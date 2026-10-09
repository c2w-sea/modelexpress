# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write changed routed experts into live FusedMoE tensors without a reload.

A changed expert is rebuilt from its six checkpoint tensors with vLLM's own
per-rank loader and backend conversion, then copied into its slot of the
existing kernel tensors. Only backends whose conversion is independent per
expert are eligible; other modules keep the whole-module reload.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

import torch
from torch import nn

_EXPERT = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale_inv)$"
)
_SHARD = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
_SHARD_DIM = {"w1": 0, "w3": 0, "w2": 1}
_PROJS = ("gate_proj", "up_proj", "down_proj")
_KINDS = ("weight", "weight_scale_inv")
_PER_EXPERT_BACKENDS = frozenset({"FLASHINFER_TRTLLM"})
_CHUNK = 32

logger = logging.getLogger(__name__)


class ExpertPatchError(RuntimeError):
    pass


@dataclass
class ExpertGroup:
    module: nn.Module
    experts: list[int] = field(default_factory=list)

    def names(self, prefix: str) -> list[str]:
        return [
            f"{prefix}.{e}.{proj}.{kind}"
            for e in self.experts
            for proj in _PROJS
            for kind in _KINDS
        ]


def _live_tensors(module: nn.Module) -> list[torch.Tensor]:
    scale = module.quant_method.weight_scale_name
    return [
        module.w13_weight,
        module.w2_weight,
        getattr(module, f"w13_{scale}"),
        getattr(module, f"w2_{scale}"),
    ]


def _scale_copies(module: nn.Module) -> list[list[torch.Tensor]] | None:
    """Every live copy of the w13 and w2 scales.

    A layerwise reload rebuilds the MoE kernel config from temporary tensors and
    then restores the layer's original storage, so eager calls read the config's
    copy while captured graphs and P2P read the layer's.
    """
    _w13, _w2, w13_scale, w2_scale = _live_tensors(module)
    config = getattr(module.quant_method, "moe_quant_config", None)
    copies = []
    for own, kernel in ((w13_scale, getattr(config, "w1_scale", None)),
                        (w2_scale, getattr(config, "w2_scale", None))):
        if not isinstance(kernel, torch.Tensor):
            return None
        if kernel.data_ptr() == own.data_ptr():
            copies.append([own])
        elif (kernel.shape, kernel.dtype, kernel.device) == (own.shape, own.dtype, own.device):
            copies.append([own, kernel])
        else:
            return None
    return copies


def _write_targets(module: nn.Module) -> list[list[torch.Tensor]]:
    copies = _scale_copies(module)
    if copies is None:
        raise ExpertPatchError("MoE kernel scales do not match the layer's scale tensors")
    w13, w2, _s13, _s2 = _live_tensors(module)
    return [[w13], [w2], *copies]


def _patchable(module: nn.Module) -> bool:
    method = getattr(module, "quant_method", None)
    backend = getattr(getattr(method, "fp8_backend", None), "name", None)
    if backend not in _PER_EXPERT_BACKENDS or not getattr(method, "block_quant", False):
        return False
    if not getattr(module.moe_config, "is_act_and_mul", False):
        return False
    return _scale_copies(module) is not None


def _module_for(model: nn.Module, prefix: str) -> nn.Module | None:
    for path in (f"{prefix}.routed_experts", prefix):
        try:
            module = model.get_submodule(path)
        except AttributeError:
            continue
        if hasattr(module, "w13_weight"):
            return module
    return None


def plan_expert_patch(
    model: nn.Module, changed: Iterable[str], checkpoint_names: Iterable[str]
) -> tuple[dict[str, ExpertGroup], set[str]]:
    """Split changed tensor names into patchable experts and everything else."""
    names = set(checkpoint_names)
    candidates: dict[str, set[int]] = {}
    rest: set[str] = set()
    for name in changed:
        match = _EXPERT.match(name)
        if match is None:
            rest.add(name)
            continue
        candidates.setdefault(match["prefix"], set()).add(int(match["expert"]))
    groups: dict[str, ExpertGroup] = {}
    rejected = []
    for prefix, experts in sorted(candidates.items()):
        module = _module_for(model, prefix)
        group = ExpertGroup(module, sorted(experts)) if module is not None else None
        if group is None or not _patchable(module) or not set(group.names(prefix)) <= names:
            rejected.append(prefix)
            rest.update(n for n in changed if n.startswith(f"{prefix}."))
            continue
        groups[prefix] = group
    if rejected:
        logger.info("Expert patch ineligible for %d modules, e.g. %s", len(rejected), rejected[:3])
    return groups, rest


def patch_experts(
    module: nn.Module,
    prefix: str,
    experts: list[int],
    tensors: Iterator[tuple[str, torch.Tensor]],
    device: torch.device,
) -> int:
    """Consume the experts' checkpoint tensors in ``ExpertGroup.names`` order."""
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        convert_to_fp8_moe_kernel_format,
    )

    method = module.quant_method
    tp_rank = module.moe_config.tp_rank
    tp_size = module.moe_config.moe_parallel_config.tp_size
    targets = _write_targets(module)
    pointers = [t.data_ptr() for copies in targets for t in copies]
    patched = 0
    for start in range(0, len(experts), _CHUNK):
        chunk = experts[start : start + _CHUNK]
        loaded: dict[tuple[int, str, str], torch.Tensor] = {}
        for _ in range(len(chunk) * len(_PROJS) * len(_KINDS)):
            name, tensor = next(tensors)
            match = _EXPERT.match(name)
            if match is None or match["prefix"] != prefix:
                raise ExpertPatchError(f"unexpected tensor {name!r} for {prefix!r}")
            loaded[(int(match["expert"]), match["proj"], match["kind"])] = tensor
        w13, w2, s13, s2 = _load_chunk(module, chunk, loaded, tp_rank, tp_size, device)
        converted = convert_to_fp8_moe_kernel_format(
            fp8_backend=method.fp8_backend,
            layer=module,
            w13=w13,
            w2=w2,
            w13_scale=s13,
            w2_scale=s2,
            w13_input_scale=None,
            w2_input_scale=None,
        )
        local = [module._map_global_expert_id_to_local_expert_id(e) for e in chunk]
        if any(e < 0 for e in local):
            raise ExpertPatchError(f"{prefix}: experts {chunk} are not all local")
        index = torch.tensor(local, dtype=torch.long, device=targets[0][0].device)
        for copies, value in zip(targets, converted, strict=True):
            for target in copies:
                if value.shape[1:] != target.shape[1:] or value.dtype != target.dtype:
                    raise ExpertPatchError(
                        f"{prefix}: converted {tuple(value.shape)} {value.dtype} does not "
                        f"fit live {tuple(target.shape)} {target.dtype}"
                    )
                target.index_copy_(0, index, value.to(target.device))
        patched += len(chunk)
    if [t.data_ptr() for copies in _write_targets(module) for t in copies] != pointers:
        raise ExpertPatchError(f"{prefix}: live expert storage changed")
    return patched


def _load_chunk(
    module: nn.Module,
    chunk: list[int],
    loaded: Mapping[tuple[int, str, str], torch.Tensor],
    tp_rank: int,
    tp_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    def first(proj, kind):
        return loaded[(chunk[0], proj, kind)]

    gate, down = first("gate_proj", "weight"), first("down_proj", "weight")
    gate_scale = first("gate_proj", "weight_scale_inv")
    down_scale = first("down_proj", "weight_scale_inv")
    k = len(chunk)

    def empty(rows, cols, like):
        size = cols * like.element_size()
        return torch.zeros((k, rows, size), dtype=torch.uint8, device=device).view(like.dtype)

    w13 = empty(2 * (gate.shape[0] // tp_size), gate.shape[1], gate)
    s13 = empty(2 * (gate_scale.shape[0] // tp_size), gate_scale.shape[1], gate_scale)
    w2 = empty(down.shape[0], down.shape[1] // tp_size, down)
    s2 = empty(down_scale.shape[0], down_scale.shape[1] // tp_size, down_scale)
    for j, expert in enumerate(chunk):
        for proj in _PROJS:
            shard = _SHARD[proj]
            for kind, target in (("weight", w2 if shard == "w2" else w13),
                                 ("weight_scale_inv", s2 if shard == "w2" else s13)):
                module._load_model_weight_or_group_weight_scale(
                    shard_dim=_SHARD_DIM[shard],
                    expert_data=target[j],
                    shard_id=shard,
                    loaded_weight=loaded[(expert, proj, kind)].to(device),
                    tp_rank=tp_rank,
                )
    return w13, w2, s13.to(torch.float32), s2.to(torch.float32)


__all__ = ["ExpertGroup", "ExpertPatchError", "patch_experts", "plan_expert_patch"]
