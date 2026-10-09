# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply routed-expert XOR deltas straight to live FusedMoE kernel tensors.

The live expert layout is a byte permutation of the checkpoint for weights and
a permutation plus clamp for scales. Both maps are derived once per module shape
by sending position codes through vLLM's own per-rank loader and kernel-format
conversion. Changed weight bytes are XORed into the live bytes on the device;
scales are rebuilt and written as absolute values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import torch
import zstandard
from torch import nn

from modelexpress_rl.inference.engines.vllm.expert_patch import (
    _EXPERT,
    _SHARD,
    _SHARD_DIM,
    _live_tensors,
    _module_for,
    _patchable,
)

_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
_CODE_BITS = 24


class SparseExpertError(RuntimeError):
    pass


@dataclass
class SparsePlan:
    module: nn.Module
    weights: list[str] = field(default_factory=list)
    scales: list[str] = field(default_factory=list)


def plan_sparse_experts(
    model: nn.Module, changed: Iterable[str], delta_names: Iterable[str]
) -> tuple[dict[str, SparsePlan], set[str]]:
    """Split changed names into sparse-writable expert tensors and the rest."""
    delta = set(delta_names)
    plans: dict[str, SparsePlan] = {}
    rejected: set[str] = set()
    rest: set[str] = set()
    for name in sorted(changed):
        match = _EXPERT.match(name)
        prefix = match["prefix"] if match else None
        if match is None or name not in delta or prefix in rejected:
            rest.add(name)
            continue
        if prefix not in plans:
            module = _module_for(model, prefix)
            if module is None or not _patchable(module):
                rejected.add(prefix)
                rest.add(name)
                continue
            plans[prefix] = SparsePlan(module)
        bucket = plans[prefix].weights if match["kind"] == "weight" else plans[prefix].scales
        bucket.append(name)
    for prefix in rejected:
        rest.update(n for n in changed if n.startswith(f"{prefix}."))
    return plans, rest


def xor_changes(payload: memoryview | bytes) -> tuple[np.ndarray, np.ndarray]:
    """Byte positions and XOR values of the non-zero bytes of one zstd payload."""
    data = np.frombuffer(zstandard.ZstdDecompressor().decompress(bytes(payload)), dtype=np.uint8)
    whole = data.size // 8 * 8
    words = np.flatnonzero(data[:whole].view(np.uint64))
    rows, cols = np.nonzero(data[:whole].reshape(-1, 8)[words])
    positions = words[rows] * 8 + cols
    tail = np.flatnonzero(data[whole:]) + whole
    positions = np.concatenate([positions, tail]).astype(np.int64)
    return positions, data[positions]


class ExpertLayout:
    """Checkpoint element -> live element offset within one expert slot."""

    def __init__(self, module: nn.Module, shapes: Mapping[tuple[str, str], tuple[int, ...]], device):
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
            convert_to_fp8_moe_kernel_format,
        )

        self._convert = convert_to_fp8_moe_kernel_format
        self.module = module
        self.shapes = shapes
        self.device = device
        live = _live_tensors(module)
        self.slots = [t[0].numel() for t in live]
        self.maps: dict[tuple[str, str], torch.Tensor] = {}
        for (proj, kind), shape in shapes.items():
            self.maps[(proj, kind)] = self._derive(proj, kind, shape, live)

    def _derive(self, proj, kind, shape, live) -> torch.Tensor:
        numel = int(np.prod(shape))
        if numel >= 1 << _CODE_BITS:
            raise SparseExpertError(f"{proj}.{kind} has too many elements for position codes")
        codes = torch.arange(1, numel + 1, dtype=torch.int64, device=self.device).reshape(shape)
        target = 1 if _SHARD[proj] == "w2" else 0
        if kind == "weight_scale_inv":
            out = self._run(proj, kind, codes.to(torch.float32), live)[target + 2]
            recovered = out.reshape(-1).to(torch.int64)
        else:
            dtype = live[target].dtype
            size = torch.empty((), dtype=dtype).element_size()
            bits = 8 * size
            recovered = None
            for shift in range(0, _CODE_BITS, bits):
                digit = ((codes >> shift) & ((1 << bits) - 1)).to(_INT_VIEW[size]).view(dtype)
                out = self._run(proj, kind, digit, live)[target]
                plane = out.reshape(-1).view(_INT_VIEW[size]).to(torch.int64) & ((1 << bits) - 1)
                recovered = plane << shift if recovered is None else recovered | (plane << shift)
        owned = torch.nonzero(recovered, as_tuple=True)[0]
        mapping = torch.full((numel,), -1, dtype=torch.int64, device=self.device)
        mapping[recovered[owned] - 1] = owned
        return mapping

    def _run(self, proj, kind, value, live):
        module = self.module
        shard = _SHARD[proj]
        tp_size = module.moe_config.moe_parallel_config.tp_size
        temps = _load_format_temps(module, self.shapes, tp_size, self.device)
        index = (1 if shard == "w2" else 0) + (2 if kind == "weight_scale_inv" else 0)
        module._load_model_weight_or_group_weight_scale(
            shard_dim=_SHARD_DIM[shard],
            expert_data=temps[index][0],
            shard_id=shard,
            loaded_weight=value,
            tp_rank=module.moe_config.tp_rank,
        )
        return self._convert(
            fp8_backend=module.quant_method.fp8_backend,
            layer=module,
            w13=temps[0],
            w2=temps[1],
            w13_scale=temps[2],
            w2_scale=temps[3],
            w13_input_scale=None,
            w2_input_scale=None,
        )


def _load_format_temps(module, shapes, tp_size, device):
    gate = shapes[("gate_proj", "weight")]
    down = shapes[("down_proj", "weight")]
    gate_scale = shapes[("gate_proj", "weight_scale_inv")]
    down_scale = shapes[("down_proj", "weight_scale_inv")]
    live = _live_tensors(module)

    def zeros(rows, cols, dtype):
        size = torch.empty((), dtype=dtype).element_size()
        return torch.zeros((1, rows, cols * size), dtype=torch.uint8, device=device).view(dtype)

    return [
        zeros(2 * (gate[0] // tp_size), gate[1], live[0].dtype),
        zeros(down[0], down[1] // tp_size, live[1].dtype),
        zeros(2 * (gate_scale[0] // tp_size), gate_scale[1], torch.float32),
        zeros(down_scale[0], down_scale[1] // tp_size, torch.float32),
    ]


def layout_for(module, shapes, device, cache: dict) -> ExpertLayout:
    key = (tuple(sorted(shapes.items())), tuple(tuple(t.shape) for t in _live_tensors(module)))
    layout = cache.get(key)
    if layout is None:
        layout = ExpertLayout(module, shapes, device)
        cache[key] = layout
    layout.module = module
    return layout


@torch.no_grad()
def apply_sparse_experts(
    prefix: str,
    plan: SparsePlan,
    shapes: Mapping[tuple[str, str], tuple[int, ...]],
    changes: Callable[[str], tuple[np.ndarray, np.ndarray]],
    target_scale: Callable[[str], torch.Tensor],
    device: torch.device,
    cache: dict,
) -> int:
    """Write one module's changed experts; return the number of changed bytes."""
    module = plan.module
    layout = layout_for(module, shapes, device, cache)
    live = _live_tensors(module)
    pointers = [t.data_ptr() for t in live]
    written = 0
    batches: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {0: ([], []), 1: ([], [])}
    for name in plan.weights:
        match = _EXPERT.match(name)
        expert = module._map_global_expert_id_to_local_expert_id(int(match["expert"]))
        if expert < 0:
            raise SparseExpertError(f"{name} is not local")
        target = 1 if _SHARD[match["proj"]] == "w2" else 0
        size = live[target].element_size()
        positions, values = changes(name)
        if positions.size == 0:
            continue
        pos = torch.from_numpy(positions).to(device)
        mapped = layout.maps[(match["proj"], "weight")][pos // size]
        keep = mapped >= 0
        offsets = (expert * layout.slots[target] + mapped[keep]) * size + pos[keep] % size
        batches[target][0].append(offsets)
        batches[target][1].append(torch.from_numpy(values).to(device)[keep])
    for target, (offsets, values) in batches.items():
        if not offsets:
            continue
        flat = live[target].view(-1).view(torch.uint8)
        index = torch.cat(offsets)
        flat[index] = flat[index] ^ torch.cat(values)
        written += index.numel()
    for name in plan.scales:
        match = _EXPERT.match(name)
        expert = module._map_global_expert_id_to_local_expert_id(int(match["expert"]))
        target = 3 if _SHARD[match["proj"]] == "w2" else 2
        mapping = layout.maps[(match["proj"], "weight_scale_inv")]
        keep = torch.nonzero(mapping >= 0, as_tuple=True)[0]
        value = target_scale(name).to(device=device, dtype=torch.float32).reshape(-1)[keep]
        flat = live[target].view(-1)
        flat[expert * layout.slots[target] + mapping[keep]] = value.clamp(min=1e-10).to(flat.dtype)
    if [t.data_ptr() for t in _live_tensors(module)] != pointers:
        raise SparseExpertError(f"{prefix}: live expert storage changed")
    return written


__all__ = ["SparseExpertError", "SparsePlan", "apply_sparse_experts", "plan_sparse_experts", "xor_changes"]
