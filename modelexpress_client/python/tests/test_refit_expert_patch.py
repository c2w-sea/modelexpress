# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch
from modelexpress_rl.inference.engines.vllm.expert_patch import plan_expert_patch
from modelexpress_rl.inference.engines.vllm.installer import _VllmInstaller
from modelexpress_rl.inference.plan import PreparedCheckpointArtifact
from modelexpress_rl.inference.receiver import DeltaChange, PreparedCheckpoint
from safetensors.torch import save_file
from torch import nn

LAYERWISE = "vllm.model_executor.model_loader.reload.layerwise"
ORACLE = "vllm.model_executor.layers.fused_moe.oracle.fp8"
EXPERTS, INTER, HIDDEN, TP, RANK = 4, 4, 2, 2, 1
PREFIX = "model.layers.0.mlp.experts"
LIVE = "model.layers.0.mlp.experts.routed_experts"
PROJS = ("gate_proj", "up_proj", "down_proj")


def _expert_tensors(seed):
    out = {}
    for e in range(EXPERTS):
        base = seed + 100 * e
        out[f"{PREFIX}.{e}.gate_proj.weight"] = torch.arange(INTER * HIDDEN) + base + 0.0
        out[f"{PREFIX}.{e}.up_proj.weight"] = torch.arange(INTER * HIDDEN) + base + 50.0
        out[f"{PREFIX}.{e}.down_proj.weight"] = torch.arange(HIDDEN * INTER) + base + 70.0
        out[f"{PREFIX}.{e}.gate_proj.weight_scale_inv"] = torch.tensor([base + 1.0, base + 2.0])
        out[f"{PREFIX}.{e}.up_proj.weight_scale_inv"] = torch.tensor([base + 3.0, base + 4.0])
        out[f"{PREFIX}.{e}.down_proj.weight_scale_inv"] = torch.tensor([[base + 5.0, base + 6.0]])
    for name in list(out):
        if name.endswith("proj.weight"):
            rows = HIDDEN if ".down_proj." in name else INTER
            out[name] = out[name].reshape(rows, -1)
        elif ".down_proj." not in name:
            out[name] = out[name].reshape(2, 1)
    return out


BASE = {**_expert_tensors(0), "model.norm.weight": torch.tensor([1.0, 2.0])}


def _convert(w13, w2, s13, s2):
    """Fake kernel layout: per-expert gate/up swap plus a transpose."""
    half = w13.shape[1] // 2
    w13 = torch.cat([w13[:, half:], w13[:, :half]], 1).transpose(1, 2).contiguous()
    s13 = torch.cat([s13[:, s13.shape[1] // 2 :], s13[:, : s13.shape[1] // 2]], 1)
    return w13, w2.transpose(1, 2).contiguous(), s13.clamp(min=1e-10), s2.clamp(min=1e-10)


class _Experts(nn.Module):
    def __init__(self, backend="FLASHINFER_TRTLLM"):
        super().__init__()
        self.moe_config = SimpleNamespace(
            tp_rank=RANK,
            is_act_and_mul=True,
            moe_parallel_config=SimpleNamespace(tp_size=TP),
        )
        w13, w2, s13, s2 = self._full(BASE)
        self.w13_weight = nn.Parameter(w13, requires_grad=False)
        self.w2_weight = nn.Parameter(w2, requires_grad=False)
        self.w13_weight_scale_inv = nn.Parameter(s13, requires_grad=False)
        self.w2_weight_scale_inv = nn.Parameter(s2, requires_grad=False)
        self.quant_method = SimpleNamespace(
            fp8_backend=SimpleNamespace(name=backend),
            block_quant=True,
            weight_scale_name="weight_scale_inv",
            moe_quant_config=SimpleNamespace(
                w1_scale=self.w13_weight_scale_inv, w2_scale=self.w2_weight_scale_inv
            ),
        )

    @staticmethod
    def narrow(tensor, dim):
        size = tensor.shape[dim] // TP
        return tensor.narrow(dim, RANK * size, size)

    @classmethod
    def _full(cls, tensors):
        def get(e, proj, kind):
            return tensors[f"{PREFIX}.{e}.{proj}.{kind}"]

        w13 = torch.stack([
            torch.cat([cls.narrow(get(e, "gate_proj", "weight"), 0),
                       cls.narrow(get(e, "up_proj", "weight"), 0)])
            for e in range(EXPERTS)
        ])
        s13 = torch.stack([
            torch.cat([cls.narrow(get(e, "gate_proj", "weight_scale_inv"), 0),
                       cls.narrow(get(e, "up_proj", "weight_scale_inv"), 0)])
            for e in range(EXPERTS)
        ])
        w2 = torch.stack([cls.narrow(get(e, "down_proj", "weight"), 1) for e in range(EXPERTS)])
        s2 = torch.stack([cls.narrow(get(e, "down_proj", "weight_scale_inv"), 1) for e in range(EXPERTS)])
        return _convert(w13, w2, s13, s2)

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return expert_id

    def _load_model_weight_or_group_weight_scale(
        self, shard_dim, expert_data, shard_id, loaded_weight, tp_rank
    ):
        assert tp_rank == RANK
        loaded = self.narrow(loaded_weight, shard_dim)
        if shard_id == "w2":
            expert_data.copy_(loaded)
            return
        half = expert_data.shape[shard_dim] // 2
        offset = 0 if shard_id == "w1" else half
        expert_data.narrow(shard_dim, offset, half).copy_(loaded)


class _Model(nn.Module):
    def __init__(self, experts=None):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.experts = nn.Module()
        self.model.layers[0].mlp.experts.routed_experts = experts or _Experts()
        self.loads = []

    def load_weights(self, weights):
        batch = [name for name, _tensor in weights]
        self.loads.append(sorted(batch))
        return set(batch)


@pytest.fixture
def fake_vllm(monkeypatch):
    state = SimpleNamespace(events=[], convert=_convert_entry)

    @contextmanager
    def current_config(_config):
        yield

    names = [
        "vllm",
        "vllm.config",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.attention",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.oracle",
        ORACLE,
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.default_loader",
        "vllm.model_executor.model_loader.reload",
        LAYERWISE,
    ]
    modules = {name: ModuleType(name) for name in names}
    modules["vllm.config"].set_current_vllm_config = current_config
    modules["vllm.model_executor.layers.attention"].is_deferred_attention_layer = (
        lambda _layer: False
    )
    modules["vllm.model_executor.model_loader.default_loader"].DefaultModelLoader = object
    modules[ORACLE].convert_to_fp8_moe_kernel_format = (
        lambda **kwargs: state.convert(**kwargs)
    )
    layerwise = modules[LAYERWISE]
    layerwise.LAYERWISE_INFO = {}
    layerwise.initialize_layerwise_reload = lambda _model: state.events.append("initialize")
    layerwise.finalize_layerwise_reload = lambda _model, _config: state.events.append("finalize")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setenv("MX_REFIT_DELTA_SURGICAL", "true")
    monkeypatch.setenv("MX_REFIT_DELTA_EXPERT_PATCH", "true")
    return state


def _convert_entry(*, fp8_backend, layer, w13, w2, w13_scale, w2_scale, **_):
    assert fp8_backend.name == "FLASHINFER_TRTLLM"
    return _convert(w13, w2, w13_scale, w2_scale)


def _checkpoint(tmp_path, tensors):
    path = tmp_path / "checkpoint"
    path.mkdir(exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, path / "model.safetensors")
    return path


def _installer(model):
    config = SimpleNamespace(load_config=SimpleNamespace(load_format="modelexpress"), quant_config=None)
    return _VllmInstaller(
        model=model,
        vllm_config=config,
        model_config=SimpleNamespace(model="/launch", revision=None),
        device=torch.device("cpu"),
    )


def _target(changes):
    target = dict(BASE)
    for name, value in changes.items():
        target[name] = value
    return target


def _install(tmp_path, model, target, changed, deferred=None):
    installer = _installer(model)
    path = _checkpoint(tmp_path, BASE if deferred is not None else target)
    installer._live_version = "v1"
    lineage = (DeltaChange("v1", "v2", frozenset(changed)),)
    return installer.install(
        PreparedCheckpointArtifact(
            PreparedCheckpoint("v2", path, {}, delta_changes=lineage, deferred=deferred)
        )
    )


def _deferred_delta(tmp_path, base_tensors, target_tensors):
    import safetensors.numpy
    from modelexpress_rl.inference.receiver import DeferredDelta
    from modelexpress_rl.utils import checksum_factory, compress_delta, compute_delta

    artifact = tmp_path / "delta"
    artifact.mkdir()
    payloads, checksums = {}, {}
    for name, target in target_tensors.items():
        raw_target = target.contiguous().view(torch.uint8).numpy()
        raw_base = base_tensors[name].contiguous().view(torch.uint8).numpy()
        delta, _ = compute_delta(raw_target, raw_base)
        payloads[name] = compress_delta(delta)
        checksum = checksum_factory("adler32")
        checksum.update(raw_target)
        checksums[name] = checksum.hexdigest()
    (artifact / "delta.safetensors").write_bytes(
        safetensors.numpy.save(payloads, metadata=checksums)
    )
    return DeferredDelta(
        artifact=artifact,
        index_metadata={"compression_format": "zstd", "checksum_format": "adler32"},
        weight_map={name: "delta.safetensors" for name in target_tensors},
    )


def _expected(target):
    return _Experts._full(target)


def _live(model):
    m = model.model.layers[0].mlp.experts.routed_experts
    return m.w13_weight, m.w2_weight, m.w13_weight_scale_inv, m.w2_weight_scale_inv


def test_plan_splits_routed_expert_tensors_from_other_changes():
    model = _Model()
    changed = {f"{PREFIX}.2.up_proj.weight", f"{PREFIX}.3.down_proj.weight_scale_inv", "model.norm.weight"}

    patch, rest = plan_expert_patch(model, changed, BASE)

    assert rest == {"model.norm.weight"}
    assert list(patch) == [PREFIX]
    assert patch[PREFIX].experts == [2, 3]
    assert patch[PREFIX].module is model.model.layers[0].mlp.experts.routed_experts


@pytest.mark.parametrize("reason", ["backend", "scale_geometry"])
def test_plan_leaves_unsupported_expert_modules_to_the_module_reload(reason):
    experts = _Experts(backend="DEEPGEMM" if reason == "backend" else "FLASHINFER_TRTLLM")
    if reason == "scale_geometry":
        experts.quant_method.moe_quant_config.w1_scale = experts.w13_weight_scale_inv[:, :1].clone()
    changed = {f"{PREFIX}.1.gate_proj.weight"}

    patch, rest = plan_expert_patch(_Model(experts), changed, BASE)

    assert patch == {}
    assert rest == changed


def test_changed_experts_are_written_in_place_without_a_module_reload(fake_vllm, tmp_path, caplog):
    changes = {
        f"{PREFIX}.1.up_proj.weight": BASE[f"{PREFIX}.1.up_proj.weight"] + 1000,
        f"{PREFIX}.3.down_proj.weight_scale_inv": torch.tensor([[0.1, 9.0]]),
    }
    target = _target(changes)
    model = _Model()
    pointers = [t.data_ptr() for t in _live(model)]
    before = [t.clone() for t in _live(model)]

    with caplog.at_level(logging.INFO):
        metrics = _install(tmp_path, model, target, changes)

    for live, expected, old in zip(_live(model), _expected(target), before):
        assert torch.equal(live, expected)
        assert torch.equal(live[[0, 2]], old[[0, 2]])
    assert [t.data_ptr() for t in _live(model)] == pointers
    assert fake_vllm.events == []
    assert model.loads == []
    assert metrics["perf/mx_receive_patched_experts"] == 2
    assert metrics["perf/mx_receive_surgical_fallback"] == 0
    assert "patched_experts=2" in caplog.text


def test_other_changes_reload_their_modules_and_experts_are_patched(fake_vllm, tmp_path):
    changes = {
        f"{PREFIX}.0.gate_proj.weight": BASE[f"{PREFIX}.0.gate_proj.weight"] * 3,
        "model.norm.weight": torch.tensor([7.0, 8.0]),
    }
    target = _target(changes)
    model = _Model()

    _install(tmp_path, model, target, changes)

    assert model.loads == [["model.norm.weight"]]
    assert fake_vllm.events == ["initialize", "finalize"]
    for live, expected in zip(_live(model), _expected(target)):
        assert torch.equal(live, expected)


def test_deferred_experts_are_rebuilt_from_parent_and_delta(fake_vllm, tmp_path):
    name = f"{PREFIX}.2.gate_proj.weight"
    changes = {name: BASE[name] + 5}
    target = _target(changes)
    deferred = _deferred_delta(tmp_path, BASE, changes)
    model = _Model()

    _install(tmp_path, model, target, changes, deferred=deferred)

    for live, expected in zip(_live(model), _expected(target)):
        assert torch.equal(live, expected)


def test_patch_failure_reloads_the_whole_expert_module(fake_vllm, tmp_path):
    def fail(**_kwargs):
        raise RuntimeError("conversion failed")

    fake_vllm.convert = fail
    name = f"{PREFIX}.1.up_proj.weight"
    target = _target({name: BASE[name] + 1})
    model = _Model()

    metrics = _install(tmp_path, model, target, {name})

    assert model.loads == [sorted(n for n in BASE if n.startswith(PREFIX))]
    assert metrics["perf/mx_receive_surgical_fallback"] == 1
    assert metrics["perf/mx_receive_patched_experts"] == 0


def test_expert_patch_is_opt_in(fake_vllm, tmp_path, monkeypatch):
    monkeypatch.setenv("MX_REFIT_DELTA_EXPERT_PATCH", "false")
    name = f"{PREFIX}.1.up_proj.weight"
    target = _target({name: BASE[name] + 1})
    model = _Model()

    _install(tmp_path, model, target, {name})

    assert model.loads == [sorted(n for n in BASE if n.startswith(PREFIX))]


def test_kernel_scale_copies_left_by_a_reload_are_written_too(fake_vllm, tmp_path):
    experts = _Experts()
    config = experts.quant_method.moe_quant_config
    config.w1_scale = experts.w13_weight_scale_inv.detach().clone()
    config.w2_scale = experts.w2_weight_scale_inv.detach().clone()
    name = f"{PREFIX}.2.up_proj.weight_scale_inv"
    changes = {name: torch.tensor([[33.0], [44.0]])}
    target = _target(changes)
    model = _Model(experts)

    metrics = _install(tmp_path, model, target, changes)

    expected = _expected(target)
    assert metrics["perf/mx_receive_patched_experts"] == 1
    assert torch.equal(experts.w13_weight_scale_inv, expected[2])
    assert torch.equal(config.w1_scale, expected[2])
    assert torch.equal(config.w2_scale, expected[3])
