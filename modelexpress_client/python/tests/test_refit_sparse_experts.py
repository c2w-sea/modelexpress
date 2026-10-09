# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from modelexpress_rl.inference.engines.vllm.sparse_experts import (
    apply_sparse_experts,
    plan_sparse_experts,
    xor_changes,
)
from modelexpress_rl.utils import compress_delta, compute_delta

_spec = importlib.util.spec_from_file_location(
    "expert_patch_fakes", Path(__file__).with_name("test_refit_expert_patch.py")
)
fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fakes)
BASE, PREFIX = fakes.BASE, fakes.PREFIX
ORACLE = "vllm.model_executor.layers.fused_moe.oracle.fp8"


@pytest.fixture
def fake_oracle(monkeypatch):
    names = ["vllm", "vllm.model_executor", "vllm.model_executor.layers",
             "vllm.model_executor.layers.fused_moe", "vllm.model_executor.layers.fused_moe.oracle", ORACLE]
    modules = {name: ModuleType(name) for name in names}
    modules[ORACLE].convert_to_fp8_moe_kernel_format = fakes._convert_entry
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _shapes():
    return {(n.split(".")[-2], n.split(".")[-1]): tuple(BASE[n].shape)
            for n in BASE if n.startswith(f"{PREFIX}.0.")}


def _payload(base, target):
    raw_target = target.contiguous().view(torch.uint8).numpy()
    raw_base = base.contiguous().view(torch.uint8).numpy()
    delta, _ = compute_delta(raw_target, raw_base)
    return compress_delta(delta)


def _apply(changes):
    target = {**BASE, **changes}
    model = fakes._Model()
    module = model.model.layers[0].mlp.experts.routed_experts
    plans, rest = plan_sparse_experts(model, changes, changes)
    assert rest == set()
    payloads = {n: _payload(BASE[n], target[n]) for n in changes}
    pointers = [t.data_ptr() for t in fakes._live(model)]
    written = apply_sparse_experts(
        PREFIX, plans[PREFIX], _shapes(),
        changes=lambda name: xor_changes(payloads[name]),
        target_scale=lambda name: target[name],
        device=torch.device("cpu"), cache={},
    )
    assert [t.data_ptr() for t in fakes._live(model)] == pointers
    return model, target, written, module


def test_xor_changes_returns_positions_and_bytes_of_nonzero_xor():
    base = np.zeros(37, dtype=np.uint8)
    target = base.copy()
    target[[0, 9, 36]] = [5, 7, 9]
    delta, _ = compute_delta(target, base)

    positions, values = xor_changes(compress_delta(delta))

    assert positions.tolist() == [0, 9, 36]
    assert values.tolist() == [5, 7, 9]


def test_plan_routes_unsupported_modules_and_other_tensors_to_reload():
    model = fakes._Model(fakes._Experts(backend="DEEPGEMM"))
    changed = {f"{PREFIX}.1.up_proj.weight", "model.norm.weight"}

    plans, rest = plan_sparse_experts(model, changed, changed)

    assert plans == {}
    assert rest == changed


def test_changed_weight_bytes_are_xored_into_the_live_layout(fake_oracle):
    name = f"{PREFIX}.2.up_proj.weight"
    changed = BASE[name].clone()
    changed[1, 0] += 17
    changed[3, 1] = -4.0

    model, target, written, _ = _apply({name: changed})

    for live, expected in zip(fakes._live(model), fakes._expected(target)):
        assert torch.equal(live, expected)
    assert 0 < written <= 8


def test_rows_owned_by_another_rank_are_not_written(fake_oracle):
    name = f"{PREFIX}.1.gate_proj.weight"
    changed = BASE[name].clone()
    changed[0, 0] += 1  # rank 0 rows; the fake installer is rank 1

    model, target, written, _ = _apply({name: changed})

    assert written == 0
    for live, expected in zip(fakes._live(model), fakes._expected(BASE)):
        assert torch.equal(live, expected)


def test_scales_and_down_weights_are_written_as_absolute_values(fake_oracle):
    up_scale = f"{PREFIX}.3.up_proj.weight_scale_inv"
    down = f"{PREFIX}.0.down_proj.weight"
    down_scale = f"{PREFIX}.0.down_proj.weight_scale_inv"
    changes = {
        up_scale: torch.tensor([[700.0], [0.75]]),
        down: BASE[down] * 2,
        down_scale: torch.tensor([[11.0, 0.9]]),
    }

    model, target, _, _ = _apply(changes)

    for live, expected in zip(fakes._live(model), fakes._expected(target)):
        assert torch.equal(live, expected)


def test_near_zero_scales_are_clamped_like_the_kernel_conversion(fake_oracle):
    name = f"{PREFIX}.1.down_proj.weight_scale_inv"

    model, target, _, module = _apply({name: torch.tensor([[3.0, 1e-30]])})

    for live, expected in zip(fakes._live(model), fakes._expected(target)):
        assert torch.equal(live, expected)
    assert module.w2_weight_scale_inv[1].min().item() == pytest.approx(1e-10)


@pytest.fixture
def fake_vllm(monkeypatch, fake_oracle):
    from contextlib import contextmanager
    from types import SimpleNamespace

    state = SimpleNamespace(events=[])

    @contextmanager
    def current_config(_config):
        yield

    names = ["vllm.config", "vllm.model_executor.layers.attention", "vllm.model_executor.model_loader",
             "vllm.model_executor.model_loader.default_loader", "vllm.model_executor.model_loader.reload",
             fakes.LAYERWISE]
    modules = {name: ModuleType(name) for name in names}
    modules["vllm.config"].set_current_vllm_config = current_config
    modules["vllm.model_executor.layers.attention"].is_deferred_attention_layer = lambda _layer: False
    modules["vllm.model_executor.model_loader.default_loader"].DefaultModelLoader = object
    layerwise = modules[fakes.LAYERWISE]
    layerwise.LAYERWISE_INFO = {}
    layerwise.initialize_layerwise_reload = lambda _model: state.events.append("initialize")
    layerwise.finalize_layerwise_reload = lambda _model, _config: state.events.append("finalize")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setenv("MX_REFIT_DELTA_SURGICAL", "true")
    monkeypatch.setenv("MX_REFIT_DELTA_SPARSE_WRITE", "true")
    return state


def _install_sparse(tmp_path, changes):
    target = {**BASE, **changes}
    deferred = fakes._deferred_delta(tmp_path, BASE, changes)
    model = fakes._Model()
    metrics = fakes._install(tmp_path, model, target, set(changes), deferred=deferred)
    return model, target, metrics


def test_installer_writes_sparse_experts_without_a_reload(fake_vllm, tmp_path, caplog):
    import logging

    name = f"{PREFIX}.2.up_proj.weight"
    changed = BASE[name].clone()
    changed[3, 0] += 9

    with caplog.at_level(logging.INFO):
        model, target, metrics = _install_sparse(tmp_path, {name: changed})

    for live, expected in zip(fakes._live(model), fakes._expected(target)):
        assert torch.equal(live, expected)
    assert model.loads == []
    assert fake_vllm.events == []
    assert metrics["perf/mx_receive_sparse_expert_bytes"] > 0
    assert "sparse_expert_bytes=" in caplog.text


def test_installer_reloads_other_modules_and_writes_experts_sparsely(fake_vllm, tmp_path):
    name = f"{PREFIX}.0.down_proj.weight"
    model, target, _ = _install_sparse(
        tmp_path, {name: BASE[name] + 1, "model.norm.weight": torch.tensor([7.0, 8.0])}
    )

    assert model.loads == [["model.norm.weight"]]
    for live, expected in zip(fakes._live(model), fakes._expected(target)):
        assert torch.equal(live, expected)


def test_sparse_failure_reloads_the_whole_expert_module(fake_vllm, tmp_path, monkeypatch):
    from modelexpress_rl.inference.engines.vllm import installer

    def fail(*_args, **_kwargs):
        raise RuntimeError("scatter failed")

    monkeypatch.setattr(installer, "apply_sparse_experts", fail)
    name = f"{PREFIX}.1.up_proj.weight"

    model, _, metrics = _install_sparse(tmp_path, {name: BASE[name] + 1})

    assert model.loads == [sorted(n for n in BASE if n.startswith(PREFIX))]
    assert metrics["perf/mx_receive_surgical_fallback"] == 1


def test_sparse_scale_writes_reach_kernel_scale_copies(fake_oracle):
    model = fakes._Model()
    module = model.model.layers[0].mlp.experts.routed_experts
    config = module.quant_method.moe_quant_config
    config.w2_scale = module.w2_weight_scale_inv.detach().clone()
    name = f"{PREFIX}.3.down_proj.weight_scale_inv"
    changes = {name: torch.tensor([[5.0, 6.5]])}
    target = {**BASE, **changes}
    plans, _ = plan_sparse_experts(model, changes, changes)

    apply_sparse_experts(PREFIX, plans[PREFIX], _shapes(), changes={}.__getitem__,
                         target_scale=lambda n: target[n], device=torch.device("cpu"), cache={})

    expected = fakes._expected(target)
    assert torch.equal(module.w2_weight_scale_inv, expected[3])
    assert torch.equal(config.w2_scale, expected[3])


def test_shared_decode_splits_modules_across_ranks_and_shares_results(tmp_path):
    import threading
    from collections import Counter

    from modelexpress_rl.inference.engines.vllm.sparse_experts import shared_changes

    names = {f"p{m}": [f"p{m}.{t}" for t in range(3)] for m in range(5)}
    decoded = Counter()
    lock = threading.Lock()

    def decode(name):
        with lock:
            decoded[name] += 1
        seed = sum(map(ord, name))
        return np.array([seed, seed + 1], dtype=np.int64), np.array([1, 2], dtype=np.uint8)

    results = {}

    def rank(r):
        results[r] = {prefix: {n: (p.tolist(), v.tolist()) for n, (p, v) in changes.items()}
                      for prefix, changes in shared_changes(names, r, 2, tmp_path, decode, timeout=10)}

    threads = [threading.Thread(target=rank, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert set(decoded.values()) == {1} and len(decoded) == 15
    assert results[0] == results[1]
    assert list(results[0]) == sorted(names)


def test_shared_decode_raises_for_a_module_that_never_appears(tmp_path):
    from modelexpress_rl.inference.engines.vllm.sparse_experts import shared_changes

    names = {"p0": ["p0.a"], "p1": ["p1.a"]}
    results = shared_changes(names, 0, 2, tmp_path, lambda n: (np.zeros(0, np.int64), np.zeros(0, np.uint8)),
                             timeout=0.2)

    prefix, changes = next(results)
    assert prefix == "p0"
    with pytest.raises(TimeoutError, match="p1"):
        next(results)
