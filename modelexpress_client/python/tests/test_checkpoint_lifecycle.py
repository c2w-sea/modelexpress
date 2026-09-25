# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Condition
from types import SimpleNamespace

import pytest
from modelexpress_rl import (
    CheckpointCollectiveContext,
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    ObjectStorageGeneratorConfig,
    ObjectStorageType,
    VllmGeneratorContext,
    WeightSource,
    WeightVersionRef,
)
from modelexpress_rl.inference.checkpoint_selection import CheckpointChanges
from modelexpress_rl.inference.checkpoint_transaction import (
    refit_checkpoint_collectively,
)
from modelexpress_rl.inference.client import StagedWeightHandle
from modelexpress_rl.inference.methods.canonical_delta import CanonicalDeltaUpdateMethod
from modelexpress_rl.inference.plan import PreparedCheckpointArtifact
from modelexpress_rl.inference.receiver import PreparedCheckpoint
from modelexpress_rl.inference.session import SessionUpdate, WeightUpdateSession


def _context(**kwargs):
    return CheckpointCollectiveContext(
        **{
            "world_size": 1,
            "all_gather": lambda value, timeout: (value,),
            "safe_point": lambda: None,
            "fence": lambda: None,
        }
        | kwargs
    )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_collective_requires_finite_positive_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        _context(timeout_seconds=timeout)


def test_config_rejects_missing_lifecycle_and_unknown_mode():
    context = VllmGeneratorContext(model=None, vllm_config=None)
    assert (
        ModelExpressGeneratorConfig(engine_context=context).checkpoint_install_mode
        == "full"
    )
    with pytest.raises(ValueError, match="CheckpointCollectiveContext"):
        ModelExpressGeneratorConfig(
            engine_context=context, checkpoint_install_mode="partial_if_supported"
        )
    with pytest.raises(ValueError, match="checkpoint_install_mode"):
        ModelExpressGeneratorConfig(
            engine_context=context, checkpoint_install_mode="partial"
        )


def test_config_requires_object_storage_in_source_order(tmp_path):
    kwargs = {
        "engine_context": VllmGeneratorContext(
            model=None, vllm_config=None, checkpoint_collective=_context()
        ),
        "checkpoint_install_mode": "partial_if_supported",
        "object_storage": ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="base",
            seed_checkpoint_path=None,
            refit_checkpoint_dir=str(tmp_path),
        ),
    }
    with pytest.raises(ValueError, match="object_storage"):
        ModelExpressGeneratorConfig(
            **(kwargs | {"object_storage": None}),
            source_order=(WeightSource.GENERATOR,),
        )
    for order in (
        None,
        (WeightSource.OBJECT_STORAGE,),
        (WeightSource.GENERATOR, WeightSource.OBJECT_STORAGE),
    ):
        config = ModelExpressGeneratorConfig(**kwargs, source_order=order)
        assert config.checkpoint_install_mode == "partial_if_supported"


def _run_clients(tmp_path, *, failure=None, unsupported=False, noop=None, repeat=False):
    condition = Condition()
    rounds = {}
    sequences = [0, 0]
    states = [
        {"paused": False, "fenced": False, "weight": "base", "published": "base"}
        for _ in range(2)
    ]
    events = []
    clients = []

    def gather(rank, value, timeout):
        assert timeout == 2
        with condition:
            seq = sequences[rank]
            sequences[rank] += 1
            current = rounds.setdefault(seq, {})
            current[rank] = value
            condition.notify_all()
            if not condition.wait_for(lambda: len(current) == 2, timeout=timeout):
                raise TimeoutError("rank lost")
            return tuple(current[i] for i in range(2))

    def run(rank):
        state = states[rank]

        def step(name):
            events.append((name, rank))
            if failure == (name, rank):
                raise RuntimeError(f"injected {name}")

        @contextmanager
        def safe_point():
            state["paused"] = True
            step("pause")
            yield
            step("resume")
            assert all(s["weight"] == "target" for s in states)
            assert all(s["published"] == "target" for s in states)
            if not state["fenced"]:
                state["paused"] = False

        def fence():
            state["paused"] = True
            state["fenced"] = True
            step("fence")

        def unpublish():
            state["published"] = None
            step("unpublish")

        def publish(version):
            assert version == "target"
            assert all(s["weight"] == "target" for s in states)
            assert all(c._serving_version_id == "target" for c in clients)
            assert all(("lease", i) in events for i in range(2))
            state["published"] = version
            step("publish")

        class Method(CanonicalDeltaUpdateMethod):
            def __init__(self):
                pass

            @contextmanager
            def installation_context(self, prepared, *, activate=True):
                assert activate is False
                yield

            def activate(self, prepared):
                step("activate")

        class Installer:
            def install_checkpoint_collectively(self, prepared, **kwargs):
                def copy():
                    assert all(s["paused"] and s["published"] is None for s in states)
                    state["weight"] = "target"
                    step("copy")

                def prepare(_):
                    step("stage")
                    if unsupported and rank == 1:
                        return None
                    return SimpleNamespace(install=copy)

                return refit_checkpoint_collectively(
                    prepared,
                    **kwargs,
                    prepare_partial=prepare,
                    install_full=lambda: (step("full"), copy()),
                    synchronize=lambda: step("sync"),
                )

        client = ModelExpressGeneratorClient()
        client._serving_version_id = "base"
        client._checkpoint_collective = _context(
            world_size=2,
            all_gather=lambda value, timeout: gather(rank, value, timeout),
            safe_point=safe_point,
            fence=fence,
            timeout_seconds=2,
        )
        clients.append(client)
        update = SessionUpdate(
            plan=SimpleNamespace(
                method=Method(),
                installer=Installer(),
                source=SimpleNamespace(kind=WeightSource.OBJECT_STORAGE),
                version=SimpleNamespace(version_id="target"),
            ),
            prepared=PreparedCheckpointArtifact(
                PreparedCheckpoint(
                    "target",
                    tmp_path,
                    {},
                    CheckpointChanges("base", "target", frozenset({"weight"})),
                )
            ),
            lease=SimpleNamespace(close=lambda: step("lease")),
        )
        client._runtime = SimpleNamespace(
            session=WeightUpdateSession(planner=None, start_lease=None),
            unpublish_runtime_tensors=unpublish,
            publish_runtime_tensors=publish,
        )
        if noop == "all" or (noop == "one" and rank == 1):
            update = None
            client._serving_version_id = "target"
            state["weight"] = state["published"] = "target"
        handle = StagedWeightHandle(client=client, version_id="target", update=update)
        client._active_handle = handle
        try:
            result = client.apply_weight(handle)
            if repeat:
                assert client.apply_weight(handle) == result
            step("returned")
            return result
        except BaseException as error:  # noqa: BLE001 - inspect both rank outcomes
            with pytest.raises(RuntimeError, match="worker recovery"):
                client.apply_weight(handle)
            with pytest.raises(RuntimeError, match="worker recovery"):
                client.stage_weight(version=WeightVersionRef("next"))
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, range(2)))
    return results, states, events


@pytest.mark.parametrize("unsupported", [False, True])
def test_configured_client_installs_then_publishes_and_resumes(tmp_path, unsupported):
    results, states, events = _run_clients(tmp_path, unsupported=unsupported)
    assert results == ["full" if unsupported else "partial"] * 2
    assert all(
        not s["paused"] and not s["fenced"] and s["published"] == "target"
        for s in states
    )
    assert sum(name == "full" for name, _ in events) == (2 if unsupported else 0)


@pytest.mark.parametrize(
    "phase",
    [
        "pause",
        "unpublish",
        "stage",
        "copy",
        "sync",
        "activate",
        "lease",
        "publish",
        "resume",
    ],
)
def test_configured_client_fences_every_rank_on_lifecycle_failure(tmp_path, phase):
    results, states, events = _run_clients(tmp_path, failure=(phase, 1))
    assert all(isinstance(result, BaseException) for result in results)
    assert all(s["fenced"] and s["paused"] and s["published"] is None for s in states)
    assert not any(name == "returned" for name, _ in events)


def test_initialize_retains_configured_lifecycle(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    import modelexpress_rl.inference.client as client_module

    collective = _context()
    runtime = MagicMock(initial_version_id="base")
    monkeypatch.setattr(
        client_module, "initialize_generator_runtime", lambda **kw: runtime
    )
    monkeypatch.setattr(
        ModelExpressGeneratorClient,
        "_validate_initial_serving_version",
        lambda *args: None,
    )
    monkeypatch.setattr(
        ModelExpressGeneratorClient, "_register_worker", lambda *args: None
    )
    monkeypatch.setattr(
        ModelExpressGeneratorClient, "_renew_worker_registration", lambda *args: None
    )
    config = ModelExpressGeneratorConfig(
        model_name="test/model",
        engine_context=VllmGeneratorContext(
            model=None,
            vllm_config=None,
            checkpoint_collective=collective,
        ),
        object_storage=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="base",
            seed_checkpoint_path=None,
            refit_checkpoint_dir=str(tmp_path),
        ),
        source_order=(WeightSource.OBJECT_STORAGE,),
        checkpoint_install_mode="partial_if_supported",
    )
    client = ModelExpressGeneratorClient.initialize(config)
    try:
        assert client._checkpoint_collective is collective
    finally:
        client.close()


def test_safe_point_cannot_suppress_transaction_failure():
    from contextlib import suppress

    from modelexpress_rl.inference.checkpoint_lifecycle import run_checkpoint_lifecycle

    events = []

    def fail():
        raise RuntimeError("copy failed")

    with pytest.raises(RuntimeError, match="installed"):
        run_checkpoint_lifecycle(
            _context(safe_point=lambda: suppress(BaseException)),
            serving_version="base",
            target_version="target",
            validate=lambda: None,
            unpublish=lambda: None,
            install=fail,
            publish=lambda: events.append("publish"),
            fence=lambda: events.append("fence"),
        )
    assert events == ["fence"]


def test_lifecycle_transport_timeout_fences_before_any_publication():
    from modelexpress_rl.inference.checkpoint_lifecycle import run_checkpoint_lifecycle

    events = []

    def lost_rank(value, timeout):
        assert timeout == 0.5
        raise TimeoutError("rank lost")

    with pytest.raises(TimeoutError, match="rank lost"):
        run_checkpoint_lifecycle(
            _context(all_gather=lost_rank, timeout_seconds=0.5),
            serving_version="base",
            target_version="target",
            validate=lambda: None,
            unpublish=lambda: None,
            install=lambda: events.append("install"),
            publish=lambda: events.append("publish"),
            fence=lambda: events.append("fence"),
        )
    assert events == ["fence"]


@pytest.mark.parametrize("noop", ["all", "one"])
def test_no_op_requires_every_rank_to_agree(tmp_path, noop):
    results, states, events = _run_clients(tmp_path, noop=noop)
    assert not any(name == "copy" for name, _ in events)
    if noop == "all":
        assert results == [None, None]
        assert not any(s["fenced"] for s in states)
    else:
        assert all(isinstance(result, BaseException) for result in results)
        assert all(s["fenced"] and s["published"] is None for s in states)


def test_repeated_apply_does_not_reinstall(tmp_path):
    results, states, events = _run_clients(tmp_path, repeat=True)
    assert results == ["partial", "partial"]
    assert sum(name == "copy" for name, _ in events) == 2
    assert not any(s["fenced"] for s in states)


def _run_routes(tmp_path, routes):
    """Apply one update per rank; routes are "checkpoint" or "p2p"."""
    condition = Condition()
    rounds = {}
    sequences = [0] * len(routes)
    events = []

    def gather(rank, value, timeout):
        with condition:
            seq = sequences[rank]
            sequences[rank] += 1
            current = rounds.setdefault(seq, {})
            current[rank] = value
            condition.notify_all()
            if not condition.wait_for(
                lambda: len(current) == len(routes), timeout=timeout
            ):
                raise TimeoutError("rank lost")
            return tuple(current[i] for i in range(len(routes)))

    class Method:
        @contextmanager
        def installation_context(self, prepared):
            yield

    class Installer:
        def install(self, prepared):
            events.append(("install", prepared))
            return {"perf/mx_receive_install_time": 1.0}

    def run(rank):
        client = ModelExpressGeneratorClient()
        client._serving_version_id = "base"
        client._checkpoint_collective = _context(
            world_size=len(routes),
            all_gather=lambda value, timeout: gather(rank, value, timeout),
            fence=lambda: events.append(("fence", rank)),
            timeout_seconds=2,
        )
        if routes[rank] == "checkpoint":
            method = CanonicalDeltaUpdateMethod.__new__(CanonicalDeltaUpdateMethod)
            prepared = PreparedCheckpointArtifact(
                PreparedCheckpoint("target", tmp_path, {})
            )
        else:
            method = Method()
            prepared = SimpleNamespace(metrics={})
        update = SessionUpdate(
            plan=SimpleNamespace(
                method=method,
                installer=Installer(),
                source=SimpleNamespace(kind=WeightSource.GENERATOR),
                version=SimpleNamespace(version_id="target"),
            ),
            prepared=prepared,
            lease=SimpleNamespace(close=lambda: None),
        )
        client._runtime = SimpleNamespace(
            session=WeightUpdateSession(planner=None, start_lease=None),
            unpublish_runtime_tensors=lambda: events.append(("unpublish", rank)),
            publish_runtime_tensors=lambda v: events.append(("publish", rank)),
        )
        handle = StagedWeightHandle(client=client, version_id="target", update=update)
        client._active_handle = handle
        try:
            return client.apply_weight(handle), client
        except BaseException as error:  # noqa: BLE001 - inspect every rank
            return error, client

    with ThreadPoolExecutor(max_workers=len(routes)) as executor:
        results = list(executor.map(run, range(len(routes))))
    return results, events


def test_partial_mode_installs_p2p_updates_without_the_checkpoint_lifecycle(tmp_path):
    results, events = _run_routes(tmp_path, ["p2p", "p2p"])
    assert [result for result, _ in results] == [
        {"perf/mx_receive_install_time": 1.0}
    ] * 2
    assert all(client._serving_version_id == "target" for _, client in results)
    assert not any(name == "fence" for name, _ in events)
    assert sorted(name for name, _ in events if name != "install") == [
        "publish", "publish", "unpublish", "unpublish",
    ]


def test_partial_mode_fences_when_ranks_choose_different_routes(tmp_path):
    from modelexpress_rl.inference.checkpoint_transaction import (
        CheckpointTransactionError,
    )

    results, events = _run_routes(tmp_path, ["checkpoint", "p2p"])
    assert all(
        isinstance(result, CheckpointTransactionError) for result, _ in results
    )
    assert sorted(rank for name, rank in events if name == "fence") == [0, 1]
    assert not any(name == "install" for name, _ in events)
