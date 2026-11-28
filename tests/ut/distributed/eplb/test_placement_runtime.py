# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.eplb import placement_runtime as runtime
from vllm_ascend.distributed.eplb.placement_runtime import (
    PlacementSnapshot,
    RolePlacement,
    RuntimeRoleSpec,
    load_placement_snapshot,
    parse_placement_snapshot,
    plan_placement_sources,
    validate_snapshot_for_runtime,
)


def make_snapshot_dict(*, with_draft: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "version": 1,
        "ep_size": 2,
        "target": {
            "num_logical_experts": 4,
            "num_redundant_experts": 2,
            "physical_to_logical_map": [[0, 1, 2, 0, 1, 3]],
        },
    }
    if with_draft:
        data["draft"] = {
            "num_logical_experts": 2,
            "num_redundant_experts": 0,
            "physical_to_logical_map": [[0, 1]],
        }
    return data


def write_snapshot(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


class TestPlacementSnapshotSchema(unittest.TestCase):
    def test_load_target_and_optional_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            for with_draft in (False, True):
                path = Path(directory) / f"placement_{int(with_draft)}.json"
                raw = make_snapshot_dict(with_draft=with_draft)
                write_snapshot(path, raw)
                loaded = load_placement_snapshot(path)
                parsed = parse_placement_snapshot(raw)
                self.assertEqual(loaded, parsed)
                self.assertEqual(loaded.digest(), parsed.digest())
                self.assertEqual(loaded.draft is not None, with_draft)

    def test_rejects_invalid_snapshots(self):
        cases = [
            ("version bool", lambda d: d.update(version=True), "version"),
            ("version two", lambda d: d.update(version=2), "version"),
            ("extra top key", lambda d: d.update(provenance={}), "must contain"),
            ("extra target key", lambda d: d["target"].update(model="m"), "must contain exactly"),
            ("bad ep_size", lambda d: d.update(ep_size=4), "divisible by ep_size"),
            ("missing expert", lambda d: d["target"].update(physical_to_logical_map=[[0, 1, 2, 0, 1, 2]]), "cover"),
            ("duplicate on rank", lambda d: d["target"].update(physical_to_logical_map=[[0, 0, 1, 2, 3, 1]]), "rank 0"),
            ("boolean expert id", lambda d: d["target"].update(physical_to_logical_map=[[False, 1, 2, 0, 1, 3]]), "ID"),
            ("wrong slot count", lambda d: d["target"].update(physical_to_logical_map=[[0, 1, 2, 3]]), "slot count"),
            ("empty layers", lambda d: d["target"].update(physical_to_logical_map=[]), "non-empty list"),
        ]
        for label, mutate, pattern in cases:
            raw = make_snapshot_dict()
            mutate(raw)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, pattern):
                parse_placement_snapshot(raw)

    def test_runtime_spec_validation_and_missing_draft(self):
        snap = parse_placement_snapshot(make_snapshot_dict(with_draft=True))
        target_spec = RuntimeRoleSpec(num_logical_experts=4, num_redundant_experts=2, num_layers=1)
        draft_spec = RuntimeRoleSpec(num_logical_experts=2, num_redundant_experts=0, num_layers=1)
        validate_snapshot_for_runtime(snap, 2, target_spec, draft_spec)

        with self.assertRaisesRegex(ValueError, "ep_size mismatch"):
            validate_snapshot_for_runtime(snap, 4, target_spec, draft_spec)
        with self.assertRaisesRegex(ValueError, "target num_logical_experts mismatch"):
            validate_snapshot_for_runtime(
                snap, 2, RuntimeRoleSpec(num_logical_experts=8, num_redundant_experts=2, num_layers=1), draft_spec
            )
        with self.assertRaisesRegex(ValueError, "draft num_redundant_experts mismatch"):
            validate_snapshot_for_runtime(
                snap, 2, target_spec, RuntimeRoleSpec(num_logical_experts=2, num_redundant_experts=2, num_layers=1)
            )
        target_only = parse_placement_snapshot(make_snapshot_dict(with_draft=False))
        with self.assertRaisesRegex(ValueError, "missing the expected draft"):
            validate_snapshot_for_runtime(target_only, 2, target_spec, draft_spec)

    def test_source_plan_and_json_errors(self):
        old = (0, 1, 2, 3, 0, 1)
        target = (1, 0, 3, 2, 0, 1)
        ranks, slots = plan_placement_sources(old, target, 2)
        self.assertEqual((ranks[0][0], slots[0][0]), (0, 1))
        self.assertEqual((ranks[0][1], slots[0][1]), (0, 0))
        self.assertEqual((ranks[0][2], slots[0][2]), (1, 0))
        for rank in range(2):
            for slot in range(3):
                self.assertEqual(old[ranks[rank][slot] * 3 + slots[rank][slot]], target[rank * 3 + slot])
        with self.assertRaisesRegex(ValueError, "No source"):
            plan_placement_sources((0, 1), (0, 2), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                load_placement_snapshot(path)


class TestPlacementRuntime(unittest.TestCase):
    def setUp(self):
        self.group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)

        def gather(output, value, *, group):
            output[:] = [value] * group.size()

        self.gather = patch.object(runtime.dist, "all_gather_object", side_effect=gather)
        self.gather.start()
        self.addCleanup(self.gather.stop)
        self.barrier = patch.object(runtime.dist, "barrier")
        self.barrier.start()
        self.addCleanup(self.barrier.stop)
        self.leader = patch.object(runtime.dist, "get_global_rank", return_value=0)
        self.leader.start()
        self.addCleanup(self.leader.stop)
        self.broadcast = patch.object(runtime.dist, "broadcast_object_list", side_effect=lambda *_, **__: None)
        self.broadcast.start()
        self.addCleanup(self.broadcast.stop)
        self.target_spec = RuntimeRoleSpec(num_logical_experts=4, num_redundant_experts=2, num_layers=1)
        self.draft_spec = RuntimeRoleSpec(num_logical_experts=2, num_redundant_experts=0, num_layers=1)
        self.snapshot = parse_placement_snapshot(make_snapshot_dict())

    def test_collective_validation_and_consensus_errors(self):
        with (
            patch.object(
                runtime.dist,
                "all_gather_object",
                side_effect=lambda out, _v, *, group: out.__setitem__(slice(None), [None, "ValueError: bad"]),
            ),
            self.assertRaisesRegex(ValueError, "rank 1.*bad"),
        ):
            runtime.collective_call(lambda: 42, self.group, "test")

        with (
            patch.object(
                runtime.dist,
                "all_gather_object",
                side_effect=lambda out, val, *, group: out.__setitem__(
                    slice(None), [None, None] if val is None else [val, "other"]
                ),
            ),
            self.assertRaisesRegex(ValueError, "differs across"),
        ):
            runtime.verify_placement_consensus(self.snapshot, self.group)

    def test_loading_validates_target_and_draft_specs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "deployment.json")
            write_snapshot(path, make_snapshot_dict(with_draft=True))
            loaded = runtime.read_placement_collectively(path, self.target_spec, self.group, draft_spec=self.draft_spec)
            self.assertIsNotNone(loaded.draft)
            wrong_target = RuntimeRoleSpec(num_logical_experts=8, num_redundant_experts=2, num_layers=1)
            with self.assertRaisesRegex(ValueError, "target num_logical_experts mismatch"):
                runtime.read_placement_collectively(path, wrong_target, self.group, draft_spec=self.draft_spec)
            wrong_draft = RuntimeRoleSpec(num_logical_experts=4, num_redundant_experts=0, num_layers=1)
            with self.assertRaisesRegex(ValueError, "draft num_logical_experts mismatch"):
                runtime.read_placement_collectively(path, self.target_spec, self.group, draft_spec=wrong_draft)

    def test_loading_reads_only_on_leader_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "deployment.json")
            write_snapshot(path, make_snapshot_dict())
            with patch.object(runtime.dist, "broadcast_object_list", side_effect=lambda *_, **__: None) as bcast:
                loaded = runtime.read_placement_collectively(path, self.target_spec, self.group)
            self.assertEqual(loaded, self.snapshot)
            self.assertEqual(bcast.call_args.kwargs["src"], 0)

            text = Path(path).read_text(encoding="utf-8")
            self.group.rank = lambda: 1
            with patch.object(
                runtime.dist, "broadcast_object_list", side_effect=lambda obj, *, src, group: obj.__setitem__(0, text)
            ):
                loaded_peer = runtime.read_placement_collectively(
                    str(Path(directory) / "absent.json"), self.target_spec, self.group
                )
            self.assertEqual(loaded_peer, self.snapshot)

            self.group.rank = lambda: 0
            with self.assertRaisesRegex(FileNotFoundError, "could not be read on EPLB rank 0"):
                runtime.read_placement_collectively(str(Path(directory) / "missing.json"), self.target_spec, self.group)

    def _make_install_state(self, rank: int, *, tensor_lists: bool = False):
        old = torch.tensor([[0, 1, 2, 3, 0, 1]])
        global_weights = torch.arange(4, dtype=torch.float32).reshape(4, 1) + 10
        local = old[0, rank * 3 : (rank + 1) * 3]
        weights: list[Any] = [global_weights[local].clone(), (global_weights[local] * 2).clone()]
        buffers: list[Any] = [torch.zeros_like(weight) for weight in weights]
        if tensor_lists:
            weights = [list(weight.unbind()) for weight in weights]
            buffers = [list(buffer.unbind()) for buffer in buffers]

        class Communicator:
            def set_stream(self, stream):
                pass

            def set_transfer_context(self, mapping, layer_idx):
                pass

            def add_send(self, tensors, peer, expert):
                pass

            def add_recv(self, tensors, peer, expert):
                for index, tensor in enumerate(tensors):
                    tensor.copy_(global_weights[expert] * (index + 1))

            def execute(self):
                pass

        model_state = SimpleNamespace(
            model=SimpleNamespace(num_logical_experts=4, num_physical_experts=6, expert_weights=[weights]),
            physical_to_logical_map=old.clone(),
            logical_to_physical_map=torch.full((1, 4, 3), -1, dtype=torch.long),
            logical_replica_count=torch.zeros((1, 4), dtype=torch.long),
            communicator=Communicator(),
            expert_buffer=buffers,
        )
        state = SimpleNamespace(
            model_states={"target": model_state},
            async_worker=None,
            prepare_movement_resources=MagicMock(),
            release_movement_resources=MagicMock(),
            placement_device_group=lambda: SimpleNamespace(size=lambda: 2, rank=lambda: rank),
            refresh_placement_layer=MagicMock(),
        )
        return state, model_state, global_weights

    def test_install_local_swap_remote_exchange_and_guards(self):
        target = (1, 0, 3, 2, 0, 1)
        snapshot = PlacementSnapshot(version=1, ep_size=2, target=RolePlacement(4, 2, (target,)))
        for rank in range(2):
            for tensor_lists in (False, True):
                with self.subTest(rank=rank, tensor_lists=tensor_lists):
                    self.group.rank = lambda r=rank: r
                    state, model_state, global_weights = self._make_install_state(rank, tensor_lists=tensor_lists)
                    expected = global_weights[list(target[rank * 3 : (rank + 1) * 3])]
                    state.refresh_placement_layer.side_effect = lambda committed, _layer, exp=expected: (
                        torch.testing.assert_close(torch.stack(list(committed.model.expert_weights[0][0])), exp)
                    )
                    runtime.install_placement(state, snapshot, self.group, MagicMock())
                    state.refresh_placement_layer.assert_called_once_with(model_state, 0)
                    self.assertEqual(model_state.logical_replica_count.tolist(), [[2, 2, 1, 1]])
                    state.prepare_movement_resources.assert_called_once_with()
                    state.release_movement_resources.assert_called_once_with()

        state, _, _ = self._make_install_state(0)
        unchanged = PlacementSnapshot(version=1, ep_size=2, target=RolePlacement(4, 2, ((0, 1, 2, 3, 0, 1),)))
        runtime.install_placement(state, unchanged, self.group, MagicMock())
        state.prepare_movement_resources.assert_not_called()

        state.async_worker = object()
        with self.assertRaisesRegex(ValueError, "precede the background worker"):
            runtime.install_placement(state, self.snapshot, self.group, MagicMock())

        state.async_worker = None
        with_draft = parse_placement_snapshot(make_snapshot_dict(with_draft=True))
        with self.assertRaisesRegex(ValueError, "missing its draft model state"):
            runtime.install_placement(state, with_draft, self.group, MagicMock(), model_states=state.model_states)

    def test_pp_stage_slicing_and_collective_read(self):
        raw = make_snapshot_dict(with_draft=True)
        raw["target"]["physical_to_logical_map"] = [[0, 1, 2, 0, 1, 3], [1, 2, 3, 0, 1, 2], [2, 3, 0, 1, 2, 0]]
        snap = parse_placement_snapshot(raw)
        s0 = RuntimeRoleSpec(num_logical_experts=4, num_redundant_experts=2, num_layers=1)
        s2 = RuntimeRoleSpec(num_logical_experts=4, num_redundant_experts=2, num_layers=2)
        specs = [(s0, None), (None, None), (s2, self.draft_spec)]

        self.assertEqual(runtime.slice_pp_stage_roles(snap, 2, 1, specs), {})
        r2 = runtime.slice_pp_stage_roles(snap, 2, 2, specs)
        self.assertEqual(r2["target"].physical_to_logical_map, ((1, 2, 3, 0, 1, 2), (2, 3, 0, 1, 2, 0)))
        self.assertEqual(r2["draft"].physical_to_logical_map, ((0, 1),))
        with self.assertRaisesRegex(ValueError, "target layer count mismatch"):
            runtime.slice_pp_stage_roles(snap, 2, 0, [(s0, None), (s0, self.draft_spec)])

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "pp.json")
            write_snapshot(path, raw)
            pp_group = SimpleNamespace(size=lambda: 3, rank=lambda: 2)
            text = Path(path).read_text(encoding="utf-8")

            def gather_any(out, val, *, group):
                out[:] = [(specs[0], None), (specs[1], None), (specs[2], None)] if group is pp_group else [val] * 2

            def bcast_any(obj, *, src, group):
                if group is pp_group:
                    obj[0] = text

            with (
                patch.object(runtime.dist, "all_gather_object", side_effect=gather_any),
                patch.object(runtime.dist, "broadcast_object_list", side_effect=bcast_any),
            ):
                loaded, stage_roles = runtime.read_pp_placement_collectively(
                    str(Path(directory) / "absent_on_stage2.json"),
                    lambda: specs[2],
                    self.group,
                    pp_group,
                    2,
                )
            self.assertEqual(loaded, snap)
            self.assertEqual(stage_roles, r2)
