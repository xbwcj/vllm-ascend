# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.distributed.eplb.placement_runtime import RuntimeRoleSpec
from vllm_ascend.worker.v2 import eplb as module
from vllm_ascend.worker.v2.eplb import AscendEPLBController


class TestStaticController(unittest.TestCase):
    def make_controller(self, *, log_balancedness: bool = False):
        controller = AscendEPLBController.__new__(AscendEPLBController)
        controller.mode = "static"
        controller._static_observability_enabled = log_balancedness
        controller.expert_placement_path = "deployment.json"
        controller.state = MagicMock()
        controller.device = "npu:0"  # type: ignore[assignment]
        controller.eplb_policy = MagicMock()
        controller.parallel_config = SimpleNamespace(
            enable_eplb=True,
            eplb_config=SimpleNamespace(log_balancedness=log_balancedness),
        )
        controller.load_collection_phase = "all"
        controller._load_collection_phase_matched = True
        controller.state.update_static_observation_schedule.return_value = False
        controller.state._static_observation_due = False
        controller.state._static_recording_enabled = False
        controller._placement_model_config = None
        controller._draft_placement_model_config = None
        return controller

    def test_static_prepare_load_sets_static_mode_without_export(self):
        controller = self.make_controller()
        state = MagicMock()
        with patch.object(module, "AscendEplbState", return_value=state):
            controller.prepare_load()
        self.assertTrue(state.static_mode)
        state.enable_placement_export.assert_not_called()

    def test_static_controller_skips_runtime_hooks(self):
        controller = self.make_controller()
        controller.prepare_forward(object(), 4)
        controller.step()
        controller.step(is_dummy=True, is_profile=True)
        controller.state.disable_static_recording.assert_called_once_with()
        controller.state.set_static_recording_enabled.assert_not_called()
        controller.state.should_record_tensor.fill_.assert_not_called()
        controller.state.prepare_forward.assert_not_called()
        controller.state.step.assert_not_called()
        controller.state.step_static_observability.assert_not_called()

    def test_static_controller_schedules_lightweight_observation(self):
        controller = self.make_controller(log_balancedness=True)
        controller.state.update_static_observation_schedule.return_value = True
        controller.prepare_forward(object(), 4)
        controller.state.update_static_observation_schedule.assert_called_once_with()
        controller.state.should_record_tensor.fill_.assert_called_once_with(True)
        controller.state.set_static_recording_enabled.assert_called_once_with(True)
        self.assertTrue(controller.state._is_load_sampling_step)
        self.assertTrue(controller.state._should_collect_local_load)
        controller.state.prepare_forward.assert_not_called()

        controller.step()
        controller.state.step_static_observability.assert_called_once_with(
            is_dummy=False,
            is_profile=False,
        )
        controller.state.step.assert_not_called()

    def test_static_initialization_loads_before_install_without_worker(self):
        controller = self.make_controller()
        target_spec = RuntimeRoleSpec(num_logical_experts=4, num_redundant_experts=2, num_layers=1)
        controller._target_runtime_spec = MagicMock(return_value=target_spec)  # type: ignore[method-assign]
        controller._optional_draft_runtime_spec = MagicMock(return_value=None)  # type: ignore[method-assign]
        model_states = {"target": object()}
        controller._placement_model_states = MagicMock(return_value=model_states)  # type: ignore[method-assign]
        group = object()
        stream = object()
        snapshot = SimpleNamespace(
            digest=lambda: "test-digest",
            target=SimpleNamespace(num_layers=1),
            draft=None,
        )
        with (
            patch.object(module, "get_ep_group", return_value=SimpleNamespace(cpu_group=group)),
            patch.object(module, "collective_call", side_effect=lambda fn, *_: fn()),
            patch.object(module, "read_placement_collectively", return_value=snapshot) as read,
            patch.object(module, "install_placement") as install,
            patch.object(module.torch.accelerator, "current_stream", return_value=stream),
        ):
            controller.maybe_start_async_loop(True)
        read.assert_called_once_with(
            "deployment.json",
            target_spec,
            group,
            draft_spec=None,
        )
        install.assert_called_once_with(
            controller.state,
            snapshot,
            group,
            stream,
            model_states=model_states,
        )
        controller.state.start_async_loop.assert_not_called()

    def test_static_initialization_passes_draft_model_state(self):
        controller = self.make_controller()
        target_spec = RuntimeRoleSpec(num_logical_experts=4, num_redundant_experts=2, num_layers=1)
        draft_spec = RuntimeRoleSpec(num_logical_experts=2, num_redundant_experts=0, num_layers=1)
        controller._target_runtime_spec = MagicMock(return_value=target_spec)  # type: ignore[method-assign]
        controller._optional_draft_runtime_spec = MagicMock(return_value=draft_spec)  # type: ignore[method-assign]
        model_states = {"target": object(), "draft": object()}
        controller._placement_model_states = MagicMock(return_value=model_states)  # type: ignore[method-assign]
        group = object()
        stream = object()
        snapshot = SimpleNamespace(
            digest=lambda: "multi-digest",
            target=SimpleNamespace(num_layers=1),
            draft=SimpleNamespace(num_layers=1),
        )
        with (
            patch.object(module, "get_ep_group", return_value=SimpleNamespace(cpu_group=group)),
            patch.object(module, "collective_call", side_effect=lambda fn, *_: fn()),
            patch.object(module, "read_placement_collectively", return_value=snapshot) as read,
            patch.object(module, "install_placement") as install,
            patch.object(module.torch.accelerator, "current_stream", return_value=stream),
        ):
            controller.maybe_start_async_loop(True)
        read.assert_called_once_with("deployment.json", target_spec, group, draft_spec=draft_spec)
        install.assert_called_once_with(controller.state, snapshot, group, stream, model_states=model_states)

    def test_dummy_loading_does_not_install_or_start_worker(self):
        controller = self.make_controller()
        with patch.object(module, "install_placement") as install:
            controller.maybe_start_async_loop(True, load_dummy_weights=True)
            controller.maybe_start_async_loop(False)
        install.assert_not_called()
        controller.state.start_async_loop.assert_not_called()

    def test_static_pp_slicing_and_phase_filtered_observation(self):
        controller = self.make_controller(log_balancedness=True)
        controller.load_collection_phase = "decode"
        controller.set_batch_phase(batch_has_prefill=True)
        controller.prepare_forward(object(), 4)
        controller.step()
        controller.state.update_static_observation_schedule.assert_not_called()
        controller.state.step_static_observability.assert_not_called()

        ep_group, pp_group, stream = object(), object(), object()
        snapshot = SimpleNamespace(
            digest=lambda: "pp-digest",
            target=SimpleNamespace(num_layers=2),
            draft=SimpleNamespace(num_layers=1),
        )
        stage_roles = {"target": object(), "draft": object()}
        controller._stage_placement_model_states = MagicMock(return_value={"target": "t", "draft": "d"})  # type: ignore[method-assign]
        with (
            patch.object(module, "_get_pp_topology", return_value=(1, 2, pp_group)),
            patch.object(module, "get_ep_group", return_value=SimpleNamespace(cpu_group=ep_group)),
            patch.object(module, "read_pp_placement_collectively", return_value=(snapshot, stage_roles)) as read_pp,
            patch.object(module, "install_placement") as install,
            patch.object(module.torch.accelerator, "current_stream", return_value=stream),
        ):
            controller.maybe_start_async_loop(True)
            read_pp.return_value = (snapshot, {})
            controller.maybe_start_async_loop(False)
        self.assertEqual(read_pp.call_count, 2)
        install.assert_called_once_with(
            controller.state,
            snapshot,
            ep_group,
            stream,
            model_states={"target": "t", "draft": "d"},
            role_placements=stage_roles,
        )
