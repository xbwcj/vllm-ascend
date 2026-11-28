# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from typing import Any

import torch
import torch.nn as nn
from vllm.config.utils import hash_config_cached
from vllm.distributed import get_ep_group
from vllm.distributed import parallel_state as _parallel_state
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    is_mixture_of_experts,
)
from vllm.v1.worker.gpu.eplb_utils import EPLBController

from vllm_ascend.ascend_config import EplbConfig
from vllm_ascend.device.hardware_profile import MoECommPolicy, get_current_hardware_profile
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbState
from vllm_ascend.distributed.eplb.placement_runtime import (
    RuntimeRoleSpec,
    build_runtime_role_spec,
    collective_call,
    install_placement,
    read_placement_collectively,
    read_pp_placement_collectively,
)
from vllm_ascend.distributed.eplb.policy.factory import create_eplb_policy
from vllm_ascend.patch.platform.patch_eplb import resolve_ascend_eplb_communicator


def _get_pp_topology() -> tuple[int, int, Any | None]:
    if getattr(_parallel_state, "_PP", None) is None and get_pp_group is _parallel_state.get_pp_group:
        return 0, 1, None
    try:
        pp_group = get_pp_group()
    except AssertionError:
        return 0, 1, None
    return pp_group.rank_in_group, pp_group.world_size, getattr(pp_group, "cpu_group", None)


def _config_hash(config: Any) -> str:
    try:
        return hash_config_cached(config)
    except TypeError:
        if hasattr(config, "compute_hash"):
            return str(config.compute_hash())
        raise


def is_eplb_load_collection_phase_matched(
    load_collection_phase: str,
    batch_has_prefill: bool,
) -> bool:
    """Return whether the batch belongs to the configured collection phase."""
    if load_collection_phase == "all":
        return True
    batch_phase = "prefill" if batch_has_prefill else "decode"
    return load_collection_phase == batch_phase


def _unwrap_moe(model: nn.Module) -> nn.Module:
    if not is_mixture_of_experts(model) and isinstance(model, SupportsMultiModal):
        return model.get_language_model()
    return model


class AscendEPLBController(EPLBController):
    """Construct Ascend state and apply phase-filtered load collection."""

    def __init__(
        self,
        parallel_config: Any,
        device: torch.device,
        ascend_eplb_config: EplbConfig | None = None,
    ) -> None:
        super().__init__(parallel_config, device)
        ascend_eplb_config = ascend_eplb_config or EplbConfig()
        self.load_collection_phase = ascend_eplb_config.load_collection_phase
        self.expert_placement_path = ascend_eplb_config.expert_placement_path
        # The communicator choice needs one binding class on every EPLB rank,
        # so reach a group consensus before the policy or state is built.
        stair_config = (
            resolve_ascend_eplb_communicator(parallel_config, ascend_eplb_config)
            if parallel_config.enable_eplb
            else ascend_eplb_config.stair_config
        )
        self.eplb_policy = create_eplb_policy(
            parallel_config.eplb_config.policy,
            stair_config,
        )
        self._load_collection_phase_matched = True
        self.mode = ascend_eplb_config.mode
        eplb_config = getattr(parallel_config, "eplb_config", None)
        self._static_observability_enabled = self.mode == "static" and bool(
            getattr(eplb_config, "log_balancedness", False)
        )
        self._placement_model_config: Any | None = None
        self._draft_placement_model_config: Any | None = None

    def prepare_load(self) -> None:
        self.state = None
        self._has_registered_models = False
        self._placement_model_config = None
        self._draft_placement_model_config = None
        if self.parallel_config.enable_eplb:
            self.state = AscendEplbState(self.parallel_config, self.device, self.eplb_policy)
            self.state.static_mode = self.mode == "static"
            if (
                self.mode == "dynamic"
                and self.expert_placement_path is not None
                and hasattr(self.state, "enable_placement_export")
            ):
                self.state.enable_placement_export(self.expert_placement_path)

    def maybe_register_model(self, model: nn.Module, model_config: Any, *args, **kwargs) -> bool:
        added = super().maybe_register_model(model, model_config, *args, **kwargs)
        if added:
            self._placement_model_config = model_config
            if self.state is not None and self.mode == "static":
                self.state.disable_static_recording(force=True)
        return added

    def _target_placement_model_state(self) -> Any:
        if self.state is None or self._placement_model_config is None:
            raise ValueError("EPLB deployments require exactly one registered target MoE model")
        model_hash = _config_hash(self._placement_model_config)
        model_state = self.state.model_states.get(model_hash)
        if model_state is None:
            raise ValueError("EPLB deployment could not locate the registered target model state")
        return model_state

    def _draft_placement_model_state(self) -> Any:
        if self.state is None or self._draft_placement_model_config is None:
            raise ValueError("EPLB deployment could not locate the registered draft model state")
        model_hash = _config_hash(self._draft_placement_model_config)
        model_state = self.state.model_states.get(model_hash)
        if model_state is None:
            raise ValueError("EPLB deployment could not locate the registered draft model state")
        return model_state

    def _target_runtime_spec(self) -> RuntimeRoleSpec:
        return build_runtime_role_spec(self._target_placement_model_state())

    def _optional_target_runtime_spec(self) -> RuntimeRoleSpec | None:
        if self._placement_model_config is None:
            return None
        return build_runtime_role_spec(self._target_placement_model_state())

    def _optional_draft_runtime_spec(self) -> RuntimeRoleSpec | None:
        if self._draft_placement_model_config is None:
            return None
        return build_runtime_role_spec(self._draft_placement_model_state())

    def _placement_model_states(self) -> dict[str, Any]:
        model_states = {"target": self._target_placement_model_state()}
        if self._draft_placement_model_config is not None:
            model_states["draft"] = self._draft_placement_model_state()
        return model_states

    def _stage_placement_model_states(self) -> dict[str, Any]:
        model_states: dict[str, Any] = {}
        if self._placement_model_config is not None:
            model_states["target"] = self._target_placement_model_state()
        if self._draft_placement_model_config is not None:
            model_states["draft"] = self._draft_placement_model_state()
        return model_states

    def maybe_start_async_loop(
        self,
        eplb_models_added: bool,
        load_dummy_weights: bool = False,
    ) -> None:
        if self.mode != "static":
            if (
                not load_dummy_weights
                and self.state is not None
                and self.expert_placement_path is not None
                and hasattr(self.state, "setup_pp_placement_export")
            ):
                self.state.setup_pp_placement_export()
            super().maybe_start_async_loop(eplb_models_added, load_dummy_weights=load_dummy_weights)
            return
        if load_dummy_weights or self.state is None:
            return
        pp_rank, pp_size, pp_cpu_group = _get_pp_topology()
        if pp_size <= 1:
            if not eplb_models_added:
                return
            group = get_ep_group().cpu_group
            target_spec = collective_call(self._target_runtime_spec, group, "static target spec")
            draft_spec = collective_call(
                self._optional_draft_runtime_spec,
                group,
                "static draft spec",
            )
            assert self.expert_placement_path is not None
            snapshot = read_placement_collectively(
                self.expert_placement_path,
                target_spec,
                group,
                draft_spec=draft_spec,
            )
            install_placement(
                self.state,
                snapshot,
                group,
                torch.accelerator.current_stream(self.device),
                model_states=self._placement_model_states(),
            )
        else:
            if pp_cpu_group is None:
                raise RuntimeError("PP CPU group is required for multi-stage static EPLB.")
            assert self.expert_placement_path is not None
            ep_cpu_group = get_ep_group().cpu_group
            snapshot, stage_roles = read_pp_placement_collectively(
                self.expert_placement_path,
                lambda: (self._optional_target_runtime_spec(), self._optional_draft_runtime_spec()),
                ep_cpu_group,
                pp_cpu_group,
                pp_rank,
            )
            if stage_roles:
                install_placement(
                    self.state,
                    snapshot,
                    ep_cpu_group,
                    torch.accelerator.current_stream(self.device),
                    model_states=self._stage_placement_model_states(),
                    role_placements=stage_roles,
                )
        logger.info(
            "Ascend static EPLB placement installed: sha256=%s target_layers=%d draft_layers=%d",
            snapshot.digest(),
            snapshot.target.num_layers,
            0 if snapshot.draft is None else snapshot.draft.num_layers,
        )

    def step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        if self.mode == "static":
            if (
                self.state is not None
                and self._static_observability_enabled
                and (is_dummy or is_profile or self._load_collection_phase_matched)
            ):
                self.state.step_static_observability(
                    is_dummy=is_dummy,
                    is_profile=is_profile,
                )
            return
        super().step(is_dummy=is_dummy, is_profile=is_profile)

    def set_batch_phase(self, batch_has_prefill: bool) -> None:
        self._load_collection_phase_matched = is_eplb_load_collection_phase_matched(
            self.load_collection_phase,
            batch_has_prefill,
        )

    def maybe_register_speculator(
        self,
        speculator: Any | None,
        speculative_config: Any | None,
        load_dummy_weights: bool,
    ) -> bool:
        # The upstream controller checks target EPLB, which can differ from
        # the replicated draft's setting.
        if speculator is not None:
            draft_vllm_config = getattr(speculator, "draft_vllm_config", None)
            draft_parallel_config = (
                draft_vllm_config.parallel_config
                if draft_vllm_config is not None
                else speculator.vllm_config.parallel_config
            )
            if not draft_parallel_config.enable_eplb:
                return False
        added = super().maybe_register_speculator(speculator, speculative_config, load_dummy_weights)
        if added:
            self._draft_placement_model_config = getattr(
                speculative_config,
                "draft_model_config",
                None,
            )
            if self.state is not None and self.mode == "static":
                self.state.disable_static_recording(force=True)
        return added

    def prepare_forward(
        self,
        model_config: Any,
        num_unpadded_tokens: int,
        ubatch_slices: list | None = None,
    ) -> None:
        state = self.state
        if state is None or not self.parallel_config.enable_eplb:
            return
        if self.mode == "static":
            if not self._static_observability_enabled or not self._load_collection_phase_matched:
                state.disable_static_recording()
                state._static_observation_due = False
                state._is_load_sampling_step = False
                state._should_collect_local_load = False
                return
            if state._static_observation_due and state._static_recording_enabled:
                state.step_static_observability()
            should_record = state.update_static_observation_schedule() and state.should_record_tensor is not None
            state.set_static_recording_enabled(should_record)
            if state.should_record_tensor is not None:
                state.should_record_tensor.fill_(should_record)
            if should_record and state.allgather_record_tensor is not None:
                state.allgather_record_tensor.fill_(-1)
            state._is_load_sampling_step = should_record
            state._should_collect_local_load = should_record
            return
        supports_allgather = get_current_hardware_profile().moe_comm_policy is MoECommPolicy.CAPACITY_AND_WORLD_SIZE
        if not state.uses_custom_load_stats or supports_allgather:
            state.prepare_forward(model_config, num_unpadded_tokens, ubatch_slices)
            if not supports_allgather:
                return
        if supports_allgather:
            state._uses_phase_filter = self.load_collection_phase != "all"
            if state.allgather_record_tensor is not None:
                state.allgather_record_tensor.fill_(-1)
        # AllGather needs source validity; other paths use operator counts.
        if state.should_record_tensor is not None:
            is_sampling = state._should_record_current_step(log_stats=self.parallel_config.eplb_config.log_balancedness)
            should_record = is_sampling and self._load_collection_phase_matched
            state.should_record_tensor.fill_(should_record)
            state._is_load_sampling_step = is_sampling
            state._should_collect_local_load = should_record
            if should_record and not supports_allgather:
                state._has_fresh_recorded_load = True

    # The two supported upstream revisions intentionally expose different signatures.
    def setup_from_mapping(
        self,
        model_config: Any,
        expanded_physical_to_logical: torch.Tensor,
    ) -> None:
        # Elastic EP reuses the live state and communicator on verified main.
        super().setup_from_mapping(model_config, expanded_physical_to_logical)
