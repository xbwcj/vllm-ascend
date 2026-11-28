# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Version-1 EPLB placement snapshot loading, source planning, and static installation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed.eplb import eplb_state as upstream_state
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from vllm_ascend.distributed.eplb.explicit_transfer import stage_explicit_layer_transfer

PLACEMENT_SNAPSHOT_VERSION = 1
_REQUIRED_KEYS = frozenset({"version", "ep_size", "target"})
_MODEL_KEYS = frozenset({"num_logical_experts", "num_redundant_experts", "physical_to_logical_map"})
_T = TypeVar("_T")


def _require_int(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class RolePlacement:
    num_logical_experts: int
    num_redundant_experts: int
    physical_to_logical_map: tuple[tuple[int, ...], ...]

    @property
    def num_physical_experts(self) -> int:
        return self.num_logical_experts + self.num_redundant_experts

    @property
    def num_layers(self) -> int:
        return len(self.physical_to_logical_map)


@dataclass(frozen=True)
class PlacementSnapshot:
    version: int
    ep_size: int
    target: RolePlacement
    draft: RolePlacement | None = None

    def digest(self) -> str:
        payload: dict[str, Any] = {
            "version": self.version,
            "ep_size": self.ep_size,
            "target": {
                "num_logical_experts": self.target.num_logical_experts,
                "num_redundant_experts": self.target.num_redundant_experts,
                "physical_to_logical_map": [list(row) for row in self.target.physical_to_logical_map],
            },
        }
        if self.draft is not None:
            payload["draft"] = {
                "num_logical_experts": self.draft.num_logical_experts,
                "num_redundant_experts": self.draft.num_redundant_experts,
                "physical_to_logical_map": [list(row) for row in self.draft.physical_to_logical_map],
            }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RuntimeRoleSpec:
    num_logical_experts: int
    num_redundant_experts: int
    num_layers: int


def build_runtime_role_spec(model_state: Any) -> RuntimeRoleSpec:
    num_logical = int(model_state.model.num_logical_experts)
    num_layers, num_physical = (int(dim) for dim in model_state.physical_to_logical_map.shape)
    model_physical = getattr(model_state.model, "num_physical_experts", None)
    if model_physical is not None and int(model_physical) != num_physical:
        raise ValueError(
            f"Runtime physical expert count ({int(model_physical)}) does not match mapping width ({num_physical})"
        )
    num_redundant = num_physical - num_logical
    if num_logical < 1 or num_redundant < 0 or num_layers < 1:
        raise ValueError("Invalid runtime MoE expert dimensions")
    return RuntimeRoleSpec(
        num_logical_experts=num_logical,
        num_redundant_experts=num_redundant,
        num_layers=num_layers,
    )


def _parse_role_entry(entry: Any, role: str, ep_size: int) -> RolePlacement:
    if not isinstance(entry, dict) or set(entry) != _MODEL_KEYS:
        raise ValueError(f"EPLB {role} placement must contain exactly {sorted(_MODEL_KEYS)}")
    num_logical = _require_int(entry["num_logical_experts"], f"{role}.num_logical_experts", 1)
    num_redundant = _require_int(entry["num_redundant_experts"], f"{role}.num_redundant_experts", 0)
    num_physical = num_logical + num_redundant
    if num_physical % ep_size != 0:
        raise ValueError(f"EPLB {role} physical expert count ({num_physical}) must be divisible by ep_size")
    mapping = entry["physical_to_logical_map"]
    if not isinstance(mapping, list) or not mapping or any(not isinstance(row, list) for row in mapping):
        raise ValueError(f"EPLB {role}.physical_to_logical_map must be a non-empty list of layer lists")

    slots_per_rank = num_physical // ep_size
    expected = set(range(num_logical))
    rows: list[tuple[int, ...]] = []
    for layer_idx, row in enumerate(mapping):
        if len(row) != num_physical:
            raise ValueError(f"EPLB {role} layer {layer_idx} has an invalid physical slot count")
        if any(type(e) is not int or not 0 <= e < num_logical for e in row):
            raise ValueError(f"EPLB {role} layer {layer_idx} contains an invalid logical expert ID")
        if set(row) != expected:
            raise ValueError(f"EPLB {role} layer {layer_idx} does not cover every logical expert")
        for rank in range(ep_size):
            slot_slice = row[rank * slots_per_rank : (rank + 1) * slots_per_rank]
            if len(set(slot_slice)) != slots_per_rank:
                raise ValueError(f"EPLB {role} layer {layer_idx} contains duplicate logical experts on rank {rank}")
        rows.append(tuple(row))
    return RolePlacement(
        num_logical_experts=num_logical,
        num_redundant_experts=num_redundant,
        physical_to_logical_map=tuple(rows),
    )


def parse_placement_snapshot(snapshot: Any) -> PlacementSnapshot:
    if not isinstance(snapshot, dict):
        raise ValueError("EPLB placement snapshot must be a JSON object")
    if type(snapshot.get("version")) is not int or snapshot["version"] != PLACEMENT_SNAPSHOT_VERSION:
        raise ValueError(f"Unsupported EPLB placement version: {snapshot.get('version')!r}")
    keys = set(snapshot)
    if _REQUIRED_KEYS - keys or keys - _REQUIRED_KEYS - {"draft"}:
        raise ValueError(f"EPLB placement snapshot must contain {sorted(_REQUIRED_KEYS)} and optional 'draft'")
    ep_size = _require_int(snapshot["ep_size"], "ep_size", 1)
    target = _parse_role_entry(snapshot["target"], "target", ep_size)
    draft = _parse_role_entry(snapshot["draft"], "draft", ep_size) if "draft" in snapshot else None
    return PlacementSnapshot(
        version=PLACEMENT_SNAPSHOT_VERSION,
        ep_size=ep_size,
        target=target,
        draft=draft,
    )


def load_placement_snapshot(path: str | os.PathLike[str]) -> PlacementSnapshot:
    with open(path, encoding="utf-8") as source:
        return parse_placement_snapshot(json.load(source))


def _validate_role_against_spec(placement: RolePlacement, role: str, spec: RuntimeRoleSpec) -> None:
    if placement.num_logical_experts != spec.num_logical_experts:
        raise ValueError(
            f"EPLB {role} num_logical_experts mismatch: "
            f"file={placement.num_logical_experts}, runtime={spec.num_logical_experts}"
        )
    if placement.num_redundant_experts != spec.num_redundant_experts:
        raise ValueError(
            f"EPLB {role} num_redundant_experts mismatch: "
            f"file={placement.num_redundant_experts}, runtime={spec.num_redundant_experts}"
        )
    if placement.num_layers != spec.num_layers:
        raise ValueError(f"EPLB {role} layer count mismatch: file={placement.num_layers}, runtime={spec.num_layers}")


def validate_snapshot_for_runtime(
    snapshot: PlacementSnapshot,
    ep_size: int,
    target_spec: RuntimeRoleSpec,
    draft_spec: RuntimeRoleSpec | None = None,
) -> None:
    if snapshot.ep_size != ep_size:
        raise ValueError(f"EPLB placement ep_size mismatch: file={snapshot.ep_size}, runtime={ep_size}")
    _validate_role_against_spec(snapshot.target, "target", target_spec)
    if draft_spec is not None:
        if snapshot.draft is None:
            raise ValueError("EPLB placement snapshot is missing the expected draft placement")
        _validate_role_against_spec(snapshot.draft, "draft", draft_spec)


def _validate_role_experts(placement: RolePlacement, role: str, spec: RuntimeRoleSpec) -> None:
    if placement.num_logical_experts != spec.num_logical_experts:
        raise ValueError(
            f"EPLB {role} num_logical_experts mismatch: "
            f"file={placement.num_logical_experts}, runtime={spec.num_logical_experts}"
        )
    if placement.num_redundant_experts != spec.num_redundant_experts:
        raise ValueError(
            f"EPLB {role} num_redundant_experts mismatch: "
            f"file={placement.num_redundant_experts}, runtime={spec.num_redundant_experts}"
        )


def slice_pp_stage_roles(
    snapshot: PlacementSnapshot,
    ep_size: int,
    pp_rank: int,
    stage_specs: Sequence[tuple[RuntimeRoleSpec | None, RuntimeRoleSpec | None]],
) -> dict[str, RolePlacement]:
    if snapshot.ep_size != ep_size:
        raise ValueError(f"EPLB placement ep_size mismatch: file={snapshot.ep_size}, runtime={ep_size}")
    target_specs = [spec for spec, _ in stage_specs if spec is not None]
    if not target_specs:
        raise ValueError("EPLB deployments require at least one PP stage with target MoE layers")
    for spec in target_specs:
        _validate_role_experts(snapshot.target, "target", spec)
    total_target_layers = sum(spec.num_layers for spec in target_specs)
    if snapshot.target.num_layers != total_target_layers:
        raise ValueError(
            f"EPLB target layer count mismatch: file={snapshot.target.num_layers}, runtime={total_target_layers}"
        )

    draft_specs = [spec for _, spec in stage_specs if spec is not None]
    if draft_specs:
        if snapshot.draft is None:
            raise ValueError("EPLB placement snapshot is missing the expected draft placement")
        for spec in draft_specs:
            _validate_role_experts(snapshot.draft, "draft", spec)
        total_draft_layers = sum(spec.num_layers for spec in draft_specs)
        if snapshot.draft.num_layers != total_draft_layers:
            raise ValueError(
                f"EPLB draft layer count mismatch: file={snapshot.draft.num_layers}, runtime={total_draft_layers}"
            )
    elif snapshot.draft is not None:
        raise ValueError("Static EPLB placement is missing its draft model state")

    local_target, local_draft = stage_specs[pp_rank]
    roles: dict[str, RolePlacement] = {}
    if local_target is not None:
        t_off = sum(spec.num_layers for spec, _ in stage_specs[:pp_rank] if spec is not None)
        roles["target"] = RolePlacement(
            num_logical_experts=snapshot.target.num_logical_experts,
            num_redundant_experts=snapshot.target.num_redundant_experts,
            physical_to_logical_map=snapshot.target.physical_to_logical_map[t_off : t_off + local_target.num_layers],
        )
    if local_draft is not None:
        assert snapshot.draft is not None
        d_off = sum(spec.num_layers for _, spec in stage_specs[:pp_rank] if spec is not None)
        roles["draft"] = RolePlacement(
            num_logical_experts=snapshot.draft.num_logical_experts,
            num_redundant_experts=snapshot.draft.num_redundant_experts,
            physical_to_logical_map=snapshot.draft.physical_to_logical_map[d_off : d_off + local_draft.num_layers],
        )
    return roles


def plan_placement_sources(
    old: tuple[int, ...],
    new: tuple[int, ...],
    ep_size: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Choose one exact old source, preferring unchanged and rank-local slots."""
    if not old or len(old) != len(new) or ep_size < 1 or len(old) % ep_size:
        raise ValueError("EPLB source planning requires equally sized dense EP placements")
    slots = len(old) // ep_size
    owners: dict[int, list[int]] = {}
    for physical, expert in enumerate(old):
        owners.setdefault(expert, []).append(physical)
    source_ranks, source_slots = [], []
    for rank in range(ep_size):
        ranks, offsets = [], []
        for slot in range(slots):
            physical = rank * slots + slot
            candidates = owners.get(new[physical], [])
            if not candidates:
                raise ValueError(f"No source owns logical expert {new[physical]}")
            source = min(candidates, key=lambda index: (index != physical, index // slots != rank, index))
            ranks.append(source // slots)
            offsets.append(source % slots)
        source_ranks.append(ranks)
        source_slots.append(offsets)
    return source_ranks, source_slots


def collective_call(operation: Callable[[], _T], group: Any, description: str) -> _T:
    """Report local validation/I/O failures before any peer starts migration."""
    result = None
    error = None
    try:
        result = operation()
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    errors: list[Any] = [None] * group.size()
    dist.all_gather_object(errors, error, group=group)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message is not None]
    if failures:
        raise ValueError(f"EPLB {description} failed: {'; '.join(failures)}")
    return result  # type: ignore[return-value]


def verify_placement_consensus(snapshot: PlacementSnapshot, group: Any) -> str:
    digest = collective_call(snapshot.digest, group, "placement validation")
    digests: list[Any] = [None] * group.size()
    dist.all_gather_object(digests, digest, group=group)
    if len(set(digests)) != 1:
        raise ValueError("EPLB placement differs across EP ranks")
    return digest


def read_placement_collectively(
    path: str,
    target_spec: RuntimeRoleSpec,
    group: Any,
    *,
    draft_spec: RuntimeRoleSpec | None = None,
) -> PlacementSnapshot:
    """Read the placement snapshot on EP rank 0 and broadcast it to every rank."""
    payload: list[str | None] = [None]
    if group.rank() == 0:
        try:
            with open(path, encoding="utf-8") as source:
                payload[0] = source.read()
        except OSError as error:
            payload[0] = f"ERROR: {type(error).__name__}: {error}"
    dist.broadcast_object_list(payload, src=dist.get_global_rank(group, 0), group=group)
    agreed = payload[0]
    if agreed is None or agreed.startswith("ERROR:"):
        raise FileNotFoundError(f"EPLB placement {path!r} could not be read on EPLB rank 0: {agreed}")

    def decode() -> PlacementSnapshot:
        snapshot = parse_placement_snapshot(json.loads(agreed))
        validate_snapshot_for_runtime(snapshot, group.size(), target_spec, draft_spec)
        return snapshot

    snapshot = collective_call(decode, group, "placement loading")
    verify_placement_consensus(snapshot, group)
    return snapshot


def read_pp_placement_collectively(
    path: str,
    local_specs_fn: Callable[[], tuple[RuntimeRoleSpec | None, RuntimeRoleSpec | None]],
    ep_group: Any,
    pp_group: Any,
    pp_rank: int,
) -> tuple[PlacementSnapshot, dict[str, RolePlacement]]:
    """Read the snapshot on global rank 0, broadcast across PP+EP, and slice per PP stage."""
    local_pair: tuple[RuntimeRoleSpec | None, RuntimeRoleSpec | None] | None = None
    local_err: str | None = None
    try:
        local_pair = local_specs_fn()
    except Exception as exc:
        local_err = f"{type(exc).__name__}: {exc}"
    ep_items: list[Any] = [None] * ep_group.size()
    dist.all_gather_object(ep_items, (local_pair, local_err), group=ep_group)
    ep_errs = [f"ep_rank {r}: {err}" for r, (_, err) in enumerate(ep_items) if err is not None]
    stage_err = "; ".join(ep_errs) if ep_errs else None
    if stage_err is None and len({item[0] for item in ep_items}) != 1:
        stage_err = "stage MoE specs differ across EP ranks"
    stage_pair = ep_items[0][0]

    stage_packet: list[Any] = [None, None]
    if ep_group.rank() == 0:
        pp_text: list[str | None] = [None]
        if pp_rank == 0:
            try:
                with open(path, encoding="utf-8") as source:
                    pp_text[0] = source.read()
            except OSError as error:
                pp_text[0] = f"ERROR: {type(error).__name__}: {error}"
        dist.broadcast_object_list(pp_text, src=dist.get_global_rank(pp_group, 0), group=pp_group)
        pp_gathered: list[Any] = [None] * pp_group.size()
        dist.all_gather_object(pp_gathered, (stage_pair, stage_err), group=pp_group)
        stage_packet = [pp_text[0], pp_gathered]
    dist.broadcast_object_list(stage_packet, src=dist.get_global_rank(ep_group, 0), group=ep_group)

    agreed_text, pp_gathered = stage_packet
    pp_errors = [f"pp_rank {r}: {err}" for r, (_, err) in enumerate(pp_gathered) if err is not None]
    if pp_errors:
        raise ValueError(f"EPLB static PP spec collection failed: {'; '.join(pp_errors)}")
    if agreed_text is None or agreed_text.startswith("ERROR:"):
        raise FileNotFoundError(f"EPLB placement {path!r} could not be read on global rank 0: {agreed_text}")
    all_stage_specs = [pair for pair, _ in pp_gathered]

    def decode() -> tuple[PlacementSnapshot, dict[str, RolePlacement]]:
        snapshot = parse_placement_snapshot(json.loads(agreed_text))
        roles = slice_pp_stage_roles(snapshot, ep_group.size(), pp_rank, all_stage_specs)
        return snapshot, roles

    snapshot, roles = collective_call(decode, ep_group, "PP placement loading")
    verify_placement_consensus(snapshot, ep_group)
    return snapshot, roles


def _synchronize(stream: Any) -> None:
    with gpu_sync_allowed():
        stream.synchronize()


def _commit_layer_mapping(model_state: Any, target: torch.Tensor, layer_idx: int) -> None:
    logical_map, replicas = upstream_state.compute_logical_maps(
        target.unsqueeze(0),
        model_state.model.num_logical_experts,
    )
    destination = model_state.logical_to_physical_map[layer_idx]
    padding = destination.shape[-1] - logical_map.shape[-1]
    if padding < 0:
        raise ValueError("Static EPLB placement exceeds the allocated replica capacity")
    logical_map = torch.nn.functional.pad(logical_map[0], (0, padding), value=-1)
    model_state.physical_to_logical_map[layer_idx].copy_(target)
    destination.copy_(logical_map)
    model_state.logical_replica_count[layer_idx].copy_(replicas[0])


def _move_from_buffer_for_expert_views(
    expert_weights: Sequence[Any],
    expert_weight_buffers: Sequence[Any],
    transfer_metadata: Any,
    new_indices: np.ndarray,
    ep_rank: int,
) -> None:
    """Install buffered weights for either tensor or Tensor List views."""
    upstream_state.move_from_buffer(
        expert_weights=expert_weights,
        expert_weights_buffers=expert_weight_buffers,
        transfer_metadata=transfer_metadata,
        new_indices=new_indices,
        ep_rank=ep_rank,
    )


def _roles_digest(roles: Mapping[str, RolePlacement]) -> str:
    payload = {
        role: {
            "num_logical_experts": entry.num_logical_experts,
            "num_redundant_experts": entry.num_redundant_experts,
            "physical_to_logical_map": [list(row) for row in entry.physical_to_logical_map],
        }
        for role, entry in sorted(roles.items())
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def install_placement(
    state: Any,
    snapshot: PlacementSnapshot,
    group: Any,
    stream: Any,
    *,
    model_states: Mapping[str, Any] | None = None,
    role_placements: Mapping[str, RolePlacement] | None = None,
) -> None:
    """Install a version-1 placement snapshot once, before profiling or graph capture."""
    if role_placements is not None and not role_placements:
        return

    def prepare() -> tuple[dict[str, tuple[Any, torch.Tensor, torch.Tensor, list[Any]]], dict[str, RolePlacement]]:
        if state.async_worker is not None:
            raise RuntimeError("Static EPLB installation must precede the background worker")
        if role_placements is None:
            if model_states is None:
                if len(state.model_states) != 1 or snapshot.draft is not None:
                    raise ValueError("Static EPLB with one model requires exactly one model state")
                resolved_states = {"target": next(iter(state.model_states.values()))}
            else:
                if "target" not in model_states:
                    raise ValueError("Static EPLB placement is missing its target model state")
                if snapshot.draft is not None and "draft" not in model_states:
                    raise ValueError("Static EPLB placement is missing its draft model state")
                resolved_states = dict(model_states)
            target_spec = build_runtime_role_spec(resolved_states["target"])
            draft_spec = build_runtime_role_spec(resolved_states["draft"]) if "draft" in resolved_states else None
            validate_snapshot_for_runtime(snapshot, group.size(), target_spec, draft_spec)
            requested_placements: dict[str, RolePlacement] = {"target": snapshot.target}
            if snapshot.draft is not None:
                requested_placements["draft"] = snapshot.draft
        else:
            if model_states is None:
                raise ValueError("Static EPLB stage installation requires explicit model_states")
            resolved_states = dict(model_states)
            requested_placements = dict(role_placements)
            for role, requested in requested_placements.items():
                if role not in resolved_states:
                    raise ValueError(f"Static EPLB placement is missing its {role} model state")
                _validate_role_against_spec(requested, role, build_runtime_role_spec(resolved_states[role]))

        prepared_roles: dict[str, tuple[Any, torch.Tensor, torch.Tensor, list[Any]]] = {}
        old_roles: dict[str, RolePlacement] = {}
        for role, requested in requested_placements.items():
            model_state = resolved_states[role]
            with gpu_sync_allowed():
                old = model_state.physical_to_logical_map.cpu().clone()
            target = torch.tensor(
                requested.physical_to_logical_map,
                dtype=old.dtype,
                device="cpu",
            )
            if target.shape != old.shape:
                raise ValueError(f"Static EPLB cannot change the number of allocated expert slots for {role}")
            plans = [
                plan_placement_sources(
                    tuple(before.tolist()),
                    tuple(after.tolist()),
                    group.size(),
                )
                for before, after in zip(old, target)
            ]
            prepared_roles[role] = (model_state, old, target, plans)
            old_roles[role] = RolePlacement(
                num_logical_experts=requested.num_logical_experts,
                num_redundant_experts=requested.num_redundant_experts,
                physical_to_logical_map=tuple(tuple(layer) for layer in old.tolist()),
            )
        return prepared_roles, old_roles

    prepared, old_roles = collective_call(prepare, group, "installation preflight")
    verify_placement_consensus(snapshot, group)
    old_digest = collective_call(lambda: _roles_digest(old_roles), group, "old layout validation")
    digests: list[Any] = [None] * group.size()
    dist.all_gather_object(digests, old_digest, group=group)
    if len(set(digests)) != 1:
        raise ValueError("EPLB placement differs across EP ranks")

    changed_layers = {
        role: torch.nonzero((old != target).any(dim=1)).flatten().tolist()
        for role, (_, old, target, _) in prepared.items()
    }
    if not any(changed_layers.values()):
        return
    state.prepare_movement_resources()
    try:
        _synchronize(stream)
        for role, (model_state, old, target, plans) in prepared.items():
            if not changed_layers[role]:
                continue
            communicator = model_state.communicator
            communicator.set_stream(stream)
            for layer_idx in changed_layers[role]:
                source_ranks, source_slots = plans[layer_idx]
                metadata = stage_explicit_layer_transfer(
                    old[layer_idx],
                    target[layer_idx],
                    np.asarray(source_ranks),
                    np.asarray(source_slots),
                    model_state.model.expert_weights[layer_idx],
                    model_state.expert_buffer,
                    state.placement_device_group(),
                    communicator,
                    stream,
                    layer_idx,
                )
                _synchronize(stream)
                dist.barrier(group=group)
                _move_from_buffer_for_expert_views(
                    expert_weights=model_state.model.expert_weights[layer_idx],
                    expert_weight_buffers=model_state.expert_buffer,
                    transfer_metadata=metadata,
                    new_indices=target[layer_idx].numpy(),
                    ep_rank=group.rank(),
                )
                _commit_layer_mapping(model_state, target[layer_idx], layer_idx)
                state.refresh_placement_layer(model_state, layer_idx)
                _synchronize(stream)
                dist.barrier(group=group)
    finally:
        state.release_movement_resources()
