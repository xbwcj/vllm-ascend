# EPLB Deployment Import and Static Initialization

Ascend EPLB can reuse a placement snapshot learned by an earlier calibration
run. The snapshot JSON contains committed physical-to-logical expert mappings,
not uncommitted planner targets and not expert weights. This document describes
the static consumer side (`eplb_config.mode="static"`).

## Snapshot representation

`vllm_ascend/distributed/eplb/placement_runtime.py` parses and validates the
`version: 1` placement snapshot schema into `PlacementSnapshot` and
`RolePlacement`:

- `version`: integer `1`.
- `ep_size`: expert-parallel world size (`>= 1`).
- `target`: required object containing `num_logical_experts`,
  `num_redundant_experts`, and `physical_to_logical_map` (one flattened global
  physical-slot list per target MoE layer).
- `draft`: optional object with the same fields (`num_logical_experts`,
  `num_redundant_experts`, `physical_to_logical_map`) for a registered
  speculative MoE draft that enables EPLB.

For EP size `N` and `S` slots per rank, physical slot `r * S + s` belongs to
rank `r`, local slot `s`. The value at this position identifies its logical
expert. For example, `[0, 1, 2, 0, 1, 3]` with `ep_size=2` places logical
experts `0` and `1` on both ranks, and one replica each of experts `2` and `3`.

Validation requires equal rank capacity (`num_physical_experts % ep_size == 0`),
complete logical-expert coverage per layer, valid integer logical expert IDs,
no duplicate logical experts within the same rank on any layer, and exact
matching against runtime `ep_size`, `num_logical_experts`,
`num_redundant_experts`, and MoE layer count. Derived logical maps and
replica-routing tables are rebuilt at installation time instead of serialized.

## Import transaction

`mode="static"` keeps `--enable-eplb` enabled because it controls the router
and expert slot allocation. The controller installs the snapshot instead of
starting the dynamic EPLB worker. Installation happens before memory profiling
and graph capture:

1. Load the checkpoint in the initial physical layout and expose the
   quantization method's actual compute-weight views.
2. Read `expert_placement_path` on global rank 0, broadcast it across PP and EP
   groups, and validate the snapshot schema and whole-model runtime dimensions
   (slicing `target` and `draft` layer maps per PP stage when `PP > 1`) before
   backend registration or weight movement.
3. Snapshot and verify consensus on the initial layout, then build an exact
   source coordinate for each target slot via `plan_placement_sources`
   (preferring unchanged slots, then rank-local copies).
4. Initialize the existing HIXL session if needed, without starting a
   background thread.
5. Stage one changed layer into the existing expert buffer through
   `stage_explicit_layer_transfer`.
6. Synchronize the stream and reach an EP CPU barrier before overwriting live
   sources.
7. Install buffered weights using upstream `move_from_buffer`, rebuild and copy
   the logical maps and replica counts, and refresh routing tables in place.
8. Synchronize before reusing buffers for the next layer, then release
   temporary HIXL registrations after all changed layers are installed.

This sequence supports within-rank swaps because old sources are staged into
buffers before any live destination slot is overwritten.

## Static serving

During serving:

- controller `prepare_forward` and `step` return without dynamic collection or
  rearrangement;
- state `step`, profile rearrangement, and async worker startup are guarded;
- ordinary forwards keep each layer's `load_recording_enabled=False`, omitting
  token recording and AllGather eligibility/counting from graph capture;
- logical-to-physical expert-replica routing remains active.

`eplb_config.log_balancedness=true` opts into a lightweight rank-local load
sample at `step_interval` so data-parallel prefill engines never enter a device
collective from the hot path. It never triggers planning or migration.

## Support boundary

Static import requires `--enable-eplb` and `eplb_config.expert_placement_path`
pointing to a `.json` file readable on global rank 0. Pipeline parallelism
(`pipeline_parallel_size >= 1`) and registered MoE speculative drafts with EPLB
enabled are supported. `load_collection_phase` is permitted in static mode and
filters rank-local observation steps when `eplb_config.log_balancedness=true`.

Existing Ascend EPLB quantization and hardware constraints, including A2
redundancy restrictions and communicator limits, still apply. Static
installation uses the same communicator and weight movement implementation as
dynamic EPLB, without runtime fallback after a partially failed migration.

## Validation

CPU unit tests cover `version: 1` snapshot parsing, invalid schema and runtime
mismatch rejection, source planning, collective error propagation and
consensus, local swaps, remote-source installation, Tensor List views,
routing-table identity, and static forward/step guards.

Before merge, exercise a snapshot with changed layers on real Ascend hardware
following the repository's
[PR testing requirements](../contribution/testing.md#pr-testing-requirements).
