# Expert Parallelism Load Balancer (EPLB)

## Feature Introduction

EPLB moves redundant expert replicas between NPUs so that an expert-parallel MoE
model keeps its per-rank token load balanced while serving. Rearrangement is
asynchronous and does not pause in-flight requests. Ascend runs EPLB on Model
Runner v2 with the STAIR policy by default, and keeps upstream `--eplb-config`
fields unchanged.

This page covers the two Ascend extensions: exporting the committed placement to
a file, and importing that file at startup instead of rebalancing online.

| Capability | Fields under `additional_config.eplb_config` |
| --- | --- |
| Export the placement file | `expert_placement_path` + `export_after_rearranges` (`mode` stays `dynamic`) |
| Import the placement file | `mode` = `static` + `expert_placement_path` |

### Constraints and Limitations

| Category | Description |
| --- | --- |
| Hardware | A2 requires `num_redundant_experts=0`; redundant experts need A3 or 950PR&950DT. |
| Deployment | PD disaggregation and PD co-location are both supported; export and import require PP=1. |
| Model | MoE models with expert parallelism; see [Weight formats](#weight-formats). |
| Import | Same model and revision, quantization, dtype, EP/TP/DP configuration and redundant-expert count. |
| Other | `/dev/shm` must be mounted; `hixl` movement needs the CANN HIXL runtime, otherwise it falls back to Gloo. |

### Weight formats

| Format | EPLB | Notes |
| --- | --- | --- |
| BF16 / FP16 | Yes | Unquantized expert weights and biases. |
| W8A8 / W8A8 Dynamic | Yes | Per-expert weight and scale tensors. |
| W4A8 | Yes | Per-expert weight, scale and scale-bias tensors. |
| W4A4 MXFP / W8A8 MXFP / W4A8 MXFP | Yes | 950PR&950DT, native ND expert tensors. |

## Feature Usage

### Step 1: Export the placement file

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{"num_redundant_experts":16}' \
  --additional-config '{"eplb_config":{"expert_placement_path":"/data/eplb/deployment.json","export_after_rearranges":5}}'
```

Serve real traffic. After the configured number of completed rearrangements the
worker exports once, in the first forward gap, and keeps serving. Skipped and
profile rearrangements do not count. The file stores placements and validation
metadata, not weights, and its name gains a `.<kv_role>.dp<rank>` suffix
(`deployment.standalone.dp0.json` without PD disaggregation). All ranks must be
given the same path.

### Step 2: Import the placement file

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{"num_redundant_experts":16}' \
  --additional-config '{"eplb_config":{"mode":"static","expert_placement_path":"/data/eplb/deployment.standalone.dp0.json"}}'
```

The placement is installed once, before profiling and graph capture, and no
runtime load is collected or moved.

- `--enable-eplb` is still required, `load_collection_phase` must stay `all`, and
  `export_after_rearranges` is rejected.
- A file is only installed on the role it was calibrated for (`prefill`,
  `decode` or `mixed`), so tune P and D nodes separately; PD co-location can
  share one file.
- Only EP rank 0 has to read the file; the other ranks receive it by broadcast.

### MTP draft

A registered MoE draft inherits `--enable-eplb` from the target: in `dynamic` mode
it is rebalanced together with the target, in `static` mode both placements come
from the same file.

### Verifying the Feature

```bash
# export
grep -E "Ascend EPLB deployment export" vllm.log
# import
grep "Ascend static EPLB deployment installed" vllm.log
```

Expected:

```text
Ascend EPLB deployment export armed after 5 rearrange cycles
Ascend EPLB deployment export running in a forward gap: /data/eplb/deployment.standalone.dp0.json
Ascend EPLB deployment exported: /data/eplb/deployment.standalone.dp0.json
Ascend static EPLB deployment installed: scope=mixed sha256=... layers=...
```

`grep -E "EPLB step:.*balancedness=" vllm.log` shows the load window the export
was taken from; `1` means perfectly balanced.

## Configuration Parameters

Upstream `--eplb-config` fields (`window_size`, `step_interval`,
`num_redundant_experts`, `policy`, `communicator`, ...) are passed through
unchanged. The Ascend fields below live in `additional_config.eplb_config` and
all require `--enable-eplb`:

| Parameter | Default | Description |
| --- | --- | --- |
| `mode` | `dynamic` | `dynamic` rebalances online, `static` installs a saved placement once. |
| `expert_placement_path` | unset | Export target in `dynamic` mode, input file in `static` mode. |
| `export_after_rearranges` | unset | `dynamic` only, requires `expert_placement_path`. |
| `load_collection_phase` | `all` | `all`, `prefill` or `decode`; STAIR only. |
| `stair_config` | - | Advanced STAIR knobs, defaults fit most workloads. |

Unknown fields are warned about and ignored.

## Tuning Suggestions

The STAIR defaults fit most workloads. Change these fields only after a
measurement shows a need. They live under `additional_config.eplb_config.stair_config`
and are range checked at startup.

| Parameter | Default | Range | Description |
| --- | --- | --- | --- |
| `load_window_bins` | `64` | 2-256 | Chronological bins used to compress the load window. |
| `load_risk_quantile` | `0.75` | (0.5, 1.0) | One-sided normal quantile for the expert and rank risk score. |
| `relative_balance_threshold` | `0.95` | (0, 1] | Rebalance when the current balance is at or below this fraction of the last committed one. |
| `absolute_balance_threshold` | `0.90` | (0, 1] | Rebalance when the current balance is at or below this value. |
| `rank_transfer_limit` | `-1` | -1 or positive | Transfers per rank in one layer plan; `-1` removes the limit. |
| `cross_node_transfer_limit` | `-1` | -1 or >= 0 | Cross-node transfers per node in one layer plan; `0` disables them. |
| `replica_search_num_stages` | `4` | 1-8 | Risk-ordered expert groups handled by the replica search. |
| `replica_search_radius` | `8` | 0-32 | Search radius around the greedy extra-replica budget. |
| `replica_search_beam_size` | `64` | 1-128 | Replica-count candidates kept after each stage. |
| `placement_search_backtrack_limit` | `32` | 0-64 | Feasible-branch reversals while one candidate is placed; `0` disables them. |
