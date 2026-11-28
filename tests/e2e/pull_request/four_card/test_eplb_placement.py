# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.four_card.test_qwen3_mrv2_eplb import MODEL, PROMPTS, _assert_expected_answers
from vllm_ascend.distributed.eplb.placement import load_placement

CALIBRATION_MAX_ROUNDS = 4


def _exported_files(path: str) -> list[Path]:
    """Exports are tagged with the P/D role and DP rank, so match by prefix."""
    target = Path(path)
    return sorted(target.parent.glob(f"{target.stem}.*{target.suffix}"))


def _snapshot_worker(worker: Any) -> dict[str, Any]:
    state = worker.model_runner.eplb.state
    model_state = next(iter(state.model_states.values()))
    samples = []
    # Include compute weights and quantization metadata, in physical slot order.
    for layer_idx in (0, model_state.model.num_moe_layers - 1):
        for view in model_state.model.expert_weights[layer_idx]:
            samples.append([tensor.reshape(-1)[:8].cpu().tolist() for tensor in view])
    return {
        "placement": model_state.physical_to_logical_map.cpu().tolist(),
        "samples": samples,
        "static": state.static_mode,
        "no_worker": state.async_worker is None,
        "recording": [layer.eplb_state.load_recording_enabled for layer in model_state.model.moe_layers],
    }


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="eplb",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_and_piecewise",
)
@patch.dict(
    "os.environ",
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        # collective_rpc transports the `_snapshot_worker` callable across the
        # engine boundary; this fallback is needed only for this test.
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.7, max_wait_seconds=180)
@pytest.mark.parametrize("communicator", ["torch_gloo", None], ids=["gloo", "auto"])
def test_dynamic_export_static_restart(tmp_path, capfd, communicator):
    path = str(tmp_path / "deployment.json")
    kwargs: dict[str, Any] = {
        "tensor_parallel_size": 4,
        "enable_expert_parallel": True,
        "enable_eplb": True,
        "max_model_len": 2048,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 2048,
        "quantization": "ascend",
        "block_size": 128,
        "gpu_memory_utilization": 0.7,
        "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
        "eplb_config": {"window_size": 2, "step_interval": 2, "num_redundant_experts": 4},
    }
    if communicator is not None:
        kwargs["eplb_config"]["communicator"] = communicator
    kwargs["additional_config"] = {
        "eplb_config": {"expert_placement_path": path, "export_after_rearranges": 1},
    }
    exported: list[Path] = []
    with VllmRunner(MODEL, **kwargs) as runner:
        initial = runner.model.collective_rpc(_snapshot_worker)
        for _ in range(CALIBRATION_MAX_ROUNDS):
            outputs = runner.generate_greedy(PROMPTS, max_tokens=32)
            _assert_expected_answers(outputs, "dynamic calibration")
            exported = _exported_files(path)
            if exported:
                break
        before = runner.model.collective_rpc(_snapshot_worker)
        assert before[0]["placement"] != initial[0]["placement"], "Calibration did not produce any changed layers"
        assert len(exported) == 1, f"Expected one exported deployment, got {exported}"
    expected = [list(layer) for layer in load_placement(exported[0]).physical_to_logical]
    capfd.readouterr()
    kwargs["additional_config"] = {
        "eplb_config": {"mode": "static", "expert_placement_path": str(exported[0])},
    }
    with VllmRunner(MODEL, **kwargs) as runner:
        after = runner.model.collective_rpc(_snapshot_worker)
        assert len(before) == len(after) == 4
        for dynamic, static in zip(before, after):
            assert static["placement"] == dynamic["placement"] == expected
            assert static["samples"] == dynamic["samples"]
            assert static["static"] and static["no_worker"]
            assert not any(static["recording"])
        outputs = runner.generate_greedy(PROMPTS, max_tokens=32)
        _assert_expected_answers(outputs, "static deployment")
        final = runner.model.collective_rpc(_snapshot_worker)
        assert final == after
    captured = capfd.readouterr()
    assert "Ascend async EPLB cycle committed" not in captured.out + captured.err
