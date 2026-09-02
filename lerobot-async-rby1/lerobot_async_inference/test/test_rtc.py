from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from lerobot_async_inference.configs import RobotClientConfig, cosine_ramp
from lerobot_async_inference.near_grasp_replay_analysis import execute_sequential_rtc_condition
from lerobot_async_inference.policy_server import PolicyServer
from lerobot_async_inference.rtc import (
    RTCDiagnosticsWriter,
    RTCRequest,
    RTCState,
    latency_to_delay_frames,
    overlap_metrics,
    slice_previous_policy_chunk,
)


class RecordingPolicy:
    def __init__(self, output: torch.Tensor):
        self.output = output
        self.calls: list[dict] = []

    def predict_action_chunk(self, observation, **kwargs):
        self.calls.append(kwargs)
        return self.output.clone()


def _bare_server(policy: RecordingPolicy, *, rtc_enabled: bool) -> PolicyServer:
    server = object.__new__(PolicyServer)
    server.policy = policy
    server.actions_per_chunk = policy.output.shape[1]
    server.rtc_execution_horizon = 10
    server.rtc_enabled = rtc_enabled
    return server


def test_rtc_disabled_preserves_exact_predict_call() -> None:
    policy = RecordingPolicy(torch.zeros(1, 4, 3))
    server = _bare_server(policy, rtc_enabled=False)
    output = server._get_action_chunk({"state": torch.zeros(1)})
    assert output.shape == (1, 4, 3)
    assert policy.calls == [{}]


def test_rtc_config_construction() -> None:
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=10.0,
        execution_horizon=10,
    )
    assert config.enabled is True
    assert config.prefix_attention_schedule is RTCAttentionSchedule.EXP


def test_local_guided_rtc_processor_runs_with_synthetic_prefix() -> None:
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

    processor = RTCProcessor(
        RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.EXP, execution_horizon=10)
    )
    output = processor.denoise_step(
        x_t=torch.zeros(1, 6, 3),
        prev_chunk_left_over=torch.ones(3, 3),
        inference_delay=2,
        time=0.5,
        original_denoise_step_partial=lambda value: value * 0.25,
    )
    assert output.shape == (1, 6, 3)
    assert bool(torch.isfinite(output).all())


def test_no_previous_chunk_bypasses_rtc() -> None:
    request = RTCState(fps=15).prepare(current_timestep=0)
    assert request.applied is False
    assert request.bypass_reason == "no previous chunk"


def test_previous_leftover_calls_guided_path() -> None:
    policy = RecordingPolicy(torch.zeros(1, 4, 3))
    server = _bare_server(policy, rtc_enabled=True)
    prefix = torch.ones(2, 3)
    request = RTCRequest(prefix, prefix, 2, 3, 2.25, True, None)
    server._get_action_chunk({}, rtc_request=request)
    assert policy.calls[0]["prev_chunk_left_over"] is prefix
    assert policy.calls[0]["inference_delay"] == 3
    assert policy.calls[0]["execution_horizon"] == 10


def test_previous_policy_chunk_slicing_uses_unexecuted_tail() -> None:
    chunk = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    leftover, shift = slice_previous_policy_chunk(chunk, previous_timestep=10, current_timestep=14)
    assert shift == 4
    assert torch.equal(leftover, chunk[4:])


def test_inference_delay_uses_ceil_frame_semantics() -> None:
    frames, frames_float = latency_to_delay_frames(0.150, 15)
    assert frames == 3
    assert abs(frames_float - 2.25) < 1e-12


def test_short_prefix_uses_local_rtc_dynamic_horizon_behavior() -> None:
    state = RTCState(fps=15)
    state.complete(
        raw_chunk=torch.ones(2, 3),
        robot_chunk=torch.ones(2, 3),
        timestep=5,
        latency_s=0.01,
    )
    request = state.prepare(current_timestep=5)
    assert request.applied is True
    assert request.prefix.shape[0] == 2  # RTCProcessor clamps execution_horizon to this length.


def test_cosine_ramp_downstream_input_is_unchanged() -> None:
    old = torch.tensor([1.0, 2.0])
    new = torch.tensor([3.0, 4.0])
    before = cosine_ramp(old, new, overlap_index=1, overlap_count=3)
    request = RTCRequest(torch.zeros(2, 2), None, 0, 0, 0.0, True, None)
    after = cosine_ramp(old, new, overlap_index=1, overlap_count=3)
    assert request.applied and torch.equal(before, after)


def test_rtc_flags_default_disabled_and_do_not_change_refill_config() -> None:
    config = RobotClientConfig(robot=object(), actions_per_chunk=50)
    assert config.rtc_enabled is False
    assert config.chunk_size_threshold == 0.5
    assert config.actions_per_chunk == 50


def test_rtc_diagnostic_jsonl_and_overlap_metrics(tmp_path) -> None:
    writer = RTCDiagnosticsWriter(tmp_path)
    metrics = overlap_metrics(torch.zeros(4, 2), torch.zeros(4, 2))
    writer.write({"request_id": 7, "rtc_applied": True, **metrics})
    path = writer.path
    writer.close()
    record = json.loads(path.read_text())
    assert record["request_id"] == 7
    assert record["aligned_overlap_length"] == 4
    assert record["post_rtc_overlap_rmse"] == 0.0


class SequentialPolicy:
    def __init__(self):
        self.config = SimpleNamespace()
        self.calls: list[dict] = []

    def eval(self):
        return self

    def reset(self):
        pass

    def predict_action_chunk(self, batch, noise=None, **kwargs):
        self.calls.append(kwargs)
        return noise[:, :6, :3] + batch["offset"]


def test_offline_rtc_replay_threads_consecutive_captures() -> None:
    policy = SequentialPolicy()
    selected = [
        {"capture_id": 1, "policy_timestamp": 10.0},
        {"capture_id": 2, "policy_timestamp": 10.2},
    ]
    results = execute_sequential_rtc_condition(
        policy=policy,
        selected=selected,
        batches={1: {"offset": torch.tensor([0.0])}, 2: {"offset": torch.tensor([1.0])}},
        postprocessor=lambda action: action,
        device=torch.device("cpu"),
        num_runs=3,
        fixed_noise=torch.zeros(1, 6, 3),
        fps=15,
        execution_horizon=10,
    )
    assert results[2].robot_actions.shape == (3, 6, 3)
    assert policy.calls[0] == {}
    assert policy.calls[1]["prev_chunk_left_over"].shape == (3, 3)


def _run_diagnostics_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        test_rtc_diagnostic_jsonl_and_overlap_metrics(Path(directory))


def load_tests(loader, tests, pattern):  # noqa: ARG001
    suite = unittest.TestSuite()
    for function in (
        test_rtc_disabled_preserves_exact_predict_call,
        test_rtc_config_construction,
        test_local_guided_rtc_processor_runs_with_synthetic_prefix,
        test_no_previous_chunk_bypasses_rtc,
        test_previous_leftover_calls_guided_path,
        test_previous_policy_chunk_slicing_uses_unexecuted_tail,
        test_inference_delay_uses_ceil_frame_semantics,
        test_short_prefix_uses_local_rtc_dynamic_horizon_behavior,
        test_cosine_ramp_downstream_input_is_unchanged,
        test_rtc_flags_default_disabled_and_do_not_change_refill_config,
        _run_diagnostics_test,
        test_offline_rtc_replay_threads_consecutive_captures,
    ):
        suite.addTest(unittest.FunctionTestCase(function))
    return suite
