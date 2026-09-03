from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lerobot_async_inference.configs import PolicyServerConfig, RobotClientConfig, cosine_ramp
from lerobot_async_inference.helpers import TimedObservation
from lerobot_async_inference.near_grasp_replay_analysis import execute_sequential_rtc_condition
from lerobot_async_inference.policy_server import PolicyServer
from lerobot_async_inference.rtc import (
    RTCDiagnosticsWriter,
    RTCRequest,
    RTCState,
    RollingLatencyEstimator,
    latency_to_delay_frames,
    overlap_metrics,
    slice_previous_policy_chunk,
)


class RecordingPolicy:
    def __init__(self, output: torch.Tensor):
        self.config = SimpleNamespace(image_features={})
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
    assert request.bypass_reason == "no_previous_chunk"
    assert request.delay_estimator_ready is False


def test_previous_leftover_calls_guided_path() -> None:
    policy = RecordingPolicy(torch.zeros(1, 4, 3))
    server = _bare_server(policy, rtc_enabled=True)
    prefix = torch.ones(2, 3)
    request = RTCRequest(
        prefix=prefix,
        previous_robot_leftover=prefix,
        shift=2,
        inference_delay_frames=3,
        delay_frames_float=2.25,
        applied=True,
        bypass_reason=None,
    )
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


def test_empty_delay_estimator_is_not_ready() -> None:
    estimator = RollingLatencyEstimator()
    assert estimator.ready is False
    assert estimator.window_count == 0
    assert estimator.window_max_s is None
    assert estimator.estimate_delay_frames(15) is None


def test_single_210ms_sample_at_15fps_uses_four_frames() -> None:
    estimator = RollingLatencyEstimator()
    estimator.add(0.210)
    estimate = estimator.estimate_delay_frames(15)
    assert estimate is not None
    assert estimate[0] == 4
    assert math.isclose(estimate[1], 3.15)


def test_rolling_max_of_180_210_260ms_uses_four_frames() -> None:
    estimator = RollingLatencyEstimator()
    for latency_s in (0.180, 0.210, 0.260):
        estimator.add(latency_s)
    assert estimator.window_max_s == 0.260
    estimate = estimator.estimate_delay_frames(15)
    assert estimate is not None
    assert estimate[0] == 4
    assert math.isclose(estimate[1], 3.9)


def test_320ms_spike_uses_five_frames() -> None:
    estimator = RollingLatencyEstimator()
    for latency_s in (0.180, 0.210, 0.260, 0.320):
        estimator.add(latency_s)
    assert estimator.window_max_s == 0.320
    estimate = estimator.estimate_delay_frames(15)
    assert estimate is not None
    assert estimate[0] == 5
    assert math.isclose(estimate[1], 4.8)


def test_delay_decreases_after_spike_leaves_rolling_window() -> None:
    estimator = RollingLatencyEstimator()
    estimator.add(0.320)
    spike_estimate = estimator.estimate_delay_frames(15)
    assert spike_estimate is not None and spike_estimate[0] == 5
    steady_latencies_s = (
        0.180,
        0.210,
        0.260,
        0.200,
        0.190,
        0.230,
        0.240,
        0.220,
        0.200,
        0.180,
    )
    for latency_s in steady_latencies_s:
        estimator.add(latency_s)
    assert estimator.window_count == 10
    assert estimator.window_max_s == 0.260
    estimate = estimator.estimate_delay_frames(15)
    assert estimate is not None
    assert estimate[0] == 4
    assert math.isclose(estimate[1], 3.9)


def _complete_state(state: RTCState, *, timestep: int, latency_s: float) -> None:
    chunk = torch.ones(10, 3)
    state.complete(
        raw_chunk=chunk,
        robot_chunk=chunk,
        timestep=timestep,
        latency_s=latency_s,
    )


def test_first_cold_start_latency_is_excluded_from_estimator() -> None:
    state = RTCState(fps=15)
    first = state.prepare(current_timestep=0)
    assert first.bypass_reason == "no_previous_chunk"
    _complete_state(state, timestep=0, latency_s=0.495)
    assert state.delay_estimator.ready is False
    assert state.delay_estimator.window_count == 0
    assert state.delay_estimator.window_max_s is None
    _complete_state(state, timestep=1, latency_s=0.210)
    assert state.delay_estimator.window_count == 1
    assert state.delay_estimator.window_max_s == 0.210


def test_second_request_bypasses_for_delay_estimator_warmup() -> None:
    state = RTCState(fps=15)
    _complete_state(state, timestep=0, latency_s=0.495)
    second = state.prepare(current_timestep=1)
    assert second.applied is False
    assert second.bypass_reason == "delay_estimator_warmup"
    assert second.inference_delay_frames == 0
    assert second.delay_estimator_window_count == 0


def test_third_request_applies_rtc_using_previous_steady_state_latency() -> None:
    state = RTCState(fps=15)
    _complete_state(state, timestep=0, latency_s=0.495)
    second = state.prepare(current_timestep=1)
    assert second.bypass_reason == "delay_estimator_warmup"
    _complete_state(state, timestep=1, latency_s=0.210)

    third = state.prepare(current_timestep=2)
    assert third.applied is True
    assert third.bypass_reason is None
    assert third.inference_delay_frames == 4
    assert third.delay_estimator_ready is True
    assert third.delay_estimator_window_count == 1
    assert third.delay_estimator_window_max_ms == 210.0

    _complete_state(state, timestep=2, latency_s=0.320)
    assert third.inference_delay_frames == 4
    fourth = state.prepare(current_timestep=3)
    assert fourth.inference_delay_frames == 5
    assert fourth.delay_estimator_window_count == 2
    assert fourth.delay_estimator_window_max_ms == 320.0


def test_short_prefix_uses_local_rtc_dynamic_horizon_behavior() -> None:
    state = RTCState(fps=15)
    state.complete(
        raw_chunk=torch.ones(2, 3),
        robot_chunk=torch.ones(2, 3),
        timestep=4,
        latency_s=0.5,
    )
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
    request = RTCRequest(
        prefix=torch.zeros(2, 2),
        previous_robot_leftover=None,
        shift=0,
        inference_delay_frames=0,
        delay_frames_float=0.0,
        applied=True,
        bypass_reason=None,
    )
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


def test_live_server_sequence_and_delay_estimator_diagnostics(tmp_path) -> None:
    server = PolicyServer(PolicyServerConfig(fps=15))
    policy = RecordingPolicy(torch.zeros(1, 10, 3))
    server.policy = policy
    server.preprocessor = lambda observation: observation
    server.postprocessor = lambda action: action
    server.lerobot_features = {}
    server.actions_per_chunk = 10
    server.rtc_enabled = True
    server._rtc_state = RTCState(fps=15)
    server._rtc_diagnostics = RTCDiagnosticsWriter(tmp_path)
    diagnostics_path = server._rtc_diagnostics.path

    with patch(
        "lerobot_async_inference.policy_server.raw_observation_to_observation",
        side_effect=lambda observation, features, images: observation,
    ):
        for timestep in range(3):
            server._predict_action_chunk(
                TimedObservation(
                    timestamp=float(timestep),
                    timestep=timestep,
                    observation={"state": torch.zeros(1)},
                    must_go=True,
                )
            )
    server.stop()

    assert policy.calls[0] == {}
    assert policy.calls[1] == {}
    assert policy.calls[2]["inference_delay"] >= 1
    assert policy.calls[2]["prev_chunk_left_over"].shape == (9, 3)

    records = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]
    assert len(records) == 3
    assert records[0]["rtc_bypass_reason"] == "no_previous_chunk"
    assert records[1]["rtc_bypass_reason"] == "delay_estimator_warmup"
    assert records[2]["rtc_applied"] is True
    assert records[2]["delay_estimator_ready"] is True
    assert records[2]["delay_estimator_window_count"] == 1
    assert records[2]["delay_estimator_window_max_ms"] == records[1]["inference_latency_ms"]
    for record in records:
        assert "inference_latency_ms" in record
        assert "inference_delay_frames" in record


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
        {"capture_id": 3, "policy_timestamp": 10.4},
    ]
    results = execute_sequential_rtc_condition(
        policy=policy,
        selected=selected,
        batches={
            1: {"offset": torch.tensor([0.0])},
            2: {"offset": torch.tensor([1.0])},
            3: {"offset": torch.tensor([2.0])},
        },
        postprocessor=lambda action: action,
        device=torch.device("cpu"),
        num_runs=3,
        fixed_noise=torch.zeros(1, 6, 3),
        fps=15,
        execution_horizon=10,
    )
    assert results[3].robot_actions.shape == (3, 6, 3)
    assert policy.calls[0] == {}
    assert policy.calls[1] == {}
    assert policy.calls[2]["prev_chunk_left_over"].shape == (3, 3)


def _run_diagnostics_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        test_rtc_diagnostic_jsonl_and_overlap_metrics(Path(directory))


def _run_live_server_sequence_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        test_live_server_sequence_and_delay_estimator_diagnostics(Path(directory))


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
        test_empty_delay_estimator_is_not_ready,
        test_single_210ms_sample_at_15fps_uses_four_frames,
        test_rolling_max_of_180_210_260ms_uses_four_frames,
        test_320ms_spike_uses_five_frames,
        test_delay_decreases_after_spike_leaves_rolling_window,
        test_first_cold_start_latency_is_excluded_from_estimator,
        test_second_request_bypasses_for_delay_estimator_warmup,
        test_third_request_applies_rtc_using_previous_steady_state_latency,
        test_short_prefix_uses_local_rtc_dynamic_horizon_behavior,
        test_cosine_ramp_downstream_input_is_unchanged,
        test_rtc_flags_default_disabled_and_do_not_change_refill_config,
        _run_diagnostics_test,
        _run_live_server_sequence_test,
        test_offline_rtc_replay_threads_consecutive_captures,
    ):
        suite.addTest(unittest.FunctionTestCase(function))
    return suite
