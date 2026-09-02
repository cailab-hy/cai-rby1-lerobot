from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lerobot_async_inference.diagnostic_capture import PolicyBatchCaptureWriter
from lerobot_async_inference.configs import PolicyServerConfig
from lerobot_async_inference.frozen_noise_analysis import execute_condition
from lerobot_async_inference.helpers import TimedObservation
from lerobot_async_inference.near_grasp_replay_analysis import (
    align_shifted_chunks,
    disagreement_metrics,
    estimate_frame_shift,
    roughness_metrics,
)
from lerobot_async_inference.policy_server import PolicyServer


LEFT = list(range(7))


def test_capture_is_disabled_by_default() -> None:
    config = PolicyServerConfig()
    server = PolicyServer(config)
    assert config.diagnostic_capture_policy_batches is False
    assert server._diagnostic_capture is None


class DeterministicNoisePolicy:
    def __init__(self) -> None:
        self.training = True

    def eval(self):
        self.training = False
        return self

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch, noise=None):
        assert noise is not None
        return noise[:, :6, :7] + batch["observation.state"][:, None, :7]


def test_fixed_noise_deterministic_replay() -> None:
    result = execute_condition(
        policy=DeterministicNoisePolicy(),
        frozen_batch={"observation.state": torch.zeros(1, 7)},
        postprocessor=lambda action: action,
        device=torch.device("cpu"),
        num_runs=3,
        fixed_noise=torch.randn(1, 6, 9),
    )
    assert torch.equal(result.robot_actions[0], result.robot_actions[1])
    assert torch.equal(result.robot_actions[0], result.robot_actions[2])


def test_smooth_chunk_has_low_roughness() -> None:
    time = torch.arange(50, dtype=torch.float32)[:, None]
    chunk = time.repeat(1, 7) * 0.001
    metrics = roughness_metrics(chunk, LEFT)
    assert metrics["d2q_max"] < 1e-6
    assert metrics["oscillation"] != metrics["oscillation"]  # no eligible reversals => NaN


def test_alternating_chunk_has_high_oscillation() -> None:
    values = torch.tensor([0.0, 0.01] * 25)[:, None].repeat(1, 7)
    metrics = roughness_metrics(values, LEFT)
    assert metrics["oscillation"] == 1.0
    assert metrics["d2q_p95"] >= 0.019


def test_known_chunk_shift_aligns_overlap() -> None:
    trajectory = torch.arange(70, dtype=torch.float32)[:, None].repeat(1, 7)
    old, new = trajectory[:50], trajectory[20:70]
    aligned_old, aligned_new = align_shifted_chunks(old, new, 20)
    assert aligned_old.shape == (30, 7)
    assert torch.equal(aligned_old, aligned_new)


def test_identical_aligned_chunks_have_zero_disagreement() -> None:
    chunk = torch.randn(20, 7)
    metrics = disagreement_metrics(chunk, chunk.clone(), LEFT)
    assert metrics["rmse"] == 0.0
    assert metrics["max_abs_diff"] == 0.0
    assert metrics["direction_mismatch"] == 0.0


def test_opposite_direction_chunks_have_high_direction_disagreement() -> None:
    time = torch.arange(20, dtype=torch.float32)[:, None]
    old = time.repeat(1, 7) * 0.01
    new = -old
    metrics = disagreement_metrics(old, new, LEFT)
    assert metrics["direction_mismatch"] == 1.0
    assert metrics["delta_cosine_similarity"] < -0.999


def test_metadata_time_difference_maps_to_frame_shift() -> None:
    shift, delta_s, source, confidence = estimate_frame_shift(
        {"policy_timestamp": 100.0}, {"policy_timestamp": 101.67}, fps=15
    )
    assert shift == 25
    assert abs(delta_s - 1.67) < 1e-9
    assert source == "policy_timestamp"
    assert confidence in {"high", "medium"}


def test_background_capture_writes_bounded_cpu_artifacts(tmp_path) -> None:
    writer = PolicyBatchCaptureWriter(tmp_path, max_captures=1, logger=logging.getLogger(__name__))
    source = {"observation.state": torch.tensor([[1.0, 2.0]]), "task": ["test"]}
    prepared = writer.prepare(source, {"policy_timestamp": 1.0})
    assert prepared is not None
    source["observation.state"].add_(10)
    writer.submit(
        prepared,
        raw_chunk=torch.zeros(4, 3),
        robot_chunk=torch.ones(4, 3),
        metadata={"inference_latency_ms": 2.0},
    )
    assert writer.prepare(source, {}) is None
    writer.close()

    saved = torch.load(tmp_path / "capture_000.pt", weights_only=False)
    assert saved["observation.state"].device.type == "cpu"
    assert torch.equal(saved["observation.state"], torch.tensor([[1.0, 2.0]]))
    assert (tmp_path / "raw_chunk_000.pt").is_file()
    assert (tmp_path / "robot_chunk_000.pt").is_file()
    metadata = json.loads((tmp_path / "capture_metadata.jsonl").read_text().strip())
    assert metadata["capture_id"] == 0
    assert metadata["raw_chunk_shape"] == [4, 3]
    assert metadata["batch_cpu_clone_ms"] >= 0
    assert metadata["background_save_and_checksum_ms"] >= 0


class MutatingDeploymentPolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(image_features={}, num_steps=10)

    def predict_action_chunk(self, batch):
        batch["observation.state"].add_(100)
        return torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)


def test_server_capture_is_immediately_before_policy_and_before_postprocess(tmp_path) -> None:
    server = PolicyServer(
        PolicyServerConfig(
            diagnostic_capture_policy_batches=True,
            diagnostic_capture_dir=str(tmp_path),
            diagnostic_capture_max=1,
        )
    )
    server.policy = MutatingDeploymentPolicy()
    server.preprocessor = lambda observation: {
        "observation.state": torch.tensor([[1.0, 2.0, 3.0]]),
        "observation.language.tokens": torch.tensor([[4, 5]]),
        "task": ["test task"],
    }
    server.postprocessor = lambda action: action + 10
    server.lerobot_features = {}
    server.actions_per_chunk = 4
    server.pretrained_name_or_path = "/checkpoint"
    timed = TimedObservation(timestamp=100.0, timestep=7, observation={}, must_go=True)

    with patch(
        "lerobot_async_inference.policy_server.raw_observation_to_observation",
        side_effect=lambda observation, features, images: observation,
    ):
        actions = server._predict_action_chunk(timed)
    server.stop()

    captured = torch.load(tmp_path / "capture_000.pt", weights_only=False)
    raw = torch.load(tmp_path / "raw_chunk_000.pt", weights_only=False)
    robot = torch.load(tmp_path / "robot_chunk_000.pt", weights_only=False)
    assert torch.equal(captured["observation.state"], torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(raw, torch.arange(12, dtype=torch.float32).reshape(4, 3))
    assert torch.equal(robot, raw + 10)
    assert torch.equal(torch.stack([action.get_action() for action in actions]), robot)
    metadata = json.loads((tmp_path / "capture_metadata.jsonl").read_text().strip())
    assert metadata["policy_timestamp"] == 100.0
    assert metadata["timestep"] == 7
    assert metadata["initial_request"] is True
    assert metadata["checkpoint"] == "/checkpoint"


def _run_background_capture_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        test_background_capture_writes_bounded_cpu_artifacts(Path(directory))


def _run_server_capture_integration_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        test_server_capture_is_immediately_before_policy_and_before_postprocess(Path(directory))


def load_tests(loader, tests, pattern):  # noqa: ARG001
    suite = unittest.TestSuite()
    for function in (
        test_capture_is_disabled_by_default,
        test_fixed_noise_deterministic_replay,
        test_smooth_chunk_has_low_roughness,
        test_alternating_chunk_has_high_oscillation,
        test_known_chunk_shift_aligns_overlap,
        test_identical_aligned_chunks_have_zero_disagreement,
        test_opposite_direction_chunks_have_high_direction_disagreement,
        test_metadata_time_difference_maps_to_frame_shift,
        _run_background_capture_test,
        _run_server_capture_integration_test,
    ):
        suite.addTest(unittest.FunctionTestCase(function))
    return suite
