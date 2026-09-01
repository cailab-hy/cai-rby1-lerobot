from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lerobot_async_inference.configs import PolicyServerConfig
from lerobot_async_inference.frozen_noise_analysis import (
    analyze_actions,
    clone_to_cpu,
    execute_condition,
    nested_checksum,
)
from lerobot_async_inference.policy_server import PolicyServer


class FakePolicy:
    def __init__(self) -> None:
        self.training = True
        self.reset_count = 0
        self.seen_noise: list[torch.Tensor | None] = []

    def eval(self) -> "FakePolicy":
        self.training = False
        return self

    def reset(self) -> None:
        self.reset_count += 1

    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.seen_noise.append(None if noise is None else noise.detach().clone())
        # Simulate a policy that mutates both its batch and its per-call noise.
        batch["observation.state"].add_(100)
        if noise is None:
            return torch.full((1, 4, 3), float(self.reset_count))
        output = noise[:, :4, :3].clone()
        noise.add_(100)
        return output


def test_recursive_clone_preserves_source_and_tensor_values() -> None:
    source = {
        "tensor": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "nested": [torch.tensor([True]), (torch.tensor([2], dtype=torch.int64), "task")],
    }
    checksum = nested_checksum(source)
    cloned = clone_to_cpu(source)

    cloned["tensor"].add_(10)
    cloned["nested"][0].logical_not_()

    assert nested_checksum(source) == checksum
    assert source["tensor"][0, 0].item() == 0
    assert source["nested"][0].item() is True


def test_random_and_fixed_conditions_reset_clone_and_stack() -> None:
    frozen = {"observation.state": torch.zeros(1, 3)}
    fixed_noise = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    fixed_checksum = nested_checksum(fixed_noise)

    random_policy = FakePolicy()
    random_result = execute_condition(
        policy=random_policy,
        frozen_batch=frozen,
        postprocessor=lambda action: action + 0.5,
        device=torch.device("cpu"),
        num_runs=3,
        fixed_noise=None,
    )
    assert random_policy.reset_count == 3
    assert random_policy.seen_noise == [None, None, None]
    assert random_result.raw_actions.shape == (3, 4, 3)
    assert random_result.robot_actions.shape == (3, 4, 3)
    assert torch.equal(frozen["observation.state"], torch.zeros(1, 3))

    fixed_policy = FakePolicy()
    fixed_result = execute_condition(
        policy=fixed_policy,
        frozen_batch=frozen,
        postprocessor=lambda action: action,
        device=torch.device("cpu"),
        num_runs=3,
        fixed_noise=fixed_noise,
    )
    assert fixed_policy.reset_count == 3
    assert all(torch.equal(noise, fixed_noise) for noise in fixed_policy.seen_noise)
    assert torch.equal(fixed_result.raw_actions[0], fixed_result.raw_actions[1])
    assert nested_checksum(fixed_noise) == fixed_checksum


def test_delta_second_delta_and_sign_flip_metrics() -> None:
    # Delta for joint 0 is +1, -1, +1; second delta is -2, +2.
    actions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]])
    groups = {
        "right_arm": [0],
        "left_arm": [1],
        "arms": [0, 1],
        "gripper": [1],
    }
    metrics, run_differences, oscillations = analyze_actions(actions, groups, 0.005)
    lookup = {(row["scope"], row["metric"]): row["value"] for row in metrics}

    assert lookup[("arms", "within_chunk_delta_max")] == 1.0
    assert lookup[("arms", "within_chunk_second_delta_max")] == 2.0
    assert lookup[("right_arm", "oscillation_ratio")] == 1.0
    assert torch.isnan(torch.tensor(lookup[("left_arm", "oscillation_ratio")]))
    assert run_differences == []
    right_joint = next(
        row
        for row in oscillations
        if row["scope"] == "right_arm" and row["action_index"] == 0
    )
    assert right_joint["eligible_pairs"] == 2


def test_server_dump_is_one_shot_and_cpu_cloned(tmp_path) -> None:
    path = tmp_path / "nested" / "frozen.pt"
    server = PolicyServer(
        PolicyServerConfig(dump_frozen_policy_batch=str(path))
    )
    observation = {
        "observation.state": torch.tensor([[1.0, 2.0]]),
        "observation.language.tokens": torch.tensor([[1, 2]], dtype=torch.int64),
    }

    server._dump_frozen_policy_batch_once(observation, timestep=7)
    observation["observation.state"].add_(10)
    server._dump_frozen_policy_batch_once(observation, timestep=8)

    saved = torch.load(path, weights_only=False)
    assert saved["observation.state"].device.type == "cpu"
    assert torch.equal(saved["observation.state"], torch.tensor([[1.0, 2.0]]))
    assert server._frozen_batch_dumped is True


def _run_server_dump_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        test_server_dump_is_one_shot_and_cpu_cloned(Path(directory))


def load_tests(loader, tests, pattern):  # noqa: ARG001
    """Expose the pytest-style functions to stdlib unittest discovery too."""
    suite = unittest.TestSuite()
    for function in (
        test_recursive_clone_preserves_source_and_tensor_values,
        test_random_and_fixed_conditions_reset_clone_and_stack,
        test_delta_second_delta_and_sign_flip_metrics,
        _run_server_dump_test,
    ):
        suite.addTest(unittest.FunctionTestCase(function))
    return suite
