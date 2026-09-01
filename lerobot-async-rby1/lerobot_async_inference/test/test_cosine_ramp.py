import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from lerobot_async_inference.configs import (
    RobotClientConfig,
    cosine_ramp,
    cosine_ramp_alpha,
    get_aggregate_function,
)
from lerobot_async_inference.helpers import TimedAction
from lerobot_async_inference.robot_client import RobotClient


def make_action(timestep: int, value: float, timestamp_offset: float = 0.0) -> TimedAction:
    return TimedAction(
        timestamp=float(timestep) + timestamp_offset,
        timestep=timestep,
        action=torch.tensor([value], dtype=torch.float32),
    )


class CosineRampMathTest(unittest.TestCase):
    def test_client_config_selects_cosine_ramp_by_name(self):
        config = RobotClientConfig(
            robot=object(),
            actions_per_chunk=50,
            aggregate_fn_name="cosine_ramp",
        )

        self.assertIs(config.aggregate_fn, cosine_ramp)

    def test_alpha_is_strictly_monotonic_inside_unit_interval(self):
        alphas = [cosine_ramp_alpha(index, 10) for index in range(10)]

        self.assertGreater(alphas[0], 0.0)
        self.assertLess(alphas[-1], 1.0)
        self.assertTrue(all(left < right for left, right in zip(alphas, alphas[1:])))

    def test_alpha_is_symmetric(self):
        alphas = [cosine_ramp_alpha(index, 10) for index in range(10)]

        for index, alpha in enumerate(alphas):
            self.assertAlmostEqual(alpha, 1.0 - alphas[-1 - index], places=12)

    def test_odd_overlap_midpoint_is_one_half(self):
        self.assertAlmostEqual(cosine_ramp_alpha(2, 5), 0.5, places=12)

    def test_single_overlap_blends_equally(self):
        merged = cosine_ramp(torch.tensor([0.0]), torch.tensor([1.0]))

        self.assertAlmostEqual(merged.item(), 0.5, places=7)

    def test_equal_actions_remain_equal_for_every_alpha(self):
        action = torch.tensor([0.25, -0.5, 1.5])

        for index in range(10):
            merged = cosine_ramp(
                action,
                action,
                overlap_index=index,
                overlap_count=10,
            )
            torch.testing.assert_close(merged, action)


class CosineRampQueueMergeTest(unittest.TestCase):
    def make_client(self, latest_action: int, old_actions: list[TimedAction]) -> RobotClient:
        client = RobotClient.__new__(RobotClient)
        client.action_queue = Queue()
        for action in old_actions:
            client.action_queue.put(action)
        client.action_queue_lock = threading.Lock()
        client.latest_action_lock = threading.Lock()
        client.latest_action = latest_action
        client.latest_action_tensor = None
        return client

    @staticmethod
    def queued_actions(client: RobotClient) -> list[TimedAction]:
        with client.action_queue_lock:
            return list(client.action_queue.queue)

    def test_synthetic_overlap_uses_temporal_ramp_and_keeps_incoming_only(self):
        old_actions = [make_action(timestep, 0.0) for timestep in range(50, 54)]
        incoming_actions = [make_action(timestep, 1.0, 0.5) for timestep in range(50, 55)]
        client = self.make_client(latest_action=49, old_actions=old_actions)

        client._aggregate_action_queues(
            incoming_actions,
            get_aggregate_function("cosine_ramp"),
        )

        merged = self.queued_actions(client)
        self.assertEqual([action.get_timestep() for action in merged], list(range(50, 55)))
        overlap_values = [action.get_action().item() for action in merged[:4]]
        self.assertTrue(
            all(left < right for left, right in zip(overlap_values, overlap_values[1:]))
        )
        self.assertGreater(overlap_values[0], 0.0)
        self.assertLess(overlap_values[-1], 1.0)
        self.assertEqual(merged[4].get_action().item(), 1.0)
        self.assertEqual(merged[2].get_timestamp(), 52.5)

    def test_existing_pairwise_modes_are_unchanged(self):
        expected_by_mode = {
            "weighted_average": 6.9,
            "average": 5.5,
            "conservative": 4.1,
        }

        for mode, expected in expected_by_mode.items():
            with self.subTest(mode=mode):
                client = self.make_client(49, [make_action(50, 2.0)])
                client._aggregate_action_queues(
                    [make_action(50, 9.0)],
                    get_aggregate_function(mode),
                )
                self.assertAlmostEqual(
                    self.queued_actions(client)[0].get_action().item(), expected, places=6
                )

    def test_true_merge_changes_only_overlap_values(self):
        old_actions = [make_action(timestep, 0.0) for timestep in range(50, 75)]
        incoming_actions = [make_action(timestep, 1.0) for timestep in range(55, 105)]
        client = self.make_client(latest_action=49, old_actions=old_actions)

        client._aggregate_action_queues(
            incoming_actions,
            get_aggregate_function("cosine_ramp"),
        )

        merged = self.queued_actions(client)
        self.assertEqual([action.get_timestep() for action in merged], list(range(50, 105)))
        by_timestep = {action.get_timestep(): action for action in merged}
        for timestep in range(50, 55):
            self.assertIs(by_timestep[timestep], old_actions[timestep - 50])
        for timestep in range(55, 75):
            self.assertGreater(by_timestep[timestep].get_action().item(), 0.0)
            self.assertLess(by_timestep[timestep].get_action().item(), 1.0)
        for timestep in range(75, 105):
            self.assertIs(by_timestep[timestep], incoming_actions[timestep - 55])

    def test_timing_diagnostics_log_ramp_samples_and_boundary_metrics(self):
        client = self.make_client(
            latest_action=49,
            old_actions=[make_action(timestep, 0.0) for timestep in range(50, 54)],
        )
        client.latest_action_tensor = torch.tensor([0.0])
        client.robot = SimpleNamespace(action_features={"left_arm_0": float})
        client.logger = Mock()

        client._aggregate_action_queues(
            [make_action(timestep, 1.0) for timestep in range(50, 55)],
            get_aggregate_function("cosine_ramp"),
            timing={},
        )

        messages = [call.args[0] for call in client.logger.debug.call_args_list]
        self.assertTrue(any(message.startswith("[COSINE_RAMP]") for message in messages))
        self.assertTrue(
            any(message.startswith("[COSINE_RAMP][BOUNDARY]") for message in messages)
        )


if __name__ == "__main__":
    unittest.main()
