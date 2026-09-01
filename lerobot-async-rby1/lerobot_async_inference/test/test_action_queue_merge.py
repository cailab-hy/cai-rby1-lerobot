import threading
import unittest
from queue import Queue

import torch

from lerobot_async_inference.configs import get_aggregate_function
from lerobot_async_inference.helpers import TimedAction
from lerobot_async_inference.robot_client import RobotClient


def make_action(timestep: int, value: float, timestamp_offset: float = 0.0) -> TimedAction:
    return TimedAction(
        timestamp=float(timestep) + timestamp_offset,
        timestep=timestep,
        action=torch.tensor([value], dtype=torch.float32),
    )


class ActionQueueMergeTest(unittest.TestCase):
    def make_client(self, latest_action: int, old_actions: list[TimedAction]) -> RobotClient:
        client = RobotClient.__new__(RobotClient)
        client.action_queue = Queue()
        for action in old_actions:
            client.action_queue.put(action)
        client.action_queue_lock = threading.Lock()
        client.latest_action_lock = threading.Lock()
        client.latest_action = latest_action
        return client

    def queue_actions(self, client: RobotClient) -> list[TimedAction]:
        with client.action_queue_lock:
            return list(client.action_queue.queue)

    def assert_queue_timesteps(self, client: RobotClient, expected: list[int]) -> list[TimedAction]:
        actions = self.queue_actions(client)
        timesteps = [action.get_timestep() for action in actions]
        self.assertEqual(timesteps, expected)
        self.assertEqual(timesteps, sorted(timesteps))
        self.assertEqual(len(timesteps), len(set(timesteps)))
        return actions

    def test_old_only_overlap_and_incoming_only_are_merged(self):
        old_actions = [make_action(timestep, timestep) for timestep in range(50, 58)]
        incoming_actions = [
            make_action(timestep, 1000 + timestep, timestamp_offset=0.5)
            for timestep in range(55, 61)
        ]
        client = self.make_client(latest_action=49, old_actions=old_actions)

        client._aggregate_action_queues(
            incoming_actions,
            get_aggregate_function("weighted_average"),
        )

        merged = self.assert_queue_timesteps(client, list(range(50, 61)))
        merged_by_timestep = {action.get_timestep(): action for action in merged}
        for timestep in range(50, 55):
            self.assertIs(merged_by_timestep[timestep], old_actions[timestep - 50])
        for timestep in range(55, 58):
            expected = 0.3 * timestep + 0.7 * (1000 + timestep)
            self.assertAlmostEqual(merged_by_timestep[timestep].get_action().item(), expected, places=3)
            self.assertEqual(merged_by_timestep[timestep].get_timestamp(), timestep + 0.5)
        for timestep in range(58, 61):
            self.assertIs(merged_by_timestep[timestep], incoming_actions[timestep - 55])

    def test_incoming_chunk_starting_later_does_not_interpolate_gap(self):
        old_actions = [make_action(timestep, timestep) for timestep in range(50, 55)]
        incoming_actions = [make_action(timestep, timestep) for timestep in range(60, 63)]
        client = self.make_client(latest_action=49, old_actions=old_actions)

        client._aggregate_action_queues(incoming_actions)

        self.assert_queue_timesteps(client, [50, 51, 52, 53, 54, 60, 61, 62])

    def test_old_only_actions_beyond_previous_guard_are_preserved(self):
        old_actions = [make_action(timestep, timestep) for timestep in range(50, 60)]
        incoming_actions = [make_action(timestep, timestep) for timestep in range(65, 68)]
        client = self.make_client(latest_action=49, old_actions=old_actions)

        client._aggregate_action_queues(incoming_actions)

        merged = self.assert_queue_timesteps(client, [*range(50, 60), 65, 66, 67])
        for old_action, merged_action in zip(old_actions, merged[: len(old_actions)], strict=True):
            self.assertIs(merged_action, old_action)

    def test_completely_overlapping_actions_are_aggregated_once(self):
        old_actions = [make_action(timestep, timestep) for timestep in range(50, 53)]
        incoming_actions = [make_action(timestep, 1000 + timestep) for timestep in range(50, 53)]
        client = self.make_client(latest_action=49, old_actions=old_actions)
        calls: list[tuple[float, float]] = []

        def aggregate_once(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
            calls.append((old.item(), new.item()))
            return new

        client._aggregate_action_queues(incoming_actions, aggregate_once)

        self.assert_queue_timesteps(client, [50, 51, 52])
        self.assertEqual(calls, [(50.0, 1050.0), (51.0, 1051.0), (52.0, 1052.0)])

    def test_incoming_only_actions_are_added(self):
        incoming_actions = [make_action(timestep, timestep) for timestep in range(50, 53)]
        client = self.make_client(latest_action=49, old_actions=[])

        client._aggregate_action_queues(incoming_actions)

        merged = self.assert_queue_timesteps(client, [50, 51, 52])
        self.assertEqual(merged, incoming_actions)

    def test_old_only_actions_are_preserved_for_empty_incoming_chunk(self):
        old_actions = [make_action(timestep, timestep) for timestep in range(50, 53)]
        client = self.make_client(latest_action=49, old_actions=old_actions)

        client._aggregate_action_queues([])

        merged = self.assert_queue_timesteps(client, [50, 51, 52])
        for old_action, merged_action in zip(old_actions, merged, strict=True):
            self.assertIs(merged_action, old_action)

    def test_stale_actions_are_filtered_from_both_queues(self):
        old_actions = [make_action(timestep, timestep) for timestep in range(51, 55)]
        incoming_actions = [make_action(timestep, 1000 + timestep) for timestep in range(50, 56)]
        client = self.make_client(latest_action=52, old_actions=old_actions)

        client._aggregate_action_queues(
            incoming_actions,
            get_aggregate_function("weighted_average"),
        )

        self.assert_queue_timesteps(client, [53, 54, 55])

    def test_full_regression_range_preserves_leading_old_only_actions(self):
        old_actions = [make_action(timestep, timestep) for timestep in range(50, 75)]
        incoming_actions = [make_action(timestep, 1000 + timestep) for timestep in range(55, 105)]
        client = self.make_client(latest_action=49, old_actions=old_actions)

        client._aggregate_action_queues(
            incoming_actions,
            get_aggregate_function("weighted_average"),
        )

        merged = self.assert_queue_timesteps(client, list(range(50, 105)))
        for timestep in range(50, 55):
            self.assertIs(merged[timestep - 50], old_actions[timestep - 50])


if __name__ == "__main__":
    unittest.main()
