import json
import threading
import unittest
from queue import Queue
from types import SimpleNamespace

import torch

from lerobot_async_inference.configs import AGGREGATE_FUNCTIONS
from lerobot_async_inference.helpers import TimedAction
from lerobot_async_inference.robot_client import RobotClient


def timed_action(timestep: int, values: list[float]) -> TimedAction:
    return TimedAction(timestamp=float(timestep), timestep=timestep, action=torch.tensor(values))


def make_client(
    *,
    debug: bool,
    action_feature_names: list[str],
    crossfade: bool = False,
) -> RobotClient:
    client = RobotClient.__new__(RobotClient)
    client.config = SimpleNamespace(
        debug_weighted_aggregation=debug,
        aggregate_fn_name="weighted_average",
        direction_reversal_epsilon=1e-8,
        arm_temporal_crossfade=crossfade,
    )
    client.robot = SimpleNamespace(action_features=action_feature_names)
    client.action_queue = Queue()
    client.action_queue_lock = threading.Lock()
    client.latest_action_lock = threading.Lock()
    client.latest_action = 9
    client.latest_executed_action = None
    client._chunk_transition_id = 0
    client._weighted_aggregation_log_queue = Queue(maxsize=100)
    client._weighted_aggregation_dropped_transitions = 0
    return client


def run_crossfade_scenario(
    blend_count: int,
    *,
    crossfade: bool,
    debug: bool = True,
) -> tuple[RobotClient, list[dict]]:
    names = ["left_arm_0", "left_arm_1", "left_gripper_0"]
    client = make_client(debug=debug, action_feature_names=names, crossfade=crossfade)
    old_values = [0.0, 1.0, 0.2]
    incoming_values = [1.0, 3.0, 0.8]
    for timestep in range(10, 15 + blend_count):
        client.action_queue.put(timed_action(timestep, old_values))
    incoming = [
        timed_action(timestep, incoming_values)
        for timestep in range(10, 16 + blend_count)
    ]
    client._aggregate_action_queues(
        incoming,
        AGGREGATE_FUNCTIONS["weighted_average"],
        transition_id=1,
    )
    records = (
        client._weighted_aggregation_log_queue.get_nowait() if debug else []
    )
    return client, records


def queue_actions(client: RobotClient) -> dict[int, torch.Tensor]:
    return {
        action.get_timestep(): action.get_action()
        for action in client.action_queue.queue
    }


class WeightedAggregationLoggerTest(unittest.TestCase):
    def test_guard_and_exact_weighted_result(self):
        client = make_client(debug=True, action_feature_names=["left_arm_0", "left_arm_1"])
        for timestep in range(10, 21):
            client.action_queue.put(timed_action(timestep, [0.0, 0.0]))
        incoming = [timed_action(timestep, [1.0, 2.0]) for timestep in range(10, 31)]

        client._aggregate_action_queues(
            incoming,
            AGGREGATE_FUNCTIONS["weighted_average"],
            transition_id=1,
        )

        records = client._weighted_aggregation_log_queue.get_nowait()
        json.dumps(records)
        header = records[0]
        summary = records[-1]
        details = {record["timestep"]: record for record in records if record["record_type"] == "timestep_detail"}
        self.assertEqual(header["guard_preserved_timesteps"], [10, 11, 12, 13, 14])
        self.assertEqual(header["first_blended_timestep"], 15)
        self.assertFalse(details[10]["aggregation_applied"])
        self.assertEqual(details[10]["preservation_reason"], "guard_steps")
        self.assertTrue(details[15]["is_first_blended_timestep"])
        self.assertEqual(details[15]["old_weight"], 0.3)
        self.assertEqual(details[15]["incoming_weight"], 0.7)
        self.assertEqual(summary["first_blend_old_weight"], 0.3)
        self.assertEqual(summary["first_blend_incoming_weight"], 0.7)
        self.assertTrue(torch.equal(torch.tensor(details[15]["merged_action"]), torch.tensor([0.7, 1.4])))

        actual = {
            action.get_timestep(): action.get_action()
            for action in client.action_queue.queue
        }
        self.assertTrue(torch.equal(torch.tensor(details[15]["merged_action"]), actual[15]))

    def test_large_disagreement(self):
        client = make_client(
            debug=True,
            action_feature_names=[f"left_arm_{index}" for index in range(6)],
        )
        old_action = torch.zeros(6)
        incoming_action = torch.zeros(6)
        incoming_action[5] = 0.15
        merged_action = 0.3 * old_action + 0.7 * incoming_action
        record = client._weighted_aggregation_detail_record(
            transition_id=1,
            timestep=15,
            region="blended",
            old_action=old_action,
            incoming_action=incoming_action,
            merged_action=merged_action,
            old_weight=0.3,
            incoming_weight=0.7,
            blend_index=1,
            blend_count=1,
            arm_crossfade_enabled=False,
            arm_old_weight=0.3,
            arm_incoming_weight=0.7,
            arm_indices=[0, 1, 2, 3, 4, 5],
            arm_feature_names=[f"left_arm_{index}" for index in range(6)],
            first_blended_timestep=15,
            is_first_new_only_timestep=False,
            old_actions={15: old_action},
            merged_actions={15: merged_action},
            action_feature_names=list(client.robot.action_features),
        )
        self.assertAlmostEqual(record["old_incoming_max_abs_delta"], 0.15, places=6)
        self.assertEqual(record["old_incoming_max_abs_delta_index"], 5)
        self.assertEqual(record["old_incoming_max_abs_delta_name"], "left_arm_5")

    def test_direction_reversal(self):
        client = make_client(debug=True, action_feature_names=["left_arm_0"])
        record = client._direction_reversal_fields(
            timestep=2,
            old_actions={0: torch.tensor([0.0]), 1: torch.tensor([0.1]), 2: torch.tensor([0.2])},
            merged_actions={0: torch.tensor([0.0]), 1: torch.tensor([0.1]), 2: torch.tensor([0.05])},
            action_feature_names=["left_arm_0"],
        )
        self.assertEqual(record["direction_reversal_indices"], [0])
        self.assertEqual(record["direction_reversal_names"], ["left_arm_0"])

    def test_debug_on_off_queue_equivalence(self):
        clients = [
            make_client(debug=False, action_feature_names=["left_arm_0", "left_arm_1"]),
            make_client(debug=True, action_feature_names=["left_arm_0", "left_arm_1"]),
        ]
        for client in clients:
            for timestep in range(10, 21):
                client.action_queue.put(timed_action(timestep, [timestep / 10, -timestep / 10]))
            incoming = [
                timed_action(timestep, [timestep / 20, timestep / 30])
                for timestep in range(10, 31)
            ]
            client._aggregate_action_queues(
                incoming,
                AGGREGATE_FUNCTIONS["weighted_average"],
                transition_id=1,
            )

        queues = [list(client.action_queue.queue) for client in clients]
        self.assertEqual(
            [action.get_timestep() for action in queues[0]],
            [action.get_timestep() for action in queues[1]],
        )
        for without_debug, with_debug in zip(*queues):
            self.assertTrue(torch.equal(without_debug.get_action(), with_debug.get_action()))


class ArmTemporalCrossfadeTest(unittest.TestCase):
    def test_case_1_three_step_arm_crossfade_and_logger(self):
        client, records = run_crossfade_scenario(3, crossfade=True)
        actions = queue_actions(client)
        expected_weights = [0.25, 0.5, 0.75]
        for timestep, incoming_weight in zip(range(15, 18), expected_weights):
            expected = torch.tensor(
                [incoming_weight, 1.0 + 2.0 * incoming_weight, 0.62]
            )
            torch.testing.assert_close(actions[timestep], expected)

        details = {
            record["timestep"]: record
            for record in records
            if record["record_type"] == "timestep_detail"
        }
        self.assertEqual(records[0]["arm_indices"], [0, 1])
        self.assertEqual(records[0]["arm_feature_names"], ["left_arm_0", "left_arm_1"])
        self.assertEqual(records[0]["non_arm_feature_names"], ["left_gripper_0"])
        for blend_index, timestep in enumerate(range(15, 18), start=1):
            self.assertEqual(details[timestep]["blend_index"], blend_index)
            self.assertEqual(details[timestep]["blend_count"], 3)
            self.assertAlmostEqual(
                details[timestep]["arm_incoming_weight"],
                expected_weights[blend_index - 1],
            )
            self.assertAlmostEqual(
                details[timestep]["arm_old_weight"],
                1.0 - expected_weights[blend_index - 1],
            )
            self.assertEqual(details[timestep]["non_arm_old_weight"], 0.3)
            self.assertEqual(details[timestep]["non_arm_incoming_weight"], 0.7)
        self.assertAlmostEqual(records[-1]["first_blend_arm_old_merged_max_delta"], 0.5)

    def test_case_2_four_step_arm_crossfade(self):
        client, _ = run_crossfade_scenario(4, crossfade=True, debug=False)
        actions = queue_actions(client)
        for timestep, incoming_weight in zip(range(15, 19), [0.2, 0.4, 0.6, 0.8]):
            self.assertAlmostEqual(actions[timestep][0].item(), incoming_weight, places=6)

    def test_case_3_one_step_arm_crossfade(self):
        client, _ = run_crossfade_scenario(1, crossfade=True, debug=False)
        actions = queue_actions(client)
        torch.testing.assert_close(actions[15][:2], torch.tensor([0.5, 2.0]))

    def test_case_4_gripper_unchanged(self):
        without_crossfade, _ = run_crossfade_scenario(3, crossfade=False, debug=False)
        with_crossfade, _ = run_crossfade_scenario(3, crossfade=True, debug=False)
        baseline = queue_actions(without_crossfade)
        crossfaded = queue_actions(with_crossfade)
        for timestep in [15, 16, 17]:
            self.assertTrue(torch.equal(baseline[timestep][2], crossfaded[timestep][2]))

    def test_case_5_guard_unchanged(self):
        without_crossfade, _ = run_crossfade_scenario(3, crossfade=False, debug=False)
        with_crossfade, _ = run_crossfade_scenario(3, crossfade=True, debug=False)
        baseline = queue_actions(without_crossfade)
        crossfaded = queue_actions(with_crossfade)
        for timestep in range(10, 15):
            self.assertTrue(torch.equal(baseline[timestep], crossfaded[timestep]))
            self.assertTrue(torch.equal(crossfaded[timestep], torch.tensor([0.0, 1.0, 0.2])))

    def test_case_6_new_only_unchanged(self):
        client, records = run_crossfade_scenario(3, crossfade=True)
        actions = queue_actions(client)
        self.assertTrue(torch.equal(actions[18], torch.tensor([1.0, 3.0, 0.8])))
        summary = records[-1]
        self.assertEqual(summary["first_new_only_timestep"], 18)
        self.assertIsNotNone(summary["last_blend_to_new_only_arm_max_delta"])
        new_only_detail = next(
            record
            for record in records
            if record.get("is_first_new_only_timestep")
        )
        self.assertEqual(new_only_detail["aggregation_region"], "new_only")

    def test_case_7_crossfade_disabled_equivalence(self):
        client, _ = run_crossfade_scenario(3, crossfade=False, debug=False)
        actions = queue_actions(client)
        old_action = torch.tensor([0.0, 1.0, 0.2])
        incoming_action = torch.tensor([1.0, 3.0, 0.8])
        expected_blend = AGGREGATE_FUNCTIONS["weighted_average"](
            old_action, incoming_action
        )
        for timestep in range(10, 15):
            self.assertTrue(torch.equal(actions[timestep], old_action))
        for timestep in range(15, 18):
            self.assertTrue(torch.equal(actions[timestep], expected_blend))
        self.assertTrue(torch.equal(actions[18], incoming_action))

    def test_case_8_arm_result_numerical_check(self):
        client, _ = run_crossfade_scenario(3, crossfade=True, debug=False)
        actions = queue_actions(client)
        self.assertTrue(torch.equal(actions[15][:2], torch.tensor([0.25, 1.5])))


if __name__ == "__main__":
    unittest.main()
