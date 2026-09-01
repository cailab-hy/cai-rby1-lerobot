import threading
import unittest
from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import torch

from lerobot_async_inference.configs import RobotClientConfig
from lerobot_async_inference.helpers import TimedAction
from lerobot_async_inference.robot_client import RobotClient


def make_action(timestep: int) -> TimedAction:
    return TimedAction(
        timestamp=float(timestep),
        timestep=timestep,
        action=torch.tensor([float(timestep)]),
    )


def action_timing() -> dict[str, float | int]:
    return {
        "queue_size_before": 2,
        "queue_size_after": 1,
        "queue_lock_wait_ms": 0.01,
        "queue_pop_ms": 0.02,
        "action_to_dict_ms": 0.03,
        "latest_action_update_ms": 0.01,
        "robot_send_action_ms": 4.0,
        "action_log_write_ms": 0.2,
        "control_loop_action_total_ms": 4.5,
    }


class TimingDiagnosticsTest(unittest.TestCase):
    def make_client(self) -> RobotClient:
        client = RobotClient.__new__(RobotClient)
        client.action_queue = Queue()
        client.action_queue_lock = threading.Lock()
        client.latest_action_lock = threading.Lock()
        client.latest_action = -1
        client.logger = Mock()
        return client

    def test_diagnostics_are_disabled_by_default(self):
        field = RobotClientConfig.__dataclass_fields__["timing_diagnostics"]
        self.assertIs(field.default, False)

    def test_empty_queue_merge_populates_safe_timing_metadata(self):
        client = self.make_client()
        timing = {}

        client._aggregate_action_queues([], timing=timing)

        self.assertEqual(timing["old_queue_size"], 0)
        self.assertIsNone(timing["old_queue_first_timestep"])
        self.assertEqual(timing["incoming_size"], 0)
        self.assertIsNone(timing["incoming_first_timestep"])
        self.assertEqual(timing["final_queue_size"], 0)
        self.assertGreaterEqual(timing["aggregate_lock_wait_ms"], 0.0)
        self.assertGreaterEqual(timing["aggregate_total_ms"], 0.0)

    def test_merge_without_diagnostics_adds_no_timing_log(self):
        client = self.make_client()

        client._aggregate_action_queues([make_action(0)])

        timing_messages = [
            call.args[0]
            for call in client.logger.debug.call_args_list
            if call.args and isinstance(call.args[0], str) and "[TIMING]" in call.args[0]
        ]
        self.assertEqual(timing_messages, [])

    def test_first_action_and_only_large_interval_trigger_stall(self):
        client = self.make_client()
        client.config = SimpleNamespace(environment_dt=1 / 15)
        client._timing_history = deque(maxlen=10)
        client._last_action_perf_time = None
        client._last_action_timestep = None

        with (
            patch(
                "lerobot_async_inference.robot_client.time.perf_counter",
                side_effect=[1.0, 1.067, 1.397],
            ),
            patch(
                "lerobot_async_inference.robot_client.time.time",
                side_effect=[10.0, 10.067, 10.397],
            ),
        ):
            client._emit_action_timing(48, action_timing())
            client._emit_action_timing(49, action_timing())
            client._timing_history.append(
                {
                    "action": action_timing(),
                    "observation": {
                        "robot_get_observation_ms": 250.0,
                        "transport_preprocess_ms": 6.0,
                        "serialization_ms": 2.0,
                        "grpc_send_observation_ms": 12.0,
                        "control_loop_observation_ms": 270.0,
                    },
                    "loop_total_ms": 279.0,
                    "sleep_budget_ms": 0.0,
                }
            )
            client._emit_action_timing(50, action_timing())

        self.assertEqual(client.logger.warning.call_count, 1)
        warning_format = client.logger.warning.call_args.args[0]
        self.assertIn("[TIMING][STALL]", warning_format)
        self.assertEqual(client.logger.warning.call_args.args[2:4], (49, 50))

    def test_control_loop_action_populates_fields_and_handles_first_action(self):
        client = self.make_client()
        client.config = SimpleNamespace(environment_dt=1 / 15)
        client.robot = SimpleNamespace(
            action_features={"joint": float},
            send_action=Mock(side_effect=lambda action: action),
        )
        client.action_queue.put(make_action(0))
        client.action_queue_size = []
        client._timing_history = deque(maxlen=10)
        client._last_action_perf_time = None
        client._last_action_timestep = None
        timing = {}

        with patch("builtins.open", mock_open()):
            performed = client.control_loop_action(timing=timing)

        self.assertEqual(performed, {"joint": 0.0})
        self.assertEqual(timing["queue_size_before"], 1)
        self.assertEqual(timing["queue_size_after"], 0)
        self.assertIsNone(timing["action_interval_ms"])
        self.assertGreaterEqual(timing["queue_lock_wait_ms"], 0.0)
        self.assertGreaterEqual(timing["robot_send_action_ms"], 0.0)
        self.assertEqual(client.logger.warning.call_count, 0)

    def test_send_observation_reports_serialization_and_rpc_separately(self):
        client = self.make_client()
        client.config = SimpleNamespace(
            backend="grpc",
            image_resize_scale=1.0,
            jpeg_compression=False,
        )
        client.backend = "grpc"
        client.camera_keys = ()
        client.shutdown_event = threading.Event()
        client.stub = SimpleNamespace(SendObservations=Mock(return_value=None))
        # Use the real type because send_observation validates it.
        from lerobot_async_inference.helpers import TimedObservation

        observation = TimedObservation(timestamp=1.0, timestep=0, observation={"state": [0.0]})
        timing = {}

        sent = client.send_observation(observation, timing=timing)

        self.assertTrue(sent)
        self.assertGreater(timing["serialized_bytes"], 0)
        self.assertGreaterEqual(timing["serialization_ms"], 0.0)
        self.assertGreaterEqual(timing["grpc_send_observation_ms"], 0.0)
        self.assertEqual(timing["transport_preprocess_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
