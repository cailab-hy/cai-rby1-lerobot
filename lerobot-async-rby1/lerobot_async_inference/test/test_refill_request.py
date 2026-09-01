import pickle
import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import torch

from lerobot_async_inference.helpers import TimedAction
from lerobot_async_inference.robot_client import RobotClient


def make_action(timestep: int) -> TimedAction:
    return TimedAction(
        timestamp=float(timestep),
        timestep=timestep,
        action=torch.tensor([float(timestep)]),
    )


class RefillRequestTest(unittest.TestCase):
    def make_client(self, queue_size: int = 0, timing_diagnostics: bool = False) -> RobotClient:
        client = RobotClient.__new__(RobotClient)
        client.backend = "grpc"
        client.action_queue = Queue()
        for timestep in range(queue_size):
            client.action_queue.put(make_action(timestep))
        client.action_queue_lock = threading.Lock()
        client.latest_action_lock = threading.Lock()
        client.latest_action = -1
        client.action_chunk_size = 50
        client._chunk_size_threshold = 0.5
        client._refill_in_flight = False
        client._refill_request_sent = False
        client._refill_request_queue_size = None
        client._refill_request_queue_ratio = None
        client.must_go = threading.Event()
        client.must_go.set()
        client.shutdown_event = threading.Event()
        client.start_barrier = Mock()
        client.logger = Mock()
        client.config = SimpleNamespace(
            client_device="cpu",
            aggregate_fn=lambda old, new: new,
            timing_diagnostics=timing_diagnostics,
        )
        return client

    @staticmethod
    def drain_to(client: RobotClient, queue_size: int) -> None:
        with client.action_queue_lock:
            while client.action_queue.qsize() > queue_size:
                client.action_queue.get_nowait()

    def run_receiver_once(self, client: RobotClient, data: bytes) -> None:
        calls = 0

        def get_actions(_request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return SimpleNamespace(data=data)
            client.shutdown_event.set()
            return SimpleNamespace(data=b"")

        client.stub = SimpleNamespace(GetActions=get_actions)
        client.receive_actions()

    def test_threshold_crossing_reserves_exactly_at_half_chunk(self):
        client = self.make_client(queue_size=26)

        self.assertFalse(client._ready_to_send_observation())
        self.drain_to(client, 25)
        self.assertTrue(client._ready_to_send_observation())
        self.assertTrue(client._refill_in_flight)
        self.assertEqual(client._refill_request_queue_size, 25)
        self.assertEqual(client._refill_request_queue_ratio, 0.5)

    def test_in_flight_refill_blocks_duplicate_requests(self):
        client = self.make_client(queue_size=25)
        self.assertTrue(client._ready_to_send_observation())

        for queue_size in (24, 20, 10, 1, 0):
            self.drain_to(client, queue_size)
            self.assertFalse(client._ready_to_send_observation())

    def test_successful_chunk_merge_clears_refill_state(self):
        client = self.make_client(queue_size=20, timing_diagnostics=True)
        client._refill_in_flight = True
        client._refill_request_sent = True
        incoming = [make_action(timestep) for timestep in range(20, 70)]

        self.run_receiver_once(client, pickle.dumps(incoming))

        self.assertFalse(client._refill_in_flight)
        self.assertEqual(client.action_queue.qsize(), 70)
        refill_logs = [
            call.args[0]
            for call in client.logger.info.call_args_list
            if call.args and "[TIMING][REFILL_COMPLETE]" in call.args[0]
        ]
        self.assertEqual(len(refill_logs), 1)

    def test_next_threshold_cycle_can_reserve_one_new_request(self):
        client = self.make_client(queue_size=25)
        self.assertTrue(client._ready_to_send_observation())
        client._clear_refill_in_flight()

        with client.action_queue_lock:
            client.action_queue = Queue()
            for timestep in range(50):
                client.action_queue.put(make_action(timestep))

        self.assertFalse(client._ready_to_send_observation())
        self.drain_to(client, 25)
        self.assertTrue(client._ready_to_send_observation())
        self.assertFalse(client._ready_to_send_observation())

    def test_empty_response_releases_sent_request_for_starvation_retry(self):
        client = self.make_client(queue_size=0)
        client._refill_in_flight = True
        client._refill_request_sent = True

        def get_empty(_request):
            client.shutdown_event.set()
            return SimpleNamespace(data=b"")

        client.stub = SimpleNamespace(GetActions=get_empty)
        client.receive_actions()

        self.assertFalse(client._refill_in_flight)
        self.assertTrue(client._ready_to_send_observation())

    def test_startup_forces_one_observation_and_marks_it_must_go(self):
        client = self.make_client(queue_size=0, timing_diagnostics=True)
        client.action_chunk_size = -1
        client.robot = SimpleNamespace(get_observation=Mock(return_value={"state": [0.0]}))
        sent_observations = []

        def send_observation(observation, timing=None):
            sent_observations.append(observation)
            return True

        client.send_observation = Mock(side_effect=send_observation)

        self.assertTrue(client._ready_to_send_observation())
        client.control_loop_observation(task="test", force_refill=True)

        self.assertEqual(len(sent_observations), 1)
        self.assertTrue(sent_observations[0].must_go)
        self.assertTrue(client._refill_in_flight)
        self.assertTrue(client._refill_request_sent)
        self.assertFalse(client._ready_to_send_observation())
        refill_logs = [
            call.args[0]
            for call in client.logger.info.call_args_list
            if call.args and "[TIMING][REFILL_REQUEST]" in call.args[0]
        ]
        self.assertEqual(len(refill_logs), 1)

    def test_failed_observation_send_rolls_back_for_retry(self):
        client = self.make_client(queue_size=0)
        client.action_chunk_size = -1
        client.robot = SimpleNamespace(get_observation=Mock(return_value={"state": [0.0]}))
        client.send_observation = Mock(return_value=False)

        self.assertTrue(client._ready_to_send_observation())
        client.control_loop_observation(task="test", force_refill=True)

        self.assertFalse(client._refill_in_flight)
        self.assertTrue(client.must_go.is_set())
        self.assertTrue(client._ready_to_send_observation())

    def test_receiver_rpc_error_rolls_back_for_retry(self):
        client = self.make_client(queue_size=0)
        client._refill_in_flight = True
        client._refill_request_sent = True

        def raise_rpc_error(_request):
            client.shutdown_event.set()
            raise grpc.RpcError("disconnected")

        client.stub = SimpleNamespace(GetActions=raise_rpc_error)
        client.receive_actions()

        self.assertFalse(client._refill_in_flight)
        self.assertTrue(client.must_go.is_set())


if __name__ == "__main__":
    unittest.main()
