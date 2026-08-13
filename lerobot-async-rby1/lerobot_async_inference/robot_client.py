# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```

GR00T ZMQ example:

python -m lerobot.async_inference.robot_client \
  --backend=groot_zmq \
  --robot.type=rby1 \
  --server_address=127.0.0.1:5555  \
  --task="pick up the can" \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average


Pi05 ZMQ example:

python -m lerobot.async_inference.robot_client \
  --backend=pi05_zmq \
  --robot.type=rby1 \
  --server_address=127.0.0.1:5556 \
  --task="move the object to the target" \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average




"""

import json
import logging
import pickle  # nosec
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pprint import pformat
from queue import Empty, Full, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()

from .policy.groot_zmq import (
    GR00TZMQClient,
    build_groot_n16_observation,
    groot_n16_action_dict_to_timed_actions,
    validate_groot_robot_compatibility,
)

from .policy.pi05_zmq import (
    Pi05ZMQClient,
    build_pi05_observation,
    pi05_action_dict_to_timed_actions,
    validate_pi05_robot_compatibility,
)



from .configs import RobotClientConfig
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    resize_image_by_scale,
    scale_lerobot_image_features,
    visualize_action_queue_size,
)

LOG_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

RAW_ACTION_LOG = f"raw_actions_{LOG_RUN_ID}.jsonl"
FINAL_ACTION_LOG = f"final_actions_{LOG_RUN_ID}.jsonl"
OBSERVATION_LOG = f"observations_{LOG_RUN_ID}.jsonl"
PREFETCH_EVENT_LOG = f"prefetch_events_{LOG_RUN_ID}.jsonl"
CHUNK_TRANSITION_LOG = f"chunk_transitions_{LOG_RUN_ID}.jsonl"
WEIGHTED_AGGREGATION_LOG = f"weighted_aggregation_{LOG_RUN_ID}.jsonl"

REMOTE_ZMQ_BACKENDS = {
    "groot_zmq": {
        "label": "GR00T",
        "client_cls": GR00TZMQClient,
        "validator": validate_groot_robot_compatibility,
        "build_observation": build_groot_n16_observation,
        "convert_actions": groot_n16_action_dict_to_timed_actions,
    },
    "pi05_zmq": {
        "label": "Pi05",
        "client_cls": Pi05ZMQClient,
        "validator": validate_pi05_robot_compatibility,
        "build_observation": build_pi05_observation,
        "convert_actions": pi05_action_dict_to_timed_actions,
    },
}

def get_remote_backend_spec(backend: str) -> dict[str, Any] | None:
    return REMOTE_ZMQ_BACKENDS.get(backend)



class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()
        self.backend = getattr(config, "backend", "grpc")

        self.policy_config = None
        self.channel = None
        self.stub = None
        self.remote_client = None

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        if self.uses_grpc_backend:
            lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
            self.transport_image_keys = [
                key.removeprefix("observation.images.")
                for key, feature in lerobot_features.items()
                if feature.get("dtype") in {"image", "video"}
            ]
            lerobot_features = scale_lerobot_image_features(
                lerobot_features,
                config.image_resize_scale,
            )

            self.policy_config = RemotePolicyConfig(
                policy_type=config.policy_type,
                pretrained_name_or_path=config.pretrained_name_or_path,
                lerobot_features=lerobot_features,
                actions_per_chunk=config.actions_per_chunk,
                device=config.policy_device,
                transport_image_scale=config.image_resize_scale,
            )
            self.channel = grpc.insecure_channel(
                self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
            )
            self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
            self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        elif self.uses_remote_zmq_backend:
            backend_spec = get_remote_backend_spec(self.backend)
            if backend_spec is None:
                raise ValueError(f"Unsupported remote backend: {self.backend}")

            backend_spec["validator"](
                self.robot,
                front_camera_key=self.config.front_camera_key,
                left_wrist_camera_key=self.config.left_wrist_camera_key,
                right_wrist_camera_key=self.config.right_wrist_camera_key,
            )

            self.logger.info(
                f"Initializing {backend_spec['label']} ZMQ client to connect to server at {self.server_address}"
            )

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        # Latest-only observation queue for async ZMQ inference.
        # maxsize=1 avoids running inference on stale observations.
        self.remote_observation_queue = Queue(maxsize=1)
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

        # gRPC-only one-shot prefetch scheduling state. None of this enters policy observations.
        self.prefetch_state_lock = threading.Lock()
        self.prefetch_log_lock = threading.Lock()
        self.prefetch_request_pending = False
        self.pending_prefetch_request_id = None
        self.pending_prefetch_observation_timestep = None
        self.prefetch_request_sent_monotonic_time = None
        self._prefetch_request_sequence = 0
        self._prefetch_trigger_request_id = None
        self._previous_action_queue_ratio = None
        self._prefetch_pending_skip_logged = False

        # Optional chunk diagnostics are produced in the receiver thread and written by a
        # dedicated writer so file I/O never blocks the control loop.
        self._chunk_transition_id = 0
        self._chunk_transition_log_queue = None
        self._chunk_transition_writer_thread = None
        self._recent_action_periods = deque(maxlen=max(1, self.config.fps))
        self._previous_action_monotonic_time = None
        self.previous_executed_timestep = None
        if self.config.debug_chunk_transitions:
            self._chunk_transition_log_queue = Queue()
            self._chunk_transition_writer_thread = threading.Thread(
                target=self._chunk_transition_writer,
                name="chunk-transition-writer",
                daemon=True,
            )
            self._chunk_transition_writer_thread.start()

        self._weighted_aggregation_log_queue = None
        self._weighted_aggregation_writer_thread = None
        self._weighted_aggregation_dropped_transitions = 0
        self.latest_executed_action = None
        if self.config.debug_weighted_aggregation:
            self._weighted_aggregation_log_queue = Queue(
                maxsize=self.config.weighted_aggregation_logger_queue_maxsize
            )
            self._weighted_aggregation_writer_thread = threading.Thread(
                target=self._weighted_aggregation_writer,
                name="weighted-aggregation-writer",
                daemon=True,
            )
            self._weighted_aggregation_writer_thread.start()

        # The gRPC sender publishes compact image transport timing/shape metadata here;
        # the next final-action record consumes it without retaining any image pixels.
        self.observation_debug_lock = threading.Lock()
        self._pending_observation_transport_debug = None
        self._last_send_observation_diagnostics = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()
    
    @property
    def uses_grpc_backend(self) -> bool:
        return self.backend == "grpc"

    @property
    def uses_groot_backend(self) -> bool:
        return self.backend == "groot_zmq"
    
    @property
    def uses_pi05_backend(self) -> bool:
        return self.backend == "pi05_zmq"

    @property
    def uses_remote_zmq_backend(self) -> bool:
        return self.backend in REMOTE_ZMQ_BACKENDS

    @property
    def remote_backend_name(self) -> str:
        if self.uses_groot_backend:
            return "GR00T"
        if self.uses_pi05_backend:
            return "Pi05"
        return "Remote"

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            if self.uses_grpc_backend:
                # client-server handshake
                start_time = time.perf_counter()
                self.stub.Ready(services_pb2.Empty())
                end_time = time.perf_counter()
                self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

                # send policy instructions
                policy_config_bytes = pickle.dumps(self.policy_config)
                policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

                self.logger.info("Sending policy instructions to policy server")
                self.logger.debug(
                    f"Policy type: {self.policy_config.policy_type} | "
                    f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                    f"Device: {self.policy_config.device}"
                )

                self.stub.SendPolicyInstructions(policy_setup)

            elif self.uses_remote_zmq_backend:
                backend_spec = get_remote_backend_spec(self.backend)
                if backend_spec is None:
                    raise ValueError(f"Unsupported remote backend: {self.backend}")

                self.remote_client = backend_spec["client_cls"](
                    server_address=self.server_address,
                    timeout_ms=self.config.zmq_timeout_ms,
                )

                error_name = backend_spec["label"]

                if not self.remote_client.ping():
                    self.logger.error(f"Failed to connect to {error_name} inference server")
                    self.remote_client.close()
                    self.remote_client = None
                    return False

            else:
                raise ValueError(f"Unsupported backend: {self.backend}")

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        if self._chunk_transition_log_queue is not None:
            self._chunk_transition_log_queue.put(None)
        if self._chunk_transition_writer_thread is not None:
            self._chunk_transition_writer_thread.join(timeout=2.0)
        if self._weighted_aggregation_log_queue is not None:
            try:
                self._weighted_aggregation_log_queue.put_nowait(None)
            except Full:
                # The daemon writer will finish whatever was already queued; shutdown must not block.
                pass
        if self._weighted_aggregation_writer_thread is not None:
            self._weighted_aggregation_writer_thread.join(timeout=2.0)

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        if self.channel is not None:
            self.channel.close()
            self.logger.debug("Client stopped, channel closed")

        if self.remote_client is not None:
            self.remote_client.close()
            self.remote_client = None
            self.logger.debug("Remote ZMQ client closed")

    def _prefetch_state_snapshot(self) -> dict[str, Any]:
        with self.prefetch_state_lock:
            return {
                "prefetch_request_pending": self.prefetch_request_pending,
                "pending_prefetch_request_id": self.pending_prefetch_request_id,
                "pending_prefetch_observation_timestep": (
                    self.pending_prefetch_observation_timestep
                ),
                "prefetch_request_sent_monotonic_time": (
                    self.prefetch_request_sent_monotonic_time
                ),
            }

    def _log_prefetch_event(
        self,
        event: str,
        *,
        timestep: int | None,
        queue_size: int,
        queue_ratio: float,
        request_id: int | None,
        **fields: Any,
    ) -> None:
        record = {
            "event": event,
            "wall_time": time.time(),
            "monotonic_time": time.perf_counter(),
            "timestep": timestep,
            "queue_size": queue_size,
            "queue_ratio": queue_ratio,
            "request_id": request_id,
            "prefetch_request_id": request_id,
            **self._prefetch_state_snapshot(),
            **fields,
        }
        try:
            with self.prefetch_log_lock:
                with open(PREFETCH_EVENT_LOG, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
        except Exception as error:
            # Debug logging must never block scheduling recovery or acknowledgement handling.
            self.logger.warning("Failed to write prefetch event %s: %s", event, error)

    def _action_queue_ratio(self) -> tuple[int, float]:
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
        if self.action_chunk_size <= 0:
            return queue_size, 0.0
        return queue_size, queue_size / self.action_chunk_size

    def _chunk_transition_writer(self) -> None:
        """Drain chunk transition records without blocking inference or robot control."""
        assert self._chunk_transition_log_queue is not None
        try:
            with open(CHUNK_TRANSITION_LOG, "a", encoding="utf-8") as stream:
                while True:
                    record = self._chunk_transition_log_queue.get()
                    if record is None:
                        break
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
        except Exception as error:
            self.logger.warning("Chunk transition writer stopped after an error: %s", error)

    def _next_chunk_transition_id(self) -> int:
        self._chunk_transition_id += 1
        return self._chunk_transition_id

    def _weighted_aggregation_writer(self) -> None:
        """Write complete transition batches without blocking the action receiver."""
        assert self._weighted_aggregation_log_queue is not None
        try:
            with open(WEIGHTED_AGGREGATION_LOG, "a", encoding="utf-8") as stream:
                while True:
                    records = self._weighted_aggregation_log_queue.get()
                    if records is None:
                        break
                    for record in records:
                        stream.write(json.dumps(record) + "\n")
                    stream.flush()
        except Exception as error:
            self.logger.warning("Weighted aggregation writer stopped after an error: %s", error)

    @staticmethod
    def _is_arm_joint_action_feature(name: str) -> bool:
        """Match RBY1 joint-mode arm features without including grippers or EE poses."""
        prefix, separator, joint_index = name.rpartition("_")
        return (
            separator == "_"
            and prefix in {"left_arm", "right_arm"}
            and joint_index.isdigit()
        )

    @classmethod
    def _arm_action_metadata(
        cls, action_feature_names: list[str]
    ) -> tuple[list[int], list[str]]:
        arm_indices = [
            index
            for index, name in enumerate(action_feature_names)
            if cls._is_arm_joint_action_feature(name)
        ]
        return arm_indices, [action_feature_names[index] for index in arm_indices]

    @staticmethod
    def _action_delta_fields(
        prefix: str,
        delta: torch.Tensor,
        action_feature_names: list[str],
    ) -> dict[str, Any]:
        """Convert one action delta to compact JSON diagnostics."""
        summary_prefix = {
            "old_vs_incoming": "old_incoming",
            "old_vs_merged": "old_merged",
            "incoming_vs_merged": "incoming_merged",
        }.get(prefix, prefix)
        delta_cpu = delta.detach().cpu()
        delta_values = delta_cpu.tolist()
        if not delta_values:
            return {
                f"action_delta_{prefix}": [],
                f"{summary_prefix}_max_abs_delta": None,
                f"{summary_prefix}_max_abs_delta_index": None,
                f"{summary_prefix}_max_abs_delta_name": None,
                f"{summary_prefix}_max_abs_delta_rad": None,
                f"{summary_prefix}_max_abs_delta_deg": None,
            }

        max_index = int(torch.argmax(torch.abs(delta_cpu)).item())
        max_delta = float(torch.abs(delta_cpu[max_index]).item())
        max_name = (
            action_feature_names[max_index]
            if max_index < len(action_feature_names)
            else None
        )
        is_arm_joint = bool(
            max_name
            and (max_name.startswith("left_arm_") or max_name.startswith("right_arm_"))
        )
        return {
            f"action_delta_{prefix}": delta_values,
            f"{summary_prefix}_max_abs_delta": max_delta,
            f"{summary_prefix}_max_abs_delta_index": max_index,
            f"{summary_prefix}_max_abs_delta_name": max_name,
            f"{summary_prefix}_max_abs_delta_rad": max_delta if is_arm_joint else None,
            f"{summary_prefix}_max_abs_delta_deg": (
                max_delta * 180.0 / 3.141592653589793 if is_arm_joint else None
            ),
        }

    @staticmethod
    def _arm_delta_fields(
        prefix: str,
        delta: torch.Tensor,
        arm_indices: list[int],
        action_feature_names: list[str],
    ) -> dict[str, Any]:
        if not arm_indices:
            return {
                f"{prefix}_max_abs_delta": None,
                f"{prefix}_max_abs_delta_index": None,
                f"{prefix}_max_abs_delta_name": None,
                f"{prefix}_max_abs_delta_rad": None,
                f"{prefix}_max_abs_delta_deg": None,
            }

        delta_cpu = delta.detach().cpu()
        local_max_index = int(torch.argmax(torch.abs(delta_cpu[arm_indices])).item())
        max_index = arm_indices[local_max_index]
        max_delta = float(torch.abs(delta_cpu[max_index]).item())
        return {
            f"{prefix}_max_abs_delta": max_delta,
            f"{prefix}_max_abs_delta_index": max_index,
            f"{prefix}_max_abs_delta_name": action_feature_names[max_index],
            f"{prefix}_max_abs_delta_rad": max_delta,
            f"{prefix}_max_abs_delta_deg": max_delta * 180.0 / 3.141592653589793,
        }

    def _direction_reversal_fields(
        self,
        *,
        timestep: int,
        old_actions: dict[int, torch.Tensor],
        merged_actions: dict[int, torch.Tensor],
        action_feature_names: list[str],
    ) -> dict[str, Any]:
        previous_timestep = timestep - 1
        if previous_timestep not in old_actions or previous_timestep not in merged_actions:
            return {
                "old_velocity_proxy": None,
                "merged_velocity_proxy": None,
                "direction_reversal_indices": [],
                "direction_reversal_names": [],
                "direction_reversal_count": 0,
            }

        old_velocity = old_actions[timestep] - old_actions[previous_timestep]
        merged_velocity = merged_actions[timestep] - merged_actions[previous_timestep]
        old_velocity_cpu = old_velocity.detach().cpu()
        merged_velocity_cpu = merged_velocity.detach().cpu()
        epsilon = self.config.direction_reversal_epsilon
        reversal_indices = []
        for index, name in enumerate(action_feature_names):
            if not (name.startswith("left_arm_") or name.startswith("right_arm_")):
                continue
            old_delta = float(old_velocity_cpu[index].item())
            merged_delta = float(merged_velocity_cpu[index].item())
            if abs(old_delta) > epsilon and abs(merged_delta) > epsilon and old_delta * merged_delta < 0:
                reversal_indices.append(index)

        return {
            "old_velocity_proxy": old_velocity_cpu.tolist(),
            "merged_velocity_proxy": merged_velocity_cpu.tolist(),
            "direction_reversal_indices": reversal_indices,
            "direction_reversal_names": [action_feature_names[index] for index in reversal_indices],
            "direction_reversal_count": len(reversal_indices),
        }

    def _weighted_aggregation_detail_record(
        self,
        *,
        transition_id: int,
        timestep: int,
        region: str,
        old_action: torch.Tensor,
        incoming_action: torch.Tensor,
        merged_action: torch.Tensor,
        old_weight: float | None,
        incoming_weight: float | None,
        blend_index: int | None,
        blend_count: int,
        arm_crossfade_enabled: bool,
        arm_old_weight: float | None,
        arm_incoming_weight: float | None,
        arm_indices: list[int],
        arm_feature_names: list[str],
        first_blended_timestep: int | None,
        is_first_new_only_timestep: bool,
        old_actions: dict[int, torch.Tensor],
        merged_actions: dict[int, torch.Tensor],
        action_feature_names: list[str],
    ) -> dict[str, Any]:
        record = {
            "record_type": "timestep_detail",
            "transition_id": transition_id,
            "timestep": timestep,
            "aggregation_region": region,
            "aggregation_applied": region == "blended",
            "preservation_reason": "guard_steps" if region == "guard_preserved" else None,
            "is_first_blended_timestep": timestep == first_blended_timestep,
            "is_first_new_only_timestep": is_first_new_only_timestep,
            "blend_index": blend_index,
            "blend_count": blend_count,
            "arm_crossfade_enabled": arm_crossfade_enabled,
            "arm_crossfade_schedule": "linear" if arm_crossfade_enabled else None,
            "arm_old_weight": arm_old_weight,
            "arm_incoming_weight": arm_incoming_weight,
            "non_arm_old_weight": old_weight,
            "non_arm_incoming_weight": incoming_weight,
            "arm_indices": arm_indices,
            "arm_feature_names": arm_feature_names,
            "old_action": old_action.detach().cpu().tolist(),
            "incoming_action": incoming_action.detach().cpu().tolist(),
            # This is the exact tensor inserted into future_action_queue.
            "merged_action": merged_action.detach().cpu().tolist(),
            "old_weight": old_weight,
            "incoming_weight": incoming_weight,
            "raw_old_weight": old_weight,
            "raw_incoming_weight": incoming_weight,
            "weight_normalization_applied": False,
            "normalized_old_weight": None,
            "normalized_incoming_weight": None,
            "contributing_sources": (
                ["old_queue", "incoming_chunk"]
                if region == "blended"
                else ["old_queue"]
            ),
            "contributing_weights": (
                [old_weight, incoming_weight] if region == "blended" else None
            ),
            "contributing_weights_by_group": (
                {
                    "arm": {
                        "old_weight": arm_old_weight,
                        "incoming_weight": arm_incoming_weight,
                    },
                    "non_arm": {
                        "old_weight": old_weight,
                        "incoming_weight": incoming_weight,
                    },
                }
                if region == "blended"
                else None
            ),
        }
        record.update(
            self._action_delta_fields(
                "old_vs_incoming", incoming_action - old_action, action_feature_names
            )
        )
        record.update(
            self._action_delta_fields(
                "old_vs_merged", merged_action - old_action, action_feature_names
            )
        )
        record.update(
            self._action_delta_fields(
                "incoming_vs_merged", merged_action - incoming_action, action_feature_names
            )
        )
        record.update(
            self._arm_delta_fields(
                "arm_old_incoming",
                incoming_action - old_action,
                arm_indices,
                action_feature_names,
            )
        )
        record.update(
            self._arm_delta_fields(
                "arm_old_merged",
                merged_action - old_action,
                arm_indices,
                action_feature_names,
            )
        )
        record.update(
            self._direction_reversal_fields(
                timestep=timestep,
                old_actions=old_actions,
                merged_actions=merged_actions,
                action_feature_names=action_feature_names,
            )
        )
        return record

    def _log_weighted_aggregation_transition(
        self,
        *,
        transition_id: int,
        latest_action: int,
        incoming_source_timestep: int | None,
        source_prefetch_request_id: int | None,
        old_actions: dict[int, torch.Tensor],
        incoming_actions: dict[int, torch.Tensor],
        merged_actions: dict[int, torch.Tensor],
        blended_results: dict[int, dict[str, Any]],
        aggregation_weights: tuple[float, float] | None,
        previous_executed_action: list[float] | None,
    ) -> None:
        if not self.config.debug_weighted_aggregation:
            return

        action_feature_names = list(self.robot.action_features)
        arm_indices, arm_feature_names = self._arm_action_metadata(action_feature_names)
        old_timesteps = sorted(old_actions)
        incoming_timesteps = sorted(incoming_actions)
        merged_timesteps = sorted(merged_actions)
        overlap_timesteps = sorted(set(old_timesteps) & set(incoming_timesteps))
        guard_until = latest_action + 5
        guard_preserved_timesteps = [
            timestep
            for timestep in old_timesteps
            if latest_action < timestep <= guard_until and timestep in merged_actions
        ]
        guard_preserved_overlap_timesteps = sorted(
            set(guard_preserved_timesteps) & set(incoming_timesteps)
        )
        blended_timesteps = sorted(blended_results)
        first_blended_timestep = blended_timesteps[0] if blended_timesteps else None
        last_blended_timestep = blended_timesteps[-1] if blended_timesteps else None
        new_only_timesteps = sorted(
            (set(incoming_timesteps) - set(old_timesteps)) & set(merged_timesteps)
        )
        first_new_only_timestep = next(
            (
                timestep
                for timestep in new_only_timesteps
                if last_blended_timestep is not None and timestep > last_blended_timestep
            ),
            None,
        )
        old_weight, incoming_weight = aggregation_weights or (None, None)

        header = {
            "record_type": "transition_header",
            "transition_id": transition_id,
            "wall_time": time.time(),
            "monotonic_time": time.perf_counter(),
            "latest_action": latest_action,
            "incoming_source_timestep": incoming_source_timestep,
            "prefetch_request_id": source_prefetch_request_id,
            "aggregate_fn_name": self.config.aggregate_fn_name,
            "actual_aggregation_weights": (
                {"old_weight": old_weight, "incoming_weight": incoming_weight}
                if aggregation_weights is not None
                else None
            ),
            "non_arm_aggregation_weights": (
                {"old_weight": old_weight, "incoming_weight": incoming_weight}
                if aggregation_weights is not None
                else None
            ),
            "weight_schedule": (
                "constant_per_blended_timestep" if aggregation_weights is not None else None
            ),
            "arm_temporal_crossfade": self.config.arm_temporal_crossfade,
            "arm_crossfade_schedule": (
                "linear" if self.config.arm_temporal_crossfade else None
            ),
            "arm_indices": arm_indices,
            "arm_feature_names": arm_feature_names,
            "non_arm_feature_names": [
                name for index, name in enumerate(action_feature_names) if index not in arm_indices
            ],
            "guard_steps": 5,
            "old_queue_size": len(old_timesteps),
            "incoming_chunk_size": len(incoming_timesteps),
            "new_queue_size": len(merged_timesteps),
            "old_queue_first_timestep": old_timesteps[0] if old_timesteps else None,
            "old_queue_last_timestep": old_timesteps[-1] if old_timesteps else None,
            "incoming_first_timestep": incoming_timesteps[0] if incoming_timesteps else None,
            "incoming_last_timestep": incoming_timesteps[-1] if incoming_timesteps else None,
            "overlap_first_timestep": overlap_timesteps[0] if overlap_timesteps else None,
            "overlap_last_timestep": overlap_timesteps[-1] if overlap_timesteps else None,
            "overlap_count": len(overlap_timesteps),
            "guard_preserved_timesteps": guard_preserved_timesteps,
            "guard_preserved_overlap_timesteps": guard_preserved_overlap_timesteps,
            "blended_timesteps": blended_timesteps,
            "first_blended_timestep": first_blended_timestep,
            "last_blended_timestep": last_blended_timestep,
            "first_new_only_timestep": first_new_only_timestep,
            "action_feature_names": action_feature_names,
            "direction_reversal_epsilon": self.config.direction_reversal_epsilon,
        }

        detail_records = []
        for timestep in guard_preserved_overlap_timesteps:
            detail_records.append(
                self._weighted_aggregation_detail_record(
                    transition_id=transition_id,
                    timestep=timestep,
                    region="guard_preserved",
                    old_action=old_actions[timestep],
                    incoming_action=incoming_actions[timestep],
                    merged_action=merged_actions[timestep],
                    old_weight=None,
                    incoming_weight=None,
                    blend_index=None,
                    blend_count=len(blended_timesteps),
                    arm_crossfade_enabled=False,
                    arm_old_weight=None,
                    arm_incoming_weight=None,
                    arm_indices=arm_indices,
                    arm_feature_names=arm_feature_names,
                    first_blended_timestep=first_blended_timestep,
                    is_first_new_only_timestep=False,
                    old_actions=old_actions,
                    merged_actions=merged_actions,
                    action_feature_names=action_feature_names,
                )
            )
        for timestep in blended_timesteps:
            blend_result = blended_results[timestep]
            detail_records.append(
                self._weighted_aggregation_detail_record(
                    transition_id=transition_id,
                    timestep=timestep,
                    region="blended",
                    old_action=blend_result["old_action"],
                    incoming_action=blend_result["incoming_action"],
                    merged_action=blend_result["merged_action"],
                    old_weight=old_weight,
                    incoming_weight=incoming_weight,
                    blend_index=blend_result["blend_index"],
                    blend_count=blend_result["blend_count"],
                    arm_crossfade_enabled=blend_result["arm_crossfade_enabled"],
                    arm_old_weight=blend_result["arm_old_weight"],
                    arm_incoming_weight=blend_result["arm_incoming_weight"],
                    arm_indices=arm_indices,
                    arm_feature_names=arm_feature_names,
                    first_blended_timestep=first_blended_timestep,
                    is_first_new_only_timestep=False,
                    old_actions=old_actions,
                    merged_actions=merged_actions,
                    action_feature_names=action_feature_names,
                )
            )

        if first_new_only_timestep is not None:
            first_new_only_action = merged_actions[first_new_only_timestep]
            detail_records.append(
                {
                    "record_type": "timestep_detail",
                    "transition_id": transition_id,
                    "timestep": first_new_only_timestep,
                    "aggregation_region": "new_only",
                    "aggregation_applied": False,
                    "preservation_reason": "incoming_only",
                    "is_first_blended_timestep": False,
                    "is_first_new_only_timestep": True,
                    "blend_index": None,
                    "blend_count": len(blended_timesteps),
                    "arm_crossfade_enabled": False,
                    "arm_crossfade_schedule": None,
                    "arm_old_weight": 0.0,
                    "arm_incoming_weight": 1.0,
                    "non_arm_old_weight": 0.0,
                    "non_arm_incoming_weight": 1.0,
                    "arm_indices": arm_indices,
                    "arm_feature_names": arm_feature_names,
                    "old_action": None,
                    "incoming_action": incoming_actions[first_new_only_timestep]
                    .detach()
                    .cpu()
                    .tolist(),
                    "merged_action": first_new_only_action.detach().cpu().tolist(),
                }
            )

        def maximum_detail(field: str) -> dict[str, Any] | None:
            candidates = [
                record for record in detail_records if record.get(field) is not None
            ]
            return max(candidates, key=lambda record: record[field]) if candidates else None

        max_old_incoming = maximum_detail("old_incoming_max_abs_delta")
        max_old_merged = maximum_detail("old_merged_max_abs_delta")
        first_blend = next(
            (record for record in detail_records if record["is_first_blended_timestep"]),
            None,
        )
        last_blend = next(
            (
                record
                for record in detail_records
                if record.get("timestep") == last_blended_timestep
                and record.get("aggregation_region") == "blended"
            ),
            None,
        )

        blend_to_new_only = {
            "last_blended_action": last_blend["merged_action"] if last_blend else None,
            "first_new_only_action": (
                merged_actions[first_new_only_timestep].detach().cpu().tolist()
                if first_new_only_timestep is not None
                else None
            ),
            "action_delta_last_blend_to_new_only": None,
            "last_blend_to_new_only_arm_max_delta": None,
            "last_blend_to_new_only_arm_max_abs_delta": None,
            "last_blend_to_new_only_arm_max_delta_index": None,
            "last_blend_to_new_only_arm_max_delta_name": None,
            "last_blend_to_new_only_arm_max_delta_rad": None,
            "last_blend_to_new_only_arm_max_delta_deg": None,
        }
        if last_blended_timestep is not None and first_new_only_timestep is not None:
            boundary_delta = (
                merged_actions[first_new_only_timestep]
                - merged_actions[last_blended_timestep]
            )
            blend_to_new_only.update(
                {
                    "action_delta_last_blend_to_new_only": boundary_delta
                    .detach()
                    .cpu()
                    .tolist(),
                    **self._arm_delta_fields(
                        "last_blend_to_new_only_arm",
                        boundary_delta,
                        arm_indices,
                        action_feature_names,
                    ),
                }
            )
            blend_to_new_only["last_blend_to_new_only_arm_max_delta"] = (
                blend_to_new_only["last_blend_to_new_only_arm_max_abs_delta"]
            )

        next_timestep = merged_timesteps[0] if merged_timesteps else None
        next_merged_action = merged_actions.get(next_timestep)
        executed_to_merged = {}
        if previous_executed_action is not None and next_merged_action is not None:
            previous_tensor = torch.as_tensor(
                previous_executed_action,
                dtype=next_merged_action.dtype,
                device=next_merged_action.device,
            )
            executed_to_merged = {
                "previous_executed_action": previous_executed_action,
                "next_merged_timestep": next_timestep,
                "next_merged_action": next_merged_action.detach().cpu().tolist(),
                **self._action_delta_fields(
                    "executed_to_merged",
                    next_merged_action - previous_tensor,
                    action_feature_names,
                ),
            }
        else:
            executed_to_merged = {
                "previous_executed_action": previous_executed_action,
                "next_merged_timestep": next_timestep,
                "next_merged_action": (
                    next_merged_action.detach().cpu().tolist()
                    if next_merged_action is not None
                    else None
                ),
                "action_delta_executed_to_merged": None,
                "executed_to_merged_max_abs_delta": None,
                "executed_to_merged_max_abs_delta_index": None,
                "executed_to_merged_max_abs_delta_name": None,
                "executed_to_merged_max_abs_delta_rad": None,
                "executed_to_merged_max_abs_delta_deg": None,
            }

        summary = {
            **header,
            "record_type": "transition_summary",
            "guard_preserved_count": len(guard_preserved_timesteps),
            "blended_count": len(blended_timesteps),
            "max_old_incoming_delta": (
                max_old_incoming["old_incoming_max_abs_delta"]
                if max_old_incoming
                else None
            ),
            "max_old_incoming_delta_timestep": (
                max_old_incoming["timestep"] if max_old_incoming else None
            ),
            "max_old_incoming_delta_joint": (
                max_old_incoming["old_incoming_max_abs_delta_name"]
                if max_old_incoming
                else None
            ),
            "max_old_merged_delta": (
                max_old_merged["old_merged_max_abs_delta"] if max_old_merged else None
            ),
            "max_old_merged_delta_timestep": (
                max_old_merged["timestep"] if max_old_merged else None
            ),
            "max_old_merged_delta_joint": (
                max_old_merged["old_merged_max_abs_delta_name"]
                if max_old_merged
                else None
            ),
            "first_blend_old_weight": first_blend["old_weight"] if first_blend else None,
            "first_blend_incoming_weight": (
                first_blend["incoming_weight"] if first_blend else None
            ),
            "first_blend_old_incoming_max_delta": (
                first_blend["old_incoming_max_abs_delta"] if first_blend else None
            ),
            "first_blend_old_merged_max_delta": (
                first_blend["old_merged_max_abs_delta"] if first_blend else None
            ),
            "first_blend_direction_reversal_count": (
                first_blend["direction_reversal_count"] if first_blend else 0
            ),
            "first_blend_arm_old_incoming_max_delta": (
                first_blend["arm_old_incoming_max_abs_delta"] if first_blend else None
            ),
            "first_blend_arm_old_merged_max_delta": (
                first_blend["arm_old_merged_max_abs_delta"] if first_blend else None
            ),
            "last_blend_arm_old_merged_max_delta": (
                last_blend["arm_old_merged_max_abs_delta"] if last_blend else None
            ),
            **blend_to_new_only,
            **executed_to_merged,
            "debug_event_drop_count": self._weighted_aggregation_dropped_transitions,
        }

        try:
            assert self._weighted_aggregation_log_queue is not None
            self._weighted_aggregation_log_queue.put_nowait([header, *detail_records, summary])
        except Full:
            self._weighted_aggregation_dropped_transitions += 1

    @staticmethod
    def _internal_gap_details(timesteps: list[int]) -> tuple[list[int], list[dict[str, Any]]]:
        missing_timesteps = []
        gap_ranges = []
        for previous_timestep, next_timestep in zip(timesteps, timesteps[1:]):
            if next_timestep <= previous_timestep + 1:
                continue
            missing = list(range(previous_timestep + 1, next_timestep))
            missing_timesteps.extend(missing)
            gap_ranges.append(
                {
                    "previous_timestep": previous_timestep,
                    "next_timestep": next_timestep,
                    "gap_size": len(missing),
                    "missing_timesteps": missing,
                }
            )
        return missing_timesteps, gap_ranges

    @classmethod
    def _build_chunk_transition_record(
        cls,
        *,
        latest_action: int,
        old_timesteps: list[int],
        incoming_timesteps: list[int],
        updated_timesteps: list[int],
        guard_steps: int = 5,
    ) -> dict[str, Any]:
        """Build timestep-only diagnostics without changing queue contents or actions."""
        old_timesteps = list(old_timesteps)
        incoming_timesteps = list(incoming_timesteps)
        updated_timesteps = list(updated_timesteps)
        old_set = set(old_timesteps)
        incoming_set = set(incoming_timesteps)
        updated_set = set(updated_timesteps)

        overlap_timesteps = sorted(old_set & incoming_set)
        old_only_future_timesteps = sorted(
            timestep
            for timestep in old_set - incoming_set
            if timestep > latest_action
        )
        incoming_only_timesteps = sorted(incoming_set - old_set)
        stale_incoming_timesteps = sorted(
            timestep for timestep in incoming_set if timestep <= latest_action
        )

        guard_until = latest_action + guard_steps
        incoming_guard_filtered_timesteps = sorted(
            timestep
            for timestep in incoming_set
            if latest_action < timestep <= guard_until
        )
        old_guard_preserved_timesteps = sorted(
            timestep
            for timestep in old_set & updated_set
            if latest_action < timestep <= guard_until
        )
        old_guard_timesteps = sorted(
            timestep
            for timestep in old_set
            if latest_action < timestep <= guard_until
        )
        old_future_dropped_timesteps = sorted(
            timestep
            for timestep in old_set - updated_set
            if timestep > latest_action
        )
        incoming_dropped_timesteps = sorted(incoming_set - updated_set)

        expected_next = latest_action + 1
        actual_next = updated_timesteps[0] if updated_timesteps else None
        gap_detected = actual_next != expected_next
        missing_timesteps = (
            list(range(expected_next, actual_next))
            if actual_next is not None and actual_next > expected_next
            else []
        )

        if not gap_detected:
            gap_origin = None
        elif not old_timesteps:
            gap_origin = "queue_exhausted_before_chunk_arrival"
        elif expected_next in old_set and expected_next not in updated_set:
            gap_origin = "merge_removed_expected_old_action"
        elif expected_next not in old_set:
            gap_origin = "gap_already_present_before_merge"
        else:
            gap_origin = "unknown"

        internal_missing_timesteps, internal_gap_ranges = cls._internal_gap_details(
            updated_timesteps
        )
        internal_missing_set = set(internal_missing_timesteps)
        internal_missing_present_in_old_queue = sorted(internal_missing_set & old_set)
        internal_missing_absent_from_old_queue = sorted(internal_missing_set - old_set)
        internal_missing_guard_filtered_timesteps = sorted(
            internal_missing_set & set(incoming_guard_filtered_timesteps)
        )
        internal_missing_old_guard_preserved_timesteps = sorted(
            internal_missing_set & set(old_guard_preserved_timesteps)
        )
        internal_missing_merge_dropped_old_timesteps = sorted(
            internal_missing_set & set(old_future_dropped_timesteps)
        )
        internal_missing_guard_filtered_without_old_backup = sorted(
            internal_missing_set
            & set(incoming_guard_filtered_timesteps)
            - old_set
        )
        internal_missing_absent_from_both_queues = sorted(
            internal_missing_set - old_set - incoming_set
        )
        classified_internal_missing = (
            set(internal_missing_guard_filtered_without_old_backup)
            | set(internal_missing_merge_dropped_old_timesteps)
            | set(internal_missing_absent_from_both_queues)
        )
        internal_missing_unclassified_timesteps = sorted(
            internal_missing_set - classified_internal_missing
        )

        internal_gap_origin_components = []
        if internal_missing_guard_filtered_without_old_backup:
            internal_gap_origin_components.append("guard_filtered_without_old_backup")
        if internal_missing_merge_dropped_old_timesteps:
            internal_gap_origin_components.append("merge_removed_old_future_action")
        if internal_missing_absent_from_both_queues:
            internal_gap_origin_components.append("gap_already_present_before_merge")
        if internal_missing_unclassified_timesteps:
            internal_gap_origin_components.append("unknown")

        if not internal_gap_ranges:
            internal_gap_origin = None
        elif len(internal_gap_origin_components) == 1:
            internal_gap_origin = internal_gap_origin_components[0]
        elif len(internal_gap_origin_components) > 1:
            internal_gap_origin = "mixed"
        else:
            internal_gap_origin = "unknown"

        return {
            "latest_action": latest_action,
            "guard_steps": guard_steps,
            "guard_until": guard_until,
            "old_queue_was_empty": not old_timesteps,
            "old_queue_size": len(old_timesteps),
            "old_queue_first": old_timesteps[0] if old_timesteps else None,
            "old_queue_last": old_timesteps[-1] if old_timesteps else None,
            "old_queue_timesteps": old_timesteps,
            "incoming_size": len(incoming_timesteps),
            "incoming_chunk_size": len(incoming_timesteps),
            "incoming_first": incoming_timesteps[0] if incoming_timesteps else None,
            "incoming_last": incoming_timesteps[-1] if incoming_timesteps else None,
            "incoming_timesteps": incoming_timesteps,
            "overlap_first": overlap_timesteps[0] if overlap_timesteps else None,
            "overlap_last": overlap_timesteps[-1] if overlap_timesteps else None,
            "overlap_count": len(overlap_timesteps),
            "overlap_timesteps": overlap_timesteps,
            "old_only_future_timesteps": old_only_future_timesteps,
            "incoming_only_timesteps": incoming_only_timesteps,
            "stale_incoming_timesteps": stale_incoming_timesteps,
            "incoming_guard_filtered_timesteps": incoming_guard_filtered_timesteps,
            "old_guard_timesteps": old_guard_timesteps,
            "old_guard_preserved_timesteps": old_guard_preserved_timesteps,
            "all_available_old_guard_timesteps_preserved": (
                old_guard_preserved_timesteps == old_guard_timesteps
            ),
            "old_future_dropped_timesteps": old_future_dropped_timesteps,
            "incoming_dropped_timesteps": incoming_dropped_timesteps,
            "new_queue_size": len(updated_timesteps),
            "updated_queue_size": len(updated_timesteps),
            "new_queue_first": actual_next,
            "new_queue_last": updated_timesteps[-1] if updated_timesteps else None,
            "new_queue_timesteps": updated_timesteps,
            "updated_queue_timesteps": updated_timesteps,
            "expected_next": expected_next,
            "actual_next": actual_next,
            "continuity_ok": not gap_detected,
            "continuity_warning": gap_detected,
            "warning": "[CHUNK GAP WARNING]" if gap_detected else None,
            "gap_detected": gap_detected,
            "gap_size": len(missing_timesteps),
            "missing_timesteps": missing_timesteps,
            "gap_origin": gap_origin,
            "expected_next_present_in_old_queue": expected_next in old_set,
            "internal_gap_detected": bool(internal_gap_ranges),
            "internal_gap_count": len(internal_gap_ranges),
            "internal_missing_timesteps": internal_missing_timesteps,
            "internal_gap_ranges": internal_gap_ranges,
            "internal_warning": (
                "[CHUNK INTERNAL GAP WARNING]" if internal_gap_ranges else None
            ),
            "internal_missing_present_in_old_queue": internal_missing_present_in_old_queue,
            "internal_missing_absent_from_old_queue": internal_missing_absent_from_old_queue,
            "internal_missing_guard_filtered_timesteps": (
                internal_missing_guard_filtered_timesteps
            ),
            "internal_missing_old_guard_preserved_timesteps": (
                internal_missing_old_guard_preserved_timesteps
            ),
            "internal_missing_merge_dropped_old_timesteps": (
                internal_missing_merge_dropped_old_timesteps
            ),
            "internal_missing_guard_filtered_without_old_backup": (
                internal_missing_guard_filtered_without_old_backup
            ),
            "internal_missing_absent_from_both_queues": (
                internal_missing_absent_from_both_queues
            ),
            "internal_missing_unclassified_timesteps": (
                internal_missing_unclassified_timesteps
            ),
            "internal_gap_origin": internal_gap_origin,
            "internal_gap_origin_components": internal_gap_origin_components,
            "any_continuity_gap": gap_detected or bool(internal_gap_ranges),
        }

    def _log_chunk_transition(
        self,
        *,
        transition_id: int,
        latest_action: int,
        old_timesteps: list[int],
        incoming_timesteps: list[int],
        updated_timesteps: list[int],
        source_prefetch_request_id: int | None,
        queue_update_time_ms: float,
    ) -> None:
        if not self.config.debug_chunk_transitions:
            return

        transition_start = time.perf_counter()
        record = {
            "event": "action_chunk_transition",
            "wall_time": time.time(),
            "monotonic_time": time.perf_counter(),
            "transition_id": transition_id,
            "backend": self.backend,
            "configured_client_fps": self.config.fps,
            "recent_actual_action_fps": (
                1.0 / (sum(self._recent_action_periods) / len(self._recent_action_periods))
                if self._recent_action_periods
                else None
            ),
            "aggregate_fn_name": self.config.aggregate_fn_name,
            **self._build_chunk_transition_record(
                latest_action=latest_action,
                old_timesteps=old_timesteps,
                incoming_timesteps=incoming_timesteps,
                updated_timesteps=updated_timesteps,
            ),
            "source_prefetch_request_id": source_prefetch_request_id,
            "queue_update_time_ms": queue_update_time_ms,
            **self._prefetch_state_snapshot(),
        }
        record["transition_processing_time_ms"] = (
            time.perf_counter() - transition_start
        ) * 1000
        assert self._chunk_transition_log_queue is not None
        self._chunk_transition_log_queue.put_nowait(record)

    def _consume_prefetch_trigger(self, observation_timestep: int) -> int | None:
        with self.prefetch_state_lock:
            request_id = self._prefetch_trigger_request_id
            if request_id is None or request_id != self.pending_prefetch_request_id:
                return None
            self._prefetch_trigger_request_id = None
            self.pending_prefetch_observation_timestep = observation_timestep
            return request_id

    def _mark_prefetch_sent(self, request_id: int, observation_timestep: int) -> None:
        sent_time = time.perf_counter()
        with self.prefetch_state_lock:
            if not (
                self.prefetch_request_pending
                and self.pending_prefetch_request_id == request_id
            ):
                return
            self.pending_prefetch_observation_timestep = observation_timestep
            self.prefetch_request_sent_monotonic_time = sent_time

        queue_size, queue_ratio = self._action_queue_ratio()
        self._log_prefetch_event(
            "prefetch_sent",
            timestep=observation_timestep,
            queue_size=queue_size,
            queue_ratio=queue_ratio,
            request_id=request_id,
        )

    def _rollback_prefetch_request(
        self,
        request_id: int,
        observation_timestep: int,
        reason: str,
    ) -> None:
        with self.prefetch_state_lock:
            if self.pending_prefetch_request_id != request_id:
                return
            self.prefetch_request_pending = False
            self.pending_prefetch_request_id = None
            self.pending_prefetch_observation_timestep = None
            self.prefetch_request_sent_monotonic_time = None
            self._prefetch_trigger_request_id = None
            self._previous_action_queue_ratio = self._chunk_size_threshold + 1.0
            self._prefetch_pending_skip_logged = False

        queue_size, queue_ratio = self._action_queue_ratio()
        self._log_prefetch_event(
            "prefetch_pending_cleared",
            timestep=observation_timestep,
            queue_size=queue_size,
            queue_ratio=queue_ratio,
            request_id=request_id,
            clear_reason=reason,
        )

    def _handle_prefetch_acknowledgement(
        self,
        source_prefetch_request_id: int | None,
        source_timestep: int | None,
    ) -> None:
        if source_prefetch_request_id is None:
            return

        queue_size, queue_ratio = self._action_queue_ratio()
        self._log_prefetch_event(
            "prefetch_response_received",
            timestep=source_timestep,
            queue_size=queue_size,
            queue_ratio=queue_ratio,
            request_id=source_prefetch_request_id,
        )

        with self.prefetch_state_lock:
            pending_request_id = self.pending_prefetch_request_id
            if not (
                self.prefetch_request_pending
                and pending_request_id == source_prefetch_request_id
            ):
                matched = False
            else:
                matched = True
                self.prefetch_request_pending = False
                self.pending_prefetch_request_id = None
                self.pending_prefetch_observation_timestep = None
                self.prefetch_request_sent_monotonic_time = None
                self._prefetch_trigger_request_id = None
                self._prefetch_pending_skip_logged = False

        if matched:
            self._log_prefetch_event(
                "prefetch_pending_cleared",
                timestep=source_timestep,
                queue_size=queue_size,
                queue_ratio=queue_ratio,
                request_id=source_prefetch_request_id,
                clear_reason="matching_response",
            )
        else:
            self._log_prefetch_event(
                "prefetch_duplicate_response",
                timestep=source_timestep,
                queue_size=queue_size,
                queue_ratio=queue_ratio,
                request_id=source_prefetch_request_id,
                pending_request_id=pending_request_id,
            )

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.uses_grpc_backend:
            raise RuntimeError("send_observation is only valid for the gRPC backend")
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            grpc_send_start = time.perf_counter()
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            grpc_send_time = time.perf_counter() - grpc_send_start
            self._last_send_observation_diagnostics = {
                "payload_bytes": len(observation_bytes),
                "pickle_serialize_ms": serialize_time * 1000,
                "grpc_send_ms": grpc_send_time * 1000,
                "total_ms": (serialize_time + grpc_send_time) * 1000,
                "success": True,
            }
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self._last_send_observation_diagnostics = {
                "payload_bytes": len(observation_bytes),
                "pickle_serialize_ms": serialize_time * 1000,
                "grpc_send_ms": (time.perf_counter() - grpc_send_start) * 1000,
                "total_ms": (time.perf_counter() - start_time) * 1000,
                "success": False,
            }
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _resize_observation_images_for_transport(
        self,
        raw_observation: RawObservation,
    ) -> tuple[RawObservation, dict[str, float], dict[str, dict[str, Any]]]:
        """Resize only configured camera frames, preserving aspect ratio and all non-image values."""
        scale = self.config.image_resize_scale
        if scale == 1.0:
            return raw_observation, {}, {}

        transport_observation = dict(raw_observation)
        resize_timings_ms = {}
        image_transport = {}
        for camera_key in self.transport_image_keys:
            if camera_key not in raw_observation:
                continue
            image = raw_observation[camera_key]
            resize_start = time.perf_counter()
            resized_image = resize_image_by_scale(image, scale)
            resize_timings_ms[camera_key] = (time.perf_counter() - resize_start) * 1000
            transport_observation[camera_key] = resized_image
            image_transport[camera_key] = {
                "original_shape": list(image.shape),
                "transport_shape": list(resized_image.shape),
                "original_bytes": int(image.nbytes),
                "transport_bytes": int(resized_image.nbytes),
            }

        return transport_observation, resize_timings_ms, image_transport

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        *,
        transition_id: int | None = None,
        source_prefetch_request_id: int | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2


        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        debug_weighted_aggregation = self.config.debug_weighted_aggregation
        arm_temporal_crossfade = self.config.arm_temporal_crossfade
        arm_indices = (
            self._arm_action_metadata(list(self.robot.action_features))[0]
            if arm_temporal_crossfade
            else []
        )
        incoming_action_queue = (
            {action.get_timestep(): action.get_action() for action in incoming_actions}
            if debug_weighted_aggregation
            else None
        )
        blended_results = {} if debug_weighted_aggregation else None
        aggregation_weights = (
            getattr(aggregate_fn, "aggregation_weights", None)
            if debug_weighted_aggregation
            else None
        )

        with self.latest_action_lock:
            latest_action = self.latest_action
            previous_executed_action = (
                list(self.latest_executed_action)
                if debug_weighted_aggregation and self.latest_executed_action is not None
                else None
            )


        ##############################################################
        ######  새로 오는 action들 처음 N개 버리고 그 뒤부터 합치는 방식 ######

        guard_steps = 5
        guard_until = latest_action + guard_steps
        track_blend_metadata = arm_temporal_crossfade or debug_weighted_aggregation
        blended_timesteps = (
            sorted(
                {
                    action.get_timestep()
                    for action in incoming_actions
                    if action.get_timestep() > guard_until
                    and action.get_timestep() in current_action_queue
                }
            )
            if track_blend_metadata
            else []
        )
        blend_count = len(blended_timesteps)
        blend_index_by_timestep = {
            timestep: index
            for index, timestep in enumerate(blended_timesteps, start=1)
        }
        # old 2개 더 남김
        for old_action in internal_queue:
            if latest_action < old_action.get_timestep() <= guard_until:
                future_action_queue.put(old_action)
        ##############################################################

        for new_action in incoming_actions:
            # with self.latest_action_lock:
            #     latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            if new_action.get_timestep() <= guard_until:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            old_action_tensor = current_action_queue[new_action.get_timestep()]
            incoming_action_tensor = new_action.get_action()
            merged_action_tensor = aggregate_fn(old_action_tensor, incoming_action_tensor)
            blend_index = blend_index_by_timestep.get(new_action.get_timestep())
            arm_crossfade_applied = arm_temporal_crossfade and bool(arm_indices)
            if arm_crossfade_applied:
                assert blend_index is not None
                arm_incoming_weight = blend_index / (blend_count + 1)
                arm_old_weight = 1.0 - arm_incoming_weight
                # Preserve the exact existing aggregate result for every non-arm
                # dimension, then overwrite only identified arm-joint dimensions.
                merged_action_tensor = merged_action_tensor.clone()
                merged_action_tensor[arm_indices] = (
                    arm_old_weight * old_action_tensor[arm_indices]
                    + arm_incoming_weight * incoming_action_tensor[arm_indices]
                )
            else:
                if aggregation_weights is not None:
                    arm_old_weight, arm_incoming_weight = aggregation_weights
                else:
                    arm_old_weight, arm_incoming_weight = None, None
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=merged_action_tensor,
                    source_prefetch_request_id=getattr(
                        new_action, "source_prefetch_request_id", None
                    ),
                )
            )
            if blended_results is not None:
                # Capture references to the exact operands and result used above. No second
                # aggregation is performed for logging.
                blended_results[new_action.get_timestep()] = {
                    "old_action": old_action_tensor,
                    "incoming_action": incoming_action_tensor,
                    "merged_action": merged_action_tensor,
                    "blend_index": blend_index,
                    "blend_count": blend_count,
                    "arm_crossfade_enabled": arm_crossfade_applied,
                    "arm_old_weight": arm_old_weight,
                    "arm_incoming_weight": arm_incoming_weight,
                }

        with self.action_queue_lock:
            self.action_queue = future_action_queue

        if debug_weighted_aggregation:
            assert incoming_action_queue is not None
            assert blended_results is not None
            if transition_id is None:
                transition_id = self._next_chunk_transition_id()
            merged_action_queue = {
                action.get_timestep(): action.get_action()
                for action in future_action_queue.queue
            }
            try:
                self._log_weighted_aggregation_transition(
                    transition_id=transition_id,
                    latest_action=latest_action,
                    incoming_source_timestep=(
                        incoming_actions[0].get_timestep() if incoming_actions else None
                    ),
                    source_prefetch_request_id=source_prefetch_request_id,
                    old_actions=current_action_queue,
                    incoming_actions=incoming_action_queue,
                    merged_actions=merged_action_queue,
                    blended_results=blended_results,
                    aggregation_weights=aggregation_weights,
                    previous_executed_action=previous_executed_action,
                )
            except Exception as error:
                # Instrumentation must never affect the already-completed queue update.
                self.logger.warning("Failed to build weighted aggregation diagnostics: %s", error)

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the gRPC policy server"""
        if not self.uses_grpc_backend:
            self.logger.debug("receive_actions called for non-gRPC backend; skipping")
            return
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start
                source_prefetch_request_ids = {
                    getattr(action, "source_prefetch_request_id", None)
                    for action in timed_actions
                    if getattr(action, "source_prefetch_request_id", None) is not None
                }
                if len(source_prefetch_request_ids) == 1:
                    source_prefetch_request_id = next(iter(source_prefetch_request_ids))
                else:
                    source_prefetch_request_id = None
                    if len(source_prefetch_request_ids) > 1:
                        self.logger.error(
                            "Received one action chunk with inconsistent prefetch request IDs: %s",
                            sorted(source_prefetch_request_ids),
                        )

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                with self.latest_action_lock:
                    latest_action = self.latest_action
                capture_transition = self.config.debug_chunk_transitions or verbose
                if capture_transition:
                    old_size, old_timesteps = self._inspect_action_queue()
                    incoming_timesteps = [action.get_timestep() for action in timed_actions]
                else:
                    old_size, old_timesteps, incoming_timesteps = 0, [], []

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    self.logger.debug(f"Current latest action: {latest_action}")

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                transition_id = (
                    self._next_chunk_transition_id()
                    if self.config.debug_chunk_transitions
                    or self.config.debug_weighted_aggregation
                    else None
                )
                start_time = time.perf_counter()
                self._aggregate_action_queues(
                    timed_actions,
                    self.config.aggregate_fn,
                    transition_id=transition_id,
                    source_prefetch_request_id=source_prefetch_request_id,
                )
                queue_update_time = time.perf_counter() - start_time

                source_timestep = timed_actions[0].get_timestep() if timed_actions else None
                self._handle_prefetch_acknowledgement(
                    source_prefetch_request_id,
                    source_timestep,
                )

                if capture_transition:
                    new_size, new_timesteps = self._inspect_action_queue()
                else:
                    new_size, new_timesteps = 0, []
                if self.config.debug_chunk_transitions:
                    assert transition_id is not None
                    self._log_chunk_transition(
                        transition_id=transition_id,
                        latest_action=latest_action,
                        old_timesteps=old_timesteps,
                        incoming_timesteps=incoming_timesteps,
                        updated_timesteps=new_timesteps,
                        source_prefetch_request_id=source_prefetch_request_id,
                        queue_update_time_ms=queue_update_time * 1000,
                    )

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[:1]}:{old_timesteps[-1:]} | "
                        f"Incoming action steps: {incoming_timesteps[:1]}:{incoming_timesteps[-1:]} | "
                        f"Updated action steps: {new_timesteps[:1]}:{new_timesteps[-1:]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        if self.config.debug_chunk_transitions:
            action_monotonic_time = time.perf_counter()
            if self._previous_action_monotonic_time is not None:
                self._recent_action_periods.append(
                    action_monotonic_time - self._previous_action_monotonic_time
                )
            self._previous_action_monotonic_time = action_monotonic_time

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
            # 2배속 소모를 원한다면...
            # timed_action = self.action_queue.get_nowait() 
        get_end = time.perf_counter() - get_start

        logging_actions = self._action_tensor_to_action_dict(timed_action.get_action())

        previous_executed_timestep = self.previous_executed_timestep
        current_executed_timestep = timed_action.get_timestep()
        timestep_delta = (
            current_executed_timestep - previous_executed_timestep
            if previous_executed_timestep is not None
            else None
        )
        self.previous_executed_timestep = current_executed_timestep

        with self.latest_action_lock:
            self.latest_action = current_executed_timestep
            if self.config.debug_weighted_aggregation:
                self.latest_executed_action = [
                    logging_actions[name] for name in self.robot.action_features
                ]

        _performed_action = self.robot.send_action(logging_actions)

        # Loggggging action to file
        prefetch_state = self._prefetch_state_snapshot()
        with self.observation_debug_lock:
            observation_transport = self._pending_observation_transport_debug
            self._pending_observation_transport_debug = None
        observation_timings = None
        if observation_transport is not None:
            transport_timings = observation_transport["timings_ms"]
            send_timings = transport_timings["send_observation"]
            observation_timings = {
                "robot_get_observation": transport_timings["robot_get_observation"],
                "image_resize": transport_timings["image_resize"],
                "send_observation": send_timings["total_ms"],
                "send_observation_breakdown": send_timings,
                "total": transport_timings["total"],
            }

        record = {
            "wall_time": time.time(),
            "policy_timestamp": timed_action.get_timestamp(),
            "timestep": timed_action.get_timestep(),
            "action": logging_actions,
            "source_prefetch_request_id": getattr(
                timed_action, "source_prefetch_request_id", None
            ),
            "previous_executed_timestep": previous_executed_timestep,
            "current_executed_timestep": current_executed_timestep,
            "timestep_delta": timestep_delta,
            "timestep_continuity_warning": (
                timestep_delta is not None and timestep_delta != 1
            ),
            "timings_ms": {"observation": observation_timings},
            "observation_transport": observation_transport,
            **prefetch_state,
        }

        with open(FINAL_ACTION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")       



        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        if not self.uses_grpc_backend:
            with self.action_queue_lock:
                return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

        queue_size, queue_ratio = self._action_queue_ratio()
        with self.latest_action_lock:
            latest_action = self.latest_action
        now = time.perf_counter()

        timed_out_request_id = None
        timed_out_duration_ms = None
        with self.prefetch_state_lock:
            if (
                self.prefetch_request_pending
                and self.prefetch_request_sent_monotonic_time is not None
                and now - self.prefetch_request_sent_monotonic_time
                >= self.config.prefetch_request_timeout
            ):
                timed_out_request_id = self.pending_prefetch_request_id
                timed_out_duration_ms = (
                    now - self.prefetch_request_sent_monotonic_time
                ) * 1000
                self.prefetch_request_pending = False
                self.pending_prefetch_request_id = None
                self.pending_prefetch_observation_timestep = None
                self.prefetch_request_sent_monotonic_time = None
                self._prefetch_trigger_request_id = None
                # Re-arm crossing so a non-empty queue retries once; an empty queue uses must_go.
                self._previous_action_queue_ratio = self._chunk_size_threshold + 1.0
                self._prefetch_pending_skip_logged = False

        if timed_out_request_id is not None:
            self._log_prefetch_event(
                "prefetch_timeout",
                timestep=max(latest_action, 0),
                queue_size=queue_size,
                queue_ratio=queue_ratio,
                request_id=timed_out_request_id,
                pending_duration_ms=timed_out_duration_ms,
            )
            self._log_prefetch_event(
                "prefetch_pending_cleared",
                timestep=max(latest_action, 0),
                queue_size=queue_size,
                queue_ratio=queue_ratio,
                request_id=timed_out_request_id,
                clear_reason="timeout",
            )

        triggered_request_id = None
        previous_queue_ratio = None
        log_pending_skip = False
        skipped_pending_request_id = None
        with self.action_queue_lock:
            queue_empty = self.action_queue.empty()

        with self.prefetch_state_lock:
            previous_queue_ratio = self._previous_action_queue_ratio
            empty_queue_fallback = queue_empty

            if self.prefetch_request_pending:
                ready = False
                if queue_ratio <= self._chunk_size_threshold and not self._prefetch_pending_skip_logged:
                    self._prefetch_pending_skip_logged = True
                    log_pending_skip = True
                    skipped_pending_request_id = self.pending_prefetch_request_id
            elif empty_queue_fallback:
                # Preserve startup/emergency retries. The observation sender marks the
                # first empty-queue request as must_go and clears it after a successful send.
                ready = True
            else:
                threshold_crossed = (
                    previous_queue_ratio is not None
                    and previous_queue_ratio > self._chunk_size_threshold
                    and queue_ratio <= self._chunk_size_threshold
                )
                ready = threshold_crossed
                if threshold_crossed:
                    self._prefetch_request_sequence += 1
                    triggered_request_id = self._prefetch_request_sequence
                    self.prefetch_request_pending = True
                    self.pending_prefetch_request_id = triggered_request_id
                    self.pending_prefetch_observation_timestep = max(latest_action, 0)
                    self.prefetch_request_sent_monotonic_time = now
                    self._prefetch_trigger_request_id = triggered_request_id
                    self._prefetch_pending_skip_logged = False

            self._previous_action_queue_ratio = queue_ratio

        if log_pending_skip:
            self._log_prefetch_event(
                "prefetch_skipped_already_pending",
                timestep=max(latest_action, 0),
                queue_size=queue_size,
                queue_ratio=queue_ratio,
                request_id=skipped_pending_request_id,
            )

        if triggered_request_id is not None:
            self._log_prefetch_event(
                "prefetch_triggered",
                timestep=max(latest_action, 0),
                queue_size=queue_size,
                queue_ratio=queue_ratio,
                request_id=triggered_request_id,
                previous_queue_ratio=previous_queue_ratio,
                threshold=self._chunk_size_threshold,
            )

        return ready

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        prefetch_request_id = None
        observation = None
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            capture_start = time.perf_counter()
            raw_observation: RawObservation = self.robot.get_observation()
            capture_time_ms = (time.perf_counter() - capture_start) * 1000
            transport_observation, image_resize_ms, image_transport = (
                self._resize_observation_images_for_transport(raw_observation)
            )
            raw_observation["task"] = task
            if transport_observation is not raw_observation:
                transport_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=transport_observation,
                timestep=max(latest_action, 0),
            )
            prefetch_request_id = self._consume_prefetch_trigger(observation.get_timestep())
            if prefetch_request_id is not None:
                observation.prefetch_requested = True
                observation.prefetch_request_id = prefetch_request_id

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            observation_sent = self.send_observation(observation)
            with self.observation_debug_lock:
                self._pending_observation_transport_debug = {
                    "observation_timestamp": observation.get_timestamp(),
                    "observation_timestep": observation.get_timestep(),
                    "image_resize_scale": self.config.image_resize_scale,
                    "images": image_transport,
                    "timings_ms": {
                        "robot_get_observation": capture_time_ms,
                        "image_resize": image_resize_ms,
                        "send_observation": self._last_send_observation_diagnostics,
                        "total": (time.perf_counter() - start_time) * 1000,
                    },
                }

            if prefetch_request_id is not None:
                if observation_sent:
                    self._mark_prefetch_sent(
                        prefetch_request_id,
                        observation.get_timestep(),
                    )
                else:
                    self._rollback_prefetch_request(
                        prefetch_request_id,
                        observation.get_timestep(),
                        "send_failed",
                    )

            self.logger.debug(
                f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go}, "
                f"Prefetch: {observation.prefetch_requested}, "
                f"Prefetch request id: {observation.prefetch_request_id})"
            )
            if observation.must_go and observation_sent:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            if prefetch_request_id is None:
                with self.prefetch_state_lock:
                    prefetch_request_id = self._prefetch_trigger_request_id
            if prefetch_request_id is not None:
                with self.latest_action_lock:
                    latest_action = self.latest_action
                observation_timestep = (
                    observation.get_timestep() if observation is not None else max(latest_action, 0)
                )
                self._rollback_prefetch_request(
                    prefetch_request_id,
                    observation_timestep,
                    "observation_sender_exception",
                )
            self.logger.error(f"Error in observation sender: {e}")


    def _build_remote_observation(self, raw_observation: RawObservation) -> dict[str, Any]:
        backend_spec = get_remote_backend_spec(self.backend)
        if backend_spec is None:
            raise ValueError(f"Unsupported remote backend: {self.backend}")

        builder = backend_spec["build_observation"]
        return builder(
            raw_observation,
            front_camera_key=self.config.front_camera_key,
            left_wrist_camera_key=self.config.left_wrist_camera_key,
            right_wrist_camera_key=self.config.right_wrist_camera_key,
        )


    def _convert_remote_actions(
        self,
        action_dict: dict[str, Any],
        observation: TimedObservation,
    ) -> list[TimedAction]:
        backend_spec = get_remote_backend_spec(self.backend)
        if backend_spec is None:
            raise ValueError(f"Unsupported remote backend: {self.backend}")

        converter = backend_spec["convert_actions"]
        return converter(
            action_dict,
            timestamp=observation.get_timestamp(),
            timestep=observation.get_timestep(),
            environment_dt=self.config.environment_dt,
            client_device=self.config.client_device,
        )

    def _put_latest_remote_observation(
        self,
        observation: TimedObservation,
        remote_observation: dict[str, Any],
    ) -> None:
        """Keep only the most recent observation for async ZMQ inference."""
        try:
            self.remote_observation_queue.put_nowait((observation, remote_observation))
            return
        except Exception:
            pass

        try:
            _ = self.remote_observation_queue.get_nowait()
        except Empty:
            pass
        self.remote_observation_queue.put_nowait((observation, remote_observation))

    def control_loop_remote_observation(self, task: str, verbose: bool = False) -> RawObservation | None:
        """Capture an observation and enqueue it for the async ZMQ inference thread."""
        if not self.uses_remote_zmq_backend:
            raise RuntimeError("control_loop_remote_observation is only valid for the remote ZMQ backends")

        try:
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()


            #### Loggggging observation to file

            obs_log = {}

            for key, value in raw_observation.items():
                if key == "task":
                    continue

                shape = getattr(value, "shape", None)

                # image는 저장 안 함
                if shape is not None and len(shape) >= 2:
                    continue

                if hasattr(value, "tolist"):
                    value = value.tolist()

                obs_log[key] = value

            record = {
                "wall_time": observation.get_timestamp(),
                "timestep": observation.get_timestep(),
                "latest_action": latest_action,
                "queue_size": current_queue_size,
                "must_go": observation.must_go,
                "observation": obs_log,
            }

            with open(OBSERVATION_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")




            remote_observation = self._build_remote_observation(raw_observation)
            self._put_latest_remote_observation(observation, remote_observation)

            self.logger.debug(
                f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go}) | "
                f"Queued {self.remote_backend_name} observation #{observation.get_timestep()}"
            )

            if observation.must_go:
                self.must_go.clear()

            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | "
                    f"Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in {self.remote_backend_name} observation loop: {e}")
            return None

    def receive_remote_actions(self, verbose: bool = False):
        """Receive actions from a remote ZMQ policy server in a background thread."""
        if not self.uses_remote_zmq_backend:
            self.logger.debug("receive_remote_actions called for non-ZMQ backend; skipping")
            return
        if self.remote_client is None:
            raise RuntimeError(
                f"{self.remote_backend_name} client not started. "
                "Run RobotClient.start() before requesting actions."
            )

        self.start_barrier.wait()
        self.logger.info(f"{self.remote_backend_name} action receiving thread starting")

        while self.running:
            try:
                observation, remote_observation = self.remote_observation_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                request_start = time.perf_counter()
                action_dict = self.remote_client.get_action(remote_observation)
                request_time = time.perf_counter() - request_start

                ### Loggggging action to file

                raw_actions = action_dict.get("actions")

                if hasattr(raw_actions, "tolist"):
                    raw_actions = raw_actions.tolist()

                record = {
                    "wall_time": time.time(),
                    "request_time_ms": request_time * 1000,
                    "observation_timestamp": observation.get_timestamp(),
                    "observation_timestep": observation.get_timestep(),
                    "actions": raw_actions,
                }

                with open(RAW_ACTION_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")


                timed_actions = self._convert_remote_actions(action_dict, observation)
                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                capture_transition = self.config.debug_chunk_transitions or verbose
                if capture_transition and timed_actions:
                    old_size, old_timesteps = self._inspect_action_queue()
                else:
                    old_size, old_timesteps = 0, []
                with self.latest_action_lock:
                    latest_action = self.latest_action
                incoming_timesteps = (
                    [action.get_timestep() for action in timed_actions]
                    if capture_transition
                    else []
                )

                queue_update_start = time.perf_counter()
                transition_id = (
                    self._next_chunk_transition_id()
                    if self.config.debug_chunk_transitions
                    or self.config.debug_weighted_aggregation
                    else None
                )
                self._aggregate_action_queues(
                    timed_actions,
                    self.config.aggregate_fn,
                    transition_id=transition_id,
                    source_prefetch_request_id=None,
                )
                queue_update_time = time.perf_counter() - queue_update_start

                if capture_transition:
                    new_size, new_timesteps = self._inspect_action_queue()
                else:
                    new_size, new_timesteps = 0, []
                if self.config.debug_chunk_transitions:
                    assert transition_id is not None
                    self._log_chunk_transition(
                        transition_id=transition_id,
                        latest_action=latest_action,
                        old_timesteps=old_timesteps,
                        incoming_timesteps=incoming_timesteps,
                        updated_timesteps=new_timesteps,
                        source_prefetch_request_id=None,
                        queue_update_time_ms=queue_update_time * 1000,
                    )

                # After receiving actions, the next empty queue triggers must-go processing.
                self.must_go.set()

                self.logger.debug(
                    f"{self.remote_backend_name} action request for obs #{observation.get_timestep()} "
                    f"took {request_time * 1000:.2f}ms"
                )

                if verbose and timed_actions:
                    self.logger.info(
                        f"Received {self.remote_backend_name} action chunk for step #{incoming_timesteps[0]} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Request time: {request_time * 1000:.2f}ms"
                    )

                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items "
                        f"({old_timesteps[:1] + old_timesteps[-1:] if old_timesteps else []}) | "
                        f"After: {new_size} items "
                        f"({new_timesteps[:1] + new_timesteps[-1:] if new_timesteps else []})"
                    )

            except Exception as e:
                self.logger.error(f"Error in {self.remote_backend_name} action receiving loop: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        if self.uses_grpc_backend or self.uses_remote_zmq_backend:
            self.start_barrier.wait()
        
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                if self.uses_grpc_backend:
                    _captured_observation = self.control_loop_observation(task, verbose)

                elif self.uses_remote_zmq_backend:
                    _captured_observation = self.control_loop_remote_observation(task, verbose)
                else:
                    raise ValueError(f"Unsupported backend: {self.backend}")

                

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    # TODO: Assert if checking robot support is still needed with the plugin system
    # if cfg.robot.type not in SUPPORTED_ROBOTS:
    #     raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        action_receiver_thread = None

        if client.uses_grpc_backend:
            client.logger.info("Starting gRPC action receiver thread...")
            action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
            action_receiver_thread.start()

        elif client.uses_remote_zmq_backend:
            client.logger.info(f"Starting {client.remote_backend_name} ZMQ action receiver thread...")
            action_receiver_thread = threading.Thread(target=client.receive_remote_actions, daemon=True)
            action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            if action_receiver_thread is not None:
                if client.uses_remote_zmq_backend:
                    action_receiver_thread.join(timeout=1.0)
                else:
                    action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()  # run the client
