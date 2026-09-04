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
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import numpy as np
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



from .camera_image_logger import CameraImageWriter
from .configs import RobotClientConfig, cosine_ramp, cosine_ramp_alpha
from .diagnostic_logger import AsyncJSONLWriter
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
    visualize_action_queue_size,
)
from .image_transport import JPEG_QUALITY, encode_observation_images
from .trajectory import GripperPostprocessor, JerkLimitedTrajectory
from .urdf_limits import (
    build_operational_profile,
    load_active_urdf_limits,
    validate_arm_action_map,
)

LOG_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

RAW_ACTION_LOG = f"raw_actions_{LOG_RUN_ID}.jsonl"
FINAL_ACTION_LOG = f"final_actions_{LOG_RUN_ID}.jsonl"
OBSERVATION_LOG = f"observations_{LOG_RUN_ID}.jsonl"

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
        if config.trajectory_postprocess.enabled:
            trajectory_cfg = config.trajectory_postprocess
            if trajectory_cfg.limits_source == "active_urdf":
                configured_model = str(getattr(config.robot, "model", "auto")).lower()
                active_model = str(trajectory_cfg.active_model).lower()
                if configured_model != "auto" and configured_model != active_model:
                    raise ValueError(
                        "trajectory_postprocess active_model disagrees with robot.model: "
                        f"{trajectory_cfg.active_model!r} != {configured_model!r}"
                    )
                validate_arm_action_map(self.robot.action_features)
                urdf_limits = load_active_urdf_limits(
                    Path(trajectory_cfg.sdk_models_dir),
                    trajectory_cfg.active_model,
                    trajectory_cfg.urdf_version,
                )
                profile = build_operational_profile(trajectory_cfg.profile, urdf_limits)
                trajectory_cfg.arms.position_limits = profile.position_limits
                trajectory_cfg.arms.velocity_limits = profile.velocity_limits
                trajectory_cfg.arms.acceleration_limits = profile.acceleration_limits
                trajectory_cfg.arms.jerk_limits = profile.jerk_limits
                self.logger.info(
                    "Loaded trajectory limits from %s (sha256=%s, profile=%s)",
                    urdf_limits.path,
                    urdf_limits.sha256,
                    profile.name,
                )
            arm_names = [name for name in self.robot.action_features if "_arm_" in name]
            limit_maps = {
                "position": config.trajectory_postprocess.arms.position_limits,
                "velocity": config.trajectory_postprocess.arms.velocity_limits,
                "acceleration": config.trajectory_postprocess.arms.acceleration_limits,
                "jerk": config.trajectory_postprocess.arms.jerk_limits,
            }
            incomplete = {
                label: [name for name in arm_names if name not in values]
                for label, values in limit_maps.items()
                if any(name not in values for name in arm_names)
            }
            if not arm_names or incomplete:
                raise ValueError(
                    "trajectory_postprocess cannot be enabled without explicit limits for every "
                    f"arm action; missing={incomplete or 'arm joint-mode features'}"
                )
        self.robot.connect()
        self.camera_keys = tuple(self.robot.cameras.keys())
        self.backend = getattr(config, "backend", "grpc")
        self.camera_image_writer = (
            CameraImageWriter(
                Path(config.camera_image_log_dir) / LOG_RUN_ID,
                self.camera_keys,
                config.camera_image_save_every_n,
                self.logger,
            )
            if config.save_camera_images
            else None
        )
        if self.camera_image_writer is not None:
            self.logger.info(
                "Camera image capture enabled: directory=%s | cameras=%s | every_n=%d",
                self.camera_image_writer.directory,
                list(self.camera_keys),
                config.camera_image_save_every_n,
            )

        self.policy_config = None
        self.channel = None
        self.stub = None
        self.remote_client = None

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        if self.uses_grpc_backend:
            lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

            self.policy_config = RemotePolicyConfig(
                config.policy_type,
                config.pretrained_name_or_path,
                lerobot_features,
                config.actions_per_chunk,
                config.policy_device,
                timing_diagnostics=config.timing_diagnostics,
                rtc_enabled=config.rtc_enabled,
                rtc_mode=config.rtc_mode,
                rtc_execution_horizon=config.rtc_execution_horizon,
                rtc_max_guidance_weight=config.rtc_max_guidance_weight,
                rtc_prefix_attention_schedule=config.rtc_prefix_attention_schedule,
                rtc_diagnostics_dir=config.rtc_diagnostics_dir,
            )
            self.channel = grpc.insecure_channel(
                self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
            )
            self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
            self.logger.info(f"Initializing client to connect to server at {self.server_address}")
            self.logger.info(
                "Image transport: resize scale = %.3f, JPEG compression = %s, JPEG quality = %d",
                self.config.image_resize_scale,
                self.config.jpeg_compression,
                JPEG_QUALITY,
            )

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
        self.latest_action_tensor = None
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        # Latest-only observation queue for async ZMQ inference.
        # maxsize=1 avoids running inference on stale observations.
        self.remote_observation_queue = Queue(maxsize=1)
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        # Protected by action_queue_lock. True includes the short reservation
        # window between threshold detection and SendObservations completion.
        self._refill_in_flight = False
        self._refill_request_sent = False
        self._refill_request_queue_size = None
        self._refill_request_queue_ratio = None
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        trajectory_cfg = config.trajectory_postprocess
        self._chunk_counter = 0
        self._diagnostic_sample_counter = 0
        self._action_diagnostic_writer = (
            AsyncJSONLWriter(
                trajectory_cfg.logging.path,
                max_queue_size=trajectory_cfg.logging.max_queue_size,
            )
            if trajectory_cfg.logging.enabled
            else None
        )
        # Preserve the legacy final_actions JSONL schema while moving its disk
        # IO off the real-time path.
        self._final_action_writer = AsyncJSONLWriter(FINAL_ACTION_LOG)
        self._trajectory: JerkLimitedTrajectory | None = None
        self._gripper_postprocessor: GripperPostprocessor | None = None
        self._trajectory_target: TimedAction | None = None
        self._trajectory_last_tick: float | None = None
        self._last_sent_action: dict[str, float] | None = None
        self._measured_position_reason: str | None = None
        if trajectory_cfg.enabled:
            feature_names = list(self.robot.action_features)
            arm_names = [name for name in feature_names if "_arm_" in name]
            gripper_names = [name for name in feature_names if "_gripper_" in name]
            if not arm_names:
                raise ValueError(
                    "trajectory_postprocess requires joint-mode arm action features"
                )
            self._trajectory = JerkLimitedTrajectory(
                arm_names,
                trajectory_cfg.arms.velocity_limits,
                trajectory_cfg.arms.acceleration_limits,
                trajectory_cfg.arms.jerk_limits,
                trajectory_cfg.arms.position_limits,
            )
            self._gripper_postprocessor = GripperPostprocessor(
                gripper_names,
                trajectory_cfg.grippers.mode,
                trajectory_cfg.grippers.rate_limits,
            )
            self.logger.info(
                "Trajectory post-processing enabled: policy_rate=%sHz control_rate=%sHz "
                "arms=%s grippers=%s",
                config.fps,
                trajectory_cfg.control_rate_hz,
                arm_names,
                trajectory_cfg.grippers.mode,
            )

        # Diagnostic state is allocated only when explicitly requested. It contains
        # scalar timings and timestep metadata, never actions or observations.
        self._timing_history = deque(maxlen=10) if config.timing_diagnostics else None
        self._last_action_perf_time = None
        self._last_action_timestep = None
        if config.timing_diagnostics and hasattr(self.robot, "_timing_diagnostics"):
            # Rby1 consumes this private opt-in marker for per-camera read timing.
            # Other robot implementations simply ignore it.
            self.robot._timing_diagnostics = True

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

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
                self._clear_refill_in_flight()
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
        self._clear_refill_in_flight()

        camera_image_writer = getattr(self, "camera_image_writer", None)
        if camera_image_writer is not None:
            camera_image_writer.close()

        for writer_name in ("_action_diagnostic_writer", "_final_action_writer"):
            writer = getattr(self, writer_name, None)
            if writer is not None:
                writer.close()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        if self.channel is not None:
            self.channel.close()
            self.logger.debug("Client stopped, channel closed")

        if self.remote_client is not None:
            self.remote_client.close()
            self.remote_client = None
            self.logger.debug("Remote ZMQ client closed")

    def _clear_refill_in_flight(self) -> bool:
        """Clear the refill state and return its previous value."""
        with self.action_queue_lock:
            was_in_flight = self._refill_in_flight
            self._refill_in_flight = False
            self._refill_request_sent = False
            self._refill_request_queue_size = None
            self._refill_request_queue_ratio = None
        return was_in_flight

    def _refill_rpc_state(self) -> tuple[bool, bool]:
        with self.action_queue_lock:
            return self._refill_in_flight, self._refill_request_sent

    def send_observation(
        self,
        obs: TimedObservation,
        timing: dict[str, Any] | None = None,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.uses_grpc_backend:
            raise RuntimeError("send_observation is only valid for the gRPC backend")
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        observation_to_send = obs
        transport_preprocess_ms = 0.0
        resize_ms = 0.0
        jpeg_encode_ms = 0.0
        transport_bytes = 0
        if self.config.image_resize_scale != 1.0 or self.config.jpeg_compression:
            transport_observation, image_stats = encode_observation_images(
                obs.get_observation(),
                self.camera_keys,
                self.config.image_resize_scale,
                self.config.jpeg_compression,
            )
            observation_to_send = replace(obs, observation=transport_observation)
            transport_preprocess_ms = image_stats.total_time * 1000
            resize_ms = image_stats.resize_time * 1000
            jpeg_encode_ms = image_stats.jpeg_encode_time * 1000
            transport_bytes = image_stats.transport_bytes
            compression_ratio = (
                image_stats.original_bytes / image_stats.transport_bytes
                if image_stats.transport_bytes
                else 0.0
            )
            self.logger.debug(
                "Image transport preprocessing: images=%d | resize=%.3fms | JPEG encode=%.3fms | "
                "total=%.3fms | original=%d bytes | transport=%d bytes | compression=%.2fx",
                image_stats.image_count,
                image_stats.resize_time * 1000,
                image_stats.jpeg_encode_time * 1000,
                image_stats.total_time * 1000,
                image_stats.original_bytes,
                image_stats.transport_bytes,
                compression_ratio,
            )

        camera_image_writer = getattr(self, "camera_image_writer", None)
        if camera_image_writer is not None:
            camera_image_writer.submit(
                observation_to_send.get_observation(),
                wall_time=observation_to_send.get_timestamp(),
                timestep=observation_to_send.get_timestep(),
            )

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(observation_to_send)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        if timing is not None:
            timing.update(
                transport_preprocess_ms=transport_preprocess_ms,
                resize_ms=resize_ms,
                jpeg_encode_ms=jpeg_encode_ms,
                transport_bytes=transport_bytes,
                serialized_bytes=len(observation_bytes),
                serialization_ms=serialize_time * 1000,
            )

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            grpc_start = time.perf_counter() if timing is not None else 0.0
            _ = self.stub.SendObservations(observation_iterator)
            if timing is not None:
                timing["grpc_send_observation_ms"] = (
                    time.perf_counter() - grpc_start
                ) * 1000
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    @staticmethod
    def _tensor_list(value: torch.Tensor | None) -> list[float] | None:
        if value is None:
            return None
        return value.detach().cpu().flatten().tolist()

    def _submit_diagnostic(self, record: dict[str, Any], *, downsample: bool = False) -> None:
        writer = getattr(self, "_action_diagnostic_writer", None)
        if writer is None:
            return
        if downsample:
            self._diagnostic_sample_counter += 1
            factor = self.config.trajectory_postprocess.logging.downsample
            if (self._diagnostic_sample_counter - 1) % factor:
                return
        if not writer.submit(record) and writer.dropped_records in {1, 10, 100}:
            self.logger.warning(
                "Action diagnostic queue full; dropped_records=%d", writer.dropped_records
            )

    def _ingest_action_chunk(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
        *,
        receive_wall_time: float | None = None,
        receive_monotonic_time: float | None = None,
        timing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Attach one clock-domain mapping, discard stale work, then merge."""
        receive_wall = time.time() if receive_wall_time is None else receive_wall_time
        receive_mono = time.monotonic() if receive_monotonic_time is None else receive_monotonic_time
        self._chunk_counter = getattr(self, "_chunk_counter", 0) + 1
        chunk_id = self._chunk_counter
        with self.action_queue_lock:
            queue_depth_before = self.action_queue.qsize()

        future: list[TimedAction] = []
        stale = 0
        for action in incoming_actions:
            scheduled = receive_mono + (action.get_timestamp() - receive_wall)
            metadata = dict(getattr(action, "metadata", {}) or {})
            metadata.update(
                chunk_id=chunk_id,
                source_chunk_id=chunk_id,
                absolute_timestep=action.get_timestep(),
                scheduled_execution_time=scheduled,
                scheduled_execution_wall_time=action.get_timestamp(),
                chunk_received_monotonic_time=receive_mono,
                chunk_received_wall_time=receive_wall,
            )
            metadata.setdefault("overlap_index", None)
            metadata.setdefault("overlap_length", 0)
            metadata.setdefault("cosine_alpha", None)
            metadata.setdefault("old_action", None)
            metadata.setdefault("new_action", self._tensor_list(action.get_action()))
            metadata.setdefault("blended_action", self._tensor_list(action.get_action()))
            metadata.setdefault("old_new_disagreement_norm", None)
            action.metadata = metadata
            if scheduled <= receive_mono:
                stale += 1
            else:
                future.append(action)

        merge_timing = timing if timing is not None else {}
        self._aggregate_action_queues(
            future,
            aggregate_fn,
            timing=merge_timing,
            current_monotonic_time=receive_mono,
        )
        with self.action_queue_lock:
            queue_depth_after = self.action_queue.qsize()
        stale += int(merge_timing.get("stale_actions_dropped", 0))
        first_metadata = (
            getattr(incoming_actions[0], "metadata", {}) if incoming_actions else {}
        )
        record = {
            "record_type": "chunk",
            "chunk_id": chunk_id,
            "source_observation_timestep": first_metadata.get(
                "source_observation_timestep",
                incoming_actions[0].get_timestep() if incoming_actions else None,
            ),
            "chunk_received_monotonic_time": receive_mono,
            "chunk_received_wall_time": receive_wall,
            "inference_start_time": first_metadata.get("inference_start_time"),
            "inference_end_time": first_metadata.get("inference_end_time"),
            "queue_depth_before": queue_depth_before,
            "queue_depth_after": queue_depth_after,
            "stale_actions_dropped": stale,
        }
        self._submit_diagnostic(record)
        return record

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        timing: dict[str, Any] | None = None,
        current_monotonic_time: float | None = None,
    ):
        """Merge queued and incoming future actions, aggregating matching timesteps."""
        diagnostics = timing is not None
        aggregate_start = time.perf_counter() if diagnostics else 0.0
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        snapshot_wait_start = time.perf_counter() if diagnostics else 0.0
        with self.action_queue_lock:
            snapshot_acquired = time.perf_counter() if diagnostics else 0.0
            internal_queue = list(self.action_queue.queue)
        snapshot_end = time.perf_counter() if diagnostics else 0.0

        with self.latest_action_lock:
            latest_action = self.latest_action
            latest_action_tensor = getattr(self, "latest_action_tensor", None)

        merge_compute_start = time.perf_counter() if diagnostics else 0.0
        stale_old = 0
        old_by_timestep = {
            action.get_timestep(): action
            for action in internal_queue
            if action.get_timestep() > latest_action
            and not (
                current_monotonic_time is not None
                and (getattr(action, "metadata", {}) or {}).get("scheduled_execution_time")
                is not None
                and action.metadata["scheduled_execution_time"] <= current_monotonic_time
            )
        }
        stale_old = len(internal_queue) - len(old_by_timestep)
        incoming_by_timestep = {
            action.get_timestep(): action
            for action in incoming_actions
            if action.get_timestep() > latest_action
        }

        old_timesteps = set(old_by_timestep)
        incoming_timesteps = set(incoming_by_timestep)
        overlap_timesteps = sorted(old_timesteps & incoming_timesteps)
        overlap_index_by_timestep = {
            timestep: overlap_index
            for overlap_index, timestep in enumerate(overlap_timesteps)
        }
        use_cosine_ramp = aggregate_fn is cosine_ramp
        merge_compute_end = time.perf_counter() if diagnostics else 0.0

        queue_rebuild_start = time.perf_counter() if diagnostics else 0.0
        future_action_queue = Queue()
        for timestep in sorted(old_timesteps | incoming_timesteps):
            if timestep not in old_by_timestep:
                future_action_queue.put(incoming_by_timestep[timestep])
            elif timestep not in incoming_by_timestep:
                future_action_queue.put(old_by_timestep[timestep])
            else:
                old_action = old_by_timestep[timestep]
                new_action = incoming_by_timestep[timestep]
                if use_cosine_ramp:
                    overlap_index = overlap_index_by_timestep[timestep]
                    overlap_count = len(overlap_timesteps)
                    aggregated_action = cosine_ramp(
                        old_action.get_action(),
                        new_action.get_action(),
                        overlap_index=overlap_index,
                        overlap_count=overlap_count,
                    )
                else:
                    aggregated_action = aggregate_fn(
                        old_action.get_action(), new_action.get_action()
                    )
                metadata = dict(getattr(new_action, "metadata", {}) or {})
                metadata.update(
                    overlap_index=overlap_index_by_timestep[timestep],
                    overlap_length=len(overlap_timesteps),
                    cosine_alpha=(
                        cosine_ramp_alpha(
                            overlap_index_by_timestep[timestep], len(overlap_timesteps)
                        )
                        if use_cosine_ramp
                        else None
                    ),
                    old_action=self._tensor_list(old_action.get_action()),
                    new_action=self._tensor_list(new_action.get_action()),
                    blended_action=self._tensor_list(aggregated_action),
                    old_new_disagreement_norm=float(
                        torch.linalg.vector_norm(
                            old_action.get_action().detach().flatten()
                            - new_action.get_action().detach().flatten()
                        ).item()
                    ),
                )
                future_action_queue.put(
                    TimedAction(
                        timestamp=new_action.get_timestamp(),
                        timestep=timestep,
                        action=aggregated_action,
                        metadata=metadata,
                    )
                )
        queue_rebuild_end = time.perf_counter() if diagnostics else 0.0

        if use_cosine_ramp and diagnostics and overlap_timesteps:
            overlap_count = len(overlap_timesteps)
            alpha_first = cosine_ramp_alpha(0, overlap_count)
            alpha_mid = cosine_ramp_alpha(overlap_count // 2, overlap_count)
            alpha_last = cosine_ramp_alpha(overlap_count - 1, overlap_count)
            self.logger.debug(
                "[COSINE_RAMP] overlap_count=%d first_timestep=%d last_timestep=%d "
                "alpha_first=%.4f alpha_mid=%.4f alpha_last=%.4f",
                overlap_count,
                overlap_timesteps[0],
                overlap_timesteps[-1],
                alpha_first,
                alpha_mid,
                alpha_last,
            )

            final_by_timestep = {
                action.get_timestep(): action for action in future_action_queue.queue
            }
            first_final_timestep = min(final_by_timestep, default=None)
            start_metrics = self._action_boundary_metrics(
                latest_action_tensor,
                (
                    final_by_timestep[first_final_timestep].get_action()
                    if first_final_timestep is not None
                    else None
                ),
            )
            first_incoming_only_after_overlap = min(
                (
                    timestep
                    for timestep in incoming_timesteps - old_timesteps
                    if timestep > overlap_timesteps[-1]
                ),
                default=None,
            )
            end_metrics = self._action_boundary_metrics(
                final_by_timestep[overlap_timesteps[-1]].get_action(),
                (
                    final_by_timestep[first_incoming_only_after_overlap].get_action()
                    if first_incoming_only_after_overlap is not None
                    else None
                ),
            )
            self.logger.debug(
                "[COSINE_RAMP][BOUNDARY] start_transition_max_abs_delta=%s "
                "start_transition_l2_delta=%s end_transition_max_abs_delta=%s "
                "end_transition_l2_delta=%s",
                self._format_boundary_metric(start_metrics, "max_abs_delta"),
                self._format_boundary_metric(start_metrics, "l2_delta"),
                self._format_boundary_metric(end_metrics, "max_abs_delta"),
                self._format_boundary_metric(end_metrics, "l2_delta"),
            )

        self.logger.debug(
            "Action queue merge: old=%d | incoming=%d | old_only=%d | overlap=%d | "
            "incoming_only=%d | final=%d",
            len(old_timesteps),
            len(incoming_timesteps),
            len(old_timesteps - incoming_timesteps),
            len(old_timesteps & incoming_timesteps),
            len(incoming_timesteps - old_timesteps),
            future_action_queue.qsize(),
        )

        replace_wait_start = time.perf_counter() if diagnostics else 0.0
        with self.action_queue_lock:
            replace_acquired = time.perf_counter() if diagnostics else 0.0
            self.action_queue = future_action_queue
        replace_end = time.perf_counter() if diagnostics else 0.0

        if timing is not None:
            old_queue_timesteps = [action.get_timestep() for action in internal_queue]
            incoming_all_timesteps = [action.get_timestep() for action in incoming_actions]
            final_timesteps = [action.get_timestep() for action in future_action_queue.queue]
            timing.update(
                aggregate_lock_wait_ms=(
                    (snapshot_acquired - snapshot_wait_start)
                    + (replace_acquired - replace_wait_start)
                )
                * 1000,
                old_queue_snapshot_ms=(snapshot_end - snapshot_acquired) * 1000,
                merge_compute_ms=(merge_compute_end - merge_compute_start) * 1000,
                queue_rebuild_ms=(queue_rebuild_end - queue_rebuild_start) * 1000,
                queue_replace_or_update_ms=(replace_end - replace_wait_start) * 1000,
                aggregate_total_ms=(replace_end - aggregate_start) * 1000,
                latest_action_timestep=latest_action,
                old_queue_size=len(old_queue_timesteps),
                old_queue_first_timestep=(old_queue_timesteps[0] if old_queue_timesteps else None),
                old_queue_last_timestep=(old_queue_timesteps[-1] if old_queue_timesteps else None),
                incoming_size=len(incoming_all_timesteps),
                incoming_first_timestep=(incoming_all_timesteps[0] if incoming_all_timesteps else None),
                incoming_last_timestep=(incoming_all_timesteps[-1] if incoming_all_timesteps else None),
                final_queue_size=len(final_timesteps),
                final_queue_first_timestep=(final_timesteps[0] if final_timesteps else None),
                final_queue_last_timestep=(final_timesteps[-1] if final_timesteps else None),
                stale_actions_dropped=stale_old,
            )

    def _action_boundary_metrics(
        self,
        first: torch.Tensor | None,
        second: torch.Tensor | None,
    ) -> dict[str, float] | None:
        """Compute arm-joint deltas without affecting queue or control semantics."""
        if first is None or second is None:
            return None

        first_flat = first.detach().flatten()
        second_flat = second.detach().flatten()
        if first_flat.shape != second_flat.shape or first_flat.numel() == 0:
            return None

        action_features = getattr(getattr(self, "robot", None), "action_features", None)
        feature_names = list(action_features) if action_features is not None else []
        arm_indices = [
            index
            for index, name in enumerate(feature_names)
            if "_arm_" in name and index < first_flat.numel()
        ]
        if arm_indices:
            first_flat = first_flat[arm_indices]
            second_flat = second_flat[arm_indices]

        delta = second_flat - first_flat
        return {
            "max_abs_delta": delta.abs().max().item(),
            "l2_delta": torch.linalg.vector_norm(delta).item(),
        }

    @staticmethod
    def _format_boundary_metric(metrics: dict[str, float] | None, key: str) -> str:
        return "n/a" if metrics is None else f"{metrics[key]:.6f}"

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
                rpc_waited_for_refill, refill_was_sent_at_rpc_start = self._refill_rpc_state()
                rpc_start = time.perf_counter() if self.config.timing_diagnostics else 0.0
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                rpc_end = time.perf_counter() if self.config.timing_diagnostics else 0.0
                if len(actions_chunk.data) == 0:
                    # Only an RPC that started while a refill was outstanding may
                    # release it. This avoids an older polling RPC racing with a
                    # newly reserved request.
                    if rpc_waited_for_refill and refill_was_sent_at_rpc_start:
                        self._clear_refill_in_flight()
                        self.must_go.set()
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()
                receive_monotonic_time = time.monotonic()
                processing_start = time.perf_counter() if self.config.timing_diagnostics else 0.0

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                device_transfer_start = (
                    time.perf_counter() if self.config.timing_diagnostics else 0.0
                )
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")
                device_transfer_ms = (
                    (time.perf_counter() - device_transfer_start) * 1000
                    if self.config.timing_diagnostics
                    else 0.0
                )

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Client receive wall time: {receive_time:.6f} | "
                        f"Action policy wall time: {timed_actions[0].get_timestamp():.6f} | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                aggregate_timing = {} if self.config.timing_diagnostics else None
                self._ingest_action_chunk(
                    timed_actions,
                    self.config.aggregate_fn,
                    receive_wall_time=receive_time,
                    receive_monotonic_time=receive_monotonic_time,
                    timing=aggregate_timing,
                )
                queue_update_time = time.perf_counter() - start_time
                queue_updated = time.perf_counter() if self.config.timing_diagnostics else 0.0

                refill_was_in_flight = self._clear_refill_in_flight()

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if self.config.timing_diagnostics:
                    receiver_processing_total_ms = (time.perf_counter() - processing_start) * 1000
                    incoming_first = aggregate_timing["incoming_first_timestep"]
                    incoming_last = aggregate_timing["incoming_last_timestep"]
                    self.logger.info(
                        "[TIMING][CHUNK_RECEIVED] event_wall_time=%.6f chunk_id=%s "
                        "chunk_first=%s chunk_last=%s rpc_wait_ms=%.3f deserialize_ms=%.3f "
                        "device_transfer_ms=%.3f post_receive_processing_ms=%.3f "
                        "aggregate_wait_for_lock_ms=%.3f aggregate_compute_ms=%.3f "
                        "old_queue_snapshot_ms=%.3f queue_rebuild_ms=%.3f "
                        "queue_replace_or_update_ms=%.3f aggregate_total_ms=%.3f "
                        "receiver_processing_total_ms=%.3f latest_action_timestep=%s "
                        "old_queue=%s:%s:%s incoming=%s:%s:%s final_queue=%s:%s:%s",
                        receive_time,
                        incoming_first,
                        incoming_first,
                        incoming_last,
                        (rpc_end - rpc_start) * 1000,
                        deserialize_time * 1000,
                        device_transfer_ms,
                        (queue_updated - processing_start) * 1000,
                        aggregate_timing["aggregate_lock_wait_ms"],
                        aggregate_timing["merge_compute_ms"],
                        aggregate_timing["old_queue_snapshot_ms"],
                        aggregate_timing["queue_rebuild_ms"],
                        aggregate_timing["queue_replace_or_update_ms"],
                        aggregate_timing["aggregate_total_ms"],
                        receiver_processing_total_ms,
                        aggregate_timing["latest_action_timestep"],
                        aggregate_timing["old_queue_size"],
                        aggregate_timing["old_queue_first_timestep"],
                        aggregate_timing["old_queue_last_timestep"],
                        aggregate_timing["incoming_size"],
                        incoming_first,
                        incoming_last,
                        aggregate_timing["final_queue_size"],
                        aggregate_timing["final_queue_first_timestep"],
                        aggregate_timing["final_queue_last_timestep"],
                    )
                    with self.latest_action_lock:
                        latest_action_for_refill = self.latest_action
                    self.logger.info(
                        "[TIMING][REFILL_COMPLETE] event_wall_time=%.6f chunk_id=%s "
                        "latest_action=%s old_queue_size=%s final_queue_size=%s "
                        "refill_in_flight_before=%s refill_in_flight=False",
                        time.time(),
                        incoming_first,
                        latest_action_for_refill,
                        aggregate_timing["old_queue_size"],
                        aggregate_timing["final_queue_size"],
                        refill_was_in_flight,
                    )

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self._clear_refill_in_flight()
                self.must_go.set()
                self.logger.error(f"Error receiving actions: {e}")

            except Exception as e:
                self._clear_refill_in_flight()
                self.must_go.set()
                self.logger.error(f"Error processing received actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            if self.action_queue.empty():
                return False
            metadata = getattr(self.action_queue.queue[0], "metadata", {}) or {}
            scheduled = metadata.get("scheduled_execution_time")
            return scheduled is None or scheduled <= time.monotonic()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def _read_measured_joint_positions(self) -> tuple[dict[str, float] | None, str | None]:
        try:
            reader = getattr(self.robot, "get_joint_positions", None)
            measured = reader() if callable(reader) else self.robot.get_observation()
            result = {
                name: float(measured[name])
                for name in self.robot.action_features
                if name in measured and np.isfinite(float(measured[name]))
            }
            arm_names = self._trajectory.joint_names if self._trajectory is not None else ()
            if all(name in result for name in arm_names):
                return result, None
            return None, "robot observation did not contain every commanded arm joint"
        except Exception as exc:
            return None, f"measured joint read unavailable: {type(exc).__name__}: {exc}"

    def reset_trajectory(self, measured_position: dict[str, float] | None = None) -> bool:
        """Reset post-processing from measured state after an episode reset."""
        if self._trajectory is None:
            return False
        reason = None
        if measured_position is None:
            measured_position, reason = self._read_measured_joint_positions()
        if measured_position is None and self._last_sent_action is not None:
            measured_position = self._last_sent_action
            reason = "measured position unavailable; seeded from last sent command"
        if measured_position is None:
            self._measured_position_reason = reason or "no measured or previously sent command available"
            self.logger.error("Cannot reset trajectory: %s", self._measured_position_reason)
            return False
        try:
            self._trajectory.reset(
                [measured_position[name] for name in self._trajectory.joint_names]
            )
            if self._gripper_postprocessor is not None:
                gripper_values = [
                    measured_position.get(
                        name,
                        self._last_sent_action.get(name, 0.0)
                        if self._last_sent_action
                        else 0.0,
                    )
                    for name in self._gripper_postprocessor.joint_names
                ]
                self._gripper_postprocessor.reset(gripper_values)
            self._trajectory_last_tick = time.monotonic()
            self._measured_position_reason = reason
            return True
        except (KeyError, ValueError) as exc:
            self._measured_position_reason = f"invalid reset position: {exc}"
            self.logger.error("Cannot reset trajectory: %s", self._measured_position_reason)
            return False

    def _pop_due_waypoint(self, now_monotonic: float) -> tuple[TimedAction | None, int]:
        """Consume due waypoints once, returning only the newest after a stall."""
        due: list[TimedAction] = []
        with self.action_queue_lock:
            while not self.action_queue.empty():
                candidate = self.action_queue.queue[0]
                scheduled = (getattr(candidate, "metadata", {}) or {}).get(
                    "scheduled_execution_time", now_monotonic
                )
                if scheduled > now_monotonic:
                    break
                due.append(self.action_queue.get_nowait())
        if not due:
            return None, 0
        newest = due[-1]
        dropped = len(due) - 1
        if dropped:
            self.logger.warning(
                "Control loop missed %d policy waypoints; dropped them without catch-up burst",
                dropped,
            )
        return newest, dropped

    def control_loop_trajectory_action(self, now_monotonic: float | None = None) -> dict[str, Any] | None:
        """Generate and send at most one high-rate command for this control tick."""
        if self._trajectory is None:
            raise RuntimeError("trajectory post-processing is disabled")
        now = time.monotonic() if now_monotonic is None else now_monotonic
        waypoint, delayed_drops = self._pop_due_waypoint(now)
        if waypoint is not None:
            self._trajectory_target = waypoint
            with self.latest_action_lock:
                self.latest_action = waypoint.get_timestep()
                self.latest_action_tensor = waypoint.get_action()
        if self._trajectory_target is None:
            return None
        if not self._trajectory.initialized and not self.reset_trajectory():
            return None

        nominal_dt = 1.0 / self.config.trajectory_postprocess.control_rate_hz
        dt = nominal_dt if self._trajectory_last_tick is None else now - self._trajectory_last_tick
        if dt <= 0:
            dt = nominal_dt
        self._trajectory_last_tick = now
        target_action = self._action_tensor_to_action_dict(self._trajectory_target.get_action())
        trajectory_error = None
        if waypoint is not None:
            try:
                self._trajectory.set_target(
                    [target_action[name] for name in self._trajectory.joint_names]
                )
            except (KeyError, ValueError) as exc:
                trajectory_error = f"invalid target: {exc}"

        sample = (
            self._trajectory.hold_last_valid(trajectory_error, dt)
            if trajectory_error is not None
            else self._trajectory.step(dt)
        )
        limited_action = dict(target_action)
        for index, name in enumerate(self._trajectory.joint_names):
            limited_action[name] = float(sample.position[index])

        gripper = self._gripper_postprocessor
        if gripper is not None and waypoint is not None and gripper.joint_names:
            gripper_position = gripper.update(
                [target_action[name] for name in gripper.joint_names], dt
            )
            for index, name in enumerate(gripper.joint_names):
                limited_action[name] = float(gripper_position[index])
        elif gripper is not None and gripper.position is not None:
            for index, name in enumerate(gripper.joint_names):
                limited_action[name] = float(gripper.position[index])

        measured, measured_reason = self._read_measured_joint_positions()
        actual_send_monotonic = time.monotonic()
        actual_send_wall = time.time()
        performed = self.robot.send_action(limited_action)
        self._last_sent_action = dict(limited_action)

        final_record = {
            "wall_time": actual_send_wall,
            "policy_timestamp": self._trajectory_target.get_timestamp(),
            "timestep": self._trajectory_target.get_timestep(),
            "action": limited_action,
        }
        self._final_action_writer.submit(final_record)

        metadata = getattr(self._trajectory_target, "metadata", {}) or {}
        velocity = {name: float(sample.velocity[i]) for i, name in enumerate(self._trajectory.joint_names)}
        acceleration = {
            name: float(sample.acceleration[i]) for i, name in enumerate(self._trajectory.joint_names)
        }
        jerk = {name: float(sample.jerk[i]) for i, name in enumerate(self._trajectory.joint_names)}
        tracking_error = (
            {name: measured[name] - limited_action[name] for name in self._trajectory.joint_names}
            if measured is not None
            else None
        )
        self._submit_diagnostic(
            {
                "record_type": "action",
                "wall_timestamp": actual_send_wall,
                "monotonic_timestamp": actual_send_monotonic,
                "absolute_timestep": self._trajectory_target.get_timestep(),
                "scheduled_execution_time": metadata.get("scheduled_execution_time"),
                "actual_send_time": actual_send_monotonic,
                "source_chunk_id": metadata.get("source_chunk_id"),
                "overlap_index": metadata.get("overlap_index"),
                "overlap_length": metadata.get("overlap_length", 0),
                "cosine_alpha": metadata.get("cosine_alpha"),
                "old_action": metadata.get("old_action"),
                "new_action": metadata.get("new_action"),
                "blended_action": metadata.get("blended_action"),
                "limited_action": limited_action,
                "old_new_disagreement_norm": metadata.get("old_new_disagreement_norm"),
                "command_velocity": velocity,
                "command_acceleration": acceleration,
                "command_jerk": jerk,
                "measured_joint_position": measured,
                "measured_joint_position_reason": measured_reason,
                "position_tracking_error": tracking_error,
                "stale_actions_dropped": delayed_drops,
                "trajectory_valid": sample.valid,
                "trajectory_error": sample.error,
            },
            downsample=True,
        )
        return performed

    def _emit_action_timing(
        self,
        timestep: int,
        timing: dict[str, Any],
        action_event_perf: float | None = None,
        action_event_wall: float | None = None,
    ) -> None:
        """Record one scalar action event and warn when its interval exceeds two periods."""
        if action_event_perf is None:
            action_event_perf = time.perf_counter()
        if action_event_wall is None:
            action_event_wall = time.time()
        interval_ms = None
        if self._last_action_perf_time is not None:
            interval_ms = (action_event_perf - self._last_action_perf_time) * 1000

        timing["action_interval_ms"] = interval_ms
        timing["timestep"] = timestep
        timing["event_wall_time"] = action_event_wall
        self.logger.debug(
            "[TIMING][ACTION] event_wall_time=%.6f timestep=%s interval_ms=%s "
            "queue_size_before=%s queue_size_after=%s queue_lock_wait_ms=%.3f "
            "queue_pop_ms=%.3f action_to_dict_ms=%.3f robot_send_action_ms=%.3f "
            "latest_action_update_ms=%.3f action_log_write_ms=%.3f "
            "control_loop_action_total_ms=%.3f",
            action_event_wall,
            timestep,
            f"{interval_ms:.3f}" if interval_ms is not None else "n/a",
            timing["queue_size_before"],
            timing["queue_size_after"],
            timing["queue_lock_wait_ms"],
            timing["queue_pop_ms"],
            timing["action_to_dict_ms"],
            timing["robot_send_action_ms"],
            timing["latest_action_update_ms"],
            timing["action_log_write_ms"],
            timing["control_loop_action_total_ms"],
        )

        if interval_ms is not None and interval_ms > 2 * self.config.environment_dt * 1000:
            previous_loop = self._timing_history[-1] if self._timing_history else {}
            previous_action = previous_loop.get("action", {})
            previous_observation = previous_loop.get("observation", {})
            self.logger.warning(
                "[TIMING][STALL] event_wall_time=%.6f prev_timestep=%s current_timestep=%s "
                "action_interval_ms=%.3f target_ms=%.3f queue_size_before=%s queue_size_after=%s "
                "queue_lock_wait_ms=%.3f queue_pop_ms=%.3f robot_send_action_ms=%.3f "
                "robot_get_observation_ms=%.3f transport_preprocess_ms=%.3f "
                "serialization_ms=%.3f grpc_send_observation_ms=%.3f "
                "control_loop_observation_ms=%.3f loop_total_ms=%.3f sleep_budget_ms=%.3f "
                "current_queue_lock_wait_ms=%.3f current_robot_send_action_ms=%.3f",
                action_event_wall,
                self._last_action_timestep,
                timestep,
                interval_ms,
                self.config.environment_dt * 1000,
                timing["queue_size_before"],
                timing["queue_size_after"],
                previous_action.get("queue_lock_wait_ms", 0.0),
                previous_action.get("queue_pop_ms", 0.0),
                previous_action.get("robot_send_action_ms", 0.0),
                previous_observation.get("robot_get_observation_ms", 0.0),
                previous_observation.get("transport_preprocess_ms", 0.0),
                previous_observation.get("serialization_ms", 0.0),
                previous_observation.get("grpc_send_observation_ms", 0.0),
                previous_observation.get("control_loop_observation_ms", 0.0),
                previous_loop.get("loop_total_ms", 0.0),
                previous_loop.get("sleep_budget_ms", 0.0),
                timing["queue_lock_wait_ms"],
                timing["robot_send_action_ms"],
            )

        self._last_action_perf_time = action_event_perf
        self._last_action_timestep = timestep

    def control_loop_action(
        self,
        verbose: bool = False,
        timing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        action_start = time.perf_counter() if timing is not None else 0.0
        get_start = time.perf_counter()
        with self.action_queue_lock:
            lock_acquired = time.perf_counter() if timing is not None else 0.0
            queue_size_before = self.action_queue.qsize()
            self.action_queue_size.append(queue_size_before)
            # Get action from queue
            pop_start = time.perf_counter() if timing is not None else 0.0
            timed_action = self.action_queue.get_nowait()
            delayed_drops = 0
            scheduled = (getattr(timed_action, "metadata", {}) or {}).get(
                "scheduled_execution_time"
            )
            if scheduled is not None:
                now_monotonic = time.monotonic()
                while not self.action_queue.empty():
                    next_action = self.action_queue.queue[0]
                    next_scheduled = (getattr(next_action, "metadata", {}) or {}).get(
                        "scheduled_execution_time"
                    )
                    if next_scheduled is None or next_scheduled > now_monotonic:
                        break
                    timed_action = self.action_queue.get_nowait()
                    delayed_drops += 1
            pop_end = time.perf_counter() if timing is not None else 0.0
            # 2배속 소모를 원한다면...
            # timed_action = self.action_queue.get_nowait() 
        get_end = time.perf_counter() - get_start

        action_to_dict_start = time.perf_counter() if timing is not None else 0.0
        logging_actions = self._action_tensor_to_action_dict(timed_action.get_action())
        action_to_dict_end = time.perf_counter() if timing is not None else 0.0

        latest_action_update_start = time.perf_counter() if timing is not None else 0.0
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
            self.latest_action_tensor = timed_action.get_action()
        latest_action_update_end = time.perf_counter() if timing is not None else 0.0

        robot_send_start = time.perf_counter() if timing is not None else 0.0
        _performed_action = self.robot.send_action(logging_actions)
        robot_send_end = time.perf_counter() if timing is not None else 0.0

        # Loggggging action to file

        record = {
            "wall_time": time.time(),
            "policy_timestamp": timed_action.get_timestamp(),
            "timestep": timed_action.get_timestep(),
            "action": logging_actions,
        }

        action_log_start = time.perf_counter() if timing is not None else 0.0
        final_writer = getattr(self, "_final_action_writer", None)
        if final_writer is not None:
            final_writer.submit(record)
        else:
            # Compatibility for lightweight test clients constructed with
            # __new__; production clients always use the background writer.
            with open(FINAL_ACTION_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        action_log_end = time.perf_counter() if timing is not None else 0.0

        metadata = getattr(timed_action, "metadata", {}) or {}
        self._submit_diagnostic(
            {
                "record_type": "action",
                "wall_timestamp": record["wall_time"],
                "monotonic_timestamp": robot_send_end if timing is not None else time.monotonic(),
                "absolute_timestep": timed_action.get_timestep(),
                "scheduled_execution_time": metadata.get("scheduled_execution_time"),
                "actual_send_time": robot_send_end if timing is not None else time.monotonic(),
                "source_chunk_id": metadata.get("source_chunk_id"),
                "overlap_index": metadata.get("overlap_index"),
                "overlap_length": metadata.get("overlap_length", 0),
                "cosine_alpha": metadata.get("cosine_alpha"),
                "old_action": metadata.get("old_action"),
                "new_action": metadata.get("new_action"),
                "blended_action": metadata.get(
                    "blended_action", self._tensor_list(timed_action.get_action())
                ),
                "limited_action": logging_actions,
                "old_new_disagreement_norm": metadata.get("old_new_disagreement_norm"),
                "command_velocity": None,
                "command_acceleration": None,
                "command_jerk": None,
                "measured_joint_position": None,
                "measured_joint_position_reason": "trajectory post-processing disabled",
                "position_tracking_error": None,
                "stale_actions_dropped": delayed_drops,
            },
            downsample=True,
        )

        if timing is not None:
            timing.update(
                queue_size_before=queue_size_before,
                queue_size_after=max(0, queue_size_before - 1 - delayed_drops),
                queue_lock_wait_ms=(lock_acquired - get_start) * 1000,
                queue_pop_ms=(pop_end - pop_start) * 1000,
                action_to_dict_ms=(action_to_dict_end - action_to_dict_start) * 1000,
                latest_action_update_ms=(
                    latest_action_update_end - latest_action_update_start
                )
                * 1000,
                robot_send_action_ms=(robot_send_end - robot_send_start) * 1000,
                action_log_write_ms=(action_log_end - action_log_start) * 1000,
                control_loop_action_total_ms=(action_log_end - action_start) * 1000,
            )
            self._emit_action_timing(
                timed_action.get_timestep(),
                timing,
                action_event_perf=robot_send_end,
                action_event_wall=record["wall_time"],
            )



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
        """Reserve one gRPC refill when the queue reaches its threshold."""
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            if not self.uses_grpc_backend:
                return queue_size / self.action_chunk_size <= self._chunk_size_threshold

            if self._refill_in_flight:
                return False

            queue_ratio = (
                queue_size / self.action_chunk_size if self.action_chunk_size > 0 else 0.0
            )
            if queue_ratio > self._chunk_size_threshold:
                return False

            # Reserve before observation capture so the next control iteration
            # cannot create a duplicate request. Failed sends roll this back.
            self._refill_in_flight = True
            self._refill_request_sent = False
            self._refill_request_queue_size = queue_size
            self._refill_request_queue_ratio = queue_ratio
            return True

    def control_loop_observation(
        self,
        task: str,
        verbose: bool = False,
        timing: dict[str, Any] | None = None,
        force_refill: bool = False,
    ) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            robot_observation_start = time.perf_counter() if timing is not None else 0.0
            raw_observation: RawObservation = self.robot.get_observation()
            robot_observation_end = time.perf_counter() if timing is not None else 0.0
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = force_refill or (
                    self.must_go.is_set() and self.action_queue.empty()
                )
                current_queue_size = self.action_queue.qsize()
                refill_queue_size = self._refill_request_queue_size
                refill_queue_ratio = self._refill_request_queue_ratio

            if timing is not None:
                timing["robot_get_observation_ms"] = (
                    robot_observation_end - robot_observation_start
                ) * 1000

            observation_sent = self.send_observation(observation, timing=timing)
            observation_end = time.perf_counter() if timing is not None else 0.0

            if force_refill:
                if observation_sent:
                    with self.action_queue_lock:
                        # A very fast receiver may already have completed and
                        # cleared this request; never resurrect that state.
                        if self._refill_in_flight:
                            self._refill_request_sent = True
                else:
                    self._clear_refill_in_flight()
                    self.must_go.set()

            if (
                observation_sent
                and force_refill
                and self.config.timing_diagnostics
            ):
                self.logger.info(
                    "[TIMING][REFILL_REQUEST] event_wall_time=%.6f timestep=%s "
                    "queue_size=%s queue_ratio=%.3f refill_in_flight_before=False must_go=%s",
                    time.time(),
                    observation.get_timestep(),
                    refill_queue_size if refill_queue_size is not None else current_queue_size,
                    refill_queue_ratio if refill_queue_ratio is not None else 0.0,
                    observation.must_go,
                )

            if timing is not None:
                timing["control_loop_observation_ms"] = (
                    observation_end - start_time
                ) * 1000
                self.logger.debug(
                    "[TIMING][OBSERVATION] event_wall_time=%.6f timestep=%s must_go=%s "
                    "queue_size=%s robot_get_observation_ms=%.3f transport_preprocess_ms=%.3f "
                    "resize_ms=%.3f jpeg_encode_ms=%.3f serialization_ms=%.3f "
                    "grpc_send_observation_ms=%.3f transport_bytes=%s serialized_bytes=%s "
                    "control_loop_observation_ms=%.3f",
                    time.time(),
                    observation.get_timestep(),
                    observation.must_go,
                    current_queue_size,
                    timing["robot_get_observation_ms"],
                    timing.get("transport_preprocess_ms", 0.0),
                    timing.get("resize_ms", 0.0),
                    timing.get("jpeg_encode_ms", 0.0),
                    timing.get("serialization_ms", 0.0),
                    timing.get("grpc_send_observation_ms", 0.0),
                    timing.get("transport_bytes", 0),
                    timing.get("serialized_bytes", 0),
                    timing["control_loop_observation_ms"],
                )

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
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
            if force_refill:
                self._clear_refill_in_flight()
                self.must_go.set()
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

            camera_image_writer = getattr(self, "camera_image_writer", None)
            if camera_image_writer is not None:
                camera_image_writer.submit(
                    observation.get_observation(),
                    wall_time=observation.get_timestamp(),
                    timestep=observation.get_timestep(),
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

                if verbose and timed_actions:
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        with self.latest_action_lock:
                            old_timesteps = [self.latest_action]
                else:
                    old_size, old_timesteps = 0, []

                queue_update_start = time.perf_counter()
                self._ingest_action_chunk(
                    timed_actions,
                    self.config.aggregate_fn,
                    receive_wall_time=time.time(),
                    receive_monotonic_time=time.monotonic(),
                )
                queue_update_time = time.perf_counter() - queue_update_start

                # After receiving actions, the next empty queue triggers must-go processing.
                self.must_go.set()

                self.logger.debug(
                    f"{self.remote_backend_name} action request for obs #{observation.get_timestep()} "
                    f"took {request_time * 1000:.2f}ms"
                )

                if verbose and timed_actions:
                    new_size, new_timesteps = self._inspect_action_queue()
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

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

    def _policy_observation_loop(self, task: str, verbose: bool) -> None:
        """Run policy observations at policy FPS, independently of command ticks."""
        period = self.config.environment_dt
        next_tick = time.monotonic()
        while self.running:
            if self._ready_to_send_observation():
                if self.uses_grpc_backend:
                    self.control_loop_observation(task, verbose, force_refill=True)
                elif self.uses_remote_zmq_backend:
                    self.control_loop_remote_observation(task, verbose)
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay <= 0:
                # Skip missed observation slots; never issue a catch-up burst.
                next_tick = time.monotonic() + period
                delay = period
            self.shutdown_event.wait(delay)

    def _high_rate_control_loop(self, task: str, verbose: bool) -> tuple[Observation, Action]:
        self.start_barrier.wait()
        self.logger.info("High-rate trajectory control loop starting")
        observation_thread = threading.Thread(
            target=self._policy_observation_loop,
            args=(task, verbose),
            name="policy-observation-loop",
            daemon=True,
        )
        observation_thread.start()
        period = 1.0 / self.config.trajectory_postprocess.control_rate_hz
        next_tick = time.monotonic()
        performed = None
        try:
            while self.running:
                performed = self.control_loop_trajectory_action(time.monotonic()) or performed
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay <= 0:
                    # Re-anchor after a stall. There is exactly one send attempt
                    # per iteration and no accumulated deadline catch-up.
                    next_tick = time.monotonic() + period
                    delay = period
                self.shutdown_event.wait(delay)
        finally:
            observation_thread.join(timeout=1.0)
        return None, performed

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        if self.config.trajectory_postprocess.enabled:
            return self._high_rate_control_loop(task, verbose)
        # Wait at barrier for synchronized start
        if self.uses_grpc_backend or self.uses_remote_zmq_backend:
            self.start_barrier.wait()
        
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            loop_timing = {} if self.config.timing_diagnostics else None
            """Control loop: (1) Performing actions, when available"""
            actions_available_start = (
                time.perf_counter() if self.config.timing_diagnostics else 0.0
            )
            has_actions = self.actions_available()
            if loop_timing is not None:
                loop_timing["actions_available_ms"] = (
                    time.perf_counter() - actions_available_start
                ) * 1000
            if has_actions:
                action_timing = {} if loop_timing is not None else None
                _performed_action = self.control_loop_action(verbose, timing=action_timing)
                if loop_timing is not None:
                    loop_timing["action"] = action_timing

            """Control loop: (2) Streaming observations to the remote policy server"""
            ready_start = time.perf_counter() if self.config.timing_diagnostics else 0.0
            ready_to_send = self._ready_to_send_observation()
            if loop_timing is not None:
                loop_timing["ready_to_send_observation_ms"] = (
                    time.perf_counter() - ready_start
                ) * 1000
            if ready_to_send:
                if self.uses_grpc_backend:
                    observation_timing = {} if loop_timing is not None else None
                    _captured_observation = self.control_loop_observation(
                        task,
                        verbose,
                        timing=observation_timing,
                        force_refill=True,
                    )
                    if loop_timing is not None:
                        loop_timing["observation"] = observation_timing

                elif self.uses_remote_zmq_backend:
                    _captured_observation = self.control_loop_remote_observation(task, verbose)
                else:
                    raise ValueError(f"Unsupported backend: {self.backend}")

                

            loop_elapsed = time.perf_counter() - control_loop_start
            if loop_timing is not None:
                loop_timing["control_loop_action_ms"] = loop_timing.get("action", {}).get(
                    "control_loop_action_total_ms", 0.0
                )
                loop_timing["control_loop_observation_ms"] = loop_timing.get(
                    "observation", {}
                ).get("control_loop_observation_ms", 0.0)
                loop_timing["loop_total_ms"] = loop_elapsed * 1000
                loop_timing["sleep_budget_ms"] = max(
                    0, self.config.environment_dt - loop_elapsed
                ) * 1000
                self.logger.debug(
                    "[TIMING][CONTROL_LOOP] event_wall_time=%.6f loop_total_ms=%.3f "
                    "actions_available_ms=%.3f control_loop_action_ms=%.3f "
                    "ready_to_send_observation_ms=%.3f control_loop_observation_ms=%.3f "
                    "sleep_budget_ms=%.3f",
                    time.time(),
                    loop_timing["loop_total_ms"],
                    loop_timing["actions_available_ms"],
                    loop_timing["control_loop_action_ms"],
                    loop_timing["ready_to_send_observation_ms"],
                    loop_timing["control_loop_observation_ms"],
                    loop_timing["sleep_budget_ms"],
                )
                self._timing_history.append(loop_timing)

            self.logger.debug(f"Control loop (ms): {loop_elapsed * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            sleep_budget = max(
                0, self.config.environment_dt - (time.perf_counter() - control_loop_start)
            )
            if loop_timing is not None:
                # Keep the snapshot used by a later STALL summary aligned with the
                # actual sleep call, including diagnostic logging overhead.
                loop_timing["sleep_budget_ms"] = sleep_budget * 1000
            time.sleep(sleep_budget)

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
