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
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.types import PolicyAction

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .diagnostic_capture import PolicyBatchCaptureWriter
from .frozen_noise_analysis import clone_to_cpu
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)
from .image_transport import decode_observation_images
from .rtc import RTCDiagnosticsWriter, RTCRequest, RTCState, overlap_metrics


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None
        self._frozen_batch_dump_lock = threading.Lock()
        self._frozen_batch_dumped = False
        self._diagnostic_capture = (
            PolicyBatchCaptureWriter(
                config.diagnostic_capture_dir,
                config.diagnostic_capture_max,
                self.logger,
            )
            if config.diagnostic_capture_policy_batches
            else None
        )

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.timing_diagnostics = False
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None
        self.pretrained_name_or_path: str | None = None
        self.rtc_enabled = False
        self.rtc_mode = "guided"
        self.rtc_execution_horizon = 10
        self.rtc_max_guidance_weight = 10.0
        self.rtc_prefix_attention_schedule = "EXP"
        self._rtc_state: RTCState | None = None
        self._rtc_diagnostics: RTCDiagnosticsWriter | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

        self.timing_diagnostics = False
        self._rtc_state = None
        if self._rtc_diagnostics is not None:
            self._rtc_diagnostics.close()
            self._rtc_diagnostics = None

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self.pretrained_name_or_path = policy_specs.pretrained_name_or_path
        self.timing_diagnostics = getattr(policy_specs, "timing_diagnostics", False)
        if self.timing_diagnostics:
            self.logger.info("[TIMING] Server diagnostics enabled by client policy setup")

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

        self.rtc_enabled = bool(getattr(policy_specs, "rtc_enabled", False))
        self.rtc_mode = str(getattr(policy_specs, "rtc_mode", "guided")).lower()
        self.rtc_execution_horizon = int(getattr(policy_specs, "rtc_execution_horizon", 10))
        self.rtc_max_guidance_weight = float(
            getattr(policy_specs, "rtc_max_guidance_weight", 10.0)
        )
        self.rtc_prefix_attention_schedule = str(
            getattr(policy_specs, "rtc_prefix_attention_schedule", "EXP")
        ).upper()
        if self.rtc_enabled:
            if self.policy_type != "smolvla" or self.rtc_mode != "guided":
                raise ValueError(
                    "Guided RTC is supported only for policy_type=smolvla and rtc_mode=guided"
                )
            from lerobot.configs import RTCAttentionSchedule
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            rtc_config = RTCConfig(
                enabled=True,
                prefix_attention_schedule=RTCAttentionSchedule[
                    self.rtc_prefix_attention_schedule
                ],
                max_guidance_weight=self.rtc_max_guidance_weight,
                execution_horizon=self.rtc_execution_horizon,
            )
            # Runtime-only injection. from_pretrained has already loaded the checkpoint,
            # and no checkpoint/config files are written.
            self.policy.config.rtc_config = rtc_config
            self.policy.init_rtc_processor()
            self._rtc_state = RTCState(self.config.fps)
            self._rtc_diagnostics = RTCDiagnosticsWriter(
                getattr(policy_specs, "rtc_diagnostics_dir", "outputs")
            )
            self.logger.info(
                "[RTC] enabled=True mode=%s execution_horizon=%d "
                "max_guidance_weight=%.1f schedule=%s diagnostics=%s",
                self.rtc_mode,
                self.rtc_execution_horizon,
                self.rtc_max_guidance_weight,
                self.rtc_prefix_attention_schedule,
                self._rtc_diagnostics.path,
            )

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # event correlation only; never used for a duration
        receive_start = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        receive_end = time.perf_counter() if self.timing_diagnostics else 0.0
        observation_received_wall_time = (
            time.time() if self.timing_diagnostics else receive_time
        )

        start_deserialize = time.perf_counter() if self.timing_diagnostics else receive_start
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        transport_decode_start = time.perf_counter() if self.timing_diagnostics else 0.0
        decoded_observation, image_stats = decode_observation_images(timed_observation.get_observation())
        transport_decode_time = (
            time.perf_counter() - transport_decode_start if self.timing_diagnostics else 0.0
        )
        if image_stats.image_count:
            timed_observation.observation = decoded_observation
            self.logger.debug(
                "Image transport decode: images=%d | JPEG decode=%.3fms | restore resize=%.3fms | "
                "total=%.3fms",
                image_stats.image_count,
                image_stats.jpeg_decode_time * 1000,
                image_stats.restore_resize_time * 1000,
                image_stats.total_time * 1000,
            )

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"Client event wall time: {obs_timestamp:.6f}"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        if self.timing_diagnostics:
            self.logger.debug(
                "[TIMING][SERVER_OBSERVATION] event_wall_time=%.6f rpc_start_wall_time=%.6f "
                "chunk_id=%s "
                "grpc_receive_ms=%.3f pickle_deserialize_ms=%.3f "
                "server_transport_decode_ms=%.3f jpeg_decode_ms=%.3f "
                "restore_resize_ms=%.3f received_bytes=%s",
                observation_received_wall_time,
                receive_time,
                obs_timestep,
                (receive_end - receive_start) * 1000,
                deserialize_time * 1000,
                transport_decode_time * 1000,
                image_stats.jpeg_decode_time * 1000,
                image_stats.restore_resize_time * 1000,
                len(received_bytes),
            )

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            observation_queue_wait_ms = (
                (time.perf_counter() - getactions_starts) * 1000
                if self.timing_diagnostics
                else 0.0
            )
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            policy_timing = {} if self.timing_diagnostics else None
            action_chunk = self._predict_action_chunk(obs, timing=policy_timing)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)
            chunk_ready_wall_time = time.time() if self.timing_diagnostics else 0.0

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            response_delay = max(
                0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts)
            )
            time.sleep(response_delay)  # sleep controls inference latency

            if self.timing_diagnostics:
                chunk_first = action_chunk[0].get_timestep() if action_chunk else None
                chunk_last = action_chunk[-1].get_timestep() if action_chunk else None
                self.logger.info(
                    "[TIMING][SERVER_CHUNK] event_wall_time=%.6f chunk_ready_wall_time=%.6f "
                    "chunk_id=%s chunk_first=%s chunk_last=%s observation_queue_wait_ms=%.3f "
                    "server_raw_observation_prepare_ms=%.3f server_policy_preprocess_ms=%.3f "
                    "server_inference_ms=%.3f server_policy_postprocess_ms=%.3f "
                    "server_action_chunk_build_ms=%.3f server_action_serialize_ms=%.3f "
                    "server_total_policy_ms=%.3f response_delay_ms=%.3f get_actions_total_ms=%.3f",
                    time.time(),
                    chunk_ready_wall_time,
                    obs.get_timestep(),
                    chunk_first,
                    chunk_last,
                    observation_queue_wait_ms,
                    policy_timing["server_raw_observation_prepare_ms"],
                    policy_timing["server_policy_preprocess_ms"],
                    policy_timing["server_inference_ms"],
                    policy_timing["server_policy_postprocess_ms"],
                    policy_timing["server_action_chunk_build_ms"],
                    serialize_time * 1000,
                    policy_timing["server_total_policy_ms"],
                    response_delay * 1000,
                    (time.perf_counter() - getactions_starts) * 1000,
                )

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(
        self,
        observation: dict[str, torch.Tensor],
        rtc_request: RTCRequest | None = None,
    ) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        if rtc_request is not None and rtc_request.applied:
            chunk = self.policy.predict_action_chunk(
                observation,
                inference_delay=rtc_request.inference_delay_frames,
                prev_chunk_left_over=rtc_request.prefix,
                execution_horizon=self.rtc_execution_horizon,
            )
        else:
            # Preserve the exact pre-RTC call path when disabled or on the first request.
            chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _dump_frozen_policy_batch_once(
        self, observation: dict[str, Any], *, timestep: int
    ) -> None:
        """Save one exact post-preprocessor batch without mutating inference input."""
        dump_path = self.config.dump_frozen_policy_batch
        if dump_path is None or self._frozen_batch_dumped:
            return

        with self._frozen_batch_dump_lock:
            if self._frozen_batch_dumped:
                return

            path = Path(dump_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            frozen_batch = clone_to_cpu(observation)
            torch.save(frozen_batch, path)
            self._frozen_batch_dumped = True
            self.logger.info(
                "[FROZEN_OBS] saved policy-ready batch | path=%s | timestep=%s | keys=%s",
                path,
                timestep,
                sorted(map(str, observation.keys())),
            )

    def _predict_action_chunk(
        self,
        observation_t: TimedObservation,
        timing: dict[str, float] | None = None,
    ) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        diagnostic_request_kind = (
            "initial" if self.last_processed_obs is None else "refill"
        ) if self._diagnostic_capture is not None else None
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        # This is the exact object shape/content passed to SmolVLA below. The
        # helper recursively clones tensors to CPU, so neither this batch nor
        # policy inference is changed by the diagnostic capture.
        if self.config.dump_frozen_policy_batch is not None:
            self._dump_frozen_policy_batch_once(
                observation, timestep=observation_t.get_timestep()
            )

        prepared_capture = None
        if self._diagnostic_capture is not None:
            task = observation.get("task")
            if isinstance(task, (list, tuple)) and len(task) == 1:
                task = task[0]
            if not isinstance(task, (str, int, float, bool, type(None))):
                task = repr(task)
            prepared_capture = self._diagnostic_capture.prepare(
                observation,
                {
                    "wall_time": time.time(),
                    "policy_timestamp": observation_t.get_timestamp(),
                    "timestep": observation_t.get_timestep(),
                    "must_go": observation_t.must_go,
                    "request_kind": diagnostic_request_kind,
                    "initial_request": diagnostic_request_kind == "initial",
                    "refill_request": diagnostic_request_kind == "refill",
                    # The client execution queue is not transported to the
                    # server; report that explicitly instead of guessing.
                    "queue_size": None,
                    "executed_timestep": None,
                    "server_observation_queue_size": self.observation_queue.qsize(),
                    "task": task,
                    "checkpoint": self.pretrained_name_or_path,
                    "num_steps": getattr(self.policy.config, "num_steps", None),
                    "fps": self.config.fps,
                    "actions_per_chunk": self.actions_per_chunk,
                },
            )

        """3. Get action chunk"""
        rtc_request = (
            self._rtc_state.prepare(observation_t.get_timestep())
            if self.rtc_enabled and self._rtc_state is not None
            else None
        )
        if rtc_request is not None:
            if rtc_request.applied:
                self.logger.info(
                    "[RTC] guided: prefix_len=%d inference_delay=%d shift=%d",
                    rtc_request.prefix.shape[0],
                    rtc_request.inference_delay_frames,
                    rtc_request.shift,
                )
            else:
                self.logger.info("[RTC] bypass: %s", rtc_request.bypass_reason)
        inference_start_wall_time = time.time() if prepared_capture is not None else None
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation, rtc_request=rtc_request)
        inference_time = time.perf_counter() - start_inference
        inference_end_wall_time = time.time() if prepared_capture is not None else None
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape
        raw_capture_clone_started = time.perf_counter() if prepared_capture is not None else 0.0
        raw_action_tensor_for_capture = None
        raw_action_tensor_for_rtc = None
        if prepared_capture is not None:
            raw_action_tensor_for_capture = action_tensor.squeeze(0).detach().clone()
        if self.rtc_enabled:
            raw_action_tensor_for_rtc = action_tensor.squeeze(0).detach().clone()
        raw_capture_clone_ms = (
            (time.perf_counter() - raw_capture_clone_started) * 1000
            if prepared_capture is not None
            else 0.0
        )

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        robot_action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        action_tensor = robot_action_tensor
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()
        if self.rtc_enabled and not bool(torch.isfinite(action_tensor).all()):
            raise RuntimeError("Policy produced NaN or Inf; refusing to return an action chunk")
        if self.rtc_enabled and self._rtc_state is not None:
            if raw_action_tensor_for_rtc is None or not bool(
                torch.isfinite(raw_action_tensor_for_rtc).all()
            ):
                raise RuntimeError("RTC produced NaN or Inf; refusing to return an action chunk")
            metrics = overlap_metrics(
                rtc_request.previous_robot_leftover if rtc_request is not None else None,
                action_tensor,
            )
            self._rtc_state.complete(
                raw_chunk=raw_action_tensor_for_rtc,
                robot_chunk=action_tensor,
                timestep=observation_t.get_timestep(),
                latency_s=inference_time,
            )
            record = {
                "request_id": observation_t.get_timestep(),
                "policy_timestamp": observation_t.get_timestamp(),
                "rtc_enabled": True,
                "rtc_applied": bool(rtc_request and rtc_request.applied),
                "rtc_bypass_reason": rtc_request.bypass_reason if rtc_request else None,
                "mode": self.rtc_mode,
                "execution_horizon": self.rtc_execution_horizon,
                "max_guidance_weight": self.rtc_max_guidance_weight,
                "prefix_schedule": self.rtc_prefix_attention_schedule,
                "inference_latency_ms": inference_time * 1000,
                "inference_delay_frames": (
                    rtc_request.inference_delay_frames if rtc_request else 0
                ),
                "delay_frames_float": rtc_request.delay_frames_float if rtc_request else 0.0,
                "delay_estimator_ready": bool(
                    rtc_request and rtc_request.delay_estimator_ready
                ),
                "delay_estimator_window_count": (
                    rtc_request.delay_estimator_window_count if rtc_request else 0
                ),
                "delay_estimator_window_max_ms": (
                    rtc_request.delay_estimator_window_max_ms if rtc_request else None
                ),
                "prev_leftover_count": (
                    int(rtc_request.prefix.shape[0])
                    if rtc_request is not None and rtc_request.prefix is not None
                    else 0
                ),
                "previous_chunk_shift": rtc_request.shift if rtc_request else 0,
                **metrics,
            }
            self.logger.info(
                "[RTC] enabled=True mode=%s execution_horizon=%d max_guidance_weight=%.1f "
                "schedule=%s inference_latency_ms=%.3f inference_delay_frames=%d "
                "delay_frames_float=%.3f delay_estimator_ready=%s "
                "delay_estimator_window_count=%d delay_estimator_window_max_ms=%s "
                "prev_chunk_left_over_length=%d rtc_applied=%s rtc_bypass_reason=%s",
                self.rtc_mode,
                self.rtc_execution_horizon,
                self.rtc_max_guidance_weight,
                self.rtc_prefix_attention_schedule,
                inference_time * 1000,
                record["inference_delay_frames"],
                record["delay_frames_float"],
                record["delay_estimator_ready"],
                record["delay_estimator_window_count"],
                record["delay_estimator_window_max_ms"],
                record["prev_leftover_count"],
                record["rtc_applied"],
                record["rtc_bypass_reason"],
            )
            if self._rtc_diagnostics is not None:
                self._rtc_diagnostics.write(record)
        if prepared_capture is not None:
            self._diagnostic_capture.submit(
                prepared_capture,
                raw_action_tensor_for_capture,
                action_tensor,
                {
                    "inference_start_time": inference_start_wall_time,
                    "inference_end_time": inference_end_wall_time,
                    "inference_latency_ms": inference_time * 1000,
                    "raw_preserve_clone_ms": raw_capture_clone_ms,
                },
            )
        policy_postprocess_end = time.perf_counter() if timing is not None else 0.0

        """5. Convert to TimedAction list"""
        chunk_build_start = time.perf_counter() if timing is not None else 0.0
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        if timing is not None:
            timing.update(
                server_raw_observation_prepare_ms=prepare_time * 1000,
                server_policy_preprocess_ms=preprocessing_time * 1000,
                server_inference_ms=inference_time * 1000,
                server_policy_postprocess_ms=(policy_postprocess_end - start_postprocess) * 1000,
                server_action_chunk_build_ms=(postprocess_stops - chunk_build_start) * 1000,
                server_total_policy_ms=(postprocess_stops - start_prepare) * 1000,
            )

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        if self._diagnostic_capture is not None:
            self._diagnostic_capture.close()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    try:
        server.wait_for_termination()
    finally:
        policy_server.stop()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
