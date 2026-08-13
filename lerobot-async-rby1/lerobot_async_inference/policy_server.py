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
python -m lerobot_async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1 \
     --debug_observation_lifecycle=true
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import asdict
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
from .observation_lifecycle import ObservationLifecycleLogger


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)
    _processed_prefetch_history_size = 256

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()
        self._prefetch_request_ids_lock = threading.Lock()
        self._recent_prefetch_request_ids = deque()
        self._recent_prefetch_request_id_set = set()

        self.last_processed_obs = None

        # Debug-only sidecar state. No lifecycle objects or writer thread exist when disabled.
        self.observation_lifecycle = (
            ObservationLifecycleLogger(
                config.observation_lifecycle_log_dir,
                queue_maxsize=config.observation_lifecycle_logger_queue_maxsize,
            )
            if config.debug_observation_lifecycle
            else None
        )
        self._active_inference_observation_id = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.transport_image_scale = 1.0
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

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

        with self._prefetch_request_ids_lock:
            self._recent_prefetch_request_ids.clear()
            self._recent_prefetch_request_id_set.clear()

    @staticmethod
    def _prefetch_metadata(obs: TimedObservation) -> tuple[bool, int | None]:
        requested = bool(getattr(obs, "prefetch_requested", False))
        request_id = getattr(obs, "prefetch_request_id", None)
        valid_request_id = isinstance(request_id, int) and not isinstance(request_id, bool) and request_id > 0
        return requested and valid_request_id, request_id if valid_request_id else None

    def _reserve_prefetch_request_id(self, request_id: int) -> bool:
        """Reserve a bounded request ID before enqueue so duplicates cannot reach inference."""
        with self._prefetch_request_ids_lock:
            if request_id in self._recent_prefetch_request_id_set:
                return False

            if len(self._recent_prefetch_request_ids) >= self._processed_prefetch_history_size:
                expired_request_id = self._recent_prefetch_request_ids.popleft()
                self._recent_prefetch_request_id_set.remove(expired_request_id)

            self._recent_prefetch_request_ids.append(request_id)
            self._recent_prefetch_request_id_set.add(request_id)
            return True

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
            f"Device: {policy_specs.device} | "
            f"Transport image scale: {getattr(policy_specs, 'transport_image_scale', 1.0)}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self.transport_image_scale = getattr(policy_specs, "transport_image_scale", 1.0)

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

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

        if self.observation_lifecycle is not None:
            request_stream_start_monotonic = time.perf_counter()
            inference_busy_at_stream_start = self._active_inference_observation_id is not None
            active_inference_at_stream_start = self._active_inference_observation_id
        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        try:
            received_bytes = receive_bytes_in_chunks(
                request_iterator, None, self.shutdown_event, self.logger
            )  # blocking call while looping over request_iterator
            if self.observation_lifecycle is not None:
                payload_received_monotonic = time.perf_counter()
            timed_observation = pickle.loads(received_bytes)  # nosec
        except Exception as e:
            if self.observation_lifecycle is not None:
                self.observation_lifecycle.record_global(
                    "observation_receive_error",
                    stage="receive_or_deserialize",
                    exception_type=type(e).__name__,
                    short_error_message=str(e)[:500],
                    request_stream_start_monotonic_time=request_stream_start_monotonic,
                )
            raise
        deserialize_time = time.perf_counter() - start_deserialize

        if self.observation_lifecycle is not None:
            observation_received_monotonic = time.perf_counter()
            self.observation_lifecycle.register(
                timed_observation,
                receive_monotonic_time=observation_received_monotonic,
                client_id=client_id,
                client_timestamp=timed_observation.get_timestamp(),
                server_receive_wall_time_unix=receive_time,
                server_configured_fps=self.config.fps,
                serialized_payload_bytes=len(received_bytes),
                request_stream_start_monotonic_time=request_stream_start_monotonic,
                payload_received_monotonic_time=payload_received_monotonic,
                deserialize_end_monotonic_time=observation_received_monotonic,
                grpc_payload_receive_ms=(
                    payload_received_monotonic - request_stream_start_monotonic
                )
                * 1000,
                deserialization_ms=(
                    observation_received_monotonic - payload_received_monotonic
                )
                * 1000,
                receive_and_deserialize_ms=deserialize_time * 1000,
                inference_busy_at_receive=inference_busy_at_stream_start,
                active_inference_observation_id=active_inference_at_stream_start,
                inference_busy_after_deserialize=self._active_inference_observation_id is not None,
                active_inference_observation_id_after_deserialize=(
                    self._active_inference_observation_id
                ),
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
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
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

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        obs = None
        error_stage = "observation_dequeue"
        try:
            getactions_starts = time.perf_counter()
            if self.observation_lifecycle is not None:
                queue_size_before_dequeue = self.observation_queue.qsize()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            if self.observation_lifecycle is not None:
                dequeue_time = time.perf_counter()
                queue_size_after_dequeue = self.observation_queue.qsize()
                source_prefetch_request_id = self._prefetch_metadata(obs)[1]
                inference_trigger_reason = self.observation_lifecycle.summary_value(
                    obs, "inference_trigger_reason"
                )
                receive_monotonic_time = self.observation_lifecycle.summary_value(
                    obs, "receive_monotonic_time"
                )
                receive_to_dequeue_ms = (
                    (dequeue_time - receive_monotonic_time) * 1000
                    if receive_monotonic_time is not None
                    else None
                )
                self.observation_lifecycle.record(
                    obs,
                    "observation_dequeued_for_processing",
                    summary_updates={
                        "dequeued_for_processing": True,
                        "selected_for_inference": True,
                        "dequeue_monotonic_time": dequeue_time,
                        "receive_to_dequeue_ms": receive_to_dequeue_ms,
                    },
                    queue_size_before_dequeue=queue_size_before_dequeue,
                    queue_size_after_dequeue=queue_size_after_dequeue,
                    dequeue_monotonic_time=dequeue_time,
                    receive_to_dequeue_ms=receive_to_dequeue_ms,
                    must_go=obs.must_go,
                    source_prefetch_request_id=source_prefetch_request_id,
                    inference_trigger_reason=inference_trigger_reason,
                    current_disposition="selected_for_inference",
                )
                self.observation_lifecycle.record(
                    obs,
                    "observation_selected_for_inference",
                    must_go=obs.must_go,
                    source_prefetch_request_id=source_prefetch_request_id,
                    inference_trigger_reason=inference_trigger_reason,
                )

            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            if self.observation_lifecycle is not None:
                observation_id = self.observation_lifecycle.observation_id(obs)
                self._active_inference_observation_id = observation_id
            error_stage = "policy_processing"
            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time
            if (
                self.observation_lifecycle is not None
                and self._active_inference_observation_id == observation_id
            ):
                self._active_inference_observation_id = None

            error_stage = "response_serialization"
            start_time = time.perf_counter()
            if self.observation_lifecycle is not None:
                self.observation_lifecycle.record(
                    obs,
                    "response_serialize_start",
                    response_serialize_start_monotonic_time=start_time,
                    source_prefetch_request_id=self._prefetch_metadata(obs)[1],
                )
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            response_serialize_end = time.perf_counter()
            serialize_time = response_serialize_end - start_time
            if self.observation_lifecycle is not None:
                self.observation_lifecycle.record(
                    obs,
                    "response_serialize_end",
                    summary_updates={
                        "response_serialize_start_monotonic_time": start_time,
                        "response_serialize_end_monotonic_time": response_serialize_end,
                        "response_serialization_ms": serialize_time * 1000,
                    },
                    response_serialize_start_monotonic_time=start_time,
                    response_serialize_end_monotonic_time=response_serialize_end,
                    response_serialization_ms=serialize_time * 1000,
                    serialized_response_bytes=len(actions_bytes),
                    source_prefetch_request_id=self._prefetch_metadata(obs)[1],
                )

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

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

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            if self.observation_lifecycle is not None:
                response_return_time = time.perf_counter()
                receive_monotonic_time = self.observation_lifecycle.summary_value(
                    obs, "receive_monotonic_time"
                )
                model_inference_end = self.observation_lifecycle.summary_value(
                    obs, "model_inference_end_monotonic_time"
                )
                chunk_ready_time = self.observation_lifecycle.summary_value(
                    obs, "chunk_ready_monotonic_time"
                )
                response_metrics = {
                    "response_return_monotonic_time": response_return_time,
                    "inference_end_to_response_return_ms": (
                        (response_return_time - model_inference_end) * 1000
                        if model_inference_end is not None
                        else None
                    ),
                    "chunk_ready_to_response_return_ms": (
                        (response_return_time - chunk_ready_time) * 1000
                        if chunk_ready_time is not None
                        else None
                    ),
                    "receive_to_response_return_ms": (
                        (response_return_time - receive_monotonic_time) * 1000
                        if receive_monotonic_time is not None
                        else None
                    ),
                    "response_returned": True,
                    "source_prefetch_request_id": self._prefetch_metadata(obs)[1],
                }
                self.observation_lifecycle.record(
                    obs,
                    "response_return",
                    summary_updates=response_metrics,
                    **response_metrics,
                )
                self.observation_lifecycle.finalize(obs, "response_returned")

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self._active_inference_observation_id = None
            if self.observation_lifecycle is not None:
                if obs is None:
                    self.observation_lifecycle.record_global(
                        "observation_processing_error",
                        stage=error_stage,
                        exception_type=type(e).__name__,
                        short_error_message=str(e)[:500],
                    )
                else:
                    self.observation_lifecycle.record(
                        obs,
                        "observation_processing_error",
                        stage=error_stage,
                        exception_type=type(e).__name__,
                        short_error_message=str(e)[:500],
                    )
                    self.observation_lifecycle.finalize(
                        obs,
                        "error",
                        error_stage=error_stage,
                        exception_type=type(e).__name__,
                        short_error_message=str(e)[:500],
                    )
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            if self.observation_lifecycle is not None:
                queue_size = self.observation_queue.qsize()
                self.observation_lifecycle.record(
                    obs,
                    "observation_filtered_already_predicted",
                    summary_updates={
                        "enqueue_attempted": False,
                        "enqueue_success": False,
                        "filter_reason": "timestep_already_predicted",
                    },
                    observation_queue_size_before=queue_size,
                    observation_queue_size_after=queue_size,
                    observation_queue_maxsize=self.observation_queue.maxsize,
                    enqueue_attempted=False,
                    enqueue_success=False,
                )
            return False

        observations_similar_result = observations_similar(
            obs, previous_obs, lerobot_features=self.lerobot_features
        )
        if self.observation_lifecycle is not None:
            compared_with_observation_id = self.observation_lifecycle.observation_id(previous_obs)
            self.observation_lifecycle.record(
                obs,
                "observation_similarity_checked",
                summary_updates={
                    "similarity_check_performed": True,
                    "observations_similar_result": observations_similar_result,
                    "compared_with_observation_id": compared_with_observation_id,
                    "compared_with_observation_timestep": previous_obs.get_timestep(),
                },
                similarity_check_performed=True,
                observations_similar_result=observations_similar_result,
                compared_with_observation_id=compared_with_observation_id,
                compared_with_observation_timestep=previous_obs.get_timestep(),
            )

        if observations_similar_result:
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            if self.observation_lifecycle is not None:
                queue_size = self.observation_queue.qsize()
                self.observation_lifecycle.record(
                    obs,
                    "observation_filtered_as_similar",
                    summary_updates={
                        "similarity_filtered": True,
                        "enqueue_attempted": False,
                        "enqueue_success": False,
                        "filter_reason": "observations_similar",
                    },
                    observation_queue_size_before=queue_size,
                    observation_queue_size_after=queue_size,
                    observation_queue_maxsize=self.observation_queue.maxsize,
                    enqueue_attempted=False,
                    enqueue_success=False,
                )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if self.observation_lifecycle is not None:
            self.observation_lifecycle.ensure_registered(obs)

        prefetch_requested = bool(getattr(obs, "prefetch_requested", False))
        is_valid_prefetch, prefetch_request_id = self._prefetch_metadata(obs)
        similarity_bypassed_for_prefetch = False
        predicted_timestep_check_bypassed_for_prefetch = False

        if prefetch_requested and not is_valid_prefetch:
            self.logger.warning(
                "Observation #%s requested prefetch without a valid positive integer request ID; "
                "using normal scheduling",
                obs.get_timestep(),
            )
            if self.observation_lifecycle is not None:
                self.observation_lifecycle.record(
                    obs,
                    "prefetch_invalid_metadata",
                    prefetch_metadata_valid=False,
                )

        if is_valid_prefetch and not self._reserve_prefetch_request_id(prefetch_request_id):
            self.logger.warning(
                "Ignoring duplicate prefetch request id=%s for observation #%s",
                prefetch_request_id,
                obs.get_timestep(),
            )
            if self.observation_lifecycle is not None:
                queue_size = self.observation_queue.qsize()
                self.observation_lifecycle.record(
                    obs,
                    "duplicate_prefetch_request",
                    summary_updates={
                        "enqueue_attempted": False,
                        "enqueue_success": False,
                        "inference_trigger_reason": "duplicate_prefetch_request",
                    },
                    observation_queue_size_before=queue_size,
                    observation_queue_size_after=queue_size,
                    observation_queue_maxsize=self.observation_queue.maxsize,
                    enqueue_attempted=False,
                    enqueue_success=False,
                )
                self.observation_lifecycle.finalize(obs, "duplicate_prefetch_request")
            return False

        if is_valid_prefetch:
            should_enqueue = True
            must_go_reason = None
            inference_trigger_reason = "prefetch"
            similarity_bypassed_for_prefetch = True
            predicted_timestep_check_bypassed_for_prefetch = True
            if self.observation_lifecycle is not None:
                self.observation_lifecycle.record(
                    obs,
                    "observation_filters_bypassed_for_prefetch",
                    summary_updates={
                        "similarity_bypassed_for_prefetch": True,
                        "predicted_timestep_check_bypassed_for_prefetch": True,
                    },
                    similarity_check_performed=False,
                    similarity_bypassed_for_prefetch=True,
                    predicted_timestep_check_bypassed_for_prefetch=True,
                )
        elif obs.must_go:
            should_enqueue = True
            must_go_reason = "client_requested"
            inference_trigger_reason = "must_go"
            if self.observation_lifecycle is not None:
                self.observation_lifecycle.record(
                    obs,
                    "observation_similarity_check_skipped",
                    similarity_check_performed=False,
                    similarity_check_skipped_reason="must_go",
                )
        elif self.last_processed_obs is None:
            should_enqueue = True
            must_go_reason = None
            inference_trigger_reason = "initial"
            if self.observation_lifecycle is not None:
                self.observation_lifecycle.record(
                    obs,
                    "observation_similarity_check_skipped",
                    similarity_check_performed=False,
                    similarity_check_skipped_reason="no_previous_processed_observation",
                )
        else:
            should_enqueue = self._obs_sanity_checks(obs, self.last_processed_obs)
            must_go_reason = None
            inference_trigger_reason = "normal_similarity_pass" if should_enqueue else None

        if self.observation_lifecycle is not None:
            self.observation_lifecycle.record(
                obs,
                "observation_scheduling_decision",
                summary_updates={
                    "must_go": obs.must_go,
                    "must_go_reason": must_go_reason,
                    "similarity_bypassed_for_prefetch": similarity_bypassed_for_prefetch,
                    "predicted_timestep_check_bypassed_for_prefetch": (
                        predicted_timestep_check_bypassed_for_prefetch
                    ),
                    "inference_trigger_reason": inference_trigger_reason,
                },
                must_go=obs.must_go,
                must_go_reason=must_go_reason,
                similarity_bypassed_for_prefetch=similarity_bypassed_for_prefetch,
                predicted_timestep_check_bypassed_for_prefetch=(
                    predicted_timestep_check_bypassed_for_prefetch
                ),
                inference_trigger_reason=inference_trigger_reason,
                enqueue_selected=should_enqueue,
            )

            if not should_enqueue:
                filter_reason = self.observation_lifecycle.summary_value(obs, "filter_reason")
                disposition = (
                    "filtered_already_predicted"
                    if filter_reason == "timestep_already_predicted"
                    else "filtered_as_similar"
                )
                self.observation_lifecycle.finalize(obs, disposition)

        if should_enqueue:
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            queue_size_before = (
                self.observation_queue.qsize() if self.observation_lifecycle is not None else None
            )
            dropped_observation = None
            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                dropped_observation = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")
                if self.observation_lifecycle is not None:
                    replacement_observation_id = self.observation_lifecycle.observation_id(obs)
                    dropped_observation_id = self.observation_lifecycle.observation_id(
                        dropped_observation
                    )
                    self.observation_lifecycle.record(
                        dropped_observation,
                        "observation_overwritten",
                        summary_updates={
                            "overwritten": True,
                            "overwritten_by_observation_id": replacement_observation_id,
                        },
                        overwritten_by_observation_id=replacement_observation_id,
                        dropped_observation_id=dropped_observation_id,
                        replacement_observation_id=replacement_observation_id,
                        observation_queue_maxsize=self.observation_queue.maxsize,
                    )
                    self.observation_lifecycle.finalize(
                        dropped_observation,
                        "overwritten_before_processing",
                        overwritten_by_observation_id=replacement_observation_id,
                    )

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            if self.observation_lifecycle is not None:
                queue_size_after = self.observation_queue.qsize()
                dropped_observation_id = (
                    self.observation_lifecycle.observation_id(dropped_observation)
                    if dropped_observation is not None
                    else None
                )
                self.observation_lifecycle.record(
                    obs,
                    "observation_queued",
                    summary_updates={
                        "queued": True,
                        "enqueue_attempted": True,
                        "enqueue_success": True,
                        "observation_queue_size_before": queue_size_before,
                        "observation_queue_size_after": queue_size_after,
                        "observation_queue_maxsize": self.observation_queue.maxsize,
                    },
                    observation_queue_size_before=queue_size_before,
                    observation_queue_size_after=queue_size_after,
                    observation_queue_maxsize=self.observation_queue.maxsize,
                    enqueue_attempted=True,
                    enqueue_success=True,
                    dropped_observation_id=dropped_observation_id,
                    replacement_observation_id=(
                        self.observation_lifecycle.observation_id(obs)
                        if dropped_observation is not None
                        else None
                    ),
                    current_disposition="queued",
                )
            return True

        return False

    def _time_action_chunk(
        self,
        t_0: float,
        action_chunk: list[torch.Tensor],
        i_0: int,
        source_prefetch_request_id: int | None = None,
    ) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(
                timestamp=t_0 + i * self.config.environment_dt,
                timestep=i_0 + i,
                action=action,
                source_prefetch_request_id=source_prefetch_request_id,
            )
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        if self.observation_lifecycle is not None:
            self.observation_lifecycle.record(
                observation_t,
                "policy_processing_start",
                summary_updates={"policy_processing_start_monotonic_time": start_prepare},
                policy_processing_start_monotonic_time=start_prepare,
            )
            self.observation_lifecycle.record(
                observation_t,
                "observation_preparation_start",
                preparation_start_monotonic_time=start_prepare,
            )
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            resize_images=self.transport_image_scale == 1.0,
        )
        preparation_end = time.perf_counter()
        prepare_time = preparation_end - start_prepare
        if self.observation_lifecycle is not None:
            self.observation_lifecycle.record(
                observation_t,
                "observation_preparation_end",
                summary_updates={"observation_preparation_ms": prepare_time * 1000},
                preparation_start_monotonic_time=start_prepare,
                preparation_end_monotonic_time=preparation_end,
                observation_preparation_ms=prepare_time * 1000,
            )

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        if self.observation_lifecycle is not None:
            self.observation_lifecycle.record(
                observation_t,
                "preprocessing_start",
                preprocessing_start_monotonic_time=start_preprocess,
            )
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_end = time.perf_counter()
        preprocessing_time = preprocessing_end - start_preprocess
        if self.observation_lifecycle is not None:
            self.observation_lifecycle.record(
                observation_t,
                "preprocessing_end",
                summary_updates={
                    "preprocessing_start_monotonic_time": start_preprocess,
                    "preprocessing_end_monotonic_time": preprocessing_end,
                    "preprocessing_ms": preprocessing_time * 1000,
                },
                preprocessing_start_monotonic_time=start_preprocess,
                preprocessing_end_monotonic_time=preprocessing_end,
                preprocessing_ms=preprocessing_time * 1000,
            )

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        if self.observation_lifecycle is not None:
            dequeue_time = self.observation_lifecycle.summary_value(
                observation_t, "dequeue_monotonic_time"
            )
            dequeue_to_inference_start_ms = (
                (start_inference - dequeue_time) * 1000 if dequeue_time is not None else None
            )
            self.observation_lifecycle.record(
                observation_t,
                "inference_start",
                summary_updates={
                    "model_inference_start_monotonic_time": start_inference,
                    "dequeue_to_inference_start_ms": dequeue_to_inference_start_ms,
                },
                model_inference_start_monotonic_time=start_inference,
                dequeue_to_inference_start_ms=dequeue_to_inference_start_ms,
            )
            self.observation_lifecycle.record(
                observation_t,
                "model_inference_start",
                model_inference_start_monotonic_time=start_inference,
            )
        action_tensor = self._get_action_chunk(observation)
        model_inference_end = time.perf_counter()
        inference_time = model_inference_end - start_inference
        if self.observation_lifecycle is not None:
            inference_metrics = {
                "model_inference_start_monotonic_time": start_inference,
                "model_inference_end_monotonic_time": model_inference_end,
                "model_inference_ms": inference_time * 1000,
                "inference_duration_ms": inference_time * 1000,
            }
            self.observation_lifecycle.record(
                observation_t,
                "model_inference_end",
                summary_updates=inference_metrics,
                **inference_metrics,
            )
            self.observation_lifecycle.record(
                observation_t,
                "inference_end",
                model_inference_start_monotonic_time=start_inference,
                model_inference_end_monotonic_time=model_inference_end,
                inference_duration_ms=inference_time * 1000,
            )
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        if self.observation_lifecycle is not None:
            self.observation_lifecycle.record(
                observation_t,
                "postprocessing_start",
                postprocessing_start_monotonic_time=start_postprocess,
            )
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        """5. Convert to TimedAction list"""
        source_prefetch_request_id = self._prefetch_metadata(observation_t)[1]
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(),
            list(action_tensor),
            observation_t.get_timestep(),
            source_prefetch_request_id=source_prefetch_request_id,
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess
        if self.observation_lifecycle is not None:
            self.observation_lifecycle.record(
                observation_t,
                "postprocessing_end",
                summary_updates={
                    "postprocessing_start_monotonic_time": start_postprocess,
                    "postprocessing_end_monotonic_time": postprocess_stops,
                    "postprocessing_ms": postprocessing_time * 1000,
                },
                postprocessing_start_monotonic_time=start_postprocess,
                postprocessing_end_monotonic_time=postprocess_stops,
                postprocessing_ms=postprocessing_time * 1000,
            )

            action_chunk_size = len(action_chunk)
            chunk_ready_metrics = {
                "source_observation_id": self.observation_lifecycle.observation_id(observation_t),
                "source_observation_timestep": observation_t.get_timestep(),
                "source_prefetch_request_id": source_prefetch_request_id,
                "policy_timestamp": observation_t.get_timestamp(),
                "action_chunk_size": action_chunk_size,
                "action_chunk_first_timestep": (
                    action_chunk[0].get_timestep() if action_chunk_size else None
                ),
                "action_chunk_last_timestep": (
                    action_chunk[-1].get_timestep() if action_chunk_size else None
                ),
                "chunk_ready_monotonic_time": postprocess_stops,
                "total_policy_processing_ms": (postprocess_stops - start_prepare) * 1000,
                "inference_end_to_chunk_ready_ms": (postprocess_stops - model_inference_end) * 1000,
                "inference_completed": True,
            }
            self.observation_lifecycle.record(
                observation_t,
                "action_chunk_ready",
                summary_updates=chunk_ready_metrics,
                **chunk_ready_metrics,
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
        if self.observation_lifecycle is not None:
            self.observation_lifecycle.close()
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
        if policy_server.observation_lifecycle is not None:
            policy_server.observation_lifecycle.close()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
