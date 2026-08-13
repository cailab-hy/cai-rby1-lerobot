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

"""Low-overhead, asynchronous observation lifecycle instrumentation.

The policy server only constructs this logger when lifecycle debugging is
enabled. Producers enqueue small metadata dictionaries without waiting for disk
I/O; a daemon thread owns the JSONL file and performs all writes.
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
import weakref
from datetime import datetime
from pathlib import Path
from queue import Full, Queue
from typing import Any

from .helpers import TimedObservation


class ObservationLifecycleLogger:
    """Track observations across receive, queueing, inference, and response."""

    def __init__(self, log_dir: str | Path, queue_maxsize: int = 10_000):
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        log_directory = Path(log_dir)
        log_directory.mkdir(parents=True, exist_ok=True)
        self.path = log_directory / f"policy_server_observation_lifecycle_{timestamp}.jsonl"

        self._events: Queue[dict[str, Any] | object] = Queue(maxsize=queue_maxsize)
        self._stop_token = object()
        self._sequence = itertools.count(1)
        self._state_lock = threading.Lock()
        self._identities: dict[int, tuple[weakref.ReferenceType[TimedObservation], int]] = {}
        self._summaries: dict[int, dict[str, Any]] = {}
        self._closed = False
        self.dropped_event_count = 0
        self._stream = self.path.open("a", encoding="utf-8")

        self._writer_thread = threading.Thread(
            target=self._write_events,
            name="observation-lifecycle-writer",
            daemon=True,
        )
        self._writer_thread.start()

    @staticmethod
    def _timestamp_fields() -> dict[str, str | float]:
        return {
            "wall_time": datetime.now().astimezone().isoformat(timespec="microseconds"),
            "monotonic_time": time.perf_counter(),
        }

    def _enqueue_event(self, event: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._events.put_nowait(event)
        except Full:
            self.dropped_event_count += 1
            # Avoid turning an overflowing debug queue into another hot-path bottleneck.
            if self.dropped_event_count == 1 or self.dropped_event_count & (
                self.dropped_event_count - 1
            ) == 0:
                logging.getLogger("policy_server").warning(
                    "Observation lifecycle logger queue is full; dropped %d debug events",
                    self.dropped_event_count,
                )

    def _write_events(self) -> None:
        stream = self._stream
        write_failed = False
        written_since_flush = 0
        try:
            while True:
                event = self._events.get()
                if event is self._stop_token:
                    break
                if write_failed:
                    continue
                try:
                    stream.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
                    written_since_flush += 1
                    if written_since_flush >= 64:
                        stream.flush()
                        written_since_flush = 0
                except Exception:
                    write_failed = True
                    logging.getLogger("policy_server").exception(
                        "Observation lifecycle writer failed; subsequent debug events will be discarded"
                    )

            if not write_failed:
                stopped_event = {
                    "event": "lifecycle_logger_stopped",
                    **self._timestamp_fields(),
                    "dropped_lifecycle_event_count": self.dropped_event_count,
                }
                stream.write(
                    json.dumps(stopped_event, separators=(",", ":"), ensure_ascii=False) + "\n"
                )
                stream.flush()
        finally:
            stream.close()

    def _observation_id_locked(self, observation: TimedObservation) -> int | None:
        identity = self._identities.get(id(observation))
        if identity is None:
            return None
        observation_ref, observation_id = identity
        if observation_ref() is observation:
            return observation_id
        return None

    def register(
        self,
        observation: TimedObservation,
        *,
        receive_monotonic_time: float | None = None,
        **fields: Any,
    ) -> int:
        """Assign a sidecar sequence ID and record the receive event."""
        if receive_monotonic_time is None:
            receive_monotonic_time = time.perf_counter()
        prefetch_requested = bool(getattr(observation, "prefetch_requested", False))
        prefetch_request_id = getattr(observation, "prefetch_request_id", None)
        with self._state_lock:
            existing_id = self._observation_id_locked(observation)
            if existing_id is not None:
                return existing_id

            observation_id = next(self._sequence)
            self._identities[id(observation)] = (weakref.ref(observation), observation_id)
            self._summaries[observation_id] = {
                "observation_id": observation_id,
                "observation_timestep": observation.get_timestep(),
                "observation_timestamp": observation.get_timestamp(),
                "received": True,
                "queued": False,
                "overwritten": False,
                "dequeued_for_processing": False,
                "similarity_check_performed": False,
                "similarity_filtered": False,
                "selected_for_inference": False,
                "inference_completed": False,
                "response_returned": False,
                "must_go": observation.must_go,
                "prefetch_requested": prefetch_requested,
                "prefetch_request_id": prefetch_request_id,
                "receive_monotonic_time": receive_monotonic_time,
                **fields,
            }

        self._enqueue_event(
            {
                "event": "observation_received",
                **self._timestamp_fields(),
                "observation_id": observation_id,
                "observation_timestep": observation.get_timestep(),
                "observation_timestamp": observation.get_timestamp(),
                "must_go": observation.must_go,
                "prefetch_requested": prefetch_requested,
                "prefetch_request_id": prefetch_request_id,
                **fields,
            }
        )
        return observation_id

    def ensure_registered(self, observation: TimedObservation) -> int:
        observation_id = self.observation_id(observation)
        if observation_id is not None:
            return observation_id
        return self.register(observation, tracking_origin="internal_entrypoint")

    def observation_id(self, observation: TimedObservation | None) -> int | None:
        if observation is None:
            return None
        with self._state_lock:
            return self._observation_id_locked(observation)

    def summary_value(self, observation: TimedObservation, key: str) -> Any:
        with self._state_lock:
            observation_id = self._observation_id_locked(observation)
            if observation_id is None:
                return None
            return self._summaries.get(observation_id, {}).get(key)

    def is_finalized(self, observation: TimedObservation) -> bool:
        return self.summary_value(observation, "final_disposition") is not None

    def record(
        self,
        observation: TimedObservation,
        event: str,
        *,
        summary_updates: dict[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        observation_id = self.ensure_registered(observation)
        prefetch_requested = bool(getattr(observation, "prefetch_requested", False))
        prefetch_request_id = getattr(observation, "prefetch_request_id", None)
        with self._state_lock:
            summary = self._summaries[observation_id]
            if summary_updates:
                summary.update(summary_updates)

        self._enqueue_event(
            {
                "event": event,
                **self._timestamp_fields(),
                "observation_id": observation_id,
                "observation_timestep": observation.get_timestep(),
                "prefetch_requested": prefetch_requested,
                "prefetch_request_id": prefetch_request_id,
                **fields,
            }
        )

    def record_global(self, event: str, **fields: Any) -> None:
        self._enqueue_event({"event": event, **self._timestamp_fields(), **fields})

    def finalize(self, observation: TimedObservation, disposition: str, **fields: Any) -> bool:
        """Emit exactly one terminal summary for an observation."""
        observation_id = self.ensure_registered(observation)
        with self._state_lock:
            summary = self._summaries[observation_id]
            if "final_disposition" in summary:
                return False
            summary.update(fields)
            summary["final_disposition"] = disposition
            outcome = dict(summary)

        self._enqueue_event(
            {
                "event": "observation_outcome",
                **self._timestamp_fields(),
                **outcome,
            }
        )
        return True

    def close(self) -> None:
        if self._closed:
            return

        with self._state_lock:
            unresolved_ids = [
                observation_id
                for observation_id, summary in self._summaries.items()
                if "final_disposition" not in summary
            ]
            unresolved_outcomes = []
            for observation_id in unresolved_ids:
                summary = self._summaries[observation_id]
                summary.update(final_disposition="unknown", final_reason="logger_closed_before_completion")
                unresolved_outcomes.append(dict(summary))

        for outcome in unresolved_outcomes:
            self._enqueue_event(
                {
                    "event": "observation_outcome",
                    **self._timestamp_fields(),
                    **outcome,
                }
            )

        # Blocking is acceptable during shutdown and guarantees a complete, parseable file.
        self._events.put(self._stop_token)
        self._writer_thread.join()
        self._closed = True
