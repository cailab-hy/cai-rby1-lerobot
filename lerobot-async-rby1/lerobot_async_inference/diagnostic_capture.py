"""Non-invasive policy-ready batch and generated-chunk capture.

Only CPU copies are made on the inference thread. Disk I/O and checksums run
on one background writer so a diagnostic rollout does not synchronously wait
for ``torch.save``. The class is never instantiated when capture is disabled.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any

import torch

from .frozen_noise_analysis import clone_to_cpu, nested_checksum


@dataclass
class PreparedCapture:
    capture_id: int
    batch: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class _CaptureJob:
    prepared: PreparedCapture
    raw_chunk: torch.Tensor
    robot_chunk: torch.Tensor


class PolicyBatchCaptureWriter:
    """Capture a bounded sequence without mutating policy inputs or outputs."""

    _STOP = object()

    def __init__(self, directory: str | Path, max_captures: int, logger: logging.Logger):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_captures = max_captures
        self.logger = logger
        self._reserved = 0
        self._next_id = self._find_next_capture_id()
        self._lock = threading.Lock()
        self._queue: Queue[_CaptureJob | object] = Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="policy-diagnostic-capture-writer",
            daemon=True,
        )
        self._thread.start()

    def _find_next_capture_id(self) -> int:
        ids: list[int] = []
        for path in self.directory.glob("capture_*.pt"):
            try:
                ids.append(int(path.stem.rsplit("_", maxsplit=1)[1]))
            except ValueError:
                continue
        return max(ids, default=-1) + 1

    def prepare(
        self, batch: dict[str, Any], metadata: dict[str, Any]
    ) -> PreparedCapture | None:
        """Reserve an id and clone the exact pre-inference batch to CPU."""
        with self._lock:
            if self._closed or self._reserved >= self.max_captures:
                return None
            capture_id = self._next_id
            self._next_id += 1
            self._reserved += 1

        clone_started = time.perf_counter()
        cpu_batch = clone_to_cpu(batch)
        clone_ms = (time.perf_counter() - clone_started) * 1000
        return PreparedCapture(
            capture_id=capture_id,
            batch=cpu_batch,
            metadata={**metadata, "batch_cpu_clone_ms": clone_ms},
        )

    def submit(
        self,
        prepared: PreparedCapture,
        raw_chunk: torch.Tensor,
        robot_chunk: torch.Tensor,
        metadata: dict[str, Any],
    ) -> None:
        """Clone generated chunks to CPU and enqueue all disk writes."""
        chunk_clone_started = time.perf_counter()
        raw_cpu = raw_chunk.detach().clone().cpu()
        robot_cpu = robot_chunk.detach().clone().cpu()
        chunk_clone_ms = (time.perf_counter() - chunk_clone_started) * 1000
        prepared.metadata.update(metadata)
        prepared.metadata["chunk_cpu_clone_ms"] = chunk_clone_ms
        prepared.metadata["writer_queue_depth_before_enqueue"] = self._queue.qsize()
        # An unbounded in-process Queue makes this a non-blocking handoff. Keep
        # the field explicit; disk time is measured separately by the writer.
        prepared.metadata["enqueue_ms"] = 0.0
        self._queue.put_nowait(_CaptureJob(prepared, raw_cpu, robot_cpu))

    def _writer_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is self._STOP:
                    return
                assert isinstance(job, _CaptureJob)
                self._write(job)
            except Exception:
                self.logger.exception("[DIAGNOSTIC_CAPTURE] background write failed")
            finally:
                self._queue.task_done()

    def _write(self, job: _CaptureJob) -> None:
        capture_id = job.prepared.capture_id
        suffix = f"{capture_id:03d}"
        save_started = time.perf_counter()
        torch.save(job.prepared.batch, self.directory / f"capture_{suffix}.pt")
        torch.save(job.raw_chunk, self.directory / f"raw_chunk_{suffix}.pt")
        torch.save(job.robot_chunk, self.directory / f"robot_chunk_{suffix}.pt")

        metadata = dict(job.prepared.metadata)
        state = job.prepared.batch.get("observation.state")
        metadata.update(
            capture_id=capture_id,
            observation_state_checksum=(
                nested_checksum(state) if isinstance(state, torch.Tensor) else None
            ),
            batch_checksum=nested_checksum(job.prepared.batch),
            raw_chunk_checksum=nested_checksum(job.raw_chunk),
            robot_chunk_checksum=nested_checksum(job.robot_chunk),
            raw_chunk_shape=list(job.raw_chunk.shape),
            robot_chunk_shape=list(job.robot_chunk.shape),
        )
        metadata["background_save_and_checksum_ms"] = (
            time.perf_counter() - save_started
        ) * 1000
        with (self.directory / "capture_metadata.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, sort_keys=True) + "\n")

        self.logger.info(
            "[DIAGNOSTIC_CAPTURE] saved id=%03d batch=%s raw_shape=%s robot_shape=%s "
            "batch_clone_ms=%.3f chunk_clone_ms=%.3f background_ms=%.3f",
            capture_id,
            self.directory / f"capture_{suffix}.pt",
            tuple(job.raw_chunk.shape),
            tuple(job.robot_chunk.shape),
            metadata["batch_cpu_clone_ms"],
            metadata["chunk_cpu_clone_ms"],
            metadata["background_save_and_checksum_ms"],
        )

    def flush(self) -> None:
        self._queue.join()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(self._STOP)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            self.logger.warning("[DIAGNOSTIC_CAPTURE] writer did not stop within 30 seconds")
