"""Asynchronous camera-image capture for robot-client observations."""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from typing import Any

import cv2
import numpy as np


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class _ImageSnapshot:
    camera_key: str
    data: bytes | np.ndarray
    encoding: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _CaptureJob:
    capture_id: int
    wall_time: float
    timestep: int
    images: tuple[_ImageSnapshot, ...]


class CameraImageWriter:
    """Write sent camera observations without blocking the control loop on disk I/O."""

    _STOP = object()

    def __init__(
        self,
        directory: str | Path,
        camera_keys: tuple[str, ...],
        save_every_n: int,
        logger: logging.Logger,
        *,
        queue_size: int = 64,
    ) -> None:
        if save_every_n <= 0:
            raise ValueError("save_every_n must be positive")
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.camera_keys = camera_keys
        self.save_every_n = save_every_n
        self.logger = logger
        self._observation_count = 0
        self._capture_count = 0
        self._dropped_count = 0
        self._written_count = 0
        self._queue: Queue[_CaptureJob | object] = Queue(maxsize=queue_size)
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="camera-image-writer",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _safe_component(value: str) -> str:
        safe = _SAFE_COMPONENT.sub("_", value).strip("._")
        return safe or "camera"

    @staticmethod
    def _snapshot(camera_key: str, value: Any) -> _ImageSnapshot | None:
        if (
            isinstance(value, dict)
            and value.get("encoding") == "jpeg"
            and isinstance(value.get("data"), bytes)
        ):
            shape = value.get("transport_shape", ())
            return _ImageSnapshot(camera_key, bytes(value["data"]), "jpeg", tuple(shape))

        if isinstance(value, dict) and value.get("encoding") == "raw_resized":
            value = value.get("data")

        if isinstance(value, np.ndarray):
            return _ImageSnapshot(
                camera_key,
                np.ascontiguousarray(value).copy(),
                "rgb",
                tuple(int(dimension) for dimension in value.shape),
            )
        return None

    def submit(
        self,
        observation: dict[str, Any],
        *,
        wall_time: float,
        timestep: int,
    ) -> bool:
        """Queue one observation for saving; return whether it was accepted."""
        with self._lock:
            if self._closed:
                return False
            observation_index = self._observation_count
            self._observation_count += 1
            if observation_index % self.save_every_n:
                return False
            capture_id = self._capture_count
            self._capture_count += 1

        images = tuple(
            snapshot
            for camera_key in self.camera_keys
            if camera_key in observation
            for snapshot in [self._snapshot(camera_key, observation[camera_key])]
            if snapshot is not None
        )
        if not images:
            return False

        try:
            self._queue.put_nowait(_CaptureJob(capture_id, wall_time, timestep, images))
            return True
        except Full:
            with self._lock:
                self._dropped_count += 1
                dropped_count = self._dropped_count
            if dropped_count == 1 or dropped_count % 50 == 0:
                self.logger.warning(
                    "[CAMERA_CAPTURE] writer queue full; dropped capture sets=%d",
                    dropped_count,
                )
            return False

    def _writer_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is self._STOP:
                    return
                assert isinstance(job, _CaptureJob)
                self._write(job)
            except Exception:
                self.logger.exception("[CAMERA_CAPTURE] background write failed")
            finally:
                self._queue.task_done()

    def _write(self, job: _CaptureJob) -> None:
        timestamp_us = round(job.wall_time * 1_000_000)
        files: dict[str, str] = {}
        shapes: dict[str, list[int]] = {}
        for image in job.images:
            camera_dir = self.directory / self._safe_component(image.camera_key)
            camera_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"frame_{job.capture_id:06d}_t{job.timestep:08d}_{timestamp_us}.jpg"
            )
            path = camera_dir / filename
            if image.encoding == "jpeg":
                path.write_bytes(image.data)
            else:
                assert isinstance(image.data, np.ndarray)
                if image.data.ndim != 3 or image.data.shape[2] != 3:
                    raise ValueError(
                        f"Camera observation '{image.camera_key}' must have HWC RGB shape"
                    )
                bgr = cv2.cvtColor(image.data, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(path), bgr):
                    raise OSError(f"Failed to save camera image: {path}")
            files[image.camera_key] = str(path.relative_to(self.directory))
            shapes[image.camera_key] = list(image.shape)

        record = {
            "capture_id": job.capture_id,
            "wall_time": job.wall_time,
            "timestep": job.timestep,
            "files": files,
            "shapes": shapes,
        }
        with (self.directory / "manifest.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        with self._lock:
            self._written_count += 1

    def close(self) -> None:
        """Flush accepted jobs and stop the background writer."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(self._STOP)
        self._thread.join()
        self.logger.info(
            "[CAMERA_CAPTURE] stopped directory=%s saved_sets=%d dropped_sets=%d",
            self.directory,
            self._written_count,
            self._dropped_count,
        )
