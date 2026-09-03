"""Guided RTC runtime helpers for the RB-Y1 asynchronous policy server."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


SUPPORTED_RTC_MODE = "guided"
DEFAULT_DELAY_ESTIMATOR_WINDOW_SIZE = 10


def latency_to_delay_frames(latency_s: float, fps: float) -> tuple[int, float]:
    """Convert a measured latency to the conservative frame delay used by LeRobot RTC."""
    if not math.isfinite(latency_s) or latency_s < 0:
        raise ValueError(f"latency_s must be finite and non-negative, got {latency_s}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    frames_float = latency_s * fps
    return math.ceil(frames_float), frames_float


class RollingLatencyEstimator:
    """Estimate RTC delay from the maximum of a bounded recent-latency window."""

    def __init__(self, window_size: int = DEFAULT_DELAY_ESTIMATOR_WINDOW_SIZE) -> None:
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError(f"window_size must be a positive integer, got {window_size}")
        self._latencies_s: deque[float] = deque(maxlen=window_size)

    @property
    def ready(self) -> bool:
        return bool(self._latencies_s)

    @property
    def window_count(self) -> int:
        return len(self._latencies_s)

    @property
    def window_max_s(self) -> float | None:
        return max(self._latencies_s) if self._latencies_s else None

    def add(self, latency_s: float) -> None:
        if not math.isfinite(latency_s) or latency_s < 0:
            raise ValueError(f"latency_s must be finite and non-negative, got {latency_s}")
        self._latencies_s.append(float(latency_s))

    def estimate_delay_frames(self, fps: float) -> tuple[int, float] | None:
        """Return the rolling-max delay, or ``None`` until one sample is available."""
        latency_s = self.window_max_s
        if latency_s is None:
            return None
        return latency_to_delay_frames(latency_s, fps)


def slice_previous_policy_chunk(
    previous_chunk: torch.Tensor | None,
    previous_timestep: int | None,
    current_timestep: int,
) -> tuple[torch.Tensor | None, int]:
    """Remove the already-executed temporal prefix from an original policy chunk."""
    if previous_chunk is None or previous_timestep is None:
        return None, 0
    if previous_chunk.ndim != 2:
        raise ValueError(
            f"previous policy chunk must have shape (T, A), got {tuple(previous_chunk.shape)}"
        )
    shift = current_timestep - previous_timestep
    if shift < 0:
        raise ValueError(
            f"current timestep {current_timestep} precedes previous chunk {previous_timestep}"
        )
    if shift >= previous_chunk.shape[0]:
        return previous_chunk[:0].clone(), shift
    return previous_chunk[shift:].clone(), shift


def overlap_metrics(previous: torch.Tensor | None, current: torch.Tensor) -> dict[str, Any]:
    """Compare a leftover previous trajectory to a newly generated trajectory."""
    empty = {
        "aligned_overlap_length": 0,
        "post_rtc_overlap_rmse": None,
        "post_rtc_overlap_mean_abs": None,
        "post_rtc_overlap_p95": None,
        "post_rtc_overlap_max": None,
        "post_rtc_direction_mismatch": None,
        "post_rtc_derivative_cosine_similarity": None,
    }
    if previous is None or previous.ndim != 2 or current.ndim != 2:
        return empty
    overlap = min(previous.shape[0], current.shape[0])
    dimensions = min(previous.shape[1], current.shape[1])
    if overlap <= 0 or dimensions <= 0:
        return empty
    old = previous[:overlap, :dimensions].detach().to(dtype=torch.float64, device="cpu")
    new = current[:overlap, :dimensions].detach().to(dtype=torch.float64, device="cpu")
    difference = old - new
    old_delta = old[1:] - old[:-1]
    new_delta = new[1:] - new[:-1]
    eligible = (old_delta.abs() > 0.005) & (new_delta.abs() > 0.005)
    mismatch = eligible & ((old_delta * new_delta) < 0)
    old_flat = old_delta.flatten()
    new_flat = new_delta.flatten()
    denominator = float(torch.linalg.vector_norm(old_flat) * torch.linalg.vector_norm(new_flat))
    return {
        "aligned_overlap_length": overlap,
        "post_rtc_overlap_rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "post_rtc_overlap_mean_abs": float(difference.abs().mean()),
        "post_rtc_overlap_p95": float(torch.quantile(difference.abs().flatten(), 0.95)),
        "post_rtc_overlap_max": float(difference.abs().max()),
        "post_rtc_direction_mismatch": (
            float(mismatch.sum() / eligible.sum()) if int(eligible.sum()) else None
        ),
        "post_rtc_derivative_cosine_similarity": (
            float(torch.dot(old_flat, new_flat) / denominator) if denominator > 0 else None
        ),
    }


@dataclass
class RTCRequest:
    prefix: torch.Tensor | None
    previous_robot_leftover: torch.Tensor | None
    shift: int
    inference_delay_frames: int
    delay_frames_float: float
    applied: bool
    bypass_reason: str | None
    delay_estimator_ready: bool = False
    delay_estimator_window_count: int = 0
    delay_estimator_window_max_ms: float | None = None


class RTCState:
    """Per-client state needed to connect consecutive asynchronous policy requests."""

    def __init__(
        self,
        fps: float,
        latency_window_size: int = DEFAULT_DELAY_ESTIMATOR_WINDOW_SIZE,
    ) -> None:
        self.fps = fps
        self.previous_raw_chunk: torch.Tensor | None = None
        self.previous_robot_chunk: torch.Tensor | None = None
        self.previous_timestep: int | None = None
        self.delay_estimator = RollingLatencyEstimator(latency_window_size)
        self._completed_inference_count = 0

    def prepare(self, current_timestep: int) -> RTCRequest:
        raw_leftover, shift = slice_previous_policy_chunk(
            self.previous_raw_chunk, self.previous_timestep, current_timestep
        )
        robot_leftover, _ = slice_previous_policy_chunk(
            self.previous_robot_chunk, self.previous_timestep, current_timestep
        )
        delay_estimate = self.delay_estimator.estimate_delay_frames(self.fps)
        delay, delay_float = delay_estimate if delay_estimate is not None else (0, 0.0)
        reason = None
        if raw_leftover is None:
            reason = "no_previous_chunk"
        elif raw_leftover.shape[0] == 0:
            reason = "previous_chunk_exhausted"
        elif not bool(torch.isfinite(raw_leftover).all()):
            reason = "previous_chunk_non_finite"
        elif delay_estimate is None:
            reason = "delay_estimator_warmup"
        window_max_s = self.delay_estimator.window_max_s
        return RTCRequest(
            prefix=None if reason else raw_leftover,
            previous_robot_leftover=robot_leftover,
            shift=shift,
            inference_delay_frames=delay,
            delay_frames_float=delay_float,
            applied=reason is None,
            bypass_reason=reason,
            delay_estimator_ready=self.delay_estimator.ready,
            delay_estimator_window_count=self.delay_estimator.window_count,
            delay_estimator_window_max_ms=(
                window_max_s * 1000 if window_max_s is not None else None
            ),
        )

    def complete(
        self,
        *,
        raw_chunk: torch.Tensor,
        robot_chunk: torch.Tensor,
        timestep: int,
        latency_s: float,
    ) -> None:
        # The first completed inference is a cold start. It establishes the
        # previous chunk only and must never influence RTC delay estimation.
        is_cold_start = self._completed_inference_count == 0
        self.previous_raw_chunk = raw_chunk.detach().clone()
        self.previous_robot_chunk = robot_chunk.detach().clone()
        self.previous_timestep = timestep
        if not is_cold_start:
            self.delay_estimator.add(latency_s)
        self._completed_inference_count += 1


class RTCDiagnosticsWriter:
    """Small synchronized JSONL writer, created only when RTC is enabled."""

    def __init__(self, directory: str | Path):
        output_dir = Path(directory).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = output_dir / f"rtc_diagnostics_{stamp}.jsonl"
        self._stream = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._stream.write(json.dumps({"wall_time": time.time(), **record}, sort_keys=True) + "\n")
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.close()
