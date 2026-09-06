"""Guided RTC runtime helpers for the RB-Y1 asynchronous policy server."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

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


@dataclass
class RTCRequest:
    prefix: torch.Tensor | None
    inference_delay_frames: int
    applied: bool


class RTCState:
    """Per-client state needed to connect consecutive asynchronous policy requests."""

    def __init__(
        self,
        fps: float,
        latency_window_size: int = DEFAULT_DELAY_ESTIMATOR_WINDOW_SIZE,
    ) -> None:
        self.fps = fps
        self.previous_raw_chunk: torch.Tensor | None = None
        self.previous_timestep: int | None = None
        self.delay_estimator = RollingLatencyEstimator(latency_window_size)
        self._completed_inference_count = 0

    def prepare(self, current_timestep: int) -> RTCRequest:
        raw_leftover, _ = slice_previous_policy_chunk(
            self.previous_raw_chunk, self.previous_timestep, current_timestep
        )
        delay_estimate = self.delay_estimator.estimate_delay_frames(self.fps)
        delay = delay_estimate[0] if delay_estimate is not None else 0
        reason = None
        if raw_leftover is None:
            reason = "no_previous_chunk"
        elif raw_leftover.shape[0] == 0:
            reason = "previous_chunk_exhausted"
        elif not bool(torch.isfinite(raw_leftover).all()):
            reason = "previous_chunk_non_finite"
        elif delay_estimate is None:
            reason = "delay_estimator_warmup"
        return RTCRequest(
            prefix=None if reason else raw_leftover,
            inference_delay_frames=delay,
            applied=reason is None,
        )

    def complete(
        self,
        *,
        raw_chunk: torch.Tensor,
        timestep: int,
        latency_s: float,
    ) -> None:
        # The first completed inference is a cold start. It establishes the
        # previous chunk only and must never influence RTC delay estimation.
        is_cold_start = self._completed_inference_count == 0
        self.previous_raw_chunk = raw_chunk.detach().clone()
        self.previous_timestep = timestep
        if not is_cold_start:
            self.delay_estimator.add(latency_s)
        self._completed_inference_count += 1
