"""Guided RTC runtime helpers for the RB-Y1 asynchronous policy server."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


SUPPORTED_RTC_MODE = "guided"


def latency_to_delay_frames(latency_s: float, fps: float) -> tuple[int, float]:
    """Convert a measured latency to the conservative frame delay used by LeRobot RTC."""
    if not math.isfinite(latency_s) or latency_s < 0:
        raise ValueError(f"latency_s must be finite and non-negative, got {latency_s}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    frames_float = latency_s * fps
    return math.ceil(frames_float), frames_float


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


class RTCState:
    """Per-client state needed to connect consecutive asynchronous policy requests."""

    def __init__(self, fps: float):
        self.fps = fps
        self.previous_raw_chunk: torch.Tensor | None = None
        self.previous_robot_chunk: torch.Tensor | None = None
        self.previous_timestep: int | None = None
        self.latencies_s: list[float] = []

    def prepare(self, current_timestep: int) -> RTCRequest:
        raw_leftover, shift = slice_previous_policy_chunk(
            self.previous_raw_chunk, self.previous_timestep, current_timestep
        )
        robot_leftover, _ = slice_previous_policy_chunk(
            self.previous_robot_chunk, self.previous_timestep, current_timestep
        )
        predicted_latency = max(self.latencies_s, default=0.0)
        delay, delay_float = latency_to_delay_frames(predicted_latency, self.fps)
        reason = None
        if raw_leftover is None:
            reason = "no previous chunk"
        elif raw_leftover.shape[0] == 0:
            reason = "previous chunk exhausted"
        elif not bool(torch.isfinite(raw_leftover).all()):
            reason = "previous chunk contains NaN or Inf"
        return RTCRequest(
            prefix=None if reason else raw_leftover,
            previous_robot_leftover=robot_leftover,
            shift=shift,
            inference_delay_frames=delay,
            delay_frames_float=delay_float,
            applied=reason is None,
            bypass_reason=reason,
        )

    def complete(
        self,
        *,
        raw_chunk: torch.Tensor,
        robot_chunk: torch.Tensor,
        timestep: int,
        latency_s: float,
    ) -> None:
        self.previous_raw_chunk = raw_chunk.detach().clone()
        self.previous_robot_chunk = robot_chunk.detach().clone()
        self.previous_timestep = timestep
        self.latencies_s.append(latency_s)


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
