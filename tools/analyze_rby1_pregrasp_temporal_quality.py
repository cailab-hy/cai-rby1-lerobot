#!/usr/bin/env python3
"""Read-only pre-grasp smoothness and action/state lag diagnostics for RB-Y1.

Only numeric parquet columns are read; videos and dataset files are never modified.
Metric primitives are reused from ``analyze_rby1_demo_smoothness.py`` so first
difference, second difference, and oscillation definitions remain consistent.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analyze_rby1_demo_smoothness import (
    DimensionMapping,
    build_dimension_mapping,
    descriptive_stats,
    episode_tasks,
    oscillation_counts,
    resolve_feature_key,
    temporal_differences,
)


EXPECTED_EPISODES = 200
EXPECTED_FRAMES = 59_729
EXPECTED_FPS = 15.0
WINDOWS = ("whole", "approach", "pregrasp", "grasp_local")


@dataclass(frozen=True)
class GripperDirection:
    open_is_high: bool
    low_plateau: float
    high_plateau: float
    initial_median: float
    agreement_fraction: float

    @property
    def open_value(self) -> float:
        return self.high_plateau if self.open_is_high else self.low_plateau

    @property
    def close_value(self) -> float:
        return self.low_plateau if self.open_is_high else self.high_plateau


@dataclass(frozen=True)
class GraspDetection:
    detected: bool
    frame: int | None
    before: float
    after: float
    magnitude: float
    confidence: float
    gripper_min: float
    gripper_max: float
    largest_deltas: tuple[float, ...]


@dataclass
class WindowMetrics:
    delta_max: np.ndarray
    delta2_max: np.ndarray
    delta_stats: dict[str, float]
    delta2_stats: dict[str, float]
    reversals: np.ndarray
    eligible: np.ndarray
    joint_delta_stats: list[dict[str, float]]
    joint_delta2_stats: list[dict[str, float]]

    @property
    def oscillation(self) -> float:
        denominator = int(self.eligible.sum())
        return float(self.reversals.sum() / denominator) if denominator else math.nan


@dataclass(frozen=True)
class LagPoint:
    lag: int
    samples: int
    position_rmse: float
    velocity_corr: float
    magnitude_corr: float
    velocity_rmse: float
    joint_corr: tuple[float, ...]


@dataclass
class LagResult:
    points: list[LagPoint]
    best_position: int | None
    best_corr: int | None
    best_velocity_rmse: int | None
    representative: int | None
    confidence: str


def _finite(value: float) -> bool:
    return value is not None and math.isfinite(float(value))


def _fmt(value: float | int | None, digits: int = 6) -> str:
    if value is None or not _finite(float(value)):
        return "N/A"
    return f"{float(value):.{digits}g}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _task_group(task: str) -> str:
    lowered = task.lower()
    if "cup" in lowered:
        return "CUP"
    if "bowl" in lowered:
        return "BOWL"
    return "OTHER"


def _split_sequences(batch: Mapping[str, Any], keys: Sequence[str]) -> dict[str, dict[int, np.ndarray]]:
    episodes = np.asarray(batch["episode_index"], dtype=np.int64).reshape(-1)
    frames = np.asarray(batch["frame_index"], dtype=np.int64).reshape(-1)
    result: dict[str, dict[int, np.ndarray]] = {key: {} for key in keys}
    for episode in sorted(int(value) for value in np.unique(episodes)):
        positions = np.flatnonzero(episodes == episode)
        positions = positions[np.argsort(frames[positions], kind="stable")]
        for key in keys:
            result[key][episode] = np.asarray(batch[key], dtype=np.float64)[positions]
    return result


def load_numeric_data(dataset: Any, action_key: str, state_key: str) -> tuple[dict[str, Any], dict[str, dict[int, np.ndarray]]]:
    columns = [action_key, state_key, "episode_index", "frame_index", "index", "task_index"]
    view = dataset.hf_dataset.select_columns(columns).with_format("numpy")
    batch = view[:]
    return batch, _split_sequences(batch, (action_key, state_key))


def sanity_check(
    dataset: Any,
    batch: Mapping[str, Any],
    action_key: str,
    state_key: str,
    fps: float,
) -> None:
    version = importlib.metadata.version("lerobot")
    episodes = np.asarray(batch["episode_index"], dtype=np.int64).reshape(-1)
    frames = np.asarray(batch["frame_index"], dtype=np.int64).reshape(-1)
    indices = np.asarray(batch["index"], dtype=np.int64).reshape(-1)
    action = np.asarray(batch[action_key])
    state = np.asarray(batch[state_key])
    print("Dataset sanity check")
    print(f"  LeRobot version        : {version}")
    print(f"  episodes / frames / fps: {dataset.num_episodes} / {dataset.num_frames} / {dataset.fps}")
    print(f"  episode range          : {episodes.min()} ... {episodes.max()}")
    print(f"  global index range     : {indices.min()} ... {indices.max()}")
    print(f"  action / state arrays  : {action.shape} / {state.shape}")

    warnings: list[str] = []
    if dataset.num_episodes != EXPECTED_EPISODES:
        warnings.append(f"expected {EXPECTED_EPISODES} episodes, found {dataset.num_episodes}")
    if dataset.num_frames != EXPECTED_FRAMES:
        warnings.append(f"expected {EXPECTED_FRAMES} frames, found {dataset.num_frames}")
    if not math.isclose(float(dataset.fps), fps, abs_tol=1e-9):
        raise ValueError(f"--fps={fps} differs from dataset fps={dataset.fps}")
    if not math.isclose(fps, EXPECTED_FPS, abs_tol=1e-9):
        warnings.append(f"expected {EXPECTED_FPS:g} FPS, analyzing {fps:g} FPS")
    expected_episodes = np.arange(dataset.num_episodes)
    if not np.array_equal(np.unique(episodes), expected_episodes):
        raise ValueError("episode_index is not contiguous from 0 to num_episodes-1")
    if not np.array_equal(indices, np.arange(dataset.num_frames)):
        raise ValueError("global index is not contiguous from 0 to num_frames-1")
    for episode in expected_episodes:
        episode_frames = frames[episodes == episode]
        if not np.array_equal(episode_frames, np.arange(len(episode_frames))):
            raise ValueError(f"episode {episode} frame_index is not contiguous from zero")
    for key, values in ((action_key, action), (state_key, state)):
        metadata_shape = tuple(dataset.features[key].get("shape") or ())
        if metadata_shape != (16,) or values.ndim != 2 or values.shape[1] != 16:
            raise ValueError(f"{key} is not 16-dimensional: metadata={metadata_shape}, data={values.shape}")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    print("  contiguous indices     : PASS")


def print_mapping(label: str, mapping: DimensionMapping) -> None:
    print(f"\n{label} dimension mapping (from metadata)")
    for index, name in enumerate(mapping.names):
        print(f"  [{index:02d}] {name:<20} -> {mapping.group_for_index(index)}")


def infer_gripper_direction(
    gripper_by_episode: Mapping[int, np.ndarray], fps: float
) -> tuple[GripperDirection, dict[str, float]]:
    all_values = np.concatenate([np.asarray(values, dtype=np.float64) for values in gripper_by_episode.values()])
    quantile_levels = (0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0)
    quantiles = np.quantile(all_values, quantile_levels)
    low, high = float(quantiles[2]), float(quantiles[4])
    if high - low <= 1e-8:
        raise ValueError("Left gripper has no separable open/closed plateaus")
    initial_values = []
    high_votes = 0
    initial_frames = max(3, int(round(fps)))
    for values in gripper_by_episode.values():
        initial = float(np.median(values[: min(initial_frames, len(values))]))
        initial_values.append(initial)
        high_votes += int(abs(initial - high) <= abs(initial - low))
    open_is_high = high_votes >= len(initial_values) / 2
    agreement = max(high_votes, len(initial_values) - high_votes) / len(initial_values)
    direction = GripperDirection(
        open_is_high=open_is_high,
        low_plateau=low,
        high_plateau=high,
        initial_median=float(np.median(initial_values)),
        agreement_fraction=float(agreement),
    )
    names = ("min", "q01", "q10", "q50", "q90", "q99", "max")
    return direction, {name: float(value) for name, value in zip(names, quantiles, strict=True)}


def _has_consecutive(mask: np.ndarray, count: int) -> bool:
    if count <= 0:
        return True
    if len(mask) < count:
        return False
    return bool(np.convolve(mask.astype(np.int64), np.ones(count, dtype=np.int64), mode="valid").max() >= count)


def detect_grasp_close(
    values: np.ndarray,
    direction: GripperDirection,
    *,
    hold_frames: int,
    transition_min_fraction: float,
    confirm_window_frames: int,
    min_close_fraction: float,
) -> GraspDetection:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    low, high = np.quantile(x, [0.10, 0.90])
    open_value, close_value = (high, low) if direction.open_is_high else (low, high)
    span = abs(float(close_value - open_value))
    largest = tuple(float(value) for value in sorted(np.abs(np.diff(x)), reverse=True)[:5])
    if span <= 1e-8 or len(x) < hold_frames + 2:
        return GraspDetection(False, None, math.nan, math.nan, 0.0, 0.0, float(x.min()), float(x.max()), largest)

    progress = (x - open_value) / (close_value - open_value)
    open_side = progress <= transition_min_fraction
    close_side = progress >= 1.0 - transition_min_fraction
    candidates = np.flatnonzero(
        (progress[1:] >= transition_min_fraction) & (progress[:-1] < transition_min_fraction)
    ) + 1
    for frame in candidates:
        before_slice = open_side[max(0, frame - hold_frames) : frame]
        if len(before_slice) < hold_frames or not bool(np.all(before_slice)):
            continue
        stop = min(len(x), frame + confirm_window_frames)
        confirmation = close_side[frame:stop]
        close_positions = np.flatnonzero(confirmation)
        if not len(close_positions):
            continue
        # A slow but valid closure may only reach the close plateau near the end
        # of the confirmation horizon. Measure persistence after that first
        # arrival, rather than penalizing the preceding transition ramp. This
        # still rejects short close/open glitches because their remaining tail
        # is dominated by the reopened plateau.
        close_tail = confirmation[close_positions[0] :]
        close_fraction = float(np.mean(close_tail))
        if close_fraction < min_close_fraction or not _has_consecutive(close_tail, hold_frames):
            continue
        before = float(np.median(x[frame - hold_frames : frame]))
        after_values = x[frame:stop][confirmation]
        after = float(np.median(after_values)) if len(after_values) else float(x[frame])
        magnitude = abs(after - before)
        if magnitude < transition_min_fraction * span:
            continue
        range_score = min(1.0, span / max(abs(direction.high_plateau - direction.low_plateau), 1e-8))
        magnitude_score = min(1.0, magnitude / span)
        confidence = float(0.45 * close_fraction + 0.30 * magnitude_score + 0.25 * range_score)
        return GraspDetection(
            True,
            int(frame),
            before,
            after,
            magnitude,
            confidence,
            float(x.min()),
            float(x.max()),
            largest,
        )
    return GraspDetection(False, None, math.nan, math.nan, 0.0, 0.0, float(x.min()), float(x.max()), largest)


def window_bounds(num_frames: int, grasp_frame: int | None, fps: float, args: argparse.Namespace) -> dict[str, tuple[int, int]]:
    bounds = {"whole": (0, num_frames)}
    if grasp_frame is None:
        return {**bounds, "approach": (0, 0), "pregrasp": (0, 0), "grasp_local": (0, 0)}
    approach_start = grasp_frame - int(round(args.approach_start_seconds * fps))
    approach_end = grasp_frame - int(round(args.approach_end_seconds * fps))
    pregrasp_start = grasp_frame - int(round(args.pregrasp_seconds * fps))
    local = int(round(args.grasp_local_seconds * fps))
    bounds["approach"] = (max(0, approach_start), min(num_frames, max(0, approach_end)))
    bounds["pregrasp"] = (max(0, pregrasp_start), min(num_frames, grasp_frame + 1))
    bounds["grasp_local"] = (max(0, grasp_frame - local), min(num_frames, grasp_frame + local + 1))
    return bounds


def compute_window_metrics(
    values: np.ndarray,
    indices: Sequence[int],
    start: int,
    end: int,
    threshold: float,
) -> WindowMetrics:
    selected = np.asarray(values, dtype=np.float64)[start:end, list(indices)]
    if selected.ndim != 2:
        selected = np.empty((0, len(indices)), dtype=np.float64)
    delta, delta2 = temporal_differences(selected) if len(selected) else (np.empty((0, len(indices))), np.empty((0, len(indices))))
    delta_max = np.max(np.abs(delta), axis=1) if len(delta) else np.empty(0)
    delta2_max = np.max(np.abs(delta2), axis=1) if len(delta2) else np.empty(0)
    reversals, eligible = oscillation_counts(delta, threshold)
    return WindowMetrics(
        delta_max=delta_max,
        delta2_max=delta2_max,
        delta_stats=descriptive_stats(delta_max),
        delta2_stats=descriptive_stats(delta2_max),
        reversals=reversals,
        eligible=eligible,
        joint_delta_stats=[descriptive_stats(np.abs(delta[:, joint]), (50, 95, 99)) for joint in range(len(indices))],
        joint_delta2_stats=[descriptive_stats(np.abs(delta2[:, joint]), (50, 95, 99)) for joint in range(len(indices))],
    )


def aggregate_metrics(metrics: Sequence[WindowMetrics]) -> dict[str, float]:
    delta = np.concatenate([item.delta_max for item in metrics if len(item.delta_max)]) if any(len(item.delta_max) for item in metrics) else np.empty(0)
    delta2 = np.concatenate([item.delta2_max for item in metrics if len(item.delta2_max)]) if any(len(item.delta2_max) for item in metrics) else np.empty(0)
    reversals = sum(int(item.reversals.sum()) for item in metrics)
    eligible = sum(int(item.eligible.sum()) for item in metrics)
    episode_ratios = np.asarray([item.oscillation for item in metrics], dtype=np.float64)
    return {
        **{f"delta_{key}": value for key, value in descriptive_stats(delta).items()},
        **{f"delta2_{key}": value for key, value in descriptive_stats(delta2).items()},
        "oscillation_pooled": reversals / eligible if eligible else math.nan,
        "oscillation_episode_mean": float(np.nanmean(episode_ratios)) if np.isfinite(episode_ratios).any() else math.nan,
        "oscillation_reversals": reversals,
        "oscillation_evaluated_pairs": eligible,
    }


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.std(x[mask]) <= 1e-12 or np.std(y[mask]) <= 1e-12:
        return math.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def estimate_lag(
    action: np.ndarray,
    state: np.ndarray,
    action_indices: Sequence[int],
    state_indices: Sequence[int],
    start: int,
    end: int,
    lag_min: int,
    lag_max: int,
    motion_threshold: float,
    min_samples: int = 5,
) -> LagResult:
    a = np.asarray(action, dtype=np.float64)[:, list(action_indices)]
    s = np.asarray(state, dtype=np.float64)[:, list(state_indices)]
    da, ds = np.diff(a, axis=0), np.diff(s, axis=0)
    base_t = np.arange(max(1, start + 1), min(end, len(a)), dtype=np.int64)
    points: list[LagPoint] = []
    for lag in range(lag_min, lag_max + 1):
        state_t = base_t + lag
        valid = (state_t >= 1) & (state_t < len(s))
        t = base_t[valid]
        u = state_t[valid]
        if len(t):
            moving = np.linalg.norm(da[t - 1], axis=1) > motion_threshold
            t, u = t[moving], u[moving]
        if len(t) < min_samples:
            points.append(LagPoint(lag, int(len(t)), math.nan, math.nan, math.nan, math.nan, tuple(math.nan for _ in action_indices)))
            continue
        pos_error = a[t] - s[u]
        av, sv = da[t - 1], ds[u - 1]
        joint_corr = tuple(_corr(av[:, joint], sv[:, joint]) for joint in range(av.shape[1]))
        points.append(
            LagPoint(
                lag=lag,
                samples=int(len(t)),
                position_rmse=float(np.sqrt(np.mean(np.sum(pos_error**2, axis=1)))),
                velocity_corr=_corr(av, sv),
                magnitude_corr=_corr(np.linalg.norm(av, axis=1), np.linalg.norm(sv, axis=1)),
                velocity_rmse=float(np.sqrt(np.mean(np.sum((av - sv) ** 2, axis=1)))),
                joint_corr=joint_corr,
            )
        )

    def best(attribute: str, maximize: bool) -> int | None:
        valid_points = [point for point in points if _finite(getattr(point, attribute))]
        if not valid_points:
            return None
        selector = max if maximize else min
        return selector(valid_points, key=lambda point: getattr(point, attribute)).lag

    best_position = best("position_rmse", False)
    best_corr = best("velocity_corr", True)
    best_velocity = best("velocity_rmse", False)
    available = [value for value in (best_position, best_corr, best_velocity) if value is not None]
    if len(available) < 3:
        representative, confidence = (int(np.median(available)), "LOW") if available else (None, "NONE")
    else:
        representative = int(np.median(available))
        spread = max(available) - min(available)
        confidence = "HIGH" if spread == 0 else "MEDIUM" if spread <= 1 else "LOW"
    return LagResult(points, best_position, best_corr, best_velocity, representative, confidence)


def lag_distribution_stats(values: Sequence[int | float | None]) -> dict[str, float]:
    array = np.asarray([value for value in values if value is not None and _finite(float(value))], dtype=np.float64)
    if not len(array):
        return {key: math.nan for key in ("mean", "std", "p10", "p25", "p50", "p75", "p90", "min", "max")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        **{f"p{q}": float(np.percentile(array, q)) for q in (10, 25, 50, 75, 90)},
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _metric_columns(prefix: str, metrics: WindowMetrics) -> dict[str, Any]:
    return {
        f"{prefix}_delta_median": metrics.delta_stats["median"],
        f"{prefix}_delta_p90": metrics.delta_stats["p90"],
        f"{prefix}_delta_p95": metrics.delta_stats["p95"],
        f"{prefix}_delta_p99": metrics.delta_stats["p99"],
        f"{prefix}_delta_max": metrics.delta_stats["max"],
        f"{prefix}_delta2_median": metrics.delta2_stats["median"],
        f"{prefix}_delta2_p90": metrics.delta2_stats["p90"],
        f"{prefix}_delta2_p95": metrics.delta2_stats["p95"],
        f"{prefix}_delta2_p99": metrics.delta2_stats["p99"],
        f"{prefix}_delta2_max": metrics.delta2_stats["max"],
        f"{prefix}_oscillation": metrics.oscillation,
        f"{prefix}_oscillation_reversals": int(metrics.reversals.sum()),
        f"{prefix}_oscillation_evaluated_pairs": int(metrics.eligible.sum()),
    }


def _lag_columns(prefix: str, result: LagResult) -> dict[str, Any]:
    return {
        f"{prefix}_best_lag_position": result.best_position,
        f"{prefix}_best_lag_corr": result.best_corr,
        f"{prefix}_best_lag_velocity_rmse": result.best_velocity_rmse,
        f"{prefix}_representative_lag": result.representative,
        f"{prefix}_lag_confidence": result.confidence,
    }


def build_summary_rows(
    episode_records: Mapping[int, dict[str, Any]], groups: Mapping[str, Sequence[int]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names = (
        "action_delta_p95",
        "action_delta2_p95",
        "action_oscillation_pooled",
        "action_oscillation_episode_mean",
        "state_delta_p95",
        "state_delta2_p95",
        "state_oscillation_pooled",
        "state_oscillation_episode_mean",
    )
    for group, episode_ids in groups.items():
        aggregate: dict[str, dict[str, dict[str, float]]] = {feature: {} for feature in ("action", "state")}
        for feature in ("action", "state"):
            for window in WINDOWS:
                aggregate[feature][window] = aggregate_metrics(
                    [episode_records[episode]["metrics"][feature][window] for episode in episode_ids]
                )
        for metric_name in metric_names:
            feature, suffix = metric_name.split("_", 1)
            rows.append(
                {
                    "group": group,
                    "metric": metric_name,
                    **{window: aggregate[feature][window][suffix] for window in ("whole", "approach", "pregrasp")},
                }
            )
    return rows


def build_lag_summary_rows(
    episode_records: Mapping[int, dict[str, Any]], groups: Mapping[str, Sequence[int]], fps: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, episode_ids in groups.items():
        for window in ("whole", "pregrasp"):
            results = [episode_records[episode]["lags"][window] for episode in episode_ids]
            fields = {
                "representative": [result.representative for result in results],
                "position": [result.best_position for result in results],
                "velocity_corr": [result.best_corr for result in results],
                "velocity_rmse": [result.best_velocity_rmse for result in results],
            }
            for field, values in fields.items():
                stats = lag_distribution_stats(values)
                for metric, value in stats.items():
                    rows.append({"group": group, "window": window, "metric": f"{field}_{metric}_frames", "value": value})
                    rows.append({"group": group, "window": window, "metric": f"{field}_{metric}_ms", "value": value * 1000 / fps})
            confidence_counts = Counter(result.confidence for result in results)
            for confidence in ("HIGH", "MEDIUM", "LOW", "NONE"):
                rows.append({"group": group, "window": window, "metric": f"confidence_{confidence.lower()}_count", "value": confidence_counts[confidence]})
            rows.append({"group": group, "window": window, "metric": "negative_representative_count", "value": sum(result.representative is not None and result.representative < 0 for result in results)})
    return rows


def _summary_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["group"]), str(row["metric"])): row for row in rows}


def interpret(summary_rows: Sequence[Mapping[str, Any]], lag_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    smooth = _summary_lookup(summary_rows)
    lag = {(str(row["group"]), str(row["window"]), str(row["metric"])): float(row["value"]) for row in lag_rows}
    whole_osc = float(smooth[("ALL", "action_oscillation_pooled")]["whole"])
    pre_osc = float(smooth[("ALL", "action_oscillation_pooled")]["pregrasp"])
    action_d2 = float(smooth[("ALL", "action_delta2_p95")]["pregrasp"])
    state_d2 = float(smooth[("ALL", "state_delta2_p95")]["pregrasp"])
    state_osc = float(smooth[("ALL", "state_oscillation_pooled")]["pregrasp"])
    median_lag = lag[("ALL", "whole", "representative_p50_frames")]
    p10_lag = lag[("ALL", "whole", "representative_p10_frames")]
    p90_lag = lag[("ALL", "whole", "representative_p90_frames")]
    lag_std = lag[("ALL", "whole", "representative_std_frames")]
    high_agreement = lag[("ALL", "whole", "confidence_high_count")]
    medium_agreement = lag[("ALL", "whole", "confidence_medium_count")]
    low_agreement = lag[("ALL", "whole", "confidence_low_count")]
    q1_high = _finite(pre_osc) and pre_osc > whole_osc + 0.02 and pre_osc > max(0.02, 2 * whole_osc)
    q2_smoother = (_finite(action_d2) and action_d2 > 0 and state_d2 / action_d2 < 0.7) or (
        _finite(pre_osc) and pre_osc > 0.01 and state_osc < 0.5 * pre_osc
    )
    q3_positive = _finite(median_lag) and median_lag > 0 and p10_lag >= 0
    q4_variable = _finite(p90_lag) and _finite(p10_lag) and ((p90_lag - p10_lag) > 2 or lag_std > 1.5)
    lines = [
        f"Q1. Pre-grasp demo substantially more oscillatory? {'YES' if q1_high else 'NO / weak evidence'} "
        f"(pooled whole={_fmt(whole_osc)}, pre-grasp={_fmt(pre_osc)}).",
        f"Q2. Measured state substantially smoother near grasp? {'YES' if q2_smoother else 'NO / mixed'} "
        f"(delta2 p95 action={_fmt(action_d2)}, state={_fmt(state_d2)}; oscillation state={_fmt(state_osc)}).",
        f"Q3. Consistent positive action-to-state lag? {'YES' if q3_positive else 'NO / not clearly'} "
        f"(whole representative median={_fmt(median_lag)} frames, p10={_fmt(p10_lag)}, p90={_fmt(p90_lag)}; "
        f"three-metric confidence HIGH/MEDIUM/LOW={int(high_agreement)}/{int(medium_agreement)}/{int(low_agreement)}).",
        f"Q4. Lag highly variable across episodes? {'YES' if q4_variable else 'NO / relatively concentrated'} "
        f"(p90-p10={_fmt(p90_lag - p10_lag)} frames, std={_fmt(lag_std)}).",
    ]
    if q1_high and q4_variable:
        lines.append("Overall: Both command-quality and temporal synchronization are plausible contributors.")
    elif q1_high:
        lines.append("Overall: Human commands contain substantial local corrective motion; the controller may attenuate it, while training targets retain it.")
    elif q4_variable:
        lines.append("Overall: Demonstration commands are locally smooth, but episode-varying command/state alignment is a plausible contributor.")
    else:
        lines.append("Overall: Neither local demo oscillation nor lag inconsistency strongly explains policy hunting; model/inference/execution remains more likely.")
    lines.append("Interpretation uses conservative heuristics; raw pooled counts, episode distributions, and lag curves are authoritative.")
    return lines


def plot_summary(output_dir: Path, summary_rows: Sequence[Mapping[str, Any]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = _summary_lookup(summary_rows)
    windows = ("whole", "approach", "pregrasp")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    specs = (("delta_p95", "Delta q p95"), ("delta2_p95", "Delta2 q p95"), ("oscillation_pooled", "Oscillation ratio"))
    x = np.arange(len(windows))
    for axis, (suffix, title) in zip(axes, specs, strict=True):
        for offset, group in zip((-0.25, 0, 0.25), ("ALL", "CUP", "BOWL"), strict=True):
            row = lookup[(group, f"action_{suffix}")]
            axis.bar(x + offset, [row[window] for window in windows], width=0.24, label=group)
        state_row = lookup[("ALL", f"state_{suffix}")]
        axis.plot(x, [state_row[window] for window in windows], "ko--", label="ALL state")
        axis.set_title(title)
        axis.set_xticks(x, windows)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("rad/frame")
    axes[1].set_ylabel("rad/frame^2")
    axes[2].legend(fontsize=8)
    path = output_dir / "pregrasp_smoothness_summary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_lag_histogram(output_dir: Path, records: Mapping[int, dict[str, Any]], groups: Mapping[str, Sequence[int]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, window in zip(axes, ("whole", "pregrasp"), strict=True):
        for group in ("CUP", "BOWL"):
            values = [records[episode]["lags"][window].representative for episode in groups[group]]
            values = [value for value in values if value is not None]
            axis.hist(values, bins=np.arange(-3.5, 11.5, 1), alpha=0.55, label=group)
        axis.set_title(f"{window} representative lag")
        axis.set_xlabel("lag [frames], positive = action -> future state")
        axis.set_ylabel("episodes")
        axis.grid(alpha=0.2)
        axis.legend()
    path = output_dir / "action_state_lag_histogram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def select_lag_examples(records: Mapping[int, dict[str, Any]]) -> list[tuple[str, int]]:
    available = [(episode, record["lags"]["whole"]) for episode, record in records.items() if record["lags"]["whole"].representative is not None]
    if not available:
        return []
    values = np.asarray([result.representative for _, result in available], dtype=float)
    median = float(np.median(values))
    selected = [
        ("median", min(available, key=lambda pair: abs(pair[1].representative - median))[0]),
        ("lowest", min(available, key=lambda pair: pair[1].representative)[0]),
        ("highest", max(available, key=lambda pair: pair[1].representative)[0]),
    ]
    low = next((episode for episode, result in available if result.confidence == "LOW"), None)
    if low is not None:
        selected.append(("low-confidence", low))
    return selected


def plot_lag_examples(output_dir: Path, records: Mapping[int, dict[str, Any]]) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = select_lag_examples(records)
    if not examples:
        return None
    fig, axes = plt.subplots(len(examples), 3, figsize=(14, 3.4 * len(examples)), squeeze=False, constrained_layout=True)
    for row, (label, episode) in enumerate(examples):
        result = records[episode]["lags"]["whole"]
        lags = [point.lag for point in result.points]
        for axis, attribute, title in zip(axes[row], ("position_rmse", "velocity_corr", "velocity_rmse"), ("position RMSE", "velocity correlation", "velocity RMSE"), strict=True):
            axis.plot(lags, [getattr(point, attribute) for point in result.points], "o-")
            axis.grid(alpha=0.25)
            axis.set_title(f"{label}: ep {episode} — {title}")
            axis.set_xlabel("lag [frames]")
    path = output_dir / "action_state_lag_curve_examples.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_worst_pregrasp(
    output_dir: Path,
    records: Mapping[int, dict[str, Any]],
    action_sequences: Mapping[int, np.ndarray],
    state_sequences: Mapping[int, np.ndarray],
    gripper_index: int,
    action_left: Sequence[int],
    state_left: Sequence[int],
    fps: float,
    count: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(
        records,
        key=lambda episode: max(
            records[episode]["metrics"]["action"]["pregrasp"].oscillation
            if _finite(records[episode]["metrics"]["action"]["pregrasp"].oscillation) else -math.inf,
            records[episode]["metrics"]["action"]["pregrasp"].delta2_stats["p95"],
        ),
        reverse=True,
    )[:count]
    paths: list[Path] = []
    for rank, episode in enumerate(ranked, start=1):
        action, state = action_sequences[episode], state_sequences[episode]
        grasp = records[episode]["grasp"].frame
        time = np.arange(len(action)) / fps
        action_metrics = compute_window_metrics(action, action_left, 0, len(action), 0.005)
        state_metrics = compute_window_metrics(state, state_left, 0, len(state), 0.005)
        fig, axes = plt.subplots(5, 1, figsize=(14, 15), constrained_layout=True)
        axes[0].plot(time, action[:, list(action_left)], linewidth=0.8)
        axes[0].set_title("Left-arm action trajectories")
        axes[1].plot(time, state[:, list(state_left)], linewidth=0.8)
        axes[1].set_title("Left-arm measured state trajectories")
        axes[2].plot(time[1:], action_metrics.delta_max, label="action")
        axes[2].plot(time[1:], state_metrics.delta_max, label="state", alpha=0.8)
        axes[2].set_title("max |Delta q|")
        axes[3].plot(time[2:], action_metrics.delta2_max, label="action")
        axes[3].plot(time[2:], state_metrics.delta2_max, label="state", alpha=0.8)
        axes[3].set_title("max |Delta2 q|")
        axes[4].plot(time, action[:, gripper_index], color="black")
        axes[4].set_title("Left-gripper command")
        for axis in axes:
            if grasp is not None:
                axis.axvline(grasp / fps, color="red", linestyle="--", label="grasp")
            axis.grid(alpha=0.25)
        axes[2].legend()
        axes[3].legend()
        axes[4].set_xlabel("time [s]")
        fig.suptitle(f"Worst pre-grasp rank {rank}: episode {episode} — {records[episode]['task']}")
        path = output_dir / f"worst_pregrasp_episode_{episode:03d}.png"
        fig.savefig(path, dpi=145)
        plt.close(fig)
        paths.append(path)
    return paths


def run_self_tests() -> None:
    frames = 120
    monotonic = np.linspace(0, 1, frames)[:, None]
    smooth = compute_window_metrics(monotonic, (0,), 0, frames, 0.005)
    assert smooth.oscillation == 0.0
    alternating = ((-1.0) ** np.arange(frames))[:, None] * 0.1
    rough = compute_window_metrics(alternating, (0,), 0, frames, 0.005)
    assert rough.oscillation > 0.95

    rng = np.random.default_rng(7)
    action = np.cumsum(rng.normal(0, 0.03, size=(frames, 2)), axis=0)
    state = np.empty_like(action)
    state[:2] = action[0]
    state[2:] = action[:-2]
    result = estimate_lag(action, state, (0, 1), (0, 1), 0, frames, -3, 10, 0.005)
    assert result.best_position == result.best_corr == result.best_velocity_rmse == 2
    noisy = state + rng.normal(0, 0.001, size=state.shape)
    noisy_result = estimate_lag(action, noisy, (0, 1), (0, 1), 0, frames, -3, 10, 0.005)
    assert abs(noisy_result.representative - 2) <= 1

    variability = lag_distribution_stats([0, 2, 4, 6])
    assert variability["p90"] - variability["p10"] > 4

    direction = GripperDirection(True, 0.0, 1.0, 1.0, 1.0)
    gripper = np.r_[np.ones(30), np.linspace(1, 0, 5), np.zeros(30)]
    grasp = detect_grasp_close(gripper, direction, hold_frames=3, transition_min_fraction=0.25, confirm_window_frames=15, min_close_fraction=0.5)
    assert grasp.detected and 30 <= grasp.frame <= 34
    no_grasp = detect_grasp_close(np.ones(60), direction, hold_frames=3, transition_min_fraction=0.25, confirm_window_frames=15, min_close_fraction=0.5)
    assert not no_grasp.detected

    episode_a = compute_window_metrics(np.array([[0.0], [1.0]]), (0,), 0, 2, 0.005)
    episode_b = compute_window_metrics(np.array([[100.0], [101.0]]), (0,), 0, 2, 0.005)
    pooled = aggregate_metrics([episode_a, episode_b])
    assert pooled["delta_max"] == 1.0
    print("Synthetic tests passed: smooth, alternating, exact/noisy +2 lag, lag variability, grasp/no-grasp, and episode boundaries.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-repo-id")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rby1_pregrasp_temporal_analysis"))
    parser.add_argument("--action-feature-key")
    parser.add_argument("--state-feature-key")
    parser.add_argument("--pregrasp-seconds", type=float, default=2.0)
    parser.add_argument("--approach-start-seconds", type=float, default=4.0)
    parser.add_argument("--approach-end-seconds", type=float, default=2.0)
    parser.add_argument("--grasp-local-seconds", type=float, default=0.5)
    parser.add_argument("--oscillation-threshold", type=float, default=0.005)
    parser.add_argument("--lag-min-frames", type=int, default=-3)
    parser.add_argument("--lag-max-frames", type=int, default=10)
    parser.add_argument("--lag-motion-threshold", type=float, default=0.005)
    parser.add_argument("--lag-min-samples", type=int, default=5)
    parser.add_argument("--gripper-hold-frames", type=int, default=3)
    parser.add_argument("--gripper-transition-min-fraction", type=float, default=0.25)
    parser.add_argument("--gripper-confirm-window-frames", type=int, default=20)
    parser.add_argument("--gripper-min-close-fraction", type=float, default=0.5)
    parser.add_argument("--worst-count", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test and (args.dataset_root is None or args.dataset_repo_id is None):
        parser.error("--dataset-root and --dataset-repo-id are required unless --self-test is used")
    if args.fps <= 0 or args.pregrasp_seconds <= 0:
        parser.error("FPS and window durations must be positive")
    if not 0 < args.gripper_transition_min_fraction < 0.5:
        parser.error("--gripper-transition-min-fraction must be in (0, 0.5)")
    if not 0 < args.gripper_min_close_fraction <= 1:
        parser.error("--gripper-min-close-fraction must be in (0, 1]")
    if args.lag_min_frames > args.lag_max_frames:
        parser.error("--lag-min-frames must not exceed --lag-max-frames")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_tests()
        return 0

    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, download_videos=False)
    action_key = resolve_feature_key(dataset.features, args.action_feature_key, "action", required=True)
    state_key = resolve_feature_key(dataset.features, args.state_feature_key, "state", required=True)
    action_mapping = build_dimension_mapping(action_key, dataset.features[action_key])
    state_mapping = build_dimension_mapping(state_key, dataset.features[state_key])
    batch, sequences = load_numeric_data(dataset, action_key, state_key)
    sanity_check(dataset, batch, action_key, state_key, args.fps)
    print_mapping("ACTION", action_mapping)
    print_mapping("STATE", state_mapping)
    if tuple(action_mapping.names[index] for index in action_mapping.left_arm) != tuple(
        state_mapping.names[index] for index in state_mapping.left_arm
    ):
        raise ValueError("Action/state left-arm metadata names are not aligned")

    tasks = episode_tasks(dataset)
    gripper_index = action_mapping.left_gripper[0]
    gripper_sequences = {episode: values[:, gripper_index] for episode, values in sequences[action_key].items()}
    direction, gripper_stats = infer_gripper_direction(gripper_sequences, args.fps)
    print("\nLeft-gripper distribution")
    print("  " + ", ".join(f"{key}={value:.6g}" for key, value in gripper_stats.items()))
    print(
        f"  inferred open={'HIGH' if direction.open_is_high else 'LOW'}, close={'LOW' if direction.open_is_high else 'HIGH'}; "
        f"initial median={direction.initial_median:.6g}, episode agreement={direction.agreement_fraction:.1%}"
    )
    print(f"\nLag movement mask: ||Delta action_left[t]||_2 > {args.lag_motion_threshold:g} rad/frame")

    records: dict[int, dict[str, Any]] = {}
    grasp_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    lag_curve_rows: list[dict[str, Any]] = []
    motion_bin_rows: list[dict[str, Any]] = []
    for episode in sorted(sequences[action_key]):
        action = sequences[action_key][episode]
        state = sequences[state_key][episode]
        task = tasks.get(episode, "")
        grasp = detect_grasp_close(
            action[:, gripper_index],
            direction,
            hold_frames=args.gripper_hold_frames,
            transition_min_fraction=args.gripper_transition_min_fraction,
            confirm_window_frames=args.gripper_confirm_window_frames,
            min_close_fraction=args.gripper_min_close_fraction,
        )
        bounds = window_bounds(len(action), grasp.frame, args.fps, args)
        metrics = {feature: {} for feature in ("action", "state")}
        for feature, values, indices in (
            ("action", action, action_mapping.left_arm),
            ("state", state, state_mapping.left_arm),
        ):
            for window, (start, end) in bounds.items():
                metrics[feature][window] = compute_window_metrics(values, indices, start, end, args.oscillation_threshold)
        lags = {
            window: estimate_lag(
                action,
                state,
                action_mapping.left_arm,
                state_mapping.left_arm,
                *bounds[window],
                args.lag_min_frames,
                args.lag_max_frames,
                args.lag_motion_threshold,
                args.lag_min_samples,
            )
            for window in ("whole", "pregrasp")
        }
        records[episode] = {"task": task, "task_group": _task_group(task), "grasp": grasp, "bounds": bounds, "metrics": metrics, "lags": lags}

        episode_gripper_quantiles = np.quantile(
            action[:, gripper_index], (0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0)
        )
        grasp_row = {
            "episode_index": episode,
            "task": task,
            "num_frames": len(action),
            "grasp_frame_action": grasp.frame,
            "grasp_time_sec": grasp.frame / args.fps if grasp.frame is not None else math.nan,
            "gripper_before": grasp.before,
            "gripper_after": grasp.after,
            "gripper_transition_magnitude": grasp.magnitude,
            "grasp_detected": grasp.detected,
            "detection_confidence": grasp.confidence,
            "gripper_min": grasp.gripper_min,
            "gripper_q01": float(episode_gripper_quantiles[1]),
            "gripper_q10": float(episode_gripper_quantiles[2]),
            "gripper_q50": float(episode_gripper_quantiles[3]),
            "gripper_q90": float(episode_gripper_quantiles[4]),
            "gripper_q99": float(episode_gripper_quantiles[5]),
            "gripper_max": grasp.gripper_max,
            "largest_gripper_deltas": json.dumps(grasp.largest_deltas),
        }
        grasp_rows.append(grasp_row)
        row: dict[str, Any] = dict(grasp_row)
        row["task_group"] = _task_group(task)
        for window in WINDOWS:
            row.update(_metric_columns(f"{window}_action", metrics["action"][window]))
            row.update(_metric_columns(f"{window}_state", metrics["state"][window]))
        row.update(_lag_columns("whole", lags["whole"]))
        row.update(_lag_columns("pregrasp", lags["pregrasp"]))
        episode_rows.append(row)

        pre_a, pre_s = metrics["action"]["pregrasp"], metrics["state"]["pregrasp"]
        for local_joint, (action_index, state_index) in enumerate(zip(action_mapping.left_arm, state_mapping.left_arm, strict=True)):
            a_rev, a_eligible = int(pre_a.reversals[local_joint]), int(pre_a.eligible[local_joint])
            s_rev, s_eligible = int(pre_s.reversals[local_joint]), int(pre_s.eligible[local_joint])
            joint_rows.append(
                {
                    "episode_index": episode,
                    "task": task,
                    "joint_name": action_mapping.names[action_index],
                    "action_dimension": action_index,
                    "state_dimension": state_index,
                    "action_delta_p95": pre_a.joint_delta_stats[local_joint]["p95"],
                    "action_delta2_p95": pre_a.joint_delta2_stats[local_joint]["p95"],
                    "action_oscillation": a_rev / a_eligible if a_eligible else math.nan,
                    "action_reversals": a_rev,
                    "action_evaluated_pairs": a_eligible,
                    "state_delta_p95": pre_s.joint_delta_stats[local_joint]["p95"],
                    "state_delta2_p95": pre_s.joint_delta2_stats[local_joint]["p95"],
                    "state_oscillation": s_rev / s_eligible if s_eligible else math.nan,
                    "state_reversals": s_rev,
                    "state_evaluated_pairs": s_eligible,
                }
            )
        for window, result in lags.items():
            for point in result.points:
                curve_row = {
                    "episode_index": episode,
                    "task": task,
                    "window": window,
                    "lag_frames": point.lag,
                    "lag_ms": point.lag * 1000 / args.fps,
                    "movement_samples": point.samples,
                    "position_rmse": point.position_rmse,
                    "velocity_corr": point.velocity_corr,
                    "velocity_magnitude_corr": point.magnitude_corr,
                    "velocity_rmse": point.velocity_rmse,
                }
                for joint, value in zip((action_mapping.names[index] for index in action_mapping.left_arm), point.joint_corr, strict=True):
                    curve_row[f"velocity_corr_{joint}"] = value
                lag_curve_rows.append(curve_row)

        if grasp.frame is not None:
            for bin_index in range(4):
                start_seconds = 2.0 - 0.5 * bin_index
                end_seconds = start_seconds - 0.5
                start = max(0, grasp.frame - int(round(start_seconds * args.fps)))
                end = max(0, grasp.frame - int(round(end_seconds * args.fps))) + (1 if bin_index == 3 else 0)
                item = compute_window_metrics(action, action_mapping.left_arm, start, min(len(action), end), args.oscillation_threshold)
                motion_bin_rows.append(
                    {
                        "episode_index": episode,
                        "task": task,
                        "bin_start_sec_before_grasp": start_seconds,
                        "bin_end_sec_before_grasp": end_seconds,
                        "delta_median": item.delta_stats["median"],
                        "delta_p95": item.delta_stats["p95"],
                        "oscillation": item.oscillation,
                        "reversals": int(item.reversals.sum()),
                        "evaluated_pairs": int(item.eligible.sum()),
                    }
                )

    detected = [episode for episode, record in records.items() if record["grasp"].detected]
    failed = [episode for episode in records if episode not in detected]
    print(f"\nGrasp detection: {len(detected)}/{len(records)} episodes")
    for episode in failed:
        grasp = records[episode]["grasp"]
        print(f"  FAILED ep {episode}: task={records[episode]['task']!r}, min/max={grasp.gripper_min:.6g}/{grasp.gripper_max:.6g}, largest deltas={grasp.largest_deltas}")

    groups = {
        "ALL": list(records),
        "CUP": [episode for episode, record in records.items() if record["task_group"] == "CUP"],
        "BOWL": [episode for episode, record in records.items() if record["task_group"] == "BOWL"],
    }
    summary_rows = build_summary_rows(records, groups)
    lag_summary_rows = build_lag_summary_rows(records, groups, args.fps)
    interpretations = interpret(summary_rows, lag_summary_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "episode_pregrasp_metrics.csv", episode_rows)
    _write_csv(args.output_dir / "lag_curves.csv", lag_curve_rows)
    _write_csv(args.output_dir / "joint_pregrasp_metrics.csv", joint_rows)
    _write_csv(args.output_dir / "grasp_detection.csv", grasp_rows)
    _write_csv(args.output_dir / "pregrasp_summary.csv", summary_rows)
    _write_csv(args.output_dir / "lag_summary.csv", lag_summary_rows)
    _write_csv(args.output_dir / "pregrasp_motion_bins.csv", motion_bin_rows)
    (args.output_dir / "interpretation.txt").write_text("\n".join(interpretations) + "\n", encoding="utf-8")
    plot_paths = [
        plot_summary(args.output_dir, summary_rows),
        plot_lag_histogram(args.output_dir, records, groups),
    ]
    lag_examples = plot_lag_examples(args.output_dir, records)
    if lag_examples is not None:
        plot_paths.append(lag_examples)
    plot_paths.extend(
        plot_worst_pregrasp(
            args.output_dir,
            records,
            sequences[action_key],
            sequences[state_key],
            gripper_index,
            action_mapping.left_arm,
            state_mapping.left_arm,
            args.fps,
            args.worst_count,
        )
    )

    lookup = _summary_lookup(summary_rows)
    print("\nWhole vs Approach vs Pre-grasp (pooled left-arm metrics)")
    for group in ("ALL", "CUP", "BOWL"):
        print(f"  {group}")
        print(f"    {'metric':<28}{'whole':>12}{'approach':>12}{'pregrasp':>12}")
        for metric in ("action_delta_p95", "action_delta2_p95", "action_oscillation_pooled", "state_delta2_p95", "state_oscillation_pooled"):
            row = lookup[(group, metric)]
            print(f"    {metric:<28}{_fmt(row['whole']):>12}{_fmt(row['approach']):>12}{_fmt(row['pregrasp']):>12}")

    lag_lookup = {(row["group"], row["window"], row["metric"]): row["value"] for row in lag_summary_rows}
    print("\nRepresentative action -> state lag distribution [frames]")
    for group in ("ALL", "CUP", "BOWL"):
        for window in ("whole", "pregrasp"):
            values = [lag_lookup[(group, window, f"representative_{metric}_frames")] for metric in ("mean", "std", "p10", "p50", "p90", "min", "max")]
            print(f"  {group:4s} {window:9s}: mean={_fmt(values[0])}, std={_fmt(values[1])}, p10/p50/p90={_fmt(values[2])}/{_fmt(values[3])}/{_fmt(values[4])}, min/max={_fmt(values[5])}/{_fmt(values[6])}")

    worst_delta2 = sorted(records, key=lambda episode: records[episode]["metrics"]["action"]["pregrasp"].delta2_stats["p95"], reverse=True)[: args.worst_count]
    worst_oscillation = sorted(
        records,
        key=lambda episode: records[episode]["metrics"]["action"]["pregrasp"].oscillation
        if _finite(records[episode]["metrics"]["action"]["pregrasp"].oscillation)
        else -math.inf,
        reverse=True,
    )[: args.worst_count]
    whole_median = lag_lookup[("ALL", "whole", "representative_p50_frames")]
    pregrasp_median = lag_lookup[("ALL", "pregrasp", "representative_p50_frames")]
    whole_outliers = sorted(
        [episode for episode in records if records[episode]["lags"]["whole"].representative != whole_median],
        key=lambda episode: abs(records[episode]["lags"]["whole"].representative - whole_median),
        reverse=True,
    )
    pregrasp_outliers = sorted(
        [episode for episode in records if records[episode]["lags"]["pregrasp"].representative != pregrasp_median],
        key=lambda episode: abs(records[episode]["lags"]["pregrasp"].representative - pregrasp_median),
        reverse=True,
    )
    print(f"\nWorst pre-grasp delta2 episodes: {worst_delta2}")
    print(f"Worst pre-grasp oscillation episodes: {worst_oscillation}")
    print(
        "Whole-lag non-median episodes: "
        f"{[(episode, records[episode]['lags']['whole'].representative, records[episode]['lags']['whole'].confidence) for episode in whole_outliers]}"
    )
    print(
        "Pre-grasp-lag non-median episodes: "
        f"{[(episode, records[episode]['lags']['pregrasp'].representative, records[episode]['lags']['pregrasp'].confidence) for episode in pregrasp_outliers]}"
    )
    print("\nAutomatic interpretation")
    for line in interpretations:
        print(f"  {line}")
    print(f"\nOutputs: {args.output_dir.resolve()}")
    for path in sorted(args.output_dir.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
