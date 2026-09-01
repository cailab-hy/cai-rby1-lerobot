#!/usr/bin/env python3
"""Read-only temporal-smoothness diagnostics for RB-Y1 LeRobot datasets.

The script deliberately computes differences inside each episode.  It never
modifies a dataset and it does not decode image/video features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_SMOLVLA_REFERENCE = {
    "arm_delta_median": 0.02407157,
    "arm_delta_p95": 0.053782679,
    "arm_delta_max": 0.056116834,
    "arm_delta2_median": 0.027014613,
    "arm_delta2_p95": 0.052834749,
    "arm_delta2_max": 0.05941534,
    "right_arm_oscillation": 0.90625,
    "left_arm_oscillation": 0.24117647,
}

STAT_PERCENTILES = (50, 90, 95, 99)
JOINT_STAT_PERCENTILES = (50, 95, 99)


@dataclass(frozen=True)
class DimensionMapping:
    names: tuple[str, ...]
    right_arm: tuple[int, ...]
    left_arm: tuple[int, ...]
    right_gripper: tuple[int, ...]
    left_gripper: tuple[int, ...]

    @property
    def arm(self) -> tuple[int, ...]:
        return self.right_arm + self.left_arm

    @property
    def gripper(self) -> tuple[int, ...]:
        return self.right_gripper + self.left_gripper

    def group_for_index(self, index: int) -> str:
        for group in ("right_arm", "left_arm", "right_gripper", "left_gripper"):
            if index in getattr(self, group):
                return group
        raise KeyError(index)


@dataclass
class EpisodeFeatureMetrics:
    values: np.ndarray
    delta: np.ndarray
    delta2: np.ndarray
    arm_delta_max: np.ndarray
    arm_delta2_max: np.ndarray
    arm_delta_stats: dict[str, float]
    arm_delta2_stats: dict[str, float]
    oscillation_counts: dict[int, tuple[int, int]]


@dataclass
class FeatureAnalysis:
    key: str
    mapping: DimensionMapping
    episodes: dict[int, EpisodeFeatureMetrics]
    arm_delta_stats: dict[str, float]
    arm_delta2_stats: dict[str, float]
    joint_rows: list[dict[str, Any]]
    gripper_rows: list[dict[str, Any]]
    group_oscillation: dict[str, tuple[int, int, float]]


def temporal_differences(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return first and second frame differences for one episode."""
    q = np.asarray(sequence, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"Expected a [frames, dimensions] array, got shape {q.shape}")
    return np.diff(q, axis=0), np.diff(q, n=2, axis=0)


def oscillation_counts(delta: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return sign-reversal and eligible-pair counts for every dimension."""
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"Expected a [transitions, dimensions] array, got shape {d.shape}")
    if threshold < 0:
        raise ValueError("Oscillation threshold must be non-negative")
    if len(d) < 2:
        zeros = np.zeros(d.shape[1], dtype=np.int64)
        return zeros.copy(), zeros
    previous, current = d[:-1], d[1:]
    eligible = (np.abs(previous) > threshold) & (np.abs(current) > threshold)
    reversals = eligible & (np.signbit(previous) != np.signbit(current))
    return reversals.sum(axis=0), eligible.sum(axis=0)


def descriptive_stats(values: np.ndarray, percentiles: Sequence[int] = STAT_PERCENTILES) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    keys = ["median" if percentile == 50 else f"p{percentile}" for percentile in percentiles]
    if flat.size == 0:
        return {**dict.fromkeys(keys, math.nan), "max": math.nan}
    result = {
        "median" if percentile == 50 else f"p{percentile}": float(np.percentile(flat, percentile))
        for percentile in percentiles
    }
    result["max"] = float(np.max(flat))
    return result


def safe_ratio(numerator: float, denominator: float) -> float:
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return math.nan
    if denominator == 0:
        return math.inf if numerator > 0 else math.nan
    return numerator / denominator


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_dimension_mapping(feature_key: str, feature: Mapping[str, Any]) -> DimensionMapping:
    """Classify dimensions solely from metadata names, never from positions."""
    shape = tuple(feature.get("shape") or ())
    if len(shape) != 1:
        raise ValueError(f"Feature {feature_key!r} must be a one-dimensional vector; metadata shape={shape}")
    dimension = int(shape[0])
    names = feature.get("names")
    if not isinstance(names, (list, tuple)) or len(names) != dimension:
        raise ValueError(
            f"Feature {feature_key!r} needs one metadata name per dimension to safely map RB-Y1 joints; "
            f"shape={shape}, names={names!r}"
        )

    groups: dict[str, list[int]] = defaultdict(list)
    unknown: list[str] = []
    for index, raw_name in enumerate(names):
        name = _normalized_name(str(raw_name))
        side = "right" if "right" in name else "left" if "left" in name else None
        is_gripper = "gripper" in name or "grip" in name
        if side is None:
            unknown.append(str(raw_name))
        elif is_gripper:
            groups[f"{side}_gripper"].append(index)
        elif "arm" in name or "joint" in name:
            groups[f"{side}_arm"].append(index)
        else:
            unknown.append(str(raw_name))

    expected = {"right_arm": 7, "left_arm": 7, "right_gripper": 1, "left_gripper": 1}
    counts = {group: len(groups[group]) for group in expected}
    if unknown or counts != expected:
        raise ValueError(
            f"Could not safely derive the RB-Y1 7+7 arm and 1+1 gripper mapping for {feature_key!r} "
            f"from metadata names. counts={counts}, unclassified={unknown}. names={list(names)}"
        )
    return DimensionMapping(
        names=tuple(str(name) for name in names),
        right_arm=tuple(groups["right_arm"]),
        left_arm=tuple(groups["left_arm"]),
        right_gripper=tuple(groups["right_gripper"]),
        left_gripper=tuple(groups["left_gripper"]),
    )


def resolve_feature_key(
    features: Mapping[str, Mapping[str, Any]], requested: str | None, role: str, required: bool
) -> str | None:
    if requested:
        if requested not in features:
            raise ValueError(f"Requested {role} feature {requested!r} does not exist. Available: {list(features)}")
        return requested

    canonical = "action" if role == "action" else "observation.state"
    if canonical in features:
        return canonical
    tokens = ("action", "command") if role == "action" else ("state", "proprio")
    candidates = [
        key
        for key, feature in features.items()
        if any(token in key.lower() for token in tokens) and len(tuple(feature.get("shape") or ())) == 1
    ]
    if len(candidates) == 1:
        return candidates[0]
    if required:
        raise ValueError(
            f"Could not uniquely discover the {role} vector feature. Candidates={candidates}; "
            f"available={list(features)}. Pass --{role}-feature-key explicitly."
        )
    if candidates:
        print(f"WARNING: ambiguous state candidates {candidates}; observation state analysis is skipped.")
    else:
        print("WARNING: no observation state vector found; state analysis is skipped.")
    return None


def _episode_metric(values: np.ndarray, mapping: DimensionMapping, threshold: float) -> EpisodeFeatureMetrics:
    delta, delta2 = temporal_differences(values)
    arm_indices = list(mapping.arm)
    arm_delta_max = np.max(np.abs(delta[:, arm_indices]), axis=1) if len(delta) else np.empty(0)
    arm_delta2_max = np.max(np.abs(delta2[:, arm_indices]), axis=1) if len(delta2) else np.empty(0)
    reversal, eligible = oscillation_counts(delta, threshold)
    return EpisodeFeatureMetrics(
        values=np.asarray(values, dtype=np.float64),
        delta=delta,
        delta2=delta2,
        arm_delta_max=arm_delta_max,
        arm_delta2_max=arm_delta2_max,
        arm_delta_stats=descriptive_stats(arm_delta_max),
        arm_delta2_stats=descriptive_stats(arm_delta2_max),
        oscillation_counts={
            index: (int(reversal[index]), int(eligible[index])) for index in range(len(mapping.names))
        },
    )


def analyze_episode_sequences(
    feature_key: str,
    episode_sequences: Mapping[int, np.ndarray],
    mapping: DimensionMapping,
    threshold: float,
) -> FeatureAnalysis:
    """Analyze pre-split episodes, preserving boundaries in every metric."""
    episodes = {
        int(episode_index): _episode_metric(sequence, mapping, threshold)
        for episode_index, sequence in episode_sequences.items()
    }
    all_arm_delta = _concatenate([episode.arm_delta_max for episode in episodes.values()])
    all_arm_delta2 = _concatenate([episode.arm_delta2_max for episode in episodes.values()])

    joint_rows: list[dict[str, Any]] = []
    gripper_rows: list[dict[str, Any]] = []
    totals: dict[int, list[int]] = {index: [0, 0] for index in range(len(mapping.names))}
    for episode in episodes.values():
        for index, (reversals, eligible) in episode.oscillation_counts.items():
            totals[index][0] += reversals
            totals[index][1] += eligible

    for index, joint_name in enumerate(mapping.names):
        group = mapping.group_for_index(index)
        delta_abs = _concatenate([np.abs(episode.delta[:, index]) for episode in episodes.values()])
        delta2_abs = _concatenate([np.abs(episode.delta2[:, index]) for episode in episodes.values()])
        delta_stats = descriptive_stats(delta_abs, JOINT_STAT_PERCENTILES)
        delta2_stats = descriptive_stats(delta2_abs, JOINT_STAT_PERCENTILES)
        reversals, eligible = totals[index]
        row = {
            "feature_key": feature_key,
            "joint_name": joint_name,
            "dimension_index": index,
            "group": group,
            "num_delta": int(delta_abs.size),
            **{f"delta_abs_{key}": value for key, value in delta_stats.items()},
            "num_delta2": int(delta2_abs.size),
            **{f"delta2_abs_{key}": value for key, value in delta2_stats.items()},
            "oscillation_threshold": threshold,
            "oscillation_reversals": reversals,
            "oscillation_evaluated_pairs": eligible,
            "oscillation_ratio": reversals / eligible if eligible else math.nan,
        }
        (joint_rows if group.endswith("arm") else gripper_rows).append(row)

    group_oscillation: dict[str, tuple[int, int, float]] = {}
    for group in ("right_arm", "left_arm"):
        indices = getattr(mapping, group)
        reversals = sum(totals[index][0] for index in indices)
        eligible = sum(totals[index][1] for index in indices)
        group_oscillation[group] = (reversals, eligible, reversals / eligible if eligible else math.nan)

    return FeatureAnalysis(
        key=feature_key,
        mapping=mapping,
        episodes=episodes,
        arm_delta_stats=descriptive_stats(all_arm_delta),
        arm_delta2_stats=descriptive_stats(all_arm_delta2),
        joint_rows=joint_rows,
        gripper_rows=gripper_rows,
        group_oscillation=group_oscillation,
    )


def _concatenate(arrays: Iterable[np.ndarray]) -> np.ndarray:
    nonempty = [np.asarray(array) for array in arrays if np.asarray(array).size]
    return np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float64)


def load_episode_sequences(
    dataset: Any, feature_keys: Sequence[str]
) -> tuple[dict[str, dict[int, np.ndarray]], dict[int, int]]:
    """Load only numeric columns and split/sort them by episode."""
    hf_dataset = dataset.hf_dataset
    required_columns = [*feature_keys, "episode_index"]
    if "frame_index" in hf_dataset.column_names:
        required_columns.append("frame_index")
    view = hf_dataset.select_columns(required_columns).with_format("numpy")
    batch = view[:]
    episode_indices = np.asarray(batch["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices = (
        np.asarray(batch["frame_index"], dtype=np.int64).reshape(-1)
        if "frame_index" in batch
        else np.arange(len(episode_indices), dtype=np.int64)
    )
    result: dict[str, dict[int, np.ndarray]] = {key: {} for key in feature_keys}
    frame_counts: dict[int, int] = {}
    for episode_index in sorted(int(value) for value in np.unique(episode_indices)):
        positions = np.flatnonzero(episode_indices == episode_index)
        positions = positions[np.argsort(frame_indices[positions], kind="stable")]
        frame_counts[episode_index] = int(len(positions))
        for key in feature_keys:
            values = np.asarray(batch[key])[positions]
            if values.ndim != 2:
                raise ValueError(f"Loaded feature {key!r} has shape {values.shape}; expected [frames, dimensions]")
            result[key][episode_index] = values.astype(np.float64, copy=False)
    return result, frame_counts


def episode_tasks(dataset: Any) -> dict[int, str]:
    tasks: dict[int, str] = {}
    metadata = getattr(dataset.meta, "episodes", None)
    if metadata is None:
        return tasks
    for row_index in range(len(metadata)):
        row = metadata[row_index]
        episode_index = int(row.get("episode_index", row_index))
        raw_tasks = row.get("tasks", [])
        if isinstance(raw_tasks, str):
            tasks[episode_index] = raw_tasks
        elif raw_tasks:
            tasks[episode_index] = " | ".join(str(task) for task in raw_tasks)
    return tasks


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_episode_rows(
    frame_counts: Mapping[int, int], tasks: Mapping[int, str], analyses: Mapping[str, FeatureAnalysis]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_index in sorted(frame_counts):
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "num_frames": frame_counts[episode_index],
            "tasks": tasks.get(episode_index, ""),
        }
        for label in ("action", "state"):
            analysis = analyses.get(label)
            if analysis is None or episode_index not in analysis.episodes:
                for order in ("delta", "delta2"):
                    for statistic in ("median", "p90", "p95", "p99", "max"):
                        row[f"{label}_arm_{order}_{statistic}"] = math.nan
                row[f"{label}_right_arm_oscillation_ratio"] = math.nan
                row[f"{label}_left_arm_oscillation_ratio"] = math.nan
                continue
            metrics = analysis.episodes[episode_index]
            for order, stats in (("delta", metrics.arm_delta_stats), ("delta2", metrics.arm_delta2_stats)):
                for statistic, value in stats.items():
                    row[f"{label}_arm_{order}_{statistic}"] = value
            for group in ("right_arm", "left_arm"):
                reversals = sum(metrics.oscillation_counts[index][0] for index in getattr(analysis.mapping, group))
                eligible = sum(metrics.oscillation_counts[index][1] for index in getattr(analysis.mapping, group))
                row[f"{label}_{group}_oscillation_ratio"] = reversals / eligible if eligible else math.nan
        # Keep the short, requested column names as aliases for action metrics.
        row["right_arm_oscillation_ratio"] = row["action_right_arm_oscillation_ratio"]
        row["left_arm_oscillation_ratio"] = row["action_left_arm_oscillation_ratio"]
        rows.append(row)
    return rows


def _format_number(value: float, width: int = 14) -> str:
    if value is None or not math.isfinite(value):
        text = "inf" if value is not None and math.isinf(value) else "N/A"
    else:
        text = f"{value:.8g}"
    return f"{text:>{width}}"


def print_schema(dataset: Any, selected: Mapping[str, str | None], mappings: Mapping[str, DimensionMapping]) -> None:
    print("\nDataset schema")
    print(f"  repo_id: {dataset.repo_id}")
    print(f"  root: {dataset.root}")
    print(f"  episodes: {dataset.num_episodes}, frames: {dataset.num_frames}, metadata fps: {dataset.fps}")
    for key, feature in dataset.features.items():
        print(f"  {key}: dtype={feature.get('dtype')}, shape={tuple(feature.get('shape') or ())}")
    for label, key in selected.items():
        if key is None:
            continue
        print(f"\n{label.upper()} mapping: feature={key!r}, dimension={len(mappings[label].names)}")
        mapping = mappings[label]
        for index, name in enumerate(mapping.names):
            print(f"  [{index:02d}] {name:<24} -> {mapping.group_for_index(index)}")


def print_feature_analysis(label: str, analysis: FeatureAnalysis) -> None:
    print(f"\n{label.upper()} dataset-wide arm metrics (per-frame max over 14 arm joints)")
    print("  metric       median          p90          p95          p99          max")
    for metric, stats in (("|delta|", analysis.arm_delta_stats), ("|delta2|", analysis.arm_delta2_stats)):
        print(f"  {metric:<8}" + "".join(_format_number(stats[key]) for key in ("median", "p90", "p95", "p99", "max")))

    print(f"\n{label.upper()} per-arm-joint metrics")
    for row in analysis.joint_rows:
        print(f"  {row['joint_name']} ({row['group']})")
        print(
            f"    |delta|  median={row['delta_abs_median']:.8g}, p95={row['delta_abs_p95']:.8g}, "
            f"p99={row['delta_abs_p99']:.8g}, max={row['delta_abs_max']:.8g}"
        )
        print(
            f"    |delta2| median={row['delta2_abs_median']:.8g}, p95={row['delta2_abs_p95']:.8g}, "
            f"p99={row['delta2_abs_p99']:.8g}, max={row['delta2_abs_max']:.8g}"
        )
        print(
            f"    oscillation={_format_number(row['oscillation_ratio'], 0).strip()} "
            f"({row['oscillation_reversals']}/{row['oscillation_evaluated_pairs']} eligible pairs)"
        )
    for group in ("right_arm", "left_arm"):
        reversals, eligible, ratio = analysis.group_oscillation[group]
        print(
            f"  {group} aggregate oscillation: {_format_number(ratio, 0).strip()} "
            f"({reversals}/{eligible} eligible pairs)"
        )

    print(f"\n{label.upper()} gripper metrics (normalized units; excluded from arm statistics)")
    for row in analysis.gripper_rows:
        print(
            f"  {row['joint_name']}: |delta| median={row['delta_abs_median']:.8g}, "
            f"p95={row['delta_abs_p95']:.8g}, p99={row['delta_abs_p99']:.8g}, max={row['delta_abs_max']:.8g}; "
            f"|delta2| median={row['delta2_abs_median']:.8g}, p95={row['delta2_abs_p95']:.8g}, "
            f"p99={row['delta2_abs_p99']:.8g}, max={row['delta2_abs_max']:.8g}; "
            f"oscillation={_format_number(row['oscillation_ratio'], 0).strip()}"
        )


def print_episode_metrics(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\nEpisode arm metrics")
    for row in rows:
        print(
            f"  episode {int(row['episode_index'])}: frames={int(row['num_frames'])}, "
            f"tasks={row.get('tasks', '')!r}"
        )
        for label in ("action", "state"):
            delta = ", ".join(
                f"{stat}={_format_number(float(row[f'{label}_arm_delta_{stat}']), 0).strip()}"
                for stat in ("median", "p90", "p95", "p99", "max")
            )
            delta2 = ", ".join(
                f"{stat}={_format_number(float(row[f'{label}_arm_delta2_{stat}']), 0).strip()}"
                for stat in ("median", "p90", "p95", "p99", "max")
            )
            right_oscillation = _format_number(float(row[f"{label}_right_arm_oscillation_ratio"]), 0).strip()
            left_oscillation = _format_number(float(row[f"{label}_left_arm_oscillation_ratio"]), 0).strip()
            print(f"    {label:<6} |delta|  {delta}")
            print(f"           |delta2| {delta2}")
            print(f"           oscillation right={right_oscillation}, left={left_oscillation}")


def print_worst_rankings(rows: Sequence[Mapping[str, Any]], top_n: int = 10) -> None:
    criteria = (
        ("action arm delta p95", "action_arm_delta_p95"),
        ("action arm delta2 p95", "action_arm_delta2_p95"),
        ("action arm max delta", "action_arm_delta_max"),
        ("right-arm oscillation", "action_right_arm_oscillation_ratio"),
        ("left-arm oscillation", "action_left_arm_oscillation_ratio"),
    )
    print(f"\nWorst episode rankings (top {min(top_n, len(rows))})")
    for title, key in criteria:
        ranked = sorted(
            rows,
            key=lambda row: float(row[key]) if math.isfinite(float(row[key])) else -math.inf,
            reverse=True,
        )[:top_n]
        rendered = ", ".join(
            f"ep {int(row['episode_index'])}: {_format_number(float(row[key]), 0).strip()}"
            for row in ranked
        )
        print(f"  {title}: {rendered}")


def _summary_value(analysis: FeatureAnalysis | None, order: str, statistic: str) -> float:
    if analysis is None:
        return math.nan
    stats = analysis.arm_delta_stats if order == "delta" else analysis.arm_delta2_stats
    return stats[statistic]


def print_comparison(analyses: Mapping[str, FeatureAnalysis], reference: Mapping[str, float] | None) -> None:
    action, state = analyses.get("action"), analyses.get("state")
    if reference is None:
        print("\nSmolVLA comparison disabled.")
        return
    metrics = (
        ("Arm delta median", "delta", "median", "arm_delta_median"),
        ("Arm delta p95", "delta", "p95", "arm_delta_p95"),
        ("Arm delta p99", "delta", "p99", None),
        ("Arm delta max", "delta", "max", "arm_delta_max"),
        ("Arm delta2 median", "delta2", "median", "arm_delta2_median"),
        ("Arm delta2 p95", "delta2", "p95", "arm_delta2_p95"),
        ("Arm delta2 p99", "delta2", "p99", None),
        ("Arm delta2 max", "delta2", "max", "arm_delta2_max"),
    )
    print("\nComparison (raw per-frame robot units)")
    print(f"  {'Metric':<25}{'DEMO ACTION':>15}{'DEMO STATE':>15}{'FIXED SMOLVLA':>17}")
    print("  " + "-" * 72)
    for title, order, statistic, ref_key in metrics:
        ref_value = float(reference[ref_key]) if ref_key else math.nan
        print(
            f"  {title:<25}{_format_number(_summary_value(action, order, statistic), 15)}"
            f"{_format_number(_summary_value(state, order, statistic), 15)}{_format_number(ref_value, 17)}"
        )
    for side, title in (("right", "Right oscillation"), ("left", "Left oscillation")):
        group = f"{side}_arm"
        action_value = action.group_oscillation[group][2] if action else math.nan
        state_value = state.group_oscillation[group][2] if state else math.nan
        ref_value = float(reference[f"{side}_arm_oscillation"])
        print(
            f"  {title:<25}{_format_number(action_value, 15)}"
            f"{_format_number(state_value, 15)}{_format_number(ref_value, 17)}"
        )

    print("\nSmolVLA / demonstration ratios")
    for order, label in (("delta", "Delta q p95"), ("delta2", "Delta2 q p95")):
        ref_value = float(reference[f"arm_{order}_p95"])
        for target_label, target in (("action", action), ("state", state)):
            ratio = safe_ratio(ref_value, _summary_value(target, order, "p95"))
            print(f"  {label} / demo {target_label}: {_format_number(ratio, 0).strip()}x")

    print("\nConservative diagnostic interpretation")
    action_ratios = [
        safe_ratio(float(reference["arm_delta_p95"]), _summary_value(action, "delta", "p95")),
        safe_ratio(float(reference["arm_delta2_p95"]), _summary_value(action, "delta2", "p95")),
    ]
    finite_ratios = [ratio for ratio in action_ratios if math.isfinite(ratio)]
    if finite_ratios and min(finite_ratios) >= 2.0:
        print("  Demonstrations are substantially smoother than fixed-noise SmolVLA.")
        print("  Temporal roughness is likely introduced or amplified by policy learning/inference.")
    elif finite_ratios and max(finite_ratios) <= 1.5 and min(finite_ratios) >= (1 / 1.5):
        print("  Demo action roughness is comparable to SmolVLA.")
        print("  Dataset trajectory quality may be an important contributor.")
    else:
        print("  Demo action and SmolVLA roughness differ by metric; no single source is identified.")
    if action is not None and state is not None:
        action_state_ratios = [
            safe_ratio(_summary_value(action, order, "p95"), _summary_value(state, order, "p95"))
            for order in ("delta", "delta2")
        ]
        if any(math.isfinite(ratio) and ratio >= 2.0 for ratio in action_state_ratios):
            print("  Command trajectory is noisy while the measured robot trajectory is smoother.")
            print("  Robot/controller filtering may have hidden command noise during demonstration collection.")
    print("  Treat these threshold-based statements as clues; use the raw metrics and plots for diagnosis.")


def plot_worst_episodes(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    action: FeatureAnalysis,
    state: FeatureAnalysis | None,
    fps: float,
    count: int,
) -> list[Path]:
    if count <= 0:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(
        rows,
        key=lambda row: float(row["action_arm_delta2_p95"])
        if math.isfinite(float(row["action_arm_delta2_p95"]))
        else -math.inf,
        reverse=True,
    )[:count]
    paths: list[Path] = []
    state_name_to_index = {name: index for index, name in enumerate(state.mapping.names)} if state else {}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for rank, row in enumerate(ranked, start=1):
        episode_index = int(row["episode_index"])
        episode = action.episodes[episode_index]
        time = np.arange(len(episode.values)) / fps
        fig, axes = plt.subplots(4, 1, figsize=(14, 13), constrained_layout=True)
        for axis, group, title in (
            (axes[0], "right_arm", "Right arm positions"),
            (axes[1], "left_arm", "Left arm positions"),
        ):
            for joint_offset, index in enumerate(getattr(action.mapping, group)):
                name = action.mapping.names[index]
                color = colors[joint_offset % len(colors)]
                axis.plot(
                    time,
                    episode.values[:, index],
                    label=f"action:{name}",
                    linewidth=1.1,
                    color=color,
                )
                state_index = state_name_to_index.get(name)
                if state is not None and state_index is not None and episode_index in state.episodes:
                    state_values = state.episodes[episode_index].values[:, state_index]
                    if len(state_values) == len(time):
                        axis.plot(
                            time,
                            state_values,
                            "--",
                            label=f"state:{name}",
                            linewidth=0.8,
                            alpha=0.7,
                            color=color,
                        )
            axis.set_title(title)
            axis.set_ylabel("position [rad]")
            axis.grid(alpha=0.25)
            axis.legend(ncol=4, fontsize=7)
        axes[2].plot(time[1:], episode.arm_delta_max, label="action", linewidth=1.2)
        axes[3].plot(time[2:], episode.arm_delta2_max, label="action", linewidth=1.2)
        if state is not None and episode_index in state.episodes:
            state_episode = state.episodes[episode_index]
            if len(state_episode.values) == len(time):
                axes[2].plot(time[1:], state_episode.arm_delta_max, "--", label="state", linewidth=1.0)
                axes[3].plot(time[2:], state_episode.arm_delta2_max, "--", label="state", linewidth=1.0)
        axes[2].set_title("Maximum absolute first difference over arm joints")
        axes[3].set_title("Maximum absolute second difference over arm joints")
        for axis in axes[2:]:
            axis.set_ylabel("robot units / frame")
            axis.grid(alpha=0.25)
            axis.legend()
        axes[3].set_xlabel("time [s]")
        task = str(row.get("tasks", ""))
        fig.suptitle(f"Worst rank {rank}: episode {episode_index}" + (f" — {task}" if task else ""))
        path = output_dir / f"worst_episode_{episode_index:06d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def load_reference(args: argparse.Namespace) -> dict[str, float] | None:
    if args.no_smolvla_reference:
        return None
    reference = dict(DEFAULT_SMOLVLA_REFERENCE)
    if args.smolvla_reference_json:
        with args.smolvla_reference_json.open(encoding="utf-8") as handle:
            overrides = json.load(handle)
        unknown = set(overrides) - set(reference)
        if unknown:
            raise ValueError(f"Unknown SmolVLA reference keys: {sorted(unknown)}")
        reference.update({key: float(value) for key, value in overrides.items()})
    return reference


def run_self_tests() -> None:
    mapping = DimensionMapping(
        names=tuple(
            [f"right_arm_{i}" for i in range(7)]
            + [f"left_arm_{i}" for i in range(7)]
            + ["right_gripper_0", "left_gripper_0"]
        ),
        right_arm=tuple(range(7)),
        left_arm=tuple(range(7, 14)),
        right_gripper=(14,),
        left_gripper=(15,),
    )

    constant = np.ones((4, 16))
    delta, delta2 = temporal_differences(constant)
    np.testing.assert_array_equal(delta, 0)
    np.testing.assert_array_equal(delta2, 0)

    linear = np.repeat(np.arange(4, dtype=float)[:, None], 16, axis=1)
    delta, delta2 = temporal_differences(linear)
    np.testing.assert_array_equal(delta, 1)
    np.testing.assert_array_equal(delta2, 0)

    alternating = np.zeros((5, 16))
    alternating[:, 0] = [0, 1, 0, 1, 0]
    reversal, eligible = oscillation_counts(temporal_differences(alternating)[0], 0.005)
    assert reversal[0] == 3 and eligible[0] == 3

    # If the two episodes were concatenated, the 1 -> 100 boundary would add a
    # spurious 99-unit delta.  The aggregate must contain only the two real deltas.
    boundary_sequences = {
        0: np.repeat(np.array([[0.0], [1.0]]), 16, axis=1),
        1: np.repeat(np.array([[100.0], [101.0]]), 16, axis=1),
    }
    analysis = analyze_episode_sequences("action", boundary_sequences, mapping, 0.005)
    assert analysis.arm_delta_stats["max"] == 1.0
    assert sum(episode.delta.shape[0] for episode in analysis.episodes.values()) == 2
    assert all(
        counts[1] == 0
        for episode in analysis.episodes.values()
        for counts in episode.oscillation_counts.values()
    )
    print("Self-tests passed: constant, linear, alternating, and episode-boundary cases.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", help="LeRobot dataset repo ID (for example local/rby1-table-bussing-v3)")
    parser.add_argument("--dataset-root", type=Path, help="Optional local LeRobot dataset root")
    parser.add_argument("--fps", type=float, help="Expected FPS; validated against dataset metadata")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo_smoothness"))
    parser.add_argument("--action-feature-key", help="Override automatic action-vector feature discovery")
    parser.add_argument("--state-feature-key", help="Override automatic observation-state feature discovery")
    parser.add_argument("--oscillation-delta-threshold", type=float, default=0.005)
    parser.add_argument("--worst-top-n", type=int, default=10)
    parser.add_argument("--num-worst-plots", type=int, default=5, choices=range(0, 6), metavar="0..5")
    parser.add_argument(
        "--smolvla-reference-json", type=Path, help="JSON overrides for the built-in fixed-noise reference"
    )
    parser.add_argument("--no-smolvla-reference", action="store_true")
    parser.add_argument(
        "--self-test", action="store_true", help="Run synthetic regression tests without loading a dataset"
    )
    args = parser.parse_args(argv)
    if not args.self_test and not args.dataset_repo_id:
        parser.error("--dataset-repo-id is required unless --self-test is used")
    if args.oscillation_delta_threshold < 0:
        parser.error("--oscillation-delta-threshold must be non-negative")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.worst_top_n <= 0:
        parser.error("--worst-top-n must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_tests()
        return 0

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        download_videos=False,
    )
    dataset_fps = float(dataset.fps)
    if args.fps is not None and not math.isclose(args.fps, dataset_fps, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"--fps={args.fps} does not match dataset metadata fps={dataset_fps}")
    fps = args.fps if args.fps is not None else dataset_fps

    action_key = resolve_feature_key(dataset.features, args.action_feature_key, "action", required=True)
    state_key = resolve_feature_key(dataset.features, args.state_feature_key, "state", required=False)
    selected = {"action": action_key, "state": state_key}
    mappings = {
        label: build_dimension_mapping(key, dataset.features[key])
        for label, key in selected.items()
        if key is not None
    }
    print_schema(dataset, selected, mappings)

    feature_keys = [key for key in (action_key, state_key) if key is not None]
    sequences, frame_counts = load_episode_sequences(dataset, feature_keys)
    analyses = {
        label: analyze_episode_sequences(key, sequences[key], mappings[label], args.oscillation_delta_threshold)
        for label, key in selected.items()
        if key is not None
    }
    tasks = episode_tasks(dataset)
    rows = build_episode_rows(frame_counts, tasks, analyses)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "episode_metrics.csv", rows)
    for label, analysis in analyses.items():
        _write_csv(args.output_dir / f"joint_metrics_{label}.csv", analysis.joint_rows)
        _write_csv(args.output_dir / f"gripper_metrics_{label}.csv", analysis.gripper_rows)

    print_episode_metrics(rows)
    for label in ("action", "state"):
        if label in analyses:
            print_feature_analysis(label, analyses[label])
    print_worst_rankings(rows, args.worst_top_n)
    print_comparison(analyses, load_reference(args))

    plot_paths = plot_worst_episodes(
        args.output_dir,
        rows,
        analyses["action"],
        analyses.get("state"),
        fps,
        min(args.num_worst_plots, len(rows)),
    )
    print(f"\nOutputs written to {args.output_dir.resolve()}")
    for path in (
        args.output_dir / "episode_metrics.csv",
        *[args.output_dir / f"joint_metrics_{label}.csv" for label in analyses],
        *[args.output_dir / f"gripper_metrics_{label}.csv" for label in analyses],
        *plot_paths,
    ):
        print(f"  {path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
