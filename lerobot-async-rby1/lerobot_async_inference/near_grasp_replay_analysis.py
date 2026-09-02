"""Robot-free replay analysis for consecutive near-grasp policy captures."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from .frozen_noise_analysis import execute_condition, nested_checksum


PERCENTILES = (0.50, 0.90, 0.95, 0.99)
PERCENTILE_NAMES = ("median", "p90", "p95", "p99")


def _finite_median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(torch.tensor(finite, dtype=torch.float64).median()) if finite else math.nan


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _distribution(values: torch.Tensor, prefix: str) -> dict[str, float]:
    flat = values.detach().to(torch.float64).flatten()
    if flat.numel() == 0:
        return {f"{prefix}_{name}": math.nan for name in (*PERCENTILE_NAMES, "max")}
    quantiles = torch.quantile(flat, torch.tensor(PERCENTILES, dtype=torch.float64))
    result = {
        f"{prefix}_{name}": float(value)
        for name, value in zip(PERCENTILE_NAMES, quantiles, strict=True)
    }
    result[f"{prefix}_max"] = float(flat.max())
    return result


def roughness_metrics(
    chunk: torch.Tensor, left_indices: list[int], threshold: float = 0.005
) -> dict[str, float]:
    """Delta, second-delta, and thresholded sign-reversal metrics."""
    if chunk.ndim != 2:
        raise ValueError(f"Expected chunk shape (T, A), got {tuple(chunk.shape)}")
    left = chunk[:, left_indices].to(torch.float64)
    delta = left[1:] - left[:-1]
    second_delta = delta[1:] - delta[:-1]
    previous = delta[:-1]
    current = delta[1:]
    eligible = (previous.abs() > threshold) & (current.abs() > threshold)
    flips = eligible & ((previous * current) < 0)
    result = {}
    result.update(_distribution(delta.abs(), "dq"))
    result.update(_distribution(second_delta.abs(), "d2q"))
    result["oscillation"] = (
        float(flips.sum() / eligible.sum()) if int(eligible.sum()) else math.nan
    )
    result["oscillation_flips"] = int(flips.sum())
    result["oscillation_eligible"] = int(eligible.sum())
    return result


def joint_roughness_rows(
    chunk: torch.Tensor,
    left_indices: list[int],
    threshold: float,
    **labels: Any,
) -> list[dict[str, Any]]:
    return [
        {
            **labels,
            "joint": joint,
            "action_index": index,
            **roughness_metrics(chunk, [index], threshold),
        }
        for joint, index in enumerate(left_indices)
    ]


def estimate_frame_shift(
    previous_metadata: dict[str, Any],
    current_metadata: dict[str, Any],
    fps: float,
) -> tuple[int | None, float | None, str, str]:
    """Estimate the new observation's horizon offset on the old chunk grid."""
    for key in ("policy_timestamp", "wall_time"):
        previous = previous_metadata.get(key)
        current = current_metadata.get(key)
        if isinstance(previous, (int, float)) and isinstance(current, (int, float)):
            delta_s = float(current) - float(previous)
            if delta_s < 0:
                continue
            shift = int(round(delta_s * fps))
            residual = abs(delta_s - shift / fps)
            confidence = "high" if residual <= 0.25 / fps else "medium"
            if key == "wall_time":
                confidence = "medium" if confidence == "high" else "low"
            return shift, delta_s, key, confidence
    return None, None, "unavailable", "none"


def align_shifted_chunks(
    previous: torch.Tensor, current: torch.Tensor, shift: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if previous.ndim != 2 or current.ndim != 2:
        raise ValueError("Alignment expects two (T, A) chunks")
    if shift < 0:
        raise ValueError(f"Frame shift must be non-negative, got {shift}")
    overlap = min(previous.shape[0] - shift, current.shape[0])
    if overlap <= 0:
        return previous[:0], current[:0]
    return previous[shift : shift + overlap], current[:overlap]


def disagreement_metrics(
    previous: torch.Tensor,
    current: torch.Tensor,
    left_indices: list[int],
    direction_threshold: float = 0.005,
) -> dict[str, float]:
    if previous.shape != current.shape:
        raise ValueError(f"Aligned shapes differ: {tuple(previous.shape)} vs {tuple(current.shape)}")
    if previous.shape[0] == 0:
        return {
            "rmse": math.nan,
            "mean_abs_diff": math.nan,
            "p95_abs_diff": math.nan,
            "max_abs_diff": math.nan,
            "direction_mismatch": math.nan,
            "direction_eligible": 0,
            "delta_cosine_similarity": math.nan,
        }
    old = previous[:, left_indices].to(torch.float64)
    new = current[:, left_indices].to(torch.float64)
    difference = old - new
    old_delta = old[1:] - old[:-1]
    new_delta = new[1:] - new[:-1]
    eligible = (old_delta.abs() > direction_threshold) & (
        new_delta.abs() > direction_threshold
    )
    mismatch = eligible & ((old_delta * new_delta) < 0)
    old_flat = old_delta.flatten()
    new_flat = new_delta.flatten()
    denominator = float(torch.linalg.vector_norm(old_flat) * torch.linalg.vector_norm(new_flat))
    cosine = float(torch.dot(old_flat, new_flat) / denominator) if denominator > 0 else math.nan
    return {
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "mean_abs_diff": float(difference.abs().mean()),
        "p95_abs_diff": float(torch.quantile(difference.abs().flatten(), 0.95)),
        "max_abs_diff": float(difference.abs().max()),
        "direction_mismatch": (
            float(mismatch.sum() / eligible.sum()) if int(eligible.sum()) else math.nan
        ),
        "direction_eligible": int(eligible.sum()),
        "delta_cosine_similarity": cosine,
    }


def observation_change(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    previous_state = previous.get("observation.state")
    current_state = current.get("observation.state")
    state_norm = math.nan
    if isinstance(previous_state, torch.Tensor) and isinstance(current_state, torch.Tensor):
        if previous_state.shape == current_state.shape:
            state_norm = float(
                torch.linalg.vector_norm(
                    current_state.to(torch.float64) - previous_state.to(torch.float64)
                )
            )

    image_differences: list[torch.Tensor] = []
    for key in sorted(set(previous) & set(current)):
        if not str(key).startswith("observation.images."):
            continue
        old_image, new_image = previous[key], current[key]
        if (
            isinstance(old_image, torch.Tensor)
            and isinstance(new_image, torch.Tensor)
            and old_image.shape == new_image.shape
        ):
            image_differences.append(
                (new_image.to(torch.float32) - old_image.to(torch.float32)).abs().flatten()
            )
    if image_differences:
        image_diff = torch.cat(image_differences)
        image_mean = float(image_diff.mean())
        image_p95 = float(torch.quantile(image_diff, 0.95))
    else:
        image_mean = image_p95 = math.nan
    return {
        "state_change": state_norm,
        "image_mean_abs_change": image_mean,
        "image_p95_abs_change": image_p95,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def _load_metadata(directory: Path) -> list[dict[str, Any]]:
    path = directory / "capture_metadata.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Capture metadata not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid metadata JSON at line {line_number}: {error}") from error
    return sorted(rows, key=lambda row: int(row["capture_id"]))


def _load_chunk(path: Path) -> torch.Tensor:
    chunk = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(chunk, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(chunk).__name__}")
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk.squeeze(0)
    if chunk.ndim != 2:
        raise ValueError(f"Expected (T, A) in {path}, got {tuple(chunk.shape)}")
    return chunk.detach().cpu()


def _pairwise_random_metrics(runs: torch.Tensor, indices: list[int]) -> dict[str, float]:
    rmses, maxima = [], []
    for first, second in itertools.combinations(range(runs.shape[0]), 2):
        difference = runs[first, :, indices] - runs[second, :, indices]
        rmses.append(float(torch.sqrt(torch.mean(difference.square()))))
        maxima.append(float(difference.abs().max()))
    return {
        "across_run_std": float(runs[:, :, indices].std(dim=0, unbiased=False).mean()),
        "run_to_run_rmse_mean": sum(rmses) / len(rmses) if rmses else math.nan,
        "run_to_run_rmse_max": max(rmses, default=math.nan),
        "run_to_run_max_abs_diff_mean": sum(maxima) / len(maxima) if maxima else math.nan,
        "run_to_run_max_abs_diff_max": max(maxima, default=math.nan),
    }


def _live_distribution_comparison(
    live: torch.Tensor, fixed: torch.Tensor, random_runs: torch.Tensor, indices: list[int]
) -> dict[str, float]:
    live_left = live[:, indices]
    fixed_left = fixed[:, indices]
    random_left = random_runs[:, :, indices]
    live_fixed = live_left - fixed_left
    random_mean = random_left.mean(dim=0)
    random_std = random_left.std(dim=0, unbiased=False)
    safe_std = random_std.clamp_min(1e-8)
    z = ((live_left - random_mean) / safe_std).abs()
    lower = torch.quantile(random_left, 0.025, dim=0)
    upper = torch.quantile(random_left, 0.975, dim=0)
    return {
        "live_fixed_rmse": float(torch.sqrt(torch.mean(live_fixed.square()))),
        "live_fixed_max_abs_diff": float(live_fixed.abs().max()),
        "live_random_mean_abs_z": float(z.mean()),
        "live_fraction_inside_random_95pct": float(
            ((live_left >= lower) & (live_left <= upper)).to(torch.float32).mean()
        ),
    }


def _extract_left_action(action: Any, indices: list[int]) -> list[float] | None:
    if isinstance(action, list) and len(action) > max(indices):
        return [float(action[index]) for index in indices]
    if not isinstance(action, dict):
        return None
    matching = [(key, value) for key, value in action.items() if "left_arm" in key]
    if len(matching) < len(indices):
        return None

    def key_number(item: tuple[str, Any]) -> tuple[int, str]:
        numbers = re.findall(r"\d+", item[0])
        return (int(numbers[-1]) if numbers else 10_000, item[0])

    matching.sort(key=key_number)
    return [float(value) for _, value in matching[: len(indices)]]


def _analyze_final_actions(
    path: Path,
    captures: list[dict[str, Any]],
    fixed_chunks: dict[int, torch.Tensor],
    left_indices: list[int],
    fps: float,
    threshold: float,
) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    rows = []
    for metadata in captures:
        capture_id = int(metadata["capture_id"])
        start = metadata.get("policy_timestamp")
        if not isinstance(start, (int, float)):
            rows.append({"capture_id": capture_id, "alignment_confidence": "none"})
            continue
        horizon = fixed_chunks[capture_id].shape[0]
        end = float(start) + (horizon - 1) / fps
        selected: list[tuple[float, list[float]]] = []
        for record in records:
            timestamp = record.get("policy_timestamp")
            values = _extract_left_action(record.get("action"), left_indices)
            if isinstance(timestamp, (int, float)) and values is not None:
                if float(start) - 0.5 / fps <= float(timestamp) <= end + 0.5 / fps:
                    selected.append((float(timestamp), values))
        selected.sort()
        executed = torch.tensor([values for _, values in selected], dtype=torch.float32)
        coverage = len(selected) / horizon
        residuals = [
            abs((timestamp - float(start)) * fps - round((timestamp - float(start)) * fps))
            for timestamp, _ in selected
        ]
        median_residual = _finite_median(residuals)
        confidence = (
            "high"
            if coverage >= 0.8 and median_residual <= 0.25
            else "medium"
            if coverage >= 0.5
            else "low"
            if selected
            else "none"
        )
        row: dict[str, Any] = {
            "capture_id": capture_id,
            "alignment_confidence": confidence,
            "executed_action_count": len(selected),
            "expected_action_count": horizon,
            "coverage": coverage,
            "median_grid_residual_frames": median_residual,
        }
        if len(selected) >= 3:
            executed_metrics = roughness_metrics(
                executed, list(range(len(left_indices))), threshold
            )
            row.update({f"final_{key}": value for key, value in executed_metrics.items()})
        rows.append(row)
    return rows


def _load_demo_reference(path: Path | None, task: str | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    desired_group = "BOWL" if task and "bowl" in task.lower() else "ALL"
    selected = [row for row in rows if row.get("group") == desired_group]
    if not selected:
        selected = [row for row in rows if row.get("group") == "ALL"]
    values = {row["metric"]: float(row["pregrasp"]) for row in selected}
    return {
        "source": str(path),
        "group": desired_group if selected else None,
        "dq_p95": values.get("action_delta_p95"),
        "d2q_p95": values.get("action_delta2_p95"),
        "oscillation": values.get("action_oscillation_pooled"),
    }


def _interpret(
    capture_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    fixed_d2 = _finite_median([row["fixed_d2q_p95"] for row in capture_rows])
    fixed_osc = _finite_median([row["fixed_osc"] for row in capture_rows])
    live_d2 = _finite_median([row["live_d2q_p95"] for row in capture_rows])
    live_osc = _finite_median([row["live_osc"] for row in capture_rows])
    random_std = _finite_median([row["random_across_run_std"] for row in capture_rows])
    pair_rmse = max((row["rmse"] for row in pair_rows if math.isfinite(row["rmse"])), default=math.nan)
    pair_mismatch = max(
        (row["direction_mismatch"] for row in pair_rows if math.isfinite(row["direction_mismatch"])),
        default=math.nan,
    )
    small_observation_large_disagreement = any(
        math.isfinite(row["state_change"])
        and row["state_change"] <= args.small_state_change_threshold
        and (
            (math.isfinite(row["rmse"]) and row["rmse"] >= args.replanning_rmse_threshold)
            or (
                math.isfinite(row["direction_mismatch"])
                and row["direction_mismatch"] >= args.direction_mismatch_threshold
            )
        )
        for row in pair_rows
    )
    findings = []
    if fixed_d2 >= args.rough_d2q_threshold or fixed_osc >= args.rough_oscillation_threshold:
        findings.append(
            {
                "case": 1,
                "name": "MODEL WITHIN-CHUNK ROUGHNESS",
                "triggered": True,
                "text": "The model generates temporally rough trajectories even for a frozen near-grasp observation. The dominant problem is model/training-side within-chunk roughness.",
            }
        )
    if random_std >= args.sampling_std_threshold:
        findings.append(
            {
                "case": 2,
                "name": "SAMPLING STOCHASTICITY",
                "triggered": True,
                "text": "Flow sampling stochasticity materially changes near-grasp trajectories.",
            }
        )
    if small_observation_large_disagreement:
        findings.append(
            {
                "case": 3,
                "name": "OBSERVATION/REPLANNING SENSITIVITY",
                "triggered": True,
                "text": "Small closed-loop observation changes cause large replanning changes. Near-grasp hunting is likely driven by policy sensitivity / replanning.",
            }
        )
    final_d2 = _finite_median(
        [row.get("final_d2q_p95", math.nan) for row in final_rows if row.get("alignment_confidence") == "high"]
    )
    fixed_smooth = fixed_d2 < args.rough_d2q_threshold and (
        not math.isfinite(fixed_osc) or fixed_osc < args.rough_oscillation_threshold
    )
    live_smooth = live_d2 < args.rough_d2q_threshold and (
        not math.isfinite(live_osc) or live_osc < args.rough_oscillation_threshold
    )
    execution_side_rough = math.isfinite(final_d2) and final_d2 >= (
        args.execution_roughness_ratio * max(fixed_d2, live_d2, 1e-9)
    )
    if fixed_smooth and live_smooth and execution_side_rough:
        findings.append(
            {
                "case": 4,
                "name": "EXECUTION/MERGE SIDE",
                "triggered": True,
                "text": "Roughness is introduced after policy generation, likely in chunk aggregation/client/execution.",
            }
        )
    if not findings:
        findings.append(
            {
                "case": 0,
                "name": "NO HEURISTIC TRIGGER",
                "triggered": False,
                "text": "No configured root-cause threshold was crossed; inspect the plots and continuous metrics.",
            }
        )
    return [
        {
            **finding,
            "evidence": {
                "median_fixed_d2q_p95": fixed_d2,
                "median_fixed_oscillation": fixed_osc,
                "median_live_d2q_p95": live_d2,
                "median_live_oscillation": live_osc,
                "median_random_across_run_std": random_std,
                "max_aligned_pair_rmse": pair_rmse,
                "max_aligned_direction_mismatch": pair_mismatch,
                "median_high_confidence_final_d2q_p95": final_d2,
            },
        }
        for finding in findings
    ]


def _plots(
    output_dir: Path,
    captures: list[dict[str, Any]],
    capture_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    fixed_chunks: dict[int, torch.Tensor],
    left_indices: list[int],
    fps: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = [int(row["capture_id"]) for row in captures]
    x = torch.arange(len(ids), dtype=torch.float32).numpy()
    width = 0.26
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for axis, metric, title in (
        (axes[0], "dq_p95", "Left-arm |Delta q| p95"),
        (axes[1], "d2q_p95", "Left-arm |Delta2 q| p95"),
    ):
        axis.bar(x - width, [row[f"live_{metric}"] for row in capture_rows], width, label="live")
        axis.bar(x, [row[f"fixed_{metric}"] for row in capture_rows], width, label="fixed")
        axis.bar(x + width, [row[f"random_mean_{metric}"] for row in capture_rows], width, label="random mean")
        axis.set_ylabel("rad")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[1].set_xticks(x, [f"{capture_id:03d}" for capture_id in ids])
    axes[1].set_xlabel("capture_id")
    figure.tight_layout()
    figure.savefig(output_dir / "near_grasp_chunk_roughness.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - width / 2, [row["random_across_run_std"] for row in capture_rows], width, label="random across-run std")
    axis.bar(x + width / 2, [row["live_fixed_rmse"] for row in capture_rows], width, label="live vs fixed RMSE")
    axis.set_xticks(x, [f"{capture_id:03d}" for capture_id in ids])
    axis.set_xlabel("capture_id")
    axis.set_ylabel("rad")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "random_vs_fixed.png", dpi=160)
    plt.close(figure)

    first_timestamp = float(captures[0].get("policy_timestamp", 0.0))
    figure, axes = plt.subplots(7, 1, figsize=(13, 15), sharex=True)
    for metadata in captures:
        capture_id = int(metadata["capture_id"])
        timestamp = float(metadata.get("policy_timestamp", first_timestamp))
        start_frame = round((timestamp - first_timestamp) * fps)
        chunk = fixed_chunks[capture_id]
        frames = torch.arange(chunk.shape[0]).numpy() + start_frame
        for joint, action_index in enumerate(left_indices):
            axes[joint].plot(frames, chunk[:, action_index].numpy(), label=f"capture {capture_id:03d}")
            axes[joint].set_ylabel(f"L{joint} rad")
            axes[joint].grid(alpha=0.2)
    axes[0].legend(ncol=min(5, len(captures)), fontsize=8)
    axes[-1].set_xlabel(f"frame on shared {fps:g} Hz policy-time grid")
    figure.suptitle("Consecutive observations replayed with one shared fixed noise")
    figure.tight_layout()
    figure.savefig(output_dir / "consecutive_fixed_noise_chunks.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for row in pair_rows:
        axis.scatter(row["state_change"], row["rmse"])
        axis.annotate(f"{row['previous_capture_id']:03d}->{row['capture_id']:03d}", (row["state_change"], row["rmse"]))
    axis.set_xlabel("policy-ready state L2 change")
    axis.set_ylabel("overlap-aligned fixed-chunk RMSE (rad)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "chunk_disagreement_vs_observation_change.png", dpi=160)
    plt.close(figure)


def _json_clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    return value


def _capture_overhead_summary(metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    critical_values = []
    background_values = []
    component_names = (
        "batch_cpu_clone_ms",
        "raw_preserve_clone_ms",
        "chunk_cpu_clone_ms",
        "enqueue_ms",
    )
    for row in metadata_rows:
        components = [row.get(name) for name in component_names]
        if all(isinstance(value, (int, float)) for value in components):
            critical_values.append(sum(float(value) for value in components))
        background = row.get("background_save_and_checksum_ms")
        if isinstance(background, (int, float)):
            background_values.append(float(background))
    return {
        "critical_path_components": list(component_names),
        "critical_path_ms_median": _finite_median(critical_values),
        "critical_path_ms_max": max(critical_values, default=math.nan),
        "background_save_checksum_ms_median": _finite_median(background_values),
        "background_save_checksum_ms_max": max(background_values, default=math.nan),
        "captures_measured": len(critical_values),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--capture-dir", "--capture_dir", dest="capture_dir", required=True)
    parser.add_argument("--last-n", "--last_n", dest="last_n", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="outputs/near_grasp_replay_analysis")
    parser.add_argument("--random-runs-per-capture", type=int, default=20)
    parser.add_argument("--fixed-runs-per-capture", type=int, default=3)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--left-arm-indices", default="7:14")
    parser.add_argument("--oscillation-threshold", type=float, default=0.005)
    parser.add_argument("--final-actions")
    parser.add_argument("--demonstration-summary")
    parser.add_argument("--determinism-atol", type=float, default=1e-6)
    parser.add_argument("--rough-d2q-threshold", type=float, default=0.02)
    parser.add_argument("--rough-oscillation-threshold", type=float, default=0.10)
    parser.add_argument("--sampling-std-threshold", type=float, default=0.01)
    parser.add_argument("--small-state-change-threshold", type=float, default=0.25)
    parser.add_argument("--replanning-rmse-threshold", type=float, default=0.05)
    parser.add_argument("--direction-mismatch-threshold", type=float, default=0.50)
    parser.add_argument("--execution-roughness-ratio", type=float, default=2.0)
    return parser


def _parse_slice(spec: str, dimension: int) -> list[int]:
    if ":" in spec:
        start_text, stop_text = spec.split(":", maxsplit=1)
        start = int(start_text) if start_text else 0
        stop = int(stop_text) if stop_text else dimension
        indices = list(range(start, stop))
    else:
        indices = [int(value) for value in spec.split(",")]
    if not indices or min(indices) < 0 or max(indices) >= dimension:
        raise ValueError(f"Invalid indices {indices} for action dimension {dimension}")
    return indices


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.last_n <= 0 or args.random_runs_per_capture < 2 or args.fixed_runs_per_capture < 3:
        raise ValueError("last-n > 0, random runs >= 2, and fixed runs >= 3 are required")
    capture_dir = Path(args.capture_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = _load_metadata(capture_dir)
    selected = metadata_rows[-args.last_n :]
    if not selected:
        raise ValueError(f"No captures found in {capture_dir}")
    captured_num_steps = {
        int(row["num_steps"])
        for row in selected
        if isinstance(row.get("num_steps"), (int, float))
    }
    if captured_num_steps and captured_num_steps != {args.num_steps}:
        raise ValueError(
            f"Captured num_steps={sorted(captured_num_steps)} does not match replay "
            f"--num-steps={args.num_steps}"
        )
    captured_fps = {
        float(row["fps"])
        for row in selected
        if isinstance(row.get("fps"), (int, float))
    }
    if captured_fps and any(abs(value - args.fps) > 1e-9 for value in captured_fps):
        raise ValueError(f"Captured fps={sorted(captured_fps)} does not match replay --fps={args.fps}")

    batches, live_raw_chunks, live_chunks = {}, {}, {}
    for metadata in selected:
        capture_id = int(metadata["capture_id"])
        suffix = f"{capture_id:03d}"
        batch_path = capture_dir / f"capture_{suffix}.pt"
        batch = torch.load(batch_path, map_location="cpu", weights_only=False)
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch in {batch_path}")
        if metadata.get("batch_checksum") and nested_checksum(batch) != metadata["batch_checksum"]:
            raise RuntimeError(f"Checksum mismatch for {batch_path}")
        batches[capture_id] = batch
        live_raw_chunks[capture_id] = _load_chunk(capture_dir / f"raw_chunk_{suffix}.pt")
        live_chunks[capture_id] = _load_chunk(capture_dir / f"robot_chunk_{suffix}.pt")

    action_dimension = next(iter(live_chunks.values())).shape[1]
    left_indices = _parse_slice(args.left_arm_indices, action_dimension)
    if len(left_indices) != 7:
        raise ValueError(f"Expected seven left-arm joints, got {left_indices}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    from lerobot.policies import get_policy_class, make_pre_post_processors

    checkpoint = Path(args.checkpoint).expanduser()
    captured_checkpoints = {
        str(row["checkpoint"])
        for row in selected
        if isinstance(row.get("checkpoint"), str)
    }
    if captured_checkpoints and str(checkpoint) not in captured_checkpoints:
        print(
            "WARNING: replay checkpoint path differs from capture metadata: "
            f"captured={sorted(captured_checkpoints)} replay={checkpoint}"
        )
    policy = get_policy_class("smolvla").from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()
    policy.config.num_steps = args.num_steps
    if getattr(policy, "model", None) is not None and policy.model.config is not policy.config:
        policy.model.config.num_steps = args.num_steps
    if hasattr(policy.model, "_rtc_enabled") and policy.model._rtc_enabled():
        raise RuntimeError("Near-grasp replay requires RTC disabled, matching the live rollout")
    _, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        postprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    noise_shape = (1, int(policy.config.chunk_size), int(policy.config.max_action_dim))
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.noise_seed)
        with torch.inference_mode():
            fixed_noise = policy.model.sample_noise(noise_shape, device).detach().clone()
    fixed_noise_checksum = nested_checksum(fixed_noise)
    torch.save(fixed_noise.detach().cpu(), output_dir / "fixed_noise.pt")

    fixed_chunks: dict[int, torch.Tensor] = {}
    random_chunks: dict[int, torch.Tensor] = {}
    fixed_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    print(f"Selected captures: {[int(row['capture_id']) for row in selected]}")
    print(f"Shared fixed noise: shape={noise_shape} seed={args.noise_seed} checksum={fixed_noise_checksum}")
    for metadata in selected:
        capture_id = int(metadata["capture_id"])
        batch = batches[capture_id]
        horizon = live_chunks[capture_id].shape[0]
        fixed = execute_condition(
            policy=policy,
            frozen_batch=batch,
            postprocessor=postprocessor,
            device=device,
            num_runs=args.fixed_runs_per_capture,
            fixed_noise=fixed_noise,
        )
        random = execute_condition(
            policy=policy,
            frozen_batch=batch,
            postprocessor=postprocessor,
            device=device,
            num_runs=args.random_runs_per_capture,
            fixed_noise=None,
        )
        if fixed.noise_checksum != fixed_noise_checksum:
            raise RuntimeError(f"Fixed-noise fairness checksum failed for capture {capture_id}")
        fixed_robot = fixed.robot_actions[:, :horizon, :action_dimension]
        random_robot = random.robot_actions[:, :horizon, :action_dimension]
        fixed_chunks[capture_id] = fixed_robot[0]
        random_chunks[capture_id] = random_robot
        fixed_differences = [
            float((fixed_robot[run] - fixed_robot[0]).abs().max())
            for run in range(1, fixed_robot.shape[0])
        ]
        if any(value > args.determinism_atol for value in fixed_differences):
            print(f"WARNING capture {capture_id}: fixed replay is not deterministic: {fixed_differences}")

        live_metric = roughness_metrics(live_chunks[capture_id], left_indices, args.oscillation_threshold)
        live_raw_metric = roughness_metrics(
            live_raw_chunks[capture_id], left_indices, args.oscillation_threshold
        )
        for run in range(fixed_robot.shape[0]):
            metric = roughness_metrics(fixed_robot[run], left_indices, args.oscillation_threshold)
            fixed_rows.append({"capture_id": capture_id, "run": run, "max_abs_diff_from_run0": 0.0 if run == 0 else fixed_differences[run - 1], **metric})
            joint_rows.extend(joint_roughness_rows(fixed_robot[run], left_indices, args.oscillation_threshold, capture_id=capture_id, condition="fixed", run=run))
        random_metrics = []
        for run in range(random_robot.shape[0]):
            metric = roughness_metrics(random_robot[run], left_indices, args.oscillation_threshold)
            random_metrics.append(metric)
            random_rows.append({"capture_id": capture_id, "run": run, **metric})
            joint_rows.extend(joint_roughness_rows(random_robot[run], left_indices, args.oscillation_threshold, capture_id=capture_id, condition="random", run=run))
        joint_rows.extend(joint_roughness_rows(live_chunks[capture_id], left_indices, args.oscillation_threshold, capture_id=capture_id, condition="live", run=0))
        joint_rows.extend(joint_roughness_rows(live_raw_chunks[capture_id], left_indices, args.oscillation_threshold, capture_id=capture_id, condition="live_raw_policy_units", run=0))
        fixed_metric = roughness_metrics(fixed_robot[0], left_indices, args.oscillation_threshold)
        random_variability = _pairwise_random_metrics(random_robot, left_indices)
        comparison = _live_distribution_comparison(live_chunks[capture_id], fixed_robot[0], random_robot, left_indices)
        comparison_rows.append({"capture_id": capture_id, **random_variability, **comparison})
        capture_rows.append(
            {
                "capture_id": capture_id,
                "live_dq_p95": live_metric["dq_p95"],
                "live_d2q_p95": live_metric["d2q_p95"],
                "live_osc": live_metric["oscillation"],
                "live_raw_dq_p95": live_raw_metric["dq_p95"],
                "live_raw_d2q_p95": live_raw_metric["d2q_p95"],
                "live_raw_osc": live_raw_metric["oscillation"],
                "fixed_dq_p95": fixed_metric["dq_p95"],
                "fixed_d2q_p95": fixed_metric["d2q_p95"],
                "fixed_osc": fixed_metric["oscillation"],
                "random_mean_dq_p95": _finite_mean([metric["dq_p95"] for metric in random_metrics]),
                "random_mean_d2q_p95": _finite_mean([metric["d2q_p95"] for metric in random_metrics]),
                "random_mean_osc": _finite_mean([metric["oscillation"] for metric in random_metrics]),
                "random_across_run_std": random_variability["across_run_std"],
                "fixed_repeat_max_abs_diff": max(fixed_differences, default=0.0),
                **comparison,
                "state_change_from_prev": math.nan,
                "aligned_fixed_chunk_rmse_from_prev": math.nan,
                "aligned_direction_mismatch_from_prev": math.nan,
            }
        )
        print(
            f"capture {capture_id:03d}: live={tuple(live_chunks[capture_id].shape)} "
            f"fixed_d2_p95={fixed_metric['d2q_p95']:.6f} "
            f"random_std={random_variability['across_run_std']:.6f} "
            f"fixed_repeat_max={max(fixed_differences, default=0.0):.3g}"
        )

    pair_rows = []
    row_by_id = {int(row["capture_id"]): row for row in capture_rows}
    for previous_metadata, metadata in zip(selected, selected[1:]):
        previous_id, capture_id = int(previous_metadata["capture_id"]), int(metadata["capture_id"])
        shift, delta_s, source, confidence = estimate_frame_shift(previous_metadata, metadata, args.fps)
        change = observation_change(batches[previous_id], batches[capture_id])
        row: dict[str, Any] = {
            "previous_capture_id": previous_id,
            "capture_id": capture_id,
            "timestamp_delta_s": delta_s,
            "frame_shift": shift,
            "alignment_source": source,
            "alignment_confidence": confidence,
            **change,
        }
        if shift is None:
            disagreement = disagreement_metrics(fixed_chunks[previous_id][:0], fixed_chunks[capture_id][:0], left_indices)
            row["overlap_frames"] = 0
        else:
            old_aligned, new_aligned = align_shifted_chunks(fixed_chunks[previous_id], fixed_chunks[capture_id], shift)
            disagreement = disagreement_metrics(old_aligned, new_aligned, left_indices, args.oscillation_threshold)
            row["overlap_frames"] = old_aligned.shape[0]
            if not old_aligned.shape[0]:
                row["alignment_confidence"] = "none"
        row.update(disagreement)
        pair_rows.append(row)
        summary_row = row_by_id[capture_id]
        summary_row["state_change_from_prev"] = change["state_change"]
        summary_row["aligned_fixed_chunk_rmse_from_prev"] = disagreement["rmse"]
        summary_row["aligned_direction_mismatch_from_prev"] = disagreement["direction_mismatch"]

    final_rows = []
    if args.final_actions:
        final_rows = _analyze_final_actions(Path(args.final_actions).expanduser(), selected, fixed_chunks, left_indices, args.fps, args.oscillation_threshold)
        for row in final_rows:
            policy_row = row_by_id[int(row["capture_id"])]
            row.update(
                policy_live_raw_dq_p95=policy_row["live_raw_dq_p95"],
                policy_live_raw_d2q_p95=policy_row["live_raw_d2q_p95"],
                policy_live_robot_dq_p95=policy_row["live_dq_p95"],
                policy_live_robot_d2q_p95=policy_row["live_d2q_p95"],
                policy_fixed_robot_dq_p95=policy_row["fixed_dq_p95"],
                policy_fixed_robot_d2q_p95=policy_row["fixed_d2q_p95"],
            )

    demo_path = Path(args.demonstration_summary).expanduser() if args.demonstration_summary else None
    if demo_path is None:
        repository_root = Path(__file__).resolve().parents[2]
        candidate = repository_root / "outputs/rby1_pregrasp_temporal_analysis/pregrasp_summary.csv"
        demo_path = candidate if candidate.is_file() else None
    task = next((row.get("task") for row in reversed(selected) if isinstance(row.get("task"), str)), None)
    demo_reference = _load_demo_reference(demo_path, task)
    interpretations = _interpret(capture_rows, pair_rows, final_rows, args)
    capture_overhead = _capture_overhead_summary(selected)

    _write_csv(output_dir / "capture_summary.csv", capture_rows)
    _write_csv(output_dir / "fixed_replay_metrics.csv", fixed_rows)
    _write_csv(output_dir / "random_replay_metrics.csv", random_rows)
    _write_csv(output_dir / "consecutive_chunk_disagreement.csv", pair_rows)
    _write_csv(output_dir / "joint_metrics.csv", joint_rows)
    _write_csv(output_dir / "live_replay_comparison.csv", comparison_rows)
    if final_rows:
        _write_csv(output_dir / "final_action_comparison.csv", final_rows)
    demo_comparison_rows = []
    if demo_reference:
        demo_comparison_rows = [
            {
                "source": "demonstration_pregrasp",
                "dq_p95": demo_reference["dq_p95"],
                "d2q_p95": demo_reference["d2q_p95"],
                "oscillation": demo_reference["oscillation"],
            },
            {
                "source": "model_near_grasp_fixed_noise_median",
                "dq_p95": _finite_median([row["fixed_dq_p95"] for row in capture_rows]),
                "d2q_p95": _finite_median([row["fixed_d2q_p95"] for row in capture_rows]),
                "oscillation": _finite_median([row["fixed_osc"] for row in capture_rows]),
            },
        ]
        _write_csv(output_dir / "demonstration_comparison.csv", demo_comparison_rows)
    torch.save(fixed_chunks, output_dir / "fixed_replay_chunks.pt")
    torch.save(random_chunks, output_dir / "random_replay_chunks.pt")
    _plots(output_dir, selected, capture_rows, pair_rows, fixed_chunks, left_indices, args.fps)

    summary = {
        "checkpoint": str(checkpoint),
        "capture_dir": str(capture_dir),
        "selected_capture_ids": [int(row["capture_id"]) for row in selected],
        "device": str(device),
        "fps": args.fps,
        "num_steps": args.num_steps,
        "action_dimension": action_dimension,
        "left_arm_indices": left_indices,
        "fixed_noise_shape": list(noise_shape),
        "fixed_noise_seed": args.noise_seed,
        "fixed_noise_checksum": fixed_noise_checksum,
        "fixed_noise_reused_for_every_capture": True,
        "fixed_runs_per_capture": args.fixed_runs_per_capture,
        "fixed_noise_deterministic_all": all(
            row["fixed_repeat_max_abs_diff"] <= args.determinism_atol
            for row in capture_rows
        ),
        "determinism_atol": args.determinism_atol,
        "random_runs_per_capture": args.random_runs_per_capture,
        "oscillation_threshold_rad": args.oscillation_threshold,
        "capture_summary": capture_rows,
        "consecutive_disagreement": pair_rows,
        "final_action_alignment": final_rows,
        "demonstration_pregrasp_reference": demo_reference,
        "demonstration_comparison": demo_comparison_rows,
        "interpretations": interpretations,
        "live_capture_overhead": capture_overhead,
        "heuristic_thresholds": {
            key: getattr(args, key)
            for key in (
                "rough_d2q_threshold",
                "rough_oscillation_threshold",
                "sampling_std_threshold",
                "small_state_change_threshold",
                "replanning_rmse_threshold",
                "direction_mismatch_threshold",
                "execution_roughness_ratio",
            )
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_clean(summary), stream, indent=2, sort_keys=True)

    print("\nCapture summary")
    for row in capture_rows:
        print(
            f"  {row['capture_id']:03d} live_d2={row['live_d2q_p95']:.6f} "
            f"fixed_d2={row['fixed_d2q_p95']:.6f} random_std={row['random_across_run_std']:.6f} "
            f"aligned_prev_rmse={row['aligned_fixed_chunk_rmse_from_prev']:.6f}"
        )
    if demo_reference:
        print("\nDemonstration comparison")
        print("  source                                  dq_p95     d2q_p95   oscillation")
        for row in demo_comparison_rows:
            print(
                f"  {row['source']:<38} {row['dq_p95']:>10.6f} "
                f"{row['d2q_p95']:>10.6f} {row['oscillation']:>13.6f}"
            )
    if capture_overhead["captures_measured"]:
        print(
            "Capture critical-path overhead: "
            f"median={capture_overhead['critical_path_ms_median']:.3f}ms "
            f"max={capture_overhead['critical_path_ms_max']:.3f}ms; "
            "disk/checksum work ran in the background"
        )
    print("Interpretation")
    for finding in interpretations:
        print(f"  CASE {finding['case']} {finding['name']}: {finding['text']}")
    print(f"Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
