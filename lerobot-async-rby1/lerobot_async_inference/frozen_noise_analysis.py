"""Offline diagnostics for frozen SmolVLA observations.

This module deliberately has no robot, camera, serial, or control-loop imports.
The server imports only :func:`clone_to_cpu` for its optional one-shot capture;
LeRobot policy imports are deferred until the standalone diagnostic is run.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


def clone_to_cpu(obj: Any) -> Any:
    """Recursively clone an object, moving every tensor to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone().cpu()
    if isinstance(obj, dict):
        return {key: clone_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clone_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        values = (clone_to_cpu(value) for value in obj)
        if hasattr(obj, "_fields"):
            return type(obj)(*values)
        return tuple(values)
    return copy.deepcopy(obj)


def clone_to_device(obj: Any, device: torch.device | str) -> Any:
    """Recursively make a fresh clone, moving tensors to ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone().to(device)
    if isinstance(obj, dict):
        return {key: clone_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clone_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        values = (clone_to_device(value, device) for value in obj)
        if hasattr(obj, "_fields"):
            return type(obj)(*values)
        return tuple(values)
    return copy.deepcopy(obj)


def nested_checksum(obj: Any) -> str:
    """Return a stable SHA-256 checksum including tensor values and metadata."""
    digest = hashlib.sha256()

    def update(value: Any, path: str) -> None:
        digest.update(path.encode())
        digest.update(type(value).__qualname__.encode())
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda item: repr(item)):
                digest.update(repr(key).encode())
                update(value[key], f"{path}.{key!r}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                update(item, f"{path}[{index}]")
        else:
            digest.update(repr(value).encode())

    update(obj, "root")
    return digest.hexdigest()


def iter_tensor_descriptions(obj: Any, path: str = "batch") -> Iterable[str]:
    if isinstance(obj, torch.Tensor):
        yield (
            f"{path}: shape={tuple(obj.shape)} dtype={obj.dtype} "
            f"device={obj.device}"
        )
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_tensor_descriptions(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from iter_tensor_descriptions(value, f"{path}[{index}]")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def postprocess_action_chunk(
    raw_actions: torch.Tensor, postprocessor: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Apply the deployment postprocessor exactly as PolicyServer does."""
    if raw_actions.ndim != 3:
        raise ValueError(f"Expected raw action shape (B, T, A), got {tuple(raw_actions.shape)}")
    processed = [postprocessor(raw_actions[:, step, :]) for step in range(raw_actions.shape[1])]
    if not all(isinstance(action, torch.Tensor) for action in processed):
        raise TypeError("SmolVLA postprocessor must return torch.Tensor actions")
    return torch.stack(processed, dim=1)


@dataclass
class TrialResults:
    raw_actions: torch.Tensor
    robot_actions: torch.Tensor
    latencies_s: list[float]


def execute_condition(
    *,
    policy: Any,
    frozen_batch: dict[str, Any],
    postprocessor: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    num_runs: int,
    fixed_noise: torch.Tensor | None,
) -> TrialResults:
    """Run RANDOM (``fixed_noise=None``) or a fixed external-noise condition."""
    raw_runs: list[torch.Tensor] = []
    robot_runs: list[torch.Tensor] = []
    latencies: list[float] = []
    frozen_checksum = nested_checksum(frozen_batch)
    noise_checksum = nested_checksum(fixed_noise) if fixed_noise is not None else None

    policy.eval()
    for _ in range(num_runs):
        policy.reset()
        batch_i = clone_to_device(frozen_batch, device)
        if nested_checksum(batch_i) != frozen_checksum:
            raise RuntimeError("Fresh trial batch does not match the frozen observation")
        noise_i = fixed_noise.detach().clone() if fixed_noise is not None else None

        _synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            raw = policy.predict_action_chunk(batch_i, noise=noise_i)
        _synchronize(device)
        latencies.append(time.perf_counter() - started)

        if raw.ndim != 3:
            raise ValueError(f"Policy returned action shape {tuple(raw.shape)}; expected (B, T, A)")
        if raw.shape[0] != 1:
            raise ValueError(f"Diagnostic currently expects batch size 1, got {raw.shape[0]}")
        # Preserve the model output before the deployment postprocessor sees a
        # view of it; a future processor step may legally operate in-place.
        raw_cpu = raw.squeeze(0).detach().clone().cpu()
        robot = postprocess_action_chunk(raw, postprocessor)
        raw_runs.append(raw_cpu)
        robot_runs.append(robot.squeeze(0).detach().clone().cpu())

        if fixed_noise is not None and nested_checksum(fixed_noise) != noise_checksum:
            raise RuntimeError("The shared fixed-noise tensor was mutated during a trial")

    if nested_checksum(frozen_batch) != frozen_checksum:
        raise RuntimeError("The source frozen batch was mutated during the experiment")
    return TrialResults(
        raw_actions=torch.stack(raw_runs),
        robot_actions=torch.stack(robot_runs),
        latencies_s=latencies,
    )


def _summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().to(torch.float64).flatten()
    if flat.numel() == 0:
        return {key: math.nan for key in ("mean", "median", "p95", "p99", "max")}
    return {
        "mean": flat.mean().item(),
        "median": flat.median().item(),
        "p95": torch.quantile(flat, 0.95).item(),
        "p99": torch.quantile(flat, 0.99).item(),
        "max": flat.max().item(),
    }


def _parse_indices(spec: str, action_dim: int) -> list[int]:
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_text, stop_text = part.split(":", maxsplit=1)
            start = int(start_text) if start_text else 0
            stop = int(stop_text) if stop_text else action_dim
            indices.extend(range(start, stop))
        else:
            indices.append(int(part))
    if not indices or len(set(indices)) != len(indices):
        raise ValueError(f"Index specification must be nonempty and unique: {spec!r}")
    if min(indices) < 0 or max(indices) >= action_dim:
        raise ValueError(f"Indices {indices} exceed actual action dimension {action_dim}")
    return indices


def _oscillation_scores(
    actions: torch.Tensor, indices: list[int], threshold: float
) -> tuple[float, list[tuple[int, float, int]]]:
    delta = actions[:, 1:, indices] - actions[:, :-1, indices]
    previous = delta[:, :-1, :]
    current = delta[:, 1:, :]
    eligible = (previous.abs() > threshold) & (current.abs() > threshold)
    flips = eligible & ((previous * current) < 0)
    eligible_per_joint = eligible.sum(dim=(0, 1))
    flips_per_joint = flips.sum(dim=(0, 1))
    rows: list[tuple[int, float, int]] = []
    for local_index, action_index in enumerate(indices):
        denominator = int(eligible_per_joint[local_index])
        ratio = (
            float(flips_per_joint[local_index]) / denominator
            if denominator
            else math.nan
        )
        rows.append((action_index, ratio, denominator))
    total_eligible = int(eligible.sum())
    group_ratio = float(flips.sum()) / total_eligible if total_eligible else math.nan
    return group_ratio, rows


def analyze_actions(
    actions: torch.Tensor,
    groups: dict[str, list[int]],
    oscillation_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate stochasticity, temporal roughness, and oscillation metrics."""
    if actions.ndim != 3:
        raise ValueError(f"Expected stacked actions (R, T, A), got {tuple(actions.shape)}")
    metric_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    oscillation_rows: list[dict[str, Any]] = []
    scopes = {"all": list(range(actions.shape[-1])), **groups}

    std_over_runs = actions.std(dim=0, unbiased=False)
    delta = actions[:, 1:, :] - actions[:, :-1, :]
    second_delta = delta[:, 1:, :] - delta[:, :-1, :]

    for scope, indices in scopes.items():
        for name, value in _summary(std_over_runs[:, indices]).items():
            metric_rows.append(
                {"scope": scope, "metric": f"across_run_std_{name}", "value": value}
            )
        delta_max_joint = delta[:, :, indices].abs().amax(dim=-1)
        for name, value in _summary(delta_max_joint).items():
            metric_rows.append(
                {"scope": scope, "metric": f"within_chunk_delta_{name}", "value": value}
            )
        second_max_joint = second_delta[:, :, indices].abs().amax(dim=-1)
        for name, value in _summary(second_max_joint).items():
            metric_rows.append(
                {"scope": scope, "metric": f"within_chunk_second_delta_{name}", "value": value}
            )
        group_ratio, joint_rows = _oscillation_scores(
            actions, indices, oscillation_threshold
        )
        metric_rows.append(
            {"scope": scope, "metric": "oscillation_ratio", "value": group_ratio}
        )
        for action_index, ratio, eligible_count in joint_rows:
            oscillation_rows.append(
                {
                    "scope": scope,
                    "action_index": action_index,
                    "ratio": ratio,
                    "eligible_pairs": eligible_count,
                }
            )

    reference = actions[0]
    for run in range(1, actions.shape[0]):
        difference = actions[run] - reference
        run_rows.append(
            {
                "run": run,
                "rmse_vs_run0": difference.square().mean().sqrt().item(),
                "l2_vs_run0": torch.linalg.vector_norm(difference).item(),
                "max_abs_vs_run0": difference.abs().max().item(),
            }
        )
    return metric_rows, run_rows, oscillation_rows


def _metric_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    return {(row["scope"], row["metric"]): float(row["value"]) for row in rows}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_action_csv(path: Path, conditions: dict[str, torch.Tensor]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "run", "timestep", "action_index", "value"])
        for condition, actions in conditions.items():
            for run in range(actions.shape[0]):
                for timestep in range(actions.shape[1]):
                    for action_index in range(actions.shape[2]):
                        writer.writerow(
                            [
                                condition,
                                run,
                                timestep,
                                action_index,
                                float(actions[run, timestep, action_index]),
                            ]
                        )


def _print_metric_section(
    condition: str, stage: str, rows: list[dict[str, Any]], run_rows: list[dict[str, Any]]
) -> None:
    lookup = _metric_lookup(rows)
    print(f"\n[{condition} / {stage}]")
    for scope in ("all", "arms", "right_arm", "left_arm", "gripper"):
        if (scope, "across_run_std_mean") not in lookup:
            continue
        print(
            f"  {scope:10s} across-run std "
            f"mean={lookup[(scope, 'across_run_std_mean')]:.8g} "
            f"median={lookup[(scope, 'across_run_std_median')]:.8g} "
            f"p95={lookup[(scope, 'across_run_std_p95')]:.8g} "
            f"p99={lookup[(scope, 'across_run_std_p99')]:.8g} "
            f"max={lookup[(scope, 'across_run_std_max')]:.8g}"
        )
    for prefix, label in (
        ("within_chunk_delta", "|delta q| max-joint"),
        ("within_chunk_second_delta", "|delta^2 q| max-joint"),
    ):
        print(
            f"  arms {label}: "
            f"median={lookup[('arms', prefix + '_median')]:.8g} "
            f"p95={lookup[('arms', prefix + '_p95')]:.8g} "
            f"p99={lookup[('arms', prefix + '_p99')]:.8g} "
            f"max={lookup[('arms', prefix + '_max')]:.8g}"
        )
    if run_rows:
        for metric in ("rmse_vs_run0", "l2_vs_run0", "max_abs_vs_run0"):
            values = torch.tensor([row[metric] for row in run_rows])
            summary = _summary(values)
            print(
                f"  run-to-run {metric}: median={summary['median']:.8g} "
                f"p95={summary['p95']:.8g} max={summary['max']:.8g}"
            )


def _interpret(
    random_metrics: dict[tuple[str, str], float],
    fixed_metrics: dict[tuple[str, str], float],
    *,
    determinism_atol: float,
    variance_ratio: float,
    rough_delta_threshold: float,
    rough_second_delta_threshold: float,
) -> list[str]:
    conclusions: list[str] = []
    random_mean = random_metrics[("all", "across_run_std_mean")]
    fixed_mean = fixed_metrics[("all", "across_run_std_mean")]
    fixed_max = fixed_metrics[("all", "across_run_std_max")]

    if fixed_max > determinism_atol:
        conclusions.append(
            "Output is changing even with identical observation and identical noise. "
            "Check hidden policy state, batch mutation, dropout/training mode, CUDA "
            "nondeterminism, or preprocessing state."
        )
    else:
        if random_mean > determinism_atol and random_mean > variance_ratio * max(
            fixed_mean, torch.finfo(torch.float64).eps
        ):
            conclusions.append(
                "Flow-matching random sampling is a major source of chunk-to-chunk variability."
            )
        delta_p95 = fixed_metrics[("arms", "within_chunk_delta_p95")]
        second_p95 = fixed_metrics[("arms", "within_chunk_second_delta_p95")]
        if (
            delta_p95 > rough_delta_threshold
            or second_p95 > rough_second_delta_threshold
        ):
            conclusions.append(
                "The generated trajectory itself is temporally rough even with deterministic "
                "noise. This points toward model/dataset trajectory quality rather than sampling "
                "stochasticity for the within-chunk component."
            )
    if not conclusions:
        conclusions.append(
            "No configured diagnostic heuristic fired; inspect the raw metrics and choose "
            "domain-specific thresholds for this robot and control rate."
        )
    return conclusions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Separate SmolVLA flow-noise variability from within-chunk roughness."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frozen_batch", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_runs", type=int, default=30)
    parser.add_argument("--output_dir", default="outputs/frozen_noise_analysis")
    parser.add_argument("--also_test_zero_noise", action="store_true")
    parser.add_argument("--oscillation_threshold", type=float, default=0.005)
    parser.add_argument("--right_arm_indices", default="0:7")
    parser.add_argument("--left_arm_indices", default="7:14")
    parser.add_argument("--gripper_indices", default="14:16")
    parser.add_argument("--determinism_atol", type=float, default=1e-6)
    parser.add_argument("--variance_ratio", type=float, default=10.0)
    parser.add_argument("--rough_delta_threshold", type=float, default=0.005)
    parser.add_argument("--rough_second_delta_threshold", type=float, default=0.005)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_runs < 2:
        raise ValueError("--num_runs must be at least 2")
    if args.oscillation_threshold < 0:
        raise ValueError("--oscillation_threshold must be non-negative")

    checkpoint = Path(args.checkpoint).expanduser()
    frozen_path = Path(args.frozen_batch).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    # Deferred imports keep server capture and metric unit tests independent of
    # heavyweight model/tokenizer initialization.
    from lerobot.policies import get_policy_class, make_pre_post_processors

    frozen_batch = torch.load(frozen_path, map_location="cpu", weights_only=False)
    if not isinstance(frozen_batch, dict):
        raise TypeError(f"Frozen batch must be a dict, got {type(frozen_batch).__name__}")
    checksum_before = nested_checksum(frozen_batch)

    policy_class = get_policy_class("smolvla")
    policy = policy_class.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()
    _, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        postprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    action_feature = policy.config.action_feature
    if action_feature is None:
        raise ValueError("Checkpoint config does not define the canonical action feature")
    actual_action_dim = int(action_feature.shape[0])
    groups = {
        "right_arm": _parse_indices(args.right_arm_indices, actual_action_dim),
        "left_arm": _parse_indices(args.left_arm_indices, actual_action_dim),
        "gripper": _parse_indices(args.gripper_indices, actual_action_dim),
    }
    groups["arms"] = groups["right_arm"] + groups["left_arm"]

    state = frozen_batch.get("observation.state")
    if not isinstance(state, torch.Tensor) or state.ndim < 1:
        raise ValueError("Frozen policy batch is missing tensor key 'observation.state'")
    batch_size = int(state.shape[0])
    if batch_size != 1:
        raise ValueError(f"Expected a captured real-observation batch size of 1, got {batch_size}")

    noise_shape = (
        batch_size,
        int(policy.config.chunk_size),
        int(policy.config.max_action_dim),
    )
    with torch.inference_mode():
        fixed_noise = policy.model.sample_noise(noise_shape, device).detach().clone()
    sanity_batch = clone_to_device(frozen_batch, device)
    batch_tensors = list(iter_tensor_descriptions(sanity_batch))

    print("Sanity checks")
    print(f"  checkpoint={checkpoint}")
    print(f"  frozen_batch={frozen_path}")
    print(f"  device={device}")
    print(f"  policy.training={policy.training}")
    print(f"  chunk_size={policy.config.chunk_size}")
    print(f"  max_action_dim={policy.config.max_action_dim}")
    print(f"  actual_action_dim={actual_action_dim}")
    print(f"  num_steps={policy.config.num_steps}")
    print(f"  batch keys={sorted(map(str, frozen_batch.keys()))}")
    for description in batch_tensors:
        print(f"  {description}")
    print(
        f"  fixed_noise: shape={tuple(fixed_noise.shape)} dtype={fixed_noise.dtype} "
        f"device={fixed_noise.device}"
    )
    print(f"Frozen batch checksum before run: {checksum_before}")
    del sanity_batch

    conditions: dict[str, TrialResults] = {}
    print(f"\nRunning RANDOM_NOISE ({args.num_runs} trials)...")
    conditions["RANDOM_NOISE"] = execute_condition(
        policy=policy,
        frozen_batch=frozen_batch,
        postprocessor=postprocessor,
        device=device,
        num_runs=args.num_runs,
        fixed_noise=None,
    )
    print(f"Running FIXED_NOISE ({args.num_runs} trials)...")
    conditions["FIXED_NOISE"] = execute_condition(
        policy=policy,
        frozen_batch=frozen_batch,
        postprocessor=postprocessor,
        device=device,
        num_runs=args.num_runs,
        fixed_noise=fixed_noise,
    )
    if args.also_test_zero_noise:
        print(f"Running ZERO_NOISE ({args.num_runs} trials; optional diagnostic)...")
        conditions["ZERO_NOISE"] = execute_condition(
            policy=policy,
            frozen_batch=frozen_batch,
            postprocessor=postprocessor,
            device=device,
            num_runs=args.num_runs,
            fixed_noise=torch.zeros_like(fixed_noise),
        )

    metric_rows_all: list[dict[str, Any]] = []
    run_rows_all: list[dict[str, Any]] = []
    oscillation_rows_all: list[dict[str, Any]] = []
    metrics_by_condition_stage: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for condition, results in conditions.items():
        for stage, actions in (
            ("raw_policy", results.raw_actions),
            ("robot_unit", results.robot_actions),
        ):
            rows, run_rows, oscillation_rows = analyze_actions(
                actions, groups, args.oscillation_threshold
            )
            metrics_by_condition_stage[(condition, stage)] = rows
            for row in rows:
                metric_rows_all.append({"condition": condition, "stage": stage, **row})
            for row in run_rows:
                run_rows_all.append({"condition": condition, "stage": stage, **row})
            for row in oscillation_rows:
                oscillation_rows_all.append({"condition": condition, "stage": stage, **row})
            _print_metric_section(condition, stage, rows, run_rows)

    fixed_robot = conditions["FIXED_NOISE"].robot_actions
    fixed_difference = (fixed_robot[1] - fixed_robot[0]).abs()
    print("\nFIXED trial0 vs trial1 (robot units):")
    print(f"  max_abs_diff={fixed_difference.max().item():.8g}")
    print(f"  mean_abs_diff={fixed_difference.mean().item():.8g}")

    comparison_specs = [
        ("Across-run mean std", "all", "across_run_std_mean"),
        ("Across-run p95 std", "all", "across_run_std_p95"),
        ("Across-run max std", "all", "across_run_std_max"),
        ("Within-chunk delta q p95", "arms", "within_chunk_delta_p95"),
        ("Within-chunk delta q max", "arms", "within_chunk_delta_max"),
        ("Within-chunk delta^2 q p95", "arms", "within_chunk_second_delta_p95"),
        ("Within-chunk delta^2 q max", "arms", "within_chunk_second_delta_max"),
        ("Right-arm oscillation ratio", "right_arm", "oscillation_ratio"),
        ("Left-arm oscillation ratio", "left_arm", "oscillation_ratio"),
    ]
    random_lookup = _metric_lookup(metrics_by_condition_stage[("RANDOM_NOISE", "robot_unit")])
    fixed_lookup = _metric_lookup(metrics_by_condition_stage[("FIXED_NOISE", "robot_unit")])
    comparison_rows = [
        {
            "metric": label,
            "RANDOM_NOISE": random_lookup[(scope, metric)],
            "FIXED_NOISE": fixed_lookup[(scope, metric)],
        }
        for label, scope, metric in comparison_specs
    ]
    print("\nRANDOM vs FIXED (robot units)")
    print(f"{'Metric':38s} {'RANDOM':>14s} {'FIXED':>14s}")
    print("-" * 68)
    for row in comparison_rows:
        print(
            f"{row['metric']:38s} {row['RANDOM_NOISE']:14.8g} "
            f"{row['FIXED_NOISE']:14.8g}"
        )

    conclusions = _interpret(
        random_lookup,
        fixed_lookup,
        determinism_atol=args.determinism_atol,
        variance_ratio=args.variance_ratio,
        rough_delta_threshold=args.rough_delta_threshold,
        rough_second_delta_threshold=args.rough_second_delta_threshold,
    )
    print("\nCONCLUSION (configured heuristics; verify against raw metrics):")
    for conclusion in conclusions:
        print(f"  - {conclusion}")

    for condition, results in conditions.items():
        stem = condition.lower().replace("_noise", "")
        torch.save(results.raw_actions, output_dir / f"{stem}_raw_actions.pt")
        torch.save(results.robot_actions, output_dir / f"{stem}_robot_actions.pt")
        # Short name denotes the final deployment/robot-unit action tensor.
        torch.save(results.robot_actions, output_dir / f"{stem}_actions.pt")

    _write_action_csv(
        output_dir / "robot_actions.csv",
        {condition: result.robot_actions for condition, result in conditions.items()},
    )
    _write_csv(
        output_dir / "metrics_summary.csv",
        metric_rows_all,
        ["condition", "stage", "scope", "metric", "value"],
    )
    _write_csv(
        output_dir / "run_differences.csv",
        run_rows_all,
        ["condition", "stage", "run", "rmse_vs_run0", "l2_vs_run0", "max_abs_vs_run0"],
    )
    _write_csv(
        output_dir / "oscillation_by_joint.csv",
        oscillation_rows_all,
        ["condition", "stage", "scope", "action_index", "ratio", "eligible_pairs"],
    )
    _write_csv(
        output_dir / "random_vs_fixed.csv",
        comparison_rows,
        ["metric", "RANDOM_NOISE", "FIXED_NOISE"],
    )

    checksum_after = nested_checksum(frozen_batch)
    metadata = {
        "checkpoint": str(checkpoint),
        "frozen_batch": str(frozen_path),
        "device": str(device),
        "num_runs": args.num_runs,
        "chunk_size": int(policy.config.chunk_size),
        "max_action_dim": int(policy.config.max_action_dim),
        "actual_action_dim": actual_action_dim,
        "num_steps": int(policy.config.num_steps),
        "groups": groups,
        "oscillation_threshold": args.oscillation_threshold,
        "determinism_atol": args.determinism_atol,
        "variance_ratio": args.variance_ratio,
        "rough_delta_threshold": args.rough_delta_threshold,
        "rough_second_delta_threshold": args.rough_second_delta_threshold,
        "frozen_checksum_before": checksum_before,
        "frozen_checksum_after": checksum_after,
        "frozen_unchanged": checksum_before == checksum_after,
        "fixed_noise_checksum": nested_checksum(fixed_noise),
        "latencies_s": {
            condition: result.latencies_s for condition, result in conditions.items()
        },
        "conclusions": conclusions,
    }
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)

    print(f"\nFrozen batch checksum after run:  {checksum_after}")
    print(f"unchanged={checksum_before == checksum_after}")
    print(f"Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
