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
    frozen_checksum: str
    noise_checksum: str | None


def execute_condition(
    *,
    policy: Any,
    frozen_batch: dict[str, Any],
    postprocessor: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    num_runs: int,
    fixed_noise: torch.Tensor | None,
    warmup_runs: int = 0,
) -> TrialResults:
    """Run RANDOM (``fixed_noise=None``) or a fixed external-noise condition."""
    raw_runs: list[torch.Tensor] = []
    robot_runs: list[torch.Tensor] = []
    latencies: list[float] = []
    frozen_checksum = nested_checksum(frozen_batch)
    noise_checksum = nested_checksum(fixed_noise) if fixed_noise is not None else None

    policy.eval()
    for run in range(warmup_runs + num_runs):
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
        elapsed = time.perf_counter() - started

        if fixed_noise is not None and nested_checksum(fixed_noise) != noise_checksum:
            raise RuntimeError("The shared fixed-noise tensor was mutated during a trial")
        if run < warmup_runs:
            continue

        latencies.append(elapsed)

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

    if nested_checksum(frozen_batch) != frozen_checksum:
        raise RuntimeError("The source frozen batch was mutated during the experiment")
    return TrialResults(
        raw_actions=torch.stack(raw_runs),
        robot_actions=torch.stack(robot_runs),
        latencies_s=latencies,
        frozen_checksum=frozen_checksum,
        noise_checksum=noise_checksum,
    )


def _summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().to(torch.float64).flatten()
    if flat.numel() == 0:
        return {
            key: math.nan
            for key in ("mean", "median", "p90", "p95", "p99", "min", "max")
        }
    return {
        "mean": flat.mean().item(),
        "median": flat.median().item(),
        "p90": torch.quantile(flat, 0.90).item(),
        "p95": torch.quantile(flat, 0.95).item(),
        "p99": torch.quantile(flat, 0.99).item(),
        "min": flat.min().item(),
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
                "mean_abs_vs_run0": difference.abs().mean().item(),
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


def diagnose_questions(
    random_metrics: dict[tuple[str, str], float],
    fixed_metrics: dict[tuple[str, str], float],
    *,
    determinism_atol: float,
    variance_ratio: float,
    rough_delta_threshold: float,
    rough_second_delta_threshold: float,
) -> dict[str, Any]:
    """Answer Q1-Q3 and select a primary A/B/C diagnostic classification.

    C has highest priority because a nondeterministic fixed condition invalidates
    clean attribution to either sampling noise or one deterministic trajectory.
    If A and B are both observed, B is primary for the reported within-chunk
    jitter, while A is retained as a secondary chunk-to-chunk finding.
    """
    random_mean = random_metrics[("all", "across_run_std_mean")]
    fixed_mean = fixed_metrics[("all", "across_run_std_mean")]
    fixed_max = fixed_metrics[("all", "across_run_std_max")]
    fixed_delta_p95 = fixed_metrics[("arms", "within_chunk_delta_p95")]
    fixed_second_delta_p95 = fixed_metrics[
        ("arms", "within_chunk_second_delta_p95")
    ]
    std_ratio = random_mean / max(fixed_mean, torch.finfo(torch.float64).eps)

    q3_fixed_changes = fixed_max > determinism_atol
    q1_random_major = (
        not q3_fixed_changes
        and random_mean > determinism_atol
        and std_ratio > variance_ratio
    )
    q2_fixed_rough = (
        fixed_delta_p95 > rough_delta_threshold
        or fixed_second_delta_p95 > rough_second_delta_threshold
    )

    if q3_fixed_changes:
        q1_answer = "INCONCLUSIVE"
        primary = "C"
        primary_text = (
            "Fixed observation + fixed noise is not repeatable; investigate policy state, "
            "batch/preprocessing mutation, training mode, or CUDA nondeterminism first."
        )
        secondary: list[str] = []
    else:
        q1_answer = "YES" if q1_random_major else "NO"
        secondary = []
        if q2_fixed_rough:
            primary = "B"
            primary_text = (
                "The fixed-noise trajectory remains temporally rough; prioritize dataset "
                "trajectory quality, training, and temporal regularization."
            )
            if q1_random_major:
                secondary.append(
                    "A: Random sampling also contributes substantial chunk-to-chunk variability."
                )
        elif q1_random_major:
            primary = "A"
            primary_text = (
                "Random flow-matching sampling is the detected source; consider deterministic "
                "noise or sampling averaging."
            )
        else:
            primary = "INCONCLUSIVE"
            primary_text = (
                "Neither A, B, nor C crossed the configured thresholds; inspect raw metrics and "
                "set robot/control-rate-specific thresholds."
            )

    return {
        "q1": {
            "answer": q1_answer,
            "question": "Does random flow noise substantially change chunks for one observation?",
            "evidence": {
                "random_across_run_mean_std": random_mean,
                "fixed_across_run_mean_std": fixed_mean,
                "random_to_fixed_std_ratio": std_ratio,
                "required_ratio": variance_ratio,
            },
        },
        "q2": {
            "answer": "YES" if q2_fixed_rough else "NO",
            "question": "Is the fixed-noise trajectory temporally rough within a chunk?",
            "evidence": {
                "fixed_arm_delta_p95": fixed_delta_p95,
                "delta_threshold": rough_delta_threshold,
                "fixed_arm_second_delta_p95": fixed_second_delta_p95,
                "second_delta_threshold": rough_second_delta_threshold,
            },
        },
        "q3": {
            "answer": "YES" if q3_fixed_changes else "NO",
            "question": "Does output change with fixed observation and fixed noise?",
            "evidence": {
                "fixed_across_run_max_std": fixed_max,
                "determinism_atol": determinism_atol,
            },
        },
        "primary_classification": primary,
        "primary_interpretation": primary_text,
        "secondary_findings": secondary,
    }


def _diagnosis_csv_rows(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_key in ("q1", "q2", "q3"):
        question = diagnosis[question_key]
        for metric, value in question["evidence"].items():
            rows.append(
                {
                    "question": question_key.upper(),
                    "answer": question["answer"],
                    "metric": metric,
                    "value": value,
                }
            )
    rows.append(
        {
            "question": "PRIMARY",
            "answer": diagnosis["primary_classification"],
            "metric": "interpretation",
            "value": diagnosis["primary_interpretation"],
        }
    )
    for index, finding in enumerate(diagnosis["secondary_findings"], start=1):
        rows.append(
            {
                "question": "SECONDARY",
                "answer": str(index),
                "metric": "finding",
                "value": finding,
            }
        )
    return rows


def _parse_num_steps_list(spec: str) -> list[int]:
    values = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not values:
        raise ValueError("--num-steps-list must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("Every --num-steps-list value must be positive")
    if len(values) != len(set(values)):
        raise ValueError("--num-steps-list values must be unique")
    return values


def _is_torch_compiled(callable_object: Any) -> bool:
    """Best-effort runtime check for a callable returned by ``torch.compile``."""
    return hasattr(callable_object, "_torchdynamo_orig_callable")


def _load_fixed_noise(
    path: Path,
    *,
    expected_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(saved, dict):
        for key in ("fixed_noise", "noise"):
            if isinstance(saved.get(key), torch.Tensor):
                saved = saved[key]
                break
    if not isinstance(saved, torch.Tensor):
        raise TypeError(
            f"Fixed-noise file must contain a tensor (or fixed_noise/noise key), got {type(saved).__name__}"
        )
    if tuple(saved.shape) != expected_shape:
        raise ValueError(
            f"Fixed noise shape {tuple(saved.shape)} does not match expected {expected_shape}"
        )
    if not saved.is_floating_point():
        raise TypeError(f"Fixed noise must be floating point, got {saved.dtype}")
    return saved.detach().clone().to(device)


def _num_steps_interpretation(
    step_values: list[int], lookups: dict[int, dict[tuple[str, str], float]]
) -> dict[str, Any]:
    """Return a deliberately conservative, explicitly heuristic interpretation."""
    ordered = sorted(step_values)
    if len(ordered) < 2:
        return {
            "classification": "INSUFFICIENT_CONDITIONS",
            "text": "At least two num_steps conditions are needed to assess a trend.",
        }

    second_p95 = [
        lookups[step][("arms", "within_chunk_second_delta_p95")] for step in ordered
    ]
    delta_p95 = [lookups[step][("arms", "within_chunk_delta_p95")] for step in ordered]
    oscillation = [
        torch.tensor(
            [
                lookups[step][("right_arm", "oscillation_ratio")],
                lookups[step][("left_arm", "oscillation_ratio")],
            ],
            dtype=torch.float64,
        ).nanmean().item()
        for step in ordered
    ]

    def relative_reduction(values: list[float]) -> float:
        baseline = abs(values[0])
        if baseline <= torch.finfo(torch.float64).eps:
            return 0.0
        return (values[0] - values[-1]) / baseline

    second_reduction = relative_reduction(second_p95)
    delta_reduction = relative_reduction(delta_p95)
    oscillation_reduction = relative_reduction(oscillation)
    second_monotonic = all(b <= a for a, b in zip(second_p95, second_p95[1:], strict=False))
    oscillation_monotonic = all(
        b <= a for a, b in zip(oscillation, oscillation[1:], strict=False)
    )

    # These thresholds only choose wording; all raw metrics and reductions are
    # emitted so the heuristic is never presented as a statistical test.
    if (
        second_monotonic
        and oscillation_monotonic
        and second_reduction >= 0.20
        and oscillation_reduction >= 0.20
    ):
        classification = "SIGNIFICANT_IMPROVEMENT"
        text = (
            "Increasing Euler integration resolution materially improves temporal smoothness. "
            "Coarse flow integration is likely contributing to rough action trajectories."
        )
    elif (
        abs(delta_reduction) < 0.05
        and abs(second_reduction) < 0.05
        and abs(oscillation_reduction) < 0.05
    ):
        classification = "LITTLE_OR_NO_IMPROVEMENT"
        text = (
            "Increasing Euler integration resolution does not materially improve temporal smoothness. "
            "The roughness is more likely associated with the learned flow field / trajectory "
            "distribution than solver resolution."
        )
    else:
        classification = "MIXED_OR_MODEST_CHANGE"
        text = (
            "The num_steps effect is mixed or modest; the raw roughness metrics do not support a "
            "strong attribution to Euler resolution alone."
        )
    return {
        "classification": classification,
        "text": text,
        "ordered_steps": ordered,
        "delta_p95": delta_p95,
        "second_delta_p95": second_p95,
        "mean_arm_oscillation": oscillation,
        "delta_p95_reduction": delta_reduction,
        "second_delta_p95_reduction": second_reduction,
        "mean_arm_oscillation_reduction": oscillation_reduction,
        "heuristic": (
            "significant requires monotonic >=20% reductions in both robot-unit arm second-delta "
            "p95 and mean arm oscillation; little/no change requires delta p95, second-delta p95, "
            "and oscillation absolute reductions all <5%"
        ),
    }


def _run_num_steps_analysis(
    *,
    args: argparse.Namespace,
    policy: Any,
    frozen_batch: dict[str, Any],
    postprocessor: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    output_dir: Path,
    groups: dict[str, list[int]],
    checkpoint: Path,
    frozen_path: Path,
    actual_action_dim: int,
    checkpoint_compile_model: bool,
) -> int:
    step_values = _parse_num_steps_list(args.num_steps_list)
    noise_shape = (
        1,
        int(policy.config.chunk_size),
        int(policy.config.max_action_dim),
    )
    if args.fixed_noise_file:
        fixed_noise_path = Path(args.fixed_noise_file).expanduser()
        fixed_noise = _load_fixed_noise(
            fixed_noise_path, expected_shape=noise_shape, device=device
        )
        noise_source = str(fixed_noise_path)
    else:
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(args.noise_seed)
            with torch.inference_mode():
                fixed_noise = policy.model.sample_noise(noise_shape, device).detach().clone()
        noise_source = f"generated once with seed={args.noise_seed}"

    frozen_checksum = nested_checksum(frozen_batch)
    noise_checksum = nested_checksum(fixed_noise)
    torch.save(fixed_noise.detach().cpu(), output_dir / "fixed_noise.pt")

    print("Sanity checks: num_steps experiment")
    print(f"  checkpoint={checkpoint}")
    print(f"  frozen_batch={frozen_path}")
    print(f"  device={device}")
    print(f"  num_steps_list={step_values}")
    print(f"  runs_per_step={args.runs_per_step} (plus one unmeasured warmup)")
    print(f"  checkpoint compile_model={checkpoint_compile_model}")
    print(f"  runtime compile_model={policy.config.compile_model}")
    print(f"  sample_actions torch-compiled={_is_torch_compiled(policy.model.sample_actions)}")
    print(f"  RTC enabled={policy.model._rtc_enabled()}")
    print(f"  fixed_noise source={noise_source}")
    print(
        f"  fixed_noise shape={tuple(fixed_noise.shape)} dtype={fixed_noise.dtype} "
        f"device={fixed_noise.device}"
    )
    print(f"  fixed_noise checksum={noise_checksum}")
    print(f"  frozen batch checksum={frozen_checksum}")

    if policy.config.compile_model or _is_torch_compiled(policy.model.sample_actions):
        raise RuntimeError("num_steps diagnostic requires an eagerly constructed policy")
    if policy.model._rtc_enabled():
        raise RuntimeError("num_steps diagnostic requires RTC to be disabled")

    conditions: dict[int, TrialResults] = {}
    metric_rows_all: list[dict[str, Any]] = []
    oscillation_rows_all: list[dict[str, Any]] = []
    run_rows_all: list[dict[str, Any]] = []
    robot_lookups: dict[int, dict[tuple[str, str], float]] = {}
    latency_summaries: dict[int, dict[str, float]] = {}

    for num_steps in step_values:
        policy.config.num_steps = num_steps
        if policy.model.config is not policy.config:
            policy.model.config.num_steps = num_steps
        if policy.config.num_steps != num_steps or policy.model.config.num_steps != num_steps:
            raise RuntimeError(f"Failed to apply num_steps={num_steps} to the inference model")

        print(f"\nRunning num_steps={num_steps}...")
        print(
            f"  runtime num_steps: policy.config={policy.config.num_steps} "
            f"model.config={policy.model.config.num_steps}"
        )
        result = execute_condition(
            policy=policy,
            frozen_batch=frozen_batch,
            postprocessor=postprocessor,
            device=device,
            num_runs=args.runs_per_step,
            fixed_noise=fixed_noise,
            warmup_runs=1,
        )
        conditions[num_steps] = result
        if result.noise_checksum != noise_checksum or result.frozen_checksum != frozen_checksum:
            raise RuntimeError(f"Fairness checksum mismatch at num_steps={num_steps}")

        for stage, actions in (
            ("raw_policy", result.raw_actions),
            ("robot_unit", result.robot_actions),
        ):
            rows, _, oscillation_rows = analyze_actions(
                actions[:1], groups, args.oscillation_threshold
            )
            if stage == "robot_unit":
                robot_lookups[num_steps] = _metric_lookup(rows)
            metric_rows_all.extend(
                {"num_steps": num_steps, "stage": stage, **row} for row in rows
            )
            oscillation_rows_all.extend(
                {"num_steps": num_steps, "stage": stage, **row}
                for row in oscillation_rows
            )
            # Determinism comparisons need all measured runs, while temporal
            # metrics deliberately use only representative run 0.
            _, all_run_rows, _ = analyze_actions(
                actions, groups, args.oscillation_threshold
            )
            run_rows_all.extend(
                {"num_steps": num_steps, "stage": stage, **row}
                for row in all_run_rows
            )

        latency_ms = torch.tensor(result.latencies_s, dtype=torch.float64) * 1000.0
        latency_summaries[num_steps] = _summary(latency_ms)
        robot_differences = [
            {
                "run": run,
                "max": (result.robot_actions[run] - result.robot_actions[0]).abs().max().item(),
                "mean": (result.robot_actions[run] - result.robot_actions[0]).abs().mean().item(),
            }
            for run in range(1, args.runs_per_step)
        ]
        for difference in robot_differences:
            print(
                f"  run{difference['run']} vs run0 robot-unit: "
                f"max_abs_diff={difference['max']:.8g} "
                f"mean_abs_diff={difference['mean']:.8g}"
            )
        deterministic = all(
            difference["max"] <= args.determinism_atol for difference in robot_differences
        )
        print(f"  deterministic within atol={args.determinism_atol:g}: {deterministic}")
        print(
            f"  checksums: fixed_noise={result.noise_checksum} "
            f"frozen_batch={result.frozen_checksum}"
        )

        torch.save(result.robot_actions[0], output_dir / f"actions_steps_{num_steps}.pt")
        torch.save(result.raw_actions[0], output_dir / f"raw_actions_steps_{num_steps}.pt")
        torch.save(
            {
                "raw_policy": result.raw_actions,
                "robot_unit": result.robot_actions,
                "latencies_s": result.latencies_s,
            },
            output_dir / f"all_runs_steps_{num_steps}.pt",
        )

    metric_specs = [
        ("delta_q_median", "arms", "within_chunk_delta_median"),
        ("delta_q_p90", "arms", "within_chunk_delta_p90"),
        ("delta_q_p95", "arms", "within_chunk_delta_p95"),
        ("delta_q_p99", "arms", "within_chunk_delta_p99"),
        ("delta_q_max", "arms", "within_chunk_delta_max"),
        ("second_delta_q_median", "arms", "within_chunk_second_delta_median"),
        ("second_delta_q_p90", "arms", "within_chunk_second_delta_p90"),
        ("second_delta_q_p95", "arms", "within_chunk_second_delta_p95"),
        ("second_delta_q_p99", "arms", "within_chunk_second_delta_p99"),
        ("second_delta_q_max", "arms", "within_chunk_second_delta_max"),
        ("right_oscillation", "right_arm", "oscillation_ratio"),
        ("left_oscillation", "left_arm", "oscillation_ratio"),
    ]
    summary_rows = [
        {
            "metric": label,
            **{
                f"steps={step}": robot_lookups[step][(scope, metric)]
                for step in step_values
            },
        }
        for label, scope, metric in metric_specs
    ]
    for statistic in ("mean", "median", "min", "max"):
        summary_rows.append(
            {
                "metric": f"inference_latency_{statistic}_ms",
                **{
                    f"steps={step}": latency_summaries[step][statistic]
                    for step in step_values
                },
            }
        )

    print("\nRobot-unit temporal roughness summary")
    header = f"{'Metric':36s}" + "".join(f" {f'steps={step}':>16s}" for step in step_values)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        values = "".join(f" {float(row[f'steps={step}']):16.8g}" for step in step_values)
        print(f"{row['metric']:36s}{values}")

    baseline_step = step_values[0]
    latency_increase = {
        step: latency_summaries[step]["mean"] / latency_summaries[baseline_step]["mean"]
        for step in step_values
    }
    print(f"\nMean latency multipliers vs steps={baseline_step}:")
    for step in step_values:
        print(f"  steps={step}: {latency_increase[step]:.4f}x")

    if 10 in robot_lookups:
        reference = {
            "delta_q_median": 0.02407157,
            "delta_q_p95": 0.053782679,
            "delta_q_max": 0.056116834,
            "second_delta_q_median": 0.027014613,
            "second_delta_q_p95": 0.052834749,
            "second_delta_q_max": 0.05941534,
            "right_oscillation": 0.90625,
            "left_oscillation": 0.24117647,
        }
        current = {row[0]: robot_lookups[10][(row[1], row[2])] for row in metric_specs}
        print("\nsteps=10 reference sanity check (different noise may legitimately differ):")
        for metric, reference_value in reference.items():
            print(
                f"  {metric}: current={current[metric]:.8g} "
                f"previous={reference_value:.8g} diff={current[metric] - reference_value:+.8g}"
            )

    interpretation = _num_steps_interpretation(step_values, robot_lookups)
    print(f"\nINTERPRETATION [{interpretation['classification']}]")
    print(f"  {interpretation['text']}")
    if "heuristic" in interpretation:
        print(f"  Heuristic only: {interpretation['heuristic']}")

    _write_csv(
        output_dir / "num_steps_summary.csv",
        summary_rows,
        ["metric", *(f"steps={step}" for step in step_values)],
    )
    _write_csv(
        output_dir / "num_steps_metrics.csv",
        metric_rows_all,
        ["num_steps", "stage", "scope", "metric", "value"],
    )
    _write_csv(
        output_dir / "num_steps_run_differences.csv",
        run_rows_all,
        [
            "num_steps",
            "stage",
            "run",
            "rmse_vs_run0",
            "l2_vs_run0",
            "mean_abs_vs_run0",
            "max_abs_vs_run0",
        ],
    )
    _write_csv(
        output_dir / "num_steps_oscillation_by_joint.csv",
        oscillation_rows_all,
        ["num_steps", "stage", "scope", "action_index", "ratio", "eligible_pairs"],
    )

    frozen_checksum_after = nested_checksum(frozen_batch)
    metadata = {
        "checkpoint": str(checkpoint),
        "frozen_batch": str(frozen_path),
        "device": str(device),
        "num_steps": step_values,
        "runs_per_step": args.runs_per_step,
        "warmup_runs_per_step": 1,
        "chunk_size": int(policy.config.chunk_size),
        "max_action_dim": int(policy.config.max_action_dim),
        "actual_action_dim": actual_action_dim,
        "groups": groups,
        "oscillation_threshold": args.oscillation_threshold,
        "checkpoint_compile_model": checkpoint_compile_model,
        "runtime_compile_model": bool(policy.config.compile_model),
        "sample_actions_torch_compiled": _is_torch_compiled(policy.model.sample_actions),
        "rtc_enabled": bool(policy.model._rtc_enabled()),
        "fixed_noise_source": noise_source,
        "fixed_noise_checksum": noise_checksum,
        "fixed_noise_checksums_by_step": {
            str(step): conditions[step].noise_checksum for step in step_values
        },
        "frozen_checksum_before": frozen_checksum,
        "frozen_checksum_after": frozen_checksum_after,
        "frozen_checksums_by_step": {
            str(step): conditions[step].frozen_checksum for step in step_values
        },
        "frozen_unchanged": frozen_checksum == frozen_checksum_after,
        "latency_ms": latency_summaries,
        "latency_multiplier_vs_first_condition": latency_increase,
        "interpretation": interpretation,
    }
    with (output_dir / "num_steps_summary.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)

    print(f"\nFrozen batch checksum after run: {frozen_checksum_after}")
    print(f"Frozen batch unchanged={frozen_checksum == frozen_checksum_after}")
    print(f"Results written to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Separate SmolVLA flow-noise variability from within-chunk roughness."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frozen_batch", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_runs", type=int, default=30)
    parser.add_argument(
        "--num_steps_list",
        "--num-steps-list",
        dest="num_steps_list",
        help="Run fixed-noise Euler-resolution mode, for example 10,20,40.",
    )
    parser.add_argument(
        "--runs_per_step",
        "--runs-per-step",
        dest="runs_per_step",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--fixed_noise_file",
        "--fixed-noise-file",
        dest="fixed_noise_file",
        help="Reuse a saved fixed-noise tensor instead of generating one.",
    )
    parser.add_argument("--noise_seed", "--noise-seed", dest="noise_seed", type=int, default=0)
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
    if args.num_steps_list is None and args.num_runs < 2:
        raise ValueError("--num_runs must be at least 2")
    if args.num_steps_list is not None and args.runs_per_step < 2:
        raise ValueError("--runs-per-step must be at least 2 for determinism checks")
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
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    frozen_batch = torch.load(frozen_path, map_location="cpu", weights_only=False)
    if not isinstance(frozen_batch, dict):
        raise TypeError(f"Frozen batch must be a dict, got {type(frozen_batch).__name__}")
    checksum_before = nested_checksum(frozen_batch)

    policy_class = get_policy_class("smolvla")
    checkpoint_compile_model = False
    if args.num_steps_list is not None:
        # compile_model is consumed inside VLAFlowMatching.__init__. Loading and
        # overriding the config before policy construction prevents a compiled
        # sample_actions callable from ever being installed.
        policy_config = PreTrainedConfig.from_pretrained(checkpoint)
        checkpoint_compile_model = bool(policy_config.compile_model)
        policy_config.compile_model = False
        policy_config.device = str(device)
        # RTC changes the Euler velocity path and is outside this single-variable
        # experiment. None also avoids constructing a debug-only RTC processor.
        policy_config.rtc_config = None
        policy = policy_class.from_pretrained(checkpoint, config=policy_config)
    else:
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

    if args.num_steps_list is not None:
        return _run_num_steps_analysis(
            args=args,
            policy=policy,
            frozen_batch=frozen_batch,
            postprocessor=postprocessor,
            device=device,
            output_dir=output_dir,
            groups=groups,
            checkpoint=checkpoint,
            frozen_path=frozen_path,
            actual_action_dim=actual_action_dim,
            checkpoint_compile_model=checkpoint_compile_model,
        )

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

    diagnosis = diagnose_questions(
        random_lookup,
        fixed_lookup,
        determinism_atol=args.determinism_atol,
        variance_ratio=args.variance_ratio,
        rough_delta_threshold=args.rough_delta_threshold,
        rough_second_delta_threshold=args.rough_second_delta_threshold,
    )
    print("\nQ1-Q3 DIAGNOSTIC ANSWERS (configured heuristics):")
    for question_key in ("q1", "q2", "q3"):
        question = diagnosis[question_key]
        evidence = ", ".join(
            f"{name}={value:.8g}" if isinstance(value, float) else f"{name}={value}"
            for name, value in question["evidence"].items()
        )
        print(
            f"  {question_key.upper()}: {question['answer']} | "
            f"{question['question']} | {evidence}"
        )
    print(
        f"\nPRIMARY_CLASSIFICATION: {diagnosis['primary_classification']}\n"
        f"  {diagnosis['primary_interpretation']}"
    )
    for finding in diagnosis["secondary_findings"]:
        print(f"  Secondary: {finding}")

    for condition, results in conditions.items():
        stem = condition.lower().replace("_noise", "")
        torch.save(results.raw_actions, output_dir / f"{stem}_raw_actions.pt")
        torch.save(results.robot_actions, output_dir / f"{stem}_robot_actions.pt")
        # Short name denotes the final deployment/robot-unit action tensor.
        torch.save(results.robot_actions, output_dir / f"{stem}_actions.pt")
    # Keep the exact x_1 sample so a later num_steps sweep can reproduce the
    # fixed-noise baseline with --fixed-noise-file.
    torch.save(fixed_noise.detach().cpu(), output_dir / "fixed_noise.pt")

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
        [
            "condition",
            "stage",
            "run",
            "rmse_vs_run0",
            "l2_vs_run0",
            "mean_abs_vs_run0",
            "max_abs_vs_run0",
        ],
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
    _write_csv(
        output_dir / "diagnostic_answers.csv",
        _diagnosis_csv_rows(diagnosis),
        ["question", "answer", "metric", "value"],
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
        "diagnosis": diagnosis,
    }
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)

    print(f"\nFrozen batch checksum after run:  {checksum_after}")
    print(f"unchanged={checksum_before == checksum_after}")
    print(f"Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
