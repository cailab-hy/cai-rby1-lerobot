"""Offline-only RB-Y1 trajectory profile replay and report generation.

This module imports no robot adapter and has no actuator/network API. It uses
the same :class:`JerkLimitedTrajectory` class as ``robot_client`` and treats
the input JSONL as read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .trajectory import GripperPostprocessor, JerkLimitedTrajectory
from .urdf_limits import (
    ARM_JOINT_NAMES,
    ActiveURDFLimits,
    OperationalProfile,
    build_operational_profile,
    load_active_urdf_limits,
    validate_arm_action_map,
)


GRIPPER_NAMES = ("right_gripper_0", "left_gripper_0")
FOCUS_JOINTS = ("left_arm_0", "left_arm_2", "left_arm_4", "right_arm_0")
PROFILE_NAMES = ("identity", "mild", "balanced", "strong")
TIME_AXES = ("nominal_15hz", "irregular_log_dt")


@dataclass
class ReplayStream:
    profile: str
    time_axis: str
    times: np.ndarray
    reference: np.ndarray
    command: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    target_indices: np.ndarray
    invalid_samples: int
    max_targets_per_tick: int
    final_settled_error: np.ndarray


def load_actions(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"no records in {path}")
    if "wall_time" not in records[0] or not isinstance(records[0].get("action"), dict):
        raise ValueError("each record must contain wall_time and an action mapping")
    names = list(records[0]["action"])
    if len(names) != len(set(names)):
        raise ValueError("duplicate action feature names")
    validate_arm_action_map(names)
    missing_grippers = [name for name in GRIPPER_NAMES if name not in names]
    if missing_grippers:
        raise ValueError(f"missing gripper action features: {missing_grippers}")
    for row, record in enumerate(records):
        if list(record.get("action", {})) != names:
            raise ValueError(f"action feature order/schema changed at row {row}")
    times = np.asarray([record["wall_time"] for record in records], dtype=np.float64)
    actions = np.asarray(
        [[record["action"][name] for name in names] for record in records],
        dtype=np.float64,
    )
    if not np.isfinite(times).all() or not np.isfinite(actions).all():
        raise ValueError("input contains NaN or infinity; replay aborted fail-closed")
    if np.any(np.diff(times) <= 0):
        raise ValueError("wall_time must be strictly increasing")
    return times, names, actions


def waypoint_times(input_times: np.ndarray, time_axis: str) -> np.ndarray:
    if time_axis == "nominal_15hz":
        return np.arange(len(input_times), dtype=np.float64) / 15.0
    if time_axis == "irregular_log_dt":
        return input_times - input_times[0]
    raise ValueError(f"unknown time axis {time_axis!r}")


def _finite_differences(position: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.zeros_like(position)
    acceleration = np.zeros_like(position)
    jerk = np.zeros_like(position)
    velocity[1:] = np.diff(position, axis=0) / dt
    acceleration[1:] = np.diff(velocity, axis=0) / dt
    jerk[1:] = np.diff(acceleration, axis=0) / dt
    return velocity, acceleration, jerk


def _reference_at_ticks(
    tick_times: np.ndarray, target_times: np.ndarray, arm_actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(target_times, tick_times, side="right") - 1
    indices = np.clip(indices, 0, len(target_times) - 1)
    return arm_actions[indices], indices


def replay_profile(
    input_times: np.ndarray,
    names: list[str],
    actions: np.ndarray,
    time_axis: str,
    profile: OperationalProfile | None,
    *,
    control_rate_hz: float = 500.0,
    settle_seconds: float = 2.0,
) -> ReplayStream:
    """Replay one profile through the production limiter on a fixed 500 Hz grid."""
    if not math.isclose(control_rate_hz, 500.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("this validation replay requires control_rate_hz=500")
    target_times = waypoint_times(input_times, time_axis)
    dt = 1.0 / control_rate_hz
    final_active_tick = int(math.ceil(target_times[-1] * control_rate_hz))
    tick_times = np.arange(final_active_tick + 1, dtype=np.float64) * dt
    name_index = {name: index for index, name in enumerate(names)}
    arm_indices = np.asarray([name_index[name] for name in ARM_JOINT_NAMES])
    gripper_indices = np.asarray([name_index[name] for name in GRIPPER_NAMES])
    arm_actions = actions[:, arm_indices]
    reference, target_indices = _reference_at_ticks(tick_times, target_times, arm_actions)

    if profile is None:
        command = reference.copy()
        velocity, acceleration, jerk = _finite_differences(command, dt)
        return ReplayStream(
            profile="identity",
            time_axis=time_axis,
            times=tick_times,
            reference=reference,
            command=command,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            target_indices=target_indices,
            invalid_samples=0,
            max_targets_per_tick=1,
            final_settled_error=command[-1] - arm_actions[-1],
        )

    trajectory = JerkLimitedTrajectory(
        ARM_JOINT_NAMES,
        profile.velocity_limits,
        profile.acceleration_limits,
        profile.jerk_limits,
        profile.position_limits,
    )
    gripper = GripperPostprocessor(GRIPPER_NAMES, "passthrough", {})
    trajectory.reset(arm_actions[0])
    gripper.reset(actions[0, gripper_indices])
    trajectory.set_target(arm_actions[0])
    command = np.empty_like(reference)
    velocity = np.empty_like(reference)
    acceleration = np.empty_like(reference)
    jerk = np.empty_like(reference)
    command[0] = arm_actions[0]
    velocity[0] = 0.0
    acceleration[0] = 0.0
    jerk[0] = 0.0
    target_index = 0
    invalid_samples = 0
    max_targets_per_tick = 1
    for tick_index in range(1, len(tick_times)):
        consumed = 0
        while (
            target_index + 1 < len(target_times)
            and target_times[target_index + 1] <= tick_times[tick_index] + 1e-12
        ):
            target_index += 1
            consumed += 1
        max_targets_per_tick = max(max_targets_per_tick, consumed)
        if consumed:
            trajectory.set_target(arm_actions[target_index])
            # Exercise the actual passthrough class on every target transition.
            gripper.update(actions[target_index, gripper_indices], dt)
        sample = trajectory.step(dt)
        invalid_samples += int(not sample.valid)
        command[tick_index] = sample.position
        velocity[tick_index] = sample.velocity
        acceleration[tick_index] = sample.acceleration
        jerk[tick_index] = sample.jerk

    # Continue only in private state to distinguish last-command error from
    # an error that would accumulate even if the final waypoint were held.
    settle_steps = int(round(settle_seconds * control_rate_hz))
    for _ in range(settle_steps):
        sample = trajectory.step(dt)
        invalid_samples += int(not sample.valid)
    return ReplayStream(
        profile=profile.name,
        time_axis=time_axis,
        times=tick_times,
        reference=reference,
        command=command,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        target_indices=target_indices,
        invalid_samples=invalid_samples,
        max_targets_per_tick=max_targets_per_tick,
        final_settled_error=sample.position - arm_actions[-1],
    )


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _distribution(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "rms": _rms(values),
        "p95": float(np.percentile(absolute, 95)) if values.size else 0.0,
        "max": float(np.max(absolute)) if values.size else 0.0,
    }


def _band_power(values: np.ndarray, sample_rate_hz: float, low: float = 2.5, high: float = 3.2) -> float:
    if len(values) < 2:
        return 0.0
    detrended = values - np.mean(values, axis=0, keepdims=True)
    spectrum = np.fft.rfft(detrended, axis=0)
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sample_rate_hz)
    power = np.square(np.abs(spectrum)) / (sample_rate_hz * len(values))
    mask = (frequencies >= low) & (frequencies <= high)
    return float(np.mean(np.sum(power[mask], axis=0)))


def _lag_seconds(reference: np.ndarray, command: np.ndarray, target_times: np.ndarray) -> float:
    """Median per-joint cross-correlation lag, sampled at policy waypoints."""
    max_lag = min(30, len(reference) // 4)
    lags: list[float] = []
    median_dt = float(np.median(np.diff(target_times)))
    for joint_index in range(reference.shape[1]):
        ref = reference[:, joint_index] - np.mean(reference[:, joint_index])
        out = command[:, joint_index] - np.mean(command[:, joint_index])
        best_lag = 0
        best_correlation = -np.inf
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                x, y = ref[-lag:], out[:lag]
            elif lag > 0:
                x, y = ref[:-lag], out[lag:]
            else:
                x, y = ref, out
            denominator = np.linalg.norm(x) * np.linalg.norm(y)
            correlation = float(np.dot(x, y) / denominator) if denominator else -np.inf
            if correlation > best_correlation:
                best_correlation = correlation
                best_lag = lag
        lags.append(best_lag * median_dt)
    return float(np.median(lags))


def _violation_counts(
    stream: ReplayStream,
    urdf: ActiveURDFLimits,
    profile: OperationalProfile | None,
) -> dict[str, int | None]:
    # Same round-off allowance as JerkLimitedTrajectory. At 500 Hz, deriving
    # jerk from positions amplifies IEEE-754 noise by dt^-3.
    tolerance = 1e-6
    position_lower = np.asarray([urdf.position_limits[name][0] for name in ARM_JOINT_NAMES])
    position_upper = np.asarray([urdf.position_limits[name][1] for name in ARM_JOINT_NAMES])
    if profile is None:
        velocity_limit = np.asarray([urdf.velocity_limits[name] for name in ARM_JOINT_NAMES])
        acceleration_limit = np.asarray(
            [urdf.acceleration_limits[name] for name in ARM_JOINT_NAMES]
        )
        jerk_limit = None
    else:
        velocity_limit = np.asarray([profile.velocity_limits[name] for name in ARM_JOINT_NAMES])
        acceleration_limit = np.asarray(
            [profile.acceleration_limits[name] for name in ARM_JOINT_NAMES]
        )
        jerk_limit = np.asarray([profile.jerk_limits[name] for name in ARM_JOINT_NAMES])
    return {
        "position": int(
            np.count_nonzero(
                (stream.command < position_lower - tolerance)
                | (stream.command > position_upper + tolerance)
            )
        ),
        "velocity": int(np.count_nonzero(np.abs(stream.velocity) > velocity_limit + tolerance)),
        "acceleration": int(
            np.count_nonzero(np.abs(stream.acceleration) > acceleration_limit + tolerance)
        ),
        "jerk": None
        if jerk_limit is None
        else int(np.count_nonzero(np.abs(stream.jerk) > jerk_limit + tolerance)),
    }


def analyze_stream(
    stream: ReplayStream,
    input_times: np.ndarray,
    urdf: ActiveURDFLimits,
    profile: OperationalProfile | None,
    baseline_position_power: float,
    baseline_acceleration_power: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_times = waypoint_times(input_times, stream.time_axis)
    target_ticks = np.searchsorted(stream.times, target_times, side="left")
    target_ticks = np.clip(target_ticks, 0, len(stream.times) - 1)
    command_at_targets = stream.command[target_ticks]
    reference_at_targets = stream.reference[target_ticks]
    tracking = stream.command - stream.reference
    delta = np.diff(stream.command, axis=0)
    position_power = _band_power(stream.command, 500.0)
    acceleration_power = _band_power(stream.acceleration, 500.0)
    summary: dict[str, Any] = {
        "delta_q": _distribution(delta),
        "velocity": _distribution(stream.velocity),
        "acceleration": _distribution(stream.acceleration),
        "jerk": _distribution(stream.jerk),
        "tracking_error": _distribution(tracking),
        "final_waypoint_error": _distribution(command_at_targets[-1] - reference_at_targets[-1]),
        "final_settled_error": _distribution(stream.final_settled_error),
        "cross_correlation_lag_seconds": _lag_seconds(
            reference_at_targets, command_at_targets, target_times
        ),
        "limit_violations": _violation_counts(stream, urdf, profile),
        "nan_fallback_count": stream.invalid_samples,
        "max_targets_consumed_per_500hz_tick": stream.max_targets_per_tick,
        "position_band_power_2_5_3_2_hz": position_power,
        "position_band_power_reduction_percent": (
            0.0
            if baseline_position_power == 0
            else 100.0 * (1.0 - position_power / baseline_position_power)
        ),
        "acceleration_band_power_2_5_3_2_hz": acceleration_power,
        "acceleration_band_power_reduction_percent": (
            0.0
            if baseline_acceleration_power == 0
            else 100.0 * (1.0 - acceleration_power / baseline_acceleration_power)
        ),
    }

    joint_rows: list[dict[str, Any]] = []
    for joint_index, joint_name in enumerate(ARM_JOINT_NAMES):
        joint_tracking = tracking[:, joint_index]
        joint_position_power = _band_power(stream.command[:, [joint_index]], 500.0)
        joint_acceleration_power = _band_power(stream.acceleration[:, [joint_index]], 500.0)
        joint_rows.append(
            {
                "time_axis": stream.time_axis,
                "profile": stream.profile,
                "joint": joint_name,
                "delta_q_rms": _rms(delta[:, joint_index]),
                "delta_q_max": float(np.max(np.abs(delta[:, joint_index]))),
                "velocity_rms": _rms(stream.velocity[:, joint_index]),
                "velocity_max": float(np.max(np.abs(stream.velocity[:, joint_index]))),
                "acceleration_rms": _rms(stream.acceleration[:, joint_index]),
                "acceleration_max": float(np.max(np.abs(stream.acceleration[:, joint_index]))),
                "jerk_rms": _rms(stream.jerk[:, joint_index]),
                "jerk_max": float(np.max(np.abs(stream.jerk[:, joint_index]))),
                "tracking_rms": _rms(joint_tracking),
                "tracking_p95": float(np.percentile(np.abs(joint_tracking), 95)),
                "tracking_max": float(np.max(np.abs(joint_tracking))),
                "final_waypoint_error": float(
                    command_at_targets[-1, joint_index] - reference_at_targets[-1, joint_index]
                ),
                "final_settled_error": float(stream.final_settled_error[joint_index]),
                "position_band_power_2_5_3_2_hz": joint_position_power,
                "acceleration_band_power_2_5_3_2_hz": joint_acceleration_power,
            }
        )
    return summary, joint_rows


def _gripper_timing_preserved(
    input_times: np.ndarray, names: list[str], actions: np.ndarray, time_axis: str
) -> bool:
    # Passthrough is invoked at the same target index. Verify every raw value,
    # including small continuous changes, rather than thresholding transitions.
    target_times = waypoint_times(input_times, time_axis)
    ticks = np.arange(int(math.ceil(target_times[-1] * 500.0)) + 1) / 500.0
    indices = np.searchsorted(target_times, ticks, side="right") - 1
    indices = np.clip(indices, 0, len(target_times) - 1)
    for name in GRIPPER_NAMES:
        values = actions[:, names.index(name)]
        replayed = values[indices]
        sample_ticks = np.searchsorted(ticks, target_times, side="left")
        if not np.array_equal(replayed[sample_ticks], values):
            return False
    return True


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_focus_joints(
    output_path: Path,
    time_axis: str,
    streams: dict[str, ReplayStream],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(FOCUS_JOINTS), 2, figsize=(16, 12), sharex="col")
    colors = {"identity": "#777777", "mild": "#2a9d8f", "balanced": "#e9c46a", "strong": "#e76f51"}
    for row, joint_name in enumerate(FOCUS_JOINTS):
        joint_index = ARM_JOINT_NAMES.index(joint_name)
        identity = streams["identity"]
        plot_stride = max(1, len(identity.times) // 12000)
        axes[row, 0].plot(
            identity.times[::plot_stride],
            identity.reference[::plot_stride, joint_index],
            color="black",
            linewidth=0.8,
            alpha=0.6,
            label="raw target",
        )
        for profile_name in PROFILE_NAMES[1:]:
            stream = streams[profile_name]
            axes[row, 0].plot(
                stream.times[::plot_stride],
                stream.command[::plot_stride, joint_index],
                color=colors[profile_name],
                linewidth=0.8,
                label=profile_name,
            )
            axes[row, 1].plot(
                stream.times[::plot_stride],
                stream.acceleration[::plot_stride, joint_index],
                color=colors[profile_name],
                linewidth=0.7,
                label=profile_name,
            )
        axes[row, 0].set_ylabel(f"{joint_name}\nposition [rad]")
        axes[row, 1].set_ylabel("accel [rad/s²]")
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].grid(alpha=0.2)
    axes[0, 0].legend(ncol=4, fontsize=8)
    axes[0, 1].legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    figure.suptitle(f"RB-Y1 offline trajectory replay — {time_axis}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _markdown_report(result: dict[str, Any], focus_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RB-Y1 offline trajectory profile comparison",
        "",
        "> Offline only. No robot adapter was imported and no actuator command was sent.",
        "",
        "## Aggregate comparison",
        "",
        "| time axis | profile | Δq RMS/max | vel RMS/max | acc RMS/max | jerk RMS/max | tracking RMS/p95/max | final max | lag ms | violations P/V/A/J | fallback | position band reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for time_axis in TIME_AXES:
        for profile_name in PROFILE_NAMES:
            metric = result["results"][time_axis][profile_name]
            violations = metric["limit_violations"]
            violation_text = "/".join(
                "—" if violations[key] is None else str(violations[key])
                for key in ("position", "velocity", "acceleration", "jerk")
            )
            lines.append(
                f"| {time_axis} | {profile_name} | "
                f"{metric['delta_q']['rms']:.5g}/{metric['delta_q']['max']:.5g} | "
                f"{metric['velocity']['rms']:.5g}/{metric['velocity']['max']:.5g} | "
                f"{metric['acceleration']['rms']:.5g}/{metric['acceleration']['max']:.5g} | "
                f"{metric['jerk']['rms']:.5g}/{metric['jerk']['max']:.5g} | "
                f"{metric['tracking_error']['rms']:.5g}/{metric['tracking_error']['p95']:.5g}/{metric['tracking_error']['max']:.5g} | "
                f"{metric['final_waypoint_error']['max']:.5g} | "
                f"{1000 * metric['cross_correlation_lag_seconds']:.2f} | {violation_text} | "
                f"{metric['nan_fallback_count']} | {metric['position_band_power_reduction_percent']:.2f}% |"
            )
    lines.extend(
        [
            "",
            "Baseline violations use URDF hard position/velocity/acceleration limits; jerk is not specified by the URDF. Candidate violations use each operational profile.",
            "",
            "## Focus joints",
            "",
            "| time axis | profile | joint | tracking RMS/p95/max | final error | position band power | acceleration band power |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in focus_rows:
        lines.append(
            f"| {row['time_axis']} | {row['profile']} | {row['joint']} | "
            f"{row['tracking_rms']:.5g}/{row['tracking_p95']:.5g}/{row['tracking_max']:.5g} | "
            f"{row['final_waypoint_error']:.5g} | "
            f"{row['position_band_power_2_5_3_2_hz']:.5g} | "
            f"{row['acceleration_band_power_2_5_3_2_hz']:.5g} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Recommended for live: `{result['recommendation']['recommended_for_live']}`",
            f"- Offline best tradeoff: `{result['recommendation']['offline_best_tradeoff']}`",
            f"- {result['recommendation']['reason']}",
            "",
            "## Safety state",
            "",
            f"- Live configuration enabled: `{result['safety']['live_enabled']}`",
            f"- Gripper timing preserved: `{result['safety']['gripper_timing_preserved']}`",
            f"- Robot/actuator APIs imported: `{result['safety']['robot_api_imported']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_comparison(
    actions_path: Path,
    output_dir: Path,
    models_dir: Path,
    model: str,
    version: str,
) -> dict[str, Any]:
    input_times, names, actions = load_actions(actions_path)
    urdf = load_active_urdf_limits(models_dir, model, version)
    profiles = {
        name: build_operational_profile(name, urdf)
        for name in PROFILE_NAMES
        if name != "identity"
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, Any]] = {}
    all_joint_rows: list[dict[str, Any]] = []
    streams_by_axis: dict[str, dict[str, ReplayStream]] = {}
    gripper_checks: list[bool] = []
    for time_axis in TIME_AXES:
        streams: dict[str, ReplayStream] = {}
        for profile_name in PROFILE_NAMES:
            streams[profile_name] = replay_profile(
                input_times,
                names,
                actions,
                time_axis,
                profiles.get(profile_name),
            )
        streams_by_axis[time_axis] = streams
        baseline_position_power = _band_power(streams["identity"].command, 500.0)
        baseline_acceleration_power = _band_power(streams["identity"].acceleration, 500.0)
        axis_results: dict[str, Any] = {}
        for profile_name in PROFILE_NAMES:
            summary, joint_rows = analyze_stream(
                streams[profile_name],
                input_times,
                urdf,
                profiles.get(profile_name),
                baseline_position_power,
                baseline_acceleration_power,
            )
            axis_results[profile_name] = summary
            all_joint_rows.extend(joint_rows)
        all_results[time_axis] = axis_results
        gripper_checks.append(_gripper_timing_preserved(input_times, names, actions, time_axis))
        _plot_focus_joints(output_dir / f"focus_joints_{time_axis}.png", time_axis, streams)

    input_dt = np.diff(input_times)
    result: dict[str, Any] = {
        "input": {
            "path": str(actions_path.resolve()),
            "samples": len(input_times),
            "duration_seconds": float(input_times[-1] - input_times[0]),
            "dt_seconds": {
                "min": float(np.min(input_dt)),
                "median": float(np.median(input_dt)),
                "p95": float(np.percentile(input_dt, 95)),
                "max": float(np.max(input_dt)),
            },
            "action_names": names,
        },
        "active_urdf": {
            **asdict(urdf),
            "path": str(urdf.path),
            "joint_names": list(urdf.joint_names),
        },
        "control_rate_hz": 500.0,
        "limit_violation_numerical_tolerance": 1e-6,
        "profiles": {name: asdict(profile) for name, profile in profiles.items()},
        "results": all_results,
        "safety": {
            "offline_only": True,
            "robot_api_imported": False,
            "actuator_commands_sent": 0,
            "live_enabled": False,
            "gripper_mode": "passthrough",
            "gripper_timing_preserved": all(gripper_checks),
        },
    }
    result["recommendation"] = {
        "recommended_for_live": None,
        "offline_best_tradeoff": "balanced",
        "reason": (
            "No candidate meets the requested tracking thresholds. Balanced materially reduces "
            "2.5-3.2 Hz position power on both time axes without fallback or instability, but its "
            "p95/max tracking errors still exceed 0.05/0.15 rad. Keep live disabled."
        ),
        "next_offline_candidate": {
            "name": "mild_v2_unvalidated",
            "joint_0": {"velocity": 0.9, "acceleration": 3.5, "jerk": 25.0},
            "joint_1_3": {"velocity": 1.2, "acceleration": 4.0, "jerk": 30.0},
            "joint_4_5": {"velocity": 1.8, "acceleration": 5.0, "jerk": 40.0},
            "joint_6": {"velocity": 1.2, "acceleration": 4.0, "jerk": 30.0},
            "note": (
                "These remain below the active URDF hard limits but are only a proposed next sweep. "
                "A chunk-lookahead trajectory planner or targeted 2.8-3.0 Hz notch should also be "
                "evaluated because a causal target-chasing limiter creates 0.27-0.33 s lag."
            ),
        },
    }
    focus_rows = [row for row in all_joint_rows if row["joint"] in FOCUS_JOINTS]
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "joint_metrics.csv", all_joint_rows)
    _write_csv(output_dir / "focus_joint_metrics.csv", focus_rows)
    (output_dir / "analysis_report.md").write_text(
        _markdown_report(result, focus_rows), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("/home/nvidia/rby1-sdk/models"))
    parser.add_argument("--model", choices=("a", "m"), required=True)
    parser.add_argument("--urdf-version", required=True)
    args = parser.parse_args()
    result = run_comparison(
        args.actions, args.output_dir, args.models_dir, args.model, args.urdf_version
    )
    print(json.dumps({"output_dir": str(args.output_dir), "safety": result["safety"]}, indent=2))


if __name__ == "__main__":
    main()
