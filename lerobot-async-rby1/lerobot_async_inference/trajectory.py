"""Stateful, dependency-free joint trajectory post-processing.

The generator works in command space.  Its velocity, acceleration and jerk are
the finite differences of the positions actually returned to the caller, so
the limits checked in tests and logs are the same limits seen by the actuator
command stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TrajectorySample:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    valid: bool = True
    error: str | None = None


class JerkLimitedTrajectory:
    """Online per-joint position command generator with persistent state."""

    def __init__(
        self,
        joint_names: Sequence[str],
        velocity_limits: Mapping[str, float],
        acceleration_limits: Mapping[str, float],
        jerk_limits: Mapping[str, float],
        position_limits: Mapping[str, Sequence[float]],
    ) -> None:
        self.joint_names = tuple(joint_names)
        self.velocity_limit = self._positive_array("velocity", velocity_limits)
        self.acceleration_limit = self._positive_array("acceleration", acceleration_limits)
        self.jerk_limit = self._positive_array("jerk", jerk_limits)
        missing_position = [name for name in self.joint_names if name not in position_limits]
        if missing_position:
            raise ValueError(f"missing position limits for: {missing_position}")
        bounds = np.asarray([position_limits[name] for name in self.joint_names], dtype=np.float64)
        if bounds.shape != (len(self.joint_names), 2) or not np.isfinite(bounds).all():
            raise ValueError("position limits must be finite [lower, upper] pairs")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("each position lower limit must be below its upper limit")
        self.position_lower = bounds[:, 0]
        self.position_upper = bounds[:, 1]
        self.position: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        self.acceleration: np.ndarray | None = None
        self.target: np.ndarray | None = None
        self._velocity_braking = np.zeros(len(self.joint_names), dtype=bool)

    def _positive_array(self, label: str, values: Mapping[str, float]) -> np.ndarray:
        missing = [name for name in self.joint_names if name not in values]
        if missing:
            raise ValueError(f"missing {label} limits for: {missing}")
        result = np.asarray([values[name] for name in self.joint_names], dtype=np.float64)
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError(f"{label} limits must be finite and positive")
        return result

    @property
    def initialized(self) -> bool:
        return self.position is not None

    def reset(self, measured_position: Sequence[float]) -> None:
        position = np.asarray(measured_position, dtype=np.float64)
        if position.shape != (len(self.joint_names),) or not np.isfinite(position).all():
            raise ValueError("initial measured position must contain one finite value per joint")
        self.position = np.clip(position, self.position_lower, self.position_upper)
        self.velocity = np.zeros_like(self.position)
        self.acceleration = np.zeros_like(self.position)
        self.target = self.position.copy()
        self._velocity_braking[:] = False

    def set_target(self, target_position: Sequence[float]) -> None:
        target = np.asarray(target_position, dtype=np.float64)
        if target.shape != (len(self.joint_names),) or not np.isfinite(target).all():
            raise ValueError("target position must contain one finite value per joint")
        self.target = np.clip(target, self.position_lower, self.position_upper)

    def hold_last_valid(self, error: str, dt: float | None = None) -> TrajectorySample:
        if not self.initialized:
            raise RuntimeError("trajectory has not been reset")
        zeros = np.zeros_like(self.position)
        held_acceleration = zeros.copy()
        held_jerk = zeros.copy()
        if dt is not None and np.isfinite(dt) and dt > 0:
            held_acceleration = -self.velocity / dt
            held_jerk = (held_acceleration - self.acceleration) / dt
        # Holding is a safety fallback. Reset derivatives because the next valid
        # tick must start from the command that was actually held.
        self.velocity = zeros.copy()
        self.acceleration = zeros.copy()
        return TrajectorySample(
            self.position.copy(), zeros, held_acceleration, held_jerk, False, error
        )

    @staticmethod
    def _stopping_distance(
        speed: float, acceleration: float, acceleration_limit: float, jerk_limit: float, dt: float
    ) -> float:
        """Continuous S-curve stopping distance in the direction of travel."""
        if speed <= 0:
            return 0.0
        del dt  # caller adds a discrete one-tick margin
        ramp_time = max(0.0, (acceleration + acceleration_limit) / jerk_limit)
        stop_during_ramp = (
            acceleration
            + np.sqrt(max(0.0, acceleration * acceleration + 2.0 * jerk_limit * speed))
        ) / jerk_limit
        if stop_during_ramp <= ramp_time:
            t = stop_during_ramp
            return max(
                0.0,
                speed * t
                + 0.5 * acceleration * t * t
                - jerk_limit * t * t * t / 6.0,
            )
        speed_after_ramp = (
            speed
            + acceleration * ramp_time
            - 0.5 * jerk_limit * ramp_time * ramp_time
        )
        distance_during_ramp = (
            speed * ramp_time
            + 0.5 * acceleration * ramp_time * ramp_time
            - jerk_limit * ramp_time * ramp_time * ramp_time / 6.0
        )
        return max(0.0, distance_during_ramp) + max(0.0, speed_after_ramp) ** 2 / (
            2.0 * acceleration_limit
        )

    def step(self, dt: float) -> TrajectorySample:
        if not self.initialized or self.target is None:
            raise RuntimeError("trajectory has not been reset")
        if not np.isfinite(dt) or dt <= 0:
            return self.hold_last_valid(f"invalid dt: {dt!r}")

        q = self.position
        v = self.velocity
        a = self.acceleration
        error = self.target - q

        # A conservative reachable-speed target makes the controller start
        # braking before the waypoint.  The d/dt cap also prevents crossing a
        # fixed target during ordinary monotonic motion.
        direction = np.sign(error)
        distance = np.abs(error)
        safe_v_acc = np.sqrt(2.0 * self.acceleration_limit * distance)
        # Reserve velocity headroom for an unexpectedly long next control
        # interval; the hard projection below still uses the configured limit.
        desired_speed = np.minimum.reduce(
            # The 250 ms convergence horizon prevents a finite-rate command
            # from racing all the way to a waypoint before jerk-limited
            # braking can take effect.
            [0.8 * self.velocity_limit, safe_v_acc, distance / max(0.25, dt)]
        )
        desired_v = direction * desired_speed
        desired_a = (desired_v - v) / dt
        for index in range(len(self.joint_names)):
            directed_speed = direction[index] * v[index]
            directed_acceleration = direction[index] * a[index]
            stopping_distance = self._stopping_distance(
                directed_speed,
                directed_acceleration,
                self.acceleration_limit[index],
                self.jerk_limit[index],
                dt,
            )
            acceleration_headroom = max(directed_acceleration, 0.0) ** 2 / (
                2.0 * self.jerk_limit[index]
            )
            approaching_velocity_limit = (
                directed_speed
                + acceleration_headroom
                + max(directed_acceleration, 0.0) * dt
                >= self.velocity_limit[index]
            )
            # One velocity tick of margin accounts for the command being held
            # until the next irregularly timed invocation.
            braking_for_position = (
                1.5 * stopping_distance + max(directed_speed, 0.0) * dt
                >= distance[index]
            )
            if approaching_velocity_limit:
                self._velocity_braking[index] = True
            if self._velocity_braking[index] and directed_acceleration <= 0:
                self._velocity_braking[index] = False
            if braking_for_position or self._velocity_braking[index]:
                desired_a[index] = -direction[index] * self.acceleration_limit[index]

        # Project the requested acceleration into the intersection induced by
        # jerk, acceleration, and velocity bounds.  Because prior samples were
        # generated by the same projection this interval remains feasible.
        a_low = np.maximum(-self.acceleration_limit, a - self.jerk_limit * dt)
        a_high = np.minimum(self.acceleration_limit, a + self.jerk_limit * dt)
        a_low = np.maximum(a_low, (-self.velocity_limit - v) / dt)
        a_high = np.minimum(a_high, (self.velocity_limit - v) / dt)
        next_a = np.clip(desired_a, a_low, a_high)
        next_v = v + next_a * dt

        proposed = q + next_v * dt
        crosses = (error > 0) & (proposed > self.target) | (error < 0) & (proposed < self.target)
        if np.any(crosses):
            # Prefer the exact non-crossing boundary when it is dynamically
            # feasible. A target reversal can make immediate non-crossing
            # physically impossible; in that case retain the bounded state.
            boundary_v = error / dt
            boundary_a = (boundary_v - v) / dt
            feasible = crosses & (boundary_a >= a_low) & (boundary_a <= a_high)
            next_a[feasible] = boundary_a[feasible]
            next_v[feasible] = boundary_v[feasible]

        next_q = np.clip(q + next_v * dt, self.position_lower, self.position_upper)
        actual_v = (next_q - q) / dt
        actual_a = (actual_v - v) / dt
        actual_j = (actual_a - a) / dt

        arrays = (next_q, actual_v, actual_a, actual_j)
        if not all(np.isfinite(value).all() for value in arrays):
            return self.hold_last_valid("non-finite trajectory result", dt)

        # Reconstructing jerk from positions near O(1 rad) divides floating
        # point round-off by dt^3. At 500 Hz that amplification is O(1e8), so
        # use a small absolute numerical tolerance while keeping the generated
        # state projected to the exact configured bounds above.
        tolerance = 1e-6
        if (
            np.any(np.abs(actual_v) > self.velocity_limit + tolerance)
            or np.any(np.abs(actual_a) > self.acceleration_limit + tolerance)
            or np.any(np.abs(actual_j) > self.jerk_limit + tolerance)
        ):
            return self.hold_last_valid("trajectory constraint projection failed", dt)

        self.position = next_q
        self.velocity = actual_v
        self.acceleration = actual_a
        return TrajectorySample(
            next_q.copy(), actual_v.copy(), actual_a.copy(), actual_j.copy()
        )


class GripperPostprocessor:
    """Keep grippers out of arm smoothing; optionally apply a scalar rate limit."""

    def __init__(self, joint_names: Sequence[str], mode: str, rate_limits: Mapping[str, float]):
        self.joint_names = tuple(joint_names)
        if mode not in {"passthrough", "rate_limited"}:
            raise ValueError("gripper mode must be 'passthrough' or 'rate_limited'")
        self.mode = mode
        self.rate_limits = dict(rate_limits)
        if mode == "rate_limited":
            missing = [name for name in self.joint_names if name not in rate_limits]
            if missing:
                raise ValueError(f"missing gripper rate limits for: {missing}")
            if any(not np.isfinite(rate_limits[name]) or rate_limits[name] <= 0 for name in self.joint_names):
                raise ValueError("gripper rate limits must be finite and positive")
        self.position: np.ndarray | None = None

    def reset(self, position: Sequence[float]) -> None:
        value = np.asarray(position, dtype=np.float64)
        if value.shape != (len(self.joint_names),) or not np.isfinite(value).all():
            raise ValueError("invalid gripper reset position")
        self.position = value.copy()

    def update(self, target: Sequence[float], dt: float) -> np.ndarray:
        value = np.asarray(target, dtype=np.float64)
        if value.shape != (len(self.joint_names),) or not np.isfinite(value).all():
            return self.position.copy() if self.position is not None else np.zeros(len(self.joint_names))
        if self.mode == "passthrough" or self.position is None:
            self.position = value.copy()
        else:
            rates = np.asarray([self.rate_limits[name] for name in self.joint_names])
            self.position += np.clip(value - self.position, -rates * dt, rates * dt)
        return self.position.copy()
