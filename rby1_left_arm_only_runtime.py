"""Private, left-leader-only runtime used by rby1_record_left_arm_only.sh.

This module is deliberately not imported by either RB-Y1 plugin package.  Its
two temporary LeRobot device types are registered only when the dedicated
recording launcher imports this file.

Follower robot: both arms move to the ready pose on connect, then only the
left arm receives per-frame commands.  Leader hardware: only the left arm
moves to its ready pose; the right leader remains untouched at the exact pose
where it was when this program started.
"""

from __future__ import annotations

import importlib.resources
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot_robot_rby1.config_rby1 import Rby1Config
from lerobot_robot_rby1.rby1 import Rby1
from lerobot_teleoperator_rby1.config_rby1_leader_arm import Rby1LeaderArmConfig
from lerobot_teleoperator_rby1.constants import (
    LEFT_ARM_NAMES,
    LEFT_ARM_Q_MAX,
    LEFT_ARM_Q_MIN,
    RIGHT_ARM_NAMES,
)

logger = logging.getLogger(__name__)

# Physical LEFT leader arm. The damaged physical RIGHT arm is IDs 0..6 with
# tool 0x80 and must never be addressed by this runtime.
_LEFT_MOTOR_IDS = list(range(7, 14))
_LEFT_TOOL_ID = 0x81
_TOOL_HOLD_FAILSAFE_AFTER = 100  # 1 second at the default 100 Hz loop
_FALLBACK_URDF_PATH = "/home/nvidia/rby1-sdk/models/leader_arm/model.urdf"
_ACTIVE_FOLLOWER: "Rby1BothDataLeftControl | None" = None


@RobotConfig.register_subclass("rby1_both_data_left_control")
@dataclass
class Rby1BothDataLeftControlConfig(Rby1Config):
    """Expose both arms in data while sending continuous commands only to the left."""


class Rby1BothDataLeftControl(Rby1):
    """RB-Y1 follower that records both arms but leaves the right arm uncommanded."""

    config_class = Rby1BothDataLeftControlConfig
    # Keep the original robot_type so this temporary controller can append to
    # the existing rby1 dataset without rewriting its metadata.
    name = "rby1"

    def __init__(self, config: Rby1BothDataLeftControlConfig) -> None:
        super().__init__(config)
        self._right_data_lock = threading.Lock()
        self._right_hold_action: dict[str, float] = {}
        self._camera_frame_cache: dict[str, np.ndarray] = {}
        self._camera_misses: dict[str, int] = {}

    def connect(self, calibrate: bool = True) -> None:
        global _ACTIVE_FOLLOWER
        try:
            super().connect(calibrate)
        except BaseException:  # noqa: BLE001
            self._cleanup_failed_connect()
            raise
        _ACTIVE_FOLLOWER = self

    def _cleanup_failed_connect(self) -> None:
        """Release resources acquired before Rby1.connect() raised."""
        if self._gripper is not None:
            try:
                self._gripper.disconnect()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to disconnect gripper after partial connect.")
            self._gripper = None
        for cam in self.cameras.values():
            if getattr(cam, "is_connected", False):
                try:
                    cam.disconnect()
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to disconnect camera after partial connect.")
        if self._robot is not None:
            try:
                self._robot.cancel_control()
                self._robot.disable_control_manager()
                if self._config.use_gripper:
                    self._robot.set_tool_flange_output_voltage("right", 0)
                    self._robot.set_tool_flange_output_voltage("left", 0)
                self._robot.disconnect()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to disconnect RB-Y1 after partial connect.")
            self._robot = None
        self._model = None
        self._stream = None
        self._is_connected = False
        logger.info("Partial RB-Y1 connection cleaned up safely.")

    def disconnect(self) -> None:
        global _ACTIVE_FOLLOWER
        if _ACTIVE_FOLLOWER is self:
            _ACTIVE_FOLLOWER = None
        super().disconnect()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        state = self._robot.get_state()
        model = self._model
        observation: dict[str, Any] = {}
        self._read_group(observation, state.position, model, "")
        if self._config.use_velocity:
            self._read_group(observation, state.velocity, model, ".vel")
        if self._config.use_torque:
            self._read_group(observation, state.torque, model, ".torque")
        self._read_ee_observation(observation, state)

        if self._config.use_gripper and self._gripper is not None:
            gripper_pos = self._gripper.get_positions()
            if self._config.use_right_arm:
                observation["right_gripper_0"] = 1.0 - float(gripper_pos[0])
            if self._config.use_left_arm:
                observation["left_gripper_0"] = float(gripper_pos[1])

        for cam_key, cam in self.cameras.items():
            try:
                frame = cam.async_read()
            except TimeoutError:
                misses = self._camera_misses.get(cam_key, 0) + 1
                self._camera_misses[cam_key] = misses
                cached = self._camera_frame_cache.get(cam_key)
                if cached is None or misses > 15:
                    raise
                if misses in (1, 5, 15):
                    logger.warning(
                        "Camera %s timed out (%d/15); reusing its last valid frame.",
                        cam_key,
                        misses,
                    )
                frame = cached.copy()
            else:
                misses = self._camera_misses.get(cam_key, 0)
                if misses:
                    logger.info("Camera %s recovered after %d missed frame(s).", cam_key, misses)
                self._camera_misses[cam_key] = 0
                self._camera_frame_cache[cam_key] = frame.copy()
            observation[cam_key] = frame

        # Cache the measured, stationary right-side state.  The teleoperator
        # writes these exact values into the action columns, while send_action
        # below deliberately omits that side from the physical command.
        right_hold = {
            key: float(value)
            for key, value in observation.items()
            if key.startswith("right_arm_") or key == "right_gripper_0"
        }
        with self._right_data_lock:
            self._right_hold_action = right_hold
        return observation

    def get_right_hold_action(self) -> dict[str, float]:
        with self._right_data_lock:
            return self._right_hold_action.copy()

    def _send_joint_action(self, rby: Any, action: dict[str, Any]) -> None:
        # Keep use_right_arm=True everywhere else so observations and dataset
        # features contain both arms.  Omit it only while building the command.
        original = self._config.use_right_arm
        self._config.use_right_arm = False
        try:
            super()._send_joint_action(rby, action)
        finally:
            self._config.use_right_arm = original

    def _send_gripper_action(self, action: dict[str, Any]) -> None:
        # The right gripper is recorded but is not sent a new target.
        original = self._config.use_right_arm
        self._config.use_right_arm = False
        try:
            super()._send_gripper_action(action)
        finally:
            self._config.use_right_arm = original


@TeleoperatorConfig.register_subclass("rby1_left_leader_only_private")
@dataclass
class Rby1LeftLeaderOnlyConfig(Rby1LeaderArmConfig):
    """Configuration for the private single-left-leader implementation."""

    # Dataset value for the uncommanded right gripper (1.0 means open).
    right_gripper_hold: float = 1.0


class Rby1LeftLeaderOnly(Teleoperator):
    """Read/control only physical-left Dynamixel IDs 7..13 and tool ID 0x81.

    The damaged right leader motors (IDs 0..6) and right tool (0x80) are never pinged,
    read, torque-enabled, mode-switched, or commanded by this class. Therefore
    the right leader receives no ready-pose trajectory and stays at its startup
    position; only the left leader moves to its configured ready pose.
    """

    config_class = Rby1LeftLeaderOnlyConfig
    name = "rby1_left_leader_only_private"

    def __init__(self, config: Rby1LeftLeaderOnlyConfig) -> None:
        super().__init__(config)
        self._config = config
        self._is_connected = False
        self._bus: Any = None
        self._rby: Any = None
        self._dyn_robot: Any = None
        self._dyn_state: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread_error: BaseException | None = None

        self._left_q = np.deg2rad(config.left_init_q_deg).astype(np.float64)
        self._left_trigger = 0.0
        self._hold_q = self._left_q.copy()
        self._mode: int | None = None
        self._init_start_q = self._left_q.copy()
        self._init_t0 = 0.0
        self._init_done = False
        self._reset_requested = False
        self._tool_read_failures = 0
        self._last_button = 0

        self._torque_constant = np.array(
            [1.6591, 1.6591, 1.6591, 1.3043, 1.3043, 1.3043, 1.3043] * 2,
            dtype=np.float64,
        )

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def action_features(self) -> dict[str, type]:
        # Both sides stay in the dataset schema.  Only left values come from
        # physical leader hardware; right values are the fixed ready pose.
        features = {name: float for name in RIGHT_ARM_NAMES + LEFT_ARM_NAMES}
        if self._config.use_gripper:
            features["right_gripper_0"] = float
            features["left_gripper_0"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")

        import rby1_sdk as rby

        self._rby = rby
        # Resolve once and use the same concrete path for both latency setup
        # and DynamixelBus. Older installations expose /dev/rby1_master_arm;
        # passing the unresolved new name to DynamixelBus cannot open it.
        device_name = rby.upc.resolve_leader_arm_device_name()
        logger.info("Opening single-left leader bus at %s", device_name)
        rby.upc.initialize_device(device_name)
        self._bus = rby.DynamixelBus(device_name)
        if not self._bus.open_port():
            raise ConnectionError("Could not open the left leader-arm serial port.")
        if not self._bus.set_baud_rate(self._bus.DefaultBaudrate):
            raise ConnectionError("Could not set the left leader-arm baud rate.")

        required_ids = _LEFT_MOTOR_IDS + [_LEFT_TOOL_ID]
        missing = [device_id for device_id in required_ids if not self._bus.ping(device_id)]
        if missing:
            raise RuntimeError(f"Left leader-arm devices not responding: {missing}")

        self._bus.set_torque_constant(self._torque_constant.tolist())
        self._init_dynamics(self._resolve_urdf_path())

        initial = self._read_left_state()
        self._left_q = initial[0]
        self._hold_q = self._left_q.copy()
        self._begin_init(self._left_q)
        self._set_left_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)

        self._stop.clear()
        self._thread_error = None
        self._thread = threading.Thread(
            target=self._control_loop,
            name="rby1-left-leader-only",
            daemon=True,
        )
        self._is_connected = True
        self._thread.start()
        logger.info(
            "Single-left leader started: LEFT is moving to ready pose using IDs "
            "7..13 and 0x81; RIGHT IDs 0..6 and 0x80 remain untouched at their "
            "startup position."
        )
        deadline = time.monotonic() + self._config.startup_wait_time
        while time.monotonic() < deadline:
            if self._thread_error is not None:
                error = self._thread_error
                self.disconnect()
                raise RuntimeError("Single-left leader control loop failed.") from error
            time.sleep(0.05)

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._bus is not None and self._rby is not None:
            try:
                self._bus.group_sync_write_torque_enable(
                    _LEFT_MOTOR_IDS, self._rby.DynamixelBus.TorqueDisable
                )
                logger.info("Single-left leader torque disabled for IDs %s.", _LEFT_MOTOR_IDS)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to disable single-left leader torque: %s", exc)
        self._bus = None
        self._is_connected = False
        logger.info("Single-left leader disconnected; right leader was never addressed.")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def request_reset(self) -> None:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._lock:
            self._reset_requested = True
            self._init_done = False

    def is_reset_complete(self) -> bool:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._raise_thread_error()
        with self._lock:
            return self._init_done and not self._reset_requested

    def reset(self) -> None:
        self.request_reset()
        deadline = time.time() + max(self._config.init_duration + 2.0, 2.0)
        while time.time() < deadline:
            if self.is_reset_complete():
                return
            time.sleep(0.05)
        logger.warning("Single-left leader reset timed out; continuing.")

    def get_action(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._raise_thread_error()
        with self._lock:
            left_q = self._left_q.copy()
            left_trigger = self._left_trigger

        left_q[6] += np.deg2rad(self._config.left_wrist_offset_deg)
        left_q = np.clip(left_q, LEFT_ARM_Q_MIN, LEFT_ARM_Q_MAX)

        right_hold = (
            _ACTIVE_FOLLOWER.get_right_hold_action()
            if _ACTIVE_FOLLOWER is not None
            else {}
        )
        missing_right = [name for name in RIGHT_ARM_NAMES if name not in right_hold]
        if missing_right:
            raise RuntimeError(
                "Follower has not supplied the measured right-arm hold state: "
                f"{missing_right}"
            )

        action = {**right_hold}
        action.update({name: float(left_q[i]) for i, name in enumerate(LEFT_ARM_NAMES)})
        if self._config.use_gripper:
            action.setdefault("right_gripper_0", float(self._config.right_gripper_hold))
            action["left_gripper_0"] = float(
                np.clip(left_trigger / self._config.gripper_trigger_max, 0.0, 1.0)
            )
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def _control_loop(self) -> None:
        period = 1.0 / self._config.control_frequency
        try:
            while not self._stop.is_set():
                tick = time.monotonic()
                q, qvel, trigger, button = self._read_left_state()
                with self._lock:
                    reset = self._reset_requested
                    if reset:
                        self._reset_requested = False
                if reset:
                    self._begin_init(q)

                elapsed = time.monotonic() - self._init_t0
                if elapsed < self._config.init_duration:
                    ratio = elapsed / max(self._config.init_duration, 1e-6)
                    smooth = 0.5 * (1.0 - np.cos(np.pi * ratio))
                    target = self._init_start_q + smooth * (
                        np.deg2rad(self._config.left_init_q_deg) - self._init_start_q
                    )
                    self._set_left_mode(self._rby.DynamixelBus.CurrentBasedPositionControlMode)
                    self._write_left_position(target)
                    output_q = target
                    self._hold_q = target.copy()
                else:
                    with self._lock:
                        self._init_done = True
                    if button == 1:
                        self._set_left_mode(self._rby.DynamixelBus.CurrentControlMode)
                        torque = self._left_gravity(q)
                        min_q = np.deg2rad(self._config.min_q_deg[7:14])
                        max_q = np.deg2rad(self._config.max_q_deg[7:14])
                        torque += self._config.joint_limit_barrier * (
                            np.maximum(min_q - q, 0.0) + np.minimum(max_q - q, 0.0)
                        )
                        torque += np.asarray(self._config.viscous_gain[7:14]) * qvel
                        limit = np.asarray(self._config.torque_limit[7:14])
                        torque = np.clip(torque, -limit, limit)
                        self._write_left_torque(torque * self._config.gravity_comp_scale)
                        self._hold_q = q.copy()
                        output_q = q
                    else:
                        self._set_left_mode(
                            self._rby.DynamixelBus.CurrentBasedPositionControlMode
                        )
                        self._write_left_position(self._hold_q)
                        output_q = self._hold_q

                with self._lock:
                    self._left_q = output_q.copy()
                    self._left_trigger = trigger

                self._stop.wait(max(0.0, period - (time.monotonic() - tick)))
        except BaseException as exc:  # noqa: BLE001
            self._thread_error = exc
            self._stop.set()
            logger.exception("Single-left leader control loop stopped.")
            try:
                self._bus.group_sync_write_torque_enable(
                    _LEFT_MOTOR_IDS, self._rby.DynamixelBus.TorqueDisable
                )
            except Exception:  # noqa: BLE001
                logger.exception("Emergency left-leader torque disable failed.")

    def _read_left_state(self) -> tuple[np.ndarray, np.ndarray, float, int]:
        states = self._bus.get_motor_states(_LEFT_MOTOR_IDS)
        if not states or len(states) != 7:
            raise RuntimeError("Failed to read all seven left leader motors.")
        by_id = {motor_id: state for motor_id, state in states}
        if any(motor_id not in by_id for motor_id in _LEFT_MOTOR_IDS):
            raise RuntimeError("Incomplete left leader motor state response.")
        q = np.array([by_id[i].position for i in _LEFT_MOTOR_IDS], dtype=np.float64)
        qvel = np.array([by_id[i].velocity for i in _LEFT_MOTOR_IDS], dtype=np.float64)
        tool = self._bus.read_button_status(_LEFT_TOOL_ID)
        if tool is None:
            self._tool_read_failures += 1
            if (
                self._tool_read_failures in (25, _TOOL_HOLD_FAILSAFE_AFTER)
                or self._tool_read_failures % 500 == 0
            ):
                logger.warning(
                    "Left leader button/trigger read missed (%d); using a safe fallback.",
                    self._tool_read_failures,
                )
            with self._lock:
                trigger = self._left_trigger
            # Short dropout: preserve the last button state to avoid a mode
            # twitch. Persistent dropout: fail safe to button released (hold
            # the current left-arm pose) without killing the motor loop.
            button = (
                self._last_button
                if self._tool_read_failures <= _TOOL_HOLD_FAILSAFE_AFTER
                else 0
            )
            return q, qvel, trigger, button
        if self._tool_read_failures:
            if self._tool_read_failures >= 25:
                logger.info(
                    "Left leader button/trigger communication recovered after %d misses.",
                    self._tool_read_failures,
                )
        self._tool_read_failures = 0
        button_state = tool[1]
        self._last_button = int(button_state.button)
        return q, qvel, float(button_state.trigger), self._last_button

    def _set_left_mode(self, mode: int) -> None:
        if self._mode == mode:
            return
        self._bus.group_sync_write_torque_enable(
            _LEFT_MOTOR_IDS, self._rby.DynamixelBus.TorqueDisable
        )
        self._bus.group_sync_write_operating_mode([(i, mode) for i in _LEFT_MOTOR_IDS])
        self._bus.group_sync_write_torque_enable(
            _LEFT_MOTOR_IDS, self._rby.DynamixelBus.TorqueEnable
        )
        self._mode = mode

    def _write_left_position(self, target: np.ndarray) -> None:
        limit = np.asarray(self._config.torque_limit[7:14], dtype=np.float64)
        self._bus.group_sync_write_send_torque(
            [(i, float(limit[j])) for j, i in enumerate(_LEFT_MOTOR_IDS)]
        )
        self._bus.group_sync_write_send_position(
            [(i, float(target[j])) for j, i in enumerate(_LEFT_MOTOR_IDS)]
        )

    def _write_left_torque(self, torque: np.ndarray) -> None:
        self._bus.group_sync_write_send_torque(
            [(i, float(torque[j])) for j, i in enumerate(_LEFT_MOTOR_IDS)]
        )

    def _begin_init(self, current_q: np.ndarray) -> None:
        self._init_start_q = current_q.copy()
        self._init_t0 = time.monotonic()
        with self._lock:
            self._init_done = False

    def _left_gravity(self, left_q: np.ndarray) -> np.ndarray:
        # The SDK's leader URDF joint order is left then right.  The unused
        # right side is fixed at its ready pose solely for the dynamics model.
        right_q = np.deg2rad(self._config.right_init_q_deg)
        self._dyn_state.set_q(np.concatenate([left_q, right_q]))
        self._dyn_robot.compute_forward_kinematics(self._dyn_state)
        gravity = np.asarray(self._dyn_robot.compute_gravity_term(self._dyn_state))
        return gravity[:7] * 0.5

    def _init_dynamics(self, model_path: str) -> None:
        model = self._rby.dynamics.load_robot_from_urdf(model_path, "Base")
        self._dyn_robot = self._rby.dynamics.Robot(model)
        self._dyn_state = self._dyn_robot.make_state(
            [
                "Base", "Link_0R", "Link_1R", "Link_2R", "Link_3R",
                "Link_4R", "Link_5R", "Link_6R", "Link_0L", "Link_1L",
                "Link_2L", "Link_3L", "Link_4L", "Link_5L", "Link_6L",
            ],
            [
                "J0_Shoulder_Pitch_R", "J1_Shoulder_Roll_R", "J2_Shoulder_Yaw_R",
                "J3_Elbow_R", "J4_Wrist_Yaw1_R", "J5_Wrist_Pitch_R",
                "J6_Wrist_Yaw2_R", "J7_Shoulder_Pitch_L", "J8_Shoulder_Roll_L",
                "J9_Shoulder_Yaw_L", "J10_Elbow_L", "J11_Wrist_Yaw1_L",
                "J12_Wrist_Pitch_L", "J13_Wrist_Yaw2_L",
            ],
        )
        self._dyn_state.set_gravity([0, 0, 0, 0, 0, -9.81])

    def _resolve_urdf_path(self) -> str:
        if self._config.leader_arm_model_path:
            return self._config.leader_arm_model_path
        sdk_path = str(importlib.resources.files("rby1_sdk"))
        candidate = os.path.abspath(os.path.join(sdk_path, "..", "models", "leader_arm", "model.urdf"))
        if os.path.isfile(candidate):
            return candidate
        if os.path.isfile(_FALLBACK_URDF_PATH):
            return _FALLBACK_URDF_PATH
        raise FileNotFoundError("Could not find the leader-arm URDF.")

    def _raise_thread_error(self) -> None:
        if self._thread_error is not None:
            raise RuntimeError("Single-left leader control loop failed.") from self._thread_error
