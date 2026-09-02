#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Override these when using a different robot or a slower trajectory:
#   RBY1_ADDRESS=192.168.30.1:50051 READY_POSE_DURATION=8 ./rby1_ready_pose.sh
ADDRESS="${RBY1_ADDRESS:-192.168.1.201:50051}"
MODEL="${RBY1_MODEL:-auto}"
VERSION="${RBY1_VERSION:-auto}"
MINIMUM_TIME="${READY_POSE_DURATION:-5.0}"
HOLD_TIME="${READY_POSE_HOLD_TIME:-1.0}"

if [[ -n "${RBY1_PYTHON:-}" ]]; then
  PYTHON_BIN="${RBY1_PYTHON}"
elif [[ -x /home/nvidia/miniforge3/envs/lerobot/bin/python ]]; then
  PYTHON_BIN=/home/nvidia/miniforge3/envs/lerobot/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Error: Python was not found. Activate the lerobot conda environment first." >&2
  exit 1
fi

echo "WARNING: The robot will move to its ready pose."
echo "  address : ${ADDRESS}"
echo "  duration: ${MINIMUM_TIME} s"
echo "Check the robot's surroundings. Press Ctrl+C within 3 seconds to cancel."
sleep 3

export PYTHONPATH="${SCRIPT_DIR}/lerobot-robot-rby1${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" - "${ADDRESS}" "${MODEL}" "${VERSION}" "${MINIMUM_TIME}" "${HOLD_TIME}" <<'PY'
import logging
import sys
import time

import rby1_sdk as rby

from lerobot_robot_rby1.command_builders import build_ready_pose_command
from lerobot_robot_rby1.config_rby1 import ready_pose_for_version
from lerobot_robot_rby1.model_probe import resolve_model_version


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("rby1-ready-pose")

address, model_arg, version_arg = sys.argv[1:4]
minimum_time = float(sys.argv[4])
hold_time = float(sys.argv[5])
if minimum_time <= 0 or hold_time < 0:
    raise ValueError("READY_POSE_DURATION must be > 0 and READY_POSE_HOLD_TIME must be >= 0")

model, version = resolve_model_version(model_arg, version_arg, address)
if model == "ub":
    raise RuntimeError("The current ready-pose definition supports RB-Y1 A/M models only.")

body_position, _, _, head_position = ready_pose_for_version(version)
robot = rby.create_robot(address, model)
connected = False
control_enabled = False
motion_finished = False

try:
    log.info("Connecting to RB-Y1 at %s (model=%s, version=%s)", address, model, version)
    if not robot.connect():
        raise ConnectionError(f"Failed to connect to RB-Y1 at {address}")
    connected = True

    if not robot.is_power_on(".*") and not robot.power_on(".*"):
        raise RuntimeError("Failed to power on RB-Y1 actuators")
    if not robot.is_servo_on(".*") and not robot.servo_on(".*"):
        raise RuntimeError("Failed to turn on RB-Y1 servos")

    manager_state = robot.get_control_manager_state()
    if manager_state.state in (
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ):
        log.warning("Clearing the control-manager fault")
        if not robot.reset_fault_control_manager():
            raise RuntimeError("Failed to clear the control-manager fault")

    if not robot.enable_control_manager():
        raise RuntimeError("Failed to enable the control manager")
    control_enabled = True

    control_state = robot.get_control_manager_state().control_state
    if control_state != rby.ControlManagerState.ControlState.Idle:
        log.warning("Cancelling the active control command before ready-pose motion")
        robot.cancel_control()
        time.sleep(1.0)

    if not robot.wait_for_control_ready(1000):
        raise RuntimeError("Control manager did not become ready within 1 second")

    robot.set_parameter("joint_position_command.cutoff_frequency", "5")
    command = build_ready_pose_command(
        rby,
        body_position,
        head_position,
        minimum_time,
        hold_time,
    )
    log.info("Moving to ready pose over at least %.1f seconds", minimum_time)
    feedback = robot.send_command(
        rby.RobotCommandBuilder().set_command(command),
        1,
    ).get()
    if feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError(f"Ready-pose command failed: {feedback.finish_code}")

    motion_finished = True
    log.info("Ready pose reached")
finally:
    if connected and not motion_finished:
        try:
            robot.cancel_control()
        except Exception:
            pass
    if connected and control_enabled:
        try:
            robot.disable_control_manager()
        except Exception as exc:
            log.warning("Failed to disable the control manager: %s", exc)
    if connected:
        try:
            robot.disconnect()
        except Exception as exc:
            log.warning("Failed to disconnect cleanly: %s", exc)
PY
