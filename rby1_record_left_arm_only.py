#!/usr/bin/env python3
"""Private launcher for rby1_record_left_arm_only.sh."""

import sys
import time

# Importing this module registers the two private device types before LeRobot
# parses command-line arguments.  No normal lerobot-record invocation sees them.
import rby1_left_arm_only_runtime  # noqa: F401

from lerobot.scripts.lerobot_record import main


_REQUIRED_CAMERA_SERIALS = {
    "front": "260322274450",
    "left": "260322274992",
    "right": "260322276006",
}


def require_all_cameras(timeout_s: int = 15) -> None:
    """Fail before robot motion when any required RealSense is absent."""
    import pyrealsense2 as rs

    deadline = time.monotonic() + timeout_s
    detected: set[str] = set()
    while time.monotonic() < deadline:
        context = rs.context()
        detected = {
            device.get_info(rs.camera_info.serial_number)
            for device in context.query_devices()
        }
        if set(_REQUIRED_CAMERA_SERIALS.values()) <= detected:
            print(f"Camera preflight OK: {sorted(detected)}", flush=True)
            return
        time.sleep(1.0)

    missing = {
        name: serial
        for name, serial in _REQUIRED_CAMERA_SERIALS.items()
        if serial not in detected
    }
    print(
        f"Error: camera preflight failed; missing RealSense camera(s): {missing}. "
        "Reconnect USB/power before retrying. The robot was not started.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    require_all_cameras()
    main()
