#!/usr/bin/env python3
"""Run the existing LeRobot client while relaying its camera frames to Web v2."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2

try:
    from .frame_relay import RelayWriter
except ImportError:
    from frame_relay import RelayWriter


class PolicyFrameRelay:
    def __init__(self, robot: Any, relay_directory: Path, fps: int, quality: int) -> None:
        self.robot = robot
        self.fps = max(1, fps)
        self.quality = quality
        self.stop_event = threading.Event()
        available_keys = [key for key in robot.cameras if (relay_directory / f"{key}.frame").is_file()]
        self.writer = RelayWriter(relay_directory, available_keys)
        self.keys = available_keys
        self.thread = threading.Thread(
            target=self._run,
            name="rby1-web-v2-frame-relay",
            daemon=True,
        )

    def start(self) -> None:
        print(f"[WEB V2] Relaying policy cameras to browser: {', '.join(self.keys) or 'none'}", flush=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.writer.close()

    def _run(self) -> None:
        interval = 1.0 / self.fps
        while not self.stop_event.is_set():
            started = time.perf_counter()
            for key in self.keys:
                camera = self.robot.cameras.get(key)
                if camera is None:
                    continue
                try:
                    # read_latest() peeks at the RealSense background buffer and
                    # does not clear the event consumed by policy async_read().
                    rgb_frame = camera.read_latest(max_age_ms=1000)
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        bgr_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.quality],
                    )
                    if ok:
                        self.writer.write(key, encoded.tobytes())
                except Exception:
                    # Preview failures must never interrupt policy observation
                    # capture or robot control.
                    continue
            self.stop_event.wait(max(0.0, interval - (time.perf_counter() - started)))


def install_relay_patch() -> None:
    relay_value = os.environ.get("RBY1_WEB_RELAY_DIR")
    if not relay_value:
        print("[WEB V2] RBY1_WEB_RELAY_DIR is missing; continuing without preview relay.", flush=True)
        return

    from lerobot_async_inference import robot_client as client_module

    original_init = client_module.RobotClient.__init__
    original_stop = client_module.RobotClient.stop
    relay_directory = Path(relay_value)
    relay_fps = int(os.environ.get("RBY1_WEB_RELAY_FPS", "15"))
    relay_quality = int(os.environ.get("RBY1_WEB_RELAY_QUALITY", "75"))

    def patched_init(self: Any, config: Any) -> None:
        original_init(self, config)
        self._rby1_web_frame_relay = None
        try:
            relay = PolicyFrameRelay(self.robot, relay_directory, relay_fps, relay_quality)
            self._rby1_web_frame_relay = relay
            relay.start()
        except Exception as exc:
            print(f"[WEB V2] Preview relay unavailable: {exc}", file=sys.stderr, flush=True)

    def patched_stop(self: Any) -> None:
        relay = getattr(self, "_rby1_web_frame_relay", None)
        try:
            if relay is not None:
                relay.stop()
                self._rby1_web_frame_relay = None
        finally:
            original_stop(self)

    client_module.RobotClient.__init__ = patched_init
    client_module.RobotClient.stop = patched_stop


def apply_task_override() -> None:
    task = os.environ.get("RBY1_WEB_TASK")
    if not task:
        return
    for index, argument in enumerate(sys.argv):
        if argument.startswith("--task="):
            sys.argv[index] = f"--task={task}"
            break
        if argument == "--task" and index + 1 < len(sys.argv):
            sys.argv[index + 1] = task
            break
    else:
        sys.argv.append(f"--task={task}")
    print(f"[WEB V2] Task override: {task}", flush=True)


def main() -> int:
    apply_task_override()
    install_relay_patch()
    from lerobot_async_inference.robot_client import async_client

    async_client()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[WEB V2] Robot client interrupted.", file=sys.stderr)
        raise
