#!/usr/bin/env python3
"""Run the existing LeRobot client while relaying its camera frames to AI GUI v3."""

from __future__ import annotations

import json
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


class VoicePolicyControl:
    """Small file-backed control channel owned only by AI GUI v3."""

    def __init__(self, path: Path, default_task: str) -> None:
        self.path = path
        self.task = default_task
        self.paused = False
        self.revision = -1
        self.mtime_ns = -1
        self.initialized = False
        self.accepting_actions = False
        self.lock = threading.Lock()

    def refresh(self) -> tuple[str | None, bool, str]:
        try:
            stat = self.path.stat()
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            revision = int(payload.get("revision", -1))
            paused = bool(payload.get("paused", False))
            task = str(payload.get("instruction") or self.task)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            with self.lock:
                return None, self.paused, self.task

        with self.lock:
            if revision == self.revision:
                self.mtime_ns = stat.st_mtime_ns
                return None, self.paused, self.task

            was_initialized = self.initialized
            was_paused = self.paused
            old_task = self.task
            self.initialized = True
            self.revision = revision
            self.mtime_ns = stat.st_mtime_ns
            self.paused = paused
            self.task = task

            if not was_initialized:
                event = "initial_pause" if paused else "initial_run"
            elif paused and not was_paused:
                event = "pause"
            elif not paused and was_paused:
                event = "resume"
            elif task != old_task:
                event = "task"
            else:
                event = None

            if paused or event in {"resume", "task", "initial_pause"}:
                self.accepting_actions = False
            return event, self.paused, self.task

    def enable_actions(self) -> None:
        with self.lock:
            if not self.paused:
                self.accepting_actions = True

    def actions_enabled(self) -> bool:
        with self.lock:
            return self.initialized and not self.paused and self.accepting_actions


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
        print(f"[AI GUI V3] Relaying policy cameras to browser: {', '.join(self.keys) or 'none'}", flush=True)
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
        print("[AI GUI V3] RBY1_WEB_RELAY_DIR is missing; continuing without preview relay.", flush=True)
        return

    from lerobot_async_inference import robot_client as client_module

    original_init = client_module.RobotClient.__init__
    original_stop = client_module.RobotClient.stop
    original_aggregate = client_module.RobotClient._aggregate_action_queues
    relay_directory = Path(relay_value)
    control_value = os.environ.get("RBY1_AI_CONTROL_FILE")
    relay_fps = int(os.environ.get("RBY1_WEB_RELAY_FPS", "15"))
    relay_quality = int(os.environ.get("RBY1_WEB_RELAY_QUALITY", "75"))

    def clear_actions(self: Any, *, advance_timestep: bool) -> None:
        with self.action_queue_lock:
            self.action_queue.queue.clear()
        if advance_timestep:
            configured_chunk = int(self.config.actions_per_chunk or 0)
            known_chunk = max(int(self.action_chunk_size), 0)
            step = max(configured_chunk, known_chunk, 1) + 10
            with self.latest_action_lock:
                self.latest_action = max(int(self.latest_action), 0) + step
        self.must_go.set()

    def hold_current_pose(self: Any) -> None:
        try:
            observation = self.robot.get_observation()
            hold_action = {
                key: float(observation[key])
                for key in self.robot.action_features
                if key in observation
            }
            if len(hold_action) != len(self.robot.action_features):
                missing = sorted(set(self.robot.action_features).difference(hold_action))
                self.logger.warning(f"AI pause could not build complete hold action; missing: {missing}")
                return
            self.robot.send_action(hold_action)
            self.logger.info("AI policy paused; current robot pose is being held.")
        except Exception as exc:
            self.logger.error(f"AI pause hold command failed: {exc}")

    def patched_init(self: Any, config: Any) -> None:
        original_init(self, config)
        self._rby1_web_frame_relay = None
        self._rby1_ai_voice_control = (
            VoicePolicyControl(Path(control_value), config.task) if control_value else None
        )
        try:
            relay = PolicyFrameRelay(self.robot, relay_directory, relay_fps, relay_quality)
            self._rby1_web_frame_relay = relay
            relay.start()
        except Exception as exc:
            print(f"[AI GUI V3] Preview relay unavailable: {exc}", file=sys.stderr, flush=True)

    def patched_stop(self: Any) -> None:
        relay = getattr(self, "_rby1_web_frame_relay", None)
        try:
            if relay is not None:
                relay.stop()
                self._rby1_web_frame_relay = None
        finally:
            original_stop(self)

    def patched_aggregate(self: Any, incoming_actions: Any, aggregate_fn: Any = None) -> None:
        control = getattr(self, "_rby1_ai_voice_control", None)
        if control is not None and not control.actions_enabled():
            clear_actions(self, advance_timestep=False)
            return
        original_aggregate(self, incoming_actions, aggregate_fn)

    def patched_control_loop(self: Any, task: str, verbose: bool = False) -> tuple[Any, Any]:
        if self.uses_grpc_backend or self.uses_remote_zmq_backend:
            self.start_barrier.wait()

        self.logger.info("Control loop thread starting")
        performed_action = None
        captured_observation = None
        control = getattr(self, "_rby1_ai_voice_control", None)
        current_task = task

        while self.running:
            loop_started = time.perf_counter()
            paused = False
            if control is not None:
                event, paused, current_task = control.refresh()
                if event in {"pause", "initial_pause"}:
                    clear_actions(self, advance_timestep=True)
                    hold_current_pose(self)
                elif event in {"resume", "task"}:
                    clear_actions(self, advance_timestep=True)
                    self.logger.info(f"AI policy resumed with task: {current_task}")
                if event in {"resume", "task", "initial_run"}:
                    control.enable_actions()

            if paused:
                clear_actions(self, advance_timestep=False)
                time.sleep(min(0.1, self.config.environment_dt))
                continue

            if self.actions_available():
                performed_action = self.control_loop_action(verbose)

            if self._ready_to_send_observation():
                if self.uses_grpc_backend:
                    captured_observation = self.control_loop_observation(current_task, verbose)
                elif self.uses_remote_zmq_backend:
                    captured_observation = self.control_loop_remote_observation(current_task, verbose)
                else:
                    raise ValueError(f"Unsupported backend: {self.backend}")

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - loop_started) * 1000:.2f}")
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - loop_started)))

        return captured_observation, performed_action

    client_module.RobotClient.__init__ = patched_init
    client_module.RobotClient.stop = patched_stop
    client_module.RobotClient._aggregate_action_queues = patched_aggregate
    client_module.RobotClient.control_loop = patched_control_loop


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
    print(f"[AI GUI V3] Task override: {task}", flush=True)


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
        print("\n[AI GUI V3] Robot client interrupted.", file=sys.stderr)
        raise
