#!/usr/bin/env python3
"""RB-Y1 browser GUI with three RealSense previews and client controls."""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

try:
    from .frame_relay import RelayStore
except ImportError:
    from frame_relay import RelayStore

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
CONFIG_PATH = APP_DIR / "config.json"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    required = {"robot_client_script", "host", "port", "cameras"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"config.json missing key(s): {', '.join(sorted(missing))}")
    if len(config["cameras"]) != 3:
        raise ValueError("config.json must contain exactly three cameras")

    keys = [item.get("key") for item in config["cameras"]]
    serials = [item.get("serial") for item in config["cameras"]]
    if any(not value for value in keys + serials):
        raise ValueError("Every camera needs a key and serial")
    if len(set(keys)) != 3 or len(set(serials)) != 3:
        raise ValueError("Camera keys and serials must be unique")
    policy_keys = config.get("policy_camera_keys", [])
    if not policy_keys or not set(policy_keys).issubset(keys):
        raise ValueError("policy_camera_keys must be a non-empty subset of camera keys")
    tasks = config.get("tasks", [])
    task_ids = [task.get("id") for task in tasks]
    if not tasks or any(not task_id for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("tasks must contain unique, non-empty ids")
    if any(not task.get("instruction") or not task.get("label") for task in tasks):
        raise ValueError("Every task needs a label and instruction")
    if config.get("default_task_id") not in task_ids:
        raise ValueError("default_task_id must match one of the configured tasks")
    return config


def resolve_script(config: dict[str, Any]) -> Path:
    path = Path(config["robot_client_script"])
    return (APP_DIR / path).resolve() if not path.is_absolute() else path


def validate(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    target = resolve_script(config)
    if not target.is_file():
        errors.append(f"Robot client script not found: {target}")
    for filename in ("index.html", "app.css", "app.js"):
        if not (STATIC_DIR / filename).is_file():
            errors.append(f"Static file not found: {STATIC_DIR / filename}")
    for path in (APP_DIR / "live_robot_client.py", APP_DIR / "bin" / "lerobot-robot-client"):
        if not path.is_file():
            errors.append(f"Web v2 helper not found: {path}")
    for module_name in ("cv2", "numpy", "pyrealsense2"):
        try:
            __import__(module_name)
        except Exception as exc:  # pragma: no cover - target environment specific
            errors.append(f"Cannot import {module_name}: {exc}")
    return errors


class EventLog:
    def __init__(self, max_items: int = 800) -> None:
        self._items: collections.deque[dict[str, Any]] = collections.deque(maxlen=max_items)
        self._sequence = 0
        self._lock = threading.Lock()

    def add(self, message: str, level: str = "info") -> None:
        with self._lock:
            self._sequence += 1
            self._items.append(
                {
                    "id": self._sequence,
                    "time": time.strftime("%H:%M:%S"),
                    "level": level,
                    "message": message,
                }
            )

    def after(self, sequence: int) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._items if item["id"] > sequence]


@dataclass(frozen=True)
class CameraSpec:
    key: str
    title: str
    serial: str


class CameraStream:
    def __init__(
        self,
        spec: CameraSpec,
        width: int,
        height: int,
        fps: int,
        jpeg_quality: int,
        event_log: EventLog,
    ) -> None:
        self.spec = spec
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.event_log = event_log
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.thread: threading.Thread | None = None
        self.jpeg: bytes | None = None
        self.version = 0
        self.state = "idle"
        self.message = "대기 중"

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name=f"web-camera-{self.spec.key}",
            daemon=True,
        )
        self.thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def snapshot(self) -> tuple[bytes | None, int, str]:
        with self.condition:
            return self.jpeg, self.version, self.state

    def status(self) -> dict[str, Any]:
        with self.condition:
            return {
                "key": self.spec.key,
                "title": self.spec.title,
                "serial": self.spec.serial,
                "state": self.state,
                "message": self.message,
                "fps": self.fps,
            }

    def _set_state(self, state: str, message: str) -> None:
        with self.condition:
            self.state = state
            self.message = message
            if state == "stopped":
                self.jpeg = None
            self.condition.notify_all()

    def _run(self) -> None:
        import cv2
        import numpy as np
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(self.spec.serial)
        rs_config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        started = False
        try:
            self._set_state("connecting", "연결 중")
            pipeline.start(rs_config)
            started = True
            self._set_state("connected", f"연결됨 · {self.fps} FPS")
            self.event_log.add(f"{self.spec.title} camera connected ({self.spec.serial}).")

            while not self.stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=700)
                except RuntimeError:
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not ok:
                    continue
                with self.condition:
                    self.jpeg = encoded.tobytes()
                    self.version += 1
                    self.condition.notify_all()
        except Exception as exc:
            message = f"연결 실패: {exc}"
            self._set_state("error", message)
            self.event_log.add(f"{self.spec.title} camera error: {exc}", "error")
        finally:
            if started:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            if self.stop_event.is_set():
                self._set_state("stopped", "미리보기 중지됨")


class CameraManager:
    def __init__(self, config: dict[str, Any], event_log: EventLog) -> None:
        self.config = config
        self.event_log = event_log
        self.lock = threading.Lock()
        self.generation = 0
        self.streams: dict[str, CameraStream] = {}
        self.specs = [
            CameraSpec(item["key"], item["title"], item["serial"])
            for item in config["cameras"]
        ]

    def start_all(self) -> None:
        self.start_keys([spec.key for spec in self.specs])

    def start_keys(self, keys: list[str]) -> None:
        width = int(self.config.get("camera_width", 640))
        height = int(self.config.get("camera_height", 480))
        fps = int(self.config.get("camera_fps", 15))
        quality = int(self.config.get("jpeg_quality", 75))
        key_set = set(keys)
        with self.lock:
            self.generation += 1
            new_streams = {
                spec.key: CameraStream(spec, width, height, fps, quality, self.event_log)
                for spec in self.specs
                if spec.key in key_set
            }
            self.streams.update(new_streams)
        for stream in new_streams.values():
            stream.start()
        if new_streams:
            self.event_log.add(f"Camera previews started: {', '.join(new_streams)}.")

    def stop_all(self) -> None:
        self.stop_keys([spec.key for spec in self.specs])

    def stop_keys(self, keys: list[str]) -> None:
        key_set = set(keys)
        with self.lock:
            streams = [stream for key, stream in self.streams.items() if key in key_set]
        for stream in streams:
            stream.stop()
        if streams:
            self.event_log.add(f"Camera previews released: {', '.join(stream.spec.key for stream in streams)}.")

    def restart_all(self) -> None:
        self.stop_all()
        self.start_all()

    def statuses(self) -> list[dict[str, Any]]:
        with self.lock:
            streams = dict(self.streams)
        results = []
        for spec in self.specs:
            stream = streams.get(spec.key)
            if stream:
                results.append(stream.status())
            else:
                results.append(
                    {
                        "key": spec.key,
                        "title": spec.title,
                        "serial": spec.serial,
                        "state": "idle",
                        "message": "대기 중",
                        "fps": int(self.config.get("camera_fps", 15)),
                    }
                )
        return results

    def frame(self, key: str) -> tuple[bytes | None, tuple[int, int], str] | None:
        with self.lock:
            stream = self.streams.get(key)
            generation = self.generation
        if stream is None:
            return None
        jpeg, version, state = stream.snapshot()
        return jpeg, (generation, version), state

    def has_key(self, key: str) -> bool:
        return any(spec.key == key for spec in self.specs)


class RobotClientManager:
    def __init__(
        self,
        config: dict[str, Any],
        cameras: CameraManager,
        relay: RelayStore,
        task_provider: Callable[[], dict[str, Any]],
        event_log: EventLog,
    ) -> None:
        self.config = config
        self.cameras = cameras
        self.relay = relay
        self.task_provider = task_provider
        self.event_log = event_log
        self.policy_camera_keys = list(config["policy_camera_keys"])
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.state = "stopped"
        self.return_code: int | None = None
        self.active_task_id: str | None = None
        self.shutdown_event = threading.Event()

    def status(self) -> dict[str, Any]:
        with self.lock:
            process = self.process
            return {
                "state": self.state,
                "pid": process.pid if process and process.poll() is None else None,
                "return_code": self.return_code,
                "active_task_id": self.active_task_id,
            }

    def start(self) -> tuple[bool, str]:
        with self.lock:
            if self.state in {"starting", "running", "stopping"}:
                return False, "Robot client is already active."
            self.state = "starting"
            self.return_code = None

        target = resolve_script(self.config)
        if not target.is_file():
            with self.lock:
                self.state = "stopped"
            return False, f"Script not found: {target}"

        self.event_log.add("Handing policy cameras to the Web v2 client relay...")
        selected_task = self.task_provider()
        self.relay.reset()
        self.cameras.stop_keys(self.policy_camera_keys)
        if self.shutdown_event.is_set():
            with self.lock:
                self.state = "stopped"
            return False, "Web GUI is shutting down."

        try:
            environment = os.environ.copy()
            environment["PATH"] = f"{APP_DIR / 'bin'}{os.pathsep}{environment.get('PATH', '')}"
            environment["RBY1_WEB_RELAY_DIR"] = str(self.relay.directory)
            environment["RBY1_WEB_RELAY_FPS"] = str(int(self.config.get("camera_fps", 15)))
            environment["RBY1_WEB_RELAY_QUALITY"] = str(int(self.config.get("jpeg_quality", 75)))
            environment["RBY1_WEB_TASK"] = selected_task["instruction"]
            process = subprocess.Popen(
                ["bash", str(target)],
                cwd=str(target.parent),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            with self.lock:
                self.state = "stopped"
                self.active_task_id = None
            self.event_log.add(f"Robot client startup failed: {exc}", "error")
            self.cameras.start_keys(self.policy_camera_keys)
            return False, str(exc)

        with self.lock:
            self.process = process
            self.state = "running"
            self.active_task_id = selected_task["id"]
        self.event_log.add(f"Robot client started (PID {process.pid}).")
        self.event_log.add(
            f"Task {selected_task['number']}. {selected_task['label']}: {selected_task['instruction']}"
        )
        threading.Thread(
            target=self._read_process,
            args=(process,),
            name="web-robot-client-output",
            daemon=True,
        ).start()
        return True, "Robot client started."

    def _read_process(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                self.event_log.add(line.rstrip("\n"), "client")
        finally:
            return_code = process.wait()
            with self.lock:
                if self.process is process:
                    self.process = None
                    self.state = "stopped"
                    self.return_code = return_code
                    self.active_task_id = None
            self.event_log.add(f"Robot client exited with code {return_code}.")
            if not self.shutdown_event.is_set():
                self.relay.reset()
                self.cameras.start_keys(self.policy_camera_keys)

    def stop(self) -> tuple[bool, str]:
        with self.lock:
            process = self.process
            if not process or process.poll() is not None:
                return False, "Robot client is not running."
            self.state = "stopping"
        self.event_log.add("Sending SIGINT to the robot client...")
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return False, "Robot client already exited."
        threading.Thread(
            target=self._terminate_after_timeout,
            args=(process,),
            name="web-robot-client-stop",
            daemon=True,
        ).start()
        return True, "Stop signal sent."

    def _terminate_after_timeout(self, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        self.event_log.add("Client did not stop after 5 seconds; sending SIGTERM.", "warning")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def shutdown(self) -> None:
        self.shutdown_event.set()
        with self.lock:
            process = self.process
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
        self.cameras.stop_all()


class WebApplication:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.log = EventLog()
        self.task_lock = threading.Lock()
        self.tasks = {task["id"]: dict(task) for task in config["tasks"]}
        self.selected_task_id = str(config["default_task_id"])
        self.policy_camera_keys = list(config["policy_camera_keys"])
        relay_directory = Path(tempfile.mkdtemp(prefix="rby1-web-v2-"))
        self.relay = RelayStore(relay_directory, self.policy_camera_keys)
        self.cameras = CameraManager(config, self.log)
        self.client = RobotClientManager(config, self.cameras, self.relay, self.selected_task, self.log)
        self.stop_event = threading.Event()

    def selected_task(self) -> dict[str, Any]:
        with self.task_lock:
            return dict(self.tasks[self.selected_task_id])

    def select_task(self, task_id: str) -> tuple[bool, str]:
        if self.client_active():
            return False, "Task cannot be changed while the robot client is active."
        with self.task_lock:
            task = self.tasks.get(task_id)
            if task is None:
                return False, f"Unknown task: {task_id}"
            self.selected_task_id = task_id
            selected = dict(task)
        self.log.add(f"Selected task {selected['number']}. {selected['label']}.")
        return True, selected["instruction"]

    def client_active(self) -> bool:
        return self.client.status()["state"] in {"starting", "running", "stopping"}

    def camera_statuses(self) -> list[dict[str, Any]]:
        statuses = {item["key"]: item for item in self.cameras.statuses()}
        if self.client_active():
            now = time.time()
            for key in self.policy_camera_keys:
                _jpeg, _sequence, timestamp = self.relay.read(key)
                fresh = timestamp is not None and now - timestamp < 2.0
                status = statuses[key]
                status["state"] = "connected" if fresh else "connecting"
                status["message"] = (
                    f"정책 프레임 공유 중 · {status['fps']} FPS" if fresh else "정책 프레임 대기 중"
                )
        return [statuses[item["key"]] for item in self.config["cameras"]]

    def frame(self, key: str) -> tuple[bytes | None, tuple[Any, ...], str] | None:
        if self.client_active() and key in self.policy_camera_keys:
            jpeg, sequence, timestamp = self.relay.read(key)
            fresh = timestamp is not None and time.time() - timestamp < 2.0
            return jpeg if fresh else None, ("relay", key, sequence), "connected" if fresh else "connecting"
        direct = self.cameras.frame(key)
        if direct is None:
            return None
        jpeg, version, state = direct
        return jpeg, ("direct", *version), state

    def status(self) -> dict[str, Any]:
        selected_task = self.selected_task()
        return {
            "cameras": self.camera_statuses(),
            "client": self.client.status(),
            "task": selected_task["instruction"],
            "tasks": list(self.tasks.values()),
            "selected_task_id": selected_task["id"],
            "model": self.config.get("displayed_model", "Configured in shell script"),
            "script": resolve_script(self.config).name,
            "version": self.config.get("web_version", "web-v2"),
        }

    def shutdown(self) -> None:
        self.stop_event.set()
        self.client.shutdown()
        self.relay.close(cleanup=True)


class RBY1HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: WebApplication) -> None:
        self.app = app
        super().__init__(address, RequestHandler)


class RequestHandler(BaseHTTPRequestHandler):
    server: RBY1HTTPServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path == "/app.css":
            self._send_file(STATIC_DIR / "app.css", "text/css; charset=utf-8")
        elif parsed.path == "/app.js":
            self._send_file(STATIC_DIR / "app.js", "text/javascript; charset=utf-8")
        elif parsed.path == "/api/status":
            self._send_json(self.server.app.status())
        elif parsed.path == "/api/logs":
            query = parse_qs(parsed.query)
            try:
                after = int(query.get("after", ["0"])[0])
            except ValueError:
                after = 0
            self._send_json({"items": self.server.app.log.after(after)})
        elif parsed.path.startswith("/camera/") and parsed.path.endswith(".mjpg"):
            key = parsed.path.removeprefix("/camera/").removesuffix(".mjpg")
            self._stream_camera(key)
        else:
            self._send_json({"ok": False, "message": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length:
            self.rfile.read(content_length)

        # A custom same-origin header blocks drive-by HTML form submissions.
        # This is CSRF hardening, not user authentication.
        if self.headers.get("X-RBY1-Control") != "web-v2":
            self._send_json(
                {"ok": False, "message": "Missing web control header."},
                HTTPStatus.FORBIDDEN,
            )
            return

        if parsed.path == "/api/client/start":
            ok, message = self.server.app.client.start()
            self._send_json({"ok": ok, "message": message}, HTTPStatus.OK if ok else HTTPStatus.CONFLICT)
        elif parsed.path == "/api/client/stop":
            ok, message = self.server.app.client.stop()
            self._send_json({"ok": ok, "message": message}, HTTPStatus.OK if ok else HTTPStatus.CONFLICT)
        elif parsed.path == "/api/cameras/restart":
            client_state = self.server.app.client.status()["state"]
            if client_state != "stopped":
                self._send_json(
                    {"ok": False, "message": "Cameras are reserved by the robot client."},
                    HTTPStatus.CONFLICT,
                )
                return
            self.server.app.cameras.restart_all()
            self._send_json({"ok": True, "message": "Camera reconnect requested."})
        elif parsed.path == "/api/task/select":
            query = parse_qs(parsed.query)
            task_id = query.get("id", [""])[0]
            ok, message = self.server.app.select_task(task_id)
            self._send_json(
                {"ok": ok, "message": message},
                HTTPStatus.OK if ok else HTTPStatus.CONFLICT,
            )
        else:
            self._send_json({"ok": False, "message": "Not found"}, HTTPStatus.NOT_FOUND)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self._send_json({"ok": False, "message": "File not found"}, HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _stream_camera(self, key: str) -> None:
        if not self.server.app.cameras.has_key(key):
            self._send_json({"ok": False, "message": "Unknown camera"}, HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        previous: tuple[Any, ...] | None = None
        try:
            while not self.server.app.stop_event.is_set():
                result = self.server.app.frame(key)
                if result is None:
                    time.sleep(0.1)
                    continue
                jpeg, version, _state = result
                if jpeg is None or version == previous:
                    time.sleep(0.04)
                    continue
                previous = version
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, format_string: str, *args: Any) -> None:
        if self.command != "GET" or not self.path.startswith(("/api/", "/camera/")):
            super().log_message(format_string, *args)


def local_addresses(port: int) -> list[str]:
    addresses = {"127.0.0.1"}
    try:
        result = subprocess.run(
            ["ip", "-json", "address", "show", "up"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        for interface in json.loads(result.stdout or "[]"):
            for address in interface.get("addr_info", []):
                if address.get("family") == "inet" and address.get("scope") == "global":
                    addresses.add(address["local"])
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(item[4][0])
    except socket.gaierror:
        pass
    return [f"http://{address}:{port}" for address in sorted(addresses)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RB-Y1 browser control GUI")
    parser.add_argument("--host", help="Listen address (default: config.json)")
    parser.add_argument("--port", type=int, help="Listen port (default: config.json)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration without opening cameras or a server",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config()
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    errors = validate(config)
    if errors:
        print("Web GUI check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    if args.check:
        print("Web GUI check passed")
        print(f"Robot client: {resolve_script(config)}")
        for camera in config["cameras"]:
            print(f"{camera['title']}: {camera['serial']}")
        return 0

    host = args.host or str(config["host"])
    port = args.port or int(config["port"])
    app = WebApplication(config)
    server = RBY1HTTPServer((host, port), app)
    app.log.add("RB-Y1 Web GUI server started.")
    app.cameras.start_all()

    print("RB-Y1 Web GUI is running.")
    for address in local_addresses(port):
        print(f"  {address}")
    print(f"  Remote browser: http://<JETSON-IP>:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping RB-Y1 Web GUI...")
    finally:
        app.stop_event.set()
        server.shutdown()
        server.server_close()
        app.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
