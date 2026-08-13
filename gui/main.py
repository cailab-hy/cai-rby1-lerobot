#!/usr/bin/env python3
"""Standalone RB-Y1 camera monitor and robot-client launcher.

All configuration and implementation live inside ``gui/``. The existing
robot client shell script is invoked as-is and is never rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    required = {"robot_client_script", "cameras"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"config.json missing key(s): {', '.join(sorted(missing))}")
    if len(config["cameras"]) != 3:
        raise ValueError("config.json must contain exactly three cameras")

    serials = [camera.get("serial") for camera in config["cameras"]]
    if any(not serial for serial in serials) or len(set(serials)) != 3:
        raise ValueError("Each camera needs a unique, non-empty serial")
    return config


def script_path(config: dict[str, Any]) -> Path:
    path = Path(config["robot_client_script"])
    return (APP_DIR / path).resolve() if not path.is_absolute() else path


def dependency_check(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    target = script_path(config)
    if not target.is_file():
        errors.append(f"Robot client script not found: {target}")

    for module_name in ("tkinter", "cv2", "PIL", "pyrealsense2"):
        try:
            __import__(module_name)
        except Exception as exc:  # pragma: no cover - depends on target system
            errors.append(f"Cannot import {module_name}: {exc}")
    return errors


@dataclass(frozen=True)
class CameraSpec:
    key: str
    title: str
    serial: str


class CameraStream:
    """Own one RealSense pipeline from a background thread."""

    def __init__(
        self,
        spec: CameraSpec,
        width: int,
        height: int,
        fps: int,
        event_queue: queue.Queue[tuple[Any, ...]],
    ) -> None:
        self.spec = spec
        self.width = width
        self.height = height
        self.fps = fps
        self.event_queue = event_queue
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame: Any | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name=f"camera-{self.spec.key}",
            daemon=True,
        )
        self.thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def frame(self) -> Any | None:
        with self.frame_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def _emit_state(self, state: str, message: str) -> None:
        self.event_queue.put(("camera_state", self.spec.key, state, message))

    def _run(self) -> None:
        import cv2
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
            self._emit_state("connecting", "연결 중")
            pipeline.start(rs_config)
            started = True
            self._emit_state("connected", f"연결됨 · {self.fps} FPS")

            while not self.stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=700)
                except RuntimeError:
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = cv2.cvtColor(
                    __import__("numpy").asanyarray(color_frame.get_data()),
                    cv2.COLOR_BGR2RGB,
                )
                with self.frame_lock:
                    self.latest_frame = frame
        except Exception as exc:
            self._emit_state("error", f"연결 실패: {exc}")
        finally:
            if started:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            with self.frame_lock:
                self.latest_frame = None
            if self.stop_event.is_set():
                self._emit_state("stopped", "미리보기 중지됨")


class RBY1ControlGUI:
    BG = "#111827"
    PANEL = "#1f2937"
    PANEL_DARK = "#0b1220"
    TEXT = "#f3f4f6"
    MUTED = "#9ca3af"
    GREEN = "#34d399"
    RED = "#ef4444"
    BLUE = "#2563eb"

    def __init__(self, root: Any, config: dict[str, Any]) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.config = config
        self.events: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self.streams: dict[str, CameraStream] = {}
        self.camera_states: dict[str, str] = {}
        self.camera_labels: dict[str, Any] = {}
        self.camera_status_labels: dict[str, Any] = {}
        self.client_process: subprocess.Popen[str] | None = None
        self.closing = False

        self.root.title("RB-Y1 VLA CONTROL")
        self.root.configure(bg=self.BG)
        self.root.geometry("1450x900")
        self.root.minsize(1050, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_styles()
        self._build_ui()
        self._make_streams()
        self.root.after(150, self.start_cameras)
        self.root.after(50, self._drain_events)
        self.root.after(80, self._refresh_frames)

    def _build_styles(self) -> None:
        style = self.ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground="#374151", background="#374151")

    def _build_ui(self) -> None:
        tk = self.tk

        header = tk.Frame(self.root, bg=self.BG, padx=22, pady=16)
        header.pack(fill="x")
        tk.Label(
            header,
            text="RB-Y1 VLA CONTROL",
            bg=self.BG,
            fg=self.TEXT,
            font=("DejaVu Sans", 21, "bold"),
        ).pack(side="left")
        self.summary_label = tk.Label(
            header,
            text="Cameras 0/3  |  Client stopped",
            bg=self.BG,
            fg=self.MUTED,
            font=("DejaVu Sans", 11),
        )
        self.summary_label.pack(side="right")

        camera_row = tk.Frame(self.root, bg=self.BG, padx=14)
        camera_row.pack(fill="both", expand=True)
        for column, camera in enumerate(self.config["cameras"]):
            camera_row.grid_columnconfigure(column, weight=1, uniform="camera")
            card = tk.Frame(
                camera_row,
                bg=self.PANEL,
                highlightthickness=1,
                highlightbackground="#374151",
                padx=8,
                pady=8,
            )
            card.grid(row=0, column=column, sticky="nsew", padx=6)
            camera_row.grid_rowconfigure(0, weight=1)

            tk.Label(
                card,
                text=camera["title"],
                bg=self.PANEL,
                fg=self.TEXT,
                font=("DejaVu Sans", 13, "bold"),
            ).pack(pady=(2, 7))
            preview = tk.Label(
                card,
                text="D405 CAMERA\n대기 중",
                bg="#030712",
                fg=self.MUTED,
                font=("DejaVu Sans", 12),
            )
            preview.pack(fill="both", expand=True)
            self.camera_labels[camera["key"]] = preview

            status = tk.Label(
                card,
                text=f"○ {camera['serial']}",
                bg=self.PANEL,
                fg=self.MUTED,
                anchor="w",
                font=("DejaVu Sans", 9),
            )
            status.pack(fill="x", pady=(7, 0))
            self.camera_status_labels[camera["key"]] = status

        controls = tk.Frame(self.root, bg=self.PANEL, padx=20, pady=14)
        controls.pack(fill="x", padx=20, pady=(14, 10))
        controls.grid_columnconfigure(0, weight=3)
        controls.grid_columnconfigure(1, weight=1)

        tk.Label(controls, text="VLA Task (script configuration)", bg=self.PANEL, fg=self.MUTED).grid(
            row=0, column=0, sticky="w"
        )
        tk.Label(controls, text="Inference Model", bg=self.PANEL, fg=self.MUTED).grid(
            row=0, column=1, sticky="w", padx=(14, 0)
        )

        task_value = tk.StringVar(value=self.config.get("displayed_task", "Configured in shell script"))
        task_entry = tk.Entry(
            controls,
            textvariable=task_value,
            state="readonly",
            readonlybackground="#374151",
            fg=self.TEXT,
            relief="flat",
            font=("DejaVu Sans", 11),
        )
        task_entry.grid(row=1, column=0, sticky="ew", ipady=9, pady=(5, 12))

        model_value = tk.StringVar(value=self.config.get("displayed_model", "Configured in shell script"))
        model_entry = tk.Entry(
            controls,
            textvariable=model_value,
            state="readonly",
            readonlybackground="#374151",
            fg=self.TEXT,
            relief="flat",
            font=("DejaVu Sans", 11),
        )
        model_entry.grid(row=1, column=1, sticky="ew", padx=(14, 0), ipady=9, pady=(5, 12))

        button_row = tk.Frame(controls, bg=self.PANEL)
        button_row.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.reconnect_button = tk.Button(
            button_row,
            text="↻ 카메라 다시 연결",
            command=self.restart_cameras,
            bg="#374151",
            fg=self.TEXT,
            activebackground="#4b5563",
            activeforeground=self.TEXT,
            relief="flat",
            padx=16,
            pady=10,
        )
        self.reconnect_button.pack(side="left")

        self.stop_button = tk.Button(
            button_row,
            text="■ 클라이언트 정지",
            command=self.stop_client,
            state="disabled",
            bg="#7f1d1d",
            fg="white",
            disabledforeground="#9ca3af",
            relief="flat",
            padx=18,
            pady=10,
        )
        self.stop_button.pack(side="right")
        self.start_button = tk.Button(
            button_row,
            text="▶ 로봇 클라이언트 실행",
            command=self.start_client,
            bg=self.BLUE,
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=10,
        )
        self.start_button.pack(side="right", padx=(0, 10))

        log_box = tk.Frame(self.root, bg=self.BG, padx=20)
        log_box.pack(fill="both", pady=(0, 16))
        tk.Label(log_box, text="Client Log", bg=self.BG, fg=self.MUTED).pack(anchor="w", pady=(0, 5))
        self.log = tk.Text(
            log_box,
            height=8,
            bg=self.PANEL_DARK,
            fg="#d1d5db",
            insertbackground="white",
            relief="flat",
            padx=10,
            pady=8,
            state="disabled",
            font=("DejaVu Sans Mono", 9),
        )
        self.log.pack(fill="both")
        self._append_log("GUI ready. Camera preview is starting.")

    def _make_streams(self) -> None:
        width = int(self.config.get("camera_width", 640))
        height = int(self.config.get("camera_height", 480))
        fps = int(self.config.get("camera_fps", 15))
        self.streams = {
            camera["key"]: CameraStream(
                CameraSpec(camera["key"], camera["title"], camera["serial"]),
                width,
                height,
                fps,
                self.events,
            )
            for camera in self.config["cameras"]
        }

    def start_cameras(self) -> None:
        if self.closing or (self.client_process and self.client_process.poll() is None):
            return
        self.reconnect_button.configure(state="disabled")
        self.camera_states.clear()
        for stream in self.streams.values():
            stream.start()
        self.root.after(1200, lambda: self.reconnect_button.configure(state="normal"))

    def stop_cameras(self) -> None:
        for stream in self.streams.values():
            stream.stop()

    def restart_cameras(self) -> None:
        if self.client_process and self.client_process.poll() is None:
            self._append_log("Camera reconnect is unavailable while the robot client is running.")
            return
        self.stop_cameras()
        self._make_streams()
        self.start_cameras()
        self._append_log("Camera reconnect requested.")

    def start_client(self) -> None:
        if self.client_process and self.client_process.poll() is None:
            return

        target = script_path(self.config)
        if not target.is_file():
            self._append_log(f"ERROR: script not found: {target}")
            return

        self._append_log("Releasing camera previews before starting the robot client...")
        self.stop_cameras()
        self.reconnect_button.configure(state="disabled")
        for label in self.camera_labels.values():
            label.configure(image="", text="ROBOT CLIENT USING CAMERA")
            label.image = None

        try:
            process = subprocess.Popen(
                ["bash", str(target)],
                cwd=str(target.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            self._append_log(f"ERROR: failed to start client: {exc}")
            self._make_streams()
            self.start_cameras()
            return

        self.client_process = process
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._append_log(f"Started {target.name} (PID {process.pid})")
        self._update_summary()
        threading.Thread(
            target=self._read_process,
            args=(process,),
            name="robot-client-output",
            daemon=True,
        ).start()

    def _read_process(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                self.events.put(("log", line.rstrip("\n")))
        finally:
            return_code = process.wait()
            self.events.put(("process_done", process, return_code))

    def stop_client(self) -> None:
        process = self.client_process
        if not process or process.poll() is not None:
            return
        self._append_log("Sending SIGINT to the robot client...")
        self.stop_button.configure(state="disabled")
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        self.root.after(5000, lambda: self._terminate_if_running(process))

    def _terminate_if_running(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            self._append_log("Client did not stop after 5 seconds; sending SIGTERM.")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _drain_events(self) -> None:
        if self.closing:
            return
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    self._append_log(event[1])
                elif kind == "camera_state":
                    _, key, state, message = event
                    self.camera_states[key] = state
                    color = self.GREEN if state == "connected" else self.RED if state == "error" else self.MUTED
                    serial = self.streams[key].spec.serial
                    self.camera_status_labels[key].configure(text=f"● {message} · {serial}", fg=color)
                    self._update_summary()
                elif kind == "process_done":
                    _, process, return_code = event
                    if self.client_process is process:
                        self.client_process = None
                        self.start_button.configure(state="normal")
                        self.stop_button.configure(state="disabled")
                        self._append_log(f"Robot client exited with code {return_code}.")
                        self._make_streams()
                        self.start_cameras()
                        self._update_summary()
        except queue.Empty:
            pass
        self.root.after(50, self._drain_events)

    def _refresh_frames(self) -> None:
        if self.closing:
            return
        from PIL import Image, ImageTk

        if not (self.client_process and self.client_process.poll() is None):
            for key, stream in self.streams.items():
                frame = stream.frame()
                if frame is None:
                    continue
                label = self.camera_labels[key]
                target_width = max(label.winfo_width(), 320)
                target_height = max(label.winfo_height(), 240)
                image = Image.fromarray(frame)
                image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                label.configure(image=photo, text="")
                label.image = photo
        self.root.after(80, self._refresh_frames)

    def _update_summary(self) -> None:
        connected = sum(state == "connected" for state in self.camera_states.values())
        running = bool(self.client_process and self.client_process.poll() is None)
        client_text = "Client running" if running else "Client stopped"
        self.summary_label.configure(
            text=f"Cameras {connected}/3  |  {client_text}",
            fg=self.GREEN if running else self.MUTED,
        )

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{timestamp}] {message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def close(self) -> None:
        from tkinter import messagebox

        process = self.client_process
        if process and process.poll() is None:
            if not messagebox.askyesno(
                "RB-Y1 VLA CONTROL",
                "Robot client is running. Stop it and close the GUI?",
            ):
                return

        self.closing = True
        self.stop_cameras()
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
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RB-Y1 VLA control GUI")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and dependencies without opening cameras or a window",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config()
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    errors = dependency_check(config)
    if errors:
        print("GUI check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    if args.check:
        print("GUI check passed")
        print(f"Robot client: {script_path(config)}")
        for camera in config["cameras"]:
            print(f"{camera['title']}: {camera['serial']}")
        return 0

    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Cannot open the graphical display: {exc}", file=sys.stderr)
        print(
            "Run ./gui/run_gui.sh from the Jetson desktop, or use SSH X forwarding.",
            file=sys.stderr,
        )
        return 1
    RBY1ControlGUI(root, config)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
