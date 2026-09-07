#!/usr/bin/env python3
"""Standalone Korean/English speech-to-text service.

This runs on the machine holding the microphone and an NVIDIA GPU - normally
the laptop the operator opens the GUI on - not on the Jetson. The aarch64
ctranslate2 wheels are built without CUDA, so Whisper on the Jetson would fall
back to ARM CPU and take seconds per command.

Keeping it here also preserves the original arrangement: speech is recognized
locally in front of the operator, and only the resulting task command travels
to the robot.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from .whisper_stt import WhisperSTT
except ImportError:
    from whisper_stt import WhisperSTT

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
MAX_AUDIO_BYTES = 8 * 1024 * 1024


class RequestHandler(BaseHTTPRequestHandler):
    server: "STTServer"

    def _cors(self) -> None:
        # The GUI is served from the Jetson on another origin, so the browser
        # sends a preflight before it will post audio here.
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", "*"))
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-RBY1-Control")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Vary", "Origin")

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlparse(self.path).path in ("/health", "/"):
            self._send_json({"ok": True, **self.server.stt.status()})
        else:
            self._send_json({"ok": False, "message": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path != "/transcribe":
            self._send_json({"ok": False, "message": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length > MAX_AUDIO_BYTES:
            self._send_json({"ok": False, "message": "Audio upload too large."},
                            HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        body = self.rfile.read(length) if length else b""
        if not body:
            self._send_json({"ok": False, "message": "Empty audio upload."}, HTTPStatus.BAD_REQUEST)
            return

        requested = (parse_qs(parsed.query).get("lang", [""])[0] or "").strip().lower()
        language = requested if requested in self.server.stt.languages else None
        try:
            result = self.server.stt.transcribe(body, language=language)
        except (ValueError, wave.Error) as exc:
            self._send_json({"ok": False, "message": f"Malformed audio upload: {exc}"},
                            HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            print(f"Speech recognition failed: {exc}", file=sys.stderr)
            self._send_json({"ok": False, "message": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if result.get("ok") and result.get("text"):
            print(f"[{result['language']}] {result['text']}  "
                  f"({result['elapsed_ms']} ms, {result['audio_seconds']:.1f} s audio)", flush=True)
        self._send_json(result)

    def log_message(self, format_string: str, *args: Any) -> None:
        pass


class STTServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], stt: WhisperSTT) -> None:
        self.stt = stt
        super().__init__(address, RequestHandler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RB-Y1 Korean/English speech recognition service")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Listen address (default: 127.0.0.1; use 0.0.0.0 to serve other devices)")
    parser.add_argument("--port", type=int, default=8002, help="Listen port (default: 8002)")
    parser.add_argument("--check", action="store_true",
                        help="Load the model and exit without binding a port")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    stt = WhisperSTT(config, logger=lambda message, level="info": print(f"[{level}] {message}", flush=True))
    if not stt.enabled:
        print("stt.enabled is false in config.json; nothing to serve.", file=sys.stderr)
        return 1

    if args.check:
        stt._warm_up()  # noqa: SLF001 - deliberate synchronous warm-up for the check path
        status = stt.status()
        print(f"Speech recognition check: {status['state']} "
              f"({status['model']} · {status['device']} · {status['compute_type']})")
        return 0 if status["state"] == "ready" else 1

    server = STTServer((args.host, args.port), stt)
    stt.warm_up_async()
    print(f"RB-Y1 speech recognition service on http://{args.host}:{args.port}")
    print(f"  model {stt.model_name}, languages {', '.join(stt.languages)}")
    print("  POST /transcribe (audio/wav)   GET /health")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping speech recognition service...")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
