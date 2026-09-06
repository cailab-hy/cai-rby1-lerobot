"""Background writer for the standard executed-action JSONL log."""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class AsyncJSONLWriter:
    """Keep the upstream action-log schema without blocking high-rate control."""

    def __init__(self, path: str | Path, *, max_queue_size: int = 4096) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: Queue[dict[str, Any] | None] = Queue(maxsize=max_queue_size)
        self._closed = False
        self.dropped_records = 0
        self._thread = threading.Thread(target=self._run, name="action-log-writer", daemon=True)
        self._thread.start()

    def submit(self, record: dict[str, Any]) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait(record)
            return True
        except Full:
            self.dropped_records += 1
            return False

    def _run(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                while True:
                    try:
                        item = self._queue.get(timeout=0.25)
                    except Empty:
                        continue
                    if item is None:
                        self._queue.task_done()
                        break
                    stream.write(
                        json.dumps(
                            _json_safe(item), allow_nan=False, separators=(",", ":")
                        )
                        + "\n"
                    )
                    self._queue.task_done()
                stream.flush()
        except Exception:
            logging.getLogger(__name__).exception("executed-action log writer failed")

    def close(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(None, timeout=timeout)
        except Full:
            logging.getLogger(__name__).warning(
                "executed-action log queue did not drain before timeout"
            )
            return
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logging.getLogger(__name__).warning(
                "executed-action log writer did not flush before timeout"
            )

    def __enter__(self) -> "AsyncJSONLWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
