"""Small mmap-based latest-frame relay shared by the web server and client shim."""

from __future__ import annotations

import mmap
import os
import struct
import time
from pathlib import Path

HEADER = struct.Struct("<QIIQ")  # sequence, jpeg length, reserved, timestamp_ns
DEFAULT_BUFFER_SIZE = 1024 * 1024


def _validate_key(key: str) -> str:
    if not key or not all(character.isalnum() or character in "_-" for character in key):
        raise ValueError(f"Invalid relay key: {key!r}")
    return key


def relay_path(directory: Path, key: str) -> Path:
    return directory / f"{_validate_key(key)}.frame"


class RelayStore:
    """Server-side relay files with lock-free, sequence-checked reads."""

    def __init__(
        self,
        directory: Path,
        keys: list[str],
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        self.directory = directory
        self.keys = list(keys)
        self.buffer_size = buffer_size
        self.maps: dict[str, mmap.mmap] = {}
        self.files: dict[str, object] = {}
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        for key in self.keys:
            path = relay_path(directory, key)
            file = path.open("w+b")
            file.truncate(buffer_size)
            self.files[key] = file
            self.maps[key] = mmap.mmap(file.fileno(), buffer_size, access=mmap.ACCESS_WRITE)
        self.reset()

    def reset(self) -> None:
        for mapping in self.maps.values():
            HEADER.pack_into(mapping, 0, 0, 0, 0, 0)

    def read(self, key: str) -> tuple[bytes | None, int, float | None]:
        mapping = self.maps.get(key)
        if mapping is None:
            return None, 0, None
        for _ in range(3):
            sequence_before, length, _reserved, timestamp_ns = HEADER.unpack_from(mapping, 0)
            if sequence_before == 0 or sequence_before % 2 or length <= 0:
                return None, sequence_before, None
            if length > self.buffer_size - HEADER.size:
                return None, sequence_before, None
            jpeg = bytes(mapping[HEADER.size : HEADER.size + length])
            sequence_after = struct.unpack_from("<Q", mapping, 0)[0]
            if sequence_before == sequence_after and sequence_after % 2 == 0:
                return jpeg, sequence_after, timestamp_ns / 1_000_000_000
        return None, 0, None

    def close(self, cleanup: bool = True) -> None:
        for mapping in self.maps.values():
            mapping.close()
        for file in self.files.values():
            file.close()
        self.maps.clear()
        self.files.clear()
        if cleanup:
            for key in self.keys:
                try:
                    relay_path(self.directory, key).unlink()
                except FileNotFoundError:
                    pass
            try:
                self.directory.rmdir()
            except OSError:
                pass


class RelayWriter:
    """Client-side writer. A new JPEG atomically replaces the prior frame."""

    def __init__(self, directory: Path, keys: list[str]) -> None:
        self.directory = directory
        self.maps: dict[str, mmap.mmap] = {}
        self.files: dict[str, object] = {}
        self.buffer_sizes: dict[str, int] = {}
        for key in keys:
            path = relay_path(directory, key)
            file = path.open("r+b", buffering=0)
            size = os.fstat(file.fileno()).st_size
            if size <= HEADER.size:
                file.close()
                raise ValueError(f"Relay file is too small: {path}")
            self.files[key] = file
            self.buffer_sizes[key] = size
            self.maps[key] = mmap.mmap(file.fileno(), size, access=mmap.ACCESS_WRITE)

    def write(self, key: str, jpeg: bytes) -> bool:
        mapping = self.maps.get(key)
        if mapping is None or len(jpeg) > self.buffer_sizes[key] - HEADER.size:
            return False
        current = struct.unpack_from("<Q", mapping, 0)[0]
        if current % 2:
            current += 1
        writing_sequence = current + 1
        struct.pack_into("<Q", mapping, 0, writing_sequence)
        mapping[HEADER.size : HEADER.size + len(jpeg)] = jpeg
        struct.pack_into("<IIQ", mapping, 8, len(jpeg), 0, time.time_ns())
        struct.pack_into("<Q", mapping, 0, writing_sequence + 1)
        return True

    def close(self) -> None:
        for mapping in self.maps.values():
            mapping.close()
        for file in self.files.values():
            file.close()
        self.maps.clear()
        self.files.clear()
