# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transport-only resizing and JPEG encoding for robot camera observations."""

import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

JPEG_QUALITY = 85

_IMAGE_TRANSPORT_KEY = "__image_transport__"
_IMAGE_TRANSPORT_VERSION = "lerobot_image_transport_v1"
_RGB_CHANNEL_ORDER = "RGB"


@dataclass(frozen=True)
class ImageEncodeStats:
    resize_time: float = 0.0
    jpeg_encode_time: float = 0.0
    total_time: float = 0.0
    original_bytes: int = 0
    transport_bytes: int = 0
    image_count: int = 0


@dataclass(frozen=True)
class ImageDecodeStats:
    jpeg_decode_time: float = 0.0
    restore_resize_time: float = 0.0
    total_time: float = 0.0
    image_count: int = 0


def validate_image_resize_scale(scale: float) -> None:
    """Validate the client-side spatial scale before the robot is started."""
    try:
        normalized_scale = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"image_resize_scale must satisfy 0 < scale <= 1.0, got {scale!r}"
        ) from exc

    if (
        isinstance(scale, bool)
        or not math.isfinite(normalized_scale)
        or not 0 < normalized_scale <= 1.0
    ):
        raise ValueError(f"image_resize_scale must satisfy 0 < scale <= 1.0, got {scale!r}")


def _validate_rgb_image(image: Any, camera_key: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Camera observation '{camera_key}' must be a NumPy array, got {type(image)}")
    if image.dtype != np.uint8:
        raise ValueError(f"Camera observation '{camera_key}' must have dtype uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Camera observation '{camera_key}' must have HWC shape with 3 RGB channels, got {image.shape}"
        )
    return image


def _transport_image(
    *,
    encoding: str,
    data: bytes | np.ndarray,
    original_shape: tuple[int, int, int],
    transport_shape: tuple[int, int, int],
) -> dict[str, Any]:
    return {
        _IMAGE_TRANSPORT_KEY: _IMAGE_TRANSPORT_VERSION,
        "encoding": encoding,
        "data": data,
        "original_shape": original_shape,
        "transport_shape": transport_shape,
        "channel_order": _RGB_CHANNEL_ORDER,
    }


def encode_observation_images(
    observation: Mapping[str, Any],
    camera_keys: Iterable[str],
    image_resize_scale: float,
    jpeg_compression: bool,
) -> tuple[dict[str, Any], ImageEncodeStats]:
    """Shallow-copy an observation and transform only its declared camera entries."""
    validate_image_resize_scale(image_resize_scale)
    total_start = time.perf_counter()
    transport_observation = dict(observation)
    resize_time = 0.0
    jpeg_encode_time = 0.0
    original_bytes = 0
    transport_bytes = 0
    image_count = 0

    for camera_key in camera_keys:
        if camera_key not in observation:
            continue

        original_image = _validate_rgb_image(observation[camera_key], camera_key)
        image = original_image
        original_shape = tuple(int(dimension) for dimension in original_image.shape)
        original_bytes += original_image.nbytes
        image_count += 1

        if image_resize_scale < 1.0:
            height, width = original_image.shape[:2]
            transport_width = max(1, round(width * image_resize_scale))
            transport_height = max(1, round(height * image_resize_scale))
            resize_start = time.perf_counter()
            image = cv2.resize(
                original_image,
                (transport_width, transport_height),
                interpolation=cv2.INTER_AREA,
            )
            resize_time += time.perf_counter() - resize_start

        if jpeg_compression:
            encode_start = time.perf_counter()
            bgr_image = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode(
                ".jpg",
                bgr_image,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            jpeg_encode_time += time.perf_counter() - encode_start
            if not success:
                raise ValueError(f"JPEG encoding failed for camera observation '{camera_key}'")
            data = encoded.tobytes()
            transport_bytes += len(data)
            transport_observation[camera_key] = _transport_image(
                encoding="jpeg",
                data=data,
                original_shape=original_shape,
                transport_shape=tuple(int(dimension) for dimension in image.shape),
            )
        else:
            transport_bytes += image.nbytes
            transport_observation[camera_key] = _transport_image(
                encoding="raw_resized",
                data=image,
                original_shape=original_shape,
                transport_shape=tuple(int(dimension) for dimension in image.shape),
            )

    stats = ImageEncodeStats(
        resize_time=resize_time,
        jpeg_encode_time=jpeg_encode_time,
        total_time=time.perf_counter() - total_start,
        original_bytes=original_bytes,
        transport_bytes=transport_bytes,
        image_count=image_count,
    )
    return transport_observation, stats


def _is_transport_image(value: Any) -> bool:
    return isinstance(value, dict) and value.get(_IMAGE_TRANSPORT_KEY) == _IMAGE_TRANSPORT_VERSION


def _shape_from_metadata(value: Mapping[str, Any], name: str, camera_key: str) -> tuple[int, int, int]:
    shape = value.get(name)
    if not isinstance(shape, tuple) or len(shape) != 3 or any(
        not isinstance(dimension, int) or dimension <= 0 for dimension in shape
    ):
        raise ValueError(f"Invalid {name} for transported camera observation '{camera_key}': {shape}")
    if shape[2] != 3:
        raise ValueError(f"Transported camera observation '{camera_key}' must have 3 channels, got {shape}")
    return shape


def decode_observation_images(
    observation: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ImageDecodeStats]:
    """Restore marked transport images while leaving legacy observations untouched."""
    transport_keys = [key for key, value in observation.items() if _is_transport_image(value)]
    if not transport_keys:
        return observation, ImageDecodeStats()

    total_start = time.perf_counter()
    decoded_observation = dict(observation)
    jpeg_decode_time = 0.0
    restore_resize_time = 0.0

    for camera_key in transport_keys:
        value = observation[camera_key]
        original_shape = _shape_from_metadata(value, "original_shape", camera_key)
        transport_shape = _shape_from_metadata(value, "transport_shape", camera_key)
        if value.get("channel_order") != _RGB_CHANNEL_ORDER:
            raise ValueError(
                f"Unsupported channel order for transported camera observation '{camera_key}': "
                f"{value.get('channel_order')!r}"
            )

        encoding = value.get("encoding")
        if encoding == "jpeg":
            data = value.get("data")
            if not isinstance(data, bytes):
                raise TypeError(f"JPEG data for camera observation '{camera_key}' must be bytes")
            decode_start = time.perf_counter()
            bgr_image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr_image is None:
                raise ValueError(f"JPEG decoding failed for camera observation '{camera_key}'")
            image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            jpeg_decode_time += time.perf_counter() - decode_start
        elif encoding == "raw_resized":
            image = _validate_rgb_image(value.get("data"), camera_key)
        else:
            raise ValueError(
                f"Unsupported encoding for transported camera observation '{camera_key}': {encoding!r}"
            )

        if image.shape != transport_shape:
            raise ValueError(
                f"Transport shape mismatch for camera observation '{camera_key}': "
                f"metadata={transport_shape}, decoded={image.shape}"
            )

        if image.shape != original_shape:
            restore_start = time.perf_counter()
            image = cv2.resize(
                image,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            restore_resize_time += time.perf_counter() - restore_start

        decoded_observation[camera_key] = _validate_rgb_image(image, camera_key)

    stats = ImageDecodeStats(
        jpeg_decode_time=jpeg_decode_time,
        restore_resize_time=restore_resize_time,
        total_time=time.perf_counter() - total_start,
        image_count=len(transport_keys),
    )
    return decoded_observation, stats
