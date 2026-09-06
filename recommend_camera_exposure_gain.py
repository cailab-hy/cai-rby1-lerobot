#!/usr/bin/env python3
"""Recommend RealSense RGB exposure/gain from LeRobot dataset statistics.

This utility performs a measurement-based, coarse-to-fine search.  It only
touches color-sensor exposure, gain, and the auto-exposure switch; white balance
is intentionally left unchanged.  By default every original camera setting is
restored.  Pass --apply to leave the final recommendations active.

The target is the compact, globally aggregated ``meta/stats.json`` summary, so
the result is a photometric approximation rather than a reconstruction of the
dataset's full image distribution.

Example for the RB-Y1 configuration in this repository::

    python recommend_camera_exposure_gain.py \
      --stats /home/nvidia/.cache/huggingface/lerobot/local/\
rby1-table-bussing-v3/meta/stats.json

Use --camera-key-map whenever camera labels differ from dataset feature names::

    --camera-key-map '{"camera1":"observation.images.front",\
"camera2":"observation.images.left","camera3":"observation.images.right"}'
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


# This mapping was verified against the repository's RB-Y1 launch configuration
# and its local recording command history.  Override it with --cameras after any
# camera remount or replacement.
DEFAULT_CAMERAS = {
    "front": "260322274450",
    "left": "260322274992",
    "right": "260322276006",
}

# Score weights.  Available photometric terms are re-normalized to sum to one,
# which makes mean+std-only stats a supported fallback.
PHOTOMETRIC_WEIGHTS = {
    "mean": 0.30,
    "q50": 0.25,
    "central_range": 0.20,
    "std": 0.15,
    "outer_range": 0.10,
}
CLIPPING_WEIGHT = 0.25
GAIN_PENALTY_WEIGHT = 0.02
ALLOWED_CLIPPED_RATIO = 0.01
DARK_THRESHOLD = 5.0 / 255.0
BRIGHT_THRESHOLD = 250.0 / 255.0
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
STAT_FIELDS = ("mean", "q50", "q10", "q90", "std", "q01", "q99")
QUANTILE_FIELDS = ("q01", "q10", "q50", "q90", "q99")


@dataclass(frozen=True)
class OptionRange:
    minimum: float
    maximum: float
    step: float
    default: float


@dataclass
class TargetStats:
    dataset_key: str
    source_scale: float
    rgb: dict[str, np.ndarray]
    available_fields: list[str]
    warnings: list[str] = field(default_factory=list)

    def luminance(self, stat_name: str) -> float:
        return float(np.dot(self.rgb[stat_name], LUMA_WEIGHTS))

    def serializable(self) -> dict[str, Any]:
        return {
            "dataset_key": self.dataset_key,
            "source_scale": self.source_scale,
            "available_fields": self.available_fields,
            "rgb": {name: values.tolist() for name, values in self.rgb.items()},
            "luminance": {
                name: self.luminance(name) for name in self.available_fields
            },
            "warnings": self.warnings,
        }


@dataclass
class CaptureStats:
    rgb_mean: list[float]
    rgb_std: list[float]
    rgb_q01: list[float]
    rgb_q10: list[float]
    rgb_q50: list[float]
    rgb_q90: list[float]
    rgb_q99: list[float]
    luma_mean: float
    luma_std: float
    luma_q01: float
    luma_q10: float
    luma_q50: float
    luma_q90: float
    luma_q99: float
    near_black_ratio: float
    near_white_ratio: float


@dataclass
class CandidateResult:
    stage: str
    exposure: float
    gain: float
    score: float
    score_components: dict[str, float]
    captured: CaptureStats
    preview_rgb: np.ndarray = field(repr=False)

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("preview_rgb")
        return value


def _json_object(text: str, option_name: str) -> dict[str, str]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} must be valid JSON: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{option_name} must be a non-empty JSON object")
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise ValueError(f"{option_name} keys and values must all be strings")
    if any(not key.strip() or not item.strip() for key, item in value.items()):
        raise ValueError(f"{option_name} keys and values must not be empty")
    return {key.strip(): item.strip() for key, item in value.items()}


def _flatten_rgb(value: Any, field_name: str, dataset_key: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{dataset_key}.{field_name} is not a numeric RGB statistic"
        ) from exc
    if array.size != 3:
        raise ValueError(
            f"{dataset_key}.{field_name} must contain exactly three RGB values; "
            f"found shape {np.asarray(value).shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{dataset_key}.{field_name} contains NaN or infinity")
    return array


def discover_image_keys(stats_document: Mapping[str, Any]) -> list[str]:
    """Return image-like keys by inspecting values instead of assuming a schema."""
    found: list[str] = []
    for key, value in stats_document.items():
        if not isinstance(value, Mapping) or "mean" not in value:
            continue
        try:
            if np.asarray(value["mean"]).size == 3:
                found.append(str(key))
        except Exception:
            continue
    return sorted(found)


def _infer_scale(raw: Mapping[str, np.ndarray], dataset_key: str) -> tuple[float, list[str]]:
    values = np.concatenate(list(raw.values()))
    minimum = float(values.min())
    maximum = float(values.max())
    tolerance = 1e-6
    if minimum < -tolerance:
        raise ValueError(
            f"{dataset_key} contains an impossible negative RGB statistic ({minimum})"
        )
    if maximum <= 1.0 + tolerance:
        return 1.0, ["RGB scale inferred as 0..1 from the observed statistic range"]
    if maximum <= 255.0 + tolerance:
        return 255.0, ["RGB scale inferred as 0..255 from the observed statistic range"]
    raise ValueError(
        f"{dataset_key} contains an impossible RGB statistic ({maximum}); "
        "expected a 0..1 or 0..255 scale"
    )


def parse_target_stats(
    stats_document: Mapping[str, Any], dataset_key: str
) -> TargetStats:
    if dataset_key not in stats_document:
        available = discover_image_keys(stats_document)
        raise ValueError(
            f"Dataset camera key {dataset_key!r} is missing. "
            f"Available image keys: {available}"
        )
    entry = stats_document[dataset_key]
    if not isinstance(entry, Mapping):
        raise ValueError(f"{dataset_key} must map to an object in stats.json")
    if "mean" not in entry:
        raise ValueError(f"{dataset_key} is missing required field 'mean'")

    raw: dict[str, np.ndarray] = {}
    for name in STAT_FIELDS + ("min", "max"):
        if name in entry:
            raw[name] = _flatten_rgb(entry[name], name, dataset_key)
    if "std" in raw and np.any(raw["std"] < 0):
        raise ValueError(f"{dataset_key}.std contains a negative value")
    scale, warnings = _infer_scale(raw, dataset_key)
    normalized = {name: np.clip(values / scale, 0.0, 1.0) for name, values in raw.items()}

    if "std" in raw and np.any(raw["std"] > scale + 1e-6):
        raise ValueError(f"{dataset_key}.std exceeds the inferred RGB scale")
    for name in ("mean",) + QUANTILE_FIELDS + ("min", "max"):
        if name in raw and (np.any(raw[name] < -1e-6) or np.any(raw[name] > scale + 1e-6)):
            raise ValueError(f"{dataset_key}.{name} is outside the inferred 0..{scale:g} range")

    present_quantiles = [name for name in QUANTILE_FIELDS if name in normalized]
    for lower, upper in zip(present_quantiles, present_quantiles[1:]):
        if np.any(normalized[lower] > normalized[upper] + 1e-6):
            raise ValueError(
                f"{dataset_key} has non-monotonic quantiles: {lower} > {upper}"
            )
    if "min" in normalized and np.any(normalized["min"] > normalized["mean"] + 1e-6):
        raise ValueError(f"{dataset_key}.min is greater than mean")
    if "max" in normalized and np.any(normalized["max"] < normalized["mean"] - 1e-6):
        raise ValueError(f"{dataset_key}.max is less than mean")

    available = [name for name in STAT_FIELDS if name in normalized]
    if "std" not in normalized:
        warnings.append("std is absent; the score will omit its contrast term")
    if not all(name in normalized for name in ("q10", "q50", "q90")):
        warnings.append("central quantiles are incomplete; available score terms will be re-weighted")
    if not all(name in normalized for name in ("q01", "q99")):
        warnings.append("outer quantiles are incomplete; the outer-range term will be omitted")

    return TargetStats(dataset_key, scale, normalized, available, warnings)


def load_stats(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"stats.json does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"stats.json root must be a JSON object: {path}")
    return document


def resolve_camera_key_map(
    cameras: Mapping[str, str],
    stats_document: Mapping[str, Any],
    explicit_map: Mapping[str, str] | None,
) -> dict[str, str]:
    image_keys = discover_image_keys(stats_document)
    if explicit_map is not None:
        missing_labels = sorted(set(cameras) - set(explicit_map))
        extra_labels = sorted(set(explicit_map) - set(cameras))
        if missing_labels or extra_labels:
            raise ValueError(
                "--camera-key-map keys must exactly match --cameras keys; "
                f"missing={missing_labels}, extra={extra_labels}"
            )
        mapping = dict(explicit_map)
    else:
        mapping = {label: f"observation.images.{label}" for label in cameras}
        unresolved = {label: key for label, key in mapping.items() if key not in image_keys}
        if unresolved:
            raise ValueError(
                "Camera labels do not exactly match dataset image keys, so no mapping was guessed. "
                f"Unresolved defaults: {unresolved}. Available image keys: {image_keys}. "
                "Pass an explicit --camera-key-map JSON object."
            )
    duplicate_keys = sorted({key for key in mapping.values() if list(mapping.values()).count(key) > 1})
    if duplicate_keys:
        raise ValueError(f"--camera-key-map reuses dataset keys: {duplicate_keys}")
    missing_keys = sorted(set(mapping.values()) - set(image_keys))
    if missing_keys:
        raise ValueError(
            f"Mapped dataset keys are missing or are not RGB image stats: {missing_keys}. "
            f"Available image keys: {image_keys}"
        )
    return mapping


def snap_to_range(value: float, option_range: OptionRange) -> float:
    value = min(max(float(value), option_range.minimum), option_range.maximum)
    steps = round((value - option_range.minimum) / option_range.step)
    snapped = option_range.minimum + steps * option_range.step
    return min(max(snapped, option_range.minimum), option_range.maximum)


def unique_snapped(values: Iterable[float], option_range: OptionRange) -> list[float]:
    result: list[float] = []
    seen: set[float] = set()
    for value in values:
        snapped = snap_to_range(value, option_range)
        identity = round(snapped, 9)
        if identity not in seen:
            seen.add(identity)
            result.append(snapped)
    return result


def logarithmic_candidates(
    option_range: OptionRange, count: int, include: Sequence[float]
) -> list[float]:
    if option_range.minimum > 0:
        base = np.geomspace(option_range.minimum, option_range.maximum, count)
    else:
        base = np.linspace(option_range.minimum, option_range.maximum, count)
    return sorted(unique_snapped([*base, *include], option_range))


def linear_candidates(
    option_range: OptionRange, count: int, include: Sequence[float]
) -> list[float]:
    base = np.linspace(option_range.minimum, option_range.maximum, count)
    return sorted(unique_snapped([*base, *include], option_range))


def nearest_spacing(value: float, candidates: Sequence[float], minimum: float) -> float:
    distances = [abs(value - item) for item in candidates if not math.isclose(value, item)]
    return max(min(distances) / 3.0 if distances else minimum, minimum)


def compute_capture_stats(frames_rgb: Sequence[np.ndarray]) -> CaptureStats:
    if not frames_rgb:
        raise ValueError("At least one measurement frame is required")
    sampled: list[np.ndarray] = []
    for frame in frames_rgb:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError("RealSense RGB frames must be uint8 NumPy arrays")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB frame, got shape {frame.shape}")
        height, width = frame.shape[:2]
        scale = min(1.0, 160.0 / width, 120.0 / height)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        sampled.append(frame.astype(np.float64) / 255.0)
    pixels = np.concatenate([frame.reshape(-1, 3) for frame in sampled], axis=0)
    luma = pixels @ LUMA_WEIGHTS
    rgb_quantiles = np.quantile(pixels, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    luma_quantiles = np.quantile(luma, [0.01, 0.10, 0.50, 0.90, 0.99])
    return CaptureStats(
        rgb_mean=pixels.mean(axis=0).tolist(),
        rgb_std=pixels.std(axis=0).tolist(),
        rgb_q01=rgb_quantiles[0].tolist(),
        rgb_q10=rgb_quantiles[1].tolist(),
        rgb_q50=rgb_quantiles[2].tolist(),
        rgb_q90=rgb_quantiles[3].tolist(),
        rgb_q99=rgb_quantiles[4].tolist(),
        luma_mean=float(luma.mean()),
        luma_std=float(luma.std()),
        luma_q01=float(luma_quantiles[0]),
        luma_q10=float(luma_quantiles[1]),
        luma_q50=float(luma_quantiles[2]),
        luma_q90=float(luma_quantiles[3]),
        luma_q99=float(luma_quantiles[4]),
        near_black_ratio=float(np.mean(luma <= DARK_THRESHOLD)),
        near_white_ratio=float(np.mean(luma >= BRIGHT_THRESHOLD)),
    )


def score_capture(
    captured: CaptureStats,
    target: TargetStats,
    gain: float,
    gain_range: OptionRange,
) -> tuple[float, dict[str, float]]:
    differences: dict[str, float] = {
        "mean": abs(captured.luma_mean - target.luminance("mean")),
    }
    if "q50" in target.rgb:
        differences["q50"] = abs(captured.luma_q50 - target.luminance("q50"))
    if "q10" in target.rgb and "q90" in target.rgb:
        target_range = target.luminance("q90") - target.luminance("q10")
        differences["central_range"] = abs(
            (captured.luma_q90 - captured.luma_q10) - target_range
        )
    if "std" in target.rgb:
        differences["std"] = abs(captured.luma_std - target.luminance("std"))
    if "q01" in target.rgb and "q99" in target.rgb:
        target_range = target.luminance("q99") - target.luminance("q01")
        differences["outer_range"] = abs(
            (captured.luma_q99 - captured.luma_q01) - target_range
        )

    active_weight = sum(PHOTOMETRIC_WEIGHTS[name] for name in differences)
    photometric_terms = {
        name: difference * PHOTOMETRIC_WEIGHTS[name] / active_weight
        for name, difference in differences.items()
    }
    photometric = sum(photometric_terms.values())
    clipping = CLIPPING_WEIGHT * (
        max(0.0, captured.near_black_ratio - ALLOWED_CLIPPED_RATIO)
        + max(0.0, captured.near_white_ratio - ALLOWED_CLIPPED_RATIO)
    )
    gain_span = gain_range.maximum - gain_range.minimum
    normalized_gain = (gain - gain_range.minimum) / gain_span if gain_span else 0.0
    gain_penalty = GAIN_PENALTY_WEIGHT * normalized_gain
    components = {
        **{f"photometric_{name}": value for name, value in photometric_terms.items()},
        "photometric_total": photometric,
        "clipping_penalty": clipping,
        "gain_penalty": gain_penalty,
        "normalized_gain": normalized_gain,
    }
    return photometric + clipping + gain_penalty, components


class CameraTuner:
    """Thin owner for the repository's installed LeRobot RealSenseCamera."""

    def __init__(
        self,
        label: str,
        serial: str,
        target: TargetStats,
        *,
        width: int,
        height: int,
        fps: int,
        settle_frames: int,
        sample_frames: int,
        max_candidates: int | None,
    ) -> None:
        self.label = label
        self.serial = serial
        self.target = target
        self.width = width
        self.height = height
        self.fps = fps
        self.settle_frames = settle_frames
        self.sample_frames = sample_frames
        self.max_candidates = max_candidates
        self.camera: Any = None
        self.sensor: Any = None
        self.rs: Any = None
        self.exposure_range: OptionRange | None = None
        self.gain_range: OptionRange | None = None
        self.original: dict[str, float] = {}
        self.candidates: list[CandidateResult] = []
        self._cache: dict[tuple[float, float], CandidateResult] = {}
        self.recommendation: CandidateResult | None = None
        self.baseline: CandidateResult | None = None

    @staticmethod
    def _option_range(sensor: Any, option: Any, label: str) -> OptionRange:
        if not sensor.supports(option):
            raise RuntimeError(f"Color sensor does not support required option: {label}")
        value = sensor.get_option_range(option)
        values = [float(value.min), float(value.max), float(value.step), float(value.default)]
        if not all(math.isfinite(item) for item in values):
            raise RuntimeError(f"Color sensor returned a non-finite {label} range: {values}")
        if value.min > value.max or value.step <= 0:
            raise RuntimeError(f"Color sensor returned an invalid {label} range: {values}")
        return OptionRange(*values)

    def connect(self) -> None:
        try:
            import pyrealsense2 as rs
            from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
            from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        except ImportError as exc:
            raise RuntimeError(
                "This utility requires the repository's LeRobot environment and pyrealsense2. "
                "Run it with /home/nvidia/miniforge3/envs/lerobot/bin/python."
            ) from exc

        self.rs = rs
        config = RealSenseCameraConfig(
            serial_number_or_name=self.serial,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )
        self.camera = RealSenseCamera(config)
        self.camera.connect(warmup=True)
        self.sensor = self.camera._get_color_sensor()

        for option, label in (
            (rs.option.enable_auto_exposure, "enable_auto_exposure"),
            (rs.option.exposure, "exposure"),
            (rs.option.gain, "gain"),
        ):
            if not self.sensor.supports(option):
                raise RuntimeError(
                    f"Camera {self.label} ({self.serial}) does not support required option {label}"
                )
        self.exposure_range = self._option_range(self.sensor, rs.option.exposure, "exposure")
        self.gain_range = self._option_range(self.sensor, rs.option.gain, "gain")
        self.original = {
            "auto_exposure": float(self.sensor.get_option(rs.option.enable_auto_exposure)),
            "exposure": float(self.sensor.get_option(rs.option.exposure)),
            "gain": float(self.sensor.get_option(rs.option.gain)),
        }

    def _set_manual(self, exposure: float, gain: float) -> tuple[float, float]:
        assert self.sensor is not None and self.rs is not None
        assert self.exposure_range is not None and self.gain_range is not None
        exposure = snap_to_range(exposure, self.exposure_range)
        gain = snap_to_range(gain, self.gain_range)
        self.sensor.set_option(self.rs.option.enable_auto_exposure, 0.0)
        self.sensor.set_option(self.rs.option.exposure, exposure)
        self.sensor.set_option(self.rs.option.gain, gain)
        return (
            float(self.sensor.get_option(self.rs.option.exposure)),
            float(self.sensor.get_option(self.rs.option.gain)),
        )

    def restore(self) -> None:
        if self.sensor is None or not self.original:
            return
        # Restore values while manual, then restore the original AE switch last.
        self.sensor.set_option(self.rs.option.enable_auto_exposure, 0.0)
        self.sensor.set_option(self.rs.option.exposure, self.original["exposure"])
        self.sensor.set_option(self.rs.option.gain, self.original["gain"])
        self.sensor.set_option(
            self.rs.option.enable_auto_exposure, self.original["auto_exposure"]
        )

    def apply_recommendation(self) -> None:
        if self.recommendation is None:
            raise RuntimeError(f"Camera {self.label} has no recommendation to apply")
        self._set_manual(self.recommendation.exposure, self.recommendation.gain)

    def disconnect(self) -> None:
        if self.camera is not None and self.camera.is_connected:
            self.camera.disconnect()

    def _capture(self) -> tuple[CaptureStats, np.ndarray]:
        frames: list[np.ndarray] = []
        for index in range(self.settle_frames + self.sample_frames):
            frame = self.camera.read()
            if index >= self.settle_frames:
                frames.append(np.ascontiguousarray(frame).copy())
        return compute_capture_stats(frames), frames[len(frames) // 2]

    def evaluate(self, exposure: float, gain: float, stage: str) -> CandidateResult | None:
        assert self.exposure_range is not None and self.gain_range is not None
        exposure = snap_to_range(exposure, self.exposure_range)
        gain = snap_to_range(gain, self.gain_range)
        key = (exposure, gain)
        if key in self._cache:
            return self._cache[key]
        if self.max_candidates is not None and len(self.candidates) >= self.max_candidates:
            return None

        actual_exposure, actual_gain = self._set_manual(exposure, gain)
        captured, preview = self._capture()
        score, components = score_capture(
            captured, self.target, actual_gain, self.gain_range
        )
        result = CandidateResult(
            stage,
            actual_exposure,
            actual_gain,
            score,
            components,
            captured,
            preview,
        )
        self._cache[key] = result
        self.candidates.append(result)
        print(
            f"  [{len(self.candidates):02d}] {stage:<16} "
            f"exposure={format_number(actual_exposure):>7} "
            f"gain={format_number(actual_gain):>4} score={score:.6f}"
        )
        return result

    def search(self) -> CandidateResult:
        assert self.exposure_range is not None and self.gain_range is not None
        original_exposure = self.original["exposure"]
        original_gain = self.original["gain"]
        self.baseline = self.evaluate(original_exposure, original_gain, "baseline")
        if self.baseline is None:
            raise RuntimeError("Candidate budget must allow the baseline measurement")

        low_gain = min(original_gain, self.gain_range.default)
        exposure_values = logarithmic_candidates(
            self.exposure_range,
            7,
            [original_exposure, self.exposure_range.default],
        )
        for exposure in exposure_values:
            self.evaluate(exposure, low_gain, "coarse_exposure")

        best = min(self.candidates, key=lambda item: item.score)
        gain_values = linear_candidates(
            self.gain_range,
            5,
            [original_gain, self.gain_range.default],
        )
        for gain in gain_values:
            self.evaluate(best.exposure, gain, "gain_search")

        best = min(self.candidates, key=lambda item: item.score)
        exposure_radius = nearest_spacing(
            best.exposure, exposure_values, self.exposure_range.step
        )
        gain_radius = nearest_spacing(best.gain, gain_values, self.gain_range.step)
        local_exposures = unique_snapped(
            [best.exposure - exposure_radius, best.exposure, best.exposure + exposure_radius],
            self.exposure_range,
        )
        local_gains = unique_snapped(
            [best.gain - gain_radius, best.gain, best.gain + gain_radius],
            self.gain_range,
        )
        for exposure in local_exposures:
            for gain in local_gains:
                self.evaluate(exposure, gain, "local_refinement")

        self.recommendation = min(self.candidates, key=lambda item: item.score)
        # Re-apply the winner so --apply and the final preview correspond to it.
        self._set_manual(self.recommendation.exposure, self.recommendation.gain)
        return self.recommendation

    def report(self, settings_action: str) -> dict[str, Any]:
        assert self.baseline is not None and self.recommendation is not None
        assert self.exposure_range is not None and self.gain_range is not None
        improvement = score_improvement(self.baseline.score, self.recommendation.score)
        return {
            "serial": self.serial,
            "dataset_key": self.target.dataset_key,
            "original_auto_exposure": self.original["auto_exposure"],
            "original_exposure": self.original["exposure"],
            "original_gain": self.original["gain"],
            "recommended_exposure": self.recommendation.exposure,
            "recommended_gain": self.recommendation.gain,
            "baseline_score": self.baseline.score,
            "recommended_score": self.recommendation.score,
            "improvement_percent": improvement,
            "settings_action": settings_action,
            "exposure_range": asdict(self.exposure_range),
            "gain_range": asdict(self.gain_range),
            "target": self.target.serializable(),
            "baseline": self.baseline.serializable(),
            "recommended": self.recommendation.serializable(),
            "candidate_count": len(self.candidates),
        }


def score_improvement(baseline: float, recommended: float) -> float:
    if baseline <= np.finfo(float).eps:
        return 0.0
    return 100.0 * (baseline - recommended) / baseline


def format_number(value: float) -> str:
    return str(int(round(value))) if math.isclose(value, round(value), abs_tol=1e-9) else f"{value:g}"


def format_rgb(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def save_rgb_png(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to save preview image: {path}")


def save_candidate_csv(path: Path, tuners: Sequence[CameraTuner]) -> None:
    fields = [
        "camera",
        "serial",
        "stage",
        "exposure",
        "gain",
        "score",
        "photometric_total",
        "clipping_penalty",
        "gain_penalty",
        "luma_mean",
        "luma_std",
        "luma_q10",
        "luma_q50",
        "luma_q90",
        "near_black_ratio",
        "near_white_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for tuner in tuners:
            for candidate in tuner.candidates:
                writer.writerow(
                    {
                        "camera": tuner.label,
                        "serial": tuner.serial,
                        "stage": candidate.stage,
                        "exposure": candidate.exposure,
                        "gain": candidate.gain,
                        "score": candidate.score,
                        "photometric_total": candidate.score_components["photometric_total"],
                        "clipping_penalty": candidate.score_components["clipping_penalty"],
                        "gain_penalty": candidate.score_components["gain_penalty"],
                        "luma_mean": candidate.captured.luma_mean,
                        "luma_std": candidate.captured.luma_std,
                        "luma_q10": candidate.captured.luma_q10,
                        "luma_q50": candidate.captured.luma_q50,
                        "luma_q90": candidate.captured.luma_q90,
                        "near_black_ratio": candidate.captured.near_black_ratio,
                        "near_white_ratio": candidate.captured.near_white_ratio,
                    }
                )


def print_target_summary(
    stats_path: Path,
    image_keys: Sequence[str],
    camera_key_map: Mapping[str, str],
    targets: Mapping[str, TargetStats],
) -> None:
    print(f"Stats: {stats_path}")
    print(f"Discovered RGB image keys: {list(image_keys)}")
    print(f"Camera key map: {dict(camera_key_map)}")
    for label, target in targets.items():
        print(f"\nCamera: {label}")
        print(f"  Dataset key : {target.dataset_key}")
        print(f"  Source scale: 0..{target.source_scale:g}")
        print(f"  Fields      : {target.available_fields}")
        print(f"  RGB mean    : {format_rgb(target.rgb['mean'])}")
        if "std" in target.rgb:
            print(f"  RGB std     : {format_rgb(target.rgb['std'])}")
        for warning in target.warnings:
            print(f"  Note        : {warning}")


def print_results(tuners: Sequence[CameraTuner], settings_action: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}")
    print("RB-Y1 RGB Exposure/Gain Recommendation")
    print(separator)
    for tuner in tuners:
        baseline = tuner.baseline
        recommendation = tuner.recommendation
        assert baseline is not None and recommendation is not None
        print(f"\nCamera: {tuner.label}")
        print(f"Serial: {tuner.serial}")
        print("\nDataset target")
        print(f"  RGB mean: {format_rgb(tuner.target.rgb['mean'])}")
        if "std" in tuner.target.rgb:
            print(f"  RGB std : {format_rgb(tuner.target.rgb['std'])}")
        print("\nCurrent")
        print(f"  Exposure: {format_number(baseline.exposure)}")
        print(f"  Gain: {format_number(baseline.gain)}")
        print(f"  RGB mean: {format_rgb(baseline.captured.rgb_mean)}")
        print(f"  Score: {baseline.score:.6f}")
        print("\nRecommended")
        print(f"  Exposure: {format_number(recommendation.exposure)}")
        print(f"  Gain: {format_number(recommendation.gain)}")
        print(f"  RGB mean: {format_rgb(recommendation.captured.rgb_mean)}")
        print(f"  Score: {recommendation.score:.6f}")
        print(f"\nImprovement: {score_improvement(baseline.score, recommendation.score):.2f}%")
        print(separator)
    print(f"Camera settings: {settings_action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure RealSense RGB frames and recommend exposure/gain that approximate "
            "LeRobot meta/stats.json. Global stats are only a compact photometric target; "
            "they do not preserve the dataset's full image distribution."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stats", type=Path, help="Path to LeRobot meta/stats.json")
    source.add_argument(
        "--dataset-root",
        type=Path,
        help="LeRobot dataset root; meta/stats.json is appended automatically",
    )
    parser.add_argument(
        "--cameras",
        default=json.dumps(DEFAULT_CAMERAS),
        help="JSON object mapping logical camera labels to RealSense serial numbers",
    )
    parser.add_argument(
        "--camera-key-map",
        help=(
            "JSON object mapping camera labels to exact stats.json keys. If omitted, only "
            "the exact observation.images.<label> mapping is accepted"
        ),
    )
    parser.add_argument(
        "--camera",
        action="append",
        help="Tune only this camera label; repeat to select more than one",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--settle-frames", type=int, default=5)
    parser.add_argument("--sample-frames", type=int, default=10)
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Maximum newly measured candidates per camera (for limited searches)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/camera_tuning")
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Leave recommended manual values active; default restores original settings",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate stats and mappings, print targets, and exit without opening cameras",
    )
    return parser


def _validate_positive_arguments(args: argparse.Namespace) -> None:
    for name in ("width", "height", "fps", "settle_frames", "sample_frames"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_candidates is not None and args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")
    if args.validate_only and args.apply:
        raise ValueError("--validate-only and --apply cannot be used together")


def main(argv: Sequence[str] | None = None) -> int:
    # Keep candidate progress visible when stdout is captured by a launcher.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_positive_arguments(args)
        stats_path = (
            args.stats.expanduser()
            if args.stats is not None
            else args.dataset_root.expanduser() / "meta" / "stats.json"
        )
        all_cameras = _json_object(args.cameras, "--cameras")
        if len(set(all_cameras.values())) != len(all_cameras):
            raise ValueError("--cameras contains duplicate serial numbers")

        if args.camera:
            unknown = sorted(set(args.camera) - set(all_cameras))
            if unknown:
                raise ValueError(
                    f"Unknown --camera label(s): {unknown}; configured cameras: {list(all_cameras)}"
                )
            selected = list(dict.fromkeys(args.camera))
        else:
            selected = list(all_cameras)
        cameras = {label: all_cameras[label] for label in selected}

        explicit_map = (
            _json_object(args.camera_key_map, "--camera-key-map")
            if args.camera_key_map
            else None
        )
        if explicit_map is not None:
            unknown_map_labels = sorted(set(explicit_map) - set(all_cameras))
            if unknown_map_labels:
                raise ValueError(
                    "--camera-key-map contains labels absent from --cameras: "
                    f"{unknown_map_labels}"
                )
            explicit_map = {
                label: explicit_map[label] for label in selected if label in explicit_map
            }
        stats_document = load_stats(stats_path)
        camera_key_map = resolve_camera_key_map(cameras, stats_document, explicit_map)
        targets = {
            label: parse_target_stats(stats_document, camera_key_map[label])
            for label in cameras
        }
    except ValueError as exc:
        parser.error(str(exc))

    image_keys = discover_image_keys(stats_document)
    print_target_summary(stats_path, image_keys, camera_key_map, targets)
    if args.validate_only:
        print("\nValidation successful; cameras were not opened and no settings were changed.")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir.expanduser() / timestamp
    tuners = [
        CameraTuner(
            label,
            serial,
            targets[label],
            width=args.width,
            height=args.height,
            fps=args.fps,
            settle_frames=args.settle_frames,
            sample_frames=args.sample_frames,
            max_candidates=args.max_candidates,
        )
        for label, serial in cameras.items()
    ]
    completed = False
    settings_action = (
        "recommended settings applied (auto exposure disabled)"
        if args.apply
        else "original settings restored"
    )

    def _interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt)
    try:
        print("\nOpening RealSense cameras (RGB only; no robot connection)...")
        for tuner in tuners:
            tuner.connect()
            print(
                f"  {tuner.label}: serial={tuner.serial}, "
                f"exposure_range={asdict(tuner.exposure_range)}, "
                f"gain_range={asdict(tuner.gain_range)}"
            )

        for tuner in tuners:
            print(f"\nSearching camera {tuner.label} ({tuner.serial})...")
            tuner.search()

        if args.apply:
            for tuner in tuners:
                tuner.apply_recommendation()
        else:
            for tuner in tuners:
                tuner.restore()

        run_dir.mkdir(parents=True, exist_ok=False)
        for tuner in tuners:
            assert tuner.baseline is not None and tuner.recommendation is not None
            save_rgb_png(run_dir / f"{tuner.label}_original.png", tuner.baseline.preview_rgb)
            save_rgb_png(
                run_dir / f"{tuner.label}_recommended.png",
                tuner.recommendation.preview_rgb,
            )
        save_candidate_csv(run_dir / "candidates.csv", tuners)
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stats_path": str(stats_path.resolve()),
            "mode": "apply" if args.apply else "recommendation_only",
            "settings_action": settings_action,
            "camera_key_map": camera_key_map,
            "capture": {
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "settle_frames": args.settle_frames,
                "sample_frames": args.sample_frames,
            },
            "score": {
                "photometric_weights": PHOTOMETRIC_WEIGHTS,
                "clipping_weight": CLIPPING_WEIGHT,
                "gain_penalty_weight": GAIN_PENALTY_WEIGHT,
                "allowed_clipped_ratio": ALLOWED_CLIPPED_RATIO,
                "dark_threshold": DARK_THRESHOLD,
                "bright_threshold": BRIGHT_THRESHOLD,
            },
            "cameras": {
                tuner.label: tuner.report(settings_action) for tuner in tuners
            },
            "limitation": (
                "LeRobot stats.json contains globally aggregated image statistics, not the "
                "full RGB distribution; recommendations are photometric approximations."
            ),
        }
        report_path = run_dir / "exposure_gain_recommendation.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        completed = True
        print_results(tuners, settings_action)
        print(f"Report: {report_path}")
        print(f"Candidates: {run_dir / 'candidates.csv'}")
        print(f"Previews: {run_dir}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted; restoring original camera settings...", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if not completed:
            for tuner in reversed(tuners):
                try:
                    tuner.restore()
                except Exception as exc:
                    print(
                        f"WARNING: failed to restore {tuner.label} ({tuner.serial}): {exc}",
                        file=sys.stderr,
                    )
        for tuner in reversed(tuners):
            try:
                tuner.disconnect()
            except Exception as exc:
                print(
                    f"WARNING: failed to disconnect {tuner.label} ({tuner.serial}): {exc}",
                    file=sys.stderr,
                )
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
