"""Fail-closed RB-Y1 arm limit loading from a versioned SDK URDF."""

from __future__ import annotations

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ARM_JOINT_NAMES = tuple(
    [f"right_arm_{index}" for index in range(7)]
    + [f"left_arm_{index}" for index in range(7)]
)
SUPPORTED_MODELS = {"a": "A", "m": "M"}
SI_UNIT_MARKER = "Nm, rad/s^2, rad/s, (min) rad, (max) rad"


@dataclass(frozen=True)
class ActiveURDFLimits:
    model: str
    version: str
    robot_name: str
    path: Path
    sha256: str
    joint_names: tuple[str, ...]
    position_limits: dict[str, tuple[float, float]]
    velocity_limits: dict[str, float]
    acceleration_limits: dict[str, float]
    units: dict[str, str]


@dataclass(frozen=True)
class OperationalProfile:
    name: str
    position_limits: dict[str, tuple[float, float]]
    velocity_limits: dict[str, float]
    acceleration_limits: dict[str, float]
    jerk_limits: dict[str, float]


_PROFILE_BY_INDEX: dict[str, dict[int, tuple[float, float, float]]] = {
    "mild": {
        0: (0.8, 2.5, 15.0),
        1: (1.0, 3.0, 20.0),
        2: (1.0, 3.0, 20.0),
        3: (1.0, 3.0, 20.0),
        4: (1.5, 4.0, 30.0),
        5: (1.5, 4.0, 30.0),
        6: (1.0, 3.0, 20.0),
    },
    "balanced": {
        0: (0.6, 1.5, 7.5),
        1: (0.8, 2.5, 12.5),
        2: (0.8, 2.5, 12.5),
        3: (0.8, 2.5, 12.5),
        4: (1.2, 3.0, 20.0),
        5: (1.2, 3.0, 20.0),
        6: (0.8, 2.0, 10.0),
    },
    "strong": {
        0: (0.5, 1.0, 5.0),
        1: (0.7, 2.0, 10.0),
        2: (0.7, 2.0, 10.0),
        3: (0.7, 2.0, 10.0),
        4: (1.0, 2.5, 15.0),
        5: (1.0, 2.5, 15.0),
        6: (0.7, 1.5, 7.5),
    },
}


def _finite_attribute(element: ET.Element, name: str, joint_name: str) -> float:
    raw = element.get(name)
    if raw is None:
        raise ValueError(f"URDF joint {joint_name!r} is missing limit attribute {name!r}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"URDF joint {joint_name!r} has non-numeric {name}={raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"URDF joint {joint_name!r} has non-finite {name}={raw!r}")
    return value


def versioned_urdf_path(models_dir: Path, model: str, version: str) -> Path:
    """Resolve only the exact versioned model path; never fall back to model.urdf."""
    normalized_model = str(model).lower().strip()
    normalized_version = str(version).lower().strip().removeprefix("v")
    if normalized_model not in SUPPORTED_MODELS:
        raise ValueError(
            f"unsupported RB-Y1 model {model!r}; expected one of {sorted(SUPPORTED_MODELS)}"
        )
    if not re.fullmatch(r"\d+\.\d+", normalized_version):
        raise ValueError(f"invalid RB-Y1 URDF version {version!r}; expected for example '1.2'")
    path = Path(models_dir) / f"rby1{normalized_model}" / "urdf" / f"model_v{normalized_version}.urdf"
    if not path.is_file():
        raise FileNotFoundError(
            f"exact RB-Y1 {normalized_model.upper()} v{normalized_version} URDF not found: {path}"
        )
    return path


def load_active_urdf_limits(models_dir: Path, model: str, version: str) -> ActiveURDFLimits:
    """Load all 14 arm limits and reject any model, version, name, or unit ambiguity."""
    normalized_model = str(model).lower().strip()
    normalized_version = str(version).lower().strip().removeprefix("v")
    path = versioned_urdf_path(models_dir, normalized_model, normalized_version)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    # URDF angles are SI by specification. RB-Y1's non-standard acceleration
    # field is accepted only when the SDK file explicitly documents its units.
    if text.count(SI_UNIT_MARKER) < len(ARM_JOINT_NAMES):
        raise ValueError(
            "URDF does not explicitly document arm limits as rad, rad/s, and rad/s^2"
        )
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"invalid URDF XML: {path}: {exc}") from exc

    expected_robot_name = f"RBY1_{SUPPORTED_MODELS[normalized_model]}_v{normalized_version}"
    robot_name = root.get("name", "")
    if robot_name != expected_robot_name:
        raise ValueError(
            f"URDF identity mismatch: expected robot name {expected_robot_name!r}, got {robot_name!r}"
        )

    joint_elements: dict[str, ET.Element] = {}
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        if name in ARM_JOINT_NAMES:
            if name in joint_elements:
                raise ValueError(f"duplicate URDF arm joint {name!r}")
            joint_elements[name] = joint
    missing = [name for name in ARM_JOINT_NAMES if name not in joint_elements]
    if missing:
        raise ValueError(f"URDF is missing required arm joints: {missing}")

    position: dict[str, tuple[float, float]] = {}
    velocity: dict[str, float] = {}
    acceleration: dict[str, float] = {}
    for name in ARM_JOINT_NAMES:
        joint = joint_elements[name]
        if joint.get("type") != "revolute":
            raise ValueError(f"URDF arm joint {name!r} must be revolute")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"URDF arm joint {name!r} has no limit element")
        lower = _finite_attribute(limit, "lower", name)
        upper = _finite_attribute(limit, "upper", name)
        maximum_velocity = _finite_attribute(limit, "velocity", name)
        maximum_acceleration = _finite_attribute(limit, "acceleration", name)
        if lower >= upper or maximum_velocity <= 0 or maximum_acceleration <= 0:
            raise ValueError(f"URDF arm joint {name!r} has invalid limits")
        position[name] = (lower, upper)
        velocity[name] = maximum_velocity
        acceleration[name] = maximum_acceleration

    return ActiveURDFLimits(
        model=normalized_model,
        version=normalized_version,
        robot_name=robot_name,
        path=path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        joint_names=ARM_JOINT_NAMES,
        position_limits=position,
        velocity_limits=velocity,
        acceleration_limits=acceleration,
        units={
            "position": "rad",
            "velocity": "rad/s",
            "acceleration": "rad/s^2",
            "jerk": "rad/s^3",
        },
    )


def build_operational_profile(
    name: str, urdf_limits: ActiveURDFLimits
) -> OperationalProfile:
    """Build a task profile and ensure it does not exceed manufacturer limits."""
    normalized = str(name).lower().strip()
    if normalized not in _PROFILE_BY_INDEX:
        raise ValueError(f"unknown trajectory profile {name!r}; expected {sorted(_PROFILE_BY_INDEX)}")
    velocity: dict[str, float] = {}
    acceleration: dict[str, float] = {}
    jerk: dict[str, float] = {}
    violations: list[str] = []
    for joint_name in ARM_JOINT_NAMES:
        joint_index = int(joint_name.rsplit("_", 1)[1])
        candidate_velocity, candidate_acceleration, candidate_jerk = _PROFILE_BY_INDEX[normalized][joint_index]
        velocity[joint_name] = candidate_velocity
        acceleration[joint_name] = candidate_acceleration
        jerk[joint_name] = candidate_jerk
        if candidate_velocity > urdf_limits.velocity_limits[joint_name]:
            violations.append(
                f"{joint_name} velocity {candidate_velocity} > URDF {urdf_limits.velocity_limits[joint_name]}"
            )
        if candidate_acceleration > urdf_limits.acceleration_limits[joint_name]:
            violations.append(
                f"{joint_name} acceleration {candidate_acceleration} > URDF {urdf_limits.acceleration_limits[joint_name]}"
            )
    if violations:
        raise ValueError("operational profile exceeds manufacturer limits: " + "; ".join(violations))
    return OperationalProfile(
        name=normalized,
        position_limits=dict(urdf_limits.position_limits),
        velocity_limits=velocity,
        acceleration_limits=acceleration,
        jerk_limits=jerk,
    )


def validate_arm_action_map(action_names: Mapping[str, object] | list[str] | tuple[str, ...]) -> None:
    """Require an exact and complete 14-joint arm feature map."""
    names = set(action_names)
    actual_arm_names = {name for name in names if name.startswith(("right_arm_", "left_arm_"))}
    expected = set(ARM_JOINT_NAMES)
    if actual_arm_names != expected:
        raise ValueError(
            "arm action/URDF joint map mismatch: "
            f"missing={sorted(expected - actual_arm_names)}, unexpected={sorted(actual_arm_names - expected)}"
        )
