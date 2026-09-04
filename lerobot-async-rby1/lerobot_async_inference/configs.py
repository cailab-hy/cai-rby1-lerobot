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

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import torch

from lerobot.robots.config import RobotConfig

from .constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
    DEFAULT_ZMQ_TIMEOUT_MS,
    SUPPORTED_BACKENDS,
)
from .image_transport import validate_image_resize_scale


def cosine_ramp_alpha(overlap_index: int, overlap_count: int) -> float:
    """Return the old-to-new blend weight for one position in an overlap."""
    if overlap_count <= 0:
        raise ValueError(f"overlap_count must be positive, got {overlap_count}")
    if overlap_index < 0 or overlap_index >= overlap_count:
        raise ValueError(
            f"overlap_index must be in [0, {overlap_count}), got {overlap_index}"
        )

    s = (overlap_index + 1) / (overlap_count + 1)
    return 0.5 * (1.0 - math.cos(math.pi * s))


def cosine_ramp(
    old: torch.Tensor,
    new: torch.Tensor,
    *,
    overlap_index: int = 0,
    overlap_count: int = 1,
) -> torch.Tensor:
    """Blend one overlap action using its position in the complete overlap."""
    alpha = cosine_ramp_alpha(overlap_index, overlap_count)
    return (1.0 - alpha) * old + alpha * new


# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
    "cosine_ramp": cosine_ramp,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class JointTrajectoryLimitsConfig:
    """Explicit command-space limits, keyed by LeRobot action feature name."""

    position_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_limits: dict[str, float] = field(default_factory=dict)
    acceleration_limits: dict[str, float] = field(default_factory=dict)
    jerk_limits: dict[str, float] = field(default_factory=dict)


@dataclass
class GripperTrajectoryConfig:
    mode: str = "passthrough"
    rate_limits: dict[str, float] = field(default_factory=dict)


@dataclass
class TrajectoryLoggingConfig:
    enabled: bool = True
    downsample: int = 1
    path: str = "logs/action_pipeline.jsonl"
    max_queue_size: int = 4096


@dataclass
class TrajectoryPostprocessConfig:
    """Opt-in high-rate trajectory generation after chunk aggregation.

    Live use remains opt-in. ``active_urdf`` requires an exact model/version
    pair and combines URDF hard position limits with a named operational
    profile. It never guesses a model or falls back to a generic URDF.
    """

    enabled: bool = False
    limits_source: str = "active_urdf"
    control_rate_hz: float | None = 500.0
    profile: str = "balanced"
    active_model: str | None = None
    urdf_version: str | None = None
    sdk_models_dir: str = "/home/nvidia/rby1-sdk/models"
    interpolation: str = "jerk_limited"
    arms: JointTrajectoryLimitsConfig = field(default_factory=JointTrajectoryLimitsConfig)
    grippers: GripperTrajectoryConfig = field(default_factory=GripperTrajectoryConfig)
    logging: TrajectoryLoggingConfig = field(default_factory=TrajectoryLoggingConfig)

    def __post_init__(self) -> None:
        if self.limits_source not in {"active_urdf", "explicit"}:
            raise ValueError(
                "trajectory_postprocess.limits_source must be active_urdf or explicit"
            )
        if self.profile not in {"mild", "balanced", "strong"}:
            raise ValueError(
                "trajectory_postprocess.profile must be mild, balanced, or strong"
            )
        if self.interpolation != "jerk_limited":
            raise ValueError("trajectory_postprocess.interpolation must be 'jerk_limited'")
        if self.enabled and (
            self.control_rate_hz is None
            or not math.isfinite(self.control_rate_hz)
            or self.control_rate_hz <= 0
        ):
            raise ValueError(
                "trajectory_postprocess.control_rate_hz must be explicitly set and positive when enabled"
            )
        if self.enabled and self.limits_source == "active_urdf" and (
            not self.active_model or not self.urdf_version
        ):
            raise ValueError(
                "active_urdf limits require explicit active_model and urdf_version; "
                "automatic model guessing is disabled"
            )
        if self.enabled and self.limits_source == "active_urdf" and not self.sdk_models_dir.strip():
            raise ValueError("trajectory_postprocess.sdk_models_dir cannot be empty")
        if self.grippers.mode not in {"passthrough", "rate_limited"}:
            raise ValueError(
                "trajectory_postprocess.grippers.mode must be passthrough or rate_limited"
            )
        if self.logging.downsample <= 0:
            raise ValueError("trajectory_postprocess.logging.downsample must be positive")
        if self.logging.max_queue_size <= 0:
            raise ValueError("trajectory_postprocess.logging.max_queue_size must be positive")
        if self.logging.enabled and not self.logging.path.strip():
            raise ValueError("trajectory_postprocess.logging.path cannot be empty")


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "Target inference latency in seconds"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "Timeout for observation queue in seconds"}
    )

    # Optional, one-shot offline diagnostic capture. Keeping this disabled has
    # no effect on the inference path.
    dump_frozen_policy_batch: str | None = field(
        default=None,
        metadata={
            "help": "Save the first policy-ready (post-preprocessor) batch for offline diagnostics"
        },
    )

    # Optional multi-observation diagnostic. Disabled by default so the
    # production inference path does not clone tensors or create a writer.
    diagnostic_capture_policy_batches: bool = field(
        default=False,
        metadata={"help": "Capture every policy-ready batch and generated action chunk"},
    )
    diagnostic_capture_dir: str = field(
        default="outputs/near_grasp_capture",
        metadata={"help": "Directory for policy batch/chunk diagnostic captures"},
    )
    diagnostic_capture_max: int = field(
        default=50,
        metadata={"help": "Maximum diagnostic captures per policy-server process"},
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

        if self.diagnostic_capture_max <= 0:
            raise ValueError(
                "diagnostic_capture_max must be positive, "
                f"got {self.diagnostic_capture_max}"
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """Create a PolicyServerConfig from a dictionary."""
        return cls(**config_dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "inference_latency": self.inference_latency,
            "logging": self.logging,
        }


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """


    # Robot configuration (for CLI usage - robot instance will be created from this)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk"})

    # Remote inference backend configuration
    backend: str = field(
        default="grpc",
        metadata={"help": f"Remote backend to use. Options: {SUPPORTED_BACKENDS}"},
    )

    # Policy configuration
    policy_type: str = field(default="act", metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(default="dummy", metadata={"help": "Pretrained model name or path"})


    # Task instruction for the robot to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})
    client_device: str = field(
        default="cpu",
        metadata={
            "help": "Device to move actions to after receiving from server (e.g., for downstream planners)"
        },
    )

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    image_resize_scale: float = field(
        default=1.0,
        metadata={"help": "Scale applied to camera images immediately before network transport"},
    )
    jpeg_compression: bool = field(
        default=False,
        metadata={"help": "Compress camera images as JPEG immediately before network transport"},
    )
    save_camera_images: bool = field(
        default=False,
        metadata={"help": "Save camera images sent by the client in a background writer"},
    )
    camera_image_log_dir: str = field(
        default="logs/camera_capture",
        metadata={"help": "Base directory for timestamped camera-image capture runs"},
    )
    camera_image_save_every_n: int = field(
        default=1,
        metadata={"help": "Save every Nth observation sent to the policy server"},
    )
    image_crop_params: dict[str, tuple[int, int, int, int]] = field(
        default_factory=dict,
        metadata={
            "help": "Optional per-camera crop parameters as (top, left, height, width), applied on the client before sending observations"
        },
    )
    zmq_timeout_ms: int = field(
        default=DEFAULT_ZMQ_TIMEOUT_MS,
        metadata={"help": "ZMQ send/recv timeout in milliseconds for the GR00T backend"},
    )
    front_camera_key: str = field(
        default="front",
        metadata={"help": "Robot observation key mapped to the front camera"},
    )
    right_wrist_camera_key: str = field(
        default="right",
        metadata={"help": "Robot observation key mapped to the right wrist camera"},
    )
    left_wrist_camera_key: str = field(
        default="left",
        metadata={"help": "Robot observation key mapped to the left wrist camera"},
    )

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )
    trajectory_postprocess: TrajectoryPostprocessConfig = field(
        default_factory=TrajectoryPostprocessConfig,
        metadata={"help": "Opt-in jerk-limited high-rate action post-processing"},
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )
    timing_diagnostics: bool = field(
        default=False,
        metadata={"help": "Enable low-overhead control-loop timing diagnostics"},
    )
    rtc_enabled: bool = field(
        default=False,
        metadata={"help": "Enable inference-time guided Real-Time Chunking for SmolVLA"},
    )
    rtc_mode: str = field(
        default="guided",
        metadata={"help": "RTC mode (only guided is valid for ordinary SmolVLA checkpoints)"},
    )
    rtc_execution_horizon: int = field(
        default=10, metadata={"help": "RTC prefix guidance execution horizon in frames"}
    )
    rtc_max_guidance_weight: float = field(
        default=10.0, metadata={"help": "Maximum guided RTC correction weight"}
    )
    rtc_prefix_attention_schedule: str = field(
        default="EXP", metadata={"help": "RTC prefix attention schedule: ZEROS, ONES, LINEAR, or EXP"}
    )
    rtc_diagnostics_dir: str = field(
        default="outputs",
        metadata={"help": "Server-side directory for rtc_diagnostics_<timestamp>.jsonl"},
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if isinstance(self.trajectory_postprocess, dict):
            trajectory_data = dict(self.trajectory_postprocess)
            if isinstance(trajectory_data.get("arms"), dict):
                trajectory_data["arms"] = JointTrajectoryLimitsConfig(
                    **trajectory_data["arms"]
                )
            if isinstance(trajectory_data.get("grippers"), dict):
                trajectory_data["grippers"] = GripperTrajectoryConfig(
                    **trajectory_data["grippers"]
                )
            if isinstance(trajectory_data.get("logging"), dict):
                trajectory_data["logging"] = TrajectoryLoggingConfig(
                    **trajectory_data["logging"]
                )
            self.trajectory_postprocess = TrajectoryPostprocessConfig(**trajectory_data)

        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}, got {self.backend!r}")

        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if self.backend == "grpc":
            if not self.policy_type:
                raise ValueError("policy_type cannot be empty when backend='grpc'")

            if not self.pretrained_name_or_path:
                raise ValueError("pretrained_name_or_path cannot be empty when backend='grpc'")

            if not self.policy_device:
                raise ValueError("policy_device cannot be empty when backend='grpc'")

        if not self.client_device:
            raise ValueError("client_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if (
            self.trajectory_postprocess.enabled
            and self.trajectory_postprocess.control_rate_hz < self.fps
        ):
            raise ValueError(
                "trajectory_postprocess.control_rate_hz must be at least the policy fps"
            )

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.rtc_mode.lower() != "guided":
            raise ValueError("Only --rtc_mode=guided is supported for this SmolVLA checkpoint")
        self.rtc_mode = self.rtc_mode.lower()
        if self.rtc_execution_horizon <= 0:
            raise ValueError("rtc_execution_horizon must be positive")
        if not math.isfinite(self.rtc_max_guidance_weight) or self.rtc_max_guidance_weight <= 0:
            raise ValueError("rtc_max_guidance_weight must be finite and positive")
        self.rtc_prefix_attention_schedule = self.rtc_prefix_attention_schedule.upper()
        if self.rtc_prefix_attention_schedule not in {"ZEROS", "ONES", "LINEAR", "EXP"}:
            raise ValueError(
                "rtc_prefix_attention_schedule must be one of ZEROS, ONES, LINEAR, EXP"
            )
        if self.rtc_enabled and self.backend != "grpc":
            raise ValueError("RTC is supported only by the gRPC SmolVLA backend")

        validate_image_resize_scale(self.image_resize_scale)

        if self.save_camera_images and not self.camera_image_log_dir.strip():
            raise ValueError("camera_image_log_dir cannot be empty when camera image saving is enabled")
        if self.camera_image_save_every_n <= 0:
            raise ValueError("camera_image_save_every_n must be positive")

        if self.zmq_timeout_ms <= 0:
            raise ValueError(f"zmq_timeout_ms must be positive, got {self.zmq_timeout_ms}")

        if not self.front_camera_key:
            raise ValueError("front_camera_key cannot be empty")

        if not self.right_wrist_camera_key:
            raise ValueError("right_wrist_camera_key cannot be empty")

        if not self.left_wrist_camera_key:
            raise ValueError("left_wrist_camera_key cannot be empty")

        normalized_crop_params = {}
        for key, value in self.image_crop_params.items():
            if len(value) != 4:
                raise ValueError(
                    f"image_crop_params['{key}'] must have four values (top, left, height, width), got {value}"
                )
            top, left, height, width = (int(v) for v in value)
            if top < 0 or left < 0:
                raise ValueError(
                    f"image_crop_params['{key}'] must use non-negative top/left offsets, got {value}"
                )
            if height <= 0 or width <= 0:
                raise ValueError(
                    f"image_crop_params['{key}'] must use positive height/width, got {value}"
                )
            normalized_crop_params[key] = (top, left, height, width)
        self.image_crop_params = normalized_crop_params
        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """Create a RobotClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "backend": self.backend,
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "client_device": self.client_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "image_resize_scale": self.image_resize_scale,
            "jpeg_compression": self.jpeg_compression,
            "save_camera_images": self.save_camera_images,
            "camera_image_log_dir": self.camera_image_log_dir,
            "camera_image_save_every_n": self.camera_image_save_every_n,
            "image_crop_params": self.image_crop_params,
            "zmq_timeout_ms": self.zmq_timeout_ms,
            "front_camera_key": self.front_camera_key,
            "left_wrist_camera_key": self.left_wrist_camera_key,
            "right_wrist_camera_key": self.right_wrist_camera_key,
            "task": self.task,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "timing_diagnostics": self.timing_diagnostics,
            "aggregate_fn_name": self.aggregate_fn_name,
            "trajectory_postprocess": asdict(self.trajectory_postprocess),
            "rtc_enabled": self.rtc_enabled,
            "rtc_mode": self.rtc_mode,
            "rtc_execution_horizon": self.rtc_execution_horizon,
            "rtc_max_guidance_weight": self.rtc_max_guidance_weight,
            "rtc_prefix_attention_schedule": self.rtc_prefix_attention_schedule,
            "rtc_diagnostics_dir": self.rtc_diagnostics_dir,
        }
