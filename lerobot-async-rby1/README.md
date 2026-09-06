# LeRobot Async Inference (RB-Y1)

A standalone package that extracts LeRobot's `async_inference` module for RB-Y1.
It supports the GR00T-ZMQ / Pi0.5-ZMQ backends alongside the gRPC backend.

## Installation

```bash
pip install -e .   # this package
```

## Structure

| Module | Description |
|--------|-------------|
| `policy_server.py` | gRPC policy server |
| `robot_client.py` | Robot client (gRPC / groot_zmq / pi05_zmq) |
| `robot_replay.py` | Replay recorded actions (JSONL) |
| `trajectory.py` | Stateful jerk-limited command generation |
| `policy/groot_zmq.py` | GR00T N1.6 ZMQ client |
| `policy/pi05_zmq.py` | Pi0.5 ZMQ client |

## CLI

```bash
# Policy server
lerobot-policy-server --host=0.0.0.0 --port=8080

# Robot client (gRPC)
lerobot-robot-client --server_address=127.0.0.1:8080 --robot.type=rby1

# Robot client (GR00T ZMQ)
lerobot-robot-client --backend=groot_zmq --robot.type=rby1 --server_address=127.0.0.1:5555

# Action replay
lerobot-robot-replay --robot.type=rby1 --actions-file=actions.jsonl --fps=30
```

Running via `python -m lerobot_async_inference.robot_client` works the same way.

## Action aggregation and trajectory post-processing

The gRPC server assigns each chunk action an absolute timestep and a wall-clock
policy timestamp at the configured policy rate (15 Hz in the RB-Y1 launch
scripts). The client converts that timestamp once, at chunk receipt, into its
local monotonic clock domain. Queue overlap is matched by absolute timestep;
`cosine_ramp` remains exactly
`(1-alpha)*old + alpha*new`, where
`alpha=(1-cos(pi*(i+1)/(N+1)))/2`. Old-only prefixes and incoming-only tails
are retained. Actions already stale when received are discarded, and a delayed
control tick selects only the newest due waypoint rather than sending a burst.

RB-Y1 joint-mode action order is `right_arm_0..6`, `left_arm_0..6`, then
`right_gripper_0`, `left_gripper_0`. Arm positions are radians. Grippers use the
normalised LeRobot convention (1=open, 0=closed), so they are deliberately not
fed through the arm limiter. The default gripper mode is immediate passthrough;
an independent optional rate limiter is available.

Trajectory generation is disabled by default. See
`trajectory_postprocess.example.yaml`. With `limits_source=active_urdf`, live
use requires an exact `active_model` and `urdf_version`; the parser refuses a
generic URDF fallback and validates the complete 14-joint map and SI units.
Position, manufacturer velocity, and manufacturer acceleration limits come
from that versioned URDF. The named mild/balanced/strong task profile supplies
lower operational velocity/acceleration and jerk limits and is rejected if it
exceeds the URDF. `limits_source=explicit` still requires all four maps for all
enabled arm joints. The SDK model reports a 2 ms internal update period, but
the host-side 500 Hz loop still needs end-to-end timing validation;
`normal_min_time` remains an independent SDK trajectory setting.

The high-rate loop uses actual monotonic `dt`, starts from
`Rby1.get_joint_positions()` at reset, and falls back to the last command only
when a measurement read fails. Other robot adapters without a lightweight
reader fall back to `get_observation()`; if neither produces all arm joints and
there is no prior command, no command is sent. Call `client.reset_trajectory()`
after any external episode reset.

## gRPC camera image transport

The robot still captures each camera at the resolution and FPS configured under
`--robot.cameras`. The following client-only options transform camera images
after capture and immediately before gRPC serialization. The policy server
detects the transport metadata automatically and restores every image to its
original HWC RGB `uint8` shape before policy preprocessing.

Legacy behavior (the defaults):

```bash
lerobot-robot-client \
  ... \
  --image_resize_scale=1.0 \
  --jpeg_compression=false
```

Resize only (`640x480 -> 320x240 -> network -> 640x480`):

```bash
lerobot-robot-client \
  ... \
  --image_resize_scale=0.5 \
  --jpeg_compression=false
```

JPEG only (`640x480 -> JPEG -> network -> decode -> 640x480`):

```bash
lerobot-robot-client \
  ... \
  --image_resize_scale=1.0 \
  --jpeg_compression=true
```

Resize and JPEG
(`640x480 -> 320x240 -> JPEG -> network -> decode -> 640x480`):

```bash
lerobot-robot-client \
  ... \
  --image_resize_scale=0.5 \
  --jpeg_compression=true
```

`image_resize_scale` must satisfy `0 < scale <= 1.0`. JPEG quality is fixed at
85 and is intentionally not exposed as another CLI option. No matching policy
server option is needed:

```bash
lerobot-policy-server --host=0.0.0.0 --port=8080 --fps=15
```

### Client camera-image capture

The client can asynchronously save the exact camera images sent with inference
requests. When JPEG transport is enabled, the writer reuses the transmitted
JPEG bytes instead of encoding each image again.

```bash
lerobot-robot-client \
  ... \
  --jpeg_compression=true \
  --save_camera_images=true \
  --camera_image_log_dir=logs/camera_capture \
  --camera_image_save_every_n=1
```

Each run creates a timestamped directory containing one subdirectory per camera
and a `manifest.jsonl` that maps image paths to observation timestamps and
timesteps. `camera_image_save_every_n` counts inference observations, not every
control-loop frame. Disk writes happen on a bounded background queue; capture
sets are dropped with a warning if the writer cannot keep up.
