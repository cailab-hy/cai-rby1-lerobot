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

## Frozen SmolVLA noise diagnostic

The policy server can save the first batch after transport decoding, normal
observation preparation, and the complete policy preprocessor. This is a
one-shot diagnostic capture; inference continues normally after the save.

```bash
lerobot-policy-server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=15 \
  --dump_frozen_policy_batch=/home/cai/frozen_smolvla_batch.pt
```

After stopping the robot rollout, run the robot-free analysis from this
package directory:

```bash
python tools/analyze_smolvla_frozen_noise.py \
  --checkpoint=/path/to/checkpoints/020000/pretrained_model \
  --frozen_batch=/home/cai/frozen_smolvla_batch.pt \
  --device=cuda \
  --num_runs=30 \
  --output_dir=outputs/frozen_noise_analysis
```

The default action grouping matches RB-Y1 checkpoint order: right arm `0:7`,
left arm `7:14`, and grippers `14:16`. Use the corresponding `--*_indices`
options if analyzing a checkpoint with a different layout.

The terminal and `diagnostic_answers.csv` explicitly answer Q1/Q2/Q3 and emit
one primary classification: C takes priority when fixed-noise repeatability is
broken; otherwise B is primary when fixed-noise trajectories remain rough,
then A when only random-noise variability is significant.

## Near-grasp multi-observation replay diagnostic

This optional mode captures every fully preprocessed, policy-ready batch and
the corresponding raw and robot-unit policy chunks before client-side queue
aggregation. It is disabled by default. With it disabled, no capture writer is
created and inference output/control behavior is unchanged.

Start the policy server on `192.168.1.9`:

```bash
lerobot-policy-server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=15 \
  --diagnostic_capture_policy_batches=true \
  --diagnostic_capture_dir=/home/cai/rby1-lerobot/cai-rby1-lerobot/outputs/near_grasp_capture \
  --diagnostic_capture_max=50
```

Start the unchanged 40K rollout from the robot computer (the camera serials
match `run_lerobot_robot_client_smolVLA_v3.sh`):

```bash
lerobot-robot-client \
  --backend=grpc \
  --server_address=192.168.1.9:8080 \
  --robot.type=rby1 \
  --robot.address=192.168.1.201:50051 \
  --robot.model=auto \
  --robot.use_right_arm=true \
  --robot.use_left_arm=true \
  --robot.use_torso=false \
  --robot.use_mobile_base=false \
  --robot.use_gripper=true \
  --robot.action_mode=joint \
  --robot.cameras='{
    "camera1":{"type":"intelrealsense","serial_number_or_name":"260322274450","fps":15,"width":640,"height":480},
    "camera2":{"type":"intelrealsense","serial_number_or_name":"260322274992","fps":15,"width":640,"height":480},
    "camera3":{"type":"intelrealsense","serial_number_or_name":"260322276006","fps":15,"width":640,"height":480}
  }' \
  --policy_type=smolvla \
  --pretrained_name_or_path=/home/cai/rby1-lerobot/cai-rby1-lerobot/outputs/smolVLA_bs32_ViT_VLM_expert_left_only/checkpoints/040000/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=cosine_ramp \
  --task="Pick up the bowl and place it in the box." \
  --fps=15 \
  --image_resize_scale=1.0 \
  --jpeg_compression=true \
  --timing_diagnostics=true
```

Stop the client immediately when the first near-grasp `+-+-` jitter appears,
or just before grasp if it does not appear. The final 3--5 completed captures
will then be the most useful ones. Stopping the policy server normally flushes
the background writer. Each capture contains `capture_NNN.pt`,
`raw_chunk_NNN.pt`, and `robot_chunk_NNN.pt`; `capture_metadata.jsonl` contains
checksums, timestamps, inference latency, shapes, and capture overhead.

Run the robot-free replay from the repository root:

```bash
python tools/analyze_near_grasp_chunk_replay.py \
  --checkpoint=/home/cai/rby1-lerobot/cai-rby1-lerobot/outputs/smolVLA_bs32_ViT_VLM_expert_left_only/checkpoints/040000/pretrained_model \
  --capture-dir=outputs/near_grasp_capture \
  --last-n=5 \
  --device=cuda \
  --random-runs-per-capture=20 \
  --output-dir=outputs/near_grasp_replay_analysis
```

Add `--final-actions=/path/to/final_actions_*.jsonl` for best-effort comparison
against the merged/executed stream. Alignment uses the policy timestamps at 15
Hz: observations separated by `d` frames compare `old[d:]` with `new[:-d]`.
The most diagnostic columns are fixed-noise `d2q_p95`/oscillation, random
across-run standard deviation, aligned consecutive fixed-chunk RMSE/direction
mismatch, and (when high-confidence final-action alignment exists) final versus
policy-generated roughness.
