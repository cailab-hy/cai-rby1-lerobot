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
