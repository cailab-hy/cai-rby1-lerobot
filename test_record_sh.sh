#!/usr/bin/env bash

set -euo pipefail

# Store the dataset beside this script even when the script is launched
# from a different working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v lerobot-record >/dev/null 2>&1; then
  echo "Error: lerobot-record was not found. Run 'conda activate lerobot' first." >&2
  exit 1
fi

exec lerobot-record \
  --robot.type=rby1 \
  --robot.address=192.168.1.201:50051 \
  --robot.model=a \
  --robot.use_gripper=true \
  --robot.cameras='{
    "front": {
      "type": "intelrealsense",
      "serial_number_or_name": "260322274450",
      "fps": 15,
      "width": 640,
      "height": 480
    },
    "left": {
      "type": "intelrealsense",
      "serial_number_or_name": "260322274992",
      "fps": 15,
      "width": 640,
      "height": 480
    },
    "right": {
      "type": "intelrealsense",
      "serial_number_or_name": "260322276006",
      "fps": 15,
      "width": 640,
      "height": 480
    }
  }' \
  --teleop.type=rby1_leader_arm \
  --dataset.repo_id=local/rby1-table-bussing-v2 \
  --dataset.root="${SCRIPT_DIR}/datasets_test" \
  --dataset.push_to_hub=false \
  --dataset.single_task="Use the left arm to place the colored dishes into the box on the left." \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=30 \
  --dataset.fps=15 \
  --dataset.video=true \
  --dataset.rgb_encoder.vcodec=libsvtav1 \
  --dataset.rgb_encoder.pix_fmt=yuv420p \
  --dataset.rgb_encoder.video_backend=pyav \
  --dataset.rgb_encoder.extra_options='{}' \
  --display_data=false \
  --dataset.reset_time_s=3 \
  --resume=true

# resume : 처음만 false, 그 뒤는 true로 바꿔서 이어 녹화
