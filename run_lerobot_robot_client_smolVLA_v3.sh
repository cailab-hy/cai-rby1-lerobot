#!/usr/bin/env bash
set -euo pipefail

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
    "camera1": {
      "type": "intelrealsense",
      "serial_number_or_name": "260322274450",
      "fps": 15,
      "width": 640,
      "height": 480
    },
    "camera2": {
      "type": "intelrealsense",
      "serial_number_or_name": "260322274992",
      "fps": 15,
      "width": 640,
      "height": 480
    },
    "camera3": {
      "type": "intelrealsense",
      "serial_number_or_name": "260322276006",
      "fps": 15,
      "width": 640,
      "height": 480
    }
  }' \
  --policy_type=smolvla \
  --pretrained_name_or_path=/home/cai/rby1-lerobot/cai-rby1-lerobot/outputs/smolVLA_bs32_ViT_VLM_expert/checkpoints/020000/pretrained_model \
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

# Pick up the bowl and place it in the box.
# Pick up the cup and place it in the box.