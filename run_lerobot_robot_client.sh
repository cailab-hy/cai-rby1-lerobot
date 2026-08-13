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
  }' \
  --policy_type=pi0 \
  --pretrained_name_or_path=/home/cai/rby1-lerobot/rby1-lerobot/outputs/pi0_training_v0.1/checkpoints/030000/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --image_resize_scale=0.5 \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.25 \
  --aggregate_fn_name=weighted_average \
  --arm_temporal_crossfade=true \
  --debug_chunk_transitions=false \
  --debug_weighted_aggregation=false \
  --task="Put the dish into the dish box with the left arm." \
  --fps=15
