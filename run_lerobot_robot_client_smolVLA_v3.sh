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
      "height": 480,
      "exposure": 6336,
      "gain": 16
    },
    "camera2": {
      "type": "intelrealsense",
      "serial_number_or_name": "260322274992",
      "fps": 15,
      "width": 640,
      "height": 480,
      "exposure": 6336,
      "gain": 16
    },
    "camera3": {
      "type": "intelrealsense",
      "serial_number_or_name": "260322276006",
      "fps": 15,
      "width": 640,
      "height": 480,
      "exposure": 6336,
      "gain": 16
    }
  }' \
  --policy_type=smolvla \
  --pretrained_name_or_path=/home/cai/rby1-lerobot/cai-rby1-lerobot/outputs/smolVLA_left_bs32_ViT_VLM_expert/checkpoints/020000/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=cosine_ramp \
  --task="Pick up the bowl and place it in the box." \
  --fps=15 \
  --image_resize_scale=1.0 \
  --jpeg_compression=true \
  --save_camera_images=true \
  --camera_image_save_every_n=1 \
  --robot.use_impedance=true \
  --robot.impedance_damping_ratio=1.0 \
  --save_camera_images=true \
  --camera_image_log_dir=logs/camera_capture \
  --camera_image_save_every_n=1 \
  --robot.use_impedance=true \
  --robot.impedance_damping_ratio=1.0

# --- Language task settings ---
# Pick up the bowl and place it in the box.
# Pick up the cup and place it in the box.

# --- aggregate_fn_name settings  ---
# weighted_average  # 새로 추론된 action을 70% 반영
# cosine_ramp       # action chunk 전체에 걸쳐 기존 action에서 새 action으로 부드럽게 전환하는 방식

# --- Img save settings ---
#   --save_camera_images=true \
#   --camera_image_log_dir=logs/camera_capture \
#   --camera_image_save_every_n=1 \
#   --robot.use_impedance=true \
#   --robot.impedance_damping_ratio=1.0

# --- Realsense camera settings ---
#       "exposure": 8333,
#       "gain": 16
