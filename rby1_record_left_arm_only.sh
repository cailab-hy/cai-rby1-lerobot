#!/usr/bin/env bash

set -euo pipefail

# Left-arm-only collection profile.
# Existing recording scripts and datasets are intentionally left untouched.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${SCRIPT_DIR}/datasets_realsense_v2"

# Override this at launch if the shared robot IP changes again:
#   RBY1_ROBOT_ADDRESS=192.168.x.x:50051 ./rby1_record_left_arm_only.sh
ROBOT_ADDRESS="${RBY1_ROBOT_ADDRESS:-192.168.1.201:50051}"

# A resumable dataset needs the complete metadata set. A failed first launch
# can leave only info.json; preserve that partial directory and start clean.
if [[ -f "${DATASET_ROOT}/meta/info.json" \
   && -f "${DATASET_ROOT}/meta/tasks.parquet" \
   && -f "${DATASET_ROOT}/meta/stats.json" ]]; then
  RESUME=true
elif [[ -e "${DATASET_ROOT}" ]]; then
  INCOMPLETE_ROOT="${DATASET_ROOT}.incomplete-$(date -u +%Y%m%d-%H%M%S)"
  mv -- "${DATASET_ROOT}" "${INCOMPLETE_ROOT}"
  echo "Incomplete dataset metadata was preserved at: ${INCOMPLETE_ROOT}" >&2
  RESUME=false
else
  RESUME=false
fi

if ! command -v lerobot-record >/dev/null 2>&1; then
  echo "Error: lerobot-record was not found. Run 'conda activate lerobot' first." >&2
  exit 1
fi

PYTHON_BIN="$(dirname -- "$(command -v lerobot-record)")/python"

# The true flags below retain both arms in the dataset schema. The private
# follower omits right-side physical commands, and the private leader runtime
# addresses only the physical-left IDs 7..13 and tool 0x81.
echo "Follower: both arms move to the ready pose; only the left arm is controlled afterward."
echo "Leader: only the left arm moves to ready; the right arm stays untouched at its startup position."
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/rby1_record_left_arm_only.py" \
  --robot.type=rby1_both_data_left_control \
  --robot.address="${ROBOT_ADDRESS}" \
  --robot.model=a \
  --robot.use_torso=false \
  --robot.use_right_arm=true \
  --robot.use_left_arm=true \
  --robot.use_gripper=true \
  --robot.move_to_ready_on_connect=true \
  --robot.reset_right_arm_on_record=false \
  --robot.reset_left_arm_on_record=false \
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
  --teleop.type=rby1_left_leader_only_private \
  --teleop.use_torso=false \
  --teleop.use_right_arm=true \
  --teleop.use_left_arm=true \
  --teleop.use_gripper=true \
  --teleop.reset_right_arm_on_record=false \
  --teleop.reset_left_arm_on_record=false \
  --dataset.repo_id=local/rby1-table-bussing-v2 \
  --dataset.root="${DATASET_ROOT}" \
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
  --resume="${RESUME}"
