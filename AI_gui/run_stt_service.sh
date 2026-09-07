#!/usr/bin/env bash
# Speech recognition service. Run this on the machine with the microphone and
# an NVIDIA GPU (the operator's laptop), not on the Jetson.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then
  echo "Error: activate the lerobot environment first: conda activate lerobot" >&2
  exit 1
fi

exec python "${SCRIPT_DIR}/stt_service.py" "$@"
