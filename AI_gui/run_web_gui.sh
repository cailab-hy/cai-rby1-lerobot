#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then
  echo "Error: activate the lerobot environment first: conda activate lerobot" >&2
  exit 1
fi

exec python "${SCRIPT_DIR}/server.py" "$@"
