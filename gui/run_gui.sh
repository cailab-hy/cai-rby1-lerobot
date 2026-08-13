#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then
  echo "Error: activate the lerobot environment first: conda activate lerobot" >&2
  exit 1
fi

# A terminal opened through SSH or a non-desktop login often does not inherit
# the local Jetson desktop display. Reuse the logged-in user's Xwayland/X11
# session when it is available.
if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X0 ]]; then
  export DISPLAY=:0
fi

if [[ -n "${DISPLAY:-}" && -z "${XAUTHORITY:-}" ]]; then
  RUNTIME_XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
  USER_XAUTHORITY="${HOME}/.Xauthority"
  if [[ -r "${RUNTIME_XAUTHORITY}" ]]; then
    export XAUTHORITY="${RUNTIME_XAUTHORITY}"
  elif [[ -r "${USER_XAUTHORITY}" ]]; then
    export XAUTHORITY="${USER_XAUTHORITY}"
  fi
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "Error: no graphical display was found." >&2
  echo "Log in to the Jetson desktop, or reconnect SSH with X forwarding (ssh -X)." >&2
  exit 1
fi

exec python "${SCRIPT_DIR}/main.py" "$@"
