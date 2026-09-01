#!/usr/bin/env python3
"""CLI wrapper for the robot-free frozen SmolVLA noise diagnostic."""

import sys
from pathlib import Path

# Make direct source-tree execution work without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_async_inference.frozen_noise_analysis import main


if __name__ == "__main__":
    raise SystemExit(main())
