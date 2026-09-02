from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from analyze_rby1_pregrasp_temporal_quality import (  # noqa: E402
    GripperDirection,
    aggregate_metrics,
    compute_window_metrics,
    detect_grasp_close,
    estimate_lag,
    lag_distribution_stats,
)


class TemporalQualitySyntheticTests(unittest.TestCase):
    def test_smooth_monotonic_has_no_oscillation(self) -> None:
        values = np.linspace(0, 1, 100)[:, None]
        metrics = compute_window_metrics(values, (0,), 0, len(values), 0.005)
        self.assertEqual(metrics.oscillation, 0.0)

    def test_alternating_has_high_oscillation(self) -> None:
        values = (((-1.0) ** np.arange(100)) * 0.1)[:, None]
        metrics = compute_window_metrics(values, (0,), 0, len(values), 0.005)
        self.assertGreater(metrics.oscillation, 0.95)

    def test_exact_positive_two_frame_lag(self) -> None:
        rng = np.random.default_rng(1)
        action = np.cumsum(rng.normal(0, 0.03, size=(150, 3)), axis=0)
        state = np.empty_like(action)
        state[:2] = action[0]
        state[2:] = action[:-2]
        result = estimate_lag(action, state, (0, 1, 2), (0, 1, 2), 0, 150, -3, 10, 0.005)
        self.assertEqual((result.best_position, result.best_corr, result.best_velocity_rmse), (2, 2, 2))

    def test_noisy_positive_two_frame_lag(self) -> None:
        rng = np.random.default_rng(2)
        action = np.cumsum(rng.normal(0, 0.03, size=(150, 3)), axis=0)
        state = np.empty_like(action)
        state[:2] = action[0]
        state[2:] = action[:-2]
        state += rng.normal(0, 0.001, size=state.shape)
        result = estimate_lag(action, state, (0, 1, 2), (0, 1, 2), 0, 150, -3, 10, 0.005)
        self.assertLessEqual(abs(result.representative - 2), 1)

    def test_different_episode_lags_are_variable(self) -> None:
        stats = lag_distribution_stats([0, 2, 4, 6])
        self.assertGreater(stats["p90"] - stats["p10"], 4)
        self.assertGreater(stats["std"], 2)

    def test_open_to_close_grasp(self) -> None:
        direction = GripperDirection(True, 0.0, 1.0, 1.0, 1.0)
        values = np.r_[np.ones(30), np.linspace(1, 0, 5), np.zeros(30)]
        result = detect_grasp_close(
            values,
            direction,
            hold_frames=3,
            transition_min_fraction=0.25,
            confirm_window_frames=15,
            min_close_fraction=0.5,
        )
        self.assertTrue(result.detected)
        self.assertIn(result.frame, range(30, 35))

    def test_no_close_is_not_detected(self) -> None:
        direction = GripperDirection(True, 0.0, 1.0, 1.0, 1.0)
        result = detect_grasp_close(
            np.ones(60),
            direction,
            hold_frames=3,
            transition_min_fraction=0.25,
            confirm_window_frames=15,
            min_close_fraction=0.5,
        )
        self.assertFalse(result.detected)
        self.assertIsNone(result.frame)

    def test_transient_close_then_reopen_is_rejected(self) -> None:
        direction = GripperDirection(True, 0.0, 1.0, 1.0, 1.0)
        values = np.r_[np.ones(8), [0.2, 0.0, 0.0], np.ones(49)]
        result = detect_grasp_close(
            values,
            direction,
            hold_frames=3,
            transition_min_fraction=0.25,
            confirm_window_frames=20,
            min_close_fraction=0.5,
        )
        self.assertFalse(result.detected)

    def test_episode_boundaries_do_not_leak(self) -> None:
        first = compute_window_metrics(np.array([[0.0], [1.0]]), (0,), 0, 2, 0.005)
        second = compute_window_metrics(np.array([[100.0], [101.0]]), (0,), 0, 2, 0.005)
        pooled = aggregate_metrics([first, second])
        self.assertEqual(pooled["delta_max"], 1.0)
        self.assertEqual(pooled["oscillation_evaluated_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
