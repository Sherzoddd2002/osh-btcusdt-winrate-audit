from __future__ import annotations

import unittest

import numpy as np

from selectivity_sweep_1d import (
    thresholds_for_pct,
    trade_quality_stats,
    wilson_interval_pct,
)


class SelectivityThresholdTests(unittest.TestCase):
    def test_more_selective_threshold_is_at_least_as_high(self):
        p = np.array([
            [0.10, 0.80, 0.10],
            [0.15, 0.70, 0.15],
            [0.20, 0.60, 0.20],
            [0.25, 0.50, 0.25],
            [0.30, 0.40, 0.30],
            [0.35, 0.30, 0.35],
            [0.40, 0.20, 0.40],
            [0.45, 0.10, 0.45],
        ])
        up_2, down_2 = thresholds_for_pct(p, 0.02)
        up_10, down_10 = thresholds_for_pct(p, 0.10)
        self.assertGreaterEqual(up_2, up_10)
        self.assertGreaterEqual(down_2, down_10)

    def test_threshold_rejects_invalid_percentile(self):
        p = np.array([[0.2, 0.6, 0.2]])
        with self.assertRaises(ValueError):
            thresholds_for_pct(p, 0.0)
        with self.assertRaises(ValueError):
            thresholds_for_pct(p, 1.0)

    def test_wilson_interval_is_bounded_and_contains_point_estimate(self):
        interval = wilson_interval_pct(60, 100)
        self.assertEqual(len(interval), 2)
        self.assertLess(interval[0], 60.0)
        self.assertGreater(interval[1], 60.0)
        self.assertGreaterEqual(interval[0], 0.0)
        self.assertLessEqual(interval[1], 100.0)

    def test_wilson_interval_is_null_without_trades(self):
        self.assertEqual(wilson_interval_pct(0, 0), [None, None])

    def test_net_payoff_drawdown_and_loss_streak(self):
        result = trade_quality_stats(np.array([0.010, -0.005, -0.010, 0.020]))
        self.assertAlmostEqual(result["avg_net_win_bps"], 150.0)
        self.assertAlmostEqual(result["avg_net_loss_bps"], 75.0)
        self.assertAlmostEqual(result["net_payoff_ratio"], 2.0)
        self.assertEqual(result["max_consecutive_losses"], 2)
        self.assertLess(result["max_trade_sequence_drawdown_pct"], 0.0)
        self.assertLess(result["win_rate_wilson95_pct"][0], 50.0)
        self.assertGreater(result["win_rate_wilson95_pct"][1], 50.0)

    def test_empty_trade_quality_is_json_friendly(self):
        result = trade_quality_stats(np.array([], dtype=float))
        self.assertEqual(result["max_consecutive_losses"], 0)
        self.assertIsNone(result["net_payoff_ratio"])
        self.assertEqual(result["win_rate_wilson95_pct"], [None, None])


if __name__ == "__main__":
    unittest.main()
