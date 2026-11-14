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


class SelectivitySweepFlowTests(unittest.TestCase):
    def test_three_thresholds_share_one_fold_model_and_are_reported_separately(self):
        from unittest.mock import patch
        import pandas as pd
        import selectivity_sweep_1d as sweep

        n = 1500
        frame = pd.DataFrame({
            "candle_time": pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC"),
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.1),
            "low": np.full(n, 99.9),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
            "trade_count": np.full(n, 100.0),
            "quote_volume": np.full(n, 100000.0),
            "taker_buy_base_volume": np.full(n, 500.0),
            "taker_buy_quote_volume": np.full(n, 50000.0),
        })
        labels = pd.DataFrame({
            "label3": np.zeros(n, dtype=np.int8),
            "exit_ret": np.full(n, 0.005),
            "sample_weight": np.ones(n),
            "valid": np.ones(n, dtype=bool),
        })
        probabilities = np.tile(np.array([0.05, 0.15, 0.80]), (n, 1))

        with (
            patch.object(sweep.pv, "make_features",
                         return_value=(pd.DataFrame({"feature": np.ones(n)}), ["feature"])),
            patch.object(sweep.rv, "barrier_labels", return_value=labels),
            patch.object(sweep.pv, "make_folds", return_value=[(0, 500, 502, 900)]),
            patch.object(sweep.rv, "fit_model", return_value=object()) as fit_mock,
            patch.object(sweep.rv, "predict_proba_3class",
                         side_effect=lambda model, X: probabilities[:len(X)]),
        ):
            result = sweep.run_selectivity_sweep(
                frame, selection_pcts=(0.02, 0.05, 0.10),
                cost_bps=10, folds_count=1, train_days=4,
                test_days=2, holdout_days=2,
            )

        self.assertEqual(set(result["selections"]), {"2%", "5%", "10%"})
        self.assertEqual(fit_mock.call_count, 1)
        for selection in result["selections"].values():
            self.assertEqual(len(selection["folds"]), 1)
            self.assertEqual(selection["calibrated_top_pct_per_side"] in (2.0, 5.0, 10.0), True)
            self.assertGreater(selection["total_oos_trades"], 0)
            self.assertEqual(selection["untouched_holdout"]["trades"], 1)
        self.assertFalse(result["metadata"]["live_orders"])



if __name__ == "__main__":
    unittest.main()
