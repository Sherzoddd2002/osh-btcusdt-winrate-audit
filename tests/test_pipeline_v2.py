from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import pipeline_v2 as pv


class PipelineV2HardeningTests(unittest.TestCase):
    def test_synthetic_loader_still_passes_strict_normalization(self):
        frame = pv.load_synthetic(1200, seed=7)
        self.assertEqual(len(frame), 1200)
        self.assertTrue(frame["candle_time"].is_monotonic_increasing)

    def test_normalizer_rejects_gap_without_deduplicating(self):
        frame = pv.load_synthetic(200, seed=7)
        frame = frame.drop(index=100).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "non-15-minute"):
            pv._normalise(frame)

    def test_normalizer_rejects_non_numeric_required_values(self):
        frame = pv.load_synthetic(200, seed=7)
        frame.loc[25, "volume"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing/non-numeric"):
            pv._normalise(frame)

    def test_evaluation_uses_fixed_train_threshold(self):
        probabilities = np.array([0.99, 0.80, 0.74, 0.20])
        y = np.array([1, 1, 0, 0])
        exit_returns = np.array([0.02, 0.03, 0.01, -0.01])
        hold_returns = np.array([0.01, 0.02, 0.00, -0.02])
        result = pv.evaluate_fold(
            probabilities, y, exit_returns, hold_returns,
            cost_bps=10, horizon=1, threshold=0.75,
        )
        self.assertEqual(result["trades"], 2)
        self.assertAlmostEqual(result["mean_gross_bps"], 250.0, places=6)

    def test_empty_threshold_selection_is_handled(self):
        result = pv.evaluate_fold(
            np.array([0.1, 0.2]), np.array([0, 0]),
            np.array([0.01, 0.02]), np.array([0.01, 0.02]),
            cost_bps=10, horizon=1, threshold=0.99,
        )
        self.assertEqual(result["trades"], 0)
        self.assertIsNone(result["mean_net_bps"])

    def test_high_low_same_bar_double_hit_counts_as_stop(self):
        n = 40
        frame = pd.DataFrame({
            "candle_time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.05),
            "low": np.full(n, 99.95),
            "close": np.full(n, 100.0),
        })
        frame.loc[6, "high"] = 100.30
        frame.loc[6, "low"] = 99.70
        with patch.object(pv, "EPS", 1e-12):
            labels = pv.triple_barrier_labels(
                frame, pd.DataFrame(index=frame.index),
                horizon=2, vol_window=4, cost_bps=10,
            )
        self.assertEqual(int(labels.loc[5, "label3"]), -1)
        self.assertLess(labels.loc[5, "exit_ret"], 0)


if __name__ == "__main__":
    unittest.main()
