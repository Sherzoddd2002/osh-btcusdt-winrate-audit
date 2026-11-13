from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research_v3 as rv


def market_frame(n: int = 160) -> pd.DataFrame:
    close = np.full(n, 100.0)
    open_ = np.full(n, 100.0)
    return pd.DataFrame({
        "candle_time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": open_,
        "high": np.full(n, 100.05),
        "low": np.full(n, 99.95),
        "close": close,
        "volume": np.full(n, 1000.0),
        "trade_count": np.full(n, 100.0),
        "quote_volume": np.full(n, 100000.0),
        "taker_buy_base_volume": np.full(n, 500.0),
        "taker_buy_quote_volume": np.full(n, 50000.0),
    })


class ResearchV3DataTests(unittest.TestCase):
    def test_valid_market_frame_is_normalized(self):
        frame = rv.validate_market_data(market_frame())
        self.assertEqual(len(frame), 160)
        self.assertEqual(str(frame["candle_time"].dt.tz), "UTC")

    def test_gap_is_rejected_instead_of_silently_dropping_rows(self):
        frame = market_frame()
        frame.loc[20, "candle_time"] += pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError, "Non-contiguous"):
            rv.validate_market_data(frame)

    def test_duplicate_timestamp_is_rejected(self):
        frame = market_frame()
        frame.loc[10, "candle_time"] = frame.loc[9, "candle_time"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            rv.validate_market_data(frame)

    def test_invalid_ohlc_is_rejected(self):
        frame = market_frame()
        frame.loc[10, "high"] = 99.0
        with self.assertRaisesRegex(ValueError, "OHLC"):
            rv.validate_market_data(frame)


class ResearchV3BarrierTests(unittest.TestCase):
    def test_first_return_does_not_wrap_from_last_close(self):
        frame = market_frame(140)
        frame.loc[len(frame) - 1, "close"] = 200.0
        with patch.object(rv, "VOL_WIN", 4):
            labels = rv.barrier_labels(frame, h=2, cost_ret=0.001)
        self.assertTrue(np.isnan(labels.loc[0, "sigma"]))
        # A wrapped np.roll return made this row spuriously finite.
        self.assertTrue(np.isnan(labels.loc[1, "sigma"]))

    def test_high_low_touch_is_detected_even_when_close_does_not_cross(self):
        frame = market_frame(40)
        frame.loc[6, "high"] = 100.30
        frame.loc[6, "low"] = 99.95
        with patch.object(rv, "VOL_WIN", 4):
            labels = rv.barrier_labels(frame, h=2, cost_ret=0.001)
        self.assertEqual(int(labels.loc[5, "label3"]), 1)
        self.assertEqual(labels.loc[5, "exit_reason"], "take_profit")
        self.assertAlmostEqual(labels.loc[5, "exit_ret"], 0.0015, places=7)

    def test_same_bar_double_touch_is_conservative_stop(self):
        frame = market_frame(40)
        frame.loc[6, "high"] = 100.30
        frame.loc[6, "low"] = 99.70
        with patch.object(rv, "VOL_WIN", 4):
            labels = rv.barrier_labels(frame, h=2, cost_ret=0.001)
        self.assertEqual(int(labels.loc[5, "label3"]), -1)
        self.assertEqual(labels.loc[5, "exit_reason"], "stop_loss")
        self.assertLess(labels.loc[5, "exit_ret"], 0)

    def test_stop_gap_is_filled_at_the_worse_open(self):
        frame = market_frame(40)
        frame.loc[7, "open"] = 99.50
        frame.loc[7, "high"] = 100.05
        frame.loc[7, "low"] = 99.40
        with patch.object(rv, "VOL_WIN", 4):
            labels = rv.barrier_labels(frame, h=2, cost_ret=0.001)
        self.assertEqual(int(labels.loc[5, "label3"]), -1)
        self.assertAlmostEqual(labels.loc[5, "exit_ret"], -0.005, places=7)


class ResearchV3CausalityTests(unittest.TestCase):
    def test_external_alignment_waits_until_two_minute_bar_is_closed(self):
        btc = pd.DataFrame({
            "candle_time": pd.to_datetime(
                ["2024-01-01T10:00:00Z", "2024-01-01T10:15:00Z"]
            )
        })
        external = pd.DataFrame({
            "datetime": pd.to_datetime([
                "2024-01-01T10:12:00Z",
                "2024-01-01T10:14:00Z",
                "2024-01-01T10:28:00Z",
            ]),
            "GLD_Close": [100.0, 200.0, 101.0],
        })
        features = rv.align_external_features(btc, external, max_age_min=60, source_bar_minutes=2)
        self.assertAlmostEqual(features.loc[0, "GLD_age_min"], np.log1p(1.0), places=7)
        self.assertTrue(np.isnan(features.loc[0, "GLD_ret1"]))
        # 10:14 source bar is only available at 10:16, after the 10:15 signal.
        self.assertAlmostEqual(features.loc[1, "GLD_ret1"], np.log(101.0 / 100.0), places=7)

    def test_unavailable_external_observation_is_flagged_stale(self):
        btc = pd.DataFrame({
            "candle_time": pd.to_datetime(["2024-01-01T10:00:00Z"])
        })
        external = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-01-01T10:14:00Z"]),
            "GLD_Close": [100.0],
        })
        features = rv.align_external_features(btc, external, source_bar_minutes=2)
        self.assertEqual(int(features.loc[0, "GLD_stale"]), 1)

    def test_probability_columns_are_aligned_by_class_label(self):
        class FakeModel:
            classes_ = np.array([2, 0, 1])

            def predict_proba(self, X):
                return np.array([[0.5, 0.2, 0.3]])

        result = rv.predict_proba_3class(FakeModel(), pd.DataFrame({"x": [1]}))
        np.testing.assert_allclose(result, np.array([[0.2, 0.3, 0.5]]))

    def test_selection_uses_frozen_threshold_and_not_test_rank(self):
        p_up = np.array([0.80, 0.80, 0.80, 0.10])
        p_down = np.zeros(4)
        returns = np.array([0.02, 0.03, 0.04, 0.01])
        idx, sides, net = rv.select_trades(
            p_up, p_down, returns, h=1, cost=0.001, thr_up=0.75, thr_dn=0.99
        )
        np.testing.assert_array_equal(idx, np.array([0, 1, 2]))
        np.testing.assert_array_equal(sides, np.array([1, 1, 1]))
        self.assertEqual(len(net), 3)

    def test_short_side_uses_opposite_underlying_return(self):
        p_up = np.array([0.1, 0.1, 0.1])
        p_down = np.array([0.9, 0.9, 0.1])
        returns = np.array([-0.01, -0.02, 0.0])
        idx, sides, net = rv.select_trades(
            p_up, p_down, returns, h=1, cost=0.001, thr_up=0.8, thr_dn=0.8
        )
        np.testing.assert_array_equal(idx, np.array([0, 1]))
        np.testing.assert_array_equal(sides, np.array([-1, -1]))
        self.assertTrue((net > 0).all())

    def test_non_overlapping_entries_respect_horizon(self):
        probabilities = np.ones(9)
        returns = np.full(9, 0.01)
        idx, _, _ = rv.select_trades(
            probabilities, np.zeros(9), returns,
            h=3, cost=0.001, thr_up=0.5, thr_dn=1.1
        )
        np.testing.assert_array_equal(idx, np.array([0, 3]))

    def test_empty_metrics_are_json_safe(self):
        result = rv.metrics(
            np.array([]), np.array([]), cost_bps=10, h=4,
            benchmark=np.array([]), n_bars=100,
        )
        self.assertEqual(result["trades"], 0)
        self.assertIsNone(result["expectancy_bps"])
        self.assertIsNone(result["profit_factor"])
        import json
        json.dumps(result, allow_nan=False)

    def test_bootstrap_is_reproducible(self):
        values = np.linspace(-0.002, 0.003, 100)
        a = rv.block_bootstrap_ci(values, block=20, reps=100, seed=11)
        b = rv.block_bootstrap_ci(values, block=20, reps=100, seed=11)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
