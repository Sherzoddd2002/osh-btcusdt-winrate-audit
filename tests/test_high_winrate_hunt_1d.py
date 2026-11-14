from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import high_winrate_hunt_1d as hunt


def market_frame(n: int = 80) -> pd.DataFrame:
    pattern = np.resize(np.array([0.0010, -0.0006, 0.0008, -0.0011]), n)
    close = 100.0 * np.exp(np.cumsum(pattern))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.0001
    low = np.minimum(open_, close) * 0.9999
    return pd.DataFrame({
        "candle_time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    })


class AsymmetricBarrierTests(unittest.TestCase):
    def test_long_and_short_have_independent_holding_times(self):
        frame = market_frame()
        i = 20
        h = 3
        with patch.object(hunt, "VOL_WIN", 4):
            sigma = (
                np.log(frame["close"]).diff().rolling(4, min_periods=2).std().iloc[i]
                * np.sqrt(h)
            )
        entry = float(frame.loc[i + 1, "open"])
        # Long TP on the first future candle; short TP only on the second.
        frame.loc[i + 1, "high"] = entry * (1 + 0.70 * sigma)
        frame.loc[i + 1, "low"] = entry * (1 + 0.10 * sigma)
        frame.loc[i + 1, "close"] = entry * (1 + 0.20 * sigma)
        frame.loc[i + 2, "open"] = entry * (1 + 0.20 * sigma)
        frame.loc[i + 2, "high"] = entry * (1 + 0.30 * sigma)
        frame.loc[i + 2, "low"] = entry * (1 - 0.60 * sigma)
        frame.loc[i + 2, "close"] = entry * (1 - 0.40 * sigma)

        with patch.object(hunt, "VOL_WIN", 4):
            labels = hunt.asymmetric_barrier_labels(frame, h, 0.5, 1.0, vol_win=4, chunk_size=7)

        self.assertEqual(int(labels.loc[i, "long_label"]), 1)
        self.assertEqual(int(labels.loc[i, "short_label"]), 1)
        self.assertEqual(int(labels.loc[i, "long_holding_bars"]), 1)
        self.assertEqual(int(labels.loc[i, "short_holding_bars"]), 2)

    def test_tp_gap_does_not_assume_price_improvement(self):
        frame = market_frame()
        i = 20
        h = 3
        with patch.object(hunt, "VOL_WIN", 4):
            sigma = (
                np.log(frame["close"]).diff().rolling(4, min_periods=2).std().iloc[i]
                * np.sqrt(h)
            )
        entry = float(frame.loc[i + 1, "open"])
        frame.loc[i + 1, "high"] = entry * (1 + 0.10 * sigma)
        frame.loc[i + 1, "low"] = entry * (1 - 0.10 * sigma)
        frame.loc[i + 1, "close"] = entry
        # Gap above the long TP on the next candle open.
        frame.loc[i + 2, "open"] = entry * (1 + 1.20 * sigma)
        frame.loc[i + 2, "high"] = entry * (1 + 1.30 * sigma)
        frame.loc[i + 2, "low"] = entry * (1 + 1.10 * sigma)
        frame.loc[i + 2, "close"] = entry * (1 + 1.25 * sigma)

        with patch.object(hunt, "VOL_WIN", 4):
            labels = hunt.asymmetric_barrier_labels(frame, h, 0.5, 1.0, vol_win=4)

        self.assertEqual(int(labels.loc[i, "long_label"]), 1)
        self.assertAlmostEqual(
            float(labels.loc[i, "long_exit"]),
            0.5 * float(labels.loc[i, "sigma"]),
            places=7,
        )

    def test_short_trade_returns_are_measured_on_entry_notional(self):
        frame = market_frame()
        i = 20
        entry = float(frame.loc[i + 1, "open"])
        # A 10% fall should produce +10% PnL on the entry notional,
        # not +11.11% from the reciprocal-price formula.
        frame.loc[i + 1, "open"] = entry
        frame.loc[i + 1, "close"] = entry * 0.90
        frame.loc[i + 1, "high"] = entry * 1.001
        frame.loc[i + 1, "low"] = entry * 0.899

        labels = hunt.asymmetric_barrier_labels(
            frame, h=1, tp_mult=1000.0, sl_mult=1000.0, vol_win=4
        )
        self.assertEqual(int(labels.loc[i, "short_label"]), 0)
        self.assertAlmostEqual(float(labels.loc[i, "short_exit"]), 0.10, places=7)
        self.assertAlmostEqual(float(labels.loc[i, "short_horizon_exit"]), 0.10, places=7)

    def test_same_bar_double_touch_is_a_loss_for_both_sides(self):
        frame = market_frame()
        i = 20
        h = 2
        with patch.object(hunt, "VOL_WIN", 4):
            sigma = (
                np.log(frame["close"]).diff().rolling(4, min_periods=2).std().iloc[i]
                * np.sqrt(h)
            )
        entry = float(frame.loc[i + 1, "open"])
        frame.loc[i + 1, "high"] = entry * (1 + 2 * sigma)
        frame.loc[i + 1, "low"] = entry * (1 - 2 * sigma)
        frame.loc[i + 1, "close"] = entry

        with patch.object(hunt, "VOL_WIN", 4):
            labels = hunt.asymmetric_barrier_labels(frame, h, 0.5, 1.0, vol_win=4)
        self.assertEqual(int(labels.loc[i, "long_label"]), -1)
        self.assertEqual(int(labels.loc[i, "short_label"]), -1)


class FundingTests(unittest.TestCase):
    def test_time_window_features_use_settlement_events_not_repeated_bars(self):
        frame = pd.DataFrame({
            "candle_time": pd.date_range(
                "2024-01-01T00:00:00Z", periods=96, freq="15min"
            )
        })
        funding = pd.DataFrame({
            "funding_time": pd.to_datetime([
                "2024-01-01T00:00:00Z",
                "2024-01-01T08:00:00Z",
                "2024-01-01T16:00:00Z",
                "2024-01-02T00:00:00Z",
            ]),
            "funding_rate": [0.001, 0.003, -0.001, 0.005],
        })
        features = hunt.funding_features(frame, funding)
        self.assertAlmostEqual(features.loc[32, "fund_last"], 0.003, places=10)
        self.assertAlmostEqual(features.loc[32, "fund_mean_1d"], 0.002, places=10)
        self.assertAlmostEqual(features.loc[64, "fund_mean_1d"], 0.001, places=10)
        self.assertEqual(int(features.loc[32, "fund_available"]), 1)

    def test_funding_cashflow_uses_mark_proxy_and_conservative_boundaries(self):
        times = pd.to_datetime([
            "2024-01-01T00:00:00Z",
            "2024-01-01T08:00:00Z",
            "2024-01-01T16:00:00Z",
        ])
        ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        rates = np.array([0.001, -0.002, -0.001])
        marks = np.array([100.0, 110.0, 120.0])

        pnl = hunt.funding_pnl_for_trade(
            1, times[0], times[2], 100.0, ns, rates, marks
        )
        # Entry-boundary payment (-10 bps) + interior credit (+22 bps);
        # exit-boundary credit is deliberately ignored.
        self.assertAlmostEqual(pnl, 0.0012, places=10)

    def test_mark_proxy_uses_only_last_completed_candle(self):
        frame = pd.DataFrame({
            "candle_time": pd.to_datetime([
                "2024-01-01T23:30:00Z",
                "2024-01-01T23:45:00Z",
            ]),
            "close": [100.0, 101.0],
        })
        funding = pd.DataFrame({
            "funding_time": pd.to_datetime(["2024-01-01T23:52:00Z"]),
            "funding_rate": [0.001],
        })
        mapped = hunt.funding_mark_price_proxy(frame, funding)
        self.assertEqual(float(mapped.loc[0, "mark_price_proxy"]), 100.0)


class ReportingTests(unittest.TestCase):
    def test_fees_apply_to_entry_and_exit_notional(self):
        # At unchanged prices: maker 2 bps on both sides + 2 bps slip = 6 bps.
        self.assertAlmostEqual(hunt._execution_cost_fraction(0.0, 1, False), 0.0006)
        # A 10% profitable short exits at 90% of entry notional:
        # maker fees are 2 bps * (1.00 + 0.90), plus 2 bps slip.
        self.assertAlmostEqual(hunt._execution_cost_fraction(0.10, -1, False), 0.00058)
        # Taker stress is 5 bps * (1.00 + 0.90) = 9.5 bps.
        self.assertAlmostEqual(hunt._execution_cost_fraction(0.10, -1, True), 0.00095)

    def test_capital_path_respects_position_cap_and_reserve(self):
        result = hunt._capital_path(np.array([0.01, -0.005]), cap=25.0)
        self.assertEqual(result["position_cap_usd"], 25.0)
        self.assertEqual(result["cash_reserve_pct"], 10.0)
        self.assertAlmostEqual(result["final_balance_usd"], 100.125, places=2)
        self.assertFalse(result["ruined"])

    def test_metrics_report_win_rate_confidence_interval_and_capital_paths(self):
        result = hunt.trade_metrics(
            np.array([0.01, -0.005, 0.01]),
            np.array([0.011, -0.004, 0.011]),
            np.array([0.001, 0.001]),
            288,
        )
        self.assertEqual(result["trades"], 3)
        self.assertAlmostEqual(result["win_rate_pct"], 66.67, places=2)
        self.assertEqual(len(result["win_rate_ci95_pct"]), 2)
        self.assertEqual(
            [p["position_cap_usd"] for p in result["capital_paths"]],
            [25.0, 50.0, 100.0],
        )

    def test_funding_duplicates_and_bad_units_fail_closed(self):
        duplicate = pd.DataFrame({
            "funding_time": pd.to_datetime([
                "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"
            ]),
            "funding_rate": [0.001, 0.002],
        })
        with self.assertRaisesRegex(ValueError, "duplicate"):
            hunt._validate_funding(duplicate)

        bad_units = pd.DataFrame({
            "funding_time": pd.to_datetime([
                "2024-01-01T00:00:00Z", "2024-01-01T08:00:00Z"
            ]),
            "funding_rate": [0.001, 1.0],
        })
        with self.assertRaisesRegex(ValueError, "50%"):
            hunt._validate_funding(bad_units)


if __name__ == "__main__":
    unittest.main()
