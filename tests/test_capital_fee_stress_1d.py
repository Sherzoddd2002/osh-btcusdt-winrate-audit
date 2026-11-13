from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research_v3 as rv
import capital_fee_stress_1d as stress


def candles_frame(n: int = 1500) -> pd.DataFrame:
    return pd.DataFrame({
        "candle_time": pd.date_range(
            "2023-01-01", periods=n, freq="15min", tz="UTC"
        ),
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


def funding_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    times = pd.date_range(start.floor("h"), end.ceil("h"), freq="4h", tz="UTC")
    return pd.DataFrame({
        "symbol": "BTCUSDT",
        "fundingTime": np.asarray([pd.Timestamp(t).value // 1_000_000 for t in times], dtype="int64"),
        "fundingRate": np.full(len(times), 0.0001),
        "markPrice": np.full(len(times), 100.0),
    })


class FundingValidationTests(unittest.TestCase):
    def test_api_shaped_funding_history_is_normalized(self):
        raw = pd.DataFrame({
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "fundingTime": [1760000000000, 1760014400000],
            "fundingRate": ["0.0001", "-0.0001"],
            "markPrice": ["100000", "101000"],
        })
        result = stress.validate_funding_history(raw)
        self.assertEqual(len(result), 2)
        self.assertEqual(str(result["funding_time"].dt.tz), "UTC")
        self.assertAlmostEqual(float(result.loc[0, "funding_rate"]), 0.0001)
        self.assertAlmostEqual(float(result.loc[1, "mark_price"]), 101000.0)

    def test_duplicate_or_missing_mark_price_fails_closed(self):
        duplicate = pd.DataFrame({
            "fundingTime": [1760000000000, 1760000000000],
            "fundingRate": [0.0001, 0.0001],
            "markPrice": [100000, 100000],
        })
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            stress.validate_funding_history(duplicate)

        missing = pd.DataFrame({
            "fundingTime": [1760000000000],
            "fundingRate": [0.0001],
        })
        with self.assertRaisesRegex(ValueError, "markPrice"):
            stress.validate_funding_history(missing)

    def test_infinite_funding_rate_is_rejected(self):
        raw = pd.DataFrame({
            "fundingTime": [1760000000000],
            "fundingRate": [np.inf],
            "markPrice": [100000.0],
        })
        with self.assertRaisesRegex(ValueError, "finite"):
            stress.validate_funding_history(raw)

    def test_funding_series_with_large_gap_is_rejected(self):
        raw = pd.DataFrame({
            "fundingTime": [1760000000000, 1760040000000],
            "fundingRate": [0.0001, 0.0001],
            "markPrice": [100000, 100000],
        })
        with self.assertRaisesRegex(ValueError, "gap"):
            stress.validate_funding_history(raw)

    def test_long_pays_positive_funding_short_receives_it(self):
        entry = pd.Timestamp("2026-01-01T00:00:00Z")
        exit_bar_open = pd.Timestamp("2026-01-02T00:00:00Z")
        exit_time = exit_bar_open + pd.Timedelta(minutes=15)
        times = pd.to_datetime([
            "2026-01-01T00:00:00Z",  # ambiguous entry candle
            "2026-01-01T08:00:00Z",  # unambiguous during the trade
            "2026-01-02T00:15:00Z",  # ambiguous exit boundary
        ], utc=True)
        times_ns = np.asarray([pd.Timestamp(t).value for t in times], dtype=np.int64)
        rates = np.full(3, 0.001)
        marks = np.full(3, 100.0)

        long_cost, long_events, long_ambiguous = stress.trade_funding_cost(
            1, 100.0, entry, exit_time, exit_bar_open,
            times_ns, rates, marks,
        )
        short_cost, short_events, short_ambiguous = stress.trade_funding_cost(
            -1, 100.0, entry, exit_time, exit_bar_open,
            times_ns, rates, marks,
        )
        self.assertAlmostEqual(long_cost, 30.0)
        self.assertAlmostEqual(short_cost, -10.0)
        self.assertEqual(long_events, 3)
        self.assertEqual(long_ambiguous, 2)
        self.assertEqual(short_ambiguous, 2)


class ExecutionAndCapitalTests(unittest.TestCase):
    def test_commission_exit_fee_scales_with_exit_notional(self):
        trade = {
            "entry_price": 100.0,
            "exit_price": 101.0,
            "directional_gross_return": 0.01,
            "funding_cost_bps": 1.0,
        }
        result = stress.add_execution_costs(
            trade, entry_fee_bps=2.0, exit_fee_bps=5.0,
            slippage_bps=3.0, bnb_discount_pct=0.0,
        )
        self.assertAlmostEqual(result["commission_cost_bps"], 7.05)
        self.assertAlmostEqual(result["net_return_bps"], 88.95)

    def test_bnb_discount_applies_only_to_commission_not_slippage_or_funding(self):
        trade = {
            "entry_price": 100.0,
            "exit_price": 100.0,
            "directional_gross_return": 0.01,
            "funding_cost_bps": 1.0,
        }
        result = stress.add_execution_costs(
            trade, entry_fee_bps=2.0, exit_fee_bps=5.0,
            slippage_bps=3.0, bnb_discount_pct=10.0,
        )
        self.assertAlmostEqual(result["commission_cost_bps"], 6.3)
        self.assertAlmostEqual(result["net_return_bps"], 89.7)

    def test_100_dollar_cap_reserves_ten_percent_of_equity(self):
        trades = [{
            "directional_gross_return": 0.01,
            "commission_cost_bps": 10.0,
            "slippage_cost_bps": 2.0,
            "funding_cost_bps": 1.0,
            "entry_fee_bps": 5.0,
            "mae_return": -0.02,
        }]
        result = stress.simulate_capital(
            trades, capital_usdt=100.0, position_cap_usdt=100.0,
            cash_reserve_pct=10.0,
        )
        self.assertAlmostEqual(result["initial_notional_usdt"], 90.0)
        self.assertAlmostEqual(result["gross_pnl_usdt"], 0.90)
        self.assertAlmostEqual(result["commission_paid_usdt"], 0.09)
        self.assertAlmostEqual(result["slippage_cost_usdt"], 0.018)
        self.assertAlmostEqual(result["funding_paid_usdt"], 0.009)
        self.assertAlmostEqual(result["net_pnl_usdt"], 0.783)
        self.assertAlmostEqual(result["ending_balance_usdt"], 100.783)


class FullPipelineSmokeTests(unittest.TestCase):
    def test_fee_scenarios_share_fold_models_across_three_selectivity_thresholds(self):
        n = 1500
        frame = candles_frame(n)
        labels = pd.DataFrame({
            "label3": np.zeros(n, dtype=np.int8),
            "exit_ret": np.full(n, 0.005),
            "sample_weight": np.ones(n),
            "valid": np.ones(n, dtype=bool),
            "holding_bars": np.ones(n, dtype=float),
            "path_high": np.full(n, 100.1),
            "path_low": np.full(n, 99.9),
            "entry_price": np.full(n, 100.0),
        })
        probabilities = np.tile(np.array([0.05, 0.15, 0.80]), (n, 1))
        funding = funding_frame(frame["candle_time"].iloc[0], frame["candle_time"].iloc[-1])

        with (
            patch.object(
                stress.pv, "make_features",
                return_value=(pd.DataFrame({"feature": np.ones(n)}), ["feature"]),
            ),
            patch.object(stress.rv, "barrier_labels", return_value=labels),
            patch.object(stress.pv, "make_folds", return_value=[(0, 500, 502, 900)]),
            patch.object(stress.rv, "fit_model", return_value=object()) as fit_mock,
            patch.object(
                stress.rv, "predict_proba_3class",
                side_effect=lambda model, X: probabilities[:len(X)],
            ),
        ):
            result = stress.run_capital_fee_stress(
                frame, funding,
                selection_pcts=(0.02, 0.05, 0.10),
                slippage_grid=(0.0, 2.0, 5.0),
                maker_fee_bps=2.0, taker_fee_bps=5.0,
                capital_sizes=(25.0, 50.0, 100.0),
                folds_count=1, train_days=4, test_days=2, holdout_days=2,
            )

        self.assertEqual(fit_mock.call_count, 7)  # seven unique fee+slippage label costs
        self.assertEqual(len(result["scenarios"]), 9)  # three fee modes x three slippage settings
        scenario = next(iter(result["scenarios"].values()))
        self.assertEqual(set(scenario["selections"]), {"2%", "5%", "10%"})
        for selection in scenario["selections"].values():
            self.assertGreater(selection["oos"]["trades"], 0)
            self.assertIn("100", selection["oos"]["capital_scenarios"])
            self.assertEqual(selection["untouched_holdout"]["trades"], 1)
        self.assertFalse(result["metadata"]["live_orders"])


class BarrierDurationTests(unittest.TestCase):
    def test_barrier_labels_retain_holding_bars_and_path_extrema(self):
        frame = candles_frame(40)
        frame.loc[6, "high"] = 100.30
        frame.loc[6, "low"] = 99.90
        with patch.object(rv, "VOL_WIN", 4):
            labels = rv.barrier_labels(frame, h=2, cost_ret=0.001)
        self.assertEqual(int(labels.loc[5, "holding_bars"]), 1)
        self.assertAlmostEqual(float(labels.loc[5, "entry_price"]), 100.0)
        self.assertAlmostEqual(float(labels.loc[5, "path_high"]), 100.30)
        self.assertAlmostEqual(float(labels.loc[5, "path_low"]), 99.90)


if __name__ == "__main__":
    unittest.main()
