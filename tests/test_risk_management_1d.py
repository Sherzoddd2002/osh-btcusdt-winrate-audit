from __future__ import annotations

import unittest

import numpy as np

from risk_management_1d import (
    run_risk_budget_grid,
    simulate_risk_budget,
    summarize_trade_attribution,
)


class TradeAttributionTests(unittest.TestCase):
    def test_summary_splits_long_short_and_exit_reasons(self):
        result = summarize_trade_attribution(
            net_returns=np.array([0.01, -0.02, 0.03, -0.01]),
            sides=np.array([1, 1, -1, -1]),
            exit_reasons=np.array(["take_profit", "stop_loss", "time", "take_profit"]),
            mae_returns=np.array([-0.005, -0.025, -0.01, -0.004]),
            mfe_returns=np.array([0.012, 0.015, 0.04, 0.005]),
            holding_bars=np.array([1, 2, 4, 1]),
        )

        self.assertEqual(result["trades"], 4)
        self.assertEqual(result["win_rate_pct"], 50.0)
        self.assertEqual(result["by_side"]["LONG"]["trades"], 2)
        self.assertEqual(result["by_side"]["SHORT"]["trades"], 2)
        self.assertEqual(result["by_exit_reason"]["stop_loss"]["trades"], 1)
        self.assertEqual(result["exit_reason_share_pct"]["take_profit"], 50.0)

    def test_empty_attribution_is_json_friendly(self):
        empty = np.array([], dtype=float)
        result = summarize_trade_attribution(
            empty, np.array([], dtype=int), np.array([], dtype=str),
            empty, empty, empty,
        )
        self.assertEqual(result["trades"], 0)
        self.assertIsNone(result["expectancy_bps"])
        self.assertEqual(result["by_side"]["LONG"]["trades"], 0)

    def test_attribution_rejects_misaligned_arrays(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            summarize_trade_attribution(
                [0.01, -0.01], [1], ["time"], [-0.01], [0.02], [2]
            )


class RiskBudgetTests(unittest.TestCase):
    def test_position_size_is_derived_from_stop_distance(self):
        common = dict(
            net_returns=[0.01, 0.01],
            mae_returns=[-0.005, -0.005],
            entry_dates=["2024-01-01", "2024-01-02"],
            risk_per_trade_pct=0.005,
        )
        tight = simulate_risk_budget(
            stop_distance_returns=[0.01, 0.01], **common
        )
        wide = simulate_risk_budget(
            stop_distance_returns=[0.02, 0.02], **common
        )

        self.assertGreater(
            tight["mean_position_notional_usd"],
            wide["mean_position_notional_usd"],
        )
        self.assertEqual(tight["base_signal_expectancy_bps"], 100.0)
        self.assertEqual(wide["base_signal_expectancy_bps"], 100.0)

    def test_daily_loss_brake_blocks_later_same_day_entries(self):
        result = simulate_risk_budget(
            net_returns=[-0.05, -0.01, 0.01],
            mae_returns=[-0.05, -0.01, -0.002],
            stop_distance_returns=[0.01, 0.01, 0.01],
            entry_dates=["2024-01-01"] * 3,
            risk_per_trade_pct=0.01,
            daily_loss_limit_pct=2.0,
        )

        self.assertEqual(result["executed_trades"], 1)
        self.assertEqual(result["skipped_daily_loss_brake"], 2)
        self.assertEqual(result["daily_loss_brake_days"], 1)

    def test_drawdown_halt_prevents_new_entries(self):
        result = simulate_risk_budget(
            net_returns=[-1.0, 0.10],
            mae_returns=[-1.0, -0.01],
            stop_distance_returns=[0.01, 0.01],
            entry_dates=["2024-01-01", "2024-01-02"],
            risk_per_trade_pct=0.01,
            max_drawdown_stop_pct=15.0,
        )

        self.assertTrue(result["drawdown_halt_triggered"])
        self.assertEqual(result["executed_trades"], 1)
        self.assertEqual(result["skipped_after_drawdown_halt"], 1)
        self.assertGreater(result["max_closed_trade_drawdown_pct"], 15.0)

    def test_loss_streak_cooldown_is_applied_to_following_trade(self):
        result = simulate_risk_budget(
            net_returns=[-0.05, -0.05, -0.05, 0.05],
            mae_returns=[-0.05, -0.05, -0.05, -0.01],
            stop_distance_returns=[0.01] * 4,
            entry_dates=["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            risk_per_trade_pct=0.01,
            daily_loss_limit_pct=10.0,
            max_drawdown_stop_pct=90.0,
            max_consecutive_losses=2,
            cooldown_trades=1,
        )

        self.assertEqual(result["executed_trades"], 3)
        self.assertEqual(result["skipped_loss_streak_cooldown"], 1)
        self.assertEqual(result["max_consecutive_losses"], 2)

    def test_risk_grid_does_not_change_underlying_signal_edge(self):
        result = run_risk_budget_grid(
            net_returns=[0.01, -0.02, 0.015],
            mae_returns=[-0.005, -0.02, -0.004],
            stop_distance_returns=[0.01, 0.02, 0.01],
            entry_dates=["2024-01-01", "2024-01-02", "2024-01-03"],
            risk_levels_pct=(0.25, 0.5, 1.0),
        )

        expectancies = {
            row["base_signal_expectancy_bps"] for row in result.values()
        }
        self.assertEqual(expectancies, {16.667})
        self.assertEqual(len(result), 3)

    def test_risk_simulation_rejects_positive_mae(self):
        with self.assertRaisesRegex(ValueError, "non-positive"):
            simulate_risk_budget(
                net_returns=[0.01],
                mae_returns=[0.001],
                stop_distance_returns=[0.01],
                entry_dates=["2024-01-01"],
                risk_per_trade_pct=0.005,
            )


if __name__ == "__main__":
    unittest.main()
