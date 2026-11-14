import unittest
import numpy as np

from execution_costs_1d import (
    SCENARIO_LEGACY_MAKER_MAKER,
    SCENARIO_TAKER_ALL,
    SCENARIO_TAKER_ENTRY_TP_MAKER,
    build_execution_scenarios,
    calculate_trade_costs,
)


class ExecutionCostTests(unittest.TestCase):
    def test_hybrid_uses_taker_entry_and_exit_type_specific_fee(self):
        result = build_execution_scenarios(
            np.array([0.01, -0.01, 0.005]),
            np.array([1, -1, 1]),
            np.array(["take_profit", "stop_loss", "time"], dtype=object),
            np.zeros(3),
        )
        hybrid = result[SCENARIO_TAKER_ENTRY_TP_MAKER]
        self.assertAlmostEqual(hybrid["total_cost_return"][0] * 10_000, 9.02, places=8)
        self.assertAlmostEqual(hybrid["total_cost_return"][1] * 10_000, 12.05, places=8)
        self.assertAlmostEqual(hybrid["total_cost_return"][2] * 10_000, 12.025, places=8)

    def test_legacy_and_all_taker_scenarios_are_ordered_by_cost(self):
        result = build_execution_scenarios(
            np.array([0.01, -0.02]),
            np.array([1, -1]),
            np.array(["take_profit", "time"], dtype=object),
            np.zeros(2),
        )
        legacy = result[SCENARIO_LEGACY_MAKER_MAKER]["total_cost_return"]
        hybrid = result[SCENARIO_TAKER_ENTRY_TP_MAKER]["total_cost_return"]
        all_taker = result[SCENARIO_TAKER_ALL]["total_cost_return"]
        self.assertTrue(np.all(legacy < hybrid))
        self.assertTrue(np.all(hybrid <= all_taker))
        self.assertTrue(np.all(all_taker > 0))

    def test_funding_is_added_after_all_execution_costs(self):
        result = build_execution_scenarios(
            np.array([0.001]),
            np.array([1]),
            np.array(["take_profit"], dtype=object),
            np.array([-0.0002]),
        )
        row = result[SCENARIO_TAKER_ENTRY_TP_MAKER]
        expected = 0.001 - row["total_cost_return"][0] - 0.0002
        self.assertAlmostEqual(row["net_return"][0], expected, places=12)

    def test_short_exit_notional_uses_short_exit_value(self):
        costs = calculate_trade_costs(
            -0.02, -1, "stop_loss", entry_fee_bps=5, exit_fee_bps=5
        )
        self.assertAlmostEqual(costs["exit_notional_ratio"], 1.02, places=12)
        self.assertAlmostEqual(costs["exit_commission_return"] * 10_000, 5.10, places=8)

    def test_misaligned_arrays_raise(self):
        with self.assertRaises(ValueError):
            build_execution_scenarios(
                np.array([0.01]), np.array([1, -1]),
                np.array(["take_profit"], dtype=object), np.array([0.0])
            )

    def test_invalid_fee_rates_raise(self):
        with self.assertRaises(ValueError):
            calculate_trade_costs(
                0.01, 1, "take_profit", entry_fee_bps=-1, exit_fee_bps=2
            )


if __name__ == "__main__":
    unittest.main()
