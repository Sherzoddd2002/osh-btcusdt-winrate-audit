from __future__ import annotations

import unittest
from decimal import Decimal

from demo_executor.client import DEMO_FUTURES_URL
from demo_executor.logic import (
    barrier_prices, choose_side, decimal_floor_step, quantity_for_cap,
    round_price_to_tick,
)


class DemoLogicTests(unittest.TestCase):
    def test_only_demo_url_is_configured(self):
        self.assertEqual(DEMO_FUTURES_URL, "https://demo-fapi.binance.com")
        self.assertNotIn("https://fapi.binance.com", DEMO_FUTURES_URL)

    def test_thresholds_control_signals(self):
        self.assertIsNone(choose_side(.5, .5, .8, .8))
        self.assertEqual(choose_side(.85, .4, .8, .8), "LONG")
        self.assertEqual(choose_side(.4, .9, .8, .8), "SHORT")

    def test_dual_signal_chooses_greater_normalized_excess(self):
        self.assertEqual(choose_side(.90, .88, .80, .70), "SHORT")

    def test_h96_c2_barriers(self):
        self.assertAlmostEqual(barrier_prices("LONG", 100000, .02)[0], 101500)
        self.assertAlmostEqual(barrier_prices("LONG", 100000, .02)[1], 98000)
        self.assertAlmostEqual(barrier_prices("SHORT", 100000, .02)[0], 98500)
        self.assertAlmostEqual(barrier_prices("SHORT", 100000, .02)[1], 102000)

    def test_quantity_rounds_down(self):
        q = quantity_for_cap(100, 100000, "0.001", "0.001", 50)
        self.assertEqual(q, Decimal("0.001"))
        self.assertLessEqual(float(q) * 100000, 100.000001)

    def test_never_increases_cap_to_exchange_minimum(self):
        with self.assertRaisesRegex(ValueError, "minimum quantity"):
            quantity_for_cap(50, 100000, "0.001", "0.001", 50)

    def test_min_notional_enforced(self):
        with self.assertRaisesRegex(ValueError, "minimum notional"):
            quantity_for_cap(50, 10000, "0.0001", "0.0001", 100)

    def test_decimal_tick_rounding(self):
        self.assertEqual(round_price_to_tick(100.011, "0.01", "down"), Decimal("100.01"))
        self.assertEqual(round_price_to_tick(100.011, "0.01", "up"), Decimal("100.02"))

    def test_floor_step(self):
        self.assertEqual(decimal_floor_step("0.00199", "0.001"), Decimal("0.001"))


if __name__ == "__main__":
    unittest.main()
