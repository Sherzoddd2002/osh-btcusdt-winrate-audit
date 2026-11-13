from __future__ import annotations

import unittest

import pandas as pd

from audit_btc_integrity import audit_frame


def audit_frame_fixture(n: int = 4) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    open_time = times.astype("int64") // 1_000_000
    return pd.DataFrame({
        "candle_time": times,
        "open_time": open_time,
        "close_time": open_time + 899999,
        "open": 100.0,
        "high": 100.1,
        "low": 99.9,
        "close": 100.0,
        "volume": 10.0,
        "quote_volume": 1000.0,
        "trade_count": 100.0,
        "taker_buy_base_volume": 5.0,
        "taker_buy_quote_volume": 500.0,
    })


class IntegrityAuditTests(unittest.TestCase):
    def test_first_nat_diff_is_not_reported_as_a_gap(self):
        report = audit_frame(audit_frame_fixture())
        self.assertEqual(report["non_interval_gaps"], 0)
        self.assertEqual(report["missing_expected_candles"], 0)
        self.assertEqual(report["structural_verdict"], "PASS")

    def test_real_missing_interval_is_reported(self):
        frame = audit_frame_fixture(5).drop(index=2).reset_index(drop=True)
        report = audit_frame(frame)
        self.assertEqual(report["non_interval_gaps"], 1)
        self.assertEqual(report["missing_expected_candles"], 1)
        self.assertEqual(report["structural_verdict"], "FAIL")

    def test_invalid_timestamp_fails_closed(self):
        frame = audit_frame_fixture()
        frame.loc[0, "candle_time"] = None
        with self.assertRaisesRegex(ValueError, "candle_time"):
            audit_frame(frame)


if __name__ == "__main__":
    unittest.main()
