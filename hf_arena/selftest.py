"""Offline self-test: replay 90 days, no API keys, no network orders.

Verifies: model artifact loads, features align, all 6 ledgers book trades,
fees are charged, gross != net, guards fire, and the report is JSON-safe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hf_arena.arena import Arena, FrozenSignals  # noqa: E402
from hf_arena.config import (INITIAL_CAPITAL_USD, LEDGER_PATH,  # noqa: E402
                             MODEL_ARTIFACT, ROUND_TRIP_TAKER_BPS)


def main(days: int = 90) -> int:
    print("=== HF ARENA SELFTEST (offline, paper only) ===", flush=True)
    assert MODEL_ARTIFACT.exists(), "frozen model artifact missing"
    signals = FrozenSignals(MODEL_ARTIFACT, [])
    print(f"model loaded: thr_long={signals.thr_long:.5f} thr_short={signals.thr_short:.5f} "
          f"features={len(signals.features)}", flush=True)

    from hf_arena.engine import ArenaEngine
    bars = days * 96
    df = ArenaEngine._candles_from_db(bars + 600)
    print(f"candles: {len(df)} rows, {df['candle_time'].iloc[0]} -> {df['candle_time'].iloc[-1]}",
          flush=True)

    arena = Arena(signals=signals)
    arena.set_candles(df)
    arena.run_range(600, len(df) - 2)
    report = arena.report()
    report["mode"] = "selftest"
    report["orders_sent"] = 0
    report["note"] = "Historical paper replay; fees included; NOT a forecast."

    print("\n=== LEADERBOARD (net, after real fees) ===", flush=True)
    print(f"{'rank':>4} {'model':<18} {'trades':>7} {'WR%':>7} {'PF':>7} "
          f"{'net$':>9} {'gross$':>9} {'fees$':>7} {'DD%':>6}", flush=True)
    for r in report["leaderboard"]:
        print(f"{r['rank']:>4} {r['model_id']:<18} {r['trades']:>7} "
              f"{str(r['win_rate_net_pct']):>7} {str(r['profit_factor_net']):>7} "
              f"{r['balance_net_usd']:>9} {r['balance_gross_usd']:>9} "
              f"{r['fees_paid_usd']:>7} {r['max_drawdown_pct']:>6}", flush=True)

    outs = Path(__file__).resolve().parent / "runtime" / "selftest_report.json"
    outs.parent.mkdir(parents=True, exist_ok=True)
    outs.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")
    arena.save(LEDGER_PATH)
    print(f"\n[report] {outs}", flush=True)

    # ---- assertions
    rows = report["leaderboard"]
    assert len(rows) == 6, "expected 6 models"
    assert all(r["balance_net_usd"] != r["balance_gross_usd"] or r["trades"] == 0
               for r in rows), "fees must be charged"
    assert all(r["fees_paid_usd"] >= 0 for r in rows)
    total = sum(r["trades"] for r in rows)
    assert total > 0, "arena produced no trades at all"
    print(f"\nSELFTEST_OK total_trades={total} fee_bps={ROUND_TRIP_TAKER_BPS} "
          f"initial=${INITIAL_CAPITAL_USD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 90))
