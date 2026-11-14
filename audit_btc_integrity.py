"""Fail-closed structural audit for public.btcusdt_15m.

This reports data consistency only. It cannot independently prove exchange
authenticity; independent Binance reconciliation is a separate requirement.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NUMERIC = [
    "open", "high", "low", "close", "volume", "quote_volume",
    "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume",
]
TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


def audit_frame(frame: pd.DataFrame, interval_minutes: int = 15) -> dict:
    """Audit an already loaded DataFrame without DB/file side effects."""
    if frame.empty:
        raise ValueError("Cannot audit an empty dataset.")
    if "candle_time" not in frame.columns:
        raise ValueError("Missing candle_time column.")
    missing = [c for c in NUMERIC if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required numeric columns: {missing}")

    df = frame.copy()
    df["candle_time"] = pd.to_datetime(df["candle_time"], utc=True, errors="coerce")
    raw = df[NUMERIC].copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    order = df.sort_values("candle_time").index
    df = df.loc[order].reset_index(drop=True)
    raw = raw.loc[order].reset_index(drop=True)
    times = df["candle_time"]
    if times.isna().any():
        raise ValueError("Invalid or missing candle_time values detected.")
    delta = times.diff().dropna()
    expected = pd.Timedelta(minutes=interval_minutes)
    bad_intervals = delta[delta != expected]
    duplicate_times = int(times.duplicated().sum())

    res = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "period_utc": [str(times.min()), str(times.max())],
        "duplicate_candle_time": duplicate_times,
        # diff().dropna() deliberately excludes the first row's NaT.
        "non_interval_gaps": int(len(bad_intervals)),
        "missing_expected_candles": int(
            sum(max(0, int(d / expected) - 1) for d in bad_intervals if d > expected)
        ),
        "non_numeric_or_missing": {
            col: int(df[col].isna().sum()) for col in NUMERIC
        },
    }

    anomalies = {}
    for col in NUMERIC:
        numeric = df[col]
        nonnum = raw[col].notna() & numeric.isna()
        anomalies[col] = {
            "non_numeric": int(nonnum.sum()),
            "abs_gt_1e15": int((numeric.abs() > 1e15).sum()),
            "negative": int((numeric < 0).sum()),
        }
    res["column_anomalies"] = anomalies

    prices_valid = df[["open", "high", "low", "close"]].notna().all(axis=1)
    d = df.loc[prices_valid].copy()
    o, h, l, c = (d[k].astype(float) for k in ("open", "high", "low", "close"))
    res["valid_ohlc_rows"] = int(len(d))
    res["nonpositive_price_rows"] = int(((d[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum())
    res["ohlc_violations"] = int(
        ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)).sum()
    )
    res["zero_volume_rows"] = int((df["volume"] == 0).sum())
    res["zero_trade_count_rows"] = int((df["trade_count"] == 0).sum())
    res["taker_base_gt_volume_rows"] = int(
        (df["taker_buy_base_volume"] > df["volume"] + 1e-8).sum()
    )
    res["taker_quote_gt_quote_volume_rows"] = int(
        (df["taker_buy_quote_volume"] > df["quote_volume"] + 1e-6).sum()
    )

    if {"open_time", "close_time"}.issubset(df.columns):
        open_ms = pd.to_numeric(df["open_time"], errors="coerce")
        close_ms = pd.to_numeric(df["close_time"], errors="coerce")
        res["close_time_width_wrong"] = int(((close_ms - open_ms) != interval_minutes * 60_000 - 1).sum())
        res["duplicate_open_time"] = int(open_ms.duplicated().sum())
    else:
        res["close_time_width_wrong"] = None
        res["duplicate_open_time"] = None

    consecutive = d[["open", "close"]].copy()
    if len(consecutive) > 1:
        res["open_equals_prev_close_pct"] = round(
            float((consecutive["open"].to_numpy()[1:] == consecutive["close"].to_numpy()[:-1]).mean() * 100), 3
        )
    else:
        res["open_equals_prev_close_pct"] = None
    returns = np.log(c / c.shift(1)).dropna()
    res["lag1_return_autocorr"] = (
        round(float(returns.autocorr(1)), 5) if len(returns) > 2 else None
    )
    res["absolute_return_mean_bps"] = (
        round(float(returns.abs().mean() * 10_000), 3) if len(returns) else None
    )
    res["last_5_closes"] = [
        [str(t), round(float(value), 8)]
        for t, value in zip(d["candle_time"].tail(5), d["close"].tail(5))
    ]
    age_days = (pd.Timestamp.now(tz="UTC") - times.max()).total_seconds() / 86_400
    res["data_age_days"] = round(float(age_days), 3)

    negative_rows = int((df[NUMERIC] < 0).any(axis=1).sum())
    res["negative_numeric_rows"] = negative_rows
    structural_errors = (
        res["duplicate_candle_time"] + res["non_interval_gaps"]
        + res["nonpositive_price_rows"] + res["ohlc_violations"]
        + sum(res["non_numeric_or_missing"].values()) + negative_rows
        + res["taker_base_gt_volume_rows"]
        + res["taker_quote_gt_quote_volume_rows"]
        + (res["close_time_width_wrong"] or 0)
        + (res["duplicate_open_time"] or 0)
    )
    res["structural_verdict"] = "PASS" if structural_errors == 0 else "FAIL"
    res["authenticity_note"] = (
        "PASS means structural checks passed only; exchange authenticity requires "
        "independent source reconciliation."
    )
    return res


def load_from_db(db_url: str, table: str) -> pd.DataFrame:
    if not TABLE_RE.fullmatch(table):
        raise ValueError("Unsafe table name. Expected schema.table.")
    from sqlalchemy import create_engine

    engine = create_engine(db_url, pool_pre_ping=True)
    sql = f"""
        SELECT to_timestamp(open_time / 1000.0) AS candle_time, *
        FROM {table}
        ORDER BY open_time ASC
    """
    try:
        with engine.connect() as connection:
            frame = pd.read_sql_query(sql, connection)
    finally:
        engine.dispose()
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit BTCUSDT 15m candle structure.")
    parser.add_argument(
        "--db-url",
        default=os.getenv("OSH_DB_URL", "postgresql+psycopg2://postgres@127.0.0.1:5433/osh"),
    )
    parser.add_argument("--table", default="public.btcusdt_15m")
    parser.add_argument("--out", default=str(HERE / "integrity_report.json"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = audit_frame(load_from_db(args.db_url, args.table))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for key, value in report.items():
        print(f"{key:<32} {value}")
    print(f"[report] {output.resolve()}")


if __name__ == "__main__":
    main()
