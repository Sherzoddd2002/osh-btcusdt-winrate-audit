"""Integrity audit of public.btcusdt_15m — detects corrupted / synthetic candles."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

R = Path(os.getenv("OSH_DATA_DIR", str(Path(__file__).resolve().parent)))

ENGINE = create_engine("postgresql+psycopg2://postgres@127.0.0.1:5433/osh")
df = pd.read_sql("""
    SELECT to_timestamp(open_time/1000.0) AS candle_time, *
    FROM public.btcusdt_15m ORDER BY open_time
""", ENGINE)
df = df.drop(columns=[c for c in ["ignore_value"] if c in df.columns])
df["candle_time"] = pd.to_datetime(df["candle_time"], utc=True)

NUM = ["open", "high", "low", "close", "volume", "quote_volume",
       "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume"]
raw = df[NUM].copy()
for c in NUM:
    df[c] = pd.to_numeric(raw[c], errors="coerce")

res = {}
res["rows"] = int(len(df))
res["period"] = [str(df.candle_time.min()), str(df.candle_time.max())]
res["duplicate_open_time"] = int(df["open_time"].duplicated().sum())
res["non_15m_gaps"] = int((df["candle_time"].diff() != pd.Timedelta(minutes=15)).sum())

# --- corrupted / non-numeric / absurd values ----------------------------
bad = {}
for c in NUM:
    nonnum = raw[c].notna() & df[c].isna()
    huge = df[c].abs() > 1e15
    neg = df[c] < 0
    bad[c] = {"non_numeric": int(nonnum.sum()),
              "abs_gt_1e15": int(huge.sum()),
              "negative": int(neg.sum())}
res["column_anomalies"] = bad
sus = df[(df["trade_count"].abs() > 1e15) | (df["trade_count"].isna() & raw["trade_count"].notna())]
res["corrupted_trade_count_rows"] = [
    {"time": str(t), "raw_head": str(x)[:60], "raw_len": len(str(x))}
    for t, x in zip(sus.candle_time.head(8), raw.loc[sus.index, "trade_count"].head(8))
]
res["corrupted_trade_count_count"] = int(len(sus))

# valid subset for the remaining structural checks
ok = df[NUM].notna().all(axis=1) & (df[NUM].abs() < 1e15).all(axis=1)
d = df.loc[ok].copy()
o, h, l, c = (d[k].astype(float) for k in ["open", "high", "low", "close"])
v = d["volume"].astype(float)
tb = d["taker_buy_base_volume"].astype(float)
qv = d["quote_volume"].astype(float)
tc = d["trade_count"].astype(float)

res["valid_rows"] = int(len(d))
res["ohlc_violations"] = int(((h < np.maximum(o, c) - 1e-9) | (l > np.minimum(o, c) + 1e-9)).sum())
res["nonpositive_price"] = int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum())
res["zero_volume"] = int((v <= 0).sum())
res["zero_trade_count"] = int((tc <= 0).sum())
res["taker_gt_volume"] = int((tb > v + 1e-6).sum())
res["close_time_width_wrong"] = int((d["close_time"] - d["open_time"] != 899999).sum())
vwap = qv / v.replace(0, np.nan)
res["vwap_outside_hilo_pct"] = round(float(((vwap < l * 0.995) | (vwap > h * 1.005)).mean() * 100), 3)

# --- synthetic data detectors -------------------------------------------
res["open_equals_prev_close_pct"] = round(float((o.iloc[1:].values == c.iloc[:-1].values).mean() * 100), 3)
res["no_upper_wick_pct"] = round(float(((h - np.maximum(o, c)).abs() < 1e-9).mean() * 100), 3)
res["no_lower_wick_pct"] = round(float(((np.minimum(o, c) - l).abs() < 1e-9).mean() * 100), 3)
res["exact_doji_pct"] = round(float(((c - o).abs() < 1e-9).mean() * 100), 3)

ret = np.log(c / c.shift(1)).dropna()
up = (c > o)
prev_up = up.shift(1).fillna(False).astype(bool)
prev_dn = (~up).shift(1).fillna(False).astype(bool)
res["lag1_return_autocorr"] = round(float(ret.autocorr(1)), 5)
res["up_after_up_pct"] = round(float((prev_up & up).sum() / prev_up.sum() * 100), 3)
res["down_after_down_pct"] = round(float((prev_dn & ~up).sum() / prev_dn.sum() * 100), 3)
res["up_rate_pct"] = round(float(up.mean() * 100), 3)
res["ret_abs_mean_bps"] = round(float(ret.abs().mean() * 10000), 2)

res["last_5_closes"] = [[str(t), round(float(x), 2)] for t, x in zip(d.candle_time.tail(5), c.tail(5))]
tod = d.assign(hour=d.candle_time.dt.hour).groupby("hour")["trade_count"].mean()
res["trade_count_hour_minmax_ratio"] = round(float(tod.max() / tod.min()), 3)

# is the 'last' candle aligned with the run date?
res["days_data_lag"] = int((pd.Timestamp("2026-11-13", tz="UTC") - d.candle_time.max()).days)

(R / "integrity_report.json").write_text(
    json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
for k, v_ in res.items():
    print(f"{k:<32} {v_}")
