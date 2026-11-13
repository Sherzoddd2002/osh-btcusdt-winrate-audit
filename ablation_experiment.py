"""Controlled ablation: do external cross-asset / news features add out-of-sample edge?

All four configs share IDENTICAL candles, labels, folds, thresholds and costs.
Only the feature matrix changes. This is the only fair way to answer
"will adding gold / news help the signal?".
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline_v2 as pv  # noqa: E402

R = Path(os.getenv("OSH_DATA_DIR", str(HERE)))
COST = 10.0
H = 4
TOP = 0.05

# ------------------------------------------------------------------ BTC
eng = create_engine("postgresql+psycopg2://postgres@127.0.0.1:5433/osh")
df = pd.read_sql("""SELECT to_timestamp(open_time/1000.0) AS candle_time, *
                    FROM public.btcusdt_15m ORDER BY open_time""", eng)
df = df.drop(columns=[c for c in ["ignore_value"] if c in df.columns])
df["candle_time"] = pd.to_datetime(df["candle_time"], utc=True)
for c in ["open", "high", "low", "close", "volume", "quote_volume", "trade_count",
          "taker_buy_base_volume", "taker_buy_quote_volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().sort_values("candle_time").reset_index(drop=True)

X_base, base_feats = pv.make_features(df)
lab = pv.triple_barrier_labels(df, X_base, horizon=H, cost_bps=COST)

# ------------------------------------------------------- external features
g = pd.read_csv(R / "WIDE_GLD_AUX_2MIN.csv")
g["datetime"] = pd.to_datetime(g["datetime"], utc=True)
g = g.sort_values("datetime").set_index("datetime")
closes = [c for c in g.columns if c.endswith("_Close")]
g15 = g[closes].resample("15min").last()
g15.index.name = "candle_time"
ext = df[["candle_time"]].merge(g15.reset_index(), on="candle_time", how="left")
X_ext = pd.DataFrame(index=df.index)
for c in closes:
    a = c.replace("_Close", "")
    p = ext[c].ffill()                      # backward-only, no look-ahead
    r1 = np.log(p / p.shift(1))
    X_ext[f"{a}_ret1"] = r1
    X_ext[f"{a}_ret4"] = np.log(p / p.shift(4))
    X_ext[f"{a}_z96"] = (p / p.rolling(96, min_periods=24).mean() - 1) / \
                        (r1.rolling(96, min_periods=24).std() + 1e-9)
X_ext["xasset_coverage"] = ext[closes[0]].notna().astype("int8")

# news -> numeric (strict backward asof, no look-ahead)
n = pd.read_csv(R / "macro_data" / "news_raw.csv")
n["dt"] = pd.to_datetime(n["published_at"], utc=True, errors="coerce")
n = n.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
BULL = ["surge", "rally", "gain", "bullish", "record high", "beat", "higher",
        "cut rates", "safe haven", "positive", "growth", "expansion", "strong"]
BEAR = ["drop", "crash", "fall", "bearish", "selloff", "miss", "lower", "hike",
        "strong dollar", "negative", "decline", "weak", "risk", "fear"]


def score(t):
    s = str(t).lower()
    return float(sum(w in s for w in BULL) - sum(w in s for w in BEAR))


n["sent"] = (n["title"].fillna("") + " " + n["description"].fillna("")).map(score)
n["n24"] = n.set_index("dt")["sent"].rolling("24h").count().to_numpy()
n["s24"] = n.set_index("dt")["sent"].rolling("24h").mean().to_numpy()
n["cum"] = n["sent"].expanding().mean()
left = pd.DataFrame({"candle_time": df["candle_time"]})
X_news = pd.DataFrame(index=df.index)
for col, name in [("sent", "news_sent"), ("s24", "news_sent24h"),
                  ("n24", "news_count24h"), ("cum", "news_cum")]:
    tmp = n[["dt", col]].dropna().rename(columns={"dt": "candle_time"})
    mg = pd.merge_asof(left.sort_values("candle_time"), tmp.sort_values("candle_time"),
                       on="candle_time", direction="backward")
    X_news[name] = mg[col].fillna(0).to_numpy()
X_news["news_coverage"] = X_news["news_sent"].ne(0).astype("int8")

# ------------------------------------------------- common evaluation window
start = pd.Timestamp("2024-02-20", tz="UTC")
end = pd.Timestamp("2026-02-19", tz="UTC")
cov = df["candle_time"].between(start, end) & lab["valid"]
keep = cov & X_base.notna().all(axis=1)
idx = np.flatnonzero(keep.to_numpy())

Xb = X_base.iloc[idx].reset_index(drop=True).fillna(0)
Xe = X_ext.iloc[idx].reset_index(drop=True).replace([np.inf, -np.inf], 0).fillna(0)
Xn = X_news.iloc[idx].reset_index(drop=True).replace([np.inf, -np.inf], 0).fillna(0)
L = lab.iloc[idx].reset_index(drop=True)
T = df["candle_time"].iloc[idx].reset_index(drop=True)
print(f"common window: {T.iloc[0]} -> {T.iloc[-1]}  rows={len(T):,}")
print(f"features: base={Xb.shape[1]}  xasset={Xe.shape[1]}  news={Xn.shape[1]}")

min_train, test_len, embargo = 96 * 180, 96 * 90, H + 2
folds = pv.make_folds(len(T), min_train, test_len, embargo, 6)
print(f"folds: {len(folds)}")

y = L["y_long"].astype("int8")
w = L["sample_weight"].astype(float)
r_exit = L["exit_ret"].to_numpy()
r_hold = L["fwd_ret_h"].to_numpy()

configs = {
    "A_BASE": Xb,
    "B_BASE_XASSET": pd.concat([Xb, Xe], axis=1),
    "C_BASE_NEWS": pd.concat([Xb, Xn], axis=1),
    "D_BASE_BOTH": pd.concat([Xb, Xe, Xn], axis=1),
}

results = {}
for name, X in configs.items():
    per_fold = []
    for i, (tr0, tr1, te0, te1) in enumerate(folds, 1):
        model = pv.build_model()
        model = pv.fit_with_early_stop(model, X.iloc[tr0:tr1], y.iloc[tr0:tr1], w.iloc[tr0:tr1])
        p_cal = model.predict_proba(X.iloc[tr1 - test_len:tr1])[:, 1]
        thr = float(np.quantile(p_cal, 1 - TOP))
        p = model.predict_proba(X.iloc[te0:te1])[:, 1]
        res = pv.evaluate_fold(p, y.iloc[te0:te1].to_numpy(),
                               r_exit[te0:te1], r_hold[te0:te1], COST, H, TOP)
        per_fold.append({
            "fold": i, "start": str(T.iloc[te0])[:10],
            "trades": res["trades"],
            "win_rate_pct": res["win_rate_after_cost_pct"],
            "mean_net_bps": res["mean_net_bps"],
            "auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"])
                         .roc_auc_score(y.iloc[te0:te1], p)),
            "threshold": round(thr, 5),
        })
    nets = [f["mean_net_bps"] for f in per_fold]
    wr = [f["win_rate_pct"] for f in per_fold]
    results[name] = {
        "folds": per_fold,
        "median_net_bps": float(np.median(nets)),
        "mean_net_bps": float(np.mean(nets)),
        "positive_folds": int(sum(v > 0 for v in nets)),
        "mean_win_rate_pct": float(np.mean(wr)),
        "median_auc": float(np.median([f["auc"] for f in per_fold])),
        "total_trades": int(sum(f["trades"] for f in per_fold)),
        "feature_count": int(X.shape[1]),
    }
    print(f"{name:<16} median_net={results[name]['median_net_bps']:+.2f}bps  "
          f"win={results[name]['mean_win_rate_pct']:.2f}%  "
          f"pos_folds={results[name]['positive_folds']}/{len(folds)}  AUC={results[name]['median_auc']:.4f}")

base = results["A_BASE"]["median_net_bps"]
for k in results:
    results[k]["delta_vs_base_bps"] = round(results[k]["median_net_bps"] - base, 3)

out = {"common_window": [str(T.iloc[0]), str(T.iloc[-1])], "rows": int(len(T)),
       "cost_bps": COST, "horizon_bars": H, "folds": len(folds),
       "configs": results,
       "verdict_note": "Only a controlled ablation can answer whether external data helps."}
(R / "ablation_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
print("\nDELTA vs A_BASE (median net bps):")
for k in ["B_BASE_XASSET", "C_BASE_NEWS", "D_BASE_BOTH"]:
    print(f"  {k:<16} {results[k]['delta_vs_base_bps']:+.3f}")
