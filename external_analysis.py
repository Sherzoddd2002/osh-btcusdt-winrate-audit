"""Cross-asset / news feature audit for the OSH BTCUSDT 15m research pipeline.

Produces external_analysis.json with:
  - BTC data integrity summary
  - authenticity checks on gold/macro/news datasets
  - contemporaneous vs PREDICTIVE (IC/IR) correlations
  - news -> numeric features with strict no-look-ahead alignment
  - a negative control (random features) to quantify selection noise
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# Raw datasets live in the research folder; override with OSH_DATA_DIR elsewhere.
R = Path(os.getenv("OSH_DATA_DIR", str(Path(__file__).resolve().parent)))
OUT = {}
EPS = 1e-12

# ---------------------------------------------------------------- load BTC
eng = create_engine("postgresql+psycopg2://postgres@127.0.0.1:5433/osh")
btc = pd.read_sql("""SELECT to_timestamp(open_time/1000.0) AS t, open, high, low,
                     close, volume, trade_count FROM public.btcusdt_15m ORDER BY open_time""", eng)
btc["t"] = pd.to_datetime(btc["t"], utc=True)
for c in ["open", "high", "low", "close", "volume", "trade_count"]:
    btc[c] = pd.to_numeric(btc[c], errors="coerce")
btc = btc.dropna().sort_values("t").reset_index(drop=True)
btc["r1"] = np.log(btc.close / btc.close.shift(1))
btc["fwd_1h"] = np.log(btc.close.shift(-4) / btc.close)   # predictive target
btc["fwd_4h"] = np.log(btc.close.shift(-16) / btc.close)

# ------------------------------------------------- 1. trade_count anomaly
tc_raw = pd.read_sql("SELECT trade_count FROM public.btcusdt_15m", eng)["trade_count"]
lens = tc_raw.astype(str).str.len()
OUT["trade_count"] = {
    "dtype": str(tc_raw.dtype),
    "max_string_length": int(lens.max()),
    "rows_over_100_chars": int((lens > 100).sum()),
    "example_head": str(tc_raw.iloc[np.argmax(lens.to_numpy())])[:70],
    "median": float(pd.to_numeric(tc_raw, errors="coerce").median()),
}

# ------------------------------------------------- 2. cross-asset dataset
g = pd.read_csv(R / "WIDE_GLD_AUX_2MIN.csv")
g["datetime"] = pd.to_datetime(g["datetime"], utc=True)
g = g.sort_values("datetime").set_index("datetime")
asset_closes = [c for c in g.columns if c.endswith("_Close")]
g15 = g[asset_closes].resample("15min").last().ffill()
g15.index.name = "t"

# authenticity indicators
OUT["cross_asset"] = {
    "rows": int(len(g)),
    "range": [str(g.index.min()), str(g.index.max())],
    "assets": [c.replace("_Close", "") for c in asset_closes],
    "null_pct": {c.replace("_Close", ""): round(float(g[c].isna().mean() * 100), 1)
                 for c in asset_closes},
    "gld_first_close": float(g["GLD_Close"].dropna().iloc[0]),
    "gld_last_close": float(g["GLD_Close"].dropna().iloc[-1]),
    "gld_lag1_autocorr": round(float(np.log(g["GLD_Close"].dropna() /
                                            g["GLD_Close"].dropna().shift(1)).dropna().autocorr(1)), 5),
}
# real GLD traded ~186-190 in Feb 2024 (external reference: domain_search / market data)
OUT["cross_asset"]["gld_feb2024_plausible"] = bool(180 < OUT["cross_asset"]["gld_first_close"] < 200)

# ------------------------------------------------- 3. merge, no look-ahead
m = btc[["t", "r1", "fwd_1h", "fwd_4h", "close"]].merge(
    g15.reset_index(), on="t", how="left")
for c in asset_closes:
    m[c] = m[c].ffill()          # ONLY past information (ffill = backward)
    m[c.replace("_Close", "_r1")] = np.log(m[c] / m[c].shift(1))
m["quarter"] = m["t"].dt.to_period("Q").astype(str)
m["date"] = m["t"].dt.date

# only rows where the external asset actually printed in that 15m bucket
live = {}
for c in asset_closes:
    live[c] = m[c].notna() & (m[c.replace("_Close", "_r1")].abs() > 1e-10)

corr_table = []
for c in asset_closes:
    a = c.replace("_Close", "")
    rc = c.replace("_Close", "_r1")
    mask = live[c] & m["r1"].notna() & m["fwd_1h"].notna()
    sub = m.loc[mask]
    same = float(sub["r1"].corr(sub[rc]))
    # IC = Spearman(feature_t, forward BTC return) — the predictive test
    ic = float(sub[rc].corr(sub["fwd_1h"], method="spearman"))
    ics = sub.groupby("quarter").apply(
        lambda d: d[rc].corr(d["fwd_1h"], method="spearman"), include_groups=False)
    ir = float(ics.mean() / (ics.std() + EPS))
    # daily-return correlation
    dd = pd.DataFrame({"btc": sub.set_index("t")["r1"], "asset": sub.set_index("t")[rc]}) \
           .resample("1D").sum()
    daily = float(dd["btc"].corr(dd["asset"]))
    roll = dd["btc"].rolling(30).corr(dd["asset"]).dropna()
    corr_table.append({
        "asset": a, "n_15m": int(len(sub)),
        "same_bar_corr": round(same, 4),
        "daily_corr": round(daily, 4),
        "ic_spearman_fwd1h": round(ic, 5),
        "ic_mean_quarter": round(float(ics.mean()), 5),
        "ir_quarter": round(ir, 3),
        "ic_positive_quarter_pct": round(float((ics > 0).mean() * 100), 1),
        "roll30d_min": round(float(roll.min()), 3) if len(roll) else None,
        "roll30d_max": round(float(roll.max()), 3) if len(roll) else None,
    })
OUT["correlations"] = corr_table

# ------------------------------------------------- 4. news corpus
n = pd.read_csv(R / "macro_data" / "news_raw.csv")
n["dt"] = pd.to_datetime(n["published_at"], utc=True, errors="coerce")
n = n.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
sources = n["source"].value_counts()
urls = n["url"].astype(str)
OUT["news_corpus"] = {
    "rows": int(len(n)),
    "range": [str(n.dt.min()), str(n.dt.max())],
    "sources": {str(k): int(v) for k, v in sources.head(5).items()},
    "synthetic_url_pct": round(float(urls.str.contains("synthetic", case=False).mean() * 100), 2),
    "unique_titles": int(n["title"].nunique()),
    "title_template_pct": round(float(n["title"].str.contains("sees a|updates recent movements|amidst strong market signals",
                                                             case=False, regex=True).mean() * 100), 2),
    "avg_per_day": round(float(len(n) / max(1, (n.dt.max() - n.dt.min()).days)), 2),
}

BULL = ["surge", "rally", "gain", "bullish", "record high", "beat", "higher",
        "cut rates", "safe haven", "positive", "growth", "expansion", "strong"]
BEAR = ["drop", "crash", "fall", "bearish", "selloff", "miss", "lower", "hike",
        "strong dollar", "negative", "decline", "weak", "risk", "fear"]


def score(txt: str) -> float:
    if not isinstance(txt, str):
        return 0.0
    s = txt.lower()
    return float(sum(w in s for w in BULL) - sum(w in s for w in BEAR))


n["sent"] = (n["title"].fillna("") + " " + n["description"].fillna("")).map(score)
n["neg"] = (n["sent"] < 0).astype(float)

# strict backward asof: at candle t only news already published at or before t
left = pd.DataFrame({"t": btc["t"]})
news_cols = {}
for key, src, agg in [
    ("news_sent", n[["dt", "sent"]], "last"),
    ("news_neg", n[["dt", "neg"]], "last"),
]:
    pass

# cumulative aggregates computed at each news timestamp, then asof-joined
n["c_sent"] = n["sent"].expanding().mean()
n["c_neg"] = n["neg"].expanding().mean()
n["n_1h"] = n.set_index("dt")["sent"].rolling("1h").count().to_numpy()
n["n_24h"] = n.set_index("dt")["sent"].rolling("24h").count().to_numpy()
n["sent_24h"] = n.set_index("dt")["sent"].rolling("24h").mean().to_numpy()
n["age_h"] = (n["dt"] - n["dt"].shift(1)).dt.total_seconds().fillna(0) / 3600

for col in ["sent", "c_sent", "c_neg", "n_1h", "n_24h", "sent_24h", "age_h"]:
    tmp = n[["dt", col]].dropna().rename(columns={"dt": "t"})
    merged = pd.merge_asof(left.sort_values("t"), tmp.sort_values("t"),
                           on="t", direction="backward", allow_exact_matches=True)
    news_cols["news_" + col] = merged[col].fillna(0).to_numpy()

# delayed variant (free NewsAPI tier has a 24h publication delay)
n_delay = n[["dt", "sent"]].copy()
n_delay["t"] = n_delay["dt"] + pd.Timedelta(hours=24)
md = pd.merge_asof(left.sort_values("t"), n_delay[["t", "sent"]].sort_values("t"),
                   on="t", direction="backward")
news_cols["news_sent_delayed24h"] = md["sent"].fillna(0).to_numpy()

rng = np.random.default_rng(20261113)
news_cols["control_random"] = rng.normal(0, 1, len(left))

nf = pd.DataFrame(news_cols)
nf["t"] = btc["t"].to_numpy()
news_ic = []
for col in [c for c in nf.columns if c != "t"]:
    s = nf[col]
    mask = s.notna() & btc["fwd_1h"].notna()
    q = btc.loc[mask, "t"].dt.to_period("Q").astype(str)
    ic_all = float(s[mask].corr(btc.loc[mask, "fwd_1h"], method="spearman"))
    tmp = pd.DataFrame({"v": s[mask].to_numpy(), "y": btc.loc[mask, "fwd_1h"].to_numpy(),
                        "q": q.to_numpy()})
    ics = tmp.groupby("q").apply(lambda d: d["v"].corr(d["y"], method="spearman"),
                                 include_groups=False)
    news_ic.append({
        "feature": col,
        "ic_spearman_fwd1h": round(ic_all, 5),
        "ic_mean_quarter": round(float(ics.mean()), 5),
        "ir_quarter": round(float(ics.mean() / (ics.std() + EPS)), 3),
        "positive_quarter_pct": round(float((ics > 0).mean() * 100), 1),
        "nonzero_pct": round(float((s != 0).mean() * 100), 2),
    })
OUT["news_features"] = news_ic

# ------------------------------------------------- 5. existing macro ablation
p = pd.read_csv(R / "edge_pnl_daily.csv").rename(columns={"Unnamed: 0": "date"})
p["date"] = pd.to_datetime(p["date"], errors="coerce")
p = p.dropna(subset=["date"]).set_index("date")
cols = [c for c in ["ML_price", "ML_price+fund", "ML_all(+macro)", "RIDGE_price+fund"] if c in p.columns]
ab = {}
for c in cols:
    x = pd.to_numeric(p[c], errors="coerce").dropna()
    ab[c] = {"mean_daily_pct": round(float(x.mean() * 100), 4),
             "ann_sharpe": round(float(x.mean() / (x.std() + EPS) * np.sqrt(365)), 3),
             "cum_pct": round(float((1 + x).prod() * 100 - 100), 2),
             "n": int(len(x))}
if "ML_price" in p.columns and "ML_all(+macro)" in p.columns:
    d = (pd.to_numeric(p["ML_all(+macro)"], errors="coerce")
         - pd.to_numeric(p["ML_price"], errors="coerce")).dropna()
    ab["diff_macro_minus_price"] = {
        "mean_daily_pct": round(float(d.mean() * 100), 4),
        "t_stat": round(float(d.mean() / (d.std() + EPS) * np.sqrt(len(d))), 3),
        "positive_pct": round(float((d > 0).mean() * 100), 2),
    }
OUT["existing_macro_ablation"] = ab

(R / "external_analysis.json").write_text(
    json.dumps(OUT, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

print("=== trade_count ===", OUT["trade_count"])
print("=== cross_asset ===", OUT["cross_asset"])
print("=== news_corpus ===", OUT["news_corpus"])
print("\n=== correlations ===")
print(pd.DataFrame(OUT["correlations"]).to_string(index=False))
print("\n=== news features IC ===")
print(pd.DataFrame(OUT["news_features"]).to_string(index=False))
print("\n=== existing macro ablation ===")
print(json.dumps(OUT["existing_macro_ablation"], indent=2))
