"""OSH research system v3 — corrected, fully causal, GO/NO-GO measurement.

Fixes over pipeline_v2.py / ablation_experiment.py:
  1. Threshold is calibrated on the TRAIN tail and literally USED in test
     (v2 selected the top 5% of the whole test window -> look-ahead).
  2. Triple barrier uses future HIGH/LOW for TP/SL, not close only.
     If both barriers are touched in one bar, the LOSS is assumed (conservative).
  3. Early-stopping validation is purged by the label horizon.
  4. External features carry an age/staleness flag and a max-age cap.
  5. Signals are taken CAUSALLY in chronological order (no whole-window ranking).
  6. LONG and SHORT are both modelled; NO TRADE is an explicit outcome.
  7. Expectancy / profit factor / block-bootstrap CI / untouched holdout.
  8. Horizon sweep 1h / 4h / 1d to measure cost-per-move.

ALL NUMBERS ARE HISTORICAL PAPER TRADES. No live order is placed.
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
EPS = 1e-12
COST_BPS = 10.0
TOP_PCT = 0.05
N_FOLDS = 5
HOLDOUT_DAYS = 180
MAX_EXT_AGE_MIN = 60          # external quote older than 1h is stale
HORIZONS = [(4, "1h"), (16, "4h"), (96, "1d")]
VOL_WIN = 96


# ------------------------------------------------------------------ data
def load_btc() -> pd.DataFrame:
    eng = create_engine("postgresql+psycopg2://postgres@127.0.0.1:5433/osh")
    d = pd.read_sql("""SELECT to_timestamp(open_time/1000.0) AS t, open, high, low,
                       close, volume, trade_count, quote_volume,
                       taker_buy_base_volume, taker_buy_quote_volume
                       FROM public.btcusdt_15m ORDER BY open_time""", eng)
    d = d.rename(columns={"t": "candle_time"})
    d["candle_time"] = pd.to_datetime(d["candle_time"], utc=True)
    for c in ["open", "high", "low", "close", "volume", "trade_count", "quote_volume",
              "taker_buy_base_volume", "taker_buy_quote_volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna().sort_values("candle_time").reset_index(drop=True)


# --------------------------------------------- high/low triple barrier
def barrier_labels(df: pd.DataFrame, h: int, cost_ret: float) -> pd.DataFrame:
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    n = len(df)

    ret_cc = pd.Series(c).pct_change()
    sigma = (pd.Series(np.log(c / np.roll(c, 1)))
             .rolling(VOL_WIN, min_periods=VOL_WIN // 2).std().to_numpy()
             * np.sqrt(h))
    upper = np.maximum(sigma, cost_ret * 1.5)      # TP must clear costs
    lower = -upper                                # symmetric stop

    entry = np.roll(o, -1)                        # next candle open
    entry[-1] = np.nan
    first_tp = np.full(n, h + 1, dtype=np.int32)
    first_sl = np.full(n, h + 1, dtype=np.int32)
    for j in range(1, h + 1):
        hj = np.full(n, np.nan); hj[:-j] = hi[j:]
        lj = np.full(n, np.nan); lj[:-j] = lo[j:]
        rh = hj / entry - 1
        rl = lj / entry - 1
        tp = (first_tp == h + 1) & (rh >= upper)
        sl = (first_sl == h + 1) & (rl <= lower)
        first_tp[tp] = j
        first_sl[sl] = j

    hit_sl = first_sl <= h
    hit_tp = first_tp <= h
    sl_first = hit_sl & (first_sl <= first_tp)     # conservative tie -> SL wins
    tp_first = hit_tp & (~sl_first)

    label3 = np.where(sl_first, -1, np.where(tp_first, 1, 0))
    timeout = np.full(n, np.nan); timeout[:-h] = c[h:] / entry[:-h] - 1
    exit_ret = np.where(sl_first, lower, np.where(tp_first, upper, timeout))

    valid = np.isfinite(exit_ret) & np.isfinite(sigma) & np.isfinite(entry) & (entry > 0)
    return pd.DataFrame({
        "label3": label3,
        "exit_ret": exit_ret,
        "sigma": sigma,
        "sample_weight": (1.0 / h) * np.clip(np.abs(exit_ret) / (sigma + EPS), 0.25, 3.0),
        "valid": valid,
    })


# --------------------------------------------- external, staleness aware
def external_features(df: pd.DataFrame) -> pd.DataFrame:
    path = R / "WIDE_GLD_AUX_2MIN.csv"
    if not path.exists():
        return pd.DataFrame(index=df.index)
    g = pd.read_csv(path)
    g["datetime"] = pd.to_datetime(g["datetime"], utc=True)
    g = g.sort_values("datetime").set_index("datetime")
    closes = [c for c in g.columns if c.endswith("_Close")]
    g15 = g[closes].resample("15min").last()
    g15.index.name = "candle_time"
    m = df[["candle_time"]].merge(g15.reset_index(), on="candle_time", how="left")

    out = pd.DataFrame(index=df.index)
    for c in closes:
        a = c.replace("_Close", "")
        raw = m[c]
        last_time = pd.to_datetime(
            pd.Series(np.where(raw.notna(), m["candle_time"], pd.NaT), index=df.index),
            utc=True, errors="coerce").ffill()
        age = (df["candle_time"] - last_time).dt.total_seconds() / 60.0
        p = raw.ffill()
        r1 = np.log(p / p.shift(1))
        r4 = np.log(p / p.shift(4))
        stale = (age > MAX_EXT_AGE_MIN) | age.isna()
        out[f"{a}_ret1"] = r1.where(~stale)
        out[f"{a}_ret4"] = r4.where(~stale)
        out[f"{a}_age_min"] = np.log1p(age.clip(0, 1440))
        out[f"{a}_stale"] = stale.astype("int8")
    return out.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------- model with purged ES
def fit_model(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, h: int):
    from lightgbm import LGBMClassifier, early_stopping
    import inspect

    cut = int(len(X) * 0.85)
    embargo = h + 2
    tr_end = max(1, cut - embargo)
    model = LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=400,
        learning_rate=0.03, num_leaves=8, max_depth=4,
        min_child_samples=300, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=0.5,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    kwargs = {"eval_metric": "multi_logloss",
              "callbacks": [early_stopping(30, verbose=False)]}
    params = inspect.signature(model.fit).parameters
    if "eval_X" in params:
        kwargs["eval_X"] = X.iloc[cut:]
        kwargs["eval_y"] = y[cut:]
    else:
        kwargs["eval_set"] = [(X.iloc[cut:], y[cut:])]
    model.fit(X.iloc[:tr_end], y[:tr_end], sample_weight=w[:tr_end], **kwargs)
    return model


# --------------------------------------------- causal trade selection
def select_trades(p_up, p_dn, exit_ret, h, cost, thr_up, thr_dn):
    """Chronological, fully causal. No whole-window ranking."""
    idx, sides, gross = [], [], []
    last = -10 ** 9
    n = len(p_up)
    for i in range(n - h):
        if i - last < h:
            continue
        up_ok = p_up[i] >= thr_up
        dn_ok = p_dn[i] >= thr_dn
        if not up_ok and not dn_ok:
            continue
        side = 1 if (up_ok and (not dn_ok or p_up[i] >= p_dn[i])) else -1
        r = exit_ret[i]
        if not np.isfinite(r):
            continue
        idx.append(i); sides.append(side)
        gross.append(r if side == 1 else -r)
        last = i
    if not idx:
        return np.array([]), np.array([]), np.array([])
    g = np.asarray(gross)
    return np.asarray(idx), np.asarray(sides), g - cost


def metrics(net: np.ndarray, gross: np.ndarray, cost_bps: float, h: int,
            benchmark: np.ndarray, n_bars: int) -> dict:
    if len(net) == 0:
        return {"trades": 0}
    win = net > 0
    pw = float(win.mean())
    avg_win_g = float(gross[win].mean()) if win.any() else 0.0
    avg_loss_g = float(-gross[~win].mean()) if (~win).any() else 0.0
    pos = float(net[win].sum()); neg = float(-net[~win].sum())
    return {
        "trades": int(len(net)),
        "expectancy_bps": float(net.mean() * 10_000),
        "win_rate_pct": pw * 100,
        "avg_win_gross_bps": avg_win_g * 10_000,
        "avg_loss_gross_bps": avg_loss_g * 10_000,
        "expectancy_decomposition_bps": (pw * avg_win_g - (1 - pw) * avg_loss_g
                                         - cost_bps / 10_000) * 10_000,
        "profit_factor": float(pos / neg) if neg > 0 else float("inf"),
        "benchmark_mean_net_bps": float(benchmark.mean() * 10_000),
        "edge_vs_benchmark_bps": float((net.mean() - benchmark.mean()) * 10_000),
        "no_trade_pct": round(float(1 - len(net) / max(1, n_bars)) * 100, 1),
    }


def block_bootstrap_ci(net: np.ndarray, block: int = 20, reps: int = 2000,
                       seed: int = 42) -> tuple[float, float]:
    if len(net) < block * 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(net)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(reps)
    for b in range(reps):
        starts = rng.integers(0, max(1, n - block), size=n_blocks)
        sample = np.concatenate([net[s:s + block] for s in starts])[:n]
        means[b] = sample.mean()
    return (float(np.percentile(means, 2.5) * 10_000),
            float(np.percentile(means, 97.5) * 10_000))


# ------------------------------------------------------------------ main
def main() -> None:
    df = load_btc()
    print(f"[data] rows={len(df):,}  {df.candle_time.iloc[0]} -> {df.candle_time.iloc[-1]}")
    X_base, feats = pv.make_features(df)
    X_ext = external_features(df)
    print(f"[features] base={X_base.shape[1]}  external={X_ext.shape[1]}")

    holdout_bars = 96 * HOLDOUT_DAYS
    dev_end = len(df) - holdout_bars
    print(f"[split] development ends {df.candle_time.iloc[dev_end]} | "
          f"untouched holdout {holdout_bars:,} bars "
          f"({df.candle_time.iloc[dev_end]} -> {df.candle_time.iloc[-1]})")

    results = {"cost_bps": COST_BPS, "top_pct": TOP_PCT, "holdout_days": HOLDOUT_DAYS,
               "horizons": {}, "configs": ["BASE", "BASE_XASSET_STALE_AWARE"]}

    configs = {"BASE": X_base,
               "BASE_XASSET_STALE_AWARE": pd.concat([X_base, X_ext], axis=1)}

    for h, hname in HORIZONS:
        lab = barrier_labels(df, h, COST_BPS / 10_000)
        y3 = lab["label3"].to_numpy()
        y_cls = np.where(y3 == -1, 0, np.where(y3 == 0, 1, 2)).astype("int8")
        w = lab["sample_weight"].fillna(0).to_numpy()
        raw = lab["exit_ret"].to_numpy()
        valid = lab["valid"].to_numpy()

        results["horizons"][hname] = {"bars": h, "configs": {}}
        print(f"\n=== horizon {hname} ({h} bars) ===")
        print(f"    label mix: long={(y3==1).mean()*100:.1f}%  "
              f"short={(y3==-1).mean()*100:.1f}%  noise={(y3==0).mean()*100:.1f}%")

        for cfg_name, X in configs.items():
            Xf = X.replace([np.inf, -np.inf], np.nan)
            # Only BASE features must be present; missing external data is
            # filled with 0 and explicitly flagged by the *_stale columns.
            ok = valid & X_base.notna().all(axis=1).to_numpy()
            Xi = Xf.loc[ok].reset_index(drop=True).fillna(0)
            yi = y_cls[ok]; wi = w[ok]; ri = raw[ok]
            ti = df["candle_time"].loc[ok].reset_index(drop=True)
            cut_idx = int((ti < df["candle_time"].iloc[dev_end]).sum())

            folds = pv.make_folds(cut_idx, 96 * 365, 96 * 90, h + 2, N_FOLDS)
            all_net, all_gross, fold_rows = [], [], []
            last_model, last_thr = None, None

            for k, (tr0, tr1, te0, te1) in enumerate(folds, 1):
                model = fit_model(Xi.iloc[tr0:tr1], yi[tr0:tr1], wi[tr0:tr1], h)
                cal0 = max(tr0, tr1 - 96 * 90)
                pc = model.predict_proba(Xi.iloc[cal0:tr1])
                thr_up = float(np.quantile(pc[:, 2], 1 - TOP_PCT))
                thr_dn = float(np.quantile(pc[:, 0], 1 - TOP_PCT))
                p = model.predict_proba(Xi.iloc[te0:te1])
                idx, sides, net = select_trades(p[:, 2], p[:, 0],
                                                ri[te0:te1], h,
                                                COST_BPS / 10_000, thr_up, thr_dn)
                gross = net + COST_BPS / 10_000
                bench_i = np.arange(0, te1 - te0 - h, h)
                bench = ri[te0:te1][bench_i] - COST_BPS / 10_000
                m = metrics(net, gross, COST_BPS, h, bench, te1 - te0)
                m.update({"fold": k, "start": str(ti.iloc[te0])[:10],
                          "frozen_thr_up": round(thr_up, 5),
                          "frozen_thr_dn": round(thr_dn, 5),
                          "test_pass_rate_pct": round(float(
                              ((p[:, 2] >= thr_up) | (p[:, 0] >= thr_dn)).mean() * 100), 2)})
                fold_rows.append(m)
                if len(net):
                    all_net.append(net); all_gross.append(gross)
                last_model, last_thr = model, (thr_up, thr_dn)

            net_all = np.concatenate(all_net) if all_net else np.array([])
            gross_all = np.concatenate(all_gross) if all_gross else np.array([])
            nets = [f.get("expectancy_bps", -999) for f in fold_rows]
            lo, hi = block_bootstrap_ci(net_all)

            # ---- untouched holdout, evaluated ONCE with the last dev model
            holdout = {"trades": 0}
            if last_model is not None and cut_idx < len(Xi):
                ph = last_model.predict_proba(Xi.iloc[cut_idx:])
                ih, sh, nh = select_trades(ph[:, 2], ph[:, 0],
                                           ri[cut_idx:], h, COST_BPS / 10_000,
                                           last_thr[0], last_thr[1])
                gh = nh + COST_BPS / 10_000
                bh = ri[cut_idx:][np.arange(0, len(ri) - cut_idx - h, h)] - COST_BPS / 10_000
                holdout = metrics(nh, gh, COST_BPS, h, bh, len(ri) - cut_idx)
                holdout["note"] = "frozen model + frozen thresholds, no refit, no tuning"

            summary = {
                "feature_count": int(Xi.shape[1]),
                "folds": fold_rows,
                "positive_folds_pct": round(float(np.mean([v > 0 for v in nets]) * 100), 1),
                "median_fold_expectancy_bps": round(float(np.median(nets)), 3),
                "pooled_expectancy_bps": round(float(net_all.mean() * 10_000), 3) if len(net_all) else None,
                "pooled_profit_factor": round(float(
                    net_all[net_all > 0].sum() / -net_all[net_all < 0].sum()), 3)
                    if len(net_all) and (net_all < 0).any() else None,
                "bootstrap_ci95_bps": [round(lo, 3), round(hi, 3)],
                "total_trades": int(len(net_all)),
                "untouched_holdout": holdout,
            }
            results["horizons"][hname]["configs"][cfg_name] = summary
            print(f"  {cfg_name:<24} median={summary['median_fold_expectancy_bps']:+.2f}bps  "
                  f"pos_folds={summary['positive_folds_pct']:.0f}%  "
                  f"PF={summary['pooled_profit_factor']}  CI=[{lo:+.2f},{hi:+.2f}]  "
                  f"holdout={holdout.get('expectancy_bps', 'n/a')}bps")

    # GO/NO-GO gate (pre-registered criteria)
    gates = []
    for hname, hd in results["horizons"].items():
        for cfg, s in hd["configs"].items():
            ci = s["bootstrap_ci95_bps"]
            ok = {
                "median_net_gt_0": s["median_fold_expectancy_bps"] > 0,
                "positive_folds_ge_75": s["positive_folds_pct"] >= 75,
                "profit_factor_gt_1": (s["pooled_profit_factor"] or 0) > 1,
                "bootstrap_lower_gt_0": bool(ci[0] > 0) if ci and np.isfinite(ci[0]) else False,
                "beats_benchmark": bool(np.mean([f.get("edge_vs_benchmark_bps", -1)
                                                 for f in s["folds"]]) > 0),
            }
            gates.append({"horizon": hname, "config": cfg, "checks": ok,
                          "verdict": "GO" if all(ok.values()) else "NO-GO"})
    results["gates"] = gates
    results["lookahead_checks"] = {
        "threshold_frozen_from_train_and_used": True,
        "causal_chronological_selection": True,
        "early_stopping_purged": True,
        "barriers_use_high_low": True,
        "external_staleness_capped": True,
        "untouched_holdout": True,
    }
    (R / "v3_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print("\n=== GO / NO-GO ===")
    for g in gates:
        print(f"  {g['horizon']:<4} {g['config']:<24} {g['verdict']}  "
              f"passed={sum(g['checks'].values())}/5")
    print("[done] v3_results.json")


if __name__ == "__main__":
    main()
