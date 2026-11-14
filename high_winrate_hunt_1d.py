"""OSH high-win-rate hunt — 1d horizon, PRE-REGISTERED, paper only.

Goal (fixed BEFORE running): find whether ANY pre-registered configuration can
reach >=60% holdout win rate AND positive net expectancy AND PF>1 under
realistic costs. No config is re-tuned after seeing holdout. Each config
touches the 180-day untouched holdout exactly ONCE.

Pre-registered configs (7 total, all reported, no cherry-picking):
  C1 sym  (1.00, 1.00)  — reference, equivalent to v3 1d
  C2 asym (0.75, 1.00)
  C3 asym (0.50, 1.00)
  C4 asym (0.75, 1.50)
  C5 asym (0.50, 1.50)
  C6 naive(0.25, 3.00)  — TRAP DEMO: mechanically high win rate, expected to lose
  C7 asym (0.75, 1.00) + crypto-native funding features

Selection: frozen train-tail quantile thresholds, top_pct=0.10 per side,
non-overlapping trades (max 1 trade/day at h=96 bars).

Primary cost scenario: maker/maker 4 bps round-trip + 2 bps slippage + REAL
funding PnL from binance_btcusdt_funding_history.csv.
Stress scenario: taker/taker 10 bps round-trip (flat).

This script never places orders. live_orders = False.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline_v2 as pv  # noqa: E402
import research_v3 as rv  # noqa: E402

BARS_PER_DAY = 96
H = 96                      # 1 day horizon, fixed
TOP_PCT = 0.10              # per-side selection quantile, fixed
FOLDS = 5
TRAIN_DAYS = 365
TEST_DAYS = 90
HOLDOUT_DAYS = 180
VOL_WIN = 96
DATA_END_UTC = pd.Timestamp("2026-10-31 23:45:00", tz="UTC")  # current month excluded (as in user's stress test)

# Pre-registered grid: (name, tp_mult, sl_mult, use_funding_features)
CONFIGS = [
    ("C1_sym_1.00_1.00",        1.00, 1.00, False),
    ("C2_asym_0.75_1.00",       0.75, 1.00, False),
    ("C3_asym_0.50_1.00",       0.50, 1.00, False),
    ("C4_asym_0.75_1.50",       0.75, 1.50, False),
    ("C5_asym_0.50_1.50",       0.50, 1.50, False),
    ("C6_naive_0.25_3.00_TRAP", 0.25, 3.00, False),
    ("C7_asym_0.75_1.00_FUND",  0.75, 1.00, True),
]

# Cost scenarios (round-trip bps, flat part). Primary adds real funding PnL.
FEE_BPS_PRIMARY = 6.0    # maker/maker 4 + 2 slip
FEE_BPS_STRESS = 10.0    # taker/taker flat

CAPITAL_USD = 100.0

FUNDING_CSV = HERE / "binance_btcusdt_funding_history.csv"


# ------------------------------------------------------------------ labels
def asymmetric_barrier_labels(df: pd.DataFrame, h: int, tp_mult: float, sl_mult: float) -> pd.DataFrame:
    """Long- and short-perspective first-barrier events in ONE pass.

    Long:  TP = +tp_mult*sigma, SL = -sl_mult*sigma.
    Short: TP = -tp_mult*sigma, SL = +sl_mult*sigma.
    Same conservative rules as research_v3.barrier_labels:
    next-open entry, same-bar double touch => stop, stop gaps fill at the
    worse open, TP gaps do not assume improvement, timeout at horizon close.
    """
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    n = len(df)

    log_ret = np.log(pd.Series(c) / pd.Series(c).shift(1))
    sigma = (log_ret.rolling(VOL_WIN, min_periods=max(2, VOL_WIN // 2)).std().to_numpy() * np.sqrt(h))
    entry = np.full(n, np.nan)
    entry[:-1] = o[1:]

    out = {
        "long_label": np.zeros(n, dtype=np.int8),
        "long_exit": np.full(n, np.nan),
        "short_label": np.zeros(n, dtype=np.int8),
        "short_exit": np.full(n, np.nan),
        "exit_reason": np.full(n, "invalid", dtype=object),
        "holding_bars": np.full(n, np.nan),
        "sigma": sigma,
        "entry_price": entry,
    }

    for i in range(n - h):
        e = entry[i]
        s = sigma[i]
        if not np.isfinite(e) or e <= 0 or not np.isfinite(s):
            continue
        long_tp = e * (1.0 + tp_mult * s)
        long_sl = e * (1.0 - sl_mult * s)
        short_tp = e * (1.0 - tp_mult * s)
        short_sl = e * (1.0 + sl_mult * s)

        long_done = short_done = False
        hold = h
        for j in range(1, h + 1):
            k = i + j
            # ---- long perspective
            if not long_done:
                hit_tp = hi[k] >= long_tp
                hit_sl = lo[k] <= long_sl
                if hit_tp and hit_sl:  # same-bar double touch: conservative loss
                    out["long_label"][i] = -1
                    out["long_exit"][i] = min(o[k], long_sl) / e - 1.0
                    long_done = True
                elif hit_sl:
                    out["long_label"][i] = -1
                    out["long_exit"][i] = min(o[k], long_sl) / e - 1.0
                    long_done = True
                elif hit_tp:
                    out["long_label"][i] = 1
                    out["long_exit"][i] = max(o[k], long_tp) / e - 1.0
                    long_done = True
            # ---- short perspective
            if not short_done:
                hit_tp = lo[k] <= short_tp
                hit_sl = hi[k] >= short_sl
                if hit_tp and hit_sl:
                    out["short_label"][i] = -1
                    out["short_exit"][i] = e / max(o[k], short_sl) - 1.0
                    short_done = True
                elif hit_sl:
                    out["short_label"][i] = -1
                    out["short_exit"][i] = e / max(o[k], short_sl) - 1.0
                    short_done = True
                elif hit_tp:
                    out["short_label"][i] = 1
                    out["short_exit"][i] = e / min(o[k], short_tp) - 1.0
                    short_done = True
            if long_done and short_done:
                hold = j
                break

        if not long_done:
            out["long_label"][i] = 0
            out["long_exit"][i] = c[i + h] / e - 1.0
        if not short_done:
            out["short_label"][i] = 0
            out["short_exit"][i] = e / c[i + h] - 1.0

        # shared diagnostics (long perspective drives exit_reason)
        out["holding_bars"][i] = hold
        lr = out["long_label"][i]
        out["exit_reason"][i] = "take_profit" if lr == 1 else ("stop_loss" if lr == -1 else "time")

    valid = np.isfinite(out["long_exit"]) & np.isfinite(sigma) & np.isfinite(entry) & (entry > 0)
    weight = (1.0 / h) * np.clip(np.abs(out["long_exit"]) / (sigma + 1e-12), 0.25, 3.0)
    weight[~valid] = np.nan
    out["sample_weight"] = weight
    out["valid"] = valid
    return pd.DataFrame(out, index=df.index)


# ------------------------------------------------------------------ funding
def load_funding(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path)
    f["funding_time"] = pd.to_datetime(f["funding_time"], utc=True)
    f = f.sort_values("funding_time").reset_index(drop=True)
    f["funding_rate"] = pd.to_numeric(f["funding_rate"], errors="coerce")
    return f.dropna(subset=["funding_rate"])


def funding_features(df: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Causal features from SETTLED funding events only (asof backward)."""
    left = pd.DataFrame({"signal_time": pd.to_datetime(df["candle_time"], utc=True) + pd.Timedelta(minutes=15)})
    right = funding[["funding_time", "funding_rate"]].rename(columns={"funding_time": "settled_at"})
    merged = pd.merge_asof(left.sort_values("signal_time"), right,
                           left_on="signal_time", right_on="settled_at", direction="backward")
    rate = merged["funding_rate"].astype(float)
    settled_at = merged["settled_at"]

    out = pd.DataFrame(index=df.index)
    out["fund_last"] = rate.to_numpy()
    out["fund_mean_1d"] = rate.rolling(3, min_periods=1).mean().to_numpy()
    out["fund_mean_3d"] = rate.rolling(9, min_periods=3).mean().to_numpy()
    m90 = rate.rolling(90, min_periods=30).mean()
    s90 = rate.rolling(90, min_periods=30).std()
    out["fund_z_30d"] = ((rate - m90) / (s90 + 1e-12)).to_numpy()
    age_h = (merged["signal_time"] - settled_at).dt.total_seconds() / 3600.0
    out["fund_hours_since"] = age_h.clip(lower=0).to_numpy()
    # deterministic schedule 00/08/16 UTC -> hours to next settlement (no leakage)
    t = pd.to_datetime(df["candle_time"], utc=True) + pd.Timedelta(minutes=15)
    hour_frac = t.dt.hour + t.dt.minute / 60.0
    to_next = (8.0 - (hour_frac % 8.0)) % 8.0
    out["fund_hours_to_next"] = to_next.to_numpy()
    return out.replace([np.inf, -np.inf], np.nan)


def funding_pnl_for_trade(side: int, entry_time: pd.Timestamp, exit_time: pd.Timestamp,
                          fund_times_ns: np.ndarray, fund_rates: np.ndarray) -> float:
    """Signed funding cash flow as a return fraction. Long pays positive rates."""
    lo = np.searchsorted(fund_times_ns, entry_time.value, side="right")
    hi = np.searchsorted(fund_times_ns, exit_time.value, side="right")
    total = float(fund_rates[lo:hi].sum()) if hi > lo else 0.0
    return -side * total


# ------------------------------------------------------------------ selection
def select_dual(p_long_up, p_short_up, thr_long, thr_short,
                long_exit, short_exit, h, fee_ret, funding_pnls=None):
    """Frozen thresholds; non-overlap; side = bigger normalized exceedance."""
    n = len(p_long_up)
    idx, sides, gross, net = [], [], [], []
    last = -10**9
    for i in range(max(0, n - h)):
        if i - last < h:
            continue
        lu = np.isfinite(p_long_up[i]) and p_long_up[i] >= thr_long
        su = np.isfinite(p_short_up[i]) and p_short_up[i] >= thr_short
        if not lu and not su:
            continue
        if lu and su:
            ex_l = (p_long_up[i] - thr_long) / max(1e-9, 1.0 - thr_long)
            ex_s = (p_short_up[i] - thr_short) / max(1e-9, 1.0 - thr_short)
            side = 1 if ex_l >= ex_s else -1
        else:
            side = 1 if lu else -1
        g = long_exit[i] if side == 1 else short_exit[i]
        if not np.isfinite(g):
            continue
        fund = funding_pnls[i] if funding_pnls is not None else 0.0
        idx.append(i)
        sides.append(side)
        gross.append(g)
        net.append(g - fee_ret + fund)
        last = i
    return (np.asarray(idx, dtype=int), np.asarray(sides, dtype=np.int8),
            np.asarray(gross, dtype=float), np.asarray(net, dtype=float))


# ------------------------------------------------------------------ metrics
def trade_metrics(net, gross, benchmark, n_bars):
    if len(net) == 0:
        return {"trades": 0}
    net = np.asarray(net, float)
    gross = np.asarray(gross, float)
    win = net > 0
    pos = float(net[win].sum())
    neg = float(-net[~win].sum())
    exp = float(net.mean() * 1e4)
    bench = float(np.mean(benchmark) * 1e4) if len(benchmark) else None
    return {
        "trades": int(len(net)),
        "trades_per_day": round(len(net) / max(1e-9, n_bars / BARS_PER_DAY), 3),
        "win_rate_pct": round(float(win.mean() * 100), 2),
        "expectancy_bps": round(exp, 3),
        "avg_win_net_bps": round(float(net[win].mean() * 1e4), 2) if win.any() else 0.0,
        "avg_loss_net_bps": round(float(-net[~win].mean() * 1e4), 2) if (~win).any() else 0.0,
        "payoff_ratio": round(float(net[win].mean() / -net[~win].mean()), 3) if win.any() and (~win).any() else None,
        "profit_factor": round(pos / neg, 4) if neg > 0 else None,
        "benchmark_mean_net_bps": round(bench, 3) if bench is not None else None,
        "edge_vs_benchmark_bps": round(exp - bench, 3) if bench is not None else None,
        "final_balance_100usd": round(float(CAPITAL_USD * np.prod(1.0 + net)), 2),
        "no_trade_pct": round(float(1 - len(net) / max(1, n_bars)) * 100, 1),
    }


def bootstrap_ci(net, block=20, reps=2000, seed=42):
    net = np.asarray(net, float)
    if len(net) < block * 2:
        return None, None
    rng = np.random.default_rng(seed)
    n = len(net)
    nb = int(np.ceil(n / block))
    means = np.empty(reps)
    for b in range(reps):
        starts = rng.integers(0, n - block + 1, size=nb)
        means[b] = np.concatenate([net[s:s + block] for s in starts])[:n].mean()
    lo, hi_ = np.percentile(means, [2.5, 97.5])
    return round(float(lo * 1e4), 3), round(float(hi_ * 1e4), 3)


# ------------------------------------------------------------------ main run
def main():
    print("=== OSH HIGH-WINRATE HUNT 1d — PRE-REGISTERED, PAPER ONLY ===", flush=True)
    print(f"horizon=1d | top_pct={TOP_PCT}/side | folds={FOLDS} | holdout={HOLDOUT_DAYS}d "
          f"| fee_primary={FEE_BPS_PRIMARY}bps+funding | stress={FEE_BPS_STRESS}bps", flush=True)

    db_url = "postgresql+psycopg2://postgres@127.0.0.1:5433/osh"
    df = rv.load_btc(db_url, "public.btcusdt_15m")
    df = df[df["candle_time"] <= DATA_END_UTC].reset_index(drop=True)
    print(f"data: {len(df)} rows, {df['candle_time'].iloc[0]} -> {df['candle_time'].iloc[-1]}", flush=True)

    funding = load_funding(FUNDING_CSV)
    fund_times_ns = funding["funding_time"].values.astype("datetime64[ns]").astype(np.int64)
    fund_rates = funding["funding_rate"].to_numpy(float)
    print(f"funding: {len(funding)} settlements", flush=True)

    X_base, base_feats = pv.make_features(df)
    X_fund = funding_features(df, funding)
    print(f"features: BASE={X_base.shape[1]}, FUNDING={X_fund.shape[1]}", flush=True)

    holdout_bars = BARS_PER_DAY * HOLDOUT_DAYS
    dev_end = len(df) - holdout_bars
    dev_cut_time = df["candle_time"].iloc[dev_end]

    results = {"metadata": {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_end_utc": str(DATA_END_UTC),
        "rows": int(len(df)),
        "preregistered_configs": [c[0] for c in CONFIGS],
        "selection": f"frozen train-tail quantile top {TOP_PCT} per side, non-overlap",
        "costs": f"primary {FEE_BPS_PRIMARY}bps + real funding; stress {FEE_BPS_STRESS}bps flat",
        "live_orders": False,
    }, "configs": {}, "gates": []}

    for name, tp_m, sl_m, use_fund in CONFIGS:
        print(f"\n--- {name} (tp={tp_m}σ, sl={sl_m}σ, funding_features={use_fund}) ---", flush=True)
        lab = asymmetric_barrier_labels(df, H, tp_m, sl_m)
        valid = lab["valid"].to_numpy(bool)

        X = pd.concat([X_base, X_fund], axis=1) if use_fund else X_base
        Xf = X.replace([np.inf, -np.inf], np.nan)
        ok = valid & X_base.notna().all(axis=1).to_numpy()
        Xi = Xf.loc[ok].reset_index(drop=True).fillna(0.0)
        ti = df["candle_time"].loc[ok].reset_index(drop=True)
        li = lab.loc[ok].reset_index(drop=True)
        cut_idx = int((ti < dev_cut_time).sum())

        y_long = np.where(li["long_label"] == -1, 0, np.where(li["long_label"] == 0, 1, 2)).astype("int8")
        y_short = np.where(li["short_label"] == -1, 0, np.where(li["short_label"] == 0, 1, 2)).astype("int8")
        w = li["sample_weight"].to_numpy(float)
        long_exit = li["long_exit"].to_numpy(float)
        short_exit = li["short_exit"].to_numpy(float)

        fold_list = pv.make_folds(cut_idx, BARS_PER_DAY * TRAIN_DAYS, BARS_PER_DAY * TEST_DAYS, H + 2, FOLDS)
        symmetric = abs(tp_m - sl_m) < 1e-12

        fold_rows, all_net, all_gross = [], [], []
        last_state = None
        for fold_no, (tr0, tr1, te0, te1) in enumerate(fold_list, 1):
            cal0 = max(tr0 + 100, tr1 - BARS_PER_DAY * min(90, TRAIN_DAYS))
            try:
                m_long = rv.fit_model(Xi.iloc[tr0:cal0], y_long[tr0:cal0], w[tr0:cal0], H)
                m_short = m_long if symmetric else rv.fit_model(Xi.iloc[tr0:cal0], y_short[tr0:cal0], w[tr0:cal0], H)
            except ValueError as exc:
                print(f"  fold {fold_no}: TRAIN FAILED ({exc})", flush=True)
                continue

            p_cal_long = rv.predict_proba_3class(m_long, Xi.iloc[cal0:tr1])
            thr_long = float(np.quantile(p_cal_long[:, 2], 1.0 - TOP_PCT))
            if symmetric:
                thr_short = float(np.quantile(p_cal_long[:, 0], 1.0 - TOP_PCT))
            else:
                p_cal_short = rv.predict_proba_3class(m_short, Xi.iloc[cal0:tr1])
                thr_short = float(np.quantile(p_cal_short[:, 2], 1.0 - TOP_PCT))

            p_t_long = rv.predict_proba_3class(m_long, Xi.iloc[te0:te1])
            if symmetric:
                p_short_up = p_t_long[:, 0]
            else:
                p_t_short = rv.predict_proba_3class(m_short, Xi.iloc[te0:te1])
                p_short_up = p_t_short[:, 2]

            # per-trade funding pnl (only for bars we might trade)
            le = long_exit[te0:te1]
            se = short_exit[te0:te1]
            hold = li["holding_bars"].to_numpy(float)[te0:te1]
            times = ti.iloc[te0:te1].reset_index(drop=True)
            fund_arr = np.zeros(te1 - te0)
            for i in range(te1 - te0 - H):
                hb = hold[i]
                if not np.isfinite(hb):
                    continue
                t0 = times.iloc[i + 1]                      # entry at next open
                t1 = times.iloc[i + int(hb)] + pd.Timedelta(minutes=15)
                fund_arr[i] = funding_pnl_for_trade(1, t0, t1, fund_times_ns, fund_rates)  # long-perspective pnl

            idx, sides, gross, net = [], [], [], []
            last_e = -10**9
            for i in range(max(0, te1 - te0 - H)):
                if i - last_e < H:
                    continue
                lu = p_t_long[i, 2] >= thr_long
                su = p_short_up[i] >= thr_short
                if not lu and not su:
                    continue
                if lu and su:
                    ex_l = (p_t_long[i, 2] - thr_long) / max(1e-9, 1.0 - thr_long)
                    ex_s = (p_short_up[i] - thr_short) / max(1e-9, 1.0 - thr_short)
                    side = 1 if ex_l >= ex_s else -1
                else:
                    side = 1 if lu else -1
                g = le[i] if side == 1 else se[i]
                if not np.isfinite(g):
                    continue
                fpnl = fund_arr[i] * side  # fund_arr is long-perspective pnl; short flips sign
                idx.append(i)
                sides.append(side)
                gross.append(g)
                net.append(g - FEE_BPS_PRIMARY / 1e4 + fpnl)
                last_e = i
            idx = np.asarray(idx, int)
            sides = np.asarray(sides, np.int8)
            gross = np.asarray(gross, float)
            net = np.asarray(net, float)

            bench_idx = np.arange(0, max(0, te1 - te0 - H), H)
            benchmark = le[bench_idx] - FEE_BPS_PRIMARY / 1e4
            row = trade_metrics(net, gross, benchmark, te1 - te0)
            row["fold"] = fold_no
            row["test_start"] = str(times.iloc[0])
            row["frozen_thr_long"] = round(thr_long, 5)
            row["frozen_thr_short"] = round(thr_short, 5)
            row["long_trades"] = int((sides == 1).sum())
            row["short_trades"] = int((sides == -1).sum())
            row["note"] = "primary cost incl. real funding"
            fold_rows.append(row)
            if len(net):
                all_net.append(net)
                all_gross.append(gross)
            last_state = (m_long, m_short, thr_long, thr_short)
            print(f"  fold {fold_no}: trades={row['trades']} wr={row.get('win_rate_pct')}% "
                  f"exp={row.get('expectancy_bps')}bps PF={row.get('profit_factor')}", flush=True)

        pooled = np.concatenate(all_net) if all_net else np.array([], float)
        pooled_g = np.concatenate(all_gross) if all_gross else np.array([], float)
        ci_lo, ci_hi = bootstrap_ci(pooled)
        exps = [r["expectancy_bps"] for r in fold_rows if r.get("expectancy_bps") is not None]
        pos_folds = float(np.mean(np.asarray(exps) > 0) * 100) if exps else 0.0

        # ---------------- holdout: ONE touch
        holdout = {"trades": 0}
        holdout_stress = {"trades": 0}
        if last_state is not None and cut_idx < len(Xi):
            m_long, m_short, thr_long, thr_short = last_state
            p_h_long = rv.predict_proba_3class(m_long, Xi.iloc[cut_idx:])
            p_h_short_up = p_h_long[:, 0] if symmetric else rv.predict_proba_3class(m_short, Xi.iloc[cut_idx:])[:, 2]
            le_h = long_exit[cut_idx:]
            se_h = short_exit[cut_idx:]
            hold_h = li["holding_bars"].to_numpy(float)[cut_idx:]
            times_h = ti.iloc[cut_idx:].reset_index(drop=True)
            nh = len(le_h)
            idx, sides, gross_h, net_h, net_stress = [], [], [], [], []
            last_e = -10**9
            for i in range(max(0, nh - H)):
                if i - last_e < H:
                    continue
                lu = p_h_long[i, 2] >= thr_long
                su = p_h_short_up[i] >= thr_short
                if not lu and not su:
                    continue
                if lu and su:
                    ex_l = (p_h_long[i, 2] - thr_long) / max(1e-9, 1.0 - thr_long)
                    ex_s = (p_h_short_up[i] - thr_short) / max(1e-9, 1.0 - thr_short)
                    side = 1 if ex_l >= ex_s else -1
                else:
                    side = 1 if lu else -1
                g = le_h[i] if side == 1 else se_h[i]
                if not np.isfinite(g) or not np.isfinite(hold_h[i]):
                    continue
                t0 = times_h.iloc[i + 1]
                t1 = times_h.iloc[i + int(hold_h[i])] + pd.Timedelta(minutes=15)
                fpnl = funding_pnl_for_trade(side, t0, t1, fund_times_ns, fund_rates)
                idx.append(i)
                sides.append(side)
                gross_h.append(g)
                net_h.append(g - FEE_BPS_PRIMARY / 1e4 + fpnl)
                net_stress.append(g - FEE_BPS_STRESS / 1e4)
                last_e = i
            idx = np.asarray(idx, int)
            sides = np.asarray(sides, np.int8)
            net_h = np.asarray(net_h, float)
            gross_h = np.asarray(gross_h, float)
            net_stress = np.asarray(net_stress, float)
            bench_h = le_h[np.arange(0, max(0, nh - H), H)] - FEE_BPS_PRIMARY / 1e4
            holdout = trade_metrics(net_h, gross_h, bench_h, nh)
            holdout_stress = trade_metrics(net_stress, gross_h, bench_h, nh)
            # exit-reason breakdown
            if len(idx):
                ll = li["long_label"].to_numpy()[cut_idx:][idx]
                holdout["tp_share_pct"] = round(float((ll == 1).mean() * 100), 1)
                holdout["sl_share_pct"] = round(float((ll == -1).mean() * 100), 1)
                holdout["timeout_share_pct"] = round(float((ll == 0).mean() * 100), 1)
                holdout["long_trades"] = int((sides == 1).sum())
                holdout["short_trades"] = int((sides == -1).sum())
                hci_lo, hci_hi = bootstrap_ci(net_h)
                holdout["bootstrap_ci95_bps"] = [hci_lo, hci_hi]

        pos = float(pooled[pooled > 0].sum()) if len(pooled) else 0.0
        neg = float(-pooled[pooled < 0].sum()) if len(pooled) else 0.0
        summary = {
            "tp_mult": tp_m, "sl_mult": sl_m, "funding_features": bool(use_fund),
            "feature_count": int(Xi.shape[1]),
            "folds": fold_rows,
            "positive_folds_pct": round(pos_folds, 1),
            "median_fold_expectancy_bps": round(float(np.median(exps)), 3) if exps else None,
            "pooled_oos": trade_metrics(pooled, pooled_g,
                                        np.array([], float), sum(FOLDS * BARS_PER_DAY * TEST_DAYS for _ in [1])),
            "bootstrap_ci95_bps": [ci_lo, ci_hi],
            "untouched_holdout_primary": holdout,
            "untouched_holdout_stress10bps": holdout_stress,
        }
        summary["pooled_oos"]["profit_factor"] = round(pos / neg, 4) if neg > 0 else None
        results["configs"][name] = summary

        checks = {
            "oos_expectancy_positive": bool(summary["pooled_oos"].get("expectancy_bps") is not None
                                            and summary["pooled_oos"]["expectancy_bps"] > 0),
            "oos_pf_ge_1_2": bool(summary["pooled_oos"].get("profit_factor") is not None
                                  and summary["pooled_oos"]["profit_factor"] >= 1.2),
            "positive_folds_ge_80": bool(pos_folds >= 80),
            "bootstrap_lower_gt_0": bool(ci_lo is not None and ci_lo > 0),
            "holdout_winrate_ge_60": bool(holdout.get("win_rate_pct") is not None
                                          and holdout["win_rate_pct"] >= 60.0),
            "holdout_expectancy_positive": bool(holdout.get("expectancy_bps") is not None
                                                and holdout["expectancy_bps"] > 0),
            "holdout_pf_gt_1": bool(holdout.get("profit_factor") is not None
                                    and holdout["profit_factor"] > 1.0),
            "holdout_min_30_trades": bool(holdout.get("trades", 0) >= 30),
        }
        verdict = "GO" if all(checks.values()) else "NO-GO"
        results["gates"].append({"config": name, "checks": checks,
                                 "checks_passed": int(sum(checks.values())), "verdict": verdict})
        print(f"  => OOS wr={summary['pooled_oos'].get('win_rate_pct')}% exp={summary['pooled_oos'].get('expectancy_bps')}bps | "
              f"HOLDOUT wr={holdout.get('win_rate_pct')}% exp={holdout.get('expectancy_bps')}bps "
              f"bal$100={holdout.get('final_balance_100usd')} | {verdict} ({sum(checks.values())}/8)", flush=True)

    out = HERE / "high_winrate_hunt_1d.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"\n[report] {out}", flush=True)
    print("\n=== FINAL TABLE ===", flush=True)
    print(f"{'config':<28} {'OOS wr%':>8} {'OOS exp':>8} {'HLD wr%':>8} {'HLD exp':>8} {'HLD $100':>9} verdict", flush=True)
    for g in results["gates"]:
        s = results["configs"][g["config"]]
        h_ = s["untouched_holdout_primary"]
        print(f"{g['config']:<28} {s['pooled_oos'].get('win_rate_pct'):>8} "
              f"{s['pooled_oos'].get('expectancy_bps'):>8} {h_.get('win_rate_pct'):>8} "
              f"{h_.get('expectancy_bps'):>8} {h_.get('final_balance_100usd'):>9} {g['verdict']} ({g['checks_passed']}/8)", flush=True)


if __name__ == "__main__":
    main()
