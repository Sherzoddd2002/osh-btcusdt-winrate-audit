"""
OSH / BTCUSDT 15m — research pipeline v2
========================================

What this fixes compared with "OSH MAIN.ipynb":

1.  Only STATIONARY, RELATIVE, LAGGED features. No raw price / raw volume levels.
2.  Cost-aware TRIPLE-BARRIER labels instead of "is the next candle green?".
3.  PURGED WALK-FORWARD validation (many out-of-sample folds), not one 70/15/15 split.
4.  Threshold is calibrated on the TRAIN tail, never on the test fold.
5.  Non-overlapping trades + realistic round-trip cost (default 10 bps).
6.  Reports the DISTRIBUTION of fold results, not a single lucky number.
7.  Training-serving symmetry: one make_features() used everywhere.

Run locally without a database (self-test):

    python pipeline_v2.py --synthetic --rows 80000 --folds 4

Run against your PostgreSQL table:

    set OSH_DB_URL=postgresql+psycopg2://USER:PASSWORD@127.0.0.1:5433/osh
    python pipeline_v2.py --table public.btcusdt_15m --cost-bps 10 --out report.json

NOTE: this is a research/validation skeleton, NOT a trading bot.
No model guarantees profit. Crypto trading carries substantial risk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

TF_MIN = 15
BARS_PER_DAY = 96
EPS = 1e-12


# ---------------------------------------------------------------------
# 0. DATA
# ---------------------------------------------------------------------

REQUIRED = [
    "candle_time", "open", "high", "low", "close", "volume",
    "quote_volume", "trade_count",
    "taker_buy_base_volume", "taker_buy_quote_volume",
]


def load_from_db(db_url: str, table: str) -> pd.DataFrame:
    from sqlalchemy import create_engine

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*[.][A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError("Unsafe table name. Expected schema.table.")
    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            df = pd.read_sql_query(
                f"""
                SELECT to_timestamp(open_time / 1000.0) AS candle_time, *
                FROM {table}
                ORDER BY open_time ASC
                """,
                connection,
            )
    finally:
        engine.dispose()
    return _normalise(df)


def load_synthetic(n: int = 80_000, seed: int = 7) -> pd.DataFrame:
    """A random walk with volatility clustering + mild short-term reversal.

    It contains NO real edge. Its only purpose is to prove the pipeline runs
    and to show what an honest 'no edge' report looks like.
    """
    rng = np.random.default_rng(seed)
    log_vol = np.empty(n)
    log_vol[0] = np.log(0.0010)
    for i in range(1, n):
        log_vol[i] = 0.97 * log_vol[i - 1] + 0.03 * np.log(0.0010) + rng.normal(0, 0.06)
    sigma = np.exp(log_vol)

    r = rng.normal(0, sigma)
    r[1:] -= 0.04 * r[:-1]  # tiny microstructure-like reversal
    close = 30_000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.00005, n))

    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.00035, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.00035, n)))
    vol = rng.lognormal(6.0, 0.45, n) * (1 + 3 * sigma / sigma.mean())
    buy_frac = np.clip(rng.normal(0.5, 0.08, n), 0.02, 0.98)

    t0 = pd.Timestamp("2021-01-01", tz="UTC")
    times = pd.date_range(t0, periods=n, freq=f"{TF_MIN}min")

    df = pd.DataFrame({
        "candle_time": times,
        "open": open_, "high": hi, "low": lo, "close": close,
        "volume": vol,
        "quote_volume": vol * close,
        "trade_count": rng.poisson(20_000, n).astype(float),
        "taker_buy_base_volume": vol * buy_frac,
        "taker_buy_quote_volume": vol * close * buy_frac,
    })
    return _normalise(df)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize only; refuse to silently discard or repair research candles."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()
    if "ignore_value" in out.columns:
        out = out.drop(columns=["ignore_value"])
    out["candle_time"] = pd.to_datetime(out["candle_time"], utc=True, errors="coerce")
    for col in REQUIRED[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out[REQUIRED].isna().any().any():
        bad = out[REQUIRED].isna().sum()
        raise ValueError(f"Missing/non-numeric required market values: {bad[bad > 0].to_dict()}")
    out = out.sort_values("candle_time").reset_index(drop=True)
    if out["candle_time"].duplicated().any():
        raise ValueError("Duplicate candle_time rows detected; refusing automatic deduplication.")

    dt = out["candle_time"].diff().dropna()
    gaps = int((dt != pd.Timedelta(minutes=TF_MIN)).sum())
    if gaps:
        raise ValueError(f"Found {gaps} non-{TF_MIN}-minute intervals; research labels would cross gaps.")

    o, h, l, close = (out[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    invalid_ohlc = (o <= 0) | (h <= 0) | (l <= 0) | (close <= 0) | (h < np.maximum(o, close)) | (l > np.minimum(o, close)) | (h < l)
    if invalid_ohlc.any():
        raise ValueError(f"Found {int(invalid_ohlc.sum())} invalid OHLC rows.")
    if (out[["volume", "quote_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume"]] < 0).any().any():
        raise ValueError("Negative volume or trade-count values detected.")
    if (out["taker_buy_base_volume"] > out["volume"] + 1e-8).any():
        raise ValueError("taker_buy_base_volume exceeds total volume.")
    if (out["taker_buy_quote_volume"] > out["quote_volume"] + 1e-6).any():
        raise ValueError("taker_buy_quote_volume exceeds quote volume.")

    print(f"[data] rows={len(out):,}  gaps=0  duplicates=0  invalid_ohlc=0  "
          f"{out['candle_time'].min()} -> {out['candle_time'].max()}")
    return out


# ---------------------------------------------------------------------
# 1. FEATURES — everything here is known at candle t CLOSE.
#    No future data. No absolute price/volume levels.
# ---------------------------------------------------------------------

def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    f = pd.DataFrame(index=df.index)

    o, h, l, c = (df[k].astype(float) for k in ["open", "high", "low", "close"])
    v = df["volume"].astype(float).clip(lower=EPS)
    qv = df["quote_volume"].astype(float)
    tbq = df["taker_buy_quote_volume"].astype(float)

    rng = (h - l).replace(0, np.nan)
    ret_oc = np.log(c / o.replace(0, np.nan))
    ret_cc = np.log(c / c.shift(1))

    # --- candle shape, all relative (scale-free) -----------------------
    f["ret_oc"] = ret_oc
    f["range_rel"] = rng / c
    f["close_position"] = ((c - l) / rng).clip(-1, 2)
    f["upper_wick_rel"] = (h - np.maximum(o, c)) / c
    f["lower_wick_rel"] = (np.minimum(o, c) - l) / c
    f["body_vs_range"] = (c - o).abs() / rng

    # --- order-flow imbalance (one single version, no duplicates) -----
    buy_ratio = (df["taker_buy_base_volume"].astype(float) / v).clip(0, 1)
    f["flow_imbalance"] = 2 * buy_ratio - 1
    f["flow_imbalance_q"] = (2 * (tbq / qv.replace(0, np.nan)) - 1)
    f["flow_imbalance_z96"] = (
        (f["flow_imbalance"] - f["flow_imbalance"].rolling(96, min_periods=24).mean())
        / f["flow_imbalance"].rolling(96, min_periods=24).std().replace(0, np.nan)
    )

    # --- multi-horizon momentum / mean reversion ----------------------
    for k in (1, 2, 4, 8, 16, 32, 96):
        f[f"ret_{k}"] = np.log(c / c.shift(k))

    # --- volatility & regime ------------------------------------------
    for k in (16, 96, 384):
        s = ret_cc.rolling(k, min_periods=k // 2).std()
        f[f"vol_{k}"] = s
        f[f"vol_ratio_{k}"] = s / s.rolling(384, min_periods=96).mean().replace(0, np.nan)

    f["range_z96"] = (
        f["range_rel"] / f["range_rel"].rolling(96, min_periods=24).std().replace(0, np.nan)
    )
    f["dist_sma96"] = c / c.rolling(96, min_periods=24).mean() - 1
    f["dist_sma384"] = c / c.rolling(384, min_periods=96).mean() - 1
    f["ret_skew96"] = ret_cc.rolling(96, min_periods=48).skew()
    f["autocorr_lag1_96"] = (
        ret_cc.rolling(96, min_periods=48)
        .apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1]
               if np.std(x[:-1]) > 0 and np.std(x[1:]) > 0 else 0.0, raw=True)
    )

    # --- volume / activity anomalies, as z-scores ---------------------
    # clip lower: real exchanges sometimes emit zero-volume / zero-trade
    # candles, and log(0) would create -inf rows that get silently dropped.
    for col, name in [(v, "volume"), (df["trade_count"].astype(float), "trades"), (qv, "quote")]:
        safe = col.clip(lower=EPS)
        m = safe.rolling(96, min_periods=24).mean()
        s = safe.rolling(96, min_periods=24).std().replace(0, np.nan)
        f[f"{name}_z96"] = (safe - m) / s
        f[f"{name}_chg"] = np.log(safe / safe.shift(1))

    # --- session only. Drop month / day-of-week sin-cos: spurious. ----
    hour = df["candle_time"].dt.hour
    f["session_asia"] = ((hour >= 0) & (hour < 8)).astype("int8")
    f["session_eu"] = ((hour >= 8) & (hour < 13)).astype("int8")
    f["session_us"] = ((hour >= 13) & (hour < 21)).astype("int8")
    f["is_weekend"] = (df["candle_time"].dt.dayofweek >= 5).astype("int8")

    features = list(f.columns)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f, features


# ---------------------------------------------------------------------
# 2. LABELS — volatility-scaled triple barrier, Net-of-cost aware
# ---------------------------------------------------------------------

def triple_barrier_labels(
    df: pd.DataFrame,
    features: pd.DataFrame,
    horizon: int = 4,
    vol_window: int = 96,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """OHLC triple barrier using past volatility, next-open entry and adverse gap fills."""
    if horizon < 1 or vol_window < 2 or cost_bps < 0:
        raise ValueError("Invalid horizon, volatility window, or trading cost.")
    c = df["close"].astype(float).to_numpy()
    o = df["open"].astype(float).to_numpy()
    hi = df["high"].astype(float).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    ret_cc = np.log(pd.Series(c) / pd.Series(c).shift(1))
    sigma = (ret_cc.rolling(vol_window, min_periods=max(2, vol_window // 2)).std()
             * np.sqrt(horizon)).to_numpy()
    cost_ret = cost_bps / 10_000.0
    barrier_vol = np.maximum(sigma, cost_ret * 1.5)
    upper = pt_mult * barrier_vol
    lower = -sl_mult * barrier_vol
    entry = np.full(len(df), np.nan)
    entry[:-1] = o[1:]
    fwd_h = np.full(len(df), np.nan)
    exit_ret = np.full(len(df), np.nan)
    label3 = np.zeros(len(df), dtype=float)

    for i in range(len(df) - horizon):
        if not np.isfinite(entry[i]) or entry[i] <= 0 or not np.isfinite(upper[i]):
            continue
        tp = entry[i] * (1.0 + upper[i])
        sl = entry[i] * (1.0 + lower[i])
        resolved = False
        for j in range(1, horizon + 1):
            k = i + j
            hit_up = hi[k] >= tp
            hit_dn = lo[k] <= sl
            if not (hit_up or hit_dn):
                continue
            if hit_dn:  # same-candle double touch is a stop-loss
                fill = min(o[k], sl)
                label3[i] = -1
            else:
                fill = max(o[k], tp)
                label3[i] = 1
            exit_ret[i] = fill / entry[i] - 1.0
            resolved = True
            break
        if not resolved:
            exit_ret[i] = c[i + horizon] / entry[i] - 1.0
            label3[i] = 0
        fwd_h[i] = c[i + horizon] / entry[i] - 1.0

    y_long = ((label3 == 1) & (upper > cost_ret)).astype(float)
    valid = np.isfinite(fwd_h) & np.isfinite(exit_ret) & np.isfinite(sigma) & (entry > 0)
    sample_weight = (1.0 / horizon) * np.clip(
        np.abs(fwd_h) / (sigma + EPS), 0.25, 3.0
    )
    out = pd.DataFrame({
        "label3": label3, "y_long": y_long, "fwd_ret_h": fwd_h,
        "exit_ret": exit_ret, "sigma": sigma, "sample_weight": sample_weight,
        "valid": valid,
    }, index=df.index)
    out.loc[~valid, ["label3", "y_long", "fwd_ret_h", "exit_ret", "sample_weight"]] = np.nan
    return out


# ---------------------------------------------------------------------
# 3. MODEL
# ---------------------------------------------------------------------

def build_model(n_estimators: int = 220):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            objective="binary", n_estimators=n_estimators,
            learning_rate=0.03, num_leaves=8, max_depth=4,
            min_child_samples=300, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=0.5,
            random_state=42, n_jobs=-1, verbosity=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=8, learning_rate=0.04,
            l2_regularization=5.0, min_samples_leaf=300, random_state=42,
        )


def fit_with_early_stop(model, X, y, w, embargo: int = 0):
    """Use a chronological validation tail with a label-horizon purge before it."""
    import inspect

    cut = int(len(X) * 0.85)
    fit_end = cut - max(0, int(embargo))
    if fit_end < 2 or cut >= len(X):
        raise ValueError("Training fold too short for purged early stopping.")
    x_train, y_train, w_train = X.iloc[:fit_end], y.iloc[:fit_end], w.iloc[:fit_end]
    x_valid, y_valid = X.iloc[cut:], y.iloc[cut:]

    try:
        from lightgbm import early_stopping
    except ImportError:
        model.fit(x_train, y_train, sample_weight=w_train)
        return model

    model.set_params(n_estimators=400)
    kwargs = {"eval_metric": "binary_logloss",
              "callbacks": [early_stopping(30, verbose=False)]}
    params = inspect.signature(model.fit).parameters
    if "eval_X" in params:
        kwargs["eval_X"] = x_valid
        kwargs["eval_y"] = y_valid
    else:
        kwargs["eval_set"] = [(x_valid, y_valid)]
    model.fit(x_train, y_train, sample_weight=w_train, **kwargs)
    return model


# ---------------------------------------------------------------------
# 4. PURGED WALK-FORWARD
# ---------------------------------------------------------------------

def make_folds(n: int, min_train: int, test_len: int, embargo: int, max_folds: int):
    """Build every chronological fold, then keep the MOST RECENT max_folds.

    Taking the first N folds would only test old years; the newest regimes are
    exactly what we need to judge whether the model generalises.
    """
    folds, start = [], min_train
    while start + test_len + embargo <= n:
        folds.append((0, start - embargo, start, start + test_len))
        start += test_len
    return folds[-max_folds:] if len(folds) > max_folds else folds


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a win rate — honest uncertainty, not just a %."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def non_overlapping(mask_idx: np.ndarray, horizon: int) -> np.ndarray:
    """Keep a signal only if the previous position is already closed."""
    kept, last = [], -10**9
    for i in mask_idx:
        if i - last >= horizon:
            kept.append(i)
            last = i
    return np.asarray(kept, dtype=int)


def evaluate_fold(
    prob: np.ndarray, y: np.ndarray,
    exit_ret: np.ndarray, hold_ret: np.ndarray,
    cost_bps: float, horizon: int, threshold: float,
) -> dict:
    """Evaluate only predictions exceeding a frozen TRAIN-derived threshold."""
    if not np.isfinite(threshold):
        raise ValueError("A finite train-calibrated threshold is required.")
    cost = cost_bps / 10_000.0
    candidates = np.flatnonzero(np.asarray(prob, dtype=float) >= threshold)
    sel = non_overlapping(candidates, horizon)
    if len(sel) == 0:
        return {
            "trades": 0, "gross_bps_list": [], "mean_gross_bps": None,
            "mean_net_bps": None, "median_net_bps": None,
            "win_rate_after_cost_pct": None, "per_trade_sharpe": None,
            "compounded_net_pct": None, "benchmark_mean_net_bps": None,
            "edge_vs_benchmark_bps": None, "precision_label_pct": None,
        }

    gross = np.asarray(exit_ret, dtype=float)[sel]
    net = gross - cost
    hit = float((net > 0).mean())
    mean_net_bps = float(net.mean() * 10_000)
    sharpe = float(net.mean() / (net.std() + EPS))
    bench_idx = non_overlapping(np.arange(len(prob)), horizon)
    bench = np.asarray(hold_ret, dtype=float)[bench_idx] - cost

    return {
        "trades": int(len(sel)),
        "precision_label_pct": float(np.asarray(y)[sel].mean() * 100),
        "gross_bps_list": [float(x * 10_000) for x in gross],
        "mean_gross_bps": float(gross.mean() * 10_000),
        "mean_net_bps": mean_net_bps,
        "median_net_bps": float(np.median(net) * 10_000),
        "win_rate_after_cost_pct": hit * 100,
        "per_trade_sharpe": sharpe,
        "compounded_net_pct": float((np.prod(1 + net) - 1) * 100),
        "benchmark_mean_net_bps": float(bench.mean() * 10_000) if len(bench) else None,
        "edge_vs_benchmark_bps": (
            float((net.mean() - bench.mean()) * 10_000) if len(bench) else None
        ),
    }


# ---------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rows", type=int, default=80_000)
    ap.add_argument("--db-url", default=os.getenv(
        "OSH_DB_URL", "postgresql+psycopg2://postgres@127.0.0.1:5433/osh"),
        help="Password-less DSN is fine: libpq reads %APPDATA%/postgresql/pgpass.conf")
    ap.add_argument("--table", default="public.btcusdt_15m")
    ap.add_argument("--horizon", type=int, default=4, help="bars held (4 = 1h)")
    ap.add_argument("--cost-bps", type=float, default=10.0,
                    help="REALISTIC round-trip cost. Binance futures taker ~8-12 bps RT.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--top-pct", type=float, default=0.05)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.synthetic:
        df = load_synthetic(args.rows)
        data_source = "SYNTHETIC (self-test, no real market data)"
    else:
        try:
            df = load_from_db(args.db_url, args.table)
        except Exception as exc:
            raise SystemExit(
                f"[data] DB ga ulanib bo'lmadi ({type(exc).__name__}). "
                "OSH_DB_URL ni to'g'ri o'rnating yoki --synthetic bilan sinab ko'ring."
            ) from None
        data_source = f"PostgreSQL {args.table}"

    X, features = make_features(df)
    lab = triple_barrier_labels(df, X, horizon=args.horizon, cost_bps=args.cost_bps)

    ok = lab["valid"] & X.notna().all(axis=1)
    X = X.loc[ok].reset_index(drop=True)
    lab = lab.loc[ok].reset_index(drop=True)
    times = df["candle_time"].loc[ok].reset_index(drop=True)
    print(f"[features] {len(features)} stationary features | usable rows={len(X):,}")
    print(f"[labels] +1={(lab.label3 == 1).mean() * 100:.2f}%  "
          f"-1={(lab.label3 == -1).mean() * 100:.2f}%  "
          f"0(noise)={(lab.label3 == 0).mean() * 100:.2f}%")

    min_train = BARS_PER_DAY * args.train_days
    test_len = BARS_PER_DAY * args.test_days
    embargo = args.horizon + 2
    folds = make_folds(len(X), min_train, test_len, embargo, args.folds)
    if not folds:
        raise ValueError("Not enough rows for the requested train/test windows. "
                         "Lower --train-days/--test-days or use more data.")
    print(f"[cv] {len(folds)} purged walk-forward folds | train>={min_train:,} "
          f"test={test_len:,} embargo={embargo}")

    y = lab["y_long"].astype("int8")
    w = lab["sample_weight"].astype(float)
    r_exit = lab["exit_ret"].to_numpy()
    r_hold = lab["fwd_ret_h"].to_numpy()

    from sklearn.metrics import roc_auc_score, brier_score_loss

    rows = []
    for i, (tr0, tr1, te0, te1) in enumerate(folds, 1):
        # Reserve the train tail exclusively for threshold calibration.
        # It is not used for fitting or early stopping.
        cal0 = max(tr0 + 100, tr1 - test_len)
        if cal0 >= tr1:
            raise ValueError("Training fold is too short for a separate calibration tail.")
        model = build_model()
        model = fit_with_early_stop(
            model, X.iloc[tr0:cal0], y.iloc[tr0:cal0], w.iloc[tr0:cal0],
            embargo=args.horizon + 2,
        )
        p_cal = model.predict_proba(X.iloc[cal0:tr1])[:, 1]
        thr = float(np.quantile(p_cal, 1 - args.top_pct))

        p_test = model.predict_proba(X.iloc[te0:te1])[:, 1]
        y_test = y.iloc[te0:te1].to_numpy()

        res = evaluate_fold(p_test, y_test,
                            r_exit[te0:te1], r_hold[te0:te1],
                            args.cost_bps, args.horizon, threshold=thr)
        # constant baseline uses the TRAIN base rate, never the test labels
        train_base = float(y.iloc[tr0:tr1].mean())
        res.update({
            "fold": i,
            "test_start": str(times.iloc[te0]),
            "test_end": str(times.iloc[te1 - 1]),
            "roc_auc": float(roc_auc_score(y_test, p_test)),
            "brier": float(brier_score_loss(y_test, p_test)),
            "brier_constant": float(brier_score_loss(
                y_test, np.full(len(y_test), train_base))),
            "threshold_from_train": round(thr, 5),
            "test_pass_rate_pct": float((p_test >= thr).mean() * 100),
        })
        rows.append(res)
        net_display = res["mean_net_bps"]
        net_text = f"{net_display:+.2f}bps" if net_display is not None else "no trades"
        print(f"[fold {i}] AUC={res['roc_auc']:.4f}  Brier={res['brier']:.4f} "
              f"(const {res['brier_constant']:.4f})  trades={res['trades']}  "
              f"net={net_text}  pass={res['test_pass_rate_pct']:.1f}%")

    rep = pd.DataFrame(rows)
    net = rep["mean_net_bps"]
    pos = (net > 0).mean() * 100

    # ---- cost sensitivity: win rate is a FUNCTION of the assumed cost ----
    all_gross = np.array([g for row in rows for g in row["gross_bps_list"]])
    cost_rows = []
    for c in (0, 1, 2, 4, 8, 10, 20):
        n = all_gross - c
        wins = int((n > 0).sum())
        lo, hi = wilson_ci(wins, len(n))
        cost_rows.append({
            "cost_bps": c,
            "trades": len(n),
            "win_rate_pct": float(wins / len(n) * 100) if len(n) else 0.0,
            "ci_low_pct": float(lo), "ci_high_pct": float(hi),
            "mean_net_bps": float(n.mean()) if len(n) else 0.0,
            "compounded_net_pct": float((np.prod(1 + n / 10_000) - 1) * 100)
                                  if len(n) else 0.0,
        })
    n_cfg = all_gross - args.cost_bps
    wins_cfg = int((n_cfg > 0).sum())
    lo_cfg, hi_cfg = wilson_ci(wins_cfg, len(n_cfg))
    at_cost = {
        "win_rate_pct": float(wins_cfg / len(n_cfg) * 100) if len(n_cfg) else 0.0,
        "ci_low_pct": float(lo_cfg), "ci_high_pct": float(hi_cfg),
        "mean_net_bps": float(n_cfg.mean()) if len(n_cfg) else 0.0,
    }

    print("\n" + "=" * 68)
    print("WALK-FORWARD SUMMARY  (this is the number that matters)")
    print("=" * 68)
    print(f"folds                     : {len(rep)}")
    print(f"total paper trades        : {len(all_gross):,}")
    print(f"win rate @ {args.cost_bps:.0f} bps       : {at_cost['win_rate_pct']:.2f}%  "
          f"(95% CI {at_cost['ci_low_pct']:.2f}–{at_cost['ci_high_pct']:.2f}%)")
    print(f"gross win rate (0 bps)    : {cost_rows[0]['win_rate_pct']:.2f}%")
    print(f"median net / trade        : {net.median():+.3f} bps")
    print(f"mean net / trade          : {net.mean():+.3f} bps")
    print(f"worst fold                : {net.min():+.3f} bps")
    print(f"positive folds            : {pos:.1f}%   (require >= 75%)")
    print(f"median AUC                : {rep['roc_auc'].median():.4f}  (0.50 = coin)")
    print(f"median Brier              : {rep['brier'].median():.4f}  "
          f"(constant = {rep['brier_constant'].median():.4f})")
    print(f"median edge vs buy&hold   : {rep['edge_vs_benchmark_bps'].median():+.3f} bps")
    print(f"round-trip cost assumed   : {args.cost_bps:.1f} bps")
    print("\n  cost(bps)  win_rate%   mean_net_bps   compounded%")
    for r in cost_rows:
        print(f"     {r['cost_bps']:>5}     {r['win_rate_pct']:>6.2f}     "
              f"{r['mean_net_bps']:>+9.2f}     {r['compounded_net_pct']:>+9.2f}")

    verdict = ("GO to shadow/paper trading" if pos >= 75 and net.median() > 0
               and rep["roc_auc"].median() > 0.52
               else "NO-GO: no robust net edge. Do not place live orders.")
    print(f"\nVERDICT                   : {verdict}")
    print("=" * 68)
    print("Reminder: many configurations x one dataset = selection bias. "
          "Apply Deflated Sharpe / keep an untouched final 6-12 months.")
    print("ALL NUMBERS ARE HISTORICAL PAPER TRADES. No live order was placed.")

    if args.out:
        summary = {
            "data_source": data_source,
            "rows": int(len(df)),
            "first_candle": str(df["candle_time"].iloc[0]),
            "last_candle": str(df["candle_time"].iloc[-1]),
            "usable_rows": int(len(X)),
            "feature_count": len(features),
            "folds": len(rep),
            "total_paper_trades": int(len(all_gross)),
            "win_rate_pct": at_cost["win_rate_pct"],
            "win_rate_ci95": [at_cost["ci_low_pct"], at_cost["ci_high_pct"]],
            "gross_win_rate_pct": cost_rows[0]["win_rate_pct"],
            "mean_net_bps": float(net.mean()),
            "median_net_bps": float(net.median()),
            "worst_fold_bps": float(net.min()),
            "positive_folds_pct": float(pos),
            "median_auc": float(rep["roc_auc"].median()),
            "median_brier": float(rep["brier"].median()),
            "brier_constant": float(rep["brier_constant"].median()),
            "median_edge_vs_benchmark_bps": float(rep["edge_vs_benchmark_bps"].median()),
            "cost_bps": args.cost_bps,
            "mode": "PAPER / SHADOW — no live orders, no capital at risk",
            "verdict": verdict,
        }
        Path(args.out).write_text(
            json.dumps({"config": {k: v for k, v in vars(args).items() if k != "db_url"},
                        "summary": summary, "features": features,
                        "cost_sensitivity": cost_rows, "folds": rows},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[report] written to {args.out}")


if __name__ == "__main__":
    main()
