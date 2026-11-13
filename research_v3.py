"""OSH Research v3 — causal, reproducible, fail-closed paper research.

This module never places orders. It deliberately stops on malformed or gapped
market data rather than silently dropping rows and changing label horizons.
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

import pipeline_v2 as pv

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("OSH_DATA_DIR", str(HERE)))
EPS = 1e-12
TF_MINUTES = 15
BARS_PER_DAY = 96
COST_BPS = 10.0
TOP_PCT = 0.05
N_FOLDS = 5
HOLDOUT_DAYS = 180
MAX_EXT_AGE_MIN = 60
SOURCE_BAR_MINUTES = 2
HORIZONS = [(4, "1h"), (16, "4h"), (96, "1d")]
VOL_WIN = 96
REQUIRED = [
    "candle_time", "open", "high", "low", "close", "volume",
    "trade_count", "quote_volume", "taker_buy_base_volume",
    "taker_buy_quote_volume",
]
TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


# ------------------------------------------------------------------ data
def validate_market_data(df: pd.DataFrame, interval_minutes: int = TF_MINUTES) -> pd.DataFrame:
    """Return a normalized copy or raise; never silently repair research input."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Market data is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Market data is empty")

    out = df[REQUIRED].copy()
    out["candle_time"] = pd.to_datetime(out["candle_time"], utc=True, errors="coerce")
    for col in REQUIRED[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out.isna().any().any():
        counts = out.isna().sum()
        bad = {k: int(v) for k, v in counts[counts > 0].items()}
        raise ValueError(f"Missing/non-numeric market values; refusing to drop rows: {bad}")

    out = out.sort_values("candle_time").reset_index(drop=True)
    if out["candle_time"].duplicated().any():
        raise ValueError("Duplicate candle_time values detected")
    delta = out["candle_time"].diff().dropna()
    expected = pd.Timedelta(minutes=interval_minutes)
    if not delta.eq(expected).all():
        bad_count = int((delta != expected).sum())
        raise ValueError(f"Non-contiguous candles: {bad_count} intervals are not {interval_minutes} minutes")

    prices = out[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise ValueError("OHLC prices must all be strictly positive")
    o, h, l, c = (out[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    invalid_ohlc = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)
    if invalid_ohlc.any():
        raise ValueError(f"OHLC structure violations: {int(invalid_ohlc.sum())}")

    nonnegative = ["volume", "trade_count", "quote_volume",
                   "taker_buy_base_volume", "taker_buy_quote_volume"]
    if (out[nonnegative] < 0).any().any():
        raise ValueError("Volume and trade-count fields cannot be negative")
    if (out["taker_buy_base_volume"] > out["volume"] + 1e-8).any():
        raise ValueError("taker_buy_base_volume exceeds total base volume")
    if (out["taker_buy_quote_volume"] > out["quote_volume"] + 1e-6).any():
        raise ValueError("taker_buy_quote_volume exceeds total quote volume")
    return out


def load_btc(db_url: str, table: str = "public.btcusdt_15m") -> pd.DataFrame:
    """Load one explicitly named PostgreSQL table; table identifier is allow-listed."""
    if not TABLE_RE.fullmatch(table):
        raise ValueError("Unsafe table name. Expected schema.table using letters, digits and underscores.")
    if not db_url:
        raise ValueError("DB URL is missing. Set OSH_DB_URL or pass --db-url.")
    from sqlalchemy import create_engine

    engine = create_engine(db_url, pool_pre_ping=True)
    sql = f"""
        SELECT to_timestamp(open_time / 1000.0) AS candle_time,
               open, high, low, close, volume, trade_count, quote_volume,
               taker_buy_base_volume, taker_buy_quote_volume
        FROM {table}
        ORDER BY open_time ASC
    """
    try:
        with engine.connect() as connection:
            raw = pd.read_sql_query(sql, connection)
    finally:
        engine.dispose()
    return validate_market_data(raw)


# -------------------------------------- high/low triple barrier labels
def barrier_labels(df: pd.DataFrame, h: int, cost_ret: float) -> pd.DataFrame:
    """First upper/lower barrier event using future OHLC and next-open entry.

    Same-bar double hits are pessimistically treated as stop-losses. Stop gaps
    fill at the worse opening price; take-profit gaps do not assume price
    improvement. Timeout trades exit at the horizon close.
    """
    if h < 1 or cost_ret < 0:
        raise ValueError("h must be >= 1 and cost_ret must be >= 0")
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    n = len(df)

    log_returns = np.log(pd.Series(c) / pd.Series(c).shift(1))
    sigma = (log_returns.rolling(VOL_WIN, min_periods=max(2, VOL_WIN // 2))
             .std().to_numpy() * np.sqrt(h))
    entry = np.full(n, np.nan)
    entry[:-1] = o[1:]
    barrier = np.maximum(sigma, cost_ret * 1.5)

    labels = np.zeros(n, dtype=np.int8)
    exit_ret = np.full(n, np.nan)
    exit_reason = np.full(n, "invalid", dtype=object)

    for i in range(n - h):
        if not np.isfinite(entry[i]) or entry[i] <= 0 or not np.isfinite(barrier[i]):
            continue
        upper_price = entry[i] * (1.0 + barrier[i])
        lower_price = entry[i] * (1.0 - barrier[i])
        resolved = False

        for j in range(1, h + 1):
            k = i + j
            hit_up = hi[k] >= upper_price
            hit_dn = lo[k] <= lower_price
            if not (hit_up or hit_dn):
                continue

            if hit_dn:  # includes a same-bar double touch: conservative loss
                fill = min(o[k], lower_price)  # adverse gap-through is not ignored
                labels[i] = -1
                exit_reason[i] = "stop_loss"
            else:
                fill = max(o[k], upper_price)
                labels[i] = 1
                exit_reason[i] = "take_profit"
            exit_ret[i] = fill / entry[i] - 1.0
            resolved = True
            break

        if not resolved:
            exit_ret[i] = c[i + h] / entry[i] - 1.0
            labels[i] = 0
            exit_reason[i] = "time"
            
    valid = np.isfinite(exit_ret) & np.isfinite(sigma) & np.isfinite(entry) & (entry > 0)
    weight = (1.0 / h) * np.clip(np.abs(exit_ret) / (sigma + EPS), 0.25, 3.0)
    weight[~valid] = np.nan
    return pd.DataFrame({
        "label3": labels,
        "exit_ret": exit_ret,
        "sigma": sigma,
        "sample_weight": weight,
        "exit_reason": exit_reason,
        "valid": valid,
    }, index=df.index)


# ----------------------------------- causal, availability-aware external
def align_external_features(
    btc: pd.DataFrame,
    source: pd.DataFrame,
    max_age_min: int = MAX_EXT_AGE_MIN,
    source_bar_minutes: int = SOURCE_BAR_MINUTES,
) -> pd.DataFrame:
    """Align external closes only after their source bars have actually closed.

    WIDE_GLD_AUX_2MIN timestamps are treated as source-bar OPEN timestamps.
    A source observation is available at datetime + source_bar_minutes. BTC
    features are queried at BTC candle close (open timestamp + 15 minutes).
    """
    out = pd.DataFrame(index=btc.index)
    if source is None or source.empty:
        return out
    if "datetime" not in source.columns:
        raise ValueError("External CSV must contain a datetime column")

    src = source.copy()
    src["datetime"] = pd.to_datetime(src["datetime"], utc=True, errors="coerce")
    src = src.dropna(subset=["datetime"]).sort_values("datetime")
    src = src.drop_duplicates(subset=["datetime"], keep="last")
    close_cols = [c for c in src.columns if c.endswith("_Close")]
    if not close_cols:
        return out

    left = pd.DataFrame({
        "signal_time": pd.to_datetime(btc["candle_time"], utc=True)
                       + pd.Timedelta(minutes=TF_MINUTES)
    }).sort_values("signal_time")

    for col in close_cols:
        asset = col[:-6]
        right = src[["datetime", col]].copy()
        right[col] = pd.to_numeric(right[col], errors="coerce")
        right = right.dropna(subset=[col])
        right = right[right[col] > 0]
        right["available_at"] = right["datetime"] + pd.Timedelta(minutes=source_bar_minutes)
        right = right[["available_at", col]].sort_values("available_at")
        if right.empty:
            continue

        merged = pd.merge_asof(
            left, right, left_on="signal_time", right_on="available_at",
            direction="backward", allow_exact_matches=True,
        )
        price = merged[col].astype(float).reset_index(drop=True)
        quote_time = merged["available_at"]
        age = (merged["signal_time"] - quote_time).dt.total_seconds() / 60.0
        stale = age.isna() | (age > max_age_min) | (age < 0)
        r1 = np.log(price / price.shift(1))
        r4 = np.log(price / price.shift(4))
        out[f"{asset}_ret1"] = r1.where(~stale).to_numpy()
        out[f"{asset}_ret4"] = r4.where(~stale).to_numpy()
        age_safe = age.fillna(max_age_min + 1).clip(lower=0)
        out[f"{asset}_age_min"] = np.log1p(age_safe).to_numpy()
        out[f"{asset}_stale"] = stale.astype("int8").to_numpy()

    return out.replace([np.inf, -np.inf], np.nan)


def external_features(
    df: pd.DataFrame,
    path: Path | str | None = None,
    max_age_min: int = MAX_EXT_AGE_MIN,
    source_bar_minutes: int = SOURCE_BAR_MINUTES,
) -> pd.DataFrame:
    path = Path(path) if path is not None else DATA_DIR / "WIDE_GLD_AUX_2MIN.csv"
    if not path.exists():
        return pd.DataFrame(index=df.index)
    source = pd.read_csv(path)
    return align_external_features(df, source, max_age_min, source_bar_minutes)


# ----------------------------------------------------------- model
def fit_model(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, h: int):
    """Train LightGBM with a chronological validation tail and a purged gap."""
    from lightgbm import LGBMClassifier, early_stopping

    y = np.asarray(y, dtype=np.int8)
    if not np.array_equal(np.unique(y), np.array([0, 1, 2])):
        raise ValueError("Training fold must contain all three barrier classes [0, 1, 2].")
    if len(X) < 100:
        raise ValueError("Not enough rows for purged early-stopping training.")

    cut = int(len(X) * 0.85)
    fit_end = cut - (h + 2)
    if fit_end < 50 or cut >= len(X):
        raise ValueError("Train fold is too short for validation plus horizon embargo.")

    model = LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=600,
        learning_rate=0.03, num_leaves=8, max_depth=4,
        min_child_samples=300, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=0.5,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        X.iloc[:fit_end], y[:fit_end],
        sample_weight=np.asarray(w)[:fit_end],
        eval_set=[(X.iloc[cut:], y[cut:])],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(40, verbose=False)],
    )
    return model


def predict_proba_3class(model, X: pd.DataFrame) -> np.ndarray:
    """Return columns in fixed order: down=0, neutral=1, up=2."""
    raw = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(model, "classes_", np.arange(raw.shape[1])))
    if raw.ndim != 2 or raw.shape[0] != len(X):
        raise ValueError("Model returned an invalid probability matrix.")
    aligned = np.zeros((len(X), 3), dtype=float)
    for j, cls in enumerate(classes):
        cls = int(cls)
        if cls not in (0, 1, 2):
            raise ValueError(f"Unexpected model class: {cls}")
        aligned[:, cls] = raw[:, j]
    if not np.isfinite(aligned).all():
        raise ValueError("Model probabilities contain non-finite values.")
    row_sum = aligned.sum(axis=1)
    if len(row_sum) and not np.allclose(row_sum, 1.0, atol=1e-5):
        raise ValueError("Model probabilities do not sum to one; class coverage is incomplete.")
    return aligned


# ------------------------------------------------ causal trade selection
def select_trades(p_up, p_dn, exit_ret, h, cost, thr_up, thr_dn):
    """Use frozen TRAIN thresholds; scan forward and prohibit overlap."""
    p_up = np.asarray(p_up, dtype=float)
    p_dn = np.asarray(p_dn, dtype=float)
    exit_ret = np.asarray(exit_ret, dtype=float)
    if not (len(p_up) == len(p_dn) == len(exit_ret)):
        raise ValueError("Probability and return arrays must have equal length")
    if h < 1 or cost < 0:
        raise ValueError("h must be >= 1 and cost must be >= 0")

    idx, sides, net = [], [], []
    last_entry = -10**9
    for i in range(max(0, len(p_up) - h)):
        if i - last_entry < h:
            continue
        up_ok = np.isfinite(p_up[i]) and p_up[i] >= thr_up
        dn_ok = np.isfinite(p_dn[i]) and p_dn[i] >= thr_dn
        if not up_ok and not dn_ok:
            continue
        side = 1 if up_ok and (not dn_ok or p_up[i] >= p_dn[i]) else -1
        underlying_return = exit_ret[i]
        if not np.isfinite(underlying_return):
            continue
        idx.append(i)
        sides.append(side)
        directional_gross = underlying_return if side == 1 else -underlying_return
        net.append(directional_gross - cost)
        last_entry = i

    return np.asarray(idx, dtype=int), np.asarray(sides, dtype=np.int8), np.asarray(net, dtype=float)


# ----------------------------------------------------------- metrics
def metrics(
    net: np.ndarray,
    gross: np.ndarray,
    cost_bps: float,
    h: int,
    benchmark: np.ndarray,
    n_bars: int,
) -> dict:
    net = np.asarray(net, dtype=float)
    gross = np.asarray(gross, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)
    if len(net) == 0:
        return {"trades": 0, "expectancy_bps": None, "win_rate_pct": None,
                "profit_factor": None, "benchmark_mean_net_bps": None,
                "edge_vs_benchmark_bps": None, "no_trade_pct": 100.0}
    if len(net) != len(gross):
        raise ValueError("net and gross arrays must have equal lengths")

    win = net > 0
    positive = float(net[win].sum())
    negative = float(-net[~win].sum())
    benchmark_mean = float(benchmark.mean() * 10_000) if len(benchmark) else None
    expectancy = float(net.mean() * 10_000)
    return {
        "trades": int(len(net)),
        "expectancy_bps": expectancy,
        "win_rate_pct": float(win.mean() * 100),
        "avg_win_gross_bps": float(gross[win].mean() * 10_000) if win.any() else 0.0,
        "avg_loss_gross_bps": float(-gross[~win].mean() * 10_000) if (~win).any() else 0.0,
        "profit_factor": float(positive / negative) if negative > 0 else None,
        "benchmark_mean_net_bps": benchmark_mean,
        "edge_vs_benchmark_bps": (
            float(expectancy - benchmark_mean) if benchmark_mean is not None else None
        ),
        "no_trade_pct": round(float(1 - len(net) / max(1, n_bars)) * 100, 1),
    }


def block_bootstrap_ci(
    net: np.ndarray, block: int = 20, reps: int = 2000, seed: int = 42
) -> tuple[float | None, float | None]:
    """Percentile CI from chronological, contiguous blocks of executed trades."""
    net = np.asarray(net, dtype=float)
    if block < 1 or reps < 1:
        raise ValueError("block and reps must be positive")
    if len(net) < block * 2:
        return None, None
    rng = np.random.default_rng(seed)
    n = len(net)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(reps)
    for b in range(reps):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([net[s:s + block] for s in starts])[:n]
        means[b] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo * 10_000), float(hi * 10_000)


# -------------------------------------------------------------- research
def run_research(
    df: pd.DataFrame,
    external: pd.DataFrame | None = None,
    cost_bps: float = COST_BPS,
    top_pct: float = TOP_PCT,
    folds_count: int = N_FOLDS,
    train_days: int = 365,
    test_days: int = 90,
    holdout_days: int = HOLDOUT_DAYS,
    max_ext_age_min: int = MAX_EXT_AGE_MIN,
    horizons: list[tuple[int, str]] | None = None,
    source: str = "provided DataFrame",
    source_bar_minutes: int = SOURCE_BAR_MINUTES,
) -> dict:
    if cost_bps < 0 or not 0 < top_pct < 1:
        raise ValueError("cost_bps must be non-negative and top_pct must be in (0, 1)")
    if min(folds_count, train_days, test_days, holdout_days) < 1:
        raise ValueError("folds and all day windows must be positive")
    df = validate_market_data(df)
    horizons = horizons or HORIZONS

    X_base, base_features = pv.make_features(df)
    X_ext = align_external_features(
        df, external, max_age_min=max_ext_age_min,
        source_bar_minutes=source_bar_minutes,
    ) if external is not None else pd.DataFrame(index=df.index)

    configs = {"BASE": X_base}
    if X_ext.shape[1] > 0:
        configs["BASE_XASSET_STALE_AWARE"] = pd.concat([X_base, X_ext], axis=1)

    holdout_bars = BARS_PER_DAY * holdout_days
    if len(df) <= holdout_bars:
        raise ValueError("Dataset is shorter than the requested untouched holdout.")
    dev_end = len(df) - holdout_bars
    dev_cut_time = df["candle_time"].iloc[dev_end]

    results = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "rows": int(len(df)),
            "first_candle_utc": str(df["candle_time"].iloc[0]),
            "last_candle_utc": str(df["candle_time"].iloc[-1]),
            "interval_minutes": TF_MINUTES,
            "validation": "strict contiguous UTC 15-minute candles; no silent row drops",
        },
        "config": {
            "cost_bps": float(cost_bps), "top_pct": float(top_pct),
            "folds": int(folds_count), "train_days": int(train_days),
            "test_days": int(test_days), "holdout_days": int(holdout_days),
            "max_external_age_min": int(max_ext_age_min),
            "source_bar_minutes": int(source_bar_minutes),
            "barrier_rule": "same-bar double touch => stop; stop gaps fill at worse open",
            "threshold_rule": "quantile calibrated on train tail and frozen for test",
        },
        "horizons": {},
        "gates": [],
        "live_orders": False,
    }

    for h, hname in horizons:
        lab = barrier_labels(df, h, cost_bps / 10_000.0)
        y3 = lab["label3"].to_numpy()
        y_cls = np.where(y3 == -1, 0, np.where(y3 == 0, 1, 2)).astype("int8")
        weights = lab["sample_weight"].to_numpy(float)
        raw_returns = lab["exit_ret"].to_numpy(float)
        valid = lab["valid"].to_numpy(bool)
        results["horizons"][hname] = {"bars": int(h), "configs": {}}

        for cfg_name, X in configs.items():
            Xf = X.replace([np.inf, -np.inf], np.nan)
            ok = valid & X_base.notna().all(axis=1).to_numpy()
            Xi = Xf.loc[ok].reset_index(drop=True).fillna(0.0)
            yi = y_cls[ok]
            wi = weights[ok]
            ri = raw_returns[ok]
            ti = df["candle_time"].loc[ok].reset_index(drop=True)
            cut_idx = int((ti < dev_cut_time).sum())

            min_train = BARS_PER_DAY * train_days
            test_len = BARS_PER_DAY * test_days
            fold_list = pv.make_folds(cut_idx, min_train, test_len, h + 2, folds_count)
            if not fold_list:
                raise ValueError(
                    f"Not enough development rows for {hname}: need at least "
                    f"{train_days} train days + {test_days} test days and embargo."
                )

            all_net, fold_rows = [], []
            last_model, last_thresholds = None, None

            for fold_no, (tr0, tr1, te0, te1) in enumerate(fold_list, 1):
                # The final calibration tail is excluded from both fitting and
                # early stopping, preventing threshold-selection feedback.
                cal0 = max(tr0 + 100, tr1 - BARS_PER_DAY * min(90, train_days))
                if cal0 >= tr1:
                    raise ValueError("Train fold is too short for an untouched calibration tail.")
                model = fit_model(Xi.iloc[tr0:cal0], yi[tr0:cal0], wi[tr0:cal0], h)
                p_cal = predict_proba_3class(model, Xi.iloc[cal0:tr1])
                thr_up = float(np.quantile(p_cal[:, 2], 1.0 - top_pct))
                thr_dn = float(np.quantile(p_cal[:, 0], 1.0 - top_pct))

                p_test = predict_proba_3class(model, Xi.iloc[te0:te1])
                selected, sides, net = select_trades(
                    p_test[:, 2], p_test[:, 0], ri[te0:te1], h,
                    cost_bps / 10_000.0, thr_up, thr_dn,
                )
                gross = net + cost_bps / 10_000.0
                bench_idx = np.arange(0, max(0, te1 - te0 - h), h)
                benchmark = ri[te0:te1][bench_idx] - cost_bps / 10_000.0
                row = metrics(net, gross, cost_bps, h, benchmark, te1 - te0)
                row.update({
                    "fold": int(fold_no),
                    "test_start_utc": str(ti.iloc[te0]),
                    "test_end_utc": str(ti.iloc[te1 - 1]),
                    "frozen_threshold_up": thr_up,
                    "frozen_threshold_down": thr_dn,
                    "test_pass_rate_pct": float(
                        ((p_test[:, 2] >= thr_up) | (p_test[:, 0] >= thr_dn)).mean() * 100
                    ),
                })
                fold_rows.append(row)
                if len(net):
                    all_net.append(net)
                last_model, last_thresholds = model, (thr_up, thr_dn)

            pooled = np.concatenate(all_net) if all_net else np.array([], dtype=float)
            pooled_gross = pooled + cost_bps / 10_000.0
            ci_lo, ci_hi = block_bootstrap_ci(pooled)
            fold_expectancies = [
                row["expectancy_bps"] for row in fold_rows
                if row["expectancy_bps"] is not None
            ]
            positive_folds = (
                float(np.mean(np.asarray(fold_expectancies) > 0) * 100)
                if fold_expectancies else 0.0
            )
            positive_total = float(pooled[pooled > 0].sum())
            negative_total = float(-pooled[pooled < 0].sum())
            pf = positive_total / negative_total if negative_total > 0 else None

            holdout = {"trades": 0, "expectancy_bps": None, "profit_factor": None}
            if last_model is not None and cut_idx < len(Xi):
                p_holdout = predict_proba_3class(last_model, Xi.iloc[cut_idx:])
                hold_idx, hold_sides, hold_net = select_trades(
                    p_holdout[:, 2], p_holdout[:, 0], ri[cut_idx:], h,
                    cost_bps / 10_000.0, last_thresholds[0], last_thresholds[1],
                )
                hold_gross = hold_net + cost_bps / 10_000.0
                hold_bench_idx = np.arange(0, max(0, len(ri) - cut_idx - h), h)
                hold_benchmark = ri[cut_idx:][hold_bench_idx] - cost_bps / 10_000.0
                holdout = metrics(
                    hold_net, hold_gross, cost_bps, h,
                    hold_benchmark, len(ri) - cut_idx,
                )
                holdout["note"] = "frozen final development model and thresholds; no refit or tuning"

            summary = {
                "feature_count": int(Xi.shape[1]),
                "folds": fold_rows,
                "positive_folds_pct": round(positive_folds, 2),
                "median_fold_expectancy_bps": (
                    round(float(np.median(fold_expectancies)), 3)
                    if fold_expectancies else None
                ),
                "pooled_expectancy_bps": (
                    round(float(pooled.mean() * 10_000), 3) if len(pooled) else None
                ),
                "pooled_profit_factor": round(float(pf), 4) if pf is not None else None,
                "bootstrap_ci95_bps": [
                    round(ci_lo, 3) if ci_lo is not None else None,
                    round(ci_hi, 3) if ci_hi is not None else None,
                ],
                "total_trades": int(len(pooled)),
                "untouched_holdout": holdout,
            }
            results["horizons"][hname]["configs"][cfg_name] = summary

            checks = {
                "median_net_gt_0": bool(
                    summary["median_fold_expectancy_bps"] is not None
                    and summary["median_fold_expectancy_bps"] > 0
                ),
                "positive_folds_ge_75": bool(positive_folds >= 75),
                "profit_factor_gt_1": bool(pf is not None and pf > 1),
                "bootstrap_lower_gt_0": bool(ci_lo is not None and ci_lo > 0),
                "beats_benchmark": bool(
                    fold_rows and all(
                        r["edge_vs_benchmark_bps"] is not None
                        and r["edge_vs_benchmark_bps"] > 0 for r in fold_rows
                    )
                ),
                "untouched_holdout_positive": bool(
                    holdout.get("expectancy_bps") is not None
                    and holdout["expectancy_bps"] > 0
                ),
                "holdout_min_30_trades": bool(holdout.get("trades", 0) >= 30),
            }
            results["gates"].append({
                "horizon": hname,
                "config": cfg_name,
                "checks": checks,
                "verdict": "GO" if all(checks.values()) else "NO-GO",
            })

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run causal BTCUSDT 15m paper research. This script never places orders."
    )
    parser.add_argument(
        "--db-url", default=os.getenv("OSH_DB_URL", "postgresql+psycopg2://postgres@127.0.0.1:5433/osh"),
        help="PostgreSQL SQLAlchemy URL; prefer OSH_DB_URL environment variable.",
    )
    parser.add_argument("--table", default="public.btcusdt_15m")
    parser.add_argument("--cost-bps", type=float, default=COST_BPS)
    parser.add_argument("--top-pct", type=float, default=TOP_PCT)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS)
    parser.add_argument("--max-ext-age-min", type=int, default=MAX_EXT_AGE_MIN)
    parser.add_argument("--source-bar-minutes", type=int, default=SOURCE_BAR_MINUTES)
    parser.add_argument("--external-csv", default=str(DATA_DIR / "WIDE_GLD_AUX_2MIN.csv"))
    parser.add_argument("--no-external", action="store_true")
    parser.add_argument("--out", default=str(DATA_DIR / "v3_results.json"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cost_bps < 0 or not 0 < args.top_pct < 1:
        raise SystemExit("--cost-bps must be >= 0 and --top-pct must be between 0 and 1.")
    if min(args.folds, args.train_days, args.test_days, args.holdout_days) < 1:
        raise SystemExit("Fold and date-window values must all be positive.")
    if args.max_ext_age_min < 0 or args.source_bar_minutes < 0:
        raise SystemExit("External age and source-bar minutes must be non-negative.")

    df = load_btc(args.db_url, args.table)
    external = None
    if not args.no_external and Path(args.external_csv).exists():
        external = pd.read_csv(args.external_csv)
    elif not args.no_external:
        print(f"[external] file not found; running BASE only: {args.external_csv}")

    results = run_research(
        df, external=external, cost_bps=args.cost_bps, top_pct=args.top_pct,
        folds_count=args.folds, train_days=args.train_days,
        test_days=args.test_days, holdout_days=args.holdout_days,
        max_ext_age_min=args.max_ext_age_min, source=f"PostgreSQL {args.table}",
        source_bar_minutes=args.source_bar_minutes,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False),
                      encoding="utf-8")

    print("\n=== OSH RESEARCH v3 — PAPER ONLY ===")
    for horizon, hd in results["horizons"].items():
        for config, summary in hd["configs"].items():
            holdout = summary["untouched_holdout"]
            print(
                f"{horizon:>3} {config:<24} "
                f"median={summary['median_fold_expectancy_bps']} bps | "
                f"positive_folds={summary['positive_folds_pct']}% | "
                f"PF={summary['pooled_profit_factor']} | "
                f"holdout={holdout.get('expectancy_bps')} bps"
            )
    for gate in results["gates"]:
        passed = sum(bool(v) for v in gate["checks"].values())
        print(f"{gate['horizon']:>3} {gate['config']:<24} {gate['verdict']} ({passed}/7 gates)")
    print(f"[report] {output.resolve()}")
    print("[safety] live_orders=false; no API trading keys or order endpoints are used.")


if __name__ == "__main__":
    main()
