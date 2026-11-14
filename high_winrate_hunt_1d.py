"""Correctness-hardened 1d asymmetric-barrier experiment. PAPER ONLY.

Seven configurations are fixed before evaluation. No tuning is performed on
the 180-day final holdout. The script never sends orders or uses API keys.

Important: a positive historical backtest is not proof of future profitability.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline_v2 as pv
import research_v3 as rv

BARS_PER_DAY = 96
H = 96
TOP_PCT = 0.10
FOLDS = 5
TRAIN_DAYS = 365
TEST_DAYS = 90
HOLDOUT_DAYS = 180
VOL_WIN = 96
DATA_END_UTC = pd.Timestamp("2026-10-31 23:45:00", tz="UTC")

CONFIGS = [
    ("C1_sym_1.00_1.00",        1.00, 1.00, False),
    ("C2_asym_0.75_1.00",       0.75, 1.00, False),
    ("C3_asym_0.50_1.00",       0.50, 1.00, False),
    ("C4_asym_0.75_1.50",       0.75, 1.50, False),
    ("C5_asym_0.50_1.50",       0.50, 1.50, False),
    ("C6_naive_0.25_3.00_TRAP", 0.25, 3.00, False),
    ("C7_asym_0.75_1.00_FUND",  0.75, 1.00, True),
]

FEE_BPS_PRIMARY = 6.0
FEE_BPS_STRESS = 10.0
INITIAL_CAPITAL = 100.0
CASH_RESERVE_PCT = 10.0
POSITION_CAPS = (25.0, 50.0, 100.0)
FUNDING_MAX_AGE_HOURS = 24.0
TF_MINUTES = 15


def _validate_funding(funding: pd.DataFrame) -> pd.DataFrame:
    required = {"funding_time", "funding_rate"}
    missing = sorted(required - set(funding.columns))
    if missing:
        raise ValueError(f"Funding CSV missing columns: {missing}")
    out = funding[["funding_time", "funding_rate"]].copy()
    out["funding_time"] = pd.to_datetime(out["funding_time"], utc=True, errors="coerce")
    out["funding_rate"] = pd.to_numeric(out["funding_rate"], errors="coerce")
    if out.isna().any().any():
        raise ValueError("Funding CSV contains missing or non-numeric settlement data.")
    out = out.sort_values("funding_time").reset_index(drop=True)
    if out["funding_time"].duplicated().any():
        raise ValueError("Funding CSV contains duplicate settlement timestamps.")
    if (out["funding_rate"].abs() > 0.50).any():
        raise ValueError("Funding rate outside the 50% sanity bound; inspect the source units.")
    return out


def load_funding(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Funding history not found: {path}. Generate it with capital_fee_stress_1d.py first."
        )
    return _validate_funding(pd.read_csv(path))


def funding_mark_price_proxy(
    df: pd.DataFrame, funding: pd.DataFrame
) -> pd.DataFrame:
    """Attach the last completed BTC 15m close at each funding settlement.

    Binance Vision funding archives do not contain the actual settlement
    markPrice. This causal close-price proxy is explicitly approximate.
    """
    candles = pd.DataFrame({
        "completed_at": pd.to_datetime(df["candle_time"], utc=True)
                        + pd.Timedelta(minutes=TF_MINUTES),
        "close_proxy": pd.to_numeric(df["close"], errors="coerce"),
    }).sort_values("completed_at")
    events = funding.sort_values("funding_time").copy()
    if len(candles):
        first_completed = candles["completed_at"].iloc[0]
        last_completed = candles["completed_at"].iloc[-1]
        events = events.loc[
            (events["funding_time"] >= first_completed)
            & (events["funding_time"] <= last_completed)
        ].copy()
    if events.empty:
        raise ValueError("No funding settlements overlap the available completed BTC candles.")
    mapped = pd.merge_asof(
        events,
        candles,
        left_on="funding_time",
        right_on="completed_at",
        direction="backward",
        allow_exact_matches=True,
    )
    age = (mapped["funding_time"] - mapped["completed_at"]).dt.total_seconds() / 60.0
    if mapped["close_proxy"].isna().any() or (age < 0).any() or (age > TF_MINUTES).any():
        raise ValueError("Could not map every funding settlement to a recent completed 15m close.")
    if (mapped["close_proxy"] <= 0).any():
        raise ValueError("Invalid mark-price proxy in funding history.")
    mapped["mark_price_proxy"] = mapped["close_proxy"].astype(float)
    return mapped[["funding_time", "funding_rate", "mark_price_proxy"]]


def asymmetric_barrier_labels(
    df: pd.DataFrame,
    h: int,
    tp_mult: float,
    sl_mult: float,
    vol_win: int = VOL_WIN,
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    """Compute independent long/short first-touch labels in bounded-memory chunks.

    Signal is known at candle i close; entry is candle i+1 open. Same-bar
    double-touch is a loss for that side. Stops use adverse gap fills. TP fills
    are fixed at the barrier and never assume price improvement on a gap.
    """
    if h < 1 or tp_mult <= 0 or sl_mult <= 0 or vol_win < 2:
        raise ValueError("h, tp_mult and sl_mult must be positive; vol_win >= 2.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    if n <= h + vol_win:
        raise ValueError("Not enough candles for the requested volatility window and horizon.")

    log_returns = np.log(pd.Series(close)).diff()
    sigma = (
        log_returns.rolling(vol_win, min_periods=max(2, vol_win // 2))
        .std().to_numpy(dtype=float) * np.sqrt(h)
    )
    entry = np.full(n, np.nan, dtype=float)
    entry[:-1] = open_[1:]

    long_label = np.full(n, np.nan, dtype=float)
    short_label = np.full(n, np.nan, dtype=float)
    long_exit = np.full(n, np.nan, dtype=float)
    short_exit = np.full(n, np.nan, dtype=float)
    long_hold = np.full(n, np.nan, dtype=float)
    short_hold = np.full(n, np.nan, dtype=float)
    long_reason = np.full(n, "invalid", dtype=object)
    short_reason = np.full(n, "invalid", dtype=object)
    long_horizon = np.full(n, np.nan, dtype=float)
    short_horizon = np.full(n, np.nan, dtype=float)

    n_events = n - h
    for start in range(0, n_events, chunk_size):
        end = min(n_events, start + chunk_size)
        rows = np.arange(start, end)
        e = entry[start:end]
        s = sigma[start:end]
        valid = np.isfinite(e) & (e > 0) & np.isfinite(s) & (s > 0)
        if not valid.any():
            continue

        long_tp = e * (1.0 + tp_mult * s)
        long_sl = e * (1.0 - sl_mult * s)
        short_tp = e * (1.0 - tp_mult * s)
        short_sl = e * (1.0 + sl_mult * s)

        # windows[r] contains candles r+1 through r+h, inclusive.
        hi_w = np.lib.stride_tricks.sliding_window_view(high[start + 1:end + h], h)
        lo_w = np.lib.stride_tricks.sliding_window_view(low[start + 1:end + h], h)
        local = np.arange(end - start)

        def resolve_side(tp: np.ndarray, sl: np.ndarray, is_short: bool):
            tp_hits = (lo_w <= tp[:, None]) if is_short else (hi_w >= tp[:, None])
            sl_hits = (hi_w >= sl[:, None]) if is_short else (lo_w <= sl[:, None])
            any_hits = tp_hits | sl_hits
            has_hit = any_hits.any(axis=1) & valid
            first = any_hits.argmax(axis=1)
            k = rows + first + 1
            first_tp = tp_hits[local, first]
            first_sl = sl_hits[local, first]
            stop = has_hit & first_sl
            take = has_hit & ~first_sl & first_tp
            timed = valid & ~has_hit

            labels = np.full(end - start, np.nan, dtype=float)
            exits = np.full(end - start, np.nan, dtype=float)
            holds = np.full(end - start, np.nan, dtype=float)
            reasons = np.full(end - start, "invalid", dtype=object)

            labels[stop] = -1.0
            labels[take] = 1.0
            labels[timed] = 0.0
            holds[has_hit & valid] = first[has_hit & valid] + 1
            holds[timed] = h
            reasons[stop] = "stop_loss"
            reasons[take] = "take_profit"
            reasons[timed] = "time"

            if is_short:
                stop_fill = np.maximum(open_[k], sl)
                take_fill = tp
                exits[stop] = e[stop] / stop_fill[stop] - 1.0
                exits[take] = e[take] / take_fill[take] - 1.0
                horizon_price = close[rows + h]
                exits[timed] = e[timed] / horizon_price[timed] - 1.0
            else:
                stop_fill = np.minimum(open_[k], sl)
                take_fill = tp
                exits[stop] = stop_fill[stop] / e[stop] - 1.0
                exits[take] = take_fill[take] / e[take] - 1.0
                horizon_price = close[rows + h]
                exits[timed] = horizon_price[timed] / e[timed] - 1.0
            return labels, exits, holds, reasons

        ll, le, lh, lr = resolve_side(long_tp, long_sl, False)
        slabel, se, sh, sr = resolve_side(short_tp, short_sl, True)

        long_label[start:end] = ll
        short_label[start:end] = slabel
        long_exit[start:end] = le
        short_exit[start:end] = se
        long_hold[start:end] = lh
        short_hold[start:end] = sh
        long_reason[start:end] = lr
        short_reason[start:end] = sr

        usable = valid
        long_horizon[start:end][usable] = close[rows[usable] + h] / e[usable] - 1.0
        short_horizon[start:end][usable] = e[usable] / close[rows[usable] + h] - 1.0

    valid_out = (
        np.isfinite(long_exit) & np.isfinite(short_exit)
        & np.isfinite(long_hold) & np.isfinite(short_hold)
        & np.isfinite(entry) & (entry > 0)
    )
    # Equal observation weights: no outcome-magnitude weighting that could
    # silently alter the classification objective.
    weight_long = np.where(valid_out, 1.0, np.nan)
    weight_short = np.where(valid_out, 1.0, np.nan)

    return pd.DataFrame({
        "long_label": long_label,
        "short_label": short_label,
        "long_exit": long_exit,
        "short_exit": short_exit,
        "long_holding_bars": long_hold,
        "short_holding_bars": short_hold,
        "long_exit_reason": long_reason,
        "short_exit_reason": short_reason,
        "long_horizon_exit": long_horizon,
        "short_horizon_exit": short_horizon,
        "sigma": sigma,
        "entry_price": entry,
        "long_sample_weight": weight_long,
        "short_sample_weight": weight_short,
        "valid": valid_out,
    }, index=df.index)


def funding_features(df: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Causal funding features computed on settlement events, not repeated bars."""
    events = _validate_funding(funding).set_index("funding_time")
    rates = events["funding_rate"].astype(float)
    event_features = pd.DataFrame(index=events.index)
    event_features["fund_last"] = rates
    event_features["fund_mean_1d"] = rates.rolling("1D", min_periods=1).mean()
    event_features["fund_mean_3d"] = rates.rolling("3D", min_periods=1).mean()
    event_features["fund_events_1d"] = rates.rolling("1D", min_periods=1).count()
    mean_30d = rates.rolling("30D", min_periods=5).mean()
    std_30d = rates.rolling("30D", min_periods=5).std()
    event_features["fund_z_30d"] = (rates - mean_30d) / (std_30d + 1e-12)

    left = pd.DataFrame({
        "row_id": np.arange(len(df), dtype=int),
        "signal_time": pd.to_datetime(df["candle_time"], utc=True)
                       + pd.Timedelta(minutes=TF_MINUTES),
    }).sort_values("signal_time")
    right = event_features.reset_index().rename(columns={"funding_time": "settled_at"})
    right = right.sort_values("settled_at")
    merged = pd.merge_asof(
        left, right,
        left_on="signal_time", right_on="settled_at",
        direction="backward", allow_exact_matches=True,
    ).sort_values("row_id")

    age_hours = (
        merged["signal_time"] - merged["settled_at"]
    ).dt.total_seconds() / 3600.0
    available = age_hours.notna() & (age_hours >= 0) & (age_hours <= FUNDING_MAX_AGE_HOURS)

    result = pd.DataFrame(index=df.index)
    for col in event_features.columns:
        result[col] = merged[col].to_numpy(dtype=float)
    result["fund_hours_since"] = age_hours.to_numpy(dtype=float)
    result["fund_available"] = available.astype("int8").to_numpy()
    result.loc[~available, event_features.columns] = np.nan
    return result.replace([np.inf, -np.inf], np.nan)


def funding_pnl_for_trade(
    side: int,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_price: float,
    fund_times_ns: np.ndarray,
    fund_rates: np.ndarray,
    fund_marks: np.ndarray,
) -> float:
    """Funding return relative to entry notional; ambiguous boundary credits ignored.

    Event cashflow as a fraction of entry notional is
    -side * funding_rate * (settlement mark proxy / entry price).
    Funding at an exact entry/exit timestamp is ambiguous from 15m OHLC;
    include payments but ignore credits at those boundaries.
    """
    if side not in (-1, 1) or not np.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("Invalid funding trade side or entry price.")
    if exit_time < entry_time:
        raise ValueError("Trade exit cannot precede entry.")
    lo = np.searchsorted(fund_times_ns, entry_time.value, side="left")
    hi = np.searchsorted(fund_times_ns, exit_time.value, side="right")
    if hi <= lo:
        return 0.0

    event_times = fund_times_ns[lo:hi]
    rates = fund_rates[lo:hi]
    marks = fund_marks[lo:hi]
    cashflows = -float(side) * rates * (marks / float(entry_price))
    boundary = (event_times == entry_time.value) | (event_times == exit_time.value)
    cashflows = np.where(boundary & (cashflows > 0), 0.0, cashflows)
    return float(cashflows.sum())


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = wins / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, centre - half) * 100.0, min(1.0, centre + half) * 100.0


def block_bootstrap_ci(
    values: np.ndarray, block: int = 20, reps: int = 2000, seed: int = 42
) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    if block < 1 or reps < 1:
        raise ValueError("block and reps must be positive.")
    if len(values) < 2 * block:
        return None, None
    rng = np.random.default_rng(seed)
    count = int(np.ceil(len(values) / block))
    means = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, len(values) - block + 1, size=count)
        means[i] = np.concatenate([values[s:s + block] for s in starts])[:len(values)].mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low * 10_000), float(high * 10_000)


def _capital_path(
    net_returns: np.ndarray,
    cap: float,
    initial: float = INITIAL_CAPITAL,
    reserve_pct: float = CASH_RESERVE_PCT,
) -> dict:
    equity = float(initial)
    peak = equity
    max_dd = 0.0
    ruined = False
    for r in np.asarray(net_returns, dtype=float):
        if equity <= 0:
            ruined = True
            break
        notional = min(float(cap), equity * (1.0 - reserve_pct / 100.0))
        end_equity = equity + notional * float(r)
        if end_equity <= 0:
            equity = 0.0
            ruined = True
        else:
            equity = end_equity
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return {
        "position_cap_usd": float(cap),
        "initial_capital_usd": float(initial),
        "final_balance_usd": round(equity, 2),
        "pnl_usd": round(equity - initial, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "cash_reserve_pct": float(reserve_pct),
        "ruined": bool(ruined),
        "sizing": "min(position_cap, 90% of current equity); no leverage",
    }


def trade_metrics(
    net: np.ndarray,
    gross: np.ndarray,
    benchmark: np.ndarray,
    n_bars: int,
) -> dict:
    net = np.asarray(net, dtype=float)
    gross = np.asarray(gross, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)
    if len(net) != len(gross):
        raise ValueError("net and gross must have equal lengths.")
    if len(net) == 0:
        return {
            "trades": 0, "win_rate_pct": None, "win_rate_ci95_pct": None,
            "expectancy_bps": None, "profit_factor": None,
            "benchmark_mean_net_bps": None, "edge_vs_benchmark_bps": None,
            "capital_paths": [],
        }
    if not np.isfinite(net).all() or not np.isfinite(gross).all():
        raise ValueError("Trade returns contain non-finite values.")

    wins = net > 0
    n_wins = int(wins.sum())
    ci_low, ci_high = wilson_interval(n_wins, len(net))
    positive = float(net[wins].sum())
    negative = float(-net[~wins].sum())
    expectancy = float(net.mean() * 10_000.0)
    benchmark_mean = float(benchmark.mean() * 10_000.0) if len(benchmark) else None
    paths = [_capital_path(net, cap) for cap in POSITION_CAPS]
    cap100 = next(p for p in paths if p["position_cap_usd"] == 100.0)

    return {
        "trades": int(len(net)),
        "trades_per_day": round(len(net) / max(1e-9, n_bars / BARS_PER_DAY), 3),
        "win_rate_pct": round(float(wins.mean() * 100.0), 2),
        "win_rate_ci95_pct": [round(ci_low, 2), round(ci_high, 2)],
        "expectancy_bps": round(expectancy, 3),
        "avg_win_net_bps": round(float(net[wins].mean() * 10_000.0), 2) if wins.any() else 0.0,
        "avg_loss_net_bps": round(float(-net[~wins].mean() * 10_000.0), 2) if (~wins).any() else 0.0,
        "payoff_ratio": (
            round(float(net[wins].mean() / -net[~wins].mean()), 3)
            if wins.any() and (~wins).any() else None
        ),
        "profit_factor": round(positive / negative, 4) if negative > 0 else None,
        "benchmark_mean_net_bps": round(benchmark_mean, 3) if benchmark_mean is not None else None,
        "edge_vs_benchmark_bps": (
            round(expectancy - benchmark_mean, 3) if benchmark_mean is not None else None
        ),
        "capital_paths": paths,
        "final_balance_100usd_cap": cap100["final_balance_usd"],
        "max_drawdown_100usd_cap_pct": cap100["max_drawdown_pct"],
    }


def fit_side_model(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    horizon: int,
):
    """Fit a 3-outcome classifier, supporting a genuinely absent class safely."""
    import inspect
    from lightgbm import LGBMClassifier, early_stopping

    y = np.asarray(y, dtype=np.int8)
    classes = np.unique(y)
    if len(classes) < 2 or not np.all(np.isin(classes, [0, 1, 2])):
        raise ValueError(f"Side training has insufficient outcome classes: {classes.tolist()}")
    if 2 not in classes:
        raise ValueError("Training fold contains no take-profit class; cannot learn this side.")
    if len(X) < 200:
        raise ValueError("Not enough observations for side-model training.")

    encoded = np.searchsorted(classes, y).astype(np.int8)
    cut = int(len(X) * 0.85)
    fit_end = cut - (horizon + 2)
    if fit_end < 100 or cut >= len(X):
        raise ValueError("Training fold too short for purged early stopping.")

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=8,
        max_depth=4,
        min_child_samples=300,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    x_train = X.iloc[:fit_end]
    y_train = encoded[:fit_end]
    w_train = np.asarray(sample_weight, dtype=float)[:fit_end]
    x_valid = X.iloc[cut:]
    y_valid = encoded[cut:]
    expected_train_classes = np.arange(len(classes), dtype=np.int8)
    if not np.array_equal(np.unique(y_train), expected_train_classes):
        raise ValueError(
            "Purged model-fit segment lost an outcome class; refusing a model "
            "whose early-stopping validation contains an unseen class."
        )

    fit_kwargs = {
        "eval_metric": "multi_logloss",
        "callbacks": [early_stopping(40, verbose=False)],
    }
    params = inspect.signature(model.fit).parameters
    if "eval_X" in params and "eval_y" in params:
        fit_kwargs["eval_X"] = x_valid
        fit_kwargs["eval_y"] = y_valid
    else:
        fit_kwargs["eval_set"] = [(x_valid, y_valid)]
    model.fit(x_train, y_train, sample_weight=w_train, **fit_kwargs)
    return model, classes


def predict_side_proba(model, classes: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=float)
    aligned = np.zeros((len(X), 3), dtype=float)
    for j, cls in enumerate(np.asarray(classes, dtype=int)):
        aligned[:, int(cls)] = raw[:, j]
    if not np.isfinite(aligned).all():
        raise ValueError("Model probabilities contain non-finite values.")
    if len(aligned) and (
        (aligned < -1e-10).any()
        or not np.allclose(aligned.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("Model probabilities do not sum to one.")
    return aligned


def _execute_signals(
    p_long: np.ndarray,
    p_short: np.ndarray,
    thr_long: float,
    thr_short: float,
    lab: pd.DataFrame,
    times: pd.Series,
    fund_times_ns: np.ndarray,
    fund_rates: np.ndarray,
    fund_marks: np.ndarray,
) -> dict:
    """Apply frozen thresholds, independent exits and funding per selected side."""
    n = len(times)
    if not (len(p_long) == len(p_short) == len(lab) == n):
        raise ValueError("Signals, timestamps and labels are not aligned.")

    indices, sides, gross, funding_returns, holding = [], [], [], [], []
    last_entry = -10**9
    for i in range(max(0, n - H)):
        if i - last_entry < H:
            continue
        go_long = np.isfinite(p_long[i]) and p_long[i] >= thr_long
        go_short = np.isfinite(p_short[i]) and p_short[i] >= thr_short
        if not go_long and not go_short:
            continue

        if go_long and go_short:
            excess_long = (p_long[i] - thr_long) / max(1e-9, 1.0 - thr_long)
            excess_short = (p_short[i] - thr_short) / max(1e-9, 1.0 - thr_short)
            side = 1 if excess_long >= excess_short else -1
        else:
            side = 1 if go_long else -1

        if side == 1:
            ret = float(lab["long_exit"].iloc[i])
            held = lab["long_holding_bars"].iloc[i]
        else:
            ret = float(lab["short_exit"].iloc[i])
            held = lab["short_holding_bars"].iloc[i]
        if not np.isfinite(ret) or not np.isfinite(held):
            continue
        held = int(held)
        if held < 1 or i + held >= n:
            continue

        entry_price = float(lab["entry_price"].iloc[i])
        entry_time = pd.Timestamp(times.iloc[i + 1])
        exit_time = pd.Timestamp(times.iloc[i + held]) + pd.Timedelta(minutes=TF_MINUTES)
        f_pnl = funding_pnl_for_trade(
            side, entry_time, exit_time, entry_price,
            fund_times_ns, fund_rates, fund_marks,
        )
        indices.append(i)
        sides.append(side)
        gross.append(ret)
        funding_returns.append(f_pnl)
        holding.append(held)
        last_entry = i

    gross_arr = np.asarray(gross, dtype=float)
    funding_arr = np.asarray(funding_returns, dtype=float)
    return {
        "indices": np.asarray(indices, dtype=int),
        "sides": np.asarray(sides, dtype=np.int8),
        "gross": gross_arr,
        "funding": funding_arr,
        "net_primary": gross_arr - FEE_BPS_PRIMARY / 10_000.0 + funding_arr,
        "net_stress": gross_arr - FEE_BPS_STRESS / 10_000.0 + funding_arr,
        "holding_bars": np.asarray(holding, dtype=int),
    }


def _long_benchmark(
    lab: pd.DataFrame,
    times: pd.Series,
    fund_times_ns: np.ndarray,
    fund_rates: np.ndarray,
    fund_marks: np.ndarray,
    fee_bps: float,
) -> np.ndarray:
    values = []
    for i in range(0, max(0, len(times) - H), H):
        ret = float(lab["long_horizon_exit"].iloc[i])
        entry_price = float(lab["entry_price"].iloc[i])
        entry_time = pd.Timestamp(times.iloc[i + 1])
        exit_time = pd.Timestamp(times.iloc[i + H]) + pd.Timedelta(minutes=TF_MINUTES)
        funding = funding_pnl_for_trade(
            1, entry_time, exit_time, entry_price,
            fund_times_ns, fund_rates, fund_marks,
        )
        values.append(ret - fee_bps / 10_000.0 + funding)
    return np.asarray(values, dtype=float)


def _metrics_with_ci(
    execution: dict,
    benchmark: np.ndarray,
    n_bars: int,
    key: str,
) -> dict:
    result = trade_metrics(execution[key], execution["gross"], benchmark, n_bars)
    ci = block_bootstrap_ci(execution[key])
    result["bootstrap_ci95_expectancy_bps"] = [
        round(ci[0], 3) if ci[0] is not None else None,
        round(ci[1], 3) if ci[1] is not None else None,
    ]
    return result


def _train_predict(
    X: pd.DataFrame,
    y_long: np.ndarray,
    y_short: np.ndarray,
    w_long: np.ndarray,
    w_short: np.ndarray,
    train_start: int,
    calibration_start: int,
    train_end: int,
) -> tuple:
    """Train only before calibration; purge all label horizons before calibration."""
    model_data_end = calibration_start - (H + 2)
    if model_data_end <= train_start or model_data_end > train_end:
        raise ValueError("Insufficient purged training observations before calibration.")
    X_train = X.iloc[train_start:model_data_end]
    ml, cls_l = fit_side_model(
        X_train, y_long[train_start:model_data_end],
        w_long[train_start:model_data_end], H,
    )
    ms, cls_s = fit_side_model(
        X_train, y_short[train_start:model_data_end],
        w_short[train_start:model_data_end], H,
    )
    X_cal = X.iloc[calibration_start:train_end]
    if len(X_cal) < 100:
        raise ValueError("Calibration tail is too short.")
    p_cal_l = predict_side_proba(ml, cls_l, X_cal)[:, 2]
    p_cal_s = predict_side_proba(ms, cls_s, X_cal)[:, 2]
    if not np.isfinite(p_cal_l).all() or not np.isfinite(p_cal_s).all():
        raise ValueError("Calibration probabilities are not finite.")
    if p_cal_l.max() <= 0 or p_cal_s.max() <= 0:
        raise ValueError("At least one side has no positive TP probability.")
    thr_l = float(np.quantile(p_cal_l, 1.0 - TOP_PCT))
    thr_s = float(np.quantile(p_cal_s, 1.0 - TOP_PCT))
    return ml, cls_l, ms, cls_s, thr_l, thr_s


def run_config(
    name: str,
    tp_mult: float,
    sl_mult: float,
    use_funding: bool,
    X_base: pd.DataFrame,
    X_funding: pd.DataFrame,
    labels: pd.DataFrame,
    times: pd.Series,
    dev_cut_time: pd.Timestamp,
    funding_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict:
    X = pd.concat([X_base, X_funding], axis=1) if use_funding else X_base
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix contains non-finite values after safe fill.")

    cut_idx = int((times < dev_cut_time).sum())
    fold_list = pv.make_folds(
        cut_idx,
        BARS_PER_DAY * TRAIN_DAYS,
        BARS_PER_DAY * TEST_DAYS,
        H + 2,
        FOLDS,
    )
    if len(fold_list) != FOLDS:
        raise ValueError(
            f"Expected {FOLDS} chronological folds but got {len(fold_list)}; "
            "refusing to report a partial experiment."
        )

    y_long = np.where(labels["long_label"].to_numpy() == -1, 0,
                      np.where(labels["long_label"].to_numpy() == 0, 1, 2)).astype(np.int8)
    y_short = np.where(labels["short_label"].to_numpy() == -1, 0,
                       np.where(labels["short_label"].to_numpy() == 0, 1, 2)).astype(np.int8)
    w_long = labels["long_sample_weight"].to_numpy(dtype=float)
    w_short = labels["short_sample_weight"].to_numpy(dtype=float)
    ft, fr, fm = funding_arrays

    fold_results = []
    pooled_primary, pooled_stress, pooled_gross = [], [], []
    pooled_benchmark_primary, pooled_benchmark_stress = [], []
    last_model_state = None
    total_oos_bars = 0

    for fold_no, (tr0, tr1, te0, te1) in enumerate(fold_list, 1):
        cal0 = max(tr0 + 100, tr1 - BARS_PER_DAY * min(90, TRAIN_DAYS))
        model_state = _train_predict(
            X, y_long, y_short, w_long, w_short, tr0, cal0, tr1
        )
        ml, cls_l, ms, cls_s, thr_l, thr_s = model_state

        X_test = X.iloc[te0:te1]
        p_l = predict_side_proba(ml, cls_l, X_test)[:, 2]
        p_s = predict_side_proba(ms, cls_s, X_test)[:, 2]
        test_labels = labels.iloc[te0:te1].reset_index(drop=True)
        test_times = times.iloc[te0:te1].reset_index(drop=True)

        execution = _execute_signals(
            p_l, p_s, thr_l, thr_s, test_labels, test_times, ft, fr, fm
        )
        benchmark = _long_benchmark(
            test_labels, test_times, ft, fr, fm, FEE_BPS_PRIMARY
        )
        benchmark_stress = _long_benchmark(
            test_labels, test_times, ft, fr, fm, FEE_BPS_STRESS
        )
        primary = _metrics_with_ci(execution, benchmark, te1 - te0, "net_primary")
        stress = _metrics_with_ci(execution, benchmark_stress, te1 - te0, "net_stress")
        primary.update({
            "fold": fold_no,
            "test_start": str(test_times.iloc[0]),
            "frozen_threshold_long": round(thr_l, 7),
            "frozen_threshold_short": round(thr_s, 7),
            "long_trades": int((execution["sides"] == 1).sum()),
            "short_trades": int((execution["sides"] == -1).sum()),
            "mean_funding_bps": round(float(execution["funding"].mean() * 10_000), 3)
                                if len(execution["funding"]) else None,
        })
        fold_results.append({"primary": primary, "stress10bps": stress})
        pooled_primary.extend(execution["net_primary"].tolist())
        pooled_stress.extend(execution["net_stress"].tolist())
        pooled_gross.extend(execution["gross"].tolist())
        pooled_benchmark_primary.extend(benchmark.tolist())
        pooled_benchmark_stress.extend(benchmark_stress.tolist())
        total_oos_bars += te1 - te0
        last_model_state = model_state

    if last_model_state is None:
        raise ValueError("No final model was trained; holdout cannot be evaluated.")

    pooled_execution = {
        "net_primary": np.asarray(pooled_primary, dtype=float),
        "net_stress": np.asarray(pooled_stress, dtype=float),
        "gross": np.asarray(pooled_gross, dtype=float),
    }
    # Fold metrics are also reported individually; pooled bootstrap preserves
    # contiguous blocks of executed trades, not independent IID observations.
    oos_primary = _metrics_with_ci(
        pooled_execution,
        np.asarray(pooled_benchmark_primary, dtype=float),
        total_oos_bars,
        "net_primary",
    )
    oos_stress = _metrics_with_ci(
        pooled_execution,
        np.asarray(pooled_benchmark_stress, dtype=float),
        total_oos_bars,
        "net_stress",
    )
    fold_expectancies = [
        f["primary"]["expectancy_bps"] for f in fold_results
        if f["primary"]["expectancy_bps"] is not None
    ]
    positive_folds_pct = (
        float(np.mean(np.asarray(fold_expectancies) > 0) * 100.0)
        if fold_expectancies else 0.0
    )

    ml, cls_l, ms, cls_s, thr_l, thr_s = last_model_state
    X_hold = X.iloc[cut_idx:]
    p_hold_l = predict_side_proba(ml, cls_l, X_hold)[:, 2]
    p_hold_s = predict_side_proba(ms, cls_s, X_hold)[:, 2]
    hold_labels = labels.iloc[cut_idx:].reset_index(drop=True)
    hold_times = times.iloc[cut_idx:].reset_index(drop=True)
    hold_execution = _execute_signals(
        p_hold_l, p_hold_s, thr_l, thr_s,
        hold_labels, hold_times, ft, fr, fm,
    )
    hold_benchmark = _long_benchmark(
        hold_labels, hold_times, ft, fr, fm, FEE_BPS_PRIMARY
    )
    hold_benchmark_stress = _long_benchmark(
        hold_labels, hold_times, ft, fr, fm, FEE_BPS_STRESS
    )
    hold_primary = _metrics_with_ci(
        hold_execution, hold_benchmark, len(hold_times), "net_primary"
    )
    hold_stress = _metrics_with_ci(
        hold_execution, hold_benchmark_stress, len(hold_times), "net_stress"
    )

    if len(hold_execution["indices"]):
        selected = hold_execution["indices"]
        sides = hold_execution["sides"]
        long_reason = hold_labels["long_exit_reason"].to_numpy()[selected]
        short_reason = hold_labels["short_exit_reason"].to_numpy()[selected]
        reasons = np.where(sides == 1, long_reason, short_reason)
        hold_primary["long_trades"] = int((sides == 1).sum())
        hold_primary["short_trades"] = int((sides == -1).sum())
        hold_primary["tp_share_pct"] = round(float((reasons == "take_profit").mean() * 100), 2)
        hold_primary["sl_share_pct"] = round(float((reasons == "stop_loss").mean() * 100), 2)
        hold_primary["timeout_share_pct"] = round(float((reasons == "time").mean() * 100), 2)

    return {
        "tp_mult_sigma": tp_mult,
        "sl_mult_sigma": sl_mult,
        "funding_features": bool(use_funding),
        "feature_count": int(X.shape[1]),
        "folds": fold_results,
        "positive_oos_folds_pct": round(positive_folds_pct, 2),
        "median_fold_expectancy_bps": (
            round(float(np.median(fold_expectancies)), 3) if fold_expectancies else None
        ),
        "pooled_oos_primary": oos_primary,
        "pooled_oos_stress10bps": oos_stress,
        "untouched_holdout_primary": hold_primary,
        "untouched_holdout_stress10bps": hold_stress,
        "final_frozen_thresholds": {
            "long_tp_probability": round(thr_l, 7),
            "short_tp_probability": round(thr_s, 7),
        },
    }


def gate_result(name: str, result: dict | None, error: str | None = None) -> dict:
    if result is None:
        return {
            "config": name,
            "error": error,
            "checks": {},
            "checks_passed": 0,
            "checks_total": 10,
            "verdict": "NO-GO",
        }
    oos = result["pooled_oos_primary"]
    hold = result["untouched_holdout_primary"]
    stress = result["untouched_holdout_stress10bps"]
    ci = oos.get("bootstrap_ci95_expectancy_bps", [None, None])
    checks = {
        "oos_expectancy_positive": oos.get("expectancy_bps") is not None and oos["expectancy_bps"] > 0,
        "oos_pf_ge_1_2": oos.get("profit_factor") is not None and oos["profit_factor"] >= 1.2,
        "positive_oos_folds_ge_80pct": result["positive_oos_folds_pct"] >= 80.0,
        "oos_bootstrap_lower_gt_zero": ci[0] is not None and ci[0] > 0,
        "holdout_win_rate_ge_60pct": hold.get("win_rate_pct") is not None and hold["win_rate_pct"] >= 60.0,
        "holdout_expectancy_positive": hold.get("expectancy_bps") is not None and hold["expectancy_bps"] > 0,
        "holdout_pf_gt_1": hold.get("profit_factor") is not None and hold["profit_factor"] > 1.0,
        "holdout_stress_expectancy_positive": stress.get("expectancy_bps") is not None and stress["expectancy_bps"] > 0,
        "holdout_min_30_trades": hold.get("trades", 0) >= 30,
        "holdout_bootstrap_lower_gt_zero": (
            hold.get("bootstrap_ci95_expectancy_bps", [None, None])[0] is not None
            and hold["bootstrap_ci95_expectancy_bps"][0] > 0
        ),
    }
    passed = int(sum(bool(v) for v in checks.values()))
    return {
        "config": name,
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "verdict": "GO_TO_SHADOW_ONLY" if passed == len(checks) else "NO-GO",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correctness-hardened 1d BTCUSDT asymmetric barrier experiment (paper only)."
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("OSH_DB_URL", "postgresql+psycopg2://postgres@127.0.0.1:5433/osh"),
    )
    parser.add_argument("--table", default="public.btcusdt_15m")
    parser.add_argument("--funding-csv", default="binance_btcusdt_funding_history.csv")
    parser.add_argument("--data-end-utc", default=str(DATA_END_UTC))
    parser.add_argument("--out", default="high_winrate_hunt_1d_fixed.json")
    args = parser.parse_args()

    print("=== OSH HIGH-WINRATE HUNT 1d — CORRECTNESS HARDENED / PAPER ONLY ===", flush=True)
    print("Fixed 7-config experiment; no holdout tuning; live_orders=false", flush=True)

    df = rv.load_btc(args.db_url, args.table)
    cutoff = pd.Timestamp(args.data_end_utc)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    df = df.loc[df["candle_time"] <= cutoff].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("No BTC candles remain before --data-end-utc.")
    if len(df) < BARS_PER_DAY * (HOLDOUT_DAYS + TRAIN_DAYS + FOLDS * TEST_DAYS + 30):
        raise SystemExit("Not enough BTC history for the fixed walk-forward and holdout protocol.")
    print(
        f"[data] rows={len(df):,} from {df['candle_time'].iloc[0]} to {df['candle_time'].iloc[-1]}",
        flush=True,
    )

    funding = load_funding(Path(args.funding_csv))
    funding = funding.loc[funding["funding_time"] <= cutoff + pd.Timedelta(minutes=15)].reset_index(drop=True)
    mapped_funding = funding_mark_price_proxy(df, funding)
    ft = mapped_funding["funding_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    fr = mapped_funding["funding_rate"].to_numpy(dtype=float)
    fm = mapped_funding["mark_price_proxy"].to_numpy(dtype=float)
    if len(ft) < 100 or not (np.diff(ft) > 0).all():
        raise SystemExit("Funding settlements are insufficient or not strictly increasing.")
    max_gap_hours = float(np.diff(ft).max() / 3_600_000_000_000) if len(ft) > 1 else None
    if max_gap_hours is not None and max_gap_hours > 24.0:
        raise SystemExit(
            f"Funding settlement gap {max_gap_hours:.2f}h exceeds the 24h fail-closed limit."
        )
    print(
        f"[funding] settlements={len(ft):,}; mark-price=previous completed 15m close proxy; "
        f"max_gap={max_gap_hours:.2f}h",
        flush=True,
    )

    X_base, base_features = pv.make_features(df)
    X_funding = funding_features(df, funding)
    labels_by_config = {}

    dev_end = len(df) - BARS_PER_DAY * HOLDOUT_DAYS
    dev_cut_time = pd.Timestamp(df["candle_time"].iloc[dev_end])
    results = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_source": f"PostgreSQL {args.table}",
            "data_rows": int(len(df)),
            "first_candle_utc": str(df["candle_time"].iloc[0]),
            "last_candle_utc": str(df["candle_time"].iloc[-1]),
            "development_holdout_cut_utc": str(dev_cut_time),
            "holdout_days": HOLDOUT_DAYS,
            "horizon_bars": H,
            "horizon_days": 1,
            "selection": f"top {TOP_PCT:.0%} per side from frozen calibration quantiles; <=1 entry/day",
            "preregistered_configs": [c[0] for c in CONFIGS],
            "primary_cost_bps": FEE_BPS_PRIMARY,
            "stress_cost_bps": FEE_BPS_STRESS,
            "fees": "maker/maker 4 bps RT plus 2 bps RT slippage scenario",
            "funding_source": str(Path(args.funding_csv).resolve()),
            "funding_settlements": int(len(ft)),
            "funding_first_utc": str(mapped_funding["funding_time"].iloc[0]),
            "funding_last_utc": str(mapped_funding["funding_time"].iloc[-1]),
            "funding_max_gap_hours": round(max_gap_hours, 3) if max_gap_hours is not None else None,
            "funding_mark_price_method": (
                "previous completed BTCUSDT 15m close at/before settlement; "
                "proxy only, not the true Binance markPrice"
            ),
            "ambiguous_funding_boundary": "payments charged; credits ignored",
            "capital_model": (
                "1x no leverage; position=min(cap,90% current equity); "
                "no exchange liquidation/order-book fill model"
            ),
            "live_orders": False,
        },
        "configs": {},
        "gates": [],
    }

    for name, tp_mult, sl_mult, use_funding in CONFIGS:
        print(
            f"\n--- {name}: TP={tp_mult} sigma / SL={sl_mult} sigma / funding_features={use_funding} ---",
            flush=True,
        )
        try:
            labels = asymmetric_barrier_labels(df, H, tp_mult, sl_mult)
            valid = (
                labels["valid"].to_numpy(dtype=bool)
                & X_base.notna().all(axis=1).to_numpy(dtype=bool)
            )
            label_df = labels.loc[valid].reset_index(drop=True)
            times = df.loc[valid, "candle_time"].reset_index(drop=True)
            if len(times) < 1000:
                raise ValueError("Too few valid label rows.")
            deltas = times.diff().dropna()
            if not deltas.eq(pd.Timedelta(minutes=TF_MINUTES)).all():
                raise ValueError(
                    "Valid training rows are not contiguous 15m bars; refusing to compress time."
                )
            Xb = X_base.loc[valid].reset_index(drop=True)
            Xf = X_funding.loc[valid].reset_index(drop=True)
            result = run_config(
                name, tp_mult, sl_mult, use_funding,
                Xb, Xf, label_df, times, dev_cut_time, (ft, fr, fm),
            )
            results["configs"][name] = result
            gate = gate_result(name, result)
            results["gates"].append(gate)
            hold = result["untouched_holdout_primary"]
            print(
                f"[{name}] OOS WR={result['pooled_oos_primary'].get('win_rate_pct')}% "
                f"EXP={result['pooled_oos_primary'].get('expectancy_bps')} bps; "
                f"HOLD WR={hold.get('win_rate_pct')}% "
                f"EXP={hold.get('expectancy_bps')} bps "
                f"$100cap={hold.get('final_balance_100usd_cap')} "
                f"{gate['verdict']} ({gate['checks_passed']}/{gate['checks_total']})",
                flush=True,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            results["configs"][name] = {"status": "INVALID_OR_NOT_EVALUATED", "error": message}
            results["gates"].append(gate_result(name, None, message))
            print(f"[{name}] INVALID / NO-GO: {message}", flush=True)

    candidate_gates = [
        g for g in results["gates"] if not g["config"].endswith("_TRAP")
    ]
    all_pass = any(g["verdict"] == "GO_TO_SHADOW_ONLY" for g in candidate_gates)
    results["summary"] = {
        "configs_evaluated": int(sum("status" not in results["configs"][g["config"]] for g in results["gates"])),
        "configs_total": len(CONFIGS),
        "candidate_configs_total": len(candidate_gates),
        "trap_demo_excluded_from_go_decision": True,
        "any_config_passed_all_gates": all_pass,
        "overall_verdict": "GO_TO_SHADOW_ONLY" if all_pass else "NO-GO",
        "note": (
            "Even a passing result permits shadow/paper validation only, never automatic live trading."
        ),
    }

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\n[report] {out_path.resolve()}", flush=True)
    print("\n=== FINAL TABLE ===", flush=True)
    print(f"{'config':<28} {'OOS WR':>8} {'OOS EXP':>9} {'HOLD WR':>8} {'HOLD EXP':>9} {'HOLD $100':>10} verdict")
    for gate in results["gates"]:
        r = results["configs"][gate["config"]]
        if "status" in r:
            print(f"{gate['config']:<28} {'—':>8} {'—':>9} {'—':>8} {'—':>9} {'—':>10} NO-GO")
            continue
        oos = r["pooled_oos_primary"]
        hold = r["untouched_holdout_primary"]
        print(
            f"{gate['config']:<28} {str(oos.get('win_rate_pct')):>8} "
            f"{str(oos.get('expectancy_bps')):>9} {str(hold.get('win_rate_pct')):>8} "
            f"{str(hold.get('expectancy_bps')):>9} "
            f"{str(hold.get('final_balance_100usd_cap')):>10} {gate['verdict']}"
        )
    print("\nNo live orders were sent. Do not optimize against this holdout.", flush=True)


if __name__ == "__main__":
    main()
