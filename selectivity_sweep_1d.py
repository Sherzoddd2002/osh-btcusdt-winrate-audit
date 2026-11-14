"""One-day BTCUSDT selectivity sweep: compare top 2%, 5%, and 10% signals.

Paper research only. Uses the same causal labels, fold builder, model,
non-overlapping execution, costs, and untouched holdout as research_v3.py.
All signal thresholds are calibrated on each training fold's held-back
calibration tail; neither test folds nor the final holdout set thresholds.
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
ONE_DAY_HORIZON = 96
DEFAULT_SELECTION_PCTS = (0.02, 0.05, 0.10)
Z95 = 1.959963984540054


def thresholds_for_pct(p_cal: np.ndarray, pct: float) -> tuple[float, float]:
    """Calibrate per-side probability cutoffs using development calibration data only."""
    if not 0.0 < float(pct) < 1.0:
        raise ValueError("selection pct must be strictly between 0 and 1")
    p_cal = np.asarray(p_cal, dtype=float)
    if p_cal.ndim != 2 or p_cal.shape[1] != 3 or len(p_cal) == 0:
        raise ValueError("calibration probabilities must be a non-empty N x 3 array")
    if not np.isfinite(p_cal).all():
        raise ValueError("calibration probabilities contain non-finite values")
    return (
        float(np.quantile(p_cal[:, 2], 1.0 - pct)),
        float(np.quantile(p_cal[:, 0], 1.0 - pct)),
    )


def wilson_interval_pct(wins: int, total: int) -> list[float | None]:
    """Two-sided 95% Wilson interval for the observed net win proportion."""
    if total <= 0:
        return [None, None]
    if wins < 0 or wins > total:
        raise ValueError("wins must be between zero and total")
    p = wins / total
    z2 = Z95 * Z95
    den = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / den
    half = Z95 * np.sqrt((p * (1.0 - p) / total) + z2 / (4.0 * total * total)) / den
    return [float(max(0.0, center - half) * 100.0),
            float(min(1.0, center + half) * 100.0)]


def trade_quality_stats(net: np.ndarray) -> dict:
    """Supplement expectancy/PF with payoff, drawdown, loss streak, and win-rate CI."""
    net = np.asarray(net, dtype=float)
    if not np.isfinite(net).all():
        raise ValueError("net trade returns must be finite")
    n = len(net)
    if n == 0:
        return {
            "win_rate_wilson95_pct": [None, None],
            "avg_net_win_bps": None,
            "avg_net_loss_bps": None,
            "net_payoff_ratio": None,
            "max_trade_sequence_drawdown_pct": None,
            "max_consecutive_losses": 0,
        }

    winners = net[net > 0]
    losers = net[net <= 0]
    avg_win = float(winners.mean() * 10_000) if len(winners) else None
    avg_loss = float(-losers.mean() * 10_000) if len(losers) else None
    payoff = (
        float(winners.mean() / abs(losers.mean()))
        if len(winners) and len(losers) and abs(losers.mean()) > 0
        else None
    )

    # Trade-sequence compounding assumes 1x notional per selected, non-overlapping trade.
    if np.any(net <= -1.0):
        drawdown = -100.0
    else:
        equity = np.concatenate(([1.0], np.cumprod(1.0 + net)))
        peaks = np.maximum.accumulate(equity)
        drawdown = float(np.min(equity / peaks - 1.0) * 100.0)

    max_loss_streak = 0
    current_streak = 0
    for is_loss in (net <= 0):
        current_streak = current_streak + 1 if is_loss else 0
        max_loss_streak = max(max_loss_streak, current_streak)

    return {
        "win_rate_wilson95_pct": wilson_interval_pct(int(len(winners)), n),
        "avg_net_win_bps": avg_win,
        "avg_net_loss_bps": avg_loss,
        "net_payoff_ratio": payoff,
        "max_trade_sequence_drawdown_pct": drawdown,
        "max_consecutive_losses": int(max_loss_streak),
    }


def _json_number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def run_selectivity_sweep(
    df: pd.DataFrame,
    selection_pcts: tuple[float, ...] | list[float] = DEFAULT_SELECTION_PCTS,
    cost_bps: float = rv.COST_BPS,
    folds_count: int = rv.N_FOLDS,
    train_days: int = 365,
    test_days: int = 90,
    holdout_days: int = rv.HOLDOUT_DAYS,
    source: str = "provided DataFrame",
) -> dict:
    """Train one model per fold, then compare fixed calibration thresholds on 1d BTC."""
    pcts = tuple(float(x) for x in selection_pcts)
    if not pcts or len(set(pcts)) != len(pcts):
        raise ValueError("selection_pcts must be non-empty and unique")
    if any(not 0.0 < x < 1.0 for x in pcts):
        raise ValueError("every selection pct must be strictly between 0 and 1")
    if cost_bps < 0 or min(folds_count, train_days, test_days, holdout_days) < 1:
        raise ValueError("cost must be non-negative and all date/fold windows positive")

    df = rv.validate_market_data(df)
    X_base, _ = pv.make_features(df)
    labels = rv.barrier_labels(df, ONE_DAY_HORIZON, cost_bps / 10_000.0)
    label3 = labels["label3"].to_numpy()
    y_all = np.where(label3 == -1, 0, np.where(label3 == 0, 1, 2)).astype("int8")
    valid = labels["valid"].to_numpy(bool)
    weights = labels["sample_weight"].to_numpy(float)
    returns = labels["exit_ret"].to_numpy(float)

    ok = valid & X_base.notna().all(axis=1).to_numpy()
    Xi = X_base.loc[ok].reset_index(drop=True).fillna(0.0)
    yi = y_all[ok]
    wi = weights[ok]
    ri = returns[ok]
    ti = df["candle_time"].loc[ok].reset_index(drop=True)
    if len(Xi) == 0:
        raise ValueError("No valid one-day training rows remain after feature/label validation.")

    holdout_bars = BARS_PER_DAY * holdout_days
    if len(df) <= holdout_bars:
        raise ValueError("Dataset is shorter than the requested untouched holdout.")
    dev_cut_time = df["candle_time"].iloc[len(df) - holdout_bars]
    cut_idx = int((ti < dev_cut_time).sum())
    min_train = BARS_PER_DAY * train_days
    test_len = BARS_PER_DAY * test_days
    fold_list = pv.make_folds(cut_idx, min_train, test_len, ONE_DAY_HORIZON + 2, folds_count)
    if not fold_list:
        raise ValueError(
            f"Not enough development rows for 1d: need at least {train_days} train days "
            f"+ {test_days} test days, plus embargo."
        )

    buckets = {
        pct: {"folds": [], "net_parts": [], "benchmark_parts": [], "test_bars": 0}
        for pct in pcts
    }
    last_model = None
    last_thresholds: dict[float, tuple[float, float]] = {}

    for fold_no, (tr0, tr1, te0, te1) in enumerate(fold_list, 1):
        # The calibration tail is excluded from model fitting and early stopping.
        cal0 = max(tr0 + 100, tr1 - BARS_PER_DAY * min(90, train_days))
        if cal0 >= tr1:
            raise ValueError("Train fold is too short for a held-back calibration tail.")

        model = rv.fit_model(Xi.iloc[tr0:cal0], yi[tr0:cal0], wi[tr0:cal0],
                             ONE_DAY_HORIZON)
        p_cal = rv.predict_proba_3class(model, Xi.iloc[cal0:tr1])
        p_test = rv.predict_proba_3class(model, Xi.iloc[te0:te1])
        test_returns = ri[te0:te1]

        benchmark_idx = np.arange(0, max(0, te1 - te0 - ONE_DAY_HORIZON),
                                  ONE_DAY_HORIZON)
        benchmark = test_returns[benchmark_idx] - cost_bps / 10_000.0

        for pct in pcts:
            thr_up, thr_dn = thresholds_for_pct(p_cal, pct)
            selected, sides, net = rv.select_trades(
                p_test[:, 2], p_test[:, 0], test_returns, ONE_DAY_HORIZON,
                cost_bps / 10_000.0, thr_up, thr_dn,
            )
            gross = net + cost_bps / 10_000.0
            row = rv.metrics(
                net, gross, cost_bps, ONE_DAY_HORIZON, benchmark, te1 - te0
            )
            row.update(trade_quality_stats(net))
            row.update({
                "fold": int(fold_no),
                "test_start_utc": str(ti.iloc[te0]),
                "test_end_utc": str(ti.iloc[te1 - 1]),
                "calibrated_top_pct_per_side": pct * 100.0,
                "frozen_threshold_up": thr_up,
                "frozen_threshold_down": thr_dn,
                "test_signal_rate_pct": float(
                    ((p_test[:, 2] >= thr_up) | (p_test[:, 0] >= thr_dn)).mean() * 100.0
                ),
                "test_bars": int(te1 - te0),
            })
            bucket = buckets[pct]
            bucket["folds"].append(row)
            if len(net):
                bucket["net_parts"].append(net)
            if len(benchmark):
                bucket["benchmark_parts"].append(benchmark)
            bucket["test_bars"] += int(te1 - te0)
            last_thresholds[pct] = (thr_up, thr_dn)

        last_model = model

    if last_model is None:
        raise RuntimeError("No model was trained; refusing to create an empty sweep.")

    p_holdout = rv.predict_proba_3class(last_model, Xi.iloc[cut_idx:])
    summaries = {}
    gates = []

    for pct in pcts:
        bucket = buckets[pct]
        pooled = (
            np.concatenate(bucket["net_parts"])
            if bucket["net_parts"] else np.array([], dtype=float)
        )
        pooled_benchmark = (
            np.concatenate(bucket["benchmark_parts"])
            if bucket["benchmark_parts"] else np.array([], dtype=float)
        )
        pooled_gross = pooled + cost_bps / 10_000.0
        pooled_metrics = rv.metrics(
            pooled, pooled_gross, cost_bps, ONE_DAY_HORIZON,
            pooled_benchmark, bucket["test_bars"],
        )
        pooled_metrics.update(trade_quality_stats(pooled))

        fold_expectancies = [
            float(row["expectancy_bps"]) for row in bucket["folds"]
            if row.get("expectancy_bps") is not None
        ]
        # Empty-trade folds count as not-positive for consistency, but are
        # excluded from the median expectancy calculation.
        positive_fold_count = sum(
            row.get("expectancy_bps") is not None and row["expectancy_bps"] > 0
            for row in bucket["folds"]
        )
        positive_folds_pct = positive_fold_count / max(1, len(bucket["folds"])) * 100.0
        ci_lo, ci_hi = rv.block_bootstrap_ci(pooled)
        median_fold_expectancy = (
            float(np.median(fold_expectancies)) if fold_expectancies else None
        )

        thr_up, thr_dn = last_thresholds[pct]
        hold_idx, hold_sides, hold_net = rv.select_trades(
            p_holdout[:, 2], p_holdout[:, 0], ri[cut_idx:],
            ONE_DAY_HORIZON, cost_bps / 10_000.0, thr_up, thr_dn,
        )
        hold_gross = hold_net + cost_bps / 10_000.0
        hold_benchmark_idx = np.arange(
            0, max(0, len(ri) - cut_idx - ONE_DAY_HORIZON), ONE_DAY_HORIZON
        )
        hold_benchmark = ri[cut_idx:][hold_benchmark_idx] - cost_bps / 10_000.0
        holdout = rv.metrics(
            hold_net, hold_gross, cost_bps, ONE_DAY_HORIZON,
            hold_benchmark, len(ri) - cut_idx,
        )
        holdout.update(trade_quality_stats(hold_net))
        holdout.update({
            "note": "frozen final development model and calibration thresholds; no refit or tuning",
            "calibrated_top_pct_per_side": pct * 100.0,
            "frozen_threshold_up": thr_up,
            "frozen_threshold_down": thr_dn,
        })

        checks = {
            "median_fold_expectancy_positive": bool(
                median_fold_expectancy is not None and median_fold_expectancy > 0
            ),
            "positive_folds_at_least_75_pct": bool(positive_folds_pct >= 75.0),
            "pooled_profit_factor_at_least_1_30": bool(
                pooled_metrics.get("profit_factor") is not None
                and pooled_metrics["profit_factor"] >= 1.30
            ),
            "bootstrap_lower_bound_positive": bool(ci_lo is not None and ci_lo > 0),
            "every_fold_beats_benchmark": bool(
                bucket["folds"] and all(
                    row.get("edge_vs_benchmark_bps") is not None
                    and row["edge_vs_benchmark_bps"] > 0 for row in bucket["folds"]
                )
            ),
            "holdout_expectancy_positive": bool(
                holdout.get("expectancy_bps") is not None
                and holdout["expectancy_bps"] > 0
            ),
            "holdout_profit_factor_above_1": bool(
                holdout.get("profit_factor") is not None
                and holdout["profit_factor"] > 1.0
            ),
            "holdout_at_least_30_trades": bool(holdout.get("trades", 0) >= 30),
        }

        key = f"{pct * 100:g}%"
        summaries[key] = {
            "calibrated_top_pct_per_side": pct * 100.0,
            "folds": bucket["folds"],
            "positive_folds_pct": round(positive_folds_pct, 2),
            "median_fold_expectancy_bps": _json_number(median_fold_expectancy),
            "pooled_oos": pooled_metrics,
            "bootstrap_ci95_expectancy_bps": [
                _json_number(ci_lo), _json_number(ci_hi)
            ],
            "total_oos_trades": int(len(pooled)),
            "untouched_holdout": holdout,
            "checks": checks,
            "verdict": "GO" if all(checks.values()) else "NO-GO",
        }
        gates.append({
            "horizon": "1d",
            "selection": key,
            "checks_passed": int(sum(checks.values())),
            "checks_total": len(checks),
            "verdict": summaries[key]["verdict"],
        })

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "rows": int(len(df)),
            "first_candle_utc": str(df["candle_time"].iloc[0]),
            "last_candle_utc": str(df["candle_time"].iloc[-1]),
            "interval_minutes": rv.TF_MINUTES,
            "horizon": "1d",
            "horizon_bars": ONE_DAY_HORIZON,
            "model": "BASE only; same trained model per fold shared across all selectivity thresholds",
            "selection_rule": (
                "Each side's probability cutoff is the (1-pct) quantile of that "
                "fold's held-back development calibration tail; frozen before test."
            ),
            "execution_rule": (
                "Next-open entry labels from research_v3; non-overlapping one-day "
                "trades; same-bar double-touch is a stop; adverse stop gaps retained."
            ),
            "holdout_rule": "Final holdout is never used to set thresholds, fit, early-stop, or tune.",
            "live_orders": False,
        },
        "config": {
            "cost_bps_round_trip": float(cost_bps),
            "selection_pcts_per_side": [float(x) for x in pcts],
            "folds": int(folds_count),
            "train_days": int(train_days),
            "test_days": int(test_days),
            "holdout_days": int(holdout_days),
        },
        "selections": summaries,
        "gates": gates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare 1d BTCUSDT signal selectivity. Paper research only; no orders."
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv(
            "OSH_DB_URL",
            "postgresql+psycopg2://postgres@127.0.0.1:5433/osh",
        ),
        help="SQLAlchemy PostgreSQL URL; prefer OSH_DB_URL.",
    )
    parser.add_argument("--table", default="public.btcusdt_15m")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--selection-pcts", default="0.02,0.05,0.10")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--holdout-days", type=int, default=180)
    parser.add_argument("--out", default="selectivity_sweep_1d.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        pcts = tuple(float(x.strip()) for x in args.selection_pcts.split(",") if x.strip())
    except ValueError as exc:
        raise SystemExit("--selection-pcts must be comma-separated decimal fractions, e.g. 0.02,0.05,0.10") from exc
    if not pcts or len(set(pcts)) != len(pcts) or any(not 0 < x < 1 for x in pcts):
        raise SystemExit("--selection-pcts must contain unique values strictly between 0 and 1.")
    if args.cost_bps < 0 or min(args.folds, args.train_days, args.test_days, args.holdout_days) < 1:
        raise SystemExit("Cost must be non-negative and fold/date windows must be positive.")

    frame = rv.load_btc(args.db_url, args.table)
    results = run_selectivity_sweep(
        frame,
        selection_pcts=pcts,
        cost_bps=args.cost_bps,
        folds_count=args.folds,
        train_days=args.train_days,
        test_days=args.test_days,
        holdout_days=args.holdout_days,
        source=f"PostgreSQL {args.table}",
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== OSH 1D SELECTIVITY SWEEP — PAPER ONLY ===")
    print("Selection | OOS trades | OOS win% | OOS PF | OOS exp(bps) | Holdout trades | Holdout win% | Holdout exp(bps) | Verdict")
    for name, row in results["selections"].items():
        oos = row["pooled_oos"]
        hold = row["untouched_holdout"]
        print(
            f"{name:>9} | {oos.get('trades', 0):>10} | "
            f"{oos.get('win_rate_pct') if oos.get('win_rate_pct') is not None else 'n/a':>8} | "
            f"{oos.get('profit_factor') if oos.get('profit_factor') is not None else 'n/a':>7} | "
            f"{oos.get('expectancy_bps') if oos.get('expectancy_bps') is not None else 'n/a':>13} | "
            f"{hold.get('trades', 0):>14} | "
            f"{hold.get('win_rate_pct') if hold.get('win_rate_pct') is not None else 'n/a':>12} | "
            f"{hold.get('expectancy_bps') if hold.get('expectancy_bps') is not None else 'n/a':>17} | "
            f"{row['verdict']}"
        )
    print(f"[report] {output.resolve()}")
    print("[safety] live_orders=false; no Binance order endpoint or trading key is used.")


if __name__ == "__main__":
    main()
