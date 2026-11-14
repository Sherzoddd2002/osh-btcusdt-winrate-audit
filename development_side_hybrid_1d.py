"""Development-only side-specific barrier / hybrid strategy research.

The final 180-day holdout is deliberately excluded from feature generation,
label generation, model fitting, threshold calibration, scoring and risk sizing.
This is discovery research only; the hybrid hypothesis was inspired by prior
development OOS side-attribution and therefore needs a genuinely later forward
paper period before it can be considered independently validated.
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
import high_winrate_hunt_1d as hunt
from execution_costs_1d import (
    DEFAULT_MAKER_FEE_BPS, DEFAULT_TAKER_FEE_BPS,
    DEFAULT_ROUND_TRIP_SLIPPAGE_BPS,
    SCENARIO_LEGACY_MAKER_MAKER, SCENARIO_TAKER_ALL,
    SCENARIO_TAKER_ENTRY_TP_MAKER,
    build_execution_scenarios, calculate_trade_costs,
)
from risk_management_1d import run_risk_budget_grid, summarize_trade_attribution


BARS_PER_DAY = 96
HOLDOUT_DAYS = 180
TRAIN_DAYS = 365
TEST_DAYS = 90
FOLDS = 5
H = 96
SL_MULT = 1.0
VARIANTS = (
    ("C2_BASE_TP_0.75_BOTH", 0.75, 0.75),
    ("C3_BASE_TP_0.50_BOTH", 0.50, 0.50),
    ("H1_HYBRID_LONG_0.50_SHORT_0.75", 0.50, 0.75),
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _benchmark_with_execution_fees(
    labels: pd.DataFrame,
    times: pd.Series,
    funding_times_ns: np.ndarray,
    funding_rates: np.ndarray,
    funding_marks: np.ndarray,
    *,
    maker_fee_bps: float,
    taker_fee_bps: float,
    round_trip_slippage_bps: float,
    bnb_discount_pct: float,
    use_maker_exit: bool,
) -> np.ndarray:
    """24h long benchmark under legacy Maker/Maker or Taker/Taker costs."""
    values = []
    for i in range(0, max(0, len(times) - H), H):
        gross = float(labels["long_horizon_exit"].iloc[i])
        entry_price = float(labels["entry_price"].iloc[i])
        entry_time = pd.Timestamp(times.iloc[i + 1])
        exit_time = pd.Timestamp(times.iloc[i + H]) + pd.Timedelta(minutes=hunt.TF_MINUTES)
        funding = hunt.funding_pnl_for_trade(
            1, entry_time, exit_time, entry_price,
            funding_times_ns, funding_rates, funding_marks,
        )
        fee = maker_fee_bps if use_maker_exit else taker_fee_bps
        costs = calculate_trade_costs(
            gross, 1, "time",
            entry_fee_bps=fee,
            exit_fee_bps=fee,
            round_trip_slippage_bps=round_trip_slippage_bps,
            bnb_discount_pct=bnb_discount_pct,
        )
        values.append(gross - costs["total_cost_return"] + funding)
    return np.asarray(values, dtype=float)


def _scenario_cost_summary(scenarios: dict[str, dict[str, np.ndarray]]) -> dict:
    result = {}
    for name, parts in scenarios.items():
        count = len(parts["net_return"])
        def avg(key: str):
            return round(float(np.mean(parts[key]) * 10_000.0), 4) if count else None
        result[name] = {
            "trades": count,
            "mean_entry_commission_bps": avg("entry_commission_return"),
            "mean_exit_commission_bps": avg("exit_commission_return"),
            "mean_total_commission_bps": avg("commission_return"),
            "mean_slippage_bps": avg("slippage_return"),
            "mean_total_execution_cost_bps": avg("total_cost_return"),
            "mean_funding_return_bps": avg("funding_return"),
            "net_expectancy_bps": avg("net_return"),
        }
    return result


def _class_ids(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_long = np.where(
        labels["long_label"].to_numpy() == -1, 0,
        np.where(labels["long_label"].to_numpy() == 0, 1, 2),
    ).astype(np.int8)
    y_short = np.where(
        labels["short_label"].to_numpy() == -1, 0,
        np.where(labels["short_label"].to_numpy() == 0, 1, 2),
    ).astype(np.int8)
    return y_long, y_short


def run_variant(
    name: str,
    tp_long: float,
    tp_short: float,
    candles: pd.DataFrame,
    X_base: pd.DataFrame,
    funding_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    fee_schedule: tuple[float, float, float, float],
) -> dict:
    labels_all = hunt.asymmetric_barrier_labels(
        candles, H, tp_long, SL_MULT, tp_mult_short=tp_short
    )
    valid, warmup_rows = hunt._contiguous_label_valid_mask(labels_all, X_base)
    labels = labels_all.loc[valid].reset_index(drop=True)
    X = X_base.loc[valid].reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    times = candles.loc[valid, "candle_time"].reset_index(drop=True)

    if len(times) > 1 and not times.diff().dropna().eq(
        pd.Timedelta(minutes=hunt.TF_MINUTES)
    ).all():
        raise ValueError("Development training rows are not contiguous 15m bars.")

    cut_idx = len(times)
    folds = pv.make_folds(
        cut_idx,
        BARS_PER_DAY * TRAIN_DAYS,
        BARS_PER_DAY * TEST_DAYS,
        H + 2,
        FOLDS,
    )
    if len(folds) != FOLDS:
        raise ValueError(
            f"{name}: expected {FOLDS} development folds, got {len(folds)}."
        )

    y_long, y_short = _class_ids(labels)
    w_long = labels["long_sample_weight"].to_numpy(dtype=float)
    w_short = labels["short_sample_weight"].to_numpy(dtype=float)
    ft, fr, fm = funding_arrays

    fold_rows = []
    pooled_net: list[float] = []
    pooled_stress: list[float] = []
    pooled_legacy: list[float] = []
    pooled_fee_parts = {
        key: {part: [] for part in (
            "entry_commission_return", "exit_commission_return",
            "commission_return", "slippage_return", "total_cost_return",
            "funding_return", "net_return",
        )}
        for key in (SCENARIO_LEGACY_MAKER_MAKER, SCENARIO_TAKER_ENTRY_TP_MAKER, SCENARIO_TAKER_ALL)
    }
    pooled_gross: list[float] = []
    pooled_sides: list[int] = []
    pooled_reasons: list[str] = []
    pooled_mae: list[float] = []
    pooled_mfe: list[float] = []
    pooled_stop_distance: list[float] = []
    pooled_dates: list[str] = []
    pooled_holding: list[int] = []
    pooled_benchmark = []
    pooled_benchmark_stress = []
    pooled_benchmark_legacy = []
    total_test_bars = 0

    for fold_no, (tr0, tr1, te0, te1) in enumerate(folds, 1):
        cal0 = max(tr0 + 100, tr1 - BARS_PER_DAY * min(90, TRAIN_DAYS))
        model_state = hunt._train_predict(
            X, y_long, y_short, w_long, w_short, tr0, cal0, tr1
        )
        ml, cls_l, ms, cls_s, thr_l, thr_s = model_state
        X_test = X.iloc[te0:te1]
        p_l = hunt.predict_side_proba(ml, cls_l, X_test)[:, 2]
        p_s = hunt.predict_side_proba(ms, cls_s, X_test)[:, 2]
        test_labels = labels.iloc[te0:te1].reset_index(drop=True)
        test_times = times.iloc[te0:te1].reset_index(drop=True)

        execution = hunt._execute_signals(
            p_l, p_s, thr_l, thr_s, test_labels, test_times, ft, fr, fm, SL_MULT
        )
        maker_fee_bps, taker_fee_bps, slippage_bps, bnb_discount_pct = fee_schedule
        fee_scenarios = build_execution_scenarios(
            execution["gross"], execution["sides"], execution["exit_reasons"],
            execution["funding"],
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            round_trip_slippage_bps=slippage_bps,
            bnb_discount_pct=bnb_discount_pct,
        )
        execution["net_legacy"] = fee_scenarios[SCENARIO_LEGACY_MAKER_MAKER]["net_return"]
        execution["net_primary"] = fee_scenarios[SCENARIO_TAKER_ENTRY_TP_MAKER]["net_return"]
        execution["net_stress"] = fee_scenarios[SCENARIO_TAKER_ALL]["net_return"]
        benchmark = _benchmark_with_execution_fees(
            test_labels, test_times, ft, fr, fm,
            maker_fee_bps=maker_fee_bps, taker_fee_bps=taker_fee_bps,
            round_trip_slippage_bps=slippage_bps, bnb_discount_pct=bnb_discount_pct,
            use_maker_exit=False,
        )
        benchmark_stress = benchmark.copy()
        benchmark_legacy = _benchmark_with_execution_fees(
            test_labels, test_times, ft, fr, fm,
            maker_fee_bps=maker_fee_bps, taker_fee_bps=taker_fee_bps,
            round_trip_slippage_bps=slippage_bps, bnb_discount_pct=bnb_discount_pct,
            use_maker_exit=True,
        )
        for scenario_name, parts in fee_scenarios.items():
            for part_name, values in parts.items():
                pooled_fee_parts[scenario_name][part_name].extend(values.tolist())
        primary = hunt._metrics_with_ci(
            execution, benchmark, te1 - te0, "net_primary"
        )
        stress = hunt._metrics_with_ci(
            execution, benchmark_stress, te1 - te0, "net_stress"
        )
        legacy = hunt._metrics_with_ci(
            execution, benchmark_legacy, te1 - te0, "net_legacy"
        )
        primary["execution_cost_assumptions"] = {
            "entry": "Taker",
            "take_profit_exit": "Maker conditional on passive limit fill",
            "stop_loss_exit": "Taker",
            "time_exit": "Taker",
        }
        stress["execution_cost_assumptions"] = {
            "entry": "Taker", "all_exit_reasons": "Taker"
        }
        attribution = summarize_trade_attribution(
            execution["net_primary"],
            execution["sides"],
            execution["exit_reasons"],
            execution["mae_returns"],
            execution["mfe_returns"],
            execution["holding_bars"],
        )
        primary["trade_attribution"] = attribution
        primary.update({
            "fold": fold_no,
            "test_start_utc": str(test_times.iloc[0]),
            "test_end_utc": str(test_times.iloc[-1]),
            "long_trades": int((execution["sides"] == 1).sum()),
            "short_trades": int((execution["sides"] == -1).sum()),
            "frozen_threshold_long": float(thr_l),
            "frozen_threshold_short": float(thr_s),
        })
        fold_rows.append({"primary": primary, "all_taker": stress, "legacy_maker_maker": legacy})

        pooled_net.extend(execution["net_primary"].tolist())
        pooled_stress.extend(execution["net_stress"].tolist())
        pooled_legacy.extend(execution["net_legacy"].tolist())
        pooled_gross.extend(execution["gross"].tolist())
        pooled_sides.extend(execution["sides"].tolist())
        pooled_reasons.extend(execution["exit_reasons"].tolist())
        pooled_mae.extend(execution["mae_returns"].tolist())
        pooled_mfe.extend(execution["mfe_returns"].tolist())
        pooled_stop_distance.extend(execution["stop_distance_returns"].tolist())
        pooled_dates.extend(execution["entry_dates"].tolist())
        pooled_holding.extend(execution["holding_bars"].tolist())
        pooled_benchmark.extend(benchmark.tolist())
        pooled_benchmark_stress.extend(benchmark_stress.tolist())
        pooled_benchmark_legacy.extend(benchmark_legacy.tolist())
        total_test_bars += te1 - te0

    net = np.asarray(pooled_net, dtype=float)
    gross = np.asarray(pooled_gross, dtype=float)
    stress_net = np.asarray(pooled_stress, dtype=float)
    legacy_net = np.asarray(pooled_legacy, dtype=float)
    if len(net) == 0:
        raise ValueError(f"{name}: no development OOS trades were selected.")

    pooled_execution = {
        "net_primary": net,
        "net_stress": stress_net,
        "net_legacy": legacy_net,
        "gross": gross,
    }
    primary = hunt._metrics_with_ci(
        pooled_execution, np.asarray(pooled_benchmark, dtype=float),
        total_test_bars, "net_primary",
    )
    stress = hunt._metrics_with_ci(
        pooled_execution, np.asarray(pooled_benchmark_stress, dtype=float),
        total_test_bars, "net_stress",
    )
    legacy = hunt._metrics_with_ci(
        pooled_execution, np.asarray(pooled_benchmark_legacy, dtype=float),
        total_test_bars, "net_legacy",
    )
    pooled_fee_scenarios = {
        scenario: {part: np.asarray(values, dtype=float) for part, values in parts.items()}
        for scenario, parts in pooled_fee_parts.items()
    }
    fold_expectancies = [
        row["primary"]["expectancy_bps"] for row in fold_rows
        if row["primary"]["expectancy_bps"] is not None
    ]
    positive_folds_pct = (
        float(np.mean(np.asarray(fold_expectancies) > 0) * 100.0)
        if fold_expectancies else 0.0
    )
    attribution = summarize_trade_attribution(
        net,
        np.asarray(pooled_sides, dtype=np.int8),
        np.asarray(pooled_reasons, dtype=object),
        np.asarray(pooled_mae, dtype=float),
        np.asarray(pooled_mfe, dtype=float),
        np.asarray(pooled_holding, dtype=float),
    )
    risk = run_risk_budget_grid(
        net,
        np.asarray(pooled_mae, dtype=float),
        np.asarray(pooled_stop_distance, dtype=float),
        np.asarray(pooled_dates, dtype=object),
        risk_levels_pct=(0.25, 0.5, 1.0),
        initial_capital=hunt.INITIAL_CAPITAL,
        max_notional=100.0,
        cash_reserve_pct=hunt.CASH_RESERVE_PCT,
        cost_buffer_bps=12.0,
        daily_loss_limit_pct=2.0,
        max_drawdown_stop_pct=15.0,
        max_consecutive_losses=3,
        cooldown_trades=1,
    )

    ci = primary.get("bootstrap_ci95_expectancy_bps", [None, None])
    risk_half = risk.get("risk_0.5pct", {})
    checks = {
        "net_win_rate_ge_60pct": primary.get("win_rate_pct") is not None
        and primary["win_rate_pct"] >= 60.0,
        "net_expectancy_positive": primary.get("expectancy_bps") is not None
        and primary["expectancy_bps"] > 0,
        "profit_factor_ge_1_25": primary.get("profit_factor") is not None
        and primary["profit_factor"] >= 1.25,
        "positive_oos_folds_ge_80pct": positive_folds_pct >= 80.0,
        "bootstrap_lower_gt_zero": ci[0] is not None and ci[0] > 0,
        "taker_stress_expectancy_positive": stress.get("expectancy_bps") is not None
        and stress["expectancy_bps"] > 0,
        "half_pct_risk_drawdown_below_15pct": risk_half.get(
            "max_closed_trade_drawdown_pct"
        ) is not None and risk_half["max_closed_trade_drawdown_pct"] < 15.0,
        "at_least_100_oos_trades": primary.get("trades", 0) >= 100,
    }
    last_primary = fold_rows[-1]["primary"]
    return {
        "tp_long_sigma": tp_long,
        "tp_short_sigma": tp_short,
        "sl_sigma": SL_MULT,
        "funding_features": False,
        "feature_count": int(X.shape[1]),
        "folds": fold_rows,
        "pooled_oos_primary": primary,
        "pooled_oos_all_taker": stress,
        "pooled_oos_legacy_maker_maker": legacy,
        "execution_cost_summary": _scenario_cost_summary(pooled_fee_scenarios),
        "positive_oos_folds_pct": round(positive_folds_pct, 3),
        "median_fold_expectancy_bps": (
            float(np.median(fold_expectancies)) if fold_expectancies else None
        ),
        "development_oos_trade_attribution": attribution,
        "development_oos_risk_budget_scenarios": risk,
        "development_checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "verdict": (
            "DEVELOPMENT_RESEARCH_CANDIDATE"
            if all(checks.values()) else "DISCOVERY_ONLY_NO_GO"
        ),
        "last_oos_fold_thresholds": {
            "long_tp_probability": last_primary["frozen_threshold_long"],
            "short_tp_probability": last_primary["frozen_threshold_short"],
        },
    }


def run_development_hunt(
    df_all: pd.DataFrame,
    funding_all: pd.DataFrame,
    holdout_days: int = HOLDOUT_DAYS,
    fee_schedule: tuple[float, float, float, float] = (
        DEFAULT_MAKER_FEE_BPS, DEFAULT_TAKER_FEE_BPS,
        DEFAULT_ROUND_TRIP_SLIPPAGE_BPS, 0.0,
    ),
) -> dict:
    if holdout_days < 1:
        raise ValueError("holdout_days must be positive.")
    df_all = rv.validate_market_data(df_all)
    holdout_rows = BARS_PER_DAY * holdout_days
    if len(df_all) <= holdout_rows:
        raise ValueError("Not enough rows to exclude the requested holdout.")
    dev_end_idx = len(df_all) - holdout_rows
    dev_cut_time = pd.Timestamp(df_all["candle_time"].iloc[dev_end_idx])
    # Critical: all features, labels and model data are built only after slicing
    # away the final holdout. The held-out candle values are never scored here.
    dev = df_all.iloc[:dev_end_idx].reset_index(drop=True)
    if len(dev) < BARS_PER_DAY * (TRAIN_DAYS + FOLDS * TEST_DAYS + 30):
        raise ValueError("Development-only history is too short for five OOS folds.")

    funding = funding_all.loc[
        funding_all["funding_time"] < dev_cut_time
    ].reset_index(drop=True)
    mapped = hunt.funding_mark_price_proxy(dev, funding)
    ft = mapped["funding_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    fr = mapped["funding_rate"].to_numpy(dtype=float)
    fm = mapped["mark_price_proxy"].to_numpy(dtype=float)
    if len(ft) < 100 or not (np.diff(ft) > 0).all():
        raise ValueError("Development funding settlements are insufficient or not strictly increasing.")
    funding_gap_hours = (
        float(np.diff(ft).max() / 3_600_000_000_000) if len(ft) > 1 else None
    )
    if funding_gap_hours is not None and funding_gap_hours > hunt.FUNDING_MAX_AGE_HOURS:
        raise ValueError(
            f"Development funding gap {funding_gap_hours:.2f}h exceeds the fail-closed limit."
        )
    funding_arrays = (ft, fr, fm)

    X_base, feature_names = pv.make_features(dev)
    output = {
        "metadata": {
            "experiment_version": "OSH-DEVELOPMENT-SIDE-HYBRID-2026-11-14-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "PostgreSQL BTCUSDT 15m; development slice only",
            "development_rows": int(len(dev)),
            "development_first_utc": str(dev["candle_time"].iloc[0]),
            "development_last_utc": str(dev["candle_time"].iloc[-1]),
            "excluded_holdout_start_utc": str(dev_cut_time),
            "excluded_holdout_days": holdout_days,
            "holdout_policy": (
                "The final holdout is removed before feature/label generation; "
                "no model scoring, threshold selection or risk sizing uses it."
            ),
            "hypothesis_warning": (
                "The hybrid long/short pairing was inspired by previously observed "
                "development OOS side attribution. This run is discovery research, "
                "not independent validation."
            ),
            "selection": "top 10% calibrated TP probability per side; at most one entry per 96 bars",
            "fee_schedule_bps": {
                "maker_per_side": fee_schedule[0],
                "taker_per_side": fee_schedule[1],
                "round_trip_slippage": fee_schedule[2],
                "bnb_discount_pct": fee_schedule[3],
                "source": "configurable assumptions; verify personal rates in Binance account",
            },
            "primary_costs": (
                "Taker entry; TP exit Maker conditional on passive limit fill; "
                "SL and TIME exits Taker; round-trip slippage included"
            ),
            "stress_costs": (
                "Taker entry and Taker exit for every exit reason; "
                "round-trip slippage included"
            ),
            "legacy_comparison": "Maker entry and Maker exit for all trades; optimistic baseline only",
            "maker_fill_warning": (
                "15m OHLC does not prove passive queue execution or Maker fill at TP; "
                "all-Taker scenario must also remain viable"
            ),
            "risk_grid": "0.25%, 0.5%, 1.0% stop-distance risk; development OOS only",
        },
        "variants": {},
    }

    for name, tp_long, tp_short in VARIANTS:
        print(
            f"\n--- {name}: LONG TP={tp_long}σ / SHORT TP={tp_short}σ / SL={SL_MULT}σ ---",
            flush=True,
        )
        result = run_variant(
            name, tp_long, tp_short, dev, X_base, funding_arrays, fee_schedule
        )
        output["variants"][name] = result
        stats = result["pooled_oos_primary"]
        print(
            f"[{name}] WR={stats.get('win_rate_pct')}% "
            f"EXP={stats.get('expectancy_bps')}bps "
            f"PF={stats.get('profit_factor')} "
            f"positive_folds={result['positive_oos_folds_pct']}% "
            f"checks={result['checks_passed']}/{result['checks_total']} "
            f"{result['verdict']}",
            flush=True,
        )
    return _json_safe(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only side-specific barrier research; never uses final holdout."
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv(
            "OSH_DB_URL",
            "postgresql+psycopg2://postgres@127.0.0.1:5433/osh",
        ),
    )
    parser.add_argument("--table", default="public.btcusdt_15m")
    parser.add_argument("--funding-csv", default="binance_btcusdt_funding_history.csv")
    parser.add_argument("--data-end-utc", default="2026-10-31T23:45:00Z")
    parser.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS)
    parser.add_argument("--maker-fee-bps", type=float, default=DEFAULT_MAKER_FEE_BPS)
    parser.add_argument("--taker-fee-bps", type=float, default=DEFAULT_TAKER_FEE_BPS)
    parser.add_argument("--round-trip-slippage-bps", type=float, default=DEFAULT_ROUND_TRIP_SLIPPAGE_BPS)
    parser.add_argument("--bnb-discount-pct", type=float, default=0.0)
    parser.add_argument("--out", default="development_side_hybrid_1d.json")
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.data_end_utc)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    all_candles = rv.load_btc(args.db_url, args.table)
    all_candles = all_candles.loc[
        all_candles["candle_time"] <= cutoff
    ].reset_index(drop=True)
    funding = hunt.load_funding(Path(args.funding_csv))
    funding = funding.loc[
        funding["funding_time"] <= cutoff + pd.Timedelta(minutes=15)
    ].reset_index(drop=True)

    result = run_development_hunt(
        all_candles, funding, args.holdout_days,
        (args.maker_fee_bps, args.taker_fee_bps, args.round_trip_slippage_bps, args.bnb_discount_pct),
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\n[report] {output.resolve()}", flush=True)
    print("\n=== DEVELOPMENT-ONLY FINAL TABLE ===", flush=True)
    for name, row in result["variants"].items():
        stats = row["pooled_oos_primary"]
        print(
            f"{name:<34} WR={stats.get('win_rate_pct')}% "
            f"EXP={stats.get('expectancy_bps')}bps PF={stats.get('profit_factor')} "
            f"trades={stats.get('trades')} {row['verdict']}"
        )
    print(
        "\nFinal 180-day holdout was excluded. Discovery results require a later "
        "independent forward paper period; no live orders.",
        flush=True,
    )


if __name__ == "__main__":
    main()
