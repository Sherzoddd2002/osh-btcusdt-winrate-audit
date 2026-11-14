"""Frozen, order-free forward paper evaluation for H96 C2.

The final 180-day historical holdout is never used. The model is fitted once on
pre-holdout candles, calibrated on a separate 90-day tail, serialized locally,
and reused unchanged for every subsequent forward snapshot. No exchange API,
credentials, or order placement are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pipeline_v2 as pv
import research_v3 as rv
import high_winrate_hunt_1d as hunt
from execution_costs_1d import build_execution_scenarios
from risk_management_1d import summarize_trade_attribution


BARS_PER_DAY = 96
HORIZON_BARS = 96
VOLATILITY_HORIZON_BARS = 96
ENTRY_COOLDOWN_BARS = 96
TRAIN_CALIBRATION_DAYS = 90
TOP_PCT = 0.10

TP_LONG_SIGMA = 0.75
TP_SHORT_SIGMA = 0.75
SL_SIGMA = 1.0

MAKER_FEE_BPS = 2.0
TAKER_FEE_BPS = 5.0
ROUND_TRIP_SLIPPAGE_BPS = 2.0
BNB_DISCOUNT_PCT = 0.0

INITIAL_CAPITAL_USD = 100.0
POSITION_CAP_USD = 50.0
CASH_RESERVE_PCT = 10.0

MIN_FORWARD_DAYS = 180
MIN_CLOSED_TRADES = 100
MAX_CANDLE_AGE_HOURS = 2.0
FUNDING_MAX_AGE_HOURS = 24.0

STRATEGY_ID = "OSH-BTCUSDT-15M-H96-C2-FORWARD-PAPER-v1"
VARIANT_KEY = "H96_C2_BASE_TP_0.75_BOTH"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _validate_discovery_report(path: Path) -> tuple[pd.Timestamp, str]:
    report = _read_json(path)
    metadata = report.get("metadata", {})
    variant = report.get("variants", {}).get(VARIANT_KEY)
    if not isinstance(variant, dict):
        raise ValueError(f"{path} does not contain {VARIANT_KEY}.")
    if int(metadata.get("excluded_holdout_days", 0)) != 180:
        raise ValueError("Discovery report does not confirm a 180-day excluded holdout.")
    if int(variant.get("time_exit_bars", -1)) != HORIZON_BARS:
        raise ValueError("Discovery report TIME horizon does not match locked H96.")
    if float(variant.get("tp_long_sigma", -1)) != TP_LONG_SIGMA:
        raise ValueError("Discovery report LONG TP does not match locked H96 C2.")
    if float(variant.get("tp_short_sigma", -1)) != TP_SHORT_SIGMA:
        raise ValueError("Discovery report SHORT TP does not match locked H96 C2.")
    if float(variant.get("sl_sigma", -1)) != SL_SIGMA:
        raise ValueError("Discovery report SL does not match locked H96 C2.")
    raw_cut = metadata.get("excluded_holdout_start_utc")
    if not raw_cut:
        raise ValueError("Discovery report is missing excluded_holdout_start_utc.")
    holdout_start = pd.Timestamp(raw_cut)
    if holdout_start.tzinfo is None:
        holdout_start = holdout_start.tz_localize("UTC")
    else:
        holdout_start = holdout_start.tz_convert("UTC")
    if holdout_start.tzinfo is None:
        raise ValueError("Could not normalize the holdout start to UTC.")
    digest = sha256_bytes(path.read_bytes())
    return holdout_start, digest


def _classes(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_long = np.where(
        labels["long_label"].to_numpy() == -1, 0,
        np.where(labels["long_label"].to_numpy() == 0, 1, 2),
    ).astype(np.int8)
    y_short = np.where(
        labels["short_label"].to_numpy() == -1, 0,
        np.where(labels["short_label"].to_numpy() == 0, 1, 2),
    ).astype(np.int8)
    return y_long, y_short


def _timestamp(value: str, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    else:
        result = result.tz_convert("UTC")
    if pd.isna(result):
        raise ValueError(f"{name} must be a valid timestamp.")
    return result


def _train_frozen_model(train_candles: pd.DataFrame) -> tuple[dict, dict]:
    labels_all = hunt.asymmetric_barrier_labels(
        train_candles,
        HORIZON_BARS,
        TP_LONG_SIGMA,
        SL_SIGMA,
        tp_mult_short=TP_SHORT_SIGMA,
        volatility_horizon=VOLATILITY_HORIZON_BARS,
    )
    X_all, feature_names = pv.make_features(train_candles)
    valid, warmup_rows = hunt._contiguous_label_valid_mask(
        labels_all, X_all, min_rows=BARS_PER_DAY * 455 + 100,
    )
    labels = labels_all.loc[valid].reset_index(drop=True)
    X = X_all.loc[valid].reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    if list(X.columns) != list(feature_names):
        raise ValueError("Training feature order unexpectedly changed.")

    times = train_candles.loc[valid, "candle_time"].reset_index(drop=True)
    if len(times) > 1 and not times.diff().dropna().eq(
        pd.Timedelta(minutes=15)
    ).all():
        raise ValueError("Training timeline is not contiguous after label validation.")

    y_long, y_short = _classes(labels)
    w_long = labels["long_sample_weight"].to_numpy(dtype=float)
    w_short = labels["short_sample_weight"].to_numpy(dtype=float)

    calibration_rows = BARS_PER_DAY * TRAIN_CALIBRATION_DAYS
    calibration_start = len(X) - calibration_rows
    if calibration_start <= BARS_PER_DAY * 365:
        raise ValueError(
            "Not enough pre-holdout data for at least 365 days of fit plus a 90-day calibration tail."
        )

    model_state = hunt._train_predict(
        X, y_long, y_short, w_long, w_short,
        0, calibration_start, len(X),
        horizon_bars=HORIZON_BARS,
    )
    model_long, classes_long, model_short, classes_short, threshold_long, threshold_short = model_state
    artifact = {
        "strategy_id": STRATEGY_ID,
        "model_long": model_long,
        "classes_long": classes_long,
        "model_short": model_short,
        "classes_short": classes_short,
        "threshold_long": float(threshold_long),
        "threshold_short": float(threshold_short),
        "feature_names": list(feature_names),
    }
    training_info = {
        "training_first_candle_utc": str(times.iloc[0]),
        "training_last_label_candle_utc": str(times.iloc[-1]),
        "training_valid_rows": int(len(X)),
        "feature_warmup_rows": int(warmup_rows),
        "calibration_days": TRAIN_CALIBRATION_DAYS,
        "calibration_rows": int(calibration_rows),
        "top_probability_pct_per_side": TOP_PCT * 100.0,
        "threshold_long": float(threshold_long),
        "threshold_short": float(threshold_short),
        "feature_count": int(len(feature_names)),
        "feature_names": list(feature_names),
    }
    return artifact, training_info


def _trade_rows(
    execution: dict,
    signal_times: pd.Series,
    primary_costs: dict[str, np.ndarray],
    stress_costs: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for j, idx in enumerate(execution["indices"]):
        held = int(execution["holding_bars"][j])
        side = int(execution["sides"][j])
        exit_reason = str(execution["exit_reasons"][j])
        rows.append({
            "signal_candle_close_utc": (
                pd.Timestamp(signal_times.iloc[int(idx)]) + pd.Timedelta(minutes=15)
            ).isoformat(),
            "entry_open_utc": pd.Timestamp(signal_times.iloc[int(idx) + 1]).isoformat(),
            "exit_candle_close_utc": (
                pd.Timestamp(signal_times.iloc[int(idx) + held]) + pd.Timedelta(minutes=15)
            ).isoformat(),
            "side": "LONG" if side == 1 else "SHORT",
            "exit_reason": exit_reason,
            "holding_bars": held,
            "gross_return_bps": round(float(execution["gross"][j]) * 10_000.0, 4),
            "funding_return_bps": round(float(execution["funding"][j]) * 10_000.0, 4),
            "primary_net_bps": round(float(primary_costs["net_return"][j]) * 10_000.0, 4),
            "all_taker_net_bps": round(float(stress_costs["net_return"][j]) * 10_000.0, 4),
            "entry_commission_bps": round(float(primary_costs["entry_commission_return"][j]) * 10_000.0, 4),
            "exit_commission_bps": round(float(primary_costs["exit_commission_return"][j]) * 10_000.0, 4),
            "slippage_bps": round(float(primary_costs["slippage_return"][j]) * 10_000.0, 4),
        })
    return rows


def _metric_bundle(
    net: np.ndarray,
    gross: np.ndarray,
    n_bars: int,
    key: str,
) -> dict:
    execution = {"gross": gross, key: net}
    metrics = hunt._metrics_with_ci(
        execution,
        np.asarray([], dtype=float),
        n_bars,
        key,
        bootstrap_alpha=0.05,
        bootstrap_reps=10_000,
    )
    metrics["bootstrap_method"] = "20-trade moving block bootstrap, 10,000 reps; nominal 95% CI"
    return metrics


def run(args: argparse.Namespace) -> dict:
    report_path = Path(args.discovery_report)
    holdout_start, report_sha = _validate_discovery_report(report_path)
    historical_end = _timestamp(args.historical_data_end_utc, "historical_data_end_utc")
    forward_start = _timestamp(args.forward_start_utc, "forward_start_utc")
    if forward_start <= historical_end:
        raise ValueError("Forward period must start strictly after the historical data cutoff.")
    if holdout_start >= historical_end:
        raise ValueError("Holdout start must precede the historical data cutoff.")
    if int(args.holdout_days) != 180:
        raise ValueError("This locked run requires the same 180-day untouched holdout.")

    all_candles = rv.load_btc(
        args.db_url,
        args.table,
    )
    now = pd.Timestamp.now(tz="UTC")
    latest_closed_open = now.floor("15min") - pd.Timedelta(minutes=15)
    all_candles = all_candles.loc[
        all_candles["candle_time"] <= latest_closed_open
    ].reset_index(drop=True)
    if all_candles.empty:
        raise ValueError("No completed BTCUSDT 15m candles are available.")

    latest_close_time = pd.Timestamp(all_candles["candle_time"].iloc[-1]) + pd.Timedelta(minutes=15)
    candle_age_hours = (now - latest_close_time).total_seconds() / 3600.0
    if candle_age_hours < -1e-9 or candle_age_hours > float(args.max_candle_age_hours):
        raise ValueError(
            f"BTC candle feed is stale: last completed close {latest_close_time.isoformat()}, "
            f"age={candle_age_hours:.2f}h; max allowed={args.max_candle_age_hours:.2f}h."
        )

    train_candles = all_candles.loc[
        all_candles["candle_time"] < holdout_start
    ].reset_index(drop=True)
    forward_candles = all_candles.loc[
        all_candles["candle_time"] >= forward_start
    ].reset_index(drop=True)
    if train_candles.empty:
        raise ValueError("No pre-holdout training data found.")
    if forward_candles.empty:
        raise ValueError("No post-cutoff forward candles found.")
    if pd.Timestamp(train_candles["candle_time"].iloc[-1]) >= holdout_start:
        raise ValueError("Training data crossed into the excluded holdout.")
    if pd.Timestamp(forward_candles["candle_time"].iloc[0]) < forward_start:
        raise ValueError("Forward scoring data starts before the locked start.")
    if len(forward_candles) < VOLATILITY_HORIZON_BARS + HORIZON_BARS + 10:
        raise ValueError(
            "Forward period needs at least 490 completed 15m candles for feature warm-up and closed outcomes."
        )

    train_hash = sha256_bytes(
        pd.util.hash_pandas_object(train_candles, index=False).to_numpy().tobytes()
    )
    artifact_path = Path(args.model_artifact)
    lock_path = Path(args.lock_file)
    if lock_path.exists() != artifact_path.exists():
        raise RuntimeError(
            "Model lock and model artifact must either both exist or both be absent. "
            "Do not overwrite an incomplete lock; inspect the files first."
        )

    if lock_path.exists():
        lock = _read_json(lock_path)
        expected = {
            "strategy_id": STRATEGY_ID,
            "discovery_report_sha256": report_sha,
            "training_data_sha256": train_hash,
            "holdout_start_utc": holdout_start.isoformat(),
            "historical_data_end_utc": historical_end.isoformat(),
            "forward_start_utc": forward_start.isoformat(),
        }
        for key, value in expected.items():
            if lock.get(key) != value:
                raise RuntimeError(
                    f"Existing model lock mismatch for {key}. "
                    "Do not reuse this lock with changed data/config; create a new test ID."
                )
        if sha256_bytes(artifact_path.read_bytes()) != lock.get("model_artifact_sha256"):
            raise RuntimeError("Frozen model artifact hash mismatch.")
        artifact = joblib.load(artifact_path)
        training_info = lock["training_info"]
        if artifact.get("strategy_id") != STRATEGY_ID:
            raise RuntimeError("Loaded model artifact has the wrong strategy ID.")
    else:
        artifact, training_info = _train_frozen_model(train_candles)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, artifact_path)
        model_hash = sha256_bytes(artifact_path.read_bytes())
        lock = {
            "strategy_id": STRATEGY_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "discovery_report_sha256": report_sha,
            "training_data_sha256": train_hash,
            "holdout_start_utc": holdout_start.isoformat(),
            "historical_data_end_utc": historical_end.isoformat(),
            "forward_start_utc": forward_start.isoformat(),
            "parameters": {
                "tp_long_sigma": TP_LONG_SIGMA,
                "tp_short_sigma": TP_SHORT_SIGMA,
                "sl_sigma": SL_SIGMA,
                "time_exit_bars": HORIZON_BARS,
                "volatility_horizon_bars": VOLATILITY_HORIZON_BARS,
                "entry_cooldown_bars": ENTRY_COOLDOWN_BARS,
                "maker_fee_bps_per_side": MAKER_FEE_BPS,
                "taker_fee_bps_per_side": TAKER_FEE_BPS,
                "round_trip_slippage_bps": ROUND_TRIP_SLIPPAGE_BPS,
                "bnb_discount_pct": BNB_DISCOUNT_PCT,
                "position_cap_usd": POSITION_CAP_USD,
                "initial_capital_usd": INITIAL_CAPITAL_USD,
                "cash_reserve_pct": CASH_RESERVE_PCT,
            },
            "training_info": training_info,
            "model_artifact_sha256": model_hash,
            "paper_only": True,
            "live_orders": False,
        }
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            json.dumps(lock, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    X_forward_all, forward_feature_names = pv.make_features(forward_candles)
    if list(forward_feature_names) != list(artifact["feature_names"]):
        raise ValueError("Forward feature schema differs from the frozen training schema.")
    labels_forward_all = hunt.asymmetric_barrier_labels(
        forward_candles,
        HORIZON_BARS,
        TP_LONG_SIGMA,
        SL_SIGMA,
        tp_mult_short=TP_SHORT_SIGMA,
        volatility_horizon=VOLATILITY_HORIZON_BARS,
    )
    valid_forward, forward_warmup_rows = hunt._contiguous_label_valid_mask(
        labels_forward_all, X_forward_all, min_rows=1,
    )
    if not valid_forward.any():
        raise ValueError("No fully warmed-up and closed forward outcomes yet.")
    # Keep the final H invalid-label rows in the aligned arrays. The executor
    # already stops H rows before the end, so truncating here would discard a
    # second H-bar window and misalign the intended 96-bar signal horizon.
    first_ready = int(forward_warmup_rows)
    labels_forward = labels_forward_all.iloc[first_ready:].reset_index(drop=True)
    X_forward = X_forward_all.iloc[first_ready:].reset_index(drop=True).replace(
        [np.inf, -np.inf], np.nan
    )
    signal_times = forward_candles.iloc[first_ready:]["candle_time"].reset_index(drop=True)
    if not labels_forward["valid"].iloc[:-HORIZON_BARS].all():
        raise ValueError("Forward label timeline has an internal invalid row after warm-up.")

    p_long = hunt.predict_side_proba(
        artifact["model_long"], artifact["classes_long"], X_forward
    )[:, 2]
    p_short = hunt.predict_side_proba(
        artifact["model_short"], artifact["classes_short"], X_forward
    )[:, 2]

    funding_all = hunt.load_funding(Path(args.funding_csv))
    first_completed = pd.Timestamp(forward_candles["candle_time"].iloc[0]) + pd.Timedelta(minutes=15)
    last_completed = pd.Timestamp(forward_candles["candle_time"].iloc[-1]) + pd.Timedelta(minutes=15)
    funding_forward = funding_all.loc[
        (funding_all["funding_time"] >= first_completed)
        & (funding_all["funding_time"] <= last_completed)
    ].reset_index(drop=True)
    mapped_funding = hunt.funding_mark_price_proxy(forward_candles, funding_forward)
    if mapped_funding.empty:
        raise ValueError("No mappable funding settlements in the forward window.")
    funding_age = (
        last_completed - pd.Timestamp(mapped_funding["funding_time"].iloc[-1])
    ).total_seconds() / 3600.0
    if funding_age > FUNDING_MAX_AGE_HOURS:
        raise ValueError(
            f"Funding file is stale by {funding_age:.2f}h; max allowed={FUNDING_MAX_AGE_HOURS:.2f}h."
        )
    funding_times_ns = mapped_funding["funding_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    funding_rates = mapped_funding["funding_rate"].to_numpy(dtype=float)
    funding_marks = mapped_funding["mark_price_proxy"].to_numpy(dtype=float)

    execution = hunt._execute_signals(
        p_long,
        p_short,
        float(artifact["threshold_long"]),
        float(artifact["threshold_short"]),
        labels_forward,
        signal_times,
        funding_times_ns,
        funding_rates,
        funding_marks,
        SL_SIGMA,
        horizon_bars=HORIZON_BARS,
        entry_cooldown_bars=ENTRY_COOLDOWN_BARS,
    )

    scenarios = build_execution_scenarios(
        execution["gross"],
        execution["sides"],
        execution["exit_reasons"],
        execution["funding"],
        maker_fee_bps=MAKER_FEE_BPS,
        taker_fee_bps=TAKER_FEE_BPS,
        round_trip_slippage_bps=ROUND_TRIP_SLIPPAGE_BPS,
        bnb_discount_pct=BNB_DISCOUNT_PCT,
    )
    primary_costs = scenarios["taker_entry_tp_maker_stop_time_taker"]
    all_taker_costs = scenarios["taker_entry_taker_exit"]
    n_forward_bars = max(1, int(len(signal_times) - HORIZON_BARS))

    primary_metrics = _metric_bundle(
        primary_costs["net_return"],
        execution["gross"],
        n_forward_bars,
        "net_primary",
    )
    all_taker_metrics = _metric_bundle(
        all_taker_costs["net_return"],
        execution["gross"],
        n_forward_bars,
        "net_all_taker",
    )
    if len(execution["indices"]):
        primary_metrics["trade_attribution"] = summarize_trade_attribution(
            primary_costs["net_return"],
            execution["sides"],
            execution["exit_reasons"],
            execution["mae_returns"],
            execution["mfe_returns"],
            execution["holding_bars"],
        )
        trades = _trade_rows(
            execution, signal_times, primary_costs, all_taker_costs,
        )
    else:
        primary_metrics["trade_attribution"] = None
        trades = []

    latest_close = pd.Timestamp(forward_candles["candle_time"].iloc[-1]) + pd.Timedelta(minutes=15)
    forward_days = max(
        0.0,
        (latest_close - forward_start).total_seconds() / 86_400.0,
    )
    cap50 = next(
        (x for x in primary_metrics.get("capital_paths", []) if x["position_cap_usd"] == POSITION_CAP_USD),
        None,
    )
    stress_cap50 = next(
        (x for x in all_taker_metrics.get("capital_paths", []) if x["position_cap_usd"] == POSITION_CAP_USD),
        None,
    )
    ci_primary = primary_metrics.get("bootstrap_ci95_expectancy_bps", [None, None])
    ci_stress = all_taker_metrics.get("bootstrap_ci95_expectancy_bps", [None, None])
    enough_sample = (
        forward_days >= MIN_FORWARD_DAYS
        and int(primary_metrics.get("trades", 0)) >= MIN_CLOSED_TRADES
    )
    checks = {
        "forward_days_ge_180": forward_days >= MIN_FORWARD_DAYS,
        "completed_trades_ge_100": int(primary_metrics.get("trades", 0)) >= MIN_CLOSED_TRADES,
        "primary_win_rate_ge_60pct": (
            primary_metrics.get("win_rate_pct") is not None
            and primary_metrics["win_rate_pct"] >= 60.0
        ),
        "primary_expectancy_positive": (
            primary_metrics.get("expectancy_bps") is not None
            and primary_metrics["expectancy_bps"] > 0
        ),
        "primary_profit_factor_ge_1_25": (
            primary_metrics.get("profit_factor") is not None
            and primary_metrics["profit_factor"] >= 1.25
        ),
        "all_taker_expectancy_positive": (
            all_taker_metrics.get("expectancy_bps") is not None
            and all_taker_metrics["expectancy_bps"] > 0
        ),
        "primary_bootstrap_lower_gt_zero": (
            ci_primary[0] is not None and ci_primary[0] > 0
        ),
        "all_taker_bootstrap_lower_gt_zero": (
            ci_stress[0] is not None and ci_stress[0] > 0
        ),
        "position_cap_50_drawdown_lt_15pct": (
            cap50 is not None and cap50["max_drawdown_pct"] < 15.0
        ),
        "all_taker_cap_50_drawdown_lt_15pct": (
            stress_cap50 is not None and stress_cap50["max_drawdown_pct"] < 15.0
        ),
    }
    if not enough_sample:
        verdict = "INCOMPLETE_FORWARD_SAMPLE"
    elif all(checks.values()):
        verdict = "FORWARD_PAPER_PASS_CANDIDATE"
    else:
        verdict = "FORWARD_PAPER_NO_GO"

    output = {
        "metadata": {
            "strategy_id": STRATEGY_ID,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "paper_only": True,
            "live_orders": False,
            "historical_discovery_report": str(report_path),
            "historical_discovery_report_sha256": report_sha,
            "historical_data_end_utc": historical_end.isoformat(),
            "excluded_holdout_start_utc": holdout_start.isoformat(),
            "excluded_holdout_days": 180,
            "forward_start_utc": forward_start.isoformat(),
            "forward_first_candle_utc": str(forward_candles["candle_time"].iloc[0]),
            "forward_last_completed_close_utc": latest_close.isoformat(),
            "forward_days": round(forward_days, 3),
            "forward_rows": int(len(forward_candles)),
            "forward_feature_warmup_rows": int(forward_warmup_rows),
            "latest_candle_age_hours": round(candle_age_hours, 4),
            "funding_latest_age_hours": round(funding_age, 4),
            "training_data_sha256": train_hash,
            "model_artifact": str(artifact_path),
            "model_lock_file": str(lock_path),
            "model_artifact_sha256": lock["model_artifact_sha256"],
            "thresholds_locked": {
                "long": float(artifact["threshold_long"]),
                "short": float(artifact["threshold_short"]),
            },
            "parameters": lock["parameters"],
            "execution_assumptions": {
                "primary": "Taker entry; TP Maker conditional on passive limit fill; SL/TIME Taker",
                "all_taker": "Taker entry and Taker exit for every exit reason",
                "slippage": "2 bps round trip",
                "funding_mark_price": "last completed BTC 15m close proxy; not exchange mark price",
                "maker_fill_warning": "15m OHLC does not prove passive queue execution or Maker fill",
            },
            "forward_readiness_policy": {
                "minimum_days": MIN_FORWARD_DAYS,
                "minimum_completed_trades": MIN_CLOSED_TRADES,
                "retrain_during_forward_test": False,
                "threshold_changes": "forbidden",
                "holdout_usage": "forbidden",
            },
        },
        "training": training_info,
        "forward_primary": primary_metrics,
        "forward_all_taker": all_taker_metrics,
        "checks": checks,
        "checks_passed": int(sum(bool(v) for v in checks.values())),
        "checks_total": len(checks),
        "verdict": verdict,
        "trades": trades,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Order-free, frozen H96 C2 forward paper evaluation."
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
    parser.add_argument("--discovery-report", default="development_time_exit_sweep_1d.json")
    parser.add_argument("--historical-data-end-utc", default="2026-10-31T23:45:00Z")
    parser.add_argument("--forward-start-utc", default="2026-11-01T00:00:00Z")
    parser.add_argument("--holdout-days", type=int, default=180)
    parser.add_argument("--max-candle-age-hours", type=float, default=MAX_CANDLE_AGE_HOURS)
    parser.add_argument("--model-artifact", default="forward_paper_h96_c2_model.joblib")
    parser.add_argument("--lock-file", default="forward_paper_h96_c2_lock.json")
    parser.add_argument("--out", default="forward_paper_h96_c2.json")
    args = parser.parse_args()

    if args.max_candle_age_hours <= 0:
        raise SystemExit("--max-candle-age-hours must be positive.")
    try:
        result = run(args)
    except Exception as exc:
        raise SystemExit(f"[FAIL-CLOSED] {type(exc).__name__}: {exc}") from exc

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    primary = result["forward_primary"]
    stress = result["forward_all_taker"]
    print(f"\n[report] {output.resolve()}")
    print(f"[model] {result['metadata']['model_artifact']}")
    print(f"[lock]  {result['metadata']['model_lock_file']}")
    print(f"[forward] days={result['metadata']['forward_days']} "
          f"trades={primary.get('trades')} "
          f"WR={primary.get('win_rate_pct')}% "
          f"EXP={primary.get('expectancy_bps')}bps "
          f"PF={primary.get('profit_factor')} "
          f"ALL_TAKER_EXP={stress.get('expectancy_bps')}bps "
          f"verdict={result['verdict']}")
    print("No exchange orders were sent. The historical 180-day holdout was excluded.")


if __name__ == "__main__":
    main()
