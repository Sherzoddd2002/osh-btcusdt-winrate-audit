from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

import research_v3 as rv
import forward_paper_h96_c2 as frozen


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    report_path = Path(args.discovery_report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    holdout_start, report_hash = frozen._validate_discovery_report(report_path)
    historical_end = frozen._timestamp(args.historical_data_end_utc, "historical_data_end_utc")
    forward_start = frozen._timestamp(args.forward_start_utc, "forward_start_utc")
    if holdout_start >= historical_end or forward_start <= historical_end:
        raise ValueError("Historical cutoff, holdout, and forward start are inconsistent.")

    model_path, lock_path = Path(args.model_artifact), Path(args.lock_file)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not lock_path.is_absolute():
        lock_path = ROOT / lock_path
    if model_path.exists() != lock_path.exists():
        raise RuntimeError("Model and lock must both exist or both be absent; inspect before repair.")
    if model_path.exists():
        lock = frozen._read_json(lock_path)
        if lock.get("model_artifact_sha256") != digest(model_path):
            raise RuntimeError("Existing frozen model hash does not match its lock.")
        print(json.dumps({"status": "REUSED_LOCKED_MODEL", "thresholds": lock["thresholds_locked"]}, indent=2))
        return

    db_url = args.db_url or os.getenv(
        "OSH_DB_URL", "postgresql+psycopg2://postgres@127.0.0.1:5433/osh"
    )
    candles = rv.load_btc(db_url, args.table)
    train = candles.loc[candles["candle_time"] < holdout_start].reset_index(drop=True)
    if train.empty or train["candle_time"].iloc[-1] >= holdout_start:
        raise ValueError("Training set is empty or crosses the excluded 180-day holdout.")
    if train["candle_time"].iloc[-1] > historical_end:
        raise ValueError("Training data extends beyond the declared historical cutoff.")

    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(train, index=False).to_numpy().tobytes()
    ).hexdigest()
    artifact, training = frozen._train_frozen_model(train)

    params = {
        "tp_long_sigma": 0.75, "tp_short_sigma": 0.75, "sl_sigma": 1.0,
        "time_exit_bars": 96, "volatility_horizon_bars": 96,
        "entry_cooldown_bars": 96, "position_cap_usd": 50,
        "leverage": 1, "margin_type": "ISOLATED",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    temp_model = model_path.with_suffix(model_path.suffix + ".tmp")
    temp_lock = lock_path.with_suffix(lock_path.suffix + ".tmp")
    joblib.dump(artifact, temp_model)
    model_hash = digest(temp_model)
    lock = {
        "strategy_id": frozen.STRATEGY_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "discovery_report_sha256": report_hash,
        "training_data_sha256": data_hash,
        "holdout_start_utc": holdout_start.isoformat(),
        "historical_data_end_utc": historical_end.isoformat(),
        "forward_start_utc": forward_start.isoformat(),
        "parameters": params,
        "training_info": training,
        "thresholds_locked": {
            "long": float(artifact["threshold_long"]),
            "short": float(artifact["threshold_short"]),
        },
        "model_artifact_sha256": model_hash,
        "paper_only": True,
        "live_orders": False,
        "demo_execution_allowed": True,
        "demo_base_url": "https://demo-fapi.binance.com",
    }
    temp_lock.write_text(json.dumps(lock, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp_model.replace(model_path)
    temp_lock.replace(lock_path)
    print(json.dumps({
        "status": "MODEL_FROZEN",
        "model_sha256": model_hash,
        "thresholds_locked": lock["thresholds_locked"],
        "training_first": training["training_first_candle_utc"],
        "training_last": training["training_last_label_candle_utc"],
        "holdout_start": holdout_start.isoformat(),
        "warning": "Holdout is excluded. Model artifact and lock are local only.",
    }, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Freeze H96 C2 using only pre-holdout candles.")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--table", default="public.btcusdt_15m")
    parser.add_argument("--discovery-report", default="development_time_exit_sweep_1d.json")
    parser.add_argument("--historical-data-end-utc", default="2026-10-31T23:45:00Z")
    parser.add_argument("--forward-start-utc", default="2026-11-01T00:00:00Z")
    parser.add_argument("--model-artifact", default="forward_paper_h96_c2_model.joblib")
    parser.add_argument("--lock-file", default="forward_paper_h96_c2_lock.json")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
