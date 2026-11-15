"""HF Arena engine — local-only orchestration, paper-first.

Modes:
  replay : historical candles from the local PostgreSQL DB, zero orders
  demo   : live 15m candles from the pinned Futures Testnet host; real orders
           only when explicitly armed by the user

Credentials are stored ONLY in Windows Credential Manager under a separate
service name. They are never written to the repo, browser storage, or logs.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .arena import Arena, FrozenSignals
from .client import DemoClient
from .config import (
    AUDIT_PATH, CRED_USERS, DEMO_FUTURES_URL, INITIAL_CAPITAL_USD,
    KEYRING_SERVICE, LEDGER_PATH, LOCAL_PORT, MAX_NOTIONAL_USD, MODEL_ARTIFACT,
    MODELS, ROUND_TRIP_TAKER_BPS, SYMBOL,
)
from .executor import DemoExecutor

DB_URL = os.getenv("OSH_DB_URL", "postgresql+psycopg2://postgres@127.0.0.1:5433/osh")
TABLE = "public.btcusdt_15m"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArenaEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.arena = Arena()
        self.executor = DemoExecutor(self.audit)
        self.mode = "idle"                 # idle | replay | demo
        self.running = False
        self.thread: threading.Thread | None = None
        self.last_bar_time: str = ""
        self.last_error = ""
        self.api_error_count = 0
        self.connect_report: dict | None = None
        self.replay_report: dict | None = None
        self.started_at = ""
        self._seen_bars: set[str] = set()
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- audit
    def audit(self, event: str, payload: dict | None = None) -> None:
        row = {"time": _now(), "event": event, "mode": self.mode}
        if payload:
            row.update(payload)
        safe = {k: v for k, v in row.items()
                if k not in {"api_key", "api_secret", "secret", "signature"}}
        try:
            with AUDIT_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(safe, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------- model
    def _load_signals(self) -> FrozenSignals | None:
        if not MODEL_ARTIFACT.exists():
            self.last_error = f"Frozen model artifact not found: {MODEL_ARTIFACT.name}"
            return None
        try:
            return FrozenSignals(MODEL_ARTIFACT, [])
        except Exception as exc:
            self.last_error = f"Frozen model load failed: {type(exc).__name__}: {exc}"
            return None

    # ------------------------------------------------------------- credentials
    @staticmethod
    def _keyring():
        import keyring
        backend = keyring.get_keyring()
        name = type(backend).__name__
        if "Fail" in name or "None" in name:
            raise RuntimeError("Windows Credential Manager is unavailable on this PC.")
        return keyring

    def save_credentials(self, api_key: str, api_secret: str) -> dict:
        key = (api_key or "").strip()
        secret = (api_secret or "").strip()
        if len(key) < 12 or len(secret) < 12:
            raise ValueError("Key/secret look too short. Use Futures TESTNET keys only.")
        keyring = self._keyring()
        keyring.set_password(KEYRING_SERVICE, CRED_USERS[0], key)
        keyring.set_password(KEYRING_SERVICE, CRED_USERS[1], secret)
        if keyring.get_password(KEYRING_SERVICE, CRED_USERS[0]) != key:
            raise RuntimeError("Credential verification failed; nothing was saved.")
        self.audit("CREDENTIALS_SAVED", {"service": KEYRING_SERVICE,
                                         "storage": "Windows Credential Manager"})
        return {"saved": True, "storage": "Windows Credential Manager (local PC only)",
                "note": "Keys never leave this PC; not in GitHub, not in browser storage."}

    def reset_credentials(self) -> dict:
        try:
            keyring = self._keyring()
            for user in CRED_USERS:
                try:
                    keyring.delete_password(KEYRING_SERVICE, user)
                except Exception:
                    pass
        except Exception:
            pass
        self.executor.client = None
        self.connect_report = None
        self.audit("CREDENTIALS_RESET", {})
        return {"reset": True}

    def _credentials(self) -> tuple[str, str]:
        keyring = self._keyring()
        key = keyring.get_password(KEYRING_SERVICE, CRED_USERS[0])
        secret = keyring.get_password(KEYRING_SERVICE, CRED_USERS[1])
        if not key or not secret:
            raise RuntimeError("No Testnet API keys saved on this PC. Enter them in the site first.")
        return key, secret

    def has_credentials(self) -> bool:
        try:
            key, secret = self._credentials()
            return bool(key and secret)
        except Exception:
            return False

    # ------------------------------------------------------------- connect
    def connect(self) -> dict:
        key, secret = self._credentials()
        client = DemoClient(key, secret)
        try:
            report = self.executor.connect(client)
        except Exception:
            client.close()
            raise
        self.connect_report = report
        self.last_error = ""
        return report

    # ------------------------------------------------------------- candles
    @staticmethod
    def _candles_from_db(limit_bars: int = 20000) -> pd.DataFrame:
        """Tail-only SQL read: avoids loading all 240k candles on every replay."""
        import research_v3 as rv  # read-only research helper, not modified
        from sqlalchemy import create_engine

        engine = create_engine(DB_URL, pool_pre_ping=True)
        sql = f"""
            SELECT to_timestamp(open_time / 1000.0) AS candle_time,
                   open, high, low, close, volume, trade_count, quote_volume,
                   taker_buy_base_volume, taker_buy_quote_volume
            FROM {TABLE}
            ORDER BY open_time DESC
            LIMIT {int(limit_bars)}
        """
        try:
            with engine.connect() as conn:
                raw = pd.read_sql_query(sql, conn)
        finally:
            engine.dispose()
        raw = raw.iloc[::-1].reset_index(drop=True)
        return rv.validate_market_data(raw)

    @staticmethod
    def _candles_from_klines(raw: list) -> pd.DataFrame:
        rows = []
        for k in raw:
            rows.append({
                "candle_time": pd.to_datetime(int(k[0]), unit="ms", utc=True),
                "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                "close": float(k[4]), "volume": float(k[5]),
                "quote_volume": float(k[7]), "trade_count": float(k[8]),
                "taker_buy_base_volume": float(k[9]),
                "taker_buy_quote_volume": float(k[10]),
            })
        return pd.DataFrame(rows).sort_values("candle_time").reset_index(drop=True)

    # ------------------------------------------------------------- replay
    def run_replay(self, days: int = 90) -> dict:
        """Historical paper replay. Zero orders, zero keys, zero network."""
        with self.lock:
            if self.running:
                raise RuntimeError("Engine is already running.")
            self.mode = "replay"
            self.running = True
            self.started_at = _now()
        try:
            signals = self._load_signals()
            df = self._candles_from_db()
            bars = int(days * 96)
            df = df.tail(bars + 600).reset_index(drop=True)
            arena = Arena(signals=signals)
            arena.set_candles(df)
            arena.run_range(600, len(df) - 2)
            self.arena = arena
            report = arena.report()
            report["mode"] = "replay"
            report["data_source"] = f"local PostgreSQL {TABLE} (last {days} days)"
            report["orders_sent"] = 0
            report["note"] = ("Historical PAPER replay for model comparison. Not a forecast, "
                              "not a live result. Fees included in net.")
            self.replay_report = report
            arena.save(LEDGER_PATH)
            self.audit("REPLAY_DONE", {"days": days, "bars": len(df),
                                       "models": len(MODELS)})
            self.last_error = ""
            return report
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.audit("REPLAY_ERROR", {"error": self.last_error})
            raise
        finally:
            with self.lock:
                self.running = False
                self.mode = "idle"

    # ------------------------------------------------------------- demo loop
    def start_demo(self, champion: str, confirm_orders: bool) -> dict:
        if champion not in {m["id"] for m in MODELS}:
            raise ValueError("Unknown champion model.")
        if self.running:
            raise RuntimeError("Engine is already running.")
        if not self.has_credentials():
            raise RuntimeError("Save Testnet API keys first.")
        report = self.connect()
        if not report.get("preflight_ok"):
            raise RuntimeError("Preflight blocked: " + "; ".join(report.get("blockers", [])))
        signals = self._load_signals()
        client = self.executor.client
        raw = client.klines(SYMBOL, "15m", 500)
        df = self._candles_from_klines(raw)
        # Only CLOSED candles: drop the current, still-forming bar.
        df = df.iloc[:-1].reset_index(drop=True)
        self.arena = Arena(signals=signals)
        self.arena.set_candles(df)
        self._seen_bars = {str(t) for t in df["candle_time"].tail(20)}
        self.executor.champion = champion
        if confirm_orders:
            self.executor.arm(champion, True)
        self.audit("DEMO_STARTED", {"champion": champion,
                                    "orders_enabled": bool(confirm_orders),
                                    "host": DEMO_FUTURES_URL})
        with self.lock:
            self.mode = "demo"
            self.running = True
            self.started_at = _now()
        self.thread = threading.Thread(target=self._loop, daemon=True,
                                       name="osh-hf-arena")
        self.thread.start()
        self.last_error = ""
        return {"started": True, "mode": "demo", "champion": champion,
                "orders_enabled": bool(confirm_orders),
                "note": "Paper leaderboard for all models; real orders only for champion."}

    def _loop(self) -> None:
        client = self.executor.client
        while True:
            with self.lock:
                if not self.running:
                    break
            try:
                time.sleep(20)
                raw = client.klines(SYMBOL, "15m", 500)
                df = self._candles_from_klines(raw).iloc[:-1].reset_index(drop=True)
                latest = str(df["candle_time"].iloc[-1])
                if latest in self._seen_bars:
                    continue
                self._seen_bars.add(latest)
                self._seen_bars = set(list(self._seen_bars)[-40:])
                self.arena.set_candles(df)
                self.arena.step(len(df) - 2)
                self.last_bar_time = latest
                self.api_error_count = 0
                # Champion signal -> optional real Testnet order
                events = self.arena.events[-30:]
                for ev in reversed(events):
                    if ev.get("event") == "SIGNAL" and ev.get("model") == self.executor.champion:
                        if ev.get("time", "") >= self.started_at:
                            led = self.arena.ledgers[self.executor.champion]
                            if led.open_positions:
                                p = led.open_positions[-1]
                                self.executor.maybe_send({
                                    "model_id": self.executor.champion,
                                    "side": p.side, "notional_usd": p.notional_usd,
                                    "tp_price": p.tp_price, "sl_price": p.sl_price,
                                    "confidence": p.confidence,
                                })
                        break
                self.arena.save(LEDGER_PATH)
            except Exception as exc:
                self.api_error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.audit("LOOP_ERROR", {"error": self.last_error,
                                          "consecutive": self.api_error_count})
                if self.api_error_count >= 3:
                    self.executor.disarm("3 consecutive loop errors")
                    with self.lock:
                        self.running = False
                    break
        self.audit("LOOP_STOPPED", {})

    def stop(self) -> dict:
        with self.lock:
            self.running = False
        self.executor.disarm("user stop")
        self.audit("ENGINE_STOPPED", {})
        return {"running": False, "orders_enabled": False}

    def kill(self) -> dict:
        with self.lock:
            self.running = False
        out = self.executor.kill()
        out["running"] = False
        return out

    # ------------------------------------------------------------- status
    def status(self) -> dict:
        lb = self.arena.leaderboard()
        return {
            "engine": "OSH HF Model Arena",
            "version": "2026-11-15-v1",
            "mode": self.mode,
            "running": self.running,
            "started_at_utc": self.started_at,
            "last_bar_utc": self.last_bar_time,
            "last_error": self.last_error,
            "host": DEMO_FUTURES_URL,
            "symbol": SYMBOL,
            "initial_capital_usd": INITIAL_CAPITAL_USD,
            "max_notional_usd": MAX_NOTIONAL_USD,
            "leverage": 1,
            "fee_bps_round_trip": ROUND_TRIP_TAKER_BPS,
            "fees_removed": False,
            "fee_note": "Komissiya hisobdan olib tashlanmagan. Gross faqat diagnostika.",
            "credentials_saved": self.has_credentials(),
            "credentials_storage": "Windows Credential Manager (local PC)",
            "connected": self.connect_report is not None,
            "preflight": self.connect_report,
            "orders_enabled": self.executor.enabled,
            "champion": self.executor.champion,
            "last_order_action": self.executor.last_action,
            "api_error_count": self.api_error_count,
            "models": len(MODELS),
            "leaderboard": lb,
            "bars_seen": self.arena.bars_seen if hasattr(self.arena, "bars_seen") else len(self.arena.bars),
            "replay_available": self.replay_report is not None,
            "paper_only_note": "Faqat Testnet / demo. Hech qanday real kapital xavf ostida emas.",
        }
