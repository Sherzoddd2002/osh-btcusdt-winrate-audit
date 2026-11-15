from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pipeline_v2 as pv
import high_winrate_hunt_1d as hunt
from demo_executor.client import BinanceDemoClient, BinanceAPIError
from demo_executor.logic import (
    INTERVAL_MS, barrier_prices, candles_from_binance_klines, choose_side,
    decimal_floor_step, format_decimal, quantity_for_cap, round_price_to_tick,
    sigma_from_history, symbol_rules,
)

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_ID = "OSH-BTCUSDT-15M-H96-C2-FORWARD-PAPER-v1"
SYMBOL = os.getenv("OSH_SYMBOL", "BTCUSDT").upper()
POSITION_CAP = float(os.getenv("OSH_POSITION_CAP_USD", "50"))
MAX_DRAWDOWN_PCT = float(os.getenv("OSH_MAX_DRAWDOWN_PCT", "15"))
if not 0 < POSITION_CAP <= 50:
    raise ValueError("OSH_POSITION_CAP_USD must be > 0 and no greater than the locked $50 cap.")
if not 0 < MAX_DRAWDOWN_PCT <= 15:
    raise ValueError("OSH_MAX_DRAWDOWN_PCT must be > 0 and no greater than 15%.")
MODEL_PATH = Path(os.getenv("OSH_MODEL_ARTIFACT", "forward_paper_h96_c2_model.joblib"))
LOCK_PATH = Path(os.getenv("OSH_MODEL_LOCK", "forward_paper_h96_c2_lock.json"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH
if not LOCK_PATH.is_absolute():
    LOCK_PATH = ROOT / LOCK_PATH
RUNTIME = ROOT / "demo_executor" / "runtime"
STATE_FILE = RUNTIME / "state.json"
EVENT_FILE = RUNTIME / "events.jsonl"


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def random_client_id(prefix):
    return (prefix + uuid.uuid4().hex[:24]).upper()[:36]


def safe_float(value, default=0.0):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


class DemoEngine:
    def __init__(self):
        # Generic BINANCE_API_KEY variables are deliberately ignored: they may be Live keys.
        key = os.getenv("BINANCE_DEMO_API_KEY") or os.getenv("BINANCE_FUTURES_API_KEY")
        secret = os.getenv("BINANCE_DEMO_API_SECRET") or os.getenv("BINANCE_FUTURES_API_SECRET")
        stored = self._read_keyring_credentials()
        if stored:
            key, secret = stored
        self.credentials_configured = bool(key and secret)
        self.client = BinanceDemoClient(key, secret) if self.credentials_configured else None
        self.credentials_source = "windows_credential_manager" if stored else ("demo_environment" if key and secret else None)
        self.orders_enabled = os.getenv("OSH_ENABLE_DEMO_ORDERS", "false").lower() == "true"
        self.lock = threading.RLock()
        self.accept_entries = False
        self.state_name = "STOPPED"
        self.last_error = None
        self.last_signal = None
        self.last_candle = None
        self.account_state = None
        self.multi_assets_mode = None
        self.rules = None
        self.model = None
        self.model_sha = None
        self.thread = None
        self.halt = threading.Event()
        self.events = []
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.state = self._read_state()

    @staticmethod
    def _keyring_module():
        try:
            import keyring
        except ImportError as exc:
            raise RuntimeError("Secure credential storage is missing. Install requirements.txt first.") from exc
        backend = keyring.get_keyring()
        if getattr(backend, "priority", 0) <= 0:
            raise RuntimeError("Windows Credential Manager backend is unavailable; credentials were not saved.")
        return keyring

    @classmethod
    def _read_keyring_credentials(cls):
        try:
            keyring = cls._keyring_module()
            key = keyring.get_password("OSH_Quant_Lab_Binance_Demo", "api_key")
            secret = keyring.get_password("OSH_Quant_Lab_Binance_Demo", "api_secret")
            return (key, secret) if key and secret else None
        except Exception:
            return None

    def save_credentials(self, api_key, api_secret):
        key = str(api_key or "").strip()
        secret = str(api_secret or "").strip()
        if not key or not secret or len(key) > 256 or len(secret) > 512:
            raise ValueError("Both Futures Testnet API fields are required and must be within their size limits.")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("Stop the engine before replacing API credentials.")
            if self.state.get("trade"):
                raise RuntimeError("A saved Testnet position is still being tracked. Do not replace its credentials.")
            keyring = self._keyring_module()
            client = BinanceDemoClient(key, secret)
            try:
                keyring.set_password("OSH_Quant_Lab_Binance_Demo", "api_key", key)
                keyring.set_password("OSH_Quant_Lab_Binance_Demo", "api_secret", secret)
                if (keyring.get_password("OSH_Quant_Lab_Binance_Demo", "api_key") != key or
                    keyring.get_password("OSH_Quant_Lab_Binance_Demo", "api_secret") != secret):
                    raise RuntimeError("Windows Credential Manager read-back verification failed.")
            except Exception:
                for username in ("api_key", "api_secret"):
                    try:
                        keyring.delete_password("OSH_Quant_Lab_Binance_Demo", username)
                    except Exception:
                        pass
                client.close()
                raise
            old = self.client
            self.client = client
            self.credentials_configured = True
            self.credentials_source = "windows_credential_manager"
            self.orders_enabled = False
            self.account_state = None
            self.rules = None
            self.last_error = None
            if old:
                old.close()
            return {"saved": True, "storage": "Windows Credential Manager", "credentials_configured": True}

    def reset_credentials(self):
        # Reset is fail-closed: it never deletes credentials while a position or order exists.
        with self.lock:
            self.accept_entries = False
            if self.state.get("trade"):
                self.state_name = "MANAGING_OPEN_DEMO_POSITION"
                raise RuntimeError("A bot-tracked position exists. New entries are disabled, but credentials are retained so TP/SL management can continue.")
            client = self.client
            if client:
                client.sync_time()
                amount = self.position_amount(client.positions(SYMBOL))
                orders = client.open_orders(SYMBOL)
                algo_orders = client.open_algo_orders(SYMBOL)
                if abs(amount) > 1e-12 or orders or algo_orders:
                    self.state_name = "STOPPED"
                    raise RuntimeError("Testnet BTCUSDT position or open orders exist. Nothing was deleted. Reconcile the Futures Testnet account first.")
            self.halt.set()
            thread = self.thread
        if thread and thread.is_alive():
            thread.join(timeout=8)
            if thread.is_alive():
                raise RuntimeError("Engine has not stopped yet. Credentials were retained; retry Reset shortly.")
        with self.lock:
            keyring = self._keyring_module()
            for username in ("api_key", "api_secret"):
                try:
                    keyring.delete_password("OSH_Quant_Lab_Binance_Demo", username)
                except keyring.errors.PasswordDeleteError:
                    pass
            if self.client:
                self.client.close()
            self.client = None
            self.credentials_configured = False
            self.credentials_source = None
            self.orders_enabled = False
            self.accept_entries = False
            self.account_state = None
            self.rules = None
            self.state_name = "STOPPED"
            self.last_error = None
            self.halt.clear()
            return {"reset": True, "credentials_configured": False, "note": "Model, cooldown state and audit log were preserved."}

    def _read_state(self):
        if not STATE_FILE.exists():
            return {"last_entry_candle_ms": None, "peak_equity": None, "trade": None}
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError("Local Testnet state is not a JSON object.")
        data.setdefault("last_entry_candle_ms", None)
        data.setdefault("peak_equity", None)
        data.setdefault("trade", None)
        return data

    def _save(self):
        temp = STATE_FILE.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temp.replace(STATE_FILE)

    def _event(self, name, **fields):
        row = {"time_utc": now_utc(), "event": name, **fields}
        self.events.append(row)
        self.events = self.events[-40:]
        with EVENT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def _client(self):
        if self.client is None:
            raise RuntimeError("No Futures Testnet API credentials saved. Enter Futures Testnet API credentials on the GitHub Pages terminal first.")
        return self.client

    def _load_model(self):
        if not MODEL_PATH.exists() or not LOCK_PATH.exists():
            raise RuntimeError("Frozen model missing. Run python -m demo_executor.prepare_model first.")
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        if lock.get("strategy_id") != STRATEGY_ID or lock.get("model_artifact_sha256") != sha:
            raise RuntimeError("Model/lock integrity validation failed.")
        if lock.get("paper_only") is not True or lock.get("live_orders") is not False:
            raise RuntimeError("Model lock must be paper_only=true and live_orders=false.")
        if lock.get("demo_execution_allowed") is not True:
            raise RuntimeError("This model lock does not allow Testnet execution.")
        expected = {
            "tp_long_sigma": 0.75, "tp_short_sigma": 0.75, "sl_sigma": 1.0,
            "time_exit_bars": 96, "volatility_horizon_bars": 96,
            "entry_cooldown_bars": 96, "position_cap_usd": 50,
            "leverage": 1, "margin_type": "ISOLATED",
        }
        locked = lock.get("parameters", {})
        if any(locked.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Model lock parameters do not match the frozen H96 C2 configuration.")
        if self.model is None or self.model_sha != sha:
            artifact = joblib.load(MODEL_PATH)
            if artifact.get("strategy_id") != STRATEGY_ID:
                raise RuntimeError("Unexpected model strategy id.")
            thresholds = lock.get("thresholds_locked", {})
            if (
                float(thresholds.get("long", -1)) != float(artifact["threshold_long"])
                or float(thresholds.get("short", -1)) != float(artifact["threshold_short"])
            ):
                raise RuntimeError("Locked probability thresholds do not match the model artifact.")
            if not artifact.get("feature_names"):
                raise RuntimeError("Frozen model has no feature schema.")
            self.model, self.model_sha = artifact, sha
        return self.model

    def model_ready(self):
        try:
            self._load_model()
            return True
        except Exception:
            return False

    @staticmethod
    def position_amount(rows):
        for row in rows:
            if row.get("positionSide") == "BOTH":
                return safe_float(row.get("positionAmt"))
        return 0.0

    def _connect(self):
        client = self._client()
        client.sync_time()
        self.rules = symbol_rules(client.exchange_info(SYMBOL), SYMBOL)
        account = client.account()
        mode = client.position_mode()
        margin_mode = client.multi_asset_mode()
        self.multi_assets_mode = bool(margin_mode.get("multiAssetsMargin"))
        self.account_state = {
            "available_balance_usdt": safe_float(account.get("availableBalance")),
            "wallet_balance_usdt": safe_float(account.get("totalWalletBalance")),
            "equity_usdt": safe_float(account.get("totalMarginBalance")),
            "unrealized_pnl_usdt": safe_float(account.get("totalUnrealizedProfit")),
        }
        return {
            "connected": True,
            "base_url": "https://testnet.binancefuture.com",
            "environment": "BINANCE USDⓈ-M FUTURES LEGACY TESTNET ONLY",
            "symbol": SYMBOL,
            "hedge_mode": bool(mode.get("dualSidePosition")),
            "multi_assets_mode": self.multi_assets_mode,
            "rules": self.rules,
            "multi_assets_mode": self.multi_assets_mode,
            "account": self.account_state,
        }

    def connect_report(self):
        report = self._connect()
        report["position_amount"] = self.position_amount(self._client().positions(SYMBOL))
        report["open_orders_count"] = len(self._client().open_orders(SYMBOL))
        return report

    def status(self):
        trade = self.state.get("trade")
        signal = self.last_signal or {}
        return {
            "environment": "BINANCE USDⓈ-M FUTURES LEGACY TESTNET ONLY",
            "base_url": "https://testnet.binancefuture.com",
            "symbol": SYMBOL,
            "paper_only": True, "live_orders": False,
            "credentials_configured": self.credentials_configured,
            "credentials_source": self.credentials_source,
            "credential_storage": "Windows Credential Manager",
            "demo_orders_enabled": self.orders_enabled,
            "model_ready": self.model_ready(),
            "engine_state": self.state_name,
            "accepting_new_entries": self.accept_entries,
            "position_cap_usd": POSITION_CAP,
            "max_drawdown_pct": MAX_DRAWDOWN_PCT,
            "last_candle_time_utc": self.last_candle,
            "last_signal": signal,
            "active_trade": trade,
            "account": self.account_state,
            "last_error": self.last_error,
            "events": list(reversed(self.events[-20:])),
        }

    def start(self, confirm_demo_orders=False):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("Testnet engine is already running.")
            if not self.orders_enabled and confirm_demo_orders is not True:
                raise RuntimeError("Explicit Testnet order confirmation is required from the Start button.")
            self._load_model()
            info = self._connect()
            if info["hedge_mode"]:
                raise RuntimeError("Futures Testnet is in Hedge Mode. Switch it manually to One-way Mode.")
            if info.get("multi_assets_mode"):
                raise RuntimeError(
                    "Futures Testnet is in Multi-Assets Mode. H96 C2 requires Single-Asset Mode "
                    "because it uses ISOLATED margin. Switch the Futures M menu to Single-Asset Mode "
                    "and run Check Futures Testnet again. No orders were sent."
                )
            client = self._client()
            positions = self.position_amount(client.positions(SYMBOL))
            orders = client.open_orders(SYMBOL)
            algo_orders = client.open_algo_orders(SYMBOL)
            trade = self.state.get("trade")
            if trade:
                # Only reconcile orders bearing the IDs persisted by this bot.
                normal_ids = {x.get("origClientOrderId") for x in orders}
                algo_ids = {x.get("clientAlgoId") for x in algo_orders}
                unexpected_normal = normal_ids - {trade.get("tp_id")}
                unexpected_algo = algo_ids - {trade.get("sl_id")}
                if unexpected_normal or unexpected_algo:
                    raise RuntimeError(
                        "Unexpected BTCUSDT orders exist. The bot will not touch manual orders; "
                        "reconcile the Testnet account before restarting."
                    )

                if abs(positions) > 1e-12:
                    # Query by the saved client IDs: a filled/triggered exit may no longer
                    # appear in openOrders/openAlgoOrders but still needs reconciliation.
                    try:
                        saved_tp = client.order(SYMBOL, trade.get("tp_id"))
                        tp_status = str(saved_tp.get("status", "")).upper()
                    except BinanceAPIError as exc:
                        if exc.code not in {-2011, -2013}:
                            raise
                        tp_status = "MISSING"
                    try:
                        saved_sl = client.algo_order(SYMBOL, trade.get("sl_id"))
                        sl_status = str(saved_sl.get("algoStatus", "")).upper()
                    except BinanceAPIError as exc:
                        if exc.code not in {-2011, -2013}:
                            raise
                        sl_status = "MISSING"

                    if tp_status == "MISSING" or sl_status == "MISSING":
                        raise RuntimeError(
                            "Saved Testnet position is missing its TP or Algo STOP. "
                            "Inspect and protect/close it manually before restarting."
                        )
                else:
                    # Position is already flat; only the saved protective orders may be canceled.
                    self._cancel_owned(trade.get("tp_id"))
                    self._cancel_algo_owned(trade.get("sl_id"))
                    self.state["trade"] = None
                    self._save()
                    trade = None
            elif abs(positions) > 1e-12 or orders or algo_orders:
                raise RuntimeError(
                    "BTCUSDT has an unrecognized position/order. Reconcile it manually in Futures Testnet."
                )
            if not trade:
                client.set_isolated(SYMBOL)
                client.set_leverage(SYMBOL, 1)
            frame, _ = self._closed_candles()
            self.last_candle = frame["candle_time"].iloc[-1].isoformat()
            self.last_processed_ms = int(frame["candle_time"].iloc[-1].timestamp() * 1000)
            # The explicit Start confirmation enables Demo order sending for this process only.
            self.orders_enabled = True
            self.accept_entries = True
            self.halt.clear()
            self.state_name = "RUNNING_WAITING_FOR_NEXT_15M_CLOSE"
            self.last_error = None
            self.thread = threading.Thread(target=self._loop, name="osh-binance-demo", daemon=True)
            self.thread.start()
            self._event("ENGINE_STARTED", symbol=SYMBOL, notional_cap_usd=POSITION_CAP, leverage=1, margin="ISOLATED")
            return {"started": True, "message": "Waiting for the next newly closed 15m candle.", "connection": info}

    def stop_entries(self):
        with self.lock:
            self.accept_entries = False
            self._event("NEW_ENTRIES_DISABLED")
            self.state_name = "STOPPING" if not self.state.get("trade") else "MANAGING_OPEN_DEMO_POSITION"
            return {"stopping": True, "message": "No new entries. Any open position remains protected and monitored."}

    def _closed_candles(self):
        client = self._client()
        frame, close_times = candles_from_binance_klines(client.klines(SYMBOL, 500))
        now_ms = int(time.time() * 1000) + client.offset_ms
        keep = [x < now_ms for x in close_times]
        frame = frame.loc[keep].reset_index(drop=True)
        if frame.empty:
            raise RuntimeError("No fully closed Futures Testnet candles.")
        return frame, [x for x, use in zip(close_times, keep) if use]

    def _loop(self):
        while not self.halt.is_set():
            try:
                frame, _ = self._closed_candles()
                candle_ms = int(frame["candle_time"].iloc[-1].timestamp() * 1000)
                self.last_candle = frame["candle_time"].iloc[-1].isoformat()
                self._monitor_trade(candle_ms)
                if candle_ms > self.last_processed_ms:
                    gap = candle_ms - self.last_processed_ms
                    self.last_processed_ms = candle_ms
                    if gap == INTERVAL_MS and self.accept_entries and not self.state.get("trade"):
                        self._evaluate_signal(frame, candle_ms)
                    elif gap > INTERVAL_MS:
                        self._event("MISSED_CANDLES", count=gap // INTERVAL_MS - 1, action="skip late entries")
                if not self.accept_entries and not self.state.get("trade"):
                    self.state_name = "STOPPED"
                    self._event("ENGINE_STOPPED")
                    break
                self.state_name = "MANAGING_OPEN_DEMO_POSITION" if self.state.get("trade") else "RUNNING_WAITING_FOR_NEXT_15M_CLOSE"
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.accept_entries = False
                self.state_name = "ERROR_MANAGING_POSITION" if self.state.get("trade") else "ERROR_STOPPED"
                self._event("ENGINE_ERROR", error=self.last_error)
                if not self.state.get("trade"):
                    break
            self.halt.wait(5)

    def _evaluate_signal(self, frame, candle_ms):
        artifact = self._load_model()
        X, names = pv.make_features(frame)
        if list(names) != list(artifact["feature_names"]):
            raise RuntimeError("Testnet serving features differ from the frozen training schema.")
        latest = X.iloc[[-1]].replace([np.inf, -np.inf], np.nan)
        p_long = float(hunt.predict_side_proba(artifact["model_long"], artifact["classes_long"], latest)[0, 2])
        p_short = float(hunt.predict_side_proba(artifact["model_short"], artifact["classes_short"], latest)[0, 2])
        tl, ts = float(artifact["threshold_long"]), float(artifact["threshold_short"])
        side = choose_side(p_long, p_short, tl, ts)
        self.last_signal = {
            "candle_open_time_utc": frame["candle_time"].iloc[-1].isoformat(),
            "p_long": p_long, "p_short": p_short,
            "threshold_long": tl, "threshold_short": ts,
            "decision": side or "NO_SIGNAL",
        }
        self._event("SIGNAL_EVALUATED", **self.last_signal)
        if side is None:
            return
        previous = self.state.get("last_entry_candle_ms")
        if previous is not None and candle_ms - int(previous) < 96 * INTERVAL_MS:
            self._event("SIGNAL_SKIPPED_COOLDOWN")
            return
        account = self._client().account()
        equity = safe_float(account.get("totalMarginBalance"), safe_float(account.get("totalWalletBalance")))
        available = safe_float(account.get("availableBalance"))
        peak = max(equity, safe_float(self.state.get("peak_equity"), equity))
        self.state["peak_equity"] = peak
        self._save()
        dd = (peak - equity) / peak * 100 if peak > 0 else 100
        if dd >= MAX_DRAWDOWN_PCT:
            self.accept_entries = False
            self._event("DRAWDOWN_ENTRY_HALT", drawdown_pct=dd, equity_usdt=equity)
            return
        cap = min(POSITION_CAP, available * 0.90)
        price = self._client().ticker_price(SYMBOL)
        rules = self.rules or symbol_rules(self._client().exchange_info(SYMBOL), SYMBOL)
        try:
            qty = quantity_for_cap(cap, price, rules["market_step_size"], rules["market_min_qty"], rules["min_notional"])
        except ValueError as exc:
            self._event("SIGNAL_SKIPPED_EXCHANGE_MINIMUM", side=side, reason=str(exc), position_cap_usd=cap)
            return
        limit_qty = decimal_floor_step(qty, rules["limit_step_size"])
        if limit_qty != qty or qty < DecimalValue(rules["limit_min_qty"]):
            self._event(
                "SIGNAL_SKIPPED_LOT_SIZE_MISMATCH",
                side=side,
                market_qty=format_decimal(qty),
                compatible_limit_qty=format_decimal(limit_qty),
                reason="Market and protective LIMIT quantity filters do not accept the same size.",
            )
            return
        sigma = sigma_from_history(frame)
        if not self.accept_entries:
            self._event("ENTRY_ABORTED_STOP_REQUEST", side=side)
            return
        self._enter(side, qty, sigma, candle_ms, {"p_long": p_long, "p_short": p_short})

    def _enter(self, side, qty, sigma, candle_ms, probs):
        if not self.accept_entries:
            self._event("ENTRY_ABORTED_STOP_REQUEST", side=side)
            return
        client = self._client()
        entry_id = random_client_id("OSHEN")
        # Persist cooldown before POST; do not blindly retry a timed-out MARKET order.
        self.state["last_entry_candle_ms"] = int(candle_ms)
        self._save()
        entry_params = {
            "symbol": SYMBOL, "side": "BUY" if side == "LONG" else "SELL",
            "type": "MARKET", "quantity": format_decimal(qty),
            "newClientOrderId": entry_id, "newOrderRespType": "RESULT",
        }
        try:
            entry = client.place_order(entry_params)
        except Exception:
            try:
                entry = client.order(SYMBOL, entry_id)
            except Exception as exc:
                self._event("ENTRY_RESULT_UNKNOWN", client_order_id=entry_id, action="manual Testnet reconciliation; no retry")
                raise RuntimeError("Market entry result unknown; no duplicate order was sent.") from exc
        filled = safe_float(entry.get("executedQty"))
        avg = safe_float(entry.get("avgPrice"))
        if avg <= 0 and filled > 0:
            avg = safe_float(entry.get("cumQuote")) / filled
        if str(entry.get("status", "")).upper() != "FILLED" or filled <= 0 or avg <= 0:
            if filled > 0:
                self.state["trade"] = {
                    "side": side, "quantity": filled, "tp_id": None, "sl_id": None,
                    "phase": "EXIT_REQUESTED", "signal_candle_open_ms": int(candle_ms),
                }
                self._save()
                self._close_reduce_only("PARTIAL_ENTRY")
            self._event("ENTRY_NOT_FILLED", status=entry.get("status"), executed_qty=filled)
            return

        tp, sl = barrier_prices(side, avg, sigma)
        rules = self.rules or symbol_rules(client.exchange_info(SYMBOL), SYMBOL)
        exact_filled = DecimalValue(filled)
        if (
            decimal_floor_step(exact_filled, rules["market_step_size"]) != exact_filled
            or decimal_floor_step(exact_filled, rules["limit_step_size"]) != exact_filled
            or exact_filled < DecimalValue(rules["limit_min_qty"])
        ):
            self.state["trade"] = {
                "side": side, "quantity": filled, "tp_id": None, "sl_id": None,
                "phase": "EXIT_REQUESTED", "signal_candle_open_ms": int(candle_ms),
            }
            self._save()
            self._close_reduce_only("FILLED_QUANTITY_NOT_COMPATIBLE_WITH_PROTECTIVE_FILTERS")
            raise RuntimeError("Filled quantity is incompatible with protective exit filters.")
        if side == "LONG":
            tp_price = round_price_to_tick(tp, rules["tick_size"], "down")
            sl_price = round_price_to_tick(sl, rules["tick_size"], "up")
        else:
            tp_price = round_price_to_tick(tp, rules["tick_size"], "up")
            sl_price = round_price_to_tick(sl, rules["tick_size"], "down")
        current = client.ticker_price(SYMBOL)
        tp_crossed = current >= float(tp_price) if side == "LONG" else current <= float(tp_price)
        sl_crossed = current <= float(sl_price) if side == "LONG" else current >= float(sl_price)
        if tp_crossed or sl_crossed:
            self.state["trade"] = {"side": side, "quantity": filled, "tp_id": None, "sl_id": None}
            self._save()
            self._close_reduce_only("BARRIER_ALREADY_CROSSED_AFTER_ENTRY")
            self._event("BARRIER_CROSSED_DURING_ENTRY", side=side, current_price=current)
            return
        if (side == "LONG" and not float(sl_price) < avg < float(tp_price)) or (
            side == "SHORT" and not float(tp_price) < avg < float(sl_price)
        ):
            self.state["trade"] = {"side": side, "quantity": filled, "tp_id": None, "sl_id": None}
            self._save()
            self._close_reduce_only("INVALID_BARRIER_ROUNDING")
            raise RuntimeError("Rounded TP/SL levels invalid; reduce-only close requested.")

        tp_id, sl_id = random_client_id("OSHTP"), random_client_id("OSHSL")
        self.state["trade"] = {
            "side": side, "quantity": filled, "entry_price": avg,
            "entry_order_id": entry.get("orderId"), "entry_client_id": entry_id,
            "signal_candle_open_ms": int(candle_ms),
            "signal_candle_time_utc": datetime.fromtimestamp(candle_ms/1000, timezone.utc).isoformat(),
            "sigma_96": sigma, "tp_price": float(tp_price), "sl_price": float(sl_price),
            "tp_id": tp_id, "sl_id": sl_id, "phase": "PROTECTING",
            "p_long": probs["p_long"], "p_short": probs["p_short"],
        }
        self._save()
        exit_side = "SELL" if side == "LONG" else "BUY"
        try:
            # Set the stop first. Take-profit is a post-only conditional Maker limit.
            client.place_algo_order({
                "algoType": "CONDITIONAL",
                "symbol": SYMBOL,
                "side": exit_side,
                "type": "STOP_MARKET",
                "triggerPrice": format_decimal(sl_price),
                "quantity": format_decimal(DecimalQty(filled, rules["market_step_size"])),
                "reduceOnly": True,
                "workingType": "CONTRACT_PRICE",
                "clientAlgoId": sl_id,
            })
            client.place_order({
                "symbol": SYMBOL, "side": exit_side, "type": "LIMIT",
                "timeInForce": "GTX", "price": format_decimal(tp_price),
                "quantity": format_decimal(DecimalQty(filled, rules["limit_step_size"])),
                "reduceOnly": True, "newClientOrderId": tp_id,
            })
        except Exception as exc:
            self._cancel_algo_owned(sl_id)
            self._cancel_owned(tp_id)
            self._close_reduce_only("PROTECTIVE_ORDER_SETUP_FAILED")
            self._event("PROTECTION_FAILED_CLOSE_REQUESTED", error=f"{type(exc).__name__}: {exc}")
            raise RuntimeError("Could not establish both Testnet protective exits; reduce-only close requested.") from exc
        self.state["trade"]["phase"] = "PROTECTED"
        self._save()
        self._event("DEMO_ENTRY_FILLED", side=side, quantity=filled, entry_price=avg,
                    tp_price=float(tp_price), sl_price=float(sl_price), entry_order_id=entry.get("orderId"))

    def _cancel_owned(self, client_id):
        if not client_id:
            return
        try:
            self._client().cancel_order(SYMBOL, client_id)
        except BinanceAPIError as exc:
            if exc.code not in {-2011, -2013}:
                self._event("CANCEL_WARNING", client_id=client_id, error=str(exc))
        except Exception as exc:
            self._event("CANCEL_WARNING", client_id=client_id, error=type(exc).__name__)

    def _cancel_algo_owned(self, client_algo_id):
        if not client_algo_id:
            return
        try:
            self._client().cancel_algo_order(SYMBOL, client_algo_id)
        except BinanceAPIError as exc:
            if exc.code not in {-2011, -2013}:
                self._event("ALGO_CANCEL_WARNING", client_algo_id=client_algo_id, error=str(exc))
        except Exception as exc:
            self._event("ALGO_CANCEL_WARNING", client_algo_id=client_algo_id, error=type(exc).__name__)

    def _close_reduce_only(self, reason):
        trade = self.state.get("trade") or {}
        trade["phase"] = "EXIT_REQUESTED"
        self._cancel_owned(trade.get("tp_id"))
        self._cancel_algo_owned(trade.get("sl_id"))
        rows = self._client().positions(SYMBOL)
        amount = self.position_amount(rows)
        if abs(amount) <= 1e-12:
            self.state["trade"] = None
            self._save()
            return
        rules = self.rules or symbol_rules(self._client().exchange_info(SYMBOL), SYMBOL)
        qty = decimal_floor_step(abs(amount), rules["market_step_size"])
        if qty <= 0:
            raise RuntimeError("Could not derive a valid reduce-only close quantity.")
        close_id = trade.get("close_id") or random_client_id("OSHCL")
        trade["close_id"] = close_id
        self.state["trade"] = trade
        self._save()
        params = {
            "symbol": SYMBOL, "side": "SELL" if amount > 0 else "BUY",
            "type": "MARKET", "quantity": format_decimal(qty),
            "reduceOnly": True, "newClientOrderId": close_id, "newOrderRespType": "RESULT",
        }
        try:
            result = self._client().place_order(params)
        except Exception:
            try:
                result = self._client().order(SYMBOL, close_id)
            except Exception as exc:
                self._event("REDUCE_ONLY_CLOSE_UNKNOWN", reason=reason, client_order_id=close_id)
                raise RuntimeError("Reduce-only close result unknown; inspect Testnet account.") from exc
        self._event("REDUCE_ONLY_CLOSE_SENT", reason=reason, status=result.get("status"), order_id=result.get("orderId"))
        remaining = self.position_amount(self._client().positions(SYMBOL))
        if abs(remaining) <= 1e-12:
            self.state["trade"] = None
        else:
            order_status = str(result.get("status", "")).upper()
            trade["close_id"] = close_id if order_status in {"NEW", "PARTIALLY_FILLED"} else None
            trade["phase"] = "EXIT_REQUESTED"
            self.state["trade"] = trade
        self._save()

    def _monitor_trade(self, candle_ms):
        trade = self.state.get("trade")
        if not trade:
            return
        client = self._client()
        amount = self.position_amount(client.positions(SYMBOL))
        if abs(amount) <= 1e-12:
            self._cancel_owned(trade.get("tp_id"))
            self._cancel_algo_owned(trade.get("sl_id"))
            self.state["trade"] = None
            self._save()
            self._event("DEMO_POSITION_CLOSED", side=trade.get("side"),
                        note="Check Futures Testnet order history for exact realized PnL, commission and funding.")
            return

        if trade.get("phase") == "EXIT_REQUESTED":
            close_id = trade.get("close_id")
            if close_id:
                try:
                    close_status = str(client.order(SYMBOL, close_id).get("status", "")).upper()
                    if close_status in {"NEW", "PARTIALLY_FILLED"}:
                        return
                except BinanceAPIError as exc:
                    if exc.code not in {-2011, -2013}:
                        raise
            trade["close_id"] = None
            self.state["trade"] = trade
            self._save()
            self._close_reduce_only("RETRY_REMAINING_POSITION")
            return
        tp_status, sl_status = "UNKNOWN", "UNKNOWN"
        try:
            tp_status = str(client.order(SYMBOL, trade["tp_id"]).get("status", "")).upper() if trade.get("tp_id") else "UNKNOWN"
        except BinanceAPIError as exc:
            if exc.code not in {-2011, -2013}:
                raise
        try:
            algo = client.algo_order(SYMBOL, trade["sl_id"]) if trade.get("sl_id") else {}
            sl_status = str(algo.get("algoStatus", "")).upper() or "UNKNOWN"
        except BinanceAPIError as exc:
            if exc.code not in {-2011, -2013}:
                raise

        if tp_status in {"FILLED", "PARTIALLY_FILLED"} or sl_status in {"FILLED", "PARTIALLY_FILLED"}:
            trade["phase"] = "EXIT_REQUESTED"
            self.state["trade"] = trade
            self._save()
            self._close_reduce_only("PROTECTIVE_EXIT_FILLED_OR_PARTIAL")
            return
        if tp_status not in {"NEW"} or sl_status not in {"NEW"}:
            trade["phase"] = "EXIT_REQUESTED"
            self.state["trade"] = trade
            self._save()
            self._close_reduce_only("PROTECTIVE_ORDER_MISSING")
            return

        elapsed = (int(candle_ms) - int(trade["signal_candle_open_ms"])) // INTERVAL_MS
        if elapsed >= 96:
            trade["phase"] = "EXIT_REQUESTED"
            self.state["trade"] = trade
            self._save()
            self._close_reduce_only("TIME_EXIT_96_BARS")


def DecimalValue(value):
    from decimal import Decimal
    return Decimal(str(value))


def DecimalQty(value, step):
    from decimal import Decimal, ROUND_DOWN
    x, s = Decimal(str(value)), Decimal(str(step))
    return (x / s).to_integral_value(rounding=ROUND_DOWN) * s
