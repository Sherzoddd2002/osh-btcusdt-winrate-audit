"""Optional real-order layer for the Binance Futures TESTNET only.

The arena always runs in paper/shadow mode. Real orders are sent ONLY for the
single user-selected champion model, ONLY after explicit confirmation, and only
to the pinned non-production host. Every guard failure closes the order path.

Design rules (professional):
  - unique client order ids + dedup before every send
  - no retry of MARKET entry after an ambiguous error
  - reconciliation before any new entry (no stacked positions)
  - 1x leverage, ISOLATED, one-way mode, $100 notional cap
  - kill switch cancels owned orders and stops new entries
"""
from __future__ import annotations

import secrets
import time
from datetime import datetime, timezone

from .client import DemoAPIError, DemoClient
from .config import (
    LEVERAGE, MARGIN_TYPE, MAX_NOTIONAL_USD, ORDER_EXECUTION_LOCK_REASON,
    SYMBOL, TESTNET_ORDER_EXECUTION_ENABLED,
)


def _cid(prefix: str) -> str:
    return f"OSHHF{prefix}{secrets.token_hex(5)}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_rules(exchange_info: dict, symbol: str) -> dict:
    for s in exchange_info.get("symbols", []):
        if s.get("symbol") != symbol:
            continue
        rules = {"price_tick": 0.01, "qty_step": 0.001, "min_qty": 0.001,
                 "min_notional": 5.0, "price_precision": 2, "qty_precision": 3}
        for f in s.get("filters", []):
            ft = f.get("filterType")
            if ft == "PRICE_FILTER":
                rules["price_tick"] = float(f["tickSize"])
            elif ft == "LOT_SIZE":
                rules["qty_step"] = float(f["stepSize"])
                rules["min_qty"] = float(f["minQty"])
            elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
                rules["min_notional"] = float(f.get("notional", f.get("minNotional", 5.0)))
        rules["price_precision"] = int(s.get("pricePrecision", 2))
        rules["qty_precision"] = int(s.get("quantityPrecision", 3))
        return rules
    raise RuntimeError(f"{symbol} not found in Testnet exchangeInfo; refusing orders.")


def floor_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    n = int(value / step + 1e-9)
    return round(n * step, 10)


class DemoExecutor:
    def __init__(self, audit):
        self.audit = audit
        self.client: DemoClient | None = None
        self.rules: dict | None = None
        self.enabled = False
        self.champion = ""
        self.sent_ids: set[str] = set()
        self.active_trade: dict | None = None
        self.api_errors = 0
        self.last_action = "idle"

    # ------------------------------------------------------------- connect
    def connect(self, client: DemoClient) -> dict:
        self.client = client
        client.sync_time()
        info = client.exchange_info()
        self.rules = extract_rules(info, SYMBOL)
        account = client.account()
        dual = client.position_mode()
        multi = client.multi_assets_mode()
        positions = client.position_risk(SYMBOL)
        available = next((float(a.get("availableBalance", 0))
                          for a in account.get("assets", []) if a.get("asset") == "USDT"), None)
        report = {
            "host": "https://testnet.binancefuture.com (non-production)",
            "time_sync_ok": True,
            "account_read_ok": True,
            "available_usdt": available,
            "dual_position_mode": bool(dual.get("dualSidePosition")),
            "multi_assets_mode": bool(multi.get("multiAssetsMargin")),
            "open_position_qty": sum(abs(float(p.get("positionAmt", 0))) for p in positions),
            "symbol": SYMBOL,
            "filters": self.rules,
            "leverage": LEVERAGE, "margin_type": MARGIN_TYPE,
            "max_notional_usd": MAX_NOTIONAL_USD,
            "checked_at_utc": _now(),
        }
        report["preflight_ok"] = (
            not report["dual_position_mode"]
            and not report["multi_assets_mode"]
            and report["open_position_qty"] == 0
            and (available is None or available >= MAX_NOTIONAL_USD)
        )
        report["blockers"] = []
        if report["dual_position_mode"]:
            report["blockers"].append("Hedge Mode ON — switch to One-way in Futures settings.")
        if report["multi_assets_mode"]:
            report["blockers"].append("Multi-Assets Mode ON — ISOLATED is required; switch OFF.")
        if report["open_position_qty"]:
            report["blockers"].append("An open BTCUSDT position exists — close it before starting.")
        if available is not None and available < MAX_NOTIONAL_USD:
            report["blockers"].append("Testnet available balance is below the $100 cap.")
        self.audit("CONNECT", report)
        return report

    def arm(self, champion: str, confirm: bool) -> None:
        if not TESTNET_ORDER_EXECUTION_ENABLED:
            raise RuntimeError(ORDER_EXECUTION_LOCK_REASON)
        if not confirm:
            raise ValueError("Explicit confirmation is required to enable Testnet orders.")
        if self.client is None or self.rules is None:
            raise RuntimeError("Run the read-only connect check first.")
        self.client.set_isolated(SYMBOL)
        self.client.set_leverage(SYMBOL, LEVERAGE)
        self.enabled = True
        self.champion = champion
        self.api_errors = 0
        self.last_action = "armed"
        self.audit("ENGINE_ARMED", {"champion": champion, "leverage": LEVERAGE,
                                    "margin": MARGIN_TYPE, "max_notional": MAX_NOTIONAL_USD})

    def disarm(self, reason: str) -> None:
        self.enabled = False
        self.last_action = f"disarmed: {reason}"
        self.audit("ENGINE_DISARMED", {"reason": reason})

    # ------------------------------------------------------------- reconcile
    def reconcile(self) -> dict:
        if self.client is None:
            return {"ok": False, "reason": "no client"}
        open_orders = self.client.open_orders(SYMBOL)
        positions = self.client.position_risk(SYMBOL)
        qty = sum(abs(float(p.get("positionAmt", 0))) for p in positions)
        owned = [o for o in open_orders
                 if str(o.get("clientOrderId", "")).startswith("OSHHF")]
        if qty == 0 and self.active_trade:
            self.audit("POSITION_CLOSED_BY_EXCHANGE", {"trade": self.active_trade["entry_id"]})
            self.active_trade = None
        return {"ok": True, "open_orders": len(owned), "position_qty": qty,
                "has_active_trade": self.active_trade is not None}

    # ------------------------------------------------------------- entry
    def maybe_send(self, signal: dict) -> dict | None:
        """Send one MARKET entry + protective orders for the champion signal.

        `signal` keys: side (+1/-1), notional_usd, tp_price, sl_price, confidence,
        entry_bar. Returns None when the order path is closed or blocked.
        """
        if not self.enabled or self.client is None or self.rules is None:
            return None
        if signal.get("model_id") and signal["model_id"] != self.champion:
            return None
        if self.api_errors >= 3:
            self.disarm("3 consecutive API errors")
            return None
        try:
            state = self.reconcile()
            if not state["ok"] or state["position_qty"] > 0 or state["has_active_trade"]:
                return {"status": "SKIPPED", "reason": "position or orders already open"}
            price = self.client.ticker_price(SYMBOL)
            notional = min(float(signal["notional_usd"]), MAX_NOTIONAL_USD)
            raw_qty = notional / price
            qty = floor_step(raw_qty, self.rules["qty_step"])
            if qty < self.rules["min_qty"] or qty * price < self.rules["min_notional"]:
                self.audit("SIGNAL_SKIPPED_MIN_NOTIONAL", {"qty": qty, "price": price})
                return {"status": "SKIPPED", "reason": "below exchange minimum"}
            side = "BUY" if signal["side"] == 1 else "SELL"
            entry_id = _cid("EN")
            if entry_id in self.sent_ids:
                return {"status": "SKIPPED", "reason": "duplicate client id"}
            self.sent_ids.add(entry_id)
            params = {
                "symbol": SYMBOL, "side": side, "type": "MARKET",
                "quantity": f"{qty:.{self.rules['qty_precision']}f}",
                "newClientOrderId": entry_id,
                "newOrderRespType": "RESULT",
            }
            entry = self.client.place_order(params)
            self.api_errors = 0
            self.active_trade = {
                "entry_id": entry_id, "order_id": entry.get("orderId"),
                "side": side, "qty": qty, "price": price,
                "confidence": signal.get("confidence"),
                "opened_at_utc": _now(),
            }
            self.last_action = f"MARKET {side} {qty} @ ~{price}"
            self.audit("DEMO_ORDER_SENT", {"client_id": entry_id, "side": side,
                                           "qty": qty, "price": price,
                                           "notional_usd": round(qty * price, 2)})
            # Protective orders are best-effort; failures are logged, never retried blindly.
            close_side = "SELL" if side == "BUY" else "BUY"
            try:
                sl_id = _cid("SL")
                self.sent_ids.add(sl_id)
                self.client.place_order({
                    "symbol": SYMBOL, "side": close_side, "type": "STOP_MARKET",
                    "stopPrice": f"{float(signal['sl_price']):.{self.rules['price_precision']}f}",
                    "quantity": f"{qty:.{self.rules['qty_precision']}f}",
                    "newClientOrderId": sl_id, "closePosition": "true",
                })
                self.audit("PROTECTIVE_SL_SENT", {"client_id": sl_id})
            except (DemoAPIError, RuntimeError) as exc:
                self.audit("PROTECTIVE_SL_WARNING", {"error": str(exc)})
            try:
                tp_id = _cid("TP")
                self.sent_ids.add(tp_id)
                self.client.place_order({
                    "symbol": SYMBOL, "side": close_side, "type": "LIMIT",
                    "price": f"{float(signal['tp_price']):.{self.rules['price_precision']}f}",
                    "quantity": f"{qty:.{self.rules['qty_precision']}f}",
                    "timeInForce": "GTC", "newClientOrderId": tp_id,
                })
                self.audit("PROTECTIVE_TP_SENT", {"client_id": tp_id})
            except (DemoAPIError, RuntimeError) as exc:
                self.audit("PROTECTIVE_TP_WARNING", {"error": str(exc)})
            return {"status": "SENT", "client_id": entry_id, "side": side,
                    "qty": qty, "price": price}
        except (DemoAPIError, RuntimeError, ValueError) as exc:
            self.api_errors += 1
            self.audit("DEMO_ORDER_ERROR", {"error": str(exc),
                                            "consecutive": self.api_errors})
            if self.api_errors >= 3:
                self.disarm("3 consecutive API errors")
            return {"status": "ERROR", "reason": str(exc)}

    def kill(self) -> dict:
        """Cancel every arena-owned open order; stop all new entries."""
        cancelled = []
        if self.client is not None:
            try:
                for o in self.client.open_orders(SYMBOL):
                    cid = str(o.get("clientOrderId", ""))
                    if cid.startswith("OSHHF"):
                        try:
                            self.client.cancel_order(SYMBOL, cid)
                            cancelled.append(cid)
                        except (DemoAPIError, RuntimeError):
                            pass
            except (DemoAPIError, RuntimeError):
                pass
        self.disarm("kill switch")
        self.audit("KILL_SWITCH", {"cancelled": cancelled})
        return {"cancelled": cancelled, "armed": False}
