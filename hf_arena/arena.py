"""HF Model Arena — fair, simultaneous A/B evaluation of trading tactics.

Every model sees the SAME candles, SAME frozen probabilities, SAME fees and
starts from the SAME $100. Only cooldown, selectivity and sizing differ.
Two balances are kept per model:
  - net:   after real commission + slippage  (the ONLY decision basis)
  - gross: before costs (diagnostic; never a GO basis, never hidden)

This module is pure computation. It never places orders and never reads keys.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    INITIAL_CAPITAL_USD, MAX_CONCURRENT_PER_MODEL, MAX_DRAWDOWN_PCT,
    MAX_DAILY_LOSS_USD, MAX_NOTIONAL_USD, MAX_TRADES_PER_HOUR,
    MAKER_FEE_BPS_PER_SIDE, MODELS, ROUND_TRIP_TAKER_BPS,
    SLIPPAGE_ROUND_TRIP_BPS, TAKER_FEE_BPS_PER_SIDE,
)

FEATURE_COLUMNS_REQUIRED = 37


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fees_bps(prefer_maker: bool) -> float:
    """Round-trip cost. Maker entry is an UNOPTIMISTIC assumption only when filled."""
    if prefer_maker:
        return 2 * MAKER_FEE_BPS_PER_SIDE + SLIPPAGE_ROUND_TRIP_BPS
    return ROUND_TRIP_TAKER_BPS


NS_PER_BAR = 15 * 60 * 1_000_000_000


@dataclass
class Position:
    side: int                 # +1 long / -1 short
    entry_price: float
    entry_bar: int
    entry_time: str
    notional_usd: float
    tp_price: float
    sl_price: float
    hold_bars: int
    confidence: float
    entry_ns: int = 0
    hold_ns: int = 0
    client_id: str = ""


@dataclass
class Trade:
    model_id: str
    side: int
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    gross_ret: float
    fee_bps: float
    net_ret: float
    notional_usd: float
    confidence: float
    reason: str
    gross_pnl_usd: float
    net_pnl_usd: float


@dataclass
class ModelLedger:
    model_id: str
    label: str
    balance_net: float = INITIAL_CAPITAL_USD
    balance_gross: float = INITIAL_CAPITAL_USD
    peak_net: float = INITIAL_CAPITAL_USD
    max_dd_pct: float = 0.0
    fees_paid_usd: float = 0.0
    trades: list = field(default_factory=list)
    open_positions: list = field(default_factory=list)
    last_entry_bar: int = -10**9
    last_entry_ns: int = -1
    hourly_entries: list = field(default_factory=list)
    daily_start_balance: float = INITIAL_CAPITAL_USD
    daily_date: str = ""
    halted_reason: str = ""

    def snapshot(self) -> dict:
        records = [t for t in self.trades]
        nets = np.asarray([t.net_ret for t in records], float)
        grosses = np.asarray([t.gross_ret for t in records], float)
        wins = nets > 0 if len(nets) else np.asarray([], bool)
        pos = float(nets[wins].sum()) if wins.any() else 0.0
        neg = float(-nets[~wins].sum()) if len(nets) and (~wins).any() else 0.0
        return {
            "model_id": self.model_id,
            "label": self.label,
            "balance_net_usd": round(self.balance_net, 2),
            "balance_gross_usd": round(self.balance_gross, 2),
            "pnl_net_usd": round(self.balance_net - INITIAL_CAPITAL_USD, 2),
            "pnl_gross_usd": round(self.balance_gross - INITIAL_CAPITAL_USD, 2),
            "return_net_pct": round((self.balance_net / INITIAL_CAPITAL_USD - 1) * 100, 2),
            "trades": len(records),
            "wins": int(wins.sum()) if len(nets) else 0,
            "win_rate_net_pct": round(float(wins.mean() * 100), 2) if len(nets) else None,
            "win_rate_gross_pct": round(float((grosses > 0).mean() * 100), 2) if len(grosses) else None,
            "profit_factor_net": round(pos / neg, 4) if neg > 0 else None,
            "expectancy_net_bps": round(float(nets.mean() * 1e4), 3) if len(nets) else None,
            "expectancy_gross_bps": round(float(grosses.mean() * 1e4), 3) if len(grosses) else None,
            "fees_paid_usd": round(self.fees_paid_usd, 3),
            "max_drawdown_pct": round(self.max_dd_pct, 2),
            "open_positions": len(self.open_positions),
            "halted_reason": self.halted_reason,
        }


class FrozenSignals:
    """Read-only wrapper over the user's frozen H96 C2 forward-paper model."""

    def __init__(self, artifact_path: Path, feature_names: list[str]):
        import joblib

        artifact = joblib.load(artifact_path)
        self.long = artifact["model_long"]
        self.short = artifact["model_short"]
        self.classes_long = np.asarray(artifact["classes_long"], int)
        self.classes_short = np.asarray(artifact["classes_short"], int)
        self.thr_long = float(artifact["threshold_long"])
        self.thr_short = float(artifact["threshold_short"])
        self.features = list(artifact.get("feature_names", feature_names))
        if len(self.features) != FEATURE_COLUMNS_REQUIRED:
            raise ValueError(f"Frozen artifact expects {len(self.features)} features, "
                             f"expected {FEATURE_COLUMNS_REQUIRED}.")

    def _align(self, model, classes, X: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(model.predict_proba(X[self.features].astype(float)), float)
        out = np.zeros((len(X), 3), float)
        for j, cls in enumerate(classes):
            out[:, int(cls)] = raw[:, j]
        return out

    def probabilities(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p_long = self._align(self.long, self.classes_long, X)
        p_short = self._align(self.short, self.classes_short, X)
        return p_long, p_short


class Arena:
    def __init__(self, signals: FrozenSignals | None = None):
        self.signals = signals
        self.ledgers = {
            m["id"]: ModelLedger(model_id=m["id"], label=m["label"]) for m in MODELS
        }
        self.models = {m["id"]: m for m in MODELS}
        self.bars: list[dict] = []
        self.p_long: np.ndarray | None = None
        self.p_short: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.events: list[dict] = []
        self.started_at = _now()

    # ------------------------------------------------------------- features
    def set_candles(self, df: pd.DataFrame) -> None:
        """Recompute stationary features and frozen probabilities for the window."""
        import pipeline_v2 as pv  # local import: research code, not modified

        X, _ = pv.make_features(df.reset_index(drop=True))
        log_ret = np.log(df["close"].astype(float) / df["close"].astype(float).shift(1))
        self.sigma = (log_ret.rolling(96, min_periods=48).std().to_numpy(float))
        self.bars = df.to_dict("records")
        if self.signals is not None:
            Xf = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            pl, ps = self.signals.probabilities(Xf)
            self.p_long, self.p_short = pl, ps
        else:
            n = len(df)
            self.p_long = np.tile([0.34, 0.32, 0.34], (n, 1))
            self.p_short = np.tile([0.34, 0.32, 0.34], (n, 1))

    # ------------------------------------------------------------- sizing
    @staticmethod
    def confidence(p: float, threshold: float) -> float:
        if threshold >= 1.0:
            return 0.0
        return float(np.clip((p - threshold) / (1.0 - threshold), 0.0, 1.0))

    def notional(self, model: dict, conf: float) -> float:
        base = float(model.get("fixed_notional_usd", 12.5))
        if model.get("sizing") == "confidence":
            # Confidence-weighted: 50%..100% of the model's base slice.
            base *= 0.5 + 0.5 * conf
        return float(min(MAX_NOTIONAL_USD, max(1.0, base)))

    def hold_bars(self, model: dict) -> int:
        return int(min(96, max(16, model["cooldown_bars"] * 4)))

    # ------------------------------------------------------------- guards
    def _guard(self, led: ModelLedger, bar_time: str) -> bool:
        if led.halted_reason:
            return False
        if led.max_dd_pct >= MAX_DRAWDOWN_PCT:
            led.halted_reason = f"max drawdown {led.max_dd_pct:.1f}% >= {MAX_DRAWDOWN_PCT}%"
        day = str(bar_time)[:10]
        if day != led.daily_date:
            led.daily_date = day
            led.daily_start_balance = led.balance_net
        if led.daily_start_balance - led.balance_net >= MAX_DAILY_LOSS_USD:
            led.halted_reason = f"daily loss ${MAX_DAILY_LOSS_USD} reached"
        if led.balance_net <= 0:
            led.halted_reason = "account depleted"
        if led.halted_reason:
            self.events.append({"time": _now(), "event": "MODEL_HALTED",
                                "model": led.model_id, "reason": led.halted_reason})
            return False
        return True

    def _rate_ok(self, led: ModelLedger, bar_ns: int) -> bool:
        hour_ns = 3600 * 1_000_000_000
        led.hourly_entries = [t for t in led.hourly_entries if bar_ns - t < hour_ns]
        return len(led.hourly_entries) < MAX_TRADES_PER_HOUR

    # ------------------------------------------------------------- exits
    def _manage_open(self, led: ModelLedger, model: dict, i: int, bar: dict) -> None:
        bar_ns = int(pd.Timestamp(bar["candle_time"]).value)
        remaining = []
        for p in led.open_positions:
            hi, lo = float(bar["high"]), float(bar["low"])
            hit_tp = hi >= p.tp_price if p.side == 1 else lo <= p.tp_price
            hit_sl = lo <= p.sl_price if p.side == 1 else hi >= p.sl_price
            expired = (bar_ns - p.entry_ns) >= p.hold_ns if p.hold_ns else \
                (i - p.entry_bar >= p.hold_bars)
            if not (hit_tp or hit_sl or expired):
                remaining.append(p)
                continue
            # Conservative: same-bar double touch is treated as a stop.
            if hit_sl and (hit_tp or not hit_tp):
                exit_price = min(float(bar["open"]), p.sl_price) if p.side == 1 \
                    else max(float(bar["open"]), p.sl_price)
                reason = "stop_loss"
            elif hit_tp:
                exit_price = max(float(bar["open"]), p.tp_price) if p.side == 1 \
                    else min(float(bar["open"]), p.tp_price)
                reason = "take_profit"
            else:
                exit_price = float(bar["close"])
                reason = "time"
            self._close(led, model, p, exit_price, reason, str(bar["candle_time"]))
        led.open_positions = remaining

    def _close(self, led: ModelLedger, model: dict, p: Position,
               exit_price: float, reason: str, exit_time: str) -> None:
        raw = exit_price / p.entry_price - 1.0
        gross_ret = raw if p.side == 1 else -raw
        fee = fees_bps(bool(model.get("prefer_maker"))) / 1e4
        net_ret = gross_ret - fee
        gross_pnl = p.notional_usd * gross_ret
        net_pnl = p.notional_usd * net_ret
        led.balance_gross += gross_pnl
        led.balance_net += net_pnl
        led.fees_paid_usd += p.notional_usd * fee
        led.peak_net = max(led.peak_net, led.balance_net)
        dd = (led.peak_net - led.balance_net) / led.peak_net * 100 if led.peak_net > 0 else 0.0
        led.max_dd_pct = max(led.max_dd_pct, dd)
        led.trades.append(Trade(
            model_id=led.model_id, side=p.side, entry_time=p.entry_time,
            exit_time=exit_time, entry_price=p.entry_price, exit_price=exit_price,
            gross_ret=gross_ret, fee_bps=fee * 1e4, net_ret=net_ret,
            notional_usd=p.notional_usd, confidence=p.confidence, reason=reason,
            gross_pnl_usd=gross_pnl, net_pnl_usd=net_pnl,
        ))
        self.events.append({"time": _now(), "event": "TRADE_CLOSED", "model": led.model_id,
                            "reason": reason, "net_pnl_usd": round(net_pnl, 4),
                            "balance_net": round(led.balance_net, 2)})

    # ------------------------------------------------------------- entries
    def _maybe_enter(self, led: ModelLedger, model: dict, i: int, bar: dict) -> None:
        bar_ns = int(pd.Timestamp(bar["candle_time"]).value)
        cooldown_ns = int(model["cooldown_bars"]) * NS_PER_BAR
        if led.last_entry_ns > 0 and bar_ns - led.last_entry_ns < cooldown_ns:
            return
        if i - led.last_entry_bar < model["cooldown_bars"]:
            return
        if len(led.open_positions) >= MAX_CONCURRENT_PER_MODEL:
            return
        if not self._guard(led, str(bar["candle_time"])) or not self._rate_ok(led, bar_ns):
            return
        if self.p_long is None or i >= len(self.p_long) or i + 1 >= len(self.bars):
            return
        sig = self.signals
        thr_l = sig.thr_long if sig else 0.5184
        thr_s = sig.thr_short if sig else 0.4989
        pl, ps = float(self.p_long[i, 2]), float(self.p_short[i, 2])
        # Selectivity: require the probability to clear the frozen threshold,
        # then apply the model's own top_pct as an extra confidence floor.
        extra = max(0.0, (0.10 - float(model.get("top_pct", 0.10))) * 0.35)
        long_ok = pl >= thr_l + extra
        short_ok = ps >= thr_s + extra
        if not (long_ok or short_ok):
            return
        if long_ok and short_ok:
            side = 1 if (pl - thr_l) >= (ps - thr_s) else -1
            p = pl if side == 1 else ps
            thr = thr_l if side == 1 else thr_s
        elif long_ok:
            side, p, thr = 1, pl, thr_l
        else:
            side, p, thr = -1, ps, thr_s
        conf = self.confidence(p, thr)
        notional = self.notional(model, conf)
        entry = float(self.bars[i + 1]["open"])
        sig1 = float(self.sigma[i]) if self.sigma is not None and np.isfinite(self.sigma[i]) else 0.004
        hold = self.hold_bars(model)
        barrier_sigma = max(1e-5, sig1 * math.sqrt(hold))
        tp = barrier_sigma * float(model["tp_sigma"])
        sl = barrier_sigma * float(model["sl_sigma"])
        tp_price = entry * (1 + tp) if side == 1 else entry * (1 - tp)
        sl_price = entry * (1 - sl) if side == 1 else entry * (1 + sl)
        entry_ns = int(pd.Timestamp(self.bars[i + 1]["candle_time"]).value)
        pos = Position(side=side, entry_price=entry, entry_bar=i + 1,
                       entry_time=str(self.bars[i + 1]["candle_time"]),
                       notional_usd=notional, tp_price=tp_price, sl_price=sl_price,
                       hold_bars=hold, confidence=conf,
                       entry_ns=entry_ns, hold_ns=int(hold) * NS_PER_BAR)
        led.open_positions.append(pos)
        led.last_entry_bar = i
        led.last_entry_ns = entry_ns
        led.hourly_entries.append(entry_ns)
        self.events.append({"time": _now(), "event": "SIGNAL", "model": led.model_id,
                            "side": "LONG" if side == 1 else "SHORT",
                            "confidence": round(conf, 4), "notional_usd": round(notional, 2),
                            "entry_price": round(entry, 2)})

    # ------------------------------------------------------------- step
    def step(self, i: int) -> None:
        if i < 1 or i >= len(self.bars) - 1:
            return
        bar = self.bars[i]
        for model in MODELS:
            led = self.ledgers[model["id"]]
            self._manage_open(led, model, i, bar)
            self._maybe_enter(led, model, i, bar)

    def run_range(self, start: int, end: int) -> None:
        for i in range(max(1, start), min(end, len(self.bars) - 1)):
            self.step(i)

    # ------------------------------------------------------------- report
    def leaderboard(self) -> list[dict]:
        rows = [self.ledgers[m["id"]].snapshot() for m in MODELS]
        rows.sort(key=lambda r: r["balance_net_usd"], reverse=True)
        for rank, r in enumerate(rows, 1):
            r["rank"] = rank
        return rows

    def report(self) -> dict:
        return {
            "generated_at_utc": _now(),
            "started_at_utc": self.started_at,
            "initial_capital_usd": INITIAL_CAPITAL_USD,
            "max_notional_usd": MAX_NOTIONAL_USD,
            "leverage": 1,
            "fee_bps_taker_round_trip": ROUND_TRIP_TAKER_BPS,
            "fee_bps_maker_attempt_round_trip": fees_bps(True),
            "fee_note": "Fees are charged on every trade. Gross is diagnostic only; "
                        "net is the only decision basis. Commission is NEVER removed.",
            "models": [dict(m) for m in MODELS],
            "leaderboard": self.leaderboard(),
            "bars_seen": len(self.bars),
            "events": self.events[-200:],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.report()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
                       encoding="utf-8")
        tmp.replace(path)
