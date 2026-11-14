from __future__ import annotations

import math
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any

import numpy as np
import pandas as pd

INTERVAL_MS = 900_000
HORIZON_BARS = 96
VOLATILITY_WINDOW = 96
TP_SIGMA = 0.75
SL_SIGMA = 1.0


def decimal_floor_step(value: float | Decimal, step: str | float) -> Decimal:
    value_d, step_d = Decimal(str(value)), Decimal(str(step))
    if value_d <= 0 or step_d <= 0:
        raise ValueError("Value and step must be positive.")
    return (value_d / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d


def format_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def round_price_to_tick(value: float, tick: str | float, direction: str) -> Decimal:
    value_d, tick_d = Decimal(str(value)), Decimal(str(tick))
    if value_d <= 0 or tick_d <= 0:
        raise ValueError("Price and tick size must be positive.")
    mode = ROUND_UP if direction == "up" else ROUND_DOWN
    return (value_d / tick_d).to_integral_value(rounding=mode) * tick_d


def choose_side(p_long: float, p_short: float, threshold_long: float, threshold_short: float) -> str | None:
    values = (p_long, p_short, threshold_long, threshold_short)
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError("Probabilities and thresholds must be finite.")
    if any(not 0 <= float(x) <= 1 for x in values):
        raise ValueError("Probabilities and thresholds must lie in [0, 1].")
    long_ok, short_ok = p_long >= threshold_long, p_short >= threshold_short
    if not long_ok and not short_ok:
        return None
    if long_ok and short_ok:
        long_excess = (p_long - threshold_long) / max(1e-9, 1 - threshold_long)
        short_excess = (p_short - threshold_short) / max(1e-9, 1 - threshold_short)
        return "LONG" if long_excess >= short_excess else "SHORT"
    return "LONG" if long_ok else "SHORT"


def sigma_from_history(candles: pd.DataFrame) -> float:
    if len(candles) < VOLATILITY_WINDOW + 2:
        raise ValueError("Need at least 98 completed 15m candles for the H96 sigma.")
    close = pd.to_numeric(candles["close"], errors="coerce")
    returns = np.log(close).diff()
    sigma = returns.rolling(VOLATILITY_WINDOW, min_periods=48).std().iloc[-1] * math.sqrt(HORIZON_BARS)
    value = float(sigma)
    if not math.isfinite(value) or not 0 < value < 0.50:
        raise ValueError(f"Invalid volatility estimate: {value!r}")
    return value


def barrier_prices(side: str, entry: float, sigma: float) -> tuple[float, float]:
    if side not in {"LONG", "SHORT"} or not math.isfinite(entry) or entry <= 0:
        raise ValueError("Invalid side or entry price.")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be positive and finite.")
    if side == "LONG":
        return entry * (1 + TP_SIGMA * sigma), entry * (1 - SL_SIGMA * sigma)
    return entry * (1 - TP_SIGMA * sigma), entry * (1 + SL_SIGMA * sigma)


def quantity_for_cap(notional_cap: float, price: float, step: str, min_qty: str, min_notional: float) -> Decimal:
    if not math.isfinite(notional_cap) or not math.isfinite(price) or notional_cap <= 0 or price <= 0:
        raise ValueError("Notional cap and price must be positive finite values.")
    qty = decimal_floor_step(Decimal(str(notional_cap)) / Decimal(str(price)), step)
    min_q = Decimal(str(min_qty))
    notional = qty * Decimal(str(price))
    if qty < min_q:
        raise ValueError(
            f"Position cap {notional_cap:.2f} USDT is below symbol minimum quantity; "
            f"minimum is about {float(min_q * Decimal(str(price))):.2f} USDT."
        )
    if float(notional) + 1e-8 < min_notional:
        raise ValueError(
            f"Position cap {notional_cap:.2f} USDT is below Binance minimum notional "
            f"{min_notional:.2f} USDT. The bot will not increase the size."
        )
    if float(notional) > notional_cap + 1e-7:
        raise ValueError("Rounded quantity would exceed the configured notional cap.")
    return qty


def candles_from_binance_klines(raw: list[list[Any]]) -> tuple[pd.DataFrame, list[int]]:
    if not raw:
        raise ValueError("Binance returned no klines.")
    rows, close_times = [], []
    for item in raw:
        if len(item) < 11:
            raise ValueError("Unexpected Binance kline response.")
        rows.append({
            "candle_time": pd.to_datetime(int(item[0]), unit="ms", utc=True),
            "open": float(item[1]), "high": float(item[2]), "low": float(item[3]),
            "close": float(item[4]), "volume": float(item[5]),
            "quote_volume": float(item[7]), "trade_count": float(item[8]),
            "taker_buy_base_volume": float(item[9]),
            "taker_buy_quote_volume": float(item[10]),
        })
        close_times.append(int(item[6]))
    frame = pd.DataFrame(rows).sort_values("candle_time").reset_index(drop=True)
    if frame["candle_time"].duplicated().any():
        raise ValueError("Binance returned duplicate 15m candles.")
    if not frame["candle_time"].diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Demo Futures candle series has a 15m gap.")
    return frame, close_times


def symbol_rules(exchange_info: dict, symbol: str) -> dict:
    match = next((x for x in exchange_info.get("symbols", []) if x.get("symbol") == symbol), None)
    if not match or match.get("status") != "TRADING" or match.get("contractType") != "PERPETUAL":
        raise ValueError(f"{symbol} is not an active Demo USDⓈ-M perpetual.")
    filters = {x.get("filterType"): x for x in match.get("filters", [])}
    market_lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE")
    limit_lot = filters.get("LOT_SIZE") or market_lot
    price_filter = filters.get("PRICE_FILTER")
    if not market_lot or not limit_lot or not price_filter:
        raise ValueError("The symbol quantity/price filters are incomplete.")
    min_filter = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {}
    min_notional = float(min_filter.get("notional") or min_filter.get("minNotional") or 0)
    return {
        "market_step_size": market_lot["stepSize"],
        "market_min_qty": market_lot["minQty"],
        "limit_step_size": limit_lot["stepSize"],
        "limit_min_qty": limit_lot["minQty"],
        "tick_size": price_filter["tickSize"],
        "min_notional": min_notional,
    }
