"""Execution-cost scenarios for BTCUSDT USDⓈ-M Futures (paper research only).

All rates are in basis points (bps) of notional. Entry commission is charged
on entry notional and exit commission on exit notional.
"""
from __future__ import annotations

from typing import Mapping
import numpy as np

DEFAULT_MAKER_FEE_BPS = 2.0
DEFAULT_TAKER_FEE_BPS = 5.0
DEFAULT_ROUND_TRIP_SLIPPAGE_BPS = 2.0

SCENARIO_LEGACY_MAKER_MAKER = "legacy_maker_entry_maker_exit"
SCENARIO_TAKER_ENTRY_TP_MAKER = "taker_entry_tp_maker_stop_time_taker"
SCENARIO_TAKER_ALL = "taker_entry_taker_exit"

# Stable schema used by the fold-level aggregation and reporting layer.
SCENARIO_COMPONENT_KEYS = (
    "gross_return",
    "funding_return",
    "entry_commission_return",
    "exit_commission_return",
    "slippage_return",
    "commission_return",
    "total_cost_return",
    "net_return",
)


def _validate_rates(maker_fee_bps: float, taker_fee_bps: float, slippage_bps: float) -> None:
    rates = (maker_fee_bps, taker_fee_bps, slippage_bps)
    if not all(np.isfinite(x) and x >= 0 for x in rates):
        raise ValueError("Fee and slippage rates must be finite, non-negative bps values.")


def calculate_trade_costs(
    gross_return: float,
    side: int,
    exit_reason: str,
    *,
    entry_fee_bps: float,
    exit_fee_bps: float,
    round_trip_slippage_bps: float = DEFAULT_ROUND_TRIP_SLIPPAGE_BPS,
    bnb_discount_pct: float = 0.0,
) -> dict[str, float]:
    """Return per-trade cost fractions relative to entry notional."""
    _validate_rates(entry_fee_bps, exit_fee_bps, round_trip_slippage_bps)
    if not np.isfinite(bnb_discount_pct) or not 0 <= bnb_discount_pct <= 100:
        raise ValueError("bnb_discount_pct must be between 0 and 100.")
    discount = 1.0 - bnb_discount_pct / 100.0
    if side not in (-1, 1):
        raise ValueError("side must be +1 (long) or -1 (short).")
    if exit_reason not in {"take_profit", "stop_loss", "time"}:
        raise ValueError(f"Unsupported exit reason: {exit_reason!r}")
    if not np.isfinite(gross_return):
        raise ValueError("gross_return must be finite.")
    exit_ratio = 1.0 + gross_return if side == 1 else 1.0 - gross_return
    if not np.isfinite(exit_ratio) or exit_ratio <= 0:
        raise ValueError("Exit notional ratio must be positive.")
    entry_fee = entry_fee_bps * discount / 10_000.0
    exit_fee = exit_fee_bps * exit_ratio * discount / 10_000.0
    slippage = round_trip_slippage_bps / 10_000.0
    return {
        "entry_commission_return": float(entry_fee),
        "exit_commission_return": float(exit_fee),
        "slippage_return": float(slippage),
        "total_cost_return": float(entry_fee + exit_fee + slippage),
        "exit_notional_ratio": float(exit_ratio),
    }


def build_execution_scenarios(
    gross_returns: np.ndarray,
    sides: np.ndarray,
    exit_reasons: np.ndarray,
    funding_returns: np.ndarray,
    *,
    maker_fee_bps: float = DEFAULT_MAKER_FEE_BPS,
    taker_fee_bps: float = DEFAULT_TAKER_FEE_BPS,
    round_trip_slippage_bps: float = DEFAULT_ROUND_TRIP_SLIPPAGE_BPS,
    bnb_discount_pct: float = 0.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Return legacy, conditional TP-maker, and all-taker net trade arrays.

    The TP-maker case is conditional: OHLC bars do not prove queue priority or a
    maker fill. It is not a guaranteed limit-order execution model.
    """
    _validate_rates(maker_fee_bps, taker_fee_bps, round_trip_slippage_bps)
    if not np.isfinite(bnb_discount_pct) or not 0 <= bnb_discount_pct <= 100:
        raise ValueError("bnb_discount_pct must be between 0 and 100.")
    discount = 1.0 - bnb_discount_pct / 100.0
    gross = np.asarray(gross_returns, dtype=float)
    side = np.asarray(sides, dtype=np.int8)
    reasons = np.asarray(exit_reasons, dtype=object)
    funding = np.asarray(funding_returns, dtype=float)
    if not (len(gross) == len(side) == len(reasons) == len(funding)):
        raise ValueError("gross_returns, sides, exit_reasons and funding_returns must align.")
    if not np.isfinite(gross).all() or not np.isfinite(funding).all():
        raise ValueError("Trade returns must be finite.")
    if len(side) and not np.isin(side, (-1, 1)).all():
        raise ValueError("Every side must be +1 or -1.")
    if len(reasons) and not np.isin(reasons, ("take_profit", "stop_loss", "time")).all():
        raise ValueError("Unknown exit reason in trade array.")

    profiles: Mapping[str, tuple[float, np.ndarray]] = {
        SCENARIO_LEGACY_MAKER_MAKER: (
            maker_fee_bps, np.full(len(gross), maker_fee_bps, dtype=float)
        ),
        SCENARIO_TAKER_ENTRY_TP_MAKER: (
            taker_fee_bps, np.where(reasons == "take_profit", maker_fee_bps, taker_fee_bps)
        ),
        SCENARIO_TAKER_ALL: (
            taker_fee_bps, np.full(len(gross), taker_fee_bps, dtype=float)
        ),
    }
    output: dict[str, dict[str, np.ndarray]] = {}
    for name, (entry_rate, exit_rates) in profiles.items():
        entry_fee = np.full(len(gross), entry_rate * discount / 10_000.0, dtype=float)
        exit_ratio = np.where(side == 1, 1.0 + gross, 1.0 - gross)
        if len(exit_ratio) and ((~np.isfinite(exit_ratio)).any() or (exit_ratio <= 0).any()):
            raise ValueError("Exit notional ratio must be positive and finite.")
        exit_fee = exit_rates * exit_ratio * discount / 10_000.0
        slippage = np.full(len(gross), round_trip_slippage_bps / 10_000.0, dtype=float)
        total_cost = entry_fee + exit_fee + slippage
        output[name] = {
            "gross_return": gross.copy(),
            "funding_return": funding.copy(),
            "entry_commission_return": entry_fee,
            "exit_commission_return": exit_fee,
            "slippage_return": slippage,
            "commission_return": entry_fee + exit_fee,
            "total_cost_return": total_cost,
            "net_return": gross - total_cost + funding,
        }
    return output
