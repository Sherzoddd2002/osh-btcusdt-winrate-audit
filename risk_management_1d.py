"""Development-only trade attribution and conservative risk-budget simulation.

This module does not generate signals or alter trade outcomes. It reports
side/exit attribution and simulates position sizing from pre-trade stop distance.
Risk overlays are intended for development walk-forward OOS trades only.
"""
from __future__ import annotations

import numpy as np


def _aligned_arrays(**values: object) -> dict[str, np.ndarray]:
    arrays = {name: np.asarray(value) for name, value in values.items()}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) > 1:
        raise ValueError("Trade attribution inputs must have equal lengths.")
    return arrays


def _summary(
    net: np.ndarray,
    mae: np.ndarray,
    mfe: np.ndarray,
    holding_bars: np.ndarray,
) -> dict:
    n = int(len(net))
    if n == 0:
        return {
            "trades": 0,
            "win_rate_pct": None,
            "expectancy_bps": None,
            "profit_factor": None,
            "avg_win_bps": None,
            "avg_loss_bps": None,
            "payoff_ratio": None,
            "avg_mae_bps": None,
            "worst_mae_bps": None,
            "avg_mfe_bps": None,
            "avg_holding_bars": None,
        }

    winners = net > 0
    losers = ~winners
    win_sum = float(net[winners].sum())
    loss_sum = float(-net[losers].sum())
    avg_win = float(net[winners].mean()) if winners.any() else None
    avg_loss = float(-net[losers].mean()) if losers.any() else None
    return {
        "trades": n,
        "wins": int(winners.sum()),
        "losses_or_breakeven": int(losers.sum()),
        "win_rate_pct": round(float(winners.mean() * 100.0), 3),
        "expectancy_bps": round(float(net.mean() * 10_000.0), 3),
        "profit_factor": round(win_sum / loss_sum, 5) if loss_sum > 0 else None,
        "avg_win_bps": round(avg_win * 10_000.0, 3) if avg_win is not None else None,
        "avg_loss_bps": round(avg_loss * 10_000.0, 3) if avg_loss is not None else None,
        "payoff_ratio": round(avg_win / avg_loss, 5)
        if avg_win is not None and avg_loss is not None and avg_loss > 0 else None,
        "avg_mae_bps": round(float(mae.mean() * 10_000.0), 3),
        "worst_mae_bps": round(float(mae.min() * 10_000.0), 3),
        "avg_mfe_bps": round(float(mfe.mean() * 10_000.0), 3),
        "avg_holding_bars": round(float(holding_bars.mean()), 3),
    }


def summarize_trade_attribution(
    net_returns: object,
    sides: object,
    exit_reasons: object,
    mae_returns: object,
    mfe_returns: object,
    holding_bars: object,
) -> dict:
    """Summarize net results by LONG/SHORT and by first-touch exit reason.

    Returns are decimal fractions relative to entry notional. MAE is the
    worst directional excursion during the realized holding interval.
    """
    arrays = _aligned_arrays(
        net_returns=net_returns,
        sides=sides,
        exit_reasons=exit_reasons,
        mae_returns=mae_returns,
        mfe_returns=mfe_returns,
        holding_bars=holding_bars,
    )
    net = arrays["net_returns"].astype(float)
    sides_arr = arrays["sides"].astype(int)
    reasons = arrays["exit_reasons"].astype(str)
    mae = arrays["mae_returns"].astype(float)
    mfe = arrays["mfe_returns"].astype(float)
    holding = arrays["holding_bars"].astype(float)

    if not np.isfinite(net).all() or not np.isfinite(mae).all():
        raise ValueError("Trade returns and MAE must be finite.")
    if not np.isfinite(mfe).all() or not np.isfinite(holding).all():
        raise ValueError("MFE and holding bars must be finite.")
    if not np.isin(sides_arr, (-1, 1)).all():
        raise ValueError("Trade sides must be -1 (SHORT) or +1 (LONG).")
    if (holding <= 0).any():
        raise ValueError("Every executed trade must have positive holding bars.")

    result = _summary(net, mae, mfe, holding)
    result["by_side"] = {
        label: _summary(
            net[sides_arr == side],
            mae[sides_arr == side],
            mfe[sides_arr == side],
            holding[sides_arr == side],
        )
        for label, side in (("LONG", 1), ("SHORT", -1))
    }
    result["by_exit_reason"] = {
        reason: _summary(
            net[reasons == reason],
            mae[reasons == reason],
            mfe[reasons == reason],
            holding[reasons == reason],
        )
        for reason in sorted(set(reasons.tolist()))
    }
    result["exit_reason_share_pct"] = {
        reason: round(float(np.mean(reasons == reason) * 100.0), 3)
        for reason in sorted(set(reasons.tolist()))
    }
    return result


def simulate_risk_budget(
    net_returns: object,
    mae_returns: object,
    stop_distance_returns: object,
    entry_dates: object,
    *,
    risk_per_trade_pct: float,
    initial_capital: float = 100.0,
    max_notional: float = 100.0,
    cash_reserve_pct: float = 10.0,
    cost_buffer_bps: float = 6.0,
    daily_loss_limit_pct: float = 2.0,
    max_drawdown_stop_pct: float = 15.0,
    max_consecutive_losses: int = 3,
    cooldown_trades: int = 1,
) -> dict:
    """Simulate stop-distance sizing and predeclared account-level risk brakes.

    risk_per_trade_pct is a fraction (0.005 means 0.5%). No leverage is used.
    The trade sequence is assumed to be non-overlapping and chronological.
    The risk overlay changes dollar sizing, not the underlying signal edge.
    """
    arrays = _aligned_arrays(
        net_returns=net_returns,
        mae_returns=mae_returns,
        stop_distance_returns=stop_distance_returns,
        entry_dates=entry_dates,
    )
    net = arrays["net_returns"].astype(float)
    mae = arrays["mae_returns"].astype(float)
    stops = arrays["stop_distance_returns"].astype(float)
    dates = np.asarray([str(x)[:10] for x in arrays["entry_dates"]], dtype=str)

    if not np.isfinite(net).all() or not np.isfinite(mae).all():
        raise ValueError("Net returns and MAE must be finite.")
    if not np.isfinite(stops).all() or (stops <= 0).any():
        raise ValueError("Every trade must have a finite positive stop distance.")
    if (mae > 1e-12).any():
        raise ValueError("MAE must be a non-positive adverse return.")
    if len(dates) > 1 and (dates[1:] < dates[:-1]).any():
        raise ValueError("Risk simulation trades must be in chronological order.")
    if not 0 < risk_per_trade_pct <= 0.02:
        raise ValueError("risk_per_trade_pct must be in (0, 0.02].")
    if initial_capital <= 0 or max_notional <= 0:
        raise ValueError("Capital and max_notional must be positive.")
    if not 0 <= cash_reserve_pct < 100:
        raise ValueError("cash_reserve_pct must be in [0, 100).")
    if cost_buffer_bps < 0:
        raise ValueError("cost_buffer_bps must be non-negative.")
    if not 0 < daily_loss_limit_pct <= 10:
        raise ValueError("daily_loss_limit_pct must be in (0, 10].")
    if not 0 < max_drawdown_stop_pct < 100:
        raise ValueError("max_drawdown_stop_pct must be in (0, 100).")
    if max_consecutive_losses < 1 or cooldown_trades < 0:
        raise ValueError("Loss-streak and cooldown settings are invalid.")

    equity = float(initial_capital)
    peak = equity
    max_closed_dd = 0.0
    max_intratrade_dd = 0.0
    current_day = None
    day_start_equity = equity
    day_realized_pnl = 0.0
    daily_brake_active = False
    halted_by_drawdown = False
    loss_streak = max_loss_streak = 0
    cooldown_left = 0
    daily_brake_days = 0
    skipped_daily = skipped_cooldown = skipped_drawdown = 0
    notionals: list[float] = []
    dollar_risks: list[float] = []
    executed_returns: list[float] = []
    pnl_total = 0.0
    cost_buffer = cost_buffer_bps / 10_000.0

    for i, (ret, adverse, stop, day) in enumerate(zip(net, mae, stops, dates)):
        if day != current_day:
            current_day = day
            day_start_equity = equity
            day_realized_pnl = 0.0
            daily_brake_active = False

        if halted_by_drawdown:
            skipped_drawdown += 1
            continue
        if daily_brake_active:
            skipped_daily += 1
            continue
        if cooldown_left > 0:
            cooldown_left -= 1
            skipped_cooldown += 1
            continue
        if equity <= 0:
            halted_by_drawdown = True
            skipped_drawdown += 1
            continue

        risk_budget_usd = equity * risk_per_trade_pct
        risk_fraction = stop + cost_buffer
        notional = min(
            max_notional,
            equity * (1.0 - cash_reserve_pct / 100.0),
            risk_budget_usd / risk_fraction,
        )
        if notional <= 0 or not np.isfinite(notional):
            raise ValueError("Risk sizing produced a non-positive or non-finite notional.")

        planned_stop_risk = notional * stop
        dollar_risks.append(planned_stop_risk)
        notionals.append(notional)

        trough_equity = equity + notional * min(float(adverse), 0.0) - notional * cost_buffer
        max_intratrade_dd = max(
            max_intratrade_dd,
            max(0.0, 1.0 - max(0.0, trough_equity) / max(peak, 1e-12)),
        )

        pnl = notional * float(ret)
        equity += pnl
        pnl_total += pnl
        day_realized_pnl += pnl
        executed_returns.append(float(ret))
        peak = max(peak, equity)
        if peak > 0:
            max_closed_dd = max(max_closed_dd, max(0.0, (peak - equity) / peak))

        if pnl < 0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

        if day_realized_pnl <= -day_start_equity * daily_loss_limit_pct / 100.0:
            daily_brake_active = True
            daily_brake_days += 1
        if peak > 0 and (peak - equity) / peak >= max_drawdown_stop_pct / 100.0:
            halted_by_drawdown = True
        if loss_streak >= max_consecutive_losses:
            cooldown_left = max(cooldown_left, cooldown_trades)
            loss_streak = 0

    executed = np.asarray(executed_returns, dtype=float)
    return {
        "risk_per_trade_pct": round(risk_per_trade_pct * 100.0, 3),
        "initial_capital": float(initial_capital),
        "max_notional": float(max_notional),
        "cash_reserve_pct": float(cash_reserve_pct),
        "daily_loss_limit_pct": float(daily_loss_limit_pct),
        "max_drawdown_stop_pct": float(max_drawdown_stop_pct),
        "max_consecutive_losses_before_cooldown": int(max_consecutive_losses),
        "cooldown_trades": int(cooldown_trades),
        "signals_seen": int(len(net)),
        "executed_trades": int(len(executed)),
        "skipped_daily_loss_brake": int(skipped_daily),
        "skipped_loss_streak_cooldown": int(skipped_cooldown),
        "skipped_after_drawdown_halt": int(skipped_drawdown),
        "daily_loss_brake_days": int(daily_brake_days),
        "drawdown_halt_triggered": bool(halted_by_drawdown),
        "ending_equity": round(float(equity), 4),
        "net_pnl_usd": round(float(pnl_total), 4),
        "return_pct": round((equity / initial_capital - 1.0) * 100.0, 4),
        "max_closed_trade_drawdown_pct": round(max_closed_dd * 100.0, 4),
        "max_intratrade_mae_drawdown_pct": round(max_intratrade_dd * 100.0, 4),
        "max_consecutive_losses": int(max_loss_streak),
        "mean_position_notional_usd": round(float(np.mean(notionals)), 4) if notionals else None,
        "mean_planned_stop_risk_usd": round(float(np.mean(dollar_risks)), 4) if dollar_risks else None,
        "executed_trade_win_rate_pct": round(float(np.mean(executed > 0) * 100.0), 3) if len(executed) else None,
        "executed_trade_expectancy_bps": round(float(executed.mean() * 10_000.0), 3) if len(executed) else None,
        "base_signal_win_rate_pct": round(float(np.mean(net > 0) * 100.0), 3) if len(net) else None,
        "base_signal_expectancy_bps": round(float(net.mean() * 10_000.0), 3) if len(net) else None,
        "note": (
            "Development OOS risk-budget simulation only; sizing and brakes do not "
            "improve the underlying per-trade expectancy. No leverage or live orders."
        ),
    }


def run_risk_budget_grid(
    net_returns: object,
    mae_returns: object,
    stop_distance_returns: object,
    entry_dates: object,
    *,
    risk_levels_pct: tuple[float, ...] = (0.25, 0.5, 1.0),
    initial_capital: float = 100.0,
    max_notional: float = 100.0,
    cash_reserve_pct: float = 10.0,
    cost_buffer_bps: float = 6.0,
    daily_loss_limit_pct: float = 2.0,
    max_drawdown_stop_pct: float = 15.0,
    max_consecutive_losses: int = 3,
    cooldown_trades: int = 1,
) -> dict:
    """Run a preregistered sizing grid; caller must supply development data only."""
    levels = tuple(float(x) for x in risk_levels_pct)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("risk_levels_pct must be non-empty and unique.")
    if any(not 0 < x <= 2.0 for x in levels):
        raise ValueError("Risk levels must be percentages in (0, 2].")
    return {
        f"risk_{level:g}pct": simulate_risk_budget(
            net_returns,
            mae_returns,
            stop_distance_returns,
            entry_dates,
            risk_per_trade_pct=level / 100.0,
            initial_capital=initial_capital,
            max_notional=max_notional,
            cash_reserve_pct=cash_reserve_pct,
            cost_buffer_bps=cost_buffer_bps,
            daily_loss_limit_pct=daily_loss_limit_pct,
            max_drawdown_stop_pct=max_drawdown_stop_pct,
            max_consecutive_losses=max_consecutive_losses,
            cooldown_trades=cooldown_trades,
        )
        for level in levels
    }
