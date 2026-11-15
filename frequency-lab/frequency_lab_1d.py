"""OSH Frequency & Capital Lab — PRE-REGISTERED, PAPER ONLY, SEPARATE ARTIFACT.

Question: can we get >=1 trade/day AND positive $100 PnL WITHOUT leverage,
without touching the frozen H96 C2 forward-paper strategy or the Testnet terminal?

Mechanism: entry cooldown. H96 C2 uses cooldown=96 bars (max 1 trade/day).
Shortening it allows overlapping 1d-hold positions. To respect $100 / 1x,
each position is sized at capital * cooldown / horizon, so maximum concurrent
notional never exceeds $100. This isolates frequency from leverage.

Pre-registered grid (5 configs, all reported, holdout touched once each,
NO re-tuning after seeing holdout): cooldown 96, 48, 24, 12, 6 bars.

Fixed for all configs: tp=0.75 sigma, sl=1.0 sigma, h=96, top_pct=10%/side,
frozen train-calibrated thresholds, 37 BASE features, 5 purged WF folds
(365d train / 90d test), 180d untouched holdout, real funding PnL,
and the same fee model as the corrected high-winrate-hunt framework.

Reads NO API keys, sends NO orders, imports NO engine/lock/model artifact.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline_v2 as pv  # noqa: E402
import research_v3 as rv  # noqa: E402
import high_winrate_hunt_1d as hw  # noqa: E402

H = 96
TP_SIGMA, SL_SIGMA = 0.75, 1.0
BARS_PER_DAY = 96
FOLDS = 5
TRAIN_DAYS, TEST_DAYS = 365, 90
HOLDOUT_DAYS = 180
DATA_END_UTC = pd.Timestamp("2026-10-31 23:45:00", tz="UTC")
CAPITAL_USD = 100.0
COOLDOWNS = [96, 48, 24, 12, 6]  # PRE-REGISTERED, fixed before running
FUNDING_CSV = HERE / "binance_btcusdt_funding_history.csv"


def max_concurrent(indices, holding):
    if len(indices) == 0:
        return 0
    ev = []
    for i, hb in zip(indices, holding):
        ev.append((int(i) + 1, 1))
        ev.append((int(i) + int(hb) + 1, -1))
    ev.sort()
    cur = mx = 0
    for _, d in ev:
        cur += d
        mx = max(mx, cur)
    return mx


def capital_path(net, frac):
    """Fixed-fractional sizing: equity *= (1 + net_i * frac). No leverage."""
    bal, peak, max_dd = CAPITAL_USD, CAPITAL_USD, 0.0
    for r in net:
        bal *= (1.0 + float(r) * frac)
        peak = max(peak, bal)
        max_dd = max(max_dd, (peak - bal) / peak)
    return bal, max_dd * 100.0


def summarize(net, gross, indices, holding, reasons, sides, frac, n_bars, benchmark):
    net = np.asarray(net, float)
    if len(net) == 0:
        return {"trades": 0}
    win = net > 0
    pos, neg = float(net[win].sum()), float(-net[~win].sum())
    bal, dd = capital_path(net, frac)
    conc = max_concurrent(indices, holding)
    lo, hi = hw.block_bootstrap_ci(net)
    bmean = float(np.mean(benchmark) * 1e4) if len(benchmark) else None
    return {
        "trades": int(len(net)),
        "trades_per_day": round(len(net) / max(1e-9, n_bars / BARS_PER_DAY), 3),
        "win_rate_pct": round(float(win.mean() * 100), 2),
        "expectancy_bps": round(float(net.mean() * 1e4), 3),
        "avg_win_net_bps": round(float(net[win].mean() * 1e4), 2) if win.any() else 0.0,
        "avg_loss_net_bps": round(float(-net[~win].mean() * 1e4), 2) if (~win).any() else 0.0,
        "payoff_ratio": round(float(net[win].mean() / -net[~win].mean()), 3) if win.any() and (~win).any() else None,
        "profit_factor": round(pos / neg, 4) if neg > 0 else None,
        "benchmark_mean_net_bps": round(bmean, 3) if bmean is not None else None,
        "position_notional_usd": round(CAPITAL_USD * frac, 2),
        "max_concurrent_positions": int(conc),
        "max_notional_usd": round(conc * CAPITAL_USD * frac, 2),
        "final_balance_100usd": round(float(bal), 2),
        "pnl_usd": round(float(bal - CAPITAL_USD), 2),
        "max_drawdown_pct": round(float(dd), 2),
        "bootstrap_ci95_bps": [
            round(lo, 3) if lo is not None else None,
            round(hi, 3) if hi is not None else None,
        ],
        "by_reason": {rs: int((np.asarray(reasons) == rs).sum())
                      for rs in ("take_profit", "stop_loss", "time")},
        "long_trades": int((np.asarray(sides) == 1).sum()),
        "short_trades": int((np.asarray(sides) == -1).sum()),
    }


def main():
    print("=== OSH FREQUENCY & CAPITAL LAB — PRE-REGISTERED, PAPER ONLY ===", flush=True)
    print(f"tp={TP_SIGMA}sigma sl={SL_SIGMA}sigma h={H} | top={hw.TOP_PCT}/side | "
          f"cooldowns={COOLDOWNS} | ${CAPITAL_USD:.0f}, 1x, no leverage", flush=True)

    df = rv.load_btc("postgresql+psycopg2://postgres@127.0.0.1:5433/osh", "public.btcusdt_15m")
    df = df[df["candle_time"] <= DATA_END_UTC].reset_index(drop=True)
    funding = hw.load_funding(FUNDING_CSV)
    fmarks = hw.funding_mark_price_proxy(df, funding)
    ft = fmarks["funding_time"].values.astype("datetime64[ns]").astype(np.int64)
    fr = fmarks["funding_rate"].to_numpy(float)
    fm = fmarks["mark_price_proxy"].to_numpy(float)
    print(f"data: {len(df)} rows to {df['candle_time'].iloc[-1]} | funding {len(fmarks)}", flush=True)

    X_base, _ = pv.make_features(df)
    X_base = X_base.replace([np.inf, -np.inf], np.nan)
    labels = hw.asymmetric_barrier_labels(df, H, TP_SIGMA, SL_SIGMA,
                                          volatility_horizon=H)
    valid, warmup_rows = hw._contiguous_label_valid_mask(labels, X_base)
    labels = labels.loc[valid].reset_index(drop=True)
    times = df.loc[valid, "candle_time"].reset_index(drop=True)
    deltas = times.diff().dropna()
    if not deltas.eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Label-valid timeline is not contiguous 15m bars; refusing to compress time.")
    X = X_base.loc[valid].reset_index(drop=True)
    print(f"[features] warmup_excluded={warmup_rows} retained={int(valid.sum())} "
          f"retained_nan_rows={int(X.isna().any(axis=1).sum())}", flush=True)

    dev_cut_time = df["candle_time"].iloc[len(df) - BARS_PER_DAY * HOLDOUT_DAYS]
    cut_idx = int((times < dev_cut_time).sum())
    fold_list = pv.make_folds(cut_idx, BARS_PER_DAY * TRAIN_DAYS,
                              BARS_PER_DAY * TEST_DAYS, H + 2, FOLDS)
    if len(fold_list) != FOLDS:
        raise ValueError(f"Expected {FOLDS} folds, got {len(fold_list)}")
    print(f"dev rows={cut_idx} holdout rows={len(X)-cut_idx} holdout_start={dev_cut_time}", flush=True)

    y_long = np.where(labels["long_label"].to_numpy() == -1, 0,
                      np.where(labels["long_label"].to_numpy() == 0, 1, 2)).astype(np.int8)
    y_short = np.where(labels["short_label"].to_numpy() == -1, 0,
                       np.where(labels["short_label"].to_numpy() == 0, 1, 2)).astype(np.int8)
    w_long = labels["long_sample_weight"].to_numpy(float)
    w_short = labels["short_sample_weight"].to_numpy(float)
    funding_arrays = (ft, fr, fm)

    # ---- train once per fold; reuse predictions across all cooldown configs
    blocks = []
    for fold_no, (tr0, tr1, te0, te1) in enumerate(fold_list, 1):
        cal0 = max(tr0 + 100, tr1 - BARS_PER_DAY * min(90, TRAIN_DAYS))
        ml, cls_l, ms, cls_s, thr_l, thr_s = hw._train_predict(
            X, y_long, y_short, w_long, w_short, tr0, cal0, tr1, horizon_bars=H)
        lab_te = labels.iloc[te0:te1].reset_index(drop=True)
        t_te = times.iloc[te0:te1].reset_index(drop=True)
        p_l = hw.predict_side_proba(ml, cls_l, X.iloc[te0:te1])[:, 2]
        p_s = hw.predict_side_proba(ms, cls_s, X.iloc[te0:te1])[:, 2]
        bench = hw._long_benchmark(lab_te, t_te, ft, fr, fm, stress=False)
        blocks.append({"fold": fold_no, "p_l": p_l, "p_s": p_s,
                       "thr_l": thr_l, "thr_s": thr_s,
                       "lab": lab_te, "times": t_te, "n_bars": te1 - te0,
                       "benchmark": bench})
        print(f"fold {fold_no}: thresholds long={thr_l:.5f} short={thr_s:.5f}", flush=True)

    # ---- holdout: frozen final development model, evaluated once per config
    tr0, tr1, _, _ = fold_list[-1]
    cal0 = max(tr0 + 100, tr1 - BARS_PER_DAY * min(90, TRAIN_DAYS))
    ml, cls_l, ms, cls_s, thr_l_h, thr_s_h = hw._train_predict(
        X, y_long, y_short, w_long, w_short, tr0, cal0, tr1, horizon_bars=H)
    lab_h = labels.iloc[cut_idx:].reset_index(drop=True)
    t_h = times.iloc[cut_idx:].reset_index(drop=True)
    hold_block = {
        "p_l": hw.predict_side_proba(ml, cls_l, X.iloc[cut_idx:])[:, 2],
        "p_s": hw.predict_side_proba(ms, cls_s, X.iloc[cut_idx:])[:, 2],
        "thr_l": thr_l_h, "thr_s": thr_s_h,
        "lab": lab_h, "times": t_h, "n_bars": len(X) - cut_idx,
        "benchmark": hw._long_benchmark(lab_h, t_h, ft, fr, fm, stress=False),
    }

    results = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "OSH-FREQUENCY-CAPITAL-LAB-2026-11-15-v1",
            "separate_artifact": True,
            "touches_forward_paper_engine": False,
            "touches_h96_c2_lock_or_model": False,
            "paper_only": True, "live_orders": False,
            "data_end_utc": str(DATA_END_UTC),
            "holdout_start_utc": str(dev_cut_time),
            "holdout_days": HOLDOUT_DAYS,
            "rows": int(len(df)),
            "preregistered": {
                "cooldown_bars": COOLDOWNS,
                "barrier": f"tp={TP_SIGMA}sigma sl={SL_SIGMA}sigma h={H}",
                "selection": f"frozen train-tail top {hw.TOP_PCT} per side",
                "position_rule": "notional = capital * cooldown / horizon; 1x, no leverage",
                "holdout_policy": "each config evaluated once; no selection or retuning on holdout",
                "fee_model": "maker 2bps/side + 2bps slip (primary); taker 5bps/side (stress)",
            },
        },
        "configs": {}, "gates": [],
    }

    for c in COOLDOWNS:
        frac = c / H
        name = f"C{c:02d}_max{96//c:02d}perday"
        print(f"\n--- {name}: cooldown={c} bars, notional=${CAPITAL_USD*frac:.2f} ---", flush=True)
        fold_rows, fold_pool = [], {"net": [], "gross": [], "idx": [], "hold": [],
                                    "reason": [], "side": [], "bench": []}
        for b in blocks:
            ex = hw._execute_signals(
                b["p_l"], b["p_s"], b["thr_l"], b["thr_s"], b["lab"], b["times"],
                ft, fr, fm, SL_SIGMA, horizon_bars=H, entry_cooldown_bars=c)
            s = summarize(ex["net_primary"], ex["gross"], ex["indices"],
                          ex["holding_bars"], ex["exit_reasons"], ex["sides"],
                          frac, b["n_bars"], b["benchmark"])
            s["fold"] = b["fold"]
            fold_rows.append(s)
            fold_pool["net"].extend(ex["net_primary"].tolist())
            fold_pool["gross"].extend(ex["gross"].tolist())
            fold_pool["idx"].extend(ex["indices"].tolist())
            fold_pool["hold"].extend(ex["holding_bars"].tolist())
            fold_pool["reason"].extend(ex["exit_reasons"].tolist())
            fold_pool["side"].extend(ex["sides"].tolist())
            fold_pool["bench"].extend(b["benchmark"].tolist())
            print(f"  fold {b['fold']}: trades={s.get('trades')} wr={s.get('win_rate_pct')}% "
                  f"exp={s.get('expectancy_bps')}bps", flush=True)

        oos = summarize(np.asarray(fold_pool["net"]), np.asarray(fold_pool["gross"]),
                        np.asarray(fold_pool["idx"]), np.asarray(fold_pool["hold"]),
                        np.asarray(fold_pool["reason"]), np.asarray(fold_pool["side"]),
                        frac, sum(b["n_bars"] for b in blocks), np.asarray(fold_pool["bench"]))
        fold_exps = [r["expectancy_bps"] for r in fold_rows
                     if r.get("expectancy_bps") is not None]
        pos_folds = float(np.mean(np.asarray(fold_exps) > 0) * 100) if fold_exps else 0.0

        ex_h = hw._execute_signals(
            hold_block["p_l"], hold_block["p_s"], hold_block["thr_l"], hold_block["thr_s"],
            hold_block["lab"], hold_block["times"], ft, fr, fm, SL_SIGMA,
            horizon_bars=H, entry_cooldown_bars=c)
        hold = summarize(ex_h["net_primary"], ex_h["gross"], ex_h["indices"],
                         ex_h["holding_bars"], ex_h["exit_reasons"], ex_h["sides"],
                         frac, hold_block["n_bars"], hold_block["benchmark"])
        hold_stress = summarize(ex_h["net_stress"], ex_h["gross"], ex_h["indices"],
                                ex_h["holding_bars"], ex_h["exit_reasons"], ex_h["sides"],
                                frac, hold_block["n_bars"], hold_block["benchmark"])

        checks = {
            "trades_per_day_ge_1": bool(oos.get("trades_per_day") is not None and oos["trades_per_day"] >= 1.0),
            "oos_expectancy_positive": bool(oos.get("expectancy_bps") is not None and oos["expectancy_bps"] > 0),
            "oos_pf_ge_1_2": bool(oos.get("profit_factor") is not None and oos["profit_factor"] >= 1.2),
            "oos_bootstrap_lower_gt_0": bool(oos["bootstrap_ci95_bps"][0] is not None and oos["bootstrap_ci95_bps"][0] > 0),
            "positive_folds_ge_80": bool(pos_folds >= 80),
            "holdout_expectancy_positive": bool(hold.get("expectancy_bps") is not None and hold["expectancy_bps"] > 0),
            "holdout_pf_gt_1": bool(hold.get("profit_factor") is not None and hold["profit_factor"] > 1.0),
            "holdout_100usd_profit": bool(hold.get("final_balance_100usd") is not None and hold["final_balance_100usd"] > CAPITAL_USD),
            "max_drawdown_le_15pct": bool(hold.get("max_drawdown_pct") is not None and hold["max_drawdown_pct"] <= 15.0),
            "max_notional_le_100usd": bool(hold.get("max_notional_usd") is not None and hold["max_notional_usd"] <= CAPITAL_USD + 1e-6),
            "holdout_stress_expectancy_positive": bool(hold_stress.get("expectancy_bps") is not None and hold_stress["expectancy_bps"] > 0),
        }
        verdict = "GO" if all(checks.values()) else "NO-GO"
        results["configs"][name] = {
            "cooldown_bars": c, "max_trades_per_day": 96 // c, "position_fraction": frac,
            "folds": fold_rows, "positive_folds_pct": round(pos_folds, 1),
            "pooled_oos_primary": oos,
            "untouched_holdout_primary": hold,
            "untouched_holdout_all_taker": hold_stress,
        }
        results["gates"].append({"config": name, "checks": checks,
                                 "checks_passed": int(sum(checks.values())),
                                 "checks_total": len(checks), "verdict": verdict})
        print(f"  => OOS {oos.get('trades')} savdo / {oos.get('trades_per_day')}/kun | "
              f"WR {oos.get('win_rate_pct')}% exp {oos.get('expectancy_bps')}bps $100->{oos.get('final_balance_100usd')}", flush=True)
        print(f"  => HOLD {hold.get('trades')} savdo / {hold.get('trades_per_day')}/kun | "
              f"WR {hold.get('win_rate_pct')}% exp {hold.get('expectancy_bps')}bps $100->{hold.get('final_balance_100usd')} "
              f"DD {hold.get('max_drawdown_pct')}% maxnotional ${hold.get('max_notional_usd')} | "
              f"{verdict} ({sum(checks.values())}/{len(checks)})", flush=True)

    out = HERE / "frequency_lab_1d.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"\n[report] {out}", flush=True)
    print("\n=== FREQUENCY FRONTIER ===", flush=True)
    print(f"{'config':<22} {'tr/d':>7} {'OOS exp':>9} {'OOS $100':>10} "
          f"{'HLD exp':>9} {'HLD $100':>10} {'DD%':>7} verdict", flush=True)
    for g in results["gates"]:
        s = results["configs"][g["config"]]
        o, h_ = s["pooled_oos_primary"], s["untouched_holdout_primary"]
        print(f"{g['config']:<22} {str(o.get('trades_per_day')):>7} "
              f"{str(o.get('expectancy_bps')):>9} {str(o.get('final_balance_100usd')):>10} "
              f"{str(h_.get('expectancy_bps')):>9} {str(h_.get('final_balance_100usd')):>10} "
              f"{str(h_.get('max_drawdown_pct')):>7} {g['verdict']} "
              f"({g['checks_passed']}/{g['checks_total']})", flush=True)


if __name__ == "__main__":
    main()
