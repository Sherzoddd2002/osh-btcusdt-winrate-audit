"""Frozen, pre-registered configuration for the HF model arena.

All values below are fixed before any forward/demo run. No parameter is tuned
from live results. Same data feed, same fees, same $100 virtual balance for every
model, so the leaderboard is a fair A/B comparison.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ safety
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
BAR_MINUTES = 15
DEMO_FUTURES_URL = "https://testnet.binancefuture.com"  # legacy USD-M Testnet host
# No environment override. No production/live host. No fallback.
LOCAL_PORT = int(os.getenv("OSH_HF_PORT", "8766"))  # 8766 != existing 8765 terminal
PAGES_ORIGIN = "https://sherzoddd2002.github.io"
KEYRING_SERVICE = "OSH_HF_Arena_Binance_Demo"  # separate from H96 C2 keyring entry
CRED_USERS = ("api_key", "api_secret")

INITIAL_CAPITAL_USD = 100.0
MAX_NOTIONAL_USD = 100.0       # hard cap: never above the $100 account
LEVERAGE = 1
MARGIN_TYPE = "ISOLATED"
# Fail-closed research gate: no HF tactic has passed net holdout/forward GO criteria.
# Keep the HF Arena in PAPER/SHADOW mode until a separately reviewed release enables orders.
TESTNET_ORDER_EXECUTION_ENABLED = False
ORDER_EXECUTION_LOCK_REASON = (
    "HF Testnet order execution is locked: all six tactics remain NO-GO after net-cost replay. "
    "Use PAPER/SHADOW mode only until an independently validated forward GO and execution review."
)
MAX_CONCURRENT_PER_MODEL = 8

# Fees are REAL and never removed. Gross is shown separately for transparency.
MAKER_FEE_BPS_PER_SIDE = float(os.getenv("OSH_HF_MAKER_BPS", "2.0"))
TAKER_FEE_BPS_PER_SIDE = float(os.getenv("OSH_HF_TAKER_BPS", "5.0"))
SLIPPAGE_ROUND_TRIP_BPS = float(os.getenv("OSH_HF_SLIP_BPS", "2.0"))
ROUND_TRIP_TAKER_BPS = 2 * TAKER_FEE_BPS_PER_SIDE + SLIPPAGE_ROUND_TRIP_BPS

# Risk guards (hard stops; cannot be disabled from the website)
MAX_DRAWDOWN_PCT = 15.0
MAX_DAILY_LOSS_USD = 10.0
MAX_TRADES_PER_HOUR = 12
KILL_ON_API_ERRORS = 3

# Model artifact (READ ONLY). Never modified by this package.
MODEL_ARTIFACT = ROOT / "forward_paper_h96_c2_model.joblib"
MODEL_LOCK = ROOT / "forward_paper_h96_c2_lock.json"

# Local-only runtime state
RUNTIME_DIR = ROOT / "hf_arena" / "runtime"
LEDGER_PATH = RUNTIME_DIR / "arena_ledger.json"
AUDIT_PATH = RUNTIME_DIR / "arena_events.jsonl"

# ------------------------------------------------------------------ arena
# Each model: name, entry cooldown in bars, TP/SL in sigma, signal selectivity,
# sizing mode, and fee mode. Confidence sizing scales notional by model certainty.
# "net" is the only basis for any GO/NO-GO decision; "gross" is diagnostic only.
MODELS = [
    {
        "id": "A_H96_BASE", "label": "H96 C2 · baseline (1/kun)",
        "cooldown_bars": 96, "tp_sigma": 0.75, "sl_sigma": 1.0,
        "top_pct": 0.10, "sizing": "fixed", "fixed_notional_usd": 50.0,
        "note": "Sizning hozirgi H96 C2 taktikasi — taqqoslash nuqtasi.",
    },
    {
        "id": "B_HF24", "label": "HF-24 · 4/kun",
        "cooldown_bars": 24, "tp_sigma": 0.75, "sl_sigma": 1.0,
        "top_pct": 0.10, "sizing": "confidence", "fixed_notional_usd": 25.0,
        "note": "Ishonchlilikka qarab hajm, 4 savdo/kun.",
    },
    {
        "id": "C_HF12", "label": "HF-12 · 8/kun",
        "cooldown_bars": 12, "tp_sigma": 0.75, "sl_sigma": 1.0,
        "top_pct": 0.10, "sizing": "confidence", "fixed_notional_usd": 12.5,
        "note": "Tez savdo + ishonchli hajm.",
    },
    {
        "id": "D_HF6", "label": "HF-6 · 16/kun",
        "cooldown_bars": 6, "tp_sigma": 0.75, "sl_sigma": 1.0,
        "top_pct": 0.10, "sizing": "confidence", "fixed_notional_usd": 6.25,
        "note": "Eng tez variant; komissiya ta'siri eng kuchli.",
    },
    {
        "id": "E_HF12_TIGHT", "label": "HF-12 · faqat kuchli signallar",
        "cooldown_bars": 12, "tp_sigma": 0.75, "sl_sigma": 1.0,
        "top_pct": 0.05, "sizing": "confidence", "fixed_notional_usd": 12.5,
        "note": "Kam savdo, yuqori ishonch (top 5%).",
    },
    {
        "id": "F_HF12_MAKER", "label": "HF-12 · maker urinish",
        "cooldown_bars": 12, "tp_sigma": 0.75, "sl_sigma": 1.0,
        "top_pct": 0.10, "sizing": "confidence", "fixed_notional_usd": 12.5,
        "prefer_maker": True,
        "note": "Komissiyani kamaytirish uchun limit entry (fill kafolatlanmagan).",
    },
]

# Pre-registered GO criteria applied to the FORWARD period only (never to history).
GO_CRITERIA = {
    "min_forward_trades": 30,
    "min_net_win_rate_pct": 55.0,
    "min_net_profit_factor": 1.10,
    "min_final_balance_usd": 100.0,
    "max_drawdown_pct": 15.0,
    "require_net_positive_after_fees": True,
    "require_positive_in_both_fee_modes": False,
}
