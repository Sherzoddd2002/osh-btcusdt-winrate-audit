# OSH Quant Lab · BTCUSDT 15m

**Current research snapshot: H96 C2 · 2026-11-14.** This project contains a static, order-free research dashboard and reproducible local paper-research code. No real exchange orders are sent and no capital is at risk from this repository.

> **Research status: DISCOVERY ONLY / NO-GO.** H96 C2 produced a positive pooled historical OOS capital simulation, but it has not passed all statistical and risk gates. Past simulation results do not guarantee future performance. This is not investment advice.

## Latest dashboard

The [OSH BTCUSDT H96 C2 dashboard](https://sherzoddd2002.github.io/osh-btcusdt-winrate-audit/) is published from GitHub Pages after the update is merged into main. The page is fully static and has no backend, exchange connection, API keys, tracking scripts, or order placement code.

Machine-readable result: [h96_c2_results_snapshot.json](h96_c2_results_snapshot.json).

## H96 C2 — latest development OOS result

The current leading candidate is H96_C2_BASE_TP_0.75_BOTH:

| Metric | Result |
|---|---:|
| OOS trades | 241 |
| Net win rate | 58.92% |
| Net expectancy | +9.463 bps / trade |
| Profit Factor | 1.1596 |
| Positive folds | 4 / 5 (80%) |
| Median fold expectancy | +7.621 bps |
| All-Taker expectancy | +7.875 bps |
| Passed development checks | 5 / 8 |

### Pooled OOS capital simulation

Each capital path starts with a separate $100 initial balance. These rows are not five independent fold balances added together.

| Position cap | Primary final balance | Primary PnL | Primary max DD | All-Taker final balance | All-Taker PnL | All-Taker max DD |
|---:|---:|---:|---:|---:|---:|---:|
| $25 | $105.70 | +$5.70 | 4.65% | $104.74 | +$4.74 | 4.77% |
| $50 | $111.40 | +$11.40 | 8.74% | $109.49 | +$9.49 | 9.02% |
| $100 | $121.06 | +$21.06 | 15.84% | $117.02 | +$17.02 | 16.38% |

Primary assumes Taker entry, a conditional passive Maker take-profit exit, and Taker stop/time exits. All-Taker stress assumes Taker entry and Taker for every exit. The $100 cap breaches the 15% drawdown guardrail in both scenarios; $50 is the selected forward-paper cap.

### Strategy parameters (frozen for forward paper)

- TP LONG / SHORT: 0.75σ / 0.75σ
- Stop Loss: 1.00σ
- TIME exit: 96 × 15m bars (24 hours)
- Entry cooldown: 96 bars (24 hours)
- Volatility scaling horizon: 96 bars
- Assumed fee: Maker 2 bps/side, Taker 5 bps/side
- Round-trip slippage: 2 bps; BNB discount: 0%
- Forward paper capital: $100 starting balance, $50 position cap, 10% cash reserve, no leverage

### Why this remains NO-GO

Three of eight development checks failed:

1. Net win rate is below 60% (58.92%).
2. Profit Factor is below 1.25 (1.1596).
3. The familywise-adjusted bootstrap lower bound is not greater than zero.

The multiple-hypothesis bootstrap used 10,000 replications and six tested hypotheses (C2/C3 × 48/64/96 bars), with per-test alpha 0.00833. Four of five folds were positive, but one fold was negative. TIME exits were weak: 47 trades, 29.79% win rate, and −38.732 bps expectancy. These findings are diagnostics, not permission to tune against the final holdout.

## Validation and limitations

- The reported 241 trades are pooled development OOS trades across five purged walk-forward folds.
- The final 180-day holdout was excluded and must remain untouched until a separately specified final evaluation.
- A 15m OHLC bar cannot prove passive Maker queue position or an actual Maker fill. The Primary scenario's Maker TP assumption is conditional and optimistic relative to a verified order-book replay.
- Funding uses the previous completed BTCUSDT 15m close as a mark-price proxy, not the actual Binance Mark Price.
- A positive historical capital path is not a return forecast. Risk, fees, slippage, funding, and execution can differ in future conditions.
- The local forward runner is paper-only and refuses stale candles. Its forward report has not yet been produced; no live order is sent.

## Frozen forward paper runner

The forward runner is in [forward_paper_h96_c2.py](forward_paper_h96_c2.py), on branch research/forward-paper-h96-c2 (PR #4). It locks the model artifact, training-data hash, discovery-report hash, thresholds, and configuration. It does not retrain during the forward window and does not use the untouched holdout for tuning.

Run locally only after PostgreSQL candles and funding history are current and the freshness checks pass:

~~~powershell
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

.\.venv\Scripts\python.exe forward_paper_h96_c2.py --table public.btcusdt_15m --funding-csv .\binance_btcusdt_funding_history.csv --discovery-report .\development_time_exit_sweep_1d.json --historical-data-end-utc "2026-10-31T23:45:00Z" --forward-start-utc "2026-11-01T00:00:00Z" --holdout-days 180 --out .\forward_paper_h96_c2.json
~~~

Default frozen files:
- forward_paper_h96_c2_model.joblib — local model artifact
- forward_paper_h96_c2_lock.json — integrity/configuration lock
- forward_paper_h96_c2.json — local forward report

Do not manually edit the model, lock, thresholds, or report. Keep local artifacts and credentials out of Git. The forward sample is incomplete until it reaches both 180 calendar days and 100 completed trades. The verdict then remains subject to every predeclared performance and risk gate.

## Repository safety

- index.html is a fully static GitHub Pages dashboard.
- h96_c2_results_snapshot.json contains only aggregate research values, not raw candles or account data.
- No API keys, database passwords, .env files, exchange credentials, or raw trade database are published.
- No live trading, private keys, or order-placement integration exists in the dashboard.
- Earlier v2/v3 reports and pages are legacy research snapshots; the H96 C2 dashboard above is the current development OOS summary.

GitHub Pages source: main / (root). Merge the dashboard update PR into main to publish it.


## Local Binance Futures Testnet terminal

The public GitHub Pages dashboard is informational only. It cannot access a local `.env` file, so Futures Testnet order execution is a separate app served only from `127.0.0.1`.

### Setup (Windows PowerShell)

Use Binance USDⓈ-M Futures **Demo Trading API keys only**. Do not paste Live keys. The API host is hard-coded to `https://testnet.binancefuture.com` (non-production Futures Testnet/Demo-compatible REST host).

Install the dependencies (including Windows Credential Manager support):

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

Freeze the H96 C2 model once using only pre-holdout candles. Use the local discovery-report path generated during the time-exit sweep:

~~~powershell
.\.venv\Scripts\python.exe -m demo_executor.prepare_model --table public.btcusdt_15m --discovery-report .\development_time_exit_sweep_1d.json --historical-data-end-utc "2026-10-31T23:45:00Z" --forward-start-utc "2026-11-01T00:00:00Z"
~~~

The local `forward_paper_h96_c2_model.joblib` and `forward_paper_h96_c2_lock.json` files are ignored by Git. Existing artifacts are hash-checked and reused; changed artifacts are not silently retrained.

For this local Futures Testnet executor, the REST host is pinned to `https://testnet.binancefuture.com` because `demo-fapi.binance.com` is currently TLS-reset on the target network while the legacy Futures Testnet host successfully serves the public `/fapi` market-data endpoints. Binance documentation currently lists `demo-fapi.binance.com` as the primary Testnet/Demo REST URL, while its Futures SDK/testing material still references the legacy host; therefore this is a network-compatibility fallback, not a guarantee that every Demo API key will authenticate there. The first account check is read-only and must succeed before any strategy start. There is no production URL fallback.

Start the local Testnet engine on your PC with one click: double-click `START_DEMO_ENGINE.bat` in the repository folder. The script creates `.venv` if needed, installs missing dependencies, pins the local port to `8765`, keeps Testnet order sending OFF, and starts the local API. Keep its console window open while using the GitHub Pages controls.

Manual alternative:

~~~powershell
$env:OSH_LOCAL_PORT = "8765"
$env:OSH_ENABLE_DEMO_ORDERS = "false"
.\.venv\Scripts\python.exe -m demo_executor.app
~~~

Open the GitHub Pages dashboard at `https://sherzoddd2002.github.io/osh-btcusdt-winrate-audit/` and use its **H96 C2 Futures Testnet terminal** section:

1. Press **Connect local engine**. In Microsoft Edge, allow this GitHub Pages origin to access localhost/local network when prompted. The browser talks only to `127.0.0.1:8765`.
2. Enter API Key and Secret created for the Futures Testnet environment and press **Save API on this PC**. The backend stores them in Windows Credential Manager; they are not written to GitHub, browser local storage, or the repository.
3. Press **Check Binance Demo**. Verify the fixed Demo URL, available balance, BTCUSDT exchange filters, and One-way position mode.
4. Press **Start H96 C2** and explicitly confirm Testnet order sending. This button is the only normal path that enables Testnet entries for the current app process. No Live endpoint exists.
5. **Stop new entries** prevents new trades while the engine continues managing an existing Demo position.
6. **Reset saved API** first disables new entries, then checks the Demo account. It refuses to delete credentials if a position or regular/Algo order is present. On a safe reset, only the API credentials are deleted; model, cooldown state and audit log are preserved.

The local Python engine must remain running on your computer while the GitHub Pages controls are in use. If the engine or PC stops, open Testnet orders at Binance may remain; inspect the account before restarting.

### Execution controls

- The process binds only to `127.0.0.1`; the browser receives no API key or secret.
- The only exchange host is `https://testnet.binancefuture.com`, the non-production Futures Testnet REST host. It is separate from current Binance Futures Demo Trading (`demo.binance.com` / `demo-fapi.binance.com`). No Live endpoint, URL override or production fallback exists.
- One-way position mode is required. The app uses isolated margin and 1x leverage; it does not change position mode.
- It skips signals when Binance's current minimum quantity/notional exceeds the $50 position cap. It never silently increases order size.
- Market entry is followed by a reduce-only STOP_MARKET Algo order (`POST /fapi/v1/algoOrder`, `triggerPrice`) and a post-only GTX LIMIT take-profit (`POST /fapi/v1/order`). The conditional stop is queried/canceled through the Algo API; the 96-bar time exit uses a reduce-only market close. Binance moved USDⓈ-M conditional order types to the Algo service from 2025-12-09.
- One entry per 96 completed 15m bars; a 15% Demo-equity drawdown disables new entries. Existing positions remain under management.
- Unknown order responses are reconciled by client order ID rather than blindly retrying market orders.
- If the process or PC stops, open Demo orders may remain at Binance. Check the Testnet account manually before restarting.

This is a Futures Testnet execution integration test, not validation of profitability. H96 C2 remains DISCOVERY ONLY / NO-GO (58.92% historical OOS win rate, PF 1.1596, 5/8 development gates). Demo fills do not demonstrate real-market Maker priority or live performance. Keep all actual credentials, the root `.env`, model artifacts and runtime state off GitHub.
