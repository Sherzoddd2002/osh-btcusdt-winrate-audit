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
