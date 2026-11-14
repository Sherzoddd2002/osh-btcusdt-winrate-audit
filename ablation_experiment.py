"""Legacy ablation runner disabled because its historical results are biased.

Reasons:
- The train-calibrated threshold was calculated but never used.
- Test trades were selected by ranking the full test window (look-ahead).
- Two-minute cross-asset bars were resampled without explicit availability-time
  alignment to the BTC candle close.
- The old news source was synthetic and must not be interpreted as BTC news.

Use research_v3.py instead. It runs BASE versus availability-aware external
features using frozen thresholds and an untouched holdout. Old
ablation_results.json is retained only as an archived historical artifact.
"""


def main() -> None:
    raise SystemExit(
        "This legacy ablation experiment is disabled: its test-window ranking "
        "and external-data timing were not causal. Run research_v3.py and do "
        "not use old ablation_results.json as evidence of predictive edge."
    )


if __name__ == "__main__":
    main()
