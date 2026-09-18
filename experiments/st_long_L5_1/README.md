# ST LONG L5-1 — Spike Transformer NEWEST_FIRST

**Direction:** LONG
**Status:** DEPLOYED
**Dates:** Started 2026-Q3, Deployed 2026-08
**Parent:** Replaces S3 as primary LONG scorer

## Hypothesis
Non-stationary Transformer with FiLM regime conditioning can predict 6%+ upside spikes better than the Conv1D S3 model, using walkforward validation across 3 folds.

## Config
- Architecture: DeStationary Attention + FiLM conditioning + Attention Pooling
- d_model=64, n_heads=4, d_ff=128, n_blocks=2, dropout=0.1
- 128 time-steps, 33 sequence features, 22 regime features
- ASL loss: gamma_neg=5.0, gamma_pos=0.1, clip=0.10
- Folds: F5/F6/F7 (walkforward, newest-first priority)
- 6 models total: L5-1 F5/F6/F7 + ASL6 F5/F6/F7
- GCS data: `gs://<GCS_BUCKET>/Algos/RA_MoE/long_L5_1_data/`
- GCS models: `gs://<GCS_BUCKET>/Algos/RA_MoE/long_L5_1_models/`
- BQ model_versions: `spike_transformer_long_L5_1_fold{5,6,7}` + `spike_transformer_long_L5_2_asl6_fold{5,6,7}`

## Results

| Metric | 6:15 AM | 6:45 AM |
|---|---|---|
| Avg PnL/day | +1.88% | +0.48% |
| Sharpe | 12.3 | 4.4 |
| Trades (Aug 3-18) | 90 | 113 |
| Tickers | 28 | 26 |
| EW Total | +22.57% | +5.71% |

## Decision
**DEPLOYED** — 6:15 AM timing only. NEWEST_FIRST strategy with M_AGREEMENT filter. 6:45 AM does not meet 1.8% threshold.

## Failed Improvements
- A/B/C variants (L5-3/4/5): all < L5-1 baseline
- v14 optimizer: failed
- ASL6-645 entry-offset: harmful (-61% at 6:45)
- All P1-P4 improvement tiers: 0 wins out of 31 ideas

## Scripts
- Training: `ra_moe/ra_moe_long_L5_1_{data_prep,train,optimizer}.py` + fold6/fold7 variants
- Scoring: `scoring/daily_score_st_long.py`, `scoring/daily_score_st_long_645.py`
- Backtest: `backtesting/backtest_st_long_prioritization.py`
