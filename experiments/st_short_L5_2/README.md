# ST SHORT L5-2 — Spike Transformer HYB_F7_30

**Direction:** SHORT
**Status:** DEPLOYED
**Dates:** Started 2026-Q3, Upgraded to L5-2 2026-08-17
**Parent:** Upgraded from L5-1 SHORT

## Hypothesis
L5-2 training data produces better SHORT spike predictions than L5-1, and a hybrid fold selection (F5+F6+F7 with F7 gated at val_b >= 3.0%) maximizes per-ticker coverage.

## Config
- Architecture: Same as L5-1 (DeStationary + FiLM, ~100K params)
- 33 sequence features, 22 regime features
- Folds: F5/F6/F7 (hybrid selection per ticker)
- F7 gate: val_b_avg_pnl >= 3.0%
- 26 tickers: L2_F5 (12), L2_F6 (7), L2_F7 (7)
- GCS data: `gs://time-series-bucket-ys/Algos/RA_MoE/short_L5_2_data/`
- GCS models: `gs://time-series-bucket-ys/Algos/RA_MoE/short_L5_2_models/`
- BQ model_versions: `spike_transformer_short_L5_2_fold{5,6,7}`

## Results

**4-Strategy Comparison (Jul 10 - Aug 13, 33 days):**

| Strategy | Avg PnL | WR | PF | EqW | Trades | Tickers |
|---|---|---|---|---|---|---|
| **L5-2 HYB** | **+2.84%** | **78%** | **4.39** | **+56.30%** | 76 | 23 |
| L5-1 HYB | +2.64% | 81% | 4.39 | +46.19% | 79 | 21 |
| Combined | +2.53% | -- | -- | +55.16% | 107 | 30 |
| Recency | +2.29% | 73% | -- | +47.60% | 135 | 30 |

**Trade Prioritization (Aug 3-18, 12 days):**

| Timing | Best Strategy | Avg PnL/day | Sharpe | Trades | Deployable? |
|---|---|---|---|---|---|
| **6:15 AM** | **M_AGREEMENT** | **+3.20%** | **15.0** | 20 | YES |
| 6:45 AM | M_AGREEMENT | +1.24% | 7.7 | 29 | NO |

## Decision
**DEPLOYED** — L5-2 HYB_F7_30 with M_AGREEMENT filter at 6:15 AM. Dominates L5-1 by +10.10% EqW.

## Scripts
- Training: `ra_moe/ra_moe_short_L5_2_{data_prep,train,optimizer}.py` + fold6/fold7 variants
- Scoring: `scoring/daily_score_short.py`, `scoring/daily_score_short_645.py`
- Backtest: `backtesting/backtest_st_short_prioritization.py`
