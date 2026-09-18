# ST SHORT X-ASSET — BTC/ETH Lead-Lag Features

**Direction:** SHORT
**Status:** PENDING (optimizer next)
**Dates:** Started 2026-08-20
**Parent:** Builds on L5-2 SHORT baseline

## Hypothesis
BTC and ETH prices lead crypto-correlated equities by 5-15 minutes. Adding BTC/ETH lead-lag features to the SHORT model should improve predictions for crypto-adjacent tickers.

## Config
- Architecture: Same DeStationary + FiLM backbone
- 47 sequence features (+14 BTC/ETH: ROC 1/5/15/60min, relative volume, momentum, correlation vs SPY)
- 26 regime features (+4: BTC 60-bar return, BTC rolling vol, ETH-BTC spread, crypto-equity correlation)
- Fold: 5 (train through 2026-06-13, val Jun 16 - Jul 15, test Jul 16 - Aug 16)
- 99 tickers
- GCS data: `gs://<GCS_BUCKET>/Algos/RA_MoE/short_L5_2_xasset_data/`
- GCS models: `gs://<GCS_BUCKET>/Algos/RA_MoE/short_L5_2_xasset_models/`
- BQ model_version: `spike_transformer_short_L5_2_xasset_fold5`

## Results (Backtest vs Baseline)

| Metric | Baseline | X-ASSET | Delta |
|---|---|---|---|
| Avg PnL (all) | -- | -- | **+0.42%** |
| Avg PnL (crypto) | -- | -- | **+1.16%** |
| Deployable tickers | 20 | 29 | +9 |
| Deploy avg PnL | 2.76% | 3.35% | +0.59% |

**Top crypto lifts:** MSTR +4.46%, WULF +3.03%, COIN +2.40%, CLSK +2.02%
**Non-crypto surprise:** AEHR +3.99%, RDW +3.14%, VST +1.97%

**This is the first positive ML experiment** in the project's history (31 prior ideas all failed).

## Decision
**PENDING** — Optimizer run required: `ra_moe/ra_moe_short_L5_2_xasset_optimizer.py` (GPU, Colab)

## Scripts
- Data prep: `ra_moe/ra_moe_short_L5_2_xasset_data_prep.py`
- Training: `ra_moe/ra_moe_short_L5_2_xasset_train.py`
- Optimizer: `ra_moe/ra_moe_short_L5_2_xasset_optimizer.py`
- Backtest: `backtesting/backtest_st_short_xasset_vs_baseline.py`
- Data build: `data_ops/build_crypto_master_inference.py`
