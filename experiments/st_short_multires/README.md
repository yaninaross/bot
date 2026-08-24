# ST SHORT Multi-Resolution — Dual Transformer Backbone

**Direction:** SHORT
**Status:** TESTED (v1 overfit, v2 pending optimizer)
**Dates:** Started 2026-08
**Parent:** Builds on L5-2 SHORT baseline

## Hypothesis
Adding a parallel 15-minute-bar Transformer branch captures longer-horizon context (~32 hours) that the 1-minute-only model misses.

## Config

### v1
- Architecture: Dual TransformerBackbone — 1-min (2 blocks) + 15-min (1 block)
- Late fusion: concat(64, 64) → Dense(64, gelu) → FiLM → head
- ~153K params
- 12 low-res features: close/high/low returns, normalized volume, range, RSI-14, RVOL, ROC-4/8/16, real-bar flag, time encoding
- MAX_TRAIN_SAMPLES = 50,000
- GCS: `short_L5_2_multires_data/` + `short_L5_2_multires_models/`

### v2
- Architecture: Identical to v1
- MAX_TRAIN_SAMPLES = 150,000 (3x data to address v1 overfitting)
- Samples/param ratio: v1 = 0.32, v2 = ~1.0
- GCS: `short_L5_2_multires_v2_data/` + `short_L5_2_multires_v2_models/`

## Results
- **v1:** Overfitting — 153K params on ~49K samples is too few
- **v2:** Training complete, optimizer pending

## Decision
**PENDING** — v2 optimizer: `ra_moe/ra_moe_short_L5_2_multires_v2_optimizer.py` (GPU, Colab)

## Scripts
- v1: `ra_moe/ra_moe_short_L5_2_multires_{data_prep,train,optimizer}.py`
- v2: `ra_moe/ra_moe_short_L5_2_multires_v2_{data_prep,train,optimizer}.py`
