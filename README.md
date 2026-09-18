# bot-RAG — Intraday Spike Prediction System

Predicts 6-10%+ intraday price spikes in ~65 volatile US equities during the first trading hour. Uses a Spike Transformer (DeStationary Attention + FiLM regime conditioning) with walkforward validation.

## Production Status

| Direction | Strategy | Avg PnL/day | Tickers | Timing |
|-----------|----------|-------------|---------|--------|
| **LONG** | NEWEST_FIRST (L5-1 + ASL6, 6 models) | +1.88% | 28 | 6:15 AM |
| **SHORT** | L5-2 HYB_F7_30 + M_AGREEMENT | +3.20% | 26 | 6:15 AM |

## Directory Structure

```
bot-RAG/
├── pipeline/          # Daily data refresh (OHLCV → micro → stitch)
├── scoring/           # Production scorers (S3, ST LONG, ST SHORT)
├── ra_moe/            # Spike Transformer training pipelines (data_prep, train, optimizer)
├── training/          # Legacy model training (exp_240-500, data_utils)
├── optimization/      # Walk-forward optimizers (v12 series)
├── backtesting/       # Active backtest scripts
├── analysis/          # Analysis, oracle, model selection scripts
├── data_ops/          # Bulk rebuilds, backfills, audits, migrations
├── experiments/       # Per-experiment tracking (README per experiment)
├── research/          # Future proposals
├── moirai/            # MOIRAI foundation model scripts
├── logs/              # Walk-forward optimizer logs
├── archive/           # All superseded scripts (~234 files)
├── EXPERIMENT_LOG.md  # Central experiment registry
└── MODEL_EVALUATION_GUIDE.md
```

## Daily Pipeline (4 stages)

```
OHLCV (Polygon) → Microstructure (T&Q) → Stitch (master_inference) → Score (5 scorers)
```

| Stage | Scripts | Runtime |
|-------|---------|---------|
| 1. OHLCV | `pipeline/daily_ohlcv_{1,2}.py` | ~5 min |
| 2. Micro | `pipeline/daily_micro_{1,2,3}.py` | ~15 min |
| 3. Stitch | `pipeline/daily_stitch_master.py` | ~8 min |
| 4. Score | `scoring/daily_score_{s3,st_long,st_long_645,short,short_645}.py` | ~5 min each |

## Key Architecture

- **Model**: DeStationary Attention + FiLM conditioning + Attention Pooling
- **Size**: d_model=64, n_heads=4, d_ff=128, n_blocks=2 (~100K params)
- **Input**: 128 time-steps, 33 seq features, 22 regime features
- **Loss**: ASL (gamma_neg=5/6, gamma_pos=0.1, clip=0.10)
- **Target**: Binary spike probability (single head)
- **Validation**: Walkforward with OPT/TEST windows per fold

## Infrastructure

- **GCS**: `gs://<GCS_BUCKET>/` (Data/Inference_v2/, Algos/RA_MoE/)
- **BQ**: `<GCP_PROJECT>.scoring_logs.wf_optimization_results`
- **Scripts**: All run as Colab cells (no `if __name__`, config vars at top)
- **Seeds**: 15 per experiment (seed = exp*100 + 01-15)

## Deploy Threshold

- Avg PnL >= 1.8%
- Win rate >= 60%
- 2+ trades/month per ticker
- Portfolio >= 5 tickers after selective filter
