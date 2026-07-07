# Model Evaluation Guide

How to evaluate spike-model results end-to-end — from scoring to strategy selection.

---

## Prerequisites

Before running any evaluation, ensure the data pipeline is current:

| Step | Script | Runtime | What it does |
|------|--------|---------|--------------|
| 1. OHLCV | `daily_ohlcv_1.py` + `daily_ohlcv_2.py` | ~5 min | Pull 1-min bars from Polygon for all tickers + context assets |
| 2. Microstructure | `daily_micro_all.py` | ~10 min | Build micro features (spread, OBI, volume delta) for 20 tickers |
| 3. Stitch | `daily_stitch_master.py` | ~8 min | Combine OHLCV + micro + context → `master_inference.parquet` per ticker on GCS |

All three are incremental — they only fetch/build new dates. Run them daily or whenever you need fresh data.

**GCS data locations:**
- `gs://time-series-bucket-ys/Data/Inference_v2/{TICKER}_master_inference.parquet`
- `gs://time-series-bucket-ys/Data/Inference/Microstructure/{TICKER}_micro.parquet`

**BQ table:** `ys-time-series.scoring_logs.wf_optimization_results` (WF optimizer configs, model_version `robust_wf_v12`)

---

## Step 1: Score All Experiments (GPU Required)

**Script:** `score_all_experiments.py`
**Runtime:** ~2-3 hours on T4 GPU (72 tickers × 5 experiments × 15 seeds)
**When to re-run:** After updating master_inference data, or when extending to new tickers

```
# Colab cell — needs GPU runtime
# Scores experiments 152, 160, 175, 185, 200
# Writes CSVs to gs://time-series-bucket-ys/Algos/Results_scored_20260703/
```

**What it produces:** One CSV per (ticker, experiment, seed):
`Algos/Results_scored_20260703/{exp}_{ticker}_seed{NN}_scores.csv`

Each CSV contains columns: `Date`, `Prob_Spike`, plus features. Only dates after `SCORE_CUTOFF = 2026-04-01` are scored (test period only).

**Staleness check:** The script auto-skips (ticker, seed) pairs where the CSV on GCS is newer than the master_inference file. To force re-score, delete the CSV on GCS first.

**Fallback scores:** All backtest scripts also read from `Algos/Results_all_v2/` as fallback for any missing CSVs (these are older full-history scores).

**Verify:** After scoring, spot-check that CSVs exist:
```python
from google.cloud import storage
client = storage.Client()
bucket = client.bucket("time-series-bucket-ys")
blobs = list(bucket.list_blobs(prefix="Algos/Results_scored_20260703/185_CLSK"))
print(f"CLSK exp-185 seeds scored: {len(blobs)}")  # expect 15
```

---

## Step 2: Baseline Strategy Comparison (No GPU)

**Script:** `backtest_strategy_csv.py`
**Runtime:** ~2-3 minutes
**Purpose:** Establish the S3 Deployed benchmark and compare 5 basic strategies

### What it tests

| # | Strategy | Description |
|---|----------|-------------|
| 1 | **S3 Deployed** (benchmark) | BQ family-agreement gate, one experiment per ticker, limit_entry |
| 2 | Composite seed=50% | All tickers, fire when 50%+ of individual seeds agree |
| 3 | S3 + GATE lb=10 | S3 + skip ticker if last 10 trades avg PnL < 0 |
| 4 | S3 + GATE lb=7 | S3 + performance gate with lookback 7 |
| 5 | Composite + GATE lb=10 | Seed=50% composite + performance gate |

### Config

- **Tickers:** AMD, APLD, CIFR, CLSK, FLNC, HUT, KEEL, PLUG, RCAT, RIOT, WULF (11)
- **Test period:** 2026-04-01 to 2026-07-03
- **Entry:** Minute 375 (06:15 PST), limit_entry with per-ticker dip_pct and max_wait
- **Slippage:** 0.15%
- **Deploy threshold:** 1.8% avg PnL (selective portfolio filter)

### What to look for

- S3 Deployed avg PnL should be ~2.5-3.0% (this is the benchmark to beat)
- GATE strategies should show better risk-adjusted metrics (Sharpe, MaxDD) vs S3
- Win rate should be 60%+ for viable strategies
- Trade count: 2+ trades/month is the target frequency
- June performance separately — regime degradation is real, check if strategies survive

### Output

- Summary table per strategy: avg PnL, total PnL, trades, WR, MaxDD, Sharpe
- Selective portfolio (tickers with avg PnL >= 1.8%)
- Cumulative PnL equity curves (chart)

---

## Step 3: Composite Method Comparison (No GPU)

**Script:** `backtest_composite_csv.py`
**Runtime:** ~1-2 minutes
**Purpose:** Find the best signal-gating method across experiments

### What it tests

5 composite methods × 2 variants (STATIC + ADAPTIVE):

| # | Method | Parameters | How it gates |
|---|--------|------------|--------------|
| 1 | Integer family agree | agree=1, 2, 3 | Count experiment families with prob >= thresh; require >= agree (min 2 families present) |
| 2 | Pct family agree | pct=50%, 60%, 75% | Same but fraction of families must agree |
| 3 | Seed-level agree | seed=40%, 50%, 60%, 75% | Count individual seeds across all exps; require pct fraction |
| 4 | **Precision-weighted vote** | threshold=50% | Weight each experiment's vote by historical precision; fire when weighted signal > 0.5 |
| 5 | Teacher-aware voting | pct=60% | Like pct-family but only count experiments that trained on the ticker |

**STATIC variant:** Fixed prob_thresh from BQ config per ticker.
**ADAPTIVE variant:** prob_thresh shifts +0.10 when trailing precision < 40%, -0.10 when > 60%.

### Config

Same 11 tickers, same test period, same limit_entry sim as Step 2.

### What to look for

- **Winner from prior run:** STATIC weighted>50% (+3.42% avg, 69% WR, -18.7% MaxDD)
- ADAPTIVE mostly hurts — collapses portfolios from 7-8 to 3-4 tickers
- Compare selective portfolio size: methods that pass too few tickers are unreliable
- Compare against S3 Deployed benchmark from Step 2

### Output

- Table A: S3 Benchmark (deployed tickers only)
- Table B: All STATIC × method combos with selective portfolios
- Table C: All ADAPTIVE × method combos with selective portfolios
- Table D: Methods that beat the benchmark

### Key decision

The winning composite method from this step feeds into Step 4. Currently: **Weighted>50%** (precision-weighted vote, static threshold).

---

## Step 4: Optimization Strategy Comparison (No GPU)

**Script:** `backtest_optimization_csv.py`
**Runtime:** ~5-10 minutes (grid search across 33 combos)
**Purpose:** Find the best optimization strategy × composite combination

### What it tests

11 strategies × 3 composites = 33 combinations:

**Strategies:**

| # | Strategy | Source | Description |
|---|----------|--------|-------------|
| 1 | STATIC (BQ) | BQ config | Fixed BQ-optimal params, no rolling adaptation |
| 2 | GATE lb=7 | BQ config | Skip ticker if last 7 trades avg PnL < 0 |
| 3 | DECAY hl=30d | Grid search | Exponential decay weighting, half-life 30 days |
| 4 | DECAY hl=60d | Grid search | Exponential decay weighting, half-life 60 days |
| 5 | HYBRID 30d+gate10 | Grid search | Decay 30d + gate lookback 10 |
| 6 | HYBRID 60d+gate10 | Grid search | Decay 60d + gate lookback 10 |
| 7 | HYBRID 30d+gate21 | Grid search | Decay 30d + gate lookback 21 |
| 8 | ADAPTIVE THRESH | BQ config | Dynamic prob_thresh (+0.10 when precision < 40%, -0.10 when > 60%) |
| 9 | MOMENTUM 10d | BQ config | Skip ticker if 10-day price return <= 0 |
| 10 | SELECTIVE DECAY | BQ + grid | Use BQ config unless recent performance negative → fallback to decay |
| 11 | VOL GATE | BQ config | Skip ticker if 5-day vol / 20-day vol > 2.0 |

**Composites:**

| Composite | Description |
|-----------|-------------|
| S3 Deployed | BQ best experiment, S3 family-agreement tickers only (benchmark) |
| Weighted>50% | Precision-weighted vote across all experiments |
| Adapt Wt>50% | Weighted vote + adaptive prob_thresh |

**Grid search (DECAY/HYBRID strategies):**
- TP: [0.06, 0.10, 0.16, 0.22]
- SL: [0.03, 0.06, 0.10, 0.14] (constrained: SL <= TP)
- Prob threshold: [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
- Re-optimization: every 5 trading days

### Config

Same 11 tickers, test period 2026-04-01 to 2026-07-03.
Training data starts 2025-12-01 (needed for decay weighting lookback).

### What to look for

- Which strategy × composite combo has the best avg PnL while maintaining 2+ trades/month
- STATIC + Weighted>50% is the composite-step winner — does any optimization strategy improve on it?
- GATE/HYBRID should improve risk-adjusted metrics (Sharpe, MaxDD) at some cost to avg PnL
- DECAY strategies find their own TP/SL via grid search — compare to BQ-optimal configs
- June column in Table A: strategies that survive June regime shift are more robust
- Charts: cumulative PnL curves reveal drawdown timing and recovery

### Output

- Table A: 11×3 matrix — full-period avg PnL, trades, WR per (strategy, composite)
- Table B: June-only matrix — same structure, June dates only
- Table C: Per-ticker breakdown for top 3 combos
- Table D: Monthly breakdown for top 3 combos
- Chart 1: Cumulative total PnL by composite method
- Chart 2: All 11 strategies per composite
- Chart 3: Monthly PnL bars

---

## Evaluation Workflow Summary

```
                  ┌──────────────────────────────┐
                  │   Data Pipeline (daily)       │
                  │   OHLCV → Micro → Stitch      │
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │   score_all_experiments.py    │
                  │   GPU • ~2-3 hrs              │
                  │   72 tickers × 5 exps × 15    │
                  │   seeds → CSVs on GCS         │
                  └──────────────┬───────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
   ┌──────────▼────────┐ ┌──────▼────────┐ ┌────────▼──────────┐
   │ Step 2: Strategy   │ │ Step 3:       │ │ Step 4:            │
   │ Baseline           │ │ Composite     │ │ Optimization       │
   │ 5 strategies       │ │ 5 methods     │ │ 11 strats × 3      │
   │ ~2-3 min           │ │ × 2 variants  │ │ composites         │
   │                    │ │ ~1-2 min      │ │ ~5-10 min          │
   └────────────────────┘ └───────────────┘ └────────────────────┘
         ▲                       │                    ▲
         │                       │                    │
         │              Winning composite ────────────┘
         │              feeds into Step 4
         │
    S3 benchmark for
    all comparisons
```

**Run order:** Steps 2, 3, 4 can run in any order (all read the same CSVs from Step 1). But logically:
1. Run **Step 2** first to establish the S3 baseline
2. Run **Step 3** to find the best composite method
3. Run **Step 4** with the winning composite to find the best optimization strategy

---

## Decision Framework

### Is a model ready to deploy?

1. **Avg PnL >= 1.8%** on the selective portfolio (DEPLOY_THRESHOLD)
2. **Win rate >= 60%** (below this, variance kills you)
3. **2+ trades/month** per ticker on average (MIN_TRADES_PER_MONTH)
4. **June survival:** Strategy must not collapse in the most recent month (regime test)
5. **Portfolio size >= 5 tickers** after selective filter (diversification)

### When to re-evaluate

- After training new experiments or walkforward folds
- After re-scoring with `score_all_experiments.py` (new data)
- Monthly — to check if regime shift has degraded the current strategy
- After any change to the WF optimizer configs in BQ

### What NOT to optimize for

- Precision alone — a strategy with 95% precision and 0.5% avg PnL is useless
- Number of trades — more trades at lower quality destroys capital
- Backtesting Sharpe in isolation — check that raw PnL clears 1.8% first

---

## Reference: Scripts Not in the Main Evaluation Flow

These scripts exist for specific investigations, not routine evaluation:

| Script | Purpose |
|--------|---------|
| `backtest_deep_optimization.py` | 72-ticker universe version of Step 4 (TF-free, 11×2 matrix). Use when evaluating across the full scored universe, not just the 11-ticker S3 set |
| `analyze_composite_scores.py` | 72-ticker composite analysis with BQ configs. Source code for composite gate logic |
| `backtest_strategy_comparison.py` | Live TF-scoring version of Step 2 (requires GPU). Use to verify CSV scores match live inference |
| `backtest_portfolio_v2.py` | Pipeline verification — confirms data loading, feature engineering, and scaling match training. Run after changing any data pipeline code |
| `backtest_tier_comparison.py` | 6-family investigation with live TF. Use for deep-dive on experiment family performance |
| `backtest_top5_composites.py` | Efficiency-frontier composites. Use for dip_pct / max_wait sensitivity analysis |
| `daily_score_combined.py` | Production daily scorer for S3. Not an evaluation tool — this generates live trading signals |
