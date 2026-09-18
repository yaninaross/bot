# Exp 220 — Balanced Teacher Model for Non-Crypto Spikers

## Problem

Current deployed models (Exp 152-185) use crypto-dominated teacher sets:
- TRAIN_TEACHERS_13: ~10/13 crypto miners
- TEACHERS_31: ~20/31 crypto-adjacent

The universal model learns spike patterns weighted by training sample distribution.
When 65-75% of training data is crypto miners, the model optimizes for crypto microstructure —
probability calibration, feature weights, and attention patterns are all crypto-tuned.

Non-crypto tickers get scored as "other" with a generic embedding, producing weak configs
that fail the S3 deployment gate (1.8% avg PnL + family agreement).

**Exp 103-106 proved this is solvable:** defense-specific teacher pools produced
LUNR 6.62%, RKLB 5.57%, KTOS 2.20% PnL — all above deployment threshold.
But those experiments are old, not walkforward, and not in the S3 pipeline.

## Goal

Train a walkforward model (6 folds, 15 seeds) with a balanced teacher set that gives
non-crypto high-beta spikers their own embeddings and equal training weight.
Same architecture as Exp 185 (proven). Same 10% spike threshold (viable for target tickers).

## Teacher Set: TRAIN_TEACHERS_220

~18 tickers, roughly 50/50 crypto vs non-crypto.

### Non-crypto spikers (from spike analysis, ≥2 spike-5%/mo AND positive Jun RS):

| Ticker | Sector       | 5%/mo | 10%/mo | Notes                          |
|--------|-------------|-------|--------|--------------------------------|
| OKLO   | Nuclear     | 7.8   | 2.96   | 111% of crypto baseline        |
| LUNR   | Space       | 7.7   | 2.67   | 110%, Exp 103 had 6.62% PnL   |
| SMR    | Nuclear     | 7.2   | 2.63   | 103%                           |
| HIMS   | Healthcare  | 6.7   | 1.63   | 96%, NOT currently scored      |
| RKLB   | Space       | 6.3   | 1.54   | 90%, Exp 103 had 5.57% PnL    |
| ACHR   | eVTOL       | 5.1   | 1.25   | 73%, already in TEACHERS_31   |
| SMCI   | AI infra    | 5.0   | 1.13   | 71%, in FULL_SCORED_UNIVERSE   |
| JOBY   | eVTOL       | ~4    | ~1.0   | In FULL_SCORED_UNIVERSE        |
| KTOS   | Defense     | ~3    | 0.25   | Exp 103 had 2.20% PnL         |

### Crypto miners (strongest, for balance):

| Ticker | 5%/mo | 10%/mo | Notes                   |
|--------|-------|--------|-------------------------|
| HUT    | ~9    | ~3.7   | Top deployed ticker     |
| CIFR   | ~8    | ~2.5   | Consistently deployed   |
| CLSK   | ~8    | ~2.3   | Consistently deployed   |
| KEEL   | ~9    | ~3.7   | Highest spike rate      |
| RIOT   | ~6    | ~0.9   | Baseline crypto miner   |

### Optional additions (evaluate data availability):

| Ticker | Sector    | 5%/mo | Notes                    |
|--------|----------|-------|--------------------------|
| MRNA   | Biotech  | ~4    | High vol, NOT scored     |
| CRSP   | Biotech  | ~3    | Gene editing, NOT scored |
| COIN   | Crypto   | ~5    | Exchange, not miner      |
| HOOD   | Fintech  | ~4    | Retail broker, volatile  |
| VRT    | AI infra | ~3    | Data center power        |
| VST    | Energy   | ~3    | Nuclear/power            |

**Final teacher count target: 15-20 tickers.**

## Architecture

Same as Exp 185. No changes needed:
- Universal model with per-ticker asset embeddings
- Same feature set (OHLCV + microstructure + context bars)
- Same SPIKE_THRESHOLD = 0.10 (10%)
- Same lookback window

The asset embedding is the key mechanism — each teacher gets its own learned embedding,
so the model can learn ticker-specific patterns while sharing feature weights.

## Walkforward Structure

Same as Exp 180-185:
- 6 folds (220-225)
- Fold IDs: 220, 221, 222, 223, 224, 225
- 15 seeds per fold (seed = fold_id * 100 + 01 through 15)
- Same train/val/test date splits as 180-185

## Data Requirements

### Already have master_inference on GCS:
- All crypto miners (HUT, CIFR, CLSK, KEEL, RIOT, etc.)
- LUNR, RKLB, OKLO, SMR, ACHR — in TEACHERS_31 / FULL_SCORED
- SMCI, JOBY, KTOS — in FULL_SCORED_UNIVERSE

### Need to build (OHLCV → micro → stitch):
- HIMS — not currently in any pipeline
- MRNA — not currently in any pipeline
- CRSP — not currently in any pipeline
- COIN — may exist, need to check
- HOOD — may exist, need to check

**Check:** `gs://<GCS_BUCKET>/Data/Inference_v2/{ticker}/` for each candidate.

## Implementation Steps

### Step 1: Data audit
Check GCS for existing master_inference parquets for all candidate teachers.
List which tickers need OHLCV + micro + stitch.

### Step 2: Build missing data
For each missing ticker:
```
bulk_ohlcv.py          → download OHLCV from Polygon
backfill_micro_parallel.py  → build microstructure features
bulk_stitch_master.py  → create master_inference parquet
```
Add new tickers to the ticker lists in these scripts.

### Step 3: Create training script
Copy `exp_185_train.py` (or `exp_200_walkforward_train.py` as reference).
Changes:
- `EXP_ID = 220` (through 225 for folds)
- `TRAIN_TEACHERS = [...]` — the balanced list above
- Everything else stays the same

Output: `exp_220_walkforward_train.py`

### Step 4: Train (user runs)
6 folds × 15 seeds = 90 model instances.
Models saved to: `gs://<GCS_BUCKET>/Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction/`
Named: `UNIVERSAL_S{seed}_M{exp_id}.h5`

### Step 5: Score
Add Exp 220-225 to `score_all_experiments.py`.
Score the full universe (72+ tickers) with the new models.
Results to: `gs://<GCS_BUCKET>/Algos/Results_scored_20260703/` (or new dated folder).

### Step 6: WF optimize
Create `optimize_wf_v12_220.py` through `_225.py` (copy from existing per-fold scripts).
Or add 220-225 family to existing multi-experiment optimizer.
Results to BQ: `wf_optimization_results` table.

### Step 7: Deploy
Add `'220_XSECTOR'` to `WF_DEPLOY_EXPERIMENTS` in `daily_score_combined.py`.
Remove LUNR from `WF_EXCLUDE_TICKERS` (it should now have a proper model).
S3 gate evaluates non-crypto tickers through a model trained for them.

## Expected Outcomes

- Non-crypto spikers (OKLO, LUNR, SMR, RKLB, ACHR) get proper embeddings
- Model learns both crypto AND non-crypto spike patterns
- S3 gate sees configs with 1.8%+ PnL for non-crypto tickers
- Portfolio diversifies from 7/9 crypto to ~50/50 crypto/non-crypto
- Sector rotation impact reduced (not all eggs in WGMI basket)

## Risk

- Diluting crypto training signal may slightly hurt crypto-only performance
- Some non-crypto tickers may not have enough 10% spike days for reliable training
- Data pipeline for new tickers (HIMS, MRNA, CRSP) needs full micro build
- 90 training runs is ~12-24 hours on current VM

## Relationship to Existing Experiments

| Exp   | Teacher Set     | Focus           | Status        |
|-------|----------------|-----------------|---------------|
| 152   | TEACHERS_13    | Crypto miners   | Deployed      |
| 160   | TEACHERS_13    | Crypto miners   | Deployed      |
| 175   | TEACHERS_13    | Crypto miners   | Deployed      |
| 185   | TEACHERS_31    | Crypto-heavy    | Deployed      |
| 200   | TEACHERS_31    | CNN-Transformer | Bug in scoring|
| 103   | Defense pool   | Defense only    | NOT in S3     |
| 210   | 35 dip tickers | Dip-then-rip    | Different arch|
| **220** | **Balanced 18** | **Crypto + non-crypto** | **NEW** |

Exp 220 is NOT a replacement for 185 — it's an addition. Both score simultaneously.
The S3 gate picks the best config per ticker across all experiments.
