# Scoring & Optimization Instructions — Strategy S3

Updated: 2026-06-22

---

## What This Project Does (read this first)

This is a **crypto/equity spike prediction system**. A Transformer model predicts whether a ticker will spike 10%+ intraday. Walk-forward optimization finds the best (TP, SL, entry_time, prob_threshold) configs per ticker per model fold, then a family agreement gate selects which tickers to trade live.

### Key concepts

- **Experiment / Fold**: Each model is trained with a different data cutoff date. Fold 180 = cutoff Aug 2025, 181 = Sep 2025, ..., 185 = Jan 2026. Similarly 170-175 is an older fold family. Each fold produces 15 seed models.
- **Seed**: Each fold trains 15 independent models (seeds 1-15). Seed number = `fold_base * 100 + seed_index` (e.g., fold 185 → seeds 18501-18515).
- **TPSL Mode**: 9 different take-profit / stop-loss strategies: `static`, `atr`, `atr_adaptive`, `regime_scaled`, `trailing_decay`, `regime_adaptive`, `atr_sl`, `grace`, `limit_entry`.
- **Walk-forward optimization**: Grid sweep over (prob_threshold × TPSL configs) on optimization window → validate on validation window → test on holdout. Surviving configs are logged to BQ.
- **Stitched walkforward**: For fold families (170-175 or 180-185), seed scores from all 6 folds are concatenated into one continuous series before optimization. Results are labeled with the first experiment (e.g., `170_XSECTOR`).
- **Single-fold optimization**: Each fold optimized independently. Results labeled per fold (e.g., `184_XSECTOR`).
- **Family agreement gate (S3)**: A ticker qualifies for live trading only if it has positive WF results across >= 2 experiment families (152, 160, 170, 180, 185). This is the deployed strategy.
- **BQ table**: `<GCP_PROJECT>.scoring_logs.wf_optimization_results`, tag `model_version = 'robust_wf_v12'`.

### The pipeline

```
Score (GPU, Colab)     →  Optimize (CPU, VM/Colab)  →  BQ table  →  S3 gate auto-selects
score_missing_*.py        optimize_wf_v12_*.py          wf_optimization_results
```

The user does NOT need to modify `daily_score_combined.py` after optimization — it queries BQ dynamically.

---

## What the user typically asks for

1. **"Run optimization for fold X"** → Find the right `optimize_wf_v12_*.py` script from the table below, launch in tmux. Don't create new scripts unless the fold is genuinely missing.
2. **"Check optimization status"** → Query BQ via python (`from google.cloud import bigquery`). The `bq` CLI often has auth issues — always use the Python client. Count rows, tickers, modes per experiment.
3. **"Score missing tickers"** → Run `score_missing_180_185.py` in Colab (needs GPU). This generates seed CSVs on GCS before optimization can run.
4. **"Run backtesting"** → Use `backtest_tier_comparison.py` or `compare_strategies.py`. These query BQ and simulate trades.

### Things NOT to do

- Don't create new optimizer scripts unless the fold family definition is genuinely missing. Check EXP_FAMILIES in the existing scripts first.
- Don't modify the optimizer grid (TPSL_PAIRS, LIMIT_ENTRY_CONFIGS, PROB_THRESHOLDS) without being asked.
- Don't run training — user handles that separately.
- Don't use `bq` CLI for queries — use `from google.cloud import bigquery` Python client (ADC auth works, CLI auth often expires).
- Don't use GCS FUSE mount — use `google.cloud.storage` API.
- Don't backfill data — ffill prices, zero-fill volumes, never bfill.

---

## Step 1: Score Missing Tickers (score_missing_180_185.py)

**What it does:** Runs 15 model seeds per fold across tickers that were added after training.
Each (ticker, fold, seed) produces a CSV at `Algos/Results_all_v2/Universal_Exp{fold}_XSECTOR/{ticker}/Seed_{seed}_Results.csv`.

**Script:** `score_missing_180_185.py`

**Run in:** Colab with GPU (for TF inference speed)

### Configuration

```python
# Line ~135 — which folds to process (empty = all 6)
FOLD_RANGE = []   # [] = all folds, [4,5] = fold 184+185 only

# Line ~160 — tickers to score
MISSING_TICKERS = [...]  # edit this list to add new tickers
```

### How it works

1. **Pre-scan** — checks GCS for existing CSVs, builds list of missing (ticker, seed) pairs per fold
2. **Market context** — loads SPY/VIX/TLT/IWM once (shared across all folds)
3. **Per fold:** loads scaling stats + models, then for each ticker: engineer → scale → score all seeds → free memory → next ticker
4. Uploads CSVs to GCS as it goes. Resumable — skips already-scored pairs.

### Scoring status (2026-06-22)

All 180-185 folds: DONE (26 tickers scored across all 6 folds).

### Memory notes

- Processes one ticker at a time (OOM fix applied 2026-06-22)
- Peak RAM ~300MB per ticker

---

## Step 2: WF Optimization

**What it does:** For each ticker with scored CSVs, sweeps a grid of (TP, SL, entry_time, prob_threshold, tpsl_mode) combinations across 9 modes. Writes results to BQ table `wf_optimization_results`.

**Run in:** VM via tmux (recommended) or Colab. CPU only, no GPU needed.

### Which optimizer script to use

| Family | Script | ACTIVE_FAMILY | Stitch? | Notes |
|--------|--------|---------------|---------|-------|
| 170-175 stitched | `optimize_wf_v12_170.py` | `walkforward` | Yes | 6 folds stitched, OPT Apr-Jan, VAL Feb-Mar, TEST Apr-Jun |
| 180-185 stitched | `optimize_wf_v12.py` | `walkforward_185` | Yes | 6 folds stitched, same windows |
| 185 only | `optimize_wf_v12_185.py` | `single_185` | No | OPT Jan-Mar, VAL Apr-May, TEST May-Jun |
| 184 only | `optimize_wf_v12_184.py` | `single_184` | No | Same windows as 185 |
| 183 only | `optimize_wf_v12_183.py` | `single_183` | No | Same windows |
| 182 only | `optimize_wf_v12_182.py` | `single_182` | No | Same windows |
| 181 only | `optimize_wf_v12_181.py` | `single_181` | No | Same windows |
| 180 only | `optimize_wf_v12_180.py` | `single_180` | No | Same windows |
| 152+160 combined | `optimize_wf_v12.py` | `v12_160_152` | No | Older experiments |

**Creating a new per-fold script:** Copy from an existing per-fold script (e.g., `optimize_wf_v12_182.py`). Only 3 lines differ:
1. `RUN_MODES` comment (cosmetic)
2. The `single_XXX` family dict entry in `EXP_FAMILIES`
3. `ACTIVE_FAMILY = 'single_XXX'`

### Configuration (already set in each script)

```python
ACTIVE_FAMILY = 'single_185'   # or 'walkforward', 'walkforward_185', etc.
TICKER_SUBSET = []             # empty = all ~72 tickers; or specify a list
SKIP_COMPLETED = True          # skips tickers already in BQ
```

### How it works

1. Loads `_FULL_TICKERS` (~72 tickers) and checks BQ for already-completed (experiment, ticker, mode) triples
2. Loads regime data (VIX/SPY/TLT) for the full eval window
3. For each ticker:
   - Loads OHLCV + seed scores (15 seeds, or stitched across 6 folds)
   - For each of the 9 TPSL modes: grid sweep over `PROB_THRESHOLDS × TPSL configs`
   - Per seed: optimize on OPT window → validate on VAL window → test on holdout
   - Ensemble: average all seed probabilities, re-optimize prob_thresh, test
   - Saves per-seed + ensemble rows to BQ
4. Prints summary tables at the end

### Running on VM via tmux (recommended)

Colab notebooks timeout when browser disconnects (all notebooks die simultaneously — it's session timeout, NOT OOM). Use tmux on the VM instead:

```bash
# Launch optimizer in background tmux session
tmux new-session -d -s wf_170 "python3 optimize_wf_v12_170.py 2>&1 | tee /tmp/wf_170.log"

# Check status
tmux capture-pane -t wf_170 -p | tail -20

# Or check the log file
tail -30 /tmp/wf_170.log

# Attach to watch live
tmux attach -t wf_170
# (Ctrl-B D to detach without stopping)

# List all sessions
tmux list-sessions

# Check process resource usage
ps aux | grep optimize_wf | grep -v grep
```

**VM capacity:** 4 CPU / 15GB RAM. Can run 2 optimizer scripts in parallel (each uses ~5GB peak). Use separate tmux sessions:

```bash
tmux new-session -d -s wf_sess1 "python3 optimize_wf_v12_180.py && python3 optimize_wf_v12_181.py"
tmux new-session -d -s wf_sess2 "python3 optimize_wf_v12_182.py && python3 optimize_wf_v12_183.py"
```

### Checking optimization status via BQ

Always use the Python client (not `bq` CLI):

```python
from google.cloud import bigquery
client = bigquery.Client()
q = """
SELECT experiment, COUNT(*) as cnt, COUNT(DISTINCT ticker) as tickers,
       COUNT(DISTINCT tpsl_mode) as modes, MAX(run_timestamp) as last_update
FROM `<GCP_PROJECT>.scoring_logs.wf_optimization_results`
WHERE model_version = 'robust_wf_v12'
  AND (experiment LIKE '170%' OR experiment LIKE '180%')
GROUP BY experiment ORDER BY experiment
"""
for row in client.query(q):
    print(f'{row.experiment:<30} {row.cnt:>6} rows  {row.tickers:>3} tickers  {row.modes} modes')
```

For fold-level summary:
```python
q = """
SELECT SPLIT(experiment, '_')[OFFSET(0)] as fold,
       COUNT(*) as total_rows, COUNT(DISTINCT ticker) as tickers,
       COUNT(DISTINCT tpsl_mode) as modes
FROM `<GCP_PROJECT>.scoring_logs.wf_optimization_results`
WHERE model_version = 'robust_wf_v12'
  AND experiment LIKE '18%'
GROUP BY fold ORDER BY fold
"""
```

### Optimization status (2026-06-22)

| Family | Rows | Tickers | Modes | Status |
|--------|-----:|--------:|------:|--------|
| 170-175 (stitched) | 16,690 | 41 | 9 | IN PROGRESS — ~43 tickers remaining, running in tmux |
| 180 (single) | 24,743 | 49 | 9 | DONE |
| 181 (single) | 3,310 | 49 | 9 | DONE |
| 182 (single) | 3,993 | 47 | 9 | DONE |
| 183 (single) | 4,677 | 48 | 9 | DONE |
| 184 (single) | 4,506 | 47 | 9 | DONE |
| 185 (single) | 4,190 | 39 | 9 | DONE (10 tickers fewer — may need rerun) |
| 152+160 | — | — | — | Previously completed |

### After optimization completes

The S3 gate in `daily_score_combined.py` automatically queries BQ at runtime. No script changes needed. New tickers that pass:
- `mean_va_pnl >= 1.8%`
- `n_seeds >= 3`
- `limit_entry` mode
- majority-positive val_b across experiment families
- `n_fams >= 2`

...will appear in the daily scoring output.

---

## Step 3: Validate (backtest_strategies.py)

After optimization, re-run the S3 backtest to validate the expanded ticker list:

```python
# In backtest_strategies.py / compare_strategies.py — they query BQ dynamically
# Just run it, no config changes needed
```

Compare: new ticker count, total trades, avg PnL, June 2026 survival.

---

## Daily Pipeline Order (for reference)

Run in Colab in this order:

1. `daily_ohlcv_1.py` — OHLCV A-L (74 tickers)
2. `daily_ohlcv_2.py` — OHLCV M-Z (74 tickers)
3. `daily_micro_all.py` — microstructure (20 tickers)
4. `daily_stitch_master.py` — stitch OHLCV+micro → master_inference (20 tickers + 4 context)
5. `daily_score_combined.py` — Strategy S3 scoring + BQ logging

**IMPORTANT:** Step 4 creates context asset masters (VIX, SPY, TLT, IWM). If IWM master is missing, EXP-152 (RCAT) will crash. Run stitch at least once to create it.

---

## Current S3 Live Tickers (as of 2026-06-22)

9 tickers passed the BQ gate:

| Ticker | Exp | Avg PnL | Agree | Seeds |
|--------|-----|---------|-------|-------|
| APLD | 180 | 2.91% | 75% | 4 |
| CIFR | 185 | 6.58% | 100% | 3 |
| CLSK | 160 | 3.60% | 100% | 3 |
| FLNC | 175 | 2.36% | 100% | 3 |
| HUT | 175 | 2.35% | 100% | 15 |
| KEEL | 175 | 3.49% | 100% | 6 |
| PL | 180 | 2.18% | 100% | 3 |
| RCAT | 152 | 3.45% | 100% | 3 |
| RIOT | 185 | 3.49% | 100% | 11 |

IREN dropped (1/2 families positive = FAIL). May recover after 170-175 optimization adds more family data.

---

## Troubleshooting

- **OOM in score_missing:** Processes one ticker at a time (fixed 2026-06-22). If still OOM, use `FOLD_RANGE` to do one fold at a time.
- **IWM crash in daily scoring:** Run `daily_stitch_master.py` to create IWM_master_inference.parquet.
- **Stale data (score_date != today):** Pipeline hasn't run — check OHLCV scripts first.
- **No configs from BQ:** Check `wf_optimization_results` has rows with `model_version = 'robust_wf_v12'` and `survived_a = TRUE`.
- **Colab timeout:** All notebooks die simultaneously = session timeout (not OOM). Use tmux on VM instead.
- **TPSL_PAIRS KeyError for limit_entry:** Fixed in all per-fold scripts (line ~2601). `limit_entry` uses `LIMIT_ENTRY_CONFIGS`, not `TPSL_PAIRS`.
- **`bq` CLI auth errors:** Use `from google.cloud import bigquery` Python client. ADC auth works; CLI auth expires.
- **GCS data loading:** Always use `google.cloud.storage` API, never GCS FUSE mount.

---

## Script inventory

| Script | Purpose | Run where |
|--------|---------|-----------|
| `optimize_wf_v12.py` | Main optimizer (all families) | VM tmux or Colab |
| `optimize_wf_v12_170.py` | 170-175 stitched walkforward | VM tmux |
| `optimize_wf_v12_180.py` - `_185.py` | Per-fold single optimizers | VM tmux or Colab |
| `score_missing_180_185.py` | Score 26 new tickers (inference) | Colab GPU |
| `daily_score_combined.py` | Daily S3 live scoring | Colab |
| `daily_ohlcv_1.py` / `_2.py` | Daily OHLCV data pull | Colab |
| `daily_micro_all.py` | Daily microstructure data | Colab |
| `daily_stitch_master.py` | Stitch OHLCV + micro | Colab |
| `backtest_tier_comparison.py` | Strategy backtesting (G/H) | Colab |
| `compare_strategies.py` | BQ-only strategy comparison (S1-S11) | Colab |
| `analyze_wf_v12.py` | WF results analysis | Colab |
| `analyze_composite_v3.py` | Composite score deep-dive | Colab |
| `rebuild_micro_incremental.py` | Rebuild micro data for specific tickers | Colab |
