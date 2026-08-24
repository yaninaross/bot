# data_ops — Data Operations Scripts

Scripts for bulk data downloads, backfills, rebuilds, audits, and diagnostics.
None of these are part of the daily scoring pipeline (see `pipeline/` and `scoring/`).

## Data Architecture

All data lives on GCS: `gs://time-series-bucket-ys/`

```
Data/Inference/          ← raw 1-min OHLCV parquets ({TICKER}_1m_full_historical_data.parquet)
Data/Inference_v2/       ← master_inference parquets ({TICKER}_master_inference.parquet)
Data/Microstructure/     ← microstructure feature parquets ({TICKER}_microstructure.parquet)
Data/TradesQuotes/       ← raw T&Q from Polygon ({TICKER}/{TICKER}_trades_{DATE}.parquet)
```

Pipeline per ticker: **OHLCV → T&Q → Microstructure → Master Inference**

Master inference = OHLCV joined with microstructure features (Spread, OBI, Volume_Delta, etc.).
Context assets (SPY, VIX, TLT, IWM) get OHLCV-only masters (no T&Q needed).

## Authentication

- **Colab / interactive**: `gcloud auth application-default login` — expires after ~1 hour in non-interactive sessions.
- **VM long-running**: Use `--sa` flag or `USE_SERVICE_ACCOUNT=1` env var. Uses VM compute service account (`compute_engine.Credentials()`) which auto-refreshes via metadata server, never expires.
- **Never use GCS FUSE mount** — unreliable. Always use `google.cloud.storage` API.

## Key Scripts

### Bulk Refresh (primary)

| Script | Purpose |
|--------|---------|
| `bulk_micro_all.py` | **Main refresh pipeline.** Downloads missing T&Q from Polygon, rebuilds micro + master_inference. Auto-detects stale tickers, prioritizes most-stale first. Skips dead tickers (SMLR, BITF, MIGI, XLE). |
| `run_bulk_refresh.sh` | Launcher for `bulk_micro_all.py --sa` with timestamped logging to `/tmp/`. |

**Usage:**
```bash
# Auto-detect stale tickers, service account auth (VM)
python bulk_micro_all.py --sa

# Force all tickers
python bulk_micro_all.py --all --sa

# Specific tickers only
python bulk_micro_all.py MARA KEEL --sa

# Long-running on VM (survives SSH disconnect)
screen -dmS bulk_refresh ./run_bulk_refresh.sh
screen -r bulk_refresh     # reattach to monitor
# Ctrl+A, D to detach
```

### Bulk Downloads

| Script | Purpose |
|--------|---------|
| `bulk_ohlcv.py` | One-time OHLCV backfill for all assets + context. |
| `bulk_ohlcv_download.py` | Bulk 1-min OHLCV from Polygon, saves to GCS. |
| `bulk_tq_download.py` | Bulk raw T&Q from Polygon, parallel threads. Idempotent — skips existing dates. |
| `bulk_stitch_master.py` | One-time OHLCV + micro join for all assets. No prerequisite checks. |
| `bulk_update_outdated.py` | Auto-updates outdated master_inference files. For Colab parallel use. |

### Backfills

| Script | Purpose |
|--------|---------|
| `backfill_micro_parallel.py` | Bulk T&Q backfill for Exp 400 teachers (Colab). |
| `backfill_micro_vm.py` | T&Q backfill for remaining scoring assets (VM). |
| `backfill_micro_june.py` | Backfill T&Q through June 30 for all 81 assets. |

### Rebuilds

| Script | Purpose |
|--------|---------|
| `rebuild_master_all.py` | Full rebuild of all master_inference from existing raw data. Writes to `Inference_v2/`. |
| `rebuild_master_trimmed.py` | Rebuild with trimmed date range. |
| `rebuild_micro_from_raw.py` | Rebuild microstructure from raw T&Q. |
| `rebuild_micro_incremental.py` | Incremental micro rebuild — only processes new T&Q days. |
| `rebuild_1a_refetch_scored.py` | Re-fetch raw data for scored tickers. |
| `rebuild_1b_refetch_unscored.py` | Re-fetch raw data for unscored tickers. |
| `rebuild_2_delete_downstream.py` | Delete derived parquets so they can be rebuilt fresh. |

### Audits & Diagnostics

| Script | Purpose |
|--------|---------|
| `audit_master_coverage.py` | Data coverage audit for Exp 400 teacher candidates. |
| `audit_master_quality.py` | Check which masters have real micro data vs OHLCV-only. |
| `audit_scored_vs_micro.py` | Compare scored results against micro data. |
| `audit_wf_v12_completeness.py` | Check walkforward v12 result completeness. |
| `diagnose_incomplete_tickers.py` | Root-cause why tickers fail the 5% micro quality gate. |
| `diagnose_scoring.py` | Compare live model scoring vs pre-computed CSVs. |
| `debug_s3_gate.py` | Debug S3 gate scoring issues. |
| `list_tq_coverage.py` | T&Q data coverage scanner — date ranges and file counts. |

### Migrations & Utilities

| Script | Purpose |
|--------|---------|
| `migrate_bitf_to_keel.py` | Rename BITF data files to KEEL (ticker change). |
| `migrate_migi_to_bgde.py` | Rename MIGI data files to BGDE (ticker change). |
| `copy_results_to_v2.py` | Copy scoring results from `Results_all/` to `Results_all_v2/`. |
| `delete_derived_parquets.py` | Delete micro + master parquets for fresh rebuild. Keeps raw OHLCV and T&Q. |
| `check_migi.py` | Verify MIGI/BGDE data state. |
| `check_training_artifacts.py` | Verify training output files exist. |
| `check_v2_artifacts.py` | Verify v2 pipeline artifacts. |
| `verify_exp230_spike_rates.py` | Verify spike rates for Exp 230. |
| `pull_deploy_configs.py` | Pull deploy configs from BQ/GCS. |
| `pull_v2_deploy_configs.py` | Pull v2 deploy configs. |
| `build_crypto_master_inference.py` | Build master_inference for crypto tickers. |
| `build_micro_153.py` | Build microstructure for Exp 153 tickers. |
| `time_machine_score.py` | Re-score historical timestamps, compare training vs scoring pipeline. |
| `profit_calculator.py` | Strategy return projector with daily compounding + quarterly tax. |

## Dead Tickers (skip list)

These are excluded from all bulk operations:
- **SMLR** — delisted Jan 2026
- **BITF** → renamed to **KEEL**
- **MIGI** → renamed to **BGDE**
- **XLE** — context asset, not traded

## Conventions

- All scripts are Colab-ready (no `if __name__` guard). Use config variables at top.
- Never `bfill` in `heal_data` — forward-fill prices, zero-fill volumes.
- Polygon API rate limit: `POLYGON_SLEEP = 0.35` between requests.
- Master inference files go to `Data/Inference_v2/` (not `Data/Inference/`).
