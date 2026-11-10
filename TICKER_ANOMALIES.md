# Ticker Anomalies & Rename Handling

## BITF → KEEL (Nasdaq rename, 2026-04-06)

**What happened:** BITF was renamed to KEEL on Nasdaq on April 6, 2026.

**Raw T&Q data — the empty stub problem:** `bulk_tq_download.py` ran for KEEL
before the migration script. It queried Polygon for all dates back to 2022, but
Polygon returns no data for "KEEL" before the rename date (it was "BITF" then).
The downloader writes empty 976-byte parquet stubs for dates with no data
(`pd.DataFrame().to_parquet()`). This created ~2,226 empty stub files in
`Data/TradesQuotes/KEEL/`.

When `migrate_bitf_to_keel.py --copy` ran later, it checked `if dst_blob.exists()`
and skipped every file — the empty stubs blocked all real BITF data from being
copied. Result: KEEL/ has 2,278 files but only 52 contain real data (26 post-rename
trading days). BITF/ still has all the real historical data (~2,146 files with data).

**Fix (two steps):**

Step 1 — Re-run migration to overwrite empty stubs with real BITF data:
```python
# In migrate_bitf_to_keel.py, set COPY_MODE = True and run the cell.
# The script now checks file size >2KB before skipping (fixed 2026-05-15).
```

Step 2 — Force rebuild microstructure from the now-complete raw T&Q:
```python
# In rebuild_micro_incremental.py, set:
RUN_TICKERS = ['KEEL']
FORCE_REBUILD = True
# Then run the cell.
```

**Scoring / training:** All scripts use `TICKER_FILE_MAP = {'BITF': 'KEEL'}` to
load data files under the KEEL name. Embedding indices are preserved — KEEL
inherits BITF's index position in train_teachers lists (e.g. EXP-97 index 9).

**After force rebuild:** master_inference is automatically rebuilt. No retraining
needed — models were trained with BITF in the teacher list and the embedding
index mapping handles the alias.

---

## MIGI → BGDE (Nasdaq rename)

**What happened:** MIGI was renamed to BGDE on Nasdaq.

**Handling:** Same pattern as BITF→KEEL. `TICKER_FILE_MAP = {'MIGI': 'BGDE'}`
in all scoring/training scripts. Data files are stored under the BGDE name.
Embedding index inherited from MIGI's position in train_teachers.

**Status:** No known data issues — micro was rebuilt correctly.

---

## SMLR (Delisted, January 2026)

**What happened:** SMLR was delisted from Nasdaq around January 2026.

**Training:** Kept in `TRAIN_TEACHERS` lists (e.g. EXP-97 index 15) for embedding
index stability — removing it would shift all subsequent indices and break
saved model weights.

**Scoring:** Excluded from `daily_score_combined.py` — no live data available.
WF optimizer shows it as deploy-ready (12.92% avg PnL) based on historical
validation, but it cannot be traded.

**Data:** Historical OHLCV, micro, and master_inference files remain in GCS.
No new data is generated.

---

## General: Rename Handling Checklist

When a ticker is renamed on an exchange:

1. **Raw T&Q:** Copy all files from old to new directory in `Data/TradesQuotes/`
2. **Microstructure:** Force rebuild under new name:
   `main(ticker_list=['NEW_TICKER'], force=True)`
3. **OHLCV:** Download fresh under new ticker name via `bulk_ohlcv_download.py`
4. **TICKER_FILE_MAP:** Add `{'OLD': 'NEW'}` in all scripts:
   - `daily_score_combined.py`
   - `exp_*.py` (training scripts)
   - `daily_micro_all.py`
   - `rebuild_micro_incremental.py` (if ticker appears in ALL_TICKERS)
5. **Train teachers:** Keep old name in the list for index stability
6. **Embedding index:** The `asset_index` loop propagates aliases automatically:
   ```python
   if 'OLD' in _idx:
       _idx['NEW'] = _idx['OLD']
   ```
7. **Bulk download scripts:** Add new ticker name to download lists

---

## Reverse-Split Price & `Spread / Close` Distortion in `Master_Inference_v2` (Audited 2026-11-09)

**What happened:** In `Data/Inference_v2/{TICKER}_master_inference.parquet`, 1-minute bar `Open`, `High`, `Low`, `Close`, and `Volume` are fetched from Polygon with `adjusted=True` (backward-adjusted for all subsequent reverse splits), whereas `Spread` (`ask_price - bid_price` from `Data/Quotes/`) and `Total_Vol` (raw trade volume from `Data/Trades/`) are **unadjusted point-in-time values**.

**Why this distorts both Rule-Based and `T&Q` ML backtests if uncorrected:**
1. **Fake `$10–$220+` Open Prices on Sub-`$2` Penny Stocks:** When a penny stock trading at `$0.45–$1.60` in 2024–2025 later executes a `1-for-20` to `1-for-60,000` reverse split in 2025–2026, its historical split-adjusted `Open` appears to be `$10.20–$17,043.00`, slipping past `$10.00 <= open_price <= $100.00` price floors.
2. **Fake Ultra-Tight Percentage Spreads (`Spread / Close`):** Dividing an unadjusted `$0.02` penny-stock spread by a `30x–450x` split-adjusted `Close` shrinks `tq_spread_pct` from a toxic `2.0%` to an artificial `0.005%–0.05%`, deceiving `T&Q` ML models into treating illiquid penny stocks as institutional-grade liquidity.

**29 Affected Tickers in the 157-Stock `T&Q` Universe (`2024-06-01 → 2026-11-06`):**
- **Extreme Reverse Splits (`80x–60,000x`):** `VIVK` (`60,000x`, true med open `$0.84`), `SDOT` (`2,000x`, true med open `$3.20`), `AZI` (`500x`, true med open `$0.79`), `BIAF` (`450x`, true med open `$1.37`), `VEEE` (`370x`, true med open `$1.67`), `BIYA` (`250x`, true med open `$1.17`), `MNTS` (`249.9x`, true med open `$3.90`), `AKAN` (`140.6x`, true med open `$2.46`), `MASK` (`100x`, true med open `$1.40`), `EHGO` (`80x`, true med open `$2.31`), `WETO` (`80x`).
- **Heavy Reverse Splits (`8x–35x`):** `VRAX` (`35x`), `OPTT` (`30x`, `$0.44`), `RVSN` (`30x`, `$0.45`), `IXHL` (`30x`, `$0.49`), `FCEL` (`30x`, `$9.09`), `AIRE` (`25x`, `$0.62`), `OMEX` (`25x`, `$1.27`), `SCNX` (`25x`, `$0.89`), `ARQQ` (`25x`, `$18.44`), `ASST` (`20x`, `$2.58`), `DUO` (`16x`, `$1.14`), `CGTL` (`15x`, `$1.31`), `IBO` (`12.6x`, `$1.20`), `FUBO` (`12x`, `$3.81`), `MOVE` (`10x`), `BNAI` (`10x`, `$2.21`), `DGNX` (`8x`, `$23.35`), `BKSY` (`8x`, `$18.67`).

**Standard Fix (Applied in `audit_and_run_unadj_clean_142_tq.py`, `run_expanded_tq_universe_full_analysis.py` & `deep_synergy_forensics_and_architectures.py`):**
- Compute the exact point-in-time split factor via either Polygon daily `adjusted=false` (`split_factor = open_px / unadj_open_px`) or directly from `Master_Inference_v2` (`split_factor = median(Total_Vol / Volume)`), requiring `split_factor <= 1.25` and `$10.00 <= unadj_open_px <= $100.00`.
- Multiply `tq_spread_pct` by `split_factor` to restore the true unadjusted percentage spread.


