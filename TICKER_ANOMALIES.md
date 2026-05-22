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
