"""
daily_stitch_master.py — Atomic OHLCV + Micro tail-join to master_inference
============================================================================
Run AFTER daily_micro_all.py + daily_ohlcv_1.py + daily_ohlcv_2.py complete.
Checks pipeline_status.json, then for each ticker:
  1. Loads OHLCV tail (last TAIL_DAYS)
  2. Loads micro tail (last TAIL_DAYS) — skipped for context assets
  3. Left-joins OHLCV + micro (trimmed to micro range), heals gaps
  4. Concats with head of existing master_inference
  5. Saves atomic master_inference parquet

Context assets (VIX, SPY, TLT, IWM) get OHLCV-only masters for regime filter.
Uses google.cloud.storage API + local /tmp reads to avoid gcsfs memory overhead.
"""

import gc
import os
import ctypes
import json
import warnings
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage as gcs_storage

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TARGET_TICKERS = [
    # S3 deploy tickers (11)
    'APLD', 'CIFR', 'CLSK', 'FLNC', 'HUT', 'IREN', 'KEEL', 'MARA', 'PL', 'RCAT', 'RIOT',
    # Train teachers needed for model inference (not already above)
    'ASTS', 'BBAI', 'LUNR', 'MVIS', 'WULF',
    # Additional tickers for broader WF coverage
    'ALAB', 'OKLO', 'RKLB', 'RDW', 'SMCI',
    # SHORT RULE_1 tickers (not already above)
    'BE', 'ONDS', 'WOLF',
]

CONTEXT_ASSETS = ['VIX', 'TLT', 'IWM', 'SPY']

TICKER_DATA_START = {
    'FCEL': pd.Timestamp("2025-08-01", tz='UTC'),
    'MU':   pd.Timestamp("2026-01-01", tz='UTC'),
}
DATA_START_DEFAULT = pd.Timestamp("2022-01-01", tz='UTC')

TAIL_DAYS = 5
MAX_WORKERS = 2
BATCH_SIZE = 20

BUCKET_NAME     = "time-series-bucket-ys"
GCS_OHLCV_DIR   = "Data/Inference"
GCS_MICRO_DIR   = "Data/Inference/Microstructure"
GCS_MASTER_DIR  = "Data/Inference_v2"
GCS_STATUS_BLOB = "Data/Inference/pipeline_status.json"

gcs_client = gcs_storage.Client()
bucket     = gcs_client.bucket(BUCKET_NAME)


# ==============================================================================
# GCS I/O — download to /tmp, read locally, upload from /tmp
# ==============================================================================
def _download(blob_path):
    local = f"/tmp/_stitch_{os.path.basename(blob_path)}"
    bucket.blob(blob_path).download_to_filename(local)
    return local

def _read_parquet(blob_path):
    local = _download(blob_path)
    df = pd.read_parquet(local, engine='pyarrow')
    os.remove(local)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df

def _read_arrow(blob_path):
    """Read parquet from GCS as Arrow table (~40% less memory than pandas)."""
    local = _download(blob_path)
    table = pq.read_table(local)
    os.remove(local)
    return table

def _find_ts_col(table):
    """Find the timestamp/index column name in an Arrow table."""
    meta = table.schema.pandas_metadata
    if meta:
        for c in meta.get('index_columns', []):
            if isinstance(c, str) and c in table.schema.names:
                return c
    for f in table.schema:
        if pa.types.is_timestamp(f.type):
            return f.name
    return table.schema.names[0]

def _arrow_max_ts(table, ts_col):
    """Get max timestamp as tz-aware pd.Timestamp."""
    val = pc.max(table.column(ts_col)).as_py()
    ts = pd.Timestamp(val)
    return ts if ts.tzinfo else ts.tz_localize('UTC')

def _ts_scalar(ts, arrow_type):
    """Convert pd.Timestamp to Arrow scalar matching the column's type."""
    if arrow_type.tz:
        t = ts if ts.tzinfo else ts.tz_localize('UTC')
    else:
        t = ts.tz_localize(None) if ts.tzinfo else ts
    return pa.array([t], type=arrow_type)[0]

def _arrow_filter_tail(table, ts_col, cutoff_ts):
    """Filter Arrow table to rows >= cutoff_ts, return as pandas DataFrame."""
    scalar = _ts_scalar(cutoff_ts, table.column(ts_col).type)
    mask = pc.greater_equal(table.column(ts_col), scalar)
    df = table.filter(mask).to_pandas()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df

def _save_parquet(df, blob_path):
    local = f"/tmp/_stitch_{os.path.basename(blob_path)}"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, local, row_group_size=500_000)
    del table
    bucket.blob(blob_path).upload_from_filename(local)
    os.remove(local)


# ==============================================================================
# PIPELINE STATUS
# ==============================================================================
def _read_pipeline_status():
    blob = bucket.blob(GCS_STATUS_BLOB)
    try:
        return json.loads(blob.download_as_text())
    except Exception:
        return {}

def _write_pipeline_status(stage, status_val):
    blob = bucket.blob(GCS_STATUS_BLOB)
    try:
        existing = json.loads(blob.download_as_text())
    except Exception:
        existing = {}
    existing[stage] = {
        'status': status_val,
        'timestamp': pd.Timestamp.now(tz='UTC').isoformat(),
    }
    blob.upload_from_string(json.dumps(existing, indent=2))

def _check_prerequisites():
    status = _read_pipeline_status()
    today = pd.Timestamp.now(tz='UTC').normalize()
    missing = []
    for stage in ['daily_micro', 'ohlcv_1', 'ohlcv_2']:
        info = status.get(stage, {})
        if info.get('status') != 'complete':
            missing.append(f"{stage}: not complete")
            continue
        ts = pd.Timestamp(info.get('timestamp', '2000-01-01'))
        if ts.tz is None:
            ts = ts.tz_localize('UTC')
        if ts.normalize() < today:
            missing.append(f"{stage}: stale ({ts.strftime('%Y-%m-%d')})")
    return missing


# ==============================================================================
# HEAL (forward-fill OHLC, zero-fill volume cols)
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# STITCH ONE TICKER
# ==============================================================================
def stitch_ticker(ticker, is_context=False):
    tag = f"  {ticker}:"
    ohlcv_blob  = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
    micro_blob  = f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet"
    master_blob = f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet"

    # ── Load OHLCV as Arrow → extract tail as pandas (small) ──
    try:
        ohlcv_t = _read_arrow(ohlcv_blob)
    except Exception:
        print(f"{tag} ❌ No OHLCV — skip")
        return False

    ts_col = _find_ts_col(ohlcv_t)
    ohlcv_max = _arrow_max_ts(ohlcv_t, ts_col)
    tail_start = ohlcv_max - timedelta(days=TAIL_DAYS)
    df_ohlcv_tail = _arrow_filter_tail(ohlcv_t, ts_col, tail_start)
    del ohlcv_t
    _release_memory()

    # ── Load micro tail as Arrow → pandas (small) — skip for context ──
    micro_tail = pd.DataFrame()
    if not is_context:
        try:
            micro_t = _read_arrow(micro_blob)
            ts_col_m = _find_ts_col(micro_t)
            micro_tail = _arrow_filter_tail(micro_t, ts_col_m, tail_start)
            del micro_t
            _release_memory()
        except Exception:
            pass

    # ── Join tail (trim OHLCV to micro range — no zeroed-micro bars) ──
    if is_context:
        df_tail = df_ohlcv_tail
    elif micro_tail.empty:
        print(f"{tag} ⚠️  No micro — keeping existing master")
        del df_ohlcv_tail, micro_tail
        _release_memory()
        return False
    else:
        micro_max = micro_tail.index.max()
        ohlcv_tail_max = df_ohlcv_tail.index.max()
        if ohlcv_tail_max > micro_max:
            trimmed = int((df_ohlcv_tail.index > micro_max).sum())
            df_ohlcv_tail = df_ohlcv_tail.loc[df_ohlcv_tail.index <= micro_max]
            print(f"{tag} ✂️  Trimmed {trimmed} bars past micro ({micro_max})")
        if df_ohlcv_tail.empty:
            print(f"{tag} ⚠️  No OHLCV in micro range — skip")
            del df_ohlcv_tail, micro_tail
            _release_memory()
            return False
        df_tail = df_ohlcv_tail.join(micro_tail, how='left')
    del df_ohlcv_tail, micro_tail
    _release_memory()

    df_tail = heal_data(df_tail)

    # ── Try loading existing master as Arrow ──
    try:
        master_t = _read_arrow(master_blob)
    except Exception:
        master_t = None

    if master_t is not None:
        # ── Tail stitch via Arrow streaming write ──
        # Peak memory = Arrow table only (~40% less than pandas)
        # Zero-copy slice for head, stream to disk, free before writing tail
        ts_col_m = _find_ts_col(master_t)
        col = master_t.column(ts_col_m)
        scalar = _ts_scalar(tail_start, col.type)
        n_head = int(pc.sum(pc.less(col, scalar)).as_py())
        head_t = master_t.slice(0, n_head)

        tail_arrow = pa.Table.from_pandas(df_tail)
        aligned = []
        for field in head_t.schema:
            if field.name in tail_arrow.column_names:
                c = tail_arrow.column(field.name)
                if c.type != field.type:
                    c = c.cast(field.type, safe=False)
                aligned.append(c)
            else:
                aligned.append(pa.nulls(tail_arrow.num_rows, type=field.type))
        tail_aligned = pa.table(aligned, schema=head_t.schema)
        del tail_arrow

        local_out = f"/tmp/_stitch_{os.path.basename(master_blob)}"
        writer = pq.ParquetWriter(local_out, head_t.schema)
        writer.write_table(head_t, row_group_size=500_000)
        n_total = n_head + tail_aligned.num_rows
        del head_t, master_t
        _release_memory()
        writer.write_table(tail_aligned)
        writer.close()
        del tail_aligned, df_tail

        bucket.blob(master_blob).upload_from_filename(local_out)
        os.remove(local_out)
        _release_memory()

    else:
        # ── Full rebuild (no existing master) — pandas path ──
        print(f"{tag} full rebuild (no existing master)")
        del df_tail
        _release_memory()

        df_ohlcv_full = _read_parquet(ohlcv_blob)
        trim_start = TICKER_DATA_START.get(ticker, DATA_START_DEFAULT)
        df_ohlcv_full = df_ohlcv_full.loc[df_ohlcv_full.index >= trim_start]

        if not is_context:
            try:
                df_micro_full = _read_parquet(micro_blob)
                micro_max = df_micro_full.index.max()
                df_ohlcv_full = df_ohlcv_full.loc[df_ohlcv_full.index <= micro_max]
                df_master = df_ohlcv_full.join(df_micro_full, how='left')
                del df_micro_full
                _release_memory()
            except Exception:
                print(f"{tag} ⚠️  No micro — cannot rebuild")
                del df_ohlcv_full
                _release_memory()
                return False
        else:
            df_master = df_ohlcv_full.copy()
        del df_ohlcv_full
        _release_memory()
        df_master = heal_data(df_master)
        df_master = df_master[df_master.index.notna()].sort_index()
        _save_parquet(df_master, master_blob)
        n_total = len(df_master)
        del df_master
        _release_memory()

    print(f"{tag} ✅ {n_total:,} rows")
    return True


# ==============================================================================
# MAIN
# ==============================================================================
import time as _time

print("🚀 daily_stitch_master.py — Parallel OHLCV + Micro Stitch")
print(f"   Run time : {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}")
print(f"   Tickers  : {len(TARGET_TICKERS)} + {len(CONTEXT_ASSETS)} context")
print(f"   Workers  : {MAX_WORKERS}")

missing = _check_prerequisites()
if missing:
    print(f"\n   ⚠️  PREREQUISITE WARNING:")
    for m in missing:
        print(f"      - {m}")
    print(f"   Proceeding anyway — master files may use stale source data.\n")

_write_pipeline_status('stitch', 'running')
_t0 = _time.time()

success = 0
failed  = 0

ALL_TICKERS = [(t, False) for t in TARGET_TICKERS] + [(t, True) for t in CONTEXT_ASSETS]
batches = [ALL_TICKERS[i:i+BATCH_SIZE] for i in range(0, len(ALL_TICKERS), BATCH_SIZE)]

for b_idx, batch in enumerate(batches, 1):
    print(f"\n── Batch {b_idx}/{len(batches)} ({len(batch)} tickers) ──")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(stitch_ticker, ticker, is_context): ticker
            for ticker, is_context in batch
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                if future.result():
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  {ticker}: ❌ Error: {e}")
                failed += 1
    _release_memory()

_write_pipeline_status('stitch', 'complete')
elapsed = _time.time() - _t0

print(f"\n{'═'*52}")
print(f"🎉 daily_stitch_master.py COMPLETE")
print(f"   Success: {success}  |  Failed: {failed}")
print(f"   Elapsed: {elapsed:.0f}s  ({elapsed/60:.1f}m)")
print(f"{'═'*52}")
