"""
bulk_stitch_master.py — One-time OHLCV + Micro join for ALL assets
===================================================================
Bulk version of daily_stitch_master.py. Rebuilds master_inference parquets
by joining OHLCV + microstructure data. Context assets get OHLCV-only masters.

NO prerequisite checks (pipeline_status.json) — runs unconditionally.
NOT part of the daily scoring pipeline — use for one-time data preparation.
Colab-ready (no if __name__).
"""

import gc, os, ctypes, warnings, sys, subprocess
import time as _time
import pandas as pd

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect(); gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception: pass
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "pyarrow", "google-cloud-storage", "--quiet"])
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME    = "time-series-bucket-ys"
GCS_OHLCV_DIR  = "Data/Inference"
GCS_MICRO_DIR  = "Data/Inference/Microstructure"
GCS_MASTER_DIR = "Data/Inference_v2"

# All tickers that need master_inference files
TARGET_TICKERS = [
    'ACHR', 'ALAB', 'AMD', 'APLD', 'ASTS', 'AVAV', 'BBAI',
    'BE', 'BGDE', 'BHP', 'BKSY', 'BTBT',
    'CAT', 'CCJ', 'CEG', 'CIFR', 'CLSK', 'COIN', 'COP', 'CORZ', 'CVX',
    'DUK', 'EQIX', 'ETN', 'FCEL', 'FCX', 'FLNC',
    'GD', 'GE', 'GEV', 'HOOD', 'HUT', 'IREN',
    'JOBY', 'KEEL', 'KTOS',
    'LHX', 'LIN', 'LMT', 'LUNR',
    'MARA', 'MNTS', 'MOD', 'MP', 'MSTR', 'MU', 'MVIS',
    'NBIS', 'NEE', 'NNDM', 'NOC', 'NVDA',
    'OKLO', 'PL', 'PLTR', 'PLUG', 'POWL',
    'RCAT', 'RDW', 'RIOT', 'RKLB', 'RTX',
    'SMCI', 'SMR', 'SOUN', 'SPIR',
    'TSLA', 'VOYG', 'VRT', 'VST', 'WULF', 'XOM',
]

CONTEXT_ASSETS = ['VIX', 'TLT', 'IWM', 'SPY']

TICKER_DATA_START = {
    'FCEL': pd.Timestamp("2025-08-01", tz='UTC'),
    'MU':   pd.Timestamp("2026-01-01", tz='UTC'),
}
DATA_START_DEFAULT = pd.Timestamp("2022-01-01", tz='UTC')

TAIL_DAYS   = 30
MAX_WORKERS = 2
BATCH_SIZE  = 20

gcs_client = gcs_storage.Client()
bucket     = gcs_client.bucket(BUCKET_NAME)


# ==============================================================================
# GCS I/O
# ==============================================================================
def _download(blob_path):
    local = f"/tmp/_bulk_stitch_{os.path.basename(blob_path)}"
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
    local = _download(blob_path)
    table = pq.read_table(local)
    os.remove(local)
    return table

def _find_ts_col(table):
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
    val = pc.max(table.column(ts_col)).as_py()
    ts = pd.Timestamp(val)
    return ts if ts.tzinfo else ts.tz_localize('UTC')

def _ts_scalar(ts, arrow_type):
    if arrow_type.tz:
        t = ts if ts.tzinfo else ts.tz_localize('UTC')
    else:
        t = ts.tz_localize(None) if ts.tzinfo else ts
    return pa.array([t], type=arrow_type)[0]

def _arrow_filter_tail(table, ts_col, cutoff_ts):
    scalar = _ts_scalar(cutoff_ts, table.column(ts_col).type)
    mask = pc.greater_equal(table.column(ts_col), scalar)
    df = table.filter(mask).to_pandas()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df

def _save_parquet(df, blob_path):
    local = f"/tmp/_bulk_stitch_{os.path.basename(blob_path)}"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, local, row_group_size=500_000)
    del table
    bucket.blob(blob_path).upload_from_filename(local)
    os.remove(local)


# ==============================================================================
# HEAL
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

    try:
        ohlcv_t = _read_arrow(ohlcv_blob)
    except Exception:
        print(f"{tag} No OHLCV — skip")
        return False

    ts_col = _find_ts_col(ohlcv_t)
    ohlcv_max = _arrow_max_ts(ohlcv_t, ts_col)
    tail_start = ohlcv_max - timedelta(days=TAIL_DAYS)
    df_ohlcv_tail = _arrow_filter_tail(ohlcv_t, ts_col, tail_start)
    del ohlcv_t
    _release_memory()

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

    if is_context:
        df_tail = df_ohlcv_tail
    elif micro_tail.empty:
        print(f"{tag} No micro — keeping existing master")
        del df_ohlcv_tail, micro_tail
        _release_memory()
        return False
    else:
        micro_max = micro_tail.index.max()
        ohlcv_tail_max = df_ohlcv_tail.index.max()
        if ohlcv_tail_max > micro_max:
            trimmed = int((df_ohlcv_tail.index > micro_max).sum())
            df_ohlcv_tail = df_ohlcv_tail.loc[df_ohlcv_tail.index <= micro_max]
            print(f"{tag} Trimmed {trimmed} bars past micro ({micro_max})")
        if df_ohlcv_tail.empty:
            print(f"{tag} No OHLCV in micro range — skip")
            del df_ohlcv_tail, micro_tail
            _release_memory()
            return False
        df_tail = df_ohlcv_tail.join(micro_tail, how='left')
    del df_ohlcv_tail, micro_tail
    _release_memory()

    df_tail = heal_data(df_tail)

    try:
        master_t = _read_arrow(master_blob)
    except Exception:
        master_t = None

    if master_t is not None:
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

        local_out = f"/tmp/_bulk_stitch_{os.path.basename(master_blob)}"
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
                print(f"{tag} No micro — cannot rebuild")
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

    print(f"{tag} done — {n_total:,} rows")
    return True


# ==============================================================================
# MAIN
# ==============================================================================
print("=" * 70)
print("  BULK STITCH — Rebuild master_inference for ALL assets")
print(f"  {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}")
print(f"  Tickers: {len(TARGET_TICKERS)} + {len(CONTEXT_ASSETS)} context")
print(f"  Workers: {MAX_WORKERS}  |  Tail: {TAIL_DAYS} days")
print("=" * 70)

_t0 = _time.time()
success = 0
failed  = 0

ALL_TICKERS = [(t, True) for t in CONTEXT_ASSETS] + [(t, False) for t in sorted(set(TARGET_TICKERS))]
batches = [ALL_TICKERS[i:i+BATCH_SIZE] for i in range(0, len(ALL_TICKERS), BATCH_SIZE)]

for b_idx, batch in enumerate(batches, 1):
    print(f"\n-- Batch {b_idx}/{len(batches)} ({len(batch)} tickers) --")
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
                print(f"  {ticker}: ERROR: {e}")
                failed += 1
    _release_memory()

elapsed = _time.time() - _t0

print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  Success: {success}  |  Failed: {failed}")
print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")
print()
print("  Next: re-run score-only scripts to extend score CSVs through June 30")
print("=" * 70)
