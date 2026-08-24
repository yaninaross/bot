"""
daily_ohlcv_2.py — OHLCV Refresh (batch 2 of 2, G-X)
=====================================================
Fetches 1-min OHLCV from Polygon, appends to GCS parquets.
Run in parallel with daily_ohlcv_1.py.
Stitch step is handled separately by daily_stitch_master.py.

Batch 2 (34 deploy tickers G-X + 1 sector proxy):
  Deploy: GEV, HUT, IREN, KEEL, KTOS, LUNR, MARA, MNTS, MSTR, MXL,
          NBIS, NEE, NTLA, ONDS, PL, PLTR, PLUG, RCAT, RDW, RIOT,
          RKLB, RXT, SLS, SMCI, SMR, SNDK, SOUN, TSLA, VOYG, VRT,
          VST, WOLF, WULF, XOM
  Sector proxy: ITA (OHLCV-only, for regime features)
"""

import time
import pandas as pd
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from polygon import RESTClient
from google.cloud import storage as gcs_storage
import gc
import ctypes
import json
import warnings

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TARGET_TICKERS = [
    'GEV', 'HUT', 'IREN', 'KEEL', 'KTOS', 'LUNR', 'MARA', 'MNTS', 'MSTR', 'MXL',
    'NBIS', 'NEE', 'NTLA', 'ONDS', 'PL', 'PLTR', 'PLUG', 'RCAT', 'RDW', 'RIOT',
    'RKLB', 'RXT', 'SLS', 'SMCI', 'SMR', 'SNDK', 'SOUN', 'TSLA', 'VOYG', 'VRT',
    'VST', 'WOLF', 'WULF', 'XOM',
]

CONTEXT_ASSETS = ['ITA']

POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
BUCKET_NAME     = "time-series-bucket-ys"

GCS_OHLCV_DIR   = f"gs://{BUCKET_NAME}/Data/Inference"
GCS_STATUS_BLOB = "Data/Inference/pipeline_status.json"

poly_client = RESTClient(POLYGON_API_KEY)
gcs_client  = gcs_storage.Client()


def _write_pipeline_status(stage, status_val):
    bucket = gcs_client.bucket(BUCKET_NAME)
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


# ==============================================================================
# POLYGON OHLCV FETCH
# ==============================================================================
def fetch_polygon_aggs(ticker, start_dt, end_dt, chunk_months=1, sleep_secs=2):
    start_dt = pd.Timestamp(start_dt, tz='UTC') if not isinstance(start_dt, pd.Timestamp) \
               else (start_dt.tz_localize('UTC') if start_dt.tzinfo is None else start_dt)
    end_dt   = pd.Timestamp(end_dt,   tz='UTC') if not isinstance(end_dt,   pd.Timestamp) \
               else (end_dt.tz_localize('UTC')   if end_dt.tzinfo   is None else end_dt)

    all_aggs    = []
    chunk_start = start_dt
    total_days  = (end_dt - start_dt).days

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + relativedelta(months=chunk_months), end_dt)
        try:
            for a in poly_client.list_aggs(ticker, 1, 'minute', chunk_start, chunk_end, limit=50000):
                all_aggs.append({'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                                 'Open': a.open, 'High': a.high, 'Low': a.low,
                                 'Close': a.close, 'Volume': a.volume})
        except Exception as e:
            print(f"      ⚠️  Chunk {chunk_start.date()} → {chunk_end.date()} failed: {e}. Retrying...")
            time.sleep(10)
            try:
                for a in poly_client.list_aggs(ticker, 1, 'minute', chunk_start, chunk_end, limit=50000):
                    all_aggs.append({'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                                     'Open': a.open, 'High': a.high, 'Low': a.low,
                                     'Close': a.close, 'Volume': a.volume})
            except Exception as e2:
                print(f"      ❌ Chunk skipped: {e2}")

        if total_days > 30:
            print(f"      📦 {chunk_start.date()} → {chunk_end.date()}: {len(all_aggs):,} bars")
        chunk_start = chunk_end
        if chunk_start < end_dt:
            time.sleep(sleep_secs)

    if not all_aggs:
        return pd.DataFrame()
    df = pd.DataFrame(all_aggs).set_index('timestamp')
    return df[~df.index.duplicated(keep='last')].sort_index()


# ==============================================================================
# MAIN LOOP
# ==============================================================================
print("🚀 daily_ohlcv_2.py — OHLCV Batch 2")
fetch_end = pd.Timestamp.now(tz='UTC')
print(f"   Run time      : {fetch_end.strftime('%Y-%m-%d %H:%M UTC')}")
print(f"   Scored Assets : {len(TARGET_TICKERS)}")
print(f"   Context Assets: {len(CONTEXT_ASSETS)}")

ALL_ASSETS = [(t, False) for t in list(set(TARGET_TICKERS))] + [(t, True) for t in list(set(CONTEXT_ASSETS))]

for ticker, is_context in ALL_ASSETS:
    print(f"\n{'─'*52}")
    print(f"🔄 {ticker}{'  [context — OHLCV only]' if is_context else ''}")
    clean_ticker = ticker.replace("I:", "").replace("X:", "")

    ohlcv_uri = f"{GCS_OHLCV_DIR}/{clean_ticker}_1m_full_historical_data.parquet"

    try:
        df_ohlcv = pd.read_parquet(ohlcv_uri)
        if df_ohlcv.index.tz is None:
            df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
        ohlcv_start = df_ohlcv.index.max() + timedelta(minutes=1)
        print(f"   Existing OHLCV: {len(df_ohlcv):,} rows, last: {df_ohlcv.index.max()}")
    except Exception:
        print(f"   🌱 No OHLCV found — will fetch full history from 2022-01-01")
        df_ohlcv    = pd.DataFrame()
        ohlcv_start = pd.Timestamp("2022-01-01", tz='UTC')

    if ohlcv_start < fetch_end:
        print(f"   ⏳ Fetching OHLCV from {ohlcv_start.date()}...")
        new_aggs = fetch_polygon_aggs(ticker, ohlcv_start, fetch_end)
        if not new_aggs.empty:
            df_ohlcv = pd.concat([df_ohlcv, new_aggs])
            df_ohlcv = df_ohlcv[~df_ohlcv.index.duplicated(keep='last')].sort_index()
            df_ohlcv.to_parquet(ohlcv_uri, engine='pyarrow')
            print(f"   📥 {len(new_aggs):,} new bars  |  total: {len(df_ohlcv):,}")
        else:
            print(f"   ℹ️  No new OHLCV bars returned")
    else:
        print(f"   ✅ OHLCV already up to date")

    del df_ohlcv
    _release_memory()

_write_pipeline_status('ohlcv_2', 'complete')
print("\n🎉 daily_ohlcv_2.py COMPLETE.")
