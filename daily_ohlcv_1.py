"""
daily_ohlcv_1.py — OHLCV Refresh + Master Inference Stitch (batch 1 of 2)
==========================================================================
Run in parallel with daily_ohlcv_2.py for faster pipeline.

Batch 1 (10 tickers + context):
  RCAT, ASTS, HUT, SPIR, CIFR, WULF, RKLB, KEEL, IREN, CLSK
  Context: ITA, I:VIX, TLT, IWM, SPY

Run AFTER daily_micro_all.py has finished updating GCS.
"""

import os
import time
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from polygon import RESTClient
from google.cloud import storage as gcs_storage
import gc
import ctypes
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
    'RCAT', 'ASTS', 'HUT', 'SPIR',        # 152/160 teachers
    'CIFR', 'WULF', 'RKLB',               # 152/160 teachers
    'KEEL', 'IREN', 'CLSK',               # 152/160 teachers
]

CONTEXT_ASSETS = ['ITA', 'I:VIX', 'TLT', 'IWM', 'SPY']

POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
BUCKET_NAME     = "time-series-bucket-ys"

GCS_OHLCV_DIR   = f"gs://{BUCKET_NAME}/Data/Inference"
GCS_MICRO_DIR   = f"gs://{BUCKET_NAME}/Data/Inference/Microstructure"
GCS_MASTER_DIR  = f"gs://{BUCKET_NAME}/Data/Inference_v2"

poly_client = RESTClient(POLYGON_API_KEY)
gcs_client  = gcs_storage.Client()

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
# HEAL (forward-fill OHLC, zero-fill volume cols)
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df

# ==============================================================================
# MAIN LOOP
# ==============================================================================
if __name__ == "__main__":
    print("🚀 daily_ohlcv_1.py — OHLCV Batch 1 of 2")
    fetch_end = pd.Timestamp.now(tz='UTC')
    print(f"   Run time      : {fetch_end.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"   Scored Assets : {len(TARGET_TICKERS)}")
    print(f"   Context Assets: {len(CONTEXT_ASSETS)}")

    ALL_ASSETS = [(t, False) for t in list(set(TARGET_TICKERS))] + [(t, True) for t in list(set(CONTEXT_ASSETS))]

    for ticker, is_context in ALL_ASSETS:
        print(f"\n{'─'*52}")
        print(f"🔄 {ticker}{'  [context — OHLCV only]' if is_context else ''}")
        clean_ticker = ticker.replace("I:", "").replace("X:", "")

        ohlcv_uri  = f"{GCS_OHLCV_DIR}/{clean_ticker}_1m_full_historical_data.parquet"
        micro_uri  = f"{GCS_MICRO_DIR}/{clean_ticker}_1m_microstructure.parquet"
        master_uri = f"{GCS_MASTER_DIR}/{clean_ticker}_master_inference.parquet"

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

        new_bars_fetched = False
        if ohlcv_start < fetch_end:
            print(f"   ⏳ Fetching OHLCV from {ohlcv_start.date()}...")
            new_aggs = fetch_polygon_aggs(ticker, ohlcv_start, fetch_end)
            if not new_aggs.empty:
                df_ohlcv = pd.concat([df_ohlcv, new_aggs])
                df_ohlcv = df_ohlcv[~df_ohlcv.index.duplicated(keep='last')].sort_index()
                df_ohlcv.to_parquet(ohlcv_uri, engine='pyarrow')
                print(f"   📥 {len(new_aggs):,} new bars  |  total: {len(df_ohlcv):,}")
                new_bars_fetched = True
            else:
                print(f"   ℹ️  No new OHLCV bars returned")
        else:
            print(f"   ✅ OHLCV already up to date")

        # Trim OHLCV to last 5 days — full data already saved to GCS
        ohlcv_max_ts = df_ohlcv.index.max()
        if len(df_ohlcv) > 10000:
            df_ohlcv = df_ohlcv[df_ohlcv.index >= ohlcv_max_ts - timedelta(days=5)].copy()
            _release_memory()

        df_micro = pd.DataFrame()
        if not is_context:
            try:
                df_micro = pd.read_parquet(micro_uri)
                if df_micro.index.tz is None:
                    df_micro.index = df_micro.index.tz_localize('UTC')
                micro_total = len(df_micro)
                if len(df_micro) > 10000:
                    df_micro = df_micro[df_micro.index >= ohlcv_max_ts - timedelta(days=5)].copy()
                    _release_memory()
                print(f"   📂 Micro loaded: {micro_total:,} rows, trimmed to {len(df_micro):,}")
            except Exception:
                print(f"   ⚠️  No micro found for {ticker} — run daily_micro scripts first")

        need_stitch = new_bars_fetched
        if not need_stitch and not is_context:
            try:
                bucket = gcs_client.bucket(BUCKET_NAME)
                master_blob = bucket.blob(master_uri.replace(f"gs://{BUCKET_NAME}/", ""))
                micro_blob  = bucket.blob(micro_uri.replace(f"gs://{BUCKET_NAME}/", ""))
                if master_blob.exists() and micro_blob.exists():
                    master_blob.reload()
                    micro_blob.reload()
                    if micro_blob.updated > master_blob.updated:
                        need_stitch = True
                        print(f"   🔄 Micro newer than master_inference — re-stitching")
            except Exception:
                pass

        if need_stitch:
            tail_start = ohlcv_max_ts - timedelta(days=3)
            # df_ohlcv/df_micro already trimmed to 5 days — extract 3-day tail, then free
            df_ohlcv_tail = df_ohlcv[df_ohlcv.index >= tail_start].copy()
            micro_tail = pd.DataFrame()
            if not is_context and not df_micro.empty:
                micro_tail = df_micro[df_micro.index >= tail_start].copy()
            del df_ohlcv, df_micro; _release_memory()

            try:
                df_existing_master = pd.read_parquet(master_uri)
                if df_existing_master.index.tz is None:
                    df_existing_master.index = df_existing_master.index.tz_localize('UTC')

                df_tail = df_ohlcv_tail
                if not micro_tail.empty:
                    df_tail = df_tail.join(micro_tail, how='left')
                del df_ohlcv_tail, micro_tail; _release_memory()
                df_tail = heal_data(df_tail)

                df_head = df_existing_master[df_existing_master.index < tail_start]
                df_unified = pd.concat([df_head, df_tail])
                df_unified = df_unified[~df_unified.index.duplicated(keep='last')]
                del df_existing_master, df_head, df_tail
                print(f"   📐 Tail-only heal from {tail_start.date()}")
            except Exception:
                print(f"   📐 No existing master — full heal")
                for _v in [df_ohlcv_tail, micro_tail]:
                    try: del _v
                    except: pass
                _release_memory()
                df_ohlcv_full = pd.read_parquet(ohlcv_uri)
                if df_ohlcv_full.index.tz is None:
                    df_ohlcv_full.index = df_ohlcv_full.index.tz_localize('UTC')
                if not is_context:
                    try:
                        df_micro_full = pd.read_parquet(micro_uri)
                        if df_micro_full.index.tz is None:
                            df_micro_full.index = df_micro_full.index.tz_localize('UTC')
                        df_unified = df_ohlcv_full.join(df_micro_full, how='left')
                        del df_micro_full
                    except Exception:
                        df_unified = df_ohlcv_full.copy()
                else:
                    df_unified = df_ohlcv_full.copy()
                del df_ohlcv_full
                df_unified = heal_data(df_unified)

            _release_memory()
            df_unified = df_unified[df_unified.index.notna()].sort_index()
            df_unified.to_parquet(master_uri, engine='pyarrow')
            print(f"   ✅ master_inference saved  ({len(df_unified):,} rows)")
            del df_unified; _release_memory()
        else:
            print(f"   ⏭️  Skipping stitch: No new data to process.")
            del df_ohlcv, df_micro; _release_memory()

    print("\n🎉 daily_ohlcv_1.py COMPLETE.")
