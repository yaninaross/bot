"""
bootstrap_missing_tickers.py — Generate master_inference parquets for tickers
missing from the daily pipeline (AMD, NVDA, etc.)

Fetches full 1-minute OHLCV history from Polygon (2022-01-01 to now),
joins with any existing micro data in GCS, heals, and uploads
master_inference.parquet to GCS.

Run once on Colab Enterprise, then add tickers to daily_data_all.py
for ongoing updates.

Usage:  python bootstrap_missing_tickers.py
"""

import os
import io
import time
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from polygon import RESTClient
from google.cloud import storage as gcs_storage
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TICKERS_TO_BOOTSTRAP = ['AMD', 'NVDA']

POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
BUCKET_NAME     = "time-series-bucket-ys"
INFERENCE_DIR   = "Data/Inference"
MICRO_DIR       = "Data/Inference/Microstructure"

HISTORY_START = "2022-01-01"

poly_client = RESTClient(POLYGON_API_KEY)
gcs_client  = gcs_storage.Client()
bucket      = gcs_client.bucket(BUCKET_NAME)

try:
    from google.colab import auth
    auth.authenticate_user()
except:
    pass

# ==============================================================================
# POLYGON OHLCV FETCH (same as daily_data_all.py)
# ==============================================================================
def fetch_polygon_aggs(ticker, start_dt, end_dt, chunk_months=1, sleep_secs=2):
    """Fetch 1-minute OHLCV bars from Polygon in monthly chunks."""
    start_dt = pd.Timestamp(start_dt, tz='UTC') if not isinstance(start_dt, pd.Timestamp) \
               else (start_dt.tz_localize('UTC') if start_dt.tzinfo is None else start_dt)
    end_dt   = pd.Timestamp(end_dt,   tz='UTC') if not isinstance(end_dt,   pd.Timestamp) \
               else (end_dt.tz_localize('UTC')   if end_dt.tzinfo   is None else end_dt)

    all_aggs    = []
    chunk_start = start_dt

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + relativedelta(months=chunk_months), end_dt)
        try:
            for a in poly_client.list_aggs(ticker, 1, 'minute', chunk_start, chunk_end, limit=50000):
                all_aggs.append({'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                                 'Open': a.open, 'High': a.high, 'Low': a.low,
                                 'Close': a.close, 'Volume': a.volume})
        except Exception as e:
            print(f"      Chunk {chunk_start.date()} -> {chunk_end.date()} failed: {e}. Retrying...")
            time.sleep(10)
            try:
                for a in poly_client.list_aggs(ticker, 1, 'minute', chunk_start, chunk_end, limit=50000):
                    all_aggs.append({'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                                     'Open': a.open, 'High': a.high, 'Low': a.low,
                                     'Close': a.close, 'Volume': a.volume})
            except Exception as e2:
                print(f"      Chunk skipped: {e2}")

        print(f"      {chunk_start.date()} -> {chunk_end.date()}: {len(all_aggs):,} bars total")
        chunk_start = chunk_end
        if chunk_start < end_dt:
            time.sleep(sleep_secs)

    if not all_aggs:
        return pd.DataFrame()
    df = pd.DataFrame(all_aggs).set_index('timestamp')
    return df[~df.index.duplicated(keep='last')].sort_index()

# ==============================================================================
# HEAL (ffill prices, zero-fill volume — NO backfill)
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print(f"BOOTSTRAP — Generating master_inference for: {TICKERS_TO_BOOTSTRAP}")
    fetch_end = pd.Timestamp.now(tz='UTC')
    print(f"   History: {HISTORY_START} -> {fetch_end.strftime('%Y-%m-%d %H:%M UTC')}")

    for ticker in TICKERS_TO_BOOTSTRAP:
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")

        clean = ticker.replace("I:", "").replace("X:", "")
        ohlcv_blob_path  = f"{INFERENCE_DIR}/{clean}_1m_full_historical_data.parquet"
        micro_blob_path  = f"{MICRO_DIR}/{clean}_1m_microstructure.parquet"
        master_blob_path = f"{INFERENCE_DIR}/{clean}_master_inference.parquet"

        # --- A. Check for existing OHLCV in GCS ---
        ohlcv_blob = bucket.blob(ohlcv_blob_path)
        if ohlcv_blob.exists():
            print(f"   Found existing OHLCV — loading...")
            df_ohlcv = pd.read_parquet(io.BytesIO(ohlcv_blob.download_as_bytes()))
            if df_ohlcv.index.tz is None:
                df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
            fetch_start = df_ohlcv.index.max() + timedelta(minutes=1)
            print(f"   Existing: {len(df_ohlcv):,} rows, last: {df_ohlcv.index.max()}")
        else:
            print(f"   No existing OHLCV — fetching full history from {HISTORY_START}")
            df_ohlcv = pd.DataFrame()
            fetch_start = pd.Timestamp(HISTORY_START, tz='UTC')

        # --- B. Fetch OHLCV from Polygon ---
        if fetch_start < fetch_end:
            print(f"   Fetching OHLCV from {fetch_start.date()}...")
            new_aggs = fetch_polygon_aggs(ticker, fetch_start, fetch_end)
            if not new_aggs.empty:
                df_ohlcv = pd.concat([df_ohlcv, new_aggs])
                df_ohlcv = df_ohlcv[~df_ohlcv.index.duplicated(keep='last')].sort_index()
                print(f"   Fetched {len(new_aggs):,} new bars | total: {len(df_ohlcv):,}")
            else:
                print(f"   No new bars returned from Polygon")
        else:
            print(f"   OHLCV already up to date")

        if df_ohlcv.empty:
            print(f"   SKIPPING {ticker} — no OHLCV data available")
            continue

        # --- C. Save raw OHLCV ---
        buf = io.BytesIO()
        df_ohlcv.to_parquet(buf, engine='pyarrow')
        buf.seek(0)
        bucket.blob(ohlcv_blob_path).upload_from_file(buf, content_type='application/octet-stream')
        print(f"   Saved OHLCV: {ohlcv_blob_path} ({len(df_ohlcv):,} rows)")

        # --- D. Load micro if it exists ---
        micro_blob = bucket.blob(micro_blob_path)
        if micro_blob.exists():
            print(f"   Found micro data — joining...")
            df_micro = pd.read_parquet(io.BytesIO(micro_blob.download_as_bytes()))
            if df_micro.index.tz is None:
                df_micro.index = df_micro.index.tz_localize('UTC')
            df_master = df_ohlcv.join(df_micro, how='left')
            print(f"   Micro: {len(df_micro):,} rows, last: {df_micro.index.max()}")
        else:
            print(f"   No micro data found — micro cols will be zero-filled")
            df_master = df_ohlcv.copy()

        # --- E. Heal ---
        print(f"   Healing data (resample + ffill + zero-fill, no backfill)...")
        df_master = heal_data(df_master)
        df_master = df_master[df_master.index.notna()].sort_index()

        # --- F. Save master_inference ---
        buf = io.BytesIO()
        df_master.to_parquet(buf, engine='pyarrow')
        buf.seek(0)
        bucket.blob(master_blob_path).upload_from_file(buf, content_type='application/octet-stream')
        print(f"   Saved master_inference: {master_blob_path} ({len(df_master):,} rows)")

        # --- Summary ---
        print(f"\n   {ticker} DONE:")
        print(f"      Date range: {df_master.index.min()} -> {df_master.index.max()}")
        print(f"      Total bars: {len(df_master):,}")
        cols_present = [c for c in ['Spread','OBI','Volume_Delta','Trade_Count','Delta_Pct','bsize','asize','Total_Vol']
                        if c in df_master.columns and df_master[c].abs().sum() > 0]
        if cols_present:
            print(f"      Micro cols with data: {cols_present}")
        else:
            print(f"      Micro cols: all zero-filled (no T&Q data)")

    print(f"\n{'='*60}")
    print(f"BOOTSTRAP COMPLETE")
    print(f"{'='*60}")
    print(f"\nREMINDER: Add these tickers to daily_data_all.py TARGET_TICKERS")
    print(f"for ongoing daily updates.")
