"""
bootstrap_defense_analysis.py — Check & generate master_inference for defense analysis
======================================================================================
Checks every ticker needed by analyze_defense_cluster.py.  If the
master_inference.parquet already exists in GCS it is skipped.  Otherwise the
script:

  1. Loads or fetches OHLCV (1-min bars from Polygon, 2022-01-01 → now)
  2. Saves the raw OHLCV parquet to GCS
  3. Joins with microstructure parquet if one exists in GCS
  4. Heals (resample 1-min, ffill prices, zero-fill volume — NO backfill)
  5. Uploads the resulting master_inference.parquet to GCS

This matches exactly how data was built for model training (exp_103.py,
daily_data_all.py, bootstrap_missing_tickers.py).

Usage:  python bootstrap_defense_analysis.py
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

# --- All 39 candidate tickers from analyze_defense_cluster.py ---
ALL_CANDIDATES = [
    # Current defense teachers
    'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS',
    'LUNR', 'SPIR', 'JOBY', 'NNDM', 'MVIS',
    # Current defense scoring extras
    'RCAT', 'ACHR',
    # Crypto cluster
    'APLD', 'BTBT', 'CIFR', 'CLSK', 'COIN', 'CORZ',
    'HOOD', 'HUT', 'IREN', 'MARA', 'MIGI', 'MSTR',
    'RIOT', 'WULF',
    # New candidates
    'ALAB', 'AMD', 'BE', 'NBIS', 'NVDA', 'PLUG',
    'SMCI', 'SOUN', 'TSLA', 'VRT',
    # Large-cap defense
    'KEEL',
]

# Context assets needed by the analysis script
CONTEXT_ASSETS = ['ITA', 'VIX', 'TLT', 'IWM']

# Context assets use Polygon ticker format for API calls
CONTEXT_POLYGON_MAP = {
    'VIX': 'I:VIX',
    'ITA': 'ITA',
    'TLT': 'TLT',
    'IWM': 'IWM',
}

TICKER_FILE_MAP = {'BITF': 'KEEL'}

POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
BUCKET_NAME     = "time-series-bucket-ys"
INFERENCE_DIR   = "Data/Inference"
MICRO_DIR       = "Data/Inference/Microstructure"

HISTORY_START = "2022-01-01"

poly_client = RESTClient(POLYGON_API_KEY)
gcs_client  = gcs_storage.Client()
bucket      = gcs_client.bucket(BUCKET_NAME)

# ==============================================================================
# POLYGON OHLCV FETCH (identical to daily_data_all.py / bootstrap_missing_tickers.py)
# ==============================================================================
def fetch_polygon_aggs(ticker, start_dt, end_dt, chunk_months=1, sleep_secs=2):
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
# HEAL (identical to exp_103.py / daily_data_all.py — resample + ffill + zero-fill, NO backfill)
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# GCS HELPERS
# ==============================================================================
def gcs_exists(blob_path):
    return bucket.blob(blob_path).exists()

def upload_df(df, blob_path):
    buf = io.BytesIO()
    df.to_parquet(buf, engine='pyarrow')
    buf.seek(0)
    bucket.blob(blob_path).upload_from_file(buf, content_type='application/octet-stream')

def download_df(blob_path):
    blob = bucket.blob(blob_path)
    return pd.read_parquet(io.BytesIO(blob.download_as_bytes()))


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  BOOTSTRAP — master_inference for defense cluster analysis")
    print("=" * 70)

    fetch_end = pd.Timestamp.now(tz='UTC')
    print(f"  History : {HISTORY_START} -> {fetch_end.strftime('%Y-%m-%d %H:%M UTC')}")

    # Build full ticker list: candidates + context (deduplicated)
    all_tickers = []
    seen = set()
    for t in ALL_CANDIDATES + CONTEXT_ASSETS:
        if t not in seen:
            all_tickers.append(t)
            seen.add(t)

    print(f"  Tickers : {len(all_tickers)} ({len(ALL_CANDIDATES)} candidates + {len(CONTEXT_ASSETS)} context)")

    # ── Phase 1: Check which master_inference files exist ──────────────────
    print(f"\n{'─'*70}")
    print("  Phase 1: Checking existing master_inference files in GCS...")
    print(f"{'─'*70}")

    exists_list = []
    missing_list = []

    for ticker in all_tickers:
        clean = ticker.replace("I:", "").replace("X:", "")
        file_ticker = TICKER_FILE_MAP.get(clean, clean)
        master_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"

        if gcs_exists(master_path):
            # Check freshness: load just enough to see last timestamp
            try:
                df_check = download_df(master_path)
                if df_check.index.tz is None:
                    df_check.index = df_check.index.tz_localize('UTC')
                last_ts = df_check.index.max()
                rows = len(df_check)
                del df_check
                print(f"  OK   {ticker:8s}  {rows:>10,} rows  last: {last_ts}")
                exists_list.append(ticker)
            except Exception as e:
                print(f"  ERR  {ticker:8s}  exists but can't read: {e}")
                missing_list.append(ticker)
        else:
            print(f"  MISS {ticker:8s}  — will generate")
            missing_list.append(ticker)

    print(f"\n  Summary: {len(exists_list)} exist, {len(missing_list)} missing")

    if not missing_list:
        print("\n  All master_inference files present — nothing to do.")
        print("=" * 70)
        raise SystemExit(0)

    # ── Phase 2: Generate missing master_inference files ───────────────────
    print(f"\n{'─'*70}")
    print(f"  Phase 2: Generating {len(missing_list)} missing master_inference files...")
    print(f"{'─'*70}")

    success = []
    failed  = []

    for i, ticker in enumerate(missing_list, 1):
        print(f"\n[{i}/{len(missing_list)}] {ticker}")
        print(f"{'─'*50}")

        clean = ticker.replace("I:", "").replace("X:", "")
        file_ticker = TICKER_FILE_MAP.get(clean, clean)

        ohlcv_path  = f"{INFERENCE_DIR}/{file_ticker}_1m_full_historical_data.parquet"
        micro_path  = f"{MICRO_DIR}/{file_ticker}_1m_microstructure.parquet"
        master_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"

        # Determine Polygon ticker for API calls
        polygon_ticker = CONTEXT_POLYGON_MAP.get(ticker, ticker)

        # --- A. Load or fetch OHLCV ---
        if gcs_exists(ohlcv_path):
            print(f"   Found existing OHLCV — loading...")
            df_ohlcv = download_df(ohlcv_path)
            if df_ohlcv.index.tz is None:
                df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
            ohlcv_start = df_ohlcv.index.max() + timedelta(minutes=1)
            print(f"   Existing: {len(df_ohlcv):,} rows, last: {df_ohlcv.index.max()}")
        else:
            print(f"   No existing OHLCV — fetching full history from {HISTORY_START}")
            df_ohlcv = pd.DataFrame()
            ohlcv_start = pd.Timestamp(HISTORY_START, tz='UTC')

        # Fetch any new data from Polygon
        if ohlcv_start < fetch_end:
            print(f"   Fetching OHLCV from {ohlcv_start.date()} via Polygon ({polygon_ticker})...")
            new_aggs = fetch_polygon_aggs(polygon_ticker, ohlcv_start, fetch_end)
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
            failed.append(ticker)
            continue

        # --- B. Save raw OHLCV ---
        upload_df(df_ohlcv, ohlcv_path)
        print(f"   Saved OHLCV: {ohlcv_path} ({len(df_ohlcv):,} rows)")

        # --- C. Join with micro if it exists ---
        is_context = ticker in CONTEXT_ASSETS
        if not is_context and gcs_exists(micro_path):
            print(f"   Found micro data — joining...")
            df_micro = download_df(micro_path)
            if df_micro.index.tz is None:
                df_micro.index = df_micro.index.tz_localize('UTC')
            df_master = df_ohlcv.join(df_micro, how='left')
            print(f"   Micro: {len(df_micro):,} rows, last: {df_micro.index.max()}")
            del df_micro
        else:
            if is_context:
                print(f"   Context asset — OHLCV only (no micro needed)")
            else:
                print(f"   No micro data found — micro cols will be zero-filled")
            df_master = df_ohlcv.copy()

        del df_ohlcv

        # --- D. Heal (resample + ffill + zero-fill, NO backfill) ---
        print(f"   Healing data...")
        df_master = heal_data(df_master)
        df_master = df_master[df_master.index.notna()].sort_index()

        # --- E. Save master_inference ---
        upload_df(df_master, master_path)
        print(f"   Saved master_inference: {master_path} ({len(df_master):,} rows)")

        # --- Summary ---
        print(f"   {ticker} DONE:")
        print(f"      Date range : {df_master.index.min()} -> {df_master.index.max()}")
        print(f"      Total bars : {len(df_master):,}")
        micro_present = [c for c in ['Spread','OBI','Volume_Delta','Trade_Count',
                                      'Delta_Pct','bsize','asize','Total_Vol']
                         if c in df_master.columns and df_master[c].abs().sum() > 0]
        if micro_present:
            print(f"      Micro cols : {micro_present}")
        else:
            print(f"      Micro cols : all zero-filled (no T&Q data)")

        del df_master
        success.append(ticker)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BOOTSTRAP COMPLETE")
    print(f"  Generated : {len(success)}/{len(missing_list)}")
    if success:
        print(f"  Success   : {', '.join(success)}")
    if failed:
        print(f"  Failed    : {', '.join(failed)}")
    print(f"{'='*70}")

    if success:
        print(f"\n  NOTE: For ongoing daily updates, add these tickers to")
        print(f"  daily_data_all.py TARGET_TICKERS (and daily_micro_all.py")
        print(f"  if you want T&Q micro features).")
