"""
daily_data_100.py — OHLCV Refresh + Master Inference Stitch for Exp-100
=======================================================================
Fetches 1-minute OHLCV bars from Polygon for ALL assets needed by Exp-100
(Defense Sector Model), then stitches each master_inference.parquet by
joining OHLCV with the micro parquet already written by daily_micro_100.py.

Covers:
  Teachers/Scoring: AVAV, KTOS, PLTR, BBAI, RKLB, ASTS, LUNR, SPIR, JOBY, NNDM, MVIS,
                    GD, LMT, NOC, RTX, LHX, GE, RCAT, ACHR
  Context:          ITA, VIX, TLT, IWM
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
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# ⚙️  CONFIGURATION
# ==============================================================================
TARGET_TICKERS = [
    # Defense Tech / UAVs
    'AVAV', 'KTOS', 'PLTR', 'BBAI',
    # Space & Dual-Use
    'RKLB', 'ASTS', 'LUNR', 'SPIR',
    # New Speculative Teachers
    'JOBY', 'NNDM', 'MVIS',
    # Target Primes & Secondary Scorers
    'GD', 'LMT', 'NOC', 'RTX', 'LHX', 'GE', 'RCAT', 'ACHR'
]

# OHLCV-only context assets (no T&Q micro needed)
CONTEXT_ASSETS = ['ITA', 'VIX', 'TLT', 'IWM']

POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
BUCKET_NAME     = "time-series-bucket-ys"

GCS_OHLCV_DIR = f"gs://{BUCKET_NAME}/Data/Inference"
GCS_MICRO_DIR = f"gs://{BUCKET_NAME}/Data/Inference/Microstructure"

poly_client = RESTClient(POLYGON_API_KEY)
gcs_client  = gcs_storage.Client()

# ==============================================================================
# 🛠️  POLYGON OHLCV FETCH
# ==============================================================================
def fetch_polygon_aggs(ticker, start_dt, end_dt, chunk_months=1, sleep_secs=2):
    """Fetch 1-minute OHLCV bars from Polygon in monthly chunks."""
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
# 🛠️  HEAL (forward-fill OHLC, zero-fill volume cols)
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df

# ==============================================================================
# 🚀  MAIN LOOP
# ==============================================================================
if __name__ == "__main__":
    print("🚀 daily_data_100.py — Exp-100 OHLCV Refresh + Master Inference Stitch")
    fetch_end = pd.Timestamp.now(tz='UTC')
    print(f"   Run time      : {fetch_end.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"   Teachers      : {TARGET_TICKERS}")
    print(f"   Context assets: {CONTEXT_ASSETS}")

    ALL_ASSETS = [(t, False) for t in TARGET_TICKERS] + [(t, True) for t in CONTEXT_ASSETS]

    for ticker, is_context in ALL_ASSETS:
        print(f"\n{'─'*52}")
        print(f"🔄 {ticker}{'  [context — OHLCV only]' if is_context else ''}")
        clean_ticker = ticker.replace("I:", "").replace("X:", "")

        ohlcv_uri  = f"{GCS_OHLCV_DIR}/{clean_ticker}_1m_full_historical_data.parquet"
        micro_uri  = f"{GCS_MICRO_DIR}/{clean_ticker}_1m_microstructure.parquet"
        master_uri = f"{GCS_OHLCV_DIR}/{clean_ticker}_master_inference.parquet"

        # ── A. Load existing OHLCV ───────────────────────────────────────────────
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

        # ── B. Fetch new OHLCV ──────────────────────────────────────────────────
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

        # ── C. Load micro (teachers only) ────────────────────────────────────────
        df_micro = pd.DataFrame()
        if not is_context:
            try:
                df_micro = pd.read_parquet(micro_uri)
                if df_micro.index.tz is None:
                    df_micro.index = df_micro.index.tz_localize('UTC')
                print(f"   📂 Micro loaded: {len(df_micro):,} rows, last: {df_micro.index.max()}")
            except Exception:
                print(f"   ⚠️  No micro found for {ticker} — run daily_micro_100.py first")

        # ── D. Stitch master_inference.parquet ───────────────────────────────────
        if not is_context and not df_micro.empty:
            df_unified = df_ohlcv.join(df_micro, how='left')
        else:
            df_unified = df_ohlcv.copy()

        df_unified = heal_data(df_unified)
        df_unified = df_unified[df_unified.index.notna()].sort_index()
        df_unified.to_parquet(master_uri, engine='pyarrow')
        print(f"   ✅ master_inference saved  ({len(df_unified):,} rows)")

    print("\n🎉 daily_data_100.py COMPLETE — exp-100 master_inference files ready.")
