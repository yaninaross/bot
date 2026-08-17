"""
daily_micro_all.py — Microstructure Bootstrapper (batch 1 of 2, A-F)
=====================================================================
Downloads T&Q raw data from Polygon and computes microstructure
parquets for 20 deploy tickers (A-F alphabetical split).

Fully incremental and idempotent: already up-to-date tickers are skipped,
gaps are filled from GCS cache or Polygon API.

Batch 1: ACHR, AEHR, ALAB, AMD, APLD, ARM, ASTS, AVAV, BBAI, BE,
         BTBT, CAT, CEG, CIFR, CLSK, COIN, CORZ, CVX, ETN, FLNC
"""

import os
import time
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from polygon import RESTClient
from google.cloud import storage as gcs_storage
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# ⚙️  CONFIGURATION
# ==============================================================================
TARGET_TICKERS = [
    'ACHR', 'AEHR', 'ALAB', 'AMD', 'APLD', 'ARM', 'ASTS', 'AVAV', 'BBAI', 'BE',
    'BTBT', 'CAT', 'CEG', 'CIFR', 'CLSK', 'COIN', 'CORZ', 'CVX', 'ETN', 'FLNC',
]

POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
BUCKET_NAME     = "time-series-bucket-ys"

GCS_MICRO_DIR = f"gs://{BUCKET_NAME}/Data/Inference/Microstructure"
GCS_RAW_DIR   = f"gs://{BUCKET_NAME}/Data/TradesQuotes"

# T&Q data starts from 2025-09-01 (bulk download range)
HISTORY_START = "2025-09-01"

GCS_STATUS_URI = f"gs://{BUCKET_NAME}/Data/Inference/pipeline_status.json"

poly_client = RESTClient(POLYGON_API_KEY)
gcs_client  = gcs_storage.Client()

# ==============================================================================
# 🛠️  GCS HELPERS
# ==============================================================================
def _blob_path(gcs_uri):
    return gcs_uri.replace(f"gs://{BUCKET_NAME}/", "")

def _gcs_exists(gcs_uri):
    return gcs_client.bucket(BUCKET_NAME).blob(_blob_path(gcs_uri)).exists()

def _is_complete_trading_day(gcs_uri, kind):
    """Check if a cached T&Q file covers US market hours (13:30-20:00 UTC)."""
    try:
        df = pd.read_parquet(gcs_uri)
        if df.empty:
            return False
        ts_col = 'timestamp' if 'timestamp' in df.columns else None
        if ts_col is None:
            for c in ('ns', 'sip_timestamp', 'participant_timestamp'):
                if c in df.columns:
                    ts_col = c
                    break
        if ts_col is None:
            return False
        col = df[ts_col]
        if pd.api.types.is_integer_dtype(col):
            col = pd.to_datetime(col, unit='ns', utc=True)
        elif pd.api.types.is_datetime64_any_dtype(col):
            if col.dt.tz is None:
                col = col.dt.tz_localize('UTC')
        else:
            col = pd.to_datetime(col, utc=True)
        last_ts = col.max()
        # Market closes ~20:00 UTC (1:00 PM PT); require last trade >= 20:00
        # to confirm full session coverage (not mid-day truncation)
        return last_ts.hour >= 20
    except Exception:
        return False

def _recent_business_days(n=7):
    """Return set of the last N business day date strings (YYYY-MM-DD)."""
    today = pd.Timestamp.now(tz='UTC').normalize()
    dates = pd.bdate_range(end=today, periods=n + 1, freq='B')
    return {d.strftime('%Y-%m-%d') for d in dates}

import json

def _write_pipeline_status(stage, status):
    """Write pipeline status to GCS so downstream scripts can check."""
    blob = gcs_client.bucket(BUCKET_NAME).blob(_blob_path(GCS_STATUS_URI))
    try:
        existing = json.loads(blob.download_as_text())
    except Exception:
        existing = {}
    existing[stage] = {
        'status': status,
        'timestamp': datetime.now(pytz.utc).isoformat(),
    }
    blob.upload_from_string(json.dumps(existing), content_type='application/json')

def read_pipeline_status():
    """Read pipeline status from GCS. Returns dict or empty dict."""
    blob = gcs_client.bucket(BUCKET_NAME).blob(_blob_path(GCS_STATUS_URI))
    try:
        return json.loads(blob.download_as_text())
    except Exception:
        return {}

# ==============================================================================
# 🛠️  SCHEMA NORMALIZATION
# ==============================================================================
def normalize_raw_schema(df, kind):
    if df is None or df.empty:
        return df
    for ts_col in ('ns', 'sip_timestamp', 'participant_timestamp'):
        if ts_col in df.columns and 'timestamp' not in df.columns:
            df = df.rename(columns={ts_col: 'timestamp'})
            break
    if 'timestamp' not in df.columns:
        return pd.DataFrame()
    col = df['timestamp']
    try:
        if pd.api.types.is_integer_dtype(col):
            df['timestamp'] = pd.to_datetime(col, unit='ns', utc=True)
        elif pd.api.types.is_datetime64_any_dtype(col):
            if col.dt.tz is None:
                df['timestamp'] = col.dt.tz_localize('UTC')
        else:
            df['timestamp'] = pd.to_datetime(col, utc=True)
    except Exception:
        return pd.DataFrame()
    if kind == 'quotes':
        df = df.rename(columns={'bid': 'bid_price', 'ask': 'ask_price',
                                'bsize': 'bid_size', 'asize': 'ask_size'})
    return df

def _valid_raw_file(fname, clean_ticker, kind):
    fname    = os.path.basename(fname)
    expected = f"{clean_ticker}_{kind}_"
    if not fname.startswith(expected):
        print(f"      ⚠️  Skipping misnamed file: {fname}")
        return False
    return True

# ==============================================================================
# 🛠️  POLYGON T&Q FETCH
# ==============================================================================
def fetch_polygon_trades(ticker, start_dt, end_dt):
    trades = []
    for t in poly_client.list_trades(ticker, timestamp_gte=start_dt, timestamp_lte=end_dt, limit=50000):
        trades.append({'timestamp': pd.to_datetime(t.sip_timestamp, unit='ns', utc=True),
                       'price': t.price, 'size': t.size})
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)

def fetch_polygon_quotes(ticker, start_dt, end_dt):
    quotes = []
    for q in poly_client.list_quotes(ticker, timestamp_gte=start_dt, timestamp_lte=end_dt, limit=50000):
        quotes.append({'timestamp': pd.to_datetime(q.sip_timestamp, unit='ns', utc=True),
                       'ask_price': q.ask_price, 'ask_size': q.ask_size,
                       'bid_price': q.bid_price, 'bid_size': q.bid_size})
    if not quotes:
        return pd.DataFrame()
    return pd.DataFrame(quotes)

# ==============================================================================
# 🧮  MICROSTRUCTURE ENGINE
# ==============================================================================
def process_raw_to_microstructure(trades_df, quotes_df):
    if trades_df is not None and not trades_df.empty:
        trades_df = trades_df.set_index('timestamp')
        trades_df['Direction']    = np.sign(trades_df['price'].diff())
        trades_df['Direction']    = trades_df['Direction'].mask(trades_df['Direction'] == 0).ffill().fillna(0)
        trades_df['Volume_Delta'] = trades_df['size'] * trades_df['Direction']
        t_1m = trades_df.resample('1min').agg(
            {'size': 'sum', 'Volume_Delta': 'sum', 'price': 'count'}
        ).rename(columns={'size': 'Total_Vol', 'price': 'Trade_Count'}).fillna(0)
    else:
        t_1m = pd.DataFrame(columns=['Total_Vol', 'Volume_Delta', 'Trade_Count'])

    if quotes_df is not None and not quotes_df.empty:
        quotes_df = quotes_df.set_index('timestamp')
        quotes_df['Spread'] = quotes_df['ask_price'] - quotes_df['bid_price']
        quotes_df['OBI']    = (quotes_df['bid_size'] - quotes_df['ask_size']) / \
                               (quotes_df['bid_size'] + quotes_df['ask_size'] + 1e-9)
        q_1m = quotes_df.resample('1min').agg(
            {'Spread': 'mean', 'OBI': 'mean', 'bid_size': 'sum', 'ask_size': 'sum'}
        ).rename(columns={'bid_size': 'bsize', 'ask_size': 'asize'}).ffill()
    else:
        q_1m = pd.DataFrame(columns=['Spread', 'OBI', 'bsize', 'asize'])

    daily_df = pd.concat([q_1m, t_1m], axis=1).ffill().fillna(0)
    daily_df['Delta_Pct'] = daily_df['Volume_Delta'] / (daily_df['Total_Vol'] + 1e-9)
    return daily_df

# ==============================================================================
# 🚀  MAIN LOOP
# ==============================================================================
if __name__ == "__main__":
    print("🚀 daily_micro_all.py — Microstructure Batch 1 (A-F)")
    run_ts    = datetime.now(pytz.utc)
    fetch_end = run_ts
    print(f"   Run time : {run_ts.strftime('%Y-%m-%d %H:%M UTC')}")
    _write_pipeline_status('daily_micro_1', 'running')

    # Deduplicate to be safe
    ALL_TICKERS = list(set(TARGET_TICKERS))
    print(f"   Total Assets: {len(ALL_TICKERS)}")

    for ticker in ALL_TICKERS:
        print(f"\n{'─'*52}")
        print(f"🔄 {ticker}")
        clean_ticker = ticker.replace("I:", "").replace("X:", "")
        micro_uri    = f"{GCS_MICRO_DIR}/{clean_ticker}_1m_microstructure.parquet"

        # ── Load existing micro to find the gap start ────────────────────────────
        try:
            df_micro    = pd.read_parquet(micro_uri)
            if df_micro.index.tz is None:
                df_micro.index = df_micro.index.tz_localize('UTC')
            micro_start = df_micro.index.max() + timedelta(minutes=1)
            print(f"   Existing micro: {len(df_micro):,} rows, last row: {df_micro.index.max()}")
        except Exception:
            df_micro    = pd.DataFrame()
            micro_start = pd.Timestamp(HISTORY_START, tz='UTC')
            print(f"   No existing micro — starting from {HISTORY_START}")

        if micro_start >= fetch_end:
            print("   ✅ Already up to date — nothing to fetch")
            continue

        # ── Gap-aware T&Q fill: process DAY-BY-DAY to avoid OOM ────────────────
        # Each day's raw ticks (millions for NVDA) are reduced to ~390 micro bars,
        # then discarded before loading the next day.
        print(f"   ⏳ Gap: {micro_start.date()} → {fetch_end.date()}...")
        gap_dates       = pd.date_range(micro_start.date(), fetch_end.date(), freq='B')
        micro_chunks    = []
        days_disk, days_api, days_empty = 0, 0, 0

        recent_dates = _recent_business_days(2)

        for gap_date in gap_dates:
            date_str = gap_date.strftime('%Y-%m-%d')
            t_fname  = f"{clean_ticker}_trades_{date_str}.parquet"
            q_fname  = f"{clean_ticker}_quotes_{date_str}.parquet"
            t_uri    = f"{GCS_RAW_DIR}/{clean_ticker}/{t_fname}"
            q_uri    = f"{GCS_RAW_DIR}/{clean_ticker}/{q_fname}"

            dt, dq = pd.DataFrame(), pd.DataFrame()

            cache_exists = _gcs_exists(t_uri) and _gcs_exists(q_uri)
            use_cache = False

            if cache_exists:
                if date_str in recent_dates:
                    # Recent date — validate that cache covers market hours
                    if _is_complete_trading_day(t_uri, 'trades'):
                        use_cache = True
                    else:
                        print(f"      🔄 {date_str}: stale cache (pre-market only) — re-fetching")
                else:
                    # Older date — trust the cache
                    use_cache = True

            if use_cache:
                try:
                    if _valid_raw_file(t_fname, clean_ticker, 'trades') and \
                       _valid_raw_file(q_fname, clean_ticker, 'quotes'):
                        dt = normalize_raw_schema(pd.read_parquet(t_uri), 'trades')
                        dq = normalize_raw_schema(pd.read_parquet(q_uri), 'quotes')
                        days_disk += 1
                    else:
                        raise ValueError("misnamed file")
                except Exception as e:
                    print(f"      ⚠️  {date_str}: disk read failed — {e}")
                    continue
            else:
                # Not in GCS or stale cache — call Polygon and upload raw files
                day_start = pd.Timestamp(gap_date, tz='UTC')
                day_end   = min(day_start + timedelta(days=1), fetch_end)
                try:
                    dt = fetch_polygon_trades(clean_ticker, day_start, day_end)
                    dq = fetch_polygon_quotes(clean_ticker, day_start, day_end)
                    if not dt.empty:
                        dt.to_parquet(t_uri, engine='pyarrow')
                    if not dq.empty:
                        dq.to_parquet(q_uri, engine='pyarrow')
                    days_api += 1
                    if dt.empty and dq.empty and date_str in recent_dates:
                        print(f"      ⚠️  {date_str}: Polygon returned empty — data may not be available yet")
                except Exception as e:
                    print(f"      ⚠️  {date_str}: API error — {e}")
                    continue

            if dt.empty and dq.empty:
                days_empty += 1
                continue

            # Trim to micro_start to avoid double-counting
            if not dt.empty:
                dt = dt[dt['timestamp'] >= micro_start]
            if not dq.empty:
                dq = dq[dq['timestamp'] >= micro_start]

            if dt.empty and dq.empty:
                continue

            # Process this single day → ~390 micro bars, then free raw ticks
            day_micro = process_raw_to_microstructure(dt, dq)
            del dt, dq  # free raw ticks immediately
            if not day_micro.empty:
                micro_chunks.append(day_micro)

        print(f"      📥 {days_disk} days from disk / {days_api} days from API / {days_empty} empty")

        if not micro_chunks:
            print("      ℹ️  No T&Q data in gap — micro unchanged")
            continue

        new_micro = pd.concat(micro_chunks).sort_index()
        del micro_chunks  # free chunk list

        # Deduplicate against existing micro before appending
        if not df_micro.empty:
            new_micro = new_micro[new_micro.index > df_micro.index.max()]

        if new_micro.empty:
            print("      ⏭️  All new micro rows already covered — skipping")
            continue

        df_micro = pd.concat([df_micro, new_micro]).sort_index()
        df_micro = df_micro[~df_micro.index.duplicated(keep='last')]
        df_micro.to_parquet(micro_uri, engine='pyarrow')
        print(f"      ✅ Appended {len(new_micro):,} rows  |  total: {len(df_micro):,} rows")

    _write_pipeline_status('daily_micro_1', 'complete')
    print("\n🎉 daily_micro_all.py COMPLETE (batch 1).")
