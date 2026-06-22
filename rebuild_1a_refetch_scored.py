"""
rebuild_1a_refetch_scored.py — Refetch T&Q + Rebuild Micro (22 scored tickers)
===============================================================================
  1. AUDIT   — find missing business days in T&Q cache for scored tickers
  2. REFETCH — download missing dates from Polygon API
  3. REBUILD MICRO — rebuild microstructure parquets from fresh T&Q data

Run in parallel with rebuild_1b_refetch_unscored.py.
Run AFTER market close (after 20:00 UTC / 1:00 PM Pacific).
After both complete, run rebuild_2_delete_downstream.py.
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
# CONFIGURATION
# ==============================================================================
DRY_RUN = False

POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
BUCKET_NAME     = "time-series-bucket-ys"
GCS_MICRO_DIR   = f"gs://{BUCKET_NAME}/Data/Inference/Microstructure"
GCS_RAW_DIR     = f"gs://{BUCKET_NAME}/Data/TradesQuotes"

SCORED_TICKERS = [
    'ACHR', 'ALAB', 'APLD', 'ASTS', 'BE',
    'BTBT', 'CIFR', 'CLSK', 'FCEL', 'IREN',
    'JOBY', 'KEEL', 'LUNR', 'OKLO', 'PL',
    'PLUG', 'RCAT', 'RDW', 'RIOT', 'RKLB',
    'SMCI', 'WULF',
]

CORRUPTION_START = "2026-03-01"

poly_client = RESTClient(POLYGON_API_KEY)
gcs_client  = gcs_storage.Client()
bucket      = gcs_client.bucket(BUCKET_NAME)

# ==============================================================================
# T&Q FETCH + MICROSTRUCTURE ENGINE
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
# INIT
# ==============================================================================
print(f"🚀 rebuild_1a — Scored tickers ({len(SCORED_TICKERS)} tickers)")

run_ts = datetime.now(pytz.utc)
if run_ts.hour < 20 and run_ts.weekday() < 5:
    print(f"\n⚠️  WARNING: Current time is {run_ts.strftime('%H:%M UTC')}.")
    print(f"   Market may still be open. Run after 20:00 UTC for complete data.\n")

# ==============================================================================
# STEP 1: AUDIT — Find missing business days
# ==============================================================================
print("=" * 60)
print("STEP 1: AUDIT — Finding missing T&Q dates since", CORRUPTION_START)
print("=" * 60)

today = pd.Timestamp.now(tz='UTC').normalize().tz_localize(None)
expected_dates = set(
    d.strftime('%Y-%m-%d')
    for d in pd.bdate_range(CORRUPTION_START, today, freq='B')
)

missing_by_ticker = {}
total_missing = 0

for ticker in SCORED_TICKERS:
    prefix = f"Data/TradesQuotes/{ticker}/{ticker}_trades_"
    blobs = list(bucket.list_blobs(prefix=prefix))
    existing = set()
    for b in blobs:
        date_part = b.name.split('_')[-1].replace('.parquet', '').replace('.csv', '')
        if date_part >= CORRUPTION_START:
            existing.add(date_part)

    missing = sorted(expected_dates - existing)
    if missing:
        missing_by_ticker[ticker] = missing
        total_missing += len(missing)

print(f"\nTotal missing date-ticker combos: {total_missing}")
print(f"Tickers with gaps: {len(missing_by_ticker)} / {len(SCORED_TICKERS)}")
print()

for ticker in sorted(missing_by_ticker.keys()):
    dates = missing_by_ticker[ticker]
    print(f"  {ticker:>6}: {len(dates)} missing  ({dates[0]} → {dates[-1]})")

if DRY_RUN:
    print("\n  DRY_RUN = True — set to False and re-run to refetch.")

# ==============================================================================
# STEP 2: REFETCH — Download missing dates from Polygon
# ==============================================================================
if not DRY_RUN:
    print(f"\n{'=' * 60}")
    print("STEP 2: REFETCH — Downloading missing T&Q from Polygon")
    print("=" * 60)

    fetched = 0
    api_errors = 0

    for ticker in sorted(missing_by_ticker.keys()):
        missing_dates = missing_by_ticker[ticker]
        clean_ticker = ticker.replace("I:", "").replace("X:", "")
        print(f"\n  {ticker}: {len(missing_dates)} dates to fetch")

        for date_str in missing_dates:
            day_start = pd.Timestamp(date_str, tz='UTC')
            day_end   = day_start + timedelta(days=1)

            t_uri = f"{GCS_RAW_DIR}/{clean_ticker}/{clean_ticker}_trades_{date_str}.parquet"
            q_uri = f"{GCS_RAW_DIR}/{clean_ticker}/{clean_ticker}_quotes_{date_str}.parquet"

            try:
                dt = fetch_polygon_trades(clean_ticker, day_start, day_end)
                dq = fetch_polygon_quotes(clean_ticker, day_start, day_end)

                if not dt.empty:
                    dt.to_parquet(t_uri, engine='pyarrow')
                if not dq.empty:
                    dq.to_parquet(q_uri, engine='pyarrow')

                if dt.empty and dq.empty:
                    pd.DataFrame({'timestamp': [], 'price': [], 'size': []}).to_parquet(t_uri, engine='pyarrow')
                    pd.DataFrame({'timestamp': [], 'ask_price': [], 'bid_price': []}).to_parquet(q_uri, engine='pyarrow')

                fetched += 1
                if fetched % 50 == 0:
                    print(f"    ... {fetched} dates fetched so far")

            except Exception as e:
                print(f"    ❌ {ticker} {date_str}: {e}")
                api_errors += 1

            time.sleep(0.5)

    print(f"\n  Fetched {fetched} dates, {api_errors} errors")

# ==============================================================================
# STEP 3: REBUILD MICRO — rebuild microstructure from fresh T&Q
# ==============================================================================
if not DRY_RUN:
    print(f"\n{'=' * 60}")
    print("STEP 3: REBUILD MICRO — rebuilding microstructure parquets")
    print("=" * 60)

    for ticker in SCORED_TICKERS:
        print(f"\n{'─' * 52}")
        print(f"  🔄 {ticker}")
        clean_ticker = ticker.replace("I:", "").replace("X:", "")
        micro_uri = f"{GCS_MICRO_DIR}/{clean_ticker}_1m_microstructure.parquet"

        try:
            df_micro = pd.read_parquet(micro_uri)
            if df_micro.index.tz is None:
                df_micro.index = df_micro.index.tz_localize('UTC')
            print(f"     Existing micro: {len(df_micro):,} rows, last: {df_micro.index.max()}")
        except Exception:
            df_micro = pd.DataFrame()
            print(f"     No existing micro")

        rebuild_start = pd.Timestamp(CORRUPTION_START, tz='UTC')
        all_dates = sorted(
            d.strftime('%Y-%m-%d')
            for d in pd.bdate_range(CORRUPTION_START, today, freq='B')
        )

        micro_chunks = []
        for date_str in all_dates:
            t_uri = f"{GCS_RAW_DIR}/{clean_ticker}/{clean_ticker}_trades_{date_str}.parquet"
            q_uri = f"{GCS_RAW_DIR}/{clean_ticker}/{clean_ticker}_quotes_{date_str}.parquet"

            try:
                t_blob = bucket.blob(t_uri.replace(f"gs://{BUCKET_NAME}/", ""))
                q_blob = bucket.blob(q_uri.replace(f"gs://{BUCKET_NAME}/", ""))
                if not t_blob.exists() or not q_blob.exists():
                    continue

                dt = normalize_raw_schema(pd.read_parquet(t_uri), 'trades')
                dq = normalize_raw_schema(pd.read_parquet(q_uri), 'quotes')

                if (dt is None or dt.empty) and (dq is None or dq.empty):
                    continue

                day_micro = process_raw_to_microstructure(dt, dq)
                del dt, dq
                if not day_micro.empty:
                    micro_chunks.append(day_micro)
            except Exception as e:
                print(f"     ⚠️  {date_str}: {e}")

        if not micro_chunks:
            print(f"     ℹ️  No T&Q data for rebuild window")
            continue

        new_micro = pd.concat(micro_chunks).sort_index()

        if not df_micro.empty:
            df_keep = df_micro[df_micro.index < rebuild_start]
            df_micro = pd.concat([df_keep, new_micro]).sort_index()
        else:
            df_micro = new_micro

        df_micro = df_micro[~df_micro.index.duplicated(keep='last')]
        df_micro.to_parquet(micro_uri, engine='pyarrow')
        print(f"     ✅ Micro rebuilt: {len(df_micro):,} rows (replaced from {CORRUPTION_START})")

    print(f"\n{'=' * 60}")
    print("NOTEBOOK 1A COMPLETE — scored tickers refetched + micro rebuilt")
    print("After 1B finishes, run rebuild_2_delete_downstream.py")
    print("=" * 60)
