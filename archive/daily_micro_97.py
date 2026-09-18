"""
daily_micro_97.py — Microstructure Bootstrapper for Exp-97
===========================================================
Downloads T&Q raw data from Polygon and computes microstructure
parquets for ALL assets needed by exp_97:

  Original (exp-95): MARA, IREN, WULF, APLD, RIOT, CLSK, CORZ, MSTR, COIN
  New (exp-97):      BITF, HUT, CIFR, BTBT, MIGI, HOOD, SMLR

This is a self-contained bootstrap — safe to run independently of
the prod daily_micro.py. Fully incremental and idempotent: already
up-to-date tickers are skipped, gaps are filled from GCS cache or
Polygon API.

Context assets (SMH, MOD, VIX, etc.) have no T&Q data — they are
handled by daily_data_97.py (OHLCV only).
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
# All TRAIN_TEACHERS for exp_97 — original 9 (exp-95) + 7 new crypto-cluster assets
TARGET_TICKERS = [
    # Original exp-95 teachers
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR', 'COIN',
    # New exp-97 teachers
    'BITF', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'SMLR',
]

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
BUCKET_NAME     = os.environ.get("GCS_BUCKET", "")

GCS_MICRO_DIR = f"gs://{BUCKET_NAME}/Data/Inference/Microstructure"
GCS_RAW_DIR   = f"gs://{BUCKET_NAME}/Data/TradesQuotes"

# Full history start — we want as much data as possible for training
HISTORY_START = "2022-01-01"

poly_client = RESTClient(POLYGON_API_KEY)
gcs_client  = gcs_storage.Client()

# ==============================================================================
# 🛠️  GCS HELPERS
# ==============================================================================
def _blob_path(gcs_uri):
    return gcs_uri.replace(f"gs://{BUCKET_NAME}/", "")

def _gcs_exists(gcs_uri):
    return gcs_client.bucket(BUCKET_NAME).blob(_blob_path(gcs_uri)).exists()

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
print("🚀 daily_micro_97.py — Exp-97 Microstructure Bootstrapper (all 16 teachers)")
run_ts    = datetime.now(pytz.utc)
fetch_end = run_ts
print(f"   Run time : {run_ts.strftime('%Y-%m-%d %H:%M UTC')}")
print(f"   Tickers  : {TARGET_TICKERS}")

for ticker in TARGET_TICKERS:
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

    # ── Gap-aware T&Q fill: per business day ────────────────────────────────
    print(f"   ⏳ Gap: {micro_start.date()} → {fetch_end.date()}...")
    gap_dates               = pd.date_range(micro_start.date(), fetch_end.date(), freq='B')
    all_trades, all_quotes  = [], []
    days_disk, days_api     = 0, 0

    for gap_date in gap_dates:
        date_str = gap_date.strftime('%Y-%m-%d')
        t_fname  = f"{clean_ticker}_trades_{date_str}.parquet"
        q_fname  = f"{clean_ticker}_quotes_{date_str}.parquet"
        t_uri    = f"{GCS_RAW_DIR}/{clean_ticker}/{t_fname}"
        q_uri    = f"{GCS_RAW_DIR}/{clean_ticker}/{q_fname}"

        if _gcs_exists(t_uri) and _gcs_exists(q_uri):
            try:
                if _valid_raw_file(t_fname, clean_ticker, 'trades') and \
                   _valid_raw_file(q_fname, clean_ticker, 'quotes'):
                    dt = normalize_raw_schema(pd.read_parquet(t_uri), 'trades')
                    dq = normalize_raw_schema(pd.read_parquet(q_uri), 'quotes')
                    if not dt.empty: all_trades.append(dt)
                    if not dq.empty: all_quotes.append(dq)
                    days_disk += 1
                else:
                    raise ValueError("misnamed file")
            except Exception as e:
                print(f"      ⚠️  {date_str}: disk read failed — {e}")
        else:
            # Not in GCS — call Polygon and upload raw files for future runs
            day_start = pd.Timestamp(gap_date, tz='UTC')
            day_end   = min(day_start + timedelta(days=1), fetch_end)
            try:
                dt = fetch_polygon_trades(clean_ticker, day_start, day_end)
                dq = fetch_polygon_quotes(clean_ticker, day_start, day_end)
                if not dt.empty:
                    dt.to_parquet(t_uri, engine='pyarrow')
                    all_trades.append(dt)
                if not dq.empty:
                    dq.to_parquet(q_uri, engine='pyarrow')
                    all_quotes.append(dq)
                days_api += 1
            except Exception as e:
                print(f"      ⚠️  {date_str}: API error — {e}")

    print(f"      📥 {days_disk} days from disk / {days_api} days from API")

    if not all_trades and not all_quotes:
        print("      ℹ️  No T&Q data in gap — micro unchanged")
        continue

    raw_trades = pd.concat(all_trades).sort_values('timestamp') if all_trades else pd.DataFrame()
    raw_quotes = pd.concat(all_quotes).sort_values('timestamp') if all_quotes else pd.DataFrame()

    # Trim to micro_start to avoid double-counting already-saved rows
    if not raw_trades.empty:
        raw_trades = raw_trades[raw_trades['timestamp'] >= micro_start]
    if not raw_quotes.empty:
        raw_quotes = raw_quotes[raw_quotes['timestamp'] >= micro_start]

    if raw_trades.empty and raw_quotes.empty:
        print("      ⏭️  Gap trimmed to zero — micro already up to date")
        continue

    print("      🧮 Calculating microstructure...")
    new_micro = process_raw_to_microstructure(raw_trades, raw_quotes)

    if new_micro.empty:
        print("      ⚠️  Micro computation returned empty — skipping")
        continue

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

print("\n🎉 daily_micro_97.py COMPLETE.")
print("   Next step: run daily_data_97.py to stitch master_inference files.")
