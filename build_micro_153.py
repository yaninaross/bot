"""
build_micro_153.py — Build microstructure + master_inference for EXP-153
========================================================================
One-shot script to build missing micro and master_inference parquets
for the 6 tickers that need them before EXP-153 training can run.

Need micro + master: FCEL, FLNC, MP, POWL, SMR
Need master only:    ALAB (micro exists, no master)

Run in Colab or any environment with GCS access.
"""

import os
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import storage as gcs_storage
import warnings

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

BUCKET_NAME     = "time-series-bucket-ys"
GCS_MICRO_DIR   = f"gs://{BUCKET_NAME}/Data/Inference/Microstructure"
GCS_RAW_DIR     = f"gs://{BUCKET_NAME}/Data/TradesQuotes"
GCS_OHLCV_DIR   = f"gs://{BUCKET_NAME}/Data/Inference"

HISTORY_START = "2025-09-01"

NEED_MICRO = ['FCEL', 'FLNC', 'MP', 'POWL', 'SMR']
NEED_MASTER_ONLY = ['ALAB']
ALL_TICKERS = NEED_MICRO + NEED_MASTER_ONLY

gcs_client  = gcs_storage.Client()

def _blob_path(gcs_uri):
    return gcs_uri.replace(f"gs://{BUCKET_NAME}/", "")

def _gcs_exists(gcs_uri):
    return gcs_client.bucket(BUCKET_NAME).blob(_blob_path(gcs_uri)).exists()

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
print("build_micro_153.py — Building micro + master for EXP-153 new tickers")
run_ts    = datetime.now(pytz.utc)
fetch_end = run_ts
print(f"   Run time: {run_ts.strftime('%Y-%m-%d %H:%M UTC')}")

for ticker in ALL_TICKERS:
    print(f"\n{'─'*52}")
    print(f"  {ticker}")
    micro_uri = f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet"

    # ── STEP 1: Build or load micro ──────────────────────────────────
    if ticker in NEED_MICRO:
        print(f"   Building microstructure from T&Q...")
        try:
            df_micro = pd.read_parquet(micro_uri)
            if df_micro.index.tz is None:
                df_micro.index = df_micro.index.tz_localize('UTC')
            micro_start = df_micro.index.max() + timedelta(minutes=1)
            print(f"   Existing micro: {len(df_micro):,} rows, last: {df_micro.index.max()}")
        except Exception:
            df_micro = pd.DataFrame()
            micro_start = pd.Timestamp(HISTORY_START, tz='UTC')
            print(f"   No existing micro — starting from {HISTORY_START}")

        if micro_start >= fetch_end:
            print("   Micro already up to date")
        else:
            gap_dates = pd.date_range(micro_start.date(), fetch_end.date(), freq='B')
            micro_chunks = []
            days_disk, days_empty = 0, 0

            for gap_date in gap_dates:
                date_str = gap_date.strftime('%Y-%m-%d')
                t_uri = f"{GCS_RAW_DIR}/{ticker}/{ticker}_trades_{date_str}.parquet"
                q_uri = f"{GCS_RAW_DIR}/{ticker}/{ticker}_quotes_{date_str}.parquet"
                dt, dq = pd.DataFrame(), pd.DataFrame()

                if _gcs_exists(t_uri) and _gcs_exists(q_uri):
                    try:
                        dt = normalize_raw_schema(pd.read_parquet(t_uri), 'trades')
                        dq = normalize_raw_schema(pd.read_parquet(q_uri), 'quotes')
                        days_disk += 1
                    except Exception as e:
                        print(f"      {date_str}: read failed — {e}")
                        continue
                else:
                    days_empty += 1
                    continue

                if dt.empty and dq.empty:
                    days_empty += 1
                    continue

                if not dt.empty:
                    dt = dt[dt['timestamp'] >= micro_start]
                if not dq.empty:
                    dq = dq[dq['timestamp'] >= micro_start]
                if dt.empty and dq.empty:
                    continue

                day_micro = process_raw_to_microstructure(dt, dq)
                del dt, dq
                if not day_micro.empty:
                    micro_chunks.append(day_micro)

            print(f"      {days_disk} from GCS / {days_empty} empty or missing")

            if micro_chunks:
                new_micro = pd.concat(micro_chunks).sort_index()
                del micro_chunks
                if not df_micro.empty:
                    new_micro = new_micro[new_micro.index > df_micro.index.max()]
                if not new_micro.empty:
                    df_micro = pd.concat([df_micro, new_micro]).sort_index()
                    df_micro = df_micro[~df_micro.index.duplicated(keep='last')]
                    df_micro.to_parquet(micro_uri, engine='pyarrow')
                    print(f"      Micro saved: {len(df_micro):,} rows")
                else:
                    print(f"      No new micro rows")
            else:
                print(f"      No T&Q data found in gap")
    else:
        try:
            df_micro = pd.read_parquet(micro_uri)
            if df_micro.index.tz is None:
                df_micro.index = df_micro.index.tz_localize('UTC')
            print(f"   Micro loaded: {len(df_micro):,} rows")
        except Exception:
            df_micro = pd.DataFrame()
            print(f"   No micro found — master will be OHLCV-only")

    # ── STEP 2: Build master_inference ───────────────────────────────
    ohlcv_uri  = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
    master_uri = f"{GCS_OHLCV_DIR}/{ticker}_master_inference.parquet"

    try:
        df_ohlcv = pd.read_parquet(ohlcv_uri)
        if df_ohlcv.index.tz is None:
            df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
        print(f"   OHLCV loaded: {len(df_ohlcv):,} rows")
    except Exception:
        print(f"   No OHLCV found — skipping master build")
        continue

    if not df_micro.empty:
        df_master = df_ohlcv.join(df_micro, how='left')
    else:
        df_master = df_ohlcv.copy()

    df_master = heal_data(df_master)
    df_master = df_master[df_master.index.notna()].sort_index()
    df_master.to_parquet(master_uri, engine='pyarrow')
    print(f"   master_inference saved: {len(df_master):,} rows")

print(f"\nbuild_micro_153.py COMPLETE")
