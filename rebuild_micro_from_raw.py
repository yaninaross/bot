"""
rebuild_micro_from_raw.py — Rebuild ALL microstructure from raw T&Q (2022+)
============================================================================
Scans Data/TradesQuotes/{ticker}/ in GCS for all available daily raw files,
processes them into 1-min microstructure bars, and writes the full
microstructure parquet. Then rebuilds master_inference with the merged data.

This is the one-time backfill script to get micro data from 2022 onward
so models can actually learn from microstructure features.

For each ticker it:
  1. Scans GCS for all raw daily T&Q parquets (2022+)
  2. Processes each day into 1-min micro bars (Spread, OBI, bsize, asize,
     Total_Vol, Volume_Delta, Trade_Count, Delta_Pct)
  3. Saves the rebuilt microstructure parquet to GCS
  4. Rebuilds master_inference by re-joining OHLCV + new micro (replaces old file)

After this script completes:
  - All master_inference files will have real micro data wherever raw T&Q exists
  - You MUST retrain models to regenerate scaling_stats JSON files
    (old stats had zero IQR for micro columns)
  - Old Results CSVs from training are stale — new training will overwrite them

Processes day-by-day to avoid OOM on tickers with heavy tick data (NVDA etc).

Usage:
  python rebuild_micro_from_raw.py                    # all tickers
  python rebuild_micro_from_raw.py RIOT CLSK IREN     # specific tickers
"""

import sys
import os
import io
import gc
import warnings

import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta
from google.cloud import storage

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"
GCS_RAW_DIR = "Data/TradesQuotes"
GCS_MICRO_DIR = "Data/Inference/Microstructure"
GCS_OHLCV_DIR = "Data/Inference"

ALL_TICKERS = [
    # Exp-150/151 (Energy / AI Power)
    'VST', 'CEG', 'GEV', 'BE', 'CCJ', 'NEE', 'DUK',
    # Exp-190 (Space / Dual-Use)
    'PL', 'RDW', 'BKSY', 'MNTS', 'VOYG',
    # Defense Primes
    'GD', 'LHX', 'LMT', 'NOC', 'RTX', 'GE',
    # EXP-151/152 additions (energy expansion + cross-sector spikers)
    'OKLO', 'SMR', 'FCEL', 'FLNC', 'PLUG', 'MP', 'POWL', 'MOD', 'ALAB',
]
# Context assets (SMH, ITA, MOD, VIX, TLT, IWM, BTCUSD, SPY) are OHLCV-only — no micro needed

DATA_START = "2022-01-01"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# ==============================================================================
# SCHEMA NORMALIZATION (from daily_micro_all.py)
# ==============================================================================
def normalize_raw_schema(df, kind):
    if df is None or df.empty:
        return pd.DataFrame()
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


# ==============================================================================
# MICROSTRUCTURE ENGINE (from daily_micro_all.py)
# ==============================================================================
def process_raw_to_microstructure(trades_df, quotes_df):
    if trades_df is not None and not trades_df.empty:
        trades_df = trades_df.set_index('timestamp')
        trades_df['Direction'] = np.sign(trades_df['price'].diff())
        trades_df['Direction'] = trades_df['Direction'].mask(
            trades_df['Direction'] == 0).ffill().fillna(0)
        trades_df['Volume_Delta'] = trades_df['size'] * trades_df['Direction']
        t_1m = trades_df.resample('1min').agg(
            {'size': 'sum', 'Volume_Delta': 'sum', 'price': 'count'}
        ).rename(columns={'size': 'Total_Vol', 'price': 'Trade_Count'}).fillna(0)
    else:
        t_1m = pd.DataFrame(columns=['Total_Vol', 'Volume_Delta', 'Trade_Count'])

    if quotes_df is not None and not quotes_df.empty:
        quotes_df = quotes_df.set_index('timestamp')
        quotes_df['Spread'] = quotes_df['ask_price'] - quotes_df['bid_price']
        quotes_df['OBI'] = (quotes_df['bid_size'] - quotes_df['ask_size']) / \
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
# HEAL (from rebuild_master_all.py)
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
# MAIN
# ==============================================================================
def main(ticker_list=None):
    print("=" * 70)
    print("  REBUILD MICROSTRUCTURE FROM RAW T&Q DATA (2022+)")
    print(f"  Run time: {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    if ticker_list:
        tickers = [t.upper() for t in ticker_list]
    else:
        cli_tickers = [a.upper() for a in sys.argv[1:]
                       if not a.startswith('-') and not a.endswith('.json')]
        tickers = cli_tickers if cli_tickers else ALL_TICKERS
    print(f"  Tickers: {len(tickers)} — {tickers}")

    summary = []

    for ticker in tickers:
        print(f"\n{'─' * 70}")
        print(f"  {ticker}")
        print(f"{'─' * 70}")

        raw_prefix = f"{GCS_RAW_DIR}/{ticker}/"

        # ── Step 1: Discover all raw T&Q files in GCS ──
        print(f"  Scanning {raw_prefix}...")
        blobs = list(bucket.list_blobs(prefix=raw_prefix))
        if not blobs:
            print(f"  No raw T&Q files found — skipping")
            summary.append((ticker, 'NO_RAW_DATA', 0, 0, '-', '-'))
            continue

        # Parse dates from filenames
        trade_dates = set()
        quote_dates = set()
        for b in blobs:
            fname = os.path.basename(b.name)
            if fname.startswith(f"{ticker}_trades_") and fname.endswith('.parquet'):
                date_str = fname.replace(f"{ticker}_trades_", '').replace('.parquet', '')
                try:
                    d = pd.Timestamp(date_str)
                    if d >= pd.Timestamp(DATA_START):
                        trade_dates.add(date_str)
                except:
                    pass
            elif fname.startswith(f"{ticker}_quotes_") and fname.endswith('.parquet'):
                date_str = fname.replace(f"{ticker}_quotes_", '').replace('.parquet', '')
                try:
                    d = pd.Timestamp(date_str)
                    if d >= pd.Timestamp(DATA_START):
                        quote_dates.add(date_str)
                except:
                    pass

        # Dates where we have both trades AND quotes
        both_dates = sorted(trade_dates & quote_dates)
        trade_only = sorted(trade_dates - quote_dates)
        quote_only = sorted(quote_dates - trade_dates)

        print(f"  Raw files: {len(trade_dates)} trade days, {len(quote_dates)} quote days")
        print(f"  Both T&Q: {len(both_dates)} days")
        if both_dates:
            print(f"  Date range: {both_dates[0]} → {both_dates[-1]}")
        if trade_only:
            print(f"  Trade-only days: {len(trade_only)} (will process with empty quotes)")
        if quote_only:
            print(f"  Quote-only days: {len(quote_only)} (will process with empty trades)")

        # All processable dates
        all_dates = sorted(set(both_dates) | set(trade_only) | set(quote_only))
        if not all_dates:
            print(f"  No processable dates after {DATA_START} — skipping")
            summary.append((ticker, 'NO_DATES', 0, 0, '-', '-'))
            continue

        # ── Step 2: Process day-by-day ──
        print(f"  Processing {len(all_dates)} days...")
        micro_chunks = []
        processed = 0
        errors = 0

        for i, date_str in enumerate(all_dates):
            t_uri = f"{GCS_RAW_DIR}/{ticker}/{ticker}_trades_{date_str}.parquet"
            q_uri = f"{GCS_RAW_DIR}/{ticker}/{ticker}_quotes_{date_str}.parquet"

            try:
                dt = pd.DataFrame()
                dq = pd.DataFrame()

                t_blob = bucket.blob(t_uri)
                if t_blob.exists():
                    dt = pd.read_parquet(io.BytesIO(t_blob.download_as_bytes()))
                    dt = normalize_raw_schema(dt, 'trades')

                q_blob = bucket.blob(q_uri)
                if q_blob.exists():
                    dq = pd.read_parquet(io.BytesIO(q_blob.download_as_bytes()))
                    dq = normalize_raw_schema(dq, 'quotes')

                if dt.empty and dq.empty:
                    continue

                day_micro = process_raw_to_microstructure(dt, dq)
                del dt, dq
                gc.collect()

                if not day_micro.empty:
                    micro_chunks.append(day_micro)
                    processed += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    Error on {date_str}: {e}")
                elif errors == 6:
                    print(f"    ... suppressing further errors")

            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(all_dates)} days processed...")

        print(f"  Processed: {processed} days, errors: {errors}")

        if not micro_chunks:
            print(f"  No micro data produced — skipping")
            summary.append((ticker, 'NO_OUTPUT', 0, errors, '-', '-'))
            continue

        # ── Step 3: Concatenate and save micro ──
        print(f"  Concatenating {len(micro_chunks)} day chunks...")
        df_micro = pd.concat(micro_chunks).sort_index()
        df_micro = df_micro[~df_micro.index.duplicated(keep='last')]
        del micro_chunks
        gc.collect()

        micro_path = f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet"
        print(f"  Saving to {micro_path}...")
        print(f"  Shape: {df_micro.shape}, range: {df_micro.index.min()} → {df_micro.index.max()}")

        buf = io.BytesIO()
        df_micro.to_parquet(buf, engine='pyarrow')
        buf.seek(0)
        bucket.blob(micro_path).upload_from_file(buf)
        print(f"  Micro saved: {len(df_micro):,} rows")

        # ── Step 4: Rebuild master_inference ──
        ohlcv_uri = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
        master_uri = f"{GCS_OHLCV_DIR}/{ticker}_master_inference.parquet"

        ohlcv_blob = bucket.blob(ohlcv_uri)
        if ohlcv_blob.exists():
            print(f"  Rebuilding master_inference...")
            df_ohlcv = pd.read_parquet(io.BytesIO(ohlcv_blob.download_as_bytes()))
            if df_ohlcv.index.tz is None:
                df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
            print(f"  OHLCV: {len(df_ohlcv):,} rows")

            if df_micro.index.tz is None:
                df_micro.index = df_micro.index.tz_localize('UTC')

            df_unified = df_ohlcv.join(df_micro, how='left')
            raw_rows = len(df_unified)

            print(f"  Healing {raw_rows:,} rows...")
            df_unified = heal_data(df_unified)
            df_unified = df_unified[df_unified.index.notna()].sort_index()

            buf = io.BytesIO()
            df_unified.to_parquet(buf, engine='pyarrow')
            buf.seek(0)
            bucket.blob(master_uri).upload_from_file(buf)
            print(f"  master_inference saved: {len(df_unified):,} rows")

            micro_notnull = df_unified['Trade_Count'].notna().sum() if 'Trade_Count' in df_unified.columns else 0
            micro_nonzero = (df_unified['Trade_Count'] > 0).sum() if 'Trade_Count' in df_unified.columns else 0
            summary.append((ticker, 'OK', processed, errors,
                            f"{len(df_micro):,}", f"{micro_nonzero:,}/{len(df_unified):,}"))

            del df_ohlcv, df_unified
        else:
            print(f"  No OHLCV found at {ohlcv_uri} — micro saved but master not rebuilt")
            summary.append((ticker, 'MICRO_ONLY', processed, errors,
                            f"{len(df_micro):,}", 'no OHLCV'))

        del df_micro
        gc.collect()

    # ── Summary ──
    print(f"\n\n{'=' * 70}")
    print(f"  REBUILD SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Ticker':<8} {'Status':<12} {'Days':>6} {'Errs':>5} {'MicroRows':>12} {'TC>0/Total':>18}")
    print(f"  {'─'*8} {'─'*12} {'─'*6} {'─'*5} {'─'*12} {'─'*18}")
    for ticker, status, days, errs, micro_rows, coverage in summary:
        print(f"  {ticker:<8} {status:<12} {days:>6} {errs:>5} {micro_rows:>12} {coverage:>18}")

    print(f"\n{'=' * 70}")
    print(f"  REBUILD COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
