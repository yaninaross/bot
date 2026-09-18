"""
rebuild_micro_incremental.py — Incremental Microstructure Rebuild
=================================================================
Like rebuild_micro_from_raw.py but checks for existing microstructure files
and only processes NEW T&Q days. Appends to existing micro parquet, then
rebuilds master_inference.

For each ticker it:
  1. Checks if microstructure parquet already exists in GCS
  2. If yes, finds the last date and only processes T&Q after that date
  3. If no, processes all available T&Q (full rebuild)
  4. Appends new micro rows to existing, deduplicates, saves
  5. Rebuilds master_inference from OHLCV + updated micro

Usage:
  python rebuild_micro_incremental.py                          # all tickers
  python rebuild_micro_incremental.py BBAI RCAT LUNR CIFR     # specific tickers
"""

import sys
import os
import io
import gc
import warnings

import numpy as np
import pandas as pd
import pytz
from datetime import datetime
from google.cloud import storage

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
GCS_RAW_DIR    = "Data/TradesQuotes"
GCS_MICRO_DIR  = "Data/Inference/Microstructure"
GCS_OHLCV_DIR  = "Data/Inference"
GCS_MASTER_DIR = "Data/Inference_v2"

ALL_TICKERS = [
    # S3 deploy tickers (10)
    'APLD', 'CIFR', 'CLSK', 'FLNC', 'HUT', 'IREN', 'KEEL', 'PL', 'RCAT', 'RIOT',
    # Train teachers needed for model inference (not already above)
    'ASTS', 'BBAI', 'LUNR', 'MVIS', 'WULF',
    # Additional tickers for broader WF coverage
    'ALAB', 'OKLO', 'RKLB', 'RDW', 'SMCI',
]

DATA_START = "2022-01-01"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)


# ==============================================================================
# SCHEMA NORMALIZATION
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
# MICROSTRUCTURE ENGINE
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
# HEAL
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
def main(ticker_list=None, force=False):
    force_rebuild = force or ('--force' in sys.argv)

    print("=" * 70)
    print("  INCREMENTAL MICROSTRUCTURE REBUILD")
    if force_rebuild:
        print("  MODE: --force (ignoring existing micro, full rebuild)")
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
        micro_path = f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet"

        # ── Step 1: Check existing microstructure ──
        micro_blob = bucket.blob(micro_path)
        df_existing_micro = None
        last_micro_date = None

        if force_rebuild:
            print(f"  --force: ignoring existing micro, full rebuild")
        elif micro_blob.exists():
            try:
                micro_blob.reload()
                size_mb = micro_blob.size / (1024 * 1024)
                df_existing_micro = pd.read_parquet(
                    io.BytesIO(micro_blob.download_as_bytes()))
                last_micro_date = df_existing_micro.index.max()
                last_date_str = pd.Timestamp(last_micro_date).strftime('%Y-%m-%d')
                print(f"  Existing micro: {len(df_existing_micro):,} rows, "
                      f"{size_mb:.1f} MB, last date: {last_date_str}")
            except Exception as e:
                print(f"  Could not read existing micro: {e}")
                print(f"  Will do full rebuild")
                df_existing_micro = None
        else:
            print(f"  No existing micro — full rebuild")

        # ── Step 2: Discover raw T&Q files ──
        print(f"  Scanning {raw_prefix}...")
        blobs = list(bucket.list_blobs(prefix=raw_prefix))
        if not blobs:
            print(f"  No raw T&Q files found — skipping")
            summary.append((ticker, 'NO_RAW_DATA', 0, 0, '-', '-'))
            if df_existing_micro is not None:
                del df_existing_micro
            continue

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
                except Exception:
                    pass
            elif fname.startswith(f"{ticker}_quotes_") and fname.endswith('.parquet'):
                date_str = fname.replace(f"{ticker}_quotes_", '').replace('.parquet', '')
                try:
                    d = pd.Timestamp(date_str)
                    if d >= pd.Timestamp(DATA_START):
                        quote_dates.add(date_str)
                except Exception:
                    pass

        all_raw_dates = sorted(
            (trade_dates | quote_dates) & (trade_dates | quote_dates))
        both_dates = sorted(trade_dates & quote_dates)
        trade_only = sorted(trade_dates - quote_dates)
        quote_only = sorted(quote_dates - trade_dates)
        all_raw_dates = sorted(set(both_dates) | set(trade_only) | set(quote_only))

        print(f"  Raw files: {len(trade_dates)} trade days, {len(quote_dates)} quote days")

        # ── Step 3: Filter to only NEW dates ──
        if last_micro_date is not None:
            cutoff = pd.Timestamp(last_micro_date).strftime('%Y-%m-%d')
            dates_to_process = [d for d in all_raw_dates if d > cutoff]
            skipped = len(all_raw_dates) - len(dates_to_process)
            print(f"  Already have micro through {cutoff}")
            print(f"  Skipping {skipped} days, processing {len(dates_to_process)} new days")
        else:
            dates_to_process = all_raw_dates
            print(f"  Processing all {len(dates_to_process)} days (no existing micro)")

        if not dates_to_process:
            print(f"  Up to date — nothing to process")
            micro_rows = f"{len(df_existing_micro):,}" if df_existing_micro is not None else '-'
            summary.append((ticker, 'UP_TO_DATE', 0, 0, micro_rows, '-'))
            if df_existing_micro is not None:
                del df_existing_micro
            continue

        # ── Step 4: Process new days ──
        print(f"  Processing {len(dates_to_process)} days...")
        micro_chunks = []
        processed = 0
        errors = 0

        for i, date_str in enumerate(dates_to_process):
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
                print(f"    {i + 1}/{len(dates_to_process)} days processed...")

        print(f"  Processed: {processed} days, errors: {errors}")

        if not micro_chunks and df_existing_micro is None:
            print(f"  No micro data produced — skipping")
            summary.append((ticker, 'NO_OUTPUT', 0, errors, '-', '-'))
            continue

        # ── Step 5: Merge with existing micro ──
        if micro_chunks:
            df_new_micro = pd.concat(micro_chunks).sort_index()
            del micro_chunks
            gc.collect()
            print(f"  New micro: {len(df_new_micro):,} rows "
                  f"({df_new_micro.index.min()} → {df_new_micro.index.max()})")
        else:
            df_new_micro = None

        if df_existing_micro is not None and df_new_micro is not None:
            if df_existing_micro.index.tz is not None and df_new_micro.index.tz is None:
                df_new_micro.index = df_new_micro.index.tz_localize(
                    df_existing_micro.index.tz)
            elif df_existing_micro.index.tz is None and df_new_micro.index.tz is not None:
                df_existing_micro.index = df_existing_micro.index.tz_localize(
                    df_new_micro.index.tz)

            before = len(df_existing_micro)
            df_micro = pd.concat([df_existing_micro, df_new_micro])
            df_micro = df_micro[~df_micro.index.duplicated(keep='last')]
            df_micro.sort_index(inplace=True)
            added = len(df_micro) - before
            print(f"  Appended {added:,} new rows (total: {len(df_micro):,})")
            del df_existing_micro, df_new_micro
        elif df_new_micro is not None:
            df_micro = df_new_micro
            print(f"  Fresh micro: {len(df_micro):,} rows")
            del df_new_micro
        else:
            df_micro = df_existing_micro
            del df_existing_micro

        gc.collect()

        # ── Step 6: Save micro ──
        print(f"  Saving to {micro_path}...")
        print(f"  Shape: {df_micro.shape}, "
              f"range: {df_micro.index.min()} → {df_micro.index.max()}")

        buf = io.BytesIO()
        df_micro.to_parquet(buf, engine='pyarrow')
        buf.seek(0)
        bucket.blob(micro_path).upload_from_file(buf)
        print(f"  Micro saved: {len(df_micro):,} rows")

        # ── Step 7: Rebuild master_inference ──
        ohlcv_uri = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
        master_uri = f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet"

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
            del df_ohlcv
            df_unified = df_unified[~df_unified.index.duplicated(keep='last')]

            print(f"  Healing {len(df_unified):,} rows...")
            df_unified = heal_data(df_unified)
            df_unified = df_unified[df_unified.index.notna()].sort_index()

            buf = io.BytesIO()
            df_unified.to_parquet(buf, engine='pyarrow')
            buf.seek(0)
            bucket.blob(master_uri).upload_from_file(buf)
            print(f"  master_inference saved: {len(df_unified):,} rows")

            micro_nonzero = (df_unified['Trade_Count'] > 0).sum() \
                if 'Trade_Count' in df_unified.columns else 0
            summary.append((ticker, 'OK', processed, errors,
                            f"{len(df_micro):,}",
                            f"{micro_nonzero:,}/{len(df_unified):,}"))
            del df_unified
        else:
            print(f"  No OHLCV at {ohlcv_uri} — micro saved, master not rebuilt")
            summary.append((ticker, 'MICRO_ONLY', processed, errors,
                            f"{len(df_micro):,}", 'no OHLCV'))

        del df_micro
        gc.collect()

    # ── Summary ──
    print(f"\n\n{'=' * 70}")
    print(f"  INCREMENTAL REBUILD SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Ticker':<8} {'Status':<12} {'NewDays':>8} {'Errs':>5} "
          f"{'MicroRows':>12} {'TC>0/Total':>18}")
    print(f"  {'─'*8} {'─'*12} {'─'*8} {'─'*5} {'─'*12} {'─'*18}")
    for ticker, status, days, errs, micro_rows, coverage in summary:
        print(f"  {ticker:<8} {status:<12} {days:>8} {errs:>5} "
              f"{micro_rows:>12} {coverage:>18}")

    print(f"\n{'=' * 70}")
    print(f"  REBUILD COMPLETE")
    print(f"{'=' * 70}")


# ==============================================================================
# RUN CONFIG — edit these before running the cell
# ==============================================================================
RUN_TICKERS = ['BGDE']        # set to None for all tickers
FORCE_REBUILD = True           # True = ignore existing micro, rebuild from scratch

main(ticker_list=RUN_TICKERS, force=FORCE_REBUILD)
