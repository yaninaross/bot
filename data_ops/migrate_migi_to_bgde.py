"""
migrate_migi_to_bgde.py — Copy MIGI T&Q data into BGDE folder, rebuild micro + master
======================================================================================
MIGI (Mawson Infrastructure) renamed to BGDE (Big Digital Energy) on 2026-04-30.

BGDE already has:
  - OHLCV (from daily feed under new ticker) — no copy needed
  - T&Q from 2026-03-02 onward (post-rename) — 150 files
  - master_inference (but with near-zero micro because micro was built from sparse BGDE T&Q)

MIGI has:
  - T&Q from 2022-01-03 to 2026-06-12 — 2,320 files (full history under old ticker)
  - Micro with 6.9% nonzero Volume_Delta — passes 5% threshold

This script:
  1. Copies MIGI T&Q files into BGDE folder (skips dates BGDE already has)
  2. Rebuilds micro from all T&Q in the BGDE folder (MIGI history + BGDE post-rename)
  3. Trims OHLCV to 2022+ and rebuilds master_inference

After this, BGDE will have full micro coverage and pass the scoring quality gate.

Usage (Colab cell):
  COPY_MODE = True
  main(copy=COPY_MODE)
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
SRC_TICKER  = "MIGI"
DST_TICKER  = "BGDE"

GCS_RAW_DIR    = "Data/TradesQuotes"
GCS_MICRO_DIR  = "Data/Inference/Microstructure"
GCS_OHLCV_DIR  = "Data/Inference"
GCS_MASTER_DIR = "Data/Inference_v2"

DATA_START = pd.Timestamp("2022-01-01", tz='UTC')

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)


# ==============================================================================
# SCHEMA NORMALIZATION (from rebuild_micro_from_raw.py)
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
# MICROSTRUCTURE ENGINE (from rebuild_micro_from_raw.py)
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
# HEAL (from rebuild_micro_from_raw.py)
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
def main(copy=False):
    dry_run = not copy and "--copy" not in sys.argv

    print("=" * 70)
    print(f"  MIGRATE {SRC_TICKER} -> {DST_TICKER}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'COPY + REBUILD'}")
    print(f"  Run time: {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Copy MIGI T&Q files into BGDE folder
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  PHASE 1: Copy {SRC_TICKER} T&Q -> {DST_TICKER} folder")
    print(f"  {'─' * 60}")

    src_prefix = f"{GCS_RAW_DIR}/{SRC_TICKER}/"
    src_blobs = list(client.list_blobs(BUCKET_NAME, prefix=src_prefix))
    print(f"  {SRC_TICKER} T&Q files: {len(src_blobs)}")

    dst_prefix = f"{GCS_RAW_DIR}/{DST_TICKER}/"
    dst_blobs = set(b.name for b in client.list_blobs(BUCKET_NAME, prefix=dst_prefix))
    print(f"  {DST_TICKER} T&Q files already: {len(dst_blobs)}")

    copied, skipped, errors = 0, 0, 0

    for blob in src_blobs:
        fname = os.path.basename(blob.name)
        if not fname.endswith(".parquet"):
            continue
        if not (fname.startswith(f"{SRC_TICKER}_trades_") or
                fname.startswith(f"{SRC_TICKER}_quotes_")):
            continue

        date_str = fname.replace(f"{SRC_TICKER}_trades_", "").replace(
            f"{SRC_TICKER}_quotes_", "").replace(".parquet", "")
        dst_fname = fname.replace(f"{SRC_TICKER}_", f"{DST_TICKER}_", 1)
        dst_path = f"{GCS_RAW_DIR}/{DST_TICKER}/{dst_fname}"

        if dst_path in dst_blobs:
            skipped += 1
            continue

        if dry_run:
            copied += 1
        else:
            try:
                bucket.copy_blob(blob, bucket, dst_path)
                copied += 1
                if copied % 100 == 0:
                    print(f"    Copied {copied} files...")
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    Error copying {fname}: {e}")

    action = "would copy" if dry_run else "copied"
    print(f"  T&Q: {action} {copied}, skipped {skipped} (already exist), {errors} errors")

    if dry_run:
        print(f"\n  DRY RUN complete. Set COPY_MODE = True to execute.")
        return

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Rebuild micro from all BGDE T&Q (MIGI history + BGDE new)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  PHASE 2: Rebuild {DST_TICKER} microstructure from merged T&Q")
    print(f"  {'─' * 60}")

    all_blobs = list(client.list_blobs(BUCKET_NAME, prefix=dst_prefix))
    trade_dates = set()
    quote_dates = set()
    for b in all_blobs:
        fname = os.path.basename(b.name)
        if fname.startswith(f"{DST_TICKER}_trades_") and fname.endswith('.parquet'):
            d = fname.replace(f"{DST_TICKER}_trades_", "").replace(".parquet", "")
            try:
                if pd.Timestamp(d) >= DATA_START:
                    trade_dates.add(d)
            except Exception:
                pass
        elif fname.startswith(f"{DST_TICKER}_quotes_") and fname.endswith('.parquet'):
            d = fname.replace(f"{DST_TICKER}_quotes_", "").replace(".parquet", "")
            try:
                if pd.Timestamp(d) >= DATA_START:
                    quote_dates.add(d)
            except Exception:
                pass

    both_dates = sorted(trade_dates & quote_dates)
    trade_only = sorted(trade_dates - quote_dates)
    quote_only = sorted(quote_dates - trade_dates)
    all_dates = sorted(set(both_dates) | set(trade_only) | set(quote_only))

    print(f"  Total {DST_TICKER} T&Q: {len(trade_dates)} trade days, {len(quote_dates)} quote days")
    print(f"  Both T&Q: {len(both_dates)}, trade-only: {len(trade_only)}, quote-only: {len(quote_only)}")
    print(f"  Processing {len(all_dates)} days...")

    micro_chunks = []
    processed = 0
    proc_errors = 0

    for i, date_str in enumerate(all_dates):
        t_uri = f"{GCS_RAW_DIR}/{DST_TICKER}/{DST_TICKER}_trades_{date_str}.parquet"
        q_uri = f"{GCS_RAW_DIR}/{DST_TICKER}/{DST_TICKER}_quotes_{date_str}.parquet"

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
            proc_errors += 1
            if proc_errors <= 5:
                print(f"    Error on {date_str}: {e}")

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(all_dates)} days processed...")

    print(f"  Processed: {processed} days, errors: {proc_errors}")

    if not micro_chunks:
        print(f"  No micro data produced!")
        return

    df_micro = pd.concat(micro_chunks).sort_index()
    df_micro = df_micro[~df_micro.index.duplicated(keep='last')]
    del micro_chunks
    gc.collect()

    micro_path = f"{GCS_MICRO_DIR}/{DST_TICKER}_1m_microstructure.parquet"
    print(f"  Micro: {len(df_micro):,} rows ({df_micro.index.min()} -> {df_micro.index.max()})")

    vd_nz_micro = (df_micro['Volume_Delta'] != 0).sum()
    print(f"  Volume_Delta nonzero in micro: {vd_nz_micro:,} / {len(df_micro):,} "
          f"= {vd_nz_micro/len(df_micro)*100:.1f}%")

    buf = io.BytesIO()
    df_micro.to_parquet(buf, engine='pyarrow')
    buf.seek(0)
    bucket.blob(micro_path).upload_from_file(buf)
    print(f"  Saved: {micro_path}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Rebuild master_inference (OHLCV trimmed to 2022+)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n  PHASE 3: Rebuild {DST_TICKER} master_inference (trimmed to 2022+)")
    print(f"  {'─' * 60}")

    ohlcv_path = f"{GCS_OHLCV_DIR}/{DST_TICKER}_1m_full_historical_data.parquet"
    master_path = f"{GCS_MASTER_DIR}/{DST_TICKER}_master_inference.parquet"

    ohlcv_blob = bucket.blob(ohlcv_path)
    if not ohlcv_blob.exists():
        print(f"  No OHLCV at {ohlcv_path}")
        print(f"  Micro saved but master_inference NOT rebuilt.")
        del df_micro
        return

    df_ohlcv = pd.read_parquet(io.BytesIO(ohlcv_blob.download_as_bytes()))
    if df_ohlcv.index.tz is None:
        df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')

    before_trim = len(df_ohlcv)
    df_ohlcv = df_ohlcv[df_ohlcv.index >= DATA_START]
    trimmed = before_trim - len(df_ohlcv)
    print(f"  OHLCV: {before_trim:,} rows, trimmed {trimmed:,} pre-2022 rows, kept {len(df_ohlcv):,}")

    if df_micro.index.tz is None:
        df_micro.index = df_micro.index.tz_localize('UTC')

    df_master = df_ohlcv.join(df_micro, how='left')
    df_master = df_master[~df_master.index.duplicated(keep='last')]
    del df_ohlcv, df_micro
    gc.collect()

    print(f"  Healing {len(df_master):,} rows...")
    df_master = heal_data(df_master)
    df_master = df_master[df_master.index.notna()].sort_index()

    vd_nz = (df_master['Volume_Delta'] != 0).sum() if 'Volume_Delta' in df_master.columns else 0
    tc_nz = (df_master['Trade_Count'] > 0).sum() if 'Trade_Count' in df_master.columns else 0
    vd_pct = vd_nz / len(df_master) * 100
    tc_pct = tc_nz / len(df_master) * 100
    threshold_status = "PASS" if vd_pct >= 5.0 else "FAIL"

    print(f"  Master: {len(df_master):,} rows")
    print(f"  Volume_Delta nonzero: {vd_nz:,} / {len(df_master):,} = {vd_pct:.1f}% -- {threshold_status}")
    print(f"  Trade_Count > 0:      {tc_nz:,} / {len(df_master):,} = {tc_pct:.1f}%")

    buf = io.BytesIO()
    df_master.to_parquet(buf, engine='pyarrow')
    buf.seek(0)
    bucket.blob(master_path).upload_from_file(buf)

    size_mb = buf.tell() / (1024 * 1024)
    print(f"  Saved: {master_path} ({size_mb:.1f} MB)")

    del df_master
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  MIGRATION COMPLETE: {SRC_TICKER} -> {DST_TICKER}")
    print(f"  T&Q copied: {copied}, micro days: {processed}, master VD%: {vd_pct:.1f}% ({threshold_status})")
    print(f"\n  Next steps:")
    print(f"  1. Run rebuild_master_trimmed.py for BE, FCEL, MSTR")
    print(f"  2. Run exp_152_score_only.py + exp_160_score_only.py")
    print(f"  3. Run optimize_wf_v12.py")
    print(f"{'=' * 70}")


# ==============================================================================
# RUN CONFIG — edit before running in Colab
# ==============================================================================
COPY_MODE = True    # False = dry run (list files), True = copy + rebuild

main(copy=COPY_MODE)
