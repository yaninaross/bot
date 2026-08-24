"""
backfill_micro_parallel.py — Bulk T&Q backfill + micro rebuild (Exp 400)
=========================================================================
Downloads missing T&Q from Polygon for Exp 400 teachers with incomplete
micro coverage. Gap-aware: scans GCS for existing raw T&Q files, downloads
only dates without cached data (beginning, middle, or end of range).

After download:
  - Rebuilds microstructure parquet from ALL raw T&Q
  - Rebuilds master_inference by joining OHLCV + new micro

3 phases:
  Phase 1: Download missing T&Q from Polygon → GCS cache
  Phase 2: Rebuild micro from all raw T&Q → micro parquet
  Phase 3: Stitch OHLCV + micro → master_inference parquet

Colab-ready. Uses google.cloud.storage API (not FUSE).
"""

import os, io, time, gc, ctypes, sys, subprocess, warnings
import numpy as np, pandas as pd, pytz
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

try:
    from polygon import RESTClient
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "polygon-api-client", "google-cloud-storage", "pyarrow", "--quiet"])
    from polygon import RESTClient

from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
BUCKET_NAME     = "time-series-bucket-ys"
GCS_RAW_DIR     = "Data/TradesQuotes"
GCS_MICRO_DIR   = "Data/Inference/Microstructure"
GCS_OHLCV_DIR   = "Data/Inference"
GCS_MASTER_DIR  = "Data/Inference_v2"

# Exp 400 teachers with MICRO_LATE from audit
TARGET_TICKERS = [
    # Severe MICRO_LATE (43mo gap — micro only from Sep 2025)
    'ACMR', 'ARM', 'BEAM', 'FCEL', 'MRNA', 'NTLA', 'TWST', 'MU',
    # Mild gaps (3-12mo)
    'CEG', 'SMR', 'SOUN', 'LUNR',
]

HISTORY_START   = pd.Timestamp("2022-01-01", tz='UTC')
FETCH_END       = pd.Timestamp.now(tz='UTC').normalize()
POLYGON_SLEEP   = 0.35

# Set to True to skip Phase 1 if T&Q already downloaded (re-run rebuild only)
SKIP_DOWNLOAD   = False

gcs_client = gcs_storage.Client()
bucket     = gcs_client.bucket(BUCKET_NAME)


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
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# PHASE 1: SCAN GCS + DOWNLOAD MISSING T&Q
# ==============================================================================
def scan_existing_tq(ticker):
    """Scan GCS for existing raw T&Q files. Returns (trade_dates, quote_dates)."""
    prefix = f"{GCS_RAW_DIR}/{ticker}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    trade_dates = set()
    quote_dates = set()
    for b in blobs:
        fname = os.path.basename(b.name)
        if fname.startswith(f"{ticker}_trades_") and fname.endswith('.parquet'):
            date_str = fname.replace(f"{ticker}_trades_", '').replace('.parquet', '')
            try:
                pd.Timestamp(date_str)
                trade_dates.add(date_str)
            except Exception:
                pass
        elif fname.startswith(f"{ticker}_quotes_") and fname.endswith('.parquet'):
            date_str = fname.replace(f"{ticker}_quotes_", '').replace('.parquet', '')
            try:
                pd.Timestamp(date_str)
                quote_dates.add(date_str)
            except Exception:
                pass
    return trade_dates, quote_dates


def download_missing_tq(ticker):
    """Download T&Q for dates missing from GCS. Returns (downloaded, existing, empty, errors)."""
    trade_dates, quote_dates = scan_existing_tq(ticker)
    both_existing = trade_dates & quote_dates

    full_range = pd.bdate_range(HISTORY_START, FETCH_END)
    all_needed = {d.strftime('%Y-%m-%d') for d in full_range}

    missing = sorted(all_needed - both_existing)

    if not missing:
        print(f"  [{ticker:6s}] All {len(all_needed)} bdays cached — skip download")
        return 0, len(all_needed), 0, 0

    first_existing = min(both_existing) if both_existing else "none"
    last_existing  = max(both_existing) if both_existing else "none"
    print(f"  [{ticker:6s}] GCS cache: {len(both_existing)} dates ({first_existing} -> {last_existing})")
    print(f"  [{ticker:6s}] Missing:   {len(missing)} dates ({missing[0]} -> {missing[-1]})")

    poly = RESTClient(POLYGON_API_KEY)
    downloaded = 0
    empty = 0
    errors = 0

    for i, date_str in enumerate(missing):
        day_start = pd.Timestamp(date_str, tz='UTC')
        day_end   = day_start + timedelta(days=1)

        try:
            trades = []
            for t in poly.list_trades(ticker, timestamp_gte=day_start,
                                      timestamp_lte=day_end, limit=50000):
                trades.append({
                    'timestamp': pd.to_datetime(t.sip_timestamp, unit='ns', utc=True),
                    'price': t.price, 'size': t.size,
                })

            quotes = []
            for q in poly.list_quotes(ticker, timestamp_gte=day_start,
                                      timestamp_lte=day_end, limit=50000):
                quotes.append({
                    'timestamp': pd.to_datetime(q.sip_timestamp, unit='ns', utc=True),
                    'ask_price': q.ask_price, 'ask_size': q.ask_size,
                    'bid_price': q.bid_price, 'bid_size': q.bid_size,
                })

            time.sleep(POLYGON_SLEEP)

            dt = pd.DataFrame(trades) if trades else pd.DataFrame()
            dq = pd.DataFrame(quotes) if quotes else pd.DataFrame()

            if dt.empty and dq.empty:
                empty += 1
                continue

            if not dt.empty:
                buf = io.BytesIO()
                dt.to_parquet(buf, engine='pyarrow')
                buf.seek(0)
                bucket.blob(
                    f"{GCS_RAW_DIR}/{ticker}/{ticker}_trades_{date_str}.parquet"
                ).upload_from_file(buf)

            if not dq.empty:
                buf = io.BytesIO()
                dq.to_parquet(buf, engine='pyarrow')
                buf.seek(0)
                bucket.blob(
                    f"{GCS_RAW_DIR}/{ticker}/{ticker}_quotes_{date_str}.parquet"
                ).upload_from_file(buf)

            downloaded += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [{ticker:6s}]   Error {date_str}: {e}")
            elif errors == 6:
                print(f"  [{ticker:6s}]   ... suppressing further errors")

        if (i + 1) % 100 == 0:
            print(f"  [{ticker:6s}]   {i+1}/{len(missing)} — "
                  f"{downloaded} dl / {empty} empty / {errors} err")

    total_cached = downloaded + len(both_existing)
    print(f"  [{ticker:6s}] Done: +{downloaded} new, {total_cached} total cached, "
          f"{empty} empty days, {errors} errors")
    return downloaded, len(both_existing), empty, errors


# ==============================================================================
# PHASE 2: REBUILD MICRO FROM ALL RAW T&Q
# ==============================================================================
def rebuild_micro(ticker):
    """Rebuild full micro parquet from all raw T&Q on GCS. Returns row count."""
    prefix = f"{GCS_RAW_DIR}/{ticker}/"
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        print(f"  [{ticker:6s}] No raw T&Q on GCS — cannot rebuild micro")
        return 0

    trade_dates = set()
    quote_dates = set()
    for b in blobs:
        fname = os.path.basename(b.name)
        if fname.startswith(f"{ticker}_trades_") and fname.endswith('.parquet'):
            date_str = fname.replace(f"{ticker}_trades_", '').replace('.parquet', '')
            try:
                if pd.Timestamp(date_str, tz='UTC') >= HISTORY_START:
                    trade_dates.add(date_str)
            except Exception:
                pass
        elif fname.startswith(f"{ticker}_quotes_") and fname.endswith('.parquet'):
            date_str = fname.replace(f"{ticker}_quotes_", '').replace('.parquet', '')
            try:
                if pd.Timestamp(date_str, tz='UTC') >= HISTORY_START:
                    quote_dates.add(date_str)
            except Exception:
                pass

    all_dates = sorted(trade_dates | quote_dates)
    if not all_dates:
        print(f"  [{ticker:6s}] No dates >= {HISTORY_START.strftime('%Y-%m-%d')} — skip")
        return 0

    print(f"  [{ticker:6s}] Processing {len(all_dates)} days "
          f"({all_dates[0]} -> {all_dates[-1]})")

    micro_chunks = []
    processed = 0
    errors = 0

    for i, date_str in enumerate(all_dates):
        t_path = f"{GCS_RAW_DIR}/{ticker}/{ticker}_trades_{date_str}.parquet"
        q_path = f"{GCS_RAW_DIR}/{ticker}/{ticker}_quotes_{date_str}.parquet"

        try:
            dt = pd.DataFrame()
            dq = pd.DataFrame()

            t_blob = bucket.blob(t_path)
            if t_blob.exists():
                dt = pd.read_parquet(io.BytesIO(t_blob.download_as_bytes()))
                dt = normalize_raw_schema(dt, 'trades')

            q_blob = bucket.blob(q_path)
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
                print(f"  [{ticker:6s}]   Error {date_str}: {e}")
            elif errors == 6:
                print(f"  [{ticker:6s}]   ... suppressing further errors")

        if (i + 1) % 200 == 0:
            print(f"  [{ticker:6s}]   {i+1}/{len(all_dates)} days processed...")

    print(f"  [{ticker:6s}] Processed: {processed} days, {errors} errors")

    if not micro_chunks:
        print(f"  [{ticker:6s}] No micro data produced")
        return 0

    df_micro = pd.concat(micro_chunks).sort_index()
    df_micro = df_micro[~df_micro.index.duplicated(keep='last')]
    del micro_chunks
    gc.collect()

    micro_path = f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet"
    buf = io.BytesIO()
    df_micro.to_parquet(buf, engine='pyarrow')
    buf.seek(0)
    bucket.blob(micro_path).upload_from_file(buf)

    print(f"  [{ticker:6s}] Micro saved: {len(df_micro):,} rows "
          f"({df_micro.index.min()} -> {df_micro.index.max()})")

    row_count = len(df_micro)
    del df_micro
    _release_memory()
    return row_count


# ==============================================================================
# PHASE 3: REBUILD MASTER (OHLCV + MICRO -> master_inference)
# ==============================================================================
def rebuild_master(ticker):
    """Stitch OHLCV + micro -> master_inference. Returns row count."""
    ohlcv_path  = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
    micro_path  = f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet"
    master_path = f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet"

    ohlcv_blob = bucket.blob(ohlcv_path)
    if not ohlcv_blob.exists():
        print(f"  [{ticker:6s}] No OHLCV — cannot build master")
        return 0

    micro_blob = bucket.blob(micro_path)
    if not micro_blob.exists():
        print(f"  [{ticker:6s}] No micro — cannot build master")
        return 0

    print(f"  [{ticker:6s}] Loading OHLCV + micro...")
    df_ohlcv = pd.read_parquet(io.BytesIO(ohlcv_blob.download_as_bytes()))
    if df_ohlcv.index.tz is None:
        df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')

    df_micro = pd.read_parquet(io.BytesIO(micro_blob.download_as_bytes()))
    if df_micro.index.tz is None:
        df_micro.index = df_micro.index.tz_localize('UTC')

    print(f"  [{ticker:6s}] OHLCV: {len(df_ohlcv):,} rows, "
          f"Micro: {len(df_micro):,} rows")

    df_master = df_ohlcv.join(df_micro, how='left')
    del df_ohlcv, df_micro
    _release_memory()

    df_master = heal_data(df_master)
    df_master = df_master[df_master.index.notna()].sort_index()

    buf = io.BytesIO()
    df_master.to_parquet(buf, engine='pyarrow')
    buf.seek(0)
    bucket.blob(master_path).upload_from_file(buf)

    micro_nonzero = (df_master['Trade_Count'] > 0).sum() if 'Trade_Count' in df_master.columns else 0
    print(f"  [{ticker:6s}] Master saved: {len(df_master):,} rows, "
          f"TC>0: {micro_nonzero:,} "
          f"({df_master.index.min()} -> {df_master.index.max()})")

    row_count = len(df_master)
    del df_master
    _release_memory()
    return row_count


# ==============================================================================
# MAIN
# ==============================================================================
_t0 = time.time()

print("=" * 70)
print("  BACKFILL T&Q + REBUILD MICRO/MASTER — Exp 400 Teachers")
print(f"  {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print(f"  Tickers: {len(TARGET_TICKERS)} — {TARGET_TICKERS}")
print(f"  Date range: {HISTORY_START.strftime('%Y-%m-%d')} -> "
      f"{FETCH_END.strftime('%Y-%m-%d')}")
print(f"  SKIP_DOWNLOAD: {SKIP_DOWNLOAD}")
print("=" * 70)

# ── Phase 1: Download missing T&Q from Polygon ──
dl_summary = []
if not SKIP_DOWNLOAD:
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Download missing T&Q from Polygon")
    print(f"{'='*70}")

    for ticker in TARGET_TICKERS:
        print()
        downloaded, existing, empty, errors = download_missing_tq(ticker)
        dl_summary.append((ticker, downloaded, existing, empty, errors))

    print(f"\n  Phase 1 Summary:")
    print(f"  {'Ticker':<8} {'New DL':>8} {'Existing':>8} {'Empty':>6} {'Errors':>6}")
    print(f"  {'---'*16}")
    for ticker, dl, ex, emp, err in dl_summary:
        print(f"  {ticker:<8} {dl:>8} {ex:>8} {emp:>6} {err:>6}")
    total_dl = sum(d[1] for d in dl_summary)
    print(f"\n  Total new downloads: {total_dl}")
else:
    print(f"\n  PHASE 1: SKIPPED (SKIP_DOWNLOAD = True)")

# ── Phase 2: Rebuild micro from raw T&Q ──
print(f"\n{'='*70}")
print(f"  PHASE 2: Rebuild microstructure from all raw T&Q")
print(f"{'='*70}")

micro_summary = []
for ticker in TARGET_TICKERS:
    print()
    rows = rebuild_micro(ticker)
    micro_summary.append((ticker, rows))

# ── Phase 3: Stitch OHLCV + micro -> master_inference ──
print(f"\n{'='*70}")
print(f"  PHASE 3: Stitch OHLCV + micro -> master_inference")
print(f"{'='*70}")

master_summary = []
for ticker in TARGET_TICKERS:
    print()
    rows = rebuild_master(ticker)
    master_summary.append((ticker, rows))

# ── Final summary ──
elapsed = time.time() - _t0
print(f"\n{'='*70}")
print(f"  FINAL SUMMARY")
print(f"{'='*70}")
print(f"  {'Ticker':<8} {'T&Q DL':>8} {'Micro Rows':>12} {'Master Rows':>12}")
print(f"  {'---'*16}")
for i, ticker in enumerate(TARGET_TICKERS):
    dl = dl_summary[i][1] if dl_summary else 0
    micro = micro_summary[i][1]
    master = master_summary[i][1]
    print(f"  {ticker:<8} {dl:>8} {micro:>12,} {master:>12,}")

print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")
print(f"\n  Next: run audit_master_coverage.py to verify coverage")
print(f"  Then: update ONLY_TICKERS in rebuild_micro_from_raw.py if needed")
print("=" * 70)
