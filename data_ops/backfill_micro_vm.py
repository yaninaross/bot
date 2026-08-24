"""
backfill_micro_vm.py — VM T&Q backfill for remaining scoring assets
====================================================================
Companion to backfill_micro_parallel.py (Colab, 12 Exp 400 teachers).
Downloads T&Q for ALL other scoring pool tickers on this VM.

Per-ticker pipeline:
  1. Scan GCS for existing raw T&Q → find missing dates
  2. Download missing dates from Polygon (skip cached)
  3. If new T&Q downloaded OR no master exists: rebuild micro + master

Gap-aware and resumable — restarting skips already-downloaded dates.
Sequential processing (Polygon API is the bottleneck).
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

# ==============================================================================
# CONFIG
# ==============================================================================
POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
BUCKET_NAME     = "time-series-bucket-ys"
GCS_RAW_DIR     = "Data/TradesQuotes"
GCS_MICRO_DIR   = "Data/Inference/Microstructure"
GCS_OHLCV_DIR   = "Data/Inference"
GCS_MASTER_DIR  = "Data/Inference_v2"

# All scoring pool tickers MINUS 12 already on Colab
COLAB_TICKERS = {
    'ACMR', 'ARM', 'BEAM', 'FCEL', 'MRNA', 'NTLA', 'TWST', 'MU',
    'CEG', 'SMR', 'SOUN', 'LUNR',
}

ALL_SCORING = [
    # --- Original pipeline tickers ---
    'ACHR', 'AEHR', 'ALAB', 'AMD', 'APLD', 'ASTS', 'AVAV',
    'BBAI', 'BE', 'BGDE', 'BHP', 'BKSY', 'BTBT',
    'CAT', 'CCJ', 'CIFR', 'CLSK', 'COIN', 'COP', 'CORZ', 'CRDO', 'CVX',
    'DUK', 'EQIX', 'ETN', 'FCX', 'FLNC',
    'GD', 'GE', 'GEV', 'HOOD', 'HUT', 'IREN',
    'JOBY', 'KEEL', 'KTOS',
    'LHX', 'LIN', 'LMT',
    'MARA', 'MNTS', 'MOD', 'MP', 'MSTR', 'MVIS', 'MXL',
    'NBIS', 'NEE', 'NEOV', 'NNE', 'NNDM', 'NOC', 'NVDA',
    'OKLO', 'ONDS', 'PL', 'PLTR', 'PLUG', 'POWL',
    'RCAT', 'RDW', 'RIOT', 'RKLB', 'RTX',
    'SLS', 'SMCI', 'SPIR',
    'TSLA', 'UCTT', 'VOYG', 'VRT', 'VST', 'WOLF', 'WULF', 'XOM',
    # --- Exp 400 scoring pool additions ---
    'BABA', 'CLF', 'ENPH', 'EQT', 'F', 'FSLR', 'GM',
    'KMI', 'LNG',
    'NET', 'NIO', 'NRG', 'NXE',
    'RIVN', 'ROKU',
    'SQ',
    'UPST',
    'XPEV',
]

TARGET_TICKERS = sorted(set(ALL_SCORING) - COLAB_TICKERS)

HISTORY_START   = pd.Timestamp("2022-01-01", tz='UTC')
FETCH_END       = pd.Timestamp.now(tz='UTC').normalize()
POLYGON_SLEEP   = 0.35

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
    trade_dates, quote_dates = scan_existing_tq(ticker)
    both_existing = trade_dates & quote_dates

    full_range = pd.bdate_range(HISTORY_START, FETCH_END)
    all_needed = {d.strftime('%Y-%m-%d') for d in full_range}
    missing = sorted(all_needed - both_existing)

    if not missing:
        print(f"  [{ticker:6s}] All {len(all_needed)} bdays cached — skip", flush=True)
        return 0, len(all_needed), 0, 0

    first_ex = min(both_existing) if both_existing else "none"
    last_ex  = max(both_existing) if both_existing else "none"
    print(f"  [{ticker:6s}] Cached: {len(both_existing)} ({first_ex} -> {last_ex}), "
          f"missing: {len(missing)} ({missing[0]} -> {missing[-1]})", flush=True)

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
                print(f"  [{ticker:6s}]   Err {date_str}: {e}", flush=True)

        if (i + 1) % 100 == 0:
            print(f"  [{ticker:6s}]   {i+1}/{len(missing)} — "
                  f"{downloaded} dl / {empty} empty / {errors} err", flush=True)

    print(f"  [{ticker:6s}] DL done: +{downloaded}, "
          f"{downloaded + len(both_existing)} total, "
          f"{empty} empty, {errors} err", flush=True)
    return downloaded, len(both_existing), empty, errors


# ==============================================================================
# PHASE 2: REBUILD MICRO FROM ALL RAW T&Q
# ==============================================================================
def rebuild_micro(ticker):
    prefix = f"{GCS_RAW_DIR}/{ticker}/"
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        print(f"  [{ticker:6s}] No raw T&Q — skip micro rebuild", flush=True)
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
        return 0

    print(f"  [{ticker:6s}] Micro: {len(all_dates)} days "
          f"({all_dates[0]} -> {all_dates[-1]})", flush=True)

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
            if errors <= 3:
                print(f"  [{ticker:6s}]   Err {date_str}: {e}", flush=True)

        if (i + 1) % 200 == 0:
            print(f"  [{ticker:6s}]   {i+1}/{len(all_dates)} days...", flush=True)

    if not micro_chunks:
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
          f"({df_micro.index.min()} -> {df_micro.index.max()})", flush=True)

    row_count = len(df_micro)
    del df_micro
    _release_memory()
    return row_count


# ==============================================================================
# PHASE 3: REBUILD MASTER (OHLCV + MICRO -> master_inference)
# ==============================================================================
def rebuild_master(ticker):
    ohlcv_path  = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
    micro_path  = f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet"
    master_path = f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet"

    ohlcv_blob = bucket.blob(ohlcv_path)
    if not ohlcv_blob.exists():
        print(f"  [{ticker:6s}] No OHLCV — skip master", flush=True)
        return 0

    micro_blob = bucket.blob(micro_path)
    if not micro_blob.exists():
        print(f"  [{ticker:6s}] No micro — skip master", flush=True)
        return 0

    df_ohlcv = pd.read_parquet(io.BytesIO(ohlcv_blob.download_as_bytes()))
    if df_ohlcv.index.tz is None:
        df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')

    df_micro = pd.read_parquet(io.BytesIO(micro_blob.download_as_bytes()))
    if df_micro.index.tz is None:
        df_micro.index = df_micro.index.tz_localize('UTC')

    df_master = df_ohlcv.join(df_micro, how='left')
    del df_ohlcv, df_micro
    _release_memory()

    df_master = heal_data(df_master)
    df_master = df_master[df_master.index.notna()].sort_index()

    buf = io.BytesIO()
    df_master.to_parquet(buf, engine='pyarrow')
    buf.seek(0)
    bucket.blob(master_path).upload_from_file(buf)

    tc_nz = (df_master['Trade_Count'] > 0).sum() if 'Trade_Count' in df_master.columns else 0
    print(f"  [{ticker:6s}] Master: {len(df_master):,} rows, TC>0: {tc_nz:,}", flush=True)

    row_count = len(df_master)
    del df_master
    _release_memory()
    return row_count


# ==============================================================================
# PER-TICKER PIPELINE
# ==============================================================================
def process_one_ticker(ticker, idx, total):
    print(f"\n{'─'*60}", flush=True)
    print(f"  [{idx}/{total}] {ticker}", flush=True)
    print(f"{'─'*60}", flush=True)

    # Phase 1: download
    downloaded, existing, empty, errors = download_missing_tq(ticker)

    # Decide if rebuild needed
    master_path = f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet"
    has_master = bucket.blob(master_path).exists()

    need_rebuild = (downloaded > 0) or (not has_master)

    if not need_rebuild:
        print(f"  [{ticker:6s}] Master exists + no new T&Q — skip rebuild", flush=True)
        return {
            'ticker': ticker, 'downloaded': downloaded, 'existing': existing,
            'empty': empty, 'errors': errors,
            'micro_rows': 0, 'master_rows': 0, 'action': 'skip_rebuild',
        }

    reason = "no master" if not has_master else f"+{downloaded} new T&Q"
    print(f"  [{ticker:6s}] Rebuilding ({reason})...", flush=True)

    # Phase 2: rebuild micro
    micro_rows = rebuild_micro(ticker)

    # Phase 3: rebuild master
    master_rows = 0
    if micro_rows > 0:
        master_rows = rebuild_master(ticker)

    return {
        'ticker': ticker, 'downloaded': downloaded, 'existing': existing,
        'empty': empty, 'errors': errors,
        'micro_rows': micro_rows, 'master_rows': master_rows, 'action': 'rebuilt',
    }


# ==============================================================================
# MAIN
# ==============================================================================
_t0 = time.time()

bdays = len(pd.bdate_range(HISTORY_START, FETCH_END))

print("=" * 60)
print("  BACKFILL T&Q — VM (remaining scoring assets)")
print(f"  {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print(f"  Tickers: {len(TARGET_TICKERS)}")
print(f"  Range: {HISTORY_START.strftime('%Y-%m-%d')} -> "
      f"{FETCH_END.strftime('%Y-%m-%d')} ({bdays} bdays)")
print(f"  Colab handling: {sorted(COLAB_TICKERS)}")
print("=" * 60)

results = []
for idx, ticker in enumerate(TARGET_TICKERS, 1):
    try:
        r = process_one_ticker(ticker, idx, len(TARGET_TICKERS))
        results.append(r)
    except Exception as e:
        print(f"  [{ticker:6s}] FATAL: {e}", flush=True)
        results.append({
            'ticker': ticker, 'downloaded': 0, 'existing': 0,
            'empty': 0, 'errors': 1,
            'micro_rows': 0, 'master_rows': 0, 'action': 'FAILED',
        })

# ── Summary ──
elapsed = time.time() - _t0
print(f"\n\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")

total_dl = sum(r['downloaded'] for r in results)
rebuilt  = [r for r in results if r['action'] == 'rebuilt']
skipped  = [r for r in results if r['action'] == 'skip_rebuild']
failed   = [r for r in results if r['action'] == 'FAILED']

print(f"  Total tickers: {len(results)}")
print(f"  New T&Q downloaded: {total_dl} days across all tickers")
print(f"  Rebuilt micro+master: {len(rebuilt)}")
print(f"  Skipped (already OK): {len(skipped)}")
print(f"  Failed: {len(failed)}")
print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

if rebuilt:
    print(f"\n  Rebuilt tickers:")
    print(f"  {'Ticker':<8} {'DL':>6} {'Micro':>12} {'Master':>12}")
    print(f"  {'─'*8} {'─'*6} {'─'*12} {'─'*12}")
    for r in rebuilt:
        print(f"  {r['ticker']:<8} {r['downloaded']:>6} "
              f"{r['micro_rows']:>12,} {r['master_rows']:>12,}")

if failed:
    print(f"\n  FAILED: {[r['ticker'] for r in failed]}")

print(f"\n  Next: run audit_master_coverage.py to verify coverage")
print("=" * 60)
