# %% ── Bulk Incremental Update — Outdated Master Inference ────────────────────
#
# Auto-updates master_inference files that are behind today's date.
# Per-ticker pipeline:
#   1. OHLCV: download missing 1-min bars from Polygon aggs, append
#   2. Micro: download missing T&Q from Polygon, process, append to existing
#   3. Master: re-stitch OHLCV + micro
#
# Incremental — only downloads/processes data for dates NOT in existing files.
# Designed to run on Colab in PARALLEL with VM daily pipeline.
# Resumable: restart skips tickers already brought up to date.
#
# Usage:
#   Run as Colab cell, or:
#   python bulk_update_outdated.py              # all outdated tickers
#   python bulk_update_outdated.py BABA CLF     # specific tickers only

import os, io, time, gc, ctypes, sys, subprocess, warnings
import numpy as np, pandas as pd, pytz
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

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
    _IN_COLAB = True
    auth.authenticate_user()
except (ImportError, Exception):
    _IN_COLAB = False


# ==============================================================================
# CONFIG
# ==============================================================================
POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
BUCKET_NAME     = "time-series-bucket-ys"
GCS_RAW_DIR     = "Data/TradesQuotes"
GCS_MICRO_DIR   = "Data/Inference/Microstructure"
GCS_OHLCV_DIR   = "Data/Inference"
GCS_MASTER_DIR  = "Data/Inference_v2"

POLYGON_SLEEP   = 0.35
HISTORY_START   = pd.Timestamp("2022-01-01", tz='UTC')

# Skip tickers being processed by VM bulk_micro_all.py (or delisted/renamed)
SKIP_TICKERS = {
    'DELL', 'ZTG', 'INTC', 'WDC', 'SCKT', 'IRDM', 'CJMB', 'NCEW',
    'FLEX', 'CIEN', 'DFNS', 'MRVL', 'BW', 'APP', 'ATEX', 'FIX',
    'STX', 'DD', 'DOV', 'SWK', 'CENX', 'VLO', 'SBLK', 'SHOP',
    'META', 'AAOI', 'NUA', 'LITE', 'ORCL', 'ANET', 'SPOT', 'NFLX',
    'NRG', 'KGS', 'CVNA', 'CAVA', 'ANF', 'STRL', 'EME', 'COHR',
    'SMLR',          # delisted
    'BITF', 'MIGI',  # renamed to KEEL / BGDE
}

# Outdated tickers (from freshness scan on 2026-08-17).
# Sorted least-stale first so quick wins complete first.
# Edit as needed, or pass tickers on CLI.
OUTDATED_TICKERS = [
    # Aug 13 (3 days old)
    'BLZE', 'BNY', 'BUUU', 'MGRT', 'SIF', 'SNDK',
    # Aug 12 (4 days old)
    'COP', 'OKLO', 'ROMA',
    # Aug 11 (5 days old)
    'BKSY', 'CCJ', 'EQIX', 'FCX', 'HOOD', 'JOBY', 'LIN', 'MOD', 'MP',
    'MU', 'MVIS', 'NNDM', 'NVDA', 'PLUG', 'POWL', 'RCAT', 'RDW',
    'TSLA', 'VOYG', 'VRT',
    # Aug 10 (6 days old)
    'BGDE', 'BHP', 'DUK', 'GD', 'GE', 'LHX', 'LMT', 'NOC', 'RTX', 'SPIR',
    # Aug 5 (11 days old)
    'CRNG',
    # Jul 20 (27 days old)
    'ACMR', 'BEAM', 'CRDO', 'MRNA', 'NEOV', 'NNE', 'NTLA', 'TWST', 'UCTT',
    # Very stale
    'BABA', 'CLF', 'XLE',
]

gcs_client = gcs_storage.Client()
bucket     = gcs_client.bucket(BUCKET_NAME)
poly       = RESTClient(POLYGON_API_KEY)


# ==============================================================================
# GCS HELPERS
# ==============================================================================
def _download_blob(blob_path, local_path):
    os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
    bucket.blob(blob_path).download_to_filename(local_path)

def _upload_from_file(buf, blob_path):
    bucket.blob(blob_path).upload_from_file(buf)

def _blob_exists(blob_path):
    return bucket.blob(blob_path).exists()


# ==============================================================================
# SCHEMA NORMALIZATION (from bulk_micro_all.py)
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
# MICROSTRUCTURE ENGINE (from bulk_micro_all.py)
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
# HEAL (ffill OHLC, zero-fill volumes — no bfill)
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
# PHASE 1: UPDATE OHLCV (incremental from Polygon aggs)
# ==============================================================================
def fetch_polygon_aggs(ticker, start_dt, end_dt, chunk_months=1, sleep_secs=2):
    """Chunked Polygon aggs download — matches daily_ohlcv_1.py format."""
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
            for a in poly.list_aggs(ticker, 1, 'minute', chunk_start, chunk_end, limit=50000):
                all_aggs.append({
                    'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                    'Open': a.open, 'High': a.high, 'Low': a.low,
                    'Close': a.close, 'Volume': a.volume,
                })
        except Exception as e:
            print(f"      Chunk {chunk_start.date()}->{chunk_end.date()} failed: {e}. Retrying...")
            time.sleep(10)
            try:
                for a in poly.list_aggs(ticker, 1, 'minute', chunk_start, chunk_end, limit=50000):
                    all_aggs.append({
                        'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                        'Open': a.open, 'High': a.high, 'Low': a.low,
                        'Close': a.close, 'Volume': a.volume,
                    })
            except Exception as e2:
                print(f"      Chunk skipped: {e2}")

        if total_days > 30:
            print(f"      {chunk_start.date()}->{chunk_end.date()}: {len(all_aggs):,} bars")
        chunk_start = chunk_end
        if chunk_start < end_dt:
            time.sleep(sleep_secs)

    if not all_aggs:
        return pd.DataFrame()
    df = pd.DataFrame(all_aggs).set_index('timestamp')
    return df[~df.index.duplicated(keep='last')].sort_index()


def update_ohlcv(ticker):
    """Download missing 1-min OHLCV from Polygon aggs, append to existing file."""
    ohlcv_blob = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
    local_path = f"/tmp/_upd_{ticker}_ohlcv.parquet"
    fetch_end  = pd.Timestamp.now(tz='UTC')

    if _blob_exists(ohlcv_blob):
        _download_blob(ohlcv_blob, local_path)
        df_existing = pd.read_parquet(local_path)
        os.remove(local_path)
        if df_existing.index.tz is None:
            df_existing.index = df_existing.index.tz_localize('UTC')
        ohlcv_start = df_existing.index.max() + timedelta(minutes=1)
        print(f"  [{ticker:6s}] OHLCV last: {df_existing.index.max().strftime('%Y-%m-%d %H:%M')}")
    else:
        print(f"  [{ticker:6s}] No OHLCV — fetching full history")
        df_existing = pd.DataFrame()
        ohlcv_start = HISTORY_START

    if ohlcv_start >= fetch_end:
        print(f"  [{ticker:6s}] OHLCV up to date")
        del df_existing; _release_memory()
        return 0

    print(f"  [{ticker:6s}] OHLCV fetching {ohlcv_start.date()} -> {fetch_end.date()}...", end='', flush=True)
    new_aggs = fetch_polygon_aggs(ticker, ohlcv_start, fetch_end)

    if new_aggs.empty:
        print(f" 0 bars")
        del df_existing; _release_memory()
        return 0

    if not df_existing.empty:
        df_combined = pd.concat([df_existing, new_aggs])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()
    else:
        df_combined = new_aggs

    local_out = f"/tmp/_upd_{ticker}_ohlcv_out.parquet"
    df_combined.to_parquet(local_out, engine='pyarrow')
    bucket.blob(ohlcv_blob).upload_from_filename(local_out)
    os.remove(local_out)

    n_new = len(new_aggs)
    print(f" +{n_new:,} bars (total: {len(df_combined):,})")

    del df_existing, new_aggs, df_combined
    _release_memory()
    return n_new


# ==============================================================================
# PHASE 2: UPDATE MICRO (incremental T&Q download + process + append)
# ==============================================================================
def update_micro(ticker):
    """Download missing T&Q from Polygon, process into micro, append to existing."""
    micro_blob = f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet"
    local_micro = f"/tmp/_upd_{ticker}_micro.parquet"
    fetch_end = pd.Timestamp.now(tz='UTC').normalize()

    # Load existing micro to find last date
    df_micro_existing = None
    if _blob_exists(micro_blob):
        _download_blob(micro_blob, local_micro)
        df_micro_existing = pd.read_parquet(local_micro)
        os.remove(local_micro)
        if df_micro_existing.index.tz is None:
            df_micro_existing.index = df_micro_existing.index.tz_localize('UTC')
        last_micro_date = df_micro_existing.index.max()
        print(f"  [{ticker:6s}] Micro last: {last_micro_date.strftime('%Y-%m-%d %H:%M')}")
        start_date = last_micro_date.normalize()  # re-process last day for safety
    else:
        print(f"  [{ticker:6s}] No existing micro — full T&Q download needed")
        start_date = HISTORY_START
        last_micro_date = HISTORY_START

    bdays = pd.bdate_range(start_date, fetch_end)
    if len(bdays) == 0:
        print(f"  [{ticker:6s}] Micro up to date")
        del df_micro_existing; _release_memory()
        return 0

    dates_needed = [d.strftime('%Y-%m-%d') for d in bdays]

    # Check which dates already have raw T&Q cached in GCS
    prefix = f"{GCS_RAW_DIR}/{ticker}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    cached_trade_dates = set()
    cached_quote_dates = set()
    for b in blobs:
        fname = os.path.basename(b.name)
        if fname.startswith(f"{ticker}_trades_"):
            d = fname.replace(f"{ticker}_trades_", '').replace('.parquet', '')
            cached_trade_dates.add(d)
        elif fname.startswith(f"{ticker}_quotes_"):
            d = fname.replace(f"{ticker}_quotes_", '').replace('.parquet', '')
            cached_quote_dates.add(d)
    cached_both = cached_trade_dates & cached_quote_dates

    dates_to_download = [d for d in dates_needed if d not in cached_both]
    n_cached = len(dates_needed) - len(dates_to_download)
    print(f"  [{ticker:6s}] Micro: {len(dates_needed)} bdays — "
          f"{n_cached} cached, {len(dates_to_download)} to download")

    # Download missing T&Q from Polygon
    downloaded = 0
    dl_errors  = 0
    for date_str in dates_to_download:
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
            dl_errors += 1
            if dl_errors <= 3:
                print(f"  [{ticker:6s}]   DL err {date_str}: {e}")

    if downloaded > 0:
        print(f"  [{ticker:6s}] Downloaded +{downloaded} T&Q days"
              f"{f' ({dl_errors} errors)' if dl_errors else ''}")

    # Process dates in range into micro rows
    micro_chunks = []
    proc_errors  = 0
    for i, date_str in enumerate(dates_needed):
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
            del dt, dq; gc.collect()

            if not day_micro.empty:
                micro_chunks.append(day_micro)
        except Exception as e:
            proc_errors += 1
            if proc_errors <= 3:
                print(f"  [{ticker:6s}]   Proc err {date_str}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  [{ticker:6s}]   {i+1}/{len(dates_needed)} days...", flush=True)

    if not micro_chunks:
        print(f"  [{ticker:6s}] No new micro data from T&Q")
        del df_micro_existing; _release_memory()
        return 0

    df_new_micro = pd.concat(micro_chunks).sort_index()
    df_new_micro = df_new_micro[~df_new_micro.index.duplicated(keep='last')]
    del micro_chunks; gc.collect()

    # Append: keep existing head + new tail
    if df_micro_existing is not None and not df_micro_existing.empty:
        df_head = df_micro_existing[df_micro_existing.index < start_date]
        df_combined = pd.concat([df_head, df_new_micro]).sort_index()
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        del df_head
    else:
        df_combined = df_new_micro

    buf = io.BytesIO()
    df_combined.to_parquet(buf, engine='pyarrow')
    buf.seek(0)
    _upload_from_file(buf, micro_blob)

    n_new = len(df_new_micro)
    print(f"  [{ticker:6s}] Micro updated: +{n_new:,} rows -> "
          f"{len(df_combined):,} total "
          f"({df_combined.index.max().strftime('%Y-%m-%d %H:%M')})")

    del df_micro_existing, df_new_micro, df_combined
    _release_memory()
    return n_new


# ==============================================================================
# PHASE 3: REBUILD MASTER (OHLCV + MICRO -> master_inference)
# ==============================================================================
def rebuild_master(ticker):
    """Join updated OHLCV + micro into master_inference (same as bulk_micro_all.py)."""
    ohlcv_blob  = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
    micro_blob  = f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet"
    master_blob = f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet"

    if not _blob_exists(ohlcv_blob):
        print(f"  [{ticker:6s}] No OHLCV — skip master rebuild")
        return 0
    if not _blob_exists(micro_blob):
        print(f"  [{ticker:6s}] No micro — skip master rebuild")
        return 0

    df_ohlcv = pd.read_parquet(io.BytesIO(bucket.blob(ohlcv_blob).download_as_bytes()))
    if df_ohlcv.index.tz is None:
        df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')

    df_micro = pd.read_parquet(io.BytesIO(bucket.blob(micro_blob).download_as_bytes()))
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
    _upload_from_file(buf, master_blob)

    tc_nz = (df_master['Trade_Count'] > 0).sum() if 'Trade_Count' in df_master.columns else 0
    print(f"  [{ticker:6s}] Master: {len(df_master):,} rows, "
          f"TC>0: {tc_nz:,}, "
          f"last: {df_master.index.max().strftime('%Y-%m-%d %H:%M')}")

    n = len(df_master)
    del df_master
    _release_memory()
    return n


# ==============================================================================
# PER-TICKER PIPELINE
# ==============================================================================
def process_one_ticker(ticker, idx, total):
    print(f"\n{'─'*60}", flush=True)
    print(f"  [{idx}/{total}] {ticker}", flush=True)
    print(f"{'─'*60}", flush=True)
    t0 = time.time()

    n_ohlcv = update_ohlcv(ticker)
    n_micro = update_micro(ticker)

    n_master = 0
    if n_ohlcv > 0 or n_micro > 0:
        n_master = rebuild_master(ticker)
    else:
        print(f"  [{ticker:6s}] No new data — master unchanged")

    elapsed = time.time() - t0
    print(f"  [{ticker:6s}] Done in {elapsed:.0f}s", flush=True)
    return {
        'ticker': ticker, 'ohlcv_new': n_ohlcv, 'micro_new': n_micro,
        'master_rows': n_master, 'elapsed': elapsed, 'status': 'OK',
    }


# ==============================================================================
# MAIN
# ==============================================================================
_t0 = time.time()

cli_args = [a.upper() for a in sys.argv[1:]
            if not a.startswith('-')
            and not a.endswith('.py')
            and not a.endswith('.json')
            and '/' not in a
            and len(a) <= 6
            and a.isalpha()]
if cli_args:
    TARGET_TICKERS = [t for t in cli_args if t not in SKIP_TICKERS]
else:
    TARGET_TICKERS = [t for t in OUTDATED_TICKERS if t not in SKIP_TICKERS]

print("=" * 60)
print("  BULK INCREMENTAL UPDATE — OUTDATED MASTER INFERENCE")
print(f"  {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print(f"  Tickers: {len(TARGET_TICKERS)}")
print(f"  Skipped: {len(SKIP_TICKERS)} (VM running / delisted / renamed)")
print("=" * 60)

results = []
for idx, ticker in enumerate(TARGET_TICKERS, 1):
    try:
        r = process_one_ticker(ticker, idx, len(TARGET_TICKERS))
        results.append(r)
    except Exception as e:
        print(f"  [{ticker:6s}] FATAL: {e}", flush=True)
        results.append({
            'ticker': ticker, 'ohlcv_new': 0, 'micro_new': 0,
            'master_rows': 0, 'elapsed': 0, 'status': f'FAILED: {e}',
        })

# ── Summary ──
elapsed_total = time.time() - _t0

print(f"\n\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")

ok      = [r for r in results if r['status'] == 'OK']
updated = [r for r in ok if r['master_rows'] > 0]
skipped = [r for r in ok if r['master_rows'] == 0]
failed  = [r for r in results if r['status'] != 'OK']

print(f"  Total tickers:    {len(results)}")
print(f"  Updated (master): {len(updated)}")
print(f"  Unchanged:        {len(skipped)}")
print(f"  Failed:           {len(failed)}")
print(f"  Elapsed:          {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)")

if updated:
    print(f"\n  Updated tickers:")
    print(f"  {'Ticker':<8} {'OHLCV+':>8} {'Micro+':>8} {'Master':>12} {'Time':>6}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*12} {'─'*6}")
    for r in updated:
        print(f"  {r['ticker']:<8} {r['ohlcv_new']:>8,} {r['micro_new']:>8,} "
              f"{r['master_rows']:>12,} {r['elapsed']:>5.0f}s")

if failed:
    print(f"\n  FAILED:")
    for r in failed:
        print(f"    {r['ticker']}: {r['status']}")

print(f"\n{'='*60}")
