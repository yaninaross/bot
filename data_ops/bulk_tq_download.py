"""
bulk_tq_download.py — Bulk Raw Trades & Quotes Downloader
==========================================================
Downloads raw T&Q data from Polygon and saves to GCS as daily parquet files.
Does NOT compute microstructure — that's done by daily_micro_all.py.

Uses parallel threads to download multiple days concurrently (I/O bound).

Fully idempotent: skips dates that already have both trades + quotes in GCS.

Usage:
  python bulk_tq_download.py                  # all tickers
  python bulk_tq_download.py MARA KEEL ASTS   # specific tickers only
"""

import sys
import os
import io
import time
import threading
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from polygon import RESTClient
from google.cloud import storage as gcs_storage
import gc
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_KEY = '2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs'

BUCKET_NAME = "time-series-bucket-ys"

# Full history from 2022-01-01 needed for training
START_DATE = '2022-01-01'

# Number of parallel download threads per ticker (keep low to avoid OOM on 16GB VMs)
MAX_WORKERS = 4

# Process this many days per batch to limit memory (avoids submitting 1000+ futures at once)
BATCH_SIZE = 50

def _fresh_bucket():
    """Create a fresh GCS client+bucket to avoid stale auth tokens on long runs."""
    return gcs_storage.Client().bucket(BUCKET_NAME)

_bucket = [_fresh_bucket()]
def _get_bucket():
    return _bucket[0]
def _refresh_bucket():
    _bucket[0] = _fresh_bucket()

def list_existing_dates(clean_ticker):
    """List all dates that have BOTH trades + quotes in GCS (via API, not FUSE cache)."""
    prefix = f"Data/TradesQuotes/{clean_ticker}/{clean_ticker}_"
    blobs = set(b.name for b in _get_bucket().list_blobs(prefix=prefix))
    trade_dates = set()
    quote_dates = set()
    for name in blobs:
        base = os.path.basename(name)
        if base.startswith(f"{clean_ticker}_trades_") and base.endswith('.parquet'):
            date_str = base.replace(f"{clean_ticker}_trades_", "").replace(".parquet", "")
            trade_dates.add(date_str)
        elif base.startswith(f"{clean_ticker}_quotes_") and base.endswith('.parquet'):
            date_str = base.replace(f"{clean_ticker}_quotes_", "").replace(".parquet", "")
            quote_dates.add(date_str)
    return trade_dates & quote_dates  # only dates with BOTH files

# ==============================================================================
# ASSET LISTS
# ==============================================================================
def discover_all_tq_tickers():
    """Auto-discover ALL tickers that have TradesQuotes folders on GCS."""
    prefix = 'Data/TradesQuotes/'
    b = _get_bucket()
    iterator = b.list_blobs(prefix=prefix, delimiter='/')
    for _ in iterator:
        pass  # must consume to populate .prefixes
    tickers = []
    for p in iterator.prefixes:
        ticker = p.replace(prefix, '').rstrip('/')
        if ticker:
            tickers.append(ticker)
    return sorted(tickers)

HARDCODED_TQ_TICKERS = [
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
    'COIN', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'KEEL', 'BITF',
    'SMCI', 'NVDA', 'VRT', 'AMD', 'TSLA',
    'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS', 'LUNR', 'SPIR',
    'JOBY', 'NNDM', 'MVIS', 'RCAT', 'ACHR',
    'VST', 'CEG', 'BE', 'GEV', 'NEE', 'DUK', 'CCJ', 'XOM', 'CVX', 'COP',
    'PL', 'RDW', 'BKSY', 'MNTS', 'VOYG', 'LMT', 'NOC', 'RTX',
    'CAT', 'FCX', 'BHP', 'ETN', 'LIN',
    'OKLO', 'SMR', 'FCEL', 'FLNC', 'PLUG', 'MP', 'POWL', 'MOD', 'ALAB',
    'AEHR', 'CRDO', 'MXL', 'WOLF', 'NEOV', 'NNE', 'ONDS', 'SLS', 'UCTT',
    'NBIS', 'SOUN', 'GD', 'LHX', 'GE', 'EQIX', 'MU', 'MRNA', 'NTLA', 'TWST', 'BEAM',
]

print("  Discovering all tickers from GCS TradesQuotes...")
ALL_TQ_TICKERS = discover_all_tq_tickers()
if not ALL_TQ_TICKERS:
    print("  GCS discovery returned 0 — falling back to hardcoded list")
    ALL_TQ_TICKERS = list(dict.fromkeys(HARDCODED_TQ_TICKERS))
else:
    extra = set(ALL_TQ_TICKERS) - set(HARDCODED_TQ_TICKERS)
    if extra:
        print(f"  Found {len(ALL_TQ_TICKERS)} tickers on GCS "
              f"({len(extra)} beyond hardcoded list: {sorted(extra)[:10]}...)")
    else:
        print(f"  Found {len(ALL_TQ_TICKERS)} tickers on GCS")

# Thread-local Polygon clients (RESTClient is not thread-safe)
_thread_local = threading.local()

def _get_client():
    if not hasattr(_thread_local, 'client'):
        _thread_local.client = RESTClient(API_KEY)
    return _thread_local.client

# ==============================================================================
# POLYGON T&Q FETCH
# ==============================================================================
def fetch_polygon_trades(ticker, start_dt, end_dt):
    client = _get_client()
    trades = []
    for t in client.list_trades(ticker, timestamp_gte=start_dt, timestamp_lte=end_dt, limit=50000):
        trades.append({'timestamp': pd.to_datetime(t.sip_timestamp, unit='ns', utc=True),
                       'price': t.price, 'size': t.size})
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)

def fetch_polygon_quotes(ticker, start_dt, end_dt):
    client = _get_client()
    quotes = []
    for q in client.list_quotes(ticker, timestamp_gte=start_dt, timestamp_lte=end_dt, limit=50000):
        quotes.append({'timestamp': pd.to_datetime(q.sip_timestamp, unit='ns', utc=True),
                       'ask_price': q.ask_price, 'ask_size': q.ask_size,
                       'bid_price': q.bid_price, 'bid_size': q.bid_size})
    if not quotes:
        return pd.DataFrame()
    return pd.DataFrame(quotes)

# ==============================================================================
# SINGLE DAY DOWNLOAD (called per thread)
# ==============================================================================
def _upload_df_to_gcs(df, gcs_path, compression='snappy'):
    """Serialize DataFrame to parquet in memory, upload via GCS API (no gcsfs needed)."""
    buf = io.BytesIO()
    if df.empty:
        pd.DataFrame().to_parquet(buf, engine='pyarrow')
    else:
        df.to_parquet(buf, engine='pyarrow', compression=compression)
    buf.seek(0)
    blob = _get_bucket().blob(gcs_path)
    blob.upload_from_file(buf, content_type='application/octet-stream')

def download_one_day(clean, ticker_dir_unused, gap_date):
    """Download trades + quotes for one ticker-day. Returns (status, date_str, info).
    Writes via GCS API (not FUSE mount or gs:// URIs) for reliable persistence."""
    date_str = gap_date.strftime('%Y-%m-%d')
    t_path = f"Data/TradesQuotes/{clean}/{clean}_trades_{date_str}.parquet"
    q_path = f"Data/TradesQuotes/{clean}/{clean}_quotes_{date_str}.parquet"

    day_start = pd.Timestamp(gap_date, tz='UTC')
    day_end   = day_start + pd.Timedelta(days=1)

    try:
        dt = fetch_polygon_trades(clean, day_start, day_end)
        dq = fetch_polygon_quotes(clean, day_start, day_end)

        _upload_df_to_gcs(dt, t_path)
        _upload_df_to_gcs(dq, q_path)

        if dt.empty and dq.empty:
            return ('holiday', date_str, '')
        else:
            t_rows = len(dt)
            q_rows = len(dq)
            return ('ok', date_str, f"trades:{t_rows:,}  quotes:{q_rows:,}")

    except Exception as e:
        return ('error', date_str, str(e))

# ==============================================================================
# CONFIG: TICKER FILTER
# ==============================================================================
# Set to a list of tickers to process only those, or None to auto-detect stale.
# Auto-detect scans all tickers and finds those missing recent business days.
ONLY_TICKERS = None

# Auto-detect: a ticker is "stale" if it's missing more than this many recent biz days
STALE_THRESHOLD_DAYS = 3

# ==============================================================================
# STALE TICKER DETECTION
# ==============================================================================
def find_stale_tickers(ticker_list, threshold=STALE_THRESHOLD_DAYS):
    """Scan all tickers, return stale ones sorted by freshness (most recent first)."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    recent_biz = pd.bdate_range(end=end_date, periods=threshold + 2, freq='B')
    recent_strs = {d.strftime('%Y-%m-%d') for d in recent_biz}

    stale = []
    up_to_date = []

    print(f"  Scanning {len(ticker_list)} tickers for staleness "
          f"(missing >{threshold} of last {len(recent_strs)} biz days)...")

    for ticker in ticker_list:
        clean = ticker.replace("I:", "").replace("X:", "")
        existing = list_existing_dates(clean)
        missing = recent_strs - existing
        latest = max(existing) if existing else '1970-01-01'
        if len(missing) > threshold:
            stale.append((ticker, len(missing), latest))
        else:
            up_to_date.append(ticker)

    # Sort by latest date DESCENDING — most fresh first (smallest gap, fastest to finish)
    stale.sort(key=lambda x: x[2], reverse=True)

    print(f"  Result: {len(stale)} stale, {len(up_to_date)} up-to-date")
    if stale:
        print(f"\n  {'#':<4} {'Ticker':<8} {'Missing':>8} {'Latest':>12}")
        print(f"  {'─'*34}")
        for i, (t, miss, latest) in enumerate(stale, 1):
            print(f"  {i:<4} {t:<8} {miss:>8} {latest:>12}")
    print()

    return [t for t, _, _ in stale]


# ==============================================================================
# MAIN
# ==============================================================================
end_date = datetime.now().strftime('%Y-%m-%d')

if ONLY_TICKERS:
    tickers = ONLY_TICKERS
else:
    tickers = find_stale_tickers(list(dict.fromkeys(ALL_TQ_TICKERS)))

if not tickers:
    print("  All tickers up to date — nothing to download.")
else:
    total = len(tickers)
    biz_dates = pd.date_range(START_DATE, end_date, freq='B')

    print(f"{'='*60}")
    print(f"  BULK T&Q DOWNLOAD — {total} tickers")
    print(f"  Range: {START_DATE} → {end_date}  ({len(biz_dates)} biz days)")
    print(f"  Workers: {MAX_WORKERS} parallel threads per ticker")
    print(f"  Output: gs://{BUCKET_NAME}/Data/TradesQuotes/")
    print(f"{'='*60}\n")

    total_downloaded = 0
    total_skipped    = 0
    total_errors     = 0

    for i, ticker in enumerate(tickers, 1):
        _refresh_bucket()

        clean = ticker.replace("I:", "").replace("X:", "")

        t_start = time.time()
        print(f"[{i}/{total}] {ticker}  ({len(biz_dates)} days)")

        ticker_downloaded = 0
        ticker_skipped    = 0
        ticker_errors     = 0

        existing_dates = list_existing_dates(clean)
        dates_to_download = []
        for d in biz_dates:
            if d.strftime('%Y-%m-%d') in existing_dates:
                ticker_skipped += 1
            else:
                dates_to_download.append(d)

        if not dates_to_download:
            print(f"      All {len(biz_dates)} days cached — skipping")
        else:
            print(f"      {ticker_skipped} cached, {len(dates_to_download)} to download...")

            if i == 1:
                test_date = dates_to_download[0]
                test_ds = test_date.strftime('%Y-%m-%d')
                download_one_day(clean, None, test_date)
                t_blob = _get_bucket().blob(f"Data/TradesQuotes/{clean}/{clean}_trades_{test_ds}.parquet")
                q_blob = _get_bucket().blob(f"Data/TradesQuotes/{clean}/{clean}_quotes_{test_ds}.parquet")
                if t_blob.exists() and q_blob.exists():
                    t_blob.reload()
                    q_blob.reload()
                    print(f"      VERIFY OK: {test_ds} confirmed in GCS "
                          f"({t_blob.size:,} + {q_blob.size:,} bytes)")
                else:
                    print(f"      VERIFY FAILED: {test_ds} NOT found in GCS after write!")
                    print(f"      Check: service account permissions on bucket {BUCKET_NAME}")
                    raise RuntimeError("GCS write verification failed")
                dates_to_download = dates_to_download[1:]
                ticker_downloaded += 1

            if dates_to_download:
                done_count = 0
                total_to_do = len(dates_to_download)

                for batch_start in range(0, total_to_do, BATCH_SIZE):
                    batch = dates_to_download[batch_start:batch_start + BATCH_SIZE]
                    _refresh_bucket()

                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = {
                            executor.submit(download_one_day, clean, None, d): d
                            for d in batch
                        }

                        for future in as_completed(futures):
                            status, date_str, info = future.result()
                            done_count += 1
                            if status == 'holiday':
                                ticker_downloaded += 1
                            elif status == 'ok':
                                ticker_downloaded += 1
                                if done_count % 10 == 0 or done_count == total_to_do:
                                    print(f"      [{done_count}/{total_to_do}] "
                                          f"{date_str}  {info}")
                            elif status == 'error':
                                ticker_errors += 1
                                print(f"      [{done_count}/{total_to_do}] "
                                      f"{date_str}  ERROR: {info}")

                    gc.collect()

        elapsed = time.time() - t_start
        print(f"      Done: {ticker_downloaded} downloaded, {ticker_skipped} cached, "
              f"{ticker_errors} errors  ({elapsed:.0f}s)")

        total_downloaded += ticker_downloaded
        total_skipped    += ticker_skipped
        total_errors     += ticker_errors

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"  Downloaded: {total_downloaded} ticker-days")
    print(f"  Skipped:    {total_skipped} ticker-days (already cached)")
    print(f"  Errors:     {total_errors}")
    print(f"{'='*60}")
