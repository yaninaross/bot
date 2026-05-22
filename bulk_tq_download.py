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
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_KEY = '2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs'

BUCKET_NAME = "time-series-bucket-ys"

# Full history from 2022-01-01 needed for training
START_DATE = '2022-01-01'

# Number of parallel download threads per ticker
MAX_WORKERS = 10

gcs_client = gcs_storage.Client()
bucket     = gcs_client.bucket(BUCKET_NAME)

def list_existing_dates(clean_ticker):
    """List all dates that have BOTH trades + quotes in GCS (via API, not FUSE cache)."""
    prefix = f"Data/TradesQuotes/{clean_ticker}/{clean_ticker}_"
    blobs = set(b.name for b in bucket.list_blobs(prefix=prefix))
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
ALL_TQ_TICKERS = [
    # --- Exp-97 (Crypto Cluster) — train teachers ---
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
    'COIN', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'KEEL', 'BITF',
    # 'SMLR' — REMOVED: delisted ~Jan 2026
    # --- Exp-97 — additional scoring pool ---
    'SMCI', 'NVDA', 'VRT', 'AMD', 'TSLA',
    # SMH, TLT, IWM excluded — context-only assets (OHLCV only, no micro needed)
    # --- Exp-100 (Defense Sector) ---
    'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS', 'LUNR', 'SPIR',
    'JOBY', 'NNDM', 'MVIS', 'RCAT', 'ACHR',
    # --- Energy & AI Power ---
    'VST', 'CEG', 'BE', 'GEV', 'NEE', 'DUK', 'CCJ', 'XOM', 'CVX', 'COP',
    # --- Space & Dual-Use Defense ---
    'PL', 'RDW', 'BKSY', 'MNTS', 'VOYG', 'LMT', 'NOC', 'RTX',
    # --- Industrials & Infrastructure ---
    'CAT', 'FCX', 'BHP', 'ETN', 'LIN',
    # --- EXP-151/152 additions (energy expansion + cross-sector spikers) ---
    'OKLO', 'SMR', 'FCEL', 'FLNC', 'PLUG', 'MP', 'POWL', 'MOD', 'ALAB',
]

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
    blob = bucket.blob(gcs_path)
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
# MAIN
# ==============================================================================
if __name__ == '__main__':
    end_date = datetime.now().strftime('%Y-%m-%d')

    # CLI filter: specific tickers or all
    # Filter out Jupyter kernel args (e.g. -f kernel.json)
    cli_args = [a for a in sys.argv[1:] if not a.startswith('-') and not a.endswith('.json')]
    if cli_args:
        tickers = [t.upper() for t in cli_args]
        invalid = [t for t in tickers if t not in ALL_TQ_TICKERS]
        if invalid:
            print(f"   Warning: {invalid} not in standard ticker list, downloading anyway")
    else:
        tickers = list(dict.fromkeys(ALL_TQ_TICKERS))

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
        clean = ticker.replace("I:", "").replace("X:", "")

        t_start = time.time()
        print(f"[{i}/{total}] {ticker}  ({len(biz_dates)} days)")

        ticker_downloaded = 0
        ticker_skipped    = 0
        ticker_errors     = 0

        # Pre-scan via GCS API (FUSE cache can be stale)
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

            # First ticker: verify write actually persists to GCS
            if i == 1:
                test_date = dates_to_download[0]
                test_ds = test_date.strftime('%Y-%m-%d')
                download_one_day(clean, None, test_date)
                t_blob = bucket.blob(f"Data/TradesQuotes/{clean}/{clean}_trades_{test_ds}.parquet")
                q_blob = bucket.blob(f"Data/TradesQuotes/{clean}/{clean}_quotes_{test_ds}.parquet")
                if t_blob.exists() and q_blob.exists():
                    t_blob.reload()
                    q_blob.reload()
                    print(f"      VERIFY OK: {test_ds} confirmed in GCS ({t_blob.size:,} + {q_blob.size:,} bytes)")
                else:
                    print(f"      VERIFY FAILED: {test_ds} NOT found in GCS after write!")
                    print(f"      Check: service account permissions on bucket {BUCKET_NAME}")
                    sys.exit(1)
                dates_to_download = dates_to_download[1:]
                ticker_downloaded += 1

            if dates_to_download:
                done_count = 0
                total_to_do = len(dates_to_download)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {
                        executor.submit(download_one_day, clean, None, d): d
                        for d in dates_to_download
                    }

                    for future in as_completed(futures):
                        status, date_str, info = future.result()
                        done_count += 1
                        if status == 'holiday':
                            ticker_downloaded += 1
                        elif status == 'ok':
                            ticker_downloaded += 1
                            print(f"      [{done_count}/{total_to_do}] {date_str}  {info}")
                        elif status == 'error':
                            ticker_errors += 1
                            print(f"      [{done_count}/{total_to_do}] {date_str}  ERROR: {info}")

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
