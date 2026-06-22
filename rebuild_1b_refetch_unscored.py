"""
rebuild_1b_refetch_unscored.py — Refetch T&Q (unscored tickers only)
=====================================================================
  1. AUDIT   — find missing business days in T&Q cache for non-scored tickers
  2. REFETCH — download missing dates from Polygon API

No micro rebuild needed — these tickers don't have microstructure files.
Run in parallel with rebuild_1a_refetch_scored.py.
Run AFTER market close (after 20:00 UTC / 1:00 PM Pacific).
"""

import os
import time
import pytz
import pandas as pd
from datetime import datetime, timedelta
from polygon import RESTClient
from google.cloud import storage as gcs_storage
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DRY_RUN = False

POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
BUCKET_NAME     = "time-series-bucket-ys"
GCS_RAW_DIR     = f"gs://{BUCKET_NAME}/Data/TradesQuotes"

SCORED_TICKERS = {
    'ACHR', 'ALAB', 'APLD', 'ASTS', 'BE',
    'BTBT', 'CIFR', 'CLSK', 'FCEL', 'IREN',
    'JOBY', 'KEEL', 'LUNR', 'OKLO', 'PL',
    'PLUG', 'RCAT', 'RDW', 'RIOT', 'RKLB',
    'SMCI', 'WULF',
}

CORRUPTION_START = "2026-03-01"

poly_client = RESTClient(POLYGON_API_KEY)
gcs_client  = gcs_storage.Client()
bucket      = gcs_client.bucket(BUCKET_NAME)

# ==============================================================================
# T&Q FETCH
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
# DISCOVER UNSCORED TICKERS
# ==============================================================================
blobs_iter = bucket.list_blobs(prefix='Data/TradesQuotes/', delimiter='/')
list(blobs_iter)
ALL_TICKERS = sorted([p.replace('Data/TradesQuotes/', '').rstrip('/') for p in blobs_iter.prefixes])
UNSCORED_TICKERS = [t for t in ALL_TICKERS if t not in SCORED_TICKERS]
print(f"Discovered {len(ALL_TICKERS)} total tickers, {len(UNSCORED_TICKERS)} unscored")

run_ts = datetime.now(pytz.utc)
if run_ts.hour < 20 and run_ts.weekday() < 5:
    print(f"\n⚠️  WARNING: Current time is {run_ts.strftime('%H:%M UTC')}.")
    print(f"   Market may still be open. Run after 20:00 UTC for complete data.\n")

# ==============================================================================
# STEP 1: AUDIT — Find missing business days
# ==============================================================================
print("=" * 60)
print("STEP 1: AUDIT — Finding missing T&Q dates since", CORRUPTION_START)
print("=" * 60)

today = pd.Timestamp.now(tz='UTC').normalize().tz_localize(None)
expected_dates = set(
    d.strftime('%Y-%m-%d')
    for d in pd.bdate_range(CORRUPTION_START, today, freq='B')
)

missing_by_ticker = {}
total_missing = 0

for ticker in UNSCORED_TICKERS:
    prefix = f"Data/TradesQuotes/{ticker}/{ticker}_trades_"
    blobs = list(bucket.list_blobs(prefix=prefix))
    existing = set()
    for b in blobs:
        date_part = b.name.split('_')[-1].replace('.parquet', '').replace('.csv', '')
        if date_part >= CORRUPTION_START:
            existing.add(date_part)

    missing = sorted(expected_dates - existing)
    if missing:
        missing_by_ticker[ticker] = missing
        total_missing += len(missing)

print(f"\nTotal missing date-ticker combos: {total_missing}")
print(f"Tickers with gaps: {len(missing_by_ticker)} / {len(UNSCORED_TICKERS)}")
print()

for ticker in sorted(missing_by_ticker.keys()):
    dates = missing_by_ticker[ticker]
    print(f"  {ticker:>6}: {len(dates)} missing  ({dates[0]} → {dates[-1]})")

if DRY_RUN:
    print("\n  DRY_RUN = True — set to False and re-run to refetch.")

# ==============================================================================
# STEP 2: REFETCH — Download missing dates from Polygon
# ==============================================================================
if not DRY_RUN:
    print(f"\n{'=' * 60}")
    print("STEP 2: REFETCH — Downloading missing T&Q from Polygon")
    print("=" * 60)

    fetched = 0
    api_errors = 0

    for ticker in sorted(missing_by_ticker.keys()):
        missing_dates = missing_by_ticker[ticker]
        clean_ticker = ticker.replace("I:", "").replace("X:", "")
        print(f"\n  {ticker}: {len(missing_dates)} dates to fetch")

        for date_str in missing_dates:
            day_start = pd.Timestamp(date_str, tz='UTC')
            day_end   = day_start + timedelta(days=1)

            t_uri = f"{GCS_RAW_DIR}/{clean_ticker}/{clean_ticker}_trades_{date_str}.parquet"
            q_uri = f"{GCS_RAW_DIR}/{clean_ticker}/{clean_ticker}_quotes_{date_str}.parquet"

            try:
                dt = fetch_polygon_trades(clean_ticker, day_start, day_end)
                dq = fetch_polygon_quotes(clean_ticker, day_start, day_end)

                if not dt.empty:
                    dt.to_parquet(t_uri, engine='pyarrow')
                if not dq.empty:
                    dq.to_parquet(q_uri, engine='pyarrow')

                if dt.empty and dq.empty:
                    pd.DataFrame({'timestamp': [], 'price': [], 'size': []}).to_parquet(t_uri, engine='pyarrow')
                    pd.DataFrame({'timestamp': [], 'ask_price': [], 'bid_price': []}).to_parquet(q_uri, engine='pyarrow')

                fetched += 1
                if fetched % 50 == 0:
                    print(f"    ... {fetched} dates fetched so far")

            except Exception as e:
                print(f"    ❌ {ticker} {date_str}: {e}")
                api_errors += 1

            time.sleep(0.5)

    print(f"\n  Fetched {fetched} dates, {api_errors} errors")

    print(f"\n{'=' * 60}")
    print("NOTEBOOK 1B COMPLETE — unscored tickers refetched")
    print("After 1A finishes, run rebuild_2_delete_downstream.py")
    print("=" * 60)
