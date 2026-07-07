"""
bulk_ohlcv.py — One-time OHLCV backfill for ALL assets + context
=================================================================
Bulk version of daily_ohlcv_1.py + daily_ohlcv_2.py merged into one script.
Fetches missing 1-min OHLCV bars from Polygon, appends to GCS parquets.

NOT part of the daily scoring pipeline — use for one-time data preparation.
Colab-ready (no if __name__).
"""

import time, gc, ctypes, sys, subprocess, warnings
import pandas as pd
from datetime import timedelta
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect(); gc.collect()
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
GCS_OHLCV_DIR   = f"gs://{BUCKET_NAME}/Data/Inference"

# All tickers from daily_ohlcv_1 + daily_ohlcv_2
TARGET_TICKERS = [
    # Batch 1 (A-K)
    'ACHR', 'ALAB', 'AMD', 'APLD', 'ASTS', 'AVAV', 'BBAI',
    'BE', 'BGDE', 'BHP', 'BITF', 'BKSY', 'BTBT',
    'CAT', 'CCJ', 'CEG', 'CIFR', 'CLSK', 'COIN', 'COP', 'CORZ', 'CVX',
    'DUK', 'EQIX', 'ETN', 'FCEL', 'FCX', 'FLNC',
    'GD', 'GE', 'GEV', 'HOOD', 'HUT', 'IREN', 'JOBY', 'KEEL', 'KTOS',
    # Batch 2 (L-X)
    'LHX', 'LIN', 'LMT', 'LUNR', 'MARA', 'MIGI', 'MNTS',
    'MOD', 'MP', 'MSTR', 'MU', 'MVIS', 'NBIS', 'NEE', 'NNDM', 'NOC', 'NVDA',
    'OKLO', 'PL', 'PLTR', 'PLUG', 'POWL',
    'RCAT', 'RDW', 'RIOT', 'RKLB', 'RTX',
    'SMCI', 'SMR', 'SOUN', 'SPIR',
    'TSLA', 'VOYG', 'VRT', 'VST', 'WULF', 'XOM',
]

CONTEXT_ASSETS = ['I:VIX', 'TLT', 'IWM', 'SPY']

poly_client = RESTClient(POLYGON_API_KEY)
gcs_client  = gcs_storage.Client()


# ==============================================================================
# POLYGON OHLCV FETCH
# ==============================================================================
def fetch_polygon_aggs(ticker, start_dt, end_dt, chunk_months=1, sleep_secs=2):
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
            for a in poly_client.list_aggs(ticker, 1, 'minute', chunk_start, chunk_end, limit=50000):
                all_aggs.append({'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                                 'Open': a.open, 'High': a.high, 'Low': a.low,
                                 'Close': a.close, 'Volume': a.volume})
        except Exception as e:
            print(f"      Chunk {chunk_start.date()} -> {chunk_end.date()} failed: {e}. Retrying...")
            time.sleep(10)
            try:
                for a in poly_client.list_aggs(ticker, 1, 'minute', chunk_start, chunk_end, limit=50000):
                    all_aggs.append({'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                                     'Open': a.open, 'High': a.high, 'Low': a.low,
                                     'Close': a.close, 'Volume': a.volume})
            except Exception as e2:
                print(f"      Chunk skipped: {e2}")

        if total_days > 30:
            print(f"      {chunk_start.date()} -> {chunk_end.date()}: {len(all_aggs):,} bars")
        chunk_start = chunk_end
        if chunk_start < end_dt:
            time.sleep(sleep_secs)

    if not all_aggs:
        return pd.DataFrame()
    df = pd.DataFrame(all_aggs).set_index('timestamp')
    return df[~df.index.duplicated(keep='last')].sort_index()


# ==============================================================================
# MAIN
# ==============================================================================
print("=" * 70)
print("  BULK OHLCV BACKFILL — All assets + context")
print("=" * 70)

fetch_end = pd.Timestamp.now(tz='UTC')
print(f"  Run time: {fetch_end.strftime('%Y-%m-%d %H:%M UTC')}")
print(f"  Assets:  {len(TARGET_TICKERS)}")
print(f"  Context: {len(CONTEXT_ASSETS)}")

ALL_ASSETS = (
    [(t, True) for t in CONTEXT_ASSETS] +
    [(t, False) for t in sorted(set(TARGET_TICKERS))]
)

summary = {'updated': [], 'up_to_date': [], 'failed': []}

for ai, (ticker, is_context) in enumerate(ALL_ASSETS, 1):
    print(f"\n{'─'*52}")
    tag = "[CONTEXT]" if is_context else ""
    print(f"[{ai}/{len(ALL_ASSETS)}] {ticker} {tag}")
    clean_ticker = ticker.replace("I:", "").replace("X:", "")

    ohlcv_uri = f"{GCS_OHLCV_DIR}/{clean_ticker}_1m_full_historical_data.parquet"

    try:
        df_ohlcv = pd.read_parquet(ohlcv_uri)
        if df_ohlcv.index.tz is None:
            df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
        ohlcv_start = df_ohlcv.index.max() + timedelta(minutes=1)
        print(f"   Existing: {len(df_ohlcv):,} rows, last: {df_ohlcv.index.max()}")
    except Exception:
        print(f"   No OHLCV found — fetching from 2022-01-01")
        df_ohlcv    = pd.DataFrame()
        ohlcv_start = pd.Timestamp("2022-01-01", tz='UTC')

    if ohlcv_start >= fetch_end:
        print(f"   Already up to date")
        summary['up_to_date'].append(clean_ticker)
        del df_ohlcv
        _release_memory()
        continue

    print(f"   Fetching from {ohlcv_start.date()}...")
    try:
        new_aggs = fetch_polygon_aggs(ticker, ohlcv_start, fetch_end)
        if not new_aggs.empty:
            df_ohlcv = pd.concat([df_ohlcv, new_aggs])
            df_ohlcv = df_ohlcv[~df_ohlcv.index.duplicated(keep='last')].sort_index()
            df_ohlcv.to_parquet(ohlcv_uri, engine='pyarrow')
            print(f"   +{len(new_aggs):,} bars -> total: {len(df_ohlcv):,}")
            summary['updated'].append(clean_ticker)
        else:
            print(f"   No new bars returned")
            summary['up_to_date'].append(clean_ticker)
    except Exception as e:
        print(f"   FAILED: {e}")
        summary['failed'].append(clean_ticker)

    del df_ohlcv
    _release_memory()


# ==============================================================================
# SUMMARY
# ==============================================================================
print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  Updated:     {len(summary['updated']):>3}")
print(f"  Up to date:  {len(summary['up_to_date']):>3}")
print(f"  Failed:      {len(summary['failed']):>3}  {summary['failed']}")
print()
print("  Next: run bulk_stitch_master.py")
print("=" * 70)
