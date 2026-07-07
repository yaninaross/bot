"""
backfill_micro_parallel.py — Parallel T&Q backfill for this VM
===============================================================
Context assets (SPY/VIX/TLT/IWM) first, then remaining assets in
REVERSE alphabetical order (Z→A) so it doesn't collide with
backfill_micro_june.py running in Colab (A→Z).

Usage:  python3 backfill_micro_parallel.py
"""

import os, time, gc, ctypes, json, sys, subprocess, warnings
import numpy as np, pandas as pd, pytz
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

GCS_MICRO_DIR = f"gs://{BUCKET_NAME}/Data/Inference/Microstructure"
GCS_RAW_DIR   = f"gs://{BUCKET_NAME}/Data/TradesQuotes"

FETCH_END = pd.Timestamp("2026-07-01", tz='UTC')
HISTORY_START = "2025-09-01"

MAX_WORKERS = 4
POLYGON_SLEEP = 0.3

ALL_TICKERS = [
    'ACHR', 'ALAB', 'AMD', 'APLD', 'ASTS', 'AVAV', 'BBAI',
    'BE', 'BGDE', 'BHP', 'BKSY', 'BTBT',
    'CAT', 'CCJ', 'CEG', 'CIFR', 'CLSK', 'COIN', 'COP', 'CORZ', 'CVX',
    'DUK', 'EQIX', 'ETN', 'FCEL', 'FCX', 'FLNC',
    'GD', 'GE', 'GEV', 'HOOD', 'HUT', 'IREN',
    'JOBY', 'KEEL', 'KTOS',
    'LHX', 'LIN', 'LMT', 'LUNR',
    'MARA', 'MNTS', 'MOD', 'MP', 'MSTR', 'MU', 'MVIS',
    'NBIS', 'NEE', 'NNDM', 'NOC', 'NVDA',
    'OKLO', 'PL', 'PLTR', 'PLUG', 'POWL',
    'RCAT', 'RDW', 'RIOT', 'RKLB', 'RTX',
    'SMCI', 'SMR', 'SOUN', 'SPIR',
    'TSLA', 'VOYG', 'VRT', 'VST', 'WULF', 'XOM',
]

# Context assets are OHLCV-only in master_inference (no micro needed)
CONTEXT_OHLCV_ONLY = {'SPY', 'VIX', 'TLT', 'IWM'}
SKIP_MICRO = {'ITA', 'XLE'}
SKIP_DEAD  = {'SMLR', 'BITF', 'MIGI'}
POLYGON_TICKER_MAP = {'VIX': 'I:VIX'}

print_lock = threading.Lock()
api_lock   = threading.Lock()

gcs_client = gcs_storage.Client()


# ==============================================================================
# HELPERS
# ==============================================================================
def _log(ticker, msg):
    with print_lock:
        print(f"  [{ticker:6s}] {msg}", flush=True)

def _blob_path(gcs_uri):
    return gcs_uri.replace(f"gs://{BUCKET_NAME}/", "")

def _gcs_exists(gcs_uri):
    return gcs_client.bucket(BUCKET_NAME).blob(_blob_path(gcs_uri)).exists()

def normalize_raw_schema(df, kind):
    if df is None or df.empty:
        return df
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

def _valid_raw_file(fname, clean_ticker, kind):
    return os.path.basename(fname).startswith(f"{clean_ticker}_{kind}_")


# ==============================================================================
# POLYGON FETCH (serialized through api_lock to respect rate limits)
# ==============================================================================
def fetch_polygon_trades(ticker, start_dt, end_dt):
    poly_ticker = POLYGON_TICKER_MAP.get(ticker, ticker)
    poly = RESTClient(POLYGON_API_KEY)
    trades = []
    try:
        for t in poly.list_trades(poly_ticker,
                timestamp_gte=start_dt, timestamp_lte=end_dt, limit=50000):
            trades.append({'timestamp': pd.to_datetime(t.sip_timestamp, unit='ns', utc=True),
                           'price': t.price, 'size': t.size})
    except Exception as e:
        _log(ticker, f"API trades error: {e}")
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)

def fetch_polygon_quotes(ticker, start_dt, end_dt):
    poly_ticker = POLYGON_TICKER_MAP.get(ticker, ticker)
    poly = RESTClient(POLYGON_API_KEY)
    quotes = []
    try:
        for q in poly.list_quotes(poly_ticker,
                timestamp_gte=start_dt, timestamp_lte=end_dt, limit=50000):
            quotes.append({'timestamp': pd.to_datetime(q.sip_timestamp, unit='ns', utc=True),
                           'ask_price': q.ask_price, 'ask_size': q.ask_size,
                           'bid_price': q.bid_price, 'bid_size': q.bid_size})
    except Exception as e:
        _log(ticker, f"API quotes error: {e}")
    if not quotes:
        return pd.DataFrame()
    return pd.DataFrame(quotes)


# ==============================================================================
# MICROSTRUCTURE ENGINE
# ==============================================================================
def process_raw_to_microstructure(trades_df, quotes_df):
    if trades_df is not None and not trades_df.empty:
        trades_df = trades_df.set_index('timestamp')
        trades_df['Direction']    = np.sign(trades_df['price'].diff())
        trades_df['Direction']    = trades_df['Direction'].mask(trades_df['Direction'] == 0).ffill().fillna(0)
        trades_df['Volume_Delta'] = trades_df['size'] * trades_df['Direction']
        t_1m = trades_df.resample('1min').agg(
            {'size': 'sum', 'Volume_Delta': 'sum', 'price': 'count'}
        ).rename(columns={'size': 'Total_Vol', 'price': 'Trade_Count'}).fillna(0)
    else:
        t_1m = pd.DataFrame(columns=['Total_Vol', 'Volume_Delta', 'Trade_Count'])

    if quotes_df is not None and not quotes_df.empty:
        quotes_df = quotes_df.set_index('timestamp')
        quotes_df['Spread'] = quotes_df['ask_price'] - quotes_df['bid_price']
        quotes_df['OBI']    = (quotes_df['bid_size'] - quotes_df['ask_size']) / \
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
# PER-TICKER WORKER
# ==============================================================================
def process_ticker(ticker):
    clean_ticker = ticker
    micro_uri = f"{GCS_MICRO_DIR}/{clean_ticker}_1m_microstructure.parquet"

    try:
        df_micro = pd.read_parquet(micro_uri)
        if df_micro.index.tz is None:
            df_micro.index = df_micro.index.tz_localize('UTC')
        micro_start = df_micro.index.max() + timedelta(minutes=1)
        _log(ticker, f"existing {len(df_micro):,} rows, last: {df_micro.index.max()}")
    except Exception:
        df_micro = pd.DataFrame()
        micro_start = pd.Timestamp(HISTORY_START, tz='UTC')
        _log(ticker, f"no existing micro — from {HISTORY_START}")

    if micro_start >= FETCH_END:
        _log(ticker, "up to date")
        return ticker, 'up_to_date'

    gap_dates = pd.date_range(micro_start.date(), FETCH_END.date(), freq='B')
    micro_chunks = []
    days_disk, days_api, days_empty, days_err = 0, 0, 0, 0

    for gap_date in gap_dates:
        date_str = gap_date.strftime('%Y-%m-%d')
        t_fname = f"{clean_ticker}_trades_{date_str}.parquet"
        q_fname = f"{clean_ticker}_quotes_{date_str}.parquet"
        t_uri = f"{GCS_RAW_DIR}/{clean_ticker}/{t_fname}"
        q_uri = f"{GCS_RAW_DIR}/{clean_ticker}/{q_fname}"

        dt, dq = pd.DataFrame(), pd.DataFrame()

        cache_t = _gcs_exists(t_uri)
        cache_q = _gcs_exists(q_uri)

        if cache_t and cache_q:
            try:
                if _valid_raw_file(t_fname, clean_ticker, 'trades') and \
                   _valid_raw_file(q_fname, clean_ticker, 'quotes'):
                    dt = normalize_raw_schema(pd.read_parquet(t_uri), 'trades')
                    dq = normalize_raw_schema(pd.read_parquet(q_uri), 'quotes')
                    days_disk += 1
                else:
                    raise ValueError("misnamed")
            except Exception as e:
                days_err += 1
                continue
        else:
            day_start = pd.Timestamp(gap_date, tz='UTC')
            day_end = day_start + timedelta(days=1)
            with api_lock:
                dt = fetch_polygon_trades(clean_ticker, day_start, day_end)
                dq = fetch_polygon_quotes(clean_ticker, day_start, day_end)
                time.sleep(POLYGON_SLEEP)
            try:
                if not dt.empty:
                    dt.to_parquet(t_uri, engine='pyarrow')
                if not dq.empty:
                    dq.to_parquet(q_uri, engine='pyarrow')
                days_api += 1
            except Exception as e:
                days_err += 1
                continue

        if dt.empty and dq.empty:
            days_empty += 1
            continue

        if not dt.empty:
            dt = dt[dt['timestamp'] >= micro_start]
        if not dq.empty:
            dq = dq[dq['timestamp'] >= micro_start]

        if dt.empty and dq.empty:
            continue

        day_micro = process_raw_to_microstructure(dt, dq)
        del dt, dq
        if not day_micro.empty:
            micro_chunks.append(day_micro)

    _log(ticker, f"{days_disk} disk / {days_api} API / {days_empty} empty / {days_err} err")

    if not micro_chunks:
        _log(ticker, "no new data")
        return ticker, 'no_data'

    new_micro = pd.concat(micro_chunks).sort_index()
    del micro_chunks

    if not df_micro.empty:
        new_micro = new_micro[new_micro.index > df_micro.index.max()]

    if new_micro.empty:
        _log(ticker, "already covered")
        return ticker, 'up_to_date'

    df_micro = pd.concat([df_micro, new_micro]).sort_index()
    df_micro = df_micro[~df_micro.index.duplicated(keep='last')]
    df_micro.to_parquet(micro_uri, engine='pyarrow')
    _log(ticker, f"DONE +{len(new_micro):,} rows, total {len(df_micro):,}, last: {df_micro.index.max()}")

    del df_micro, new_micro
    _release_memory()
    return ticker, 'updated'


# ==============================================================================
# MAIN — context first, then remaining Z→A
# ==============================================================================
print("=" * 70)
print("  BACKFILL MICRO (PARALLEL) — VM edition")
print(f"  {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print(f"  Workers: {MAX_WORKERS}  |  Polygon sleep: {POLYGON_SLEEP}s")
print("=" * 70)

eligible = sorted(set(ALL_TICKERS) - SKIP_MICRO - SKIP_DEAD - CONTEXT_OHLCV_ONLY)
# Reverse alphabetical so this doesn't collide with backfill_micro_june.py (A→Z)
ordered = sorted(eligible, reverse=True)

print(f"\n  Context (SPY/VIX/TLT/IWM): OHLCV-only, no micro needed")
print(f"  Skipping: {sorted(SKIP_MICRO | SKIP_DEAD | CONTEXT_OHLCV_ONLY)}")
print(f"  Assets (Z→A): {len(ordered)} tickers\n")

summary = {'up_to_date': [], 'updated': [], 'no_data': [], 'failed': []}

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {pool.submit(process_ticker, t): t for t in ordered}
    for fut in as_completed(futures):
        ticker = futures[fut]
        try:
            _, status = fut.result()
            summary[status].append(ticker)
        except Exception as e:
            with print_lock:
                print(f"  [{ticker:6s}] EXCEPTION: {e}", flush=True)
            summary['failed'].append(ticker)

print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  Up to date:  {len(summary['up_to_date']):>3}")
print(f"  Updated:     {len(summary['updated']):>3}  {summary['updated']}")
print(f"  No T&Q data: {len(summary['no_data']):>3}  {summary['no_data']}")
print(f"  Failed:      {len(summary['failed']):>3}  {summary['failed']}")

print(f"\n  Note: SPY/VIX/TLT/IWM need OHLCV update (daily_ohlcv), not micro.")

print("=" * 70)
