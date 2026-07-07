"""
backfill_micro_june.py — Download missing T&Q + rebuild micro through June 30
==============================================================================
Targets ALL 81 assets with master_inference files in GCS.
Fully incremental: loads existing micro, finds the gap, fetches only missing
days from GCS cache or Polygon API, appends.

Run in Colab Enterprise. No if __name__ guard.
"""

import os, time, gc, ctypes, json, sys, subprocess, warnings
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

GCS_MICRO_DIR = f"gs://{BUCKET_NAME}/Data/Inference/Microstructure"
GCS_RAW_DIR   = f"gs://{BUCKET_NAME}/Data/TradesQuotes"

FETCH_END = pd.Timestamp("2026-07-01", tz='UTC')
HISTORY_START = "2025-09-01"

# All 81 tickers from master_inference (GCS names)
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

# Sector ETFs — OHLCV-only, no T&Q on Polygon
SKIP_MICRO = {'ITA', 'XLE'}

# Delisted / renamed stubs — skip
SKIP_DEAD = {'SMLR', 'BITF', 'MIGI'}

# Polygon API uses I:VIX for CBOE VIX index
POLYGON_TICKER_MAP = {'VIX': 'I:VIX'}

poly_client = RESTClient(POLYGON_API_KEY)
gcs_client  = gcs_storage.Client()


# ==============================================================================
# GCS HELPERS
# ==============================================================================
def _blob_path(gcs_uri):
    return gcs_uri.replace(f"gs://{BUCKET_NAME}/", "")

def _gcs_exists(gcs_uri):
    return gcs_client.bucket(BUCKET_NAME).blob(_blob_path(gcs_uri)).exists()


# ==============================================================================
# SCHEMA NORMALIZATION
# ==============================================================================
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
    expected = f"{clean_ticker}_{kind}_"
    if not os.path.basename(fname).startswith(expected):
        return False
    return True


# ==============================================================================
# POLYGON T&Q FETCH
# ==============================================================================
def fetch_polygon_trades(ticker, start_dt, end_dt):
    poly_ticker = POLYGON_TICKER_MAP.get(ticker, ticker)
    trades = []
    try:
        for t in poly_client.list_trades(poly_ticker,
                timestamp_gte=start_dt, timestamp_lte=end_dt, limit=50000):
            trades.append({'timestamp': pd.to_datetime(t.sip_timestamp, unit='ns', utc=True),
                           'price': t.price, 'size': t.size})
    except Exception as e:
        print(f"      API trades error: {e}")
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)

def fetch_polygon_quotes(ticker, start_dt, end_dt):
    poly_ticker = POLYGON_TICKER_MAP.get(ticker, ticker)
    quotes = []
    try:
        for q in poly_client.list_quotes(poly_ticker,
                timestamp_gte=start_dt, timestamp_lte=end_dt, limit=50000):
            quotes.append({'timestamp': pd.to_datetime(q.sip_timestamp, unit='ns', utc=True),
                           'ask_price': q.ask_price, 'ask_size': q.ask_size,
                           'bid_price': q.bid_price, 'bid_size': q.bid_size})
    except Exception as e:
        print(f"      API quotes error: {e}")
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
# MAIN
# ==============================================================================
print("=" * 70)
print("  BACKFILL MICRO — All assets through June 30, 2026")
print(f"  Run time: {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 70)

TICKERS = sorted(set(ALL_TICKERS) - SKIP_MICRO - SKIP_DEAD - CONTEXT_OHLCV_ONLY)
print(f"  Tickers: {len(TICKERS)}")
print(f"  Skipping (no micro needed): {sorted(SKIP_MICRO | SKIP_DEAD | CONTEXT_OHLCV_ONLY)}")
print(f"  Fetch end: {FETCH_END.date()}")

summary = {'up_to_date': [], 'updated': [], 'failed': [], 'no_data': []}

for ti, ticker in enumerate(TICKERS, 1):
    print(f"\n{'─'*52}")
    print(f"[{ti}/{len(TICKERS)}] {ticker}")
    clean_ticker = ticker
    micro_uri = f"{GCS_MICRO_DIR}/{clean_ticker}_1m_microstructure.parquet"

    # ── Load existing micro to find gap start ──
    try:
        df_micro = pd.read_parquet(micro_uri)
        if df_micro.index.tz is None:
            df_micro.index = df_micro.index.tz_localize('UTC')
        micro_start = df_micro.index.max() + timedelta(minutes=1)
        print(f"   Existing: {len(df_micro):,} rows, last: {df_micro.index.max()}")
    except Exception:
        df_micro = pd.DataFrame()
        micro_start = pd.Timestamp(HISTORY_START, tz='UTC')
        print(f"   No existing micro — starting from {HISTORY_START}")

    if micro_start >= FETCH_END:
        print("   Already up to date")
        summary['up_to_date'].append(ticker)
        continue

    # ── Day-by-day T&Q fill ──
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
                    raise ValueError("misnamed file")
            except Exception as e:
                print(f"      {date_str}: disk read failed — {e}")
                days_err += 1
                continue
        else:
            day_start = pd.Timestamp(gap_date, tz='UTC')
            day_end = day_start + timedelta(days=1)
            try:
                dt = fetch_polygon_trades(clean_ticker, day_start, day_end)
                dq = fetch_polygon_quotes(clean_ticker, day_start, day_end)
                if not dt.empty:
                    dt.to_parquet(t_uri, engine='pyarrow')
                if not dq.empty:
                    dq.to_parquet(q_uri, engine='pyarrow')
                days_api += 1
                time.sleep(0.25)
            except Exception as e:
                print(f"      {date_str}: API error — {e}")
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

    print(f"      {days_disk} disk / {days_api} API / {days_empty} empty / {days_err} errors")

    if not micro_chunks:
        print("      No new T&Q data found")
        summary['no_data'].append(ticker)
        del df_micro
        _release_memory()
        continue

    new_micro = pd.concat(micro_chunks).sort_index()
    del micro_chunks

    if not df_micro.empty:
        new_micro = new_micro[new_micro.index > df_micro.index.max()]

    if new_micro.empty:
        print("      All rows already covered")
        summary['up_to_date'].append(ticker)
        del df_micro, new_micro
        _release_memory()
        continue

    df_micro = pd.concat([df_micro, new_micro]).sort_index()
    df_micro = df_micro[~df_micro.index.duplicated(keep='last')]
    df_micro.to_parquet(micro_uri, engine='pyarrow')
    print(f"      Appended {len(new_micro):,} rows — total: {len(df_micro):,}")
    print(f"      New last date: {df_micro.index.max()}")
    summary['updated'].append(ticker)

    del df_micro, new_micro
    _release_memory()


# ==============================================================================
# SUMMARY
# ==============================================================================
print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  Up to date:  {len(summary['up_to_date']):>3}  {summary['up_to_date']}")
print(f"  Updated:     {len(summary['updated']):>3}  {summary['updated']}")
print(f"  No T&Q data: {len(summary['no_data']):>3}  {summary['no_data']}")
print(f"  Failed:      {len(summary['failed']):>3}  {summary['failed']}")
print()

print(f"  Note: SPY/VIX/TLT/IWM need OHLCV update (daily_ohlcv), not micro.")

print()
print("  Next steps:")
print("  1. Run daily_ohlcv_1.py + daily_ohlcv_2.py (if OHLCV also needs extending)")
print("  2. Run daily_stitch_master.py (joins OHLCV + micro → master_inference)")
print("  3. Re-run score-only scripts to extend score CSVs through June 30")
print("=" * 70)
