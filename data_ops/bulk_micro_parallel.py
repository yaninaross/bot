# %% ── Bulk Micro Download + Rebuild — PARALLEL VERSION ───────────────────────
#
# Parallelized rewrite of bulk_micro_all.py:
#   - Downloads multiple days concurrently within each ticker (ThreadPoolExecutor)
#   - Processes multiple tickers concurrently
#   - Global semaphore to respect Polygon API rate limits
#   - Same GCS paths, same rebuild logic, same output format
#
# Safe to run alongside the original sequential script IF different tickers.
#
# Usage:
#   python bulk_micro_parallel.py --sa                        # auto-stale, parallel
#   python bulk_micro_parallel.py --sa --all                  # force all
#   python bulk_micro_parallel.py --sa MARA KEEL ASTS         # specific tickers
#   python bulk_micro_parallel.py --sa --exclude AAPL,MSFT    # stale minus exclusions
#
#   --day-workers N    concurrent day downloads per ticker (default: 8)
#   --ticker-workers N concurrent tickers (default: 3)
#   --api-limit N      max concurrent Polygon API calls (default: 12)
#   --least-stale-first  process least-stale tickers first (refresh active tickers fast)

import os, io, time, gc, ctypes, sys, subprocess, warnings, threading
import numpy as np, pandas as pd, pytz
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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

ALL_TICKERS = [
    'APLD', 'CIFR', 'CLSK', 'FLNC', 'HUT', 'IREN', 'KEEL', 'MARA', 'PL', 'RCAT', 'RIOT',
    'ASTS', 'BBAI', 'LUNR', 'MVIS', 'WULF',
    'ALAB', 'OKLO', 'RKLB', 'RDW', 'SMCI',
    'BE', 'ONDS', 'SMR', 'WOLF',
    'ARM', 'BKSY', 'BTBT', 'CORZ', 'HOOD', 'MU', 'NNDM', 'NVDA', 'PLUG', 'SLS', 'VOYG',
    'AEHR', 'AVAV', 'CCJ', 'JOBY', 'SOUN',
    'AMD', 'COIN', 'MSTR', 'BGDE',
    'ACHR', 'GD', 'GE', 'KTOS', 'LHX', 'LMT', 'NOC', 'PLTR', 'RTX', 'SPIR',
    'CEG', 'COP', 'CVX', 'DUK', 'FCEL', 'GEV', 'NEE', 'VST', 'XOM',
    'BHP', 'CAT', 'ETN', 'EQIX', 'FCX', 'LIN', 'MOD', 'POWL',
    'MNTS', 'MP', 'NBIS', 'TSLA', 'VRT',
    'ACMR', 'BEAM', 'CLF', 'CRDO', 'MRNA', 'MXL', 'NEOV', 'NNE', 'NTLA', 'TWST', 'UCTT',
    'BABA',
    'AXTI', 'BLZE', 'BNY', 'BUUU', 'CRNG', 'MGRT', 'ROMA', 'RXT', 'SIF', 'SNDK',
    'SPY', 'VIX', 'TLT', 'IWM', 'ITA',
    'DELL', 'ZTG',
    'AAPL', 'AMZN', 'MSFT', 'META',
    'RIVN', 'NIO', 'XPEV', 'LI', 'NKLA',
    'GME', 'DJT',
    'BAC',
    'FSR',
    'AVGO', 'LRCX', 'AMAT', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'ANET',
    'INTC', 'QCOM', 'ADI', 'NXPI', 'MPWR', 'AMKR',
    'SNOW', 'DDOG', 'DT', 'MDB', 'CRM', 'NOW', 'ADBE', 'WDAY',
    'AI', 'PATH', 'TWLO', 'HUBS',
    'CRWD', 'PANW', 'ZS', 'S', 'FTNT', 'OKTA', 'NET', 'CYBR',
    'CIEN',
    'APP', 'RBLX', 'SQ', 'SHOP', 'TTD',
]

SKIP_TICKERS = {
    'SMLR', 'BITF', 'MIGI', 'XLE', 'BTCUSD', 'ETHUSD', 'SQQQ',
}

STALENESS_THRESHOLD_DAYS = 1
HISTORY_START   = pd.Timestamp("2022-01-01", tz='UTC')
FETCH_END       = pd.Timestamp.now(tz='UTC').normalize()

SKIP_DOWNLOAD   = False
FORCE_REBUILD   = False

# ── Parallelism config (overridable via CLI) ──
DAY_WORKERS     = 8     # concurrent day downloads per ticker
TICKER_WORKERS  = 3     # concurrent tickers
API_LIMIT       = 12    # max concurrent Polygon API calls (global)
POLYGON_SLEEP   = 0.12  # sleep after each API day-fetch (per-thread)

# ── Parse CLI ──
USE_SERVICE_ACCOUNT = '--sa' in sys.argv
_cli_args = [a for a in sys.argv[1:] if not a.startswith('-')]

_exclude_set = set()
LEAST_STALE_FIRST = '--least-stale-first' in sys.argv
for i, a in enumerate(sys.argv):
    if a == '--exclude' and i + 1 < len(sys.argv):
        _exclude_set = {t.strip().upper() for t in sys.argv[i + 1].split(',')}
    if a == '--day-workers' and i + 1 < len(sys.argv):
        DAY_WORKERS = int(sys.argv[i + 1])
    if a == '--ticker-workers' and i + 1 < len(sys.argv):
        TICKER_WORKERS = int(sys.argv[i + 1])
    if a == '--api-limit' and i + 1 < len(sys.argv):
        API_LIMIT = int(sys.argv[i + 1])

# Global rate limiter
_api_semaphore = threading.Semaphore(API_LIMIT)
_print_lock = threading.Lock()
_stats_lock = threading.Lock()

def tprint(msg):
    with _print_lock:
        print(msg, flush=True)


# ==============================================================================
# GCS CLIENT
# ==============================================================================
def _init_gcs_client():
    if USE_SERVICE_ACCOUNT:
        from google.auth import compute_engine
        creds = compute_engine.Credentials()
        client = gcs_storage.Client(credentials=creds, project='yanina-everything')
        tprint("  Auth: VM compute service account (never expires)")
    else:
        client = gcs_storage.Client()
        tprint("  Auth: Application Default Credentials (may expire)")
    return client

gcs_client = _init_gcs_client()
bucket     = gcs_client.bucket(BUCKET_NAME)

_last_refresh = time.time()
_REFRESH_INTERVAL = 1800
_refresh_lock = threading.Lock()

def _refresh_bucket():
    global gcs_client, bucket, _last_refresh
    with _refresh_lock:
        if time.time() - _last_refresh > _REFRESH_INTERVAL:
            gcs_client = _init_gcs_client()
            bucket = gcs_client.bucket(BUCKET_NAME)
            _last_refresh = time.time()

def _get_bucket():
    _refresh_bucket()
    return bucket


# ==============================================================================
# STALENESS SCANNER
# ==============================================================================
def scan_master_staleness(tickers, today=None):
    if today is None:
        today = pd.Timestamp.now(tz='UTC').normalize()
    tprint("  Scanning master_inference staleness on GCS...")
    blobs = list(bucket.list_blobs(prefix=f"{GCS_MASTER_DIR}/"))
    master_blobs = {
        os.path.basename(b.name).replace('_master_inference.parquet', ''): b
        for b in blobs if 'master_inference' in b.name and b.name.endswith('.parquet')
    }
    staleness = []
    for ticker in tickers:
        if ticker in SKIP_TICKERS or ticker in _exclude_set:
            continue
        blob = master_blobs.get(ticker)
        if blob is None:
            staleness.append((ticker, 9999, 'NO_MASTER'))
            continue
        gcs_updated = blob.updated.replace(tzinfo=pytz.UTC) if blob.updated.tzinfo is None else blob.updated
        days_since = (today - gcs_updated.replace(hour=0, minute=0, second=0, microsecond=0)).days
        staleness.append((ticker, days_since, blob.updated.strftime('%Y-%m-%d')))
    if LEAST_STALE_FIRST:
        staleness.sort(key=lambda x: x[1])
    else:
        staleness.sort(key=lambda x: -x[1])
    return staleness

def filter_stale_tickers(staleness, threshold_days=STALENESS_THRESHOLD_DAYS):
    stale = [(t, d, info) for t, d, info in staleness if d > threshold_days]
    fresh = [(t, d, info) for t, d, info in staleness if d <= threshold_days]
    return stale, fresh


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
# PHASE 1: SCAN GCS + PARALLEL DOWNLOAD MISSING T&Q
# ==============================================================================
def scan_existing_tq(ticker):
    bkt = _get_bucket()
    prefix = f"{GCS_RAW_DIR}/{ticker}/"
    blobs = list(bkt.list_blobs(prefix=prefix))
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


def _download_one_day(ticker, date_str, poly):
    """Download trades + quotes for one day. Returns ('ok', 'empty', or 'error')."""
    day_start = pd.Timestamp(date_str, tz='UTC')
    day_end   = day_start + timedelta(days=1)
    bkt = _get_bucket()

    with _api_semaphore:
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

        except Exception as e:
            return 'error', str(e)

    dt = pd.DataFrame(trades) if trades else pd.DataFrame()
    dq = pd.DataFrame(quotes) if quotes else pd.DataFrame()

    if dt.empty and dq.empty:
        return 'empty', ''

    if not dt.empty:
        buf = io.BytesIO()
        dt.to_parquet(buf, engine='pyarrow')
        buf.seek(0)
        bkt.blob(f"{GCS_RAW_DIR}/{ticker}/{ticker}_trades_{date_str}.parquet"
                 ).upload_from_file(buf)

    if not dq.empty:
        buf = io.BytesIO()
        dq.to_parquet(buf, engine='pyarrow')
        buf.seek(0)
        bkt.blob(f"{GCS_RAW_DIR}/{ticker}/{ticker}_quotes_{date_str}.parquet"
                 ).upload_from_file(buf)

    return 'ok', ''


def download_missing_tq(ticker):
    trade_dates, quote_dates = scan_existing_tq(ticker)
    both_existing = trade_dates & quote_dates

    full_range = pd.bdate_range(HISTORY_START, FETCH_END)
    all_needed = {d.strftime('%Y-%m-%d') for d in full_range}
    missing = sorted(all_needed - both_existing)

    if not missing:
        tprint(f"  [{ticker:6s}] All {len(all_needed)} bdays cached — skip")
        return 0, len(all_needed), 0, 0

    first_ex = min(both_existing) if both_existing else "none"
    last_ex  = max(both_existing) if both_existing else "none"
    tprint(f"  [{ticker:6s}] Cached: {len(both_existing)} ({first_ex} -> {last_ex}), "
           f"missing: {len(missing)} ({missing[0]} -> {missing[-1]})")

    poly = RESTClient(POLYGON_API_KEY)
    downloaded = 0
    empty = 0
    errors = 0
    error_msgs = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=DAY_WORKERS) as executor:
        futures = {
            executor.submit(_download_one_day, ticker, date_str, poly): date_str
            for date_str in missing
        }

        done_count = 0
        _log_interval = 25 if len(missing) < 500 else 50
        for future in as_completed(futures):
            done_count += 1
            status, msg = future.result()
            if status == 'ok':
                downloaded += 1
            elif status == 'empty':
                empty += 1
            else:
                errors += 1
                if len(error_msgs) < 5:
                    error_msgs.append(f"{futures[future]}: {msg}")

            if done_count == 1 or done_count % _log_interval == 0:
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta_min = (len(missing) - done_count) / rate / 60 if rate > 0 else 0
                tprint(f"  [{ticker:6s}]   {done_count}/{len(missing)} — "
                       f"{downloaded} dl / {empty} empty / {errors} err  "
                       f"({rate:.1f} days/s, ETA {eta_min:.0f}m)")

    for msg in error_msgs:
        tprint(f"  [{ticker:6s}]   Err {msg}")

    elapsed = time.time() - t0
    tprint(f"  [{ticker:6s}] DL done: +{downloaded}, "
           f"{downloaded + len(both_existing)} total, "
           f"{empty} empty, {errors} err  ({elapsed:.0f}s)")
    return downloaded, len(both_existing), empty, errors


# ==============================================================================
# PHASE 2: REBUILD MICRO FROM ALL RAW T&Q
# ==============================================================================
def rebuild_micro(ticker):
    bkt = _get_bucket()
    prefix = f"{GCS_RAW_DIR}/{ticker}/"
    blobs = list(bkt.list_blobs(prefix=prefix))

    if not blobs:
        tprint(f"  [{ticker:6s}] No raw T&Q — skip micro rebuild")
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

    tprint(f"  [{ticker:6s}] Micro: {len(all_dates)} days "
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

            t_blob = bkt.blob(t_path)
            if t_blob.exists():
                dt = pd.read_parquet(io.BytesIO(t_blob.download_as_bytes()))
                dt = normalize_raw_schema(dt, 'trades')

            q_blob = bkt.blob(q_path)
            if q_blob.exists():
                dq = pd.read_parquet(io.BytesIO(q_blob.download_as_bytes()))
                dq = normalize_raw_schema(dq, 'quotes')

            if dt.empty and dq.empty:
                continue

            day_micro = process_raw_to_microstructure(dt, dq)
            del dt, dq

            if not day_micro.empty:
                micro_chunks.append(day_micro)
                processed += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                tprint(f"  [{ticker:6s}]   Err {date_str}: {e}")

        if (i + 1) % 200 == 0:
            tprint(f"  [{ticker:6s}]   {i+1}/{len(all_dates)} days...")

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
    bkt.blob(micro_path).upload_from_file(buf)

    tprint(f"  [{ticker:6s}] Micro saved: {len(df_micro):,} rows "
           f"({df_micro.index.min()} -> {df_micro.index.max()})")

    row_count = len(df_micro)
    del df_micro
    _release_memory()
    return row_count


# ==============================================================================
# PHASE 3: REBUILD MASTER
# ==============================================================================
def rebuild_master(ticker):
    bkt = _get_bucket()
    ohlcv_path  = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
    micro_path  = f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet"
    master_path = f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet"

    ohlcv_blob = bkt.blob(ohlcv_path)
    if not ohlcv_blob.exists():
        tprint(f"  [{ticker:6s}] No OHLCV — skip master")
        return 0

    micro_blob = bkt.blob(micro_path)
    if not micro_blob.exists():
        tprint(f"  [{ticker:6s}] No micro — skip master")
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
    bkt.blob(master_path).upload_from_file(buf)

    tc_nz = (df_master['Trade_Count'] > 0).sum() if 'Trade_Count' in df_master.columns else 0
    tprint(f"  [{ticker:6s}] Master: {len(df_master):,} rows, TC>0: {tc_nz:,}")

    row_count = len(df_master)
    del df_master
    _release_memory()
    return row_count


# ==============================================================================
# PER-TICKER PIPELINE
# ==============================================================================
def process_one_ticker(ticker, idx, total):
    t0 = time.time()
    tprint(f"\n{'─'*60}")
    tprint(f"  [{idx}/{total}] {ticker}")
    tprint(f"{'─'*60}")

    downloaded = 0
    existing = 0
    empty = 0
    errors = 0

    if not SKIP_DOWNLOAD:
        downloaded, existing, empty, errors = download_missing_tq(ticker)

    bkt = _get_bucket()
    master_path = f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet"
    has_master = bkt.blob(master_path).exists()

    need_rebuild = FORCE_REBUILD or (downloaded > 0) or (not has_master)

    if not need_rebuild:
        tprint(f"  [{ticker:6s}] Master exists + no new T&Q — skip rebuild")
        return {
            'ticker': ticker, 'downloaded': downloaded, 'existing': existing,
            'empty': empty, 'errors': errors,
            'micro_rows': 0, 'master_rows': 0, 'action': 'skip_rebuild',
            'elapsed': time.time() - t0,
        }

    reason = "forced" if FORCE_REBUILD else ("no master" if not has_master else f"+{downloaded} new T&Q")
    tprint(f"  [{ticker:6s}] Rebuilding ({reason})...")

    micro_rows = rebuild_micro(ticker)

    master_rows = 0
    if micro_rows > 0:
        master_rows = rebuild_master(ticker)

    elapsed = time.time() - t0
    tprint(f"  [{ticker:6s}] DONE in {elapsed:.0f}s ({elapsed/60:.1f}m)")

    return {
        'ticker': ticker, 'downloaded': downloaded, 'existing': existing,
        'empty': empty, 'errors': errors,
        'micro_rows': micro_rows, 'master_rows': master_rows, 'action': 'rebuilt',
        'elapsed': elapsed,
    }


# ==============================================================================
# MAIN
# ==============================================================================
_t0 = time.time()

cli_args = [a for a in sys.argv[1:] if not a.endswith('.json')]
force_all = '--all' in cli_args
ticker_args = [a.upper() for a in _cli_args if a.upper() not in ('--SA', '--ALL')]

if ticker_args:
    TARGET_TICKERS = [t for t in ticker_args if t not in _exclude_set]
    staleness_info = None
elif force_all:
    TARGET_TICKERS = [t for t in dict.fromkeys(ALL_TICKERS)
                      if t not in SKIP_TICKERS and t not in _exclude_set]
    staleness_info = None
else:
    all_candidates = [t for t in dict.fromkeys(ALL_TICKERS)
                      if t not in SKIP_TICKERS and t not in _exclude_set]
    staleness = scan_master_staleness(all_candidates)
    stale, fresh = filter_stale_tickers(staleness, STALENESS_THRESHOLD_DAYS)
    staleness_info = (stale, fresh)
    TARGET_TICKERS = [t for t, _, _ in stale]

    print(f"\n  Staleness scan results:")
    print(f"    Total assets:      {len(staleness)}")
    print(f"    Up to date (<={STALENESS_THRESHOLD_DAYS}d): {len(fresh)}")
    print(f"    Stale (>{STALENESS_THRESHOLD_DAYS}d):       {len(stale)}")
    print(f"    Excluded (CLI):    {len(_exclude_set)} ({', '.join(sorted(_exclude_set))})")
    print(f"    Skipped (dead):    {len(SKIP_TICKERS)} ({', '.join(sorted(SKIP_TICKERS))})")
    if stale:
        print(f"\n  Priority queue (most stale first):")
        print(f"  {'#':>4} {'Ticker':<8} {'Stale':>6} {'Last Upload':>12}")
        print(f"  {'─'*4} {'─'*8} {'─'*6} {'─'*12}")
        for i, (t, d, info) in enumerate(stale, 1):
            label = f"{d}d" if d < 9999 else "NEW"
            print(f"  {i:>4} {t:<8} {label:>6} {info:>12}")

bdays = len(pd.bdate_range(HISTORY_START, FETCH_END))

print(f"\n{'='*60}")
print("  BULK MICRO DOWNLOAD + REBUILD — PARALLEL")
print(f"  {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print(f"  Tickers: {len(TARGET_TICKERS)}")
print(f"  Range: {HISTORY_START.strftime('%Y-%m-%d')} -> "
      f"{FETCH_END.strftime('%Y-%m-%d')} ({bdays} bdays)")
print(f"  Mode: {'explicit list' if ticker_args else ('all (--all)' if force_all else 'auto-stale')}")
print(f"  Parallelism: {DAY_WORKERS} day-workers × {TICKER_WORKERS} ticker-workers, "
      f"API limit {API_LIMIT}")
print(f"  Order: {'LEAST stale first (refresh active tickers fast)' if LEAST_STALE_FIRST else 'MOST stale first (default)'}")
print(f"  Excluded: {sorted(_exclude_set) if _exclude_set else 'none'}")
print(f"  SKIP_DOWNLOAD: {SKIP_DOWNLOAD}")
print(f"  FORCE_REBUILD: {FORCE_REBUILD}")
print("=" * 60)

if not TARGET_TICKERS:
    print("\n  All tickers up to date — nothing to do.")
    sys.exit(0)

# ── Process tickers with thread pool ──
results = []

if TICKER_WORKERS <= 1:
    for idx, ticker in enumerate(TARGET_TICKERS, 1):
        try:
            r = process_one_ticker(ticker, idx, len(TARGET_TICKERS))
            results.append(r)
        except Exception as e:
            tprint(f"  [{ticker:6s}] FATAL: {e}")
            results.append({
                'ticker': ticker, 'downloaded': 0, 'existing': 0,
                'empty': 0, 'errors': 1,
                'micro_rows': 0, 'master_rows': 0, 'action': 'FAILED',
                'elapsed': 0,
            })
else:
    with ThreadPoolExecutor(max_workers=TICKER_WORKERS) as executor:
        futures = {}
        for idx, ticker in enumerate(TARGET_TICKERS, 1):
            f = executor.submit(process_one_ticker, ticker, idx, len(TARGET_TICKERS))
            futures[f] = ticker

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                r = future.result()
                results.append(r)
            except Exception as e:
                tprint(f"  [{ticker:6s}] FATAL: {e}")
                results.append({
                    'ticker': ticker, 'downloaded': 0, 'existing': 0,
                    'empty': 0, 'errors': 1,
                    'micro_rows': 0, 'master_rows': 0, 'action': 'FAILED',
                    'elapsed': 0,
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
print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m, {elapsed/3600:.1f}h)")

if rebuilt:
    print(f"\n  Rebuilt tickers:")
    print(f"  {'Ticker':<8} {'DL':>6} {'Micro':>12} {'Master':>12} {'Time':>8}")
    print(f"  {'─'*8} {'─'*6} {'─'*12} {'─'*12} {'─'*8}")
    for r in sorted(rebuilt, key=lambda x: x['ticker']):
        t_str = f"{r.get('elapsed', 0)/60:.0f}m"
        print(f"  {r['ticker']:<8} {r['downloaded']:>6} "
              f"{r['micro_rows']:>12,} {r['master_rows']:>12,} {t_str:>8}")

if failed:
    print(f"\n  FAILED: {[r['ticker'] for r in failed]}")

if not ticker_args and not force_all and staleness_info:
    stale_set = {t for t, _, _ in staleness_info[0]}
    rebuilt_set = {r['ticker'] for r in rebuilt}
    still_stale = stale_set - rebuilt_set - {r['ticker'] for r in skipped}
    if still_stale:
        print(f"\n  WARNING: {len(still_stale)} tickers still stale after run:")
        print(f"    {', '.join(sorted(still_stale))}")

print("=" * 60)
