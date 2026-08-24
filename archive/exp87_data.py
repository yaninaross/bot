import os
import time
import pytz
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from polygon import RESTClient
from google.cloud import storage as gcs_storage
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# ⚙️ 1. CONFIGURATION & GCS PATHS
# ==============================================================================
# All assets with T&Q raw data available — build microstructure for all of them
# SMH only has data through 2023-02-10 (279 files) — not enough, stays OHLCV-only via CONTEXT_ASSETS
# MOD, TLT, IWM, VIX, SPY, BTCUSD have 0 T&Q files — OHLCV context only
TARGET_TICKERS = [
    'APLD', 'NVDA', 'SMCI', 'VRT', 'MARA', 'CORZ', 'IREN', 'WULF',
    'RIOT', 'CLSK', 'AMD', 'PLTR', 'COIN', 'MSTR', 'TSLA',
]
CONTEXT_ASSETS = ['SMH', 'MOD', 'TLT', 'IWM', 'I:VIX', 'SPY', 'X:BTCUSD']
ALL_ASSETS = list(set(TARGET_TICKERS + CONTEXT_ASSETS))

HISTORY_START = "2022-01-01"

BUCKET_NAME = "time-series-bucket-ys"

# GCS URI paths (gs://) — used for pandas read/write via gcsfs (FUSE-independent)
GCS_OHLCV_DIR = f"gs://{BUCKET_NAME}/Data/Inference"
GCS_MICRO_DIR = f"gs://{BUCKET_NAME}/Data/Inference/Microstructure"
GCS_RAW_DIR   = f"gs://{BUCKET_NAME}/Data/TradesQuotes"

# FUSE paths — kept only for os.makedirs of local temp dirs (not for GCS reads/writes)
BASE_PATH = "/gcs/time-series-bucket-ys"
OHLCV_DIR = f"{BASE_PATH}/Data/Inference"
MICRO_DIR = f"{BASE_PATH}/Data/Inference/Microstructure"
RAW_DIR   = f"{BASE_PATH}/Data/TradesQuotes"

# ==============================================================================
# 🔁 FULL_REBUILD MODE
# ------------------------------------------------------------------------------
# True  → delete existing microstructure and rebuild from all historical raw files.
# False → normal daily incremental update (fetch today's data, append to micro).
#
# NORMALIZE_RAW_FILES: rewrite old-schema raw files in GCS in-place.
# Not needed for FULL_REBUILD — build_microstructure_from_scratch normalizes
# at read time. Only run this as a one-off cleanup if needed.
# ==============================================================================
FULL_REBUILD        = False
NORMALIZE_RAW_FILES = False   # skip expensive in-place schema fix during rebuild

# Tickers already completed in a previous run — skip to save time.
# Reset to set() after a clean full run.
SKIP_TICKERS = set()

for d in [OHLCV_DIR, MICRO_DIR, RAW_DIR]:
    os.makedirs(d, exist_ok=True)

poly_client = RESTClient("2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
gcs_client  = gcs_storage.Client()


def _gcs_blob_path(gcs_uri):
    """Strip gs://bucket/ prefix to get the blob path."""
    return gcs_uri.replace(f"gs://{BUCKET_NAME}/", "")

def _gcs_exists(gcs_uri):
    """Check if a GCS blob exists using the SDK (FUSE-independent)."""
    return gcs_client.bucket(BUCKET_NAME).blob(_gcs_blob_path(gcs_uri)).exists()

def _list_raw_blobs(clean_ticker):
    """List raw parquet filenames for a ticker using GCS SDK (FUSE-independent)."""
    prefix = f"Data/TradesQuotes/{clean_ticker}/"
    return sorted(
        b.name.split('/')[-1]
        for b in gcs_client.list_blobs(BUCKET_NAME, prefix=prefix)
        if b.name.endswith('.parquet')
    )

# ==============================================================================
# 🛠️ 2. SCHEMA NORMALIZATION
# ==============================================================================
def normalize_raw_schema(df, kind):
    """
    Normalize a raw trades/quotes DataFrame to the canonical schema.

    Handles all known timestamp column name variants:
      ns, sip_timestamp, participant_timestamp → timestamp
    Handles all known timestamp types:
      int (nanoseconds), datetime (no tz), datetime (with tz), object/string

    Old bulk script (wrong schema):
      trades : ns (int or datetime), price, size
      quotes : ns (int or datetime), bid, ask, bsize, asize
    Canonical schema (exp87_data / new bulk script):
      trades : timestamp (UTC datetime64), price, size
      quotes : timestamp (UTC datetime64), bid_price, ask_price, bid_size, ask_size

    Returns empty DataFrame if no usable timestamp column can be found.
    """
    if df is None or df.empty:
        return df

    # ── Timestamp column — normalize name ────────────────────────────────────
    for ts_col in ('ns', 'sip_timestamp', 'participant_timestamp'):
        if ts_col in df.columns and 'timestamp' not in df.columns:
            df = df.rename(columns={ts_col: 'timestamp'})
            break

    if 'timestamp' not in df.columns:
        return pd.DataFrame()   # no usable timestamp — signal failure

    # ── Timestamp column — normalize type ────────────────────────────────────
    col = df['timestamp']
    try:
        if pd.api.types.is_integer_dtype(col):
            df['timestamp'] = pd.to_datetime(col, unit='ns', utc=True)
        elif pd.api.types.is_datetime64_any_dtype(col):
            if col.dt.tz is None:
                df['timestamp'] = col.dt.tz_localize('UTC')
        else:
            # object / string — try generic parse
            df['timestamp'] = pd.to_datetime(col, utc=True)
    except Exception:
        return pd.DataFrame()   # unparseable timestamp — signal failure

    # ── Quote column renames ──────────────────────────────────────────────────
    if kind == 'quotes':
        df = df.rename(columns={
            'bid':   'bid_price',
            'ask':   'ask_price',
            'bsize': 'bid_size',
            'asize': 'ask_size',
        })

    return df


def _valid_raw_file(fname, clean_ticker, kind):
    """
    Return True only if the filename starts with the expected
    '{clean_ticker}_trades_' or '{clean_ticker}_quotes_' prefix.
    Rejects misnamed files (e.g. SMCI_trades_*.parquet stored in APLD folder).
    Accepts either a bare filename or a full path.
    """
    fname    = os.path.basename(fname)
    expected = f"{clean_ticker}_{kind}_"
    if not fname.startswith(expected):
        print(f"      ⚠️  Skipping misnamed file: {fname}  (expected prefix '{expected}')")
        return False
    return True


def _needs_normalization(df, kind):
    """Check if a loaded DataFrame needs schema normalization."""
    if 'ns' in df.columns:
        return True
    if kind == 'quotes' and 'bid' in df.columns:
        return True
    return False


def normalize_ticker_raw_files(clean_ticker):
    """
    Scan all raw parquet files for a ticker and normalize schema in-place.
    Uses GCS SDK for file listing and gs:// URIs for reads/writes (FUSE-independent).
    Only rewrites files that actually need fixing.
    """
    files = _list_raw_blobs(clean_ticker)
    if not files:
        print(f"   ⚠️  No raw files for {clean_ticker} in GCS — skipping normalization")
        return

    total, fixed, errors = len(files), 0, 0

    for fname in files:
        gcs_uri = f"{GCS_RAW_DIR}/{clean_ticker}/{fname}"
        kind    = 'quotes' if '_quotes_' in fname else 'trades'

        try:
            df = pd.read_parquet(gcs_uri)
            if not _needs_normalization(df, kind):
                continue
            df = normalize_raw_schema(df, kind)
            df.to_parquet(gcs_uri, engine='pyarrow', compression='snappy')
            fixed += 1
        except Exception as e:
            print(f"      ⚠️  Could not normalize {fname}: {e}")
            errors += 1

    status = f"fixed {fixed}/{total}"
    if errors:
        status += f", {errors} errors"
    print(f"   🔧 [{clean_ticker}] Raw file normalization: {status}")


# ==============================================================================
# 🛠️ 3. POLYGON API FETCH HELPERS
# ==============================================================================
def fetch_polygon_aggs(ticker, start_dt, end_dt, chunk_months=1, sleep_secs=2):
    """Fetches 1m OHLCV bars from Polygon in monthly chunks."""
    start_dt = pd.Timestamp(start_dt, tz='UTC') if not isinstance(start_dt, pd.Timestamp) else start_dt.tz_localize('UTC') if start_dt.tzinfo is None else start_dt
    end_dt   = pd.Timestamp(end_dt,   tz='UTC') if not isinstance(end_dt,   pd.Timestamp) else end_dt.tz_localize('UTC')   if end_dt.tzinfo   is None else end_dt

    all_aggs = []
    chunk_start = start_dt
    total_days = (end_dt - start_dt).days

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + relativedelta(months=chunk_months), end_dt)
        try:
            for a in poly_client.list_aggs(ticker, 1, 'minute', chunk_start, chunk_end, limit=50000):
                all_aggs.append({'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                                 'Open': a.open, 'High': a.high, 'Low': a.low,
                                 'Close': a.close, 'Volume': a.volume})
        except Exception as e:
            print(f"      ⚠️ Chunk {chunk_start.date()} → {chunk_end.date()} failed: {e}. Retrying after 10s...")
            time.sleep(10)
            try:
                for a in poly_client.list_aggs(ticker, 1, 'minute', chunk_start, chunk_end, limit=50000):
                    all_aggs.append({'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                                     'Open': a.open, 'High': a.high, 'Low': a.low,
                                     'Close': a.close, 'Volume': a.volume})
            except Exception as e2:
                print(f"      ❌ Chunk {chunk_start.date()} → {chunk_end.date()} skipped after retry: {e2}")

        if total_days > 30:
            print(f"      📦 {chunk_start.date()} → {chunk_end.date()}: {len(all_aggs):,} bars so far")
        chunk_start = chunk_end
        if chunk_start < end_dt:
            time.sleep(sleep_secs)

    if not all_aggs:
        return pd.DataFrame()
    df = pd.DataFrame(all_aggs).set_index('timestamp')
    return df[~df.index.duplicated(keep='last')].sort_index()


def fetch_polygon_trades(ticker, start_dt, end_dt):
    """Fetches raw nanosecond trades from Polygon."""
    trades = []
    for t in poly_client.list_trades(ticker, timestamp_gte=start_dt, timestamp_lte=end_dt, limit=50000):
        trades.append({'timestamp': pd.to_datetime(t.sip_timestamp, unit='ns', utc=True),
                       'price': t.price, 'size': t.size})
    if not trades: return pd.DataFrame()
    return pd.DataFrame(trades)


def fetch_polygon_quotes(ticker, start_dt, end_dt):
    """Fetches raw nanosecond NBBO quotes from Polygon."""
    quotes = []
    for q in poly_client.list_quotes(ticker, timestamp_gte=start_dt, timestamp_lte=end_dt, limit=50000):
        quotes.append({'timestamp': pd.to_datetime(q.sip_timestamp, unit='ns', utc=True),
                       'ask_price': q.ask_price, 'ask_size': q.ask_size,
                       'bid_price': q.bid_price, 'bid_size': q.bid_size})
    if not quotes: return pd.DataFrame()
    return pd.DataFrame(quotes)


# ==============================================================================
# 🧮 4. MICROSTRUCTURE MATH ENGINE
# ==============================================================================
def process_raw_to_microstructure(trades_df, quotes_df):
    if trades_df is not None and not trades_df.empty:
        trades_df = trades_df.set_index('timestamp')
        trades_df['Direction'] = np.sign(trades_df['price'].diff())
        trades_df['Direction'] = trades_df['Direction'].mask(trades_df['Direction'] == 0).ffill().fillna(0)
        trades_df['Volume_Delta'] = trades_df['size'] * trades_df['Direction']
        t_1m = trades_df.resample('1min').agg({
            'size': 'sum',
            'Volume_Delta': 'sum',
            'price': 'count'
        }).rename(columns={'size': 'Total_Vol', 'price': 'Trade_Count'}).fillna(0)
    else:
        t_1m = pd.DataFrame(columns=['Total_Vol', 'Volume_Delta', 'Trade_Count'])

    if quotes_df is not None and not quotes_df.empty:
        quotes_df = quotes_df.set_index('timestamp')
        quotes_df['Spread'] = quotes_df['ask_price'] - quotes_df['bid_price']
        quotes_df['OBI'] = (quotes_df['bid_size'] - quotes_df['ask_size']) / (quotes_df['bid_size'] + quotes_df['ask_size'] + 1e-9)
        q_1m = quotes_df.resample('1min').agg({
            'Spread': 'mean',
            'OBI': 'mean',
            'bid_size': 'sum',
            'ask_size': 'sum'
        }).rename(columns={'bid_size': 'bsize', 'ask_size': 'asize'}).ffill()
    else:
        q_1m = pd.DataFrame(columns=['Spread', 'OBI', 'bsize', 'asize'])

    daily_df = pd.concat([q_1m, t_1m], axis=1).ffill().fillna(0)
    daily_df['Delta_Pct'] = daily_df['Volume_Delta'] / (daily_df['Total_Vol'] + 1e-9)
    return daily_df


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# 🔁 5. FULL MICROSTRUCTURE REBUILD FROM RAW FILES
# ==============================================================================
def build_microstructure_from_scratch(clean_ticker):
    """
    Read every historical raw trades/quotes parquet file for a ticker (sorted by
    date) and assemble a complete microstructure parquet from scratch.
    Uses GCS SDK for file listing and gs:// URIs for reads (FUSE-independent).
    Files are normalized on-the-fly so mixed schemas are handled transparently.
    Returns the complete microstructure DataFrame (not saved here).
    """
    all_files   = _list_raw_blobs(clean_ticker)
    trade_files = [f for f in all_files if f.startswith(f"{clean_ticker}_trades_")]

    if not trade_files:
        print(f"   ⚠️  No raw trade files for {clean_ticker} in GCS — cannot rebuild micro")
        return pd.DataFrame()

    # Build a set of available quote filenames for O(1) lookup
    quote_files_set = set(f for f in all_files if f.startswith(f"{clean_ticker}_quotes_"))

    print(f"   📂 [{clean_ticker}] Rebuilding micro from {len(trade_files)} raw date files...")
    all_micro = []

    for i, tf in enumerate(trade_files, 1):
        date_str = tf.replace(f"{clean_ticker}_trades_", "").replace(".parquet", "")
        qf       = f"{clean_ticker}_quotes_{date_str}.parquet"
        t_uri    = f"{GCS_RAW_DIR}/{clean_ticker}/{tf}"
        q_uri    = f"{GCS_RAW_DIR}/{clean_ticker}/{qf}"

        try:
            if _valid_raw_file(tf, clean_ticker, 'trades'):
                raw_trades = normalize_raw_schema(pd.read_parquet(t_uri), 'trades')
            else:
                raw_trades = pd.DataFrame()
        except Exception as e:
            print(f"      ⚠️  {date_str} trades read error: {e}")
            raw_trades = pd.DataFrame()

        try:
            if qf in quote_files_set and _valid_raw_file(qf, clean_ticker, 'quotes'):
                raw_quotes = normalize_raw_schema(pd.read_parquet(q_uri), 'quotes')
            else:
                raw_quotes = pd.DataFrame()
        except Exception as e:
            print(f"      ⚠️  {date_str} quotes read error: {e}")
            raw_quotes = pd.DataFrame()

        if raw_trades.empty and raw_quotes.empty:
            continue

        try:
            day_micro = process_raw_to_microstructure(raw_trades, raw_quotes)
            if not day_micro.empty:
                all_micro.append(day_micro)
        except Exception as e:
            print(f"      ⚠️  {date_str} micro calc error: {e}")

        if i % 100 == 0:
            print(f"      ⏳ Processed {i}/{len(trade_files)} days...")

    if not all_micro:
        print(f"   ⚠️  No micro data produced for {clean_ticker}")
        return pd.DataFrame()

    df_micro = pd.concat(all_micro).sort_index()
    df_micro = df_micro[~df_micro.index.duplicated(keep='last')]
    print(f"   ✅ [{clean_ticker}] Rebuilt {len(df_micro):,} micro rows from {len(all_micro)} days")
    return df_micro


# ==============================================================================
# 🚀 6. MAIN LOOP
# ==============================================================================
print(f"🚀 INITIALIZING DECOUPLED HARVESTER  |  FULL_REBUILD={FULL_REBUILD}")
today_str = datetime.now(pytz.utc).strftime('%Y-%m-%d')
fetch_end = pd.Timestamp.now(tz='UTC')

# GCS FUSE is unreliable for subdirectory discovery — all GCS reads/writes
# now use gs:// URIs (gcsfs) or the GCS SDK directly, not FUSE paths.
# The FUSE warming below is informational only; no logic depends on it.
print("ℹ️  GCS paths: using gs:// URIs + SDK (FUSE not relied upon for reads/writes)")

# ── Optional: normalize raw file schemas in GCS in-place ─────────────────────
# Skipped by default — build_microstructure_from_scratch normalizes at read time.
if NORMALIZE_RAW_FILES:
    print("\n" + "="*60)
    print("🔧 Normalizing raw file schemas in GCS...")
    print("="*60)
    for ticker in ALL_ASSETS:
        if ticker in CONTEXT_ASSETS:
            continue
        clean_ticker = ticker.replace("I:", "").replace("X:", "")
        normalize_ticker_raw_files(clean_ticker)
    print("\n✅ Raw file normalization complete.\n")

if FULL_REBUILD:
    print("\n" + "="*60)
    print("🔁 FULL REBUILD MODE — rebuilding microstructure from raw files")
    print("   (schema normalized at read time — no pre-pass needed)")
    print("="*60 + "\n")

# ── PER-TICKER LOOP ───────────────────────────────────────────────────────────
for ticker in ALL_ASSETS:
    if ticker in SKIP_TICKERS:
        print(f"   ⏭️  Skipping {ticker} (already completed)")
        continue
    print(f"\n" + "-"*50)
    print(f"🔄 Processing {ticker}...")
    clean_ticker = ticker.replace("I:", "").replace("X:", "")

    # gs:// URIs for all GCS reads/writes — FUSE-independent
    ohlcv_uri  = f"{GCS_OHLCV_DIR}/{clean_ticker}_1m_full_historical_data.parquet"
    micro_uri  = f"{GCS_MICRO_DIR}/{clean_ticker}_1m_microstructure.parquet"
    master_uri = f"{GCS_OHLCV_DIR}/{clean_ticker}_master_inference.parquet"

    # Daily raw file gs:// URIs (used in incremental mode only)
    raw_t_uri  = f"{GCS_RAW_DIR}/{clean_ticker}/{clean_ticker}_trades_{today_str}.parquet"
    raw_q_uri  = f"{GCS_RAW_DIR}/{clean_ticker}/{clean_ticker}_quotes_{today_str}.parquet"

    # ── A. OHLCV anchor ──────────────────────────────────────────────────────
    try:
        df_ohlcv = pd.read_parquet(ohlcv_uri)
        if df_ohlcv.index.tz is None: df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
        ohlcv_start = df_ohlcv.index.max() + timedelta(minutes=1)
    except Exception:
        print(f"   🌱 No historical OHLCV — bootstrapping from {HISTORY_START}...")
        df_ohlcv = pd.DataFrame()
        ohlcv_start = pd.Timestamp(HISTORY_START, tz='UTC')

    # ── B. Microstructure anchor ─────────────────────────────────────────────
    if ticker not in CONTEXT_ASSETS:
        if FULL_REBUILD:
            # Delete existing micro via GCS SDK so we start clean
            micro_blob = gcs_client.bucket(BUCKET_NAME).blob(
                f"Data/Inference/Microstructure/{clean_ticker}_1m_microstructure.parquet"
            )
            if micro_blob.exists():
                micro_blob.delete()
                print(f"   🗑️  Deleted existing microstructure for {clean_ticker}")
            df_micro    = pd.DataFrame()
            micro_start = ohlcv_start
        else:
            try:
                df_micro = pd.read_parquet(micro_uri)
                if df_micro.index.tz is None: df_micro.index = df_micro.index.tz_localize('UTC')
                micro_start = df_micro.index.max() + timedelta(minutes=1)
            except Exception:
                df_micro    = pd.DataFrame()
                micro_start = ohlcv_start
    else:
        df_micro    = pd.DataFrame()
        micro_start = fetch_end

    # ── C. FETCH OHLCV ───────────────────────────────────────────────────────
    if ohlcv_start < fetch_end:
        print(f"   ⏳ [OHLCV] Fetching from {ohlcv_start}...")
        new_aggs = fetch_polygon_aggs(ticker, ohlcv_start, fetch_end)
        if not new_aggs.empty:
            print(f"      📥 {len(new_aggs):,} new 1m bars.")
            df_ohlcv = pd.concat([df_ohlcv, new_aggs])
            df_ohlcv = df_ohlcv[~df_ohlcv.index.duplicated(keep='last')].sort_index()
            df_ohlcv.to_parquet(ohlcv_uri, engine='pyarrow')
    else:
        print("   ✅ [OHLCV] Already up to date.")

    # ── D. MICROSTRUCTURE ────────────────────────────────────────────────────
    new_micro = None

    if ticker not in CONTEXT_ASSETS:
        if FULL_REBUILD:
            # Rebuild from all historical raw files
            new_micro = build_microstructure_from_scratch(clean_ticker)

        elif micro_start < fetch_end:
            # Gap-aware daily update:
            # Loop every business day in the gap. Use raw file from disk if it
            # exists (bulk download may have pre-populated it), otherwise call
            # Polygon API for that specific day and save the raw file.
            # This guarantees 100% coverage regardless of how many days were missed.
            print(f"   ⏳ [MICRO] Gap from {micro_start.date()} → {fetch_end.date()}...")

            gap_dates      = pd.date_range(micro_start.date(), fetch_end.date(), freq='B')
            all_trades     = []
            all_quotes     = []
            days_from_disk = 0
            days_from_api  = 0

            for gap_date in gap_dates:
                date_str = gap_date.strftime('%Y-%m-%d')
                t_fname  = f"{clean_ticker}_trades_{date_str}.parquet"
                q_fname  = f"{clean_ticker}_quotes_{date_str}.parquet"
                t_uri    = f"{GCS_RAW_DIR}/{clean_ticker}/{t_fname}"
                q_uri    = f"{GCS_RAW_DIR}/{clean_ticker}/{q_fname}"

                if _gcs_exists(t_uri) and _gcs_exists(q_uri):
                    # Raw files already in GCS — validate name, normalize and reuse
                    try:
                        if _valid_raw_file(t_fname, clean_ticker, 'trades') and \
                           _valid_raw_file(q_fname, clean_ticker, 'quotes'):
                            dt = normalize_raw_schema(pd.read_parquet(t_uri), 'trades')
                            dq = normalize_raw_schema(pd.read_parquet(q_uri), 'quotes')
                            if not dt.empty: all_trades.append(dt)
                            if not dq.empty: all_quotes.append(dq)
                            days_from_disk += 1
                        else:
                            # Misnamed files — fall through to API fetch
                            raise ValueError("misnamed file, fetching from API")
                    except Exception as e:
                        print(f"      ⚠️  {date_str}: GCS read error — {e}")
                else:
                    # Not in GCS — fetch from Polygon and upload
                    day_start = pd.Timestamp(gap_date, tz='UTC')
                    day_end   = day_start + timedelta(days=1)
                    try:
                        dt = fetch_polygon_trades(clean_ticker, day_start, min(day_end, fetch_end))
                        dq = fetch_polygon_quotes(clean_ticker, day_start, min(day_end, fetch_end))
                        if not dt.empty:
                            dt.to_parquet(t_uri, engine='pyarrow')
                            all_trades.append(dt)
                        if not dq.empty:
                            dq.to_parquet(q_uri, engine='pyarrow')
                            all_quotes.append(dq)
                        days_from_api += 1
                    except Exception as e:
                        print(f"      ⚠️  {date_str}: API error — {e}")

            print(f"      📥 {days_from_disk} days from disk / {days_from_api} days from API")

            raw_trades = pd.concat(all_trades).sort_values('timestamp') if all_trades else pd.DataFrame()
            raw_quotes = pd.concat(all_quotes).sort_values('timestamp') if all_quotes else pd.DataFrame()

            # Trim to micro_start to avoid reprocessing already-covered rows
            if not raw_trades.empty: raw_trades = raw_trades[raw_trades['timestamp'] >= micro_start]
            if not raw_quotes.empty: raw_quotes = raw_quotes[raw_quotes['timestamp'] >= micro_start]

            if not raw_trades.empty or not raw_quotes.empty:
                print("      🧮 Calculating microstructure...")
                new_micro = process_raw_to_microstructure(raw_trades, raw_quotes)
        else:
            print("   ✅ [MICRO] Already up to date.")

    # ── E. SAVE MICROSTRUCTURE ───────────────────────────────────────────────
    if new_micro is not None and not new_micro.empty:
        if not df_micro.empty:
            new_micro = new_micro[new_micro.index > df_micro.index.max()]

        if not new_micro.empty:
            df_micro = pd.concat([df_micro, new_micro]).sort_index()
            df_micro = df_micro[~df_micro.index.duplicated(keep='last')]
            df_micro.to_parquet(micro_uri, engine='pyarrow')
            print(f"      ✅ Saved {len(new_micro):,} new micro rows.")
        else:
            print("      ⏭️ Micro already up to date — skipped append.")

    # ── F. STITCH & SAVE MASTER INFERENCE FILE ───────────────────────────────
    print("      💾 Finalizing master inference file...")

    if ticker not in CONTEXT_ASSETS and not df_micro.empty:
        df_unified = df_ohlcv.join(df_micro, how='left')
    else:
        df_unified = df_ohlcv.copy()

    df_unified = heal_data(df_unified)
    df_unified = df_unified[df_unified.index.notna()].sort_index()

    df_unified.to_parquet(master_uri, engine='pyarrow')
    print(f"   ✅ Saved: {clean_ticker}_master_inference.parquet")

print("\n🎉 HARVEST COMPLETE.")
