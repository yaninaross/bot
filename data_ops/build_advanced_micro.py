# %% ── Build Advanced Microstructure Features ─────────────────────────────────
#
# Computes 12 new features from raw trades + quotes (Lee-Ready signed):
#
# Lee-Ready Order Flow (4):
#   1. LR_OFI           — (buy_vol - sell_vol) / total_vol (signed, Lee-Ready)
#   2. LR_VPIN          — |buy_vol - sell_vol| / total_vol (unsigned pressure)
#   3. LR_Buy_Ratio     — n_buys / (n_buys + n_sells) (count-based imbalance)
#   4. LR_Size_Imbalance — (avg_buy_size - avg_sell_size) / (avg_buy + avg_sell)
#
# Information Theory (2):
#   5. PE               — Permutation Entropy of trade-sign sequence (m=5)
#   6. LZC              — Lempel-Ziv Complexity of binary buy/sell sequence
#
# Tick-Level Dynamics (2):
#   7. Tick_Volatility   — std of trade-to-trade returns (intra-bar micro-vol)
#   8. Shock_Persist     — Lag-1 autocorrelation of tick returns
#
# Trade Distribution (2):
#   9. Trade_Size_CV     — std(size)/mean(size) — algo vs organic detection
#  10. Trade_Intensity   — trades per second (urgency measure)
#
# Quote Dynamics (2):
#  11. Quote_Intensity   — quote updates per minute (market maker activity)
#  12. Spread_CV         — std(spread)/mean(spread) within bar
#
# Reads:  gs://{GCS_BUCKET}/Data/TradesQuotes/{ticker}/{ticker}_trades_{date}.parquet
#         gs://{GCS_BUCKET}/Data/TradesQuotes/{ticker}/{ticker}_quotes_{date}.parquet
#
# Writes: gs://{GCS_BUCKET}/Data/Inference/Microstructure_Advanced/{ticker}_1m_advanced_micro.parquet
#
# Incremental: detects existing output, processes only new dates.
# Colab-ready, config vars at top.

import os, sys, time, gc, ctypes, warnings
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta
from math import log2, factorial
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-cloud-storage", "pyarrow", "--quiet"])
    from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

import matplotlib
try:
    from google.colab import auth as _
    _IN_COLAB = True
except ImportError:
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
GCS_TQ_DIR  = 'Data/TradesQuotes'
GCS_OUT_DIR = 'Data/Inference/Microstructure_Advanced'

DATA_START = '2024-08-01'

# Set to None to process all tickers discovered from GCS,
# or list specific tickers to process
TARGET_TICKERS = None  # None = auto-discover from GCS

PE_ORDER = 5             # permutation entropy embedding dimension
MIN_TRADES_PER_BAR = 6   # bars with fewer trades get NaN (filled later)

DAY_WORKERS = 4          # concurrent day downloads per ticker
TICKER_BATCH_SIZE = 5    # tickers to process before gc

ADVANCED_MICRO_COLS = [
    'LR_OFI', 'LR_VPIN', 'LR_Buy_Ratio', 'LR_Size_Imbalance',
    'PE', 'LZC',
    'Tick_Volatility', 'Shock_Persist',
    'Trade_Size_CV', 'Trade_Intensity',
    'Quote_Intensity', 'Spread_CV',
]

gcs_client = gcs_storage.Client()
bucket = gcs_client.bucket(BUCKET_NAME)


def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ==============================================================================
# GCS HELPERS
# ==============================================================================
def _download_to_tmp(blob_path):
    local = f"/tmp/_adv_micro_{os.path.basename(blob_path)}"
    bucket.blob(blob_path).download_to_filename(local)
    return local


def _read_parquet_gcs(blob_path):
    try:
        local = _download_to_tmp(blob_path)
        df = pd.read_parquet(local, engine='pyarrow')
        os.remove(local)
        return df
    except Exception:
        return None


def _save_parquet_gcs(df, blob_path):
    local = f"/tmp/_adv_micro_{os.path.basename(blob_path)}"
    df.to_parquet(local, engine='pyarrow')
    bucket.blob(blob_path).upload_from_filename(local)
    size_mb = os.path.getsize(local) / 1e6
    os.remove(local)
    return size_mb


# ==============================================================================
# TICKER DISCOVERY
# ==============================================================================
def discover_tickers():
    """Find all tickers that have TradesQuotes data on GCS."""
    seen = set()
    tickers = []
    prefix = f'{GCS_TQ_DIR}/'
    iterator = bucket.list_blobs(prefix=prefix, delimiter='/')
    for blob in iterator:
        pass  # must consume to populate .prefixes
    for p in iterator.prefixes:
        ticker = p.replace(prefix, '').rstrip('/')
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return sorted(tickers)


def list_trade_dates(ticker, start_date):
    """List all available trade dates for a ticker from GCS."""
    prefix = f'{GCS_TQ_DIR}/{ticker}/{ticker}_trades_'
    dates = []
    for blob in bucket.list_blobs(prefix=prefix):
        fname = blob.name.split('/')[-1]
        if not fname.endswith('.parquet'):
            continue
        date_str = fname.replace(f'{ticker}_trades_', '').replace('.parquet', '')
        if date_str >= start_date:
            dates.append(date_str)
    return sorted(dates)


# ==============================================================================
# LEE-READY TRADE SIGNING
# ==============================================================================
def lee_ready_sign(trades_df, quotes_df):
    """
    Sign each trade as buy (+1) or sell (-1) using Lee-Ready algorithm:
    - Compare trade price to quote midpoint at time of trade
    - If price > mid: buy. If price < mid: sell.
    - If price == mid: use tick rule (compare to previous trade price)
    """
    if trades_df.empty or quotes_df.empty:
        return trades_df.assign(sign=np.nan, tick_return=np.nan)

    trades = trades_df.sort_values('timestamp').copy()
    quotes = quotes_df[['timestamp', 'bid_price', 'ask_price']].sort_values('timestamp').copy()

    signed = pd.merge_asof(
        trades, quotes, on='timestamp', direction='backward'
    )

    mid = (signed['bid_price'] + signed['ask_price']) / 2
    signed['sign'] = np.where(
        signed['price'] > mid, 1.0,
        np.where(signed['price'] < mid, -1.0, 0.0)
    )

    zero_mask = signed['sign'] == 0.0
    if zero_mask.any():
        tick_sign = np.sign(signed['price'].diff())
        signed.loc[zero_mask, 'sign'] = tick_sign[zero_mask]

    signed['sign'] = signed['sign'].replace(0.0, np.nan).ffill().fillna(1.0)
    signed['tick_return'] = signed['price'].pct_change().fillna(0.0)

    return signed


# ==============================================================================
# FEATURE COMPUTATIONS
# ==============================================================================
def permutation_entropy(signs, m=PE_ORDER):
    """
    Normalized permutation entropy of a sequence.
    Returns value in [0, 1]. Low = predictable/coordinated, High = random.
    """
    n = len(signs)
    if n < m + 1:
        return np.nan

    patterns = []
    for i in range(n - m + 1):
        window = signs[i:i + m]
        pattern = tuple(np.argsort(np.argsort(window)))
        patterns.append(pattern)

    counts = Counter(patterns)
    total = len(patterns)
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = log2(factorial(m))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def lempel_ziv_complexity(binary_seq):
    """
    Normalized Lempel-Ziv complexity (LZ76) of a binary sequence.
    Input: array-like of 0/1 values.
    Returns value in (0, ~1]. Low = repetitive, High = complex/random.
    """
    n = len(binary_seq)
    if n < 2:
        return np.nan

    s = ''.join(str(int(b)) for b in binary_seq)
    sub_strings = set()
    c = 0
    start = 0

    for i in range(n):
        sub = s[start:i + 1]
        if sub not in sub_strings:
            sub_strings.add(sub)
            c += 1
            start = i + 1

    if start < n:
        c += 1

    norm = n / log2(n) if n > 1 else 1.0
    return c / norm


def compute_bar_features(signed_trades_in_bar, quotes_in_bar):
    """
    Compute all 12 features for trades within a single 1-minute bar.
    """
    df = signed_trades_in_bar
    n = len(df)

    if n < MIN_TRADES_PER_BAR:
        return {col: np.nan for col in ADVANCED_MICRO_COLS}

    signs = df['sign'].values
    sizes = df['size'].values.astype(np.float64)
    tick_rets = df['tick_return'].values

    buy_mask = signs > 0
    sell_mask = signs < 0

    n_buys = buy_mask.sum()
    n_sells = sell_mask.sum()
    buy_vol = sizes[buy_mask].sum()
    sell_vol = sizes[sell_mask].sum()
    total_vol = buy_vol + sell_vol

    # ── Lee-Ready Order Flow (4) ──
    lr_ofi = (buy_vol - sell_vol) / (total_vol + 1e-9)
    lr_vpin = abs(buy_vol - sell_vol) / (total_vol + 1e-9)
    lr_buy_ratio = n_buys / (n_buys + n_sells + 1e-9)

    avg_buy_size = sizes[buy_mask].mean() if n_buys > 0 else 0.0
    avg_sell_size = sizes[sell_mask].mean() if n_sells > 0 else 0.0
    lr_size_imb = (avg_buy_size - avg_sell_size) / (avg_buy_size + avg_sell_size + 1e-9)

    # ── Information Theory (2) ──
    pe = permutation_entropy(signs, PE_ORDER)
    binary = (signs > 0).astype(int)
    lzc = lempel_ziv_complexity(binary)

    # ── Tick-Level Dynamics (2) ──
    tick_vol = np.std(tick_rets[1:]) if n > 2 else 0.0

    shock_persist = 0.0
    if n > 3:
        rets = tick_rets[1:]
        if len(rets) > 2 and np.std(rets) > 1e-10:
            x, y = rets[:-1], rets[1:]
            mx, my = x.mean(), y.mean()
            sx, sy = x.std(), y.std()
            if sx > 1e-10 and sy > 1e-10:
                shock_persist = float(np.mean((x - mx) * (y - my)) / (sx * sy))

    # ── Trade Distribution (2) ──
    trade_size_cv = float(np.std(sizes) / (np.mean(sizes) + 1e-9))

    ts_range = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
    trade_intensity = n / (ts_range + 1e-9) if ts_range > 0 else float(n)

    # ── Quote Dynamics (2) ──
    n_quotes = len(quotes_in_bar) if quotes_in_bar is not None else 0
    quote_intensity = float(n_quotes)

    spread_cv = 0.0
    if quotes_in_bar is not None and len(quotes_in_bar) > 1:
        spreads = (quotes_in_bar['ask_price'] - quotes_in_bar['bid_price']).values
        spread_mean = np.mean(spreads)
        if spread_mean > 1e-8:
            spread_cv = float(np.std(spreads) / spread_mean)

    return {
        'LR_OFI': lr_ofi,
        'LR_VPIN': lr_vpin,
        'LR_Buy_Ratio': lr_buy_ratio,
        'LR_Size_Imbalance': lr_size_imb,
        'PE': pe,
        'LZC': lzc,
        'Tick_Volatility': tick_vol,
        'Shock_Persist': shock_persist,
        'Trade_Size_CV': trade_size_cv,
        'Trade_Intensity': trade_intensity,
        'Quote_Intensity': quote_intensity,
        'Spread_CV': spread_cv,
    }


# ==============================================================================
# PROCESS ONE DAY
# ==============================================================================
def process_one_day(ticker, date_str):
    """
    Load trades + quotes for one ticker-day, compute advanced micro features.
    Returns DataFrame with 1-min index and 12 feature columns, or None on failure.
    """
    t_blob = f'{GCS_TQ_DIR}/{ticker}/{ticker}_trades_{date_str}.parquet'
    q_blob = f'{GCS_TQ_DIR}/{ticker}/{ticker}_quotes_{date_str}.parquet'

    trades_df = _read_parquet_gcs(t_blob)
    quotes_df = _read_parquet_gcs(q_blob)

    if trades_df is None or trades_df.empty:
        return None
    if quotes_df is None or quotes_df.empty:
        return None

    for df in [trades_df, quotes_df]:
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            elif df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

    signed = lee_ready_sign(trades_df, quotes_df)
    del trades_df

    if signed.empty:
        return None

    signed['bar'] = signed['timestamp'].dt.floor('1min')
    quotes_df['bar'] = quotes_df['timestamp'].dt.floor('1min')

    quotes_grouped = {ts: grp for ts, grp in quotes_df.groupby('bar')}
    del quotes_df

    results = []
    for bar_ts, trade_group in signed.groupby('bar'):
        q_group = quotes_grouped.get(bar_ts)
        feats = compute_bar_features(trade_group, q_group)
        feats['timestamp'] = bar_ts
        results.append(feats)

    del signed, quotes_grouped

    if not results:
        return None

    bar_df = pd.DataFrame(results)
    bar_df = bar_df.set_index('timestamp').sort_index()
    return bar_df


# ==============================================================================
# PROCESS ONE TICKER (all dates)
# ==============================================================================
def process_ticker(ticker, dates_to_process):
    """Process all dates for one ticker, return concatenated DataFrame."""
    if not dates_to_process:
        return None

    all_bars = []
    errors = 0

    def _process_date(date_str):
        try:
            return process_one_day(ticker, date_str)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=DAY_WORKERS) as pool:
        futures = {pool.submit(_process_date, d): d for d in dates_to_process}
        for future in as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                all_bars.append(result)
            elif result is None:
                errors += 1

    if not all_bars:
        return None

    combined = pd.concat(all_bars).sort_index()
    combined = combined[~combined.index.duplicated(keep='last')]

    combined = combined.ffill().fillna(0.0)

    return combined


# ==============================================================================
# MAIN
# ==============================================================================
t_total = time.time()

print(f"{'=' * 80}")
print(f"  BUILD ADVANCED MICROSTRUCTURE FEATURES (12 columns)")
print(f"  Features: {', '.join(ADVANCED_MICRO_COLS)}")
print(f"  Data start: {DATA_START}")
print(f"  Output: gs://{BUCKET_NAME}/{GCS_OUT_DIR}/")
print(f"{'=' * 80}")

# -- Discover tickers --
if TARGET_TICKERS is None:
    print(f"\n  Discovering tickers from GCS TradesQuotes...")
    all_tickers = discover_tickers()
    print(f"  Found {len(all_tickers)} tickers")
else:
    all_tickers = sorted(TARGET_TICKERS)
    print(f"\n  Processing {len(all_tickers)} specified tickers")

# -- Process each ticker --
total_success = 0
total_skip = 0
total_fail = 0
ticker_stats = []

for ti, ticker in enumerate(all_tickers):
    t0 = time.time()
    print(f"\n{'─' * 60}")
    print(f"  [{ti+1}/{len(all_tickers)}] {ticker}")

    out_blob = f'{GCS_OUT_DIR}/{ticker}_1m_advanced_micro.parquet'
    existing_df = None
    existing_max_date = None

    try:
        existing_df = _read_parquet_gcs(out_blob)
        if existing_df is not None and not existing_df.empty:
            if existing_df.index.tz is None:
                existing_df.index = existing_df.index.tz_localize('UTC')
            existing_max_date = existing_df.index.max().strftime('%Y-%m-%d')
            print(f"    Existing: {len(existing_df):,} rows through {existing_max_date}")
    except Exception:
        pass

    all_dates = list_trade_dates(ticker, DATA_START)
    if not all_dates:
        print(f"    No trade files from {DATA_START} — skip")
        total_skip += 1
        continue

    if existing_max_date:
        dates_to_process = [d for d in all_dates if d > existing_max_date]
    else:
        dates_to_process = all_dates

    if not dates_to_process:
        print(f"    Already up to date — skip")
        total_skip += 1
        continue

    print(f"    Processing {len(dates_to_process)} dates "
          f"({dates_to_process[0]} → {dates_to_process[-1]})")

    new_df = process_ticker(ticker, dates_to_process)

    if new_df is None or new_df.empty:
        print(f"    No valid data produced — skip")
        total_fail += 1
        continue

    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]
    else:
        combined = new_df

    del existing_df, new_df

    size_mb = _save_parquet_gcs(combined, out_blob)
    elapsed = time.time() - t0

    print(f"    ✅ {len(combined):,} rows, {size_mb:.1f} MB ({elapsed:.0f}s)")
    ticker_stats.append({
        'ticker': ticker, 'rows': len(combined),
        'dates': len(dates_to_process), 'time_s': elapsed
    })
    total_success += 1

    del combined
    if (ti + 1) % TICKER_BATCH_SIZE == 0:
        _release_memory()

total_elapsed = time.time() - t_total


# ==============================================================================
# SUMMARY CHART
# ==============================================================================
if ticker_stats:
    stats_df = pd.DataFrame(ticker_stats)
    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, len(stats_df) * 0.25)))

    axes[0].barh(stats_df['ticker'], stats_df['rows'], color='#4a90d9')
    axes[0].set_xlabel('Rows')
    axes[0].set_title('Advanced Micro Rows per Ticker')
    axes[0].invert_yaxis()

    axes[1].barh(stats_df['ticker'], stats_df['time_s'], color='#e8a838')
    axes[1].set_xlabel('Seconds')
    axes[1].set_title('Processing Time per Ticker')
    axes[1].invert_yaxis()

    axes[2].barh(stats_df['ticker'], stats_df['dates'], color='#50b848')
    axes[2].set_xlabel('Dates')
    axes[2].set_title('Dates Processed')
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig('/tmp/advanced_micro_summary.png', dpi=100, bbox_inches='tight')
    plt.show()
    print(f"\n  Chart saved: /tmp/advanced_micro_summary.png")

print(f"\n{'=' * 80}")
print(f"  ADVANCED MICROSTRUCTURE BUILD COMPLETE")
print(f"  Features: {len(ADVANCED_MICRO_COLS)} columns")
print(f"  {', '.join(ADVANCED_MICRO_COLS)}")
print(f"  Success: {total_success}  |  Skipped: {total_skip}  |  Failed: {total_fail}")
print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"  Output: gs://{BUCKET_NAME}/{GCS_OUT_DIR}/")
print(f"{'=' * 80}")
