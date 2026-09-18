"""
analyze_opening_clusters.py — Opening Price Path Clustering
============================================================
Extracts the normalized minute-by-minute price path for the first 60 minutes
after market open (6:30 AM PST) for ALL OHLCV assets in GCS.

Runs k-means clustering (k=2..6) on the path shapes to discover natural
groupings. For each k, reports:
  - Silhouette score (cluster quality)
  - Centroid paths (average shape per cluster)
  - Per-cluster stats: EOD PnL, dip depth, runup, which tickers dominate
  - Per-ticker cluster distribution (which assets belong to which pattern)
  - Sector breakdown

Designed to answer:
  1. How many natural opening patterns exist?
  2. What does each pattern look like?
  3. Which assets consistently show which pattern?
  4. Which assets are good training candidates for a pattern classifier?

Run in Colab or any environment with GCS access.
"""

import gc
import io
import ctypes
import warnings
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
        gc.collect()
except Exception:
    def _malloc_trim():
        gc.collect()

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import storage as gcs_storage

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "--quiet"])
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference"

PST = pytz.timezone('America/Los_Angeles')

MARKET_OPEN_MINUTE = 390      # 6:30 AM PST
SWING_WINDOW_MINUTES = 60     # first 60 minutes
PATH_RESAMPLE_MINUTES = 5     # resample path to 5-min intervals (12 points)
EOD_MINUTE = 780              # 1:00 PM PST

K_RANGE = range(2, 7)         # test k=2,3,4,5,6
N_KMEANS_INIT = 20
RANDOM_STATE = 42

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

EXCLUDE_TICKERS = {
    'SPY', 'VIX', 'TLT', 'IWM', 'SMH', 'ITA', 'XLE', 'BTCUSD',
    'QQQ', 'HYG', 'USO', 'SLV', 'MOD',
}

DATA_START_DATE = "2022-01-01"

MIN_DAYS_PER_TICKER = 50
MIN_CONSISTENCY_PCT = 0.40     # ticker must be >= 40% in one cluster to be "consistent"

SECTORS = {
    'crypto':      ['RIOT', 'CLSK', 'HUT', 'MARA', 'BTBT', 'CIFR', 'IREN', 'APLD',
                    'CORZ', 'COIN', 'MSTR', 'KEEL', 'WULF', 'HOOD'],
    'defense':     ['LUNR', 'RKLB', 'KTOS', 'AVAV', 'LMT', 'NOC', 'RTX', 'LHX',
                    'GD', 'BKSY', 'PL', 'RDW', 'MNTS', 'NNDM', 'SPIR'],
    'energy':      ['FCEL', 'PLUG', 'BE', 'FLNC', 'SMR', 'OKLO', 'CCJ', 'CEG',
                    'NEE', 'DUK', 'VST', 'GEV', 'NRG', 'TLN'],
    'space_evtol': ['ASTS', 'ACHR', 'JOBY', 'RCAT', 'BBAI', 'SMLR', 'VOYG'],
    'nuclear':     ['SMR', 'OKLO', 'CCJ', 'CEG', 'NXE', 'NBIS', 'BWXT'],
    'tech':        ['NVDA', 'AMD', 'PLTR', 'SMCI', 'TSLA', 'VRT', 'POWL',
                    'ALAB', 'MVIS', 'MU'],
}
SECTOR_LOOKUP = {}
for _sector, _tickers in SECTORS.items():
    for _t in _tickers:
        if _t not in SECTOR_LOOKUP:
            SECTOR_LOOKUP[_t] = _sector

W = 140

client = gcs_storage.Client()
bucket = client.bucket(BUCKET_NAME)


# ==============================================================================
# DISCOVER TICKERS
# ==============================================================================
def discover_tickers():
    blobs = bucket.list_blobs(prefix=f"{INFERENCE_DIR}/")
    tickers = set()
    for b in blobs:
        name = b.name.replace(f"{INFERENCE_DIR}/", "")
        if '/' in name:
            continue
        if name.endswith('_1m_full_historical_data.parquet'):
            t = name.replace('_1m_full_historical_data.parquet', '')
            tickers.add(t)
        elif name.endswith('_master_inference.parquet'):
            t = name.replace('_master_inference.parquet', '')
            tickers.add(t)
    tickers -= EXCLUDE_TICKERS
    return sorted(tickers)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_ohlcv(ticker):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    for suffix in ['_1m_full_historical_data', '_master_inference']:
        blob_path = f"{INFERENCE_DIR}/{file_ticker}{suffix}.parquet"
        blob = bucket.blob(blob_path)
        if blob.exists():
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if dt_cols:
                df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
            else:
                num_col = next((c for c in df.columns
                               if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                              df.columns[0])
                df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
            df = df.set_index('Datetime')
            df = df[~df.index.duplicated(keep='first')]
            std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                       'close': 'Close', 'volume': 'Volume'}
            lower_map = {k.lower(): v for k, v in std_map.items()}
            df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                          for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
            df = df[df.index.notna()].sort_index()
            cutoff = pd.Timestamp(DATA_START_DATE, tz='UTC')
            df = df[df.index >= cutoff]
            keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            return df[keep]
    return None


def trim_to_market_hours(df):
    """Keep only 6:00 AM - 2:00 PM PST bars. Avoids resampling non-market hours."""
    pst_idx = df.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute
    mask = (mins >= 360) & (mins <= 840)
    return df[mask].copy()


def heal_market_hours(df):
    """Ffill prices, zero-fill volume on already-trimmed market-hours data."""
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# EXTRACT NORMALIZED PRICE PATHS (returns compact arrays, not dicts)
# ==============================================================================
N_PATH_POINTS = SWING_WINDOW_MINUTES // PATH_RESAMPLE_MINUTES

def extract_daily_paths(df):
    """
    For each trading day, extract the normalized price path as % change from
    open price, resampled to PATH_RESAMPLE_MINUTES intervals.

    Returns: paths (N x 12 float32), eod_pnls (N float32), max_dds, max_rus, end_pcts
    """
    pst_idx = df.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute

    window_start = MARKET_OPEN_MINUTE
    window_end = MARKET_OPEN_MINUTE + SWING_WINDOW_MINUTES

    in_window = (mins >= window_start) & (mins <= window_end)
    in_eod = (mins >= EOD_MINUTE) & (df['Volume'] > 0)

    window_df = df[in_window]
    w_mins = mins[in_window]
    w_dates = pst_idx[in_window].date

    eod_df = df[in_eod]
    eod_dates = pst_idx[in_eod].date
    eod_closes = pd.Series(eod_df['Close'].values, index=eod_dates)
    eod_closes = eod_closes.groupby(eod_closes.index).first()

    close_vals = window_df['Close'].values
    vol_vals = window_df['Volume'].values
    minute_vals = w_mins.values if hasattr(w_mins, 'values') else np.array(w_mins)

    paths_list = []
    eod_list = []
    dd_list = []
    ru_list = []
    end_list = []
    date_list = []

    unique_dates = pd.unique(w_dates)
    date_arr = np.array(w_dates)

    for date in unique_dates:
        day_mask = date_arr == date
        day_close = close_vals[day_mask]
        day_vol = vol_vals[day_mask]
        day_mins = minute_vals[day_mask]

        real_mask = day_vol > 0
        if real_mask.sum() < 5:
            continue

        first_real = np.argmax(real_mask)
        open_price = float(day_close[first_real])
        if open_price <= 0:
            continue

        pct = (day_close - open_price) / open_price
        offsets = day_mins - window_start

        path = np.full(N_PATH_POINTS, np.nan, dtype=np.float32)
        for i in range(N_PATH_POINTS):
            b_start = i * PATH_RESAMPLE_MINUTES
            b_end = (i + 1) * PATH_RESAMPLE_MINUTES
            bmask = (offsets >= b_start) & (offsets < b_end)
            if bmask.any():
                path[i] = np.mean(pct[bmask])

        if np.isnan(path).sum() > N_PATH_POINTS // 3:
            continue
        for i in range(N_PATH_POINTS):
            if np.isnan(path[i]):
                path[i] = path[i-1] if i > 0 else 0.0

        eod_close = float(eod_closes.get(date, np.nan))
        eod_pnl = (eod_close - open_price) / open_price if not np.isnan(eod_close) else np.nan

        paths_list.append(path)
        eod_list.append(np.float32(eod_pnl))
        dd_list.append(np.float32(np.min(path)))
        ru_list.append(np.float32(np.max(path)))
        end_list.append(path[-1])
        date_list.append(date)

    if not paths_list:
        return None, None, None, None, None, None

    return (np.array(paths_list, dtype=np.float32),
            np.array(eod_list, dtype=np.float32),
            np.array(dd_list, dtype=np.float32),
            np.array(ru_list, dtype=np.float32),
            np.array(end_list, dtype=np.float32),
            np.array(date_list))


# ==============================================================================
# MAIN — COLLECT ALL PATHS
# ==============================================================================
print(f"{'='*W}")
print(f"  OPENING PRICE PATH CLUSTERING")
print(f"  Window: first {SWING_WINDOW_MINUTES} min after open, {PATH_RESAMPLE_MINUTES}-min resolution ({N_PATH_POINTS} points)")
print(f"  Data: {DATA_START_DATE} → present")
print(f"{'='*W}")
print()

print("Discovering tickers...")
all_tickers = discover_tickers()
print(f"  Found {len(all_tickers)} tickers")
print()

path_chunks = []
ticker_chunks = []
eod_chunks = []
dd_chunks = []
ru_chunks = []
end_chunks = []
date_chunks = []
ticker_day_counts = {}

for i, ticker in enumerate(all_tickers):
    try:
        df = load_ohlcv(ticker)
        if df is None:
            continue
        df = trim_to_market_hours(df)
        df = heal_market_hours(df)
        paths, eods, dds, rus, ends, dates = extract_daily_paths(df)
        del df
        _malloc_trim()

        if paths is None or len(paths) < MIN_DAYS_PER_TICKER:
            n = 0 if paths is None else len(paths)
            print(f"  [{i+1}/{len(all_tickers)}] {ticker}: {n} days (< {MIN_DAYS_PER_TICKER}, skip)")
            continue

        n_days = len(paths)
        path_chunks.append(paths)
        eod_chunks.append(eods)
        dd_chunks.append(dds)
        ru_chunks.append(rus)
        end_chunks.append(ends)
        date_chunks.append(dates)
        ticker_chunks.append(np.full(n_days, ticker))
        ticker_day_counts[ticker] = n_days
        print(f"  [{i+1}/{len(all_tickers)}] {ticker}: {n_days} days")
    except Exception as e:
        print(f"  [{i+1}/{len(all_tickers)}] {ticker}: ERROR — {e}")

path_matrix = np.concatenate(path_chunks, axis=0)
tickers_arr = np.concatenate(ticker_chunks)
eod_arr = np.concatenate(eod_chunks)
max_dd_arr = np.concatenate(dd_chunks)
max_ru_arr = np.concatenate(ru_chunks)
end_pct_arr = np.concatenate(end_chunks)
dates_arr = np.concatenate(date_chunks)
del path_chunks, ticker_chunks, eod_chunks, dd_chunks, ru_chunks, end_chunks, date_chunks
_malloc_trim()

print(f"\n  Total: {len(path_matrix)} day-paths from {len(ticker_day_counts)} tickers")


# ==============================================================================
# K-MEANS CLUSTERING — SWEEP k=2..6
# ==============================================================================
print()
print(f"{'='*W}")
print(f"  K-MEANS CLUSTERING (k={K_RANGE.start}..{K_RANGE.stop-1})")
print(f"{'='*W}")

scaler = StandardScaler()
X = scaler.fit_transform(path_matrix)

best_k = None
best_score = -1
km_results = {}

for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=N_KMEANS_INIT, random_state=RANDOM_STATE, max_iter=500)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(50000, len(X)))

    centroids_scaled = km.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)

    km_results[k] = {
        'labels': labels,
        'centroids': centroids,
        'silhouette': sil,
        'km': km,
    }

    if sil > best_score:
        best_score = sil
        best_k = k

    sizes = [np.sum(labels == c) for c in range(k)]
    pcts = [s / len(labels) for s in sizes]
    print(f"\n  k={k}  silhouette={sil:.4f}  {'<-- BEST' if k == best_k else ''}")
    print(f"    Cluster sizes: {', '.join(f'C{c}={sizes[c]} ({pcts[c]:.0%})' for c in range(k))}")

print(f"\n  Best k = {best_k} (silhouette = {best_score:.4f})")


# ==============================================================================
# DETAILED ANALYSIS FOR EACH K
# ==============================================================================
def ascii_path(path, width=50, height=7):
    """Render a price path as ASCII art."""
    mn = min(path)
    mx = max(path)
    rng = mx - mn if mx != mn else 0.01
    lines = []
    for row in range(height):
        level = mx - (row / (height - 1)) * rng
        line = []
        for val in path:
            if abs(val - level) <= rng / (height - 1) / 2:
                line.append('█')
            elif row == height - 1 and abs(val - mn) < rng / (height - 1):
                line.append('▄')
            elif row == 0 and abs(val - mx) < rng / (height - 1):
                line.append('▀')
            else:
                line.append(' ')
        label = f"{level:>+6.1%}"
        lines.append(f"      {label} │{''.join(line)}│")
    time_labels = ''.join(f'{i*PATH_RESAMPLE_MINUTES:>{width//N_PATH_POINTS}}' for i in range(N_PATH_POINTS))
    lines.append(f"             └{'─' * N_PATH_POINTS}┘")
    lines.append(f"              {time_labels}  (min from open)")
    return '\n'.join(lines)


def analyze_k(k):
    res = km_results[k]
    labels = res['labels']
    centroids = res['centroids']

    print()
    print(f"{'='*W}")
    print(f"  DETAILED ANALYSIS: k={k}  (silhouette={res['silhouette']:.4f})")
    print(f"{'='*W}")

    cluster_names = {}
    for c in range(k):
        path = centroids[c]
        trough_pos = np.argmin(path)
        peak_pos = np.argmax(path)
        end_val = path[-1]
        min_val = path.min()
        max_val = path.max()

        if min_val > -0.002 and max_val > 0.003:
            name = "UP"
        elif max_val < 0.002 and min_val < -0.003:
            name = "DOWN"
        elif trough_pos < peak_pos and min_val < -0.003:
            name = "DIP_FIRST"
        elif peak_pos < trough_pos and max_val > 0.003:
            name = "RIP_FIRST"
        elif abs(min_val) < 0.003 and abs(max_val) < 0.003:
            name = "FLAT"
        else:
            name = "MIXED"

        n_same = sum(1 for n in cluster_names.values() if n == name)
        if n_same > 0:
            name = f"{name}_{n_same+1}"
        cluster_names[c] = name

    for c in range(k):
        mask = labels == c
        n = mask.sum()
        pct = n / len(labels)
        name = cluster_names[c]
        path = centroids[c]

        c_eod = eod_arr[mask]
        c_dd = max_dd_arr[mask]
        c_ru = max_ru_arr[mask]
        c_end = end_pct_arr[mask]

        print(f"\n  {'─'*W}")
        print(f"  CLUSTER {c}: {name}  ({n} days, {pct:.0%} of total)")
        print(f"  {'─'*W}")

        print(f"    Centroid path (% from open):")
        pts = '  '.join(f"{path[i]:>+.2%}" for i in range(N_PATH_POINTS))
        print(f"      [{pts}]")
        print()
        print(ascii_path(path))
        print()

        valid_eod = c_eod[~np.isnan(c_eod)]
        print(f"    Stats:")
        print(f"      Max drawdown:   med={np.median(c_dd):>+.2%}  mean={np.mean(c_dd):>+.2%}")
        print(f"      Max runup:      med={np.median(c_ru):>+.2%}  mean={np.mean(c_ru):>+.2%}")
        print(f"      End of window:  med={np.median(c_end):>+.2%}  mean={np.mean(c_end):>+.2%}")
        if len(valid_eod) > 0:
            print(f"      EOD PnL:        med={np.median(valid_eod):>+.2%}  mean={np.mean(valid_eod):>+.2%}  "
                  f"win_rate={np.mean(valid_eod > 0):.0%}")

        c_tickers = tickers_arr[mask]
        ticker_counts = pd.Series(c_tickers).value_counts()
        top_tickers = ticker_counts.head(15)
        print(f"\n    Top tickers in this cluster:")
        print(f"      {'Ticker':<8} {'Days':>5}  {'% of ticker':>11}  {'Sector':<12}")
        for tk, cnt in top_tickers.items():
            tk_total = ticker_day_counts.get(tk, cnt)
            pct_of_tk = cnt / tk_total
            sec = SECTOR_LOOKUP.get(tk, 'other')
            print(f"      {tk:<8} {cnt:>5}  {pct_of_tk:>10.0%}  {sec:<12}")

    # ── PER-TICKER CLUSTER DISTRIBUTION ──────────────────────────────────────
    print()
    print(f"  {'='*W}")
    print(f"  PER-TICKER CLUSTER DISTRIBUTION (k={k})")
    print(f"  {'─'*W}")

    header_parts = [f"  {'Ticker':<8}  {'Sector':<12}  {'Days':>5}"]
    for c in range(k):
        header_parts.append(f"  {cluster_names[c]:>12}")
    header_parts.append(f"  {'Dominant':>12}  {'Consist':>7}")
    print(''.join(header_parts))
    print(f"  {'─'*W}")

    ticker_profiles = []
    for ticker in sorted(ticker_day_counts.keys()):
        tk_mask = tickers_arr == ticker
        tk_labels = labels[tk_mask]
        n_days = len(tk_labels)
        if n_days < MIN_DAYS_PER_TICKER:
            continue

        dist = {}
        for c in range(k):
            dist[c] = np.sum(tk_labels == c) / n_days

        dominant_c = max(dist, key=dist.get)
        consistency = dist[dominant_c]
        sec = SECTOR_LOOKUP.get(ticker, 'other')

        parts = [f"  {ticker:<8}  {sec:<12}  {n_days:>5}"]
        for c in range(k):
            parts.append(f"  {dist[c]:>11.0%}")
        parts.append(f"  {cluster_names[dominant_c]:>12}  {consistency:>6.0%}")
        print(''.join(parts))

        ticker_profiles.append({
            'ticker': ticker,
            'sector': sec,
            'n_days': n_days,
            'dominant_cluster': dominant_c,
            'dominant_name': cluster_names[dominant_c],
            'consistency': consistency,
            **{f'pct_C{c}': dist[c] for c in range(k)},
        })

    profiles_df = pd.DataFrame(ticker_profiles)

    # ── CONSISTENT TICKERS PER CLUSTER ───────────────────────────────────────
    print()
    print(f"  {'='*W}")
    print(f"  CONSISTENT TICKERS (>= {MIN_CONSISTENCY_PCT:.0%} in one cluster)")
    print(f"  {'─'*W}")

    for c in range(k):
        name = cluster_names[c]
        consistent = profiles_df[profiles_df[f'pct_C{c}'] >= MIN_CONSISTENCY_PCT].copy()
        consistent = consistent.sort_values(f'pct_C{c}', ascending=False)
        n_consist = len(consistent)
        print(f"\n    {name} ({n_consist} tickers):")
        if consistent.empty:
            print(f"      (none)")
        else:
            for _, row in consistent.iterrows():
                bar = '█' * int(row[f'pct_C{c}'] * 30)
                print(f"      {row['ticker']:<8} {row['sector']:<12} "
                      f"{row[f'pct_C{c}']:>5.0%}  {bar}")

    # ── SECTOR BREAKDOWN ─────────────────────────────────────────────────────
    print()
    print(f"  {'='*W}")
    print(f"  SECTOR × CLUSTER BREAKDOWN")
    print(f"  {'─'*W}")

    header = f"  {'Sector':<14}  {'Count':>5}"
    for c in range(k):
        header += f"  {cluster_names[c]:>12}"
    print(header)
    print(f"  {'─'*W}")

    for sec in sorted(profiles_df['sector'].unique()):
        sec_df = profiles_df[profiles_df['sector'] == sec]
        parts = [f"  {sec:<14}  {len(sec_df):>5}"]
        for c in range(k):
            avg_pct = sec_df[f'pct_C{c}'].mean()
            parts.append(f"  {avg_pct:>11.0%}")
        print(''.join(parts))

    # ── TRAINING ASSET RECOMMENDATIONS ───────────────────────────────────────
    print()
    print(f"  {'='*W}")
    print(f"  TRAINING ASSET RECOMMENDATIONS (k={k})")
    print(f"  {'─'*W}")
    print(f"  Assets with highest consistency per cluster, suitable as training teachers")
    print()

    for c in range(k):
        name = cluster_names[c]
        consistent = profiles_df[profiles_df[f'pct_C{c}'] >= MIN_CONSISTENCY_PCT].copy()
        consistent = consistent.sort_values(f'pct_C{c}', ascending=False)

        top = consistent.head(10)
        if top.empty:
            print(f"    {name}: no consistent assets (all < {MIN_CONSISTENCY_PCT:.0%})")
            continue

        c_mask = labels == c
        c_eod_valid = eod_arr[c_mask & ~np.isnan(eod_arr)]
        avg_eod = np.mean(c_eod_valid) if len(c_eod_valid) > 0 else np.nan

        print(f"    {name} (avg EOD PnL when in this cluster: {avg_eod:>+.2%}):")
        print(f"      {'Ticker':<8} {'Sector':<12} {'Consist':>7}  {'Days':>5}  {'Action'}")
        for _, row in top.iterrows():
            action = ""
            if 'DIP' in name:
                action = "LIMIT ENTRY candidate"
            elif 'UP' in name:
                action = "MARKET BUY at open"
            elif 'DOWN' in name:
                action = "AVOID"
            elif 'RIP' in name:
                action = "MARKET BUY at open (or skip)"
            print(f"      {row['ticker']:<8} {row['sector']:<12} "
                  f"{row[f'pct_C{c}']:>6.0%}  {row['n_days']:>5}  {action}")
        print()

    return profiles_df


# ==============================================================================
# RUN ANALYSIS FOR ALL K VALUES
# ==============================================================================
all_profiles = {}
for k in K_RANGE:
    profiles = analyze_k(k)
    all_profiles[k] = profiles


# ==============================================================================
# TEMPORAL DYNAMICS — HOW CLUSTERS SHIFT OVER TIME (best-k only)
# ==============================================================================
print()
print(f"{'='*W}")
print(f"  TEMPORAL DYNAMICS — CLUSTER FREQUENCY BY MONTH (k={best_k})")
print(f"{'='*W}")

best_res = km_results[best_k]
best_labels = best_res['labels']
best_centroids = best_res['centroids']

cluster_names_best = {}
for c in range(best_k):
    path = best_centroids[c]
    trough_pos = np.argmin(path)
    peak_pos = np.argmax(path)
    min_val = path.min()
    max_val = path.max()
    if min_val > -0.002 and max_val > 0.003:
        name = "UP"
    elif max_val < 0.002 and min_val < -0.003:
        name = "DOWN"
    elif trough_pos < peak_pos and min_val < -0.003:
        name = "DIP_FIRST"
    elif peak_pos < trough_pos and max_val > 0.003:
        name = "RIP_FIRST"
    elif abs(min_val) < 0.003 and abs(max_val) < 0.003:
        name = "FLAT"
    else:
        name = "MIXED"
    n_same = sum(1 for n in cluster_names_best.values() if n == name)
    if n_same > 0:
        name = f"{name}_{n_same+1}"
    cluster_names_best[c] = name

up_cluster = None
down_cluster = None
for c, n in cluster_names_best.items():
    if n == 'UP':
        up_cluster = c
    if n == 'DOWN':
        down_cluster = c

months_str = np.array([f"{d.year}-{d.month:02d}" for d in dates_arr])
unique_months = sorted(set(months_str))

# ── A. Aggregate: all tickers, cluster % by month ────────────────────────────
print()
print(f"  A) AGGREGATE CLUSTER MIX BY MONTH")
print(f"  {'─'*W}")

header = f"  {'Month':<8}  {'Days':>5}"
for c in range(best_k):
    header += f"  {cluster_names_best[c]:>8}"
if up_cluster is not None:
    header += f"  {'UP_EOD':>8}  {'UP bar'}"
print(header)
print(f"  {'─'*W}")

monthly_up_pct = {}
for month in unique_months:
    m_mask = months_str == month
    m_labels = best_labels[m_mask]
    n_month = len(m_labels)
    if n_month < 10:
        continue

    parts = [f"  {month:<8}  {n_month:>5}"]
    for c in range(best_k):
        pct = np.sum(m_labels == c) / n_month
        parts.append(f"  {pct:>7.0%}")

    if up_cluster is not None:
        up_mask = m_mask & (best_labels == up_cluster)
        up_pct = up_mask.sum() / n_month
        monthly_up_pct[month] = up_pct
        up_eod = eod_arr[up_mask]
        valid_up_eod = up_eod[~np.isnan(up_eod)]
        avg_eod = np.mean(valid_up_eod) if len(valid_up_eod) > 0 else np.nan
        eod_str = f"{avg_eod:>+7.1%}" if not np.isnan(avg_eod) else "    N/A"
        bar = '█' * int(up_pct * 60)
        parts.append(f"  {eod_str}  {bar}")

    print(''.join(parts))

if monthly_up_pct:
    pcts = list(monthly_up_pct.values())
    first_half = pcts[:len(pcts)//2]
    second_half = pcts[len(pcts)//2:]
    if first_half and second_half:
        avg_early = np.mean(first_half)
        avg_late = np.mean(second_half)
        print()
        print(f"  UP cluster trend:")
        print(f"    First half avg:  {avg_early:.1%}")
        print(f"    Second half avg: {avg_late:.1%}")
        change = avg_late - avg_early
        pct_change = change / avg_early * 100 if avg_early > 0 else 0
        direction = "INCREASING" if change > 0.005 else "DECREASING" if change < -0.005 else "STABLE"
        print(f"    Change: {change:>+.1%} ({pct_change:>+.0f}%)  →  {direction}")

# ── B. UP cluster quality over time (EOD PnL + win rate by month) ────────────
if up_cluster is not None:
    print()
    print(f"  {'='*W}")
    print(f"  B) UP CLUSTER QUALITY OVER TIME")
    print(f"  {'─'*W}")
    print(f"  {'Month':<8}  {'UP days':>7}  {'Med EOD':>8}  {'Mean EOD':>9}  {'WinRate':>7}  "
          f"{'Med Amp':>8}  {'Quality bar'}")
    print(f"  {'─'*W}")

    for month in unique_months:
        m_mask = months_str == month
        up_mask = m_mask & (best_labels == up_cluster)
        n_up = up_mask.sum()
        if n_up < 3:
            continue

        up_eod = eod_arr[up_mask]
        valid = up_eod[~np.isnan(up_eod)]
        if len(valid) == 0:
            continue

        med_eod = np.median(valid)
        mean_eod = np.mean(valid)
        wr = np.mean(valid > 0)
        up_ru = max_ru_arr[up_mask]
        med_amp = np.median(up_ru)

        bar_len = max(0, int(med_eod * 200))
        bar = '█' * min(bar_len, 50)
        print(f"  {month:<8}  {n_up:>7}  {med_eod:>+7.1%}  {mean_eod:>+8.1%}  "
              f"{wr:>6.0%}  {med_amp:>+7.1%}  {bar}")

# ── C. DOWN cluster severity over time ───────────────────────────────────────
if down_cluster is not None:
    print()
    print(f"  {'='*W}")
    print(f"  C) DOWN CLUSTER SEVERITY OVER TIME")
    print(f"  {'─'*W}")
    print(f"  {'Month':<8}  {'DN days':>7}  {'Med EOD':>8}  {'Mean EOD':>9}  {'Med DD':>8}  "
          f"{'Severity bar'}")
    print(f"  {'─'*W}")

    for month in unique_months:
        m_mask = months_str == month
        dn_mask = m_mask & (best_labels == down_cluster)
        n_dn = dn_mask.sum()
        if n_dn < 3:
            continue

        dn_eod = eod_arr[dn_mask]
        valid = dn_eod[~np.isnan(dn_eod)]
        if len(valid) == 0:
            continue

        med_eod = np.median(valid)
        mean_eod = np.mean(valid)
        dn_dd = max_dd_arr[dn_mask]
        med_dd = np.median(dn_dd)

        bar_len = max(0, int(abs(med_eod) * 200))
        bar = '▓' * min(bar_len, 50)
        print(f"  {month:<8}  {n_dn:>7}  {med_eod:>+7.1%}  {mean_eod:>+8.1%}  "
              f"{med_dd:>+7.1%}  {bar}")

# ── D. Per-ticker UP% trend (deployed + high-swing tickers) ──────────────────
if up_cluster is not None:
    print()
    print(f"  {'='*W}")
    print(f"  D) PER-TICKER UP CLUSTER % BY QUARTER")
    print(f"  {'─'*W}")

    quarters_str = np.array([f"{d.year}Q{(d.month-1)//3+1}" for d in dates_arr])
    unique_quarters = sorted(set(quarters_str))

    deployed_tickers = {'ALAB', 'BE', 'OKLO', 'RKLB', 'PL', 'RDW', 'SMCI'}
    high_swing = set()
    for tk, cnt in ticker_day_counts.items():
        tk_mask = tickers_arr == tk
        tk_up = (best_labels[tk_mask] == up_cluster).sum()
        if tk_up / cnt >= 0.18:
            high_swing.add(tk)

    show_tickers = sorted(deployed_tickers | high_swing)

    header = f"  {'Ticker':<8}  {'Sector':<12}"
    for q in unique_quarters:
        header += f"  {q:>7}"
    header += f"  {'Trend'}"
    print(header)
    print(f"  {'─'*W}")

    for ticker in show_tickers:
        tk_mask = tickers_arr == ticker
        if tk_mask.sum() == 0:
            continue
        sec = SECTOR_LOOKUP.get(ticker, 'other')
        parts = [f"  {ticker:<8}  {sec:<12}"]

        qtr_pcts = []
        for q in unique_quarters:
            q_mask = tk_mask & (quarters_str == q)
            n_q = q_mask.sum()
            if n_q < 10:
                parts.append(f"  {'—':>7}")
                continue
            up_pct = (best_labels[q_mask] == up_cluster).sum() / n_q
            qtr_pcts.append(up_pct)
            parts.append(f"  {up_pct:>6.0%}")

        if len(qtr_pcts) >= 4:
            first = np.mean(qtr_pcts[:len(qtr_pcts)//2])
            last = np.mean(qtr_pcts[len(qtr_pcts)//2:])
            delta = last - first
            if delta > 0.03:
                trend = f"  ↑ +{delta:.0%}"
            elif delta < -0.03:
                trend = f"  ↓ {delta:+.0%}"
            else:
                trend = f"  → {delta:+.0%}"
            parts.append(trend)

        print(''.join(parts))

# ── E. Recent 6-month snapshot vs historical ─────────────────────────────────
if up_cluster is not None:
    print()
    print(f"  {'='*W}")
    print(f"  E) RECENT 6 MONTHS vs HISTORICAL — UP CLUSTER RATE")
    print(f"  {'─'*W}")

    from datetime import timedelta as _td
    max_date = max(dates_arr)
    cutoff_date = datetime(max_date.year, max_date.month, max_date.day) - _td(days=180)
    cutoff_d = cutoff_date.date()

    recent_mask = np.array([d >= cutoff_d for d in dates_arr])
    hist_mask = ~recent_mask

    print(f"  {'Ticker':<8} {'Sector':<12}  {'Hist UP%':>8}  {'Recent UP%':>10}  "
          f"{'Delta':>7}  {'Hist EOD':>8}  {'Rec EOD':>8}  {'Signal'}")
    print(f"  {'─'*W}")

    for ticker in sorted(ticker_day_counts.keys()):
        tk_mask = tickers_arr == ticker
        n_hist = (tk_mask & hist_mask).sum()
        n_recent = (tk_mask & recent_mask).sum()
        if n_hist < 50 or n_recent < 20:
            continue

        hist_idx = tk_mask & hist_mask
        recent_idx = tk_mask & recent_mask
        hist_up = (best_labels[hist_idx] == up_cluster).sum() / n_hist
        recent_up = (best_labels[recent_idx] == up_cluster).sum() / n_recent
        delta = recent_up - hist_up

        hist_up_idx = hist_idx & (best_labels == up_cluster)
        hist_eod_vals = eod_arr[hist_up_idx]
        hist_eod_valid = hist_eod_vals[~np.isnan(hist_eod_vals)]
        hist_eod = np.median(hist_eod_valid) if len(hist_eod_valid) > 0 else np.nan

        rec_up_idx = recent_idx & (best_labels == up_cluster)
        rec_eod_vals = eod_arr[rec_up_idx]
        rec_eod_valid = rec_eod_vals[~np.isnan(rec_eod_vals)]
        rec_eod = np.median(rec_eod_valid) if len(rec_eod_valid) > 0 else np.nan

        signal = ""
        if delta > 0.05:
            signal = "MORE UP days"
        elif delta < -0.05:
            signal = "FEWER UP days"
        if not np.isnan(rec_eod) and not np.isnan(hist_eod) and rec_eod < hist_eod * 0.5:
            signal += " + WEAKER spikes" if signal else "WEAKER spikes"

        sec = SECTOR_LOOKUP.get(ticker, 'other')
        hist_eod_str = f"{hist_eod:>+7.1%}" if not np.isnan(hist_eod) else "    N/A"
        rec_eod_str = f"{rec_eod:>+7.1%}" if not np.isnan(rec_eod) else "    N/A"
        print(f"  {ticker:<8} {sec:<12}  {hist_up:>7.0%}  {recent_up:>9.0%}  "
              f"{delta:>+6.0%}  {hist_eod_str}  {rec_eod_str}  {signal}")


# ==============================================================================
# FINAL SUMMARY — RECOMMENDED K
# ==============================================================================
print()
print(f"{'='*W}")
print(f"  SUMMARY — OPTIMAL CLUSTER COUNT")
print(f"{'='*W}")
print()
print(f"  {'k':>3}  {'Silhouette':>10}  {'Notes'}")
print(f"  {'─'*60}")
for k in K_RANGE:
    sil = km_results[k]['silhouette']
    marker = " <-- BEST" if k == best_k else ""
    labels = km_results[k]['labels']
    sizes = [np.sum(labels == c) for c in range(k)]
    min_pct = min(s / len(labels) for s in sizes)
    notes = f"smallest cluster = {min_pct:.0%}"
    if min_pct < 0.10:
        notes += " (very small — may be noise)"
    print(f"  {k:>3}  {sil:>10.4f}  {notes}{marker}")

print()
print(f"  Recommendation: use k={best_k}")
print(f"  Silhouette score measures cluster separation (higher = more distinct)")
print()

best_labels = km_results[best_k]['labels']
best_centroids = km_results[best_k]['centroids']
print(f"  Best-k={best_k} centroid summary:")
for c in range(best_k):
    path = best_centroids[c]
    trough = np.argmin(path)
    peak = np.argmax(path)
    print(f"    C{c}: min={path.min():>+.2%} @ +{trough*PATH_RESAMPLE_MINUTES}min  "
          f"max={path.max():>+.2%} @ +{peak*PATH_RESAMPLE_MINUTES}min  "
          f"end={path[-1]:>+.2%}")

print()
print(f"{'='*W}")
print(f"  Done. {len(path_matrix)} day-paths from {len(ticker_day_counts)} tickers clustered.")
print(f"{'='*W}")
