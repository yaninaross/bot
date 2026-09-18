"""
analyze_new_asset_spikes.py — New Asset Spike Screener (1-min Intraday)
=======================================================================
Pulls 1-min OHLCV from Polygon for ~70 candidate tickers across 5 sectors.
Measures TRUE intraday spike: Close at market open bar → max High through close.

Spike definition (matches training label):
  For each trading day, at the ENTRY_MINUTE bar (default 6:15 PT / 9:15 ET):
    spike = (max_high from entry to 13:00 PT) / close_at_entry - 1

This is the same measurement the model uses for spike probability labels.

Windows:
  - Full 12 months (Jul 2025 → Jul 2026)
  - Recent 3 months (Apr 2026 → Jul 2026) — focus window

1-min parquets cached to /tmp/spike_screener_1m/ per ticker (one file each).
Re-runs skip already-downloaded tickers.

Colab-ready (config vars, no if __name__).
"""

import time, gc, ctypes, sys, subprocess, warnings, os
from collections import defaultdict
from dateutil.relativedelta import relativedelta
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

try:
    from polygon import RESTClient
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "polygon-api-client", "--quiet", "--break-system-packages"])
    from polygon import RESTClient

try:
    import matplotlib
    try:
        from google.colab import auth
        _IN_COLAB = True
    except ImportError:
        matplotlib.use('Agg')
        _IN_COLAB = False
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if _IN_COLAB:
    try:
        auth.authenticate_user()
    except Exception:
        pass


# ==============================================================================
# CONFIG
# ==============================================================================
POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
CACHE_DIR = '/tmp/spike_screener_1m'
os.makedirs(CACHE_DIR, exist_ok=True)

DATA_START = '2025-07-01'
DATA_END   = '2026-07-10'

RECENT_START = '2026-04-01'

ENTRY_MINUTE  = 375    # 6:15 PT = 6*60+15
EOD_MINUTE    = 780    # 13:00 PT
MARKET_OPEN   = 390    # 6:30 PT (regular open)

SPIKE_THRESHOLDS = [0.03, 0.05, 0.08, 0.10]

MIN_SPIKE_10PCT_PER_MONTH = 2.0
MIN_SPIKE_8PCT_PER_MONTH  = 3.0

W = 120

# ==============================================================================
# TICKER LISTS BY SECTOR
# ==============================================================================
CANDIDATE_SECTORS = {
    'Crypto Mining (Baseline)': {
        'stocks': ['HUT', 'CIFR', 'CLSK', 'RIOT', 'WULF', 'MARA', 'KEEL',
                   'APLD', 'BTBT', 'CORZ', 'IREN'],
        'is_baseline': True,
    },
    'Adv Semiconductor & AI Hardware': {
        'stocks': ['AMAT', 'KLAC', 'ALAB', 'ICHR', 'ACMR', 'MXL', 'AEHR',
                   'LASR', 'NVDA', 'AMD', 'ARM', 'LSCC', 'RMBS',
                   'CRDO', 'MU', 'UCTT', 'SITM'],
    },
    'High-Beta Biotech & Genomics': {
        'stocks': ['APGE', 'MRNA', 'TWST', 'SLS', 'INNV', 'BBIO', 'TXG',
                   'CRSP', 'BEAM', 'NTLA', 'IONS', 'VKTX', 'CYTK'],
    },
    'Utilities / Nuclear / Smart Grid': {
        'stocks': ['BE', 'WOLF', 'PLUG', 'FCEL', 'OKLO', 'NNE',
                   'VST', 'FRVO', 'NEOV', 'SMR', 'CCJ', 'GEV', 'POWL'],
    },
    'Retail Squeeze & Small-Cap Growth': {
        'stocks': ['TDUP', 'OPEN', 'RIVN', 'SOFI', 'RRGB', 'CBRL',
                   'MYE', 'ATEX', 'ONDS', 'UPST'],
    },
    'Cyclical Materials & Consumer': {
        'stocks': ['AA', 'X', 'LAC', 'MP', 'CHWY'],
    },
}

ALL_STOCKS = []
STOCK_SECTOR = {}
for sector, cfg in CANDIDATE_SECTORS.items():
    for s in cfg['stocks']:
        if s not in STOCK_SECTOR:
            ALL_STOCKS.append(s)
            STOCK_SECTOR[s] = sector


# ==============================================================================
# 1-MIN DATA LOADING — GCS first, Polygon fallback, local cache
# ==============================================================================
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
TICKER_FILE_MAP = {'BITF': 'KEEL'}

from google.cloud import storage as gcs_storage
try:
    from google.colab import auth as _colab_auth
    _colab_auth.authenticate_user()
except Exception:
    pass
_gcs_client = gcs_storage.Client()
_gcs_bucket = _gcs_client.bucket(BUCKET_NAME)

poly_client = RESTClient(POLYGON_API_KEY)


def _load_gcs_parquet(gcs_uri):
    df = pd.read_parquet(gcs_uri)
    df = df.reset_index()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    else:
        num_col = next((c for c in df.columns if c in ('Timestamp', 'Date', 'Time')), df.columns[0])
        df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
    df = df.set_index('Datetime').sort_index()
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep='first')]
    df.index = df.index.tz_convert('America/Los_Angeles')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            return None
    return df


def load_from_gcs(ticker):
    file_ticker = TICKER_FILE_MAP.get(ticker, ticker)
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        for folder in ['Data/Inference_v2', 'Data/Inference']:
            uri = f"gs://{BUCKET_NAME}/{folder}/{file_ticker}{suffix}.parquet"
            try:
                df = _load_gcs_parquet(uri)
                if df is not None and len(df) > 100:
                    start = pd.Timestamp(DATA_START).tz_localize('America/Los_Angeles')
                    df = df[df.index >= start]
                    return df
            except Exception:
                continue
    return None


def fetch_1min_bars(ticker, start, end, chunk_months=1, sleep_secs=2):
    start_dt = pd.Timestamp(start, tz='UTC')
    end_dt   = pd.Timestamp(end,   tz='UTC')

    all_aggs    = []
    chunk_start = start_dt

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + relativedelta(months=chunk_months), end_dt)
        try:
            for a in poly_client.list_aggs(ticker, 1, 'minute',
                                            chunk_start, chunk_end, limit=50000):
                all_aggs.append({
                    'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                    'Open': a.open, 'High': a.high, 'Low': a.low,
                    'Close': a.close, 'Volume': a.volume,
                })
        except Exception as e:
            print(f"      chunk {chunk_start.date()}→{chunk_end.date()} failed: {e}, retrying...")
            time.sleep(10)
            try:
                for a in poly_client.list_aggs(ticker, 1, 'minute',
                                                chunk_start, chunk_end, limit=50000):
                    all_aggs.append({
                        'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                        'Open': a.open, 'High': a.high, 'Low': a.low,
                        'Close': a.close, 'Volume': a.volume,
                    })
            except Exception as e2:
                print(f"      chunk SKIPPED: {e2}")

        chunk_start = chunk_end
        if chunk_start < end_dt:
            time.sleep(sleep_secs)

    if not all_aggs:
        return None

    df = pd.DataFrame(all_aggs).set_index('timestamp')
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df.index = df.index.tz_convert('America/Los_Angeles')
    return df


def load_ticker_1m(ticker):
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_1m.parquet")

    # 1. Local cache (fastest)
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('America/Los_Angeles')
        return df, 'cache'

    # 2. GCS bucket (already have OHLCV for many tickers)
    df = load_from_gcs(ticker)
    if df is not None and len(df) > 100:
        return df, 'gcs'

    # 3. Polygon API (download for new tickers)
    print(f"Polygon download...", end=" ", flush=True)
    df = fetch_1min_bars(ticker, DATA_START, DATA_END)
    if df is not None and len(df) > 0:
        df.to_parquet(cache_path)
        print(f"{len(df):,} bars cached", end=" ")
        return df, 'polygon'
    else:
        print("FAILED", end=" ")
        return None, 'failed'


def load_all_data():
    data = {}
    sources = {'gcs': 0, 'cache': 0, 'polygon': 0, 'failed': 0}
    n = len(ALL_STOCKS)
    for i, ticker in enumerate(ALL_STOCKS):
        print(f"  [{i+1}/{n}] {ticker}...", end=" ")
        df, source = load_ticker_1m(ticker)
        sources[source] = sources.get(source, 0) + 1
        if df is not None and len(df) > 100:
            data[ticker] = df
            print(f"{len(df):,} bars ({source})")
        else:
            print(f"SKIP ({source})")

    print(f"\n  Sources: GCS={sources['gcs']}, cache={sources['cache']}, "
          f"Polygon={sources['polygon']}, failed={sources['failed']}")
    return data


# ==============================================================================
# INTRADAY SPIKE COMPUTATION
# ==============================================================================
def compute_daily_spikes(df):
    """For each trading day, compute spike from entry_minute to EOD."""
    times = (df.index.hour * 60 + df.index.minute).values
    dates = df.index.date

    unique_dates = sorted(set(dates))
    daily_spikes = []
    prev_close = np.nan

    for d in unique_dates:
        day_mask = dates == d
        day_times  = times[day_mask]
        day_highs  = df['High'].values[day_mask]
        day_closes = df['Close'].values[day_mask]
        day_vols   = df['Volume'].values[day_mask]

        entry_mask = day_times == ENTRY_MINUTE
        if entry_mask.sum() == 0:
            entry_mask = day_times == MARKET_OPEN
            if entry_mask.sum() == 0:
                eod_bars = day_times <= EOD_MINUTE
                if eod_bars.sum() > 0:
                    prev_close = day_closes[eod_bars][-1]
                continue

        entry_idx = np.where(entry_mask)[0][0]
        entry_price = day_closes[entry_idx]
        if entry_price <= 0:
            continue

        post_entry = (day_times > ENTRY_MINUTE) & (day_times <= EOD_MINUTE)
        if post_entry.sum() == 0:
            continue

        day_lows = df['Low'].values[day_mask]
        max_high = day_highs[post_entry].max()
        min_low  = day_lows[post_entry].min()

        spike_up   = (max_high - entry_price) / entry_price
        spike_down = (entry_price - min_low) / entry_price

        day_vol = day_vols.sum()

        eod_bars = day_times <= EOD_MINUTE
        close_at_eod = day_closes[eod_bars][-1] if eod_bars.sum() > 0 else entry_price

        daily_spikes.append({
            'date': d,
            'entry_price': entry_price,
            'max_high': max_high,
            'min_low': min_low,
            'spike_up': spike_up,
            'spike_down': spike_down,
            'volume': day_vol,
            'close_at_eod': close_at_eod,
            'prev_day_close': prev_close,
        })

        prev_close = close_at_eod

    return pd.DataFrame(daily_spikes)


def apply_spike_filters(df_spikes):
    """Add quality filter columns: fade_ratio, gap_pct, vol_ratio, qualification flags."""
    if df_spikes is None or len(df_spikes) == 0:
        return df_spikes

    df_spikes['vol_sma20'] = df_spikes['volume'].rolling(20, min_periods=5).mean()
    df_spikes['vol_ratio'] = df_spikes['volume'] / df_spikes['vol_sma20']

    denom = df_spikes['max_high'] - df_spikes['entry_price']
    numer = df_spikes['max_high'] - df_spikes['close_at_eod']
    df_spikes['fade_ratio'] = np.where(denom > 0, numer / denom, np.nan)
    df_spikes['fade_ratio'] = df_spikes['fade_ratio'].clip(0, 2)

    df_spikes['gap_pct'] = np.where(
        df_spikes['prev_day_close'] > 0,
        np.abs(df_spikes['entry_price'] - df_spikes['prev_day_close']) / df_spikes['prev_day_close'],
        np.nan
    )

    df_spikes['gap_ok'] = df_spikes['gap_pct'].fillna(0) <= 0.02
    df_spikes['vol_ok'] = df_spikes['vol_ratio'].fillna(0) > 1.5
    df_spikes['qualified'] = df_spikes['gap_ok'] & df_spikes['vol_ok']

    return df_spikes


def compute_spike_metrics(df_spikes):
    """Compute spike frequency metrics from daily spike dataframe."""
    if df_spikes is None or len(df_spikes) < 10:
        return None

    n_days = len(df_spikes)
    n_months = max(1.0, n_days / 21.0)

    metrics = {
        'n_days': n_days,
        'n_months': round(n_months, 1),
        'avg_spike_up': df_spikes['spike_up'].mean(),
        'p75_spike_up': df_spikes['spike_up'].quantile(0.75),
        'p90_spike_up': df_spikes['spike_up'].quantile(0.90),
        'max_spike_up': df_spikes['spike_up'].max(),
        'avg_spike_down': df_spikes['spike_down'].mean(),
        'avg_daily_vol': df_spikes['volume'].mean(),
    }

    has_filters = 'qualified' in df_spikes.columns

    for thresh in SPIKE_THRESHOLDS:
        pct = int(thresh * 100)
        spike_mask = df_spikes['spike_up'] >= thresh
        spike_days = spike_mask.sum()
        metrics[f'spike_{pct}_count'] = spike_days
        metrics[f'spike_{pct}_per_mo'] = spike_days / n_months
        if spike_days > 0:
            metrics[f'spike_{pct}_avg_mag'] = df_spikes.loc[spike_mask, 'spike_up'].mean()
            metrics[f'spike_{pct}_avg_dd']  = df_spikes.loc[spike_mask, 'spike_down'].mean()
        else:
            metrics[f'spike_{pct}_avg_mag'] = 0.0
            metrics[f'spike_{pct}_avg_dd']  = 0.0

        if has_filters:
            q_mask = spike_mask & df_spikes['qualified']
            q_days = q_mask.sum()
            metrics[f'q_spike_{pct}_count'] = q_days
            metrics[f'q_spike_{pct}_per_mo'] = q_days / n_months
            metrics[f'q_spike_{pct}_pass_rate'] = q_days / spike_days if spike_days > 0 else 0.0

    if has_filters:
        spike5_mask = df_spikes['spike_up'] >= 0.05
        if spike5_mask.sum() > 0:
            metrics['avg_fade_ratio'] = df_spikes.loc[spike5_mask, 'fade_ratio'].mean()
            metrics['avg_gap_pct'] = df_spikes.loc[spike5_mask, 'gap_pct'].mean()
            metrics['avg_vol_ratio'] = df_spikes.loc[spike5_mask, 'vol_ratio'].mean()
            metrics['pct_gap_ok'] = df_spikes.loc[spike5_mask, 'gap_ok'].mean()
            metrics['pct_vol_ok'] = df_spikes.loc[spike5_mask, 'vol_ok'].mean()
        else:
            for k in ('avg_fade_ratio', 'avg_gap_pct', 'avg_vol_ratio'):
                metrics[k] = np.nan
            metrics['pct_gap_ok'] = 0.0
            metrics['pct_vol_ok'] = 0.0

    return metrics


# ==============================================================================
# MAIN
# ==============================================================================
print("=" * W)
print(f"  NEW ASSET SPIKE SCREENER — 1-min Intraday Analysis")
print(f"  Spike = (max High after entry) / (Close at entry) − 1")
print(f"  Entry: minute {ENTRY_MINUTE} (6:15 PT)  |  EOD: minute {EOD_MINUTE} (13:00 PT)")
print(f"  Full window: {DATA_START} → {DATA_END}  |  Focus: {RECENT_START} → {DATA_END}")
print("=" * W)

data = load_all_data()
_release_memory()

print(f"\n  Computing intraday spike metrics for {len(data)} tickers...")

results = []
for ticker in ALL_STOCKS:
    if ticker not in data:
        continue
    df = data[ticker]
    sector = STOCK_SECTOR[ticker]

    df_spikes = compute_daily_spikes(df)
    if len(df_spikes) < 10:
        print(f"    {ticker}: only {len(df_spikes)} trading days, skipping")
        continue

    df_spikes = apply_spike_filters(df_spikes)
    full_metrics = compute_spike_metrics(df_spikes)
    if full_metrics is None:
        continue

    recent_spikes = df_spikes[df_spikes['date'] >= pd.Timestamp(RECENT_START).date()]
    recent_metrics = compute_spike_metrics(recent_spikes)

    avg_price = df_spikes['entry_price'].mean()
    avg_dollar_vol = full_metrics['avg_daily_vol'] * avg_price

    row = {'Ticker': ticker, 'Sector': sector, 'avg_dollar_vol': avg_dollar_vol}
    for k, v in full_metrics.items():
        row[f'f_{k}'] = v
    if recent_metrics:
        for k, v in recent_metrics.items():
            row[f'r_{k}'] = v
    else:
        for k in full_metrics:
            row[f'r_{k}'] = 0

    results.append(row)

df_results = pd.DataFrame(results)

del data
_release_memory()
print(f"  {len(df_results)} tickers with valid spike data")


# ==============================================================================
# TABLE A: PER-TICKER SPIKE PROFILE (sorted by recent 8%+ spike freq)
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE A: PER-TICKER INTRADAY SPIKE PROFILE — 12mo Full + 3mo Recent")
print(f"  Spike measured: Close@{ENTRY_MINUTE}min → max High through {EOD_MINUTE}min")
print("=" * W)

df_sorted = df_results.sort_values('r_spike_8_per_mo', ascending=False)

print(f"\n  {'Ticker':<7} {'Sector':<35} "
      f"{'│ 12mo':>5} {'5%/mo':>6} {'8%/mo':>6} {'10%/mo':>7} {'AvgSp':>6} "
      f"{'│ 3mo':>4} {'5%/mo':>6} {'8%/mo':>6} {'10%/mo':>7} {'AvgSp':>6} "
      f"{'│ $Vol':>7}")
print(f"  {'─'*7} {'─'*35} {'─'*5} {'─'*6} {'─'*6} {'─'*7} {'─'*6} "
      f"{'─'*4} {'─'*6} {'─'*6} {'─'*7} {'─'*6} {'─'*7}")

for _, row in df_sorted.iterrows():
    is_baseline = CANDIDATE_SECTORS.get(row['Sector'], {}).get('is_baseline', False)
    marker = '▶' if is_baseline else ' '
    dollar_vol_m = row.get('avg_dollar_vol', 0) / 1e6

    print(f" {marker}{row['Ticker']:<6} {row['Sector'][:35]:<35} "
          f"│{row.get('f_n_days', 0):>4}d "
          f"{row.get('f_spike_5_per_mo', 0):>5.1f} "
          f"{row.get('f_spike_8_per_mo', 0):>5.1f} "
          f"{row.get('f_spike_10_per_mo', 0):>6.2f} "
          f"{row.get('f_avg_spike_up', 0)*100:>5.1f}% "
          f"│{row.get('r_n_days', 0):>3}d "
          f"{row.get('r_spike_5_per_mo', 0):>5.1f} "
          f"{row.get('r_spike_8_per_mo', 0):>5.1f} "
          f"{row.get('r_spike_10_per_mo', 0):>6.2f} "
          f"{row.get('r_avg_spike_up', 0)*100:>5.1f}% "
          f"│{dollar_vol_m:>6.1f}M")


# ==============================================================================
# TABLE B: SECTOR SUMMARY vs CRYPTO BASELINE
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE B: SECTOR SUMMARY — Average Spike Rates vs Crypto Mining Baseline")
print("=" * W)

sector_summary = []
for sector, cfg in CANDIDATE_SECTORS.items():
    sector_rows = df_results[df_results['Sector'] == sector]
    if len(sector_rows) == 0:
        continue
    summary = {
        'Sector': sector,
        'N': len(sector_rows),
        'f_5_avg': sector_rows['f_spike_5_per_mo'].mean(),
        'f_8_avg': sector_rows['f_spike_8_per_mo'].mean(),
        'f_10_avg': sector_rows['f_spike_10_per_mo'].mean(),
        'r_5_avg': sector_rows['r_spike_5_per_mo'].mean(),
        'r_8_avg': sector_rows['r_spike_8_per_mo'].mean(),
        'r_10_avg': sector_rows['r_spike_10_per_mo'].mean(),
        'f_avg_up': sector_rows['f_avg_spike_up'].mean(),
        'r_avg_up': sector_rows['r_avg_spike_up'].mean(),
        'is_baseline': cfg.get('is_baseline', False),
    }
    sector_summary.append(summary)

df_sector = pd.DataFrame(sector_summary).sort_values('r_10_avg', ascending=False)

crypto_baseline = df_sector[df_sector['is_baseline']].iloc[0] if any(df_sector['is_baseline']) else None

print(f"\n  {'Sector':<38} {'N':>2} "
      f"{'│ 12mo 5%':>8} {'8%':>5} {'10%':>6} {'AvgSp':>6} "
      f"{'│ 3mo 5%':>7} {'8%':>5} {'10%':>6} {'AvgSp':>6} "
      f"{'│ vs BL':>6}")
print(f"  {'─'*38} {'─'*2} {'─'*8} {'─'*5} {'─'*6} {'─'*6} "
      f"{'─'*7} {'─'*5} {'─'*6} {'─'*6} {'─'*6}")

for _, row in df_sector.iterrows():
    marker = '★' if row['is_baseline'] else ' '
    vs_bl = ''
    if crypto_baseline is not None and not row['is_baseline'] and crypto_baseline['r_10_avg'] > 0:
        ratio = row['r_10_avg'] / crypto_baseline['r_10_avg'] * 100
        vs_bl = f"{ratio:>5.0f}%"

    print(f" {marker}{row['Sector'][:37]:<37} {row['N']:>2} "
          f"│ {row['f_5_avg']:>5.1f} {row['f_8_avg']:>5.1f} {row['f_10_avg']:>5.2f} "
          f"{row['f_avg_up']*100:>5.1f}% "
          f"│ {row['r_5_avg']:>4.1f} {row['r_8_avg']:>5.1f} {row['r_10_avg']:>5.2f} "
          f"{row['r_avg_up']*100:>5.1f}% "
          f"│{vs_bl:>6}")


# ==============================================================================
# TABLE C: TOP CANDIDATES — TRAINING FEASIBILITY SCORE
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE C: TOP CANDIDATES — Ranked by Training Feasibility Score")
print(f"  Score = 0.4 × recent_10%/mo + 0.3 × recent_8%/mo + 0.2 × full_10%/mo + 0.1 × upside_rank")
print("=" * W)

df_non_baseline = df_results[
    ~df_results['Sector'].apply(lambda s: CANDIDATE_SECTORS.get(s, {}).get('is_baseline', False))
].copy()

if len(df_non_baseline) > 0:
    up_rank = df_non_baseline['r_avg_spike_up'].rank(pct=True)
    df_non_baseline['score'] = (
        0.4 * df_non_baseline['r_spike_10_per_mo'] +
        0.3 * df_non_baseline['r_spike_8_per_mo'] +
        0.2 * df_non_baseline['f_spike_10_per_mo'] +
        0.1 * up_rank * df_non_baseline['r_spike_10_per_mo'].max()
    )
    df_ranked = df_non_baseline.sort_values('score', ascending=False)

    print(f"\n  {'Rk':>3} {'Ticker':<7} {'Sector':<35} "
          f"{'Score':>6} {'│ R.10%/mo':>9} {'R.8%/mo':>8} {'F.10%/mo':>9} "
          f"{'R.AvgSp':>7} {'R.MaxSp':>8} {'$Vol(M)':>8}")
    print(f"  {'─'*3} {'─'*7} {'─'*35} {'─'*6} {'─'*9} {'─'*8} {'─'*9} {'─'*7} {'─'*8} {'─'*8}")

    for rank, (_, row) in enumerate(df_ranked.head(40).iterrows(), 1):
        meets_10 = '✓' if row.get('r_spike_10_per_mo', 0) >= MIN_SPIKE_10PCT_PER_MONTH else ' '
        dollar_vol_m = row.get('avg_dollar_vol', 0) / 1e6
        print(f"  {rank:>3} {row['Ticker']:<7} {row['Sector'][:35]:<35} "
              f"{row['score']:>5.2f}{meets_10} "
              f"│ {row.get('r_spike_10_per_mo', 0):>7.2f} "
              f"{row.get('r_spike_8_per_mo', 0):>7.1f} "
              f"{row.get('f_spike_10_per_mo', 0):>8.2f} "
              f"{row.get('r_avg_spike_up', 0)*100:>6.1f}% "
              f"{row.get('r_max_spike_up', 0)*100:>7.1f}% "
              f"{dollar_vol_m:>7.1f}M")


# ==============================================================================
# TABLE D: TREND — 3mo vs 12mo spike rate change
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE D: SPIKE TREND — Recent 3mo vs Full 12mo (acceleration / deceleration)")
print("=" * W)

df_trend = df_results.copy()
df_trend['trend_10'] = df_trend['r_spike_10_per_mo'] - df_trend['f_spike_10_per_mo']
df_trend['trend_8']  = df_trend['r_spike_8_per_mo']  - df_trend['f_spike_8_per_mo']
df_trend = df_trend.sort_values('trend_10', ascending=False)

print(f"\n  {'Ticker':<7} {'Sector':<35} "
      f"{'│ 10%/mo Full':>13} {'Recent':>7} {'Δ':>6} "
      f"{'│ 8%/mo Full':>12} {'Recent':>7} {'Δ':>6} {'Signal':>8}")
print(f"  {'─'*7} {'─'*35} {'─'*13} {'─'*7} {'─'*6} {'─'*12} {'─'*7} {'─'*6} {'─'*8}")

for _, row in df_trend.iterrows():
    trend_10 = row['trend_10']
    trend_8  = row['trend_8']
    if trend_10 > 0.5:
        signal = 'ACCEL'
    elif trend_10 < -0.5:
        signal = 'DECEL'
    else:
        signal = 'FLAT'

    print(f"  {row['Ticker']:<7} {row['Sector'][:35]:<35} "
          f"│ {row.get('f_spike_10_per_mo', 0):>10.2f} "
          f"{row.get('r_spike_10_per_mo', 0):>7.2f} "
          f"{trend_10:>+5.1f} "
          f"│ {row.get('f_spike_8_per_mo', 0):>9.1f} "
          f"{row.get('r_spike_8_per_mo', 0):>7.1f} "
          f"{trend_8:>+5.1f} "
          f"{signal:>8}")


# ==============================================================================
# TABLE E: FINAL SHORTLIST — meets 10% spike threshold for training
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE E: FINAL SHORTLIST — Tickers with ≥{MIN_SPIKE_10PCT_PER_MONTH:.0f} spikes at 10%/month (recent 3mo)")
print(f"  These are viable training candidates for next experiment teacher sets")
print("=" * W)

shortlist = df_results[
    (df_results['r_spike_10_per_mo'] >= MIN_SPIKE_10PCT_PER_MONTH)
].sort_values('r_spike_10_per_mo', ascending=False)

already_teachers = {
    'CIFR', 'IREN', 'KEEL', 'HUT', 'APLD', 'WULF', 'RIOT', 'CLSK', 'BTBT',
    'ALAB', 'MARA', 'RCAT', 'ASTS', 'LUNR', 'BBAI', 'RKLB', 'SPIR', 'BKSY',
    'MNTS', 'PLUG', 'BE', 'FCEL', 'FLNC', 'OKLO', 'SMR', 'MVIS', 'BGDE',
    'RDW', 'PL', 'VOYG', 'SMLR', 'AMD', 'NVDA', 'TSLA', 'PLTR', 'SOUN',
    'HOOD', 'SMCI', 'VRT', 'NBIS', 'KTOS', 'AVAV', 'ACHR', 'JOBY', 'NNDM',
    'CCJ', 'CEG',
}

print(f"\n  {'Ticker':<7} {'Sector':<35} "
      f"{'10%/mo':>7} {'8%/mo':>7} {'5%/mo':>7} "
      f"{'AvgSp':>6} {'MaxSp':>6} {'AvgDD':>6} "
      f"{'$Vol(M)':>8} {'Status':>14}")
print(f"  {'─'*7} {'─'*35} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*14}")

new_candidates = []
for _, row in shortlist.iterrows():
    ticker = row['Ticker']
    in_teachers = ticker in already_teachers
    status = 'IN TEACHERS' if in_teachers else 'NEW'
    dollar_vol_m = row.get('avg_dollar_vol', 0) / 1e6

    print(f"  {ticker:<7} {row['Sector'][:35]:<35} "
          f"{row.get('r_spike_10_per_mo', 0):>6.2f} "
          f"{row.get('r_spike_8_per_mo', 0):>6.1f} "
          f"{row.get('r_spike_5_per_mo', 0):>6.1f} "
          f"{row.get('r_avg_spike_up', 0)*100:>5.1f}% "
          f"{row.get('r_max_spike_up', 0)*100:>5.1f}% "
          f"{row.get('r_avg_spike_down', 0)*100:>5.1f}% "
          f"{dollar_vol_m:>7.1f}M "
          f"{status:>14}")

    if not in_teachers:
        new_candidates.append(ticker)


# ==============================================================================
# TABLE F: WIDER NET — 8%+ but below 10% threshold
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE F: WIDER NET — Tickers with ≥{MIN_SPIKE_8PCT_PER_MONTH:.0f} spikes at 8%/month (recent 3mo)")
print(f"  Not yet meeting 10% threshold but worth monitoring")
print("=" * W)

wider = df_results[
    (df_results['r_spike_8_per_mo'] >= MIN_SPIKE_8PCT_PER_MONTH) &
    (df_results['r_spike_10_per_mo'] < MIN_SPIKE_10PCT_PER_MONTH)
].sort_values('r_spike_8_per_mo', ascending=False)

print(f"\n  {'Ticker':<7} {'Sector':<35} "
      f"{'10%/mo':>7} {'8%/mo':>7} {'5%/mo':>7} "
      f"{'AvgSp':>6} {'$Vol(M)':>8} {'Status':>14}")
print(f"  {'─'*7} {'─'*35} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*8} {'─'*14}")

for _, row in wider.iterrows():
    ticker = row['Ticker']
    in_teachers = ticker in already_teachers
    status = 'IN TEACHERS' if in_teachers else 'NEW'
    dollar_vol_m = row.get('avg_dollar_vol', 0) / 1e6

    print(f"  {ticker:<7} {row['Sector'][:35]:<35} "
          f"{row.get('r_spike_10_per_mo', 0):>6.2f} "
          f"{row.get('r_spike_8_per_mo', 0):>6.1f} "
          f"{row.get('r_spike_5_per_mo', 0):>6.1f} "
          f"{row.get('r_avg_spike_up', 0)*100:>5.1f}% "
          f"{dollar_vol_m:>7.1f}M "
          f"{status:>14}")


# ==============================================================================
# TABLE G: SPIKE QUALITY — Qualified (gap≤2% + vol>1.5×SMA20) vs Raw
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE G: SPIKE QUALITY — Qualified spikes vs Raw (recent 3mo)")
print(f"  Qualified = Gap Insulator (overnight gap ≤ 2%) + Volume Filter (vol > 1.5×SMA20)")
print(f"  Fade Ratio: 0 = held at high, 1 = faded to entry price by EOD")
print("=" * W)

df_quality = df_results[df_results['r_spike_5_per_mo'] >= 1.0].copy()
df_quality = df_quality.sort_values('r_spike_10_per_mo', ascending=False)

has_q_cols = 'r_q_spike_10_per_mo' in df_quality.columns

if has_q_cols and len(df_quality) > 0:
    print(f"\n  {'Ticker':<7} {'Sector':<30} "
          f"{'│ Raw':>4} {'10%':>5} {'8%':>5} "
          f"{'│ Qual':>5} {'10%':>5} {'8%':>5} "
          f"{'│Pass%':>6} "
          f"{'│Fade':>5} {'Gap%':>5} {'VolR':>5} "
          f"{'│GapOK':>6} {'VolOK':>6}")
    print(f"  {'─'*7} {'─'*30} "
          f"{'─'*4} {'─'*5} {'─'*5} "
          f"{'─'*5} {'─'*5} {'─'*5} "
          f"{'─'*6} "
          f"{'─'*5} {'─'*5} {'─'*5} "
          f"{'─'*6} {'─'*6}")

    for _, row in df_quality.iterrows():
        raw_10 = row.get('r_spike_10_per_mo', 0)
        raw_8  = row.get('r_spike_8_per_mo', 0)
        q_10   = row.get('r_q_spike_10_per_mo', 0)
        q_8    = row.get('r_q_spike_8_per_mo', 0)
        pass_r = row.get('r_q_spike_10_pass_rate', 0) * 100 if raw_10 > 0 else 0
        fade   = row.get('r_avg_fade_ratio', float('nan'))
        gap    = row.get('r_avg_gap_pct', float('nan'))
        volr   = row.get('r_avg_vol_ratio', float('nan'))
        g_ok   = row.get('r_pct_gap_ok', 0) * 100
        v_ok   = row.get('r_pct_vol_ok', 0) * 100

        fade_s = f"{fade:.2f}" if not np.isnan(fade) else "  - "
        gap_s  = f"{gap*100:.1f}" if not np.isnan(gap) else " - "
        volr_s = f"{volr:.1f}" if not np.isnan(volr) else " - "

        print(f"  {row['Ticker']:<7} {row['Sector'][:30]:<30} "
              f"│   {raw_10:>4.1f} {raw_8:>5.1f} "
              f"│    {q_10:>4.1f} {q_8:>5.1f} "
              f"│{pass_r:>5.0f}% "
              f"│{fade_s:>4} {gap_s:>4}% {volr_s:>4}x "
              f"│{g_ok:>5.0f}% {v_ok:>5.0f}%")
else:
    print("\n  (no quality filter data available — need close_at_eod/prev_day_close in spike data)")


# ==============================================================================
# SUMMARY
# ==============================================================================
print()
print("=" * W)
print(f"  SUMMARY")
print("=" * W)

n_total = len(df_results)
n_shortlist = len(shortlist)
n_new = len(new_candidates)
n_wider = len(wider)

print(f"\n  Tickers analyzed:        {n_total}")
print(f"  Meet 10%/mo threshold:   {n_shortlist} ({n_new} NEW, not in current teachers)")
print(f"  Meet 8%/mo wider net:    {n_wider} (below 10% threshold)")
print(f"\n  NEW candidates for next experiment teacher set:")
if new_candidates:
    for c in new_candidates:
        r = df_results[df_results['Ticker'] == c].iloc[0]
        print(f"    {c:<7} {STOCK_SECTOR[c][:35]:<35} "
              f"10%/mo={r.get('r_spike_10_per_mo',0):.1f}  "
              f"8%/mo={r.get('r_spike_8_per_mo',0):.1f}  "
              f"avg_spike={r.get('r_avg_spike_up',0)*100:.1f}%")
else:
    print(f"    (none found meeting threshold)")

print(f"\n  Data cached at: {CACHE_DIR}/")
_release_memory()
print(f"\n  Done.")
