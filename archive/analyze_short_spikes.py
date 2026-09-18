"""
analyze_short_spikes.py — Short-Side Spike / Liquidation Dump Screener
=======================================================================
Measures downside spike: Close at entry bar → min Low through close.
  spike_down = (entry_price − min_low) / entry_price

Goal: find the right target threshold for Exp 400 (short-side model).
Compares -6%, -8%, -10%, -12% thresholds across all tickers.

Windows:
  - Full 12 months (Jul 2025 → Jul 2026)
  - Recent 3 months (Apr 2026 → Jul 2026) — focus window

Also reports:
  - Recovery ratio: how much of the dump reverses by EOD
  - Volume surge during dumps (liquidity proxy)
  - Intraday PnL if you had shorted at entry and covered at dump low

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
DATA_END   = '2026-07-19'

RECENT_START = '2026-04-01'

ENTRY_MINUTE  = 375    # 6:15 PT = 6*60+15
EOD_MINUTE    = 780    # 13:00 PT
MARKET_OPEN   = 390    # 6:30 PT (regular open)

# Short-side thresholds to evaluate
DROP_THRESHOLDS = [0.04, 0.06, 0.08, 0.10, 0.12]

W = 120

# ==============================================================================
# TICKER LISTS — all assets we have data for
# ==============================================================================
CANDIDATE_SECTORS = {
    'Crypto Mining': {
        'stocks': ['HUT', 'CIFR', 'CLSK', 'RIOT', 'WULF', 'MARA', 'KEEL',
                   'APLD', 'BTBT', 'CORZ', 'IREN', 'MSTR', 'COIN', 'HOOD'],
    },
    'Semiconductor & AI': {
        'stocks': ['AEHR', 'CRDO', 'MXL', 'UCTT', 'ALAB', 'AMD', 'NVDA',
                   'ARM', 'MU', 'SMCI', 'VRT'],
    },
    'Biotech & Genomics': {
        'stocks': ['MRNA', 'TWST', 'SLS', 'BEAM', 'NTLA'],
    },
    'Energy / Nuclear': {
        'stocks': ['BE', 'WOLF', 'PLUG', 'FCEL', 'OKLO', 'NNE', 'NEOV',
                   'SMR', 'VST', 'CEG', 'GEV', 'POWL', 'FLNC', 'CCJ'],
    },
    'Space / Defense': {
        'stocks': ['AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS', 'LUNR',
                   'SPIR', 'JOBY', 'NNDM', 'MVIS', 'RCAT', 'ACHR',
                   'PL', 'RDW', 'BKSY', 'MNTS', 'VOYG', 'LMT', 'NOC', 'RTX'],
    },
    'Industrials / Infra': {
        'stocks': ['CAT', 'FCX', 'BHP', 'ETN', 'LIN', 'GD', 'LHX', 'GE'],
    },
    'Other High-Beta': {
        'stocks': ['NBIS', 'SOUN', 'ONDS', 'EQIX', 'TSLA', 'MP', 'MOD',
                   'BGDE', 'XOM', 'CVX', 'COP', 'DUK', 'NEE'],
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
# DATA LOADING — GCS first, Polygon fallback, local cache
# ==============================================================================
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

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
    all_aggs = []
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
            print(f"      chunk {chunk_start.date()}→{chunk_end.date()} failed: {e}")
            time.sleep(10)
            try:
                for a in poly_client.list_aggs(ticker, 1, 'minute',
                                                chunk_start, chunk_end, limit=50000):
                    all_aggs.append({
                        'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                        'Open': a.open, 'High': a.high, 'Low': a.low,
                        'Close': a.close, 'Volume': a.volume,
                    })
            except Exception:
                pass
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
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('America/Los_Angeles')
        return df, 'cache'
    df = load_from_gcs(ticker)
    if df is not None and len(df) > 100:
        return df, 'gcs'
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
# INTRADAY SPIKE COMPUTATION — BOTH SIDES
# ==============================================================================
def compute_daily_spikes(df):
    times = (df.index.hour * 60 + df.index.minute).values
    dates = df.index.date

    unique_dates = sorted(set(dates))
    daily_spikes = []
    prev_close = np.nan

    for d in unique_dates:
        day_mask = dates == d
        day_times  = times[day_mask]
        day_highs  = df['High'].values[day_mask]
        day_lows   = df['Low'].values[day_mask]
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

        max_high = day_highs[post_entry].max()
        min_low  = day_lows[post_entry].min()

        spike_up   = (max_high - entry_price) / entry_price
        spike_down = (entry_price - min_low) / entry_price  # positive = drop

        day_vol = day_vols.sum()
        eod_bars = day_times <= EOD_MINUTE
        close_at_eod = day_closes[eod_bars][-1] if eod_bars.sum() > 0 else entry_price

        # Recovery ratio for dumps: how much of the drop was recovered by EOD
        # 0 = no recovery (closed at low), 1 = full recovery (closed at entry)
        if spike_down > 0.02:
            drop_amount = entry_price - min_low
            recovery = (close_at_eod - min_low) / drop_amount if drop_amount > 0 else 0
            recovery = max(0, min(2.0, recovery))  # clip
        else:
            recovery = np.nan

        # Time to min low (minutes after entry)
        post_entry_lows = day_lows[post_entry]
        post_entry_times = day_times[post_entry]
        min_low_idx = np.argmin(post_entry_lows)
        time_to_low = post_entry_times[min_low_idx] - ENTRY_MINUTE

        # Short PnL: if shorted at entry, covered at min_low
        short_pnl = spike_down  # gross short profit = drop magnitude

        # Volume in the dump window (entry to min_low time)
        dump_window = post_entry & (day_times <= post_entry_times[min_low_idx])
        dump_vol = day_vols[day_mask][dump_window].sum() if dump_window.sum() > 0 else 0

        daily_spikes.append({
            'date': d,
            'entry_price': entry_price,
            'max_high': max_high,
            'min_low': min_low,
            'spike_up': spike_up,
            'spike_down': spike_down,
            'volume': day_vol,
            'dump_volume': dump_vol,
            'close_at_eod': close_at_eod,
            'prev_day_close': prev_close,
            'recovery_ratio': recovery,
            'time_to_low': time_to_low,
            'short_pnl': short_pnl,
        })

        prev_close = close_at_eod

    return pd.DataFrame(daily_spikes)


def compute_drop_metrics(df_spikes, label=''):
    if df_spikes is None or len(df_spikes) < 10:
        return None

    n_days = len(df_spikes)
    n_months = max(1.0, n_days / 21.0)

    metrics = {
        'n_days': n_days,
        'n_months': round(n_months, 1),
        'avg_drop': df_spikes['spike_down'].mean(),
        'p75_drop': df_spikes['spike_down'].quantile(0.75),
        'p90_drop': df_spikes['spike_down'].quantile(0.90),
        'max_drop': df_spikes['spike_down'].max(),
        'avg_spike_up': df_spikes['spike_up'].mean(),
        'avg_daily_vol': df_spikes['volume'].mean(),
    }

    for thresh in DROP_THRESHOLDS:
        pct = int(thresh * 100)
        drop_mask = df_spikes['spike_down'] >= thresh
        drop_days = drop_mask.sum()
        metrics[f'drop_{pct}_count'] = drop_days
        metrics[f'drop_{pct}_per_mo'] = drop_days / n_months

        if drop_days > 0:
            drops = df_spikes[drop_mask]
            metrics[f'drop_{pct}_avg_mag'] = drops['spike_down'].mean()
            metrics[f'drop_{pct}_avg_recovery'] = drops['recovery_ratio'].mean()
            metrics[f'drop_{pct}_avg_time_to_low'] = drops['time_to_low'].mean()
            metrics[f'drop_{pct}_median_time_to_low'] = drops['time_to_low'].median()
            # Volume surge during dumps vs average day
            avg_vol = df_spikes['volume'].mean()
            if avg_vol > 0:
                metrics[f'drop_{pct}_vol_surge'] = drops['volume'].mean() / avg_vol
            else:
                metrics[f'drop_{pct}_vol_surge'] = 0
            # Simulated short PnL (gross, before slippage)
            metrics[f'drop_{pct}_avg_short_pnl'] = drops['spike_down'].mean()
        else:
            metrics[f'drop_{pct}_avg_mag'] = 0.0
            metrics[f'drop_{pct}_avg_recovery'] = 0.0
            metrics[f'drop_{pct}_avg_time_to_low'] = 0.0
            metrics[f'drop_{pct}_median_time_to_low'] = 0.0
            metrics[f'drop_{pct}_vol_surge'] = 0.0
            metrics[f'drop_{pct}_avg_short_pnl'] = 0.0

    return metrics


# ==============================================================================
# MAIN
# ==============================================================================
print("=" * W)
print(f"  SHORT-SIDE SPIKE / LIQUIDATION DUMP SCREENER — Exp 400 Target Analysis")
print(f"  Drop = (entry_price − min_Low) / entry_price")
print(f"  Entry: minute {ENTRY_MINUTE} (6:15 PT)  |  EOD: minute {EOD_MINUTE} (13:00 PT)")
print(f"  Full window: {DATA_START} → {DATA_END}  |  Focus: {RECENT_START} → {DATA_END}")
print(f"  Thresholds: {', '.join(f'{t:.0%}' for t in DROP_THRESHOLDS)}")
print("=" * W)

data = load_all_data()
_release_memory()

print(f"\n  Computing short-side spike metrics for {len(data)} tickers...")

results = []
all_spikes = {}  # keep for distribution analysis

for ticker in ALL_STOCKS:
    if ticker not in data:
        continue
    df = data[ticker]
    sector = STOCK_SECTOR[ticker]

    df_spikes = compute_daily_spikes(df)
    if len(df_spikes) < 10:
        print(f"    {ticker}: only {len(df_spikes)} trading days, skipping")
        continue

    all_spikes[ticker] = df_spikes

    full_metrics = compute_drop_metrics(df_spikes, 'full')
    if full_metrics is None:
        continue

    recent_spikes = df_spikes[df_spikes['date'] >= pd.Timestamp(RECENT_START).date()]
    recent_metrics = compute_drop_metrics(recent_spikes, 'recent')

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
print(f"  {len(df_results)} tickers with valid data")


# ==============================================================================
# TABLE A: THRESHOLD COMPARISON — which target is best?
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE A: THRESHOLD COMPARISON — Avg Drops/Month Across All Tickers")
print(f"  Goal: find threshold with ≥2 events/month for enough training signal")
print("=" * W)

print(f"\n  {'Threshold':<12} │ {'12mo Avg/mo':>12} {'Med/mo':>8} {'Max/mo':>8} {'N tickers≥2':>12}"
      f" │ {'3mo Avg/mo':>11} {'Med/mo':>8} {'Max/mo':>8} {'N tickers≥2':>12}")
print(f"  {'─'*12} │ {'─'*12} {'─'*8} {'─'*8} {'─'*12}"
      f" │ {'─'*11} {'─'*8} {'─'*8} {'─'*12}")

for thresh in DROP_THRESHOLDS:
    pct = int(thresh * 100)
    f_col = f'f_drop_{pct}_per_mo'
    r_col = f'r_drop_{pct}_per_mo'

    f_vals = df_results[f_col]
    r_vals = df_results[r_col]

    f_ge2 = (f_vals >= 2.0).sum()
    r_ge2 = (r_vals >= 2.0).sum()

    print(f"  {thresh:>10.0%}   │ {f_vals.mean():>10.2f} {f_vals.median():>8.2f} "
          f"{f_vals.max():>8.2f} {f_ge2:>10}"
          f"   │ {r_vals.mean():>9.2f} {r_vals.median():>8.2f} "
          f"{r_vals.max():>8.2f} {r_ge2:>10}")


# ==============================================================================
# TABLE B: PER-TICKER DROP PROFILE — sorted by recent 8% drop frequency
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE B: PER-TICKER DROP PROFILE — 12mo Full + 3mo Recent")
print(f"  Drop = max intraday drawdown from 6:15 entry to 13:00 exit")
print("=" * W)

df_sorted = df_results.sort_values('r_drop_8_per_mo', ascending=False)

print(f"\n  {'Ticker':<7} {'Sector':<22} "
      f"│{'12m':>3} {'4%':>5} {'6%':>5} {'8%':>5} {'10%':>5} {'12%':>5} {'MaxDr':>6} "
      f"│{'3m':>3} {'4%':>5} {'6%':>5} {'8%':>5} {'10%':>5} {'12%':>5} {'MaxDr':>6} "
      f"│{'$Vol':>6}")
print(f"  {'─'*7} {'─'*22} "
      f"│{'─'*3} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*6} "
      f"│{'─'*3} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*6} "
      f"│{'─'*6}")

for _, row in df_sorted.iterrows():
    dvol = row.get('avg_dollar_vol', 0) / 1e6
    print(f"  {row['Ticker']:<7} {row['Sector'][:22]:<22} "
          f"│{row.get('f_n_days',0):>3.0f} "
          f"{row.get('f_drop_4_per_mo',0):>5.1f} "
          f"{row.get('f_drop_6_per_mo',0):>5.1f} "
          f"{row.get('f_drop_8_per_mo',0):>5.1f} "
          f"{row.get('f_drop_10_per_mo',0):>5.1f} "
          f"{row.get('f_drop_12_per_mo',0):>5.1f} "
          f"{row.get('f_max_drop',0)*100:>5.1f}% "
          f"│{row.get('r_n_days',0):>3.0f} "
          f"{row.get('r_drop_4_per_mo',0):>5.1f} "
          f"{row.get('r_drop_6_per_mo',0):>5.1f} "
          f"{row.get('r_drop_8_per_mo',0):>5.1f} "
          f"{row.get('r_drop_10_per_mo',0):>5.1f} "
          f"{row.get('r_drop_12_per_mo',0):>5.1f} "
          f"{row.get('r_max_drop',0)*100:>5.1f}% "
          f"│{dvol:>5.0f}M")


# ==============================================================================
# TABLE C: SECTOR SUMMARY
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE C: SECTOR SUMMARY — Average Drops/Month")
print("=" * W)

sector_summary = []
for sector in CANDIDATE_SECTORS:
    sector_rows = df_results[df_results['Sector'] == sector]
    if len(sector_rows) == 0:
        continue
    summary = {'Sector': sector, 'N': len(sector_rows)}
    for thresh in DROP_THRESHOLDS:
        pct = int(thresh * 100)
        summary[f'f_{pct}'] = sector_rows[f'f_drop_{pct}_per_mo'].mean()
        summary[f'r_{pct}'] = sector_rows[f'r_drop_{pct}_per_mo'].mean()
    summary['f_max'] = sector_rows['f_max_drop'].mean()
    summary['r_max'] = sector_rows['r_max_drop'].mean()
    sector_summary.append(summary)

df_sector = pd.DataFrame(sector_summary).sort_values('r_8', ascending=False)

print(f"\n  {'Sector':<24} {'N':>2} "
      f"│{'12mo 6%':>7} {'8%':>5} {'10%':>5} {'12%':>5} "
      f"│{'3mo 6%':>6} {'8%':>5} {'10%':>5} {'12%':>5}")
print(f"  {'─'*24} {'─'*2} "
      f"│{'─'*7} {'─'*5} {'─'*5} {'─'*5} "
      f"│{'─'*6} {'─'*5} {'─'*5} {'─'*5}")

for _, row in df_sector.iterrows():
    print(f"  {row['Sector'][:24]:<24} {row['N']:>2} "
          f"│{row.get('f_6',0):>6.1f} {row.get('f_8',0):>5.1f} "
          f"{row.get('f_10',0):>5.1f} {row.get('f_12',0):>5.1f} "
          f"│{row.get('r_6',0):>5.1f} {row.get('r_8',0):>5.1f} "
          f"{row.get('r_10',0):>5.1f} {row.get('r_12',0):>5.1f}")


# ==============================================================================
# TABLE D: DUMP QUALITY — Recovery, Timing, Volume Surge
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE D: DUMP QUALITY AT EACH THRESHOLD (Recent 3mo)")
print(f"  Recovery: 0=stayed at low, 1=fully recovered by EOD")
print(f"  Time: minutes from entry to min_low")
print(f"  VolSurge: dump day volume / average day volume")
print("=" * W)

for thresh in [0.06, 0.08, 0.10]:
    pct = int(thresh * 100)
    print(f"\n  ── {thresh:.0%} DROPS ──")

    col_count = f'r_drop_{pct}_count'
    col_mag   = f'r_drop_{pct}_avg_mag'
    col_rec   = f'r_drop_{pct}_avg_recovery'
    col_time  = f'r_drop_{pct}_avg_time_to_low'
    col_mtime = f'r_drop_{pct}_median_time_to_low'
    col_vol   = f'r_drop_{pct}_vol_surge'

    has_drops = df_results[df_results[col_count] > 0].sort_values(col_count, ascending=False)

    if len(has_drops) == 0:
        print(f"    No tickers had {thresh:.0%}+ drops in recent 3 months")
        continue

    print(f"  {'Ticker':<7} {'Drops':>5} {'AvgMag':>7} {'Recovery':>9} "
          f"{'AvgMin':>7} {'MedMin':>7} {'VolSrg':>7}")
    print(f"  {'─'*7} {'─'*5} {'─'*7} {'─'*9} {'─'*7} {'─'*7} {'─'*7}")

    for _, row in has_drops.iterrows():
        print(f"  {row['Ticker']:<7} {row[col_count]:>5.0f} "
              f"{row[col_mag]*100:>6.1f}% "
              f"{row[col_rec]:>8.2f} "
              f"{row[col_time]:>6.0f}m "
              f"{row[col_mtime]:>6.0f}m "
              f"{row[col_vol]:>6.1f}x")


# ==============================================================================
# TABLE E: ASYMMETRY — UP vs DOWN spike comparison
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE E: ASYMMETRY — UP vs DOWN Spike Rate (Recent 3mo)")
print(f"  Ratio > 1 = more dumps than spikes (short-biased)")
print("=" * W)

df_asym = df_results.copy()
for thresh in [0.06, 0.08, 0.10]:
    pct = int(thresh * 100)
    # We need up-spike rates too — compute from all_spikes
    for idx, row in df_asym.iterrows():
        ticker = row['Ticker']
        if ticker in all_spikes:
            spikes = all_spikes[ticker]
            recent = spikes[spikes['date'] >= pd.Timestamp(RECENT_START).date()]
            n_months = max(1.0, len(recent) / 21.0)
            up_count = (recent['spike_up'] >= thresh).sum()
            df_asym.at[idx, f'r_up_{pct}_per_mo'] = up_count / n_months
            dn_count = (recent['spike_down'] >= thresh).sum()
            if up_count > 0:
                df_asym.at[idx, f'ratio_{pct}'] = dn_count / up_count
            else:
                df_asym.at[idx, f'ratio_{pct}'] = 999 if dn_count > 0 else 0

df_asym = df_asym.sort_values('r_drop_8_per_mo', ascending=False)

print(f"\n  {'Ticker':<7} {'Sector':<22} "
      f"│{'UP 8%':>5} {'DN 8%':>5} {'Ratio':>6} "
      f"│{'UP10%':>5} {'DN10%':>5} {'Ratio':>6} "
      f"│{'Bias':>8}")
print(f"  {'─'*7} {'─'*22} "
      f"│{'─'*5} {'─'*5} {'─'*6} "
      f"│{'─'*5} {'─'*5} {'─'*6} "
      f"│{'─'*8}")

for _, row in df_asym.head(40).iterrows():
    up8 = row.get('r_up_8_per_mo', 0)
    dn8 = row.get('r_drop_8_per_mo', 0)
    r8  = row.get('ratio_8', 0)
    up10 = row.get('r_up_10_per_mo', 0)
    dn10 = row.get('r_drop_10_per_mo', 0)
    r10  = row.get('ratio_10', 0)

    if r8 > 1.2:
        bias = 'SHORT▼'
    elif r8 < 0.8:
        bias = 'LONG▲'
    else:
        bias = 'NEUTRAL'

    r8_s = f"{r8:.1f}" if r8 < 100 else "∞"
    r10_s = f"{r10:.1f}" if r10 < 100 else "∞"

    print(f"  {row['Ticker']:<7} {row['Sector'][:22]:<22} "
          f"│{up8:>5.1f} {dn8:>5.1f} {r8_s:>6} "
          f"│{up10:>5.1f} {dn10:>5.1f} {r10_s:>6} "
          f"│{bias:>8}")


# ==============================================================================
# TABLE F: DISTRIBUTION — P50/P75/P90/P95/P99 of daily max drawdowns
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE F: DRAWDOWN DISTRIBUTION — Recent 3mo Percentiles")
print(f"  Helps pick the right target threshold")
print("=" * W)

print(f"\n  {'Ticker':<7} {'Sector':<22} "
      f"│{'P50':>6} {'P75':>6} {'P90':>6} {'P95':>6} {'P99':>6} {'Max':>6} {'Mean':>6}")
print(f"  {'─'*7} {'─'*22} "
      f"│{'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

percentile_data = []
for ticker in ALL_STOCKS:
    if ticker not in all_spikes:
        continue
    spikes = all_spikes[ticker]
    recent = spikes[spikes['date'] >= pd.Timestamp(RECENT_START).date()]
    if len(recent) < 10:
        continue

    drops = recent['spike_down']
    pdata = {
        'Ticker': ticker,
        'Sector': STOCK_SECTOR[ticker],
        'p50': drops.quantile(0.50),
        'p75': drops.quantile(0.75),
        'p90': drops.quantile(0.90),
        'p95': drops.quantile(0.95),
        'p99': drops.quantile(0.99),
        'max': drops.max(),
        'mean': drops.mean(),
    }
    percentile_data.append(pdata)

    print(f"  {ticker:<7} {STOCK_SECTOR[ticker][:22]:<22} "
          f"│{pdata['p50']*100:>5.1f}% "
          f"{pdata['p75']*100:>5.1f}% "
          f"{pdata['p90']*100:>5.1f}% "
          f"{pdata['p95']*100:>5.1f}% "
          f"{pdata['p99']*100:>5.1f}% "
          f"{pdata['max']*100:>5.1f}% "
          f"{pdata['mean']*100:>5.1f}%")

# Cross-ticker summary
if percentile_data:
    pdf = pd.DataFrame(percentile_data)
    print(f"\n  {'AVERAGE':<7} {'(all tickers)':<22} "
          f"│{pdf['p50'].mean()*100:>5.1f}% "
          f"{pdf['p75'].mean()*100:>5.1f}% "
          f"{pdf['p90'].mean()*100:>5.1f}% "
          f"{pdf['p95'].mean()*100:>5.1f}% "
          f"{pdf['p99'].mean()*100:>5.1f}% "
          f"{pdf['max'].mean()*100:>5.1f}% "
          f"{pdf['mean'].mean()*100:>5.1f}%")
    print(f"  {'MEDIAN':<7} {'(all tickers)':<22} "
          f"│{pdf['p50'].median()*100:>5.1f}% "
          f"{pdf['p75'].median()*100:>5.1f}% "
          f"{pdf['p90'].median()*100:>5.1f}% "
          f"{pdf['p95'].median()*100:>5.1f}% "
          f"{pdf['p99'].median()*100:>5.1f}% "
          f"{pdf['max'].median()*100:>5.1f}% "
          f"{pdf['mean'].median()*100:>5.1f}%")


# ==============================================================================
# TABLE G: SHORTLIST — Viable Exp 400 teachers at each threshold
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE G: VIABLE TEACHERS BY THRESHOLD (≥2.0 drops/month recent 3mo)")
print("=" * W)

for thresh in DROP_THRESHOLDS:
    pct = int(thresh * 100)
    col = f'r_drop_{pct}_per_mo'
    viable = df_results[df_results[col] >= 2.0].sort_values(col, ascending=False)

    print(f"\n  ── {thresh:.0%} threshold: {len(viable)} tickers with ≥2/month ──")
    if len(viable) > 0:
        for _, row in viable.iterrows():
            dvol = row.get('avg_dollar_vol', 0) / 1e6
            print(f"    {row['Ticker']:<7} {row['Sector'][:25]:<25} "
                  f"{row[col]:>5.1f}/mo  "
                  f"avgMag={row.get(f'r_drop_{pct}_avg_mag',0)*100:>5.1f}%  "
                  f"recovery={row.get(f'r_drop_{pct}_avg_recovery',0):.2f}  "
                  f"$vol={dvol:.0f}M")
    else:
        print(f"    (none)")


# ==============================================================================
# CHARTS
# ==============================================================================
if HAS_MPL:
    print(f"\n  Generating charts...")

    # Chart 1: Threshold comparison — how many tickers qualify at each level
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    thresholds_pct = [int(t * 100) for t in DROP_THRESHOLDS]
    for period, prefix, color in [('12mo Full', 'f', '#1f77b4'), ('3mo Recent', 'r', '#ff7f0e')]:
        counts = []
        for thresh in DROP_THRESHOLDS:
            pct = int(thresh * 100)
            col = f'{prefix}_drop_{pct}_per_mo'
            counts.append((df_results[col] >= 2.0).sum())
        ax1.plot(thresholds_pct, counts, 'o-', label=period, color=color, linewidth=2, markersize=8)
    ax1.set_xlabel('Drop Threshold (%)')
    ax1.set_ylabel('Number of Tickers with ≥2/month')
    ax1.set_title('Viable Teacher Count by Threshold', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Chart 2: Distribution of P90 drawdowns across tickers
    ax2 = axes[1]
    if percentile_data:
        p90s = sorted([p['p90'] * 100 for p in percentile_data], reverse=True)
        ax2.barh(range(len(p90s)), p90s, color='#d62728', alpha=0.7)
        tickers_sorted = [p['Ticker'] for p in sorted(percentile_data, key=lambda x: -x['p90'])]
        ax2.set_yticks(range(len(tickers_sorted)))
        ax2.set_yticklabels(tickers_sorted, fontsize=6)
        for thresh in [6, 8, 10, 12]:
            ax2.axvline(x=thresh, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
            ax2.text(thresh + 0.2, len(p90s) - 1, f'{thresh}%', fontsize=8, color='gray')
        ax2.set_xlabel('P90 Daily Max Drawdown (%)')
        ax2.set_title('P90 Drawdown by Ticker (Recent 3mo)', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('/tmp/short_spike_threshold_analysis.png', dpi=150, bbox_inches='tight')
    print(f"    Chart 1 saved: /tmp/short_spike_threshold_analysis.png")
    plt.show()

    # Chart 3: UP vs DOWN spike rates side by side
    fig2, ax3 = plt.subplots(figsize=(14, 7))
    if 'r_up_8_per_mo' in df_asym.columns:
        top30 = df_asym.head(30)
        x = np.arange(len(top30))
        w = 0.35
        ax3.barh(x - w/2, top30['r_up_8_per_mo'].values, w, label='UP 8%+ spikes/mo', color='#2ca02c', alpha=0.7)
        ax3.barh(x + w/2, top30['r_drop_8_per_mo'].values, w, label='DOWN 8%+ drops/mo', color='#d62728', alpha=0.7)
        ax3.set_yticks(x)
        ax3.set_yticklabels(top30['Ticker'].values, fontsize=8)
        ax3.set_xlabel('Events per Month (Recent 3mo)')
        ax3.set_title('UP vs DOWN 8%+ Events — Top 30 Tickers', fontsize=13, fontweight='bold')
        ax3.axvline(x=2.0, color='blue', linewidth=1, linestyle='--', alpha=0.5, label='2/mo threshold')
        ax3.legend()
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('/tmp/short_spike_up_vs_down.png', dpi=150, bbox_inches='tight')
    print(f"    Chart 2 saved: /tmp/short_spike_up_vs_down.png")
    plt.show()


# ==============================================================================
# RECOMMENDATION
# ==============================================================================
print()
print("=" * W)
print(f"  RECOMMENDATION — Exp 400 Target Threshold")
print("=" * W)

for thresh in DROP_THRESHOLDS:
    pct = int(thresh * 100)
    col = f'r_drop_{pct}_per_mo'
    viable = df_results[df_results[col] >= 2.0]
    print(f"\n  {thresh:>5.0%} target:  {len(viable):>2} tickers with ≥2/mo")
    if len(viable) > 0:
        avg_mag = df_results[df_results[col] >= 2.0][f'r_drop_{pct}_avg_mag'].mean()
        avg_rec = df_results[df_results[col] >= 2.0][f'r_drop_{pct}_avg_recovery'].mean()
        print(f"           avg magnitude when hit: {avg_mag*100:.1f}%")
        print(f"           avg recovery by EOD: {avg_rec:.2f}")
        print(f"           tickers: {', '.join(viable.sort_values(col, ascending=False)['Ticker'].tolist())}")

print(f"""
  DECISION FRAMEWORK:
  - If using SAME architecture as long-spike model:
    Pick the threshold that gives ~20 teachers with ≥2.0 events/month
    (matches Exp 230 long-side teacher count)
  - Lower recovery ratio = better (dump doesn't reverse → short stays profitable)
  - Volume surge > 1.5x confirms liquidation (not just drift)
  - Check UP vs DOWN asymmetry: tickers that dump MORE than they spike
    are better short candidates
""")

_release_memory()
print(f"  Done.")
