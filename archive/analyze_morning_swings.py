"""
analyze_morning_swings.py — Morning Swing Size Distribution (30-min window)
============================================================================
Measures the magnitude of price swings in the first 30 minutes after market
open across all tickers. Answers: what fraction of days produce 1%, 3%, 5%+
moves, and how does this vary by ticker/sector/month?

Window:  6:30 AM PST (open) → 7:00 AM PST (+30 min)
Data:    2026 only, all available tickers in GCS

Metrics per day/ticker:
  - max_up:    peak high vs open price (best-case long entry at open)
  - max_down:  worst low vs open price (worst-case long entry at open)
  - ret_30m:   close at +30m vs open
  - time_peak: minutes from open to max_up
  - time_trough: minutes from open to max_down

Output tables:
  1. Overall swing size cumulative distribution
  2. Up swing distribution by sector
  3. Down swing distribution by sector
  4. Per-ticker swing profiles (median, p75, p90, p95 up swing)
  5. Monthly evolution of swing sizes
  6. Asymmetry analysis (up/down ratio per ticker)
  7. Time-to-peak distribution (how fast do moves happen?)
  8. Trainability: UP-day prevalence at various thresholds

Colab-ready. Uses google.cloud.storage API (not FUSE).
"""

import os, gc, sys, subprocess, warnings, ctypes
from collections import defaultdict
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect(); gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "google-cloud-storage", "pyarrow", "gcsfs", "--quiet"])
    from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"
OHLCV_DIR = "Data/Inference"
INFERENCE_DIR = "Data/Inference_v2"

DATA_START = '2025-12-20'
ANALYSIS_START = '2026-01-01'
ANALYSIS_END = '2026-06-30'
MONTHS = ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06']
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

MKT_OPEN_MIN  = 390   # 6:30 AM PST
SWING_END_MIN = 420   # 7:00 AM PST (+30 min)
MKT_CLOSE_MIN = 780   # 1:00 PM PST

SWING_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

# ==============================================================================
# TICKER GROUPS
# ==============================================================================
GROUPS = {
    'deployed': [
        'HUT', 'FLNC', 'CIFR', 'CLSK', 'KEEL', 'RCAT', 'RIOT', 'APLD', 'WULF',
    ],
    'crypto': [
        'IREN', 'BTBT', 'MARA', 'ALAB', 'CORZ', 'COIN', 'MSTR', 'HOOD', 'BGDE',
    ],
    'defense_space': [
        'ASTS', 'LUNR', 'BBAI', 'RKLB', 'SPIR', 'BKSY', 'MNTS',
        'KTOS', 'AVAV', 'NNDM', 'MVIS',
    ],
    'energy_nuclear': [
        'PLUG', 'BE', 'FCEL', 'OKLO', 'SMR',
    ],
    'growth_evtol': [
        'JOBY', 'ACHR', 'RDW', 'PL', 'VOYG', 'MP',
    ],
    'large_cap': [
        'SMCI', 'AMD', 'NVDA', 'TSLA', 'PLTR', 'SOUN', 'VRT', 'VST', 'MU',
        'MOD', 'POWL', 'NBIS',
    ],
    'defense_prime': [
        'GD', 'GE', 'GEV', 'LHX', 'LMT', 'NOC', 'RTX',
    ],
    'commodity': [
        'BHP', 'CAT', 'CCJ', 'CEG', 'COP', 'CVX', 'XOM', 'FCX',
    ],
    'utility_infra': [
        'DUK', 'EQIX', 'ETN', 'LIN', 'NEE',
    ],
    'benchmark': [
        'SPY', 'QQQ', 'IWM', 'TLT',
    ],
}

TICKER_TO_GROUP = {}
ALL_TICKERS = []
for grp, tickers in GROUPS.items():
    for t in tickers:
        TICKER_TO_GROUP[t] = grp
        ALL_TICKERS.append(t)

gcs_client = gcs_storage.Client()


# ==============================================================================
# DATA LOADING
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def load_ohlcv_gcs(ticker):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for dir_path in [INFERENCE_DIR, OHLCV_DIR]:
        for suffix in ['_master_inference.parquet', '_1m_full_historical_data.parquet']:
            uri = f"gs://{BUCKET_NAME}/{dir_path}/{clean}{suffix}"
            try:
                df = pd.read_parquet(uri)
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                if dt_cols:
                    df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
                else:
                    num_col = next((c for c in df.columns
                                   if c in ('Datetime','Date','Time','Timestamp','Level_0')),
                                  df.columns[0])
                    df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
                df = df.set_index('Datetime').sort_index()
                df = df[~df.index.duplicated(keep='first')]
                std_map = {'open':'Open','high':'High','low':'Low',
                           'close':'Close','volume':'Volume'}
                lower_map = {k.lower():v for k,v in std_map.items()}
                df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                              for c in df.columns]
                start = pd.Timestamp(DATA_START, tz='UTC')
                end = pd.Timestamp(ANALYSIS_END, tz='UTC') + pd.Timedelta(days=2)
                df = df[(df.index >= start) & (df.index <= end)]
                if df.empty:
                    continue
                df = heal_data(df)
                df.index = df.index.tz_convert(PST)
                return df
            except Exception:
                continue
    return None


# ==============================================================================
# SWING EXTRACTION
# ==============================================================================
def extract_swings(df, ticker, group):
    records = []
    all_dates = sorted(set(df.index.date))
    analysis_start_date = pd.Timestamp(ANALYSIS_START).date()
    analysis_end_date = pd.Timestamp(ANALYSIS_END).date()

    for date in all_dates:
        if date < analysis_start_date or date > analysis_end_date:
            continue

        day = df[df.index.date == date]
        mins = np.array(day.index.hour * 60 + day.index.minute)

        mkt_mask = (mins >= MKT_OPEN_MIN) & (mins <= MKT_CLOSE_MIN)
        mkt_bars = day[mkt_mask]
        if len(mkt_bars) < 5:
            continue
        if mkt_bars.iloc[0].get('Volume', 1) == 0:
            continue

        open_price = mkt_bars.iloc[0]['Open']
        if open_price <= 0 or np.isnan(open_price):
            continue

        # 30-min window: open to +30m
        swing_mask = (mins >= MKT_OPEN_MIN) & (mins <= SWING_END_MIN)
        swing_bars = day[swing_mask]
        if len(swing_bars) < 5:
            continue

        highs = swing_bars['High'].values
        lows = swing_bars['Low'].values
        closes = swing_bars['Close'].values

        max_high = np.nanmax(highs)
        min_low = np.nanmin(lows)

        max_up = (max_high - open_price) / open_price
        max_down = (min_low - open_price) / open_price
        ret_30m = (closes[-1] - open_price) / open_price

        # time to peak / trough (minutes from open)
        peak_idx = np.nanargmax(highs)
        trough_idx = np.nanargmin(lows)
        swing_mins = np.array(swing_bars.index.hour * 60 + swing_bars.index.minute)
        time_peak = swing_mins[peak_idx] - MKT_OPEN_MIN
        time_trough = swing_mins[trough_idx] - MKT_OPEN_MIN

        # volume in 30-min window
        vol_30m = swing_bars['Volume'].sum()

        # pre-open gap (if we have pre-market data)
        pre_mask = (mins >= MKT_OPEN_MIN - 30) & (mins < MKT_OPEN_MIN)
        pre_bars = day[pre_mask]
        prev_close_mask = mins < MKT_OPEN_MIN
        prev_bars = day[prev_close_mask]

        gap = np.nan
        if len(prev_bars) > 0:
            prev_close = prev_bars.iloc[-1]['Close']
            if prev_close > 0:
                gap = (open_price - prev_close) / prev_close

        # EOD return (for continuation analysis)
        eod_close = mkt_bars.iloc[-1]['Close']
        ret_eod = (eod_close - open_price) / open_price

        # peak-to-close drawdown within 30 min (how much of the peak do you keep?)
        peak_ret = max_up
        hold_ratio = ret_30m / max_up if max_up > 0.001 else np.nan

        records.append({
            'date': date,
            'month': pd.Timestamp(date).strftime('%Y-%m'),
            'ticker': ticker,
            'group': group,
            'open_price': open_price,
            'max_up': max_up,
            'max_down': max_down,
            'ret_30m': ret_30m,
            'ret_eod': ret_eod,
            'time_peak': time_peak,
            'time_trough': time_trough,
            'vol_30m': vol_30m,
            'gap': gap,
            'hold_ratio': hold_ratio,
        })

    return records


# ==============================================================================
# LOAD DATA
# ==============================================================================
W = 120
print("=" * W)
print(f"  MORNING SWING SIZE ANALYSIS (30-min window)")
print(f"  Window: 6:30 AM → 7:00 AM PST | Data: Jan-Jun 2026")
print("=" * W)

print(f"\n  Loading OHLCV data...")
all_records = []
loaded = 0
failed = []

for ticker in ALL_TICKERS:
    df = load_ohlcv_gcs(ticker)
    if df is None:
        failed.append(ticker)
        continue
    group = TICKER_TO_GROUP.get(ticker, 'other')
    recs = extract_swings(df, ticker, group)
    all_records.extend(recs)
    loaded += 1
    del df
    _release_memory()

print(f"  Loaded: {loaded}/{len(ALL_TICKERS)} tickers, {len(all_records)} observations")
if failed:
    print(f"  Failed: {', '.join(failed)}")

data = pd.DataFrame(all_records)
data['abs_up'] = data['max_up'].abs() * 100
data['abs_down'] = data['max_down'].abs() * 100
data['abs_ret'] = data['ret_30m'].abs() * 100

_release_memory()


# ==============================================================================
# TABLE 1: OVERALL SWING SIZE CUMULATIVE DISTRIBUTION
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 1: CUMULATIVE DISTRIBUTION — % of days with swing >= threshold")
print("=" * W)
print(f"  N = {len(data)} ticker-days")
print()

print(f"  {'Threshold':>10}", end="")
print(f"  {'Up>=':>8}  {'Down>=':>8}  {'|Ret30m|>=':>10}  {'Ret30m>':>8}  {'Ret30m<':>9}")
print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*9}")

for thresh in SWING_THRESHOLDS:
    pct_up = (data['max_up'] >= thresh / 100).mean() * 100
    pct_down = (data['max_down'] <= -thresh / 100).mean() * 100
    pct_abs_ret = (data['ret_30m'].abs() >= thresh / 100).mean() * 100
    pct_ret_pos = (data['ret_30m'] >= thresh / 100).mean() * 100
    pct_ret_neg = (data['ret_30m'] <= -thresh / 100).mean() * 100

    print(f"  {thresh:>9.1f}%  {pct_up:>7.1f}%  {pct_down:>7.1f}%  {pct_abs_ret:>9.1f}%  "
          f"{pct_ret_pos:>7.1f}%  {pct_ret_neg:>8.1f}%")

# summary stats
print(f"\n  Summary statistics (% moves):")
for metric, label in [('max_up', 'Max up (peak)'), ('max_down', 'Max down (trough)'),
                       ('ret_30m', 'Return @30m')]:
    vals = data[metric] * 100
    print(f"    {label:<20} mean={vals.mean():+.2f}%  median={vals.median():+.2f}%  "
          f"std={vals.std():.2f}%  p5={vals.quantile(0.05):+.2f}%  p95={vals.quantile(0.95):+.2f}%")


# ==============================================================================
# TABLE 2: UP SWING DISTRIBUTION BY SECTOR
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 2: UP SWING (peak high vs open) — % of days >= threshold, by sector")
print("=" * W)

sector_order = ['deployed', 'crypto', 'defense_space', 'energy_nuclear',
                'growth_evtol', 'large_cap', 'defense_prime', 'commodity',
                'utility_infra', 'benchmark']

thresh_cols = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

print(f"  {'Sector':<17} {'N':>5}", end="")
for t in thresh_cols:
    print(f"  {f'>={t:.0f}%':>6}" if t >= 1 else f"  {f'>={t}%':>6}", end="")
print(f"  {'median':>7}  {'p90':>7}  {'p95':>7}")

print(f"  {'─'*17} {'─'*5}", end="")
for _ in thresh_cols:
    print(f"  {'─'*6}", end="")
print(f"  {'─'*7}  {'─'*7}  {'─'*7}")

for sector in sector_order:
    sdf = data[data['group'] == sector]
    if sdf.empty:
        continue
    print(f"  {sector:<17} {len(sdf):>5}", end="")
    for t in thresh_cols:
        pct = (sdf['max_up'] >= t / 100).mean() * 100
        print(f"  {pct:>5.0f}%", end="")
    med = sdf['max_up'].median() * 100
    p90 = sdf['max_up'].quantile(0.90) * 100
    p95 = sdf['max_up'].quantile(0.95) * 100
    print(f"  {med:>6.2f}%  {p90:>6.2f}%  {p95:>6.2f}%")

# total row
print(f"  {'─'*17} {'─'*5}", end="")
for _ in thresh_cols:
    print(f"  {'─'*6}", end="")
print(f"  {'─'*7}  {'─'*7}  {'─'*7}")
print(f"  {'ALL':<17} {len(data):>5}", end="")
for t in thresh_cols:
    pct = (data['max_up'] >= t / 100).mean() * 100
    print(f"  {pct:>5.0f}%", end="")
med = data['max_up'].median() * 100
p90 = data['max_up'].quantile(0.90) * 100
p95 = data['max_up'].quantile(0.95) * 100
print(f"  {med:>6.2f}%  {p90:>6.2f}%  {p95:>6.2f}%")


# ==============================================================================
# TABLE 3: DOWN SWING DISTRIBUTION BY SECTOR
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 3: DOWN SWING (worst low vs open) — % of days <= -threshold, by sector")
print("=" * W)

print(f"  {'Sector':<17} {'N':>5}", end="")
for t in thresh_cols:
    print(f"  {f'<=-{t:.0f}%':>6}" if t >= 1 else f"  {f'<=-{t}%':>7}", end="")
print(f"  {'median':>7}  {'p10':>7}  {'p5':>7}")

print(f"  {'─'*17} {'─'*5}", end="")
for _ in thresh_cols:
    print(f"  {'─'*6}", end="")
print(f"  {'─'*7}  {'─'*7}  {'─'*7}")

for sector in sector_order:
    sdf = data[data['group'] == sector]
    if sdf.empty:
        continue
    print(f"  {sector:<17} {len(sdf):>5}", end="")
    for t in thresh_cols:
        pct = (sdf['max_down'] <= -t / 100).mean() * 100
        print(f"  {pct:>5.0f}%", end="")
    med = sdf['max_down'].median() * 100
    p10 = sdf['max_down'].quantile(0.10) * 100
    p5 = sdf['max_down'].quantile(0.05) * 100
    print(f"  {med:>6.2f}%  {p10:>6.2f}%  {p5:>6.2f}%")

print(f"  {'─'*17} {'─'*5}", end="")
for _ in thresh_cols:
    print(f"  {'─'*6}", end="")
print(f"  {'─'*7}  {'─'*7}  {'─'*7}")
print(f"  {'ALL':<17} {len(data):>5}", end="")
for t in thresh_cols:
    pct = (data['max_down'] <= -t / 100).mean() * 100
    print(f"  {pct:>5.0f}%", end="")
med = data['max_down'].median() * 100
p10 = data['max_down'].quantile(0.10) * 100
p5 = data['max_down'].quantile(0.05) * 100
print(f"  {med:>6.2f}%  {p10:>6.2f}%  {p5:>6.2f}%")


# ==============================================================================
# TABLE 4: PER-TICKER SWING PROFILES
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 4: PER-TICKER UP SWING PROFILES (sorted by median)")
print("=" * W)

print(f"  {'Ticker':<6} {'Grp':<14} {'N':>4}  {'Mean':>6}  {'Med':>6}  {'p75':>6}  "
      f"{'p90':>6}  {'p95':>6}  {'>=1%':>5}  {'>=2%':>5}  {'>=3%':>5}  {'>=5%':>5}")
print(f"  {'─'*6} {'─'*14} {'─'*4}  {'─'*6}  {'─'*6}  {'─'*6}  "
      f"{'─'*6}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")

ticker_stats = []
for ticker in sorted(set(data['ticker'])):
    tdf = data[data['ticker'] == ticker]
    ups = tdf['max_up'] * 100
    ticker_stats.append({
        'ticker': ticker,
        'group': tdf.iloc[0]['group'],
        'n': len(tdf),
        'mean': ups.mean(),
        'median': ups.median(),
        'p75': ups.quantile(0.75),
        'p90': ups.quantile(0.90),
        'p95': ups.quantile(0.95),
        'pct_1': (ups >= 1).mean() * 100,
        'pct_2': (ups >= 2).mean() * 100,
        'pct_3': (ups >= 3).mean() * 100,
        'pct_5': (ups >= 5).mean() * 100,
    })

ticker_stats.sort(key=lambda x: -x['median'])

for s in ticker_stats:
    print(f"  {s['ticker']:<6} {s['group']:<14} {s['n']:>4}  {s['mean']:>5.2f}%  {s['median']:>5.2f}%  "
          f"{s['p75']:>5.2f}%  {s['p90']:>5.2f}%  {s['p95']:>5.2f}%  "
          f"{s['pct_1']:>4.0f}%  {s['pct_2']:>4.0f}%  {s['pct_3']:>4.0f}%  {s['pct_5']:>4.0f}%")


# ==============================================================================
# TABLE 5: MONTHLY EVOLUTION
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 5: MONTHLY EVOLUTION — swing size percentiles over time")
print("=" * W)

# focus on deployed + crypto (the tradeable universe)
tradeable_groups = {'deployed', 'crypto', 'defense_space', 'energy_nuclear'}
tdata = data[data['group'].isin(tradeable_groups)]

print(f"\n  UP SWINGS (tradeable universe: {len(tdata)} obs)")
print(f"  {'Month':>7} {'N':>5}  {'Mean':>6}  {'Med':>6}  {'p75':>6}  {'p90':>6}  "
      f"{'>=1%':>5}  {'>=2%':>5}  {'>=3%':>5}  {'>=5%':>5}")
print(f"  {'─'*7} {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  "
      f"{'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")

for month, mlabel in zip(MONTHS, MONTH_LABELS):
    mdf = tdata[tdata['month'] == month]
    if mdf.empty:
        continue
    ups = mdf['max_up'] * 100
    print(f"  {mlabel:>7} {len(mdf):>5}  {ups.mean():>5.2f}%  {ups.median():>5.2f}%  "
          f"{ups.quantile(0.75):>5.2f}%  {ups.quantile(0.90):>5.2f}%  "
          f"{(ups >= 1).mean()*100:>4.0f}%  {(ups >= 2).mean()*100:>4.0f}%  "
          f"{(ups >= 3).mean()*100:>4.0f}%  {(ups >= 5).mean()*100:>4.0f}%")

print(f"\n  DOWN SWINGS (tradeable universe)")
print(f"  {'Month':>7} {'N':>5}  {'Mean':>6}  {'Med':>6}  {'p25':>6}  {'p10':>6}  "
      f"{'<=-1%':>5}  {'<=-2%':>5}  {'<=-3%':>5}  {'<=-5%':>5}")
print(f"  {'─'*7} {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  "
      f"{'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")

for month, mlabel in zip(MONTHS, MONTH_LABELS):
    mdf = tdata[tdata['month'] == month]
    if mdf.empty:
        continue
    downs = mdf['max_down'] * 100
    print(f"  {mlabel:>7} {len(mdf):>5}  {downs.mean():>5.2f}%  {downs.median():>5.2f}%  "
          f"{downs.quantile(0.25):>5.2f}%  {downs.quantile(0.10):>5.2f}%  "
          f"{(downs <= -1).mean()*100:>4.0f}%  {(downs <= -2).mean()*100:>4.0f}%  "
          f"{(downs <= -3).mean()*100:>4.0f}%  {(downs <= -5).mean()*100:>4.0f}%")

print(f"\n  NET RETURN @30m (tradeable universe)")
print(f"  {'Month':>7} {'N':>5}  {'Mean':>7}  {'Med':>7}  "
      f"{'%>0':>5}  {'%>+1%':>6}  {'%>+2%':>6}  {'%<-1%':>6}  {'%<-2%':>6}")
print(f"  {'─'*7} {'─'*5}  {'─'*7}  {'─'*7}  "
      f"{'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")

for month, mlabel in zip(MONTHS, MONTH_LABELS):
    mdf = tdata[tdata['month'] == month]
    if mdf.empty:
        continue
    rets = mdf['ret_30m'] * 100
    print(f"  {mlabel:>7} {len(mdf):>5}  {rets.mean():>+6.2f}%  {rets.median():>+6.2f}%  "
          f"{(rets > 0).mean()*100:>4.0f}%  "
          f"{(rets > 1).mean()*100:>5.0f}%  {(rets > 2).mean()*100:>5.0f}%  "
          f"{(rets < -1).mean()*100:>5.0f}%  {(rets < -2).mean()*100:>5.0f}%")


# ==============================================================================
# TABLE 6: UP/DOWN ASYMMETRY PER TICKER
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 6: UP/DOWN ASYMMETRY — ratio of median up swing to median |down swing|")
print("=" * W)
print(f"  Ratio > 1.0 = upside-biased, < 1.0 = downside-biased")
print()

print(f"  {'Ticker':<6} {'Grp':<14}  {'MedUp':>6}  {'Med|Dn|':>7}  {'Ratio':>6}  {'Bias':>12}  "
      f"{'MeanUp':>7}  {'Mean|Dn|':>8}  {'MeanRatio':>9}")
print(f"  {'─'*6} {'─'*14}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*12}  "
      f"{'─'*7}  {'─'*8}  {'─'*9}")

asym_stats = []
for ticker in sorted(set(data['ticker'])):
    tdf = data[data['ticker'] == ticker]
    med_up = tdf['max_up'].median() * 100
    med_dn = abs(tdf['max_down'].median()) * 100
    mean_up = tdf['max_up'].mean() * 100
    mean_dn = abs(tdf['max_down'].mean()) * 100
    ratio = med_up / med_dn if med_dn > 0.01 else 99
    mean_ratio = mean_up / mean_dn if mean_dn > 0.01 else 99

    if ratio > 1.15:
        bias = "UP-BIASED"
    elif ratio < 0.85:
        bias = "DOWN-BIASED"
    else:
        bias = "SYMMETRIC"

    asym_stats.append({
        'ticker': ticker, 'group': tdf.iloc[0]['group'],
        'med_up': med_up, 'med_dn': med_dn, 'ratio': ratio,
        'mean_up': mean_up, 'mean_dn': mean_dn, 'mean_ratio': mean_ratio,
        'bias': bias,
    })

asym_stats.sort(key=lambda x: -x['ratio'])

for s in asym_stats:
    print(f"  {s['ticker']:<6} {s['group']:<14}  {s['med_up']:>5.2f}%  {s['med_dn']:>6.2f}%  "
          f"{s['ratio']:>5.2f}  {s['bias']:>12}  "
          f"{s['mean_up']:>6.2f}%  {s['mean_dn']:>7.2f}%  {s['mean_ratio']:>8.2f}")

n_up = sum(1 for s in asym_stats if s['bias'] == 'UP-BIASED')
n_dn = sum(1 for s in asym_stats if s['bias'] == 'DOWN-BIASED')
n_sym = sum(1 for s in asym_stats if s['bias'] == 'SYMMETRIC')
print(f"\n  Summary: {n_up} up-biased, {n_dn} down-biased, {n_sym} symmetric")


# ==============================================================================
# TABLE 7: TIME-TO-PEAK DISTRIBUTION
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 7: TIME-TO-PEAK — how quickly do up moves happen? (minutes from open)")
print("=" * W)

# only consider days with meaningful up moves (>=1%)
big_up = data[data['max_up'] >= 0.01].copy()
print(f"  Days with >=1% up swing: {len(big_up)} ({len(big_up)/len(data)*100:.1f}% of all)")
print()

time_bins = [0, 5, 10, 15, 20, 25, 30]
time_labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20-25m', '25-30m']

print(f"  {'Sector':<17} {'N':>5}", end="")
for tl in time_labels:
    print(f"  {tl:>7}", end="")
print(f"  {'MeanMin':>8}")

print(f"  {'─'*17} {'─'*5}", end="")
for _ in time_labels:
    print(f"  {'─'*7}", end="")
print(f"  {'─'*8}")

for sector in sector_order:
    sdf = big_up[big_up['group'] == sector]
    if len(sdf) < 5:
        continue
    print(f"  {sector:<17} {len(sdf):>5}", end="")
    for i in range(len(time_bins) - 1):
        lo, hi = time_bins[i], time_bins[i + 1]
        pct = ((sdf['time_peak'] >= lo) & (sdf['time_peak'] < hi)).mean() * 100
        print(f"  {pct:>6.1f}%", end="")
    print(f"  {sdf['time_peak'].mean():>7.1f}m")

# total
print(f"  {'─'*17} {'─'*5}", end="")
for _ in time_labels:
    print(f"  {'─'*7}", end="")
print(f"  {'─'*8}")
print(f"  {'ALL':<17} {len(big_up):>5}", end="")
for i in range(len(time_bins) - 1):
    lo, hi = time_bins[i], time_bins[i + 1]
    pct = ((big_up['time_peak'] >= lo) & (big_up['time_peak'] < hi)).mean() * 100
    print(f"  {pct:>6.1f}%", end="")
print(f"  {big_up['time_peak'].mean():>7.1f}m")


# ==============================================================================
# TABLE 8: CONTINUATION — does +30m return predict EOD return?
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 8: CONTINUATION — if UP at +30m, what happens by EOD?")
print("=" * W)

ret_bins = [0.5, 1.0, 2.0, 3.0, 5.0]

print(f"  {'30m Return':>12}  {'N':>5}  {'EOD Mean':>8}  {'EOD Med':>8}  "
      f"{'%EOD>0':>7}  {'%EOD>+1%':>8}  {'%Gives Back':>11}")
print(f"  {'─'*12}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*11}")

for thresh in ret_bins:
    up_days = tdata[tdata['ret_30m'] >= thresh / 100]
    if len(up_days) < 5:
        print(f"  {f'>+{thresh}%':>12}  {len(up_days):>5}  {'—':>8}  {'—':>8}  {'—':>7}  {'—':>8}  {'—':>11}")
        continue
    eod = up_days['ret_eod'] * 100
    gives_back = (up_days['ret_eod'] < up_days['ret_30m']).mean() * 100
    print(f"  {f'>+{thresh}%':>12}  {len(up_days):>5}  {eod.mean():>+7.2f}%  {eod.median():>+7.2f}%  "
          f"{(eod > 0).mean()*100:>6.0f}%  {(eod > 1).mean()*100:>7.0f}%  {gives_back:>10.0f}%")

# same for down
print()
for thresh in ret_bins:
    dn_days = tdata[tdata['ret_30m'] <= -thresh / 100]
    if len(dn_days) < 5:
        print(f"  {f'<-{thresh}%':>12}  {len(dn_days):>5}  {'—':>8}  {'—':>8}  {'—':>7}  {'—':>8}  {'—':>11}")
        continue
    eod = dn_days['ret_eod'] * 100
    worsens = (dn_days['ret_eod'] < dn_days['ret_30m']).mean() * 100
    print(f"  {f'<-{thresh}%':>12}  {len(dn_days):>5}  {eod.mean():>+7.2f}%  {eod.median():>+7.2f}%  "
          f"{(eod > 0).mean()*100:>6.0f}%  {(eod > 1).mean()*100:>7.0f}%  {worsens:>10.0f}%")


# ==============================================================================
# TABLE 9: TRAINABILITY — UP-day prevalence at each threshold
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 9: TRAINABILITY — class balance for binary UP model")
print("=" * W)
print(f"  'UP' = max peak within 30 min >= threshold")
print(f"  Ideal class balance for training: 15-35% positive rate")
print()

# by sector for key thresholds
key_thresholds = [1.0, 1.5, 2.0, 3.0, 5.0]

print(f"  {'Sector':<17} {'N':>5}", end="")
for t in key_thresholds:
    print(f"  {'UP@'+f'{t}%':>7}", end="")
print()
print(f"  {'─'*17} {'─'*5}", end="")
for _ in key_thresholds:
    print(f"  {'─'*7}", end="")
print()

for sector in sector_order:
    sdf = data[data['group'] == sector]
    if sdf.empty:
        continue
    print(f"  {sector:<17} {len(sdf):>5}", end="")
    for t in key_thresholds:
        rate = (sdf['max_up'] >= t / 100).mean() * 100
        marker = " *" if 15 <= rate <= 35 else "  " if 5 <= rate <= 50 else " !"
        print(f"  {rate:>5.0f}%{marker[1]}", end="")
    print()

print(f"  {'─'*17} {'─'*5}", end="")
for _ in key_thresholds:
    print(f"  {'─'*7}", end="")
print()
print(f"  {'ALL':<17} {len(data):>5}", end="")
for t in key_thresholds:
    rate = (data['max_up'] >= t / 100).mean() * 100
    print(f"  {rate:>5.0f}%", end="")
print()

print(f"\n  * = ideal training range (15-35%)")

# best threshold per sector
print(f"\n  Best UP threshold per sector (closest to 25% positive rate):")
for sector in sector_order:
    sdf = data[data['group'] == sector]
    if sdf.empty:
        continue
    best_t = None
    best_dist = 999
    for t_pct in np.arange(0.5, 10.1, 0.25):
        rate = (sdf['max_up'] >= t_pct / 100).mean()
        dist = abs(rate - 0.25)
        if dist < best_dist:
            best_dist = dist
            best_t = t_pct
    actual_rate = (sdf['max_up'] >= best_t / 100).mean() * 100
    print(f"    {sector:<17}  threshold={best_t:.1f}%  → {actual_rate:.0f}% positive rate")


# ==============================================================================
# TABLE 10: HOLD RATIO — what fraction of peak is kept at +30m?
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 10: HOLD RATIO — fraction of peak up move retained at +30m")
print("=" * W)
print(f"  Hold ratio = ret_30m / max_up  (1.0 = held all gains, 0.0 = round-tripped)")
print()

# only days with meaningful up moves
sig_up = data[data['max_up'] >= 0.01].copy()

print(f"  {'Sector':<17} {'N':>5}  {'MeanHold':>8}  {'MedHold':>8}  "
      f"{'%Hold>0.8':>9}  {'%Hold>0.5':>9}  {'%Hold<0':>8}")
print(f"  {'─'*17} {'─'*5}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*8}")

for sector in sector_order:
    sdf = sig_up[sig_up['group'] == sector]
    if len(sdf) < 5:
        continue
    hr = sdf['hold_ratio']
    print(f"  {sector:<17} {len(sdf):>5}  {hr.mean():>7.2f}  {hr.median():>7.2f}  "
          f"{(hr > 0.8).mean()*100:>8.0f}%  {(hr > 0.5).mean()*100:>8.0f}%  "
          f"{(hr < 0).mean()*100:>7.0f}%")

print(f"  {'─'*17} {'─'*5}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*8}")
hr = sig_up['hold_ratio']
print(f"  {'ALL':<17} {len(sig_up):>5}  {hr.mean():>7.2f}  {hr.median():>7.2f}  "
      f"{(hr > 0.8).mean()*100:>8.0f}%  {(hr > 0.5).mean()*100:>8.0f}%  "
      f"{(hr < 0).mean()*100:>7.0f}%")


# ==============================================================================
# VERDICT
# ==============================================================================
print()
print("=" * W)
print(f"  VERDICT")
print("=" * W)

# compute key stats for tradeable universe
t_ups = tdata['max_up'] * 100
t_rets = tdata['ret_30m'] * 100

pct_1up = (tdata['max_up'] >= 0.01).mean() * 100
pct_2up = (tdata['max_up'] >= 0.02).mean() * 100
pct_3up = (tdata['max_up'] >= 0.03).mean() * 100
pct_5up = (tdata['max_up'] >= 0.05).mean() * 100

print(f"""
  Tradeable universe ({len(tdata)} ticker-days):
    >=1% up swing:  {pct_1up:.0f}% of days
    >=2% up swing:  {pct_2up:.0f}% of days
    >=3% up swing:  {pct_3up:.0f}% of days
    >=5% up swing:  {pct_5up:.0f}% of days

  Median up swing: {t_ups.median():.2f}%
  Mean up swing:   {t_ups.mean():.2f}%
  Mean ret @30m:   {t_rets.mean():+.2f}%
""")

# find best threshold for ~25% rate
best_t_overall = None
best_dist_overall = 999
for t_pct in np.arange(0.5, 10.1, 0.25):
    rate = (tdata['max_up'] >= t_pct / 100).mean()
    dist = abs(rate - 0.25)
    if dist < best_dist_overall:
        best_dist_overall = dist
        best_t_overall = t_pct

actual = (tdata['max_up'] >= best_t_overall / 100).mean() * 100
print(f"  Recommended UP threshold for tradeable universe: {best_t_overall:.1f}%")
print(f"    → {actual:.0f}% positive rate (target: ~25%)")
print(f"    → Binary model: predict 'will peak exceed {best_t_overall:.1f}% within 30 min of open?'")

# per-sector recommendations
print(f"\n  Per-sector threshold recommendations (targeting ~25% positive rate):")
for sector in ['deployed', 'crypto', 'defense_space', 'energy_nuclear']:
    sdf = data[data['group'] == sector]
    best_t = None
    best_d = 999
    for t_pct in np.arange(0.5, 10.1, 0.25):
        rate = (sdf['max_up'] >= t_pct / 100).mean()
        d = abs(rate - 0.25)
        if d < best_d:
            best_d = d
            best_t = t_pct
    rate = (sdf['max_up'] >= best_t / 100).mean() * 100
    print(f"    {sector:<17}  {best_t:.1f}%  → {rate:.0f}% positive")

print()
print("=" * W)
print(f"  Analysis complete.")
print("=" * W)
