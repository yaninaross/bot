"""
analyze_sector_spikes.py — Sector Spike Profile Analysis
=========================================================
For each sector receiving capital inflows (from sector rotation analysis),
evaluate daily spike frequency and magnitude of individual stocks.

Goal: identify sectors where the intraday spike model could work.
Compares all sectors to the crypto mining baseline.

Data: Polygon daily bars, 2-year window (Jul 2024 → Jul 2026).
Cached at /tmp/sector_spike_daily.pkl for fast re-runs.

Tables:
  A: Sector ETF daily range comparison (avg, p75, p90, max)
  B: Per-stock spike frequency at 3%, 5%, 8%, 10% thresholds
  C: Sector-level spike summary (avg across stocks per sector)
  D: Top 30 stocks ranked by 5%+ upside spike frequency
  E: Crypto miners vs top candidates — head-to-head
  F: Model feasibility scorecard (spike freq × inflow rank)

Charts:
  1: Sector ETF average daily range (bar chart)
  2: Spike frequency by sector at each threshold
  3: Scatter: 5%+ spike frequency vs avg spike magnitude
  4: Distribution of daily upside moves by sector (box plot)

Colab-ready (no if __name__).
"""

import time, gc, ctypes, sys, subprocess, warnings, pickle, os
from collections import defaultdict
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
CACHE_FILE = '/tmp/sector_spike_daily.pkl'

DATA_START = '2024-07-01'
DATA_END   = '2026-07-05'

SPIKE_THRESHOLDS = [0.03, 0.05, 0.08, 0.10]

SECTOR_STOCKS = {
    'Crypto Mining': {
        'etf': 'WGMI',
        'jun_rs': -9.5,
        'stocks': ['HUT', 'CIFR', 'CLSK', 'RIOT', 'WULF', 'MARA', 'KEEL', 'APLD',
                    'BTBT', 'CORZ'],
    },
    'Semiconductors': {
        'etf': 'SMH',
        'jun_rs': +9.6,
        'stocks': ['NVDA', 'AMD', 'SMCI', 'MU', 'MRVL', 'AVGO', 'VRT', 'ANET'],
    },
    'Defense / Aerospace': {
        'etf': 'ITA',
        'jun_rs': +7.1,
        'stocks': ['KTOS', 'LUNR', 'RKLB', 'JOBY', 'ACHR', 'LMT', 'RTX', 'NOC'],
    },
    'Utilities / Nuclear': {
        'etf': 'XLU',
        'jun_rs': +6.9,
        'stocks': ['VST', 'CEG', 'OKLO', 'SMR', 'NEE', 'DUK', 'NRG'],
    },
    'Healthcare / Biotech': {
        'etf': 'XLV',
        'jun_rs': +9.0,
        'stocks': ['MRNA', 'CRSP', 'HIMS', 'EXAS', 'BMRN', 'UNH', 'LLY'],
    },
    'Financials / Fintech': {
        'etf': 'XLF',
        'jun_rs': +5.9,
        'stocks': ['HOOD', 'COIN', 'SOFI', 'SQ', 'GS', 'MS', 'JPM'],
    },
    'Industrials': {
        'etf': 'XLI',
        'jun_rs': +9.1,
        'stocks': ['GEV', 'ETN', 'UBER', 'GE', 'CAT', 'PLTR'],
    },
}

ALL_ETFS = list(set(s['etf'] for s in SECTOR_STOCKS.values())) + ['SPY']
ALL_STOCKS = []
STOCK_SECTOR = {}
for sector, cfg in SECTOR_STOCKS.items():
    for s in cfg['stocks']:
        if s not in STOCK_SECTOR:
            ALL_STOCKS.append(s)
            STOCK_SECTOR[s] = sector

W = 110
poly_client = RESTClient(POLYGON_API_KEY)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def fetch_daily_bars(ticker, start, end, sleep_secs=2):
    all_bars = []
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    try:
        for a in poly_client.list_aggs(ticker, 1, 'day', start_dt, end_dt, limit=50000):
            all_bars.append({
                'Date': pd.to_datetime(a.timestamp, unit='ms').normalize(),
                'Open': a.open, 'High': a.high, 'Low': a.low,
                'Close': a.close, 'Volume': a.volume,
            })
    except Exception as e:
        print(f"    {ticker}: fetch failed ({e}), retrying...")
        time.sleep(10)
        try:
            for a in poly_client.list_aggs(ticker, 1, 'day', start_dt, end_dt, limit=50000):
                all_bars.append({
                    'Date': pd.to_datetime(a.timestamp, unit='ms').normalize(),
                    'Open': a.open, 'High': a.high, 'Low': a.low,
                    'Close': a.close, 'Volume': a.volume,
                })
        except Exception as e2:
            print(f"    {ticker}: FAILED ({e2})")
            return None
    if not all_bars:
        print(f"    {ticker}: no bars returned")
        return None
    df = pd.DataFrame(all_bars).set_index('Date').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    time.sleep(sleep_secs)
    return df


def load_all_data():
    if os.path.exists(CACHE_FILE):
        print(f"  Loading cached data from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        cached_tickers = set(data.keys())
        all_needed = set(ALL_ETFS) | set(ALL_STOCKS)
        missing = all_needed - cached_tickers
        if not missing:
            print(f"    {len(data)} tickers loaded from cache")
            return data
        print(f"    {len(cached_tickers)} cached, {len(missing)} missing: {sorted(missing)}")
    else:
        data = {}
        missing = set(ALL_ETFS) | set(ALL_STOCKS)

    print(f"  Downloading {len(missing)} tickers from Polygon (daily bars)...")
    for i, ticker in enumerate(sorted(missing)):
        print(f"    [{i+1}/{len(missing)}] {ticker}...", end=" ")
        df = fetch_daily_bars(ticker, DATA_START, DATA_END)
        if df is not None:
            data[ticker] = df
            print(f"{len(df)} bars")
        else:
            print("SKIP")

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Cached {len(data)} tickers to {CACHE_FILE}")
    return data


# ==============================================================================
# SPIKE METRICS
# ==============================================================================
def compute_spike_metrics(df):
    if df is None or len(df) < 20:
        return None

    df = df[df['Volume'] > 0].copy()
    if len(df) < 20:
        return None

    df['Daily_Range'] = (df['High'] - df['Low']) / df['Open']
    df['Upside'] = (df['High'] - df['Open']) / df['Open']
    df['Downside'] = (df['Open'] - df['Low']) / df['Open']
    df['Close_Return'] = df['Close'].pct_change()

    n_days = len(df)
    metrics = {
        'n_days': n_days,
        'avg_daily_range': df['Daily_Range'].mean(),
        'p75_daily_range': df['Daily_Range'].quantile(0.75),
        'p90_daily_range': df['Daily_Range'].quantile(0.90),
        'max_daily_range': df['Daily_Range'].max(),
        'avg_upside': df['Upside'].mean(),
        'p90_upside': df['Upside'].quantile(0.90),
        'max_upside': df['Upside'].max(),
        'avg_downside': df['Downside'].mean(),
        'avg_volume': df['Volume'].mean(),
        'volatility_20d': df['Close_Return'].rolling(20).std().mean() * np.sqrt(252),
    }

    for thresh in SPIKE_THRESHOLDS:
        pct = int(thresh * 100)
        spike_days = (df['Upside'] >= thresh).sum()
        metrics[f'spike_{pct}pct_count'] = spike_days
        metrics[f'spike_{pct}pct_freq'] = spike_days / n_days
        metrics[f'spike_{pct}pct_per_month'] = spike_days / max(1, n_days / 21)

        if spike_days > 0:
            spike_mask = df['Upside'] >= thresh
            metrics[f'spike_{pct}pct_avg_mag'] = df.loc[spike_mask, 'Upside'].mean()
            metrics[f'spike_{pct}pct_avg_dd'] = df.loc[spike_mask, 'Downside'].mean()
        else:
            metrics[f'spike_{pct}pct_avg_mag'] = 0
            metrics[f'spike_{pct}pct_avg_dd'] = 0

    jun_mask = (df.index >= '2026-06-01') & (df.index <= '2026-06-30')
    jun = df[jun_mask]
    if len(jun) > 0:
        metrics['jun_avg_range'] = jun['Daily_Range'].mean()
        metrics['jun_avg_upside'] = jun['Upside'].mean()
        for thresh in SPIKE_THRESHOLDS:
            pct = int(thresh * 100)
            metrics[f'jun_spike_{pct}pct'] = (jun['Upside'] >= thresh).sum()
    else:
        metrics['jun_avg_range'] = 0
        metrics['jun_avg_upside'] = 0
        for thresh in SPIKE_THRESHOLDS:
            pct = int(thresh * 100)
            metrics[f'jun_spike_{pct}pct'] = 0

    return metrics


# ==============================================================================
# MAIN
# ==============================================================================
print("=" * W)
print(f"  SECTOR SPIKE PROFILE ANALYSIS — Evaluating Inflow Sectors for Model Feasibility")
print("=" * W)

data = load_all_data()

print(f"\n  Computing spike metrics for {len(ALL_STOCKS)} stocks + {len(ALL_ETFS)} ETFs...")

etf_metrics = {}
for etf in ALL_ETFS:
    if etf in data:
        m = compute_spike_metrics(data[etf])
        if m:
            etf_metrics[etf] = m

stock_metrics = {}
for stock in ALL_STOCKS:
    if stock in data:
        m = compute_spike_metrics(data[stock])
        if m:
            stock_metrics[stock] = m
            sector = STOCK_SECTOR[stock]
        else:
            print(f"    {stock}: insufficient data")
    else:
        print(f"    {stock}: not in cache")


# ==============================================================================
# TABLE A: SECTOR ETF DAILY RANGE COMPARISON
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE A: SECTOR ETF DAILY RANGE — (High-Low)/Open")
print("=" * W)

sorted_etfs = sorted(etf_metrics.keys(),
                     key=lambda e: etf_metrics[e]['avg_daily_range'], reverse=True)
print(f"\n  {'ETF':<6} {'Sector':<22} {'Jun RS':>6} {'AvgRng':>7} {'P75':>7} "
      f"{'P90':>7} {'MaxRng':>7} {'AvgUp':>7} {'P90Up':>7} {'JunRng':>7}")
print(f"  {'─'*6} {'─'*22} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

for etf in sorted_etfs:
    m = etf_metrics[etf]
    sector_name = '—'
    jun_rs = 0
    for sn, cfg in SECTOR_STOCKS.items():
        if cfg['etf'] == etf:
            sector_name = sn[:22]
            jun_rs = cfg['jun_rs']
            break
    if etf == 'SPY':
        sector_name = 'Benchmark'

    print(f"  {etf:<6} {sector_name:<22} {jun_rs:>+5.1f}% "
          f"{m['avg_daily_range']:>6.1%} {m['p75_daily_range']:>6.1%} "
          f"{m['p90_daily_range']:>6.1%} {m['max_daily_range']:>6.1%} "
          f"{m['avg_upside']:>6.1%} {m['p90_upside']:>6.1%} "
          f"{m.get('jun_avg_range', 0):>6.1%}")


# ==============================================================================
# TABLE B: PER-STOCK SPIKE FREQUENCY
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE B: PER-STOCK SPIKE FREQUENCY — Upside (High-Open)/Open >= Threshold")
print("=" * W)

for sector_name, cfg in SECTOR_STOCKS.items():
    etf = cfg['etf']
    jun_rs = cfg['jun_rs']
    marker = '★' if jun_rs > 0 else '▼'
    print(f"\n  {marker} {sector_name} ({etf}, Jun RS: {jun_rs:+.1f}%)")
    print(f"    {'Stock':<7} {'Days':>5} {'AvgRng':>7} {'AvgUp':>6} "
          f"{'≥3%':>6} {'≥5%':>6} {'≥8%':>6} {'≥10%':>6} "
          f"{'5%/mo':>6} {'10%/mo':>7} {'Annl σ':>7}")
    print(f"    {'─'*7} {'─'*5} {'─'*7} {'─'*6} "
          f"{'─'*6} {'─'*6} {'─'*6} {'─'*6} "
          f"{'─'*6} {'─'*7} {'─'*7}")

    sector_stocks = [s for s in cfg['stocks'] if s in stock_metrics]
    sector_stocks.sort(key=lambda s: stock_metrics[s]['spike_5pct_freq'], reverse=True)

    for stock in sector_stocks:
        m = stock_metrics[stock]
        vol = m.get('volatility_20d', 0)
        print(f"    {stock:<7} {m['n_days']:>5} {m['avg_daily_range']:>6.1%} "
              f"{m['avg_upside']:>5.1%} "
              f"{m['spike_3pct_freq']:>5.1%} {m['spike_5pct_freq']:>5.1%} "
              f"{m['spike_8pct_freq']:>5.1%} {m['spike_10pct_freq']:>5.1%} "
              f"{m['spike_5pct_per_month']:>5.1f} {m['spike_10pct_per_month']:>6.2f} "
              f"{vol:>6.0%}")

    if sector_stocks:
        avg_5 = np.mean([stock_metrics[s]['spike_5pct_freq'] for s in sector_stocks])
        avg_10 = np.mean([stock_metrics[s]['spike_10pct_freq'] for s in sector_stocks])
        print(f"    {'SECTOR AVG':<7} {'':>5} {'':>7} {'':>6} "
              f"{'':>6} {avg_5:>5.1%} {'':>6} {avg_10:>5.1%}")


# ==============================================================================
# TABLE C: SECTOR-LEVEL SPIKE SUMMARY
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE C: SECTOR-LEVEL SPIKE SUMMARY — Averages Across Stocks")
print("=" * W)

sector_summaries = {}
for sector_name, cfg in SECTOR_STOCKS.items():
    stocks = [s for s in cfg['stocks'] if s in stock_metrics]
    if not stocks:
        continue
    summary = {
        'n_stocks': len(stocks),
        'etf': cfg['etf'],
        'jun_rs': cfg['jun_rs'],
    }
    for key in ['avg_daily_range', 'avg_upside', 'spike_3pct_freq', 'spike_5pct_freq',
                'spike_8pct_freq', 'spike_10pct_freq', 'spike_5pct_per_month',
                'spike_10pct_per_month', 'volatility_20d',
                'jun_avg_range', 'jun_avg_upside']:
        vals = [stock_metrics[s].get(key, 0) for s in stocks]
        summary[key] = np.mean(vals)
    sector_summaries[sector_name] = summary

sorted_sectors = sorted(sector_summaries.keys(),
                        key=lambda s: sector_summaries[s]['spike_5pct_freq'], reverse=True)

print(f"\n  {'Sector':<24} {'ETF':>4} {'JunRS':>6} {'#Stk':>4} "
      f"{'AvgRng':>7} {'≥3%':>6} {'≥5%':>6} {'≥8%':>6} {'≥10%':>6} "
      f"{'5%/mo':>6} {'10%/mo':>7} {'σ':>6}")
print(f"  {'─'*24} {'─'*4} {'─'*6} {'─'*4} "
      f"{'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*6} "
      f"{'─'*6} {'─'*7} {'─'*6}")

for sector in sorted_sectors:
    s = sector_summaries[sector]
    marker = '★' if s['jun_rs'] > 0 else ' '
    print(f" {marker}{sector:<24} {s['etf']:>4} {s['jun_rs']:>+5.1f}% {s['n_stocks']:>4} "
          f"{s['avg_daily_range']:>6.1%} {s['spike_3pct_freq']:>5.1%} "
          f"{s['spike_5pct_freq']:>5.1%} {s['spike_8pct_freq']:>5.1%} "
          f"{s['spike_10pct_freq']:>5.1%} {s['spike_5pct_per_month']:>5.1f} "
          f"{s['spike_10pct_per_month']:>6.2f} {s['volatility_20d']:>5.0%}")

crypto = sector_summaries.get('Crypto Mining', {})
if crypto:
    print(f"\n  Crypto Mining baseline: {crypto['spike_5pct_freq']:.1%} of days have ≥5% upside, "
          f"{crypto['spike_10pct_freq']:.1%} have ≥10%")
    print(f"  That is {crypto['spike_5pct_per_month']:.1f} spike-5% days/month and "
          f"{crypto['spike_10pct_per_month']:.2f} spike-10% days/month")


# ==============================================================================
# TABLE D: TOP 30 STOCKS BY 5%+ SPIKE FREQUENCY
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE D: TOP 30 STOCKS — Ranked by 5%+ Upside Spike Frequency")
print("=" * W)

all_ranked = sorted(stock_metrics.items(),
                    key=lambda x: x[1]['spike_5pct_freq'], reverse=True)

print(f"\n  {'Rk':>3} {'Stock':<7} {'Sector':<22} {'Days':>5} "
      f"{'AvgRng':>7} {'≥5%':>6} {'≥10%':>6} "
      f"{'5%/mo':>6} {'10%/mo':>7} {'AvgSpk5':>8} {'AvgDD5':>7} {'σ':>6}")
print(f"  {'─'*3} {'─'*7} {'─'*22} {'─'*5} "
      f"{'─'*7} {'─'*6} {'─'*6} "
      f"{'─'*6} {'─'*7} {'─'*8} {'─'*7} {'─'*6}")

for rank, (stock, m) in enumerate(all_ranked[:30], 1):
    sector = STOCK_SECTOR.get(stock, '?')[:22]
    vol = m.get('volatility_20d', 0)
    is_crypto = 'Crypto' in STOCK_SECTOR.get(stock, '')
    marker = '●' if is_crypto else '○'
    print(f"  {rank:>3} {marker}{stock:<6} {sector:<22} {m['n_days']:>5} "
          f"{m['avg_daily_range']:>6.1%} {m['spike_5pct_freq']:>5.1%} "
          f"{m['spike_10pct_freq']:>5.1%} {m['spike_5pct_per_month']:>5.1f} "
          f"{m['spike_10pct_per_month']:>6.2f} {m['spike_5pct_avg_mag']:>7.1%} "
          f"{m['spike_5pct_avg_dd']:>6.1%} {vol:>5.0%}")

print(f"\n  ● = crypto mining (baseline)   ○ = other sector")


# ==============================================================================
# TABLE E: CRYPTO MINERS vs TOP NON-CRYPTO CANDIDATES
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE E: HEAD-TO-HEAD — Crypto Miners vs Top Non-Crypto Candidates")
print("=" * W)

crypto_stocks = [s for s in ALL_STOCKS if 'Crypto' in STOCK_SECTOR.get(s, '') and s in stock_metrics]
non_crypto = [(s, m) for s, m in all_ranked if 'Crypto' not in STOCK_SECTOR.get(s, '')]
top_non_crypto = [s for s, m in non_crypto[:15]]

print(f"\n  CRYPTO MINERS (deployed model baseline):")
print(f"    {'Stock':<7} {'AvgRng':>7} {'≥5%':>6} {'≥10%':>6} {'5%/mo':>6} "
      f"{'10%/mo':>7} {'AvgSpkMag':>10} {'JunRng':>7} {'Jun5%':>5}")
print(f"    {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*10} {'─'*7} {'─'*5}")
for stock in sorted(crypto_stocks, key=lambda s: stock_metrics[s]['spike_5pct_freq'], reverse=True):
    m = stock_metrics[stock]
    print(f"    {stock:<7} {m['avg_daily_range']:>6.1%} {m['spike_5pct_freq']:>5.1%} "
          f"{m['spike_10pct_freq']:>5.1%} {m['spike_5pct_per_month']:>5.1f} "
          f"{m['spike_10pct_per_month']:>6.2f} {m['spike_5pct_avg_mag']:>9.1%} "
          f"{m.get('jun_avg_range', 0):>6.1%} {m.get('jun_spike_5pct', 0):>5}")

print(f"\n  TOP NON-CRYPTO CANDIDATES:")
print(f"    {'Stock':<7} {'Sector':<20} {'AvgRng':>7} {'≥5%':>6} {'≥10%':>6} {'5%/mo':>6} "
      f"{'10%/mo':>7} {'AvgSpkMag':>10} {'JunRng':>7} {'Jun5%':>5}")
print(f"    {'─'*7} {'─'*20} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*10} {'─'*7} {'─'*5}")
for stock in top_non_crypto:
    m = stock_metrics[stock]
    sector = STOCK_SECTOR.get(stock, '?')[:20]
    print(f"    {stock:<7} {sector:<20} {m['avg_daily_range']:>6.1%} {m['spike_5pct_freq']:>5.1%} "
          f"{m['spike_10pct_freq']:>5.1%} {m['spike_5pct_per_month']:>5.1f} "
          f"{m['spike_10pct_per_month']:>6.2f} {m['spike_5pct_avg_mag']:>9.1%} "
          f"{m.get('jun_avg_range', 0):>6.1%} {m.get('jun_spike_5pct', 0):>5}")


# ==============================================================================
# TABLE F: MODEL FEASIBILITY SCORECARD
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE F: MODEL FEASIBILITY SCORECARD")
print(f"  (Combining spike frequency with sector inflow strength)")
print("=" * W)

print(f"\n  Scoring: Feasibility = spike_5pct_per_month × (1 + jun_rs/10)")
print(f"  Higher = better candidate for model expansion")
print(f"  Minimum viable: ≥2 spike-5% days/month AND from positive-RS sector\n")

feasibility = []
for stock, m in stock_metrics.items():
    sector = STOCK_SECTOR.get(stock, '?')
    jun_rs = 0
    for sn, cfg in SECTOR_STOCKS.items():
        if stock in cfg['stocks']:
            jun_rs = cfg['jun_rs']
            break
    spike_pm = m['spike_5pct_per_month']
    score = spike_pm * (1 + jun_rs / 10) if jun_rs > 0 else spike_pm * 0.5
    feasibility.append({
        'stock': stock,
        'sector': sector,
        'jun_rs': jun_rs,
        'spike_5pct_pm': spike_pm,
        'spike_10pct_pm': m['spike_10pct_per_month'],
        'avg_range': m['avg_daily_range'],
        'score': score,
        'is_crypto': 'Crypto' in sector,
    })

feasibility.sort(key=lambda x: x['score'], reverse=True)

print(f"  {'Rk':>3} {'Stock':<7} {'Sector':<22} {'JunRS':>6} "
      f"{'5%/mo':>6} {'10%/mo':>7} {'AvgRng':>7} {'Score':>7} {'Verdict':<20}")
print(f"  {'─'*3} {'─'*7} {'─'*22} {'─'*6} "
      f"{'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*20}")

for rank, f in enumerate(feasibility[:30], 1):
    if f['is_crypto']:
        verdict = 'BASELINE (deployed)'
    elif f['jun_rs'] <= 0:
        verdict = 'OUTFLOW sector'
    elif f['spike_5pct_pm'] < 2.0:
        verdict = 'Too few spikes'
    elif f['spike_5pct_pm'] >= 4.0:
        verdict = 'STRONG CANDIDATE'
    elif f['spike_5pct_pm'] >= 2.0:
        verdict = 'POSSIBLE candidate'
    else:
        verdict = '—'

    marker = '●' if f['is_crypto'] else '◆' if 'CANDIDATE' in verdict else '○'
    print(f"  {rank:>3} {marker}{f['stock']:<6} {f['sector']:<22} {f['jun_rs']:>+5.1f}% "
          f"{f['spike_5pct_pm']:>5.1f} {f['spike_10pct_pm']:>6.2f} "
          f"{f['avg_range']:>6.1%} {f['score']:>6.1f} {verdict:<20}")

non_crypto_candidates = [f for f in feasibility
                         if not f['is_crypto'] and f['jun_rs'] > 0 and f['spike_5pct_pm'] >= 2.0]
print(f"\n  Non-crypto stocks with ≥2 spike-5% days/month AND positive Jun RS: "
      f"{len(non_crypto_candidates)}")
if non_crypto_candidates:
    print(f"  → {', '.join(f['stock'] for f in non_crypto_candidates)}")


# ==============================================================================
# DIAGNOSIS
# ==============================================================================
print()
print("=" * W)
print(f"  DIAGNOSIS")
print("=" * W)

if crypto:
    crypto_5pm = crypto['spike_5pct_per_month']
    crypto_10pm = crypto['spike_10pct_per_month']
    print(f"\n  CRYPTO MINING BASELINE:")
    print(f"    Avg 5%+ spike days/month: {crypto_5pm:.1f}")
    print(f"    Avg 10%+ spike days/month: {crypto_10pm:.2f}")
    print(f"    Avg daily range: {crypto['avg_daily_range']:.1%}")
    print(f"    Annualized vol: {crypto['volatility_20d']:.0%}")

    print(f"\n  SECTOR COMPARISON (vs crypto baseline):")
    for sector in sorted_sectors:
        if sector == 'Crypto Mining':
            continue
        s = sector_summaries[sector]
        ratio_5 = s['spike_5pct_per_month'] / max(0.01, crypto_5pm)
        ratio_10 = s['spike_10pct_per_month'] / max(0.01, crypto_10pm)
        inflow = '↑' if s['jun_rs'] > 0 else '↓'
        print(f"    {inflow} {sector:<24} 5%/mo: {s['spike_5pct_per_month']:>4.1f} "
              f"({ratio_5:>4.0%} of crypto)   10%/mo: {s['spike_10pct_per_month']:>5.2f} "
              f"({ratio_10:>4.0%} of crypto)")

    print(f"\n  KEY FINDINGS:")
    high_spike_inflow = [s for s in sorted_sectors
                         if s != 'Crypto Mining'
                         and sector_summaries[s]['jun_rs'] > 0
                         and sector_summaries[s]['spike_5pct_per_month'] >= 2.0]
    low_spike_inflow = [s for s in sorted_sectors
                        if s != 'Crypto Mining'
                        and sector_summaries[s]['jun_rs'] > 0
                        and sector_summaries[s]['spike_5pct_per_month'] < 2.0]

    if high_spike_inflow:
        print(f"    Sectors with BOTH inflows AND ≥2 spike-5% days/month:")
        for s in high_spike_inflow:
            ss = sector_summaries[s]
            print(f"      ✓ {s} ({ss['etf']}, Jun RS {ss['jun_rs']:+.1f}%, "
                  f"{ss['spike_5pct_per_month']:.1f} spike days/mo)")
    else:
        print(f"    NO sectors have both inflows AND ≥2 spike-5% days/month")

    if low_spike_inflow:
        print(f"    Sectors with inflows but INSUFFICIENT spikes:")
        for s in low_spike_inflow:
            ss = sector_summaries[s]
            print(f"      ✗ {s} ({ss['etf']}, Jun RS {ss['jun_rs']:+.1f}%, "
                  f"only {ss['spike_5pct_per_month']:.1f} spike days/mo)")

    print(f"\n  IMPLICATION FOR MODEL EXPANSION:")
    if non_crypto_candidates:
        by_sector = defaultdict(list)
        for f in non_crypto_candidates:
            by_sector[f['sector']].append(f['stock'])
        for sector, stocks in sorted(by_sector.items(), key=lambda x: -len(x[1])):
            print(f"    {sector}: {', '.join(stocks)} — worth investigating")
    print(f"    Note: ETF-level spikes are muted by diversification.")
    print(f"    Individual stocks within inflow sectors may have much higher spike rates.")
    print(f"    This table shows individual stocks, not ETFs.")


# ==============================================================================
# CHARTS
# ==============================================================================
if HAS_MPL:
    print(f"\n  Generating charts...")

    SECTOR_COLORS = {
        'Crypto Mining': '#FF6B00',
        'Semiconductors': '#2196F3',
        'Defense / Aerospace': '#4CAF50',
        'Utilities / Nuclear': '#9C27B0',
        'Healthcare / Biotech': '#E91E63',
        'Financials / Fintech': '#00BCD4',
        'Industrials': '#795548',
    }

    # Chart 1: Sector average daily range
    fig, ax = plt.subplots(figsize=(12, 5))
    sectors_plot = [s for s in sorted_sectors if s in sector_summaries]
    ranges = [sector_summaries[s]['avg_daily_range'] * 100 for s in sectors_plot]
    colors = [SECTOR_COLORS.get(s, '#666') for s in sectors_plot]
    bars = ax.bar(range(len(sectors_plot)), ranges, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(sectors_plot)))
    ax.set_xticklabels([s.split('/')[0].strip()[:12] for s in sectors_plot],
                       rotation=30, ha='right', fontsize=10)
    for bar, val, sector in zip(bars, ranges, sectors_plot):
        rs = sector_summaries[sector]['jun_rs']
        label = f'{val:.1f}%\n(RS:{rs:+.0f}%)'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                label, ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Avg Daily Range (%)', fontsize=12)
    ax.set_title('Average Daily Range by Sector (2-year)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('/tmp/chart_sector_range.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Chart 2: Spike frequency by sector at each threshold
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, thresh in zip(axes, SPIKE_THRESHOLDS):
        pct = int(thresh * 100)
        key = f'spike_{pct}pct_per_month'
        vals = [sector_summaries[s].get(key, 0) for s in sectors_plot]
        colors = [SECTOR_COLORS.get(s, '#666') for s in sectors_plot]
        ax.barh(range(len(sectors_plot)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(sectors_plot)))
        ax.set_yticklabels([s.split('/')[0].strip()[:12] for s in sectors_plot], fontsize=9)
        ax.set_xlabel('Days/Month', fontsize=10)
        ax.set_title(f'≥{pct}% Upside Spikes/Month', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
    plt.suptitle('Spike Frequency by Sector and Threshold', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/tmp/chart_sector_spike_freq.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Chart 3: Scatter — 5%+ spike freq vs avg spike magnitude
    fig, ax = plt.subplots(figsize=(12, 8))
    for stock, m in stock_metrics.items():
        sector = STOCK_SECTOR.get(stock, '?')
        color = SECTOR_COLORS.get(sector, '#666')
        freq = m['spike_5pct_per_month']
        mag = m['spike_5pct_avg_mag'] * 100 if m['spike_5pct_avg_mag'] > 0 else 0
        size = 80 if 'Crypto' in sector else 50
        marker = 's' if 'Crypto' in sector else 'o'
        ax.scatter(freq, mag, c=color, s=size, marker=marker, alpha=0.7,
                   edgecolors='white', linewidth=0.5)
        if freq > 3.0 or mag > 8.0:
            ax.annotate(stock, (freq, mag), fontsize=8, alpha=0.8,
                        xytext=(5, 5), textcoords='offset points')

    for sector, color in SECTOR_COLORS.items():
        ax.scatter([], [], c=color, s=60, label=sector.split('/')[0].strip())
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.axvline(x=2.0, color='gray', linestyle='--', alpha=0.4)
    ax.text(2.1, ax.get_ylim()[1] * 0.95, 'min viable\n(2/mo)', fontsize=8,
            alpha=0.5, va='top')
    ax.set_xlabel('5%+ Spike Days per Month', fontsize=12)
    ax.set_ylabel('Avg Spike Magnitude on 5%+ Days (%)', fontsize=12)
    ax.set_title('Spike Frequency vs Magnitude — All Stocks', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/chart_spike_scatter.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Chart 4: Box plot of daily upside by sector
    fig, ax = plt.subplots(figsize=(12, 5))
    box_data = []
    box_labels = []
    box_colors = []
    for sector in sectors_plot:
        cfg = SECTOR_STOCKS[sector]
        sector_upside = []
        for stock in cfg['stocks']:
            if stock in data and data[stock] is not None:
                df = data[stock]
                df_v = df[df['Volume'] > 0]
                upside = ((df_v['High'] - df_v['Open']) / df_v['Open'] * 100).values
                sector_upside.extend(upside)
        if sector_upside:
            box_data.append(sector_upside)
            box_labels.append(sector.split('/')[0].strip()[:12])
            box_colors.append(SECTOR_COLORS.get(sector, '#666'))

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    showfliers=False, whis=[5, 95])
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.4, label='5% threshold')
    ax.axhline(y=10, color='darkred', linestyle='--', alpha=0.4, label='10% threshold')
    ax.set_ylabel('Daily Upside (High-Open)/Open (%)', fontsize=12)
    ax.set_title('Distribution of Daily Upside Moves by Sector', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('/tmp/chart_sector_upside_box.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"  Charts saved to /tmp/chart_sector_*.png")


print()
print("=" * W)
print(f"  Analysis complete. {len(stock_metrics)} stocks across {len(SECTOR_STOCKS)} sectors evaluated.")
print("=" * W)
