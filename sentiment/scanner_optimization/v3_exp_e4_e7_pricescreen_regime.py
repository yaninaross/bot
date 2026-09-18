"""
V3 Scanner Optimization — E4 (Pure Price Screen vs LLM) & E7 (Market Regime Filter)

E4: Compare a simple gap-fade price screen against LLM-based ticker selection.
    - Get full universe of >3% gappers from Polygon grouped daily API
    - Compute gap-fade returns for the full universe
    - Compare with LLM-flagged subset performance
    - Test ORB-enhanced variants

E7: Test whether SPY market regime affects scanner trade performance.
    - Classify each eval_date by SPY 5-day SMA regime
    - Analyze PnL by regime, direction, and SPY intraday trend
    - Compute alpha (excess return over market)

WARNING: Only 4 eval dates — regime analysis is VERY low-n.
"""

import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = "os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../') + '/sentiment/scanner_optimization"
RESULTS = os.path.join(BASE, "results")
CSV_PATH = os.path.join(RESULTS, "v2_features_full.csv")
DAILY_PATH = os.path.join(RESULTS, "v2_daily_data.json")
SPY_PATH = os.path.join(RESULTS, "v2_spy_daily.json")
MINUTE_PATH = os.path.join(RESULTS, "v3_minute_bars.json")

EVAL_DATES = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"]
PREV_DATES = {
    "2026-09-08": "2026-09-04",  # Sep 5-7 are Sat/Sun/Labor Day
    "2026-09-09": "2026-09-08",
    "2026-09-10": "2026-09-09",
    "2026-09-11": "2026-09-10",
}
POLYGON_KEY = os.environ["POLYGON_API_KEY"]
MIN_GAP_PCT = 3.0

# ── Helper functions ───────────────────────────────────────────────────────────

def trimmed_mean(arr, lo=5, hi=95):
    """5-95% trimmed mean for robustness."""
    arr = np.array([x for x in arr if not np.isnan(x)])
    if len(arr) == 0:
        return np.nan
    lo_val, hi_val = np.percentile(arr, [lo, hi])
    trimmed = arr[(arr >= lo_val) & (arr <= hi_val)]
    return np.mean(trimmed) if len(trimmed) > 0 else np.nan


def safe_median(arr):
    arr = np.array([x for x in arr if not np.isnan(x)])
    return np.median(arr) if len(arr) > 0 else np.nan


def win_rate(arr):
    arr = np.array([x for x in arr if not np.isnan(x)])
    if len(arr) == 0:
        return np.nan
    return np.mean(arr > 0) * 100


def fmt(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:+.{decimals}f}" if val != 0 else f"{val:.{decimals}f}"


# ── Load data ──────────────────────────────────────────────────────────────────
print("=" * 80)
print("V3 SCANNER OPTIMIZATION — E4 & E7")
print("=" * 80)

df = pd.read_csv(CSV_PATH)
with open(SPY_PATH) as f:
    spy_daily = json.load(f)
with open(MINUTE_PATH) as f:
    minute_bars = json.load(f)

print(f"Loaded {len(df)} LLM-flagged trades, {len(spy_daily)} SPY daily bars, "
      f"{len(minute_bars['ticker_bars'])} ticker minute bar sets")

# ═══════════════════════════════════════════════════════════════════════════════
# E4: PURE PRICE SCREEN VS LLM SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("E4: PURE PRICE SCREEN VS LLM SELECTION")
print("=" * 80)

# ── Step 1: Get full universe of gappers from Polygon ──────────────────────────
print("\n--- Step 1: Fetching full gapper universe from Polygon ---")

from polygon import RESTClient
poly = RESTClient(POLYGON_KEY)

# Fetch grouped daily bars for each date + previous date
grouped_data = {}  # date -> {ticker: {open, high, low, close, volume}}
dates_to_fetch = set()
for d in EVAL_DATES:
    dates_to_fetch.add(d)
    dates_to_fetch.add(PREV_DATES[d])

for date in sorted(dates_to_fetch):
    print(f"  Fetching grouped daily for {date}...", end=" ")
    try:
        results = poly.get_grouped_daily_aggs(date=date)
        day_data = {}
        for r in results:
            # Filter to common stocks — skip warrants, units, rights
            ticker = r.ticker
            if any(ticker.endswith(suf) for suf in ['W', 'WS', 'U', 'R']):
                continue
            day_data[ticker] = {
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume,
            }
        grouped_data[date] = day_data
        print(f"{len(day_data)} tickers")
    except Exception as e:
        print(f"ERROR: {e}")
        grouped_data[date] = {}
    time.sleep(0.25)  # Rate limit courtesy

# ── Step 2: Compute gaps and build gapper universe ─────────────────────────────
print("\n--- Step 2: Computing gaps for full universe ---")

all_gappers = []  # list of dicts
for eval_date in EVAL_DATES:
    prev_date = PREV_DATES[eval_date]
    today = grouped_data.get(eval_date, {})
    yesterday = grouped_data.get(prev_date, {})

    common_tickers = set(today.keys()) & set(yesterday.keys())
    date_gappers = 0
    for ticker in common_tickers:
        prev_close = yesterday[ticker]['close']
        today_open = today[ticker]['open']
        today_close = today[ticker]['close']
        today_high = today[ticker]['high']
        today_low = today[ticker]['low']
        today_vol = today[ticker]['volume']

        if prev_close <= 0 or today_open <= 0:
            continue

        gap_pct = (today_open - prev_close) / prev_close * 100
        abs_gap = abs(gap_pct)

        if abs_gap < MIN_GAP_PCT:
            continue

        # Gap-fade return: short the gap direction
        # If gap up, we short -> profit if close < open
        # If gap down, we go long -> profit if close > open
        # Unified: gap_fade_pnl = -sign(gap) * (close - open) / open * 100
        gap_sign = 1 if gap_pct > 0 else -1
        gap_fade_pnl = -gap_sign * (today_close - today_open) / today_open * 100

        # Intraday range for context
        intraday_range = (today_high - today_low) / today_open * 100 if today_open > 0 else 0

        all_gappers.append({
            'ticker': ticker,
            'eval_date': eval_date,
            'gap_pct': gap_pct,
            'abs_gap_pct': abs_gap,
            'gap_direction': 'UP' if gap_pct > 0 else 'DOWN',
            'today_open': today_open,
            'today_close': today_close,
            'today_volume': today_vol,
            'gap_fade_pnl': gap_fade_pnl,
            'intraday_range_pct': intraday_range,
            'prev_close': prev_close,
        })
        date_gappers += 1

    print(f"  {eval_date}: {len(common_tickers)} common tickers -> {date_gappers} gappers (>{MIN_GAP_PCT}% gap)")

gappers_df = pd.DataFrame(all_gappers)
print(f"\nTotal gappers: {len(gappers_df)}")

# Filter to reasonable price / volume (remove penny stocks and ultra-low vol)
gappers_df_filtered = gappers_df[
    (gappers_df['today_open'] >= 1.0) &  # Price >= $1
    (gappers_df['today_volume'] >= 10000)  # Min 10k volume
].copy()
print(f"After price >= $1 and vol >= 10k: {len(gappers_df_filtered)}")

# ── Step 3: Compare universe vs LLM-flagged ────────────────────────────────────
print("\n--- Step 3: Universe vs LLM-flagged comparison ---")

# Identify which LLM-flagged tickers are in the gapper universe
df['in_universe'] = df.apply(
    lambda row: row['ticker'] in set(
        gappers_df_filtered[gappers_df_filtered['eval_date'] == row['eval_date']]['ticker']
    ),
    axis=1
)

# Summary table
print(f"\n{'Category':<35} {'N':>6} {'Mean PnL':>10} {'Median':>10} {'Trim Mean':>10} {'WinRate':>8}")
print("-" * 85)

# Full universe (all gappers)
u_pnl = gappers_df_filtered['gap_fade_pnl'].dropna().values
print(f"{'Full gapper universe (gap-fade)':<35} {len(u_pnl):>6} "
      f"{np.mean(u_pnl):>+10.3f} {np.median(u_pnl):>+10.3f} "
      f"{trimmed_mean(u_pnl):>+10.3f} {win_rate(u_pnl):>7.1f}%")

# Gap-up only (SHORT bias universe)
gap_up = gappers_df_filtered[gappers_df_filtered['gap_direction'] == 'UP']
up_pnl = gap_up['gap_fade_pnl'].dropna().values
print(f"{'  Gap-up only (fade = SHORT)':<35} {len(up_pnl):>6} "
      f"{np.mean(up_pnl):>+10.3f} {np.median(up_pnl):>+10.3f} "
      f"{trimmed_mean(up_pnl):>+10.3f} {win_rate(up_pnl):>7.1f}%")

# Gap-down only (LONG bias universe)
gap_dn = gappers_df_filtered[gappers_df_filtered['gap_direction'] == 'DOWN']
dn_pnl = gap_dn['gap_fade_pnl'].dropna().values
print(f"{'  Gap-down only (fade = LONG)':<35} {len(dn_pnl):>6} "
      f"{np.mean(dn_pnl):>+10.3f} {np.median(dn_pnl):>+10.3f} "
      f"{trimmed_mean(dn_pnl):>+10.3f} {win_rate(dn_pnl):>7.1f}%")

# LLM-flagged trades — 60m PnL
llm_pnl = df['pnl_60m_nosl'].dropna().values
print(f"{'LLM-flagged (60m no-SL)':<35} {len(llm_pnl):>6} "
      f"{np.mean(llm_pnl):>+10.3f} {np.median(llm_pnl):>+10.3f} "
      f"{trimmed_mean(llm_pnl):>+10.3f} {win_rate(llm_pnl):>7.1f}%")

# LLM-flagged EOD PnL
llm_eod = df['pnl_eod_nosl'].dropna().values
print(f"{'LLM-flagged (EOD no-SL)':<35} {len(llm_eod):>6} "
      f"{np.mean(llm_eod):>+10.3f} {np.median(llm_eod):>+10.3f} "
      f"{trimmed_mean(llm_eod):>+10.3f} {win_rate(llm_eod):>7.1f}%")

# LLM non-earnings
llm_noearn = df[df['is_earnings'] == False]
ne_pnl = llm_noearn['pnl_60m_nosl'].dropna().values
print(f"{'LLM-flagged excl earnings (60m)':<35} {len(ne_pnl):>6} "
      f"{np.mean(ne_pnl):>+10.3f} {np.median(ne_pnl):>+10.3f} "
      f"{trimmed_mean(ne_pnl):>+10.3f} {win_rate(ne_pnl):>7.1f}%")

# ── Step 4: Per-date comparison ────────────────────────────────────────────────
print("\n--- Step 4: Per-date comparison ---")
print(f"\n{'Date':<14} {'Univ N':>7} {'Univ PnL':>10} {'LLM N':>6} {'LLM PnL':>10} {'LLM adds?':>10}")
print("-" * 65)

date_comparison = {}
for d in EVAL_DATES:
    u_day = gappers_df_filtered[gappers_df_filtered['eval_date'] == d]
    l_day = df[df['eval_date'] == d]

    u_mean = np.mean(u_day['gap_fade_pnl'].dropna()) if len(u_day) > 0 else np.nan
    l_mean = np.mean(l_day['pnl_60m_nosl'].dropna()) if len(l_day['pnl_60m_nosl'].dropna()) > 0 else np.nan

    adds_value = "YES" if (not np.isnan(l_mean) and not np.isnan(u_mean) and l_mean > u_mean) else "NO"
    if np.isnan(l_mean) or np.isnan(u_mean):
        adds_value = "N/A"

    date_comparison[d] = {
        'univ_n': len(u_day),
        'univ_pnl': u_mean,
        'llm_n': len(l_day),
        'llm_pnl': l_mean,
    }
    print(f"{d:<14} {len(u_day):>7} {fmt(u_mean):>10} {len(l_day):>6} {fmt(l_mean):>10} {adds_value:>10}")

# ── Step 5: ORB-enhanced price screen ──────────────────────────────────────────
print("\n--- Step 5: ORB-enhanced comparison (LLM-flagged subset) ---")

# For LLM-flagged trades, we have ORB direction
# ORB confirms gap-fade: if gap is UP and ORB is SHORT, or gap is DOWN and ORB is LONG
df['gap_fade_dir'] = df['gap_pct'].apply(lambda x: 'SHORT' if x > 0 else 'LONG')
df['orb_confirms_fade'] = df['orb_dir'] == df['gap_fade_dir']

# Group 1: All LLM-flagged
# Group 2: LLM + ORB confirms
# Group 3: Price screen concept (just gap > 3% from our CSV, ORB confirms)

# For ORB analysis, use the original direction-specific PnL
# The pnl_60m_nosl is already direction-adjusted (positive = trade worked)

print(f"\n{'Filter':<40} {'N':>5} {'Mean':>8} {'Median':>8} {'TrMean':>8} {'WR%':>6}")
print("-" * 80)

# All LLM trades
all_60 = df['pnl_60m_nosl'].dropna()
print(f"{'All LLM trades':<40} {len(all_60):>5} {np.mean(all_60):>+8.3f} "
      f"{np.median(all_60):>+8.3f} {trimmed_mean(all_60.values):>+8.3f} {win_rate(all_60.values):>5.1f}%")

# LLM + ORB confirms LLM direction
orb_match = df[df['orb_dir'] == df['direction']]
orb_pnl = orb_match['pnl_60m_nosl'].dropna()
print(f"{'LLM + ORB confirms LLM direction':<40} {len(orb_pnl):>5} {np.mean(orb_pnl):>+8.3f} "
      f"{np.median(orb_pnl):>+8.3f} {trimmed_mean(orb_pnl.values):>+8.3f} {win_rate(orb_pnl.values):>5.1f}%")

# LLM + ORB confirms gap-fade
orb_fade = df[df['orb_confirms_fade'] == True]
fade_pnl = orb_fade['pnl_60m_nosl'].dropna()
print(f"{'LLM + ORB confirms gap-fade':<40} {len(fade_pnl):>5} {np.mean(fade_pnl):>+8.3f} "
      f"{np.median(fade_pnl):>+8.3f} {trimmed_mean(fade_pnl.values):>+8.3f} {win_rate(fade_pnl.values):>5.1f}%")

# Non-earnings + ORB confirms LLM dir
ne_orb = df[(df['is_earnings'] == False) & (df['orb_dir'] == df['direction'])]
ne_orb_pnl = ne_orb['pnl_60m_nosl'].dropna()
print(f"{'Non-earn + ORB confirms LLM dir':<40} {len(ne_orb_pnl):>5} {np.mean(ne_orb_pnl):>+8.3f} "
      f"{np.median(ne_orb_pnl):>+8.3f} {trimmed_mean(ne_orb_pnl.values):>+8.3f} {win_rate(ne_orb_pnl.values):>5.1f}%")

# Price screen concept: abs_gap > 3%, not earnings, ORB confirms gap-fade
ps_orb = df[(df['abs_gap_pct'] >= 3.0) & (df['is_earnings'] == False) & (df['orb_confirms_fade'] == True)]
ps_orb_pnl = ps_orb['pnl_60m_nosl'].dropna()
print(f"{'PriceScreen(>3%) + no-earn + ORB fade':<40} {len(ps_orb_pnl):>5} {np.mean(ps_orb_pnl):>+8.3f} "
      f"{np.median(ps_orb_pnl):>+8.3f} {trimmed_mean(ps_orb_pnl.values):>+8.3f} {win_rate(ps_orb_pnl.values):>5.1f}%")

# Gap magnitude buckets
print("\n--- Gap magnitude buckets (all gappers, universe) ---")
print(f"{'Gap Range':<20} {'N':>6} {'Mean PnL':>10} {'Median':>10} {'WR%':>8}")
print("-" * 60)
for lo, hi in [(3, 5), (5, 10), (10, 20), (20, 50), (50, 200)]:
    bucket = gappers_df_filtered[(gappers_df_filtered['abs_gap_pct'] >= lo) & (gappers_df_filtered['abs_gap_pct'] < hi)]
    if len(bucket) == 0:
        continue
    bp = bucket['gap_fade_pnl'].dropna().values
    print(f"{lo}-{hi}%{'':<15} {len(bp):>6} {np.mean(bp):>+10.3f} {np.median(bp):>+10.3f} {win_rate(bp):>7.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# E7: MARKET REGIME FILTER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("E7: MARKET REGIME FILTER")
print("=" * 80)
print("\nWARNING: Only 4 eval dates — regime analysis is VERY low-n. "
      "Treat as directional only.\n")

# ── Regime classification ──────────────────────────────────────────────────────
print("--- Regime classification ---")

spy_dates = sorted(spy_daily.keys())

def get_spy_sma5(eval_date):
    """Compute SPY 5-day SMA from the 5 trading days BEFORE eval_date."""
    idx = spy_dates.index(eval_date) if eval_date in spy_dates else -1
    if idx < 5:
        return np.nan
    prev_5 = spy_dates[idx-5:idx]
    closes = [spy_daily[d]['close'] for d in prev_5]
    return np.mean(closes)

regime_info = {}
for d in EVAL_DATES:
    prev_d = PREV_DATES[d]
    sma5 = get_spy_sma5(d)
    prev_close = spy_daily[prev_d]['close'] if prev_d in spy_daily else np.nan

    if np.isnan(sma5) or np.isnan(prev_close):
        regime = "unknown"
        distance_pct = np.nan
    else:
        distance_pct = (prev_close - sma5) / sma5 * 100
        if abs(distance_pct) < 0.1:
            regime = "range_bound"
        elif distance_pct > 0:
            regime = "trending_up"
        else:
            regime = "trending_down"

    # SPY intraday return: use daily bar close vs minute bar open at 09:30
    # (minute bars only extend to ~13:00, so use daily close from v2_spy_daily.json)
    spy_mb = minute_bars['spy_bars'].get(d, {})
    spy_open_930 = spy_mb.get('09:30', {}).get('o', np.nan)
    spy_daily_close = spy_daily[d]['close'] if d in spy_daily else np.nan
    spy_intra_ret = (spy_daily_close - spy_open_930) / spy_open_930 * 100 if (
        not np.isnan(spy_open_930) and spy_open_930 > 0 and not np.isnan(spy_daily_close)
    ) else np.nan

    # SPY morning momentum (first 30 min)
    spy_1000 = spy_mb.get('09:59', spy_mb.get('10:00', {}))
    spy_morning_close = spy_1000.get('c', np.nan) if spy_1000 else np.nan
    spy_morning_ret = (spy_morning_close - spy_open_930) / spy_open_930 * 100 if (
        not np.isnan(spy_open_930) and spy_open_930 > 0 and not np.isnan(spy_morning_close)
    ) else np.nan

    # SPY 60m return (for alpha calc)
    spy_1030 = spy_mb.get('10:29', spy_mb.get('10:30', {}))
    spy_60m_close = spy_1030.get('c', np.nan) if spy_1030 else np.nan
    spy_60m_ret = (spy_60m_close - spy_open_930) / spy_open_930 * 100 if (
        not np.isnan(spy_open_930) and spy_open_930 > 0 and not np.isnan(spy_60m_close)
    ) else np.nan

    regime_info[d] = {
        'regime': regime,
        'sma5': sma5,
        'prev_close': prev_close,
        'distance_pct': distance_pct,
        'spy_intra_ret': spy_intra_ret,
        'spy_morning_ret': spy_morning_ret,
        'spy_60m_ret': spy_60m_ret,
        'spy_open': spy_open_930,
    }

    print(f"  {d}: regime={regime:<15} SPY prev_close={prev_close:.2f}  SMA5={sma5:.2f}  "
          f"dist={fmt(distance_pct)}%  intra_ret={fmt(spy_intra_ret)}%  "
          f"morn_ret={fmt(spy_morning_ret)}%  60m_ret={fmt(spy_60m_ret)}%")

# ── Per-regime trade performance ───────────────────────────────────────────────
print("\n--- Per-regime trade performance ---")

df['regime'] = df['eval_date'].map(lambda d: regime_info[d]['regime'])
df['spy_intra_ret'] = df['eval_date'].map(lambda d: regime_info[d]['spy_intra_ret'])
df['spy_morning_ret'] = df['eval_date'].map(lambda d: regime_info[d]['spy_morning_ret'])
df['spy_60m_ret'] = df['eval_date'].map(lambda d: regime_info[d]['spy_60m_ret'])

print(f"\n{'Regime':<18} {'N':>5} {'60m Mean':>10} {'60m Med':>10} {'60m TrM':>10} {'EOD Mean':>10} {'WR%':>7}")
print("-" * 75)
for regime in sorted(df['regime'].unique()):
    rdf = df[df['regime'] == regime]
    p60 = rdf['pnl_60m_nosl'].dropna().values
    peod = rdf['pnl_eod_nosl'].dropna().values
    print(f"{regime:<18} {len(rdf):>5} {np.mean(p60):>+10.3f} {np.median(p60):>+10.3f} "
          f"{trimmed_mean(p60):>+10.3f} {np.mean(peod):>+10.3f} {win_rate(p60):>6.1f}%")

# ── LONG vs SHORT by regime ───────────────────────────────────────────────────
print("\n--- LONG vs SHORT by regime ---")
print(f"\n{'Regime':<18} {'Dir':<7} {'N':>5} {'60m Mean':>10} {'60m Med':>10} {'WR%':>7}")
print("-" * 60)
for regime in sorted(df['regime'].unique()):
    for direction in ['LONG', 'SHORT']:
        rdf = df[(df['regime'] == regime) & (df['direction'] == direction)]
        p60 = rdf['pnl_60m_nosl'].dropna().values
        if len(p60) == 0:
            continue
        print(f"{regime:<18} {direction:<7} {len(rdf):>5} {np.mean(p60):>+10.3f} "
              f"{np.median(p60):>+10.3f} {win_rate(p60):>6.1f}%")

# ── Per-date SPY vs trade PnL ─────────────────────────────────────────────────
print("\n--- Per-date: SPY return vs avg trade PnL ---")
print(f"\n{'Date':<14} {'Regime':<15} {'SPY Intra':>10} {'SPY Morn':>10} {'Avg 60m':>10} "
      f"{'LONG 60m':>10} {'SHORT 60m':>10} {'N':>5}")
print("-" * 85)

date_stats = {}
for d in EVAL_DATES:
    ri = regime_info[d]
    ddf = df[df['eval_date'] == d]
    avg_60 = np.mean(ddf['pnl_60m_nosl'].dropna()) if len(ddf['pnl_60m_nosl'].dropna()) > 0 else np.nan
    long_60 = np.mean(ddf[ddf['direction'] == 'LONG']['pnl_60m_nosl'].dropna())
    short_60 = np.mean(ddf[ddf['direction'] == 'SHORT']['pnl_60m_nosl'].dropna())

    date_stats[d] = {
        'spy_intra': ri['spy_intra_ret'],
        'spy_morn': ri['spy_morning_ret'],
        'avg_60m': avg_60,
        'long_60m': long_60 if not np.isnan(long_60) else np.nan,
        'short_60m': short_60 if not np.isnan(short_60) else np.nan,
        'n': len(ddf),
    }

    print(f"{d:<14} {ri['regime']:<15} {fmt(ri['spy_intra_ret']):>10} "
          f"{fmt(ri['spy_morning_ret']):>10} {fmt(avg_60):>10} "
          f"{fmt(long_60):>10} {fmt(short_60):>10} {len(ddf):>5}")

# ── SPY correlation (very low-n) ──────────────────────────────────────────────
print("\n--- SPY intraday correlation (n=4, directional only) ---")
spy_rets = [date_stats[d]['spy_intra'] for d in EVAL_DATES]
avg_pnls = [date_stats[d]['avg_60m'] for d in EVAL_DATES]
valid = [(s, p) for s, p in zip(spy_rets, avg_pnls) if not np.isnan(s) and not np.isnan(p)]
corr = np.nan
pval = np.nan
if len(valid) >= 3:
    spy_arr, pnl_arr = zip(*valid)
    corr, pval = stats.pearsonr(spy_arr, pnl_arr)
    print(f"  Pearson r(SPY_intra, avg_trade_60m) = {corr:+.3f}  p={pval:.3f}  n={len(valid)}")
else:
    print("  Insufficient data for correlation")

spy_morns = [date_stats[d]['spy_morn'] for d in EVAL_DATES]
valid_m = [(s, p) for s, p in zip(spy_morns, avg_pnls) if not np.isnan(s) and not np.isnan(p)]
corr_m = np.nan
pval_m = np.nan
if len(valid_m) >= 3:
    spy_m_arr, pnl_m_arr = zip(*valid_m)
    corr_m, pval_m = stats.pearsonr(spy_m_arr, pnl_m_arr)
    print(f"  Pearson r(SPY_morning, avg_trade_60m) = {corr_m:+.3f}  p={pval_m:.3f}  n={len(valid_m)}")

# ── Alpha analysis ────────────────────────────────────────────────────────────
print("\n--- Alpha analysis (excess return over market) ---")

# Compute alpha at 60m: trade_pnl - sign * SPY_60m_return
df['alpha_60m'] = df['pnl_60m_nosl'] - df['sign'] * df['spy_60m_ret']
alpha_60 = df['alpha_60m'].dropna().values
print(f"  Alpha at 60m:  mean={np.mean(alpha_60):+.3f}  median={np.median(alpha_60):+.3f}  "
      f"trim_mean={trimmed_mean(alpha_60):+.3f}  n={len(alpha_60)}")
print(f"  Alpha > 0: {np.mean(alpha_60 > 0)*100:.1f}% of trades")

# Use existing alpha columns
alpha_t1 = df['alpha_t1_close'].dropna().values
alpha_t2 = df['alpha_t2_close'].dropna().values
print(f"  Alpha T+1 close: mean={np.mean(alpha_t1):+.3f}  median={np.median(alpha_t1):+.3f}  n={len(alpha_t1)}")
print(f"  Alpha T+2 close: mean={np.mean(alpha_t2):+.3f}  median={np.median(alpha_t2):+.3f}  n={len(alpha_t2)}")

# Alpha by direction
for direction in ['LONG', 'SHORT']:
    ddf = df[df['direction'] == direction]
    a60 = ddf['alpha_60m'].dropna().values
    at1 = ddf['alpha_t1_close'].dropna().values
    if len(a60) > 0:
        print(f"  {direction} alpha 60m: mean={np.mean(a60):+.3f}  median={np.median(a60):+.3f}  "
              f"positive={np.mean(a60>0)*100:.1f}%  n={len(a60)}")

# Alpha by regime
print("\n  Alpha by regime:")
for regime in sorted(df['regime'].unique()):
    rdf = df[df['regime'] == regime]
    a60 = rdf['alpha_60m'].dropna().values
    if len(a60) > 0:
        print(f"    {regime:<18} alpha_60m mean={np.mean(a60):+.3f}  median={np.median(a60):+.3f}  n={len(a60)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── E4 Charts ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("E4: Pure Price Screen vs LLM Selection", fontsize=14, fontweight='bold')

# Chart 1: Full universe vs LLM-flagged avg PnL
ax = axes[0, 0]
categories = ['Full Universe\n(gap-fade EOD)', 'Gap-Up\n(fade=SHORT)', 'Gap-Down\n(fade=LONG)',
              'LLM-flagged\n(60m)', 'LLM no-earn\n(60m)']
means = [np.mean(u_pnl), np.mean(up_pnl), np.mean(dn_pnl), np.mean(llm_pnl), np.mean(ne_pnl)]
medians_vals = [np.median(u_pnl), np.median(up_pnl), np.median(dn_pnl), np.median(llm_pnl), np.median(ne_pnl)]
counts = [len(u_pnl), len(up_pnl), len(dn_pnl), len(llm_pnl), len(ne_pnl)]
colors = ['#4a86c8', '#c85a4a', '#5aaa5a', '#c8a04a', '#7a5ac8']

x = np.arange(len(categories))
width = 0.35
bars1 = ax.bar(x - width/2, means, width, label='Mean', color=colors, alpha=0.8)
bars2 = ax.bar(x + width/2, medians_vals, width, label='Median', color=colors, alpha=0.4)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax.set_ylabel('PnL (%)')
ax.set_title('Avg PnL: Universe vs LLM')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8)
ax.legend(fontsize=8)
for i, (m, n) in enumerate(zip(means, counts)):
    ax.text(i - width/2, m + (0.05 if m >= 0 else -0.15), f'n={n}', ha='center', fontsize=7)

# Chart 2: Per-date universe size and PnL
ax = axes[0, 1]
x = np.arange(len(EVAL_DATES))
width = 0.35
univ_pnls = [date_comparison[d]['univ_pnl'] for d in EVAL_DATES]
llm_pnls = [date_comparison[d]['llm_pnl'] for d in EVAL_DATES]
ax.bar(x - width/2, univ_pnls, width, label='Universe (gap-fade)', color='#4a86c8', alpha=0.8)
ax.bar(x + width/2, llm_pnls, width, label='LLM (60m)', color='#c8a04a', alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax.set_ylabel('Avg PnL (%)')
ax.set_title('Per-Date: Universe vs LLM PnL')
ax.set_xticks(x)
date_labels = [d.split('-')[-1] for d in EVAL_DATES]
ax.set_xticklabels([f"Sep {d}" for d in date_labels])
ax.legend(fontsize=8)
# Add n labels
for i, d in enumerate(EVAL_DATES):
    ax.text(i - width/2, 0.02, f"n={date_comparison[d]['univ_n']}", ha='center', fontsize=7, rotation=90, va='bottom')
    ax.text(i + width/2, 0.02, f"n={date_comparison[d]['llm_n']}", ha='center', fontsize=7, rotation=90, va='bottom')

# Chart 3: ORB-enhanced comparison
ax = axes[1, 0]
orb_labels = ['All LLM', 'LLM+ORB\n(confirms dir)', 'LLM+ORB\n(fade)', 'No-earn+ORB\n(confirms dir)',
              'PriceScr+\nno-earn+ORB']
orb_means_list = []
orb_ns = []
orb_wrs = []
for label, subset_pnl in [
    ('All LLM', all_60.values),
    ('LLM+ORB dir', orb_pnl.values),
    ('LLM+ORB fade', fade_pnl.values),
    ('NE+ORB dir', ne_orb_pnl.values),
    ('PS+NE+ORB', ps_orb_pnl.values),
]:
    orb_means_list.append(np.mean(subset_pnl) if len(subset_pnl) > 0 else 0)
    orb_ns.append(len(subset_pnl))
    orb_wrs.append(win_rate(subset_pnl))

orb_colors = ['#888888', '#4a86c8', '#c85a4a', '#5aaa5a', '#c8a04a']
bars = ax.bar(range(len(orb_labels)), orb_means_list, color=orb_colors, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax.set_ylabel('Avg PnL 60m (%)')
ax.set_title('ORB-Enhanced Filters')
ax.set_xticks(range(len(orb_labels)))
ax.set_xticklabels(orb_labels, fontsize=8)
for i, (m, n, w) in enumerate(zip(orb_means_list, orb_ns, orb_wrs)):
    offset = 0.05 if m >= 0 else -0.2
    ax.text(i, m + offset, f'n={n}\nWR={w:.0f}%', ha='center', fontsize=7)

# Chart 4: Gap magnitude bucket performance (universe)
ax = axes[1, 1]
buckets = [(3, 5), (5, 10), (10, 20), (20, 50), (50, 200)]
bucket_labels = []
bucket_means = []
bucket_ns = []
for lo, hi in buckets:
    bucket = gappers_df_filtered[(gappers_df_filtered['abs_gap_pct'] >= lo) & (gappers_df_filtered['abs_gap_pct'] < hi)]
    if len(bucket) == 0:
        continue
    bp = bucket['gap_fade_pnl'].dropna().values
    bucket_labels.append(f"{lo}-{hi}%")
    bucket_means.append(np.mean(bp))
    bucket_ns.append(len(bp))

bcolors = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(bucket_labels)))
bars = ax.bar(range(len(bucket_labels)), bucket_means, color=bcolors, alpha=0.8, edgecolor='#333')
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax.set_ylabel('Avg Gap-Fade PnL (%)')
ax.set_title('Universe Gap-Fade by Gap Magnitude')
ax.set_xticks(range(len(bucket_labels)))
ax.set_xticklabels(bucket_labels, fontsize=9)
for i, (m, n) in enumerate(zip(bucket_means, bucket_ns)):
    offset = 0.05 if m >= 0 else -0.25
    ax.text(i, m + offset, f'n={n}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(BASE, "v3_e4_pricescreen.png"), dpi=150, bbox_inches='tight')
print(f"\nSaved E4 chart: {os.path.join(BASE, 'v3_e4_pricescreen.png')}")

# ── E7 Charts ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("E7: Market Regime Filter (n=4 dates — VERY low-n)", fontsize=14, fontweight='bold')

# Chart 1: Regime -> avg PnL
ax = axes[0, 0]
regimes = sorted(df['regime'].unique())
regime_means_60 = []
regime_means_eod = []
regime_ns_plot = []
for regime in regimes:
    rdf = df[df['regime'] == regime]
    p60 = rdf['pnl_60m_nosl'].dropna().values
    peod = rdf['pnl_eod_nosl'].dropna().values
    regime_means_60.append(np.mean(p60) if len(p60) > 0 else 0)
    regime_means_eod.append(np.mean(peod) if len(peod) > 0 else 0)
    regime_ns_plot.append(len(rdf))

x = np.arange(len(regimes))
width = 0.35
ax.bar(x - width/2, regime_means_60, width, label='60m', color='#4a86c8', alpha=0.8)
ax.bar(x + width/2, regime_means_eod, width, label='EOD', color='#c85a4a', alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax.set_ylabel('Avg PnL (%)')
ax.set_title('PnL by Market Regime')
ax.set_xticks(x)
ax.set_xticklabels(regimes, fontsize=9)
ax.legend(fontsize=8)
for i, n in enumerate(regime_ns_plot):
    ax.text(i, max(regime_means_60[i], regime_means_eod[i]) + 0.05, f'n={n}', ha='center', fontsize=8)

# Chart 2: SPY return vs trade PnL scatter (per-day, labeled)
ax = axes[0, 1]
spy_x = [regime_info[d]['spy_intra_ret'] for d in EVAL_DATES]
trade_y = [date_stats[d]['avg_60m'] for d in EVAL_DATES]
valid_mask = [not (np.isnan(s) or np.isnan(t)) for s, t in zip(spy_x, trade_y)]
spy_x_v = [s for s, v in zip(spy_x, valid_mask) if v]
trade_y_v = [t for t, v in zip(trade_y, valid_mask) if v]
dates_v = [d for d, v in zip(EVAL_DATES, valid_mask) if v]
ax.scatter(spy_x_v, trade_y_v, s=100, c='#4a86c8', zorder=3, edgecolors='black')
for s, t, d in zip(spy_x_v, trade_y_v, dates_v):
    ax.annotate(d.split('-')[-1] + ' Sep', (s, t), textcoords="offset points", xytext=(8, 8), fontsize=8)
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('SPY Intraday Return (%)')
ax.set_ylabel('Avg Trade PnL 60m (%)')
ax.set_title('SPY Return vs Trade PnL (per-day)')
if len(spy_x_v) >= 3:
    slope, intercept = np.polyfit(spy_x_v, trade_y_v, 1)
    x_line = np.linspace(min(spy_x_v) - 0.1, max(spy_x_v) + 0.1, 50)
    ax.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.5, linewidth=1)
    r_scatter, _ = stats.pearsonr(spy_x_v, trade_y_v) if len(spy_x_v) >= 3 else (np.nan, np.nan)
    ax.text(0.05, 0.95, f'r={r_scatter:+.2f}\nn={len(spy_x_v)}', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Chart 3: Alpha distribution histogram
ax = axes[1, 0]
alpha_data = df['alpha_t1_close'].dropna().values
# Clip extremes for visibility
alpha_clipped = np.clip(alpha_data, -20, 20)
ax.hist(alpha_clipped, bins=40, color='#5aaa5a', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axvline(x=0, color='red', linewidth=1.5, linestyle='-')
ax.axvline(x=np.mean(alpha_data), color='blue', linewidth=1.5, linestyle='--', label=f'Mean={np.mean(alpha_data):+.2f}')
ax.axvline(x=np.median(alpha_data), color='orange', linewidth=1.5, linestyle='--', label=f'Median={np.median(alpha_data):+.2f}')
ax.set_xlabel('Alpha T+1 Close (%)')
ax.set_ylabel('Count')
ax.set_title('Alpha Distribution (T+1 Close)')
ax.legend(fontsize=8)
ax.text(0.95, 0.95, f'n={len(alpha_data)}\n>0: {np.mean(alpha_data>0)*100:.0f}%',
        transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Chart 4: LONG vs SHORT PnL by regime
ax = axes[1, 1]
regimes_for_chart = sorted(df['regime'].unique())
x = np.arange(len(regimes_for_chart))
width = 0.35
long_means = []
short_means = []
long_ns = []
short_ns = []
for regime in regimes_for_chart:
    ldf = df[(df['regime'] == regime) & (df['direction'] == 'LONG')]
    sdf = df[(df['regime'] == regime) & (df['direction'] == 'SHORT')]
    lp = ldf['pnl_60m_nosl'].dropna().values
    sp = sdf['pnl_60m_nosl'].dropna().values
    long_means.append(np.mean(lp) if len(lp) > 0 else 0)
    short_means.append(np.mean(sp) if len(sp) > 0 else 0)
    long_ns.append(len(lp))
    short_ns.append(len(sp))

ax.bar(x - width/2, long_means, width, label='LONG', color='#4a86c8', alpha=0.8)
ax.bar(x + width/2, short_means, width, label='SHORT', color='#c85a4a', alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax.set_ylabel('Avg PnL 60m (%)')
ax.set_title('LONG vs SHORT by Regime')
ax.set_xticks(x)
ax.set_xticklabels(regimes_for_chart, fontsize=9)
ax.legend(fontsize=8)
for i in range(len(regimes_for_chart)):
    ax.text(i - width/2, long_means[i] + 0.03, f'n={long_ns[i]}', ha='center', fontsize=7)
    ax.text(i + width/2, short_means[i] + 0.03, f'n={short_ns[i]}', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(BASE, "v3_e7_regime.png"), dpi=150, bbox_inches='tight')
print(f"Saved E7 chart: {os.path.join(BASE, 'v3_e7_regime.png')}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS JSON
# ═══════════════════════════════════════════════════════════════════════════════

results_json = {
    "experiment": "V3 E4 + E7",
    "eval_dates": EVAL_DATES,
    "e4_price_screen": {
        "full_universe": {
            "n_gappers": len(gappers_df_filtered),
            "gap_fade_mean_pnl": float(np.mean(u_pnl)),
            "gap_fade_median_pnl": float(np.median(u_pnl)),
            "gap_fade_trimmed_mean": float(trimmed_mean(u_pnl)),
            "win_rate": float(win_rate(u_pnl)),
        },
        "gap_up_subset": {
            "n": len(up_pnl),
            "fade_mean_pnl": float(np.mean(up_pnl)),
            "fade_median_pnl": float(np.median(up_pnl)),
        },
        "gap_down_subset": {
            "n": len(dn_pnl),
            "fade_mean_pnl": float(np.mean(dn_pnl)),
            "fade_median_pnl": float(np.median(dn_pnl)),
        },
        "llm_flagged": {
            "n": len(llm_pnl),
            "mean_pnl_60m": float(np.mean(llm_pnl)),
            "median_pnl_60m": float(np.median(llm_pnl)),
            "trimmed_mean_60m": float(trimmed_mean(llm_pnl)),
            "win_rate": float(win_rate(llm_pnl)),
        },
        "llm_non_earnings": {
            "n": len(ne_pnl),
            "mean_pnl_60m": float(np.mean(ne_pnl)),
        },
        "per_date": {d: {
            "universe_n": date_comparison[d]['univ_n'],
            "universe_pnl": float(date_comparison[d]['univ_pnl']) if not np.isnan(date_comparison[d]['univ_pnl']) else None,
            "llm_n": date_comparison[d]['llm_n'],
            "llm_pnl": float(date_comparison[d]['llm_pnl']) if not np.isnan(date_comparison[d]['llm_pnl']) else None,
        } for d in EVAL_DATES},
        "orb_enhanced": {
            "all_llm": {"n": len(all_60), "mean": float(np.mean(all_60))},
            "llm_orb_confirms_dir": {"n": len(orb_pnl), "mean": float(np.mean(orb_pnl))},
            "llm_orb_fade": {"n": len(fade_pnl), "mean": float(np.mean(fade_pnl))},
            "non_earn_orb_dir": {"n": len(ne_orb_pnl), "mean": float(np.mean(ne_orb_pnl))},
            "pricescreen_ne_orb_fade": {"n": len(ps_orb_pnl), "mean": float(np.mean(ps_orb_pnl)) if len(ps_orb_pnl) > 0 else None},
        },
        "limitations": [
            "No market cap filter (data unavailable) — universe includes micro/small caps",
            "Universe gap-fade uses EOD close, LLM-flagged uses 60m PnL — different horizons",
            "Only 4 eval dates",
        ],
    },
    "e7_regime": {
        "warning": "VERY low-n: only 4 eval dates",
        "regime_classification": {d: {
            "regime": regime_info[d]['regime'],
            "spy_sma5": float(regime_info[d]['sma5']),
            "spy_prev_close": float(regime_info[d]['prev_close']),
            "distance_pct": float(regime_info[d]['distance_pct']),
            "spy_intra_ret": float(regime_info[d]['spy_intra_ret']) if not np.isnan(regime_info[d]['spy_intra_ret']) else None,
            "spy_morning_ret": float(regime_info[d]['spy_morning_ret']) if not np.isnan(regime_info[d]['spy_morning_ret']) else None,
        } for d in EVAL_DATES},
        "alpha": {
            "alpha_60m_mean": float(np.mean(alpha_60)),
            "alpha_60m_median": float(np.median(alpha_60)),
            "alpha_60m_positive_pct": float(np.mean(alpha_60 > 0) * 100),
            "alpha_t1_close_mean": float(np.mean(alpha_t1)),
            "alpha_t1_close_median": float(np.median(alpha_t1)),
            "alpha_t2_close_mean": float(np.mean(alpha_t2)),
            "alpha_t2_close_median": float(np.median(alpha_t2)),
        },
        "spy_correlation": {
            "spy_intra_vs_avg_pnl_r": float(corr) if not np.isnan(corr) else None,
            "spy_intra_vs_avg_pnl_p": float(pval) if not np.isnan(pval) else None,
            "spy_morning_vs_avg_pnl_r": float(corr_m) if not np.isnan(corr_m) else None,
        },
    },
}

out_path = os.path.join(RESULTS, "v3_e4_e7_results.json")
with open(out_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"\nSaved results: {out_path}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
E4 — Pure Price Screen vs LLM:
  Full universe gappers (>3%): {len(gappers_df_filtered)} tickers across 4 dates
  Universe gap-fade avg PnL:   {np.mean(u_pnl):+.3f}% (median {np.median(u_pnl):+.3f}%)
  LLM-flagged avg PnL (60m):   {np.mean(llm_pnl):+.3f}% (median {np.median(llm_pnl):+.3f}%)
  LLM adds selection value:    {'YES' if np.mean(llm_pnl) > np.mean(u_pnl) else 'NO'} (mean), {'YES' if np.median(llm_pnl) > np.median(u_pnl) else 'NO'} (median)
  Best ORB-enhanced filter:    Non-earn+ORB confirms dir: {np.mean(ne_orb_pnl):+.3f}% (n={len(ne_orb_pnl)})

E7 — Market Regime:
  Regimes found: {', '.join(sorted(df['regime'].unique()))}
  SPY intra vs avg PnL corr:  r={fmt(corr)} (n={len(valid)}, NOT significant)
  Alpha 60m:                   {np.mean(alpha_60):+.3f}% mean ({np.mean(alpha_60>0)*100:.0f}% positive)
  Alpha T+1 close:             {np.mean(alpha_t1):+.3f}% mean
  Alpha T+2 close:             {np.mean(alpha_t2):+.3f}% mean

  CAVEAT: 4 dates is far too few for regime conclusions.
  These results are DIRECTIONAL ONLY — not actionable without more data.
""")

print("Done.")
