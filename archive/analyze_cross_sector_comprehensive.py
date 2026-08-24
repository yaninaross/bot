"""
Comprehensive Cross-Sector Market Analysis — 2-Year View (Jul 2024 → Jul 2026)
=================================================================================
Uses INDEPENDENT data (Yahoo Finance daily bars) to validate conclusions.

Covers:
  - 15 sector ETFs + SPY/QQQ/IWM/BTC
  - ~40 individual speculative/growth stocks across 10 sectors
  - Monthly returns, volatility, Up/Down skew, drawdown
  - Focus on last 2 months (Jun-Jul 2026) vs historical baseline

NOT limited to GCS bucket — downloads from Yahoo Finance.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

W = 130

# ──────────────────────────────────────────────────────────────────────────────
# UNIVERSE
# ──────────────────────────────────────────────────────────────────────────────

SECTOR_ETFS = {
    'SPY':  'S&P 500',
    'QQQ':  'Nasdaq 100',
    'IWM':  'Russell 2000',
    'XLK':  'Tech',
    'XLF':  'Financials',
    'XLE':  'Energy',
    'XLV':  'Healthcare',
    'XLI':  'Industrials',
    'XLY':  'Consumer Disc',
    'XLP':  'Consumer Staple',
    'XLU':  'Utilities',
    'XLB':  'Materials',
    'XLRE': 'Real Estate',
    'XLC':  'Communication',
    'SMH':  'Semiconductor',
    'XBI':  'Biotech',
    'ARKK': 'Innov/Disrupt',
    'WGMI': 'Crypto Miners',
    'TAN':  'Solar',
    'ICLN': 'Clean Energy',
    'ITA':  'Aero/Defense',
    'IBB':  'Biotech Lg',
}

SPECULATIVE_STOCKS = {
    'crypto': {
        'label': 'Crypto',
        'tickers': ['RIOT', 'CLSK', 'MARA', 'COIN', 'MSTR', 'HUT', 'CIFR', 'IREN', 'WULF', 'CORZ'],
    },
    'defense': {
        'label': 'Defense',
        'tickers': ['RKLB', 'LUNR', 'KTOS', 'AVAV', 'PL', 'BKSY', 'RDW'],
    },
    'semiconductor': {
        'label': 'Semicon',
        'tickers': ['SMCI', 'ALAB', 'ARM', 'CRDO', 'MXL', 'WOLF', 'AEHR'],
    },
    'clean_energy': {
        'label': 'CleanEnergy',
        'tickers': ['FCEL', 'PLUG', 'BE', 'FLNC', 'ENPH', 'FSLR'],
    },
    'nuclear': {
        'label': 'Nuclear',
        'tickers': ['SMR', 'OKLO', 'NNE', 'CCJ', 'CEG', 'NXE'],
    },
    'biotech': {
        'label': 'Biotech',
        'tickers': ['MRNA', 'BEAM', 'NTLA', 'CRSP', 'TWST', 'EXAS'],
    },
    'space_evtol': {
        'label': 'Space/eVTOL',
        'tickers': ['ASTS', 'ACHR', 'JOBY', 'RCAT', 'SOUN'],
    },
    'ev_auto': {
        'label': 'EV/Auto',
        'tickers': ['TSLA', 'RIVN', 'LCID', 'NIO', 'LI'],
    },
    'fintech': {
        'label': 'Fintech',
        'tickers': ['SQ', 'UPST', 'HOOD', 'AFRM', 'SOFI'],
    },
    'consumer': {
        'label': 'Consumer',
        'tickers': ['CVNA', 'W', 'CHWY', 'DASH', 'ABNB', 'ROKU'],
    },
}

START_DATE = '2024-07-01'
END_DATE = '2026-07-18'

print("=" * W)
print("  COMPREHENSIVE MARKET ANALYSIS — 2-Year View with Independent Data")
print("  Yahoo Finance daily bars • Sector ETFs + Speculative Stocks")
print("=" * W)

# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n  Downloading data from Yahoo Finance ({START_DATE} → {END_DATE})...")

all_etf_tickers = list(SECTOR_ETFS.keys())
all_spec_tickers = []
for sec_info in SPECULATIVE_STOCKS.values():
    all_spec_tickers.extend(sec_info['tickers'])
all_spec_tickers = sorted(set(all_spec_tickers))

all_download = all_etf_tickers + all_spec_tickers
print(f"    ETFs: {len(all_etf_tickers)}, Speculative stocks: {len(all_spec_tickers)}, Total: {len(all_download)}")

raw = yf.download(all_download, start=START_DATE, end=END_DATE, group_by='ticker', progress=False, auto_adjust=True)

# Parse into per-ticker DataFrames
ticker_data = {}
for ticker in all_download:
    try:
        if len(all_download) > 1:
            cols = raw[ticker] if ticker in raw.columns.get_level_values(0) else None
        else:
            cols = raw
        if cols is None or cols.empty:
            continue
        df = cols[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
        if len(df) < 20:
            continue
        ticker_data[ticker] = df
    except Exception:
        continue

print(f"    Loaded: {len(ticker_data)} tickers with data")
missing = [t for t in all_download if t not in ticker_data]
if missing:
    print(f"    Missing: {', '.join(missing[:15])}{'...' if len(missing) > 15 else ''}")


# ──────────────────────────────────────────────────────────────────────────────
# COMPUTE MONTHLY METRICS
# ──────────────────────────────────────────────────────────────────────────────

MONTHS_2Y = []
for year in [2024, 2025, 2026]:
    for month in range(1, 13):
        if year == 2024 and month < 7:
            continue
        if year == 2026 and month > 7:
            continue
        m_start = f'{year}-{month:02d}-01'
        if month == 12:
            m_end = f'{year+1}-01-01'
        else:
            m_end = f'{year}-{month+1:02d}-01'
        m_label = f"{datetime(year, month, 1).strftime('%b')}{str(year)[-2:]}"
        MONTHS_2Y.append((m_label, m_start, m_end, year, month))

def compute_monthly_metrics(df, ticker):
    """Compute monthly metrics from daily OHLCV."""
    rows = []
    for m_label, m_start, m_end, year, month in MONTHS_2Y:
        m = df[(df.index >= m_start) & (df.index < m_end)]
        if len(m) < 3:
            continue

        close = m['Close'].values
        high = m['High'].values
        low = m['Low'].values
        opn = m['Open'].values
        vol = m['Volume'].values

        daily_return = np.diff(close) / close[:-1]
        daily_range = (high - low) / np.where(close > 0, close, np.nan)
        daily_up = np.maximum(close - opn, 0) / np.where(opn > 0, opn, np.nan)
        daily_down = np.minimum(close - opn, 0) / np.where(opn > 0, opn, np.nan)

        avg_up = np.nanmean(daily_up)
        avg_down_abs = abs(np.nanmean(daily_down))
        up_down = avg_up / avg_down_abs if avg_down_abs > 0.0001 else 999

        # Large move days (using daily range > X% of close)
        pct_big_days_4 = float(np.nanmean(daily_range > 0.04))
        pct_big_days_6 = float(np.nanmean(daily_range > 0.06))
        pct_big_days_8 = float(np.nanmean(daily_range > 0.08))

        # Large UP days specifically (close > open by X%)
        pct_up_4 = float(np.nanmean(daily_up > 0.04))
        pct_up_6 = float(np.nanmean(daily_up > 0.06))

        # Drawdown from month start
        peak = np.maximum.accumulate(close)
        dd = (close - peak) / peak
        max_dd = float(np.min(dd))

        rows.append({
            'ticker': ticker,
            'month': m_label,
            'year': year,
            'month_num': month,
            'n_days': len(m),
            'monthly_return': float(close[-1] / close[0] - 1),
            'avg_daily_range': float(np.nanmean(daily_range)),
            'avg_daily_return': float(np.nanmean(daily_return)) if len(daily_return) > 0 else 0,
            'daily_vol': float(np.nanstd(daily_return)) if len(daily_return) > 1 else 0,
            'up_down_ratio': up_down,
            'pct_green_days': float(np.mean(daily_return > 0)) if len(daily_return) > 0 else 0.5,
            'pct_big_days_4': pct_big_days_4,
            'pct_big_days_6': pct_big_days_6,
            'pct_big_days_8': pct_big_days_8,
            'pct_up_4': pct_up_4,
            'pct_up_6': pct_up_6,
            'max_drawdown': max_dd,
            'avg_dollar_volume': float(np.mean(vol * close)),
            'price_start': float(close[0]),
            'price_end': float(close[-1]),
        })
    return rows


all_monthly = []

for ticker in sorted(ticker_data.keys()):
    df = ticker_data[ticker]
    rows = compute_monthly_metrics(df, ticker)
    all_monthly.extend(rows)

mdf = pd.DataFrame(all_monthly)
print(f"  Computed: {len(mdf)} ticker-month observations")


# ──────────────────────────────────────────────────────────────────────────────
# PART A: SECTOR ETF ANALYSIS (2-year monthly grid)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'═' * W}")
print(f"  PART A: SECTOR ETF MONTHLY RETURNS — 2-Year Grid")
print(f"  Focus: is July 2026 unusual? Is it broad or sector-specific?")
print(f"{'═' * W}")

focus_etfs = ['SPY', 'QQQ', 'IWM', 'WGMI', 'SMH', 'XBI', 'ARKK', 'XLE', 'XLV',
              'XLI', 'XLF', 'XLY', 'XLU', 'ITA', 'ICLN', 'TAN']
recent_months = [m[0] for m in MONTHS_2Y if m[3] >= 2026]

print(f"\n  {'ETF':<6} {'Sector':<16}", end="")
for ml in recent_months:
    print(f" {ml:>6}", end="")
print(f"  {'2Y avg':>6}")
print(f"  {'─'*6} {'─'*16}", end="")
for _ in recent_months:
    print(f" {'─'*6}", end="")
print(f"  {'─'*6}")

for etf in focus_etfs:
    if etf not in ticker_data:
        continue
    label = SECTOR_ETFS.get(etf, etf)
    edf = mdf[mdf['ticker'] == etf]

    print(f"  {etf:<6} {label:<16}", end="")

    all_rets = []
    for ml in recent_months:
        row = edf[edf['month'] == ml]
        if len(row) > 0:
            r = row.iloc[0]['monthly_return']
            all_rets.append(r)
            print(f" {r:>+5.1%}", end="")
        else:
            print(f" {'—':>6}", end="")

    avg_ret = edf['monthly_return'].mean() if len(edf) > 0 else np.nan
    print(f"  {avg_ret:>+5.1%}" if not np.isnan(avg_ret) else f"  {'—':>6}")


# ──────────────────────────────────────────────────────────────────────────────
# PART B: VOLATILITY REGIME — Is July 2026 unusually quiet?
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'═' * W}")
print(f"  PART B: DAILY VOLATILITY (avg intraday range) — Is July unusually quiet?")
print(f"  Comparing Jul26 vs 2-year average and recent months")
print(f"{'═' * W}")

vol_etfs = ['SPY', 'QQQ', 'IWM', 'WGMI', 'SMH', 'ARKK', 'XBI', 'XLE', 'ITA', 'ICLN']
compare_months = ['Apr26', 'May26', 'Jun26', 'Jul26']

print(f"\n  {'ETF':<6} {'Sector':<16} {'2Y avg':>7}", end="")
for ml in compare_months:
    print(f" {ml:>6}", end="")
print(f"  {'Jul vs 2Y':>9}")
print(f"  {'─'*6} {'─'*16} {'─'*7}", end="")
for _ in compare_months:
    print(f" {'─'*6}", end="")
print(f"  {'─'*9}")

for etf in vol_etfs:
    if etf not in ticker_data:
        continue
    label = SECTOR_ETFS.get(etf, etf)
    edf = mdf[mdf['ticker'] == etf]

    avg_2y = edf['avg_daily_range'].mean()
    jul_row = edf[edf['month'] == 'Jul26']
    jul_vol = jul_row.iloc[0]['avg_daily_range'] if len(jul_row) > 0 else np.nan

    print(f"  {etf:<6} {label:<16} {avg_2y:>6.1%}", end="")
    for ml in compare_months:
        row = edf[edf['month'] == ml]
        if len(row) > 0:
            print(f" {row.iloc[0]['avg_daily_range']:>5.1%}", end="")
        else:
            print(f" {'—':>6}", end="")

    if not np.isnan(jul_vol) and avg_2y > 0:
        pct = jul_vol / avg_2y - 1
        print(f"  {pct:>+8.0%}")
    else:
        print(f"  {'—':>9}")


# ──────────────────────────────────────────────────────────────────────────────
# PART C: UP/DOWN RATIO — Direction of moves
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'═' * W}")
print(f"  PART C: UP/DOWN RATIO — Where are the moves going?")
print(f"  >1.0 = upside dominant, <1.0 = downside dominant")
print(f"{'═' * W}")

print(f"\n  {'ETF':<6} {'Sector':<16} {'2Y avg':>7}", end="")
for ml in compare_months:
    print(f" {ml:>6}", end="")
print(f"  {'Jul26':>6}")
print(f"  {'─'*6} {'─'*16} {'─'*7}", end="")
for _ in compare_months:
    print(f" {'─'*6}", end="")
print(f"  {'─'*6}")

for etf in vol_etfs:
    if etf not in ticker_data:
        continue
    label = SECTOR_ETFS.get(etf, etf)
    edf = mdf[mdf['ticker'] == etf]

    avg_2y = edf['up_down_ratio'].mean()

    print(f"  {etf:<6} {label:<16} {avg_2y:>6.2f}", end="")
    for ml in compare_months:
        row = edf[edf['month'] == ml]
        if len(row) > 0:
            val = row.iloc[0]['up_down_ratio']
            marker = '*' if val < 0.7 else (' ' if val < 1.0 else '+')
            print(f" {val:>5.2f}{marker}", end="")
        else:
            print(f" {'—':>6}", end="")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PART D: SPECULATIVE STOCKS BY SECTOR — Monthly returns + vol
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'═' * W}")
print(f"  PART D: SPECULATIVE STOCKS — Sector avg monthly returns")
print(f"  (avg return of individual stocks in each sector)")
print(f"{'═' * W}")

spec_monthly = {}
for sec_key, sec_info in SPECULATIVE_STOCKS.items():
    label = sec_info['label']
    sec_tickers = [t for t in sec_info['tickers'] if t in ticker_data]
    if not sec_tickers:
        continue

    sec_data = mdf[mdf['ticker'].isin(sec_tickers)]
    monthly_avg = sec_data.groupby('month').agg(
        avg_return=('monthly_return', 'mean'),
        avg_range=('avg_daily_range', 'mean'),
        avg_ud=('up_down_ratio', 'mean'),
        avg_big4=('pct_big_days_4', 'mean'),
        avg_up4=('pct_up_4', 'mean'),
        n_tickers=('ticker', 'nunique'),
    )
    spec_monthly[sec_key] = monthly_avg

# Table: sector returns
print(f"\n  {'Sector':<14} {'#':>2}", end="")
for ml in compare_months:
    print(f" {ml:>7}", end="")
print(f"  {'2Y avg':>7}")
print(f"  {'─'*14} {'─'*2}", end="")
for _ in compare_months:
    print(f" {'─'*7}", end="")
print(f"  {'─'*7}")

for sec_key, sec_info in SPECULATIVE_STOCKS.items():
    label = sec_info['label']
    if sec_key not in spec_monthly:
        continue
    sm = spec_monthly[sec_key]

    sec_tickers = [t for t in sec_info['tickers'] if t in ticker_data]
    print(f"  {label:<14} {len(sec_tickers):>2}", end="")

    for ml in compare_months:
        if ml in sm.index:
            print(f" {sm.loc[ml, 'avg_return']:>+6.1%}", end="")
        else:
            print(f" {'—':>7}", end="")

    sec_data = mdf[mdf['ticker'].isin(sec_tickers)]
    overall = sec_data['monthly_return'].mean()
    print(f"  {overall:>+6.1%}")


# ──────────────────────────────────────────────────────────────────────────────
# PART E: LARGE MOVE FREQUENCY — % of days with >4% daily range
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'═' * W}")
print(f"  PART E: LARGE MOVE DAYS (>4% daily range) — speculative stocks")
print(f"  (proxy for spike opportunity)")
print(f"{'═' * W}")

print(f"\n  {'Sector':<14}", end="")
for ml in compare_months:
    print(f" {ml:>7}", end="")
print(f"  {'2Y avg':>7}  {'Jul vs 2Y':>9}")
print(f"  {'─'*14}", end="")
for _ in compare_months:
    print(f" {'─'*7}", end="")
print(f"  {'─'*7}  {'─'*9}")

for sec_key, sec_info in SPECULATIVE_STOCKS.items():
    label = sec_info['label']
    if sec_key not in spec_monthly:
        continue
    sm = spec_monthly[sec_key]

    print(f"  {label:<14}", end="")
    jul_val = None
    vals_all = []
    for ml in compare_months:
        if ml in sm.index:
            v = sm.loc[ml, 'avg_big4']
            print(f" {v:>6.0%}", end="")
            if ml == 'Jul26':
                jul_val = v
        else:
            print(f" {'—':>7}", end="")

    sec_tickers = [t for t in sec_info['tickers'] if t in ticker_data]
    sec_data = mdf[mdf['ticker'].isin(sec_tickers)]
    avg_2y = sec_data['pct_big_days_4'].mean()
    print(f"  {avg_2y:>6.0%}", end="")

    if jul_val is not None and avg_2y > 0:
        print(f"  {jul_val / avg_2y - 1:>+8.0%}")
    else:
        print(f"  {'—':>9}")


# ──────────────────────────────────────────────────────────────────────────────
# PART F: UP-MOVE FREQUENCY — % of days with >4% close-over-open
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'═' * W}")
print(f"  PART F: BIG UP DAYS (close > open by 4%+) — speculative stocks")
print(f"  (direct proxy for spike model opportunity)")
print(f"{'═' * W}")

print(f"\n  {'Sector':<14}", end="")
for ml in compare_months:
    print(f" {ml:>7}", end="")
print(f"  {'2Y avg':>7}  {'Jul vs 2Y':>9}")
print(f"  {'─'*14}", end="")
for _ in compare_months:
    print(f" {'─'*7}", end="")
print(f"  {'─'*7}  {'─'*9}")

for sec_key, sec_info in SPECULATIVE_STOCKS.items():
    label = sec_info['label']
    if sec_key not in spec_monthly:
        continue
    sm = spec_monthly[sec_key]

    print(f"  {label:<14}", end="")
    jul_val = None
    for ml in compare_months:
        if ml in sm.index:
            v = sm.loc[ml, 'avg_up4']
            print(f" {v:>6.0%}", end="")
            if ml == 'Jul26':
                jul_val = v
        else:
            print(f" {'—':>7}", end="")

    sec_tickers = [t for t in sec_info['tickers'] if t in ticker_data]
    sec_data = mdf[mdf['ticker'].isin(sec_tickers)]
    avg_2y = sec_data['pct_up_4'].mean()
    print(f"  {avg_2y:>6.0%}", end="")

    if jul_val is not None and avg_2y > 0.001:
        print(f"  {jul_val / avg_2y - 1:>+8.0%}")
    else:
        print(f"  {'—':>9}")


# ──────────────────────────────────────────────────────────────────────────────
# PART G: HISTORICAL CONTEXT — How does Jul26 compare to other bear months?
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'═' * W}")
print(f"  PART G: HISTORICAL CONTEXT — Where does Jul26 rank in 2 years?")
print(f"{'═' * W}")

for etf, label in [('SPY', 'S&P 500'), ('IWM', 'Russell 2000'), ('WGMI', 'Crypto Miners'), ('SMH', 'Semiconductor')]:
    if etf not in ticker_data:
        continue
    edf = mdf[mdf['ticker'] == etf].sort_values('monthly_return')
    jul = edf[edf['month'] == 'Jul26']

    if len(jul) == 0:
        continue

    jul_ret = jul.iloc[0]['monthly_return']
    rank = int((edf['monthly_return'] < jul_ret).sum()) + 1
    total = len(edf)
    percentile = rank / total

    print(f"\n  {etf} ({label}):")
    print(f"    Jul26 return: {jul_ret:+.1%} — rank {rank}/{total} (percentile: {percentile:.0%})")
    print(f"    Worst 5 months:", end="")
    for _, row in edf.head(5).iterrows():
        print(f"  {row['month']} {row['monthly_return']:+.1%}", end="")
    print()
    print(f"    Best 5 months: ", end="")
    for _, row in edf.tail(5).iterrows():
        print(f"  {row['month']} {row['monthly_return']:+.1%}", end="")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PART H: BREADTH — How many sectors are positive vs negative?
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'═' * W}")
print(f"  PART H: MARKET BREADTH — How many sector ETFs are positive each month?")
print(f"{'═' * W}")

breadth_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB',
                'XLRE', 'XLC', 'SMH', 'XBI', 'ARKK', 'ITA', 'ICLN']

recent_months_ext = [m[0] for m in MONTHS_2Y if m[3] >= 2025 and m[4] >= 7 or m[3] >= 2026]
# Simpler: last 12 months
last_12 = [m[0] for m in MONTHS_2Y[-12:]]

print(f"\n  {'Month':<8} {'#Pos':>4} {'#Neg':>4} {'Breadth':>8} {'SPY':>7} {'Regime':>12}")
print(f"  {'─'*8} {'─'*4} {'─'*4} {'─'*8} {'─'*7} {'─'*12}")

for ml in last_12:
    pos = 0
    neg = 0
    for etf in breadth_etfs:
        row = mdf[(mdf['ticker'] == etf) & (mdf['month'] == ml)]
        if len(row) > 0:
            if row.iloc[0]['monthly_return'] > 0:
                pos += 1
            else:
                neg += 1
    total = pos + neg
    breadth = pos / total if total > 0 else 0

    spy_row = mdf[(mdf['ticker'] == 'SPY') & (mdf['month'] == ml)]
    spy_ret = spy_row.iloc[0]['monthly_return'] if len(spy_row) > 0 else np.nan

    if breadth > 0.7:
        regime = 'BROAD BULL'
    elif breadth > 0.5:
        regime = 'MIXED BULL'
    elif breadth > 0.3:
        regime = 'MIXED BEAR'
    else:
        regime = 'BROAD BEAR'

    print(f"  {ml:<8} {pos:>4} {neg:>4} {breadth:>7.0%}  {spy_ret:>+6.1%}  {regime:>12}" if not np.isnan(spy_ret) else f"  {ml:<8} {pos:>4} {neg:>4} {breadth:>7.0%}  {'—':>7}  {regime:>12}")


# ──────────────────────────────────────────────────────────────────────────────
# PART I: PER-TICKER SNAPSHOT — Top speculative names
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'═' * W}")
print(f"  PART I: INDIVIDUAL SPECULATIVE STOCK SNAPSHOT — July 2026")
print(f"  Sorted by monthly return within each sector")
print(f"{'═' * W}")

for sec_key, sec_info in SPECULATIVE_STOCKS.items():
    label = sec_info['label']
    sec_tickers = [t for t in sec_info['tickers'] if t in ticker_data]
    if not sec_tickers:
        continue

    jul_data = mdf[(mdf['ticker'].isin(sec_tickers)) & (mdf['month'] == 'Jul26')]
    jun_data = mdf[(mdf['ticker'].isin(sec_tickers)) & (mdf['month'] == 'Jun26')]

    if len(jul_data) == 0:
        continue

    print(f"\n  ── {label} ──")
    print(f"  {'Ticker':<7} {'JulRet':>7} {'JunRet':>7} │ {'Vol':>5} {'U/D':>5} {'Big4%':>5} {'Up4%':>5} │ {'MaxDD':>6} {'%Grn':>5}")
    print(f"  {'─'*7} {'─'*7} {'─'*7} │ {'─'*5} {'─'*5} {'─'*5} {'─'*5} │ {'─'*6} {'─'*5}")

    for _, row in jul_data.sort_values('monthly_return').iterrows():
        ticker = row['ticker']
        jun_row = jun_data[jun_data['ticker'] == ticker]
        jun_ret = jun_row.iloc[0]['monthly_return'] if len(jun_row) > 0 else np.nan

        print(f"  {ticker:<7} {row['monthly_return']:>+6.1%} ", end="")
        print(f"{jun_ret:>+6.1%}" if not np.isnan(jun_ret) else f"{'—':>7}", end="")
        print(f" │ {row['avg_daily_range']:>4.1%} {row['up_down_ratio']:>4.1f} {row['pct_big_days_4']:>4.0%} {row['pct_up_4']:>4.0%}", end="")
        print(f" │ {row['max_drawdown']:>+5.0%} {row['pct_green_days']:>4.0%}")


# ──────────────────────────────────────────────────────────────────────────────
# FINAL DIAGNOSIS
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'═' * W}")
print(f"  FINAL DIAGNOSIS — Validated with independent data")
print(f"{'═' * W}")

# Count bearish ETFs in July
jul_etf_data = mdf[(mdf['ticker'].isin(breadth_etfs)) & (mdf['month'] == 'Jul26')]
n_neg_etfs = int((jul_etf_data['monthly_return'] < 0).sum())
n_total_etfs = len(jul_etf_data)

spy_jul = mdf[(mdf['ticker'] == 'SPY') & (mdf['month'] == 'Jul26')]
spy_ret = spy_jul.iloc[0]['monthly_return'] if len(spy_jul) > 0 else np.nan

wgmi_jul = mdf[(mdf['ticker'] == 'WGMI') & (mdf['month'] == 'Jul26')]
wgmi_ret = wgmi_jul.iloc[0]['monthly_return'] if len(wgmi_jul) > 0 else np.nan

# Average speculative returns by sector
print(f"\n  MARKET STATE:")
print(f"    SPY July return:     {spy_ret:+.1%}" if not np.isnan(spy_ret) else "    SPY: no data")
print(f"    Sector breadth:      {n_neg_etfs}/{n_total_etfs} sector ETFs negative")
print(f"    WGMI (crypto):       {wgmi_ret:+.1%}" if not np.isnan(wgmi_ret) else "    WGMI: no data")

print(f"\n  SPECULATIVE SECTOR RETURNS (Jul26):")
for sec_key, sec_info in SPECULATIVE_STOCKS.items():
    label = sec_info['label']
    sec_tickers = [t for t in sec_info['tickers'] if t in ticker_data]
    sec_jul = mdf[(mdf['ticker'].isin(sec_tickers)) & (mdf['month'] == 'Jul26')]
    if len(sec_jul) > 0:
        avg_ret = sec_jul['monthly_return'].mean()
        avg_ud = sec_jul['up_down_ratio'].mean()
        avg_big = sec_jul['pct_big_days_4'].mean()
        print(f"    {label:<14} ret: {avg_ret:>+6.1%}  U/D: {avg_ud:.2f}  big days: {avg_big:.0%}")

# Confirmation check
print(f"\n  VALIDATION vs INTRADAY ANALYSIS:")
print(f"    ┌──────────────────────────────────────────────────────────────────────────────────┐")
print(f"    │  Finding from intraday (GCS) data          │ Confirmed by daily (YF) data?      │")
print(f"    │───────────────────────────────────────────────────────────────────────────────────│")

# Check each conclusion
conclusions = []

# 1. Broad bear
n_bear_spec = sum(1 for sk in SPECULATIVE_STOCKS
                  if any(t in ticker_data for t in SPECULATIVE_STOCKS[sk]['tickers'])
                  and mdf[(mdf['ticker'].isin([t for t in SPECULATIVE_STOCKS[sk]['tickers'] if t in ticker_data])) & (mdf['month'] == 'Jul26')]['monthly_return'].mean() < -0.02)
n_spec_total = sum(1 for sk in SPECULATIVE_STOCKS if any(t in ticker_data for t in SPECULATIVE_STOCKS[sk]['tickers']))

confirmed_bear = n_bear_spec >= n_spec_total * 0.7
print(f"    │  11/12 sectors bearish in July              │ {'YES' if confirmed_bear else 'NO':>3} — {n_bear_spec}/{n_spec_total} spec sectors down   │")

# 2. Crypto hit harder
crypto_tks = [t for t in SPECULATIVE_STOCKS['crypto']['tickers'] if t in ticker_data]
others_tks = [t for sk, si in SPECULATIVE_STOCKS.items() if sk != 'crypto'
              for t in si['tickers'] if t in ticker_data]
crypto_avg = mdf[(mdf['ticker'].isin(crypto_tks)) & (mdf['month'] == 'Jul26')]['monthly_return'].mean()
others_avg = mdf[(mdf['ticker'].isin(others_tks)) & (mdf['month'] == 'Jul26')]['monthly_return'].mean()
confirmed_crypto = crypto_avg < others_avg
print(f"    │  Crypto hit harder than others              │ {'YES' if confirmed_crypto else 'NO':>3} — crypto {crypto_avg:+.1%} vs others {others_avg:+.1%}  │")

# 3. Up/Down ratio < 1.0
jul_ud = mdf[mdf['month'] == 'Jul26']['up_down_ratio'].mean()
confirmed_ud = jul_ud < 1.0
print(f"    │  Up/Down ratio < 1.0 (downside dominant)    │ {'YES' if confirmed_ud else 'NO':>3} — avg U/D = {jul_ud:.2f}                     │")

# 4. Volume declining
jun_vol = mdf[mdf['month'] == 'Jun26']['avg_dollar_volume'].mean()
jul_vol = mdf[mdf['month'] == 'Jul26']['avg_dollar_volume'].mean()
vol_chg = jul_vol / jun_vol - 1 if jun_vol > 0 else 0
confirmed_vol = vol_chg < -0.10
print(f"    │  Volume declining                           │ {'YES' if confirmed_vol else 'NO':>3} — {vol_chg:+.0%} Jun→Jul                     │")

print(f"    └──────────────────────────────────────────────────────────────────────────────────┘")

all_confirmed = all([confirmed_bear, confirmed_crypto, confirmed_ud])
if all_confirmed:
    print(f"\n    ALL KEY FINDINGS CONFIRMED with independent data.")
    print(f"    This is not a data artifact — it's a real market regime shift.")
else:
    print(f"\n    PARTIAL CONFIRMATION — some findings differ with broader data.")

print(f"\n{'═' * W}")
print(f"  ANALYSIS COMPLETE — {len(ticker_data)} tickers, {len(SECTOR_ETFS)} ETFs + {len(all_spec_tickers)} stocks")
print(f"{'═' * W}")
