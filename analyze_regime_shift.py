"""
analyze_regime_shift.py -- Market Regime Shift Analysis: Backtest vs Live
=========================================================================
Compares April-May 2026 (WF backtest test period) vs June 1-8 2026 (live loss period).

Loads SPY, VIX, TLT, IWM + individual tickers (FCEL, LUNR, MVIS, SMCI)
from GCS parquet, computes daily regime metrics, and prints a structured report.

Run in Colab. Uses google.cloud.storage API (not FUSE).
"""

import numpy as np
import pandas as pd
import pytz
from datetime import datetime
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
PST = pytz.timezone('America/Los_Angeles')

# Period definitions
BACKTEST_START = pd.Timestamp('2026-04-01', tz='UTC')
BACKTEST_END   = pd.Timestamp('2026-05-29 23:59:59', tz='UTC')
LIVE_START     = pd.Timestamp('2026-06-01', tz='UTC')
LIVE_END       = pd.Timestamp('2026-06-08 23:59:59', tz='UTC')

# Pre-market window in PST: 6:00-6:30 AM
PM_START_MIN = 360   # 6*60
PM_END_MIN   = 390   # 6*60 + 30

# Regular hours in PST: 6:30 AM - 1:00 PM
RH_START_MIN = 390
RH_END_MIN   = 780   # 13*60

# Context tickers
REGIME_TICKERS = {
    'SPY': 'SPY_1m_full_historical_data.parquet',
    'VIX': 'I:VIX_1m_full_historical_data.parquet',
    'TLT': 'TLT_1m_full_historical_data.parquet',
    'IWM': 'IWM_1m_full_historical_data.parquet',
}

# Individual tickers with heavy live losses
LOSS_TICKERS = ['FCEL', 'LUNR', 'MVIS', 'SMCI']

W = 120

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_ohlcv(ticker, filename=None):
    """Load OHLCV parquet from GCS."""
    if filename is None:
        filename = f"{ticker}_1m_full_historical_data.parquet"
    uri = f"gs://{BUCKET_NAME}/{OHLCV_DIR}/{filename}"
    try:
        df = pd.read_parquet(uri)
    except Exception as e:
        print(f"  ERROR loading {ticker}: {e}")
        return None

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.sort_index()

    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df['Volume'] = df['Volume'].fillna(0)

    return df


def filter_period(df, start, end):
    """Filter dataframe to a date range."""
    return df[(df.index >= start) & (df.index <= end)]


# ==============================================================================
# DAILY METRICS COMPUTATION
# ==============================================================================
def compute_daily_metrics(df, ticker_name):
    """Compute daily regime metrics from 1-min OHLCV."""
    df_pst = df.copy()
    df_pst['pst_time'] = df_pst.index.tz_convert(PST)
    df_pst['date'] = df_pst['pst_time'].dt.date
    df_pst['mins'] = df_pst['pst_time'].dt.hour * 60 + df_pst['pst_time'].dt.minute

    # Pre-market bars: 6:00-6:30 AM PST
    pm_mask = (df_pst['mins'] >= PM_START_MIN) & (df_pst['mins'] < PM_END_MIN)
    # Regular hours bars: 6:30 AM - 1:00 PM PST
    rh_mask = (df_pst['mins'] >= RH_START_MIN) & (df_pst['mins'] <= RH_END_MIN)

    daily_rows = []

    for date, day_df in df_pst.groupby('date'):
        pm_bars = day_df[pm_mask.loc[day_df.index]]
        rh_bars = day_df[rh_mask.loc[day_df.index]]

        if len(pm_bars) == 0 and len(rh_bars) == 0:
            continue

        row = {'date': date, 'ticker': ticker_name}

        # Daily open-to-close return (using first/last bar of the day with volume)
        real_bars = day_df[day_df['Volume'] > 0]
        if len(real_bars) >= 2:
            day_open = real_bars['Open'].iloc[0]
            day_close = real_bars['Close'].iloc[-1]
            row['daily_return'] = (day_close / day_open - 1) if day_open > 0 else 0
        else:
            row['daily_return'] = 0

        # Daily high-low range (ATR proxy)
        if len(real_bars) > 0:
            day_high = real_bars['High'].max()
            day_low = real_bars['Low'].min()
            day_mid = (day_high + day_low) / 2
            row['daily_range_pct'] = (day_high - day_low) / day_mid if day_mid > 0 else 0
        else:
            row['daily_range_pct'] = 0

        # Pre-market volatility (std of log returns)
        pm_real = pm_bars[pm_bars['Volume'] > 0]
        if len(pm_real) >= 3:
            pm_log_ret = np.log(pm_real['Close'] / pm_real['Close'].shift(1)).dropna()
            row['pm_volatility'] = pm_log_ret.std() if len(pm_log_ret) > 0 else 0
            row['pm_range_pct'] = (pm_real['High'].max() - pm_real['Low'].min()) / pm_real['Close'].mean() if pm_real['Close'].mean() > 0 else 0
            row['pm_bars_count'] = len(pm_real)
        else:
            row['pm_volatility'] = 0
            row['pm_range_pct'] = 0
            row['pm_bars_count'] = len(pm_real) if len(pm_real) > 0 else 0

        # Regular hours volatility
        rh_real = rh_bars[rh_bars['Volume'] > 0]
        if len(rh_real) >= 3:
            rh_log_ret = np.log(rh_real['Close'] / rh_real['Close'].shift(1)).dropna()
            row['rh_volatility'] = rh_log_ret.std() if len(rh_log_ret) > 0 else 0
        else:
            row['rh_volatility'] = 0

        # Pre-market volume
        row['pm_volume'] = pm_real['Volume'].sum() if len(pm_real) > 0 else 0

        # VIX-specific: use Close as the level
        if ticker_name == 'VIX':
            if len(pm_real) > 0:
                row['vix_level'] = pm_real['Close'].median()
            elif len(real_bars) > 0:
                row['vix_level'] = real_bars['Close'].median()
            else:
                row['vix_level'] = np.nan

        daily_rows.append(row)

    return pd.DataFrame(daily_rows)


# ==============================================================================
# SPIKE ANALYSIS FOR INDIVIDUAL TICKERS
# ==============================================================================
def compute_spike_metrics(df, ticker_name, threshold=0.06):
    """Compute pre-market spike behavior metrics."""
    df_pst = df.copy()
    df_pst['pst_time'] = df_pst.index.tz_convert(PST)
    df_pst['date'] = df_pst['pst_time'].dt.date
    df_pst['mins'] = df_pst['pst_time'].dt.hour * 60 + df_pst['pst_time'].dt.minute

    # Pre-market bars with volume
    pm_mask = (df_pst['mins'] >= PM_START_MIN) & (df_pst['mins'] < PM_END_MIN) & (df_pst['Volume'] > 0)
    # Safe hours for future max (up to 1 PM PST)
    safe_mask = ((df_pst['pst_time'].dt.hour < 13) | ((df_pst['pst_time'].dt.hour == 13) & (df_pst['pst_time'].dt.minute == 0))) & (df_pst['Volume'] > 0)

    daily_rows = []
    for date, day_df in df_pst.groupby('date'):
        pm_bars = day_df[pm_mask.loc[day_df.index]]
        if len(pm_bars) == 0:
            continue

        safe_bars = day_df[safe_mask.loc[day_df.index]]

        row = {'date': date, 'ticker': ticker_name}

        # Overnight gap: compare first PM bar open to previous day's close
        row['pm_open'] = pm_bars['Open'].iloc[0]
        row['pm_close'] = pm_bars['Close'].iloc[-1]
        row['pm_high'] = pm_bars['High'].max()
        row['pm_low'] = pm_bars['Low'].min()

        # Pre-market range
        pm_mid = (row['pm_high'] + row['pm_low']) / 2
        row['pm_range_pct'] = (row['pm_high'] - row['pm_low']) / pm_mid if pm_mid > 0 else 0

        # Pre-market return (open to close of PM window)
        row['pm_return'] = (row['pm_close'] / row['pm_open'] - 1) if row['pm_open'] > 0 else 0

        # Future max from PM close through 1 PM
        future_bars = safe_bars[safe_bars.index > pm_bars.index[-1]]
        if len(future_bars) > 0:
            future_max = future_bars['High'].max()
            max_gain = (future_max / row['pm_close'] - 1) if row['pm_close'] > 0 else 0
            row['max_gain'] = max(0, max_gain)
            row['spiked'] = 1 if max_gain >= threshold else 0
        else:
            row['max_gain'] = 0
            row['spiked'] = 0

        # PM volume
        row['pm_volume'] = pm_bars['Volume'].sum()

        daily_rows.append(row)

    return pd.DataFrame(daily_rows)


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================
print(f"{'='*W}")
print(f"  MARKET REGIME SHIFT ANALYSIS")
print(f"  Backtest period:  April 1 - May 29, 2026")
print(f"  Live loss period: June 1 - 8, 2026")
print(f"{'='*W}")

# --- Load regime tickers ---
print(f"\n--- Loading regime tickers ---")
regime_data = {}
for ticker, filename in REGIME_TICKERS.items():
    print(f"  Loading {ticker}...", end=' ')
    df = load_ohlcv(ticker, filename)
    if df is not None:
        # Filter to our analysis window (March-June 2026)
        df = df[df.index >= pd.Timestamp('2026-03-01', tz='UTC')]
        metrics = compute_daily_metrics(df, ticker)
        regime_data[ticker] = metrics
        print(f"OK ({len(metrics)} days)")
    else:
        print("FAILED")

# --- Load loss tickers ---
print(f"\n--- Loading loss tickers ---")
loss_data = {}
loss_spike_data = {}
for ticker in LOSS_TICKERS:
    print(f"  Loading {ticker}...", end=' ')
    df = load_ohlcv(ticker)
    if df is not None:
        df_window = df[df.index >= pd.Timestamp('2026-03-01', tz='UTC')]
        metrics = compute_daily_metrics(df_window, ticker)
        spike_metrics = compute_spike_metrics(df_window, ticker, threshold=0.06)
        loss_data[ticker] = metrics
        loss_spike_data[ticker] = spike_metrics
        print(f"OK ({len(metrics)} days, {len(spike_metrics)} PM days)")
    else:
        print("FAILED")


# ==============================================================================
# TABLE 1: MARKET REGIME COMPARISON
# ==============================================================================
def period_stats(metrics_df, start, end):
    """Compute summary stats for a period."""
    mask = (pd.to_datetime(metrics_df['date']) >= start.tz_localize(None)) & \
           (pd.to_datetime(metrics_df['date']) <= end.tz_localize(None))
    period = metrics_df[mask]
    if len(period) == 0:
        return {}
    stats = {
        'n_days': len(period),
        'avg_daily_return': period['daily_return'].mean(),
        'med_daily_return': period['daily_return'].median(),
        'std_daily_return': period['daily_return'].std(),
        'avg_range_pct': period['daily_range_pct'].mean(),
        'avg_pm_vol': period['pm_volatility'].mean(),
        'avg_rh_vol': period['rh_volatility'].mean(),
        'pm_vs_rh_ratio': period['pm_volatility'].mean() / period['rh_volatility'].mean() if period['rh_volatility'].mean() > 0 else 0,
        'max_drawdown_day': period['daily_return'].min(),
        'best_day': period['daily_return'].max(),
        'pct_down_days': (period['daily_return'] < 0).mean(),
        'cumulative_return': (1 + period['daily_return']).prod() - 1,
    }
    if 'vix_level' in period.columns:
        vix_vals = period['vix_level'].dropna()
        if len(vix_vals) > 0:
            stats['avg_vix'] = vix_vals.mean()
            stats['med_vix'] = vix_vals.median()
            stats['max_vix'] = vix_vals.max()
            stats['min_vix'] = vix_vals.min()
            stats['vix_trend'] = vix_vals.iloc[-1] - vix_vals.iloc[0] if len(vix_vals) > 1 else 0
    return stats


print(f"\n\n{'='*W}")
print(f"  TABLE 1: MARKET REGIME COMPARISON — Backtest (Apr-May) vs Live (Jun 1-8)")
print(f"{'='*W}")

for ticker in ['SPY', 'VIX', 'TLT', 'IWM']:
    if ticker not in regime_data:
        print(f"\n  {ticker}: NO DATA")
        continue

    bt_stats = period_stats(regime_data[ticker], BACKTEST_START, BACKTEST_END)
    lv_stats = period_stats(regime_data[ticker], LIVE_START, LIVE_END)

    if not bt_stats or not lv_stats:
        print(f"\n  {ticker}: Incomplete period data")
        continue

    print(f"\n  --- {ticker} ---")
    print(f"  {'Metric':<30} {'Backtest (Apr-May)':>20} {'Live (Jun 1-8)':>20} {'Change':>15}")
    print(f"  {'─'*30} {'─'*20} {'─'*20} {'─'*15}")

    metrics_to_show = [
        ('Trading days',      'n_days',           '{:.0f}',  '{:.0f}'),
        ('Avg daily return',  'avg_daily_return',  '{:+.3%}', '{:+.3%}'),
        ('Median daily return','med_daily_return', '{:+.3%}', '{:+.3%}'),
        ('Cumulative return', 'cumulative_return', '{:+.2%}', '{:+.2%}'),
        ('Daily return stdev','std_daily_return',  '{:.4f}',  '{:.4f}'),
        ('Avg daily range %', 'avg_range_pct',     '{:.3%}',  '{:.3%}'),
        ('Avg PM volatility', 'avg_pm_vol',        '{:.5f}',  '{:.5f}'),
        ('Avg RH volatility', 'avg_rh_vol',        '{:.5f}',  '{:.5f}'),
        ('PM/RH vol ratio',   'pm_vs_rh_ratio',    '{:.2f}x', '{:.2f}x'),
        ('Worst day',         'max_drawdown_day',   '{:+.3%}', '{:+.3%}'),
        ('Best day',          'best_day',           '{:+.3%}', '{:+.3%}'),
        ('Pct down days',     'pct_down_days',      '{:.0%}',  '{:.0%}'),
    ]

    if ticker == 'VIX':
        metrics_to_show += [
            ('Avg VIX level',  'avg_vix',   '{:.2f}', '{:.2f}'),
            ('Median VIX',     'med_vix',   '{:.2f}', '{:.2f}'),
            ('Max VIX',        'max_vix',   '{:.2f}', '{:.2f}'),
            ('Min VIX',        'min_vix',   '{:.2f}', '{:.2f}'),
            ('VIX period trend','vix_trend', '{:+.2f}','{:+.2f}'),
        ]

    for label, key, fmt_bt, fmt_lv in metrics_to_show:
        bt_val = bt_stats.get(key, np.nan)
        lv_val = lv_stats.get(key, np.nan)
        if pd.isna(bt_val) or pd.isna(lv_val):
            continue

        bt_str = fmt_bt.format(bt_val)
        lv_str = fmt_lv.format(lv_val)

        # Change indicator
        if key in ('n_days',):
            change_str = ''
        elif bt_val != 0:
            pct_change = (lv_val / bt_val - 1) * 100
            if abs(pct_change) > 50:
                change_str = f'{pct_change:+.0f}% !!'
            elif abs(pct_change) > 20:
                change_str = f'{pct_change:+.0f}% !'
            else:
                change_str = f'{pct_change:+.0f}%'
        else:
            change_str = 'N/A'

        print(f"  {label:<30} {bt_str:>20} {lv_str:>20} {change_str:>15}")


# ==============================================================================
# TABLE 2: IWM vs SPY RELATIVE PERFORMANCE (Small-cap sentiment)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 2: SMALL-CAP SENTIMENT — IWM vs SPY Relative Performance")
print(f"{'='*W}")

if 'SPY' in regime_data and 'IWM' in regime_data:
    spy_df = regime_data['SPY'].set_index('date')
    iwm_df = regime_data['IWM'].set_index('date')
    merged = spy_df[['daily_return']].rename(columns={'daily_return': 'spy_ret'}).join(
        iwm_df[['daily_return']].rename(columns={'daily_return': 'iwm_ret'}),
        how='inner'
    )
    merged['iwm_minus_spy'] = merged['iwm_ret'] - merged['spy_ret']

    for period_name, start, end in [
        ('Backtest (Apr-May)', BACKTEST_START, BACKTEST_END),
        ('Live (Jun 1-8)', LIVE_START, LIVE_END),
    ]:
        mask = (pd.to_datetime(merged.index) >= start.tz_localize(None)) & \
               (pd.to_datetime(merged.index) <= end.tz_localize(None))
        p = merged[mask]
        if len(p) == 0:
            continue
        print(f"\n  {period_name} ({len(p)} days):")
        print(f"    Avg IWM-SPY spread:  {p['iwm_minus_spy'].mean():+.3%}")
        print(f"    Days IWM > SPY:      {(p['iwm_minus_spy'] > 0).sum()}/{len(p)} ({(p['iwm_minus_spy'] > 0).mean():.0%})")
        print(f"    Avg IWM return:      {p['iwm_ret'].mean():+.3%}")
        print(f"    Avg SPY return:      {p['spy_ret'].mean():+.3%}")
        print(f"    Corr(IWM, SPY):      {p['iwm_ret'].corr(p['spy_ret']):.3f}")
        print(f"    IWM cumulative:      {(1 + p['iwm_ret']).prod() - 1:+.2%}")
        print(f"    SPY cumulative:      {(1 + p['spy_ret']).prod() - 1:+.2%}")

    # Day-by-day for June
    june_mask = pd.to_datetime(merged.index) >= LIVE_START.tz_localize(None)
    june = merged[june_mask]
    if len(june) > 0:
        print(f"\n  Day-by-day June:")
        print(f"  {'Date':<12} {'SPY':>8} {'IWM':>8} {'IWM-SPY':>10} {'Direction'}")
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
        for date, r in june.iterrows():
            direction = 'IWM LEADS' if r['iwm_minus_spy'] > 0.002 else ('SPY LEADS' if r['iwm_minus_spy'] < -0.002 else 'EVEN')
            print(f"  {str(date):<12} {r['spy_ret']:>+7.2%} {r['iwm_ret']:>+7.2%} {r['iwm_minus_spy']:>+9.3%} {direction}")
else:
    print("  Missing SPY or IWM data")


# ==============================================================================
# TABLE 3: DAILY VIX LEVELS (DETAILED)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 3: DAILY VIX LEVELS — April through June 2026")
print(f"{'='*W}")

if 'VIX' in regime_data:
    vix_df = regime_data['VIX']
    vix_df_sorted = vix_df.sort_values('date')

    # Weekly summary
    vix_df_sorted['week'] = pd.to_datetime(vix_df_sorted['date']).dt.isocalendar().week
    vix_df_sorted['year_week'] = pd.to_datetime(vix_df_sorted['date']).dt.strftime('%Y-W%U')

    print(f"\n  Weekly VIX summary:")
    print(f"  {'Week':<12} {'Avg VIX':>8} {'Max VIX':>8} {'Min VIX':>8} {'Trend':>8}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for yw, grp in vix_df_sorted.groupby('year_week'):
        vix_vals = grp['vix_level'].dropna()
        if len(vix_vals) == 0:
            continue
        trend = vix_vals.iloc[-1] - vix_vals.iloc[0] if len(vix_vals) > 1 else 0
        trend_str = f'{trend:+.2f}'
        print(f"  {yw:<12} {vix_vals.mean():>8.2f} {vix_vals.max():>8.2f} {vix_vals.min():>8.2f} {trend_str:>8}")

    # Day-by-day for May-June transition
    may_jun = vix_df_sorted[pd.to_datetime(vix_df_sorted['date']) >= pd.Timestamp('2026-05-20')]
    if len(may_jun) > 0:
        print(f"\n  Day-by-day VIX (May 20 - Jun 8):")
        print(f"  {'Date':<12} {'VIX Level':>10} {'Daily Chg':>10} {'Regime'}")
        print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*15}")
        prev_vix = None
        for _, r in may_jun.iterrows():
            vix = r.get('vix_level', np.nan)
            if pd.isna(vix):
                continue
            chg = vix - prev_vix if prev_vix is not None else 0
            if vix > 25:
                regime = 'HIGH FEAR'
            elif vix > 20:
                regime = 'ELEVATED'
            elif vix > 16:
                regime = 'NORMAL'
            else:
                regime = 'LOW VOL'
            print(f"  {str(r['date']):<12} {vix:>10.2f} {chg:>+9.2f} {regime}")
            prev_vix = vix


# ==============================================================================
# TABLE 4: INDIVIDUAL TICKER PRE-MARKET SPIKE ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 4: PRE-MARKET SPIKE ANALYSIS — Loss Tickers")
print(f"  Threshold: 6% spike (consistent with model training)")
print(f"{'='*W}")

for ticker in LOSS_TICKERS:
    if ticker not in loss_spike_data:
        print(f"\n  {ticker}: NO DATA")
        continue

    spike_df = loss_spike_data[ticker]

    for period_name, start, end in [
        ('Backtest (Apr-May)', BACKTEST_START, BACKTEST_END),
        ('Live (Jun 1-8)', LIVE_START, LIVE_END),
    ]:
        mask = (pd.to_datetime(spike_df['date']) >= start.tz_localize(None)) & \
               (pd.to_datetime(spike_df['date']) <= end.tz_localize(None))
        p = spike_df[mask]
        if len(p) == 0:
            print(f"\n  {ticker} - {period_name}: No PM data")
            continue

        n_days = len(p)
        n_spikes = p['spiked'].sum()
        spike_rate = n_spikes / n_days if n_days > 0 else 0
        avg_max_gain = p['max_gain'].mean()
        avg_pm_range = p['pm_range_pct'].mean()
        avg_pm_return = p['pm_return'].mean()
        avg_pm_vol = p['pm_volume'].mean()

        print(f"\n  {ticker} - {period_name} ({n_days} PM days):")
        print(f"    Spike days (>=6%):   {n_spikes}/{n_days} = {spike_rate:.0%}")
        print(f"    Avg max gain:        {avg_max_gain:.2%}")
        print(f"    Avg PM range:        {avg_pm_range:.2%}")
        print(f"    Avg PM return:       {avg_pm_return:+.2%}")
        print(f"    Avg PM volume:       {avg_pm_vol:,.0f}")

    # Day-by-day June detail
    june_mask = pd.to_datetime(spike_df['date']) >= LIVE_START.tz_localize(None)
    june = spike_df[june_mask]
    if len(june) > 0:
        print(f"\n  {ticker} - June day-by-day:")
        print(f"  {'Date':<12} {'PM Rng%':>8} {'PM Ret':>8} {'MaxGain':>8} {'Spike?':>7} {'PM Vol':>12}")
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*12}")
        for _, r in june.iterrows():
            spike_flag = 'YES' if r['spiked'] else 'no'
            print(f"  {str(r['date']):<12} {r['pm_range_pct']:>7.2%} {r['pm_return']:>+7.2%} "
                  f"{r['max_gain']:>7.2%} {spike_flag:>7} {r['pm_volume']:>12,.0f}")


# ==============================================================================
# TABLE 5: OVERNIGHT GAP ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 5: OVERNIGHT GAP ANALYSIS — Loss Tickers")
print(f"{'='*W}")

for ticker in LOSS_TICKERS:
    if ticker not in loss_spike_data:
        continue

    spike_df = loss_spike_data[ticker].sort_values('date')

    # Compute overnight gap: today's PM open vs yesterday's PM close
    spike_df = spike_df.copy()
    spike_df['prev_close'] = spike_df['pm_close'].shift(1)
    spike_df['overnight_gap'] = (spike_df['pm_open'] / spike_df['prev_close'] - 1)
    spike_df.loc[spike_df['prev_close'] <= 0, 'overnight_gap'] = np.nan

    for period_name, start, end in [
        ('Backtest (Apr-May)', BACKTEST_START, BACKTEST_END),
        ('Live (Jun 1-8)', LIVE_START, LIVE_END),
    ]:
        mask = (pd.to_datetime(spike_df['date']) >= start.tz_localize(None)) & \
               (pd.to_datetime(spike_df['date']) <= end.tz_localize(None))
        p = spike_df[mask].dropna(subset=['overnight_gap'])
        if len(p) == 0:
            continue

        avg_gap = p['overnight_gap'].mean()
        med_gap = p['overnight_gap'].median()
        pct_gap_up = (p['overnight_gap'] > 0).mean()
        pct_gap_down = (p['overnight_gap'] < 0).mean()
        avg_abs_gap = p['overnight_gap'].abs().mean()

        print(f"\n  {ticker} - {period_name} ({len(p)} days):")
        print(f"    Avg overnight gap:   {avg_gap:+.2%}")
        print(f"    Median gap:          {med_gap:+.2%}")
        print(f"    Avg |gap|:           {avg_abs_gap:.2%}")
        print(f"    Gap up days:         {pct_gap_up:.0%}")
        print(f"    Gap down days:       {pct_gap_down:.0%}")


# ==============================================================================
# TABLE 6: CORRELATED SELL-OFF DAYS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 6: CORRELATED SELL-OFF DAYS — Multiple tickers down together")
print(f"{'='*W}")

# Merge all loss tickers daily returns
if loss_data:
    all_returns = pd.DataFrame()
    for ticker, metrics in loss_data.items():
        m = metrics[['date', 'daily_return']].rename(columns={'daily_return': ticker})
        if all_returns.empty:
            all_returns = m.set_index('date')
        else:
            all_returns = all_returns.join(m.set_index('date'), how='outer')

    # Add SPY for reference
    if 'SPY' in regime_data:
        spy_m = regime_data['SPY'][['date', 'daily_return']].rename(columns={'daily_return': 'SPY'})
        all_returns = all_returns.join(spy_m.set_index('date'), how='left')

    # Filter to analysis window
    all_returns = all_returns[(pd.to_datetime(all_returns.index) >= BACKTEST_START.tz_localize(None))]

    # Find days where 3+ loss tickers are down >2%
    loss_cols = [t for t in LOSS_TICKERS if t in all_returns.columns]
    all_returns['n_down_2pct'] = (all_returns[loss_cols] < -0.02).sum(axis=1)
    all_returns['avg_loss_ret'] = all_returns[loss_cols].mean(axis=1)

    corr_selloffs = all_returns[all_returns['n_down_2pct'] >= 2].sort_values('avg_loss_ret')

    if len(corr_selloffs) > 0:
        print(f"\n  Days with 2+ tickers down >2% (worst first):")
        print(f"  {'Date':<12} ", end='')
        for t in loss_cols:
            print(f"{'  ' + t:>8}", end='')
        print(f"{'  SPY':>8} {'#Down':>6}")
        print(f"  {'─'*12} ", end='')
        for t in loss_cols:
            print(f"{'─'*8}", end='')
        print(f"{'─'*8} {'─'*6}")

        for date, r in corr_selloffs.head(20).iterrows():
            print(f"  {str(date):<12} ", end='')
            for t in loss_cols:
                val = r.get(t, np.nan)
                if pd.isna(val):
                    print(f"{'N/A':>8}", end='')
                else:
                    marker = ' !!' if val < -0.05 else (' !' if val < -0.02 else '')
                    print(f"{val:>+7.1%}{marker[-1] if marker else ' '}", end='')
            spy_val = r.get('SPY', np.nan)
            if pd.isna(spy_val):
                print(f"{'N/A':>8}", end='')
            else:
                print(f"{spy_val:>+7.1%} ", end='')
            print(f"{int(r['n_down_2pct']):>5}")

        # Count by period
        for period_name, start, end in [
            ('Backtest (Apr-May)', BACKTEST_START, BACKTEST_END),
            ('Live (Jun 1-8)', LIVE_START, LIVE_END),
        ]:
            mask = (pd.to_datetime(corr_selloffs.index) >= start.tz_localize(None)) & \
                   (pd.to_datetime(corr_selloffs.index) <= end.tz_localize(None))
            n = mask.sum()
            total_mask = (pd.to_datetime(all_returns.index) >= start.tz_localize(None)) & \
                         (pd.to_datetime(all_returns.index) <= end.tz_localize(None))
            total = total_mask.sum()
            print(f"\n  {period_name}: {n} correlated sell-off days out of {total} trading days ({n/total:.0%})" if total > 0 else f"\n  {period_name}: no data")
    else:
        print("  No correlated sell-off days found")


# ==============================================================================
# TABLE 7: SUMMARY COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 7: REGIME SHIFT SUMMARY")
print(f"{'='*W}")

print(f"""
  COMPARISON: Backtest (Apr-May) vs Live (Jun 1-8)
  ─────────────────────────────────────────────────
""")

# Build summary
summary_items = []

# VIX
if 'VIX' in regime_data:
    bt_vix = period_stats(regime_data['VIX'], BACKTEST_START, BACKTEST_END)
    lv_vix = period_stats(regime_data['VIX'], LIVE_START, LIVE_END)
    if bt_vix and lv_vix:
        bt_avg = bt_vix.get('avg_vix', 0)
        lv_avg = lv_vix.get('avg_vix', 0)
        delta = lv_avg - bt_avg
        flag = 'CHANGED' if abs(delta) > 3 else 'SIMILAR'
        summary_items.append(('VIX Level', f'{bt_avg:.1f}', f'{lv_avg:.1f}', f'{delta:+.1f}', flag))

# SPY
if 'SPY' in regime_data:
    bt_spy = period_stats(regime_data['SPY'], BACKTEST_START, BACKTEST_END)
    lv_spy = period_stats(regime_data['SPY'], LIVE_START, LIVE_END)
    if bt_spy and lv_spy:
        summary_items.append(('SPY avg daily ret', f"{bt_spy['avg_daily_return']:+.3%}", f"{lv_spy['avg_daily_return']:+.3%}",
                             f"{lv_spy['avg_daily_return'] - bt_spy['avg_daily_return']:+.3%}",
                             'CHANGED' if abs(lv_spy['avg_daily_return'] - bt_spy['avg_daily_return']) > 0.003 else 'SIMILAR'))
        summary_items.append(('SPY daily vol (std)', f"{bt_spy['std_daily_return']:.4f}", f"{lv_spy['std_daily_return']:.4f}",
                             f"{lv_spy['std_daily_return']/bt_spy['std_daily_return']:.2f}x" if bt_spy['std_daily_return'] > 0 else 'N/A',
                             'CHANGED' if lv_spy['std_daily_return'] / bt_spy['std_daily_return'] > 1.5 or lv_spy['std_daily_return'] / bt_spy['std_daily_return'] < 0.67 else 'SIMILAR'))

# IWM
if 'IWM' in regime_data:
    bt_iwm = period_stats(regime_data['IWM'], BACKTEST_START, BACKTEST_END)
    lv_iwm = period_stats(regime_data['IWM'], LIVE_START, LIVE_END)
    if bt_iwm and lv_iwm:
        summary_items.append(('IWM avg daily ret', f"{bt_iwm['avg_daily_return']:+.3%}", f"{lv_iwm['avg_daily_return']:+.3%}",
                             f"{lv_iwm['avg_daily_return'] - bt_iwm['avg_daily_return']:+.3%}",
                             'CHANGED' if abs(lv_iwm['avg_daily_return'] - bt_iwm['avg_daily_return']) > 0.003 else 'SIMILAR'))

# Spike rates for loss tickers
for ticker in LOSS_TICKERS:
    if ticker in loss_spike_data:
        spike_df = loss_spike_data[ticker]
        bt_mask = (pd.to_datetime(spike_df['date']) >= BACKTEST_START.tz_localize(None)) & \
                  (pd.to_datetime(spike_df['date']) <= BACKTEST_END.tz_localize(None))
        lv_mask = (pd.to_datetime(spike_df['date']) >= LIVE_START.tz_localize(None)) & \
                  (pd.to_datetime(spike_df['date']) <= LIVE_END.tz_localize(None))
        bt_p = spike_df[bt_mask]
        lv_p = spike_df[lv_mask]
        if len(bt_p) > 0 and len(lv_p) > 0:
            bt_rate = bt_p['spiked'].mean()
            lv_rate = lv_p['spiked'].mean()
            bt_gain = bt_p['max_gain'].mean()
            lv_gain = lv_p['max_gain'].mean()
            summary_items.append((f'{ticker} spike rate', f'{bt_rate:.0%}', f'{lv_rate:.0%}',
                                 f'{lv_rate - bt_rate:+.0%}',
                                 'DECAYED' if lv_rate < bt_rate * 0.5 else ('SIMILAR' if abs(lv_rate - bt_rate) < 0.1 else 'CHANGED')))
            summary_items.append((f'{ticker} avg max gain', f'{bt_gain:.2%}', f'{lv_gain:.2%}',
                                 f'{lv_gain - bt_gain:+.2%}',
                                 'DECAYED' if lv_gain < bt_gain * 0.5 else 'SIMILAR'))

# Print summary table
print(f"  {'Metric':<25} {'Backtest':>15} {'Live':>15} {'Delta':>12} {'Status':>10}")
print(f"  {'─'*25} {'─'*15} {'─'*15} {'─'*12} {'─'*10}")
for label, bt_val, lv_val, delta, status in summary_items:
    flag = ' <<' if status in ('CHANGED', 'DECAYED') else ''
    print(f"  {label:<25} {bt_val:>15} {lv_val:>15} {delta:>12} {status:>10}{flag}")


# ==============================================================================
# VERDICT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

n_changed = sum(1 for _, _, _, _, s in summary_items if s in ('CHANGED', 'DECAYED'))
n_total = len(summary_items)
print(f"""
  Regime metrics flagged: {n_changed}/{n_total} changed or decayed

  Key questions answered:
  1. VIX level shift?      Check table above
  2. SPY direction change?  Check table above
  3. Small-cap rotation?    Check IWM vs SPY spread
  4. Spike pattern decay?   Check individual ticker spike rates
  5. Correlated sell-offs?  Check Table 6 for clustered losses

  If {n_changed}/{n_total} or more metrics show CHANGED/DECAYED, the environment
  has fundamentally shifted and the WF test period is no longer representative.
""")

print(f"{'='*W}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*W}")
