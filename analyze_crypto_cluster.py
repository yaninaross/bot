"""
analyze_crypto_cluster.py — Deep asset analysis for Exp-97/110 crypto cluster
==============================================================================
Memory-efficient: trims pre-2022 data on load, processes single-asset analyses
one ticker at a time, only holds close-price series for cross-asset work.

Analyses:
  1. Spike frequency & magnitude (TRAIN / OPT / TEST / LIVE periods)
  2. Volatility profile (ATR%, daily range, tail distribution)
  3. Liquidity & microstructure (volume, spread, trade count, OBI)
  4. Correlation structure (inter-teacher, teacher-context)
  5. Intraday seasonality (spike rate by 30-min bucket)
  6. BTC beta & residual vol (miner-specific)
  7. Short-term momentum structure (autocorrelation, mean-reversion)

Usage:  python analyze_crypto_cluster.py
"""

import sys
import subprocess
import os
import io
import gc
import warnings

import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import pandas_ta as ta
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--quiet"])
    import pandas_ta as ta

from google.cloud import storage

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_ID = "110_MICRO"

TRAIN_TEACHERS = [
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
    'COIN', 'BITF', 'HUT', 'CIFR', 'BTBT', 'HOOD',
]

SCORING_POOL_EXTRAS = ['SMCI', 'NVDA', 'VRT', 'AMD', 'PLTR', 'TSLA']
CONTEXT_ASSETS = ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD']

DEPLOY_TICKERS = ['CLSK', 'IREN', 'RIOT', 'APLD']

ALL_ANALYSIS_TICKERS = TRAIN_TEACHERS + SCORING_POOL_EXTRAS

TICKER_FILE_MAP = {'BITF': 'KEEL'}
BUCKET_NAME = "time-series-bucket-ys"
INFERENCE_DIR = "Data/Inference"

SPIKE_THRESHOLD = 0.10
DATA_START_DATE = "2022-01-01"
TRAIN_CUTOFF_DATE = "2025-06-30"
OPT_START = "2025-07-01"
OPT_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-03-31"
LIVE_START = "2026-04-01"
LIVE_END = datetime.now(pytz.UTC).strftime("%Y-%m-%d")

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
KEEP_COLS = ['Open', 'High', 'Low', 'Close', 'Volume'] + MICRO_COLS
ROC_HORIZONS = [5, 30, 60, 120]
PST = pytz.timezone('America/Los_Angeles')

PERIODS = [
    ('TRAIN', DATA_START_DATE, TRAIN_CUTOFF_DATE),
    ('OPT',   OPT_START,       OPT_END),
    ('TEST',  TEST_START,      TEST_END),
    ('LIVE',  LIVE_START,      LIVE_END),
]

client = storage.Client()

# ==============================================================================
# DATA LOADING — trims to 2022+ and drops unused columns
# ==============================================================================
def load_master(ticker):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob = client.bucket(BUCKET_NAME).blob(
        f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet")
    if not blob.exists():
        print(f"   Missing {file_ticker}_master_inference.parquet")
        return None
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
    std_map = {
        'Total_vol': 'Total_Vol', 'Trade_count': 'Trade_Count',
        'Volume_delta': 'Volume_Delta', 'Delta_pct': 'Delta_Pct',
        'Obi': 'OBI', 'Bsize': 'bsize', 'Asize': 'asize',
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume',
    }
    lower_map = {k.lower(): v for k, v in std_map.items()}
    df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                  for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
    for c in MICRO_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df = df[df.index.notna()].sort_index()

    # ── TRIM to 2022+ and keep only needed columns ──
    cutoff = pd.Timestamp(DATA_START_DATE, tz='UTC')
    df = df[df.index >= cutoff]
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep]
    return df


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def load_and_heal(ticker):
    """Load, trim, heal one ticker. Returns None on failure."""
    df = load_master(ticker)
    if df is None:
        return None
    df = heal_data(df)
    bars = len(df)
    print(f"      {ticker}: {bars:,} bars (trimmed), "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


# ==============================================================================
# ANALYSIS FUNCTIONS (single-ticker, stateless)
# ==============================================================================

def compute_spike_stats(df, ticker, period_name, start_date, end_date):
    mask = (df.index >= pd.Timestamp(start_date, tz='UTC')) & \
           (df.index < pd.Timestamp(end_date, tz='UTC'))
    sub = df[mask]
    if sub.empty:
        return None

    pst_idx = sub.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute
    is_entry_window = (mins >= 360) & (mins <= 390)
    is_real = sub['Volume'] > 0
    is_safe = (pst_idx.hour < 13) | ((pst_idx.hour == 13) & (pst_idx.minute == 0))

    safe_high = np.where(is_safe & is_real, sub['High'].values, -1.0)
    safe_high_s = pd.Series(safe_high, index=sub.index)
    future_max = safe_high_s.groupby(safe_high_s.index.date).transform(
        lambda x: x[::-1].cummax()[::-1])

    close_safe = sub['Close'].replace(0, np.nan)
    spike_pct = ((future_max / close_safe) - 1).clip(lower=0).fillna(0)
    is_spike = (spike_pct >= SPIKE_THRESHOLD)

    entry_mask = is_entry_window & is_real
    entry_bars = sub[entry_mask]
    entry_spikes = is_spike[entry_mask]
    entry_mags = spike_pct[entry_mask]

    if entry_bars.empty:
        return None

    n_days = entry_bars.index.normalize().nunique()
    n_spike_bars = entry_spikes.sum()
    spike_rate = n_spike_bars / len(entry_bars) if len(entry_bars) > 0 else 0

    spike_mags = entry_mags[entry_spikes]
    spike_days = entry_bars[entry_spikes].index.normalize().nunique()
    days_with_spike_rate = spike_days / n_days if n_days > 0 else 0

    return {
        'ticker': ticker, 'period': period_name,
        'n_days': n_days, 'n_entry_bars': len(entry_bars),
        'n_spike_bars': int(n_spike_bars), 'spike_rate': spike_rate,
        'days_with_spike': spike_days, 'days_with_spike_pct': days_with_spike_rate,
        'spike_mag_mean': spike_mags.mean() if len(spike_mags) > 0 else 0,
        'spike_mag_median': spike_mags.median() if len(spike_mags) > 0 else 0,
        'spike_mag_p90': spike_mags.quantile(0.90) if len(spike_mags) > 0 else 0,
        'spike_mag_max': spike_mags.max() if len(spike_mags) > 0 else 0,
    }


def compute_volatility_profile(df, ticker, start_date, end_date):
    mask = (df.index >= pd.Timestamp(start_date, tz='UTC')) & \
           (df.index < pd.Timestamp(end_date, tz='UTC'))
    sub = df[mask]
    if sub.empty or 'Close' not in sub.columns:
        return None

    pst_idx = sub.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute
    mkt_hours = sub[(mins >= 390) & (mins <= 780) & (sub['Volume'] > 0)]
    if mkt_hours.empty or len(mkt_hours) < 200:
        return None

    returns_1m = mkt_hours['Close'].pct_change().dropna()
    if len(returns_1m) < 100:
        return None

    daily_hl = mkt_hours.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    daily_range_pct = ((daily_hl['High'] - daily_hl['Low']) / daily_hl['Close']).dropna()

    atr_14 = ta.atr(mkt_hours['High'], mkt_hours['Low'], mkt_hours['Close'], length=14)
    atr_pct = (atr_14 / mkt_hours['Close']).dropna()

    kurtosis = scipy_stats.kurtosis(returns_1m, fisher=True)
    skewness = scipy_stats.skew(returns_1m)

    pos_ret = returns_1m[returns_1m > 0]
    neg_ret = returns_1m[returns_1m < 0]
    up_tail = pos_ret.quantile(0.95) if len(pos_ret) > 10 else 0
    dn_tail = neg_ret.quantile(0.05) if len(neg_ret) > 10 else 0
    tail_ratio = (up_tail / abs(dn_tail)
                  if len(pos_ret) > 10 and len(neg_ret) > 10 and dn_tail != 0 else 0)

    return {
        'ticker': ticker,
        'avg_price': mkt_hours['Close'].mean(),
        'daily_range_pct_mean': daily_range_pct.mean() * 100,
        'atr_pct_mean': atr_pct.mean() * 100 if len(atr_pct) > 0 else 0,
        'ret_1m_std': returns_1m.std() * 100,
        'annualized_vol': returns_1m.std() * np.sqrt(252 * 390) * 100,
        'kurtosis': kurtosis, 'skewness': skewness,
        'up_tail_95': up_tail * 100, 'dn_tail_05': dn_tail * 100,
        'tail_ratio': tail_ratio,
        'pct_bars_up': (returns_1m > 0).mean() * 100,
    }


def compute_liquidity_profile(df, ticker, start_date, end_date):
    mask = (df.index >= pd.Timestamp(start_date, tz='UTC')) & \
           (df.index < pd.Timestamp(end_date, tz='UTC'))
    sub = df[mask]
    if sub.empty:
        return None

    pst_idx = sub.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute
    mkt_hours = sub[(mins >= 390) & (mins <= 780) & (sub['Volume'] > 0)]
    if mkt_hours.empty:
        return None

    entry_window = sub[(mins >= 360) & (mins <= 390) & (sub['Volume'] > 0)]

    avg_vol_1m = mkt_hours['Volume'].mean()
    med_vol_1m = mkt_hours['Volume'].median()
    dollar_vol = (mkt_hours['Volume'] * mkt_hours['Close']).mean()

    avg_spread = mkt_hours['Spread'].mean() if 'Spread' in mkt_hours.columns else np.nan
    close_mean = mkt_hours['Close'].mean()
    spread_pct = (avg_spread / close_mean * 100) if not np.isnan(avg_spread) and close_mean > 0 else np.nan

    avg_tc = mkt_hours['Trade_Count'].mean() if 'Trade_Count' in mkt_hours.columns else np.nan
    avg_obi = mkt_hours['OBI'].mean() if 'OBI' in mkt_hours.columns else np.nan

    entry_vol = entry_window['Volume'].mean() if not entry_window.empty else 0
    has_micro = mkt_hours['Trade_Count'].sum() > 0 if 'Trade_Count' in mkt_hours.columns else False

    return {
        'ticker': ticker,
        'avg_vol_1m': avg_vol_1m, 'med_vol_1m': med_vol_1m,
        'avg_dollar_vol_1m': dollar_vol,
        'avg_spread': avg_spread, 'spread_pct': spread_pct,
        'avg_trade_count': avg_tc, 'avg_obi': avg_obi,
        'entry_window_vol': entry_vol, 'has_micro_data': has_micro,
    }


def compute_momentum_structure(df, ticker, start_date, end_date):
    mask = (df.index >= pd.Timestamp(start_date, tz='UTC')) & \
           (df.index < pd.Timestamp(end_date, tz='UTC'))
    sub = df[mask]
    if sub.empty:
        return None

    pst_idx = sub.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute
    mkt = sub[(mins >= 390) & (mins <= 780) & (sub['Volume'] > 0)]
    if len(mkt) < 200:
        return None

    ret_1m = mkt['Close'].pct_change().dropna()
    ret_5m = mkt['Close'].pct_change(5).dropna()

    acf_1 = ret_1m.autocorr(lag=1) if len(ret_1m) > 10 else 0
    acf_5 = ret_1m.autocorr(lag=5) if len(ret_1m) > 10 else 0
    acf_30 = ret_1m.autocorr(lag=30) if len(ret_1m) > 30 else 0

    daily_ret = mkt['Close'].resample('D').last().dropna().pct_change().dropna()
    daily_acf_1 = daily_ret.autocorr(lag=1) if len(daily_ret) > 3 else 0

    big_up = ret_5m[ret_5m > ret_5m.quantile(0.95)]
    if len(big_up) > 5:
        following = []
        for ts in big_up.index:
            future = ret_5m.loc[ts:].iloc[1:7]
            if len(future) > 0:
                following.append(future.mean())
        mr = np.mean(following) if following else 0
    else:
        mr = 0

    return {
        'ticker': ticker,
        'acf_1m_lag1': acf_1, 'acf_1m_lag5': acf_5, 'acf_1m_lag30': acf_30,
        'daily_acf_lag1': daily_acf_1,
        'mean_rev_after_spike': mr * 100,
    }


def compute_intraday_seasonality(df, ticker, start_date, end_date):
    mask = (df.index >= pd.Timestamp(start_date, tz='UTC')) & \
           (df.index < pd.Timestamp(end_date, tz='UTC'))
    sub = df[mask]
    if sub.empty:
        return None

    pst_idx = sub.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute
    is_real = sub['Volume'] > 0
    is_safe = (pst_idx.hour < 13) | ((pst_idx.hour == 13) & (pst_idx.minute == 0))

    safe_high = np.where(is_safe & is_real, sub['High'].values, -1.0)
    safe_high_s = pd.Series(safe_high, index=sub.index)
    future_max = safe_high_s.groupby(safe_high_s.index.date).transform(
        lambda x: x[::-1].cummax()[::-1])
    close_safe = sub['Close'].replace(0, np.nan)
    is_spike = (future_max > (close_safe * (1 + SPIKE_THRESHOLD)))

    buckets = {}
    for bucket_start in range(360, 780, 30):
        bucket_end = bucket_start + 30
        bucket_mask = (mins >= bucket_start) & (mins < bucket_end) & is_real
        n = bucket_mask.sum()
        if n > 0:
            rate = is_spike[bucket_mask].mean()
            h, m = divmod(bucket_start, 60)
            buckets[f"{h:02d}:{m:02d}"] = rate

    return {'ticker': ticker, 'buckets': buckets}


# ==============================================================================
# CROSS-ASSET ANALYSES (use daily close series — low memory)
# ==============================================================================

def extract_daily_close(df, start_date, end_date):
    """Extract daily close from a healed df — compact representation."""
    mask = (df.index >= pd.Timestamp(start_date, tz='UTC')) & \
           (df.index < pd.Timestamp(end_date, tz='UTC'))
    sub = df[mask]
    if sub.empty:
        return None
    pst_idx = sub.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute
    mkt = sub[(mins >= 390) & (mins <= 780) & (sub['Volume'] > 0)]
    if mkt.empty:
        return None
    return mkt['Close'].resample('D').last().dropna()


def extract_5min_close(df, start_date, end_date):
    """Extract 5-min close during market hours — for correlations."""
    mask = (df.index >= pd.Timestamp(start_date, tz='UTC')) & \
           (df.index < pd.Timestamp(end_date, tz='UTC'))
    sub = df[mask]
    if sub.empty:
        return None
    pst_idx = sub.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute
    mkt = sub[(mins >= 390) & (mins <= 780) & (sub['Volume'] > 0)]
    if mkt.empty:
        return None
    return mkt['Close'].resample('5min').last().dropna()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 80)
    print("  CRYPTO CLUSTER DEEP ANALYSIS — Exp-97/110")
    print("  14 Teachers + 6 Scoring Pool Extras + Context Assets")
    print(f"  LIVE period: {LIVE_START} → {LIVE_END}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # PHASE 1: Single-ticker analyses (load → analyze → free)
    # ------------------------------------------------------------------
    spike_rows = []
    vol_rows = []
    liq_rows = []
    mom_rows = []
    season_results = {}

    daily_closes = {}
    fivemin_closes = {}

    print("\n[PHASE 1] Per-ticker analysis (load → compute → free)...")
    all_tickers = ALL_ANALYSIS_TICKERS + CONTEXT_ASSETS
    for i, ticker in enumerate(all_tickers, 1):
        print(f"\n   [{i}/{len(all_tickers)}] {ticker}...")
        df = load_and_heal(ticker)
        if df is None:
            print(f"      {ticker}: FAILED — skipping")
            continue

        is_analysis = ticker in ALL_ANALYSIS_TICKERS

        if is_analysis:
            for period_name, start, end in PERIODS:
                r = compute_spike_stats(df, ticker, period_name, start, end)
                if r:
                    spike_rows.append(r)

            r = compute_volatility_profile(df, ticker, DATA_START_DATE, TRAIN_CUTOFF_DATE)
            if r:
                r['teacher'] = 'TEACHER' if ticker in TRAIN_TEACHERS else 'EXTRA'
                vol_rows.append(r)

            r = compute_liquidity_profile(df, ticker, DATA_START_DATE, TRAIN_CUTOFF_DATE)
            if r:
                r['teacher'] = 'TEACHER' if ticker in TRAIN_TEACHERS else 'EXTRA'
                liq_rows.append(r)

            r = compute_momentum_structure(df, ticker, DATA_START_DATE, TRAIN_CUTOFF_DATE)
            if r:
                r['teacher'] = 'TEACHER' if ticker in TRAIN_TEACHERS else 'EXTRA'
                mom_rows.append(r)

            if ticker in DEPLOY_TICKERS:
                r = compute_intraday_seasonality(df, ticker, DATA_START_DATE, TRAIN_CUTOFF_DATE)
                if r:
                    season_results[ticker] = r

        dc = extract_daily_close(df, DATA_START_DATE, TRAIN_CUTOFF_DATE)
        if dc is not None:
            daily_closes[ticker] = dc

        if ticker in TRAIN_TEACHERS or ticker in CONTEXT_ASSETS:
            fc = extract_5min_close(df, DATA_START_DATE, TRAIN_CUTOFF_DATE)
            if fc is not None:
                fivemin_closes[ticker] = fc

        del df
        gc.collect()

    # ------------------------------------------------------------------
    # PRINT PHASE 1 RESULTS
    # ------------------------------------------------------------------

    # ── 1. SPIKE FREQUENCY ──
    print("\n" + "=" * 80)
    print("  [1/7] SPIKE FREQUENCY & MAGNITUDE")
    print("=" * 80)

    if spike_rows:
        spike_df = pd.DataFrame(spike_rows)

        for period_name in ['TRAIN', 'OPT', 'TEST', 'LIVE']:
            sub = spike_df[spike_df['period'] == period_name].copy()
            if sub.empty:
                continue
            print(f"\n   --- {period_name} Period ---")
            sub = sub.sort_values('spike_rate', ascending=False)
            sub['teacher'] = sub['ticker'].apply(lambda t: 'TEACHER' if t in TRAIN_TEACHERS else 'EXTRA')
            sub['deploy'] = sub['ticker'].apply(lambda t: '***' if t in DEPLOY_TICKERS else '')

            print(f"   {'Ticker':<8} {'Type':<8} {'Days':>5} {'SpikeRate':>9} {'DaysW/Spike':>12} "
                  f"{'MagMean':>8} {'MagMed':>7} {'MagP90':>7} {'MagMax':>7} {'Dply':>4}")
            print(f"   {'─'*8} {'─'*8} {'─'*5} {'─'*9} {'─'*12} "
                  f"{'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*4}")
            for _, r in sub.iterrows():
                print(f"   {r['ticker']:<8} {r['teacher']:<8} {r['n_days']:>5} "
                      f"{r['spike_rate']:>8.1%} {r['days_with_spike_pct']:>11.1%} "
                      f"{r['spike_mag_mean']:>7.1%} {r['spike_mag_median']:>6.1%} "
                      f"{r['spike_mag_p90']:>6.1%} {r['spike_mag_max']:>6.1%} {r['deploy']:>4}")

        print("\n   --- Spike Rate Stability (TRAIN → OPT → TEST → LIVE) ---")
        pivot = {}
        for _, row in spike_df.iterrows():
            t = row['ticker']
            if t not in pivot:
                pivot[t] = {}
            pivot[t][row['period']] = row['spike_rate']
        pivot = pd.DataFrame(pivot).T
        for src, dst, col in [('TRAIN','OPT','OPT/TRN'), ('OPT','TEST','TST/OPT'), ('TEST','LIVE','LIV/TST')]:
            if src in pivot.columns and dst in pivot.columns:
                pivot[col] = pivot[dst] / pivot[src].replace(0, np.nan)
        if 'TRAIN' in pivot.columns:
            pivot = pivot.sort_values('TRAIN', ascending=False)
        print(pivot.round(4).to_string())

    # ── 2. VOLATILITY ──
    print("\n" + "=" * 80)
    print("  [2/7] VOLATILITY PROFILE (Training Period)")
    print("=" * 80)

    if vol_rows:
        vol_df = pd.DataFrame(vol_rows).sort_values('daily_range_pct_mean', ascending=False)
        print(f"\n   {'Ticker':<8} {'Type':<8} {'AvgPx':>8} {'DlyRng%':>8} {'ATR%':>6} "
              f"{'1mStd%':>7} {'AnnVol%':>8} {'Kurt':>6} {'Skew':>6} {'UpTail':>7} "
              f"{'DnTail':>7} {'TailR':>6}")
        print(f"   {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} "
              f"{'─'*7} {'─'*8} {'─'*6} {'─'*6} {'─'*7} "
              f"{'─'*7} {'─'*6}")
        for _, r in vol_df.iterrows():
            d = '***' if r['ticker'] in DEPLOY_TICKERS else ''
            print(f"   {r['ticker']:<8} {r['teacher']:<8} {r['avg_price']:>7.1f} "
                  f"{r['daily_range_pct_mean']:>7.2f}% {r['atr_pct_mean']:>5.2f}% "
                  f"{r['ret_1m_std']:>6.3f}% {r['annualized_vol']:>7.1f}% "
                  f"{r['kurtosis']:>5.1f} {r['skewness']:>5.2f} "
                  f"{r['up_tail_95']:>6.3f}% {r['dn_tail_05']:>6.3f}% "
                  f"{r['tail_ratio']:>5.2f} {d}")

    # ── 3. LIQUIDITY ──
    print("\n" + "=" * 80)
    print("  [3/7] LIQUIDITY & MICROSTRUCTURE (Training Period)")
    print("=" * 80)

    if liq_rows:
        liq_df = pd.DataFrame(liq_rows).sort_values('avg_dollar_vol_1m', ascending=False)
        print(f"\n   {'Ticker':<8} {'Type':<8} {'AvgVol/m':>10} {'$Vol/m':>12} "
              f"{'Spread':>8} {'Sprd%':>6} {'Trades':>7} {'OBI':>6} "
              f"{'EntryVol':>10} {'Micro':>5}")
        print(f"   {'─'*8} {'─'*8} {'─'*10} {'─'*12} "
              f"{'─'*8} {'─'*6} {'─'*7} {'─'*6} "
              f"{'─'*10} {'─'*5}")
        for _, r in liq_df.iterrows():
            d = '***' if r['ticker'] in DEPLOY_TICKERS else ''
            print(f"   {r['ticker']:<8} {r['teacher']:<8} {r['avg_vol_1m']:>10,.0f} "
                  f"${r['avg_dollar_vol_1m']:>10,.0f} "
                  f"{r['avg_spread']:>7.3f} {r['spread_pct']:>5.2f}% "
                  f"{r['avg_trade_count']:>6.0f} {r['avg_obi']:>5.2f} "
                  f"{r['entry_window_vol']:>10,.0f} {'Y' if r['has_micro_data'] else 'N':>5} {d}")

    # ── 4. MOMENTUM ──
    print("\n" + "=" * 80)
    print("  [4/7] MOMENTUM STRUCTURE & AUTOCORRELATION (Training Period)")
    print("=" * 80)

    if mom_rows:
        mom_df = pd.DataFrame(mom_rows).sort_values('acf_1m_lag1', ascending=False)
        print(f"\n   {'Ticker':<8} {'Type':<8} {'ACF(1)':>7} {'ACF(5)':>7} {'ACF(30)':>8} "
              f"{'DlyACF':>7} {'MR%':>7}")
        print(f"   {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*7}")
        for _, r in mom_df.iterrows():
            d = '***' if r['ticker'] in DEPLOY_TICKERS else ''
            print(f"   {r['ticker']:<8} {r['teacher']:<8} {r['acf_1m_lag1']:>6.3f} "
                  f"{r['acf_1m_lag5']:>6.3f} {r['acf_1m_lag30']:>7.3f} "
                  f"{r['daily_acf_lag1']:>6.3f} {r['mean_rev_after_spike']:>6.3f}% {d}")

    # ── 5. INTRADAY SEASONALITY ──
    print("\n" + "=" * 80)
    print("  [5/7] INTRADAY SPIKE SEASONALITY (Training Period)")
    print("=" * 80)

    for ticker in DEPLOY_TICKERS:
        if ticker not in season_results:
            continue
        buckets = season_results[ticker]['buckets']
        if not buckets:
            continue
        print(f"\n   {ticker} — Spike rate by 30-min bucket (PST):")
        for bucket, rate in sorted(buckets.items()):
            bar = '█' * int(rate * 200)
            entry = ' ← ENTRY' if bucket == '06:00' else ''
            print(f"      {bucket}  {rate:>6.2%}  {bar}{entry}")

    # ------------------------------------------------------------------
    # PHASE 2: Cross-asset analyses (using compact daily/5min series)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  [6/7] BTC BETA & RESIDUAL VOLATILITY (Training Period)")
    print("=" * 80)

    btc_results = {}
    if 'BTCUSD' in daily_closes:
        btc_daily = daily_closes['BTCUSD'].pct_change().dropna()
        for ticker in ALL_ANALYSIS_TICKERS:
            if ticker not in daily_closes or ticker == 'BTCUSD':
                continue
            asset_daily = daily_closes[ticker].pct_change().dropna()
            merged = pd.concat([btc_daily, asset_daily], axis=1, keys=['BTC', ticker]).dropna()
            if len(merged) < 20:
                continue
            cov = np.cov(merged['BTC'], merged[ticker])
            beta = cov[0, 1] / cov[0, 0] if cov[0, 0] > 0 else 0
            corr = merged.corr().iloc[0, 1]
            residuals = merged[ticker] - beta * merged['BTC']
            resid_vol = residuals.std() * np.sqrt(252) * 100
            btc_results[ticker] = {
                'btc_beta': beta, 'btc_corr': corr,
                'r_squared': corr ** 2, 'residual_vol_ann': resid_vol,
            }

    if btc_results:
        btc_rows = sorted(btc_results.items(), key=lambda x: x[1]['btc_beta'], reverse=True)
        print(f"\n   {'Ticker':<8} {'Type':<8} {'Beta':>6} {'Corr':>6} {'R²':>6} {'ResidVol%':>10}")
        print(f"   {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*10}")
        for ticker, v in btc_rows:
            t = 'TEACHER' if ticker in TRAIN_TEACHERS else 'EXTRA'
            d = '***' if ticker in DEPLOY_TICKERS else ''
            print(f"   {ticker:<8} {t:<8} {v['btc_beta']:>5.2f} "
                  f"{v['btc_corr']:>5.2f} {v['r_squared']:>5.2f} "
                  f"{v['residual_vol_ann']:>9.1f}% {d}")

    # ── 6. CORRELATION MATRIX ──
    print("\n" + "=" * 80)
    print("  [7/7] CROSS-ASSET CORRELATIONS (Training Period, 5-min returns)")
    print("=" * 80)

    if len(fivemin_closes) >= 2:
        prices_5m = pd.DataFrame(fivemin_closes)
        returns_5m = prices_5m.pct_change().dropna()
        corr_matrix = returns_5m.corr()

        teachers_in = [t for t in TRAIN_TEACHERS if t in corr_matrix.index]
        if len(teachers_in) >= 2:
            print("\n   Teacher-to-Teacher mean correlations:")
            teacher_corr = corr_matrix.loc[teachers_in, teachers_in]
            for ticker in teachers_in:
                others = teacher_corr.loc[ticker].drop(ticker, errors='ignore')
                d = '***' if ticker in DEPLOY_TICKERS else ''
                print(f"   {ticker:<8} avg={others.mean():>5.3f}  "
                      f"max={others.max():>5.3f} ({others.idxmax()})  "
                      f"min={others.min():>5.3f} ({others.idxmin()}) {d}")

            ctx_in = [t for t in CONTEXT_ASSETS if t in corr_matrix.columns]
            if ctx_in:
                print("\n   Teacher-to-Context correlations:")
                for ticker in teachers_in:
                    parts = '  '.join([f"{c}={corr_matrix.loc[ticker, c]:>5.3f}" for c in ctx_in])
                    d = '***' if ticker in DEPLOY_TICKERS else ''
                    print(f"   {ticker:<8} {parts} {d}")

        del prices_5m, returns_5m
        gc.collect()

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SUMMARY: TEACHER vs NON-TEACHER COMPARISON")
    print("=" * 80)

    if spike_rows:
        spike_df = pd.DataFrame(spike_rows)
        train_spikes = spike_df[spike_df['period'] == 'TRAIN'].copy()
        train_spikes['group'] = train_spikes['ticker'].apply(
            lambda t: 'TEACHER' if t in TRAIN_TEACHERS else 'EXTRA')
        summary = train_spikes.groupby('group').agg({
            'spike_rate': ['mean', 'median', 'std'],
            'spike_mag_mean': ['mean', 'median'],
            'days_with_spike_pct': ['mean'],
            'n_spike_bars': ['sum', 'mean'],
        })
        print("\n   Spike Statistics (Training Period):")
        print(summary.round(4).to_string())

    if vol_rows:
        vol_all = pd.DataFrame(vol_rows)
        vol_all['group'] = vol_all['ticker'].apply(
            lambda t: 'TEACHER' if t in TRAIN_TEACHERS else 'EXTRA')
        vs = vol_all.groupby('group').agg({
            'daily_range_pct_mean': ['mean', 'median'],
            'annualized_vol': ['mean', 'median'],
            'kurtosis': ['mean'],
            'tail_ratio': ['mean'],
        })
        print("\n   Volatility Statistics (Training Period):")
        print(vs.round(3).to_string())

    if btc_results:
        print("\n   BTC Beta Summary:")
        t_betas = [v['btc_beta'] for t, v in btc_results.items() if t in TRAIN_TEACHERS]
        e_betas = [v['btc_beta'] for t, v in btc_results.items() if t in SCORING_POOL_EXTRAS]
        if t_betas:
            print(f"      Teachers:  avg={np.mean(t_betas):.2f}  "
                  f"std={np.std(t_betas):.2f}  [{min(t_betas):.2f}, {max(t_betas):.2f}]")
        if e_betas:
            print(f"      Extras:    avg={np.mean(e_betas):.2f}  "
                  f"std={np.std(e_betas):.2f}  [{min(e_betas):.2f}, {max(e_betas):.2f}]")

    # ── DEPLOY PROFILE CARDS ──
    print("\n" + "=" * 80)
    print("  DEPLOY CANDIDATE PROFILE CARDS")
    print("=" * 80)

    for ticker in DEPLOY_TICKERS:
        print(f"\n   ┌─── {ticker} ───────────────────────────────────┐")

        if spike_rows:
            for pn in ['TRAIN', 'OPT', 'TEST', 'LIVE']:
                r = spike_df[(spike_df['ticker'] == ticker) & (spike_df['period'] == pn)]
                if not r.empty:
                    r = r.iloc[0]
                    print(f"   │ {pn:<5}: spike_rate={r['spike_rate']:.1%}  "
                          f"mag={r['spike_mag_mean']:.1%}  days_w_spike={r['days_with_spike_pct']:.1%}")

        if vol_rows:
            vr = [r for r in vol_rows if r['ticker'] == ticker]
            if vr:
                vr = vr[0]
                print(f"   │ Vol: range={vr['daily_range_pct_mean']:.2f}%  "
                      f"ann={vr['annualized_vol']:.1f}%  "
                      f"kurt={vr['kurtosis']:.1f}  skew={vr['skewness']:.2f}")

        if ticker in btc_results:
            br = btc_results[ticker]
            print(f"   │ BTC: beta={br['btc_beta']:.2f}  corr={br['btc_corr']:.2f}  "
                  f"R²={br['r_squared']:.2f}  resid_vol={br['residual_vol_ann']:.1f}%")

        if liq_rows:
            lr = [r for r in liq_rows if r['ticker'] == ticker]
            if lr:
                lr = lr[0]
                print(f"   │ Liq: vol/m={lr['avg_vol_1m']:,.0f}  "
                      f"$/m={lr['avg_dollar_vol_1m']:,.0f}  "
                      f"spread={lr['spread_pct']:.2f}%")

        if mom_rows:
            mr = [r for r in mom_rows if r['ticker'] == ticker]
            if mr:
                mr = mr[0]
                print(f"   │ Mom: ACF(1)={mr['acf_1m_lag1']:.3f}  "
                      f"ACF(5)={mr['acf_1m_lag5']:.3f}  "
                      f"MR={mr['mean_rev_after_spike']:.3f}%")

        print(f"   └──────────────────────────────────────────────┘")

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
