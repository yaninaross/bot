"""
analyze_defense_cluster.py — Defense cluster candidate screening
================================================================
Analyzes ALL 39 potential target assets (regardless of current experiment
assignment) to find the best candidates for the Exp-103 defense cluster.

Key additions vs v1:
  - Expanded from 19 to 39 tickers (all assets with micro data, minus
    context/index/delisted/large-cap-no-spike)
  - Cluster-fit metrics: ITA corr, IWM corr, mean teacher correlation
  - Final CANDIDATE RANKING dashboard sorted by composite score

Analyses:
  1. Spike frequency & magnitude (TRAIN / OPT / TEST / LIVE periods)
  2. Volatility profile (ATR%, daily range, tail distribution)
  3. Liquidity & microstructure (volume, spread, trade count, OBI)
  4. Cluster fit (ITA corr, IWM corr, mean teacher corr)
  5. Intraday seasonality (spike rate by 30-min bucket, top candidates)
  6. Momentum structure (autocorrelation, mean-reversion)
  7. CANDIDATE RANKING — composite score

Usage:  python analyze_defense_cluster.py
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

# --- Current Exp-103 teachers (for cluster-fit comparison) ---
CURRENT_TEACHERS = [
    'AVAV', 'KTOS', 'PLTR', 'BBAI',
    'RKLB', 'ASTS', 'LUNR', 'SPIR',
    'JOBY', 'NNDM', 'MVIS',
]

# --- Reference: top teachers that actually spike (for mean-teacher-corr) ---
TOP_TEACHERS = ['LUNR', 'BBAI', 'SPIR', 'ASTS', 'RCAT', 'RKLB']

# --- ALL 39 candidate tickers to analyze ---
ALL_CANDIDATES = [
    # Current defense teachers
    'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS',
    'LUNR', 'SPIR', 'JOBY', 'NNDM', 'MVIS',
    # Current defense scoring extras
    'RCAT', 'ACHR',
    # Crypto cluster (evaluate for defense fit too)
    'APLD', 'BTBT', 'CIFR', 'CLSK', 'COIN', 'CORZ',
    'HOOD', 'HUT', 'IREN', 'MARA', 'MIGI', 'MSTR',
    'RIOT', 'WULF',
    # New candidates (have micro data, not in any model)
    'ALAB', 'AMD', 'BE', 'NBIS', 'NVDA', 'PLUG',
    'SMCI', 'SOUN', 'TSLA', 'VRT',
    # Large-cap defense (include for completeness, will score low)
    'KEEL',
]

CONTEXT_ASSETS = ['ITA', 'VIX', 'TLT', 'IWM']

TICKER_FILE_MAP = {'BITF': 'KEEL'}
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference"

SPIKE_THRESHOLD = 0.10
DATA_START_DATE = "2022-01-01"
TRAIN_CUTOFF_DATE = "2025-10-01"
OPT_START = "2025-10-02"
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
# DATA LOADING
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
    df = load_master(ticker)
    if df is None:
        return None
    df = heal_data(df)
    bars = len(df)
    print(f"      {ticker}: {bars:,} bars, "
          f"{df.index.min().date()} -> {df.index.max().date()}")
    return df


# ==============================================================================
# ANALYSIS FUNCTIONS
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
# CROSS-ASSET HELPERS
# ==============================================================================

def extract_daily_close(df, start_date, end_date):
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
    print("=" * 90)
    print("  DEFENSE CLUSTER CANDIDATE SCREENING — Exp-103")
    print(f"  {len(ALL_CANDIDATES)} candidates + {len(CONTEXT_ASSETS)} context assets")
    print(f"  TRAIN: {DATA_START_DATE} -> {TRAIN_CUTOFF_DATE}")
    print(f"  LIVE:  {LIVE_START} -> {LIVE_END}")
    print("=" * 90)

    # ------------------------------------------------------------------
    # PHASE 1: Per-ticker analysis (load -> analyze -> free)
    # ------------------------------------------------------------------
    spike_rows = []
    vol_rows = []
    liq_rows = []
    mom_rows = []

    daily_closes = {}
    fivemin_closes = {}

    print("\n[PHASE 1] Per-ticker analysis...")
    all_tickers = list(dict.fromkeys(ALL_CANDIDATES + CONTEXT_ASSETS))
    for i, ticker in enumerate(all_tickers, 1):
        print(f"\n   [{i}/{len(all_tickers)}] {ticker}...")
        df = load_and_heal(ticker)
        if df is None:
            print(f"      FAILED — skipping")
            continue

        is_candidate = ticker in ALL_CANDIDATES

        if is_candidate:
            for period_name, start, end in PERIODS:
                r = compute_spike_stats(df, ticker, period_name, start, end)
                if r:
                    spike_rows.append(r)

            r = compute_volatility_profile(df, ticker, DATA_START_DATE, TRAIN_CUTOFF_DATE)
            if r:
                vol_rows.append(r)

            r = compute_liquidity_profile(df, ticker, DATA_START_DATE, TRAIN_CUTOFF_DATE)
            if r:
                liq_rows.append(r)

            r = compute_momentum_structure(df, ticker, DATA_START_DATE, TRAIN_CUTOFF_DATE)
            if r:
                mom_rows.append(r)

        dc = extract_daily_close(df, DATA_START_DATE, TRAIN_CUTOFF_DATE)
        if dc is not None:
            daily_closes[ticker] = dc

        fc = extract_5min_close(df, DATA_START_DATE, TRAIN_CUTOFF_DATE)
        if fc is not None:
            fivemin_closes[ticker] = fc

        del df
        gc.collect()

    # ------------------------------------------------------------------
    # PHASE 2: Cross-asset cluster-fit metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  [PHASE 2] CLUSTER-FIT METRICS")
    print("=" * 90)

    # --- ITA correlation (daily) ---
    ita_corr_map = {}
    if 'ITA' in daily_closes:
        ita_daily = daily_closes['ITA'].pct_change().dropna()
        for ticker in ALL_CANDIDATES:
            if ticker not in daily_closes:
                continue
            asset_daily = daily_closes[ticker].pct_change().dropna()
            merged = pd.concat([ita_daily, asset_daily], axis=1, keys=['ITA', ticker]).dropna()
            if len(merged) < 20:
                continue
            corr = merged.corr().iloc[0, 1]
            cov = np.cov(merged['ITA'], merged[ticker])
            beta = cov[0, 1] / cov[0, 0] if cov[0, 0] > 0 else 0
            ita_corr_map[ticker] = {'ita_corr': corr, 'ita_beta': beta}

    # --- IWM correlation (daily) ---
    iwm_corr_map = {}
    if 'IWM' in daily_closes:
        iwm_daily = daily_closes['IWM'].pct_change().dropna()
        for ticker in ALL_CANDIDATES:
            if ticker not in daily_closes:
                continue
            asset_daily = daily_closes[ticker].pct_change().dropna()
            merged = pd.concat([iwm_daily, asset_daily], axis=1, keys=['IWM', ticker]).dropna()
            if len(merged) < 20:
                continue
            corr = merged.corr().iloc[0, 1]
            cov = np.cov(merged['IWM'], merged[ticker])
            beta = cov[0, 1] / cov[0, 0] if cov[0, 0] > 0 else 0
            iwm_corr_map[ticker] = {'iwm_corr': corr, 'iwm_beta': beta}

    # --- Mean teacher correlation (5-min returns vs TOP_TEACHERS) ---
    teacher_corr_map = {}
    if len(fivemin_closes) >= 2:
        prices_5m = pd.DataFrame(fivemin_closes)
        returns_5m = prices_5m.pct_change().dropna()
        corr_matrix = returns_5m.corr()

        top_in = [t for t in TOP_TEACHERS if t in corr_matrix.columns]
        for ticker in ALL_CANDIDATES:
            if ticker not in corr_matrix.columns:
                continue
            others = [t for t in top_in if t != ticker]
            if not others:
                continue
            mean_corr = corr_matrix.loc[ticker, others].mean()
            max_corr = corr_matrix.loc[ticker, others].max()
            max_corr_name = corr_matrix.loc[ticker, others].idxmax()
            teacher_corr_map[ticker] = {
                'mean_teacher_corr': mean_corr,
                'max_teacher_corr': max_corr,
                'most_similar': max_corr_name,
            }

        del prices_5m, returns_5m
        gc.collect()

    # ------------------------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------------------------

    # -- 1. SPIKE FREQUENCY --
    print("\n" + "=" * 90)
    print("  [1/7] SPIKE FREQUENCY & MAGNITUDE")
    print("=" * 90)

    spike_df = pd.DataFrame()
    if spike_rows:
        spike_df = pd.DataFrame(spike_rows)
        for period_name in ['TRAIN', 'OPT', 'TEST', 'LIVE']:
            sub = spike_df[spike_df['period'] == period_name].copy()
            if sub.empty:
                continue
            print(f"\n   --- {period_name} Period ---")
            sub = sub.sort_values('spike_rate', ascending=False)
            sub['cur'] = sub['ticker'].apply(
                lambda t: 'DEF' if t in CURRENT_TEACHERS else
                          ('XTRA' if t in ['RCAT','ACHR'] else 'NEW'))
            print(f"   {'Ticker':<8} {'Grp':<5} {'Days':>5} {'SpikeRate':>9} {'DaysW/Spk':>10} "
                  f"{'MagMean':>8} {'MagMed':>7} {'MagP90':>7} {'MagMax':>7}")
            print(f"   {'─'*8} {'─'*5} {'─'*5} {'─'*9} {'─'*10} "
                  f"{'─'*8} {'─'*7} {'─'*7} {'─'*7}")
            for _, r in sub.iterrows():
                print(f"   {r['ticker']:<8} {r['cur']:<5} {r['n_days']:>5} "
                      f"{r['spike_rate']:>8.1%} {r['days_with_spike_pct']:>9.1%} "
                      f"{r['spike_mag_mean']:>7.1%} {r['spike_mag_median']:>6.1%} "
                      f"{r['spike_mag_p90']:>6.1%} {r['spike_mag_max']:>6.1%}")

        print("\n   --- Spike Rate Stability (TRAIN -> OPT -> TEST -> LIVE) ---")
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

    # -- 2. VOLATILITY --
    print("\n" + "=" * 90)
    print("  [2/7] VOLATILITY PROFILE (Training Period)")
    print("=" * 90)

    if vol_rows:
        vol_df = pd.DataFrame(vol_rows).sort_values('daily_range_pct_mean', ascending=False)
        print(f"\n   {'Ticker':<8} {'AvgPx':>8} {'DlyRng%':>8} {'ATR%':>6} "
              f"{'1mStd%':>7} {'AnnVol%':>8} {'Kurt':>7} {'Skew':>6} {'TailR':>6}")
        print(f"   {'─'*8} {'─'*8} {'─'*8} {'─'*6} "
              f"{'─'*7} {'─'*8} {'─'*7} {'─'*6} {'─'*6}")
        for _, r in vol_df.iterrows():
            print(f"   {r['ticker']:<8} {r['avg_price']:>7.1f} "
                  f"{r['daily_range_pct_mean']:>7.2f}% {r['atr_pct_mean']:>5.2f}% "
                  f"{r['ret_1m_std']:>6.3f}% {r['annualized_vol']:>7.1f}% "
                  f"{r['kurtosis']:>6.1f} {r['skewness']:>5.2f} "
                  f"{r['tail_ratio']:>5.2f}")

    # -- 3. LIQUIDITY --
    print("\n" + "=" * 90)
    print("  [3/7] LIQUIDITY & MICROSTRUCTURE (Training Period)")
    print("=" * 90)

    if liq_rows:
        liq_df = pd.DataFrame(liq_rows).sort_values('avg_dollar_vol_1m', ascending=False)
        print(f"\n   {'Ticker':<8} {'AvgVol/m':>10} {'$Vol/m':>12} "
              f"{'Sprd%':>6} {'Trades':>7} {'OBI':>6} {'Micro':>5}")
        print(f"   {'─'*8} {'─'*10} {'─'*12} "
              f"{'─'*6} {'─'*7} {'─'*6} {'─'*5}")
        for _, r in liq_df.iterrows():
            print(f"   {r['ticker']:<8} {r['avg_vol_1m']:>10,.0f} "
                  f"${r['avg_dollar_vol_1m']:>10,.0f} "
                  f"{r['spread_pct']:>5.2f}% "
                  f"{r['avg_trade_count']:>6.0f} {r['avg_obi']:>5.2f} "
                  f"{'Y' if r['has_micro_data'] else 'N':>5}")

    # -- 4. CLUSTER FIT --
    print("\n" + "=" * 90)
    print("  [4/7] CLUSTER-FIT METRICS")
    print(f"        ITA corr = defense sector exposure")
    print(f"        IWM corr = small-cap risk-on factor")
    print(f"        Teacher corr = avg 5-min corr with top teachers ({', '.join(TOP_TEACHERS)})")
    print("=" * 90)

    cluster_rows = []
    for ticker in ALL_CANDIDATES:
        row = {'ticker': ticker}
        if ticker in ita_corr_map:
            row['ita_corr'] = ita_corr_map[ticker]['ita_corr']
            row['ita_beta'] = ita_corr_map[ticker]['ita_beta']
        else:
            row['ita_corr'] = np.nan
            row['ita_beta'] = np.nan
        if ticker in iwm_corr_map:
            row['iwm_corr'] = iwm_corr_map[ticker]['iwm_corr']
            row['iwm_beta'] = iwm_corr_map[ticker]['iwm_beta']
        else:
            row['iwm_corr'] = np.nan
            row['iwm_beta'] = np.nan
        if ticker in teacher_corr_map:
            row['teacher_corr'] = teacher_corr_map[ticker]['mean_teacher_corr']
            row['most_similar'] = teacher_corr_map[ticker]['most_similar']
        else:
            row['teacher_corr'] = np.nan
            row['most_similar'] = ''
        cluster_rows.append(row)

    cluster_df = pd.DataFrame(cluster_rows).set_index('ticker')
    cluster_df = cluster_df.sort_values('teacher_corr', ascending=False)

    print(f"\n   {'Ticker':<8} {'ITA_Corr':>9} {'ITA_Beta':>9} {'IWM_Corr':>9} "
          f"{'IWM_Beta':>9} {'Tchr_Corr':>10} {'MostSim':<8}")
    print(f"   {'─'*8} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*10} {'─'*8}")
    for ticker, r in cluster_df.iterrows():
        grp = 'DEF' if ticker in CURRENT_TEACHERS else ('XTRA' if ticker in ['RCAT','ACHR'] else 'NEW')
        print(f"   {ticker:<8} {r['ita_corr']:>8.3f} {r['ita_beta']:>8.2f} "
              f"{r['iwm_corr']:>8.3f} {r['iwm_beta']:>8.2f} "
              f"{r['teacher_corr']:>9.3f} {r['most_similar']:<8} [{grp}]")

    # -- 5. INTRADAY SEASONALITY (top spike candidates only) --
    print("\n" + "=" * 90)
    print("  [5/7] INTRADAY SPIKE SEASONALITY (top 10 by spike rate)")
    print("=" * 90)

    top_spike_tickers = []
    if not spike_df.empty:
        train_sr = spike_df[spike_df['period'] == 'TRAIN'].sort_values('spike_rate', ascending=False)
        top_spike_tickers = train_sr['ticker'].head(10).tolist()

    season_results = {}
    if top_spike_tickers:
        for ticker in top_spike_tickers:
            df = load_and_heal(ticker)
            if df is None:
                continue
            r = compute_intraday_seasonality(df, ticker, DATA_START_DATE, TRAIN_CUTOFF_DATE)
            if r:
                season_results[ticker] = r
            del df
            gc.collect()

        for ticker in top_spike_tickers:
            if ticker not in season_results:
                continue
            buckets = season_results[ticker]['buckets']
            if not buckets:
                continue
            print(f"\n   {ticker} — Spike rate by 30-min bucket (PST):")
            for bucket, rate in sorted(buckets.items()):
                bar = '#' * int(rate * 200)
                entry = ' <- ENTRY' if bucket == '06:00' else ''
                print(f"      {bucket}  {rate:>6.2%}  {bar}{entry}")

    # -- 6. MOMENTUM --
    print("\n" + "=" * 90)
    print("  [6/7] MOMENTUM STRUCTURE (Training Period)")
    print("=" * 90)

    if mom_rows:
        mom_df = pd.DataFrame(mom_rows).sort_values('acf_1m_lag1', ascending=False)
        print(f"\n   {'Ticker':<8} {'ACF(1)':>7} {'ACF(5)':>7} {'ACF(30)':>8} "
              f"{'DlyACF':>7} {'MR%':>7}")
        print(f"   {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*7}")
        for _, r in mom_df.iterrows():
            print(f"   {r['ticker']:<8} {r['acf_1m_lag1']:>6.3f} "
                  f"{r['acf_1m_lag5']:>6.3f} {r['acf_1m_lag30']:>7.3f} "
                  f"{r['daily_acf_lag1']:>6.3f} {r['mean_rev_after_spike']:>6.3f}%")

    # ==================================================================
    # [7/7] CANDIDATE RANKING — composite dashboard
    # ==================================================================
    print("\n" + "=" * 90)
    print("  [7/7] CANDIDATE RANKING")
    print("=" * 90)

    ranking_rows = []
    for ticker in ALL_CANDIDATES:
        row = {'ticker': ticker}

        # Spike rate (TRAIN)
        if not spike_df.empty:
            train_match = spike_df[(spike_df['ticker'] == ticker) & (spike_df['period'] == 'TRAIN')]
            if not train_match.empty:
                row['spike_train'] = train_match.iloc[0]['spike_rate']
                row['spike_mag'] = train_match.iloc[0]['spike_mag_mean']
            else:
                row['spike_train'] = 0
                row['spike_mag'] = 0
            # OOS spike rate (avg of OPT + TEST + LIVE)
            oos = spike_df[(spike_df['ticker'] == ticker) & (spike_df['period'].isin(['OPT','TEST','LIVE']))]
            row['spike_oos'] = oos['spike_rate'].mean() if not oos.empty else 0
        else:
            row['spike_train'] = 0
            row['spike_mag'] = 0
            row['spike_oos'] = 0

        # Cluster fit
        row['ita_corr'] = ita_corr_map.get(ticker, {}).get('ita_corr', 0)
        row['iwm_corr'] = iwm_corr_map.get(ticker, {}).get('iwm_corr', 0)
        row['teacher_corr'] = teacher_corr_map.get(ticker, {}).get('mean_teacher_corr', 0)

        # Volatility
        vol_match = [v for v in vol_rows if v['ticker'] == ticker]
        row['ann_vol'] = vol_match[0]['annualized_vol'] if vol_match else 0
        row['daily_range'] = vol_match[0]['daily_range_pct_mean'] if vol_match else 0

        # Liquidity
        liq_match = [l for l in liq_rows if l['ticker'] == ticker]
        row['dollar_vol'] = liq_match[0]['avg_dollar_vol_1m'] if liq_match else 0

        # Group
        if ticker in CURRENT_TEACHERS:
            row['group'] = 'DEF'
        elif ticker in ['RCAT', 'ACHR']:
            row['group'] = 'XTRA'
        else:
            row['group'] = 'NEW'

        ranking_rows.append(row)

    rank_df = pd.DataFrame(ranking_rows)

    # Filter: must have >1% spike rate in TRAIN to be viable
    rank_df = rank_df[rank_df['spike_train'] > 0.01].copy()

    # Sort by spike_train descending
    rank_df = rank_df.sort_values('spike_train', ascending=False)

    print(f"\n   Showing {len(rank_df)} tickers with >1% TRAIN spike rate")
    print(f"\n   {'Ticker':<8} {'Grp':<5} {'SpkTRN':>7} {'SpkOOS':>7} {'SpkMag':>7} "
          f"{'AnnVol':>7} {'DlyRng':>7} {'$Vol':>10} "
          f"{'ITA_r':>6} {'IWM_r':>6} {'Tchr_r':>7}")
    print(f"   {'─'*8} {'─'*5} {'─'*7} {'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*7} {'─'*10} "
          f"{'─'*6} {'─'*6} {'─'*7}")
    for _, r in rank_df.iterrows():
        print(f"   {r['ticker']:<8} {r['group']:<5} "
              f"{r['spike_train']:>6.1%} {r['spike_oos']:>6.1%} {r['spike_mag']:>6.1%} "
              f"{r['ann_vol']:>6.0f}% {r['daily_range']:>6.2f}% "
              f"${r['dollar_vol']:>9,.0f} "
              f"{r['ita_corr']:>5.2f} {r['iwm_corr']:>5.2f} {r['teacher_corr']:>6.3f}")

    # --- Quick summary ---
    print(f"\n   --- CLUSTER FIT SUMMARY ---")
    print(f"   Minimum viable: spike_train > 5%, spike_oos > 2%, ann_vol > 50%")
    viable = rank_df[(rank_df['spike_train'] > 0.05) &
                     (rank_df['spike_oos'] > 0.02) &
                     (rank_df['ann_vol'] > 50)]
    viable = viable.sort_values('teacher_corr', ascending=False)
    print(f"   {len(viable)} viable candidates:")
    for _, r in viable.iterrows():
        flag = '<-- NEW' if r['group'] == 'NEW' else ''
        print(f"      {r['ticker']:<8} spike={r['spike_train']:.1%}/{r['spike_oos']:.1%}  "
              f"ITA_r={r['ita_corr']:.2f}  IWM_r={r['iwm_corr']:.2f}  "
              f"tchr_r={r['teacher_corr']:.3f}  {flag}")

    print("\n" + "=" * 90)
    print("  ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
