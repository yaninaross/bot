"""
analyze_energy_cluster.py — Energy cluster candidate screening (OHLCV-only)
============================================================================
Analyzes energy + AI-power + industrial tickers to assess viability for
EXP-155 (Energy cluster retrain with mixed teachers, EXP-105 style).

Uses OHLCV data only (not master_inference) so we can include new assets
that haven't been through the full TQ->micro->master pipeline yet.

Analyses:
  1. Spike frequency & magnitude (TRAIN / OPT / TEST / LIVE periods)
  2. Volatility profile (ATR%, daily range, tail distribution)
  3. Liquidity (volume, dollar volume, entry-window zero-vol %)
  4. Cluster fit (XLE corr, IWM corr, mean teacher corr)
  5. Intraday seasonality (spike rate by 30-min bucket)
  6. Momentum structure (autocorrelation, mean-reversion)
  7. Multi-threshold spike analysis (6%, 8%, 10%, 15%)
  8. Pairwise correlation heatmap
  9. CANDIDATE RANKING — composite score

Usage:  python analyze_energy_cluster.py
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
try:
    from scipy import stats as scipy_stats
except (ImportError, Exception):
    scipy_stats = None

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

# --- Current EXP-150 teachers ---
CURRENT_TEACHERS = ['VST', 'CEG', 'GEV', 'BE', 'CCJ', 'NEE', 'DUK']

# --- Reference: high-spike teachers for cluster-fit correlation ---
TOP_TEACHERS = ['VST', 'CEG', 'GEV', 'BE', 'CCJ']

# --- ALL candidates to analyze ---
ALL_CANDIDATES = [
    # Current EXP-150 teachers
    'VST', 'CEG', 'GEV', 'BE', 'CCJ', 'NEE', 'DUK',
    # AI power / data center crossovers (FULL pipeline)
    'VRT', 'SMCI', 'GE',
    # Clean energy / hydrogen
    'PLUG', 'ENPH', 'FCEL',
    # Mining / commodities (electrification)
    'FCX', 'CLF', 'MP',
    # Oil & gas
    'XOM', 'CVX', 'COP',
    # Industrials / power infra
    'ETN', 'CAT', 'BHP', 'LIN',
    # Data center / AI infra (partial pipeline)
    'EQIX', 'ALAB',
    # --- Energy expansion pool (2026 high-alpha) ---
    # Unregulated power / IPPs
    'TLN', 'NRG', 'PEG', 'BEPC',
    # Advanced nuclear & fuel cycle
    'BWXT', 'OKLO', 'SMR', 'NXE',
    # Data center grid & power mgmt
    'POWL', 'MOD', 'HUBB',
    # Natural gas feedstocks
    'WMB', 'EQT', 'LNG', 'KMI',
    # High-beta storage & tech derivatives
    'FLNC', 'FSLR', 'NXT', 'TPL',
    # --- Cross-sector volatile teachers (EXP-105 philosophy) ---
    # Crypto crossovers (proven in EXP-105: stable 10%+ spike rates)
    'CIFR', 'IREN', 'HUT', 'MARA', 'KEEL',
    'APLD', 'WULF', 'CLSK', 'RIOT', 'MSTR', 'COIN',
    # Defense crossovers (proven spikers from EXP-103/105)
    'LUNR', 'RKLB', 'RCAT', 'ASTS', 'BBAI',
    'KTOS', 'NNDM', 'MVIS', 'ACHR',
]

CONTEXT_ASSETS = ['XLE', 'VIX', 'TLT', 'IWM', 'SMH']

BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference"

SPIKE_THRESHOLDS = [0.04, 0.06, 0.08, 0.10, 0.15]
DEFAULT_SPIKE = 0.10

DATA_START_DATE = "2022-01-01"
TRAIN_CUTOFF_DATE = "2025-10-01"
OPT_START = "2025-10-02"
OPT_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-03-31"
LIVE_START = "2026-04-01"
LIVE_END = datetime.now(pytz.UTC).strftime("%Y-%m-%d")

OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
PST = pytz.timezone('America/Los_Angeles')

PERIODS = [
    ('TRAIN', DATA_START_DATE, TRAIN_CUTOFF_DATE),
    ('OPT',   OPT_START,       OPT_END),
    ('TEST',  TEST_START,      TEST_END),
    ('LIVE',  LIVE_START,      LIVE_END),
]

client = storage.Client()

# ==============================================================================
# DATA LOADING — OHLCV only (no master_inference required)
# ==============================================================================
def load_ohlcv(ticker):
    """Load OHLCV parquet from GCS. Falls back to master_inference if OHLCV missing."""
    clean = ticker.replace("I:", "").replace("X:", "")
    bucket_obj = client.bucket(BUCKET_NAME)

    ohlcv_path = f"{INFERENCE_DIR}/{clean}_1m_full_historical_data.parquet"
    master_path = f"{INFERENCE_DIR}/{clean}_master_inference.parquet"

    blob = bucket_obj.blob(ohlcv_path)
    source = 'OHLCV'
    if not blob.exists():
        blob = bucket_obj.blob(master_path)
        source = 'master'
        if not blob.exists():
            print(f"   Missing data for {clean}")
            return None, source

    df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    col_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume',
    }
    df.columns = [col_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]

    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    elif 'Timestamp' in df.columns:
        ts_col = df['Timestamp']
        if pd.api.types.is_numeric_dtype(ts_col):
            if ts_col.iloc[0] > 1e12:
                df['Datetime'] = pd.to_datetime(ts_col, unit='ms', utc=True)
            else:
                df['Datetime'] = pd.to_datetime(ts_col, unit='s', utc=True)
        else:
            df['Datetime'] = pd.to_datetime(ts_col, utc=True)
    else:
        num_col = next((c for c in df.columns
                       if c in ('Datetime', 'Date', 'Time', 'Level_0')),
                      df.columns[0])
        df['Datetime'] = pd.to_datetime(df[num_col], utc=True)

    df = df.set_index('Datetime')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df[~df.index.duplicated(keep='first')]
    df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()

    df = df[df.index.notna()].sort_index()
    cutoff = pd.Timestamp(DATA_START_DATE, tz='UTC')
    df = df[df.index >= cutoff]

    keep = [c for c in OHLCV_COLS if c in df.columns]
    df = df[keep]
    return df, source


def heal_ohlcv(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)
    return df


def load_and_heal(ticker):
    result = load_ohlcv(ticker)
    if result[0] is None:
        return None
    df, source = result
    df = heal_ohlcv(df)
    bars = len(df)
    print(f"      {ticker}: {bars:,} bars, "
          f"{df.index.min().date()} -> {df.index.max().date()}  [{source}]")
    return df


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def compute_spike_stats(df, ticker, period_name, start_date, end_date,
                        spike_threshold=DEFAULT_SPIKE):
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
    is_spike = (spike_pct >= spike_threshold)
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
        'ticker': ticker, 'period': period_name, 'threshold': spike_threshold,
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
    if scipy_stats is not None:
        kurtosis = scipy_stats.kurtosis(returns_1m, fisher=True)
        skewness = scipy_stats.skew(returns_1m)
    else:
        m = returns_1m - returns_1m.mean()
        s = returns_1m.std()
        kurtosis = float((m ** 4).mean() / s ** 4 - 3) if s > 0 else 0
        skewness = float((m ** 3).mean() / s ** 3) if s > 0 else 0
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
    entry_window = sub[(mins >= 360) & (mins <= 390)]
    avg_vol_1m = mkt_hours['Volume'].mean()
    med_vol_1m = mkt_hours['Volume'].median()
    dollar_vol = (mkt_hours['Volume'] * mkt_hours['Close']).mean()
    entry_vol = entry_window['Volume'].mean() if not entry_window.empty else 0
    entry_zero_pct = ((entry_window['Volume'] == 0).sum() / len(entry_window) * 100
                      if not entry_window.empty else 100)
    return {
        'ticker': ticker,
        'avg_vol_1m': avg_vol_1m, 'med_vol_1m': med_vol_1m,
        'avg_dollar_vol_1m': dollar_vol,
        'entry_window_vol': entry_vol, 'entry_zero_vol_pct': entry_zero_pct,
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
    is_spike = (future_max > (close_safe * (1 + DEFAULT_SPIKE)))
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
    print("  ENERGY / AI-POWER CLUSTER CANDIDATE SCREENING — EXP-155")
    print(f"  {len(ALL_CANDIDATES)} candidates + {len(CONTEXT_ASSETS)} context assets")
    print(f"  Current EXP-150 teachers: {', '.join(CURRENT_TEACHERS)}")
    print(f"  Data source: OHLCV only (no micro/master required)")
    print(f"  TRAIN: {DATA_START_DATE} -> {TRAIN_CUTOFF_DATE}")
    print(f"  LIVE:  {LIVE_START} -> {LIVE_END}")
    print("=" * 90)

    # ------------------------------------------------------------------
    # PHASE 1: Per-ticker analysis
    # ------------------------------------------------------------------
    spike_rows = []
    multi_thresh_rows = []
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

            for thresh in SPIKE_THRESHOLDS:
                r = compute_spike_stats(df, ticker, 'TRAIN',
                                        DATA_START_DATE, TRAIN_CUTOFF_DATE, thresh)
                if r:
                    multi_thresh_rows.append(r)

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

    def ticker_group(t):
        if t in CURRENT_TEACHERS:
            return 'CUR'
        elif t in ['VRT', 'SMCI', 'GE', 'ALAB', 'EQIX']:
            return 'AI-PW'
        elif t in ['PLUG', 'ENPH', 'FCEL', 'FLNC', 'FSLR']:
            return 'CLEAN'
        elif t in ['XOM', 'CVX', 'COP', 'WMB', 'EQT', 'LNG', 'KMI']:
            return 'OIL'
        elif t in ['FCX', 'CLF', 'MP', 'BHP', 'NXE']:
            return 'MINE'
        elif t in ['ETN', 'CAT', 'LIN', 'POWL', 'MOD', 'HUBB', 'NXT', 'TPL']:
            return 'INDUS'
        elif t in ['TLN', 'NRG', 'PEG', 'BEPC', 'BWXT', 'OKLO', 'SMR']:
            return 'POWER'
        elif t in ['CIFR', 'IREN', 'HUT', 'MARA', 'KEEL',
                    'APLD', 'WULF', 'CLSK', 'RIOT', 'MSTR', 'COIN']:
            return 'CRYPT'
        elif t in ['LUNR', 'RKLB', 'RCAT', 'ASTS', 'BBAI',
                    'KTOS', 'NNDM', 'MVIS', 'ACHR']:
            return 'DEF'
        else:
            return 'OTHER'

    xle_corr_map = {}
    if 'XLE' in daily_closes:
        xle_daily = daily_closes['XLE'].pct_change().dropna()
        for ticker in ALL_CANDIDATES:
            if ticker not in daily_closes:
                continue
            asset_daily = daily_closes[ticker].pct_change().dropna()
            merged = pd.concat([xle_daily, asset_daily], axis=1, keys=['XLE', ticker]).dropna()
            if len(merged) < 20:
                continue
            corr = merged.corr().iloc[0, 1]
            cov = np.cov(merged['XLE'], merged[ticker])
            beta = cov[0, 1] / cov[0, 0] if cov[0, 0] > 0 else 0
            xle_corr_map[ticker] = {'xle_corr': corr, 'xle_beta': beta}

    smh_corr_map = {}
    if 'SMH' in daily_closes:
        smh_daily = daily_closes['SMH'].pct_change().dropna()
        for ticker in ALL_CANDIDATES:
            if ticker not in daily_closes:
                continue
            asset_daily = daily_closes[ticker].pct_change().dropna()
            merged = pd.concat([smh_daily, asset_daily], axis=1, keys=['SMH', ticker]).dropna()
            if len(merged) < 20:
                continue
            corr = merged.corr().iloc[0, 1]
            smh_corr_map[ticker] = {'smh_corr': corr}

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
            iwm_corr_map[ticker] = {'iwm_corr': corr}

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

    # -- 1. SPIKE FREQUENCY (default threshold) --
    print("\n" + "=" * 90)
    print(f"  [1/9] SPIKE FREQUENCY & MAGNITUDE (threshold={DEFAULT_SPIKE:.0%})")
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
            sub['grp'] = sub['ticker'].apply(ticker_group)
            print(f"   {'Ticker':<8} {'Grp':<6} {'Days':>5} {'SpikeRate':>9} {'DaysW/Spk':>10} "
                  f"{'MagMean':>8} {'MagMed':>7} {'MagP90':>7} {'MagMax':>7}")
            print(f"   {'─'*8} {'─'*6} {'─'*5} {'─'*9} {'─'*10} "
                  f"{'─'*8} {'─'*7} {'─'*7} {'─'*7}")
            for _, r in sub.iterrows():
                print(f"   {r['ticker']:<8} {r['grp']:<6} {r['n_days']:>5} "
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
        for src, dst, col in [('TRAIN','OPT','OPT/TRN'), ('OPT','TEST','TST/OPT'),
                              ('TEST','LIVE','LIV/TST')]:
            if src in pivot.columns and dst in pivot.columns:
                pivot[col] = pivot[dst] / pivot[src].replace(0, np.nan)
        if 'TRAIN' in pivot.columns:
            pivot = pivot.sort_values('TRAIN', ascending=False)
        print(pivot.round(4).to_string())

    # -- 2. VOLATILITY --
    print("\n" + "=" * 90)
    print("  [2/9] VOLATILITY PROFILE (Training Period)")
    print("=" * 90)

    if vol_rows:
        vol_df = pd.DataFrame(vol_rows).sort_values('daily_range_pct_mean', ascending=False)
        print(f"\n   {'Ticker':<8} {'Grp':<6} {'AvgPx':>8} {'DlyRng%':>8} {'ATR%':>6} "
              f"{'1mStd%':>7} {'AnnVol%':>8} {'Kurt':>7} {'Skew':>6} {'TailR':>6}")
        print(f"   {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*6} "
              f"{'─'*7} {'─'*8} {'─'*7} {'─'*6} {'─'*6}")
        for _, r in vol_df.iterrows():
            grp = ticker_group(r['ticker'])
            print(f"   {r['ticker']:<8} {grp:<6} {r['avg_price']:>7.1f} "
                  f"{r['daily_range_pct_mean']:>7.2f}% {r['atr_pct_mean']:>5.2f}% "
                  f"{r['ret_1m_std']:>6.3f}% {r['annualized_vol']:>7.1f}% "
                  f"{r['kurtosis']:>6.1f} {r['skewness']:>5.2f} "
                  f"{r['tail_ratio']:>5.2f}")

    # -- 3. LIQUIDITY --
    print("\n" + "=" * 90)
    print("  [3/9] LIQUIDITY (Training Period)")
    print("=" * 90)

    if liq_rows:
        liq_df = pd.DataFrame(liq_rows).sort_values('avg_dollar_vol_1m', ascending=False)
        print(f"\n   {'Ticker':<8} {'Grp':<6} {'AvgVol/m':>10} {'$Vol/m':>12} {'EntV0%':>7}")
        print(f"   {'─'*8} {'─'*6} {'─'*10} {'─'*12} {'─'*7}")
        for _, r in liq_df.iterrows():
            grp = ticker_group(r['ticker'])
            print(f"   {r['ticker']:<8} {grp:<6} {r['avg_vol_1m']:>10,.0f} "
                  f"${r['avg_dollar_vol_1m']:>10,.0f} "
                  f"{r['entry_zero_vol_pct']:>6.1f}%")

    # -- 4. CLUSTER FIT --
    print("\n" + "=" * 90)
    print("  [4/9] CLUSTER-FIT METRICS")
    print(f"        XLE corr = energy sector exposure")
    print(f"        SMH corr = AI/semiconductor demand (power consumption driver)")
    print(f"        IWM corr = small-cap risk-on factor")
    print(f"        Teacher corr = avg 5-min corr with top teachers ({', '.join(TOP_TEACHERS)})")
    print("=" * 90)

    cluster_rows = []
    for ticker in ALL_CANDIDATES:
        row = {'ticker': ticker}
        row['xle_corr'] = xle_corr_map.get(ticker, {}).get('xle_corr', np.nan)
        row['xle_beta'] = xle_corr_map.get(ticker, {}).get('xle_beta', np.nan)
        row['smh_corr'] = smh_corr_map.get(ticker, {}).get('smh_corr', np.nan)
        row['iwm_corr'] = iwm_corr_map.get(ticker, {}).get('iwm_corr', np.nan)
        row['teacher_corr'] = teacher_corr_map.get(ticker, {}).get('mean_teacher_corr', np.nan)
        row['most_similar'] = teacher_corr_map.get(ticker, {}).get('most_similar', '')
        cluster_rows.append(row)

    cluster_df = pd.DataFrame(cluster_rows).set_index('ticker')
    cluster_df = cluster_df.sort_values('teacher_corr', ascending=False)

    print(f"\n   {'Ticker':<8} {'Grp':<6} {'XLE_Corr':>9} {'XLE_Beta':>9} {'SMH_Corr':>9} "
          f"{'IWM_Corr':>9} {'Tchr_Corr':>10} {'MostSim':<8}")
    print(f"   {'─'*8} {'─'*6} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*10} {'─'*8}")
    for ticker, r in cluster_df.iterrows():
        grp = ticker_group(ticker)
        xle = f"{r['xle_corr']:>8.3f}" if not np.isnan(r['xle_corr']) else '     N/A'
        xle_b = f"{r['xle_beta']:>8.2f}" if not np.isnan(r['xle_beta']) else '     N/A'
        smh = f"{r['smh_corr']:>8.3f}" if not np.isnan(r['smh_corr']) else '     N/A'
        iwm = f"{r['iwm_corr']:>8.3f}" if not np.isnan(r['iwm_corr']) else '     N/A'
        tch = f"{r['teacher_corr']:>9.3f}" if not np.isnan(r['teacher_corr']) else '      N/A'
        print(f"   {ticker:<8} {grp:<6} {xle} {xle_b} {smh} "
              f"{iwm} {tch} {r['most_similar']:<8}")

    # -- 5. INTRADAY SEASONALITY --
    print("\n" + "=" * 90)
    print("  [5/9] INTRADAY SPIKE SEASONALITY (current + top new candidates)")
    print("=" * 90)

    season_tickers = CURRENT_TEACHERS[:4] + ['VRT', 'SMCI', 'PLUG', 'FCX',
                                              'MARA', 'IREN', 'LUNR', 'OKLO']
    season_results = {}
    for ticker in season_tickers:
        df = load_and_heal(ticker)
        if df is None:
            continue
        r = compute_intraday_seasonality(df, ticker, DATA_START_DATE, TRAIN_CUTOFF_DATE)
        if r:
            season_results[ticker] = r
        del df
        gc.collect()

    for ticker in season_tickers:
        if ticker not in season_results:
            continue
        buckets = season_results[ticker]['buckets']
        if not buckets:
            continue
        grp = ticker_group(ticker)
        print(f"\n   {ticker} [{grp}] — Spike rate by 30-min bucket (PST):")
        for bucket, rate in sorted(buckets.items()):
            bar = '#' * int(rate * 200)
            entry = ' <- ENTRY' if bucket == '06:00' else ''
            print(f"      {bucket}  {rate:>6.2%}  {bar}{entry}")

    # -- 6. MOMENTUM --
    print("\n" + "=" * 90)
    print("  [6/9] MOMENTUM STRUCTURE (Training Period)")
    print("=" * 90)

    if mom_rows:
        mom_df = pd.DataFrame(mom_rows).sort_values('acf_1m_lag1', ascending=False)
        print(f"\n   {'Ticker':<8} {'Grp':<6} {'ACF(1)':>7} {'ACF(5)':>7} {'ACF(30)':>8} "
              f"{'DlyACF':>7} {'MR%':>7}")
        print(f"   {'─'*8} {'─'*6} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*7}")
        for _, r in mom_df.iterrows():
            grp = ticker_group(r['ticker'])
            print(f"   {r['ticker']:<8} {grp:<6} {r['acf_1m_lag1']:>6.3f} "
                  f"{r['acf_1m_lag5']:>6.3f} {r['acf_1m_lag30']:>7.3f} "
                  f"{r['daily_acf_lag1']:>6.3f} {r['mean_rev_after_spike']:>6.3f}%")

    # -- 7. MULTI-THRESHOLD SPIKE ANALYSIS --
    print("\n" + "=" * 90)
    print("  [7/9] MULTI-THRESHOLD SPIKE ANALYSIS (Training Period)")
    print(f"        Thresholds: {', '.join(f'{t:.0%}' for t in SPIKE_THRESHOLDS)}")
    print("=" * 90)

    if multi_thresh_rows:
        mt_df = pd.DataFrame(multi_thresh_rows)
        pivot_mt = mt_df.pivot_table(
            index='ticker', columns='threshold', values='spike_rate', aggfunc='first')
        pivot_mt.columns = [f'{t:.0%}' for t in pivot_mt.columns]
        pivot_mt = pivot_mt.sort_values(pivot_mt.columns[0], ascending=False)

        pivot_days = mt_df.pivot_table(
            index='ticker', columns='threshold', values='days_with_spike_pct', aggfunc='first')
        pivot_days.columns = [f'{t:.0%}_days' for t in pivot_days.columns]

        print(f"\n   Spike RATE by threshold (% of entry bars with spike):")
        print(f"\n   {'Ticker':<8} {'Grp':<6}", end='')
        for col in pivot_mt.columns:
            print(f" {col:>8}", end='')
        print()
        print(f"   {'─'*8} {'─'*6}", end='')
        for _ in pivot_mt.columns:
            print(f" {'─'*8}", end='')
        print()
        for ticker, r in pivot_mt.iterrows():
            grp = ticker_group(ticker)
            print(f"   {ticker:<8} {grp:<6}", end='')
            for val in r.values:
                print(f" {val:>7.1%}", end='')
            print()

        print(f"\n   Days WITH spike by threshold (% of trading days):")
        pivot_days = pivot_days.loc[pivot_mt.index]
        print(f"\n   {'Ticker':<8} {'Grp':<6}", end='')
        for col in pivot_days.columns:
            label = col.replace('_days', '')
            print(f" {label:>8}", end='')
        print()
        print(f"   {'─'*8} {'─'*6}", end='')
        for _ in pivot_days.columns:
            print(f" {'─'*8}", end='')
        print()
        for ticker, r in pivot_days.iterrows():
            grp = ticker_group(ticker)
            print(f"   {ticker:<8} {grp:<6}", end='')
            for val in r.values:
                print(f" {val:>7.1%}", end='')
            print()

    # -- 8. PAIRWISE CORRELATION HEATMAP --
    print("\n" + "=" * 90)
    print("  [8/9] PAIRWISE CORRELATION (5-min returns, current + top candidates)")
    print("=" * 90)

    corr_tickers = [t for t in CURRENT_TEACHERS + ['VRT', 'SMCI', 'GE', 'PLUG', 'FCX', 'XOM',
                                                    'MARA', 'IREN', 'CIFR', 'LUNR', 'RKLB',
                                                    'OKLO', 'FSLR', 'TLN']
                    if t in fivemin_closes]
    if len(corr_tickers) >= 2:
        prices_5m = pd.DataFrame({t: fivemin_closes[t] for t in corr_tickers})
        returns_5m = prices_5m.pct_change().dropna()
        corr_mat = returns_5m.corr()

        print(f"\n   {'':>8}", end='')
        for t in corr_tickers:
            print(f" {t:>6}", end='')
        print()
        for t1 in corr_tickers:
            grp = ticker_group(t1)
            print(f"   {t1:<6} [{grp[0]}]", end='')
            for t2 in corr_tickers:
                val = corr_mat.loc[t1, t2]
                if t1 == t2:
                    print(f"    ---", end='')
                else:
                    print(f" {val:>5.2f}", end='')
            print()

    # ==================================================================
    # [9/9] CANDIDATE RANKING
    # ==================================================================
    print("\n" + "=" * 90)
    print("  [9/9] CANDIDATE RANKING — EXP-155 Energy Cluster Retrain")
    print("=" * 90)

    ranking_rows = []
    for ticker in ALL_CANDIDATES:
        row = {'ticker': ticker, 'group': ticker_group(ticker)}

        if not spike_df.empty:
            train_match = spike_df[(spike_df['ticker'] == ticker) & (spike_df['period'] == 'TRAIN')]
            if not train_match.empty:
                row['spike_train'] = train_match.iloc[0]['spike_rate']
                row['spike_mag'] = train_match.iloc[0]['spike_mag_mean']
            else:
                row['spike_train'] = 0
                row['spike_mag'] = 0
            oos = spike_df[(spike_df['ticker'] == ticker) &
                           (spike_df['period'].isin(['OPT', 'TEST', 'LIVE']))]
            row['spike_oos'] = oos['spike_rate'].mean() if not oos.empty else 0
        else:
            row['spike_train'] = 0
            row['spike_mag'] = 0
            row['spike_oos'] = 0

        row['xle_corr'] = xle_corr_map.get(ticker, {}).get('xle_corr', 0)
        row['smh_corr'] = smh_corr_map.get(ticker, {}).get('smh_corr', 0)
        row['iwm_corr'] = iwm_corr_map.get(ticker, {}).get('iwm_corr', 0)
        row['teacher_corr'] = teacher_corr_map.get(ticker, {}).get('mean_teacher_corr', 0)

        vol_match = [v for v in vol_rows if v['ticker'] == ticker]
        row['ann_vol'] = vol_match[0]['annualized_vol'] if vol_match else 0
        row['daily_range'] = vol_match[0]['daily_range_pct_mean'] if vol_match else 0

        liq_match = [l for l in liq_rows if l['ticker'] == ticker]
        row['dollar_vol'] = liq_match[0]['avg_dollar_vol_1m'] if liq_match else 0
        row['entry_zero'] = liq_match[0]['entry_zero_vol_pct'] if liq_match else 100

        ranking_rows.append(row)

    rank_df = pd.DataFrame(ranking_rows)
    rank_df = rank_df.sort_values('spike_train', ascending=False)

    print(f"\n   {'Ticker':<8} {'Grp':<6} {'SpkTRN':>7} {'SpkOOS':>7} {'SpkMag':>7} "
          f"{'AnnVol':>7} {'DlyRng':>7} {'$Vol':>10} {'EntV0%':>7} "
          f"{'XLE_r':>6} {'SMH_r':>6} {'Tchr_r':>7}")
    print(f"   {'─'*8} {'─'*6} {'─'*7} {'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*7} {'─'*10} {'─'*7} "
          f"{'─'*6} {'─'*6} {'─'*7}")
    for _, r in rank_df.iterrows():
        print(f"   {r['ticker']:<8} {r['group']:<6} "
              f"{r['spike_train']:>6.1%} {r['spike_oos']:>6.1%} {r['spike_mag']:>6.1%} "
              f"{r['ann_vol']:>6.0f}% {r['daily_range']:>6.2f}% "
              f"${r['dollar_vol']:>9,.0f} {r['entry_zero']:>6.1f}% "
              f"{r['xle_corr']:>5.2f} {r['smh_corr']:>5.2f} {r['teacher_corr']:>6.3f}")

    # --- Recommendations ---
    print(f"\n   --- RECOMMENDATIONS ---")

    print(f"\n   A) High-spike candidates (>3% at 10% threshold):")
    high_spike = rank_df[rank_df['spike_train'] > 0.03].sort_values('spike_train', ascending=False)
    for _, r in high_spike.iterrows():
        cur = ' (current)' if r['ticker'] in CURRENT_TEACHERS else ''
        print(f"      {r['ticker']:<8} [{r['group']:<5}] spike={r['spike_train']:.1%}  "
              f"vol={r['ann_vol']:.0f}%  XLE={r['xle_corr']:.2f}  "
              f"SMH={r['smh_corr']:.2f}{cur}")

    print(f"\n   B) Moderate-spike candidates (1-3% at 10% — may need lower threshold):")
    moderate = rank_df[(rank_df['spike_train'] > 0.01) &
                       (rank_df['spike_train'] <= 0.03)].sort_values('spike_train', ascending=False)
    for _, r in moderate.iterrows():
        cur = ' (current)' if r['ticker'] in CURRENT_TEACHERS else ''
        print(f"      {r['ticker']:<8} [{r['group']:<5}] spike={r['spike_train']:.1%}  "
              f"vol={r['ann_vol']:.0f}%  XLE={r['xle_corr']:.2f}{cur}")

    print(f"\n   C) Low/no spike at 10% (need 4-6% threshold or skip):")
    low = rank_df[rank_df['spike_train'] <= 0.01].sort_values('spike_train', ascending=False)
    for _, r in low.iterrows():
        print(f"      {r['ticker']:<8} [{r['group']:<5}] spike={r['spike_train']:.1%}  "
              f"vol={r['ann_vol']:.0f}%")

    # --- Threshold recommendation ---
    print(f"\n   --- THRESHOLD RECOMMENDATION ---")
    if multi_thresh_rows:
        mt_df = pd.DataFrame(multi_thresh_rows)
        print(f"\n   Tickers with >3% spike rate at each threshold:")
        for thresh in SPIKE_THRESHOLDS:
            sub = mt_df[mt_df['threshold'] == thresh]
            viable_at = sub[sub['spike_rate'] > 0.03]
            tickers_list = ', '.join(viable_at.sort_values('spike_rate', ascending=False)['ticker'].tolist())
            print(f"      {thresh:>4.0%}: {len(viable_at):>2}/{len(sub)} tickers  "
                  f"(avg rate: {sub['spike_rate'].mean():.1%})  [{tickers_list}]")

    # --- Proposed teacher composition (data-driven) ---
    print(f"\n   --- PROPOSED EXP-155 TEACHER COMPOSITION ---")
    print(f"   (EXP-105 formula: ~35% core + ~40% cross-sector volatile + ~25% borderline)")
    print()

    teacher_thresh = 0.03
    scoring_thresh = 0.01
    teacher_candidates = rank_df[rank_df['spike_train'] > teacher_thresh].copy()
    scoring_candidates = rank_df[(rank_df['spike_train'] > scoring_thresh) &
                                 (rank_df['spike_train'] <= teacher_thresh)].copy()
    reject_candidates = rank_df[rank_df['spike_train'] <= scoring_thresh].copy()

    group_order = ['CUR', 'POWER', 'AI-PW', 'CLEAN', 'CRYPT', 'DEF',
                   'OIL', 'MINE', 'INDUS', 'OTHER']
    group_labels = {
        'CUR': 'Energy Core (current EXP-150)',
        'POWER': 'Power / Nuclear expansion',
        'AI-PW': 'AI Power / Data Center',
        'CLEAN': 'Clean Energy / Storage',
        'CRYPT': 'Crypto crossovers (proven in 105)',
        'DEF': 'Defense crossovers (proven in 103/105)',
        'OIL': 'Oil & Gas',
        'MINE': 'Mining / Commodities',
        'INDUS': 'Industrials / Grid',
        'OTHER': 'Other',
    }

    print(f"   TEACHER CANDIDATES (spike_rate > {teacher_thresh:.0%} at 10% threshold):")
    for grp in group_order:
        sub = teacher_candidates[teacher_candidates['group'] == grp]
        if sub.empty:
            continue
        label = group_labels.get(grp, grp)
        tickers = ', '.join(sub.sort_values('spike_train', ascending=False)['ticker'].tolist())
        print(f"      [{grp:<6}] {label}: {tickers}")
    print(f"      Total: {len(teacher_candidates)} candidates")

    print(f"\n   SCORING-ONLY (spike {scoring_thresh:.0%}-{teacher_thresh:.0%} — score but don't train):")
    for grp in group_order:
        sub = scoring_candidates[scoring_candidates['group'] == grp]
        if sub.empty:
            continue
        tickers = ', '.join(sub.sort_values('spike_train', ascending=False)['ticker'].tolist())
        print(f"      [{grp:<6}] {tickers}")

    print(f"\n   DROP / SKIP (spike < {scoring_thresh:.0%} — no value as teacher or scorer):")
    if not reject_candidates.empty:
        tickers = ', '.join(reject_candidates.sort_values('spike_train', ascending=False)['ticker'].tolist())
        print(f"      {tickers}")

    # Composition summary
    if not teacher_candidates.empty:
        group_counts = teacher_candidates['group'].value_counts()
        total_t = len(teacher_candidates)
        print(f"\n   COMPOSITION MIX:")
        for grp in group_order:
            cnt = group_counts.get(grp, 0)
            if cnt > 0:
                pct = cnt / total_t * 100
                print(f"      {grp:<6}: {cnt:>2} teachers ({pct:>4.0f}%)")
        energy_core = group_counts.get('CUR', 0) + group_counts.get('POWER', 0) + group_counts.get('AI-PW', 0) + group_counts.get('CLEAN', 0)
        cross_sector = group_counts.get('CRYPT', 0) + group_counts.get('DEF', 0)
        other_sector = total_t - energy_core - cross_sector
        print(f"\n      Energy-adjacent:  {energy_core} ({energy_core/total_t*100:.0f}%)")
        print(f"      Cross-sector:     {cross_sector} ({cross_sector/total_t*100:.0f}%)")
        if other_sector > 0:
            print(f"      Other:            {other_sector} ({other_sector/total_t*100:.0f}%)")
        print(f"\n      EXP-105 target was ~35% core / ~40% cross / ~25% borderline")

    print("\n" + "=" * 90)
    print("  ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
