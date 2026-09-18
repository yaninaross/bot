# %% ── RA-MoE Data Loader ─────────────────────────────────────────────────────
#
# Shared data utilities for Regime-Adaptive Mixture-of-Experts model.
# Loads master_inference parquet from GCS, engineers features, computes
# regime descriptors, builds walk-forward splits, and assembles TF datasets.
#
# Usage:  from ra_moe.data_loader import (load_master, build_fold_datasets,
#                                          WALK_FORWARD_FOLDS, ALL_TICKERS)

import os, io, gc, sys, warnings, ctypes
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings('ignore')

try:
    import pandas_ta as ta
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta",
                           "--no-deps", "--quiet"])
    import pandas_ta as ta

import tensorflow as tf
from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONSTANTS
# ==============================================================================
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"
PST = pytz.timezone('America/Los_Angeles')

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
EXCLUDE_FROM_TRADING = {
    'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA',  # context / ETFs
    'BITF', 'MIGI',                               # renamed (use KEEL, BGDE)
    'SMLR',                                        # delisted
}

TIME_STEPS    = 128       # lookback window (minutes)
EMBED_DIM     = 16
ROC_HORIZONS  = [5, 30, 60]

# Target horizons for forward return prediction (minutes)
TARGET_HORIZONS = [15, 30, 60]

# Walk-forward folds (Option B: 4 monthly, expanding from Aug 2024)
DATA_START   = '2024-08-01'
WALK_FORWARD_FOLDS = [
    {'fold': 1, 'train_end': '2026-02-28', 'val_start': '2026-03-01',
     'val_end': '2026-03-31', 'test_start': '2026-04-01', 'test_end': '2026-04-30'},
    {'fold': 2, 'train_end': '2026-03-31', 'val_start': '2026-04-01',
     'val_end': '2026-04-30', 'test_start': '2026-05-01', 'test_end': '2026-05-31'},
    {'fold': 3, 'train_end': '2026-04-30', 'val_start': '2026-05-01',
     'val_end': '2026-05-31', 'test_start': '2026-06-01', 'test_end': '2026-06-30'},
    {'fold': 4, 'train_end': '2026-05-31', 'val_start': '2026-06-01',
     'val_end': '2026-06-30', 'test_start': '2026-07-01', 'test_end': '2026-08-04'},
]

# Per-bar feature columns (asset-level)
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RVOL', 'ATR_14', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

# Context columns (market-wide, appended per bar)
CONTEXT_COLS = []
for _ca in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_ca}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_ca}_ROC_{_h}m')
CONTEXT_COLS.append('Time_Pos')

ALL_FEATURE_COLS = ASSET_COLS + CONTEXT_COLS

# Regime feature columns (for router input)
REGIME_COLS = [
    'regime_rvol_15', 'regime_rvol_60', 'regime_rvol_128',
    'regime_trend_60', 'regime_trend_128',
    'regime_autocorr_60',
    'regime_xsect_disp',
    'regime_vix_level', 'regime_vix_velocity',
    'regime_time_sin', 'regime_time_cos',
    'regime_spy_ret_60',
]

# Scaling
SCALING_CLIP = 10.0

gcs_client = gcs_storage.Client()


# ==============================================================================
# DATA LOADING
# ==============================================================================
_master_cache = {}


def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def load_master(ticker):
    """Load master_inference parquet from GCS. Returns DataFrame with UTC DatetimeIndex."""
    if ticker in _master_cache:
        return _master_cache[ticker]

    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"

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

        for mc in MICRO_COLS:
            if mc not in df.columns:
                df[mc] = 0.0

        df = df[df.index.notna()].sort_index()
        _master_cache[ticker] = df
        return df

    except Exception as e:
        print(f"    load_master({ticker}): FAILED -- {e}")
        return None


def heal_data(df):
    """Resample to 1-min, ffill prices, zero-fill volumes. Never backfill."""
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume'] + MICRO_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def clear_cache():
    """Release all cached DataFrames."""
    _master_cache.clear()
    _release_memory()


# ==============================================================================
# CONTEXT ASSET LOADING
# ==============================================================================
def load_context_assets():
    """Load SPY/VIX/TLT/IWM and compute context ROC features."""
    frames = []
    for ca in CONTEXT_ASSETS:
        df = load_master(ca)
        if df is None:
            raise RuntimeError(f"Cannot load context asset {ca}")
        df = heal_data(df)

        anchor_roc = df['Close'].pct_change(1)
        cols = {f'{ca}_Anchor_ROC': anchor_roc}
        for h in ROC_HORIZONS:
            cols[f'{ca}_ROC_{h}m'] = df['Close'].pct_change(h)

        frame = pd.DataFrame(cols, index=df.index)
        frames.append(frame)

    market_df = pd.concat(frames, axis=1).ffill().fillna(0)
    return market_df


# ==============================================================================
# FEATURE ENGINEERING — PER ASSET
# ==============================================================================
def engineer_features(df, market_df):
    """Compute per-bar features for one ticker. Adds market context columns."""
    daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
        df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c

    df['RSI_14'] = ta.rsi(df['Close'], length=14)

    vol_real = df['Volume'].where(df['Volume'] > 0)
    df['RVOL'] = df['Volume'] / vol_real.rolling(50, min_periods=5).mean().ffill().clip(lower=1e-9)

    real_mask = df['Volume'] > 0
    atr_raw = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_14'] = atr_raw.where(real_mask).ffill().fillna(0)

    for h in ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)

    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)

    df = df.join(market_df, how='left').ffill().fillna(0)

    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ==============================================================================
# REGIME FEATURES — MARKET-WIDE (for router)
# ==============================================================================
def compute_regime_features(market_df, universe_returns=None):
    """
    Compute regime descriptor vector from market-wide data.
    Returns DataFrame with REGIME_COLS, same index as market_df.

    Args:
        market_df: output of load_context_assets() (has SPY/VIX ROCs)
        universe_returns: DataFrame of per-ticker 1-min returns (columns=tickers)
                         for cross-sectional dispersion. Optional.
    """
    rf = pd.DataFrame(index=market_df.index)

    spy_close = None
    spy_master = _master_cache.get('SPY')
    if spy_master is not None:
        spy_close = spy_master['Close'].reindex(market_df.index).ffill()

    # Realized volatility at multiple windows (from SPY returns)
    spy_ret = market_df['SPY_Anchor_ROC'].fillna(0)
    for w, name in [(15, 'regime_rvol_15'), (60, 'regime_rvol_60'),
                     (128, 'regime_rvol_128')]:
        rf[name] = spy_ret.rolling(w, min_periods=max(w // 2, 1)).std().fillna(0)

    # Trend strength: cumulative return / realized vol (Sharpe-like)
    for w, name in [(60, 'regime_trend_60'), (128, 'regime_trend_128')]:
        cum_ret = spy_ret.rolling(w, min_periods=max(w // 2, 1)).sum()
        rvol = rf[f'regime_rvol_{w}'].clip(lower=1e-8)
        rf[name] = (cum_ret / rvol).fillna(0).clip(-10, 10)

    # Return autocorrelation (trailing 60 bars)
    rf['regime_autocorr_60'] = spy_ret.rolling(60, min_periods=30).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0,
        raw=True).fillna(0)

    # Cross-sectional dispersion
    if universe_returns is not None and len(universe_returns.columns) > 1:
        rf['regime_xsect_disp'] = universe_returns.std(axis=1).fillna(0)
    else:
        rf['regime_xsect_disp'] = spy_ret.rolling(60).std().fillna(0)

    # VIX level and velocity
    vix_col = 'VIX_Anchor_ROC'
    if vix_col in market_df.columns:
        vix_cum = market_df[vix_col].cumsum()
        rf['regime_vix_level'] = vix_cum.rolling(60, min_periods=1).mean().fillna(0)
        rf['regime_vix_velocity'] = vix_cum.diff(15).fillna(0)
    else:
        rf['regime_vix_level'] = 0.0
        rf['regime_vix_velocity'] = 0.0

    # Time-of-day cyclical encoding
    try:
        idx_pst = market_df.index.tz_convert(PST)
    except Exception:
        idx_pst = market_df.index.tz_localize('UTC').tz_convert(PST)
    mins_of_day = idx_pst.hour * 60 + idx_pst.minute
    rf['regime_time_sin'] = np.sin(2 * np.pi * mins_of_day / 1440)
    rf['regime_time_cos'] = np.cos(2 * np.pi * mins_of_day / 1440)

    # SPY 60-bar return (simple momentum)
    rf['regime_spy_ret_60'] = spy_ret.rolling(60, min_periods=30).sum().fillna(0)

    rf = rf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return rf[REGIME_COLS]


# ==============================================================================
# TARGET: FORWARD RETURNS
# ==============================================================================
def compute_forward_returns(df, horizons=None):
    """
    Compute forward returns at specified horizons.
    Returns DataFrame with columns 'fwd_ret_{h}' for each horizon h.
    Only computes for valid market-hours bars (Volume > 0).
    """
    if horizons is None:
        horizons = TARGET_HORIZONS

    targets = pd.DataFrame(index=df.index)
    close = df['Close']

    for h in horizons:
        fwd = close.shift(-h)
        targets[f'fwd_ret_{h}'] = (fwd - close) / close.clip(lower=1e-8)

    targets = targets.replace([np.inf, -np.inf], np.nan)
    return targets


# ==============================================================================
# SCALING
# ==============================================================================
def compute_scaling_stats(data_dict, feature_cols):
    """
    Compute median/IQR scaling stats from training data.

    Args:
        data_dict: {ticker: DataFrame} with feature columns
        feature_cols: list of column names to scale

    Returns:
        dict of {col: {'med': float, 'iqr': float}}
    """
    stats = {}
    skip_cols = {'Time_Pos', 'Bar_Is_Real'}

    all_vals = defaultdict(list)
    for ticker, df in data_dict.items():
        for col in feature_cols:
            if col in skip_cols or col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) > 0:
                sample = vals[::max(1, len(vals) // 50000)]
                all_vals[col].append(sample)

    for col, val_list in all_vals.items():
        combined = np.concatenate(val_list)
        med = float(np.median(combined))
        q75 = float(np.percentile(combined, 75))
        q25 = float(np.percentile(combined, 25))
        iqr = max(q75 - q25, 1e-8)
        stats[col] = {'med': med, 'iqr': iqr}

    return stats


def apply_scaling(df, stats, feature_cols):
    """Apply median/IQR scaling. Returns scaled copy."""
    df_s = df.copy()
    skip_cols = {'Time_Pos', 'Bar_Is_Real'}
    for col in feature_cols:
        if col in skip_cols or col not in stats or col not in df_s.columns:
            continue
        df_s[col] = ((df_s[col] - stats[col]['med'])
                     / stats[col]['iqr']).clip(-SCALING_CLIP, SCALING_CLIP)
    return df_s


def apply_regime_scaling(regime_df, stats=None):
    """Scale regime features. If stats=None, computes from the data (train mode)."""
    if stats is None:
        stats = {}
        for col in regime_df.columns:
            vals = regime_df[col].dropna().values
            med = float(np.median(vals)) if len(vals) > 0 else 0.0
            q75 = float(np.percentile(vals, 75)) if len(vals) > 0 else 1.0
            q25 = float(np.percentile(vals, 25)) if len(vals) > 0 else 0.0
            stats[col] = {'med': med, 'iqr': max(q75 - q25, 1e-8)}

    scaled = regime_df.copy()
    for col in regime_df.columns:
        if col in stats:
            scaled[col] = ((scaled[col] - stats[col]['med'])
                           / stats[col]['iqr']).clip(-SCALING_CLIP, SCALING_CLIP)
    return scaled, stats


# ==============================================================================
# VALID ROW FILTER
# ==============================================================================
def get_valid_mask(df):
    """
    Returns boolean mask for rows valid for training/evaluation.
    Valid = market hours (6:00-12:30 PST) AND real bar (volume > 0)
    AND enough future bars for target computation.
    """
    idx_pst = df.index.tz_convert(PST)
    mins = idx_pst.hour * 60 + idx_pst.minute
    market_open = (mins >= 360) & (mins <= 750)  # 6:00 AM - 12:30 PM PST
    real_bar = df['Volume'] > 0 if 'Volume' in df.columns else True
    return market_open & real_bar


# ==============================================================================
# WINDOWED DATASET CONSTRUCTION
# ==============================================================================
def build_windows(feature_df, regime_df, target_df, valid_mask,
                  window_size=TIME_STEPS, asset_id=0):
    """
    Build windowed arrays from aligned DataFrames.

    Returns:
        X_seq:    (N, T, F_asset) — per-bar feature sequences
        X_regime: (N, F_regime)   — regime features at prediction time
        X_id:     (N, 1)          — asset ID
        Y:        (N, n_horizons) — forward returns at each horizon
        dates:    (N,)            — prediction timestamps (for diagnostics)
    """
    feat_vals = feature_df.values.astype(np.float32)
    regime_vals = regime_df.values.astype(np.float32)
    target_vals = target_df.values.astype(np.float32)
    mask = valid_mask.values

    n_total = len(feature_df)
    max_horizon = max(TARGET_HORIZONS)

    indices = []
    for i in range(window_size, n_total - max_horizon):
        if mask[i]:
            indices.append(i)

    if not indices:
        return None

    indices = np.array(indices)
    N = len(indices)

    X_seq = np.zeros((N, window_size, feat_vals.shape[1]), dtype=np.float32)
    X_regime = np.zeros((N, regime_vals.shape[1]), dtype=np.float32)
    X_id = np.full((N, 1), asset_id, dtype=np.int32)
    Y = np.zeros((N, target_vals.shape[1]), dtype=np.float32)

    for j, i in enumerate(indices):
        X_seq[j] = feat_vals[i - window_size:i]
        X_regime[j] = regime_vals[i]
        Y[j] = target_vals[i]

    dates = feature_df.index[indices]

    nan_mask = ~np.isnan(Y).any(axis=1)
    if nan_mask.sum() < len(nan_mask):
        X_seq = X_seq[nan_mask]
        X_regime = X_regime[nan_mask]
        X_id = X_id[nan_mask]
        Y = Y[nan_mask]
        dates = dates[nan_mask]

    return X_seq, X_regime, X_id, Y, dates


# ==============================================================================
# DISCOVER TRADABLE TICKERS
# ==============================================================================
def discover_tickers():
    """List all tradable tickers with master_inference files in GCS."""
    bucket = gcs_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=f'{INFERENCE_DIR}/', delimiter='/'))
    master_files = [b.name for b in blobs if b.name.endswith('_master_inference.parquet')]
    tickers = sorted([f.split('/')[-1].replace('_master_inference.parquet', '')
                      for f in master_files])
    tradable = [t for t in tickers if t not in EXCLUDE_FROM_TRADING]
    return tradable


# ==============================================================================
# FULL PIPELINE: BUILD FOLD DATASETS
# ==============================================================================
def build_fold_datasets(fold_config, tickers=None, max_tickers=None, verbose=True):
    """
    Build train/val/test datasets for one walk-forward fold.

    Args:
        fold_config: dict with keys train_end, val_start, val_end,
                     test_start, test_end
        tickers: list of tickers to use (None = discover all)
        max_tickers: limit for fast debugging
        verbose: print progress

    Returns:
        dict with keys 'train', 'val', 'test', each containing:
            X_seq, X_regime, X_id, Y, dates
        Also: 'scaling_stats', 'regime_stats', 'ticker_index', 'n_features'
    """
    if tickers is None:
        tickers = discover_tickers()
    if max_tickers:
        tickers = tickers[:max_tickers]

    ticker_index = {t: i for i, t in enumerate(tickers)}

    if verbose:
        print(f"  Fold {fold_config.get('fold', '?')}: "
              f"train→{fold_config['train_end']} | "
              f"val {fold_config['val_start']}→{fold_config['val_end']} | "
              f"test {fold_config['test_start']}→{fold_config['test_end']}")
        print(f"  Loading {len(tickers)} tickers...")

    # Date boundaries (UTC)
    data_start = pd.Timestamp(DATA_START, tz='UTC')
    train_end  = pd.Timestamp(fold_config['train_end'], tz='UTC') + pd.Timedelta(days=1)
    val_start  = pd.Timestamp(fold_config['val_start'], tz='UTC')
    val_end    = pd.Timestamp(fold_config['val_end'], tz='UTC') + pd.Timedelta(days=1)
    test_start = pd.Timestamp(fold_config['test_start'], tz='UTC')
    test_end   = pd.Timestamp(fold_config['test_end'], tz='UTC') + pd.Timedelta(days=1)

    # Step 1: Load context
    if verbose:
        print(f"  Loading context assets...")
    market_df = load_context_assets()

    # Step 2: Load & engineer features for all tickers
    all_ticker_data = {}
    universe_returns = {}

    for i, ticker in enumerate(tickers):
        df = load_master(ticker)
        if df is None:
            continue

        df = df[df.index >= data_start - pd.Timedelta(days=7)]
        df = heal_data(df)
        df = engineer_features(df, market_df)

        targets = compute_forward_returns(df)
        valid = get_valid_mask(df)

        universe_returns[ticker] = df['Close'].pct_change(1)

        all_ticker_data[ticker] = {
            'features': df[ALL_FEATURE_COLS].copy(),
            'targets': targets,
            'valid': valid,
        }

        _master_cache.pop(ticker, None)

        if verbose and (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(tickers)} loaded")
            _release_memory()

    if verbose:
        print(f"  {len(all_ticker_data)} tickers loaded successfully")

    # Step 3: Compute regime features
    if verbose:
        print(f"  Computing regime features...")
    univ_ret_df = pd.DataFrame(universe_returns).reindex(market_df.index).fillna(0)
    regime_df = compute_regime_features(market_df, univ_ret_df)

    # Step 4: Compute scaling stats from TRAINING data only
    if verbose:
        print(f"  Computing scaling stats (train period only)...")
    train_data_for_stats = {}
    for ticker, data in all_ticker_data.items():
        train_slice = data['features'][data['features'].index < train_end]
        if len(train_slice) > 0:
            train_data_for_stats[ticker] = train_slice

    scaling_stats = compute_scaling_stats(train_data_for_stats, ALL_FEATURE_COLS)
    del train_data_for_stats

    regime_train = regime_df[regime_df.index < train_end]
    _, regime_stats = apply_regime_scaling(regime_train)

    # Step 5: Apply scaling & build windows per split
    splits = {
        'train': (data_start, train_end),
        'val':   (val_start, val_end),
        'test':  (test_start, test_end),
    }

    result = {}
    for split_name, (s_start, s_end) in splits.items():
        all_X_seq, all_X_regime, all_X_id, all_Y, all_dates = [], [], [], [], []

        for ticker, data in all_ticker_data.items():
            feat = data['features']
            tgt  = data['targets']
            valid = data['valid']

            # Need window_size bars BEFORE the split start for lookback
            lookback_start = s_start - pd.Timedelta(minutes=TIME_STEPS + 10)
            mask = (feat.index >= lookback_start) & (feat.index < s_end)
            feat_slice = feat[mask].copy()
            tgt_slice  = tgt.reindex(feat_slice.index).copy()
            valid_slice = valid.reindex(feat_slice.index, fill_value=False)

            if len(feat_slice) < TIME_STEPS + max(TARGET_HORIZONS) + 1:
                continue

            feat_scaled = apply_scaling(feat_slice, scaling_stats, ALL_FEATURE_COLS)
            regime_slice = regime_df.reindex(feat_slice.index).fillna(0)
            regime_scaled, _ = apply_regime_scaling(regime_slice, regime_stats)

            # Restrict valid mask to actual split period (not lookback)
            in_split = (feat_slice.index >= s_start) & (feat_slice.index < s_end)
            valid_final = valid_slice & in_split

            asset_id = ticker_index.get(ticker, len(tickers))
            arrays = build_windows(feat_scaled, regime_scaled, tgt_slice,
                                   valid_final, TIME_STEPS, asset_id)
            if arrays is None:
                continue

            X_seq, X_regime, X_id, Y, dates = arrays
            all_X_seq.append(X_seq)
            all_X_regime.append(X_regime)
            all_X_id.append(X_id)
            all_Y.append(Y)
            all_dates.append(dates)

        if all_X_seq:
            result[split_name] = {
                'X_seq':    np.concatenate(all_X_seq, axis=0),
                'X_regime': np.concatenate(all_X_regime, axis=0),
                'X_id':     np.concatenate(all_X_id, axis=0),
                'Y':        np.concatenate(all_Y, axis=0),
                'dates':    np.concatenate(all_dates, axis=0) if all_dates else np.array([]),
            }
            if verbose:
                n = result[split_name]['X_seq'].shape[0]
                print(f"    {split_name}: {n:,} samples, "
                      f"X_seq={result[split_name]['X_seq'].shape}, "
                      f"X_regime={result[split_name]['X_regime'].shape}")
        else:
            result[split_name] = None
            if verbose:
                print(f"    {split_name}: NO SAMPLES")

    result['scaling_stats'] = scaling_stats
    result['regime_stats'] = regime_stats
    result['ticker_index'] = ticker_index
    result['n_asset_features'] = len(ALL_FEATURE_COLS)
    result['n_regime_features'] = len(REGIME_COLS)

    _release_memory()
    return result


def make_tf_dataset(split_data, batch_size=256, shuffle=True):
    """
    Convert split arrays into a tf.data.Dataset.

    Returns dataset yielding ((X_seq, X_regime, X_id), Y) tuples.
    """
    if split_data is None:
        return None

    ds = tf.data.Dataset.from_tensor_slices((
        {
            'seq_input':    split_data['X_seq'],
            'regime_input': split_data['X_regime'],
            'asset_id':     split_data['X_id'],
        },
        split_data['Y'],
    ))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(split_data['X_seq']), 100_000),
                        reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
