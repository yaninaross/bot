# %% ── Spike Transformer SHORT L5-1: Data Prep (Fold 6) ─────────────────────────
#
# Fold 6 variant with shorter test window for fold advancement comparison.
# Identical to ra_moe_short_L5_1_data_prep.py except fold dates.
#
# V2 of SHORT data prep with SECTOR-AWARE REGIME FEATURES.
#
# Changes vs ra_moe_short_01_data_prep.py:
#   1. Sector proxy context assets: MARA (crypto), MSTR (BTC/growth), ITA (defense)
#   2. 6 new regime features (22 total vs 16):
#      - regime_crypto_ret_60:   crypto sector health (60-min MARA return)
#      - regime_crypto_ret_128:  crypto sector health (128-min MARA return)
#      - regime_crypto_rvol:     crypto sector volatility
#      - regime_growth_vs_value: IWM-SPY relative strength
#      - regime_defense_rs:      ITA-SPY relative strength
#      - regime_safe_haven:      TLT-SPY flow differential
#   3. Explicit TRAIN_TICKERS list (65 assets with updated master_inference)
#   4. Separate GCS paths: short_L5_1_data/ (won't overwrite existing)
#
# Target: (close - min_forward_low) / close >= 6% (price drops 6%+)
#
# Self-contained Colab cell. No local imports needed.
# GPU NOT required. Colab-ready.
#
# Input:  gs://time-series-bucket-ys/Data/Inference_v2/*_master_inference.parquet
# Output: gs://time-series-bucket-ys/Algos/RA_MoE/short_L5_1_data/fold_6_{train,val,test}.npz
#         gs://time-series-bucket-ys/Algos/RA_MoE/short_L5_1_data/meta.npz

import os, sys, time, gc, ctypes, warnings
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

from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

import matplotlib
try:
    from google.colab import auth as _
    _IN_COLAB = True
except ImportError:
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIG
# ==============================================================================
DATA_DIR  = '/tmp/ra_moe_short_L5_1_data'
os.makedirs(DATA_DIR, exist_ok=True)

GCS_BUCKET      = 'time-series-bucket-ys'
GCS_INFER_DIR   = 'Data/Inference_v2'
GCS_DATA_PREFIX = 'Algos/RA_MoE/short_L5_1_data'

MAX_TICKERS       = None
FOLDS_TO_BUILD    = [6]
SUBSAMPLE_EVERY   = 30
MAX_TRAIN_SAMPLES = 50_000

SEED = 42
W    = 100
PST  = pytz.timezone('America/Los_Angeles')

# -- Walk-forward folds --
DATA_START = '2024-08-01'
WALK_FORWARD_FOLDS = [
    {'fold': 6, 'train_end': '2026-06-23', 'val_start': '2026-06-24',
     'val_end': '2026-07-21', 'test_start': '2026-07-22', 'test_end': '2026-08-10'},
]

# -- Ticker filtering --
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
SECTOR_PROXY_ASSETS = ['MARA', 'MSTR', 'ITA']
EXCLUDE_FROM_TRADING = {
    'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA',
    'BITF', 'MIGI', 'SMLR',
}

TRAIN_TICKERS = [
    'AEHR', 'ALAB', 'APLD', 'ARM', 'ASTS', 'AVAV', 'BBAI', 'BE', 'BHP',
    'BKSY', 'BTBT', 'CAT', 'CCJ', 'CEG', 'CIFR', 'CLSK', 'COP', 'CORZ',
    'CVX', 'DUK', 'EQIX', 'ETN', 'FCEL', 'FCX', 'FLNC', 'GE', 'GEV',
    'HOOD', 'HUT', 'IREN', 'JOBY', 'KEEL', 'KTOS', 'LHX', 'LIN', 'LMT',
    'LUNR', 'MARA', 'MOD', 'MU', 'MVIS', 'NEE', 'NNDM', 'NVDA', 'OKLO',
    'ONDS', 'PL', 'PLTR', 'PLUG', 'RCAT', 'RDW', 'RIOT', 'RKLB', 'RTX',
    'SLS', 'SMCI', 'SMR', 'SOUN', 'SPIR', 'VOYG', 'VST', 'WOLF', 'WULF',
    'XOM',
]

# -- Feature columns (same asset+context as before) --
ROC_HORIZONS = [5, 30, 60]
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RVOL', 'ATR_14', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)
CONTEXT_COLS = []
for _ca in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_ca}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_ca}_ROC_{_h}m')
CONTEXT_COLS.append('Time_Pos')
ALL_FEATURE_COLS = ASSET_COLS + CONTEXT_COLS

# -- Regime features: original 16 + 6 new sector-aware --
REGIME_COLS = [
    'regime_rvol_15', 'regime_rvol_60', 'regime_rvol_128',
    'regime_trend_60', 'regime_trend_128',
    'regime_autocorr_60', 'regime_xsect_disp',
    'regime_vix_level', 'regime_vix_velocity',
    'regime_time_sin', 'regime_time_cos',
    'regime_spy_ret_60',
    'regime_vix_close',
    'regime_spy_tlt_corr',
    'regime_breadth',
    'regime_skew_60',
    # NEW: sector-aware features
    'regime_crypto_ret_60',
    'regime_crypto_ret_128',
    'regime_crypto_rvol',
    'regime_growth_vs_value',
    'regime_defense_rs',
    'regime_safe_haven',
]

TIME_STEPS        = 128
TARGET_HORIZONS   = [15, 30, 60]
SCALING_CLIP      = 10.0

# -- SHORT-specific target config --
DUMP_THRESHOLD        = 0.06
SAMPLE_WEIGHT_CAP     = 10.0
WEIGHT_DENOMINATOR    = 0.02

N_ASSET_FEATURES  = len(ALL_FEATURE_COLS)   # 33
N_REGIME_FEATURES = len(REGIME_COLS)         # 22


# ==============================================================================
# GCS HELPERS
# ==============================================================================
_gcs_client = storage.Client()

def _gcs_bucket():
    return _gcs_client.bucket(GCS_BUCKET)

def upload_to_gcs(local_path, prefix):
    fname = os.path.basename(local_path)
    blob_path = f'{prefix}/{fname}'
    _gcs_bucket().blob(blob_path).upload_from_filename(local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"    -> gs://{GCS_BUCKET}/{blob_path} ({size_mb:.1f} MB)")


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
    if ticker in _master_cache:
        return _master_cache[ticker]
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    uri = f"gs://{GCS_BUCKET}/{GCS_INFER_DIR}/{file_ticker}_master_inference.parquet"
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
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume'] + MICRO_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df

def clear_cache():
    _master_cache.clear()
    _release_memory()


# ==============================================================================
# CONTEXT ASSETS + SECTOR PROXIES
# ==============================================================================
def load_context_assets():
    filter_start = pd.Timestamp(DATA_START, tz='UTC') - pd.Timedelta(days=180)
    frames = []
    vix_close = None
    for ca in CONTEXT_ASSETS:
        df = load_master(ca)
        if df is None:
            raise RuntimeError(f"Cannot load context asset {ca}")
        df = df[df.index >= filter_start]
        df = heal_data(df)
        if ca == 'VIX':
            vix_close = df['Close'].copy()
        cols = {f'{ca}_Anchor_ROC': df['Close'].pct_change(1)}
        for h in ROC_HORIZONS:
            cols[f'{ca}_ROC_{h}m'] = df['Close'].pct_change(h)
        frames.append(pd.DataFrame(cols, index=df.index))

    # Sector proxy assets — loaded for regime features only, not as sequence features
    sector_proxy_data = {}
    for sp in SECTOR_PROXY_ASSETS:
        df = load_master(sp)
        if df is None:
            print(f"    WARNING: sector proxy {sp} not found, will zero-fill")
            continue
        df = df[df.index >= filter_start]
        df = heal_data(df)
        sector_proxy_data[sp] = df['Close'].pct_change(1)

    _master_cache.clear()
    result = pd.concat(frames, axis=1).ffill().fillna(0)
    if vix_close is not None:
        result['VIX_Close'] = vix_close.reindex(result.index).ffill().fillna(0)

    for sp, ret_series in sector_proxy_data.items():
        result[f'{sp}_ret'] = ret_series.reindex(result.index).ffill().fillna(0)

    return result


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def engineer_features(df, market_df):
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
    # Join only the standard context columns (not sector proxy columns)
    ctx_cols = [c for c in market_df.columns
                if not c.endswith('_ret') and c != 'VIX_Close']
    df = df.join(market_df[ctx_cols], how='left').ffill().fillna(0)
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ==============================================================================
# REGIME FEATURES (with sector-aware additions)
# ==============================================================================
def compute_regime_features(market_df, universe_returns=None):
    rf = pd.DataFrame(index=market_df.index)
    spy_ret = market_df['SPY_Anchor_ROC'].fillna(0)

    # --- Original 16 features (unchanged) ---
    for w, name in [(15, 'regime_rvol_15'), (60, 'regime_rvol_60'),
                     (128, 'regime_rvol_128')]:
        rf[name] = spy_ret.rolling(w, min_periods=max(w // 2, 1)).std().fillna(0)
    for w, name in [(60, 'regime_trend_60'), (128, 'regime_trend_128')]:
        cum_ret = spy_ret.rolling(w, min_periods=max(w // 2, 1)).sum()
        rvol = rf[f'regime_rvol_{w}'].clip(lower=1e-8)
        rf[name] = (cum_ret / rvol).fillna(0).clip(-10, 10)

    spy_vals = spy_ret.values.astype(np.float64)
    n = len(spy_vals)
    autocorr = np.full(n, 0.0)
    w_ac = 60
    for i in range(w_ac, n):
        chunk = spy_vals[i - w_ac:i]
        if np.count_nonzero(~np.isnan(chunk)) >= 30:
            x, y = chunk[:-1], chunk[1:]
            mx, my = x.mean(), y.mean()
            sx, sy = x.std(), y.std()
            if sx > 1e-12 and sy > 1e-12:
                autocorr[i] = ((x - mx) * (y - my)).mean() / (sx * sy)
    rf['regime_autocorr_60'] = autocorr

    if universe_returns is not None and len(universe_returns.columns) > 1:
        rf['regime_xsect_disp'] = universe_returns.std(axis=1).fillna(0)
    else:
        rf['regime_xsect_disp'] = spy_ret.rolling(60).std().fillna(0)

    vix_col = 'VIX_Anchor_ROC'
    if vix_col in market_df.columns:
        vix_cum = market_df[vix_col].cumsum()
        rf['regime_vix_level'] = vix_cum.rolling(60, min_periods=1).mean().fillna(0)
        rf['regime_vix_velocity'] = vix_cum.diff(15).fillna(0)
    else:
        rf['regime_vix_level'] = 0.0
        rf['regime_vix_velocity'] = 0.0

    try:
        idx_pst = market_df.index.tz_convert(PST)
    except Exception:
        idx_pst = market_df.index.tz_localize('UTC').tz_convert(PST)
    mins_of_day = idx_pst.hour * 60 + idx_pst.minute
    rf['regime_time_sin'] = np.sin(2 * np.pi * mins_of_day / 1440)
    rf['regime_time_cos'] = np.cos(2 * np.pi * mins_of_day / 1440)

    rf['regime_spy_ret_60'] = spy_ret.rolling(60, min_periods=30).sum().fillna(0)

    if 'VIX_Close' in market_df.columns:
        rf['regime_vix_close'] = market_df['VIX_Close'].ffill().fillna(0)
    else:
        rf['regime_vix_close'] = 0.0

    tlt_ret = (market_df['TLT_Anchor_ROC'].fillna(0)
               if 'TLT_Anchor_ROC' in market_df.columns
               else pd.Series(0.0, index=market_df.index))
    rf['regime_spy_tlt_corr'] = spy_ret.rolling(
        60, min_periods=30).corr(tlt_ret).fillna(0).clip(-1, 1)

    if universe_returns is not None and len(universe_returns.columns) > 1:
        cum_rets_60 = universe_returns.rolling(60, min_periods=30).sum()
        rf['regime_breadth'] = (cum_rets_60 > 0).mean(axis=1).fillna(0.5)
    else:
        rf['regime_breadth'] = 0.5

    rolling_mean = spy_ret.rolling(60, min_periods=30).mean()
    rolling_std = spy_ret.rolling(60, min_periods=30).std().clip(lower=1e-12)
    standardized = (spy_ret - rolling_mean) / rolling_std
    rf['regime_skew_60'] = standardized.pow(3).rolling(
        60, min_periods=30).mean().fillna(0).clip(-5, 5)

    # --- NEW: 6 sector-aware features ---
    # Crypto sector health (MARA as proxy)
    crypto_ret = (market_df['MARA_ret'].fillna(0)
                  if 'MARA_ret' in market_df.columns
                  else pd.Series(0.0, index=market_df.index))
    rf['regime_crypto_ret_60'] = crypto_ret.rolling(
        60, min_periods=30).sum().fillna(0)
    rf['regime_crypto_ret_128'] = crypto_ret.rolling(
        128, min_periods=60).sum().fillna(0)
    rf['regime_crypto_rvol'] = crypto_ret.rolling(
        60, min_periods=30).std().fillna(0)

    # Growth vs value: IWM (small-cap/growth) relative strength vs SPY
    iwm_ret = (market_df['IWM_Anchor_ROC'].fillna(0)
               if 'IWM_Anchor_ROC' in market_df.columns
               else pd.Series(0.0, index=market_df.index))
    rf['regime_growth_vs_value'] = (
        iwm_ret.rolling(60, min_periods=30).sum() -
        spy_ret.rolling(60, min_periods=30).sum()
    ).fillna(0)

    # Defense sector relative strength (ITA as proxy)
    defense_ret = (market_df['ITA_ret'].fillna(0)
                   if 'ITA_ret' in market_df.columns
                   else pd.Series(0.0, index=market_df.index))
    rf['regime_defense_rs'] = (
        defense_ret.rolling(60, min_periods=30).sum() -
        spy_ret.rolling(60, min_periods=30).sum()
    ).fillna(0)

    # Safe haven flow: TLT relative strength vs SPY (bond vs equity)
    rf['regime_safe_haven'] = (
        tlt_ret.rolling(60, min_periods=30).sum() -
        spy_ret.rolling(60, min_periods=30).sum()
    ).fillna(0)

    rf = rf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return rf[REGIME_COLS]


# ==============================================================================
# DUMP TARGETS (SHORT-specific)
# ==============================================================================
def compute_dump_targets(df):
    close = df['Close']
    low = df['Low']
    targets = pd.DataFrame(index=df.index)
    continuous = pd.DataFrame(index=df.index)
    weights = pd.DataFrame(index=df.index)
    for h in TARGET_HORIZONS:
        shifted_low = low.shift(-1)
        fwd_min_low = shifted_low.iloc[::-1].rolling(h, min_periods=1).min().iloc[::-1]
        max_dump = ((close - fwd_min_low) / close.clip(lower=1e-8)).replace(
            [np.inf, -np.inf], np.nan)
        dump = (max_dump >= DUMP_THRESHOLD).astype(float)
        w = (max_dump / WEIGHT_DENOMINATOR).clip(0, SAMPLE_WEIGHT_CAP)
        w = w.where(dump == 1, 1.0)
        targets[f'dump_{h}'] = dump
        continuous[f'max_dump_{h}'] = max_dump
        weights[f'weight_{h}'] = w
    return targets, continuous, weights


# ==============================================================================
# SCALING
# ==============================================================================
def compute_scaling_stats(data_dict, feature_cols):
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
        stats[col] = {'med': med, 'iqr': max(q75 - q25, 1e-8)}
    return stats

def apply_scaling(df, stats, feature_cols):
    df_s = df.copy()
    skip_cols = {'Time_Pos', 'Bar_Is_Real'}
    for col in feature_cols:
        if col in skip_cols or col not in stats or col not in df_s.columns:
            continue
        df_s[col] = ((df_s[col] - stats[col]['med'])
                     / stats[col]['iqr']).clip(-SCALING_CLIP, SCALING_CLIP)
    return df_s

def apply_regime_scaling(regime_df, stats=None):
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
# VALID MASK
# ==============================================================================
def get_valid_mask(df):
    idx_pst = df.index.tz_convert(PST)
    mins = idx_pst.hour * 60 + idx_pst.minute
    market_open = (mins >= 360) & (mins <= 750)
    real_bar = df['Volume'] > 0 if 'Volume' in df.columns else True
    return market_open & real_bar


# ==============================================================================
# WINDOW BUILDING
# ==============================================================================
def build_windows(feature_df, regime_df, target_df, continuous_df, weight_df,
                  valid_mask, window_size=TIME_STEPS, asset_id=0, subsample_every=1):
    feat_vals = feature_df.values.astype(np.float32)
    regime_vals = regime_df.values.astype(np.float32)
    target_vals = target_df.values.astype(np.float32)
    cont_vals = continuous_df.values.astype(np.float32)
    weight_vals = weight_df.values.astype(np.float32)
    mask = valid_mask.values
    n_total = len(feature_df)
    max_horizon = max(TARGET_HORIZONS)
    indices = []
    count = 0
    for i in range(window_size, n_total - max_horizon):
        if mask[i]:
            if count % subsample_every == 0:
                indices.append(i)
            count += 1
    if not indices:
        return None
    indices = np.array(indices)
    N = len(indices)
    X_seq = np.zeros((N, window_size, feat_vals.shape[1]), dtype=np.float32)
    X_regime = np.zeros((N, regime_vals.shape[1]), dtype=np.float32)
    X_id = np.full((N, 1), asset_id, dtype=np.int32)
    Y = np.zeros((N, target_vals.shape[1]), dtype=np.float32)
    Y_cont = np.zeros((N, cont_vals.shape[1]), dtype=np.float32)
    W_sample = np.zeros((N, weight_vals.shape[1]), dtype=np.float32)
    for j, i in enumerate(indices):
        X_seq[j] = feat_vals[i - window_size:i]
        X_regime[j] = regime_vals[i]
        Y[j] = target_vals[i]
        Y_cont[j] = cont_vals[i]
        W_sample[j] = weight_vals[i]
    timestamps = feature_df.index[indices]
    nan_mask = ~np.isnan(Y).any(axis=1) & ~np.isnan(Y_cont).any(axis=1)
    if nan_mask.sum() < len(nan_mask):
        X_seq = X_seq[nan_mask]
        X_regime = X_regime[nan_mask]
        X_id = X_id[nan_mask]
        Y = Y[nan_mask]
        Y_cont = Y_cont[nan_mask]
        W_sample = W_sample[nan_mask]
        timestamps = timestamps[nan_mask]
    return X_seq, X_regime, X_id, Y, Y_cont, W_sample, timestamps


# ==============================================================================
# REGIME LABELING
# ==============================================================================
def assign_regime_labels(regime_df, train_end):
    train_rvol = regime_df.loc[regime_df.index < train_end, 'regime_rvol_128']
    t33 = float(train_rvol.quantile(0.33))
    t67 = float(train_rvol.quantile(0.67))
    rvol = regime_df['regime_rvol_128']
    labels = pd.Series(1, index=regime_df.index, dtype=int)
    labels[rvol <= t33] = 0
    labels[rvol >= t67] = 2
    return labels, ['LOW_VOL', 'NORMAL', 'HIGH_VOL'], (t33, t67)


# ==============================================================================
# MAIN
# ==============================================================================
t_total = time.time()

print(f"{'=' * W}")
print(f"  SPIKE TRANSFORMER SHORT L5-1: DATA PREP (SECTOR-AWARE FiLM) — FOLD 6")
print(f"{'=' * W}")
print(f"  Config:")
print(f"    Tickers:          {len(TRAIN_TICKERS)} (explicit list)")
print(f"    Folds to build:   {FOLDS_TO_BUILD}")
print(f"    Subsample train:  every {SUBSAMPLE_EVERY}th window")
print(f"    Max train samples:{MAX_TRAIN_SAMPLES:,}")
print(f"    Window size:      {TIME_STEPS}")
print(f"    Target horizons:  {TARGET_HORIZONS}")
print(f"    Dump threshold:   {DUMP_THRESHOLD*100:.1f}%")
print(f"    Sample wt cap:    {SAMPLE_WEIGHT_CAP}")
print(f"    Asset features:   {N_ASSET_FEATURES}")
print(f"    Regime features:  {N_REGIME_FEATURES} (was 16, +6 sector-aware)")
print(f"    Sector proxies:   {SECTOR_PROXY_ASSETS}")
print(f"    GCS output:       gs://{GCS_BUCKET}/{GCS_DATA_PREFIX}/")
print(f"{'=' * W}")

# -- 1. Use explicit ticker list --
print(f"\n  Using explicit TRAIN_TICKERS list ({len(TRAIN_TICKERS)} tickers)")
all_tickers = sorted(TRAIN_TICKERS)
print(f"    {all_tickers[:10]}...")
ticker_index = {t: i for i, t in enumerate(all_tickers)}

# -- 2. Load context assets + sector proxies --
print(f"\n  Loading context assets ({', '.join(CONTEXT_ASSETS)}) "
      f"+ sector proxies ({', '.join(SECTOR_PROXY_ASSETS)})...")
t0 = time.time()
market_df = load_context_assets()
sector_cols = [c for c in market_df.columns if c.endswith('_ret')]
print(f"    Market data: {market_df.shape} ({time.time()-t0:.1f}s)")
print(f"    Sector proxy cols: {sector_cols}")

# -- 3. Load all tickers + engineer features --
print(f"\n  Loading and engineering features for {len(all_tickers)} tickers...")
data_start = pd.Timestamp(DATA_START, tz='UTC')
all_ticker_data = {}
universe_returns = {}

for i, ticker in enumerate(all_tickers):
    df = load_master(ticker)
    if df is None:
        continue
    df = df[df.index >= data_start - pd.Timedelta(days=7)]
    df = heal_data(df)
    df = engineer_features(df, market_df)
    targets, continuous, sample_wts = compute_dump_targets(df)
    valid = get_valid_mask(df)
    universe_returns[ticker] = df['Close'].pct_change(1)
    all_ticker_data[ticker] = {
        'features': df[ALL_FEATURE_COLS].astype(np.float32).copy(),
        'targets': targets.astype(np.float32).copy(),
        'continuous': continuous.astype(np.float32).copy(),
        'weights': sample_wts.astype(np.float32).copy(),
        'valid': valid,
    }
    del df, targets, continuous, sample_wts
    _master_cache.pop(ticker, None)
    if (i + 1) % 20 == 0:
        print(f"    {i+1}/{len(all_tickers)} loaded")
        _release_memory()

print(f"  {len(all_ticker_data)} tickers loaded successfully")
loaded_tickers = sorted(all_ticker_data.keys())

# -- 4. Compute regime features (with sector-aware additions) --
print(f"\n  Computing regime features (22 = 16 original + 6 sector-aware)...")
t0 = time.time()
univ_ret_df = pd.DataFrame(universe_returns).reindex(market_df.index).fillna(0)
regime_df = compute_regime_features(market_df, univ_ret_df)
del univ_ret_df
_release_memory()
print(f"    Regime features: {regime_df.shape} ({time.time()-t0:.1f}s)")
print(f"    Columns: {list(regime_df.columns)}")

# -- 5. Build each fold --
for fold_config in WALK_FORWARD_FOLDS:
    fold_id = fold_config['fold']
    if fold_id not in FOLDS_TO_BUILD:
        continue

    print(f"\n{'─' * W}")
    print(f"  FOLD {fold_id}")
    print(f"{'─' * W}")
    print(f"    Train: {DATA_START} → {fold_config['train_end']}")
    print(f"    Val:   {fold_config['val_start']} → {fold_config['val_end']}")
    print(f"    Test:  {fold_config['test_start']} → {fold_config['test_end']}")

    train_end  = pd.Timestamp(fold_config['train_end'], tz='UTC') + pd.Timedelta(days=1)
    val_start  = pd.Timestamp(fold_config['val_start'], tz='UTC')
    val_end    = pd.Timestamp(fold_config['val_end'], tz='UTC') + pd.Timedelta(days=1)
    test_start = pd.Timestamp(fold_config['test_start'], tz='UTC')
    test_end   = pd.Timestamp(fold_config['test_end'], tz='UTC') + pd.Timedelta(days=1)

    print(f"    Computing scaling stats (train period)...")
    train_data_for_stats = {}
    for ticker, data in all_ticker_data.items():
        train_slice = data['features'][data['features'].index < train_end]
        if len(train_slice) > 0:
            train_data_for_stats[ticker] = train_slice
    scaling_stats = compute_scaling_stats(train_data_for_stats, ALL_FEATURE_COLS)
    del train_data_for_stats

    regime_train = regime_df[regime_df.index < train_end]
    _, regime_stats = apply_regime_scaling(regime_train)

    regime_labels, regime_names, regime_thresholds = assign_regime_labels(
        regime_df, train_end)

    splits_config = [
        ('train', data_start, train_end, SUBSAMPLE_EVERY),
        ('val',   val_start, val_end, 1),
        ('test',  test_start, test_end, 1),
    ]

    all_dates = regime_df.index.normalize().unique().sort_values()
    date_to_day = {d: i for i, d in enumerate(all_dates)}

    per_ticker_budget = MAX_TRAIN_SAMPLES // max(len(loaded_tickers), 1)
    is_last_fold = (fold_id == max(FOLDS_TO_BUILD))

    accum = {s: {'X_seq': [], 'X_regime': [], 'X_id': [], 'Y': [],
                 'Y_cont': [], 'W_sample': [],
                 'days': [], 'date_ints': [], 'minutes': [], 'regime_ids': [],
                 'n_tickers': 0} for s, *_ in splits_config}

    print(f"\n    Building windows (per-ticker budget: {per_ticker_budget:,} for train)...")
    t0 = time.time()

    for ti, ticker in enumerate(list(all_ticker_data.keys())):
        if is_last_fold:
            data = all_ticker_data.pop(ticker)
        else:
            data = all_ticker_data[ticker]
        feat = data['features']
        tgt  = data['targets']
        cont = data['continuous']
        wgt  = data['weights']
        valid = data['valid']

        for split_name, s_start, s_end, subsample in splits_config:
            lookback_start = s_start - pd.Timedelta(minutes=TIME_STEPS + 10)
            mask = (feat.index >= lookback_start) & (feat.index < s_end)
            feat_slice = feat[mask].copy()
            tgt_slice  = tgt.reindex(feat_slice.index).copy()
            cont_slice = cont.reindex(feat_slice.index).copy()
            wgt_slice  = wgt.reindex(feat_slice.index).copy()
            valid_slice = valid.reindex(feat_slice.index, fill_value=False)

            if len(feat_slice) < TIME_STEPS + max(TARGET_HORIZONS) + 1:
                continue

            feat_scaled = apply_scaling(feat_slice, scaling_stats, ALL_FEATURE_COLS)
            regime_slice = regime_df.reindex(feat_slice.index).fillna(0)
            regime_scaled, _ = apply_regime_scaling(regime_slice, regime_stats)

            in_split = (feat_slice.index >= s_start) & (feat_slice.index < s_end)
            valid_final = valid_slice & in_split

            asset_id = ticker_index.get(ticker, len(all_tickers))
            arrays = build_windows(feat_scaled, regime_scaled, tgt_slice,
                                   cont_slice, wgt_slice,
                                   valid_final, TIME_STEPS, asset_id,
                                   subsample_every=subsample)
            if arrays is None:
                continue

            X_seq, X_regime, X_id, Y, Y_cont, W_sample, timestamps = arrays

            if split_name == 'train' and len(Y) > per_ticker_budget:
                rng = np.random.default_rng(SEED + asset_id)
                keep = rng.choice(len(Y), per_ticker_budget, replace=False)
                keep.sort()
                X_seq = X_seq[keep]
                X_regime = X_regime[keep]
                X_id = X_id[keep]
                Y = Y[keep]
                Y_cont = Y_cont[keep]
                W_sample = W_sample[keep]
                timestamps = timestamps[keep]

            days = np.array([date_to_day.get(ts.normalize(), -1)
                             for ts in timestamps], dtype=np.int32)
            date_ints = np.array(
                [int(ts.normalize().strftime('%Y%m%d')) for ts in timestamps],
                dtype=np.int32)
            ts_pst = timestamps.tz_convert(PST)
            minutes = np.array(
                [t.hour * 60 + t.minute for t in ts_pst], dtype=np.int32)
            rids = np.array([regime_labels.get(ts, 1)
                             for ts in timestamps], dtype=np.int32)

            a = accum[split_name]
            a['X_seq'].append(X_seq.astype(np.float16))
            a['X_regime'].append(X_regime.astype(np.float16))
            a['X_id'].append(X_id)
            a['Y'].append(Y)
            a['Y_cont'].append(Y_cont)
            a['W_sample'].append(W_sample)
            a['days'].append(days)
            a['date_ints'].append(date_ints)
            a['minutes'].append(minutes)
            a['regime_ids'].append(rids)
            a['n_tickers'] += 1

        del data
        if (ti + 1) % 20 == 0:
            _release_memory()
            print(f"      {ti+1}/{len(loaded_tickers)} tickers processed")

    elapsed = time.time() - t0
    print(f"    Window building done ({elapsed:.0f}s)")

    if not all_ticker_data:
        _release_memory()
        print(f"    Ticker data freed (last fold)")

    for split_name, *_ in splits_config:
        a = accum[split_name]
        if not a['X_seq']:
            print(f"      {split_name}: NO SAMPLES")
            continue

        n_tickers = a['n_tickers']

        def _concat_and_free(key, dtype=None):
            arr = np.concatenate(a[key], axis=0)
            a[key].clear()
            a[key] = None
            _release_memory()
            if dtype is not None and arr.dtype != dtype:
                arr = arr.astype(dtype)
            return arr

        X_seq_all    = _concat_and_free('X_seq', np.float32)
        X_regime_all = _concat_and_free('X_regime', np.float32)
        X_id_all     = _concat_and_free('X_id')
        Y_all        = _concat_and_free('Y')
        Y_cont_all   = _concat_and_free('Y_cont')
        W_all        = _concat_and_free('W_sample')
        days_all      = _concat_and_free('days')
        date_ints_all = _concat_and_free('date_ints')
        minutes_all   = _concat_and_free('minutes')
        regime_all   = _concat_and_free('regime_ids')

        n = len(Y_all)
        dump_rates = Y_all.mean(axis=0)
        print(f"      {split_name}: {n:,} samples from {n_tickers} tickers")
        print(f"        X_seq:    {X_seq_all.shape}")
        print(f"        X_regime: {X_regime_all.shape}")
        print(f"        Y:        {Y_all.shape} (binary dump targets)")
        print(f"        Dump rates: " + ", ".join(
            [f"{h}m={dump_rates[i]*100:.1f}%"
             for i, h in enumerate(TARGET_HORIZONS)]))
        print(f"        Avg max dump (positives): " + ", ".join(
            [f"{h}m={Y_cont_all[Y_all[:,i]==1, i].mean()*100:.2f}%"
             if (Y_all[:,i]==1).sum() > 0 else f"{h}m=N/A"
             for i, h in enumerate(TARGET_HORIZONS)]))

        fname = f'fold_{fold_id}_{split_name}.npz'
        local_path = os.path.join(DATA_DIR, fname)
        np.savez_compressed(local_path,
                            X_seq=X_seq_all,
                            X_regime=X_regime_all,
                            X_id=X_id_all,
                            Y=Y_all,
                            Y_continuous=Y_cont_all,
                            W_sample=W_all,
                            days=days_all,
                            date_ints=date_ints_all,
                            minutes=minutes_all,
                            regime_ids=regime_all)
        upload_to_gcs(local_path, GCS_DATA_PREFIX)

        del X_seq_all, X_regime_all, X_id_all, Y_all, Y_cont_all, W_all
        del days_all, date_ints_all, minutes_all, regime_all
        accum[split_name] = None
        _release_memory()

    regime_schedule = np.array([regime_labels.get(d, 1) for d in all_dates],
                                dtype=np.int32)

    print(f"\n    Regime distribution (train):")
    train_days = [d for d in all_dates if d < train_end]
    for rid, rname in enumerate(regime_names):
        n_r = sum(1 for d in train_days if regime_labels.get(d, -1) == rid)
        print(f"      {rname:>12}: {n_r} days ({n_r/max(len(train_days),1)*100:.0f}%)")

# -- 6. Save metadata --
print(f"\n  Saving metadata...")
meta_path = os.path.join(DATA_DIR, 'meta.npz')
np.savez(meta_path,
         regime_schedule=regime_schedule,
         regime_names=np.array(regime_names),
         regime_thresholds=np.array(regime_thresholds),
         ticker_names=np.array(loaded_tickers),
         ticker_index=np.array(list(ticker_index.items()), dtype=object),
         n_asset_features=N_ASSET_FEATURES,
         n_regime_features=N_REGIME_FEATURES,
         feature_cols=np.array(ALL_FEATURE_COLS),
         regime_cols=np.array(REGIME_COLS),
         scaling_stats=np.array(scaling_stats, dtype=object),
         regime_stats=np.array(regime_stats, dtype=object),
         target_type='dump',
         dump_threshold=DUMP_THRESHOLD)
upload_to_gcs(meta_path, GCS_DATA_PREFIX)


# ==============================================================================
# DIAGNOSTIC CHART
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  DIAGNOSTICS")
print(f"{'=' * W}")

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('Spike Transformer SHORT L5-1 — Data Diagnostics (Sector-Aware FiLM) — Fold 6',
             fontsize=14, fontweight='bold')

# [0,0] Regime features over time
ax = axes[0, 0]
regime_daily = regime_df.resample('D').mean().dropna(how='all')
ax.plot(regime_daily.index, regime_daily['regime_rvol_128'], label='RVol 128', alpha=0.8)
ax.plot(regime_daily.index, regime_daily['regime_trend_128'], label='Trend 128', alpha=0.8)
ax.set_title('Regime Features (daily avg)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# [0,1] VIX proxy
ax = axes[0, 1]
if 'regime_vix_level' in regime_daily.columns:
    ax.plot(regime_daily.index, regime_daily['regime_vix_level'], color='purple', alpha=0.8)
ax.set_title('VIX Level Proxy')
ax.grid(True, alpha=0.3)

# [0,2] NEW: Sector health features
ax = axes[0, 2]
ax.plot(regime_daily.index, regime_daily['regime_crypto_ret_60'],
        label='Crypto Health (60m)', color='orange', alpha=0.8)
ax.plot(regime_daily.index, regime_daily['regime_growth_vs_value'],
        label='Growth vs Value', color='green', alpha=0.8)
ax.plot(regime_daily.index, regime_daily['regime_defense_rs'],
        label='Defense RS', color='red', alpha=0.8)
ax.plot(regime_daily.index, regime_daily['regime_safe_haven'],
        label='Safe Haven', color='blue', alpha=0.8)
ax.set_title('NEW: Sector Health Features')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# [1,0] Regime label distribution
ax = axes[1, 0]
for rid, rname in enumerate(regime_names):
    mask = regime_labels == rid
    daily_mask = regime_labels.resample('D').first().dropna()
    counts = (daily_mask == rid).rolling(20).mean()
    ax.plot(counts.index, counts, label=rname, alpha=0.8)
ax.set_title('Regime Fraction (20-day rolling)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# [1,1] Ticker sample counts
ax = axes[1, 1]
for fold_config in WALK_FORWARD_FOLDS:
    fold_id = fold_config['fold']
    if fold_id not in FOLDS_TO_BUILD:
        continue
    for split in ['train', 'val', 'test']:
        fpath = os.path.join(DATA_DIR, f'fold_{fold_id}_{split}.npz')
        if os.path.exists(fpath):
            data = np.load(fpath)
            ids = data['X_id'].flatten()
            unique, counts = np.unique(ids, return_counts=True)
            ax.bar(unique + (0.25 if split == 'val' else 0.5 if split == 'test' else 0),
                   counts, width=0.25, alpha=0.7, label=f'F{fold_id} {split}')
ax.set_xlabel('Ticker ID')
ax.set_ylabel('Sample Count')
ax.set_title('Samples per Ticker')
ax.legend(fontsize=7)

# [1,2] NEW: Crypto vol overlay
ax = axes[1, 2]
ax.plot(regime_daily.index, regime_daily['regime_crypto_rvol'],
        label='Crypto RVol', color='orange', alpha=0.8)
ax.plot(regime_daily.index, regime_daily['regime_rvol_128'],
        label='SPY RVol 128', color='blue', alpha=0.8)
ax.set_title('Crypto vs SPY Volatility')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = os.path.join(DATA_DIR, 'data_diagnostics.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
upload_to_gcs(chart_path, GCS_DATA_PREFIX)
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# SUMMARY
# ==============================================================================
total_elapsed = time.time() - t_total
print(f"\n{'=' * W}")
print(f"  SHORT L5-1 DATA PREP COMPLETE (SECTOR-AWARE FiLM) — FOLD 6")
print(f"{'=' * W}")
print(f"  Tickers:          {len(loaded_tickers)}")
print(f"  Asset features:   {N_ASSET_FEATURES}")
print(f"  Regime features:  {N_REGIME_FEATURES} (16 original + 6 sector-aware)")
print(f"  New features:     crypto_ret_60, crypto_ret_128, crypto_rvol,")
print(f"                    growth_vs_value, defense_rs, safe_haven")
print(f"  Sector proxies:   {SECTOR_PROXY_ASSETS}")
print(f"  Regime names:     {regime_names}")
print(f"  Folds built:      {FOLDS_TO_BUILD}")
print(f"  Target:           DUMP >= {DUMP_THRESHOLD*100:.0f}% (binary)")
print(f"  Total time:       {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

for fold_id in FOLDS_TO_BUILD:
    for split in ['train', 'val', 'test']:
        fpath = os.path.join(DATA_DIR, f'fold_{fold_id}_{split}.npz')
        if os.path.exists(fpath):
            data = np.load(fpath)
            n = data['X_seq'].shape[0]
            sz = os.path.getsize(fpath) / 1e6
            print(f"    fold_{fold_id}_{split}: {n:,} samples ({sz:.0f} MB)")

print(f"\n  GCS: gs://{GCS_BUCKET}/{GCS_DATA_PREFIX}/")
print(f"  Next: run ra_moe_short_L5_1_train.py")
print(f"{'=' * W}")
