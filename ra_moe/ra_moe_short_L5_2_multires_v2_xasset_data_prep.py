# %% ── Spike Transformer SHORT L5-2 MULTI-RES v2 + X-ASSET: Data Prep ────────────
#
# EXPERIMENT: Multi-Resolution Input v2 + Cross-Asset Lead-Lag Features
#
# Combines the two best SHORT experiment variants:
#   1. Multi-Res v2: dual backbone (1m + 15m), 150k training samples
#   2. X-ASSET: BTC/ETH lead-lag sequence + regime features
#
# Architecture:
#   - backbone_1m:  128 x 47 features at 1-min resolution (33 baseline + 14 BTC/ETH)
#   - backbone_15m: 128 x 12 features at 15-min resolution (~32h lookback)
#   - 26 regime features (22 baseline + 4 lead-lag)
#   - 150k training samples (from v2)
#
# GCS paths: short_L5_2_multires_v2_xasset_data/
#
# Self-contained Colab cell. GPU NOT required. Colab-ready.

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
DATA_DIR  = '/tmp/ra_moe_short_L5_2_multires_v2_xasset_data'
os.makedirs(DATA_DIR, exist_ok=True)

GCS_BUCKET      = os.environ.get("GCS_BUCKET", "")
GCS_INFER_DIR   = 'Data/Inference_v2'
GCS_DATA_PREFIX = 'Algos/RA_MoE/short_L5_2_multires_v2_xasset_data'

MAX_TICKERS       = None
FOLDS_TO_BUILD    = [5]
SUBSAMPLE_EVERY   = 30
MAX_TRAIN_SAMPLES = 150_000

SEED = 42
W    = 100
PST  = pytz.timezone('America/Los_Angeles')

DATA_START = '2024-08-01'
WALK_FORWARD_FOLDS = [
    {'fold': 5, 'train_end': '2026-06-13', 'val_start': '2026-06-16',
     'val_end': '2026-07-15', 'test_start': '2026-07-16', 'test_end': '2026-08-16'},
]

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
SECTOR_PROXY_ASSETS = ['MARA', 'MSTR', 'ITA']

LEAD_LAG_ASSETS = ['BTC-USD', 'ETH-USD']
LEAD_LAG_LABELS = {'BTC-USD': 'BTC', 'ETH-USD': 'ETH'}

EXCLUDE_FROM_TRADING = {
    'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA',
    'BITF', 'MIGI', 'SMLR',
}

TRAIN_TICKERS = [
    'ACHR', 'ACMR', 'AEHR', 'ALAB', 'AMD', 'APLD', 'ARM', 'ASTS', 'AVAV',
    'AXTI', 'BABA', 'BBAI', 'BEAM', 'BE', 'BGDE', 'BHP', 'BKSY', 'BLZE',
    'BNY', 'BTBT', 'BUUU', 'CAT', 'CCJ', 'CEG', 'CIFR', 'CLF', 'CLSK',
    'COIN', 'COP', 'CORZ', 'CRDO', 'CRNG', 'CVX', 'DUK', 'EQIX', 'ETN',
    'FCEL', 'FCX', 'FLNC', 'GD', 'GE', 'GEV', 'HOOD', 'HUT', 'IREN',
    'JOBY', 'KEEL', 'KTOS', 'LHX', 'LIN', 'LMT', 'LUNR', 'MARA', 'MGRT',
    'MNTS', 'MOD', 'MP', 'MRNA', 'MSTR', 'MU', 'MVIS', 'MXL', 'NBIS',
    'NEE', 'NEOV', 'NNDM', 'NNE', 'NOC', 'NTLA', 'NVDA', 'OKLO', 'ONDS',
    'PL', 'PLTR', 'PLUG', 'POWL', 'RCAT', 'RDW', 'RIOT', 'RKLB', 'ROMA',
    'RTX', 'RXT', 'SIF', 'SLS', 'SMCI', 'SMR', 'SNDK', 'SOUN', 'SPIR',
    'TSLA', 'TWST', 'UCTT', 'VOYG', 'VRT', 'VST', 'WOLF', 'WULF', 'XOM',
]

# -- 1-min feature columns: baseline 33 + 14 lead-lag = 47 --
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

LEAD_LAG_COLS = []
for _ll in LEAD_LAG_ASSETS:
    _label = LEAD_LAG_LABELS[_ll]
    LEAD_LAG_COLS.extend([
        f'{_label}_ROC_1m',
        f'{_label}_ROC_5m',
        f'{_label}_ROC_15m',
        f'{_label}_ROC_60m',
        f'{_label}_RVOL',
        f'{_label}_Momentum',
        f'{_label}_Corr_vs_SPY',
    ])

ALL_FEATURE_COLS = ASSET_COLS + CONTEXT_COLS + LEAD_LAG_COLS

# -- 15-min feature columns (NEW: 12 features) --
LOWRES_COLS = [
    'lr_Close_ret', 'lr_High_ret', 'lr_Low_ret', 'lr_Volume_norm',
    'lr_Range', 'lr_RSI_14', 'lr_RVOL', 'lr_ROC_4', 'lr_ROC_8',
    'lr_ROC_16', 'lr_Bar_Is_Real', 'lr_Time_Pos',
]

# -- Regime features: baseline 22 + 4 lead-lag = 26 --
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
    'regime_crypto_ret_60',
    'regime_crypto_ret_128',
    'regime_crypto_rvol',
    'regime_growth_vs_value',
    'regime_defense_rs',
    'regime_safe_haven',
    'regime_btc_ret_60',
    'regime_btc_rvol',
    'regime_eth_btc_spread',
    'regime_crypto_equity_corr',
]

TIME_STEPS        = 128
TIME_STEPS_15M    = 128
LOWRES_PERIOD     = 15
TARGET_HORIZONS   = [15, 30, 60]
SCALING_CLIP      = 10.0

DUMP_THRESHOLD        = 0.06
SAMPLE_WEIGHT_CAP     = 10.0
WEIGHT_DENOMINATOR    = 0.02

N_ASSET_FEATURES  = len(ALL_FEATURE_COLS)   # 47 (33 baseline + 14 lead-lag)
N_LOWRES_FEATURES = len(LOWRES_COLS)         # 12
N_REGIME_FEATURES = len(REGIME_COLS)         # 26 (22 baseline + 4 lead-lag)


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


def load_lead_lag_asset(ticker):
    """Load a 24/7 crypto asset (BTC-USD, ETH-USD) from GCS."""
    file_ticker = ticker.replace('-', '')
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{GCS_BUCKET}/{GCS_INFER_DIR}/{file_ticker}{suffix}.parquet"
        try:
            df = pd.read_parquet(uri)
            break
        except Exception:
            continue
    else:
        for suffix in ['_master_inference', '_1m_full_historical_data']:
            uri = f"gs://{GCS_BUCKET}/{GCS_INFER_DIR}/{ticker}{suffix}.parquet"
            try:
                df = pd.read_parquet(uri)
                break
            except Exception:
                continue
        else:
            print(f"    load_lead_lag_asset({ticker}): NOT FOUND — will zero-fill")
            return None

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
    std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
               'close': 'Close', 'volume': 'Volume'}
    lower_map = {k.lower(): v for k, v in std_map.items()}
    df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                  for c in df.columns]
    df = df[df.index.notna()].sort_index()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    if 'Volume' not in df.columns:
        df['Volume'] = 0.0
    else:
        df['Volume'] = df['Volume'].fillna(0)
    return df


def clear_cache():
    _master_cache.clear()
    _release_memory()


# ==============================================================================
# CONTEXT ASSETS + SECTOR PROXIES + LEAD-LAG ASSETS
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

    spy_ret = result['SPY_Anchor_ROC'].fillna(0)
    for ll_ticker in LEAD_LAG_ASSETS:
        label = LEAD_LAG_LABELS[ll_ticker]
        print(f"    Loading lead-lag asset: {ll_ticker} -> {label}")
        df_ll = load_lead_lag_asset(ll_ticker)
        if df_ll is not None:
            df_ll = df_ll[df_ll.index >= filter_start]
            close = df_ll['Close'].reindex(result.index, method='ffill').fillna(method='ffill')
            volume = df_ll['Volume'].reindex(result.index).fillna(0)
            ret_1m = close.pct_change(1).fillna(0)

            result[f'{label}_ROC_1m'] = ret_1m
            result[f'{label}_ROC_5m'] = close.pct_change(5).fillna(0)
            result[f'{label}_ROC_15m'] = close.pct_change(15).fillna(0)
            result[f'{label}_ROC_60m'] = close.pct_change(60).fillna(0)

            vol_real = volume.where(volume > 0)
            vol_mean = vol_real.rolling(50, min_periods=5).mean().ffill().clip(lower=1e-9)
            result[f'{label}_RVOL'] = (volume / vol_mean).fillna(1.0)

            result[f'{label}_Momentum'] = ret_1m.rolling(30, min_periods=15).sum().fillna(0)

            result[f'{label}_Corr_vs_SPY'] = ret_1m.rolling(
                15, min_periods=10).corr(spy_ret).fillna(0).clip(-1, 1)

            result[f'{label}_ret'] = ret_1m
        else:
            for col in [f'{label}_ROC_1m', f'{label}_ROC_5m', f'{label}_ROC_15m',
                        f'{label}_ROC_60m', f'{label}_RVOL', f'{label}_Momentum',
                        f'{label}_Corr_vs_SPY', f'{label}_ret']:
                result[col] = 0.0

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result


# ==============================================================================
# FEATURE ENGINEERING (1-min, identical to baseline)
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
    ctx_cols = [c for c in market_df.columns
                if not c.endswith('_ret') and c != 'VIX_Close']
    df = df.join(market_df[ctx_cols], how='left').ffill().fillna(0)
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ==============================================================================
# 15-MIN FEATURE ENGINEERING (NEW)
# ==============================================================================
def build_15min_features(df_1m):
    """Resample 1-min OHLCV to 15-min bars and compute features."""
    ohlcv_15 = df_1m[['Open', 'High', 'Low', 'Close', 'Volume']].resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).ffill()

    lr = pd.DataFrame(index=ohlcv_15.index)
    lr['lr_Close_ret'] = ohlcv_15['Close'].pct_change(1)
    lr['lr_High_ret'] = (ohlcv_15['High'] - ohlcv_15['Open']) / ohlcv_15['Open'].clip(lower=1e-8)
    lr['lr_Low_ret'] = (ohlcv_15['Low'] - ohlcv_15['Open']) / ohlcv_15['Open'].clip(lower=1e-8)

    vol_real = ohlcv_15['Volume'].where(ohlcv_15['Volume'] > 0)
    vol_mean = vol_real.rolling(50, min_periods=5).mean().ffill().clip(lower=1e-9)
    lr['lr_Volume_norm'] = (ohlcv_15['Volume'] / vol_mean).fillna(1.0)

    lr['lr_Range'] = (ohlcv_15['High'] - ohlcv_15['Low']) / ohlcv_15['Close'].clip(lower=1e-8)
    lr['lr_RSI_14'] = ta.rsi(ohlcv_15['Close'], length=14)

    lr['lr_RVOL'] = lr['lr_Volume_norm']

    lr['lr_ROC_4'] = ohlcv_15['Close'].pct_change(4)
    lr['lr_ROC_8'] = ohlcv_15['Close'].pct_change(8)
    lr['lr_ROC_16'] = ohlcv_15['Close'].pct_change(16)

    lr['lr_Bar_Is_Real'] = (ohlcv_15['Volume'] > 0).astype(float)

    try:
        idx_pst = ohlcv_15.index.tz_convert(PST)
    except Exception:
        idx_pst = ohlcv_15.index
    mins = idx_pst.hour * 60 + idx_pst.minute
    lr['lr_Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)

    lr = lr.replace([np.inf, -np.inf], np.nan).fillna(0)
    return lr[LOWRES_COLS]


# ==============================================================================
# REGIME FEATURES (baseline 22 + 4 lead-lag = 26)
# ==============================================================================
def compute_regime_features(market_df, universe_returns=None):
    rf = pd.DataFrame(index=market_df.index)
    spy_ret = market_df['SPY_Anchor_ROC'].fillna(0)

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

    crypto_ret = (market_df['MARA_ret'].fillna(0)
                  if 'MARA_ret' in market_df.columns
                  else pd.Series(0.0, index=market_df.index))
    rf['regime_crypto_ret_60'] = crypto_ret.rolling(60, min_periods=30).sum().fillna(0)
    rf['regime_crypto_ret_128'] = crypto_ret.rolling(128, min_periods=60).sum().fillna(0)
    rf['regime_crypto_rvol'] = crypto_ret.rolling(60, min_periods=30).std().fillna(0)

    iwm_ret = (market_df['IWM_Anchor_ROC'].fillna(0)
               if 'IWM_Anchor_ROC' in market_df.columns
               else pd.Series(0.0, index=market_df.index))
    rf['regime_growth_vs_value'] = (
        iwm_ret.rolling(60, min_periods=30).sum() -
        spy_ret.rolling(60, min_periods=30).sum()
    ).fillna(0)

    defense_ret = (market_df['ITA_ret'].fillna(0)
                   if 'ITA_ret' in market_df.columns
                   else pd.Series(0.0, index=market_df.index))
    rf['regime_defense_rs'] = (
        defense_ret.rolling(60, min_periods=30).sum() -
        spy_ret.rolling(60, min_periods=30).sum()
    ).fillna(0)

    rf['regime_safe_haven'] = (
        tlt_ret.rolling(60, min_periods=30).sum() -
        spy_ret.rolling(60, min_periods=30).sum()
    ).fillna(0)

    btc_ret = (market_df['BTC_ret'].fillna(0)
               if 'BTC_ret' in market_df.columns
               else pd.Series(0.0, index=market_df.index))
    eth_ret = (market_df['ETH_ret'].fillna(0)
               if 'ETH_ret' in market_df.columns
               else pd.Series(0.0, index=market_df.index))

    rf['regime_btc_ret_60'] = btc_ret.rolling(
        60, min_periods=30).sum().fillna(0)
    rf['regime_btc_rvol'] = btc_ret.rolling(
        60, min_periods=30).std().fillna(0)
    rf['regime_eth_btc_spread'] = (
        eth_ret.rolling(60, min_periods=30).sum() -
        btc_ret.rolling(60, min_periods=30).sum()
    ).fillna(0)
    rf['regime_crypto_equity_corr'] = btc_ret.rolling(
        60, min_periods=30).corr(spy_ret).fillna(0).clip(-1, 1)

    rf = rf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return rf[REGIME_COLS]


# ==============================================================================
# DUMP TARGETS (identical to baseline)
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
    skip_cols = {'Time_Pos', 'Bar_Is_Real', 'lr_Time_Pos', 'lr_Bar_Is_Real'}
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
    skip_cols = {'Time_Pos', 'Bar_Is_Real', 'lr_Time_Pos', 'lr_Bar_Is_Real'}
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
# WINDOW BUILDING (with 15-min low-res branch)
# ==============================================================================
def build_windows_multires(feature_df, lr_df, regime_df, target_df, continuous_df,
                           weight_df, valid_mask, window_size=TIME_STEPS,
                           lr_window_size=TIME_STEPS_15M, lr_period=LOWRES_PERIOD,
                           asset_id=0, subsample_every=1):
    feat_vals = feature_df.values.astype(np.float32)
    regime_vals = regime_df.values.astype(np.float32)
    target_vals = target_df.values.astype(np.float32)
    cont_vals = continuous_df.values.astype(np.float32)
    weight_vals = weight_df.values.astype(np.float32)
    mask = valid_mask.values
    n_total = len(feature_df)
    max_horizon = max(TARGET_HORIZONS)

    lr_vals = lr_df.values.astype(np.float32)
    lr_index = lr_df.index
    feat_index = feature_df.index

    indices = []
    count = 0
    min_start = max(window_size, lr_window_size * lr_period)
    for i in range(min_start, n_total - max_horizon):
        if mask[i]:
            if count % subsample_every == 0:
                indices.append(i)
            count += 1
    if not indices:
        return None
    indices = np.array(indices)
    N = len(indices)

    n_lr_feats = lr_vals.shape[1]
    X_seq = np.zeros((N, window_size, feat_vals.shape[1]), dtype=np.float32)
    X_seq_15m = np.zeros((N, lr_window_size, n_lr_feats), dtype=np.float32)
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

        current_time = feat_index[i]
        lr_mask_before = lr_index <= current_time
        lr_count = lr_mask_before.sum()
        if lr_count >= lr_window_size:
            lr_end_idx = lr_count
            lr_start_idx = lr_end_idx - lr_window_size
            X_seq_15m[j] = lr_vals[lr_start_idx:lr_end_idx]
        elif lr_count > 0:
            available = lr_vals[:lr_count]
            pad_len = lr_window_size - lr_count
            X_seq_15m[j] = np.concatenate([
                np.zeros((pad_len, n_lr_feats), dtype=np.float32),
                available
            ], axis=0)

    timestamps = feature_df.index[indices]
    nan_mask = ~np.isnan(Y).any(axis=1) & ~np.isnan(Y_cont).any(axis=1)
    if nan_mask.sum() < len(nan_mask):
        X_seq = X_seq[nan_mask]
        X_seq_15m = X_seq_15m[nan_mask]
        X_regime = X_regime[nan_mask]
        X_id = X_id[nan_mask]
        Y = Y[nan_mask]
        Y_cont = Y_cont[nan_mask]
        W_sample = W_sample[nan_mask]
        timestamps = timestamps[nan_mask]
    return X_seq, X_seq_15m, X_regime, X_id, Y, Y_cont, W_sample, timestamps


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
print(f"  SPIKE TRANSFORMER SHORT L5-2 MULTI-RES v2 + X-ASSET: DATA PREP")
print(f"  Dual-Resolution (1m + 15m) + BTC/ETH Lead-Lag — 150k samples")
print(f"{'=' * W}")
print(f"  Config:")
print(f"    Tickers:          {len(TRAIN_TICKERS)}")
print(f"    Folds to build:   {FOLDS_TO_BUILD}")
print(f"    1-min branch:     {TIME_STEPS} steps x {N_ASSET_FEATURES} features (33 base + 14 lead-lag)")
print(f"    15-min branch:    {TIME_STEPS_15M} steps x {N_LOWRES_FEATURES} features (~32h lookback)")
print(f"    Regime features:  {N_REGIME_FEATURES} (22 base + 4 lead-lag)")
print(f"    Lead-lag assets:  {LEAD_LAG_ASSETS}")
print(f"    GCS output:       gs://{GCS_BUCKET}/{GCS_DATA_PREFIX}/")
print(f"{'=' * W}")

all_tickers = sorted(TRAIN_TICKERS)
ticker_index = {t: i for i, t in enumerate(all_tickers)}

print(f"\n  Loading context assets + lead-lag ({', '.join(LEAD_LAG_ASSETS)})...")
t0 = time.time()
market_df = load_context_assets()
print(f"    Market data: {market_df.shape} ({time.time()-t0:.1f}s)")

print(f"\n  Loading tickers and building dual-res features...")
data_start = pd.Timestamp(DATA_START, tz='UTC')
all_ticker_data = {}
universe_returns = {}

_train_end_approx = pd.Timestamp(
    WALK_FORWARD_FOLDS[0]['train_end'], tz='UTC') + pd.Timedelta(days=1)
_scaling_samples_1m = defaultdict(list)
_scaling_samples_15m = defaultdict(list)
_skip_cols_1m = {'Time_Pos', 'Bar_Is_Real'}
_skip_cols_15m = {'lr_Time_Pos', 'lr_Bar_Is_Real'}

for i, ticker in enumerate(all_tickers):
    df = load_master(ticker)
    if df is None:
        continue
    df = df[df.index >= data_start - pd.Timedelta(days=14)]
    df = heal_data(df)

    # 1-min features (baseline)
    df_feat = engineer_features(df, market_df)
    targets, continuous, sample_wts = compute_dump_targets(df_feat)
    valid = get_valid_mask(df_feat)

    # 15-min features (NEW)
    lr_features = build_15min_features(df)

    # Collect float32 scaling samples BEFORE float16 conversion
    feat_f32 = df_feat[ALL_FEATURE_COLS]
    train_mask = feat_f32.index < _train_end_approx
    feat_train = feat_f32[train_mask]
    if len(feat_train) > 0:
        for col in ALL_FEATURE_COLS:
            if col in _skip_cols_1m or col not in feat_train.columns:
                continue
            vals = feat_train[col].dropna().values
            if len(vals) > 0:
                stride = max(1, len(vals) // 5000)
                _scaling_samples_1m[col].append(vals[::stride].astype(np.float32).copy())
    lr_train = lr_features[lr_features.index < _train_end_approx]
    if len(lr_train) > 0:
        for col in LOWRES_COLS:
            if col in _skip_cols_15m or col not in lr_train.columns:
                continue
            vals = lr_train[col].dropna().values
            if len(vals) > 0:
                stride = max(1, len(vals) // 5000)
                _scaling_samples_15m[col].append(vals[::stride].astype(np.float32).copy())

    universe_returns[ticker] = df['Close'].pct_change(1)
    all_ticker_data[ticker] = {
        'features': df_feat[ALL_FEATURE_COLS].clip(-65000, 65000).astype(np.float16).copy(),
        'lr_features': lr_features.clip(-65000, 65000).astype(np.float16).copy(),
        'targets': targets.astype(np.float32).copy(),
        'continuous': continuous.astype(np.float32).copy(),
        'weights': sample_wts.astype(np.float32).copy(),
        'valid': valid,
    }
    del df, df_feat, targets, continuous, sample_wts, lr_features, feat_f32
    _master_cache.pop(ticker, None)
    if (i + 1) % 20 == 0:
        print(f"    {i+1}/{len(all_tickers)} loaded")
        _release_memory()

print(f"  {len(all_ticker_data)} tickers loaded")
loaded_tickers = sorted(all_ticker_data.keys())

print(f"\n  Computing regime features (26 = 22 baseline + 4 lead-lag)...")
t0 = time.time()
univ_ret_df = pd.DataFrame(universe_returns).reindex(market_df.index).fillna(0)
regime_df = compute_regime_features(market_df, univ_ret_df)
del univ_ret_df, market_df
_release_memory()
print(f"    Regime: {regime_df.shape} ({time.time()-t0:.1f}s)")

# -- Build folds --
for fold_config in WALK_FORWARD_FOLDS:
    fold_id = fold_config['fold']
    if fold_id not in FOLDS_TO_BUILD:
        continue

    print(f"\n{'─' * W}")
    print(f"  FOLD {fold_id}")
    print(f"{'─' * W}")

    train_end  = pd.Timestamp(fold_config['train_end'], tz='UTC') + pd.Timedelta(days=1)
    val_start  = pd.Timestamp(fold_config['val_start'], tz='UTC')
    val_end    = pd.Timestamp(fold_config['val_end'], tz='UTC') + pd.Timedelta(days=1)
    test_start = pd.Timestamp(fold_config['test_start'], tz='UTC')
    test_end   = pd.Timestamp(fold_config['test_end'], tz='UTC') + pd.Timedelta(days=1)

    # Scaling stats from pre-collected float32 samples (avoids float16 precision loss)
    scaling_stats_1m = {}
    for col, sample_list in _scaling_samples_1m.items():
        combined = np.concatenate(sample_list)
        med = float(np.median(combined))
        q75 = float(np.percentile(combined, 75))
        q25 = float(np.percentile(combined, 25))
        scaling_stats_1m[col] = {'med': med, 'iqr': max(q75 - q25, 1e-8)}
    print(f"    1-min scaling stats: {len(scaling_stats_1m)} features "
          f"(from {sum(len(s) for s in _scaling_samples_1m.values()):,} float32 samples)")
    del _scaling_samples_1m

    scaling_stats_15m = {}
    for col, sample_list in _scaling_samples_15m.items():
        combined = np.concatenate(sample_list)
        med = float(np.median(combined))
        q75 = float(np.percentile(combined, 75))
        q25 = float(np.percentile(combined, 25))
        scaling_stats_15m[col] = {'med': med, 'iqr': max(q75 - q25, 1e-8)}
    print(f"    15-min scaling stats: {len(scaling_stats_15m)} features")
    del _scaling_samples_15m
    _release_memory()

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

    accum = {s: {'X_seq': [], 'X_seq_15m': [], 'X_regime': [], 'X_id': [],
                 'Y': [], 'Y_cont': [], 'W_sample': [],
                 'days': [], 'date_ints': [], 'minutes': [], 'regime_ids': [],
                 'n_tickers': 0} for s, *_ in splits_config}

    print(f"\n    Building dual-resolution windows...")
    t0 = time.time()

    for ti, ticker in enumerate(list(all_ticker_data.keys())):
        if is_last_fold:
            data = all_ticker_data.pop(ticker)
        else:
            data = all_ticker_data[ticker]
        feat = data['features']
        lr_feat = data['lr_features']
        tgt  = data['targets']
        cont = data['continuous']
        wgt  = data['weights']
        valid = data['valid']

        for split_name, s_start, s_end, subsample in splits_config:
            lookback_start = s_start - pd.Timedelta(minutes=TIME_STEPS_15M * LOWRES_PERIOD + 100)
            mask = (feat.index >= lookback_start) & (feat.index < s_end)
            feat_slice = feat[mask].copy()
            tgt_slice  = tgt.reindex(feat_slice.index).copy()
            cont_slice = cont.reindex(feat_slice.index).copy()
            wgt_slice  = wgt.reindex(feat_slice.index).copy()
            valid_slice = valid.reindex(feat_slice.index, fill_value=False)

            lr_mask = (lr_feat.index >= lookback_start) & (lr_feat.index < s_end)
            lr_slice = lr_feat[lr_mask].copy()

            if len(feat_slice) < TIME_STEPS + max(TARGET_HORIZONS) + 1:
                continue

            # Convert float16→float32 before scaling (matches working v2 script)
            feat_f32 = feat_slice.astype(np.float32)
            lr_f32 = lr_slice.astype(np.float32)

            # Diagnostic: NaN check before/after scaling (train split, all tickers)
            if split_name == 'train':
                n_pre_1m = np.isnan(feat_f32.values).sum() + np.isinf(feat_f32.values).sum()
                n_pre_lr = np.isnan(lr_f32.values).sum() + np.isinf(lr_f32.values).sum()
                if n_pre_1m > 0 or n_pre_lr > 0:
                    print(f"      DIAG {ticker}: PRE-SCALE NaN/inf: 1m={n_pre_1m:,}, 15m={n_pre_lr:,}")

            feat_scaled = apply_scaling(feat_f32, scaling_stats_1m, ALL_FEATURE_COLS)
            lr_scaled = apply_scaling(lr_f32, scaling_stats_15m, LOWRES_COLS)
            del feat_f32, lr_f32

            if split_name == 'train':
                n_post_1m = np.isnan(feat_scaled.values).sum() + np.isinf(feat_scaled.values).sum()
                n_post_lr = np.isnan(lr_scaled.values).sum() + np.isinf(lr_scaled.values).sum()
                if n_post_1m > 0 or n_post_lr > 0:
                    print(f"      DIAG {ticker}: POST-SCALE NaN/inf: 1m={n_post_1m:,}, 15m={n_post_lr:,}")
            regime_slice = regime_df.reindex(feat_slice.index).fillna(0)
            regime_scaled, _ = apply_regime_scaling(regime_slice, regime_stats)

            in_split = (feat_slice.index >= s_start) & (feat_slice.index < s_end)
            valid_final = valid_slice & in_split

            asset_id = ticker_index.get(ticker, len(all_tickers))
            arrays = build_windows_multires(
                feat_scaled, lr_scaled, regime_scaled, tgt_slice,
                cont_slice, wgt_slice, valid_final, TIME_STEPS,
                TIME_STEPS_15M, LOWRES_PERIOD, asset_id, subsample)
            if arrays is None:
                continue

            X_seq, X_seq_15m, X_regime, X_id, Y, Y_cont, W_sample, timestamps = arrays

            if split_name == 'train':
                n_win_1m = np.isnan(X_seq).sum() + np.isinf(X_seq).sum()
                n_win_lr = np.isnan(X_seq_15m).sum() + np.isinf(X_seq_15m).sum()
                if n_win_1m > 0 or n_win_lr > 0:
                    print(f"      DIAG {ticker}: POST-WINDOW NaN/inf: 1m={n_win_1m:,}, 15m={n_win_lr:,} (N={len(X_seq)})")
                elif ti == 0:
                    print(f"      DIAG {ticker}: POST-WINDOW clean (0 NaN/inf)")

            if split_name == 'train' and len(Y) > per_ticker_budget:
                rng = np.random.default_rng(SEED + asset_id)
                keep = rng.choice(len(Y), per_ticker_budget, replace=False)
                keep.sort()
                X_seq = X_seq[keep]
                X_seq_15m = X_seq_15m[keep]
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
            a['X_seq_15m'].append(X_seq_15m.astype(np.float16))
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

        X_seq_all     = _concat_and_free('X_seq', np.float32)
        X_seq_15m_all = _concat_and_free('X_seq_15m', np.float32)
        X_regime_all  = _concat_and_free('X_regime', np.float32)
        X_id_all      = _concat_and_free('X_id')
        Y_all         = _concat_and_free('Y')
        Y_cont_all    = _concat_and_free('Y_cont')
        W_all         = _concat_and_free('W_sample')
        days_all      = _concat_and_free('days')
        date_ints_all = _concat_and_free('date_ints')
        minutes_all   = _concat_and_free('minutes')
        regime_all    = _concat_and_free('regime_ids')

        # NaN guard: replace any residual NaN/inf with 0
        for arr_name, arr in [('X_seq', X_seq_all), ('X_seq_15m', X_seq_15m_all),
                               ('X_regime', X_regime_all)]:
            n_nan = np.isnan(arr).sum() + np.isinf(arr).sum()
            if n_nan > 0:
                print(f"        WARNING: {arr_name} has {n_nan:,} NaN/inf — replacing with 0")
                np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        n = len(Y_all)
        dump_rates = Y_all.mean(axis=0)
        print(f"      {split_name}: {n:,} samples from {n_tickers} tickers")
        print(f"        X_seq:     {X_seq_all.shape} (1-min)")
        print(f"        X_seq_15m: {X_seq_15m_all.shape} (15-min)")
        print(f"        X_regime:  {X_regime_all.shape}")

        fname = f'fold_{fold_id}_{split_name}.npz'
        local_path = os.path.join(DATA_DIR, fname)
        np.savez_compressed(local_path,
                            X_seq=X_seq_all,
                            X_seq_15m=X_seq_15m_all,
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

        del X_seq_all, X_seq_15m_all, X_regime_all, X_id_all
        del Y_all, Y_cont_all, W_all
        del days_all, date_ints_all, minutes_all, regime_all
        accum[split_name] = None
        _release_memory()

    regime_schedule = np.array([regime_labels.get(d, 1) for d in all_dates],
                                dtype=np.int32)

# Save metadata
print(f"\n  Saving metadata...")
meta_path = os.path.join(DATA_DIR, 'meta.npz')
np.savez(meta_path,
         regime_schedule=regime_schedule,
         regime_names=np.array(regime_names),
         regime_thresholds=np.array(regime_thresholds),
         ticker_names=np.array(loaded_tickers),
         ticker_index=np.array(list(ticker_index.items()), dtype=object),
         n_asset_features=N_ASSET_FEATURES,
         n_lowres_features=N_LOWRES_FEATURES,
         n_regime_features=N_REGIME_FEATURES,
         feature_cols=np.array(ALL_FEATURE_COLS),
         lowres_cols=np.array(LOWRES_COLS),
         regime_cols=np.array(REGIME_COLS),
         target_type='dump',
         dump_threshold=DUMP_THRESHOLD)
upload_to_gcs(meta_path, GCS_DATA_PREFIX)


# ==============================================================================
# SUMMARY
# ==============================================================================
total_elapsed = time.time() - t_total
print(f"\n{'=' * W}")
print(f"  SHORT L5-2 MULTI-RES v2 + X-ASSET DATA PREP COMPLETE")
print(f"  Dual-Resolution: 1-min ({TIME_STEPS}x{N_ASSET_FEATURES}) + 15-min ({TIME_STEPS_15M}x{N_LOWRES_FEATURES}), 150k cap")
print(f"{'=' * W}")
print(f"  Tickers:          {len(loaded_tickers)}")
print(f"  1-min features:   {N_ASSET_FEATURES} (33 baseline + 14 BTC/ETH lead-lag)")
print(f"  15-min features:  {N_LOWRES_FEATURES} (Close_ret, Range, RSI, ROC_4/8/16, etc.)")
print(f"  Regime features:  {N_REGIME_FEATURES} (22 baseline + 4 lead-lag)")
print(f"  Lead-lag assets:  {LEAD_LAG_ASSETS}")
print(f"  Total time:       {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

for fold_id in FOLDS_TO_BUILD:
    for split in ['train', 'val', 'test']:
        fpath = os.path.join(DATA_DIR, f'fold_{fold_id}_{split}.npz')
        if os.path.exists(fpath):
            data = np.load(fpath)
            n = data['X_seq'].shape[0]
            sz = os.path.getsize(fpath) / 1e6
            print(f"    fold_{fold_id}_{split}: {n:,} samples ({sz:.0f} MB)")
            if 'X_seq_15m' in data:
                print(f"      X_seq_15m: {data['X_seq_15m'].shape}")

print(f"\n  GCS: gs://{GCS_BUCKET}/{GCS_DATA_PREFIX}/")
print(f"  Next: run ra_moe_short_L5_2_multires_v2_xasset_train.py")
print(f"{'=' * W}")
