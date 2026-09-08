# %% ── ST LONG — A/B Test: MR-v2 (33f) vs MR-v2-ADV (45f) ──────────────────
#
# Pure A/B comparison of baseline Multi-Res v2 F8 model (33 features)
# against the new MR-v2-ADV F8 model (45 features = 33 baseline + 12 advanced micro).
#
# Advanced micro features (12 extra columns):
#   LR_OFI, LR_VPIN, LR_Buy_Ratio, LR_Size_Imbalance,
#   PE, LZC, Tick_Volatility, Shock_Persist,
#   Trade_Size_CV, Trade_Intensity, Quote_Intensity, Spread_CV
#
# Both models use identical architecture (MultiResSpikeTransformer, dual backbone
# 1m + 15m), regime features (22), and 15-min features (12) — only the 1-min
# input width differs (33 vs 45).
#
# NO multi-model consensus/ensemble/agreement — single-model only per family.
# GPU recommended. Colab-ready.

import os, sys, gc, ctypes, subprocess, warnings, random
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

try:
    from numba import jit
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numba", "--quiet"])
    from numba import jit

import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, LayerNormalization
from keras import layers as _kl
import h5py
from google.cloud import storage as gcs_storage

if tf.config.list_physical_devices('GPU'):
    keras.mixed_precision.set_dtype_policy('mixed_float16')

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None

try:
    from google.colab import auth
    _IN_COLAB = True
    try:
        auth.authenticate_user()
    except Exception:
        pass
except ImportError:
    _IN_COLAB = False

import matplotlib
if not _IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME      = "time-series-bucket-ys"
WF_BQ_TABLE      = "ys-time-series.scoring_logs.wf_optimization_results"
INFERENCE_DIR    = "Data/Inference_v2"
PST              = pytz.timezone('America/Los_Angeles')

BACKTEST_START   = '2026-08-10'
BACKTEST_END     = '2026-08-24'

TAIL_DAYS        = 60

SCORE_MINUTE     = 375
ENTRY_MINUTE     = 385
EXIT_TIME_MIN    = 780
SLIPPAGE_PCT     = 0.0015

BASE_PNL_THRESH  = 0.018
MIN_TRADES_PER_MONTH = 2.0

TIME_STEPS       = 128
D_MODEL          = 64
N_HEADS          = 4
D_FF             = 128
N_BLOCKS         = 2
DROPOUT_RATE     = 0.1
FILM_HIDDEN      = 64
N_HORIZONS       = 3
N_REGIME_F       = 22

N_ASSET_F_BASELINE = 33
N_ASSET_F_ADV      = 45

N_BLOCKS_15M     = 1
TIME_STEPS_15M   = 128
N_LOWRES_F       = 12

FOLDS = [8]
HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}
HORIZON_FROM_LABEL = {'15m': 0, '30m': 1, '60m': 2}

FAMILIES = {
    'MR-v2': {
        'gcs_data':   'Algos/RA_MoE/long_L5_1_multires_v2_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_1_multires_v2_models',
        'arch': 'multires',
        'n_asset_f': 33,
        'bq_versions': {
            8: 'spike_transformer_long_L5_1_multires_v2_fold8',
        },
    },
    'MR-v2-ADV': {
        'gcs_data':   'Algos/RA_MoE/long_L5_1_multires_v2_adv_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_1_multires_v2_adv_models',
        'arch': 'multires_adv',
        'n_asset_f': 45,
        'bq_versions': {
            8: 'spike_transformer_long_L5_1_multires_v2_adv_fold8',
        },
    },
}

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR', 'BTCUSD', 'ETHUSD'}

LOWRES_COLS = [
    'lr_Close_ret', 'lr_High_ret', 'lr_Low_ret', 'lr_Volume_norm',
    'lr_Range', 'lr_RSI_14', 'lr_RVOL', 'lr_ROC_4', 'lr_ROC_8',
    'lr_ROC_16', 'lr_Bar_Is_Real', 'lr_Time_Pos',
]

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60]
ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RVOL', 'ATR_14', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_ASSETS_LIST = ['SPY', 'VIX', 'TLT', 'IWM']
SECTOR_PROXY_ASSETS = ['MARA', 'MSTR', 'ITA']

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS_LIST:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS.append('Time_Pos')

ALL_COLS = ASSET_COLS + CONTEXT_COLS

MICRO_COLS_ADVANCED = [
    'LR_OFI', 'LR_VPIN', 'LR_Buy_Ratio', 'LR_Size_Imbalance',
    'PE', 'LZC', 'Tick_Volatility', 'Shock_Persist',
    'Trade_Size_CV', 'Trade_Intensity', 'Quote_Intensity', 'Spread_CV',
]

ALL_COLS_ADV = ALL_COLS + MICRO_COLS_ADVANCED  # 33 + 12 = 45

REGIME_COLS = [
    'regime_rvol_15', 'regime_rvol_60', 'regime_rvol_128',
    'regime_trend_60', 'regime_trend_128', 'regime_autocorr_60',
    'regime_xsect_disp', 'regime_vix_level', 'regime_vix_velocity',
    'regime_time_sin', 'regime_time_cos', 'regime_spy_ret_60',
    'regime_vix_close', 'regime_spy_tlt_corr', 'regime_breadth',
    'regime_skew_60',
    'regime_crypto_ret_60', 'regime_crypto_ret_128', 'regime_crypto_rvol',
    'regime_growth_vs_value', 'regime_defense_rs', 'regime_safe_haven',
]

W = 120
SEED = 42
np.random.seed(SEED)

gcs_client = gcs_storage.Client()

_NEEDED_COLS = {'Open', 'High', 'Low', 'Close', 'Volume',
                'open', 'high', 'low', 'close', 'volume',
                'Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
                'bsize', 'asize', 'Total_Vol',
                'Total_vol', 'Trade_count', 'Volume_delta', 'Delta_pct',
                'Obi', 'Bsize', 'Asize',
                'Datetime', 'Date', 'Time', 'Timestamp', 'Level_0'}


# ==============================================================================
# BQ CONFIG LOADING — per-family, single fold
# ==============================================================================
def load_deploy_configs():
    if bigquery is None:
        raise RuntimeError("BigQuery not available")
    bq_client = bigquery.Client()

    family_configs = {}
    for family_name, fam in FAMILIES.items():
        family_configs[family_name] = {}
        for fold_id, model_version in fam['bq_versions'].items():
            q = f"""
            SELECT ticker, experiment, tpsl_mode, prob_thresh, tp_pct, sl_pct,
                   grace_minutes, dip_pct, max_wait_minutes,
                   val_b_avg_pnl, val_b_win_rate, val_b_count
            FROM `{WF_BQ_TABLE}`
            WHERE model_version = '{model_version}'
              AND survived_a = TRUE
              AND val_b_avg_pnl >= {BASE_PNL_THRESH}
            """
            rows = bq_client.query(q).to_dataframe()
            if len(rows) == 0:
                print(f"    {family_name} F{fold_id}: 0 tickers")
                continue
            best = rows.loc[rows.groupby('ticker')['val_b_avg_pnl'].idxmax()]
            for _, row in best.iterrows():
                tk = row['ticker']
                if tk in EXCLUDE_TICKERS:
                    continue
                exp = row['experiment']
                h_part = exp.replace(f'{model_version}_', '').split('_')[0]
                horizon_idx = HORIZON_FROM_LABEL.get(h_part, 1)
                mode = row['tpsl_mode']
                if pd.isna(mode) or mode == '':
                    if '_GRACE' in exp:
                        mode = 'grace'
                    elif '_LIMIT' in exp:
                        mode = 'limit_entry'
                    else:
                        mode = 'static'
                family_configs[family_name][tk] = {
                    'family': family_name,
                    'fold': fold_id,
                    'horizon': horizon_idx,
                    'prob_thresh': float(row['prob_thresh']),
                    'tp_pct': float(row['tp_pct']),
                    'sl_pct': float(row['sl_pct']),
                    'mode': mode,
                    'grace_minutes': int(row['grace_minutes']) if pd.notna(row['grace_minutes']) else 0,
                    'dip_pct': float(row['dip_pct']) if pd.notna(row['dip_pct']) else 0,
                    'max_wait': int(row['max_wait_minutes']) if pd.notna(row['max_wait_minutes']) else 0,
                    'val_b_avg_pnl': float(row['val_b_avg_pnl']),
                }
            print(f"    {family_name} F{fold_id}: {len(family_configs[family_name])} tickers")

    return family_configs


# ==============================================================================
# DATA LOADING (master_inference parquets from GCS)
# ==============================================================================
_master_cache = {}


def load_master(ticker, tail_days=None):
    cache_key = (ticker, tail_days)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    local_path = f"/tmp/_bt_long_ab_{file_ticker}_master.parquet"
    try:
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
        import pyarrow.parquet as pq
        schema = pq.read_schema(local_path)
        all_cols = [f.name for f in schema]
        use_cols = [c for c in all_cols if c in _NEEDED_COLS]
        if not use_cols:
            use_cols = None
        df = pd.read_parquet(local_path, columns=use_cols)
        os.remove(local_path)
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            num_col = next(
                (c for c in df.columns
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
        if tail_days is not None:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
            df = df[df.index >= cutoff]
            if df.empty:
                return None
        _master_cache[cache_key] = df
        return df
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        return None


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# MARKET CONTEXT & FEATURE ENGINEERING
# ==============================================================================
def build_market_context():
    frames = []
    vix_close = None
    for ca in CONTEXT_ASSETS_LIST:
        df = load_master(ca, tail_days=TAIL_DAYS)
        if df is None:
            raise RuntimeError(f"Cannot load context asset {ca}")
        df = df.sort_index()
        df = heal_data(df)
        if ca == 'VIX':
            vix_close = df['Close'].copy()
        cols = {f'{ca}_Anchor_ROC': df['Close'].pct_change(1)}
        for h in ROC_HORIZONS:
            cols[f'{ca}_ROC_{h}m'] = df['Close'].pct_change(h)
        frames.append(pd.DataFrame(cols, index=df.index))
    result = pd.concat(frames, axis=1).ffill().fillna(0)
    if vix_close is not None:
        result['VIX_Close'] = vix_close.reindex(result.index).ffill().fillna(0)
    for sp in SECTOR_PROXY_ASSETS:
        df = load_master(sp, tail_days=TAIL_DAYS)
        if df is not None:
            df = heal_data(df.sort_index())
            result[f'{sp}_ret'] = df['Close'].pct_change(1).reindex(
                result.index).ffill().fillna(0)
    return result


def engineer_asset(ticker, market_df):
    df = load_master(ticker, tail_days=TAIL_DAYS)
    if df is None:
        return None
    df = df.sort_index()
    df = heal_data(df)
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
    df = df.join(market_df, how='left').ffill().fillna(0)
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def engineer_asset_adv(ticker, market_df):
    """Engineer features for ADV model: baseline 33 features + 12 advanced micro."""
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is None:
        return None

    # Load advanced micro parquet from GCS
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"Data/Inference/Microstructure_Advanced/{file_ticker}_1m_advanced_micro.parquet"
    local_path = f"/tmp/_bt_long_ab_{file_ticker}_adv_micro.parquet"

    try:
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
        df_adv = pd.read_parquet(local_path)
        os.remove(local_path)

        # Ensure datetime index
        df_adv = df_adv.reset_index()
        if isinstance(df_adv.columns, pd.MultiIndex):
            df_adv.columns = [c[0] if isinstance(c, tuple) else c for c in df_adv.columns]
        dt_cols = [c for c in df_adv.columns if pd.api.types.is_datetime64_any_dtype(df_adv[c])]
        if dt_cols:
            df_adv['Datetime'] = pd.to_datetime(df_adv[dt_cols[0]], utc=True)
        else:
            cand = next(
                (c for c in df_adv.columns
                 if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                df_adv.columns[0])
            df_adv['Datetime'] = pd.to_datetime(df_adv[cand], utc=True)
        df_adv = df_adv.set_index('Datetime')
        df_adv = df_adv[~df_adv.index.duplicated(keep='first')].sort_index()

        # Keep only the 12 advanced columns
        avail = [c for c in MICRO_COLS_ADVANCED if c in df_adv.columns]
        if avail:
            df_feat = df_feat.join(df_adv[avail], how='left')

    except Exception as e:
        print(f"      [ADV] Failed to load advanced micro for {ticker}: {e}")

    # Zero-fill any missing advanced columns
    for c in MICRO_COLS_ADVANCED:
        if c not in df_feat.columns:
            df_feat[c] = 0.0
        else:
            df_feat[c] = df_feat[c].fillna(0.0)

    if os.path.exists(local_path):
        try:
            os.remove(local_path)
        except Exception:
            pass

    return df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_15min_features(df_1m):
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
    for i in range(60, n):
        chunk = spy_vals[i - 60:i]
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
    return rf[REGIME_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)


# ==============================================================================
# MODEL ARCHITECTURE (MultiResSpikeTransformer only)
# ==============================================================================
class _SeriesStationarization(_kl.Layer):
    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        std = tf.math.reduce_std(x, axis=1, keepdims=True) + 1e-5
        return (x - mean) / std, tf.concat([mean[:, 0, :], std[:, 0, :]], axis=-1)

class _DeStationaryAttention(_kl.Layer):
    def __init__(self, d_model, n_heads, dropout=0.1, **kw):
        super().__init__(**kw)
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.scale = tf.sqrt(tf.cast(self.d_head, tf.float32))
        self.wq = Dense(d_model); self.wk = Dense(d_model)
        self.wv = Dense(d_model); self.wo = Dense(d_model)
        self.attn_drop = Dropout(dropout)
        self.tau_proj = Dense(n_heads)
        self.sigma_proj = Dense(d_model)
    def call(self, x, stats, training=False):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        H, D = self.n_heads, self.d_head
        Q = tf.transpose(tf.reshape(self.wq(x), (B, T, H, D)), (0, 2, 1, 3))
        K = tf.transpose(tf.reshape(self.wk(x), (B, T, H, D)), (0, 2, 1, 3))
        V = tf.transpose(tf.reshape(self.wv(x), (B, T, H, D)), (0, 2, 1, 3))
        scores = tf.matmul(Q, K, transpose_b=True) / tf.cast(self.scale, Q.dtype)
        tau = tf.reshape(self.tau_proj(stats), (B, H, 1, 1))
        scores = scores + tau
        attn = self.attn_drop(tf.nn.softmax(scores, axis=-1), training=training)
        out = tf.reshape(tf.transpose(tf.matmul(attn, V), (0, 2, 1, 3)),
                         (B, T, self.d_model))
        sigma = tf.nn.softplus(self.sigma_proj(stats))
        return self.wo(out * sigma[:, tf.newaxis, :])

class TransformerBlock(_kl.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kw):
        super().__init__(**kw)
        self.attn = _DeStationaryAttention(d_model, n_heads, dropout, name='destat_attn')
        self.ffn = keras.Sequential([
            Dense(d_ff, activation='gelu'), Dense(d_model), Dropout(dropout)])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.drop = Dropout(dropout)
    def call(self, x, stats, training=False):
        x = x + self.drop(self.attn(self.norm1(x), stats, training=training),
                          training=training)
        return x + self.ffn(self.norm2(x), training=training)

class TransformerBackbone(_kl.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len,
                 dropout=0.1, **kw):
        super().__init__(**kw)
        self.stationarize = _SeriesStationarization(name='stationarize')
        self.input_proj = Dense(d_model)
        self.pos_embed = self.add_weight(
            name='pos_embed', shape=(1, seq_len, d_model),
            initializer='glorot_normal', trainable=True)
        self.input_drop = Dropout(dropout)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout,
                                        name=f'block_{i}')
                       for i in range(n_blocks)]
        self.final_norm = LayerNormalization()
    def call(self, x, training=False):
        x_norm, stats = self.stationarize(x)
        h = self.input_proj(x_norm) + self.pos_embed
        h = self.input_drop(h, training=training)
        for block in self.blocks:
            h = block(h, stats, training=training)
        return self.final_norm(h)

class _AttentionPooling(_kl.Layer):
    def __init__(self, d_model, **kw):
        super().__init__(**kw)
        self.query = self.add_weight(
            name='pool_query', shape=(1, 1, d_model),
            initializer='glorot_normal', trainable=True)
        self.scale = tf.sqrt(tf.cast(d_model, tf.float32))
    def call(self, x, training=False):
        scores = tf.matmul(self.query, x, transpose_b=True) / tf.cast(self.scale, x.dtype)
        return tf.matmul(tf.nn.softmax(scores, axis=-1), x)[:, 0, :]

class _FiLMConditioner(_kl.Layer):
    def __init__(self, d_model, hidden_dim=64, **kw):
        super().__init__(**kw)
        self.mlp = keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(hidden_dim, activation='relu')])
        self.gamma_proj = Dense(d_model)
        self.beta_proj = Dense(d_model)
    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        return (1.0 + self.gamma_proj(h)) * x + self.beta_proj(h)

class MultiResSpikeTransformer(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.backbone_1m = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_BLOCKS,
            TIME_STEPS, DROPOUT_RATE, name='backbone_1m')
        self.pool_1m = _AttentionPooling(D_MODEL, name='pool_1m')
        self.backbone_15m = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_BLOCKS_15M,
            TIME_STEPS_15M, DROPOUT_RATE, name='backbone_15m')
        self.pool_15m = _AttentionPooling(D_MODEL, name='pool_15m')
        self.fusion = Dense(D_MODEL, activation='gelu', name='fusion')
        self.film = _FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
        self.drop = Dropout(DROPOUT_RATE)
        self.head = Dense(N_HORIZONS, activation='sigmoid', name='spike_head')
    def call(self, inputs, training=False):
        h_1m = self.backbone_1m(inputs['seq_input'], training=training)
        h_1m = self.pool_1m(h_1m, training=training)
        h_15m = self.backbone_15m(inputs['seq_15m_input'], training=training)
        h_15m = self.pool_15m(h_15m, training=training)
        h = self.fusion(tf.concat([h_1m, h_15m], axis=-1))
        h = self.film(h, inputs['regime_input'], training=training)
        return self.head(self.drop(h, training=training))


# ==============================================================================
# MODEL LOADING (h5py weight loader)
# ==============================================================================
_model_baseline = None
_model_adv = None
_fold_weights = {}
_current_key = None
_family_meta = {}


def _load_keras3_weights(model, filepath):
    with h5py.File(filepath, 'r') as f:
        _load_layer_recursive(model, f)

def _load_layer_recursive(layer, h5_group):
    if 'vars' in h5_group:
        vg = h5_group['vars']
        own_vars = layer.trainable_weights + layer.non_trainable_weights
        for i, var in enumerate(own_vars):
            key = str(i)
            if key in vg:
                var.assign(np.array(vg[key]))
    for key in sorted(h5_group.keys()):
        if key in ('vars', 'optimizer'):
            continue
        if not isinstance(h5_group[key], h5py.Group):
            continue
        child = getattr(layer, key, None)
        if child is None:
            continue
        if isinstance(child, _kl.Layer):
            _load_layer_recursive(child, h5_group[key])
        elif isinstance(child, (list, tuple)):
            _load_list_recursive(child, h5_group[key])

def _load_list_recursive(items, h5_group):
    used_names = {}
    for item in items:
        if not isinstance(item, _kl.Layer):
            continue
        cname = type(item).__name__
        snake = ''.join(f'_{c.lower()}' if c.isupper() else c
                        for c in cname).lstrip('_')
        if snake in used_names:
            used_names[snake] += 1
            snake = f"{snake}_{used_names[snake]}"
        else:
            used_names[snake] = 0
        if snake in h5_group:
            _load_layer_recursive(item, h5_group[snake])

def _ensure_file(gcs_path, local_path):
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    gcs_client.bucket(BUCKET_NAME).blob(gcs_path).download_to_filename(local_path)
    return local_path

def _get_model_for_family(family):
    if family == 'MR-v2-ADV':
        return _model_adv
    return _model_baseline

def _ensure_fold_loaded(family, fold_id):
    global _current_key
    key = (family, fold_id)
    if _current_key != key:
        model = _get_model_for_family(family)
        _load_keras3_weights(model, _fold_weights[key])
        _current_key = key

def load_models_and_meta(family_configs):
    global _model_baseline, _model_adv, _fold_weights, _current_key, _family_meta

    for family_name, fam in FAMILIES.items():
        local_meta = f'/tmp/ra_moe_long_ab_{family_name}_meta.npz'
        meta_path = _ensure_file(f"{fam['gcs_data']}/meta.npz", local_meta)
        meta = np.load(meta_path, allow_pickle=True)
        meta_dict = {
            'scaling_stats': meta['scaling_stats'].item(),
            'regime_stats': meta['regime_stats'].item(),
        }
        if 'scaling_stats_15m' in meta:
            meta_dict['scaling_stats_15m'] = meta['scaling_stats_15m'].item()
        _family_meta[family_name] = meta_dict
        print(f"   {family_name} meta loaded")

    keras.backend.clear_session()
    gc.collect()

    # Build baseline model (33 features)
    _model_baseline = MultiResSpikeTransformer(name='multires_baseline')
    dummy_seq_33 = tf.zeros((1, TIME_STEPS, N_ASSET_F_BASELINE))
    dummy_15m = tf.zeros((1, TIME_STEPS_15M, N_LOWRES_F))
    dummy_reg = tf.zeros((1, N_REGIME_F))
    _model_baseline({'seq_input': dummy_seq_33, 'seq_15m_input': dummy_15m, 'regime_input': dummy_reg})
    print(f"   MultiResSpikeTransformer BASELINE (33f): {_model_baseline.count_params():,} params")

    # Build ADV model (45 features)
    _model_adv = MultiResSpikeTransformer(name='multires_adv')
    dummy_seq_45 = tf.zeros((1, TIME_STEPS, N_ASSET_F_ADV))
    _model_adv({'seq_input': dummy_seq_45, 'seq_15m_input': dummy_15m, 'regime_input': dummy_reg})
    print(f"   MultiResSpikeTransformer ADV (45f):      {_model_adv.count_params():,} params")

    # Download weights for each family/fold
    for family_name, fam in FAMILIES.items():
        cfgs = family_configs.get(family_name, {})
        for fold_id in fam['bq_versions']:
            local_dir = f'/tmp/ra_moe_long_ab_{family_name}_models/fold_{fold_id}'
            weights_path = os.path.join(local_dir, 'model.weights.h5')
            if not os.path.exists(weights_path):
                os.makedirs(local_dir, exist_ok=True)
                prefix = f"{fam['gcs_models']}/fold_{fold_id}"
                bucket = gcs_client.bucket(BUCKET_NAME)
                for blob in bucket.list_blobs(prefix=prefix):
                    fname = blob.name.split('/')[-1]
                    blob.download_to_filename(os.path.join(local_dir, fname))
            _fold_weights[(family_name, fold_id)] = weights_path
            n_tickers = len(cfgs)
            print(f"   {family_name} F{fold_id} weights cached ({n_tickers} tickers)")

    # Pre-load baseline weights
    first_key = ('MR-v2', 8)
    if first_key in _fold_weights:
        _load_keras3_weights(_model_baseline, _fold_weights[first_key])
        _current_key = first_key


def apply_scaling(df, stats, cols):
    df_s = df.copy()
    for col in cols:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# NUMBA TRADE SIMULATION — LONG DIRECTION
# ==============================================================================
@jit(nopython=True)
def run_trade_sim(entry_indices, prices, highs, lows, date_ints, time_mins,
                  tp, sl, exit_time):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        ep = prices[idx]; tgt = ep * (1 + tp); stp = ep * (1 - sl)
        oc = 0.0; xp = ep; xt = 0; done = False; xi = idx
        for k in range(idx + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (xp - ep) / ep; xi = k; done = True; break
            sl_h = lows[k] <= stp; tp_h = highs[k] >= tgt
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (xp - ep) / ep; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (xp - ep) / ep; xi = np_
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = ep; exit_a[tc] = xp; out_a[tc] = xt
        tc += 1; ltd = ed; nxt = cp + 1
        while nxt < ne:
            if entry_indices[nxt] > xi: break
            nxt += 1
        cp = nxt
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]

@jit(nopython=True)
def run_trade_sim_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
                        tp, sl, exit_time, grace_minutes):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        ep = prices[idx]; tgt = ep * (1 + tp); stp = ep * (1 - sl)
        et = time_mins[idx]; oc = 0.0; xp = ep; xt = 0; done = False; xi = idx
        for k in range(idx + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (xp - ep) / ep; xi = k; done = True; break
            elapsed = time_mins[k] - et; sl_act = elapsed >= grace_minutes
            tp_h = highs[k] >= tgt; sl_h = sl_act and (lows[k] <= stp)
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (xp - ep) / ep; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (xp - ep) / ep; xi = np_
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = ep; exit_a[tc] = xp; out_a[tc] = xt
        tc += 1; ltd = ed; nxt = cp + 1
        while nxt < ne:
            if entry_indices[nxt] > xi: break
            nxt += 1
        cp = nxt
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]

@jit(nopython=True)
def run_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints, time_mins,
                              tp, sl, exit_time, dip_pct, max_wait_minutes):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        sp = prices[idx]; lp = sp * (1.0 - dip_pct); st = time_mins[idx]
        fi = -1; fp = 0.0
        for k in range(idx + 1, np_):
            if date_ints[k] != ed: break
            if time_mins[k] - st > max_wait_minutes: break
            if lows[k] <= lp: fp = lp; fi = k; break
        if fi < 0: cp += 1; ltd = ed; continue
        tgt = fp * (1 + tp); stp = fp * (1 - sl)
        oc = 0.0; xp = fp; xt = 0; done = False; xi = fi
        for k in range(fi + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (xp - fp) / fp; xi = k; done = True; break
            sl_h = lows[k] <= stp; tp_h = highs[k] >= tgt
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (xp - fp) / fp; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (xp - fp) / fp; xi = np_
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = fp; exit_a[tc] = xp; out_a[tc] = xt
        tc += 1; ltd = ed; nxt = cp + 1
        while nxt < ne:
            if entry_indices[nxt] > xi: break
            nxt += 1
        cp = nxt
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]

@jit(nopython=True)
def run_trade_sim_limit_entry_with_nofills(entry_indices, prices, highs, lows, date_ints, time_mins,
                                           tp, sl, exit_time, dip_pct, max_wait_minutes):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); filled_a = np.empty(ne, np.int64)
    tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        sp = prices[idx]; lp = sp * (1.0 - dip_pct); st = time_mins[idx]
        fi = -1; fp = 0.0
        for k in range(idx + 1, np_):
            if date_ints[k] != ed: break
            if time_mins[k] - st > max_wait_minutes: break
            if lows[k] <= lp: fp = lp; fi = k; break
        if fi < 0:
            pnl_a[tc] = 0.0; date_a[tc] = ed; entry_a[tc] = sp
            exit_a[tc] = sp; out_a[tc] = 0; filled_a[tc] = 0
            tc += 1; ltd = ed; cp += 1; continue
        tgt = fp * (1 + tp); stp = fp * (1 - sl)
        oc = 0.0; xp = fp; xt = 0; done = False; xi = fi
        for k in range(fi + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (xp - fp) / fp; xi = k; done = True; break
            sl_h = lows[k] <= stp; tp_h = highs[k] >= tgt
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (xp - fp) / fp; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (xp - fp) / fp; xi = np_
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = fp; exit_a[tc] = xp
        out_a[tc] = xt; filled_a[tc] = 1
        tc += 1; ltd = ed; nxt = cp + 1
        while nxt < ne:
            if entry_indices[nxt] > xi: break
            nxt += 1
        cp = nxt
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc], filled_a[:tc]

def _run_sim(indices, ohlcv, cfg, include_nofills=False):
    p, h, l, d, t = ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'].astype(np.int64)
    m = cfg['mode']
    if m == 'grace':
        r = run_trade_sim_grace(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'],
                                EXIT_TIME_MIN, cfg.get('grace_minutes', 0))
        pnl, dd, ep, xp, oc = r
        filled = np.ones(len(pnl), dtype=np.int64)
    elif m == 'limit_entry':
        if include_nofills:
            r = run_trade_sim_limit_entry_with_nofills(
                indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'],
                EXIT_TIME_MIN, cfg.get('dip_pct', 0), cfg.get('max_wait', 30))
            pnl, dd, ep, xp, oc, filled = r
        else:
            r = run_trade_sim_limit_entry(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'],
                                          EXIT_TIME_MIN, cfg.get('dip_pct', 0), cfg.get('max_wait', 30))
            pnl, dd, ep, xp, oc = r
            filled = np.ones(len(pnl), dtype=np.int64)
    else:
        r = run_trade_sim(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)
        pnl, dd, ep, xp, oc = r
        filled = np.ones(len(pnl), dtype=np.int64)
    if len(pnl) > 0 and SLIPPAGE_PCT > 0:
        slip_mask = filled.astype(np.float64)
        pnl = pnl - 2 * SLIPPAGE_PCT * slip_mask
    return pnl, dd, ep, xp, oc, filled


# ==============================================================================
# SCORING + TRADE SIM — A/B variant (family-aware feature columns)
# ==============================================================================
def score_and_backtest_ticker(ticker, cfg, df_feat, regime_df, df_15m=None):
    family = cfg['family']
    scaling_stats = _family_meta[family]['scaling_stats']
    regime_stats = _family_meta[family]['regime_stats']

    # Determine which feature columns to use
    if family == 'MR-v2-ADV':
        feature_cols = ALL_COLS_ADV
    else:
        feature_cols = ALL_COLS

    df_scaled = apply_scaling(df_feat, scaling_stats, feature_cols)

    if df_15m is not None:
        scaling_stats_15m = _family_meta[family].get('scaling_stats_15m')
        if scaling_stats_15m:
            df_15m_scaled = apply_scaling(df_15m, scaling_stats_15m, LOWRES_COLS)
        else:
            df_15m_scaled = df_15m.copy()
        XL = df_15m_scaled[LOWRES_COLS].values.astype(np.float32)

    bt_start = pd.Timestamp(BACKTEST_START, tz='UTC')
    bt_end = pd.Timestamp(BACKTEST_END, tz='UTC') + pd.Timedelta(days=1)

    score_bars = []
    for i in range(TIME_STEPS, len(df_feat)):
        ts = df_feat.index[i]
        if ts < bt_start or ts >= bt_end:
            continue
        bar_pst = ts.tz_convert(PST)
        bar_min = bar_pst.hour * 60 + bar_pst.minute
        if bar_min != SCORE_MINUTE:
            continue
        if 'Volume' in df_feat.columns and float(df_feat['Volume'].iloc[i]) == 0:
            continue
        score_bars.append(i)

    if not score_bars:
        return []

    fold_id = cfg['fold']
    _ensure_fold_loaded(family, fold_id)
    model = _get_model_for_family(family)

    XA = df_scaled[feature_cols].values.astype(np.float32)
    h_idx = cfg['horizon']
    signal_dates = []
    signal_meta = {}

    for row_idx in score_bars:
        seq_batch = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]

        regime_at_entry = regime_df.iloc[regime_df.index.get_indexer(
            [df_feat.index[row_idx]], method='pad')[0]]
        regime_scaled = np.zeros(N_REGIME_F, dtype=np.float32)
        for ri, col in enumerate(REGIME_COLS):
            val = float(regime_at_entry[col]) if col in regime_at_entry.index else 0.0
            if col in regime_stats:
                s = regime_stats[col]
                val = np.clip((val - s['med']) / s['iqr'], -10, 10)
            regime_scaled[ri] = val
        regime_batch = regime_scaled[np.newaxis]

        if df_15m is not None:
            ts_1m = df_feat.index[row_idx]
            lr_idx = df_15m_scaled.index.get_indexer([ts_1m], method='pad')[0]
            lr_end = lr_idx + 1
            lr_start = max(0, lr_end - TIME_STEPS_15M)
            lr_window = XL[lr_start:lr_end]
            if len(lr_window) < TIME_STEPS_15M:
                pad = np.zeros((TIME_STEPS_15M - len(lr_window), N_LOWRES_F), dtype=np.float32)
                lr_window = np.concatenate([pad, lr_window], axis=0)
            lr_batch = lr_window[np.newaxis]
            preds = model(
                {'seq_input': seq_batch, 'seq_15m_input': lr_batch, 'regime_input': regime_batch},
                training=False
            ).numpy().flatten()
        else:
            preds = model(
                {'seq_input': seq_batch, 'regime_input': regime_batch},
                training=False
            ).numpy().flatten()

        prob = float(preds[h_idx])
        if prob >= cfg['prob_thresh']:
            bar_pst = df_feat.index[row_idx].tz_convert(PST)
            date_int = bar_pst.year * 10000 + bar_pst.month * 100 + bar_pst.day
            signal_dates.append((row_idx, date_int, prob))
            signal_meta[date_int] = {
                'atr_14': float(df_feat['ATR_14'].iloc[row_idx]),
                'rvol': float(df_feat['RVOL'].iloc[row_idx]),
                'gap_pct': float(df_feat['Gap_Pct'].iloc[row_idx]),
                'rsi_14': float(df_feat['RSI_14'].iloc[row_idx]) if 'RSI_14' in df_feat.columns else 50.0,
            }

    if not signal_dates:
        return []

    df_bt = df_feat[(df_feat.index >= bt_start) & (df_feat.index < bt_end)].copy()
    df_bt_pst = df_bt.index.tz_convert(PST)

    ohlcv = {
        'prices': df_bt['Close'].values.copy(),
        'highs': df_bt['High'].values.copy(),
        'lows': df_bt['Low'].values.copy(),
        'dates': (df_bt_pst.year * 10000 + df_bt_pst.month * 100 + df_bt_pst.day).values.astype(np.int64),
        'times': (df_bt_pst.hour * 60 + df_bt_pst.minute).values,
    }

    entry_indices = []
    signal_probs = {}
    for row_idx, date_int, prob in signal_dates:
        for j in range(len(df_bt)):
            bt_pst = df_bt_pst[j]
            bt_date_int = bt_pst.year * 10000 + bt_pst.month * 100 + bt_pst.day
            bt_min = bt_pst.hour * 60 + bt_pst.minute
            if bt_date_int == date_int and bt_min == ENTRY_MINUTE:
                entry_indices.append(j)
                signal_probs[date_int] = prob
                break

    if not entry_indices:
        return []

    pnl, dates, ep, xp, outcomes, fill_flags = _run_sim(
        np.array(entry_indices, dtype=np.int64), ohlcv, cfg, include_nofills=True)

    trades = []
    for i in range(len(pnl)):
        d = int(dates[i])
        date_str = f"{d//10000}-{(d%10000)//100:02d}-{d%100:02d}"
        meta = signal_meta.get(d, {})
        prob = signal_probs.get(d, 0.0)
        thresh = cfg['prob_thresh']
        ev = prob * cfg['tp_pct'] - (1 - prob) * cfg['sl_pct']
        is_filled = bool(fill_flags[i])
        trades.append({
            'Ticker': ticker,
            'Date': date_str,
            'PnL': float(pnl[i]),
            'Entry': float(ep[i]),
            'Exit': float(xp[i]),
            'Outcome': 'NO_FILL' if not is_filled else ('TP' if outcomes[i]==1 else 'SL' if outcomes[i]==-1 else 'EOD'),
            'Filled': is_filled,
            'Mode': cfg['mode'],
            'Horizon': HORIZON_LABELS[h_idx],
            'Family': family,
            'Prob': prob,
            'Thresh': thresh,
            'TP_Pct': cfg['tp_pct'],
            'SL_Pct': cfg['sl_pct'],
            'Val_B': cfg['val_b_avg_pnl'],
            'ATR_14': meta.get('atr_14', 0.0),
            'RVOL': meta.get('rvol', 0.0),
            'Gap_Pct': meta.get('gap_pct', 0.0),
        })
    return trades


# ==============================================================================
# MAIN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  ST LONG — A/B TEST: MR-v2 (33f) vs MR-v2-ADV (45f)")
print(f"  {BACKTEST_START} -> {BACKTEST_END} | Fold 8 only")
print(f"  Score: {SCORE_MINUTE//60}:{SCORE_MINUTE%60:02d} PT | Entry: {ENTRY_MINUTE//60}:{ENTRY_MINUTE%60:02d} PT | "
      f"Exit: {EXIT_TIME_MIN//60}:{EXIT_TIME_MIN%60:02d} PT")
print(f"  Slippage: {2*SLIPPAGE_PCT:.2%} RT | Deploy threshold: >{BASE_PNL_THRESH:.1%}")
print(f"{'='*W}")

print(f"\n  Loading deploy configs from BQ (per-family, F8 only)...")
FAMILY_CONFIGS = load_deploy_configs()

# Collect all unique tickers across both families
all_tickers_set = set()
for family_name, cfgs in FAMILY_CONFIGS.items():
    all_tickers_set.update(cfgs.keys())
    print(f"    {family_name}: {len(cfgs)} tickers — {', '.join(sorted(cfgs.keys()))}")
ALL_TICKERS = sorted(all_tickers_set)
print(f"    Union: {len(ALL_TICKERS)} unique tickers")

print(f"\n  Loading models and scaling stats...")
load_models_and_meta(FAMILY_CONFIGS)

print(f"\n  Building market context...")
market_df = build_market_context()
_master_cache.clear()

print(f"  Building universe returns for regime features ({len(ALL_TICKERS)} tickers)...")
_univ_rets = {}
for _t in ALL_TICKERS:
    _df = load_master(_t, tail_days=TAIL_DAYS)
    if _df is not None:
        _df = heal_data(_df.sort_index())
        _univ_rets[_t] = _df['Close'].pct_change(1)
        _master_cache.pop((_t, TAIL_DAYS), None)
_univ_ret_df = pd.DataFrame(_univ_rets).reindex(market_df.index).fillna(0)
print(f"    Universe returns: {_univ_ret_df.shape[1]} tickers loaded")

print(f"  Computing regime features (22 cols)...")
regime_df = compute_regime_features(market_df, _univ_ret_df)
del _univ_ret_df, _univ_rets


# ==============================================================================
# PHASE 1: Score all tickers for each family independently
# ==============================================================================
family_trades = {}
_15m_cache = {}

for family_name in ['MR-v2', 'MR-v2-ADV']:
    cfgs = FAMILY_CONFIGS.get(family_name, {})
    if not cfgs:
        print(f"\n  {family_name}: No tickers configured, skipping.")
        family_trades[family_name] = []
        continue

    print(f"\n  Scoring {family_name} ({len(cfgs)} tickers)...")
    trades_list = []

    for ticker in sorted(cfgs.keys()):
        cfg = cfgs[ticker]
        print(f"    {ticker} ({cfg['mode']}, H={HORIZON_LABELS[cfg['horizon']]})...", end="")

        # Use ADV feature engineering for the ADV family
        if family_name == 'MR-v2-ADV':
            df_feat = engineer_asset_adv(ticker, market_df)
        else:
            df_feat = engineer_asset(ticker, market_df)

        if df_feat is None:
            print(" SKIP (load failed)")
            continue

        # Build 15-min features (same for both families)
        if ticker not in _15m_cache:
            _15m_cache[ticker] = build_15min_features(df_feat)
        df_15m = _15m_cache[ticker]

        trades = score_and_backtest_ticker(ticker, cfg, df_feat, regime_df, df_15m=df_15m)
        if trades:
            n_filled = sum(1 for t in trades if t.get('Filled', True))
            n_nofill = len(trades) - n_filled
            filled_pnls = [t['PnL'] for t in trades if t.get('Filled', True)]
            avg_str = f"avg={np.mean(filled_pnls):+.2%}" if filled_pnls else "avg=N/A"
            nf_str = f" ({n_nofill} no-fill)" if n_nofill > 0 else ""
            print(f" {n_filled}t{nf_str} {avg_str}")
            trades_list.extend(trades)
        else:
            print(" 0 signals")
        _master_cache.pop((ticker, TAIL_DAYS), None)

    family_trades[family_name] = trades_list
    print(f"    {family_name} total: {len(trades_list)} trades")

_master_cache.clear()
_15m_cache.clear()


# ==============================================================================
# PHASE 2: Compute per-family statistics
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  A/B COMPARISON RESULTS")
print(f"{'='*W}")

family_stats = {}
family_dfs = {}

for family_name in ['MR-v2', 'MR-v2-ADV']:
    trades = family_trades[family_name]
    if not trades:
        family_stats[family_name] = {}
        family_dfs[family_name] = pd.DataFrame()
        continue

    df_t = pd.DataFrame(trades)
    family_dfs[family_name] = df_t

    filled_df = df_t[df_t['Filled'] == True] if 'Filled' in df_t.columns else df_t
    n_total = len(df_t)
    n_filled = len(filled_df)
    n_nofill = n_total - n_filled

    if n_filled > 0:
        avg_pnl = filled_df['PnL'].mean()
        total_pnl = filled_df['PnL'].sum()
        win_rate = (filled_df['PnL'] > 0).mean()
        n_tickers = filled_df['Ticker'].nunique()
        n_days = filled_df['Date'].nunique()

        # Per-ticker stats
        tk_stats = filled_df.groupby('Ticker')['PnL'].agg(['mean', 'count', 'sum'])
        tk_stats.columns = ['avg_pnl', 'n_trades', 'total_pnl']

        # Daily equity curve
        daily = filled_df.groupby('Date')['PnL'].mean()
        cum_pnl = daily.cumsum()
        max_dd = 0
        peak = 0
        for c in cum_pnl.values:
            peak = max(peak, c)
            max_dd = max(max_dd, peak - c)

        sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    else:
        avg_pnl = total_pnl = win_rate = sharpe = max_dd = 0.0
        n_tickers = n_days = 0
        tk_stats = pd.DataFrame()

    family_stats[family_name] = {
        'n_tickers': n_tickers,
        'n_total': n_total,
        'n_filled': n_filled,
        'n_nofill': n_nofill,
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'n_days': n_days,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'tk_stats': tk_stats,
    }

# Print summary table
s_base = family_stats.get('MR-v2', {})
s_adv = family_stats.get('MR-v2-ADV', {})

def _delta(a, b):
    return a - b if (a is not None and b is not None) else 0.0

print(f"\n  {'':20s} {'Baseline (33f)':>16s} {'ADV (45f)':>16s} {'Delta':>12s}")
print(f"  {'─'*70}")
print(f"  {'Tickers deployed:':<20s} {s_base.get('n_tickers',0):>16d} "
      f"{s_adv.get('n_tickers',0):>16d} "
      f"{_delta(s_adv.get('n_tickers',0), s_base.get('n_tickers',0)):>+12d}")
print(f"  {'Total trades:':<20s} {s_base.get('n_filled',0):>16d} "
      f"{s_adv.get('n_filled',0):>16d} "
      f"{_delta(s_adv.get('n_filled',0), s_base.get('n_filled',0)):>+12d}")
print(f"  {'No-fills:':<20s} {s_base.get('n_nofill',0):>16d} "
      f"{s_adv.get('n_nofill',0):>16d} "
      f"{_delta(s_adv.get('n_nofill',0), s_base.get('n_nofill',0)):>+12d}")
print(f"  {'Avg PnL/trade:':<20s} {s_base.get('avg_pnl',0):>15.2%} "
      f"{s_adv.get('avg_pnl',0):>15.2%} "
      f"{_delta(s_adv.get('avg_pnl',0), s_base.get('avg_pnl',0)):>+11.2%}")
print(f"  {'Win Rate:':<20s} {s_base.get('win_rate',0):>15.1%} "
      f"{s_adv.get('win_rate',0):>15.1%} "
      f"{_delta(s_adv.get('win_rate',0), s_base.get('win_rate',0)):>+11.1%}")
print(f"  {'Total PnL:':<20s} {s_base.get('total_pnl',0):>15.2%} "
      f"{s_adv.get('total_pnl',0):>15.2%} "
      f"{_delta(s_adv.get('total_pnl',0), s_base.get('total_pnl',0)):>+11.2%}")
print(f"  {'Sharpe:':<20s} {s_base.get('sharpe',0):>16.1f} "
      f"{s_adv.get('sharpe',0):>16.1f} "
      f"{_delta(s_adv.get('sharpe',0), s_base.get('sharpe',0)):>+12.1f}")
print(f"  {'Max Drawdown:':<20s} {s_base.get('max_dd',0):>15.2%} "
      f"{s_adv.get('max_dd',0):>15.2%} "
      f"{_delta(s_adv.get('max_dd',0), s_base.get('max_dd',0)):>+11.2%}")

# Winner
if s_adv.get('avg_pnl', 0) > s_base.get('avg_pnl', 0):
    winner = "MR-v2-ADV (45f)"
    delta_pnl = s_adv.get('avg_pnl', 0) - s_base.get('avg_pnl', 0)
else:
    winner = "MR-v2 Baseline (33f)"
    delta_pnl = s_base.get('avg_pnl', 0) - s_adv.get('avg_pnl', 0)
print(f"\n  WINNER by Avg PnL: {winner} (delta={delta_pnl:+.2%})")


# ==============================================================================
# PHASE 3: Per-ticker comparison
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PER-TICKER COMPARISON")
print(f"{'='*W}")

tk_base = s_base.get('tk_stats', pd.DataFrame())
tk_adv = s_adv.get('tk_stats', pd.DataFrame())

# Merge per-ticker stats
all_tk = sorted(set(
    list(tk_base.index if not tk_base.empty else []) +
    list(tk_adv.index if not tk_adv.empty else [])
))

comparison_rows = []
print(f"\n  {'Ticker':<8} {'Base PnL':>10} {'Base #':>7} {'ADV PnL':>10} {'ADV #':>7} {'Delta':>10} {'Winner':>10}")
print(f"  {'─'*70}")

for tk in all_tk:
    b_pnl = tk_base.loc[tk, 'avg_pnl'] if (not tk_base.empty and tk in tk_base.index) else None
    b_n = int(tk_base.loc[tk, 'n_trades']) if (not tk_base.empty and tk in tk_base.index) else 0
    a_pnl = tk_adv.loc[tk, 'avg_pnl'] if (not tk_adv.empty and tk in tk_adv.index) else None
    a_n = int(tk_adv.loc[tk, 'n_trades']) if (not tk_adv.empty and tk in tk_adv.index) else 0

    b_str = f"{b_pnl:+.2%}" if b_pnl is not None else "N/A"
    a_str = f"{a_pnl:+.2%}" if a_pnl is not None else "N/A"

    if b_pnl is not None and a_pnl is not None:
        delta = a_pnl - b_pnl
        d_str = f"{delta:+.2%}"
        w_str = "ADV" if delta > 0 else ("BASE" if delta < 0 else "TIE")
    elif a_pnl is not None:
        delta = a_pnl
        d_str = "ADV only"
        w_str = "ADV"
    elif b_pnl is not None:
        delta = -b_pnl
        d_str = "Base only"
        w_str = "BASE"
    else:
        delta = 0.0
        d_str = "N/A"
        w_str = "N/A"

    comparison_rows.append({
        'Ticker': tk,
        'base_pnl': b_pnl if b_pnl is not None else 0.0,
        'base_n': b_n,
        'adv_pnl': a_pnl if a_pnl is not None else 0.0,
        'adv_n': a_n,
        'delta': delta if isinstance(delta, float) else 0.0,
    })

    print(f"  {tk:<8} {b_str:>10} {b_n:>7} {a_str:>10} {a_n:>7} {d_str:>10} {w_str:>10}")

comp_df = pd.DataFrame(comparison_rows)

# Count wins
n_adv_wins = sum(1 for r in comparison_rows if r['delta'] > 0)
n_base_wins = sum(1 for r in comparison_rows if r['delta'] < 0)
n_ties = sum(1 for r in comparison_rows if r['delta'] == 0)
print(f"\n  ADV wins: {n_adv_wins} | Base wins: {n_base_wins} | Ties: {n_ties}")


# ==============================================================================
# PHASE 4: COMPARISON CHARTS
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle(f'ST LONG A/B Test: MR-v2 Baseline (33f) vs MR-v2-ADV (45f)\n'
             f'{BACKTEST_START} -> {BACKTEST_END} | Fold 8 | '
             f'Slippage: {2*SLIPPAGE_PCT:.2%} RT',
             fontsize=13, fontweight='bold')

# ── Chart 1: Cumulative PnL equity curves ──
ax = axes[0, 0]
for family_name, color, label in [
    ('MR-v2', '#1565C0', 'Baseline (33f)'),
    ('MR-v2-ADV', '#E65100', 'ADV (45f)')
]:
    df_t = family_dfs.get(family_name)
    if df_t is None or df_t.empty:
        continue
    filled_df = df_t[df_t['Filled'] == True] if 'Filled' in df_t.columns else df_t
    if filled_df.empty:
        continue
    daily = filled_df.groupby('Date')['PnL'].mean().sort_index()
    dates_dt = pd.to_datetime(daily.index)
    cum = daily.cumsum().values * 100
    # Prepend zero
    dates_dt = dates_dt.insert(0, dates_dt[0] - pd.Timedelta(days=1))
    cum = np.insert(cum, 0, 0.0)
    ax.plot(dates_dt, cum, color=color, linewidth=2.5, label=label, alpha=0.9)

ax.axhline(0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Cumulative Avg Daily PnL (%)')
ax.set_title('Chart 1: Cumulative PnL — Equity Curves')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ── Chart 2: Per-ticker avg PnL comparison (horizontal bars) ──
ax = axes[0, 1]
if not comp_df.empty:
    sorted_comp = comp_df.sort_values('delta', ascending=True)
    y_pos = range(len(sorted_comp))
    colors_bar = ['#2E7D32' if d > 0 else '#C62828' if d < 0 else '#757575'
                  for d in sorted_comp['delta']]
    ax.barh(list(y_pos), sorted_comp['delta'].values * 100, color=colors_bar, alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sorted_comp['Ticker'].values, fontsize=7)
    ax.axvline(0, color='black', linewidth=1.0)
    ax.set_xlabel('Delta Avg PnL (ADV - Baseline) [%]')
    ax.set_title('Chart 2: Per-Ticker PnL Delta (green=ADV wins)')
    ax.grid(axis='x', alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No comparison data', ha='center', va='center', transform=ax.transAxes)

# ── Chart 3: Scatter plot — baseline vs ADV ──
ax = axes[1, 0]
if not comp_df.empty:
    # Only plot tickers present in both families
    both_mask = (comp_df['base_n'] > 0) & (comp_df['adv_n'] > 0)
    comp_both = comp_df[both_mask]

    if not comp_both.empty:
        ax.scatter(comp_both['base_pnl'] * 100, comp_both['adv_pnl'] * 100,
                   c='#1565C0', s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
        for _, row in comp_both.iterrows():
            ax.annotate(row['Ticker'],
                        (row['base_pnl'] * 100, row['adv_pnl'] * 100),
                        fontsize=6, xytext=(4, 4), textcoords='offset points')

        # Diagonal line (no difference)
        all_vals = list(comp_both['base_pnl'] * 100) + list(comp_both['adv_pnl'] * 100)
        mn, mx = min(all_vals) - 0.5, max(all_vals) + 0.5
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1, alpha=0.5, label='No difference')
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)

    ax.set_xlabel('Baseline Avg PnL (%)')
    ax.set_ylabel('ADV Avg PnL (%)')
    ax.set_title('Chart 3: Scatter — above line = ADV improvement')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No comparison data', ha='center', va='center', transform=ax.transAxes)

# ── Chart 4: Summary stats as text ──
ax = axes[1, 1]
ax.axis('off')
summary_lines = [
    f"{'':20s} {'Baseline (33f)':>16s} {'ADV (45f)':>16s} {'Delta':>12s}",
    f"{'─'*66}",
    f"{'Tickers deployed:':<20s} {s_base.get('n_tickers',0):>16d} {s_adv.get('n_tickers',0):>16d} {_delta(s_adv.get('n_tickers',0), s_base.get('n_tickers',0)):>+12d}",
    f"{'Total trades:':<20s} {s_base.get('n_filled',0):>16d} {s_adv.get('n_filled',0):>16d} {_delta(s_adv.get('n_filled',0), s_base.get('n_filled',0)):>+12d}",
    f"{'Avg PnL/trade:':<20s} {s_base.get('avg_pnl',0):>15.2%} {s_adv.get('avg_pnl',0):>15.2%} {_delta(s_adv.get('avg_pnl',0), s_base.get('avg_pnl',0)):>+11.2%}",
    f"{'Win Rate:':<20s} {s_base.get('win_rate',0):>15.1%} {s_adv.get('win_rate',0):>15.1%} {_delta(s_adv.get('win_rate',0), s_base.get('win_rate',0)):>+11.1%}",
    f"{'Total PnL:':<20s} {s_base.get('total_pnl',0):>15.2%} {s_adv.get('total_pnl',0):>15.2%} {_delta(s_adv.get('total_pnl',0), s_base.get('total_pnl',0)):>+11.2%}",
    f"{'Sharpe:':<20s} {s_base.get('sharpe',0):>16.1f} {s_adv.get('sharpe',0):>16.1f} {_delta(s_adv.get('sharpe',0), s_base.get('sharpe',0)):>+12.1f}",
    f"{'Max Drawdown:':<20s} {s_base.get('max_dd',0):>15.2%} {s_adv.get('max_dd',0):>15.2%} {_delta(s_adv.get('max_dd',0), s_base.get('max_dd',0)):>+11.2%}",
    f"{'─'*66}",
    f"WINNER: {winner} (delta={delta_pnl:+.2%})",
    f"ADV wins: {n_adv_wins} | Base wins: {n_base_wins} | Ties: {n_ties}",
]
ax.text(0.05, 0.95, '\n'.join(summary_lines),
        transform=ax.transAxes, fontsize=9, fontfamily='monospace',
        verticalalignment='top')
ax.set_title('Chart 4: Summary Statistics')

plt.tight_layout()
chart_path = '/tmp/backtest_st_long_adv_micro_ab.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# CLEANUP
# ==============================================================================
keras.backend.clear_session()
gc.collect()
try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass

print(f"\n{'='*W}")
print(f"  A/B TEST COMPLETE: MR-v2 (33f) vs MR-v2-ADV (45f)")
print(f"{'='*W}")
