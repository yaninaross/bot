# %% ── ST SHORT — 9-Model Trade Prioritization Comparison ─────────────────────
#
# EXPANDED from 6-model to 9-model by adding Multi-Res v2 (MR-v2) F6/F7/F8.
#
# 9 models:
#   L5-1 F5/F6/F7  (standard SpikeTransformer, single backbone)
#   L5-2 F5/F6/F7  (standard SpikeTransformer, single backbone)
#   MR-v2 F6/F7/F8 (MultiResSpikeTransformer, dual backbone 1m+15m)
#
# MR-v2 architecture: dual Transformer backbone (1m: 2 blocks + 15m: 1 block),
# 153K params, late fusion concat(64,64)->Dense(64,gelu), FiLM conditioning.
# Requires separate model object and 15-min feature engineering.
#
# Phase 7: K sweep (1→9) — M_CONSENSUS (9) vs M_CONS_6NEWEST (6) vs M_CONS_5NEWEST (5)
# Phase 10: Model overlap analysis (signal agreement, Jaccard, correlation)
#
# Model subsets:
#   M_CONSENSUS   (9): all models
#   M_CONS_6NEWEST(6): L5-1 F6/F7 + L5-2 F6/F7 + MR-v2 F7/F8 (drop F5s + MR-v2 F6)
#   M_CONS_5NEWEST(5): L5-1 F7 + L5-2 F7 + MR-v2 F6/F7/F8 (newest per std family + all MR-v2)
#
# SHORT = price down is profit. TP hit when Low <= tp_price.
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
BUCKET_NAME      = os.environ.get("GCS_BUCKET", "")
WF_BQ_TABLE      = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
INFERENCE_DIR    = "Data/Inference_v2"
PST              = pytz.timezone('America/Los_Angeles')

BACKTEST_START   = '2026-08-03'
BACKTEST_END     = '2026-08-24'

TAIL_DAYS        = 45

SCORE_MINUTE     = 375
ENTRY_MINUTE     = 385
EXIT_TIME_MIN    = 780
SLIPPAGE_PCT     = 0.0015

BASE_PNL_THRESH  = 0.018
F7_VAL_B_THRESH  = 0.030
MIN_TRADES_PER_MONTH = 2.0

MAX_TRADES       = 3
RANDOM_ITERS     = 50

TIME_STEPS       = 128
D_MODEL          = 64
N_HEADS          = 4
D_FF             = 128
N_BLOCKS         = 2
DROPOUT_RATE     = 0.1
FILM_HIDDEN      = 64
N_HORIZONS       = 3
N_ASSET_F        = 33
N_REGIME_F       = 22

FAMILIES = {
    'L5-1': {
        'gcs_data':   'Algos/RA_MoE/short_L5_1_data',
        'gcs_models': 'Algos/RA_MoE/short_L5_1_models',
        'arch': 'standard',
        'bq_versions': {
            5: 'spike_transformer_short_L5_1_fold5',
            6: 'spike_transformer_short_L5_1_fold6',
            7: 'spike_transformer_short_L5_1_fold7',
        },
    },
    'L5-2': {
        'gcs_data':   'Algos/RA_MoE/short_L5_2_data',
        'gcs_models': 'Algos/RA_MoE/short_L5_2_models',
        'arch': 'standard',
        'bq_versions': {
            5: 'spike_transformer_short_L5_2_fold5',
            6: 'spike_transformer_short_L5_2_fold6',
            7: 'spike_transformer_short_L5_2_fold7',
        },
    },
    'MR-v2': {
        'gcs_data':   'Algos/RA_MoE/short_L5_2_multires_v2_data',
        'gcs_models': 'Algos/RA_MoE/short_L5_2_multires_v2_models',
        'arch': 'multires',
        'bq_versions': {
            6: 'spike_transformer_short_L5_2_multires_v2_fold6',
            7: 'spike_transformer_short_L5_2_multires_v2_fold7',
            8: 'spike_transformer_short_L5_2_multires_v2_fold8',
        },
    },
}

FOLDS = [5, 6, 7, 8]
HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}
HORIZON_FROM_LABEL = {'15m': 0, '30m': 1, '60m': 2}

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR', 'BTCUSD', 'ETHUSD'}

N_BLOCKS_15M     = 1
TIME_STEPS_15M   = 128
N_LOWRES_F       = 12

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
# BQ CONFIG LOADING — Hybrid F5+F6+F7 across L5-1 + L5-2
# ==============================================================================
def load_deploy_configs():
    if bigquery is None:
        raise RuntimeError("BigQuery not available")
    bq_client = bigquery.Client()

    family_fold_cfgs = {}
    for family_name, fam in FAMILIES.items():
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
                family_fold_cfgs[(family_name, fold_id)] = {}
                print(f"    {family_name} F{fold_id}: 0 tickers")
                continue
            best = rows.loc[rows.groupby('ticker')['val_b_avg_pnl'].idxmax()]
            cfgs = {}
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
                cfgs[tk] = {
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
            family_fold_cfgs[(family_name, fold_id)] = cfgs
            print(f"    {family_name} F{fold_id}: {len(cfgs)} tickers")

    # Hybrid selection: best val_b per ticker across all (family, fold) combos
    # F7 gated at val_b >= 3.0%
    deploy = {}
    all_tickers = set()
    for cfgs in family_fold_cfgs.values():
        all_tickers |= set(cfgs.keys())

    for tk in sorted(all_tickers):
        best_fold_key, best_val, best_cfg = None, -999, None
        for (family_name, fold_id), cfgs in family_fold_cfgs.items():
            if tk not in cfgs:
                continue
            cfg = cfgs[tk]
            v = cfg['val_b_avg_pnl']
            if fold_id >= 7 and v < F7_VAL_B_THRESH:
                continue
            if v > best_val:
                best_fold_key = (family_name, fold_id)
                best_val = v
                best_cfg = cfg
        if best_cfg is not None:
            deploy[tk] = best_cfg

    source_counts = {}
    for tk, cfg in deploy.items():
        key = f"{cfg['family']} F{cfg['fold']}"
        source_counts.setdefault(key, []).append(tk)
    for src in sorted(source_counts):
        tickers = sorted(source_counts[src])
        print(f"    Hybrid {src}: {len(tickers)} — {', '.join(tickers)}")
    print(f"    Total: {len(deploy)} tickers (Hybrid F5+F6+F7+F8)")
    return deploy, family_fold_cfgs


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
    local_path = f"/tmp/_bt_short_prio_{file_ticker}_master.parquet"
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
# MODEL ARCHITECTURE (SpikeTransformer)
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

class SpikeTransformer(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.backbone = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_BLOCKS,
            TIME_STEPS, DROPOUT_RATE, name='backbone')
        self.attn_pool = _AttentionPooling(D_MODEL, name='attn_pool')
        self.film = _FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
        self.drop = Dropout(DROPOUT_RATE)
        self.head = Dense(N_HORIZONS, activation='sigmoid', name='spike_head')
    def call(self, inputs, training=False):
        h = self.backbone(inputs['seq_input'], training=training)
        h = self.attn_pool(h, training=training)
        h = self.film(h, inputs['regime_input'], training=training)
        return self.head(self.drop(h, training=training))


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
_model_std = None
_model_mr = None
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
    arch = FAMILIES[family].get('arch', 'standard')
    return _model_mr if arch == 'multires' else _model_std

def _ensure_fold_loaded(family, fold_id):
    global _current_key
    key = (family, fold_id)
    if _current_key != key:
        model = _get_model_for_family(family)
        _load_keras3_weights(model, _fold_weights[key])
        _current_key = key

def load_models_and_meta(deploy_configs, all_model_configs=None):
    global _model_std, _model_mr, _fold_weights, _current_key, _family_meta
    all_cfgs = list(deploy_configs.values())
    if all_model_configs:
        for cfg_list in all_model_configs.values():
            all_cfgs.extend(cfg_list)
    needed_families = sorted(set(cfg['family'] for cfg in all_cfgs))
    for family_name in needed_families:
        fam = FAMILIES[family_name]
        local_meta = f'/tmp/ra_moe_short_{family_name}_meta.npz'
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

    need_std = any(FAMILIES[f].get('arch', 'standard') == 'standard' for f in needed_families)
    need_mr  = any(FAMILIES[f].get('arch', 'standard') == 'multires' for f in needed_families)

    if need_std:
        _model_std = SpikeTransformer(name='spike_transformer')
        dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_F))
        dummy_reg = tf.zeros((1, N_REGIME_F))
        _model_std({'seq_input': dummy_seq, 'regime_input': dummy_reg})
        print(f"   Standard SpikeTransformer: {_model_std.count_params():,} params")

    if need_mr:
        _model_mr = MultiResSpikeTransformer(name='multires_spike_transformer')
        dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_F))
        dummy_15m = tf.zeros((1, TIME_STEPS_15M, N_LOWRES_F))
        dummy_reg = tf.zeros((1, N_REGIME_F))
        _model_mr({'seq_input': dummy_seq, 'seq_15m_input': dummy_15m, 'regime_input': dummy_reg})
        print(f"   MultiResSpikeTransformer: {_model_mr.count_params():,} params")

    needed_keys = sorted(set((cfg['family'], cfg['fold']) for cfg in all_cfgs))
    for family_name, fold_id in needed_keys:
        fam = FAMILIES[family_name]
        local_dir = f'/tmp/ra_moe_short_{family_name}_models/fold_{fold_id}'
        weights_path = os.path.join(local_dir, 'model.weights.h5')
        if not os.path.exists(weights_path):
            os.makedirs(local_dir, exist_ok=True)
            prefix = f"{fam['gcs_models']}/fold_{fold_id}"
            bucket = gcs_client.bucket(BUCKET_NAME)
            for blob in bucket.list_blobs(prefix=prefix):
                fname = blob.name.split('/')[-1]
                blob.download_to_filename(os.path.join(local_dir, fname))
        _fold_weights[(family_name, fold_id)] = weights_path
        n_tickers = sum(1 for c in deploy_configs.values()
                        if c['family'] == family_name and c['fold'] == fold_id)
        print(f"   {family_name} F{fold_id} weights cached ({n_tickers} tickers)")

    first_key = needed_keys[0]
    first_model = _get_model_for_family(first_key[0])
    _load_keras3_weights(first_model, _fold_weights[first_key])
    _current_key = first_key
    print(f"   {len(needed_keys)} (family,fold) combos, {len(deploy_configs)} tickers")


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
# NUMBA TRADE SIMULATION — SHORT DIRECTION
# TP hit when Low <= tp_price (price drops to target)
# SL hit when High >= sl_price (price rallies against us)
# EOD PnL = (entry - exit) / entry
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
        ep = prices[idx]; tgt = ep * (1 - tp); stp = ep * (1 + sl)
        oc = 0.0; xp = ep; xt = 0; done = False; xi = idx
        for k in range(idx + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (ep - xp) / ep; xi = k; done = True; break
            tp_h = lows[k] <= tgt; sl_h = highs[k] >= stp
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (ep - xp) / ep; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (ep - xp) / ep; xi = np_
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
        ep = prices[idx]; tgt = ep * (1 - tp); stp = ep * (1 + sl)
        et = time_mins[idx]; oc = 0.0; xp = ep; xt = 0; done = False; xi = idx
        for k in range(idx + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (ep - xp) / ep; xi = k; done = True; break
            elapsed = time_mins[k] - et; sl_act = elapsed >= grace_minutes
            tp_h = lows[k] <= tgt; sl_h = sl_act and (highs[k] >= stp)
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (ep - xp) / ep; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (ep - xp) / ep; xi = np_
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
        sp = prices[idx]; lp = sp * (1.0 + dip_pct); st = time_mins[idx]
        fi = -1; fp = 0.0
        for k in range(idx + 1, np_):
            if date_ints[k] != ed: break
            if time_mins[k] - st > max_wait_minutes: break
            if highs[k] >= lp: fp = lp; fi = k; break
        if fi < 0: cp += 1; ltd = ed; continue
        tgt = fp * (1 - tp); stp = fp * (1 + sl)
        oc = 0.0; xp = fp; xt = 0; done = False; xi = fi
        for k in range(fi + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (fp - xp) / fp; xi = k; done = True; break
            tp_h = lows[k] <= tgt; sl_h = highs[k] >= stp
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (fp - xp) / fp; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (fp - xp) / fp; xi = np_
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
        sp = prices[idx]; lp = sp * (1.0 + dip_pct); st = time_mins[idx]
        fi = -1; fp = 0.0
        for k in range(idx + 1, np_):
            if date_ints[k] != ed: break
            if time_mins[k] - st > max_wait_minutes: break
            if highs[k] >= lp: fp = lp; fi = k; break
        if fi < 0:
            pnl_a[tc] = 0.0; date_a[tc] = ed; entry_a[tc] = sp
            exit_a[tc] = sp; out_a[tc] = 0; filled_a[tc] = 0
            tc += 1; ltd = ed; cp += 1; continue
        tgt = fp * (1 - tp); stp = fp * (1 + sl)
        oc = 0.0; xp = fp; xt = 0; done = False; xi = fi
        for k in range(fi + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (fp - xp) / fp; xi = k; done = True; break
            tp_h = lows[k] <= tgt; sl_h = highs[k] >= stp
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (fp - xp) / fp; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (fp - xp) / fp; xi = np_
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
# SCORING + TRADE SIM — enriched with signal metadata for prioritization
# ==============================================================================
def score_and_backtest_ticker(ticker, cfg, df_feat, regime_df, df_15m=None):
    family = cfg['family']
    scaling_stats = _family_meta[family]['scaling_stats']
    regime_stats = _family_meta[family]['regime_stats']
    is_multires = FAMILIES[family].get('arch', 'standard') == 'multires'

    df_scaled = apply_scaling(df_feat, scaling_stats, ALL_COLS)

    if is_multires and df_15m is not None:
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

    XA = df_scaled[ALL_COLS].values.astype(np.float32)
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

        if is_multires and df_15m is not None:
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
            'Family': f"{cfg['family']} F{cfg['fold']}",
            'Prob': prob,
            'Thresh': thresh,
            'Prob_Margin': (prob - thresh) / max(thresh, 1e-6),
            'EV': ev,
            'Val_B': cfg['val_b_avg_pnl'],
            'TP_Pct': cfg['tp_pct'],
            'SL_Pct': cfg['sl_pct'],
            'RR_Ratio': cfg['tp_pct'] / max(cfg['sl_pct'], 1e-6),
            'ATR_14': meta.get('atr_14', 0.0),
            'RVOL': meta.get('rvol', 0.0),
            'Gap_Pct': meta.get('gap_pct', 0.0),
        })
    return trades


# ==============================================================================
# PRIORITIZATION STRATEGIES
# ==============================================================================
def _minmax_norm(vals):
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-12:
        return [0.5] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]

def select_all(day_trades, max_k, rng=None):
    return day_trades

def select_random(day_trades, max_k, rng=None):
    if rng is None:
        rng = random.Random(SEED)
    if len(day_trades) <= max_k:
        return day_trades
    return rng.sample(day_trades, max_k)

def select_prob_raw(day_trades, max_k, rng=None):
    return sorted(day_trades, key=lambda t: -t['Prob'])[:max_k]

def select_prob_margin(day_trades, max_k, rng=None):
    return sorted(day_trades, key=lambda t: -t['Prob_Margin'])[:max_k]

def select_expected_value(day_trades, max_k, rng=None):
    return sorted(day_trades, key=lambda t: -t['EV'])[:max_k]

def select_val_b(day_trades, max_k, rng=None):
    return sorted(day_trades, key=lambda t: -t['Val_B'])[:max_k]

def select_composite(day_trades, max_k, rng=None):
    if len(day_trades) <= max_k:
        return day_trades
    pm = _minmax_norm([t['Prob_Margin'] for t in day_trades])
    ev = _minmax_norm([t['EV'] for t in day_trades])
    vb = _minmax_norm([t['Val_B'] for t in day_trades])
    scores = [0.4 * pm[i] + 0.3 * ev[i] + 0.3 * vb[i] for i in range(len(day_trades))]
    ranked = sorted(zip(scores, day_trades), key=lambda x: -x[0])
    return [t for _, t in ranked[:max_k]]

def select_inverse_prob(day_trades, max_k, rng=None):
    return sorted(day_trades, key=lambda t: t['Prob'])[:max_k]


STRATEGIES = {
    'ALL_TRADES':   select_all,
    'RANDOM':       select_random,
    'PROB_RAW':     select_prob_raw,
    'PROB_MARGIN':  select_prob_margin,
    'EXPECTED_VAL': select_expected_value,
    'VAL_B_BEST':   select_val_b,
    'COMPOSITE':    select_composite,
    'INVERSE_PROB': select_inverse_prob,
}


def evaluate_strategy(all_trades_df, strategy_fn, max_k, n_iters=1):
    dates = sorted(all_trades_df['Date'].unique())
    is_random = (strategy_fn == select_random)
    actual_iters = RANDOM_ITERS if is_random else 1

    has_filled_col = 'Filled' in all_trades_df.columns

    iter_daily = []
    for iteration in range(actual_iters):
        rng = random.Random(SEED + iteration)
        selected = []
        for date in dates:
            day = all_trades_df[all_trades_df['Date'] == date].to_dict('records')
            picked = strategy_fn(day, max_k, rng)
            selected.extend(picked)

        if not selected:
            iter_daily.append([])
            continue

        sel_df = pd.DataFrame(selected)
        daily_ew = []
        for date in dates:
            day = sel_df[sel_df['Date'] == date]
            if len(day) == 0:
                continue
            n_selected = len(day)
            if has_filled_col and 'Filled' in day.columns:
                n_filled = int(day['Filled'].sum())
                filled_day = day[day['Filled'] == True]
                filled_pnl = filled_day['PnL'].mean() if len(filled_day) > 0 else 0.0
                roac = filled_day['PnL'].sum() / n_selected if n_selected > 0 else 0.0
            else:
                n_filled = n_selected
                filled_pnl = day['PnL'].mean()
                roac = day['PnL'].mean()
            daily_ew.append({
                'Date': date,
                'EW': filled_pnl,
                'ROAC': roac,
                'N': n_selected,
                'N_Filled': n_filled,
                'Tickers': sorted(day['Ticker'].unique().tolist()),
            })
        iter_daily.append(daily_ew)

    if is_random and iter_daily:
        all_dates_set = sorted(set(d['Date'] for run in iter_daily for d in run))
        avg_daily = []
        for date in all_dates_set:
            ew_vals, roac_vals, n_vals, nf_vals = [], [], [], []
            for run in iter_daily:
                for d in run:
                    if d['Date'] == date:
                        ew_vals.append(d['EW'])
                        roac_vals.append(d['ROAC'])
                        n_vals.append(d['N'])
                        nf_vals.append(d['N_Filled'])
            avg_daily.append({
                'Date': date,
                'EW': np.mean(ew_vals) if ew_vals else 0,
                'ROAC': np.mean(roac_vals) if roac_vals else 0,
                'N': np.mean(n_vals) if n_vals else 0,
                'N_Filled': np.mean(nf_vals) if nf_vals else 0,
                'Tickers': [],
            })
        return avg_daily
    else:
        return iter_daily[0] if iter_daily else []


def compute_stats(daily_ew_list):
    if not daily_ew_list:
        return {}
    ew_vals = [d['EW'] for d in daily_ew_list]
    roac_vals = [d.get('ROAC', d['EW']) for d in daily_ew_list]
    n_vals = [d['N'] for d in daily_ew_list]
    nf_vals = [d.get('N_Filled', d['N']) for d in daily_ew_list]
    cum = np.cumsum(ew_vals)
    cum_roac = np.cumsum(roac_vals)
    avg = np.mean(ew_vals)
    avg_roac = np.mean(roac_vals)
    std = np.std(ew_vals) if len(ew_vals) > 1 else 0
    std_roac = np.std(roac_vals) if len(roac_vals) > 1 else 0
    sharpe = (avg / std * np.sqrt(252)) if std > 0 else 0
    sharpe_roac = (avg_roac / std_roac * np.sqrt(252)) if std_roac > 0 else 0
    total_trades = sum(n_vals)
    total_filled = sum(nf_vals)
    win_days = sum(1 for v in ew_vals if v > 0)
    max_dd = 0
    peak = 0
    for c in cum:
        peak = max(peak, c)
        dd = peak - c
        max_dd = max(max_dd, dd)
    cap_util = total_filled / total_trades if total_trades > 0 else 1.0
    return {
        'cum_ew': cum[-1],
        'avg_daily': avg,
        'std_daily': std,
        'sharpe': sharpe,
        'total_trades': total_trades,
        'avg_trades_day': np.mean(n_vals),
        'n_days': len(daily_ew_list),
        'win_day_pct': win_days / len(daily_ew_list) if daily_ew_list else 0,
        'max_drawdown': max_dd,
        'best_day': max(ew_vals),
        'worst_day': min(ew_vals),
        'total_filled': total_filled,
        'cap_util': cap_util,
        'cum_roac': cum_roac[-1],
        'avg_daily_roac': avg_roac,
        'sharpe_roac': sharpe_roac,
    }


# ==============================================================================
# MAIN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  ST SHORT — TRADE PRIORITIZATION COMPARISON")
print(f"  {BACKTEST_START} -> {BACKTEST_END} | MAX_TRADES = {MAX_TRADES}")
print(f"  Score: {SCORE_MINUTE//60}:{SCORE_MINUTE%60:02d} PT | Entry: {ENTRY_MINUTE//60}:{ENTRY_MINUTE%60:02d} PT | "
      f"Exit: {EXIT_TIME_MIN//60}:{EXIT_TIME_MIN%60:02d} PT")
print(f"  Slippage: {2*SLIPPAGE_PCT:.2%} RT | Deploy threshold: >{BASE_PNL_THRESH:.1%}")
print(f"{'='*W}")

print(f"\n  Loading deploy configs from BQ (Hybrid F5+F6+F7+F8, 3 families)...")
DEPLOY_CONFIGS, FAMILY_FOLD_CFGS = load_deploy_configs()
DEPLOY_TICKERS = sorted(DEPLOY_CONFIGS.keys())

ALL_MODEL_CONFIGS = {}
for (family_name, fold_id), cfgs in FAMILY_FOLD_CFGS.items():
    for tk, cfg in cfgs.items():
        ALL_MODEL_CONFIGS.setdefault(tk, []).append(cfg)
n_multi = sum(1 for v in ALL_MODEL_CONFIGS.values() if len(v) > 1)
print(f"    Multi-model tickers: {n_multi}/{len(ALL_MODEL_CONFIGS)} "
      f"(tickers with configs in >1 model)")

print(f"\n  Loading models and scaling stats...")
load_models_and_meta(DEPLOY_CONFIGS, ALL_MODEL_CONFIGS)

print(f"\n  Building market context...")
market_df = build_market_context()
_master_cache.clear()

print(f"  Computing regime features (22 cols)...")
regime_df = compute_regime_features(market_df)


# ==============================================================================
# PHASE 1: Score all tickers, simulate all trades
# ==============================================================================
print(f"\n  Phase 1: Scoring and simulating ALL {len(DEPLOY_TICKERS)} tickers...")
all_trades = []
_15m_cache = {}
for ticker in DEPLOY_TICKERS:
    cfg = DEPLOY_CONFIGS[ticker]
    print(f"    {ticker} ({cfg['family']} F{cfg['fold']}, {cfg['mode']}, "
          f"H={HORIZON_LABELS[cfg['horizon']]})...", end="")
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is None:
        print(" SKIP (load failed)")
        continue
    df_15m = None
    is_mr = FAMILIES[cfg['family']].get('arch', 'standard') == 'multires'
    if is_mr:
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
        all_trades.extend(trades)
    else:
        print(" 0 signals")
    _master_cache.pop((ticker, TAIL_DAYS), None)

_master_cache.clear()


# ==============================================================================
# PHASE 2: Analyze signal distribution
# ==============================================================================
if not all_trades:
    print(f"\n  NO TRADES in backtest window. Cannot compare strategies.")
else:
    df_all = pd.DataFrame(all_trades)
    all_dates = sorted(df_all['Date'].unique())

    print(f"\n\n{'='*W}")
    print(f"  SIGNAL DISTRIBUTION ANALYSIS")
    print(f"{'='*W}\n")

    daily_counts = df_all.groupby('Date').size()
    contested_days = (daily_counts > MAX_TRADES).sum()
    total_days = len(daily_counts)
    n_filled_total = df_all['Filled'].sum() if 'Filled' in df_all.columns else len(df_all)
    n_nofill_total = len(df_all) - n_filled_total
    fill_rate = n_filled_total / len(df_all) if len(df_all) > 0 else 1.0
    n_limit = len(df_all[df_all['Mode'] == 'limit_entry']) if 'Mode' in df_all.columns else 0
    n_limit_nofill = len(df_all[(df_all['Mode'] == 'limit_entry') & (df_all['Filled'] == False)]) if 'Filled' in df_all.columns and 'Mode' in df_all.columns else 0

    print(f"  Total signals:     {len(df_all)} ({n_filled_total} filled, {n_nofill_total} no-fill)")
    print(f"  Fill rate:         {fill_rate:.0%}")
    if n_limit > 0:
        limit_fill_rate = (n_limit - n_limit_nofill) / n_limit
        print(f"  Limit entry:       {n_limit} signals, {n_limit - n_limit_nofill} filled ({limit_fill_rate:.0%})")
    print(f"  Active days:       {total_days}")
    print(f"  Avg signals/day:   {daily_counts.mean():.1f}")
    print(f"  Max signals/day:   {daily_counts.max()}")
    print(f"  Contested days:    {contested_days}/{total_days} "
          f"(days with >{MAX_TRADES} signals, where selection matters)")
    print(f"  Uncontested days:  {total_days - contested_days} (all strategies identical)\n")

    print(f"  {'Date':<12} {'Signals':>8} {'Contested':>10}  Tickers")
    print(f"  {'─'*W}")
    for date in all_dates:
        day = df_all[df_all['Date'] == date]
        n = len(day)
        contested = 'YES' if n > MAX_TRADES else ''
        tickers = ', '.join(sorted(day['Ticker'].unique()))
        print(f"  {date:<12} {n:>8} {contested:>10}  {tickers}")


    # ==========================================================================
    # PHASE 3: Evaluate each strategy
    # ==========================================================================
    print(f"\n\n{'='*W}")
    print(f"  STRATEGY COMPARISON (MAX_TRADES = {MAX_TRADES})")
    print(f"{'='*W}\n")

    strategy_results = {}
    strategy_daily = {}
    for name, fn in STRATEGIES.items():
        print(f"  Evaluating {name}...", end="")
        daily = evaluate_strategy(df_all, fn, MAX_TRADES)
        stats = compute_stats(daily)
        strategy_results[name] = stats
        strategy_daily[name] = daily
        if stats:
            print(f" EW={stats['cum_ew']:+.2%}, Sharpe={stats['sharpe']:.1f}, "
                  f"trades={stats['total_trades']:.0f}")
        else:
            print(" NO TRADES")

    print(f"\n  {'Strategy':<14} {'Cum EW':>8} {'Avg/Day':>8} {'Sharpe':>7} "
          f"{'Trades':>7} {'Filled':>7} {'CapUtil':>7} {'ROAC':>8} {'Sh_ROAC':>7} "
          f"{'Tr/Day':>6} {'WinDay%':>8} {'MaxDD':>7}")
    print(f"  {'─'*W}")

    ranked = sorted(strategy_results.items(), key=lambda x: x[1].get('cum_ew', -999), reverse=True)
    for name, s in ranked:
        if not s:
            continue
        print(f"  {name:<14} {s['cum_ew']:>+7.2%} {s['avg_daily']:>+7.2%} "
              f"{s['sharpe']:>7.1f} {s['total_trades']:>7.0f} "
              f"{s.get('total_filled', s['total_trades']):>7.0f} "
              f"{s.get('cap_util', 1.0):>6.0%} "
              f"{s.get('avg_daily_roac', s['avg_daily']):>+7.2%} "
              f"{s.get('sharpe_roac', s['sharpe']):>7.1f} "
              f"{s['avg_trades_day']:>5.1f} {s['win_day_pct']:>7.0%} "
              f"{s['max_drawdown']:>6.2%}")

    best_name = ranked[0][0] if ranked else 'NONE'
    best_stats = ranked[0][1] if ranked else {}
    print(f"\n  WINNER: {best_name}")
    if best_name != 'ALL_TRADES' and 'ALL_TRADES' in strategy_results:
        all_ew = strategy_results['ALL_TRADES'].get('cum_ew', 0)
        best_ew = best_stats.get('cum_ew', 0)
        delta = best_ew - all_ew
        print(f"  vs ALL_TRADES: {delta:+.2%} cumulative EW difference")


    # ==========================================================================
    # PHASE 4: Daily detail for top strategies
    # ==========================================================================
    top_strats = [name for name, _ in ranked[:4] if name != 'ALL_TRADES']
    if 'ALL_TRADES' not in [n for n, _ in ranked[:4]]:
        top_strats = ['ALL_TRADES'] + top_strats[:3]
    else:
        top_strats = [n for n, _ in ranked[:4]]

    print(f"\n\n{'='*W}")
    print(f"  DAILY PNL COMPARISON — TOP STRATEGIES")
    print(f"{'='*W}\n")

    header_strats = '  '.join(f'{s:>12}' for s in top_strats)
    print(f"  {'Date':<12} {header_strats}")
    print(f"  {'─'*W}")

    for date in all_dates:
        vals = []
        for sname in top_strats:
            daily = strategy_daily.get(sname, [])
            day_entry = next((d for d in daily if d['Date'] == date), None)
            if day_entry:
                vals.append(f"{day_entry['EW']:>+7.2%}({day_entry['N']:.0f}t)")
            else:
                vals.append(f"{'—':>12}")
        print(f"  {date:<12} {'  '.join(vals)}")


    # ==========================================================================
    # CONTESTED DAY DEEP DIVE
    # ==========================================================================
    print(f"\n\n{'='*W}")
    print(f"  CONTESTED DAY DEEP DIVE (days with >{MAX_TRADES} signals)")
    print(f"{'='*W}\n")

    for date in all_dates:
        day = df_all[df_all['Date'] == date]
        if len(day) <= MAX_TRADES:
            continue
        print(f"  {date} — {len(day)} signals (selecting {MAX_TRADES}):")
        print(f"    {'Ticker':<7} {'Prob':>6} {'Margin':>7} {'EV':>7} "
              f"{'Val_B':>6} {'RR':>5} {'Mode':<12} {'PnL':>7} {'Out':<8}")
        for _, t in day.sort_values('Prob', ascending=False).iterrows():
            fill_marker = '' if t.get('Filled', True) else ' [NF]'
            print(f"    {t['Ticker']:<7} {t['Prob']:>5.3f} {t['Prob_Margin']:>+6.1%} "
                  f"{t['EV']:>+6.3f} {t['Val_B']:>5.1%} {t['RR_Ratio']:>4.1f} "
                  f"{t['Mode']:<12} {t['PnL']:>+6.2%} {t['Outcome']:<8}{fill_marker}")

        print(f"    Selections:")
        for sname in ['PROB_RAW', 'PROB_MARGIN', 'EXPECTED_VAL', 'VAL_B_BEST', 'COMPOSITE']:
            fn = STRATEGIES[sname]
            picked = fn(day.to_dict('records'), MAX_TRADES)
            picked_tickers = [p['Ticker'] for p in picked]
            picked_pnl = np.mean([p['PnL'] for p in picked])
            print(f"      {sname:<14} -> {', '.join(picked_tickers):<30} EW: {picked_pnl:+.2%}")
        print()


    # ==========================================================================
    # SIGNAL METADATA CORRELATION WITH PNL
    # ==========================================================================
    print(f"\n{'='*W}")
    print(f"  SIGNAL FEATURE CORRELATIONS WITH PNL")
    print(f"{'='*W}\n")

    corr_features = ['Prob', 'Prob_Margin', 'EV', 'Val_B', 'RR_Ratio',
                      'ATR_14', 'RVOL', 'Gap_Pct']
    for feat in corr_features:
        if feat in df_all.columns:
            corr = df_all[feat].corr(df_all['PnL'])
            if pd.isna(corr):
                print(f"  {feat:<14} r=  NaN  (constant feature)")
                continue
            bar = '#' * int(abs(corr) * 50)
            sign = '+' if corr >= 0 else '-'
            print(f"  {feat:<14} r={corr:+.3f}  {sign}{bar}")
    print(f"\n  Positive r = higher feature value -> higher PnL (good for ranking)")


    # ==========================================================================
    # CHARTS
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f'ST SHORT — Trade Prioritization Comparison\n'
                 f'{BACKTEST_START} -> {BACKTEST_END} | MAX_TRADES = {MAX_TRADES} | '
                 f'{len(df_all)} total signals, {contested_days} contested days',
                 fontsize=13, fontweight='bold')

    colors_map = {
        'ALL_TRADES': '#1565C0',
        'RANDOM': '#757575',
        'PROB_RAW': '#2E7D32',
        'PROB_MARGIN': '#00838F',
        'EXPECTED_VAL': '#E65100',
        'VAL_B_BEST': '#6A1B9A',
        'COMPOSITE': '#C62828',
        'INVERSE_PROB': '#F9A825',
    }

    ax = axes[0, 0]
    for name in STRATEGIES:
        daily = strategy_daily.get(name, [])
        if not daily:
            continue
        dates_dt = pd.to_datetime([d['Date'] for d in daily])
        cum = np.cumsum([d['EW'] for d in daily]) * 100
        lw = 2.5 if name in (best_name, 'ALL_TRADES') else 1.2
        ls = '-' if name != 'RANDOM' else '--'
        ax.plot(dates_dt, cum, color=colors_map.get(name, '#333'),
                linewidth=lw, linestyle=ls, label=name, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylabel('Cumulative EW PnL (%)')
    ax.set_title('Equity Curves — All Strategies')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax = axes[0, 1]
    strat_names = [n for n, _ in ranked if strategy_results.get(n)]
    cum_vals = [strategy_results[n]['cum_ew'] * 100 for n in strat_names]
    bar_colors = [colors_map.get(n, '#333') for n in strat_names]
    bars = ax.barh(range(len(strat_names)), cum_vals, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(strat_names)))
    ax.set_yticklabels(strat_names, fontsize=9)
    for i, (name, val) in enumerate(zip(strat_names, cum_vals)):
        s = strategy_results[name]
        ax.text(val + 0.3 if val >= 0 else val - 0.3,
                i, f" Sh={s['sharpe']:.1f}", va='center', fontsize=7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Cumulative EW PnL (%)')
    ax.set_title('Strategy Ranking')
    ax.grid(axis='x', alpha=0.3)

    ax = axes[1, 0]
    all_daily = {d['Date']: d['EW'] for d in strategy_daily.get('ALL_TRADES', [])}
    for name in top_strats:
        if name == 'ALL_TRADES':
            continue
        daily = strategy_daily.get(name, [])
        if not daily:
            continue
        dates_dt = pd.to_datetime([d['Date'] for d in daily])
        diffs = [(d['EW'] - all_daily.get(d['Date'], 0)) * 100 for d in daily]
        ax.bar(dates_dt, diffs, alpha=0.5, width=0.6,
               color=colors_map.get(name, '#333'), label=name)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylabel('EW PnL diff vs ALL_TRADES (%)')
    ax.set_title('Daily Delta vs ALL_TRADES')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax = axes[1, 1]
    date_labels = pd.to_datetime(all_dates)
    counts = [len(df_all[df_all['Date'] == d]) for d in all_dates]
    bar_colors_count = ['#C62828' if c > MAX_TRADES else '#2E7D32' for c in counts]
    ax.bar(date_labels, counts, color=bar_colors_count, alpha=0.8, width=0.8)
    ax.axhline(MAX_TRADES, color='black', linewidth=1.5, linestyle='--',
               label=f'MAX_TRADES = {MAX_TRADES}')
    for i, (dt, c) in enumerate(zip(date_labels, counts)):
        if c > MAX_TRADES:
            ax.text(dt, c + 0.2, f'{c}', ha='center', fontsize=8, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylabel('Signals')
    ax.set_title('Daily Signal Count (red = contested)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    chart_path = '/tmp/backtest_st_short_prioritization_9model.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved: {chart_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()


    # ==========================================================================
    # PHASE 5: MULTI-MODEL SCORING
    # ==========================================================================
    print(f"\n\n{'='*W}")
    print(f"  PHASE 5: MULTI-MODEL SCORING")
    print(f"  Scoring each ticker with all available (family, fold) models")
    print(f"{'='*W}")

    multi_trades = []
    ticker_model_counts = {}
    _15m_cache_p5 = {}
    for ticker in DEPLOY_TICKERS:
        available = ALL_MODEL_CONFIGS.get(ticker, [])
        if not available:
            continue
        df_feat = engineer_asset(ticker, market_df)
        if df_feat is None:
            continue
        n_models_scored = 0
        for cfg in available:
            df_15m = None
            is_mr = FAMILIES[cfg['family']].get('arch', 'standard') == 'multires'
            if is_mr:
                if ticker not in _15m_cache_p5:
                    _15m_cache_p5[ticker] = build_15min_features(df_feat)
                df_15m = _15m_cache_p5[ticker]
            trades = score_and_backtest_ticker(ticker, cfg, df_feat, regime_df, df_15m=df_15m)
            for t in trades:
                t['Model_Source'] = f"{cfg['family']} F{cfg['fold']}"
            if trades:
                n_models_scored += 1
            multi_trades.extend(trades)
        ticker_model_counts[ticker] = n_models_scored
        _master_cache.pop((ticker, TAIL_DAYS), None)
        n_avail = len(available)
        print(f"    {ticker}: {n_avail} models available, "
              f"{n_models_scored} fired signals")

    _master_cache.clear()

    if multi_trades:
        df_multi = pd.DataFrame(multi_trades)
        multi_dates = sorted(df_multi['Date'].unique())

        ticker_day_agg = {}
        for date in multi_dates:
            day = df_multi[df_multi['Date'] == date]
            for ticker in day['Ticker'].unique():
                tk_day = day[day['Ticker'] == ticker]
                n_models = len(tk_day)
                probs = tk_day['Prob'].values
                ticker_day_agg[(ticker, date)] = {
                    'n_models': n_models,
                    'avg_prob': float(np.mean(probs)),
                    'max_prob': float(np.max(probs)),
                    'min_prob': float(np.min(probs)),
                    'prob_spread': float(np.max(probs) - np.min(probs)),
                    'candidates': tk_day.to_dict('records'),
                }

        print(f"\n  Multi-model pool: {len(df_multi)} total candidates "
              f"across {df_multi['Ticker'].nunique()} tickers, {len(multi_dates)} days")
        print(f"  Models per signal day (avg): "
              f"{df_multi.groupby(['Ticker','Date']).size().mean():.1f}")


        # ==================================================================
        # MULTI-MODEL STRATEGIES
        # ==================================================================
        def _resolve_ticker_model(candidates, method='max_prob'):
            if len(candidates) == 1:
                return candidates[0]
            if method == 'max_prob':
                return max(candidates, key=lambda c: c['Prob'])
            elif method == 'best_val_b':
                return max(candidates, key=lambda c: c['Val_B'])
            elif method == 'newest_fold':
                fold_order = {'F8': 0, 'F7': 1, 'F6': 2, 'F5': 3}
                family_order = {'MR-v2': 0, 'L5-2': 1, 'L5-1': 2}
                def fold_key(c):
                    src = c.get('Model_Source', c.get('Family', ''))
                    parts = src.split()
                    fam = parts[0] if parts else ''
                    fold = parts[1] if len(parts) > 1 else ''
                    return (fold_order.get(fold, 9), family_order.get(fam, 9), -c['Prob'])
                return min(candidates, key=fold_key)
            return candidates[0]

        def select_multi_consensus(day_trades_multi, max_k, rng=None):
            by_ticker = defaultdict(list)
            for t in day_trades_multi:
                by_ticker[t['Ticker']].append(t)
            resolved = []
            for ticker, cands in by_ticker.items():
                best = _resolve_ticker_model(cands, 'max_prob')
                best = dict(best)
                best['N_Models'] = len(cands)
                best['Avg_Prob_Multi'] = np.mean([c['Prob'] for c in cands])
                resolved.append(best)
            resolved.sort(key=lambda t: (-t['N_Models'], -t['Prob']))
            return resolved[:max_k]

        def select_multi_ensemble_avg(day_trades_multi, max_k, rng=None):
            by_ticker = defaultdict(list)
            for t in day_trades_multi:
                by_ticker[t['Ticker']].append(t)
            resolved = []
            for ticker, cands in by_ticker.items():
                best = _resolve_ticker_model(cands, 'max_prob')
                best = dict(best)
                best['Avg_Prob_Multi'] = np.mean([c['Prob'] for c in cands])
                resolved.append(best)
            resolved.sort(key=lambda t: -t['Avg_Prob_Multi'])
            return resolved[:max_k]

        def select_multi_max_prob(day_trades_multi, max_k, rng=None):
            by_ticker = defaultdict(list)
            for t in day_trades_multi:
                by_ticker[t['Ticker']].append(t)
            resolved = []
            for ticker, cands in by_ticker.items():
                best = _resolve_ticker_model(cands, 'max_prob')
                resolved.append(best)
            resolved.sort(key=lambda t: -t['Prob'])
            return resolved[:max_k]

        def select_multi_newest_fold(day_trades_multi, max_k, rng=None):
            by_ticker = defaultdict(list)
            for t in day_trades_multi:
                by_ticker[t['Ticker']].append(t)
            resolved = []
            for ticker, cands in by_ticker.items():
                best = _resolve_ticker_model(cands, 'newest_fold')
                resolved.append(best)
            resolved.sort(key=lambda t: -t['Prob'])
            return resolved[:max_k]

        def select_multi_agreement(day_trades_multi, max_k, rng=None):
            by_ticker = defaultdict(list)
            for t in day_trades_multi:
                by_ticker[t['Ticker']].append(t)
            resolved = []
            for ticker, cands in by_ticker.items():
                if len(cands) < 2:
                    continue
                probs = [c['Prob'] for c in cands]
                spread = max(probs) - min(probs)
                if spread > 0.15:
                    continue
                best = _resolve_ticker_model(cands, 'max_prob')
                best = dict(best)
                best['Avg_Prob_Multi'] = np.mean(probs)
                best['Prob_Spread'] = spread
                resolved.append(best)
            resolved.sort(key=lambda t: -t.get('Avg_Prob_Multi', t['Prob']))
            return resolved[:max_k]

        OLD_FAMILIES = {'L5-1', 'L5-2'}
        NEW_FAMILIES = {'MR-v2'}

        def _filter_by_families(day_trades, families):
            return [t for t in day_trades
                    if t.get('Model_Source', '').split()[0] in families]

        def select_consensus_6old(day_trades_multi, max_k, rng=None):
            return select_multi_consensus(
                _filter_by_families(day_trades_multi, OLD_FAMILIES), max_k, rng)

        def select_consensus_3new(day_trades_multi, max_k, rng=None):
            return select_multi_consensus(
                _filter_by_families(day_trades_multi, NEW_FAMILIES), max_k, rng)

        def select_consensus_5newest(day_trades_multi, max_k, rng=None):
            newest_versions = {
                ('L5-1', 'F7'), ('L5-2', 'F7'),
                ('MR-v2', 'F6'), ('MR-v2', 'F7'), ('MR-v2', 'F8'),
            }
            def _is_newest(t):
                parts = t.get('Model_Source', '').split()
                return (parts[0], parts[1]) in newest_versions if len(parts) >= 2 else False
            filtered = [t for t in day_trades_multi if _is_newest(t)]
            return select_multi_consensus(filtered, max_k, rng)

        def select_consensus_6newest(day_trades_multi, max_k, rng=None):
            newest_6 = {
                ('L5-1', 'F6'), ('L5-1', 'F7'),
                ('L5-2', 'F6'), ('L5-2', 'F7'),
                ('MR-v2', 'F7'), ('MR-v2', 'F8'),
            }
            def _is_in_6(t):
                parts = t.get('Model_Source', '').split()
                return (parts[0], parts[1]) in newest_6 if len(parts) >= 2 else False
            filtered = [t for t in day_trades_multi if _is_in_6(t)]
            return select_multi_consensus(filtered, max_k, rng)

        MULTI_STRATEGIES = {
            'M_CONSENSUS':    select_multi_consensus,
            'M_CONS_6OLD':    select_consensus_6old,
            'M_CONS_3NEW':    select_consensus_3new,
            'M_CONS_6NEWEST': select_consensus_6newest,
            'M_CONS_5NEWEST': select_consensus_5newest,
            'M_ENSEMBLE_AVG': select_multi_ensemble_avg,
            'M_MAX_PROB':     select_multi_max_prob,
            'M_NEWEST_FOLD':  select_multi_newest_fold,
            'M_AGREEMENT':    select_multi_agreement,
        }


        # ==================================================================
        # PHASE 6: EVALUATE MULTI-MODEL STRATEGIES
        # ==================================================================
        print(f"\n\n{'='*W}")
        print(f"  MULTI-MODEL STRATEGY COMPARISON (MAX_TRADES = {MAX_TRADES})")
        print(f"{'='*W}\n")

        multi_strategy_results = {}
        multi_strategy_daily = {}

        for name, fn in MULTI_STRATEGIES.items():
            selected = []
            for date in multi_dates:
                day = df_multi[df_multi['Date'] == date].to_dict('records')
                picked = fn(day, MAX_TRADES)
                selected.extend(picked)

            if not selected:
                multi_strategy_results[name] = {}
                multi_strategy_daily[name] = []
                print(f"  {name}: NO TRADES")
                continue

            sel_df = pd.DataFrame(selected)
            daily_ew = []
            for date in multi_dates:
                day = sel_df[sel_df['Date'] == date]
                if len(day) == 0:
                    continue
                n_selected = len(day)
                if 'Filled' in day.columns:
                    n_filled = int(day['Filled'].sum())
                    filled_day = day[day['Filled'] == True]
                    filled_pnl = filled_day['PnL'].mean() if len(filled_day) > 0 else 0.0
                    roac = filled_day['PnL'].sum() / n_selected if n_selected > 0 else 0.0
                else:
                    n_filled = n_selected
                    filled_pnl = day['PnL'].mean()
                    roac = day['PnL'].mean()
                daily_ew.append({
                    'Date': date,
                    'EW': filled_pnl,
                    'ROAC': roac,
                    'N': n_selected,
                    'N_Filled': n_filled,
                    'Tickers': sorted(day['Ticker'].unique().tolist()),
                })

            stats = compute_stats(daily_ew)
            multi_strategy_results[name] = stats
            multi_strategy_daily[name] = daily_ew
            cu_str = f", CapUtil={stats.get('cap_util', 1.0):.0%}" if stats.get('cap_util', 1.0) < 1.0 else ""
            print(f"  {name}: EW={stats['cum_ew']:+.2%}, "
                  f"Sharpe={stats['sharpe']:.1f}, trades={stats['total_trades']:.0f}"
                  f"{cu_str}")


        # ==================================================================
        # COMBINED COMPARISON: SINGLE-MODEL + MULTI-MODEL
        # ==================================================================
        print(f"\n\n{'='*W}")
        print(f"  COMBINED STRATEGY COMPARISON — SINGLE-MODEL + MULTI-MODEL")
        print(f"{'='*W}\n")

        all_results = {}
        all_daily = {}
        for name, s in strategy_results.items():
            all_results[name] = s
            all_daily[name] = strategy_daily[name]
        for name, s in multi_strategy_results.items():
            all_results[name] = s
            all_daily[name] = multi_strategy_daily[name]

        combined_ranked = sorted(all_results.items(),
                                  key=lambda x: x[1].get('cum_ew', -999), reverse=True)

        print(f"  {'Strategy':<16} {'Type':<7} {'Cum EW':>8} {'Avg/Day':>8} "
              f"{'Sharpe':>7} {'Trades':>7} {'Filled':>7} {'CapUtil':>7} "
              f"{'ROAC':>8} {'Sh_ROAC':>7} {'WinDay%':>8}")
        print(f"  {'─'*W}")

        for name, s in combined_ranked:
            if not s:
                continue
            stype = 'MULTI' if name.startswith('M_') else 'SINGLE'
            print(f"  {name:<16} {stype:<7} {s['cum_ew']:>+7.2%} "
                  f"{s['avg_daily']:>+7.2%} {s['sharpe']:>7.1f} "
                  f"{s['total_trades']:>7.0f} "
                  f"{s.get('total_filled', s['total_trades']):>7.0f} "
                  f"{s.get('cap_util', 1.0):>6.0%} "
                  f"{s.get('avg_daily_roac', s['avg_daily']):>+7.2%} "
                  f"{s.get('sharpe_roac', s['sharpe']):>7.1f} "
                  f"{s['win_day_pct']:>7.0%}")

        overall_best = combined_ranked[0][0] if combined_ranked else 'NONE'
        best_single = next((n for n, _ in combined_ranked if not n.startswith('M_')), None)
        best_multi = next((n for n, _ in combined_ranked if n.startswith('M_')), None)

        print(f"\n  OVERALL WINNER:     {overall_best}")
        if best_single:
            print(f"  Best single-model:  {best_single} "
                  f"(EW={all_results[best_single].get('cum_ew',0):+.2%})")
        if best_multi:
            print(f"  Best multi-model:   {best_multi} "
                  f"(EW={all_results[best_multi].get('cum_ew',0):+.2%})")
        if best_single and best_multi:
            delta = (all_results[best_multi].get('cum_ew', 0) -
                     all_results[best_single].get('cum_ew', 0))
            winner_type = "multi-model" if delta > 0 else "single-model"
            print(f"  Multi vs Single:    {delta:+.2%} "
                  f"({winner_type} wins)")


        # ==================================================================
        # PHASE 7: MAX_TRADES SWEEP (1-6) — TOP 5 STRATEGIES
        # ==================================================================
        SWEEP_RANGE = range(1, 10)
        TOP5_STRATS = {
            'M_CONSENSUS':    ('multi',  select_multi_consensus),
            'M_CONS_6NEWEST': ('multi',  select_consensus_6newest),
            'M_CONS_5NEWEST': ('multi',  select_consensus_5newest),
            'M_CONS_6OLD':    ('multi',  select_consensus_6old),
            'M_CONS_3NEW':    ('multi',  select_consensus_3new),
            'M_AGREEMENT':    ('multi',  select_multi_agreement),
        }

        print(f"\n\n{'='*W}")
        print(f"  PHASE 7: MAX_TRADES SWEEP (1 → 9) — CONSENSUS VARIANTS + TOP")
        print(f"{'='*W}\n")

        sweep_results = {}
        for sname, (stype, fn) in TOP5_STRATS.items():
            sweep_results[sname] = {}
            source_df = df_multi if stype == 'multi' else df_all
            source_dates = sorted(source_df['Date'].unique())

            for max_k in SWEEP_RANGE:
                selected = []
                for date in source_dates:
                    day = source_df[source_df['Date'] == date].to_dict('records')
                    picked = fn(day, max_k)
                    selected.extend(picked)

                if not selected:
                    sweep_results[sname][max_k] = {}
                    continue

                sel_df = pd.DataFrame(selected)
                daily_ew = []
                for date in source_dates:
                    day_sel = sel_df[sel_df['Date'] == date]
                    if len(day_sel) == 0:
                        continue
                    n_sel = len(day_sel)
                    if 'Filled' in day_sel.columns:
                        n_f = int(day_sel['Filled'].sum())
                        f_day = day_sel[day_sel['Filled'] == True]
                        f_pnl = f_day['PnL'].mean() if len(f_day) > 0 else 0.0
                        roac_v = f_day['PnL'].sum() / n_sel if n_sel > 0 else 0.0
                    else:
                        n_f = n_sel
                        f_pnl = day_sel['PnL'].mean()
                        roac_v = day_sel['PnL'].mean()
                    daily_ew.append({
                        'Date': date,
                        'EW': f_pnl,
                        'ROAC': roac_v,
                        'N': n_sel,
                        'N_Filled': n_f,
                    })
                sweep_results[sname][max_k] = compute_stats(daily_ew)

        print(f"  {'Strategy':<16}", end="")
        for k in SWEEP_RANGE:
            print(f"  {'K='+str(k):>10}", end="")
        print()
        print(f"  {'─'*W}")

        metrics_to_sweep = [
            ('Cum EW', 'cum_ew', lambda v: f"{v:>+9.2%}"),
            ('Cum ROAC', 'cum_roac', lambda v: f"{v:>+9.2%}"),
            ('Sharpe', 'sharpe', lambda v: f"{v:>10.1f}"),
            ('Sh ROAC', 'sharpe_roac', lambda v: f"{v:>10.1f}"),
            ('CapUtil', 'cap_util', lambda v: f"{v:>9.0%}"),
            ('Avg/Day', 'avg_daily', lambda v: f"{v:>+9.2%}"),
            ('MaxDD', 'max_drawdown', lambda v: f"{v:>9.2%}"),
            ('Trades', 'total_trades', lambda v: f"{v:>10.0f}"),
            ('WinDay%', 'win_day_pct', lambda v: f"{v:>9.0%}"),
        ]

        for metric_name, metric_key, fmt_fn in metrics_to_sweep:
            print(f"\n  — {metric_name} —")
            rows = []
            for sname in TOP5_STRATS:
                vals = []
                for k in SWEEP_RANGE:
                    s = sweep_results[sname].get(k, {})
                    vals.append(s.get(metric_key, 0))
                best_k = max(SWEEP_RANGE, key=lambda kk: sweep_results[sname].get(kk, {}).get(metric_key, -999))
                rows.append((sname, vals, best_k))

            for sname, vals, best_k in rows:
                print(f"  {sname:<16}", end="")
                for i, v in enumerate(vals):
                    marker = ' *' if (i + 1) == best_k and metric_key in ('cum_ew', 'sharpe') else '  '
                    print(f"{fmt_fn(v)}{marker}", end="")
                print()

        print(f"\n  (* = best K for that strategy)")

        best_overall_ew = -999
        best_overall_combo = ('', 0)
        best_overall_sharpe = -999
        best_sharpe_combo = ('', 0)
        for sname in TOP5_STRATS:
            for k in SWEEP_RANGE:
                s = sweep_results[sname].get(k, {})
                ew = s.get('cum_ew', -999)
                sh = s.get('sharpe', -999)
                if ew > best_overall_ew:
                    best_overall_ew = ew
                    best_overall_combo = (sname, k)
                if sh > best_overall_sharpe:
                    best_overall_sharpe = sh
                    best_sharpe_combo = (sname, k)

        print(f"\n  BEST by Cum EW:  {best_overall_combo[0]} @ K={best_overall_combo[1]} "
              f"({best_overall_ew:+.2%})")
        print(f"  BEST by Sharpe:  {best_sharpe_combo[0]} @ K={best_sharpe_combo[1]} "
              f"(Sharpe {best_overall_sharpe:.1f})")

        # Sweep chart
        fig3, axes3 = plt.subplots(2, 2, figsize=(20, 14))
        fig3.suptitle(f'ST SHORT — MAX_TRADES Sweep (1→9) — Consensus Variants\n'
                      f'{BACKTEST_START} → {BACKTEST_END}',
                      fontsize=13, fontweight='bold')

        sweep_colors = {
            'M_CONSENSUS': '#D81B60',
            'M_CONS_6NEWEST': '#8E24AA',
            'M_CONS_5NEWEST': '#FF6F00',
            'M_CONS_6OLD': '#1E88E5',
            'M_CONS_3NEW': '#00ACC1',
            'M_AGREEMENT': '#43A047',
        }

        ks = list(SWEEP_RANGE)

        ax = axes3[0, 0]
        for sname in TOP5_STRATS:
            vals = [sweep_results[sname].get(k, {}).get('cum_ew', 0) * 100 for k in ks]
            ax.plot(ks, vals, 'o-', color=sweep_colors[sname], linewidth=2,
                    markersize=8, label=sname)
        ax.set_xlabel('MAX_TRADES (K)')
        ax.set_ylabel('Cumulative EW PnL (%)')
        ax.set_title('Cumulative PnL vs Position Cap')
        ax.set_xticks(ks)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes3[0, 1]
        for sname in TOP5_STRATS:
            vals = [sweep_results[sname].get(k, {}).get('sharpe', 0) for k in ks]
            ax.plot(ks, vals, 's-', color=sweep_colors[sname], linewidth=2,
                    markersize=8, label=sname)
        ax.set_xlabel('MAX_TRADES (K)')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe vs Position Cap')
        ax.set_xticks(ks)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes3[1, 0]
        for sname in TOP5_STRATS:
            vals = [sweep_results[sname].get(k, {}).get('avg_daily', 0) * 100 for k in ks]
            ax.plot(ks, vals, '^-', color=sweep_colors[sname], linewidth=2,
                    markersize=8, label=sname)
        ax.set_xlabel('MAX_TRADES (K)')
        ax.set_ylabel('Avg Daily EW PnL (%)')
        ax.set_title('Avg Daily PnL vs Position Cap')
        ax.set_xticks(ks)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes3[1, 1]
        for sname in TOP5_STRATS:
            vals = [sweep_results[sname].get(k, {}).get('max_drawdown', 0) * 100 for k in ks]
            ax.plot(ks, vals, 'D-', color=sweep_colors[sname], linewidth=2,
                    markersize=8, label=sname)
        ax.set_xlabel('MAX_TRADES (K)')
        ax.set_ylabel('Max Drawdown (%)')
        ax.set_title('Max Drawdown vs Position Cap')
        ax.set_xticks(ks)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.invert_yaxis()

        plt.tight_layout()
        chart3_path = '/tmp/backtest_st_short_prioritization_9model_sweep.png'
        plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
        print(f"\n  Sweep chart saved: {chart3_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()


        # ==================================================================
        # MULTI-MODEL DEEP DIVE
        # ==================================================================
        print(f"\n\n{'='*W}")
        print(f"  MULTI-MODEL TICKER COVERAGE")
        print(f"{'='*W}\n")

        print(f"  {'Ticker':<7} {'Models':>7} {'Configs':>8} "
              f"{'HYB Source':<10} {'Avg Prob':>8} {'HYB PnL':>7}")
        print(f"  {'─'*80}")

        for ticker in sorted(DEPLOY_TICKERS):
            n_configs = len(ALL_MODEL_CONFIGS.get(ticker, []))
            n_fired = ticker_model_counts.get(ticker, 0)
            hyb_cfg = DEPLOY_CONFIGS[ticker]
            hyb_src = f"{hyb_cfg['family']} F{hyb_cfg['fold']}"

            tk_multi = df_multi[df_multi['Ticker'] == ticker]
            avg_prob = tk_multi['Prob'].mean() if len(tk_multi) > 0 else 0

            tk_single = df_all[df_all['Ticker'] == ticker]
            hyb_pnl = tk_single['PnL'].mean() if len(tk_single) > 0 else 0

            print(f"  {ticker:<7} {n_fired:>3}/{n_configs:<3} "
                  f"{n_configs:>8} {hyb_src:<10} "
                  f"{avg_prob:>7.3f} {hyb_pnl:>+6.2%}")


        # ==================================================================
        # CONTESTED DAY — MULTI-MODEL DEEP DIVE
        # ==================================================================
        print(f"\n\n{'='*W}")
        print(f"  MULTI-MODEL CONTESTED DAYS — WHO PICKS WHAT")
        print(f"{'='*W}\n")

        for date in multi_dates:
            day_multi = df_multi[df_multi['Date'] == date]
            tickers_with_signal = day_multi['Ticker'].nunique()
            if tickers_with_signal <= MAX_TRADES:
                continue

            print(f"  {date} — {len(day_multi)} candidates across "
                  f"{tickers_with_signal} tickers:")

            by_ticker = defaultdict(list)
            for _, row in day_multi.iterrows():
                by_ticker[row['Ticker']].append(row)

            for ticker in sorted(by_ticker.keys()):
                cands = by_ticker[ticker]
                models = ', '.join(f"{c['Model_Source']}({c['Prob']:.3f})"
                                   for c in sorted(cands,
                                   key=lambda x: -x['Prob']))
                pnls = [c['PnL'] for c in cands]
                print(f"    {ticker:<7} [{len(cands)} models] {models}  "
                      f"PnL range: {min(pnls):+.2%} to {max(pnls):+.2%}")

            print(f"    Selections:")
            for sname in ['M_CONSENSUS', 'M_ENSEMBLE_AVG', 'M_MAX_PROB', 'M_AGREEMENT']:
                fn = MULTI_STRATEGIES[sname]
                picked = fn(day_multi.to_dict('records'), MAX_TRADES)
                if picked:
                    picked_detail = [f"{p['Ticker']}({p.get('Model_Source', p.get('Family',''))[:5]})"
                                     for p in picked]
                    picked_pnl = np.mean([p['PnL'] for p in picked])
                    print(f"      {sname:<16} -> {', '.join(picked_detail):<35} "
                          f"EW: {picked_pnl:+.2%}")
                else:
                    print(f"      {sname:<16} -> NO TRADES")
            print()


        # ==================================================================
        # COMBINED CHARTS — SINGLE + MULTI
        # ==================================================================
        fig2, axes2 = plt.subplots(2, 2, figsize=(20, 14))
        fig2.suptitle(f'ST SHORT — Combined Prioritization: Single-Model + Multi-Model\n'
                      f'{BACKTEST_START} -> {BACKTEST_END} | MAX_TRADES = {MAX_TRADES}',
                      fontsize=13, fontweight='bold')

        multi_colors = {
            'M_CONSENSUS': '#D81B60',
            'M_CONS_6OLD': '#1E88E5',
            'M_CONS_3NEW': '#00ACC1',
            'M_CONS_6NEWEST': '#8E24AA',
            'M_CONS_5NEWEST': '#FF6F00',
            'M_ENSEMBLE_AVG': '#FFB300',
            'M_MAX_PROB': '#5C6BC0',
            'M_NEWEST_FOLD': '#78909C',
            'M_AGREEMENT': '#43A047',
        }
        all_colors = {**colors_map, **multi_colors}

        ax = axes2[0, 0]
        for name in list(STRATEGIES.keys()) + list(MULTI_STRATEGIES.keys()):
            daily = all_daily.get(name, [])
            if not daily:
                continue
            dates_dt = pd.to_datetime([d['Date'] for d in daily])
            cum = np.cumsum([d['EW'] for d in daily]) * 100
            is_multi = name.startswith('M_')
            lw = 2.0 if name == overall_best else 1.0
            ls = '--' if is_multi else '-'
            alpha = 0.9 if name == overall_best else 0.6
            ax.plot(dates_dt, cum, color=all_colors.get(name, '#333'),
                    linewidth=lw, linestyle=ls, label=name, alpha=alpha)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.set_ylabel('Cumulative EW PnL (%)')
        ax.set_title('All Strategies — Equity Curves')
        ax.legend(fontsize=6, ncol=3, loc='upper left')
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax = axes2[0, 1]
        comb_names = [n for n, _ in combined_ranked if all_results.get(n)]
        comb_vals = [all_results[n]['cum_ew'] * 100 for n in comb_names]
        comb_colors = [all_colors.get(n, '#333') for n in comb_names]
        ax.barh(range(len(comb_names)), comb_vals, color=comb_colors, alpha=0.8)
        ax.set_yticks(range(len(comb_names)))
        labels = [f"{'[M] ' if n.startswith('M_') else ''}{n}" for n in comb_names]
        ax.set_yticklabels(labels, fontsize=8)
        for i, (name, val) in enumerate(zip(comb_names, comb_vals)):
            s = all_results[name]
            ax.text(val + 0.2 if val >= 0 else val - 0.2,
                    i, f"Sh={s['sharpe']:.1f}", va='center', fontsize=7)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Cumulative EW PnL (%)')
        ax.set_title('Combined Ranking (solid=single, [M]=multi)')
        ax.grid(axis='x', alpha=0.3)

        ax = axes2[1, 0]
        config_counts = [len(ALL_MODEL_CONFIGS.get(tk, [])) for tk in DEPLOY_TICKERS]
        fired_counts = [ticker_model_counts.get(tk, 0) for tk in DEPLOY_TICKERS]
        x_pos = range(len(DEPLOY_TICKERS))
        ax.bar(x_pos, config_counts, alpha=0.4, color='#1565C0', label='Configs available')
        ax.bar(x_pos, fired_counts, alpha=0.7, color='#2E7D32', label='Models fired')
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(DEPLOY_TICKERS, rotation=90, fontsize=6)
        ax.set_ylabel('Number of models')
        ax.set_title('Per-Ticker Model Coverage')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        ax = axes2[1, 1]
        single_names = [n for n in strategy_results if strategy_results[n]]
        multi_names = [n for n in multi_strategy_results if multi_strategy_results[n]]
        for name in single_names:
            s = strategy_results[name]
            ax.scatter(s['sharpe'], s['cum_ew'] * 100,
                      color=all_colors.get(name, '#333'), s=100,
                      marker='o', label=name, zorder=5)
            ax.annotate(name, (s['sharpe'], s['cum_ew']*100),
                       fontsize=6, xytext=(5, 5), textcoords='offset points')
        for name in multi_names:
            s = multi_strategy_results[name]
            ax.scatter(s['sharpe'], s['cum_ew'] * 100,
                      color=all_colors.get(name, '#333'), s=100,
                      marker='D', label=name, zorder=5)
            ax.annotate(name, (s['sharpe'], s['cum_ew']*100),
                       fontsize=6, xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Cumulative EW PnL (%)')
        ax.set_title('Sharpe vs PnL (o=single, D=multi)')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        chart2_path = '/tmp/backtest_st_short_prioritization_9model_combined.png'
        plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
        print(f"\n  Combined chart saved: {chart2_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()


        # ==================================================================
        # DEPLOYED STRATEGY CHARTS — M_CONS_5NEWEST daily PnL + comparisons
        # ==================================================================
        DEPLOYED = 'M_CONS_5NEWEST'
        deployed_daily = multi_strategy_daily.get(DEPLOYED, [])
        if deployed_daily:
            compare_strats = [
                (DEPLOYED, multi_strategy_daily, multi_strategy_results),
                ('M_CONSENSUS', multi_strategy_daily, multi_strategy_results),
                ('M_CONS_6NEWEST', multi_strategy_daily, multi_strategy_results),
                ('VAL_B_BEST', strategy_daily, strategy_results),
            ]

            fig_dep, axes_dep = plt.subplots(2, 2, figsize=(20, 14))
            dep_stats = multi_strategy_results.get(DEPLOYED, {})
            dep_cum = dep_stats.get('cum_ew', 0)
            dep_sharpe = dep_stats.get('sharpe', 0)
            dep_trades = dep_stats.get('total_trades', 0)
            fig_dep.suptitle(
                f'ST SHORT — DEPLOYED: {DEPLOYED}\n'
                f'{BACKTEST_START} -> {BACKTEST_END} | '
                f'EW {dep_cum:+.2%} | Sharpe {dep_sharpe:.1f} | '
                f'{dep_trades:.0f} trades | MAX_TRADES {MAX_TRADES}',
                fontsize=13, fontweight='bold')

            # [0,0] Deployed equity curve with daily annotations
            ax = axes_dep[0, 0]
            dates_dt = pd.to_datetime([d['Date'] for d in deployed_daily])
            cum_vals = np.cumsum([d['EW'] for d in deployed_daily]) * 100
            ax.plot(dates_dt, cum_vals, 'b-o', linewidth=2.5, markersize=6,
                    label=DEPLOYED, zorder=5)
            ax.fill_between(dates_dt, 0, cum_vals, alpha=0.12, color='blue')
            ax.axhline(0, color='black', linewidth=0.8)
            for i, d in enumerate(deployed_daily):
                ax.annotate(f"{d['EW']*100:+.1f}%",
                            (dates_dt[i], cum_vals[i]),
                            textcoords="offset points", xytext=(0, 12),
                            fontsize=7, ha='center',
                            color='green' if d['EW'] > 0 else 'red')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.set_ylabel('Cumulative EW PnL (%)')
            ax.set_title(f'{DEPLOYED} — Equity Curve')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # [0,1] Daily PnL bar chart for deployed
            ax = axes_dep[0, 1]
            ew_vals = [d['EW'] * 100 for d in deployed_daily]
            bar_colors_dep = ['#2E7D32' if v > 0 else '#C62828' for v in ew_vals]
            ax.bar(dates_dt, ew_vals, color=bar_colors_dep, alpha=0.8, width=0.8)
            for i, d in enumerate(deployed_daily):
                ax.text(dates_dt[i],
                        ew_vals[i] + (0.08 if ew_vals[i] >= 0 else -0.25),
                        f"{d['N']}t", ha='center', fontsize=7)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.set_ylabel('Daily EW PnL (%)')
            ax.set_title(f'{DEPLOYED} — Daily PnL')
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # [1,0] Comparison equity curves — deployed vs top alternatives
            ax = axes_dep[1, 0]
            dep_compare_colors = {
                DEPLOYED: '#FF6F00',
                'M_CONSENSUS': '#D81B60',
                'M_CONS_6NEWEST': '#8E24AA',
                'VAL_B_BEST': '#6A1B9A',
            }
            for sname, s_daily_dict, s_results_dict in compare_strats:
                s_daily = s_daily_dict.get(sname, [])
                s_stats = s_results_dict.get(sname, {})
                if not s_daily:
                    continue
                s_dates = pd.to_datetime([d['Date'] for d in s_daily])
                s_cum = np.cumsum([d['EW'] for d in s_daily]) * 100
                s_ew = s_stats.get('cum_ew', 0)
                s_sh = s_stats.get('sharpe', 0)
                lw = 2.5 if sname == DEPLOYED else 1.5
                ax.plot(s_dates, s_cum,
                        color=dep_compare_colors.get(sname, '#555'),
                        linewidth=lw,
                        label=f'{sname} (EW={s_ew:+.1%}, Sh={s_sh:.1f})')
            ax.axhline(0, color='black', linewidth=0.8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.set_ylabel('Cumulative EW PnL (%)')
            ax.set_title('Deployed vs Top Alternatives')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # [1,1] Daily PnL table (text)
            ax = axes_dep[1, 1]
            ax.axis('off')
            table_lines = [f"{'Date':<12} {'#Tr':>4} {'EW PnL':>8} {'Cum':>8}  Tickers"]
            table_lines.append('─' * 60)
            cum_running = 0.0
            for d in deployed_daily:
                cum_running += d['EW']
                tks = ', '.join(d.get('Tickers', []))
                table_lines.append(
                    f"{d['Date']:<12} {d['N']:>4} {d['EW']*100:>+7.2f}% "
                    f"{cum_running*100:>+7.2f}%  {tks}")
            table_text = '\n'.join(table_lines)
            ax.text(0.02, 0.98, table_text, transform=ax.transAxes,
                    fontsize=7, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            ax.set_title(f'{DEPLOYED} — Daily Detail', fontsize=11)

            plt.tight_layout()
            dep_chart_path = '/tmp/backtest_st_short_deployed_m_cons5newest.png'
            plt.savefig(dep_chart_path, dpi=150, bbox_inches='tight')
            print(f"\n  Deployed strategy chart saved: {dep_chart_path}")
            if _IN_COLAB:
                plt.show()
            plt.close()

    else:
        print(f"\n  No multi-model trades — skipping multi-model comparison.")


    # ==========================================================================
    # PHASE 8: CAPITAL UTILIZATION ANALYSIS
    # ==========================================================================
    print(f"\n\n{'='*W}")
    print(f"  PHASE 8: CAPITAL UTILIZATION ANALYSIS")
    print(f"  How unfilled limit orders affect portfolio-level returns")
    print(f"{'='*W}\n")

    has_nofills = 'Filled' in df_all.columns and not df_all['Filled'].all()

    if not has_nofills:
        print(f"  No unfilled limit orders in this backtest period.")
        print(f"  All strategies have 100% capital utilization — EW = ROAC.")
    else:
        # Per-mode fill rate summary
        mode_summary = df_all.groupby('Mode').agg(
            Total=('Filled', 'count'),
            Filled=('Filled', 'sum'),
        )
        mode_summary['Fill_Rate'] = mode_summary['Filled'] / mode_summary['Total']
        mode_summary['No_Fill'] = mode_summary['Total'] - mode_summary['Filled']
        filled_pnls_by_mode = df_all[df_all['Filled'] == True].groupby('Mode')['PnL'].mean()
        mode_summary['Avg_PnL_Filled'] = filled_pnls_by_mode

        print(f"  Entry Mode Fill Rates:")
        print(f"    {'Mode':<14} {'Total':>6} {'Filled':>7} {'No-Fill':>8} {'FillRate':>9} {'AvgPnL':>8}")
        print(f"    {'─'*60}")
        for mode, row in mode_summary.iterrows():
            avg_pnl_str = f"{row['Avg_PnL_Filled']:+.2%}" if pd.notna(row['Avg_PnL_Filled']) else 'N/A'
            print(f"    {mode:<14} {row['Total']:>6.0f} {row['Filled']:>7.0f} "
                  f"{row['No_Fill']:>8.0f} {row['Fill_Rate']:>8.0%} {avg_pnl_str:>8}")

        # Per-ticker fill rate for limit_entry tickers
        limit_tickers = df_all[df_all['Mode'] == 'limit_entry']
        if len(limit_tickers) > 0:
            print(f"\n  Per-Ticker Limit Entry Fill Rates:")
            tk_fill = limit_tickers.groupby('Ticker').agg(
                Total=('Filled', 'count'),
                Filled=('Filled', 'sum'),
            )
            tk_fill['Fill_Rate'] = tk_fill['Filled'] / tk_fill['Total']
            tk_fill = tk_fill.sort_values('Fill_Rate')
            print(f"    {'Ticker':<7} {'Signals':>8} {'Filled':>7} {'FillRate':>9}")
            print(f"    {'─'*35}")
            for tk, row in tk_fill.iterrows():
                print(f"    {tk:<7} {row['Total']:>8.0f} {row['Filled']:>7.0f} {row['Fill_Rate']:>8.0%}")

        # EW vs ROAC comparison across ALL strategies
        print(f"\n  EW vs ROAC — Strategy Comparison:")
        print(f"  (EW = avg PnL of filled trades only; ROAC = total PnL / total allocated slots)")
        print(f"\n    {'Strategy':<16} {'Cum EW':>8} {'CumROAC':>8} {'Drag':>7} "
              f"{'CapUtil':>7} {'Filled':>7}/{' Total':<6} {'AvgEW':>8} {'AvgROAC':>8}")
        print(f"    {'─'*90}")

        all_strat_data = {}
        all_strat_data.update(strategy_results)
        if multi_trades:
            all_strat_data.update(multi_strategy_results)

        roac_ranked = sorted(all_strat_data.items(),
                             key=lambda x: x[1].get('cum_roac', x[1].get('cum_ew', -999)),
                             reverse=True)
        for name, s in roac_ranked:
            if not s:
                continue
            cum_ew = s.get('cum_ew', 0)
            cum_roac = s.get('cum_roac', cum_ew)
            drag = cum_ew - cum_roac
            cap_util = s.get('cap_util', 1.0)
            total_filled = s.get('total_filled', s['total_trades'])
            total_trades = s['total_trades']
            avg_ew = s.get('avg_daily', 0)
            avg_roac = s.get('avg_daily_roac', avg_ew)
            print(f"    {name:<16} {cum_ew:>+7.2%} {cum_roac:>+7.2%} {drag:>+6.2%} "
                  f"{cap_util:>6.0%} {total_filled:>5.0f} / {total_trades:<5.0f} "
                  f"{avg_ew:>+7.2%} {avg_roac:>+7.2%}")

        # Ranking shift analysis
        ew_rank = sorted(all_strat_data.items(),
                         key=lambda x: x[1].get('cum_ew', -999), reverse=True)
        roac_rank_list = sorted(all_strat_data.items(),
                                key=lambda x: x[1].get('cum_roac', x[1].get('cum_ew', -999)),
                                reverse=True)
        ew_rank_names = [n for n, s in ew_rank if s]
        roac_rank_names = [n for n, s in roac_rank_list if s]

        rank_shifts = []
        for name in ew_rank_names:
            ew_pos = ew_rank_names.index(name)
            roac_pos = roac_rank_names.index(name) if name in roac_rank_names else ew_pos
            shift = ew_pos - roac_pos
            if shift != 0:
                rank_shifts.append((name, ew_pos + 1, roac_pos + 1, shift))

        if rank_shifts:
            print(f"\n  Ranking Shifts (EW rank -> ROAC rank):")
            for name, ew_r, roac_r, shift in rank_shifts:
                arrow = '↑' if shift > 0 else '↓'
                print(f"    {name:<16} #{ew_r} -> #{roac_r}  ({arrow}{abs(shift)})")
        else:
            print(f"\n  No ranking shifts between EW and ROAC — fill rates are uniform.")

        # ROAC winners
        if roac_rank_names:
            best_roac = roac_rank_names[0]
            best_ew_name = ew_rank_names[0]
            print(f"\n  WINNER by EW:    {best_ew_name} ({all_strat_data[best_ew_name].get('cum_ew', 0):+.2%})")
            print(f"  WINNER by ROAC:  {best_roac} ({all_strat_data[best_roac].get('cum_roac', 0):+.2%})")
            if best_roac != best_ew_name:
                print(f"  *** ROAC flips the winner! Capital utilization matters. ***")
            else:
                print(f"  Same winner — capital utilization did not change the outcome.")

        # Capital utilization chart
        fig_cu, axes_cu = plt.subplots(2, 2, figsize=(20, 14))
        fig_cu.suptitle(f'ST SHORT — Capital Utilization Analysis\n'
                        f'{BACKTEST_START} -> {BACKTEST_END} | MAX_TRADES = {MAX_TRADES}',
                        fontsize=13, fontweight='bold')

        cu_colors = {
            'ALL_TRADES': '#1565C0', 'RANDOM': '#757575',
            'PROB_RAW': '#2E7D32', 'PROB_MARGIN': '#00838F',
            'EXPECTED_VAL': '#E65100', 'VAL_B_BEST': '#6A1B9A',
            'COMPOSITE': '#C62828', 'INVERSE_PROB': '#F9A825',
            'M_CONSENSUS': '#D81B60', 'M_CONS_6OLD': '#1E88E5',
            'M_CONS_3NEW': '#00ACC1', 'M_CONS_6NEWEST': '#8E24AA',
            'M_CONS_5NEWEST': '#FF6F00', 'M_ENSEMBLE_AVG': '#FFB300',
            'M_MAX_PROB': '#5C6BC0', 'M_NEWEST_FOLD': '#78909C',
            'M_AGREEMENT': '#43A047',
        }

        # 1: EW vs ROAC cumulative comparison
        ax = axes_cu[0, 0]
        all_daily_data = {}
        all_daily_data.update(strategy_daily)
        if multi_trades:
            all_daily_data.update(multi_strategy_daily)

        top_cu_strats = [n for n, _ in roac_ranked[:6] if all_strat_data.get(n)]
        for name in top_cu_strats:
            daily = all_daily_data.get(name, [])
            if not daily:
                continue
            dates_dt = pd.to_datetime([d['Date'] for d in daily])
            cum_ew = np.cumsum([d['EW'] for d in daily]) * 100
            cum_roac = np.cumsum([d.get('ROAC', d['EW']) for d in daily]) * 100
            c = cu_colors.get(name, '#333')
            ax.plot(dates_dt, cum_ew, color=c, linewidth=1.5, linestyle='-',
                    label=f'{name} EW', alpha=0.9)
            ax.plot(dates_dt, cum_roac, color=c, linewidth=1.5, linestyle='--',
                    label=f'{name} ROAC', alpha=0.6)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.set_ylabel('Cumulative PnL (%)')
        ax.set_title('EW (solid) vs ROAC (dashed) — Top 6')
        ax.legend(fontsize=5, ncol=2, loc='upper left')
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2: Capital utilization bar chart
        ax = axes_cu[0, 1]
        cu_names = [n for n, s in roac_ranked if s]
        cu_vals = [all_strat_data[n].get('cap_util', 1.0) * 100 for n in cu_names]
        cu_bar_colors = [cu_colors.get(n, '#333') for n in cu_names]
        bars = ax.barh(range(len(cu_names)), cu_vals, color=cu_bar_colors, alpha=0.8)
        ax.set_yticks(range(len(cu_names)))
        ax.set_yticklabels(cu_names, fontsize=8)
        for i, val in enumerate(cu_vals):
            ax.text(val + 0.5, i, f"{val:.0f}%", va='center', fontsize=7)
        ax.set_xlabel('Capital Utilization (%)')
        ax.set_title('Capital Utilization by Strategy')
        ax.set_xlim(0, 105)
        ax.axvline(100, color='black', linewidth=0.8, linestyle='--')
        ax.grid(axis='x', alpha=0.3)

        # 3: ROAC drag (EW - ROAC) per strategy
        ax = axes_cu[1, 0]
        drag_names = [n for n, s in roac_ranked if s and s.get('cap_util', 1.0) < 1.0]
        if drag_names:
            drag_vals = [(all_strat_data[n]['cum_ew'] - all_strat_data[n].get('cum_roac', all_strat_data[n]['cum_ew'])) * 100
                         for n in drag_names]
            drag_bar_colors = [cu_colors.get(n, '#333') for n in drag_names]
            ax.barh(range(len(drag_names)), drag_vals, color=drag_bar_colors, alpha=0.8)
            ax.set_yticks(range(len(drag_names)))
            ax.set_yticklabels(drag_names, fontsize=8)
            ax.set_xlabel('Idle Capital Drag (EW - ROAC, %)')
            ax.set_title('PnL Lost to Unfilled Orders')
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No unfilled orders in selected strategies',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title('PnL Lost to Unfilled Orders')

        # 4: Sharpe EW vs Sharpe ROAC scatter
        ax = axes_cu[1, 1]
        for name in cu_names:
            s = all_strat_data[name]
            sh_ew = s.get('sharpe', 0)
            sh_roac = s.get('sharpe_roac', sh_ew)
            cu = s.get('cap_util', 1.0)
            marker = 'D' if name.startswith('M_') else 'o'
            sz = 60 + (1 - cu) * 400
            ax.scatter(sh_ew, sh_roac, color=cu_colors.get(name, '#333'),
                      s=sz, marker=marker, alpha=0.8, zorder=5)
            ax.annotate(name, (sh_ew, sh_roac), fontsize=5,
                       xytext=(4, 4), textcoords='offset points')
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('Sharpe (EW — filled trades only)')
        ax.set_ylabel('Sharpe (ROAC — includes idle capital)')
        ax.set_title('Sharpe EW vs ROAC (larger dot = lower fill rate)')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        chart_cu_path = '/tmp/backtest_st_short_capital_utilization_9model.png'
        plt.savefig(chart_cu_path, dpi=150, bbox_inches='tight')
        print(f"\n  Capital utilization chart saved: {chart_cu_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()


    # ==========================================================================
    # PHASE 8b: DEPLOYED STRATEGY — EW vs ROAC HEAD-TO-HEAD
    # Isolates M_AGREEMENT (deployed) and shows the exact impact of
    # accounting for unfilled limit orders
    # ==========================================================================
    DEPLOYED_STRATEGY = 'M_AGREEMENT'
    deploy_stats = (multi_strategy_results.get(DEPLOYED_STRATEGY, {})
                    if multi_trades else strategy_results.get(DEPLOYED_STRATEGY, {}))
    deploy_daily = (multi_strategy_daily.get(DEPLOYED_STRATEGY, [])
                    if multi_trades else strategy_daily.get(DEPLOYED_STRATEGY, []))

    if deploy_stats and deploy_daily:
        print(f"\n\n{'='*W}")
        print(f"  PHASE 8b: DEPLOYED STRATEGY HEAD-TO-HEAD — {DEPLOYED_STRATEGY}")
        print(f"  EW (ignoring unfilled orders) vs ROAC (penalizing unfilled slots)")
        print(f"{'='*W}\n")

        ds = deploy_stats
        cum_ew = ds.get('cum_ew', 0)
        cum_roac = ds.get('cum_roac', cum_ew)
        avg_ew = ds.get('avg_daily', 0)
        avg_roac = ds.get('avg_daily_roac', avg_ew)
        sh_ew = ds.get('sharpe', 0)
        sh_roac = ds.get('sharpe_roac', sh_ew)
        cap_util = ds.get('cap_util', 1.0)
        total_t = ds.get('total_trades', 0)
        total_f = ds.get('total_filled', total_t)
        drag_cum = cum_ew - cum_roac
        drag_avg = avg_ew - avg_roac
        drag_sh = sh_ew - sh_roac

        print(f"  {'Metric':<22} {'EW (filled only)':>18} {'ROAC (capital-adj)':>20} {'Drag':>10}")
        print(f"  {'─'*75}")
        print(f"  {'Cumulative PnL':<22} {cum_ew:>+17.2%} {cum_roac:>+19.2%} {drag_cum:>+9.2%}")
        print(f"  {'Avg Daily PnL':<22} {avg_ew:>+17.2%} {avg_roac:>+19.2%} {drag_avg:>+9.2%}")
        print(f"  {'Sharpe Ratio':<22} {sh_ew:>17.1f} {sh_roac:>19.1f} {drag_sh:>+9.1f}")
        print(f"  {'Trades':<22} {total_f:>17.0f} {total_t:>19.0f}")
        print(f"  {'Capital Utilization':<22} {'':>17} {cap_util:>18.0%}")

        if drag_cum > 0:
            pct_overstated = (drag_cum / cum_roac * 100) if cum_roac != 0 else 0
            print(f"\n  EW overstates cumulative returns by {drag_cum:+.2%} "
                  f"({pct_overstated:.1f}% relative)")
        else:
            print(f"\n  No overstatement — all trades filled.")

        print(f"\n  Per-Day Breakdown:")
        print(f"    {'Date':<12} {'Selected':>8} {'Filled':>7} {'Unfilled':>8} "
              f"{'EW':>9} {'ROAC':>9} {'Drag':>8} {'Tickers'}")
        print(f"    {'─'*100}")

        for d in deploy_daily:
            n_sel = d['N']
            n_filled = d.get('N_Filled', n_sel)
            n_unfilled = n_sel - n_filled
            ew_val = d['EW']
            roac_val = d.get('ROAC', ew_val)
            drag_d = ew_val - roac_val
            tickers = ', '.join(d.get('Tickers', []))
            nf_marker = ' **' if n_unfilled > 0 else ''
            print(f"    {d['Date']:<12} {n_sel:>8.0f} {n_filled:>7.0f} {n_unfilled:>8.0f} "
                  f"{ew_val:>+8.2%} {roac_val:>+8.2%} {drag_d:>+7.2%} "
                  f"{tickers}{nf_marker}")

        nf_days = sum(1 for d in deploy_daily
                      if d.get('N_Filled', d['N']) < d['N'])
        total_days_deploy = len(deploy_daily)
        print(f"\n  ** = day with unfilled limit orders "
              f"({nf_days}/{total_days_deploy} days affected)")

        # Mini equity curve chart
        fig8b, ax8b = plt.subplots(1, 1, figsize=(14, 6))
        dates_dt = pd.to_datetime([d['Date'] for d in deploy_daily])
        cum_ew_arr = np.cumsum([d['EW'] for d in deploy_daily]) * 100
        cum_roac_arr = np.cumsum([d.get('ROAC', d['EW']) for d in deploy_daily]) * 100

        ax8b.fill_between(dates_dt, cum_roac_arr, cum_ew_arr,
                          alpha=0.2, color='#C62828', label='Drag (unfilled capital)')
        ax8b.plot(dates_dt, cum_ew_arr, 'o-', color='#2E7D32', linewidth=2.5,
                  markersize=7, label=f'EW (filled only): {cum_ew:+.2%}')
        ax8b.plot(dates_dt, cum_roac_arr, 's--', color='#1565C0', linewidth=2.5,
                  markersize=7, label=f'ROAC (capital-adjusted): {cum_roac:+.2%}')

        for i, d in enumerate(deploy_daily):
            if d.get('N_Filled', d['N']) < d['N']:
                ax8b.axvline(dates_dt[i], color='#C62828', alpha=0.15, linewidth=8)

        ax8b.axhline(0, color='black', linewidth=0.8)
        ax8b.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax8b.set_ylabel('Cumulative PnL (%)')
        ax8b.set_title(f'{DEPLOYED_STRATEGY} — EW vs ROAC Equity Curve\n'
                       f'Red shading = capital drag from unfilled limit orders | '
                       f'Red bars = days with unfills',
                       fontsize=11)
        ax8b.legend(fontsize=10, loc='upper left')
        ax8b.grid(alpha=0.3)
        plt.setp(ax8b.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        chart8b_path = '/tmp/backtest_st_short_deploy_ew_vs_roac_9model.png'
        plt.savefig(chart8b_path, dpi=150, bbox_inches='tight')
        print(f"\n  Deploy EW vs ROAC chart saved: {chart8b_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()


    # ==========================================================================
    # PHASE 9: MODE OVERRIDE ANALYSIS
    # For each limit_entry ticker, simulate static entry with same TP/SL
    # and compare ROAC to determine if limit_entry is actually optimal
    # ==========================================================================
    limit_tickers_phase9 = {tk: cfg for tk, cfg in DEPLOY_CONFIGS.items()
                            if cfg['mode'] == 'limit_entry'}

    if limit_tickers_phase9:
        print(f"\n\n{'='*W}")
        print(f"  PHASE 9: MODE OVERRIDE ANALYSIS")
        print(f"  For each limit_entry ticker: simulate static entry with same TP/SL")
        print(f"  Question: would switching to static improve ROAC?")
        print(f"{'='*W}\n")

        print(f"  Re-simulating {len(limit_tickers_phase9)} limit_entry tickers "
              f"with static override...\n")

        mode_comparison = []

        for ticker in sorted(limit_tickers_phase9.keys()):
            cfg_limit = limit_tickers_phase9[ticker]
            cfg_static = dict(cfg_limit)
            cfg_static['mode'] = 'static'
            cfg_static['dip_pct'] = 0
            cfg_static['max_wait'] = 0

            df_feat = engineer_asset(ticker, market_df)
            if df_feat is None:
                print(f"    {ticker}: SKIP (load failed)")
                continue

            family = cfg_limit['family']
            fold_id = cfg_limit['fold']
            is_mr = FAMILIES[family].get('arch', 'standard') == 'multires'
            _ensure_fold_loaded(family, fold_id)

            scaling_stats = _family_meta[family]['scaling_stats']
            regime_stats_local = _family_meta[family]['regime_stats']
            df_scaled = apply_scaling(df_feat, scaling_stats, ALL_COLS)

            df_15m_scaled = None
            if is_mr:
                df_15m = build_15min_features(df_feat)
                scaling_stats_15m = _family_meta[family].get('scaling_stats_15m')
                if scaling_stats_15m:
                    df_15m_scaled = apply_scaling(df_15m, scaling_stats_15m, LOWRES_COLS)
                else:
                    df_15m_scaled = df_15m.copy()

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
                print(f"    {ticker}: no score bars")
                continue

            XA = df_scaled[ALL_COLS].values.astype(np.float32)
            if is_mr:
                XL = df_15m_scaled[LOWRES_COLS].values.astype(np.float32)
            h_idx = cfg_limit['horizon']
            signal_dates = []
            model = _get_model_for_family(family)

            for row_idx in score_bars:
                seq_batch = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]
                regime_at_entry = regime_df.iloc[regime_df.index.get_indexer(
                    [df_feat.index[row_idx]], method='pad')[0]]
                regime_scaled = np.zeros(N_REGIME_F, dtype=np.float32)
                for ri, col in enumerate(REGIME_COLS):
                    val = float(regime_at_entry[col]) if col in regime_at_entry.index else 0.0
                    if col in regime_stats_local:
                        s = regime_stats_local[col]
                        val = np.clip((val - s['med']) / s['iqr'], -10, 10)
                    regime_scaled[ri] = val
                regime_batch = regime_scaled[np.newaxis]

                inputs = {'seq_input': seq_batch, 'regime_input': regime_batch}
                if is_mr:
                    ts_1m = df_feat.index[row_idx]
                    lr_idx = df_15m_scaled.index.get_indexer([ts_1m], method='pad')[0]
                    lr_start = max(0, lr_idx - TIME_STEPS_15M + 1)
                    lr_slice = XL[lr_start:lr_idx + 1]
                    if len(lr_slice) < TIME_STEPS_15M:
                        pad = np.zeros((TIME_STEPS_15M - len(lr_slice), N_LOWRES_F), dtype=np.float32)
                        lr_slice = np.concatenate([pad, lr_slice], axis=0)
                    inputs['seq_15m_input'] = lr_slice[np.newaxis]

                preds = model(inputs, training=False).numpy().flatten()

                prob = float(preds[h_idx])
                if prob >= cfg_limit['prob_thresh']:
                    bar_pst_local = df_feat.index[row_idx].tz_convert(PST)
                    date_int = bar_pst_local.year * 10000 + bar_pst_local.month * 100 + bar_pst_local.day
                    signal_dates.append((row_idx, date_int, prob))

            if not signal_dates:
                print(f"    {ticker}: no signals above threshold")
                continue

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
            for row_idx, date_int, prob in signal_dates:
                for j in range(len(df_bt)):
                    bt_pst_j = df_bt_pst[j]
                    bt_date_int = bt_pst_j.year * 10000 + bt_pst_j.month * 100 + bt_pst_j.day
                    bt_min = bt_pst_j.hour * 60 + bt_pst_j.minute
                    if bt_date_int == date_int and bt_min == ENTRY_MINUTE:
                        entry_indices.append(j)
                        break

            if not entry_indices:
                print(f"    {ticker}: no entry bars found")
                continue

            n_signals = len(entry_indices)
            idx_arr = np.array(entry_indices, dtype=np.int64)

            pnl_lim, dates_lim, _, _, outcomes_lim, filled_lim = _run_sim(
                idx_arr, ohlcv, cfg_limit, include_nofills=True)

            pnl_sta, dates_sta, _, _, outcomes_sta, filled_sta = _run_sim(
                idx_arr, ohlcv, cfg_static, include_nofills=False)

            n_filled_lim = int(np.sum(filled_lim)) if len(filled_lim) > 0 else 0
            n_filled_sta = int(np.sum(filled_sta)) if len(filled_sta) > 0 else n_signals

            fill_rate_lim = n_filled_lim / n_signals if n_signals > 0 else 0
            fill_rate_sta = 1.0

            filled_pnl_lim = pnl_lim[filled_lim == 1] if len(filled_lim) > 0 else pnl_lim
            avg_pnl_lim = float(np.mean(filled_pnl_lim)) if len(filled_pnl_lim) > 0 else 0
            avg_pnl_sta = float(np.mean(pnl_sta)) if len(pnl_sta) > 0 else 0

            total_pnl_lim = float(np.sum(pnl_lim))
            total_pnl_sta = float(np.sum(pnl_sta))

            roac_lim = total_pnl_lim / n_signals if n_signals > 0 else 0
            roac_sta = total_pnl_sta / n_signals if n_signals > 0 else 0

            winner = 'STATIC' if roac_sta > roac_lim else 'LIMIT'
            delta_roac = roac_sta - roac_lim

            alt_cfgs = []
            for (fam_name, fld), cfgs_dict in FAMILY_FOLD_CFGS.items():
                if ticker in cfgs_dict:
                    alt_cfg = cfgs_dict[ticker]
                    if alt_cfg['mode'] != 'limit_entry':
                        alt_cfgs.append(alt_cfg)

            mode_comparison.append({
                'Ticker': ticker,
                'Family': f"{cfg_limit['family']} F{cfg_limit['fold']}",
                'Signals': n_signals,
                'Lim_Filled': n_filled_lim,
                'Lim_FillRate': fill_rate_lim,
                'Lim_AvgPnL': avg_pnl_lim,
                'Lim_TotalPnL': total_pnl_lim,
                'Lim_ROAC': roac_lim,
                'Sta_Filled': n_filled_sta,
                'Sta_AvgPnL': avg_pnl_sta,
                'Sta_TotalPnL': total_pnl_sta,
                'Sta_ROAC': roac_sta,
                'Winner': winner,
                'Delta_ROAC': delta_roac,
                'TP': cfg_limit['tp_pct'],
                'SL': cfg_limit['sl_pct'],
                'Dip': cfg_limit.get('dip_pct', 0),
                'MaxWait': cfg_limit.get('max_wait', 0),
                'Alt_NonLimit_Cfgs': len(alt_cfgs),
            })

            print(f"    {ticker} ({cfg_limit['family']} F{cfg_limit['fold']}): "
                  f"{n_signals} signals | "
                  f"LIMIT: {n_filled_lim}/{n_signals} filled ({fill_rate_lim:.0%}), "
                  f"avg={avg_pnl_lim:+.2%}, ROAC={roac_lim:+.2%} | "
                  f"STATIC: {n_filled_sta}/{n_signals}, "
                  f"avg={avg_pnl_sta:+.2%}, ROAC={roac_sta:+.2%} | "
                  f"-> {winner} ({delta_roac:+.2%})")

            _master_cache.pop((ticker, TAIL_DAYS), None)

        _master_cache.clear()

        if mode_comparison:
            df_mc = pd.DataFrame(mode_comparison)

            print(f"\n\n  {'─'*W}")
            print(f"  MODE OVERRIDE SUMMARY — Limit Entry vs Static (same TP/SL)")
            print(f"  {'─'*W}\n")
            print(f"    {'Ticker':<7} {'Source':<10} {'Sig':>4} "
                  f"{'L.Fill':>6} {'L.Rate':>6} {'L.AvgPnL':>9} {'L.ROAC':>8} "
                  f"{'S.AvgPnL':>9} {'S.ROAC':>8} "
                  f"{'Winner':>7} {'ΔROAC':>8} {'Dip%':>6} {'Wait':>5}")
            print(f"    {'─'*110}")

            for _, row in df_mc.sort_values('Delta_ROAC', ascending=False).iterrows():
                print(f"    {row['Ticker']:<7} {row['Family']:<10} {row['Signals']:>4} "
                      f"{row['Lim_Filled']:>4}/{row['Signals']:<1} {row['Lim_FillRate']:>5.0%} "
                      f"{row['Lim_AvgPnL']:>+8.2%} {row['Lim_ROAC']:>+7.2%} "
                      f"{row['Sta_AvgPnL']:>+8.2%} {row['Sta_ROAC']:>+7.2%} "
                      f"{row['Winner']:>7} {row['Delta_ROAC']:>+7.2%} "
                      f"{row['Dip']:>5.1%} {row['MaxWait']:>4}m")

            n_static_wins = (df_mc['Winner'] == 'STATIC').sum()
            n_limit_wins = (df_mc['Winner'] == 'LIMIT').sum()
            avg_delta = df_mc['Delta_ROAC'].mean()
            override_tickers = df_mc[df_mc['Winner'] == 'STATIC']['Ticker'].tolist()

            print(f"\n  Results: STATIC wins {n_static_wins}/{len(df_mc)}, "
                  f"LIMIT wins {n_limit_wins}/{len(df_mc)}")
            print(f"  Avg ΔROAC (static - limit): {avg_delta:+.2%}")

            if override_tickers:
                print(f"\n  RECOMMENDED OVERRIDES (switch to static):")
                for tk in override_tickers:
                    row = df_mc[df_mc['Ticker'] == tk].iloc[0]
                    print(f"    {tk}: ROAC {row['Lim_ROAC']:+.2%} -> {row['Sta_ROAC']:+.2%} "
                          f"({row['Delta_ROAC']:+.2%}), "
                          f"fill rate was {row['Lim_FillRate']:.0%}")
            else:
                print(f"\n  No overrides recommended — limit_entry wins for all tickers.")

            still_limit = df_mc[df_mc['Winner'] == 'LIMIT']
            if len(still_limit) > 0:
                print(f"\n  KEEP LIMIT ENTRY:")
                for _, row in still_limit.iterrows():
                    print(f"    {row['Ticker']}: limit ROAC {row['Lim_ROAC']:+.2%} > "
                          f"static ROAC {row['Sta_ROAC']:+.2%} "
                          f"(fill rate {row['Lim_FillRate']:.0%}, "
                          f"but filled PnL {row['Lim_AvgPnL']:+.2%} compensates)")

            # Portfolio impact: what if we override all STATIC-winning tickers?
            if override_tickers:
                orig_limit_pnl = df_mc['Lim_TotalPnL'].sum()
                override_pnl = 0.0
                for _, row in df_mc.iterrows():
                    if row['Winner'] == 'STATIC':
                        override_pnl += row['Sta_TotalPnL']
                    else:
                        override_pnl += row['Lim_TotalPnL']

                total_sigs = df_mc['Signals'].sum()
                orig_roac = orig_limit_pnl / total_sigs if total_sigs > 0 else 0
                new_roac = override_pnl / total_sigs if total_sigs > 0 else 0

                print(f"\n  PORTFOLIO IMPACT (limit_entry tickers only):")
                print(f"    Original (all limit):  Total PnL {orig_limit_pnl:+.2%}, "
                      f"ROAC {orig_roac:+.2%} across {total_sigs} signal-days")
                print(f"    With overrides:        Total PnL {override_pnl:+.2%}, "
                      f"ROAC {new_roac:+.2%} across {total_sigs} signal-days")
                print(f"    Delta:                 {override_pnl - orig_limit_pnl:+.2%} total, "
                      f"{new_roac - orig_roac:+.2%} ROAC")

            # Chart: per-ticker ROAC comparison
            fig9, axes9 = plt.subplots(1, 2, figsize=(18, 8))
            fig9.suptitle(f'ST SHORT — Mode Override Analysis: Limit Entry vs Static\n'
                          f'{BACKTEST_START} -> {BACKTEST_END}',
                          fontsize=13, fontweight='bold')

            ax = axes9[0]
            tickers_sorted = df_mc.sort_values('Delta_ROAC', ascending=True)['Ticker'].values
            y_pos = np.arange(len(tickers_sorted))
            roac_lim_vals = [float(df_mc[df_mc['Ticker']==tk]['Lim_ROAC'].iloc[0]) * 100
                             for tk in tickers_sorted]
            roac_sta_vals = [float(df_mc[df_mc['Ticker']==tk]['Sta_ROAC'].iloc[0]) * 100
                             for tk in tickers_sorted]

            bar_h = 0.35
            bars1 = ax.barh(y_pos - bar_h/2, roac_lim_vals, bar_h,
                           label='Limit Entry ROAC', color='#E65100', alpha=0.8)
            bars2 = ax.barh(y_pos + bar_h/2, roac_sta_vals, bar_h,
                           label='Static ROAC', color='#1565C0', alpha=0.8)

            for i, tk in enumerate(tickers_sorted):
                row = df_mc[df_mc['Ticker'] == tk].iloc[0]
                winner_marker = ' *' if row['Winner'] == 'STATIC' else ''
                fill_str = f" ({row['Lim_FillRate']:.0%})"
                ax.annotate(f"{tk}{fill_str}{winner_marker}",
                           xy=(0, y_pos[i]), fontsize=7,
                           va='center', ha='right',
                           xytext=(-5, 0), textcoords='offset points')

            ax.set_yticks(y_pos)
            ax.set_yticklabels([''] * len(tickers_sorted))
            ax.set_xlabel('ROAC per signal-day (%)')
            ax.set_title('ROAC: Limit Entry vs Static Override\n(* = static wins, % = limit fill rate)')
            ax.legend(fontsize=8, loc='lower right')
            ax.axvline(0, color='black', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)

            ax = axes9[1]
            fill_rates = df_mc['Lim_FillRate'].values * 100
            delta_roacs = df_mc['Delta_ROAC'].values * 100
            colors9 = ['#1565C0' if w == 'STATIC' else '#E65100'
                       for w in df_mc['Winner'].values]
            ax.scatter(fill_rates, delta_roacs, c=colors9, s=80, alpha=0.8, zorder=5)
            for _, row in df_mc.iterrows():
                ax.annotate(row['Ticker'],
                           (row['Lim_FillRate'] * 100, row['Delta_ROAC'] * 100),
                           fontsize=7, xytext=(4, 4), textcoords='offset points')
            ax.axhline(0, color='black', linewidth=1, linestyle='--')
            ax.set_xlabel('Limit Entry Fill Rate (%)')
            ax.set_ylabel('ΔROAC: Static minus Limit (%)')
            ax.set_title('Fill Rate vs Mode Override Benefit\n'
                         '(blue = switch to static, orange = keep limit)')
            ax.grid(alpha=0.3)

            plt.tight_layout()
            chart9_path = '/tmp/backtest_st_short_mode_override_9model.png'
            plt.savefig(chart9_path, dpi=150, bbox_inches='tight')
            print(f"\n  Mode override chart saved: {chart9_path}")
            if _IN_COLAB:
                plt.show()
            plt.close()
    else:
        print(f"\n  Phase 9 skipped — no limit_entry tickers in deploy configs.")


    # ==========================================================================
    # PHASE 10: MODEL OVERLAP / SIGNAL AGREEMENT ANALYSIS
    # ==========================================================================
    if multi_trades:
        print(f"\n\n{'='*W}")
        print(f"  PHASE 10: MODEL OVERLAP & SIGNAL AGREEMENT")
        print(f"  Analyzing signal overlap across all 9 models")
        print(f"{'='*W}")

        df_overlap = pd.DataFrame(multi_trades)
        model_names = sorted(df_overlap['Model_Source'].unique())
        n_models_total = len(model_names)
        print(f"\n  Models found: {n_models_total}")
        for mn in model_names:
            cnt = len(df_overlap[df_overlap['Model_Source'] == mn])
            print(f"    {mn}: {cnt} signals")

        model_signal_sets = {}
        model_prob_maps = {}
        for mn in model_names:
            sub = df_overlap[df_overlap['Model_Source'] == mn]
            sig_keys = set(zip(sub['Ticker'], sub['Date']))
            model_signal_sets[mn] = sig_keys
            model_prob_maps[mn] = {(r['Ticker'], r['Date']): r['Prob']
                                   for _, r in sub.iterrows()}

        print(f"\n  --- Pairwise Jaccard Similarity (signals as ticker-date pairs) ---")
        jaccard_matrix = np.zeros((n_models_total, n_models_total))
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                s1, s2 = model_signal_sets[m1], model_signal_sets[m2]
                union = len(s1 | s2)
                inter = len(s1 & s2)
                jaccard_matrix[i, j] = inter / union if union > 0 else 0.0

        header = f"{'':>12s}" + ''.join(f"  {mn:>10s}" for mn in model_names)
        print(f"  {header}")
        for i, mn in enumerate(model_names):
            row_str = f"  {mn:>12s}"
            for j in range(n_models_total):
                row_str += f"  {jaccard_matrix[i,j]:>10.3f}"
            print(row_str)

        off_diag = []
        for i in range(n_models_total):
            for j in range(i + 1, n_models_total):
                off_diag.append(jaccard_matrix[i, j])
        print(f"\n  Avg pairwise Jaccard: {np.mean(off_diag):.3f} "
              f"(min={np.min(off_diag):.3f}, max={np.max(off_diag):.3f})")

        print(f"\n  --- Pairwise Prob Correlation (where both models fire) ---")
        corr_matrix = np.full((n_models_total, n_models_total), np.nan)
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    continue
                common = model_signal_sets[m1] & model_signal_sets[m2]
                if len(common) < 5:
                    continue
                p1 = [model_prob_maps[m1][k] for k in common]
                p2 = [model_prob_maps[m2][k] for k in common]
                corr_matrix[i, j] = float(np.corrcoef(p1, p2)[0, 1])

        header2 = f"{'':>12s}" + ''.join(f"  {mn:>10s}" for mn in model_names)
        print(f"  {header2}")
        for i, mn in enumerate(model_names):
            row_str = f"  {mn:>12s}"
            for j in range(n_models_total):
                v = corr_matrix[i, j]
                row_str += f"  {v:>10.3f}" if not np.isnan(v) else f"  {'N/A':>10s}"
            print(row_str)

        off_corr = [corr_matrix[i, j] for i in range(n_models_total)
                     for j in range(i + 1, n_models_total)
                     if not np.isnan(corr_matrix[i, j])]
        if off_corr:
            print(f"\n  Avg pairwise prob correlation: {np.mean(off_corr):.3f} "
                  f"(min={np.min(off_corr):.3f}, max={np.max(off_corr):.3f})")

        print(f"\n  --- Family-Level Overlap ---")
        family_sets = {}
        for mn in model_names:
            fam = mn.split()[0]
            if fam not in family_sets:
                family_sets[fam] = set()
            family_sets[fam] |= model_signal_sets[mn]
        fam_names = sorted(family_sets.keys())
        for i, f1 in enumerate(fam_names):
            for j, f2 in enumerate(fam_names):
                if j <= i:
                    continue
                s1, s2 = family_sets[f1], family_sets[f2]
                union = len(s1 | s2)
                inter = len(s1 & s2)
                jacc = inter / union if union > 0 else 0
                only1 = len(s1 - s2)
                only2 = len(s2 - s1)
                print(f"    {f1} vs {f2}: Jaccard={jacc:.3f}, "
                      f"overlap={inter}, {f1}-only={only1}, {f2}-only={only2}")

        print(f"\n  --- Signal Breadth Distribution ---")
        signal_keys_all = set()
        for s in model_signal_sets.values():
            signal_keys_all |= s
        breadth_counts = {}
        for key in signal_keys_all:
            n = sum(1 for mn in model_names if key in model_signal_sets[mn])
            breadth_counts[n] = breadth_counts.get(n, 0) + 1
        total_uniq = len(signal_keys_all)
        for n_agree in sorted(breadth_counts.keys()):
            cnt = breadth_counts[n_agree]
            pct = cnt / total_uniq * 100
            print(f"    {n_agree} model(s) agree: {cnt} signals ({pct:.1f}%)")

        solo_signals = [k for k in signal_keys_all
                        if sum(1 for mn in model_names if k in model_signal_sets[mn]) == 1]
        if solo_signals:
            solo_by_model = defaultdict(int)
            for k in solo_signals:
                for mn in model_names:
                    if k in model_signal_sets[mn]:
                        solo_by_model[mn] += 1
            print(f"\n  Unique-only signals by model:")
            for mn in model_names:
                if mn in solo_by_model:
                    print(f"    {mn}: {solo_by_model[mn]} unique signals")

        fig10, axes10 = plt.subplots(1, 3, figsize=(22, 7))

        ax = axes10[0]
        mask = np.zeros_like(jaccard_matrix, dtype=bool)
        im = ax.imshow(jaccard_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(n_models_total))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_models_total))
        ax.set_yticklabels(model_names, fontsize=8)
        for i in range(n_models_total):
            for j in range(n_models_total):
                ax.text(j, i, f"{jaccard_matrix[i,j]:.2f}",
                        ha='center', va='center', fontsize=7,
                        color='white' if jaccard_matrix[i,j] > 0.5 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title('Jaccard Similarity\n(signal overlap)')

        ax = axes10[1]
        corr_plot = np.where(np.isnan(corr_matrix), 0, corr_matrix)
        im2 = ax.imshow(corr_plot, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n_models_total))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_models_total))
        ax.set_yticklabels(model_names, fontsize=8)
        for i in range(n_models_total):
            for j in range(n_models_total):
                v = corr_matrix[i, j]
                txt = f"{v:.2f}" if not np.isnan(v) else "N/A"
                ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                        color='white' if abs(corr_plot[i,j]) > 0.5 else 'black')
        plt.colorbar(im2, ax=ax, shrink=0.8)
        ax.set_title('Probability Correlation\n(where both fire)')

        ax = axes10[2]
        breadth_x = sorted(breadth_counts.keys())
        breadth_y = [breadth_counts[x] for x in breadth_x]
        bars = ax.bar(breadth_x, breadth_y, color='#1565C0', alpha=0.8,
                      edgecolor='white', linewidth=0.5)
        for bar, cnt in zip(bars, breadth_y):
            pct = cnt / total_uniq * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{cnt}\n({pct:.0f}%)", ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('Number of Models Agreeing')
        ax.set_ylabel('Number of Signals')
        ax.set_title(f'Signal Breadth Distribution\n({total_uniq} unique ticker-day signals)')
        ax.set_xticks(breadth_x)
        ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Phase 10: Model Overlap & Signal Agreement (9 Models)', fontsize=14, y=1.02)
        plt.tight_layout()
        chart10_path = '/tmp/backtest_st_short_model_overlap.png'
        plt.savefig(chart10_path, dpi=150, bbox_inches='tight')
        print(f"\n  Overlap chart saved: {chart10_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()
    else:
        print(f"\n  Phase 10 skipped — no multi_trades data available.")


    # ==========================================================================
    # PHASE 11: PER-MODEL TEMPORAL DEGRADATION ANALYSIS
    # Do newer folds (F7, F8) outperform older folds (F5, F6)?
    # Does each model degrade over the backtest window?
    # ==========================================================================
    if multi_trades:
        print(f"\n\n{'='*W}")
        print(f"  PHASE 11: MODEL TEMPORAL DEGRADATION ANALYSIS")
        print(f"  Does training on more recent data produce better models?")
        print(f"  Does each model degrade over the backtest window?")
        print(f"{'='*W}")

        df_temporal = pd.DataFrame(multi_trades)
        df_temporal['Date_dt'] = pd.to_datetime(df_temporal['Date'])
        df_temporal_filled = df_temporal[df_temporal['Filled'] == True].copy() if 'Filled' in df_temporal.columns else df_temporal.copy()

        model_names_t = sorted(df_temporal_filled['Model_Source'].unique())

        # --- 11a: Per-model aggregate stats ---
        print(f"\n  --- 11a: Per-Model Aggregate Performance ---\n")
        print(f"  {'Model':<14} {'Signals':>8} {'Filled':>7} {'AvgPnL':>8} {'WinRate':>8} "
              f"{'CumPnL':>8} {'Sharpe':>7} {'AvgProb':>8} {'Tickers':>8}")
        print(f"  {'─'*90}")

        model_agg = {}
        for mn in model_names_t:
            sub = df_temporal[df_temporal['Model_Source'] == mn]
            sub_f = df_temporal_filled[df_temporal_filled['Model_Source'] == mn]
            n_total = len(sub)
            n_filled = len(sub_f)
            if n_filled == 0:
                model_agg[mn] = {'avg_pnl': 0, 'cum_pnl': 0, 'win_rate': 0,
                                 'sharpe': 0, 'n_filled': 0, 'n_total': 0}
                continue
            pnls = sub_f['PnL'].values
            avg_pnl = float(np.mean(pnls))
            cum_pnl = float(np.sum(pnls))
            win_rate = float(np.mean(pnls > 0))
            std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 0
            sharpe = (avg_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0
            avg_prob = float(sub_f['Prob'].mean())
            n_tickers = sub_f['Ticker'].nunique()

            parts = mn.split()
            family = parts[0] if parts else ''
            fold = int(parts[1].replace('F', '')) if len(parts) > 1 else 0

            model_agg[mn] = {
                'avg_pnl': avg_pnl, 'cum_pnl': cum_pnl, 'win_rate': win_rate,
                'sharpe': sharpe, 'n_filled': n_filled, 'n_total': n_total,
                'avg_prob': avg_prob, 'n_tickers': n_tickers,
                'family': family, 'fold': fold,
            }
            print(f"  {mn:<14} {n_total:>8} {n_filled:>7} {avg_pnl:>+7.2%} "
                  f"{win_rate:>7.0%} {cum_pnl:>+7.2%} {sharpe:>7.1f} "
                  f"{avg_prob:>7.3f} {n_tickers:>8}")

        # --- 11b: Fold-age analysis (within each family) ---
        print(f"\n  --- 11b: Fold-Age Analysis (newer fold = more recent training data) ---\n")

        families_in_data = sorted(set(m['family'] for m in model_agg.values() if m.get('family')))
        for family in families_in_data:
            fam_models = [(mn, m) for mn, m in model_agg.items()
                          if m.get('family') == family and m['n_filled'] > 0]
            fam_models.sort(key=lambda x: x[1]['fold'])
            if len(fam_models) < 2:
                continue
            print(f"  {family}:")
            folds = [m['fold'] for _, m in fam_models]
            pnls = [m['avg_pnl'] for _, m in fam_models]
            for mn, m in fam_models:
                trend_marker = ''
                print(f"    F{m['fold']}: avg={m['avg_pnl']:+.2%}, "
                      f"win={m['win_rate']:.0%}, n={m['n_filled']}, "
                      f"sharpe={m['sharpe']:.1f}")
            if len(folds) >= 2:
                fold_corr = float(np.corrcoef(folds, pnls)[0, 1]) if np.std(pnls) > 0 else 0
                newest = fam_models[-1]
                oldest = fam_models[0]
                delta = newest[1]['avg_pnl'] - oldest[1]['avg_pnl']
                print(f"    Fold-PnL correlation: r={fold_corr:+.3f} "
                      f"({'newer=better' if fold_corr > 0.1 else 'newer=worse' if fold_corr < -0.1 else 'no trend'})")
                print(f"    Newest vs oldest delta: {delta:+.2%} "
                      f"(F{newest[1]['fold']} vs F{oldest[1]['fold']})")
            print()

        # --- 11c: Per-model daily PnL over time (degradation within window) ---
        print(f"  --- 11c: Per-Model Daily Performance Over Time ---\n")

        all_bt_dates = sorted(df_temporal_filled['Date'].unique())

        model_daily_pnl = {}
        for mn in model_names_t:
            sub = df_temporal_filled[df_temporal_filled['Model_Source'] == mn]
            daily = sub.groupby('Date')['PnL'].agg(['mean', 'count', 'sum']).reset_index()
            daily.columns = ['Date', 'avg_pnl', 'n_trades', 'total_pnl']
            model_daily_pnl[mn] = daily

        print(f"  {'Date':<12}", end="")
        for mn in model_names_t:
            label = mn.replace(' ', '')
            print(f"  {label:>12}", end="")
        print()
        print(f"  {'─'*14 + '─'*14*len(model_names_t)}")

        for date in all_bt_dates:
            print(f"  {date:<12}", end="")
            for mn in model_names_t:
                daily = model_daily_pnl[mn]
                day_row = daily[daily['Date'] == date]
                if len(day_row) > 0:
                    avg = day_row['avg_pnl'].iloc[0]
                    n = int(day_row['n_trades'].iloc[0])
                    print(f"  {avg:>+6.2%}({n:>2}t)", end="")
                else:
                    print(f"  {'—':>12}", end="")
            print()

        # --- 11d: Per-model rolling cumulative PnL ---
        print(f"\n  --- 11d: Per-Model Cumulative PnL ---\n")
        for mn in model_names_t:
            daily = model_daily_pnl[mn]
            if len(daily) == 0:
                continue
            cum = daily['avg_pnl'].cumsum()
            daily_vals = daily['avg_pnl'].values
            n_pos = sum(1 for v in daily_vals if v > 0)
            n_neg = sum(1 for v in daily_vals if v < 0)

            if len(daily_vals) >= 3:
                half = len(daily_vals) // 2
                first_half_avg = np.mean(daily_vals[:half])
                second_half_avg = np.mean(daily_vals[half:])
                trend = second_half_avg - first_half_avg
                trend_label = "DEGRADING" if trend < -0.005 else "IMPROVING" if trend > 0.005 else "STABLE"
            else:
                first_half_avg = np.mean(daily_vals)
                second_half_avg = first_half_avg
                trend = 0
                trend_label = "TOO FEW DAYS"

            print(f"  {mn:<14} cum={cum.iloc[-1]:+.2%}, "
                  f"days +{n_pos}/-{n_neg}, "
                  f"1st half avg={first_half_avg:+.2%}, "
                  f"2nd half avg={second_half_avg:+.2%}, "
                  f"trend={trend:+.2%} [{trend_label}]")

        # --- 11e: Cross-family newest-vs-oldest summary ---
        print(f"\n  --- 11e: OVERALL — Does More Recent Training Data Help? ---\n")

        oldest_pnls = []
        newest_pnls = []
        for family in families_in_data:
            fam_models = [(mn, m) for mn, m in model_agg.items()
                          if m.get('family') == family and m['n_filled'] > 0]
            fam_models.sort(key=lambda x: x[1]['fold'])
            if len(fam_models) < 2:
                continue
            oldest_pnls.append(fam_models[0][1]['avg_pnl'])
            newest_pnls.append(fam_models[-1][1]['avg_pnl'])

        if oldest_pnls and newest_pnls:
            avg_oldest = np.mean(oldest_pnls)
            avg_newest = np.mean(newest_pnls)
            delta_overall = avg_newest - avg_oldest
            print(f"  Avg across families:")
            print(f"    Oldest fold avg PnL:  {avg_oldest:+.2%}")
            print(f"    Newest fold avg PnL:  {avg_newest:+.2%}")
            print(f"    Delta (newest-oldest): {delta_overall:+.2%}")
            if delta_overall > 0.002:
                print(f"    -> NEWER DATA HELPS: retraining on recent data adds {delta_overall:+.2%} per trade")
                print(f"    -> Supports frequent retraining (every 4-6 weeks)")
            elif delta_overall < -0.002:
                print(f"    -> NEWER DATA HURTS: older folds outperform by {-delta_overall:+.2%}")
                print(f"    -> Retraining may not be needed; older patterns still valid")
            else:
                print(f"    -> INCONCLUSIVE: no meaningful difference between old and new folds")
                print(f"    -> Consider extending backtest window for more data points")

        # --- 11f: Degradation within window (slope of daily PnL) ---
        print(f"\n  --- 11f: Within-Window Degradation (is each model getting worse over time?) ---\n")
        print(f"  {'Model':<14} {'Slope':>9} {'R²':>6} {'Verdict':<14} "
              f"{'Days':>5} {'LastDay':>12}")
        print(f"  {'─'*70}")

        for mn in model_names_t:
            daily = model_daily_pnl[mn]
            if len(daily) < 3:
                print(f"  {mn:<14} {'N/A':>9} {'N/A':>6} {'TOO FEW DAYS':<14} "
                      f"{len(daily):>5}")
                continue
            x = np.arange(len(daily), dtype=np.float64)
            y = daily['avg_pnl'].values.astype(np.float64)
            slope, intercept = np.polyfit(x, y, 1)
            y_hat = slope * x + intercept
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if slope < -0.002 and r2 > 0.1:
                verdict = "DEGRADING"
            elif slope > 0.002 and r2 > 0.1:
                verdict = "IMPROVING"
            else:
                verdict = "STABLE"

            last_date = daily['Date'].iloc[-1]
            print(f"  {mn:<14} {slope:>+8.4f} {r2:>5.2f} {verdict:<14} "
                  f"{len(daily):>5} {last_date:>12}")

        # --- Charts ---
        n_chart_models = len(model_names_t)
        fig11, axes11 = plt.subplots(2, 2, figsize=(22, 16))
        fig11.suptitle(f'Phase 11: Model Temporal Degradation Analysis\n'
                       f'{BACKTEST_START} → {BACKTEST_END} | 9 Models',
                       fontsize=14, fontweight='bold')

        model_colors_11 = {}
        cmap = plt.cm.tab10
        for i, mn in enumerate(model_names_t):
            model_colors_11[mn] = cmap(i / max(n_chart_models - 1, 1))

        # Chart 1: Per-model cumulative equity curves
        ax = axes11[0, 0]
        for mn in model_names_t:
            daily = model_daily_pnl[mn]
            if len(daily) == 0:
                continue
            dates_dt = pd.to_datetime(daily['Date'])
            cum = daily['avg_pnl'].cumsum() * 100
            parts = mn.split()
            fold = int(parts[1].replace('F', '')) if len(parts) > 1 else 0
            lw = 2.5 if fold >= 7 else 1.2
            ls = '-' if fold >= 7 else '--'
            ax.plot(dates_dt, cum, color=model_colors_11[mn], linewidth=lw,
                    linestyle=ls, label=mn, alpha=0.85)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.set_ylabel('Cumulative Avg PnL (%)')
        ax.set_title('Per-Model Equity Curves\n(solid/thick = newest folds F7+F8)')
        ax.legend(fontsize=7, ncol=2, loc='best')
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Chart 2: Fold-age vs avg PnL (scatter, grouped by family)
        ax = axes11[0, 1]
        family_markers = {'L5-1': 'o', 'L5-2': 's', 'MR-v2': 'D'}
        family_colors_11 = {'L5-1': '#1565C0', 'L5-2': '#2E7D32', 'MR-v2': '#D81B60'}
        for mn, m in model_agg.items():
            if m['n_filled'] == 0:
                continue
            fam = m.get('family', '')
            fold = m.get('fold', 0)
            marker = family_markers.get(fam, 'o')
            color = family_colors_11.get(fam, '#333')
            ax.scatter(fold, m['avg_pnl'] * 100, s=120 + m['n_filled'] * 8,
                      marker=marker, color=color, alpha=0.8, zorder=5,
                      edgecolors='black', linewidth=0.5)
            ax.annotate(mn, (fold, m['avg_pnl'] * 100),
                       fontsize=7, xytext=(6, 6), textcoords='offset points')

        for fam in families_in_data:
            fam_pts = [(m['fold'], m['avg_pnl'] * 100)
                       for m in model_agg.values()
                       if m.get('family') == fam and m['n_filled'] > 0]
            if len(fam_pts) >= 2:
                fam_pts.sort()
                xs, ys = zip(*fam_pts)
                ax.plot(xs, ys, '--', color=family_colors_11.get(fam, '#333'),
                        alpha=0.4, linewidth=1.5)

        ax.set_xlabel('Fold Number (higher = more recent training data)')
        ax.set_ylabel('Avg PnL per Trade (%)')
        ax.set_title('Fold Age vs Performance\n(size = # trades; dashed = family trend)')
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker=family_markers[f],
                                  color=family_colors_11[f], linestyle='None',
                                  markersize=10, label=f)
                           for f in families_in_data if f in family_markers]
        ax.legend(handles=legend_elements, fontsize=9)
        ax.grid(alpha=0.3)

        # Chart 3: First-half vs second-half per model (degradation check)
        ax = axes11[1, 0]
        mn_labels = []
        first_halves = []
        second_halves = []
        for mn in model_names_t:
            daily = model_daily_pnl[mn]
            if len(daily) < 2:
                continue
            vals = daily['avg_pnl'].values
            half = len(vals) // 2
            first_halves.append(np.mean(vals[:half]) * 100)
            second_halves.append(np.mean(vals[half:]) * 100)
            mn_labels.append(mn)

        if mn_labels:
            y_pos = np.arange(len(mn_labels))
            bar_h = 0.35
            ax.barh(y_pos - bar_h/2, first_halves, bar_h, label='1st Half',
                    color='#1565C0', alpha=0.8)
            ax.barh(y_pos + bar_h/2, second_halves, bar_h, label='2nd Half',
                    color='#E65100', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(mn_labels, fontsize=8)
            for i in range(len(mn_labels)):
                delta = second_halves[i] - first_halves[i]
                arrow = '↑' if delta > 0.1 else '↓' if delta < -0.1 else '→'
                ax.annotate(f" {arrow}{delta:+.1f}%",
                           xy=(max(first_halves[i], second_halves[i]), y_pos[i]),
                           fontsize=7, va='center')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Avg Daily PnL (%)')
        ax.set_title('1st Half vs 2nd Half of Backtest Window\n(orange worse = degradation)')
        ax.legend(fontsize=9)
        ax.grid(axis='x', alpha=0.3)

        # Chart 4: Family-level fold comparison (bar chart)
        ax = axes11[1, 1]
        bar_data = {}
        for mn, m in model_agg.items():
            if m['n_filled'] == 0:
                continue
            fam = m.get('family', '')
            fold = m.get('fold', 0)
            bar_data.setdefault(fam, []).append((fold, m['avg_pnl'] * 100, m['n_filled']))

        x_offset = 0
        x_ticks = []
        x_labels = []
        for fam in sorted(bar_data.keys()):
            entries = sorted(bar_data[fam], key=lambda e: e[0])
            for fold, pnl, n in entries:
                color = family_colors_11.get(fam, '#333')
                alpha = 0.4 + 0.2 * (fold - min(e[0] for e in entries))
                bar = ax.bar(x_offset, pnl, color=color, alpha=min(alpha, 1.0),
                             edgecolor='black', linewidth=0.5, width=0.7)
                ax.text(x_offset, pnl + (0.1 if pnl >= 0 else -0.3),
                        f"n={n}", ha='center', fontsize=7)
                x_ticks.append(x_offset)
                x_labels.append(f"{fam}\nF{fold}")
                x_offset += 1
            x_offset += 0.5

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('Avg PnL per Trade (%)')
        ax.set_title('Per-Family Fold Comparison\n(darker = newer fold)')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart11_path = '/tmp/backtest_st_short_temporal_degradation_9model.png'
        plt.savefig(chart11_path, dpi=150, bbox_inches='tight')
        print(f"\n  Temporal degradation chart saved: {chart11_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()
    else:
        print(f"\n  Phase 11 skipped — no multi_trades data available.")


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
print(f"  SHORT PRIORITIZATION COMPARISON COMPLETE")
print(f"{'='*W}")
