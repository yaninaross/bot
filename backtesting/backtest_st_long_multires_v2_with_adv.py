# %% ── ST LONG — 9-Model Trade Prioritization: MR-v2-ADV F8 replaces MR-v2 F8 ─
#
# Same 9-model framework, but MR-v2 F8 (33 features) is REPLACED by
# MR-v2-ADV F8 (45 features = 33 baseline + 12 advanced microstructure).
#
# 9 models:
#   L5-1 F5/F6/F7  (standard SpikeTransformer, 33 features)
#   ASL6 F5/F6/F7  (standard SpikeTransformer, 33 features)
#   MR-v2 F6/F7    (MultiResSpikeTransformer, 33 features)
#   MR-v2-ADV F8   (MultiResSpikeTransformer, 45 features — 12 Lee-Ready micro)
#
# Three model objects: _model_std (standard 33f), _model_mr (multires 33f),
# _model_adv (multires 45f). MR-v2-ADV F8 uses engineer_asset_adv() which loads
# 12 extra columns from Microstructure_Advanced parquets on GCS.
#
# LONG = price up is profit. TP hit when High >= tp_price.
# Limit entry for LONG: wait for DIP (lows[k] <= signal_price * (1 - dip_pct))
#
# Model subsets tested:
#   M_CONSENSUS     (9): all models
#   M_CONS_6OLD     (6): L5-1 F5/F6/F7 + ASL6 F5/F6/F7 (no MR-v2/ADV)
#   M_CONS_3NEW     (3): MR-v2 F6/F7 + MR-v2-ADV F8
#   M_CONS_6NEWEST  (6): L5-1 F6/F7 + ASL6 F6/F7 + MR-v2 F7 + MR-v2-ADV F8
#   M_CONS_NO_F7    (3): L5-1 F6 + ASL6 F6 + MR-v2-ADV F8 (fold 7 removed)
#   M_CONS_5NEWEST  (5): L5-1 F7 + ASL6 F7 + MR-v2 F6/F7 + MR-v2-ADV F8
#
# Fill compensation: limit_entry mode tracks no-fills; ROAC = total_pnl / n_signals.
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

BACKTEST_START   = '2026-08-18'
BACKTEST_END     = '2026-09-07'

TAIL_DAYS        = 45

SCORE_MINUTE     = 375
ENTRY_MINUTE     = 385
EXIT_TIME_MIN    = 780
SLIPPAGE_PCT     = 0.0015

BASE_PNL_THRESH  = 0.018
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
N_ASSET_F_ADV    = 45
N_REGIME_F       = 22

FAMILIES = {
    'L5-1': {
        'gcs_data':   'Algos/RA_MoE/long_L5_1_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_1_models',
        'arch': 'standard',
        'bq_versions': {
            5: 'spike_transformer_long_L5_1_fold5',
            6: 'spike_transformer_long_L5_1_fold6',
            7: 'spike_transformer_long_L5_1_fold7',
        },
    },
    'ASL6': {
        'gcs_data':   'Algos/RA_MoE/long_L5_2_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_2_asl6_models',
        'arch': 'standard',
        'bq_versions': {
            5: 'spike_transformer_long_L5_2_asl6_fold5',
            6: 'spike_transformer_long_L5_2_asl6_fold6',
            7: 'spike_transformer_long_L5_2_asl6_fold7',
        },
    },
    'MR-v2': {
        'gcs_data':   'Algos/RA_MoE/long_L5_1_multires_v2_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_1_multires_v2_models',
        'arch': 'multires',
        'bq_versions': {
            6: 'spike_transformer_long_L5_1_multires_v2_fold6',
            7: 'spike_transformer_long_L5_1_multires_v2_fold7',
        },
    },
    'MR-v2-ADV': {
        'gcs_data':   'Algos/RA_MoE/long_L5_1_multires_v2_adv_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_1_multires_v2_adv_models',
        'arch': 'multires_adv',
        'bq_versions': {
            8: 'spike_transformer_long_L5_1_multires_v2_adv_fold8',
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

MICRO_COLS_ADVANCED = [
    'LR_OFI', 'LR_VPIN', 'LR_Buy_Ratio', 'LR_Size_Imbalance',
    'PE', 'LZC', 'Tick_Volatility', 'Shock_Persist',
    'Trade_Size_CV', 'Trade_Intensity', 'Quote_Intensity', 'Spread_CV',
]

ASSET_COLS_ADV = (['Gap_Pct', 'RSI_14', 'RVOL', 'ATR_14', 'Bar_Is_Real'] +
                  [f'ROC_{h}m' for h in ROC_HORIZONS] +
                  MICRO_COLS + MICRO_COLS_ADVANCED)
ALL_COLS_ADV = ASSET_COLS_ADV + CONTEXT_COLS  # 28 + 17 = 45

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
# BQ CONFIG LOADING — Newest First across L5-1 + ASL6 + MR-v2 + MR-v2-ADV
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

    # Newest First: MR-v2-ADV F8 > MR-v2 F7 > MR-v2 F6 > L5-1/ASL6 F7 > F6 > F5
    NEWEST_FIRST_ORDER = [
        ('MR-v2-ADV', 8), ('MR-v2', 7), ('MR-v2', 6),
        ('L5-1', 7), ('ASL6', 7),
        ('L5-1', 6), ('ASL6', 6),
        ('L5-1', 5), ('ASL6', 5),
    ]

    deploy = {}
    for family, fold in NEWEST_FIRST_ORDER:
        for tk, cfg in family_fold_cfgs.get((family, fold), {}).items():
            if tk not in deploy:
                deploy[tk] = cfg

    source_counts = {}
    for tk, cfg in deploy.items():
        key = f"{cfg['family']} F{cfg['fold']}"
        source_counts.setdefault(key, []).append(tk)
    for src in sorted(source_counts):
        tickers = sorted(source_counts[src])
        print(f"    Newest-First {src}: {len(tickers)} — {', '.join(tickers)}")
    print(f"    Total: {len(deploy)} tickers (Newest First across 4 families)")
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
    local_path = f"/tmp/_bt_long_mrv2_{file_ticker}_master.parquet"
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
    """Engineer features for MR-v2-ADV: baseline 33 + 12 advanced micro = 45."""
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is None:
        return None
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"Data/Inference/Microstructure_Advanced/{file_ticker}_1m_advanced_micro.parquet"
    local_path = f"/tmp/_bt_long_adv_{file_ticker}_adv_micro.parquet"
    try:
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
        df_adv = pd.read_parquet(local_path)
        os.remove(local_path)
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
        avail = [c for c in MICRO_COLS_ADVANCED if c in df_adv.columns]
        if avail:
            df_feat = df_feat.join(df_adv[avail], how='left')
    except Exception as e:
        print(f"      [ADV] Failed to load advanced micro for {ticker}: {e}")
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
# MODEL ARCHITECTURE (SpikeTransformer + MultiResSpikeTransformer)
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
_model_adv = None
_fold_weights = {}
_current_key_std = None
_current_key_mr = None
_current_key_adv = None
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
    if arch == 'multires_adv':
        return _model_adv
    elif arch == 'multires':
        return _model_mr
    return _model_std

def _ensure_fold_loaded(family, fold_id):
    global _current_key_std, _current_key_mr, _current_key_adv
    key = (family, fold_id)
    arch = FAMILIES[family].get('arch', 'standard')
    if arch == 'multires_adv':
        if _current_key_adv != key:
            _load_keras3_weights(_model_adv, _fold_weights[key])
            _current_key_adv = key
    elif arch == 'multires':
        if _current_key_mr != key:
            _load_keras3_weights(_model_mr, _fold_weights[key])
            _current_key_mr = key
    else:
        if _current_key_std != key:
            _load_keras3_weights(_model_std, _fold_weights[key])
            _current_key_std = key

def load_models_and_meta(deploy_configs, all_model_configs=None):
    global _model_std, _model_mr, _model_adv, _fold_weights
    global _current_key_std, _current_key_mr, _current_key_adv, _family_meta
    all_cfgs = list(deploy_configs.values())
    if all_model_configs:
        for cfg_list in all_model_configs.values():
            all_cfgs.extend(cfg_list)
    needed_families = sorted(set(cfg['family'] for cfg in all_cfgs))
    for family_name in needed_families:
        fam = FAMILIES[family_name]
        local_meta = f'/tmp/ra_moe_long_{family_name}_meta.npz'
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
    need_adv = any(FAMILIES[f].get('arch', 'standard') == 'multires_adv' for f in needed_families)

    if need_std:
        _model_std = SpikeTransformer(name='spike_transformer')
        dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_F))
        dummy_reg = tf.zeros((1, N_REGIME_F))
        _model_std({'seq_input': dummy_seq, 'regime_input': dummy_reg})
        print(f"   Standard SpikeTransformer (33f): {_model_std.count_params():,} params")

    if need_mr:
        _model_mr = MultiResSpikeTransformer(name='multires_spike_transformer')
        dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_F))
        dummy_15m = tf.zeros((1, TIME_STEPS_15M, N_LOWRES_F))
        dummy_reg = tf.zeros((1, N_REGIME_F))
        _model_mr({'seq_input': dummy_seq, 'seq_15m_input': dummy_15m, 'regime_input': dummy_reg})
        print(f"   MultiResSpikeTransformer (33f): {_model_mr.count_params():,} params")

    if need_adv:
        _model_adv = MultiResSpikeTransformer(name='multires_adv')
        dummy_seq_45 = tf.zeros((1, TIME_STEPS, N_ASSET_F_ADV))
        dummy_15m = tf.zeros((1, TIME_STEPS_15M, N_LOWRES_F))
        dummy_reg = tf.zeros((1, N_REGIME_F))
        _model_adv({'seq_input': dummy_seq_45, 'seq_15m_input': dummy_15m, 'regime_input': dummy_reg})
        print(f"   MultiResSpikeTransformer ADV (45f): {_model_adv.count_params():,} params")

    needed_keys = sorted(set((cfg['family'], cfg['fold']) for cfg in all_cfgs))
    for family_name, fold_id in needed_keys:
        fam = FAMILIES[family_name]
        local_dir = f'/tmp/ra_moe_long_{family_name}_models/fold_{fold_id}'
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

    for fk in needed_keys:
        fam_name, fid = fk
        arch = FAMILIES[fam_name].get('arch', 'standard')
        model = _get_model_for_family(fam_name)
        _load_keras3_weights(model, _fold_weights[fk])
        if arch == 'multires_adv':
            _current_key_adv = fk
        elif arch == 'multires':
            _current_key_mr = fk
        else:
            _current_key_std = fk
        break
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
# NUMBA TRADE SIMULATION — LONG DIRECTION
# TP hit when High >= tp_price (price rises to target)
# SL hit when Low <= sl_price (price drops against us)
# EOD PnL = (exit - entry) / entry
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
# SCORING + TRADE SIM — enriched with signal metadata for prioritization
# ==============================================================================
def score_and_backtest_ticker(ticker, cfg, df_feat, regime_df, df_15m=None):
    family = cfg['family']
    scaling_stats = _family_meta[family]['scaling_stats']
    regime_stats = _family_meta[family]['regime_stats']
    arch = FAMILIES[family].get('arch', 'standard')
    is_multires = arch in ('multires', 'multires_adv')
    is_adv = arch == 'multires_adv'

    feature_cols = ALL_COLS_ADV if is_adv else ALL_COLS
    df_scaled = apply_scaling(df_feat, scaling_stats, feature_cols)

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
        return [], {'n_bars': 0, 'max_prob': 0, 'mean_prob': 0, 'median_prob': 0,
                    'p90_prob': 0, 'thresh': cfg['prob_thresh'], 'n_above': 0}

    fold_id = cfg['fold']
    _ensure_fold_loaded(family, fold_id)
    model = _get_model_for_family(family)

    XA = df_scaled[feature_cols].values.astype(np.float32)
    h_idx = cfg['horizon']
    signal_dates = []
    signal_meta = {}
    all_probs = []

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
        all_probs.append(prob)
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

    prob_diag = {
        'n_bars': len(all_probs),
        'max_prob': max(all_probs) if all_probs else 0,
        'mean_prob': np.mean(all_probs) if all_probs else 0,
        'median_prob': np.median(all_probs) if all_probs else 0,
        'p90_prob': np.percentile(all_probs, 90) if all_probs else 0,
        'thresh': cfg['prob_thresh'],
        'n_above': sum(1 for p in all_probs if p >= cfg['prob_thresh']),
    }

    if not signal_dates:
        return [], prob_diag

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
        return [], prob_diag

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
    return trades, prob_diag


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
print(f"  ST LONG — 9-MODEL TRADE PRIORITIZATION (MR-v2-ADV F8 replaces MR-v2 F8)")
print(f"  {BACKTEST_START} -> {BACKTEST_END} | MAX_TRADES = {MAX_TRADES}")
print(f"  Score: {SCORE_MINUTE//60}:{SCORE_MINUTE%60:02d} PT | Entry: {ENTRY_MINUTE//60}:{ENTRY_MINUTE%60:02d} PT | "
      f"Exit: {EXIT_TIME_MIN//60}:{EXIT_TIME_MIN%60:02d} PT")
print(f"  Slippage: {2*SLIPPAGE_PCT:.2%} RT | Deploy threshold: >{BASE_PNL_THRESH:.1%}")
print(f"{'='*W}")

print(f"\n  Loading deploy configs from BQ (Newest First, 4 families)...")
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

_bt_tickers = sorted(DEPLOY_CONFIGS.keys())
print(f"  Building universe returns for regime features ({len(_bt_tickers)} tickers)...")
_univ_rets = {}
for _t in _bt_tickers:
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
# PHASE 1: Score all tickers, simulate all trades
# ==============================================================================
print(f"\n  Phase 1: Scoring and simulating ALL {len(DEPLOY_TICKERS)} tickers...")
all_trades = []
all_prob_diags = {}
_15m_cache = {}
for ticker in DEPLOY_TICKERS:
    cfg = DEPLOY_CONFIGS[ticker]
    print(f"    {ticker} ({cfg['family']} F{cfg['fold']}, {cfg['mode']}, "
          f"H={HORIZON_LABELS[cfg['horizon']]})...", end="")
    arch = FAMILIES[cfg['family']].get('arch', 'standard')
    if arch == 'multires_adv':
        df_feat = engineer_asset_adv(ticker, market_df)
    else:
        df_feat = engineer_asset(ticker, market_df)
    if df_feat is None:
        print(" SKIP (load failed)")
        continue
    df_15m = None
    is_mr = arch in ('multires', 'multires_adv')
    if is_mr:
        if ticker not in _15m_cache:
            _15m_cache[ticker] = build_15min_features(df_feat)
        df_15m = _15m_cache[ticker]
    trades, prob_diag = score_and_backtest_ticker(ticker, cfg, df_feat, regime_df, df_15m=df_15m)
    all_prob_diags[ticker] = {**prob_diag, 'family': cfg['family'], 'fold': cfg['fold']}
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

# Probability diagnostics
print(f"\n\n{'='*W}")
print(f"  PROBABILITY DIAGNOSTICS — All Tickers")
print(f"{'='*W}\n")
print(f"  {'Ticker':<7} {'Family':<12} {'Bars':>5} {'Thresh':>7} {'MaxP':>7} {'P90':>7} "
      f"{'MeanP':>7} {'MedP':>7} {'Above':>6} {'Gap':>7}")
print(f"  {'─'*85}")
adv_diags = []
other_diags = []
for tk in sorted(all_prob_diags.keys()):
    d = all_prob_diags[tk]
    gap = d['max_prob'] - d['thresh']
    marker = ' <<<' if d['n_above'] == 0 and d['family'] == 'MR-v2-ADV' else ''
    line = (f"  {tk:<7} {d['family']+' F'+str(d['fold']):<12} {d['n_bars']:>5} "
            f"{d['thresh']:>6.3f} {d['max_prob']:>6.3f} {d['p90_prob']:>6.3f} "
            f"{d['mean_prob']:>6.3f} {d['median_prob']:>6.3f} {d['n_above']:>5} "
            f"{gap:>+6.3f}{marker}")
    if d['family'] == 'MR-v2-ADV':
        adv_diags.append(line)
    else:
        other_diags.append(line)
print(f"\n  — MR-v2-ADV F8 tickers —")
for line in adv_diags:
    print(line)
print(f"\n  — Other families —")
for line in other_diags:
    print(line)
n_adv_zero = sum(1 for tk, d in all_prob_diags.items()
                 if d['family'] == 'MR-v2-ADV' and d['n_above'] == 0)
n_adv_total = sum(1 for d in all_prob_diags.values() if d['family'] == 'MR-v2-ADV')
print(f"\n  ADV tickers with 0 signals: {n_adv_zero}/{n_adv_total}")


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

    print(f"  Total signals:     {len(df_all)} ({n_filled_total} filled, {n_nofill_total} no-fill)")
    print(f"  Fill rate:         {fill_rate:.0%}")
    print(f"  Active days:       {total_days}")
    print(f"  Avg signals/day:   {daily_counts.mean():.1f}")
    print(f"  Max signals/day:   {daily_counts.max()}")
    print(f"  Contested days:    {contested_days}/{total_days} "
          f"(days with >{MAX_TRADES} signals)")

    print(f"\n  {'Date':<12} {'Signals':>8} {'Contested':>10}  Tickers")
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
        if stats:
            is_random = (fn == select_random)
            rng = random.Random(SEED)
            sel_records = []
            for date in sorted(df_all['Date'].unique()):
                day = df_all[df_all['Date'] == date].to_dict('records')
                picked = fn(day, MAX_TRADES, rng)
                sel_records.extend(picked)
            if sel_records:
                sel_tmp = pd.DataFrame(sel_records)
                if 'Filled' in sel_tmp.columns:
                    filled_tmp = sel_tmp[sel_tmp['Filled'] == True]
                else:
                    filled_tmp = sel_tmp
                stats['avg_per_trade'] = float(filled_tmp['PnL'].mean()) if len(filled_tmp) > 0 else 0.0
                stats['n_filled_trades'] = len(filled_tmp)
            else:
                stats['avg_per_trade'] = 0.0
                stats['n_filled_trades'] = 0
        strategy_results[name] = stats
        strategy_daily[name] = daily
        if stats:
            print(f" EW={stats['cum_ew']:+.2%}, Sharpe={stats['sharpe']:.1f}, "
                  f"trades={stats['total_trades']:.0f}, "
                  f"avg/trade={stats['avg_per_trade']:+.2%}")
        else:
            print(" NO TRADES")

    print(f"\n  {'Strategy':<14} {'Cum EW':>8} {'Avg/Day':>8} {'Sharpe':>7} "
          f"{'Trades':>7} {'Filled':>7} {'CapUtil':>7} {'ROAC':>8} "
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
              f"{s['avg_trades_day']:>5.1f} {s['win_day_pct']:>7.0%} "
              f"{s['max_drawdown']:>6.2%}")

    best_name = ranked[0][0] if ranked else 'NONE'
    best_stats = ranked[0][1] if ranked else {}
    print(f"\n  WINNER: {best_name}")


    # ==========================================================================
    # CHARTS — Phase 1 (single-model strategies)
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f'ST LONG — Trade Prioritization (9 models, MR-v2-ADV F8 replaces MR-v2 F8)\n'
                 f'{BACKTEST_START} -> {BACKTEST_END} | MAX_TRADES = {MAX_TRADES} | '
                 f'{len(df_all)} signals, {contested_days} contested days',
                 fontsize=13, fontweight='bold')

    colors_map = {
        'ALL_TRADES': '#1565C0', 'RANDOM': '#757575',
        'PROB_RAW': '#2E7D32', 'PROB_MARGIN': '#00838F',
        'EXPECTED_VAL': '#E65100', 'VAL_B_BEST': '#6A1B9A',
        'COMPOSITE': '#C62828', 'INVERSE_PROB': '#F9A825',
    }

    ax = axes[0, 0]
    for name in STRATEGIES:
        daily = strategy_daily.get(name, [])
        if not daily:
            continue
        dates_dt = pd.to_datetime([d['Date'] for d in daily])
        cum = np.cumsum([d['EW'] for d in daily]) * 100
        dates_dt = dates_dt.insert(0, dates_dt[0] - pd.Timedelta(days=1))
        cum = np.insert(cum, 0, 0.0)
        lw = 2.5 if name in (best_name, 'ALL_TRADES') else 1.2
        ls = '-' if name != 'RANDOM' else '--'
        ax.plot(dates_dt, cum, color=colors_map.get(name, '#333'),
                linewidth=lw, linestyle=ls, label=name, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylabel('Cumulative EW PnL (%)')
    ax.set_title('Equity Curves — Single-Model Strategies')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax = axes[0, 1]
    strat_names = [n for n, _ in ranked if strategy_results.get(n)]
    cum_vals = [strategy_results[n]['cum_ew'] * 100 for n in strat_names]
    bar_colors = [colors_map.get(n, '#333') for n in strat_names]
    ax.barh(range(len(strat_names)), cum_vals, color=bar_colors, alpha=0.8)
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
    all_daily_plot = {d['Date']: d['EW'] for d in strategy_daily.get('ALL_TRADES', [])}
    top_strats = [name for name, _ in ranked[:4] if name != 'ALL_TRADES']
    if 'ALL_TRADES' not in [n for n, _ in ranked[:4]]:
        top_strats = ['ALL_TRADES'] + top_strats[:3]
    else:
        top_strats = [n for n, _ in ranked[:4]]
    for name in top_strats:
        if name == 'ALL_TRADES':
            continue
        daily = strategy_daily.get(name, [])
        if not daily:
            continue
        dates_dt = pd.to_datetime([d['Date'] for d in daily])
        diffs = [(d['EW'] - all_daily_plot.get(d['Date'], 0)) * 100 for d in daily]
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
    chart_path = '/tmp/backtest_st_long_multires_v2_with_adv.png'
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
        needs_adv = any(FAMILIES[c['family']].get('arch') == 'multires_adv' for c in available)
        if needs_adv:
            df_feat_adv = engineer_asset_adv(ticker, market_df)
        df_feat = engineer_asset(ticker, market_df)
        if df_feat is None:
            continue
        n_models_scored = 0
        for cfg in available:
            arch = FAMILIES[cfg['family']].get('arch', 'standard')
            is_adv = arch == 'multires_adv'
            df_input = df_feat_adv if (is_adv and needs_adv and df_feat_adv is not None) else df_feat
            df_15m = None
            is_mr = arch in ('multires', 'multires_adv')
            if is_mr:
                if ticker not in _15m_cache_p5:
                    _15m_cache_p5[ticker] = build_15min_features(df_input)
                df_15m = _15m_cache_p5[ticker]
            trades, _ = score_and_backtest_ticker(ticker, cfg, df_input, regime_df, df_15m=df_15m)
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

        print(f"\n  Multi-model pool: {len(df_multi)} total candidates "
              f"across {df_multi['Ticker'].nunique()} tickers, {len(multi_dates)} days")


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
                family_order = {'MR-v2-ADV': 0, 'MR-v2': 0, 'ASL6': 1, 'L5-1': 2}
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

        OLD_FAMILIES = {'L5-1', 'ASL6'}
        NEW_FAMILIES = {'MR-v2', 'MR-v2-ADV'}

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
                ('L5-1', 'F7'), ('ASL6', 'F7'),
                ('MR-v2', 'F6'), ('MR-v2', 'F7'), ('MR-v2-ADV', 'F8'),
            }
            def _is_newest(t):
                parts = t.get('Model_Source', '').split()
                return (parts[0], parts[1]) in newest_versions if len(parts) >= 2 else False
            filtered = [t for t in day_trades_multi if _is_newest(t)]
            return select_multi_consensus(filtered, max_k, rng)

        def select_consensus_6newest(day_trades_multi, max_k, rng=None):
            newest_6 = {
                ('L5-1', 'F6'), ('L5-1', 'F7'),
                ('ASL6', 'F6'), ('ASL6', 'F7'),
                ('MR-v2', 'F7'), ('MR-v2-ADV', 'F8'),
            }
            def _is_in_6(t):
                parts = t.get('Model_Source', '').split()
                return (parts[0], parts[1]) in newest_6 if len(parts) >= 2 else False
            filtered = [t for t in day_trades_multi if _is_in_6(t)]
            return select_multi_consensus(filtered, max_k, rng)

        def select_consensus_no_f7(day_trades_multi, max_k, rng=None):
            no_f7 = {
                ('L5-1', 'F6'), ('ASL6', 'F6'), ('MR-v2-ADV', 'F8'),
            }
            def _is_no_f7(t):
                parts = t.get('Model_Source', '').split()
                return (parts[0], parts[1]) in no_f7 if len(parts) >= 2 else False
            filtered = [t for t in day_trades_multi if _is_no_f7(t)]
            return select_multi_consensus(filtered, max_k, rng)

        MULTI_STRATEGIES = {
            'M_CONSENSUS':    select_multi_consensus,
            'M_CONS_6OLD':    select_consensus_6old,
            'M_CONS_3NEW':    select_consensus_3new,
            'M_CONS_6NEWEST': select_consensus_6newest,
            'M_CONS_5NEWEST': select_consensus_5newest,
            'M_CONS_NO_F7':  select_consensus_no_f7,
            'M_ENSEMBLE_AVG': select_multi_ensemble_avg,
            'M_MAX_PROB':     select_multi_max_prob,
            'M_NEWEST_FOLD':  select_multi_newest_fold,
            'M_AGREEMENT':    select_multi_agreement,
        }

        # ==================================================================
        # CONSENSUS FILTER FACTORY
        # Systematically generate MARGIN / MIN2 / M2_MARG / WFIRE variants
        # across different model subsets to find optimal consensus rules.
        # ==================================================================
        MIN_FIRE_MARGIN = 0.05

        def _make_version_filter(versions):
            def _match(t):
                parts = t.get('Model_Source', '').split()
                return (parts[0], parts[1]) in versions if len(parts) >= 2 else False
            return _match

        def _make_margin_strategy(versions):
            _match = _make_version_filter(versions)
            def _fn(day_trades_multi, max_k, rng=None):
                filtered = [t for t in day_trades_multi
                            if _match(t) and (t['Prob'] - t['Thresh']) >= MIN_FIRE_MARGIN]
                return select_multi_consensus(filtered, max_k, rng)
            return _fn

        def _make_min2_strategy(versions):
            _match = _make_version_filter(versions)
            def _fn(day_trades_multi, max_k, rng=None):
                filtered = [t for t in day_trades_multi if _match(t)]
                by_ticker = defaultdict(list)
                for t in filtered:
                    by_ticker[t['Ticker']].append(t)
                resolved = []
                for ticker, cands in by_ticker.items():
                    folds_fired = set()
                    for c in cands:
                        parts = c.get('Model_Source', '').split()
                        if len(parts) >= 2:
                            folds_fired.add(parts[1])
                    if len(folds_fired) < 2:
                        continue
                    best = _resolve_ticker_model(cands, 'max_prob')
                    best = dict(best)
                    best['N_Models'] = len(cands)
                    best['N_Folds_Fired'] = len(folds_fired)
                    resolved.append(best)
                resolved.sort(key=lambda t: (-t['N_Folds_Fired'], -t['Prob']))
                return resolved[:max_k]
            return _fn

        def _make_min2_margin_strategy(versions):
            _match = _make_version_filter(versions)
            def _fn(day_trades_multi, max_k, rng=None):
                filtered = [t for t in day_trades_multi
                            if _match(t) and (t['Prob'] - t['Thresh']) >= MIN_FIRE_MARGIN]
                by_ticker = defaultdict(list)
                for t in filtered:
                    by_ticker[t['Ticker']].append(t)
                resolved = []
                for ticker, cands in by_ticker.items():
                    folds_fired = set()
                    for c in cands:
                        parts = c.get('Model_Source', '').split()
                        if len(parts) >= 2:
                            folds_fired.add(parts[1])
                    if len(folds_fired) < 2:
                        continue
                    best = _resolve_ticker_model(cands, 'max_prob')
                    best = dict(best)
                    best['N_Models'] = len(cands)
                    best['N_Folds_Fired'] = len(folds_fired)
                    resolved.append(best)
                resolved.sort(key=lambda t: (-t['N_Folds_Fired'], -t['Prob']))
                return resolved[:max_k]
            return _fn

        def _make_wfire_strategy(versions):
            _match = _make_version_filter(versions)
            def _fn(day_trades_multi, max_k, rng=None):
                filtered = [t for t in day_trades_multi if _match(t)]
                by_ticker = defaultdict(list)
                for t in filtered:
                    by_ticker[t['Ticker']].append(t)
                resolved = []
                for ticker, cands in by_ticker.items():
                    margins = [(c['Prob'] - c['Thresh']) for c in cands if (c['Prob'] - c['Thresh']) > 0]
                    n_fires = len(margins)
                    if n_fires == 0:
                        continue
                    sum_conviction = sum(margins)
                    best = _resolve_ticker_model(cands, 'max_prob')
                    best = dict(best)
                    best['N_Models'] = n_fires
                    best['Sum_Conviction'] = sum_conviction
                    resolved.append(best)
                resolved.sort(key=lambda t: (-t['N_Models'], -t['Sum_Conviction']))
                return resolved[:max_k]
            return _fn

        # LONG-specific model groupings (9 models: L5-1 F5/F6/F7 + ASL6 F5/F6/F7 + MR-v2 F6/F7 + MR-v2-ADV F8)
        FILTER_BASE_CONFIGS = {
            'CONS_ALL':     None,
            'CONS_NO_F7':   {('L5-1', 'F6'), ('ASL6', 'F6'), ('MR-v2-ADV', 'F8')},
            'CONS_6NEWEST': {('L5-1', 'F6'), ('L5-1', 'F7'), ('ASL6', 'F6'), ('ASL6', 'F7'), ('MR-v2', 'F7'), ('MR-v2-ADV', 'F8')},
            'CONS_5NEWEST': {('L5-1', 'F7'), ('ASL6', 'F7'), ('MR-v2', 'F6'), ('MR-v2', 'F7'), ('MR-v2-ADV', 'F8')},
            'MRV2_ALL':     {('MR-v2', 'F6'), ('MR-v2', 'F7'), ('MR-v2-ADV', 'F8')},
            'OLD_ONLY':     {('L5-1', 'F5'), ('L5-1', 'F6'), ('L5-1', 'F7'), ('ASL6', 'F5'), ('ASL6', 'F6'), ('ASL6', 'F7')},
        }

        FILTER_VARIANTS = {}
        for base_name, versions in FILTER_BASE_CONFIGS.items():
            if versions is None:
                all_versions = {
                    (fam, f'F{fold}')
                    for fam in ('L5-1', 'ASL6')
                    for fold in range(5, 8)
                } | {
                    ('MR-v2', 'F6'), ('MR-v2', 'F7'), ('MR-v2-ADV', 'F8'),
                }
                versions = all_versions
            FILTER_VARIANTS[f'X_{base_name}_MARGIN']  = _make_margin_strategy(versions)
            FILTER_VARIANTS[f'X_{base_name}_MIN2']    = _make_min2_strategy(versions)
            FILTER_VARIANTS[f'X_{base_name}_M2_MARG'] = _make_min2_margin_strategy(versions)
            FILTER_VARIANTS[f'X_{base_name}_WFIRE']   = _make_wfire_strategy(versions)

        MULTI_STRATEGIES.update(FILTER_VARIANTS)


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
            if 'Filled' in sel_df.columns:
                filled_trades = sel_df[sel_df['Filled'] == True]
            else:
                filled_trades = sel_df
            stats['avg_per_trade'] = float(filled_trades['PnL'].mean()) if len(filled_trades) > 0 else 0.0
            stats['n_filled_trades'] = len(filled_trades)
            multi_strategy_results[name] = stats
            multi_strategy_daily[name] = daily_ew
            cu_str = f", CapUtil={stats.get('cap_util', 1.0):.0%}" if stats.get('cap_util', 1.0) < 1.0 else ""
            print(f"  {name}: EW={stats['cum_ew']:+.2%}, "
                  f"Sharpe={stats['sharpe']:.1f}, trades={stats['total_trades']:.0f}, "
                  f"avg/trade={stats['avg_per_trade']:+.2%}"
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

        print(f"  {'Strategy':<20} {'Type':<7} {'Cum EW':>8} {'Avg/Tr':>8} "
              f"{'Sharpe':>7} {'Trades':>7} {'Filled':>7} {'CapUtil':>7} "
              f"{'ROAC':>8} {'WinDay%':>8}")
        print(f"  {'─'*W}")

        for name, s in combined_ranked:
            if not s:
                continue
            stype = 'MULTI' if name.startswith('M_') else 'SINGLE'
            avg_tr = s.get('avg_per_trade', 0)
            deploy_marker = ' **' if avg_tr >= 0.018 else ''
            print(f"  {name:<20} {stype:<7} {s['cum_ew']:>+7.2%} "
                  f"{avg_tr:>+7.2%} {s['sharpe']:>7.1f} "
                  f"{s['total_trades']:>7.0f} "
                  f"{s.get('total_filled', s['total_trades']):>7.0f} "
                  f"{s.get('cap_util', 1.0):>6.0%} "
                  f"{s.get('avg_daily_roac', s['avg_daily']):>+7.2%} "
                  f"{s['win_day_pct']:>7.0%}{deploy_marker}")

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
            print(f"  Multi vs Single:    {delta:+.2%} ({winner_type} wins)")

        # KEY COMPARISON: 3 NEW (MR-v2 F6/F7 + MR-v2-ADV F8) vs 6 OLD (L5-1 + ASL6)
        print(f"\n  ── KEY COMPARISON: 3 NEW (MR-v2+ADV) vs 6 OLD (L5-1 + ASL6) ──")
        if 'M_CONS_3NEW' in multi_strategy_results and 'M_CONS_6OLD' in multi_strategy_results:
            s_new = multi_strategy_results['M_CONS_3NEW']
            s_old = multi_strategy_results['M_CONS_6OLD']
            if s_new and s_old:
                delta_ew = s_new.get('cum_ew', 0) - s_old.get('cum_ew', 0)
                delta_sh = s_new.get('sharpe', 0) - s_old.get('sharpe', 0)
                winner = "3 NEW (MR-v2+ADV)" if delta_ew > 0 else "6 OLD (L5-1+ASL6)"
                print(f"  3 NEW (MR-v2+ADV): EW={s_new['cum_ew']:+.2%}, Sh={s_new['sharpe']:.1f}, "
                      f"trades={s_new['total_trades']:.0f}")
                print(f"  6 OLD (L5-1+ASL6): EW={s_old['cum_ew']:+.2%}, Sh={s_old['sharpe']:.1f}, "
                      f"trades={s_old['total_trades']:.0f}")
                print(f"  Delta:             EW={delta_ew:+.2%}, Sharpe={delta_sh:+.1f} → {winner} wins")
            else:
                print(f"  One or both subsets had no trades.")
        else:
            print(f"  Subset strategies not evaluated (no BQ configs).")


        # ==================================================================
        # PHASE 7: MAX_TRADES SWEEP
        # ==================================================================
        SWEEP_RANGE = range(1, 10)
        TOP_STRATS = {
            'M_CONSENSUS':    ('multi',  select_multi_consensus),
            'M_CONS_6NEWEST': ('multi',  select_consensus_6newest),
            'M_CONS_5NEWEST': ('multi',  select_consensus_5newest),
            'M_CONS_NO_F7':  ('multi',  select_consensus_no_f7),
            'M_CONS_6OLD':    ('multi',  select_consensus_6old),
            'M_CONS_3NEW':    ('multi',  select_consensus_3new),
            'M_AGREEMENT':    ('multi',  select_multi_agreement),
        }
        for vname in sorted(FILTER_VARIANTS.keys()):
            TOP_STRATS[vname] = ('multi', FILTER_VARIANTS[vname])

        print(f"\n\n{'='*W}")
        print(f"  PHASE 7: MAX_TRADES SWEEP (1 → 9) — CONSENSUS + FILTER VARIANTS ({len(TOP_STRATS)} strategies)")
        print(f"{'='*W}\n")

        sweep_results = {}
        for sname, (stype, fn) in TOP_STRATS.items():
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
                        'Date': date, 'EW': f_pnl, 'ROAC': roac_v,
                        'N': n_sel, 'N_Filled': n_f,
                    })
                s_stats = compute_stats(daily_ew)
                if s_stats:
                    if 'Filled' in sel_df.columns:
                        f_trades = sel_df[sel_df['Filled'] == True]
                    else:
                        f_trades = sel_df
                    s_stats['avg_per_trade'] = float(f_trades['PnL'].mean()) if len(f_trades) > 0 else 0.0
                sweep_results[sname][max_k] = s_stats

        print(f"  {'Strategy':<16}", end="")
        for k in SWEEP_RANGE:
            print(f"  {'K='+str(k):>10}", end="")
        print()
        print(f"  {'─'*W}")

        metrics_to_sweep = [
            ('Cum EW', 'cum_ew', lambda v: f"{v:>+9.2%}"),
            ('Avg/Trade', 'avg_per_trade', lambda v: f"{v:>+9.2%}"),
            ('Sharpe', 'sharpe', lambda v: f"{v:>10.1f}"),
            ('Avg/Day', 'avg_daily', lambda v: f"{v:>+9.2%}"),
            ('MaxDD', 'max_drawdown', lambda v: f"{v:>9.2%}"),
            ('Trades', 'total_trades', lambda v: f"{v:>10.0f}"),
            ('WinDay%', 'win_day_pct', lambda v: f"{v:>9.0%}"),
        ]

        for metric_name, metric_key, fmt_fn in metrics_to_sweep:
            print(f"\n  — {metric_name} —")
            for sname in TOP_STRATS:
                vals = []
                for k in SWEEP_RANGE:
                    s = sweep_results[sname].get(k, {})
                    vals.append(s.get(metric_key, 0))
                best_k = max(SWEEP_RANGE, key=lambda kk: sweep_results[sname].get(kk, {}).get(metric_key, -999))
                print(f"  {sname:<16}", end="")
                for i, v in enumerate(vals):
                    marker = ' *' if (i + 1) == best_k and metric_key in ('cum_ew', 'sharpe') else '  '
                    print(f"{fmt_fn(v)}{marker}", end="")
                print()

        print(f"\n  (* = best K for that strategy)")

        best_overall_ew = -999
        best_overall_combo = ('', 0)
        for sname in TOP_STRATS:
            for k in SWEEP_RANGE:
                s = sweep_results[sname].get(k, {})
                ew = s.get('cum_ew', -999)
                if ew > best_overall_ew:
                    best_overall_ew = ew
                    best_overall_combo = (sname, k)
        print(f"\n  BEST by Cum EW:  {best_overall_combo[0]} @ K={best_overall_combo[1]} "
              f"({best_overall_ew:+.2%})")


        # ==================================================================
        # MULTI-MODEL TICKER COVERAGE
        # ==================================================================
        print(f"\n\n{'='*W}")
        print(f"  MULTI-MODEL TICKER COVERAGE")
        print(f"{'='*W}\n")

        print(f"  {'Ticker':<7} {'Models':>7} {'Configs':>8} "
              f"{'NF Source':<12} {'Avg Prob':>8} {'NF PnL':>7}")
        print(f"  {'─'*80}")

        for ticker in sorted(DEPLOY_TICKERS):
            n_configs = len(ALL_MODEL_CONFIGS.get(ticker, []))
            n_fired = ticker_model_counts.get(ticker, 0)
            nf_cfg = DEPLOY_CONFIGS[ticker]
            nf_src = f"{nf_cfg['family']} F{nf_cfg['fold']}"

            tk_multi = df_multi[df_multi['Ticker'] == ticker]
            avg_prob = tk_multi['Prob'].mean() if len(tk_multi) > 0 else 0

            tk_single = df_all[df_all['Ticker'] == ticker]
            nf_pnl = tk_single['PnL'].mean() if len(tk_single) > 0 else 0

            print(f"  {ticker:<7} {n_fired:>3}/{n_configs:<3} "
                  f"{n_configs:>8} {nf_src:<12} "
                  f"{avg_prob:>7.3f} {nf_pnl:>+6.2%}")


        # ==================================================================
        # COMBINED CHARTS — SINGLE + MULTI
        # ==================================================================
        multi_colors = {
            'M_CONSENSUS': '#D81B60',
            'M_CONS_6OLD': '#1E88E5',
            'M_CONS_3NEW': '#00ACC1',
            'M_CONS_6NEWEST': '#8E24AA',
            'M_CONS_5NEWEST': '#FF6F00',
            'M_CONS_NO_F7':  '#E53935',
            'M_ENSEMBLE_AVG': '#FFB300',
            'M_MAX_PROB': '#5C6BC0',
            'M_NEWEST_FOLD': '#78909C',
            'M_AGREEMENT': '#43A047',
        }
        all_colors = {**colors_map, **multi_colors}

        fig2, axes2 = plt.subplots(2, 2, figsize=(20, 14))
        fig2.suptitle(f'ST LONG — Combined: Single + Multi-Model (9 models, ADV F8)\n'
                      f'{BACKTEST_START} -> {BACKTEST_END} | MAX_TRADES = {MAX_TRADES}',
                      fontsize=13, fontweight='bold')

        ax = axes2[0, 0]
        for name in list(STRATEGIES.keys()) + list(MULTI_STRATEGIES.keys()):
            daily = all_daily.get(name, [])
            if not daily:
                continue
            dates_dt = pd.to_datetime([d['Date'] for d in daily])
            cum = np.cumsum([d['EW'] for d in daily]) * 100
            dates_dt = dates_dt.insert(0, dates_dt[0] - pd.Timedelta(days=1))
            cum = np.insert(cum, 0, 0.0)
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
        ax.legend(fontsize=5, ncol=3, loc='upper left')
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax = axes2[0, 1]
        comb_names = [n for n, _ in combined_ranked if all_results.get(n)]
        comb_vals = [all_results[n]['cum_ew'] * 100 for n in comb_names]
        comb_colors_bar = [all_colors.get(n, '#333') for n in comb_names]
        ax.barh(range(len(comb_names)), comb_vals, color=comb_colors_bar, alpha=0.8)
        ax.set_yticks(range(len(comb_names)))
        labels = [f"{'[M] ' if n.startswith('M_') else ''}{n}" for n in comb_names]
        ax.set_yticklabels(labels, fontsize=7)
        for i, (name, val) in enumerate(zip(comb_names, comb_vals)):
            s = all_results[name]
            ax.text(val + 0.2 if val >= 0 else val - 0.2,
                    i, f"Sh={s['sharpe']:.1f}", va='center', fontsize=6)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Cumulative EW PnL (%)')
        ax.set_title('Combined Ranking')
        ax.grid(axis='x', alpha=0.3)

        ax = axes2[1, 0]
        config_counts = [len(ALL_MODEL_CONFIGS.get(tk, [])) for tk in DEPLOY_TICKERS]
        fired_counts = [ticker_model_counts.get(tk, 0) for tk in DEPLOY_TICKERS]
        x_pos = range(len(DEPLOY_TICKERS))
        ax.bar(x_pos, config_counts, alpha=0.4, color='#1565C0', label='Configs available')
        ax.bar(x_pos, fired_counts, alpha=0.7, color='#2E7D32', label='Models fired')
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(DEPLOY_TICKERS, rotation=90, fontsize=5)
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
                       fontsize=5, xytext=(5, 5), textcoords='offset points')
        for name in multi_names:
            s = multi_strategy_results[name]
            ax.scatter(s['sharpe'], s['cum_ew'] * 100,
                      color=all_colors.get(name, '#333'), s=100,
                      marker='D', label=name, zorder=5)
            ax.annotate(name, (s['sharpe'], s['cum_ew']*100),
                       fontsize=5, xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Cumulative EW PnL (%)')
        ax.set_title('Sharpe vs PnL (o=single, D=multi)')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        chart2_path = '/tmp/backtest_st_long_multires_v2_with_adv_combined.png'
        plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
        print(f"\n  Combined chart saved: {chart2_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()

        # Sweep chart
        sweep_colors = {
            'M_CONSENSUS': '#D81B60', 'M_CONS_6NEWEST': '#8E24AA',
            'M_CONS_5NEWEST': '#FF6F00', 'M_CONS_NO_F7': '#E53935',
            'M_CONS_6OLD': '#1E88E5',
            'M_CONS_3NEW': '#00ACC1', 'M_AGREEMENT': '#43A047',
        }
        fig3, axes3 = plt.subplots(2, 2, figsize=(20, 14))
        fig3.suptitle(f'ST LONG — MAX_TRADES Sweep (1→9) — NEW (MR-v2+ADV) vs OLD\n'
                      f'{BACKTEST_START} → {BACKTEST_END}',
                      fontsize=13, fontweight='bold')
        ks = list(SWEEP_RANGE)

        ks_plot = [0] + ks

        ax = axes3[0, 0]
        for sname in TOP_STRATS:
            vals = [0.0] + [sweep_results[sname].get(k, {}).get('cum_ew', 0) * 100 for k in ks]
            ax.plot(ks_plot, vals, 'o-', color=sweep_colors.get(sname, '#333'), linewidth=2,
                    markersize=8, label=sname)
        ax.set_xlabel('MAX_TRADES (K)'); ax.set_ylabel('Cumulative EW PnL (%)')
        ax.set_title('Cumulative PnL vs Position Cap')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(ks_plot); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes3[0, 1]
        for sname in TOP_STRATS:
            vals = [0.0] + [sweep_results[sname].get(k, {}).get('sharpe', 0) for k in ks]
            ax.plot(ks_plot, vals, 's-', color=sweep_colors.get(sname, '#333'), linewidth=2,
                    markersize=8, label=sname)
        ax.set_xlabel('MAX_TRADES (K)'); ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe vs Position Cap')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(ks_plot); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes3[1, 0]
        for sname in TOP_STRATS:
            vals = [0.0] + [sweep_results[sname].get(k, {}).get('avg_daily', 0) * 100 for k in ks]
            ax.plot(ks_plot, vals, '^-', color=sweep_colors.get(sname, '#333'), linewidth=2,
                    markersize=8, label=sname)
        ax.set_xlabel('MAX_TRADES (K)'); ax.set_ylabel('Avg Daily EW PnL (%)')
        ax.set_title('Avg Daily PnL vs Position Cap')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(ks_plot); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes3[1, 1]
        for sname in TOP_STRATS:
            vals = [0.0] + [sweep_results[sname].get(k, {}).get('max_drawdown', 0) * 100 for k in ks]
            ax.plot(ks_plot, vals, 'D-', color=sweep_colors.get(sname, '#333'), linewidth=2,
                    markersize=8, label=sname)
        ax.set_xlabel('MAX_TRADES (K)'); ax.set_ylabel('Max Drawdown (%)')
        ax.set_title('Max Drawdown vs Position Cap')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(ks_plot); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.invert_yaxis()

        plt.tight_layout()
        chart3_path = '/tmp/backtest_st_long_multires_v2_with_adv_sweep.png'
        plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
        print(f"\n  Sweep chart saved: {chart3_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()


        # ==================================================================
        # PHASE 8: A/B — M_CONS_6NEWEST: MR-v2 F8 (33f) vs MR-v2-ADV F8 (45f)
        # The ONLY difference is the 6th model slot:
        #   Baseline: L5-1 F6/F7 + ASL6 F6/F7 + MR-v2 F7 + MR-v2 F8 (33f)
        #   ADV:      L5-1 F6/F7 + ASL6 F6/F7 + MR-v2 F7 + MR-v2-ADV F8 (45f)
        # ==================================================================
        print(f"\n\n{'='*W}")
        print(f"  PHASE 8: A/B COMPARISON — M_CONS_6NEWEST")
        print(f"  Baseline: L5-1 F6/F7 + ASL6 F6/F7 + MR-v2 F7 + MR-v2 F8 (33f)")
        print(f"  ADV:      L5-1 F6/F7 + ASL6 F6/F7 + MR-v2 F7 + MR-v2-ADV F8 (45f)")
        print(f"{'='*W}")

        # — Load MR-v2 F8 (baseline 33f) configs from BQ —
        print(f"\n  Loading MR-v2 F8 (baseline 33f) configs from BQ...")
        mrv2_f8_cfgs = {}
        if bigquery is not None:
            _bq_ab = bigquery.Client()
            _q_f8 = f"""
            SELECT ticker, experiment, tpsl_mode, prob_thresh, tp_pct, sl_pct,
                   grace_minutes, dip_pct, max_wait_minutes,
                   val_b_avg_pnl, val_b_win_rate, val_b_count
            FROM `{WF_BQ_TABLE}`
            WHERE model_version = 'spike_transformer_long_L5_1_multires_v2_fold8'
              AND survived_a = TRUE
              AND val_b_avg_pnl >= {BASE_PNL_THRESH}
            """
            _rows_f8 = _bq_ab.query(_q_f8).to_dataframe()
            if len(_rows_f8) > 0:
                _best_f8 = _rows_f8.loc[_rows_f8.groupby('ticker')['val_b_avg_pnl'].idxmax()]
                for _, _row in _best_f8.iterrows():
                    _tk = _row['ticker']
                    if _tk in EXCLUDE_TICKERS:
                        continue
                    _exp = _row['experiment']
                    _h_part = _exp.replace('spike_transformer_long_L5_1_multires_v2_fold8_', '').split('_')[0]
                    _h_idx = HORIZON_FROM_LABEL.get(_h_part, 1)
                    _mode = _row['tpsl_mode']
                    if pd.isna(_mode) or _mode == '':
                        if '_GRACE' in _exp:
                            _mode = 'grace'
                        elif '_LIMIT' in _exp:
                            _mode = 'limit_entry'
                        else:
                            _mode = 'static'
                    mrv2_f8_cfgs[_tk] = {
                        'family': 'MR-v2',
                        'fold': 8,
                        'horizon': _h_idx,
                        'prob_thresh': float(_row['prob_thresh']),
                        'tp_pct': float(_row['tp_pct']),
                        'sl_pct': float(_row['sl_pct']),
                        'mode': _mode,
                        'grace_minutes': int(_row['grace_minutes']) if pd.notna(_row['grace_minutes']) else 0,
                        'dip_pct': float(_row['dip_pct']) if pd.notna(_row['dip_pct']) else 0,
                        'max_wait': int(_row['max_wait_minutes']) if pd.notna(_row['max_wait_minutes']) else 0,
                        'val_b_avg_pnl': float(_row['val_b_avg_pnl']),
                    }
            print(f"    MR-v2 F8 baseline: {len(mrv2_f8_cfgs)} tickers — "
                  f"{', '.join(sorted(mrv2_f8_cfgs))}")

        # — Load MR-v2 F8 weights (same arch as MR-v2 F6/F7 → uses _model_mr) —
        _mrv2_f8_wdir = '/tmp/ra_moe_long_MR-v2_models/fold_8'
        _mrv2_f8_wpath = os.path.join(_mrv2_f8_wdir, 'model.weights.h5')
        if not os.path.exists(_mrv2_f8_wpath):
            os.makedirs(_mrv2_f8_wdir, exist_ok=True)
            _prefix_f8 = f"{FAMILIES['MR-v2']['gcs_models']}/fold_8"
            _bucket_f8 = gcs_client.bucket(BUCKET_NAME)
            for _blob in _bucket_f8.list_blobs(prefix=_prefix_f8):
                _fname = _blob.name.split('/')[-1]
                _blob.download_to_filename(os.path.join(_mrv2_f8_wdir, _fname))
        _fold_weights[('MR-v2', 8)] = _mrv2_f8_wpath
        print(f"    MR-v2 F8 weights cached")

        # — Score all MR-v2 F8 tickers —
        print(f"\n  Scoring MR-v2 F8 (baseline 33f) tickers...")
        _mrv2_f8_trades = []
        _15m_ab = {}
        for _tk, _cfg in sorted(mrv2_f8_cfgs.items()):
            _df_f = engineer_asset(_tk, market_df)
            if _df_f is None:
                print(f"      {_tk}: SKIP (load failed)")
                continue
            if _tk not in _15m_ab:
                _15m_ab[_tk] = build_15min_features(_df_f)
            _trades_f8, _ = score_and_backtest_ticker(
                _tk, _cfg, _df_f, regime_df, df_15m=_15m_ab[_tk])
            for _t in _trades_f8:
                _t['Model_Source'] = 'MR-v2 F8'
            _nf = sum(1 for _t in _trades_f8 if _t.get('Filled', True))
            _avg = (np.mean([_t['PnL'] for _t in _trades_f8 if _t.get('Filled', True)])
                    if _nf > 0 else 0)
            print(f"      {_tk} ({_cfg['mode']}, H={HORIZON_LABELS[_cfg['horizon']]}): "
                  f"{_nf}t avg={_avg:+.2%}")
            _mrv2_f8_trades.extend(_trades_f8)
            _master_cache.pop((_tk, TAIL_DAYS), None)
        _master_cache.clear()
        print(f"    MR-v2 F8 total: {len(_mrv2_f8_trades)} trade candidates")

        # — Build two 6NEWEST pools (5 common models + 1 different F8 slot) —
        _common_versions = {
            ('L5-1', 'F6'), ('L5-1', 'F7'),
            ('ASL6', 'F6'), ('ASL6', 'F7'),
            ('MR-v2', 'F7'),
        }
        def _is_common_model(t):
            _parts = t.get('Model_Source', '').split()
            return (len(_parts) >= 2 and
                    (_parts[0], _parts[1]) in _common_versions)

        _common_pool = [t for t in df_multi.to_dict('records')
                        if _is_common_model(t)]
        _adv_f8_pool = [t for t in df_multi.to_dict('records')
                        if t.get('Model_Source', '') == 'MR-v2-ADV F8']

        _pool_adv = _common_pool + _adv_f8_pool
        _pool_base = _common_pool + _mrv2_f8_trades

        _df_pool_adv = pd.DataFrame(_pool_adv) if _pool_adv else pd.DataFrame()
        _df_pool_base = pd.DataFrame(_pool_base) if _pool_base else pd.DataFrame()

        print(f"\n  Pool sizes:")
        print(f"    Common (5 models):   {len(_common_pool)} candidates")
        print(f"    + MR-v2-ADV F8:      {len(_adv_f8_pool)} → ADV pool = {len(_pool_adv)}")
        print(f"    + MR-v2 F8 baseline: {len(_mrv2_f8_trades)} → Base pool = {len(_pool_base)}")

        # — Run consensus on both pools at each K —
        _ab_dates = sorted(set(
            (list(_df_pool_adv['Date'].unique()) if len(_df_pool_adv) > 0 else []) +
            (list(_df_pool_base['Date'].unique()) if len(_df_pool_base) > 0 else [])
        ))

        def _run_consensus_pool(pool_df, dates, max_k):
            daily_ew = []
            for date in dates:
                day = pool_df[pool_df['Date'] == date].to_dict('records')
                if not day:
                    continue
                picked = select_multi_consensus(day, max_k)
                if not picked:
                    continue
                sel = pd.DataFrame(picked)
                n_sel = len(sel)
                if 'Filled' in sel.columns:
                    n_f = int(sel['Filled'].sum())
                    f_day = sel[sel['Filled'] == True]
                    f_pnl = f_day['PnL'].mean() if len(f_day) > 0 else 0.0
                    roac = f_day['PnL'].sum() / n_sel if n_sel > 0 else 0.0
                else:
                    n_f = n_sel
                    f_pnl = sel['PnL'].mean()
                    roac = sel['PnL'].mean()
                daily_ew.append({
                    'Date': date, 'EW': f_pnl, 'ROAC': roac,
                    'N': n_sel, 'N_Filled': n_f,
                    'Tickers': sorted(sel['Ticker'].unique().tolist()),
                })
            return daily_ew

        # K sweep for A/B
        AB_SWEEP = range(1, 8)
        _ab_sweep = {'ADV': {}, 'BASE': {}}
        for _k in AB_SWEEP:
            _d_adv = (_run_consensus_pool(_df_pool_adv, _ab_dates, _k)
                      if len(_df_pool_adv) > 0 else [])
            _d_base = (_run_consensus_pool(_df_pool_base, _ab_dates, _k)
                       if len(_df_pool_base) > 0 else [])
            _ab_sweep['ADV'][_k] = compute_stats(_d_adv)
            _ab_sweep['BASE'][_k] = compute_stats(_d_base)

        # Use K=MAX_TRADES for detailed comparison
        _daily_adv = (_run_consensus_pool(_df_pool_adv, _ab_dates, MAX_TRADES)
                      if len(_df_pool_adv) > 0 else [])
        _daily_base = (_run_consensus_pool(_df_pool_base, _ab_dates, MAX_TRADES)
                       if len(_df_pool_base) > 0 else [])
        _stats_adv = compute_stats(_daily_adv)
        _stats_base = compute_stats(_daily_base)

        # — Summary comparison at K=MAX_TRADES —
        print(f"\n  ── A/B SUMMARY at MAX_TRADES = {MAX_TRADES} ──")
        print(f"  {'Metric':<20} {'Base (F8 33f)':>16} {'ADV (F8 45f)':>16} {'Delta':>12}")
        print(f"  {'─'*70}")

        _ab_metrics = [
            ('Cum EW',     'cum_ew',          True),
            ('Avg Daily',  'avg_daily',       True),
            ('Sharpe',     'sharpe',          False),
            ('Trades',     'total_trades',    False),
            ('Filled',     'total_filled',    False),
            ('Cap Util',   'cap_util',        True),
            ('Win Day %',  'win_day_pct',     True),
            ('Max DD',     'max_drawdown',    True),
            ('ROAC',       'avg_daily_roac',  True),
        ]
        for _label, _key, _is_pct in _ab_metrics:
            _vb = _stats_base.get(_key, 0)
            _va = _stats_adv.get(_key, 0)
            _d = _va - _vb
            if _is_pct:
                print(f"  {_label:<20} {_vb:>15.2%} {_va:>15.2%} {_d:>+11.2%}")
            elif _key in ('total_trades', 'total_filled'):
                print(f"  {_label:<20} {_vb:>15.0f} {_va:>15.0f} {_d:>+11.0f}")
            else:
                print(f"  {_label:<20} {_vb:>15.1f} {_va:>15.1f} {_d:>+11.1f}")

        _winner_ab = ("MR-v2-ADV F8 (45f)"
                      if _stats_adv.get('cum_ew', 0) > _stats_base.get('cum_ew', 0)
                      else "MR-v2 F8 (33f)")
        _delta_ew = _stats_adv.get('cum_ew', 0) - _stats_base.get('cum_ew', 0)
        print(f"\n  VERDICT @ K={MAX_TRADES}: {_winner_ab} wins by "
              f"{abs(_delta_ew):.2%} Cum EW")

        # — K sweep table —
        print(f"\n  ── A/B SWEEP: K = {min(AB_SWEEP)} → {max(AB_SWEEP)} ──")
        print(f"\n  {'':>16}", end="")
        for _k in AB_SWEEP:
            print(f"  {'K='+str(_k):>10}", end="")
        print()
        print(f"  {'─'*90}")

        print(f"\n  — Cum EW —")
        for _variant in ('BASE', 'ADV'):
            _tag = 'F8 33f' if _variant == 'BASE' else 'F8 45f'
            print(f"  {_tag:<16}", end="")
            for _k in AB_SWEEP:
                _v = _ab_sweep[_variant].get(_k, {}).get('cum_ew', 0)
                print(f"  {_v:>+9.2%}", end="")
            print()
        print(f"  {'Delta':<16}", end="")
        for _k in AB_SWEEP:
            _d = (_ab_sweep['ADV'].get(_k, {}).get('cum_ew', 0) -
                  _ab_sweep['BASE'].get(_k, {}).get('cum_ew', 0))
            print(f"  {_d:>+9.2%}", end="")
        print()

        print(f"\n  — Sharpe —")
        for _variant in ('BASE', 'ADV'):
            _tag = 'F8 33f' if _variant == 'BASE' else 'F8 45f'
            print(f"  {_tag:<16}", end="")
            for _k in AB_SWEEP:
                _v = _ab_sweep[_variant].get(_k, {}).get('sharpe', 0)
                print(f"  {_v:>10.1f}", end="")
            print()
        print(f"  {'Delta':<16}", end="")
        for _k in AB_SWEEP:
            _d = (_ab_sweep['ADV'].get(_k, {}).get('sharpe', 0) -
                  _ab_sweep['BASE'].get(_k, {}).get('sharpe', 0))
            print(f"  {_d:>+10.1f}", end="")
        print()

        print(f"\n  — Trades —")
        for _variant in ('BASE', 'ADV'):
            _tag = 'F8 33f' if _variant == 'BASE' else 'F8 45f'
            print(f"  {_tag:<16}", end="")
            for _k in AB_SWEEP:
                _v = _ab_sweep[_variant].get(_k, {}).get('total_trades', 0)
                print(f"  {_v:>10.0f}", end="")
            print()

        # Best K for each variant
        _best_k_base = max(AB_SWEEP, key=lambda k: _ab_sweep['BASE'].get(k, {}).get('cum_ew', -999))
        _best_k_adv = max(AB_SWEEP, key=lambda k: _ab_sweep['ADV'].get(k, {}).get('cum_ew', -999))
        print(f"\n  Best K: Base={_best_k_base} "
              f"({_ab_sweep['BASE'][_best_k_base].get('cum_ew', 0):+.2%}), "
              f"ADV={_best_k_adv} "
              f"({_ab_sweep['ADV'][_best_k_adv].get('cum_ew', 0):+.2%})")

        # How many K values does ADV win?
        _adv_k_wins = sum(1 for _k in AB_SWEEP
                          if _ab_sweep['ADV'].get(_k, {}).get('cum_ew', 0) >
                             _ab_sweep['BASE'].get(_k, {}).get('cum_ew', 0))
        print(f"  ADV wins {_adv_k_wins}/{len(list(AB_SWEEP))} K values")

        # — Per-day comparison at K=MAX_TRADES —
        print(f"\n  ── PER-DAY COMPARISON (K={MAX_TRADES}) ──")
        print(f"  {'Date':<12} {'Base EW':>8} {'ADV EW':>8} {'Delta':>8} "
              f"{'Win':>5}  Base Tickers / ADV Tickers")
        print(f"  {'─'*W}")

        _base_by_date = {d['Date']: d for d in _daily_base}
        _adv_by_date = {d['Date']: d for d in _daily_adv}
        _ab_day_wins = {'BASE': 0, 'ADV': 0, 'TIE': 0}

        for _date in _ab_dates:
            _db = _base_by_date.get(_date)
            _da = _adv_by_date.get(_date)
            _ew_b = _db['EW'] if _db else 0
            _ew_a = _da['EW'] if _da else 0
            _delta = _ew_a - _ew_b
            if abs(_delta) < 1e-8:
                _w = 'TIE'
            elif _delta > 0:
                _w = 'ADV'
            else:
                _w = 'BASE'
            _ab_day_wins[_w] += 1
            _btk = ', '.join(_db['Tickers']) if _db else '—'
            _atk = ', '.join(_da['Tickers']) if _da else '—'
            print(f"  {_date:<12} {_ew_b:>+7.2%} {_ew_a:>+7.2%} {_delta:>+7.2%} "
                  f"{_w:>5}  {_btk} / {_atk}")

        print(f"\n  Day wins: Base={_ab_day_wins['BASE']}, "
              f"ADV={_ab_day_wins['ADV']}, Tie={_ab_day_wins['TIE']}")

        # — F8 slot signal comparison —
        print(f"\n  ── F8 SLOT SIGNAL BREAKDOWN ──")
        _adv_f8_tickers = set(t['Ticker'] for t in _adv_f8_pool)
        _base_f8_tickers = set(t['Ticker'] for t in _mrv2_f8_trades)
        _all_f8_tickers = sorted(_adv_f8_tickers | _base_f8_tickers)
        _overlap = _adv_f8_tickers & _base_f8_tickers

        print(f"  MR-v2-ADV F8 (45f): {len(_adv_f8_tickers)} — "
              f"{', '.join(sorted(_adv_f8_tickers))}")
        print(f"  MR-v2 F8 (33f):     {len(_base_f8_tickers)} — "
              f"{', '.join(sorted(_base_f8_tickers))}")
        print(f"  Overlap:            {len(_overlap)} — "
              f"{', '.join(sorted(_overlap))}")
        print(f"  ADV only:           "
              f"{', '.join(sorted(_adv_f8_tickers - _base_f8_tickers)) or '—'}")
        print(f"  Base only:          "
              f"{', '.join(sorted(_base_f8_tickers - _adv_f8_tickers)) or '—'}")

        print(f"\n  {'Ticker':<7} {'ADV sigs':>9} {'ADV avg':>9} "
              f"{'Base sigs':>10} {'Base avg':>10} {'Delta':>8}")
        print(f"  {'─'*60}")
        _adv_sum_pnl = 0; _adv_sum_n = 0
        _base_sum_pnl = 0; _base_sum_n = 0
        for _tk in _all_f8_tickers:
            _a_tr = [t for t in _adv_f8_pool
                     if t['Ticker'] == _tk and t.get('Filled', True)]
            _b_tr = [t for t in _mrv2_f8_trades
                     if t['Ticker'] == _tk and t.get('Filled', True)]
            _a_n = len(_a_tr); _b_n = len(_b_tr)
            _a_avg = np.mean([t['PnL'] for t in _a_tr]) if _a_n > 0 else 0
            _b_avg = np.mean([t['PnL'] for t in _b_tr]) if _b_n > 0 else 0
            _d_avg = _a_avg - _b_avg if (_a_n > 0 and _b_n > 0) else 0
            _marker = ''
            if _tk in _overlap and _a_n > 0 and _b_n > 0:
                _marker = ' ← overlap'
            elif _tk in (_adv_f8_tickers - _base_f8_tickers):
                _marker = ' (ADV only)'
            elif _tk in (_base_f8_tickers - _adv_f8_tickers):
                _marker = ' (Base only)'
            print(f"  {_tk:<7} {_a_n:>9} {_a_avg:>+8.2%} "
                  f"{_b_n:>10} {_b_avg:>+9.2%} {_d_avg:>+7.2%}{_marker}")
            _adv_sum_pnl += sum(t['PnL'] for t in _a_tr)
            _adv_sum_n += _a_n
            _base_sum_pnl += sum(t['PnL'] for t in _b_tr)
            _base_sum_n += _b_n
        _adv_avg_all = _adv_sum_pnl / _adv_sum_n if _adv_sum_n > 0 else 0
        _base_avg_all = _base_sum_pnl / _base_sum_n if _base_sum_n > 0 else 0
        print(f"  {'─'*60}")
        print(f"  {'TOTAL':<7} {_adv_sum_n:>9} {_adv_avg_all:>+8.2%} "
              f"{_base_sum_n:>10} {_base_avg_all:>+9.2%} "
              f"{_adv_avg_all - _base_avg_all:>+7.2%}")

        # — A/B equity curve + daily delta chart —
        fig_ab, axes_ab = plt.subplots(2, 2, figsize=(20, 14))
        fig_ab.suptitle(
            f'A/B: M_CONS_6NEWEST — MR-v2 F8 (33f) vs MR-v2-ADV F8 (45f)\n'
            f'{BACKTEST_START} → {BACKTEST_END} | MAX_TRADES = {MAX_TRADES}',
            fontsize=13, fontweight='bold')

        # Equity curves at K=MAX_TRADES
        ax = axes_ab[0, 0]
        if _daily_base:
            _bd = pd.to_datetime([d['Date'] for d in _daily_base])
            _bc = np.cumsum([d['EW'] for d in _daily_base]) * 100
            _bd = _bd.insert(0, _bd[0] - pd.Timedelta(days=1))
            _bc = np.insert(_bc, 0, 0.0)
            ax.plot(_bd, _bc, 'o-', color='#1565C0', linewidth=2.5, markersize=5,
                    label=f'Base F8 33f: {_stats_base.get("cum_ew", 0):+.2%}')
        if _daily_adv:
            _ad = pd.to_datetime([d['Date'] for d in _daily_adv])
            _ac = np.cumsum([d['EW'] for d in _daily_adv]) * 100
            _ad = _ad.insert(0, _ad[0] - pd.Timedelta(days=1))
            _ac = np.insert(_ac, 0, 0.0)
            ax.plot(_ad, _ac, 's-', color='#C62828', linewidth=2.5, markersize=5,
                    label=f'ADV F8 45f: {_stats_adv.get("cum_ew", 0):+.2%}')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.set_ylabel('Cumulative EW PnL (%)')
        ax.set_title(f'Equity Curves (K={MAX_TRADES})')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Daily delta bars
        ax = axes_ab[0, 1]
        if _daily_base and _daily_adv:
            _dates_both = sorted(
                set(d['Date'] for d in _daily_base) &
                set(d['Date'] for d in _daily_adv))
            _deltas_bar = []
            _dt_bar = []
            for _d in _dates_both:
                _eb = _base_by_date.get(_d, {}).get('EW', 0)
                _ea = _adv_by_date.get(_d, {}).get('EW', 0)
                _deltas_bar.append((_ea - _eb) * 100)
                _dt_bar.append(pd.to_datetime(_d))
            _cbar = ['#C62828' if d > 0 else '#1565C0' for d in _deltas_bar]
            ax.bar(_dt_bar, _deltas_bar, color=_cbar, alpha=0.8, width=0.8)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.set_ylabel('ADV − Base EW PnL (%)')
            ax.set_title('Daily Delta (red=ADV wins, blue=Base wins)')
            ax.grid(alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # K sweep: Cum EW
        ax = axes_ab[1, 0]
        _ks_ab = list(AB_SWEEP)
        _ks_plot_ab = [0] + _ks_ab
        _base_ew = [0.0] + [_ab_sweep['BASE'].get(k, {}).get('cum_ew', 0) * 100
                            for k in _ks_ab]
        _adv_ew = [0.0] + [_ab_sweep['ADV'].get(k, {}).get('cum_ew', 0) * 100
                           for k in _ks_ab]
        ax.plot(_ks_plot_ab, _base_ew, 'o-', color='#1565C0', linewidth=2.5,
                markersize=8, label='Base F8 (33f)')
        ax.plot(_ks_plot_ab, _adv_ew, 's-', color='#C62828', linewidth=2.5,
                markersize=8, label='ADV F8 (45f)')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('MAX_TRADES (K)')
        ax.set_ylabel('Cumulative EW PnL (%)')
        ax.set_title('Cum EW vs Position Cap')
        ax.set_xticks(_ks_plot_ab)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # K sweep: Sharpe
        ax = axes_ab[1, 1]
        _base_sh = [0.0] + [_ab_sweep['BASE'].get(k, {}).get('sharpe', 0)
                            for k in _ks_ab]
        _adv_sh = [0.0] + [_ab_sweep['ADV'].get(k, {}).get('sharpe', 0)
                           for k in _ks_ab]
        ax.plot(_ks_plot_ab, _base_sh, 'o-', color='#1565C0', linewidth=2.5,
                markersize=8, label='Base F8 (33f)')
        ax.plot(_ks_plot_ab, _adv_sh, 's-', color='#C62828', linewidth=2.5,
                markersize=8, label='ADV F8 (45f)')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('MAX_TRADES (K)')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe vs Position Cap')
        ax.set_xticks(_ks_plot_ab)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        _ab_path = '/tmp/backtest_st_long_ab_f8_comparison.png'
        plt.savefig(_ab_path, dpi=150, bbox_inches='tight')
        print(f"\n  A/B chart saved: {_ab_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()

    else:
        print(f"\n  No multi-model trades — skipping multi-model comparison.")


    # ==========================================================================
    # PHASE 9: CAPITAL UTILIZATION ANALYSIS
    # ==========================================================================
    print(f"\n\n{'='*W}")
    print(f"  PHASE 9: CAPITAL UTILIZATION ANALYSIS")
    print(f"  How unfilled limit orders affect portfolio-level returns")
    print(f"{'='*W}\n")

    has_nofills = 'Filled' in df_all.columns and not df_all['Filled'].all()

    if not has_nofills:
        print(f"  No unfilled limit orders in this backtest period.")
        print(f"  All strategies have 100% capital utilization — EW = ROAC.")
    else:
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

        print(f"\n  EW vs ROAC — Strategy Comparison:")
        print(f"  (EW = avg PnL of filled trades only; ROAC = total PnL / total allocated slots)")
        print(f"\n    {'Strategy':<20} {'Cum EW':>8} {'CumROAC':>8} {'Drag':>7} "
              f"{'CapUtil':>7} {'Filled':>7}/{' Total':<6} {'AvgEW':>8} {'AvgROAC':>8}")
        print(f"    {'─'*95}")

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
            print(f"    {name:<20} {cum_ew:>+7.2%} {cum_roac:>+7.2%} {drag:>+6.2%} "
                  f"{cap_util:>6.0%} {total_filled:>5.0f} / {total_trades:<5.0f} "
                  f"{avg_ew:>+7.2%} {avg_roac:>+7.2%}")

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
                print(f"    {name:<20} #{ew_r} -> #{roac_r}  ({arrow}{abs(shift)})")
        else:
            print(f"\n  No ranking shifts between EW and ROAC — fill rates are uniform.")

        if roac_rank_names:
            best_roac = roac_rank_names[0]
            best_ew_name = ew_rank_names[0]
            print(f"\n  WINNER by EW:    {best_ew_name} ({all_strat_data[best_ew_name].get('cum_ew', 0):+.2%})")
            print(f"  WINNER by ROAC:  {best_roac} ({all_strat_data[best_roac].get('cum_roac', 0):+.2%})")
            if best_roac != best_ew_name:
                print(f"  *** ROAC flips the winner! Capital utilization matters. ***")
            else:
                print(f"  Same winner — capital utilization did not change the outcome.")


    # ==========================================================================
    # PHASE 10: MODEL OVERLAP / SIGNAL AGREEMENT ANALYSIS
    # ==========================================================================
    if multi_trades:
        print(f"\n\n{'='*W}")
        print(f"  PHASE 10: MODEL OVERLAP & SIGNAL AGREEMENT")
        print(f"  Analyzing signal overlap across all {len(set(t['Model_Source'] for t in multi_trades))} models")
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

        header = f"{'':>14s}" + ''.join(f"  {mn:>12s}" for mn in model_names)
        print(f"  {header}")
        for i, mn in enumerate(model_names):
            row_str = f"  {mn:>14s}"
            for j in range(n_models_total):
                row_str += f"  {jaccard_matrix[i,j]:>12.3f}"
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

        header2 = f"{'':>14s}" + ''.join(f"  {mn:>12s}" for mn in model_names)
        print(f"  {header2}")
        for i, mn in enumerate(model_names):
            row_str = f"  {mn:>14s}"
            for j in range(n_models_total):
                v = corr_matrix[i, j]
                row_str += f"  {v:>12.3f}" if not np.isnan(v) else f"  {'N/A':>12s}"
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

        plt.suptitle(f'Phase 10: Model Overlap & Signal Agreement ({n_models_total} Models)', fontsize=14, y=1.02)
        plt.tight_layout()
        chart10_path = '/tmp/backtest_st_long_model_overlap.png'
        plt.savefig(chart10_path, dpi=150, bbox_inches='tight')
        print(f"\n  Overlap chart saved: {chart10_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()
    else:
        print(f"\n  Phase 10 skipped — no multi_trades data available.")


    # ==========================================================================
    # PHASE 11: PER-MODEL TEMPORAL DEGRADATION ANALYSIS
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

        n_chart_models = len(model_names_t)
        fig11, axes11 = plt.subplots(2, 2, figsize=(22, 16))
        fig11.suptitle(f'Phase 11: Model Temporal Degradation Analysis\n'
                       f'{BACKTEST_START} → {BACKTEST_END} | {n_chart_models} Models',
                       fontsize=14, fontweight='bold')

        model_colors_11 = {}
        cmap = plt.cm.tab10
        for i, mn in enumerate(model_names_t):
            model_colors_11[mn] = cmap(i / max(n_chart_models - 1, 1))

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
        ax.set_title('Per-Model Equity Curves\n(solid/thick = newest folds F7+)')
        ax.legend(fontsize=7, ncol=2, loc='best')
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax = axes11[0, 1]
        family_markers = {'L5-1': 'o', 'ASL6': 's', 'MR-v2': 'D', 'MR-v2-ADV': '^'}
        family_colors_11 = {'L5-1': '#1565C0', 'ASL6': '#2E7D32', 'MR-v2': '#D81B60', 'MR-v2-ADV': '#FF6F00'}
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
        legend_elements = [Line2D([0], [0], marker=family_markers.get(f, 'o'),
                                  color=family_colors_11.get(f, '#333'), linestyle='None',
                                  markersize=10, label=f)
                           for f in families_in_data if f in family_markers]
        ax.legend(handles=legend_elements, fontsize=9)
        ax.grid(alpha=0.3)

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
        chart11_path = '/tmp/backtest_st_long_temporal_degradation.png'
        plt.savefig(chart11_path, dpi=150, bbox_inches='tight')
        print(f"\n  Temporal degradation chart saved: {chart11_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()
    else:
        print(f"\n  Phase 11 skipped — no multi_trades data available.")


# ==============================================================================
# PHASE DIAGNOSTIC: MODEL PERFORMANCE vs MARKET OPPORTUNITY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  DIAGNOSTIC: WHY IS PERFORMANCE DEGRADING?")
print(f"  Model metrics vs market opportunity — LONG")
print(f"  {BACKTEST_START} → {BACKTEST_END}")
print(f"{'='*W}")

if 'df_all' in dir() and len(df_all) > 0:

    # ── D1: Signal count & probability by week ──
    df_all['Date_dt'] = pd.to_datetime(df_all['Date'])
    df_all['Week'] = df_all['Date_dt'].dt.isocalendar().week.astype(int)
    df_all['Year_Week'] = (df_all['Date_dt'].dt.isocalendar().year.astype(str) + '-W' +
                            df_all['Date_dt'].dt.isocalendar().week.astype(str).str.zfill(2))

    print(f"\n  ── D1: Signal Volume & Probability by Week ──\n")
    print(f"  {'Week':>8}  {'Signals':>8} {'Filled':>7} {'Avg Prob':>9} {'Med Prob':>9} "
          f"{'Avg Margin':>10} {'TP Rate':>8} {'Avg PnL':>8}")
    print(f"  {'─'*85}")

    for wk in sorted(df_all['Year_Week'].unique()):
        wk_df = df_all[df_all['Year_Week'] == wk]
        wk_filled = wk_df[wk_df['Filled'] == True] if 'Filled' in wk_df.columns else wk_df
        n_sig = len(wk_df)
        n_fill = len(wk_filled)
        avg_prob = wk_df['Prob'].mean()
        med_prob = wk_df['Prob'].median()
        avg_margin = (wk_df['Prob'] - wk_df['Thresh']).mean()
        tp_rate = (wk_filled['Outcome'] == 'TP').mean() * 100 if len(wk_filled) > 0 else 0
        avg_pnl = wk_filled['PnL'].mean() if len(wk_filled) > 0 else 0
        print(f"  {wk:>8}  {n_sig:>8} {n_fill:>7} {avg_prob:>9.4f} {med_prob:>9.4f} "
              f"{avg_margin:>+9.4f} {tp_rate:>7.1f}% {avg_pnl:>+7.2f}%")

    # ── D2: Trade outcome decomposition by week ──
    print(f"\n  ── D2: Outcome Breakdown by Week ──\n")
    print(f"  {'Week':>8}  {'TP':>5} {'SL':>5} {'EOD':>5} {'NO_FILL':>8} {'TP%':>6} {'SL%':>6} {'EOD%':>6}")
    print(f"  {'─'*60}")

    for wk in sorted(df_all['Year_Week'].unique()):
        wk_df = df_all[df_all['Year_Week'] == wk]
        n_tp = (wk_df['Outcome'] == 'TP').sum()
        n_sl = (wk_df['Outcome'] == 'SL').sum()
        n_eod = (wk_df['Outcome'] == 'EOD').sum()
        n_nf = (wk_df['Outcome'] == 'NO_FILL').sum()
        n_filled = n_tp + n_sl + n_eod
        tp_pct = n_tp / n_filled * 100 if n_filled > 0 else 0
        sl_pct = n_sl / n_filled * 100 if n_filled > 0 else 0
        eod_pct = n_eod / n_filled * 100 if n_filled > 0 else 0
        print(f"  {wk:>8}  {n_tp:>5} {n_sl:>5} {n_eod:>5} {n_nf:>8} {tp_pct:>5.1f}% {sl_pct:>5.1f}% {eod_pct:>5.1f}%")

    # ── D3: Per-model family firing rates ──
    print(f"\n  ── D3: Per-Model Family Performance ──\n")
    if 'Family' in df_all.columns:
        print(f"  {'Family':<20} {'N':>5} {'Filled':>7} {'TP%':>6} {'Avg PnL':>8} {'Avg Prob':>9}")
        print(f"  {'─'*60}")
        for fam in sorted(df_all['Family'].unique()):
            fam_df = df_all[df_all['Family'] == fam]
            fam_filled = fam_df[fam_df['Filled'] == True] if 'Filled' in fam_df.columns else fam_df
            n = len(fam_df)
            n_fill = len(fam_filled)
            tp_rate = (fam_filled['Outcome'] == 'TP').mean() * 100 if len(fam_filled) > 0 else 0
            avg_pnl = fam_filled['PnL'].mean() if len(fam_filled) > 0 else 0
            avg_prob = fam_df['Prob'].mean()
            print(f"  {fam:<20} {n:>5} {n_fill:>7} {tp_rate:>5.1f}% {avg_pnl:>+7.2f}% {avg_prob:>9.4f}")

    # ── D4: MFE at 15m/30m/60m for signal days vs ALL days ──
    print(f"\n  ── D4: MFE Analysis — Signal Days vs Market Opportunity ──")
    print(f"  Loading price data for MFE computation...\n")

    _deploy_tickers_diag = sorted(df_all['Ticker'].unique())
    _bt_s = pd.Timestamp(BACKTEST_START).date()
    _bt_e = pd.Timestamp(BACKTEST_END).date()

    _signal_mfe = []
    _market_mfe = []

    for _tk in _deploy_tickers_diag:
        _df = load_master(_tk, tail_days=TAIL_DAYS)
        if _df is None:
            continue
        try:
            _df_pst = _df.index.tz_convert(PST)
        except Exception:
            continue
        _mins = _df_pst.hour * 60 + _df_pst.minute
        _df = _df.copy()
        _df['_min'] = _mins
        _df['_date'] = _df_pst.date

        _signal_dates = set(
            pd.to_datetime(df_all[df_all['Ticker'] == _tk]['Date']).dt.date
        )
        _tk_trades = df_all[df_all['Ticker'] == _tk]

        for _dv, _day_df in _df.groupby('_date'):
            if _dv < _bt_s or _dv > _bt_e:
                continue
            _entry_bars = _day_df[_day_df['_min'] == ENTRY_MINUTE]
            if len(_entry_bars) == 0:
                _entry_bars = _day_df[(_day_df['_min'] >= ENTRY_MINUTE - 2) &
                                      (_day_df['_min'] <= ENTRY_MINUTE + 2)]
            if len(_entry_bars) == 0:
                continue
            _ep = float(_entry_bars.iloc[0]['Close'])
            if _ep <= 0:
                continue

            _mfe_row = {'Ticker': _tk, 'Date': str(_dv),
                        'Is_Signal': _dv in _signal_dates}

            for _wmin, _wlabel in [(15, '15m'), (30, '30m'), (60, '60m')]:
                _after = _day_df[(_day_df['_min'] > ENTRY_MINUTE) &
                                  (_day_df['_min'] <= ENTRY_MINUTE + _wmin)]
                if len(_after) >= 2:
                    _mfe_up = (float(_after['High'].max()) - _ep) / _ep * 100
                    _mfe_dn = (_ep - float(_after['Low'].min())) / _ep * 100
                else:
                    _mfe_up = _mfe_dn = 0
                _mfe_row[f'MFE_Up_{_wlabel}'] = _mfe_up
                _mfe_row[f'MFE_Dn_{_wlabel}'] = _mfe_dn

            if _dv in _signal_dates:
                _trade_row = _tk_trades[_tk_trades['Date'] == str(_dv)]
                if len(_trade_row) > 0:
                    _mfe_row['Trade_PnL'] = float(_trade_row.iloc[0]['PnL'])
                    _mfe_row['Trade_Outcome'] = _trade_row.iloc[0]['Outcome']
                    _mfe_row['Trade_TP'] = float(_trade_row.iloc[0]['TP_Pct'])
                    _mfe_row['Trade_Prob'] = float(_trade_row.iloc[0]['Prob'])
                _signal_mfe.append(_mfe_row)
            _market_mfe.append(_mfe_row)

        _master_cache.pop((_tk, TAIL_DAYS), None)
        del _df
        gc.collect()

    if _signal_mfe and _market_mfe:
        _df_sig = pd.DataFrame(_signal_mfe)
        _df_mkt = pd.DataFrame(_market_mfe)

        print(f"  {'Metric':<30} {'Signal Days':>12} {'ALL Days':>12} {'Delta':>10}")
        print(f"  {'─'*70}")
        for _wl in ['15m', '30m', '60m']:
            _s_up = _df_sig[f'MFE_Up_{_wl}'].mean()
            _m_up = _df_mkt[f'MFE_Up_{_wl}'].mean()
            print(f"  {'Avg MFE UP ' + _wl:<30} {_s_up:>11.2f}% {_m_up:>11.2f}% {_s_up - _m_up:>+9.2f}%")
        for _wl in ['15m', '30m', '60m']:
            _s_dn = _df_sig[f'MFE_Dn_{_wl}'].mean()
            _m_dn = _df_mkt[f'MFE_Dn_{_wl}'].mean()
            print(f"  {'Avg MFE DN ' + _wl:<30} {_s_dn:>11.2f}% {_m_dn:>11.2f}% {_s_dn - _m_dn:>+9.2f}%")

        print(f"\n  {'':30} {'Signal':>12} {'Market':>12}")
        print(f"  {'─'*58}")
        for _wl in ['15m', '30m', '60m']:
            _s3 = (_df_sig[f'MFE_Up_{_wl}'] >= 3.0).mean() * 100
            _m3 = (_df_mkt[f'MFE_Up_{_wl}'] >= 3.0).mean() * 100
            print(f"  {'≥3% UP rate ' + _wl:<30} {_s3:>11.1f}% {_m3:>11.1f}%")

        print(f"\n  Signal days: {len(_df_sig)}  |  All days: {len(_df_mkt)}")

        # ── D5: TP Reachability — would TP have been hit at 15m/30m/60m? ──
        if 'Trade_TP' in _df_sig.columns:
            print(f"\n  ── D5: TP Reachability (signal days only) ──\n")
            _sig_with_tp = _df_sig[_df_sig['Trade_TP'].notna()]
            if len(_sig_with_tp) > 0:
                for _wl in ['15m', '30m', '60m']:
                    _reached = (_sig_with_tp[f'MFE_Up_{_wl}'] >= _sig_with_tp['Trade_TP'] * 100).mean() * 100
                    _avg_mfe = _sig_with_tp[f'MFE_Up_{_wl}'].mean()
                    _avg_tp = _sig_with_tp['Trade_TP'].mean() * 100
                    print(f"  {_wl}: {_reached:>5.1f}% of signal days reached TP "
                          f"(avg MFE: {_avg_mfe:.2f}%, avg TP: {_avg_tp:.2f}%)")

        # ── D6: Probability calibration ──
        print(f"\n  ── D6: Probability Calibration ──")
        print(f"  (binned probability vs actual 60m MFE ≥ 3%)\n")
        if 'Trade_Prob' in _df_sig.columns:
            _sig_cal = _df_sig[_df_sig['Trade_Prob'].notna()].copy()
            _sig_cal['Hit_3pct'] = _sig_cal['MFE_Up_60m'] >= 3.0
            _bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            _sig_cal['Prob_Bin'] = pd.cut(_sig_cal['Trade_Prob'], bins=_bins)
            _cal = _sig_cal.groupby('Prob_Bin', observed=True).agg(
                N=('Hit_3pct', 'count'),
                Hit_Rate=('Hit_3pct', 'mean'),
                Avg_MFE=('MFE_Up_60m', 'mean'),
                Avg_PnL=('Trade_PnL', 'mean'),
            )
            print(f"  {'Prob Bin':<15} {'N':>5} {'Hit Rate':>10} {'Avg MFE':>9} {'Avg PnL':>9}")
            print(f"  {'─'*52}")
            for idx, row in _cal.iterrows():
                print(f"  {str(idx):<15} {row['N']:>5.0f} {row['Hit_Rate']:>9.1%} "
                      f"{row['Avg_MFE']:>8.2f}% {row['Avg_PnL']:>+8.2f}%")

        # ── D7: Diagnostic charts ──
        print(f"\n  Generating diagnostic charts...")
        fig_d, axes_d = plt.subplots(2, 3, figsize=(22, 12))
        fig_d.suptitle(f'LONG Diagnostic: Model Performance vs Market Opportunity\n'
                       f'{BACKTEST_START} → {BACKTEST_END}', fontsize=13, fontweight='bold')

        # D7a: Weekly PnL + signal count
        ax = axes_d[0, 0]
        weeks = sorted(df_all['Year_Week'].unique())
        wk_pnl = []
        wk_cnt = []
        for wk in weeks:
            wk_df = df_all[(df_all['Year_Week'] == wk) & (df_all['Filled'] == True)]
            wk_pnl.append(wk_df['PnL'].mean() if len(wk_df) > 0 else 0)
            wk_cnt.append(len(wk_df))
        ax.bar(range(len(weeks)), wk_pnl,
               color=['#2E7D32' if p >= 0 else '#C62828' for p in wk_pnl], alpha=0.7)
        ax.set_xticks(range(len(weeks)))
        ax.set_xticklabels([w.split('-W')[1] for w in weeks], fontsize=8)
        ax.set_ylabel('Avg PnL per Trade (%)')
        ax.set_title('Weekly Avg PnL')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(range(len(weeks)), wk_cnt, 'ko-', markersize=4, alpha=0.5)
        ax2.set_ylabel('Signal Count', color='grey')

        # D7b: MFE distribution — signal days vs all days
        ax = axes_d[0, 1]
        ax.hist(_df_mkt['MFE_Up_60m'], bins=50, alpha=0.5, color='grey', label='All days', density=True)
        ax.hist(_df_sig['MFE_Up_60m'], bins=30, alpha=0.7, color='#1565C0', label='Signal days', density=True)
        ax.axvline(3, color='red', linewidth=1.5, linestyle='--', label='3% spike')
        if 'Trade_TP' in _df_sig.columns and _df_sig['Trade_TP'].notna().any():
            _avg_tp_pct = _df_sig['Trade_TP'].mean() * 100
            ax.axvline(_avg_tp_pct, color='darkred', linewidth=1.5, linestyle=':', label=f'Avg TP ({_avg_tp_pct:.1f}%)')
        ax.set_xlabel('60m MFE UP (%)')
        ax.set_ylabel('Density')
        ax.set_title('60m MFE: Signal Days vs All Days')
        ax.legend(fontsize=8)
        ax.set_xlim(-1, 15)

        # D7c: Probability distribution
        ax = axes_d[0, 2]
        if 'Trade_Prob' in _df_sig.columns:
            ax.hist(_df_sig['Trade_Prob'], bins=20, color='#FF6F00', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Model Probability')
        ax.set_ylabel('Count')
        ax.set_title('Signal Probability Distribution')
        ax.grid(axis='y', alpha=0.3)

        # D7d: TP reachability by time window
        ax = axes_d[1, 0]
        if 'Trade_TP' in _df_sig.columns and _df_sig['Trade_TP'].notna().any():
            _reach_rates = []
            _windows = ['15m', '30m', '60m']
            for _wl in _windows:
                _r = (_sig_with_tp[f'MFE_Up_{_wl}'] >= _sig_with_tp['Trade_TP'] * 100).mean() * 100
                _reach_rates.append(_r)
            ax.bar(range(3), _reach_rates, color=['#C62828', '#FF6F00', '#2E7D32'], alpha=0.7)
            ax.set_xticks(range(3))
            ax.set_xticklabels(_windows)
            ax.set_ylabel('% of signal days reaching TP')
            ax.set_title('TP Reachability by Window')
            for i, r in enumerate(_reach_rates):
                ax.text(i, r + 1, f'{r:.1f}%', ha='center', fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        # D7e: Outcome by week (stacked)
        ax = axes_d[1, 1]
        outcomes_by_wk = {'TP': [], 'SL': [], 'EOD': []}
        for wk in weeks:
            wk_df = df_all[(df_all['Year_Week'] == wk) & (df_all['Filled'] == True)]
            n = len(wk_df) if len(wk_df) > 0 else 1
            outcomes_by_wk['TP'].append((wk_df['Outcome'] == 'TP').sum() / n * 100)
            outcomes_by_wk['SL'].append((wk_df['Outcome'] == 'SL').sum() / n * 100)
            outcomes_by_wk['EOD'].append((wk_df['Outcome'] == 'EOD').sum() / n * 100)
        x = range(len(weeks))
        ax.bar(x, outcomes_by_wk['TP'], color='#2E7D32', alpha=0.7, label='TP')
        ax.bar(x, outcomes_by_wk['SL'], bottom=outcomes_by_wk['TP'], color='#C62828', alpha=0.7, label='SL')
        _bot = [a + b for a, b in zip(outcomes_by_wk['TP'], outcomes_by_wk['SL'])]
        ax.bar(x, outcomes_by_wk['EOD'], bottom=_bot, color='#757575', alpha=0.7, label='EOD')
        ax.set_xticks(x)
        ax.set_xticklabels([w.split('-W')[1] for w in weeks], fontsize=8)
        ax.set_ylabel('Outcome %')
        ax.set_title('Trade Outcome by Week')
        ax.legend(fontsize=8)

        # D7f: Calibration scatter
        ax = axes_d[1, 2]
        if 'Trade_Prob' in _df_sig.columns and 'Trade_PnL' in _df_sig.columns:
            _sig_plot = _df_sig[_df_sig['Trade_PnL'].notna()]
            colors = ['#2E7D32' if p > 0 else '#C62828' for p in _sig_plot['Trade_PnL']]
            ax.scatter(_sig_plot['Trade_Prob'], _sig_plot['MFE_Up_60m'],
                      c=colors, alpha=0.5, s=20)
            ax.axhline(3, color='red', linewidth=1, linestyle='--', alpha=0.5)
            ax.set_xlabel('Model Probability')
            ax.set_ylabel('60m MFE UP (%)')
            ax.set_title('Probability vs Actual MFE\n(green=profit, red=loss)')
            ax.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        _diag_path = '/tmp/backtest_long_diagnostic.png'
        plt.savefig(_diag_path, dpi=150, bbox_inches='tight')
        print(f"  Diagnostic chart saved: {_diag_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()

    else:
        print(f"  No MFE data computed — skipping diagnostic charts.")

else:
    print(f"  No trades found — diagnostic skipped.")


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
print(f"  LONG 9-MODEL PRIORITIZATION + CONSENSUS FILTER ANALYSIS COMPLETE")
print(f"{'='*W}")
