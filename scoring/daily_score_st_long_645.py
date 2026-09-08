# %% ── Daily Scoring — ST LONG M_CONS_NO_F7 645 (L5-1 + ASL6 + MR-v2-ADV) ──
#
# 6:45 AM variant — same models and strategy as daily_score_st_long.py, scored 30 min later
#
# Spike Transformer LONG — 4-Family Hybrid + M_CONS_NO_F7 selection
#
#   - Models: L5-1 F5/F6/F7 + ASL6 F5/F6/F7 + MR-v2 F6/F7/F8/F9 + MR-v2-ADV F8
#   - 3 consensus models (NO_F7 — fold 7 removed):
#       L5-1 F6 (standard), ASL6 F6 (standard), MR-v2-ADV F8 (multires 45f)
#   - Newest First priority: MR-v2-ADV F8 > MR-v2 F9 > MR-v2 F7 > L5-1/ASL6 F7 > F6
#   - M_CONS_NO_F7: score ALL tickers with all available models,
#     count fires from 3 consensus models, rank by (n_fires DESC, best_prob DESC),
#     pick top MAX_TRADES
#   - MR-v2-ADV: MultiResSpikeTransformer (dual 1m+15m, 45 features, 157K params)
#   - Score at minute 405 (6:45 AM PT), entry at minute 415 (6:55 AM PT)
#   - FiLM regime conditioning (no separate regime filter)
#
# Run AFTER daily_stitch_master.py has finished updating GCS.

import os
import sys
import json
import gc
import subprocess

import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

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


# ==============================================================================
# CONSTANTS
# ==============================================================================
BUCKET_NAME      = "time-series-bucket-ys"
BQ_TABLE         = "ys-time-series.scoring_logs.live_predictions"
WF_BQ_TABLE      = "ys-time-series.scoring_logs.wf_optimization_results"
INFERENCE_DIR    = "Data/Inference_v2"
TAIL_DAYS        = 25
PST              = pytz.timezone('America/Los_Angeles')

FOLDS            = [5, 6, 7, 8, 9]
BASE_PNL_THRESH  = 0.018
F7_VAL_B_THRESH  = 0.005

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
            8: 'spike_transformer_long_L5_1_multires_v2_fold8',
            9: 'spike_transformer_long_L5_1_multires_v2_fold9',
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

CONSENSUS_MODELS = {
    ('L5-1', 6), ('ASL6', 6), ('MR-v2-ADV', 8),
}

NEWEST_FIRST_ORDER = [
    ('MR-v2-ADV', 8), ('MR-v2', 9), ('MR-v2', 7),
    ('L5-1', 7), ('ASL6', 7),
    ('L5-1', 6), ('ASL6', 6), ('MR-v2', 6),
    ('L5-1', 5), ('ASL6', 5),
]
SCORE_MINUTE     = 405
ENTRY_MINUTE     = 415
MAX_TRADES       = 2
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

N_BLOCKS_15M     = 1
TIME_STEPS_15M   = 128
N_LOWRES_F       = 12

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

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

HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}
HORIZON_FROM_LABEL = {'15m': 0, '30m': 1, '60m': 2}

EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

gcs_client = gcs_storage.Client()


# ==============================================================================
# BQ CONFIG LOADING — Newest First across L5-1 + ASL6 + MR-v2
# ==============================================================================
def load_deploy_configs():
    if bigquery is None:
        raise RuntimeError("BigQuery not available — cannot load deploy configs")
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
            print(f"    {family_name} F{fold_id}: {len(cfgs)} tickers — {', '.join(sorted(cfgs.keys()))}")

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
        print(f"    {src}: {len(tickers)} — {', '.join(tickers)}")
    print(f"    Total: {len(deploy)} tickers (Newest First)")
    return deploy, family_fold_cfgs


# ==============================================================================
# DATA LOADING
# ==============================================================================
_master_cache = {}
_feat_cache = {}
_15m_cache = {}


_NEEDED_COLS = {'Open', 'High', 'Low', 'Close', 'Volume',
                'open', 'high', 'low', 'close', 'volume',
                'Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
                'bsize', 'asize', 'Total_Vol',
                'Total_vol', 'Trade_count', 'Volume_delta', 'Delta_pct',
                'Obi', 'Bsize', 'Asize',
                'Datetime', 'Date', 'Time', 'Timestamp', 'Level_0'}


def load_master(ticker, tail_days=None):
    cache_key = (ticker, tail_days)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    local_path = f"/tmp/_st_long_{file_ticker}_master.parquet"
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
            num_col = next((c for c in df.columns
                           if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                          df.columns[0])
            df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
        df = df.set_index('Datetime')
        if tail_days is not None:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
            df = df[df.index >= cutoff].copy()
            if df.empty:
                return None
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
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        for c in MICRO_COLS:
            if c not in df.columns:
                df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        _master_cache[cache_key] = df
        return df
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        print(f"      load_master({ticker}): FAILED — {e}")
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
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is None:
        return None
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"Data/Inference/Microstructure_Advanced/{file_ticker}_1m_advanced_micro.parquet"
    local_path = f"/tmp/_score_long_adv_{file_ticker}_adv_micro.parquet"
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
# MODEL ARCHITECTURE  (SpikeTransformer, 22 regime features)
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
# MODEL LOADING — two instances (standard + multires), swap weights per (family, fold)
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


def load_models_and_meta(deploy_configs, family_fold_cfgs=None):
    global _model_std, _model_mr, _model_adv, _fold_weights
    global _current_key_std, _current_key_mr, _current_key_adv, _family_meta

    all_cfgs = list(deploy_configs.values())
    if family_fold_cfgs:
        for cfgs in family_fold_cfgs.values():
            all_cfgs.extend(cfgs.values())
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

    first_key = needed_keys[0]
    first_model = _get_model_for_family(first_key[0])
    _load_keras3_weights(first_model, _fold_weights[first_key])
    first_arch = FAMILIES[first_key[0]].get('arch', 'standard')
    if first_arch == 'multires_adv':
        _current_key_adv = first_key
    elif first_arch == 'multires':
        _current_key_mr = first_key
    else:
        _current_key_std = first_key
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
# CONSENSUS — score ticker with ALL available models
# ==============================================================================
def score_all_models(ticker, df_feat, regime_df, run_ts, family_fold_cfgs, df_15m=None):
    """Score a ticker with ALL available (family, fold) models, return consensus data."""
    today = run_ts.astimezone(PST).date()
    target_pst = PST.localize(datetime(today.year, today.month, today.day,
                                       SCORE_MINUTE // 60, SCORE_MINUTE % 60))
    target_utc = target_pst.astimezone(pytz.utc)

    candidates = df_feat.index[df_feat.index <= target_utc]
    if len(candidates) == 0:
        return None
    row_idx = df_feat.index.get_loc(candidates[-1])
    if row_idx < TIME_STEPS:
        return None

    all_scores = []
    for (family_name, fold_id), cfgs in family_fold_cfgs.items():
        if ticker not in cfgs:
            continue
        cfg = cfgs[ticker]
        h_idx = cfg['horizon']

        scaling_stats = _family_meta[family_name]['scaling_stats']
        regime_stats = _family_meta[family_name]['regime_stats']
        arch = FAMILIES[family_name].get('arch', 'standard')
        is_multires = arch in ('multires', 'multires_adv')
        is_adv = arch == 'multires_adv'

        feature_cols = ALL_COLS_ADV if is_adv else ALL_COLS
        df_scaled = apply_scaling(df_feat, scaling_stats, feature_cols)
        XA = df_scaled[feature_cols].values.astype(np.float32)
        seq_batch = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]

        regime_at_entry = regime_df.iloc[regime_df.index.get_indexer(
            [df_feat.index[row_idx]], method='pad')[0]]
        regime_scaled = np.zeros(N_REGIME_F, dtype=np.float32)
        for i, col in enumerate(REGIME_COLS):
            val = float(regime_at_entry[col]) if col in regime_at_entry.index else 0.0
            if col in regime_stats:
                s = regime_stats[col]
                val = np.clip((val - s['med']) / s['iqr'], -10, 10)
            regime_scaled[i] = val
        regime_batch = regime_scaled[np.newaxis]

        _ensure_fold_loaded(family_name, fold_id)
        inputs = {'seq_input': seq_batch, 'regime_input': regime_batch}

        if is_multires and df_15m is not None:
            scaling_stats_15m = _family_meta[family_name].get('scaling_stats_15m')
            if scaling_stats_15m:
                df_15m_s = apply_scaling(df_15m, scaling_stats_15m, LOWRES_COLS)
            else:
                df_15m_s = df_15m.copy()
            XL = df_15m_s[LOWRES_COLS].values.astype(np.float32)
            ts_1m = df_feat.index[row_idx]
            lr_idx = df_15m_s.index.get_indexer([ts_1m], method='pad')[0]
            lr_end = lr_idx + 1
            lr_start = max(0, lr_end - TIME_STEPS_15M)
            lr_window = XL[lr_start:lr_end]
            if len(lr_window) < TIME_STEPS_15M:
                pad = np.zeros((TIME_STEPS_15M - len(lr_window), N_LOWRES_F), dtype=np.float32)
                lr_window = np.concatenate([pad, lr_window], axis=0)
            inputs['seq_15m_input'] = lr_window[np.newaxis]

        model = _get_model_for_family(family_name)
        preds = model(inputs, training=False).numpy().flatten()
        prob = float(preds[h_idx])

        all_scores.append({
            'family': family_name,
            'fold': fold_id,
            'prob': round(prob, 4),
            'thresh': cfg['prob_thresh'],
            'fired': prob >= cfg['prob_thresh'],
            'horizon_idx': h_idx,
        })

    if not all_scores:
        return None

    consensus_scores = [s for s in all_scores
                        if (s['family'], s['fold']) in CONSENSUS_MODELS]
    fires = [s for s in consensus_scores if s['fired']]
    return {
        'n_available': len(consensus_scores),
        'n_fires': len(fires),
        'all_scores': all_scores,
        'fires': fires,
        'best_prob': max(s['prob'] for s in all_scores),
        'best_fire_prob': max(s['prob'] for s in fires) if fires else 0,
    }


# ==============================================================================
# SCORING ENGINE
# ==============================================================================
def score_ticker(ticker, cfg, df_feat, regime_df, run_ts, df_15m=None):
    today = run_ts.astimezone(PST).date()
    target_pst = PST.localize(datetime(today.year, today.month, today.day,
                                       SCORE_MINUTE // 60, SCORE_MINUTE % 60))
    target_utc = target_pst.astimezone(pytz.utc)

    candidates = df_feat.index[df_feat.index <= target_utc]
    if len(candidates) == 0:
        print(f"      {ticker}: SKIP — no data at or before {target_pst.strftime('%H:%M')} PST")
        return None
    row_idx = df_feat.index.get_loc(candidates[-1])
    bar_ts_pst = df_feat.index[row_idx].tz_convert(PST)
    score_date = bar_ts_pst.date()
    if score_date != today:
        print(f"      {ticker}: SKIP — nearest bar is {score_date} (stale master data)")
        return None
    bar_minute = bar_ts_pst.hour * 60 + bar_ts_pst.minute
    if bar_minute != SCORE_MINUTE:
        print(f"      {ticker}: SKIP — bar at {bar_ts_pst.strftime('%H:%M')} PST, need {SCORE_MINUTE//60}:{SCORE_MINUTE%60:02d}")
        return None
    if row_idx < TIME_STEPS:
        print(f"      {ticker}: SKIP — not enough history: {row_idx} < {TIME_STEPS}")
        return None
    if 'Volume' in df_feat.columns and float(df_feat['Volume'].iloc[row_idx]) == 0:
        print(f"      {ticker}: SKIP — no liquidity at {bar_ts_pst.strftime('%H:%M')} PST (Volume=0)")
        return None

    family = cfg['family']
    scaling_stats = _family_meta[family]['scaling_stats']
    regime_stats = _family_meta[family]['regime_stats']
    arch = FAMILIES[family].get('arch', 'standard')
    is_multires = arch in ('multires', 'multires_adv')
    is_adv = arch == 'multires_adv'

    feature_cols = ALL_COLS_ADV if is_adv else ALL_COLS
    df_scaled = apply_scaling(df_feat, scaling_stats, feature_cols)

    XA = df_scaled[feature_cols].values.astype(np.float32)
    seq_batch = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]

    regime_at_entry = regime_df.iloc[regime_df.index.get_indexer(
        [df_feat.index[row_idx]], method='pad')[0]]
    regime_scaled = np.zeros(N_REGIME_F, dtype=np.float32)
    for i, col in enumerate(REGIME_COLS):
        val = float(regime_at_entry[col]) if col in regime_at_entry.index else 0.0
        if col in regime_stats:
            s = regime_stats[col]
            val = np.clip((val - s['med']) / s['iqr'], -10, 10)
        regime_scaled[i] = val
    regime_batch = regime_scaled[np.newaxis]

    fold_id = cfg['fold']
    _ensure_fold_loaded(family, fold_id)

    inputs = {'seq_input': seq_batch, 'regime_input': regime_batch}
    if is_multires and df_15m is not None:
        scaling_stats_15m = _family_meta[family].get('scaling_stats_15m')
        if scaling_stats_15m:
            df_15m_s = apply_scaling(df_15m, scaling_stats_15m, LOWRES_COLS)
        else:
            df_15m_s = df_15m.copy()
        XL = df_15m_s[LOWRES_COLS].values.astype(np.float32)
        ts_1m = df_feat.index[row_idx]
        lr_idx = df_15m_s.index.get_indexer([ts_1m], method='pad')[0]
        lr_end = lr_idx + 1
        lr_start = max(0, lr_end - TIME_STEPS_15M)
        lr_window = XL[lr_start:lr_end]
        if len(lr_window) < TIME_STEPS_15M:
            pad = np.zeros((TIME_STEPS_15M - len(lr_window), N_LOWRES_F), dtype=np.float32)
            lr_window = np.concatenate([pad, lr_window], axis=0)
        inputs['seq_15m_input'] = lr_window[np.newaxis]

    model = _get_model_for_family(family)
    preds = model(inputs, training=False).numpy().flatten()

    h_idx = cfg['horizon']
    prob = float(preds[h_idx])
    signal = prob >= cfg['prob_thresh']

    entry_bar_idx = row_idx + (ENTRY_MINUTE - SCORE_MINUTE)
    if entry_bar_idx < len(df_feat) and 'Close' in df_feat.columns:
        close = float(df_feat['Close'].iloc[entry_bar_idx])
    elif 'Close' in df_feat.columns:
        close = float(df_feat['Close'].iloc[row_idx])
    else:
        close = float('nan')

    entry_price = close
    if cfg['mode'] == 'limit_entry':
        entry_price = close * (1 - cfg.get('dip_pct', 0))

    return {
        'ticker': ticker,
        'family': family,
        'fold': fold_id,
        'horizon': HORIZON_LABELS[h_idx],
        'horizon_idx': h_idx,
        'prob': round(prob, 4),
        'threshold': cfg['prob_thresh'],
        'signal': signal,
        'tp_pct': cfg['tp_pct'],
        'sl_pct': cfg['sl_pct'],
        'mode': cfg['mode'],
        'dip_pct': cfg.get('dip_pct', 0),
        'max_wait': cfg.get('max_wait', 0),
        'grace_minutes': cfg.get('grace_minutes', 0),
        'entry_minute': ENTRY_MINUTE,
        'score_date': score_date,
        'bar_ts': df_feat.index[row_idx],
        'close': round(close, 4),
        'entry_price': round(entry_price, 4),
        'all_horizon_probs': {HORIZON_LABELS[i]: round(float(preds[i]), 4)
                              for i in range(N_HORIZONS)},
    }


# ==============================================================================
# DASHBOARD
# ==============================================================================
_MODE_LABELS = {'static': 'STATIC', 'grace': 'GRACE', 'limit_entry': 'LIMIT',
                'volscaled': 'VOLSC'}


def print_dashboard(results, deploy_configs, run_ts):
    W = 120
    pst_str = run_ts.astimezone(PST).strftime('%Y-%m-%d %H:%M PST')

    source_counts = {}
    for cfg in deploy_configs.values():
        key = f"{cfg['family']} F{cfg['fold']}"
        source_counts.setdefault(key, 0)
        source_counts[key] += 1

    print()
    print(f"  {'='*W}")
    print(f"  ST LONG 645 SIGNALS — M_CONS_NO_F7 (L5-1 F6 + ASL6 F6 + MR-v2-ADV F8)   {pst_str}")
    print(f"  {'='*W}")
    print(f"  Strategy: M_CONS_NO_F7 (3 consensus, no fold 7, max={MAX_TRADES})  |  "
          f"Score: {SCORE_MINUTE // 60}:{SCORE_MINUTE % 60:02d} PST  |  "
          f"Entry: {ENTRY_MINUTE // 60}:{ENTRY_MINUTE % 60:02d} PST")
    for src in sorted(source_counts):
        src_tickers = sorted(t for t, c in deploy_configs.items()
                             if f"{c['family']} F{c['fold']}" == src)
        print(f"  {src}: {', '.join(src_tickers)} ({len(src_tickers)})")

    if not results:
        print(f"\n  No tickers scored.")
        print(f"  {'='*W}")
        return

    print()
    print(f"  {'Ticker':<6} {'Src':<8} {'H':>3}  {'SIG':<7}  "
          f"{'Buy@':>10}  {'SL@':>10}  {'TP@':>10}  "
          f"{'Prob':>5} / {'Thr':>5}  {'Mode':<8}  "
          f"{'Close':>8}  {'Fire':>5} {'Avl':>3}  "
          f"{'P(15m)':>6} {'P(30m)':>6} {'P(60m)':>6}")
    print(f"  {'─'*W}")

    for r in results:
        if r['signal']:
            sig_str = "\033[92m GO  \033[0m"
        else:
            sig_str = "  --  "
        entry_price = r['entry_price']
        buy_str = f"${entry_price:.2f}"
        sl_str = f"${entry_price * (1 - r['sl_pct']):.2f}"
        tp_str = f"${entry_price * (1 + r['tp_pct']):.2f}"
        mode_label = _MODE_LABELS.get(r['mode'], '??')
        hp = r['all_horizon_probs']
        src_label = f"{r['family']}F{r['fold']}"
        n_fires = r.get('n_fires', 0)
        n_avail = r.get('n_available', 1)
        fire_str = f"{n_fires}/{n_avail}"
        print(f"  {r['ticker']:<6} {src_label:<8} {r['horizon']:>3}  {sig_str}  "
              f"{buy_str:>10}  {sl_str:>10}  {tp_str:>10}  "
              f"{r['prob']:>5.3f} / {r['threshold']:>5.3f}  {mode_label:<8}  "
              f"{r['close']:>8.2f}  "
              f"{fire_str:>5} {n_avail:>3}  "
              f"{hp.get('15m', 0):>5.3f}  {hp.get('30m', 0):>5.3f}  {hp.get('60m', 0):>5.3f}")

    signals = [r for r in results if r['signal']]

    print()
    print(f"  {'='*W}")
    print(f"  ST LONG 645 SUMMARY")
    print(f"  {'─'*W}")
    print(f"  Scored:    {len(results)} tickers")
    print(f"  Signals:   {len(signals)}/{len(results)}" +
          (f"  ({', '.join(r['ticker'] for r in signals)})" if signals else ""))
    if signals:
        print(f"  Consensus: {len(CONSENSUS_MODELS)} models (NO_F7)")

    if signals:
        for family, fold in NEWEST_FIRST_ORDER:
            fold_sigs = [r for r in signals if r['family'] == family and r['fold'] == fold]
            if fold_sigs:
                print(f"    {family} F{fold}: {', '.join(r['ticker'] for r in fold_sigs)}")

        print()
        print(f"  \033[92mTRADE: {', '.join(r['ticker'] for r in signals)}"
              f"  ({len(signals)} position{'s' if len(signals) != 1 else ''})\033[0m")

        limit_trades = [r for r in signals if r['mode'] == 'limit_entry']
        if limit_trades:
            print()
            print(f"  {'─'*W}")
            print(f"  LIMIT ORDERS TO PLACE")
            print(f"  {'─'*W}")
            for r in limit_trades:
                c = r['close']
                dip = r.get('dip_pct', 0) or 0
                lp = r['entry_price']
                tp_price = lp * (1 + r['tp_pct'])
                sl_price = lp * (1 - r['sl_pct'])
                wait = r.get('max_wait', 30) or 30
                em = r['entry_minute']
                expire_min = em + wait
                expire_h = expire_min // 60
                expire_ampm = "AM" if expire_h < 12 else "PM"
                expire_h12 = expire_h if expire_h <= 12 else expire_h - 12
                expire_str = f"{expire_h12}:{expire_min % 60:02d}{expire_ampm} PT"
                print(f"  {r['ticker']:<6} {r['family']}F{r['fold']}  "
                      f"\033[92m${lp:>9.2f}\033[0m  "
                      f"-{dip:.1%}  "
                      f"TP ${tp_price:>9.2f}  SL ${sl_price:>9.2f}  "
                      f"Cancel {expire_str}  [{r['horizon']} spike prob={r['prob']:.3f}]")

        market_trades = [r for r in signals if r['mode'] in ('static', 'grace')]
        if market_trades:
            print()
            print(f"  {'─'*W}")
            print(f"  MARKET ORDERS (enter at {ENTRY_MINUTE // 60}:{ENTRY_MINUTE % 60:02d} PST)")
            print(f"  {'─'*W}")
            for r in market_trades:
                tp_price = r['entry_price'] * (1 + r['tp_pct'])
                sl_price = r['entry_price'] * (1 - r['sl_pct'])
                grace_str = f"  Grace={r['grace_minutes']}m" if r['mode'] == 'grace' else ""
                print(f"  {r['ticker']:<6} {r['family']}F{r['fold']}  "
                      f"\033[92mBUY @ ${r['close']:>9.2f}\033[0m  "
                      f"TP ${tp_price:>9.2f}  SL ${sl_price:>9.2f}{grace_str}  "
                      f"[{r['horizon']} spike prob={r['prob']:.3f}]")

    print()
    print(f"  Strategy: M_CONS_NO_F7 (3 consensus, no fold 7, max={MAX_TRADES})  |  "
          f"Score {SCORE_MINUTE//60}:{SCORE_MINUTE%60:02d} -> Entry {ENTRY_MINUTE//60}:{ENTRY_MINUTE%60:02d} PST  |  "
          f"FiLM conditioning")
    print(f"  {'='*W}")


# ==============================================================================
# BQ LOGGER
# ==============================================================================
_BQ_SCHEMA_DEFS = [
    ("run_timestamp",  "TIMESTAMP"),
    ("bar_timestamp",  "TIMESTAMP"),
    ("score_date",     "DATE"),
    ("experiment",     "STRING"),
    ("ticker",         "STRING"),
    ("seed",           "INT64"),
    ("prob",           "FLOAT64"),
    ("threshold",      "FLOAT64"),
    ("signal",         "STRING"),
    ("tp_pct",         "FLOAT64"),
    ("sl_pct",         "FLOAT64"),
    ("entry_minute",   "INT64"),
    ("close",          "FLOAT64"),
    ("filter_status",  "STRING"),
    ("tpsl_mode",      "STRING"),
    ("strategy",       "STRING"),
    ("model_version",  "STRING"),
]


def _ensure_bq_schema(bq_client):
    for col_name, col_type in _BQ_SCHEMA_DEFS:
        sql = (f"ALTER TABLE `{BQ_TABLE}` "
               f"ADD COLUMN IF NOT EXISTS `{col_name}` {col_type}")
        bq_client.query(sql).result()


def log_to_bigquery(results, run_ts):
    if bigquery is None or not results:
        return
    rows = []
    for r in results:
        family = r['family']
        fold_id = r['fold']
        mv = FAMILIES[family]['bq_versions'][fold_id]
        rows.append({
            'run_timestamp': run_ts.isoformat(),
            'bar_timestamp': r['bar_ts'].isoformat(),
            'score_date': str(r['score_date']),
            'experiment': f'{mv}_{r["horizon"]}',
            'ticker': r['ticker'],
            'seed': 0,
            'prob': r['prob'],
            'threshold': r['threshold'],
            'signal': 'GO' if r['signal'] else 'NO',
            'tp_pct': r['tp_pct'],
            'sl_pct': r['sl_pct'],
            'entry_minute': r['entry_minute'],
            'close': r['close'],
            'filter_status': 'OK',
            'tpsl_mode': r['mode'],
            'strategy': 'ST_LONG_M_CONS_NO_F7_645',
            'model_version': mv,
        })
    bq_client = bigquery.Client()
    _ensure_bq_schema(bq_client)
    try:
        errors = bq_client.insert_rows_json(BQ_TABLE, rows)
        if errors:
            print(f"   BigQuery errors: {errors}")
        else:
            print(f"   Logged {len(rows)} rows to {BQ_TABLE} (strategy=ST_LONG_M_CONS_NO_F7_645)")
    except Exception as e:
        print(f"   BigQuery failed: {e}")


# ==============================================================================
# STITCH FRESHNESS CHECK
# ==============================================================================
def _check_stitch_freshness():
    blob = gcs_client.bucket(BUCKET_NAME).blob("Data/Inference/pipeline_status.json")
    try:
        status = json.loads(blob.download_as_text())
    except Exception:
        print("   pipeline_status.json not found — stitch freshness unknown")
        return
    stitch = status.get('stitch_1', status.get('stitch', {}))
    if stitch.get('status') != 'complete':
        print("   daily_stitch_master has not run — master_inference may be stale")
        return
    ts = pd.Timestamp(stitch.get('timestamp', '2000-01-01'))
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    today = pd.Timestamp.now(tz='UTC').normalize()
    if ts.normalize() < today:
        print(f"   Stitch last ran {ts.strftime('%Y-%m-%d %H:%M')} — not today")


# ==============================================================================
# MAIN
# ==============================================================================
run_ts = datetime.now(pytz.utc)
pst_str = run_ts.astimezone(PST).strftime('%Y-%m-%d %I:%M %p PST')
print(f"\n{'='*80}")
print(f"  DAILY SCORING — ST LONG M_CONS_NO_F7 645 — {pst_str}")
print(f"{'='*80}")

_check_stitch_freshness()

now_pst = run_ts.astimezone(PST)
now_minutes = now_pst.hour * 60 + now_pst.minute
if now_minutes < SCORE_MINUTE:
    print(f"\n  SKIP — too early ({now_minutes} < {SCORE_MINUTE})")
else:
    print(f"\n  Loading deploy configs from BQ (Newest First)...")
    DEPLOY_CONFIGS, FAMILY_FOLD_CFGS = load_deploy_configs()
    DEPLOY_TICKERS = sorted(DEPLOY_CONFIGS.keys())

    print(f"\n  Loading models and scaling stats...")
    load_models_and_meta(DEPLOY_CONFIGS, FAMILY_FOLD_CFGS)

    print(f"  Building market context...")
    market_df = build_market_context()

    print(f"  Building universe returns for regime features ({len(DEPLOY_TICKERS)} tickers)...")
    _univ_rets = {}
    for _t in DEPLOY_TICKERS:
        _df = load_master(_t, tail_days=TAIL_DAYS)
        if _df is not None:
            _df = heal_data(_df.sort_index())
            _univ_rets[_t] = _df['Close'].pct_change(1)
    _univ_ret_df = pd.DataFrame(_univ_rets).reindex(market_df.index).fillna(0)
    print(f"    Universe returns: {_univ_ret_df.shape[1]} tickers loaded")

    print(f"  Computing regime features (22 cols)...")
    regime_df = compute_regime_features(market_df, _univ_ret_df)
    del _univ_ret_df, _univ_rets

    # --- Phase 1: Score each ticker with its hybrid (newest-first) model ---
    source_tickers = {}
    for t, cfg in DEPLOY_CONFIGS.items():
        key = (cfg['family'], cfg['fold'])
        source_tickers.setdefault(key, []).append(t)

    results = []
    for (family, fold_id) in sorted(source_tickers.keys()):
        tickers = sorted(source_tickers[(family, fold_id)])
        print(f"\n  Scoring {family} F{fold_id} ({len(tickers)} tickers)...")
        for ticker in tickers:
            cfg = DEPLOY_CONFIGS[ticker]
            arch = FAMILIES[cfg['family']].get('arch', 'standard')
            if arch == 'multires_adv':
                df_feat = engineer_asset_adv(ticker, market_df)
            else:
                df_feat = engineer_asset(ticker, market_df)
            if df_feat is None:
                print(f"      {ticker}: SKIP — load_master failed")
                continue
            _feat_cache[ticker] = df_feat
            is_mr = arch in ('multires', 'multires_adv')
            df_15m = build_15min_features(df_feat) if is_mr else None
            if df_15m is not None:
                _15m_cache[ticker] = df_15m
            result = score_ticker(ticker, cfg, df_feat, regime_df, run_ts, df_15m=df_15m)
            if result is not None:
                results.append(result)

    results.sort(key=lambda r: r['ticker'])
    scored = [r['ticker'] for r in results]
    print(f"\n  Scored {len(scored)}/{len(DEPLOY_CONFIGS)}: {', '.join(scored)}")

    # --- Phase 2: Score ALL scored tickers with ALL available models (consensus) ---
    print(f"\n  Phase 2: Multi-model consensus scoring ({len(results)} tickers × {len(CONSENSUS_MODELS)} consensus models)...")
    for r in results:
        r['n_fires'] = 0
        r['n_available'] = 1
        r['consensus'] = None
        r['best_fire_prob'] = 0.0
        r['sum_conviction'] = 0.0
        r['signal'] = False

    for r in results:
        ticker = r['ticker']
        needs_adv = any(
            FAMILIES[cfgs[ticker]['family']].get('arch') == 'multires_adv'
            for (fam, fold), cfgs in FAMILY_FOLD_CFGS.items()
            if ticker in cfgs
        )
        df_feat = _feat_cache.get(ticker)
        if needs_adv and df_feat is not None and 'LR_OFI' not in df_feat.columns:
            df_feat = engineer_asset_adv(ticker, market_df)
            _feat_cache[ticker] = df_feat
        if df_feat is None:
            df_feat = engineer_asset_adv(ticker, market_df) if needs_adv else engineer_asset(ticker, market_df)
        if df_feat is None:
            continue
        df_15m = _15m_cache.get(ticker)
        if df_15m is None:
            df_15m = build_15min_features(df_feat)
        consensus = score_all_models(ticker, df_feat, regime_df, run_ts,
                                      FAMILY_FOLD_CFGS, df_15m=df_15m)
        if consensus:
            r['consensus'] = consensus
            r['n_fires'] = consensus['n_fires']
            r['n_available'] = consensus['n_available']
            r['best_fire_prob'] = consensus['best_fire_prob']
            r['sum_conviction'] = sum(s['prob'] / max(s['thresh'], 1e-6) for s in consensus['fires'])
            if consensus['n_fires'] > 0:
                fire_strs = [f"{s['family']}F{s['fold']}={s['prob']:.3f}"
                             for s in consensus['all_scores'] if s['fired']]
                skip_strs = [f"{s['family']}F{s['fold']}={s['prob']:.3f}"
                             for s in consensus['all_scores'] if not s['fired']]
                print(f"    {ticker}: {consensus['n_fires']}/{consensus['n_available']} fired "
                      f"[{', '.join(fire_strs)}]"
                      + (f"  skip=[{', '.join(skip_strs)}]" if skip_strs else ""))

    # --- Phase 3: M_CONS_NO_F7 selection — rank by (n_fires DESC, conviction DESC) ---
    candidates = [r for r in results if r['n_fires'] > 0]
    candidates.sort(key=lambda r: (-r['n_fires'], -r['sum_conviction']))

    print(f"\n  M_CONS_NO_F7: {len(candidates)} tickers with ≥1 model firing")
    if candidates:
        for i, r in enumerate(candidates):
            marker = " <- SELECTED" if i < MAX_TRADES else ""
            print(f"    #{i+1} {r['ticker']:<6} fires={r['n_fires']}/{r['n_available']}  "
                  f"conviction={r['sum_conviction']:.3f}  hyb_prob={r['prob']:.3f}{marker}")

    selected = candidates[:MAX_TRADES]
    selected_tickers = set(r['ticker'] for r in selected)
    for r in results:
        r['signal'] = r['ticker'] in selected_tickers

    if len(candidates) > MAX_TRADES:
        dropped = candidates[MAX_TRADES:]
        drop_strs = [f"{r['ticker']}({r['n_fires']}M,conv={r['sum_conviction']:.3f})" for r in dropped]
        print(f"  Dropped ({len(dropped)}): {', '.join(drop_strs)}")

    print_dashboard(results, DEPLOY_CONFIGS, run_ts)
    log_to_bigquery(results, run_ts)

    _master_cache.clear()
    _feat_cache.clear()
    _15m_cache.clear()
    keras.backend.clear_session()
    gc.collect()
