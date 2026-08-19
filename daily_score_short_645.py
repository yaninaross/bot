# %% ── Daily Scoring — ST SHORT L5-2 HYB_F7_30 645 + M_AGREEMENT ──────────────
#
# 6:45 AM variant — same models and strategy as daily_score_short.py, scored 30 min later
#
#   - 3 models (fold 5/6/7), L5-2 architecture (22 regime features)
#   - Per-ticker fold assignment from BQ (best val_b_avg_pnl per ticker)
#   - F7 tickers only included if val_b_avg_pnl >= 3.0%
#   - M_AGREEMENT: score GO signals with ALL folds, filter by spread < 0.15
#   - MAX_TRADES cap: when signals exceed cap, prioritize by lowest spread
#   - Score at minute 405 (6:45 AM PT), entry at minute 415 (6:55 AM PT)
#   - FiLM regime conditioning (no separate regime filter)
#
# Upgraded from L5-1 to L5-2 (2026-08-16): L5-2 HYB +2.84% avg, EqW +56.30%
# vs L5-1 HYB +2.64% avg, EqW +46.19% (Jul 10 – Aug 13, 33 trading days)
#
# Run AFTER daily_stitch_master.py has finished updating GCS.

import os
import sys
import json
import gc
import subprocess

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

try:
    import tf_keras
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tf_keras", "--quiet"])

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

import tensorflow as tf
import h5py

from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras import layers as _kl
from google.cloud import storage as gcs_storage

if tf.config.list_physical_devices('GPU'):
    mixed_precision.set_global_policy('mixed_float16')

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None


# ==============================================================================
# CONSTANTS
# ==============================================================================
BUCKET_NAME      = "time-series-bucket-ys"
GCS_DATA_PREFIX  = 'Algos/RA_MoE/short_L5_2_data'
GCS_MODEL_PREFIX = 'Algos/RA_MoE/short_L5_2_models'
BQ_PREDICTIONS   = "ys-time-series.scoring_logs.live_predictions"
WF_BQ_TABLE      = "ys-time-series.scoring_logs.wf_optimization_results"
INFERENCE_DIR    = "Data/Inference_v2"
TAIL_DAYS        = 25
PST              = pytz.timezone('America/Los_Angeles')
MIN_PRICE        = 2.00

FOLDS            = [5, 6, 7]
F7_VAL_B_THRESH  = 0.030
BASE_PNL_THRESH  = 0.018
SCORE_MINUTE     = 405
ENTRY_MINUTE     = 415
MAX_TRADES              = 3
AGREEMENT_SPREAD_THRESH = 0.15
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

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

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

HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}
HORIZON_FROM_LABEL = {'15m': 0, '30m': 1, '60m': 2}

EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

BQ_MODEL_VERSIONS = {
    5: 'spike_transformer_short_L5_2_fold5',
    6: 'spike_transformer_short_L5_2_fold6',
    7: 'spike_transformer_short_L5_2_fold7',
}

gcs_client = gcs_storage.Client()


# ==============================================================================
# BQ CONFIG LOADING — HYB_F7_30 logic
# ==============================================================================
def load_deploy_configs():
    if bigquery is None:
        raise RuntimeError("BigQuery not available — cannot load deploy configs")
    bq_client = bigquery.Client()

    fold_cfgs = {}
    for fold_id, model_version in BQ_MODEL_VERSIONS.items():
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
            fold_cfgs[fold_id] = {}
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
        fold_cfgs[fold_id] = cfgs
        print(f"    F{fold_id}: {len(cfgs)} tickers — {', '.join(sorted(cfgs.keys()))}")

    deploy = {}
    all_tickers = set()
    for cfgs in fold_cfgs.values():
        all_tickers |= set(cfgs.keys())

    for tk in sorted(all_tickers):
        best_fold, best_val, best_cfg = None, -999, None
        for fold_id in FOLDS:
            if tk not in fold_cfgs.get(fold_id, {}):
                continue
            cfg = fold_cfgs[fold_id][tk]
            v = cfg['val_b_avg_pnl']
            if fold_id == 7 and v < F7_VAL_B_THRESH:
                continue
            if v > best_val:
                best_fold, best_val, best_cfg = fold_id, v, cfg
        if best_cfg is not None:
            deploy[tk] = best_cfg

    fold_counts = {}
    for tk, cfg in deploy.items():
        f = cfg['fold']
        fold_counts.setdefault(f, []).append(tk)
    for f in sorted(fold_counts):
        tickers = sorted(fold_counts[f])
        print(f"    Hybrid F{f}: {len(tickers)} — {', '.join(tickers)}")
    print(f"    Total: {len(deploy)} tickers")
    return deploy, fold_cfgs


# ==============================================================================
# DATA LOADING
# ==============================================================================
_master_cache = {}


def load_master(ticker, tail_days=None):
    cache_key = (ticker, tail_days)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
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


def compute_regime_features(market_df):
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
# MODEL ARCHITECTURE
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
        self.attn = _DeStationaryAttention(d_model, n_heads, dropout)
        self.ffn = tf.keras.Sequential([
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
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout)
                       for _ in range(n_blocks)]
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
        self.mlp = tf.keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(hidden_dim, activation='relu')])
        self.gamma_proj = Dense(d_model)
        self.beta_proj = Dense(d_model)

    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        return (1.0 + self.gamma_proj(h)) * x + self.beta_proj(h)


class SpikeTransformer(Model):
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


# ==============================================================================
# MODEL LOADING — single instance, swap weights per fold
# ==============================================================================
_model = None
_fold_weights = {}
_current_fold = None
_scaling_stats = None
_regime_stats = None


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


def _ensure_fold_loaded(fold_id):
    global _current_fold
    if _current_fold != fold_id:
        _load_keras3_weights(_model, _fold_weights[fold_id])
        _current_fold = fold_id


def load_models_and_meta(deploy_configs):
    global _model, _fold_weights, _current_fold, _scaling_stats, _regime_stats

    meta_path = _ensure_file(
        f'{GCS_DATA_PREFIX}/meta.npz', '/tmp/ra_moe_short_L5_2_meta.npz')
    meta = np.load(meta_path, allow_pickle=True)
    _scaling_stats = meta['scaling_stats'].item()
    _regime_stats = meta['regime_stats'].item()

    tf.keras.backend.clear_session()
    gc.collect()

    _model = SpikeTransformer(name='spike_transformer')
    dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_F))
    dummy_reg = tf.zeros((1, N_REGIME_F))
    _model({'seq_input': dummy_seq, 'regime_input': dummy_reg})

    needed_folds = FOLDS  # Load all 3 folds for M_AGREEMENT scoring
    for fold_id in needed_folds:
        local_dir = f'/tmp/ra_moe_short_L5_2_models/fold_{fold_id}'
        weights_path = os.path.join(local_dir, 'model.weights.h5')
        if not os.path.exists(weights_path):
            os.makedirs(local_dir, exist_ok=True)
            prefix = f'{GCS_MODEL_PREFIX}/fold_{fold_id}'
            bucket = gcs_client.bucket(BUCKET_NAME)
            for blob in bucket.list_blobs(prefix=prefix):
                fname = blob.name.split('/')[-1]
                blob.download_to_filename(os.path.join(local_dir, fname))
        _fold_weights[fold_id] = weights_path
        n_tickers = sum(1 for c in deploy_configs.values() if c['fold'] == fold_id)
        print(f"   Fold {fold_id} weights cached ({n_tickers} tickers): {weights_path}")

    first_fold = needed_folds[0]
    _load_keras3_weights(_model, _fold_weights[first_fold])
    _current_fold = first_fold
    print(f"   Model built: {_model.count_params():,} params, {len(needed_folds)} folds, "
          f"{len(deploy_configs)} tickers")


def apply_scaling(df, stats, cols):
    df_s = df.copy()
    for col in cols:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


def compute_agreement(ticker, df_feat, regime_df, run_ts, deploy_cfg, fold_cfgs):
    """Score a ticker with all available folds, return agreement metrics."""
    all_probs = {}
    h_idx = deploy_cfg['horizon']

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

    df_scaled = apply_scaling(df_feat, _scaling_stats, ALL_COLS)
    XA = df_scaled[ALL_COLS].values.astype(np.float32)
    seq_batch = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]

    regime_at_entry = regime_df.iloc[regime_df.index.get_indexer(
        [df_feat.index[row_idx]], method='pad')[0]]
    regime_scaled = np.zeros(N_REGIME_F, dtype=np.float32)
    for i, col in enumerate(REGIME_COLS):
        val = float(regime_at_entry[col]) if col in regime_at_entry.index else 0.0
        if col in _regime_stats:
            s = _regime_stats[col]
            val = np.clip((val - s['med']) / s['iqr'], -10, 10)
        regime_scaled[i] = val
    regime_batch = regime_scaled[np.newaxis]

    for fold_id in FOLDS:
        if ticker not in fold_cfgs.get(fold_id, {}):
            continue
        _ensure_fold_loaded(fold_id)
        preds = _model(
            {'seq_input': seq_batch, 'regime_input': regime_batch},
            training=False
        ).numpy().flatten()
        all_probs[fold_id] = float(preds[h_idx])

    if len(all_probs) <= 1:
        return {'n_models': len(all_probs), 'spread': 0.0, 'all_probs': all_probs, 'agreed': True}

    probs = list(all_probs.values())
    spread = max(probs) - min(probs)
    return {
        'n_models': len(all_probs),
        'spread': round(spread, 4),
        'all_probs': all_probs,
        'agreed': spread < AGREEMENT_SPREAD_THRESH,
    }


# ==============================================================================
# SCORING ENGINE
# ==============================================================================
def score_ticker(ticker, cfg, df_feat, regime_df, run_ts):
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

    if 'Close' in df_feat.columns:
        raw_close = float(df_feat['Close'].iloc[row_idx])
        if raw_close < MIN_PRICE:
            print(f"      {ticker}: SKIP — price ${raw_close:.2f} < ${MIN_PRICE:.2f}")
            return None

    df_scaled = apply_scaling(df_feat, _scaling_stats, ALL_COLS)

    XA = df_scaled[ALL_COLS].values.astype(np.float32)
    seq_batch = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]

    regime_at_entry = regime_df.iloc[regime_df.index.get_indexer(
        [df_feat.index[row_idx]], method='pad')[0]]
    regime_scaled = np.zeros(N_REGIME_F, dtype=np.float32)
    for i, col in enumerate(REGIME_COLS):
        val = float(regime_at_entry[col]) if col in regime_at_entry.index else 0.0
        if col in _regime_stats:
            s = _regime_stats[col]
            val = np.clip((val - s['med']) / s['iqr'], -10, 10)
        regime_scaled[i] = val
    regime_batch = regime_scaled[np.newaxis]

    fold_id = cfg['fold']
    _ensure_fold_loaded(fold_id)
    preds = _model(
        {'seq_input': seq_batch, 'regime_input': regime_batch},
        training=False
    ).numpy().flatten()

    h_idx = cfg['horizon']
    prob = float(preds[h_idx])
    signal = prob >= cfg['prob_thresh']

    close = float(df_feat['Close'].iloc[row_idx]) if 'Close' in df_feat.columns else float('nan')

    entry_price = close
    if cfg['mode'] == 'limit_entry':
        entry_price = close * (1 + cfg.get('dip_pct', 0))

    return {
        'ticker': ticker,
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
        'max_wait_minutes': cfg.get('max_wait', 0),
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

    fold_counts = {}
    for cfg in deploy_configs.values():
        fold_counts.setdefault(cfg['fold'], 0)
        fold_counts[cfg['fold']] += 1

    print()
    print(f"  {'='*W}")
    print(f"  ST SHORT 645 SIGNALS — L5-2 HYB_F7_30 + M_AGREEMENT   {pst_str}")
    print(f"  {'='*W}")
    print(f"  Strategy: Hybrid F5+F6+F7 + M_AGREEMENT (spread<{AGREEMENT_SPREAD_THRESH}, max={MAX_TRADES})  |  "
          f"Score: {SCORE_MINUTE // 60}:{SCORE_MINUTE % 60:02d} PST  |  "
          f"Entry: {ENTRY_MINUTE // 60}:{ENTRY_MINUTE % 60:02d} PST")
    for f in sorted(fold_counts):
        f_tickers = sorted(t for t, c in deploy_configs.items() if c['fold'] == f)
        print(f"  F{f}: {', '.join(f_tickers)} ({len(f_tickers)})")

    if not results:
        print(f"\n  No tickers scored.")
        print(f"  {'='*W}")
        return

    print()
    print(f"  {'Ticker':<6} {'F':>1}  {'H':>3}  {'SIG':<7}  "
          f"{'Short@':>10}  {'TP@':>10}  {'SL@':>10}  "
          f"{'Prob':>5} / {'Thr':>5}  {'Mode':<8}  "
          f"{'Close':>8}  {'Agr':>5} {'Spr':>5}  "
          f"{'P(15m)':>6} {'P(30m)':>6} {'P(60m)':>6}")
    print(f"  {'─'*W}")

    for r in results:
        if r['signal']:
            sig_str = "\033[92m GO  \033[0m"
        else:
            sig_str = "  --  "

        entry_price = r['entry_price']
        short_str = f"${entry_price:.2f}"
        tp_str = f"${entry_price * (1 - r['tp_pct']):.2f}"
        sl_str = f"${entry_price * (1 + r['sl_pct']):.2f}"

        mode_label = _MODE_LABELS.get(r['mode'], '??')
        hp = r['all_horizon_probs']

        n_mod = r.get('n_models', 1)
        spr = r.get('spread', 0)
        agr_str = f"{n_mod}F" if r.get('agreed', True) else f"{n_mod}X"
        spr_str = f"{spr:.2f}" if n_mod > 1 else "  —"

        print(f"  {r['ticker']:<6} {r['fold']:>1}  {r['horizon']:>3}  {sig_str}  "
              f"{short_str:>10}  {tp_str:>10}  {sl_str:>10}  "
              f"{r['prob']:>5.3f} / {r['threshold']:>5.3f}  {mode_label:<8}  "
              f"{r['close']:>8.2f}  "
              f"{agr_str:>5} {spr_str:>5}  "
              f"{hp.get('15m', 0):>5.3f}  {hp.get('30m', 0):>5.3f}  {hp.get('60m', 0):>5.3f}")

    signals = [r for r in results if r['signal']]

    print()
    print(f"  {'='*W}")
    print(f"  ST SHORT 645 SUMMARY")
    print(f"  {'─'*W}")
    print(f"  Scored:    {len(results)} tickers")
    print(f"  Signals:   {len(signals)}/{len(results)}" +
          (f"  ({', '.join(r['ticker'] for r in signals)})" if signals else ""))

    if signals:
        for fold_id in FOLDS:
            fold_sigs = [r for r in signals if r['fold'] == fold_id]
            if fold_sigs:
                print(f"    F{fold_id}: {', '.join(r['ticker'] for r in fold_sigs)}")

        print()
        print(f"  \033[91mSHORT: {', '.join(r['ticker'] for r in signals)}"
              f"  ({len(signals)} position{'s' if len(signals) != 1 else ''})\033[0m")

        limit_trades = [r for r in signals if r['mode'] == 'limit_entry']
        if limit_trades:
            print()
            print(f"  {'─'*W}")
            print(f"  LIMIT SELL ORDERS (rally-wait, enter at SHORT)")
            print(f"  {'─'*W}")
            for r in limit_trades:
                rally = r.get('dip_pct', 0) or 0
                lp = r['entry_price']
                tp_price = lp * (1 - r['tp_pct'])
                sl_price = lp * (1 + r['sl_pct'])
                wait = r.get('max_wait_minutes', 30) or 30
                em = r['entry_minute']
                expire_min = em + wait
                expire_h = expire_min // 60
                expire_ampm = "AM" if expire_h < 12 else "PM"
                expire_h12 = expire_h if expire_h <= 12 else expire_h - 12
                expire_str = f"{expire_h12}:{expire_min % 60:02d}{expire_ampm} PT"
                print(f"  {r['ticker']:<6} F{r['fold']}  "
                      f"\033[91m${lp:>9.2f}\033[0m  "
                      f"+{rally:.1%}  "
                      f"TP ${tp_price:>9.2f}  SL ${sl_price:>9.2f}  "
                      f"{expire_str:>10}  {'1:00PM PT':>10}")

        market_trades = [r for r in signals if r['mode'] in ('static', 'grace', 'volscaled')]
        if market_trades:
            print()
            print(f"  {'─'*W}")
            print(f"  MARKET SHORT ORDERS (enter at {ENTRY_MINUTE // 60}:{ENTRY_MINUTE % 60:02d} PST)")
            print(f"  {'─'*W}")
            for r in market_trades:
                tp_price = r['entry_price'] * (1 - r['tp_pct'])
                sl_price = r['entry_price'] * (1 + r['sl_pct'])
                grace_str = f"  Grace={r['grace_minutes']}m" if r['mode'] == 'grace' else ""
                print(f"  {r['ticker']:<6} F{r['fold']}  "
                      f"\033[91mSHORT @ ${r['close']:>9.2f}\033[0m  "
                      f"TP ${tp_price:>9.2f}  SL ${sl_price:>9.2f}{grace_str}  "
                      f"[{r['horizon']} dump prob={r['prob']:.3f}]")

    print()
    print(f"  Strategy: L5-2 HYB_F7_30 645 + M_AGREEMENT (spread<{AGREEMENT_SPREAD_THRESH}, max={MAX_TRADES})  |  "
          f"Score {SCORE_MINUTE//60}:{SCORE_MINUTE%60:02d} -> Entry {ENTRY_MINUTE//60}:{ENTRY_MINUTE%60:02d} PST  |  "
          f"FiLM conditioning")
    print(f"  {'='*W}")


# ==============================================================================
# BIGQUERY LOGGER
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
        sql = (f"ALTER TABLE `{BQ_PREDICTIONS}` "
               f"ADD COLUMN IF NOT EXISTS `{col_name}` {col_type}")
        bq_client.query(sql).result()


def log_to_bigquery(results, run_ts):
    if bigquery is None or not results:
        return
    rows = []
    for r in results:
        fold_id = r['fold']
        rows.append({
            'run_timestamp': run_ts.isoformat(),
            'bar_timestamp': r['bar_ts'].isoformat(),
            'score_date': str(r['score_date']),
            'experiment': f'spike_transformer_short_L5_2_fold{fold_id}_{r["horizon"]}',
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
            'strategy': 'ST_SHORT_L5_2_HYB_F7_30_645_MAGR',
            'model_version': f'spike_transformer_short_L5_2_fold{fold_id}',
        })
    bq_client = bigquery.Client()
    _ensure_bq_schema(bq_client)
    try:
        errors = bq_client.insert_rows_json(BQ_PREDICTIONS, rows)
        if errors:
            print(f"   BigQuery errors: {errors}")
        else:
            print(f"   Logged {len(rows)} rows to {BQ_PREDICTIONS} (strategy=ST_SHORT_L5_2_HYB_F7_30_645_MAGR)")
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
    stitch = status.get('stitch', {})
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
print(f"  DAILY SCORING — ST SHORT L5-2 HYB_F7_30 645 + M_AGREEMENT — {pst_str}")
print(f"{'='*80}")

_check_stitch_freshness()

now_pst = run_ts.astimezone(PST)
now_minutes = now_pst.hour * 60 + now_pst.minute
if now_minutes < SCORE_MINUTE:
    print(f"\n  SKIP — too early ({now_minutes} < {SCORE_MINUTE})")
else:
    print(f"\n  Loading deploy configs from BQ (HYB_F7_30)...")
    DEPLOY_CONFIGS, FOLD_CFGS = load_deploy_configs()

    print(f"\n  Loading models and scaling stats...")
    load_models_and_meta(DEPLOY_CONFIGS)

    print(f"  Building market context...")
    market_df = build_market_context()
    _master_cache.clear()

    print(f"  Computing regime features (22 cols)...")
    regime_df = compute_regime_features(market_df)

    fold_tickers = {}
    for t, cfg in DEPLOY_CONFIGS.items():
        fold_tickers.setdefault(cfg['fold'], []).append(t)

    results = []
    for fold_id in sorted(fold_tickers.keys()):
        tickers = sorted(fold_tickers[fold_id])
        print(f"\n  Scoring fold {fold_id} ({len(tickers)} tickers)...")
        for ticker in tickers:
            cfg = DEPLOY_CONFIGS[ticker]
            df_feat = engineer_asset(ticker, market_df)
            if df_feat is None:
                print(f"      {ticker}: SKIP — load_master failed")
                continue
            result = score_ticker(ticker, cfg, df_feat, regime_df, run_ts)
            if result is not None:
                results.append(result)
            _master_cache.pop((ticker, TAIL_DAYS), None)

    results.sort(key=lambda r: r['ticker'])
    scored = [r['ticker'] for r in results]
    print(f"\n  Scored {len(scored)}/{len(DEPLOY_CONFIGS)}: {', '.join(scored)}")

    # --- M_AGREEMENT: score GO signals with all folds ---
    go_signals = [r for r in results if r['signal']]
    print(f"\n  Computing agreement for {len(go_signals)} GO signals across all folds...")

    for r in results:
        r['agreement'] = None
        r['agreed'] = True
        r['spread'] = 0.0
        r['n_models'] = 1

    for r in go_signals:
        ticker = r['ticker']
        df_feat = engineer_asset(ticker, market_df)
        if df_feat is not None:
            agr = compute_agreement(ticker, df_feat, regime_df, run_ts,
                                    DEPLOY_CONFIGS[ticker], FOLD_CFGS)
            if agr:
                r['agreement'] = agr
                r['agreed'] = agr['agreed']
                r['spread'] = agr['spread']
                r['n_models'] = agr['n_models']
                status = "AGREED" if agr['agreed'] else "DISAGREE"
                model_strs = [f"F{fid}={p:.3f}" for fid, p in agr['all_probs'].items()]
                print(f"    {ticker}: {status} spread={agr['spread']:.3f} "
                      f"({agr['n_models']} folds: {', '.join(model_strs)})")
            _master_cache.pop((ticker, TAIL_DAYS), None)

    # --- Apply agreement filter + MAX_TRADES cap ---
    agreed_signals = [r for r in go_signals if r['agreed']]
    rejected_signals = [r for r in go_signals if not r['agreed']]

    if rejected_signals:
        print(f"\n  M_AGREEMENT filter: {len(rejected_signals)} rejected "
              f"(spread >= {AGREEMENT_SPREAD_THRESH}): "
              f"{', '.join(r['ticker'] for r in rejected_signals)}")
        for r in rejected_signals:
            r['signal'] = False

    if len(agreed_signals) > MAX_TRADES:
        agreed_signals.sort(key=lambda r: r['spread'])
        kept = agreed_signals[:MAX_TRADES]
        dropped = agreed_signals[MAX_TRADES:]
        print(f"\n  MAX_TRADES cap ({MAX_TRADES}): keeping {', '.join(r['ticker'] for r in kept)}")
        drop_strs = [f"{r['ticker']}(spread={r['spread']:.3f})" for r in dropped]
        print(f"  Dropped: {', '.join(drop_strs)}")
        for r in dropped:
            r['signal'] = False
    elif agreed_signals:
        print(f"\n  {len(agreed_signals)} signals within MAX_TRADES={MAX_TRADES} cap")

    print_dashboard(results, DEPLOY_CONFIGS, run_ts)
    log_to_bigquery(results, run_ts)

    _master_cache.clear()
    tf.keras.backend.clear_session()
    gc.collect()
