# %% ── ST LONG 645 COMPARISON — Current (ASL6) vs New (ASL6-645) @ 6:45 ──────
#
# A/B comparison of two LONG Spike Transformer strategies at 6:45 AM:
#
#   CURRENT:  L5-1 + ASL6       → score 6:45 PT, entry 6:55 PT
#   NEW_645:  L5-1 + ASL6-645   → score 6:45 PT, entry 6:55 PT
#
# Both use NEWEST_FIRST + M_AGREEMENT (spread < 0.15, MAX_TRADES = 3).
# The only model swap is ASL6 → ASL6-645 (retrained with entry-offset targets).
# L5-1 uses same BQ configs in both strategies (identical results for L5-1 tickers).
#
# OUTPUT:
#   1. Head-to-head strategy comparison (with M_AGREEMENT filter)
#   2. Per-ticker breakdown (which tickers improved/degraded)
#   3. Fold-to-fold comparison (ASL6 vs ASL6-645, raw — no agreement filter)
#   4. Charts: equity curves, fold heatmap, per-ticker waterfall
#
# GPU recommended. Colab-ready.

import os, sys, gc, ctypes, subprocess, warnings
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
    auth.authenticate_user()
except (ImportError, Exception):
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

BACKTEST_START   = '2026-08-03'
BACKTEST_END     = '2026-08-18'

TAIL_DAYS        = 45

EXIT_TIME_MIN    = 780
SLIPPAGE_PCT     = 0.0015

BASE_PNL_THRESH  = 0.018
MIN_TRADES_PER_MONTH = 2.0
MAX_TRADES               = 3
AGREEMENT_SPREAD_THRESH  = 0.15

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


# ==============================================================================
# STRATEGY DEFINITIONS — both at 6:45 AM
# ==============================================================================
SCORE_MINUTE = 405   # 6:45 AM PST
ENTRY_MINUTE = 415   # 6:55 AM PST

CURRENT_FAMILIES = {
    'L5-1': {
        'gcs_data':   'Algos/RA_MoE/long_L5_1_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_1_models',
        'bq_versions': {
            5: 'spike_transformer_long_L5_1_fold5',
            6: 'spike_transformer_long_L5_1_fold6',
            7: 'spike_transformer_long_L5_1_fold7',
        },
    },
    'ASL6': {
        'gcs_data':   'Algos/RA_MoE/long_L5_2_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_2_asl6_models',
        'bq_versions': {
            5: 'spike_transformer_long_L5_2_asl6_fold5',
            6: 'spike_transformer_long_L5_2_asl6_fold6',
            7: 'spike_transformer_long_L5_2_asl6_fold7',
        },
    },
}

CURRENT_ORDER = [
    ('L5-1', 7), ('ASL6', 7),
    ('L5-1', 6), ('ASL6', 6),
    ('L5-1', 5), ('ASL6', 5),
]

NEW645_FAMILIES = {
    'L5-1': {
        'gcs_data':   'Algos/RA_MoE/long_L5_1_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_1_models',
        'bq_versions': {
            5: 'spike_transformer_long_L5_1_fold5',
            6: 'spike_transformer_long_L5_1_fold6',
            7: 'spike_transformer_long_L5_1_fold7',
        },
    },
    'ASL6-645': {
        'gcs_data':   'Algos/RA_MoE/long_ASL6_645_data',
        'gcs_models': 'Algos/RA_MoE/long_ASL6_645_models',
        'bq_versions': {
            5: 'spike_transformer_long_ASL6_645_fold5',
            6: 'spike_transformer_long_ASL6_645_fold6',
            7: 'spike_transformer_long_ASL6_645_fold7',
        },
    },
}

NEW645_ORDER = [
    ('L5-1', 7), ('ASL6-645', 7),
    ('L5-1', 6), ('ASL6-645', 6),
    ('L5-1', 5), ('ASL6-645', 5),
]

ALL_FAMILIES = {**CURRENT_FAMILIES, **NEW645_FAMILIES}

FOLDS = [5, 6, 7]
HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}
HORIZON_FROM_LABEL = {'15m': 0, '30m': 1, '60m': 2}

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

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
# BQ CONFIG LOADING — parameterized by strategy
# ==============================================================================
def load_strategy_configs(families, order, label):
    if bigquery is None:
        raise RuntimeError("BigQuery not available")
    bq_client = bigquery.Client()

    family_fold_cfgs = {}
    for family_name, fam in families.items():
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
                print(f"    [{label}] {family_name} F{fold_id}: 0 tickers")
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
            print(f"    [{label}] {family_name} F{fold_id}: {len(cfgs)} tickers")

    deploy = {}
    for family, fold in order:
        for tk, cfg in family_fold_cfgs.get((family, fold), {}).items():
            if tk not in deploy:
                deploy[tk] = cfg

    source_counts = {}
    for tk, cfg in deploy.items():
        key = f"{cfg['family']} F{cfg['fold']}"
        source_counts.setdefault(key, []).append(tk)
    for src in sorted(source_counts):
        tickers = sorted(source_counts[src])
        print(f"    [{label}] {src}: {len(tickers)} — {', '.join(tickers)}")
    print(f"    [{label}] Total: {len(deploy)} tickers (Newest First)")
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
    local_path = f"/tmp/_bt_645_{file_ticker}_master.parquet"
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


# ==============================================================================
# MODEL LOADING (h5py weight loader)
# ==============================================================================
_model = None
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

def _ensure_fold_loaded(family, fold_id):
    global _current_key
    key = (family, fold_id)
    if _current_key != key:
        _load_keras3_weights(_model, _fold_weights[key])
        _current_key = key


def load_all_models_and_meta(all_deploy_configs, fold_cfgs_list=None):
    global _model, _fold_weights, _current_key, _family_meta
    all_cfgs = []
    for cfgs in all_deploy_configs.values():
        all_cfgs.extend(cfgs.values())
    if fold_cfgs_list:
        for fold_cfgs in fold_cfgs_list:
            for cfgs in fold_cfgs.values():
                all_cfgs.extend(cfgs.values())

    needed_families = sorted(set(cfg['family'] for cfg in all_cfgs))
    for family_name in needed_families:
        if family_name in _family_meta:
            continue
        fam = ALL_FAMILIES[family_name]
        local_meta = f'/tmp/ra_moe_long_{family_name}_meta.npz'
        meta_path = _ensure_file(f"{fam['gcs_data']}/meta.npz", local_meta)
        meta = np.load(meta_path, allow_pickle=True)
        _family_meta[family_name] = {
            'scaling_stats': meta['scaling_stats'].item(),
            'regime_stats': meta['regime_stats'].item(),
        }
        print(f"   {family_name} meta loaded")

    keras.backend.clear_session()
    gc.collect()

    _model = SpikeTransformer(name='spike_transformer')
    dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_F))
    dummy_reg = tf.zeros((1, N_REGIME_F))
    _model({'seq_input': dummy_seq, 'regime_input': dummy_reg})

    needed_keys = sorted(set((cfg['family'], cfg['fold']) for cfg in all_cfgs))
    for family_name, fold_id in needed_keys:
        if (family_name, fold_id) in _fold_weights:
            continue
        fam = ALL_FAMILIES[family_name]
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
        print(f"   {family_name} F{fold_id} weights cached")

    first_key = needed_keys[0]
    _load_keras3_weights(_model, _fold_weights[first_key])
    _current_key = first_key
    print(f"   Model built: {_model.count_params():,} params, "
          f"{len(needed_keys)} (family,fold) combos")


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
# NUMBA TRADE SIMULATION — LONG direction
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
        ep = prices[idx]; tp_p = ep * (1 + tp); sl_p = ep * (1 - sl)
        oc = 0; xp = ep; found = False
        for j in range(idx + 1, np_):
            if date_ints[j] != ed:
                xp = prices[j - 1]; break
            if time_mins[j] > exit_time:
                xp = prices[j]; break
            if highs[j] >= tp_p: xp = tp_p; oc = 1; found = True; break
            if lows[j] <= sl_p: xp = sl_p; oc = -1; found = True; break
        if not found and j == np_ - 1:
            xp = prices[-1]
        pnl_a[tc] = (xp - ep) / ep; date_a[tc] = ed
        entry_a[tc] = ep; exit_a[tc] = xp; out_a[tc] = oc
        tc += 1; ltd = ed; cp += 1
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]

@jit(nopython=True)
def run_trade_sim_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
                        tp, sl, exit_time, grace_min):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        ep = prices[idx]; tp_p = ep * (1 + tp); sl_p = ep * (1 - sl)
        entry_t = time_mins[idx]; grace_end = entry_t + grace_min
        oc = 0; xp = ep; found = False
        for j in range(idx + 1, np_):
            if date_ints[j] != ed: xp = prices[j - 1]; break
            if time_mins[j] > exit_time: xp = prices[j]; break
            if highs[j] >= tp_p: xp = tp_p; oc = 1; found = True; break
            if time_mins[j] >= grace_end:
                if lows[j] <= sl_p: xp = sl_p; oc = -1; found = True; break
        if not found and j == np_ - 1: xp = prices[-1]
        pnl_a[tc] = (xp - ep) / ep; date_a[tc] = ed
        entry_a[tc] = ep; exit_a[tc] = xp; out_a[tc] = oc
        tc += 1; ltd = ed; cp += 1
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]

@jit(nopython=True)
def run_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints, time_mins,
                              tp, sl, exit_time, dip_pct, max_wait):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        ref_p = prices[idx]; limit_p = ref_p * (1 - dip_pct)
        signal_t = time_mins[idx]; deadline = signal_t + max_wait
        filled = False; fill_idx = -1
        for j in range(idx, np_):
            if date_ints[j] != ed: break
            if time_mins[j] > deadline: break
            if lows[j] <= limit_p:
                filled = True; fill_idx = j; break
        if not filled: cp += 1; continue
        ep = limit_p; tp_p = ep * (1 + tp); sl_p = ep * (1 - sl)
        oc = 0; xp = ep; found = False
        for j in range(fill_idx + 1, np_):
            if date_ints[j] != ed: xp = prices[j - 1]; break
            if time_mins[j] > exit_time: xp = prices[j]; break
            if highs[j] >= tp_p: xp = tp_p; oc = 1; found = True; break
            if lows[j] <= sl_p: xp = sl_p; oc = -1; found = True; break
        if not found and j == np_ - 1: xp = prices[-1]
        pnl_a[tc] = (xp - ep) / ep; date_a[tc] = ed
        entry_a[tc] = ep; exit_a[tc] = xp; out_a[tc] = oc
        tc += 1; ltd = ed; cp += 1
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]


def _run_sim(indices, ohlcv, cfg):
    p, h, l, d, t = ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'].astype(np.int64)
    m = cfg['mode']
    if m == 'grace':
        r = run_trade_sim_grace(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'],
                                EXIT_TIME_MIN, cfg.get('grace_minutes', 0))
    elif m == 'limit_entry':
        r = run_trade_sim_limit_entry(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'],
                                      EXIT_TIME_MIN, cfg.get('dip_pct', 0), cfg.get('max_wait', 30))
    else:
        r = run_trade_sim(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)
    pnl, dd, ep, xp, oc = r
    if len(pnl) > 0 and SLIPPAGE_PCT > 0:
        pnl = pnl - 2 * SLIPPAGE_PCT
    return pnl, dd, ep, xp, oc


# ==============================================================================
# SCORING + TRADE SIM — parameterized by score_minute / entry_minute
# ==============================================================================
def score_and_backtest_ticker(ticker, cfg, df_feat, regime_df,
                               score_minute, entry_minute):
    family = cfg['family']
    scaling_stats = _family_meta[family]['scaling_stats']
    regime_stats = _family_meta[family]['regime_stats']

    df_scaled = apply_scaling(df_feat, scaling_stats, ALL_COLS)

    bt_start = pd.Timestamp(BACKTEST_START, tz='UTC')
    bt_end = pd.Timestamp(BACKTEST_END, tz='UTC') + pd.Timedelta(days=1)

    score_bars = []
    for i in range(TIME_STEPS, len(df_feat)):
        ts = df_feat.index[i]
        if ts < bt_start or ts >= bt_end:
            continue
        bar_pst = ts.tz_convert(PST)
        bar_min = bar_pst.hour * 60 + bar_pst.minute
        if bar_min != score_minute:
            continue
        if 'Volume' in df_feat.columns and float(df_feat['Volume'].iloc[i]) == 0:
            continue
        score_bars.append(i)

    if not score_bars:
        return [], {}

    fold_id = cfg['fold']
    _ensure_fold_loaded(family, fold_id)

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

        preds = _model(
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
            }

    if not signal_dates:
        return [], {}

    signal_bar_map = {date_int: row_idx for row_idx, date_int, prob in signal_dates}

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
            if bt_date_int == date_int and bt_min == entry_minute:
                entry_indices.append(j)
                signal_probs[date_int] = prob
                break

    if not entry_indices:
        return [], signal_bar_map

    pnl, dates, ep, xp, outcomes = _run_sim(
        np.array(entry_indices, dtype=np.int64), ohlcv, cfg)

    trades = []
    for i in range(len(pnl)):
        d = int(dates[i])
        date_str = f"{d//10000}-{(d%10000)//100:02d}-{d%100:02d}"
        meta = signal_meta.get(d, {})
        prob = signal_probs.get(d, 0.0)
        trades.append({
            'Ticker': ticker,
            'Date': date_str,
            'PnL': float(pnl[i]),
            'Entry': float(ep[i]),
            'Exit': float(xp[i]),
            'Outcome': 'TP' if outcomes[i]==1 else 'SL' if outcomes[i]==-1 else 'EOD',
            'Mode': cfg['mode'],
            'Horizon': HORIZON_LABELS[h_idx],
            'Family': f"{family} F{fold_id}",
            'Prob': prob,
            'Thresh': cfg['prob_thresh'],
            'TP_Pct': cfg['tp_pct'],
            'SL_Pct': cfg['sl_pct'],
            'Val_B': cfg['val_b_avg_pnl'],
        })
    return trades, signal_bar_map


# ==============================================================================
# COMPUTE STATS
# ==============================================================================
def compute_stats(trades_df):
    if trades_df.empty:
        return {}
    dates = sorted(trades_df['Date'].unique())
    daily_ew = []
    for date in dates:
        day = trades_df[trades_df['Date'] == date]
        daily_ew.append({
            'Date': date,
            'EW': day['PnL'].mean(),
            'N': len(day),
        })
    ew_vals = [d['EW'] for d in daily_ew]
    n_vals = [d['N'] for d in daily_ew]
    cum = np.cumsum(ew_vals)
    avg = np.mean(ew_vals)
    std = np.std(ew_vals) if len(ew_vals) > 1 else 0
    sharpe = (avg / std * np.sqrt(252)) if std > 0 else 0
    total_trades = sum(n_vals)
    win_trades = (trades_df['PnL'] > 0).sum()
    avg_pnl = trades_df['PnL'].mean()
    max_dd = 0
    peak = 0
    for c in cum:
        peak = max(peak, c)
        dd = peak - c
        max_dd = max(max_dd, dd)
    win_days = sum(1 for v in ew_vals if v > 0)
    return {
        'cum_ew': cum[-1],
        'avg_daily': avg,
        'avg_pnl_trade': avg_pnl,
        'std_daily': std,
        'sharpe': sharpe,
        'total_trades': total_trades,
        'avg_trades_day': np.mean(n_vals),
        'n_days': len(daily_ew),
        'win_rate': win_trades / total_trades if total_trades > 0 else 0,
        'win_day_pct': win_days / len(daily_ew) if daily_ew else 0,
        'max_drawdown': max_dd,
        'daily_ew': daily_ew,
        'tp_rate': (trades_df['Outcome'] == 'TP').sum() / total_trades if total_trades > 0 else 0,
    }


# ==============================================================================
# M_AGREEMENT — multi-model agreement scoring
# ==============================================================================
def compute_agreement_for_signals(ticker, cfg, df_feat, regime_df,
                                   signal_bar_map, family_fold_cfgs):
    h_idx = cfg['horizon']
    results = {}

    for date_int, bar_idx in signal_bar_map.items():
        all_probs = {}
        for (fam_name, fold_id), cfgs in family_fold_cfgs.items():
            if ticker not in cfgs:
                continue
            fam_cfg = cfgs[ticker]
            family = fam_cfg['family']
            scaling_stats = _family_meta[family]['scaling_stats']
            regime_stats = _family_meta[family]['regime_stats']

            df_scaled = apply_scaling(df_feat, scaling_stats, ALL_COLS)
            XA = df_scaled[ALL_COLS].values.astype(np.float32)
            if bar_idx < TIME_STEPS:
                continue
            seq_batch = XA[bar_idx - TIME_STEPS:bar_idx][np.newaxis]

            regime_at = regime_df.iloc[regime_df.index.get_indexer(
                [df_feat.index[bar_idx]], method='pad')[0]]
            regime_scaled = np.zeros(N_REGIME_F, dtype=np.float32)
            for ri, col in enumerate(REGIME_COLS):
                val = float(regime_at[col]) if col in regime_at.index else 0.0
                if col in regime_stats:
                    s = regime_stats[col]
                    val = np.clip((val - s['med']) / s['iqr'], -10, 10)
                regime_scaled[ri] = val
            regime_batch = regime_scaled[np.newaxis]

            _ensure_fold_loaded(fam_name, fold_id)
            preds = _model(
                {'seq_input': seq_batch, 'regime_input': regime_batch},
                training=False
            ).numpy().flatten()
            all_probs[(fam_name, fold_id)] = float(preds[h_idx])

        spread = (max(all_probs.values()) - min(all_probs.values())) if len(all_probs) > 1 else 0.0
        date_str = f"{date_int//10000}-{(date_int%10000)//100:02d}-{date_int%100:02d}"
        results[(ticker, date_str)] = {
            'spread': round(spread, 4),
            'agreed': spread < AGREEMENT_SPREAD_THRESH,
            'n_models': len(all_probs),
            'all_probs': all_probs,
        }

    return results


def apply_agreement_filter(trades, agreement_data, max_trades, spread_thresh):
    if not trades:
        return []

    dates = sorted(set(t['Date'] for t in trades))
    filtered = []
    rejected_count = 0
    capped_count = 0

    for date in dates:
        day_trades = [t for t in trades if t['Date'] == date]
        agreed = []
        for t in day_trades:
            agr = agreement_data.get((t['Ticker'], t['Date']), {})
            t['spread'] = agr.get('spread', 0.0)
            t['n_models'] = agr.get('n_models', 1)
            if agr.get('agreed', True):
                agreed.append(t)
            else:
                rejected_count += 1

        agreed.sort(key=lambda t: t.get('spread', 0))
        if len(agreed) > max_trades:
            capped_count += len(agreed) - max_trades
            agreed = agreed[:max_trades]
        filtered.extend(agreed)

    n_raw = len(trades)
    print(f"      Filter: {n_raw} raw → {len(filtered)} kept "
          f"({rejected_count} rejected by spread, {capped_count} capped at {max_trades}/day)")
    return filtered


# ==============================================================================
# MAIN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  ST LONG 645 — MODEL COMPARISON: ASL6 vs ASL6-645 (both @ 6:45)")
print(f"  {BACKTEST_START} -> {BACKTEST_END}")
print(f"  CURRENT: L5-1 + ASL6     @ {SCORE_MINUTE//60}:{SCORE_MINUTE%60:02d}/{ENTRY_MINUTE//60}:{ENTRY_MINUTE%60:02d} PT")
print(f"  NEW_645: L5-1 + ASL6-645 @ {SCORE_MINUTE//60}:{SCORE_MINUTE%60:02d}/{ENTRY_MINUTE//60}:{ENTRY_MINUTE%60:02d} PT")
print(f"  M_AGREEMENT: spread<{AGREEMENT_SPREAD_THRESH}, MAX_TRADES={MAX_TRADES}")
print(f"  Exit: {EXIT_TIME_MIN//60}:{EXIT_TIME_MIN%60:02d} PT | "
      f"Slippage: {2*SLIPPAGE_PCT:.2%} RT | Deploy threshold: >{BASE_PNL_THRESH:.1%}")
print(f"{'='*W}")


# ==============================================================================
# PHASE 1: Load BQ configs for both strategies
# ==============================================================================
print(f"\n  Phase 1: Loading BQ configs...")

print(f"\n  --- CURRENT strategy (L5-1 + ASL6) ---")
current_configs, current_fold_cfgs = load_strategy_configs(
    CURRENT_FAMILIES, CURRENT_ORDER, 'CURRENT')

print(f"\n  --- NEW_645 strategy (L5-1 + ASL6-645) ---")
new645_configs, new645_fold_cfgs = load_strategy_configs(
    NEW645_FAMILIES, NEW645_ORDER, 'NEW_645')

current_tickers = sorted(current_configs.keys())
new645_tickers = sorted(new645_configs.keys())
all_tickers = sorted(set(current_tickers) | set(new645_tickers))
common_tickers = sorted(set(current_tickers) & set(new645_tickers))

print(f"\n  CURRENT tickers: {len(current_tickers)}")
print(f"  NEW_645 tickers:  {len(new645_tickers)}")
print(f"  Common (both):    {len(common_tickers)}")
print(f"  Only CURRENT:     {sorted(set(current_tickers) - set(new645_tickers))}")
print(f"  Only NEW_645:     {sorted(set(new645_tickers) - set(current_tickers))}")


# ==============================================================================
# PHASE 2: Load models and shared data
# ==============================================================================
print(f"\n  Phase 2: Loading models and meta...")
load_all_models_and_meta(
    {'current': current_configs, 'new645': new645_configs},
    fold_cfgs_list=[current_fold_cfgs, new645_fold_cfgs]
)

print(f"\n  Building market context...")
market_df = build_market_context()
_master_cache.clear()

print(f"  Computing regime features (22 cols)...")
regime_df = compute_regime_features(market_df)


# ==============================================================================
# PHASE 3: Score CURRENT strategy (L5-1 + ASL6 @ 6:45) + M_AGREEMENT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 3: CURRENT — L5-1 + ASL6 @ 6:45 + M_AGREEMENT")
print(f"{'='*W}")

current_raw_trades = []
current_signal_bars = {}
for ticker in current_tickers:
    cfg = current_configs[ticker]
    print(f"    {ticker} ({cfg['family']} F{cfg['fold']}, {cfg['mode']}, "
          f"H={HORIZON_LABELS[cfg['horizon']]})...", end="")
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is None:
        print(" SKIP (load failed)")
        continue
    trades, sig_bars = score_and_backtest_ticker(ticker, cfg, df_feat, regime_df,
                                                  SCORE_MINUTE, ENTRY_MINUTE)
    if trades:
        print(f" {len(trades)}t raw, avg={np.mean([t['PnL'] for t in trades]):+.2%}")
        current_raw_trades.extend(trades)
        if sig_bars:
            current_signal_bars[ticker] = sig_bars
    else:
        print(" 0 signals")

print(f"\n  Computing agreement for {len(current_signal_bars)} tickers with signals...")
current_agreement = {}
for ticker, sig_bars in current_signal_bars.items():
    cfg = current_configs[ticker]
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is None:
        continue
    agr = compute_agreement_for_signals(
        ticker, cfg, df_feat, regime_df, sig_bars, current_fold_cfgs)
    current_agreement.update(agr)
    n_agreed = sum(1 for v in agr.values() if v['agreed'])
    if agr:
        spreads = [v['spread'] for v in agr.values()]
        print(f"    {ticker}: {n_agreed}/{len(agr)} agreed "
              f"(spread {min(spreads):.3f}-{max(spreads):.3f}, "
              f"{list(agr.values())[0]['n_models']}M)")
    _master_cache.pop((ticker, TAIL_DAYS), None)

_master_cache.clear()

print(f"\n  Applying M_AGREEMENT filter...")
current_trades = apply_agreement_filter(
    current_raw_trades, current_agreement, MAX_TRADES, AGREEMENT_SPREAD_THRESH)


# ==============================================================================
# PHASE 4: Score NEW_645 strategy (L5-1 + ASL6-645 @ 6:45) + M_AGREEMENT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 4: NEW_645 — L5-1 + ASL6-645 @ 6:45 + M_AGREEMENT")
print(f"{'='*W}")

new645_raw_trades = []
new645_signal_bars = {}
for ticker in new645_tickers:
    cfg = new645_configs[ticker]
    print(f"    {ticker} ({cfg['family']} F{cfg['fold']}, {cfg['mode']}, "
          f"H={HORIZON_LABELS[cfg['horizon']]})...", end="")
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is None:
        print(" SKIP (load failed)")
        continue
    trades, sig_bars = score_and_backtest_ticker(ticker, cfg, df_feat, regime_df,
                                                  SCORE_MINUTE, ENTRY_MINUTE)
    if trades:
        print(f" {len(trades)}t raw, avg={np.mean([t['PnL'] for t in trades]):+.2%}")
        new645_raw_trades.extend(trades)
        if sig_bars:
            new645_signal_bars[ticker] = sig_bars
    else:
        print(" 0 signals")

print(f"\n  Computing agreement for {len(new645_signal_bars)} tickers with signals...")
new645_agreement = {}
for ticker, sig_bars in new645_signal_bars.items():
    cfg = new645_configs[ticker]
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is None:
        continue
    agr = compute_agreement_for_signals(
        ticker, cfg, df_feat, regime_df, sig_bars, new645_fold_cfgs)
    new645_agreement.update(agr)
    n_agreed = sum(1 for v in agr.values() if v['agreed'])
    if agr:
        spreads = [v['spread'] for v in agr.values()]
        print(f"    {ticker}: {n_agreed}/{len(agr)} agreed "
              f"(spread {min(spreads):.3f}-{max(spreads):.3f}, "
              f"{list(agr.values())[0]['n_models']}M)")
    _master_cache.pop((ticker, TAIL_DAYS), None)

_master_cache.clear()

print(f"\n  Applying M_AGREEMENT filter...")
new645_trades = apply_agreement_filter(
    new645_raw_trades, new645_agreement, MAX_TRADES, AGREEMENT_SPREAD_THRESH)


# ==============================================================================
# PHASE 5: HEAD-TO-HEAD COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 5: HEAD-TO-HEAD — M_AGREEMENT FILTERED")
print(f"{'='*W}")

df_cur = pd.DataFrame(current_trades) if current_trades else pd.DataFrame()
df_new = pd.DataFrame(new645_trades) if new645_trades else pd.DataFrame()

stats_cur = compute_stats(df_cur) if not df_cur.empty else {}
stats_new = compute_stats(df_new) if not df_new.empty else {}

df_cur_raw = pd.DataFrame(current_raw_trades) if current_raw_trades else pd.DataFrame()
df_new_raw = pd.DataFrame(new645_raw_trades) if new645_raw_trades else pd.DataFrame()
stats_cur_raw = compute_stats(df_cur_raw) if not df_cur_raw.empty else {}
stats_new_raw = compute_stats(df_new_raw) if not df_new_raw.empty else {}

print(f"\n  RAW (before M_AGREEMENT):")
print(f"    CURRENT: {len(current_raw_trades)} trades, "
      f"avg={stats_cur_raw.get('avg_pnl_trade', 0):+.2%}, "
      f"cum EW={stats_cur_raw.get('cum_ew', 0):+.2%}")
print(f"    NEW_645: {len(new645_raw_trades)} trades, "
      f"avg={stats_new_raw.get('avg_pnl_trade', 0):+.2%}, "
      f"cum EW={stats_new_raw.get('cum_ew', 0):+.2%}")

print(f"\n  AFTER M_AGREEMENT (spread<{AGREEMENT_SPREAD_THRESH}, max {MAX_TRADES}/day):")

print(f"\n  {'Metric':<22} {'CURRENT':>12} {'NEW_645':>12} {'Delta':>12}")
print(f"  {'─'*60}")

comparison_metrics = [
    ('Total Trades',      'total_trades', lambda v: f"{v:.0f}"),
    ('Active Days',       'n_days',       lambda v: f"{v:.0f}"),
    ('Avg Trades/Day',    'avg_trades_day', lambda v: f"{v:.1f}"),
    ('Avg PnL/Trade',     'avg_pnl_trade', lambda v: f"{v:+.2%}"),
    ('Win Rate',          'win_rate',     lambda v: f"{v:.0%}"),
    ('TP Rate',           'tp_rate',      lambda v: f"{v:.0%}"),
    ('Cum EW PnL',        'cum_ew',       lambda v: f"{v:+.2%}"),
    ('Avg Daily EW',      'avg_daily',    lambda v: f"{v:+.2%}"),
    ('Sharpe',            'sharpe',       lambda v: f"{v:.1f}"),
    ('Win Day %',         'win_day_pct',  lambda v: f"{v:.0%}"),
    ('Max Drawdown',      'max_drawdown', lambda v: f"{v:.2%}"),
]

for label, key, fmt in comparison_metrics:
    v_cur = stats_cur.get(key, 0)
    v_new = stats_new.get(key, 0)
    delta = v_new - v_cur
    delta_str = fmt(delta) if key not in ('total_trades', 'n_days') else f"{delta:+.0f}"
    marker = ' ✓' if delta > 0 and key in ('avg_pnl_trade', 'cum_ew', 'avg_daily', 'sharpe', 'win_rate', 'tp_rate', 'win_day_pct') else ''
    marker = ' ✓' if delta < 0 and key == 'max_drawdown' else marker
    print(f"  {label:<22} {fmt(v_cur):>12} {fmt(v_new):>12} {delta_str:>12}{marker}")

cur_ew = stats_cur.get('cum_ew', 0)
new_ew = stats_new.get('cum_ew', 0)
winner = 'NEW_645' if new_ew > cur_ew else 'CURRENT'
print(f"\n  VERDICT: {winner} wins by cum EW ({abs(new_ew - cur_ew):+.2%})")


# ==============================================================================
# PER-TICKER BREAKDOWN
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PER-TICKER COMPARISON (M_AGREEMENT filtered)")
print(f"{'='*W}\n")

ticker_comparison = []
if not df_cur.empty or not df_new.empty:
    print(f"  {'Ticker':<7} {'CUR Tr':>6} {'CUR Avg':>8} {'NEW Tr':>6} {'NEW Avg':>8} "
          f"{'Delta':>8} {'Source_CUR':<12} {'Source_NEW':<12} {'Verdict':<7}")
    print(f"  {'─'*90}")

    for ticker in sorted(all_tickers):
        cur_tk = df_cur[df_cur['Ticker'] == ticker] if not df_cur.empty else pd.DataFrame()
        new_tk = df_new[df_new['Ticker'] == ticker] if not df_new.empty else pd.DataFrame()
        cur_n = len(cur_tk)
        new_n = len(new_tk)
        cur_avg = cur_tk['PnL'].mean() if cur_n > 0 else 0
        new_avg = new_tk['PnL'].mean() if new_n > 0 else 0
        delta = new_avg - cur_avg if (cur_n > 0 or new_n > 0) else 0

        src_cur = current_configs.get(ticker, {})
        src_new = new645_configs.get(ticker, {})
        src_cur_str = f"{src_cur.get('family','-')} F{src_cur.get('fold','-')}" if src_cur else '—'
        src_new_str = f"{src_new.get('family','-')} F{src_new.get('fold','-')}" if src_new else '—'

        if cur_n > 0 or new_n > 0:
            verdict = 'NEW' if delta > 0 else 'CUR' if delta < 0 else 'TIE'
            print(f"  {ticker:<7} {cur_n:>6} {cur_avg:>+7.2%} {new_n:>6} {new_avg:>+7.2%} "
                  f"{delta:>+7.2%} {src_cur_str:<12} {src_new_str:<12} {verdict:<7}")
            ticker_comparison.append({
                'Ticker': ticker, 'CUR_N': cur_n, 'CUR_Avg': cur_avg,
                'NEW_N': new_n, 'NEW_Avg': new_avg, 'Delta': delta,
                'Verdict': verdict,
            })

    if ticker_comparison:
        tc_df = pd.DataFrame(ticker_comparison)
        n_better = (tc_df['Verdict'] == 'NEW').sum()
        n_worse = (tc_df['Verdict'] == 'CUR').sum()
        n_tie = (tc_df['Verdict'] == 'TIE').sum()
        n_active = len(tc_df[(tc_df['CUR_N'] > 0) | (tc_df['NEW_N'] > 0)])
        print(f"\n  Summary: NEW_645 better for {n_better}/{n_active} tickers, "
              f"CURRENT better for {n_worse}/{n_active}, ties: {n_tie}")


# ==============================================================================
# PHASE 6: FOLD-TO-FOLD ASL6 vs ASL6-645 (raw, no agreement filter)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 6: FOLD-TO-FOLD — ASL6 vs ASL6-645 (both @ 6:45, raw)")
print(f"  Isolating model quality at same timing, no agreement filter")
print(f"{'='*W}")

fold_comparison = {}
fold_trades_cur_all = {}
fold_trades_new_all = {}

for fold_id in FOLDS:
    asl6_cfgs = current_fold_cfgs.get(('ASL6', fold_id), {})
    asl6_645_cfgs = new645_fold_cfgs.get(('ASL6-645', fold_id), {})

    overlap = sorted(set(asl6_cfgs.keys()) & set(asl6_645_cfgs.keys()))
    print(f"\n  Fold {fold_id}: ASL6 has {len(asl6_cfgs)} tickers, "
          f"ASL6-645 has {len(asl6_645_cfgs)} tickers, overlap: {len(overlap)}")

    if not overlap:
        fold_comparison[fold_id] = {'overlap': 0}
        continue

    fold_trades_cur = []
    fold_trades_new = []

    for ticker in overlap:
        df_feat = engineer_asset(ticker, market_df)
        if df_feat is None:
            continue

        cfg_asl6 = asl6_cfgs[ticker]
        trades_cur, _ = score_and_backtest_ticker(
            ticker, cfg_asl6, df_feat, regime_df,
            SCORE_MINUTE, ENTRY_MINUTE)
        for t in trades_cur:
            t['Fold'] = fold_id
        fold_trades_cur.extend(trades_cur)

        cfg_645 = asl6_645_cfgs[ticker]
        trades_new, _ = score_and_backtest_ticker(
            ticker, cfg_645, df_feat, regime_df,
            SCORE_MINUTE, ENTRY_MINUTE)
        for t in trades_new:
            t['Fold'] = fold_id
        fold_trades_new.extend(trades_new)

        _master_cache.pop((ticker, TAIL_DAYS), None)

    _master_cache.clear()

    df_fcur = pd.DataFrame(fold_trades_cur) if fold_trades_cur else pd.DataFrame()
    df_fnew = pd.DataFrame(fold_trades_new) if fold_trades_new else pd.DataFrame()

    fold_trades_cur_all[fold_id] = df_fcur
    fold_trades_new_all[fold_id] = df_fnew

    s_cur = compute_stats(df_fcur) if not df_fcur.empty else {}
    s_new = compute_stats(df_fnew) if not df_fnew.empty else {}

    fold_comparison[fold_id] = {
        'overlap': len(overlap),
        'cur_trades': s_cur.get('total_trades', 0),
        'new_trades': s_new.get('total_trades', 0),
        'cur_avg_pnl': s_cur.get('avg_pnl_trade', 0),
        'new_avg_pnl': s_new.get('avg_pnl_trade', 0),
        'cur_wr': s_cur.get('win_rate', 0),
        'new_wr': s_new.get('win_rate', 0),
        'cur_cum': s_cur.get('cum_ew', 0),
        'new_cum': s_new.get('cum_ew', 0),
        'cur_sharpe': s_cur.get('sharpe', 0),
        'new_sharpe': s_new.get('sharpe', 0),
    }


print(f"\n\n  {'─'*90}")
print(f"  FOLD-TO-FOLD SUMMARY (both @ 6:45, raw — no agreement filter)")
print(f"  {'─'*90}")
print(f"  {'Fold':<6} {'Overlap':>8} "
      f"{'ASL6 Tr':>8} {'645 Tr':>8} "
      f"{'ASL6 Avg':>9} {'645 Avg':>9} {'Δ Avg':>8} "
      f"{'ASL6 WR':>8} {'645 WR':>8} "
      f"{'ASL6 Cum':>9} {'645 Cum':>9} "
      f"{'Winner':>7}")
print(f"  {'─'*90}")

for fold_id in FOLDS:
    fc = fold_comparison[fold_id]
    if fc['overlap'] == 0:
        print(f"  F{fold_id:<5} {fc['overlap']:>8} — no overlap —")
        continue
    d_avg = fc['new_avg_pnl'] - fc['cur_avg_pnl']
    fold_w = '645' if fc['new_avg_pnl'] > fc['cur_avg_pnl'] else 'ASL6' if fc['cur_avg_pnl'] > fc['new_avg_pnl'] else 'TIE'
    print(f"  F{fold_id:<5} {fc['overlap']:>8} "
          f"{fc['cur_trades']:>8.0f} {fc['new_trades']:>8.0f} "
          f"{fc['cur_avg_pnl']:>+8.2%} {fc['new_avg_pnl']:>+8.2%} {d_avg:>+7.2%} "
          f"{fc['cur_wr']:>7.0%} {fc['new_wr']:>7.0%} "
          f"{fc['cur_cum']:>+8.2%} {fc['new_cum']:>+8.2%} "
          f"{fold_w:>7}")

folds_645_wins = sum(1 for f in FOLDS if fold_comparison[f].get('new_avg_pnl', 0) > fold_comparison[f].get('cur_avg_pnl', 0))
print(f"\n  645 wins {folds_645_wins}/{len(FOLDS)} folds by avg PnL/trade")


# ==============================================================================
# PER-TICKER FOLD DETAIL
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PER-TICKER FOLD DETAIL — ASL6 vs ASL6-645 (both @ 6:45)")
print(f"{'='*W}\n")

for fold_id in FOLDS:
    df_fcur = fold_trades_cur_all.get(fold_id, pd.DataFrame())
    df_fnew = fold_trades_new_all.get(fold_id, pd.DataFrame())
    if df_fcur.empty and df_fnew.empty:
        continue

    print(f"\n  ── Fold {fold_id} ──")
    print(f"  {'Ticker':<7} {'ASL6 Tr':>7} {'ASL6 Avg':>9} {'645 Tr':>7} "
          f"{'645 Avg':>9} {'Delta':>8} {'Winner':>7}")
    print(f"  {'─'*60}")

    all_tk = sorted(set(
        (df_fcur['Ticker'].unique().tolist() if not df_fcur.empty else []) +
        (df_fnew['Ticker'].unique().tolist() if not df_fnew.empty else [])
    ))
    for tk in all_tk:
        dk = df_fcur[df_fcur['Ticker'] == tk] if not df_fcur.empty else pd.DataFrame()
        nk = df_fnew[df_fnew['Ticker'] == tk] if not df_fnew.empty else pd.DataFrame()
        dn = len(dk); nn = len(nk)
        da = dk['PnL'].mean() if dn > 0 else 0
        na = nk['PnL'].mean() if nn > 0 else 0
        delta = na - da
        v = '645' if delta > 0 else 'ASL6' if delta < 0 else 'TIE'
        print(f"  {tk:<7} {dn:>7} {da:>+8.2%} {nn:>7} {na:>+8.2%} "
              f"{delta:>+7.2%} {v:>7}")


# ==============================================================================
# PHASE 7: CHARTS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 7: GENERATING CHARTS")
print(f"{'='*W}")

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle(f'ST LONG 645 — CURRENT (ASL6) vs NEW (ASL6-645) + M_AGREEMENT\n'
             f'{BACKTEST_START} → {BACKTEST_END} | '
             f'Both @ 6:45 PT | spread<{AGREEMENT_SPREAD_THRESH}, max {MAX_TRADES}/day',
             fontsize=13, fontweight='bold')

COLOR_CUR = '#1565C0'
COLOR_NEW = '#C62828'
COLOR_FOLD = {5: '#2E7D32', 6: '#E65100', 7: '#6A1B9A'}


# [0,0] Equity curves
ax = axes[0, 0]
for label, stats, color in [('CURRENT (ASL6)', stats_cur, COLOR_CUR),
                              ('NEW (ASL6-645)', stats_new, COLOR_NEW)]:
    daily = stats.get('daily_ew', [])
    if not daily:
        continue
    dates_dt = pd.to_datetime([d['Date'] for d in daily])
    cum = np.cumsum([d['EW'] for d in daily]) * 100
    ax.plot(dates_dt, cum, color=color, linewidth=2.5, label=label, marker='o',
            markersize=4)
ax.axhline(0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Cumulative EW PnL (%)')
ax.set_title('Strategy Equity Curves (M_AGREEMENT filtered)')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


# [0,1] Daily PnL comparison
ax = axes[0, 1]
if stats_cur.get('daily_ew') and stats_new.get('daily_ew'):
    cur_daily = {d['Date']: d['EW'] for d in stats_cur['daily_ew']}
    new_daily = {d['Date']: d['EW'] for d in stats_new['daily_ew']}
    all_dates = sorted(set(cur_daily.keys()) | set(new_daily.keys()))
    x_pos = np.arange(len(all_dates))
    bar_w = 0.35
    cur_vals = [cur_daily.get(d, 0) * 100 for d in all_dates]
    new_vals = [new_daily.get(d, 0) * 100 for d in all_dates]
    ax.bar(x_pos - bar_w/2, cur_vals, bar_w, color=COLOR_CUR, alpha=0.8, label='CURRENT')
    ax.bar(x_pos + bar_w/2, new_vals, bar_w, color=COLOR_NEW, alpha=0.8, label='NEW_645')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d[-5:] for d in all_dates], rotation=45, ha='right', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('EW PnL (%)')
    ax.set_title('Daily PnL Comparison')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)


# [0,2] Strategy metrics bar chart
ax = axes[0, 2]
metric_labels = ['Avg PnL/Tr', 'Win Rate', 'Sharpe/10', 'Cum EW']
cur_bars = [
    stats_cur.get('avg_pnl_trade', 0) * 100,
    stats_cur.get('win_rate', 0) * 100,
    stats_cur.get('sharpe', 0) / 10 * 100,
    stats_cur.get('cum_ew', 0) * 100,
]
new_bars = [
    stats_new.get('avg_pnl_trade', 0) * 100,
    stats_new.get('win_rate', 0) * 100,
    stats_new.get('sharpe', 0) / 10 * 100,
    stats_new.get('cum_ew', 0) * 100,
]
x_met = np.arange(len(metric_labels))
bar_w2 = 0.35
ax.bar(x_met - bar_w2/2, cur_bars, bar_w2, color=COLOR_CUR, alpha=0.8, label='CURRENT')
ax.bar(x_met + bar_w2/2, new_bars, bar_w2, color=COLOR_NEW, alpha=0.8, label='NEW_645')
ax.set_xticks(x_met)
ax.set_xticklabels(metric_labels, fontsize=9)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Key Metrics Comparison')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)


# [1,0] Fold-to-fold — avg PnL per trade
ax = axes[1, 0]
fold_labels = [f'F{f}' for f in FOLDS]
cur_fold_pnl = [fold_comparison[f].get('cur_avg_pnl', 0) * 100 for f in FOLDS]
new_fold_pnl = [fold_comparison[f].get('new_avg_pnl', 0) * 100 for f in FOLDS]
x_fold = np.arange(len(FOLDS))
bar_wf = 0.35
ax.bar(x_fold - bar_wf/2, cur_fold_pnl, bar_wf, color=COLOR_CUR, alpha=0.8, label='ASL6')
ax.bar(x_fold + bar_wf/2, new_fold_pnl, bar_wf, color=COLOR_NEW, alpha=0.8, label='ASL6-645')
ax.set_xticks(x_fold)
ax.set_xticklabels(fold_labels, fontsize=11)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Avg PnL/Trade (%)')
ax.set_title('Fold-to-Fold: ASL6 vs ASL6-645 (both @ 6:45)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for i, (d, n) in enumerate(zip(cur_fold_pnl, new_fold_pnl)):
    ax.text(i - bar_wf/2, d + 0.05 if d >= 0 else d - 0.15, f'{d:+.2f}%',
            ha='center', fontsize=7, color=COLOR_CUR)
    ax.text(i + bar_wf/2, n + 0.05 if n >= 0 else n - 0.15, f'{n:+.2f}%',
            ha='center', fontsize=7, color=COLOR_NEW)


# [1,1] Fold-to-fold — cumulative EW
ax = axes[1, 1]
cur_fold_cum = [fold_comparison[f].get('cur_cum', 0) * 100 for f in FOLDS]
new_fold_cum = [fold_comparison[f].get('new_cum', 0) * 100 for f in FOLDS]
ax.bar(x_fold - bar_wf/2, cur_fold_cum, bar_wf, color=COLOR_CUR, alpha=0.8, label='ASL6')
ax.bar(x_fold + bar_wf/2, new_fold_cum, bar_wf, color=COLOR_NEW, alpha=0.8, label='ASL6-645')
ax.set_xticks(x_fold)
ax.set_xticklabels(fold_labels, fontsize=11)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Cumulative EW PnL (%)')
ax.set_title('Fold-to-Fold: Cumulative PnL')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)


# [1,2] Per-ticker delta waterfall
ax = axes[1, 2]
if ticker_comparison:
    tc_sorted = sorted(ticker_comparison, key=lambda x: x['Delta'], reverse=True)
    n_show = min(15, len(tc_sorted))
    top_tickers = tc_sorted[:n_show]
    bottom_tickers = tc_sorted[-n_show:] if len(tc_sorted) > n_show else []
    show_tickers = top_tickers + [t for t in bottom_tickers if t not in top_tickers]
    show_tickers = sorted(show_tickers, key=lambda x: x['Delta'], reverse=True)

    tk_labels = [t['Ticker'] for t in show_tickers]
    tk_deltas = [t['Delta'] * 100 for t in show_tickers]
    tk_colors = [COLOR_NEW if d > 0 else COLOR_CUR for d in tk_deltas]
    y_pos = np.arange(len(tk_labels))
    ax.barh(y_pos, tk_deltas, color=tk_colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tk_labels, fontsize=7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('PnL Delta (NEW_645 - CURRENT, %)')
    ax.set_title('Per-Ticker PnL Delta (red=NEW better, blue=CUR better)')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

plt.tight_layout()
chart_path = '/tmp/backtest_st_long_645_comparison.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# FOLD EQUITY CURVES — separate chart
# ==============================================================================
if any(not fold_trades_cur_all.get(f, pd.DataFrame()).empty or
       not fold_trades_new_all.get(f, pd.DataFrame()).empty for f in FOLDS):
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 7))
    fig2.suptitle(f'Fold-to-Fold Equity Curves — ASL6 vs ASL6-645 (both @ 6:45)\n'
                  f'{BACKTEST_START} → {BACKTEST_END}',
                  fontsize=13, fontweight='bold')

    for idx, fold_id in enumerate(FOLDS):
        ax = axes2[idx]
        for label, df_fold, color, ls in [
            ('ASL6', fold_trades_cur_all.get(fold_id, pd.DataFrame()), COLOR_CUR, '-'),
            ('ASL6-645', fold_trades_new_all.get(fold_id, pd.DataFrame()), COLOR_NEW, '-'),
        ]:
            if df_fold.empty:
                continue
            stats_f = compute_stats(df_fold)
            daily = stats_f.get('daily_ew', [])
            if not daily:
                continue
            dates_dt = pd.to_datetime([d['Date'] for d in daily])
            cum = np.cumsum([d['EW'] for d in daily]) * 100
            avg_pnl = stats_f.get('avg_pnl_trade', 0)
            n_trades = stats_f.get('total_trades', 0)
            ax.plot(dates_dt, cum, color=color, linewidth=2, linestyle=ls,
                    label=f'{label} ({n_trades:.0f}t, avg={avg_pnl:+.2%})',
                    marker='o', markersize=3)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.set_ylabel('Cumulative EW PnL (%)')
        fc = fold_comparison.get(fold_id, {})
        fold_winner = '645 ✓' if fc.get('new_avg_pnl', 0) > fc.get('cur_avg_pnl', 0) else 'ASL6 ✓'
        ax.set_title(f'Fold {fold_id} — {fold_winner} (overlap: {fc.get("overlap", 0)} tickers)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    chart2_path = '/tmp/backtest_st_long_645_fold_curves.png'
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    print(f"  Fold curves saved: {chart2_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FINAL SUMMARY")
print(f"{'='*W}\n")

print(f"  STRATEGY-LEVEL (M_AGREEMENT filtered, both @ 6:45):")
print(f"    CURRENT (ASL6):     {stats_cur.get('total_trades', 0):.0f} trades, "
      f"avg PnL={stats_cur.get('avg_pnl_trade', 0):+.2%}, "
      f"cum EW={stats_cur.get('cum_ew', 0):+.2%}, "
      f"Sharpe={stats_cur.get('sharpe', 0):.1f}")
print(f"    NEW_645 (ASL6-645): {stats_new.get('total_trades', 0):.0f} trades, "
      f"avg PnL={stats_new.get('avg_pnl_trade', 0):+.2%}, "
      f"cum EW={stats_new.get('cum_ew', 0):+.2%}, "
      f"Sharpe={stats_new.get('sharpe', 0):.1f}")
print(f"    Verdict: {winner}")

cur_daily_avg = stats_cur.get('avg_daily', 0)
new_daily_avg = stats_new.get('avg_daily', 0)
deploy_tag_cur = '✓ DEPLOY' if cur_daily_avg >= BASE_PNL_THRESH else '✗ below 1.8%'
deploy_tag_new = '✓ DEPLOY' if new_daily_avg >= BASE_PNL_THRESH else '✗ below 1.8%'
print(f"\n  DEPLOY THRESHOLD ({BASE_PNL_THRESH:.1%} avg daily EW):")
print(f"    CURRENT: {cur_daily_avg:+.2%}/day → {deploy_tag_cur}")
print(f"    NEW_645: {new_daily_avg:+.2%}/day → {deploy_tag_new}")

print(f"\n  FOLD-LEVEL (ASL6 vs ASL6-645, both @ 6:45, raw):")
for fold_id in FOLDS:
    fc = fold_comparison[fold_id]
    if fc.get('overlap', 0) == 0:
        print(f"    Fold {fold_id}: no overlap")
        continue
    w = '645' if fc['new_avg_pnl'] > fc['cur_avg_pnl'] else 'ASL6'
    delta = fc['new_avg_pnl'] - fc['cur_avg_pnl']
    print(f"    Fold {fold_id}: ASL6 {fc['cur_avg_pnl']:+.2%} vs 645 {fc['new_avg_pnl']:+.2%} "
          f"(Δ={delta:+.2%}) → {w}")

if ticker_comparison:
    tc_df = pd.DataFrame(ticker_comparison)
    n_better = (tc_df['Verdict'] == 'NEW').sum()
    n_worse = (tc_df['Verdict'] == 'CUR').sum()
    print(f"\n  TICKER-LEVEL: {n_better} improved, {n_worse} degraded")


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
print(f"  645 MODEL COMPARISON COMPLETE")
print(f"{'='*W}")
