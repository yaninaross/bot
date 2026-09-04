# %% ── ST LONG NEWEST_FIRST 645 — Live Backtest (master_inference scoring) ────
#
# 6:45 AM variant — same models and strategy as backtest_st_long_newest_first_live.py,
# scored 30 min later.
#
#   - 6 models: L5-1 F5/F6/F7 + ASL6 F5/F6/F7
#   - Newest First priority: F7 > F6 > F5, L5-1 preferred within same fold
#   - Per-ticker: first (family, fold) with a BQ config wins
#   - Score at minute 405 (6:45 AM PT), entry at minute 415 (6:55 AM PT)
#   - FiLM regime conditioning (22 regime features)
#   - Numba trade simulation: static / grace / limit_entry
#
# OUTPUT: daily PnL table, per-ticker breakdown, equity curve, summary stats.
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

SCORE_MINUTE     = 405
ENTRY_MINUTE     = 415
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
N_ASSET_F        = 33
N_REGIME_F       = 22

FAMILIES = {
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

NEWEST_FIRST_ORDER = [
    ('L5-1', 7), ('ASL6', 7),
    ('L5-1', 6), ('ASL6', 6),
    ('L5-1', 5), ('ASL6', 5),
]

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
# BQ CONFIG LOADING — Newest First across L5-1 + ASL6
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
    return deploy


# ==============================================================================
# DATA LOADING (from daily_score_st_long.py — master_inference parquets)
# ==============================================================================
_master_cache = {}


def load_master(ticker, tail_days=None):
    cache_key = (ticker, tail_days)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    local_path = f"/tmp/_bt_long_645_{file_ticker}_master.parquet"
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
# MARKET CONTEXT & FEATURE ENGINEERING (from daily_score_st_long.py)
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
# MODEL ARCHITECTURE (SpikeTransformer — identical to daily_score_st_long.py)
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
# MODEL LOADING (from daily_score_st_long.py — h5py weight loader)
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

def load_models_and_meta(deploy_configs):
    global _model, _fold_weights, _current_key, _family_meta
    needed_families = sorted(set(cfg['family'] for cfg in deploy_configs.values()))
    for family_name in needed_families:
        fam = FAMILIES[family_name]
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

    needed_keys = sorted(set((cfg['family'], cfg['fold']) for cfg in deploy_configs.values()))
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
    _load_keras3_weights(_model, _fold_weights[first_key])
    _current_key = first_key
    print(f"   Model built: {_model.count_params():,} params, {len(needed_keys)} (family,fold) combos, "
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


# ==============================================================================
# NUMBA TRADE SIMULATION (from backtest_long_asl6_3fold_vs_L5_1.py)
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
# LIVE SCORING + BACKTEST — per ticker, all days in window
# ==============================================================================
def score_and_backtest_ticker(ticker, cfg, df_feat, regime_df):
    family = cfg['family']
    scaling_stats = _family_meta[family]['scaling_stats']
    regime_stats = _family_meta[family]['regime_stats']

    df_scaled = apply_scaling(df_feat, scaling_stats, ALL_COLS)

    bt_start = pd.Timestamp(BACKTEST_START, tz='UTC')
    bt_end = pd.Timestamp(BACKTEST_END, tz='UTC') + pd.Timedelta(days=1)

    df_pst = df_feat.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute

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

    XA = df_scaled[ALL_COLS].values.astype(np.float32)
    h_idx = cfg['horizon']
    signal_dates = []

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

    pnl, dates, ep, xp, outcomes = _run_sim(
        np.array(entry_indices, dtype=np.int64), ohlcv, cfg)

    trades = []
    for i in range(len(pnl)):
        d = int(dates[i])
        date_str = f"{d//10000}-{(d%10000)//100:02d}-{d%100:02d}"
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
            'Prob': signal_probs.get(d, 0.0),
            'Thresh': cfg['prob_thresh'],
        })
    return trades


# ==============================================================================
# MAIN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  ST LONG NEWEST_FIRST 645 — LIVE BACKTEST")
print(f"  {BACKTEST_START} -> {BACKTEST_END}")
print(f"  Score: {SCORE_MINUTE//60}:{SCORE_MINUTE%60:02d} PT | Entry: {ENTRY_MINUTE//60}:{ENTRY_MINUTE%60:02d} PT | "
      f"Exit: {EXIT_TIME_MIN//60}:{EXIT_TIME_MIN%60:02d} PT")
print(f"  Slippage: {2*SLIPPAGE_PCT:.2%} RT | Deploy threshold: >{BASE_PNL_THRESH:.1%}")
print(f"{'='*W}")

print(f"\n  Loading deploy configs from BQ (Newest First)...")
DEPLOY_CONFIGS = load_deploy_configs()
DEPLOY_TICKERS = sorted(DEPLOY_CONFIGS.keys())

print(f"\n  Loading models and scaling stats...")
load_models_and_meta(DEPLOY_CONFIGS)

print(f"\n  Building market context...")
market_df = build_market_context()
_master_cache.clear()

print(f"  Computing regime features (22 cols)...")
regime_df = compute_regime_features(market_df)

print(f"\n  Scoring and backtesting {len(DEPLOY_TICKERS)} tickers...")
all_trades = []
for ticker in DEPLOY_TICKERS:
    cfg = DEPLOY_CONFIGS[ticker]
    print(f"    {ticker} ({cfg['family']} F{cfg['fold']}, {cfg['mode']}, "
          f"H={HORIZON_LABELS[cfg['horizon']]})...", end="")
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is None:
        print(" SKIP (load failed)")
        continue
    trades = score_and_backtest_ticker(ticker, cfg, df_feat, regime_df)
    if trades:
        print(f" {len(trades)}t avg={np.mean([t['PnL'] for t in trades]):+.2%}")
        all_trades.extend(trades)
    else:
        print(" 0 signals")
    _master_cache.pop((ticker, TAIL_DAYS), None)

_master_cache.clear()


# ==============================================================================
# RESULTS — DAILY PNL TABLE
# ==============================================================================
if not all_trades:
    print(f"\n  NO TRADES in backtest window.")
else:
    df_trades = pd.DataFrame(all_trades)
    all_dates = sorted(df_trades['Date'].unique())
    n_trades = len(df_trades)
    n_tickers = df_trades['Ticker'].nunique()

    print(f"\n\n{'='*W}")
    print(f"  DAILY PNL — ST LONG NEWEST_FIRST 645  ({BACKTEST_START} -> {BACKTEST_END})")
    print(f"  {n_trades} trades across {n_tickers} tickers | Slippage {2*SLIPPAGE_PCT:.2%} RT")
    print(f"{'='*W}\n")

    print(f"  {'Date':<12} {'#Tr':>4} {'EW PnL':>8} {'Cum EW':>8} {'WR':>6}  {'Tickers traded'}")
    print(f"  {'─'*W}")

    cum_ew = 0.0
    daily_ew_list = []
    for date in all_dates:
        day = df_trades[df_trades['Date'] == date]
        nt = len(day)
        ew_pnl = day['PnL'].mean()
        cum_ew += ew_pnl
        wr = (day['PnL'] > 0).mean()
        daily_ew_list.append({'Date': date, 'EW': ew_pnl, 'Cum': cum_ew, 'N': nt})

        tickers_detail = []
        for _, t in day.sort_values('Ticker').iterrows():
            tickers_detail.append(f"{t['Ticker']}({t['PnL']:+.1%},{t['Outcome']})")
        print(f"  {date:<12} {nt:>4} {ew_pnl:>+7.2%} {cum_ew:>+7.2%} {wr:>5.0%}  {', '.join(tickers_detail)}")

    print(f"  {'─'*W}")
    avg_daily = np.mean([d['EW'] for d in daily_ew_list])
    std_daily = np.std([d['EW'] for d in daily_ew_list]) if len(daily_ew_list) > 1 else 0
    sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0
    print(f"  {'TOTAL':<12} {n_trades:>4} {cum_ew:>+7.2%}          "
          f"       Avg daily: {avg_daily:+.2%} | Sharpe(EW): {sharpe:.1f}")


    # ==========================================================================
    # PER-TICKER SUMMARY
    # ==========================================================================
    print(f"\n\n{'='*W}")
    print(f"  PER-TICKER SUMMARY")
    print(f"{'='*W}\n")

    print(f"  {'Ticker':<7} {'Src':<10} {'Mode':<8} {'#Tr':>4} {'Avg PnL':>8} "
          f"{'WR':>5} {'TP':>3} {'SL':>3} {'EOD':>4}  {'Daily PnL'}")
    print(f"  {'─'*W}")

    for ticker in sorted(df_trades['Ticker'].unique()):
        tk = df_trades[df_trades['Ticker'] == ticker]
        cfg = DEPLOY_CONFIGS[ticker]
        src = f"{cfg['family']} F{cfg['fold']}"
        avg = tk['PnL'].mean()
        wr = (tk['PnL'] > 0).mean()
        n_tp = (tk['Outcome'] == 'TP').sum()
        n_sl = (tk['Outcome'] == 'SL').sum()
        n_eod = (tk['Outcome'] == 'EOD').sum()
        daily_pnls = ', '.join(f"{r['Date'][-5:]}:{r['PnL']:+.1%}" for _, r in tk.iterrows())
        print(f"  {ticker:<7} {src:<10} {cfg['mode']:<8} {len(tk):>4} {avg:>+7.2%} "
              f"{wr:>4.0%} {n_tp:>3} {n_sl:>3} {n_eod:>4}  {daily_pnls}")


    # ==========================================================================
    # SUMMARY STATS
    # ==========================================================================
    print(f"\n\n{'='*W}")
    print(f"  SUMMARY STATS")
    print(f"{'='*W}")

    avg_pnl = df_trades['PnL'].mean()
    total_pnl = df_trades['PnL'].sum()
    overall_wr = (df_trades['PnL'] > 0).mean()
    n_days = len(all_dates)
    trades_per_day = n_trades / n_days if n_days > 0 else 0

    print(f"  Trades:          {n_trades}")
    print(f"  Tickers:         {n_tickers}")
    print(f"  Active days:     {n_days}")
    print(f"  Trades/day:      {trades_per_day:.1f}")
    print(f"  Avg PnL/trade:   {avg_pnl:+.2%}")
    print(f"  Total PnL:       {total_pnl:+.2%}")
    print(f"  Win rate:        {overall_wr:.0%}")
    print(f"  EW Portfolio:    {cum_ew:+.2%}")
    print(f"  EW Avg daily:    {avg_daily:+.2%}")
    print(f"  EW Sharpe:       {sharpe:.1f}")

    by_outcome = df_trades.groupby('Outcome')['PnL'].agg(['mean', 'count'])
    print(f"\n  By outcome:")
    for oc in ['TP', 'SL', 'EOD']:
        if oc in by_outcome.index:
            print(f"    {oc}: {int(by_outcome.loc[oc, 'count'])} trades, avg {by_outcome.loc[oc, 'mean']:+.2%}")

    by_mode = df_trades.groupby('Mode')['PnL'].agg(['mean', 'count'])
    print(f"\n  By mode:")
    for mode in sorted(by_mode.index):
        print(f"    {mode}: {int(by_mode.loc[mode, 'count'])} trades, avg {by_mode.loc[mode, 'mean']:+.2%}")

    by_family = df_trades.groupby('Family')['PnL'].agg(['mean', 'count'])
    print(f"\n  By source:")
    for fam in sorted(by_family.index):
        print(f"    {fam}: {int(by_family.loc[fam, 'count'])} trades, avg {by_family.loc[fam, 'mean']:+.2%}")


    # ==========================================================================
    # TODAY DETAIL
    # ==========================================================================
    today_str = BACKTEST_END
    today_trades = df_trades[df_trades['Date'] == today_str]
    if len(today_trades) > 0:
        print(f"\n\n{'='*W}")
        print(f"  TODAY ({today_str}) — TRADE DETAIL")
        print(f"{'='*W}\n")
        print(f"  {'Ticker':<7} {'Src':<10} {'H':>3} {'Mode':<8} "
              f"{'Prob':>5} {'Thr':>5} {'Entry$':>9} {'Exit$':>9} "
              f"{'PnL':>7} {'Out':<4}")
        print(f"  {'─'*80}")
        for _, t in today_trades.sort_values('Ticker').iterrows():
            print(f"  {t['Ticker']:<7} {t['Family']:<10} {t['Horizon']:>3} {t['Mode']:<8} "
                  f"{t['Prob']:>5.3f} {t['Thresh']:>5.3f} ${t['Entry']:>8.2f} ${t['Exit']:>8.2f} "
                  f"{t['PnL']:>+6.2%} {t['Outcome']:<4}")
        ew_today = today_trades['PnL'].mean()
        print(f"\n  Today EW PnL: {ew_today:+.2%} ({len(today_trades)} trades)")
    else:
        print(f"\n  No trades on {today_str}")


    # ==========================================================================
    # CHARTS
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'ST LONG NEWEST_FIRST 645 — Live Backtest\n'
                 f'{BACKTEST_START} -> {BACKTEST_END} | '
                 f'{n_trades} trades, {n_tickers} tickers | '
                 f'EW {cum_ew:+.2%} | Sharpe {sharpe:.1f}',
                 fontsize=13, fontweight='bold')

    # [0,0] EW Cumulative equity curve
    ax = axes[0, 0]
    dates_dt = pd.to_datetime([d['Date'] for d in daily_ew_list])
    cum_vals = [d['Cum'] * 100 for d in daily_ew_list]
    ax.plot(dates_dt, cum_vals, 'b-o', linewidth=2, markersize=5, label='EW Cumulative')
    ax.fill_between(dates_dt, 0, cum_vals, alpha=0.15, color='blue')
    ax.axhline(0, color='black', linewidth=0.8)
    for i, d in enumerate(daily_ew_list):
        ax.annotate(f"{d['EW']*100:+.1f}%", (dates_dt[i], cum_vals[i]),
                    textcoords="offset points", xytext=(0, 10), fontsize=7, ha='center')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylabel('Cumulative EW PnL (%)')
    ax.set_title('EW Portfolio Equity Curve')
    ax.legend(); ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # [0,1] Daily EW bar chart
    ax = axes[0, 1]
    ew_vals = [d['EW'] * 100 for d in daily_ew_list]
    colors = ['#2E7D32' if v > 0 else '#C62828' for v in ew_vals]
    ax.bar(dates_dt, ew_vals, color=colors, alpha=0.8, width=0.8)
    for i, d in enumerate(daily_ew_list):
        ax.text(dates_dt[i], ew_vals[i] + (0.05 if ew_vals[i] >= 0 else -0.15),
                f"{d['N']}t", ha='center', fontsize=7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylabel('Daily EW PnL (%)')
    ax.set_title('Daily PnL')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # [1,0] Per-ticker avg PnL bar chart
    ax = axes[1, 0]
    ticker_stats = df_trades.groupby('Ticker')['PnL'].agg(['mean', 'count'])
    ticker_stats = ticker_stats.sort_values('mean', ascending=True)
    y_pos = range(len(ticker_stats))
    colors_tk = ['#2E7D32' if v > 0 else '#C62828' for v in ticker_stats['mean']]
    ax.barh(list(y_pos), ticker_stats['mean'].values * 100, color=colors_tk, alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{t} ({int(c)}t)" for t, c in
                        zip(ticker_stats.index, ticker_stats['count'])], fontsize=7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Avg PnL/trade (%)')
    ax.set_title('Per-Ticker Performance')
    ax.grid(axis='x', alpha=0.3)

    # [1,1] Outcome distribution
    ax = axes[1, 1]
    outcome_counts = df_trades['Outcome'].value_counts()
    outcome_colors = {'TP': '#2E7D32', 'SL': '#C62828', 'EOD': '#FF8F00'}
    labels = []; sizes = []; oc_colors = []
    for oc in ['TP', 'SL', 'EOD']:
        if oc in outcome_counts.index:
            avg_p = df_trades[df_trades['Outcome'] == oc]['PnL'].mean()
            labels.append(f"{oc}\n{outcome_counts[oc]} ({avg_p:+.1%})")
            sizes.append(outcome_counts[oc])
            oc_colors.append(outcome_colors.get(oc, '#888'))
    ax.pie(sizes, labels=labels, colors=oc_colors, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 9})
    ax.set_title('Trade Outcomes')

    plt.tight_layout()
    chart_path = '/tmp/backtest_st_long_newest_first_645_live.png'
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
print(f"  BACKTEST COMPLETE")
print(f"{'='*W}")
