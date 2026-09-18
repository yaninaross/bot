# %% ── ST SHORT HYBRID — Trade Prioritization Comparison ──────────────────────
#
# When multiple SHORT signals fire on the same day and capital is limited to
# MAX_TRADES concurrent positions, which selection strategy is best?
#
# Approach:
#   1. Score all tickers across all days (same engine as live backtest)
#   2. Simulate ALL signals to get trade outcomes (SHORT direction)
#   3. Group trades by date; on days with > MAX_TRADES signals, apply each
#      prioritization strategy to select the top-K trades
#   4. Compare portfolio metrics across strategies
#
# Deploy strategy: Hybrid F5+F6+F7
#   - 2 families: L5-1 + L5-2, each with folds 5/6/7
#   - Per-ticker: best val_b fold wins; F7 gated at val_b >= 3.0%
#   - L5-2 is deployed (upgraded 2026-08-16)
#
# Strategies tested:
#   ALL_TRADES   — no cap (upper bound baseline)
#   RANDOM       — uniform random, averaged over 50 iterations
#   PROB_RAW     — highest raw model probability
#   PROB_MARGIN  — highest (prob - threshold) / threshold
#   EXPECTED_VAL — prob × tp - (1-prob) × sl
#   VAL_B_BEST   — highest historical val_b_avg_pnl from WF optimizer
#   COMPOSITE    — 0.4×prob_margin_norm + 0.3×EV_norm + 0.3×val_b_norm
#   INVERSE_PROB — lowest prob above threshold (contrarian)
#
# Multi-model strategies (Phase 5):
#   M_CONSENSUS    — rank by # models firing, ties by max prob
#   M_ENSEMBLE_AVG — rank by average prob across all models
#   M_MAX_PROB     — rank by highest single-model prob
#   M_NEWEST_FOLD  — prefer newest fold (F7>F6>F5), L5-2 preferred
#   M_AGREEMENT    — only tickers where models agree (spread < 0.15)
#
# Phase 7: K sweep (1→6) for top 5 strategies
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
BACKTEST_END     = '2026-08-18'

TAIL_DAYS        = 45

SCORE_MINUTE     = 405
ENTRY_MINUTE     = 415
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
        'bq_versions': {
            5: 'spike_transformer_short_L5_1_fold5',
            6: 'spike_transformer_short_L5_1_fold6',
            7: 'spike_transformer_short_L5_1_fold7',
        },
    },
    'L5-2': {
        'gcs_data':   'Algos/RA_MoE/short_L5_2_data',
        'gcs_models': 'Algos/RA_MoE/short_L5_2_models',
        'bq_versions': {
            5: 'spike_transformer_short_L5_2_fold5',
            6: 'spike_transformer_short_L5_2_fold6',
            7: 'spike_transformer_short_L5_2_fold7',
        },
    },
}

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
            if fold_id == 7 and v < F7_VAL_B_THRESH:
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
    print(f"    Total: {len(deploy)} tickers (Hybrid F5+F6+F7)")
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
    local_path = f"/tmp/_bt_short_prio_645_{file_ticker}_master.parquet"
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

def load_models_and_meta(deploy_configs, all_model_configs=None):
    global _model, _fold_weights, _current_key, _family_meta
    all_cfgs = list(deploy_configs.values())
    if all_model_configs:
        for cfg_list in all_model_configs.values():
            all_cfgs.extend(cfg_list)
    needed_families = sorted(set(cfg['family'] for cfg in all_cfgs))
    for family_name in needed_families:
        fam = FAMILIES[family_name]
        local_meta = f'/tmp/ra_moe_short_645_{family_name}_meta.npz'
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
        fam = FAMILIES[family_name]
        local_dir = f'/tmp/ra_moe_short_645_{family_name}_models/fold_{fold_id}'
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
# SCORING + TRADE SIM — enriched with signal metadata for prioritization
# ==============================================================================
def score_and_backtest_ticker(ticker, cfg, df_feat, regime_df):
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

    pnl, dates, ep, xp, outcomes = _run_sim(
        np.array(entry_indices, dtype=np.int64), ohlcv, cfg)

    trades = []
    for i in range(len(pnl)):
        d = int(dates[i])
        date_str = f"{d//10000}-{(d%10000)//100:02d}-{d%100:02d}"
        meta = signal_meta.get(d, {})
        prob = signal_probs.get(d, 0.0)
        thresh = cfg['prob_thresh']
        ev = prob * cfg['tp_pct'] - (1 - prob) * cfg['sl_pct']
        trades.append({
            'Ticker': ticker,
            'Date': date_str,
            'PnL': float(pnl[i]),
            'Entry': float(ep[i]),
            'Exit': float(xp[i]),
            'Outcome': 'TP' if outcomes[i]==1 else 'SL' if outcomes[i]==-1 else 'EOD',
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
            daily_ew.append({
                'Date': date,
                'EW': day['PnL'].mean(),
                'N': len(day),
                'Tickers': sorted(day['Ticker'].unique().tolist()),
            })
        iter_daily.append(daily_ew)

    if is_random and iter_daily:
        all_dates_set = sorted(set(d['Date'] for run in iter_daily for d in run))
        avg_daily = []
        for date in all_dates_set:
            ew_vals = []
            n_vals = []
            for run in iter_daily:
                for d in run:
                    if d['Date'] == date:
                        ew_vals.append(d['EW'])
                        n_vals.append(d['N'])
            avg_daily.append({
                'Date': date,
                'EW': np.mean(ew_vals) if ew_vals else 0,
                'N': np.mean(n_vals) if n_vals else 0,
                'Tickers': [],
            })
        return avg_daily
    else:
        return iter_daily[0] if iter_daily else []


def compute_stats(daily_ew_list):
    if not daily_ew_list:
        return {}
    ew_vals = [d['EW'] for d in daily_ew_list]
    n_vals = [d['N'] for d in daily_ew_list]
    cum = np.cumsum(ew_vals)
    avg = np.mean(ew_vals)
    std = np.std(ew_vals) if len(ew_vals) > 1 else 0
    sharpe = (avg / std * np.sqrt(252)) if std > 0 else 0
    total_trades = sum(n_vals)
    win_days = sum(1 for v in ew_vals if v > 0)
    max_dd = 0
    peak = 0
    for c in cum:
        peak = max(peak, c)
        dd = peak - c
        max_dd = max(max_dd, dd)
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
    }


# ==============================================================================
# MAIN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  ST SHORT 645 — TRADE PRIORITIZATION COMPARISON")
print(f"  {BACKTEST_START} -> {BACKTEST_END} | MAX_TRADES = {MAX_TRADES}")
print(f"  Score: {SCORE_MINUTE//60}:{SCORE_MINUTE%60:02d} PT | Entry: {ENTRY_MINUTE//60}:{ENTRY_MINUTE%60:02d} PT | "
      f"Exit: {EXIT_TIME_MIN//60}:{EXIT_TIME_MIN%60:02d} PT")
print(f"  Slippage: {2*SLIPPAGE_PCT:.2%} RT | Deploy threshold: >{BASE_PNL_THRESH:.1%}")
print(f"{'='*W}")

print(f"\n  Loading deploy configs from BQ (Hybrid F5+F6+F7)...")
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
    print(f"  Total signals:     {len(df_all)}")
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
          f"{'Trades':>7} {'Tr/Day':>6} {'WinDay%':>8} {'MaxDD':>7} "
          f"{'Best':>7} {'Worst':>7}")
    print(f"  {'─'*W}")

    ranked = sorted(strategy_results.items(), key=lambda x: x[1].get('cum_ew', -999), reverse=True)
    for name, s in ranked:
        if not s:
            continue
        print(f"  {name:<14} {s['cum_ew']:>+7.2%} {s['avg_daily']:>+7.2%} "
              f"{s['sharpe']:>7.1f} {s['total_trades']:>7.0f} "
              f"{s['avg_trades_day']:>5.1f} {s['win_day_pct']:>7.0%} "
              f"{s['max_drawdown']:>6.2%} {s['best_day']:>+6.2%} {s['worst_day']:>+6.2%}")

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
              f"{'Val_B':>6} {'RR':>5} {'Mode':<8} {'PnL':>7} {'Out':<4}")
        for _, t in day.sort_values('Prob', ascending=False).iterrows():
            print(f"    {t['Ticker']:<7} {t['Prob']:>5.3f} {t['Prob_Margin']:>+6.1%} "
                  f"{t['EV']:>+6.3f} {t['Val_B']:>5.1%} {t['RR_Ratio']:>4.1f} "
                  f"{t['Mode']:<8} {t['PnL']:>+6.2%} {t['Outcome']:<4}")

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
    fig.suptitle(f'ST SHORT 645 — Trade Prioritization Comparison\n'
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
    chart_path = '/tmp/backtest_st_short_prioritization_645.png'
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
    for ticker in DEPLOY_TICKERS:
        available = ALL_MODEL_CONFIGS.get(ticker, [])
        if not available:
            continue
        df_feat = engineer_asset(ticker, market_df)
        if df_feat is None:
            continue
        n_models_scored = 0
        for cfg in available:
            trades = score_and_backtest_ticker(ticker, cfg, df_feat, regime_df)
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
                fold_order = {'F7': 0, 'F6': 1, 'F5': 2}
                family_order = {'L5-2': 0, 'L5-1': 1}
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

        MULTI_STRATEGIES = {
            'M_CONSENSUS':    select_multi_consensus,
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
                daily_ew.append({
                    'Date': date,
                    'EW': day['PnL'].mean(),
                    'N': len(day),
                    'Tickers': sorted(day['Ticker'].unique().tolist()),
                })

            stats = compute_stats(daily_ew)
            multi_strategy_results[name] = stats
            multi_strategy_daily[name] = daily_ew
            print(f"  {name}: EW={stats['cum_ew']:+.2%}, "
                  f"Sharpe={stats['sharpe']:.1f}, trades={stats['total_trades']:.0f}")


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
              f"{'Sharpe':>7} {'Trades':>7} {'Tr/Day':>6} {'WinDay%':>8} "
              f"{'MaxDD':>7}")
        print(f"  {'─'*W}")

        for name, s in combined_ranked:
            if not s:
                continue
            stype = 'MULTI' if name.startswith('M_') else 'SINGLE'
            print(f"  {name:<16} {stype:<7} {s['cum_ew']:>+7.2%} "
                  f"{s['avg_daily']:>+7.2%} {s['sharpe']:>7.1f} "
                  f"{s['total_trades']:>7.0f} {s['avg_trades_day']:>5.1f} "
                  f"{s['win_day_pct']:>7.0%} {s['max_drawdown']:>6.2%}")

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
        SWEEP_RANGE = range(1, 7)
        TOP5_STRATS = {
            'M_CONSENSUS':    ('multi',  select_multi_consensus),
            'M_MAX_PROB':     ('multi',  select_multi_max_prob),
            'M_NEWEST_FOLD':  ('multi',  select_multi_newest_fold),
            'M_ENSEMBLE_AVG': ('multi',  select_multi_ensemble_avg),
            'EXPECTED_VAL':   ('single', select_expected_value),
        }

        print(f"\n\n{'='*W}")
        print(f"  PHASE 7: MAX_TRADES SWEEP (1 → 6) — TOP 5 STRATEGIES")
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
                    daily_ew.append({
                        'Date': date,
                        'EW': day_sel['PnL'].mean(),
                        'N': len(day_sel),
                    })
                sweep_results[sname][max_k] = compute_stats(daily_ew)

        print(f"  {'Strategy':<16}", end="")
        for k in SWEEP_RANGE:
            print(f"  {'K='+str(k):>10}", end="")
        print()
        print(f"  {'─'*W}")

        metrics_to_sweep = [
            ('Cum EW', 'cum_ew', lambda v: f"{v:>+9.2%}"),
            ('Sharpe', 'sharpe', lambda v: f"{v:>10.1f}"),
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
        fig3.suptitle(f'ST SHORT 645 — MAX_TRADES Sweep (1→6) — Top 5 Strategies\n'
                      f'{BACKTEST_START} → {BACKTEST_END}',
                      fontsize=13, fontweight='bold')

        sweep_colors = {
            'M_CONSENSUS': '#D81B60',
            'M_MAX_PROB': '#1E88E5',
            'M_NEWEST_FOLD': '#00ACC1',
            'M_ENSEMBLE_AVG': '#8E24AA',
            'EXPECTED_VAL': '#E65100',
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
        chart3_path = '/tmp/backtest_st_short_prioritization_645_sweep.png'
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
        fig2.suptitle(f'ST SHORT 645 — Combined Prioritization: Single-Model + Multi-Model\n'
                      f'{BACKTEST_START} -> {BACKTEST_END} | MAX_TRADES = {MAX_TRADES}',
                      fontsize=13, fontweight='bold')

        multi_colors = {
            'M_CONSENSUS': '#D81B60',
            'M_ENSEMBLE_AVG': '#8E24AA',
            'M_MAX_PROB': '#1E88E5',
            'M_NEWEST_FOLD': '#00ACC1',
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
        chart2_path = '/tmp/backtest_st_short_prioritization_645_combined.png'
        plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
        print(f"\n  Combined chart saved: {chart2_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()

    else:
        print(f"\n  No multi-model trades — skipping multi-model comparison.")


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
print(f"  SHORT 645 PRIORITIZATION COMPARISON COMPLETE")
print(f"{'='*W}")
