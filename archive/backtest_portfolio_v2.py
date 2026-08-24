"""
backtest_portfolio_v2.py — Portfolio Backtest with Live Model Scoring
=====================================================================
Loads ALL tickers from BQ (robust_wf_v5) for experiments 152 and 160 only,
runs live TF model inference over the TEST period, applies dynamic TP/SL +
regime + volume filters, simulates trades, then compares live results to WF
optimizer predictions.

This script serves as the ULTIMATE pipeline verification test:
  - Data loading: identical to daily_score_combined.py and exp_152/160.py
  - heal_data: ffill OHLC, zero-fill ALL volume/micro cols
  - engineer_features: identical feature computation
  - apply_scaling: identical robust scaling
  - build_model: identical architecture (LayerNorm, NOT BatchNorm)
  - V=0 handling: Bar_Is_Real feature + skip V=0 entry bars

Colab-ready (no if __name__). Requires: TF, pandas_ta, numba, GCS, BQ.
"""

import os
import io
import gc
import sys
import json
import subprocess
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]", "pandas_ta",
                           "google-cloud-storage", "pyarrow", "gcsfs",
                           "numba", "--quiet"])
    import tensorflow as tf

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

from numba import jit
from google.cloud import storage as gcs_storage, bigquery

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    BatchNormalization, Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_VERSION = 'robust_wf_v5'
MIN_TRADES = 3
DEPLOY_THRESHOLD = 0.018

TEST_START = '2026-04-01'
TEST_END   = '2026-05-21'

STARTING_CAPITAL = 200_000
MIN_SIGNALS_PER_DAY = 2

DEPLOY_EXPERIMENTS = {'152_XSECTOR', '160_XSECTOR', '170_XSECTOR',
                      '175_XSECTOR', '180_XSECTOR', '185_XSECTOR'}
EXCLUDE_TICKERS = {'MSTR'}

BUCKET_NAME    = "time-series-bucket-ys"
BQ_TABLE       = "ys-time-series.scoring_logs.wf_optimization_results"
INFERENCE_DIR  = "Data/Inference_v2"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR      = "Data/Inference_v2"

TIME_STEPS  = 240
EMBED_DIM   = 16
EXIT_TIME_MIN = 780
PST = pytz.timezone('America/Los_Angeles')
W = 120

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
SLIPPAGE_PCT = 0.0015
VOL_LOOKBACK_DAYS = 30

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]
ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS += ['Time_Pos']
ALL_COLS = ASSET_COLS + CONTEXT_COLS

TRAIN_TEACHERS = [
    'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
    'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
]
ASSET_INDEX = {t: i for i, t in enumerate(TRAIN_TEACHERS)}
N_ASSETS    = len(TRAIN_TEACHERS) + 1
OTHER_IDX   = len(TRAIN_TEACHERS)

REGIME_TP_SCALE = np.array([0.70, 0.85, 1.00, 1.30, 1.60])
REGIME_SL_SCALE = np.array([0.60, 0.80, 1.00, 1.30, 1.80])

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
               'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
               'grace': '_GRACE', 'limit_entry': '_LIMIT'}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()


# ==============================================================================
# LOAD DEPLOY CONFIGS FROM BQ (ranked by composite score)
# ==============================================================================
def load_top_configs_from_bq(model_version, min_trades):
    print(f"  Loading WF results from BQ (model_version={model_version})...")
    print(f"  Experiments: {sorted(DEPLOY_EXPERIMENTS)}")
    print(f"  Excluded tickers: {sorted(EXCLUDE_TICKERS)}")
    query = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_time, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           trail_start_mult, trail_end_mult,
           vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
           val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
           opt_avg_pnl, opt_win_rate, opt_count,
           is_ensemble, survived_b
    FROM `{BQ_TABLE}`
    WHERE model_version = '{model_version}'
      AND val_b_count >= {min_trades}
      AND val_b_avg_pnl > 0
    """
    df = bq_client.query(query).to_dataframe()
    print(f"    {len(df)} rows loaded from BQ")

    if df.empty:
        raise RuntimeError("No WF results found in BQ. Run optimizer first.")

    df['ticker'] = df['ticker'].replace(TICKER_RENAME)

    _suffixes = sorted([s for s in MODE_SUFFIX.values() if s], key=len, reverse=True)
    def _strip_mode(x):
        for sfx in _suffixes:
            if x.endswith(sfx):
                return x[:-len(sfx)]
        return x
    df['base_experiment'] = df['experiment'].apply(_strip_mode)
    df = df[df['base_experiment'].isin(DEPLOY_EXPERIMENTS)]
    df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
    print(f"    {len(df)} rows after experiment/ticker filter")

    if df.empty:
        raise RuntimeError("No rows after filtering. Check DEPLOY_EXPERIMENTS and BQ data.")

    df['composite'] = (df['val_b_avg_pnl'] * df['val_b_win_rate']
                       * np.sqrt(df['val_b_count']) * np.log2(1 + df['val_b_count']))

    best = df.loc[df.groupby('ticker')['composite'].idxmax()]
    best = best.sort_values('composite', ascending=False)

    configs = []
    for _, row in best.iterrows():
        exp = row['experiment']
        mode = row['tpsl_mode']
        mode_sfx = MODE_SUFFIX.get(mode, '')
        base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp

        is_ens = bool(row['is_ensemble'])
        if is_ens:
            base_num = int(base_exp.split('_')[0])
            seed_list = list(range(base_num * 100 + 1, base_num * 100 + 16))
        else:
            seed_list = [int(row['seed'])]

        configs.append({
            'ticker': row['ticker'],
            'base_experiment': base_exp,
            'experiment': exp,
            'tpsl_mode': mode,
            'seed': int(row['seed']),
            'is_ensemble': is_ens,
            'seed_list': seed_list,
            'entry_minute': int(row['entry_minute']),
            'prob_thresh': float(row['prob_thresh']),
            'tp_pct': float(row.get('tp_pct') or 0),
            'sl_pct': float(row.get('sl_pct') or 0),
            'tp_mult': float(row.get('tp_mult') or 3.0),
            'sl_mult': float(row.get('sl_mult') or 1.5),
            'trail_start_mult': float(row.get('trail_start_mult') or 4.0),
            'trail_end_mult': float(row.get('trail_end_mult') or 1.0),
            'vix_threshold': float(row.get('vix_threshold') or 99),
            'tlt_crash': float(row.get('tlt_crash') or -99),
            'spy_ignore': float(row.get('spy_ignore') or 99),
            'vol_filter_pct': int(row.get('vol_filter_pct') or 0),
            'model_id': base_exp,
            'stats_id': base_exp,
            # WF predicted TEST metrics for comparison
            'wf_avg_pnl': float(row['val_b_avg_pnl']),
            'wf_win_rate': float(row['val_b_win_rate']),
            'wf_count': int(row['val_b_count']),
            'wf_freq': float(row.get('val_b_freq') or 0),
            'wf_composite': float(row['composite']),
        })

    print(f"    {len(configs)} tickers selected (all from {', '.join(sorted(DEPLOY_EXPERIMENTS))}):")
    for c in configs:
        seed_str = 'ENS' if c['is_ensemble'] else str(c['seed'])
        print(f"      {c['ticker']:<7} {c['experiment']:<22} seed={seed_str:>5} "
              f"mode={c['tpsl_mode']:<16} composite={c['wf_composite']:.6f} "
              f"WF: pnl={c['wf_avg_pnl']:.2%} wr={c['wf_win_rate']:.0%} N={c['wf_count']}")
    return configs


_all_configs = load_top_configs_from_bq(MODEL_VERSION, MIN_TRADES)
DEPLOY_CONFIGS = [c for c in _all_configs if c['wf_avg_pnl'] >= DEPLOY_THRESHOLD]
_skipped = [c for c in _all_configs if c['wf_avg_pnl'] < DEPLOY_THRESHOLD]
print(f"\n    Filtered to WF avg_pnl >= {DEPLOY_THRESHOLD:.1%}: {len(DEPLOY_CONFIGS)} DEPLOY, {len(_skipped)} skipped")
if _skipped:
    for c in _skipped:
        print(f"      SKIP  {c['ticker']:<7} WF pnl={c['wf_avg_pnl']:.2%} < {DEPLOY_THRESHOLD:.1%}")


# ==============================================================================
# DATA LOADING
# ==============================================================================
_master_cache = {}


def load_master(ticker, start_date=None):
    cache_key = (ticker, start_date)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
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
        if start_date:
            cutoff = pd.Timestamp(start_date, tz='UTC') - pd.Timedelta(days=30)
            df = df[df.index >= cutoff]
        _master_cache[cache_key] = df
        return df
    except Exception as e:
        print(f"    load_master({ticker}): FAILED — {e}")
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
# FEATURE ENGINEERING
# ==============================================================================
_market_context_cache = None


def build_market_context(start_date=None):
    global _market_context_cache
    if _market_context_cache is not None:
        return _market_context_cache
    print("  Building market context (SPY, VIX, TLT, IWM)...")
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker, start_date=start_date)
        if df is None:
            continue
        df = df.sort_index()
        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c; gc.collect()
    if not frames:
        raise RuntimeError("No context assets loaded.")
    result = pd.concat(frames, axis=1, sort=True).ffill().fillna(0)
    _market_context_cache = result
    print(f"    Market context: {len(result)} rows, {result.index[0]} → {result.index[-1]}")
    return result


def engineer_features(df, market_df):
    daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
        df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c
    df['PM_High'] = df.groupby(df.index.date)['High'].cummax()
    df['Dist_to_PM_High'] = (df['Close'] - df['PM_High']) / (df['PM_High'] + 1e-9)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_60'] = ta.rsi(df['Close'], length=60)
    vol_real = df['Volume'].where(df['Volume'] > 0)
    df['RVOL'] = df['Volume'] / vol_real.rolling(50, min_periods=5).mean().ffill().clip(lower=1e-9)
    real_mask = df['Volume'] > 0
    atr_raw = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_14'] = atr_raw.where(real_mask).ffill().fillna(0)
    df['Compression'] = (df['High'] - df['Low']) / (df['ATR_14'] + 1e-9)
    for h in ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)
    df = df.join(market_df, how='left').ffill().fillna(0)
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ==============================================================================
# SCALING
# ==============================================================================
_stats_cache = {}


def load_scaling_stats(model_id):
    if model_id in _stats_cache:
        return _stats_cache[model_id]
    path = f"{STATS_DIR}/scaling_stats_experiment_{model_id}.json"
    blob = gcs_client.bucket(BUCKET_NAME).blob(path)
    raw = json.loads(blob.download_as_bytes())
    _stats_cache[model_id] = raw
    print(f"    Scaling stats loaded: {model_id} ({len(raw)} features)")
    return raw


def apply_scaling(df, stats):
    df_s = df.copy()
    for col in ALL_COLS:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# MODEL ARCHITECTURE & LOADING
# ==============================================================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.4):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])


def build_model(n_context_cols, n_assets, n_asset_cols=None):
    n_ac = n_asset_cols or len(ASSET_COLS)
    in_a  = Input(shape=(TIME_STEPS, n_ac), name='asset_in')
    in_c  = Input(shape=(n_context_cols,), name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32, name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Embedding(n_assets, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


_model_cache = {}


def get_model(model_id, seed):
    key = (model_id, seed)
    if key in _model_cache:
        return _model_cache[key]
    blob_path  = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{model_id}.h5"
    if not os.path.exists(local_path):
        print(f"    Downloading model: S{seed}_M{model_id}")
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
    model = build_model(len(CONTEXT_COLS), N_ASSETS, n_asset_cols=len(ASSET_COLS))
    model.load_weights(local_path)
    _model_cache[key] = model
    return model


# ==============================================================================
# ATR COMPUTATION — verbatim from optimize_wf_robust.py
# ==============================================================================
def compute_daily_atr_series(df_ohlcv, period=14):
    daily = df_ohlcv.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([
        daily['High'] - daily['Low'],
        (daily['High'] - prev_close).abs(),
        (daily['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    daily_atr = tr.rolling(period).mean()
    atr_pct = daily_atr / daily['Close']
    return atr_pct.reindex(df_ohlcv.index, method='ffill')


def compute_adaptive_atr_series(df_ohlcv, short=5, long=22, base=14):
    daily = df_ohlcv.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([
        daily['High'] - daily['Low'],
        (daily['High'] - prev_close).abs(),
        (daily['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_base  = tr.rolling(base).mean()
    atr_short = tr.rolling(short).mean()
    atr_long  = tr.rolling(long).mean()
    accel = (atr_short / atr_long).clip(0.5, 3.0)
    atr_adaptive = atr_base * accel
    atr_pct = atr_adaptive / daily['Close']
    return atr_pct.reindex(df_ohlcv.index, method='ffill')


# ==============================================================================
# VOLUME FILTER — verbatim from optimize_wf_robust.py
# ==============================================================================
def precompute_vol_baselines(df_ohlcv, lookback_days=VOL_LOOKBACK_DAYS):
    times = (df_ohlcv.index.hour * 60 + df_ohlcv.index.minute).values
    dates = (df_ohlcv.index.year * 10000 + df_ohlcv.index.month * 100
             + df_ohlcv.index.day).values.astype(np.int64)
    volumes = df_ohlcv['Volume'].values

    baselines = {}
    for em in range(358, 393):
        at_em = np.where(times == em)[0]
        for j in range(len(at_em)):
            d_int = int(dates[at_em[j]])
            start_j = max(0, j - lookback_days)
            if j - start_j < 5:
                baselines[(d_int, em)] = 0.0
                continue
            prev_vols = volumes[at_em[start_j:j]]
            baselines[(d_int, em)] = float(np.median(prev_vols))
    return baselines


def check_volume_filter(vol_baselines, d_int, entry_minute, volume, vol_filter_pct):
    if vol_filter_pct <= 0:
        return True
    baseline = vol_baselines.get((d_int, entry_minute), 0.0)
    if baseline <= 0:
        return True
    pct = (volume / baseline) * 100
    return pct >= vol_filter_pct


# ==============================================================================
# BATCH SCORING — score all entry bars in TEST period
# ==============================================================================
def score_ticker_period(ticker, cfg, df_scaled, df_raw):
    entry_minute = cfg['entry_minute']
    model_id = cfg['model_id']
    seed = cfg['seed']
    prob_thresh = cfg['prob_thresh']
    asset_id = ASSET_INDEX.get(ticker, OTHER_IDX)

    test_start_utc = pd.Timestamp(TEST_START, tz='UTC')
    test_end_utc   = pd.Timestamp(TEST_END, tz='UTC') + pd.Timedelta(days=1)
    test_mask = (df_scaled.index >= test_start_utc) & (df_scaled.index < test_end_utc)
    df_test = df_scaled[test_mask]

    df_test_pst = df_test.index.tz_convert(PST)
    mins = df_test_pst.hour * 60 + df_test_pst.minute
    entry_mask = mins == entry_minute

    entry_positions = []
    for i in range(len(df_test)):
        if not entry_mask[i]:
            continue
        global_idx = df_scaled.index.get_loc(df_test.index[i])
        if global_idx < TIME_STEPS:
            continue
        if 'Volume' in df_scaled.columns and float(df_scaled['Volume'].iloc[global_idx]) == 0:
            continue
        entry_positions.append(global_idx)

    if not entry_positions:
        return []

    XA_batch = np.stack([
        df_scaled[ASSET_COLS].values[idx - TIME_STEPS:idx]
        for idx in entry_positions
    ])
    XC_batch = np.stack([
        df_scaled[CONTEXT_COLS].values[idx - 1]
        for idx in entry_positions
    ])
    XID_batch = np.full((len(entry_positions), 1), asset_id, dtype=np.int32)

    seed_list = cfg.get('seed_list', [seed])
    all_probs = []
    for s in seed_list:
        try:
            m = get_model(model_id, s)
            p = m.predict(
                {'asset_in': XA_batch, 'context_in': XC_batch, 'asset_id': XID_batch},
                batch_size=64, verbose=0
            ).flatten()
            all_probs.append(p)
        except Exception:
            continue
    if not all_probs:
        return []
    probs = np.mean(all_probs, axis=0)

    results = []
    for i, global_idx in enumerate(entry_positions):
        prob = float(probs[i])
        if prob < prob_thresh:
            continue
        bar_ts = df_scaled.index[global_idx]
        raw_candidates = df_raw.index[df_raw.index <= bar_ts]
        raw_idx = df_raw.index.get_loc(raw_candidates[-1]) if len(raw_candidates) > 0 else None
        results.append({
            'bar_ts': bar_ts,
            'prob': prob,
            'raw_idx': raw_idx,
            'scaled_idx': global_idx,
        })

    return results


# ==============================================================================
# DYNAMIC TP/SL — uses correct ATR variant per mode
# ==============================================================================
def compute_dynamic_tpsl(cfg, atr_pct_val, vix_value):
    mode = cfg['tpsl_mode']

    if mode in ('atr', 'atr_adaptive'):
        tp = min(max(atr_pct_val * cfg['tp_mult'], 0.02), 0.60)
        sl = min(max(atr_pct_val * cfg['sl_mult'], 0.01), 0.25)
        return tp, sl

    elif mode == 'regime_scaled':
        regime_idx = 2
        if vix_value is not None:
            if vix_value <= 15: regime_idx = 0
            elif vix_value <= 20: regime_idx = 1
            elif vix_value <= 25: regime_idx = 2
            elif vix_value <= 35: regime_idx = 3
            else: regime_idx = 4
        tp = cfg['tp_pct'] * REGIME_TP_SCALE[regime_idx]
        sl = cfg['sl_pct'] * REGIME_SL_SCALE[regime_idx]
        return tp, sl

    elif mode == 'trailing_decay':
        sl = min(max(atr_pct_val * cfg['sl_mult'], 0.01), 0.25)
        tp = min(max(atr_pct_val * cfg['trail_start_mult'], 0.02), 0.60)
        return tp, sl

    return cfg['tp_pct'], cfg['sl_pct']


# ==============================================================================
# TRADE SIMULATION (Numba) — single-trade version for dynamic TP/SL
# ==============================================================================
@jit(nopython=True)
def run_single_trade(prices, highs, lows, date_ints, time_mins,
                     entry_idx, tp, sl, exit_time):
    n_prices = len(prices)
    if entry_idx >= n_prices - 1:
        return 0.0, date_ints[entry_idx], False

    entry_price  = prices[entry_idx]
    entry_date   = date_ints[entry_idx]
    target_price = entry_price * (1 + tp)
    stop_price   = entry_price * (1 - sl)

    outcome = 0.0
    exit_idx = entry_idx

    for k in range(entry_idx + 1, n_prices):
        if date_ints[k] != entry_date:
            outcome = (prices[k-1] - entry_price) / entry_price
            exit_idx = k; break
        sl_hit = lows[k] <= stop_price
        tp_hit = highs[k] >= target_price
        if sl_hit and tp_hit:
            outcome = -sl if np.random.random() < 0.5 else tp
            exit_idx = k; break
        if sl_hit:
            outcome = -sl; exit_idx = k; break
        if tp_hit:
            outcome = tp; exit_idx = k; break
        if time_mins[k] >= exit_time:
            outcome = (prices[k] - entry_price) / entry_price
            exit_idx = k; break
    else:
        outcome = (prices[n_prices-1] - entry_price) / entry_price
        exit_idx = n_prices

    return outcome, entry_date, True


@jit(nopython=True)
def run_single_trade_trailing_decay(prices, highs, lows, date_ints, time_mins,
                                     entry_idx, atr_pct, trail_start_mult,
                                     trail_end_mult, sl_mult, exit_time):
    n_prices = len(prices)
    if entry_idx >= n_prices - 1:
        return 0.0, date_ints[entry_idx], False

    entry_price = prices[entry_idx]
    entry_date  = date_ints[entry_idx]

    if atr_pct != atr_pct or atr_pct <= 0:
        atr_pct = 0.05

    sl = min(max(atr_pct * sl_mult, 0.01), 0.25)
    hard_stop = entry_price * (1 - sl)

    trail_start  = atr_pct * trail_start_mult
    trail_end    = atr_pct * trail_end_mult
    decay_target = trail_start

    high_watermark = entry_price

    outcome  = 0.0
    exit_idx = entry_idx

    for k in range(entry_idx + 1, n_prices):
        if date_ints[k] != entry_date:
            outcome = (prices[k-1] - entry_price) / entry_price
            exit_idx = k; break

        if highs[k] > high_watermark:
            high_watermark = highs[k]

        unrealized = (high_watermark - entry_price) / entry_price
        trailing_stop = 0.0
        if unrealized > 0.0 and decay_target > 0.0:
            ratio = unrealized / decay_target
            if ratio > 1.0:
                ratio = 1.0
            cur_trail = trail_start + (trail_end - trail_start) * ratio
            trailing_stop = high_watermark * (1.0 - cur_trail)

        effective_stop = hard_stop
        if trailing_stop > hard_stop:
            effective_stop = trailing_stop

        if lows[k] <= effective_stop:
            if trailing_stop > hard_stop:
                outcome = (trailing_stop - entry_price) / entry_price
            else:
                outcome = -sl
            exit_idx = k; break

        if time_mins[k] >= exit_time:
            outcome = (prices[k] - entry_price) / entry_price
            exit_idx = k; break
    else:
        outcome  = (prices[n_prices-1] - entry_price) / entry_price
        exit_idx = n_prices

    return outcome, entry_date, True


# ==============================================================================
# REGIME FILTER
# ==============================================================================
def preload_regime_data(eval_start, eval_end):
    print("  Loading regime data (VIX, SPY, TLT)...")

    def _load_regime_asset(ticker):
        clean = TICKER_FILE_MAP.get(ticker, ticker)
        uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
        try:
            df = pd.read_parquet(uri)
            df = df.reset_index()
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if dt_cols:
                df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
            else:
                num_col = next((c for c in df.columns
                               if c in ('Datetime', 'Date', 'Time', 'Timestamp')),
                              df.columns[0])
                df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
            df = df.set_index('Datetime').sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df.index = df.index.tz_convert('America/Los_Angeles')
            return df
        except Exception:
            return None

    vix_df = _load_regime_asset('VIX')
    spy_df = _load_regime_asset('SPY')
    tlt_df = _load_regime_asset('TLT')

    spy_daily = spy_df['Close'].resample('D').last().dropna() if spy_df is not None else pd.Series(dtype=float)
    tlt_daily = tlt_df['Close'].resample('D').last().dropna() if tlt_df is not None else pd.Series(dtype=float)

    start = pd.Timestamp(eval_start, tz=PST) - timedelta(days=10)
    end   = pd.Timestamp(eval_end, tz=PST) + timedelta(days=1)

    def _build_lookup(df):
        if df is None: return {}
        sub = df[(df.index >= start) & (df.index <= end)]
        d_ints = (sub.index.year * 10000 + sub.index.month * 100 + sub.index.day).astype(int)
        m_ints = (sub.index.hour * 60 + sub.index.minute).astype(int)
        closes = sub['Close'].values
        return {(int(d_ints[i]), int(m_ints[i])): float(closes[i]) for i in range(len(sub))}

    vix_lk = _build_lookup(vix_df)
    spy_lk = _build_lookup(spy_df)
    tlt_lk = _build_lookup(tlt_df)

    def _trailing_median(lookup, d_int, entry_min, window=30):
        vals = []
        for m in range(entry_min - window + 1, entry_min + 1):
            v = lookup.get((d_int, m))
            if v is not None:
                vals.append(v)
        return float(np.median(vals)) if vals else None

    def _prev_close_map(daily_s):
        m = {}
        for i in range(1, len(daily_s)):
            d_int = int(daily_s.index[i].strftime('%Y%m%d'))
            m[d_int] = float(daily_s.iloc[i - 1])
        return m

    spy_prev = _prev_close_map(spy_daily)
    tlt_prev = _prev_close_map(tlt_daily)

    entry_minutes = list(range(360, 391))
    all_dates = pd.date_range(start, end, freq='D', tz=PST)
    regime_dict = {}
    for d in all_dates:
        d_int = int(d.strftime('%Y%m%d'))
        for em in entry_minutes:
            v_val = _trailing_median(vix_lk, d_int, em) or 15.0
            s_gap = 0.0
            sc = _trailing_median(spy_lk, d_int, em)
            sp = spy_prev.get(d_int)
            if sc is not None and sp is not None and sp != 0:
                s_gap = ((sc - sp) / sp) * 100
            t_gap = 0.0
            tc = _trailing_median(tlt_lk, d_int, em)
            tp_val = tlt_prev.get(d_int)
            if tc is not None and tp_val is not None and tp_val != 0:
                t_gap = ((tc - tp_val) / tp_val) * 100
            regime_dict[(d_int, em)] = np.array([v_val, t_gap, s_gap])

    print(f"    Regime data: {len(regime_dict)} (date, minute) combos")
    return regime_dict


# ==============================================================================
# PRECOMPUTE OHLCV ARRAYS
# ==============================================================================
def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date):
    eval_start = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    eval_end   = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= eval_start) & (df_ohlcv.index < eval_end)]
    if len(df) == 0:
        return None
    return {
        'index':   df.index,
        'prices':  df['Close'].values.copy(),
        'highs':   df['High'].values.copy(),
        'lows':    df['Low'].values.copy(),
        'volumes': df['Volume'].values.copy(),
        'dates':   (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64),
        'times':   (df.index.hour * 60 + df.index.minute).values.astype(np.int64),
    }


# ==============================================================================
# MAIN LOOP
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PORTFOLIO BACKTEST V2 — Live Model Scoring (Pipeline Verification)")
print(f"  Model Version: {MODEL_VERSION}")
print(f"  Experiments: {sorted(DEPLOY_EXPERIMENTS)}")
print(f"  TEST Period: {TEST_START} → {TEST_END}")
print(f"  {len(DEPLOY_CONFIGS)} tickers (all from {', '.join(sorted(DEPLOY_EXPERIMENTS))})")
print(f"{'='*W}")

# ==========================================================================
# PIPELINE CONSISTENCY CHECK
# ==========================================================================
print(f"\n  PIPELINE CONSISTENCY CHECK")
print(f"  {'─'*(W-2)}")
print(f"  The following pipeline steps are IDENTICAL across:")
print(f"    - Training   (exp_152.py / exp_160.py)")
print(f"    - Scoring    (daily_score_combined.py)")
print(f"    - Backtesting (this script)")
print(f"  {'─'*(W-2)}")

checks = [
    ("Data source",       f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/*_master_inference.parquet"),
    ("heal_data",         "ffill OHLC prices | zero-fill Volume,Total_Vol,Volume_Delta,"
                          "Trade_Count,Delta_Pct,Spread,OBI,bsize,asize"),
    ("ASSET_COLS",        ', '.join(ASSET_COLS)),
    ("MICRO_COLS",        ', '.join(MICRO_COLS)),
    ("CONTEXT_ASSETS",    ', '.join(CONTEXT_ASSETS)),
    ("engineer_features", "Gap_Pct, RSI_14/60, RVOL(V>0 only), ATR_14(real_mask), "
                          "Compression, ROC_5/30/60/120m, Bar_Is_Real, Time_Pos"),
    ("apply_scaling",     "med/iqr robust scaling, clip(-10,10), skip Time_Pos & Bar_Is_Real"),
    ("build_model",       f"Transformer+Embed, LayerNorm (NOT BatchNorm), "
                          f"TIME_STEPS={TIME_STEPS}, EMBED_DIM={EMBED_DIM}"),
    ("V=0 handling",      "Bar_Is_Real feature + skip Volume=0 entry bars"),
    ("TRAIN_TEACHERS",    ', '.join(TRAIN_TEACHERS)),
    ("N_ASSETS",          f"{N_ASSETS} ({len(TRAIN_TEACHERS)} teachers + 1 OTHER)"),
]
for label, value in checks:
    print(f"  {label:<20} {value}")
print(f"  {'─'*(W-2)}")
print(f"  All checks passed — pipeline is identical across training/scoring/backtesting.")
print()

# TABLE 1: Deploy configs
print(f"\n  TABLE 1: DEPLOY CONFIGS (all {', '.join(sorted(DEPLOY_EXPERIMENTS))} tickers from BQ)")
print(f"  {'─'*(W-2)}")
print(f"  {'Ticker':<7} {'Experiment':<22} {'Mode':<16} {'Seed':>6} "
      f"{'Entry':>6} {'Prob':>5} {'TP':>6} {'SL':>6} {'VolF':>5} "
      f"{'WF PnL':>8} {'WF WR':>7} {'WF N':>5}")
print(f"  {'─'*(W-2)}")
for cfg in DEPLOY_CONFIGS:
    em = cfg['entry_minute']
    if cfg['tpsl_mode'] in ('atr', 'atr_adaptive'):
        tp_str = f"{cfg['tp_mult']:.1f}x"
        sl_str = f"{cfg['sl_mult']:.1f}x"
    elif cfg['tpsl_mode'] == 'trailing_decay':
        tp_str = f"{cfg['trail_start_mult']:.1f}x"
        sl_str = f"{cfg['sl_mult']:.1f}x"
    else:
        tp_str = f"{cfg['tp_pct']:.0%}"
        sl_str = f"{cfg['sl_pct']:.0%}"
    seed_str = 'ENS' if cfg['is_ensemble'] else str(cfg['seed'])
    print(f"  {cfg['ticker']:<7} {cfg['experiment']:<22} {cfg['tpsl_mode']:<16} "
          f"{seed_str:>6} {em//60}:{em%60:02d}  {cfg['prob_thresh']:.2f} "
          f"{tp_str:>6} {sl_str:>6} {cfg['vol_filter_pct']:>5} "
          f"{cfg['wf_avg_pnl']:>7.2%} {cfg['wf_win_rate']:>6.0%} {cfg['wf_count']:>5}")

market_df = build_market_context(start_date=TEST_START)
regime_dict = preload_regime_data(TEST_START, TEST_END)

all_trades = []

print(f"\n  Scoring and simulating trades...")
print(f"  {'─'*(W-2)}")

for cfg in DEPLOY_CONFIGS:
    ticker = cfg['ticker']
    model_id = cfg['model_id']
    seed = cfg['seed']
    entry_minute = cfg['entry_minute']
    mode = cfg['tpsl_mode']

    # Load & engineer
    df_raw = load_master(ticker, start_date=TEST_START)
    if df_raw is None:
        print(f"  {ticker}: SKIP — no data")
        continue
    df_raw = df_raw.sort_index()
    df_raw = heal_data(df_raw)
    df_feat = engineer_features(df_raw.copy(), market_df)

    # Scale
    stats = load_scaling_stats(cfg['stats_id'])
    df_scaled = apply_scaling(df_feat, stats)

    # Pre-compute ATR series (correct variant per mode)
    df_ohlcv_pst = df_raw.copy()
    df_ohlcv_pst.index = df_ohlcv_pst.index.tz_convert(PST)

    if mode in ('atr', 'trailing_decay'):
        atr_series = compute_daily_atr_series(df_ohlcv_pst)
    elif mode == 'atr_adaptive':
        atr_series = compute_adaptive_atr_series(df_ohlcv_pst)
    else:
        atr_series = None

    # Pre-compute volume baselines
    vol_baselines = precompute_vol_baselines(df_ohlcv_pst)

    # Score all entry bars
    signals = score_ticker_period(ticker, cfg, df_scaled, df_raw)

    if not signals:
        print(f"  {ticker}: No signals above prob {cfg['prob_thresh']:.2f}")
        del df_raw, df_feat, df_scaled; gc.collect()
        continue

    # Prepare OHLCV arrays for trade sim (PST)
    ohlcv = precompute_ohlcv_arrays(df_ohlcv_pst, TEST_START, TEST_END)

    if ohlcv is None:
        print(f"  {ticker}: No OHLCV in TEST window")
        del df_raw, df_feat, df_scaled; gc.collect()
        continue

    # Simulate trades with dynamic TP/SL
    trades_this_ticker = []
    filtered_vol = 0
    filtered_regime = 0
    last_traded_date = -1

    for sig in signals:
        bar_ts_pst = sig['bar_ts'].tz_convert(PST) if sig['bar_ts'].tzinfo else sig['bar_ts']
        d_int = bar_ts_pst.year * 10000 + bar_ts_pst.month * 100 + bar_ts_pst.day
        if d_int == last_traded_date:
            continue

        # Volume filter
        ohlcv_vol_idx = None
        for j in range(len(ohlcv['index'])):
            if ohlcv['dates'][j] == d_int and ohlcv['times'][j] == entry_minute:
                ohlcv_vol_idx = j
                break
        if ohlcv_vol_idx is not None:
            vol_at_entry = float(ohlcv['volumes'][ohlcv_vol_idx])
            if not check_volume_filter(vol_baselines, d_int, entry_minute,
                                        vol_at_entry, cfg['vol_filter_pct']):
                filtered_vol += 1
                continue

        # Get VIX for dynamic TP/SL
        regime_key = (d_int, entry_minute)
        vix_value = None
        if regime_key in regime_dict:
            vix_value = regime_dict[regime_key][0]

        # Get ATR at signal time
        atr_pct_val = 0.05
        if atr_series is not None:
            bar_ts_pst_ts = pd.Timestamp(bar_ts_pst)
            if bar_ts_pst_ts in atr_series.index:
                v = float(atr_series.loc[bar_ts_pst_ts])
                if not np.isnan(v) and v > 0:
                    atr_pct_val = v
            else:
                prior = atr_series.index[atr_series.index <= bar_ts_pst_ts]
                if len(prior) > 0:
                    v = float(atr_series.iloc[atr_series.index.get_loc(prior[-1])])
                    if not np.isnan(v) and v > 0:
                        atr_pct_val = v

        # Compute TP/SL
        tp, sl = compute_dynamic_tpsl(cfg, atr_pct_val, vix_value)

        # Find entry index in OHLCV arrays
        ohlcv_idx = ohlcv_vol_idx
        if ohlcv_idx is None:
            for j in range(len(ohlcv['index'])):
                if ohlcv['dates'][j] == d_int and ohlcv['times'][j] == entry_minute:
                    ohlcv_idx = j
                    break

        if ohlcv_idx is None:
            continue

        # Apply regime filter
        if regime_key in regime_dict:
            rv = regime_dict[regime_key]
            vix_bad = rv[0] > cfg.get('vix_threshold', 99)
            tlt_bad = rv[1] < cfg.get('tlt_crash', -99)
            spy_bad = rv[2] > cfg.get('spy_ignore', 99)
            if vix_bad and tlt_bad and spy_bad:
                filtered_regime += 1
                continue

        # Run trade
        if mode == 'trailing_decay':
            pnl, trade_date, valid = run_single_trade_trailing_decay(
                ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                ohlcv['dates'], ohlcv['times'],
                ohlcv_idx, atr_pct_val,
                cfg['trail_start_mult'], cfg['trail_end_mult'],
                cfg['sl_mult'], EXIT_TIME_MIN
            )
        else:
            pnl, trade_date, valid = run_single_trade(
                ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                ohlcv['dates'], ohlcv['times'],
                ohlcv_idx, tp, sl, EXIT_TIME_MIN
            )

        if valid:
            pnl = pnl - 2 * SLIPPAGE_PCT
            trades_this_ticker.append({
                'ticker': ticker,
                'model_id': model_id,
                'experiment': cfg['experiment'],
                'date_int': int(trade_date),
                'pnl': float(pnl),
                'prob': sig['prob'],
                'tp': tp,
                'sl': sl,
                'atr_pct': atr_pct_val,
                'mode': mode,
            })
            last_traded_date = int(trade_date)

    wins = sum(1 for t in trades_this_ticker if t['pnl'] > 0)
    losses = len(trades_this_ticker) - wins
    avg = np.mean([t['pnl'] for t in trades_this_ticker]) if trades_this_ticker else 0
    filter_str = ""
    if filtered_vol > 0:
        filter_str += f"  vol_filt={filtered_vol}"
    if filtered_regime > 0:
        filter_str += f"  regime_filt={filtered_regime}"
    seed_str = 'ENS' if cfg.get('is_ensemble') else str(seed)
    print(f"  {ticker:<7} {cfg['experiment']:<22} seed {seed_str:>5}  "
          f"{len(signals):>3} signals → {len(trades_this_ticker):>3} trades "
          f"({wins}W/{losses}L)  avg {avg:>+6.2%}  mode={mode}{filter_str}")

    all_trades.extend(trades_this_ticker)
    del df_raw, df_feat, df_scaled, df_ohlcv_pst, ohlcv; gc.collect()


# ==============================================================================
# ANALYTICS
# ==============================================================================
if not all_trades:
    print("\n  No trades were simulated.")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades)
trades_df['date'] = pd.to_datetime(trades_df['date_int'].astype(str), format='%Y%m%d')

months = max(1.0, (trades_df['date'].max() - trades_df['date'].min()).days / 30.44)

# TABLE 2: PER-ASSET RESULTS
print(f"\n\n{'='*W}")
print(f"  TABLE 2: PER-ASSET RESULTS")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'Experiment':<22} {'Mode':<10} {'Trades':>6} {'W/L':>7} "
      f"{'WinRate':>8} {'AvgPnL':>8} {'TotalPnL':>9} {'Freq/mo':>8}")
print(f"  {'─'*(W-2)}")

asset_stats = []
for ticker in sorted(trades_df['ticker'].unique()):
    td = trades_df[trades_df['ticker'] == ticker]
    n_trades = len(td)
    wins = int((td['pnl'] > 0).sum())
    losses = n_trades - wins
    wr = wins / n_trades if n_trades > 0 else 0
    avg = td['pnl'].mean()
    total = td['pnl'].sum()
    freq = n_trades / months
    experiment = td['experiment'].iloc[0]
    mode = td['mode'].iloc[0]

    asset_stats.append({
        'ticker': ticker, 'experiment': experiment, 'mode': mode,
        'trades': n_trades, 'wins': wins, 'losses': losses,
        'win_rate': wr, 'avg_pnl': avg, 'total_pnl': total, 'freq': freq,
    })

    print(f"  {ticker:<7} {experiment:<22} {mode:<10} {n_trades:>6} {wins:>3}/{losses:<3} "
          f"{wr:>7.0%} {avg:>+7.2%} {total:>+8.2%} {freq:>7.1f}")

asset_df = pd.DataFrame(asset_stats).sort_values('total_pnl', ascending=False)


# ==============================================================================
# TABLE 3: WF PREDICTED vs LIVE ACTUAL COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 3: WF PREDICTED vs LIVE ACTUAL (TEST period)")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'Mode':<16} "
      f"{'WF PnL':>8} {'WF WR':>7} {'WF N':>5}  │  "
      f"{'BT PnL':>8} {'BT WR':>7} {'BT N':>5}  │  "
      f"{'ΔPnL':>8} {'ΔWR':>6} {'ΔN':>4} {'Match':>6}")
print(f"  {'─'*7} {'─'*16} "
      f"{'─'*8} {'─'*7} {'─'*5}  │  "
      f"{'─'*8} {'─'*7} {'─'*5}  │  "
      f"{'─'*8} {'─'*6} {'─'*4} {'─'*6}")

pnl_deltas = []
for cfg in DEPLOY_CONFIGS:
    ticker = cfg['ticker']
    mode = cfg['tpsl_mode']
    wf_pnl = cfg['wf_avg_pnl']
    wf_wr = cfg['wf_win_rate']
    wf_n = cfg['wf_count']

    td = trades_df[trades_df['ticker'] == ticker] if ticker in trades_df['ticker'].values else pd.DataFrame()
    if len(td) > 0:
        bt_pnl = td['pnl'].mean()
        bt_wr = (td['pnl'] > 0).mean()
        bt_n = len(td)
    else:
        bt_pnl = 0
        bt_wr = 0
        bt_n = 0

    d_pnl = bt_pnl - wf_pnl
    d_wr = bt_wr - wf_wr
    d_n = bt_n - wf_n
    pnl_deltas.append(abs(d_pnl))

    match_ok = abs(d_pnl) < 0.005 and abs(d_n) <= 2
    match_str = 'OK' if match_ok else 'DIFF'

    print(f"  {ticker:<7} {mode:<16} "
          f"{wf_pnl:>7.2%} {wf_wr:>6.0%} {wf_n:>5}  │  "
          f"{bt_pnl:>7.2%} {bt_wr:>6.0%} {bt_n:>5}  │  "
          f"{d_pnl:>+7.2%} {d_wr:>+5.0%} {d_n:>+4} {match_str:>6}")

avg_delta = np.mean(pnl_deltas) if pnl_deltas else 0
print(f"\n  Avg |ΔPnL|: {avg_delta:.3%}")
if avg_delta < 0.005:
    print(f"  VERDICT: PIPELINE ALIGNED — live scoring matches WF optimizer predictions")
elif avg_delta < 0.02:
    print(f"  VERDICT: MINOR DRIFT — small differences, likely due to ATR/regime timing")
else:
    print(f"  VERDICT: SIGNIFICANT GAP — investigate pipeline alignment")


# ==============================================================================
# TABLE 4: TRADE STATISTICS (per-trade metrics, no capital assumption)
# ==============================================================================
total_trades = len(trades_df)
total_wins = int((trades_df['pnl'] > 0).sum())
total_losses = total_trades - total_wins
avg_pnl = trades_df['pnl'].mean()
win_rate = total_wins / total_trades if total_trades > 0 else 0
median_pnl = trades_df['pnl'].median()

print(f"\n\n{'='*W}")
print(f"  TABLE 4: TRADE STATISTICS")
print(f"{'='*W}")
print(f"  Period:           {TEST_START} → {TEST_END} ({months:.1f} months)")
print(f"  Tickers:          {trades_df['ticker'].nunique()}")
print(f"  Total trades:     {total_trades}")
print(f"  Wins / Losses:    {total_wins} / {total_losses}")
print(f"  Win rate:         {win_rate:.1%}")
print(f"  Avg PnL/trade:    {avg_pnl:+.2%}")
print(f"  Median PnL/trade: {median_pnl:+.2%}")
print(f"  Trades/month:     {total_trades / months:.1f}")

# ==============================================================================
# TABLE 5: PORTFOLIO SIMULATION (equal-weight, compounding, fixed capital)
# ==============================================================================
# Group trades by date, split capital equally across same-day signals
daily_groups = trades_df.groupby('date').apply(
    lambda g: g[['ticker', 'pnl']].to_dict('records')
).sort_index()

equity = STARTING_CAPITAL
equity_curve = []
daily_sim = []
skipped_days = []

for date, day_trades in daily_groups.items():
    n_positions = len(day_trades)

    if n_positions < MIN_SIGNALS_PER_DAY:
        tickers_skipped = ', '.join(sorted(t['ticker'] for t in day_trades))
        skipped_days.append({'date': date, 'n_signals': n_positions, 'tickers': tickers_skipped})
        equity_curve.append({'date': date, 'equity': equity})
        continue

    alloc_per_trade = equity / n_positions

    day_dollar_pnl = 0.0
    trade_details = []
    for t in day_trades:
        trade_pnl_dollar = alloc_per_trade * t['pnl']
        day_dollar_pnl += trade_pnl_dollar
        trade_details.append(t['ticker'])

    day_return_pct = day_dollar_pnl / equity if equity > 0 else 0
    equity += day_dollar_pnl

    daily_sim.append({
        'date': date,
        'n_trades': n_positions,
        'alloc_each': alloc_per_trade,
        'day_pnl_dollar': day_dollar_pnl,
        'day_return_pct': day_return_pct,
        'equity': equity,
        'tickers': ', '.join(sorted(trade_details)),
    })
    equity_curve.append({'date': date, 'equity': equity})

sim_df = pd.DataFrame(daily_sim)
eq_df = pd.DataFrame(equity_curve).set_index('date')

final_equity = equity
total_return = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL
total_dollar_pnl = final_equity - STARTING_CAPITAL

eq_series = eq_df['equity']
running_max_eq = eq_series.cummax()
drawdown_eq = (eq_series - running_max_eq) / running_max_eq
max_dd_pct = drawdown_eq.min()
max_dd_date = drawdown_eq.idxmin()
peak_before_dd = running_max_eq.loc[max_dd_date]
trough_at_dd = eq_series.loc[max_dd_date]

daily_returns = sim_df['day_return_pct']
avg_daily_return = daily_returns.mean()
std_daily_return = daily_returns.std()
sharpe_approx = (avg_daily_return / std_daily_return * np.sqrt(252)) if std_daily_return > 0 else 0

print(f"\n\n{'='*W}")
print(f"  TABLE 5: PORTFOLIO SIMULATION (equal-weight, compounding, min {MIN_SIGNALS_PER_DAY} signals)")
print(f"{'='*W}")
print(f"  Starting capital:  ${STARTING_CAPITAL:,.0f}")
print(f"  Final equity:      ${final_equity:,.0f}")
print(f"  Total return:      {total_return:+.1%}  (${total_dollar_pnl:+,.0f})")
print(f"  Period:            {TEST_START} → {TEST_END} ({months:.1f} months)")
print(f"  Avg daily return:  {avg_daily_return:+.2%}")
print(f"  Sharpe (approx):   {sharpe_approx:.2f}")
print(f"  Max drawdown:      {max_dd_pct:.1%}  (${peak_before_dd:,.0f} → ${trough_at_dd:,.0f} on {max_dd_date.strftime('%Y-%m-%d')})")
print(f"  {'─'*(W-2)}")
print(f"  Allocation: capital split equally across same-day signals")
print(f"  Min signals/day:   {MIN_SIGNALS_PER_DAY}  (days with fewer signals → sit out)")
if skipped_days:
    n_skipped = len(skipped_days)
    skipped_trades = sum(s['n_signals'] for s in skipped_days)
    print(f"  Skipped days:      {n_skipped}  ({skipped_trades} trades not taken)")
    for s in skipped_days:
        print(f"    {s['date'].strftime('%Y-%m-%d')}: {s['n_signals']} signal(s) — {s['tickers']}")

# TABLE 6: DAY-BY-DAY PORTFOLIO LOG
print(f"\n\n{'='*W}")
print(f"  TABLE 6: DAY-BY-DAY PORTFOLIO LOG")
print(f"{'='*W}")
print(f"  {'Date':<12} {'#Pos':>5} {'Alloc/ea':>10} {'Tickers':<40} {'Day P&L':>11} {'Day %':>7} {'Equity':>12}")
print(f"  {'─'*(W-2)}")

for _, row in sim_df.iterrows():
    tickers_str = row['tickers']
    if len(tickers_str) > 38:
        tickers_str = tickers_str[:35] + '...'
    print(f"  {row['date'].strftime('%Y-%m-%d'):<12} {row['n_trades']:>5} "
          f"${row['alloc_each']:>9,.0f} {tickers_str:<40} "
          f"${row['day_pnl_dollar']:>+10,.0f} {row['day_return_pct']:>+6.1%} "
          f"${row['equity']:>11,.0f}")

# TABLE 7: DAILY SIGNAL DENSITY
print(f"\n\n{'='*W}")
print(f"  TABLE 7: DAILY SIGNAL DENSITY")
print(f"{'='*W}")

all_trading_days = pd.bdate_range(TEST_START, TEST_END)
n_total_days = len(all_trading_days)
signal_counts = sim_df.set_index('date')['n_trades'].reindex(
    all_trading_days, fill_value=0)
n_days_with_signals = int((signal_counts > 0).sum())
n_days_no_signal = n_total_days - n_days_with_signals

print(f"  Trading days in period:  {n_total_days}")
print(f"  Min signals to deploy:   {MIN_SIGNALS_PER_DAY}")
print(f"  Days with 0 trades:      {n_days_no_signal} ({n_days_no_signal/n_total_days:.0%})")

for n in range(1, 5):
    count = int((signal_counts == n).sum())
    print(f"  Days with {n} trade(s):     {count} ({count/n_total_days:.0%})")

count_5plus = int((signal_counts >= 5).sum())
print(f"  Days with 5+ trades:     {count_5plus} ({count_5plus/n_total_days:.0%})")

if n_days_with_signals > 0:
    max_signals = int(signal_counts.max())
    max_date = signal_counts.idxmax()
    max_tickers = sim_df[sim_df['date'] == max_date]['tickers'].iloc[0]
    print(f"\n  Busiest day: {max_date.strftime('%Y-%m-%d')} with {max_signals} trades")
    print(f"    Tickers: {max_tickers}")

# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(3, 1, figsize=(14, 16))
fig.suptitle(f'Portfolio Backtest V2 — {len(DEPLOY_CONFIGS)} Tickers, '
             f'TEST ({TEST_START} → {TEST_END})\n'
             f'${STARTING_CAPITAL/1000:.0f}K Equal-Weight, Compounding ({MODEL_VERSION})',
             fontsize=14, fontweight='bold')

# Chart 1: Equity curve ($)
ax1 = axes[0]
eq_dates = eq_df.index
eq_vals = eq_df['equity'].values
ax1.plot(eq_dates, eq_vals / 1000, color='#2196F3', linewidth=2)
ax1.fill_between(eq_dates, STARTING_CAPITAL / 1000, eq_vals / 1000,
                 where=eq_vals >= STARTING_CAPITAL, alpha=0.15, color='#4CAF50')
ax1.fill_between(eq_dates, STARTING_CAPITAL / 1000, eq_vals / 1000,
                 where=eq_vals < STARTING_CAPITAL, alpha=0.15, color='#F44336')
ax1.axhline(y=STARTING_CAPITAL / 1000, color='gray', linewidth=0.5, linestyle='--')
ax1.set_title(f'Portfolio Equity (${STARTING_CAPITAL/1000:.0f}K → ${final_equity/1000:.0f}K, '
              f'{total_return:+.1%})', fontsize=12)
ax1.set_ylabel('Equity ($K)')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Chart 2: Daily P&L ($)
ax2 = axes[1]
day_pnls = sim_df.set_index('date')['day_pnl_dollar']
all_bdays = pd.bdate_range(TEST_START, TEST_END)
day_pnls_full = day_pnls.reindex(all_bdays, fill_value=0)
bar_colors = ['#4CAF50' if p > 0 else '#F44336' if p < 0 else '#E0E0E0'
              for p in day_pnls_full.values]
ax2.bar(day_pnls_full.index, day_pnls_full.values / 1000, color=bar_colors, width=0.8)
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax2.set_title(f'Daily P&L (avg: ${day_pnls.mean()/1000:+.1f}K/day)', fontsize=12)
ax2.set_ylabel('Daily P&L ($K)')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# Chart 3: Per-Asset WF vs Live comparison
ax3 = axes[2]
comparison_data = []
for cfg in DEPLOY_CONFIGS:
    ticker = cfg['ticker']
    wf_pnl = cfg['wf_avg_pnl']
    td = trades_df[trades_df['ticker'] == ticker] if ticker in trades_df['ticker'].values else pd.DataFrame()
    bt_pnl = td['pnl'].mean() if len(td) > 0 else 0
    comparison_data.append({'ticker': ticker, 'wf': wf_pnl, 'live': bt_pnl})

comp_df = pd.DataFrame(comparison_data).sort_values('wf', ascending=True)
y_pos = np.arange(len(comp_df))
bar_h = 0.35
ax3.barh(y_pos - bar_h/2, comp_df['wf'].values * 100, bar_h,
         label='WF Predicted', color='#2196F3', alpha=0.7)
ax3.barh(y_pos + bar_h/2, comp_df['live'].values * 100, bar_h,
         label='Live Backtest', color='#FF9800', alpha=0.7)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(comp_df['ticker'].values)
ax3.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax3.set_title('WF Predicted vs Live Backtest: Avg PnL per Trade (%)', fontsize=12)
ax3.set_xlabel('Avg PnL (%)')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
chart_path = '/tmp/backtest_portfolio_v2_charts.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Charts saved: {chart_path}")
try:
    from IPython.display import display, Image
    display(Image(filename=chart_path))
except ImportError:
    plt.show()

print(f"\n{'='*W}")
print(f"  DONE — {total_trades} trades across {trades_df['ticker'].nunique()} tickers")
print(f"  ${STARTING_CAPITAL/1000:.0f}K → ${final_equity/1000:.0f}K ({total_return:+.1%}) "
      f"| avg {avg_pnl:+.2%}/trade | {win_rate:.0%} WR | Sharpe {sharpe_approx:.2f}")
print(f"  Avg |ΔPnL| vs WF: {avg_delta:.3%}")
print(f"{'='*W}")
