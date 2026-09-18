# %% ── F403 + F425 + F435 SHORT + S3 LONG — Comparative Backtest ──────────────
#
# Seven strategies compared head-to-head:
#   1. F403_ONLY   — SHORT 9:30 (entry min 375), 2.5% dump
#   2. F425_ONLY   — SHORT 10AM (entry min 405), 2.5% dump
#   3. F435_ONLY   — SHORT 10AM (entry min 405), 8% dump
#   4. ADDITIVE    — F403 + F425 combined (deployed ADDITIVE_403)
#   5. ALL_SHORT   — F403 + F425 + F435 all three legs
#   6. S3_LONG     — S3 family agreement gate, CSV-scored, limit_entry LONG
#   7. HEDGED      — S3_LONG + ADDITIVE combined portfolio
#
# SHORT legs: live model inference from GCS-stored weights (transformer encoder
#   + embedding + MLP). Same architecture/features/scaling as training.
# S3 LONG: CSV-scored from GCS Results_all_v2 / Results_scored_20260703.
#   Numba trade simulation with limit_entry mode.
#
# GPU RECOMMENDED. Colab-ready.

import os, io, gc, sys, json, time, ctypes, warnings
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, Embedding, Flatten,
                                     Concatenate, MultiHeadAttention, Add,
                                     LayerNormalization, GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

try:
    import pandas_ta as ta
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta",
                           "--no-deps", "--quiet"])
    import pandas_ta as ta

try:
    from numba import jit
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numba", "--quiet"])
    from numba import jit

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
    auth.authenticate_user()
except (ImportError, Exception):
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from google.cloud import storage as gcs_storage, bigquery


# ==============================================================================
# 1. CONFIG
# ==============================================================================
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
GCS_MODEL_DIR = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
INFERENCE_DIR = "Data/Inference_v2"
RESULTS_DIR      = 'Algos/Results_scored_20260703'
RESULTS_FALLBACK = 'Algos/Results_all_v2'

TEST_START  = '2026-04-01'
TEST_END    = '2026-08-08'
HALF2_START = '2026-06-01'

DEPLOY_FLOOR  = 0.010
SLIPPAGE_PCT  = 0.002
SLIPPAGE_RT   = 2 * SLIPPAGE_PCT
EXIT_TIME_MIN = 780
MIN_H2_TRADES = 3

TIME_STEPS = 240
EMBED_DIM  = 16
SEEDS_PER_FOLD = 15

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_ALLOWED_MODES = ('static', 'atr_sl', 'grace')

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

# ── 9:30 feature sets (F403 — no Morning_Ext, Gap_VIX, VIX_Level) ──
ASSET_COLS_930 = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
                   'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
                  [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_COLS_930 = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS_930.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS_930.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS_930.append('Time_Pos')

ALL_COLS_930 = ASSET_COLS_930 + CONTEXT_COLS_930

# ── 10AM feature sets (shared by F425 and F435) ──
ASSET_COLS_10AM = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
                    'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real',
                    'Morning_Ext', 'Gap_VIX'] +
                   [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_COLS_10AM = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS_10AM.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS_10AM.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS_10AM += ['Time_Pos', 'VIX_Level']

ALL_COLS_10AM = ASSET_COLS_10AM + CONTEXT_COLS_10AM

# ── Leg configurations ──
LEG_F403 = {
    'name': 'F403_ONLY',
    'label': 'F403 (9:30, 2.5% dump)',
    'exp_id': '403_XSECTOR',
    'seed_base': 40300,
    'model_version': 'robust_wf_v12_short_403',
    'entry_min': 375,
    'asset_cols': ASSET_COLS_930,
    'context_cols': CONTEXT_COLS_930,
    'all_cols': ALL_COLS_930,
    'scaling_file': f'{INFERENCE_DIR}/scaling_stats_experiment_403_XSECTOR.json',
    'feature_variant': '930',
    'color': '#1565C0',
}

LEG_F425 = {
    'name': 'F425_ONLY',
    'label': 'F425 (10AM, 2.5% dump)',
    'exp_id': '425_10AM',
    'seed_base': 42500,
    'model_version': 'robust_wf_v12_short_425_10am',
    'entry_min': 405,
    'asset_cols': ASSET_COLS_10AM,
    'context_cols': CONTEXT_COLS_10AM,
    'all_cols': ALL_COLS_10AM,
    'scaling_file': f'{INFERENCE_DIR}/scaling_stats_experiment_425_10AM.json',
    'feature_variant': '10am',
    'color': '#2E7D32',
}

LEG_F435 = {
    'name': 'F435_ONLY',
    'label': 'F435 (10AM, 8% dump)',
    'exp_id': '435_10AM',
    'seed_base': 43500,
    'model_version': 'robust_wf_v12_short_435_10am',
    'entry_min': 405,
    'asset_cols': ASSET_COLS_10AM,
    'context_cols': CONTEXT_COLS_10AM,
    'all_cols': ALL_COLS_10AM,
    'scaling_file': f'{INFERENCE_DIR}/scaling_stats_experiment_435_10AM.json',
    'feature_variant': '10am',
    'color': '#E65100',
}

# ── S3 LONG gate constants ──
S3_MODEL_VERSIONS = ('robust_wf_v12', 'robust_wf_v12_stitched')
S3_MIN_SEEDS     = 3
S3_MAX_VA_OPT    = 3.0
S3_DEPLOY_THRESHOLD = 0.018
S3_EXCLUDE       = {'LUNR'}
S3_GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
               'prob_thresh', 'tp_pct', 'sl_pct',
               'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
               'grace_minutes', 'dip_pct', 'max_wait_minutes', 'vol_filter_pct']
S3_ENTRY_MINUTE = 375
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

# ── Tickers to evaluate (SHORT) ──
RULE_1 = ['BE', 'IREN', 'OKLO', 'ONDS', 'RKLB', 'WOLF']

EXPANDED_SHORT = sorted(set(RULE_1 + [
    'ACMR', 'AVAV', 'BEAM', 'BKSY', 'BTBT', 'CCJ', 'CEG', 'CIFR', 'CLSK',
    'COIN', 'COP', 'CORZ', 'FCEL', 'FLNC', 'GE', 'GEV', 'HOOD',
    'HUT', 'JOBY', 'KTOS', 'LUNR', 'MARA', 'MNTS', 'MOD', 'MP',
    'MSTR', 'MU', 'MVIS', 'MXL', 'NNE', 'RIOT', 'RCAT', 'RIVN',
    'SMR', 'SOUN', 'VOYG', 'WULF',
]))

# ── Strategy colors ──
STRATEGY_COLORS = {
    'F403_ONLY': '#1565C0',
    'F425_ONLY': '#2E7D32',
    'F435_ONLY': '#E65100',
    'ADDITIVE':  '#D81B60',
    'ALL_SHORT': '#795548',
    'S3_LONG':   '#FF8F00',
    'HEDGED':    '#7B1FA2',
}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120


# ==============================================================================
# 2. HELPERS
# ==============================================================================
def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


_master_cache = {}


def load_master(ticker):
    if ticker in _master_cache:
        return _master_cache[ticker]
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


# ==============================================================================
# 3. SHORT MODEL — build_model, detect_n_assets, load_seed_models
# ==============================================================================
def detect_n_assets(exp_id, seed_base):
    seed = seed_base + 1
    blob_path = f"{GCS_MODEL_DIR}/UNIVERSAL_S{seed}_M{exp_id}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{exp_id}.h5"
    bucket = gcs_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    if not os.path.exists(local_path):
        blob.download_to_filename(local_path)
    elif os.path.getsize(local_path) != blob.size:
        os.remove(local_path)
        blob.download_to_filename(local_path)
    import h5py
    with h5py.File(local_path, 'r') as f:
        def _find_embedding(group):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset) and 'embedding' in key.lower():
                    return item.shape[0]
                if isinstance(item, h5py.Group):
                    result = _find_embedding(item)
                    if result is not None:
                        return result
            return None
        n = _find_embedding(f)
    if n is None:
        raise RuntimeError(f"Cannot detect N_ASSETS from {blob_path}")
    return n


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.4):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])


def build_model(n_asset_features, n_context_features, n_assets=31):
    in_a  = Input(shape=(TIME_STEPS, n_asset_features), name='asset_in')
    in_c  = Input(shape=(n_context_features,),          name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,           name='asset_id')
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


def load_seed_models(leg_config):
    exp_id = leg_config['exp_id']
    seed_base = leg_config['seed_base']
    n_ac = len(leg_config['asset_cols'])
    n_cc = len(leg_config['context_cols'])
    seeds = list(range(seed_base + 1, seed_base + SEEDS_PER_FOLD + 1))
    n_assets = detect_n_assets(exp_id, seed_base)
    other_idx = n_assets - 1
    leg_config['n_assets'] = n_assets
    leg_config['other_idx'] = other_idx
    print(f"    Detected N_ASSETS={n_assets} (OTHER_IDX={other_idx})")
    bucket = gcs_client.bucket(BUCKET_NAME)
    models = []
    for seed in seeds:
        blob_path = f"{GCS_MODEL_DIR}/UNIVERSAL_S{seed}_M{exp_id}.h5"
        local_path = f"/tmp/UNIVERSAL_S{seed}_M{exp_id}.h5"
        try:
            blob = bucket.blob(blob_path)
            if not os.path.exists(local_path):
                blob.download_to_filename(local_path)
            elif os.path.getsize(local_path) != blob.size:
                os.remove(local_path)
                blob.download_to_filename(local_path)
            tf.keras.backend.clear_session()
            model = build_model(n_ac, n_cc, n_assets=n_assets)
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.load_weights(local_path)
            models.append(model)
        except Exception as e:
            print(f"    Seed {seed}: FAILED -- {e}")
    return models


# ==============================================================================
# 4. FEATURE ENGINEERING
# ==============================================================================
def build_market_context(include_vix_level=False):
    frames = []
    vix_close = None
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker)
        if df is None:
            raise RuntimeError(f"Cannot load context asset {ticker}")
        df = df.sort_index()
        df = heal_data(df)
        if ticker == 'VIX':
            vix_close = df['Close'].copy()
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
    _master_cache.clear()
    result = pd.concat(frames, axis=1).ffill().fillna(0)
    if include_vix_level and vix_close is not None:
        result['VIX_Level'] = vix_close.reindex(result.index).ffill().fillna(0)
    return result


def engineer_features_930(df, market_df):
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


def engineer_features_10am(df, market_df):
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    dates = df_pst.date

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

    open_window = (mins >= 388) & (mins <= 400) & (df['Volume'] > 0)
    temp_dates = pd.Series(dates, index=df.index)
    if open_window.any():
        daily_open_930 = df.loc[open_window].groupby(temp_dates[open_window])['Open'].first()
        mapped_open = temp_dates.map(daily_open_930)
    else:
        mapped_open = pd.Series(np.nan, index=df.index)
    after_open = mins >= 390
    df['Morning_Ext'] = np.where(
        after_open & (mapped_open.fillna(0) > 0),
        (df['Close'] / mapped_open.fillna(1).clip(lower=1e-8)) - 1,
        0.0
    )

    df = df.join(market_df, how='left').ffill().fillna(0)
    vix_level = df['VIX_Level'] if 'VIX_Level' in df.columns else pd.Series(0.0, index=df.index)
    df['Gap_VIX'] = df['Gap_Pct'] * vix_level

    df_pst2 = df.index.tz_convert(PST)
    mins2 = df_pst2.hour * 60 + df_pst2.minute
    df['Time_Pos'] = np.clip((mins2 - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ==============================================================================
# 5. SCALING
# ==============================================================================
def load_scaling_stats(gcs_path):
    blob = gcs_client.bucket(BUCKET_NAME).blob(gcs_path)
    return json.loads(blob.download_as_text())


def apply_scaling(df, stats, all_cols):
    df_s = df.copy()
    for col in all_cols:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# 6. SCORING (SHORT)
# ==============================================================================
def batch_score_ticker(df_scaled, entry_indices, models, asset_cols, context_cols, other_idx):
    XA = df_scaled[asset_cols].values.astype(np.float32)
    XC = df_scaled[context_cols].values.astype(np.float32)
    valid = [i for i in entry_indices if i >= TIME_STEPS]
    if not valid:
        return {}
    valid = np.array(valid)
    XA_batch = np.array([XA[i - TIME_STEPS:i] for i in valid])
    XC_batch = np.array([XC[i - 1] for i in valid])
    XID_batch = np.full((len(valid), 1), other_idx, dtype=np.int32)

    all_seed_probs = []
    for model in models:
        preds = model.predict(
            {'asset_in': XA_batch, 'context_in': XC_batch, 'asset_id': XID_batch},
            batch_size=min(len(valid), 4096), verbose=0
        ).flatten()
        all_seed_probs.append(preds)

    ensemble = np.mean(all_seed_probs, axis=0)
    idx_pst = df_scaled.index.tz_convert(PST)
    result = {}
    for j, i in enumerate(valid):
        d = idx_pst[i].date()
        result[d] = float(ensemble[j])
    return result


# ==============================================================================
# 7. SHORT TRADE SIMULATION
# ==============================================================================
def run_short_trade_static(prices, highs, lows, date_ints, time_mins,
                           entry_idx, tp, sl, exit_time):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    entry_price = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    target_price = entry_price * (1 - tp)
    stop_price   = entry_price * (1 + sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            return (entry_price - prices[k-1]) / entry_price, ed, True
        if highs[k] >= stop_price and lows[k] <= target_price:
            return (tp if np.random.random() < 0.5 else -sl), ed, True
        if highs[k] >= stop_price:
            return -sl, ed, True
        if lows[k] <= target_price:
            return tp, ed, True
        if time_mins[k] >= exit_time:
            return (entry_price - prices[k]) / entry_price, ed, True
    return (entry_price - prices[n-1]) / entry_price, ed, True


def run_short_trade_atr_sl(prices, highs, lows, date_ints, time_mins,
                           entry_idx, tp_pct, atr_pct_val, sl_mult, exit_time):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    entry_price = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    atr = atr_pct_val if atr_pct_val > 0 else 0.05
    sl = min(max(atr * sl_mult, 0.01), 0.25)
    target_price = entry_price * (1 - tp_pct)
    stop_price   = entry_price * (1 + sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            return (entry_price - prices[k-1]) / entry_price, ed, True
        if highs[k] >= stop_price and lows[k] <= target_price:
            return (tp_pct if np.random.random() < 0.5 else -sl), ed, True
        if highs[k] >= stop_price:
            return -sl, ed, True
        if lows[k] <= target_price:
            return tp_pct, ed, True
        if time_mins[k] >= exit_time:
            return (entry_price - prices[k]) / entry_price, ed, True
    return (entry_price - prices[n-1]) / entry_price, ed, True


def run_short_trade_grace(prices, highs, lows, date_ints, time_mins,
                          entry_idx, tp, sl, exit_time, grace_minutes):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    entry_price = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    et = time_mins[entry_idx]
    target_price = entry_price * (1 - tp)
    stop_price   = entry_price * (1 + sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            return (entry_price - prices[k-1]) / entry_price, ed, True
        in_grace = (time_mins[k] - et) <= grace_minutes
        tp_hit = lows[k] <= target_price
        sl_hit = highs[k] >= stop_price
        if not in_grace:
            if sl_hit and tp_hit:
                return (tp if np.random.random() < 0.5 else -sl), ed, True
            if sl_hit:
                return -sl, ed, True
        if tp_hit:
            return tp, ed, True
        if time_mins[k] >= exit_time:
            return (entry_price - prices[k]) / entry_price, ed, True
    return (entry_price - prices[n-1]) / entry_price, ed, True


def precompute_ohlcv_arrays(df, start_date, end_date):
    s = pd.to_datetime(start_date).tz_localize(PST)
    e = pd.to_datetime(end_date).tz_localize(PST) + timedelta(days=1)
    idx = df.index
    if str(idx.tz) != 'America/Los_Angeles':
        idx = idx.tz_convert(PST)
    mask = (idx >= s) & (idx < e)
    df_cut = df[mask]
    if len(df_cut) == 0:
        return None
    idx_cut = df_cut.index if str(df_cut.index.tz) == 'America/Los_Angeles' \
        else df_cut.index.tz_convert(PST)
    return {
        'prices': df_cut['Close'].values.astype(np.float64).copy(),
        'highs':  df_cut['High'].values.astype(np.float64).copy(),
        'lows':   df_cut['Low'].values.astype(np.float64).copy(),
        'dates':  (idx_cut.year * 10000 + idx_cut.month * 100 + idx_cut.day).values.astype(np.int64),
        'times':  (idx_cut.hour * 60 + idx_cut.minute).values.astype(np.int64),
    }


def compute_daily_atr_pct(df, period=14):
    daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([
        daily['High'] - daily['Low'],
        (daily['High'] - prev_close).abs(),
        (daily['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    daily_atr = tr.rolling(period).mean()
    atr_pct = daily_atr / daily['Close']
    return atr_pct.reindex(df.index, method='ffill')


def execute_trade(ohlcv, entry_idx, mode, tp_pct, sl_pct, sl_mult, grace_min):
    prices = ohlcv['prices']
    highs  = ohlcv['highs']
    lows   = ohlcv['lows']
    dates  = ohlcv['dates']
    times  = ohlcv['times']
    if mode == 'atr_sl':
        atr_val = ohlcv['atr_pct'][entry_idx] if 'atr_pct' in ohlcv else 0.05
        return run_short_trade_atr_sl(prices, highs, lows, dates, times,
                                      entry_idx, tp_pct, atr_val, sl_mult, EXIT_TIME_MIN)
    elif mode == 'grace':
        return run_short_trade_grace(prices, highs, lows, dates, times,
                                     entry_idx, tp_pct, sl_pct, EXIT_TIME_MIN,
                                     int(grace_min))
    else:
        return run_short_trade_static(prices, highs, lows, dates, times,
                                      entry_idx, tp_pct, sl_pct, EXIT_TIME_MIN)


# ==============================================================================
# 8. LONG TRADE SIMULATION (Numba)
# ==============================================================================
@jit(nopython=True)
def run_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints,
                              time_mins, tp, sl, exit_time, dip_pct,
                              max_wait_minutes):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    trades      = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)
    trade_count       = 0
    current_entry_ptr = 0
    last_traded_date  = -1

    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break
        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1; continue
        signal_price = prices[idx]
        limit_price  = signal_price * (1.0 - dip_pct)
        signal_time  = time_mins[idx]

        fill_idx   = -1
        fill_price = 0.0
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date: break
            if time_mins[k] - signal_time > max_wait_minutes: break
            if lows[k] <= limit_price:
                fill_price = limit_price; fill_idx = k; break

        if fill_idx < 0:
            current_entry_ptr += 1
            last_traded_date = entry_date; continue

        target_price = fill_price * (1 + tp)
        stop_price   = fill_price * (1 - sl)
        outcome    = 0.0
        trade_done = False
        exit_idx   = fill_idx

        for k in range(fill_idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - fill_price) / fill_price
                exit_idx = k; trade_done = True; break
            sl_hit = lows[k] <= stop_price
            tp_hit = highs[k] >= target_price
            if sl_hit and tp_hit:
                outcome = -sl if np.random.random() < 0.5 else tp
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (prices[k] - fill_price) / fill_price
                exit_idx = k; trade_done = True; break

        if not trade_done:
            outcome = (prices[n_prices-1] - fill_price) / fill_price
            exit_idx = n_prices
        trades[trade_count] = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date
        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return trades[:trade_count], entry_dates[:trade_count]


# ==============================================================================
# 9. S3 HELPERS
# ==============================================================================
def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    if 200 <= num <= 205: return '200'
    if 220 <= num <= 225: return '220'
    return 'other'


def load_seed_scores(ticker, experiment_base):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    bucket = gcs_client.bucket(BUCKET_NAME)
    all_results = {}
    for results_dir in [RESULTS_FALLBACK, RESULTS_DIR]:
        prefix = f"{results_dir}/Universal_Exp{experiment_base}/{clean}/Seed_"
        blobs = list(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            name = blob.name.split('/')[-1]
            if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
                continue
            seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
            d = pd.read_csv(io.BytesIO(blob.download_as_bytes()),
                            index_col=0, parse_dates=True)
            if d.index.tz is None:
                d.index = d.index.tz_localize('UTC')
            d.index = d.index.tz_convert(PST)
            d = d.sort_index()
            if seed_id in all_results:
                old = all_results[seed_id]
                combined = pd.concat([old, d])
                combined = combined[~combined.index.duplicated(keep='last')]
                all_results[seed_id] = combined.sort_index()
            else:
                all_results[seed_id] = d
    return all_results


def load_ticker_ohlcv(ticker):
    """Load OHLCV with PST index for S3 LONG tickers."""
    if ticker in _master_cache:
        return _master_cache[ticker]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}{suffix}.parquet"
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
            df = df[df.index.notna()].sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df.index = df.index.tz_convert(PST)
            std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                       'close': 'Close', 'volume': 'Volume'}
            lower_map = {k.lower(): v for k, v in std_map.items()}
            df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                          for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
            _master_cache[ticker] = df
            return df
        except Exception:
            continue
    return None


def precompute_ohlcv_arrays_pst(df_ohlcv, start_date, end_date):
    """OHLCV arrays for S3 tickers (already in PST)."""
    eval_start = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    eval_end   = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= eval_start) & (df_ohlcv.index < eval_end)]
    if len(df) == 0:
        return None
    dates = (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64)
    times = (df.index.hour * 60 + df.index.minute).values.astype(np.int64)
    return {
        'prices': df['Close'].values.astype(np.float64).copy(),
        'highs':  df['High'].values.astype(np.float64).copy(),
        'lows':   df['Low'].values.astype(np.float64).copy(),
        'dates':  dates,
        'times':  times,
    }


# ==============================================================================
# 10. BQ CONFIG LOADING
# ==============================================================================
def load_bq_configs_short(model_version, tickers):
    q = f"""
    SELECT ticker, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, sl_mult, grace_minutes,
           opt_avg_pnl, val_a_avg_pnl, val_a_win_rate, val_a_count,
           is_ensemble, model_version
    FROM `{BQ_TABLE}`
    WHERE model_version = '{model_version}'
      AND opt_avg_pnl > 0
      AND tpsl_mode IN ({", ".join(f"'{m}'" for m in BQ_ALLOWED_MODES)})
      AND is_ensemble = TRUE
    ORDER BY ticker, val_a_avg_pnl DESC
    """
    df = bq_client.query(q).to_dataframe()
    if len(df) == 0:
        print(f"    WARNING: No BQ results for {model_version}")
        return {}
    for col in ['prob_thresh', 'tp_pct', 'sl_pct', 'sl_mult', 'grace_minutes',
                'entry_minute', 'val_a_avg_pnl']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    configs = {}
    for ticker in tickers:
        t_df = df[df['ticker'] == ticker]
        if len(t_df) == 0:
            continue
        best_idx = t_df['val_a_avg_pnl'].idxmax()
        row = t_df.loc[best_idx]
        configs[ticker] = {
            'tpsl_mode': row['tpsl_mode'],
            'prob_thresh': float(row['prob_thresh']),
            'tp_pct': float(row['tp_pct']),
            'sl_pct': float(row['sl_pct']),
            'sl_mult': float(row['sl_mult']),
            'grace_minutes': int(row['grace_minutes']),
            'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else 405,
            'val_a_avg_pnl': float(row['val_a_avg_pnl']),
        }
    return configs


def load_bq_configs_s3():
    print(f"    Loading S3 LONG configs from BQ (family-agreement gate)...")
    s3_q = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           trail_start_mult, trail_end_mult, grace_minutes,
           dip_pct, max_wait_minutes, vol_filter_pct,
           val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
           val_b_avg_pnl, opt_avg_pnl, is_ensemble
    FROM `{BQ_TABLE}`
    WHERE model_version IN {str(S3_MODEL_VERSIONS)}
      AND survived_a = TRUE
      AND val_a_count >= 3
      AND val_a_avg_pnl > 0
    """
    raw_s3 = bq_client.query(s3_q).to_dataframe()
    print(f"    {len(raw_s3)} raw BQ rows")

    raw_s3 = raw_s3[raw_s3['opt_avg_pnl'] > 0].copy()
    raw_s3 = raw_s3[raw_s3['val_a_avg_pnl'] / raw_s3['opt_avg_pnl'] <= S3_MAX_VA_OPT]
    indiv = raw_s3[raw_s3['is_ensemble'] == False].copy()
    for k in S3_GRP_KEYS:
        if k not in indiv.columns: indiv[k] = 0
        else: indiv[k] = indiv[k].fillna(0)

    sa = indiv.groupby(S3_GRP_KEYS).agg(
        n_seeds=('seed', 'nunique'),
        n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= S3_DEPLOY_THRESHOLD).sum()),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        vb_pnl=('val_b_avg_pnl', 'mean'),
    ).reset_index()
    sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)

    sa = sa[sa['n_seeds'] >= S3_MIN_SEEDS].copy()
    sa_limit = sa[sa['tpsl_mode'] == 'limit_entry'].copy()
    sa_limit['score'] = (sa_limit['mean_va_pnl'] ** 2) * sa_limit['agree_18']
    sa_limit['exp_fam'] = sa_limit['experiment'].apply(assign_family)
    sa_fam = sa_limit[sa_limit['exp_fam'] != 'other'].copy()

    best_per_fam = sa_fam.loc[sa_fam.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    best_per_fam = best_per_fam[best_per_fam['mean_va_pnl'] >= S3_DEPLOY_THRESHOLD]

    agreement = {}
    for tk in best_per_fam['ticker'].unique():
        if tk in S3_EXCLUDE: continue
        tk_rows = best_per_fam[best_per_fam['ticker'] == tk]
        n_fams = len(tk_rows)
        n_pos = int((tk_rows['vb_pnl'] > 0).sum())
        agreement[tk] = {'n_fams': n_fams, 'n_pos': n_pos,
                         'majority_pos': n_pos > n_fams / 2 and n_fams >= 2}

    s3_gate_tickers = sorted([tk for tk, ag in agreement.items() if ag['majority_pos']])
    print(f"    Family agreement: {len(s3_gate_tickers)} tickers pass S3 gate")

    sa_s3 = sa_limit[sa_limit['ticker'].isin(s3_gate_tickers)].copy()
    sa_s3['score'] = (sa_s3['mean_va_pnl'] ** 2) * sa_s3['agree_18']
    best_s3 = sa_s3.loc[sa_s3.groupby('ticker')['score'].idxmax()]
    best_s3 = best_s3[best_s3['mean_va_pnl'] >= S3_DEPLOY_THRESHOLD]

    s3_configs = {}
    for _, row in best_s3.iterrows():
        tk = row['ticker']
        s3_configs[tk] = {
            'prob_thresh': float(row['prob_thresh']),
            'tp_pct': float(row['tp_pct']),
            'sl_pct': float(row['sl_pct']),
            'mode': 'limit_entry',
            'dip_pct': float(row['dip_pct']),
            'max_wait_minutes': float(row['max_wait_minutes']),
            'grace_minutes': 0,
            'experiment': row['experiment'],
            'entry_minute': int(row['entry_minute']) if pd.notna(row['entry_minute']) else S3_ENTRY_MINUTE,
            'val_a_pnl': float(row['mean_va_pnl']),
        }

    del raw_s3, indiv, sa, sa_limit, sa_fam, best_per_fam, sa_s3, best_s3
    _release_memory()
    return s3_configs


# ==============================================================================
# 11. RUN SHORT LEGS
# ==============================================================================
def run_leg(leg_config, bq_configs, market_df, all_tickers):
    leg_name = leg_config['name']
    entry_min = leg_config['entry_min']
    asset_cols = leg_config['asset_cols']
    context_cols = leg_config['context_cols']
    all_cols = leg_config['all_cols']
    variant = leg_config['feature_variant']

    print(f"\n  {'─'*W}")
    print(f"  LEG: {leg_name} ({leg_config['label']})")
    print(f"  {'─'*W}")

    # Load scaling stats
    print(f"    Loading scaling stats: {leg_config['scaling_file']}")
    try:
        stats = load_scaling_stats(leg_config['scaling_file'])
        print(f"    Scaling stats: {len(stats)} features")
    except Exception as e:
        print(f"    ERROR loading scaling stats: {e}")
        return [], {}

    # Load models
    print(f"    Loading {SEEDS_PER_FOLD} models for {leg_config['exp_id']}...")
    t0 = time.time()
    models = load_seed_models(leg_config)
    print(f"    Models loaded: {len(models)}/{SEEDS_PER_FOLD} ({time.time()-t0:.0f}s)")
    if not models:
        print(f"    ERROR: No models loaded -- skipping leg")
        return [], {}

    tickers_with_config = [t for t in all_tickers if t in bq_configs]
    print(f"    Tickers with BQ configs: {len(tickers_with_config)}")

    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt   = pd.Timestamp(TEST_END).date()
    h2_start_dt   = pd.Timestamp(HALF2_START).date()
    lookback_start = pd.Timestamp(TEST_START, tz='UTC') - pd.Timedelta(days=10)

    all_trades = []
    per_ticker = defaultdict(list)

    for ticker in sorted(tickers_with_config):
        cfg = bq_configs[ticker]
        df = load_master(ticker)
        if df is None:
            continue
        df = df[df.index >= lookback_start]
        df = heal_data(df)

        if variant == '930':
            df = engineer_features_930(df, market_df)
        else:
            df = engineer_features_10am(df, market_df)

        for col in all_cols:
            if col not in df.columns:
                df[col] = 0.0

        df_scaled = apply_scaling(df, stats, all_cols)
        ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
        if ohlcv is None:
            _master_cache.pop(ticker, None)
            continue

        # ATR for atr_sl mode
        if cfg['tpsl_mode'] == 'atr_sl':
            atr_pct = compute_daily_atr_pct(df)
            s_pst = pd.to_datetime(TEST_START).tz_localize(PST)
            e_pst = pd.to_datetime(TEST_END).tz_localize(PST) + timedelta(days=1)
            idx_c = df.index.tz_convert(PST) if str(df.index.tz) != 'America/Los_Angeles' else df.index
            atr_cut = atr_pct[(idx_c >= s_pst) & (idx_c < e_pst)]
            ohlcv['atr_pct'] = atr_cut.values.astype(np.float64)

        # Find entry bars in test period for scoring
        df_test = df_scaled[df_scaled.index >= pd.Timestamp(TEST_START, tz='UTC')]
        df_test = df_test[df_test.index <= pd.Timestamp(TEST_END, tz='UTC') + pd.Timedelta(days=1)]
        idx_pst = df_test.index.tz_convert(PST)
        mins = idx_pst.hour * 60 + idx_pst.minute
        entry_mask = (mins == entry_min) & (df_test['Volume'] > 0 if 'Volume' in df_test.columns else True)
        entry_positions_in_scaled = np.where(entry_mask)[0]

        global_entry_indices = []
        for pos in entry_positions_in_scaled:
            global_idx = df_scaled.index.get_loc(df_test.index[pos])
            global_entry_indices.append(global_idx)

        daily_probs = batch_score_ticker(
            df_scaled, global_entry_indices, models, asset_cols, context_cols,
            other_idx=leg_config['other_idx'])

        if not daily_probs:
            _master_cache.pop(ticker, None)
            continue

        # Simulate trades using OHLCV arrays
        times_arr = ohlcv['times']
        dates_arr = ohlcv['dates']
        ohlcv_entry_indices = np.where(times_arr == entry_min)[0]
        last_traded_date = -1

        for idx in ohlcv_entry_indices:
            d_int = int(dates_arr[idx])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            if d_int == last_traded_date:
                continue

            prob = daily_probs.get(d_date)
            if prob is None:
                continue
            if prob < cfg['prob_thresh']:
                continue

            pnl, trade_date, filled = execute_trade(
                ohlcv, idx, cfg['tpsl_mode'], cfg['tp_pct'], cfg['sl_pct'],
                cfg['sl_mult'], cfg['grace_minutes'])
            if not filled:
                continue
            last_traded_date = d_int

            # Apply slippage to SHORT trades
            pnl = pnl - SLIPPAGE_RT

            half = 'H2' if d_date >= h2_start_dt else 'H1'
            trade = {
                'date': d_date, 'ticker': ticker, 'pnl': pnl,
                'prob': prob, 'close': float(ohlcv['prices'][idx]),
                'mode': cfg['tpsl_mode'], 'tp': cfg['tp_pct'],
                'sl': cfg['sl_pct'] if cfg['tpsl_mode'] != 'atr_sl' else cfg['sl_mult'],
                'half': half, 'leg': leg_name, 'slot': leg_config['label'],
                'direction': 'SHORT',
            }
            all_trades.append(trade)
            per_ticker[ticker].append(trade)

        _master_cache.pop(ticker, None)
        n_trades = len(per_ticker.get(ticker, []))
        n_scored = len(daily_probs)
        if n_trades > 0:
            avg = np.mean([t['pnl'] for t in per_ticker[ticker]])
            print(f"    {ticker}: {n_scored} bars scored, {n_trades} trades, avg={avg:+.2%}")
        _release_memory()

    # Free models
    del models
    tf.keras.backend.clear_session()
    _release_memory()

    print(f"\n  {leg_name} TOTAL: {len(all_trades)} trades across "
          f"{len(per_ticker)} tickers")
    return all_trades, dict(per_ticker)


# ==============================================================================
# PORTFOLIO ANALYSIS HELPERS
# ==============================================================================
def summarize_trades(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0, 'sharpe': 0, 'trades_df': pd.DataFrame(),
                'h1_n': 0, 'h1_avg': 0, 'h1_total': 0,
                'h2_n': 0, 'h2_avg': 0, 'h2_total': 0, 'h2_wr': 0,
                'monthly': {}, 'n_tickers': 0}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    cum = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    h1 = df[df['half'] == 'H1']
    h2 = df[df['half'] == 'H2']
    daily_pnl = df.groupby('date')['pnl'].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() > 0 else 0)
    return {
        'label': label, 'n': len(df),
        'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': max_dd, 'sharpe': float(sharpe),
        'trades_df': df,
        'n_tickers': df['ticker'].nunique(),
        'h1_n': len(h1), 'h1_avg': h1['pnl'].mean() if len(h1) > 0 else 0,
        'h1_total': h1['pnl'].sum() if len(h1) > 0 else 0,
        'h2_n': len(h2), 'h2_avg': h2['pnl'].mean() if len(h2) > 0 else 0,
        'h2_total': h2['pnl'].sum() if len(h2) > 0 else 0,
        'h2_wr': (h2['pnl'] > 0).mean() if len(h2) > 0 else 0,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
    }


def compute_capital_adjusted(trades_list, deploy_floor=None):
    if not trades_list:
        return {'n_days': 0, 'daily_avg': 0, 'daily_total': 0, 'daily_wr': 0,
                'daily_dd': 0, 'avg_positions': 0, 'daily_series': pd.Series(dtype=float),
                'h2_daily_avg': 0, 'h2_daily_total': 0, 'h2_n_days': 0}
    df = pd.DataFrame(trades_list)
    if deploy_floor is not None:
        tk_avg = df.groupby('ticker')['pnl'].mean()
        good_tickers = tk_avg[tk_avg >= deploy_floor].index
        df = df[df['ticker'].isin(good_tickers)]
    if len(df) == 0:
        return {'n_days': 0, 'daily_avg': 0, 'daily_total': 0, 'daily_wr': 0,
                'daily_dd': 0, 'avg_positions': 0, 'daily_series': pd.Series(dtype=float),
                'h2_daily_avg': 0, 'h2_daily_total': 0, 'h2_n_days': 0}
    daily = df.groupby('date').agg(
        day_pnl=('pnl', 'mean'),
        n_positions=('pnl', 'count'),
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    cum = daily['day_pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    h2_mask = daily.index >= pd.Timestamp(HALF2_START)
    h2_daily = daily[h2_mask]
    return {
        'n_days': len(daily),
        'daily_avg': daily['day_pnl'].mean(),
        'daily_total': daily['day_pnl'].sum(),
        'daily_wr': (daily['day_pnl'] > 0).mean(),
        'daily_dd': max_dd,
        'avg_positions': daily['n_positions'].mean(),
        'daily_series': daily['day_pnl'],
        'h2_daily_avg': h2_daily['day_pnl'].mean() if len(h2_daily) > 0 else 0,
        'h2_daily_total': h2_daily['day_pnl'].sum() if len(h2_daily) > 0 else 0,
        'h2_n_days': len(h2_daily),
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)
t_total = time.time()

print(f"\n{'='*W}")
print(f"  F425 vs F435 SHORT + S3 LONG -- COMPARATIVE BACKTEST")
print(f"  TEST: {TEST_START} -> {TEST_END}  |  H2: {HALF2_START}+")
print(f"{'='*W}")
print(f"  LEG 1: {LEG_F425['label']} -- {len(ASSET_COLS_10AM)} asset + {len(CONTEXT_COLS_10AM)} context features")
print(f"  LEG 2: {LEG_F435['label']} -- {len(ASSET_COLS_10AM)} asset + {len(CONTEXT_COLS_10AM)} context features")
print(f"  LEG 3: S3 LONG (family agreement gate, limit_entry, CSV-scored)")
print(f"  Deploy threshold: {DEPLOY_FLOOR:.1%}")
print(f"  Slippage: {SLIPPAGE_PCT:.1%} one-way ({SLIPPAGE_RT:.2%} RT)")
print(f"  GPU: {tf.config.list_physical_devices('GPU')}")


# ── Step 1: Load BQ configs ─────────────────────────────────────────────────
print(f"\n  STEP 1: Loading BQ configs...")
cfgs_403 = load_bq_configs_short(LEG_F403['model_version'], EXPANDED_SHORT)
print(f"    F403 configs: {len(cfgs_403)} tickers")

cfgs_425 = load_bq_configs_short(LEG_F425['model_version'], EXPANDED_SHORT)
print(f"    F425 configs: {len(cfgs_425)} tickers")

cfgs_435 = load_bq_configs_short(LEG_F435['model_version'], EXPANDED_SHORT)
print(f"    F435 configs: {len(cfgs_435)} tickers")

for name, cfgs in [('F403', cfgs_403), ('F425', cfgs_425), ('F435', cfgs_435)]:
    for ticker in sorted(cfgs)[:5]:
        c = cfgs[ticker]
        print(f"    {name} {ticker}: {c['tpsl_mode']} prob>{c['prob_thresh']:.2f} "
              f"TP={c['tp_pct']*100:.0f}% val_a={c['val_a_avg_pnl']*100:+.2f}%")
    if len(cfgs) > 5:
        print(f"    ... +{len(cfgs)-5} more")

s3_configs = load_bq_configs_s3()
s3_tickers = sorted(s3_configs.keys())
print(f"    {len(s3_tickers)} S3 tickers: {', '.join(s3_tickers)}")
for tk in s3_tickers[:5]:
    c = s3_configs[tk]
    print(f"      {tk:<7} {c['experiment']:<16} Prob>{c['prob_thresh']:.2f}  "
          f"TP={c['tp_pct']:.0%}  SL={c['sl_pct']:.0%}  "
          f"dip={c['dip_pct']:.1%}  wait={c['max_wait_minutes']:.0f}m  "
          f"valA={c['val_a_pnl']:+.2%}")
if len(s3_tickers) > 5:
    print(f"      ... +{len(s3_tickers)-5} more")


# ── Step 2: Build market context ────────────────────────────────────────────
print(f"\n  STEP 2: Building market context...")
t0 = time.time()
market_df_930 = build_market_context(include_vix_level=False)
print(f"    Market context 9:30: {market_df_930.shape} ({time.time()-t0:.1f}s)")
_master_cache.clear()

t0 = time.time()
market_df_10am = build_market_context(include_vix_level=True)
print(f"    Market context 10AM: {market_df_10am.shape} ({time.time()-t0:.1f}s)")
_master_cache.clear()


# ── Step 3: Run F403 leg ────────────────────────────────────────────────────
print(f"\n  STEP 3: Scoring + trading F403 (9:30, 2.5% dump)...")
np.random.seed(42)
trades_403, per_tk_403 = run_leg(LEG_F403, cfgs_403, market_df_930, EXPANDED_SHORT)


# ── Step 4: Run F425 leg ────────────────────────────────────────────────────
print(f"\n  STEP 4: Scoring + trading F425 (10AM, 2.5% dump)...")
np.random.seed(42)
trades_425, per_tk_425 = run_leg(LEG_F425, cfgs_425, market_df_10am, EXPANDED_SHORT)


# ── Step 5: Run F435 leg ────────────────────────────────────────────────────
print(f"\n  STEP 5: Scoring + trading F435 (10AM, 8% dump)...")
np.random.seed(42)
trades_435, per_tk_435 = run_leg(LEG_F435, cfgs_435, market_df_10am, EXPANDED_SHORT)


# ── Step 6: Run S3 LONG ────────────────────────────────────────────────────
print(f"\n  STEP 6: S3 LONG -- CSV scoring + trade simulation...")

# 12. RUN S3 — CSV scoring + LONG trade sim
print(f"    Loading S3 CSV scores from GCS...")
s3_daily_probs = {}
for tk in s3_tickers:
    cfg = s3_configs[tk]
    exp = cfg['experiment']
    exp_base = exp.replace('_LIMIT', '').replace('_GRACE', '')
    entry_min = cfg.get('entry_minute', S3_ENTRY_MINUTE)

    seeds_dict = load_seed_scores(tk, exp_base)
    if not seeds_dict:
        print(f"    {tk}: no CSV scores for {exp_base}")
        continue

    avg_probs = None
    n_seeds = 0
    for seed_id, sdf in seeds_dict.items():
        if 'Prob_Spike' not in sdf.columns:
            col = [c for c in sdf.columns if 'prob' in c.lower() or 'spike' in c.lower()]
            if not col: continue
            sdf = sdf.rename(columns={col[0]: 'Prob_Spike'})
        bar_mins = sdf.index.hour * 60 + sdf.index.minute
        entry_bars = sdf[bar_mins == entry_min]
        if entry_bars.empty: continue
        daily_prob = entry_bars.groupby(entry_bars.index.date)['Prob_Spike'].first()
        if avg_probs is None:
            avg_probs = daily_prob.copy()
        else:
            avg_probs = avg_probs.add(daily_prob, fill_value=0)
        n_seeds += 1

    if n_seeds > 0 and avg_probs is not None:
        prob_dict = {}
        for dt, prob in (avg_probs / n_seeds).items():
            d_int = dt.year * 10000 + dt.month * 100 + dt.day
            prob_dict[d_int] = float(prob)
        s3_daily_probs[tk] = prob_dict
        test_start_int = int(TEST_START.replace('-', ''))
        test_end_int = int(TEST_END.replace('-', ''))
        n_test = sum(1 for d in prob_dict if test_start_int <= d <= test_end_int)
        print(f"    {tk}: {len(prob_dict)} total dates, {n_test} in test, {n_seeds} seeds ({exp_base})")
    else:
        print(f"    {tk}: no valid seed scores")

_release_memory()

# Load OHLCV and run S3 LONG trades
print(f"\n    Running S3 LONG trade simulation...")
s3_trades = []
s3_per_tk = defaultdict(list)
test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

for ticker in s3_tickers:
    cfg = s3_configs[ticker]
    df = load_ticker_ohlcv(ticker)
    if df is None:
        print(f"    {ticker}: FAILED to load OHLCV")
        continue
    df = heal_data(df.sort_index())
    ohlcv = precompute_ohlcv_arrays_pst(df, TEST_START, TEST_END)
    if ohlcv is None:
        _master_cache.pop(ticker, None)
        continue

    probs_dict = s3_daily_probs.get(ticker, {})
    if not probs_dict:
        _master_cache.pop(ticker, None)
        continue

    dates = ohlcv['dates']; times = ohlcv['times']
    entry_min = cfg.get('entry_minute', S3_ENTRY_MINUTE)
    probs = np.zeros(len(dates), dtype=np.float64)
    for i in range(len(dates)):
        if times[i] == entry_min:
            probs[i] = probs_dict.get(int(dates[i]), 0.0)

    mask = (probs >= cfg['prob_thresh']) & (times == entry_min)
    indices = np.where(mask)[0].astype(np.int64)
    if len(indices) == 0:
        _master_cache.pop(ticker, None)
        continue

    np.random.seed(42)
    pnl_arr, trade_dates = run_trade_sim_limit_entry(
        indices, ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
        ohlcv['dates'], ohlcv['times'].astype(np.int64),
        cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
        cfg['dip_pct'], cfg['max_wait_minutes'])

    # Apply slippage to LONG trades
    if len(pnl_arr) > 0:
        pnl_arr = pnl_arr - SLIPPAGE_RT

    for j in range(len(pnl_arr)):
        d_int = int(trade_dates[j])
        d_str = f"{d_int // 10000}-{(d_int % 10000) // 100:02d}-{d_int % 100:02d}"
        d_date = pd.Timestamp(d_str).date()
        half = 'H2' if d_date >= h2_start_dt else 'H1'
        trade = {
            'date': d_date, 'ticker': ticker, 'pnl': float(pnl_arr[j]),
            'half': half, 'leg': 'S3_LONG', 'slot': 'S3',
            'direction': 'LONG', 'mode': 'limit_entry',
        }
        s3_trades.append(trade)
        s3_per_tk[ticker].append(trade)

    if len(pnl_arr) > 0:
        print(f"    {ticker}: {len(pnl_arr)} trades, avg={np.mean(pnl_arr):+.2%}, "
              f"WR={(pnl_arr > 0).mean():.0%}")

    _master_cache.pop(ticker, None)
    _release_memory()

s3_per_tk = dict(s3_per_tk)
print(f"\n    S3 LONG TOTAL: {len(s3_trades)} trades across {len(s3_per_tk)} tickers")


# ==============================================================================
# 13. BUILD STRATEGIES
# ==============================================================================
print(f"\n  STEP 6: Building 5 strategies...")

# Tag trades
for t in trades_403:
    t['slot'] = 'F403'
for t in trades_425:
    t['slot'] = 'F425'
for t in trades_435:
    t['slot'] = 'F435'

# ADDITIVE = F403 + F425 (deployed ADDITIVE_403)
additive_trades = trades_403 + trades_425
additive_per_tk = defaultdict(list)
for tk, tl in per_tk_403.items():
    additive_per_tk[tk].extend(tl)
for tk, tl in per_tk_425.items():
    additive_per_tk[tk].extend(tl)
additive_per_tk = dict(additive_per_tk)

# ALL_SHORT = F403 + F425 + F435 (all three legs)
all_short_trades = trades_403 + trades_425 + trades_435
all_short_per_tk = defaultdict(list)
for tk, tl in per_tk_403.items():
    all_short_per_tk[tk].extend(tl)
for tk, tl in per_tk_425.items():
    all_short_per_tk[tk].extend(tl)
for tk, tl in per_tk_435.items():
    all_short_per_tk[tk].extend(tl)
all_short_per_tk = dict(all_short_per_tk)

# HEDGED = S3_LONG + ADDITIVE (F403+F425)
hedged_trades = additive_trades + s3_trades
hedged_per_tk = defaultdict(list)
for tk, tl in additive_per_tk.items():
    hedged_per_tk[tk].extend(tl)
for tk, tl in s3_per_tk.items():
    hedged_per_tk[tk].extend(tl)
hedged_per_tk = dict(hedged_per_tk)

# Build summaries
port_403  = summarize_trades(trades_403, 'F403_ONLY')
port_425  = summarize_trades(trades_425, 'F425_ONLY')
port_435  = summarize_trades(trades_435, 'F435_ONLY')
port_add  = summarize_trades(additive_trades, 'ADDITIVE')
port_all  = summarize_trades(all_short_trades, 'ALL_SHORT')
port_s3   = summarize_trades(s3_trades, 'S3_LONG')
port_hedge = summarize_trades(hedged_trades, 'HEDGED')

strategies = {
    'F403_ONLY': {'summary': port_403,  'per_ticker': per_tk_403,          'trades': trades_403},
    'F425_ONLY': {'summary': port_425,  'per_ticker': per_tk_425,          'trades': trades_425},
    'F435_ONLY': {'summary': port_435,  'per_ticker': per_tk_435,          'trades': trades_435},
    'ADDITIVE':  {'summary': port_add,  'per_ticker': additive_per_tk,     'trades': additive_trades},
    'ALL_SHORT': {'summary': port_all,  'per_ticker': all_short_per_tk,    'trades': all_short_trades},
    'S3_LONG':   {'summary': port_s3,   'per_ticker': s3_per_tk,           'trades': s3_trades},
    'HEDGED':    {'summary': port_hedge, 'per_ticker': hedged_per_tk,      'trades': hedged_trades},
}
STRAT_ORDER = ['F403_ONLY', 'F425_ONLY', 'F435_ONLY', 'ADDITIVE', 'ALL_SHORT', 'S3_LONG', 'HEDGED']

elapsed = time.time() - t_total
print(f"\n  Total scoring time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


# ==============================================================================
# TABLE 1: HEAD-TO-HEAD (5 strategies)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: HEAD-TO-HEAD COMPARISON — 7 STRATEGIES ({TEST_START} -> {TEST_END})")
print(f"{'='*W}")

print(f"\n  {'Strategy':<14} {'#Tk':>4} {'Trades':>6} {'Avg PnL':>8} {'Total':>9} "
      f"{'WR':>5} {'MaxDD':>7} {'Sharpe':>7}")
print(f"  {'─'*65}")

for sname in STRAT_ORDER:
    p = strategies[sname]['summary']
    if p['n'] == 0:
        print(f"  {sname:<14} -- no trades")
        continue
    print(f"  {sname:<14} {p['n_tickers']:>4} {p['n']:>6} {p['avg']:>+7.2%} "
          f"{p['total']:>+8.1%} {p['wr']:>5.0%} {p['dd']:>+7.2%} {p['sharpe']:>6.1f}")


# ==============================================================================
# TABLE 2: F403 vs F425 vs F435 DIRECT COMPARISON (per-ticker + overlap)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 2: F403 vs F425 vs F435 DIRECT COMPARISON")
print(f"{'='*W}")

all_short_tickers = sorted(set(list(per_tk_403.keys()) + list(per_tk_425.keys()) + list(per_tk_435.keys())))
print(f"\n  {'Ticker':<7} {'F403 #':>6} {'F403 Avg':>9}  "
      f"{'F425 #':>6} {'F425 Avg':>9}  "
      f"{'F435 #':>6} {'F435 Avg':>9}  {'Winner':>8}")
print(f"  {'─'*80}")

for ticker in all_short_tickers:
    t403 = per_tk_403.get(ticker, [])
    t425 = per_tk_425.get(ticker, [])
    t435 = per_tk_435.get(ticker, [])
    n403 = len(t403); n425 = len(t425); n435 = len(t435)
    avg403 = np.mean([t['pnl'] for t in t403]) if t403 else 0
    avg425 = np.mean([t['pnl'] for t in t425]) if t425 else 0
    avg435 = np.mean([t['pnl'] for t in t435]) if t435 else 0
    avgs = {'F403': avg403, 'F425': avg425, 'F435': avg435}
    active = {k: v for k, v in avgs.items() if (k == 'F403' and n403) or (k == 'F425' and n425) or (k == 'F435' and n435)}
    winner = max(active, key=active.get) if active else '--'
    f403_str = f"{n403:>6} {avg403:>+8.2%}" if n403 else f"{'--':>6} {'--':>9}"
    f425_str = f"{n425:>6} {avg425:>+8.2%}" if n425 else f"{'--':>6} {'--':>9}"
    f435_str = f"{n435:>6} {avg435:>+8.2%}" if n435 else f"{'--':>6} {'--':>9}"
    print(f"  {ticker:<7} {f403_str}  {f425_str}  {f435_str}  {winner:>8}")

# Overlap analysis
tickers_403_set = set(per_tk_403.keys())
tickers_425_set = set(per_tk_425.keys())
tickers_435_set = set(per_tk_435.keys())
all_three = tickers_403_set & tickers_425_set & tickers_435_set
f403_only = tickers_403_set - tickers_425_set - tickers_435_set
f425_only = tickers_425_set - tickers_403_set - tickers_435_set
f435_only = tickers_435_set - tickers_403_set - tickers_425_set
f403_f425 = (tickers_403_set & tickers_425_set) - tickers_435_set
f403_f435 = (tickers_403_set & tickers_435_set) - tickers_425_set
f425_f435 = (tickers_425_set & tickers_435_set) - tickers_403_set
print(f"\n  TICKER OVERLAP:")
print(f"    F403 only:        {len(f403_only):>3} -- {', '.join(sorted(f403_only)) if f403_only else 'none'}")
print(f"    F425 only:        {len(f425_only):>3} -- {', '.join(sorted(f425_only)) if f425_only else 'none'}")
print(f"    F435 only:        {len(f435_only):>3} -- {', '.join(sorted(f435_only)) if f435_only else 'none'}")
print(f"    F403+F425:        {len(f403_f425):>3} -- {', '.join(sorted(f403_f425)) if f403_f425 else 'none'}")
print(f"    F403+F435:        {len(f403_f435):>3} -- {', '.join(sorted(f403_f435)) if f403_f435 else 'none'}")
print(f"    F425+F435:        {len(f425_f435):>3} -- {', '.join(sorted(f425_f435)) if f425_f435 else 'none'}")
print(f"    All three:        {len(all_three):>3} -- {', '.join(sorted(all_three)) if all_three else 'none'}")
print(f"    Total unique:     {len(all_short_tickers):>3}")


# ==============================================================================
# TABLE 3: MONTHLY BREAKDOWN (all 5 strategies)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: MONTHLY BREAKDOWN")
print(f"{'='*W}")

all_months = sorted(set(m for sname in STRAT_ORDER
                        for m in strategies[sname]['summary'].get('monthly', {}).keys()))

for sname in STRAT_ORDER:
    p = strategies[sname]['summary']
    monthly = p.get('monthly', {})
    if p['n'] == 0:
        continue
    print(f"\n  {sname}:")
    print(f"    {'Month':<10} {'Trades':>6} {'Avg':>7} {'Total':>8} {'WR':>5}")
    print(f"    {'─'*40}")
    for m in all_months:
        if m in monthly:
            d = monthly[m]
            print(f"    {m:<10} {d['n']:>6} {d['avg']:>+6.2%} {d['total']:>+7.1%} {d['wr']:>4.0%}")
        else:
            print(f"    {m:<10}     --")


# ==============================================================================
# TABLE 4: PER-TICKER BREAKDOWN (all strategies)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 4: PER-TICKER BREAKDOWN")
print(f"{'='*W}")

all_tickers_union = sorted(set(
    tk for sname in STRAT_ORDER
    for tk in strategies[sname]['per_ticker'].keys()
))

col_w = 18
print(f"\n  {'Ticker':<7}", end="")
for sname in STRAT_ORDER:
    print(f"  {sname:<{col_w}}", end="")
print()
print(f"  {'─'*7}", end="")
for _ in STRAT_ORDER:
    print(f"  {'─'*col_w}", end="")
print()

for ticker in all_tickers_union:
    print(f"  {ticker:<7}", end="")
    for sname in STRAT_ORDER:
        per_tk = strategies[sname]['per_ticker']
        trades = per_tk.get(ticker, [])
        if trades:
            avg = np.mean([t['pnl'] for t in trades])
            n = len(trades)
            wr = sum(1 for t in trades if t['pnl'] > 0) / n * 100
            cell = f"{avg:>+5.2%} {n:>3}t {wr:>3.0f}%w"
            print(f"  {cell:<{col_w}}", end="")
        else:
            print(f"  {'--':<{col_w}}", end="")
    print()


# ==============================================================================
# TABLE 5: CAPITAL-ADJUSTED DAILY PnL (deployed tickers only)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 5: CAPITAL-ADJUSTED PnL (deployed only, floor={DEPLOY_FLOOR:.1%})")
print(f"{'='*W}")
print(f"\n  {'Strategy':<14} {'Days':>5} {'Avg/Day':>8} {'Total':>8} {'WR':>5} "
      f"{'MaxDD':>7} {'Pos/Day':>7}  {'H2 Avg':>7} {'H2 Days':>7}")
print(f"  {'─'*85}")

for sname in STRAT_ORDER:
    per_tk = strategies[sname]['per_ticker']
    deploy_tickers = [tk for tk, trades in per_tk.items()
                      if trades and np.mean([t['pnl'] for t in trades]) >= DEPLOY_FLOOR]
    if not deploy_tickers:
        print(f"  {sname:<14} (no tickers above threshold)")
        continue
    deploy_trades = []
    for tk in deploy_tickers:
        deploy_trades.extend(per_tk[tk])
    ca = compute_capital_adjusted(deploy_trades)
    print(f"  {sname:<14} {ca['n_days']:>5} {ca['daily_avg']:>+7.2%} "
          f"{ca['daily_total']:>+7.1%} {ca['daily_wr']:>4.0%} "
          f"{ca['daily_dd']:>+7.2%} {ca['avg_positions']:>6.1f}  "
          f"{ca['h2_daily_avg']:>+7.2%} {ca['h2_n_days']:>7}")
    print(f"  {'':14} Deployed: {', '.join(sorted(deploy_tickers))}")


# ==============================================================================
# TABLE 5B: CAPITAL-ADJUSTED WITH POSITIVE-ONLY TICKERS (lower threshold for SHORT)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 5B: CAPITAL-ADJUSTED (positive avg tickers, no floor)")
print(f"{'='*W}")
print(f"\n  Strategy        Days  Avg/Day    Total    WR   MaxDD Pos/Day   H2 Avg H2 Days  Tickers")
print(f"  {'─'*100}")

for sname in STRAT_ORDER:
    per_tk = strategies[sname]['per_ticker']
    pos_tickers = [tk for tk, trades in per_tk.items()
                   if trades and np.mean([t['pnl'] for t in trades]) > 0]
    if not pos_tickers:
        print(f"  {sname:<14} (no positive tickers)")
        continue
    pos_trades = []
    for tk in pos_tickers:
        pos_trades.extend(per_tk[tk])
    ca = compute_capital_adjusted(pos_trades)
    print(f"  {sname:<14} {ca['n_days']:>5} {ca['daily_avg']:>+7.2%} "
          f"{ca['daily_total']:>+7.1%} {ca['daily_wr']:>4.0%} "
          f"{ca['daily_dd']:>+7.2%} {ca['avg_positions']:>6.1f}  "
          f"{ca['h2_daily_avg']:>+7.2%} {ca['h2_n_days']:>7}  "
          f"{', '.join(sorted(pos_tickers))}")


# ==============================================================================
# TABLE 5C: DAY-LEVEL ADDITIVE ANALYSIS — S3 vs SHORT coverage
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 5C: DAY-LEVEL ADDITIVE ANALYSIS")
print(f"{'='*W}")

def daily_pnl_series(trades_list):
    if not trades_list:
        return pd.Series(dtype=float)
    df = pd.DataFrame(trades_list)
    return df.groupby('date')['pnl'].mean()

s3_daily = daily_pnl_series(s3_trades)
f403_daily = daily_pnl_series(trades_403)
f425_daily = daily_pnl_series(trades_425)
f435_daily = daily_pnl_series(trades_435)
additive_daily = daily_pnl_series(additive_trades)
short_daily = daily_pnl_series(all_short_trades)

s3_dates = set(s3_daily.index) if len(s3_daily) > 0 else set()
f403_dates = set(f403_daily.index) if len(f403_daily) > 0 else set()
f425_dates = set(f425_daily.index) if len(f425_daily) > 0 else set()
additive_dates = set(additive_daily.index) if len(additive_daily) > 0 else set()
short_dates = set(short_daily.index) if len(short_daily) > 0 else set()
all_dates = s3_dates | short_dates

s3_only_dates = s3_dates - short_dates
short_only_dates = short_dates - s3_dates
both_dates = s3_dates & short_dates

print(f"\n  COVERAGE:")
print(f"    S3 LONG trading days:    {len(s3_dates):>4}")
print(f"    F403 trading days:       {len(f403_dates):>4}")
print(f"    F425 trading days:       {len(f425_dates):>4}")
print(f"    ADDITIVE (F403+F425):    {len(additive_dates):>4}")
print(f"    ALL SHORT (3 legs):      {len(short_dates):>4}")
print(f"    All active days:         {len(all_dates):>4}")
print(f"    S3-only days:            {len(s3_only_dates):>4}  (S3 trades, no SHORT)")
print(f"    SHORT-only days:         {len(short_only_dates):>4}  (SHORT trades, no S3)")
print(f"    Both-active days:        {len(both_dates):>4}  (both trade)")

print(f"\n  DAILY PnL BY DAY TYPE (ADDITIVE = F403+F425):")
print(f"  {'Day Type':<22} {'Days':>5} {'S3 Avg':>8} {'ADDIT Avg':>10} {'Combined':>10}")
print(f"  {'─'*60}")

additive_only_dates = additive_dates - s3_dates
s3_only_from_add = s3_dates - additive_dates
both_add_dates = s3_dates & additive_dates

if s3_only_from_add:
    s3_only_avg = s3_daily[s3_daily.index.isin(s3_only_from_add)].mean()
    print(f"  {'S3-only':<22} {len(s3_only_from_add):>5} {s3_only_avg:>+7.2%} {'--':>10} {s3_only_avg:>+9.2%}")

if additive_only_dates:
    add_only_avg = additive_daily[additive_daily.index.isin(additive_only_dates)].mean()
    print(f"  {'ADDITIVE-only':<22} {len(additive_only_dates):>5} {'--':>8} {add_only_avg:>+9.2%} {add_only_avg:>+9.2%}")

if both_add_dates:
    s3_both_avg = s3_daily[s3_daily.index.isin(both_add_dates)].mean()
    add_both_avg = additive_daily[additive_daily.index.isin(both_add_dates)].mean()
    both_combined = []
    for d in both_add_dates:
        day_trades = [t for t in hedged_trades if t['date'] == d]
        if day_trades:
            both_combined.append(np.mean([t['pnl'] for t in day_trades]))
    both_avg = np.mean(both_combined) if both_combined else 0
    print(f"  {'Both-active':<22} {len(both_add_dates):>5} {s3_both_avg:>+7.2%} {add_both_avg:>+9.2%} {both_avg:>+9.2%}")

combined_daily = pd.Series(dtype=float)
hedged_all_dates = s3_dates | additive_dates
for d in hedged_all_dates:
    day_trades = [t for t in hedged_trades if t['date'] == d]
    if day_trades:
        combined_daily[d] = np.mean([t['pnl'] for t in day_trades])
if len(combined_daily) > 0:
    print(f"  {'ALL ACTIVE':<22} {len(hedged_all_dates):>5} {'':>8} {'':>10} {combined_daily.mean():>+9.2%}")

# Monthly day-type breakdown
print(f"\n  MONTHLY: ADDITIVE-only days (when S3 is silent)")
print(f"  {'Month':<12} {'ADDIT-only days':>16} {'Avg PnL':>8} {'Trades':>7}")
print(f"  {'─'*48}")
if additive_only_dates:
    add_only_df = pd.DataFrame([(d, additive_daily[d]) for d in additive_only_dates],
                                columns=['date', 'pnl'])
    add_only_df['month'] = pd.to_datetime(add_only_df['date']).dt.to_period('M')
    for month, grp in add_only_df.groupby('month'):
        n_trades_month = len([t for t in additive_trades
                             if t['date'] in set(grp['date'])])
        print(f"  {str(month):<12} {len(grp):>16} {grp['pnl'].mean():>+7.2%} {n_trades_month:>7}")

# ADDITIVE contribution to S3
print(f"\n  ADDITIVE (F403+F425) CONTRIBUTION TO S3:")
if additive_only_dates:
    add_only_avg = additive_daily[additive_daily.index.isin(additive_only_dates)].mean()
    n_add_trades_solo = len([t for t in additive_trades if t['date'] in additive_only_dates])
    print(f"    ADDITIVE trades on non-S3 days: {len(additive_only_dates)} days, "
          f"{n_add_trades_solo} trades, avg daily PnL={add_only_avg:+.2%}")
add_s3_combined_daily = pd.Series(dtype=float)
all_add_s3_trades = additive_trades + s3_trades
for d in (additive_dates | s3_dates):
    day_trades = [t for t in all_add_s3_trades if t['date'] == d]
    if day_trades:
        add_s3_combined_daily[d] = np.mean([t['pnl'] for t in day_trades])
if len(add_s3_combined_daily) > 0:
    print(f"    S3-only portfolio:     {len(s3_dates)} active days, "
          f"daily avg={s3_daily.mean():+.2%}")
    print(f"    S3+ADDITIVE portfolio: {len(add_s3_combined_daily)} active days, "
          f"daily avg={add_s3_combined_daily.mean():+.2%}")
    pct_more_days = (len(add_s3_combined_daily) - len(s3_dates)) / max(len(s3_dates), 1) * 100
    print(f"    Coverage gain: +{pct_more_days:.0f}% more active days")

# Per-leg contribution
print(f"\n  PER-LEG BREAKDOWN:")
for leg_name, leg_daily, leg_dates, leg_trades in [
    ('F403', f403_daily, f403_dates, trades_403),
    ('F425', f425_daily, f425_dates, trades_425),
    ('F435', f435_daily, set(f435_daily.index) if len(f435_daily) > 0 else set(), trades_435),
]:
    solo_dates = leg_dates - s3_dates
    if solo_dates and len(leg_daily) > 0:
        solo_avg = leg_daily[leg_daily.index.isin(solo_dates)].mean()
        n_solo = len([t for t in leg_trades if t['date'] in solo_dates])
        print(f"    {leg_name} on non-S3 days: {len(solo_dates)} days, "
              f"{n_solo} trades, avg={solo_avg:+.2%}")


# ==============================================================================
# TABLE 6: OVERLAP ANALYSIS — which tickers does each strategy trade?
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 6: TICKER OVERLAP ANALYSIS")
print(f"{'='*W}")

strat_ticker_sets = {}
for sname in STRAT_ORDER:
    strat_ticker_sets[sname] = set(strategies[sname]['per_ticker'].keys())

print(f"\n  PAIRWISE OVERLAP:")
print(f"    {'':20}", end="")
for sname in STRAT_ORDER:
    print(f"  {sname:>10}", end="")
print()

for s1 in STRAT_ORDER:
    print(f"    {s1:<20}", end="")
    for s2 in STRAT_ORDER:
        overlap_n = len(strat_ticker_sets[s1] & strat_ticker_sets[s2])
        print(f"  {overlap_n:>10}", end="")
    print()

# SHORT vs LONG overlap
short_set = strat_ticker_sets.get('ADDITIVE', set())
long_set  = strat_ticker_sets.get('S3_LONG', set())
both_dirs = short_set & long_set
only_short_dir = short_set - long_set
only_long_dir  = long_set - short_set
print(f"\n  SHORT vs LONG DIRECTION OVERLAP:")
print(f"    SHORT only:  {len(only_short_dir)} -- {', '.join(sorted(only_short_dir)) if only_short_dir else 'none'}")
print(f"    LONG only:   {len(only_long_dir)} -- {', '.join(sorted(only_long_dir)) if only_long_dir else 'none'}")
print(f"    Both dirs:   {len(both_dirs)} -- {', '.join(sorted(both_dirs)) if both_dirs else 'none'}")

# Same-day hedge analysis
if both_dirs and s3_trades and additive_trades:
    short_df = pd.DataFrame(additive_trades)
    long_df = pd.DataFrame(s3_trades)
    short_date_set = set(zip(short_df['ticker'], short_df['date']))
    long_date_set = set(zip(long_df['ticker'], long_df['date']))
    same_day_hedge = short_date_set & long_date_set
    print(f"\n  SAME-DAY HEDGES (SHORT + LONG on same ticker/date):")
    print(f"    {len(same_day_hedge)} instances")
    if same_day_hedge:
        for tk, dt in sorted(same_day_hedge)[:10]:
            s_pnl = [t['pnl'] for t in additive_trades if t['ticker'] == tk and t['date'] == dt]
            l_pnl = [t['pnl'] for t in s3_trades if t['ticker'] == tk and t['date'] == dt]
            print(f"      {tk} {dt}: SHORT={sum(s_pnl):+.2%}, LONG={sum(l_pnl):+.2%}, "
                  f"NET={sum(s_pnl)+sum(l_pnl):+.2%}")
        if len(same_day_hedge) > 10:
            print(f"      ... +{len(same_day_hedge)-10} more")


# ==============================================================================
# 15. CHARTS (2x2 grid)
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Chart [0,0]: Equity curves (all 5 strategies)
ax1 = axes[0, 0]
for sname in STRAT_ORDER:
    p = strategies[sname]['summary']
    if p['n'] == 0:
        continue
    df_t = p['trades_df'].sort_values('date')
    cum = df_t['pnl'].cumsum() * 100
    ax1.plot(pd.to_datetime(df_t['date'].values), cum.values,
             color=STRATEGY_COLORS[sname], linewidth=2.5 if sname in ('ADDITIVE', 'HEDGED') else 1.8,
             label=f"{sname} ({p['n']}tr, {p['avg']:+.2%})", alpha=0.9,
             linestyle='-' if sname != 'HEDGED' else '--')
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax1.axvline(x=pd.Timestamp(HALF2_START), color='red', linewidth=1,
            linestyle=':', alpha=0.5, label='H2 start')
ax1.set_ylabel('Cumulative PnL (%)')
ax1.set_title('Equity Curves -- All 5 Strategies', fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Chart [0,1]: Monthly avg PnL bars (F403 vs ADDITIVE vs S3)
ax2 = axes[0, 1]
if all_months:
    x = np.arange(len(all_months))
    bar_width = 0.2
    chart_strats = ['F403_ONLY', 'ADDITIVE', 'S3_LONG', 'HEDGED']
    for i, sname in enumerate(chart_strats):
        p = strategies[sname]['summary']
        if p['n'] == 0: continue
        monthly = p.get('monthly', {})
        monthly_avgs = []
        for m in all_months:
            ms = monthly.get(m)
            monthly_avgs.append(ms['avg'] * 100 if ms else 0)
        bars = ax2.bar(x + i * bar_width, monthly_avgs, bar_width,
                       label=sname, color=STRATEGY_COLORS[sname], alpha=0.85)
        for bar, val in zip(bars, monthly_avgs):
            if abs(val) > 0.01:
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + (0.05 if val >= 0 else -0.15),
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=6)
    ax2.set_xticks(x + bar_width * 1.5)
    month_labels = [pd.Timestamp(m + '-01').strftime('%b') for m in all_months]
    ax2.set_xticklabels(month_labels, fontsize=10)
    ax2.set_ylabel('Avg PnL per Trade (%)')
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
ax2.set_title('Monthly Avg PnL -- F403 vs ADDITIVE vs S3 vs HEDGED', fontsize=12, fontweight='bold')

# Chart [1,0]: Per-ticker heatmap (F403 vs F425 vs F435)
ax3 = axes[1, 0]
if all_short_tickers:
    heatmap_data = np.full((len(all_short_tickers), 3), np.nan)
    for j, (sname, per_tk) in enumerate([('F403', per_tk_403), ('F425', per_tk_425), ('F435', per_tk_435)]):
        for i, ticker in enumerate(all_short_tickers):
            trades = per_tk.get(ticker, [])
            if trades:
                heatmap_data[i, j] = np.mean([t['pnl'] for t in trades]) * 100
    vmax = max(abs(np.nanmin(heatmap_data)), abs(np.nanmax(heatmap_data)))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1
    im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(['F403 (9:30)', 'F425 (10AM)', 'F435 (8%)'], fontsize=9)
    ax3.set_yticks(range(len(all_short_tickers)))
    ax3.set_yticklabels(all_short_tickers, fontsize=7)
    for i in range(len(all_short_tickers)):
        for j in range(3):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                ax3.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                         fontsize=6, fontweight='bold',
                         color='white' if abs(val) > vmax*0.5 else 'black')
    fig.colorbar(im, ax=ax3, label='Avg PnL (%)', shrink=0.8)
ax3.set_title('Per-Ticker Heatmap -- F403 vs F425 vs F435', fontsize=12, fontweight='bold')

# Chart [1,1]: Daily PnL histogram (ADDITIVE = F403+F425)
ax4 = axes[1, 1]
if port_add['n'] > 0:
    daily_pnls = port_add['trades_df'].groupby('date')['pnl'].mean() * 100
    ax4.hist(daily_pnls, bins=30, color=STRATEGY_COLORS['ADDITIVE'], alpha=0.75,
             edgecolor='black', linewidth=0.5)
    ax4.axvline(x=daily_pnls.mean(), color='red', linewidth=2, linestyle='--',
                label=f'Mean={daily_pnls.mean():.2f}%')
    ax4.axvline(x=0, color='gray', linewidth=1, linestyle='-')
    ax4.set_xlabel('Daily Avg PnL (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
ax4.set_title('Daily PnL Distribution -- ADDITIVE (F403+F425)', fontsize=12, fontweight='bold')

fig.suptitle(f'F403 + F425 + F435 SHORT + S3 LONG -- Comparative Backtest\n'
             f'{TEST_START} -> {TEST_END}  |  Deploy: {DEPLOY_FLOOR:.1%}  |  '
             f'Slippage: {SLIPPAGE_PCT:.1%} one-way',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/tmp/backtest_425_vs_435_vs_s3.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Chart saved: /tmp/backtest_425_vs_435_vs_s3.png")


# ==============================================================================
# VERDICT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

for sname in STRAT_ORDER:
    p = strategies[sname]['summary']
    per_tk = strategies[sname]['per_ticker']
    deploy = [tk for tk, trades in per_tk.items()
              if trades and np.mean([t['pnl'] for t in trades]) >= DEPLOY_FLOOR]
    deploy_trades = []
    for tk in deploy:
        deploy_trades.extend(per_tk[tk])
    ca = compute_capital_adjusted(deploy_trades) if deploy_trades else None

    print(f"\n  {sname}:")
    print(f"    All:      {p['n']:>4} trades ({p['n_tickers']} tickers), avg={p['avg']:>+.2%}, "
          f"total={p['total']:>+.1%}, Sharpe={p['sharpe']:.1f}")
    print(f"    H1:       {p['h1_n']:>4} trades, avg={p['h1_avg']:>+.2%}")
    print(f"    H2:       {p['h2_n']:>4} trades, avg={p['h2_avg']:>+.2%}, WR={p['h2_wr']:.0%}")
    if deploy:
        dp = summarize_trades(deploy_trades, '')
        print(f"    Deployed: {dp['n']:>4} trades ({len(deploy)} tickers), "
              f"avg={dp['avg']:>+.2%}, total={dp['total']:>+.1%}")
        if ca and ca['n_days'] > 0:
            print(f"    Cap-adj:  {ca['daily_avg']:>+.2%}/day, {ca['n_days']} days, "
                  f"WR={ca['daily_wr']:.0%}, DD={ca['daily_dd']:>+.2%}, "
                  f"H2={ca['h2_daily_avg']:>+.2%}/day")
        print(f"    Tickers:  {', '.join(sorted(deploy))}")
    else:
        print(f"    Deployed: none above {DEPLOY_FLOOR:.1%}")

# ADDITIVE value verdict
print(f"\n  ADDITIVE VALUE ANALYSIS (F403+F425 = deployed ADDITIVE_403):")
if len(s3_daily) > 0 and len(additive_daily) > 0:
    hedged_daily_series = pd.Series(dtype=float)
    for d in (additive_dates | s3_dates):
        dt = [t for t in hedged_trades if t['date'] == d]
        if dt:
            hedged_daily_series[d] = np.mean([t['pnl'] for t in dt])
    print(f"    S3 alone:      {len(s3_dates)} active days, "
          f"daily avg={s3_daily.mean():+.2%}, total={s3_daily.sum():+.1%}")
    print(f"    ADDITIVE only: {len(additive_dates)} active days, "
          f"daily avg={additive_daily.mean():+.2%}, total={additive_daily.sum():+.1%}")
    print(f"    S3+ADDITIVE:   {len(hedged_daily_series)} active days, "
          f"daily avg={hedged_daily_series.mean():+.2%}, total={hedged_daily_series.sum():+.1%}")
    extra_days = len(hedged_daily_series) - len(s3_dates)
    print(f"    Extra days from ADDITIVE: {extra_days} (+{extra_days/max(len(s3_dates),1)*100:.0f}% coverage)")
    add_only_sum = additive_daily[additive_daily.index.isin(additive_dates - s3_dates)].sum()
    print(f"    PnL from ADDITIVE-only days: {add_only_sum:+.1%}")
    # Per-leg contribution
    for leg_name, leg_daily, leg_dates in [('F403', f403_daily, f403_dates), ('F425', f425_daily, f425_dates)]:
        if len(leg_daily) > 0:
            solo = leg_dates - s3_dates
            solo_sum = leg_daily[leg_daily.index.isin(solo)].sum() if solo else 0
            print(f"    {leg_name} standalone: {len(leg_dates)} days, "
                  f"daily avg={leg_daily.mean():+.2%}, non-S3 PnL={solo_sum:+.1%}")

# Best strategy summary
best_avg = max(STRAT_ORDER, key=lambda s: strategies[s]['summary']['avg'] if strategies[s]['summary']['n'] > 0 else -999)
best_sharpe = max(STRAT_ORDER, key=lambda s: strategies[s]['summary']['sharpe'] if strategies[s]['summary']['n'] > 0 else -999)
print(f"\n  BEST BY AVG PnL:  {best_avg} ({strategies[best_avg]['summary']['avg']:+.2%})")
print(f"  BEST BY SHARPE:   {best_sharpe} ({strategies[best_sharpe]['summary']['sharpe']:.1f})")

print(f"\n  Deploy threshold:  {DEPLOY_FLOOR:.1%}")
print(f"  Slippage:          {SLIPPAGE_PCT:.1%} one-way ({SLIPPAGE_RT:.2%} RT)")
print(f"  Total runtime:     {time.time()-t_total:.0f}s")
print(f"{'='*W}")
