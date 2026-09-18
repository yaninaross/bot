# %% ── F403 + F425 + F435 SHORT Combined Backtest ─────────────────────────────
#
# Five SHORT strategies compared head-to-head:
#   1. F403_ONLY   — SHORT 9:30 (entry min 375), 2.5% dump
#   2. F425_ONLY   — SHORT 10AM (entry min 405), 2.5% dump
#   3. F435_ONLY   — SHORT 10AM (entry min 405), 8% dump
#   4. ADDITIVE    — F403 + F425 combined (deployed ADDITIVE_403)
#   5. ALL_SHORT   — F403 + F425 + F435 all three legs
#
# All legs: live model inference from GCS-stored weights (transformer encoder
#   + embedding + MLP). Same architecture/features/scaling as training.
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

# ── Tickers to evaluate ──
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
# 6. SCORING
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
# 8. BQ CONFIG LOADING
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
                'entry_minute', 'val_a_avg_pnl', 'val_a_count']:
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
            'val_a_count': int(row['val_a_count']),
        }
    return configs


# ==============================================================================
# 9. RUN SHORT LEG
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

    print(f"    Loading scaling stats: {leg_config['scaling_file']}")
    try:
        stats = load_scaling_stats(leg_config['scaling_file'])
        print(f"    Scaling stats: {len(stats)} features")
    except Exception as e:
        print(f"    ERROR loading scaling stats: {e}")
        return [], {}

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

        if cfg['tpsl_mode'] == 'atr_sl':
            atr_pct = compute_daily_atr_pct(df)
            s_pst = pd.to_datetime(TEST_START).tz_localize(PST)
            e_pst = pd.to_datetime(TEST_END).tz_localize(PST) + timedelta(days=1)
            idx_c = df.index.tz_convert(PST) if str(df.index.tz) != 'America/Los_Angeles' else df.index
            atr_cut = atr_pct[(idx_c >= s_pst) & (idx_c < e_pst)]
            ohlcv['atr_pct'] = atr_cut.values.astype(np.float64)

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
print(f"  F403 + F425 + F435 SHORT COMBINED BACKTEST")
print(f"  TEST: {TEST_START} -> {TEST_END}  |  H2: {HALF2_START}+")
print(f"{'='*W}")
print(f"  LEG 1: {LEG_F403['label']} -- {len(ASSET_COLS_930)} asset + {len(CONTEXT_COLS_930)} context features")
print(f"  LEG 2: {LEG_F425['label']} -- {len(ASSET_COLS_10AM)} asset + {len(CONTEXT_COLS_10AM)} context features")
print(f"  LEG 3: {LEG_F435['label']} -- {len(ASSET_COLS_10AM)} asset + {len(CONTEXT_COLS_10AM)} context features")
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


# ==============================================================================
# BUILD STRATEGIES
# ==============================================================================
print(f"\n  Building 5 strategies...")

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

port_403 = summarize_trades(trades_403, 'F403_ONLY')
port_425 = summarize_trades(trades_425, 'F425_ONLY')
port_435 = summarize_trades(trades_435, 'F435_ONLY')
port_add = summarize_trades(additive_trades, 'ADDITIVE')
port_all = summarize_trades(all_short_trades, 'ALL_SHORT')

strategies = {
    'F403_ONLY': {'summary': port_403, 'per_ticker': per_tk_403,       'trades': trades_403},
    'F425_ONLY': {'summary': port_425, 'per_ticker': per_tk_425,       'trades': trades_425},
    'F435_ONLY': {'summary': port_435, 'per_ticker': per_tk_435,       'trades': trades_435},
    'ADDITIVE':  {'summary': port_add, 'per_ticker': additive_per_tk,  'trades': additive_trades},
    'ALL_SHORT': {'summary': port_all, 'per_ticker': all_short_per_tk, 'trades': all_short_trades},
}
STRAT_ORDER = ['F403_ONLY', 'F425_ONLY', 'F435_ONLY', 'ADDITIVE', 'ALL_SHORT']

elapsed = time.time() - t_total
print(f"\n  Total scoring time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


# ==============================================================================
# TABLE 1: HEAD-TO-HEAD
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: HEAD-TO-HEAD COMPARISON ({TEST_START} -> {TEST_END})")
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
# TABLE 2: PER-TICKER DETAIL (all legs + val_a info)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 2: PER-TICKER DETAIL")
print(f"{'='*W}")

all_short_tickers = sorted(set(
    list(per_tk_403.keys()) + list(per_tk_425.keys()) + list(per_tk_435.keys())))

print(f"\n  {'Ticker':<7} {'F403':>10} {'va#':>4}  {'F425':>10} {'va#':>4}  "
      f"{'F435':>10} {'va#':>4}")
print(f"  {'─'*55}")

for ticker in all_short_tickers:
    parts = []
    for per_tk, cfgs in [(per_tk_403, cfgs_403), (per_tk_425, cfgs_425), (per_tk_435, cfgs_435)]:
        trades = per_tk.get(ticker, [])
        cfg = cfgs.get(ticker)
        if trades:
            avg = np.mean([t['pnl'] for t in trades])
            va_n = cfg['val_a_count'] if cfg else 0
            parts.append(f"{avg:>+5.2%} {len(trades):>3}t {va_n:>4}")
        else:
            parts.append(f"{'--':>10} {'--':>4}")
    print(f"  {ticker:<7} {'  '.join(parts)}")


# ==============================================================================
# PARAMETER SWEEP — score once, filter many ways
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PARAMETER SWEEP")
print(f"  Scored {len(trades_403)}+{len(trades_425)}+{len(trades_435)} trades across "
      f"{len(all_short_tickers)} tickers. Now testing filter combinations.")
print(f"{'='*W}")

SWEEP_LEGS = [
    ('F403',),
    ('F435',),
    ('F403', 'F435'),
    ('F403', 'F425'),
    ('F425', 'F435'),
    ('F403', 'F425', 'F435'),
]
SWEEP_VA_COUNT = [0, 3, 5, 8]
SWEEP_VA_PNL   = [-999, 0.0, 0.005]
SWEEP_DEPLOY   = [0.005, 0.01, 0.015, 0.02]

LEG_MAP = {
    'F403': (trades_403, per_tk_403, cfgs_403),
    'F425': (trades_425, per_tk_425, cfgs_425),
    'F435': (trades_435, per_tk_435, cfgs_435),
}

sweep_results = []

for legs in SWEEP_LEGS:
    for va_min_count in SWEEP_VA_COUNT:
        for va_min_pnl in SWEEP_VA_PNL:
            eligible_trades = []
            eligible_tickers = set()
            for leg in legs:
                all_leg_trades, _, leg_cfgs = LEG_MAP[leg]
                ok_tickers = {tk for tk, c in leg_cfgs.items()
                              if c['val_a_count'] >= va_min_count
                              and c['val_a_avg_pnl'] >= va_min_pnl}
                eligible_trades.extend([t for t in all_leg_trades
                                        if t['ticker'] in ok_tickers])
                eligible_tickers.update(ok_tickers)

            if not eligible_trades:
                continue

            tk_trades = defaultdict(list)
            for t in eligible_trades:
                tk_trades[t['ticker']].append(t)
            n_traded = len(tk_trades)

            for deploy_floor in SWEEP_DEPLOY:
                deployed = {tk: tl for tk, tl in tk_trades.items()
                            if np.mean([t['pnl'] for t in tl]) >= deploy_floor}

                if not deployed:
                    sweep_results.append({
                        'legs': '+'.join(legs), 'va_count': va_min_count,
                        'va_pnl': va_min_pnl, 'deploy': deploy_floor,
                        'n_eligible': len(eligible_tickers), 'n_traded': n_traded,
                        'n_deployed': 0, 'trades': 0, 'avg_pnl': 0,
                        'daily_avg': 0, 'daily_wr': 0, 'n_days': 0,
                        'h2_daily': 0, 'tickers': [],
                    })
                    continue

                deploy_trades = [t for tk in deployed for t in deployed[tk]]
                ca = compute_capital_adjusted(deploy_trades)

                sweep_results.append({
                    'legs': '+'.join(legs), 'va_count': va_min_count,
                    'va_pnl': va_min_pnl, 'deploy': deploy_floor,
                    'n_eligible': len(eligible_tickers), 'n_traded': n_traded,
                    'n_deployed': len(deployed), 'trades': len(deploy_trades),
                    'avg_pnl': np.mean([t['pnl'] for t in deploy_trades]),
                    'daily_avg': ca['daily_avg'], 'daily_wr': ca['daily_wr'],
                    'n_days': ca['n_days'], 'h2_daily': ca['h2_daily_avg'],
                    'tickers': sorted(deployed.keys()),
                })

valid_results = [r for r in sweep_results if r['n_deployed'] > 0 and r['n_days'] >= 5]
valid_results.sort(key=lambda r: r['daily_avg'], reverse=True)

print(f"\n  {'Rank':>4} {'Legs':<16} {'VA#>=':>5} {'VA%>=':>6} {'Floor':>6} "
      f"{'#Elig':>5} {'#Dep':>4} {'Trd':>5} {'Avg':>7} {'$/Day':>7} {'WR':>4} "
      f"{'Days':>4} {'H2$/D':>7}  Tickers")
print(f"  {'─'*120}")

for i, r in enumerate(valid_results[:30]):
    va_pnl_str = 'any' if r['va_pnl'] < -1 else f"{r['va_pnl']:.1%}"
    star = ' *' if r['daily_avg'] >= 0.02 else ''
    print(f"  {i+1:>4} {r['legs']:<16} {r['va_count']:>5} {va_pnl_str:>6} "
          f"{r['deploy']:>5.1%} {r['n_eligible']:>5} {r['n_deployed']:>4} "
          f"{r['trades']:>5} {r['avg_pnl']:>+6.2%} {r['daily_avg']:>+6.2%} "
          f"{r['daily_wr']:>3.0%} {r['n_days']:>4} {r['h2_daily']:>+6.2%}  "
          f"{', '.join(r['tickers'])}{star}")


# ── Best combo details ──
if valid_results:
    best = valid_results[0]
    va_pnl_label = 'any' if best['va_pnl'] < -1 else f"{best['va_pnl']:.1%}"
    print(f"\n  BEST COMBO: {best['legs']} | VA#>={best['va_count']} | "
          f"VA%>={va_pnl_label} | Floor={best['deploy']:.1%}")
    print(f"    {best['n_deployed']} tickers, {best['trades']} trades, "
          f"avg={best['avg_pnl']:+.2%}, cap-adj={best['daily_avg']:+.2%}/day, "
          f"WR={best['daily_wr']:.0%}, H2={best['h2_daily']:+.2%}/day")
    print(f"    Tickers: {', '.join(best['tickers'])}")

# ── Find combos with >=2 tickers and best daily avg ──
diverse = [r for r in valid_results if r['n_deployed'] >= 2]
if diverse:
    best_d = diverse[0]
    print(f"\n  BEST WITH >=2 TICKERS: {best_d['legs']} | VA#>={best_d['va_count']} | "
          f"Floor={best_d['deploy']:.1%}")
    print(f"    {best_d['n_deployed']} tickers, {best_d['trades']} trades, "
          f"avg={best_d['avg_pnl']:+.2%}, cap-adj={best_d['daily_avg']:+.2%}/day, "
          f"WR={best_d['daily_wr']:.0%}, H2={best_d['h2_daily']:+.2%}/day")
    print(f"    Tickers: {', '.join(best_d['tickers'])}")

stable = [r for r in valid_results if r['n_deployed'] >= 3 and r['h2_daily'] > 0]
if stable:
    best_s = stable[0]
    print(f"\n  BEST WITH >=3 TICKERS + H2>0: {best_s['legs']} | VA#>={best_s['va_count']} | "
          f"Floor={best_s['deploy']:.1%}")
    print(f"    {best_s['n_deployed']} tickers, {best_s['trades']} trades, "
          f"avg={best_s['avg_pnl']:+.2%}, cap-adj={best_s['daily_avg']:+.2%}/day, "
          f"WR={best_s['daily_wr']:.0%}, H2={best_s['h2_daily']:+.2%}/day")
    print(f"    Tickers: {', '.join(best_s['tickers'])}")


# ==============================================================================
# CHARTS (2x2)
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Chart [0,0]: Equity curves — raw legs
ax1 = axes[0, 0]
for sname in STRAT_ORDER:
    p = strategies[sname]['summary']
    if p['n'] == 0:
        continue
    df_t = p['trades_df'].sort_values('date')
    cum = df_t['pnl'].cumsum() * 100
    ax1.plot(pd.to_datetime(df_t['date'].values), cum.values,
             color=STRATEGY_COLORS[sname],
             linewidth=2.5 if sname in ('ADDITIVE', 'ALL_SHORT') else 1.5,
             label=f"{sname} ({p['n']}tr, {p['avg']:+.2%})", alpha=0.9)
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax1.axvline(x=pd.Timestamp(HALF2_START), color='red', linewidth=1,
            linestyle=':', alpha=0.5, label='H2 start')
ax1.set_ylabel('Cumulative PnL (%)')
ax1.set_title('All Trades (unfiltered)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Chart [0,1]: Per-ticker heatmap
ax2 = axes[0, 1]
if all_short_tickers:
    heatmap_data = np.full((len(all_short_tickers), 3), np.nan)
    for j, per_tk in enumerate([per_tk_403, per_tk_425, per_tk_435]):
        for i, ticker in enumerate(all_short_tickers):
            trades = per_tk.get(ticker, [])
            if trades:
                heatmap_data[i, j] = np.mean([t['pnl'] for t in trades]) * 100
    vmax = max(abs(np.nanmin(heatmap_data)), abs(np.nanmax(heatmap_data)))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1
    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['F403 (9:30)', 'F425 (10AM)', 'F435 (8%)'], fontsize=9)
    ax2.set_yticks(range(len(all_short_tickers)))
    ax2.set_yticklabels(all_short_tickers, fontsize=7)
    for i in range(len(all_short_tickers)):
        for j in range(3):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                         fontsize=6, fontweight='bold',
                         color='white' if abs(val) > vmax*0.5 else 'black')
    fig.colorbar(im, ax=ax2, label='Avg PnL (%)', shrink=0.8)
ax2.set_title('Per-Ticker Avg PnL by Leg', fontsize=12, fontweight='bold')

# Chart [1,0]: Top 15 sweep combos — daily avg bar chart
ax3 = axes[1, 0]
if valid_results:
    top_n = min(15, len(valid_results))
    top = valid_results[:top_n]
    y_pos = np.arange(top_n)
    daily_avgs = [r['daily_avg'] * 100 for r in top]
    labels = [f"{r['legs']} va#>={r['va_count']} f={r['deploy']:.1%}" for r in top]
    colors_bar = ['#2E7D32' if r['h2_daily'] > 0 else '#E65100' for r in top]
    ax3.barh(y_pos, daily_avgs, color=colors_bar, alpha=0.85, edgecolor='black', linewidth=0.3)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=7)
    ax3.set_xlabel('Cap-Adj Daily Avg PnL (%)')
    ax3.axvline(x=0, color='gray', linewidth=0.5)
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
ax3.set_title('Top 15 Sweep Combos (green=H2>0)', fontsize=12, fontweight='bold')

# Chart [1,1]: Best combo equity curve (cap-adj)
ax4 = axes[1, 1]
if valid_results:
    for rank, combo in enumerate(valid_results[:3]):
        combo_trades = []
        for leg in combo['legs'].split('+'):
            all_leg_trades, _, leg_cfgs = LEG_MAP[leg]
            ok_tickers = {tk for tk, c in leg_cfgs.items()
                          if c['val_a_count'] >= combo['va_count']
                          and c['val_a_avg_pnl'] >= combo['va_pnl']}
            combo_trades.extend([t for t in all_leg_trades if t['ticker'] in ok_tickers])
        tk_avg = defaultdict(list)
        for t in combo_trades:
            tk_avg[t['ticker']].append(t['pnl'])
        dep_tickers = {tk for tk, pnls in tk_avg.items()
                       if np.mean(pnls) >= combo['deploy']}
        dep_trades = sorted([t for t in combo_trades if t['ticker'] in dep_tickers],
                            key=lambda t: t['date'])
        if not dep_trades:
            continue
        df_eq = pd.DataFrame(dep_trades)
        daily = df_eq.groupby('date')['pnl'].mean()
        daily = daily.sort_index()
        cum = daily.cumsum() * 100
        lw = 3 if rank == 0 else 1.5
        al = 1.0 if rank == 0 else 0.6
        ax4.plot(pd.to_datetime(cum.index), cum.values, linewidth=lw, alpha=al,
                 label=f"#{rank+1} {combo['legs']} ({combo['n_deployed']}tk, "
                       f"{combo['daily_avg']:+.2%}/d)")
    ax4.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax4.axvline(x=pd.Timestamp(HALF2_START), color='red', linewidth=1,
                linestyle=':', alpha=0.5, label='H2')
    ax4.set_ylabel('Cumulative Cap-Adj PnL (%)')
    ax4.legend(fontsize=8, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax4.set_title('Top 3 Combos — Cap-Adj Equity', fontsize=12, fontweight='bold')

fig.suptitle(f'SHORT Parameter Sweep — F403 + F425 + F435\n'
             f'{TEST_START} -> {TEST_END}  |  Slippage: {SLIPPAGE_RT:.2%} RT  |  '
             f'{len(valid_results)} valid combos tested',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/tmp/backtest_short_combined.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Chart saved: /tmp/backtest_short_combined.png")


# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")
print(f"  Scored: F403={len(trades_403)} trades ({len(per_tk_403)} tickers), "
      f"F425={len(trades_425)} ({len(per_tk_425)}), F435={len(trades_435)} ({len(per_tk_435)})")
print(f"  Sweep: {len(sweep_results)} combos tested, {len(valid_results)} with >=1 deployed ticker")
print(f"  Slippage: {SLIPPAGE_RT:.2%} RT (matching WF optimizer)")
print(f"  Runtime: {time.time()-t_total:.0f}s")
print(f"{'='*W}")
