# %% ── SHORT Portfolio Backtest: Spike Transformer vs F403 ─────────────────────
#
# Head-to-head comparison of two SHORT strategies:
#
#   1. F403_DEPLOYED — BE/OKLO/SMR (live model scoring, BQ configs)
#      Entry: 6:15 AM (min 375), UNIVERSAL transformer encoder, 15-seed ensemble
#
#   2. ST_SHORT — 11 deployable tickers from ST optimizer
#      Score: 6:15 AM (min 375), Entry: 6:25 AM (min 385)
#      Non-Stationary Transformer, single model, per-ticker best config
#
#   3. COMBINED — union of both (different entry times allow coexistence)
#
# Overlap: OKLO and SMR appear in both strategies but at different entry times
#          (F403 at 6:15, ST at 6:25). Combined keeps both trades.
#
# GPU RECOMMENDED for F403 model scoring + ST model scoring. Colab-ready.

import os, io, gc, sys, json, time, ctypes, warnings, uuid
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Embedding, Flatten,
                                     Concatenate, MultiHeadAttention, Add,
                                     LayerNormalization, GlobalAveragePooling1D)
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.regularizers import l2

try:
    import pandas_ta as ta
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta",
                           "--no-deps", "--quiet"])
    import pandas_ta as ta

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
BUCKET_NAME   = "time-series-bucket-ys"
GCS_MODEL_DIR = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
INFERENCE_DIR = "Data/Inference_v2"

TEST_START  = '2026-04-01'
TEST_END    = '2026-08-08'
H2_START    = '2026-06-01'

SLIPPAGE_PCT  = 0.0015
SLIPPAGE_RT   = 2 * SLIPPAGE_PCT
EXIT_TIME_MIN = 780

TIME_STEPS     = 240
EMBED_DIM      = 16
SEEDS_PER_FOLD = 15

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

# ── F403 feature columns (9:30 variant — no Morning_Ext, Gap_VIX, VIX_Level) ──
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

# ── F403 leg config ──
LEG_F403 = {
    'name': 'F403_DEPLOYED',
    'label': 'F403 (BE/OKLO/SMR)',
    'exp_id': '403_XSECTOR',
    'seed_base': 40300,
    'model_version': 'robust_wf_v12_short_403',
    'entry_min': 375,
    'asset_cols': ASSET_COLS_930,
    'context_cols': CONTEXT_COLS_930,
    'all_cols': ALL_COLS_930,
    'scaling_file': f'{INFERENCE_DIR}/scaling_stats_experiment_403_XSECTOR.json',
}

F403_TICKERS = ['BE', 'OKLO', 'SMR']

# ── ST SHORT: 11 deployable tickers (best config per ticker from optimizer) ──
ST_SCORE_MINUTE = 375
ST_ENTRY_MINUTE = 385

ST_CONFIGS = {
    'IREN':  {'horizon': 0, 'prob_thresh': 0.60, 'tp_pct': 0.08, 'sl_pct': 0.04,
              'mode': 'static'},
    'WOLF':  {'horizon': 2, 'prob_thresh': 0.70, 'tp_pct': 0.12, 'sl_pct': 0.06,
              'mode': 'static'},
    'AEHR':  {'horizon': 2, 'prob_thresh': 0.50, 'tp_pct': 0.20, 'sl_pct': 0.10,
              'mode': 'limit_entry', 'dip_pct': 0.04, 'max_wait': 90},
    'SOUN':  {'horizon': 0, 'prob_thresh': 0.60, 'tp_pct': 0.08, 'sl_pct': 0.04,
              'mode': 'grace', 'grace_minutes': 15},
    'OKLO':  {'horizon': 0, 'prob_thresh': 0.60, 'tp_pct': 0.12, 'sl_pct': 0.06,
              'mode': 'static'},
    'MVIS':  {'horizon': 2, 'prob_thresh': 0.60, 'tp_pct': 0.16, 'sl_pct': 0.08,
              'mode': 'grace', 'grace_minutes': 15},
    'VOYG':  {'horizon': 1, 'prob_thresh': 0.40, 'tp_pct': 0.06, 'sl_pct': 0.03,
              'mode': 'static'},
    'AVAV':  {'horizon': 0, 'prob_thresh': 0.30, 'tp_pct': 0.08, 'sl_pct': 0.04,
              'mode': 'static'},
    'CCJ':   {'horizon': 0, 'prob_thresh': 0.20, 'tp_pct': 0.08, 'sl_pct': 0.04,
              'mode': 'limit_entry', 'dip_pct': 0.01, 'max_wait': 30},
    'JOBY':  {'horizon': 0, 'prob_thresh': 0.60, 'tp_pct': 0.16, 'sl_pct': 0.08,
              'mode': 'limit_entry', 'dip_pct': 0.025, 'max_wait': 60},
    'SMR':   {'horizon': 0, 'prob_thresh': 0.50, 'tp_pct': 0.16, 'sl_pct': 0.08,
              'mode': 'limit_entry', 'dip_pct': 0.025, 'max_wait': 60},
}

ST_TICKERS = sorted(ST_CONFIGS.keys())

# ── ST model architecture (must match training) ──
ST_D_MODEL = 64; ST_N_HEADS = 4; ST_D_FF = 128; ST_N_BLOCKS = 2
ST_DROPOUT = 0.1; ST_FILM_HIDDEN = 64; ST_TIME_STEPS = 128
ST_N_HORIZONS = 3
ST_ASL_GAMMA_NEG = 6.0; ST_ASL_GAMMA_POS = 0.1; ST_ASL_CLIP = 0.08

GCS_ST_DATA   = 'Algos/RA_MoE/short_data'
GCS_ST_MODELS = 'Algos/RA_MoE/short_models'
ST_FOLD_ID    = 4
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

STRATEGY_COLORS = {
    'F403_DEPLOYED': '#1565C0',
    'ST_SHORT':      '#E65100',
    'COMBINED':      '#2E7D32',
}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
np.random.seed(42)
W = 100


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


# ==============================================================================
# 3. F403 MODEL ARCHITECTURE + LOADING
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


def build_f403_model(n_asset_features, n_context_features, n_assets=31):
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
    return KerasModel(inputs=[in_a, in_c, in_id], outputs=out)


def load_f403_models(leg_config):
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
            model = build_f403_model(n_ac, n_cc, n_assets=n_assets)
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.load_weights(local_path)
            models.append(model)
        except Exception as e:
            print(f"    Seed {seed}: FAILED -- {e}")
    return models


# ==============================================================================
# 4. F403 FEATURE ENGINEERING + SCALING + SCORING
# ==============================================================================
def build_market_context():
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker)
        if df is None:
            raise RuntimeError(f"Cannot load context asset {ticker}")
        df = df.sort_index()
        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
    _master_cache.clear()
    return pd.concat(frames, axis=1).ffill().fillna(0)


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
# 5. F403 TRADE SIMS (per-trade, Python)
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


def execute_f403_trade(ohlcv, entry_idx, mode, tp_pct, sl_pct, sl_mult, grace_min):
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
# 6. F403 BQ CONFIGS + RUN LEG
# ==============================================================================
def load_bq_configs_short(model_version, tickers):
    allowed_modes = ('static', 'atr_sl', 'grace')
    q = f"""
    SELECT ticker, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, sl_mult, grace_minutes,
           opt_avg_pnl, val_a_avg_pnl, val_a_win_rate, val_a_count,
           is_ensemble, model_version
    FROM `{BQ_TABLE}`
    WHERE model_version = '{model_version}'
      AND opt_avg_pnl > 0
      AND tpsl_mode IN ({", ".join(f"'{m}'" for m in allowed_modes)})
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
            'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else 375,
            'val_a_avg_pnl': float(row['val_a_avg_pnl']),
            'val_a_count': int(row['val_a_count']),
        }
    return configs


def run_f403_leg(leg_config, bq_configs, market_df, tickers):
    entry_min = leg_config['entry_min']
    asset_cols = leg_config['asset_cols']
    context_cols = leg_config['context_cols']
    all_cols = leg_config['all_cols']

    print(f"\n  Loading scaling stats...")
    stats = load_scaling_stats(leg_config['scaling_file'])
    print(f"    Scaling stats: {len(stats)} features")

    print(f"    Loading {SEEDS_PER_FOLD} F403 models...")
    t0 = time.time()
    models = load_f403_models(leg_config)
    print(f"    Models loaded: {len(models)}/{SEEDS_PER_FOLD} ({time.time()-t0:.0f}s)")
    if not models:
        return [], {}

    tickers_with_config = [t for t in tickers if t in bq_configs]
    print(f"    Tickers with BQ configs: {len(tickers_with_config)}")

    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt   = pd.Timestamp(TEST_END).date()
    h2_start_dt   = pd.Timestamp(H2_START).date()
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
        df = engineer_features_930(df, market_df)
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
        entry_mask = (mins == entry_min)
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

            pnl, trade_date, filled = execute_f403_trade(
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
                'mode': cfg['tpsl_mode'], 'tp': cfg['tp_pct'], 'sl': cfg['sl_pct'],
                'half': half, 'strategy': 'F403_DEPLOYED', 'direction': 'SHORT',
                'entry_min': entry_min,
            }
            all_trades.append(trade)
            per_ticker[ticker].append(trade)

        _master_cache.pop(ticker, None)
        n_trades = len(per_ticker.get(ticker, []))
        if n_trades > 0:
            avg = np.mean([t['pnl'] for t in per_ticker[ticker]])
            print(f"    {ticker}: {len(daily_probs)} scored, {n_trades} trades, avg={avg:+.2%}")
        _release_memory()

    del models
    tf.keras.backend.clear_session()
    _release_memory()

    print(f"\n  F403: {len(all_trades)} trades across {len(per_ticker)} tickers")
    return all_trades, dict(per_ticker)


# ==============================================================================
# 7. SPIKE TRANSFORMER MODEL (must match training class names exactly)
# ==============================================================================

class SeriesStationarization(layers.Layer):
    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        std  = tf.math.reduce_std(x, axis=1, keepdims=True) + 1e-5
        x_norm = (x - mean) / std
        stats = tf.concat([mean[:, 0, :], std[:, 0, :]], axis=-1)
        return x_norm, stats


class DeStationaryAttention(layers.Layer):
    def __init__(self, d_model, n_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model; self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = tf.sqrt(tf.cast(self.d_head, tf.float32))
        self.wq = layers.Dense(d_model); self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model); self.wo = layers.Dense(d_model)
        self.attn_drop = layers.Dropout(dropout)
        self.tau_proj = layers.Dense(n_heads)
        self.sigma_proj = layers.Dense(d_model)

    def call(self, x, stats, training=False):
        B = tf.shape(x)[0]; T = tf.shape(x)[1]
        H, D = self.n_heads, self.d_head
        Q = tf.transpose(tf.reshape(self.wq(x), (B, T, H, D)), (0, 2, 1, 3))
        K = tf.transpose(tf.reshape(self.wk(x), (B, T, H, D)), (0, 2, 1, 3))
        V = tf.transpose(tf.reshape(self.wv(x), (B, T, H, D)), (0, 2, 1, 3))
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale
        tau = tf.reshape(self.tau_proj(stats), (B, H, 1, 1))
        scores = scores + tau
        attn = self.attn_drop(tf.nn.softmax(scores, axis=-1), training=training)
        out = tf.matmul(attn, V)
        out = tf.reshape(tf.transpose(out, (0, 2, 1, 3)), (B, T, self.d_model))
        sigma = tf.nn.softplus(self.sigma_proj(stats))
        out = out * sigma[:, tf.newaxis, :]
        return self.wo(out)


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = DeStationaryAttention(d_model, n_heads, dropout)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation='gelu'),
            layers.Dense(d_model),
            layers.Dropout(dropout)])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop = layers.Dropout(dropout)

    def call(self, x, stats, training=False):
        x = x + self.drop(
            self.attn(self.norm1(x), stats, training=training), training=training)
        x = x + self.ffn(self.norm2(x), training=training)
        return x


class TransformerBackbone(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len,
                 dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stationarize = SeriesStationarization()
        self.input_proj = layers.Dense(d_model)
        self.pos_embed = self.add_weight(
            name='pos_embed', shape=(1, seq_len, d_model),
            initializer='glorot_normal', trainable=True)
        self.input_drop = layers.Dropout(dropout)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout,
                                        name=f'block_{i}')
                       for i in range(n_blocks)]
        self.final_norm = layers.LayerNormalization()

    def call(self, x, training=False):
        x_norm, stats = self.stationarize(x)
        h = self.input_proj(x_norm) + self.pos_embed
        h = self.input_drop(h, training=training)
        for block in self.blocks:
            h = block(h, stats, training=training)
        return self.final_norm(h)


class AttentionPooling(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.query = self.add_weight(
            name='pool_query', shape=(1, 1, d_model),
            initializer='glorot_normal', trainable=True)
        self.scale = tf.sqrt(tf.cast(d_model, tf.float32))

    def call(self, x, training=False):
        scores = tf.matmul(self.query, x, transpose_b=True) / self.scale
        weights = tf.nn.softmax(scores, axis=-1)
        pooled = tf.matmul(weights, x)
        return pooled[:, 0, :]


class FiLMConditioner(layers.Layer):
    def __init__(self, d_model, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu'),
        ])
        self.gamma_proj = layers.Dense(d_model)
        self.beta_proj = layers.Dense(d_model)

    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        gamma = 1.0 + self.gamma_proj(h)
        beta = self.beta_proj(h)
        return gamma * x + beta


class SpikeTransformer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = TransformerBackbone(
            ST_D_MODEL, ST_N_HEADS, ST_D_FF, ST_N_BLOCKS,
            ST_TIME_STEPS, ST_DROPOUT, name='backbone')
        self.attn_pool = AttentionPooling(ST_D_MODEL, name='attn_pool')
        self.film = FiLMConditioner(ST_D_MODEL, ST_FILM_HIDDEN, name='film')
        self.drop = layers.Dropout(ST_DROPOUT)
        self.head = layers.Dense(ST_N_HORIZONS, activation='sigmoid', name='spike_head')

    def call(self, inputs, training=False):
        seq = inputs['seq_input']
        regime = inputs['regime_input']
        h = self.backbone(seq, training=training)
        h = self.attn_pool(h, training=training)
        h = self.film(h, regime, training=training)
        h = self.drop(h, training=training)
        return self.head(h)

    def train_step(self, data):
        x, y, sw = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            preds = self(x, training=True)
            loss = st_asl_loss(y, preds, sw)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        x, y, sw = tf.keras.utils.unpack_x_y_sample_weight(data)
        preds = self(x, training=False)
        return {'loss': st_asl_loss(y, preds, sw)}


def st_asl_loss(y_true, y_pred, sample_weight=None):
    p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    pos_loss = -y_true * tf.pow(1.0 - p, ST_ASL_GAMMA_POS) * tf.math.log(p)
    p_m = tf.maximum(p - ST_ASL_CLIP, 0.0)
    neg_loss = -(1.0 - y_true) * tf.pow(p_m, ST_ASL_GAMMA_NEG) * tf.math.log(
        tf.clip_by_value(1.0 - p_m, 1e-7, 1.0))
    per_sample = tf.reduce_mean(pos_loss + neg_loss, axis=-1)
    if sample_weight is not None:
        per_sample = per_sample * sample_weight
    return tf.reduce_mean(per_sample)


# ==============================================================================
# 8. ST SCORING — load model, score npz, build daily scores
# ==============================================================================
ST_DATA_DIR  = '/tmp/ra_moe_short_data'
ST_MODEL_DIR = '/tmp/ra_moe_short_models'
os.makedirs(ST_DATA_DIR, exist_ok=True)
os.makedirs(ST_MODEL_DIR, exist_ok=True)


def _ensure_st_data(filename):
    local_path = os.path.join(ST_DATA_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    gcs_path = f'{GCS_ST_DATA}/{filename}'
    print(f"    Downloading {filename} from GCS...")
    gcs_client.bucket(BUCKET_NAME).blob(gcs_path).download_to_filename(local_path)
    return local_path


def _ensure_st_model(fold_id):
    local_dir = os.path.join(ST_MODEL_DIR, f'fold_{fold_id}')
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if os.path.exists(weights_path):
        return weights_path
    os.makedirs(local_dir, exist_ok=True)
    bucket = gcs_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=f'{GCS_ST_MODELS}/fold_{fold_id}'))
    for blob in blobs:
        fname = blob.name.split('/')[-1]
        blob.download_to_filename(os.path.join(local_dir, fname))
    return weights_path


def score_st_npz(model, split, ticker_names):
    path = _ensure_st_data(f'fold_{ST_FOLD_ID}_{split}.npz')
    data = np.load(path)
    X_seq = data['X_seq']; X_reg = data['X_regime']
    X_id = data['X_id'].flatten()
    date_ints = data['date_ints']; minutes = data['minutes']
    data.close()

    entry_mask = (minutes == ST_SCORE_MINUTE)
    X_seq = X_seq[entry_mask]; X_reg = X_reg[entry_mask]
    X_id = X_id[entry_mask]; date_ints = date_ints[entry_mask]
    n = len(X_id)
    print(f"    {split}: {n} score-minute samples")

    preds = model.predict(
        {'seq_input': X_seq.astype(np.float32),
         'regime_input': X_reg.astype(np.float32)},
        batch_size=512, verbose=0)
    del X_seq, X_reg; gc.collect()

    ticker_scores = {}
    for i in range(n):
        tid = int(X_id[i])
        d_int = int(date_ints[i])
        if tid >= len(ticker_names):
            continue
        tname = str(ticker_names[tid])
        if tname in EXCLUDE_TICKERS:
            continue
        if tname not in ticker_scores:
            ticker_scores[tname] = {h: {} for h in range(ST_N_HORIZONS)}
        for h in range(ST_N_HORIZONS):
            ticker_scores[tname][h][d_int] = float(preds[i, h])
    del preds, X_id, date_ints; gc.collect()
    return ticker_scores


def run_st_scoring():
    print(f"\n  Loading ST SHORT model...")
    meta_path = _ensure_st_data('meta.npz')
    meta = np.load(meta_path, allow_pickle=True)
    ticker_names = meta['ticker_names']

    probe_path = _ensure_st_data(f'fold_{ST_FOLD_ID}_val.npz')
    probe = np.load(probe_path)
    n_af = probe['X_seq'].shape[-1]
    n_rf = probe['X_regime'].shape[-1]
    probe.close()

    model = SpikeTransformer(name='spike_transformer')
    dummy_seq = tf.zeros((1, ST_TIME_STEPS, n_af))
    dummy_reg = tf.zeros((1, n_rf))
    model({'seq_input': dummy_seq, 'regime_input': dummy_reg})

    weights_path = _ensure_st_model(ST_FOLD_ID)
    model.load_weights(weights_path)
    print(f"    ST model loaded: {model.count_params():,} params")

    val_scores = score_st_npz(model, 'val', ticker_names)
    test_scores = score_st_npz(model, 'test', ticker_names)

    all_scores = {}
    for tname in set(val_scores.keys()) | set(test_scores.keys()):
        all_scores[tname] = {}
        for h in range(ST_N_HORIZONS):
            merged = {}
            if tname in val_scores and h in val_scores[tname]:
                merged.update(val_scores[tname][h])
            if tname in test_scores and h in test_scores[tname]:
                merged.update(test_scores[tname][h])
            all_scores[tname][h] = merged

    del model, val_scores, test_scores
    tf.keras.backend.clear_session(); gc.collect()
    return all_scores


# ==============================================================================
# 9. ST TRADE SIMS (Numba) — used for static, grace, limit_entry
# ==============================================================================

@jit(nopython=True)
def st_trade_sim_static(entry_indices, prices, highs, lows, date_ints, time_mins,
                        tp, sl, exit_time):
    n_entries = len(entry_indices); n_prices = len(prices)
    trades = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)
    tc = 0; ptr = 0; last_date = -1
    while ptr < n_entries:
        idx = entry_indices[ptr]
        if idx >= n_prices - 1: break
        ed = date_ints[idx]
        if ed == last_date: ptr += 1; continue
        ep = prices[idx]; tp_px = ep * (1 - tp); sl_px = ep * (1 + sl)
        outcome = 0.0; done = False; exit_idx = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != ed:
                outcome = (ep - prices[k-1]) / ep; exit_idx = k; done = True; break
            if highs[k] >= sl_px and lows[k] <= tp_px:
                outcome = -sl if np.random.random() < 0.5 else tp; exit_idx = k; done = True; break
            if highs[k] >= sl_px: outcome = -sl; exit_idx = k; done = True; break
            if lows[k] <= tp_px: outcome = tp; exit_idx = k; done = True; break
            if time_mins[k] >= exit_time:
                outcome = (ep - prices[k]) / ep; exit_idx = k; done = True; break
        if not done: outcome = (ep - prices[n_prices-1]) / ep; exit_idx = n_prices
        trades[tc] = outcome; entry_dates[tc] = ed; tc += 1; last_date = ed
        nxt = ptr + 1
        while nxt < n_entries:
            if entry_indices[nxt] > exit_idx: break
            nxt += 1
        ptr = nxt
    return trades[:tc], entry_dates[:tc]


@jit(nopython=True)
def st_trade_sim_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
                       tp, sl, exit_time, grace_minutes):
    n_entries = len(entry_indices); n_prices = len(prices)
    trades = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)
    tc = 0; ptr = 0; last_date = -1
    while ptr < n_entries:
        idx = entry_indices[ptr]
        if idx >= n_prices - 1: break
        ed = date_ints[idx]
        if ed == last_date: ptr += 1; continue
        ep = prices[idx]; tp_px = ep * (1 - tp); sl_px = ep * (1 + sl)
        et = time_mins[idx]; outcome = 0.0; done = False; exit_idx = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != ed:
                outcome = (ep - prices[k-1]) / ep; exit_idx = k; done = True; break
            elapsed = time_mins[k] - et; sl_on = elapsed >= grace_minutes
            tp_hit = lows[k] <= tp_px; sl_hit = sl_on and (highs[k] >= sl_px)
            if sl_hit and tp_hit:
                outcome = -sl if np.random.random() < 0.5 else tp; exit_idx = k; done = True; break
            if sl_hit: outcome = -sl; exit_idx = k; done = True; break
            if tp_hit: outcome = tp; exit_idx = k; done = True; break
            if time_mins[k] >= exit_time:
                outcome = (ep - prices[k]) / ep; exit_idx = k; done = True; break
        if not done: outcome = (ep - prices[n_prices-1]) / ep; exit_idx = n_prices
        trades[tc] = outcome; entry_dates[tc] = ed; tc += 1; last_date = ed
        nxt = ptr + 1
        while nxt < n_entries:
            if entry_indices[nxt] > exit_idx: break
            nxt += 1
        ptr = nxt
    return trades[:tc], entry_dates[:tc]


@jit(nopython=True)
def st_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints, time_mins,
                             tp, sl, exit_time, rally_pct, max_wait_minutes):
    n_entries = len(entry_indices); n_prices = len(prices)
    trades = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)
    tc = 0; ptr = 0; last_date = -1
    while ptr < n_entries:
        idx = entry_indices[ptr]
        if idx >= n_prices - 1: break
        ed = date_ints[idx]
        if ed == last_date: ptr += 1; continue
        sig_px = prices[idx]; limit_px = sig_px * (1.0 + rally_pct)
        sig_t = time_mins[idx]
        fill_idx = -1; fill_px = 0.0
        for k in range(idx + 1, n_prices):
            if date_ints[k] != ed: break
            if time_mins[k] - sig_t > max_wait_minutes: break
            if highs[k] >= limit_px: fill_px = limit_px; fill_idx = k; break
        if fill_idx < 0: ptr += 1; last_date = ed; continue
        tp_px = fill_px * (1 - tp); sl_px = fill_px * (1 + sl)
        outcome = 0.0; done = False; exit_idx = fill_idx
        for k in range(fill_idx + 1, n_prices):
            if date_ints[k] != ed:
                outcome = (fill_px - prices[k-1]) / fill_px; exit_idx = k; done = True; break
            if highs[k] >= sl_px and lows[k] <= tp_px:
                outcome = -sl if np.random.random() < 0.5 else tp; exit_idx = k; done = True; break
            if highs[k] >= sl_px: outcome = -sl; exit_idx = k; done = True; break
            if lows[k] <= tp_px: outcome = tp; exit_idx = k; done = True; break
            if time_mins[k] >= exit_time:
                outcome = (fill_px - prices[k]) / fill_px; exit_idx = k; done = True; break
        if not done: outcome = (fill_px - prices[n_prices-1]) / fill_px; exit_idx = n_prices
        trades[tc] = outcome; entry_dates[tc] = ed; tc += 1; last_date = ed
        nxt = ptr + 1
        while nxt < n_entries:
            if entry_indices[nxt] > exit_idx: break
            nxt += 1
        ptr = nxt
    return trades[:tc], entry_dates[:tc]


def _run_st_sim(indices, ohlcv, cfg):
    prices = ohlcv['prices']; highs = ohlcv['highs']; lows = ohlcv['lows']
    dates = ohlcv['dates']; times = ohlcv['times'].astype(np.int64)
    mode = cfg['mode']
    if mode == 'grace':
        pnl, d = st_trade_sim_grace(indices, prices, highs, lows, dates, times,
                                     cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
                                     cfg['grace_minutes'])
    elif mode == 'limit_entry':
        pnl, d = st_trade_sim_limit_entry(indices, prices, highs, lows, dates, times,
                                           cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
                                           cfg['dip_pct'], cfg['max_wait'])
    else:
        pnl, d = st_trade_sim_static(indices, prices, highs, lows, dates, times,
                                      cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)
    if len(pnl) > 0:
        pnl = pnl - SLIPPAGE_RT
    return pnl, d


# ==============================================================================
# 10. RUN ST STRATEGY
# ==============================================================================
def run_st_strategy(all_daily_scores):
    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt   = pd.Timestamp(TEST_END).date()
    h2_start_dt   = pd.Timestamp(H2_START).date()

    all_trades = []
    per_ticker = defaultdict(list)

    for ticker in sorted(ST_TICKERS):
        cfg = ST_CONFIGS[ticker]
        daily_scores = all_daily_scores.get(ticker, {})
        if not daily_scores:
            print(f"    {ticker}: no ST scores")
            continue

        h_idx = cfg['horizon']
        h_scores = daily_scores.get(h_idx, {})
        if not h_scores:
            print(f"    {ticker}: no scores for horizon {h_idx}")
            continue

        try:
            df_ohlcv = load_master(ticker)
            if df_ohlcv is None:
                continue
        except Exception as e:
            print(f"    {ticker}: OHLCV load failed -- {e}")
            continue

        ohlcv = precompute_ohlcv_arrays(df_ohlcv, TEST_START, TEST_END)
        if ohlcv is None:
            _master_cache.pop(ticker, None)
            continue

        dates_arr = ohlcv['dates']; times_arr = ohlcv['times']

        score_by_date = {int(d): s for d, s in h_scores.items()}

        entry_list = []
        for i in range(len(dates_arr)):
            if times_arr[i] == ST_ENTRY_MINUTE:
                day_score = score_by_date.get(int(dates_arr[i]), 0.0)
                if day_score >= cfg['prob_thresh']:
                    entry_list.append(i)
        if not entry_list:
            _master_cache.pop(ticker, None)
            continue
        indices = np.array(entry_list, dtype=np.int64)

        raw_pnl, raw_dates = _run_st_sim(indices, ohlcv, cfg)

        for j in range(len(raw_pnl)):
            d_int = int(raw_dates[j])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            half = 'H2' if d_date >= h2_start_dt else 'H1'
            trade = {
                'date': d_date, 'ticker': ticker, 'pnl': float(raw_pnl[j]),
                'prob': score_by_date.get(d_int, 0.0),
                'close': 0.0, 'mode': cfg['mode'],
                'tp': cfg['tp_pct'], 'sl': cfg['sl_pct'],
                'half': half, 'strategy': 'ST_SHORT', 'direction': 'SHORT',
                'entry_min': ST_ENTRY_MINUTE,
            }
            all_trades.append(trade)
            per_ticker[ticker].append(trade)

        _master_cache.pop(ticker, None)
        n_trades = len(per_ticker.get(ticker, []))
        if n_trades > 0:
            avg = np.mean([t['pnl'] for t in per_ticker[ticker]])
            print(f"    {ticker}: {len(h_scores)} days scored, {n_trades} trades, avg={avg:+.2%}")
        _release_memory()

    print(f"\n  ST SHORT: {len(all_trades)} trades across {len(per_ticker)} tickers")
    return all_trades, dict(per_ticker)


# ==============================================================================
# 11. PORTFOLIO ANALYSIS
# ==============================================================================
def summarize_trades(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0, 'sharpe': 0, 'trades_df': pd.DataFrame(),
                'h1_n': 0, 'h1_avg': 0, 'h2_n': 0, 'h2_avg': 0,
                'h2_wr': 0, 'monthly': {}, 'n_tickers': 0}
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
        'trades_df': df, 'n_tickers': df['ticker'].nunique(),
        'h1_n': len(h1), 'h1_avg': h1['pnl'].mean() if len(h1) > 0 else 0,
        'h2_n': len(h2), 'h2_avg': h2['pnl'].mean() if len(h2) > 0 else 0,
        'h2_wr': (h2['pnl'] > 0).mean() if len(h2) > 0 else 0,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
    }


def compute_capital_adjusted(trades_list, deploy_floor=None):
    if not trades_list:
        return {'n_days': 0, 'daily_avg': 0, 'daily_wr': 0, 'daily_dd': 0,
                'avg_positions': 0, 'daily_series': pd.Series(dtype=float),
                'h2_daily_avg': 0, 'h2_n_days': 0}
    df = pd.DataFrame(trades_list)
    if deploy_floor is not None:
        tk_avg = df.groupby('ticker')['pnl'].mean()
        good = tk_avg[tk_avg >= deploy_floor].index
        df = df[df['ticker'].isin(good)]
    if len(df) == 0:
        return {'n_days': 0, 'daily_avg': 0, 'daily_wr': 0, 'daily_dd': 0,
                'avg_positions': 0, 'daily_series': pd.Series(dtype=float),
                'h2_daily_avg': 0, 'h2_n_days': 0}
    daily = df.groupby('date').agg(day_pnl=('pnl', 'mean'), n_pos=('pnl', 'count'))
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    cum = daily['day_pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    h2_mask = daily.index >= pd.Timestamp(H2_START)
    h2_daily = daily[h2_mask]
    return {
        'n_days': len(daily),
        'daily_avg': daily['day_pnl'].mean(),
        'daily_wr': (daily['day_pnl'] > 0).mean(),
        'daily_dd': max_dd,
        'avg_positions': daily['n_pos'].mean(),
        'daily_series': daily['day_pnl'],
        'h2_daily_avg': h2_daily['day_pnl'].mean() if len(h2_daily) > 0 else 0,
        'h2_n_days': len(h2_daily),
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
t_total = time.time()

print(f"\n{'='*W}")
print(f"  SHORT PORTFOLIO BACKTEST: Spike Transformer vs F403")
print(f"  Period: {TEST_START} -> {TEST_END}  |  H2 (OOS for both): {H2_START}+")
print(f"  Slippage: {SLIPPAGE_PCT:.2%} one-way ({SLIPPAGE_RT:.2%} RT)")
print(f"  GPU: {tf.config.list_physical_devices('GPU')}")
print(f"{'='*W}")
print(f"\n  F403 DEPLOYED: {', '.join(F403_TICKERS)} (entry min {LEG_F403['entry_min']})")
print(f"  ST SHORT: {', '.join(ST_TICKERS)} (score min {ST_SCORE_MINUTE}, entry min {ST_ENTRY_MINUTE})")
overlap = sorted(set(F403_TICKERS) & set(ST_TICKERS))
if overlap:
    print(f"  OVERLAP: {', '.join(overlap)} (different entry times, both trades kept)")


# ── Step 1: Score ST SHORT ────────────────────────────────────────────────────
print(f"\n  STEP 1: Spike Transformer SHORT scoring...")
t0 = time.time()
np.random.seed(42)
st_daily_scores = run_st_scoring()
print(f"    ST scoring done ({time.time()-t0:.0f}s)")


# ── Step 2: Run ST SHORT trades ──────────────────────────────────────────────
print(f"\n  STEP 2: Running ST SHORT trades...")
np.random.seed(42)
trades_st, per_tk_st = run_st_strategy(st_daily_scores)


# ── Step 3: Load F403 configs from BQ ─────────────────────────────────────────
print(f"\n  STEP 3: Loading F403 BQ configs...")
cfgs_403 = load_bq_configs_short(LEG_F403['model_version'], F403_TICKERS)
for ticker in sorted(cfgs_403):
    c = cfgs_403[ticker]
    print(f"    {ticker}: {c['tpsl_mode']} prob>{c['prob_thresh']:.2f} "
          f"TP={c['tp_pct']*100:.0f}% SL={c['sl_pct']*100:.0f}% "
          f"val_a={c['val_a_avg_pnl']*100:+.2f}%")


# ── Step 4: Build market context + score F403 ─────────────────────────────────
print(f"\n  STEP 4: Building market context + scoring F403...")
t0 = time.time()
market_df = build_market_context()
_master_cache.clear()
print(f"    Market context: {market_df.shape} ({time.time()-t0:.1f}s)")

np.random.seed(42)
trades_403, per_tk_403 = run_f403_leg(LEG_F403, cfgs_403, market_df, F403_TICKERS)


# ── Step 5: Build combined strategy ──────────────────────────────────────────
combined_trades = trades_403 + trades_st
combined_per_tk = defaultdict(list)
for tk, tl in per_tk_403.items():
    combined_per_tk[f'{tk}_F403'].extend(tl)
for tk, tl in per_tk_st.items():
    combined_per_tk[f'{tk}_ST'].extend(tl)
combined_per_tk = dict(combined_per_tk)


port_403 = summarize_trades(trades_403, 'F403_DEPLOYED')
port_st  = summarize_trades(trades_st,  'ST_SHORT')
port_comb = summarize_trades(combined_trades, 'COMBINED')

cap_403 = compute_capital_adjusted(trades_403)
cap_st  = compute_capital_adjusted(trades_st)
cap_comb = compute_capital_adjusted(combined_trades)

elapsed = time.time() - t_total
print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


# ==============================================================================
# TABLE 1: HEAD-TO-HEAD
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: HEAD-TO-HEAD COMPARISON")
print(f"  Period: {TEST_START} -> {TEST_END}  |  H2 (OOS): {H2_START}+")
print(f"{'='*W}")

print(f"\n  {'Strategy':<16} {'#Tk':>4} {'Trades':>7} {'Avg PnL':>8} {'Total':>9} "
      f"{'WR':>5} {'MaxDD':>7} {'Sharpe':>7}  {'H2 Avg':>8} {'H2 N':>5}")
print(f"  {'─'*85}")

for p in [port_403, port_st, port_comb]:
    if p['n'] == 0:
        print(f"  {p['label']:<16} -- no trades")
        continue
    print(f"  {p['label']:<16} {p['n_tickers']:>4} {p['n']:>7} {p['avg']:>+7.2%} "
          f"{p['total']:>+8.1%} {p['wr']:>5.0%} {p['dd']:>+7.2%} {p['sharpe']:>6.1f}"
          f"  {p['h2_avg']:>+7.2%} {p['h2_n']:>5}")


# ==============================================================================
# TABLE 2: CAPITAL-ADJUSTED DAILY
# ==============================================================================
print(f"\n  {'Strategy':<16} {'Days':>5} {'Daily Avg':>10} {'Daily WR':>9} "
      f"{'DD':>8} {'Avg Pos':>8}  {'H2 Avg':>9} {'H2 Days':>8}")
print(f"  {'─'*80}")

for lbl, cap in [('F403_DEPLOYED', cap_403), ('ST_SHORT', cap_st), ('COMBINED', cap_comb)]:
    if cap['n_days'] == 0:
        print(f"  {lbl:<16} -- no trading days")
        continue
    print(f"  {lbl:<16} {cap['n_days']:>5} {cap['daily_avg']:>+9.2%} "
          f"{cap['daily_wr']:>8.0%} {cap['daily_dd']:>+8.2%} "
          f"{cap['avg_positions']:>7.1f}  {cap['h2_daily_avg']:>+8.2%} {cap['h2_n_days']:>8}")


# ==============================================================================
# TABLE 3: PER-TICKER DETAIL
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: PER-TICKER DETAIL (all trades, {TEST_START} -> {TEST_END})")
print(f"{'='*W}")

all_tickers = sorted(set(
    list(per_tk_403.keys()) + list(per_tk_st.keys())))

print(f"\n  {'Ticker':<7} {'Source':>8} {'N':>4} {'Avg PnL':>8} {'WR':>5} "
      f"{'H1 N':>5} {'H1 Avg':>8} {'H2 N':>5} {'H2 Avg':>8} {'Entry':>6}")
print(f"  {'─'*72}")

for ticker in all_tickers:
    for source, per_tk_dict, entry_label in [
        ('F403', per_tk_403, '6:15'),
        ('ST', per_tk_st, '6:25'),
    ]:
        trades = per_tk_dict.get(ticker, [])
        if not trades:
            continue
        pnls = [t['pnl'] for t in trades]
        h1 = [t['pnl'] for t in trades if t['half'] == 'H1']
        h2 = [t['pnl'] for t in trades if t['half'] == 'H2']
        print(f"  {ticker:<7} {source:>8} {len(pnls):>4} {np.mean(pnls):>+7.2%} "
              f"{np.mean(np.array(pnls) > 0):>5.0%}"
              f" {len(h1):>5} {np.mean(h1) if h1 else 0:>+7.2%}"
              f" {len(h2):>5} {np.mean(h2) if h2 else 0:>+7.2%}"
              f" {entry_label:>6}")


# ==============================================================================
# TABLE 4: MONTHLY BREAKDOWN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 4: MONTHLY BREAKDOWN")
print(f"{'='*W}")

for lbl, p in [('F403_DEPLOYED', port_403), ('ST_SHORT', port_st), ('COMBINED', port_comb)]:
    print(f"\n  {lbl}:")
    if not p['monthly']:
        print(f"    -- no trades")
        continue
    print(f"    {'Month':>7} {'N':>4} {'Avg PnL':>8} {'Total':>8} {'WR':>5}")
    print(f"    {'─'*38}")
    for month in sorted(p['monthly'].keys()):
        m = p['monthly'][month]
        oos = " (OOS)" if month >= '2026-06' else " (OPT)" if lbl == 'ST_SHORT' else ""
        print(f"    {month:>7} {m['n']:>4} {m['avg']:>+7.2%} {m['total']:>+7.2%} "
              f"{m['wr']:>5.0%}{oos}")


# ==============================================================================
# CHARTS
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ── [0,0]: Equity curves (cumulative PnL %) ──
ax = axes[0, 0]
for lbl, p, color in [
    ('F403_DEPLOYED', port_403, STRATEGY_COLORS['F403_DEPLOYED']),
    ('ST_SHORT', port_st, STRATEGY_COLORS['ST_SHORT']),
    ('COMBINED', port_comb, STRATEGY_COLORS['COMBINED']),
]:
    if p['n'] == 0: continue
    df = p['trades_df'].sort_values('date')
    df['cum_pnl'] = df['pnl'].cumsum()
    lw = 2.5 if lbl == 'COMBINED' else 1.8
    ax.plot(pd.to_datetime(df['date']), df['cum_pnl'] * 100,
            label=f"{lbl} ({p['n']}t, {p['avg']:+.2%})",
            color=color, linewidth=lw)

h2_dt = pd.Timestamp(H2_START)
ax.axvline(h2_dt, color='gray', linestyle='--', alpha=0.5, label='H2 start (OOS)')
ax.set_title('Cumulative PnL (%)')
ax.set_xlabel('Date'); ax.set_ylabel('Cumulative PnL (%)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── [0,1]: Monthly avg PnL bar chart ──
ax = axes[0, 1]
all_months = sorted(set(
    list(port_403.get('monthly', {}).keys()) +
    list(port_st.get('monthly', {}).keys()) +
    list(port_comb.get('monthly', {}).keys())
))
if all_months:
    x = np.arange(len(all_months))
    w = 0.25
    for i, (lbl, p, color) in enumerate([
        ('F403', port_403, STRATEGY_COLORS['F403_DEPLOYED']),
        ('ST', port_st, STRATEGY_COLORS['ST_SHORT']),
        ('COMB', port_comb, STRATEGY_COLORS['COMBINED']),
    ]):
        vals = [p['monthly'].get(m, {}).get('avg', 0) * 100 for m in all_months]
        ax.bar(x + i * w, vals, w, label=lbl, color=color, alpha=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(all_months, rotation=45, fontsize=8)
    ax.set_ylabel('Avg PnL (%)')
    ax.set_title('Monthly Avg PnL')
    ax.legend(fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

# ── [1,0]: Per-ticker heatmap ──
ax = axes[1, 0]
heatmap_tickers = sorted(set(list(per_tk_403.keys()) + list(per_tk_st.keys())))
heatmap_data = []
for ticker in heatmap_tickers:
    f403_trades = per_tk_403.get(ticker, [])
    st_trades = per_tk_st.get(ticker, [])
    f403_avg = np.mean([t['pnl'] for t in f403_trades]) if f403_trades else np.nan
    st_avg = np.mean([t['pnl'] for t in st_trades]) if st_trades else np.nan
    heatmap_data.append({'ticker': ticker, 'F403': f403_avg * 100, 'ST': st_avg * 100})

if heatmap_data:
    hm_df = pd.DataFrame(heatmap_data).set_index('ticker')
    im = ax.imshow(hm_df.values, cmap='RdYlGn', aspect='auto',
                   vmin=-5, vmax=5)
    ax.set_yticks(range(len(hm_df)))
    ax.set_yticklabels(hm_df.index, fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['F403', 'ST SHORT'], fontsize=9)
    for i in range(len(hm_df)):
        for j in range(2):
            v = hm_df.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:+.1f}%', ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if abs(v) > 3 else 'black')
    ax.set_title('Per-Ticker Avg PnL (%)')
    plt.colorbar(im, ax=ax, shrink=0.8)

# ── [1,1]: Portfolio equity ($200K start) ──
ax = axes[1, 1]
capital = 200_000

for lbl, p, color in [
    ('F403_DEPLOYED', port_403, STRATEGY_COLORS['F403_DEPLOYED']),
    ('ST_SHORT', port_st, STRATEGY_COLORS['ST_SHORT']),
    ('COMBINED', port_comb, STRATEGY_COLORS['COMBINED']),
]:
    if p['n'] == 0: continue
    df = p['trades_df'].sort_values('date')
    daily = df.groupby('date')['pnl'].mean()
    equity = [capital]
    for pnl_val in daily.values:
        equity.append(equity[-1] * (1 + pnl_val))
    dates_plot = [pd.Timestamp(TEST_START)] + [pd.Timestamp(d) for d in daily.index]
    lw = 2.5 if lbl == 'COMBINED' else 1.8
    ax.plot(dates_plot, [e / 1000 for e in equity],
            label=f"{lbl} (${equity[-1]/1000:.0f}K)",
            color=color, linewidth=lw)

ax.axvline(h2_dt, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Portfolio Value ($200K start)')
ax.set_ylabel('Value ($K)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}K'))

fig.suptitle(f'SHORT Portfolio: Spike Transformer vs F403 Deployed\n'
             f'{TEST_START} to {TEST_END} | Slippage: {SLIPPAGE_RT:.2%} RT',
             fontsize=13, fontweight='bold')
plt.tight_layout()

chart_path = '/tmp/backtest_st_short_vs_f403.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# GRAND SUMMARY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  GRAND SUMMARY")
print(f"{'='*W}")
print(f"\n  F403 DEPLOYED (BE/OKLO/SMR):")
print(f"    {port_403['n']} trades, {port_403['avg']:+.2%} avg, "
      f"{port_403['wr']:.0%} WR, H2: {port_403['h2_avg']:+.2%}")
print(f"    Cap-adj: {cap_403['daily_avg']:+.2%}/day, {cap_403['daily_wr']:.0%} daily WR")

print(f"\n  ST SHORT ({len(ST_TICKERS)} tickers):")
print(f"    {port_st['n']} trades, {port_st['avg']:+.2%} avg, "
      f"{port_st['wr']:.0%} WR, H2: {port_st['h2_avg']:+.2%}")
print(f"    Cap-adj: {cap_st['daily_avg']:+.2%}/day, {cap_st['daily_wr']:.0%} daily WR")

print(f"\n  COMBINED (union, {port_comb['n_tickers']} tickers):")
print(f"    {port_comb['n']} trades, {port_comb['avg']:+.2%} avg, "
      f"{port_comb['wr']:.0%} WR, H2: {port_comb['h2_avg']:+.2%}")
print(f"    Cap-adj: {cap_comb['daily_avg']:+.2%}/day, {cap_comb['daily_wr']:.0%} daily WR")

overlap_tickers = sorted(set(per_tk_403.keys()) & set(per_tk_st.keys()))
unique_st = sorted(set(per_tk_st.keys()) - set(per_tk_403.keys()))
if overlap_tickers:
    print(f"\n  OVERLAP tickers: {', '.join(overlap_tickers)}")
    print(f"  (F403 enters at 6:15 AM, ST enters at 6:25 AM — can run both)")
if unique_st:
    print(f"  ST-ONLY tickers: {', '.join(unique_st)}")
    print(f"  (New coverage not in F403)")

print(f"\n  H1 ({TEST_START} to {H2_START}): ST in-sample (biased), F403 validation")
print(f"  H2 ({H2_START} to {TEST_END}): Both out-of-sample (fair comparison)")
print(f"\n{'='*W}")
