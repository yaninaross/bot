# %% ── SHORT CSV vs Live TF Dual Backtest ─────────────────────────────────────
#
# Compares CSV-scored and live-TF-scored backtests for 4 strategies:
#   FOLD_403, FOLD_414, RANK_COMPOSITE, ORACLE
# on RULE_1 tickers. Validates that live scoring matches CSV backtesting.
#
# GPU recommended for live TF inference. Colab-ready (no if __name__).

import os, sys, io, gc, json, subprocess, ctypes, warnings
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
except ImportError:
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2
from google.cloud import storage as gcs_storage, bigquery

if tf.config.list_physical_devices('GPU'):
    mixed_precision.set_global_policy('mixed_float16')

if _IN_COLAB:
    try:
        auth.authenticate_user()
    except Exception:
        pass

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME     = "time-series-bucket-ys"
GCS_MODEL_PATH  = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
RESULTS_DIR     = "Algos/Results_all_v2"
INFERENCE_DIR   = "Data/Inference_v2"
TIME_STEPS      = 240
EMBED_DIM       = 16
N_ASSETS        = 31
OTHER_IDX       = 30
PST             = pytz.timezone('America/Los_Angeles')

TEST_START      = '2026-04-01'
TEST_END        = '2026-07-28'
HALF2_START     = '2026-06-01'
ENTRY_MIN       = 375
EXIT_TIME_MIN   = 780
SLIPPAGE_PCT    = 0.002
STARTING_CAPITAL = 200_000
SL_FIXED        = 0.03
MIN_H2_TRADES   = 3

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]
ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

SHORT_TICKERS = ['BE', 'IREN', 'OKLO', 'ONDS', 'RKLB', 'WOLF']

FOLDS_A2 = [400, 401, 402, 403, 404, 405]
FOLDS_A1 = [410, 411, 412, 413, 414, 415]
ALL_FOLDS = FOLDS_A2 + FOLDS_A1

FOLD_EXPERIMENTS    = {f: f'{f}_XSECTOR' for f in ALL_FOLDS}
FOLD_MODEL_VERSIONS = {f: f'robust_wf_v12_short_{f}' for f in ALL_FOLDS}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_ALLOWED_MODES = ('static', 'atr_sl', 'grace')

STRATEGIES_TO_TEST = ['FOLD_403', 'FOLD_414', 'RANK_COMPOSITE', 'ORACLE']
STRATEGY_COLORS = {
    'FOLD_403':       '#1565C0',
    'FOLD_414':       '#E65100',
    'RANK_COMPOSITE': '#2E7D32',
    'ORACLE':         '#D32F2F',
}
TICKER_COLORS = {
    'BE':   '#2196F3', 'IREN': '#4CAF50', 'OKLO': '#FF9800',
    'ONDS': '#E91E63', 'RKLB': '#9C27B0', 'WOLF': '#00BCD4',
}

FIXED_UNIVERSAL_CFG = {
    'fold': None,
    'tpsl_mode': 'static',
    'prob_thresh': 0.50,
    'tp_pct': 0.05,
    'sl_pct': SL_FIXED,
    'sl_mult': 0.0,
    'grace_minutes': 0,
    'entry_minute': ENTRY_MIN,
    'val_a_avg_pnl': 0.0,
    'opt_avg_pnl': 0.0,
    'config_id': 'fixed_universal',
}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110


# ==============================================================================
# CONTEXT COLS
# ==============================================================================
def _build_context_cols():
    cols = []
    for t in CONTEXT_ASSETS:
        cols.append(f'{t}_Anchor_ROC')
        for h in ROC_HORIZONS:
            cols.append(f'{t}_ROC_{h}m')
    cols.append('Time_Pos')
    return cols

CONTEXT_COLS = _build_context_cols()
ALL_COLS = ASSET_COLS + CONTEXT_COLS


# ==============================================================================
# DATA LOADING — live version (micro col handling)
# ==============================================================================
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
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        for c in MICRO_COLS:
            if c not in df.columns:
                df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        cutoff = pd.Timestamp(TEST_START, tz='UTC') - pd.Timedelta(days=60)
        df = df[df.index >= cutoff]
        _master_cache[ticker] = df
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
# MARKET CONTEXT
# ==============================================================================
def build_market_context():
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker)
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
        del df, daily_c
        gc.collect()
    if not frames:
        raise RuntimeError("No context assets loaded.")
    return pd.concat(frames, axis=1, sort=True).ffill().fillna(0)


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
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


def engineer_asset(ticker, market_df):
    df = load_master(ticker)
    if df is None:
        return None
    df = df.sort_index()
    df = heal_data(df)
    return engineer_features(df, market_df)


# ==============================================================================
# SCALING
# ==============================================================================
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
# MODEL ARCHITECTURE
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


def build_model():
    n_ac = len(ASSET_COLS)
    n_cc = len(CONTEXT_COLS)
    in_a  = Input(shape=(TIME_STEPS, n_ac), name='asset_in')
    in_c  = Input(shape=(n_cc,),            name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32, name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Embedding(N_ASSETS, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


# ==============================================================================
# MODEL LOADER
# ==============================================================================
_model_cache = {}

def get_model(seed, experiment='400_XSECTOR'):
    cache_key = (seed, experiment)
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    blob_path  = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{experiment}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{experiment}.h5"
    if not os.path.exists(local_path):
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
    model = build_model()
    model.load_weights(local_path)
    _model_cache[cache_key] = model
    return model


# ==============================================================================
# SCORING ENGINE — 15-seed ensemble at entry bar
# ==============================================================================
def score_ticker(ticker, cfg, df_scaled, df_raw, target_date):
    if len(df_scaled) == 0:
        return None
    entry_min = cfg['entry_minute']
    target_pst = PST.localize(datetime(target_date.year, target_date.month, target_date.day,
                                       entry_min // 60, entry_min % 60))
    target_utc = target_pst.astimezone(pytz.utc)
    candidates = df_scaled.index[df_scaled.index <= target_utc]
    if len(candidates) == 0:
        return None
    row_idx = df_scaled.index.get_loc(candidates[-1])
    bar_ts_pst = df_scaled.index[row_idx].tz_convert(PST)
    score_date = bar_ts_pst.date()
    if score_date != target_date:
        return None
    bar_minute = bar_ts_pst.hour * 60 + bar_ts_pst.minute
    if bar_minute != entry_min:
        return None
    if row_idx < TIME_STEPS:
        return None

    XA = df_scaled[ASSET_COLS].values
    XC = df_scaled[CONTEXT_COLS].values
    XA_b  = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]
    XC_b  = XC[row_idx - 1][np.newaxis]
    XID_b = np.array([[OTHER_IDX]], dtype=np.int32)

    base_exp = cfg['_base_experiment']
    seed_list = cfg['_seed_list']

    all_probs = []
    for s in seed_list:
        try:
            m = get_model(s, base_exp)
            p = float(m(
                {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
                training=False
            ).numpy().flatten()[0])
            all_probs.append(p)
        except Exception:
            continue
    if not all_probs:
        return None
    prob = float(np.mean(all_probs))

    return {
        'ticker': ticker,
        'prob': round(prob, 4),
        'threshold': cfg['prob_thresh'],
        'signal': prob >= cfg['prob_thresh'],
        'score_date': score_date,
        'row_idx': row_idx,
        'close': round(float(df_raw['Close'].iloc[row_idx]), 4),
    }


# ==============================================================================
# OHLCV ARRAYS
# ==============================================================================
def build_ohlcv_arrays(df_raw):
    idx_pst = df_raw.index.tz_convert(PST)
    return {
        'prices': df_raw['Close'].values.astype(np.float64),
        'highs':  df_raw['High'].values.astype(np.float64),
        'lows':   df_raw['Low'].values.astype(np.float64),
        'dates':  (idx_pst.year * 10000 + idx_pst.month * 100 + idx_pst.day).values.astype(np.int64),
        'times':  (idx_pst.hour * 60 + idx_pst.minute).values.astype(np.int64),
    }


def precompute_ohlcv_arrays(df, start_date, end_date):
    s = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    idx = df.index
    if str(idx.tz) != 'America/Los_Angeles':
        idx = idx.tz_convert('America/Los_Angeles')
    mask = (idx >= s) & (idx < e)
    df_cut = df[mask]
    if len(df_cut) == 0:
        return None
    idx_cut = df_cut.index if str(df_cut.index.tz) == 'America/Los_Angeles' \
        else df_cut.index.tz_convert('America/Los_Angeles')
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
# SHORT TRADE SIMULATION
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
        tp_hit = lows[k] <= target_price
        sl_hit = highs[k] >= stop_price
        if sl_hit and tp_hit:
            return (tp if np.random.random() < 0.5 else -sl), ed, True
        if sl_hit: return -sl, ed, True
        if tp_hit: return tp, ed, True
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
        tp_hit = lows[k] <= target_price
        sl_hit = highs[k] >= stop_price
        if sl_hit and tp_hit:
            return (tp_pct if np.random.random() < 0.5 else -sl), ed, True
        if sl_hit: return -sl, ed, True
        if tp_hit: return tp_pct, ed, True
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
        elapsed = time_mins[k] - et
        sl_active = elapsed >= grace_minutes
        tp_hit = lows[k] <= target_price
        sl_hit = sl_active and (highs[k] >= stop_price)
        if sl_hit and tp_hit:
            return (tp if np.random.random() < 0.5 else -sl), ed, True
        if sl_hit: return -sl, ed, True
        if tp_hit: return tp, ed, True
        if time_mins[k] >= exit_time:
            return (entry_price - prices[k]) / entry_price, ed, True
    return (entry_price - prices[n-1]) / entry_price, ed, True


def execute_trade(ohlcv, entry_idx, mode, tp_pct, sl_pct, sl_mult, grace_min):
    prices    = ohlcv['prices']
    highs     = ohlcv['highs']
    lows      = ohlcv['lows']
    date_ints = ohlcv['dates']
    time_mins = ohlcv['times']
    if mode == 'atr_sl':
        atr_arr = ohlcv.get('atr_pct', np.full(len(prices), 0.05))
        pnl, td, filled = run_short_trade_atr_sl(
            prices, highs, lows, date_ints, time_mins,
            entry_idx, tp_pct, float(atr_arr[entry_idx]), sl_mult, EXIT_TIME_MIN)
    elif mode == 'grace':
        pnl, td, filled = run_short_trade_grace(
            prices, highs, lows, date_ints, time_mins,
            entry_idx, tp_pct, sl_pct, EXIT_TIME_MIN, grace_min)
    else:
        pnl, td, filled = run_short_trade_static(
            prices, highs, lows, date_ints, time_mins,
            entry_idx, tp_pct, sl_pct, EXIT_TIME_MIN)
    if filled:
        pnl -= 2 * SLIPPAGE_PCT
    return pnl, td, filled


# ==============================================================================
# CSV SCORING — load seed probs + compute daily averages
# ==============================================================================
def load_seed_scores(ticker, experiment_base):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    bucket = gcs_client.bucket(BUCKET_NAME)
    prefix = f"{RESULTS_DIR}/Universal_Exp{experiment_base}/{clean}/Seed_"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        return {}
    results = {}
    for blob in blobs:
        name = blob.name.split('/')[-1]
        if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
            continue
        seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
        d = pd.read_csv(io.BytesIO(blob.download_as_bytes()),
                        index_col=0, parse_dates=True)
        if 'Prob_Spike' not in d.columns:
            continue
        if d.index.tz is None:
            d.index = d.index.tz_localize('UTC')
        d.index = d.index.tz_convert(PST)
        results[seed_id] = d['Prob_Spike'].sort_index()
    return results


def compute_daily_prob(seed_scores, entry_min):
    all_daily = defaultdict(list)
    for seed_id, series in seed_scores.items():
        bar_mins = series.index.hour * 60 + series.index.minute
        entry_vals = series[bar_mins == entry_min]
        for dt, val in entry_vals.items():
            all_daily[dt.date()].append(float(val))
    return {d: np.mean(ps) for d, ps in all_daily.items()}


def compute_csv_coverage(daily_probs, test_start, test_end):
    ts = pd.Timestamp(test_start).date()
    te = pd.Timestamp(test_end).date()
    return {d for d in daily_probs.keys() if ts <= d <= te}


# ==============================================================================
# SUMMARIZE RESULTS
# ==============================================================================
def summarize_trades(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0, 'monthly': {}, 'trades_df': pd.DataFrame()}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    cum = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    return {
        'label': label,
        'n': len(df), 'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': max_dd,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
        'trades_df': df,
    }


# ==============================================================================
# CSV BACKTEST
# ==============================================================================
def run_csv_backtest(configs, daily_probs_cache, ticker_ohlcv, label):
    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt   = pd.Timestamp(TEST_END).date()
    h2_start_dt   = pd.Timestamp(HALF2_START).date()

    all_trades = []
    per_ticker_trades = defaultdict(list)
    scored_probs = defaultdict(dict)

    for ticker in sorted(configs.keys()):
        if ticker not in ticker_ohlcv or ticker not in daily_probs_cache:
            continue
        cfg = configs[ticker]
        ohlcv = ticker_ohlcv[ticker]
        probs = daily_probs_cache[ticker]
        entry_min = cfg.get('entry_minute', ENTRY_MIN)
        prob_thresh = cfg['prob_thresh']
        mode = cfg['tpsl_mode']

        prices = ohlcv['prices']
        times  = ohlcv['times']
        dates  = ohlcv['dates']

        entry_indices = np.where(times == entry_min)[0]
        last_traded_date = -1

        for idx in entry_indices:
            d_int = int(dates[idx])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()

            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            if d_int == last_traded_date:
                continue
            prob = probs.get(d_date)
            if prob is None:
                continue

            scored_probs[ticker][d_date] = prob

            if prob < prob_thresh:
                continue

            pnl, trade_date, filled = execute_trade(
                ohlcv, idx, mode, cfg['tp_pct'], cfg['sl_pct'],
                cfg['sl_mult'], cfg['grace_minutes'])

            if not filled:
                continue

            last_traded_date = d_int
            half = 'H2' if d_date >= h2_start_dt else 'H1'
            trade = {
                'date': d_date, 'ticker': ticker, 'pnl': pnl,
                'prob': prob, 'close': float(prices[idx]),
                'mode': mode, 'tp': cfg['tp_pct'],
                'sl': cfg['sl_pct'] if mode != 'atr_sl' else cfg['sl_mult'],
                'half': half,
            }
            all_trades.append(trade)
            per_ticker_trades[ticker].append(trade)

    return all_trades, per_ticker_trades, scored_probs


# ==============================================================================
# CONFIG SELECTION (from BQ)
# ==============================================================================
def _safe_float(val, default=0.0):
    try:
        if pd.isna(val): return default
    except (TypeError, ValueError): pass
    try: return float(val)
    except (TypeError, ValueError): return default


def _bq_row_to_cfg(row):
    return {
        'fold': int(row['fold']),
        'tpsl_mode': row['tpsl_mode'],
        'prob_thresh': row['prob_thresh'],
        'tp_pct': row['tp_pct'],
        'sl_pct': row['sl_pct'],
        'sl_mult': row['sl_mult'],
        'grace_minutes': int(row['grace_minutes']),
        'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else ENTRY_MIN,
        'val_a_avg_pnl': row['val_a_avg_pnl'],
        'val_a_win_rate': row.get('val_a_win_rate', 0.0),
        'opt_avg_pnl': row['opt_avg_pnl'],
        'va_opt_ratio': row.get('va_opt_ratio', 0.0),
        'config_id': f"f{int(row['fold'])}_{row['tpsl_mode']}_t{row['prob_thresh']:.2f}_tp{row['tp_pct']:.2f}",
    }


def _pick_best(t_df, col='val_a_avg_pnl'):
    if len(t_df) == 0:
        return None
    best_idx = t_df[col].idxmax()
    return _bq_row_to_cfg(t_df.loc[best_idx])


def select_single_fold(bq_df, tickers, fold):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'] == fold)]
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker]
        cfg = _pick_best(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


def select_by_ranking(bq_df, tickers, metric):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'] == 403)].copy()
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker].copy()
        if len(t_df) == 0:
            continue

        if metric == 'overfit':
            valid = t_df[t_df['val_a_avg_pnl'] > 0].copy()
            if len(valid) == 0:
                valid = t_df
            valid['_overfit_dist'] = (valid['va_opt_ratio'] - 1.0).abs()
            best_idx = valid['_overfit_dist'].idxmin()
            selections[ticker] = _bq_row_to_cfg(valid.loc[best_idx])
        elif metric == 'win_rate':
            cfg = _pick_best(t_df, 'val_a_win_rate')
            if cfg:
                selections[ticker] = cfg
        elif metric == 'composite':
            t_df['_composite'] = t_df['val_a_avg_pnl'] * np.minimum(
                1.0, 1.0 / t_df['va_opt_ratio'].clip(lower=0.01))
            cfg = _pick_best(t_df, '_composite')
            if cfg:
                selections[ticker] = cfg
    return selections


# ==============================================================================
# LIVE BACKTEST — day-by-day TF scoring
# ==============================================================================
def run_live_backtest(configs, ticker_data_dict, trading_days, csv_coverage, label):
    h2_start_dt = pd.Timestamp(HALF2_START).date()

    all_trades = []
    per_ticker_trades = defaultdict(list)
    scored_probs = defaultdict(dict)

    for di, day in enumerate(trading_days):
        for ticker, cfg in sorted(configs.items()):
            if ticker not in ticker_data_dict:
                continue
            if day not in csv_coverage.get(ticker, set()):
                continue

            td = ticker_data_dict[ticker]
            exp = cfg['_base_experiment']
            df_scaled = td['scaled'].get(exp)
            if df_scaled is None:
                continue
            df_raw = td['df_raw']

            result = score_ticker(ticker, cfg, df_scaled, df_raw, day)
            if result is None:
                continue

            scored_probs[ticker][day] = result['prob']

            if not result['signal']:
                continue

            ohlcv = td['ohlcv']
            entry_idx = result['row_idx']
            mode = cfg['tpsl_mode']

            if mode == 'atr_sl':
                atr_arr = ohlcv.get('atr_pct', np.full(len(ohlcv['prices']), 0.05))
                atr_val = float(atr_arr[entry_idx]) if entry_idx < len(atr_arr) else 0.05
                pnl, trade_date, filled = run_short_trade_atr_sl(
                    ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                    ohlcv['dates'], ohlcv['times'],
                    entry_idx, cfg['tp_pct'], atr_val, cfg['sl_mult'], EXIT_TIME_MIN)
            elif mode == 'grace':
                pnl, trade_date, filled = run_short_trade_grace(
                    ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                    ohlcv['dates'], ohlcv['times'],
                    entry_idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
                    cfg['grace_minutes'])
            else:
                pnl, trade_date, filled = run_short_trade_static(
                    ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                    ohlcv['dates'], ohlcv['times'],
                    entry_idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)

            if not filled:
                continue

            pnl -= 2 * SLIPPAGE_PCT
            half = 'H2' if day >= h2_start_dt else 'H1'
            trade = {
                'date': day, 'ticker': ticker, 'pnl': pnl,
                'prob': result['prob'], 'close': result['close'],
                'mode': mode, 'tp': cfg['tp_pct'],
                'sl': cfg['sl_pct'] if mode != 'atr_sl' else cfg['sl_mult'],
                'half': half,
            }
            all_trades.append(trade)
            per_ticker_trades[ticker].append(trade)

        if (di + 1) % 10 == 0 or di == len(trading_days) - 1:
            print(f"      Day {di+1}/{len(trading_days)}: {day}  "
                  f"({len(all_trades)} trades so far)")

    return all_trades, per_ticker_trades, scored_probs


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT DUAL BACKTEST — CSV vs Live TF Scoring")
print(f"  STRATEGIES: {', '.join(STRATEGIES_TO_TEST)}")
print(f"  TICKERS: {', '.join(sorted(SHORT_TICKERS))} (RULE_1)")
print(f"  TEST: {TEST_START} -> {TEST_END}  |  H2: {HALF2_START} -> {TEST_END}")
print(f"  ENTRY: 6:15 PST ({ENTRY_MIN} min)  |  EXIT: 1:00 PM ({EXIT_TIME_MIN} min)")
print(f"  SLIPPAGE: {SLIPPAGE_PCT:.2%} each way  |  CAPITAL: ${STARTING_CAPITAL:,}")
print(f"{'='*W}")


# ── Step 1: Load BQ configs ──
print(f"\n  STEP 1: Loading BQ configs (all 12 folds, ensemble)...")

mv_str = ", ".join(f"'{FOLD_MODEL_VERSIONS[f]}'" for f in ALL_FOLDS)
modes_str = ", ".join(f"'{m}'" for m in BQ_ALLOWED_MODES)

q = f"""
SELECT ticker, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, sl_mult, grace_minutes,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       is_ensemble, model_version
FROM `{BQ_TABLE}`
WHERE model_version IN ({mv_str})
  AND opt_avg_pnl > 0
  AND tpsl_mode IN ({modes_str})
  AND is_ensemble = TRUE
ORDER BY ticker, model_version
"""
bq_df = bq_client.query(q).to_dataframe()
bq_df['fold'] = bq_df['model_version'].str.extract(r'_(\d+)$').astype(int)

for col in ['prob_thresh', 'tp_pct', 'sl_pct', 'sl_mult', 'grace_minutes',
            'entry_minute', 'opt_avg_pnl', 'opt_win_rate', 'opt_count',
            'val_a_avg_pnl', 'val_a_win_rate', 'val_a_count']:
    bq_df[col] = bq_df[col].apply(lambda x: _safe_float(x, 0.0))

bq_df['va_opt_ratio'] = bq_df['val_a_avg_pnl'] / bq_df['opt_avg_pnl'].clip(lower=0.001)

bq_tickers = sorted(bq_df['ticker'].unique())
print(f"    BQ: {len(bq_df)} configs, {len(bq_tickers)} tickers, {bq_df['fold'].nunique()} folds")
print(f"    RULE_1 in BQ: {[t for t in SHORT_TICKERS if t in bq_tickers]}")


# ── Step 2: Select configs for FOLD_403, FOLD_414, RANK_COMPOSITE ──
print(f"\n  STEP 2: Selecting configs for 3 strategies...")

strategies = {}
strategies['FOLD_403'] = select_single_fold(bq_df, SHORT_TICKERS, 403)
strategies['FOLD_414'] = select_single_fold(bq_df, SHORT_TICKERS, 414)
strategies['RANK_COMPOSITE'] = select_by_ranking(bq_df, SHORT_TICKERS, 'composite')

for sn in ['FOLD_403', 'FOLD_414', 'RANK_COMPOSITE']:
    cfgs = strategies[sn]
    print(f"    {sn}: {len(cfgs)} tickers")
    for ticker, cfg in sorted(cfgs.items()):
        print(f"      {ticker}: f{cfg['fold']}_{cfg['tpsl_mode']} "
              f"prob>{cfg['prob_thresh']:.2f} TP={cfg['tp_pct']:.0%} "
              f"SL={cfg['sl_pct']:.0%} val_a={cfg['val_a_avg_pnl']:+.2%}")


# ── Step 3: Load OHLCV for CSV backtest (test-period filtered) ──
print(f"\n  STEP 3: Loading OHLCV + ATR for {len(SHORT_TICKERS)} tickers...")

ticker_ohlcv_csv = {}
_ohlcv_df_cache = {}

for ticker in SHORT_TICKERS:
    try:
        df = load_master(ticker)
        if df is None:
            continue
        if isinstance(df.index, pd.MultiIndex):
            _master_cache.pop(ticker, None)
            continue
        df = heal_data(df)
        if isinstance(df.index, pd.MultiIndex):
            _master_cache.pop(ticker, None)
            continue
        ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
        if ohlcv is None:
            print(f"    {ticker}: no data in test period")
            continue
        atr_pct = compute_daily_atr_pct(df)
        atr_vals = atr_pct.reindex(df.index).fillna(0.05)
        s = pd.to_datetime(TEST_START).tz_localize('America/Los_Angeles')
        e = pd.to_datetime(TEST_END).tz_localize('America/Los_Angeles') + timedelta(days=1)
        idx = df.index
        if str(idx.tz) != 'America/Los_Angeles':
            idx = idx.tz_convert('America/Los_Angeles')
        atr_cut = atr_vals[(idx >= s) & (idx < e)]
        ohlcv['atr_pct'] = atr_cut.values.astype(np.float64)
        ticker_ohlcv_csv[ticker] = ohlcv
        _ohlcv_df_cache[ticker] = df
        n_days = len(np.unique(ohlcv['dates']))
        print(f"    {ticker}: {n_days} trading days, ATR_pct={atr_cut.median():.4f}")
    except Exception as ex:
        print(f"    {ticker}: OHLCV load failed -- {ex}")
    gc.collect()

_release_memory()


# ── Step 4: Load CSV seed scores for all needed folds ──
print(f"\n  STEP 4: Loading CSV seed scores for needed folds...")

needed_folds = set()
for sn in ['FOLD_403', 'FOLD_414', 'RANK_COMPOSITE']:
    for ticker, cfg in strategies[sn].items():
        needed_folds.add(cfg['fold'])

all_fold_probs = defaultdict(dict)
for fold in sorted(needed_folds):
    exp = FOLD_EXPERIMENTS[fold]
    n_loaded = 0
    for ticker in SHORT_TICKERS:
        if ticker not in ticker_ohlcv_csv:
            continue
        seeds = load_seed_scores(ticker, exp)
        if not seeds:
            continue
        dp = compute_daily_prob(seeds, ENTRY_MIN)
        if dp:
            all_fold_probs[ticker][fold] = dp
            n_loaded += 1
        del seeds
    print(f"    Fold {fold}: {n_loaded} tickers loaded")
    _release_memory()


# ── Step 5: Build per-strategy daily probs for CSV backtest ──
print(f"\n  STEP 5: Building per-strategy CSV probability maps...")

strategy_probs_csv = {}
for sn in ['FOLD_403', 'FOLD_414', 'RANK_COMPOSITE']:
    dp_map = {}
    for ticker, cfg in strategies[sn].items():
        if ticker not in ticker_ohlcv_csv:
            continue
        fold = cfg['fold']
        dp = all_fold_probs.get(ticker, {}).get(fold, {})
        if dp:
            dp_map[ticker] = dp
    strategy_probs_csv[sn] = dp_map
    print(f"    {sn}: {len(dp_map)} tickers with probs")


# ── Step 6: Oracle exhaustive backtest (CSV, H2-only optimization) ──
print(f"\n  STEP 6: Running exhaustive oracle backtest (CSV)...")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

all_ticker_configs = {}
for ticker in SHORT_TICKERS:
    t_df = bq_df[bq_df['ticker'] == ticker]
    if len(t_df) == 0:
        continue
    configs_list = []
    for _, bq_row in t_df.iterrows():
        configs_list.append({
            'fold': int(bq_row['fold']),
            'tpsl_mode': bq_row['tpsl_mode'],
            'prob_thresh': bq_row['prob_thresh'],
            'tp_pct': bq_row['tp_pct'],
            'sl_pct': bq_row['sl_pct'],
            'sl_mult': bq_row['sl_mult'],
            'grace_minutes': int(bq_row['grace_minutes']),
            'entry_minute': int(bq_row['entry_minute']) if bq_row['entry_minute'] > 0 else ENTRY_MIN,
            'config_id': f"f{int(bq_row['fold'])}_{bq_row['tpsl_mode']}_t{bq_row['prob_thresh']:.2f}_tp{bq_row['tp_pct']:.2f}",
        })
    configs_list.append(FIXED_UNIVERSAL_CFG.copy())
    all_ticker_configs[ticker] = configs_list

ensemble_probs_all = {}
for ticker in ticker_ohlcv_csv:
    fold_dps = all_fold_probs.get(ticker, {})
    if not fold_dps:
        continue
    all_dates = set()
    for dp in fold_dps.values():
        all_dates.update(dp.keys())
    ep_all = {}
    for d in all_dates:
        vals = [dp[d] for f, dp in fold_dps.items() if d in dp]
        if vals:
            ep_all[d] = np.mean(vals)
    ensemble_probs_all[ticker] = ep_all

oracle_fold_extras = set()
for ticker in SHORT_TICKERS:
    for cfg in all_ticker_configs.get(ticker, []):
        if cfg['fold'] is not None:
            oracle_fold_extras.add(cfg['fold'])

extra_folds = oracle_fold_extras - needed_folds
if extra_folds:
    print(f"    Loading {len(extra_folds)} extra folds for oracle: {sorted(extra_folds)}")
    for fold in sorted(extra_folds):
        exp = FOLD_EXPERIMENTS[fold]
        n_loaded = 0
        for ticker in SHORT_TICKERS:
            if ticker not in ticker_ohlcv_csv:
                continue
            if fold in all_fold_probs.get(ticker, {}):
                continue
            seeds = load_seed_scores(ticker, exp)
            if not seeds:
                continue
            dp = compute_daily_prob(seeds, ENTRY_MIN)
            if dp:
                all_fold_probs[ticker][fold] = dp
                n_loaded += 1
            del seeds
        if n_loaded > 0:
            print(f"      Fold {fold}: {n_loaded} tickers")
        _release_memory()

    for ticker in ticker_ohlcv_csv:
        fold_dps = all_fold_probs.get(ticker, {})
        if not fold_dps:
            continue
        all_dates = set()
        for dp in fold_dps.values():
            all_dates.update(dp.keys())
        ep_all = {}
        for d in all_dates:
            vals = [dp[d] for f, dp in fold_dps.items() if d in dp]
            if vals:
                ep_all[d] = np.mean(vals)
        ensemble_probs_all[ticker] = ep_all

tickers_ready = sorted(set(ticker_ohlcv_csv.keys()) & set(ensemble_probs_all.keys()))

oracle_configs = {}
total_bt = 0

for ticker in tickers_ready:
    if ticker not in all_ticker_configs:
        continue
    ohlcv = ticker_ohlcv_csv[ticker]
    best_h2_avg = -999
    best_cfg = None

    for cfg in all_ticker_configs[ticker]:
        if cfg['fold'] is not None:
            dp = all_fold_probs.get(ticker, {}).get(cfg['fold'], {})
            if not dp:
                dp = ensemble_probs_all.get(ticker, {})
        else:
            dp = ensemble_probs_all.get(ticker, {})
        if not dp:
            continue

        em = cfg['entry_minute'] if cfg['entry_minute'] > 0 else ENTRY_MIN
        thresh = cfg['prob_thresh']
        entry_indices = np.where(ohlcv['times'] == em)[0]
        h2_trades_list = []
        last_td = -1

        for idx in entry_indices:
            d_int = int(ohlcv['dates'][idx])
            if d_int == last_td:
                continue
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
            if d_date < test_start_dt or d_date > test_end_dt:
                last_td = d_int
                continue
            if d_date < h2_start_dt:
                last_td = d_int
                continue
            prob = dp.get(d_date, 0.0)
            if prob < thresh:
                last_td = d_int
                continue
            pnl, td, filled = execute_trade(
                ohlcv, idx, cfg['tpsl_mode'],
                cfg['tp_pct'], cfg['sl_pct'], cfg['sl_mult'],
                cfg['grace_minutes'])
            if filled:
                h2_trades_list.append(pnl)
            last_td = d_int

        total_bt += 1
        if len(h2_trades_list) >= MIN_H2_TRADES:
            h2_avg = np.mean(h2_trades_list)
            if h2_avg > best_h2_avg:
                best_h2_avg = h2_avg
                best_cfg = cfg.copy()

    if best_cfg is not None:
        oracle_configs[ticker] = best_cfg
        print(f"    {ticker}: oracle = f{best_cfg['fold']}_{best_cfg['tpsl_mode']} "
              f"TP={best_cfg['tp_pct']:.0%} thresh={best_cfg['prob_thresh']:.2f} -> H2={best_h2_avg:+.2%}")

print(f"    Total configs tested: {total_bt}")
print(f"    Oracle tickers: {len(oracle_configs)}")

strategies['ORACLE'] = oracle_configs

oracle_dp_map = {}
for ticker, cfg in oracle_configs.items():
    fold = cfg['fold']
    if fold is not None and fold in all_fold_probs.get(ticker, {}):
        oracle_dp_map[ticker] = all_fold_probs[ticker][fold]
    elif ticker in ensemble_probs_all:
        oracle_dp_map[ticker] = ensemble_probs_all[ticker]
strategy_probs_csv['ORACLE'] = oracle_dp_map


# ── Step 7: Run CSV backtests for all 4 strategies ──
print(f"\n  STEP 7: Running CSV backtests...")

csv_results = {}
csv_per_tk = {}
csv_scored_probs = {}

for sn in STRATEGIES_TO_TEST:
    np.random.seed(42)
    cfgs = strategies[sn]
    probs = strategy_probs_csv[sn]
    trades, per_tk, scored = run_csv_backtest(cfgs, probs, ticker_ohlcv_csv, sn)
    port = summarize_trades(trades, f'{sn}_CSV')
    csv_results[sn] = port
    csv_per_tk[sn] = per_tk
    csv_scored_probs[sn] = scored
    print(f"    {sn}: {port['n']} trades, avg={port['avg']:+.2%}, total={port['total']:+.1%}, wr={port['wr']:.0%}")


# ── Step 8: Build live TF infrastructure ──
print(f"\n  STEP 8: Building live TF infrastructure...")

all_needed_exps = set()
for sn in STRATEGIES_TO_TEST:
    for ticker, cfg in strategies[sn].items():
        fold = cfg['fold']
        if fold is not None:
            all_needed_exps.add(FOLD_EXPERIMENTS[fold])

print(f"    Need experiments: {sorted(all_needed_exps)}")

# 8a: Load scaling stats
print(f"    Loading scaling stats...")
scaling_stats = {}
for exp in sorted(all_needed_exps):
    blob = gcs_client.bucket(BUCKET_NAME).blob(
        f"{INFERENCE_DIR}/scaling_stats_experiment_{exp}.json")
    scaling_stats[exp] = json.loads(blob.download_as_text())
    print(f"      {exp}: {len(scaling_stats[exp])} features")

# 8b: Build market context + engineer features
print(f"    Building market context + engineering features...")
market_df = build_market_context()
print(f"      Market context: {len(market_df)} bars")

ticker_live_data = {}
for ticker in sorted(SHORT_TICKERS):
    df_raw = engineer_asset(ticker, market_df)
    if df_raw is None:
        print(f"      {ticker}: SKIP — no data")
        continue

    ohlcv_full = build_ohlcv_arrays(df_raw)
    atr_pct = compute_daily_atr_pct(df_raw)
    atr_vals = atr_pct.fillna(0.05)
    ohlcv_full['atr_pct'] = atr_vals.values.astype(np.float64)

    scaled_dfs = {}
    for exp in all_needed_exps:
        if exp in scaling_stats:
            scaled_dfs[exp] = apply_scaling(df_raw, scaling_stats[exp])

    ticker_live_data[ticker] = {
        'df_raw': df_raw,
        'ohlcv': ohlcv_full,
        'scaled': scaled_dfs,
    }
    n_days = len(np.unique(ohlcv_full['dates']))
    print(f"      {ticker}: {n_days} trading days, {len(scaled_dfs)} scaled variants")

del market_df
gc.collect()

# 8c: Build CSV coverage for live scoring (restrict to CSV-valid days)
print(f"    Building CSV coverage sets...")
csv_coverage = {}
for ticker in ticker_live_data:
    coverage = set()
    for fold in all_fold_probs.get(ticker, {}):
        dp = all_fold_probs[ticker][fold]
        for d in dp:
            if test_start_dt <= d <= test_end_dt:
                coverage.add(d)
    csv_coverage[ticker] = coverage
    print(f"      {ticker}: {len(coverage)} valid days in test window")

# 8d: Load TF models
print(f"    Loading TF models...")
for exp in sorted(all_needed_exps):
    alpha = exp.split('_')[0]
    seed_base = int(alpha) * 100
    seeds = list(range(seed_base + 1, seed_base + 16))
    n_loaded = 0
    for s in seeds:
        try:
            get_model(s, exp)
            n_loaded += 1
        except Exception as e:
            print(f"      {exp} seed {s} FAILED: {e}")
    print(f"      {exp}: {n_loaded}/15 models loaded")
    if n_loaded == 0:
        raise RuntimeError(f"No models loaded for {exp}")

# 8e: Augment strategy configs with _base_experiment and _seed_list
for sn in STRATEGIES_TO_TEST:
    for ticker, cfg in strategies[sn].items():
        fold = cfg['fold']
        if fold is not None:
            exp = FOLD_EXPERIMENTS[fold]
            cfg['_base_experiment'] = exp
            seed_base = fold * 100
            cfg['_seed_list'] = list(range(seed_base + 1, seed_base + 16))
        else:
            cfg['_base_experiment'] = '403_XSECTOR'
            cfg['_seed_list'] = list(range(40301, 40316))

# 8f: Get trading days
all_dates = set()
for ticker, td in ticker_live_data.items():
    o = td['ohlcv']
    for d_int in np.unique(o['dates']):
        d_str = str(int(d_int))
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if test_start_dt <= d_date <= test_end_dt and d_date.weekday() < 5:
            all_dates.add(d_date)
trading_days = sorted(all_dates)
print(f"    Trading days: {len(trading_days)} ({trading_days[0]} -> {trading_days[-1]})")


# ── Step 9: Run live TF backtests ──
print(f"\n  STEP 9: Running live TF backtests...")

live_results = {}
live_per_tk = {}
live_scored_probs = {}

for sn in STRATEGIES_TO_TEST:
    print(f"\n    ── {sn} ──")
    np.random.seed(42)
    cfgs = strategies[sn]
    trades, per_tk, scored = run_live_backtest(
        cfgs, ticker_live_data, trading_days, csv_coverage, sn)
    port = summarize_trades(trades, f'{sn}_LIVE')
    live_results[sn] = port
    live_per_tk[sn] = per_tk
    live_scored_probs[sn] = scored
    print(f"    → {sn}: {port['n']} trades, avg={port['avg']:+.2%}, "
          f"total={port['total']:+.1%}, wr={port['wr']:.0%}")

_release_memory()


# ==============================================================================
# COMPARISON ANALYSIS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  CSV vs LIVE COMPARISON RESULTS ({TEST_START} -> {TEST_END})")
print(f"{'='*W}")


# ── Table 1: CSV vs Live Summary ──
print(f"\n{'='*W}")
print(f"  TABLE 1: CSV vs LIVE SUMMARY")
print(f"{'='*W}")

header = (f"  {'Strategy':<16} │ {'CSV N':>5} {'CSV Avg':>8} {'CSV Total':>10} "
          f"│ {'Live N':>6} {'Live Avg':>9} {'Live Total':>11} │ {'Δ Avg':>7} {'Match%':>7}")
print(f"\n{header}")
print(f"  {'─'*100}")

for sn in STRATEGIES_TO_TEST:
    cp = csv_results[sn]
    lp = live_results[sn]

    csv_signal_days = set()
    if cp['n'] > 0:
        for _, row in cp['trades_df'].iterrows():
            csv_signal_days.add((row['ticker'], row['date']))

    live_signal_days = set()
    if lp['n'] > 0:
        for _, row in lp['trades_df'].iterrows():
            live_signal_days.add((row['ticker'], row['date']))

    if csv_signal_days | live_signal_days:
        match_pct = len(csv_signal_days & live_signal_days) / len(csv_signal_days | live_signal_days) * 100
    else:
        match_pct = 0

    delta_avg = lp['avg'] - cp['avg'] if cp['n'] > 0 and lp['n'] > 0 else 0

    print(f"  {sn:<16} │ {cp['n']:>5} {cp['avg']:>+8.2%} {cp['total']:>+10.1%} "
          f"│ {lp['n']:>6} {lp['avg']:>+9.2%} {lp['total']:>+11.1%} "
          f"│ {delta_avg:>+7.2%} {match_pct:>6.0f}%")


# ── Table 2: Per-Ticker Breakdown for each strategy ──
print(f"\n{'='*W}")
print(f"  TABLE 2: PER-TICKER CSV vs LIVE (per strategy)")
print(f"{'='*W}")

for sn in STRATEGIES_TO_TEST:
    print(f"\n  ── {sn} ──")
    print(f"  {'Ticker':<7} │ {'CSV N':>5} {'CSV Avg':>8} │ {'Live N':>6} {'Live Avg':>9} "
          f"│ {'Signal Match':>12} │ {'Prob r':>7} {'Prob |Δ|':>8}")
    print(f"  {'─'*85}")

    for ticker in sorted(SHORT_TICKERS):
        csv_tk = csv_per_tk.get(sn, {}).get(ticker, [])
        live_tk = live_per_tk.get(sn, {}).get(ticker, [])
        csv_n = len(csv_tk)
        live_n = len(live_tk)
        csv_avg = np.mean([t['pnl'] for t in csv_tk]) if csv_tk else 0
        live_avg = np.mean([t['pnl'] for t in live_tk]) if live_tk else 0

        csv_sig = {t['date'] for t in csv_tk}
        live_sig = {t['date'] for t in live_tk}
        union = csv_sig | live_sig
        match_pct = len(csv_sig & live_sig) / len(union) * 100 if union else 0

        csv_sp = csv_scored_probs.get(sn, {}).get(ticker, {})
        live_sp = live_scored_probs.get(sn, {}).get(ticker, {})
        common_days = sorted(set(csv_sp.keys()) & set(live_sp.keys()))
        if len(common_days) >= 3:
            csv_vals = [csv_sp[d] for d in common_days]
            live_vals = [live_sp[d] for d in common_days]
            r = np.corrcoef(csv_vals, live_vals)[0, 1] if len(set(csv_vals)) > 1 else 0
            mean_delta = np.mean([abs(csv_sp[d] - live_sp[d]) for d in common_days])
        else:
            r = 0
            mean_delta = 0

        print(f"  {ticker:<7} │ {csv_n:>5} {csv_avg:>+8.2%} │ {live_n:>6} {live_avg:>+9.2%} "
              f"│ {match_pct:>11.0f}% │ {r:>+7.3f} {mean_delta:>8.4f}")


# ── Table 3: Signal Agreement Matrix ──
print(f"\n{'='*W}")
print(f"  TABLE 3: SIGNAL AGREEMENT MATRIX")
print(f"{'='*W}")

print(f"\n  {'Strategy':<16} │ {'Both':>5} {'CSV Only':>9} {'Live Only':>10} "
      f"{'Neither':>8} │ {'Agreement':>10}")
print(f"  {'─'*75}")

for sn in STRATEGIES_TO_TEST:
    csv_sp = csv_scored_probs.get(sn, {})
    live_sp = live_scored_probs.get(sn, {})
    cfgs = strategies[sn]

    both_sig = 0
    csv_only = 0
    live_only = 0
    neither = 0

    for ticker in SHORT_TICKERS:
        if ticker not in cfgs:
            continue
        thresh = cfgs[ticker]['prob_thresh']
        csv_d = csv_sp.get(ticker, {})
        live_d = live_sp.get(ticker, {})
        common = sorted(set(csv_d.keys()) & set(live_d.keys()))

        for d in common:
            c_sig = csv_d[d] >= thresh
            l_sig = live_d[d] >= thresh
            if c_sig and l_sig:
                both_sig += 1
            elif c_sig:
                csv_only += 1
            elif l_sig:
                live_only += 1
            else:
                neither += 1

    total = both_sig + csv_only + live_only + neither
    agree_pct = (both_sig + neither) / total * 100 if total > 0 else 0

    print(f"  {sn:<16} │ {both_sig:>5} {csv_only:>9} {live_only:>10} "
          f"{neither:>8} │ {agree_pct:>9.1f}%")


# ── Table 4: Probability Statistics ──
print(f"\n{'='*W}")
print(f"  TABLE 4: PROBABILITY COMPARISON STATISTICS")
print(f"{'='*W}")

print(f"\n  {'Strategy':<16} │ {'N Scored':>8} {'Mean CSV':>9} {'Mean Live':>10} "
      f"{'Mean |Δ|':>9} {'Max |Δ|':>8} {'Corr r':>7}")
print(f"  {'─'*80}")

for sn in STRATEGIES_TO_TEST:
    csv_sp = csv_scored_probs.get(sn, {})
    live_sp = live_scored_probs.get(sn, {})

    all_csv = []
    all_live = []
    all_deltas = []

    for ticker in SHORT_TICKERS:
        csv_d = csv_sp.get(ticker, {})
        live_d = live_sp.get(ticker, {})
        common = sorted(set(csv_d.keys()) & set(live_d.keys()))
        for d in common:
            all_csv.append(csv_d[d])
            all_live.append(live_d[d])
            all_deltas.append(abs(csv_d[d] - live_d[d]))

    if len(all_csv) >= 3:
        r = np.corrcoef(all_csv, all_live)[0, 1] if len(set(all_csv)) > 1 else 0
        print(f"  {sn:<16} │ {len(all_csv):>8} {np.mean(all_csv):>9.4f} "
              f"{np.mean(all_live):>10.4f} {np.mean(all_deltas):>9.4f} "
              f"{np.max(all_deltas):>8.4f} {r:>+7.3f}")
    else:
        print(f"  {sn:<16} │ {'N/A':>8}")


# ── Verdict ──
print(f"\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

verdicts = []
for sn in STRATEGIES_TO_TEST:
    cp = csv_results[sn]
    lp = live_results[sn]

    csv_sp = csv_scored_probs.get(sn, {})
    live_sp = live_scored_probs.get(sn, {})
    all_csv_v = []
    all_live_v = []
    for ticker in SHORT_TICKERS:
        csv_d = csv_sp.get(ticker, {})
        live_d = live_sp.get(ticker, {})
        common = sorted(set(csv_d.keys()) & set(live_d.keys()))
        for d in common:
            all_csv_v.append(csv_d[d])
            all_live_v.append(live_d[d])

    r = np.corrcoef(all_csv_v, all_live_v)[0, 1] if len(all_csv_v) >= 3 and len(set(all_csv_v)) > 1 else 0

    csv_signal_days = set()
    if cp['n'] > 0:
        for _, row in cp['trades_df'].iterrows():
            csv_signal_days.add((row['ticker'], row['date']))
    live_signal_days = set()
    if lp['n'] > 0:
        for _, row in lp['trades_df'].iterrows():
            live_signal_days.add((row['ticker'], row['date']))
    union = csv_signal_days | live_signal_days
    match_pct = len(csv_signal_days & live_signal_days) / len(union) * 100 if union else 0

    status = "CONSISTENT" if match_pct >= 80 and r >= 0.7 else \
             "PARTIAL" if match_pct >= 60 or r >= 0.5 else "DIVERGENT"
    verdicts.append((sn, match_pct, r, status))
    print(f"    {sn:<16}: Signal match={match_pct:.0f}%, Prob r={r:+.3f} → {status}")

overall_match = np.mean([v[1] for v in verdicts])
overall_r = np.mean([v[2] for v in verdicts])
overall = "CONSISTENT" if overall_match >= 80 and overall_r >= 0.7 else \
          "PARTIAL" if overall_match >= 60 or overall_r >= 0.5 else "DIVERGENT"
print(f"\n    OVERALL: Signal match={overall_match:.0f}%, Prob r={overall_r:+.3f} → {overall}")
print(f"    (CONSISTENT: match>=80% AND r>=0.7 | PARTIAL: match>=60% OR r>=0.5 | DIVERGENT)")


# ==============================================================================
# CHARTS (2x2)
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# ── Chart 1: CSV vs Live Avg PnL (grouped bars) ──
ax1 = axes[0, 0]
x = np.arange(len(STRATEGIES_TO_TEST))
bar_w = 0.35
csv_avgs = [csv_results[sn]['avg'] * 100 for sn in STRATEGIES_TO_TEST]
live_avgs = [live_results[sn]['avg'] * 100 for sn in STRATEGIES_TO_TEST]

bars1 = ax1.bar(x - bar_w/2, csv_avgs, bar_w, label='CSV', color='#1565C0', alpha=0.85)
bars2 = ax1.bar(x + bar_w/2, live_avgs, bar_w, label='Live TF', color='#E65100', alpha=0.85)

ax1.set_xticks(x)
ax1.set_xticklabels(STRATEGIES_TO_TEST, fontsize=10, rotation=15, ha='right')
ax1.set_ylabel('Avg PnL per Trade (%)')
ax1.set_title('CSV vs Live TF — Average PnL', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.axhline(y=0, color='gray', linewidth=0.5)
ax1.grid(True, alpha=0.3, axis='y')

for bar_set in [bars1, bars2]:
    for bar in bar_set:
        h = bar.get_height()
        ax1.annotate(f'{h:+.2f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8)

# ── Chart 2: Signal Match Rate ──
ax2 = axes[0, 1]
match_rates = []
for sn in STRATEGIES_TO_TEST:
    cp = csv_results[sn]
    lp = live_results[sn]
    csv_sig = set()
    if cp['n'] > 0:
        for _, row in cp['trades_df'].iterrows():
            csv_sig.add((row['ticker'], row['date']))
    live_sig = set()
    if lp['n'] > 0:
        for _, row in lp['trades_df'].iterrows():
            live_sig.add((row['ticker'], row['date']))
    union = csv_sig | live_sig
    match_rates.append(len(csv_sig & live_sig) / len(union) * 100 if union else 0)

colors_bar = [STRATEGY_COLORS.get(sn, '#666') for sn in STRATEGIES_TO_TEST]
bars3 = ax2.bar(STRATEGIES_TO_TEST, match_rates, color=colors_bar, alpha=0.85)
ax2.axhline(y=80, color='green', linewidth=2, linestyle='--', label='80% threshold', alpha=0.7)
ax2.axhline(y=60, color='orange', linewidth=1.5, linestyle='--', label='60% threshold', alpha=0.5)
ax2.set_ylabel('Signal Match Rate (%)')
ax2.set_title('Signal Agreement — CSV vs Live', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

for bar, pct in zip(bars3, match_rates):
    ax2.annotate(f'{pct:.0f}%', xy=(bar.get_x() + bar.get_width()/2, pct),
                 xytext=(0, 4), textcoords='offset points',
                 ha='center', fontsize=10, fontweight='bold')

# ── Chart 3: Cumulative PnL overlay — CSV (solid) vs Live (dashed) ──
ax3 = axes[1, 0]
for sn in STRATEGIES_TO_TEST:
    color = STRATEGY_COLORS.get(sn, '#666')
    cp = csv_results[sn]
    lp = live_results[sn]

    if cp['n'] > 0:
        df_csv = cp['trades_df'].sort_values('date')
        csv_cum = df_csv['pnl'].cumsum() * 100
        ax3.plot(pd.to_datetime(df_csv['date'].values), csv_cum.values,
                 color=color, linewidth=2, label=f'{sn} CSV', linestyle='-')

    if lp['n'] > 0:
        df_live = lp['trades_df'].sort_values('date')
        live_cum = df_live['pnl'].cumsum() * 100
        ax3.plot(pd.to_datetime(df_live['date'].values), live_cum.values,
                 color=color, linewidth=2, label=f'{sn} Live', linestyle='--')

ax3.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax3.set_ylabel('Cumulative PnL (%)')
ax3.set_title('Cumulative PnL — CSV (solid) vs Live (dashed)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=7, ncol=2, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# ── Chart 4: Probability Scatter — CSV vs Live ──
ax4 = axes[1, 1]
for ticker in sorted(SHORT_TICKERS):
    csv_all = {}
    live_all = {}
    for sn in STRATEGIES_TO_TEST:
        csv_d = csv_scored_probs.get(sn, {}).get(ticker, {})
        live_d = live_scored_probs.get(sn, {}).get(ticker, {})
        for d in set(csv_d.keys()) & set(live_d.keys()):
            csv_all[d] = csv_d[d]
            live_all[d] = live_d[d]

    if csv_all:
        common = sorted(csv_all.keys())
        c_vals = [csv_all[d] for d in common]
        l_vals = [live_all[d] for d in common]
        ax4.scatter(c_vals, l_vals, alpha=0.4, s=20,
                    color=TICKER_COLORS.get(ticker, '#666'), label=ticker)

ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='y=x')
ax4.set_xlabel('CSV Prob')
ax4.set_ylabel('Live TF Prob')
ax4.set_title('Probability Correlation — CSV vs Live', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9, loc='upper left')
ax4.set_xlim(-0.02, 1.02)
ax4.set_ylim(-0.02, 1.02)
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')

fig.suptitle(f'SHORT Dual Backtest — CSV vs Live TF Scoring\n'
             f'{TEST_START} → {TEST_END}  |  '
             f'{", ".join(STRATEGIES_TO_TEST)}  |  '
             f'{len(SHORT_TICKERS)} tickers',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/tmp/backtest_short_dual.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Chart saved: /tmp/backtest_short_dual.png")


# ── Trade Log ──
print(f"\n{'='*W}")
print(f"  TRADE LOG — CSV vs Live Side-by-Side")
print(f"{'='*W}")

for sn in STRATEGIES_TO_TEST:
    print(f"\n  ── {sn} ──")
    cp = csv_results[sn]
    lp = live_results[sn]

    csv_trades = {(t['ticker'], t['date']): t for t in (cp['trades_df'].to_dict('records') if cp['n'] > 0 else [])}
    live_trades = {(t['ticker'], t['date']): t for t in (lp['trades_df'].to_dict('records') if lp['n'] > 0 else [])}

    all_keys = sorted(set(csv_trades.keys()) | set(live_trades.keys()), key=lambda x: (x[1], x[0]))
    for key in all_keys[:30]:
        ticker, d = key
        ct = csv_trades.get(key)
        lt = live_trades.get(key)
        csv_str = f"CSV={ct['pnl']:+.2%} p={ct['prob']:.3f}" if ct else "CSV=—"
        live_str = f"LIVE={lt['pnl']:+.2%} p={lt['prob']:.3f}" if lt else "LIVE=—"
        match = "✓" if ct and lt else "✗"
        print(f"    {d}  {ticker:<7}  {csv_str:<25}  {live_str:<26}  {match}")

    if len(all_keys) > 30:
        print(f"    ... ({len(all_keys) - 30} more trades)")


print(f"\n{'='*W}")
print(f"  DONE — CSV vs Live TF Dual Backtest Complete")
print(f"  Chart: /tmp/backtest_short_dual.png")
print(f"{'='*W}")
