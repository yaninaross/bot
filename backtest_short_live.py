"""
backtest_short_live.py — Live TF Scoring SHORT Model Backtest (Exp 400)
========================================================================
Backtests SHORT/dump model using live 15-seed ensemble TF inference.
Top 5 tickers from cross-alpha selection: SMR, MVIS, NTLA, SOUN, MARA

Loads configs from BQ (robust_wf_v12_short / robust_wf_v12_short_a1).
Supports static, atr_sl, and grace modes (SHORT trade direction).

Capital allocation: each day, total capital is divided equally among
all trades triggered that day. Avg PnL and total PnL ($) reported.

GPU required for inference speed. Colab-ready (no if __name__).
"""

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
N_ASSETS        = 31       # 30 teachers + 1 other (from trained model embedding)
OTHER_IDX       = 30
PST             = pytz.timezone('America/Los_Angeles')

TEST_START      = '2026-04-01'
TEST_END        = '2026-07-23'
EXIT_TIME_MIN   = 780
SLIPPAGE_PCT    = 0.002    # 0.20% each way — shorts face wider fills
STARTING_CAPITAL = 200_000

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]
ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

SHORT_TICKERS = ['SMR', 'MVIS', 'NTLA', 'SOUN', 'MARA']

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1')
BQ_ALLOWED_MODES  = ('static', 'atr_sl', 'grace')
MIN_BQ_TRADES = 3
MAX_VA_OPT_RATIO = 3.0

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
# DATA LOADING
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
# CSV COVERAGE — identifies valid trading days per ticker
# ==============================================================================
def load_seed_scores(ticker, experiment_base):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    bucket = gcs_client.bucket(BUCKET_NAME)
    all_results = {}
    prefix = f"{RESULTS_DIR}/Universal_Exp{experiment_base}/{clean}/Seed_"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        return {}
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
        all_results[seed_id] = d
    return all_results


def build_csv_coverage(ticker, base_exp, entry_min):
    seed_scores = load_seed_scores(ticker, base_exp)
    if not seed_scores:
        return set()
    valid_days = set()
    for seed_id, sdf in seed_scores.items():
        bar_mins = sdf.index.hour * 60 + sdf.index.minute
        entry_bars = sdf[bar_mins == entry_min]
        for dt in entry_bars.index:
            valid_days.add(dt.date())
    return valid_days


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
# MODEL ARCHITECTURE (standard transformer — same as Exp 185)
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
# OHLCV ARRAYS (for trade sim)
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
        if sl_hit:
            return -sl, ed, True
        if tp_hit:
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
        tp_hit = lows[k] <= target_price
        sl_hit = highs[k] >= stop_price
        if sl_hit and tp_hit:
            return (tp_pct if np.random.random() < 0.5 else -sl), ed, True
        if sl_hit:
            return -sl, ed, True
        if tp_hit:
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
        elapsed = time_mins[k] - et
        sl_active = elapsed >= grace_minutes
        tp_hit = lows[k] <= target_price
        sl_hit = sl_active and (highs[k] >= stop_price)
        if sl_hit and tp_hit:
            return (tp if np.random.random() < 0.5 else -sl), ed, True
        if sl_hit:
            return -sl, ed, True
        if tp_hit:
            return tp, ed, True
        if time_mins[k] >= exit_time:
            return (entry_price - prices[k]) / entry_price, ed, True
    return (entry_price - prices[n-1]) / entry_price, ed, True


# ==============================================================================
# BQ CONFIG LOADING
# ==============================================================================
_MODE_SUFFIXES = ('_ATRSL', '_GRACE', '_LIMIT', '_ATR', '_ATRADP',
                  '_REGIME', '_TRAIL', '_REGADP')

def _base_experiment(exp):
    for sfx in sorted(_MODE_SUFFIXES, key=len, reverse=True):
        if exp.endswith(sfx):
            return exp[:-len(sfx)]
    return exp


def load_short_configs():
    """Pick best ensemble config per ticker across all SHORT experiments (400, 410)."""
    versions_sql = ', '.join(f"'{v}'" for v in BQ_MODEL_VERSIONS)
    tickers_sql  = ', '.join(f"'{t}'" for t in SHORT_TICKERS)
    q = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           grace_minutes,
           val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
           val_b_avg_pnl, opt_avg_pnl, is_ensemble, n_seeds_avg
    FROM `{BQ_TABLE}`
    WHERE model_version IN ({versions_sql})
      AND is_ensemble = TRUE
      AND survived_a = TRUE
      AND val_a_count >= {MIN_BQ_TRADES}
      AND val_a_avg_pnl > 0
      AND ticker IN ({tickers_sql})
    """
    df = bq_client.query(q).to_dataframe()
    if df.empty:
        raise RuntimeError("No BQ ensemble rows found for short model tickers")
    print(f"    {len(df)} ensemble BQ rows loaded")

    df = df[df['opt_avg_pnl'] > 0].copy()
    df = df[df['val_a_avg_pnl'] / df['opt_avg_pnl'] <= MAX_VA_OPT_RATIO]
    print(f"    {len(df)} after overfit filter")

    df = df[df['tpsl_mode'].isin(BQ_ALLOWED_MODES)]
    print(f"    {len(df)} after mode filter ({', '.join(BQ_ALLOWED_MODES)})")

    def _safe_float(val, default=0.0):
        try:
            if pd.isna(val): return default
        except (TypeError, ValueError): pass
        try: return float(val)
        except (TypeError, ValueError): return default

    def _safe_int(val, default=0):
        try:
            if pd.isna(val): return default
        except (TypeError, ValueError): pass
        try: return int(val)
        except (TypeError, ValueError): return default

    configs = {}
    for ticker in SHORT_TICKERS:
        tk = df[df['ticker'] == ticker].copy()
        if tk.empty:
            print(f"    {ticker}: no ensemble rows found")
            continue

        best_row = tk.loc[tk['val_a_avg_pnl'].idxmax()]
        mode = best_row['tpsl_mode']
        exp = best_row['experiment']
        cfg = {
            'ticker': ticker,
            'experiment': exp,
            'alpha': exp.split('_')[0],
            'source': 'ensemble',
            'tpsl_mode': mode,
            'entry_minute': _safe_int(best_row['entry_minute'], 375),
            'prob_thresh': _safe_float(best_row['prob_thresh']),
            'tp_pct': _safe_float(best_row['tp_pct']),
            'sl_pct': _safe_float(best_row['sl_pct']),
            'sl_mult': _safe_float(best_row.get('sl_mult', 0)),
            'grace_minutes': _safe_int(best_row.get('grace_minutes', 0)),
            'val_pnl': _safe_float(best_row['val_a_avg_pnl']),
            'n_seeds': _safe_int(best_row.get('n_seeds_avg', 15)),
        }
        configs[ticker] = cfg
        print(f"    {ticker}: exp={exp} {mode} prob>{cfg['prob_thresh']:.2f} "
              f"TP={cfg['tp_pct']:.0%} SL={cfg['sl_pct']:.0%} sl_m={cfg['sl_mult']:.1f} "
              f"val_pnl={cfg['val_pnl']:.2%}")

    return configs


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT MODEL BACKTEST — Live TF Scoring (Exp 400/410)")
print(f"  TEST: {TEST_START} → {TEST_END}")
print(f"  TICKERS: {', '.join(sorted(SHORT_TICKERS))}")
print(f"  EXIT: 1:00 PM ({EXIT_TIME_MIN} min)")
print(f"  SLIPPAGE: {SLIPPAGE_PCT:.2%} each way  |  CAPITAL: ${STARTING_CAPITAL:,}")
print(f"{'='*W}")


# ── Step 1: Load BQ configs ──
print(f"\n  STEP 1: Loading BQ configs...")
deploy_configs = load_short_configs()
if not deploy_configs:
    raise RuntimeError("No deployable configs found")
print(f"  → {len(deploy_configs)} tickers configured")


# ── Step 2: Derive seed lists + load scaling stats per base experiment ──
base_experiments_used = set()
for ticker, cfg in deploy_configs.items():
    base_exp = _base_experiment(cfg['experiment'])
    cfg['_base_experiment'] = base_exp
    alpha = base_exp.split('_')[0]
    seed_base = int(alpha) * 100
    cfg['_seed_list'] = list(range(seed_base + 1, seed_base + 16))
    base_experiments_used.add(base_exp)

print(f"\n  STEP 2: Loading scaling stats...")
scaling_stats = {}
for base_exp in sorted(base_experiments_used):
    blob = gcs_client.bucket(BUCKET_NAME).blob(
        f"{INFERENCE_DIR}/scaling_stats_experiment_{base_exp}.json")
    scaling_stats[base_exp] = json.loads(blob.download_as_text())
    print(f"    {base_exp}: {len(scaling_stats[base_exp])} features")

# ── Step 2b: Build CSV coverage sets (valid trading days per ticker) ──
print(f"\n  STEP 2b: Building CSV coverage sets...")
_cov_start = pd.Timestamp(TEST_START).date()
_cov_end = pd.Timestamp(TEST_END).date()
csv_valid_days = {}
for ticker, cfg in sorted(deploy_configs.items()):
    base_exp = cfg['_base_experiment']
    entry_min = cfg['entry_minute']
    valid = build_csv_coverage(ticker, base_exp, entry_min)
    csv_valid_days[ticker] = valid
    test_valid = {d for d in valid if _cov_start <= d <= _cov_end}
    print(f"    {ticker}/{base_exp}: {len(test_valid)} valid days in test window "
          f"(of {len(valid)} total)")
    gc.collect()

# ── Step 3: Load models (15-seed ensemble per base experiment) ──
print(f"\n  STEP 3: Loading TF models...")
for base_exp in sorted(base_experiments_used):
    alpha = base_exp.split('_')[0]
    seed_base = int(alpha) * 100
    seeds = list(range(seed_base + 1, seed_base + 16))
    n_loaded = 0
    for s in seeds:
        try:
            get_model(s, base_exp)
            n_loaded += 1
        except Exception as e:
            print(f"    {base_exp} seed {s} FAILED: {e}")
    print(f"    {base_exp}: {n_loaded}/15 models loaded")
    if n_loaded == 0:
        raise RuntimeError(f"No models loaded for {base_exp} — check GCS")


# ── Step 4: Build market context + engineer features ──
print(f"\n  STEP 4: Building market context + engineering features...")
market_df = build_market_context()
print(f"    Market context built: {len(market_df)} bars")

ticker_data = {}
for ticker in sorted(deploy_configs.keys()):
    df_raw = engineer_asset(ticker, market_df)
    if df_raw is None:
        print(f"    {ticker}: SKIP — no data")
        continue
    base_exp = deploy_configs[ticker]['_base_experiment']
    df_scaled = apply_scaling(df_raw, scaling_stats[base_exp])
    ohlcv = build_ohlcv_arrays(df_raw)
    atr_pct = compute_daily_atr_pct(df_raw)
    atr_vals = atr_pct.fillna(0.05)
    ohlcv['atr_pct'] = atr_vals.values.astype(np.float64)
    ticker_data[ticker] = {
        'df_scaled': df_scaled, 'df_raw': df_raw, 'ohlcv': ohlcv,
        'cfg': deploy_configs[ticker],
    }
    n_days = len(np.unique(ohlcv['dates']))
    print(f"    {ticker}: {n_days} trading days, {len(df_scaled)} total bars")

del market_df
gc.collect()
print(f"  Data loaded for {len(ticker_data)} tickers")


# ── Step 5: Get trading days ──
test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
all_dates = set()
for ticker, td in ticker_data.items():
    o = td['ohlcv']
    for d_int in np.unique(o['dates']):
        d_str = str(int(d_int))
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if test_start_dt <= d_date <= test_end_dt and d_date.weekday() < 5:
            all_dates.add(d_date)
trading_days = sorted(all_dates)
print(f"  Trading days: {len(trading_days)} ({trading_days[0]} → {trading_days[-1]})")


# ── Step 6: Score and simulate SHORT trades ──
print(f"\n  STEP 5: Scoring + simulating SHORT trades...")

all_trades = []
per_ticker_trades = defaultdict(list)
daily_trades = defaultdict(list)
n_scored = 0
n_signals = 0
n_skipped_coverage = 0

for di, day in enumerate(trading_days):
    for ticker, td in ticker_data.items():
        if day not in csv_valid_days.get(ticker, set()):
            n_skipped_coverage += 1
            continue
        cfg = td['cfg']
        result = score_ticker(ticker, cfg, td['df_scaled'], td['df_raw'],
                              target_date=day)
        if result is None:
            continue
        n_scored += 1
        if not result['signal']:
            continue
        n_signals += 1

        o = td['ohlcv']
        entry_idx = result['row_idx']
        mode = cfg['tpsl_mode']

        if mode == 'atr_sl':
            atr_val = float(o['atr_pct'][entry_idx]) if entry_idx < len(o['atr_pct']) else 0.05
            pnl, trade_date, filled = run_short_trade_atr_sl(
                o['prices'], o['highs'], o['lows'], o['dates'], o['times'],
                entry_idx, cfg['tp_pct'], atr_val, cfg['sl_mult'], EXIT_TIME_MIN)
        elif mode == 'grace':
            pnl, trade_date, filled = run_short_trade_grace(
                o['prices'], o['highs'], o['lows'], o['dates'], o['times'],
                entry_idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
                cfg['grace_minutes'])
        else:
            pnl, trade_date, filled = run_short_trade_static(
                o['prices'], o['highs'], o['lows'], o['dates'], o['times'],
                entry_idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)

        if not filled:
            continue

        pnl -= 2 * SLIPPAGE_PCT

        trade = {
            'date': day, 'ticker': ticker, 'pnl': pnl,
            'prob': result['prob'], 'close': result['close'],
            'mode': mode, 'tp': cfg['tp_pct'],
            'sl': cfg['sl_pct'] if mode != 'atr_sl' else cfg['sl_mult'],
        }
        all_trades.append(trade)
        per_ticker_trades[ticker].append(trade)
        daily_trades[day].append(trade)

    if (di + 1) % 10 == 0 or di == len(trading_days) - 1:
        print(f"    Day {di+1}/{len(trading_days)}: {day}  "
              f"({n_scored} scored, {n_signals} signals, {len(all_trades)} trades, "
              f"{n_skipped_coverage} skipped/no-csv)")

_release_memory()


# ==============================================================================
# RESULTS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  RESULTS — SHORT MODEL LIVE SCORING ({TEST_START} → {TEST_END})")
print(f"{'='*W}")
print(f"  Scored: {n_scored}  |  Signals: {n_signals}  |  Filled: {len(all_trades)}  |  "
      f"Skipped (no CSV): {n_skipped_coverage}")

if not all_trades:
    print("\n  No trades — nothing to report.")
else:
    df_all = pd.DataFrame(all_trades).sort_values('date')

    # ── Avg PnL (simple per-trade average) ──
    avg_pnl = df_all['pnl'].mean()
    total_pnl_pct = df_all['pnl'].sum()
    wr = (df_all['pnl'] > 0).mean()
    cum = df_all['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()

    print(f"\n  Portfolio Summary:")
    print(f"    Trades:    {len(df_all)}")
    print(f"    Total PnL: {total_pnl_pct:+.1%} (simple sum)")
    print(f"    Avg PnL:   {avg_pnl:+.2%} per trade")
    print(f"    Win Rate:  {wr:.0%}")
    print(f"    Max DD:    {max_dd:+.2%}")

    # ── Total PnL ($) — capital divided among daily trades ──
    equity = STARTING_CAPITAL
    equity_curve = []
    for day in sorted(daily_trades.keys()):
        day_t = daily_trades[day]
        n_today = len(day_t)
        if n_today == 0:
            continue
        pos_size = equity / n_today
        day_dollar_pnl = sum(pos_size * t['pnl'] for t in day_t)
        equity += day_dollar_pnl
        equity_curve.append({'date': day, 'equity': equity, 'n_trades': n_today,
                             'day_pnl': day_dollar_pnl})

    df_equity = pd.DataFrame(equity_curve)
    total_dollar_pnl = equity - STARTING_CAPITAL
    total_return = total_dollar_pnl / STARTING_CAPITAL

    print(f"\n  Capital Allocation (all capital / N trades per day):")
    print(f"    Starting:  ${STARTING_CAPITAL:,.0f}")
    print(f"    Final:     ${equity:,.0f}")
    print(f"    Total P&L: ${total_dollar_pnl:+,.0f} ({total_return:+.1%})")

    # ── Monthly breakdown ──
    df_all['month'] = pd.to_datetime(df_all['date']).dt.strftime('%Y-%m')
    print(f"\n  Monthly Breakdown:")
    print(f"  {'Month':<8}  {'N':>4}  {'Total':>8}  {'Avg':>7}  {'WR':>5}  {'$ PnL':>10}")
    print(f"  {'─'*50}")
    for month in sorted(df_all['month'].unique()):
        m_df = df_all[df_all['month'] == month]
        m_total = m_df['pnl'].sum()
        m_avg = m_df['pnl'].mean()
        m_wr = (m_df['pnl'] > 0).mean()
        m_dollar = sum(
            e['day_pnl'] for e in equity_curve
            if pd.Timestamp(e['date']).strftime('%Y-%m') == month
        )
        print(f"  {month:<8}  {len(m_df):>4}  {m_total:>+8.1%}  {m_avg:>+7.2%}  "
              f"{m_wr:>5.0%}  ${m_dollar:>+9,.0f}")

    # ── Per-ticker breakdown ──
    print(f"\n  Per-Ticker:")
    print(f"  {'Ticker':<7} {'Mode':<8} {'α':>4} {'N':>4}  {'Total':>8}  {'Avg':>7}  {'WR':>5}  {'Prob>':>6}")
    print(f"  {'─'*60}")
    for ticker in sorted(per_ticker_trades.keys()):
        trades = per_ticker_trades[ticker]
        if not trades:
            continue
        cfg = deploy_configs[ticker]
        df_t = pd.DataFrame(trades)
        total = df_t['pnl'].sum()
        avg = df_t['pnl'].mean()
        wr_t = (df_t['pnl'] > 0).mean()
        print(f"  {ticker:<7} {cfg['tpsl_mode']:<8} {cfg['alpha']:>4} {len(trades):>4}  "
              f"{total:>+8.1%}  {avg:>+7.2%}  {wr_t:>5.0%}  {cfg['prob_thresh']:>6.2f}")


    # ==============================================================================
    # CHARTS
    # ==============================================================================
    print(f"\n  Generating charts...")

    TICKER_COLORS = {
        'SMR': '#2196F3', 'MVIS': '#4CAF50', 'NTLA': '#FF9800',
        'SOUN': '#E91E63', 'MARA': '#9C27B0',
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # ── Chart 1: Cumulative PnL (%) ──
    ax1 = axes[0, 0]
    cum_pnl = df_all['pnl'].cumsum() * 100
    dates_all = pd.to_datetime(df_all['date'].values)
    ax1.plot(dates_all, cum_pnl.values, color='#333333', linewidth=2.5, label='Portfolio')
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax1.fill_between(dates_all, cum_pnl.values, 0,
                     where=cum_pnl.values >= 0, alpha=0.1, color='green')
    ax1.fill_between(dates_all, cum_pnl.values, 0,
                     where=cum_pnl.values < 0, alpha=0.1, color='red')
    ax1.set_ylabel('Cumulative PnL (%)')
    ax1.set_title('SHORT Portfolio — Cumulative PnL (Live Scoring)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # ── Chart 2: Per-Ticker Cumulative PnL ──
    ax2 = axes[0, 1]
    for ticker in sorted(per_ticker_trades.keys()):
        trades = per_ticker_trades[ticker]
        if not trades:
            continue
        df_t = pd.DataFrame(trades).sort_values('date')
        cum_t = df_t['pnl'].cumsum() * 100
        ax2.plot(pd.to_datetime(df_t['date'].values), cum_t.values,
                 label=ticker, color=TICKER_COLORS.get(ticker, '#666'), linewidth=2)
    ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax2.set_ylabel('Cumulative PnL (%)')
    ax2.set_title('Per-Ticker Cumulative PnL', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # ── Chart 3: Equity Curve ($) — capital / N trades per day ──
    ax3 = axes[1, 0]
    eq_dates = pd.to_datetime([e['date'] for e in equity_curve])
    eq_vals = [e['equity'] / 1000 for e in equity_curve]
    ax3.plot(eq_dates, eq_vals, color='#333333', linewidth=2.5)
    ax3.axhline(y=STARTING_CAPITAL / 1000, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax3.set_ylabel('Equity ($K)')
    ax3.set_title(f'Portfolio Equity — ${STARTING_CAPITAL/1000:.0f}K, capital / N daily trades',
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # ── Chart 4: Monthly Avg PnL per Ticker ──
    ax4 = axes[1, 1]
    months = sorted(df_all['month'].unique())
    tickers_with_trades = sorted(per_ticker_trades.keys())
    x = np.arange(len(months))
    bar_width = 0.8 / max(len(tickers_with_trades), 1)
    for i, ticker in enumerate(tickers_with_trades):
        df_t = pd.DataFrame(per_ticker_trades[ticker])
        df_t['month'] = pd.to_datetime(df_t['date']).dt.strftime('%Y-%m')
        monthly_avg = df_t.groupby('month')['pnl'].mean()
        vals = [monthly_avg.get(m, 0) * 100 for m in months]
        ax4.bar(x + i * bar_width, vals, bar_width,
                label=ticker, color=TICKER_COLORS.get(ticker, '#666'), alpha=0.85)
    mlabels = [pd.Timestamp(m + '-01').strftime('%b') for m in months]
    ax4.set_xticks(x + bar_width * len(tickers_with_trades) / 2)
    ax4.set_xticklabels(mlabels, fontsize=11)
    ax4.set_ylabel('Avg PnL per Trade (%)')
    ax4.set_title('Monthly Avg PnL by Ticker', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=8, ncol=2)
    ax4.axhline(y=0, color='gray', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'SHORT Model Live Backtest — {TEST_START} → {TEST_END}  |  '
                 f'{len(all_trades)} trades, Avg={avg_pnl:+.2%}, WR={wr:.0%}, '
                 f'Final=${equity:,.0f}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('/tmp/backtest_short_live.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Chart saved: /tmp/backtest_short_live.png")


    # ── Trade log ──
    print(f"\n  All trades ({len(all_trades)}):")
    for _, t in df_all.iterrows():
        print(f"    {t['date']}  {t['ticker']:<7}  pnl={t['pnl']:+.2%}  "
              f"prob={t['prob']:.3f}  close=${t['close']:.2f}")


print(f"\n{'='*W}")
print(f"  DONE — {len(all_trades)} trades across {len(per_ticker_trades)} tickers")
print(f"  Total PnL: {total_pnl_pct:+.1%}  |  Avg PnL: {avg_pnl:+.2%}  |  "
      f"WR: {wr:.0%}  |  Final: ${equity:,.0f} ({total_return:+.1%})")
print(f"{'='*W}")
