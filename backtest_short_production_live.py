"""
backtest_short_production_live.py — Walk-Forward Production Simulation (Live TF Scoring)
=========================================================================================
Same walk-forward ticker selection as backtest_short_production.py but replaces CSV-scored
probabilities with live 15-seed TF model inference. Validates that live scoring produces
the same PnL as CSV scoring.

Strategies: ML_STATIC, ML_ADAPTIVE, ML_COMBINED
(ORACLE/CURRENT omitted — not needed for live scoring validation)

GPU recommended (15 seeds × 53 tickers × ~80 days = ~63K inferences).
Colab-ready (no if __name__).
"""

import os, io, gc, sys, json, subprocess, warnings, ctypes
from datetime import datetime, timedelta, date as date_type
from collections import defaultdict
import numpy as np, pandas as pd, pytz
from scipy import stats as sp_stats

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

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

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
# CONFIG — identical to backtest_short_production.py
# ==============================================================================
BUCKET_NAME    = "time-series-bucket-ys"
RESULTS_DIR    = "Algos/Results_all_v2"
INFERENCE_DIR  = "Data/Inference_v2"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"

FULL_START = '2026-04-01'
FULL_END   = '2026-07-23'
HALF1_END  = '2026-05-31'
HALF2_START = '2026-06-01'

ENTRY_MIN     = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.002

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1')

TOP_K = 8
FILTERED_VA_OPT_RATIO = 2.0
ROLLING_LOOKBACK_DAYS = 14
BURN_IN_WEEKS = 2
CONSEC_LOSS_DROP = 3
CONSEC_WEEKS_ADD = 3
ROLLING_FEAT_WEIGHT = 0.3

# TF model config
TIME_STEPS  = 240
EMBED_DIM   = 16
N_ASSETS    = 31
OTHER_IDX   = 30

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]
ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110


# ==============================================================================
# CONTEXT COLUMNS
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

def load_master(ticker, apply_cutoff=True):
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
        if apply_cutoff:
            cutoff = pd.Timestamp(FULL_START, tz='UTC') - pd.Timedelta(days=30)
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
# MARKET CONTEXT (from backtest_short_live.py)
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
        if ticker in _master_cache:
            del _master_cache[ticker]
        gc.collect()
    if not frames:
        raise RuntimeError("No context assets loaded.")
    return pd.concat(frames, axis=1, sort=True).ffill().fillna(0)


# ==============================================================================
# FEATURE ENGINEERING (from backtest_short_live.py)
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
# SCALING (from backtest_short_live.py)
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
# MODEL ARCHITECTURE (from backtest_short_live.py — standard transformer)
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
# MODEL LOADER (from backtest_short_live.py)
# ==============================================================================
_model_cache = {}

def get_model(seed, experiment):
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
# LIVE SCORING ENGINE (from backtest_short_live.py)
# ==============================================================================
def score_ticker_live(ticker, cfg, df_scaled, ohlcv, target_date):
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
        'close': round(float(ohlcv['prices'][row_idx]), 4),
    }


# ==============================================================================
# OHLCV ARRAYS (from backtest_short_live.py)
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
# SHORT TRADE SIMULATION (identical to backtest_short_production.py)
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
# HELPERS (from backtest_short_production.py)
# ==============================================================================
_MODE_SUFFIXES = ('_ATRSL', '_GRACE', '_LIMIT', '_ATR', '_ATRADP',
                  '_REGIME', '_TRAIL', '_REGADP')

def _base_experiment(exp):
    for sfx in sorted(_MODE_SUFFIXES, key=len, reverse=True):
        if exp.endswith(sfx):
            return exp[:-len(sfx)]
    return exp

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

def _row_to_cfg(ticker, row, source):
    exp = row['experiment'] if isinstance(row, pd.Series) else row.get('experiment', '')
    base_exp = _base_experiment(exp)
    return {
        'ticker': ticker,
        'experiment': exp,
        '_base_experiment': base_exp,
        'alpha': base_exp.split('_')[0],
        'source': source,
        'tpsl_mode': row['tpsl_mode'],
        'entry_minute': _safe_int(row.get('entry_minute', 375), 375),
        'prob_thresh': _safe_float(row['prob_thresh']),
        'tp_pct': _safe_float(row.get('tp_pct', 0)),
        'sl_pct': _safe_float(row.get('sl_pct', 0)),
        'sl_mult': _safe_float(row.get('sl_mult', 0)),
        'grace_minutes': _safe_int(row.get('grace_minutes', 0)),
        'val_pnl': _safe_float(row.get('val_a_avg_pnl', row.get('mean_va_pnl', 0))),
        'test_pnl': _safe_float(row.get('val_b_avg_pnl', row.get('vb_pnl', 0))),
    }


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
# BQ + CONFIG SELECTION (from backtest_short_production.py)
# ==============================================================================
def query_all_bq():
    versions_sql = ', '.join(f"'{v}'" for v in BQ_MODEL_VERSIONS)
    q = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           grace_minutes,
           opt_avg_pnl, opt_win_rate, opt_count,
           val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
           val_b_avg_pnl, val_b_win_rate, val_b_count,
           is_ensemble, survived_a, model_version
    FROM `{BQ_TABLE}`
    WHERE model_version IN ({versions_sql})
      AND survived_a = TRUE
      AND val_a_count >= 3
      AND opt_avg_pnl > 0
      AND val_a_avg_pnl > 0
    ORDER BY ticker, experiment, seed
    """
    return bq_client.query(q).to_dataframe()


def select_configs_ensemble(bq_df, ticker_list):
    ens = bq_df[bq_df['is_ensemble'] == True].copy()
    if ens.empty:
        ens = bq_df[bq_df['seed'] == -1].copy()
    ens = ens[ens['tpsl_mode'].isin(['static', 'atr_sl', 'grace'])]
    ens = ens[ens['val_a_avg_pnl'] / ens['opt_avg_pnl'] <= FILTERED_VA_OPT_RATIO]
    configs = {}
    for ticker in ticker_list:
        tk = ens[ens['ticker'] == ticker]
        if tk.empty:
            continue
        best = tk.loc[tk['val_a_avg_pnl'].idxmax()]
        configs[ticker] = _row_to_cfg(ticker, best, 'ensemble')
    return configs


# ==============================================================================
# FEATURE ENGINEERING FOR DECISION TREE (from backtest_short_production.py)
# ==============================================================================
def compute_ticker_features(bq_df):
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = bq_df[bq_df['is_ensemble'] != True].copy()

    indiv['va_opt_ratio'] = indiv['val_a_avg_pnl'] / indiv['opt_avg_pnl']

    rows = []
    for ticker, grp in indiv.groupby('ticker'):
        n = len(grp)
        filt = grp[(grp['tpsl_mode'] != 'limit_entry') &
                    (grp['va_opt_ratio'] <= FILTERED_VA_OPT_RATIO)]

        seed_grp_keys = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
                         'prob_thresh', 'tp_pct', 'sl_pct', 'sl_mult', 'grace_minutes']
        for k in seed_grp_keys:
            if k in grp.columns and grp[k].isna().any():
                grp[k] = grp[k].fillna(0)
        seed_agg = grp.groupby(seed_grp_keys).agg(
            n_seeds=('seed', 'nunique'),
            mean_va_pnl=('val_a_avg_pnl', 'mean'),
        ).reset_index()
        seed_agg['agree_18'] = 0.0
        for idx_sa, row_sa in seed_agg.iterrows():
            mask = True
            for k in seed_grp_keys:
                mask = mask & (grp[k] == row_sa[k])
            sub = grp[mask]
            n_pos = (sub['val_a_avg_pnl'] >= 0.018).sum()
            seed_agg.at[idx_sa, 'agree_18'] = n_pos / row_sa['n_seeds'] if row_sa['n_seeds'] > 0 else 0
        seed_agg['score_pnl2'] = seed_agg['mean_va_pnl'] ** 2 * seed_agg['agree_18']

        exps = grp['experiment'].apply(_base_experiment).unique()
        exp_400 = [e for e in exps if e.startswith('400')]
        exp_410 = [e for e in exps if e.startswith('410')]
        has_both = len(exp_400) > 0 and len(exp_410) > 0

        cross_agree = 0.0
        if has_both:
            a400 = seed_agg[seed_agg['ticker'].isin([ticker]) &
                            seed_agg['experiment'].apply(lambda e: _base_experiment(e).startswith('400'))]
            a410 = seed_agg[seed_agg['ticker'].isin([ticker]) &
                            seed_agg['experiment'].apply(lambda e: _base_experiment(e).startswith('410'))]
            best_400_agree = a400['agree_18'].max() if not a400.empty else 0
            best_410_agree = a410['agree_18'].max() if not a410.empty else 0
            cross_agree = min(best_400_agree, best_410_agree)

        row = {
            'ticker': ticker,
            'n_configs': n,
            'n_filt': len(filt),
            'pct_limit': (grp['tpsl_mode'] == 'limit_entry').sum() / n,
            'pct_static': (grp['tpsl_mode'] == 'static').sum() / n,
            'pct_atr_sl': (grp['tpsl_mode'] == 'atr_sl').sum() / n,
            'pct_grace': (grp['tpsl_mode'] == 'grace').sum() / n,
            'va_median': grp['val_a_avg_pnl'].median(),
            'va_mean': grp['val_a_avg_pnl'].mean(),
            'va_best': grp['val_a_avg_pnl'].max(),
            'va_worst': grp['val_a_avg_pnl'].min(),
            'va_spread': grp['val_a_avg_pnl'].max() - grp['val_a_avg_pnl'].min(),
            'va_std': grp['val_a_avg_pnl'].std(),
            'va_pos_rate': (grp['val_a_avg_pnl'] >= 0.01).mean(),
            'va_wr_median': grp['val_a_win_rate'].median(),
            'va_count_median': grp['val_a_count'].median(),
            'opt_median': grp['opt_avg_pnl'].median(),
            'opt_best': grp['opt_avg_pnl'].max(),
            'opt_wr_median': grp['opt_win_rate'].median(),
            'opt_count_median': grp['opt_count'].median(),
            'va_opt_ratio_med': grp['va_opt_ratio'].median(),
            'va_opt_ratio_std': grp['va_opt_ratio'].std(),
            'retention': grp['val_a_avg_pnl'].median() / grp['opt_avg_pnl'].median()
                         if grp['opt_avg_pnl'].median() > 0 else 0,
            'agree_18_best': seed_agg['agree_18'].max() if not seed_agg.empty else 0,
            'agree_18_median': seed_agg['agree_18'].median() if not seed_agg.empty else 0,
            'score_pnl2_best': seed_agg['score_pnl2'].max() if not seed_agg.empty else 0,
            'has_both_alphas': int(has_both),
            'cross_alpha_agree': cross_agree,
        }

        if len(filt) > 0:
            row['filt_va_median'] = filt['val_a_avg_pnl'].median()
            row['filt_va_best'] = filt['val_a_avg_pnl'].max()
            row['filt_va_pos_rate'] = (filt['val_a_avg_pnl'] >= 0.01).mean()
        else:
            row['filt_va_median'] = 0
            row['filt_va_best'] = 0
            row['filt_va_pos_rate'] = 0

        rows.append(row)

    features_df = pd.DataFrame(rows).set_index('ticker')
    features_df = features_df.fillna(0)
    return features_df


def compute_target(bq_df):
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = bq_df[bq_df['is_ensemble'] != True].copy()

    target = indiv.groupby('ticker').agg(
        vb_median=('val_b_avg_pnl', 'median'),
        vb_pos_rate=('val_b_avg_pnl', lambda x: (x > 0).mean()),
    )
    return target


# ==============================================================================
# WALK-FORWARD UTILITIES (from backtest_short_production.py)
# ==============================================================================
def generate_weeks(start_str, end_str):
    start = pd.Timestamp(start_str).date()
    end = pd.Timestamp(end_str).date()

    weeks = []
    current = start
    while current <= end:
        if current.weekday() <= 4:
            week_monday = current - timedelta(days=current.weekday())
            if week_monday < start:
                week_monday = start
            week_friday = week_monday + timedelta(days=4)
            if week_friday > end:
                week_friday = end
            week_entry = (week_monday, week_friday)
            if not weeks or weeks[-1] != week_entry:
                weeks.append(week_entry)
            current = week_friday + timedelta(days=3)
        else:
            current += timedelta(days=1)

    return weeks


def add_rolling_features(features_df, trade_history, cutoff_date, lookback_days=14):
    lookback_start = cutoff_date - timedelta(days=lookback_days)

    rolling_feats = {}
    for ticker in features_df.index:
        trades = trade_history.get(ticker, [])
        recent = [t for t in trades
                  if lookback_start <= t['date'] < cutoff_date]

        if recent:
            pnls = [t['pnl'] for t in recent]
            rolling_feats[ticker] = {
                'rolling_pnl': np.mean(pnls),
                'rolling_wr': np.mean([1 if p > 0 else 0 for p in pnls]),
                'rolling_n_trades': len(recent),
                'rolling_active': 1.0,
            }
        else:
            rolling_feats[ticker] = {
                'rolling_pnl': 0.0,
                'rolling_wr': 0.0,
                'rolling_n_trades': 0.0,
                'rolling_active': 0.0,
            }

    rolling_df = pd.DataFrame(rolling_feats).T
    rolling_df.index.name = 'ticker'
    return features_df.join(rolling_df, how='left').fillna(0)


def add_expanding_features(features_df, trade_history, cutoff_date):
    rolling_feats = {}
    for ticker in features_df.index:
        trades = trade_history.get(ticker, [])
        past = [t for t in trades if t['date'] < cutoff_date]

        if past:
            pnls = [t['pnl'] for t in past]
            rolling_feats[ticker] = {
                'rolling_pnl': np.mean(pnls),
                'rolling_wr': np.mean([1 if p > 0 else 0 for p in pnls]),
                'rolling_n_trades': len(past),
                'rolling_active': 1.0,
            }
        else:
            rolling_feats[ticker] = {
                'rolling_pnl': 0.0,
                'rolling_wr': 0.0,
                'rolling_n_trades': 0.0,
                'rolling_active': 0.0,
            }

    rolling_df = pd.DataFrame(rolling_feats).T
    rolling_df.index.name = 'ticker'
    return features_df.join(rolling_df, how='left').fillna(0)


def train_and_select(features_df, y, top_k):
    X = features_df.values.astype(np.float64)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X_s, y)
    preds = model.predict(X_s)

    pred_df = pd.DataFrame({
        'ticker': features_df.index,
        'pred_vb': preds,
    }).sort_values('pred_vb', ascending=False)

    selected = pred_df.head(top_k)['ticker'].tolist()
    pred_dict = dict(zip(pred_df['ticker'], pred_df['pred_vb']))
    return selected, pred_dict


def train_and_select_downweighted(features_df, y, top_k, n_bq_features,
                                   weight=ROLLING_FEAT_WEIGHT):
    X = features_df.values.astype(np.float64).copy()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s[:, n_bq_features:] *= weight

    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X_s, y)
    preds = model.predict(X_s)

    pred_df = pd.DataFrame({
        'ticker': features_df.index,
        'pred_vb': preds,
    }).sort_values('pred_vb', ascending=False)

    selected = pred_df.head(top_k)['ticker'].tolist()
    pred_dict = dict(zip(pred_df['ticker'], pred_df['pred_vb']))
    return selected, pred_dict


def apply_sticky_rules(current_active, model_selected, trade_history, cutoff_date,
                       candidate_streak, all_configs, ticker_data):
    result = list(current_active)

    for ticker in list(result):
        trades = trade_history.get(ticker, [])
        recent = sorted([t for t in trades if t['date'] < cutoff_date],
                        key=lambda t: t['date'])
        if len(recent) >= CONSEC_LOSS_DROP:
            tail = recent[-CONSEC_LOSS_DROP:]
            if all(t['pnl'] <= 0 for t in tail):
                result.remove(ticker)

    valid_model = [t for t in model_selected
                   if t in all_configs and t in ticker_data]

    for ticker in valid_model:
        if ticker not in result:
            streak = candidate_streak.get(ticker, 0)
            if streak >= CONSEC_WEEKS_ADD:
                result.append(ticker)

    return result


# ==============================================================================
# DAILY TRADE EXECUTION — LIVE SCORING VERSION
# ==============================================================================
def run_single_day_trade_live(ticker, trade_date, cfg, td, scaling_stats_map):
    """Score ticker live with TF models, then simulate trade if signal fires."""
    base_exp = cfg['_base_experiment']
    df_scaled = td['df_scaled_map'].get(base_exp)
    ohlcv = td['ohlcv']

    if df_scaled is None:
        return None

    result = score_ticker_live(ticker, cfg, df_scaled, ohlcv, trade_date)
    if result is None:
        return None
    if not result['signal']:
        return None

    entry_idx = result['row_idx']
    mode = cfg['tpsl_mode']
    prices = ohlcv['prices']
    highs  = ohlcv['highs']
    lows   = ohlcv['lows']
    dates  = ohlcv['dates']
    times  = ohlcv['times']

    if mode == 'atr_sl':
        atr_arr = ohlcv.get('atr_pct', np.full(len(prices), 0.05))
        atr_val = float(atr_arr[entry_idx]) if entry_idx < len(atr_arr) else 0.05
        pnl, _, filled = run_short_trade_atr_sl(
            prices, highs, lows, dates, times,
            entry_idx, cfg['tp_pct'], atr_val, cfg['sl_mult'], EXIT_TIME_MIN)
    elif mode == 'grace':
        pnl, _, filled = run_short_trade_grace(
            prices, highs, lows, dates, times,
            entry_idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['grace_minutes'])
    else:
        pnl, _, filled = run_short_trade_static(
            prices, highs, lows, dates, times,
            entry_idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)

    if not filled:
        return None

    pnl -= 2 * SLIPPAGE_PCT

    return {
        'date': trade_date, 'ticker': ticker, 'pnl': pnl,
        'prob': result['prob'], 'close': result['close'],
        'mode': mode, 'tp': cfg['tp_pct'],
        'sl': cfg['sl_pct'] if mode != 'atr_sl' else cfg['sl_mult'],
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT PRODUCTION SIMULATION — Walk-Forward Weekly (LIVE TF SCORING)")
print(f"  3 strategies: ML_STATIC, ML_ADAPTIVE, ML_COMBINED")
print(f"  Test period: {FULL_START} to {FULL_END}")
print(f"  Scoring: 15-seed TF ensemble (live inference)")
print(f"{'='*W}")


# ── PHASE 1: Load BQ + Train DecisionTree ──
print(f"\n  PHASE 1: Loading BQ data + training DecisionTree...")

print(f"    Querying BQ...")
bq_df = query_all_bq()
all_tickers = sorted(bq_df['ticker'].unique())
n_tickers = len(all_tickers)
print(f"    {len(bq_df)} BQ rows across {n_tickers} tickers")

print(f"\n    Computing features + target...")
features_df = compute_ticker_features(bq_df)
target_df = compute_target(bq_df)
combined = features_df.join(target_df, how='inner')
tickers = combined.index.tolist()
feature_cols = [c for c in features_df.columns]
y = combined['vb_median'].values.astype(np.float64)
print(f"    {len(tickers)} tickers × {len(feature_cols)} BQ features")

print(f"\n    Training static DecisionTree model...")
static_tickers, static_preds = train_and_select(
    features_df.loc[tickers], y, TOP_K)
print(f"    ML_STATIC tickers: {', '.join(static_tickers)}")

print(f"\n    Selecting ensemble configs for ALL tickers...")
all_configs = select_configs_ensemble(bq_df, all_tickers)
print(f"    {len(all_configs)} tickers have valid ensemble configs")


# ── PHASE 2: Load TF infrastructure ──
print(f"\n{'='*W}")
print(f"  PHASE 2: Loading TF models + engineering features...")
print(f"{'='*W}")

# Derive seed lists + collect base experiments
base_experiments_used = set()
for ticker, cfg in all_configs.items():
    base_exp = cfg['_base_experiment']
    alpha = base_exp.split('_')[0]
    seed_base = int(alpha) * 100
    cfg['_seed_list'] = list(range(seed_base + 1, seed_base + 16))
    base_experiments_used.add(base_exp)

print(f"    Base experiments: {', '.join(sorted(base_experiments_used))}")

# Load scaling stats per experiment
print(f"\n    Loading scaling stats...")
scaling_stats_map = {}
for base_exp in sorted(base_experiments_used):
    blob = gcs_client.bucket(BUCKET_NAME).blob(
        f"{INFERENCE_DIR}/scaling_stats_experiment_{base_exp}.json")
    scaling_stats_map[base_exp] = json.loads(blob.download_as_text())
    print(f"    {base_exp}: {len(scaling_stats_map[base_exp])} features")

# Load TF models (15 seeds per experiment)
print(f"\n    Loading TF models...")
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
            print(f"      {base_exp} seed {s} FAILED: {e}")
    print(f"    {base_exp}: {n_loaded}/15 models loaded")
    if n_loaded == 0:
        raise RuntimeError(f"No models loaded for {base_exp} — check GCS")

# Build market context
print(f"\n    Building market context...")
market_df = build_market_context()
print(f"    Market context built: {len(market_df)} bars")

# Engineer features + build OHLCV for ALL tickers
# Memory optimization: only keep df_scaled[needed cols] + ohlcv arrays; drop df_raw
print(f"\n    Engineering features for {len(all_configs)} tickers...")
ticker_data = {}
n_loaded = 0
for ticker in sorted(all_configs.keys()):
    cfg = all_configs[ticker]
    df_raw = engineer_asset(ticker, market_df)
    if df_raw is None:
        print(f"      {ticker}: SKIP — no data")
        continue

    base_exp = cfg['_base_experiment']
    df_scaled_full = apply_scaling(df_raw, scaling_stats_map[base_exp])
    df_scaled = df_scaled_full[ASSET_COLS + CONTEXT_COLS].copy()
    del df_scaled_full

    ohlcv = build_ohlcv_arrays(df_raw)
    atr_pct = compute_daily_atr_pct(df_raw)
    atr_vals = atr_pct.fillna(0.05)
    ohlcv['atr_pct'] = atr_vals.values.astype(np.float64)

    ticker_data[ticker] = {
        'df_scaled_map': {base_exp: df_scaled},
        'ohlcv': ohlcv,
    }

    del df_raw, atr_pct, atr_vals
    if ticker in _master_cache:
        del _master_cache[ticker]
    n_loaded += 1
    if n_loaded % 5 == 0:
        _release_memory()
        print(f"      {n_loaded} tickers loaded...")

del market_df
_master_cache.clear()
_release_memory()
print(f"    {n_loaded} tickers ready for live scoring")


# ── PHASE 3: Walk-Forward Weekly Loop ──
weeks = generate_weeks(FULL_START, FULL_END)
n_bq_features = len(feature_cols)
print(f"\n{'='*W}")
print(f"  PHASE 3: Walk-Forward Simulation ({len(weeks)} weeks) — LIVE SCORING")
print(f"{'='*W}")

STRAT_NAMES = ['ML_STATIC', 'ML_ADAPTIVE', 'ML_COMBINED']

strategies = {s: {'trades': [], 'weekly_tickers': []} for s in STRAT_NAMES}

trade_history = {s: defaultdict(list) for s in STRAT_NAMES}
combined_candidate_streak = defaultdict(int)
combined_prev_active = None

trading_days_list = pd.bdate_range(FULL_START, FULL_END).date

print(f"\n  {'Wk':>3} {'Period':<15} │ {'Strategy':<14} │ {'Tk':>3} │ {'Trd':>4} {'PnL':>8} {'Chg':>4}")
print(f"  {'─'*65}")

for week_idx, (wk_start, wk_end) in enumerate(weeks):
    week_days = [d for d in trading_days_list if wk_start <= d <= wk_end]
    if not week_days:
        continue

    # ── ML_STATIC: same tickers every week ──
    static_active = [t for t in static_tickers if t in all_configs and t in ticker_data]

    # ── ML_ADAPTIVE: 14-day rolling window, full retrain (original) ──
    if week_idx < BURN_IN_WEEKS:
        adaptive_active = static_active.copy()
    else:
        aug_feat = add_rolling_features(
            features_df.loc[tickers], trade_history['ML_ADAPTIVE'], wk_start,
            ROLLING_LOOKBACK_DAYS)
        adaptive_sel, _ = train_and_select(aug_feat, y, TOP_K)
        adaptive_active = [t for t in adaptive_sel
                           if t in all_configs and t in ticker_data]

    # ── ML_COMBINED: expanding + sticky hysteresis + downweighted rolling ──
    if week_idx < BURN_IN_WEEKS:
        combined_active = static_active.copy()
        combined_model_sel = static_tickers
    else:
        comb_feat = add_expanding_features(
            features_df.loc[tickers], trade_history['ML_COMBINED'], wk_start)
        combined_model_sel, _ = train_and_select_downweighted(
            comb_feat, y, TOP_K, n_bq_features)
        combined_model_valid = [t for t in combined_model_sel
                                if t in all_configs and t in ticker_data]

        for tk in tickers:
            if tk in combined_model_valid:
                combined_candidate_streak[tk] = combined_candidate_streak.get(tk, 0) + 1
            else:
                combined_candidate_streak[tk] = 0

        base = combined_prev_active if combined_prev_active else static_active
        combined_active = apply_sticky_rules(
            base, combined_model_valid,
            trade_history['ML_COMBINED'], wk_start,
            combined_candidate_streak, all_configs, ticker_data)

    combined_prev_active = combined_active

    strategy_tickers = {
        'ML_STATIC':   static_active,
        'ML_ADAPTIVE': adaptive_active,
        'ML_COMBINED': combined_active,
    }

    for sname in STRAT_NAMES:
        strategies[sname]['weekly_tickers'].append({
            'week': week_idx + 1,
            'start': wk_start,
            'end': wk_end,
            'tickers': strategy_tickers[sname],
        })

    # ── Daily trading with live scoring ──
    week_trades = {s: [] for s in STRAT_NAMES}

    for day in week_days:
        np.random.seed(42 + day.toordinal())

        for sname in STRAT_NAMES:
            active_tks = strategy_tickers[sname]
            for ticker in active_tks:
                if ticker not in ticker_data:
                    continue
                cfg = all_configs[ticker]
                td = ticker_data[ticker]

                trade = run_single_day_trade_live(
                    ticker, day, cfg, td, scaling_stats_map)
                if trade is not None:
                    trade['strategy'] = sname
                    trade['week'] = week_idx + 1
                    week_trades[sname].append(trade)
                    strategies[sname]['trades'].append(trade)
                    trade_history[sname][ticker].append(trade)

    # ── Print week summary ──
    for si, sname in enumerate(STRAT_NAMES):
        n_tk = len(strategy_tickers[sname])
        wt = week_trades[sname]
        n_trades = len(wt)
        week_pnl = sum(t['pnl'] for t in wt)

        chg = ''
        if sname in ('ML_ADAPTIVE', 'ML_COMBINED') and week_idx >= BURN_IN_WEEKS:
            prev_wk = strategies[sname]['weekly_tickers']
            if len(prev_wk) >= 2:
                prev_tks = set(prev_wk[-2]['tickers'])
                curr_tks = set(strategy_tickers[sname])
                n_chg = len(prev_tks.symmetric_difference(curr_tks))
                chg = f"±{n_chg}" if n_chg > 0 else ""

        period_str = f"{wk_start.strftime('%m/%d')}-{wk_end.strftime('%m/%d')}" if si == 0 else ""
        wk_str = f"W{week_idx+1}" if si == 0 else ""
        print(f"  {wk_str:>3} {period_str:<15} │ {sname:<14} │ {n_tk:>3} │ {n_trades:>4} {week_pnl:>+7.1%} {chg:>4}")

    if week_idx < len(weeks) - 1:
        print(f"  {'':>3} {'':15} │ {'':14} │ {'':>3} │")


# ── PHASE 4: Aggregate Results ──
print(f"\n{'='*W}")
print(f"  PHASE 4: Results (LIVE TF SCORING)")
print(f"{'='*W}")

half1_end_dt = pd.Timestamp(HALF1_END).date()
half2_start_dt = pd.Timestamp(HALF2_START).date()

print(f"\n  {'Strategy':<14} │ {'── FULL ──────────────────':} │ {'── HALF1 ──────────':} │ {'── HALF2 ──────────':}")
print(f"  {'':14} │ {'N':>4} {'Tot':>7} {'Avg':>7} {'WR':>5} {'DD':>7} │ {'N':>4} {'Tot':>7} {'Avg':>7} {'WR':>5} │ {'N':>4} {'Tot':>7} {'Avg':>7} {'WR':>5}")
print(f"  {'─'*105}")

strategy_summaries = {}
for sname in STRAT_NAMES:
    all_t = strategies[sname]['trades']
    h1_t = [t for t in all_t if t['date'] <= half1_end_dt]
    h2_t = [t for t in all_t if t['date'] >= half2_start_dt]

    f_sum = summarize_trades(all_t, f"{sname}/FULL")
    h1_sum = summarize_trades(h1_t, f"{sname}/HALF1")
    h2_sum = summarize_trades(h2_t, f"{sname}/HALF2")
    strategy_summaries[sname] = {'FULL': f_sum, 'HALF1': h1_sum, 'HALF2': h2_sum}

    print(f"  {sname:<14} │ {f_sum['n']:>4} {f_sum['total']:>+6.1%} {f_sum['avg']:>+6.2%} {f_sum['wr']:>4.0%} {f_sum['dd']:>+6.1%} │ "
          f"{h1_sum['n']:>4} {h1_sum['total']:>+6.1%} {h1_sum['avg']:>+6.2%} {h1_sum['wr']:>4.0%} │ "
          f"{h2_sum['n']:>4} {h2_sum['total']:>+6.1%} {h2_sum['avg']:>+6.2%} {h2_sum['wr']:>4.0%}")


# ── Per-ticker breakdown for ML_STATIC ──
print(f"\n  ── Per-Ticker (ML_STATIC) ──")
static_trades = strategies['ML_STATIC']['trades']
per_tk = defaultdict(list)
for t in static_trades:
    per_tk[t['ticker']].append(t)

print(f"  {'Ticker':<7} │ {'N':>4} {'Avg':>7} {'WR':>5} │ {'H1_avg':>7} {'H2_avg':>7} {'Status':>8}")
print(f"  {'─'*60}")
for ticker in sorted(per_tk.keys()):
    trades = per_tk[ticker]
    h1 = [t for t in trades if t['date'] <= half1_end_dt]
    h2 = [t for t in trades if t['date'] >= half2_start_dt]
    avg = np.mean([t['pnl'] for t in trades])
    wr = np.mean([1 if t['pnl'] > 0 else 0 for t in trades])
    h1_avg = np.mean([t['pnl'] for t in h1]) if h1 else 0
    h2_avg = np.mean([t['pnl'] for t in h2]) if h2 else 0
    status = "HOLDS" if h1_avg > 0 and h2_avg > 0 else ("DECAYED" if h1_avg > 0 else "WEAK")
    print(f"  {ticker:<7} │ {len(trades):>4} {avg:>+6.2%} {wr:>4.0%} │ {h1_avg:>+6.2%} {h2_avg:>+6.2%} {status:>8}")


# ── Weekly ticker changes for adaptive strategies ──
for adapt_name in ['ML_ADAPTIVE', 'ML_COMBINED']:
    print(f"\n  ── {adapt_name}: Weekly Ticker Selection ──")
    for wk_info in strategies[adapt_name]['weekly_tickers']:
        tks = sorted(wk_info['tickers'])
        prev_idx = wk_info['week'] - 2
        if prev_idx >= 0:
            prev_tks = set(strategies[adapt_name]['weekly_tickers'][prev_idx]['tickers'])
            added = sorted(set(tks) - prev_tks)
            removed = sorted(prev_tks - set(tks))
            delta = ""
            if added or removed:
                delta = f"  [+{','.join(added)}]" if added else ""
                delta += f"  [-{','.join(removed)}]" if removed else ""
        else:
            delta = ""
        print(f"    W{wk_info['week']:>2} ({wk_info['start'].strftime('%m/%d')}-{wk_info['end'].strftime('%m/%d')}): "
              f"{', '.join(tks)}{delta}")


# ── Monthly summary ──
print(f"\n  ── Monthly Summary ──")
months = sorted(set(t['date'].strftime('%Y-%m') for s in strategies.values() for t in s['trades']))
print(f"  {'Month':<8} │ " + ' │ '.join(f'{s:<22}' for s in STRAT_NAMES))
print(f"  {'':8} │ " + ' │ '.join(f"{'N':>3} {'Tot':>6} {'Avg':>6} {'WR':>4}" for _ in STRAT_NAMES))
print(f"  {'─'*90}")
for month in months:
    parts = []
    for sname in STRAT_NAMES:
        mt = [t for t in strategies[sname]['trades'] if t['date'].strftime('%Y-%m') == month]
        if mt:
            n = len(mt)
            tot = sum(t['pnl'] for t in mt)
            avg = np.mean([t['pnl'] for t in mt])
            wr = np.mean([1 if t['pnl'] > 0 else 0 for t in mt])
            parts.append(f"{n:>3} {tot:>+5.1%} {avg:>+5.2%} {wr:>3.0%}")
        else:
            parts.append(f"{'--':>3} {'--':>6} {'--':>6} {'--':>4}")
    print(f"  {month:<8} │ " + ' │ '.join(parts))


# ── CSV vs Live Comparison (reference from CSV-scored run) ──
print(f"\n  ── CSV vs LIVE Comparison ──")
print(f"  Reference CSV results (from backtest_short_production.py v2):")
CSV_REF = {
    'ML_STATIC':   {'n': 92,  'avg': 0.0111, 'wr': 0.55, 'dd': -0.402},
    'ML_ADAPTIVE': {'n': 104, 'avg': 0.0098, 'wr': 0.55, 'dd': -0.587},
    'ML_COMBINED': {'n': 77,  'avg': 0.0135, 'wr': 0.61, 'dd': -0.402},
}
print(f"  {'Strategy':<14} │ {'── CSV ──────────────':} │ {'── LIVE ─────────────':} │ {'Δ Avg':>7} {'Δ N':>5}")
print(f"  {'':14} │ {'N':>4} {'Avg':>7} {'WR':>5} {'DD':>7} │ {'N':>4} {'Avg':>7} {'WR':>5} {'DD':>7} │")
print(f"  {'─'*85}")
for sname in STRAT_NAMES:
    csv = CSV_REF.get(sname, {})
    live = strategy_summaries[sname]['FULL']
    csv_n = csv.get('n', 0)
    csv_avg = csv.get('avg', 0)
    csv_wr = csv.get('wr', 0)
    csv_dd = csv.get('dd', 0)
    d_avg = live['avg'] - csv_avg if live['n'] > 0 else 0
    d_n = live['n'] - csv_n
    print(f"  {sname:<14} │ {csv_n:>4} {csv_avg:>+6.2%} {csv_wr:>4.0%} {csv_dd:>+6.1%} │ "
          f"{live['n']:>4} {live['avg']:>+6.2%} {live['wr']:>4.0%} {live['dd']:>+6.1%} │ "
          f"{d_avg:>+6.2%} {d_n:>+4}")


# ── PHASE 5: Equity Curve Chart ──
print(f"\n  PHASE 5: Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('SHORT Production Simulation — Live TF Scoring', fontsize=14, fontweight='bold')

colors = {
    'ML_STATIC':   '#2563eb',
    'ML_ADAPTIVE': '#9333ea',
    'ML_COMBINED': '#ea580c',
}
line_styles = {
    'ML_STATIC': '-', 'ML_ADAPTIVE': '--', 'ML_COMBINED': '-',
}

# Panel 1: Cumulative equity curves
ax = axes[0, 0]
for sname in STRAT_NAMES:
    trades = sorted(strategies[sname]['trades'], key=lambda t: t['date'])
    if not trades:
        continue
    dates_arr = [t['date'] for t in trades]
    cum_pnl = np.cumsum([t['pnl'] for t in trades])
    lw = 2.2 if sname == 'ML_COMBINED' else 1.8
    ax.plot(dates_arr, cum_pnl * 100, label=sname, color=colors[sname],
            linewidth=lw, linestyle=line_styles[sname])
ax.axhline(0, color='#555', linewidth=0.5, linestyle='--')
ax.axvline(half2_start_dt, color='#555', linewidth=0.5, linestyle=':', alpha=0.5)
ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('Equity Curve (Live TF Scoring)')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Weekly PnL bars
ax = axes[0, 1]
bar_width = 0.25
week_labels = [f"W{i+1}" for i in range(len(weeks))]
x_pos = np.arange(len(weeks))

for si, sname in enumerate(STRAT_NAMES):
    weekly_pnls = []
    for week_idx_c in range(len(weeks)):
        wt = [t for t in strategies[sname]['trades'] if t['week'] == week_idx_c + 1]
        weekly_pnls.append(sum(t['pnl'] for t in wt) * 100 if wt else 0)
    offset = (si - 1) * bar_width
    ax.bar(x_pos + offset, weekly_pnls, bar_width, label=sname,
           color=colors[sname], alpha=0.8)

ax.axhline(0, color='#555', linewidth=0.5)
ax.set_xlabel('Week')
ax.set_ylabel('Week Total PnL (%)')
ax.set_title('Weekly PnL')
ax.set_xticks(x_pos)
ax.set_xticklabels(week_labels, fontsize=6, rotation=45)
ax.legend(fontsize=7)
ax.grid(True, axis='y', alpha=0.3)

# Panel 3: Ticker selection heatmap (COMBINED)
ax = axes[1, 0]
comb_wk = strategies['ML_COMBINED']['weekly_tickers']
all_comb_tks = sorted(set(
    t for wk in comb_wk for t in wk['tickers']))

if all_comb_tks and comb_wk:
    heatmap_data = np.zeros((len(all_comb_tks), len(comb_wk)))
    for wi, wk in enumerate(comb_wk):
        for ti, tk in enumerate(all_comb_tks):
            if tk in wk['tickers']:
                heatmap_data[ti, wi] = 1.0

    ax.imshow(heatmap_data, aspect='auto', cmap='Oranges', interpolation='nearest',
              vmin=0, vmax=1)
    ax.set_yticks(range(len(all_comb_tks)))
    ax.set_yticklabels(all_comb_tks, fontsize=7)
    ax.set_xticks(range(len(comb_wk)))
    ax.set_xticklabels([f"W{wk['week']}" for wk in comb_wk], fontsize=7, rotation=45)
    ax.set_title('ML_COMBINED: Ticker Selection by Week')
    ax.set_xlabel('Week')
else:
    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax.set_title('ML_COMBINED: Ticker Selection by Week')

# Panel 4: Trade PnL scatter — STATIC vs COMBINED
ax = axes[1, 1]
for sname in ['ML_STATIC', 'ML_COMBINED']:
    trades = strategies[sname]['trades']
    if not trades:
        continue
    dates_arr = [t['date'] for t in trades]
    pnls = [t['pnl'] * 100 for t in trades]
    ax.scatter(dates_arr, pnls, label=sname, color=colors[sname],
               alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
ax.axhline(0, color='#555', linewidth=0.5, linestyle='--')
ax.axvline(half2_start_dt, color='#555', linewidth=0.5, linestyle=':', alpha=0.5)
ax.set_ylabel('Trade PnL (%)')
ax.set_title('STATIC vs COMBINED: Individual Trades')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = '/tmp/backtest_short_production_live.png'
fig.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"    Chart saved: {chart_path}")


# ── Summary ──
print(f"\n{'='*W}")
print(f"  SUMMARY (LIVE TF SCORING)")
print(f"{'='*W}")

ml_strats_ranked = sorted(
    STRAT_NAMES,
    key=lambda s: strategy_summaries[s]['FULL']['avg'] if strategy_summaries[s]['FULL']['n'] > 0 else -999,
    reverse=True)

print(f"\n  ── Ranking by FULL avg PnL ──")
for rank, sname in enumerate(ml_strats_ranked, 1):
    fs = strategy_summaries[sname]['FULL']
    h2s = strategy_summaries[sname]['HALF2']
    marker = " ◄" if rank == 1 else ""
    print(f"    {rank}. {sname:<14} FULL: {fs['avg']:+.2%} ({fs['n']}t, {fs['wr']:.0%}WR, DD {fs['dd']:+.1%})  "
          f"H2: {h2s['avg']:+.2%} ({h2s['n']}t){marker}")

# Stability: total ticker changes per adaptive strategy
print(f"\n  ── Stability (total ticker changes across {len(weeks)} weeks) ──")
for adapt_name in ['ML_ADAPTIVE', 'ML_COMBINED']:
    n_changes = 0
    for wi in range(1, len(strategies[adapt_name]['weekly_tickers'])):
        prev = set(strategies[adapt_name]['weekly_tickers'][wi-1]['tickers'])
        curr = set(strategies[adapt_name]['weekly_tickers'][wi]['tickers'])
        n_changes += len(prev.symmetric_difference(curr))
    print(f"    {adapt_name:<14} {n_changes} changes")

print(f"\n{'='*W}")
print(f"  Done. Compare these results with CSV-scored backtest_short_production.py output.")
print(f"  If trade counts and avg PnL match closely, live scoring is confirmed.")
print(f"{'='*W}")
