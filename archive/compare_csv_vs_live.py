"""
compare_csv_vs_live.py — Side-by-Side CSV vs Live TF Probability Comparison
============================================================================
For a set of tickers, loads:
  1. CSV scores from GCS Results_all_v2/ (same source as backtest_short_production.py)
  2. Live TF 15-seed ensemble inference (same pipeline as backtest_short_live.py)

Prints per-ticker, per-day, per-seed probability comparison.
Identifies where divergence occurs (scaling? features? model weights?).

GPU recommended but not required. Colab-ready (no if __name__).
"""

import os, sys, io, gc, json, subprocess, ctypes, warnings
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')
os.environ['TF_USE_LEGACY_KERAS'] = '1'

try:
    from google.colab import auth
    _IN_COLAB = True
except (ImportError, ModuleNotFoundError):
    _IN_COLAB = False

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
    except Exception as _auth_err:
        print(f"  Auth note: {_auth_err} — continuing with default credentials")

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

COMPARE_START   = '2026-04-01'
COMPARE_END     = '2026-04-10'
COMPARE_DAYS    = 5

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]
ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1')
FILTERED_VA_OPT_RATIO = 2.0

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110


# ==============================================================================
# CONTEXT COLS (must match backtest_short_live.py exactly)
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

def load_master(ticker, apply_cutoff=False):
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
            cutoff = pd.Timestamp(COMPARE_START, tz='UTC') - pd.Timedelta(days=60)
            df = df[df.index >= cutoff]
        _master_cache[ticker] = df
        return df
    except Exception as e:
        print(f"      load_master({ticker}): FAILED — {e}")
        return None


def heal_data_csv(df):
    """heal_data as used in backtest_short_production.py — minimal."""
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def heal_data_live(df):
    """heal_data as used in backtest_short_live.py — full micro zero-fill."""
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
# CSV SCORES (from backtest_short_production.py)
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


def get_csv_probs_at_entry(seed_scores, entry_min, target_date):
    """Get per-seed probabilities from CSV at entry_minute on target_date."""
    per_seed = {}
    for seed_id, sdf in seed_scores.items():
        bar_mins = sdf.index.hour * 60 + sdf.index.minute
        day_mask = sdf.index.date == target_date
        entry_mask = bar_mins == entry_min
        matches = sdf[day_mask & entry_mask]
        if len(matches) > 0:
            per_seed[seed_id] = float(matches.iloc[0]['Prob_Spike'])
    return per_seed


# ==============================================================================
# LIVE SCORING (from backtest_short_live.py)
# ==============================================================================
def build_market_context():
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker, apply_cutoff=True)
        if df is None:
            continue
        df = df.sort_index()
        df = heal_data_live(df)
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


def apply_scaling(df, stats):
    df_s = df.copy()
    for col in ALL_COLS:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


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


def score_live_at_entry(df_scaled, entry_min, target_date, base_exp, seed_list):
    """Run live 15-seed TF inference at entry_minute on target_date. Returns per-seed probs."""
    target_pst = PST.localize(datetime(target_date.year, target_date.month, target_date.day,
                                       entry_min // 60, entry_min % 60))
    target_utc = target_pst.astimezone(pytz.utc)
    candidates = df_scaled.index[df_scaled.index <= target_utc]
    if len(candidates) == 0:
        return {}
    row_idx = df_scaled.index.get_loc(candidates[-1])
    bar_ts_pst = df_scaled.index[row_idx].tz_convert(PST)
    score_date = bar_ts_pst.date()
    if score_date != target_date:
        return {}
    bar_minute = bar_ts_pst.hour * 60 + bar_ts_pst.minute
    if bar_minute != entry_min:
        return {}
    if row_idx < TIME_STEPS:
        return {}

    XA = df_scaled[ASSET_COLS].values
    XC = df_scaled[CONTEXT_COLS].values
    XA_b  = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]
    XC_b  = XC[row_idx - 1][np.newaxis]
    XID_b = np.array([[OTHER_IDX]], dtype=np.int32)

    per_seed = {}
    for s in seed_list:
        try:
            m = get_model(s, base_exp)
            p = float(m(
                {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
                training=False
            ).numpy().flatten()[0])
            per_seed[s] = p
        except Exception as e:
            per_seed[s] = f"ERR: {e}"
    return per_seed


# ==============================================================================
# BQ CONFIG (same as backtest_short_production.py)
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
# MAIN COMPARISON
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  CSV vs LIVE TF Probability Comparison")
print(f"  Compare first {COMPARE_DAYS} trading days: {COMPARE_START} to {COMPARE_END}")
print(f"{'='*W}")


# ── Step 1: BQ configs ──
print(f"\n  Step 1: Loading BQ configs...")
bq_df = query_all_bq()
all_tickers = sorted(bq_df['ticker'].unique())
print(f"    {len(bq_df)} BQ rows, {len(all_tickers)} tickers")

all_configs = select_configs_ensemble(bq_df, all_tickers)
print(f"    {len(all_configs)} tickers with ensemble configs")

compare_tickers = sorted(all_configs.keys())[:10]
print(f"\n    Comparing first 10 tickers: {', '.join(compare_tickers)}")


# ── Step 2: Derive seed lists ──
print(f"\n  Step 2: Deriving seed lists per config...")
base_exps_needed = set()
for ticker in compare_tickers:
    cfg = all_configs[ticker]
    base_exp = cfg['_base_experiment']
    alpha = base_exp.split('_')[0]
    seed_base = int(alpha) * 100
    cfg['_seed_list'] = list(range(seed_base + 1, seed_base + 16))
    base_exps_needed.add(base_exp)
    print(f"    {ticker}: exp={cfg['experiment']}, base={base_exp}, seeds={cfg['_seed_list'][0]}-{cfg['_seed_list'][-1]}, entry_min={cfg['entry_minute']}")

print(f"\n    Base experiments needed: {sorted(base_exps_needed)}")


# ── Step 3: Load scaling stats ──
print(f"\n  Step 3: Loading scaling stats...")
scaling_stats = {}
for be in sorted(base_exps_needed):
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/scaling_stats_experiment_{be}.json"
    try:
        blob = gcs_client.bucket(BUCKET_NAME).blob(f"{INFERENCE_DIR}/scaling_stats_experiment_{be}.json")
        raw = json.loads(blob.download_as_string())
        scaling_stats[be] = raw
        print(f"    {be}: {len(raw)} columns scaled")
    except Exception as e:
        print(f"    {be}: FAILED — {e}")
        try:
            alpha = be.split('_')[0]
            blob2 = gcs_client.bucket(BUCKET_NAME).blob(f"{INFERENCE_DIR}/scaling_stats_experiment_{alpha}.json")
            raw = json.loads(blob2.download_as_string())
            scaling_stats[be] = raw
            print(f"    {be}: fallback to {alpha} — {len(raw)} columns")
        except Exception as e2:
            print(f"    {be}: fallback also FAILED — {e2}")


# ── Step 4: Load CSV scores ──
print(f"\n  Step 4: Loading CSV scores...")
csv_scores_cache = {}
for ticker in compare_tickers:
    cfg = all_configs[ticker]
    base_exp = cfg['_base_experiment']
    key = (ticker, base_exp)
    if key in csv_scores_cache:
        continue
    seed_scores = load_seed_scores(ticker, base_exp)
    if seed_scores:
        csv_scores_cache[key] = seed_scores
        print(f"    {ticker}/{base_exp}: {len(seed_scores)} seeds loaded from CSV")
    else:
        print(f"    {ticker}/{base_exp}: NO CSV SCORES FOUND")
    gc.collect()


# ── Step 5: Build market context + engineer features for live scoring ──
print(f"\n  Step 5: Building market context...")
market_df = build_market_context()
print(f"    Market context: {len(market_df)} bars, {market_df.columns.tolist()[:5]}...")

print(f"\n  Step 6: Engineering features + scaling for live scoring...")
live_data = {}
for ticker in compare_tickers:
    cfg = all_configs[ticker]
    base_exp = cfg['_base_experiment']
    if base_exp not in scaling_stats:
        print(f"    {ticker}: SKIP — no scaling stats for {base_exp}")
        continue
    df = load_master(ticker, apply_cutoff=True)
    if df is None:
        continue
    df = df.sort_index()
    df = heal_data_live(df)
    df = engineer_features(df, market_df)
    df_scaled = apply_scaling(df, scaling_stats[base_exp])
    live_data[ticker] = df_scaled[ALL_COLS].copy()
    n_bars = len(live_data[ticker])
    print(f"    {ticker}: {n_bars} bars scaled ({n_bars/390:.0f} trading days)")
    del df, df_scaled
    gc.collect()

_release_memory()


# ── Step 6: Load models ──
print(f"\n  Step 7: Pre-loading models...")
for be in sorted(base_exps_needed):
    if be not in scaling_stats:
        continue
    alpha = be.split('_')[0]
    seed_base = int(alpha) * 100
    for s in range(seed_base + 1, seed_base + 16):
        try:
            get_model(s, be)
        except Exception as e:
            print(f"    Model S{s}_M{be}: FAILED — {e}")
print(f"    {len(_model_cache)} models loaded")


# ── Step 7: Compare! ──
trading_days = pd.bdate_range(COMPARE_START, COMPARE_END).date[:COMPARE_DAYS]
print(f"\n{'='*W}")
print(f"  COMPARISON: {len(compare_tickers)} tickers × {len(trading_days)} days")
print(f"{'='*W}")

all_deltas = []

for ticker in compare_tickers:
    cfg = all_configs[ticker]
    base_exp = cfg['_base_experiment']
    entry_min = cfg['entry_minute']
    seed_list = cfg.get('_seed_list', [])
    csv_key = (ticker, base_exp)

    has_csv = csv_key in csv_scores_cache
    has_live = ticker in live_data

    print(f"\n  ┌─ {ticker} (exp={cfg['experiment']}, entry_min={entry_min}, thresh={cfg['prob_thresh']:.3f})")
    if not has_csv:
        print(f"  │  *** NO CSV SCORES ***")
    if not has_live:
        print(f"  │  *** NO LIVE DATA ***")
        continue

    df_scaled = live_data[ticker]

    for day in trading_days:
        csv_probs = {}
        if has_csv:
            csv_probs = get_csv_probs_at_entry(csv_scores_cache[csv_key], entry_min, day)

        live_probs = score_live_at_entry(df_scaled, entry_min, day, base_exp, seed_list)

        csv_mean = np.mean(list(csv_probs.values())) if csv_probs else None
        live_vals = [v for v in live_probs.values() if isinstance(v, (int, float))]
        live_mean = np.mean(live_vals) if live_vals else None

        delta = None
        if csv_mean is not None and live_mean is not None:
            delta = live_mean - csv_mean
            all_deltas.append({
                'ticker': ticker, 'date': day,
                'csv_prob': csv_mean, 'live_prob': live_mean,
                'delta': delta, 'abs_delta': abs(delta),
                'csv_signal': csv_mean >= cfg['prob_thresh'],
                'live_signal': live_mean >= cfg['prob_thresh'],
                'signal_match': (csv_mean >= cfg['prob_thresh']) == (live_mean >= cfg['prob_thresh']),
            })

        csv_str = f"{csv_mean:.4f}" if csv_mean is not None else "  N/A "
        live_str = f"{live_mean:.4f}" if live_mean is not None else "  N/A "
        delta_str = f"{delta:+.4f}" if delta is not None else "  N/A "
        thresh = cfg['prob_thresh']
        csv_sig = "SIG" if csv_mean is not None and csv_mean >= thresh else "   "
        live_sig = "SIG" if live_mean is not None and live_mean >= thresh else "   "
        match = "✓" if csv_sig == live_sig else "✗"

        print(f"  │  {day}  CSV={csv_str} {csv_sig}  LIVE={live_str} {live_sig}  Δ={delta_str}  {match}")

        # Per-seed detail for first day only
        if day == trading_days[0] and (csv_probs or live_probs):
            all_seeds = sorted(set(list(csv_probs.keys()) + [s for s in live_probs.keys()]))
            if len(all_seeds) > 0:
                print(f"  │  ── Per-Seed Detail (day 1 only) ──")
                for s in all_seeds[:5]:
                    c = csv_probs.get(s, None)
                    l = live_probs.get(s, None)
                    c_str = f"{c:.4f}" if c is not None else " N/A "
                    l_str = f"{l:.4f}" if isinstance(l, (int, float)) else f" {l}" if l else " N/A "
                    d_str = f"{l-c:+.4f}" if (isinstance(l, (int, float)) and c is not None) else " N/A "
                    print(f"  │    Seed {s:>5}: CSV={c_str}  LIVE={l_str}  Δ={d_str}")
                if len(all_seeds) > 5:
                    print(f"  │    ... ({len(all_seeds)-5} more seeds)")

    print(f"  └─")


# ── Summary ──
print(f"\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")

if all_deltas:
    df_d = pd.DataFrame(all_deltas)
    n_total = len(df_d)
    n_match = df_d['signal_match'].sum()
    n_mismatch = n_total - n_match

    print(f"\n  Total comparisons: {n_total}")
    print(f"  Signal matches:    {n_match} ({n_match/n_total:.0%})")
    print(f"  Signal mismatches: {n_mismatch} ({n_mismatch/n_total:.0%})")
    print(f"\n  Probability delta (LIVE - CSV):")
    print(f"    Mean:   {df_d['delta'].mean():+.4f}")
    print(f"    Median: {df_d['delta'].median():+.4f}")
    print(f"    Std:    {df_d['abs_delta'].std():.4f}")
    print(f"    Max:    {df_d['abs_delta'].max():.4f}")
    print(f"    Min:    {df_d['abs_delta'].min():.4f}")

    print(f"\n  Per-Ticker Average Delta:")
    for ticker, grp in df_d.groupby('ticker'):
        avg_d = grp['delta'].mean()
        max_d = grp['abs_delta'].max()
        matches = grp['signal_match'].sum()
        total = len(grp)
        print(f"    {ticker:<7}  avgΔ={avg_d:+.4f}  maxΔ={max_d:.4f}  signals={matches}/{total}")

    if n_mismatch > 0:
        print(f"\n  Signal Mismatches (CSV signal ≠ LIVE signal):")
        mismatches = df_d[df_d['signal_match'] == False]
        for _, row in mismatches.iterrows():
            print(f"    {row['ticker']} {row['date']}  CSV={row['csv_prob']:.4f}({'SIG' if row['csv_signal'] else '---'})  "
                  f"LIVE={row['live_prob']:.4f}({'SIG' if row['live_signal'] else '---'})  Δ={row['delta']:+.4f}")
else:
    print(f"\n  No valid comparisons — check data availability.")


print(f"\n{'='*W}")
print(f"  INTERPRETATION")
print(f"{'='*W}")
print(f"""
  If delta ≈ 0 (< 0.01): Models + features match. CSV and live produce same probs.
  If delta is systematic (all positive or all negative): Scaling or feature offset.
  If delta is large & random: Different model weights, or fundamental feature mismatch.
  If signal match rate < 80%: Live scoring will produce very different trade selections.
""")

_release_memory()
print(f"\n  Done.")
