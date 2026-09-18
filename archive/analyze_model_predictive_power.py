"""
analyze_model_predictive_power.py — Model vs Market: June 2026 Diagnosis
========================================================================
Comprehensive classification analysis on v11 deployed tickers to answer:
Is the model failing (calling spikes that don't exist) or is the market
simply not producing spikes?

Runs live TF model inference for ALL bars (not just GO signals) to compute
full confusion matrices (TP/FP/TN/FN), calibration curves, ROC/AUC, and
per-ticker precision-recall breakdowns across April, May, and June 2026.

Run in Colab: paste as cell. Requires: TF, pandas_ta, numba, GCS, BQ.
"""

import os, io, gc, sys, json, subprocess, warnings, ctypes
from datetime import datetime, timedelta

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

try:
    import tf_keras
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tf_keras", "--quiet"])

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]", "tf_keras", "pandas_ta",
                           "google-cloud-storage", "pyarrow", "gcsfs",
                           "numba", "--quiet"])
    import tensorflow as tf

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from numba import jit
from google.cloud import storage as gcs_storage, bigquery

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

import keras
print(f"  TF={tf.__version__}, Keras={keras.__version__}")
print(f"  Legacy Keras: {os.environ.get('TF_USE_LEGACY_KERAS', 'NOT SET')}")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2

# ==============================================================================
# CONFIG
# ==============================================================================
VERSION = 'robust_wf_v11'
TEST_START = '2026-04-01'
TEST_END   = '2026-06-09'
SPIKE_THRESHOLD = 0.10

BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")
BQ_TABLE       = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
INFERENCE_DIR  = "Data/Inference_v2"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR      = "Data/Inference_v2"

TIME_STEPS = 240; EMBED_DIM = 16; EXIT_TIME_MIN = 780
PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

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

TEACHERS_13 = ['BBAI','RCAT','LUNR','ASTS','ACHR','MVIS','WULF','CIFR','IREN','KEEL','HUT','CLSK','APLD']
TEACHERS_31 = ['CIFR','IREN','KEEL','HUT','APLD','WULF','RIOT','CLSK','BTBT','ALAB','MARA',
    'RCAT','ASTS','LUNR','BBAI','RKLB','SPIR','BKSY','MNTS','PLUG','BE','FCEL','FLNC',
    'OKLO','SMR','MVIS','BGDE','RDW','PL','VOYG','SMLR']

FAMILY_TEACHERS = {'152_XSECTOR': TEACHERS_13, '160_XSECTOR': TEACHERS_13,
    '170_XSECTOR': TEACHERS_13, '175_XSECTOR': TEACHERS_13, '185_XSECTOR': TEACHERS_31}

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL'}

W = 120

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()

def get_asset_index(model_id, ticker):
    teachers = FAMILY_TEACHERS.get(model_id, TEACHERS_13)
    return {t: i for i, t in enumerate(teachers)}.get(ticker, len(teachers))

def get_n_assets(model_id):
    return len(FAMILY_TEACHERS.get(model_id, TEACHERS_13)) + 1

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================
_master_cache = {}
_model_cache = {}
_stats_cache = {}
_market_context_cache = None

def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open','High','Low','Close']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume','Total_Vol','Volume_Delta','Trade_Count','Delta_Pct','Spread','OBI','bsize','asize']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df

def load_master(ticker, start_date=None):
    cache_key = (ticker, start_date)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        df = pd.read_parquet(uri)
        df = df.reset_index()
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            df['Datetime'] = pd.to_datetime(df[df.columns[0]], unit='ms', utc=True)
        df = df.set_index('Datetime')
        df = df[~df.index.duplicated(keep='first')]
        std_map = {'Total_vol':'Total_Vol','Trade_count':'Trade_Count','Volume_delta':'Volume_Delta',
                    'Delta_pct':'Delta_Pct','Obi':'OBI','Bsize':'bsize','Asize':'asize',
                    'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
        lower_map = {k.lower():v for k,v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        for c in MICRO_COLS:
            if c not in df.columns: df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date, tz='UTC') - pd.Timedelta(days=30)]
        _master_cache[cache_key] = df
        return df
    except Exception:
        return None

def build_market_context(start_date=None):
    global _market_context_cache
    if _market_context_cache is not None: return _market_context_cache
    print("  Building market context...")
    trim_from = pd.Timestamp(start_date, tz='UTC') - pd.Timedelta(days=7) if start_date else None
    frames = []
    for ticker in CONTEXT_ASSETS:
        clean = TICKER_FILE_MAP.get(ticker, ticker)
        uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
        try:
            d = pd.read_parquet(uri)
        except Exception:
            continue
        d = d.reset_index()
        dt_cols = [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]
        if dt_cols:
            d['Datetime'] = pd.to_datetime(d[dt_cols[0]], utc=True)
        else:
            d['Datetime'] = pd.to_datetime(d[d.columns[0]], unit='ms', utc=True)
        d = d.set_index('Datetime').sort_index()
        d = d[~d.index.duplicated(keep='first')]
        std_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
        lower_map = {k.lower():v for k,v in std_map.items()}
        d.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in d.columns]
        if trim_from is not None:
            d = d[d.index >= trim_from]
        d = d[['Close']].copy()
        d = d.resample('1min').asfreq()
        d['Close'] = d['Close'].ffill()
        daily_c = d['Close'].resample('D').last().ffill().shift(1).reindex(d.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (d['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = d['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=d.index))
        del d, daily_c; _release_memory()
        print(f"    {ticker} loaded")
    _market_context_cache = pd.concat(frames, axis=1, sort=True).ffill().fillna(0)
    del frames; _release_memory()
    return _market_context_cache

def engineer_features(df, market_df):
    daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(df.index, method='ffill')
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

def load_scaling_stats(model_id):
    if model_id in _stats_cache: return _stats_cache[model_id]
    blob = gcs_client.bucket(BUCKET_NAME).blob(f"{STATS_DIR}/scaling_stats_experiment_{model_id}.json")
    raw = json.loads(blob.download_as_bytes())
    _stats_cache[model_id] = raw
    return raw

def apply_scaling(df, stats):
    df_s = df.copy()
    for col in ALL_COLS:
        if col in ('Time_Pos','Bar_Is_Real'): continue
        if col in stats:
            df_s[col] = ((df_s[col]-stats[col]['med'])/stats[col]['iqr']).clip(-10,10).astype('float32')
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

def build_model(n_context_cols, n_assets, n_asset_cols=None):
    n_ac = n_asset_cols or len(ASSET_COLS)
    in_a = Input(shape=(TIME_STEPS, n_ac), name='asset_in')
    in_c = Input(shape=(n_context_cols,), name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32, name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Flatten()(Embedding(n_assets, EMBED_DIM, name='asset_embed')(in_id))
    e = Dense(32, activation='relu')(e)
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)

def get_model(model_id, seed):
    key = (model_id, seed)
    if key in _model_cache: return _model_cache[key]
    blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{model_id}.h5"
    if not os.path.exists(local_path):
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
    model = build_model(len(CONTEXT_COLS), get_n_assets(model_id), n_asset_cols=len(ASSET_COLS))
    model.load_weights(local_path)
    _model_cache[key] = model
    return model

# ==============================================================================
# COMPOSITE SCORE (S8_mix only — for portfolio selection)
# ==============================================================================
def compute_s8_mix(df):
    va = df['val_a_avg_pnl'].values
    va_wr = df['val_a_win_rate'].values
    va_n = df['val_a_count'].values.astype(float)
    vb = df['val_b_avg_pnl'].values
    vb_wr = df['val_b_win_rate'].values
    vb_n = df['val_b_count'].values.astype(float)
    combined_n = va_n + vb_n
    wavg_pnl = np.where(combined_n > 0, (va * va_n + vb * vb_n) / combined_n, 0)
    combined_wr = np.where(combined_n > 0, (va_wr * va_n + vb_wr * vb_n) / combined_n, 0)
    degradation = np.where(va > 0, np.abs(vb - va) / va, 1.0)
    consistency = np.clip(1 - degradation, 0.2, 1.0)
    def mo(s, e):
        try: return max(0.5, (pd.Timestamp(e) - pd.Timestamp(s)).days / 30.44)
        except: return 1.0
    va_n_mo = va_n / np.array([mo(s, e) for s, e in zip(df['val_a_start'], df['val_a_end'])])
    vb_n_mo = vb_n / np.array([mo(s, e) for s, e in zip(df['val_b_start'], df['val_b_end'])])
    combined_n_mo = (va_n_mo + vb_n_mo) / 2
    nq3 = np.exp(-0.5 * ((combined_n_mo - 3.0) / 2.5)**2)
    asym_filter = ((va >= 0.015) & (vb >= 0.010)) | ((va >= 0.010) & (vb >= 0.015))
    wr_both_50 = (va_wr >= 0.50) & (vb_wr >= 0.50)
    df['S8_mix'] = np.where(asym_filter & wr_both_50,
        wavg_pnl**2 * combined_wr**2 * nq3 * consistency * np.log2(1 + combined_n), 0)
    return df

# ==============================================================================
# STEP 1: LOAD V11 PORTFOLIO FROM BQ
# ==============================================================================
print(f"{'='*W}")
print(f"  MODEL PREDICTIVE POWER ANALYSIS — {VERSION}")
print(f"  Period: {TEST_START} → {TEST_END}  |  Spike threshold: {SPIKE_THRESHOLD*100:.0f}%")
print(f"{'='*W}")

q = f"""
SELECT ticker, experiment, seed, tpsl_mode, prob_thresh, model_version,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       tp_pct, sl_pct, tp_mult, sl_mult,
       entry_minute,
       val_a_start, val_a_end, val_b_start, val_b_end,
       survived_a, survived_b, is_ensemble
FROM `{BQ_TABLE}`
WHERE model_version = '{VERSION}'
  AND survived_a = TRUE
  AND val_b_start = '{TEST_START}'
"""
print("  Loading BQ data...")
df_bq = bq_client.query(q).to_dataframe()
df_bq = df_bq.drop_duplicates(subset=['ticker','experiment','seed','tpsl_mode','prob_thresh'], keep='first')
df_bq = compute_s8_mix(df_bq)

# Select best config per ticker — use S8_mix if available, else best val_b_avg_pnl
has_score = df_bq[df_bq['S8_mix'] > 0]
no_score = df_bq[~df_bq['ticker'].isin(has_score['ticker'].unique())]
portfolio_scored = has_score.loc[has_score.groupby('ticker')['S8_mix'].idxmax()] if len(has_score) > 0 else pd.DataFrame()
portfolio_unscored = no_score.loc[no_score.groupby('ticker')['val_b_avg_pnl'].idxmax()] if len(no_score) > 0 else pd.DataFrame()
portfolio = pd.concat([portfolio_scored, portfolio_unscored], ignore_index=True)

deployed_set = set(portfolio_scored['ticker']) if len(portfolio_scored) > 0 else set()
print(f"  {len(df_bq)} configs, {df_bq['ticker'].nunique()} unique tickers")
print(f"  → {len(portfolio_scored)} deployed (S8_mix>0) + {len(portfolio_unscored)} non-deployed = {len(portfolio)} total")

ticker_configs = {}
for _, r in portfolio.iterrows():
    exp = r['experiment']
    mode_sfx = MODE_SUFFIX.get(r['tpsl_mode'], '')
    base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp
    seed_val = int(r['seed'])
    is_ens = bool(r['is_ensemble'])

    if base_exp == '170_XSECTOR':
        base_exp = '175_XSECTOR'
        if not is_ens and 0 <= seed_val <= 14:
            seed_val = 17501 + seed_val

    ticker_configs[r['ticker']] = {
        'experiment': exp, 'base_exp': base_exp,
        'seed': seed_val, 'is_ensemble': is_ens,
        'prob_thresh': float(r['prob_thresh']),
        'tpsl_mode': r['tpsl_mode'],
        'tp_pct': float(r.get('tp_pct', 0) or 0),
        'sl_pct': float(r.get('sl_pct', 0) or 0),
        'tp_mult': float(r.get('tp_mult', 0) or 0),
        'sl_mult': float(r.get('sl_mult', 0) or 0),
        'entry_minute': int(r['entry_minute']),
    }

deployed_tickers = sorted(ticker_configs.keys())
n_dep = sum(1 for t in deployed_tickers if t in deployed_set)
n_ndep = len(deployed_tickers) - n_dep
print(f"  All tickers ({len(deployed_tickers)}): {deployed_tickers}")
print(f"    Deployed: {sorted(t for t in deployed_tickers if t in deployed_set)}")
print(f"    Non-deployed: {sorted(t for t in deployed_tickers if t not in deployed_set)}")

# Group by experiment for TABLE 9
exp_tickers = {}
for t, cfg in ticker_configs.items():
    exp_key = cfg['base_exp'].split('_')[0]
    exp_tickers.setdefault(exp_key, []).append(t)
print(f"  By experiment: {', '.join(f'{k}: {len(v)}' for k, v in exp_tickers.items())}")

# ==============================================================================
# STEP 2: LOAD DATA + RUN INFERENCE ON ALL BARS + COMPUTE GROUND TRUTH
# ==============================================================================
print(f"\n  Loading data + running inference for {len(deployed_tickers)} tickers...")
print(f"  Scoring ALL bars at each ticker's entry_minute (not just GO signals)")
market_df = build_market_context(start_date=TEST_START)

test_start_utc = pd.Timestamp(TEST_START, tz='UTC')
test_end_utc = pd.Timestamp(TEST_END, tz='UTC') + timedelta(days=1)
trim_start = test_start_utc - pd.Timedelta(days=5)

all_bars = []

for ci, ticker in enumerate(deployed_tickers):
    cfg = ticker_configs[ticker]
    print(f"  [{ci+1}/{len(deployed_tickers)}] {ticker}/{cfg['base_exp']}/S{'ENS' if cfg['is_ensemble'] else cfg['seed']}...", end=' ')

    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        df_raw = pd.read_parquet(uri)
    except Exception:
        print("SKIP (no data)"); continue
    df_raw = df_raw.reset_index()
    dt_cols = [c for c in df_raw.columns if pd.api.types.is_datetime64_any_dtype(df_raw[c])]
    if dt_cols:
        df_raw['Datetime'] = pd.to_datetime(df_raw[dt_cols[0]], utc=True)
    else:
        df_raw['Datetime'] = pd.to_datetime(df_raw[df_raw.columns[0]], unit='ms', utc=True)
    df_raw = df_raw.set_index('Datetime').sort_index()
    df_raw = df_raw[~df_raw.index.duplicated(keep='first')]
    std_map = {'Total_vol':'Total_Vol','Trade_count':'Trade_Count','Volume_delta':'Volume_Delta',
                'Delta_pct':'Delta_Pct','Obi':'OBI','Bsize':'bsize','Asize':'asize',
                'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
    lower_map = {k.lower():v for k,v in std_map.items()}
    df_raw.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df_raw.columns]
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated(keep='first')].copy()
    for c in MICRO_COLS:
        if c not in df_raw.columns: df_raw[c] = 0.0
    df_raw = df_raw[df_raw.index >= test_start_utc - pd.Timedelta(days=30)]
    df_raw = heal_data(df_raw)

    df_pst = df_raw.copy()
    df_pst.index = df_pst.index.tz_convert(PST)

    df_feat = engineer_features(df_raw.copy(), market_df)
    stats = load_scaling_stats(cfg['base_exp'])
    df_scaled = apply_scaling(df_feat, stats)

    asset_id = get_asset_index(cfg['base_exp'], ticker)
    df_test = df_scaled[(df_scaled.index >= test_start_utc) & (df_scaled.index < test_end_utc)]
    if len(df_test) == 0:
        del df_raw, df_feat, df_scaled, df_pst; _release_memory()
        print("SKIP (no test data)"); continue

    em = cfg['entry_minute']
    pst_idx = df_test.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute
    positions = [df_scaled.index.get_loc(df_test.index[i])
                 for i in range(len(df_test))
                 if mins[i] == em and df_scaled.index.get_loc(df_test.index[i]) >= TIME_STEPS
                 and ('Volume' not in df_scaled.columns or float(df_scaled['Volume'].iloc[df_scaled.index.get_loc(df_test.index[i])]) > 0)]

    if not positions:
        del df_raw, df_feat, df_scaled, df_pst; _release_memory()
        print("no valid bars"); continue

    XA = np.stack([df_scaled[ASSET_COLS].values[idx-TIME_STEPS:idx] for idx in positions]).astype(np.float32)
    XC = np.stack([df_scaled[CONTEXT_COLS].values[idx-1] for idx in positions]).astype(np.float32)
    XID = np.full((len(positions), 1), asset_id, dtype=np.int32)

    if cfg['is_ensemble']:
        seed_list = list(range(int(cfg['base_exp'].split('_')[0])*100+1, int(cfg['base_exp'].split('_')[0])*100+16))
    else:
        seed_list = [cfg['seed']]

    all_probs = []
    for s in seed_list:
        try:
            m = get_model(cfg['base_exp'], s)
            all_probs.append(m.predict({'asset_in':XA,'context_in':XC,'asset_id':XID},batch_size=64,verbose=0).flatten())
        except Exception:
            continue
    if not all_probs:
        del df_raw, df_feat, df_scaled, df_pst; _release_memory()
        print("no models"); continue

    probs = np.mean(all_probs, axis=0)

    test_s_pst = pd.Timestamp(TEST_START, tz=PST)
    test_e_pst = pd.Timestamp(TEST_END, tz=PST) + timedelta(days=1)
    ohlcv_test = df_pst[(df_pst.index >= test_s_pst) & (df_pst.index < test_e_pst)]

    for i, idx in enumerate(positions):
        ts = df_scaled.index[idx]
        ts_pst = ts.tz_convert(PST)
        prob = float(probs[i])

        entry_price = float(df_pst['Close'].iloc[df_pst.index.get_indexer([ts_pst], method='nearest')[0]])

        day_mask = (ohlcv_test.index.date == ts_pst.date()) & (ohlcv_test.index > ts_pst)
        day_bars = ohlcv_test[day_mask]

        real_bars = day_bars[(day_bars['Volume'] > 0) &
                             (day_bars.index.hour * 60 + day_bars.index.minute <= EXIT_TIME_MIN)]

        max_gain = 0.0
        max_drop = 0.0
        eod_ret = 0.0

        if len(real_bars) > 0 and entry_price > 0:
            max_gain = float((real_bars['High'].values / entry_price - 1).max())
            max_drop = float((real_bars['Low'].values / entry_price - 1).min())

        if len(day_bars) > 0 and entry_price > 0:
            eod_ret = float(day_bars['Close'].iloc[-1] / entry_price - 1)

        tp_eff = cfg['tp_pct'] if cfg['tp_pct'] > 0 else cfg['tp_mult'] * 0.05
        sl_eff = cfg['sl_pct'] if cfg['sl_pct'] > 0 else cfg['sl_mult'] * 0.05

        trade_pnl = eod_ret
        exit_type = 'EOD'
        if max_drop <= -sl_eff:
            trade_pnl = -sl_eff
            exit_type = 'SL'
        elif max_gain >= tp_eff:
            trade_pnl = tp_eff
            exit_type = 'TP'

        date_str = ts_pst.strftime('%Y-%m-%d')
        month = ts_pst.month
        if month == 4:
            period = 'April'
        elif month == 5:
            period = 'May'
        else:
            period = 'June'

        week_start = (ts_pst - timedelta(days=ts_pst.weekday())).strftime('%Y-%m-%d')

        bar = {
            'ticker': ticker,
            'date': date_str,
            'week': week_start,
            'period': period,
            'entry_minute': em,
            'prob': prob,
            'prob_thresh': cfg['prob_thresh'],
            'max_gain': max_gain,
            'max_drop': max_drop,
            'eod_ret': eod_ret,
            'trade_pnl': trade_pnl,
            'exit_type': exit_type,
            'tp_eff': tp_eff,
            'sl_eff': sl_eff,
            'entry_price': entry_price,
            'actual_spike_10': int(max_gain >= 0.10),
            'actual_spike_8': int(max_gain >= 0.08),
            'actual_spike_5': int(max_gain >= 0.05),
            'actual_spike_3': int(max_gain >= 0.03),
            'experiment': cfg['base_exp'].split('_')[0],
            'is_deployed': ticker in deployed_set,
        }
        all_bars.append(bar)

    del df_raw, df_feat, df_scaled, df_pst, XA, XC; _release_memory()
    n_go = sum(1 for b in all_bars[-len(positions):] if b['prob'] >= b['prob_thresh'])
    n_spikes = sum(1 for b in all_bars[-len(positions):] if b['actual_spike_10'])
    print(f"done ({len(positions)} bars, {n_go} GO, {n_spikes} spikes)")

    tf.keras.backend.clear_session(); _model_cache.clear(); _release_memory()

_master_cache.clear(); _release_memory()

# ==============================================================================
# BUILD ANALYSIS DATAFRAME
# ==============================================================================
df = pd.DataFrame(all_bars)
print(f"\n  Total bars scored: {len(df)}")
print(f"  Tickers: {df['ticker'].nunique()} ({df['is_deployed'].sum() // max(1, df.groupby('ticker').ngroups)} deployed per bar)")
print(f"  Deployed tickers: {sorted(df[df['is_deployed']]['ticker'].unique())}")
print(f"  Non-deployed tickers: {sorted(df[~df['is_deployed']]['ticker'].unique())}")
print(f"  By period: {df.groupby('period').size().to_dict()}")
print(f"  Actual spikes (≥10%): {df['actual_spike_10'].sum()} / {len(df)} = {df['actual_spike_10'].mean()*100:.1f}%")
print(f"  GO signals: {(df['prob'] >= df['prob_thresh']).sum()}")

df_go = df[df['prob'] >= df['prob_thresh']].copy()

# ==============================================================================
# TABLE 1: MONTHLY CONFUSION MATRIX
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 1: MONTHLY CONFUSION MATRIX — 10% spike threshold")
print(f"  Prob thresholds: 0.40, 0.45, 0.50, 0.55, 0.60")
print(f"{'='*W}")

prob_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
periods = ['April', 'May', 'June']

print(f"\n  {'Period':<7} {'Thresh':>6} {'TP':>5} {'FP':>5} {'TN':>6} {'FN':>5} "
      f"{'Prec':>6} {'Recall':>6} {'F1':>6} {'Acc':>6} {'GO/day':>6}")
print(f"  {'─'*7} {'─'*6} {'─'*5} {'─'*5} {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

cm_data = []
for period in periods:
    df_p = df[df['period'] == period]
    n_days = df_p['date'].nunique()
    for thresh in prob_thresholds:
        pred_pos = df_p['prob'] >= thresh
        actual_pos = df_p['actual_spike_10'] == 1

        tp = int((pred_pos & actual_pos).sum())
        fp = int((pred_pos & ~actual_pos).sum())
        tn = int((~pred_pos & ~actual_pos).sum())
        fn = int((~pred_pos & actual_pos).sum())

        prec = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * prec * recall / max(prec + recall, 1e-9)
        acc = (tp + tn) / max(tp + fp + tn + fn, 1)
        go_per_day = (tp + fp) / max(n_days, 1)

        cm_data.append({'period': period, 'thresh': thresh, 'tp': tp, 'fp': fp,
                        'tn': tn, 'fn': fn, 'prec': prec, 'recall': recall,
                        'f1': f1, 'acc': acc, 'go_per_day': go_per_day})

        print(f"  {period:<7} {thresh:>5.2f} {tp:>5} {fp:>5} {tn:>6} {fn:>5} "
              f"{prec*100:>5.1f}% {recall*100:>5.1f}% {f1*100:>5.1f}% {acc*100:>5.1f}% {go_per_day:>5.1f}")
    print()

df_cm = pd.DataFrame(cm_data)

# Highlight the deployed threshold level
deployed_thresh = df['prob_thresh'].mode().iloc[0] if len(df) > 0 else 0.50
print(f"  Most common deployed threshold: {deployed_thresh:.3f}")
print(f"\n  KEY COMPARISON at closest threshold:")
closest_thresh = min(prob_thresholds, key=lambda x: abs(x - deployed_thresh))
for period in periods:
    row = df_cm[(df_cm['period'] == period) & (df_cm['thresh'] == closest_thresh)].iloc[0]
    print(f"    {period}: Precision={row['prec']*100:.1f}%, Recall={row['recall']*100:.1f}%, "
          f"TP={row['tp']}, FP={row['fp']}, FN={row['fn']}, GO/day={row['go_per_day']:.1f}")

# Deployed vs Non-deployed breakdown
print(f"\n  DEPLOYED vs NON-DEPLOYED at threshold {closest_thresh}:")
for subset_label, subset_mask in [('Deployed', df['is_deployed']), ('Non-deployed', ~df['is_deployed'])]:
    df_sub = df[subset_mask]
    if len(df_sub) == 0: continue
    for period in periods:
        dp = df_sub[df_sub['period'] == period]
        if len(dp) == 0: continue
        pred_pos = dp['prob'] >= closest_thresh
        actual_pos = dp['actual_spike_10'] == 1
        tp = int((pred_pos & actual_pos).sum())
        fp = int((pred_pos & ~actual_pos).sum())
        fn = int((~pred_pos & actual_pos).sum())
        prec = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        print(f"    {subset_label:<13} {period}: Prec={prec*100:.1f}%, Recall={recall*100:.1f}%, "
              f"TP={tp}, FP={fp}, FN={fn}, Tickers={dp['ticker'].nunique()}")


# ==============================================================================
# TABLE 2: MARKET SPIKE AVAILABILITY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 2: MARKET SPIKE AVAILABILITY — Weekly spike rates (ALL bars)")
print(f"  If spikes disappeared in June, the model isn't wrong — the market changed")
print(f"{'='*W}")

spike_cols = {'≥3%': 'actual_spike_3', '≥5%': 'actual_spike_5',
              '≥8%': 'actual_spike_8', '≥10%': 'actual_spike_10'}

weekly = df.groupby('week').agg(
    n_bars=('ticker', 'size'),
    n_tickers=('ticker', 'nunique'),
    spike_3=('actual_spike_3', 'sum'),
    spike_5=('actual_spike_5', 'sum'),
    spike_8=('actual_spike_8', 'sum'),
    spike_10=('actual_spike_10', 'sum'),
    avg_max_gain=('max_gain', 'mean'),
).reset_index()

print(f"\n  {'Week':<12} {'Bars':>5} {'Tkrs':>5} {'≥3%':>5} {'rate':>6} {'≥5%':>5} {'rate':>6} "
      f"{'≥8%':>5} {'rate':>6} {'≥10%':>4} {'rate':>6} {'AvgGain':>8}")
print(f"  {'─'*12} {'─'*5} {'─'*5} {'─'*5} {'─'*6} {'─'*5} {'─'*6} {'─'*5} {'─'*6} {'─'*4} {'─'*6} {'─'*8}")

for _, r in weekly.iterrows():
    n = r['n_bars']
    print(f"  {r['week']:<12} {n:>5} {r['n_tickers']:>5} "
          f"{int(r['spike_3']):>5} {r['spike_3']/n*100:>5.1f}% "
          f"{int(r['spike_5']):>5} {r['spike_5']/n*100:>5.1f}% "
          f"{int(r['spike_8']):>5} {r['spike_8']/n*100:>5.1f}% "
          f"{int(r['spike_10']):>4} {r['spike_10']/n*100:>5.1f}% "
          f"{r['avg_max_gain']*100:>+7.2f}%")

# Monthly aggregate
print(f"\n  Monthly spike rates:")
for period in periods:
    dp = df[df['period'] == period]
    n = len(dp)
    if n == 0: continue
    print(f"    {period}: {n} bars, "
          f"≥3%: {dp['actual_spike_3'].mean()*100:.1f}%, "
          f"≥5%: {dp['actual_spike_5'].mean()*100:.1f}%, "
          f"≥8%: {dp['actual_spike_8'].mean()*100:.1f}%, "
          f"≥10%: {dp['actual_spike_10'].mean()*100:.1f}%, "
          f"avg max gain: {dp['max_gain'].mean()*100:+.2f}%")

# Per-ticker spike rate
print(f"\n  Per-ticker spike rate (≥10%) by period:")
print(f"  {'Ticker':<7} {'Dep':>3} {'April':>8} {'May':>8} {'June':>8} {'Total':>8}")
print(f"  {'─'*7} {'─'*3} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
for ticker in sorted(df['ticker'].unique()):
    is_dep = '*' if ticker in deployed_set else ' '
    parts = [f"  {ticker:<7} {is_dep:>3}"]
    for period in periods:
        dp = df[(df['ticker'] == ticker) & (df['period'] == period)]
        if len(dp) > 0:
            rate = dp['actual_spike_10'].mean() * 100
            parts.append(f"{rate:>7.1f}%")
        else:
            parts.append(f"{'—':>8}")
    dt = df[df['ticker'] == ticker]
    parts.append(f"{dt['actual_spike_10'].mean()*100:>7.1f}%")
    print(''.join(parts))
print(f"  (* = deployed)")


# ==============================================================================
# TABLE 3: PRECISION-RECALL BY MONTH × TICKER
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 3: PRECISION-RECALL BY MONTH × TICKER")
print(f"  Using each ticker's deployed probability threshold")
print(f"{'='*W}")

print(f"\n  {'Ticker':<7} {'Period':<7} {'Thresh':>6} {'TP':>4} {'FP':>4} {'FN':>4} "
      f"{'Prec':>6} {'Recall':>6} {'F1':>6} {'GO':>4} {'Spikes':>6}")
print(f"  {'─'*7} {'─'*7} {'─'*6} {'─'*4} {'─'*4} {'─'*4} {'─'*6} {'─'*6} {'─'*6} {'─'*4} {'─'*6}")

ticker_period_metrics = []
for ticker in sorted(df['ticker'].unique()):
    dt = df[df['ticker'] == ticker]
    thresh = dt['prob_thresh'].iloc[0]
    for period in periods:
        dp = dt[dt['period'] == period]
        if len(dp) == 0: continue
        pred_pos = dp['prob'] >= thresh
        actual_pos = dp['actual_spike_10'] == 1

        tp = int((pred_pos & actual_pos).sum())
        fp = int((pred_pos & ~actual_pos).sum())
        fn = int((~pred_pos & actual_pos).sum())

        prec = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * prec * recall / max(prec + recall, 1e-9)

        ticker_period_metrics.append({
            'ticker': ticker, 'period': period, 'thresh': thresh,
            'tp': tp, 'fp': fp, 'fn': fn, 'prec': prec, 'recall': recall,
            'f1': f1, 'n_go': int(pred_pos.sum()), 'n_spikes': int(actual_pos.sum()),
        })

        print(f"  {ticker:<7} {period:<7} {thresh:>5.3f} {tp:>4} {fp:>4} {fn:>4} "
              f"{prec*100:>5.1f}% {recall*100:>5.1f}% {f1*100:>5.1f}% "
              f"{int(pred_pos.sum()):>4} {int(actual_pos.sum()):>6}")

df_tpm = pd.DataFrame(ticker_period_metrics)


# ==============================================================================
# TABLE 4: CALIBRATION CURVE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 4: CALIBRATION CURVE — Prob bins vs actual spike rate, by month")
print(f"  Perfect calibration: prob bin 0.6 → 60% actual spike rate")
print(f"{'='*W}")

cal_bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]

print(f"\n  {'Bin':<10}", end='')
for period in periods:
    print(f"  {'N':>5} {'Actual%':>7}", end='')
print()
print(f"  {'─'*10}", end='')
for _ in periods:
    print(f"  {'─'*5} {'─'*7}", end='')
print()

cal_data = []
for lo, hi in cal_bins:
    label = f"{lo:.1f}-{hi:.1f}"
    print(f"  {label:<10}", end='')
    for period in periods:
        dp = df[(df['period'] == period) & (df['prob'] >= lo) & (df['prob'] < hi)]
        n = len(dp)
        if n > 0:
            actual = dp['actual_spike_10'].mean()
            cal_data.append({'bin_lo': lo, 'bin_hi': hi, 'period': period,
                             'n': n, 'actual_rate': actual, 'mid': (lo+hi)/2})
            print(f"  {n:>5} {actual*100:>6.1f}%", end='')
        else:
            print(f"  {'—':>5} {'—':>7}", end='')
    print()

df_cal = pd.DataFrame(cal_data)


# ==============================================================================
# TABLE 5: ROC / AUC BY MONTH
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 5: ROC / AUC BY MONTH")
print(f"  AUC drop = model losing discriminative power")
print(f"  AUC stable + precision drop = spike rates changed, model ordering still correct")
print(f"{'='*W}")

def compute_auc(probs, labels):
    if len(np.unique(labels)) < 2:
        return float('nan'), [], []
    sorted_indices = np.argsort(-probs)
    sorted_labels = labels[sorted_indices]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan'), [], []

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp_cum = 0
    fp_cum = 0
    for lab in sorted_labels:
        if lab == 1:
            tp_cum += 1
        else:
            fp_cum += 1
        tpr_list.append(tp_cum / n_pos)
        fpr_list.append(fp_cum / n_neg)

    auc_val = 0.0
    for j in range(1, len(fpr_list)):
        auc_val += (fpr_list[j] - fpr_list[j-1]) * (tpr_list[j] + tpr_list[j-1]) / 2
    return auc_val, fpr_list, tpr_list

roc_data = {}
print(f"\n  {'Period':<7} {'AUC':>6} {'N':>6} {'Pos':>5} {'Neg':>6} {'Pos%':>6}")
print(f"  {'─'*7} {'─'*6} {'─'*6} {'─'*5} {'─'*6} {'─'*6}")

for period in periods:
    dp = df[df['period'] == period]
    probs_arr = dp['prob'].values
    labels_arr = dp['actual_spike_10'].values.astype(int)
    auc_val, fpr, tpr = compute_auc(probs_arr, labels_arr)
    roc_data[period] = {'auc': auc_val, 'fpr': fpr, 'tpr': tpr}
    n_pos = int(labels_arr.sum())
    n_neg = len(labels_arr) - n_pos
    print(f"  {period:<7} {auc_val:>5.3f} {len(dp):>6} {n_pos:>5} {n_neg:>6} {n_pos/max(len(dp),1)*100:>5.1f}%")

# Overall AUC
probs_all = df['prob'].values
labels_all = df['actual_spike_10'].values.astype(int)
auc_overall, _, _ = compute_auc(probs_all, labels_all)
print(f"\n  Overall AUC: {auc_overall:.3f}")

if all(not np.isnan(roc_data[p]['auc']) for p in periods):
    auc_trend = roc_data['June']['auc'] - roc_data['April']['auc']
    print(f"  AUC trend (June - April): {auc_trend:+.3f}")
    if abs(auc_trend) < 0.03:
        print(f"  → STABLE: Model discriminative power unchanged")
    elif auc_trend < -0.03:
        print(f"  → DECLINING: Model losing ability to rank spike probability")
    else:
        print(f"  → IMPROVING: Model discrimination getting better")


# ==============================================================================
# TABLE 6: SIGNAL-TO-NOISE — GO SIGNAL QUALITY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 6: GO SIGNAL QUALITY — Only signals above deployed threshold")
print(f"{'='*W}")

print(f"\n  {'Period':<7} {'N':>4} {'AvgProb':>7} {'AvgGain':>8} {'MedGain':>8} {'StdGain':>8} "
      f"{'AvgDrop':>8} {'TP%':>5} {'SL%':>5} {'EOD%':>5} {'AvgPnL':>8} {'WR':>5}")
print(f"  {'─'*7} {'─'*4} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*5} {'─'*5} {'─'*5} {'─'*8} {'─'*5}")

for period in periods:
    go = df_go[df_go['period'] == period]
    if len(go) == 0:
        print(f"  {period:<7} {'—':>4}")
        continue
    n = len(go)
    pct_tp = (go['exit_type'] == 'TP').mean() * 100
    pct_sl = (go['exit_type'] == 'SL').mean() * 100
    pct_eod = (go['exit_type'] == 'EOD').mean() * 100
    wr = (go['trade_pnl'] > 0).mean() * 100
    print(f"  {period:<7} {n:>4} {go['prob'].mean():>6.3f} {go['max_gain'].mean()*100:>+7.2f}% "
          f"{go['max_gain'].median()*100:>+7.2f}% {go['max_gain'].std()*100:>7.2f}% "
          f"{go['max_drop'].mean()*100:>+7.2f}% "
          f"{pct_tp:>4.0f}% {pct_sl:>4.0f}% {pct_eod:>4.0f}% "
          f"{go['trade_pnl'].mean()*100:>+7.2f}% {wr:>4.0f}%")


# ==============================================================================
# TABLE 7: FALSE POSITIVE DEEP DIVE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 7: FALSE POSITIVE DEEP DIVE — GO signals where no ≥10% spike")
print(f"  Near-miss: ≥5% or ≥8% gain. Full wrong: <3% gain.")
print(f"{'='*W}")

for period in periods:
    go_p = df_go[df_go['period'] == period]
    fp = go_p[go_p['actual_spike_10'] == 0]
    if len(fp) == 0:
        print(f"\n  {period}: No false positives")
        continue
    n_fp = len(fp)
    near_8 = (fp['max_gain'] >= 0.08).sum()
    near_5 = (fp['max_gain'] >= 0.05).sum()
    near_3 = (fp['max_gain'] >= 0.03).sum()
    full_wrong = (fp['max_gain'] < 0.03).sum()

    print(f"\n  {period}: {n_fp} FP signals")
    print(f"    ≥8% gain (near-miss):   {near_8:>4} ({near_8/n_fp*100:>5.1f}%)")
    print(f"    ≥5% gain:               {near_5:>4} ({near_5/n_fp*100:>5.1f}%)")
    print(f"    ≥3% gain:               {near_3:>4} ({near_3/n_fp*100:>5.1f}%)")
    print(f"    <3% gain (full wrong):  {full_wrong:>4} ({full_wrong/n_fp*100:>5.1f}%)")
    print(f"    Avg max gain:           {fp['max_gain'].mean()*100:>+.2f}%")
    print(f"    Avg max drop:           {fp['max_drop'].mean()*100:>+.2f}%")
    print(f"    Avg PnL (traded):       {fp['trade_pnl'].mean()*100:>+.2f}%")

# June FP detail
fp_june = df_go[(df_go['period'] == 'June') & (df_go['actual_spike_10'] == 0)]
if len(fp_june) > 0:
    print(f"\n  June FP detail:")
    print(f"  {'Date':<12} {'Ticker':<7} {'Prob':>5} {'MaxGain':>8} {'MaxDrop':>8} {'PnL':>8} {'Exit':>4}")
    print(f"  {'─'*12} {'─'*7} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*4}")
    for _, r in fp_june.sort_values('date').iterrows():
        print(f"  {r['date']:<12} {r['ticker']:<7} {r['prob']:.2f} "
              f"{r['max_gain']*100:>+7.2f}% {r['max_drop']*100:>+7.2f}% "
              f"{r['trade_pnl']*100:>+7.2f}% {r['exit_type']:>4}")


# ==============================================================================
# TABLE 8: FALSE NEGATIVE AUDIT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 8: FALSE NEGATIVE AUDIT — Missed spikes (≥10%) the model didn't call")
print(f"{'='*W}")

for period in periods:
    dp = df[df['period'] == period]
    fn = dp[(dp['prob'] < dp['prob_thresh']) & (dp['actual_spike_10'] == 1)]
    total_spikes = dp['actual_spike_10'].sum()
    if total_spikes == 0:
        print(f"\n  {period}: No spikes occurred")
        continue

    print(f"\n  {period}: {len(fn)} missed spikes out of {total_spikes} total (miss rate: {len(fn)/total_spikes*100:.0f}%)")

    if len(fn) > 0:
        prob_bins_fn = [(0.0, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50)]
        print(f"    Prob distribution of missed spikes:")
        for lo, hi in prob_bins_fn:
            n_in_bin = ((fn['prob'] >= lo) & (fn['prob'] < hi)).sum()
            if n_in_bin > 0:
                print(f"      {lo:.2f}-{hi:.2f}: {n_in_bin} ({n_in_bin/len(fn)*100:.0f}%)")

        print(f"    Avg prob assigned: {fn['prob'].mean():.3f}")
        print(f"    Median prob:       {fn['prob'].median():.3f}")
        print(f"    Max prob:          {fn['prob'].max():.3f}")
        print(f"    Avg max gain:      {fn['max_gain'].mean()*100:+.2f}%")

        close_to_thresh = fn[fn['prob'] >= fn['prob_thresh'] - 0.05]
        print(f"    Within 5% of threshold: {len(close_to_thresh)} ({len(close_to_thresh)/len(fn)*100:.0f}%)")

        if len(fn) <= 30:
            print(f"\n    {'Date':<12} {'Ticker':<7} {'Prob':>5} {'Thresh':>6} {'MaxGain':>8}")
            print(f"    {'─'*12} {'─'*7} {'─'*5} {'─'*6} {'─'*8}")
            for _, r in fn.sort_values('prob', ascending=False).iterrows():
                gap = r['prob'] - r['prob_thresh']
                print(f"    {r['date']:<12} {r['ticker']:<7} {r['prob']:.2f} {r['prob_thresh']:.3f} "
                      f"{r['max_gain']*100:>+7.2f}%  (gap: {gap:+.3f})")


# ==============================================================================
# TABLE 9: EXPERIMENT COMPARISON (152 vs 160)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 9: EXPERIMENT COMPARISON — Which experiment family performs better?")
print(f"{'='*W}")

experiments = sorted(df['experiment'].unique())

print(f"\n  Monthly metrics by experiment:")
print(f"  {'Exp':<5} {'Period':<7} {'N':>5} {'Prec':>6} {'Recall':>6} {'AUC':>6} {'SpikeRate':>9} {'GO':>4}")
print(f"  {'─'*5} {'─'*7} {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*9} {'─'*4}")

for exp in experiments:
    de = df[df['experiment'] == exp]
    for period in periods:
        dp = de[de['period'] == period]
        if len(dp) == 0: continue

        pred_pos = dp['prob'] >= dp['prob_thresh']
        actual_pos = dp['actual_spike_10'] == 1
        tp = int((pred_pos & actual_pos).sum())
        fp = int((pred_pos & ~actual_pos).sum())
        fn = int((~pred_pos & actual_pos).sum())
        prec = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        probs_arr = dp['prob'].values
        labels_arr = dp['actual_spike_10'].values.astype(int)
        auc_val, _, _ = compute_auc(probs_arr, labels_arr)

        spike_rate = dp['actual_spike_10'].mean()
        n_go = int(pred_pos.sum())

        print(f"  {exp:<5} {period:<7} {len(dp):>5} {prec*100:>5.1f}% {recall*100:>5.1f}% "
              f"{auc_val:>5.3f} {spike_rate*100:>8.1f}% {n_go:>4}")

# Per-ticker winner
if len(experiments) >= 2:
    print(f"\n  Per-ticker June precision (which experiment wins):")
    print(f"  {'Ticker':<7}", end='')
    for exp in experiments:
        print(f"  {'Exp'+exp+' Prec':>12}", end='')
    print(f"  {'Winner':>7}")
    print(f"  {'─'*7}", end='')
    for _ in experiments:
        print(f"  {'─'*12}", end='')
    print(f"  {'─'*7}")

    for ticker in sorted(df['ticker'].unique()):
        dt_jun = df[(df['ticker'] == ticker) & (df['period'] == 'June')]
        if len(dt_jun) == 0: continue
        precs = {}
        print(f"  {ticker:<7}", end='')
        for exp in experiments:
            de = dt_jun[dt_jun['experiment'] == exp]
            if len(de) == 0:
                print(f"  {'—':>12}", end='')
                continue
            pred_pos = de['prob'] >= de['prob_thresh']
            actual_pos = de['actual_spike_10'] == 1
            tp = int((pred_pos & actual_pos).sum())
            fp = int((pred_pos & ~actual_pos).sum())
            prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else float('nan')
            precs[exp] = prec
            if np.isnan(prec):
                print(f"  {'no GO':>12}", end='')
            else:
                print(f"  {prec*100:>11.1f}%", end='')
        valid_precs = {k: v for k, v in precs.items() if not np.isnan(v)}
        if valid_precs:
            winner = max(valid_precs, key=valid_precs.get)
            print(f"  {winner:>7}")
        else:
            print(f"  {'—':>7}")


# ==============================================================================
# TABLE 10: VERDICT SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 10: VERDICT SUMMARY — Model vs Market Diagnosis")
print(f"{'='*W}")

# 1. Market spike availability
apr_spike = df[df['period'] == 'April']['actual_spike_10'].mean() if len(df[df['period'] == 'April']) > 0 else 0
may_spike = df[df['period'] == 'May']['actual_spike_10'].mean() if len(df[df['period'] == 'May']) > 0 else 0
jun_spike = df[df['period'] == 'June']['actual_spike_10'].mean() if len(df[df['period'] == 'June']) > 0 else 0

print(f"\n  1. MARKET SPIKE AVAILABILITY:")
print(f"     April: {apr_spike*100:.1f}%  |  May: {may_spike*100:.1f}%  |  June: {jun_spike*100:.1f}%")
if apr_spike > 0:
    jun_vs_apr = (jun_spike - apr_spike) / apr_spike * 100
    print(f"     June vs April change: {jun_vs_apr:+.0f}%")
    if jun_vs_apr < -30:
        print(f"     → SPIKE DROUGHT: Significantly fewer spikes in June")
    elif jun_vs_apr < -10:
        print(f"     → MILD DECLINE: Somewhat fewer spikes")
    else:
        print(f"     → STABLE: Spike availability unchanged")

# 2. Model discrimination (AUC)
print(f"\n  2. MODEL DISCRIMINATION (AUC):")
for period in periods:
    auc_val = roc_data[period]['auc']
    print(f"     {period}: {auc_val:.3f}")
if all(not np.isnan(roc_data[p]['auc']) for p in periods):
    auc_change = roc_data['June']['auc'] - roc_data['April']['auc']
    if abs(auc_change) < 0.03:
        print(f"     → STABLE: Model can still rank spike probability correctly")
    elif auc_change < -0.03:
        print(f"     → DECLINING: Model discrimination weakening")
    else:
        print(f"     → IMPROVING")

# 3. Calibration shift
print(f"\n  3. CALIBRATION SHIFT:")
for period in periods:
    cal_p = df_cal[df_cal['period'] == period]
    high_bins = cal_p[cal_p['bin_lo'] >= 0.5]
    if len(high_bins) > 0:
        avg_pred = high_bins['mid'].mean()
        avg_actual = high_bins['actual_rate'].mean()
        print(f"     {period}: Avg predicted (high bins): {avg_pred*100:.0f}%, "
              f"Avg actual: {avg_actual*100:.1f}%")

# 4. Precision / Recall
print(f"\n  4. PRECISION / RECALL (at deployed threshold):")
for period in periods:
    dp = df[df['period'] == period]
    pred_pos = dp['prob'] >= dp['prob_thresh']
    actual_pos = dp['actual_spike_10'] == 1
    tp = int((pred_pos & actual_pos).sum())
    fp = int((pred_pos & ~actual_pos).sum())
    fn = int((~pred_pos & actual_pos).sum())
    prec = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"     {period}: Precision={prec*100:.1f}%, Recall={recall*100:.1f}%, TP={tp}, FP={fp}, FN={fn}")

# 5. Per-ticker breakdown
print(f"\n  5. PER-TICKER CONCENTRATION:")
if len(df_tpm) > 0:
    june_metrics = df_tpm[df_tpm['period'] == 'June']
    if len(june_metrics) > 0:
        zero_prec = june_metrics[june_metrics['prec'] == 0]
        high_prec = june_metrics[june_metrics['prec'] >= 0.30]
        no_go = june_metrics[june_metrics['n_go'] == 0]
        print(f"     June: {len(june_metrics)} tickers scored")
        print(f"       Zero precision (all FP): {len(zero_prec)} tickers — {sorted(zero_prec['ticker'].tolist())}")
        print(f"       High precision (≥30%):   {len(high_prec)} tickers — {sorted(high_prec['ticker'].tolist())}")
        print(f"       No GO signals:           {len(no_go)} tickers — {sorted(no_go['ticker'].tolist())}")

# 6. Final verdict
print(f"\n  6. DIAGNOSIS:")
if apr_spike > 0 and jun_spike / apr_spike < 0.5:
    print(f"     PRIMARY: Market spike drought — {jun_vs_apr:+.0f}% fewer 10%+ spikes in June")
    if all(not np.isnan(roc_data[p]['auc']) for p in periods) and abs(roc_data['June']['auc'] - roc_data['April']['auc']) < 0.05:
        print(f"     SECONDARY: Model discrimination stable (AUC unchanged) — model is NOT broken")
        print(f"     → The model is calling signals in a market that isn't producing spikes")
    else:
        print(f"     SECONDARY: Model discrimination also declining — both market AND model contributing")
elif all(not np.isnan(roc_data[p]['auc']) for p in periods) and roc_data['June']['auc'] < roc_data['April']['auc'] - 0.05:
    print(f"     PRIMARY: Model failure — AUC dropped from {roc_data['April']['auc']:.3f} to {roc_data['June']['auc']:.3f}")
    print(f"     → Model losing ability to discriminate spikes from non-spikes")
else:
    print(f"     MIXED: Examine per-ticker results for concentrated failures vs systemic issues")
    print(f"     → Check if specific tickers are dragging down aggregate metrics")


# ==============================================================================
# CHARTS (matplotlib)
# ==============================================================================
if HAS_MPL:
    print(f"\n\n{'='*W}")
    print(f"  GENERATING CHARTS...")
    print(f"{'='*W}")

    colors = {'April': '#2196F3', 'May': '#4CAF50', 'June': '#F44336'}
    tickers_sorted = sorted(df['ticker'].unique())

    # ── PAGE 1: Core Diagnostics (2x2) ──────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Model Predictive Power — Core Diagnostics', fontsize=14, fontweight='bold')

    # 1A: Calibration curves
    ax = axes1[0, 0]
    for period in periods:
        cal_p = df_cal[df_cal['period'] == period]
        if len(cal_p) > 0:
            ax.plot(cal_p['mid'], cal_p['actual_rate'], 'o-', label=period,
                    color=colors[period], linewidth=2, markersize=6)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect')
    ax.set_xlabel('Predicted Probability (bin midpoint)')
    ax.set_ylabel('Actual Spike Rate (≥10%)')
    ax.set_title('Calibration Curve: Model Probability vs Reality')
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 1B: ROC curves
    ax = axes1[0, 1]
    for period in periods:
        rd = roc_data[period]
        if rd['fpr'] and rd['tpr']:
            ax.plot(rd['fpr'], rd['tpr'], label=f"{period} (AUC={rd['auc']:.3f})",
                    color=colors[period], linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves by Month')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1C: Weekly precision + spike rate timeline
    ax = axes1[1, 0]
    weekly_prec = []
    for _, wr in weekly.iterrows():
        wk = wr['week']
        dp = df_go[df_go['week'] == wk]
        prec = (dp['actual_spike_10'] == 1).mean() if len(dp) > 0 else float('nan')
        spike_rate = wr['spike_10'] / wr['n_bars'] if wr['n_bars'] > 0 else 0
        weekly_prec.append({'week': wk, 'precision': prec, 'spike_rate': spike_rate})
    wdf = pd.DataFrame(weekly_prec)
    x = range(len(wdf))
    ax.bar(x, wdf['spike_rate'] * 100, alpha=0.3, color='steelblue', label='Spike rate (all bars)')
    ax2_twin = ax.twinx()
    valid_mask = ~wdf['precision'].isna()
    ax2_twin.plot([i for i, v in enumerate(valid_mask) if v],
             wdf.loc[valid_mask, 'precision'] * 100, 'ro-', linewidth=2,
             markersize=6, label='GO precision')
    ax.set_xticks(x)
    ax.set_xticklabels(wdf['week'], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Spike Rate (%)', color='steelblue')
    ax2_twin.set_ylabel('GO Signal Precision (%)', color='red')
    ax.set_title('Weekly Spike Rate vs GO Precision')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    # 1D: Probability distribution by month (histogram)
    ax = axes1[1, 1]
    bins_hist = np.arange(0, 1.05, 0.05)
    for period in periods:
        dp = df[df['period'] == period]
        if len(dp) > 0:
            ax.hist(dp['prob'], bins=bins_hist, alpha=0.4, color=colors[period],
                    label=f"{period} (n={len(dp)})", density=True, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Model Probability')
    ax.set_ylabel('Density')
    ax.set_title('Probability Distribution by Month')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = '/tmp/model_power_1_core.png'
    fig1.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Page 1 saved: {p1}")

    # ── PAGE 2: Per-Ticker Analysis (2x2) ───────────────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 14))
    fig2.suptitle('Model Predictive Power — Per-Ticker Analysis', fontsize=14, fontweight='bold')

    # 2A: Per-ticker precision heatmap (ticker × month)
    ax = axes2[0, 0]
    heatmap_data = []
    for ticker in tickers_sorted:
        row = []
        for period in periods:
            dp = df[(df['ticker'] == ticker) & (df['period'] == period)]
            pred_pos = dp['prob'] >= dp['prob_thresh']
            actual_pos = dp['actual_spike_10'] == 1
            tp = int((pred_pos & actual_pos).sum())
            fp = int((pred_pos & ~actual_pos).sum())
            prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else float('nan')
            row.append(prec)
        heatmap_data.append(row)
    heatmap_arr = np.array(heatmap_data)
    im = ax.imshow(heatmap_arr, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods)
    ax.set_yticks(range(len(tickers_sorted)))
    ax.set_yticklabels(tickers_sorted, fontsize=7)
    ax.set_title('Precision Heatmap (Ticker × Month)')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Precision')
    for i in range(len(tickers_sorted)):
        for j in range(len(periods)):
            val = heatmap_arr[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.0f}%", ha='center', va='center',
                        fontsize=6, color='black' if val > 0.3 else 'white')

    # 2B: Per-ticker spike rate heatmap (ticker × month)
    ax = axes2[0, 1]
    spike_heatmap = []
    for ticker in tickers_sorted:
        row = []
        for period in periods:
            dp = df[(df['ticker'] == ticker) & (df['period'] == period)]
            rate = dp['actual_spike_10'].mean() if len(dp) > 0 else float('nan')
            row.append(rate)
        spike_heatmap.append(row)
    spike_arr = np.array(spike_heatmap)
    im2 = ax.imshow(spike_arr, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.3)
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods)
    ax.set_yticks(range(len(tickers_sorted)))
    ax.set_yticklabels(tickers_sorted, fontsize=7)
    ax.set_title('Spike Rate Heatmap (Ticker × Month)')
    plt.colorbar(im2, ax=ax, shrink=0.8, label='Spike Rate')
    for i in range(len(tickers_sorted)):
        for j in range(len(periods)):
            val = spike_arr[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.0f}%", ha='center', va='center',
                        fontsize=6, color='black' if val < 0.15 else 'white')

    # 2C: Per-ticker AUC bar chart
    ax = axes2[1, 0]
    ticker_aucs = {}
    for ticker in tickers_sorted:
        dt = df[df['ticker'] == ticker]
        if dt['actual_spike_10'].nunique() < 2:
            ticker_aucs[ticker] = float('nan')
            continue
        y_true = dt['actual_spike_10'].values
        y_score = dt['prob'].values
        try:
            from sklearn.metrics import roc_auc_score as _roc_auc
            ticker_aucs[ticker] = _roc_auc(y_true, y_score)
        except Exception:
            ticker_aucs[ticker] = float('nan')
    auc_vals = [ticker_aucs.get(t, float('nan')) for t in tickers_sorted]
    bar_colors = ['#2196F3' if t in deployed_set else '#90CAF9' for t in tickers_sorted]
    bars = ax.barh(range(len(tickers_sorted)), auc_vals, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.set_yticks(range(len(tickers_sorted)))
    ax.set_yticklabels(tickers_sorted, fontsize=7)
    ax.set_xlabel('AUC')
    ax.set_title('Per-Ticker AUC (dark=deployed, light=non-deployed)')
    ax.set_xlim(0.3, 1.0)
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(auc_vals):
        if not np.isnan(v):
            ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=6)

    # 2D: FP/TP breakdown by ticker (stacked bar)
    ax = axes2[1, 1]
    tp_counts = []
    fp_counts = []
    for ticker in tickers_sorted:
        dt = df[df['ticker'] == ticker]
        pred_pos = dt['prob'] >= dt['prob_thresh']
        actual_pos = dt['actual_spike_10'] == 1
        tp_counts.append(int((pred_pos & actual_pos).sum()))
        fp_counts.append(int((pred_pos & ~actual_pos).sum()))
    y_pos = range(len(tickers_sorted))
    ax.barh(y_pos, tp_counts, color='#4CAF50', label='TP', edgecolor='white', linewidth=0.5)
    ax.barh(y_pos, fp_counts, left=tp_counts, color='#F44336', label='FP', edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tickers_sorted, fontsize=7)
    ax.set_xlabel('Number of GO Signals')
    ax.set_title('TP vs FP by Ticker')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = '/tmp/model_power_2_tickers.png'
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Page 2 saved: {p2}")

    # ── PAGE 3: Signal Quality & Gains (2x2) ────────────────────────────────
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
    fig3.suptitle('Model Predictive Power — Signal Quality & Gain Analysis', fontsize=14, fontweight='bold')

    # 3A: Max gain distribution by month (GO signals only)
    ax = axes3[0, 0]
    for period in periods:
        dp = df_go[df_go['period'] == period]
        if len(dp) > 0:
            gains = dp['max_gain'].clip(-0.05, 0.25) * 100
            ax.hist(gains, bins=30, alpha=0.4, color=colors[period],
                    label=f"{period} (n={len(dp)})", density=True, edgecolor='white', linewidth=0.5)
    ax.axvline(x=10, color='black', linestyle='--', alpha=0.5, label='10% spike threshold')
    ax.set_xlabel('Max Gain (%)')
    ax.set_ylabel('Density')
    ax.set_title('Max Gain Distribution (GO Signals Only)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3B: Scatter — probability vs max gain (all bars)
    ax = axes3[0, 1]
    sample = df.sample(n=min(5000, len(df)), random_state=42) if len(df) > 5000 else df
    for period in periods:
        dp = sample[sample['period'] == period]
        if len(dp) > 0:
            ax.scatter(dp['prob'], dp['max_gain'] * 100, alpha=0.15, s=8,
                       color=colors[period], label=period)
    ax.axhline(y=10, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Model Probability')
    ax.set_ylabel('Max Gain (%)')
    ax.set_title('Probability vs Actual Max Gain')
    ax.set_ylim(-10, 30)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3C: FP near-miss analysis — max gain distribution of false positives
    ax = axes3[1, 0]
    fp_df = df_go[df_go['actual_spike_10'] == 0]
    if len(fp_df) > 0:
        fp_gains = fp_df['max_gain'] * 100
        n_bins_fp = min(30, max(5, len(fp_gains) // 3))
        ax.hist(fp_gains, bins=n_bins_fp, color='#F44336', alpha=0.6, edgecolor='white', linewidth=0.5)
        pct_8 = (fp_df['max_gain'] >= 0.08).mean() * 100
        pct_5 = (fp_df['max_gain'] >= 0.05).mean() * 100
        ax.axvline(x=10, color='black', linestyle='--', alpha=0.5, label='10% threshold')
        ax.axvline(x=8, color='orange', linestyle='--', alpha=0.5, label=f'8% ({pct_8:.0f}% of FP)')
        ax.axvline(x=5, color='blue', linestyle='--', alpha=0.5, label=f'5% ({pct_5:.0f}% of FP)')
        ax.set_xlabel('Max Gain (%)')
        ax.set_ylabel('Count')
        ax.set_title('False Positive Max Gains — Near Misses?')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No false positives', transform=ax.transAxes, ha='center')
        ax.set_title('False Positive Max Gains')
    ax.grid(True, alpha=0.3)

    # 3D: Deployed vs Non-deployed precision comparison
    ax = axes3[1, 1]
    dep_labels = ['Deployed', 'Non-deployed']
    dep_precs = {}
    for label, is_dep in [('Deployed', True), ('Non-deployed', False)]:
        dep_precs[label] = []
        for period in periods:
            if is_dep:
                dp = df[(df['is_deployed'] == True) & (df['period'] == period)]
            else:
                dp = df[(df['is_deployed'] == False) & (df['period'] == period)]
            pred_pos = dp['prob'] >= dp['prob_thresh']
            actual_pos = dp['actual_spike_10'] == 1
            tp = int((pred_pos & actual_pos).sum())
            fp = int((pred_pos & ~actual_pos).sum())
            prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else float('nan')
            dep_precs[label].append(prec * 100 if not np.isnan(prec) else 0)

    x_dep = np.arange(len(periods))
    w = 0.35
    ax.bar(x_dep - w/2, dep_precs['Deployed'], w, color='#1565C0', label='Deployed', edgecolor='white')
    ax.bar(x_dep + w/2, dep_precs['Non-deployed'], w, color='#90CAF9', label='Non-deployed', edgecolor='white')
    ax.set_xticks(x_dep)
    ax.set_xticklabels(periods)
    ax.set_ylabel('Precision (%)')
    ax.set_title('Deployed vs Non-Deployed Precision')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    p3 = '/tmp/model_power_3_signals.png'
    fig3.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Page 3 saved: {p3}")

    # ── PAGE 4: Experiment Comparison & Time Series (2x2) ───────────────────
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12))
    fig4.suptitle('Model Predictive Power — Experiment Comparison & Trends', fontsize=14, fontweight='bold')

    # 4A: Experiment precision comparison (grouped bar)
    ax = axes4[0, 0]
    exp_ids = sorted(df['experiment'].unique())
    exp_colors_list = ['#1565C0', '#E65100', '#2E7D32', '#6A1B9A']
    x_exp = np.arange(len(periods))
    bar_w = 0.8 / max(len(exp_ids), 1)
    for ei, exp_id in enumerate(exp_ids):
        precs_exp = []
        for period in periods:
            dp = df[(df['experiment'] == exp_id) & (df['period'] == period)]
            pred_pos = dp['prob'] >= dp['prob_thresh']
            actual_pos = dp['actual_spike_10'] == 1
            tp = int((pred_pos & actual_pos).sum())
            fp = int((pred_pos & ~actual_pos).sum())
            prec = tp / max(tp + fp, 1) * 100 if (tp + fp) > 0 else 0
            precs_exp.append(prec)
        offset = (ei - len(exp_ids)/2 + 0.5) * bar_w
        ax.bar(x_exp + offset, precs_exp, bar_w, label=f"Exp {exp_id}",
               color=exp_colors_list[ei % len(exp_colors_list)], edgecolor='white')
    ax.set_xticks(x_exp)
    ax.set_xticklabels(periods)
    ax.set_ylabel('Precision (%)')
    ax.set_title('Precision by Experiment × Month')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 4B: Daily GO signal count + precision time series
    ax = axes4[0, 1]
    df['date_dt'] = pd.to_datetime(df['date'])
    daily_stats = []
    for date_val in sorted(df['date_dt'].unique()):
        dd = df[df['date_dt'] == date_val]
        dd_go = dd[dd['prob'] >= dd['prob_thresh']]
        n_go = len(dd_go)
        prec_d = dd_go['actual_spike_10'].mean() if n_go > 0 else float('nan')
        daily_stats.append({'date': date_val, 'n_go': n_go, 'precision': prec_d})
    ddf = pd.DataFrame(daily_stats)
    ax.bar(range(len(ddf)), ddf['n_go'], alpha=0.4, color='steelblue', label='GO signals')
    ax_twin2 = ax.twinx()
    valid_d = ~ddf['precision'].isna()
    ax_twin2.plot([i for i, v in enumerate(valid_d) if v],
                  ddf.loc[valid_d, 'precision'] * 100, 'ro-', markersize=3,
                  linewidth=1, alpha=0.8, label='Precision')
    ax.set_xticks(range(0, len(ddf), max(1, len(ddf)//10)))
    date_labels = [d.strftime('%m/%d') for d in ddf['date']]
    ax.set_xticklabels([date_labels[i] for i in range(0, len(ddf), max(1, len(ddf)//10))],
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('# GO Signals', color='steelblue')
    ax_twin2.set_ylabel('Precision (%)', color='red')
    ax.set_title('Daily GO Signals & Precision')
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax_twin2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, loc='upper right', fontsize=7)

    # 4C: Cumulative spike rate over time (deployed vs all)
    ax = axes4[1, 0]
    for label, subset_mask in [('All tickers', df['ticker'].notna()),
                                ('Deployed only', df['is_deployed'] == True)]:
        ds = df[subset_mask].sort_values('date_dt')
        if len(ds) > 0:
            cumspike = ds['actual_spike_10'].expanding().mean() * 100
            ax.plot(range(len(cumspike)), cumspike.values, linewidth=1.5, label=label)
    ax.set_xlabel('Bar Index (chronological)')
    ax.set_ylabel('Cumulative Spike Rate (%)')
    ax.set_title('Cumulative Spike Rate Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4D: Probability threshold sensitivity — precision vs threshold
    ax = axes4[1, 1]
    thresholds_range = np.arange(0.30, 0.80, 0.02)
    for period in periods:
        dp = df[df['period'] == period]
        precs_thr = []
        for thr in thresholds_range:
            pred_pos = dp['prob'] >= thr
            actual_pos = dp['actual_spike_10'] == 1
            tp = int((pred_pos & actual_pos).sum())
            fp = int((pred_pos & ~actual_pos).sum())
            prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else float('nan')
            precs_thr.append(prec * 100)
        ax.plot(thresholds_range * 100, precs_thr, color=colors[period],
                linewidth=2, label=period)
    ax.set_xlabel('Probability Threshold (%)')
    ax.set_ylabel('Precision (%)')
    ax.set_title('Precision vs Threshold — Could Higher Threshold Fix June?')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=30, color='gray', linestyle=':', alpha=0.5)

    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    p4 = '/tmp/model_power_4_experiments.png'
    fig4.savefig(p4, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"  Page 4 saved: {p4}")

    print(f"\n  All charts saved:")
    print(f"    {p1}")
    print(f"    {p2}")
    print(f"    {p3}")
    print(f"    {p4}")

else:
    print(f"\n  matplotlib not available — skipping charts")


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ANALYSIS COMPLETE")
print(f"  {len(df)} bars scored across {df['ticker'].nunique()} tickers")
print(f"  {len(df_go)} GO signals, {df['actual_spike_10'].sum()} actual spikes")
print(f"{'='*W}")
