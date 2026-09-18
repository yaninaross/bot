"""
analyze_june_deep.py — Deep Analysis: Why V9 Failed in June 2026
================================================================
Loads master_inference data + runs TF model inference for all v9 deployed
tickers. Captures raw features, probabilities, and price paths on every
signal day, then runs 7 analysis dimensions:

  1. Feature Distribution Shift (KS-test, April-May vs June)
  2. Pre-Market Microstructure Breakdown (per ticker)
  3. Intraday Price Action Profiles (loss categorization)
  4. Cross-Ticker Loss Correlation
  5. Model Probability Calibration Decay
  6. Context Feature Analysis (SPY/VIX/TLT/IWM)
  7. Spike Rate Decay Timeline (weekly, multi-threshold)

Run in Colab: paste as cell. Requires: TF, pandas_ta, numba, GCS, BQ.
"""

import os, io, gc, sys, json, subprocess, warnings, ctypes
from datetime import datetime, timedelta
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]", "pandas_ta",
                           "google-cloud-storage", "pyarrow", "gcsfs",
                           "numba", "scipy", "--quiet"])
    import tensorflow as tf

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

from scipy import stats as sp_stats
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
    Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2

# ==============================================================================
# CONFIG
# ==============================================================================
VERSION = 'robust_wf_v9'
WINNING_SCORE = 'S8_mix'
TEST_START = '2026-04-01'
TEST_END   = '2026-06-08'
JUNE_START = '2026-06-01'

BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")
BQ_TABLE       = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
INFERENCE_DIR  = "Data/Inference_v2"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR      = "Data/Inference_v2"
OHLCV_DIR      = "Data/Inference"

TIME_STEPS = 240; EMBED_DIM = 16; EXIT_TIME_MIN = 780
PST = pytz.timezone('America/Los_Angeles')
SLIPPAGE_PCT = 0.0015
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

# ==============================================================================
# CORE FUNCTIONS (from compare_composite_scores.py)
# ==============================================================================
_master_cache = {}
_model_cache = {}
_stats_cache = {}
_market_context_cache = None

def get_asset_index(model_id, ticker):
    teachers = FAMILY_TEACHERS.get(model_id, TEACHERS_13)
    return {t: i for i, t in enumerate(teachers)}.get(ticker, len(teachers))

def get_n_assets(model_id):
    return len(FAMILY_TEACHERS.get(model_id, TEACHERS_13)) + 1

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
        cols = {f'{clean}_Anchor_ROC': (d['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{clean}_ROC_{h}m'] = d['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=d.index))
        del d, daily_c; _release_memory()
        print(f"    {clean} loaded")
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

def compute_atr(df_ohlcv, period=14):
    daily = df_ohlcv.resample('D').agg({'High':'max','Low':'min','Close':'last'}).dropna()
    pc = daily['Close'].shift(1)
    tr = pd.concat([daily['High']-daily['Low'],(daily['High']-pc).abs(),(daily['Low']-pc).abs()],axis=1).max(axis=1)
    return (tr.rolling(period).mean() / daily['Close']).reindex(df_ohlcv.index, method='ffill')

# ==============================================================================
# COMPOSITE SCORE (S8_mix only)
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
# STEP 1: LOAD V9 PORTFOLIO FROM BQ
# ==============================================================================
print(f"{'='*W}")
print(f"  DEEP ANALYSIS — Why V9 Failed in June 2026")
print(f"  Period: {TEST_START} → {TEST_END}  |  June starts: {JUNE_START}")
print(f"{'='*W}")

q = f"""
SELECT ticker, experiment, seed, tpsl_mode, prob_thresh, model_version,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       tp_pct, sl_pct, tp_mult, sl_mult, trail_start_mult, trail_end_mult,
       vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
       entry_minute,
       val_a_start, val_a_end, val_b_start, val_b_end,
       survived_a, survived_b, is_ensemble
FROM `{BQ_TABLE}`
WHERE model_version = '{VERSION}'
  AND survived_a = TRUE
  AND val_b_start = '{TEST_START}'
"""
print("  Loading BQ data...")
df_all = bq_client.query(q).to_dataframe()
df_all = df_all.drop_duplicates(subset=['ticker','experiment','seed','tpsl_mode','prob_thresh'], keep='first')
df_all = compute_s8_mix(df_all)
valid = df_all[df_all['S8_mix'] > 0]
portfolio = valid.loc[valid.groupby('ticker')['S8_mix'].idxmax()]
print(f"  {len(df_all)} configs → {len(portfolio)} deployed tickers")

# Build config lookup
ticker_configs = {}
for _, r in portfolio.iterrows():
    exp = r['experiment']
    mode_sfx = MODE_SUFFIX.get(r['tpsl_mode'], '')
    base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp
    seed_val = int(r['seed'])
    is_ens = bool(r['is_ensemble'])

    # FIX: WF optimizer stores positional index (0-14) as "seed" for stitched
    # 170-175 walkforward. Remap to fold-175 models (latest cutoff), matching
    # daily_score_combined.py which uses model_id='175_XSECTOR'.
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
print(f"  Deployed: {deployed_tickers}")
remapped = {t: c for t, c in ticker_configs.items() if c['base_exp'] == '175_XSECTOR'}
if remapped:
    print(f"  Remapped 170→175: {list(remapped.keys())}")

# ==============================================================================
# STEP 2: LOAD DATA + RUN MODEL INFERENCE + CAPTURE EVERYTHING
# ==============================================================================
print(f"\n  Loading OHLCV + running model inference for {len(deployed_tickers)} tickers...")
market_df = build_market_context(start_date=TEST_START)

test_start_utc = pd.Timestamp(TEST_START, tz='UTC')
test_end_utc = pd.Timestamp(TEST_END, tz='UTC') + timedelta(days=1)
june_start_utc = pd.Timestamp(JUNE_START, tz='UTC')
trim_start = test_start_utc - pd.Timedelta(days=5)

all_signals = []
ticker_ohlcv_pst = {}
ticker_daily_returns = {}

for ci, ticker in enumerate(deployed_tickers):
    cfg = ticker_configs[ticker]
    print(f"  [{ci+1}/{len(deployed_tickers)}] {ticker}/{cfg['base_exp']}/S{cfg['seed']}...", end=' ')

    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        df_raw = pd.read_parquet(uri)
    except Exception:
        print("SKIP"); continue
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

    # Keep OHLCV in PST for price path extraction
    df_pst = df_raw.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    test_s_pst = pd.Timestamp(TEST_START, tz=PST)
    test_e_pst = pd.Timestamp(TEST_END, tz=PST) + timedelta(days=1)
    ohlcv_test = df_pst[(df_pst.index >= test_s_pst) & (df_pst.index < test_e_pst)]
    if not ohlcv_test.empty:
        ticker_ohlcv_pst[ticker] = ohlcv_test

        # Daily returns for cross-ticker correlation
        daily = ohlcv_test[ohlcv_test['Volume'] > 0].resample('D').agg(
            {'Open': 'first', 'Close': 'last'}).dropna()
        if len(daily) > 0:
            daily['ret'] = daily['Close'] / daily['Open'] - 1
            ticker_daily_returns[ticker] = daily['ret']

    # Engineer features (raw = unscaled for distribution analysis)
    df_feat = engineer_features(df_raw.copy(), market_df)
    stats = load_scaling_stats(cfg['base_exp'])
    df_scaled = apply_scaling(df_feat, stats)

    # Model inference
    asset_id = get_asset_index(cfg['base_exp'], ticker)
    df_test = df_scaled[(df_scaled.index >= test_start_utc) & (df_scaled.index < test_end_utc)]
    if len(df_test) == 0:
        del df_raw, df_feat, df_scaled, df_pst; _release_memory()
        print("SKIP"); continue

    em = cfg['entry_minute']
    pst_idx = df_test.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute
    positions = [df_scaled.index.get_loc(df_test.index[i])
                 for i in range(len(df_test))
                 if mins[i] == em and df_scaled.index.get_loc(df_test.index[i]) >= TIME_STEPS
                 and ('Volume' not in df_scaled.columns or float(df_scaled['Volume'].iloc[df_scaled.index.get_loc(df_test.index[i])]) > 0)]

    if not positions:
        del df_raw, df_feat, df_scaled, df_pst; _release_memory()
        print("no positions"); continue

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

    # Collect signals with raw features, context, and price paths
    for i, idx in enumerate(positions):
        ts = df_scaled.index[idx]
        ts_pst = ts.tz_convert(PST)
        date_int = ts_pst.year*10000 + ts_pst.month*100 + ts_pst.day
        prob = float(probs[i])

        # Raw (unscaled) feature values at entry bar
        raw_feats = {}
        for col in ASSET_COLS:
            raw_feats[col] = float(df_feat[col].iloc[df_feat.index.get_loc(ts)] if ts in df_feat.index else 0)
        for col in CONTEXT_COLS:
            raw_feats[col] = float(df_feat[col].iloc[df_feat.index.get_loc(ts)] if ts in df_feat.index else 0)

        # Scaled context vector (what model actually sees)
        scaled_context = {}
        for col in CONTEXT_COLS:
            scaled_context[col] = float(XC[i, CONTEXT_COLS.index(col)])

        # Intraday price path from entry to EOD
        entry_price = float(df_pst['Close'].iloc[df_pst.index.get_indexer([ts_pst], method='nearest')[0]])
        day_mask = (ohlcv_test.index.date == ts_pst.date()) & (ohlcv_test.index >= ts_pst)
        day_bars = ohlcv_test[day_mask]

        max_gain = 0.0; max_drop = 0.0; eod_ret = 0.0
        max_gain_time = ''; max_drop_time = ''
        if len(day_bars) > 0 and entry_price > 0:
            highs_ret = (day_bars['High'].values / entry_price - 1)
            lows_ret = (day_bars['Low'].values / entry_price - 1)
            max_gain = float(np.max(highs_ret))
            max_drop = float(np.min(lows_ret))
            max_gain_idx = int(np.argmax(highs_ret))
            max_drop_idx = int(np.argmin(lows_ret))
            max_gain_time = day_bars.index[max_gain_idx].strftime('%H:%M') if max_gain_idx < len(day_bars) else ''
            max_drop_time = day_bars.index[max_drop_idx].strftime('%H:%M') if max_drop_idx < len(day_bars) else ''
            eod_price = float(day_bars['Close'].iloc[-1])
            eod_ret = (eod_price / entry_price - 1)

        # Determine effective TP/SL
        tp_eff = cfg['tp_pct'] if cfg['tp_pct'] > 0 else cfg['tp_mult'] * 0.05
        sl_eff = cfg['sl_pct'] if cfg['sl_pct'] > 0 else cfg['sl_mult'] * 0.05

        # Simulate trade outcome
        trade_pnl = eod_ret
        exit_type = 'EOD'
        if max_drop <= -sl_eff:
            trade_pnl = -sl_eff
            exit_type = 'SL'
        elif max_gain >= tp_eff:
            trade_pnl = tp_eff
            exit_type = 'TP'

        is_go = prob >= cfg['prob_thresh']
        period = 'june' if ts >= june_start_utc else 'apr_may'

        sig = {
            'date': date_int, 'date_str': ts_pst.strftime('%Y-%m-%d'),
            'ticker': ticker, 'prob': prob, 'threshold': cfg['prob_thresh'],
            'is_go': is_go, 'period': period,
            'entry_price': entry_price, 'entry_minute': em,
            'max_gain': max_gain, 'max_gain_time': max_gain_time,
            'max_drop': max_drop, 'max_drop_time': max_drop_time,
            'eod_ret': eod_ret, 'trade_pnl': trade_pnl, 'exit_type': exit_type,
            'tp_eff': tp_eff, 'sl_eff': sl_eff,
            'experiment': cfg['base_exp'],
        }
        sig.update({f'raw_{k}': v for k, v in raw_feats.items()})
        sig.update({f'ctx_{k}': v for k, v in scaled_context.items()})
        all_signals.append(sig)

    del df_raw, df_feat, df_scaled, df_pst, XA, XC; _release_memory()
    n_go = sum(1 for s in all_signals if s['ticker'] == ticker and s['is_go'])
    print(f"done ({len(positions)} bars, {n_go} GO)")

_master_cache.clear(); _model_cache.clear(); _release_memory()

# ==============================================================================
# BUILD SIGNAL DATAFRAME
# ==============================================================================
df_sig = pd.DataFrame(all_signals)
df_go = df_sig[df_sig['is_go']].copy()
df_go_am = df_go[df_go['period'] == 'apr_may']
df_go_jun = df_go[df_go['period'] == 'june']

print(f"\n  Total signal bars: {len(df_sig)}")
print(f"  GO signals: {len(df_go)} (Apr-May: {len(df_go_am)}, June: {len(df_go_jun)})")
print(f"  GO WR: Apr-May {(df_go_am['trade_pnl'] > 0).mean():.0%}, "
      f"June {(df_go_jun['trade_pnl'] > 0).mean():.0%}")

# ==============================================================================
# TABLE 1: FEATURE DISTRIBUTION SHIFT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 1: FEATURE DISTRIBUTION SHIFT — GO signals, April-May vs June")
print(f"  KS-test measures distribution divergence (higher = more different)")
print(f"{'='*W}")

raw_asset_cols = [f'raw_{c}' for c in ASSET_COLS]
raw_context_cols = [f'raw_{c}' for c in CONTEXT_COLS]
all_raw_cols = raw_asset_cols + raw_context_cols

shift_results = []
for raw_col in all_raw_cols:
    col_name = raw_col.replace('raw_', '')
    vals_am = df_go_am[raw_col].dropna().values
    vals_jun = df_go_jun[raw_col].dropna().values
    if len(vals_am) < 3 or len(vals_jun) < 3:
        continue
    ks_stat, ks_p = sp_stats.ks_2samp(vals_am, vals_jun)
    med_am = np.median(vals_am)
    med_jun = np.median(vals_jun)
    mean_am = np.mean(vals_am)
    mean_jun = np.mean(vals_jun)
    if abs(med_am) > 1e-6:
        pct_shift = (med_jun - med_am) / abs(med_am) * 100
    else:
        pct_shift = 0
    shift_results.append({
        'feature': col_name, 'ks_stat': ks_stat, 'ks_p': ks_p,
        'med_am': med_am, 'med_jun': med_jun, 'mean_am': mean_am,
        'mean_jun': mean_jun, 'pct_shift': pct_shift,
        'is_context': col_name in CONTEXT_COLS,
    })

shift_df = pd.DataFrame(shift_results).sort_values('ks_stat', ascending=False)

print(f"\n  {'Feature':<22} {'Apr-May Med':>12} {'June Med':>12} {'KS-stat':>8} {'p-value':>8} {'Shift':>10}")
print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*8} {'─'*8} {'─'*10}")

for _, r in shift_df.head(25).iterrows():
    sig_marker = '***' if r['ks_p'] < 0.001 else ('**' if r['ks_p'] < 0.01 else ('*' if r['ks_p'] < 0.05 else ''))
    direction = '↓' if r['pct_shift'] < -10 else ('↑' if r['pct_shift'] > 10 else '→')
    pct_str = f"{direction} {abs(r['pct_shift']):.0f}%" if abs(r['pct_shift']) > 1 else '→ ~0%'
    prefix = '[CTX] ' if r['is_context'] else ''
    print(f"  {prefix}{r['feature']:<22} {r['med_am']:>12.4f} {r['med_jun']:>12.4f} "
          f"{r['ks_stat']:>7.3f} {r['ks_p']:>7.4f}{sig_marker} {pct_str:>10}")


# ==============================================================================
# TABLE 2: PRE-MARKET MICROSTRUCTURE BREAKDOWN
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 2: PRE-MARKET MICROSTRUCTURE — GO signal bars, per ticker")
print(f"{'='*W}")

micro_features = ['raw_Total_Vol', 'raw_Trade_Count', 'raw_Spread', 'raw_OBI',
                   'raw_Volume_Delta', 'raw_Delta_Pct', 'raw_Gap_Pct', 'raw_RVOL']

print(f"\n  {'Ticker':<7} {'Period':<8} {'PM_Vol':>10} {'Trades':>8} {'Spread':>8} "
      f"{'OBI':>7} {'Gap%':>7} {'RVOL':>7} {'VolDelta':>10} {'N':>4}")
print(f"  {'─'*7} {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*10} {'─'*4}")

for ticker in sorted(df_go['ticker'].unique()):
    for period, label in [('apr_may', 'AprMay'), ('june', 'June')]:
        mask = (df_go['ticker'] == ticker) & (df_go['period'] == period)
        subset = df_go[mask]
        if len(subset) == 0:
            continue
        print(f"  {ticker:<7} {label:<8} "
              f"{subset['raw_Total_Vol'].median():>10,.0f} "
              f"{subset['raw_Trade_Count'].median():>8,.0f} "
              f"{subset['raw_Spread'].median():>8.4f} "
              f"{subset['raw_OBI'].median():>+7.3f} "
              f"{subset['raw_Gap_Pct'].median()*100:>+6.2f}% "
              f"{subset['raw_RVOL'].median():>7.2f} "
              f"{subset['raw_Volume_Delta'].median():>10,.0f} "
              f"{len(subset):>4}")


# ==============================================================================
# TABLE 3: INTRADAY PRICE ACTION PROFILES — June losses
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 3: INTRADAY PRICE PROFILES — June GO signals (all)")
print(f"{'='*W}")

def categorize_loss(row):
    if row['trade_pnl'] >= 0:
        return 'WIN'
    mg = row['max_gain']
    md = row['max_drop']
    sl = row['sl_eff']
    tp = row['tp_eff']
    if mg < 0.01:
        return 'NEVER_MOVED'
    if mg >= 0.02 and row['exit_type'] == 'SL':
        return 'GAP_REVERSAL'
    if md < -0.10:
        return 'FLASH_CRASH'
    return 'SLOW_BLEED'

df_go_jun_detail = df_go_jun.copy()
df_go_jun_detail['category'] = df_go_jun_detail.apply(categorize_loss, axis=1)

# Also categorize April-May losses for comparison
df_go_am_losses = df_go_am[df_go_am['trade_pnl'] < 0].copy()
df_go_am_losses['category'] = df_go_am_losses.apply(categorize_loss, axis=1)

print(f"\n  {'Date':<12} {'Ticker':<7} {'Entry$':>7} {'Prob':>5} {'MaxGain':>8} {'When':>6} "
      f"{'MaxDrop':>8} {'When':>6} {'EOD':>8} {'PnL':>8} {'Exit':>4} {'Category'}")
print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*5} {'─'*8} {'─'*6} {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*4} {'─'*13}")

for _, r in df_go_jun_detail.sort_values('date_str').iterrows():
    win_marker = '+' if r['trade_pnl'] > 0 else '-'
    print(f"  {r['date_str']:<12} {r['ticker']:<7} ${r['entry_price']:>6.2f} "
          f"{r['prob']:.2f} {r['max_gain']*100:>+7.2f}% {r['max_gain_time']:>6} "
          f"{r['max_drop']*100:>+7.2f}% {r['max_drop_time']:>6} "
          f"{r['eod_ret']*100:>+7.2f}% {r['trade_pnl']*100:>+7.2f}% {r['exit_type']:>4} {r['category']}")

# Category distribution comparison
print(f"\n  Loss category distribution:")
print(f"  {'Category':<15} {'Apr-May (N)':>12} {'Apr-May %':>10} {'June (N)':>10} {'June %':>10}")
print(f"  {'─'*15} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
for cat in ['NEVER_MOVED', 'GAP_REVERSAL', 'SLOW_BLEED', 'FLASH_CRASH']:
    n_am = len(df_go_am_losses[df_go_am_losses['category'] == cat])
    n_jun = len(df_go_jun_detail[(df_go_jun_detail['category'] != 'WIN') & (df_go_jun_detail['category'] == cat)])
    pct_am = n_am / max(1, len(df_go_am_losses)) * 100
    pct_jun = n_jun / max(1, len(df_go_jun_detail[df_go_jun_detail['category'] != 'WIN'])) * 100
    print(f"  {cat:<15} {n_am:>12} {pct_am:>9.0f}% {n_jun:>10} {pct_jun:>9.0f}%")


# ==============================================================================
# TABLE 4: CROSS-TICKER LOSS CORRELATION
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 4: CROSS-TICKER RETURN CORRELATION")
print(f"  Higher correlation in June = systemic (market-wide), not model-specific")
print(f"{'='*W}")

ret_df = pd.DataFrame(ticker_daily_returns)
ret_df.index = pd.to_datetime(ret_df.index).tz_localize(None)

am_mask = ret_df.index < pd.Timestamp(JUNE_START)
jun_mask = ret_df.index >= pd.Timestamp(JUNE_START)

tickers_with_data = [t for t in deployed_tickers if t in ret_df.columns]

if len(tickers_with_data) >= 3:
    corr_am = ret_df.loc[am_mask, tickers_with_data].corr()
    corr_jun = ret_df.loc[jun_mask, tickers_with_data].corr()

    # Average pairwise correlation
    n = len(tickers_with_data)
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    avg_corr_am = corr_am.values[mask_upper].mean()
    avg_corr_jun = corr_jun.values[mask_upper].mean()

    print(f"\n  Average pairwise correlation:")
    print(f"    April-May: {avg_corr_am:+.3f}")
    print(f"    June:      {avg_corr_jun:+.3f}")
    print(f"    Change:    {avg_corr_jun - avg_corr_am:+.3f}")
    if avg_corr_jun > avg_corr_am + 0.1:
        print(f"    → SYSTEMIC: tickers moving together in June (macro-driven losses)")
    else:
        print(f"    → IDIOSYNCRATIC: ticker-specific model failures")

    # Show June correlation matrix for top loss tickers
    june_losers = df_go_jun.groupby('ticker')['trade_pnl'].mean().sort_values().head(8).index.tolist()
    june_losers_in_data = [t for t in june_losers if t in tickers_with_data]
    if len(june_losers_in_data) >= 3:
        corr_sub = corr_jun.loc[june_losers_in_data, june_losers_in_data]
        print(f"\n  June correlation matrix (worst tickers):")
        header = "  " + " " * 7 + "".join(f"{t:>7}" for t in june_losers_in_data)
        print(header)
        for t1 in june_losers_in_data:
            row_str = f"  {t1:<7}"
            for t2 in june_losers_in_data:
                val = corr_sub.loc[t1, t2]
                row_str += f"{val:>+7.2f}"
            print(row_str)

    # Day-by-day: count how many tickers had negative returns
    print(f"\n  Day-by-day: fraction of tickers with negative returns")
    print(f"  {'Date':<12} {'Down/Total':>12} {'Pct Down':>10} {'Avg Return':>12}")
    print(f"  {'─'*12} {'─'*12} {'─'*10} {'─'*12}")
    for date in sorted(ret_df[jun_mask].index):
        day_rets = ret_df.loc[date].dropna()
        if len(day_rets) == 0: continue
        n_down = (day_rets < 0).sum()
        print(f"  {date.strftime('%Y-%m-%d'):<12} {n_down}/{len(day_rets):>10} "
              f"{n_down/len(day_rets)*100:>9.0f}% {day_rets.mean()*100:>+11.2f}%")


# ==============================================================================
# TABLE 5: MODEL PROBABILITY CALIBRATION DECAY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 5: PROBABILITY CALIBRATION — Does prob predict win rate?")
print(f"{'='*W}")

prob_bins = [(0.45, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.01)]

print(f"\n  {'Prob Bin':<12} {'Apr-May WR':>11} {'(N)':>5} {'June WR':>9} {'(N)':>5} {'Drift':>8}")
print(f"  {'─'*12} {'─'*11} {'─'*5} {'─'*9} {'─'*5} {'─'*8}")

for lo, hi in prob_bins:
    label = f"{lo:.2f}-{hi:.2f}"
    am_bin = df_go_am[(df_go_am['prob'] >= lo) & (df_go_am['prob'] < hi)]
    jun_bin = df_go_jun[(df_go_jun['prob'] >= lo) & (df_go_jun['prob'] < hi)]
    wr_am = (am_bin['trade_pnl'] > 0).mean() if len(am_bin) > 0 else float('nan')
    wr_jun = (jun_bin['trade_pnl'] > 0).mean() if len(jun_bin) > 0 else float('nan')
    drift = wr_jun - wr_am if not (np.isnan(wr_am) or np.isnan(wr_jun)) else float('nan')

    wr_am_str = f"{wr_am*100:.0f}%" if not np.isnan(wr_am) else "—"
    wr_jun_str = f"{wr_jun*100:.0f}%" if not np.isnan(wr_jun) else "—"
    drift_str = f"{drift*100:+.0f}%" if not np.isnan(drift) else "—"
    print(f"  {label:<12} {wr_am_str:>11} {len(am_bin):>5} {wr_jun_str:>9} {len(jun_bin):>5} {drift_str:>8}")

# Also show: probability distribution itself
print(f"\n  Probability distribution of GO signals:")
print(f"  {'Period':<10} {'Mean':>6} {'Median':>8} {'Std':>6} {'Min':>6} {'Max':>6} {'N':>5}")
print(f"  {'─'*10} {'─'*6} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*5}")
for period, label, subset in [('apr_may', 'Apr-May', df_go_am), ('june', 'June', df_go_jun)]:
    print(f"  {label:<10} {subset['prob'].mean():.3f} {subset['prob'].median():>8.3f} "
          f"{subset['prob'].std():>6.3f} {subset['prob'].min():>6.3f} {subset['prob'].max():>6.3f} "
          f"{len(subset):>5}")


# ==============================================================================
# TABLE 6: CONTEXT FEATURE ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 6: CONTEXT FEATURES — What the model sees about SPY/VIX/TLT/IWM")
print(f"  Raw (unscaled) values on GO signal bars")
print(f"{'='*W}")

ctx_raw_cols = [f'raw_{c}' for c in CONTEXT_COLS if c != 'Time_Pos']

print(f"\n  {'Context Feature':<22} {'Apr-May Med':>12} {'June Med':>12} {'Shift':>10} {'p-value':>8}")
print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*10} {'─'*8}")

for raw_col in ctx_raw_cols:
    col_name = raw_col.replace('raw_', '')
    vals_am = df_go_am[raw_col].dropna().values
    vals_jun = df_go_jun[raw_col].dropna().values
    if len(vals_am) < 3 or len(vals_jun) < 3:
        continue
    med_am = np.median(vals_am)
    med_jun = np.median(vals_jun)
    _, ks_p = sp_stats.ks_2samp(vals_am, vals_jun)
    if abs(med_am) > 1e-6:
        pct = (med_jun - med_am) / abs(med_am) * 100
        shift_str = f"{'↑' if pct > 0 else '↓'} {abs(pct):.0f}%"
    else:
        diff = med_jun - med_am
        shift_str = f"{'↑' if diff > 0 else '↓'} {abs(diff):.4f}"
    sig = '***' if ks_p < 0.001 else ('**' if ks_p < 0.01 else ('*' if ks_p < 0.05 else ''))
    print(f"  {col_name:<22} {med_am:>12.5f} {med_jun:>12.5f} {shift_str:>10} {ks_p:>7.4f}{sig}")


# ==============================================================================
# TABLE 7: SPIKE RATE DECAY TIMELINE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 7: SPIKE RATE DECAY — Weekly, all deployed tickers")
print(f"  Spike = max gain from entry bar through 1 PM PST >= threshold")
print(f"{'='*W}")

spike_thresholds = [0.03, 0.05, 0.08, 0.10]

# Use ALL signal bars (not just GO) for spike opportunity analysis
weekly_data = []
df_sig['week_start'] = pd.to_datetime(df_sig['date_str']).dt.to_period('W').dt.start_time

for week_start, week_df in df_sig.groupby('week_start'):
    row = {'week': week_start.strftime('%b %d')}
    row['n_bars'] = len(week_df)
    row['n_go'] = len(week_df[week_df['is_go']])
    go_subset = week_df[week_df['is_go']]
    row['go_wr'] = (go_subset['trade_pnl'] > 0).mean() if len(go_subset) > 0 else 0
    row['go_avg_pnl'] = go_subset['trade_pnl'].mean() if len(go_subset) > 0 else 0
    for thresh in spike_thresholds:
        n_spike = (week_df['max_gain'] >= thresh).sum()
        row[f'spike_{int(thresh*100)}pct'] = n_spike / max(1, len(week_df)) * 100
    weekly_data.append(row)

weekly_df = pd.DataFrame(weekly_data)

header = f"  {'Week':<10} {'Bars':>5} {'GO':>4} {'WR':>5} {'AvgPnL':>8}"
for t in spike_thresholds:
    header += f" {f'≥{int(t*100)}%':>6}"
print(f"\n{header}")
print(f"  {'─'*10} {'─'*5} {'─'*4} {'─'*5} {'─'*8}" + " ".join(['─'*6]*len(spike_thresholds)))

for _, r in weekly_df.iterrows():
    line = f"  {r['week']:<10} {r['n_bars']:>5} {r['n_go']:>4} {r['go_wr']*100:>4.0f}% {r['go_avg_pnl']*100:>+7.2f}%"
    for t in spike_thresholds:
        val = r[f'spike_{int(t*100)}pct']
        line += f" {val:>5.1f}%"
    print(line)

# Per-ticker spike frequency at model target (≥8%)
print(f"\n  Per-ticker spike frequency (≥8% threshold — model target):")
print(f"  {'Ticker':<7} {'Apr-May ≥8%':>12} {'(N)':>5} {'June ≥8%':>10} {'(N)':>5} {'Change':>10}")
print(f"  {'─'*7} {'─'*12} {'─'*5} {'─'*10} {'─'*5} {'─'*10}")

for ticker in sorted(df_sig['ticker'].unique()):
    am_t = df_sig[(df_sig['ticker'] == ticker) & (df_sig['period'] == 'apr_may')]
    jun_t = df_sig[(df_sig['ticker'] == ticker) & (df_sig['period'] == 'june')]
    if len(am_t) == 0: continue
    rate_am = (am_t['max_gain'] >= 0.08).mean() * 100
    rate_jun = (jun_t['max_gain'] >= 0.08).mean() * 100 if len(jun_t) > 0 else 0
    change = rate_jun - rate_am
    print(f"  {ticker:<7} {rate_am:>11.1f}% {len(am_t):>5} {rate_jun:>9.1f}% {len(jun_t):>5} {change:>+9.1f}%")


# ==============================================================================
# SUMMARY: ROOT CAUSE SYNTHESIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ROOT CAUSE SYNTHESIS")
print(f"{'='*W}")

# Compute key summary metrics
am_wr = (df_go_am['trade_pnl'] > 0).mean()
jun_wr = (df_go_jun['trade_pnl'] > 0).mean()
am_avg_pnl = df_go_am['trade_pnl'].mean()
jun_avg_pnl = df_go_jun['trade_pnl'].mean()
avg_prob_am = df_go_am['prob'].mean()
avg_prob_jun = df_go_jun['prob'].mean()

# Spike rates at model target thresholds (8%, 10%)
am_sigs = df_sig[df_sig['period']=='apr_may']
jun_sigs = df_sig[df_sig['period']=='june']
am_spike8 = (am_sigs['max_gain'] >= 0.08).mean()
jun_spike8 = (jun_sigs['max_gain'] >= 0.08).mean()
am_spike10 = (am_sigs['max_gain'] >= 0.10).mean()
jun_spike10 = (jun_sigs['max_gain'] >= 0.10).mean()

# Per-day breakdown of June
jun_daily = df_sig[df_sig['period']=='june'].groupby('date_str').agg(
    n_bars=('ticker', 'count'),
    n_go=('is_go', 'sum'),
    spike8=('max_gain', lambda x: (x >= 0.08).mean()),
    spike10=('max_gain', lambda x: (x >= 0.10).mean()),
    avg_drop=('max_drop', 'mean'),
    pct_down=('max_drop', lambda x: (x < -0.02).mean()),
).sort_index()

# Identify catastrophic vs normal June days
crash_dates = jun_daily[jun_daily['pct_down'] >= 0.80].index.tolist()
normal_dates = jun_daily[jun_daily['pct_down'] < 0.80].index.tolist()
jun_go_crash = df_go_jun[df_go_jun['date_str'].isin(crash_dates)]
jun_go_normal = df_go_jun[df_go_jun['date_str'].isin(normal_dates)]

# Top shifted features
top_shifts = shift_df.head(5)['feature'].tolist()

print(f"""
  1. PERFORMANCE COLLAPSE
     Apr-May: WR={am_wr:.0%}, avg PnL={am_avg_pnl*100:+.2f}% ({len(df_go_am)} trades)
     June:    WR={jun_wr:.0%}, avg PnL={jun_avg_pnl*100:+.2f}% ({len(df_go_jun)} trades)

  2. SPIKE OPPORTUNITY AT MODEL TARGET (≥8% / ≥10%):
     Apr-May: {am_spike8:.0%} / {am_spike10:.0%} of signal bars had ≥8% / ≥10% spikes
     June:    {jun_spike8:.0%} / {jun_spike10:.0%} of signal bars had ≥8% / ≥10% spikes

  3. JUNE DAY-BY-DAY:""")

for _, r in jun_daily.iterrows():
    day_type = 'CRASH' if r.name in crash_dates else 'ok'
    print(f"     {r.name}: {int(r['n_bars'])} bars, {int(r['n_go'])} GO, "
          f"≥8% spike: {r['spike8']:.0%}, avg drop: {r['avg_drop']*100:+.1f}%, "
          f"{r['pct_down']:.0%} down  [{day_type}]")

crash_wr = (jun_go_crash['trade_pnl'] > 0).mean() if len(jun_go_crash) > 0 else 0
normal_wr = (jun_go_normal['trade_pnl'] > 0).mean() if len(jun_go_normal) > 0 else 0

print(f"""
     Crash days ({', '.join(crash_dates)}): {len(jun_go_crash)} GO, WR={crash_wr:.0%}
     Normal days ({', '.join(normal_dates)}): {len(jun_go_normal)} GO, WR={normal_wr:.0%}

  4. MODEL DIDN'T ADAPT
     Apr-May avg GO probability: {avg_prob_am:.3f}
     June avg GO probability:    {avg_prob_jun:.3f}
     → Model confidence barely changed despite regime shift

  5. TOP SHIFTED FEATURES (most different in June vs Apr-May):
     {', '.join(top_shifts)}

  6. CROSS-TICKER PATTERN:
     {'SYSTEMIC — losses correlated across tickers (market-wide)'
      if len(tickers_with_data) >= 3 and avg_corr_jun > avg_corr_am + 0.1
      else 'MIXED — some correlation increase'}
""")


# ==============================================================================
# CHARTS
# ==============================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('V9 June Failure — Deep Analysis', fontsize=14, fontweight='bold')

    # 1. Feature shift bar chart (top 10)
    ax = axes[0, 0]
    top10 = shift_df.head(10)
    colors = ['#d32f2f' if p < 0.05 else '#999' for p in top10['ks_p']]
    ax.barh(range(len(top10)), top10['ks_stat'].values, color=colors)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10['feature'].values, fontsize=8)
    ax.set_xlabel('KS Statistic')
    ax.set_title('Feature Distribution Shift\n(red = p<0.05)')
    ax.invert_yaxis()

    # 2. Calibration curve
    ax = axes[0, 1]
    for period, label, subset, color in [
        ('apr_may', 'Apr-May', df_go_am, '#1976d2'),
        ('june', 'June', df_go_jun, '#d32f2f'),
    ]:
        prob_centers = []
        actual_wrs = []
        for lo, hi in prob_bins:
            bin_data = subset[(subset['prob'] >= lo) & (subset['prob'] < hi)]
            if len(bin_data) >= 2:
                prob_centers.append((lo + hi) / 2)
                actual_wrs.append((bin_data['trade_pnl'] > 0).mean())
        if prob_centers:
            ax.plot(prob_centers, actual_wrs, 'o-', label=label, color=color, linewidth=2)
    ax.plot([0.4, 1.0], [0.4, 1.0], '--', color='gray', alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Model Probability')
    ax.set_ylabel('Actual Win Rate')
    ax.set_title('Probability Calibration')
    ax.legend(fontsize=9)
    ax.set_xlim(0.4, 0.85)
    ax.set_ylim(-0.05, 1.05)

    # 3. Spike rate over time
    ax = axes[0, 2]
    if len(weekly_df) > 0:
        x = range(len(weekly_df))
        for t in spike_thresholds:
            col = f'spike_{int(t*100)}pct'
            ax.plot(x, weekly_df[col].values, 'o-', label=f'≥{int(t*100)}%', markersize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(weekly_df['week'].values, rotation=45, fontsize=7)
        ax.set_ylabel('Spike Rate (%)')
        ax.set_title('Spike Opportunity by Week')
        ax.legend(fontsize=8)
        ax.axvline(x=len(weekly_df)-2, color='red', linestyle='--', alpha=0.5, label='June starts')

    # 4. Weekly WR and GO signals
    ax = axes[1, 0]
    if len(weekly_df) > 0:
        x = range(len(weekly_df))
        ax.bar(x, weekly_df['go_wr'].values * 100, color=['#1976d2']*max(0,len(weekly_df)-2) + ['#d32f2f']*min(2,len(weekly_df)), alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(weekly_df['week'].values, rotation=45, fontsize=7)
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('GO Signal Win Rate by Week')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    # 5. Loss category comparison
    ax = axes[1, 1]
    categories = ['NEVER_MOVED', 'GAP_REVERSAL', 'SLOW_BLEED', 'FLASH_CRASH']
    am_counts = [len(df_go_am_losses[df_go_am_losses['category'] == c]) for c in categories]
    jun_losses = df_go_jun_detail[df_go_jun_detail['category'] != 'WIN']
    jun_counts = [len(jun_losses[jun_losses['category'] == c]) for c in categories]
    x_pos = np.arange(len(categories))
    bar_width = 0.35
    am_pcts = [c/max(1,sum(am_counts))*100 for c in am_counts]
    jun_pcts = [c/max(1,sum(jun_counts))*100 for c in jun_counts]
    ax.bar(x_pos - bar_width/2, am_pcts, bar_width, label='Apr-May', color='#1976d2', alpha=0.7)
    ax.bar(x_pos + bar_width/2, jun_pcts, bar_width, label='June', color='#d32f2f', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.replace('_', '\n') for c in categories], fontsize=8)
    ax.set_ylabel('% of Losses')
    ax.set_title('Loss Category Distribution')
    ax.legend(fontsize=9)

    # 6. Cross-ticker correlation heatmap
    ax = axes[1, 2]
    if len(tickers_with_data) >= 3:
        top_tickers = df_go_jun.groupby('ticker')['trade_pnl'].sum().sort_values().head(8).index.tolist()
        top_in_data = [t for t in top_tickers if t in tickers_with_data][:6]
        if len(top_in_data) >= 3:
            corr_sub = ret_df.loc[jun_mask, top_in_data].corr()
            im = ax.imshow(corr_sub.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
            ax.set_xticks(range(len(top_in_data)))
            ax.set_xticklabels(top_in_data, fontsize=8, rotation=45)
            ax.set_yticks(range(len(top_in_data)))
            ax.set_yticklabels(top_in_data, fontsize=8)
            for i in range(len(top_in_data)):
                for j in range(len(top_in_data)):
                    ax.text(j, i, f'{corr_sub.values[i,j]:.2f}', ha='center', va='center', fontsize=7)
            ax.set_title('June Return Correlation\n(worst tickers)')
            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    chart_path = '/tmp/june_deep_analysis.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Charts saved: {chart_path}")
    plt.close()

except ImportError:
    print("\n  (matplotlib not available — skipping charts)")

print(f"\n{'='*W}")
print(f"  DEEP ANALYSIS COMPLETE")
print(f"{'='*W}")
