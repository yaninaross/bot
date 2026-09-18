"""
backtest_signal_filters.py — Signal Quality Filter Evaluation for V9
====================================================================
Investigates WHY June GO signals failed by capturing raw microstructure,
volume, and volatility at each entry bar. Tests filter combinations that
would have blocked bad signals.

Three gaps identified in current pipeline:
  1. vol_filter_pct = 0 for 27/28 deployed tickers (filter disabled)
  2. vol_filter checks OHLCV Volume, never microstructure (Trade_Count, etc)
  3. Bar_Is_Real is a feature, not a hard gate — model can fire on V=0 bars

This script captures raw (unscaled) microstructure at every GO signal bar,
then backtests filter combinations to find which would have blocked June
losses without killing Apr-May wins.

Colab-ready (no if __name__). Requires: TF, pandas_ta, numba, GCS, BQ.
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

STARTING_CAPITAL = 200_000

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
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
               'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
               'grace': '_GRACE', 'limit_entry': '_LIMIT'}

W = 120

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()

# ==============================================================================
# CORE FUNCTIONS (from analyze_june_deep.py)
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

def build_market_context(start_date=None):
    global _market_context_cache
    if _market_context_cache is not None: return _market_context_cache
    print("  Building market context...")
    trim_from = pd.Timestamp(start_date, tz='UTC') - pd.Timedelta(days=7) if start_date else None
    frames = []
    for ticker in CONTEXT_ASSETS:
        clean = TICKER_FILE_MAP.get(ticker, ticker)
        uri = f"gs://{BUCKET_NAME}/{OHLCV_DIR}/{clean}_master_inference.parquet"
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


# ==============================================================================
# COMPOSITE SCORE (S8_mix)
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
print(f"  SIGNAL QUALITY FILTER EVALUATION — V9 Portfolio")
print(f"  Period: {TEST_START} → {TEST_END}  |  Capital: ${STARTING_CAPITAL:,.0f}")
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
        's8_mix': float(r['S8_mix']),
        'vol_filter_pct': int(r.get('vol_filter_pct', 0) or 0),
    }

deployed_tickers = sorted(ticker_configs.keys())
print(f"  Deployed: {deployed_tickers}")

vf_status = [(t, ticker_configs[t]['vol_filter_pct']) for t in deployed_tickers]
n_active = sum(1 for _, v in vf_status if v > 0)
print(f"\n  Current vol_filter_pct settings: {n_active}/{len(vf_status)} tickers have filter active")
for t, v in vf_status:
    if v > 0:
        print(f"    {t}: vol_filter_pct = {v}")


# ==============================================================================
# STEP 2: LOAD DATA + RUN MODEL INFERENCE + CAPTURE MICROSTRUCTURE
# ==============================================================================
print(f"\n  Loading OHLCV + running model inference + capturing microstructure...")
market_df = build_market_context(start_date=TEST_START)

test_start_utc = pd.Timestamp(TEST_START, tz='UTC')
test_end_utc = pd.Timestamp(TEST_END, tz='UTC') + timedelta(days=1)
june_start_utc = pd.Timestamp(JUNE_START, tz='UTC')

all_signals = []
ticker_vol_baselines = {}

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

    df_pst = df_raw.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    test_s_pst = pd.Timestamp(TEST_START, tz=PST)
    test_e_pst = pd.Timestamp(TEST_END, tz=PST) + timedelta(days=1)

    # Compute volume baseline for this ticker at entry_minute (30-day median)
    em = cfg['entry_minute']
    mins_pst = df_pst.index.hour * 60 + df_pst.index.minute
    at_entry = df_pst[mins_pst == em]
    past_real = at_entry[(at_entry.index < test_s_pst) & (at_entry['Volume'] > 0)].tail(30)
    vol_median = float(past_real['Volume'].median()) if len(past_real) >= 5 else 0.0
    tc_median = float(past_real['Trade_Count'].median()) if len(past_real) >= 5 and 'Trade_Count' in past_real.columns else 0.0
    tv_median = float(past_real['Total_Vol'].median()) if len(past_real) >= 5 and 'Total_Vol' in past_real.columns else 0.0
    ticker_vol_baselines[ticker] = {
        'vol_median': vol_median, 'tc_median': tc_median, 'tv_median': tv_median
    }

    # Engineer features (keep raw df_feat for microstructure capture)
    df_feat = engineer_features(df_raw.copy(), market_df)
    stats = load_scaling_stats(cfg['base_exp'])
    df_scaled = apply_scaling(df_feat, stats)

    asset_id = get_asset_index(cfg['base_exp'], ticker)
    df_test = df_scaled[(df_scaled.index >= test_start_utc) & (df_scaled.index < test_end_utc)]
    if len(df_test) == 0:
        del df_raw, df_feat, df_scaled, df_pst; _release_memory()
        print("SKIP"); continue

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

    # Collect GO signals with RAW microstructure at entry bar
    ohlcv_test_trade = df_pst[(df_pst.index >= test_s_pst) & (df_pst.index < test_e_pst)]
    for i, idx in enumerate(positions):
        ts = df_scaled.index[idx]
        ts_pst = ts.tz_convert(PST)
        prob = float(probs[i])

        if prob < cfg['prob_thresh']:
            continue

        entry_price = float(df_pst['Close'].iloc[df_pst.index.get_indexer([ts_pst], method='nearest')[0]])
        day_mask = (ohlcv_test_trade.index.date == ts_pst.date()) & (ohlcv_test_trade.index >= ts_pst)
        day_bars = ohlcv_test_trade[day_mask]

        max_gain = 0.0; max_drop = 0.0; eod_ret = 0.0
        if len(day_bars) > 0 and entry_price > 0:
            highs_ret = (day_bars['High'].values / entry_price - 1)
            lows_ret = (day_bars['Low'].values / entry_price - 1)
            max_gain = float(np.max(highs_ret))
            max_drop = float(np.min(lows_ret))
            eod_price = float(day_bars['Close'].iloc[-1])
            eod_ret = (eod_price / entry_price - 1)

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

        # --- RAW microstructure at entry bar (pre-scaling) ---
        raw_vol = float(df_feat['Volume'].iloc[idx]) if 'Volume' in df_feat.columns else 0.0
        raw_tc = float(df_feat['Trade_Count'].iloc[idx]) if 'Trade_Count' in df_feat.columns else 0.0
        raw_tv = float(df_feat['Total_Vol'].iloc[idx]) if 'Total_Vol' in df_feat.columns else 0.0
        raw_spread = float(df_feat['Spread'].iloc[idx]) if 'Spread' in df_feat.columns else 0.0
        raw_obi = float(df_feat['OBI'].iloc[idx]) if 'OBI' in df_feat.columns else 0.0
        raw_atr14 = float(df_feat['ATR_14'].iloc[idx]) if 'ATR_14' in df_feat.columns else 0.0
        raw_rvol = float(df_feat['RVOL'].iloc[idx]) if 'RVOL' in df_feat.columns else 0.0
        raw_bar_is_real = float(df_feat['Bar_Is_Real'].iloc[idx]) if 'Bar_Is_Real' in df_feat.columns else 0.0
        raw_gap_pct = float(df_feat['Gap_Pct'].iloc[idx]) if 'Gap_Pct' in df_feat.columns else 0.0
        raw_compression = float(df_feat['Compression'].iloc[idx]) if 'Compression' in df_feat.columns else 0.0

        dollar_vol = raw_vol * entry_price if entry_price > 0 else 0.0

        # Volume as % of baseline
        vol_pct_baseline = (raw_vol / vol_median * 100) if vol_median > 0 else 0.0

        # Lookback microstructure: avg of last 5 bars at entry_minute (rolling baseline)
        at_entry_before = at_entry[(at_entry.index < ts_pst) & (at_entry['Volume'] > 0)].tail(5)
        rolling_tc_avg = float(at_entry_before['Trade_Count'].mean()) if len(at_entry_before) > 0 and 'Trade_Count' in at_entry_before.columns else 0.0
        rolling_tv_avg = float(at_entry_before['Total_Vol'].mean()) if len(at_entry_before) > 0 and 'Total_Vol' in at_entry_before.columns else 0.0

        # VIX at entry
        vix_col = 'VIX_Anchor_ROC'
        vix_val = float(df_feat[vix_col].iloc[idx]) if vix_col in df_feat.columns else 0.0

        # IWM at entry
        iwm_col = 'IWM_ROC_30m'
        iwm_val = float(df_feat[iwm_col].iloc[idx]) if iwm_col in df_feat.columns else 0.0

        period = 'june' if ts >= june_start_utc else 'apr_may'
        date_str = ts_pst.strftime('%Y-%m-%d')

        all_signals.append({
            'date_str': date_str,
            'ticker': ticker,
            'prob': prob,
            'trade_pnl': trade_pnl,
            'exit_type': exit_type,
            'tp_eff': tp_eff,
            'sl_eff': sl_eff,
            'entry_price': entry_price,
            's8_mix': cfg['s8_mix'],
            'period': period,
            'max_gain': max_gain,
            'max_drop': max_drop,
            # Raw microstructure
            'raw_volume': raw_vol,
            'raw_trade_count': raw_tc,
            'raw_total_vol': raw_tv,
            'raw_spread': raw_spread,
            'raw_obi': raw_obi,
            'raw_atr14': raw_atr14,
            'raw_rvol': raw_rvol,
            'raw_bar_is_real': raw_bar_is_real,
            'raw_gap_pct': raw_gap_pct,
            'raw_compression': raw_compression,
            'dollar_vol': dollar_vol,
            'vol_pct_baseline': vol_pct_baseline,
            'rolling_tc_avg': rolling_tc_avg,
            'rolling_tv_avg': rolling_tv_avg,
            'vix_anchor_roc': vix_val,
            'iwm_roc_30m': iwm_val,
        })

    del df_raw, df_feat, df_scaled, df_pst, XA, XC; _release_memory()
    n_go = sum(1 for s in all_signals if s['ticker'] == ticker)
    print(f"done ({len(positions)} bars, {n_go} GO)")

_master_cache.clear(); _model_cache.clear(); _release_memory()

df_go = pd.DataFrame(all_signals)
print(f"\n  Total GO signals: {len(df_go)}")
print(f"  Apr-May: {len(df_go[df_go['period']=='apr_may'])}, June: {len(df_go[df_go['period']=='june'])}")


# ==============================================================================
# TABLE 1: RAW MICROSTRUCTURE AT EVERY GO SIGNAL
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 1: RAW MICROSTRUCTURE AT ENTRY BAR — All GO Signals")
print(f"{'='*W}")

print(f"\n  {'Date':<12} {'Ticker':<7} {'Prob':>5} {'PnL':>7} {'Exit':>4} "
      f"{'Volume':>10} {'Trade#':>7} {'TotalV':>10} {'Spread':>8} "
      f"{'ATR14':>7} {'RVOL':>6} {'$Vol':>10} {'V%Base':>7} {'BIR':>4}")
print(f"  {'─'*12} {'─'*7} {'─'*5} {'─'*7} {'─'*4} "
      f"{'─'*10} {'─'*7} {'─'*10} {'─'*8} "
      f"{'─'*7} {'─'*6} {'─'*10} {'─'*7} {'─'*4}")

for _, sig in df_go.sort_values(['date_str', 'ticker']).iterrows():
    marker = ' ' if sig['period'] == 'apr_may' else '*'
    pnl_str = f"{sig['trade_pnl']*100:>+6.2f}%"
    bir_str = f"{sig['raw_bar_is_real']:.0f}"
    print(f"{marker} {sig['date_str']:<12} {sig['ticker']:<7} {sig['prob']:.3f} {pnl_str} {sig['exit_type']:>4} "
          f"{sig['raw_volume']:>10,.0f} {sig['raw_trade_count']:>7,.0f} {sig['raw_total_vol']:>10,.0f} "
          f"{sig['raw_spread']:>8.4f} "
          f"{sig['raw_atr14']:.4f} {sig['raw_rvol']:>5.1f} ${sig['dollar_vol']:>9,.0f} "
          f"{sig['vol_pct_baseline']:>6.0f}% {bir_str:>4}")


# ==============================================================================
# TABLE 2: MICROSTRUCTURE SUMMARY — APR-MAY vs JUNE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 2: MICROSTRUCTURE COMPARISON — Apr-May Winners vs June Losers")
print(f"{'='*W}")

apr_may = df_go[df_go['period'] == 'apr_may']
june = df_go[df_go['period'] == 'june']
apr_may_wins = apr_may[apr_may['trade_pnl'] > 0]
apr_may_loss = apr_may[apr_may['trade_pnl'] <= 0]
june_wins = june[june['trade_pnl'] > 0]
june_loss = june[june['trade_pnl'] <= 0]

cols_to_compare = [
    ('raw_volume', 'Volume', '.0f'),
    ('raw_trade_count', 'Trade_Count', '.0f'),
    ('raw_total_vol', 'Total_Vol', '.0f'),
    ('raw_spread', 'Spread', '.4f'),
    ('raw_atr14', 'ATR_14', '.4f'),
    ('raw_rvol', 'RVOL', '.2f'),
    ('dollar_vol', 'Dollar_Vol', ',.0f'),
    ('vol_pct_baseline', 'Vol%Base', '.0f'),
    ('raw_bar_is_real', 'Bar_Is_Real', '.2f'),
    ('raw_gap_pct', 'Gap_Pct', '.4f'),
    ('vix_anchor_roc', 'VIX_ROC', '.4f'),
    ('iwm_roc_30m', 'IWM_30m', '.4f'),
]

groups = [
    ('Apr-May WIN', apr_may_wins),
    ('Apr-May LOSS', apr_may_loss),
    ('June WIN', june_wins),
    ('June LOSS', june_loss),
]

print(f"\n  {'Feature':<14}", end='')
for label, _ in groups:
    print(f" {label:>15}", end='')
print()
print(f"  {'─'*14}", end='')
for _ in groups:
    print(f" {'─'*15}", end='')
print()

for col, label, fmt in cols_to_compare:
    print(f"  {label:<14}", end='')
    for _, subset in groups:
        if len(subset) > 0 and col in subset.columns:
            med = subset[col].median()
            print(f" {med:>15{fmt}}", end='')
        else:
            print(f" {'—':>15}", end='')
    print()

# Count zeros
print(f"\n  Zero-count analysis:")
zero_cols = ['raw_volume', 'raw_trade_count', 'raw_total_vol', 'raw_spread']
for col in zero_cols:
    label = col.replace('raw_', '')
    print(f"    {label:<14}", end='')
    for glabel, subset in groups:
        if len(subset) > 0 and col in subset.columns:
            n_zero = (subset[col] == 0).sum()
            pct = n_zero / len(subset) * 100
            print(f"  {n_zero:>3}/{len(subset):>3} ({pct:>4.0f}%)", end='')
        else:
            print(f"  {'—':>14}", end='')
    print()


# ==============================================================================
# TABLE 3: FILTER IMPACT MATRIX
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 3: FILTER IMPACT MATRIX — What each filter blocks/keeps")
print(f"{'='*W}")

filters = [
    ('Baseline (no filter)',
     lambda s: True),
    ('Trade_Count >= 1',
     lambda s: s['raw_trade_count'] >= 1),
    ('Trade_Count >= 3',
     lambda s: s['raw_trade_count'] >= 3),
    ('Trade_Count >= 5',
     lambda s: s['raw_trade_count'] >= 5),
    ('Total_Vol >= 50',
     lambda s: s['raw_total_vol'] >= 50),
    ('Total_Vol >= 100',
     lambda s: s['raw_total_vol'] >= 100),
    ('Total_Vol >= 500',
     lambda s: s['raw_total_vol'] >= 500),
    ('Spread > 0',
     lambda s: s['raw_spread'] > 0),
    ('Dollar_Vol >= $50K',
     lambda s: s['dollar_vol'] >= 50_000),
    ('Dollar_Vol >= $25K',
     lambda s: s['dollar_vol'] >= 25_000),
    ('Vol%Baseline >= 25%',
     lambda s: s['vol_pct_baseline'] >= 25),
    ('Vol%Baseline >= 50%',
     lambda s: s['vol_pct_baseline'] >= 50),
    ('RVOL >= 0.5',
     lambda s: s['raw_rvol'] >= 0.5),
    ('ATR_14 <= 0.06',
     lambda s: s['raw_atr14'] <= 0.06),
    ('ATR_14 <= 0.08',
     lambda s: s['raw_atr14'] <= 0.08),
    ('ATR_14 <= 0.10',
     lambda s: s['raw_atr14'] <= 0.10),
    ('Bar_Is_Real == 1',
     lambda s: s['raw_bar_is_real'] == 1),
    ('Micro alive (TC>0 OR TV>0)',
     lambda s: s['raw_trade_count'] > 0 or s['raw_total_vol'] > 0),
    ('Micro alive (TC>0 AND Sp>0)',
     lambda s: s['raw_trade_count'] > 0 and s['raw_spread'] > 0),
    ('VIX_ROC < 0.03',
     lambda s: s['vix_anchor_roc'] < 0.03),
    ('IWM_30m > -0.005',
     lambda s: s['iwm_roc_30m'] > -0.005),
]

print(f"\n  {'Filter':<30} {'AM Keep':>8} {'AM WR':>6} {'AM AvgP':>8} "
      f"{'Jun Keep':>9} {'Jun WR':>7} {'Jun AvgP':>9} "
      f"{'Jun$Saved':>10} {'Net':>6}")
print(f"  {'─'*30} {'─'*8} {'─'*6} {'─'*8} {'─'*9} {'─'*7} {'─'*9} {'─'*10} {'─'*6}")

baseline_jun_pnl = june['trade_pnl'].sum() if len(june) > 0 else 0

for fname, ffunc in filters:
    am_keep = apr_may[apr_may.apply(ffunc, axis=1)] if len(apr_may) > 0 else pd.DataFrame()
    jun_keep = june[june.apply(ffunc, axis=1)] if len(june) > 0 else pd.DataFrame()

    am_n = len(am_keep)
    am_wr = (am_keep['trade_pnl'] > 0).mean() * 100 if am_n > 0 else 0
    am_avg = am_keep['trade_pnl'].mean() * 100 if am_n > 0 else 0

    jun_n = len(jun_keep)
    jun_wr = (jun_keep['trade_pnl'] > 0).mean() * 100 if jun_n > 0 else 0
    jun_avg = jun_keep['trade_pnl'].mean() * 100 if jun_n > 0 else 0

    jun_pnl_kept = jun_keep['trade_pnl'].sum() if jun_n > 0 else 0
    jun_saved = (jun_pnl_kept - baseline_jun_pnl) * 100

    am_blocked = len(apr_may) - am_n
    jun_blocked = len(june) - jun_n
    net = f"+{jun_blocked - am_blocked}" if jun_blocked > am_blocked else f"{jun_blocked - am_blocked}"

    print(f"  {fname:<30} {am_n:>3}/{len(apr_may):<4} {am_wr:>5.0f}% {am_avg:>+7.2f}% "
          f"{jun_n:>4}/{len(june):<4} {jun_wr:>5.0f}% {jun_avg:>+8.2f}% "
          f"{jun_saved:>+9.1f}% {net:>6}")


# ==============================================================================
# TABLE 4: COMBINED FILTER STRATEGIES
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 4: COMBINED FILTER STRATEGIES — Multi-criteria gates")
print(f"{'='*W}")

combo_filters = [
    ('Baseline',
     lambda s: True),

    # Microstructure gates
    ('A: TC>=1 OR TV>=50',
     lambda s: s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50),
    ('B: TC>=1 AND Spread>0',
     lambda s: s['raw_trade_count'] >= 1 and s['raw_spread'] > 0),
    ('C: TC>=3 OR TV>=100',
     lambda s: s['raw_trade_count'] >= 3 or s['raw_total_vol'] >= 100),

    # Micro + volume
    ('D: (TC>=1 OR TV>=50) + $Vol>=25K',
     lambda s: (s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50) and s['dollar_vol'] >= 25_000),
    ('E: (TC>=1 OR TV>=50) + Vol%>=25',
     lambda s: (s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50) and s['vol_pct_baseline'] >= 25),

    # Micro + ATR cap
    ('F: (TC>=1 OR TV>=50) + ATR<=0.08',
     lambda s: (s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50) and s['raw_atr14'] <= 0.08),
    ('G: (TC>=1 OR TV>=50) + ATR<=0.10',
     lambda s: (s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50) and s['raw_atr14'] <= 0.10),

    # Micro + context
    ('H: (TC>=1 OR TV>=50) + VIX<0.03',
     lambda s: (s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50) and s['vix_anchor_roc'] < 0.03),
    ('I: (TC>=1 OR TV>=50) + IWM>-0.005',
     lambda s: (s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50) and s['iwm_roc_30m'] > -0.005),

    # Full combinations
    ('J: TC>=1 + $Vol>=25K + ATR<=0.10',
     lambda s: s['raw_trade_count'] >= 1 and s['dollar_vol'] >= 25_000 and s['raw_atr14'] <= 0.10),
    ('K: (TC>=1 OR TV>=50) + ATR<=0.10 + VIX<0.03',
     lambda s: (s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50) and s['raw_atr14'] <= 0.10 and s['vix_anchor_roc'] < 0.03),
    ('L: Micro + $Vol + ATR + VIX',
     lambda s: (s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50) and s['dollar_vol'] >= 25_000 and s['raw_atr14'] <= 0.10 and s['vix_anchor_roc'] < 0.03),
]

print(f"\n  {'Filter':<38} {'AM Keep':>8} {'AM WR':>6} {'AM Avg':>8} "
      f"{'Jun Keep':>9} {'Jun WR':>7} {'Jun Avg':>9} "
      f"{'Jun$Impact':>11}")
print(f"  {'─'*38} {'─'*8} {'─'*6} {'─'*8} {'─'*9} {'─'*7} {'─'*9} {'─'*11}")

for fname, ffunc in combo_filters:
    am_keep = apr_may[apr_may.apply(ffunc, axis=1)] if len(apr_may) > 0 else pd.DataFrame()
    jun_keep = june[june.apply(ffunc, axis=1)] if len(june) > 0 else pd.DataFrame()

    am_n = len(am_keep)
    am_wr = (am_keep['trade_pnl'] > 0).mean() * 100 if am_n > 0 else 0
    am_avg = am_keep['trade_pnl'].mean() * 100 if am_n > 0 else 0

    jun_n = len(jun_keep)
    jun_wr = (jun_keep['trade_pnl'] > 0).mean() * 100 if jun_n > 0 else 0
    jun_avg = jun_keep['trade_pnl'].mean() * 100 if jun_n > 0 else 0

    jun_pnl_kept = jun_keep['trade_pnl'].sum() * 100 if jun_n > 0 else 0
    baseline_j = baseline_jun_pnl * 100
    jun_saved = jun_pnl_kept - baseline_j

    print(f"  {fname:<38} {am_n:>3}/{len(apr_may):<4} {am_wr:>5.0f}% {am_avg:>+7.2f}% "
          f"{jun_n:>4}/{len(june):<4} {jun_wr:>5.0f}% {jun_avg:>+8.2f}% "
          f"{jun_saved:>+10.1f}%")


# ==============================================================================
# TABLE 5: PORTFOLIO BACKTEST WITH FILTERS (equity curves)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 5: PORTFOLIO BACKTEST — Equity Curves with Top Filters")
print(f"{'='*W}")

def run_filtered_portfolio(df_go, filter_fn, starting_capital):
    filtered = df_go[df_go.apply(filter_fn, axis=1)].copy()
    daily_groups = filtered.groupby('date_str')
    equity = starting_capital
    curve = []
    total_trades = 0

    for date in sorted(df_go['date_str'].unique()):
        if date not in daily_groups.groups:
            curve.append({'date': date, 'equity': equity, 'day_pnl': 0, 'n_taken': 0})
            continue
        group = daily_groups.get_group(date)
        n = len(group)
        if n == 0:
            curve.append({'date': date, 'equity': equity, 'day_pnl': 0, 'n_taken': 0})
            continue
        weight = 1.0 / n
        day_pnl = sum(equity * weight * row['trade_pnl'] for _, row in group.iterrows())
        equity += day_pnl
        total_trades += n
        curve.append({'date': date, 'equity': equity, 'day_pnl': day_pnl, 'n_taken': n})

    curve_df = pd.DataFrame(curve)
    peak = curve_df['equity'].cummax()
    dd = ((curve_df['equity'] - peak) / peak).min() if len(curve_df) > 0 else 0
    active = curve_df[curve_df['n_taken'] > 0]
    if len(active) > 1:
        daily_rets = active['day_pnl'] / (active['equity'] - active['day_pnl'])
        sharpe = float(daily_rets.mean() / max(daily_rets.std(), 1e-8) * np.sqrt(252))
    else:
        sharpe = 0

    return curve_df, equity, total_trades, dd, sharpe


portfolio_filters = [
    ('Baseline (no filter)',      lambda s: True),
    ('A: TC>=1 OR TV>=50',        lambda s: s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50),
    ('C: TC>=3 OR TV>=100',       lambda s: s['raw_trade_count'] >= 3 or s['raw_total_vol'] >= 100),
    ('D: Micro + $Vol>=25K',      lambda s: (s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50) and s['dollar_vol'] >= 25_000),
    ('G: Micro + ATR<=0.10',      lambda s: (s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50) and s['raw_atr14'] <= 0.10),
    ('K: Micro+ATR+VIX',          lambda s: (s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50) and s['raw_atr14'] <= 0.10 and s['vix_anchor_roc'] < 0.03),
    ('Dollar_Vol >= $50K only',   lambda s: s['dollar_vol'] >= 50_000),
    ('Vol%Baseline >= 25% only',  lambda s: s['vol_pct_baseline'] >= 25),
]

print(f"\n  {'Filter':<30} {'Trades':>7} {'Final$':>10} {'Return':>8} {'MaxDD':>7} {'Sharpe':>7} {'JunTr':>6} {'Jun$PnL':>10}")
print(f"  {'─'*30} {'─'*7} {'─'*10} {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*10}")

portfolio_results = {}
for fname, ffunc in portfolio_filters:
    curve, final, n_trades, dd, sharpe = run_filtered_portfolio(df_go, ffunc, STARTING_CAPITAL)
    ret = (final - STARTING_CAPITAL) / STARTING_CAPITAL

    filtered = df_go[df_go.apply(ffunc, axis=1)]
    jun_filtered = filtered[filtered['period'] == 'june']
    jun_n = len(jun_filtered)

    jun_curve = curve[curve['date'] >= JUNE_START]
    if len(jun_curve) > 0:
        jun_pnl = jun_curve['day_pnl'].sum()
    else:
        jun_pnl = 0

    portfolio_results[fname] = {
        'curve': curve, 'final': final, 'n_trades': n_trades,
        'ret': ret, 'dd': dd, 'sharpe': sharpe,
        'jun_n': jun_n, 'jun_pnl': jun_pnl,
    }

    print(f"  {fname:<30} {n_trades:>7} ${final/1000:>8.1f}K {ret*100:>+7.1f}% "
          f"{dd*100:>+6.1f}% {sharpe:>6.2f} {jun_n:>6} ${jun_pnl:>+9,.0f}")


# ==============================================================================
# TABLE 6: BLOCKED TRADES DETAIL — Which specific trades did each filter block?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 6: JUNE TRADES — Blocked vs Kept by Micro Filter (TC>=1 OR TV>=50)")
print(f"{'='*W}")

micro_filter = lambda s: s['raw_trade_count'] >= 1 or s['raw_total_vol'] >= 50

jun_signals = df_go[df_go['period'] == 'june'].sort_values(['date_str', 'ticker'])
print(f"\n  {'Date':<12} {'Ticker':<7} {'Prob':>5} {'PnL':>7} {'Exit':>4} "
      f"{'TC':>5} {'TV':>8} {'Spread':>8} {'$Vol':>10} {'ATR14':>7} {'Status':>10}")
print(f"  {'─'*12} {'─'*7} {'─'*5} {'─'*7} {'─'*4} "
      f"{'─'*5} {'─'*8} {'─'*8} {'─'*10} {'─'*7} {'─'*10}")

for _, sig in jun_signals.iterrows():
    passes = micro_filter(sig)
    status = 'KEEP' if passes else 'BLOCKED'
    marker = '  ' if passes else '>>'
    pnl_str = f"{sig['trade_pnl']*100:>+6.2f}%"
    print(f"{marker}{sig['date_str']:<12} {sig['ticker']:<7} {sig['prob']:.3f} {pnl_str} {sig['exit_type']:>4} "
          f"{sig['raw_trade_count']:>5.0f} {sig['raw_total_vol']:>8,.0f} {sig['raw_spread']:>8.4f} "
          f"${sig['dollar_vol']:>9,.0f} {sig['raw_atr14']:.4f} {status:>10}")

blocked_jun = jun_signals[~jun_signals.apply(micro_filter, axis=1)]
kept_jun = jun_signals[jun_signals.apply(micro_filter, axis=1)]

if len(blocked_jun) > 0:
    blocked_pnl = blocked_jun['trade_pnl'].mean() * 100
    blocked_wr = (blocked_jun['trade_pnl'] > 0).mean() * 100
    print(f"\n  Blocked: {len(blocked_jun)} trades, WR={blocked_wr:.0f}%, AvgPnL={blocked_pnl:+.2f}%")
if len(kept_jun) > 0:
    kept_pnl = kept_jun['trade_pnl'].mean() * 100
    kept_wr = (kept_jun['trade_pnl'] > 0).mean() * 100
    print(f"  Kept:    {len(kept_jun)} trades, WR={kept_wr:.0f}%, AvgPnL={kept_pnl:+.2f}%")


# ==============================================================================
# TABLE 7: ATR-BASED POSITION SIZING ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 7: ATR-BASED POSITION SIZING — Inverse ATR weighting")
print(f"{'='*W}")

print(f"\n  If positions were sized inversely to ATR_14 (lower ATR = more capital):")
print(f"  This reduces exposure on high-volatility days without blocking signals.\n")

atr_med = df_go['raw_atr14'].median()
print(f"  ATR_14 median across all GO signals: {atr_med:.4f}")

for period_label, period_key in [('Apr-May', 'apr_may'), ('June', 'june')]:
    subset = df_go[df_go['period'] == period_key]
    if len(subset) == 0:
        continue
    print(f"\n  {period_label} ATR distribution:")
    for pct in [25, 50, 75, 90, 95]:
        val = np.percentile(subset['raw_atr14'], pct)
        print(f"    P{pct}: {val:.4f}")

    # Simulate inverse-ATR weighting
    daily_groups = subset.groupby('date_str')
    total_pnl_equal = 0
    total_pnl_invATR = 0
    for date, group in daily_groups:
        n = len(group)
        eq_weight = 1.0 / n
        atr_vals = group['raw_atr14'].values
        inv_atr = 1.0 / np.clip(atr_vals, 0.01, None)
        inv_weights = inv_atr / inv_atr.sum()

        for i, (_, row) in enumerate(group.iterrows()):
            total_pnl_equal += eq_weight * row['trade_pnl']
            total_pnl_invATR += inv_weights[i] * row['trade_pnl']

    print(f"  Equal-weight PnL:    {total_pnl_equal*100:>+8.2f}%")
    print(f"  Inv-ATR weighted:    {total_pnl_invATR*100:>+8.2f}%")
    print(f"  Delta:               {(total_pnl_invATR - total_pnl_equal)*100:>+8.2f}%")


# ==============================================================================
# TABLE 8: DOLLAR VOLUME ANALYSIS — Live scorer has $50K minimum
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 8: DOLLAR VOLUME ANALYSIS — Live scorer uses MIN_DOLLAR_VOLUME=$50K")
print(f"{'='*W}")

print(f"\n  Current live scorer: MIN_DOLLAR_VOLUME = $50,000")
print(f"  Backtests/analysis scripts: NO dollar volume check")
print(f"\n  Dollar volume distribution at entry bar:\n")

for period_label, period_key in [('Apr-May', 'apr_may'), ('June', 'june')]:
    subset = df_go[df_go['period'] == period_key]
    if len(subset) == 0:
        continue
    print(f"  {period_label}:")
    for thresh in [0, 10_000, 25_000, 50_000, 100_000]:
        below = (subset['dollar_vol'] < thresh).sum() if thresh > 0 else 0
        above = (subset['dollar_vol'] >= thresh).sum() if thresh > 0 else len(subset)
        if thresh == 0:
            print(f"    Total signals: {len(subset)}")
        else:
            pct = below / len(subset) * 100
            subset_below = subset[subset['dollar_vol'] < thresh]
            below_wr = (subset_below['trade_pnl'] > 0).mean() * 100 if len(subset_below) > 0 else 0
            below_avg = subset_below['trade_pnl'].mean() * 100 if len(subset_below) > 0 else 0
            print(f"    Below ${thresh/1000:.0f}K: {below}/{len(subset)} ({pct:.0f}%)  "
                  f"WR={below_wr:.0f}%  AvgPnL={below_avg:+.2f}%")
    print()


# ==============================================================================
# TABLE 9: PER-TICKER MICROSTRUCTURE HEALTH
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 9: PER-TICKER MICROSTRUCTURE HEALTH — Which tickers have dead micro?")
print(f"{'='*W}")

print(f"\n  {'Ticker':<7} {'GO#':>4} {'TC=0':>5} {'TV=0':>5} {'Sp=0':>5} {'BIR=0':>6} "
      f"{'AvgTC':>7} {'AvgTV':>10} {'AvgSp':>8} {'AvgATR':>7} {'WR':>5} {'AvgPnL':>8}")
print(f"  {'─'*7} {'─'*4} {'─'*5} {'─'*5} {'─'*5} {'─'*6} "
      f"{'─'*7} {'─'*10} {'─'*8} {'─'*7} {'─'*5} {'─'*8}")

for ticker in sorted(df_go['ticker'].unique()):
    tsub = df_go[df_go['ticker'] == ticker]
    n = len(tsub)
    tc0 = (tsub['raw_trade_count'] == 0).sum()
    tv0 = (tsub['raw_total_vol'] == 0).sum()
    sp0 = (tsub['raw_spread'] == 0).sum()
    bir0 = (tsub['raw_bar_is_real'] == 0).sum()
    avg_tc = tsub['raw_trade_count'].mean()
    avg_tv = tsub['raw_total_vol'].mean()
    avg_sp = tsub['raw_spread'].mean()
    avg_atr = tsub['raw_atr14'].mean()
    wr = (tsub['trade_pnl'] > 0).mean() * 100
    avg_pnl = tsub['trade_pnl'].mean() * 100

    flag = ' '
    if tc0 > n * 0.5 or tv0 > n * 0.5:
        flag = '!'
    print(f"{flag} {ticker:<7} {n:>4} {tc0:>5} {tv0:>5} {sp0:>5} {bir0:>6} "
          f"{avg_tc:>7.1f} {avg_tv:>10,.0f} {avg_sp:>8.4f} {avg_atr:>7.4f} "
          f"{wr:>4.0f}% {avg_pnl:>+7.2f}%")


# ==============================================================================
# TABLE 10: VOLUME BASELINE COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 10: VOLUME BASELINES — 30-day median at entry_minute (pre-test)")
print(f"{'='*W}")

print(f"\n  These baselines come from 30 trading days before {TEST_START}.")
print(f"  vol_filter_pct checks: entry_volume >= (pct/100) × baseline")
print(f"  Problem: baseline uses OHLCV Volume, not microstructure Trade_Count/Total_Vol\n")

print(f"  {'Ticker':<7} {'Entry':>6} {'Vol_Med':>10} {'TC_Med':>7} {'TV_Med':>10} {'VF_Pct':>7} {'Status':>10}")
print(f"  {'─'*7} {'─'*6} {'─'*10} {'─'*7} {'─'*10} {'─'*7} {'─'*10}")

for ticker in sorted(ticker_vol_baselines.keys()):
    bl = ticker_vol_baselines[ticker]
    cfg = ticker_configs[ticker]
    vf = cfg['vol_filter_pct']
    em = cfg['entry_minute']
    em_str = f"{em // 60}:{em % 60:02d}"
    status = 'ACTIVE' if vf > 0 else 'DISABLED'
    print(f"  {ticker:<7} {em_str:>6} {bl['vol_median']:>10,.0f} {bl['tc_median']:>7,.0f} "
          f"{bl['tv_median']:>10,.0f} {vf:>6}% {status:>10}")


# ==============================================================================
# CHARTS
# ==============================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Signal Quality Filter Evaluation — V9', fontsize=14, fontweight='bold')

    # 1. Equity curves for top filters
    ax = axes[0, 0]
    colors = ['#666', '#1976d2', '#d32f2f', '#4caf50', '#ff9800', '#9c27b0', '#00bcd4', '#795548']
    for i, (fname, _) in enumerate(portfolio_filters):
        r = portfolio_results[fname]
        curve = r['curve']
        dates = pd.to_datetime(curve['date'])
        eq = curve['equity'].values / 1000
        lw = 2.5 if i in (0, 2, 4) else 1.0
        alpha = 1.0 if lw > 1 else 0.6
        ax.plot(dates, eq, color=colors[i % len(colors)], linewidth=lw, alpha=alpha,
                label=fname.split(': ')[0] if ':' in fname else fname[:20])
    ax.axvline(x=pd.Timestamp(JUNE_START), color='red', linestyle='--', alpha=0.4)
    ax.set_ylabel('Equity ($K)')
    ax.set_title('Equity Curves with Filters')
    ax.legend(fontsize=6, loc='upper left')
    ax.tick_params(axis='x', rotation=30, labelsize=7)

    # 2. Microstructure distribution: winners vs losers
    ax = axes[0, 1]
    win_tc = df_go[df_go['trade_pnl'] > 0]['raw_trade_count']
    loss_tc = df_go[df_go['trade_pnl'] <= 0]['raw_trade_count']
    bins = np.arange(0, max(win_tc.max(), loss_tc.max()) + 2) - 0.5
    ax.hist(win_tc, bins=bins, alpha=0.6, color='#4caf50', label=f'Winners (n={len(win_tc)})', density=True)
    ax.hist(loss_tc, bins=bins, alpha=0.6, color='#d32f2f', label=f'Losers (n={len(loss_tc)})', density=True)
    ax.set_xlabel('Trade_Count at entry bar')
    ax.set_ylabel('Density')
    ax.set_title('Trade_Count Distribution: Winners vs Losers')
    ax.legend()

    # 3. ATR vs PnL scatter
    ax = axes[1, 0]
    am = df_go[df_go['period'] == 'apr_may']
    jn = df_go[df_go['period'] == 'june']
    ax.scatter(am['raw_atr14'], am['trade_pnl'] * 100, alpha=0.5, s=30,
               color='#1976d2', label='Apr-May', edgecolors='black', linewidth=0.3)
    ax.scatter(jn['raw_atr14'], jn['trade_pnl'] * 100, alpha=0.8, s=60,
               color='#d32f2f', label='June', edgecolors='black', linewidth=0.5, marker='D')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('ATR_14 at entry')
    ax.set_ylabel('Trade PnL (%)')
    ax.set_title('ATR_14 vs Trade Outcome')
    ax.legend()

    # 4. Dollar volume vs PnL
    ax = axes[1, 1]
    ax.scatter(am['dollar_vol'] / 1000, am['trade_pnl'] * 100, alpha=0.5, s=30,
               color='#1976d2', label='Apr-May', edgecolors='black', linewidth=0.3)
    ax.scatter(jn['dollar_vol'] / 1000, jn['trade_pnl'] * 100, alpha=0.8, s=60,
               color='#d32f2f', label='June', edgecolors='black', linewidth=0.5, marker='D')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=50, color='orange', linestyle='--', alpha=0.5, label='$50K threshold')
    ax.axvline(x=25, color='green', linestyle='--', alpha=0.5, label='$25K threshold')
    ax.set_xlabel('Dollar Volume at entry ($K)')
    ax.set_ylabel('Trade PnL (%)')
    ax.set_title('Dollar Volume vs Trade Outcome')
    ax.legend(fontsize=7)

    plt.tight_layout()
    chart_path = '/tmp/signal_filters.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Charts saved: {chart_path}")
    plt.close()

except ImportError:
    print("\n  (matplotlib not available — skipping charts)")


# ==============================================================================
# SYNTHESIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SYNTHESIS — SIGNAL QUALITY FILTER FINDINGS")
print(f"{'='*W}")

print(f"""
  GAP 1: vol_filter_pct DISABLED
    27/28 deployed tickers have vol_filter_pct = 0 (filter off).
    WF optimizer grid is [0, 25] — only tests "off" vs "25%".
    vol_filter_pct = 0 always wins optimization because it keeps more trades.
    FIX: Make microstructure gate mandatory, not optimizer-selected.

  GAP 2: vol_filter checks OHLCV Volume, not microstructure
    The filter at daily_score_combined.py:618 checks bar Volume (OHLCV feed).
    It NEVER checks Trade_Count, Total_Vol, or Spread (microstructure feed).
    A bar can have Volume > 0 from OHLCV but Trade_Count = 0 from micro feed.
    FIX: Add hard gate: require Trade_Count >= 1 OR Total_Vol >= 50.

  GAP 3: Bar_Is_Real is a feature, not a gate
    Bar_Is_Real = (Volume > 0) is a model input, not a hard filter.
    The model can fire GO even when Bar_Is_Real = 0.
    FIX: Redundant if Gap 2 is fixed (micro gate is stricter).

  GAP 4: Dollar volume not checked in backtests
    Live scorer has MIN_DOLLAR_VOLUME = $50,000 but backtests skip it.
    FIX: Add dollar volume check to all backtest/analysis scripts.
""")

# Compute best filter recommendation
best_filter = None
best_score = -999
for fname, ffunc in combo_filters:
    if fname == 'Baseline':
        continue
    am_keep = apr_may[apr_may.apply(ffunc, axis=1)] if len(apr_may) > 0 else pd.DataFrame()
    jun_keep = june[june.apply(ffunc, axis=1)] if len(june) > 0 else pd.DataFrame()
    am_n = len(am_keep)
    jun_n = len(jun_keep)
    am_loss = len(apr_may) - am_n
    jun_blocked = len(june) - jun_n

    if am_loss > len(apr_may) * 0.15:
        continue

    am_wr = (am_keep['trade_pnl'] > 0).mean() if am_n > 0 else 0
    jun_wr = (jun_keep['trade_pnl'] > 0).mean() if jun_n > 0 else 0
    jun_pnl = jun_keep['trade_pnl'].sum() if jun_n > 0 else 0

    score = jun_blocked * 2 - am_loss + (jun_pnl - baseline_jun_pnl) * 10
    if score > best_score:
        best_score = score
        best_filter = fname

if best_filter:
    print(f"  RECOMMENDED FILTER: {best_filter}")
    print(f"  Rationale: Best tradeoff between blocking June losses and preserving Apr-May wins.")
else:
    print(f"  No single filter clearly dominates — see Table 4 for tradeoffs.")

print(f"\n{'='*W}")
print(f"  SIGNAL QUALITY FILTER EVALUATION COMPLETE")
print(f"{'='*W}")
