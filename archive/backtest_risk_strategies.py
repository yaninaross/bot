"""
backtest_risk_strategies.py — Risk Management Strategy Evaluation for V9
========================================================================
Loads V9 portfolio from BQ, runs TF model inference, collects all GO signals
with trade outcomes (same as analyze_june_deep.py), then replays through
12 portfolio allocation + risk management strategies:

  Baseline: equal-weight, cap-2, cap-3
  Score-weighted: S8_mix proportional, score-ranked cap-2/3, score-tiered
  Risk management: max-risk cap, correlation-aware, VIX scaling, combined

All trades enter simultaneously (6:20-6:30 AM PST window). Risk management
is purely pre-entry: which tickers to include and how to size them.

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
DAILY_LOSS_LIMIT = 0.05
CORR_LOOKBACK = 5
CORR_THRESHOLD = 0.50
VIX_STRESS_THRESHOLD = 0.03

BUCKET_NAME    = "time-series-bucket-ys"
BQ_TABLE       = "ys-time-series.scoring_logs.wf_optimization_results"
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
print(f"  RISK MANAGEMENT STRATEGY EVALUATION — V9 Portfolio")
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
    }

deployed_tickers = sorted(ticker_configs.keys())
print(f"  Deployed: {deployed_tickers}")

# Show score distribution
scores = [(t, ticker_configs[t]['s8_mix']) for t in deployed_tickers]
scores.sort(key=lambda x: x[1], reverse=True)
print(f"\n  S8_mix scores (higher = better historical quality):")
for t, s in scores:
    print(f"    {t:<7} {s:.6f}")


# ==============================================================================
# STEP 2: LOAD DATA + RUN MODEL INFERENCE + COLLECT GO SIGNALS
# ==============================================================================
print(f"\n  Loading OHLCV + running model inference for {len(deployed_tickers)} tickers...")
market_df = build_market_context(start_date=TEST_START)

test_start_utc = pd.Timestamp(TEST_START, tz='UTC')
test_end_utc = pd.Timestamp(TEST_END, tz='UTC') + timedelta(days=1)
june_start_utc = pd.Timestamp(JUNE_START, tz='UTC')

all_signals = []
ticker_daily_returns = {}
daily_vix_anchor = {}

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

    # Daily returns for cross-ticker correlation
    df_pst = df_raw.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    test_s_pst = pd.Timestamp(TEST_START, tz=PST)
    test_e_pst = pd.Timestamp(TEST_END, tz=PST) + timedelta(days=1)
    ohlcv_test = df_pst[(df_pst.index >= test_s_pst - pd.Timedelta(days=10)) & (df_pst.index < test_e_pst)]
    if not ohlcv_test.empty:
        daily = ohlcv_test[ohlcv_test['Volume'] > 0].resample('D').agg(
            {'Open': 'first', 'Close': 'last'}).dropna()
        if len(daily) > 0:
            daily['ret'] = daily['Close'] / daily['Open'] - 1
            ticker_daily_returns[ticker] = daily['ret']

    # Engineer features
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

    # Collect GO signals
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

        # Capture VIX at entry for regime scaling
        vix_col = 'VIX_Anchor_ROC'
        vix_val = float(df_feat[vix_col].iloc[df_feat.index.get_loc(ts)] if ts in df_feat.index else 0)
        date_str = ts_pst.strftime('%Y-%m-%d')
        if date_str not in daily_vix_anchor:
            daily_vix_anchor[date_str] = vix_val

        period = 'june' if ts >= june_start_utc else 'apr_may'
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
            'vix_anchor_roc': vix_val,
            'period': period,
            'max_gain': max_gain,
            'max_drop': max_drop,
        })

    del df_raw, df_feat, df_scaled, df_pst, XA, XC; _release_memory()
    n_go = sum(1 for s in all_signals if s['ticker'] == ticker)
    print(f"done ({len(positions)} bars, {n_go} GO)")

_master_cache.clear(); _model_cache.clear(); _release_memory()

df_go = pd.DataFrame(all_signals)
print(f"\n  Total GO signals: {len(df_go)}")
print(f"  Apr-May: {len(df_go[df_go['period']=='apr_may'])}, June: {len(df_go[df_go['period']=='june'])}")


# ==============================================================================
# STEP 3: COMPUTE ROLLING PAIRWISE CORRELATIONS
# ==============================================================================
print(f"\n  Computing {CORR_LOOKBACK}-day rolling pairwise correlations...")

ret_df = pd.DataFrame(ticker_daily_returns)
ret_df.index = pd.to_datetime(ret_df.index).tz_localize(None)

daily_avg_corr = {}
for date in sorted(df_go['date_str'].unique()):
    date_ts = pd.Timestamp(date)
    go_tickers = df_go[df_go['date_str'] == date]['ticker'].unique().tolist()
    go_in_data = [t for t in go_tickers if t in ret_df.columns]
    if len(go_in_data) < 2:
        daily_avg_corr[date] = 0.0
        continue
    lookback_start = date_ts - pd.Timedelta(days=CORR_LOOKBACK + 3)
    lookback_end = date_ts - pd.Timedelta(days=1)
    window = ret_df.loc[(ret_df.index >= lookback_start) & (ret_df.index <= lookback_end), go_in_data]
    if len(window) < 3:
        daily_avg_corr[date] = 0.0
        continue
    corr_mat = window.corr()
    n = len(go_in_data)
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = corr_mat.values[mask_upper]
    valid_vals = vals[~np.isnan(vals)]
    daily_avg_corr[date] = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0

print(f"  Correlation computed for {len(daily_avg_corr)} trading days")
for d in sorted(daily_avg_corr.keys()):
    if d >= JUNE_START:
        n_go = len(df_go[df_go['date_str'] == d])
        print(f"    {d}: avg corr = {daily_avg_corr[d]:+.3f}  ({n_go} GO signals)")


# ==============================================================================
# STEP 4: STRATEGY FUNCTIONS
# ==============================================================================

def run_strategy(df_go, strategy_fn, starting_capital):
    """Run a strategy over GO signals, return equity curve and stats."""
    daily_groups = df_go.groupby('date_str')
    equity = starting_capital
    curve = []
    all_taken = []

    for date in sorted(daily_groups.groups.keys()):
        group = daily_groups.get_group(date)
        trades = group.to_dict('records')

        selected, weights = strategy_fn(trades, date, equity)

        if not selected:
            curve.append({'date': date, 'equity': equity, 'day_pnl': 0,
                          'n_taken': 0, 'n_available': len(trades),
                          'capital_deployed': 0})
            continue

        day_pnl = 0
        for i, trade in enumerate(selected):
            trade_dollar = equity * weights[i] * trade['trade_pnl']
            day_pnl += trade_dollar
            all_taken.append({**trade, 'weight': weights[i], 'dollar_pnl': trade_dollar})

        equity += day_pnl
        capital_deployed = sum(weights)
        curve.append({'date': date, 'equity': equity, 'day_pnl': day_pnl,
                      'n_taken': len(selected), 'n_available': len(trades),
                      'capital_deployed': capital_deployed})

    return pd.DataFrame(curve), equity, pd.DataFrame(all_taken)


# --- Baseline strategies ---

def strat_equal_weight(trades, date, equity):
    n = len(trades)
    return trades, [1.0 / n] * n

def strat_cap2_prob(trades, date, equity):
    ranked = sorted(trades, key=lambda t: t['prob'], reverse=True)
    selected = ranked[:2]
    n = len(selected)
    return selected, [1.0 / n] * n

def strat_cap3_prob(trades, date, equity):
    ranked = sorted(trades, key=lambda t: t['prob'], reverse=True)
    selected = ranked[:3]
    n = len(selected)
    return selected, [1.0 / n] * n


# --- Score-weighted strategies ---

def strat_score_weighted(trades, date, equity):
    scores = [t['s8_mix'] for t in trades]
    total = sum(scores)
    if total <= 0:
        n = len(trades)
        return trades, [1.0 / n] * n
    weights = [s / total for s in scores]
    return trades, weights

def strat_score_cap2(trades, date, equity):
    ranked = sorted(trades, key=lambda t: t['s8_mix'] * t['prob'], reverse=True)
    selected = ranked[:2]
    n = len(selected)
    return selected, [1.0 / n] * n

def strat_score_cap3(trades, date, equity):
    ranked = sorted(trades, key=lambda t: t['s8_mix'] * t['prob'], reverse=True)
    selected = ranked[:3]
    n = len(selected)
    return selected, [1.0 / n] * n

def strat_score_tiered(trades, date, equity):
    if len(trades) <= 2:
        n = len(trades)
        return trades, [1.0 / n] * n
    ranked = sorted(trades, key=lambda t: t['s8_mix'], reverse=True)
    n = len(ranked)
    tier1_n = max(1, n // 3)
    tier2_n = max(1, n // 3)
    tier3_n = n - tier1_n - tier2_n
    weights = []
    tier_budgets = [0.50, 0.33, 0.17]
    for i, t in enumerate(ranked):
        if i < tier1_n:
            weights.append(tier_budgets[0] / tier1_n)
        elif i < tier1_n + tier2_n:
            weights.append(tier_budgets[1] / tier2_n)
        else:
            weights.append(tier_budgets[2] / max(tier3_n, 1))
    return ranked, weights


# --- Risk management strategies ---

def strat_max_risk_cap(trades, date, equity):
    max_sl = max(t['sl_eff'] for t in trades)
    max_positions = max(1, int(DAILY_LOSS_LIMIT / max_sl))
    ranked = sorted(trades, key=lambda t: t['s8_mix'] * t['prob'], reverse=True)
    selected = ranked[:max_positions]
    n = len(selected)
    return selected, [1.0 / n] * n

def strat_corr_aware(trades, date, equity):
    avg_corr = daily_avg_corr.get(date, 0.0)
    scale = max(0.25, 1.0 - avg_corr)
    scores = [t['s8_mix'] for t in trades]
    total = sum(scores)
    if total <= 0:
        n = len(trades)
        base_weights = [1.0 / n] * n
    else:
        base_weights = [s / total for s in scores]
    weights = [w * scale for w in base_weights]
    return trades, weights

def strat_score_max_risk(trades, date, equity):
    max_sl = max(t['sl_eff'] for t in trades)
    max_positions = max(1, int(DAILY_LOSS_LIMIT / max_sl))
    ranked = sorted(trades, key=lambda t: t['s8_mix'] * t['prob'], reverse=True)
    selected = ranked[:max_positions]
    scores = [t['s8_mix'] for t in selected]
    total = sum(scores)
    if total <= 0:
        n = len(selected)
        return selected, [1.0 / n] * n
    weights = [s / total for s in scores]
    return selected, weights

def strat_vix_scaling(trades, date, equity):
    vix_val = daily_vix_anchor.get(date, 0.0)
    scale = 0.50 if vix_val > VIX_STRESS_THRESHOLD else 1.0
    scores = [t['s8_mix'] for t in trades]
    total = sum(scores)
    if total <= 0:
        n = len(trades)
        base_weights = [1.0 / n] * n
    else:
        base_weights = [s / total for s in scores]
    weights = [w * scale for w in base_weights]
    return trades, weights

def strat_combined(trades, date, equity):
    ranked = sorted(trades, key=lambda t: t['s8_mix'] * t['prob'], reverse=True)
    selected = ranked[:3]
    avg_corr = daily_avg_corr.get(date, 0.0)
    corr_scale = max(0.25, 1.0 - avg_corr)
    vix_val = daily_vix_anchor.get(date, 0.0)
    vix_scale = 0.50 if vix_val > VIX_STRESS_THRESHOLD else 1.0
    scale = min(corr_scale, vix_scale)
    scores = [t['s8_mix'] for t in selected]
    total = sum(scores)
    if total <= 0:
        n = len(selected)
        base_weights = [1.0 / n] * n
    else:
        base_weights = [s / total for s in scores]
    weights = [w * scale for w in base_weights]
    return selected, weights


STRATEGIES = [
    ('1. Equal-weight',           strat_equal_weight),
    ('2. Cap-2 (prob)',           strat_cap2_prob),
    ('3. Cap-3 (prob)',           strat_cap3_prob),
    ('4. Score-weighted',         strat_score_weighted),
    ('5. Score cap-2',            strat_score_cap2),
    ('6. Score cap-3',            strat_score_cap3),
    ('7. Score-tiered',           strat_score_tiered),
    ('8. Max-risk cap',           strat_max_risk_cap),
    ('9. Corr-aware',             strat_corr_aware),
    ('10. Score+risk cap',        strat_score_max_risk),
    ('11. VIX scaling',           strat_vix_scaling),
    ('12. Combined',              strat_combined),
]


# ==============================================================================
# STEP 5: RUN ALL STRATEGIES
# ==============================================================================
print(f"\n{'='*W}")
print(f"  RUNNING {len(STRATEGIES)} STRATEGIES...")
print(f"{'='*W}")

results = {}
for name, fn in STRATEGIES:
    curve, final, taken = run_strategy(df_go, fn, STARTING_CAPITAL)
    ret = (final - STARTING_CAPITAL) / STARTING_CAPITAL

    n_trades = len(taken)
    wr = (taken['trade_pnl'] > 0).mean() if n_trades > 0 else 0
    avg_pnl = taken['trade_pnl'].mean() if n_trades > 0 else 0

    peak = curve['equity'].cummax()
    dd = ((curve['equity'] - peak) / peak).min() if len(curve) > 0 else 0

    active = curve[curve['n_taken'] > 0]
    if len(active) > 1:
        daily_rets = active['day_pnl'] / (active['equity'] - active['day_pnl'])
        sharpe = float(daily_rets.mean() / max(daily_rets.std(), 1e-8) * np.sqrt(252))
    else:
        sharpe = 0

    # June-specific
    jun_taken = taken[taken['period'] == 'june'] if 'period' in taken.columns and n_trades > 0 else pd.DataFrame()
    jun_n = len(jun_taken)
    jun_wr = (jun_taken['trade_pnl'] > 0).mean() if jun_n > 0 else 0
    jun_pnl = jun_taken['dollar_pnl'].sum() if jun_n > 0 else 0

    jun_curve = curve[curve['date'] >= JUNE_START]
    jun_peak = 0
    jun_dd = 0
    if len(jun_curve) > 0:
        jun_eq = jun_curve['equity']
        jun_running_max = jun_eq.cummax()
        jun_dd = ((jun_eq - jun_running_max) / jun_running_max).min()

    results[name] = {
        'curve': curve, 'taken': taken, 'final': final, 'ret': ret,
        'n_trades': n_trades, 'wr': wr, 'avg_pnl': avg_pnl,
        'dd': dd, 'sharpe': sharpe,
        'jun_n': jun_n, 'jun_wr': jun_wr, 'jun_pnl': jun_pnl, 'jun_dd': jun_dd,
    }
    print(f"  {name:<25} → ${final/1000:>7.1f}K  ({ret:+.1%})  {n_trades} trades  Sharpe {sharpe:.2f}")


# ==============================================================================
# TABLE 1: STRATEGY COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 1: STRATEGY COMPARISON — ${STARTING_CAPITAL/1000:.0f}K, {TEST_START} → {TEST_END}")
print(f"{'='*W}")

print(f"\n  {'Strategy':<25} {'Trades':>6} {'WR':>5} {'AvgPnL':>8} {'Return':>8} {'MaxDD':>7} {'Sharpe':>7} {'Final$':>10}")
print(f"  {'─'*25} {'─'*6} {'─'*5} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*10}")

baseline_final = results['1. Equal-weight']['final']
for name in [n for n, _ in STRATEGIES]:
    r = results[name]
    delta = r['final'] - baseline_final
    delta_str = f"({delta:+,.0f})" if name != '1. Equal-weight' else ''
    print(f"  {name:<25} {r['n_trades']:>6} {r['wr']*100:>4.0f}% {r['avg_pnl']*100:>+7.2f}% "
          f"{r['ret']*100:>+7.1f}% {r['dd']*100:>+6.1f}% {r['sharpe']:>6.2f} "
          f"${r['final']/1000:>8.1f}K {delta_str}")


# ==============================================================================
# TABLE 2: JUNE-SPECIFIC IMPACT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 2: JUNE PERFORMANCE BY STRATEGY")
print(f"{'='*W}")

baseline_jun_pnl = results['1. Equal-weight']['jun_pnl']

print(f"\n  {'Strategy':<25} {'Jun Trades':>10} {'Jun WR':>7} {'Jun $PnL':>10} {'Jun DD':>7} {'Saved$':>10}")
print(f"  {'─'*25} {'─'*10} {'─'*7} {'─'*10} {'─'*7} {'─'*10}")

for name in [n for n, _ in STRATEGIES]:
    r = results[name]
    saved = r['jun_pnl'] - baseline_jun_pnl
    saved_str = f"${saved:+,.0f}" if name != '1. Equal-weight' else '—'
    print(f"  {name:<25} {r['jun_n']:>10} {r['jun_wr']*100:>6.0f}% "
          f"${r['jun_pnl']:>+9,.0f} {r['jun_dd']*100:>+6.1f}% {saved_str:>10}")


# ==============================================================================
# TABLE 3: DAY-BY-DAY JUNE LOG
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 3: JUNE DAY-BY-DAY — Top strategies compared")
print(f"{'='*W}")

show_strategies = ['1. Equal-weight', '4. Score-weighted', '6. Score cap-3',
                   '8. Max-risk cap', '12. Combined']

header = f"  {'Date':<12} {'Avail':>5}"
for s in show_strategies:
    short = s.split('. ')[1][:10]
    header += f"  {short:>13}"
print(f"\n{header}")
print(f"  {'─'*12} {'─'*5}" + "  ".join(['─'*13]*len(show_strategies)))

june_dates = sorted(df_go[df_go['period'] == 'june']['date_str'].unique())
for date in june_dates:
    n_avail = len(df_go[df_go['date_str'] == date])
    line = f"  {date:<12} {n_avail:>5}"
    for sname in show_strategies:
        r = results[sname]
        day_row = r['curve'][r['curve']['date'] == date]
        if len(day_row) > 0:
            dr = day_row.iloc[0]
            line += f"  {int(dr['n_taken']):>2}/${dr['day_pnl']:>+9,.0f}"
        else:
            line += f"  {'—':>13}"
    print(line)


# ==============================================================================
# TABLE 4: WORST-DAY ANALYSIS (Jun 3, 5, 8)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 4: WORST-DAY DEEP DIVE — Which tickers did each strategy select?")
print(f"{'='*W}")

worst_days = ['2026-06-03', '2026-06-05', '2026-06-08']

for day in worst_days:
    day_signals = df_go[df_go['date_str'] == day].sort_values('s8_mix', ascending=False)
    if len(day_signals) == 0:
        continue

    corr_val = daily_avg_corr.get(day, 0.0)
    vix_val = daily_vix_anchor.get(day, 0.0)

    print(f"\n  {day}  |  {len(day_signals)} GO signals  |  Corr={corr_val:+.3f}  |  VIX_ROC={vix_val:+.4f}")
    print(f"  Available signals (ranked by S8_mix):")
    print(f"    {'Ticker':<7} {'S8_mix':>10} {'Prob':>6} {'Score*P':>10} {'PnL':>8} {'Exit':>4}")
    print(f"    {'─'*7} {'─'*10} {'─'*6} {'─'*10} {'─'*8} {'─'*4}")
    for _, sig in day_signals.iterrows():
        print(f"    {sig['ticker']:<7} {sig['s8_mix']:.6f} {sig['prob']:.3f} "
              f"{sig['s8_mix']*sig['prob']:.6f} {sig['trade_pnl']*100:>+7.2f}% {sig['exit_type']:>4}")

    print(f"\n  Strategy selections:")
    for sname in show_strategies:
        taken = results[sname]['taken']
        day_taken = taken[taken['date_str'] == day] if len(taken) > 0 else pd.DataFrame()
        if len(day_taken) > 0:
            tickers = ', '.join(day_taken['ticker'].tolist())
            day_pnl = day_taken['dollar_pnl'].sum()
            cap_deployed = day_taken['weight'].sum()
            print(f"    {sname:<25} → {tickers:<30} ${day_pnl:>+9,.0f}  ({cap_deployed:.0%} capital)")
        else:
            print(f"    {sname:<25} → no trades")


# ==============================================================================
# TABLE 5: SCORE PREDICTIVE VALUE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 5: DOES S8_MIX SCORE PREDICT PERFORMANCE?")
print(f"{'='*W}")

score_med = df_go['s8_mix'].median()
high_score = df_go[df_go['s8_mix'] >= score_med]
low_score = df_go[df_go['s8_mix'] < score_med]

print(f"\n  Score median: {score_med:.6f}")
print(f"\n  {'Group':<20} {'Trades':>7} {'WR':>5} {'AvgPnL':>8} {'TotalPnL':>10}")
print(f"  {'─'*20} {'─'*7} {'─'*5} {'─'*8} {'─'*10}")

for label, subset in [('High score (≥med)', high_score), ('Low score (<med)', low_score)]:
    n = len(subset)
    wr = (subset['trade_pnl'] > 0).mean() if n > 0 else 0
    avg = subset['trade_pnl'].mean() if n > 0 else 0
    total = subset['trade_pnl'].sum() if n > 0 else 0
    print(f"  {label:<20} {n:>7} {wr*100:>4.0f}% {avg*100:>+7.2f}% {total*100:>+9.1f}%")

# By period
for period_label, period_key in [('Apr-May', 'apr_may'), ('June', 'june')]:
    print(f"\n  {period_label}:")
    psub = df_go[df_go['period'] == period_key]
    for label, mask_fn in [('High score', lambda x: x['s8_mix'] >= score_med),
                           ('Low score',  lambda x: x['s8_mix'] < score_med)]:
        subset = psub[mask_fn(psub)]
        n = len(subset)
        wr = (subset['trade_pnl'] > 0).mean() if n > 0 else 0
        avg = subset['trade_pnl'].mean() if n > 0 else 0
        print(f"    {label:<18} {n:>5} trades  WR={wr*100:.0f}%  AvgPnL={avg*100:+.2f}%")


# ==============================================================================
# CHARTS
# ==============================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Risk Management Strategy Evaluation — V9, ${STARTING_CAPITAL/1000:.0f}K', fontsize=14, fontweight='bold')

    # 1. Equity curves
    ax = axes[0, 0]
    colors = ['#666', '#1976d2', '#2196f3', '#d32f2f', '#f44336', '#ff9800',
              '#4caf50', '#9c27b0', '#00bcd4', '#795548', '#607d8b', '#e91e63']
    for i, (name, _) in enumerate(STRATEGIES):
        curve = results[name]['curve']
        dates = pd.to_datetime(curve['date'])
        eq = curve['equity'].values / 1000
        lw = 2.5 if name in ('1. Equal-weight', '6. Score cap-3', '12. Combined') else 0.8
        alpha = 1.0 if lw > 1 else 0.5
        ax.plot(dates, eq, color=colors[i % len(colors)], linewidth=lw, alpha=alpha,
                label=name.split('. ')[1])
    ax.axvline(x=pd.Timestamp(JUNE_START), color='red', linestyle='--', alpha=0.4, label='June start')
    ax.axhline(y=STARTING_CAPITAL / 1000, color='gray', linestyle=':', alpha=0.3)
    ax.set_ylabel('Equity ($K)')
    ax.set_title('Equity Curves')
    ax.legend(fontsize=6, ncol=2, loc='upper left')
    ax.tick_params(axis='x', rotation=30, labelsize=7)

    # 2. June PnL comparison
    ax = axes[0, 1]
    names_short = [n.split('. ')[1][:12] for n, _ in STRATEGIES]
    jun_pnls = [results[n]['jun_pnl'] / 1000 for n, _ in STRATEGIES]
    bar_colors = ['#4caf50' if p >= 0 else '#d32f2f' for p in jun_pnls]
    ax.barh(range(len(STRATEGIES)), jun_pnls, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(STRATEGIES)))
    ax.set_yticklabels(names_short, fontsize=8)
    ax.set_xlabel('June P&L ($K)')
    ax.set_title('June Dollar Impact by Strategy')
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.invert_yaxis()

    # 3. Risk-return scatter
    ax = axes[1, 0]
    for i, (name, _) in enumerate(STRATEGIES):
        r = results[name]
        ax.scatter(r['dd'] * 100, r['ret'] * 100, s=80, color=colors[i % len(colors)],
                   zorder=3, edgecolors='black', linewidth=0.5)
        ax.annotate(name.split('. ')[1][:10], (r['dd'] * 100, r['ret'] * 100),
                    fontsize=6, ha='left', va='bottom')
    ax.set_xlabel('Max Drawdown (%)')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Risk vs Return')

    # 4. Trades per day distribution (baseline vs combined)
    ax = axes[1, 1]
    for sname, color, label in [('1. Equal-weight', '#1976d2', 'Equal-weight'),
                                 ('12. Combined', '#d32f2f', 'Combined')]:
        curve = results[sname]['curve']
        active = curve[curve['n_taken'] > 0]
        counts = active['n_taken'].value_counts().sort_index()
        ax.bar(counts.index + (0.2 if sname == '12. Combined' else -0.2),
               counts.values, width=0.35, color=color, alpha=0.7, label=label)
    ax.set_xlabel('Positions per Day')
    ax.set_ylabel('Frequency')
    ax.set_title('Position Count Distribution')
    ax.legend()

    plt.tight_layout()
    chart_path = '/tmp/risk_strategies.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Charts saved: {chart_path}")
    plt.close()

except ImportError:
    print("\n  (matplotlib not available — skipping charts)")


# ==============================================================================
# SYNTHESIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SYNTHESIS")
print(f"{'='*W}")

best_name = max(results.keys(), key=lambda k: results[k]['sharpe'])
best = results[best_name]
baseline = results['1. Equal-weight']

print(f"""
  BASELINE (equal-weight, no risk management):
    ${STARTING_CAPITAL/1000:.0f}K → ${baseline['final']/1000:.1f}K ({baseline['ret']:+.1%})
    {baseline['n_trades']} trades, {baseline['wr']:.0%} WR, Sharpe {baseline['sharpe']:.2f}
    Max DD: {baseline['dd']:.1%}
    June: {baseline['jun_n']} trades, ${baseline['jun_pnl']:+,.0f}

  BEST RISK-ADJUSTED ({best_name}):
    ${STARTING_CAPITAL/1000:.0f}K → ${best['final']/1000:.1f}K ({best['ret']:+.1%})
    {best['n_trades']} trades, {best['wr']:.0%} WR, Sharpe {best['sharpe']:.2f}
    Max DD: {best['dd']:.1%}
    June: {best['jun_n']} trades, ${best['jun_pnl']:+,.0f}
    June savings vs baseline: ${best['jun_pnl'] - baseline['jun_pnl']:+,.0f}
""")

print(f"{'='*W}")
print(f"  RISK STRATEGY EVALUATION COMPLETE")
print(f"{'='*W}")
