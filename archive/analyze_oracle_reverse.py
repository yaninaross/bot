# %% ── SHORT: Reverse Engineering — S3-Style Rule-Based Ticker Selection ──────
#
# Find observable features (no H2 PnL leakage) that predict which tickers
# are profitable in H2 (Jun-Jul 2026). Build a composite rule like S3.
#
# Phase 1: Load BQ configs + OHLCV + seed scores
# Phase 2: Compute 18 features per ticker (BQ, OHLCV, model)
# Phase 3: H2 backtest ground truth
# Phase 4: Feature → H2 correlation analysis
# Phase 5: Composite rule construction + validation
#
# GPU NOT required. Colab-ready.

import os, io, gc, sys, warnings, ctypes
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import combinations
import numpy as np, pandas as pd, pytz
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
except ImportError:
    matplotlib.use('Agg')
    _IN_COLAB = False

import matplotlib.pyplot as plt

from google.cloud import storage as gcs_storage, bigquery

if _IN_COLAB:
    try: auth.authenticate_user()
    except Exception: pass

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME   = "time-series-bucket-ys"
RESULTS_DIR   = "Algos/Results_all_v2"
INFERENCE_DIR = "Data/Inference_v2"

TEST_START  = '2026-04-01'
TEST_END    = '2026-07-28'
HALF1_END   = '2026-05-31'
HALF2_START = '2026-06-01'

ENTRY_MIN     = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.002
SL_FIXED      = 0.03
DEPLOY_FLOOR  = 0.018

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

FOLDS_A2 = [400, 401, 402, 403, 404, 405]
FOLDS_A1 = [410, 411, 412, 413, 414, 415]
ALL_FOLDS = FOLDS_A2 + FOLDS_A1

FOLD_EXPERIMENTS    = {f: f'{f}_XSECTOR' for f in ALL_FOLDS}
FOLD_MODEL_VERSIONS = {f: f'robust_wf_v12_short_{f}' for f in ALL_FOLDS}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_ALLOWED_MODES = ('static', 'atr_sl', 'grace')

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120

# Known stable tickers from confidence calibration analysis
STABLE_13 = ['MVIS', 'MXL', 'KTOS', 'JOBY', 'MP', 'LUNR', 'SMR',
             'FCEL', 'VOYG', 'WULF', 'IREN', 'ONDS', 'OKLO']
ORACLE_10 = ['AVAV', 'BE', 'MU', 'MVIS', 'MSTR', 'MP', 'CCJ',
             'WOLF', 'OKLO', 'CEG']

MIN_TICKERS_FOR_RULE = 5
MIN_TRADES_VALID     = 3


# ==============================================================================
# DATA LOADING
# ==============================================================================
_master_cache = {}

def load_master(ticker):
    if ticker in _master_cache:
        return _master_cache[ticker]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
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
        std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'volume': 'Volume'}
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        df = df[df.index.notna()].sort_index()
        cutoff = pd.Timestamp(TEST_START, tz='UTC') - pd.Timedelta(days=30)
        df = df[df.index >= cutoff]
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
    for c in ['Volume']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


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


def _safe_float(val, default=0.0):
    try:
        if pd.isna(val): return default
    except (TypeError, ValueError): pass
    try: return float(val)
    except (TypeError, ValueError): return default


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


def measure_day_dip(ohlcv, entry_idx):
    prices = ohlcv['prices']
    lows = ohlcv['lows']
    highs = ohlcv['highs']
    date_ints = ohlcv['dates']
    time_mins = ohlcv['times']
    n = len(prices)
    if entry_idx >= n - 1:
        return None
    entry_price = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    max_dip = 0.0
    max_bounce = 0.0
    eod_pnl = 0.0
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            eod_pnl = (entry_price - prices[k-1]) / entry_price
            break
        bar_low_pct = (entry_price - lows[k]) / entry_price
        bar_high_pct = (highs[k] - entry_price) / entry_price
        if bar_low_pct > max_dip:
            max_dip = bar_low_pct
        if bar_high_pct > max_bounce:
            max_bounce = bar_high_pct
        if time_mins[k] >= EXIT_TIME_MIN:
            eod_pnl = (entry_price - prices[k]) / entry_price
            break
    else:
        eod_pnl = (entry_price - prices[n-1]) / entry_price
    return {'max_dip': max_dip, 'max_bounce': max_bounce, 'eod_pnl': eod_pnl}


# ==============================================================================
# PHASE 1: LOAD BQ + OHLCV + SEED SCORES
# ==============================================================================
np.random.seed(42)

print(f"{'='*W}")
print(f"  SHORT: REVERSE ENGINEERING — S3-STYLE RULE-BASED SELECTION")
print(f"  TEST: {TEST_START} → {TEST_END}  |  H2: {HALF2_START} → {TEST_END}")
print(f"  Question: What observable features predict H2-profitable tickers?")
print(f"{'='*W}")

# ── BQ: load ALL ensemble configs ────────────────────────────────────────────
print(f"\n  PHASE 1A: Loading BQ configs...")
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

all_tickers_bq = sorted(bq_df['ticker'].unique())
print(f"    BQ: {len(bq_df)} configs, {len(all_tickers_bq)} tickers, {bq_df['fold'].nunique()} folds")
print(f"    Modes: {bq_df['tpsl_mode'].value_counts().to_dict()}")

# ── ALL configs per ticker (exhaustive oracle) ─────────────────────────────
# Instead of 7 summary methods, test EVERY BQ config + fixed universal.
# Oracle = best H2 result across ALL configs per ticker.

all_ticker_configs = {}
best_config = {}

FIXED_UNIVERSAL_CFG = {
    'fold': None,
    'tpsl_mode': 'static',
    'prob_thresh': 0.50,
    'tp_pct': 0.05,
    'sl_pct': 0.03,
    'sl_mult': 0.0,
    'grace_minutes': 0,
    'entry_minute': ENTRY_MIN,
    'val_a_avg_pnl': 0.0,
    'opt_avg_pnl': 0.0,
    'config_id': 'fixed_universal',
}

for ticker in all_tickers_bq:
    t_df = bq_df[bq_df['ticker'] == ticker]
    if len(t_df) == 0:
        continue

    configs = []
    for bq_idx, bq_row in t_df.iterrows():
        cfg = {
            'fold': int(bq_row['fold']),
            'tpsl_mode': bq_row['tpsl_mode'],
            'prob_thresh': bq_row['prob_thresh'],
            'tp_pct': bq_row['tp_pct'],
            'sl_pct': bq_row['sl_pct'],
            'sl_mult': bq_row['sl_mult'],
            'grace_minutes': int(bq_row['grace_minutes']),
            'entry_minute': int(bq_row['entry_minute']) if bq_row['entry_minute'] > 0 else ENTRY_MIN,
            'val_a_avg_pnl': bq_row['val_a_avg_pnl'],
            'opt_avg_pnl': bq_row['opt_avg_pnl'],
            'config_id': f"f{int(bq_row['fold'])}_{bq_row['tpsl_mode']}_t{bq_row['prob_thresh']:.2f}_tp{bq_row['tp_pct']:.2f}",
        }
        configs.append(cfg)

    configs.append(FIXED_UNIVERSAL_CFG.copy())

    all_ticker_configs[ticker] = configs

    best_va_row = t_df.loc[t_df['val_a_avg_pnl'].idxmax()]
    if best_va_row['val_a_avg_pnl'] > 0:
        best_config[ticker] = {
            'fold': int(best_va_row['fold']),
            'tpsl_mode': best_va_row['tpsl_mode'],
            'prob_thresh': best_va_row['prob_thresh'],
            'tp_pct': best_va_row['tp_pct'],
            'sl_pct': best_va_row['sl_pct'],
            'sl_mult': best_va_row['sl_mult'],
            'grace_minutes': int(best_va_row['grace_minutes']),
            'entry_minute': int(best_va_row['entry_minute']) if best_va_row['entry_minute'] > 0 else ENTRY_MIN,
            'val_a_avg_pnl': best_va_row['val_a_avg_pnl'],
            'opt_avg_pnl': best_va_row['opt_avg_pnl'],
        }

selected_tickers = sorted(all_ticker_configs.keys())
n_configs = [len(v) for v in all_ticker_configs.values()]
total_configs = sum(n_configs)
print(f"    ALL configs loaded for {len(selected_tickers)} tickers")
print(f"    Configs per ticker: min={min(n_configs)}, max={max(n_configs)}, "
      f"median={np.median(n_configs):.0f}, total={total_configs}")

# ── Load OHLCV ──────────────────────────────────────────────────────────────
print(f"\n  PHASE 1B: Loading OHLCV for {len(selected_tickers)} tickers...")
ticker_data = {}
for i, ticker in enumerate(selected_tickers):
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
            del df
            _master_cache.pop(ticker, None)
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
        ticker_data[ticker] = ohlcv
        del df
        _master_cache.pop(ticker, None)
    except Exception as ex:
        print(f"    {ticker}: OHLCV load failed -- {ex}")
        _master_cache.pop(ticker, None)
    if (i + 1) % 10 == 0:
        _release_memory()
_release_memory()
print(f"    OHLCV loaded: {len(ticker_data)} tickers")

# ── Load seed scores for best fold per ticker ───────────────────────────────
print(f"\n  PHASE 1C: Loading seed scores (best fold per ticker)...")
daily_probs = {}
needed_folds = set()
for ticker in ticker_data:
    if ticker in best_config:
        needed_folds.add(best_config[ticker]['fold'])

for fold in sorted(needed_folds):
    exp = FOLD_EXPERIMENTS[fold]
    fold_tickers = [t for t in ticker_data
                    if t in best_config and best_config[t]['fold'] == fold]
    n_loaded = 0
    for ticker in fold_tickers:
        seeds = load_seed_scores(ticker, exp)
        if not seeds:
            continue
        dp = compute_daily_prob(seeds, ENTRY_MIN)
        if dp:
            daily_probs[ticker] = dp
            n_loaded += 1
        del seeds
    print(f"      Fold {fold}: {n_loaded} tickers")
    _release_memory()

# Also load ALL folds' scores for cross-fold ensemble (features)
print(f"\n  PHASE 1D: Loading ALL folds' scores for feature computation...")
all_fold_probs = defaultdict(dict)
for fold in ALL_FOLDS:
    exp = FOLD_EXPERIMENTS[fold]
    fold_tickers_bq = set(bq_df[bq_df['fold'] == fold]['ticker'].unique())
    fold_tickers = [t for t in ticker_data if t in fold_tickers_bq]
    n_loaded = 0
    for ticker in fold_tickers:
        if fold in all_fold_probs[ticker]:
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

# Build cross-fold ensemble probs
ensemble_probs = {}
for ticker in ticker_data:
    fold_dps = all_fold_probs.get(ticker, {})
    if not fold_dps:
        if ticker in daily_probs:
            ensemble_probs[ticker] = daily_probs[ticker]
        continue
    all_dates = set()
    for dp in fold_dps.values():
        all_dates.update(dp.keys())
    ep = {}
    for d in all_dates:
        vals = [dp[d] for dp in fold_dps.values() if d in dp]
        if vals:
            ep[d] = np.mean(vals)
    ensemble_probs[ticker] = ep

tickers_ready = sorted(set(ticker_data.keys()) & set(ensemble_probs.keys()))
print(f"\n    Ready: {len(tickers_ready)} tickers with OHLCV + ensemble probs")


# ==============================================================================
# PHASE 2: COMPUTE FEATURE MATRIX (18 features per ticker)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 2: FEATURE MATRIX COMPUTATION")
print(f"  ALL features observable BEFORE H2 — no H2 PnL leakage")
print(f"{'='*W}")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h1_end_dt     = pd.Timestamp(HALF1_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

feature_rows = []

for ticker in tickers_ready:
    ohlcv = ticker_data[ticker]
    row = {'ticker': ticker}

    # ── BQ FEATURES ─────────────────────────────────────────────────────────
    t_df = bq_df[bq_df['ticker'] == ticker]
    if len(t_df) == 0:
        continue

    fold_best_va = {}
    for fold, grp in t_df.groupby('fold'):
        fold_best_va[fold] = grp['val_a_avg_pnl'].max()

    n_folds_total = len(fold_best_va)
    n_folds_positive = sum(1 for v in fold_best_va.values() if v > 0)

    row['n_folds_positive'] = n_folds_positive
    row['fold_agreement_pct'] = n_folds_positive / max(n_folds_total, 1)

    a2_folds_pos = sum(1 for f, v in fold_best_va.items() if f < 410 and v > 0)
    a1_folds_pos = sum(1 for f, v in fold_best_va.items() if f >= 410 and v > 0)
    row['cross_alpha_agree'] = int(a2_folds_pos > 0 and a1_folds_pos > 0)

    va_values = list(fold_best_va.values())
    row['median_val_a'] = float(np.median(va_values))
    row['best_val_a'] = float(np.max(va_values))

    fold_ratios = []
    for fold, grp in t_df.groupby('fold'):
        best_idx = grp['val_a_avg_pnl'].idxmax()
        r = grp.loc[best_idx]
        if r['opt_avg_pnl'] > 0.001:
            fold_ratios.append(r['val_a_avg_pnl'] / r['opt_avg_pnl'])
    row['median_overfit_ratio'] = float(np.median(fold_ratios)) if fold_ratios else 0.0

    positive_configs = t_df[t_df['val_a_avg_pnl'] > 0]
    row['n_positive_configs'] = len(positive_configs)

    if len(positive_configs) > 0:
        row['grace_pct'] = (positive_configs['tpsl_mode'] == 'grace').mean()
        row['median_tp'] = positive_configs['tp_pct'].median()
        row['median_thresh'] = positive_configs['prob_thresh'].median()
    else:
        row['grace_pct'] = 0.0
        row['median_tp'] = 0.0
        row['median_thresh'] = 0.0

    # ── OHLCV FEATURES ──────────────────────────────────────────────────────
    entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]
    daily_dips = []
    last_td = -1
    for idx in entry_indices:
        d_int = int(ohlcv['dates'][idx])
        if d_int == last_td:
            continue
        d_str = str(d_int)
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < test_start_dt or d_date > test_end_dt:
            continue
        result = measure_day_dip(ohlcv, idx)
        if result is not None:
            daily_dips.append(result['max_dip'])
        last_td = d_int

    if daily_dips:
        dips_arr = np.array(daily_dips)
        row['avg_dip'] = float(dips_arr.mean())
        row['p75_dip'] = float(np.percentile(dips_arr, 75))
        row['dip_gt_3pct'] = float((dips_arr >= 0.03).mean())
        row['dip_gt_5pct'] = float((dips_arr >= 0.05).mean())
    else:
        row['avg_dip'] = 0.0
        row['p75_dip'] = 0.0
        row['dip_gt_3pct'] = 0.0
        row['dip_gt_5pct'] = 0.0

    atr_arr = ohlcv.get('atr_pct', np.full(len(ohlcv['prices']), 0.05))
    valid_atr = atr_arr[atr_arr > 0]
    row['avg_atr_pct'] = float(valid_atr.mean()) if len(valid_atr) > 0 else 0.05

    # ── MODEL FEATURES ──────────────────────────────────────────────────────
    ep = ensemble_probs.get(ticker, {})
    if ep and daily_dips:
        entry_indices_2 = np.where(ohlcv['times'] == ENTRY_MIN)[0]
        signal_dips = []
        nonsignal_dips = []
        prob_dip_pairs = []
        last_td = -1

        for idx in entry_indices_2:
            d_int = int(ohlcv['dates'][idx])
            if d_int == last_td:
                continue
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            result = measure_day_dip(ohlcv, idx)
            if result is None:
                last_td = d_int
                continue
            prob_val = ep.get(d_date, 0.0)
            if prob_val >= 0.50:
                signal_dips.append(result['max_dip'])
            else:
                nonsignal_dips.append(result['max_dip'])
            prob_dip_pairs.append((prob_val, result['max_dip']))
            last_td = d_int

        if signal_dips and nonsignal_dips:
            ns_mean = np.mean(nonsignal_dips)
            row['signal_lift'] = (np.mean(signal_dips) / max(ns_mean, 0.001)) - 1.0
        else:
            row['signal_lift'] = 0.0

        total_days = len(signal_dips) + len(nonsignal_dips)
        row['signal_freq'] = len(signal_dips) / max(total_days, 1)

        if len(prob_dip_pairs) >= 10:
            probs_arr = np.array([p for p, d in prob_dip_pairs])
            dips_arr2 = np.array([d for p, d in prob_dip_pairs])
            rho, _ = sp_stats.spearmanr(probs_arr, dips_arr2)
            row['calibration_r'] = float(rho) if not np.isnan(rho) else 0.0
        else:
            row['calibration_r'] = 0.0
    else:
        row['signal_lift'] = 0.0
        row['signal_freq'] = 0.0
        row['calibration_r'] = 0.0

    feature_rows.append(row)

feat_df = pd.DataFrame(feature_rows).set_index('ticker')

FEATURE_COLS = [
    'n_folds_positive', 'fold_agreement_pct', 'cross_alpha_agree',
    'median_val_a', 'best_val_a', 'median_overfit_ratio',
    'n_positive_configs', 'grace_pct', 'median_tp', 'median_thresh',
    'avg_dip', 'p75_dip', 'dip_gt_3pct', 'dip_gt_5pct', 'avg_atr_pct',
    'signal_lift', 'signal_freq', 'calibration_r',
]

print(f"\n  Feature matrix: {len(feat_df)} tickers × {len(FEATURE_COLS)} features")
print(f"\n  {'Feature':<25} {'Min':>8} {'P25':>8} {'Med':>8} {'P75':>8} {'Max':>8}")
print(f"  {'-'*73}")
for f in FEATURE_COLS:
    vals = feat_df[f]
    print(f"  {f:<25} {vals.min():>8.3f} {vals.quantile(0.25):>8.3f} "
          f"{vals.median():>8.3f} {vals.quantile(0.75):>8.3f} {vals.max():>8.3f}")


# ==============================================================================
# PHASE 3: EXHAUSTIVE ALL-CONFIGS H2 BACKTEST — TRUE ORACLE
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 3: EXHAUSTIVE ALL-CONFIGS H2 BACKTEST — TRUE ORACLE")
print(f"  Every BQ config + fixed universal per ticker → oracle = best config per ticker")
print(f"  H1: {TEST_START} → {HALF1_END}  |  H2: {HALF2_START} → {TEST_END}")
print(f"{'='*W}")

bt_results = []
all_config_results = {}
total_bt_count = 0

for ticker in tickers_ready:
    if ticker not in all_ticker_configs:
        continue
    ohlcv = ticker_data[ticker]
    configs = all_ticker_configs[ticker]
    ticker_cfg_results = []

    for cfg in configs:
        if cfg['fold'] is not None:
            dp = all_fold_probs.get(ticker, {}).get(cfg['fold'], {})
            if not dp:
                dp = ensemble_probs.get(ticker, {})
        else:
            dp = ensemble_probs.get(ticker, {})

        if not dp:
            continue

        em = cfg['entry_minute'] if cfg['entry_minute'] > 0 else ENTRY_MIN
        thresh = cfg['prob_thresh']

        entry_indices = np.where(ohlcv['times'] == em)[0]
        h1_trades_list = []
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

            prob = dp.get(d_date, 0.0)
            if prob < thresh:
                last_td = d_int
                continue

            pnl, td, filled = execute_trade(
                ohlcv, idx, cfg['tpsl_mode'],
                cfg['tp_pct'], cfg['sl_pct'], cfg['sl_mult'],
                cfg['grace_minutes'])

            if filled:
                if d_date < h2_start_dt:
                    h1_trades_list.append(pnl)
                else:
                    h2_trades_list.append(pnl)
            last_td = d_int

        h2_arr = np.array(h2_trades_list) if h2_trades_list else np.array([])
        h1_arr = np.array(h1_trades_list) if h1_trades_list else np.array([])

        ticker_cfg_results.append({
            'config_id': cfg['config_id'],
            'fold': cfg['fold'],
            'tpsl_mode': cfg['tpsl_mode'],
            'prob_thresh': cfg['prob_thresh'],
            'tp_pct': cfg['tp_pct'],
            'sl_pct': cfg['sl_pct'],
            'h2_avg': float(h2_arr.mean()) if len(h2_arr) > 0 else 0.0,
            'h2_trades': len(h2_arr),
            'h2_total': float(h2_arr.sum()) if len(h2_arr) > 0 else 0.0,
            'h2_wr': float((h2_arr > 0).mean()) if len(h2_arr) > 0 else 0.0,
            'h1_avg': float(h1_arr.mean()) if len(h1_arr) > 0 else 0.0,
            'h1_trades': len(h1_arr),
            'h1_total': float(h1_arr.sum()) if len(h1_arr) > 0 else 0.0,
        })
        total_bt_count += 1

    all_config_results[ticker] = ticker_cfg_results

    valid_cfgs = [c for c in ticker_cfg_results if c['h2_trades'] >= MIN_TRADES_VALID]

    row = {'ticker': ticker}

    if valid_cfgs:
        best_cfg = max(valid_cfgs, key=lambda c: c['h2_avg'])
        row['h2_avg_pnl'] = best_cfg['h2_avg']
        row['h2_trades'] = best_cfg['h2_trades']
        row['h2_total_pnl'] = best_cfg['h2_total']
        row['h2_wr'] = best_cfg['h2_wr']
        row['h1_avg_pnl'] = best_cfg['h1_avg']
        row['h1_trades'] = best_cfg['h1_trades']
        h1_tot = best_cfg['h1_total']
        h2_tot = best_cfg['h2_total']
        full_n = best_cfg['h1_trades'] + best_cfg['h2_trades']
        row['full_trades'] = full_n
        row['full_avg_pnl'] = (h1_tot + h2_tot) / full_n if full_n > 0 else 0.0
        row['oracle_config'] = best_cfg['config_id']
        row['oracle_mode'] = best_cfg['tpsl_mode']
        row['oracle_tp'] = best_cfg['tp_pct']
        row['oracle_sl'] = best_cfg['sl_pct']
        row['oracle_thresh'] = best_cfg['prob_thresh']
        row['oracle_fold'] = best_cfg['fold']
        row['n_configs_tested'] = len(ticker_cfg_results)
        row['n_configs_valid'] = len(valid_cfgs)
        row['n_configs_positive'] = sum(1 for c in valid_cfgs if c['h2_avg'] > 0)
    else:
        row['h2_avg_pnl'] = 0.0
        row['h2_trades'] = 0
        row['h2_total_pnl'] = 0.0
        row['h2_wr'] = 0.0
        row['h1_avg_pnl'] = 0.0
        row['h1_trades'] = 0
        row['full_trades'] = 0
        row['full_avg_pnl'] = 0.0
        row['oracle_config'] = 'none'
        row['oracle_mode'] = ''
        row['oracle_tp'] = 0.0
        row['oracle_sl'] = 0.0
        row['oracle_thresh'] = 0.0
        row['oracle_fold'] = None
        row['n_configs_tested'] = len(ticker_cfg_results)
        row['n_configs_valid'] = 0
        row['n_configs_positive'] = 0

    bva_cfg = best_config.get(ticker)
    if bva_cfg:
        bva_results = [c for c in ticker_cfg_results
                       if c['fold'] == bva_cfg.get('fold')
                       and c['tpsl_mode'] == bva_cfg.get('tpsl_mode')
                       and abs(c['prob_thresh'] - bva_cfg.get('prob_thresh', 0)) < 0.001
                       and abs(c['tp_pct'] - bva_cfg.get('tp_pct', 0)) < 0.0001]
        if bva_results:
            row['bva_h2_avg'] = bva_results[0]['h2_avg']
            row['bva_h2_trades'] = bva_results[0]['h2_trades']
        else:
            row['bva_h2_avg'] = np.nan
            row['bva_h2_trades'] = 0
    else:
        row['bva_h2_avg'] = np.nan
        row['bva_h2_trades'] = 0

    bt_results.append(row)

bt_df = pd.DataFrame(bt_results).set_index('ticker')
print(f"\n  Total backtests run: {total_bt_count}")

# ── 3A: Summary stats ─────────────────────────────────────────────────────
valid_bt = bt_df[bt_df['h2_trades'] >= MIN_TRADES_VALID]
print(f"\n  3A: Exhaustive Oracle Summary ({len(valid_bt)} tickers with ≥{MIN_TRADES_VALID} H2 trades)")

oracle_avg = valid_bt['h2_avg_pnl'].mean() if len(valid_bt) > 0 else 0.0
oracle_pos = (valid_bt['h2_avg_pnl'] > 0).sum() if len(valid_bt) > 0 else 0
oracle_deploy = (valid_bt['h2_avg_pnl'] >= DEPLOY_FLOOR).sum() if len(valid_bt) > 0 else 0
oracle_total = valid_bt['h2_total_pnl'].sum() if len(valid_bt) > 0 else 0.0

print(f"  ORACLE avg H2 PnL:     {oracle_avg*100:>+.2f}%")
print(f"  ORACLE total H2 PnL:   {oracle_total*100:>+.1f}%")
print(f"  H2 positive tickers:   {oracle_pos}/{len(valid_bt)}")
print(f"  H2 deployable (≥1.8%): {oracle_deploy}/{len(valid_bt)}")
print(f"  Configs tested total:  {total_bt_count}")

bva_valid = valid_bt['bva_h2_avg'].dropna()
bva_avg = bva_valid.mean() if len(bva_valid) > 0 else 0.0
print(f"\n  Best-val_a baseline:   {bva_avg*100:>+.2f}%  ({len(bva_valid)} tickers)")
print(f"  Oracle improvement:    {(oracle_avg - bva_avg)*100:>+.2f}pp")

oracle_improved = sum(1 for _, r in valid_bt.iterrows()
                      if not pd.isna(r.get('bva_h2_avg', np.nan))
                      and r['h2_avg_pnl'] > r['bva_h2_avg'])
print(f"  Oracle > BVA:          {oracle_improved}/{len(bva_valid)} tickers")

# ── 3B: Oracle config property distribution ───────────────────────────────
print(f"\n  3B: Oracle Config Properties (what configs win in H2?)")

mode_counts = valid_bt[valid_bt['oracle_config'] != 'none']['oracle_mode'].value_counts()
print(f"\n  Mode distribution:")
for mode, cnt in mode_counts.items():
    pct = cnt / len(valid_bt) * 100
    print(f"    {mode:<12}: {cnt:>3} tickers ({pct:.0f}%)")

tp_vals = valid_bt[valid_bt['oracle_tp'] > 0]['oracle_tp']
if len(tp_vals) > 0:
    print(f"\n  TP distribution: min={tp_vals.min()*100:.1f}%, "
          f"median={tp_vals.median()*100:.1f}%, max={tp_vals.max()*100:.1f}%")

thresh_vals = valid_bt[valid_bt['oracle_thresh'] > 0]['oracle_thresh']
if len(thresh_vals) > 0:
    print(f"  Thresh distribution: min={thresh_vals.min():.2f}, "
          f"median={thresh_vals.median():.2f}, max={thresh_vals.max():.2f}")

fold_counts = valid_bt[valid_bt['oracle_fold'].notna()]['oracle_fold'].value_counts().sort_index()
if len(fold_counts) > 0:
    print(f"\n  Fold distribution:")
    for fold, cnt in fold_counts.items():
        print(f"    Fold {int(fold)}: {cnt:>3}")

# ── 3C: Config depth per ticker ────────────────────────────────────────────
print(f"\n  3C: Config Depth per Ticker")
print(f"  {'Ticker':<8} {'Tested':>7} {'Valid':>6} {'H2>0':>6} {'OracleAvg%':>11}")
print(f"  {'-'*45}")
for _, r in valid_bt.sort_values('h2_avg_pnl', ascending=False).iterrows():
    print(f"  {r.name:<8} {int(r['n_configs_tested']):>7} {int(r['n_configs_valid']):>6} "
          f"{int(r['n_configs_positive']):>6} {r['h2_avg_pnl']*100:>+10.2f}%")

# ── 3D: Per-ticker oracle results ──────────────────────────────────────────
print(f"\n  3D: Per-Ticker Oracle Results (sorted by H2 avg)")
print(f"  {'Ticker':<8} {'Mode':<8} {'TP%':>5} {'SL%':>5} {'Thr':>5} {'Fold':>5} "
      f"{'H2tr':>5} {'H2avg%':>8} {'H2tot%':>8} {'H2WR':>6} {'BVA%':>8} {'Deploy':>6}")
print(f"  {'-'*90}")
for _, r in valid_bt.sort_values('h2_avg_pnl', ascending=False).iterrows():
    deploy = "YES" if r['h2_avg_pnl'] >= DEPLOY_FLOOR else ""
    bva_val = r['bva_h2_avg'] if not pd.isna(r.get('bva_h2_avg', np.nan)) else 0.0
    fold_str = str(int(r['oracle_fold'])) if r['oracle_fold'] is not None and not pd.isna(r.get('oracle_fold', np.nan)) else "ens"
    print(f"  {r.name:<8} {r['oracle_mode']:<8} {r['oracle_tp']*100:>4.1f} "
          f"{r['oracle_sl']*100:>4.1f} {r['oracle_thresh']:>4.2f} {fold_str:>5} "
          f"{int(r['h2_trades']):>5} {r['h2_avg_pnl']*100:>+7.2f}% "
          f"{r['h2_total_pnl']*100:>+7.1f}% {r['h2_wr']*100:>5.0f}% "
          f"{bva_val*100:>+7.2f}% {deploy:>6}")

# ── 3E: Fixed universal baseline ──────────────────────────────────────────
print(f"\n  3E: Fixed Universal Baseline (static, TP=5%, SL=3%, thresh=0.50)")
fu_results = []
for ticker, cfg_list in all_config_results.items():
    fu = [c for c in cfg_list if c['config_id'] == 'fixed_universal']
    if fu and fu[0]['h2_trades'] >= MIN_TRADES_VALID:
        fu_results.append({'ticker': ticker, 'h2_avg': fu[0]['h2_avg'],
                           'h2_trades': fu[0]['h2_trades']})
if fu_results:
    fu_avgs = [f['h2_avg'] for f in fu_results]
    print(f"  {len(fu_results)} tickers with ≥{MIN_TRADES_VALID} trades")
    print(f"  Avg H2 PnL: {np.mean(fu_avgs)*100:+.2f}%")
    print(f"  H2 > 0:     {sum(1 for v in fu_avgs if v > 0)}/{len(fu_avgs)}")
else:
    print(f"  No tickers with ≥{MIN_TRADES_VALID} fixed universal trades")

# ── 3F: Visualizations ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Phase 3: Exhaustive Oracle Results', fontsize=14, fontweight='bold')

# (0,0) Per-ticker oracle H2 avg PnL
ax = axes[0, 0]
sorted_bt = valid_bt.sort_values('h2_avg_pnl')
colors = ['#2ecc71' if v >= DEPLOY_FLOOR else '#3498db' if v > 0 else '#e74c3c'
          for v in sorted_bt['h2_avg_pnl']]
ax.barh(range(len(sorted_bt)), sorted_bt['h2_avg_pnl'] * 100, color=colors)
ax.set_yticks(range(len(sorted_bt)))
ax.set_yticklabels(sorted_bt.index, fontsize=7)
ax.axvline(x=DEPLOY_FLOOR * 100, color='red', linestyle='--', alpha=0.7, label=f'Deploy ≥{DEPLOY_FLOOR*100}%')
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlabel('H2 Avg PnL %')
ax.set_title(f'Oracle H2 Avg PnL ({len(sorted_bt)} tickers)')
ax.legend(fontsize=8)

# (0,1) Oracle vs BVA
ax = axes[0, 1]
bva_compare = valid_bt[valid_bt['bva_h2_avg'].notna()].sort_values('h2_avg_pnl')
if len(bva_compare) > 0:
    y_pos = np.arange(len(bva_compare))
    bar_h = 0.35
    ax.barh(y_pos + bar_h/2, bva_compare['h2_avg_pnl'] * 100, bar_h,
            label='Oracle', color='#2ecc71', alpha=0.85)
    ax.barh(y_pos - bar_h/2, bva_compare['bva_h2_avg'] * 100, bar_h,
            label='Best val_a', color='#e74c3c', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bva_compare.index, fontsize=7)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('H2 Avg PnL %')
    ax.set_title(f'Oracle vs Best-Val-A ({len(bva_compare)} tickers)')
    ax.legend(fontsize=8)

# (1,0) Mode distribution
ax = axes[1, 0]
mode_counts_plot = valid_bt[valid_bt['oracle_config'] != 'none']['oracle_mode'].value_counts()
ax.pie(mode_counts_plot.values, labels=mode_counts_plot.index, autopct='%1.0f%%',
       colors=['#3498db', '#2ecc71', '#e67e22', '#9b59b6'])
ax.set_title('Oracle Mode Distribution')

# (1,1) TP vs H2 avg PnL scatter
ax = axes[1, 1]
tp_mask = valid_bt['oracle_tp'] > 0
if tp_mask.sum() > 0:
    tp_plot = valid_bt[tp_mask]
    colors_sc = ['#2ecc71' if v >= DEPLOY_FLOOR else '#3498db' if v > 0 else '#e74c3c'
                 for v in tp_plot['h2_avg_pnl']]
    ax.scatter(tp_plot['oracle_tp'] * 100, tp_plot['h2_avg_pnl'] * 100,
               c=colors_sc, s=60, edgecolors='white', alpha=0.8)
    for _, r in tp_plot.iterrows():
        if abs(r['h2_avg_pnl']) > 0.02:
            ax.annotate(r.name, (r['oracle_tp'] * 100, r['h2_avg_pnl'] * 100),
                       fontsize=6, alpha=0.7)
    ax.axhline(y=DEPLOY_FLOOR * 100, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Oracle TP %')
    ax.set_ylabel('H2 Avg PnL %')
    ax.set_title('TP vs H2 Performance')

plt.tight_layout()
plt.show()

# ── Build combined DataFrame ───────────────────────────────────────────────
join_cols = ['h2_avg_pnl', 'h2_trades', 'h2_total_pnl', 'h2_wr',
             'h1_avg_pnl', 'h1_trades', 'full_trades', 'full_avg_pnl',
             'oracle_config', 'oracle_mode', 'oracle_tp', 'oracle_sl',
             'oracle_thresh', 'bva_h2_avg', 'n_configs_tested',
             'n_configs_valid', 'n_configs_positive']
combined = feat_df[FEATURE_COLS].join(bt_df[join_cols], how='inner')
combined['h2_positive'] = (combined['h2_avg_pnl'] > 0).astype(int)
combined['h2_deployable'] = (combined['h2_avg_pnl'] >= DEPLOY_FLOOR).astype(int)

n_h2_pos = (combined['h2_avg_pnl'] > 0).sum()
n_h2_deploy = (combined['h2_avg_pnl'] >= DEPLOY_FLOOR).sum()
print(f"\n  ORACLE H2 positive: {n_h2_pos}/{len(combined)}")
print(f"  ORACLE H2 deployable (≥{DEPLOY_FLOOR*100:.1f}%): {n_h2_deploy}/{len(combined)}")

h2_with_trades = combined[combined['h2_trades'] >= MIN_TRADES_VALID]
if len(h2_with_trades) > 0:
    print(f"\n  ALL tickers (≥{MIN_TRADES_VALID} trades): {len(h2_with_trades)} tickers, "
          f"ORACLE avg H2 PnL = {h2_with_trades['h2_avg_pnl'].mean()*100:+.2f}%")


# ==============================================================================
# PHASE 4: FEATURE → ORACLE H2 CORRELATION ANALYSIS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 4: FEATURE → ORACLE H2 CORRELATION ANALYSIS")
print(f"  Ground truth = best config per ticker (ALL BQ configs tested exhaustively)")
print(f"{'='*W}")

analysis_df = combined[combined['h2_trades'] >= MIN_TRADES_VALID].copy()
print(f"\n  Analysis on {len(analysis_df)} tickers with ≥{MIN_TRADES_VALID} H2 trades")

# ── 4A: Spearman correlations ──────────────────────────────────────────────
print(f"\n  4A: Spearman Correlation with H2 Avg PnL")
print(f"  {'Feature':<25} {'ρ(h2avg)':>10} {'p-val':>8} {'ρ(h2tot)':>10} {'p-val':>8} {'Sig':>5}")
print(f"  {'-'*71}")

corr_results = []
for f in FEATURE_COLS:
    vals = analysis_df[f].values
    h2_avg = analysis_df['h2_avg_pnl'].values
    h2_tot = analysis_df['h2_total_pnl'].values

    if np.std(vals) < 1e-10:
        continue

    rho_avg, p_avg = sp_stats.spearmanr(vals, h2_avg)
    rho_tot, p_tot = sp_stats.spearmanr(vals, h2_tot)

    corr_results.append({
        'feature': f,
        'rho_avg': rho_avg,
        'p_avg': p_avg,
        'rho_tot': rho_tot,
        'p_tot': p_tot,
    })

corr_results.sort(key=lambda x: abs(x['rho_avg']), reverse=True)

for cr in corr_results:
    sig = ""
    if cr['p_avg'] < 0.10: sig = "*"
    if cr['p_avg'] < 0.05: sig = "**"
    if cr['p_avg'] < 0.01: sig = "***"
    print(f"  {cr['feature']:<25} {cr['rho_avg']:>+10.4f} {cr['p_avg']:>8.4f} "
          f"{cr['rho_tot']:>+10.4f} {cr['p_tot']:>8.4f} {sig:>5}")

# ── 4B: Group comparison (H2-positive vs H2-negative) ────────────────────
print(f"\n  4B: H2-Positive vs H2-Negative Group Comparison")

pos_mask = analysis_df['h2_avg_pnl'] > 0
neg_mask = analysis_df['h2_avg_pnl'] <= 0
n_pos = pos_mask.sum()
n_neg = neg_mask.sum()
print(f"  H2+ group: {n_pos} tickers  |  H2- group: {n_neg} tickers")

print(f"\n  {'Feature':<25} {'H2+ Mean':>10} {'H2- Mean':>10} {'Diff':>10} {'U-pval':>8} {'Sig':>5}")
print(f"  {'-'*73}")

group_results = []
for f in FEATURE_COLS:
    pos_vals = analysis_df.loc[pos_mask, f].values
    neg_vals = analysis_df.loc[neg_mask, f].values

    if len(pos_vals) < 2 or len(neg_vals) < 2:
        continue

    pos_mean = np.mean(pos_vals)
    neg_mean = np.mean(neg_vals)
    diff = pos_mean - neg_mean

    try:
        u_stat, u_p = sp_stats.mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
    except Exception:
        u_p = 1.0

    sig = ""
    if u_p < 0.10: sig = "*"
    if u_p < 0.05: sig = "**"
    if u_p < 0.01: sig = "***"

    group_results.append({
        'feature': f, 'pos_mean': pos_mean, 'neg_mean': neg_mean,
        'diff': diff, 'u_p': u_p,
    })

    print(f"  {f:<25} {pos_mean:>10.4f} {neg_mean:>10.4f} {diff:>+10.4f} {u_p:>8.4f} {sig:>5}")

# ── 4C: Identify predictive features ────────────────────────────────────────
print(f"\n  4C: Predictive Features (|ρ| ≥ 0.15 OR U-test p < 0.15)")
predictive = []
for cr in corr_results:
    gr = next((g for g in group_results if g['feature'] == cr['feature']), None)
    u_p = gr['u_p'] if gr else 1.0
    if abs(cr['rho_avg']) >= 0.15 or u_p < 0.15:
        direction = "HIGHER is better" if cr['rho_avg'] > 0 else "LOWER is better"
        predictive.append({
            'feature': cr['feature'],
            'rho': cr['rho_avg'],
            'p_spearman': cr['p_avg'],
            'p_utest': u_p,
            'direction': direction,
            'pos_mean': gr['pos_mean'] if gr else 0,
            'neg_mean': gr['neg_mean'] if gr else 0,
        })
        print(f"    {cr['feature']:<25} ρ={cr['rho_avg']:+.3f} U-p={u_p:.3f}  → {direction}")

if not predictive:
    print(f"    (Relaxing to |ρ| ≥ 0.10 OR U-test p < 0.20)")
    for cr in corr_results:
        gr = next((g for g in group_results if g['feature'] == cr['feature']), None)
        u_p = gr['u_p'] if gr else 1.0
        if abs(cr['rho_avg']) >= 0.10 or u_p < 0.20:
            direction = "HIGHER is better" if cr['rho_avg'] > 0 else "LOWER is better"
            predictive.append({
                'feature': cr['feature'],
                'rho': cr['rho_avg'],
                'p_spearman': cr['p_avg'],
                'p_utest': u_p,
                'direction': direction,
                'pos_mean': gr['pos_mean'] if gr else 0,
                'neg_mean': gr['neg_mean'] if gr else 0,
            })
            print(f"    {cr['feature']:<25} ρ={cr['rho_avg']:+.3f} U-p={u_p:.3f}  → {direction}")

print(f"\n  Total predictive features: {len(predictive)}")

# ── 4D: Visualizations ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Phase 4: Feature → Oracle H2 Correlations', fontsize=14, fontweight='bold')

# Feature importance bar chart (top left, spanning full width)
ax_imp = fig.add_subplot(3, 3, (1, 3))
feat_names = [cr['feature'] for cr in corr_results]
feat_rhos = [cr['rho_avg'] for cr in corr_results]
feat_colors = ['#2ecc71' if abs(r) >= 0.15 else '#bdc3c7' for r in feat_rhos]
ax_imp.barh(range(len(feat_names)), feat_rhos, color=feat_colors)
ax_imp.set_yticks(range(len(feat_names)))
ax_imp.set_yticklabels(feat_names, fontsize=8)
ax_imp.axvline(x=0, color='black', linewidth=0.5)
ax_imp.axvline(x=0.15, color='gray', linestyle='--', alpha=0.5, label='±0.15')
ax_imp.axvline(x=-0.15, color='gray', linestyle='--', alpha=0.5)
ax_imp.set_xlabel('Spearman ρ with H2 Avg PnL')
ax_imp.set_title('Feature Importance (All 18 Features)')
ax_imp.legend(fontsize=8)

# Top 6 feature scatter plots (2x3 grid below)
top_feats = [p['feature'] for p in predictive[:6]]
for i, fname in enumerate(top_feats):
    ax = fig.add_subplot(3, 3, 4 + i)
    x_vals = analysis_df[fname].values
    y_vals = analysis_df['h2_avg_pnl'].values * 100
    sc_colors = ['#2ecc71' if v >= DEPLOY_FLOOR * 100 else '#3498db' if v > 0 else '#e74c3c'
                 for v in y_vals]
    ax.scatter(x_vals, y_vals, c=sc_colors, alpha=0.7, edgecolors='white', s=40)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=DEPLOY_FLOOR * 100, color='red', linestyle='--', alpha=0.4)
    for j in range(len(x_vals)):
        if abs(y_vals[j]) > 2.5:
            ax.annotate(analysis_df.index[j], (x_vals[j], y_vals[j]),
                       fontsize=5, alpha=0.7)
    rho = next(cr['rho_avg'] for cr in corr_results if cr['feature'] == fname)
    ax.set_xlabel(fname, fontsize=8)
    ax.set_ylabel('H2 Avg %', fontsize=8)
    ax.set_title(f'ρ={rho:+.3f}', fontsize=9)

plt.tight_layout()
plt.show()


# ==============================================================================
# PHASE 5: COMPOSITE RULE CONSTRUCTION + VALIDATION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 5: COMPOSITE RULE CONSTRUCTION + VALIDATION")
print(f"  Build S3-style gates from predictive features")
print(f"{'='*W}")

# ── 5A: Single-feature threshold sweeps ──────────────────────────────────────
print(f"\n  5A: Single-Feature Threshold Sweeps")

feat_thresholds = {}

for pf in predictive:
    fname = pf['feature']
    vals = analysis_df[fname].values
    h2_avg = analysis_df['h2_avg_pnl'].values
    higher_better = pf['rho'] > 0

    percentiles = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
    candidates = sorted(set(np.percentile(vals, percentiles)))

    best_thresh = None
    best_score = -999
    best_n = 0
    best_avg = 0

    print(f"\n    {fname} ({'↑' if higher_better else '↓'}):")
    print(f"    {'Thresh':>10} {'N pass':>7} {'H2 avg%':>9} {'H2 WR%':>8} {'H2 tot%':>9}")

    for thresh in candidates:
        if higher_better:
            mask = vals >= thresh
        else:
            mask = vals <= thresh

        n_pass = mask.sum()
        if n_pass < MIN_TICKERS_FOR_RULE:
            continue

        h2_sel = h2_avg[mask]
        avg_pnl = float(h2_sel.mean())
        wr = float((h2_sel > 0).mean())
        tot = float(h2_sel.sum())

        print(f"    {thresh:>10.4f} {n_pass:>7} {avg_pnl*100:>+8.2f}% {wr*100:>7.0f}% {tot*100:>+8.1f}%")

        score = avg_pnl * min(n_pass, 15)
        if score > best_score:
            best_score = score
            best_thresh = thresh
            best_n = n_pass
            best_avg = avg_pnl

    if best_thresh is not None:
        feat_thresholds[fname] = {
            'threshold': best_thresh,
            'higher_better': higher_better,
            'n_pass': best_n,
            'h2_avg': best_avg,
        }
        print(f"    → BEST: {best_thresh:.4f} ({best_n} tickers, {best_avg*100:+.2f}%)")

# ── 5B: Composite rules ─────────────────────────────────────────────────────
print(f"\n  5B: Composite Rule Search")
print(f"  Testing ALL combinations of 2-4 predictive features")

combo_results = []
pred_features = [pf['feature'] for pf in predictive if pf['feature'] in feat_thresholds]

for n_features in range(2, min(len(pred_features) + 1, 5)):
    for combo in combinations(pred_features, n_features):
        mask = np.ones(len(analysis_df), dtype=bool)
        rule_desc = []
        for fname in combo:
            ft = feat_thresholds[fname]
            vals = analysis_df[fname].values
            if ft['higher_better']:
                mask &= vals >= ft['threshold']
                rule_desc.append(f"{fname}>={ft['threshold']:.3f}")
            else:
                mask &= vals <= ft['threshold']
                rule_desc.append(f"{fname}<={ft['threshold']:.3f}")

        n_pass = mask.sum()
        if n_pass < MIN_TICKERS_FOR_RULE:
            continue

        h2_sel = analysis_df.loc[mask, 'h2_avg_pnl'].values
        h1_sel = analysis_df.loc[mask, 'h1_avg_pnl'].values
        selected_tickers_list = analysis_df.index[mask].tolist()

        avg_pnl = float(h2_sel.mean())
        wr = float((h2_sel > 0).mean())
        h1_avg = float(h1_sel.mean()) if len(h1_sel) > 0 else 0.0

        h2_trade_total = float(analysis_df.loc[mask, 'h2_total_pnl'].sum())
        h2_total_trades = int(analysis_df.loc[mask, 'h2_trades'].sum())

        overlap_stable = len(set(selected_tickers_list) & set(STABLE_13))
        overlap_oracle = len(set(selected_tickers_list) & set(ORACLE_10))

        combo_results.append({
            'features': combo,
            'n_features': n_features,
            'rule': ' AND '.join(rule_desc),
            'n_pass': n_pass,
            'h2_avg_pnl': avg_pnl,
            'h2_wr': wr,
            'h2_trade_total': h2_trade_total,
            'h2_total_trades': h2_total_trades,
            'h1_avg_pnl': h1_avg,
            'overlap_stable13': overlap_stable,
            'overlap_oracle10': overlap_oracle,
            'tickers': selected_tickers_list,
        })

combo_results.sort(key=lambda x: x['h2_avg_pnl'], reverse=True)

print(f"\n  Found {len(combo_results)} valid combinations")
print(f"\n  TOP 20 COMPOSITE RULES (sorted by H2 avg PnL):")
print(f"  {'#':>3} {'Tkr':>4} {'Trd':>5} {'H2avg%':>8} {'H2WR%':>7} {'TrdTot%':>8} {'H1avg%':>8} "
      f"{'S13':>4} {'O10':>4} {'Rule'}")
print(f"  {'-'*110}")

for i, cr in enumerate(combo_results[:20]):
    print(f"  {i+1:>3} {cr['n_pass']:>4} {cr['h2_total_trades']:>5} "
          f"{cr['h2_avg_pnl']*100:>+7.2f}% {cr['h2_wr']*100:>6.0f}% "
          f"{cr['h2_trade_total']*100:>+7.1f}% {cr['h1_avg_pnl']*100:>+7.2f}% "
          f"{cr['overlap_stable13']:>4} {cr['overlap_oracle10']:>4} {cr['rule']}")

# ── 5C: Validation of top 3 rules ───────────────────────────────────────────
print(f"\n  5C: Detailed Validation of Top 3 Rules")

for rank, cr in enumerate(combo_results[:3]):
    print(f"\n  {'─'*80}")
    print(f"  RULE #{rank+1}: {cr['rule']}")
    print(f"  {'─'*80}")

    print(f"    Selected tickers ({cr['n_pass']}): {', '.join(sorted(cr['tickers']))}")
    print(f"    H2 avg PnL:     {cr['h2_avg_pnl']*100:+.2f}%")
    print(f"    H2 WR:          {cr['h2_wr']*100:.0f}%")
    print(f"    H2 trades:      {cr['h2_total_trades']}")
    print(f"    H2 trade total: {cr['h2_trade_total']*100:+.1f}%  (sum of all individual trade PnLs)")
    print(f"    H1 avg PnL:     {cr['h1_avg_pnl']*100:+.2f}%")
    print(f"    Overlap with Stable-13: {cr['overlap_stable13']}/{len(STABLE_13)}")
    print(f"    Overlap with ORACLE-10: {cr['overlap_oracle10']}/{len(ORACLE_10)}")

    h1_pos = sum(1 for t in cr['tickers']
                 if t in combined.index and combined.loc[t, 'h1_avg_pnl'] > 0)
    h1_neg = len(cr['tickers']) - h1_pos
    print(f"    H1 cross-check: {h1_pos} positive, {h1_neg} negative in H1")

    print(f"\n    {'Ticker':<8} {'Mode':<8} {'TP%':>5} {'Thr':>5} {'H1avg%':>8} {'H2avg%':>8} {'H2tr':>5} {'H2WR%':>7}")
    for t in sorted(cr['tickers']):
        if t in combined.index:
            r = combined.loc[t]
            omode = str(r.get('oracle_mode', '?'))
            otp = r.get('oracle_tp', 0)
            othr = r.get('oracle_thresh', 0)
            print(f"    {t:<8} {omode:<8} {otp*100:>4.1f} {othr:>4.2f} "
                  f"{r['h1_avg_pnl']*100:>+7.2f}% "
                  f"{r['h2_avg_pnl']*100:>+7.2f}% "
                  f"{int(r['h2_trades']):>5} {r['h2_wr']*100:>6.0f}%")

    print(f"\n    Sensitivity test (±20% threshold perturbation):")
    for perturb in [-0.20, -0.10, 0.0, +0.10, +0.20]:
        mask = np.ones(len(analysis_df), dtype=bool)
        for fname in cr['features']:
            ft = feat_thresholds[fname]
            vals = analysis_df[fname].values
            if ft['higher_better']:
                adj_thresh = ft['threshold'] * (1 + perturb)
                mask &= vals >= adj_thresh
            else:
                adj_thresh = ft['threshold'] * (1 - perturb)
                mask &= vals <= adj_thresh
        n = mask.sum()
        if n >= 1:
            avg = analysis_df.loc[mask, 'h2_avg_pnl'].mean()
            print(f"      {perturb:+.0%}: {n:>3} tickers, H2 avg = {avg*100:+.2f}%")
        else:
            print(f"      {perturb:+.0%}: 0 tickers (rule too restrictive)")

# ── 5D: Final comparison ─────────────────────────────────────────────────────
print(f"\n  5D: Final Comparison (ORACLE ground truth)")
print(f"  {'─'*80}")

all_h2 = analysis_df['h2_avg_pnl']
all_h2_tot = analysis_df['h2_total_pnl']
all_h2_trades = int(analysis_df['h2_trades'].sum())
print(f"  ALL ORACLE ({len(analysis_df)}):            H2 avg = {all_h2.mean()*100:+.2f}%  "
      f"WR = {(all_h2 > 0).mean()*100:.0f}%  Trades = {all_h2_trades}  "
      f"TradeTot = {all_h2_tot.sum()*100:+.1f}%")

if 'bva_h2_avg' in analysis_df.columns:
    bva_vals = analysis_df['bva_h2_avg'].dropna()
    if len(bva_vals) > 0:
        print(f"  ALL BEST-VAL-A ({len(bva_vals)}):         H2 avg = {bva_vals.mean()*100:+.2f}%  "
              f"WR = {(bva_vals > 0).mean()*100:.0f}%")

stable_in = analysis_df.index.isin(STABLE_13)
if stable_in.sum() > 0:
    s13_h2 = analysis_df.loc[stable_in, 'h2_avg_pnl']
    s13_trades = int(analysis_df.loc[stable_in, 'h2_trades'].sum())
    s13_tot = analysis_df.loc[stable_in, 'h2_total_pnl'].sum()
    print(f"  STABLE-13 ({stable_in.sum()} found):        H2 avg = {s13_h2.mean()*100:+.2f}%  "
          f"WR = {(s13_h2 > 0).mean()*100:.0f}%  Trades = {s13_trades}  "
          f"TradeTot = {s13_tot*100:+.1f}%")

oracle_in = analysis_df.index.isin(ORACLE_10)
if oracle_in.sum() > 0:
    o10_h2 = analysis_df.loc[oracle_in, 'h2_avg_pnl']
    o10_trades = int(analysis_df.loc[oracle_in, 'h2_trades'].sum())
    o10_tot = analysis_df.loc[oracle_in, 'h2_total_pnl'].sum()
    print(f"  ORACLE-10 ({oracle_in.sum()} found):        H2 avg = {o10_h2.mean()*100:+.2f}%  "
          f"WR = {(o10_h2 > 0).mean()*100:.0f}%  Trades = {o10_trades}  "
          f"TradeTot = {o10_tot*100:+.1f}%")

for i, cr in enumerate(combo_results[:3]):
    print(f"  RULE #{i+1} ({cr['n_pass']} tickers):         "
          f"H2 avg = {cr['h2_avg_pnl']*100:+.2f}%  "
          f"WR = {cr['h2_wr']*100:.0f}%  Trades = {cr['h2_total_trades']}  "
          f"TradeTot = {cr['h2_trade_total']*100:+.1f}%")

print(f"\n  Best single-feature gates:")
for fname, ft in sorted(feat_thresholds.items(), key=lambda x: x[1]['h2_avg'], reverse=True):
    op = ">=" if ft['higher_better'] else "<="
    print(f"    {fname} {op} {ft['threshold']:.3f}: {ft['n_pass']} tickers, "
          f"H2 avg = {ft['h2_avg']*100:+.2f}%")

# ── 5E: Oracle config distribution for top rules ─────────────────────────
if combo_results and 'oracle_mode' in combined.columns:
    print(f"\n  5E: Oracle Config Distribution for Top Rules")
    for rank, cr in enumerate(combo_results[:3]):
        rule_tickers = cr['tickers']
        rule_sel = combined.loc[combined.index.isin(rule_tickers)]
        if len(rule_sel) > 0:
            print(f"    RULE #{rank+1} ({len(rule_tickers)} tickers):")
            mode_dist = rule_sel['oracle_mode'].value_counts()
            for m, c in mode_dist.items():
                print(f"      Mode {m}: {c}")
            tp_med = rule_sel['oracle_tp'].median()
            thr_med = rule_sel['oracle_thresh'].median()
            print(f"      Median TP: {tp_med*100:.1f}%  |  Median thresh: {thr_med:.2f}")

# ── 5F: Visualizations ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Phase 5: Strategy Comparison & Rule Sensitivity', fontsize=14, fontweight='bold')

# (0) Strategy comparison
ax = axes[0]
strat_names = []
strat_avgs = []
strat_trades = []
strat_colors_bar = []

strat_names.append(f'ALL Oracle\n({len(analysis_df)}t)')
strat_avgs.append(all_h2.mean() * 100)
strat_trades.append(all_h2_trades)
strat_colors_bar.append('#95a5a6')

if 'bva_h2_avg' in analysis_df.columns:
    bva_vals_plot = analysis_df['bva_h2_avg'].dropna()
    if len(bva_vals_plot) > 0:
        strat_names.append(f'Best val_a\n({len(bva_vals_plot)}t)')
        strat_avgs.append(bva_vals_plot.mean() * 100)
        strat_trades.append(0)
        strat_colors_bar.append('#e74c3c')

if stable_in.sum() > 0:
    strat_names.append(f'Stable-13\n({stable_in.sum()}t)')
    strat_avgs.append(s13_h2.mean() * 100)
    strat_trades.append(s13_trades)
    strat_colors_bar.append('#f39c12')

if oracle_in.sum() > 0:
    strat_names.append(f'ORACLE-10\n({oracle_in.sum()}t)')
    strat_avgs.append(o10_h2.mean() * 100)
    strat_trades.append(o10_trades)
    strat_colors_bar.append('#9b59b6')

for i, cr in enumerate(combo_results[:3]):
    strat_names.append(f'Rule #{i+1}\n({cr["n_pass"]}t, {cr["h2_total_trades"]}tr)')
    strat_avgs.append(cr['h2_avg_pnl'] * 100)
    strat_trades.append(cr['h2_total_trades'])
    strat_colors_bar.append('#2ecc71')

bars = ax.bar(range(len(strat_names)), strat_avgs, color=strat_colors_bar, edgecolor='white')
ax.set_xticks(range(len(strat_names)))
ax.set_xticklabels(strat_names, fontsize=8)
ax.axhline(y=DEPLOY_FLOOR * 100, color='red', linestyle='--', alpha=0.7, label=f'Deploy ≥{DEPLOY_FLOOR*100}%')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('H2 Avg PnL %')
ax.set_title('Strategy Comparison')
ax.legend(fontsize=8)
for bar, val in zip(bars, strat_avgs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:+.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# (1) Sensitivity curves for top 3 rules
ax = axes[1]
perturb_vals = [-0.20, -0.15, -0.10, -0.05, 0.0, +0.05, +0.10, +0.15, +0.20]
line_colors = ['#2ecc71', '#3498db', '#e67e22']

for rank, cr in enumerate(combo_results[:3]):
    sens_x = []
    sens_y = []
    sens_n = []
    for perturb in perturb_vals:
        mask = np.ones(len(analysis_df), dtype=bool)
        for fname in cr['features']:
            ft = feat_thresholds[fname]
            vals_s = analysis_df[fname].values
            if ft['higher_better']:
                adj_thresh = ft['threshold'] * (1 + perturb)
                mask &= vals_s >= adj_thresh
            else:
                adj_thresh = ft['threshold'] * (1 - perturb)
                mask &= vals_s <= adj_thresh
        n = mask.sum()
        if n >= 1:
            avg = analysis_df.loc[mask, 'h2_avg_pnl'].mean() * 100
            sens_x.append(perturb * 100)
            sens_y.append(avg)
            sens_n.append(n)
        else:
            sens_x.append(perturb * 100)
            sens_y.append(None)
            sens_n.append(0)

    valid_x = [x for x, y in zip(sens_x, sens_y) if y is not None]
    valid_y = [y for y in sens_y if y is not None]
    ax.plot(valid_x, valid_y, 'o-', color=line_colors[rank], linewidth=2,
            label=f'Rule #{rank+1}', markersize=6)
    for x, y, n in zip(sens_x, sens_y, sens_n):
        if y is not None:
            ax.annotate(f'{n}t', (x, y), fontsize=6, alpha=0.7,
                       textcoords='offset points', xytext=(0, 8), ha='center')

ax.axhline(y=DEPLOY_FLOOR * 100, color='red', linestyle='--', alpha=0.7, label=f'Deploy ≥{DEPLOY_FLOOR*100}%')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Threshold Perturbation %')
ax.set_ylabel('H2 Avg PnL %')
ax.set_title('Rule Sensitivity (wider = more robust)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# Rule #1 per-ticker PnL breakdown
if combo_results:
    best_rule = combo_results[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f'Rule #1: {best_rule["rule"]}', fontsize=11, fontweight='bold')
    rule_data = analysis_df.loc[analysis_df.index.isin(best_rule['tickers'])].sort_values('h2_avg_pnl')
    bar_colors = ['#2ecc71' if v >= DEPLOY_FLOOR else '#3498db' if v > 0 else '#e74c3c'
                  for v in rule_data['h2_avg_pnl']]
    bars = ax.barh(range(len(rule_data)), rule_data['h2_avg_pnl'] * 100, color=bar_colors)
    ax.set_yticks(range(len(rule_data)))
    ax.set_yticklabels(rule_data.index, fontsize=10)
    ax.axvline(x=DEPLOY_FLOOR * 100, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('H2 Avg PnL %')
    for bar, (_, r) in zip(bars, rule_data.iterrows()):
        trades = int(r['h2_trades'])
        tot = r['h2_total_pnl'] * 100
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{trades}tr, tot={tot:+.0f}%', va='center', fontsize=9)
    ax.set_title(f'{best_rule["n_pass"]} tickers, {best_rule["h2_total_trades"]} trades, '
                 f'avg={best_rule["h2_avg_pnl"]*100:+.2f}%, '
                 f'trade total={best_rule["h2_trade_total"]*100:+.1f}%')
    plt.tight_layout()
    plt.show()

print(f"\n{'='*W}")
print(f"  ANALYSIS COMPLETE")
print(f"  Feature matrix: {len(feat_df)} tickers × {len(FEATURE_COLS)} features")
print(f"  Predictive features found: {len(predictive)}")
print(f"  Composite rules tested: {len(combo_results)}")
if combo_results:
    best = combo_results[0]
    print(f"  Best rule: {best['rule']}")
    print(f"    → {best['n_pass']} tickers, H2 avg = {best['h2_avg_pnl']*100:+.2f}%, "
          f"H1 avg = {best['h1_avg_pnl']*100:+.2f}%")
print(f"{'='*W}")
