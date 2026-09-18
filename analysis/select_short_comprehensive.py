# %% ── Comprehensive SHORT Config Selection — 12 Unbiased Methods ────────────
#
# Tests 12 config selection methods (5 composite, 3 ML, 4 heuristic) on all
# 12 SHORT walkforward folds. Ensemble-only. No val_b/TEST data in training.
# Each fold is independent — no cross-fold referencing.
#
# Outputs:
#   /tmp/select_short_comp_summary.csv  — method comparison table
#   /tmp/select_short_comp_trades.csv   — all trades per method
#
# GPU NOT required. Colab-ready.

import os, io, gc, sys, warnings, ctypes
from datetime import datetime, timedelta
from collections import defaultdict
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

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
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
RESULTS_DIR   = "Algos/Results_all_v2"
INFERENCE_DIR = "Data/Inference_v2"

TEST_START  = '2026-04-01'
TEST_END    = '2026-07-28'
HALF1_END   = '2026-05-31'
HALF2_START = '2026-06-01'

ENTRY_MIN     = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.002

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

FOLDS_A2 = [400, 401, 402, 403, 404, 405]
FOLDS_A1 = [410, 411, 412, 413, 414, 415]
ALL_FOLDS = FOLDS_A2 + FOLDS_A1

FOLD_EXPERIMENTS    = {f: f'{f}_XSECTOR' for f in ALL_FOLDS}
FOLD_MODEL_VERSIONS = {f: f'robust_wf_v12_short_{f}' for f in ALL_FOLDS}

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_ALLOWED_MODES = ('static', 'atr_sl', 'grace')

DEPLOY_FLOOR = 0.018

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120

OUTPUT_SUMMARY = '/tmp/select_short_comp_summary.csv'
OUTPUT_TRADES  = '/tmp/select_short_comp_trades.csv'

METHOD_NAMES = [
    'A1_S3_COMPOSITE', 'A2_ANTI_OVERFIT', 'A3_WIDE_TP',
    'A4_MODE_AWARE', 'A5_COMBINED',
    'B1_RIDGE_CONFIG', 'B2_GB_CONFIG', 'B3_RIDGE_TICKER',
    'C1_FIXED_WIDE', 'C2_WIDE_TP_FILTER', 'C3_GRACE_ONLY', 'C4_HIGH_THRESH',
]


# ==============================================================================
# DATA LOADING (copied from oracle_vs_strategies.py)
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


# ==============================================================================
# SHORT TRADE SIMULATION (copied from oracle_vs_strategies.py)
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


# ==============================================================================
# PHASE 1: LOAD BQ ENSEMBLE CONFIGS
# ==============================================================================
np.random.seed(42)

print(f"{'='*W}")
print(f"  COMPREHENSIVE SHORT CONFIG SELECTION — 12 Unbiased Methods")
print(f"  TEST: {TEST_START} → {TEST_END}  |  H2: {HALF2_START} → {TEST_END}")
print(f"  Ensemble-only  |  No val_b in training  |  No cross-fold reference")
print(f"{'='*W}")

print(f"\n  PHASE 1: Loading BQ ensemble configs...")
mv_str = ", ".join(f"'{FOLD_MODEL_VERSIONS[f]}'" for f in ALL_FOLDS)
modes_str = ", ".join(f"'{m}'" for m in BQ_ALLOWED_MODES)

q = f"""
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       grace_minutes,
       opt_avg_pnl, opt_win_rate, opt_freq, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_freq, val_a_count,
       val_b_avg_pnl,
       is_ensemble, survived_a, model_version,
       vol_filter_pct, pre_range_filter_pct
FROM `{BQ_TABLE}`
WHERE model_version IN ({mv_str})
  AND opt_avg_pnl > 0
  AND tpsl_mode IN ({modes_str})
  AND is_ensemble = TRUE
ORDER BY ticker, model_version
"""
bq_df = bq_client.query(q).to_dataframe()
bq_df['fold'] = bq_df['model_version'].str.extract(r'_(\d+)$').astype(int)

print(f"    BQ: {len(bq_df)} rows ({bq_df['ticker'].nunique()} tickers, {bq_df['fold'].nunique()} folds)")
print(f"    Modes: {bq_df['tpsl_mode'].value_counts().to_dict()}")
print(f"    Folds: {sorted(bq_df['fold'].unique())}")


# ==============================================================================
# PHASE 2: FEATURE ENGINEERING (per-config row)
# ==============================================================================
print(f"\n  PHASE 2: Feature engineering...")

bq_df['is_static'] = (bq_df['tpsl_mode'] == 'static').astype(int)
bq_df['is_atr_sl'] = (bq_df['tpsl_mode'] == 'atr_sl').astype(int)
bq_df['is_grace']  = (bq_df['tpsl_mode'] == 'grace').astype(int)
bq_df['is_a2']     = (bq_df['fold'] < 410).astype(int)
bq_df['fold_norm'] = (bq_df['fold'] - 400) / 15.0

for col in ['prob_thresh', 'tp_pct', 'sl_pct', 'sl_mult', 'grace_minutes',
            'entry_minute', 'vol_filter_pct', 'pre_range_filter_pct',
            'opt_avg_pnl', 'opt_win_rate', 'opt_count', 'opt_freq',
            'val_a_avg_pnl', 'val_a_win_rate', 'val_a_count', 'val_a_freq']:
    bq_df[col] = bq_df[col].apply(lambda x: _safe_float(x, 0.0))

bq_df['va_opt_ratio'] = bq_df['val_a_avg_pnl'] / bq_df['opt_avg_pnl'].clip(lower=0.001)
bq_df['tp_sl_ratio']  = bq_df['tp_pct'] / bq_df['sl_pct'].clip(lower=0.005)
bq_df['opt_pnl_per_trade'] = bq_df['opt_avg_pnl'] / bq_df['opt_freq'].clip(lower=0.01)

HYPERPARAM_FEATURES = [
    'prob_thresh', 'tp_pct', 'sl_pct', 'sl_mult', 'grace_minutes',
    'entry_minute', 'vol_filter_pct', 'pre_range_filter_pct',
    'is_static', 'is_atr_sl', 'is_grace', 'is_a2', 'fold_norm',
]
OPT_FEATURES = [
    'opt_avg_pnl', 'opt_win_rate', 'opt_count', 'opt_freq',
]
DERIVED_FEATURES = [
    'va_opt_ratio', 'tp_sl_ratio', 'opt_pnl_per_trade',
]
ML_INPUT_FEATURES = HYPERPARAM_FEATURES + OPT_FEATURES + DERIVED_FEATURES

print(f"    {len(ML_INPUT_FEATURES)} ML input features (no val_a/val_b in inputs):")
for i, f in enumerate(ML_INPUT_FEATURES, 1):
    vals = bq_df[f]
    print(f"      {i:2d}. {f:<25} min={vals.min():>10.3f}  med={vals.median():>10.3f}  max={vals.max():>10.3f}")

n_tickers = bq_df['ticker'].nunique()
n_folds   = bq_df['fold'].nunique()
print(f"\n    {len(bq_df)} configs × {n_tickers} tickers × {n_folds} folds")
print(f"    val_a > 0: {(bq_df['val_a_avg_pnl'] > 0).sum()}/{len(bq_df)} ({100*(bq_df['val_a_avg_pnl'] > 0).mean():.0f}%)")
print(f"    val_a >= 1.8%: {(bq_df['val_a_avg_pnl'] >= 0.018).sum()}/{len(bq_df)} ({100*(bq_df['val_a_avg_pnl'] >= 0.018).mean():.0f}%)")


# ==============================================================================
# PHASE 3: SCORING METHODS — select best config per ticker
# ==============================================================================
print(f"\n  PHASE 3: Scoring — 12 methods, pick best config per ticker...")

method_selections = {}

# ── A: Composite Scores ──────────────────────────────────────────────────────
def score_s3_composite(row):
    va = row['val_a_avg_pnl']
    return (va ** 2) * int(va >= DEPLOY_FLOOR)

def score_anti_overfit(row):
    va = row['val_a_avg_pnl']
    ratio = row['va_opt_ratio']
    return va * max(0.0, 1.0 - abs(ratio - 1.0) * 0.5)

def score_wide_tp(row):
    va = row['val_a_avg_pnl']
    tp = row['tp_pct']
    return va * (1.0 + 0.5 * max(0.0, tp - 0.08) / 0.04)

def score_mode_aware(row):
    va = row['val_a_avg_pnl']
    bonus = 1.3 if row['is_grace'] else (1.1 if row['is_atr_sl'] else 1.0)
    return va * bonus

def score_combined(row):
    va = row['val_a_avg_pnl']
    tp = row['tp_pct']
    tp_bonus = 1.0 + 0.5 * max(0.0, tp - 0.08) / 0.04
    mode_bonus = 1.3 if row['is_grace'] else (1.1 if row['is_atr_sl'] else 1.0)
    ratio = row['va_opt_ratio']
    overfit_penalty = max(0.0, 1.0 - abs(ratio - 1.0) * 0.5)
    return (va ** 2) * tp_bonus * mode_bonus * overfit_penalty

COMPOSITE_METHODS = {
    'A1_S3_COMPOSITE': score_s3_composite,
    'A2_ANTI_OVERFIT': score_anti_overfit,
    'A3_WIDE_TP':      score_wide_tp,
    'A4_MODE_AWARE':   score_mode_aware,
    'A5_COMBINED':     score_combined,
}

for method_name, score_fn in COMPOSITE_METHODS.items():
    bq_df['_score'] = bq_df.apply(score_fn, axis=1)
    best_idx = bq_df.groupby('ticker')['_score'].idxmax()
    selections = {}
    for ticker, idx in best_idx.items():
        row = bq_df.loc[idx]
        if row['_score'] > 0:
            selections[ticker] = {
                'fold': int(row['fold']),
                'tpsl_mode': row['tpsl_mode'],
                'prob_thresh': row['prob_thresh'],
                'tp_pct': row['tp_pct'],
                'sl_pct': row['sl_pct'],
                'sl_mult': row['sl_mult'],
                'grace_minutes': int(row['grace_minutes']),
                'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else ENTRY_MIN,
                'score': row['_score'],
                'opt_avg_pnl': row['opt_avg_pnl'],
                'val_a_avg_pnl': row['val_a_avg_pnl'],
            }
    method_selections[method_name] = selections
    print(f"    {method_name}: {len(selections)} tickers selected")

bq_df.drop(columns=['_score'], inplace=True, errors='ignore')

# ── B: ML Methods (LOO-Ticker CV) ───────────────────────────────────────────
print(f"\n    ML Methods (LOO-Ticker CV)...")

tickers_all = sorted(bq_df['ticker'].unique())
X_ml = bq_df[ML_INPUT_FEATURES].values.astype(np.float64)
y_ml = bq_df['val_a_avg_pnl'].values.astype(np.float64)
ticker_col = bq_df['ticker'].values

ML_MODELS = {
    'B1_RIDGE_CONFIG': (Ridge, dict(alpha=1.0)),
    'B2_GB_CONFIG': (GradientBoostingRegressor, dict(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)),
}

for method_name, (model_cls, model_params) in ML_MODELS.items():
    predictions = np.zeros(len(bq_df))

    for test_ticker in tickers_all:
        train_mask = ticker_col != test_ticker
        test_mask  = ticker_col == test_ticker

        X_train, y_train = X_ml[train_mask], y_ml[train_mask]
        X_test = X_ml[test_mask]

        if len(X_train) < 10 or len(X_test) == 0:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = model_cls(**model_params)
        model.fit(X_train_s, y_train)
        predictions[test_mask] = model.predict(X_test_s)

    bq_df[f'_pred_{method_name}'] = predictions

    rho, p = sp_stats.spearmanr(y_ml, predictions)
    print(f"      {method_name}: ρ={rho:+.3f} (p={p:.4f})")

    selections = {}
    for ticker in tickers_all:
        t_mask = bq_df['ticker'] == ticker
        t_df = bq_df[t_mask]
        if len(t_df) == 0:
            continue
        best_idx = t_df[f'_pred_{method_name}'].idxmax()
        row = bq_df.loc[best_idx]
        if row[f'_pred_{method_name}'] > 0:
            selections[ticker] = {
                'fold': int(row['fold']),
                'tpsl_mode': row['tpsl_mode'],
                'prob_thresh': row['prob_thresh'],
                'tp_pct': row['tp_pct'],
                'sl_pct': row['sl_pct'],
                'sl_mult': row['sl_mult'],
                'grace_minutes': int(row['grace_minutes']),
                'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else ENTRY_MIN,
                'score': float(row[f'_pred_{method_name}']),
                'opt_avg_pnl': row['opt_avg_pnl'],
                'val_a_avg_pnl': row['val_a_avg_pnl'],
            }
    method_selections[method_name] = selections
    print(f"      {method_name}: {len(selections)} tickers selected")

# B3: RIDGE_TICKER — ticker-level features (ADAPTIVE_RELAXED pattern)
print(f"\n    B3_RIDGE_TICKER (ticker-level, ADAPTIVE_RELAXED pattern)...")
ens_df = bq_df.copy()

ticker_bq_feats = {}
for ticker, grp in ens_df.groupby('ticker'):
    n_configs = len(grp)
    n_a2 = (grp['fold'] < 410).sum()
    n_a1 = (grp['fold'] >= 410).sum()
    has_both = int(n_a2 > 0 and n_a1 > 0)

    a2_va = grp.loc[grp['fold'] < 410, 'val_a_avg_pnl']
    a1_va = grp.loc[grp['fold'] >= 410, 'val_a_avg_pnl']
    cross_agree = 0
    if len(a2_va) > 0 and len(a1_va) > 0:
        cross_agree = int(a2_va.median() > 0 and a1_va.median() > 0)

    agree_18 = (grp['opt_avg_pnl'] >= DEPLOY_FLOOR).mean()
    opt_wr_med = grp['opt_win_rate'].median()
    pct_static = grp['is_static'].mean()
    pct_atr_sl = grp['is_atr_sl'].mean()
    pct_grace = grp['is_grace'].mean()
    va_med = grp['val_a_avg_pnl'].median()
    va_pos = (grp['val_a_avg_pnl'] > 0).mean()
    va_opt_med = grp['va_opt_ratio'].median()

    ticker_bq_feats[ticker] = {
        'n_configs': n_configs,
        'has_both_alphas': has_both,
        'cross_alpha_agree': cross_agree,
        'agree_18_best': agree_18,
        'opt_wr_median': opt_wr_med,
        'pct_static': pct_static,
        'pct_atr_sl': pct_atr_sl,
        'pct_grace': pct_grace,
        'pct_limit': 0.0,
        'va_median': va_med,
        'va_pos_rate': va_pos,
        'va_opt_ratio_med': va_opt_med,
    }

BQ_FEAT_NAMES = [
    'n_configs', 'has_both_alphas', 'cross_alpha_agree', 'agree_18_best',
    'opt_wr_median', 'pct_static', 'pct_atr_sl', 'pct_grace',
    'pct_limit', 'va_median', 'va_pos_rate', 'va_opt_ratio_med',
]

feat_df = pd.DataFrame.from_dict(ticker_bq_feats, orient='index')
feat_df = feat_df[BQ_FEAT_NAMES]
feat_df['_target'] = feat_df['va_median']

X_tk = feat_df[BQ_FEAT_NAMES].values.astype(np.float64)
y_tk = feat_df['_target'].values.astype(np.float64)
tk_list = feat_df.index.tolist()

predictions_tk = np.zeros(len(tk_list))
for i, test_tk in enumerate(tk_list):
    train_mask = np.array([j for j in range(len(tk_list)) if j != i])
    if len(train_mask) < 5:
        continue
    X_train = X_tk[train_mask]
    y_train = y_tk[train_mask]
    X_test  = X_tk[i:i+1]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    predictions_tk[i] = model.predict(X_test_s)[0]

rho_tk, p_tk = sp_stats.spearmanr(y_tk, predictions_tk)
print(f"      B3_RIDGE_TICKER: ρ={rho_tk:+.3f} (p={p_tk:.4f})")

selections_b3 = {}
for i, ticker in enumerate(tk_list):
    if predictions_tk[i] <= 0:
        continue
    t_df = ens_df[ens_df['ticker'] == ticker]
    if len(t_df) == 0:
        continue
    best_idx = t_df['val_a_avg_pnl'].idxmax()
    row = ens_df.loc[best_idx]
    va_opt = row['va_opt_ratio']
    if va_opt > 5.0:
        continue
    selections_b3[ticker] = {
        'fold': int(row['fold']),
        'tpsl_mode': row['tpsl_mode'],
        'prob_thresh': row['prob_thresh'],
        'tp_pct': row['tp_pct'],
        'sl_pct': row['sl_pct'],
        'sl_mult': row['sl_mult'],
        'grace_minutes': int(row['grace_minutes']),
        'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else ENTRY_MIN,
        'score': float(predictions_tk[i]),
        'opt_avg_pnl': row['opt_avg_pnl'],
        'val_a_avg_pnl': row['val_a_avg_pnl'],
    }
method_selections['B3_RIDGE_TICKER'] = selections_b3
print(f"      B3_RIDGE_TICKER: {len(selections_b3)} tickers selected")


# ── C: Heuristic Methods ────────────────────────────────────────────────────
print(f"\n    Heuristic methods...")

# C1: FIXED_WIDE — closest to TP=12%/SL=8%/grace
TARGET_TP, TARGET_SL = 0.12, 0.08
selections_c1 = {}
for ticker in tickers_all:
    t_df = ens_df[ens_df['ticker'] == ticker]
    grace_df = t_df[t_df['tpsl_mode'] == 'grace']
    search_df = grace_df if len(grace_df) > 0 else t_df
    if len(search_df) == 0:
        continue
    dist = ((search_df['tp_pct'] - TARGET_TP).abs() + (search_df['sl_pct'] - TARGET_SL).abs())
    best_idx = dist.idxmin()
    row = ens_df.loc[best_idx]
    selections_c1[ticker] = {
        'fold': int(row['fold']),
        'tpsl_mode': row['tpsl_mode'],
        'prob_thresh': row['prob_thresh'],
        'tp_pct': row['tp_pct'],
        'sl_pct': row['sl_pct'],
        'sl_mult': row['sl_mult'],
        'grace_minutes': int(row['grace_minutes']),
        'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else ENTRY_MIN,
        'score': float(-dist.min()),
        'opt_avg_pnl': row['opt_avg_pnl'],
        'val_a_avg_pnl': row['val_a_avg_pnl'],
    }
method_selections['C1_FIXED_WIDE'] = selections_c1
print(f"    C1_FIXED_WIDE: {len(selections_c1)} tickers (target TP={TARGET_TP*100:.0f}%/SL={TARGET_SL*100:.0f}%/grace)")

# C2: WIDE_TP_FILTER — filter TP >= 10%, rank by val_a
selections_c2 = {}
for ticker in tickers_all:
    t_df = ens_df[(ens_df['ticker'] == ticker) & (ens_df['tp_pct'] >= 0.10)]
    if len(t_df) == 0:
        continue
    best_idx = t_df['val_a_avg_pnl'].idxmax()
    row = ens_df.loc[best_idx]
    if row['val_a_avg_pnl'] <= 0:
        continue
    selections_c2[ticker] = {
        'fold': int(row['fold']),
        'tpsl_mode': row['tpsl_mode'],
        'prob_thresh': row['prob_thresh'],
        'tp_pct': row['tp_pct'],
        'sl_pct': row['sl_pct'],
        'sl_mult': row['sl_mult'],
        'grace_minutes': int(row['grace_minutes']),
        'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else ENTRY_MIN,
        'score': row['val_a_avg_pnl'],
        'opt_avg_pnl': row['opt_avg_pnl'],
        'val_a_avg_pnl': row['val_a_avg_pnl'],
    }
method_selections['C2_WIDE_TP_FILTER'] = selections_c2
print(f"    C2_WIDE_TP_FILTER: {len(selections_c2)} tickers (TP >= 10%, rank by val_a)")

# C3: GRACE_ONLY — filter grace mode, rank by val_a
selections_c3 = {}
for ticker in tickers_all:
    t_df = ens_df[(ens_df['ticker'] == ticker) & (ens_df['tpsl_mode'] == 'grace')]
    if len(t_df) == 0:
        continue
    best_idx = t_df['val_a_avg_pnl'].idxmax()
    row = ens_df.loc[best_idx]
    if row['val_a_avg_pnl'] <= 0:
        continue
    selections_c3[ticker] = {
        'fold': int(row['fold']),
        'tpsl_mode': row['tpsl_mode'],
        'prob_thresh': row['prob_thresh'],
        'tp_pct': row['tp_pct'],
        'sl_pct': row['sl_pct'],
        'sl_mult': row['sl_mult'],
        'grace_minutes': int(row['grace_minutes']),
        'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else ENTRY_MIN,
        'score': row['val_a_avg_pnl'],
        'opt_avg_pnl': row['opt_avg_pnl'],
        'val_a_avg_pnl': row['val_a_avg_pnl'],
    }
method_selections['C3_GRACE_ONLY'] = selections_c3
print(f"    C3_GRACE_ONLY: {len(selections_c3)} tickers (grace mode, rank by val_a)")

# C4: HIGH_THRESH — highest prob_thresh per ticker
selections_c4 = {}
for ticker in tickers_all:
    t_df = ens_df[ens_df['ticker'] == ticker]
    if len(t_df) == 0:
        continue
    best_idx = t_df['prob_thresh'].idxmax()
    row = ens_df.loc[best_idx]
    selections_c4[ticker] = {
        'fold': int(row['fold']),
        'tpsl_mode': row['tpsl_mode'],
        'prob_thresh': row['prob_thresh'],
        'tp_pct': row['tp_pct'],
        'sl_pct': row['sl_pct'],
        'sl_mult': row['sl_mult'],
        'grace_minutes': int(row['grace_minutes']),
        'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else ENTRY_MIN,
        'score': row['prob_thresh'],
        'opt_avg_pnl': row['opt_avg_pnl'],
        'val_a_avg_pnl': row['val_a_avg_pnl'],
    }
method_selections['C4_HIGH_THRESH'] = selections_c4
print(f"    C4_HIGH_THRESH: {len(selections_c4)} tickers (highest prob_thresh)")


# ==============================================================================
# PHASE 4: LOAD OHLCV + SEED SCORES
# ==============================================================================
print(f"\n  PHASE 4: Loading OHLCV + seed scores...")

all_selected_tickers = set()
all_selected_folds = set()
for method_name, selections in method_selections.items():
    for ticker, cfg in selections.items():
        all_selected_tickers.add(ticker)
        all_selected_folds.add(cfg['fold'])

print(f"    {len(all_selected_tickers)} unique tickers across all methods")
print(f"    {len(all_selected_folds)} unique folds needed")

ticker_data = {}
for i, ticker in enumerate(sorted(all_selected_tickers)):
    df = load_master(ticker)
    if df is None:
        continue
    df = heal_data(df)
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
    if (i + 1) % 10 == 0:
        print(f"      {i+1}/{len(all_selected_tickers)} loaded")
        _release_memory()
_release_memory()
print(f"    OHLCV: {len(ticker_data)} tickers loaded")

needed_tf = set()
for method_name, selections in method_selections.items():
    for ticker, cfg in selections.items():
        if ticker in ticker_data:
            needed_tf.add((ticker, cfg['fold']))

print(f"\n    Loading seed scores for {len(needed_tf)} (ticker, fold) pairs...")
daily_probs = defaultdict(dict)
loaded_tf = set()
for fold in sorted(all_selected_folds):
    exp = FOLD_EXPERIMENTS[fold]
    fold_tickers = [t for (t, f) in needed_tf if f == fold and t in ticker_data]
    n_loaded = 0
    for ticker in fold_tickers:
        if (ticker, fold) in loaded_tf:
            continue
        seeds = load_seed_scores(ticker, exp)
        if not seeds:
            continue
        dp = compute_daily_prob(seeds, ENTRY_MIN)
        if dp:
            daily_probs[ticker][fold] = dp
            loaded_tf.add((ticker, fold))
            n_loaded += 1
        del seeds
    print(f"      Fold {fold}: {n_loaded} tickers")
    _release_memory()
print(f"    Total: {len(loaded_tf)} (ticker, fold) probability sets")


# ==============================================================================
# PHASE 5: TRADE SIMULATION
# ==============================================================================
print(f"\n  PHASE 5: Trade simulation...")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

unique_configs = {}
for method_name, selections in method_selections.items():
    for ticker, cfg in selections.items():
        config_key = (ticker, cfg['fold'], cfg['tpsl_mode'],
                      cfg['tp_pct'], cfg['sl_pct'], cfg['sl_mult'],
                      cfg['grace_minutes'])
        if config_key not in unique_configs:
            unique_configs[config_key] = cfg

print(f"    {len(unique_configs)} unique (ticker, fold, config) combos to simulate")

trade_results = {}
n_sim = 0
for config_key, cfg in unique_configs.items():
    ticker = config_key[0]
    fold = config_key[1]

    if ticker not in ticker_data:
        continue
    ohlcv = ticker_data[ticker]
    probs = daily_probs.get(ticker, {}).get(fold)
    if not probs:
        continue

    entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]
    trades_for_config = []
    last_td = -1

    for idx in entry_indices:
        d_int = int(ohlcv['dates'][idx])
        if d_int == last_td:
            continue
        d_str = str(d_int)
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < test_start_dt or d_date > test_end_dt:
            continue
        prob_val = probs.get(d_date, 0.0)
        if prob_val < cfg['prob_thresh']:
            continue

        pnl, td, filled = execute_trade(
            ohlcv, idx, cfg['tpsl_mode'],
            cfg['tp_pct'], cfg['sl_pct'], cfg['sl_mult'], cfg['grace_minutes'])
        if filled:
            trades_for_config.append({
                'date': d_date,
                'pnl': pnl,
                'ticker': ticker,
                'fold': fold,
            })
            last_td = d_int

    trade_results[config_key] = trades_for_config
    n_sim += 1

print(f"    Simulated {n_sim} configs, {sum(len(t) for t in trade_results.values())} total trades")


# ==============================================================================
# PHASE 6: EVALUATION + OUTPUT
# ==============================================================================
print(f"\n  PHASE 6: Evaluation...")
print(f"\n{'='*W}")
print(f"  METHOD COMPARISON — ALL 12 METHODS")
print(f"{'='*W}")

header = (f"  {'Method':<22} │ {'Tks':>3} {'Trd':>4} │ {'Avg%':>7} {'WR':>5} {'Total%':>8} │ "
          f"{'H2_Trd':>6} {'H2_Avg':>7} {'H2_Tot':>8} │ {'Big%':>5}")
print(header)
print(f"  {'─'*22}─┼─{'─'*3}─{'─'*4}─┼─{'─'*7}─{'─'*5}─{'─'*8}─┼─{'─'*6}─{'─'*7}─{'─'*8}─┼─{'─'*5}")

method_results = {}
all_method_trades = []

for method_name in METHOD_NAMES:
    selections = method_selections.get(method_name, {})
    method_trades = []

    for ticker, cfg in selections.items():
        config_key = (ticker, cfg['fold'], cfg['tpsl_mode'],
                      cfg['tp_pct'], cfg['sl_pct'], cfg['sl_mult'],
                      cfg['grace_minutes'])
        trades = trade_results.get(config_key, [])
        method_trades.extend(trades)

    if not method_trades:
        print(f"  {method_name:<22} │   0    0 │   0.00     0   0.00 │      0   0.00     0.00 │  0.0")
        method_results[method_name] = {
            'n_tickers': 0, 'n_trades': 0, 'avg_pnl': 0, 'wr': 0,
            'total_pnl': 0, 'h2_n': 0, 'h2_avg': 0, 'h2_total': 0, 'big_pct': 0,
        }
        continue

    pnls = np.array([t['pnl'] for t in method_trades])
    tickers_set = set(t['ticker'] for t in method_trades)
    n_trades = len(pnls)
    avg_pnl = float(pnls.mean())
    wr = float((pnls > 0).mean())
    total_pnl = float(pnls.sum())

    h2_pnls = pnls[np.array([t['date'] >= h2_start_dt for t in method_trades])]
    h2_n = len(h2_pnls)
    h2_total = float(h2_pnls.sum()) if h2_n > 0 else 0.0
    h2_avg = float(h2_pnls.mean()) if h2_n > 0 else 0.0

    big_pct = float((pnls > 0.02).sum() / n_trades * 100) if n_trades > 0 else 0.0

    print(f"  {method_name:<22} │ {len(tickers_set):>3} {n_trades:>4} │ "
          f"{avg_pnl*100:>+6.2f} {wr*100:>4.0f}% {total_pnl*100:>+7.1f} │ "
          f"{h2_n:>6} {h2_avg*100:>+6.2f} {h2_total*100:>+7.1f} │ {big_pct:>4.1f}")

    method_results[method_name] = {
        'n_tickers': len(tickers_set), 'n_trades': n_trades,
        'avg_pnl': avg_pnl, 'wr': wr, 'total_pnl': total_pnl,
        'h2_n': h2_n, 'h2_avg': h2_avg, 'h2_total': h2_total,
        'big_pct': big_pct,
    }

    for t in method_trades:
        all_method_trades.append({**t, 'method': method_name})


# ── Per-ticker breakdown for top 3 methods by H2 avg ────────────────────────
print(f"\n{'='*W}")
print(f"  PER-TICKER BREAKDOWN — TOP 3 METHODS BY H2 AVG PnL")
print(f"{'='*W}")

ranked = sorted(method_results.items(),
                key=lambda x: x[1]['h2_avg'] if x[1]['h2_n'] >= 3 else -999,
                reverse=True)

for rank, (method_name, res) in enumerate(ranked[:3], 1):
    if res['h2_n'] < 1:
        continue
    selections = method_selections.get(method_name, {})
    print(f"\n  #{rank} {method_name}  (H2 avg: {res['h2_avg']*100:+.2f}%, {res['n_trades']} trades, {res['n_tickers']} tickers)")
    print(f"  {'Ticker':<8} │ {'Fold':>4} {'Mode':<7} {'TP%':>5} {'SL%':>5} {'Thresh':>6} │ "
          f"{'Trd':>3} {'Avg%':>7} {'WR':>5} {'Total%':>8} │ "
          f"{'H2_Trd':>6} {'H2_Avg%':>8} {'H2_Tot%':>8}")
    print(f"  {'─'*8}─┼─{'─'*4}─{'─'*7}─{'─'*5}─{'─'*5}─{'─'*6}─┼─{'─'*3}─{'─'*7}─{'─'*5}─{'─'*8}─┼─{'─'*6}─{'─'*8}─{'─'*8}")

    ticker_trades = defaultdict(list)
    for t in all_method_trades:
        if t['method'] == method_name:
            ticker_trades[t['ticker']].append(t)

    ticker_rows = []
    for ticker in sorted(ticker_trades.keys()):
        trades = ticker_trades[ticker]
        pnls = np.array([t['pnl'] for t in trades])
        h2_pnls = np.array([t['pnl'] for t in trades if t['date'] >= h2_start_dt])
        cfg = selections.get(ticker, {})
        ticker_rows.append((
            ticker, cfg,
            len(pnls), float(pnls.mean()), float((pnls > 0).mean()), float(pnls.sum()),
            len(h2_pnls),
            float(h2_pnls.mean()) if len(h2_pnls) > 0 else 0.0,
            float(h2_pnls.sum()) if len(h2_pnls) > 0 else 0.0,
        ))

    ticker_rows.sort(key=lambda x: x[8], reverse=True)
    for (ticker, cfg, n, avg, wr, tot, h2n, h2a, h2t) in ticker_rows:
        fold = cfg.get('fold', 0)
        mode = cfg.get('tpsl_mode', '?')[:7]
        tp = cfg.get('tp_pct', 0) * 100
        sl = cfg.get('sl_pct', 0) * 100
        thresh = cfg.get('prob_thresh', 0)
        print(f"  {ticker:<8} │ {fold:>4} {mode:<7} {tp:>4.1f} {sl:>4.1f} {thresh:>5.3f} │ "
              f"{n:>3} {avg*100:>+6.2f} {wr*100:>4.0f}% {tot*100:>+7.1f} │ "
              f"{h2n:>6} {h2a*100:>+7.2f} {h2t*100:>+7.1f}")


# ── Config characteristics per method ────────────────────────────────────────
print(f"\n{'='*W}")
print(f"  CONFIG CHARACTERISTICS PER METHOD")
print(f"{'='*W}")

print(f"  {'Method':<22} │ {'TP_mean':>7} {'SL_mean':>7} {'Thresh':>7} │ {'static':>6} {'atr_sl':>6} {'grace':>6} │ {'va_mean':>7}")
print(f"  {'─'*22}─┼─{'─'*7}─{'─'*7}─{'─'*7}─┼─{'─'*6}─{'─'*6}─{'─'*6}─┼─{'─'*7}")

for method_name in METHOD_NAMES:
    selections = method_selections.get(method_name, {})
    if not selections:
        continue
    tps = [c['tp_pct'] for c in selections.values()]
    sls = [c['sl_pct'] for c in selections.values()]
    threshs = [c['prob_thresh'] for c in selections.values()]
    modes = [c['tpsl_mode'] for c in selections.values()]
    vas = [c['val_a_avg_pnl'] for c in selections.values()]

    n = len(selections)
    n_static = sum(1 for m in modes if m == 'static')
    n_atr = sum(1 for m in modes if m == 'atr_sl')
    n_grace = sum(1 for m in modes if m == 'grace')

    print(f"  {method_name:<22} │ {np.mean(tps)*100:>6.1f}% {np.mean(sls)*100:>6.1f}% {np.mean(threshs):>6.3f} │ "
          f"{n_static/n*100:>5.0f}% {n_atr/n*100:>5.0f}% {n_grace/n*100:>5.0f}% │ {np.mean(vas)*100:>+6.2f}%")


# ── Cross-method agreement ──────────────────────────────────────────────────
print(f"\n{'='*W}")
print(f"  CROSS-METHOD AGREEMENT — which tickers appear in how many methods")
print(f"{'='*W}")

ticker_method_count = defaultdict(int)
ticker_method_list = defaultdict(list)
for method_name in METHOD_NAMES:
    for ticker in method_selections.get(method_name, {}).keys():
        ticker_method_count[ticker] += 1
        ticker_method_list[ticker].append(method_name)

consensus = sorted(ticker_method_count.items(), key=lambda x: -x[1])
print(f"\n  {'Ticker':<8} │ {'#Methods':>8} │ Methods")
print(f"  {'─'*8}─┼─{'─'*8}─┼─{'─'*50}")

for ticker, count in consensus:
    methods_str = ', '.join(sorted(ticker_method_list[ticker]))
    if len(methods_str) > 80:
        methods_str = methods_str[:77] + '...'
    print(f"  {ticker:<8} │ {count:>8} │ {methods_str}")

print(f"\n  Consensus tiers:")
for thresh in [12, 10, 8, 6, 4]:
    n = sum(1 for _, c in consensus if c >= thresh)
    tks = [t for t, c in consensus if c >= thresh]
    print(f"    >= {thresh:2d} methods: {n:>2} tickers  {', '.join(tks[:10])}")


# ── Big-trade analysis ───────────────────────────────────────────────────────
print(f"\n{'='*W}")
print(f"  BIG-TRADE ANALYSIS — % of trades by PnL bucket")
print(f"{'='*W}")

print(f"  {'Method':<22} │ {'<-3%':>5} {'-3~-1':>5} {'-1~0':>5} {'0~1%':>5} {'1~2%':>5} {'2~3%':>5} {'3~5%':>5} {'>5%':>5}")
print(f"  {'─'*22}─┼─{'─'*5}─{'─'*5}─{'─'*5}─{'─'*5}─{'─'*5}─{'─'*5}─{'─'*5}─{'─'*5}")

for method_name in METHOD_NAMES:
    trades = [t for t in all_method_trades if t['method'] == method_name]
    if not trades:
        continue
    pnls = np.array([t['pnl'] for t in trades]) * 100
    n = len(pnls)
    buckets = [
        (pnls < -3).sum(),
        ((pnls >= -3) & (pnls < -1)).sum(),
        ((pnls >= -1) & (pnls < 0)).sum(),
        ((pnls >= 0) & (pnls < 1)).sum(),
        ((pnls >= 1) & (pnls < 2)).sum(),
        ((pnls >= 2) & (pnls < 3)).sum(),
        ((pnls >= 3) & (pnls < 5)).sum(),
        (pnls >= 5).sum(),
    ]
    pcts = [b / n * 100 for b in buckets]
    print(f"  {method_name:<22} │ {pcts[0]:>4.0f}% {pcts[1]:>4.0f}% {pcts[2]:>4.0f}% {pcts[3]:>4.0f}% "
          f"{pcts[4]:>4.0f}% {pcts[5]:>4.0f}% {pcts[6]:>4.0f}% {pcts[7]:>4.0f}%")


# ── Monthly breakdown for top method ─────────────────────────────────────────
print(f"\n{'='*W}")
print(f"  MONTHLY BREAKDOWN — TOP 3 METHODS")
print(f"{'='*W}")

for rank, (method_name, res) in enumerate(ranked[:3], 1):
    if res['n_trades'] < 1:
        continue
    trades = [t for t in all_method_trades if t['method'] == method_name]
    monthly = defaultdict(list)
    for t in trades:
        m = str(t['date'])[:7]
        monthly[m].append(t['pnl'])

    print(f"\n  #{rank} {method_name}:")
    print(f"    {'Month':<8} │ {'Trades':>6} {'Avg%':>7} {'WR':>5} {'Total%':>8}")
    print(f"    {'─'*8}─┼─{'─'*6}─{'─'*7}─{'─'*5}─{'─'*8}")
    for m in sorted(monthly.keys()):
        pnls = np.array(monthly[m])
        print(f"    {m:<8} │ {len(pnls):>6} {pnls.mean()*100:>+6.2f} {(pnls>0).mean()*100:>4.0f}% {pnls.sum()*100:>+7.1f}")


# ── Save outputs ─────────────────────────────────────────────────────────────
print(f"\n  Saving outputs...")

summary_rows = []
for method_name in METHOD_NAMES:
    r = method_results.get(method_name, {})
    summary_rows.append({
        'method': method_name,
        'n_tickers': r.get('n_tickers', 0),
        'n_trades': r.get('n_trades', 0),
        'avg_pnl': r.get('avg_pnl', 0),
        'wr': r.get('wr', 0),
        'total_pnl': r.get('total_pnl', 0),
        'h2_n': r.get('h2_n', 0),
        'h2_avg': r.get('h2_avg', 0),
        'h2_total': r.get('h2_total', 0),
        'big_pct': r.get('big_pct', 0),
    })

pd.DataFrame(summary_rows).to_csv(OUTPUT_SUMMARY, index=False)
print(f"    Summary: {OUTPUT_SUMMARY}")

trades_df = pd.DataFrame(all_method_trades)
if len(trades_df) > 0:
    trades_df.to_csv(OUTPUT_TRADES, index=False)
    print(f"    Trades: {OUTPUT_TRADES} ({len(trades_df)} rows)")

print(f"\n{'='*W}")
print(f"  DONE — {len(METHOD_NAMES)} methods evaluated")
print(f"  Best by H2 avg: {ranked[0][0]} ({ranked[0][1]['h2_avg']*100:+.2f}%)" if ranked else "  No methods with trades")
print(f"{'='*W}")
