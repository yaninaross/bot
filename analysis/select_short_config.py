# %% ── SHORT: Config Selection — 17 Methods for Picking Best BQ Config ───────
#
# Phase B of SHORT pipeline: given selected tickers (from Phase A asset
# selection), pick the BEST BQ config per ticker using only BQ-observable
# properties. Tests 14 rule-based + 3 ML methods against oracle ceiling.
#
# Phase 1: Load BQ configs + OHLCV + seed scores
# Phase 2: Define oracle ticker groups
# Phase 3: Exhaustive H2 backtest (ground truth)
# Phase 4: 17 config selection methods
# Phase 5: Evaluate all methods per group
# Phase 6: Top-3 deep dive
# Phase 7: Graphs
#
# GPU NOT required. Colab-ready.

import os, io, gc, sys, warnings, ctypes
from datetime import datetime, timedelta
from collections import defaultdict, Counter
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

try:
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

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
SL_FIXED      = 0.03
DEPLOY_FLOOR  = 0.018

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

FOLDS_A2 = [400, 401, 402, 403, 404, 405]
FOLDS_A1 = [410, 411, 412, 413, 414, 415]
ALL_FOLDS = FOLDS_A2 + FOLDS_A1

FOLD_EXPERIMENTS    = {f: f'{f}_XSECTOR' for f in ALL_FOLDS}
FOLD_MODEL_VERSIONS = {f: f'robust_wf_v12_short_{f}' for f in ALL_FOLDS}

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_ALLOWED_MODES = ('static', 'atr_sl', 'grace')

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120

MIN_H2_TRADES = 3

RULE_1 = ['BE', 'IREN', 'OKLO', 'ONDS', 'RKLB', 'WOLF']
RULE_6 = RULE_1 + ['LUNR', 'FCEL']
STABLE_13 = ['MVIS', 'MXL', 'KTOS', 'JOBY', 'MP', 'LUNR', 'SMR',
             'FCEL', 'VOYG', 'WULF', 'IREN', 'ONDS', 'OKLO']

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


# ==============================================================================
# PHASE 1: LOAD BQ + OHLCV + SEED SCORES
# ==============================================================================
np.random.seed(42)

print(f"{'='*W}")
print(f"  SHORT: CONFIG SELECTION — 17 METHODS FOR PICKING BEST BQ CONFIG")
print(f"  Phase B: Given selected tickers, pick best config per ticker")
print(f"  TEST: {TEST_START} → {TEST_END}  |  H2: {HALF2_START} → {TEST_END}")
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

# ── Build all_ticker_configs ─────────────────────────────────────────────────
all_ticker_configs = {}

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

selected_tickers = sorted(all_ticker_configs.keys())
n_configs = [len(v) for v in all_ticker_configs.values()]
total_configs = sum(n_configs)
print(f"    ALL configs: {len(selected_tickers)} tickers, {total_configs} total configs")

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

# ── Load ALL folds' seed scores ─────────────────────────────────────────────
print(f"\n  PHASE 1C: Loading ALL folds' seed scores...")
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
# PHASE 2: DEFINE ORACLE TICKER GROUPS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 2: ORACLE TICKER GROUPS")
print(f"{'='*W}")

TICKER_GROUPS = {
    'RULE_1': [t for t in RULE_1 if t in tickers_ready],
    'RULE_6': [t for t in RULE_6 if t in tickers_ready],
    'STABLE_13': [t for t in STABLE_13 if t in tickers_ready],
}

for gn, gl in TICKER_GROUPS.items():
    print(f"  {gn}: {len(gl)} tickers — {', '.join(gl)}")


# ==============================================================================
# PHASE 3: EXHAUSTIVE ALL-CONFIGS H2 BACKTEST — GROUND TRUTH
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 3: EXHAUSTIVE ALL-CONFIGS H2 BACKTEST — GROUND TRUTH")
print(f"  Every BQ config + fixed universal per ticker → per-config H2 results")
print(f"{'='*W}")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h1_end_dt     = pd.Timestamp(HALF1_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

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
            'sl_mult': cfg.get('sl_mult', 0.0),
            'grace_minutes': cfg.get('grace_minutes', 0),
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

# Populate ALL_40 group
all_40 = []
for ticker, results in all_config_results.items():
    valid = [r for r in results if r['h2_trades'] >= MIN_H2_TRADES]
    if valid:
        all_40.append(ticker)
TICKER_GROUPS['ALL_40'] = sorted(all_40)

print(f"\n  Total backtests run: {total_bt_count}")
print(f"  Tickers with valid H2 data: {len(all_40)}")
for gn, gl in TICKER_GROUPS.items():
    print(f"  {gn}: {len(gl)} tickers")


# ==============================================================================
# PHASE 4: CONFIG SELECTION METHODS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 4: 17 CONFIG SELECTION METHODS")
print(f"  Each method picks ONE config per ticker using BQ-observable properties only")
print(f"{'='*W}")


def _find_matching_result(cfg, results):
    if cfg.get('config_id') == 'fixed_universal':
        for r in results:
            if r['config_id'] == 'fixed_universal':
                return r
        return None
    for r in results:
        if (r['fold'] == cfg.get('fold')
            and r['tpsl_mode'] == cfg.get('tpsl_mode')
            and abs(r['prob_thresh'] - cfg.get('prob_thresh', 0)) < 0.001
            and abs(r['tp_pct'] - cfg.get('tp_pct', 0)) < 0.0001):
            return r
    return None


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
        'opt_avg_pnl': row['opt_avg_pnl'],
        'config_id': f"f{int(row['fold'])}_{row['tpsl_mode']}_t{row['prob_thresh']:.2f}_tp{row['tp_pct']:.2f}",
    }


def _pick_best_by_column(t_df, col='val_a_avg_pnl'):
    if len(t_df) == 0:
        return None
    best_idx = t_df[col].idxmax()
    return _bq_row_to_cfg(t_df.loc[best_idx])


# ── M01: Best val_a (baseline) ──────────────────────────────────────────────
def select_m01_best_va(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[bq_df['ticker'] == ticker]
        if len(t_df) == 0:
            continue
        cfg = _pick_best_by_column(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── M02: S3 Composite ───────────────────────────────────────────────────────
def select_m02_s3_composite(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[bq_df['ticker'] == ticker].copy()
        if len(t_df) == 0:
            continue
        t_df['_score'] = (t_df['val_a_avg_pnl'] ** 2) * (t_df['val_a_avg_pnl'] >= 0.01).astype(float)
        valid = t_df[t_df['_score'] > 0]
        if len(valid) == 0:
            cfg = _pick_best_by_column(t_df, 'val_a_avg_pnl')
        else:
            cfg = _pick_best_by_column(valid, '_score')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── M03: Anti-overfit val_a ─────────────────────────────────────────────────
def select_m03_anti_overfit(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[bq_df['ticker'] == ticker].copy()
        if len(t_df) == 0:
            continue
        t_df['_score'] = t_df['val_a_avg_pnl'] * np.maximum(
            0, 1 - np.abs(t_df['va_opt_ratio'] - 1) * 0.5)
        valid = t_df[t_df['_score'] > 0]
        if len(valid) == 0:
            cfg = _pick_best_by_column(t_df, 'val_a_avg_pnl')
        else:
            cfg = _pick_best_by_column(valid, '_score')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── M04: Static only ────────────────────────────────────────────────────────
def select_m04_static_only(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['tpsl_mode'] == 'static')]
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker]
        cfg = _pick_best_by_column(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── M05: Grace only ─────────────────────────────────────────────────────────
def select_m05_grace_only(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['tpsl_mode'] == 'grace')]
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker]
        cfg = _pick_best_by_column(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── M06: Fold-403 best ──────────────────────────────────────────────────────
def select_m06_fold403(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'] == 403)]
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker]
        cfg = _pick_best_by_column(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── M07: Alpha-2 folds (400-405) ────────────────────────────────────────────
def select_m07_alpha2_folds(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'].isin(FOLDS_A2))]
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker]
        cfg = _pick_best_by_column(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── M08: Wide TP (≥8%) ──────────────────────────────────────────────────────
def select_m08_wide_tp(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['tp_pct'] >= 0.08)]
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker]
        cfg = _pick_best_by_column(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── M09: High thresh (≥0.45) ────────────────────────────────────────────────
def select_m09_high_thresh(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['prob_thresh'] >= 0.45)]
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker]
        cfg = _pick_best_by_column(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── M10: Combined S3-style composite ────────────────────────────────────────
def select_m10_combined_s3(bq_df, tickers):
    selections = {}
    mode_bonus = {'grace': 1.3, 'atr_sl': 1.1, 'static': 1.0}
    for ticker in tickers:
        t_df = bq_df[bq_df['ticker'] == ticker].copy()
        if len(t_df) == 0:
            continue
        t_df['_mb'] = t_df['tpsl_mode'].map(mode_bonus).fillna(1.0)
        t_df['_tp_bonus'] = 1.0 + 0.5 * np.maximum(0, (t_df['tp_pct'] - 0.08) / 0.04)
        t_df['_of_pen'] = np.maximum(0, 1 - np.abs(t_df['va_opt_ratio'] - 1) * 0.5)
        t_df['_score'] = (t_df['val_a_avg_pnl'] ** 2) * t_df['_mb'] * t_df['_tp_bonus'] * t_df['_of_pen']
        valid = t_df[t_df['_score'] > 0]
        if len(valid) == 0:
            cfg = _pick_best_by_column(t_df, 'val_a_avg_pnl')
        else:
            cfg = _pick_best_by_column(valid, '_score')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── M11: Fold consensus ─────────────────────────────────────────────────────
def select_m11_fold_consensus(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[bq_df['ticker'] == ticker]
        if len(t_df) == 0:
            continue
        group_stats = []
        for (mode, tp, thresh), grp in t_df.groupby(['tpsl_mode', 'tp_pct', 'prob_thresh']):
            n_folds_pos = (grp['val_a_avg_pnl'] > 0).sum()
            med_va = grp['val_a_avg_pnl'].median()
            group_stats.append({
                'key': (mode, tp, thresh),
                'n_folds_positive': n_folds_pos,
                'median_va': med_va,
            })
        if not group_stats:
            continue
        group_stats.sort(key=lambda x: (x['n_folds_positive'], x['median_va']), reverse=True)
        best_key = group_stats[0]['key']
        grp = t_df[(t_df['tpsl_mode'] == best_key[0]) &
                    (abs(t_df['tp_pct'] - best_key[1]) < 0.0001) &
                    (abs(t_df['prob_thresh'] - best_key[2]) < 0.001)]
        if len(grp) == 0:
            continue
        cfg = _pick_best_by_column(grp, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── M12: Median val_a across folds ──────────────────────────────────────────
def select_m12_median_va(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[bq_df['ticker'] == ticker]
        if len(t_df) == 0:
            continue
        group_stats = []
        for (mode, tp, thresh), grp in t_df.groupby(['tpsl_mode', 'tp_pct', 'prob_thresh']):
            med_va = grp['val_a_avg_pnl'].median()
            group_stats.append({
                'key': (mode, tp, thresh),
                'median_va': med_va,
                'n_folds': len(grp),
            })
        if not group_stats:
            continue
        group_stats.sort(key=lambda x: x['median_va'], reverse=True)
        best_key = group_stats[0]['key']
        grp = t_df[(t_df['tpsl_mode'] == best_key[0]) &
                    (abs(t_df['tp_pct'] - best_key[1]) < 0.0001) &
                    (abs(t_df['prob_thresh'] - best_key[2]) < 0.001)]
        if len(grp) == 0:
            continue
        cfg = _pick_best_by_column(grp, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── M13: Fixed universal ────────────────────────────────────────────────────
def select_m13_fixed_universal(tickers):
    return {ticker: dict(FIXED_UNIVERSAL_CFG) for ticker in tickers}


# ── M14: Oracle (ceiling, BIASED) ──────────────────────────────────────────
def select_m14_oracle(all_config_results, tickers):
    selections = {}
    for ticker in tickers:
        results = all_config_results.get(ticker, [])
        valid = [r for r in results if r['h2_trades'] >= MIN_H2_TRADES]
        if not valid:
            continue
        best = max(valid, key=lambda r: r['h2_avg'])
        selections[ticker] = best
    return selections


# ── ML Methods (M15-M17) ────────────────────────────────────────────────────
ML_FEATURE_COLS = ['fold_norm', 'is_static', 'is_grace', 'is_atr_sl',
                   'prob_thresh', 'tp_pct', 'sl_pct',
                   'opt_avg_pnl', 'val_a_avg_pnl', 'va_opt_ratio',
                   'opt_win_rate', 'val_a_win_rate']


def _prepare_ml_data(bq_df):
    ml_df = bq_df.copy()
    ml_df['fold_norm'] = (ml_df['fold'] - 400) / 15.0
    ml_df['is_static'] = (ml_df['tpsl_mode'] == 'static').astype(int)
    ml_df['is_grace'] = (ml_df['tpsl_mode'] == 'grace').astype(int)
    ml_df['is_atr_sl'] = (ml_df['tpsl_mode'] == 'atr_sl').astype(int)
    return ml_df


def _build_ml_target(ml_df, all_config_results):
    y_target = np.zeros(len(ml_df))
    for i, (_, row) in enumerate(ml_df.iterrows()):
        tk = row['ticker']
        results = all_config_results.get(tk, [])
        matched = [r for r in results
                   if r['fold'] == row['fold']
                   and r['tpsl_mode'] == row['tpsl_mode']
                   and abs(r['prob_thresh'] - row['prob_thresh']) < 0.001
                   and abs(r['tp_pct'] - row['tp_pct']) < 0.0001]
        if matched:
            y_target[i] = matched[0]['h2_avg']
    return y_target


def select_ml_loo(ml_df, y_target, tickers, model_cls, model_params):
    selections = {}
    ticker_col = ml_df['ticker'].values
    X_all = ml_df[ML_FEATURE_COLS].values.astype(np.float64)

    for test_ticker in tickers:
        train_mask = ticker_col != test_ticker
        test_mask = ticker_col == test_ticker

        X_train, y_train = X_all[train_mask], y_target[train_mask]
        X_test = X_all[test_mask]

        if len(X_train) < 10 or X_test.shape[0] == 0:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_cls(**model_params)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        best_local_idx = int(np.argmax(preds))
        global_idx = np.where(test_mask)[0][best_local_idx]
        row = ml_df.iloc[global_idx]

        selections[test_ticker] = _bq_row_to_cfg(row)
    return selections


# ── Run all methods ─────────────────────────────────────────────────────────
print(f"\n  Running 17 config selection methods...")

method_selections = {}

method_selections['M01_BEST_VA'] = select_m01_best_va(bq_df, tickers_ready)
print(f"    M01_BEST_VA: {len(method_selections['M01_BEST_VA'])} tickers")

method_selections['M02_S3_COMPOSITE'] = select_m02_s3_composite(bq_df, tickers_ready)
print(f"    M02_S3_COMPOSITE: {len(method_selections['M02_S3_COMPOSITE'])} tickers")

method_selections['M03_ANTI_OVERFIT'] = select_m03_anti_overfit(bq_df, tickers_ready)
print(f"    M03_ANTI_OVERFIT: {len(method_selections['M03_ANTI_OVERFIT'])} tickers")

method_selections['M04_STATIC_ONLY'] = select_m04_static_only(bq_df, tickers_ready)
print(f"    M04_STATIC_ONLY: {len(method_selections['M04_STATIC_ONLY'])} tickers")

method_selections['M05_GRACE_ONLY'] = select_m05_grace_only(bq_df, tickers_ready)
print(f"    M05_GRACE_ONLY: {len(method_selections['M05_GRACE_ONLY'])} tickers")

method_selections['M06_FOLD403'] = select_m06_fold403(bq_df, tickers_ready)
print(f"    M06_FOLD403: {len(method_selections['M06_FOLD403'])} tickers")

method_selections['M07_ALPHA2_FOLDS'] = select_m07_alpha2_folds(bq_df, tickers_ready)
print(f"    M07_ALPHA2_FOLDS: {len(method_selections['M07_ALPHA2_FOLDS'])} tickers")

method_selections['M08_WIDE_TP'] = select_m08_wide_tp(bq_df, tickers_ready)
print(f"    M08_WIDE_TP: {len(method_selections['M08_WIDE_TP'])} tickers")

method_selections['M09_HIGH_THRESH'] = select_m09_high_thresh(bq_df, tickers_ready)
print(f"    M09_HIGH_THRESH: {len(method_selections['M09_HIGH_THRESH'])} tickers")

method_selections['M10_COMBINED_S3'] = select_m10_combined_s3(bq_df, tickers_ready)
print(f"    M10_COMBINED_S3: {len(method_selections['M10_COMBINED_S3'])} tickers")

method_selections['M11_FOLD_CONSENSUS'] = select_m11_fold_consensus(bq_df, tickers_ready)
print(f"    M11_FOLD_CONSENSUS: {len(method_selections['M11_FOLD_CONSENSUS'])} tickers")

method_selections['M12_MEDIAN_VA'] = select_m12_median_va(bq_df, tickers_ready)
print(f"    M12_MEDIAN_VA: {len(method_selections['M12_MEDIAN_VA'])} tickers")

method_selections['M13_FIXED_UNIVERSAL'] = select_m13_fixed_universal(tickers_ready)
print(f"    M13_FIXED_UNIVERSAL: {len(method_selections['M13_FIXED_UNIVERSAL'])} tickers")

method_selections['M14_ORACLE'] = select_m14_oracle(all_config_results, tickers_ready)
print(f"    M14_ORACLE: {len(method_selections['M14_ORACLE'])} tickers")

if _HAS_SKLEARN:
    ml_df = _prepare_ml_data(bq_df)
    y_target = _build_ml_target(ml_df, all_config_results)

    method_selections['M15_RIDGE_LOO'] = select_ml_loo(
        ml_df, y_target, tickers_ready, Ridge, {'alpha': 1.0})
    print(f"    M15_RIDGE_LOO: {len(method_selections['M15_RIDGE_LOO'])} tickers")

    method_selections['M16_GB_LOO'] = select_ml_loo(
        ml_df, y_target, tickers_ready, GradientBoostingRegressor,
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42})
    print(f"    M16_GB_LOO: {len(method_selections['M16_GB_LOO'])} tickers")

    method_selections['M17_LASSO_LOO'] = select_ml_loo(
        ml_df, y_target, tickers_ready, Lasso, {'alpha': 0.001, 'max_iter': 10000})
    print(f"    M17_LASSO_LOO: {len(method_selections['M17_LASSO_LOO'])} tickers")
else:
    print(f"    M15-M17: SKIPPED (sklearn not available)")

METHOD_ORDER = [m for m in [
    'M01_BEST_VA', 'M02_S3_COMPOSITE', 'M03_ANTI_OVERFIT',
    'M04_STATIC_ONLY', 'M05_GRACE_ONLY', 'M06_FOLD403',
    'M07_ALPHA2_FOLDS', 'M08_WIDE_TP', 'M09_HIGH_THRESH',
    'M10_COMBINED_S3', 'M11_FOLD_CONSENSUS', 'M12_MEDIAN_VA',
    'M13_FIXED_UNIVERSAL', 'M14_ORACLE',
    'M15_RIDGE_LOO', 'M16_GB_LOO', 'M17_LASSO_LOO',
] if m in method_selections]


# ==============================================================================
# PHASE 5: EVALUATE ALL METHODS PER TICKER GROUP
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 5: METHOD EVALUATION PER TICKER GROUP")
print(f"{'='*W}")


def evaluate_method(method_name, selections, ticker_group):
    h2_avgs = []
    h2_trades_total = 0
    h2_totals = []
    h2_wrs = []
    per_ticker = []

    for ticker in ticker_group:
        cfg = selections.get(ticker)
        if cfg is None:
            continue
        results = all_config_results.get(ticker, [])
        matched = _find_matching_result(cfg, results)
        if matched and matched['h2_trades'] >= MIN_H2_TRADES:
            h2_avgs.append(matched['h2_avg'])
            h2_trades_total += matched['h2_trades']
            h2_totals.append(matched['h2_total'])
            h2_wrs.append(matched['h2_wr'])
            per_ticker.append({
                'ticker': ticker,
                'h2_avg': matched['h2_avg'],
                'h2_trades': matched['h2_trades'],
                'h2_total': matched['h2_total'],
                'h2_wr': matched['h2_wr'],
                'config': cfg,
                'matched': matched,
            })

    oracle_avgs = []
    for ticker in ticker_group:
        results = all_config_results.get(ticker, [])
        valid = [r for r in results if r['h2_trades'] >= MIN_H2_TRADES]
        if valid:
            oracle_avgs.append(max(r['h2_avg'] for r in valid))

    oracle_avg = np.mean(oracle_avgs) if oracle_avgs else 0.0
    method_avg = np.mean(h2_avgs) if h2_avgs else 0.0
    capture_pct = (method_avg / oracle_avg * 100) if oracle_avg > 0 else 0.0

    return {
        'method': method_name,
        'n_tickers': len(h2_avgs),
        'h2_avg': method_avg,
        'h2_trades': h2_trades_total,
        'h2_total': sum(h2_totals) if h2_totals else 0.0,
        'h2_wr': np.mean(h2_wrs) if h2_wrs else 0.0,
        'oracle_capture': capture_pct,
        'per_ticker': per_ticker,
    }


eval_results = {}

for group_name, group_tickers in TICKER_GROUPS.items():
    if not group_tickers:
        continue
    eval_results[group_name] = {}

    print(f"\n  ── {group_name} ({len(group_tickers)} tickers) ──")
    print(f"  {'Method':<22} {'Tks':>3} {'Trd':>5} {'H2avg%':>8} {'H2WR':>6} "
          f"{'TrdTot%':>8} {'Capture':>8}")
    print(f"  {'-'*65}")

    group_results = []
    for method_name in METHOD_ORDER:
        result = evaluate_method(method_name, method_selections[method_name], group_tickers)
        eval_results[group_name][method_name] = result
        group_results.append(result)

    group_results.sort(key=lambda x: x['h2_avg'], reverse=True)

    for r in group_results:
        deploy_mark = " ✓" if r['h2_avg'] >= DEPLOY_FLOOR else ""
        print(f"  {r['method']:<22} {r['n_tickers']:>3} {r['h2_trades']:>5} "
              f"{r['h2_avg']*100:>+7.2f}% {r['h2_wr']*100:>5.0f}% "
              f"{r['h2_total']*100:>+7.1f}% {r['oracle_capture']:>7.1f}%{deploy_mark}")

    best_non_oracle = [r for r in group_results if r['method'] != 'M14_ORACLE']
    if best_non_oracle:
        best = best_non_oracle[0]
        if best['h2_avg'] >= DEPLOY_FLOOR:
            print(f"\n  >>> DEPLOY-READY: {best['method']} achieves "
                  f"{best['h2_avg']*100:+.2f}% on {group_name}")
        else:
            print(f"\n  >>> Below deploy threshold ({DEPLOY_FLOOR*100:.1f}%): "
                  f"best non-oracle = {best['method']} at {best['h2_avg']*100:+.2f}%")


# ==============================================================================
# PHASE 6: TOP-3 DEEP DIVE
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 6: TOP-3 METHOD DEEP DIVE")
print(f"{'='*W}")

for group_name in ['RULE_1', 'RULE_6']:
    if group_name not in eval_results:
        continue
    group_tickers = TICKER_GROUPS[group_name]
    ge = eval_results[group_name]

    ranked = sorted(
        [(m, r) for m, r in ge.items() if m != 'M14_ORACLE'],
        key=lambda x: x[1]['h2_avg'], reverse=True)

    oracle_result = ge.get('M14_ORACLE', {})

    print(f"\n  ── {group_name}: Top 3 Methods (vs Oracle) ──")

    for rank, (method_name, result) in enumerate(ranked[:3]):
        print(f"\n  {'─'*80}")
        print(f"  #{rank+1} {method_name}: H2 avg = {result['h2_avg']*100:+.2f}%, "
              f"Oracle capture = {result['oracle_capture']:.1f}%")
        print(f"  {'─'*80}")

        print(f"\n  {'Ticker':<8} {'Selected Config':<35} {'H2avg%':>8} {'Orcl%':>8} {'Gap':>8} {'Trd':>5}")
        print(f"  {'-'*78}")

        oracle_sels = method_selections.get('M14_ORACLE', {})

        for pt in sorted(result['per_ticker'], key=lambda x: x['h2_avg'], reverse=True):
            ticker = pt['ticker']
            cfg = pt['config']
            cfg_str = f"{cfg.get('tpsl_mode', '?')}/TP{cfg.get('tp_pct',0)*100:.0f}/f{cfg.get('fold','?')}"

            orc = oracle_sels.get(ticker, {})
            orc_avg = orc.get('h2_avg', 0.0) if orc else 0.0
            gap = pt['h2_avg'] - orc_avg

            match_str = "MATCH" if abs(gap) < 0.001 else ""
            print(f"  {ticker:<8} {cfg_str:<35} {pt['h2_avg']*100:>+7.2f}% "
                  f"{orc_avg*100:>+7.2f}% {gap*100:>+7.2f}% {pt['h2_trades']:>5} {match_str}")

        modes = Counter(pt['config'].get('tpsl_mode', '?') for pt in result['per_ticker'])
        tps = [pt['config'].get('tp_pct', 0) for pt in result['per_ticker']]
        thrs = [pt['config'].get('prob_thresh', 0) for pt in result['per_ticker']]
        folds = Counter(pt['config'].get('fold', '?') for pt in result['per_ticker'])
        print(f"\n    Config properties: modes={dict(modes)}, "
              f"median_TP={np.median(tps)*100:.1f}%, "
              f"median_thresh={np.median(thrs):.2f}, folds={dict(folds)}")

        n_matches = sum(1 for pt in result['per_ticker']
                        if abs(pt['h2_avg'] - oracle_sels.get(pt['ticker'], {}).get('h2_avg', -999)) < 0.001)
        print(f"    Oracle match rate: {n_matches}/{len(result['per_ticker'])}")


# ==============================================================================
# PHASE 7: GRAPHS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 7: VISUALIZATIONS")
print(f"{'='*W}")

# ── 7A: Method comparison bars per group ────────────────────────────────────
n_groups = len([g for g in TICKER_GROUPS if g in eval_results and eval_results[g]])
fig, axes = plt.subplots(n_groups, 1, figsize=(14, 5 * n_groups))
if n_groups == 1:
    axes = [axes]
fig.suptitle('Phase 5: Method H2 Avg PnL by Ticker Group', fontsize=14, fontweight='bold')

for ax_idx, (group_name, ge) in enumerate(eval_results.items()):
    if not ge:
        continue
    ax = axes[ax_idx]
    sorted_results = sorted(ge.values(), key=lambda r: r['h2_avg'])
    method_names = [r['method'] for r in sorted_results]
    method_avgs = [r['h2_avg'] * 100 for r in sorted_results]
    colors = ['#2ecc71' if r['method'] == 'M14_ORACLE' else
              '#f39c12' if r['h2_avg'] >= DEPLOY_FLOOR else
              '#3498db' if r['h2_avg'] > 0 else '#e74c3c'
              for r in sorted_results]

    ax.barh(range(len(method_names)), method_avgs, color=colors, edgecolor='white')
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=8)
    ax.axvline(x=DEPLOY_FLOOR * 100, color='red', linestyle='--', alpha=0.7,
               label=f'Deploy ≥{DEPLOY_FLOOR*100}%')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('H2 Avg PnL %')
    ax.set_title(f'{group_name} ({len(TICKER_GROUPS.get(group_name, []))} tickers)')
    ax.legend(fontsize=8)
    for bar, val in zip(ax.patches, method_avgs):
        if abs(val) > 0.1:
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{val:+.2f}%', va='center', fontsize=7)

plt.tight_layout()
plt.show()

# ── 7B: Method vs Oracle scatter (top 3 methods on RULE_1) ──────────────────
if 'RULE_1' in eval_results:
    ge = eval_results['RULE_1']
    ranked = sorted(
        [(m, r) for m, r in ge.items() if m != 'M14_ORACLE' and r['per_ticker']],
        key=lambda x: x[1]['h2_avg'], reverse=True)

    n_top = min(3, len(ranked))
    if n_top > 0:
        fig, axes = plt.subplots(1, n_top, figsize=(6 * n_top, 5))
        if n_top == 1:
            axes = [axes]
        fig.suptitle('Method vs Oracle H2 Avg PnL (RULE_1)', fontsize=14, fontweight='bold')

        oracle_sels = method_selections.get('M14_ORACLE', {})

        for ax_idx, (method_name, result) in enumerate(ranked[:n_top]):
            ax = axes[ax_idx]
            x_vals = []
            y_vals = []
            labels = []

            for pt in result['per_ticker']:
                orc = oracle_sels.get(pt['ticker'], {})
                orc_avg = orc.get('h2_avg', 0.0) if orc else 0.0
                x_vals.append(orc_avg * 100)
                y_vals.append(pt['h2_avg'] * 100)
                labels.append(pt['ticker'])

            if x_vals:
                mn = min(min(x_vals), min(y_vals)) - 1
                mx = max(max(x_vals), max(y_vals)) + 1
                ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.3, label='Perfect')
                sc_colors = ['#2ecc71' if y >= DEPLOY_FLOOR * 100 else '#3498db' if y > 0 else '#e74c3c'
                             for y in y_vals]
                ax.scatter(x_vals, y_vals, c=sc_colors, s=80, edgecolors='white', zorder=5)
                for x, y, lbl in zip(x_vals, y_vals, labels):
                    ax.annotate(lbl, (x, y), fontsize=8, alpha=0.8,
                               textcoords='offset points', xytext=(5, 5))
                ax.axhline(y=0, color='gray', linewidth=0.3)
                ax.axvline(x=0, color='gray', linewidth=0.3)
                ax.set_xlabel('Oracle H2 %')
                ax.set_ylabel('Method H2 %')
                ax.set_title(f'{method_name}\ncapture={result["oracle_capture"]:.0f}%')
                ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

# ── 7C: Top method per-ticker breakdown (RULE_1) ───────────────────────────
if 'RULE_1' in eval_results:
    ge = eval_results['RULE_1']
    ranked = sorted(
        [(m, r) for m, r in ge.items() if m != 'M14_ORACLE' and r['per_ticker']],
        key=lambda x: x[1]['h2_avg'], reverse=True)

    if ranked:
        best_method, best_result = ranked[0]
        oracle_sels = method_selections.get('M14_ORACLE', {})

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f'RULE_1: {best_method} vs Oracle per Ticker', fontsize=13, fontweight='bold')

        tickers_sorted = sorted(best_result['per_ticker'], key=lambda x: x['h2_avg'], reverse=True)
        ticker_names = [pt['ticker'] for pt in tickers_sorted]
        method_vals = [pt['h2_avg'] * 100 for pt in tickers_sorted]
        oracle_vals = []
        for pt in tickers_sorted:
            orc = oracle_sels.get(pt['ticker'], {})
            oracle_vals.append((orc.get('h2_avg', 0.0) if orc else 0.0) * 100)

        y_pos = np.arange(len(ticker_names))
        bar_h = 0.35
        ax.barh(y_pos + bar_h/2, oracle_vals, bar_h, label='Oracle', color='#2ecc71', alpha=0.7)
        ax.barh(y_pos - bar_h/2, method_vals, bar_h, label=best_method, color='#3498db', alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ticker_names, fontsize=10)
        ax.axvline(x=DEPLOY_FLOOR * 100, color='red', linestyle='--', alpha=0.7,
                   label=f'Deploy ≥{DEPLOY_FLOOR*100}%')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('H2 Avg PnL %')

        for i, (mv, ov) in enumerate(zip(method_vals, oracle_vals)):
            gap = mv - ov
            ax.text(max(mv, ov) + 0.3, i, f'gap={gap:+.1f}%', va='center', fontsize=8, alpha=0.7)

        ax.legend(fontsize=9)
        ax.set_title(f'Method avg={np.mean(method_vals):+.2f}%, '
                     f'Oracle avg={np.mean(oracle_vals):+.2f}%, '
                     f'Capture={best_result["oracle_capture"]:.0f}%')
        plt.tight_layout()
        plt.show()

# ── 7D: Config property distributions (top method vs oracle) ────────────────
if 'RULE_1' in eval_results:
    ge = eval_results['RULE_1']
    ranked = sorted(
        [(m, r) for m, r in ge.items() if m != 'M14_ORACLE' and r['per_ticker']],
        key=lambda x: x[1]['h2_avg'], reverse=True)

    oracle_result = ge.get('M14_ORACLE', {})

    if ranked and oracle_result and oracle_result.get('per_ticker'):
        best_method, best_result = ranked[0]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'Config Properties: {best_method} vs Oracle (RULE_1)',
                     fontsize=13, fontweight='bold')

        method_modes = [pt['config'].get('tpsl_mode', '?') for pt in best_result['per_ticker']]
        oracle_modes = [pt['matched'].get('tpsl_mode', pt['config'].get('tpsl_mode', '?'))
                       for pt in oracle_result['per_ticker']]

        ax = axes[0]
        all_modes_set = sorted(set(method_modes + oracle_modes))
        m_counts = [method_modes.count(m) for m in all_modes_set]
        o_counts = [oracle_modes.count(m) for m in all_modes_set]
        x_pos = np.arange(len(all_modes_set))
        ax.bar(x_pos - 0.2, m_counts, 0.35, label=best_method, color='#3498db')
        ax.bar(x_pos + 0.2, o_counts, 0.35, label='Oracle', color='#2ecc71')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_modes_set)
        ax.set_ylabel('Count')
        ax.set_title('Mode Distribution')
        ax.legend(fontsize=8)

        ax = axes[1]
        method_tps = [pt['config'].get('tp_pct', 0) * 100 for pt in best_result['per_ticker']]
        oracle_tps = [pt['matched'].get('tp_pct', pt['config'].get('tp_pct', 0)) * 100
                     for pt in oracle_result['per_ticker']]
        bins = np.arange(0, 20, 2)
        ax.hist(method_tps, bins=bins, alpha=0.6, label=best_method, color='#3498db')
        ax.hist(oracle_tps, bins=bins, alpha=0.6, label='Oracle', color='#2ecc71')
        ax.set_xlabel('TP %')
        ax.set_ylabel('Count')
        ax.set_title('TP Distribution')
        ax.legend(fontsize=8)

        ax = axes[2]
        method_thrs = [pt['config'].get('prob_thresh', 0) for pt in best_result['per_ticker']]
        oracle_thrs = [pt['matched'].get('prob_thresh', pt['config'].get('prob_thresh', 0))
                      for pt in oracle_result['per_ticker']]
        bins_t = np.arange(0, 0.8, 0.05)
        ax.hist(method_thrs, bins=bins_t, alpha=0.6, label=best_method, color='#3498db')
        ax.hist(oracle_thrs, bins=bins_t, alpha=0.6, label='Oracle', color='#2ecc71')
        ax.set_xlabel('Prob Threshold')
        ax.set_ylabel('Count')
        ax.set_title('Threshold Distribution')
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  FINAL SUMMARY")
print(f"{'='*W}")

for group_name, ge in eval_results.items():
    if not ge:
        continue
    oracle_r = ge.get('M14_ORACLE', {})
    oracle_avg = oracle_r.get('h2_avg', 0) if oracle_r else 0

    best_non_oracle = sorted(
        [(m, r) for m, r in ge.items() if m != 'M14_ORACLE'],
        key=lambda x: x[1]['h2_avg'], reverse=True)

    print(f"\n  {group_name}:")
    print(f"    Oracle ceiling: {oracle_avg*100:+.2f}%")
    if best_non_oracle:
        bm, br = best_non_oracle[0]
        deployable = "DEPLOY-READY" if br['h2_avg'] >= DEPLOY_FLOOR else "below threshold"
        print(f"    Best method: {bm} at {br['h2_avg']*100:+.2f}% "
              f"(capture={br['oracle_capture']:.0f}%) — {deployable}")
        print(f"    H2 trades: {br['h2_trades']}, WR: {br['h2_wr']*100:.0f}%, "
              f"Trade total: {br['h2_total']*100:+.1f}%")

    print(f"    Methods above deploy ({DEPLOY_FLOOR*100:.1f}%):")
    for m, r in best_non_oracle:
        if r['h2_avg'] >= DEPLOY_FLOOR:
            print(f"      {m}: {r['h2_avg']*100:+.2f}%, {r['h2_trades']} trades, "
                  f"capture={r['oracle_capture']:.0f}%")

print(f"\n{'='*W}")
print(f"  ANALYSIS COMPLETE")
print(f"  Methods tested: {len(METHOD_ORDER)}")
print(f"  Groups evaluated: {list(eval_results.keys())}")
print(f"{'='*W}")
