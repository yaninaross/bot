# %% ── SHORT Oracle on TEST Period (DEPLOYABLE) ───────────────────────────────
#
# Defines the absolute ceiling (TRUE ORACLE) for SHORT trading on TEST period.
# Exhaustively backtests ALL BQ configs (val_a filters REMOVED — val_a is
# anti-signal for SHORT r=-0.27) across ALL 12 walkforward folds
# (400-405 α=2.0, 410-415 α=1.0) for EVERY ticker.
#
# Oracle = best DEPLOYABLE config per ticker (avg PnL >= 1.8%, min 3 trades)
# selected with perfect foresight on TEST. Picks by total PnL among deployable
# configs. Falls back to best avg PnL if no config meets deploy threshold.
# This is the benchmark ceiling — no real strategy can beat it.
#
# Outputs:
#   /tmp/oracle_short_per_config.csv   — every config×ticker backtest result
#   /tmp/oracle_short_per_ticker.csv   — per-ticker ceiling + best config
#   /tmp/oracle_short_trades.csv       — individual trades (for strategy comparison)
#   /tmp/oracle_short_test.png         — 4-panel summary chart
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

FOLD_EXPERIMENTS   = {f: f'{f}_XSECTOR' for f in ALL_FOLDS}
FOLD_MODEL_VERSIONS = {f: f'robust_wf_v12_short_{f}' for f in ALL_FOLDS}

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_ALLOWED_MODES = ('static', 'atr_sl', 'grace')

DEPLOY_PNL_THRESHOLD = 0.018
MIN_ORACLE_TRADES    = 3

FILTERED_VA_OPT_RATIO = 2.0

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110

OUTPUT_PER_CONFIG = '/tmp/oracle_short_per_config.csv'
OUTPUT_PER_TICKER = '/tmp/oracle_short_per_ticker.csv'
OUTPUT_TRADES     = '/tmp/oracle_short_trades.csv'
OUTPUT_CHART      = '/tmp/oracle_short_test.png'


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
# HELPERS
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


# ==============================================================================
# PHASE 1: QUERY BQ — ALL CONFIGS ACROSS ALL 12 FOLDS
# ==============================================================================
print("=" * W)
print("  SHORT ORACLE — TEST PERIOD CEILING (12 folds)")
print(f"  TEST: {TEST_START} → {TEST_END}")
print(f"  HALF1: {TEST_START} → {HALF1_END}  |  HALF2: {HALF2_START} → {TEST_END}")
print("=" * W)

print(f"\n  PHASE 1: Querying BQ for all configs across 12 folds...")
mv_str = ", ".join(f"'{FOLD_MODEL_VERSIONS[f]}'" for f in ALL_FOLDS)
modes_str = ", ".join(f"'{m}'" for m in BQ_ALLOWED_MODES)

q = f"""
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       grace_minutes, vol_filter_pct, pre_range_filter_pct,
       opt_avg_pnl, opt_win_rate, opt_count, opt_freq,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       is_ensemble, survived_a, model_version
FROM `{BQ_TABLE}`
WHERE model_version IN ({mv_str})
  AND opt_avg_pnl > 0
  AND tpsl_mode IN ({modes_str})
ORDER BY ticker, experiment, seed
"""
bq_df = bq_client.query(q).to_dataframe()
bq_df['fold'] = bq_df['model_version'].str.extract(r'_(\d+)$').astype(int)

n_tickers = bq_df['ticker'].nunique()
n_indiv = len(bq_df[bq_df['is_ensemble'] == False])
n_ens = len(bq_df[bq_df['is_ensemble'] == True])
print(f"    {len(bq_df)} BQ rows: {n_indiv} individual + {n_ens} ensemble")
print(f"    {n_tickers} unique tickers across {bq_df['fold'].nunique()} folds")
for f in ALL_FOLDS:
    nf = len(bq_df[bq_df['fold'] == f])
    nt = bq_df[bq_df['fold'] == f]['ticker'].nunique()
    if nf > 0:
        print(f"      Fold {f}: {nf:4d} configs, {nt:2d} tickers")


# ==============================================================================
# PHASE 2: LOAD ALL DATA
# ==============================================================================
print(f"\n  PHASE 2: Loading data for {n_tickers} tickers...")
all_tickers = sorted(bq_df['ticker'].unique())

ticker_data = {}
for i, ticker in enumerate(all_tickers):
    df = load_master(ticker)
    if df is None:
        continue
    df = heal_data(df)
    ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
    if ohlcv is None:
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
    ticker_data[ticker] = {'ohlcv': ohlcv}
    del df
    if (i + 1) % 10 == 0:
        _release_memory()

print(f"    Loaded OHLCV for {len(ticker_data)}/{len(all_tickers)} tickers")

# Load CSV seed scores for all (ticker, base_experiment) pairs
print(f"    Loading CSV scores...")
csv_scores_cache = {}
all_exp_pairs = set()
for _, row in bq_df.iterrows():
    base_exp = _base_experiment(row['experiment'])
    all_exp_pairs.add((row['ticker'], base_exp))

n_loaded = 0
for ticker, base_exp in sorted(all_exp_pairs):
    if ticker not in ticker_data:
        continue
    if (ticker, base_exp) in csv_scores_cache:
        continue
    seed_scores = load_seed_scores(ticker, base_exp)
    if seed_scores:
        csv_scores_cache[(ticker, base_exp)] = seed_scores
        n_loaded += 1
        if n_loaded % 20 == 0:
            print(f"      {n_loaded} loaded...")

print(f"    Loaded CSV scores for {n_loaded} ticker×experiment pairs")
_release_memory()


# ==============================================================================
# PHASE 3: EXHAUSTIVE BACKTEST — EVERY CONFIG ON TEST
# ==============================================================================
print(f"\n  PHASE 3: Exhaustive backtest ({len(bq_df)} configs × {len(ticker_data)} tickers)...")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h1_end_dt     = pd.Timestamp(HALF1_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

config_results = []
all_trades = []
tickers_done = 0

for ticker in all_tickers:
    if ticker not in ticker_data:
        continue

    ohlcv = ticker_data[ticker]['ohlcv']
    ticker_rows = bq_df[bq_df['ticker'] == ticker]

    # Precompute daily ensemble probs per (base_exp, entry_minute)
    prob_cache = {}
    seen_keys = set()
    for _, r in ticker_rows[['experiment', 'entry_minute']].drop_duplicates().iterrows():
        base_exp = _base_experiment(r['experiment'])
        em_int = _safe_int(r['entry_minute'], ENTRY_MIN)
        key = (base_exp, em_int)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        seed_scores = csv_scores_cache.get((ticker, base_exp))
        if not seed_scores:
            continue
        prob_cache[key] = compute_daily_prob(seed_scores, em_int)

    # Precompute entry bar indices
    entry_idx_map = {}
    for em in ticker_rows['entry_minute'].dropna().unique():
        em_int = _safe_int(em, ENTRY_MIN)
        entry_idx_map[em_int] = np.where(ohlcv['times'] == em_int)[0]

    n_bt = 0
    for cfg_idx, bq_row in ticker_rows.iterrows():
        exp = bq_row['experiment']
        base_exp = _base_experiment(exp)
        em_int = _safe_int(bq_row.get('entry_minute', ENTRY_MIN), ENTRY_MIN)
        mode = bq_row['tpsl_mode']
        fold = int(bq_row['fold'])

        probs = prob_cache.get((base_exp, em_int))
        if not probs:
            continue

        entry_indices = entry_idx_map.get(em_int, np.array([]))
        if len(entry_indices) == 0:
            continue

        prob_thresh = _safe_float(bq_row['prob_thresh'])
        tp_pct      = _safe_float(bq_row.get('tp_pct', 0))
        sl_pct      = _safe_float(bq_row.get('sl_pct', 0))
        sl_mult     = _safe_float(bq_row.get('sl_mult', 0))
        grace_min   = _safe_int(bq_row.get('grace_minutes', 0))

        trades = []
        last_td = -1

        for idx in entry_indices:
            d_int = int(ohlcv['dates'][idx])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()

            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            if d_int == last_td:
                continue

            prob = probs.get(d_date)
            if prob is None or prob < prob_thresh:
                continue

            pnl, td, filled = execute_trade(
                ohlcv, idx, mode, tp_pct, sl_pct, sl_mult, grace_min)

            if not filled:
                continue

            last_td = d_int
            trades.append({
                'ticker': ticker,
                'date': str(d_date),
                'pnl': pnl,
                'fold': fold,
                'config_idx': cfg_idx,
                'mode': mode,
                'prob': prob,
            })

        # Summarize this config
        t_h1 = [t for t in trades if t['date'] <= str(h1_end_dt)]
        t_h2 = [t for t in trades if t['date'] >= str(h2_start_dt)]

        def _s(tl):
            if not tl:
                return 0, 0.0, 0.0, 0.0
            pnls = [t['pnl'] for t in tl]
            return len(tl), sum(pnls), np.mean(pnls), float((np.array(pnls) > 0).mean())

        nf, tf_tot, af, wf = _s(trades)
        n1, t1_tot, a1, w1 = _s(t_h1)
        n2, t2_tot, a2, w2 = _s(t_h2)

        opt_pnl = _safe_float(bq_row['opt_avg_pnl'])
        va_pnl  = _safe_float(bq_row['val_a_avg_pnl'])
        va_opt_ratio = va_pnl / opt_pnl if opt_pnl > 0 else 0

        config_results.append({
            'ticker': ticker,
            'fold': fold,
            'experiment': exp,
            'seed': int(bq_row['seed']) if not pd.isna(bq_row['seed']) else -1,
            'is_ensemble': bool(bq_row.get('is_ensemble', False)),
            'model_version': bq_row['model_version'],
            'tpsl_mode': mode,
            'entry_minute': em_int,
            'prob_thresh': prob_thresh,
            'tp_pct': tp_pct, 'sl_pct': sl_pct,
            'sl_mult': sl_mult, 'grace_minutes': grace_min,
            'opt_avg_pnl': opt_pnl,
            'val_a_avg_pnl': va_pnl,
            'va_opt_ratio': va_opt_ratio,
            'bq_vb_avg_pnl': _safe_float(bq_row.get('val_b_avg_pnl')),
            'bt_full_n': nf, 'bt_full_total': tf_tot, 'bt_full_avg': af, 'bt_full_wr': wf,
            'bt_h1_n': n1, 'bt_h1_total': t1_tot, 'bt_h1_avg': a1, 'bt_h1_wr': w1,
            'bt_h2_n': n2, 'bt_h2_total': t2_tot, 'bt_h2_avg': a2, 'bt_h2_wr': w2,
        })
        n_bt += 1

    tickers_done += 1
    if n_bt > 0:
        print(f"      [{tickers_done:2d}/{len(ticker_data)}] {ticker}: {n_bt} configs backtested")

results_df = pd.DataFrame(config_results)
print(f"\n    Total: {len(results_df)} configs backtested across {results_df['ticker'].nunique()} tickers")

results_df.to_csv(OUTPUT_PER_CONFIG, index=False)
print(f"    Saved: {OUTPUT_PER_CONFIG}")
_release_memory()


# ==============================================================================
# PHASE 4: PER-TICKER ORACLE CEILING
# ==============================================================================
print(f"\n  PHASE 4: Computing per-ticker TRUE ORACLE ceiling...")

per_ticker = []
oracle_trade_rows = []

for ticker in sorted(results_df['ticker'].unique()):
    grp = results_df[results_df['ticker'] == ticker]
    n_cfg = len(grp)
    if n_cfg == 0:
        continue

    n_pos_full = (grp['bt_full_avg'] > 0).sum()

    # TRUE ORACLE: best config by FULL test avg PnL (with deploy floor)
    deployable = grp[(grp['bt_full_avg'] >= DEPLOY_PNL_THRESHOLD) &
                     (grp['bt_full_n'] >= MIN_ORACLE_TRADES)]
    if not deployable.empty:
        best_full = deployable.loc[deployable['bt_full_total'].idxmax()]
    else:
        best_full = grp.loc[grp['bt_full_avg'].idxmax()]

    # Best by HALF2 only (recent out-of-sample)
    h2_with_trades = grp[grp['bt_h2_n'] > 0]
    if not h2_with_trades.empty:
        best_h2 = h2_with_trades.loc[h2_with_trades['bt_h2_avg'].idxmax()]
    else:
        best_h2 = best_full

    # ENSEMBLE selection (what we'd pick without oracle — best val_a ensemble)
    ens = grp[(grp['is_ensemble'] == True) &
              (grp['va_opt_ratio'] <= FILTERED_VA_OPT_RATIO)]
    if not ens.empty:
        ens_best = ens.loc[ens['val_a_avg_pnl'].idxmax()]
        ens_full = float(ens_best['bt_full_avg'])
        ens_h1   = float(ens_best['bt_h1_avg'])
        ens_h2   = float(ens_best['bt_h2_avg'])
        ens_n    = int(ens_best['bt_full_n'])
        ens_fold = int(ens_best['fold'])
    else:
        ens_full = ens_h1 = ens_h2 = 0.0
        ens_n = 0
        ens_fold = -1

    per_ticker.append({
        'ticker': ticker,
        'n_configs': n_cfg,
        'n_positive_full': n_pos_full,
        'pct_positive': n_pos_full / n_cfg,
        # TRUE ORACLE (best by FULL test PnL)
        'oracle_full_avg': float(best_full['bt_full_avg']),
        'oracle_full_wr': float(best_full['bt_full_wr']),
        'oracle_full_n': int(best_full['bt_full_n']),
        'oracle_full_total': float(best_full['bt_full_total']),
        'oracle_h1_avg': float(best_full['bt_h1_avg']),
        'oracle_h2_avg': float(best_full['bt_h2_avg']),
        'oracle_fold': int(best_full['fold']),
        'oracle_mode': best_full['tpsl_mode'],
        'oracle_tp': float(best_full['tp_pct']),
        'oracle_sl': float(best_full['sl_pct']),
        'oracle_sl_mult': float(best_full['sl_mult']),
        'oracle_prob': float(best_full['prob_thresh']),
        'oracle_exp': best_full['experiment'],
        # Best by HALF2
        'best_h2_avg': float(best_h2['bt_h2_avg']),
        'best_h2_fold': int(best_h2['fold']),
        # Ensemble selection (unbiased — what we'd actually pick)
        'ens_full_avg': ens_full,
        'ens_h1_avg': ens_h1,
        'ens_h2_avg': ens_h2,
        'ens_n': ens_n,
        'ens_fold': ens_fold,
        # Gap
        'gap_full': float(best_full['bt_full_avg']) - ens_full,
    })

    # Re-run the oracle config to get per-trade data
    ohlcv = ticker_data[ticker]['ohlcv']
    oracle_mode = best_full['tpsl_mode']
    oracle_tp   = float(best_full['tp_pct'])
    oracle_sl   = float(best_full['sl_pct'])
    oracle_sm   = float(best_full['sl_mult'])
    oracle_gm   = int(best_full['grace_minutes'])
    oracle_pt   = float(best_full['prob_thresh'])
    oracle_fold = int(best_full['fold'])
    oracle_exp  = _base_experiment(best_full['experiment'])
    oracle_em   = int(best_full['entry_minute'])

    probs = prob_cache_key = None
    seed_scores = csv_scores_cache.get((ticker, oracle_exp))
    if seed_scores:
        probs = compute_daily_prob(seed_scores, oracle_em)

    if probs:
        entry_indices = np.where(ohlcv['times'] == oracle_em)[0]
        last_td = -1
        for idx in entry_indices:
            d_int = int(ohlcv['dates'][idx])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            if d_int == last_td:
                continue
            prob = probs.get(d_date)
            if prob is None or prob < oracle_pt:
                continue
            pnl, td, filled = execute_trade(
                ohlcv, idx, oracle_mode, oracle_tp, oracle_sl, oracle_sm, oracle_gm)
            if not filled:
                continue
            last_td = d_int
            period = 'HALF1' if d_date <= h1_end_dt else 'HALF2'
            oracle_trade_rows.append({
                'ticker': ticker,
                'date': str(d_date),
                'pnl': pnl,
                'period': period,
                'fold': oracle_fold,
                'mode': oracle_mode,
                'tp_pct': oracle_tp,
                'sl_pct': oracle_sl,
                'prob': prob,
                'strategy': 'ORACLE',
            })

ceiling_df = pd.DataFrame(per_ticker).sort_values('oracle_full_avg', ascending=False)
ceiling_df.to_csv(OUTPUT_PER_TICKER, index=False)

trades_df = pd.DataFrame(oracle_trade_rows)
trades_df.to_csv(OUTPUT_TRADES, index=False)

print(f"    Saved: {OUTPUT_PER_TICKER} ({len(ceiling_df)} tickers)")
print(f"    Saved: {OUTPUT_TRADES} ({len(trades_df)} trades)")


# ==============================================================================
# PHASE 5: RESULTS OUTPUT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 5: ORACLE RESULTS")
print(f"{'='*W}")

n_total = len(ceiling_df)
n_oracle_pos = (ceiling_df['oracle_full_avg'] > 0).sum()
n_ens_pos    = (ceiling_df['ens_full_avg'] > 0).sum()
n_oracle_strong = (ceiling_df['oracle_full_avg'] > 0.01).sum()
n_deployable = (ceiling_df['oracle_full_avg'] >= DEPLOY_PNL_THRESHOLD).sum()

print(f"\n  ── SUMMARY ──")
print(f"    {n_total} tickers with backtest data (val_a filters REMOVED)")
print(f"    {n_oracle_pos} ({100*n_oracle_pos/n_total:.0f}%) have ≥1 profitable config (TRUE ORACLE)")
print(f"    {n_deployable} ({100*n_deployable/n_total:.0f}%) DEPLOYABLE — oracle avg >= {DEPLOY_PNL_THRESHOLD*100:.1f}%")
print(f"    {n_ens_pos} ({100*n_ens_pos/n_total:.0f}%) profitable with ensemble selection (unbiased)")
print(f"    {n_oracle_strong} ({100*n_oracle_strong/n_total:.0f}%) have oracle PnL > +1.0%")
print(f"    Gap: {n_oracle_pos - n_ens_pos} additional tickers profitable with oracle config")
print(f"    Avg oracle ceiling (all):        {ceiling_df['oracle_full_avg'].mean()*100:+.2f}%")
deploy_mask = ceiling_df['oracle_full_avg'] >= DEPLOY_PNL_THRESHOLD
if deploy_mask.any():
    print(f"    Avg oracle ceiling (deployable): {ceiling_df.loc[deploy_mask, 'oracle_full_avg'].mean()*100:+.2f}%")
print(f"    Avg ensemble PnL:   {ceiling_df['ens_full_avg'].mean()*100:+.2f}%")
print(f"    Avg config gap:     {ceiling_df['gap_full'].mean()*100:+.2f}%")

# ── DEPLOYABLE ORACLE PORTFOLIO (avg >= 1.8% deploy threshold) ──
oracle_deploy = ceiling_df[ceiling_df['oracle_full_avg'] >= DEPLOY_PNL_THRESHOLD]
if not trades_df.empty and not oracle_deploy.empty:
    deploy_trades = trades_df[trades_df['ticker'].isin(oracle_deploy['ticker'])]
    n_dt = len(deploy_trades)
    if n_dt > 0:
        d_avg = deploy_trades['pnl'].mean()
        d_total = deploy_trades['pnl'].sum()
        d_wr = (deploy_trades['pnl'] > 0).mean()
        print(f"\n  ── DEPLOYABLE ORACLE PORTFOLIO (avg >= {DEPLOY_PNL_THRESHOLD*100:.1f}%) ──")
        print(f"    Tickers:    {len(oracle_deploy)}")
        print(f"    Trades:     {n_dt}")
        print(f"    Avg PnL:    {d_avg*100:+.2f}%")
        print(f"    Win Rate:   {d_wr*100:.0f}%")
        print(f"    Total PnL:  {d_total*100:+.1f}%")
        print(f"    Tickers:    {', '.join(oracle_deploy['ticker'].values)}")

# ── ALL ORACLE PORTFOLIO (all oracle-positive tickers) ──
oracle_pos = ceiling_df[ceiling_df['oracle_full_avg'] > 0]
if not trades_df.empty:
    oracle_trades = trades_df[trades_df['ticker'].isin(oracle_pos['ticker'])]
    n_trades = len(oracle_trades)
    if n_trades > 0:
        avg_pnl = oracle_trades['pnl'].mean()
        total_pnl = oracle_trades['pnl'].sum()
        wr = (oracle_trades['pnl'] > 0).mean()
        cum = oracle_trades.sort_values('date')['pnl'].cumsum()
        max_dd = (cum - cum.cummax()).min()
        pnl_arr = oracle_trades['pnl'].values
        sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(252) if len(pnl_arr) > 1 and pnl_arr.std() > 0 else 0.0

        print(f"\n  ── ORACLE PORTFOLIO (oracle-positive tickers only) ──")
        print(f"    Tickers:    {len(oracle_pos)}")
        print(f"    Trades:     {n_trades}")
        print(f"    Avg PnL:    {avg_pnl*100:+.2f}%")
        print(f"    Win Rate:   {wr*100:.0f}%")
        print(f"    Total PnL:  {total_pnl*100:+.1f}%")
        print(f"    Max DD:     {max_dd*100:.1f}%")
        print(f"    Sharpe:     {sharpe:.2f}")

        # Monthly breakdown
        oracle_trades_dated = oracle_trades.copy()
        oracle_trades_dated['month'] = pd.to_datetime(oracle_trades_dated['date']).dt.strftime('%Y-%m')
        monthly = oracle_trades_dated.groupby('month').agg(
            n=('pnl', 'count'),
            total=('pnl', 'sum'),
            avg=('pnl', 'mean'),
            wr=('pnl', lambda x: (x > 0).mean())
        )
        print(f"\n    Monthly breakdown:")
        for m, r in monthly.iterrows():
            print(f"      {m}: {int(r['n']):3d} trades  {r['total']*100:+6.1f}% total  {r['avg']*100:+5.2f}% avg  {r['wr']*100:4.0f}% WR")

# ── Per-ticker table ──
print(f"\n  ── PER-TICKER ORACLE CEILING (sorted by FULL avg PnL) ──")
print(f"  {'Ticker':<8}│{'ORACLE':>8} {'WR':>5} {'N':>4} │{'H1':>8} {'H2':>8} │{'ENSEMB':>8} │{'GAP':>8} │{'Fold':>5} │ Config")
print(f"  {'─'*8}┼{'─'*18}─┼{'─'*18}┼{'─'*9}┼{'─'*9}┼{'─'*6}┼{'─'*28}")

for _, row in ceiling_df.iterrows():
    if row['oracle_mode'] == 'atr_sl':
        sl_str = f"sm={row['oracle_sl_mult']:.1f}"
    else:
        sl_str = f"sl={row['oracle_sl']:.3f}"
    cfg_str = f"{row['oracle_mode'][:6]} tp={row['oracle_tp']:.3f} {sl_str} p≥{row['oracle_prob']:.2f}"
    flag = " ★" if row['oracle_full_avg'] > 0.01 else ""
    print(f"  {row['ticker']:<8}│{row['oracle_full_avg']*100:+7.2f}% {row['oracle_full_wr']*100:4.0f}% {int(row['oracle_full_n']):3d} │"
          f"{row['oracle_h1_avg']*100:+7.2f}% {row['oracle_h2_avg']*100:+7.2f}% │"
          f"{row['ens_full_avg']*100:+7.2f}% │{row['gap_full']*100:+7.2f}% │"
          f"  {int(row['oracle_fold']):3d} │ {cfg_str}{flag}")

# ── Fold distribution of oracle configs ──
print(f"\n  ── WHICH FOLDS WIN (oracle config source) ──")
fold_counts = ceiling_df['oracle_fold'].value_counts().sort_index()
for f, cnt in fold_counts.items():
    alpha = '2.0' if f < 410 else '1.0'
    tks = ceiling_df[ceiling_df['oracle_fold'] == f]['ticker'].tolist()
    print(f"    Fold {int(f)} (α={alpha}): {cnt:2d} tickers — {', '.join(tks[:8])}")

# ── Mode distribution ──
print(f"\n  ── WINNING MODE DISTRIBUTION ──")
mode_counts = ceiling_df['oracle_mode'].value_counts()
for mode, cnt in mode_counts.items():
    print(f"    {mode:<12} {cnt:2d} ({100*cnt/len(ceiling_df):4.0f}%)")

# ── HALF2 stability ──
print(f"\n  ── HALF2 STABILITY (oracle-positive tickers) ──")
print(f"  {'Ticker':<8}│{'FULL':>8} │{'HALF1':>8} │{'HALF2':>8} │ Status")
print(f"  {'─'*8}┼{'─'*9}┼{'─'*9}┼{'─'*9}┼{'─'*12}")
for _, row in ceiling_df[ceiling_df['oracle_full_avg'] > 0].iterrows():
    h1 = row['oracle_h1_avg']
    h2 = row['oracle_h2_avg']
    if h2 > 0.01 and (h1 <= 0 or h2 >= h1 * 0.5):
        status = "HOLDS ★"
    elif h2 > 0:
        status = "WEAK"
    else:
        status = "DECAYED"
    print(f"  {row['ticker']:<8}│{row['oracle_full_avg']*100:+7.2f}% │{h1*100:+7.2f}% │{h2*100:+7.2f}% │ {status}")

# ── Config parameter ranges for winners ──
winners = ceiling_df[ceiling_df['oracle_full_avg'] > 0]
if len(winners) > 0:
    print(f"\n  ── CONFIG RANGES FOR ORACLE-POSITIVE TICKERS ({len(winners)} tickers) ──")
    print(f"    tp_pct:     {winners['oracle_tp'].min():.3f} – {winners['oracle_tp'].max():.3f}  (median {winners['oracle_tp'].median():.3f})")
    sl_static = winners[winners['oracle_mode'] != 'atr_sl']
    if len(sl_static) > 0:
        print(f"    sl_pct:     {sl_static['oracle_sl'].min():.3f} – {sl_static['oracle_sl'].max():.3f}  (median {sl_static['oracle_sl'].median():.3f})")
    print(f"    prob_th:    {winners['oracle_prob'].min():.2f} – {winners['oracle_prob'].max():.2f}  (median {winners['oracle_prob'].median():.2f})")


# ==============================================================================
# PHASE 6: CHARTS
# ==============================================================================
print(f"\n  Generating charts...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'SHORT Oracle — TEST Period Ceiling ({TEST_START} → {TEST_END})', fontsize=14, fontweight='bold')

# Panel 1: Per-ticker bar chart
ax = axes[0, 0]
tks = ceiling_df['ticker'].values
oracle_vals = ceiling_df['oracle_full_avg'].values * 100
ens_vals = ceiling_df['ens_full_avg'].values * 100
colors = ['#2E7D32' if v > 0 else '#C62828' for v in oracle_vals]
x = np.arange(len(tks))
ax.bar(x, oracle_vals, color=colors, alpha=0.7, label='Oracle (best config)')
ax.scatter(x, ens_vals, color='blue', s=20, zorder=5, label='Ensemble (val_a pick)')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(tks, rotation=90, fontsize=6)
ax.set_ylabel('Avg PnL (%)')
ax.set_title('Per-Ticker: Oracle vs Ensemble Selection')
ax.legend(fontsize=8)

# Panel 2: HALF1 vs HALF2 scatter
ax = axes[0, 1]
pos = ceiling_df[ceiling_df['oracle_full_avg'] > 0]
neg = ceiling_df[ceiling_df['oracle_full_avg'] <= 0]
if len(pos) > 0:
    ax.scatter(pos['oracle_h1_avg'] * 100, pos['oracle_h2_avg'] * 100,
               c='#2E7D32', s=40, alpha=0.7, label=f'Oracle+ ({len(pos)})')
    for _, row in pos.iterrows():
        ax.annotate(row['ticker'],
                    (row['oracle_h1_avg'] * 100, row['oracle_h2_avg'] * 100),
                    fontsize=7, alpha=0.8)
if len(neg) > 0:
    ax.scatter(neg['oracle_h1_avg'] * 100, neg['oracle_h2_avg'] * 100,
               c='#C62828', s=20, alpha=0.4, label=f'Oracle− ({len(neg)})')
lim = max(abs(ceiling_df['oracle_h1_avg'].max()), abs(ceiling_df['oracle_h2_avg'].max())) * 100 + 1
ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label='H1 = H2')
ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
ax.set_xlabel('HALF1 Avg PnL (%)')
ax.set_ylabel('HALF2 Avg PnL (%)')
ax.set_title('Oracle Stability: HALF1 vs HALF2')
ax.legend(fontsize=8)

# Panel 3: Fold distribution
ax = axes[1, 0]
fold_data = ceiling_df.groupby('oracle_fold')['oracle_full_avg'].agg(['count', 'mean'])
fold_data = fold_data.reindex(ALL_FOLDS).fillna(0)
colors_fold = ['#1565C0' if f < 410 else '#E65100' for f in ALL_FOLDS]
bars = ax.bar(range(len(ALL_FOLDS)), fold_data['count'].values, color=colors_fold, alpha=0.7)
ax.set_xticks(range(len(ALL_FOLDS)))
ax.set_xticklabels([str(f) for f in ALL_FOLDS], fontsize=9)
ax.set_ylabel('# Tickers (oracle source)')
ax.set_title('Which Fold Wins? (blue=α2.0, orange=α1.0)')
for i, (cnt, avg) in enumerate(zip(fold_data['count'].values, fold_data['mean'].values)):
    if cnt > 0:
        ax.text(i, cnt + 0.2, f'{avg*100:+.1f}%', ha='center', fontsize=7)

# Panel 4: Oracle portfolio equity curve
ax = axes[1, 1]
if not trades_df.empty:
    oracle_pos_tickers = set(ceiling_df[ceiling_df['oracle_full_avg'] > 0]['ticker'])
    curve_trades = trades_df[trades_df['ticker'].isin(oracle_pos_tickers)].sort_values('date')
    if len(curve_trades) > 0:
        cum_pnl = curve_trades['pnl'].cumsum() * 100
        ax.plot(range(len(cum_pnl)), cum_pnl.values, color='#2E7D32', linewidth=1.5)
        ax.fill_between(range(len(cum_pnl)), 0, cum_pnl.values,
                        where=cum_pnl.values >= 0, color='#2E7D32', alpha=0.1)
        ax.fill_between(range(len(cum_pnl)), 0, cum_pnl.values,
                        where=cum_pnl.values < 0, color='#C62828', alpha=0.1)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Cumulative PnL (%)')
        ax.set_title(f'Oracle Portfolio ({len(oracle_pos_tickers)} tickers, {len(curve_trades)} trades)')

        h1_mask = curve_trades['period'] == 'HALF1'
        h1_end_idx = h1_mask.sum()
        if 0 < h1_end_idx < len(curve_trades):
            ax.axvline(h1_end_idx, color='orange', linewidth=1, linestyle='--', label='HALF1|HALF2')
            ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches='tight')
print(f"  Chart saved: {OUTPUT_CHART}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  FINAL SUMMARY — SHORT ORACLE CEILING")
print(f"{'='*W}")
print(f"  Per-config backtest: {OUTPUT_PER_CONFIG} ({len(results_df)} rows)")
print(f"  Per-ticker ceiling:  {OUTPUT_PER_TICKER} ({len(ceiling_df)} tickers)")
print(f"  Oracle trades:       {OUTPUT_TRADES} ({len(trades_df)} trades)")
print(f"  Chart:               {OUTPUT_CHART}")
print(f"")
print(f"  ORACLE CEILING:  {ceiling_df['oracle_full_avg'].mean()*100:+.2f}% avg ({n_oracle_pos}/{n_total} positive)")
print(f"  ENSEMBLE (real): {ceiling_df['ens_full_avg'].mean()*100:+.2f}% avg ({n_ens_pos}/{n_total} positive)")
print(f"  CONFIG GAP:      {ceiling_df['gap_full'].mean()*100:+.2f}% avg")
print(f"")
if not trades_df.empty:
    oracle_pos_trades = trades_df[trades_df['ticker'].isin(oracle_pos['ticker'])]
    if len(oracle_pos_trades) > 0:
        print(f"  ORACLE PORTFOLIO (benchmark for strategy comparison):")
        print(f"    {len(oracle_pos)} tickers, {len(oracle_pos_trades)} trades")
        print(f"    Avg PnL:   {oracle_pos_trades['pnl'].mean()*100:+.2f}%")
        print(f"    Total PnL: {oracle_pos_trades['pnl'].sum()*100:+.1f}%")
        print(f"    Win Rate:  {(oracle_pos_trades['pnl'] > 0).mean()*100:.0f}%")
print(f"")
print(f"  → Next: Compare strategies against this oracle using {OUTPUT_TRADES}")
print(f"{'='*W}")
