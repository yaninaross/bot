"""
oracle_true_ceiling.py — Absolute Maximum Profitability per Asset
================================================================
Exhaustively backtests ALL ~2700 BQ configs for ALL tickers.
No Monte Carlo — exhaustive search is feasible at this scale.

Output:
  1. Per-ticker ceiling table: max PnL, best config, gap vs ensemble
  2. /tmp/oracle_per_config_results.csv — ~2700 rows (BQ features + backtest)
     → ML model training data for reverse-engineering selection
  3. /tmp/oracle_per_ticker_ceiling.csv — 53 rows
  4. /tmp/oracle_true_ceiling.png — 4-panel chart

GPU NOT required. Colab-ready (no if __name__).
"""

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
    try:
        auth.authenticate_user()
    except Exception:
        pass

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")
RESULTS_DIR    = "Algos/Results_all_v2"
INFERENCE_DIR  = "Data/Inference_v2"

FULL_START  = '2026-04-01'
FULL_END    = '2026-07-23'
HALF1_END   = '2026-05-31'
HALF2_START = '2026-06-01'

EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.002

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1')

FILTERED_VA_OPT_RATIO = 2.0

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110

OUTPUT_PER_CONFIG = '/tmp/oracle_per_config_results.csv'
OUTPUT_PER_TICKER = '/tmp/oracle_per_ticker_ceiling.csv'
OUTPUT_CHART      = '/tmp/oracle_true_ceiling.png'


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
        cutoff = pd.Timestamp(FULL_START, tz='UTC') - pd.Timedelta(days=30)
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
# PHASE 1: QUERY BQ
# ==============================================================================
print("=" * W)
print("  SHORT ORACLE — TRUE CEILING PER ASSET")
print("  Exhaustive backtest: ALL configs × ALL tickers")
print("=" * W)

print(f"\n  PHASE 1: Querying BQ...")
versions_sql = ', '.join(f"'{v}'" for v in BQ_MODEL_VERSIONS)
q = f"""
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       grace_minutes, vol_filter_pct, pre_range_filter_pct,
       opt_avg_pnl, opt_win_rate, opt_count, opt_freq,
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
bq_df = bq_client.query(q).to_dataframe()
n_tickers = bq_df['ticker'].nunique()
n_indiv = len(bq_df[bq_df['is_ensemble'] == False])
n_ens = len(bq_df[bq_df['is_ensemble'] == True])
print(f"    {len(bq_df)} BQ rows: {n_indiv} individual + {n_ens} ensemble across {n_tickers} tickers")


# ==============================================================================
# PHASE 2: LOAD ALL DATA
# ==============================================================================
print(f"\n  PHASE 2: Loading data for {n_tickers} tickers...")
all_tickers = sorted(bq_df['ticker'].unique())

# 2a. Load OHLCV for all tickers
ticker_data = {}
for ticker in all_tickers:
    df = load_master(ticker)
    if df is None:
        continue
    df = heal_data(df)
    ohlcv = precompute_ohlcv_arrays(df, FULL_START, FULL_END)
    if ohlcv is None:
        continue
    atr_pct = compute_daily_atr_pct(df)
    atr_vals = atr_pct.reindex(df.index).fillna(0.05)
    s = pd.to_datetime(FULL_START).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(FULL_END).tz_localize('America/Los_Angeles') + timedelta(days=1)
    idx = df.index
    if str(idx.tz) != 'America/Los_Angeles':
        idx = idx.tz_convert('America/Los_Angeles')
    atr_cut = atr_vals[(idx >= s) & (idx < e)]
    ohlcv['atr_pct'] = atr_cut.values.astype(np.float64)
    ticker_data[ticker] = {'ohlcv': ohlcv, 'df': df}
    del df
    gc.collect()

print(f"    Loaded OHLCV for {len(ticker_data)}/{len(all_tickers)} tickers")

# 2b. Load CSV scores for all (ticker, base_experiment) pairs
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
        print(f"      {ticker}/{base_exp}: {len(seed_scores)} seeds")

print(f"    Loaded CSV scores for {n_loaded} ticker×experiment pairs")
_release_memory()


# ==============================================================================
# PHASE 3: EXHAUSTIVE BACKTEST — EVERY CONFIG
# ==============================================================================
print(f"\n  PHASE 3: Exhaustive backtest ({len(bq_df)} configs)...")

full_start_dt = pd.Timestamp(FULL_START).date()
full_end_dt   = pd.Timestamp(FULL_END).date()
h1_end_dt     = pd.Timestamp(HALF1_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

results = []
tickers_done = 0

for ticker in all_tickers:
    if ticker not in ticker_data:
        continue

    ohlcv = ticker_data[ticker]['ohlcv']
    prices    = ohlcv['prices']
    highs     = ohlcv['highs']
    lows      = ohlcv['lows']
    date_ints = ohlcv['dates']
    time_mins = ohlcv['times']
    atr_arr   = ohlcv.get('atr_pct', np.full(len(prices), 0.05))

    ticker_rows = bq_df[bq_df['ticker'] == ticker]

    # Precompute daily ensemble probs per (base_exp, entry_minute)
    prob_cache = {}
    seen_keys = set()
    for _, r in ticker_rows[['experiment', 'entry_minute']].drop_duplicates().iterrows():
        base_exp = _base_experiment(r['experiment'])
        em_int = _safe_int(r['entry_minute'], 375)
        key = (base_exp, em_int)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        seed_scores = csv_scores_cache.get((ticker, base_exp))
        if not seed_scores:
            continue
        daily = {}
        for seed_id, sdf in seed_scores.items():
            bar_mins = sdf.index.hour * 60 + sdf.index.minute
            entry_bars = sdf[bar_mins == em_int]
            for dt, sr in entry_bars.iterrows():
                d = dt.date()
                if d not in daily:
                    daily[d] = []
                daily[d].append(float(sr['Prob_Spike']))
        prob_cache[key] = {d: np.mean(ps) for d, ps in daily.items()}

    # Precompute entry bar indices per entry_minute
    entry_idx_map = {}
    for em in ticker_rows['entry_minute'].dropna().unique():
        em_int = _safe_int(em, 375)
        entry_idx_map[em_int] = np.where(time_mins == em_int)[0]

    n_bt = 0
    for _, bq_row in ticker_rows.iterrows():
        exp = bq_row['experiment']
        base_exp = _base_experiment(exp)
        em_int = _safe_int(bq_row.get('entry_minute', 375), 375)
        mode = bq_row['tpsl_mode']

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
            d_int = int(date_ints[idx])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()

            if d_date < full_start_dt or d_date > full_end_dt:
                continue
            if d_int == last_td:
                continue

            prob = probs.get(d_date)
            if prob is None or prob < prob_thresh:
                continue

            if mode == 'atr_sl':
                pnl, td, filled = run_short_trade_atr_sl(
                    prices, highs, lows, date_ints, time_mins,
                    idx, tp_pct, float(atr_arr[idx]), sl_mult, EXIT_TIME_MIN)
            elif mode == 'grace':
                pnl, td, filled = run_short_trade_grace(
                    prices, highs, lows, date_ints, time_mins,
                    idx, tp_pct, sl_pct, EXIT_TIME_MIN, grace_min)
            else:
                pnl, td, filled = run_short_trade_static(
                    prices, highs, lows, date_ints, time_mins,
                    idx, tp_pct, sl_pct, EXIT_TIME_MIN)

            if not filled:
                continue

            pnl -= 2 * SLIPPAGE_PCT
            last_td = d_int
            trades.append({'date': d_date, 'pnl': pnl})

        # Split by period
        t_h1 = [t for t in trades if t['date'] <= h1_end_dt]
        t_h2 = [t for t in trades if t['date'] >= h2_start_dt]

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

        results.append({
            'ticker': ticker,
            'experiment': exp,
            'seed': int(bq_row['seed']) if not pd.isna(bq_row['seed']) else -1,
            'is_ensemble': bool(bq_row.get('is_ensemble', False)),
            'model_version': bq_row['model_version'],
            'tpsl_mode': mode,
            'entry_minute': em_int,
            'prob_thresh': prob_thresh,
            'tp_pct': tp_pct, 'sl_pct': sl_pct,
            'sl_mult': sl_mult, 'grace_minutes': grace_min,
            'is_limit_entry': int(mode == 'limit_entry'),
            'vol_filter_pct': _safe_float(bq_row.get('vol_filter_pct', 0)),
            'pre_range_filter_pct': _safe_float(bq_row.get('pre_range_filter_pct', 0)),
            # Unbiased BQ stats
            'opt_avg_pnl': opt_pnl,
            'opt_win_rate': _safe_float(bq_row.get('opt_win_rate')),
            'opt_count': _safe_float(bq_row.get('opt_count')),
            'opt_freq': _safe_float(bq_row.get('opt_freq', 0)),
            'val_a_avg_pnl': va_pnl,
            'val_a_win_rate': _safe_float(bq_row.get('val_a_win_rate')),
            'val_a_count': _safe_float(bq_row.get('val_a_count')),
            'val_a_freq': _safe_float(bq_row.get('val_a_freq', 0)),
            'va_opt_ratio': va_opt_ratio,
            # BQ biased (for validation only)
            'bq_vb_avg_pnl': _safe_float(bq_row.get('val_b_avg_pnl')),
            'bq_vb_win_rate': _safe_float(bq_row.get('val_b_win_rate')),
            'bq_vb_count': _safe_float(bq_row.get('val_b_count')),
            # CSV backtest ground truth
            'bt_full_n': nf, 'bt_full_total': tf_tot, 'bt_full_avg': af, 'bt_full_wr': wf,
            'bt_h1_n': n1, 'bt_h1_total': t1_tot, 'bt_h1_avg': a1, 'bt_h1_wr': w1,
            'bt_h2_n': n2, 'bt_h2_total': t2_tot, 'bt_h2_avg': a2, 'bt_h2_wr': w2,
        })
        n_bt += 1

    tickers_done += 1
    if n_bt > 0:
        print(f"      [{tickers_done:2d}/{len(ticker_data)}] {ticker}: {n_bt}/{len(ticker_rows)} configs backtested")

results_df = pd.DataFrame(results)
print(f"\n    Total: {len(results_df)} configs backtested across {results_df['ticker'].nunique()} tickers")

results_df.to_csv(OUTPUT_PER_CONFIG, index=False)
print(f"    Saved: {OUTPUT_PER_CONFIG}")
_release_memory()


# ==============================================================================
# PHASE 4: PER-TICKER CEILING ANALYSIS
# ==============================================================================
print(f"\n  PHASE 4: Per-ticker ceiling analysis...")

per_ticker = []
for ticker in sorted(results_df['ticker'].unique()):
    grp = results_df[results_df['ticker'] == ticker]
    n_cfg = len(grp)
    if n_cfg == 0:
        continue

    n_pos_full = (grp['bt_full_avg'] > 0).sum()

    # TRUE ORACLE: best config by FULL avg PnL
    best_full = grp.loc[grp['bt_full_avg'].idxmax()]

    # Best config by HALF2 PnL (out-of-sample oracle)
    h2_with_trades = grp[grp['bt_h2_n'] > 0]
    if not h2_with_trades.empty:
        best_h2 = grp.loc[h2_with_trades['bt_h2_avg'].idxmax()]
    else:
        best_h2 = best_full

    # ENSEMBLE selection (same logic as select_configs_ensemble)
    ens = grp[(grp['is_ensemble'] == True) &
              (grp['tpsl_mode'].isin(['static', 'atr_sl', 'grace'])) &
              (grp['va_opt_ratio'] <= FILTERED_VA_OPT_RATIO)]
    if not ens.empty:
        ens_best = ens.loc[ens['val_a_avg_pnl'].idxmax()]
        ens_full = float(ens_best['bt_full_avg'])
        ens_h2   = float(ens_best['bt_h2_avg'])
        ens_n    = int(ens_best['bt_full_n'])
    else:
        ens_full = 0.0
        ens_h2   = 0.0
        ens_n    = 0

    # Median config performance (random baseline)
    med_full = grp['bt_full_avg'].median()

    per_ticker.append({
        'ticker': ticker,
        'n_configs': n_cfg,
        'n_positive_full': n_pos_full,
        'pct_positive': n_pos_full / n_cfg,
        # True oracle (best by FULL)
        'oracle_full': float(best_full['bt_full_avg']),
        'oracle_full_wr': float(best_full['bt_full_wr']),
        'oracle_full_n': int(best_full['bt_full_n']),
        'oracle_h1': float(best_full['bt_h1_avg']),
        'oracle_h2': float(best_full['bt_h2_avg']),
        # Best by HALF2
        'best_h2': float(best_h2['bt_h2_avg']),
        'best_h2_full': float(best_h2['bt_full_avg']),
        # Ensemble method
        'ens_full': ens_full,
        'ens_h2': ens_h2,
        'ens_n': ens_n,
        # Gaps
        'gap_full': float(best_full['bt_full_avg']) - ens_full,
        'gap_h2': float(best_h2['bt_h2_avg']) - ens_h2,
        # Random baseline
        'median_full': med_full,
        # Best FULL config details
        'best_mode': best_full['tpsl_mode'],
        'best_tp': float(best_full['tp_pct']),
        'best_sl': float(best_full['sl_pct']),
        'best_prob': float(best_full['prob_thresh']),
        'best_entry': int(best_full['entry_minute']),
        'best_exp': best_full['experiment'],
    })

ceiling_df = pd.DataFrame(per_ticker).sort_values('oracle_full', ascending=False)
ceiling_df.to_csv(OUTPUT_PER_TICKER, index=False)
print(f"    Saved: {OUTPUT_PER_TICKER}")


# ==============================================================================
# PHASE 5: OUTPUT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 5: RESULTS")
print(f"{'='*W}")

n_total = len(ceiling_df)
n_oracle_pos = (ceiling_df['oracle_full'] > 0).sum()
n_ens_pos    = (ceiling_df['ens_full'] > 0).sum()
n_oracle_strong = (ceiling_df['oracle_full'] > 0.01).sum()

print(f"\n  ── SUMMARY ──")
print(f"    {n_total} tickers with backtest data")
print(f"    {n_oracle_pos} ({100*n_oracle_pos/n_total:.0f}%) have ≥1 profitable config (TRUE ORACLE)")
print(f"    {n_ens_pos} ({100*n_ens_pos/n_total:.0f}%) profitable with ensemble selection")
print(f"    {n_oracle_strong} ({100*n_oracle_strong/n_total:.0f}%) have oracle PnL > +1.0%")
print(f"    Gap: {n_oracle_pos - n_ens_pos} additional tickers become profitable with oracle config")
print(f"    Avg oracle ceiling: {ceiling_df['oracle_full'].mean()*100:+.2f}%")
print(f"    Avg ensemble PnL:   {ceiling_df['ens_full'].mean()*100:+.2f}%")
print(f"    Avg config gap:     {ceiling_df['gap_full'].mean()*100:+.2f}%")

# ── BQ val_b vs CSV backtest correlation ──
valid = results_df[(results_df['bq_vb_avg_pnl'] != 0) & (results_df['bt_full_n'] > 0)]
if len(valid) > 10:
    rho_bq, p_bq = sp_stats.spearmanr(valid['bq_vb_avg_pnl'], valid['bt_full_avg'])
    print(f"\n  ── BQ val_b vs CSV Backtest Correlation ──")
    print(f"    Spearman ρ = {rho_bq:+.3f} (p={p_bq:.4f}) on {len(valid)} configs")
    label = 'STRONG' if abs(rho_bq) > 0.5 else 'MODERATE' if abs(rho_bq) > 0.3 else 'WEAK'
    print(f"    {label} — {'BQ is reliable proxy' if abs(rho_bq) > 0.5 else 'CSV backtest is more reliable'}")

# ── Per-ticker table ──
print(f"\n  ── PER-TICKER TRUE ORACLE (sorted by oracle FULL PnL) ──")
print(f"  {'Ticker':<8}│{'ORACLE':>8} {'WR':>5} {'N':>4} │{'ENSEMB':>8} │{'GAP':>8} │{'ORC_H2':>8} {'BST_H2':>8} │{'Ncfg':>5} {'%Pos':>5}│ Best Config")
print(f"  {'─'*8}┼{'─'*18}─┼{'─'*9}┼{'─'*9}┼{'─'*18}─┼{'─'*11}┼{'─'*25}")

for _, row in ceiling_df.iterrows():
    sl_str = f"sl={row['best_sl']:.3f}" if row['best_mode'] != 'atr_sl' else f"sm={row['best_sl']:.1f}"
    cfg_str = f"{row['best_mode'][:6]} tp={row['best_tp']:.3f} {sl_str}"
    flag = " ★" if row['oracle_full'] > 0.01 else ""
    print(f"  {row['ticker']:<8}│{row['oracle_full']*100:+7.2f}% {row['oracle_full_wr']*100:4.0f}% {int(row['oracle_full_n']):3d} │"
          f"{row['ens_full']*100:+7.2f}% │{row['gap_full']*100:+7.2f}% │"
          f"{row['oracle_h2']*100:+7.2f}% {row['best_h2']*100:+7.2f}% │"
          f"{int(row['n_configs']):4d} {row['pct_positive']*100:4.0f}%│ {cfg_str}{flag}")

# ── Mode analysis ──
print(f"\n  ── WINNING MODE DISTRIBUTION (top 15 tickers by oracle) ──")
top15 = ceiling_df.head(15)
mode_counts = top15['best_mode'].value_counts()
for mode, cnt in mode_counts.items():
    tks = top15[top15['best_mode'] == mode]['ticker'].tolist()
    print(f"    {mode:<15} {cnt:2d} ({100*cnt/len(top15):4.0f}%) — {', '.join(tks)}")

# ── Config parameter ranges for winners ──
winners = ceiling_df[ceiling_df['oracle_full'] > 0]
if len(winners) > 0:
    print(f"\n  ── CONFIG RANGES FOR PROFITABLE TICKERS ({len(winners)} tickers) ──")
    print(f"    tp_pct:   {winners['best_tp'].min():.3f} – {winners['best_tp'].max():.3f}  (median {winners['best_tp'].median():.3f})")
    print(f"    sl_pct:   {winners['best_sl'].min():.3f} – {winners['best_sl'].max():.3f}  (median {winners['best_sl'].median():.3f})")
    print(f"    prob_th:  {winners['best_prob'].min():.2f} – {winners['best_prob'].max():.2f}  (median {winners['best_prob'].median():.2f})")

# ── Per-config PnL distribution ──
print(f"\n  ── PER-CONFIG PNL DISTRIBUTION (all {len(results_df)} configs) ──")
pcts = [0, 10, 25, 50, 75, 90, 100]
for p in pcts:
    v = np.percentile(results_df['bt_full_avg'], p)
    print(f"    P{p:3d}: {v*100:+7.2f}%")
n_pos_configs = (results_df['bt_full_avg'] > 0).sum()
print(f"    Positive configs: {n_pos_configs}/{len(results_df)} ({100*n_pos_configs/len(results_df):.0f}%)")

# ── HALF2 stability ──
print(f"\n  ── HALF2 STABILITY (top 15 by oracle FULL) ──")
print(f"  {'Ticker':<8}│{'FULL':>8} │{'HALF1':>8} │{'HALF2':>8} │ Status")
print(f"  {'─'*8}┼{'─'*9}┼{'─'*9}┼{'─'*9}┼{'─'*12}")
for _, row in top15.iterrows():
    h1 = row['oracle_h1']
    h2 = row['oracle_h2']
    if h2 > 0.01 and h2 >= h1 * 0.5:
        status = "HOLDS"
    elif h2 > 0:
        status = "WEAK"
    else:
        status = "DECAYED"
    print(f"  {row['ticker']:<8}│{row['oracle_full']*100:+7.2f}% │{h1*100:+7.2f}% │{h2*100:+7.2f}% │ {status}")


# ==============================================================================
# PHASE 6: CHARTS
# ==============================================================================
print(f"\n  Generating charts...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SHORT Oracle — True Ceiling per Asset', fontsize=14, fontweight='bold')

# Panel 1: Per-ticker bar chart (sorted by oracle FULL)
ax = axes[0, 0]
tks = ceiling_df['ticker'].values
oracle_vals = ceiling_df['oracle_full'].values * 100
ens_vals = ceiling_df['ens_full'].values * 100
colors = ['#2E7D32' if v > 0 else '#C62828' for v in oracle_vals]
x = np.arange(len(tks))
ax.bar(x, oracle_vals, color=colors, alpha=0.7, label='Oracle (best config)')
ax.scatter(x, ens_vals, color='blue', s=20, zorder=5, label='Ensemble selection')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(tks, rotation=90, fontsize=6)
ax.set_ylabel('Avg PnL (%)')
ax.set_title('Per-Ticker: Oracle vs Ensemble')
ax.legend(fontsize=8)

# Panel 2: Oracle vs Ensemble scatter
ax = axes[0, 1]
ax.scatter(ceiling_df['ens_full'] * 100, ceiling_df['oracle_full'] * 100,
           c=colors, s=40, alpha=0.7)
lim = max(abs(ceiling_df['oracle_full'].max()), abs(ceiling_df['ens_full'].max())) * 100 + 1
ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label='No gap')
for _, row in ceiling_df.head(10).iterrows():
    ax.annotate(row['ticker'],
                (row['ens_full'] * 100, row['oracle_full'] * 100),
                fontsize=7, alpha=0.8)
ax.set_xlabel('Ensemble PnL (%)')
ax.set_ylabel('Oracle PnL (%)')
ax.set_title('Config Selection Gap')
ax.legend(fontsize=8)

# Panel 3: BQ val_b vs CSV backtest
ax = axes[1, 0]
if len(valid) > 10:
    ax.scatter(valid['bq_vb_avg_pnl'] * 100, valid['bt_full_avg'] * 100,
               alpha=0.15, s=8, c='steelblue')
    ax.plot([-10, 10], [-10, 10], 'k--', alpha=0.3)
    ax.set_xlabel('BQ val_b PnL (%)')
    ax.set_ylabel('CSV Backtest PnL (%)')
    ax.set_title(f'BQ vs CSV Correlation (ρ={rho_bq:+.3f}, N={len(valid)})')
    ax.set_xlim(np.percentile(valid['bq_vb_avg_pnl']*100, [1, 99]))
    ax.set_ylim(np.percentile(valid['bt_full_avg']*100, [1, 99]))
else:
    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)

# Panel 4: PnL distribution histogram
ax = axes[1, 1]
bt_pnls = results_df['bt_full_avg'].values * 100
ax.hist(bt_pnls, bins=60, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(0, color='red', linewidth=1, linestyle='--')
ax.axvline(np.median(bt_pnls), color='orange', linewidth=1, label=f'Median: {np.median(bt_pnls):+.2f}%')
ax.set_xlabel('Avg PnL (%)')
ax.set_ylabel('Config Count')
ax.set_title(f'PnL Distribution ({len(results_df)} configs)')
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
print(f"  FINAL SUMMARY")
print(f"{'='*W}")
print(f"  Per-config results: {OUTPUT_PER_CONFIG} ({len(results_df)} rows)")
print(f"  Per-ticker ceiling: {OUTPUT_PER_TICKER} ({len(ceiling_df)} tickers)")
print(f"  Chart: {OUTPUT_CHART}")
print(f"")
print(f"  TRUE ORACLE ceiling:   {ceiling_df['oracle_full'].mean()*100:+.2f}% avg FULL, {ceiling_df.head(10)['oracle_full'].mean()*100:+.2f}% top-10")
print(f"  Ensemble selection:    {ceiling_df['ens_full'].mean()*100:+.2f}% avg FULL")
print(f"  Config selection gap:  {ceiling_df['gap_full'].mean()*100:+.2f}% avg")
print(f"")
print(f"  Top 5 by TRUE ORACLE:")
for _, row in ceiling_df.head(5).iterrows():
    print(f"    {row['ticker']:<8} {row['oracle_full']*100:+.2f}% FULL  {row['oracle_h2']*100:+.2f}% H2  (ens: {row['ens_full']*100:+.2f}%, gap: {row['gap_full']*100:+.2f}%)")
print(f"")
print(f"  → Use {OUTPUT_PER_CONFIG} as ML training data:")
print(f"    Features: opt/val_a stats, hyperparams, mode flags  (~15 unbiased)")
print(f"    Target:   bt_full_avg or bt_h2_avg (CSV backtest ground truth)")
print(f"    Samples:  {len(results_df)} rows — enough for neural network")
print(f"{'='*W}")
