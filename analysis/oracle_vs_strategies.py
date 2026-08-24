# %% ── Oracle vs Strategies — Exhaustive SHORT Model Evaluation ─────────────
#
# Pre-computes per-fold trades once, then sweeps ~960 unbiased strategies
# across 3 axes: ticker selection × config selection × signal method.
# Compares each to the TRUE ORACLE ceiling.
#
# Outputs:
#   /tmp/oracle_vs_strategies_results.csv  — all strategy results
#   /tmp/oracle_vs_strategies_top.csv      — top 50 strategies
#
# GPU NOT required. Colab-ready.

import os, io, gc, sys, warnings, ctypes
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np, pandas as pd, pytz

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

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

FOLDS_A2 = [400, 401, 402, 403, 404, 405]
FOLDS_A1 = [410, 411, 412, 413, 414, 415]
ALL_FOLDS = FOLDS_A2 + FOLDS_A1

FOLD_EXPERIMENTS    = {f: f'{f}_XSECTOR' for f in ALL_FOLDS}
FOLD_MODEL_VERSIONS = {f: f'robust_wf_v12_short_{f}' for f in ALL_FOLDS}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_ALLOWED_MODES = ('static', 'atr_sl', 'grace')

SEED_AGREE_THRESHOLD  = 0.018

CONFIG_METHODS = ['best_opt', 'median_opt', 'most_configs', 'conservative',
                  'worst_opt', 'h1_oracle', 'fixed_tpsl']

FIXED_TP   = 0.08
FIXED_SL   = 0.04
FIXED_MODE = 'static'
SIGNAL_METHODS = ['single_best', 'ensemble_all', 'ensemble_top3',
                  'vote_2', 'vote_3', 'vote_both_alpha']
THRESH_MULTS = [1.0, 1.25, 1.5, 2.0]

FOLD_GROUPS = {
    'ALL_12':    ALL_FOLDS,
    'A2_ALL':    FOLDS_A2,
    'A1_ALL':    FOLDS_A1,
    'EARLY_A2':  [400, 401, 402],
    'LATE_A2':   [403, 404, 405],
    'EARLY_A1':  [410, 411, 412],
    'LATE_A1':   [413, 414, 415],
    'EARLY_ALL': [400, 401, 402, 410, 411, 412],
    'CROSS_BEST': None,
}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120

OUTPUT_ALL     = '/tmp/oracle_vs_strategies_results.csv'
OUTPUT_TOP     = '/tmp/oracle_vs_strategies_top.csv'
ORACLE_CSV     = '/tmp/oracle_short_per_ticker.csv'
ORACLE_TRADES  = '/tmp/oracle_short_trades.csv'


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
# STRATEGY EVALUATION FUNCTIONS
# ==============================================================================
def make_gate_label(params):
    parts = []
    if params['min_folds'] > 1:
        parts.append(f"F>={params['min_folds']}")
    if params['alpha_req'] != 'any':
        parts.append(f"A={params['alpha_req']}")
    if params['va_cons_min'] > 0:
        parts.append(f"VC>={params['va_cons_min']:.0%}")
    if params['seed_agree_min'] > 0:
        parts.append(f"SA>={params['seed_agree_min']:.0%}")
    if params.get('min_opt_pos', 0) > 0:
        parts.append(f"OP>={params['min_opt_pos']}")
    return ','.join(parts) if parts else 'ANY'


def gate_passes(feat, available, params):
    if len(available) < params['min_folds']:
        return False
    ar = params['alpha_req']
    if ar == 'both':
        if not any(f < 410 for f in available) or not any(f >= 410 for f in available):
            return False
    elif ar == 'a2_only':
        if not any(f < 410 for f in available):
            return False
    elif ar == 'a1_only':
        if not any(f >= 410 for f in available):
            return False
    if params['va_cons_min'] > 0 and feat.get('va_consistency', 0) < params['va_cons_min']:
        return False
    if params['seed_agree_min'] > 0 and feat.get('seed_agreement', 0) < params['seed_agree_min']:
        return False
    min_op = params.get('min_opt_pos', 0)
    if min_op > 0:
        n_opt_pos = int(round(feat.get('va_consistency', 0) * feat.get('n_folds', 0)))
        if n_opt_pos < min_op:
            return False
    return True


def select_config_fold(ticker, feat, fold_configs, available, method,
                       h1_pnl=None, fold_configs_worst=None):
    if not available:
        return None
    if method == 'best_opt':
        best, best_o = None, -999
        for f in available:
            cfg = fold_configs.get((ticker, f))
            if cfg and cfg['opt_avg_pnl'] > best_o:
                best_o = cfg['opt_avg_pnl']
                best = f
        return best
    elif method == 'worst_opt':
        src = fold_configs_worst if fold_configs_worst else fold_configs
        best, best_o = None, 999
        for f in available:
            cfg = src.get((ticker, f))
            if cfg and cfg['opt_avg_pnl'] < best_o:
                best_o = cfg['opt_avg_pnl']
                best = f
        return best
    elif method == 'h1_oracle':
        if h1_pnl is None:
            return None
        best, best_h1 = None, -999
        for f in available:
            h1 = h1_pnl.get((ticker, f), 0.0)
            if h1 > best_h1:
                best_h1 = h1
                best = f
        return best
    elif method == 'fixed_tpsl':
        best, best_cnt = None, -1
        for f in available:
            cnt = feat.get('bq_counts', {}).get(f, 0)
            if cnt > best_cnt:
                best_cnt = cnt
                best = f
        return best
    elif method == 'median_opt':
        fv = [(fold_configs[(ticker, f)]['opt_avg_pnl'], f)
              for f in available if (ticker, f) in fold_configs]
        if not fv:
            return None
        fv.sort()
        return fv[len(fv) // 2][1]
    elif method == 'most_configs':
        best, best_cnt = None, -1
        for f in available:
            cnt = feat.get('bq_counts', {}).get(f, 0)
            if cnt > best_cnt:
                best_cnt = cnt
                best = f
        return best
    elif method == 'conservative':
        best, best_tp = None, 999.0
        for f in available:
            cfg = fold_configs.get((ticker, f))
            if cfg and cfg['tp_pct'] < best_tp:
                best_tp = cfg['tp_pct']
                best = f
        return best
    return None


def compute_signal_dates(ticker, selected_fold, available, daily_probs,
                         fold_configs, fold_fire_dates, method, thresh_mult=1.0):
    if method == 'single_best':
        return fold_fire_dates.get((ticker, selected_fold), set())

    elif method == 'ensemble_all':
        fps, thresholds = [], []
        for f in available:
            fp = daily_probs.get(ticker, {}).get(f)
            cfg = fold_configs.get((ticker, f))
            if fp and cfg:
                fps.append(fp)
                thresholds.append(cfg['prob_thresh'])
        if not fps:
            return set()
        med_thresh = float(np.median(thresholds)) * thresh_mult
        all_dates = set()
        for fp in fps:
            all_dates |= set(fp.keys())
        result = set()
        for d in all_dates:
            vals = [fp[d] for fp in fps if d in fp]
            if vals and np.mean(vals) >= med_thresh:
                result.add(d)
        return result

    elif method == 'ensemble_top3':
        fv = []
        for f in available:
            fp = daily_probs.get(ticker, {}).get(f)
            cfg = fold_configs.get((ticker, f))
            if fp and cfg:
                fv.append((cfg['opt_avg_pnl'], f, fp, cfg['prob_thresh']))
        fv.sort(reverse=True)
        top = fv[:3]
        if not top:
            return set()
        med_thresh = float(np.median([t[3] for t in top])) * thresh_mult
        all_dates = set()
        for _, _, fp, _ in top:
            all_dates |= set(fp.keys())
        result = set()
        for d in all_dates:
            vals = [fp[d] for _, _, fp, _ in top if d in fp]
            if vals and np.mean(vals) >= med_thresh:
                result.add(d)
        return result

    elif method in ('vote_2', 'vote_3'):
        min_votes = 2 if method == 'vote_2' else 3
        all_dates = set()
        for f in available:
            all_dates |= fold_fire_dates.get((ticker, f), set())
        result = set()
        for d in all_dates:
            votes = sum(1 for f in available if d in fold_fire_dates.get((ticker, f), set()))
            if votes >= min_votes:
                result.add(d)
        return result

    elif method == 'vote_both_alpha':
        a2 = [f for f in available if f < 410]
        a1 = [f for f in available if f >= 410]
        if not a2 or not a1:
            return set()
        all_dates = set()
        for f in available:
            all_dates |= fold_fire_dates.get((ticker, f), set())
        result = set()
        for d in all_dates:
            a2_fire = any(d in fold_fire_dates.get((ticker, f), set()) for f in a2)
            a1_fire = any(d in fold_fire_dates.get((ticker, f), set()) for f in a1)
            if a2_fire and a1_fire:
                result.add(d)
        return result

    return set()


def evaluate_trades(trades_list, oracle_pos_set, oracle_total_pnl, h2_start_dt):
    if not trades_list:
        return None
    sorted_trades = sorted(trades_list, key=lambda t: t['date'])
    pnls = np.array([t['pnl'] for t in sorted_trades])
    n = len(pnls)
    total = float(pnls.sum())
    avg = float(pnls.mean())
    wr = float((pnls > 0).mean())
    tickers_set = set(t['ticker'] for t in sorted_trades)
    if n > 1 and pnls.std() > 0:
        sharpe = float(avg / pnls.std() * np.sqrt(252))
    else:
        sharpe = 0.0
    cum = pnls.cumsum()
    max_dd = float((cum - np.maximum.accumulate(cum)).min())
    h2_pnls = pnls[np.array([t['date'] >= h2_start_dt for t in sorted_trades])]
    h2_n = len(h2_pnls)
    h2_total = float(h2_pnls.sum()) if h2_n > 0 else 0.0
    h2_avg = float(h2_pnls.mean()) if h2_n > 0 else 0.0
    oracle_overlap = len(tickers_set & oracle_pos_set) / max(len(oracle_pos_set), 1)
    oracle_capture = total / oracle_total_pnl if oracle_total_pnl > 0 else 0.0
    monthly = {}
    for t in sorted_trades:
        m = str(t['date'])[:7]
        monthly[m] = monthly.get(m, 0.0) + t['pnl']
    ticker_pnls = defaultdict(float)
    for t in sorted_trades:
        ticker_pnls[t['ticker']] += t['pnl']
    max_ticker_pnl = max(ticker_pnls.values()) if ticker_pnls else 0.0
    max_concentration = max_ticker_pnl / total if total > 0 else 1.0
    return {
        'n_tickers': len(tickers_set), 'n_trades': n,
        'avg_pnl': avg, 'wr': wr, 'total_pnl': total,
        'sharpe': sharpe, 'max_dd': max_dd,
        'h2_n': h2_n, 'h2_total': h2_total, 'h2_avg': h2_avg,
        'oracle_overlap': oracle_overlap, 'oracle_capture': oracle_capture,
        'max_concentration': max_concentration,
        'monthly': monthly, 'tickers': tickers_set,
    }


def collect_trades(gate_params, config_method, signal_method, fold_group_name,
                   ticker_features, fold_configs, daily_probs, fold_fire_dates,
                   date_trade_pnl, cross_best_folds, thresh_mult=1.0,
                   h1_pnl=None, fold_configs_worst=None,
                   date_trade_pnl_fixed=None):
    trades = []
    use_fixed = (config_method == 'fixed_tpsl')
    for ticker in sorted(ticker_features.keys()):
        feat = ticker_features[ticker]
        if fold_group_name == 'CROSS_BEST':
            available = cross_best_folds.get(ticker, [])
        elif fold_group_name == 'ALL_12':
            available = feat['available_folds']
        else:
            pool_set = set(FOLD_GROUPS.get(fold_group_name, ALL_FOLDS))
            available = [f for f in feat['available_folds'] if f in pool_set]
        if not available:
            continue
        if gate_params is not None and not gate_passes(feat, available, gate_params):
            continue
        sel_fold = select_config_fold(ticker, feat, fold_configs, available, config_method,
                                      h1_pnl=h1_pnl, fold_configs_worst=fold_configs_worst)
        if sel_fold is None:
            continue
        sig_dates = compute_signal_dates(ticker, sel_fold, available,
                                         daily_probs, fold_configs, fold_fire_dates,
                                         signal_method, thresh_mult=thresh_mult)
        if use_fixed:
            pnl_map = date_trade_pnl_fixed.get(ticker, {}) if date_trade_pnl_fixed else {}
        else:
            pnl_map = date_trade_pnl.get((ticker, sel_fold), {})
        for d in sig_dates:
            if d in pnl_map:
                trades.append({'date': d, 'ticker': ticker, 'pnl': pnl_map[d], 'fold': sel_fold})
    return trades


# ==============================================================================
# PHASE 1: LOAD DATA
# ==============================================================================
np.random.seed(42)

print(f"{'='*W}")
print(f"  ORACLE vs STRATEGIES — SHORT Model Evaluation (val_a ignored, +worst_opt/h1_oracle/fixed)")
print(f"  TEST: {TEST_START} → {TEST_END}")
print(f"  Axes: {len(CONFIG_METHODS)} configs × {len(SIGNAL_METHODS)} signals × gates × fold-groups")
print(f"{'='*W}")

print(f"\n  PHASE 1: Loading data...")
mv_str = ", ".join(f"'{FOLD_MODEL_VERSIONS[f]}'" for f in ALL_FOLDS)
modes_str = ", ".join(f"'{m}'" for m in BQ_ALLOWED_MODES)

q = f"""
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       grace_minutes, opt_avg_pnl, val_a_avg_pnl, val_a_count,
       val_b_avg_pnl, is_ensemble, survived_a, model_version
FROM `{BQ_TABLE}`
WHERE model_version IN ({mv_str})
  AND opt_avg_pnl > 0
  AND tpsl_mode IN ({modes_str})
ORDER BY ticker, model_version
"""
bq_df = bq_client.query(q).to_dataframe()
bq_df['fold'] = bq_df['model_version'].str.extract(r'_(\d+)$').astype(int)
print(f"    BQ: {len(bq_df)} rows ({bq_df['ticker'].nunique()} tickers, {bq_df['fold'].nunique()} folds)")

fold_configs = {}
fold_configs_worst = {}
n_from_ens, n_from_indiv = 0, 0
for _, row in bq_df.sort_values('opt_avg_pnl', ascending=False).iterrows():
    key = (row['ticker'], row['fold'])
    cfg_entry = {
        'tpsl_mode': row['tpsl_mode'],
        'prob_thresh': _safe_float(row['prob_thresh']),
        'tp_pct': _safe_float(row['tp_pct']),
        'sl_pct': _safe_float(row['sl_pct']),
        'sl_mult': _safe_float(row['sl_mult']),
        'grace_minutes': _safe_int(row['grace_minutes']),
        'val_a_avg_pnl': _safe_float(row['val_a_avg_pnl']),
        'opt_avg_pnl': _safe_float(row['opt_avg_pnl']),
        'experiment': row['experiment'],
        'is_ensemble': bool(row.get('is_ensemble', False)),
    }
    if key not in fold_configs:
        fold_configs[key] = cfg_entry
        if cfg_entry['is_ensemble']:
            n_from_ens += 1
        else:
            n_from_indiv += 1
    if key not in fold_configs_worst or cfg_entry['opt_avg_pnl'] < fold_configs_worst[key]['opt_avg_pnl']:
        fold_configs_worst[key] = cfg_entry
n_tickers_fc = len(set(t for t, f in fold_configs.keys()))
print(f"    fold_configs: {len(fold_configs)} entries ({n_from_ens} ensemble-best + {n_from_indiv} individual-best), {n_tickers_fc} tickers")
print(f"    fold_configs_worst: {len(fold_configs_worst)} entries (lowest opt_avg_pnl per ticker-fold)")

indiv_df = bq_df[bq_df['is_ensemble'] == False]
seed_agree_by_tf = {}
for (ticker, fold), grp in indiv_df.groupby(['ticker', 'fold']):
    n_total = len(grp)
    n_agree = (grp['opt_avg_pnl'] >= SEED_AGREE_THRESHOLD).sum()
    seed_agree_by_tf[(ticker, fold)] = n_agree / n_total if n_total > 0 else 0.0

bq_counts = bq_df.groupby(['ticker', 'fold']).size().to_dict()

all_tickers = sorted(set(t for t, f in fold_configs.keys()))
print(f"    {len(all_tickers)} tickers with surviving configs")

print(f"\n    Loading OHLCV...")
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
    _master_cache.pop(ticker, None)
    if (i + 1) % 10 == 0:
        print(f"      {i+1}/{len(all_tickers)} loaded")
        _release_memory()
_release_memory()
print(f"    OHLCV: {len(ticker_data)} tickers")

print(f"\n    Loading seed scores for all {len(ALL_FOLDS)} folds...")
daily_probs = defaultdict(dict)
total_loaded = 0
for fold in ALL_FOLDS:
    exp = FOLD_EXPERIMENTS[fold]
    fold_tickers = [t for t in all_tickers if (t, fold) in fold_configs and t in ticker_data]
    n_loaded = 0
    for ticker in fold_tickers:
        seeds = load_seed_scores(ticker, exp)
        if not seeds:
            continue
        dp = compute_daily_prob(seeds, ENTRY_MIN)
        if dp:
            daily_probs[ticker][fold] = dp
            n_loaded += 1
        del seeds
    total_loaded += n_loaded
    print(f"      Fold {fold}: {n_loaded}/{len(fold_tickers)} tickers")
    _release_memory()
print(f"    Total: {total_loaded} ticker-fold probability sets")


# ==============================================================================
# PHASE 2: PRE-COMPUTE PER-FOLD TRADES + FEATURES
# ==============================================================================
print(f"\n  PHASE 2: Pre-computing per-fold trades + ticker features...")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

date_trade_pnl = {}
fold_fire_dates_by_mult = {m: {} for m in THRESH_MULTS}
n_precomputed = 0

for ticker in sorted(ticker_data.keys()):
    ohlcv = ticker_data[ticker]['ohlcv']
    entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]

    for fold in ALL_FOLDS:
        cfg = fold_configs.get((ticker, fold))
        probs = daily_probs.get(ticker, {}).get(fold)
        if cfg is None or not probs:
            continue

        pnl_by_date = {}
        last_td = -1
        for idx in entry_indices:
            d_int = int(ohlcv['dates'][idx])
            if d_int == last_td:
                continue
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            pnl, td, filled = execute_trade(
                ohlcv, idx, cfg['tpsl_mode'],
                cfg['tp_pct'], cfg['sl_pct'], cfg['sl_mult'], cfg['grace_minutes'])
            if filled:
                pnl_by_date[d_date] = pnl
                last_td = d_int

        date_trade_pnl[(ticker, fold)] = pnl_by_date

        for mult in THRESH_MULTS:
            fire = {d for d, p in probs.items()
                    if d >= test_start_dt and d <= test_end_dt and p >= cfg['prob_thresh'] * mult}
            fold_fire_dates_by_mult[mult][(ticker, fold)] = fire
        n_precomputed += 1

print(f"    Pre-computed {n_precomputed} (ticker, fold) trade sets × {len(THRESH_MULTS)} thresholds")

ticker_features = {}
for ticker in sorted(ticker_data.keys()):
    avail = sorted([f for f in ALL_FOLDS
                    if (ticker, f) in fold_configs and f in daily_probs.get(ticker, {})])
    if not avail:
        continue
    n_a2 = sum(1 for f in avail if f < 410)
    n_a1 = sum(1 for f in avail if f >= 410)
    opt_vals = [fold_configs[(ticker, f)]['opt_avg_pnl'] for f in avail]
    opt_pos = sum(1 for v in opt_vals if v > 0)
    sa_vals = [seed_agree_by_tf[(ticker, f)] for f in avail if (ticker, f) in seed_agree_by_tf]

    ticker_features[ticker] = {
        'available_folds': avail,
        'n_folds': len(avail),
        'n_a2': n_a2, 'n_a1': n_a1,
        'has_both_alphas': n_a2 > 0 and n_a1 > 0,
        'va_consistency': opt_pos / len(avail) if avail else 0,
        'va_median': float(np.median(opt_vals)),
        'va_std': float(np.std(opt_vals)) if len(opt_vals) > 1 else 0,
        'seed_agreement': float(np.mean(sa_vals)) if sa_vals else 0,
        'bq_counts': {f: bq_counts.get((ticker, f), 0) for f in avail},
    }

cross_best_folds = {}
for ticker, feat in ticker_features.items():
    best_a2, best_a1 = None, None
    best_a2_opt, best_a1_opt = -999, -999
    for f in feat['available_folds']:
        cfg = fold_configs.get((ticker, f))
        if cfg is None:
            continue
        ov = cfg['opt_avg_pnl']
        if f < 410 and ov > best_a2_opt:
            best_a2_opt, best_a2 = ov, f
        elif f >= 410 and ov > best_a1_opt:
            best_a1_opt, best_a1 = ov, f
    folds = []
    if best_a2 is not None: folds.append(best_a2)
    if best_a1 is not None: folds.append(best_a1)
    cross_best_folds[ticker] = folds

print(f"    {len(ticker_features)} tickers with features")
for nf_min in [1, 3, 6, 9, 12]:
    ct = sum(1 for f in ticker_features.values() if f['n_folds'] >= nf_min)
    print(f"      >= {nf_min:2d} folds: {ct} tickers")

# H1 PnL per (ticker, fold) — for h1_oracle config selection
h1_pnl = {}
for (ticker, fold), pnl_map in date_trade_pnl.items():
    h1_total = sum(pnl for d, pnl in pnl_map.items() if d < h2_start_dt)
    h1_pnl[(ticker, fold)] = h1_total

# Fixed TP/SL pre-computation — config-independent trade PnL
print(f"\n    Pre-computing fixed TP/SL trades (TP={FIXED_TP*100:.0f}%, SL={FIXED_SL*100:.0f}%, {FIXED_MODE})...")
date_trade_pnl_fixed = {}
n_fixed = 0
for ticker in sorted(ticker_data.keys()):
    ohlcv = ticker_data[ticker]['ohlcv']
    entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]
    pnl_by_date = {}
    last_td = -1
    for idx in entry_indices:
        d_int = int(ohlcv['dates'][idx])
        if d_int == last_td:
            continue
        d_str = str(d_int)
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < test_start_dt or d_date > test_end_dt:
            continue
        pnl, td, filled = execute_trade(
            ohlcv, idx, FIXED_MODE, FIXED_TP, FIXED_SL, 0.0, 0)
        if filled:
            pnl_by_date[d_date] = pnl
            last_td = d_int
    if pnl_by_date:
        date_trade_pnl_fixed[ticker] = pnl_by_date
        n_fixed += 1
print(f"    Fixed trades: {n_fixed} tickers")

# OPT → TEST correlation diagnostic
print(f"\n    ── OPT → TEST CORRELATION DIAGNOSTIC ──")
opt_vals_all, test_vals_all = [], []
for (ticker, fold), pnl_map in date_trade_pnl.items():
    cfg = fold_configs.get((ticker, fold))
    if cfg is None or not pnl_map:
        continue
    test_avg = np.mean(list(pnl_map.values()))
    opt_vals_all.append(cfg['opt_avg_pnl'])
    test_vals_all.append(test_avg)
if len(opt_vals_all) > 10:
    from scipy import stats as scipy_stats
    opt_a = np.array(opt_vals_all)
    test_a = np.array(test_vals_all)
    pearson_r = np.corrcoef(opt_a, test_a)[0, 1]
    spearman_r = scipy_stats.spearmanr(opt_a, test_a).statistic
    print(f"    OPT→TEST Pearson r  = {pearson_r:+.3f}  (n={len(opt_a)} ticker-fold pairs)")
    print(f"    OPT→TEST Spearman ρ = {spearman_r:+.3f}")
    if pearson_r < -0.05:
        print(f"    ⚠ ANTI-CORRELATION: higher OPT → worse TEST. 'worst_opt' may outperform 'best_opt'.")
    elif pearson_r < 0.1:
        print(f"    ⚠ NEAR ZERO: OPT has no predictive power for TEST.")
    else:
        print(f"    ✓ Positive correlation: OPT-based selection should work.")
    # H1-only correlation
    h1_vals_all = []
    for (ticker, fold), pnl_map in date_trade_pnl.items():
        cfg = fold_configs.get((ticker, fold))
        if cfg is None or not pnl_map:
            continue
        h1_trades = [p for d, p in pnl_map.items() if d < h2_start_dt]
        h2_trades = [p for d, p in pnl_map.items() if d >= h2_start_dt]
        if h1_trades and h2_trades:
            h1_vals_all.append(np.mean(h1_trades))
    if len(h1_vals_all) > 10:
        h2_vals_corr = []
        h1_vals_corr = []
        for (ticker, fold), pnl_map in date_trade_pnl.items():
            cfg = fold_configs.get((ticker, fold))
            if cfg is None or not pnl_map:
                continue
            h1_t = [p for d, p in pnl_map.items() if d < h2_start_dt]
            h2_t = [p for d, p in pnl_map.items() if d >= h2_start_dt]
            if h1_t and h2_t:
                h1_vals_corr.append(np.mean(h1_t))
                h2_vals_corr.append(np.mean(h2_t))
        h1_a = np.array(h1_vals_corr)
        h2_a = np.array(h2_vals_corr)
        h1h2_pearson = np.corrcoef(h1_a, h2_a)[0, 1]
        h1h2_spearman = scipy_stats.spearmanr(h1_a, h2_a).statistic
        print(f"    H1→H2  Pearson r  = {h1h2_pearson:+.3f}  (n={len(h1_a)} ticker-fold pairs)")
        print(f"    H1→H2  Spearman ρ = {h1h2_spearman:+.3f}")
        if h1h2_pearson > 0.1:
            print(f"    ✓ H1 predicts H2: 'h1_oracle' should outperform OPT-based methods.")
        elif h1h2_pearson < -0.05:
            print(f"    ⚠ H1→H2 ANTI-CORRELATED: no config selection will work across halves.")
        else:
            print(f"    ⚠ H1→H2 near zero: configs that work in H1 don't predict H2.")

_release_memory()


# ==============================================================================
# PHASE 3: LOAD ORACLE BASELINE
# ==============================================================================
print(f"\n  PHASE 3: Loading oracle baseline...")
import os as _os
_missing = [f for f in (ORACLE_CSV, ORACLE_TRADES) if not _os.path.exists(f)]
if _missing:
    print(f"\n{'='*70}")
    print(f"  *** WARNING: Oracle CSV files NOT FOUND ***")
    for f in _missing:
        print(f"    MISSING: {f}")
    print(f"\n  You must run oracle_short_test.py FIRST to produce these files.")
    print(f"  Run:  %run oracle_short_test.py   (or python3 oracle_short_test.py)")
    print(f"{'='*70}\n")
    raise FileNotFoundError(f"Missing oracle files: {_missing}")

oracle_df = pd.read_csv(ORACLE_CSV)
oracle_pos_set = set(oracle_df[oracle_df['oracle_full_avg'] > 0]['ticker'].values)
oracle_total_pnl = oracle_df[oracle_df['oracle_full_avg'] > 0]['oracle_full_total'].sum()
print(f"    Oracle: {len(oracle_pos_set)} positive tickers, {oracle_total_pnl*100:+.1f}% total PnL ceiling")

oracle_trades_df = pd.read_csv(ORACLE_TRADES)
oracle_trades_total = oracle_trades_df[oracle_trades_df['ticker'].isin(oracle_pos_set)]['pnl'].sum()
print(f"    Oracle trades total: {oracle_trades_total*100:+.1f}%")


# ==============================================================================
# PHASE 4: STRATEGY SWEEP
# ==============================================================================
print(f"\n  PHASE 4: Exhaustive strategy sweep...")

gate_raw = []
for mf in range(1, 11):
    gate_raw.append((mf, 'any', 0.0, 0.0, 0))
for ar in ['any', 'both', 'a2_only', 'a1_only']:
    gate_raw.append((1, ar, 0.0, 0.0, 0))
for vc in [0.0, 0.4, 0.6, 0.8]:
    gate_raw.append((3, 'any', vc, 0.0, 0))
for sa in [0.0, 0.3, 0.5]:
    gate_raw.append((3, 'any', 0.0, sa, 0))
combined_gates = [
    (4, 'both', 0.0, 0.0, 0), (5, 'both', 0.0, 0.0, 0),
    (3, 'both', 0.6, 0.0, 0), (4, 'both', 0.6, 0.0, 0),
    (3, 'any', 0.6, 0.3, 0),  (4, 'any', 0.6, 0.3, 0),
    (3, 'both', 0.0, 0.3, 0), (5, 'any', 0.8, 0.0, 0),
    (3, 'any', 0.4, 0.5, 0),  (6, 'both', 0.6, 0.3, 0),
]
for c in combined_gates:
    gate_raw.append(c)
for op in [4, 6, 8, 10]:
    gate_raw.append((op, 'any', 0.0, 0.0, op))
    gate_raw.append((op, 'both', 0.0, 0.0, op))
for op in [6, 8]:
    gate_raw.append((op, 'any', 0.6, 0.0, op))
    gate_raw.append((op, 'any', 0.0, 0.3, op))

seen_gates = set()
unique_gates = []
for mf, ar, vc, sa, op in gate_raw:
    key = (mf, ar, vc, sa, op)
    if key not in seen_gates:
        seen_gates.add(key)
        unique_gates.append({'min_folds': mf, 'alpha_req': ar,
                             'va_cons_min': vc, 'seed_agree_min': sa,
                             'min_opt_pos': op})

n_base = len(unique_gates) * len(CONFIG_METHODS) * len(SIGNAL_METHODS) + \
        len(FOLD_GROUPS) * len(CONFIG_METHODS) * len(SIGNAL_METHODS)
n_total = n_base * len(THRESH_MULTS)
print(f"    Base strategies: {n_base} × {len(THRESH_MULTS)} threshold mults = {n_total} total")

all_results = []
n_done = 0

for mult in THRESH_MULTS:
    mult_label = f"{mult:.2f}x"
    fire_dates = fold_fire_dates_by_mult[mult]
    common_args = dict(ticker_features=ticker_features, fold_configs=fold_configs,
                       daily_probs=daily_probs, fold_fire_dates=fire_dates,
                       date_trade_pnl=date_trade_pnl, cross_best_folds=cross_best_folds,
                       thresh_mult=mult, h1_pnl=h1_pnl,
                       fold_configs_worst=fold_configs_worst,
                       date_trade_pnl_fixed=date_trade_pnl_fixed)

    print(f"\n    ── Threshold mult {mult_label} ──")
    print(f"    Running gate-based strategies...")
    for gi, gp in enumerate(unique_gates):
        g_label = make_gate_label(gp)
        for cm in CONFIG_METHODS:
            for sm in SIGNAL_METHODS:
                trades = collect_trades(gp, cm, sm, 'ALL_12', **common_args)
                metrics = evaluate_trades(trades, oracle_pos_set, oracle_trades_total, h2_start_dt)
                if metrics:
                    metrics.update({'gate': g_label, 'config': cm, 'signal': sm,
                                    'fold_group': 'ALL_12', 'thresh_mult': mult,
                                    '_gate_params': gp})
                    all_results.append(metrics)
                n_done += 1
        if (gi + 1) % 10 == 0:
            print(f"      {gi+1}/{len(unique_gates)} gates done...")

    print(f"    Running fold-group strategies...")
    for fg_name in FOLD_GROUPS:
        for cm in CONFIG_METHODS:
            for sm in SIGNAL_METHODS:
                trades = collect_trades(None, cm, sm, fg_name, **common_args)
                metrics = evaluate_trades(trades, oracle_pos_set, oracle_trades_total, h2_start_dt)
                if metrics:
                    metrics.update({'gate': 'ANY', 'config': cm, 'signal': sm,
                                    'fold_group': fg_name, 'thresh_mult': mult,
                                    '_gate_params': None})
                    all_results.append(metrics)
                n_done += 1

    n_mult = sum(1 for r in all_results if r.get('thresh_mult') == mult)
    print(f"    {mult_label}: {n_mult} strategies with trades")

print(f"\n    TOTAL: {n_done} evaluated, {len(all_results)} with trades")


# ==============================================================================
# PHASE 5: OUTPUT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 5: RESULTS")
print(f"{'='*W}")

def _thr(r):
    return f"{r.get('thresh_mult', 1.0):.2f}"

results_sorted = sorted(all_results, key=lambda r: r['total_pnl'], reverse=True)

# H2-only combo score for primary ranking
for r in all_results:
    r['h2_combo'] = r['h2_avg'] * r['h2_total'] if (r['h2_avg'] > 0 and r['h2_total'] > 0) else 0.0
h2_ranked = sorted(all_results, key=lambda r: r['h2_combo'], reverse=True)

print(f"\n  ── TOP 30 STRATEGIES BY H2 COMBO SCORE (avg × total, H2 ONLY) ──")
print(f"  {'#':>3} │{'Thr':>5}│{'FGroup':>10}│{'Gate':>22}│{'Config':>12}│{'Signal':>14}│"
      f"{'Tks':>4}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_WR':>6}{'H2_Tot%':>9}│{'Combo':>7}│{'Tr/Tk/Mo':>9}")
print(f"  {'─'*3}─┼{'─'*5}┼{'─'*10}┼{'─'*22}┼{'─'*12}┼{'─'*14}┼{'─'*9}┼{'─'*23}┼{'─'*7}┼{'─'*9}")

for i, r in enumerate(h2_ranked[:30]):
    if r['h2_combo'] <= 0:
        break
    h2_wr = 0.0
    tr_mo = r['h2_n'] / max(r['n_tickers'], 1) / 2
    print(f"  {i+1:3d} │{_thr(r):>5}│{r['fold_group']:>10}│{r['gate']:>22}│{r['config']:>12}│{r['signal']:>14}│"
          f"{r['n_tickers']:>4}{r['h2_n']:>5}│"
          f"{r['h2_avg']*100:>+7.2f}{r['wr']*100:>5.0f}%{r['h2_total']*100:>+8.1f}│"
          f"{r['h2_combo']*10000:>6.1f}│{tr_mo:>8.1f}")

# ── KEY ANALYSIS: H2 WITH HIGH AVG PnL ──
print(f"\n  ══ H2 FOCUS: STRATEGIES WITH HIGH AVG PnL IN JUN-JUL ══")

h2_pos = [r for r in all_results if r['h2_total'] > 0 and r['h2_n'] > 0]
print(f"  {len(h2_pos)} strategies with positive H2 PnL")

# Top 25 by H2 avg PnL (minimum 10 H2 trades to be meaningful)
h2_avg_sorted = sorted([r for r in h2_pos if r['h2_n'] >= 10],
                        key=lambda r: r['h2_avg'], reverse=True)
print(f"\n  ── TOP 25 H2 STRATEGIES BY AVG PnL (min 10 H2 trades) ──")
print(f"  {'#':>3} │{'Thr':>5}│{'FGroup':>10}│{'Gate':>22}│{'Config':>12}│{'Signal':>14}│"
      f"{'Tks':>4}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_WR':>6}{'H2_Tot%':>9}│{'Full%':>8}")
print(f"  {'─'*3}─┼{'─'*5}┼{'─'*10}┼{'─'*22}┼{'─'*12}┼{'─'*14}┼{'─'*9}┼{'─'*23}┼{'─'*8}")
for i, r in enumerate(h2_avg_sorted[:25]):
    h2_wr = sum(1 for t in sorted(r.get('monthly', {}).keys()) if t >= '2026-06') / max(r['h2_n'], 1) * 100
    print(f"  {i+1:3d} │{_thr(r):>5}│{r['fold_group']:>10}│{r['gate']:>22}│{r['config']:>12}│{r['signal']:>14}│"
          f"{r['n_tickers']:>4}{r['h2_n']:>5}│"
          f"{r['h2_avg']*100:>+7.2f}{r['wr']*100:>5.0f}%{r['h2_total']*100:>+8.1f}│"
          f"{r['total_pnl']*100:>+7.1f}")

# SWEET SPOT: H2 total > +30% AND H2 avg > +0.5%
print(f"\n  ── H2 SWEET SPOT: total > +30% AND avg > +0.50% ──")
sweet = sorted([r for r in h2_pos if r['h2_total'] >= 0.30 and r['h2_avg'] >= 0.005],
               key=lambda r: r['h2_avg'], reverse=True)
print(f"  {len(sweet)} strategies found")
if sweet:
    print(f"  {'#':>3} │{'Thr':>5}│{'FGroup':>10}│{'Gate':>22}│{'Config':>12}│{'Signal':>14}│"
          f"{'Tks':>4}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'Full%':>8}│{'Tr/Tk/Mo':>9}")
    print(f"  {'─'*3}─┼{'─'*5}┼{'─'*10}┼{'─'*22}┼{'─'*12}┼{'─'*14}┼{'─'*9}┼{'─'*17}┼{'─'*8}┼{'─'*9}")
    for i, r in enumerate(sweet[:20]):
        tr_per_tk_mo = r['h2_n'] / max(r['n_tickers'], 1) / 2
        print(f"  {i+1:3d} │{_thr(r):>5}│{r['fold_group']:>10}│{r['gate']:>22}│{r['config']:>12}│{r['signal']:>14}│"
              f"{r['n_tickers']:>4}{r['h2_n']:>5}│"
              f"{r['h2_avg']*100:>+7.2f}{r['h2_total']*100:>+8.1f}│"
              f"{r['total_pnl']*100:>+7.1f}│{tr_per_tk_mo:>8.1f}")

# DEPLOYMENT CANDIDATES: H2 avg > +0.50% AND H2 total > +10%, ranked by H2 combo
print(f"\n  ── DEPLOYMENT CANDIDATES (H2 ONLY): avg > +0.50% AND total > +10% ──")
deploy = sorted([r for r in h2_pos if r['h2_avg'] >= 0.005 and r['h2_total'] >= 0.10],
                key=lambda r: r['h2_combo'], reverse=True)
print(f"  {len(deploy)} strategies found")
if deploy:
    print(f"  {'#':>3} │{'Thr':>5}│{'FGroup':>10}│{'Gate':>22}│{'Config':>12}│{'Signal':>14}│"
          f"{'Tks':>4}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'Combo':>7}│{'Tr/Tk/Mo':>9}")
    for i, r in enumerate(deploy[:20]):
        tr_per_tk_mo = r['h2_n'] / max(r['n_tickers'], 1) / 2
        print(f"  {i+1:3d} │{_thr(r):>5}│{r['fold_group']:>10}│{r['gate']:>22}│{r['config']:>12}│{r['signal']:>14}│"
              f"{r['n_tickers']:>4}{r['h2_n']:>5}│"
              f"{r['h2_avg']*100:>+7.2f}{r['h2_total']*100:>+8.1f}│"
              f"{r['h2_combo']*10000:>6.1f}│{tr_per_tk_mo:>8.1f}")

# THRESHOLD MULTIPLIER IMPACT on best base strategies
print(f"\n  ── THRESHOLD MULTIPLIER IMPACT ──")
print(f"  For top base strategies (at 1.0x), show effect of raising threshold:")
base_top = sorted([r for r in all_results if r.get('thresh_mult', 1.0) == 1.0],
                  key=lambda r: r['h2_total'], reverse=True)[:5]
for base_r in base_top:
    base_key = (base_r['fold_group'], base_r['gate'], base_r['config'], base_r['signal'])
    print(f"\n  {base_key[0]}|{base_key[1]}|{base_key[2]}|{base_key[3]}:")
    print(f"  {'Mult':>6} {'Tks':>4} {'Trades':>7} {'Avg%':>7} {'WR':>5} {'Total%':>9} "
          f"{'H2_N':>5} {'H2_Avg%':>8} {'H2_Tot%':>9} {'Tr/Tk/Mo':>9}")
    for mult in THRESH_MULTS:
        variants = [r for r in all_results
                    if r['fold_group'] == base_key[0] and r['gate'] == base_key[1]
                    and r['config'] == base_key[2] and r['signal'] == base_key[3]
                    and r.get('thresh_mult', 1.0) == mult]
        if variants:
            r = variants[0]
            tr_mo = r['h2_n'] / max(r['n_tickers'], 1) / 2
            print(f"  {mult:>5.2f}x {r['n_tickers']:>4} {r['n_trades']:>7} "
                  f"{r['avg_pnl']*100:>+6.2f} {r['wr']*100:>4.0f}% {r['total_pnl']*100:>+8.1f} "
                  f"{r['h2_n']:>5} {r['h2_avg']*100:>+7.2f} {r['h2_total']*100:>+8.1f} {tr_mo:>8.1f}")
        else:
            print(f"  {mult:>5.2f}x  (no trades)")

# Top 25 by H2 total (classic view)
h2_sorted = sorted(all_results, key=lambda r: r['h2_total'], reverse=True)
print(f"\n  ── TOP 25 STRATEGIES BY HALF2 (Jun-Jul) TOTAL PnL ──")
print(f"  {'#':>3} │{'Thr':>5}│{'FGroup':>10}│{'Gate':>22}│{'Config':>12}│{'Signal':>14}│"
      f"{'Tks':>4}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'Full%':>9}")
print(f"  {'─'*3}─┼{'─'*5}┼{'─'*10}┼{'─'*22}┼{'─'*12}┼{'─'*14}┼{'─'*9}┼{'─'*17}┼{'─'*9}")
for i, r in enumerate(h2_sorted[:25]):
    print(f"  {i+1:3d} │{_thr(r):>5}│{r['fold_group']:>10}│{r['gate']:>22}│{r['config']:>12}│{r['signal']:>14}│"
          f"{r['n_tickers']:>4}{r['h2_n']:>5}│"
          f"{r['h2_avg']*100:>+7.2f}{r['h2_total']*100:>+8.1f}│"
          f"{r['total_pnl']*100:>+8.1f}")

# Per-ticker breakdown helper
def _print_per_ticker_breakdown(label, strat_r, fold_configs, ticker_features,
                                 daily_probs, fold_fire_dates_by_mult,
                                 date_trade_pnl, cross_best_folds, oracle_pos_set, h2_start_dt,
                                 h1_pnl=None, fold_configs_worst=None,
                                 date_trade_pnl_fixed=None):
    tm = strat_r.get('thresh_mult', 1.0)
    print(f"\n  ── {label} PER-TICKER BREAKDOWN ──")
    print(f"  Strategy: {_thr(strat_r)}x|{strat_r['fold_group']}|{strat_r['gate']}|"
          f"{strat_r['config']}|{strat_r['signal']}")
    fire_dates = fold_fire_dates_by_mult[tm]
    args = dict(ticker_features=ticker_features, fold_configs=fold_configs,
                daily_probs=daily_probs, fold_fire_dates=fire_dates,
                date_trade_pnl=date_trade_pnl, cross_best_folds=cross_best_folds,
                thresh_mult=tm, h1_pnl=h1_pnl, fold_configs_worst=fold_configs_worst,
                date_trade_pnl_fixed=date_trade_pnl_fixed)
    trades = collect_trades(strat_r.get('_gate_params'), strat_r['config'],
                            strat_r['signal'], strat_r['fold_group'], **args)
    if not trades:
        print("  (no trades)")
        return
    tdf = pd.DataFrame(trades)
    tdf['is_h2'] = tdf['date'].apply(lambda d: d >= h2_start_dt)
    tk_fold = tdf.groupby('ticker')['fold'].first()
    tk_stats = tdf.groupby('ticker').agg(
        n=('pnl', 'count'), total=('pnl', 'sum'),
        avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean()),
        h2_n=('is_h2', 'sum'),
        h2_total=('pnl', lambda x: x[tdf.loc[x.index, 'is_h2']].sum()),
    ).sort_values('h2_total', ascending=False)
    print(f"  {'Ticker':>8} {'Fold':>5} {'Mode':>8} {'TP%':>5} {'SL%':>5} "
          f"{'Tr':>4} {'Avg%':>7} {'WR':>5} {'Tot%':>8} "
          f"{'H2_Tr':>5} {'H2_Tot%':>8} {'Oracle?':>8}")
    for t, row in tk_stats.iterrows():
        is_oracle = '★' if t in oracle_pos_set else ''
        f_used = int(tk_fold.get(t, 0))
        cfg = fold_configs.get((t, f_used), {})
        mode = cfg.get('tpsl_mode', '?')[:8]
        tp = cfg.get('tp_pct', 0) * 100
        sl = cfg.get('sl_pct', 0) * 100
        print(f"  {t:>8} {f_used:>5} {mode:>8} {tp:>4.1f}% {sl:>4.1f}% "
              f"{int(row['n']):>4} {row['avg']*100:>+6.2f} {row['wr']*100:>4.0f}% "
              f"{row['total']*100:>+7.1f} {int(row['h2_n']):>5} "
              f"{row['h2_total']*100:>+7.1f} {is_oracle:>8}")

_bd_args = dict(fold_configs=fold_configs, ticker_features=ticker_features,
                daily_probs=daily_probs, fold_fire_dates_by_mult=fold_fire_dates_by_mult,
                date_trade_pnl=date_trade_pnl, cross_best_folds=cross_best_folds,
                oracle_pos_set=oracle_pos_set, h2_start_dt=h2_start_dt,
                h1_pnl=h1_pnl, fold_configs_worst=fold_configs_worst,
                date_trade_pnl_fixed=date_trade_pnl_fixed)

# Run breakdowns for key strategies
if results_sorted:
    _print_per_ticker_breakdown("BEST OVERALL (by total PnL)", results_sorted[0], **_bd_args)
if h2_sorted:
    _print_per_ticker_breakdown("BEST H2 TOTAL", h2_sorted[0], **_bd_args)
if sweet:
    _print_per_ticker_breakdown("BEST H2 SWEET-SPOT", sweet[0], **_bd_args)
best_1x = sorted([r for r in all_results if r.get('thresh_mult', 1.0) == 1.0],
                 key=lambda r: r['h2_total'], reverse=True)
if best_1x:
    _print_per_ticker_breakdown("BEST 1.0x THRESHOLD (most diversified)", best_1x[0], **_bd_args)

# Category comparison
print(f"\n  ── CATEGORY COMPARISON ──")
categories = {
    'Best overall':    results_sorted[0] if results_sorted else None,
    'Best H2 total':   h2_sorted[0] if h2_sorted else None,
    'Best H2 avg':     max(h2_pos, key=lambda r: r['h2_avg']) if h2_pos else None,
    'Best H2 sweet':   sweet[0] if sweet else None,
    'Best Sharpe':     max(all_results, key=lambda r: r['sharpe']) if all_results else None,
}
for cat, r in categories.items():
    if r is None:
        continue
    lbl = f"{_thr(r)}x|{r['fold_group']}|{r['gate']}|{r['config']}|{r['signal']}"
    print(f"  {cat:>16}: {lbl:<60} total={r['total_pnl']*100:>+7.1f}% "
          f"H2={r['h2_total']*100:>+6.1f}% h2_avg={r['h2_avg']*100:>+.2f}%")

# ── PARETO FRONTIER: BEST THRESHOLD PER BASE STRATEGY ──
print(f"\n  ══ PARETO: OPTIMAL THRESHOLD PER BASE STRATEGY ══")
print(f"  For each base strategy, find the threshold that maximizes avg PnL × total PnL")

strat_families = defaultdict(list)
for r in all_results:
    base_key = (r['fold_group'], r['gate'], r['config'], r['signal'])
    strat_families[base_key].append(r)

pareto_rows = []
for base_key, variants in strat_families.items():
    if not variants:
        continue
    by_mult = {r.get('thresh_mult', 1.0): r for r in variants}
    base = by_mult.get(1.0)
    if base is None or base['n_trades'] < 5:
        continue
    best_avg_r = max(variants, key=lambda r: r['avg_pnl'] if r['n_trades'] >= 5 else -999)
    best_tot_r = max(variants, key=lambda r: r['total_pnl'])
    h2_pos_variants = [r for r in variants if r['h2_n'] >= 3 and r['h2_avg'] > 0 and r['h2_total'] > 0]
    best_h2_avg_r = max(h2_pos_variants, key=lambda r: r['h2_avg']) if h2_pos_variants else max(variants, key=lambda r: r['h2_avg'] if r['h2_n'] >= 3 else -999)
    best_h2_combo_r = max(h2_pos_variants, key=lambda r: r['h2_avg'] * r['h2_total']) if h2_pos_variants else variants[0]
    pareto_rows.append({
        'base_key': base_key,
        'base_total': base['total_pnl'],
        'base_avg': base['avg_pnl'],
        'base_h2_avg': base['h2_avg'],
        'base_h2_total': base['h2_total'],
        'base_trades': base['n_trades'],
        'best_avg_mult': best_avg_r.get('thresh_mult', 1.0),
        'best_avg_val': best_avg_r['avg_pnl'],
        'best_avg_total': best_avg_r['total_pnl'],
        'best_avg_trades': best_avg_r['n_trades'],
        'best_tot_mult': best_tot_r.get('thresh_mult', 1.0),
        'best_tot_val': best_tot_r['total_pnl'],
        'best_h2a_mult': best_h2_avg_r.get('thresh_mult', 1.0),
        'best_h2a_val': best_h2_avg_r['h2_avg'],
        'best_h2a_total': best_h2_avg_r['h2_total'],
        'best_h2a_trades': best_h2_avg_r['h2_n'],
        'best_h2c_mult': best_h2_combo_r.get('thresh_mult', 1.0),
        'best_h2c_avg': best_h2_combo_r['h2_avg'],
        'best_h2c_total': best_h2_combo_r['h2_total'],
        'best_h2c_trades': best_h2_combo_r['h2_n'],
        'h2_combo_score': best_h2_combo_r['h2_avg'] * best_h2_combo_r['h2_total'] if (h2_pos_variants and best_h2_combo_r['h2_avg'] > 0 and best_h2_combo_r['h2_total'] > 0) else 0,
        'variants': by_mult,
    })

# Sort by H2 combo score (avg × total — rewards both dimensions)
pareto_sorted = sorted(pareto_rows, key=lambda p: p['h2_combo_score'], reverse=True)

print(f"\n  ── TOP 30 BASE STRATEGIES: BEST H2 THRESHOLD (by avg×total score) ──")
print(f"  Each row shows the base (1.0x) and the best threshold for H2 avg×total")
print(f"  {'#':>3} │{'FGroup':>10}│{'Gate':>18}│{'Config':>12}│{'Signal':>14}│"
      f"{'Base':>9}│{'→Thr':>5}│{'Trades':>7}│{'Avg%':>7}{'Tot%':>8}│"
      f"{'H2_Avg%':>8}{'H2_Tot%':>9}│{'Score':>7}")
print(f"  {'─'*3}─┼{'─'*10}┼{'─'*18}┼{'─'*12}┼{'─'*14}┼{'─'*9}┼{'─'*5}┼{'─'*7}┼{'─'*15}┼{'─'*17}┼{'─'*7}")
for i, p in enumerate(pareto_sorted[:30]):
    bk = p['base_key']
    print(f"  {i+1:3d} │{bk[0]:>10}│{bk[1]:>18}│{bk[2]:>12}│{bk[3]:>14}│"
          f"{p['base_total']*100:>+8.1f}│{p['best_h2c_mult']:>4.2f}x│{p['best_h2c_trades']:>7}│"
          f"{p['best_h2c_avg']*100:>+6.2f}{p['best_h2c_total']*100:>+7.1f}│"
          f"{p['best_h2a_val']*100:>+7.2f}{p['best_h2a_total']*100:>+8.1f}│"
          f"{p['h2_combo_score']*10000:>6.1f}")

# Full Pareto curves for top 10 base strategies
print(f"\n  ── FULL PARETO CURVES: TOP 10 BY H2 COMBO SCORE ──")
for i, p in enumerate(pareto_sorted[:10]):
    bk = p['base_key']
    print(f"\n  #{i+1} {bk[0]}|{bk[1]}|{bk[2]}|{bk[3]}:")
    print(f"  {'Mult':>6} {'Tks':>4} {'Trades':>7} {'Avg%':>7} {'WR':>5} {'Total%':>9} "
          f"{'H2_N':>5} {'H2_Avg%':>8} {'H2_Tot%':>9} {'Combo':>7} {'Tr/Tk/Mo':>9}")
    for mult in THRESH_MULTS:
        r = p['variants'].get(mult)
        if r:
            tr_mo = r['h2_n'] / max(r['n_tickers'], 1) / 2
            combo = r['h2_avg'] * r['h2_total'] * 10000 if r['h2_n'] >= 3 else 0
            marker = ' ◀' if mult == p['best_h2c_mult'] else ''
            print(f"  {mult:>5.2f}x {r['n_tickers']:>4} {r['n_trades']:>7} "
                  f"{r['avg_pnl']*100:>+6.2f} {r['wr']*100:>4.0f}% {r['total_pnl']*100:>+8.1f} "
                  f"{r['h2_n']:>5} {r['h2_avg']*100:>+7.2f} {r['h2_total']*100:>+8.1f} "
                  f"{combo:>6.1f} {tr_mo:>8.1f}{marker}")

# DEPLOYMENT-READY: strategies where raising threshold gets H2 avg > 1% while keeping total > 20%
print(f"\n  ── DEPLOYMENT-READY: threshold-optimized, H2 avg>1.0% AND H2 total>20% ──")
deploy_pareto = []
for p in pareto_sorted:
    for mult in THRESH_MULTS:
        r = p['variants'].get(mult)
        if r and r['h2_avg'] >= 0.01 and r['h2_total'] >= 0.20 and r['h2_n'] >= 5:
            deploy_pareto.append({
                'base_key': p['base_key'], 'mult': mult, 'r': r,
                'h2_score': r['h2_avg'] * r['h2_total'],
            })
deploy_pareto.sort(key=lambda d: d['h2_score'], reverse=True)
seen_bases = set()
deploy_unique = []
for d in deploy_pareto:
    if d['base_key'] not in seen_bases:
        seen_bases.add(d['base_key'])
        deploy_unique.append(d)
print(f"  {len(deploy_unique)} unique base strategies (best threshold each)")
if deploy_unique:
    print(f"  {'#':>3} │{'Thr':>5}│{'FGroup':>10}│{'Gate':>18}│{'Config':>12}│{'Signal':>14}│"
          f"{'Tks':>4}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'Full%':>8}│{'Tr/Tk/Mo':>9}")
    for i, d in enumerate(deploy_unique[:20]):
        r = d['r']
        bk = d['base_key']
        tr_mo = r['h2_n'] / max(r['n_tickers'], 1) / 2
        print(f"  {i+1:3d} │{d['mult']:>4.2f}x│{bk[0]:>10}│{bk[1]:>18}│{bk[2]:>12}│{bk[3]:>14}│"
              f"{r['n_tickers']:>4}{r['h2_n']:>5}│"
              f"{r['h2_avg']*100:>+7.2f}{r['h2_total']*100:>+8.1f}│"
              f"{r['total_pnl']*100:>+7.1f}│{tr_mo:>8.1f}")

# Threshold sensitivity for most_configs|single_best across fold counts
print(f"\n  ── THRESHOLD × FOLD-COUNT (config=most_configs, signal=single_best, ALL_12) ──")
print(f"  {'Mult':>6} {'F':>3} {'Tks':>4} {'Tr':>5} {'Avg%':>7} {'WR':>5} {'Total%':>9} "
      f"{'H2_N':>5} {'H2_Avg%':>8} {'H2_Tot%':>9} {'Tr/Tk/Mo':>9}")
for mult in THRESH_MULTS:
    for mf in [3, 5, 7, 9]:
        gl = make_gate_label({'min_folds': mf, 'alpha_req': 'any', 'va_cons_min': 0, 'seed_agree_min': 0})
        matches = [r for r in all_results
                   if r['gate'] == gl and r['config'] == 'most_configs'
                   and r['signal'] == 'single_best' and r['fold_group'] == 'ALL_12'
                   and r.get('thresh_mult', 1.0) == mult]
        if matches:
            r = matches[0]
            tr_mo = r['h2_n'] / max(r['n_tickers'], 1) / 2
            print(f"  {mult:>5.2f}x {mf:>3} {r['n_tickers']:>4} {r['n_trades']:>5} "
                  f"{r['avg_pnl']*100:>+6.2f} {r['wr']*100:>4.0f}% {r['total_pnl']*100:>+8.1f} "
                  f"{r['h2_n']:>5} {r['h2_avg']*100:>+7.2f} {r['h2_total']*100:>+8.1f} {tr_mo:>8.1f}")
    print()


# ── DIVERSIFIED DEPLOYMENT: min 3 tickers, max concentration < 60% ──
print(f"\n  ══ DIVERSIFIED DEPLOYMENT: ≥3 tickers, max concentration <60%, H2 avg>0.5% ══")
diversified = sorted(
    [r for r in all_results
     if r['n_tickers'] >= 3
     and r.get('max_concentration', 1.0) < 0.60
     and r['h2_avg'] >= 0.005
     and r['h2_n'] >= 10],
    key=lambda r: r['h2_avg'] * r['h2_total'], reverse=True)
print(f"  {len(diversified)} strategies found")
if diversified:
    print(f"  {'#':>3} │{'Thr':>5}│{'FGroup':>10}│{'Gate':>22}│{'Config':>12}│{'Signal':>14}│"
          f"{'Tks':>4}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'Full%':>8}│{'MaxConc':>8}")
    print(f"  {'─'*3}─┼{'─'*5}┼{'─'*10}┼{'─'*22}┼{'─'*12}┼{'─'*14}┼{'─'*9}┼{'─'*17}┼{'─'*8}┼{'─'*8}")
    for i, r in enumerate(diversified[:25]):
        print(f"  {i+1:3d} │{_thr(r):>5}│{r['fold_group']:>10}│{r['gate']:>22}│{r['config']:>12}│{r['signal']:>14}│"
              f"{r['n_tickers']:>4}{r['h2_n']:>5}│"
              f"{r['h2_avg']*100:>+7.2f}{r['h2_total']*100:>+8.1f}│"
              f"{r['total_pnl']*100:>+7.1f}│{r['max_concentration']*100:>6.0f}%")
    if diversified:
        _print_per_ticker_breakdown("BEST DIVERSIFIED", diversified[0], **_bd_args)

# ── FOLD AGREEMENT GATE ANALYSIS ──
print(f"\n  ══ FOLD AGREEMENT GATE SWEEP (OPT-positive folds as family substitute) ══")
print(f"  {'OP_min':>7} {'Alpha':>6} {'Tks':>4} (tickers passing gate at 1.0x/vote_2/best_opt)")
for op_min in [2, 4, 6, 8, 10]:
    for ar in ['any', 'both']:
        gp = {'min_folds': op_min, 'alpha_req': ar, 'va_cons_min': 0.0, 'seed_agree_min': 0.0, 'min_opt_pos': op_min}
        gl = make_gate_label(gp)
        matches = [r for r in all_results
                   if r['gate'] == gl and r['config'] == 'best_opt'
                   and r['signal'] == 'vote_2' and r['fold_group'] == 'ALL_12'
                   and r.get('thresh_mult', 1.0) == 1.0]
        if matches:
            r = matches[0]
            print(f"  OP>={op_min:2d}  {ar:>6}  {r['n_tickers']:>4} tks, {r['n_trades']:>4} tr, "
                  f"avg={r['avg_pnl']*100:>+.2f}%, H2={r['h2_avg']*100:>+.2f}%, "
                  f"H2_tot={r['h2_total']*100:>+.1f}%")
        else:
            print(f"  OP>={op_min:2d}  {ar:>6}  (no trades)")


# ==============================================================================
# PHASE 6: SAVE CSVs
# ==============================================================================
print(f"\n  PHASE 6: Saving results...")

csv_rows = []
for r in all_results:
    row = {k: v for k, v in r.items() if k not in ('tickers', 'monthly', '_gate_params')}
    monthly = r.get('monthly', {})
    for m in ['2026-04', '2026-05', '2026-06', '2026-07']:
        row[f'pnl_{m}'] = monthly.get(m, 0.0)
    csv_rows.append(row)

full_df = pd.DataFrame(csv_rows).sort_values('total_pnl', ascending=False)
full_df.to_csv(OUTPUT_ALL, index=False)
print(f"    All results: {OUTPUT_ALL} ({len(full_df)} strategies)")

top_df = full_df.head(50)
top_df.to_csv(OUTPUT_TOP, index=False)
print(f"    Top 50:      {OUTPUT_TOP}")

pareto_csv = '/tmp/oracle_vs_strategies_pareto.csv'
pareto_csv_rows = []
for p in pareto_sorted:
    bk = p['base_key']
    pareto_csv_rows.append({
        'fold_group': bk[0], 'gate': bk[1], 'config': bk[2], 'signal': bk[3],
        'base_total_pnl': p['base_total'], 'base_avg_pnl': p['base_avg'],
        'base_h2_avg': p['base_h2_avg'], 'base_h2_total': p['base_h2_total'],
        'base_trades': p['base_trades'],
        'best_h2_mult': p['best_h2c_mult'],
        'best_h2_avg': p['best_h2c_avg'], 'best_h2_total': p['best_h2c_total'],
        'best_h2_trades': p['best_h2c_trades'],
        'h2_combo_score': p['h2_combo_score'],
        'best_avg_mult': p['best_avg_mult'], 'best_avg_val': p['best_avg_val'],
    })
pd.DataFrame(pareto_csv_rows).to_csv(pareto_csv, index=False)
print(f"    Pareto:      {pareto_csv} ({len(pareto_csv_rows)} base strategies)")

# ── CONFIG METHOD HEAD-TO-HEAD ──
print(f"\n  ══ CONFIG METHOD HEAD-TO-HEAD (1.0x threshold, ALL_12, ANY gate, single_best signal) ══")
print(f"  {'Method':>14} {'Tks':>4} {'Tr':>5} {'Avg%':>7} {'WR':>5} {'Total%':>9} "
      f"{'H2_N':>5} {'H2_Avg%':>8} {'H2_Tot%':>9} {'Cap%':>5}")
for cm in CONFIG_METHODS:
    matches = [r for r in all_results
               if r['config'] == cm and r['signal'] == 'single_best'
               and r['fold_group'] == 'ALL_12' and r['gate'] == 'ANY'
               and r.get('thresh_mult', 1.0) == 1.0]
    if matches:
        r = matches[0]
        print(f"  {cm:>14} {r['n_tickers']:>4} {r['n_trades']:>5} "
              f"{r['avg_pnl']*100:>+6.2f} {r['wr']*100:>4.0f}% {r['total_pnl']*100:>+8.1f} "
              f"{r['h2_n']:>5} {r['h2_avg']*100:>+7.2f} {r['h2_total']*100:>+8.1f} "
              f"{r['oracle_capture']*100:>4.0f}%")
    else:
        print(f"  {cm:>14} (no trades)")

print(f"\n  ── CONFIG METHOD HEAD-TO-HEAD (1.0x, ALL_12, ANY gate, vote_2 signal) ──")
print(f"  {'Method':>14} {'Tks':>4} {'Tr':>5} {'Avg%':>7} {'WR':>5} {'Total%':>9} "
      f"{'H2_N':>5} {'H2_Avg%':>8} {'H2_Tot%':>9} {'Cap%':>5}")
for cm in CONFIG_METHODS:
    matches = [r for r in all_results
               if r['config'] == cm and r['signal'] == 'vote_2'
               and r['fold_group'] == 'ALL_12' and r['gate'] == 'ANY'
               and r.get('thresh_mult', 1.0) == 1.0]
    if matches:
        r = matches[0]
        print(f"  {cm:>14} {r['n_tickers']:>4} {r['n_trades']:>5} "
              f"{r['avg_pnl']*100:>+6.2f} {r['wr']*100:>4.0f}% {r['total_pnl']*100:>+8.1f} "
              f"{r['h2_n']:>5} {r['h2_avg']*100:>+7.2f} {r['h2_total']*100:>+8.1f} "
              f"{r['oracle_capture']*100:>4.0f}%")
    else:
        print(f"  {cm:>14} (no trades)")

# Best strategy per config method (by H2 combo score)
print(f"\n  ── BEST STRATEGY PER CONFIG METHOD (by H2 avg × H2 total, min 10 H2 trades) ──")
print(f"  {'Method':>14} │{'Thr':>5}│{'Gate':>18}│{'Signal':>14}│"
      f"{'Tks':>4}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'Full%':>8}")
for cm in CONFIG_METHODS:
    cm_results = [r for r in all_results if r['config'] == cm and r['h2_n'] >= 10 and r['h2_avg'] > 0]
    if cm_results:
        best_cm = max(cm_results, key=lambda r: r['h2_avg'] * r['h2_total'])
        print(f"  {cm:>14} │{_thr(best_cm):>5}│{best_cm['gate']:>18}│{best_cm['signal']:>14}│"
              f"{best_cm['n_tickers']:>4}{best_cm['h2_n']:>5}│"
              f"{best_cm['h2_avg']*100:>+7.2f}{best_cm['h2_total']*100:>+8.1f}│"
              f"{best_cm['total_pnl']*100:>+7.1f}")
    else:
        print(f"  {cm:>14} │ (no qualifying strategies)")

# Breakdowns for new methods
best_worst_opt = sorted([r for r in all_results if r['config'] == 'worst_opt' and r['h2_n'] >= 5],
                        key=lambda r: r['h2_avg'] * r['h2_total'] if r['h2_avg'] > 0 else -999,
                        reverse=True)
if best_worst_opt:
    _print_per_ticker_breakdown("BEST worst_opt STRATEGY", best_worst_opt[0], **_bd_args)

best_h1_oracle = sorted([r for r in all_results if r['config'] == 'h1_oracle' and r['h2_n'] >= 5],
                         key=lambda r: r['h2_avg'] * r['h2_total'] if r['h2_avg'] > 0 else -999,
                         reverse=True)
if best_h1_oracle:
    _print_per_ticker_breakdown("BEST h1_oracle STRATEGY", best_h1_oracle[0], **_bd_args)

best_fixed = sorted([r for r in all_results if r['config'] == 'fixed_tpsl' and r['h2_n'] >= 5],
                    key=lambda r: r['h2_avg'] * r['h2_total'] if r['h2_avg'] > 0 else -999,
                    reverse=True)
if best_fixed:
    _print_per_ticker_breakdown(f"BEST fixed_tpsl ({FIXED_MODE} TP={FIXED_TP*100:.0f}% SL={FIXED_SL*100:.0f}%)",
                                best_fixed[0], **_bd_args)


# ==============================================================================
# PHASE 7: TRADE-LEVEL FORENSICS — WHAT ACTUALLY HAPPENS AFTER ENTRY?
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 7: TRADE FORENSICS — WHAT DOES THE PRICE DO AFTER SHORT ENTRY?")
print(f"{'='*W}")

def forensic_trade(ohlcv, entry_idx, tp_pct, sl_pct, mode, sl_mult, grace_min):
    """Trace what happens after SHORT entry: max dip, max bounce, exit reason."""
    prices = ohlcv['prices']
    highs = ohlcv['highs']
    lows = ohlcv['lows']
    date_ints = ohlcv['dates']
    time_mins = ohlcv['times']
    n = len(prices)
    if entry_idx >= n - 1:
        return None
    entry_price = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    et = time_mins[entry_idx]

    # For atr_sl, compute actual SL used
    actual_sl = sl_pct
    if mode == 'atr_sl':
        atr_arr = ohlcv.get('atr_pct', np.full(n, 0.05))
        atr = float(atr_arr[entry_idx]) if atr_arr[entry_idx] > 0 else 0.05
        actual_sl = min(max(atr * sl_mult, 0.01), 0.25)

    max_dip = 0.0       # best possible SHORT PnL (max price drop from entry)
    max_bounce = 0.0     # worst adverse move BEFORE the max dip
    max_bounce_before_dip = 0.0
    running_max_bounce = 0.0
    eod_pnl = 0.0
    exit_reason = 'TIME'
    exit_pnl = 0.0
    exit_bar = 0
    dip_bar = 0

    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            eod_pnl = (entry_price - prices[k-1]) / entry_price
            exit_pnl = eod_pnl
            exit_reason = 'EOD'
            exit_bar = k - 1 - entry_idx
            break

        # Track price action
        bar_low_pct = (entry_price - lows[k]) / entry_price     # positive = price dropped (good for short)
        bar_high_pct = (highs[k] - entry_price) / entry_price   # positive = price bounced up (bad for short)

        if bar_low_pct > max_dip:
            max_dip = bar_low_pct
            max_bounce_before_dip = running_max_bounce
            dip_bar = k - entry_idx
        if bar_high_pct > running_max_bounce:
            running_max_bounce = bar_high_pct
        if bar_high_pct > max_bounce:
            max_bounce = bar_high_pct

        # Check exits
        target_price = entry_price * (1 - tp_pct)
        stop_price = entry_price * (1 + actual_sl)
        sl_active = True
        if mode == 'grace':
            sl_active = (time_mins[k] - et) >= grace_min

        tp_hit = lows[k] <= target_price
        sl_hit = sl_active and (highs[k] >= stop_price)

        if sl_hit and tp_hit:
            exit_reason = 'TP+SL'
            exit_pnl = tp_pct if np.random.random() < 0.5 else -actual_sl
            exit_bar = k - entry_idx
            break
        if sl_hit:
            exit_reason = 'SL'
            exit_pnl = -actual_sl
            exit_bar = k - entry_idx
            break
        if tp_hit:
            exit_reason = 'TP'
            exit_pnl = tp_pct
            exit_bar = k - entry_idx
            break
        if time_mins[k] >= EXIT_TIME_MIN:
            exit_reason = 'TIME'
            exit_pnl = (entry_price - prices[k]) / entry_price
            exit_bar = k - entry_idx
            break
    else:
        exit_pnl = (entry_price - prices[n-1]) / entry_price
        exit_reason = 'LAST'
        exit_bar = n - 1 - entry_idx

    exit_pnl -= 2 * SLIPPAGE_PCT

    return {
        'max_dip': max_dip,
        'max_bounce': max_bounce,
        'bounce_before_dip': max_bounce_before_dip,
        'exit_reason': exit_reason,
        'exit_pnl': exit_pnl,
        'tp_pct': tp_pct,
        'sl_pct': actual_sl,
        'money_left': max_dip - (exit_pnl + 2*SLIPPAGE_PCT) if exit_pnl >= 0 else max_dip,
        'dip_bar': dip_bar,
        'exit_bar': exit_bar,
    }

# ── 7A: FORENSIC ANALYSIS — ALL H2 SIGNAL DAYS ──
print(f"\n  ══ 7A: TRADE FORENSICS — EVERY H2 SIGNAL DAY ══")
print(f"  For each ticker × fold with H2 trades, trace what happened after entry.")
print(f"  MaxDip = max intraday drop from entry (best possible SHORT capture)")
print(f"  MaxBnc = max adverse bounce before exit")
print(f"  Left = money left on table (MaxDip - what we captured)")

all_forensics = []
for ticker in sorted(ticker_data.keys()):
    ohlcv = ticker_data[ticker]['ohlcv']
    entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]

    for fold in ALL_FOLDS:
        cfg = fold_configs.get((ticker, fold))
        probs = daily_probs.get(ticker, {}).get(fold)
        if cfg is None or not probs:
            continue

        for idx in entry_indices:
            d_int = int(ohlcv['dates'][idx])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
            if d_date < h2_start_dt or d_date > test_end_dt:
                continue
            prob = probs.get(d_date)
            if prob is None or prob < cfg['prob_thresh']:
                continue

            f_result = forensic_trade(ohlcv, idx, cfg['tp_pct'], cfg['sl_pct'],
                                       cfg['tpsl_mode'], cfg['sl_mult'], cfg['grace_minutes'])
            if f_result:
                f_result['ticker'] = ticker
                f_result['fold'] = fold
                f_result['date'] = d_date
                f_result['prob'] = prob
                f_result['mode'] = cfg['tpsl_mode']
                all_forensics.append(f_result)

print(f"  Analyzed {len(all_forensics)} H2 trades across all ticker-fold pairs")

if all_forensics:
    fdf = pd.DataFrame(all_forensics)

    # Aggregate stats
    print(f"\n  ── AGGREGATE H2 TRADE FORENSICS ──")
    print(f"  {'Metric':<30} {'Mean':>8} {'Median':>8} {'P10':>8} {'P90':>8}")
    for col, label in [('max_dip', 'Max dip (best possible)'),
                       ('max_bounce', 'Max adverse bounce'),
                       ('bounce_before_dip', 'Bounce BEFORE max dip'),
                       ('exit_pnl', 'Actual exit PnL'),
                       ('money_left', 'Money left on table')]:
        vals = fdf[col].values
        print(f"  {label:<30} {np.mean(vals)*100:>+7.2f}% {np.median(vals)*100:>+7.2f}% "
              f"{np.percentile(vals,10)*100:>+7.2f}% {np.percentile(vals,90)*100:>+7.2f}%")

    print(f"\n  Exit reason distribution:")
    for reason in ['TP', 'SL', 'TIME', 'EOD', 'TP+SL']:
        cnt = (fdf['exit_reason'] == reason).sum()
        if cnt > 0:
            subset = fdf[fdf['exit_reason'] == reason]
            print(f"    {reason:>5}: {cnt:>5} ({cnt/len(fdf)*100:>5.1f}%)  "
                  f"avg_pnl={subset['exit_pnl'].mean()*100:>+6.2f}%  "
                  f"avg_max_dip={subset['max_dip'].mean()*100:>+6.2f}%  "
                  f"avg_left={subset['money_left'].mean()*100:>+6.2f}%")

    # Key question: on SL trades, how big was the dip AFTER we got stopped out?
    sl_trades = fdf[fdf['exit_reason'] == 'SL']
    if len(sl_trades) > 0:
        print(f"\n  ── SL TRADES: DID THE PRICE DROP AFTER STOP-OUT? ──")
        print(f"  {len(sl_trades)} SL trades. After getting stopped out:")
        dip_after_sl = sl_trades['max_dip'].values
        print(f"    Max dip that DID occur: avg {np.mean(dip_after_sl)*100:+.2f}%, "
              f"median {np.median(dip_after_sl)*100:+.2f}%")
        big_dip_after = (dip_after_sl > 0.03).sum()
        print(f"    Trades where price dipped >3% after entry (we just got stopped first): "
              f"{big_dip_after}/{len(sl_trades)} ({big_dip_after/len(sl_trades)*100:.0f}%)")
        print(f"    Bounce that triggered SL: avg {sl_trades['max_bounce'].mean()*100:+.2f}%")

    # False positives: signal fired but price went UP
    no_dip = fdf[fdf['max_dip'] < 0.01]
    print(f"\n  ── FALSE POSITIVES: signal fired, price never dipped >1% ──")
    print(f"  {len(no_dip)}/{len(fdf)} trades ({len(no_dip)/len(fdf)*100:.0f}%) "
          f"had max dip < 1%")
    big_dip = fdf[fdf['max_dip'] >= 0.05]
    print(f"  {len(big_dip)}/{len(fdf)} trades ({len(big_dip)/len(fdf)*100:.0f}%) "
          f"had max dip >= 5% (the big morning drops)")
    if len(big_dip) > 0:
        print(f"    Of those {len(big_dip)} big-dip trades:")
        print(f"      Captured (exit_pnl > 0):     {(big_dip['exit_pnl']>0).sum()}")
        print(f"      Lost (exit_pnl <= 0):         {(big_dip['exit_pnl']<=0).sum()}")
        print(f"      Avg exit PnL:                 {big_dip['exit_pnl'].mean()*100:+.2f}%")
        print(f"      Avg max dip:                  {big_dip['max_dip'].mean()*100:+.2f}%")
        print(f"      Avg money left on table:      {big_dip['money_left'].mean()*100:+.2f}%")
        print(f"      Avg bounce before dip:        {big_dip['bounce_before_dip'].mean()*100:+.2f}%")
        sl_in_big = big_dip[big_dip['exit_reason'] == 'SL']
        if len(sl_in_big) > 0:
            print(f"      SL exits on big-dip days:     {len(sl_in_big)} — bounced {sl_in_big['max_bounce'].mean()*100:+.2f}% first, "
                  f"SL was {sl_in_big['sl_pct'].mean()*100:.1f}%")

    # ── 7B: PER-TICKER FORENSICS ──
    print(f"\n  ══ 7B: PER-TICKER H2 FORENSICS ══")
    print(f"  {'Ticker':<8}{'N':>4}│{'AvgDip':>7}{'AvgBnc':>7}{'AvgPnL':>7}│"
          f"{'%SL':>5}{'%TP':>5}{'%Time':>6}│"
          f"{'BigDip':>7}{'CapBig':>7}│{'AvgLeft':>8}")
    print(f"  {'─'*8}{'─'*4}┼{'─'*21}┼{'─'*16}┼{'─'*14}┼{'─'*8}")

    per_ticker = fdf.groupby('ticker')
    for ticker in sorted(per_ticker.groups.keys(),
                          key=lambda t: per_ticker.get_group(t)['max_dip'].mean(), reverse=True):
        g = per_ticker.get_group(ticker)
        n = len(g)
        n_sl = (g['exit_reason'] == 'SL').sum()
        n_tp = (g['exit_reason'] == 'TP').sum()
        n_time = (g['exit_reason'].isin(['TIME', 'EOD'])).sum()
        n_big = (g['max_dip'] >= 0.05).sum()
        n_cap_big = ((g['max_dip'] >= 0.05) & (g['exit_pnl'] > 0)).sum()
        print(f"  {ticker:<8}{n:>4}│"
              f"{g['max_dip'].mean()*100:>+6.2f}%{g['max_bounce'].mean()*100:>+6.2f}%{g['exit_pnl'].mean()*100:>+6.2f}%│"
              f"{n_sl/n*100:>4.0f}%{n_tp/n*100:>4.0f}%{n_time/n*100:>5.0f}%│"
              f"{n_big:>6}{n_cap_big:>6}{'/' if n_big else ' '}{n_big if n_big else '':>1}│"
              f"{g['money_left'].mean()*100:>+7.2f}%")

    # ── 7C: WORST LOSSES — DETAILED TRADE LOG ──
    print(f"\n  ══ 7C: BIGGEST LOSSES WHERE PRICE ACTUALLY DROPPED ══")
    print(f"  Trades where max_dip > 3% but we LOST money — SL triggered before the drop")
    bad_luck = fdf[(fdf['max_dip'] > 0.03) & (fdf['exit_pnl'] < 0)].sort_values('money_left', ascending=False)
    if len(bad_luck) > 0:
        print(f"  {len(bad_luck)} trades found")
        print(f"  {'Ticker':<8}{'Date':>12}{'Fold':>5}{'Mode':>7}│"
              f"{'MaxDip':>7}{'MaxBnc':>7}{'BncB4':>7}│"
              f"{'ExitPnL':>8}{'Exit':>5}│{'TP':>5}{'SL':>5}│{'Left':>7}")
        for _, t in bad_luck.head(30).iterrows():
            print(f"  {t['ticker']:<8}{str(t['date']):>12}{t['fold']:>5}{t['mode']:>7}│"
                  f"{t['max_dip']*100:>+6.2f}%{t['max_bounce']*100:>+6.2f}%{t['bounce_before_dip']*100:>+6.2f}%│"
                  f"{t['exit_pnl']*100:>+7.2f}%{t['exit_reason']:>5}│"
                  f"{t['tp_pct']*100:>4.0f}%{t['sl_pct']*100:>4.0f}%│"
                  f"{t['money_left']*100:>+6.2f}%")
    else:
        print(f"  None found")

    # ── 7D: WHAT TP/SL WOULD HAVE CAPTURED THE BIG DROPS? ──
    print(f"\n  ══ 7D: OPTIMAL SL FOR BIG-DIP DAYS ══")
    print(f"  On days where price dipped >5%, what SL was needed to survive the bounce?")
    big_dip_all = fdf[fdf['max_dip'] >= 0.05]
    if len(big_dip_all) > 0:
        bounces = big_dip_all['bounce_before_dip'].values
        print(f"  {len(big_dip_all)} big-dip trades (max_dip >= 5%)")
        print(f"  Bounce BEFORE the big dip:")
        for p in [50, 75, 90, 95, 99]:
            pv = np.percentile(bounces, p)
            survived = (bounces <= pv).sum()
            print(f"    P{p:02d}: {pv*100:>+6.2f}%  "
                  f"(SL at {pv*100:.1f}% would survive {survived}/{len(bounces)} = {survived/len(bounces)*100:.0f}%)")
        print(f"\n  → Recommended minimum SL to capture big drops: "
              f"{np.percentile(bounces, 90)*100:.1f}% "
              f"(survives 90% of pre-dip bounces)")

    # ── 7E: SUMMARY DIAGNOSIS ──
    print(f"\n  ══ 7E: ROOT CAUSE SUMMARY ══")
    n_total_f = len(fdf)
    n_sl_f = (fdf['exit_reason'] == 'SL').sum()
    n_tp_f = (fdf['exit_reason'] == 'TP').sum()
    avg_dip = fdf['max_dip'].mean()
    avg_pnl_f = fdf['exit_pnl'].mean()
    avg_left = fdf['money_left'].mean()
    pct_false = len(fdf[fdf['max_dip'] < 0.01]) / n_total_f * 100
    pct_big = len(fdf[fdf['max_dip'] >= 0.05]) / n_total_f * 100
    pct_sl = n_sl_f / n_total_f * 100

    print(f"  {n_total_f} H2 trades analyzed:")
    print(f"    Avg max dip available:    {avg_dip*100:+.2f}%")
    print(f"    Avg PnL captured:         {avg_pnl_f*100:+.2f}%")
    print(f"    Avg money left on table:  {avg_left*100:+.2f}%")
    print(f"    False positive rate:      {pct_false:.0f}% (price never dipped >1%)")
    print(f"    Big drop rate:            {pct_big:.0f}% (max dip >= 5%)")
    print(f"    SL exit rate:             {pct_sl:.0f}%")
    if avg_dip > 0.03 and pct_sl > 30:
        print(f"\n    DIAGNOSIS: Big drops exist ({avg_dip*100:.1f}% avg dip) but SL is too tight.")
        print(f"    {pct_sl:.0f}% of trades exit via SL. Widen SL or use grace mode.")
    elif pct_false > 40:
        print(f"\n    DIAGNOSIS: Too many false positives ({pct_false:.0f}%). Model signal quality is the bottleneck.")
        print(f"    Config optimization can't fix a signal that fires on flat days.")
    elif avg_left > 0.03:
        print(f"\n    DIAGNOSIS: TP is too tight. Avg {avg_left*100:.1f}% left on table.")
        print(f"    Widen TP to capture more of the available drop.")

print(f"\n{'='*W}")
print(f"  FINAL SUMMARY (H2 ONLY — Jun-Jul deployment evaluation)")
print(f"{'='*W}")
print(f"  Strategies evaluated:  {n_done}")
print(f"  With trades:           {len(all_results)}")

# H2 oracle ceiling
oracle_h2_trades = oracle_trades_df[
    (oracle_trades_df['ticker'].isin(oracle_pos_set)) &
    (oracle_trades_df['period'] == 'HALF2')
]
oracle_h2_total = oracle_h2_trades['pnl'].sum() if len(oracle_h2_trades) > 0 else 0.0
oracle_h2_avg = oracle_h2_trades['pnl'].mean() if len(oracle_h2_trades) > 0 else 0.0
oracle_h2_n = len(oracle_h2_trades)
print(f"  Oracle H2 ceiling:     {oracle_h2_total*100:+.1f}% total, "
      f"{oracle_h2_avg*100:+.2f}% avg, {oracle_h2_n} trades")

best_h2 = h2_ranked[0] if h2_ranked and h2_ranked[0]['h2_combo'] > 0 else None
if best_h2:
    h2_cap = best_h2['h2_total'] / oracle_h2_total if oracle_h2_total > 0 else 0.0
    print(f"\n  BEST H2 strategy:      {best_h2['h2_total']*100:+.1f}% H2 total, "
          f"{best_h2['h2_avg']*100:+.2f}% avg ({h2_cap*100:.0f}% H2 capture)")
    print(f"    → {_thr(best_h2)}x|{best_h2['fold_group']}|{best_h2['gate']}|{best_h2['config']}|{best_h2['signal']}")
    print(f"    → {best_h2['n_tickers']} tickers, {best_h2['h2_n']} H2 trades")

if diversified:
    bd = diversified[0]
    print(f"\n  BEST DIVERSIFIED:      {bd['h2_total']*100:+.1f}% H2 total, "
          f"{bd['h2_avg']*100:+.2f}% H2 avg, {bd['n_tickers']} tickers, "
          f"conc={bd['max_concentration']*100:.0f}%")
    print(f"    → {_thr(bd)}x|{bd['fold_group']}|{bd['gate']}|{bd['config']}|{bd['signal']}")
for cm_name, cm_list in [('worst_opt', best_worst_opt), ('h1_oracle', best_h1_oracle), ('fixed_tpsl', best_fixed)]:
    if cm_list:
        r = cm_list[0]
        print(f"\n  BEST {cm_name:>14}:  {r['h2_total']*100:+.1f}% H2 total, "
              f"{r['h2_avg']*100:+.2f}% H2 avg, {r['n_tickers']} tickers")
        print(f"    → {_thr(r)}x|{r['fold_group']}|{r['gate']}|{r['config']}|{r['signal']}")
print(f"")
print(f"  Run: python3 -u oracle_vs_strategies.py > /tmp/oracle_vs_strat.log 2>&1")
print(f"{'='*W}")
