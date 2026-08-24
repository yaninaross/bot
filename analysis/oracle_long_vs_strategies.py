# %% ── Oracle vs Strategies — Exhaustive LONG Model Evaluation ────────────────
#
# Pre-computes per-experiment trades once, then sweeps ~1200 unbiased strategies
# across 3 axes: ticker selection × config selection × signal method.
# Compares each to the TRUE ORACLE ceiling and both OLD/NEW S3 baselines.
#
# Outputs:
#   /tmp/oracle_long_vs_strategies_results.csv  — all strategy results
#   /tmp/oracle_long_vs_strategies_top.csv      — top 50 strategies
#   /tmp/oracle_long_vs_strategies_pareto.csv   — pareto by H2 combo score
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
BUCKET_NAME    = "time-series-bucket-ys"
RESULTS_DIR    = "Algos/Results_scored_20260703"
RESULTS_FALLBACK = "Algos/Results_all_v2"
INFERENCE_DIR  = "Data/Inference_v2"

TEST_START  = '2026-04-01'
TEST_END    = '2026-08-04'
HALF1_END   = '2026-05-31'
HALF2_START = '2026-06-01'

EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.0015

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS_BASE = ('robust_wf_v12', 'robust_wf_v12_stitched')
BQ_MODEL_VERSION_LIKES = (
    'robust_wf_v12_17%', 'robust_wf_v12_18%',
    'robust_wf_v12_20%', 'robust_wf_v12_24%',
)

MAX_VA_OPT_RATIO = 3.0
DEPLOY_THRESHOLD = 0.018
MIN_SEEDS = 3
SKIP_MODES = {'regime_scaled', 'regime_adaptive'}

GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']

CONFIG_METHODS = ['best_opt', 'median_opt', 'most_configs', 'conservative',
                  'worst_opt', 'h1_oracle', 'fixed_tpsl', 's3_score']
SIGNAL_METHODS = ['single_best', 'ensemble_all', 'ensemble_top3',
                  'vote_2', 'vote_3', 'vote_majority']
THRESH_MULTS = [1.0, 1.25, 1.5, 2.0]

FIXED_TP   = 0.08
FIXED_SL   = 0.04
FIXED_MODE = 'limit_entry'
FIXED_DIP  = 0.005
FIXED_WAIT = 15

ALL_FAMILIES = ['152', '160', '170', '180', '200', '220', '240']

FAMILY_GROUPS = {
    'ALL':       ALL_FAMILIES,
    'BASE':      ['152', '160'],
    'WF_OLD':    ['170', '180'],
    'WF_NEW':    ['200', '220', '240'],
    'WF_ALL':    ['170', '180', '200', '220', '240'],
    'CORE':      ['152', '160', '180', '200'],
    'CROSS_BEST': None,
}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120

OUTPUT_ALL     = '/tmp/oracle_long_vs_strategies_results.csv'
OUTPUT_TOP     = '/tmp/oracle_long_vs_strategies_top.csv'
OUTPUT_PARETO  = '/tmp/oracle_long_vs_strategies_pareto.csv'


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
    all_results = {}
    for results_dir in [RESULTS_FALLBACK, RESULTS_DIR]:
        prefix = f"{results_dir}/Universal_Exp{experiment_base}/{clean}/Seed_"
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            continue
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
            d = d.sort_index()
            if seed_id in all_results:
                old = all_results[seed_id]
                combined = pd.concat([old, d])
                combined = combined[~combined.index.duplicated(keep='last')]
                all_results[seed_id] = combined.sort_index()
            else:
                all_results[seed_id] = d
    return all_results


def compute_daily_prob(seed_scores, entry_min):
    all_daily = defaultdict(list)
    for seed_id, sdf in seed_scores.items():
        if isinstance(sdf, pd.Series):
            bar_mins = sdf.index.hour * 60 + sdf.index.minute
            entry_vals = sdf[bar_mins == entry_min]
            for dt, val in entry_vals.items():
                all_daily[dt.date()].append(float(val))
        else:
            bar_mins = sdf.index.hour * 60 + sdf.index.minute
            entry_vals = sdf[bar_mins == entry_min]
            for dt, sr in entry_vals.iterrows():
                all_daily[dt.date()].append(float(sr['Prob_Spike']))
    return {d: np.mean(ps) for d, ps in all_daily.items()}


# ==============================================================================
# LONG TRADE SIMULATION (buy-side)
# ==============================================================================
def run_long_trade_static(prices, highs, lows, date_ints, time_mins,
                          entry_idx, tp, sl, exit_time):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    ep = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    tgt = ep * (1 + tp)
    stp = ep * (1 - sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            return (prices[k-1] - ep) / ep, ed, True
        sl_h = lows[k] <= stp
        tp_h = highs[k] >= tgt
        if sl_h and tp_h:
            return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_h: return -sl, ed, True
        if tp_h: return tp, ed, True
        if time_mins[k] >= exit_time:
            return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


def run_long_trade_trailing_decay(prices, highs, lows, date_ints, time_mins,
                                  entry_idx, atr_pct_val, trail_start_mult,
                                  trail_end_mult, sl_mult, exit_time):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    ep = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    atr_pct = atr_pct_val if (atr_pct_val == atr_pct_val and atr_pct_val > 0) else 0.05
    sl = min(max(atr_pct * sl_mult, 0.01), 0.25)
    hard_stop = ep * (1 - sl)
    ts = atr_pct * trail_start_mult
    te = atr_pct * trail_end_mult
    hwm = ep
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            return (prices[k-1] - ep) / ep, ed, True
        if highs[k] > hwm:
            hwm = highs[k]
        unr = (hwm - ep) / ep
        trailing_stop = 0.0
        if unr > 0 and ts > 0:
            r = min(unr / ts, 1.0)
            trailing_stop = hwm * (1.0 - (ts + (te - ts) * r))
        eff = max(hard_stop, trailing_stop)
        if lows[k] <= eff:
            return ((trailing_stop - ep) / ep if trailing_stop > hard_stop else -sl), ed, True
        if time_mins[k] >= exit_time:
            return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


def run_long_trade_grace(prices, highs, lows, date_ints, time_mins,
                         entry_idx, tp, sl, exit_time, grace_minutes):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    ep = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    et = time_mins[entry_idx]
    tgt = ep * (1 + tp)
    stp = ep * (1 - sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            return (prices[k-1] - ep) / ep, ed, True
        elapsed = time_mins[k] - et
        sl_active = elapsed >= grace_minutes
        tp_h = highs[k] >= tgt
        sl_h = lows[k] <= stp
        if sl_active and sl_h and tp_h:
            return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_active and sl_h: return -sl, ed, True
        if tp_h: return tp, ed, True
        if time_mins[k] >= exit_time:
            return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


def run_long_trade_limit_entry(prices, highs, lows, date_ints, time_mins,
                               entry_idx, tp, sl, exit_time, dip_pct, max_wait):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    sp = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    et = time_mins[entry_idx]
    lp = sp * (1.0 - dip_pct)
    fill_idx = -1
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed: break
        if time_mins[k] - et > max_wait: break
        if lows[k] <= lp:
            fill_idx = k
            break
    if fill_idx < 0:
        return 0.0, ed, False
    ep = lp
    tgt = ep * (1.0 + tp)
    stp = ep * (1.0 - sl)
    for k in range(fill_idx + 1, n):
        if date_ints[k] != ed:
            return (prices[k-1] - ep) / ep, ed, True
        sl_h = lows[k] <= stp
        tp_h = highs[k] >= tgt
        if sl_h and tp_h:
            return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_h: return -sl, ed, True
        if tp_h: return tp, ed, True
        if time_mins[k] >= exit_time:
            return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


def execute_trade(ohlcv, entry_idx, cfg):
    prices    = ohlcv['prices']
    highs     = ohlcv['highs']
    lows      = ohlcv['lows']
    date_ints = ohlcv['dates']
    time_mins = ohlcv['times']
    mode = cfg['tpsl_mode']
    tp   = cfg['tp_pct']
    sl   = cfg['sl_pct']

    if mode == 'limit_entry':
        pnl, td, filled = run_long_trade_limit_entry(
            prices, highs, lows, date_ints, time_mins,
            entry_idx, tp, sl, EXIT_TIME_MIN,
            cfg.get('dip_pct', 0.005), cfg.get('max_wait_minutes', 15))
    elif mode == 'grace':
        pnl, td, filled = run_long_trade_grace(
            prices, highs, lows, date_ints, time_mins,
            entry_idx, tp, sl, EXIT_TIME_MIN, cfg.get('grace_minutes', 10))
    elif mode == 'trailing_decay':
        atr_arr = ohlcv.get('atr_pct', np.full(len(prices), 0.05))
        pnl, td, filled = run_long_trade_trailing_decay(
            prices, highs, lows, date_ints, time_mins,
            entry_idx, float(atr_arr[entry_idx]),
            cfg.get('trail_start_mult', 0), cfg.get('trail_end_mult', 0),
            cfg.get('sl_mult', 0), EXIT_TIME_MIN)
    elif mode == 'atr_sl':
        atr_arr = ohlcv.get('atr_pct', np.full(len(prices), 0.05))
        atr_val = float(atr_arr[entry_idx])
        atr_val = atr_val if atr_val > 0 else 0.05
        computed_sl = min(max(atr_val * cfg.get('sl_mult', 1.0), 0.01), 0.25)
        pnl, td, filled = run_long_trade_static(
            prices, highs, lows, date_ints, time_mins,
            entry_idx, tp, computed_sl, EXIT_TIME_MIN)
    elif mode in ('atr', 'atr_adaptive'):
        atr_arr = ohlcv.get('atr_pct', np.full(len(prices), 0.05))
        atr_val = float(atr_arr[entry_idx])
        atr_val = atr_val if atr_val > 0 else 0.05
        computed_tp = min(max(atr_val * cfg.get('tp_mult', 1.0), 0.005), 0.30)
        computed_sl = min(max(atr_val * cfg.get('sl_mult', 1.0), 0.01), 0.25)
        pnl, td, filled = run_long_trade_static(
            prices, highs, lows, date_ints, time_mins,
            entry_idx, computed_tp, computed_sl, EXIT_TIME_MIN)
    else:
        pnl, td, filled = run_long_trade_static(
            prices, highs, lows, date_ints, time_mins,
            entry_idx, tp, sl, EXIT_TIME_MIN)
    if filled:
        pnl -= 2 * SLIPPAGE_PCT
    return pnl, td, filled


# ==============================================================================
# HELPERS
# ==============================================================================
_MODE_SUFFIXES = ('_ATRSL', '_GRACE', '_LIMIT', '_ATR', '_ATRADP',
                  '_REGIME', '_TRAIL', '_REGADP', '_XSECTOR')

def _base_experiment(exp):
    for sfx in sorted(_MODE_SUFFIXES, key=len, reverse=True):
        if exp.endswith(sfx):
            return exp[:-len(sfx)]
    return exp

def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    if 200 <= num <= 205: return '200'
    if 220 <= num <= 225: return '220'
    if 240 <= num <= 245: return '240'
    return 'other'

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
    if params['min_families'] > 1:
        parts.append(f"F>={params['min_families']}")
    if params['va_cons_min'] > 0:
        parts.append(f"VC>={params['va_cons_min']:.0%}")
    if params['seed_agree_min'] > 0:
        parts.append(f"SA>={params['seed_agree_min']:.0%}")
    if params.get('fam_agree_min', 0) > 0:
        parts.append(f"FA>={params['fam_agree_min']:.0%}")
    return ','.join(parts) if parts else 'ANY'


def gate_passes(feat, available_exps, params):
    fams = set(feat['exp_family'].get(e, 'other') for e in available_exps)
    fams.discard('other')
    if len(fams) < params['min_families']:
        return False
    if params['va_cons_min'] > 0 and feat.get('va_consistency', 0) < params['va_cons_min']:
        return False
    if params['seed_agree_min'] > 0 and feat.get('seed_agreement', 0) < params['seed_agree_min']:
        return False
    fa_min = params.get('fam_agree_min', 0)
    if fa_min > 0:
        if feat.get('fam_agree_pct', 0) < fa_min:
            return False
    return True


def select_config_exp(ticker, feat, exp_configs, available, method,
                      h1_pnl=None, exp_configs_worst=None):
    if not available:
        return None
    if method == 'best_opt':
        best, best_o = None, -999
        for e in available:
            cfg = exp_configs.get((ticker, e))
            if cfg and cfg['opt_avg_pnl'] > best_o:
                best_o = cfg['opt_avg_pnl']
                best = e
        return best
    elif method == 'worst_opt':
        src = exp_configs_worst if exp_configs_worst else exp_configs
        best, best_o = None, 999
        for e in available:
            cfg = src.get((ticker, e))
            if cfg and cfg['opt_avg_pnl'] < best_o:
                best_o = cfg['opt_avg_pnl']
                best = e
        return best
    elif method == 'h1_oracle':
        if h1_pnl is None:
            return None
        best, best_h1 = None, -999
        for e in available:
            h1 = h1_pnl.get((ticker, e), 0.0)
            if h1 > best_h1:
                best_h1 = h1
                best = e
        return best
    elif method == 's3_score':
        best, best_s = None, -999
        for e in available:
            cfg = exp_configs.get((ticker, e))
            if cfg:
                score = (cfg['mean_va_pnl'] ** 2) * cfg.get('agree_18', 0)
                if score > best_s:
                    best_s = score
                    best = e
        return best
    elif method == 'fixed_tpsl':
        best, best_cnt = None, -1
        for e in available:
            cnt = feat.get('bq_counts', {}).get(e, 0)
            if cnt > best_cnt:
                best_cnt = cnt
                best = e
        return best
    elif method == 'median_opt':
        fv = [(exp_configs[(ticker, e)]['opt_avg_pnl'], e)
              for e in available if (ticker, e) in exp_configs]
        if not fv: return None
        fv.sort()
        return fv[len(fv) // 2][1]
    elif method == 'most_configs':
        best, best_cnt = None, -1
        for e in available:
            cnt = feat.get('bq_counts', {}).get(e, 0)
            if cnt > best_cnt:
                best_cnt = cnt
                best = e
        return best
    elif method == 'conservative':
        best, best_tp = None, 999.0
        for e in available:
            cfg = exp_configs.get((ticker, e))
            if cfg and cfg['tp_pct'] < best_tp:
                best_tp = cfg['tp_pct']
                best = e
        return best
    return None


def compute_signal_dates(ticker, selected_exp, available, daily_probs,
                         exp_configs, exp_fire_dates, method, thresh_mult=1.0):
    if method == 'single_best':
        return exp_fire_dates.get((ticker, selected_exp), set())

    elif method == 'ensemble_all':
        fps, thresholds = [], []
        for e in available:
            fp = daily_probs.get(ticker, {}).get(e)
            cfg = exp_configs.get((ticker, e))
            if fp and cfg:
                fps.append(fp)
                thresholds.append(cfg['prob_thresh'])
        if not fps: return set()
        med_thresh = float(np.median(thresholds)) * thresh_mult
        all_dates = set()
        for fp in fps: all_dates |= set(fp.keys())
        result = set()
        for d in all_dates:
            vals = [fp[d] for fp in fps if d in fp]
            if vals and np.mean(vals) >= med_thresh:
                result.add(d)
        return result

    elif method == 'ensemble_top3':
        fv = []
        for e in available:
            fp = daily_probs.get(ticker, {}).get(e)
            cfg = exp_configs.get((ticker, e))
            if fp and cfg:
                fv.append((cfg['opt_avg_pnl'], e, fp, cfg['prob_thresh']))
        fv.sort(reverse=True)
        top = fv[:3]
        if not top: return set()
        med_thresh = float(np.median([t[3] for t in top])) * thresh_mult
        all_dates = set()
        for _, _, fp, _ in top: all_dates |= set(fp.keys())
        result = set()
        for d in all_dates:
            vals = [fp[d] for _, _, fp, _ in top if d in fp]
            if vals and np.mean(vals) >= med_thresh:
                result.add(d)
        return result

    elif method in ('vote_2', 'vote_3', 'vote_majority'):
        min_votes = {'vote_2': 2, 'vote_3': 3, 'vote_majority': max(2, len(available) // 2 + 1)}[method]
        all_dates = set()
        for e in available:
            all_dates |= exp_fire_dates.get((ticker, e), set())
        result = set()
        for d in all_dates:
            votes = sum(1 for e in available if d in exp_fire_dates.get((ticker, e), set()))
            if votes >= min_votes:
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
    sharpe = float(avg / pnls.std() * np.sqrt(252)) if n > 1 and pnls.std() > 0 else 0.0
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


def collect_trades(gate_params, config_method, signal_method, family_group_name,
                   ticker_features, exp_configs, daily_probs, exp_fire_dates,
                   date_trade_pnl, cross_best_exps, thresh_mult=1.0,
                   h1_pnl=None, exp_configs_worst=None,
                   date_trade_pnl_fixed=None):
    trades = []
    use_fixed = (config_method == 'fixed_tpsl')
    for ticker in sorted(ticker_features.keys()):
        feat = ticker_features[ticker]
        if family_group_name == 'CROSS_BEST':
            available = cross_best_exps.get(ticker, [])
        elif family_group_name == 'ALL':
            available = feat['available_exps']
        else:
            pool_fams = set(FAMILY_GROUPS.get(family_group_name, ALL_FAMILIES))
            available = [e for e in feat['available_exps'] if feat['exp_family'].get(e) in pool_fams]
        if not available:
            continue
        if gate_params is not None and not gate_passes(feat, available, gate_params):
            continue
        sel_exp = select_config_exp(ticker, feat, exp_configs, available, config_method,
                                    h1_pnl=h1_pnl, exp_configs_worst=exp_configs_worst)
        if sel_exp is None:
            continue
        sig_dates = compute_signal_dates(ticker, sel_exp, available,
                                         daily_probs, exp_configs, exp_fire_dates,
                                         signal_method, thresh_mult=thresh_mult)
        if use_fixed:
            pnl_map = date_trade_pnl_fixed.get(ticker, {}) if date_trade_pnl_fixed else {}
        else:
            pnl_map = date_trade_pnl.get((ticker, sel_exp), {})
        for d in sig_dates:
            if d in pnl_map:
                trades.append({'date': d, 'ticker': ticker, 'pnl': pnl_map[d], 'exp': sel_exp})
    return trades


# ==============================================================================
# PHASE 1: LOAD DATA
# ==============================================================================
np.random.seed(42)

print(f"{'='*W}")
print(f"  ORACLE vs STRATEGIES — LONG Model Evaluation")
print(f"  TEST: {TEST_START} → {TEST_END}")
print(f"  Axes: {len(CONFIG_METHODS)} configs × {len(SIGNAL_METHODS)} signals × gates × family-groups")
print(f"{'='*W}")

print(f"\n  PHASE 1: Loading data...")
base_sql = ', '.join(f"'{v}'" for v in BQ_MODEL_VERSIONS_BASE)
like_clauses = '\n       '.join(f"OR model_version LIKE '{p}'" for p in BQ_MODEL_VERSION_LIKES)
q = f"""
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult, grace_minutes,
       dip_pct, max_wait_minutes, vol_filter_pct,
       opt_avg_pnl, opt_win_rate, opt_count, opt_freq,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, is_ensemble, survived_a, model_version
FROM `{BQ_TABLE}`
WHERE (model_version IN ({base_sql})
       {like_clauses})
  AND survived_a = TRUE
  AND val_a_count >= 3
  AND opt_avg_pnl > 0
  AND val_a_avg_pnl > 0
"""
bq_df = bq_client.query(q).to_dataframe()
print(f"    BQ: {len(bq_df)} rows")

bq_df = bq_df[bq_df['val_a_avg_pnl'] / bq_df['opt_avg_pnl'] <= MAX_VA_OPT_RATIO]
bq_df = bq_df[~bq_df['tpsl_mode'].isin(SKIP_MODES)]
print(f"    After filters: {len(bq_df)} rows ({bq_df['ticker'].nunique()} tickers)")

bq_df['base_exp'] = bq_df['experiment'].apply(_base_experiment)
bq_df['family'] = bq_df['experiment'].apply(assign_family)

# Aggregate across seeds to get per-config stats
for k in GRP_KEYS:
    if k not in bq_df.columns:
        bq_df[k] = 0
    else:
        bq_df[k] = bq_df[k].fillna(0)

indiv_df = bq_df[bq_df['is_ensemble'] == False].copy()
agg = indiv_df.groupby(GRP_KEYS).agg(
    n_seeds=('seed', 'nunique'),
    n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= DEPLOY_THRESHOLD).sum()),
    mean_opt=('opt_avg_pnl', 'mean'),
    mean_va_pnl=('val_a_avg_pnl', 'mean'),
    mean_va_wr=('val_a_win_rate', 'mean'),
    mean_va_count=('val_a_count', 'mean'),
    mean_vb=('val_b_avg_pnl', 'mean'),
).reset_index()
agg['agree_18'] = (agg['n_pos_seeds_18'] / agg['n_seeds']).clip(upper=1.0)
agg['base_exp'] = agg['experiment'].apply(_base_experiment)
agg['family'] = agg['experiment'].apply(assign_family)

# Best limit_entry config per (ticker, base_exp)
agg_limit = agg[(agg['tpsl_mode'] == 'limit_entry') & (agg['n_seeds'] >= MIN_SEEDS)].copy()

exp_configs = {}
exp_configs_worst = {}
for (ticker, base_exp), grp in agg_limit.groupby(['ticker', 'base_exp']):
    best_idx = grp['mean_opt'].idxmax()
    worst_idx = grp['mean_opt'].idxmin()
    best = grp.loc[best_idx]
    worst = grp.loc[worst_idx]
    def _to_cfg(row):
        return {
            'tpsl_mode': 'limit_entry',
            'prob_thresh': float(row['prob_thresh']),
            'tp_pct': float(row['tp_pct']),
            'sl_pct': float(row['sl_pct']),
            'tp_mult': float(row.get('tp_mult', 0)),
            'sl_mult': float(row.get('sl_mult', 0)),
            'trail_start_mult': float(row.get('trail_start_mult', 0)),
            'trail_end_mult': float(row.get('trail_end_mult', 0)),
            'grace_minutes': int(row.get('grace_minutes', 0)),
            'dip_pct': float(row.get('dip_pct', 0.005)),
            'max_wait_minutes': int(row.get('max_wait_minutes', 15)),
            'entry_minute': int(row['entry_minute']),
            'opt_avg_pnl': float(row['mean_opt']),
            'mean_va_pnl': float(row['mean_va_pnl']),
            'agree_18': float(row['agree_18']),
            'mean_vb': float(row['mean_vb']),
            'experiment': row['experiment'],
            'family': assign_family(row['experiment']),
            'n_seeds': int(row['n_seeds']),
        }
    exp_configs[(ticker, base_exp)] = _to_cfg(best)
    exp_configs_worst[(ticker, base_exp)] = _to_cfg(worst)

n_tickers_ec = len(set(t for t, e in exp_configs.keys()))
print(f"    exp_configs: {len(exp_configs)} (ticker, base_exp) pairs, {n_tickers_ec} tickers")

bq_counts = bq_df.groupby(['ticker', 'base_exp']).size().to_dict()
seed_agree_by_te = {}
for (ticker, base_exp), grp in indiv_df.groupby(['ticker', 'base_exp']):
    n_total = len(grp)
    n_agree = (grp['opt_avg_pnl'] >= DEPLOY_THRESHOLD).sum()
    seed_agree_by_te[(ticker, base_exp)] = n_agree / n_total if n_total > 0 else 0.0

all_tickers = sorted(set(t for t, e in exp_configs.keys()))
print(f"    {len(all_tickers)} tickers with surviving limit_entry configs")

# Load OHLCV
print(f"\n    Loading OHLCV...")
ticker_data = {}
for i, ticker in enumerate(all_tickers):
    df = load_master(ticker)
    if df is None: continue
    df = heal_data(df)
    ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
    if ohlcv is None: continue
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

# Load seed scores
print(f"\n    Loading seed scores...")
daily_probs = defaultdict(dict)
total_loaded = 0
base_exps_needed = sorted(set(e for _, e in exp_configs.keys()))
for i, be in enumerate(base_exps_needed):
    tickers_for_be = [t for t in all_tickers if (t, be) in exp_configs and t in ticker_data]
    for ticker in tickers_for_be:
        if be in daily_probs.get(ticker, {}):
            continue
        seeds = load_seed_scores(ticker, be)
        if not seeds:
            continue
        em = exp_configs[(ticker, be)]['entry_minute']
        dp = compute_daily_prob(seeds, em)
        if dp:
            daily_probs[ticker][be] = dp
            total_loaded += 1
        del seeds
    if (i + 1) % 20 == 0:
        print(f"      {i+1}/{len(base_exps_needed)} base_exps done ({total_loaded} loaded)")
        _release_memory()
_release_memory()
print(f"    Total: {total_loaded} (ticker, base_exp) probability sets")


# ==============================================================================
# PHASE 2: PRE-COMPUTE PER-EXPERIMENT TRADES + FEATURES
# ==============================================================================
print(f"\n  PHASE 2: Pre-computing per-experiment trades + ticker features...")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

date_trade_pnl = {}
exp_fire_dates_by_mult = {m: {} for m in THRESH_MULTS}
n_precomputed = 0

for ticker in sorted(ticker_data.keys()):
    ohlcv = ticker_data[ticker]['ohlcv']

    for be in base_exps_needed:
        cfg = exp_configs.get((ticker, be))
        probs = daily_probs.get(ticker, {}).get(be)
        if cfg is None or not probs:
            continue

        em = cfg['entry_minute']
        entry_indices = np.where(ohlcv['times'] == em)[0]

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
            pnl, td, filled = execute_trade(ohlcv, idx, cfg)
            if filled:
                pnl_by_date[d_date] = pnl
                last_td = d_int

        date_trade_pnl[(ticker, be)] = pnl_by_date

        for mult in THRESH_MULTS:
            fire = {d for d, p in probs.items()
                    if d >= test_start_dt and d <= test_end_dt
                    and p >= cfg['prob_thresh'] * mult}
            exp_fire_dates_by_mult[mult][(ticker, be)] = fire
        n_precomputed += 1

print(f"    Pre-computed {n_precomputed} (ticker, base_exp) trade sets × {len(THRESH_MULTS)} thresholds")

# Ticker features
ticker_features = {}
for ticker in sorted(ticker_data.keys()):
    avail = sorted([be for be in base_exps_needed
                    if (ticker, be) in exp_configs and be in daily_probs.get(ticker, {})])
    if not avail:
        continue
    exp_fam = {be: exp_configs[(ticker, be)]['family'] for be in avail}
    fams = set(exp_fam.values())
    fams.discard('other')
    opt_vals = [exp_configs[(ticker, be)]['opt_avg_pnl'] for be in avail]
    opt_pos = sum(1 for v in opt_vals if v > 0)
    sa_vals = [seed_agree_by_te.get((ticker, be), 0) for be in avail]
    vb_vals = [exp_configs[(ticker, be)]['mean_vb'] for be in avail]
    fam_vb = defaultdict(list)
    for be in avail:
        f = exp_fam[be]
        if f != 'other':
            fam_vb[f].append(exp_configs[(ticker, be)]['mean_vb'])
    n_fam_pos_vb = sum(1 for f, vbs in fam_vb.items() if np.mean(vbs) > 0)
    n_fam_total = len(fam_vb)

    ticker_features[ticker] = {
        'available_exps': avail,
        'exp_family': exp_fam,
        'n_exps': len(avail),
        'n_families': len(fams),
        'families': sorted(fams),
        'va_consistency': opt_pos / len(avail) if avail else 0,
        'va_median': float(np.median(opt_vals)),
        'seed_agreement': float(np.mean(sa_vals)) if sa_vals else 0,
        'fam_agree_pct': n_fam_pos_vb / n_fam_total if n_fam_total > 0 else 0,
        'bq_counts': {be: bq_counts.get((ticker, be), 0) for be in avail},
    }

# Cross-best: best exp from old families + best from new families
cross_best_exps = {}
for ticker, feat in ticker_features.items():
    old_fams = {'152', '160', '170', '180'}
    new_fams = {'200', '220', '240'}
    best_old, best_new = None, None
    best_old_opt, best_new_opt = -999, -999
    for e in feat['available_exps']:
        cfg = exp_configs.get((ticker, e))
        if cfg is None: continue
        fam = feat['exp_family'].get(e, 'other')
        ov = cfg['opt_avg_pnl']
        if fam in old_fams and ov > best_old_opt:
            best_old_opt, best_old = ov, e
        if fam in new_fams and ov > best_new_opt:
            best_new_opt, best_new = ov, e
    exps = []
    if best_old is not None: exps.append(best_old)
    if best_new is not None: exps.append(best_new)
    cross_best_exps[ticker] = exps

print(f"    {len(ticker_features)} tickers with features")
for nf_min in [1, 2, 3, 4, 5, 6, 7]:
    ct = sum(1 for f in ticker_features.values() if f['n_families'] >= nf_min)
    print(f"      >= {nf_min} families: {ct} tickers")

# H1 PnL per (ticker, base_exp)
h1_pnl = {}
for (ticker, be), pnl_map in date_trade_pnl.items():
    h1_total = sum(pnl for d, pnl in pnl_map.items() if d < h2_start_dt)
    h1_pnl[(ticker, be)] = h1_total

# Fixed TP/SL pre-computation
print(f"\n    Pre-computing fixed TP/SL trades (TP={FIXED_TP*100:.0f}%, SL={FIXED_SL*100:.0f}%, {FIXED_MODE})...")
fixed_cfg = {'tpsl_mode': FIXED_MODE, 'tp_pct': FIXED_TP, 'sl_pct': FIXED_SL,
             'dip_pct': FIXED_DIP, 'max_wait_minutes': FIXED_WAIT}
date_trade_pnl_fixed = {}
for ticker in sorted(ticker_data.keys()):
    ohlcv = ticker_data[ticker]['ohlcv']
    common_em = 375
    entry_indices = np.where(ohlcv['times'] == common_em)[0]
    pnl_by_date = {}
    last_td = -1
    for idx in entry_indices:
        d_int = int(ohlcv['dates'][idx])
        if d_int == last_td: continue
        d_str = str(d_int)
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < test_start_dt or d_date > test_end_dt: continue
        pnl, td, filled = execute_trade(ohlcv, idx, fixed_cfg)
        if filled:
            pnl_by_date[d_date] = pnl
            last_td = d_int
    if pnl_by_date:
        date_trade_pnl_fixed[ticker] = pnl_by_date
print(f"    Fixed trades: {len(date_trade_pnl_fixed)} tickers")

# OPT → TEST correlation diagnostic
print(f"\n    ── OPT → TEST CORRELATION DIAGNOSTIC ──")
from scipy import stats as scipy_stats
opt_vals_all, test_vals_all = [], []
for (ticker, be), pnl_map in date_trade_pnl.items():
    cfg = exp_configs.get((ticker, be))
    if cfg is None or not pnl_map: continue
    test_avg = np.mean(list(pnl_map.values()))
    opt_vals_all.append(cfg['opt_avg_pnl'])
    test_vals_all.append(test_avg)
if len(opt_vals_all) > 10:
    opt_a = np.array(opt_vals_all)
    test_a = np.array(test_vals_all)
    pearson_r = np.corrcoef(opt_a, test_a)[0, 1]
    spearman_r = scipy_stats.spearmanr(opt_a, test_a).statistic
    print(f"    OPT→TEST Pearson r  = {pearson_r:+.3f}  (n={len(opt_a)} ticker-exp pairs)")
    print(f"    OPT→TEST Spearman ρ = {spearman_r:+.3f}")
    if pearson_r < -0.05:
        print(f"    ⚠ ANTI-CORRELATION: higher OPT → worse TEST.")
    elif pearson_r < 0.1:
        print(f"    ⚠ NEAR ZERO: OPT has no predictive power.")
    else:
        print(f"    ✓ Positive: OPT-based selection should work.")

    h1_vals_corr, h2_vals_corr = [], []
    for (ticker, be), pnl_map in date_trade_pnl.items():
        cfg = exp_configs.get((ticker, be))
        if cfg is None or not pnl_map: continue
        h1_t = [p for d, p in pnl_map.items() if d < h2_start_dt]
        h2_t = [p for d, p in pnl_map.items() if d >= h2_start_dt]
        if h1_t and h2_t:
            h1_vals_corr.append(np.mean(h1_t))
            h2_vals_corr.append(np.mean(h2_t))
    if len(h1_vals_corr) > 10:
        h1_a = np.array(h1_vals_corr)
        h2_a = np.array(h2_vals_corr)
        h1h2_r = np.corrcoef(h1_a, h2_a)[0, 1]
        h1h2_rho = scipy_stats.spearmanr(h1_a, h2_a).statistic
        print(f"    H1→H2  Pearson r  = {h1h2_r:+.3f}  (n={len(h1_a)} ticker-exp pairs)")
        print(f"    H1→H2  Spearman ρ = {h1h2_rho:+.3f}")
        if h1h2_r > 0.1:
            print(f"    ✓ H1 predicts H2.")
        elif h1h2_r < -0.05:
            print(f"    ⚠ H1→H2 ANTI-CORRELATED.")
        else:
            print(f"    ⚠ H1→H2 near zero.")

_release_memory()


# ==============================================================================
# PHASE 3: COMPUTE ORACLE BASELINE (self-contained, no external files)
# ==============================================================================
print(f"\n  PHASE 3: Computing oracle baseline from pre-computed trades...")

# Oracle = best experiment per ticker (limit_entry only, from our data)
oracle_per_ticker = {}
for ticker in ticker_features:
    best_total, best_exp = -999, None
    for be in ticker_features[ticker]['available_exps']:
        pnl_map = date_trade_pnl.get((ticker, be), {})
        fire = exp_fire_dates_by_mult[1.0].get((ticker, be), set())
        total = sum(pnl_map.get(d, 0) for d in fire)
        if total > best_total:
            best_total, best_exp = total, be
    h2_pnls = []
    if best_exp:
        pnl_map = date_trade_pnl.get((ticker, best_exp), {})
        fire = exp_fire_dates_by_mult[1.0].get((ticker, best_exp), set())
        for d in fire:
            if d in pnl_map:
                if d >= h2_start_dt:
                    h2_pnls.append(pnl_map[d])
    oracle_per_ticker[ticker] = {
        'oracle_full': best_total, 'oracle_exp': best_exp,
        'oracle_h2': sum(h2_pnls) if h2_pnls else 0.0,
    }

oracle_pos_set = set(t for t, v in oracle_per_ticker.items() if v['oracle_full'] > 0)
oracle_total_pnl = sum(v['oracle_full'] for v in oracle_per_ticker.values() if v['oracle_full'] > 0)
oracle_h2_total = sum(v['oracle_h2'] for v in oracle_per_ticker.values() if v['oracle_h2'] > 0)
print(f"    Oracle (limit_entry): {len(oracle_pos_set)} positive tickers, "
      f"{oracle_total_pnl*100:+.1f}% total, H2={oracle_h2_total*100:+.1f}%")

# S3-equivalent baseline (s3_score config + family agreement gate)
# computed inline — this IS the production strategy
s3_gate = {'min_families': 2, 'va_cons_min': 0.0, 'seed_agree_min': 0.0, 'fam_agree_min': 0.5}
s3_trades = collect_trades(s3_gate, 's3_score', 'single_best', 'ALL',
                           ticker_features=ticker_features, exp_configs=exp_configs,
                           daily_probs=daily_probs,
                           exp_fire_dates=exp_fire_dates_by_mult[1.0],
                           date_trade_pnl=date_trade_pnl,
                           cross_best_exps=cross_best_exps, thresh_mult=1.0,
                           h1_pnl=h1_pnl, exp_configs_worst=exp_configs_worst,
                           date_trade_pnl_fixed=date_trade_pnl_fixed)
s3_metrics = evaluate_trades(s3_trades, oracle_pos_set, oracle_total_pnl, h2_start_dt)
if s3_metrics:
    print(f"    S3 equivalent: {s3_metrics['n_tickers']} tickers, {s3_metrics['n_trades']} trades, "
          f"avg={s3_metrics['avg_pnl']*100:+.2f}%, total={s3_metrics['total_pnl']*100:+.1f}%, "
          f"H2={s3_metrics['h2_total']*100:+.1f}%")
else:
    print(f"    S3 equivalent: no trades")
    s3_metrics = {'total_pnl': 0, 'avg_pnl': 0, 'h2_total': 0, 'n_tickers': 0, 'n_trades': 0}


# ==============================================================================
# PHASE 4: STRATEGY SWEEP
# ==============================================================================
print(f"\n  PHASE 4: Exhaustive strategy sweep...")

gate_raw = []
for mf in range(1, 8):
    gate_raw.append((mf, 0.0, 0.0, 0.0))
for vc in [0.0, 0.4, 0.6, 0.8]:
    gate_raw.append((2, vc, 0.0, 0.0))
for sa in [0.0, 0.3, 0.5]:
    gate_raw.append((2, 0.0, sa, 0.0))
for fa in [0.0, 0.4, 0.5, 0.6]:
    gate_raw.append((2, 0.0, 0.0, fa))
combined_gates = [
    (3, 0.4, 0.0, 0.5), (3, 0.6, 0.0, 0.5), (4, 0.0, 0.0, 0.5),
    (3, 0.0, 0.3, 0.5), (4, 0.6, 0.0, 0.0), (5, 0.0, 0.0, 0.5),
    (2, 0.6, 0.3, 0.0), (3, 0.4, 0.0, 0.6), (4, 0.0, 0.3, 0.5),
    (2, 0.0, 0.0, 0.6), (3, 0.0, 0.0, 0.6), (5, 0.0, 0.0, 0.0),
]
for c in combined_gates:
    gate_raw.append(c)

seen_gates = set()
unique_gates = []
for mf, vc, sa, fa in gate_raw:
    key = (mf, vc, sa, fa)
    if key not in seen_gates:
        seen_gates.add(key)
        unique_gates.append({'min_families': mf, 'va_cons_min': vc,
                             'seed_agree_min': sa, 'fam_agree_min': fa})

n_base = len(unique_gates) * len(CONFIG_METHODS) * len(SIGNAL_METHODS) + \
         len(FAMILY_GROUPS) * len(CONFIG_METHODS) * len(SIGNAL_METHODS)
n_total = n_base * len(THRESH_MULTS)
print(f"    Base strategies: {n_base} × {len(THRESH_MULTS)} threshold mults = {n_total} total")

all_results = []
n_done = 0

for mult in THRESH_MULTS:
    mult_label = f"{mult:.2f}x"
    fire_dates = exp_fire_dates_by_mult[mult]
    common_args = dict(ticker_features=ticker_features, exp_configs=exp_configs,
                       daily_probs=daily_probs, exp_fire_dates=fire_dates,
                       date_trade_pnl=date_trade_pnl, cross_best_exps=cross_best_exps,
                       thresh_mult=mult, h1_pnl=h1_pnl,
                       exp_configs_worst=exp_configs_worst,
                       date_trade_pnl_fixed=date_trade_pnl_fixed)

    print(f"\n    ── Threshold mult {mult_label} ──")
    print(f"    Running gate-based strategies...")
    for gi, gp in enumerate(unique_gates):
        g_label = make_gate_label(gp)
        for cm in CONFIG_METHODS:
            for sm in SIGNAL_METHODS:
                trades = collect_trades(gp, cm, sm, 'ALL', **common_args)
                metrics = evaluate_trades(trades, oracle_pos_set, oracle_total_pnl, h2_start_dt)
                if metrics:
                    metrics.update({'gate': g_label, 'config': cm, 'signal': sm,
                                    'family_group': 'ALL', 'thresh_mult': mult,
                                    '_gate_params': gp})
                    all_results.append(metrics)
                n_done += 1
        if (gi + 1) % 10 == 0:
            print(f"      {gi+1}/{len(unique_gates)} gates done...")

    print(f"    Running family-group strategies...")
    for fg_name in FAMILY_GROUPS:
        for cm in CONFIG_METHODS:
            for sm in SIGNAL_METHODS:
                trades = collect_trades(None, cm, sm, fg_name, **common_args)
                metrics = evaluate_trades(trades, oracle_pos_set, oracle_total_pnl, h2_start_dt)
                if metrics:
                    metrics.update({'gate': 'ANY', 'config': cm, 'signal': sm,
                                    'family_group': fg_name, 'thresh_mult': mult,
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

for r in all_results:
    r['h2_combo'] = r['h2_avg'] * r['h2_total'] if (r['h2_avg'] > 0 and r['h2_total'] > 0) else 0.0
h2_ranked = sorted(all_results, key=lambda r: r['h2_combo'], reverse=True)

print(f"\n  ── TOP 30 STRATEGIES BY TOTAL PnL ──")
print(f"  {'#':>3} │{'Thr':>5}│{'FamGrp':>12}│{'Gate':>22}│{'Config':>12}│{'Signal':>14}│"
      f"{'Tks':>4}{'N':>5}│{'Avg%':>8}{'WR':>5}{'Tot%':>9}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'Cap%':>5}")
print(f"  {'─'*3}─┼{'─'*5}┼{'─'*12}┼{'─'*22}┼{'─'*12}┼{'─'*14}┼{'─'*9}┼{'─'*22}┼{'─'*17}┼{'─'*5}")
for i, r in enumerate(results_sorted[:30]):
    print(f"  {i+1:3d} │{_thr(r):>5}│{r['family_group']:>12}│{r['gate']:>22}│{r['config']:>12}│{r['signal']:>14}│"
          f"{r['n_tickers']:>4}{r['n_trades']:>5}│"
          f"{r['avg_pnl']*100:>+7.2f}{r['wr']*100:>4.0f}%{r['total_pnl']*100:>+8.1f}│"
          f"{r['h2_avg']*100:>+7.2f}{r['h2_total']*100:>+8.1f}│"
          f"{r['oracle_capture']*100:>4.0f}%")

# H2 FOCUS
h2_pos = [r for r in all_results if r['h2_total'] > 0 and r['h2_n'] > 0]
h2_avg_sorted = sorted([r for r in h2_pos if r['h2_n'] >= 5],
                        key=lambda r: r['h2_avg'], reverse=True)

print(f"\n  ── TOP 25 STRATEGIES BY H2 AVG PnL (min 5 H2 trades) ──")
print(f"  {'#':>3} │{'Thr':>5}│{'FamGrp':>12}│{'Gate':>22}│{'Config':>12}│{'Signal':>14}│"
      f"{'Tks':>4}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'Full%':>8}")
print(f"  {'─'*3}─┼{'─'*5}┼{'─'*12}┼{'─'*22}┼{'─'*12}┼{'─'*14}┼{'─'*9}┼{'─'*17}┼{'─'*8}")
for i, r in enumerate(h2_avg_sorted[:25]):
    print(f"  {i+1:3d} │{_thr(r):>5}│{r['family_group']:>12}│{r['gate']:>22}│{r['config']:>12}│{r['signal']:>14}│"
          f"{r['n_tickers']:>4}{r['h2_n']:>5}│"
          f"{r['h2_avg']*100:>+7.2f}{r['h2_total']*100:>+8.1f}│"
          f"{r['total_pnl']*100:>+7.1f}")

# CONFIG METHOD HEAD-TO-HEAD
print(f"\n  ══ CONFIG METHOD HEAD-TO-HEAD (1.0x threshold, ALL families, ANY gate) ══")
print(f"\n  ── single_best signal ──")
print(f"  {'Method':>14} {'Tks':>4} {'Tr':>5} {'Avg%':>7} {'WR':>5} {'Total%':>9} "
      f"{'H2_N':>5} {'H2_Avg%':>8} {'H2_Tot%':>9} {'Cap%':>5}")
for cm in CONFIG_METHODS:
    matches = [r for r in all_results
               if r['config'] == cm and r['signal'] == 'single_best'
               and r['family_group'] == 'ALL' and r['gate'] == 'ANY'
               and r.get('thresh_mult', 1.0) == 1.0]
    if matches:
        r = matches[0]
        print(f"  {cm:>14} {r['n_tickers']:>4} {r['n_trades']:>5} "
              f"{r['avg_pnl']*100:>+6.2f} {r['wr']*100:>4.0f}% {r['total_pnl']*100:>+8.1f} "
              f"{r['h2_n']:>5} {r['h2_avg']*100:>+7.2f} {r['h2_total']*100:>+8.1f} "
              f"{r['oracle_capture']*100:>4.0f}%")

print(f"\n  ── vote_2 signal ──")
print(f"  {'Method':>14} {'Tks':>4} {'Tr':>5} {'Avg%':>7} {'WR':>5} {'Total%':>9} "
      f"{'H2_N':>5} {'H2_Avg%':>8} {'H2_Tot%':>9} {'Cap%':>5}")
for cm in CONFIG_METHODS:
    matches = [r for r in all_results
               if r['config'] == cm and r['signal'] == 'vote_2'
               and r['family_group'] == 'ALL' and r['gate'] == 'ANY'
               and r.get('thresh_mult', 1.0) == 1.0]
    if matches:
        r = matches[0]
        print(f"  {cm:>14} {r['n_tickers']:>4} {r['n_trades']:>5} "
              f"{r['avg_pnl']*100:>+6.2f} {r['wr']*100:>4.0f}% {r['total_pnl']*100:>+8.1f} "
              f"{r['h2_n']:>5} {r['h2_avg']*100:>+7.2f} {r['h2_total']*100:>+8.1f} "
              f"{r['oracle_capture']*100:>4.0f}%")

# Best strategy per config method
print(f"\n  ── BEST STRATEGY PER CONFIG METHOD (by H2 avg × H2 total, min 5 H2 trades) ──")
print(f"  {'Method':>14} │{'Thr':>5}│{'Gate':>18}│{'Signal':>14}│"
      f"{'Tks':>4}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'Full%':>8}")
for cm in CONFIG_METHODS:
    cm_results = [r for r in all_results if r['config'] == cm and r['h2_n'] >= 5 and r['h2_avg'] > 0]
    if cm_results:
        best_cm = max(cm_results, key=lambda r: r['h2_avg'] * r['h2_total'])
        print(f"  {cm:>14} │{_thr(best_cm):>5}│{best_cm['gate']:>18}│{best_cm['signal']:>14}│"
              f"{best_cm['n_tickers']:>4}{best_cm['h2_n']:>5}│"
              f"{best_cm['h2_avg']*100:>+7.2f}{best_cm['h2_total']*100:>+8.1f}│"
              f"{best_cm['total_pnl']*100:>+7.1f}")
    else:
        print(f"  {cm:>14} │ (no qualifying strategies)")

# PARETO: Optimal threshold per base strategy
print(f"\n  ══ PARETO: OPTIMAL THRESHOLD PER BASE STRATEGY ══")

strat_families = defaultdict(list)
for r in all_results:
    base_key = (r['family_group'], r['gate'], r['config'], r['signal'])
    strat_families[base_key].append(r)

pareto_rows = []
for base_key, variants in strat_families.items():
    if not variants: continue
    by_mult = {r.get('thresh_mult', 1.0): r for r in variants}
    base = by_mult.get(1.0)
    if base is None or base['n_trades'] < 5: continue
    h2_pos_variants = [r for r in variants if r['h2_n'] >= 3 and r['h2_avg'] > 0 and r['h2_total'] > 0]
    best_h2_combo_r = max(h2_pos_variants, key=lambda r: r['h2_avg'] * r['h2_total']) if h2_pos_variants else variants[0]
    pareto_rows.append({
        'base_key': base_key,
        'base_total': base['total_pnl'], 'base_avg': base['avg_pnl'],
        'base_h2_avg': base['h2_avg'], 'base_h2_total': base['h2_total'],
        'base_trades': base['n_trades'],
        'best_h2c_mult': best_h2_combo_r.get('thresh_mult', 1.0),
        'best_h2c_avg': best_h2_combo_r['h2_avg'],
        'best_h2c_total': best_h2_combo_r['h2_total'],
        'best_h2c_trades': best_h2_combo_r['h2_n'],
        'h2_combo_score': best_h2_combo_r['h2_avg'] * best_h2_combo_r['h2_total'] if h2_pos_variants else 0,
        'variants': by_mult,
    })

pareto_sorted = sorted(pareto_rows, key=lambda p: p['h2_combo_score'], reverse=True)

print(f"\n  ── TOP 20 BASE STRATEGIES: BEST H2 THRESHOLD ──")
print(f"  {'#':>3} │{'FamGrp':>12}│{'Gate':>18}│{'Config':>12}│{'Signal':>14}│"
      f"{'Base%':>8}│{'→Thr':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'Score':>7}")
print(f"  {'─'*3}─┼{'─'*12}┼{'─'*18}┼{'─'*12}┼{'─'*14}┼{'─'*8}┼{'─'*5}┼{'─'*17}┼{'─'*7}")
for i, p in enumerate(pareto_sorted[:20]):
    bk = p['base_key']
    print(f"  {i+1:3d} │{bk[0]:>12}│{bk[1]:>18}│{bk[2]:>12}│{bk[3]:>14}│"
          f"{p['base_total']*100:>+7.1f}│{p['best_h2c_mult']:>4.2f}x│"
          f"{p['best_h2c_avg']*100:>+7.2f}{p['best_h2c_total']*100:>+8.1f}│"
          f"{p['h2_combo_score']*10000:>6.1f}")

# Full Pareto curves for top 5
print(f"\n  ── FULL PARETO CURVES: TOP 5 BY H2 COMBO SCORE ──")
for i, p in enumerate(pareto_sorted[:5]):
    bk = p['base_key']
    print(f"\n  #{i+1} {bk[0]}|{bk[1]}|{bk[2]}|{bk[3]}:")
    print(f"  {'Mult':>6} {'Tks':>4} {'Trades':>7} {'Avg%':>7} {'WR':>5} {'Total%':>9} "
          f"{'H2_N':>5} {'H2_Avg%':>8} {'H2_Tot%':>9} {'Tr/Tk/Mo':>9}")
    for mult in THRESH_MULTS:
        r = p['variants'].get(mult)
        if r:
            n_months = max(1, len(set(str(t['date'])[:7] for t in [])))
            tr_mo = r['h2_n'] / max(r['n_tickers'], 1) / 2
            combo = r['h2_avg'] * r['h2_total'] * 10000 if r['h2_n'] >= 3 else 0
            marker = ' ◀' if mult == p['best_h2c_mult'] else ''
            print(f"  {mult:>5.2f}x {r['n_tickers']:>4} {r['n_trades']:>7} "
                  f"{r['avg_pnl']*100:>+6.2f} {r['wr']*100:>4.0f}% {r['total_pnl']*100:>+8.1f} "
                  f"{r['h2_n']:>5} {r['h2_avg']*100:>+7.2f} {r['h2_total']*100:>+8.1f} "
                  f"{tr_mo:>8.1f}{marker}")

# FAMILY AGREEMENT GATE SWEEP
print(f"\n  ══ FAMILY AGREEMENT GATE SWEEP ══")
print(f"  {'Fam_min':>7} {'FA':>5} {'Tks':>4} (at 1.0x/s3_score/single_best)")
for fam_min in [1, 2, 3, 4, 5]:
    for fa in [0.0, 0.5, 0.6]:
        gp = {'min_families': fam_min, 'va_cons_min': 0.0, 'seed_agree_min': 0.0, 'fam_agree_min': fa}
        gl = make_gate_label(gp)
        matches = [r for r in all_results
                   if r['gate'] == gl and r['config'] == 's3_score'
                   and r['signal'] == 'single_best' and r['family_group'] == 'ALL'
                   and r.get('thresh_mult', 1.0) == 1.0]
        if matches:
            r = matches[0]
            print(f"  F>={fam_min}    {fa:>4.0%} {r['n_tickers']:>4} tks, {r['n_trades']:>4} tr, "
                  f"avg={r['avg_pnl']*100:>+.2f}%, H2={r['h2_avg']*100:>+.2f}%, "
                  f"H2_tot={r['h2_total']*100:>+.1f}%")

# Per-ticker breakdown for key strategies
def _print_per_ticker_breakdown(label, strat_r, exp_configs, ticker_features,
                                 daily_probs, exp_fire_dates_by_mult,
                                 date_trade_pnl, cross_best_exps, oracle_pos_set, h2_start_dt,
                                 h1_pnl=None, exp_configs_worst=None,
                                 date_trade_pnl_fixed=None):
    tm = strat_r.get('thresh_mult', 1.0)
    print(f"\n  ── {label} PER-TICKER BREAKDOWN ──")
    print(f"  Strategy: {_thr(strat_r)}x|{strat_r['family_group']}|{strat_r['gate']}|"
          f"{strat_r['config']}|{strat_r['signal']}")
    fire_dates = exp_fire_dates_by_mult[tm]
    args = dict(ticker_features=ticker_features, exp_configs=exp_configs,
                daily_probs=daily_probs, exp_fire_dates=fire_dates,
                date_trade_pnl=date_trade_pnl, cross_best_exps=cross_best_exps,
                thresh_mult=tm, h1_pnl=h1_pnl, exp_configs_worst=exp_configs_worst,
                date_trade_pnl_fixed=date_trade_pnl_fixed)
    trades = collect_trades(strat_r.get('_gate_params'), strat_r['config'],
                            strat_r['signal'], strat_r['family_group'], **args)
    if not trades:
        print("  (no trades)")
        return
    tdf = pd.DataFrame(trades)
    tdf['is_h2'] = tdf['date'].apply(lambda d: d >= h2_start_dt)
    tk_exp = tdf.groupby('ticker')['exp'].first()
    tk_stats = tdf.groupby('ticker').agg(
        n=('pnl', 'count'), total=('pnl', 'sum'),
        avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean()),
        h2_n=('is_h2', 'sum'),
        h2_total=('pnl', lambda x: x[tdf.loc[x.index, 'is_h2']].sum()),
    ).sort_values('total', ascending=False)
    print(f"  {'Ticker':>8} {'Exp':>16} {'TP%':>5} {'SL%':>5} "
          f"{'Tr':>4} {'Avg%':>7} {'WR':>5} {'Tot%':>8} "
          f"{'H2_Tr':>5} {'H2_Tot%':>8} {'Oracle?':>8}")
    for t, row in tk_stats.iterrows():
        is_oracle = '★' if t in oracle_pos_set else ''
        e_used = tk_exp.get(t, '?')
        cfg = exp_configs.get((t, e_used), {})
        tp = cfg.get('tp_pct', 0) * 100
        sl = cfg.get('sl_pct', 0) * 100
        print(f"  {t:>8} {str(e_used)[:16]:>16} {tp:>4.1f}% {sl:>4.1f}% "
              f"{int(row['n']):>4} {row['avg']*100:>+6.2f} {row['wr']*100:>4.0f}% "
              f"{row['total']*100:>+7.1f} {int(row['h2_n']):>5} "
              f"{row['h2_total']*100:>+7.1f} {is_oracle:>8}")

_bd_args = dict(exp_configs=exp_configs, ticker_features=ticker_features,
                daily_probs=daily_probs, exp_fire_dates_by_mult=exp_fire_dates_by_mult,
                date_trade_pnl=date_trade_pnl, cross_best_exps=cross_best_exps,
                oracle_pos_set=oracle_pos_set, h2_start_dt=h2_start_dt,
                h1_pnl=h1_pnl, exp_configs_worst=exp_configs_worst,
                date_trade_pnl_fixed=date_trade_pnl_fixed)

if results_sorted:
    _print_per_ticker_breakdown("BEST OVERALL (by total PnL)", results_sorted[0], **_bd_args)
if h2_avg_sorted:
    _print_per_ticker_breakdown("BEST H2 AVG", h2_avg_sorted[0], **_bd_args)

# S3_SCORE strategy breakdown
s3_matches = [r for r in all_results
              if r['config'] == 's3_score' and r['signal'] == 'single_best'
              and r['family_group'] == 'ALL' and r['gate'] == 'ANY'
              and r.get('thresh_mult', 1.0) == 1.0]
if s3_matches:
    _print_per_ticker_breakdown("S3_SCORE STRATEGY (production equivalent)", s3_matches[0], **_bd_args)

# Category comparison
print(f"\n  ── CATEGORY COMPARISON ──")
categories = {
    'Best overall':    results_sorted[0] if results_sorted else None,
    'Best H2 avg':     max([r for r in h2_pos if r['h2_n'] >= 5], key=lambda r: r['h2_avg']) if [r for r in h2_pos if r['h2_n'] >= 5] else None,
    'Best H2 combo':   h2_ranked[0] if h2_ranked else None,
    'Best Sharpe':     max(all_results, key=lambda r: r['sharpe']) if all_results else None,
    'S3 equivalent':   s3_matches[0] if s3_matches else None,
}
print(f"  {'Category':>18} │ {'Strategy':<60} │{'Total%':>8} │{'H2_Tot%':>9} │{'H2_Avg%':>8}")
print(f"  {'─'*18}─┼{'─'*61}┼{'─'*9}┼{'─'*10}┼{'─'*8}")
for cat, r in categories.items():
    if r is None: continue
    lbl = f"{_thr(r)}x|{r['family_group']}|{r['gate']}|{r['config']}|{r['signal']}"
    print(f"  {cat:>18} │ {lbl:<60} │{r['total_pnl']*100:>+7.1f} │{r['h2_total']*100:>+8.1f} │{r['h2_avg']*100:>+7.2f}")

# DIVERSIFIED DEPLOYMENT
print(f"\n  ══ DIVERSIFIED DEPLOYMENT: ≥3 tickers, max conc <60%, H2 avg>0% ══")
diversified = sorted(
    [r for r in all_results
     if r['n_tickers'] >= 3
     and r.get('max_concentration', 1.0) < 0.60
     and r['h2_avg'] > 0
     and r['h2_n'] >= 5],
    key=lambda r: r['h2_avg'] * r['h2_total'], reverse=True)
print(f"  {len(diversified)} strategies found")
if diversified:
    print(f"  {'#':>3} │{'Thr':>5}│{'FamGrp':>12}│{'Gate':>22}│{'Config':>12}│{'Signal':>14}│"
          f"{'Tks':>4}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'Full%':>8}│{'MaxConc':>8}")
    print(f"  {'─'*3}─┼{'─'*5}┼{'─'*12}┼{'─'*22}┼{'─'*12}┼{'─'*14}┼{'─'*9}┼{'─'*17}┼{'─'*8}┼{'─'*8}")
    for i, r in enumerate(diversified[:15]):
        print(f"  {i+1:3d} │{_thr(r):>5}│{r['family_group']:>12}│{r['gate']:>22}│{r['config']:>12}│{r['signal']:>14}│"
              f"{r['n_tickers']:>4}{r['h2_n']:>5}│"
              f"{r['h2_avg']*100:>+7.2f}{r['h2_total']*100:>+8.1f}│"
              f"{r['total_pnl']*100:>+7.1f}│{r['max_concentration']*100:>6.0f}%")
    _print_per_ticker_breakdown("BEST DIVERSIFIED", diversified[0], **_bd_args)


# ==============================================================================
# PHASE 6: SAVE CSVs + CHARTS
# ==============================================================================
print(f"\n  PHASE 6: Saving results...")

csv_rows = []
for r in all_results:
    row = {k: v for k, v in r.items() if k not in ('tickers', 'monthly', '_gate_params')}
    monthly = r.get('monthly', {})
    for m in ['2026-04', '2026-05', '2026-06', '2026-07', '2026-08']:
        row[f'pnl_{m}'] = monthly.get(m, 0.0)
    csv_rows.append(row)

full_df = pd.DataFrame(csv_rows).sort_values('total_pnl', ascending=False)
full_df.to_csv(OUTPUT_ALL, index=False)
print(f"    All results: {OUTPUT_ALL} ({len(full_df)} strategies)")

top_df = full_df.head(50)
top_df.to_csv(OUTPUT_TOP, index=False)
print(f"    Top 50:      {OUTPUT_TOP}")

pareto_csv_rows = []
for p in pareto_sorted:
    bk = p['base_key']
    pareto_csv_rows.append({
        'family_group': bk[0], 'gate': bk[1], 'config': bk[2], 'signal': bk[3],
        'base_total_pnl': p['base_total'], 'base_avg_pnl': p['base_avg'],
        'base_h2_avg': p['base_h2_avg'], 'base_h2_total': p['base_h2_total'],
        'base_trades': p['base_trades'],
        'best_h2_mult': p['best_h2c_mult'],
        'best_h2_avg': p['best_h2c_avg'], 'best_h2_total': p['best_h2c_total'],
        'best_h2_trades': p['best_h2c_trades'],
        'h2_combo_score': p['h2_combo_score'],
    })
pd.DataFrame(pareto_csv_rows).to_csv(OUTPUT_PARETO, index=False)
print(f"    Pareto:      {OUTPUT_PARETO} ({len(pareto_csv_rows)} base strategies)")

# Chart: top strategies vs S3 vs oracle
print(f"\n    Generating chart...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('LONG Oracle vs Strategies — Strategy Sweep Results', fontsize=14, fontweight='bold')

# [0,0] Config method comparison
ax = axes[0, 0]
cm_data = []
for cm in CONFIG_METHODS:
    matches = [r for r in all_results
               if r['config'] == cm and r['signal'] == 'single_best'
               and r['family_group'] == 'ALL' and r['gate'] == 'ANY'
               and r.get('thresh_mult', 1.0) == 1.0]
    if matches:
        r = matches[0]
        cm_data.append((cm, r['avg_pnl'] * 100, r['h2_avg'] * 100, r['n_tickers']))
if cm_data:
    cm_names = [d[0] for d in cm_data]
    cm_full = [d[1] for d in cm_data]
    cm_h2 = [d[2] for d in cm_data]
    x = np.arange(len(cm_names))
    w = 0.35
    ax.bar(x - w/2, cm_full, w, label='Full avg PnL', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, cm_h2, w, label='H2 avg PnL', color='coral', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cm_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Avg PnL (%)')
    ax.set_title('Config Method Comparison (1.0x, ALL, ANY, single_best)')
    ax.legend(fontsize=8)

# [0,1] Total PnL distribution
ax = axes[0, 1]
totals = [r['total_pnl'] * 100 for r in all_results]
ax.hist(totals, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(0, color='red', linewidth=1, linestyle='--')
ax.axvline(np.median(totals), color='orange', linewidth=1, label=f'Median: {np.median(totals):+.1f}%')
ax.set_xlabel('Total PnL (%)')
ax.set_ylabel('Strategy Count')
ax.set_title(f'Total PnL Distribution ({len(all_results)} strategies)')
ax.legend(fontsize=8)

# [1,0] H2 avg vs H2 total scatter
ax = axes[1, 0]
h2_avgs = [r['h2_avg'] * 100 for r in all_results if r['h2_n'] > 0]
h2_tots = [r['h2_total'] * 100 for r in all_results if r['h2_n'] > 0]
ax.scatter(h2_tots, h2_avgs, alpha=0.15, s=8, c='steelblue')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('H2 Total PnL (%)')
ax.set_ylabel('H2 Avg PnL (%)')
ax.set_title('H2 Avg vs H2 Total (all strategies)')
if s3_matches:
    sr = s3_matches[0]
    ax.scatter([sr['h2_total']*100], [sr['h2_avg']*100], c='red', s=80, marker='*',
               zorder=10, label='S3 equiv')
    ax.legend(fontsize=8)

# [1,1] Family group comparison
ax = axes[1, 1]
fg_data = []
for fg in FAMILY_GROUPS:
    matches = [r for r in all_results
               if r['family_group'] == fg and r['config'] == 's3_score'
               and r['signal'] == 'single_best' and r['gate'] == 'ANY'
               and r.get('thresh_mult', 1.0) == 1.0]
    if matches:
        r = matches[0]
        fg_data.append((fg, r['total_pnl'] * 100, r['h2_total'] * 100))
if fg_data:
    fg_names = [d[0] for d in fg_data]
    fg_full = [d[1] for d in fg_data]
    fg_h2 = [d[2] for d in fg_data]
    x = np.arange(len(fg_names))
    w = 0.35
    ax.bar(x - w/2, fg_full, w, label='Full total', color='#2E7D32', alpha=0.8)
    ax.bar(x + w/2, fg_h2, w, label='H2 total', color='#C62828', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(fg_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Total PnL (%)')
    ax.set_title('Family Group Comparison (s3_score, single_best)')
    ax.legend(fontsize=8)

plt.tight_layout()
chart_path = '/tmp/oracle_long_vs_strategies.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"    Chart: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  FINAL SUMMARY")
print(f"{'='*W}")
print(f"  Strategies evaluated: {n_done}")
print(f"  Strategies with trades: {len(all_results)}")
print(f"  Oracle ceiling (limit_entry): {oracle_total_pnl*100:+.1f}% total ({len(oracle_pos_set)} positive tickers)")
print(f"  S3 equivalent: {s3_metrics['total_pnl']*100:+.1f}% total ({s3_metrics['n_tickers']} tickers, {s3_metrics['n_trades']} trades)")
print(f"")
if results_sorted:
    r = results_sorted[0]
    print(f"  BEST TOTAL: {_thr(r)}x|{r['family_group']}|{r['gate']}|{r['config']}|{r['signal']}")
    print(f"    {r['n_tickers']} tickers, {r['n_trades']} trades, "
          f"avg={r['avg_pnl']*100:+.2f}%, total={r['total_pnl']*100:+.1f}%, "
          f"H2={r['h2_total']*100:+.1f}%, capture={r['oracle_capture']*100:.0f}%")
if h2_avg_sorted:
    r = h2_avg_sorted[0]
    print(f"  BEST H2 AVG: {_thr(r)}x|{r['family_group']}|{r['gate']}|{r['config']}|{r['signal']}")
    print(f"    {r['n_tickers']} tickers, {r['h2_n']} H2 trades, "
          f"H2_avg={r['h2_avg']*100:+.2f}%, H2_total={r['h2_total']*100:+.1f}%")
if s3_matches:
    r = s3_matches[0]
    print(f"  S3 EQUIVALENT: {r['n_tickers']} tickers, {r['n_trades']} trades, "
          f"avg={r['avg_pnl']*100:+.2f}%, total={r['total_pnl']*100:+.1f}%, "
          f"H2={r['h2_total']*100:+.1f}%, capture={r['oracle_capture']*100:.0f}%")
print(f"")
print(f"  Files saved:")
print(f"    {OUTPUT_ALL} ({len(full_df)} strategies)")
print(f"    {OUTPUT_TOP}")
print(f"    {OUTPUT_PARETO} ({len(pareto_csv_rows)} base strategies)")
print(f"    {chart_path}")
print(f"{'='*W}")
