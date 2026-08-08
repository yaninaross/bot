# %% ── Oracle vs Strategies — H2-TARGETED LONG Model Evaluation ──────────────
#
# Same sweep as oracle_long_vs_strategies.py but oracle ceiling and config
# selection are optimized for H2 (Jun-Aug) instead of full period.
#
# Key question: what's the MAXIMUM achievable PnL in H2, and can any
# non-oracle strategy capture it?  If even the H2 oracle is negative,
# the models don't work post-collapse.
#
# Adds config methods: best_h2 (total), best_h2_avg (per-trade)
#
# Outputs:
#   /tmp/oracle_long_h2_results.csv   — all strategy results
#   /tmp/oracle_long_h2_top.csv       — top 50 strategies
#   /tmp/oracle_long_h2_pareto.csv    — pareto by H2 combo score
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
                  'worst_opt', 'h1_oracle', 'fixed_tpsl', 's3_score',
                  'best_h2', 'best_h2_avg']
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

OUTPUT_ALL     = '/tmp/oracle_long_h2_results.csv'
OUTPUT_TOP     = '/tmp/oracle_long_h2_top.csv'
OUTPUT_PARETO  = '/tmp/oracle_long_h2_pareto.csv'


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
                      h1_pnl=None, h2_pnl=None, h2_pnl_avg=None,
                      exp_configs_worst=None):
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
    elif method == 'best_h2':
        if h2_pnl is None:
            return None
        best, best_v = None, -999
        for e in available:
            v = h2_pnl.get((ticker, e), 0.0)
            if v > best_v:
                best_v = v
                best = e
        return best
    elif method == 'best_h2_avg':
        if h2_pnl_avg is None:
            return None
        best, best_v = None, -999
        for e in available:
            v = h2_pnl_avg.get((ticker, e), 0.0)
            if v > best_v:
                best_v = v
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


def evaluate_trades(trades_list, oracle_h2_pos_set, oracle_h2_total, h2_start_dt):
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
    h2_wr = float((h2_pnls > 0).mean()) if h2_n > 0 else 0.0
    h2_tickers = set(t['ticker'] for t in sorted_trades if t['date'] >= h2_start_dt)
    oracle_overlap = len(h2_tickers & oracle_h2_pos_set) / max(len(oracle_h2_pos_set), 1)
    oracle_h2_capture = h2_total / oracle_h2_total if oracle_h2_total > 0 else 0.0
    monthly = {}
    for t in sorted_trades:
        m = str(t['date'])[:7]
        monthly[m] = monthly.get(m, 0.0) + t['pnl']
    ticker_h2_pnls = defaultdict(float)
    for t in sorted_trades:
        if t['date'] >= h2_start_dt:
            ticker_h2_pnls[t['ticker']] += t['pnl']
    max_h2_ticker = max(ticker_h2_pnls.values()) if ticker_h2_pnls else 0.0
    max_h2_conc = max_h2_ticker / h2_total if h2_total > 0 else 1.0
    return {
        'n_tickers': len(tickers_set), 'n_trades': n,
        'avg_pnl': avg, 'wr': wr, 'total_pnl': total,
        'sharpe': sharpe, 'max_dd': max_dd,
        'h2_n': h2_n, 'h2_total': h2_total, 'h2_avg': h2_avg, 'h2_wr': h2_wr,
        'h2_tickers': len(h2_tickers),
        'oracle_overlap': oracle_overlap, 'oracle_h2_capture': oracle_h2_capture,
        'max_h2_concentration': max_h2_conc,
        'monthly': monthly, 'tickers': tickers_set,
    }


def collect_trades(gate_params, config_method, signal_method, family_group_name,
                   ticker_features, exp_configs, daily_probs, exp_fire_dates,
                   date_trade_pnl, cross_best_exps, thresh_mult=1.0,
                   h1_pnl=None, h2_pnl=None, h2_pnl_avg=None,
                   exp_configs_worst=None,
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
                                    h1_pnl=h1_pnl, h2_pnl=h2_pnl,
                                    h2_pnl_avg=h2_pnl_avg,
                                    exp_configs_worst=exp_configs_worst)
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
print(f"  H2-TARGETED ORACLE vs STRATEGIES — LONG Model")
print(f"  TEST: {TEST_START} → {TEST_END}   H2: {HALF2_START} → {TEST_END}")
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

# H1 and H2 PnL per (ticker, base_exp)
h1_pnl = {}
h2_pnl = {}
h2_pnl_avg = {}
h2_trade_counts = {}
for (ticker, be), pnl_map in date_trade_pnl.items():
    fire = exp_fire_dates_by_mult[1.0].get((ticker, be), set())
    h1_trades = [pnl_map[d] for d in fire if d in pnl_map and d < h2_start_dt]
    h2_trades = [pnl_map[d] for d in fire if d in pnl_map and d >= h2_start_dt]
    h1_pnl[(ticker, be)] = sum(h1_trades)
    h2_pnl[(ticker, be)] = sum(h2_trades)
    h2_pnl_avg[(ticker, be)] = np.mean(h2_trades) if h2_trades else 0.0
    h2_trade_counts[(ticker, be)] = len(h2_trades)

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

# OPT → H2 correlation diagnostic
print(f"\n    ── OPT → H2 CORRELATION DIAGNOSTIC ──")
from scipy import stats as scipy_stats
opt_vals_all, h2_vals_all = [], []
for (ticker, be), h2v in h2_pnl.items():
    cfg = exp_configs.get((ticker, be))
    if cfg is None: continue
    if h2_trade_counts.get((ticker, be), 0) == 0: continue
    opt_vals_all.append(cfg['opt_avg_pnl'])
    h2_vals_all.append(h2v)
if len(opt_vals_all) > 10:
    opt_a = np.array(opt_vals_all)
    h2_a  = np.array(h2_vals_all)
    pearson_r = np.corrcoef(opt_a, h2_a)[0, 1]
    spearman_r = scipy_stats.spearmanr(opt_a, h2_a).statistic
    print(f"    OPT→H2 Pearson r  = {pearson_r:+.3f}  (n={len(opt_a)} ticker-exp pairs)")
    print(f"    OPT→H2 Spearman ρ = {spearman_r:+.3f}")
    if pearson_r < -0.05:
        print(f"    ⚠ ANTI-CORRELATION: higher OPT → worse H2.")
    elif pearson_r < 0.1:
        print(f"    ⚠ NEAR ZERO: OPT has no predictive power for H2.")
    else:
        print(f"    ✓ Positive: OPT-based selection partially works for H2.")

    h1_vals_corr, h2_vals_corr = [], []
    for (ticker, be) in h2_pnl.keys():
        h1v = h1_pnl.get((ticker, be), 0)
        h2v = h2_pnl.get((ticker, be), 0)
        if h2_trade_counts.get((ticker, be), 0) > 0:
            h1_vals_corr.append(h1v)
            h2_vals_corr.append(h2v)
    if len(h1_vals_corr) > 10:
        h1_a = np.array(h1_vals_corr)
        h2_a = np.array(h2_vals_corr)
        h1h2_r = np.corrcoef(h1_a, h2_a)[0, 1]
        h1h2_rho = scipy_stats.spearmanr(h1_a, h2_a).statistic
        print(f"    H1→H2  Pearson r  = {h1h2_r:+.3f}  (n={len(h1_a)} ticker-exp pairs)")
        print(f"    H1→H2  Spearman ρ = {h1h2_rho:+.3f}")

    va_vals, h2_vals_va = [], []
    for (ticker, be), h2v in h2_pnl.items():
        cfg = exp_configs.get((ticker, be))
        if cfg is None: continue
        if h2_trade_counts.get((ticker, be), 0) == 0: continue
        va_vals.append(cfg['mean_va_pnl'])
        h2_vals_va.append(h2v)
    if len(va_vals) > 10:
        va_a = np.array(va_vals)
        h2_a = np.array(h2_vals_va)
        va_h2_r = np.corrcoef(va_a, h2_a)[0, 1]
        va_h2_rho = scipy_stats.spearmanr(va_a, h2_a).statistic
        print(f"    VA→H2  Pearson r  = {va_h2_r:+.3f}  (n={len(va_a)} ticker-exp pairs)")
        print(f"    VA→H2  Spearman ρ = {va_h2_rho:+.3f}")

_release_memory()


# ==============================================================================
# PHASE 3: COMPUTE H2 ORACLE BASELINE (self-contained)
# ==============================================================================
print(f"\n  PHASE 3: Computing H2 ORACLE ceiling...")

oracle_h2_per_ticker = {}
for ticker in ticker_features:
    best_h2_total, best_exp = -999, None
    for be in ticker_features[ticker]['available_exps']:
        h2v = h2_pnl.get((ticker, be), 0)
        if h2v > best_h2_total:
            best_h2_total, best_exp = h2v, be
    h2_n = h2_trade_counts.get((ticker, best_exp), 0) if best_exp else 0
    h2_avg_v = h2_pnl_avg.get((ticker, best_exp), 0) if best_exp else 0
    full_total = 0
    if best_exp:
        fire = exp_fire_dates_by_mult[1.0].get((ticker, best_exp), set())
        pnl_map = date_trade_pnl.get((ticker, best_exp), {})
        full_total = sum(pnl_map.get(d, 0) for d in fire)
    oracle_h2_per_ticker[ticker] = {
        'h2_total': best_h2_total, 'h2_exp': best_exp,
        'h2_n': h2_n, 'h2_avg': h2_avg_v,
        'full_total': full_total,
    }

oracle_h2_pos_set = set(t for t, v in oracle_h2_per_ticker.items() if v['h2_total'] > 0)
oracle_h2_total = sum(v['h2_total'] for v in oracle_h2_per_ticker.values() if v['h2_total'] > 0)
oracle_h2_neg_set = set(t for t, v in oracle_h2_per_ticker.items() if v['h2_total'] <= 0)

print(f"    H2 Oracle: {len(oracle_h2_pos_set)} positive tickers, "
      f"{oracle_h2_total*100:+.1f}% total H2 PnL ceiling")
print(f"    H2 Oracle NEGATIVE: {len(oracle_h2_neg_set)} tickers (even best exp loses in H2)")

# Full oracle for comparison
oracle_full_total = sum(max(v['full_total'] for v in [oracle_h2_per_ticker[t]])
                        for t in oracle_h2_pos_set)

# Per-ticker H2 oracle breakdown
h2o_sorted = sorted(oracle_h2_per_ticker.items(),
                     key=lambda x: x[1]['h2_total'], reverse=True)
print(f"\n  ── H2 ORACLE PER-TICKER ──")
print(f"  {'Ticker':>8} {'BestExp':>20} {'H2_N':>5} {'H2_Avg%':>8} {'H2_Tot%':>9} "
      f"{'Full%':>8} {'H2 Win?':>8}")
for t, v in h2o_sorted:
    marker = '✓' if v['h2_total'] > 0 else '✗'
    print(f"  {t:>8} {str(v['h2_exp'])[:20]:>20} {v['h2_n']:>5} "
          f"{v['h2_avg']*100:>+7.2f} {v['h2_total']*100:>+8.1f} "
          f"{v['full_total']*100:>+7.1f} {marker:>8}")

# S3-equivalent baseline for comparison
s3_gate = {'min_families': 2, 'va_cons_min': 0.0, 'seed_agree_min': 0.0, 'fam_agree_min': 0.5}
s3_trades = collect_trades(s3_gate, 's3_score', 'single_best', 'ALL',
                           ticker_features=ticker_features, exp_configs=exp_configs,
                           daily_probs=daily_probs,
                           exp_fire_dates=exp_fire_dates_by_mult[1.0],
                           date_trade_pnl=date_trade_pnl,
                           cross_best_exps=cross_best_exps, thresh_mult=1.0,
                           h1_pnl=h1_pnl, h2_pnl=h2_pnl, h2_pnl_avg=h2_pnl_avg,
                           exp_configs_worst=exp_configs_worst,
                           date_trade_pnl_fixed=date_trade_pnl_fixed)
s3_metrics = evaluate_trades(s3_trades, oracle_h2_pos_set, oracle_h2_total, h2_start_dt)
if s3_metrics:
    print(f"\n    S3 equivalent: {s3_metrics['n_tickers']} tickers, {s3_metrics['n_trades']} trades, "
          f"H2_avg={s3_metrics['h2_avg']*100:+.2f}%, H2_total={s3_metrics['h2_total']*100:+.1f}%, "
          f"H2_capture={s3_metrics['oracle_h2_capture']*100:.0f}%")
else:
    print(f"\n    S3 equivalent: no trades")
    s3_metrics = {'total_pnl': 0, 'avg_pnl': 0, 'h2_total': 0, 'h2_avg': 0,
                  'n_tickers': 0, 'n_trades': 0, 'h2_n': 0, 'oracle_h2_capture': 0}


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
                       thresh_mult=mult, h1_pnl=h1_pnl, h2_pnl=h2_pnl,
                       h2_pnl_avg=h2_pnl_avg,
                       exp_configs_worst=exp_configs_worst,
                       date_trade_pnl_fixed=date_trade_pnl_fixed)

    print(f"\n    ── Threshold mult {mult_label} ──")
    print(f"    Running gate-based strategies...")
    for gi, gp in enumerate(unique_gates):
        g_label = make_gate_label(gp)
        for cm in CONFIG_METHODS:
            for sm in SIGNAL_METHODS:
                trades = collect_trades(gp, cm, sm, 'ALL', **common_args)
                metrics = evaluate_trades(trades, oracle_h2_pos_set, oracle_h2_total, h2_start_dt)
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
                metrics = evaluate_trades(trades, oracle_h2_pos_set, oracle_h2_total, h2_start_dt)
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
# PHASE 5: OUTPUT — H2 FOCUSED
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 5: H2-FOCUSED RESULTS")
print(f"{'='*W}")

def _thr(r):
    return f"{r.get('thresh_mult', 1.0):.2f}"

# Primary sort: H2 total PnL
h2_sorted = sorted(all_results, key=lambda r: r['h2_total'], reverse=True)

# H2 combo
for r in all_results:
    r['h2_combo'] = r['h2_avg'] * r['h2_total'] if (r['h2_avg'] > 0 and r['h2_total'] > 0) else 0.0
h2_combo_ranked = sorted(all_results, key=lambda r: r['h2_combo'], reverse=True)

print(f"\n  ── TOP 30 STRATEGIES BY H2 TOTAL PnL ──")
print(f"  {'#':>3} │{'Thr':>5}│{'FamGrp':>12}│{'Gate':>22}│{'Config':>12}│{'Signal':>14}│"
      f"{'H2_Tks':>6}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_WR':>5}{'H2_Tot%':>9}│{'Full%':>8}│{'H2Cap%':>6}")
print(f"  {'─'*3}─┼{'─'*5}┼{'─'*12}┼{'─'*22}┼{'─'*12}┼{'─'*14}┼{'─'*11}┼{'─'*22}┼{'─'*8}┼{'─'*6}")
for i, r in enumerate(h2_sorted[:30]):
    print(f"  {i+1:3d} │{_thr(r):>5}│{r['family_group']:>12}│{r['gate']:>22}│{r['config']:>12}│{r['signal']:>14}│"
          f"{r.get('h2_tickers',0):>6}{r['h2_n']:>5}│"
          f"{r['h2_avg']*100:>+7.2f}{r.get('h2_wr',0)*100:>4.0f}%{r['h2_total']*100:>+8.1f}│"
          f"{r['total_pnl']*100:>+7.1f}│{r.get('oracle_h2_capture',0)*100:>5.0f}%")

# H2 AVG focus (min 5 trades)
h2_pos = [r for r in all_results if r['h2_total'] > 0 and r['h2_n'] > 0]
h2_avg_sorted = sorted([r for r in h2_pos if r['h2_n'] >= 5],
                        key=lambda r: r['h2_avg'], reverse=True)

print(f"\n  ── TOP 25 STRATEGIES BY H2 AVG PnL (min 5 H2 trades) ──")
print(f"  {'#':>3} │{'Thr':>5}│{'FamGrp':>12}│{'Gate':>22}│{'Config':>12}│{'Signal':>14}│"
      f"{'Tks':>4}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_WR':>5}{'H2_Tot%':>9}│{'H2Cap%':>6}")
print(f"  {'─'*3}─┼{'─'*5}┼{'─'*12}┼{'─'*22}┼{'─'*12}┼{'─'*14}┼{'─'*9}┼{'─'*22}┼{'─'*6}")
for i, r in enumerate(h2_avg_sorted[:25]):
    print(f"  {i+1:3d} │{_thr(r):>5}│{r['family_group']:>12}│{r['gate']:>22}│{r['config']:>12}│{r['signal']:>14}│"
          f"{r['n_tickers']:>4}{r['h2_n']:>5}│"
          f"{r['h2_avg']*100:>+7.2f}{r.get('h2_wr',0)*100:>4.0f}%{r['h2_total']*100:>+8.1f}│"
          f"{r.get('oracle_h2_capture',0)*100:>5.0f}%")

# CONFIG METHOD HEAD-TO-HEAD — H2 focused
print(f"\n  ══ CONFIG METHOD HEAD-TO-HEAD — H2 PERFORMANCE ══")
print(f"  (1.0x threshold, ALL families, ANY gate, single_best signal)")
print(f"\n  {'Method':>14} {'Tks':>4} {'H2_N':>5} {'H2_Avg%':>8} {'H2_WR':>5} "
      f"{'H2_Tot%':>9} {'Full%':>8} {'H2Cap%':>6}")
for cm in CONFIG_METHODS:
    matches = [r for r in all_results
               if r['config'] == cm and r['signal'] == 'single_best'
               and r['family_group'] == 'ALL' and r['gate'] == 'ANY'
               and r.get('thresh_mult', 1.0) == 1.0]
    if matches:
        r = matches[0]
        print(f"  {cm:>14} {r['n_tickers']:>4} {r['h2_n']:>5} "
              f"{r['h2_avg']*100:>+7.2f} {r.get('h2_wr',0)*100:>4.0f}% "
              f"{r['h2_total']*100:>+8.1f} {r['total_pnl']*100:>+7.1f} "
              f"{r.get('oracle_h2_capture',0)*100:>5.0f}%")

print(f"\n  ── vote_2 signal ──")
print(f"  {'Method':>14} {'Tks':>4} {'H2_N':>5} {'H2_Avg%':>8} {'H2_WR':>5} "
      f"{'H2_Tot%':>9} {'Full%':>8} {'H2Cap%':>6}")
for cm in CONFIG_METHODS:
    matches = [r for r in all_results
               if r['config'] == cm and r['signal'] == 'vote_2'
               and r['family_group'] == 'ALL' and r['gate'] == 'ANY'
               and r.get('thresh_mult', 1.0) == 1.0]
    if matches:
        r = matches[0]
        print(f"  {cm:>14} {r['n_tickers']:>4} {r['h2_n']:>5} "
              f"{r['h2_avg']*100:>+7.2f} {r.get('h2_wr',0)*100:>4.0f}% "
              f"{r['h2_total']*100:>+8.1f} {r['total_pnl']*100:>+7.1f} "
              f"{r.get('oracle_h2_capture',0)*100:>5.0f}%")

# best_h2 vs best_opt vs s3_score — THE KEY COMPARISON
print(f"\n  ══ KEY COMPARISON: H2 ORACLE vs PRODUCTION (1.0x/ALL/ANY/single_best) ══")
key_methods = ['best_h2', 'best_h2_avg', 'best_opt', 'worst_opt', 'h1_oracle', 's3_score', 'conservative']
print(f"  {'Method':>14} {'Tks':>4} {'H2_N':>5} {'H2_Avg%':>8} {'H2_Tot%':>9} "
      f"{'Full%':>8} {'H2Cap%':>6} {'Note'}")
for cm in key_methods:
    matches = [r for r in all_results
               if r['config'] == cm and r['signal'] == 'single_best'
               and r['family_group'] == 'ALL' and r['gate'] == 'ANY'
               and r.get('thresh_mult', 1.0) == 1.0]
    if matches:
        r = matches[0]
        note = ''
        if cm == 'best_h2':  note = '← H2 ORACLE (cheating)'
        if cm == 's3_score': note = '← PRODUCTION'
        if cm == 'best_opt': note = '← standard oracle'
        print(f"  {cm:>14} {r['n_tickers']:>4} {r['h2_n']:>5} "
              f"{r['h2_avg']*100:>+7.2f} {r['h2_total']*100:>+8.1f} "
              f"{r['total_pnl']*100:>+7.1f} {r.get('oracle_h2_capture',0)*100:>5.0f}% {note}")

# Best strategy per config method
print(f"\n  ── BEST STRATEGY PER CONFIG METHOD (by H2 total, min 5 H2 trades) ──")
print(f"  {'Method':>14} │{'Thr':>5}│{'Gate':>18}│{'Signal':>14}│"
      f"{'H2_N':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'H2Cap%':>6}")
for cm in CONFIG_METHODS:
    cm_results = [r for r in all_results if r['config'] == cm and r['h2_n'] >= 5]
    if cm_results:
        best_cm = max(cm_results, key=lambda r: r['h2_total'])
        print(f"  {cm:>14} │{_thr(best_cm):>5}│{best_cm['gate']:>18}│{best_cm['signal']:>14}│"
              f"{best_cm['h2_n']:>5}│"
              f"{best_cm['h2_avg']*100:>+7.2f}{best_cm['h2_total']*100:>+8.1f}│"
              f"{best_cm.get('oracle_h2_capture',0)*100:>5.0f}%")
    else:
        print(f"  {cm:>14} │ (no qualifying strategies)")

# PARETO: Optimal threshold per base strategy
print(f"\n  ══ PARETO: OPTIMAL THRESHOLD PER BASE STRATEGY (by H2 total) ══")

strat_families = defaultdict(list)
for r in all_results:
    base_key = (r['family_group'], r['gate'], r['config'], r['signal'])
    strat_families[base_key].append(r)

pareto_rows = []
for base_key, variants in strat_families.items():
    if not variants: continue
    by_mult = {r.get('thresh_mult', 1.0): r for r in variants}
    base = by_mult.get(1.0)
    if base is None or base['h2_n'] < 3: continue
    best_h2_r = max(variants, key=lambda r: r['h2_total'] if r['h2_n'] >= 3 else -999)
    pareto_rows.append({
        'base_key': base_key,
        'base_h2_total': base['h2_total'], 'base_h2_avg': base['h2_avg'],
        'base_h2_n': base['h2_n'], 'base_total': base['total_pnl'],
        'best_h2_mult': best_h2_r.get('thresh_mult', 1.0),
        'best_h2_total': best_h2_r['h2_total'],
        'best_h2_avg': best_h2_r['h2_avg'],
        'best_h2_n': best_h2_r['h2_n'],
        'h2_combo_score': best_h2_r['h2_avg'] * best_h2_r['h2_total'] if best_h2_r['h2_total'] > 0 else 0,
        'variants': by_mult,
    })

pareto_sorted = sorted(pareto_rows, key=lambda p: p['best_h2_total'], reverse=True)

print(f"\n  ── TOP 20 BASE STRATEGIES: BEST H2 THRESHOLD ──")
print(f"  {'#':>3} │{'FamGrp':>12}│{'Gate':>18}│{'Config':>12}│{'Signal':>14}│"
      f"{'Base_H2%':>9}│{'→Thr':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'H2_N':>5}")
print(f"  {'─'*3}─┼{'─'*12}┼{'─'*18}┼{'─'*12}┼{'─'*14}┼{'─'*9}┼{'─'*5}┼{'─'*17}┼{'─'*5}")
for i, p in enumerate(pareto_sorted[:20]):
    bk = p['base_key']
    print(f"  {i+1:3d} │{bk[0]:>12}│{bk[1]:>18}│{bk[2]:>12}│{bk[3]:>14}│"
          f"{p['base_h2_total']*100:>+8.1f}│{p['best_h2_mult']:>4.2f}x│"
          f"{p['best_h2_avg']*100:>+7.2f}{p['best_h2_total']*100:>+8.1f}│"
          f"{p['best_h2_n']:>5}")

# FAMILY AGREEMENT GATE SWEEP — H2 focused
print(f"\n  ══ FAMILY AGREEMENT GATE SWEEP — H2 FOCUS ══")
print(f"  {'Fam_min':>7} {'FA':>5} {'Tks':>4} (at 1.0x/best_h2/single_best)")
for fam_min in [1, 2, 3, 4, 5]:
    for fa in [0.0, 0.5, 0.6]:
        gp = {'min_families': fam_min, 'va_cons_min': 0.0, 'seed_agree_min': 0.0, 'fam_agree_min': fa}
        gl = make_gate_label(gp)
        matches = [r for r in all_results
                   if r['gate'] == gl and r['config'] == 'best_h2'
                   and r['signal'] == 'single_best' and r['family_group'] == 'ALL'
                   and r.get('thresh_mult', 1.0) == 1.0]
        if matches:
            r = matches[0]
            print(f"  F>={fam_min}    {fa:>4.0%} {r['n_tickers']:>4} tks, {r['h2_n']:>4} H2 tr, "
                  f"H2_avg={r['h2_avg']*100:>+.2f}%, H2_tot={r['h2_total']*100:>+.1f}%, "
                  f"cap={r.get('oracle_h2_capture',0)*100:.0f}%")

# Per-ticker breakdown
def _print_per_ticker_breakdown(label, strat_r, exp_configs, ticker_features,
                                 daily_probs, exp_fire_dates_by_mult,
                                 date_trade_pnl, cross_best_exps,
                                 oracle_h2_pos_set, oracle_h2_total, h2_start_dt,
                                 h1_pnl=None, h2_pnl=None, h2_pnl_avg=None,
                                 exp_configs_worst=None,
                                 date_trade_pnl_fixed=None,
                                 oracle_h2_per_ticker=None):
    tm = strat_r.get('thresh_mult', 1.0)
    print(f"\n  ── {label} PER-TICKER BREAKDOWN ──")
    print(f"  Strategy: {_thr(strat_r)}x|{strat_r['family_group']}|{strat_r['gate']}|"
          f"{strat_r['config']}|{strat_r['signal']}")
    fire_dates = exp_fire_dates_by_mult[tm]
    args = dict(ticker_features=ticker_features, exp_configs=exp_configs,
                daily_probs=daily_probs, exp_fire_dates=fire_dates,
                date_trade_pnl=date_trade_pnl, cross_best_exps=cross_best_exps,
                thresh_mult=tm, h1_pnl=h1_pnl, h2_pnl=h2_pnl, h2_pnl_avg=h2_pnl_avg,
                exp_configs_worst=exp_configs_worst,
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
    ).sort_values('h2_total', ascending=False)
    oht = oracle_h2_per_ticker or {}
    print(f"  {'Ticker':>8} {'Exp':>16} {'TP%':>5} {'SL%':>5} "
          f"{'H2_Tr':>5} {'H2_Avg%':>8} {'H2_Tot%':>8} "
          f"{'Oracle_H2%':>10} {'Gap%':>6}")
    for t, row in tk_stats.iterrows():
        e_used = tk_exp.get(t, '?')
        cfg = exp_configs.get((t, e_used), {})
        tp = cfg.get('tp_pct', 0) * 100
        sl = cfg.get('sl_pct', 0) * 100
        orc_h2 = oht.get(t, {}).get('h2_total', 0)
        h2_t = row['h2_total']
        h2_n = int(row['h2_n'])
        h2_avg = h2_t / h2_n if h2_n > 0 else 0
        gap = h2_t - orc_h2
        print(f"  {t:>8} {str(e_used)[:16]:>16} {tp:>4.1f}% {sl:>4.1f}% "
              f"{h2_n:>5} {h2_avg*100:>+7.2f} {h2_t*100:>+7.1f} "
              f"{orc_h2*100:>+9.1f} {gap*100:>+5.1f}")

_bd_args = dict(exp_configs=exp_configs, ticker_features=ticker_features,
                daily_probs=daily_probs, exp_fire_dates_by_mult=exp_fire_dates_by_mult,
                date_trade_pnl=date_trade_pnl, cross_best_exps=cross_best_exps,
                oracle_h2_pos_set=oracle_h2_pos_set, oracle_h2_total=oracle_h2_total,
                h2_start_dt=h2_start_dt,
                h1_pnl=h1_pnl, h2_pnl=h2_pnl, h2_pnl_avg=h2_pnl_avg,
                exp_configs_worst=exp_configs_worst,
                date_trade_pnl_fixed=date_trade_pnl_fixed,
                oracle_h2_per_ticker=oracle_h2_per_ticker)

if h2_sorted:
    _print_per_ticker_breakdown("BEST H2 TOTAL", h2_sorted[0], **_bd_args)
if h2_avg_sorted:
    _print_per_ticker_breakdown("BEST H2 AVG (min 5)", h2_avg_sorted[0], **_bd_args)

# best_h2 cheating oracle breakdown
h2_oracle_matches = [r for r in all_results
                     if r['config'] == 'best_h2' and r['signal'] == 'single_best'
                     and r['family_group'] == 'ALL' and r['gate'] == 'ANY'
                     and r.get('thresh_mult', 1.0) == 1.0]
if h2_oracle_matches:
    _print_per_ticker_breakdown("H2 ORACLE (cheating baseline)", h2_oracle_matches[0], **_bd_args)

# S3 equivalent
s3_matches = [r for r in all_results
              if r['config'] == 's3_score' and r['signal'] == 'single_best'
              and r['family_group'] == 'ALL' and r['gate'] == 'ANY'
              and r.get('thresh_mult', 1.0) == 1.0]
if s3_matches:
    _print_per_ticker_breakdown("S3 PRODUCTION STRATEGY", s3_matches[0], **_bd_args)

# Category comparison
print(f"\n  ── CATEGORY COMPARISON (H2 FOCUS) ──")
categories = {
    'H2 oracle':      h2_oracle_matches[0] if h2_oracle_matches else None,
    'Best H2 total':   h2_sorted[0] if h2_sorted else None,
    'Best H2 avg':     h2_avg_sorted[0] if h2_avg_sorted else None,
    'Best H2 combo':   h2_combo_ranked[0] if h2_combo_ranked else None,
    'S3 production':   s3_matches[0] if s3_matches else None,
}
print(f"  {'Category':>18} │ {'Strategy':<55} │{'H2_Tot%':>9} │{'H2_Avg%':>8} │{'H2Cap%':>6} │{'Full%':>8}")
print(f"  {'─'*18}─┼{'─'*56}┼{'─'*10}┼{'─'*9}┼{'─'*7}┼{'─'*8}")
for cat, r in categories.items():
    if r is None: continue
    lbl = f"{_thr(r)}x|{r['family_group']}|{r['gate']}|{r['config']}|{r['signal']}"
    print(f"  {cat:>18} │ {lbl:<55} │{r['h2_total']*100:>+8.1f} │{r['h2_avg']*100:>+7.2f} "
          f"│{r.get('oracle_h2_capture',0)*100:>5.0f}% │{r['total_pnl']*100:>+7.1f}")

# DIVERSIFIED DEPLOYMENT — H2 focused
print(f"\n  ══ DIVERSIFIED DEPLOYMENT: ≥3 H2 tickers, max H2 conc <60%, H2 avg>0% ══")
diversified = sorted(
    [r for r in all_results
     if r.get('h2_tickers', 0) >= 3
     and r.get('max_h2_concentration', 1.0) < 0.60
     and r['h2_avg'] > 0
     and r['h2_n'] >= 5],
    key=lambda r: r['h2_total'], reverse=True)
print(f"  {len(diversified)} strategies found")
if diversified:
    print(f"  {'#':>3} │{'Thr':>5}│{'FamGrp':>12}│{'Gate':>22}│{'Config':>12}│{'Signal':>14}│"
          f"{'H2_Tks':>6}{'H2_N':>5}│{'H2_Avg%':>8}{'H2_Tot%':>9}│{'H2Cap%':>6}")
    print(f"  {'─'*3}─┼{'─'*5}┼{'─'*12}┼{'─'*22}┼{'─'*12}┼{'─'*14}┼{'─'*11}┼{'─'*17}┼{'─'*6}")
    for i, r in enumerate(diversified[:15]):
        print(f"  {i+1:3d} │{_thr(r):>5}│{r['family_group']:>12}│{r['gate']:>22}│{r['config']:>12}│{r['signal']:>14}│"
              f"{r.get('h2_tickers',0):>6}{r['h2_n']:>5}│"
              f"{r['h2_avg']*100:>+7.2f}{r['h2_total']*100:>+8.1f}│"
              f"{r.get('oracle_h2_capture',0)*100:>5.0f}%")
    _print_per_ticker_breakdown("BEST DIVERSIFIED (H2)", diversified[0], **_bd_args)


# ==============================================================================
# PHASE 6: SAVE CSVs + CHARTS
# ==============================================================================
print(f"\n  PHASE 6: Saving results...")

csv_rows = []
for r in all_results:
    row = {k: v for k, v in r.items()
           if k not in ('tickers', 'monthly', '_gate_params')}
    monthly = r.get('monthly', {})
    for m in ['2026-04', '2026-05', '2026-06', '2026-07', '2026-08']:
        row[f'pnl_{m}'] = monthly.get(m, 0.0)
    csv_rows.append(row)

full_df = pd.DataFrame(csv_rows).sort_values('h2_total', ascending=False)
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
        'base_h2_total': p['base_h2_total'], 'base_h2_avg': p['base_h2_avg'],
        'base_h2_n': p['base_h2_n'], 'base_total': p['base_total'],
        'best_h2_mult': p['best_h2_mult'],
        'best_h2_total': p['best_h2_total'], 'best_h2_avg': p['best_h2_avg'],
        'best_h2_n': p['best_h2_n'],
        'h2_combo_score': p['h2_combo_score'],
    })
pd.DataFrame(pareto_csv_rows).to_csv(OUTPUT_PARETO, index=False)
print(f"    Pareto:      {OUTPUT_PARETO} ({len(pareto_csv_rows)} base strategies)")

# Charts
print(f"\n    Generating charts...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('H2-TARGETED ORACLE — LONG Strategy Sweep', fontsize=14, fontweight='bold')

# [0,0] Config method H2 comparison
ax = axes[0, 0]
cm_data = []
for cm in CONFIG_METHODS:
    matches = [r for r in all_results
               if r['config'] == cm and r['signal'] == 'single_best'
               and r['family_group'] == 'ALL' and r['gate'] == 'ANY'
               and r.get('thresh_mult', 1.0) == 1.0]
    if matches:
        r = matches[0]
        cm_data.append((cm, r['h2_avg'] * 100, r['h2_total'] * 100))
if cm_data:
    cm_names = [d[0] for d in cm_data]
    cm_h2_avg = [d[1] for d in cm_data]
    cm_h2_tot = [d[2] for d in cm_data]
    x = np.arange(len(cm_names))
    w = 0.35
    ax.bar(x - w/2, cm_h2_avg, w, label='H2 avg PnL', color='coral', alpha=0.8)
    ax.bar(x + w/2, cm_h2_tot, w, label='H2 total PnL', color='darkred', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cm_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('H2 PnL (%)')
    ax.set_title('Config Method: H2 Performance (1.0x, ALL, ANY)')
    ax.legend(fontsize=8)

# [0,1] H2 oracle ceiling per ticker
ax = axes[0, 1]
tk_h2 = sorted(oracle_h2_per_ticker.items(), key=lambda x: x[1]['h2_total'], reverse=True)
tk_names = [t for t, _ in tk_h2[:25]]
tk_vals = [v['h2_total'] * 100 for _, v in tk_h2[:25]]
colors = ['#2E7D32' if v > 0 else '#C62828' for v in tk_vals]
ax.barh(range(len(tk_names)), tk_vals, color=colors, alpha=0.8)
ax.set_yticks(range(len(tk_names)))
ax.set_yticklabels(tk_names, fontsize=7)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('H2 Oracle Total PnL (%)')
ax.set_title(f'H2 Oracle Ceiling by Ticker (top 25)')
ax.invert_yaxis()

# [1,0] H2 total distribution
ax = axes[1, 0]
h2_totals = [r['h2_total'] * 100 for r in all_results if r['h2_n'] > 0]
ax.hist(h2_totals, bins=50, color='coral', alpha=0.7, edgecolor='white')
ax.axvline(0, color='red', linewidth=1, linestyle='--')
if h2_totals:
    ax.axvline(np.median(h2_totals), color='orange', linewidth=1,
               label=f'Median: {np.median(h2_totals):+.1f}%')
ax.set_xlabel('H2 Total PnL (%)')
ax.set_ylabel('Strategy Count')
ax.set_title(f'H2 Total PnL Distribution ({len(h2_totals)} strategies)')
ax.legend(fontsize=8)

# [1,1] Family group H2 comparison
ax = axes[1, 1]
fg_data = []
for fg in FAMILY_GROUPS:
    matches = [r for r in all_results
               if r['family_group'] == fg and r['config'] == 'best_h2'
               and r['signal'] == 'single_best' and r['gate'] == 'ANY'
               and r.get('thresh_mult', 1.0) == 1.0]
    if matches:
        r = matches[0]
        fg_data.append((fg, r['h2_total'] * 100, r['total_pnl'] * 100))
if fg_data:
    fg_names = [d[0] for d in fg_data]
    fg_h2 = [d[1] for d in fg_data]
    fg_full = [d[2] for d in fg_data]
    x = np.arange(len(fg_names))
    w = 0.35
    ax.bar(x - w/2, fg_h2, w, label='H2 total', color='#C62828', alpha=0.8)
    ax.bar(x + w/2, fg_full, w, label='Full total', color='#2E7D32', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(fg_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Total PnL (%)')
    ax.set_title('Family Group: H2 vs Full (best_h2, single_best)')
    ax.legend(fontsize=8)

plt.tight_layout()
chart_path = '/tmp/oracle_long_h2_strategies.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"    Chart: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  H2-TARGETED FINAL SUMMARY")
print(f"{'='*W}")
print(f"  Strategies evaluated: {n_done}")
print(f"  Strategies with trades: {len(all_results)}")
print(f"  H2 Oracle ceiling: {oracle_h2_total*100:+.1f}% total ({len(oracle_h2_pos_set)} positive tickers)")
print(f"  H2 Oracle negative: {len(oracle_h2_neg_set)} tickers (even oracle loses in H2)")
print(f"  S3 equivalent H2: {s3_metrics['h2_total']*100:+.1f}% "
      f"({s3_metrics['h2_n']} trades, capture={s3_metrics['oracle_h2_capture']*100:.0f}%)")
print(f"")
if h2_sorted:
    r = h2_sorted[0]
    print(f"  BEST H2 TOTAL: {_thr(r)}x|{r['family_group']}|{r['gate']}|{r['config']}|{r['signal']}")
    print(f"    {r.get('h2_tickers',0)} H2 tickers, {r['h2_n']} H2 trades, "
          f"H2_avg={r['h2_avg']*100:+.2f}%, H2_total={r['h2_total']*100:+.1f}%, "
          f"H2_capture={r.get('oracle_h2_capture',0)*100:.0f}%")
if h2_avg_sorted:
    r = h2_avg_sorted[0]
    print(f"  BEST H2 AVG: {_thr(r)}x|{r['family_group']}|{r['gate']}|{r['config']}|{r['signal']}")
    print(f"    {r['h2_n']} H2 trades, H2_avg={r['h2_avg']*100:+.2f}%, H2_total={r['h2_total']*100:+.1f}%")
if h2_oracle_matches:
    r = h2_oracle_matches[0]
    print(f"  H2 ORACLE (cheating): {r['n_tickers']} tickers, {r['h2_n']} H2 trades, "
          f"H2_avg={r['h2_avg']*100:+.2f}%, H2_total={r['h2_total']*100:+.1f}%")
print(f"")
print(f"  KEY DIAGNOSTIC:")
if oracle_h2_total > 0:
    print(f"    H2 oracle IS positive ({oracle_h2_total*100:+.1f}%) → signal exists but strategy selection matters")
    gap_pct = (1 - s3_metrics['oracle_h2_capture']) * 100 if s3_metrics['oracle_h2_capture'] > 0 else 100
    print(f"    S3 captures {s3_metrics['oracle_h2_capture']*100:.0f}% of H2 oracle → {gap_pct:.0f}% left on table")
else:
    print(f"    H2 oracle is NEGATIVE ({oracle_h2_total*100:+.1f}%) → models fundamentally broken post-collapse")
    print(f"    No config selection method can help — need new models or regime filter")
print(f"")
print(f"  Files saved:")
print(f"    {OUTPUT_ALL} ({len(full_df)} strategies)")
print(f"    {OUTPUT_TOP}")
print(f"    {OUTPUT_PARETO} ({len(pareto_csv_rows)} base strategies)")
print(f"    {chart_path}")
print(f"{'='*W}")
