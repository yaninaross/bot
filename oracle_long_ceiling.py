"""
oracle_long_ceiling.py — LONG Oracle: True Ceiling per Asset (ALL modes)
========================================================================
Exhaustively backtests ALL LONG BQ configs for ALL tickers across ALL 7 modes:
  static, atr, atr_adaptive, atr_sl, trailing_decay, grace, limit_entry
  (skips regime_scaled, regime_adaptive — need runtime VIX data)

Then computes S3 baseline per ticker to measure the config-selection gap.

Output:
  1. /tmp/oracle_long_per_config_results.csv — ALL configs × ALL tickers
     → ML model training data (BQ features + CSV backtest ground truth)
  2. /tmp/oracle_long_per_ticker_ceiling.csv — per-ticker ceiling summary
  3. /tmp/oracle_long_ceiling.png — 4-panel chart
  4. Mode Analysis table: which mode wins empirically?

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
BUCKET_NAME    = "time-series-bucket-ys"
RESULTS_DIR    = "Algos/Results_scored_20260703"
RESULTS_FALLBACK = "Algos/Results_all_v2"
INFERENCE_DIR  = "Data/Inference_v2"

FULL_START  = '2026-04-01'
FULL_END    = '2026-07-23'
HALF1_END   = '2026-05-31'
HALF2_START = '2026-06-01'

EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.0015

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12', 'robust_wf_v12_stitched')

MAX_VA_OPT_RATIO = 3.0
DEPLOY_THRESHOLD = 0.018
MIN_SEEDS = 3

SKIP_MODES = {'regime_scaled', 'regime_adaptive'}

GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110

OUTPUT_PER_CONFIG = '/tmp/oracle_long_per_config_results.csv'
OUTPUT_PER_TICKER = '/tmp/oracle_long_per_ticker_ceiling.csv'
OUTPUT_CHART      = '/tmp/oracle_long_ceiling.png'


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
        if sl_h:
            return -sl, ed, True
        if tp_h:
            return tp, ed, True
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
        if sl_active and sl_h:
            return -sl, ed, True
        if tp_h:
            return tp, ed, True
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
        if date_ints[k] != ed:
            break
        if time_mins[k] - et > max_wait:
            break
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
        if sl_h:
            return -sl, ed, True
        if tp_h:
            return tp, ed, True
        if time_mins[k] >= exit_time:
            return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


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
# PHASE 1: QUERY BQ
# ==============================================================================
print("=" * W)
print("  LONG ORACLE — TRUE CEILING PER ASSET (ALL 7 MODES)")
print("  Exhaustive backtest: ALL configs × ALL tickers")
print("=" * W)

print(f"\n  PHASE 1: Querying BQ...")
versions_sql = ', '.join(f"'{v}'" for v in BQ_MODEL_VERSIONS)
q = f"""
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult, grace_minutes,
       dip_pct, max_wait_minutes, vol_filter_pct,
       pre_range_filter_pct,
       opt_avg_pnl, opt_win_rate, opt_count, opt_freq,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       model_version, is_ensemble
FROM `{BQ_TABLE}`
WHERE model_version IN ({versions_sql})
  AND survived_a = TRUE
  AND val_a_count >= 3
  AND opt_avg_pnl > 0
  AND val_a_avg_pnl > 0
"""
bq_df = bq_client.query(q).to_dataframe()
print(f"    {len(bq_df)} raw BQ rows loaded")

bq_df = bq_df[bq_df['val_a_avg_pnl'] / bq_df['opt_avg_pnl'] <= MAX_VA_OPT_RATIO]
print(f"    {len(bq_df)} after overfit filter (va/opt <= {MAX_VA_OPT_RATIO})")

bq_df = bq_df[~bq_df['tpsl_mode'].isin(SKIP_MODES)]
print(f"    {len(bq_df)} after dropping {SKIP_MODES}")

all_tickers = sorted(bq_df['ticker'].unique())
all_modes = sorted(bq_df['tpsl_mode'].unique())
print(f"    {len(all_tickers)} tickers, {len(all_modes)} modes: {', '.join(all_modes)}")
for m in all_modes:
    cnt = len(bq_df[bq_df['tpsl_mode'] == m])
    print(f"      {m:<20} {cnt:>6} configs")


# ==============================================================================
# PHASE 2: LOAD OHLCV + CSV SCORES
# ==============================================================================
print(f"\n  PHASE 2: Loading OHLCV + CSV scores...")
full_start_dt = pd.Timestamp(FULL_START).date()
full_end_dt   = pd.Timestamp(FULL_END).date()
h1_end_dt     = pd.Timestamp(HALF1_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

ticker_data = {}
csv_scores_cache = {}

for i, ticker in enumerate(all_tickers):
    raw = load_master(ticker)
    if raw is None:
        continue
    healed = heal_data(raw)
    ohlcv = precompute_ohlcv_arrays(healed, FULL_START, FULL_END)
    if ohlcv is None:
        continue

    atr_series = compute_daily_atr_pct(healed)
    pst_idx = healed.index.tz_convert('America/Los_Angeles') \
        if str(healed.index.tz) != 'America/Los_Angeles' else healed.index
    s_ts = pd.to_datetime(FULL_START).tz_localize('America/Los_Angeles')
    e_ts = pd.to_datetime(FULL_END).tz_localize('America/Los_Angeles') + timedelta(days=1)
    mask = (pst_idx >= s_ts) & (pst_idx < e_ts)
    atr_cut = atr_series[mask]
    ohlcv['atr_pct'] = atr_cut.values.astype(np.float64) if len(atr_cut) == len(ohlcv['prices']) \
        else np.full(len(ohlcv['prices']), 0.05)

    ticker_data[ticker] = {'ohlcv': ohlcv}

    exps = bq_df[bq_df['ticker'] == ticker]['experiment'].unique()
    base_exps = set(_base_experiment(e) for e in exps)
    for be in base_exps:
        if (ticker, be) not in csv_scores_cache:
            scores = load_seed_scores(ticker, be)
            if scores:
                csv_scores_cache[(ticker, be)] = scores

    if (i + 1) % 10 == 0:
        print(f"    [{i+1}/{len(all_tickers)}] loaded...")
        _release_memory()

print(f"    {len(ticker_data)} tickers with OHLCV, {len(csv_scores_cache)} (ticker,exp) CSV caches")
_release_memory()


# ==============================================================================
# PHASE 3: EXHAUSTIVE BACKTEST — ALL CONFIGS × ALL TICKERS × ALL MODES
# ==============================================================================
print(f"\n  PHASE 3: Exhaustive backtesting (7 modes)...")
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

        prob_thresh    = _safe_float(bq_row['prob_thresh'])
        tp_pct         = _safe_float(bq_row.get('tp_pct', 0))
        sl_pct         = _safe_float(bq_row.get('sl_pct', 0))
        tp_mult        = _safe_float(bq_row.get('tp_mult', 0))
        sl_mult        = _safe_float(bq_row.get('sl_mult', 0))
        trail_start_m  = _safe_float(bq_row.get('trail_start_mult', 0))
        trail_end_m    = _safe_float(bq_row.get('trail_end_mult', 0))
        grace_min      = _safe_int(bq_row.get('grace_minutes', 0))
        dip_pct        = _safe_float(bq_row.get('dip_pct', 0))
        max_wait_min   = _safe_int(bq_row.get('max_wait_minutes', 0))

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

            # ── Mode dispatch (LONG buy-side) ──
            if mode == 'limit_entry':
                pnl, td, filled = run_long_trade_limit_entry(
                    prices, highs, lows, date_ints, time_mins,
                    idx, tp_pct, sl_pct, EXIT_TIME_MIN, dip_pct, max_wait_min)
            elif mode == 'grace':
                pnl, td, filled = run_long_trade_grace(
                    prices, highs, lows, date_ints, time_mins,
                    idx, tp_pct, sl_pct, EXIT_TIME_MIN, grace_min)
            elif mode == 'trailing_decay':
                pnl, td, filled = run_long_trade_trailing_decay(
                    prices, highs, lows, date_ints, time_mins,
                    idx, float(atr_arr[idx]),
                    trail_start_m, trail_end_m, sl_mult, EXIT_TIME_MIN)
            elif mode == 'atr_sl':
                atr_val = float(atr_arr[idx])
                atr_val = atr_val if atr_val > 0 else 0.05
                computed_sl = min(max(atr_val * sl_mult, 0.01), 0.25)
                pnl, td, filled = run_long_trade_static(
                    prices, highs, lows, date_ints, time_mins,
                    idx, tp_pct, computed_sl, EXIT_TIME_MIN)
            elif mode in ('atr', 'atr_adaptive'):
                atr_val = float(atr_arr[idx])
                atr_val = atr_val if atr_val > 0 else 0.05
                computed_tp = min(max(atr_val * tp_mult, 0.005), 0.30)
                computed_sl = min(max(atr_val * sl_mult, 0.01), 0.25)
                pnl, td, filled = run_long_trade_static(
                    prices, highs, lows, date_ints, time_mins,
                    idx, computed_tp, computed_sl, EXIT_TIME_MIN)
            else:
                pnl, td, filled = run_long_trade_static(
                    prices, highs, lows, date_ints, time_mins,
                    idx, tp_pct, sl_pct, EXIT_TIME_MIN)

            if not filled:
                continue

            pnl -= 2 * SLIPPAGE_PCT
            last_td = d_int
            trades.append({'date': d_date, 'pnl': pnl})

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
            'tp_mult': tp_mult, 'sl_mult': sl_mult,
            'trail_start_mult': trail_start_m, 'trail_end_mult': trail_end_m,
            'grace_minutes': grace_min,
            'dip_pct': dip_pct, 'max_wait_minutes': max_wait_min,
            'is_limit_entry': int(mode == 'limit_entry'),
            'vol_filter_pct': _safe_float(bq_row.get('vol_filter_pct', 0)),
            'pre_range_filter_pct': _safe_float(bq_row.get('pre_range_filter_pct', 0)),
            'opt_avg_pnl': opt_pnl,
            'opt_win_rate': _safe_float(bq_row.get('opt_win_rate')),
            'opt_count': _safe_float(bq_row.get('opt_count')),
            'opt_freq': _safe_float(bq_row.get('opt_freq', 0)),
            'val_a_avg_pnl': va_pnl,
            'val_a_win_rate': _safe_float(bq_row.get('val_a_win_rate')),
            'val_a_count': _safe_float(bq_row.get('val_a_count')),
            'val_a_freq': _safe_float(bq_row.get('val_a_freq', 0)),
            'va_opt_ratio': va_opt_ratio,
            'bq_vb_avg_pnl': _safe_float(bq_row.get('val_b_avg_pnl')),
            'bq_vb_win_rate': _safe_float(bq_row.get('val_b_win_rate')),
            'bq_vb_count': _safe_float(bq_row.get('val_b_count')),
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
# PHASE 4: S3 BASELINE (production strategy per ticker)
# ==============================================================================
print(f"\n  PHASE 4: Computing S3 baseline per ticker...")

indiv = bq_df[bq_df['is_ensemble'] == False].copy()
for k in GRP_KEYS:
    if k not in indiv.columns:
        indiv[k] = 0
    else:
        indiv[k] = indiv[k].fillna(0)

sa = indiv.groupby(GRP_KEYS).agg(
    n_seeds=('seed', 'nunique'),
    n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= DEPLOY_THRESHOLD).sum()),
    mean_va_pnl=('val_a_avg_pnl', 'mean'),
    mean_va_wr=('val_a_win_rate', 'mean'),
    mean_va_count=('val_a_count', 'mean'),
    vb_pnl=('val_b_avg_pnl', 'mean'),
).reset_index()
sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)
sa = sa[sa['n_seeds'] >= MIN_SEEDS].copy()

sa_limit = sa[sa['tpsl_mode'] == 'limit_entry'].copy()
sa_limit['score'] = (sa_limit['mean_va_pnl'] ** 2) * sa_limit['agree_18']
sa_limit['exp_fam'] = sa_limit['experiment'].apply(assign_family)
sa_fam = sa_limit[sa_limit['exp_fam'] != 'other'].copy()

best_per_fam = sa_fam.loc[sa_fam.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
best_per_fam = best_per_fam[best_per_fam['mean_va_pnl'] >= DEPLOY_THRESHOLD]

agreement = {}
for tk in best_per_fam['ticker'].unique():
    tk_rows = best_per_fam[best_per_fam['ticker'] == tk]
    n_fams = len(tk_rows)
    n_pos = int((tk_rows['vb_pnl'] > 0).sum())
    agreement[tk] = {
        'n_fams': n_fams, 'n_pos': n_pos,
        'majority_pos': n_pos > n_fams / 2 and n_fams >= 2,
    }

s3_tickers = sorted([tk for tk, ag in agreement.items() if ag['majority_pos']])
print(f"    Family agreement: {len(s3_tickers)} tickers pass S3 gate")

sa_s3 = sa_limit[sa_limit['ticker'].isin(s3_tickers)].copy()
sa_s3['score'] = (sa_s3['mean_va_pnl'] ** 2) * sa_s3['agree_18']
best_s3 = sa_s3.loc[sa_s3.groupby('ticker')['score'].idxmax()]
best_s3 = best_s3[best_s3['mean_va_pnl'] >= DEPLOY_THRESHOLD]

s3_backtest = {}
for _, row in best_s3.iterrows():
    tk = row['ticker']
    match = results_df[
        (results_df['ticker'] == tk) &
        (results_df['experiment'] == row['experiment']) &
        (results_df['tpsl_mode'] == 'limit_entry') &
        (results_df['entry_minute'] == int(row['entry_minute'])) &
        (results_df['prob_thresh'] == float(row['prob_thresh'])) &
        (results_df['tp_pct'] == float(row['tp_pct'])) &
        (results_df['sl_pct'] == float(row['sl_pct']))
    ]
    if not match.empty:
        avg_pnl = match['bt_full_avg'].mean()
        avg_h2  = match['bt_h2_avg'].mean()
        avg_n   = int(match['bt_full_n'].mean())
    else:
        avg_pnl = 0.0
        avg_h2  = 0.0
        avg_n   = 0
    s3_backtest[tk] = {'full': avg_pnl, 'h2': avg_h2, 'n': avg_n}

print(f"    S3 baseline computed for {len(s3_backtest)} tickers: {', '.join(sorted(s3_backtest.keys()))}")


# ==============================================================================
# PHASE 5: PER-TICKER CEILING ANALYSIS
# ==============================================================================
print(f"\n  PHASE 5: Per-ticker ceiling analysis...")

per_ticker = []
for ticker in sorted(results_df['ticker'].unique()):
    grp = results_df[results_df['ticker'] == ticker]
    n_cfg = len(grp)
    if n_cfg == 0:
        continue

    n_pos_full = (grp['bt_full_avg'] > 0).sum()

    best_full = grp.loc[grp['bt_full_avg'].idxmax()]

    h2_with_trades = grp[grp['bt_h2_n'] > 0]
    if not h2_with_trades.empty:
        best_h2 = grp.loc[h2_with_trades['bt_h2_avg'].idxmax()]
    else:
        best_h2 = best_full

    s3_full = s3_backtest.get(ticker, {}).get('full', 0.0)
    s3_h2   = s3_backtest.get(ticker, {}).get('h2', 0.0)
    s3_n    = s3_backtest.get(ticker, {}).get('n', 0)

    med_full = grp['bt_full_avg'].median()

    per_ticker.append({
        'ticker': ticker,
        'n_configs': n_cfg,
        'n_positive_full': n_pos_full,
        'pct_positive': n_pos_full / n_cfg,
        'oracle_full': float(best_full['bt_full_avg']),
        'oracle_full_wr': float(best_full['bt_full_wr']),
        'oracle_full_n': int(best_full['bt_full_n']),
        'oracle_h1': float(best_full['bt_h1_avg']),
        'oracle_h2': float(best_full['bt_h2_avg']),
        'best_h2': float(best_h2['bt_h2_avg']),
        'best_h2_full': float(best_h2['bt_full_avg']),
        's3_full': s3_full,
        's3_h2': s3_h2,
        's3_n': s3_n,
        'gap_full': float(best_full['bt_full_avg']) - s3_full,
        'gap_h2': float(best_h2['bt_h2_avg']) - s3_h2,
        'median_full': med_full,
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
# PHASE 6: OUTPUT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 6: RESULTS")
print(f"{'='*W}")

n_total = len(ceiling_df)
n_oracle_pos = (ceiling_df['oracle_full'] > 0).sum()
n_s3_pos     = (ceiling_df['s3_full'] > 0).sum()
n_oracle_strong = (ceiling_df['oracle_full'] > 0.01).sum()

print(f"\n  ── SUMMARY ──")
print(f"    {n_total} tickers with backtest data")
print(f"    {n_oracle_pos} ({100*n_oracle_pos/n_total:.0f}%) have ≥1 profitable config (TRUE ORACLE)")
print(f"    {n_s3_pos} ({100*n_s3_pos/n_total:.0f}%) profitable with S3 selection")
print(f"    {n_oracle_strong} ({100*n_oracle_strong/n_total:.0f}%) have oracle PnL > +1.0%")
print(f"    Gap: {n_oracle_pos - n_s3_pos} additional tickers become profitable with oracle config")
print(f"    Avg oracle ceiling: {ceiling_df['oracle_full'].mean()*100:+.2f}%")
print(f"    Avg S3 PnL:         {ceiling_df['s3_full'].mean()*100:+.2f}%")
print(f"    Avg config gap:     {ceiling_df['gap_full'].mean()*100:+.2f}%")

# ── BQ val_b vs CSV backtest correlation ──
valid = results_df[(results_df['bq_vb_avg_pnl'] != 0) & (results_df['bt_full_n'] > 0)]
rho_bq = 0.0
if len(valid) > 10:
    rho_bq, p_bq = sp_stats.spearmanr(valid['bq_vb_avg_pnl'], valid['bt_full_avg'])
    print(f"\n  ── BQ val_b vs CSV Backtest Correlation ──")
    print(f"    Spearman ρ = {rho_bq:+.3f} (p={p_bq:.4f}) on {len(valid)} configs")
    label = 'STRONG' if abs(rho_bq) > 0.5 else 'MODERATE' if abs(rho_bq) > 0.3 else 'WEAK'
    print(f"    {label} — {'BQ is reliable proxy' if abs(rho_bq) > 0.5 else 'CSV backtest is more reliable'}")

# ── val_a vs CSV backtest correlation ──
valid_va = results_df[(results_df['val_a_avg_pnl'] != 0) & (results_df['bt_full_n'] > 0)]
if len(valid_va) > 10:
    rho_va, p_va = sp_stats.spearmanr(valid_va['val_a_avg_pnl'], valid_va['bt_full_avg'])
    print(f"\n  ── val_a vs CSV Backtest Correlation ──")
    print(f"    Spearman ρ = {rho_va:+.3f} (p={p_va:.4f}) on {len(valid_va)} configs")
    label = 'STRONG' if abs(rho_va) > 0.5 else 'MODERATE' if abs(rho_va) > 0.3 else 'WEAK'
    print(f"    {label}")

# ── Per-ticker table ──
print(f"\n  ── PER-TICKER TRUE ORACLE (sorted by oracle FULL PnL) ──")
print(f"  {'Ticker':<8}│{'ORACLE':>8} {'WR':>5} {'N':>4} │{'S3':>8} │{'GAP':>8} │{'ORC_H2':>8} {'BST_H2':>8} │{'Ncfg':>5} {'%Pos':>5}│ Best Config")
print(f"  {'─'*8}┼{'─'*18}─┼{'─'*9}┼{'─'*9}┼{'─'*18}─┼{'─'*11}┼{'─'*25}")

for _, row in ceiling_df.iterrows():
    sl_str = f"sl={row['best_sl']:.3f}" if row['best_mode'] != 'atr_sl' else f"sm={row['best_sl']:.1f}"
    cfg_str = f"{row['best_mode'][:6]} tp={row['best_tp']:.3f} {sl_str}"
    flag = " ★" if row['oracle_full'] > 0.01 else ""
    print(f"  {row['ticker']:<8}│{row['oracle_full']*100:+7.2f}% {row['oracle_full_wr']*100:4.0f}% {int(row['oracle_full_n']):3d} │"
          f"{row['s3_full']*100:+7.2f}% │{row['gap_full']*100:+7.2f}% │"
          f"{row['oracle_h2']*100:+7.2f}% {row['best_h2']*100:+7.2f}% │"
          f"{int(row['n_configs']):4d} {row['pct_positive']*100:4.0f}%│ {cfg_str}{flag}")

# ── MODE ANALYSIS (key question: is limit_entry the best mode?) ──
print(f"\n  ══ MODE ANALYSIS — Is limit_entry the best mode? ══")
print(f"\n  ── A) Winning Mode Distribution (all tickers by oracle) ──")
mode_counts = ceiling_df['best_mode'].value_counts()
for mode, cnt in mode_counts.items():
    tks = ceiling_df[ceiling_df['best_mode'] == mode]['ticker'].tolist()
    pct = 100 * cnt / len(ceiling_df)
    preview = ', '.join(tks[:10])
    if len(tks) > 10:
        preview += f' +{len(tks)-10} more'
    print(f"    {mode:<18} {cnt:3d} ({pct:4.0f}%) — {preview}")

print(f"\n  ── B) Per-Mode Aggregate Performance (all configs) ──")
print(f"  {'Mode':<18}│{'Configs':>8} │{'Avg PnL':>9} │{'Med PnL':>9} │{'Avg WR':>8} │{'%Pos':>6} │{'Avg N':>6}")
print(f"  {'─'*18}┼{'─'*9}┼{'─'*10}┼{'─'*10}┼{'─'*9}┼{'─'*7}┼{'─'*7}")
for mode in sorted(results_df['tpsl_mode'].unique()):
    mgrp = results_df[results_df['tpsl_mode'] == mode]
    with_trades = mgrp[mgrp['bt_full_n'] > 0]
    n_pos = (mgrp['bt_full_avg'] > 0).sum()
    avg_pnl = with_trades['bt_full_avg'].mean() if len(with_trades) > 0 else 0
    med_pnl = with_trades['bt_full_avg'].median() if len(with_trades) > 0 else 0
    avg_wr  = with_trades['bt_full_wr'].mean() if len(with_trades) > 0 else 0
    avg_n   = with_trades['bt_full_n'].mean() if len(with_trades) > 0 else 0
    pct_pos = 100 * n_pos / len(mgrp) if len(mgrp) > 0 else 0
    flag = " ◀" if mode == 'limit_entry' else ""
    print(f"  {mode:<18}│{len(mgrp):>8} │{avg_pnl*100:>+8.2f}% │{med_pnl*100:>+8.2f}% │{avg_wr*100:>7.1f}% │{pct_pos:>5.0f}% │{avg_n:>5.1f}{flag}")

print(f"\n  ── C) Best Oracle Mode per Ticker (top 15) ──")
top15 = ceiling_df.head(15)
limit_wins = (top15['best_mode'] == 'limit_entry').sum()
print(f"    limit_entry wins: {limit_wins}/{len(top15)} ({100*limit_wins/len(top15):.0f}%)")
other_modes = top15[top15['best_mode'] != 'limit_entry']
if len(other_modes) > 0:
    print(f"    Non-limit winners:")
    for _, row in other_modes.iterrows():
        print(f"      {row['ticker']:<8} → {row['best_mode']} ({row['oracle_full']*100:+.2f}%)")

# ── Best limit_entry config per ticker vs oracle (any mode) ──
print(f"\n  ── D) limit_entry Ceiling vs True Oracle (any mode) ──")
print(f"  {'Ticker':<8}│{'Oracle':>8} {'Mode':<12}│{'Lim Best':>8} │{'Gap':>8}")
print(f"  {'─'*8}┼{'─'*21}┼{'─'*9}┼{'─'*9}")
n_limit_matches = 0
for _, row in ceiling_df.iterrows():
    tk = row['ticker']
    lim_grp = results_df[(results_df['ticker'] == tk) & (results_df['tpsl_mode'] == 'limit_entry')]
    lim_best = lim_grp['bt_full_avg'].max() if len(lim_grp) > 0 else 0
    gap = row['oracle_full'] - lim_best
    if gap < 0.001:
        n_limit_matches += 1
    print(f"  {tk:<8}│{row['oracle_full']*100:+7.2f}% {row['best_mode']:<12}│{lim_best*100:+7.2f}% │{gap*100:+7.2f}%")

print(f"\n    limit_entry matches oracle: {n_limit_matches}/{len(ceiling_df)} ({100*n_limit_matches/len(ceiling_df):.0f}%)")

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
# PHASE 7: CHARTS
# ==============================================================================
print(f"\n  Generating charts...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('LONG Oracle — True Ceiling per Asset (All 7 Modes)', fontsize=14, fontweight='bold')

ax = axes[0, 0]
tks = ceiling_df['ticker'].values
oracle_vals = ceiling_df['oracle_full'].values * 100
s3_vals = ceiling_df['s3_full'].values * 100
colors = ['#2E7D32' if v > 0 else '#C62828' for v in oracle_vals]
x = np.arange(len(tks))
ax.bar(x, oracle_vals, color=colors, alpha=0.7, label='Oracle (best config, any mode)')
ax.scatter(x, s3_vals, color='blue', s=20, zorder=5, label='S3 selection (limit_entry)')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(tks, rotation=90, fontsize=6)
ax.set_ylabel('Avg PnL (%)')
ax.set_title('Per-Ticker: Oracle vs S3')
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.scatter(ceiling_df['s3_full'] * 100, ceiling_df['oracle_full'] * 100,
           c=colors, s=40, alpha=0.7)
lim = max(abs(ceiling_df['oracle_full'].max()), abs(ceiling_df['s3_full'].max())) * 100 + 1
ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label='No gap')
for _, row in ceiling_df.head(10).iterrows():
    ax.annotate(row['ticker'],
                (row['s3_full'] * 100, row['oracle_full'] * 100),
                fontsize=7, alpha=0.8)
ax.set_xlabel('S3 PnL (%)')
ax.set_ylabel('Oracle PnL (%)')
ax.set_title('Config Selection Gap (S3 vs Oracle)')
ax.legend(fontsize=8)

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
print(f"  S3 selection:          {ceiling_df['s3_full'].mean()*100:+.2f}% avg FULL")
print(f"  Config selection gap:  {ceiling_df['gap_full'].mean()*100:+.2f}% avg")
print(f"")
print(f"  MODE VERDICT:")
limit_oracle_pct = 100 * (ceiling_df['best_mode'] == 'limit_entry').sum() / len(ceiling_df)
print(f"    limit_entry is oracle winner for {limit_oracle_pct:.0f}% of tickers")
top_mode = ceiling_df['best_mode'].value_counts().index[0]
print(f"    Most common oracle mode: {top_mode}")
print(f"")
print(f"  Top 5 by TRUE ORACLE:")
for _, row in ceiling_df.head(5).iterrows():
    print(f"    {row['ticker']:<8} {row['oracle_full']*100:+.2f}% FULL  {row['oracle_h2']*100:+.2f}% H2  (S3: {row['s3_full']*100:+.2f}%, gap: {row['gap_full']*100:+.2f}%, mode: {row['best_mode']})")
print(f"")
print(f"  → Use {OUTPUT_PER_CONFIG} as ML training data:")
print(f"    Features: opt/val_a stats, hyperparams, mode flags  (~20 unbiased)")
print(f"    Target:   bt_full_avg or bt_h2_avg (CSV backtest ground truth)")
print(f"    Samples:  {len(results_df)} rows — enough for neural network")
print(f"{'='*W}")
