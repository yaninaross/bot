# %% ── Combined LONG + SHORT Portfolio Backtest (CSV-Scored) ─────────────────
#
# Combines deployed S3 LONG strategy with FOLD_414 SHORT strategy.
# Shows individual + combined portfolio metrics, equity curves,
# day-level clash analysis (LONG and SHORT signals on same date),
# and whether SHORT vs LONG need prioritization.
#
# LONG: S3 family-agreement gate, limit_entry, BQ-driven config selection
# SHORT: FOLD_414 best val_a per RULE_1 ticker, CSV-scored
# CSV-scored — no GPU required. Colab-ready.

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
import matplotlib.dates as mdates
from matplotlib.patches import Patch

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
RESULTS_DIR_V2 = "Algos/Results_all_v2"
INFERENCE_DIR  = "Data/Inference_v2"

TEST_START  = '2026-04-01'
TEST_END    = '2026-08-04'
HALF2_START = '2026-06-01'
ENTRY_MIN   = 375
EXIT_TIME_MIN = 780

LONG_SLIPPAGE  = 0.0015
SHORT_SLIPPAGE = 0.002
STARTING_CAPITAL = 200_000

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

# ── LONG S3 constants (match daily_score_combined.py production) ──
LONG_MODEL_VERSIONS = ('robust_wf_v12', 'robust_wf_v12_stitched')
DEPLOY_THRESHOLD = 0.018
MIN_SEEDS  = 3
MAX_VA_OPT = 3.0
DEPLOY_EXPERIMENTS = {'152_XSECTOR', '160_XSECTOR', '170_XSECTOR',
                      '175_XSECTOR',
                      '180_XSECTOR', '181_XSECTOR', '182_XSECTOR',
                      '183_XSECTOR', '184_XSECTOR', '185_XSECTOR',
                      '200_XSECTOR', '201_XSECTOR', '202_XSECTOR',
                      '203_XSECTOR', '204_XSECTOR', '205_XSECTOR',
                      '220_XSECTOR', '221_XSECTOR', '222_XSECTOR',
                      '223_XSECTOR', '224_XSECTOR', '225_XSECTOR'}
EXCLUDE_TICKERS = {'LUNR'}

GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']

# ── SHORT constants (match daily_score_short.py production) ──
SHORT_TICKERS  = ['BE', 'IREN', 'OKLO', 'ONDS', 'RKLB', 'WOLF']
SHORT_FOLD     = 414
SHORT_EXP      = '414_XSECTOR'
SHORT_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1',
                        'robust_wf_v12_short_414')
SHORT_MODES    = ('static', 'atr_sl', 'grace')

VOL_LOOKBACK_DAYS = 30
S3_VIX_THRESHOLD = 20
S3_TLT_CRASH = 0
S3_SPY_IGNORE = 0

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120

SIDE_COLORS = {'LONG': '#2196F3', 'SHORT': '#E91E63', 'COMBINED': '#4CAF50'}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def _safe_float(val, default=0.0):
    try:
        if pd.isna(val): return default
    except (TypeError, ValueError): pass
    try: return float(val)
    except (TypeError, ValueError): return default


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
        'volumes': df_cut['Volume'].values.astype(np.float64).copy(),
        'dates':  (idx_cut.year * 10000 + idx_cut.month * 100 + idx_cut.day).values.astype(np.int64),
        'times':  (idx_cut.hour * 60 + idx_cut.minute).values.astype(np.int64),
    }


def precompute_vol_baselines(df_ohlcv, lookback_days=VOL_LOOKBACK_DAYS):
    times = (df_ohlcv.index.hour * 60 + df_ohlcv.index.minute).values
    dates = (df_ohlcv.index.year * 10000 + df_ohlcv.index.month * 100 + df_ohlcv.index.day).values.astype(np.int64)
    volumes = df_ohlcv['Volume'].values
    baselines = {}
    for em in range(358, 393):
        at_em = np.where(times == em)[0]
        for j in range(len(at_em)):
            d_int = int(dates[at_em[j]])
            start_j = max(0, j - lookback_days)
            if j - start_j < 5:
                baselines[(d_int, em)] = 0.0
                continue
            baselines[(d_int, em)] = float(np.median(volumes[at_em[start_j:j]]))
    return baselines


def check_volume_filter(vol_baselines, d_int, entry_minute, volume, vol_filter_pct):
    if vol_filter_pct <= 0:
        return True
    baseline = vol_baselines.get((d_int, entry_minute), 0.0)
    if baseline <= 0:
        return True
    return (volume / baseline) * 100 >= vol_filter_pct


def preload_regime_data(eval_start, eval_end):
    print("  Loading regime data (VIX, SPY, TLT)...")
    def _load(ticker):
        clean = TICKER_FILE_MAP.get(ticker, ticker)
        uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
        try:
            d = pd.read_parquet(uri).reset_index()
            dt_cols = [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]
            if dt_cols: d['Datetime'] = pd.to_datetime(d[dt_cols[0]], utc=True)
            else:
                nc = next((c for c in d.columns if c in ('Datetime','Date','Time','Timestamp')), d.columns[0])
                d['Datetime'] = pd.to_datetime(d[nc], unit='ms', utc=True)
            d = d.set_index('Datetime').sort_index()
            d = d[~d.index.duplicated(keep='first')]
            std_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
            lower_map = {k.lower(): v for k, v in std_map.items()}
            d.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in d.columns]
            d.index = d.index.tz_convert('America/Los_Angeles')
            return d
        except Exception:
            return None

    vix_df = _load('VIX'); spy_df = _load('SPY'); tlt_df = _load('TLT')
    spy_daily = spy_df['Close'].resample('D').last().dropna() if spy_df is not None else pd.Series(dtype=float)
    tlt_daily = tlt_df['Close'].resample('D').last().dropna() if tlt_df is not None else pd.Series(dtype=float)

    start = pd.Timestamp(eval_start, tz=PST) - timedelta(days=10)
    end = pd.Timestamp(eval_end, tz=PST) + timedelta(days=1)

    def _lk(d):
        if d is None: return {}
        sub = d[(d.index >= start) & (d.index <= end)]
        di = (sub.index.year*10000 + sub.index.month*100 + sub.index.day).astype(int)
        mi = (sub.index.hour*60 + sub.index.minute).astype(int)
        return {(int(di[i]), int(mi[i])): float(sub['Close'].values[i]) for i in range(len(sub))}

    vix_lk = _lk(vix_df); spy_lk = _lk(spy_df); tlt_lk = _lk(tlt_df)

    def _med(lk, d_int, em, w=30):
        vals = [lk[(d_int, m)] for m in range(em-w+1, em+1) if (d_int, m) in lk]
        return float(np.median(vals)) if vals else None

    def _pc(daily_s):
        m = {}
        for i in range(1, len(daily_s)):
            m[int(daily_s.index[i].strftime('%Y%m%d'))] = float(daily_s.iloc[i-1])
        return m

    spy_prev = _pc(spy_daily); tlt_prev = _pc(tlt_daily)
    regime_dict = {}
    for d in pd.date_range(start, end, freq='D', tz=PST):
        d_int = int(d.strftime('%Y%m%d'))
        for em in range(360, 391):
            v = _med(vix_lk, d_int, em) or 15.0
            sc = _med(spy_lk, d_int, em); sp = spy_prev.get(d_int)
            s_gap = ((sc - sp) / sp * 100) if sc and sp and sp != 0 else 0.0
            tc = _med(tlt_lk, d_int, em); tp_val = tlt_prev.get(d_int)
            t_gap = ((tc - tp_val) / tp_val * 100) if tc and tp_val and tp_val != 0 else 0.0
            regime_dict[(d_int, em)] = np.array([v, t_gap, s_gap])
    print(f"    Regime data: {len(regime_dict)} combos")
    return regime_dict


# ==============================================================================
# CSV SCORE LOADING
# ==============================================================================
def load_seed_scores_long(ticker, experiment_base):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    bucket = gcs_client.bucket(BUCKET_NAME)
    all_results = {}
    for results_dir in [RESULTS_DIR_V2, RESULTS_DIR]:
        prefix = f"{results_dir}/Universal_Exp{experiment_base}/{clean}/Seed_"
        blobs = list(bucket.list_blobs(prefix=prefix))
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


def load_seed_scores_short(ticker, experiment_base):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    bucket = gcs_client.bucket(BUCKET_NAME)
    prefix = f"{RESULTS_DIR_V2}/Universal_Exp{experiment_base}/{clean}/Seed_"
    blobs = list(bucket.list_blobs(prefix=prefix))
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


def compute_daily_prob_long(seed_scores, entry_min):
    n_seeds = 0
    avg_probs = None
    for seed_id, sdf in seed_scores.items():
        if 'Prob_Spike' not in sdf.columns:
            col = [c for c in sdf.columns if 'prob' in c.lower() or 'spike' in c.lower()]
            if not col:
                continue
            sdf = sdf.rename(columns={col[0]: 'Prob_Spike'})
        bar_mins = sdf.index.hour * 60 + sdf.index.minute
        entry_bars = sdf[bar_mins == entry_min]
        if entry_bars.empty:
            continue
        daily_prob = entry_bars.groupby(entry_bars.index.date)['Prob_Spike'].first()
        if avg_probs is None:
            avg_probs = daily_prob.copy()
        else:
            avg_probs = avg_probs.add(daily_prob, fill_value=0)
        n_seeds += 1
    if n_seeds > 0 and avg_probs is not None:
        return (avg_probs / n_seeds).to_dict()
    return {}


def compute_daily_prob_short(seed_scores, entry_min):
    all_daily = defaultdict(list)
    for seed_id, series in seed_scores.items():
        bar_mins = series.index.hour * 60 + series.index.minute
        entry_vals = series[bar_mins == entry_min]
        for dt, val in entry_vals.items():
            all_daily[dt.date()].append(float(val))
    return {d: np.mean(ps) for d, ps in all_daily.items()}


# ==============================================================================
# LONG TRADE SIMULATION — limit_entry mode
# ==============================================================================
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
            return (prices[k - 1] - ep) / ep, ed, True
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
    return (prices[n - 1] - ep) / ep, ed, True


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


def execute_short_trade(ohlcv, entry_idx, mode, tp_pct, sl_pct, sl_mult, grace_min):
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
        pnl -= 2 * SHORT_SLIPPAGE
    return pnl, td, filled


# ==============================================================================
# STATS COMPUTATION
# ==============================================================================
def compute_stats(trades_list, label, n_alloc=1):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0, 'sharpe': 0, 'per_mo': 0, 'n_tickers': 0,
                'n_alloc': n_alloc, 'port_return': 0, 'monthly': {},
                'trades_df': pd.DataFrame()}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    n_months = max(1, df['month'].nunique())
    n_pos = max(1, n_alloc)

    df_sorted = df.sort_values('date')
    port_pnl = df_sorted['pnl'] / n_pos
    port_cum = port_pnl.cumsum()
    port_dd = (port_cum - port_cum.cummax()).min()
    port_return = port_cum.iloc[-1] if len(port_cum) > 0 else 0

    daily_pnl = df.groupby('date')['pnl'].sum() / n_pos
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() > 0 else 0)
    return {
        'label': label,
        'n': len(df), 'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': port_dd,
        'sharpe': sharpe, 'per_mo': len(df) / n_months,
        'n_tickers': df['ticker'].nunique(),
        'n_alloc': n_pos, 'port_return': port_return,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
        'trades_df': df,
    }


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
    return 'other'


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print("=" * W)
print(f"  COMBINED LONG + SHORT PORTFOLIO BACKTEST (CSV-Scored)")
print(f"  LONG:  S3 deployed (limit_entry, BQ family agreement)")
print(f"  SHORT: FOLD_414 (RULE_1 tickers, best val_a, CSV-scored)")
print(f"  TEST:  {TEST_START} → {TEST_END}")
print(f"  CAPITAL: ${STARTING_CAPITAL:,}  |  EXIT: 1:00 PM PST")
print("=" * W)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LONG S3 — BQ Config Selection (family agreement gate)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  STEP 1: Loading LONG S3 configs from BQ...")

mv_str = ", ".join(f"'{m}'" for m in LONG_MODEL_VERSIONS)
q_long = f"""
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult, grace_minutes,
       dip_pct, max_wait_minutes, vol_filter_pct,
       vix_threshold, tlt_crash, spy_ignore,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, opt_avg_pnl, is_ensemble
FROM `ys-time-series.scoring_logs.wf_optimization_results`
WHERE (model_version IN ({mv_str})
       OR model_version LIKE 'robust_wf_v12_17%'
       OR model_version LIKE 'robust_wf_v12_18%'
       OR model_version LIKE 'robust_wf_v12_20%'
       OR model_version LIKE 'robust_wf_v12_22%'
       OR model_version LIKE 'robust_wf_v12_24%')
  AND survived_a = TRUE
  AND val_a_count >= 3
  AND val_a_avg_pnl > 0
"""
raw_long = bq_client.query(q_long).to_dataframe()
print(f"    {len(raw_long)} raw LONG BQ rows")

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
               'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
               'grace': '_GRACE', 'limit_entry': '_LIMIT'}

def _base_exp(exp_name):
    for sfx in sorted(MODE_SUFFIX.values(), key=len, reverse=True):
        if sfx and exp_name.endswith(sfx):
            return exp_name[:-len(sfx)]
    return exp_name

raw_long['ticker'] = raw_long['ticker'].replace(TICKER_FILE_MAP)
raw_long['base_experiment'] = raw_long['experiment'].apply(_base_exp)
raw_long = raw_long[raw_long['base_experiment'].isin(DEPLOY_EXPERIMENTS)]
raw_long = raw_long[~raw_long['ticker'].isin(EXCLUDE_TICKERS)]
print(f"    {len(raw_long)} after experiment + ticker filter")

raw_long = raw_long[raw_long['opt_avg_pnl'] > 0].copy()
raw_long = raw_long[raw_long['val_a_avg_pnl'] / raw_long['opt_avg_pnl'] <= MAX_VA_OPT]
print(f"    {len(raw_long)} after overfit filter")

indiv = raw_long[raw_long['is_ensemble'] == False].copy()
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
    mean_va_freq=('val_a_freq', 'mean'),
    vb_pnl=('val_b_avg_pnl', 'mean'),
).reset_index()
sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)

sa = sa[sa['n_seeds'] >= MIN_SEEDS].copy()
sa_limit = sa[sa['tpsl_mode'] == 'limit_entry'].copy()
print(f"    {len(sa_limit)} limit_entry seed groups (n_seeds >= {MIN_SEEDS})")

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
print(f"    S3 tickers: {', '.join(s3_tickers)}")

sa_s3 = sa_limit[sa_limit['ticker'].isin(s3_tickers)].copy()
sa_s3['score'] = (sa_s3['mean_va_pnl'] ** 2) * sa_s3['agree_18']
best_s3 = sa_s3.loc[sa_s3.groupby('ticker')['score'].idxmax()]
best_s3 = best_s3[best_s3['mean_va_pnl'] >= DEPLOY_THRESHOLD]

long_configs = {}
for _, row in best_s3.iterrows():
    tk = row['ticker']
    long_configs[tk] = {
        'prob_thresh': float(row['prob_thresh']),
        'tp': float(row['tp_pct']),
        'sl': float(row['sl_pct']),
        'avg_pnl': float(row['mean_va_pnl']),
        'experiment': row['experiment'],
        'dip_pct': float(row['dip_pct']),
        'max_wait_minutes': float(row['max_wait_minutes']),
        'entry_minute': int(row['entry_minute']) if pd.notna(row['entry_minute']) else ENTRY_MIN,
        'vol_filter_pct': int(_safe_float(row['vol_filter_pct'], 0)),
        'vix_threshold': _safe_float(row.get('vix_threshold'), S3_VIX_THRESHOLD),
        'tlt_crash': _safe_float(row.get('tlt_crash'), S3_TLT_CRASH),
        'spy_ignore': _safe_float(row.get('spy_ignore'), S3_SPY_IGNORE),
    }

long_tickers = sorted(long_configs.keys())
print(f"    LONG configs: {len(long_tickers)} tickers")
for tk in long_tickers:
    cfg = long_configs[tk]
    vfp = cfg['vol_filter_pct']
    print(f"      {tk}: prob>{cfg['prob_thresh']:.2f} TP={cfg['tp']:.0%} SL={cfg['sl']:.0%} "
          f"dip={cfg['dip_pct']:.1%} wait={cfg['max_wait_minutes']:.0f}m "
          f"vfp={vfp} vix<{cfg['vix_threshold']:.0f} "
          f"avgPnL={cfg['avg_pnl']:.2%} ({cfg['experiment']})")

del raw_long, indiv, sa, sa_limit, sa_fam, best_per_fam, sa_s3, best_s3
_release_memory()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: SHORT FOLD_414 — BQ Config Selection
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  STEP 2: Loading SHORT configs from BQ (fold {SHORT_FOLD} with fallback)...")

short_mv_str = ", ".join(f"'{m}'" for m in SHORT_MODEL_VERSIONS)
modes_str = ", ".join(f"'{m}'" for m in SHORT_MODES)
q_short = f"""
SELECT ticker, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, sl_mult, grace_minutes,
       opt_avg_pnl, val_a_avg_pnl, val_a_win_rate, val_a_count,
       is_ensemble, model_version
FROM `ys-time-series.scoring_logs.wf_optimization_results`
WHERE model_version IN ({short_mv_str})
  AND opt_avg_pnl > 0
  AND tpsl_mode IN ({modes_str})
  AND is_ensemble = TRUE
ORDER BY ticker, model_version
"""
bq_short = bq_client.query(q_short).to_dataframe()

for col in ['prob_thresh', 'tp_pct', 'sl_pct', 'sl_mult', 'grace_minutes',
            'entry_minute', 'opt_avg_pnl', 'val_a_avg_pnl', 'val_a_win_rate', 'val_a_count']:
    bq_short[col] = bq_short[col].apply(lambda x: _safe_float(x, 0.0))

print(f"    {len(bq_short)} total SHORT BQ rows")

bq_short['fold'] = bq_short['model_version'].str.extract(r'_(\d+)$').astype(float)
short_mask = bq_short['model_version'].str.contains('short')
a1_mask = bq_short['model_version'].str.contains('_a1')
bq_short.loc[short_mask & ~a1_mask & bq_short['fold'].isna(), 'fold'] = 400.0
bq_short.loc[a1_mask & bq_short['fold'].isna(), 'fold'] = 410.0

fold_df = bq_short[bq_short['fold'] == SHORT_FOLD]
print(f"    {len(fold_df)} rows for fold {SHORT_FOLD}")

short_configs = {}
for ticker in SHORT_TICKERS:
    t_df = fold_df[fold_df['ticker'] == ticker]
    if len(t_df) == 0:
        t_df = bq_short[bq_short['ticker'] == ticker]
        if len(t_df) > 0:
            best_fold = t_df.loc[t_df['val_a_avg_pnl'].idxmax(), 'fold']
            print(f"    {ticker}: no fold {SHORT_FOLD} config -- fallback to fold {int(best_fold)}")
    if len(t_df) == 0:
        print(f"    {ticker}: no BQ config in any fold")
        continue
    best_idx = t_df['val_a_avg_pnl'].idxmax()
    row = t_df.loc[best_idx]
    short_configs[ticker] = {
        'tpsl_mode': row['tpsl_mode'],
        'prob_thresh': float(row['prob_thresh']),
        'tp_pct': float(row['tp_pct']),
        'sl_pct': float(row['sl_pct']),
        'sl_mult': float(row['sl_mult']),
        'grace_minutes': int(row['grace_minutes']),
        'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else ENTRY_MIN,
        'val_a_avg_pnl': float(row['val_a_avg_pnl']),
    }

short_active = sorted(short_configs.keys())
print(f"    SHORT configs: {len(short_active)} tickers")
for tk in short_active:
    cfg = short_configs[tk]
    print(f"      {tk}: {cfg['tpsl_mode']} prob>{cfg['prob_thresh']:.2f} "
          f"TP={cfg['tp_pct']:.0%} SL={cfg['sl_pct']:.0%} "
          f"va={cfg['val_a_avg_pnl']:+.2%}")

del bq_short, fold_df
_release_memory()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Load OHLCV for all tickers
# ══════════════════════════════════════════════════════════════════════════════
all_tickers = sorted(set(long_tickers) | set(short_active))
print(f"\n  STEP 3: Loading OHLCV for {len(all_tickers)} tickers...")

ohlcv_arrays = {}
vol_baselines_cache = {}
for ticker in all_tickers:
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
            print(f"    {ticker}: no data in test period")
            del df; _master_cache.pop(ticker, None)
            continue
        if ticker in short_active:
            atr_pct = compute_daily_atr_pct(df)
            atr_vals = atr_pct.reindex(df.index).fillna(0.05)
            s = pd.to_datetime(TEST_START).tz_localize('America/Los_Angeles')
            e = pd.to_datetime(TEST_END).tz_localize('America/Los_Angeles') + timedelta(days=1)
            idx = df.index
            if str(idx.tz) != 'America/Los_Angeles':
                idx = idx.tz_convert('America/Los_Angeles')
            atr_cut = atr_vals[(idx >= s) & (idx < e)]
            ohlcv['atr_pct'] = atr_cut.values.astype(np.float64)
        if ticker in long_tickers and 'Volume' in df.columns:
            df_pst = df.copy()
            if str(df_pst.index.tz) != 'America/Los_Angeles':
                df_pst.index = df_pst.index.tz_convert('America/Los_Angeles')
            vol_baselines_cache[ticker] = precompute_vol_baselines(df_pst)
            del df_pst
        ohlcv_arrays[ticker] = ohlcv
        n_days = len(np.unique(ohlcv['dates']))
        side = 'L+S' if ticker in long_tickers and ticker in short_active else \
               ('L' if ticker in long_tickers else 'S')
        print(f"    {ticker} [{side}]: {n_days} trading days")
        del df; _master_cache.pop(ticker, None)
    except Exception as ex:
        print(f"    {ticker}: OHLCV load failed -- {ex}")
        _master_cache.pop(ticker, None)
    gc.collect()

_release_memory()

# ── Regime filter data ──
regime_dict = preload_regime_data(TEST_START, TEST_END)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Load CSV scores
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  STEP 4: Loading CSV scores...")

# ── LONG scores ──
long_daily_probs = {}
long_experiments = set(cfg['experiment'] for cfg in long_configs.values())
print(f"    LONG: {len(long_experiments)} experiments to load")

for ticker in long_tickers:
    if ticker not in ohlcv_arrays:
        continue
    cfg = long_configs[ticker]
    exp = cfg['experiment']
    exp_base = exp.replace('_LIMIT', '')
    seeds = load_seed_scores_long(ticker, exp_base)
    if not seeds:
        print(f"    {ticker}: no CSV scores for {exp}")
        continue
    entry_min = cfg.get('entry_minute', ENTRY_MIN)
    probs = compute_daily_prob_long(seeds, entry_min)
    if probs:
        long_daily_probs[ticker] = probs
        print(f"    LONG {ticker} ({exp}): {len(seeds)} seeds, {len(probs)} dates")
    del seeds; gc.collect()

# ── SHORT scores ──
short_daily_probs = {}
print(f"    SHORT: loading fold {SHORT_FOLD} scores")

for ticker in short_active:
    if ticker not in ohlcv_arrays:
        continue
    seeds = load_seed_scores_short(ticker, SHORT_EXP)
    if not seeds:
        print(f"    {ticker}: no CSV scores for {SHORT_EXP}")
        continue
    entry_min = short_configs[ticker].get('entry_minute', ENTRY_MIN)
    probs = compute_daily_prob_short(seeds, entry_min)
    if probs:
        short_daily_probs[ticker] = probs
        print(f"    SHORT {ticker}: {len(seeds)} seeds, {len(probs)} dates")
    del seeds; gc.collect()

_release_memory()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Pre-simulate LONG trades (limit_entry)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  STEP 5: Pre-simulating LONG trades (limit_entry)...")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()

long_daily_trades = {}
for ticker in long_tickers:
    cfg = long_configs[ticker]
    ohlcv = ohlcv_arrays.get(ticker)
    if ohlcv is None:
        continue
    tp, sl = cfg['tp'], cfg['sl']
    dip_pct = cfg['dip_pct']
    max_wait = cfg['max_wait_minutes']
    entry_min = cfg.get('entry_minute', ENTRY_MIN)
    outcomes = {}
    n_unfilled = 0
    unique_dates = np.unique(ohlcv['dates'])
    for d_int in unique_dates:
        d_str = str(int(d_int))
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < test_start_dt or d_date > test_end_dt:
            continue
        mask = (ohlcv['dates'] == d_int) & (ohlcv['times'] >= entry_min)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        entry_idx = idxs[0]
        if ohlcv['prices'][entry_idx] <= 0:
            continue
        pnl, _, valid = run_long_trade_limit_entry(
            ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
            ohlcv['dates'], ohlcv['times'],
            entry_idx, tp, sl, EXIT_TIME_MIN, dip_pct, max_wait)
        if valid:
            outcomes[d_date] = pnl - LONG_SLIPPAGE
        else:
            n_unfilled += 1
    if outcomes:
        long_daily_trades[ticker] = outcomes
    print(f"    {ticker}: {len(outcomes)} filled / {len(outcomes)+n_unfilled} days")

# ── Run LONG S3 backtest (with vol_filter + regime filter) ──
long_trades_list = []
n_vol_filtered = 0
n_regime_filtered = 0
for ticker in long_tickers:
    if ticker not in long_daily_probs or ticker not in long_daily_trades:
        continue
    cfg = long_configs[ticker]
    probs = long_daily_probs[ticker]
    outcomes = long_daily_trades[ticker]
    ohlcv = ohlcv_arrays.get(ticker)
    vbl = vol_baselines_cache.get(ticker, {})
    entry_min = cfg.get('entry_minute', ENTRY_MIN)
    vfp = cfg.get('vol_filter_pct', 0)
    vix_th = cfg.get('vix_threshold', 99)
    tlt_cr = cfg.get('tlt_crash', -99)
    spy_ig = cfg.get('spy_ignore', 99)

    for d_date, pnl in sorted(outcomes.items()):
        if d_date < test_start_dt or d_date > test_end_dt:
            continue
        prob = probs.get(d_date, 0.0)
        if prob < cfg['prob_thresh']:
            continue

        d_int = d_date.year * 10000 + d_date.month * 100 + d_date.day

        if vfp > 0 and ohlcv is not None:
            mask = (ohlcv['dates'] == d_int) & (ohlcv['times'] == entry_min)
            idxs = np.where(mask)[0]
            if idxs.size > 0:
                entry_vol = float(ohlcv['volumes'][idxs[0]])
                if not check_volume_filter(vbl, d_int, entry_min, entry_vol, vfp):
                    n_vol_filtered += 1
                    continue

        rk = (d_int, entry_min)
        if rk in regime_dict:
            rv = regime_dict[rk]
            if rv[0] > vix_th and rv[1] < tlt_cr and rv[2] > spy_ig:
                n_regime_filtered += 1
                continue

        long_trades_list.append({
            'date': d_date, 'ticker': ticker, 'pnl': pnl,
            'prob': prob, 'side': 'LONG',
        })

print(f"    LONG total: {len(long_trades_list)} trades across {len(long_tickers)} tickers")
print(f"    Filtered: {n_vol_filtered} vol, {n_regime_filtered} regime")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Run SHORT backtest
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  STEP 6: Running SHORT backtest (FOLD_414)...")

short_trades_list = []
for ticker in short_active:
    if ticker not in short_daily_probs or ticker not in ohlcv_arrays:
        continue
    cfg = short_configs[ticker]
    probs = short_daily_probs[ticker]
    ohlcv = ohlcv_arrays[ticker]
    entry_min = cfg.get('entry_minute', ENTRY_MIN)
    entry_indices = np.where(ohlcv['times'] == entry_min)[0]
    last_traded = -1

    for idx in entry_indices:
        d_int = int(ohlcv['dates'][idx])
        if d_int == last_traded:
            continue
        d_str = str(d_int)
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < test_start_dt or d_date > test_end_dt:
            last_traded = d_int
            continue
        prob = probs.get(d_date, 0.0)
        if prob < cfg['prob_thresh']:
            last_traded = d_int
            continue
        pnl, td, filled = execute_short_trade(
            ohlcv, idx, cfg['tpsl_mode'],
            cfg['tp_pct'], cfg['sl_pct'], cfg['sl_mult'],
            cfg['grace_minutes'])
        if filled:
            short_trades_list.append({
                'date': d_date, 'ticker': ticker, 'pnl': pnl,
                'prob': prob, 'side': 'SHORT',
            })
        last_traded = d_int

print(f"    SHORT total: {len(short_trades_list)} trades across "
      f"{len(set(t['ticker'] for t in short_trades_list))} tickers")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: COMBINE & ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  STEP 7: Computing stats and clash analysis...")

combined_trades = long_trades_list + short_trades_list

stats_long = compute_stats(long_trades_list, 'LONG (S3)', len(long_tickers))
stats_short = compute_stats(short_trades_list, 'SHORT (414)', len(short_active))
stats_combined = compute_stats(combined_trades, 'COMBINED', len(long_tickers) + len(short_active))

# ── Clash analysis ──
long_dates = set(t['date'] for t in long_trades_list)
short_dates = set(t['date'] for t in short_trades_list)
clash_dates = sorted(long_dates & short_dates)
all_trade_dates = sorted(long_dates | short_dates)

long_by_date = defaultdict(list)
for t in long_trades_list:
    long_by_date[t['date']].append(t)
short_by_date = defaultdict(list)
for t in short_trades_list:
    short_by_date[t['date']].append(t)

long_only_dates = sorted(long_dates - short_dates)
short_only_dates = sorted(short_dates - long_dates)

overlap_tickers = set(long_tickers) & set(short_active)
same_ticker_clashes = []
for d in clash_dates:
    lt = {t['ticker'] for t in long_by_date[d]}
    st = {t['ticker'] for t in short_by_date[d]}
    overlap = lt & st
    if overlap:
        same_ticker_clashes.append((d, overlap))


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * W)
print(f"  RESULTS — LONG + SHORT COMBINED PORTFOLIO")
print("=" * W)


# ── TABLE 1: Side-by-side comparison ──
print(f"\n  TABLE 1: PORTFOLIO COMPARISON")
print(f"\n  {'Side':<15} {'#Tk':>3} {'Trd':>5} {'AvgPnL':>8} {'PortRet':>8} "
      f"{'WR':>5} {'MaxDD':>8} {'Sharpe':>7} {'T/mo':>5} {'RawSum':>8}")
print(f"  {'─'*15} {'─'*3} {'─'*5} {'─'*8} {'─'*8} "
      f"{'─'*5} {'─'*8} {'─'*7} {'─'*5} {'─'*8}")

for label, st in [('LONG (S3)', stats_long),
                   ('SHORT (414)', stats_short),
                   ('COMBINED', stats_combined)]:
    if st['n'] == 0:
        print(f"  {label:<15} — no trades")
        continue
    print(f"  {label:<15} {st['n_tickers']:>3} {st['n']:>5} {st['avg']:>+7.2%} "
          f"{st['port_return']:>+7.1%} {st['wr']:>4.0%} {st['dd']:>+7.1%} "
          f"{st['sharpe']:>6.2f} {st['per_mo']:>5.1f} {st['total']:>+7.1%}")


# ── TABLE 2: Monthly breakdown ──
months = sorted(set(list(stats_long.get('monthly', {}).keys()) +
                     list(stats_short.get('monthly', {}).keys()) +
                     list(stats_combined.get('monthly', {}).keys())))

if months:
    print(f"\n  TABLE 2: MONTHLY BREAKDOWN")
    print(f"\n  {'Month':<8} | {'──── LONG (S3) ─────────':} | "
          f"{'──── SHORT (414) ───────':} | {'──── COMBINED ─────────':}")
    print(f"  {'':8} | {'Trd':>3} {'AvgPnL':>7} {'WR':>5} {'TotPnL':>8} | "
          f"{'Trd':>3} {'AvgPnL':>7} {'WR':>5} {'TotPnL':>8} | "
          f"{'Trd':>3} {'AvgPnL':>7} {'WR':>5} {'TotPnL':>8}")
    print(f"  {'─'*90}")

    for month in months:
        row = f"  {month:<8} |"
        for st in [stats_long, stats_short, stats_combined]:
            m = st.get('monthly', {}).get(month)
            if m:
                row += f" {m['n']:>3} {m['avg']:>+6.2%} {m['wr']:>4.0%} {m['total']:>+7.1%} |"
            else:
                row += f" {'—':>3} {'—':>7} {'—':>5} {'—':>8} |"
        print(row)


# ── TABLE 3: Per-ticker breakdown ──
print(f"\n  TABLE 3: PER-TICKER BREAKDOWN")
print(f"\n  {'Ticker':<7} {'Side':<6} | {'Trades':>6} {'AvgPnL':>8} {'TotPnL':>8} {'WR':>5}")
print(f"  {'─'*50}")

for ticker in sorted(all_tickers):
    for side, trades_list in [('LONG', long_trades_list), ('SHORT', short_trades_list)]:
        tt = [t for t in trades_list if t['ticker'] == ticker]
        if not tt:
            continue
        avg = np.mean([t['pnl'] for t in tt])
        total = sum(t['pnl'] for t in tt)
        wr = np.mean([t['pnl'] > 0 for t in tt])
        print(f"  {ticker:<7} {side:<6} | {len(tt):>6} {avg:>+7.2%} {total:>+7.1%} {wr:>4.0%}")


# ── TABLE 4: Clash Analysis ──
print(f"\n  TABLE 4: CLASH ANALYSIS")
print(f"\n  Total trading days: {len(all_trade_dates)}")
print(f"  LONG-only days:    {len(long_only_dates)}")
print(f"  SHORT-only days:   {len(short_only_dates)}")
print(f"  Clash days (both): {len(clash_dates)} ({len(clash_dates)/max(1,len(all_trade_dates)):.0%})")
print(f"  Ticker overlap:    {overlap_tickers if overlap_tickers else 'NONE'}")
print(f"  Same-ticker clash: {len(same_ticker_clashes)} days")

if clash_dates:
    clash_long_pnl = [np.mean([t['pnl'] for t in long_by_date[d]]) for d in clash_dates]
    clash_short_pnl = [np.mean([t['pnl'] for t in short_by_date[d]]) for d in clash_dates]
    clash_net = [np.mean(clash_long_pnl), np.mean(clash_short_pnl)]

    noclash_long_pnl = [np.mean([t['pnl'] for t in long_by_date[d]])
                         for d in long_only_dates if d in long_by_date]
    noclash_short_pnl = [np.mean([t['pnl'] for t in short_by_date[d]])
                          for d in short_only_dates if d in short_by_date]

    print(f"\n  PnL on clash days:")
    print(f"    LONG  avg on clash days:    {np.mean(clash_long_pnl):>+.2%}")
    print(f"    SHORT avg on clash days:    {np.mean(clash_short_pnl):>+.2%}")
    print(f"    NET   avg on clash days:    {np.mean(clash_long_pnl) + np.mean(clash_short_pnl):>+.2%}")

    if noclash_long_pnl:
        print(f"    LONG  avg on non-clash:     {np.mean(noclash_long_pnl):>+.2%}")
    if noclash_short_pnl:
        print(f"    SHORT avg on non-clash:     {np.mean(noclash_short_pnl):>+.2%}")

    both_dates = sorted(long_dates & short_dates)
    daily_long_pnl_series = pd.Series(
        {d: sum(t['pnl'] for t in long_by_date[d]) for d in both_dates})
    daily_short_pnl_series = pd.Series(
        {d: sum(t['pnl'] for t in short_by_date[d]) for d in both_dates})
    if len(daily_long_pnl_series) >= 3 and len(daily_short_pnl_series) >= 3:
        corr = daily_long_pnl_series.corr(daily_short_pnl_series)
        print(f"\n  Daily PnL correlation (clash days): r = {corr:+.3f}")
        if corr < -0.2:
            print(f"    → Negative correlation: SHORT acts as hedge for LONG ✓")
        elif corr > 0.2:
            print(f"    → Positive correlation: strategies move together (amplify risk)")
        else:
            print(f"    → Low correlation: strategies are independent")

    print(f"\n  Clash day detail (first 20):")
    print(f"  {'Date':>12}  {'LONG':>8} {'#L':>3}  {'SHORT':>8} {'#S':>3}  {'NET':>8}")
    print(f"  {'─'*50}")
    for d in clash_dates[:20]:
        l_pnl = sum(t['pnl'] for t in long_by_date[d])
        l_n = len(long_by_date[d])
        s_pnl = sum(t['pnl'] for t in short_by_date[d])
        s_n = len(short_by_date[d])
        net = l_pnl + s_pnl
        flag = ' !' if (l_pnl > 0 and s_pnl > 0) else (' ~' if (l_pnl < 0 and s_pnl < 0) else '  ')
        print(f"  {str(d):>12}  {l_pnl:>+7.2%} {l_n:>3}  {s_pnl:>+7.2%} {s_n:>3}  {net:>+7.2%}{flag}")

if same_ticker_clashes:
    print(f"\n  SAME-TICKER CLASHES (LONG + SHORT on same ticker same day):")
    for d, tks in same_ticker_clashes[:10]:
        for tk in tks:
            l_t = [t for t in long_by_date[d] if t['ticker'] == tk]
            s_t = [t for t in short_by_date[d] if t['ticker'] == tk]
            l_pnl = l_t[0]['pnl'] if l_t else 0
            s_pnl = s_t[0]['pnl'] if s_t else 0
            print(f"    {d} {tk}: LONG={l_pnl:+.2%}, SHORT={s_pnl:+.2%}, NET={l_pnl+s_pnl:+.2%}")


# ── TABLE 5: Portfolio Simulation ──
print(f"\n  TABLE 5: PORTFOLIO SIMULATION (${STARTING_CAPITAL:,})")

equity_curves = {}
for label, trades, n_tk in [('LONG', long_trades_list, len(long_tickers)),
                              ('SHORT', short_trades_list, len(short_active)),
                              ('COMBINED', combined_trades, len(long_tickers) + len(short_active))]:
    if not trades:
        print(f"    {label}: no trades")
        continue
    df_t = pd.DataFrame(trades).sort_values('date')
    equity = STARTING_CAPITAL
    curve = [{'date': pd.Timestamp(TEST_START).date(), 'equity': equity}]
    peak = equity
    max_dd_pct = 0
    max_dd_dollar = 0

    for _, row in df_t.iterrows():
        alloc = equity / max(1, n_tk)
        pnl_dollar = alloc * row['pnl']
        equity += pnl_dollar
        curve.append({'date': row['date'], 'equity': equity})
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        if dd < max_dd_pct:
            max_dd_pct = dd
            max_dd_dollar = equity - peak

    total_return = (equity - STARTING_CAPITAL) / STARTING_CAPITAL
    daily_pnl = df_t.groupby('date')['pnl'].sum()
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

    print(f"\n  {label}:")
    print(f"    Final equity: ${equity:,.0f} ({total_return:+.1%})")
    print(f"    Max drawdown: ${max_dd_dollar:,.0f} ({max_dd_pct:+.1%})")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    Trades: {len(df_t)}, Trades/mo: {len(df_t)/max(1,df_t['date'].apply(lambda x: x.strftime('%Y-%m')).nunique()):.1f}")

    equity_curves[label] = pd.DataFrame(curve)


# ── TABLE 6: VERDICT ──
print(f"\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

if stats_long['n'] > 0 and stats_short['n'] > 0 and stats_combined['n'] > 0:
    delta_avg = stats_combined['avg'] - stats_long['avg']
    delta_port = stats_combined['port_return'] - stats_long['port_return']
    delta_dd = stats_combined['dd'] - stats_long['dd']
    delta_sharpe = stats_combined['sharpe'] - stats_long['sharpe']

    print(f"\n  COMBINED vs LONG-only:")
    print(f"    ΔAvgPnL:   {delta_avg:+.2%} ({'better' if delta_avg > 0 else 'worse'})")
    print(f"    ΔPortRet:  {delta_port:+.1%} ({'better' if delta_port > 0 else 'worse'})")
    print(f"    ΔMaxDD:    {delta_dd:+.1%} ({'deeper' if delta_dd < 0 else 'shallower'})")
    print(f"    ΔSharpe:   {delta_sharpe:+.2f} ({'better' if delta_sharpe > 0 else 'worse'})")

    improvements = sum([
        delta_port > 0,
        abs(stats_combined['dd']) < abs(stats_long['dd']),
        delta_sharpe > 0,
    ])

    if improvements >= 2:
        print(f"\n  → SHORT IMPROVES the portfolio ({improvements}/3 metrics better)")
        print(f"    Recommendation: DEPLOY SHORT as LONG hedge")
    elif improvements == 1:
        print(f"\n  → MIXED results — SHORT has marginal impact ({improvements}/3)")
        print(f"    Recommendation: Deploy SHORT cautiously, monitor interaction")
    else:
        print(f"\n  → SHORT HURTS the portfolio ({improvements}/3)")
        print(f"    Recommendation: Re-evaluate SHORT ticker selection or sizing")

    if len(same_ticker_clashes) > 0:
        print(f"\n  PRIORITIZATION NEEDED: {len(same_ticker_clashes)} same-ticker clashes detected")
        print(f"    These tickers appear in both LONG and SHORT: {overlap_tickers}")
        print(f"    Consider: exclude overlapping tickers from SHORT, or use time-of-day priority")
    else:
        print(f"\n  NO PRIORITIZATION NEEDED: ticker sets are disjoint")
        print(f"    LONG and SHORT can run independently without conflict")


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS (3x2 layout)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  Generating charts...")

fig, axes = plt.subplots(3, 2, figsize=(20, 18))
fig.suptitle('COMBINED LONG + SHORT PORTFOLIO BACKTEST', fontsize=16, fontweight='bold', y=0.98)


# ── Chart 1: Equity curves ──
ax1 = axes[0, 0]
_st_map = {'LONG': stats_long, 'SHORT': stats_short, 'COMBINED': stats_combined}
for label in ['LONG', 'SHORT', 'COMBINED']:
    if label not in equity_curves:
        continue
    curve = equity_curves[label]
    st = _st_map[label]
    dates = pd.to_datetime(curve['date'])
    equity = curve['equity'] / 1000
    final_ret = (curve['equity'].iloc[-1] / STARTING_CAPITAL - 1) * 100
    ax1.plot(dates, equity,
             label=f'{label} ({st["n"]}t, avg {st["avg"]:+.2%}, ret {final_ret:+.0f}%)',
             color=SIDE_COLORS[label],
             linewidth=2.5 if label == 'COMBINED' else 1.8,
             linestyle='-' if label == 'COMBINED' else '--')

ax1.axhline(y=STARTING_CAPITAL/1000, color='gray', linestyle='--', alpha=0.3)
ax1.axvline(x=pd.Timestamp(HALF2_START), color='red', linestyle=':', alpha=0.4)
ax1.text(pd.Timestamp(HALF2_START) + pd.Timedelta(days=1), ax1.get_ylim()[0],
         'H2', color='red', fontsize=8, alpha=0.6)
ax1.set_ylabel('Equity ($K)', fontsize=11)
ax1.set_title('Equity Curves ($200K start)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


# ── Chart 2: Cumulative Portfolio Return ──
ax2 = axes[0, 1]
n_tk_map = {'LONG': len(long_tickers), 'SHORT': len(short_active),
            'COMBINED': len(long_tickers) + len(short_active)}
for label, tl in [('LONG', long_trades_list), ('SHORT', short_trades_list),
                   ('COMBINED', combined_trades)]:
    if not tl:
        continue
    df_t = pd.DataFrame(tl).sort_values('date')
    n_tk = max(1, n_tk_map[label])
    cum = (df_t['pnl'] / n_tk).cumsum() * 100
    dates = pd.to_datetime(df_t['date'].values)
    final_ret = cum.iloc[-1]
    ax2.plot(dates, cum.values,
             label=f'{label} ({df_t.shape[0]}t, {final_ret:+.1f}%)',
             color=SIDE_COLORS[label],
             linewidth=2.5 if label == 'COMBINED' else 1.8,
             linestyle='-' if label == 'COMBINED' else '--')

ax2.axhline(y=0, color='gray', linewidth=0.5)
ax2.axvline(x=pd.Timestamp(HALF2_START), color='red', linestyle=':', alpha=0.4)
ax2.set_ylabel('Portfolio Return (%)', fontsize=11)
ax2.set_title('Cumulative Portfolio Return (position-weighted)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


# ── Chart 3: Drawdown comparison ──
ax3 = axes[1, 0]
for label, tl in [('LONG', long_trades_list), ('SHORT', short_trades_list),
                   ('COMBINED', combined_trades)]:
    if not tl:
        continue
    df_t = pd.DataFrame(tl).sort_values('date')
    n_tk = max(1, n_tk_map[label])
    cum = (df_t['pnl'] / n_tk).cumsum()
    dd = (cum - cum.cummax()) * 100
    dates = pd.to_datetime(df_t['date'].values)
    ax3.fill_between(dates, dd.values, 0, alpha=0.12, color=SIDE_COLORS[label])
    ax3.plot(dates, dd.values, label=label, color=SIDE_COLORS[label],
             linewidth=1.5, linestyle='-' if label == 'COMBINED' else '--')

ax3.axvline(x=pd.Timestamp(HALF2_START), color='red', linestyle=':', alpha=0.4)
ax3.set_ylabel('Drawdown (%)', fontsize=11)
ax3.set_title('Drawdown Comparison', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10, framealpha=0.9, loc='lower left')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


# ── Chart 4: Daily PnL scatter (LONG vs SHORT on clash days) ──
ax4 = axes[1, 1]
if clash_dates:
    l_pnls = [sum(t['pnl'] for t in long_by_date[d]) * 100 for d in clash_dates]
    s_pnls = [sum(t['pnl'] for t in short_by_date[d]) * 100 for d in clash_dates]
    ax4.scatter(l_pnls, s_pnls, c=SIDE_COLORS['COMBINED'], s=40, alpha=0.7, edgecolors='white')
    ax4.axhline(y=0, color='gray', linewidth=0.5)
    ax4.axvline(x=0, color='gray', linewidth=0.5)

    max_abs = max(max(abs(v) for v in l_pnls + s_pnls), 1)
    ax4.plot([-max_abs, max_abs], [max_abs, -max_abs], 'r--', alpha=0.2, label='perfect hedge')

    q1 = sum(1 for l, s in zip(l_pnls, s_pnls) if l > 0 and s > 0)
    q2 = sum(1 for l, s in zip(l_pnls, s_pnls) if l < 0 and s > 0)
    q3 = sum(1 for l, s in zip(l_pnls, s_pnls) if l < 0 and s < 0)
    q4 = sum(1 for l, s in zip(l_pnls, s_pnls) if l > 0 and s < 0)
    ax4.text(0.95, 0.95, f'L+S+ {q1}', transform=ax4.transAxes, ha='right', va='top', fontsize=8, color='green')
    ax4.text(0.05, 0.95, f'L-S+ {q2}', transform=ax4.transAxes, ha='left', va='top', fontsize=8, color='blue')
    ax4.text(0.05, 0.05, f'L-S- {q3}', transform=ax4.transAxes, ha='left', va='bottom', fontsize=8, color='red')
    ax4.text(0.95, 0.05, f'L+S- {q4}', transform=ax4.transAxes, ha='right', va='bottom', fontsize=8, color='orange')

ax4.set_xlabel('LONG daily PnL (%)', fontsize=11)
ax4.set_ylabel('SHORT daily PnL (%)', fontsize=11)
ax4.set_title(f'Daily PnL: LONG vs SHORT (clash days, n={len(clash_dates)})', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9, framealpha=0.9)
ax4.grid(True, alpha=0.3)


# ── Chart 5: Monthly performance bars ──
ax5 = axes[2, 0]
if months:
    x = np.arange(len(months))
    bar_width = 0.25
    month_labels = [m[-2:] for m in months]

    for i, (label, st) in enumerate([('LONG', stats_long), ('SHORT', stats_short),
                                      ('COMBINED', stats_combined)]):
        vals = []
        for m in months:
            ms = st.get('monthly', {}).get(m)
            vals.append(ms['avg'] * 100 if ms else 0)
        bars = ax5.bar(x + i * bar_width, vals, bar_width,
                       label=label, color=SIDE_COLORS[label], alpha=0.85)
        for bar, val in zip(bars, vals):
            if abs(val) > 0.01:
                ax5.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + (0.05 if val >= 0 else -0.15),
                         f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax5.set_xticks(x + bar_width)
    ax5.set_xticklabels(month_labels, fontsize=10)
    ax5.set_ylabel('Avg PnL per Trade (%)', fontsize=11)
    ax5.set_title('Monthly Average PnL', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10, framealpha=0.9)
    ax5.axhline(y=0, color='gray', linewidth=0.5)
    ax5.grid(True, alpha=0.3, axis='y')


# ── Chart 6: Summary metrics comparison ──
ax6 = axes[2, 1]
metric_names = ['Port Ret\n(%)', 'Avg PnL\n(%)', 'Win Rate\n(%)', 'Sharpe']
sides = ['LONG', 'SHORT', 'COMBINED']
metric_vals = {
    'LONG': [stats_long['port_return']*100, stats_long['avg']*100, stats_long['wr']*100, stats_long['sharpe']],
    'SHORT': [stats_short['port_return']*100, stats_short['avg']*100, stats_short['wr']*100, stats_short['sharpe']],
    'COMBINED': [stats_combined['port_return']*100, stats_combined['avg']*100, stats_combined['wr']*100, stats_combined['sharpe']],
}

x = np.arange(len(metric_names))
bar_width = 0.25
for i, side in enumerate(sides):
    vals = metric_vals[side]
    bars = ax6.bar(x + i * bar_width, vals, bar_width,
                   label=side, color=SIDE_COLORS[side], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.3 if val >= 0 else -1.0),
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)

ax6.set_xticks(x + bar_width)
ax6.set_xticklabels(metric_names, fontsize=10)
ax6.set_title('Key Metrics Comparison', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10, framealpha=0.9)
ax6.axhline(y=0, color='gray', linewidth=0.5)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/tmp/backtest_combined_portfolio.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n  Charts saved to /tmp/backtest_combined_portfolio.png")


# ── Final summary ──
print()
print("=" * W)
n_long = len(long_tickers)
n_short = len(short_active)
print(f"  Analysis complete: {n_long} LONG tickers (S3) + {n_short} SHORT tickers (FOLD_414)")
print(f"  {len(long_trades_list)} LONG trades + {len(short_trades_list)} SHORT trades = "
      f"{len(combined_trades)} combined")
print(f"  {len(clash_dates)} clash days, {len(same_ticker_clashes)} same-ticker clashes")
print("=" * W)
