# %% ── SHORT All-Model Comparison — 400/410/425/435 ─────────────────────────
#
# Pulls WF optimization results from BQ for ALL SHORT model families:
#   Phase A (400-405): 8% dump, 9:30 entry (min 375)
#   Phase B (410-415): 8% dump, 9:30 entry (min 375)
#   10AM v1 (425_10AM): 2.5% dump, 10:00 entry (min 405)
#   10AM v2 (435_10AM): 8% dump, 10:00 entry (min 405)  [when available]
#
# The 9:30 and 10AM models run at DIFFERENT times of day — they are ADDITIVE,
# not competing. A ticker can fire a 9:30 SHORT and a 10AM SHORT on the same day.
# This script evaluates each family standalone AND the combined portfolio.
#
# CSV-scored, GPU NOT required. Colab-ready.

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
TEST_END    = '2026-07-31'
HALF2_START = '2026-06-01'

SLIPPAGE_PCT  = 0.002
DEPLOY_FLOOR  = 0.010
SL_FIXED      = 0.03
MIN_H2_TRADES = 3

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

EXIT_TIME_MIN = 780

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_ALLOWED_MODES = ('static', 'atr_sl', 'grace')

# ── Model families ───────────────────────────────────────────────────────────
FOLDS_A = list(range(400, 406))     # Phase A: 8% dump, 9:30 entry
FOLDS_B = list(range(410, 416))     # Phase B: 8% dump, 9:30 entry
FOLDS_10AM_V1 = [425]               # 10AM v1: 2.5% dump, 10:00 entry
FOLDS_10AM_V2 = [435]               # 10AM v2: 8% dump, 10:00 entry (pending)

FAMILY_CONFIG = {
    'PHASE_A': {
        'folds': FOLDS_A,
        'entry_min': 375,
        'mv_pattern': 'robust_wf_v12_short_{fold}',
        'exp_pattern': '{fold}_XSECTOR',
        'color': '#1565C0',
        'label': '400-405 (8% dump, 9:30)',
    },
    'PHASE_B': {
        'folds': FOLDS_B,
        'entry_min': 375,
        'mv_pattern': 'robust_wf_v12_short_{fold}',
        'exp_pattern': '{fold}_XSECTOR',
        'color': '#E65100',
        'label': '410-415 (8% dump, 9:30)',
    },
    '10AM_V1': {
        'folds': FOLDS_10AM_V1,
        'entry_min': 405,
        'mv_pattern': 'robust_wf_v12_short_{fold}_10am',
        'exp_pattern': '{fold}_10AM',
        'color': '#2E7D32',
        'label': '425 (2.5% dump, 10:00)',
    },
    '10AM_V2': {
        'folds': FOLDS_10AM_V2,
        'entry_min': 405,
        'mv_pattern': 'robust_wf_v12_short_{fold}_10am',
        'exp_pattern': '{fold}_10AM',
        'color': '#9C27B0',
        'label': '435 (8% dump, 10:00)',
    },
}

# Build lookup tables
ALL_FOLDS = []
FOLD_TO_FAMILY = {}
FOLD_MODEL_VERSIONS = {}
FOLD_EXPERIMENTS = {}
FOLD_ENTRY_MIN = {}

for fam_name, fam in FAMILY_CONFIG.items():
    for fold in fam['folds']:
        ALL_FOLDS.append(fold)
        FOLD_TO_FAMILY[fold] = fam_name
        FOLD_MODEL_VERSIONS[fold] = fam['mv_pattern'].format(fold=fold)
        FOLD_EXPERIMENTS[fold] = fam['exp_pattern'].format(fold=fold)
        FOLD_ENTRY_MIN[fold] = fam['entry_min']

# ── Tickers to evaluate: union of all SHORT candidates ──────────────────────
RULE_1 = ['BE', 'IREN', 'OKLO', 'ONDS', 'RKLB', 'WOLF']

EXPANDED_SHORT = sorted(set(RULE_1 + [
    'ACMR', 'AVAV', 'BEAM', 'BKSY', 'BTBT', 'CCJ', 'CEG', 'CIFR', 'CLSK',
    'COIN', 'COP', 'CORZ', 'FCEL', 'FLNC', 'GE', 'GEV', 'HOOD',
    'HUT', 'JOBY', 'KTOS', 'LUNR', 'MARA', 'MNTS', 'MOD', 'MP',
    'MSTR', 'MU', 'MVIS', 'MXL', 'NNE', 'RIOT', 'RCAT', 'RIVN',
    'SMR', 'SOUN', 'VOYG', 'WULF',
]))

ALL_TICKERS = EXPANDED_SHORT

# ── Strategy definitions ─────────────────────────────────────────────────────
# Each strategy picks configs differently from BQ results.
# BEST_PER_FAMILY: for each family, pick best val_a per ticker (non-oracle)
# FOLD_403: single fold from Phase A
# FOLD_414: single fold from Phase B
# FOLD_425: single fold from 10AM v1
# CROSS_FAMILY_BEST: pick best val_a across ALL families (the key question)
# ORACLE: best actual H2 config (biased ceiling)

STRATEGY_NAMES = ['FOLD_403', 'FOLD_414', 'FOLD_425', 'FOLD_435',
                  'BEST_930', 'BEST_10AM',
                  'ADDITIVE_403', 'ADDITIVE_414', 'ADDITIVE_BEST',
                  'ORACLE']

STRATEGY_COLORS = {
    'FOLD_403':      '#1565C0',
    'FOLD_414':      '#E65100',
    'FOLD_425':      '#2E7D32',
    'FOLD_435':      '#9C27B0',
    'BEST_930':      '#00838F',
    'BEST_10AM':     '#558B2F',
    'ADDITIVE_403':  '#D81B60',
    'ADDITIVE_414':  '#FF6F00',
    'ADDITIVE_BEST': '#4A148C',
    'ORACLE':        '#9E9E9E',
}

STRATEGY_LABELS = {
    'FOLD_403':      'F403 alone (9:30)',
    'FOLD_414':      'F414 alone (9:30)',
    'FOLD_425':      'F425 alone (10AM, 2.5%)',
    'FOLD_435':      'F435 alone (10AM, 8%)',
    'BEST_930':      'Best val_a (any 9:30)',
    'BEST_10AM':     'Best val_a (any 10AM)',
    'ADDITIVE_403':  'F403 + best 10AM',
    'ADDITIVE_414':  'F414 + best 10AM',
    'ADDITIVE_BEST': 'Best 9:30 + Best 10AM',
    'ORACLE':        'Oracle (biased)',
}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120


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
        in_grace = (time_mins[k] - et) <= grace_minutes
        tp_hit = lows[k] <= target_price
        sl_hit = highs[k] >= stop_price
        if not in_grace:
            if sl_hit and tp_hit:
                return (tp if np.random.random() < 0.5 else -sl), ed, True
            if sl_hit: return -sl, ed, True
        if tp_hit: return tp, ed, True
        if time_mins[k] >= exit_time:
            return (entry_price - prices[k]) / entry_price, ed, True
    return (entry_price - prices[n-1]) / entry_price, ed, True


def execute_trade(ohlcv, entry_idx, mode, tp_pct, sl_pct, sl_mult, grace_min):
    prices = ohlcv['prices']
    highs  = ohlcv['highs']
    lows   = ohlcv['lows']
    dates  = ohlcv['dates']
    times  = ohlcv['times']

    if mode == 'atr_sl':
        atr_val = ohlcv['atr_pct'][entry_idx] if 'atr_pct' in ohlcv else 0.05
        return run_short_trade_atr_sl(prices, highs, lows, dates, times,
                                      entry_idx, tp_pct, atr_val, sl_mult, EXIT_TIME_MIN)
    elif mode == 'grace':
        return run_short_trade_grace(prices, highs, lows, dates, times,
                                     entry_idx, tp_pct, sl_pct, EXIT_TIME_MIN,
                                     int(grace_min))
    else:
        return run_short_trade_static(prices, highs, lows, dates, times,
                                      entry_idx, tp_pct, sl_pct, EXIT_TIME_MIN)


# ==============================================================================
# CONFIG SELECTION METHODS
# ==============================================================================
def _bq_row_to_cfg(row):
    return {
        'fold': int(row['fold']),
        'family': FOLD_TO_FAMILY.get(int(row['fold']), 'UNKNOWN'),
        'tpsl_mode': row['tpsl_mode'],
        'prob_thresh': row['prob_thresh'],
        'tp_pct': row['tp_pct'],
        'sl_pct': row['sl_pct'],
        'sl_mult': row['sl_mult'],
        'grace_minutes': int(row['grace_minutes']),
        'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0
                        else FOLD_ENTRY_MIN.get(int(row['fold']), 375),
        'val_a_avg_pnl': row['val_a_avg_pnl'],
        'opt_avg_pnl': row['opt_avg_pnl'],
        'config_id': f"f{int(row['fold'])}_{row['tpsl_mode']}_t{row['prob_thresh']:.2f}_tp{row['tp_pct']:.2f}",
    }


def _pick_best(t_df, col='val_a_avg_pnl'):
    if len(t_df) == 0:
        return None
    best_idx = t_df[col].idxmax()
    return _bq_row_to_cfg(t_df.loc[best_idx])


def select_single_fold(bq_df, tickers, fold):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'] == fold)]
        cfg = _pick_best(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


def select_cross_family_best(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[bq_df['ticker'] == ticker]
        cfg = _pick_best(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


def select_best_by_entry_time(bq_df, tickers, entry_min):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker)]
        fold_mask = t_df['fold'].map(lambda f: FOLD_ENTRY_MIN.get(f, -1) == entry_min)
        t_df = t_df[fold_mask]
        cfg = _pick_best(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ==============================================================================
# BACKTEST + SUMMARY
# ==============================================================================
def run_csv_backtest(configs, daily_probs_cache, ticker_data, label):
    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt   = pd.Timestamp(TEST_END).date()
    h2_start_dt   = pd.Timestamp(HALF2_START).date()

    all_trades = []
    per_ticker_trades = defaultdict(list)

    for ticker in sorted(configs.keys()):
        if ticker not in ticker_data or ticker not in daily_probs_cache:
            continue
        cfg = configs[ticker]
        ohlcv = ticker_data[ticker]
        probs = daily_probs_cache[ticker]
        entry_min = cfg.get('entry_minute', 375)
        prob_thresh = cfg['prob_thresh']
        mode = cfg['tpsl_mode']

        prices = ohlcv['prices']
        dates  = ohlcv['dates']
        times  = ohlcv['times']

        entry_indices = np.where(times == entry_min)[0]
        last_traded_date = -1

        for idx in entry_indices:
            d_int = int(dates[idx])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()

            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            if d_int == last_traded_date:
                continue
            prob = probs.get(d_date)
            if prob is None:
                continue
            if prob < prob_thresh:
                continue

            pnl, trade_date, filled = execute_trade(
                ohlcv, idx, mode, cfg['tp_pct'], cfg['sl_pct'],
                cfg['sl_mult'], cfg['grace_minutes'])

            if not filled:
                continue
            last_traded_date = d_int

            half = 'H2' if d_date >= h2_start_dt else 'H1'
            trade = {
                'date': d_date, 'ticker': ticker, 'pnl': pnl,
                'prob': prob, 'close': float(prices[idx]),
                'mode': mode, 'tp': cfg['tp_pct'],
                'sl': cfg['sl_pct'] if mode != 'atr_sl' else cfg['sl_mult'],
                'half': half, 'family': cfg.get('family', '?'),
                'fold': cfg.get('fold', 0),
            }
            all_trades.append(trade)
            per_ticker_trades[ticker].append(trade)

    return all_trades, per_ticker_trades


def summarize_trades(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0, 'monthly': {}, 'trades_df': pd.DataFrame(),
                'h1_n': 0, 'h1_avg': 0, 'h1_total': 0,
                'h2_n': 0, 'h2_avg': 0, 'h2_total': 0, 'h2_wr': 0}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    cum = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    h1 = df[df['half'] == 'H1']
    h2 = df[df['half'] == 'H2']
    return {
        'label': label,
        'n': len(df), 'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': max_dd,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
        'trades_df': df,
        'h1_n': len(h1), 'h1_avg': h1['pnl'].mean() if len(h1) > 0 else 0,
        'h1_total': h1['pnl'].sum() if len(h1) > 0 else 0,
        'h2_n': len(h2), 'h2_avg': h2['pnl'].mean() if len(h2) > 0 else 0,
        'h2_total': h2['pnl'].sum() if len(h2) > 0 else 0,
        'h2_wr': (h2['pnl'] > 0).mean() if len(h2) > 0 else 0,
    }


def compute_capital_adjusted(trades_list, deploy_floor=None):
    if not trades_list:
        return {'n_days': 0, 'daily_avg': 0, 'daily_total': 0, 'daily_wr': 0,
                'daily_dd': 0, 'avg_positions': 0, 'daily_series': pd.Series(dtype=float),
                'h2_daily_avg': 0, 'h2_daily_total': 0, 'h2_n_days': 0}
    df = pd.DataFrame(trades_list)
    if deploy_floor is not None:
        tk_avg = df.groupby('ticker')['pnl'].mean()
        good_tickers = tk_avg[tk_avg >= deploy_floor].index
        df = df[df['ticker'].isin(good_tickers)]
    if len(df) == 0:
        return {'n_days': 0, 'daily_avg': 0, 'daily_total': 0, 'daily_wr': 0,
                'daily_dd': 0, 'avg_positions': 0, 'daily_series': pd.Series(dtype=float),
                'h2_daily_avg': 0, 'h2_daily_total': 0, 'h2_n_days': 0}
    daily = df.groupby('date').agg(
        day_pnl=('pnl', 'mean'),
        n_positions=('pnl', 'count'),
        day_total=('pnl', 'sum'),
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    cum = daily['day_pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    h2_mask = daily.index >= pd.Timestamp(HALF2_START)
    h2_daily = daily[h2_mask]
    return {
        'n_days': len(daily),
        'daily_avg': daily['day_pnl'].mean(),
        'daily_total': daily['day_pnl'].sum(),
        'daily_wr': (daily['day_pnl'] > 0).mean(),
        'daily_dd': max_dd,
        'avg_positions': daily['n_positions'].mean(),
        'daily_series': daily['day_pnl'],
        'h2_daily_avg': h2_daily['day_pnl'].mean() if len(h2_daily) > 0 else 0,
        'h2_daily_total': h2_daily['day_pnl'].sum() if len(h2_daily) > 0 else 0,
        'h2_n_days': len(h2_daily),
    }


# ==============================================================================
# MAIN
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT ALL-MODEL COMPARISON")
print(f"  Question: Does the 10AM model (425/435) add value over 400/410?")
print(f"  TEST: {TEST_START} -> {TEST_END}  |  H2: {HALF2_START} -> {TEST_END}")
print(f"{'='*W}")
for fam_name, fam in FAMILY_CONFIG.items():
    print(f"  {fam_name:>10}: folds {fam['folds']}  entry_min={fam['entry_min']}  {fam['label']}")
print(f"  Tickers: {len(ALL_TICKERS)} candidates")
print(f"  Deploy threshold: {DEPLOY_FLOOR:.1%}")


# ── Step 1: Load BQ configs from ALL model families ─────────────────────────
print(f"\n  STEP 1: Loading BQ configs (all families)...")

all_mvs = [FOLD_MODEL_VERSIONS[f] for f in ALL_FOLDS]
mv_str = ", ".join(f"'{mv}'" for mv in all_mvs)
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

if len(bq_df) == 0:
    print("    WARNING: No BQ results found. Trying without is_ensemble filter...")
    q2 = f"""
    SELECT ticker, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, sl_mult, grace_minutes,
           opt_avg_pnl, opt_win_rate, opt_count,
           val_a_avg_pnl, val_a_win_rate, val_a_count,
           is_ensemble, model_version
    FROM `{BQ_TABLE}`
    WHERE model_version IN ({mv_str})
      AND opt_avg_pnl > 0
      AND tpsl_mode IN ({modes_str})
    ORDER BY ticker, model_version
    """
    bq_df = bq_client.query(q2).to_dataframe()

# Extract fold number from model_version
def _extract_fold(mv):
    if '_10am' in mv:
        parts = mv.replace('_10am', '').split('_')
        return int(parts[-1])
    parts = mv.split('_')
    return int(parts[-1])

bq_df['fold'] = bq_df['model_version'].apply(_extract_fold)

for col in ['prob_thresh', 'tp_pct', 'sl_pct', 'sl_mult', 'grace_minutes',
            'entry_minute', 'opt_avg_pnl', 'opt_win_rate', 'opt_count',
            'val_a_avg_pnl', 'val_a_win_rate', 'val_a_count']:
    bq_df[col] = bq_df[col].apply(lambda x: _safe_float(x, 0.0))

bq_df['family'] = bq_df['fold'].map(FOLD_TO_FAMILY)

bq_tickers = sorted(bq_df['ticker'].unique())
print(f"    BQ: {len(bq_df)} configs, {len(bq_tickers)} tickers, {bq_df['fold'].nunique()} folds")

for fam_name in FAMILY_CONFIG:
    fam_df = bq_df[bq_df['family'] == fam_name]
    n_cfgs = len(fam_df)
    n_tickers = fam_df['ticker'].nunique()
    folds_found = sorted(fam_df['fold'].unique()) if n_cfgs > 0 else []
    print(f"    {fam_name:>10}: {n_cfgs:>5} configs, {n_tickers:>3} tickers, folds={folds_found}")


# ── Step 2: Load OHLCV data ────────────────────────────────────────────────
print(f"\n  STEP 2: Loading OHLCV + ATR for {len(ALL_TICKERS)} tickers...")
ticker_data = {}

for ticker in ALL_TICKERS:
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
        n_days = len(np.unique(ohlcv['dates']))
        print(f"    {ticker}: {n_days} trading days")
        del df
        _master_cache.pop(ticker, None)
    except Exception as ex:
        print(f"    {ticker}: OHLCV load failed -- {ex}")
        _master_cache.pop(ticker, None)
    gc.collect()

_release_memory()
print(f"    OHLCV ready: {len(ticker_data)} tickers")


# ── Step 3: Load fold-specific CSV scores (per family entry_minute) ────────
print(f"\n  STEP 3: Loading CSV seed scores (all families x all tickers)...")

# all_fold_probs[ticker][fold] = {date: prob}
all_fold_probs = defaultdict(dict)
family_prob_loaded = defaultdict(int)

for fold in sorted(ALL_FOLDS):
    exp = FOLD_EXPERIMENTS[fold]
    entry_min = FOLD_ENTRY_MIN[fold]
    fam = FOLD_TO_FAMILY[fold]
    n_loaded = 0
    for ticker in ALL_TICKERS:
        if ticker not in ticker_data:
            continue
        if fold in all_fold_probs[ticker]:
            continue
        seeds = load_seed_scores(ticker, exp)
        if not seeds:
            continue
        dp = compute_daily_prob(seeds, entry_min)
        if dp:
            all_fold_probs[ticker][fold] = dp
            n_loaded += 1
        del seeds
    if n_loaded > 0:
        print(f"    Fold {fold} ({fam}, entry={entry_min}): {n_loaded} tickers")
        family_prob_loaded[fam] += n_loaded
    _release_memory()

# Build per-family ensemble probs
family_ensemble_probs = {}
for fam_name, fam in FAMILY_CONFIG.items():
    fam_folds = fam['folds']
    ens = {}
    for ticker in ticker_data:
        fold_dps = {f: all_fold_probs[ticker][f] for f in fam_folds if f in all_fold_probs.get(ticker, {})}
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
        ens[ticker] = ep
    family_ensemble_probs[fam_name] = ens

for fam_name in FAMILY_CONFIG:
    n = len(family_ensemble_probs.get(fam_name, {}))
    print(f"    {fam_name:>10} ensemble: {n} tickers")

# Global ensemble across ALL families (for oracle)
global_ensemble = {}
for ticker in ticker_data:
    all_dps = [dp for f, dp in all_fold_probs.get(ticker, {}).items()]
    if not all_dps:
        continue
    all_dates = set()
    for dp in all_dps:
        all_dates.update(dp.keys())
    ep = {}
    for d in all_dates:
        vals = [dp[d] for dp in all_dps if d in dp]
        if vals:
            ep[d] = np.mean(vals)
    global_ensemble[ticker] = ep

tickers_ready = sorted(set(ticker_data.keys()) & set(global_ensemble.keys()))
print(f"\n    Ready: {len(tickers_ready)} tickers with OHLCV + probs (any family)")

# Also check which tickers have probs from each family
for fam_name in FAMILY_CONFIG:
    ens = family_ensemble_probs.get(fam_name, {})
    ready = [t for t in ALL_TICKERS if t in ens and t in ticker_data]
    rule1_in = [t for t in RULE_1 if t in ens]
    print(f"    {fam_name:>10}: {len(ready)} tickers ready  (RULE_1: {len(rule1_in)}/6)")


# ── Step 4: Config selection + backtests ───────────────────────────────────
print(f"\n  STEP 4: Running {len(STRATEGY_NAMES)} strategies...")
print(f"  NOTE: 9:30 and 10AM models are ADDITIVE — different entry times,")
print(f"        same ticker can fire both on the same day.")

results = {}

# Single-fold standalone strategies
cfgs_403 = select_single_fold(bq_df, tickers_ready, 403)
print(f"\n  FOLD_403 (9:30): {len(cfgs_403)} tickers with configs")

cfgs_414 = select_single_fold(bq_df, tickers_ready, 414)
print(f"  FOLD_414 (9:30): {len(cfgs_414)} tickers with configs")

cfgs_425 = select_single_fold(bq_df, tickers_ready, 425)
print(f"  FOLD_425 (10AM, 2.5%): {len(cfgs_425)} tickers with configs")

cfgs_435 = select_single_fold(bq_df, tickers_ready, 435)
print(f"  FOLD_435 (10AM, 8%): {len(cfgs_435)} tickers with configs")

# Best 10AM config (prefer 435 if available, fallback to 425)
best_10am_cfgs = {}
for ticker in tickers_ready:
    if ticker in cfgs_435:
        best_10am_cfgs[ticker] = cfgs_435[ticker]
    elif ticker in cfgs_425:
        best_10am_cfgs[ticker] = cfgs_425[ticker]
print(f"  Best 10AM (435 or 425): {len(best_10am_cfgs)} tickers")

# ORACLE: best actual H2 config per ticker (biased)
test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

oracle_cfgs = {}
for ticker in tickers_ready:
    t_df = bq_df[bq_df['ticker'] == ticker]
    if len(t_df) == 0:
        continue
    ohlcv = ticker_data.get(ticker)
    if ohlcv is None:
        continue
    best_h2_avg = -999
    best_cfg = None

    for _, row in t_df.iterrows():
        cfg = _bq_row_to_cfg(row)
        fold = cfg['fold']
        if fold in all_fold_probs.get(ticker, {}):
            dp = all_fold_probs[ticker][fold]
        elif ticker in global_ensemble:
            dp = global_ensemble[ticker]
        else:
            continue

        em = cfg['entry_minute']
        thresh = cfg['prob_thresh']
        entry_indices = np.where(ohlcv['times'] == em)[0]
        h2_trades = []
        last_td = -1

        for idx in entry_indices:
            d_int = int(ohlcv['dates'][idx])
            if d_int == last_td:
                continue
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
            if d_date < h2_start_dt or d_date > test_end_dt:
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
                h2_trades.append(pnl)
            last_td = d_int

        if len(h2_trades) >= MIN_H2_TRADES:
            h2_avg = np.mean(h2_trades)
            if h2_avg > best_h2_avg:
                best_h2_avg = h2_avg
                best_cfg = cfg.copy()

    if best_cfg is not None:
        oracle_cfgs[ticker] = best_cfg

print(f"  ORACLE: {len(oracle_cfgs)} tickers with H2-positive configs")
oracle_family_counts = defaultdict(int)
for tk, cfg in oracle_cfgs.items():
    oracle_family_counts[cfg['family']] += 1
for fam, cnt in sorted(oracle_family_counts.items(), key=lambda x: -x[1]):
    print(f"    -> {fam}: won {cnt} tickers (oracle)")

# ── Helper to build prob map for a config set ──
def _build_dp_map(cfgs):
    dp_map = {}
    for ticker, cfg in cfgs.items():
        fold = cfg['fold']
        if fold in all_fold_probs.get(ticker, {}):
            dp_map[ticker] = all_fold_probs[ticker][fold]
        else:
            fam = cfg.get('family', '')
            if fam in family_ensemble_probs and ticker in family_ensemble_probs[fam]:
                dp_map[ticker] = family_ensemble_probs[fam][ticker]
            elif ticker in global_ensemble:
                dp_map[ticker] = global_ensemble[ticker]
    return dp_map

# ── Run standalone backtests ──
standalone_strats = {
    'FOLD_403': cfgs_403,
    'FOLD_414': cfgs_414,
    'FOLD_425': cfgs_425,
    'FOLD_435': cfgs_435,
    'ORACLE':   oracle_cfgs,
}

for sname, cfgs in standalone_strats.items():
    np.random.seed(42)
    if not cfgs:
        print(f"\n  {sname}: no configs -> SKIP")
        results[sname] = summarize_trades([], sname)
        continue

    dp_map = _build_dp_map(cfgs)
    trades, per_tk = run_csv_backtest(cfgs, dp_map, ticker_data, sname)
    port = summarize_trades(trades, sname)
    port['per_ticker'] = per_tk
    port['configs'] = cfgs
    port['n_tickers_active'] = len(per_tk)
    results[sname] = port

    print(f"\n  {sname} ({STRATEGY_LABELS.get(sname, '')}):")
    print(f"    {port['n_tickers_active']} tickers, {port['n']} trades, "
          f"avg={port['avg']:+.2%}, total={port['total']:+.1%}, "
          f"H2={port['h2_avg']:+.2%} ({port['h2_n']} trades)")

# ── Run ADDITIVE backtests: 9:30 + best 10AM on same day ──
# These are independent trades at different entry times — both can fire same day.
for base_name, base_cfgs in [('ADDITIVE_403', cfgs_403), ('ADDITIVE_414', cfgs_414)]:
    np.random.seed(42)
    if not base_cfgs:
        results[base_name] = summarize_trades([], base_name)
        print(f"\n  {base_name}: no 9:30 configs -> SKIP")
        continue

    # Run 9:30 leg
    dp_930 = _build_dp_map(base_cfgs)
    trades_930, per_tk_930 = run_csv_backtest(base_cfgs, dp_930, ticker_data, f"{base_name}_930")

    # Run 10AM leg (only tickers that also have a 10AM config)
    np.random.seed(42)
    dp_10am = _build_dp_map(best_10am_cfgs)
    trades_10am, per_tk_10am = run_csv_backtest(best_10am_cfgs, dp_10am, ticker_data, f"{base_name}_10am")

    # Merge: tag each trade with its entry time slot
    for t in trades_930:
        t['slot'] = '9:30'
    for t in trades_10am:
        t['slot'] = '10AM'
    combined_trades = trades_930 + trades_10am

    # Merge per_ticker
    combined_per_tk = defaultdict(list)
    for tk, tl in per_tk_930.items():
        combined_per_tk[tk].extend(tl)
    for tk, tl in per_tk_10am.items():
        combined_per_tk[tk].extend(tl)

    port = summarize_trades(combined_trades, base_name)
    port['per_ticker'] = dict(combined_per_tk)
    port['configs'] = {**base_cfgs}
    for tk, cfg in best_10am_cfgs.items():
        if tk not in port['configs']:
            port['configs'][tk] = cfg
    port['n_tickers_active'] = len(combined_per_tk)
    port['n_930'] = len(trades_930)
    port['n_10am'] = len(trades_10am)
    results[base_name] = port

    pct_10am = len(trades_10am) / max(1, len(combined_trades)) * 100
    print(f"\n  {base_name} ({STRATEGY_LABELS.get(base_name, '')}):")
    print(f"    {port['n_tickers_active']} tickers, {len(combined_trades)} trades "
          f"({len(trades_930)} @ 9:30 + {len(trades_10am)} @ 10AM = {pct_10am:.0f}% from 10AM)")
    print(f"    avg={port['avg']:+.2%}, total={port['total']:+.1%}, "
          f"H2={port['h2_avg']:+.2%} ({port['h2_n']} trades)")

# ── BEST_930: best val_a config from any 9:30 fold per ticker ──
cfgs_best_930 = select_best_by_entry_time(bq_df, tickers_ready, 375)
print(f"\n  BEST_930: {len(cfgs_best_930)} tickers with configs (best val_a across folds 400-415)")
best930_folds = defaultdict(int)
for tk, cfg in cfgs_best_930.items():
    best930_folds[cfg['fold']] += 1
for fold, cnt in sorted(best930_folds.items()):
    print(f"    fold {fold}: won {cnt} tickers")

np.random.seed(42)
dp_map = _build_dp_map(cfgs_best_930)
trades_b930, per_tk_b930 = run_csv_backtest(cfgs_best_930, dp_map, ticker_data, 'BEST_930')
port = summarize_trades(trades_b930, 'BEST_930')
port['per_ticker'] = per_tk_b930
port['configs'] = cfgs_best_930
port['n_tickers_active'] = len(per_tk_b930)
results['BEST_930'] = port
print(f"  BEST_930: {port['n_tickers_active']} tickers, {port['n']} trades, "
      f"avg={port['avg']:+.2%}, total={port['total']:+.1%}, "
      f"H2={port['h2_avg']:+.2%} ({port['h2_n']} trades)")

# ── BEST_10AM: best val_a config from any 10AM fold per ticker ──
cfgs_best_10am = select_best_by_entry_time(bq_df, tickers_ready, 405)
print(f"\n  BEST_10AM: {len(cfgs_best_10am)} tickers with configs (best val_a across 425/435)")

np.random.seed(42)
dp_map = _build_dp_map(cfgs_best_10am)
trades_b10am, per_tk_b10am = run_csv_backtest(cfgs_best_10am, dp_map, ticker_data, 'BEST_10AM')
port = summarize_trades(trades_b10am, 'BEST_10AM')
port['per_ticker'] = per_tk_b10am
port['configs'] = cfgs_best_10am
port['n_tickers_active'] = len(per_tk_b10am)
results['BEST_10AM'] = port
print(f"  BEST_10AM: {port['n_tickers_active']} tickers, {port['n']} trades, "
      f"avg={port['avg']:+.2%}, total={port['total']:+.1%}, "
      f"H2={port['h2_avg']:+.2%} ({port['h2_n']} trades)")

# ── ADDITIVE_BEST: BEST_930 + BEST_10AM ──
np.random.seed(42)
dp_930 = _build_dp_map(cfgs_best_930)
trades_ab930, per_tk_ab930 = run_csv_backtest(cfgs_best_930, dp_930, ticker_data, 'ADDITIVE_BEST_930')
np.random.seed(42)
dp_10am = _build_dp_map(cfgs_best_10am)
trades_ab10am, per_tk_ab10am = run_csv_backtest(cfgs_best_10am, dp_10am, ticker_data, 'ADDITIVE_BEST_10am')

for t in trades_ab930:
    t['slot'] = '9:30'
for t in trades_ab10am:
    t['slot'] = '10AM'
combined_ab = trades_ab930 + trades_ab10am
combined_ab_per_tk = defaultdict(list)
for tk, tl in per_tk_ab930.items():
    combined_ab_per_tk[tk].extend(tl)
for tk, tl in per_tk_ab10am.items():
    combined_ab_per_tk[tk].extend(tl)

port = summarize_trades(combined_ab, 'ADDITIVE_BEST')
port['per_ticker'] = dict(combined_ab_per_tk)
port['configs'] = {**cfgs_best_930}
for tk, cfg in cfgs_best_10am.items():
    if tk not in port['configs']:
        port['configs'][tk] = cfg
port['n_tickers_active'] = len(combined_ab_per_tk)
port['n_930'] = len(trades_ab930)
port['n_10am'] = len(trades_ab10am)
results['ADDITIVE_BEST'] = port
pct_10am = len(trades_ab10am) / max(1, len(combined_ab)) * 100
print(f"\n  ADDITIVE_BEST: {port['n_tickers_active']} tickers, {len(combined_ab)} trades "
      f"({len(trades_ab930)} @ 9:30 + {len(trades_ab10am)} @ 10AM = {pct_10am:.0f}% from 10AM)")
print(f"    avg={port['avg']:+.2%}, total={port['total']:+.1%}, "
      f"H2={port['h2_avg']:+.2%} ({port['h2_n']} trades)")


# ==============================================================================
# TABLE 1: STRATEGY COMPARISON — Full Period
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: FULL PERIOD ({TEST_START} -> {TEST_END})")
print(f"{'='*W}")

print(f"\n  {'Strategy':<20} {'Label':<28} {'Tkrs':>4} {'Trades':>6} {'(9:30+10AM)':>12} "
      f"{'Avg PnL':>8} {'Total':>9} {'WR':>5} {'MaxDD':>7}")
print(f"  {'─'*105}")

for sname in STRATEGY_NAMES:
    p = results[sname]
    biased = '*' if sname == 'ORACLE' else ' '
    lbl = STRATEGY_LABELS.get(sname, sname)[:26]
    if p['n'] == 0:
        print(f"  {sname:<20} {lbl:<28} {'—':>4} {'—':>6} {'':>12} "
              f"{'—':>8} {'—':>9} {'—':>5} {'—':>7}")
    else:
        slot_info = ''
        if 'n_930' in p:
            slot_info = f"({p['n_930']}+{p['n_10am']})"
        print(f"  {sname:<20} {lbl:<28} {p['n_tickers_active']:>4} {p['n']:>6} {slot_info:>12} "
              f"{p['avg']:>+7.2%}{biased} {p['total']:>+8.1%} {p['wr']:>5.0%} {p['dd']:>+7.2%}")

# ── Deployed-only portfolio: trades only from tickers above DEPLOY_FLOOR ──
print(f"\n  {'─'*105}")
print(f"  DEPLOYED-ONLY (tickers with avg >= {DEPLOY_FLOOR:.1%}):")
print(f"  {'─'*105}")

for sname in STRATEGY_NAMES:
    p = results[sname]
    if p['n'] == 0:
        continue
    per_tk = p.get('per_ticker', {})
    deploy_tickers = [tk for tk, trades in per_tk.items()
                      if trades and np.mean([t['pnl'] for t in trades]) >= DEPLOY_FLOOR]
    if not deploy_tickers:
        print(f"  {sname:<20} {'(no tickers above threshold)':<28}")
        continue
    deploy_trades = []
    for tk in deploy_tickers:
        deploy_trades.extend(per_tk[tk])
    dp = summarize_trades(deploy_trades, f'{sname}_deployed')
    biased = '*' if sname == 'ORACLE' else ' '
    lbl = f"deployed ({len(deploy_tickers)} tkrs)"
    print(f"  {sname:<20} {lbl:<28} {len(deploy_tickers):>4} {dp['n']:>6} {'':>12} "
          f"{dp['avg']:>+7.2%}{biased} {dp['total']:>+8.1%} {dp['wr']:>5.0%} {dp['dd']:>+7.2%}"
          f"  [{', '.join(sorted(deploy_tickers))}]")


# ==============================================================================
# TABLE 2: H2-ONLY COMPARISON
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 2: H2-ONLY ({HALF2_START} -> {TEST_END})")
print(f"{'='*W}")

print(f"\n  {'Strategy':<20} {'H2 Trd':>7} {'H2 Avg':>8} "
      f"{'H2 Total':>9} {'H2 WR':>6}  {'H1 Avg':>7} {'Edge':>5}")
print(f"  {'─'*70}")

for sname in STRATEGY_NAMES:
    p = results[sname]
    if p['n'] == 0:
        continue
    edge = 'UP' if p['h2_avg'] > p['h1_avg'] else ('DOWN' if p['h2_avg'] < p['h1_avg'] else 'FLAT')
    biased = '*' if sname == 'ORACLE' else ' '
    print(f"  {sname:<20} {p['h2_n']:>7} {p['h2_avg']:>+7.2%}{biased} "
          f"{p['h2_total']:>+8.1%} {p['h2_wr']:>5.0%}  "
          f"{p['h1_avg']:>+7.2%} {edge:>5}")

print(f"\n  {'─'*70}")
print(f"  DEPLOYED-ONLY H2 (tickers with full-period avg >= {DEPLOY_FLOOR:.1%}):")
print(f"  {'─'*70}")

for sname in STRATEGY_NAMES:
    p = results[sname]
    if p['n'] == 0:
        continue
    per_tk = p.get('per_ticker', {})
    deploy_tickers = [tk for tk, trades in per_tk.items()
                      if trades and np.mean([t['pnl'] for t in trades]) >= DEPLOY_FLOOR]
    if not deploy_tickers:
        continue
    deploy_h2 = []
    deploy_h1 = []
    for tk in deploy_tickers:
        for t in per_tk[tk]:
            if t['half'] == 'H2':
                deploy_h2.append(t)
            else:
                deploy_h1.append(t)
    if not deploy_h2:
        continue
    h2_avg = np.mean([t['pnl'] for t in deploy_h2])
    h2_total = sum(t['pnl'] for t in deploy_h2)
    h2_wr = np.mean([1 if t['pnl'] > 0 else 0 for t in deploy_h2])
    h1_avg = np.mean([t['pnl'] for t in deploy_h1]) if deploy_h1 else 0
    edge = 'UP' if h2_avg > h1_avg else ('DOWN' if h2_avg < h1_avg else 'FLAT')
    biased = '*' if sname == 'ORACLE' else ' '
    print(f"  {sname:<20} {len(deploy_h2):>7} {h2_avg:>+7.2%}{biased} "
          f"{h2_total:>+8.1%} {h2_wr:>5.0%}  "
          f"{h1_avg:>+7.2%} {edge:>5}  [{', '.join(sorted(deploy_tickers))}]")


# ==============================================================================
# TABLE 3: PER-TICKER DETAIL — Each strategy
# ==============================================================================
for sname in STRATEGY_NAMES:
    p = results[sname]
    if p['n'] == 0:
        continue

    per_tk = p.get('per_ticker', {})
    cfgs = p.get('configs', {})

    print(f"\n{'='*W}")
    print(f"  TABLE 3.{STRATEGY_NAMES.index(sname)+1}: PER-TICKER — {sname} "
          f"(total={p['total']:+.1%}, avg={p['avg']:+.2%})")
    print(f"{'='*W}")

    print(f"  {'Ticker':<7} {'Trades':>6} {'Avg':>7} {'Total':>8} {'WR':>5} "
          f"{'H2 Avg':>7} {'Fold':>5} {'Family':<10} {'Mode':<8} {'TP':>4} {'SL':>4} {'Thr':>5}")
    print(f"  {'─'*90}")

    for ticker in sorted(set(list(per_tk.keys()) + list(cfgs.keys()))):
        trades = per_tk.get(ticker, [])
        cfg = cfgs.get(ticker)
        n = len(trades)
        if n == 0:
            continue
        avg_pnl = np.mean([t['pnl'] for t in trades])
        total_pnl = sum(t['pnl'] for t in trades)
        wr = np.mean([1 for t in trades if t['pnl'] > 0]) / n * 100
        h2_trades = [t for t in trades if t['half'] == 'H2']
        h2_avg = np.mean([t['pnl'] for t in h2_trades]) if h2_trades else 0

        fold_s = f"{cfg['fold']}" if cfg else '—'
        fam_s = cfg.get('family', '?')[:9] if cfg else '—'
        mode_s = cfg['tpsl_mode'] if cfg else '—'
        tp_s = f"{cfg['tp_pct']*100:.0f}%" if cfg else '—'
        sl_s = f"{cfg['sl_pct']*100:.0f}%" if cfg and cfg['tpsl_mode'] != 'atr_sl' else \
               (f"x{cfg['sl_mult']:.1f}" if cfg else '—')
        thr_s = f"{cfg['prob_thresh']:.2f}" if cfg else '—'

        deploy_flag = ' >' if avg_pnl >= DEPLOY_FLOOR else '  '
        rule1_flag = ' R1' if ticker in RULE_1 else '   '
        print(f"  {ticker:<7} {n:>6} {avg_pnl:>+6.2%} {total_pnl:>+7.1%} {wr:>4.0f}% "
              f"{h2_avg:>+6.2%} {fold_s:>5} {fam_s:<10} {mode_s:<8} {tp_s:>4} {sl_s:>4} {thr_s:>5}"
              f"{deploy_flag}{rule1_flag}")


# ==============================================================================
# TABLE 4: ADDITIVE VALUE — What did 10AM add to each 9:30 strategy?
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 4: ADDITIVE VALUE — 10AM marginal contribution")
print(f"{'='*W}")

for base_fold, add_name in [('FOLD_403', 'ADDITIVE_403'), ('FOLD_414', 'ADDITIVE_414'),
                            ('BEST_930', 'ADDITIVE_BEST')]:
    p_base = results[base_fold]
    p_add  = results[add_name]
    if p_base['n'] == 0:
        continue

    n_930  = p_add.get('n_930', 0)
    n_10am = p_add.get('n_10am', 0)
    delta_avg = p_add['avg'] - p_base['avg'] if p_add['n'] > 0 else 0
    delta_total = p_add['total'] - p_base['total'] if p_add['n'] > 0 else 0

    print(f"\n  {base_fold} -> {add_name}:")
    print(f"    9:30 trades:  {p_base['n']:>4}  avg={p_base['avg']:>+.2%}  total={p_base['total']:>+.1%}")
    if p_add['n'] > 0:
        print(f"    + 10AM trades: {n_10am:>4}  (new signals at different entry time)")
        print(f"    = Combined:   {p_add['n']:>4}  avg={p_add['avg']:>+.2%}  total={p_add['total']:>+.1%}")
        print(f"    Delta:               avg {delta_avg:>+.2%}  total {delta_total:>+.1%}")
        # Show which tickers the 10AM leg added trades for
        tickers_10am = set()
        for tk, trades in p_add.get('per_ticker', {}).items():
            if any(t.get('slot') == '10AM' for t in trades):
                tickers_10am.add(tk)
        tickers_only_10am = tickers_10am - set(p_base.get('per_ticker', {}).keys())
        print(f"    10AM fired on: {', '.join(sorted(tickers_10am)) if tickers_10am else 'none'}")
        if tickers_only_10am:
            print(f"    10AM-only tickers (not in 9:30): {', '.join(sorted(tickers_only_10am))}")
    else:
        print(f"    No 10AM configs available — additive = same as standalone")


# ==============================================================================
# TABLE 5: 10AM STANDALONE ANALYSIS — Per-model quality
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 5: 10AM MODEL STANDALONE ANALYSIS")
print(f"  (These run SEPARATELY from 9:30 — not a replacement, an addition)")
print(f"{'='*W}")

tickers_with_425 = set(cfgs_425.keys())
tickers_with_435 = set(cfgs_435.keys())
tickers_with_10am = tickers_with_425 | tickers_with_435
tickers_with_403 = set(cfgs_403.keys())
tickers_with_414 = set(cfgs_414.keys())

print(f"\n  10AM coverage:")
print(f"    F425 (2.5% dump): {len(tickers_with_425)} tickers — {', '.join(sorted(tickers_with_425)) if tickers_with_425 else 'none'}")
print(f"    F435 (8% dump):   {len(tickers_with_435)} tickers — {', '.join(sorted(tickers_with_435)) if tickers_with_435 else 'none'}")
print(f"    Union:            {len(tickers_with_10am)} unique tickers")

# Tickers that ONLY have 10AM configs (purely additive — new capacity)
only_10am = tickers_with_10am - tickers_with_403 - tickers_with_414
if only_10am:
    print(f"\n  10AM-only tickers (no 9:30 config — pure new capacity):")
    print(f"    {', '.join(sorted(only_10am))}")

# Head-to-head quality comparison (standalone, not additive)
all_10am_strats = [s for s in ['FOLD_425', 'FOLD_435'] if results[s]['n'] > 0]
if all_10am_strats:
    all_compare_tickers = set()
    for s in all_10am_strats:
        all_compare_tickers.update(results[s].get('per_ticker', {}).keys())

    if all_compare_tickers:
        print(f"\n  10AM per-ticker quality (standalone):")
        hdr = f"  {'Ticker':<7}"
        for s in all_10am_strats:
            hdr += f" {s+' avg':>12} {s+' #tr':>6}"
        print(hdr)
        print(f"  {'─'*60}")

        for ticker in sorted(all_compare_tickers):
            row = f"  {ticker:<7}"
            for s in all_10am_strats:
                trades = results[s].get('per_ticker', {}).get(ticker, [])
                if trades:
                    avg = np.mean([t['pnl'] for t in trades])
                    row += f" {avg:>+11.2%} {len(trades):>6}"
                else:
                    row += f" {'—':>12} {'—':>6}"
            print(row)

        for s in all_10am_strats:
            per_tk = results[s].get('per_ticker', {})
            deployable = [tk for tk, trades in per_tk.items()
                          if trades and np.mean([t['pnl'] for t in trades]) >= DEPLOY_FLOOR]
            print(f"\n  {s} deployable (>= {DEPLOY_FLOOR:.1%}): "
                  f"{len(deployable)} — {', '.join(sorted(deployable)) if deployable else 'none'}")


# ==============================================================================
# TABLE 6: CAPITAL-ADJUSTED PnL — Equal-weight daily portfolio
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 6: CAPITAL-ADJUSTED PnL (equal-weight across same-day positions)")
print(f"  Deployed tickers only (avg >= {DEPLOY_FLOOR:.1%})")
print(f"{'='*W}")
print(f"  If N positions fire on the same day, each gets 1/N of capital.")
print(f"  Daily PnL = mean(trade PnLs that day). This is what you'd see in production.")

print(f"\n  {'Strategy':<20} {'Days':>5} {'Avg/Day':>8} {'Total':>8} {'WR':>5} "
      f"{'MaxDD':>7} {'Pos/Day':>7}  {'H2 Avg':>7} {'H2 Days':>7}")
print(f"  {'─'*85}")

cap_adj_results = {}
for sname in STRATEGY_NAMES:
    p = results[sname]
    if p['n'] == 0:
        continue
    per_tk = p.get('per_ticker', {})
    deploy_tickers = [tk for tk, trades in per_tk.items()
                      if trades and np.mean([t['pnl'] for t in trades]) >= DEPLOY_FLOOR]
    if not deploy_tickers:
        continue
    deploy_trades = []
    for tk in deploy_tickers:
        deploy_trades.extend(per_tk[tk])
    ca = compute_capital_adjusted(deploy_trades)
    cap_adj_results[sname] = ca
    biased = '*' if sname == 'ORACLE' else ' '
    print(f"  {sname:<20} {ca['n_days']:>5} {ca['daily_avg']:>+7.2%}{biased} "
          f"{ca['daily_total']:>+7.1%} {ca['daily_wr']:>4.0%} "
          f"{ca['daily_dd']:>+7.2%} {ca['avg_positions']:>6.1f}  "
          f"{ca['h2_daily_avg']:>+7.2%} {ca['h2_n_days']:>7}")

# Best strategy highlight
non_oracle = {k: v for k, v in cap_adj_results.items() if k != 'ORACLE' and v['n_days'] > 0}
if non_oracle:
    best_strat = max(non_oracle, key=lambda s: non_oracle[s]['daily_avg'])
    best_ca = non_oracle[best_strat]
    print(f"\n  >>> BEST DEPLOYED: {best_strat}")
    print(f"      {best_ca['daily_avg']:+.2%}/day over {best_ca['n_days']} trading days")
    print(f"      {best_ca['avg_positions']:.1f} positions/day avg")
    print(f"      H2: {best_ca['h2_daily_avg']:+.2%}/day ({best_ca['h2_n_days']} days)")


# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(4, 2, figsize=(22, 30))

# ── Chart 1: Strategy comparison bar chart ──
ax1 = axes[0, 0]
active_strats = [s for s in STRATEGY_NAMES if results[s]['n'] > 0]
x1 = np.arange(len(active_strats))
avgs = [results[s]['avg'] * 100 for s in active_strats]
colors = [STRATEGY_COLORS.get(s, '#666') for s in active_strats]
bars = ax1.bar(x1, avgs, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
for bar, s in zip(bars, active_strats):
    n = results[s]['n']
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{n}tr', ha='center', va='bottom', fontsize=9)
ax1.axhline(y=DEPLOY_FLOOR * 100, color='red', linewidth=1.5, linestyle='--',
            alpha=0.7, label=f'Deploy ({DEPLOY_FLOOR:.1%})')
ax1.axhline(y=0, color='gray', linewidth=0.5)
ax1.set_xticks(x1)
ax1.set_xticklabels([STRATEGY_LABELS.get(s, s)[:20] for s in active_strats],
                     fontsize=9, rotation=15, ha='right')
ax1.set_ylabel('Avg PnL (%)')
ax1.set_title('Full Period — Avg PnL by Strategy', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# ── Chart 2: H2 comparison ──
ax2 = axes[0, 1]
h2_avgs = [results[s]['h2_avg'] * 100 for s in active_strats]
bars2 = ax2.bar(x1, h2_avgs, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
for bar, s in zip(bars2, active_strats):
    n = results[s]['h2_n']
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{n}tr', ha='center', va='bottom', fontsize=9)
ax2.axhline(y=DEPLOY_FLOOR * 100, color='red', linewidth=1.5, linestyle='--', alpha=0.7)
ax2.axhline(y=0, color='gray', linewidth=0.5)
ax2.set_xticks(x1)
ax2.set_xticklabels([STRATEGY_LABELS.get(s, s)[:20] for s in active_strats],
                     fontsize=9, rotation=15, ha='right')
ax2.set_ylabel('H2 Avg PnL (%)')
ax2.set_title(f'H2 ({HALF2_START}+) — Avg PnL by Strategy', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# ── Chart 3: Cumulative PnL curves ──
ax3 = axes[1, 0]
for sname in STRATEGY_NAMES:
    p = results[sname]
    if p['n'] == 0:
        continue
    df_t = p['trades_df'].sort_values('date')
    cum = df_t['pnl'].cumsum() * 100
    ax3.plot(pd.to_datetime(df_t['date'].values), cum.values,
             color=STRATEGY_COLORS.get(sname, '#666'), linewidth=2.5,
             label=f"{sname} ({p['n']}tr)", alpha=0.9)
ax3.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax3.axvline(x=pd.Timestamp(HALF2_START), color='red', linewidth=1,
            linestyle=':', alpha=0.5, label='H2 start')
ax3.set_ylabel('Cumulative PnL (%)')
ax3.set_title('Cumulative PnL — All Strategies', fontsize=13, fontweight='bold')
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# ── Chart 4: Additive delta — what 10AM added to each 9:30 base ──
ax4 = axes[1, 1]
add_pairs = [('FOLD_403', 'ADDITIVE_403'), ('FOLD_414', 'ADDITIVE_414'),
             ('BEST_930', 'ADDITIVE_BEST')]
add_pairs = [(b, a) for b, a in add_pairs if results[b]['n'] > 0]
if add_pairs:
    x4 = np.arange(len(add_pairs))
    base_totals = [results[b]['total'] * 100 for b, _ in add_pairs]
    add_totals  = [results[a]['total'] * 100 for _, a in add_pairs]
    delta_totals = [a - b for b, a in zip(base_totals, add_totals)]
    labels_4 = [b.replace('FOLD_', 'F') for b, _ in add_pairs]

    ax4.bar(x4 - 0.15, base_totals, 0.3, label='9:30 alone', color='#1565C0', alpha=0.85)
    ax4.bar(x4 + 0.15, add_totals, 0.3, label='9:30 + 10AM', color='#D81B60', alpha=0.85)
    for i, d in enumerate(delta_totals):
        ax4.annotate(f'{d:+.1f}%', (x4[i] + 0.15, add_totals[i]),
                     ha='center', va='bottom', fontsize=10, fontweight='bold',
                     color='green' if d > 0 else 'red')
    ax4.set_xticks(x4)
    ax4.set_xticklabels(labels_4, fontsize=11)
    ax4.axhline(y=0, color='gray', linewidth=0.5)
    ax4.set_ylabel('Total PnL (%)')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
ax4.set_title('Additive Value — 10AM on top of 9:30', fontsize=13, fontweight='bold')

# ── Chart 5: Per-ticker avg PnL comparison (FOLD_403 vs FOLD_414 vs FOLD_425) ──
ax5 = axes[2, 0]
compare_strats = ['FOLD_403', 'FOLD_414', 'FOLD_425', 'FOLD_435']
compare_strats = [s for s in compare_strats if results[s]['n'] > 0]

all_compare_tickers = set()
for s in compare_strats:
    all_compare_tickers.update(results[s].get('per_ticker', {}).keys())
compare_tickers = sorted(all_compare_tickers)

if compare_tickers and compare_strats:
    x5 = np.arange(len(compare_tickers))
    bar_w = 0.8 / max(len(compare_strats), 1)
    for i, sname in enumerate(compare_strats):
        per_tk = results[sname].get('per_ticker', {})
        avgs = []
        for tk in compare_tickers:
            trades = per_tk.get(tk, [])
            avgs.append(np.mean([t['pnl'] for t in trades]) * 100 if trades else 0)
        offset = (i - len(compare_strats)/2 + 0.5) * bar_w
        ax5.bar(x5 + offset, avgs, bar_w,
                label=STRATEGY_LABELS.get(sname, sname)[:20],
                color=STRATEGY_COLORS.get(sname, '#666'), alpha=0.85)
    ax5.axhline(y=DEPLOY_FLOOR * 100, color='red', linewidth=1, linestyle='--', alpha=0.7)
    ax5.axhline(y=0, color='gray', linewidth=0.5)
    ax5.set_xticks(x5)
    ax5.set_xticklabels(compare_tickers, fontsize=8, rotation=45, ha='right')
    ax5.set_ylabel('Avg PnL (%)')
    ax5.set_title('Per-Ticker Avg PnL — All Standalone Folds', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

# ── Chart 6: Monthly PnL heatmap ──
ax6 = axes[2, 1]
months_all = set()
for s in STRATEGY_NAMES:
    months_all.update(results[s].get('monthly', {}).keys())
months_sorted = sorted(months_all)

active_for_heat = [s for s in STRATEGY_NAMES if results[s]['n'] > 0]
if months_sorted and active_for_heat:
    heat_data = np.zeros((len(active_for_heat), len(months_sorted)))
    for i, sname in enumerate(active_for_heat):
        monthly = results[sname].get('monthly', {})
        for j, m in enumerate(months_sorted):
            if m in monthly:
                heat_data[i, j] = monthly[m]['avg'] * 100
    vmax = max(abs(heat_data.min()), abs(heat_data.max()))
    if vmax == 0:
        vmax = 1
    im = ax6.imshow(heat_data, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax6.set_yticks(range(len(active_for_heat)))
    ax6.set_yticklabels(active_for_heat, fontsize=10)
    ax6.set_xticks(range(len(months_sorted)))
    ax6.set_xticklabels(months_sorted, fontsize=10)
    for i in range(len(active_for_heat)):
        monthly = results[active_for_heat[i]].get('monthly', {})
        for j, m in enumerate(months_sorted):
            n_trades = monthly[m]['n'] if m in monthly else 0
            ax6.text(j, i, f'{heat_data[i,j]:+.1f}%\n({n_trades})',
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     color='white' if abs(heat_data[i,j]) > vmax*0.5 else 'black')
    fig.colorbar(im, ax=ax6, label='Avg PnL (%)', shrink=0.8)
ax6.set_title('Monthly Avg PnL Heatmap', fontsize=13, fontweight='bold')

# ── Chart 7: Capital-adjusted equity curves (deployed-only) ──
ax7 = axes[3, 0]
cap_strats_plot = ['FOLD_403', 'BEST_930', 'ADDITIVE_403', 'ADDITIVE_BEST', 'ORACLE']
for sname in cap_strats_plot:
    if sname not in cap_adj_results or cap_adj_results[sname]['n_days'] == 0:
        continue
    ca = cap_adj_results[sname]
    cum = ca['daily_series'].cumsum() * 100
    ax7.plot(cum.index, cum.values,
             color=STRATEGY_COLORS.get(sname, '#666'), linewidth=2.5,
             label=f"{sname} ({ca['n_days']}d)", alpha=0.9)
ax7.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax7.axvline(x=pd.Timestamp(HALF2_START), color='red', linewidth=1,
            linestyle=':', alpha=0.5, label='H2 start')
ax7.set_ylabel('Cumulative PnL (%)')
ax7.set_title(f'Capital-Adjusted Equity — Deployed Only (>={DEPLOY_FLOOR:.0%})',
              fontsize=13, fontweight='bold')
ax7.legend(fontsize=8, loc='upper left')
ax7.grid(True, alpha=0.3)
ax7.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# ── Chart 8: Deployed-only avg PnL comparison (capital-adjusted vs raw) ──
ax8 = axes[3, 1]
cap_strats_bar = [s for s in STRATEGY_NAMES if s in cap_adj_results
                  and cap_adj_results[s]['n_days'] > 0 and s != 'ORACLE']
if cap_strats_bar:
    x8 = np.arange(len(cap_strats_bar))
    raw_avgs = []
    cap_avgs = []
    for s in cap_strats_bar:
        p = results[s]
        per_tk = p.get('per_ticker', {})
        dtk = [tk for tk, trades in per_tk.items()
               if trades and np.mean([t['pnl'] for t in trades]) >= DEPLOY_FLOOR]
        dt = []
        for tk in dtk:
            dt.extend(per_tk[tk])
        raw_avgs.append(np.mean([t['pnl'] for t in dt]) * 100 if dt else 0)
        cap_avgs.append(cap_adj_results[s]['daily_avg'] * 100)
    ax8.bar(x8 - 0.15, raw_avgs, 0.3, label='Raw per-trade avg', color='#1565C0', alpha=0.85)
    ax8.bar(x8 + 0.15, cap_avgs, 0.3, label='Capital-adjusted avg/day', color='#D81B60', alpha=0.85)
    ax8.axhline(y=DEPLOY_FLOOR * 100, color='red', linewidth=1, linestyle='--', alpha=0.7)
    ax8.axhline(y=0, color='gray', linewidth=0.5)
    ax8.set_xticks(x8)
    ax8.set_xticklabels([STRATEGY_LABELS.get(s, s)[:18] for s in cap_strats_bar],
                         fontsize=8, rotation=20, ha='right')
    ax8.set_ylabel('Avg PnL (%)')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')
ax8.set_title('Raw vs Capital-Adjusted — Deployed Only', fontsize=13, fontweight='bold')

fig.suptitle(f'SHORT All-Model Comparison — 9:30 + 10AM Additive\n'
             f'{TEST_START} -> {TEST_END}  |  Deploy: {DEPLOY_FLOOR:.1%}  |  Slip: {SLIPPAGE_PCT:.1%}',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/tmp/backtest_short_all_models.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Chart saved: /tmp/backtest_short_all_models.png")


# ==============================================================================
# VERDICT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

for sname in STRATEGY_NAMES:
    p = results[sname]
    if p['n'] == 0:
        print(f"  {sname:<20}: NO DATA (model not yet optimized?)")
        continue
    biased = ' [BIASED]' if sname == 'ORACLE' else ''
    deploy = 'DEPLOY-READY' if p['avg'] >= DEPLOY_FLOOR else 'BELOW THRESH'
    h2_deploy = 'H2-READY' if p['h2_avg'] >= DEPLOY_FLOOR else 'H2-BELOW'
    print(f"  {sname:<20}: {p['n']:>4} trades, avg={p['avg']:>+.2%}, "
          f"total={p['total']:>+.1%}, H2={p['h2_avg']:>+.2%} "
          f"-> {deploy} / {h2_deploy}{biased}")

# Deployable tickers per strategy + deployed-only portfolio estimate
print(f"\n  DEPLOYABLE TICKERS (avg >= {DEPLOY_FLOOR:.1%}):")
print(f"  {'Strategy':<20} {'#Tkrs':>5} {'Trades':>6} {'Avg':>7} {'Total':>8} {'WR':>5}  Tickers")
print(f"  {'─'*90}")
for sname in STRATEGY_NAMES:
    if sname == 'ORACLE':
        continue
    p = results[sname]
    per_tk = p.get('per_ticker', {})
    deployable = []
    deploy_trades = []
    for tk, trades in per_tk.items():
        if trades and np.mean([t['pnl'] for t in trades]) >= DEPLOY_FLOOR:
            deployable.append(tk)
            deploy_trades.extend(trades)
    if deploy_trades:
        d_avg = np.mean([t['pnl'] for t in deploy_trades])
        d_total = sum(t['pnl'] for t in deploy_trades)
        d_wr = np.mean([1 if t['pnl'] > 0 else 0 for t in deploy_trades])
        print(f"  {sname:<20} {len(deployable):>5} {len(deploy_trades):>6} {d_avg:>+6.2%} "
              f"{d_total:>+7.1%} {d_wr:>4.0%}  {', '.join(sorted(deployable))}")
    else:
        print(f"  {sname:<20} {'0':>5} {'—':>6} {'—':>7} {'—':>8} {'—':>5}  none")

# Key question — additive framing
print(f"\n  KEY QUESTION: Does the 10AM model add trades (and PnL) on top of 9:30?")
print(f"  (They run at different times — not competing, stacking.)")

for base_fold, add_name in [('FOLD_403', 'ADDITIVE_403'), ('FOLD_414', 'ADDITIVE_414'),
                            ('BEST_930', 'ADDITIVE_BEST')]:
    p_base = results[base_fold]
    p_add  = results[add_name]
    if p_base['n'] == 0:
        continue
    n_10am = p_add.get('n_10am', 0)
    if n_10am == 0:
        print(f"\n  {base_fold}: 10AM added 0 trades — no additive value yet")
        has_any_10am = results['FOLD_425']['n'] + results['FOLD_435']['n']
        if has_any_10am == 0:
            print(f"    -> No 10AM configs survived optimization")
            print(f"       F425: 2.5% dump target likely too small for TP/SL grid")
            print(f"       F435: not yet trained — will test entry-time hypothesis with aligned 8% dump")
    else:
        delta_avg = p_add['avg'] - p_base['avg']
        delta_total = p_add['total'] - p_base['total']
        print(f"\n  {base_fold} + 10AM = {add_name}:")
        print(f"    9:30 alone:  {p_base['n']} trades, avg={p_base['avg']:+.2%}")
        print(f"    + 10AM:      {n_10am} new trades")
        print(f"    Combined:    {p_add['n']} trades, avg={p_add['avg']:+.2%}")
        print(f"    Delta avg: {delta_avg:+.2%}  Delta total: {delta_total:+.1%}")
        if delta_avg > 0:
            print(f"    -> 10AM HELPS: raises avg PnL and adds volume")
        elif delta_total > 0:
            print(f"    -> 10AM MIXED: lower avg but higher total (more trades)")
        else:
            print(f"    -> 10AM HURTS: dilutes both avg and total")

print(f"\n  LONG S3 reference: +2.23% avg")
print(f"  Deploy threshold:  {DEPLOY_FLOOR:.1%}")
print(f"{'='*W}")
