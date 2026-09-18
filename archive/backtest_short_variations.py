# %% ── SHORT M06_FOLD403 Variations Backtest ────────────────────────────────
#
# Tests 25 variations of M06_FOLD403 across 5 dimensions + oracle ceiling:
#   Dim 1 — Single-Fold: all 12 folds individually (400-405, 410-415)
#   Dim 2 — Fold Ensembles: ENS_AVG_A2, ENS_AVG_ALL, ENS_VOTE_A2, ENS_TOP3_VA
#   Dim 3 — Ranking Criteria: RANK_OVERFIT, RANK_WIN_RATE, RANK_COMPOSITE
#   Dim 4 — Ticker-Specific: TKR_BEST_VA, TKR_BEST_A2, TKR_MEDIAN_VA
#   Dim 5 — Cross-Model: XMODEL_VOTE, XMODEL_AVG, XMODEL_BLEND
#
# CSV-scored (not live TF). RULE_1 tickers. GPU NOT required. Colab-ready.

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
import matplotlib.cm as cm

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
HALF2_START = '2026-06-01'

ENTRY_MIN     = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.002
STARTING_CAPITAL = 200_000
DEPLOY_FLOOR  = 0.018
SL_FIXED      = 0.03

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

SHORT_TICKERS = ['BE', 'IREN', 'OKLO', 'ONDS', 'RKLB', 'WOLF']

FOLDS_A2 = [400, 401, 402, 403, 404, 405]
FOLDS_A1 = [410, 411, 412, 413, 414, 415]
ALL_FOLDS = FOLDS_A2 + FOLDS_A1

FOLD_EXPERIMENTS    = {f: f'{f}_XSECTOR' for f in ALL_FOLDS}
FOLD_MODEL_VERSIONS = {f: f'robust_wf_v12_short_{f}' for f in ALL_FOLDS}

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_ALLOWED_MODES = ('static', 'atr_sl', 'grace')

MIN_H2_TRADES = 3
VOTE_QUORUM = 4

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

DIMENSIONS = {
    'SingleFold': [f'FOLD_{f}' for f in ALL_FOLDS],
    'Ensemble':   ['ENS_AVG_A2', 'ENS_AVG_ALL', 'ENS_VOTE_A2', 'ENS_TOP3_VA'],
    'Ranking':    ['RANK_OVERFIT', 'RANK_WIN_RATE', 'RANK_COMPOSITE'],
    'TickerSpec':  ['TKR_BEST_VA', 'TKR_BEST_A2', 'TKR_MEDIAN_VA'],
    'CrossModel': ['XMODEL_VOTE', 'XMODEL_AVG', 'XMODEL_BLEND'],
}

DIM_COLORS = {
    'SingleFold': '#1565C0',
    'Ensemble':   '#E65100',
    'Ranking':    '#2E7D32',
    'TickerSpec':  '#9C27B0',
    'CrossModel': '#795548',
    'Oracle':     '#D32F2F',
}

TICKER_COLORS = {
    'BE':   '#2196F3', 'IREN': '#4CAF50', 'OKLO': '#FF9800',
    'ONDS': '#E91E63', 'RKLB': '#9C27B0', 'WOLF': '#00BCD4',
}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110


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
# SUMMARIZE RESULTS
# ==============================================================================
def summarize_trades(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0, 'monthly': {}, 'trades_df': pd.DataFrame()}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    cum = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    return {
        'label': label,
        'n': len(df), 'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': max_dd,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
        'trades_df': df,
    }


# ==============================================================================
# BACKTEST FUNCTION — CSV-scored with fold-specific probs
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
        entry_min = cfg.get('entry_minute', ENTRY_MIN)
        prob_thresh = cfg['prob_thresh']
        mode = cfg['tpsl_mode']

        prices = ohlcv['prices']
        highs  = ohlcv['highs']
        lows   = ohlcv['lows']
        dates  = ohlcv['dates']
        times  = ohlcv['times']
        atr_arr = ohlcv.get('atr_pct', np.full(len(prices), 0.05))

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

            if mode == 'atr_sl':
                pnl, trade_date, filled = run_short_trade_atr_sl(
                    prices, highs, lows, dates, times,
                    idx, cfg['tp_pct'], float(atr_arr[idx]), cfg['sl_mult'], EXIT_TIME_MIN)
            elif mode == 'grace':
                pnl, trade_date, filled = run_short_trade_grace(
                    prices, highs, lows, dates, times,
                    idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['grace_minutes'])
            else:
                pnl, trade_date, filled = run_short_trade_static(
                    prices, highs, lows, dates, times,
                    idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)

            if not filled:
                continue

            pnl -= 2 * SLIPPAGE_PCT
            last_traded_date = d_int

            half = 'H2' if d_date >= h2_start_dt else 'H1'
            trade = {
                'date': d_date, 'ticker': ticker, 'pnl': pnl,
                'prob': prob, 'close': float(prices[idx]),
                'mode': mode, 'tp': cfg['tp_pct'],
                'sl': cfg['sl_pct'] if mode != 'atr_sl' else cfg['sl_mult'],
                'half': half,
            }
            all_trades.append(trade)
            per_ticker_trades[ticker].append(trade)

    return all_trades, per_ticker_trades


def run_vote_backtest(configs, all_fold_probs_map, vote_folds, ticker_data, label):
    """Modified backtest: trade only when >= VOTE_QUORUM folds agree (prob > thresh)."""
    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt   = pd.Timestamp(TEST_END).date()
    h2_start_dt   = pd.Timestamp(HALF2_START).date()

    all_trades = []
    per_ticker_trades = defaultdict(list)

    for ticker in sorted(configs.keys()):
        if ticker not in ticker_data:
            continue
        cfg = configs[ticker]
        ohlcv = ticker_data[ticker]
        prob_thresh = cfg['prob_thresh']
        mode = cfg['tpsl_mode']

        fold_dps = []
        for fold in vote_folds:
            dp = all_fold_probs_map.get(ticker, {}).get(fold, {})
            if dp:
                fold_dps.append(dp)
        if not fold_dps:
            continue

        prices = ohlcv['prices']
        dates  = ohlcv['dates']
        times  = ohlcv['times']
        atr_arr = ohlcv.get('atr_pct', np.full(len(prices), 0.05))

        entry_indices = np.where(times == cfg.get('entry_minute', ENTRY_MIN))[0]
        last_traded_date = -1

        for idx in entry_indices:
            d_int = int(dates[idx])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()

            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            if d_int == last_traded_date:
                continue

            votes = sum(1 for dp in fold_dps if dp.get(d_date, 0.0) >= prob_thresh)
            if votes < VOTE_QUORUM:
                continue

            if mode == 'atr_sl':
                pnl, trade_date, filled = run_short_trade_atr_sl(
                    prices, ohlcv['highs'], ohlcv['lows'], dates, times,
                    idx, cfg['tp_pct'], float(atr_arr[idx]), cfg['sl_mult'], EXIT_TIME_MIN)
            elif mode == 'grace':
                pnl, trade_date, filled = run_short_trade_grace(
                    prices, ohlcv['highs'], ohlcv['lows'], dates, times,
                    idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['grace_minutes'])
            else:
                pnl, trade_date, filled = run_short_trade_static(
                    prices, ohlcv['highs'], ohlcv['lows'], dates, times,
                    idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)

            if not filled:
                continue

            pnl -= 2 * SLIPPAGE_PCT
            last_traded_date = d_int

            avg_prob = np.mean([dp.get(d_date, 0.0) for dp in fold_dps])
            half = 'H2' if d_date >= h2_start_dt else 'H1'
            trade = {
                'date': d_date, 'ticker': ticker, 'pnl': pnl,
                'prob': avg_prob, 'close': float(prices[idx]),
                'mode': mode, 'tp': cfg['tp_pct'],
                'sl': cfg['sl_pct'] if mode != 'atr_sl' else cfg['sl_mult'],
                'half': half,
            }
            all_trades.append(trade)
            per_ticker_trades[ticker].append(trade)

    return all_trades, per_ticker_trades


# ==============================================================================
# CONFIG SELECTION METHODS
# ==============================================================================
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
        'val_a_win_rate': row.get('val_a_win_rate', 0.0),
        'opt_avg_pnl': row['opt_avg_pnl'],
        'va_opt_ratio': row.get('va_opt_ratio', 0.0),
        'config_id': f"f{int(row['fold'])}_{row['tpsl_mode']}_t{row['prob_thresh']:.2f}_tp{row['tp_pct']:.2f}",
    }


def _pick_best(t_df, col='val_a_avg_pnl'):
    if len(t_df) == 0:
        return None
    best_idx = t_df[col].idxmax()
    return _bq_row_to_cfg(t_df.loc[best_idx])


# ── Dimension 1: Single-fold selection ──
def select_single_fold(bq_df, tickers, fold):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'] == fold)]
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker]
        cfg = _pick_best(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── Dimension 3: Ranking criteria on fold 403 ──
def select_by_ranking(bq_df, tickers, metric):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'] == 403)].copy()
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker].copy()
        if len(t_df) == 0:
            continue

        if metric == 'overfit':
            valid = t_df[t_df['val_a_avg_pnl'] > 0].copy()
            if len(valid) == 0:
                valid = t_df
            valid['_overfit_dist'] = (valid['va_opt_ratio'] - 1.0).abs()
            best_idx = valid['_overfit_dist'].idxmin()
            selections[ticker] = _bq_row_to_cfg(valid.loc[best_idx])
        elif metric == 'win_rate':
            cfg = _pick_best(t_df, 'val_a_win_rate')
            if cfg:
                selections[ticker] = cfg
        elif metric == 'composite':
            t_df['_composite'] = t_df['val_a_avg_pnl'] * np.minimum(1.0, 1.0 / t_df['va_opt_ratio'].clip(lower=0.01))
            cfg = _pick_best(t_df, '_composite')
            if cfg:
                selections[ticker] = cfg
    return selections


# ── Dimension 4: Ticker-specific fold ──
def select_ticker_specific(bq_df, tickers, mode):
    selections = {}
    for ticker in tickers:
        if mode == 'best_va':
            t_df = bq_df[bq_df['ticker'] == ticker]
        elif mode == 'best_a2':
            t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'].isin(FOLDS_A2))]
            if len(t_df) == 0:
                t_df = bq_df[bq_df['ticker'] == ticker]
        elif mode == 'median_va':
            t_df = bq_df[bq_df['ticker'] == ticker]
            if len(t_df) == 0:
                continue
            per_fold = t_df.groupby('fold')['val_a_avg_pnl'].max().sort_values()
            if len(per_fold) == 0:
                continue
            median_idx = len(per_fold) // 2
            median_fold = per_fold.index[median_idx]
            t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'] == median_fold)]
        else:
            continue

        if len(t_df) == 0:
            continue
        cfg = _pick_best(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ── Dimension 2 helper: ENS_TOP3_VA config selection ──
def select_ens_top3(bq_df, tickers):
    selections = {}
    top3_folds = {}
    for ticker in tickers:
        t_df = bq_df[bq_df['ticker'] == ticker]
        if len(t_df) == 0:
            continue
        per_fold = t_df.groupby('fold')['val_a_avg_pnl'].max().sort_values(ascending=False)
        top3 = list(per_fold.head(3).index)
        top3_folds[ticker] = top3
        best_fold = top3[0]
        fold_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'] == best_fold)]
        cfg = _pick_best(fold_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections, top3_folds


# ── Dimension 5: Cross-model helpers ──
def select_xmodel_blend(bq_df, tickers):
    selections = {}
    xmodel_folds = {}
    for ticker in tickers:
        a2_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'].isin(FOLDS_A2))]
        a2_cfg = _pick_best(a2_df, 'val_a_avg_pnl') if len(a2_df) > 0 else None
        a1_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'].isin(FOLDS_A1))]
        a1_cfg = _pick_best(a1_df, 'val_a_avg_pnl') if len(a1_df) > 0 else None

        if a2_cfg and a1_cfg:
            if a2_cfg['val_a_avg_pnl'] >= a1_cfg['val_a_avg_pnl']:
                selections[ticker] = a2_cfg
            else:
                selections[ticker] = a1_cfg
            xmodel_folds[ticker] = (a2_cfg['fold'], a1_cfg['fold'])
        elif a2_cfg:
            selections[ticker] = a2_cfg
            xmodel_folds[ticker] = (a2_cfg['fold'], None)
        elif a1_cfg:
            selections[ticker] = a1_cfg
            xmodel_folds[ticker] = (None, a1_cfg['fold'])
    return selections, xmodel_folds


def get_best_folds_per_family(bq_df, tickers):
    result = {}
    for ticker in tickers:
        t_df = bq_df[bq_df['ticker'] == ticker]
        if len(t_df) == 0:
            continue
        a2_df = t_df[t_df['fold'].isin(FOLDS_A2)]
        a2_best = int(a2_df.groupby('fold')['val_a_avg_pnl'].max().idxmax()) if len(a2_df) > 0 else 403
        a1_df = t_df[t_df['fold'].isin(FOLDS_A1)]
        a1_best = int(a1_df.groupby('fold')['val_a_avg_pnl'].max().idxmax()) if len(a1_df) > 0 else 410
        result[ticker] = (a2_best, a1_best)
    return result


def run_xmodel_vote_backtest(configs, all_fold_probs_map, ticker_data, label):
    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt   = pd.Timestamp(TEST_END).date()
    h2_start_dt   = pd.Timestamp(HALF2_START).date()

    all_trades = []
    per_ticker_trades = defaultdict(list)

    for ticker in sorted(configs.keys()):
        if ticker not in ticker_data:
            continue
        cfg = configs[ticker]
        ohlcv = ticker_data[ticker]
        prob_thresh = cfg['prob_thresh']
        mode = cfg['tpsl_mode']

        a2_dps = [all_fold_probs_map.get(ticker, {}).get(f, {})
                  for f in FOLDS_A2 if all_fold_probs_map.get(ticker, {}).get(f)]
        a1_dps = [all_fold_probs_map.get(ticker, {}).get(f, {})
                  for f in FOLDS_A1 if all_fold_probs_map.get(ticker, {}).get(f)]
        if not a2_dps or not a1_dps:
            continue

        prices = ohlcv['prices']
        dates  = ohlcv['dates']
        times  = ohlcv['times']
        atr_arr = ohlcv.get('atr_pct', np.full(len(prices), 0.05))

        entry_indices = np.where(times == cfg.get('entry_minute', ENTRY_MIN))[0]
        last_traded_date = -1

        for idx in entry_indices:
            d_int = int(dates[idx])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()

            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            if d_int == last_traded_date:
                continue

            a2_pass = any(dp.get(d_date, 0.0) >= prob_thresh for dp in a2_dps)
            a1_pass = any(dp.get(d_date, 0.0) >= prob_thresh for dp in a1_dps)
            if not (a2_pass and a1_pass):
                continue

            if mode == 'atr_sl':
                pnl, trade_date, filled = run_short_trade_atr_sl(
                    prices, ohlcv['highs'], ohlcv['lows'], dates, times,
                    idx, cfg['tp_pct'], float(atr_arr[idx]), cfg['sl_mult'], EXIT_TIME_MIN)
            elif mode == 'grace':
                pnl, trade_date, filled = run_short_trade_grace(
                    prices, ohlcv['highs'], ohlcv['lows'], dates, times,
                    idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['grace_minutes'])
            else:
                pnl, trade_date, filled = run_short_trade_static(
                    prices, ohlcv['highs'], ohlcv['lows'], dates, times,
                    idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)

            if not filled:
                continue

            pnl -= 2 * SLIPPAGE_PCT
            last_traded_date = d_int

            all_probs = [dp.get(d_date, 0.0) for dp in a2_dps + a1_dps]
            avg_prob = np.mean(all_probs)
            half = 'H2' if d_date >= h2_start_dt else 'H1'
            trade = {
                'date': d_date, 'ticker': ticker, 'pnl': pnl,
                'prob': avg_prob, 'close': float(prices[idx]),
                'mode': mode, 'tp': cfg['tp_pct'],
                'sl': cfg['sl_pct'] if mode != 'atr_sl' else cfg['sl_mult'],
                'half': half,
            }
            all_trades.append(trade)
            per_ticker_trades[ticker].append(trade)

    return all_trades, per_ticker_trades


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT M06_FOLD403 VARIATIONS BACKTEST — 25 Strategies + Oracle")
print(f"  TICKERS: {', '.join(sorted(SHORT_TICKERS))} (RULE_1)")
print(f"  TEST: {TEST_START} -> {TEST_END}  |  H2: {HALF2_START} -> {TEST_END}")
print(f"  ENTRY: 6:15 PST ({ENTRY_MIN} min)  |  EXIT: 1:00 PM ({EXIT_TIME_MIN} min)")
print(f"  SLIPPAGE: {SLIPPAGE_PCT:.2%} each way  |  CAPITAL: ${STARTING_CAPITAL:,}")
print(f"{'='*W}")


# ── Step 1: Load BQ configs ──────────────────────────────────────────────────
print(f"\n  STEP 1: Loading BQ configs (all 12 folds, ensemble)...")

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

bq_tickers = sorted(bq_df['ticker'].unique())
print(f"    BQ: {len(bq_df)} configs, {len(bq_tickers)} tickers, {bq_df['fold'].nunique()} folds")
print(f"    RULE_1 in BQ: {[t for t in SHORT_TICKERS if t in bq_tickers]}")


# ── Step 2: Run ALL config selection methods ──────────────────────────────────
print(f"\n  STEP 2: Selecting configs (25 strategies)...")

strategies = {}

# Dim 1: 12 single folds
for fold in ALL_FOLDS:
    strategies[f'FOLD_{fold}'] = select_single_fold(bq_df, SHORT_TICKERS, fold)

# Dim 2: ensemble strategies (fold-403 configs — prob variation happens later)
strategies['ENS_AVG_A2']  = select_single_fold(bq_df, SHORT_TICKERS, 403)
strategies['ENS_AVG_ALL'] = select_single_fold(bq_df, SHORT_TICKERS, 403)
strategies['ENS_VOTE_A2'] = select_single_fold(bq_df, SHORT_TICKERS, 403)
ens_top3_cfgs, ens_top3_folds = select_ens_top3(bq_df, SHORT_TICKERS)
strategies['ENS_TOP3_VA'] = ens_top3_cfgs

# Dim 3: ranking criteria on fold 403
for metric in ['overfit', 'win_rate', 'composite']:
    strategies[f'RANK_{metric.upper()}'] = select_by_ranking(bq_df, SHORT_TICKERS, metric)

# Dim 4: ticker-specific
for mode in ['best_va', 'best_a2', 'median_va']:
    strategies[f'TKR_{mode.upper()}'] = select_ticker_specific(bq_df, SHORT_TICKERS, mode)

# Dim 5: cross-model (combine alpha-2 and alpha-1 families)
strategies['XMODEL_VOTE']  = select_single_fold(bq_df, SHORT_TICKERS, 403)
strategies['XMODEL_AVG']   = select_single_fold(bq_df, SHORT_TICKERS, 403)
xmodel_blend_cfgs, xmodel_blend_folds = select_xmodel_blend(bq_df, SHORT_TICKERS)
strategies['XMODEL_BLEND'] = xmodel_blend_cfgs
xmodel_best_folds = get_best_folds_per_family(bq_df, SHORT_TICKERS)

for sn, cfgs in strategies.items():
    tks = sorted(cfgs.keys())
    print(f"    {sn:<16}: {len(tks)} tickers — {', '.join(tks)}")

strat_to_dim = {}
for dim, names in DIMENSIONS.items():
    for n in names:
        strat_to_dim[n] = dim
strat_to_dim['ORACLE'] = 'Oracle'


# ── Step 3: Load OHLCV data ──────────────────────────────────────────────────
print(f"\n  STEP 3: Loading OHLCV + ATR for {len(SHORT_TICKERS)} tickers...")
ticker_data = {}

for ticker in SHORT_TICKERS:
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
        print(f"    {ticker}: {n_days} trading days, ATR_pct={atr_cut.median():.4f}")
        del df
        _master_cache.pop(ticker, None)
    except Exception as ex:
        print(f"    {ticker}: OHLCV load failed -- {ex}")
        _master_cache.pop(ticker, None)
    gc.collect()

_release_memory()


# ── Step 4: Load fold-specific CSV scores ─────────────────────────────────────
print(f"\n  STEP 4: Loading CSV seed scores (all 12 folds)...")

all_fold_probs = defaultdict(dict)
for fold in sorted(ALL_FOLDS):
    exp = FOLD_EXPERIMENTS[fold]
    n_loaded = 0
    for ticker in SHORT_TICKERS:
        if ticker not in ticker_data:
            continue
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
        print(f"    Fold {fold}: {n_loaded} tickers loaded")
    _release_memory()

# Pre-compute ensemble averages
ensemble_probs_all = {}
ensemble_probs_a2 = {}
for ticker in ticker_data:
    fold_dps = all_fold_probs.get(ticker, {})
    if not fold_dps:
        continue

    all_dates = set()
    for dp in fold_dps.values():
        all_dates.update(dp.keys())

    ep_all = {}
    ep_a2 = {}
    for d in all_dates:
        vals_all = [dp[d] for f, dp in fold_dps.items() if d in dp]
        vals_a2  = [dp[d] for f, dp in fold_dps.items() if f in FOLDS_A2 and d in dp]
        if vals_all:
            ep_all[d] = np.mean(vals_all)
        if vals_a2:
            ep_a2[d] = np.mean(vals_a2)
    ensemble_probs_all[ticker] = ep_all
    ensemble_probs_a2[ticker] = ep_a2

# ENS_TOP3_VA: per-ticker top-3 folds averaged
ens_top3_probs = {}
for ticker in ticker_data:
    folds_for_ticker = ens_top3_folds.get(ticker, [])
    if not folds_for_ticker:
        ens_top3_probs[ticker] = ensemble_probs_all.get(ticker, {})
        continue
    all_dates = set()
    for f in folds_for_ticker:
        dp = all_fold_probs.get(ticker, {}).get(f, {})
        all_dates.update(dp.keys())
    ep = {}
    for d in all_dates:
        vals = [all_fold_probs[ticker][f][d] for f in folds_for_ticker
                if f in all_fold_probs.get(ticker, {}) and d in all_fold_probs[ticker][f]]
        if vals:
            ep[d] = np.mean(vals)
    ens_top3_probs[ticker] = ep

tickers_ready = sorted(set(ticker_data.keys()) & set(ensemble_probs_all.keys()))
print(f"\n    Ready: {len(tickers_ready)} tickers with OHLCV + probs")


# ── Step 5: Build per-strategy daily probs ────────────────────────────────────
print(f"\n  STEP 5: Building per-strategy daily probability maps...")

strategy_probs = {}
for sn, cfgs in strategies.items():
    dp_map = {}
    for ticker, cfg in cfgs.items():
        if ticker not in ticker_data:
            continue
        fold = cfg['fold']

        if sn == 'ENS_AVG_A2':
            dp_map[ticker] = ensemble_probs_a2.get(ticker, {})
        elif sn == 'ENS_AVG_ALL':
            dp_map[ticker] = ensemble_probs_all.get(ticker, {})
        elif sn == 'ENS_TOP3_VA':
            dp_map[ticker] = ens_top3_probs.get(ticker, {})
        elif sn == 'ENS_VOTE_A2':
            dp_map[ticker] = ensemble_probs_a2.get(ticker, {})
        elif sn.startswith('TKR_') or sn.startswith('FOLD_'):
            if fold is not None and fold in all_fold_probs.get(ticker, {}):
                dp_map[ticker] = all_fold_probs[ticker][fold]
            elif ticker in ensemble_probs_all:
                dp_map[ticker] = ensemble_probs_all[ticker]
        elif sn.startswith('RANK_'):
            fold_403_dp = all_fold_probs.get(ticker, {}).get(403, {})
            dp_map[ticker] = fold_403_dp if fold_403_dp else ensemble_probs_all.get(ticker, {})
        elif sn == 'XMODEL_AVG':
            a2_f, a1_f = xmodel_best_folds.get(ticker, (403, 410))
            a2_dp = all_fold_probs.get(ticker, {}).get(a2_f, {})
            a1_dp = all_fold_probs.get(ticker, {}).get(a1_f, {})
            xm_dates = set(a2_dp.keys()) | set(a1_dp.keys())
            avg_dp = {}
            for d in xm_dates:
                vals = []
                if d in a2_dp: vals.append(a2_dp[d])
                if d in a1_dp: vals.append(a1_dp[d])
                if vals: avg_dp[d] = np.mean(vals)
            dp_map[ticker] = avg_dp
        elif sn == 'XMODEL_BLEND':
            folds_pair = xmodel_blend_folds.get(ticker, (None, None))
            a2_dp = all_fold_probs.get(ticker, {}).get(folds_pair[0], {}) if folds_pair[0] else {}
            a1_dp = all_fold_probs.get(ticker, {}).get(folds_pair[1], {}) if folds_pair[1] else {}
            xm_dates = set(a2_dp.keys()) | set(a1_dp.keys())
            avg_dp = {}
            for d in xm_dates:
                vals = []
                if d in a2_dp: vals.append(a2_dp[d])
                if d in a1_dp: vals.append(a1_dp[d])
                if vals: avg_dp[d] = np.mean(vals)
            dp_map[ticker] = avg_dp
        elif sn == 'XMODEL_VOTE':
            dp_map[ticker] = ensemble_probs_all.get(ticker, {})
        else:
            if fold is not None and fold in all_fold_probs.get(ticker, {}):
                dp_map[ticker] = all_fold_probs[ticker][fold]
            elif ticker in ensemble_probs_all:
                dp_map[ticker] = ensemble_probs_all[ticker]

    strategy_probs[sn] = dp_map
    print(f"    {sn:<16}: {len(dp_map)} tickers")


# ── Step 6: Exhaustive oracle backtest ────────────────────────────────────────
print(f"\n  STEP 6: Running exhaustive oracle backtest (all configs x all tickers)...")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

all_ticker_configs = {}
for ticker in SHORT_TICKERS:
    t_df = bq_df[bq_df['ticker'] == ticker]
    if len(t_df) == 0:
        continue
    configs_list = []
    for _, bq_row in t_df.iterrows():
        configs_list.append({
            'fold': int(bq_row['fold']),
            'tpsl_mode': bq_row['tpsl_mode'],
            'prob_thresh': bq_row['prob_thresh'],
            'tp_pct': bq_row['tp_pct'],
            'sl_pct': bq_row['sl_pct'],
            'sl_mult': bq_row['sl_mult'],
            'grace_minutes': int(bq_row['grace_minutes']),
            'entry_minute': int(bq_row['entry_minute']) if bq_row['entry_minute'] > 0 else ENTRY_MIN,
            'config_id': f"f{int(bq_row['fold'])}_{bq_row['tpsl_mode']}_t{bq_row['prob_thresh']:.2f}_tp{bq_row['tp_pct']:.2f}",
        })
    configs_list.append(FIXED_UNIVERSAL_CFG.copy())
    all_ticker_configs[ticker] = configs_list

oracle_configs = {}
total_bt = 0

for ticker in tickers_ready:
    if ticker not in all_ticker_configs:
        continue
    ohlcv = ticker_data[ticker]
    best_h2_avg = -999
    best_cfg = None

    for cfg in all_ticker_configs[ticker]:
        if cfg['fold'] is not None:
            dp = all_fold_probs.get(ticker, {}).get(cfg['fold'], {})
            if not dp:
                dp = ensemble_probs_all.get(ticker, {})
        else:
            dp = ensemble_probs_all.get(ticker, {})
        if not dp:
            continue

        em = cfg['entry_minute'] if cfg['entry_minute'] > 0 else ENTRY_MIN
        thresh = cfg['prob_thresh']
        entry_indices = np.where(ohlcv['times'] == em)[0]
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
            if d_date < h2_start_dt:
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
                h2_trades_list.append(pnl)
            last_td = d_int

        total_bt += 1
        if len(h2_trades_list) >= MIN_H2_TRADES:
            h2_avg = np.mean(h2_trades_list)
            if h2_avg > best_h2_avg:
                best_h2_avg = h2_avg
                best_cfg = cfg.copy()

    if best_cfg is not None:
        oracle_configs[ticker] = best_cfg
        print(f"    {ticker}: oracle = f{best_cfg['fold']}_{best_cfg['tpsl_mode']} "
              f"TP={best_cfg['tp_pct']:.0%} thresh={best_cfg['prob_thresh']:.2f} -> H2 avg={best_h2_avg:+.2%}")

print(f"    Total configs backtested: {total_bt}")
print(f"    Oracle tickers: {len(oracle_configs)}")

strategies['ORACLE'] = oracle_configs

oracle_dp_map = {}
for ticker, cfg in oracle_configs.items():
    fold = cfg['fold']
    if fold is not None and fold in all_fold_probs.get(ticker, {}):
        oracle_dp_map[ticker] = all_fold_probs[ticker][fold]
    elif ticker in ensemble_probs_all:
        oracle_dp_map[ticker] = ensemble_probs_all[ticker]
strategy_probs['ORACLE'] = oracle_dp_map


# ── Step 7: Run ALL backtests ─────────────────────────────────────────────────
print(f"\n  STEP 7: Running backtests for 26 strategies...")

strategy_results = {}
strategy_per_tk = {}

all_strat_names = [f'FOLD_{f}' for f in ALL_FOLDS] + \
    ['ENS_AVG_A2', 'ENS_AVG_ALL', 'ENS_VOTE_A2', 'ENS_TOP3_VA'] + \
    ['RANK_OVERFIT', 'RANK_WIN_RATE', 'RANK_COMPOSITE'] + \
    ['TKR_BEST_VA', 'TKR_BEST_A2', 'TKR_MEDIAN_VA'] + \
    ['XMODEL_VOTE', 'XMODEL_AVG', 'XMODEL_BLEND'] + \
    ['ORACLE']

for sn in all_strat_names:
    np.random.seed(42)
    cfgs = strategies[sn]
    probs = strategy_probs[sn]

    if sn == 'ENS_VOTE_A2':
        trades, per_tk = run_vote_backtest(cfgs, all_fold_probs, FOLDS_A2, ticker_data, sn)
    elif sn == 'XMODEL_VOTE':
        trades, per_tk = run_xmodel_vote_backtest(cfgs, all_fold_probs, ticker_data, sn)
    else:
        trades, per_tk = run_csv_backtest(cfgs, probs, ticker_data, sn)

    port = summarize_trades(trades, sn)
    strategy_results[sn] = port
    strategy_per_tk[sn] = per_tk
    dim = strat_to_dim.get(sn, '?')
    print(f"    [{dim[:6]:>6}] {sn:<16}: {port['n']:>3} trades, avg={port['avg']:+.2%}, "
          f"wr={port['wr']:.0%}, total={port['total']:+.1%}")


# ==============================================================================
# RESULTS TABLES
# ==============================================================================
oracle_avg = strategy_results['ORACLE']['avg'] if strategy_results['ORACLE']['avg'] != 0 else 1
baseline_avg = strategy_results['FOLD_403']['avg']

sorted_strats = sorted(all_strat_names, key=lambda sn: strategy_results[sn]['avg'], reverse=True)


# ── Table 1: Overall Ranking ──
print(f"\n{'='*W}")
print(f"  TABLE 1: OVERALL RANKING — All 26 Strategies ({TEST_START} -> {TEST_END})")
print(f"{'='*W}")

print(f"\n  {'Rank':>4} {'Strategy':<16} {'Dim':<10} {'Trades':>6} {'Total':>8} "
      f"{'Avg':>7} {'WR':>5} {'H2_Avg':>7} {'Oracle%':>7} {'vs403':>7}")
print(f"  {'─'*90}")

for rank, sn in enumerate(sorted_strats, 1):
    p = strategy_results[sn]
    dim = strat_to_dim.get(sn, '—')

    h2_trades = p['trades_df'][p['trades_df']['half'] == 'H2'] if p['n'] > 0 else pd.DataFrame()
    h2_avg = h2_trades['pnl'].mean() if len(h2_trades) > 0 else 0

    oracle_pct = (p['avg'] / oracle_avg * 100) if sn != 'ORACLE' else 100
    vs_403 = p['avg'] - baseline_avg if sn != 'FOLD_403' else 0

    marker = ' <-- BASELINE' if sn == 'FOLD_403' else ''
    marker = ' <-- ORACLE' if sn == 'ORACLE' else marker

    print(f"  {rank:>4} {sn:<16} {dim:<10} {p['n']:>6} {p['total']:>+8.1%} "
          f"{p['avg']:>+7.2%} {p['wr']:>5.0%} {h2_avg:>+7.2%} {oracle_pct:>6.0f}% {vs_403:>+7.2%}{marker}")


# ── Table 2: Per-Dimension Winners ──
print(f"\n{'='*W}")
print(f"  TABLE 2: PER-DIMENSION WINNERS")
print(f"{'='*W}")

print(f"\n  {'Dimension':<12} {'Winner':<16} {'Trades':>6} {'Avg':>7} {'H2_Avg':>7} "
      f"{'Oracle%':>7} {'vs FOLD_403':>11}")
print(f"  {'─'*75}")

for dim, strat_list in DIMENSIONS.items():
    valid = [(sn, strategy_results[sn]) for sn in strat_list if sn in strategy_results]
    if not valid:
        continue
    winner_sn, winner_p = max(valid, key=lambda x: x[1]['avg'])

    h2_trades = winner_p['trades_df'][winner_p['trades_df']['half'] == 'H2'] if winner_p['n'] > 0 else pd.DataFrame()
    h2_avg = h2_trades['pnl'].mean() if len(h2_trades) > 0 else 0
    oracle_pct = (winner_p['avg'] / oracle_avg * 100) if oracle_avg != 0 else 0
    vs_403 = winner_p['avg'] - baseline_avg

    print(f"  {dim:<12} {winner_sn:<16} {winner_p['n']:>6} {winner_p['avg']:>+7.2%} "
          f"{h2_avg:>+7.2%} {oracle_pct:>6.0f}% {vs_403:>+11.2%}")


# ── Table 3: Single-Fold Dimension Detail ──
print(f"\n{'='*W}")
print(f"  TABLE 3: SINGLE-FOLD COMPARISON (Dim 1)")
print(f"{'='*W}")

print(f"\n  {'Fold':>4} {'Alpha':>5} {'Trades':>6} {'Avg':>7} {'WR':>5} {'H2_Avg':>7} "
      f"{'Tickers':>7} {'vs403':>7}")
print(f"  {'─'*60}")

for fold in ALL_FOLDS:
    sn = f'FOLD_{fold}'
    p = strategy_results[sn]
    alpha = '2.0' if fold in FOLDS_A2 else '1.0'
    n_tk = len(strategy_per_tk[sn])

    h2_trades = p['trades_df'][p['trades_df']['half'] == 'H2'] if p['n'] > 0 else pd.DataFrame()
    h2_avg = h2_trades['pnl'].mean() if len(h2_trades) > 0 else 0
    vs_403 = p['avg'] - baseline_avg

    marker = ' <--' if fold == 403 else ''
    print(f"  {fold:>4} {alpha:>5} {p['n']:>6} {p['avg']:>+7.2%} {p['wr']:>5.0%} "
          f"{h2_avg:>+7.2%} {n_tk:>5}/6 {vs_403:>+7.2%}{marker}")


# ── Table 4: Per-Ticker Best Strategy ──
print(f"\n{'='*W}")
print(f"  TABLE 4: PER-TICKER BEST STRATEGY")
print(f"{'='*W}")

print(f"\n  {'Ticker':<7} {'Best Strategy':<16} {'Best Avg':>8} {'FOLD_403':>8} {'Delta':>7} {'Best N':>6}")
print(f"  {'─'*58}")

for ticker in sorted(SHORT_TICKERS):
    best_sn = None
    best_avg = -999
    for sn in all_strat_names:
        if sn == 'ORACLE':
            continue
        trades = strategy_per_tk[sn].get(ticker, [])
        if not trades:
            continue
        avg = np.mean([t['pnl'] for t in trades])
        if avg > best_avg:
            best_avg = avg
            best_sn = sn

    f403_trades = strategy_per_tk['FOLD_403'].get(ticker, [])
    f403_avg = np.mean([t['pnl'] for t in f403_trades]) if f403_trades else 0
    delta = best_avg - f403_avg if best_sn else 0
    best_n = len(strategy_per_tk.get(best_sn, {}).get(ticker, [])) if best_sn else 0

    print(f"  {ticker:<7} {(best_sn or '—'):<16} {best_avg:>+8.2%} {f403_avg:>+8.2%} "
          f"{delta:>+7.2%} {best_n:>6}")


# ── Table 5: H1/H2 Split — Top 8 ──
print(f"\n{'='*W}")
print(f"  TABLE 5: H1 vs H2 SPLIT — Top 8 Strategies")
print(f"{'='*W}")

print(f"\n  {'Strategy':<16} {'Dim':<10} | {'-- H1 --':} | {'-- H2 --':} | {'Edge':>5}")
print(f"  {'':26} | {'N':>4} {'Avg':>7} {'WR':>5} | {'N':>4} {'Avg':>7} {'WR':>5} |")
print(f"  {'─'*75}")

for sn in sorted_strats[:8]:
    dim = strat_to_dim.get(sn, '—')
    trades_df = strategy_results[sn]['trades_df']
    if len(trades_df) == 0:
        continue
    h1 = trades_df[trades_df['half'] == 'H1']
    h2 = trades_df[trades_df['half'] == 'H2']

    h1_n = len(h1); h1_avg = h1['pnl'].mean() if h1_n > 0 else 0
    h1_wr = (h1['pnl'] > 0).mean() if h1_n > 0 else 0
    h2_n = len(h2); h2_avg = h2['pnl'].mean() if h2_n > 0 else 0
    h2_wr = (h2['pnl'] > 0).mean() if h2_n > 0 else 0

    edge = 'UP' if h2_avg > h1_avg else ('DOWN' if h2_avg < h1_avg else 'FLAT')
    print(f"  {sn:<16} {dim:<10} | {h1_n:>4} {h1_avg:>+7.2%} {h1_wr:>5.0%} | "
          f"{h2_n:>4} {h2_avg:>+7.2%} {h2_wr:>5.0%} | {edge:>5}")


# ==============================================================================
# CHARTS (3x2 layout)
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(3, 2, figsize=(22, 20))


# ── Chart 1: Overall Ranking — horizontal bar ──
ax1 = axes[0, 0]
y_pos = np.arange(len(sorted_strats))
avgs = [strategy_results[sn]['avg'] * 100 for sn in sorted_strats]
colors = [DIM_COLORS.get(strat_to_dim.get(sn, '—'), '#666') for sn in sorted_strats]

ax1.barh(y_pos, avgs, color=colors, alpha=0.85, height=0.7)
ax1.axvline(x=DEPLOY_FLOOR * 100, color='red', linewidth=1.5, linestyle='--', alpha=0.7, label=f'Deploy ({DEPLOY_FLOOR:.1%})')
ax1.axvline(x=0, color='gray', linewidth=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(sorted_strats, fontsize=8)
ax1.invert_yaxis()
ax1.set_xlabel('Avg PnL per Trade (%)')
ax1.set_title('Overall Ranking — All 26 Strategies', fontsize=13, fontweight='bold')

for dim, color in DIM_COLORS.items():
    ax1.barh([], [], color=color, label=dim)
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3, axis='x')


# ── Chart 2: Per-Dimension Winners — cumulative PnL ──
ax2 = axes[0, 1]

dim_winners = {}
for dim, strat_list in DIMENSIONS.items():
    valid = [(sn, strategy_results[sn]) for sn in strat_list if sn in strategy_results]
    if valid:
        winner_sn = max(valid, key=lambda x: x[1]['avg'])[0]
        dim_winners[dim] = winner_sn

plot_strats = list(dim_winners.values()) + ['FOLD_403', 'ORACLE']
plot_strats = list(dict.fromkeys(plot_strats))

for sn in plot_strats:
    p = strategy_results[sn]
    if p['n'] == 0:
        continue
    df_t = p['trades_df'].sort_values('date')
    cum = df_t['pnl'].cumsum() * 100
    dim = strat_to_dim.get(sn, 'Oracle')
    ls = '--' if sn == 'ORACLE' else ('-.' if sn == 'FOLD_403' else '-')
    lw = 1.5 if sn in ('ORACLE', 'FOLD_403') else 2.5
    color = DIM_COLORS.get(dim, '#666')
    label = f'{sn} ({dim})'
    ax2.plot(pd.to_datetime(df_t['date'].values), cum.values,
             color=color, linewidth=lw, linestyle=ls, label=label)

ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax2.axvline(x=pd.Timestamp(HALF2_START), color='red', linewidth=1, linestyle=':', alpha=0.5)
ax2.set_ylabel('Cumulative PnL (%)')
ax2.set_title('Dimension Winners + Baseline — Cumulative PnL', fontsize=13, fontweight='bold')
ax2.legend(fontsize=7, loc='best')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


# ── Chart 3: Single-Fold Comparison — bar chart ──
ax3 = axes[1, 0]
fold_names = [f'{f}' for f in ALL_FOLDS]
fold_avgs = [strategy_results[f'FOLD_{f}']['avg'] * 100 for f in ALL_FOLDS]
fold_colors = ['#1565C0' if f in FOLDS_A2 else '#E65100' for f in ALL_FOLDS]
x = np.arange(len(ALL_FOLDS))

bars = ax3.bar(x, fold_avgs, color=fold_colors, alpha=0.85, width=0.7)
ax3.axhline(y=DEPLOY_FLOOR * 100, color='red', linewidth=1, linestyle='--', alpha=0.7, label=f'Deploy ({DEPLOY_FLOOR:.1%})')
ax3.axhline(y=baseline_avg * 100, color='green', linewidth=1.5, linestyle=':', alpha=0.7, label=f'FOLD_403 ({baseline_avg:+.2%})')
ax3.axhline(y=0, color='gray', linewidth=0.5)
ax3.set_xticks(x)
ax3.set_xticklabels(fold_names, fontsize=9)
ax3.set_ylabel('Avg PnL per Trade (%)')
ax3.set_xlabel('Fold')
ax3.set_title('Dimension 1: Single-Fold Comparison', fontsize=13, fontweight='bold')
ax3.bar([], [], color='#1565C0', label='Alpha-2 (400-405)')
ax3.bar([], [], color='#E65100', label='Alpha-1 (410-415)')
ax3.legend(fontsize=8, loc='best')
ax3.grid(True, alpha=0.3, axis='y')

for i, (v, f) in enumerate(zip(fold_avgs, ALL_FOLDS)):
    if f == 403:
        bars[i].set_edgecolor('green')
        bars[i].set_linewidth(3)


# ── Chart 4: Dimensions 2-4 Detail — grouped bar ──
ax4 = axes[1, 1]
dim2345_strats = DIMENSIONS['Ensemble'] + DIMENSIONS['Ranking'] + DIMENSIONS['TickerSpec'] + DIMENSIONS['CrossModel']
d2345_avgs = [strategy_results[sn]['avg'] * 100 for sn in dim2345_strats]
d2345_colors = []
for sn in dim2345_strats:
    dim = strat_to_dim[sn]
    d2345_colors.append(DIM_COLORS[dim])

x = np.arange(len(dim2345_strats))
ax4.bar(x, d2345_avgs, color=d2345_colors, alpha=0.85, width=0.7)
ax4.axhline(y=DEPLOY_FLOOR * 100, color='red', linewidth=1, linestyle='--', alpha=0.7)
ax4.axhline(y=baseline_avg * 100, color='green', linewidth=1.5, linestyle=':', alpha=0.7, label=f'FOLD_403 ({baseline_avg:+.2%})')
ax4.axhline(y=0, color='gray', linewidth=0.5)
ax4.set_xticks(x)
ax4.set_xticklabels([sn.replace('ENS_', '').replace('RANK_', '').replace('TKR_', '').replace('XMODEL_', 'XM_')
                      for sn in dim2345_strats], fontsize=7, rotation=35, ha='right')
ax4.set_ylabel('Avg PnL per Trade (%)')
ax4.set_title('Dims 2-5: Ensemble / Ranking / Ticker / CrossModel', fontsize=12, fontweight='bold')

for dim in ['Ensemble', 'Ranking', 'TickerSpec', 'CrossModel']:
    ax4.bar([], [], color=DIM_COLORS[dim], label=dim)
ax4.legend(fontsize=7, loc='best')
ax4.grid(True, alpha=0.3, axis='y')


# ── Chart 5: Per-Ticker Heatmap — top 8 strategies × 6 tickers ──
ax5 = axes[2, 0]
top8 = sorted_strats[:8]
heat_data = np.zeros((len(top8), len(SHORT_TICKERS)))
for i, sn in enumerate(top8):
    for j, ticker in enumerate(sorted(SHORT_TICKERS)):
        trades = strategy_per_tk[sn].get(ticker, [])
        heat_data[i, j] = np.mean([t['pnl'] for t in trades]) * 100 if trades else 0

vmax = max(abs(heat_data.min()), abs(heat_data.max()))
if vmax == 0:
    vmax = 1
im = ax5.imshow(heat_data, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
ax5.set_yticks(range(len(top8)))
ax5.set_yticklabels(top8, fontsize=9)
ax5.set_xticks(range(len(SHORT_TICKERS)))
ax5.set_xticklabels(sorted(SHORT_TICKERS), fontsize=10)
for i in range(len(top8)):
    for j in range(len(SHORT_TICKERS)):
        ax5.text(j, i, f'{heat_data[i,j]:+.1f}%', ha='center', va='center',
                 fontsize=9, fontweight='bold',
                 color='white' if abs(heat_data[i,j]) > vmax * 0.6 else 'black')
fig.colorbar(im, ax=ax5, label='Avg PnL (%)', shrink=0.8)
ax5.set_title('Per-Ticker Heatmap — Top 8 Strategies', fontsize=13, fontweight='bold')


# ── Chart 6: H1 vs H2 — Top 5 ──
ax6 = axes[2, 1]
top5 = sorted_strats[:5]
x = np.arange(len(top5))
bar_w = 0.35
h1_avgs = []
h2_avgs = []
for sn in top5:
    trades_df = strategy_results[sn]['trades_df']
    if len(trades_df) == 0:
        h1_avgs.append(0)
        h2_avgs.append(0)
        continue
    h1 = trades_df[trades_df['half'] == 'H1']
    h2 = trades_df[trades_df['half'] == 'H2']
    h1_avgs.append(h1['pnl'].mean() * 100 if len(h1) > 0 else 0)
    h2_avgs.append(h2['pnl'].mean() * 100 if len(h2) > 0 else 0)

ax6.bar(x - bar_w/2, h1_avgs, bar_w, label='H1 (Apr-May)', color='#90CAF9', alpha=0.85)
ax6.bar(x + bar_w/2, h2_avgs, bar_w, label='H2 (Jun-Jul)', color='#1565C0', alpha=0.85)
ax6.axhline(y=DEPLOY_FLOOR * 100, color='red', linewidth=1, linestyle='--', alpha=0.7)
ax6.set_xticks(x)
ax6.set_xticklabels(top5, fontsize=9, rotation=15)
ax6.set_ylabel('Avg PnL per Trade (%)')
ax6.set_title('H1 vs H2 — Top 5 Strategies', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
ax6.axhline(y=0, color='gray', linewidth=0.5)
ax6.grid(True, alpha=0.3, axis='y')


fig.suptitle(f'SHORT M06 Variations — RULE_1 ({len(SHORT_TICKERS)} tickers)\n'
             f'{TEST_START} -> {TEST_END}  |  25 strategies + Oracle  |  Deploy: {DEPLOY_FLOOR:.1%}',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/tmp/backtest_short_variations.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Chart saved: /tmp/backtest_short_variations.png")


# ==============================================================================
# FINAL VERDICT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

best_non_oracle = sorted_strats[1] if sorted_strats[0] == 'ORACLE' else sorted_strats[0]
best_p = strategy_results[best_non_oracle]
baseline_p = strategy_results['FOLD_403']
oracle_p = strategy_results['ORACLE']

print(f"\n  ORACLE:        {oracle_p['n']} trades, avg={oracle_p['avg']:+.2%}, wr={oracle_p['wr']:.0%}")
print(f"  BEST OVERALL:  {best_non_oracle} -> {best_p['n']} trades, avg={best_p['avg']:+.2%}, "
      f"wr={best_p['wr']:.0%}, oracle={best_p['avg']/oracle_avg*100:.0f}%")
print(f"  BASELINE:      FOLD_403 -> {baseline_p['n']} trades, avg={baseline_p['avg']:+.2%}, "
      f"wr={baseline_p['wr']:.0%}, oracle={baseline_p['avg']/oracle_avg*100:.0f}%")

beats_baseline = best_p['avg'] > baseline_p['avg']
improvement = best_p['avg'] - baseline_p['avg']
print(f"\n  {'BEATS' if beats_baseline else 'DOES NOT BEAT'} FOLD_403 baseline by {improvement:+.2%}")

print(f"\n  Per-dimension winners:")
for dim, strat_list in DIMENSIONS.items():
    valid = [(sn, strategy_results[sn]) for sn in strat_list if sn in strategy_results]
    if not valid:
        continue
    winner_sn, winner_p = max(valid, key=lambda x: x[1]['avg'])
    vs_403 = winner_p['avg'] - baseline_avg
    print(f"    {dim:<12}: {winner_sn:<16} avg={winner_p['avg']:+.2%} (vs 403: {vs_403:+.2%})")

print(f"\n  LONG S3 reference:  +2.23% avg")
print(f"  Deploy threshold:   {DEPLOY_FLOOR:.1%}")
print(f"{'='*W}")
