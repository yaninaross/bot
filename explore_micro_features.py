"""
explore_micro_features.py — Microstructure Feature Exploration for SHORT Ticker Selection
==========================================================================================
24 new microstructure features (from T&Q-derived columns in master_inference)
+ 25 existing features (OHLCV + signal + BQ structural) = 49 total.
Tested against TEST ORACLE (top-8 by Jun-Jul total PnL, avg_win >= 1%).
Lead time analysis at Apr 1 / May 1 / Jun 1.
7 analysis sections + 2x3 chart panel. CSV scoring only. Colab-ready.
"""

import os, io, gc, sys, warnings, ctypes
from datetime import datetime, timedelta, date as date_type
from collections import defaultdict
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
except (ImportError, ModuleNotFoundError):
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from google.cloud import storage as gcs_storage, bigquery
from scipy.stats import spearmanr, skew

if _IN_COLAB:
    try:
        auth.authenticate_user()
    except Exception as _auth_err:
        print(f"  Auth note: {_auth_err} — continuing with default credentials")

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME    = "time-series-bucket-ys"
RESULTS_DIR    = "Algos/Results_all_v2"
INFERENCE_DIR  = "Data/Inference_v2"

TEST_START = '2026-04-01'
TEST_END   = '2026-07-23'
HALF1_END  = '2026-05-31'
HALF2_START = '2026-06-01'

EXIT_TIME_MIN  = 780
SLIPPAGE_PCT   = 0.002
MIN_AVG_WIN    = 0.01

MORNING_START_MIN = 375
MORNING_END_MIN   = 450

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1')
BQ_ALLOWED_MODES  = ('static', 'atr_sl', 'grace')

TOP_K = 8
FILTERED_VA_OPT_RATIO = 2.0
MIN_SEEDS = 3
OHLCV_LOOKBACK_DAYS = 45

FEATURE_DATES = ['2026-04-01', '2026-05-01', '2026-06-01']

CHART_PATH = '/tmp/explore_micro_features.png'

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110


# ==============================================================================
# REUSED HELPERS (from analyze_oracle_reverse.py)
# ==============================================================================
_MODE_SUFFIXES = ('_ATRSL', '_GRACE', '_LIMIT', '_ATR', '_ATRADP',
                  '_REGIME', '_TRAIL', '_REGADP')

def _base_experiment(exp):
    for sfx in sorted(_MODE_SUFFIXES, key=len, reverse=True):
        if exp.endswith(sfx):
            return exp[:-len(sfx)]
    return exp

def _sf(val, d=0.0):
    try:
        if pd.isna(val): return d
    except (TypeError, ValueError): pass
    try: return float(val)
    except (TypeError, ValueError): return d

def _si(val, d=0):
    try:
        if pd.isna(val): return d
    except (TypeError, ValueError): pass
    try: return int(val)
    except (TypeError, ValueError): return d

def _row_to_cfg(ticker, row, source):
    exp = row['experiment'] if isinstance(row, pd.Series) else row.get('experiment', '')
    base_exp = _base_experiment(exp)
    return {
        'ticker': ticker, 'experiment': exp,
        '_base_experiment': base_exp,
        'alpha': base_exp.split('_')[0],
        'source': source,
        'tpsl_mode': row['tpsl_mode'],
        'entry_minute': _si(row.get('entry_minute', 375), 375),
        'prob_thresh': _sf(row['prob_thresh']),
        'tp_pct': _sf(row.get('tp_pct', 0)),
        'sl_pct': _sf(row.get('sl_pct', 0)),
        'sl_mult': _sf(row.get('sl_mult', 0)),
        'grace_minutes': _si(row.get('grace_minutes', 0)),
        'val_pnl': _sf(row.get('val_a_avg_pnl', row.get('mean_va_pnl', 0))),
        'test_pnl': _sf(row.get('val_b_avg_pnl', row.get('vb_pnl', 0))),
    }


# ==============================================================================
# DATA LOADING
# ==============================================================================
_master_cache = {}

MICRO_VOL_COLS = ['Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']
MICRO_QUOTE_COLS = ['Spread', 'OBI', 'bsize', 'asize']

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
        cutoff = pd.Timestamp(TEST_START, tz='UTC') - pd.Timedelta(days=OHLCV_LOOKBACK_DAYS)
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
    for c in ['Volume'] + MICRO_VOL_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    for c in MICRO_QUOTE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# CSV SCORES
# ==============================================================================
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

def compute_daily_probs(ticker, cfg, csv_scores_cache):
    base_exp = cfg['_base_experiment']
    entry_min = cfg['entry_minute']
    seed_scores = csv_scores_cache.get((ticker, base_exp))
    if not seed_scores:
        return {}
    all_daily = {}
    for seed_id, sdf in seed_scores.items():
        bar_mins = sdf.index.hour * 60 + sdf.index.minute
        entry_bars = sdf[bar_mins == entry_min]
        for dt, row in entry_bars.iterrows():
            d = dt.date()
            if d not in all_daily:
                all_daily[d] = []
            all_daily[d].append(float(row['Prob_Spike']))
    return {d: np.mean(ps) for d, ps in all_daily.items()}


# ==============================================================================
# OHLCV + ATR
# ==============================================================================
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

def get_daily_ohlcv(ticker):
    df = _master_cache.get(ticker)
    if df is None:
        return None
    idx = df.index
    if str(idx.tz) != 'America/Los_Angeles':
        df = df.copy()
        df.index = df.index.tz_convert('America/Los_Angeles')
    daily = df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum',
    }).dropna(subset=['Close'])
    daily = daily[daily['Volume'] > 0]
    return daily


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

def run_single_day_trade(ticker, trade_date, cfg, ohlcv, daily_probs, atr_arr):
    prob = daily_probs.get(trade_date)
    if prob is None or prob < cfg['prob_thresh']:
        return None
    entry_min = cfg['entry_minute']
    dates = ohlcv['dates']
    times = ohlcv['times']
    prices = ohlcv['prices']
    highs = ohlcv['highs']
    lows = ohlcv['lows']
    d_int = trade_date.year * 10000 + trade_date.month * 100 + trade_date.day
    candidates = np.where((times == entry_min) & (dates == d_int))[0]
    if len(candidates) == 0:
        return None
    idx = candidates[0]
    mode = cfg['tpsl_mode']
    if mode == 'atr_sl':
        atr_val = float(atr_arr[idx]) if idx < len(atr_arr) else 0.05
        pnl, _, filled = run_short_trade_atr_sl(
            prices, highs, lows, dates, times,
            idx, cfg['tp_pct'], atr_val, cfg['sl_mult'], EXIT_TIME_MIN)
    elif mode == 'grace':
        pnl, _, filled = run_short_trade_grace(
            prices, highs, lows, dates, times,
            idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['grace_minutes'])
    else:
        pnl, _, filled = run_short_trade_static(
            prices, highs, lows, dates, times,
            idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)
    if not filled:
        return None
    pnl -= 2 * SLIPPAGE_PCT
    return {
        'date': trade_date, 'ticker': ticker, 'pnl': pnl,
        'prob': prob, 'close': float(prices[idx]),
        'mode': mode, 'tp': cfg['tp_pct'],
        'sl': cfg['sl_pct'] if mode != 'atr_sl' else cfg['sl_mult'],
    }


# ==============================================================================
# BQ + CONFIG SELECTION
# ==============================================================================
def query_all_bq():
    versions_sql = ', '.join(f"'{v}'" for v in BQ_MODEL_VERSIONS)
    q = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           grace_minutes,
           opt_avg_pnl, opt_win_rate, opt_count,
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
    return bq_client.query(q).to_dataframe()

GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct', 'sl_mult', 'grace_minutes']

def select_configs_perseed(bq_df, all_tickers):
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = bq_df[bq_df['is_ensemble'] != True].copy()
    indiv = indiv[indiv['tpsl_mode'].isin(BQ_ALLOWED_MODES)]
    indiv['va_opt_ratio'] = indiv['val_a_avg_pnl'] / indiv['opt_avg_pnl']
    indiv = indiv[indiv['va_opt_ratio'] <= FILTERED_VA_OPT_RATIO]
    for k in GRP_KEYS:
        if k in indiv.columns and indiv[k].isna().any():
            indiv[k] = indiv[k].fillna(0)
    agg = indiv.groupby(GRP_KEYS).agg(
        n_seeds=('seed', 'nunique'),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
    ).reset_index()
    configs = {}
    for ticker in all_tickers:
        tk = agg[agg['ticker'] == ticker]
        strict = tk[tk['n_seeds'] >= MIN_SEEDS]
        if strict.empty:
            strict = tk[tk['n_seeds'] >= 1]
        if strict.empty:
            continue
        best = strict.loc[strict['mean_va_pnl'].idxmax()]
        row_data = best.to_dict()
        row_data['val_a_avg_pnl'] = row_data.pop('mean_va_pnl', 0)
        row_data['opt_avg_pnl'] = row_data.get('opt_avg_pnl', 1)
        configs[ticker] = _row_to_cfg(ticker, row_data, 'perseed')
    return configs

def compute_vb_ranks(bq_df):
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = bq_df[bq_df['is_ensemble'] != True].copy()
    return indiv.groupby('ticker')['val_b_avg_pnl'].median()

def compute_static_bq_features(bq_df):
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = bq_df[bq_df['is_ensemble'] != True].copy()
    indiv['va_opt_ratio'] = indiv['val_a_avg_pnl'] / indiv['opt_avg_pnl']
    rows = []
    for ticker, grp in indiv.groupby('ticker'):
        n = len(grp)
        grp = grp.copy()
        for k in GRP_KEYS:
            if k in grp.columns and grp[k].isna().any():
                grp[k] = grp[k].fillna(0)
        seed_agg = grp.groupby(GRP_KEYS).agg(
            n_seeds=('seed', 'nunique'),
            mean_va_pnl=('val_a_avg_pnl', 'mean'),
        ).reset_index()
        seed_agg['agree_18'] = 0.0
        for idx_sa, row_sa in seed_agg.iterrows():
            mask = pd.Series(True, index=grp.index)
            for k in GRP_KEYS:
                mask = mask & (grp[k] == row_sa[k])
            sub = grp[mask]
            n_pos = (sub['val_a_avg_pnl'] >= 0.018).sum()
            seed_agg.at[idx_sa, 'agree_18'] = n_pos / row_sa['n_seeds'] if row_sa['n_seeds'] > 0 else 0
        exps = grp['experiment'].apply(_base_experiment).unique()
        exp_400 = [e for e in exps if e.startswith('400')]
        exp_410 = [e for e in exps if e.startswith('410')]
        has_both = len(exp_400) > 0 and len(exp_410) > 0
        cross_agree = 0.0
        if has_both:
            a400 = seed_agg[seed_agg['experiment'].apply(
                lambda e: _base_experiment(e).startswith('400'))]
            a410 = seed_agg[seed_agg['experiment'].apply(
                lambda e: _base_experiment(e).startswith('410'))]
            best_400 = a400['agree_18'].max() if not a400.empty else 0
            best_410 = a410['agree_18'].max() if not a410.empty else 0
            cross_agree = min(best_400, best_410)
        rows.append({
            'ticker': ticker,
            'n_configs': n,
            'has_both_alphas': int(has_both),
            'cross_alpha_agree': cross_agree,
            'agree_18_best': seed_agg['agree_18'].max() if not seed_agg.empty else 0,
            'opt_wr_median': grp['opt_win_rate'].median(),
            'pct_static': (grp['tpsl_mode'] == 'static').sum() / n,
            'pct_atr_sl': (grp['tpsl_mode'] == 'atr_sl').sum() / n,
            'pct_grace': (grp['tpsl_mode'] == 'grace').sum() / n,
            'pct_limit': (grp['tpsl_mode'] == 'limit_entry').sum() / n,
            'va_median': grp['val_a_avg_pnl'].median(),
            'va_pos_rate': (grp['val_a_avg_pnl'] >= 0.01).mean(),
            'va_opt_ratio_med': grp['va_opt_ratio'].median(),
        })
    return pd.DataFrame(rows).set_index('ticker').fillna(0)


# ==============================================================================
# PAPER TRADING + TEST ORACLE
# ==============================================================================
def paper_trade_all_tickers(tradeable_tickers, trading_days, configs,
                            ticker_data, daily_probs_cache):
    all_trades = defaultdict(list)
    for day in trading_days:
        np.random.seed(42 + day.toordinal())
        for ticker in tradeable_tickers:
            if ticker not in configs or ticker not in ticker_data:
                continue
            if ticker not in daily_probs_cache:
                continue
            cfg = configs[ticker]
            ohlcv = ticker_data[ticker]['ohlcv']
            atr_arr = ohlcv.get('atr_pct', np.full(len(ohlcv['prices']), 0.05))
            trade = run_single_day_trade(
                ticker, day, cfg, ohlcv, daily_probs_cache[ticker], atr_arr)
            if trade is not None:
                all_trades[ticker].append(trade)
    return all_trades

def compute_test_oracle(all_trades, top_k, min_avg_win=0.01):
    half2_start = pd.Timestamp(HALF2_START).date()
    ticker_stats = []
    for ticker, trades in all_trades.items():
        h2 = [t['pnl'] for t in trades if t['date'] >= half2_start]
        if not h2:
            continue
        wins = [p for p in h2 if p > 0]
        avg_win = np.mean(wins) if wins else 0
        if avg_win < min_avg_win:
            continue
        ticker_stats.append({
            'ticker': ticker,
            'h2_total_pnl': sum(h2),
            'h2_n': len(h2),
            'h2_avg': np.mean(h2),
            'h2_wr': float(np.mean(np.array(h2) > 0)),
            'avg_win': avg_win,
        })
    ticker_stats.sort(key=lambda x: x['h2_total_pnl'], reverse=True)
    return ticker_stats[:top_k], ticker_stats

def generate_weeks(start_str, end_str):
    start = pd.Timestamp(start_str).date()
    end = pd.Timestamp(end_str).date()
    weeks = []
    current = start
    while current <= end:
        if current.weekday() <= 4:
            week_monday = current - timedelta(days=current.weekday())
            if week_monday < start: week_monday = start
            week_friday = week_monday + timedelta(days=4)
            if week_friday > end: week_friday = end
            week_entry = (week_monday, week_friday)
            if not weeks or weeks[-1] != week_entry:
                weeks.append(week_entry)
            current = week_friday + timedelta(days=3)
        else:
            current += timedelta(days=1)
    return weeks


# ==============================================================================
# EXISTING FEATURES (OHLCV + Signal)
# ==============================================================================
FEATURE_NAMES_OHLCV = [
    'ret_5d', 'ret_10d', 'ret_20d',
    'rvol_5d', 'rvol_20d', 'rvol_ratio',
    'drawdown_20d', 'volume_ratio', 'intraday_range_5d',
]
FEATURE_NAMES_SIGNAL = [
    'signal_freq_20d', 'signal_avg_prob_20d',
    'signal_freq_5d', 'signal_prob_trend',
]

def compute_ohlcv_features(ticker, as_of_date):
    daily = get_daily_ohlcv(ticker)
    if daily is None or len(daily) < 5:
        return {f: 0.0 for f in FEATURE_NAMES_OHLCV}
    cutoff = pd.Timestamp(as_of_date).tz_localize('America/Los_Angeles')
    d = daily[daily.index < cutoff]
    if len(d) < 5:
        return {f: 0.0 for f in FEATURE_NAMES_OHLCV}
    close = d['Close'].values
    high = d['High'].values
    low = d['Low'].values
    vol = d['Volume'].values
    def _ret(n):
        if len(close) < n + 1: return 0.0
        return (close[-1] - close[-(n+1)]) / close[-(n+1)] if close[-(n+1)] != 0 else 0.0
    def _rvol(n):
        if len(close) < n + 1: return 0.0
        rets = np.diff(close[-n-1:]) / close[-n-1:-1]
        rets = rets[np.isfinite(rets)]
        return float(np.std(rets)) if len(rets) > 1 else 0.0
    rvol_5 = _rvol(5)
    rvol_20 = _rvol(20)
    high_20 = np.max(high[-20:]) if len(high) >= 20 else np.max(high)
    dd_20 = (close[-1] - high_20) / high_20 if high_20 != 0 else 0.0
    vol_5 = np.mean(vol[-5:]) if len(vol) >= 5 else np.mean(vol)
    vol_20 = np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol)
    vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0
    n_range = min(5, len(close))
    intra = np.mean((high[-n_range:] - low[-n_range:]) / close[-n_range:]) \
        if np.all(close[-n_range:] != 0) else 0.0
    return {
        'ret_5d': _ret(5), 'ret_10d': _ret(10), 'ret_20d': _ret(20),
        'rvol_5d': rvol_5, 'rvol_20d': rvol_20,
        'rvol_ratio': rvol_5 / max(rvol_20, 1e-9),
        'drawdown_20d': dd_20, 'volume_ratio': vol_ratio,
        'intraday_range_5d': float(intra),
    }

def compute_signal_features(daily_probs, prob_thresh, as_of_date):
    if not daily_probs:
        return {f: 0.0 for f in FEATURE_NAMES_SIGNAL}
    cutoff = as_of_date
    past = {d: p for d, p in daily_probs.items() if d < cutoff}
    if not past:
        return {f: 0.0 for f in FEATURE_NAMES_SIGNAL}
    sorted_dates = sorted(past.keys(), reverse=True)
    last_20 = sorted_dates[:20]
    last_5 = sorted_dates[:5]
    fired_20 = [past[d] for d in last_20 if past[d] >= prob_thresh]
    fired_5 = [past[d] for d in last_5 if past[d] >= prob_thresh]
    freq_20 = len(fired_20) / max(len(last_20), 1)
    freq_5 = len(fired_5) / max(len(last_5), 1)
    return {
        'signal_freq_20d': freq_20,
        'signal_avg_prob_20d': float(np.mean(fired_20)) if fired_20 else 0.0,
        'signal_freq_5d': freq_5,
        'signal_prob_trend': freq_5 - freq_20,
    }


# ==============================================================================
# NEW: MICROSTRUCTURE FEATURES (24 features)
# ==============================================================================
FEATURE_NAMES_MICRO = [
    'flow_mean_5d', 'flow_mean_20d', 'flow_persistence_20d',
    'flow_std_20d', 'flow_skew_20d', 'abs_flow_mean_5d',
    'spread_mean_5d', 'spread_mean_20d', 'spread_cv_20d',
    'spread_rel', 'avg_trade_size_20d',
    'obi_mean_5d', 'obi_mean_20d', 'obi_trend', 'book_depth_ratio_20d',
    'trade_count_mean_20d', 'vol_concentration',
    'trade_count_trend', 'dollar_volume_20d',
    'morning_reversal_rate', 'close_vs_open_mean_20d',
    'dip_recovery_rate', 'open_range_20d', 'afternoon_drift_20d',
]

def _get_pst_df(ticker):
    df = _master_cache.get(ticker)
    if df is None:
        return None
    if str(df.index.tz) != 'America/Los_Angeles':
        df = df.copy()
        df.index = df.index.tz_convert('America/Los_Angeles')
    return df

def _daily_micro_agg(df, cutoff_dt, n_days, morning_start=MORNING_START_MIN, morning_end=MORNING_END_MIN):
    past = df[df.index < cutoff_dt]
    if past.empty:
        return None
    mins = past.index.hour * 60 + past.index.minute
    morning = past[(mins >= morning_start) & (mins <= morning_end)]
    if morning.empty:
        return None
    morning = morning.copy()
    morning['_date'] = morning.index.date
    dates = sorted(morning['_date'].unique(), reverse=True)[:n_days]
    if not dates:
        return None
    morning = morning[morning['_date'].isin(set(dates))]
    daily = morning.groupby('_date').agg({
        'Delta_Pct': 'mean',
        'Spread': 'mean',
        'OBI': 'mean',
        'bsize': 'sum',
        'asize': 'sum',
        'Total_Vol': 'sum',
        'Trade_Count': 'sum',
    })
    return daily

def _daily_full_agg(df, cutoff_dt, n_days):
    past = df[df.index < cutoff_dt]
    if past.empty:
        return None
    past = past.copy()
    past['_date'] = past.index.date
    dates = sorted(past['_date'].unique(), reverse=True)[:n_days]
    if not dates:
        return None
    past = past[past['_date'].isin(set(dates))]

    mins = past.index.hour * 60 + past.index.minute

    rows = []
    for d in dates:
        day_data = past[past['_date'] == d]
        day_mins = day_data.index.hour * 60 + day_data.index.minute

        morning_mask = (day_mins >= MORNING_START_MIN) & (day_mins <= MORNING_END_MIN)
        morning_vol = day_data.loc[morning_mask, 'Total_Vol'].sum() if morning_mask.any() else 0
        full_vol = day_data['Total_Vol'].sum()

        close_vals = day_data['Close'].dropna()
        open_val = close_vals.iloc[0] if len(close_vals) > 0 else np.nan
        close_val = close_vals.iloc[-1] if len(close_vals) > 0 else np.nan
        high_val = day_data['High'].max()
        low_val = day_data['Low'].min()

        first_30 = day_data[day_mins <= (day_mins.min() + 30)] if len(day_data) > 0 else day_data
        first_30_high = first_30['High'].max() if len(first_30) > 0 else np.nan
        first_30_low = first_30['Low'].min() if len(first_30) > 0 else np.nan
        close_6_30 = np.nan
        c630_mask = day_mins <= (MORNING_START_MIN + 15)
        if c630_mask.any():
            c630_data = close_vals[c630_mask]
            if len(c630_data) > 0:
                close_6_30 = c630_data.iloc[-1]

        noon_mask = (day_mins >= 720) & (day_mins <= 730)
        noon_price = day_data.loc[noon_mask, 'Close'].iloc[-1] if noon_mask.any() and len(day_data.loc[noon_mask, 'Close']) > 0 else np.nan

        rows.append({
            '_date': d,
            'morning_vol': morning_vol,
            'full_vol': full_vol,
            'open': open_val,
            'close': close_val,
            'high': high_val,
            'low': low_val,
            'first_30_high': first_30_high,
            'first_30_low': first_30_low,
            'close_6_30': close_6_30,
            'noon_price': noon_price,
        })
    return pd.DataFrame(rows)


def compute_micro_features(ticker, as_of_date, universe_spread_median=0.0):
    zeros = {f: 0.0 for f in FEATURE_NAMES_MICRO}
    df = _get_pst_df(ticker)
    if df is None:
        return zeros

    micro_cols = ['Delta_Pct', 'Spread', 'OBI', 'bsize', 'asize',
                  'Total_Vol', 'Trade_Count']
    missing = [c for c in micro_cols if c not in df.columns]
    if missing:
        return zeros

    cutoff_dt = pd.Timestamp(as_of_date).tz_localize('America/Los_Angeles')

    daily_20 = _daily_micro_agg(df, cutoff_dt, 20)
    if daily_20 is None or len(daily_20) < 3:
        return zeros
    daily_5 = daily_20.head(5)

    flow_vals = daily_20['Delta_Pct'].values
    flow_mean_5d = float(daily_5['Delta_Pct'].mean())
    flow_mean_20d = float(daily_20['Delta_Pct'].mean())
    flow_std_20d = float(np.std(flow_vals)) if len(flow_vals) > 1 else 0
    flow_skew_20d = float(skew(flow_vals)) if len(flow_vals) > 2 else 0
    abs_flow_mean_5d = float(np.mean(np.abs(daily_5['Delta_Pct'].values)))

    if len(flow_vals) > 1:
        ac = np.corrcoef(flow_vals[:-1], flow_vals[1:])[0, 1]
        flow_persistence_20d = float(ac) if not np.isnan(ac) else 0
    else:
        flow_persistence_20d = 0

    spread_vals_5 = daily_5['Spread'].values
    spread_vals_20 = daily_20['Spread'].values
    spread_mean_5d = float(np.mean(spread_vals_5))
    spread_mean_20d = float(np.mean(spread_vals_20))
    spread_cv_20d = float(np.std(spread_vals_20) / max(np.mean(spread_vals_20), 1e-9))
    spread_rel = spread_mean_20d / max(universe_spread_median, 1e-9) if universe_spread_median > 0 else 1.0

    tc_vals = daily_20['Trade_Count'].values
    tv_vals = daily_20['Total_Vol'].values
    valid_tc = tc_vals[tc_vals > 0]
    avg_trade_size_20d = float(np.mean(tv_vals[tc_vals > 0] / tc_vals[tc_vals > 0])) if len(valid_tc) > 0 else 0

    obi_5 = daily_5['OBI'].values
    obi_20 = daily_20['OBI'].values
    obi_mean_5d = float(np.mean(obi_5))
    obi_mean_20d = float(np.mean(obi_20))
    obi_trend = obi_mean_5d - obi_mean_20d

    bs = daily_20['bsize'].values
    as_ = daily_20['asize'].values
    total_book = bs + as_
    valid_book = total_book > 0
    book_depth_ratio_20d = float(np.mean(bs[valid_book] / total_book[valid_book])) if valid_book.any() else 0.5

    tc_5 = daily_5['Trade_Count'].values
    tc_20 = daily_20['Trade_Count'].values
    trade_count_mean_20d = float(np.mean(tc_20))
    tc_mean_5 = float(np.mean(tc_5)) if len(tc_5) > 0 else 0
    tc_mean_20 = float(np.mean(tc_20)) if len(tc_20) > 0 else 1
    trade_count_trend = tc_mean_5 / max(tc_mean_20, 1)

    full_daily = _daily_full_agg(df, cutoff_dt, 20)
    if full_daily is not None and len(full_daily) > 0:
        morning_vols = full_daily['morning_vol'].values
        full_vols = full_daily['full_vol'].values
        valid_fv = full_vols > 0
        vol_concentration = float(np.mean(morning_vols[valid_fv] / full_vols[valid_fv])) if valid_fv.any() else 0

        close_arr = full_daily['close'].values
        valid_close = ~np.isnan(close_arr) & (close_arr > 0)
        dollar_volume_20d = float(np.mean(full_vols[valid_close] * close_arr[valid_close])) if valid_close.any() else 0

        opens = full_daily['open'].values
        closes = full_daily['close'].values
        valid_oc = ~np.isnan(opens) & ~np.isnan(closes) & (opens > 0)
        if valid_oc.any():
            co_rets = (closes[valid_oc] - opens[valid_oc]) / opens[valid_oc]
            close_vs_open_mean_20d = float(np.mean(co_rets))
        else:
            close_vs_open_mean_20d = 0

        close_6_30 = full_daily['close_6_30'].values
        valid_rev = ~np.isnan(close_6_30) & ~np.isnan(opens) & ~np.isnan(closes) & (opens > 0)
        if valid_rev.sum() > 0:
            morning_dir = np.sign(close_6_30[valid_rev] - opens[valid_rev])
            afternoon_dir = np.sign(closes[valid_rev] - close_6_30[valid_rev])
            reversals = morning_dir != afternoon_dir
            reversals = reversals & (morning_dir != 0) & (afternoon_dir != 0)
            morning_reversal_rate = float(np.mean(reversals))
        else:
            morning_reversal_rate = 0

        lows = full_daily['low'].values
        highs = full_daily['high'].values
        valid_dip = ~np.isnan(opens) & ~np.isnan(closes) & ~np.isnan(lows) & (opens > 0)
        if valid_dip.sum() > 0:
            dipped = lows[valid_dip] < opens[valid_dip]
            recovered = closes[valid_dip] > opens[valid_dip]
            dip_recovery_rate = float(np.mean(dipped & recovered))
        else:
            dip_recovery_rate = 0

        f30_high = full_daily['first_30_high'].values
        f30_low = full_daily['first_30_low'].values
        valid_or = ~np.isnan(f30_high) & ~np.isnan(f30_low) & valid_close
        if valid_or.any():
            open_range_20d = float(np.mean((f30_high[valid_or] - f30_low[valid_or]) / close_arr[valid_or]))
        else:
            open_range_20d = 0

        noon_p = full_daily['noon_price'].values
        valid_noon = ~np.isnan(noon_p) & ~np.isnan(closes) & (noon_p > 0)
        if valid_noon.sum() > 0:
            afternoon_drift_20d = float(np.mean((closes[valid_noon] - noon_p[valid_noon]) / noon_p[valid_noon]))
        else:
            afternoon_drift_20d = 0
    else:
        vol_concentration = 0
        dollar_volume_20d = 0
        close_vs_open_mean_20d = 0
        morning_reversal_rate = 0
        dip_recovery_rate = 0
        open_range_20d = 0
        afternoon_drift_20d = 0

    return {
        'flow_mean_5d': flow_mean_5d,
        'flow_mean_20d': flow_mean_20d,
        'flow_persistence_20d': flow_persistence_20d,
        'flow_std_20d': flow_std_20d,
        'flow_skew_20d': flow_skew_20d,
        'abs_flow_mean_5d': abs_flow_mean_5d,
        'spread_mean_5d': spread_mean_5d,
        'spread_mean_20d': spread_mean_20d,
        'spread_cv_20d': spread_cv_20d,
        'spread_rel': spread_rel,
        'avg_trade_size_20d': avg_trade_size_20d,
        'obi_mean_5d': obi_mean_5d,
        'obi_mean_20d': obi_mean_20d,
        'obi_trend': obi_trend,
        'book_depth_ratio_20d': book_depth_ratio_20d,
        'trade_count_mean_20d': trade_count_mean_20d,
        'vol_concentration': vol_concentration,
        'trade_count_trend': trade_count_trend,
        'dollar_volume_20d': dollar_volume_20d,
        'morning_reversal_rate': morning_reversal_rate,
        'close_vs_open_mean_20d': close_vs_open_mean_20d,
        'dip_recovery_rate': dip_recovery_rate,
        'open_range_20d': open_range_20d,
        'afternoon_drift_20d': afternoon_drift_20d,
    }


# ==============================================================================
# COMBINED FEATURE COMPUTATION
# ==============================================================================
BQ_FEATURE_NAMES = [
    'n_configs', 'has_both_alphas', 'cross_alpha_agree', 'agree_18_best',
    'opt_wr_median', 'pct_static', 'pct_atr_sl', 'pct_grace',
    'pct_limit', 'va_median', 'va_pos_rate', 'va_opt_ratio_med',
]

ALL_FEATURE_NAMES = FEATURE_NAMES_OHLCV + FEATURE_NAMES_SIGNAL + BQ_FEATURE_NAMES + FEATURE_NAMES_MICRO

FEATURE_CATEGORY = {}
for f in FEATURE_NAMES_OHLCV: FEATURE_CATEGORY[f] = 'OHLCV'
for f in FEATURE_NAMES_SIGNAL: FEATURE_CATEGORY[f] = 'Signal'
for f in BQ_FEATURE_NAMES: FEATURE_CATEGORY[f] = 'BQ'
for f in FEATURE_NAMES_MICRO: FEATURE_CATEGORY[f] = 'Micro'


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  MICROSTRUCTURE FEATURE EXPLORATION — SHORT Ticker Selection")
print(f"  24 micro + 25 existing = 49 features vs TEST ORACLE")
print(f"  Test period: {TEST_START} to {TEST_END}")
print(f"{'='*W}")


# -- PHASE 0: Load data, paper trade, compute TEST ORACLE --
print(f"\n  PHASE 0: Loading data + TEST ORACLE...")

print(f"    Querying BQ...")
bq_df = query_all_bq()
all_tickers = sorted(bq_df['ticker'].unique())
n_tickers = len(all_tickers)
print(f"    {len(bq_df)} BQ rows across {n_tickers} tickers")

all_configs = select_configs_perseed(bq_df, all_tickers)
print(f"    {len(all_configs)} tickers have valid configs")

vb_ranks = compute_vb_ranks(bq_df)
static_feats = compute_static_bq_features(bq_df)
print(f"    {len(static_feats)} tickers x {len(static_feats.columns)} BQ features")

tradeable_tickers = sorted(set(all_configs.keys()))

print(f"\n    Loading OHLCV + micro for {n_tickers} tickers...")
ticker_data = {}
micro_available = []
for ticker in all_tickers:
    df = load_master(ticker)
    if df is None:
        continue
    df = heal_data(df)
    has_micro = all(c in df.columns for c in ['Delta_Pct', 'Spread', 'OBI', 'Trade_Count'])
    if has_micro:
        micro_cols_sum = df[['Trade_Count', 'Total_Vol']].sum()
        if micro_cols_sum['Trade_Count'] > 0:
            micro_available.append(ticker)
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
    gc.collect()
print(f"    Loaded {len(ticker_data)} tickers with OHLCV")
print(f"    {len(micro_available)} tickers have non-zero microstructure data")

print(f"\n    Loading CSV scores...")
csv_scores_cache = {}
for ticker, cfg in sorted(all_configs.items()):
    if ticker not in ticker_data:
        continue
    base_exp = cfg['_base_experiment']
    if (ticker, base_exp) in csv_scores_cache:
        continue
    seed_scores = load_seed_scores(ticker, base_exp)
    if seed_scores:
        csv_scores_cache[(ticker, base_exp)] = seed_scores
    gc.collect()
print(f"    Loaded scores for {len(csv_scores_cache)} ticker/experiment pairs")

daily_probs_cache = {}
for ticker, cfg in all_configs.items():
    if ticker not in ticker_data:
        continue
    probs = compute_daily_probs(ticker, cfg, csv_scores_cache)
    if probs:
        daily_probs_cache[ticker] = probs
print(f"    {len(daily_probs_cache)} tickers with daily probs")

_release_memory()

print(f"\n    Paper-trading ALL tickers...")
trading_days = pd.bdate_range(TEST_START, TEST_END).date
all_trades = paper_trade_all_tickers(
    tradeable_tickers, trading_days, all_configs, ticker_data, daily_probs_cache)
n_traded = sum(1 for t in all_trades.values() if len(t) > 0)
n_total_trades = sum(len(t) for t in all_trades.values())
print(f"    {n_traded}/{len(tradeable_tickers)} tickers traded, {n_total_trades} total trades")

test_oracle_list, all_ticker_h2_stats = compute_test_oracle(all_trades, TOP_K, MIN_AVG_WIN)
oracle_tickers = [t['ticker'] for t in test_oracle_list]
oracle_set = set(oracle_tickers)
test_ranks = pd.Series({t['ticker']: t['h2_total_pnl'] for t in all_ticker_h2_stats})

print(f"    TEST ORACLE: {', '.join(oracle_tickers)}")

_release_memory()


# -- PHASE 1: Compute ALL features at 3 dates --
print(f"\n  PHASE 1: Computing features at {len(FEATURE_DATES)} dates...")

all_features = {}

for as_of_str in FEATURE_DATES:
    as_of_date = pd.Timestamp(as_of_str).date()
    print(f"\n    As-of {as_of_str}:")

    universe_spreads = []
    for ticker in tradeable_tickers:
        raw = compute_micro_features(ticker, as_of_date, 0)
        if raw['spread_mean_20d'] > 0:
            universe_spreads.append(raw['spread_mean_20d'])
    univ_spread_med = float(np.median(universe_spreads)) if universe_spreads else 0.01
    print(f"      Universe median spread: {univ_spread_med:.4f}")

    features_at_date = {}
    n_with_micro = 0
    for ticker in tradeable_tickers:
        ohlcv_f = compute_ohlcv_features(ticker, as_of_date)
        probs = daily_probs_cache.get(ticker, {})
        thresh = all_configs[ticker]['prob_thresh'] if ticker in all_configs else 0.5
        signal_f = compute_signal_features(probs, thresh, as_of_date)
        bq_f = {}
        for bf in BQ_FEATURE_NAMES:
            bq_f[bf] = static_feats.at[ticker, bf] if ticker in static_feats.index and bf in static_feats.columns else 0
        micro_f = compute_micro_features(ticker, as_of_date, univ_spread_med)
        if micro_f.get('trade_count_mean_20d', 0) > 0:
            n_with_micro += 1
        combined = {**ohlcv_f, **signal_f, **bq_f, **micro_f}
        features_at_date[ticker] = combined

    all_features[as_of_str] = features_at_date
    print(f"      {len(features_at_date)} tickers, {n_with_micro} with non-zero micro")

_release_memory()

jun1_features = all_features['2026-06-01']
common_tickers = sorted(set(jun1_features.keys()) & set(test_ranks.index))
print(f"\n    Common tickers for analysis: {len(common_tickers)}")


# ==============================================================================
# SECTION 1: Micro Feature Data Quality
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 1: Microstructure Data Quality")
print(f"{'='*W}")

print(f"\n  Micro coverage (as-of June 1):")
print(f"  {'Ticker':<7} | {'TC_20d':>8} {'Spread':>8} {'OBI':>8} {'DeltaPct':>8} {'AvgTrdSz':>8} | {'OC':>3}")
print(f"  {'-'*65}")
for ticker in tradeable_tickers:
    f = jun1_features.get(ticker, {})
    oc = ' Y ' if ticker in oracle_set else ' - '
    tc = f.get('trade_count_mean_20d', 0)
    if tc > 0:
        print(f"  {ticker:<7} | {tc:>8.0f} {f.get('spread_mean_20d',0):>8.4f} "
              f"{f.get('obi_mean_20d',0):>+7.4f} {f.get('flow_mean_20d',0):>+7.4f} "
              f"{f.get('avg_trade_size_20d',0):>8.0f} | {oc}")

n_micro_tradeable = sum(1 for t in tradeable_tickers
                         if jun1_features.get(t, {}).get('trade_count_mean_20d', 0) > 0)
n_micro_oracle = sum(1 for t in oracle_tickers
                      if jun1_features.get(t, {}).get('trade_count_mean_20d', 0) > 0)
print(f"\n  Coverage: {n_micro_tradeable}/{len(tradeable_tickers)} tradeable, "
      f"{n_micro_oracle}/{len(oracle_tickers)} ORACLE tickers have micro data")


# ==============================================================================
# SECTION 2: Micro Features vs TEST ORACLE
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 2: Micro Features vs TEST ORACLE (as-of June 1)")
print(f"{'='*W}")

micro_results = []
print(f"\n  {'Feature':<25} | {'rho':>6} {'p':>7} | {'Cohen_d':>8} | {'AUC':>5} | {'OC_mean':>10} {'Non-OC':>10}")
print(f"  {'-'*90}")

for fname in FEATURE_NAMES_MICRO:
    fvals = [jun1_features.get(t, {}).get(fname, 0) for t in common_tickers]
    tvals = [test_ranks[t] for t in common_tickers]
    rho, p = spearmanr(fvals, tvals)
    if np.isnan(rho): rho = 0

    oc_vals = [jun1_features.get(t, {}).get(fname, 0) for t in common_tickers if t in oracle_set]
    noc_vals = [jun1_features.get(t, {}).get(fname, 0) for t in common_tickers if t not in oracle_set]
    oc_mean = np.mean(oc_vals) if oc_vals else 0
    noc_mean = np.mean(noc_vals) if noc_vals else 0
    pooled = np.sqrt((np.var(oc_vals) + np.var(noc_vals)) / 2) if oc_vals and noc_vals else 1
    cohens_d = (oc_mean - noc_mean) / max(pooled, 1e-9)

    fvals_np = np.array(fvals)
    labels = np.array([1 if t in oracle_set else 0 for t in common_tickers])
    concordant = discordant = 0
    for i in range(len(fvals_np)):
        for j in range(len(fvals_np)):
            if labels[i] > labels[j]:
                if fvals_np[i] > fvals_np[j]: concordant += 1
                elif fvals_np[i] < fvals_np[j]: discordant += 1
    auc = concordant / max(concordant + discordant, 1)

    micro_results.append({
        'name': fname, 'rho': rho, 'p': p, 'cohens_d': cohens_d,
        'auc': auc, 'oc_mean': oc_mean, 'noc_mean': noc_mean,
    })

    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  {fname:<25} | {rho:>+5.3f} {p:>6.4f} | {cohens_d:>+7.2f} | {auc:>5.3f} | "
          f"{oc_mean:>+9.4f} {noc_mean:>+9.4f} {sig}")

micro_by_rho = sorted(micro_results, key=lambda x: abs(x['rho']), reverse=True)
micro_by_auc = sorted(micro_results, key=lambda x: x['auc'], reverse=True)
print(f"\n  BEST MICRO by |rho|: {micro_by_rho[0]['name']} (rho={micro_by_rho[0]['rho']:+.3f})")
print(f"  BEST MICRO by AUC:  {micro_by_auc[0]['name']} (AUC={micro_by_auc[0]['auc']:.3f})")


# ==============================================================================
# SECTION 3: Combined Ranking — All 49 Features
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 3: Combined Feature Ranking — All {len(ALL_FEATURE_NAMES)} Features")
print(f"{'='*W}")

all_results = []
for fname in ALL_FEATURE_NAMES:
    fvals = [jun1_features.get(t, {}).get(fname, 0) for t in common_tickers]
    tvals = [test_ranks[t] for t in common_tickers]
    rho, p = spearmanr(fvals, tvals)
    if np.isnan(rho): rho = 0

    oc_vals = [jun1_features.get(t, {}).get(fname, 0) for t in common_tickers if t in oracle_set]
    noc_vals = [jun1_features.get(t, {}).get(fname, 0) for t in common_tickers if t not in oracle_set]
    oc_mean = np.mean(oc_vals) if oc_vals else 0
    noc_mean = np.mean(noc_vals) if noc_vals else 0
    pooled = np.sqrt((np.var(oc_vals) + np.var(noc_vals)) / 2) if oc_vals and noc_vals else 1
    cohens_d = (oc_mean - noc_mean) / max(pooled, 1e-9)

    fvals_np = np.array(fvals)
    labels = np.array([1 if t in oracle_set else 0 for t in common_tickers])
    concordant = discordant = 0
    for i in range(len(fvals_np)):
        for j in range(len(fvals_np)):
            if labels[i] > labels[j]:
                if fvals_np[i] > fvals_np[j]: concordant += 1
                elif fvals_np[i] < fvals_np[j]: discordant += 1
    auc = concordant / max(concordant + discordant, 1)

    cat = FEATURE_CATEGORY.get(fname, '?')
    all_results.append({
        'name': fname, 'category': cat, 'rho': rho, 'p': p,
        'cohens_d': cohens_d, 'auc': auc,
        'oc_mean': oc_mean, 'noc_mean': noc_mean,
    })

all_by_rho = sorted(all_results, key=lambda x: abs(x['rho']), reverse=True)

print(f"\n  {'Rk':>3} {'Feature':<25} {'Cat':>6} | {'rho':>6} {'p':>7} | {'d':>6} {'AUC':>5}")
print(f"  {'-'*70}")
for i, r in enumerate(all_by_rho):
    marker = ' <<<' if r['category'] == 'Micro' else ''
    sig = '***' if r['p'] < 0.01 else '**' if r['p'] < 0.05 else '*' if r['p'] < 0.1 else ''
    print(f"  {i+1:>3} {r['name']:<25} {r['category']:>6} | {r['rho']:>+5.3f} {r['p']:>6.4f} | "
          f"{r['cohens_d']:>+5.2f} {r['auc']:>5.3f} {sig}{marker}")

n_micro_top10 = sum(1 for r in all_by_rho[:10] if r['category'] == 'Micro')
n_micro_top20 = sum(1 for r in all_by_rho[:20] if r['category'] == 'Micro')
print(f"\n  Micro features in top-10: {n_micro_top10}/10")
print(f"  Micro features in top-20: {n_micro_top20}/20")

best_existing = max([r for r in all_results if r['category'] != 'Micro'],
                    key=lambda x: abs(x['rho']), default={'name': '?', 'rho': 0})
best_micro = max([r for r in all_results if r['category'] == 'Micro'],
                 key=lambda x: abs(x['rho']), default={'name': '?', 'rho': 0})
print(f"\n  BEST EXISTING: {best_existing['name']} (rho={best_existing['rho']:+.3f})")
print(f"  BEST MICRO:    {best_micro['name']} (rho={best_micro['rho']:+.3f})")


# ==============================================================================
# SECTION 4: Lead Time Analysis
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 4: Lead Time Analysis — Features at Apr 1 / May 1 / Jun 1")
print(f"{'='*W}")

top_features = [r['name'] for r in all_by_rho[:15]]

print(f"\n  {'Feature':<25} | {'Apr 1':>7} {'May 1':>7} {'Jun 1':>7} | {'Trend':>8}")
print(f"  {'-'*65}")

lead_time_data = {}
for fname in top_features:
    rhos = {}
    for date_str in FEATURE_DATES:
        feats = all_features[date_str]
        fvals = [feats.get(t, {}).get(fname, 0) for t in common_tickers]
        tvals = [test_ranks[t] for t in common_tickers]
        r, _ = spearmanr(fvals, tvals)
        rhos[date_str] = r if not np.isnan(r) else 0
    lead_time_data[fname] = rhos

    apr = rhos['2026-04-01']
    may = rhos['2026-05-01']
    jun = rhos['2026-06-01']
    if abs(apr) > abs(jun) * 0.8:
        trend = 'STABLE'
    elif abs(jun) > abs(apr) * 1.5:
        trend = 'LATE'
    else:
        trend = 'GROWING'
    cat = FEATURE_CATEGORY.get(fname, '?')
    marker = ' <<<' if cat == 'Micro' else ''
    print(f"  {fname:<25} | {apr:>+6.3f} {may:>+6.3f} {jun:>+6.3f} | {trend:>8}{marker}")

early_predictors = []
for fname in ALL_FEATURE_NAMES:
    apr_feats = all_features['2026-04-01']
    fvals = [apr_feats.get(t, {}).get(fname, 0) for t in common_tickers]
    tvals = [test_ranks[t] for t in common_tickers]
    r, p = spearmanr(fvals, tvals)
    if not np.isnan(r):
        early_predictors.append({'name': fname, 'rho_apr': r, 'p': p,
                                  'cat': FEATURE_CATEGORY.get(fname, '?')})
early_by_rho = sorted(early_predictors, key=lambda x: abs(x['rho_apr']), reverse=True)

print(f"\n  BEST EARLY PREDICTORS (as-of April 1):")
for i, r in enumerate(early_by_rho[:10]):
    marker = ' <<<' if r['cat'] == 'Micro' else ''
    sig = '***' if r['p'] < 0.01 else '**' if r['p'] < 0.05 else '*' if r['p'] < 0.1 else ''
    print(f"    {i+1}. {r['name']:<25} rho={r['rho_apr']:>+.3f} ({r['cat']}){sig}{marker}")


# ==============================================================================
# SECTION 5: Feature Interactions
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 5: Feature Interactions — Micro × Existing")
print(f"{'='*W}")

top3_micro = [r['name'] for r in sorted(
    [r for r in all_results if r['category'] == 'Micro'],
    key=lambda x: abs(x['rho']), reverse=True)][:3]
top3_existing = [r['name'] for r in sorted(
    [r for r in all_results if r['category'] != 'Micro'],
    key=lambda x: abs(x['rho']), reverse=True)][:3]

print(f"\n  Top-3 micro: {', '.join(top3_micro)}")
print(f"  Top-3 existing: {', '.join(top3_existing)}")

print(f"\n  2-FEATURE AND RULES (micro + existing):")
print(f"  {'Micro':<22} {'Existing':<22} | {'Captured':>8} {'FP':>4} {'Prec':>5} {'Recall':>6}")
print(f"  {'-'*75}")

best_rule = None
best_f1 = 0

for mf in top3_micro:
    for ef in top3_existing:
        mf_vals = np.array([jun1_features.get(t, {}).get(mf, 0) for t in common_tickers])
        ef_vals = np.array([jun1_features.get(t, {}).get(ef, 0) for t in common_tickers])
        labels = np.array([1 if t in oracle_set else 0 for t in common_tickers])

        best_j_combo = 0
        best_mt = best_et = 0
        best_cap = best_fp_count = 0

        mf_uniq = np.sort(np.unique(mf_vals))
        ef_uniq = np.sort(np.unique(ef_vals))
        mf_thresholds = mf_uniq[::max(1, len(mf_uniq)//10)]
        ef_thresholds = ef_uniq[::max(1, len(ef_uniq)//10)]

        for mt in mf_thresholds:
            for et in ef_thresholds:
                pred = (mf_vals >= mt) & (ef_vals >= et)
                tp = (pred & (labels == 1)).sum()
                fp = (pred & (labels == 0)).sum()
                fn = (~pred & (labels == 1)).sum()
                tn = (~pred & (labels == 0)).sum()
                sens = tp / max(tp + fn, 1)
                spec = tn / max(tn + fp, 1)
                j = sens + spec - 1
                if j > best_j_combo:
                    best_j_combo = j
                    best_mt = mt
                    best_et = et
                    best_cap = tp
                    best_fp_count = fp

        prec = best_cap / max(best_cap + best_fp_count, 1)
        recall = best_cap / max(len(oracle_set), 1)
        f1 = 2 * prec * recall / max(prec + recall, 1e-9)

        print(f"  {mf:<22} {ef:<22} | {best_cap:>5}/{len(oracle_set):<2} {best_fp_count:>4} "
              f"{prec:>4.0%} {recall:>5.0%}")

        if f1 > best_f1:
            best_f1 = f1
            best_rule = {
                'micro': mf, 'existing': ef,
                'micro_thresh': best_mt, 'existing_thresh': best_et,
                'captured': best_cap, 'fp': best_fp_count,
                'prec': prec, 'recall': recall, 'f1': f1,
            }

if best_rule:
    print(f"\n  BEST 2-FEATURE RULE:")
    print(f"    {best_rule['micro']} >= {best_rule['micro_thresh']:.4f}")
    print(f"    {best_rule['existing']} >= {best_rule['existing_thresh']:.4f}")
    print(f"    Captures: {best_rule['captured']}/{len(oracle_set)}, "
          f"FP: {best_rule['fp']}, Prec: {best_rule['prec']:.0%}, "
          f"Recall: {best_rule['recall']:.0%}, F1: {best_rule['f1']:.2f}")


# ==============================================================================
# SECTION 6: Best Rule Builder
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 6: Best Rule Builder — 3-4 Feature Rules")
print(f"{'='*W}")

all_by_auc = sorted(all_results, key=lambda x: x['auc'], reverse=True)

print(f"\n  Top-12 features by AUC for ORACLE membership:")
print(f"  {'Rk':>3} {'Feature':<25} {'Cat':>6} {'AUC':>5} {'d':>6} {'rho':>6}")
print(f"  {'-'*55}")
for i, r in enumerate(all_by_auc[:12]):
    marker = ' <<<' if r['category'] == 'Micro' else ''
    print(f"  {i+1:>3} {r['name']:<25} {r['category']:>6} {r['auc']:>5.3f} "
          f"{r['cohens_d']:>+5.2f} {r['rho']:>+5.3f}{marker}")

top4_by_auc = all_by_auc[:4]
print(f"\n  3-FEATURE AND RULE (top-3 by AUC):")
top3_auc = all_by_auc[:3]
labels = np.array([1 if t in oracle_set else 0 for t in common_tickers])

for n_feats in [3, 4]:
    feats_used = all_by_auc[:n_feats]
    feat_thresholds = {}
    for r in feats_used:
        fname = r['name']
        fvals = np.array([jun1_features.get(t, {}).get(fname, 0) for t in common_tickers])
        sorted_uniq = np.sort(np.unique(fvals))
        best_j = 0
        best_t = sorted_uniq[0]
        for thresh in sorted_uniq:
            pred_pos = labels[fvals >= thresh]
            pred_neg = labels[fvals < thresh]
            tp = pred_pos.sum()
            fp = len(pred_pos) - tp
            fn = pred_neg.sum()
            tn = len(pred_neg) - fn
            j = tp / max(tp+fn,1) + tn / max(tn+fp,1) - 1
            if j > best_j:
                best_j = j
                best_t = thresh
        feat_thresholds[fname] = best_t

    captured = 0
    fps = 0
    for ticker in common_tickers:
        passes = True
        for r in feats_used:
            fname = r['name']
            val = jun1_features.get(ticker, {}).get(fname, 0)
            if val < feat_thresholds[fname]:
                passes = False
                break
        if passes:
            if ticker in oracle_set:
                captured += 1
            else:
                fps += 1

    prec = captured / max(captured + fps, 1)
    recall = captured / max(len(oracle_set), 1)
    f1 = 2 * prec * recall / max(prec + recall, 1e-9)

    print(f"\n  {n_feats}-FEATURE AND RULE:")
    for r in feats_used:
        fname = r['name']
        cat = r['category']
        print(f"    {fname} >= {feat_thresholds[fname]:.4f}  ({cat}, AUC={r['auc']:.3f})")
    print(f"  Captures: {captured}/{len(oracle_set)}, FP: {fps}, "
          f"Prec: {prec:.0%}, Recall: {recall:.0%}, F1: {f1:.2f}")

    passed_tickers = []
    for ticker in common_tickers:
        passes = True
        for r in feats_used:
            fname = r['name']
            val = jun1_features.get(ticker, {}).get(fname, 0)
            if val < feat_thresholds[fname]:
                passes = False
                break
        if passes:
            oc = ' Y ' if ticker in oracle_set else ' - '
            h2_pnl = test_ranks.get(ticker, 0)
            passed_tickers.append((ticker, oc, h2_pnl))
    if passed_tickers:
        print(f"  Passed tickers:")
        for ticker, oc, h2 in sorted(passed_tickers, key=lambda x: -x[2]):
            print(f"    {ticker:<7} {oc} H2_total={h2:+.2%}")


# ==============================================================================
# SECTION 7: Recommendation
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 7: RECOMMENDATION")
print(f"{'='*W}")

print(f"\n  1. BEST MICRO FEATURES (by |rho| vs test_ranks):")
for i, r in enumerate(micro_by_rho[:5]):
    print(f"     {i+1}. {r['name']:<25} rho={r['rho']:+.3f}  d={r['cohens_d']:+.2f}  AUC={r['auc']:.3f}")

print(f"\n  2. IMPROVEMENT OVER EXISTING:")
print(f"     Best existing: {best_existing['name']} (rho={best_existing['rho']:+.3f})")
print(f"     Best micro:    {best_micro['name']} (rho={best_micro['rho']:+.3f})")
improvement = abs(best_micro['rho']) - abs(best_existing['rho'])
if improvement > 0:
    print(f"     MICRO IMPROVES by {improvement:+.3f}")
else:
    print(f"     Micro trails by {improvement:+.3f} — complements, doesn't replace")

print(f"\n  3. EARLY DETECTION:")
early_micro = [r for r in early_by_rho[:10] if r['cat'] == 'Micro']
if early_micro:
    print(f"     Micro features detectable at April 1:")
    for r in early_micro[:3]:
        print(f"       {r['name']:<25} rho_apr={r['rho_apr']:+.3f}")
else:
    print(f"     No micro features in top-10 early predictors")

if best_rule:
    print(f"\n  4. BEST COMBINED RULE:")
    print(f"     {best_rule['micro']} >= {best_rule['micro_thresh']:.4f} AND "
          f"{best_rule['existing']} >= {best_rule['existing_thresh']:.4f}")
    print(f"     Captures {best_rule['captured']}/{len(oracle_set)} ORACLE, "
          f"Prec={best_rule['prec']:.0%}, F1={best_rule['f1']:.2f}")

print(f"\n  5. SUGGESTED NEXT STEPS:")
print(f"     - If micro features improve rho: add to ML ticker selection model")
print(f"     - If stable at Apr/May: use as pre-filter before model scoring")
print(f"     - If micro + existing combo captures >6/8 ORACLE: deploy as rule-based gate")
print(f"\n{'='*W}")


# ==============================================================================
# CHARTS (2x3)
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle(f'Microstructure Feature Exploration — SHORT Ticker Selection\n'
             f'24 micro + 25 existing = 49 features vs TEST ORACLE ({",".join(oracle_tickers)})',
             fontsize=13, fontweight='bold')

CAT_COLORS = {'OHLCV': '#2563eb', 'Signal': '#7c3aed', 'BQ': '#d97706', 'Micro': '#dc2626'}
OC_COLOR = '#dc2626'
NOC_COLOR = '#2563eb'

# Panel (0,0): All features ranked by |rho|
ax = axes[0, 0]
top_30 = all_by_rho[:30]
y_pos = np.arange(len(top_30))
rho_vals = [r['rho'] for r in top_30]
bar_colors = [CAT_COLORS.get(r['category'], '#6b7280') for r in top_30]
ax.barh(y_pos, rho_vals, color=bar_colors, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels([r['name'] for r in top_30], fontsize=7)
ax.invert_yaxis()
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Spearman rho vs H2 total PnL')
ax.set_title('All Features Ranked by |rho|')
from matplotlib.patches import Patch
legend_items = [Patch(facecolor=c, label=k) for k, c in CAT_COLORS.items()]
ax.legend(handles=legend_items, fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel (0,1): Best micro feature scatter
ax = axes[0, 1]
bm_name = best_micro['name']
for ticker in common_tickers:
    val = jun1_features.get(ticker, {}).get(bm_name, 0)
    h2 = test_ranks.get(ticker, 0)
    color = OC_COLOR if ticker in oracle_set else NOC_COLOR
    alpha = 0.9 if ticker in oracle_set else 0.4
    size = 80 if ticker in oracle_set else 30
    ax.scatter(val, h2 * 100, c=color, s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
    if ticker in oracle_set:
        ax.annotate(ticker, (val, h2*100), fontsize=7, ha='left')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel(bm_name)
ax.set_ylabel('H2 Total PnL (%)')
ax.set_title(f'Best Micro: {bm_name} (rho={best_micro["rho"]:+.3f})')
ax.grid(True, alpha=0.3)

# Panel (0,2): Cohen's d for micro features
ax = axes[0, 2]
micro_sorted_d = sorted(micro_results, key=lambda x: abs(x['cohens_d']), reverse=True)
y_pos = np.arange(len(micro_sorted_d))
d_vals = [r['cohens_d'] for r in micro_sorted_d]
d_colors = ['#16a34a' if d > 0 else '#dc2626' for d in d_vals]
ax.barh(y_pos, d_vals, color=d_colors, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([r['name'] for r in micro_sorted_d], fontsize=7)
ax.invert_yaxis()
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel("Cohen's d (ORACLE - Non-ORACLE)")
ax.set_title('Micro Features: Effect Size')
ax.grid(True, alpha=0.3)

# Panel (1,0): Lead time heatmap
ax = axes[1, 0]
lt_features = [r['name'] for r in all_by_rho[:12]]
lt_data = np.zeros((len(lt_features), 3))
for i, fname in enumerate(lt_features):
    for j, ds in enumerate(FEATURE_DATES):
        lt_data[i, j] = lead_time_data.get(fname, {}).get(ds, 0)
im = ax.imshow(lt_data, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels(['Apr 1', 'May 1', 'Jun 1'], fontsize=9)
ax.set_yticks(range(len(lt_features)))
labels_lt = []
for fname in lt_features:
    cat = FEATURE_CATEGORY.get(fname, '?')
    labels_lt.append(f'{fname} ({cat[0]})')
ax.set_yticklabels(labels_lt, fontsize=7)
ax.set_title('Lead Time: rho vs H2 Total PnL')
fig.colorbar(im, ax=ax, shrink=0.8, label='Spearman rho')

# Panel (1,1): 2-feature scatter
ax = axes[1, 1]
if best_rule:
    mf_name = best_rule['micro']
    ef_name = best_rule['existing']
    for ticker in common_tickers:
        mf_val = jun1_features.get(ticker, {}).get(mf_name, 0)
        ef_val = jun1_features.get(ticker, {}).get(ef_name, 0)
        color = OC_COLOR if ticker in oracle_set else NOC_COLOR
        alpha = 0.9 if ticker in oracle_set else 0.4
        size = 80 if ticker in oracle_set else 30
        ax.scatter(mf_val, ef_val, c=color, s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
        if ticker in oracle_set:
            ax.annotate(ticker, (mf_val, ef_val), fontsize=7, ha='left')
    ax.axvline(best_rule['micro_thresh'], color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(best_rule['existing_thresh'], color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel(mf_name)
    ax.set_ylabel(ef_name)
    ax.set_title(f'Best 2-Feature Rule (F1={best_rule["f1"]:.2f})')
else:
    ax.set_title('No 2-feature rule found')
ax.grid(True, alpha=0.3)

# Panel (1,2): Feature coverage heatmap for ORACLE tickers
ax = axes[1, 2]
top_feats_hm = [r['name'] for r in all_by_auc[:12]]
hm_data = np.zeros((len(oracle_tickers), len(top_feats_hm)))
for i, ticker in enumerate(oracle_tickers):
    for j, fname in enumerate(top_feats_hm):
        val = jun1_features.get(ticker, {}).get(fname, 0)
        all_vals = [jun1_features.get(t, {}).get(fname, 0) for t in common_tickers]
        pctile = np.mean(np.array(all_vals) <= val) if all_vals else 0.5
        hm_data[i, j] = pctile

im = ax.imshow(hm_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(len(top_feats_hm)))
ax.set_xticklabels(top_feats_hm, fontsize=6, rotation=45, ha='right')
ax.set_yticks(range(len(oracle_tickers)))
ax.set_yticklabels(oracle_tickers, fontsize=8)
ax.set_title('ORACLE Ticker Percentiles (top-12 features)')
fig.colorbar(im, ax=ax, shrink=0.8, label='Percentile rank')

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(CHART_PATH, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Chart saved: {CHART_PATH}")

_release_memory()
