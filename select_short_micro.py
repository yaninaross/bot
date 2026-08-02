"""
select_short_micro.py — ML Ticker Selection with Microstructure Features (SHORT)
=================================================================================
5 models × 4 feature sets (BQ_ONLY / EXISTING / MICRO_ONLY / ALL) = 20 LOO-CV runs.
49 total features: 9 OHLCV + 4 signal + 12 BQ + 24 micro.
Target: H2 (Jun-Jul) total PnL per ticker.
Backtest: MODEL_TOP8 vs BQ_ONLY_TOP8 vs ORACLE vs CURRENT.
Lead-time validation at Apr 1 / May 1 / Jun 1.
7 analysis sections + 2×3 charts. CSV scoring only. Colab-ready.
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
from scipy import stats as sp_stats

from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

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
MODEL_TOP_K = 8
FILTERED_VA_OPT_RATIO = 2.0
MIN_SEEDS = 3
OHLCV_LOOKBACK_DAYS = 45

CURRENT_TICKERS = ['VOYG', 'NTLA', 'CLF', 'SMR']
FEATURE_DATES = ['2026-04-01', '2026-05-01', '2026-06-01']

CHART_PATH = '/tmp/select_short_micro.png'

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110


# ==============================================================================
# REUSED HELPERS (from explore_micro_features.py / analyze_oracle_reverse.py)
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

def compute_h2_target(all_trades):
    half2_start = pd.Timestamp(HALF2_START).date()
    h2_pnl = {}
    for ticker, trades in all_trades.items():
        h2 = [t['pnl'] for t in trades if t['date'] >= half2_start]
        if h2:
            h2_pnl[ticker] = sum(h2)
    return pd.Series(h2_pnl)


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
# MICROSTRUCTURE FEATURES (24 features)
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
        noon_price = day_data.loc[noon_mask, 'Close'].iloc[-1] \
            if noon_mask.any() and len(day_data.loc[noon_mask, 'Close']) > 0 else np.nan
        rows.append({
            '_date': d, 'morning_vol': morning_vol, 'full_vol': full_vol,
            'open': open_val, 'close': close_val, 'high': high_val, 'low': low_val,
            'first_30_high': first_30_high, 'first_30_low': first_30_low,
            'close_6_30': close_6_30, 'noon_price': noon_price,
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
        close_vs_open_mean_20d = float(np.mean((closes[valid_oc] - opens[valid_oc]) / opens[valid_oc])) if valid_oc.any() else 0
        close_6_30 = full_daily['close_6_30'].values
        valid_rev = ~np.isnan(close_6_30) & ~np.isnan(opens) & ~np.isnan(closes) & (opens > 0)
        if valid_rev.sum() > 0:
            morning_dir = np.sign(close_6_30[valid_rev] - opens[valid_rev])
            afternoon_dir = np.sign(closes[valid_rev] - close_6_30[valid_rev])
            reversals = (morning_dir != afternoon_dir) & (morning_dir != 0) & (afternoon_dir != 0)
            morning_reversal_rate = float(np.mean(reversals))
        else:
            morning_reversal_rate = 0
        lows = full_daily['low'].values
        highs = full_daily['high'].values
        valid_dip = ~np.isnan(opens) & ~np.isnan(closes) & ~np.isnan(lows) & (opens > 0)
        if valid_dip.sum() > 0:
            dip_recovery_rate = float(np.mean((lows[valid_dip] < opens[valid_dip]) & (closes[valid_dip] > opens[valid_dip])))
        else:
            dip_recovery_rate = 0
        f30_high = full_daily['first_30_high'].values
        f30_low = full_daily['first_30_low'].values
        valid_or = ~np.isnan(f30_high) & ~np.isnan(f30_low) & valid_close
        open_range_20d = float(np.mean((f30_high[valid_or] - f30_low[valid_or]) / close_arr[valid_or])) if valid_or.any() else 0
        noon_p = full_daily['noon_price'].values
        valid_noon = ~np.isnan(noon_p) & ~np.isnan(closes) & (noon_p > 0)
        afternoon_drift_20d = float(np.mean((closes[valid_noon] - noon_p[valid_noon]) / noon_p[valid_noon])) if valid_noon.sum() > 0 else 0
    else:
        vol_concentration = 0; dollar_volume_20d = 0; close_vs_open_mean_20d = 0
        morning_reversal_rate = 0; dip_recovery_rate = 0; open_range_20d = 0; afternoon_drift_20d = 0

    return {
        'flow_mean_5d': flow_mean_5d, 'flow_mean_20d': flow_mean_20d,
        'flow_persistence_20d': flow_persistence_20d, 'flow_std_20d': flow_std_20d,
        'flow_skew_20d': flow_skew_20d, 'abs_flow_mean_5d': abs_flow_mean_5d,
        'spread_mean_5d': spread_mean_5d, 'spread_mean_20d': spread_mean_20d,
        'spread_cv_20d': spread_cv_20d, 'spread_rel': spread_rel,
        'avg_trade_size_20d': avg_trade_size_20d,
        'obi_mean_5d': obi_mean_5d, 'obi_mean_20d': obi_mean_20d,
        'obi_trend': obi_trend, 'book_depth_ratio_20d': book_depth_ratio_20d,
        'trade_count_mean_20d': trade_count_mean_20d, 'vol_concentration': vol_concentration,
        'trade_count_trend': trade_count_trend, 'dollar_volume_20d': dollar_volume_20d,
        'morning_reversal_rate': morning_reversal_rate,
        'close_vs_open_mean_20d': close_vs_open_mean_20d,
        'dip_recovery_rate': dip_recovery_rate, 'open_range_20d': open_range_20d,
        'afternoon_drift_20d': afternoon_drift_20d,
    }


# ==============================================================================
# FEATURE SETS + CATEGORIES
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

FEATURE_SETS = {
    'BQ_ONLY':    BQ_FEATURE_NAMES,
    'EXISTING':   FEATURE_NAMES_OHLCV + FEATURE_NAMES_SIGNAL + BQ_FEATURE_NAMES,
    'MICRO_ONLY': FEATURE_NAMES_MICRO,
    'ALL':        ALL_FEATURE_NAMES,
}
FS_ORDER = ['BQ_ONLY', 'EXISTING', 'MICRO_ONLY', 'ALL']


# ==============================================================================
# ML FUNCTIONS
# ==============================================================================
MODELS = {
    'Ridge':            (Ridge,                     {'alpha': 1.0}),
    'Lasso':            (Lasso,                     {'alpha': 0.001, 'max_iter': 10000}),
    'DecisionTree':     (DecisionTreeRegressor,     {'max_depth': 3, 'random_state': 42}),
    'RandomForest':     (RandomForestRegressor,     {'n_estimators': 100, 'max_depth': 3,
                                                     'max_features': 'sqrt', 'random_state': 42}),
    'GradientBoosting': (GradientBoostingRegressor, {'n_estimators': 50, 'max_depth': 2,
                                                     'learning_rate': 0.1, 'random_state': 42}),
}
MODEL_ORDER = ['Ridge', 'Lasso', 'DecisionTree', 'RandomForest', 'GradientBoosting']

def train_loo_cv(X, y, model_class, model_params):
    loo = LeaveOneOut()
    predictions = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = model_class(**model_params)
        model.fit(X_train_s, y_train)
        predictions[test_idx] = model.predict(X_test_s)
    return predictions

def evaluate_model(y_true, y_pred, oracle_set, tickers, top_k=8):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
    rho, p_val = spearmanr(y_true, y_pred)
    if np.isnan(rho): rho = 0
    true_top_k_idx = set(np.argsort(y_true)[-top_k:])
    pred_top_k_idx = set(np.argsort(y_pred)[-top_k:])
    top_k_prec = len(true_top_k_idx & pred_top_k_idx) / top_k
    pred_top_k_tickers = set(tickers[i] for i in pred_top_k_idx)
    oracle_prec = len(pred_top_k_tickers & oracle_set) / top_k
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'rho': rho, 'p_val': p_val,
        'top_k_prec': top_k_prec,
        'oracle_prec': oracle_prec,
    }

def get_feature_importance(X_arr, y_arr, feature_names, model_class, model_params):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_arr)
    model = model_class(**model_params)
    model.fit(X_s, y_arr)
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = np.abs(model.coef_)
    else:
        imp = np.zeros(len(feature_names))
    return list(zip(feature_names, imp))


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  ML TICKER SELECTION WITH MICROSTRUCTURE FEATURES — SHORT")
print(f"  5 models × 4 feature sets = 20 LOO-CV runs")
print(f"  49 features: 9 OHLCV + 4 Signal + 12 BQ + 24 Micro")
print(f"  Target: H2 (Jun-Jul) total PnL | Test: {TEST_START} to {TEST_END}")
print(f"{'='*W}")


# -- PHASE 0: Load data, paper trade, compute targets --
print(f"\n  PHASE 0: Loading data + paper trading...")

print(f"    Querying BQ...")
bq_df = query_all_bq()
all_tickers = sorted(bq_df['ticker'].unique())
n_tickers = len(all_tickers)
print(f"    {len(bq_df)} BQ rows across {n_tickers} tickers")

all_configs = select_configs_perseed(bq_df, all_tickers)
print(f"    {len(all_configs)} tickers have valid configs")

static_feats = compute_static_bq_features(bq_df)
tradeable_tickers = sorted(set(all_configs.keys()))

print(f"\n    Loading OHLCV + micro for {n_tickers} tickers...")
ticker_data = {}
for ticker in all_tickers:
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
    gc.collect()
print(f"    Loaded {len(ticker_data)} tickers")

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
print(f"    TEST ORACLE: {', '.join(oracle_tickers)}")

h2_target = compute_h2_target(all_trades)
print(f"    H2 target: {len(h2_target)} tickers (no avg_win filter)")
print(f"    H2 range: {h2_target.min():+.2%} to {h2_target.max():+.2%}")

_release_memory()


# -- PHASE 1: Compute features at 3 dates --
print(f"\n  PHASE 1: Computing features at {len(FEATURE_DATES)} dates...")

all_features = {}
for as_of_str in FEATURE_DATES:
    as_of_date = pd.Timestamp(as_of_str).date()
    print(f"    As-of {as_of_str}:")
    universe_spreads = []
    for ticker in tradeable_tickers:
        raw = compute_micro_features(ticker, as_of_date, 0)
        if raw['spread_mean_20d'] > 0:
            universe_spreads.append(raw['spread_mean_20d'])
    univ_spread_med = float(np.median(universe_spreads)) if universe_spreads else 0.01

    features_at_date = {}
    for ticker in tradeable_tickers:
        ohlcv_f = compute_ohlcv_features(ticker, as_of_date)
        probs = daily_probs_cache.get(ticker, {})
        thresh = all_configs[ticker]['prob_thresh'] if ticker in all_configs else 0.5
        signal_f = compute_signal_features(probs, thresh, as_of_date)
        bq_f = {}
        for bf in BQ_FEATURE_NAMES:
            bq_f[bf] = static_feats.at[ticker, bf] if ticker in static_feats.index and bf in static_feats.columns else 0
        micro_f = compute_micro_features(ticker, as_of_date, univ_spread_med)
        features_at_date[ticker] = {**ohlcv_f, **signal_f, **bq_f, **micro_f}
    all_features[as_of_str] = features_at_date
    print(f"      {len(features_at_date)} tickers")

_release_memory()

jun1_features = all_features['2026-06-01']
common_tickers = sorted(set(jun1_features.keys()) & set(h2_target.index))
print(f"\n    Common tickers for ML: {len(common_tickers)}")


# -- PHASE 2: LOO-CV — 5 models × 4 feature sets --
print(f"\n  PHASE 2: LOO-CV — 5 models × 4 feature sets = 20 runs...")

feature_matrices = {}
for fs_name, fs_cols in FEATURE_SETS.items():
    X_data = np.array([[jun1_features[t].get(f, 0) for f in fs_cols] for t in common_tickers],
                      dtype=np.float64)
    feature_matrices[fs_name] = X_data

y_arr = np.array([h2_target[t] for t in common_tickers], dtype=np.float64)
ticker_arr = np.array(common_tickers)

cv_results = {}
cv_predictions = {}

for fs_name in FS_ORDER:
    X_arr = feature_matrices[fs_name]
    for model_name in MODEL_ORDER:
        cls, params = MODELS[model_name]
        preds = train_loo_cv(X_arr, y_arr, cls, params)
        metrics = evaluate_model(y_arr, preds, oracle_set, ticker_arr, MODEL_TOP_K)
        cv_results[(model_name, fs_name)] = metrics
        cv_predictions[(model_name, fs_name)] = preds

best_key = max(cv_results.keys(), key=lambda k: cv_results[k]['rho'])
best_model_name, best_fs_name = best_key
best_metrics = cv_results[best_key]
best_preds = cv_predictions[best_key]


# ==============================================================================
# SECTION 1: LOO-CV Results
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 1: LOO-CV Results — Spearman rho")
print(f"{'='*W}")

print(f"\n  {'Model':<20} | ", end='')
for fs in FS_ORDER:
    n_f = len(FEATURE_SETS[fs])
    print(f" {fs}({n_f})", end='')
print()
print(f"  {'-'*80}")
for mn in MODEL_ORDER:
    print(f"  {mn:<20} | ", end='')
    for fs in FS_ORDER:
        r = cv_results[(mn, fs)]
        marker = ' <<<' if (mn, fs) == best_key else ''
        sig = '*' if r['p_val'] < 0.1 else ''
        print(f"  {r['rho']:>+5.3f}{sig:<4}", end='')
    print()

print(f"\n  BEST: {best_model_name} + {best_fs_name} → rho={best_metrics['rho']:+.3f} "
      f"(p={best_metrics['p_val']:.4f})")

print(f"\n  Oracle precision (fraction of model's top-8 in ORACLE):")
print(f"  {'Model':<20} | ", end='')
for fs in FS_ORDER:
    print(f" {fs:>12}", end='')
print()
print(f"  {'-'*80}")
for mn in MODEL_ORDER:
    print(f"  {mn:<20} | ", end='')
    for fs in FS_ORDER:
        r = cv_results[(mn, fs)]
        print(f"  {r['oracle_prec']:>9.0%}   ", end='')
    print()

best_oracle_key = max(cv_results.keys(), key=lambda k: cv_results[k]['oracle_prec'])
bon, bofn = best_oracle_key
print(f"\n  Best oracle precision: {bon} + {bofn} → {cv_results[best_oracle_key]['oracle_prec']:.0%}")

micro_vs_bq = {}
for mn in MODEL_ORDER:
    bq_rho = cv_results[(mn, 'BQ_ONLY')]['rho']
    micro_rho = cv_results[(mn, 'MICRO_ONLY')]['rho']
    all_rho = cv_results[(mn, 'ALL')]['rho']
    micro_vs_bq[mn] = {'bq': bq_rho, 'micro': micro_rho, 'all': all_rho,
                        'micro_lift': micro_rho - bq_rho, 'all_lift': all_rho - bq_rho}

print(f"\n  MICRO_ONLY vs BQ_ONLY lift:")
for mn in MODEL_ORDER:
    d = micro_vs_bq[mn]
    verdict = 'MICRO WINS' if d['micro_lift'] > 0 else 'BQ WINS'
    print(f"    {mn:<20} BQ={d['bq']:+.3f} MICRO={d['micro']:+.3f} ALL={d['all']:+.3f} → {verdict} ({d['micro_lift']:+.3f})")


# ==============================================================================
# SECTION 2: Feature Importance
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 2: Feature Importance — {best_model_name} + {best_fs_name}")
print(f"{'='*W}")

best_cls, best_params = MODELS[best_model_name]
best_fs_cols = FEATURE_SETS[best_fs_name]
X_best = feature_matrices[best_fs_name]

importance = get_feature_importance(X_best, y_arr, best_fs_cols, best_cls, best_params)
importance.sort(key=lambda x: x[1], reverse=True)

print(f"\n  {'Rk':>3} {'Feature':<25} {'Cat':>6} {'Importance':>12}")
print(f"  {'-'*50}")
for i, (fname, imp) in enumerate(importance[:15]):
    cat = FEATURE_CATEGORY.get(fname, '?')
    marker = ' <<<' if cat == 'Micro' else ''
    print(f"  {i+1:>3} {fname:<25} {cat:>6} {imp:>12.4f}{marker}")

n_micro_top5 = sum(1 for fname, _ in importance[:5] if FEATURE_CATEGORY.get(fname) == 'Micro')
n_micro_top10 = sum(1 for fname, _ in importance[:10] if FEATURE_CATEGORY.get(fname) == 'Micro')
print(f"\n  Micro in top-5 importance: {n_micro_top5}/5")
print(f"  Micro in top-10 importance: {n_micro_top10}/10")


# ==============================================================================
# SECTION 3: Ticker Selection
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 3: Ticker Selection — MODEL_TOP{MODEL_TOP_K}")
print(f"{'='*W}")

scaler_full = StandardScaler()
X_full_s = scaler_full.fit_transform(X_best)
model_full = best_cls(**best_params)
model_full.fit(X_full_s, y_arr)
full_preds = model_full.predict(X_full_s)

pred_ranking = pd.Series(full_preds, index=common_tickers).sort_values(ascending=False)
model_top8 = pred_ranking.head(MODEL_TOP_K).index.tolist()
model_set = set(model_top8)

bq_cls, bq_params = MODELS[best_model_name]
X_bq = feature_matrices['BQ_ONLY']
scaler_bq = StandardScaler()
X_bq_s = scaler_bq.fit_transform(X_bq)
model_bq = bq_cls(**bq_params)
model_bq.fit(X_bq_s, y_arr)
bq_preds = model_bq.predict(X_bq_s)
bq_top8 = pd.Series(bq_preds, index=common_tickers).sort_values(ascending=False).head(MODEL_TOP_K).index.tolist()
bq_set = set(bq_top8)

overlap_oracle = len(model_set & oracle_set)
overlap_bq = len(bq_set & oracle_set)

print(f"\n  MODEL_TOP8 ({best_fs_name}): {', '.join(model_top8)}")
print(f"  BQ_ONLY_TOP8:              {', '.join(bq_top8)}")
print(f"  ORACLE:                    {', '.join(oracle_tickers)}")
print(f"\n  MODEL overlap with ORACLE: {overlap_oracle}/{MODEL_TOP_K}")
print(f"  BQ_ONLY overlap with ORACLE: {overlap_bq}/{MODEL_TOP_K}")

print(f"\n  {'Rk':>3} {'Ticker':<7} {'Actual H2':>10} {'Pred H2':>10} {'LOO Pred':>10} | {'OC':>3} {'MOD':>4} {'BQ':>3}")
print(f"  {'-'*65}")
actual_ranking = pd.Series(y_arr, index=common_tickers).sort_values(ascending=False)
for i, (ticker, actual) in enumerate(actual_ranking.items()):
    idx_t = list(common_tickers).index(ticker)
    pred = full_preds[idx_t]
    loo_pred = best_preds[idx_t]
    oc = ' Y ' if ticker in oracle_set else ' - '
    mod = ' Y ' if ticker in model_set else ' - '
    bq_mark = ' Y ' if ticker in bq_set else ' - '
    print(f"  {i+1:>3} {ticker:<7} {actual:>+9.2%} {pred:>+9.2%} {loo_pred:>+9.2%} | {oc} {mod} {bq_mark}")


# ==============================================================================
# SECTION 4: Lead Time Validation
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 4: Lead Time — rho at Apr 1 / May 1 / Jun 1")
print(f"{'='*W}")

lead_time_results = {}
for date_str in FEATURE_DATES:
    feats = all_features[date_str]
    lt_tickers = sorted(set(feats.keys()) & set(h2_target.index))
    y_lt = np.array([h2_target[t] for t in lt_tickers], dtype=np.float64)
    tk_lt = np.array(lt_tickers)

    for fs_name, fs_cols in FEATURE_SETS.items():
        X_lt = np.array([[feats[t].get(f, 0) for f in fs_cols] for t in lt_tickers], dtype=np.float64)
        preds_lt = train_loo_cv(X_lt, y_lt, best_cls, best_params)
        metrics_lt = evaluate_model(y_lt, preds_lt, oracle_set, tk_lt, MODEL_TOP_K)
        lead_time_results[(fs_name, date_str)] = metrics_lt

print(f"\n  Spearman rho (using {best_model_name}):")
print(f"  {'Feature Set':<14} | {'Apr 1':>8} {'May 1':>8} {'Jun 1':>8} | {'Trend':>8}")
print(f"  {'-'*55}")
for fs in FS_ORDER:
    apr = lead_time_results[(fs, '2026-04-01')]['rho']
    may = lead_time_results[(fs, '2026-05-01')]['rho']
    jun = lead_time_results[(fs, '2026-06-01')]['rho']
    if abs(apr) > abs(jun) * 0.8:
        trend = 'STABLE'
    elif abs(jun) > abs(apr) * 1.5:
        trend = 'LATE'
    else:
        trend = 'GROWING'
    print(f"  {fs:<14} | {apr:>+7.3f} {may:>+7.3f} {jun:>+7.3f} | {trend:>8}")

print(f"\n  Oracle precision (using {best_model_name}):")
print(f"  {'Feature Set':<14} | {'Apr 1':>8} {'May 1':>8} {'Jun 1':>8}")
print(f"  {'-'*45}")
for fs in FS_ORDER:
    apr_op = lead_time_results[(fs, '2026-04-01')]['oracle_prec']
    may_op = lead_time_results[(fs, '2026-05-01')]['oracle_prec']
    jun_op = lead_time_results[(fs, '2026-06-01')]['oracle_prec']
    print(f"  {fs:<14} | {apr_op:>7.0%} {may_op:>7.0%} {jun_op:>7.0%}")


# ==============================================================================
# SECTION 5: Backtest — H2 PnL Comparison
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 5: Backtest — H2 PnL Comparison")
print(f"{'='*W}")

current_valid = [t for t in CURRENT_TICKERS if t in set(all_trades.keys())]

strategies = {
    'MODEL_TOP8':   model_top8,
    'BQ_ONLY_TOP8': bq_top8,
    'ORACLE':       oracle_tickers,
    'CURRENT':      current_valid,
}

periods = [
    ('FULL', TEST_START, TEST_END),
    ('H1',   TEST_START, HALF1_END),
    ('H2',   HALF2_START, TEST_END),
]

bt_results = {}
for strat_name, ticker_set in strategies.items():
    for p_name, p_start, p_end in periods:
        start_d = pd.Timestamp(p_start).date()
        end_d = pd.Timestamp(p_end).date()
        trades = []
        for ticker in ticker_set:
            for t in all_trades.get(ticker, []):
                if start_d <= t['date'] <= end_d:
                    trades.append(t)
        pnls = [t['pnl'] for t in trades]
        bt_results[(strat_name, p_name)] = {
            'n': len(pnls),
            'total': sum(pnls) if pnls else 0,
            'avg': float(np.mean(pnls)) if pnls else 0,
            'wr': float(np.mean(np.array(pnls) > 0)) if pnls else 0,
            'trades': trades,
        }

print(f"\n  {'Strategy':<15} | {'─── H2 (TARGET) ──────────':>30} | {'──── FULL ─────────────':>30}")
print(f"  {'':15} | {'N':>4} {'Total':>8} {'Avg':>8} {'WR':>5} | {'N':>4} {'Total':>8} {'Avg':>8} {'WR':>5}")
print(f"  {'-'*80}")
for sn in ['MODEL_TOP8', 'BQ_ONLY_TOP8', 'ORACLE', 'CURRENT']:
    h2 = bt_results[(sn, 'H2')]
    full = bt_results[(sn, 'FULL')]
    tag = ' [biased]' if sn == 'ORACLE' else ''
    print(f"  {sn:<15} | {h2['n']:>4} {h2['total']:>+7.1%} {h2['avg']:>+7.2%} {h2['wr']:>4.0%} "
          f"| {full['n']:>4} {full['total']:>+7.1%} {full['avg']:>+7.2%} {full['wr']:>4.0%}{tag}")

print(f"\n  Monthly H2 breakdown:")
for sn in ['MODEL_TOP8', 'BQ_ONLY_TOP8', 'ORACLE']:
    h2_trades = bt_results[(sn, 'H2')]['trades']
    monthly = defaultdict(list)
    for t in h2_trades:
        monthly[t['date'].month].append(t['pnl'])
    print(f"\n  {sn}:")
    for m in sorted(monthly.keys()):
        pnls = monthly[m]
        mname = {6: 'Jun', 7: 'Jul'}.get(m, str(m))
        print(f"    {mname}: {len(pnls):>3} trades, total={sum(pnls):>+6.1%}, avg={np.mean(pnls):>+6.2%}, WR={np.mean(np.array(pnls)>0):>4.0%}")


# ==============================================================================
# SECTION 6: Per-Ticker Analysis (TP / FP / FN)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 6: Per-Ticker Analysis — MODEL vs ORACLE")
print(f"{'='*W}")

tp_tickers = model_set & oracle_set
fp_tickers = model_set - oracle_set
fn_tickers = oracle_set - model_set

print(f"\n  TRUE POSITIVE ({len(tp_tickers)}): In MODEL and ORACLE")
for ticker in sorted(tp_tickers):
    h2 = h2_target.get(ticker, 0)
    print(f"    {ticker:<7} H2={h2:+.2%}")

print(f"\n  FALSE POSITIVE ({len(fp_tickers)}): In MODEL but NOT ORACLE")
for ticker in sorted(fp_tickers):
    h2 = h2_target.get(ticker, 0)
    idx_t = list(common_tickers).index(ticker) if ticker in common_tickers else -1
    pred = full_preds[idx_t] if idx_t >= 0 else 0
    print(f"    {ticker:<7} H2={h2:+.2%}  predicted={pred:+.4f}")

print(f"\n  FALSE NEGATIVE ({len(fn_tickers)}): In ORACLE but NOT MODEL")
for ticker in sorted(fn_tickers):
    h2 = h2_target.get(ticker, 0)
    idx_t = list(common_tickers).index(ticker) if ticker in common_tickers else -1
    pred_rank = 0
    if idx_t >= 0:
        pred_rank = int((pred_ranking > full_preds[idx_t]).sum()) + 1
    print(f"    {ticker:<7} H2={h2:+.2%}  model rank={pred_rank}")


# ==============================================================================
# SECTION 7: Recommendation
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 7: RECOMMENDATION")
print(f"{'='*W}")

model_h2 = bt_results[('MODEL_TOP8', 'H2')]
bq_h2 = bt_results[('BQ_ONLY_TOP8', 'H2')]
oracle_h2 = bt_results[('ORACLE', 'H2')]

print(f"\n  1. MODEL PERFORMANCE:")
print(f"     Best model: {best_model_name} + {best_fs_name}")
print(f"     LOO-CV rho: {best_metrics['rho']:+.3f} (p={best_metrics['p_val']:.4f})")
print(f"     Oracle precision: {best_metrics['oracle_prec']:.0%}")
print(f"     H2 backtest: {model_h2['n']} trades, total={model_h2['total']:+.1%}, avg={model_h2['avg']:+.2%}")

print(f"\n  2. DO MICRO FEATURES HELP?")
for mn in MODEL_ORDER:
    d = micro_vs_bq[mn]
    if d['micro_lift'] > 0.05:
        print(f"     {mn}: YES — MICRO rho={d['micro']:+.3f} vs BQ rho={d['bq']:+.3f} (+{d['micro_lift']:.3f})")
    elif d['micro_lift'] > 0:
        print(f"     {mn}: MARGINAL — MICRO rho={d['micro']:+.3f} vs BQ rho={d['bq']:+.3f} (+{d['micro_lift']:.3f})")
    else:
        print(f"     {mn}: NO — MICRO rho={d['micro']:+.3f} vs BQ rho={d['bq']:+.3f} ({d['micro_lift']:+.3f})")

print(f"\n  3. LEAD TIME:")
apr_best = max(lead_time_results[(fs, '2026-04-01')]['rho'] for fs in FS_ORDER)
may_best = max(lead_time_results[(fs, '2026-05-01')]['rho'] for fs in FS_ORDER)
jun_best = max(lead_time_results[(fs, '2026-06-01')]['rho'] for fs in FS_ORDER)
print(f"     Best rho: Apr={apr_best:+.3f}, May={may_best:+.3f}, Jun={jun_best:+.3f}")
if apr_best > 0.2:
    print(f"     EARLY SIGNAL EXISTS — model predictive 2 months ahead")
elif may_best > 0.2:
    print(f"     MODERATE LEAD — model predictive 1 month ahead")
else:
    print(f"     LATE SIGNAL — model only predictive near the target period")

print(f"\n  4. BACKTEST COMPARISON:")
print(f"     MODEL_TOP8:   {model_h2['total']:>+7.1%} ({model_h2['n']} trades)")
print(f"     BQ_ONLY_TOP8: {bq_h2['total']:>+7.1%} ({bq_h2['n']} trades)")
print(f"     ORACLE:       {oracle_h2['total']:>+7.1%} ({oracle_h2['n']} trades) [biased ceiling]")
model_vs_bq_lift = model_h2['total'] - bq_h2['total']
if model_vs_bq_lift > 0:
    print(f"     MODEL beats BQ_ONLY by {model_vs_bq_lift:+.1%}")
else:
    print(f"     MODEL trails BQ_ONLY by {model_vs_bq_lift:+.1%}")

print(f"\n  5. DEPLOYABILITY:")
deployable = best_metrics['rho'] > 0.3 and best_metrics['oracle_prec'] >= 0.375
if deployable:
    print(f"     DEPLOYABLE — rho>{0.3} and oracle_prec>={37.5}%")
    print(f"     Suggested tickers: {', '.join(model_top8)}")
else:
    print(f"     NOT YET DEPLOYABLE — needs rho>{0.3} and oracle_prec>={37.5}%")
    print(f"     Current: rho={best_metrics['rho']:+.3f}, oracle_prec={best_metrics['oracle_prec']:.0%}")

print(f"\n{'='*W}")


# ==============================================================================
# CHARTS (2×3)
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle(f'ML Ticker Selection with Micro Features — SHORT\n'
             f'Best: {best_model_name} + {best_fs_name} (rho={best_metrics["rho"]:+.3f}) | '
             f'MODEL_TOP8: {",".join(model_top8)}',
             fontsize=13, fontweight='bold')

CAT_COLORS = {'OHLCV': '#2563eb', 'Signal': '#7c3aed', 'BQ': '#d97706', 'Micro': '#dc2626'}
FS_COLORS = {'BQ_ONLY': '#d97706', 'EXISTING': '#2563eb', 'MICRO_ONLY': '#dc2626', 'ALL': '#16a34a'}

# Panel (0,0): Model × FeatureSet rho comparison
ax = axes[0, 0]
x = np.arange(len(MODEL_ORDER))
bar_width = 0.2
for i, fs in enumerate(FS_ORDER):
    rhos = [cv_results[(mn, fs)]['rho'] for mn in MODEL_ORDER]
    ax.bar(x + i * bar_width, rhos, bar_width, label=fs, color=FS_COLORS[fs], alpha=0.85)
ax.set_xticks(x + 1.5 * bar_width)
ax.set_xticklabels(MODEL_ORDER, fontsize=8, rotation=15)
ax.set_ylabel('Spearman rho')
ax.set_title('LOO-CV: 5 Models × 4 Feature Sets')
ax.axhline(0, color='gray', linewidth=0.5)
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# Panel (0,1): Predicted vs actual scatter (best model)
ax = axes[0, 1]
for i, ticker in enumerate(common_tickers):
    actual = y_arr[i]
    pred = best_preds[i]
    if ticker in oracle_set and ticker in model_set:
        color, size, alpha = '#16a34a', 100, 0.9
    elif ticker in oracle_set:
        color, size, alpha = '#dc2626', 80, 0.9
    elif ticker in model_set:
        color, size, alpha = '#d97706', 80, 0.9
    else:
        color, size, alpha = '#6b7280', 30, 0.4
    ax.scatter(actual * 100, pred * 100, c=color, s=size, alpha=alpha,
               edgecolors='black', linewidth=0.5)
    if ticker in oracle_set or ticker in model_set:
        ax.annotate(ticker, (actual * 100, pred * 100), fontsize=7, ha='left')
lims = [min(y_arr.min(), best_preds.min()) * 100 - 2,
        max(y_arr.max(), best_preds.max()) * 100 + 2]
ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('Actual H2 Total PnL (%)')
ax.set_ylabel('LOO Predicted H2 PnL (%)')
ax.set_title(f'{best_model_name}+{best_fs_name} (rho={best_metrics["rho"]:+.3f})')
from matplotlib.patches import Patch
legend_items = [Patch(facecolor='#16a34a', label='TP (MODEL∩ORACLE)'),
                Patch(facecolor='#dc2626', label='FN (ORACLE only)'),
                Patch(facecolor='#d97706', label='FP (MODEL only)'),
                Patch(facecolor='#6b7280', label='Neither')]
ax.legend(handles=legend_items, fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel (0,2): Feature importance (top 15)
ax = axes[0, 2]
top_imp = importance[:15]
y_pos = np.arange(len(top_imp))
imp_vals = [v for _, v in top_imp]
imp_colors = [CAT_COLORS.get(FEATURE_CATEGORY.get(f, '?'), '#6b7280') for f, _ in top_imp]
ax.barh(y_pos, imp_vals, color=imp_colors, alpha=0.85)
ax.set_yticks(y_pos)
labels_imp = [f'{f} ({FEATURE_CATEGORY.get(f, "?")[0]})' for f, _ in top_imp]
ax.set_yticklabels(labels_imp, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance')
ax.set_title(f'Top-15 Features ({best_model_name})')
legend_items = [Patch(facecolor=c, label=k) for k, c in CAT_COLORS.items()]
ax.legend(handles=legend_items, fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel (1,0): Lead time — rho at 3 dates × 4 feature sets
ax = axes[1, 0]
date_labels = ['Apr 1', 'May 1', 'Jun 1']
for fs in FS_ORDER:
    rhos = [lead_time_results[(fs, d)]['rho'] for d in FEATURE_DATES]
    ax.plot(date_labels, rhos, 'o-', label=fs, color=FS_COLORS[fs], linewidth=2, markersize=8)
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_ylabel('Spearman rho')
ax.set_title(f'Lead Time: {best_model_name} rho at 3 Dates')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel (1,1): H2 cumulative PnL curves
ax = axes[1, 1]
strat_colors = {'MODEL_TOP8': '#16a34a', 'BQ_ONLY_TOP8': '#d97706',
                'ORACLE': '#2563eb', 'CURRENT': '#6b7280'}
for sn in ['MODEL_TOP8', 'BQ_ONLY_TOP8', 'ORACLE', 'CURRENT']:
    h2_trades = bt_results[(sn, 'H2')]['trades']
    if not h2_trades:
        continue
    sorted_trades = sorted(h2_trades, key=lambda t: t['date'])
    cum_pnl = np.cumsum([t['pnl'] for t in sorted_trades])
    dates_plot = [t['date'] for t in sorted_trades]
    lw = 2.5 if sn == 'MODEL_TOP8' else 1.5
    ls = '--' if sn in ('ORACLE', 'CURRENT') else '-'
    ax.plot(dates_plot, cum_pnl * 100, color=strat_colors[sn], linewidth=lw,
            linestyle=ls, label=f'{sn} ({len(h2_trades)}t)')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('H2 Equity Curves')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', linewidth=0.5)

# Panel (1,2): Per-ticker H2 PnL bars
ax = axes[1, 2]
union_tickers = sorted(model_set | oracle_set, key=lambda t: h2_target.get(t, 0), reverse=True)
y_pos = np.arange(len(union_tickers))
bar_vals = [h2_target.get(t, 0) * 100 for t in union_tickers]
bar_colors = []
for t in union_tickers:
    if t in model_set and t in oracle_set:
        bar_colors.append('#16a34a')
    elif t in oracle_set:
        bar_colors.append('#dc2626')
    else:
        bar_colors.append('#d97706')
ax.barh(y_pos, bar_vals, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
labels_tk = []
for t in union_tickers:
    tag = ''
    if t in model_set and t in oracle_set: tag = ' [TP]'
    elif t in oracle_set: tag = ' [FN]'
    else: tag = ' [FP]'
    labels_tk.append(f'{t}{tag}')
ax.set_yticklabels(labels_tk, fontsize=8)
ax.invert_yaxis()
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('H2 Total PnL (%)')
ax.set_title('Per-Ticker H2 PnL (MODEL∪ORACLE)')
legend_items = [Patch(facecolor='#16a34a', label='TP'),
                Patch(facecolor='#dc2626', label='FN (ORACLE only)'),
                Patch(facecolor='#d97706', label='FP (MODEL only)')]
ax.legend(handles=legend_items, fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(CHART_PATH, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Chart saved: {CHART_PATH}")

_release_memory()
