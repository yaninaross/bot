"""
backtest_short_candidates.py — Candidate #2 and #3 SHORT Ticker Selection
=========================================================================
Tests the two remaining unbiased approaches head-to-head:

  #2: ADAPTIVE_RELAXED — BQ-only Ridge with relaxed gates (drop cross_alpha/has_both, ratio<5.0)
  #3: ML_DYNAMIC       — 3-model competition (GBM/Ridge/RF) on 27 features with panel data

Control: ADAPTIVE (original, no gates, BQ-only Ridge)
Baselines: ORACLE (H2 ceiling) | CURRENT (deployed)

Weekly rebalancing. Zero future information. CSV scoring. No GPU.
5 analysis sections + 2x3 charts. Colab-ready.
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

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

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
BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")
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

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1',
                     'robust_wf_v12_short_stitched')
_EXP_400_FAMILY = ('400', '401', '402', '403', '404', '405', 'WF400')
BQ_ALLOWED_MODES  = ('static', 'atr_sl', 'grace')

TOP_K = 8
FILTERED_VA_OPT_RATIO = 2.0      # strict filter for config selection
RELAXED_VA_OPT_RATIO  = 5.0      # relaxed for ADAPTIVE_RELAXED universe
MIN_SEEDS = 3
OHLCV_LOOKBACK_DAYS = 45

DYNAMIC_TARGET_BLEND = 0.5
MIN_TRADES_FOR_BLEND = 2

BURN_IN_WEEKS = 2                 # ML_DYNAMIC uses FILTERED fallback during burn-in
STICKY_RETAIN = 4                 # ML_DYNAMIC retains up to 4 positive incumbents

FILTERED_MIN_CONFIGS = 5
FILTERED_VA_POS_RATE = 0.50

MODELS = {
    'GBM': (GradientBoostingRegressor, {
        'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.1,
        'random_state': 42, 'subsample': 0.8,
    }),
    'Ridge': (Ridge, {'alpha': 1.0}),
    'RF': (RandomForestRegressor, {
        'n_estimators': 50, 'max_depth': 2, 'max_features': 'sqrt',
        'random_state': 42,
    }),
}

CURRENT_TICKERS = ['VOYG', 'NTLA', 'CLF', 'SMR']

CHART_PATH = '/tmp/backtest_short_candidates.png'

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110


# ==============================================================================
# REUSED HELPERS
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
        exp_400 = [e for e in exps if e.startswith(_EXP_400_FAMILY)]
        exp_410 = [e for e in exps if e.startswith('410')]
        has_both = len(exp_400) > 0 and len(exp_410) > 0
        cross_agree = 0.0
        if has_both:
            a400 = seed_agg[seed_agg['experiment'].apply(
                lambda e: _base_experiment(e).startswith(_EXP_400_FAMILY))]
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
    return ticker_stats[:top_k]


# ==============================================================================
# FEATURE COMPUTATION (OHLCV + Signal + Micro)
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
BQ_FEATURE_NAMES = [
    'n_configs', 'has_both_alphas', 'cross_alpha_agree', 'agree_18_best',
    'opt_wr_median', 'pct_static', 'pct_atr_sl', 'pct_grace',
    'pct_limit', 'va_median', 'va_pos_rate', 'va_opt_ratio_med',
]
ALL_FEATURE_NAMES = FEATURE_NAMES_OHLCV + FEATURE_NAMES_SIGNAL + BQ_FEATURE_NAMES + FEATURE_NAMES_MICRO


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
# WALK-FORWARD FEATURE HELPERS
# ==============================================================================
def compute_all_features_at_date(tickers, as_of_date, static_bq_feats,
                                  daily_probs_cache, all_configs, feature_cache):
    cached_tickers = {}
    for ticker in tickers:
        ck = (ticker, as_of_date)
        if ck in feature_cache:
            cached_tickers[ticker] = feature_cache[ck]

    if len(cached_tickers) == len(tickers):
        return cached_tickers

    universe_spreads = []
    for ticker in tickers:
        if ticker in cached_tickers:
            sp = cached_tickers[ticker].get('spread_mean_20d', 0)
            if sp > 0:
                universe_spreads.append(sp)
            continue
        raw = compute_micro_features(ticker, as_of_date, 0)
        if raw['spread_mean_20d'] > 0:
            universe_spreads.append(raw['spread_mean_20d'])
    univ_spread_med = float(np.median(universe_spreads)) if universe_spreads else 0.01

    result = {}
    for ticker in tickers:
        if ticker in cached_tickers:
            result[ticker] = cached_tickers[ticker]
            continue
        ohlcv_f = compute_ohlcv_features(ticker, as_of_date)
        probs = daily_probs_cache.get(ticker, {})
        thresh = all_configs[ticker]['prob_thresh'] if ticker in all_configs else 0.5
        signal_f = compute_signal_features(probs, thresh, as_of_date)
        bq_f = {}
        for bf in BQ_FEATURE_NAMES:
            bq_f[bf] = static_bq_feats.at[ticker, bf] \
                if ticker in static_bq_feats.index and bf in static_bq_feats.columns else 0
        micro_f = compute_micro_features(ticker, as_of_date, univ_spread_med)
        feat_dict = {**ohlcv_f, **signal_f, **bq_f, **micro_f}
        result[ticker] = feat_dict
        feature_cache[(ticker, as_of_date)] = feat_dict
    return result

def compute_backward_target(all_trades, bq_target, tickers, as_of_date,
                             blend=DYNAMIC_TARGET_BLEND, min_trades=MIN_TRADES_FOR_BLEND):
    y = {}
    for ticker in tickers:
        bq_val = bq_target.get(ticker, 0)
        past = [t['pnl'] for t in all_trades.get(ticker, []) if t['date'] < as_of_date]
        if len(past) >= min_trades:
            actual_mean = float(np.mean(past))
            y[ticker] = blend * actual_mean + (1 - blend) * bq_val
        else:
            y[ticker] = bq_val
    return y


# ==============================================================================
# STRATEGY 1: ADAPTIVE (control — same as backtest_short_3approaches.py)
# ==============================================================================
def select_adaptive(tickers, features, all_trades, bq_target, as_of_date, top_k=TOP_K):
    """BQ-only Ridge, full universe, blended backward target."""
    valid = [t for t in tickers if t in features]
    if len(valid) < top_k:
        return valid[:top_k], {}

    y_dict = compute_backward_target(all_trades, bq_target, valid, as_of_date)

    X = np.array([[features[t].get(f, 0) for f in BQ_FEATURE_NAMES] for t in valid],
                 dtype=np.float64)
    y = np.array([y_dict.get(t, 0) for t in valid], dtype=np.float64)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_s, y)
    preds = model.predict(X_s)

    pred_dict = dict(zip(valid, preds))
    ranked = sorted(zip(valid, preds), key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in ranked[:top_k]]
    return selected, pred_dict


# ==============================================================================
# STRATEGY 2: ADAPTIVE_RELAXED (candidate #2 — relaxed gates)
# ==============================================================================
def select_adaptive_relaxed(tickers, features, all_trades, bq_target,
                            static_bq_feats, as_of_date, top_k=TOP_K):
    """BQ-only Ridge on pre-filtered universe: va_opt_ratio_med < 5.0 only.
    No cross_alpha_agree or has_both_alphas gates.
    """
    # Pre-filter: only keep tickers passing relaxed gate
    filtered = []
    blocked = []
    for t in tickers:
        if t not in features:
            continue
        if t in static_bq_feats.index:
            ratio = static_bq_feats.at[t, 'va_opt_ratio_med']
            if ratio >= RELAXED_VA_OPT_RATIO:
                blocked.append((t, f'ratio={ratio:.2f}'))
                continue
        filtered.append(t)

    if len(filtered) < top_k:
        filtered = [t for t in tickers if t in features]

    y_dict = compute_backward_target(all_trades, bq_target, filtered, as_of_date)

    X = np.array([[features[t].get(f, 0) for f in BQ_FEATURE_NAMES] for t in filtered],
                 dtype=np.float64)
    y = np.array([y_dict.get(t, 0) for t in filtered], dtype=np.float64)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_s, y)
    preds = model.predict(X_s)

    pred_dict = dict(zip(filtered, preds))
    ranked = sorted(zip(filtered, preds), key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in ranked[:top_k]]
    return selected, pred_dict, len(filtered), blocked


# ==============================================================================
# STRATEGY 3: ML_DYNAMIC (candidate #3 — 3-model competition)
# ==============================================================================
FEATURE_NAMES_PAPER = [
    'paper_pnl_mean', 'paper_pnl_recent', 'paper_wr',
    'paper_n_trades', 'paper_sharpe', 'paper_best',
]
FEATURE_NAMES_BQ_8 = [
    'n_configs', 'has_both_alphas', 'cross_alpha_agree', 'agree_18_best',
    'opt_wr_median', 'pct_static', 'pct_atr_sl', 'pct_grace',
]
ML_FEATURE_NAMES = FEATURE_NAMES_OHLCV + FEATURE_NAMES_SIGNAL + FEATURE_NAMES_PAPER + FEATURE_NAMES_BQ_8

def compute_paper_features(paper_trades):
    if not paper_trades:
        return {f: 0.0 for f in FEATURE_NAMES_PAPER}
    pnls = np.array([t['pnl'] for t in paper_trades])
    n = len(pnls)
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls)) if n > 1 else 1e-9
    recent = pnls[-5:] if n >= 5 else pnls
    return {
        'paper_pnl_mean': mean_pnl,
        'paper_pnl_recent': float(np.mean(recent)),
        'paper_wr': float(np.mean(pnls > 0)),
        'paper_n_trades': float(n),
        'paper_sharpe': mean_pnl / max(std_pnl, 1e-9),
        'paper_best': float(np.max(pnls)),
    }


def build_weekly_features_ml(tickers, static_bq_feats, daily_probs_cache,
                              configs, paper_history, as_of_date):
    """Build 27-feature DataFrame for ML_DYNAMIC."""
    rows = []
    for ticker in tickers:
        feat = {}
        feat.update(compute_ohlcv_features(ticker, as_of_date))
        probs = daily_probs_cache.get(ticker, {})
        thresh = configs[ticker]['prob_thresh'] if ticker in configs else 0.5
        feat.update(compute_signal_features(probs, thresh, as_of_date))
        feat.update(compute_paper_features(paper_history.get(ticker, [])))
        if ticker in static_bq_feats.index:
            for col in FEATURE_NAMES_BQ_8:
                feat[col] = static_bq_feats.at[ticker, col] if col in static_bq_feats.columns else 0.0
        else:
            for col in FEATURE_NAMES_BQ_8:
                feat[col] = 0.0
        rows.append({'ticker': ticker, **feat})
    df = pd.DataFrame(rows).set_index('ticker')
    df = df.reindex(columns=ML_FEATURE_NAMES).fillna(0)
    return df


def build_panel_dataset(weekly_features_history, paper_pnl_by_week, tickers):
    X_rows = []
    y_vals = []
    for week_idx, feat_df in weekly_features_history:
        pnl_map = paper_pnl_by_week.get(week_idx, {})
        for ticker in feat_df.index:
            if ticker not in tickers:
                continue
            row = feat_df.loc[ticker].values.astype(np.float64)
            target = pnl_map.get(ticker, 0.0)
            X_rows.append(row)
            y_vals.append(target)
    if not X_rows:
        return None, None
    return np.array(X_rows), np.array(y_vals)


def _loo_cv_rho(model_cls, params, X, y):
    n = len(y)
    if n < 10:
        return 0.0
    n_folds = min(5, n // 10)
    if n_folds < 2:
        return 0.0
    fold_ids = np.arange(n) % n_folds
    oo_preds = np.full(n, np.nan)
    for fold in range(n_folds):
        train_mask = fold_ids != fold
        test_mask = fold_ids == fold
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_mask])
        X_te = scaler.transform(X[test_mask])
        model = model_cls(**params)
        model.fit(X_tr, y[train_mask])
        oo_preds[test_mask] = model.predict(X_te)
    valid = ~np.isnan(oo_preds)
    if valid.sum() < 5:
        return 0.0
    rho, _ = spearmanr(y[valid], oo_preds[valid])
    return rho if not np.isnan(rho) else 0.0


def train_and_predict_ml(X_train, y_train, X_current, current_tickers, top_k):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_curr_s = scaler.transform(X_current)

    best_rho = -999
    best_name = None
    best_preds = None
    best_importances = None
    model_metrics = {}

    for name, (model_cls, params) in MODELS.items():
        cv_rho = _loo_cv_rho(model_cls, params, X_train, y_train)
        model = model_cls(**params)
        model.fit(X_train_s, y_train)
        train_preds = model.predict(X_train_s)
        train_rho, _ = spearmanr(y_train, train_preds)
        if np.isnan(train_rho):
            train_rho = 0.0
        model_metrics[name] = {'rho': cv_rho, 'train_rho': train_rho}

        if cv_rho > best_rho:
            best_rho = cv_rho
            best_name = name
            best_preds = model.predict(X_curr_s)
            if hasattr(model, 'feature_importances_'):
                best_importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                best_importances = np.abs(model.coef_)
            else:
                best_importances = np.zeros(X_train.shape[1])

    pred_df = pd.DataFrame({
        'ticker': current_tickers,
        'pred': best_preds,
    }).sort_values('pred', ascending=False)
    selected = pred_df.head(top_k)['ticker'].tolist()

    importances_dict = dict(zip(ML_FEATURE_NAMES, best_importances))

    return selected, pred_df, best_name, importances_dict, model_metrics, best_rho


def select_filtered_tickers(bq_df, top_k):
    """Rule-based FILTERED fallback for ML_DYNAMIC burn-in."""
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = bq_df[bq_df['is_ensemble'] != True].copy()
    indiv = indiv[indiv['tpsl_mode'] != 'limit_entry']
    indiv['va_opt_ratio'] = indiv['val_a_avg_pnl'] / indiv['opt_avg_pnl']
    indiv = indiv[indiv['va_opt_ratio'] <= FILTERED_VA_OPT_RATIO]
    ticker_stats = []
    for ticker, grp in indiv.groupby('ticker'):
        if len(grp) < FILTERED_MIN_CONFIGS:
            continue
        va_pos_rate = (grp['val_a_avg_pnl'] >= 0.01).mean()
        if va_pos_rate < FILTERED_VA_POS_RATE:
            continue
        ticker_stats.append({
            'ticker': ticker,
            'va_median': grp['val_a_avg_pnl'].median(),
            'va_pos_rate': va_pos_rate,
            'n_configs': len(grp),
        })
    if not ticker_stats:
        return []
    stats_df = pd.DataFrame(ticker_stats).sort_values('va_median', ascending=False)
    return stats_df.head(top_k)['ticker'].tolist()


def paper_trade_week(tickers, week_days, configs, ticker_data, daily_probs_cache):
    results = defaultdict(list)
    for day in week_days:
        np.random.seed(42 + day.toordinal())
        for ticker in tickers:
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
                results[ticker].append(trade)
    return results


def generate_weeks(start_str, end_str):
    start = pd.Timestamp(start_str).date()
    end = pd.Timestamp(end_str).date()
    weeks = []
    current = start
    while current <= end:
        if current.weekday() <= 4:
            week_monday = current - timedelta(days=current.weekday())
            if week_monday < start:
                week_monday = start
            week_friday = week_monday + timedelta(days=4)
            if week_friday > end:
                week_friday = end
            week_entry = (week_monday, week_friday)
            if not weeks or weeks[-1] != week_entry:
                weeks.append(week_entry)
            current = week_friday + timedelta(days=3)
        else:
            current += timedelta(days=1)
    return weeks


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

STRAT_NAMES = ['ADAPTIVE', 'ADAPTIVE_RELAXED', 'ML_DYNAMIC', 'ORACLE', 'CURRENT']
STRAT_COLORS = {
    'ADAPTIVE':         '#2563eb',
    'ADAPTIVE_RELAXED': '#7c3aed',
    'ML_DYNAMIC':       '#dc2626',
    'ORACLE':           '#16a34a',
    'CURRENT':          '#a3a3a3',
}

print(f"\n{'='*W}")
print(f"  CANDIDATE COMPARISON — SHORT TICKER SELECTION")
print(f"  #2: ADAPTIVE_RELAXED (BQ-Ridge, relaxed gates, ratio<{RELAXED_VA_OPT_RATIO})")
print(f"  #3: ML_DYNAMIC (3-model competition, 27 features, panel data)")
print(f"  Control: ADAPTIVE (BQ-Ridge, no gates)")
print(f"  Baselines: ORACLE (H2 ceiling) | CURRENT (deployed)")
print(f"  Test: {TEST_START} to {TEST_END}")
print(f"{'='*W}")


# -- PHASE 0: Load data + paper trade ALL --
print(f"\n  PHASE 0: Loading data...")

print(f"    Querying BQ...")
bq_df = query_all_bq()
all_tickers = sorted(bq_df['ticker'].unique())
n_tickers = len(all_tickers)
print(f"    {len(bq_df)} BQ rows across {n_tickers} tickers")

all_configs = select_configs_perseed(bq_df, all_tickers)
print(f"    {len(all_configs)} tickers have valid configs")

static_bq_feats = compute_static_bq_features(bq_df)
tradeable_tickers = sorted(set(all_configs.keys()))

bq_target = {}
for ticker in tradeable_tickers:
    bq_target[ticker] = static_bq_feats.at[ticker, 'va_median'] \
        if ticker in static_bq_feats.index else 0

filtered_tickers = select_filtered_tickers(bq_df, TOP_K)
print(f"    FILTERED (burn-in fallback): {', '.join(filtered_tickers)}")

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
trading_days = sorted(pd.bdate_range(TEST_START, TEST_END).date)
all_trades = paper_trade_all_tickers(
    tradeable_tickers, trading_days, all_configs, ticker_data, daily_probs_cache)
n_traded = sum(1 for t in all_trades.values() if len(t) > 0)
n_total_trades = sum(len(t) for t in all_trades.values())
print(f"    {n_traded}/{len(tradeable_tickers)} tickers traded, {n_total_trades} total trades")

test_oracle_list = compute_test_oracle(all_trades, TOP_K, MIN_AVG_WIN)
oracle_tickers = [t['ticker'] for t in test_oracle_list]
oracle_set = set(oracle_tickers)
print(f"    FINAL ORACLE: {', '.join(oracle_tickers)}")
for ts in test_oracle_list:
    print(f"      {ts['ticker']:<7} H2: {ts['h2_total_pnl']:>+7.2%} ({ts['h2_n']} trades, WR {ts['h2_wr']:.0%})")

_release_memory()

# -- Show relaxed gate effect --
n_full_universe = len([t for t in tradeable_tickers if t in ticker_data and t in daily_probs_cache])
relaxed_blocked = []
for t in tradeable_tickers:
    if t in static_bq_feats.index:
        r = static_bq_feats.at[t, 'va_opt_ratio_med']
        if r >= RELAXED_VA_OPT_RATIO:
            relaxed_blocked.append((t, r))
n_relaxed = n_full_universe - len(relaxed_blocked)
oracle_blocked_by_relaxed = [t for t in oracle_tickers if any(bt == t for bt, _ in relaxed_blocked)]
print(f"\n    Gate comparison:")
print(f"      Full universe: {n_full_universe} tickers")
print(f"      RELAXED (ratio<{RELAXED_VA_OPT_RATIO}): {n_relaxed} tickers pass ({len(relaxed_blocked)} blocked)")
print(f"      Oracle tickers blocked by RELAXED: {len(oracle_blocked_by_relaxed)} — {', '.join(oracle_blocked_by_relaxed) if oracle_blocked_by_relaxed else 'NONE'}")
if relaxed_blocked:
    for t, r in sorted(relaxed_blocked, key=lambda x: x[1], reverse=True)[:10]:
        oracle_mark = ' *ORACLE*' if t in oracle_set else ''
        print(f"        {t:<7} ratio={r:.2f}{oracle_mark}")


# -- PHASE 1: Walk-Forward --
weeks = generate_weeks(TEST_START, TEST_END)
trading_days_list = pd.bdate_range(TEST_START, TEST_END).date

print(f"\n{'='*W}")
print(f"  PHASE 1: Walk-Forward ({len(weeks)} weeks)")
print(f"{'='*W}")

valid_tradeable = [t for t in tradeable_tickers
                   if t in ticker_data and t in daily_probs_cache]
print(f"    {len(valid_tradeable)} tradeable tickers")

feature_cache = {}

# ADAPTIVE state
adp_daily_active = {}
adp_selection = list(valid_tradeable)

# ADAPTIVE_RELAXED state
rlx_daily_active = {}
rlx_selection = list(valid_tradeable)

# ML_DYNAMIC state
ml_daily_active = {}
ml_selection = list(valid_tradeable)
paper_history = defaultdict(list)
paper_pnl_by_week = {}
weekly_features_history = []
weekly_importances = []
weekly_model_info = []

# Rebalance log
rebalance_log = []

for week_idx, (wk_start, wk_end) in enumerate(weeks):
    week_days = [d for d in trading_days_list if wk_start <= d <= wk_end]
    if not week_days:
        continue

    rebal_date = week_days[0]

    # -- Compute 49 features for ADAPTIVE approaches --
    features = compute_all_features_at_date(
        valid_tradeable, rebal_date, static_bq_feats,
        daily_probs_cache, all_configs, feature_cache)

    # -- ADAPTIVE (control) --
    adp_selected, adp_preds = select_adaptive(
        valid_tradeable, features, all_trades, bq_target, rebal_date)

    # -- ADAPTIVE_RELAXED (candidate #2) --
    rlx_selected, rlx_preds, rlx_universe_size, rlx_blocked = select_adaptive_relaxed(
        valid_tradeable, features, all_trades, bq_target,
        static_bq_feats, rebal_date)

    adp_selection = adp_selected
    rlx_selection = rlx_selected

    # -- ML_DYNAMIC (candidate #3) --
    # Build 27-feature DataFrame
    feat_df_ml = build_weekly_features_ml(
        tradeable_tickers, static_bq_feats, daily_probs_cache,
        all_configs, paper_history, rebal_date)
    weekly_features_history.append((week_idx, feat_df_ml))

    if week_idx < BURN_IN_WEEKS:
        ml_active = [t for t in filtered_tickers
                     if t in all_configs and t in ticker_data]
        ml_model_name = 'BURN_IN'
        ml_cv_rho = 0.0
        ml_importances = {}
        ml_metrics = {}
    else:
        X_train, y_train = build_panel_dataset(
            weekly_features_history[:-1],
            paper_pnl_by_week,
            tradeable_tickers)

        if X_train is not None and len(X_train) >= 10:
            selected_ml, pred_df_ml, ml_model_name, ml_importances, ml_metrics, ml_cv_rho = \
                train_and_predict_ml(X_train, y_train,
                                     feat_df_ml.values.astype(np.float64),
                                     feat_df_ml.index.tolist(), TOP_K)

            # Sticky incumbents
            prev_ml = ml_selection if ml_selection != list(valid_tradeable) else []
            if prev_ml and STICKY_RETAIN > 0:
                incumbent_scores = []
                for tk in prev_ml:
                    ph = paper_history.get(tk, [])
                    if ph:
                        recent = [t['pnl'] for t in ph[-5:]]
                        incumbent_scores.append((tk, np.mean(recent)))
                    else:
                        incumbent_scores.append((tk, 0.0))
                incumbent_scores.sort(key=lambda x: x[1], reverse=True)
                retained = [tk for tk, pnl in incumbent_scores[:STICKY_RETAIN]
                           if pnl > 0 and tk in all_configs and tk in ticker_data]
                n_new = TOP_K - len(retained)
                new_from_ml = [t for t in selected_ml
                              if t not in retained and t in all_configs and t in ticker_data]
                ml_active = retained + new_from_ml[:n_new]
            else:
                ml_active = [t for t in selected_ml
                             if t in all_configs and t in ticker_data]

            weekly_importances.append(ml_importances)
            weekly_model_info.append({
                'week': week_idx + 1, 'model': ml_model_name,
                'rho': ml_cv_rho, 'metrics': ml_metrics,
                'n_train': len(X_train),
            })
        else:
            ml_active = [t for t in filtered_tickers
                         if t in all_configs and t in ticker_data]
            ml_model_name = 'FALLBACK'
            ml_cv_rho = 0.0
            ml_importances = {}
            ml_metrics = {}

    ml_selection = ml_active

    # -- Paper-trade ALL tickers for this week (updates ML_DYNAMIC's feature/target) --
    week_paper = paper_trade_week(
        tradeable_tickers, week_days, all_configs, ticker_data, daily_probs_cache)
    week_pnl_map = {}
    for ticker, trades in week_paper.items():
        paper_history[ticker].extend(trades)
        if trades:
            week_pnl_map[ticker] = np.mean([t['pnl'] for t in trades])
    paper_pnl_by_week[week_idx] = week_pnl_map

    # -- Track daily active sets --
    for day in week_days:
        adp_daily_active[day] = set(adp_selection)
        rlx_daily_active[day] = set(rlx_selection)
        ml_daily_active[day] = set(ml_selection)

    # -- Log --
    adp_ov = len(set(adp_selected) & oracle_set)
    rlx_ov = len(set(rlx_selected) & oracle_set)
    ml_ov = len(set(ml_selection) & oracle_set)

    rebalance_log.append({
        'week': week_idx + 1,
        'date': rebal_date,
        'adp_tickers': adp_selected.copy(),
        'rlx_tickers': rlx_selected.copy(),
        'ml_tickers': list(ml_selection).copy(),
        'adp_oracle_overlap': adp_ov,
        'rlx_oracle_overlap': rlx_ov,
        'ml_oracle_overlap': ml_ov,
        'rlx_universe_size': rlx_universe_size,
        'ml_model': ml_model_name,
        'ml_cv_rho': ml_cv_rho,
    })

    print(f"  W{week_idx+1:>2} {str(rebal_date):<12} "
          f"ADP→Oracle:{adp_ov}/{TOP_K}  "
          f"RLX→Oracle:{rlx_ov}/{TOP_K} (univ={rlx_universe_size})  "
          f"ML→Oracle:{ml_ov}/{TOP_K} ({ml_model_name})")

print(f"\n    {len(rebalance_log)} weeks completed")
_release_memory()


# -- Collect strategy trades --
print(f"\n  Collecting strategy trades...")

current_set = set(CURRENT_TICKERS)
strat_trades = {s: [] for s in STRAT_NAMES}

for ticker, trades in all_trades.items():
    for trade in trades:
        d = trade['date']
        if ticker in adp_daily_active.get(d, set()):
            strat_trades['ADAPTIVE'].append(trade)
        if ticker in rlx_daily_active.get(d, set()):
            strat_trades['ADAPTIVE_RELAXED'].append(trade)
        if ticker in ml_daily_active.get(d, set()):
            strat_trades['ML_DYNAMIC'].append(trade)
        if ticker in oracle_set:
            strat_trades['ORACLE'].append(trade)
        if ticker in current_set:
            strat_trades['CURRENT'].append(trade)

for sn in STRAT_NAMES:
    print(f"    {sn:<18}: {len(strat_trades[sn])} trades")


# ==============================================================================
# SECTION 1: Strategy Comparison
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 1: Strategy Comparison — Candidates vs Control vs Baselines")
print(f"{'='*W}")

half2_start_d = pd.Timestamp(HALF2_START).date()
test_start_d = pd.Timestamp(TEST_START).date()
test_end_d = pd.Timestamp(TEST_END).date()

def compute_bt_stats(trades, start_d=None, end_d=None):
    if start_d and end_d:
        filtered = [t for t in trades if start_d <= t['date'] <= end_d]
    else:
        filtered = trades
    pnls = [t['pnl'] for t in filtered]
    if not pnls:
        return {'n': 0, 'total': 0, 'avg': 0, 'wr': 0, 'sharpe': 0}
    total = sum(pnls)
    avg = float(np.mean(pnls))
    wr = float(np.mean(np.array(pnls) > 0))
    std = float(np.std(pnls)) if len(pnls) > 1 else 1
    sharpe = avg / std if std > 0 else 0
    return {'n': len(pnls), 'total': total, 'avg': avg, 'wr': wr, 'sharpe': sharpe}

print(f"\n  {'Strategy':<18} | {'── H2 (Jun-Jul) ─────────────────────':>40} | {'── FULL (Apr-Jul) ──────────────────':>38}")
print(f"  {'':18} | {'N':>4} {'Total':>8} {'Avg':>8} {'WR':>5} {'Sharpe':>7} | {'N':>4} {'Total':>8} {'Avg':>8} {'WR':>5} {'Sharpe':>7}")
print(f"  {'-'*108}")

h2_stats = {}
full_stats = {}
for sn in STRAT_NAMES:
    h2 = compute_bt_stats(strat_trades[sn], half2_start_d, test_end_d)
    full = compute_bt_stats(strat_trades[sn], test_start_d, test_end_d)
    h2_stats[sn] = h2
    full_stats[sn] = full
    tag = ''
    if sn == 'ORACLE': tag = ' [ceiling]'
    elif sn in ('ADAPTIVE_RELAXED', 'ML_DYNAMIC'): tag = ' [CANDIDATE]'
    elif sn == 'ADAPTIVE': tag = ' [control]'
    print(f"  {sn:<18} | {h2['n']:>4} {h2['total']:>+7.1%} {h2['avg']:>+7.2%} {h2['wr']:>4.0%} {h2['sharpe']:>+6.2f} "
          f"| {full['n']:>4} {full['total']:>+7.1%} {full['avg']:>+7.2%} {full['wr']:>4.0%} {full['sharpe']:>+6.2f}{tag}")

oracle_ceiling = h2_stats['ORACLE']['total']
print(f"\n  Candidate rankings (H2 total PnL):")
candidate_ranking = sorted(
    [(sn, h2_stats[sn]['total']) for sn in ['ADAPTIVE', 'ADAPTIVE_RELAXED', 'ML_DYNAMIC']],
    key=lambda x: x[1], reverse=True)
for rank, (sn, pnl) in enumerate(candidate_ranking, 1):
    captured = pnl / oracle_ceiling * 100 if oracle_ceiling > 0 else 0
    vs_ctrl = pnl - h2_stats['ADAPTIVE']['total'] if sn != 'ADAPTIVE' else 0
    vs_curr = pnl - h2_stats['CURRENT']['total']
    ctrl_str = f"  vs ADAPTIVE={vs_ctrl:>+6.1%}" if sn != 'ADAPTIVE' else "  (control)"
    print(f"    #{rank} {sn:<18} H2={pnl:>+7.1%}  captured={captured:>4.0f}%  vs CURRENT={vs_curr:>+6.1%}{ctrl_str}")


# ==============================================================================
# SECTION 2: Oracle Overlap Progression
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 2: Oracle Overlap — Week by Week")
print(f"{'='*W}")

print(f"\n  {'Wk':>3} {'Date':<12} {'ADP':>4} {'RLX':>4} {'ML':>4} {'RLX tickers':>40} {'ML tickers':>40}")
print(f"  {'-'*115}")

adp_overlaps = []
rlx_overlaps = []
ml_overlaps = []
for entry in rebalance_log:
    adp_overlaps.append(entry['adp_oracle_overlap'])
    rlx_overlaps.append(entry['rlx_oracle_overlap'])
    ml_overlaps.append(entry['ml_oracle_overlap'])

    rlx_in_oracle = sorted(set(entry['rlx_tickers']) & oracle_set)
    ml_in_oracle = sorted(set(entry['ml_tickers']) & oracle_set)
    rlx_str = ','.join(rlx_in_oracle) if rlx_in_oracle else '-'
    ml_str = ','.join(ml_in_oracle) if ml_in_oracle else '-'
    print(f"  W{entry['week']:>2} {str(entry['date']):<12} "
          f"{entry['adp_oracle_overlap']:>3}/{TOP_K} "
          f"{entry['rlx_oracle_overlap']:>3}/{TOP_K} "
          f"{entry['ml_oracle_overlap']:>3}/{TOP_K} "
          f"  {rlx_str:<40} {ml_str:<40}")

print(f"\n  Average oracle overlap:")
print(f"    ADAPTIVE:         {np.mean(adp_overlaps):.1f}/{TOP_K}")
print(f"    ADAPTIVE_RELAXED: {np.mean(rlx_overlaps):.1f}/{TOP_K}")
print(f"    ML_DYNAMIC:       {np.mean(ml_overlaps):.1f}/{TOP_K}")

improvement = np.mean(rlx_overlaps) - np.mean(adp_overlaps)
print(f"\n  Relaxed gates effect: {improvement:>+.1f} oracle overlap improvement")


# ==============================================================================
# SECTION 3: Ticker Selection — Frequency & Convergence
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 3: Ticker Selection — Frequency of Appearance")
print(f"{'='*W}")

n_weeks = len(rebalance_log)
ticker_freq = {'ADAPTIVE': defaultdict(int), 'ADAPTIVE_RELAXED': defaultdict(int),
               'ML_DYNAMIC': defaultdict(int)}
for entry in rebalance_log:
    for t in entry['adp_tickers']:
        ticker_freq['ADAPTIVE'][t] += 1
    for t in entry['rlx_tickers']:
        ticker_freq['ADAPTIVE_RELAXED'][t] += 1
    for t in entry['ml_tickers']:
        ticker_freq['ML_DYNAMIC'][t] += 1

all_selected_tickers = set()
for strat_freqs in ticker_freq.values():
    all_selected_tickers.update(strat_freqs.keys())

print(f"\n  {'Ticker':<7} {'ADP':>5} {'RLX':>5} {'ML':>5} {'Oracle':>7}")
print(f"  {'-'*38}")
for t in sorted(all_selected_tickers):
    adp_f = ticker_freq['ADAPTIVE'].get(t, 0)
    rlx_f = ticker_freq['ADAPTIVE_RELAXED'].get(t, 0)
    ml_f = ticker_freq['ML_DYNAMIC'].get(t, 0)
    oc = '  Y' if t in oracle_set else '   '
    if adp_f + rlx_f + ml_f >= 3:
        print(f"  {t:<7} {adp_f:>4}w {rlx_f:>4}w {ml_f:>4}w {oc}")

# Convergence: how many tickers are shared between approaches each week
print(f"\n  Weekly convergence (tickers shared between approaches):")
for entry in rebalance_log[:5]:
    adp_set = set(entry['adp_tickers'])
    rlx_set = set(entry['rlx_tickers'])
    ml_set = set(entry['ml_tickers'])
    adp_rlx = len(adp_set & rlx_set)
    adp_ml = len(adp_set & ml_set)
    rlx_ml = len(rlx_set & ml_set)
    all3 = len(adp_set & rlx_set & ml_set)
    print(f"  W{entry['week']:>2}: ADP∩RLX={adp_rlx} ADP∩ML={adp_ml} RLX∩ML={rlx_ml} ALL3={all3}")
if len(rebalance_log) > 5:
    print(f"    ... ({len(rebalance_log)-5} more weeks)")


# ==============================================================================
# SECTION 4: ML_DYNAMIC Diagnostics
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 4: ML_DYNAMIC — Model Selection & Feature Importance")
print(f"{'='*W}")

if weekly_model_info:
    print(f"\n  {'Wk':>3} {'Model':<6} {'CV rho':>8} {'N_train':>8} {'GBM':>10} {'Ridge':>10} {'RF':>10}")
    print(f"  {'-'*65}")
    model_wins = defaultdict(int)
    for info in weekly_model_info:
        model_wins[info['model']] += 1
        metrics_str = {}
        for name in ['GBM', 'Ridge', 'RF']:
            if name in info['metrics']:
                m = info['metrics'][name]
                metrics_str[name] = f"{m['rho']:+.3f}"
            else:
                metrics_str[name] = '   -'
        print(f"  W{info['week']:>2} {info['model']:<6} {info['rho']:>+7.3f} {info['n_train']:>8} "
              f"{metrics_str.get('GBM', '   -'):>10} {metrics_str.get('Ridge', '   -'):>10} {metrics_str.get('RF', '   -'):>10}")

    print(f"\n  Model wins: {dict(model_wins)}")
    cv_rhos = [info['rho'] for info in weekly_model_info]
    print(f"  Avg CV rho: {np.mean(cv_rhos):+.3f} (std={np.std(cv_rhos):.3f})")

if weekly_importances:
    avg_imp = defaultdict(list)
    for imp_dict in weekly_importances:
        for feat, val in imp_dict.items():
            avg_imp[feat].append(val)
    print(f"\n  Top 10 features by mean importance:")
    sorted_imp = sorted([(f, np.mean(v)) for f, v in avg_imp.items()],
                        key=lambda x: x[1], reverse=True)
    for rank, (feat, imp) in enumerate(sorted_imp[:10], 1):
        category = 'BQ' if feat in FEATURE_NAMES_BQ_8 else \
                   'PAPER' if feat in FEATURE_NAMES_PAPER else \
                   'SIGNAL' if feat in FEATURE_NAMES_SIGNAL else 'OHLCV'
        print(f"    #{rank:>2} {feat:<25} {imp:.4f}  [{category}]")


# ==============================================================================
# SECTION 5: Monthly PnL Breakdown
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 5: Monthly PnL Breakdown")
print(f"{'='*W}")

months = ['Apr', 'May', 'Jun', 'Jul']
month_ranges = [
    (date_type(2026, 4, 1), date_type(2026, 4, 30)),
    (date_type(2026, 5, 1), date_type(2026, 5, 31)),
    (date_type(2026, 6, 1), date_type(2026, 6, 30)),
    (date_type(2026, 7, 1), date_type(2026, 7, 23)),
]

print(f"\n  {'Strategy':<18} {'Apr':>8} {'May':>8} {'Jun':>8} {'Jul':>8}")
print(f"  {'-'*56}")
monthly_pnl = {sn: [] for sn in STRAT_NAMES}
for sn in STRAT_NAMES:
    row_vals = []
    for m_name, (m_start, m_end) in zip(months, month_ranges):
        st = compute_bt_stats(strat_trades[sn], m_start, m_end)
        row_vals.append(st['total'])
    monthly_pnl[sn] = row_vals
    tag = ' [ceiling]' if sn == 'ORACLE' else ''
    print(f"  {sn:<18} {row_vals[0]:>+7.1%} {row_vals[1]:>+7.1%} {row_vals[2]:>+7.1%} {row_vals[3]:>+7.1%}{tag}")

# -- VERDICT --
print(f"\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

best_sn, best_pnl = candidate_ranking[0]
control_pnl = h2_stats['ADAPTIVE']['total']
current_pnl = h2_stats['CURRENT']['total']

print(f"\n  ORACLE ceiling (H2):     {oracle_ceiling:>+7.1%}")
print(f"  CURRENT portfolio (H2):  {current_pnl:>+7.1%}")
print(f"  ADAPTIVE control (H2):   {control_pnl:>+7.1%}")
print(f"\n  WINNER: {best_sn} with H2 = {best_pnl:>+7.1%}")

if best_sn != 'ADAPTIVE':
    delta = best_pnl - control_pnl
    print(f"  Improvement over ADAPTIVE control: {delta:>+.1%}")
else:
    print(f"  Neither candidate beat the ADAPTIVE control")

rlx_delta = h2_stats['ADAPTIVE_RELAXED']['total'] - control_pnl
ml_delta = h2_stats['ML_DYNAMIC']['total'] - control_pnl
print(f"\n  ADAPTIVE_RELAXED vs control: {rlx_delta:>+.1%}  (relaxed gates effect)")
print(f"  ML_DYNAMIC vs control:       {ml_delta:>+.1%}  (3-model + paper features)")

rlx_avg_ov = np.mean(rlx_overlaps)
adp_avg_ov = np.mean(adp_overlaps)
ml_avg_ov = np.mean(ml_overlaps)
print(f"\n  Oracle overlap improvement:")
print(f"    ADAPTIVE_RELAXED: {rlx_avg_ov:.1f}/{TOP_K} (vs {adp_avg_ov:.1f}/{TOP_K} control = {rlx_avg_ov-adp_avg_ov:>+.1f})")
print(f"    ML_DYNAMIC:       {ml_avg_ov:.1f}/{TOP_K} (vs {adp_avg_ov:.1f}/{TOP_K} control = {ml_avg_ov-adp_avg_ov:>+.1f})")

if best_pnl > current_pnl and h2_stats[best_sn]['avg'] > 0:
    print(f"\n  RECOMMENDATION: Deploy {best_sn} — beats CURRENT by {best_pnl - current_pnl:>+.1%}")
else:
    print(f"\n  RECOMMENDATION: Keep CURRENT portfolio — no candidate improves enough")


# ==============================================================================
# CHARTS (2x3)
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('SHORT Ticker Selection — Candidate Comparison', fontsize=14, fontweight='bold', y=0.98)

# -- (0,0): Equity curves --
ax = axes[0, 0]
for sn in STRAT_NAMES:
    trades_sorted = sorted(strat_trades[sn], key=lambda t: t['date'])
    if not trades_sorted:
        continue
    dates = [t['date'] for t in trades_sorted]
    cum_pnl = np.cumsum([t['pnl'] for t in trades_sorted])
    ax.plot(dates, cum_pnl, label=sn, color=STRAT_COLORS[sn],
            linewidth=2.5 if sn in ('ADAPTIVE_RELAXED', 'ML_DYNAMIC') else 1.5,
            alpha=1.0 if sn in ('ADAPTIVE_RELAXED', 'ML_DYNAMIC', 'ORACLE') else 0.6)
ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
ax.axvline(x=pd.Timestamp(HALF2_START).date(), color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax.set_title('Equity Curves')
ax.set_ylabel('Cumulative PnL')
ax.legend(fontsize=7, loc='upper left')
ax.tick_params(axis='x', rotation=45, labelsize=7)

# -- (0,1): H2 total PnL bar chart --
ax = axes[0, 1]
h2_vals = [h2_stats[sn]['total'] for sn in STRAT_NAMES]
colors = [STRAT_COLORS[sn] for sn in STRAT_NAMES]
bars = ax.bar(range(len(STRAT_NAMES)), h2_vals, color=colors)
ax.set_xticks(range(len(STRAT_NAMES)))
ax.set_xticklabels([s.replace('ADAPTIVE_RELAXED', 'ADP_RLX').replace('ML_DYNAMIC', 'ML_DYN')
                    for s in STRAT_NAMES], fontsize=8)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('H2 Total PnL')
ax.set_ylabel('Total PnL')
for bar, val in zip(bars, h2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:+.0%}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)

# -- (0,2): Oracle overlap over weeks --
ax = axes[0, 2]
week_nums = [e['week'] for e in rebalance_log]
ax.plot(week_nums, adp_overlaps, 'o-', label='ADAPTIVE', color=STRAT_COLORS['ADAPTIVE'],
        linewidth=1.5, markersize=4)
ax.plot(week_nums, rlx_overlaps, 's-', label='ADP_RELAXED', color=STRAT_COLORS['ADAPTIVE_RELAXED'],
        linewidth=2, markersize=5)
ax.plot(week_nums, ml_overlaps, '^-', label='ML_DYNAMIC', color=STRAT_COLORS['ML_DYNAMIC'],
        linewidth=2, markersize=5)
ax.axhline(y=TOP_K, color='green', linewidth=1, linestyle='--', alpha=0.3, label=f'Perfect ({TOP_K})')
ax.set_title('Oracle Overlap per Week')
ax.set_ylabel(f'Overlap (out of {TOP_K})')
ax.set_xlabel('Week')
ax.set_ylim(-0.5, TOP_K + 0.5)
ax.legend(fontsize=7)

# -- (1,0): Ticker selection heatmap --
ax = axes[1, 0]
all_heatmap_tickers = set()
for entry in rebalance_log:
    all_heatmap_tickers.update(entry['adp_tickers'])
    all_heatmap_tickers.update(entry['rlx_tickers'])
    all_heatmap_tickers.update(entry['ml_tickers'])
heatmap_tickers = sorted(all_heatmap_tickers)
n_ht = len(heatmap_tickers)
n_wk = len(rebalance_log)

if n_ht > 0 and n_wk > 0:
    # Use 3 rows per week (ADP, RLX, ML) stacked
    heatmap = np.zeros((n_ht, n_wk * 3))
    for wi, entry in enumerate(rebalance_log):
        for ti, t in enumerate(heatmap_tickers):
            if t in entry['adp_tickers']:
                heatmap[ti, wi*3] = 1
            if t in entry['rlx_tickers']:
                heatmap[ti, wi*3+1] = 2
            if t in entry['ml_tickers']:
                heatmap[ti, wi*3+2] = 3
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', STRAT_COLORS['ADAPTIVE'],
                           STRAT_COLORS['ADAPTIVE_RELAXED'], STRAT_COLORS['ML_DYNAMIC']])
    ax.imshow(heatmap, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=3)

    # Mark oracle tickers
    for ti, t in enumerate(heatmap_tickers):
        if t in oracle_set:
            ax.text(-0.5, ti, '*', fontsize=8, color='green', ha='right', va='center', fontweight='bold')

    ax.set_yticks(range(n_ht))
    ax.set_yticklabels(heatmap_tickers, fontsize=5)
    ax.set_xlabel('Week (ADP|RLX|ML)')
    ax.set_title('Ticker Selection Heatmap')
else:
    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

# -- (1,1): Monthly PnL grouped bars --
ax = axes[1, 1]
x = np.arange(len(months))
bar_width = 0.15
active_strats = ['ADAPTIVE', 'ADAPTIVE_RELAXED', 'ML_DYNAMIC', 'ORACLE', 'CURRENT']
for i, sn in enumerate(active_strats):
    vals = monthly_pnl[sn]
    offset = (i - len(active_strats)/2 + 0.5) * bar_width
    short_name = sn.replace('ADAPTIVE_RELAXED', 'ADP_RLX').replace('ML_DYNAMIC', 'ML_DYN')
    ax.bar(x + offset, vals, bar_width, label=short_name, color=STRAT_COLORS[sn], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(months)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('Monthly PnL')
ax.set_ylabel('Total PnL')
ax.legend(fontsize=6)

# -- (1,2): ML_DYNAMIC model diagnostics --
ax = axes[1, 2]
if weekly_model_info:
    ml_weeks = [info['week'] for info in weekly_model_info]
    ml_cv_rhos = [info['rho'] for info in weekly_model_info]
    ml_models = [info['model'] for info in weekly_model_info]

    model_colors = {'GBM': '#e11d48', 'Ridge': '#2563eb', 'RF': '#16a34a', 'BURN_IN': '#a3a3a3', 'FALLBACK': '#a3a3a3'}
    for i, (w, rho, model) in enumerate(zip(ml_weeks, ml_cv_rhos, ml_models)):
        ax.bar(w, rho, color=model_colors.get(model, '#6b7280'), alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('ML_DYNAMIC: CV Rho per Week')
    ax.set_xlabel('Week')
    ax.set_ylabel('CV Spearman Rho')

    # Legend for model colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=model_colors[m], label=m)
                       for m in ['GBM', 'Ridge', 'RF'] if m in set(ml_models)]
    ax.legend(handles=legend_elements, fontsize=7)
else:
    ax.text(0.5, 0.5, 'No ML data', ha='center', va='center', transform=ax.transAxes)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(CHART_PATH, dpi=150, bbox_inches='tight')
print(f"\n  Charts saved to {CHART_PATH}")
if _IN_COLAB:
    plt.show()
else:
    plt.close()

print(f"\n{'='*W}")
print(f"  DONE")
print(f"{'='*W}")
