"""
backtest_short_compare.py — Dynamic ML Walk-Forward SHORT Ticker Selector
==========================================================================
Walk-forward weekly ticker selection backtest for SHORT models.

Three strategies compared:
  ML_DYNAMIC  — GBM/Ridge/RF retrained weekly on panel data (ticker × week)
                Dynamic features: OHLCV momentum/vol, signal frequency, paper-trade PnL
                Target = paper-trade PnL (NO data leakage)
  FILTERED    — Rule-based unbiased: excl limit_entry, overfit<=2.0, va_pos_rate>=0.50
  ORACLE      — Top-K by val_b (biased ceiling — uses test-period data) [B]

Paper-trades ALL tickers every day regardless of ML selection.
Per-seed config selection (avoids ensemble val_a bug).
CSV scoring only. No GPU required. Colab-ready.
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

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

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1')
BQ_ALLOWED_MODES  = ('static', 'atr_sl', 'grace')

TOP_K = 8
BURN_IN_WEEKS = 2
FILTERED_VA_OPT_RATIO = 2.0
FILTERED_MIN_CONFIGS = 5
FILTERED_VA_POS_RATE = 0.50
MIN_SEEDS = 3

OHLCV_LOOKBACK_DAYS = 45

STICKY_RETAIN = 4

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

FEATURE_NAMES_OHLCV = [
    'ret_5d', 'ret_10d', 'ret_20d',
    'rvol_5d', 'rvol_20d', 'rvol_ratio',
    'drawdown_20d', 'volume_ratio', 'intraday_range_5d',
]
FEATURE_NAMES_SIGNAL = [
    'signal_freq_20d', 'signal_avg_prob_20d',
    'signal_freq_5d', 'signal_prob_trend',
]
FEATURE_NAMES_PAPER = [
    'paper_pnl_mean', 'paper_pnl_recent', 'paper_wr',
    'paper_n_trades', 'paper_sharpe', 'paper_best',
]
FEATURE_NAMES_BQ = [
    'n_configs', 'has_both_alphas', 'cross_alpha_agree', 'agree_18_best',
    'opt_wr_median', 'pct_static', 'pct_atr_sl', 'pct_grace',
]
ALL_FEATURE_NAMES = (FEATURE_NAMES_OHLCV + FEATURE_NAMES_SIGNAL
                     + FEATURE_NAMES_PAPER + FEATURE_NAMES_BQ)
N_DYNAMIC_FEATURES = (len(FEATURE_NAMES_OHLCV) + len(FEATURE_NAMES_SIGNAL)
                      + len(FEATURE_NAMES_PAPER))

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110


# ==============================================================================
# HELPERS
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
    for c in ['Volume']:
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
    """Resample 1-min master data to daily bars for OHLCV feature computation."""
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
    """Per-seed aggregated config selection. Avoids ensemble val_a bug."""
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


def select_filtered_tickers(bq_df, top_k):
    """Rule-based FILTERED strategy: no limit_entry, overfit<=2.0, va_pos_rate>=0.50."""
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


def compute_oracle(bq_df, top_k):
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = bq_df[bq_df['is_ensemble'] != True].copy()
    vb = indiv.groupby('ticker').agg(vb_median=('val_b_avg_pnl', 'median'))
    return vb.nlargest(top_k, 'vb_median').index.tolist()


# ==============================================================================
# STATIC BQ FEATURES (8 structural features, computed once)
# ==============================================================================
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
        })

    return pd.DataFrame(rows).set_index('ticker').fillna(0)


# ==============================================================================
# DYNAMIC FEATURE ENGINE
# ==============================================================================
def compute_ohlcv_features(ticker, as_of_date):
    """9 OHLCV features from daily bars up to as_of_date."""
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
        if len(close) < n + 1:
            return 0.0
        return (close[-1] - close[-(n+1)]) / close[-(n+1)] if close[-(n+1)] != 0 else 0.0

    def _rvol(n):
        if len(close) < n + 1:
            return 0.0
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
        'ret_5d': _ret(5),
        'ret_10d': _ret(10),
        'ret_20d': _ret(20),
        'rvol_5d': rvol_5,
        'rvol_20d': rvol_20,
        'rvol_ratio': rvol_5 / max(rvol_20, 1e-9),
        'drawdown_20d': dd_20,
        'volume_ratio': vol_ratio,
        'intraday_range_5d': float(intra),
    }


def compute_signal_features(daily_probs, prob_thresh, as_of_date):
    """4 signal features from daily prob cache."""
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


def compute_paper_features(paper_trades):
    """6 features from paper-trade history."""
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


def build_weekly_features(tickers, static_feats, daily_probs_cache, configs,
                          paper_history, as_of_date):
    """Build feature DataFrame for all tickers at a given week boundary."""
    rows = []
    for ticker in tickers:
        feat = {}
        feat.update(compute_ohlcv_features(ticker, as_of_date))
        probs = daily_probs_cache.get(ticker, {})
        thresh = configs[ticker]['prob_thresh'] if ticker in configs else 0.5
        feat.update(compute_signal_features(probs, thresh, as_of_date))
        feat.update(compute_paper_features(paper_history.get(ticker, [])))
        if ticker in static_feats.index:
            for col in FEATURE_NAMES_BQ:
                feat[col] = static_feats.at[ticker, col] if col in static_feats.columns else 0.0
        else:
            for col in FEATURE_NAMES_BQ:
                feat[col] = 0.0
        rows.append({'ticker': ticker, **feat})

    df = pd.DataFrame(rows).set_index('ticker')
    df = df.reindex(columns=ALL_FEATURE_NAMES).fillna(0)
    return df


# ==============================================================================
# PAPER TRADING ENGINE
# ==============================================================================
def paper_trade_week(tickers, week_days, configs, ticker_data,
                     daily_probs_cache):
    """Paper-trade ALL tickers for all days in the week."""
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


# ==============================================================================
# ML TRAINING — PANEL DATA
# ==============================================================================
def build_panel_dataset(weekly_features_history, paper_pnl_by_week, tickers):
    """Flatten (ticker, week) pairs into training set.
    weekly_features_history: list of (week_idx, features_df)
    paper_pnl_by_week: dict[week_idx] -> dict[ticker] -> mean_pnl
    """
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
    """Leave-one-out CV Spearman rho. Groups by ticker (every N_TICKERS-th row)."""
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


def train_and_predict(X_train, y_train, X_current, current_tickers, top_k):
    """Train 3 models, pick best by LOO-CV Spearman rho, predict and select top-K."""
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

    importances_dict = dict(zip(ALL_FEATURE_NAMES, best_importances))

    return selected, pred_df, best_name, importances_dict, model_metrics, best_rho


# ==============================================================================
# WALK-FORWARD HELPERS
# ==============================================================================
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


def summarize_trades(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0, 'dd': 0}
    df = pd.DataFrame(trades_list).sort_values('date')
    cum = df['pnl'].cumsum()
    return {
        'label': label, 'n': len(df),
        'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(),
        'dd': (cum - cum.cummax()).min(),
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT BACKTEST — Dynamic ML Walk-Forward (v4)")
print(f"  ML: {', '.join(MODELS.keys())} retrained weekly on panel data (ticker x week)")
print(f"  Features: {len(ALL_FEATURE_NAMES)} ({N_DYNAMIC_FEATURES} dynamic + {len(FEATURE_NAMES_BQ)} static BQ)")
print(f"  Target: paper-trade PnL (no leakage)")
print(f"  Paper-trades ALL tickers every day")
print(f"  Config: per-seed aggregated (avoids ensemble val_a bug)")
print(f"  ORACLE: val_b [B] | FILTERED: rule-based unbiased")
print(f"  Test period: {TEST_START} to {TEST_END}")
print(f"{'='*W}")


# -- PHASE 0: Load data --
print(f"\n  PHASE 0: Loading data...")

print(f"    Querying BQ...")
bq_df = query_all_bq()
all_tickers = sorted(bq_df['ticker'].unique())
n_tickers = len(all_tickers)
print(f"    {len(bq_df)} BQ rows across {n_tickers} tickers")

print(f"\n    Selecting per-seed configs...")
all_configs = select_configs_perseed(bq_df, all_tickers)
print(f"    {len(all_configs)} tickers have valid per-seed configs")
for tk, cfg in sorted(all_configs.items())[:5]:
    print(f"      {tk}: {cfg['tpsl_mode']} tp={cfg['tp_pct']:.1%} "
          f"sl={cfg['sl_pct']:.1%} prob={cfg['prob_thresh']:.3f} "
          f"entry={cfg['entry_minute']}m")
if len(all_configs) > 5:
    print(f"      ... ({len(all_configs) - 5} more)")

print(f"\n    Computing static BQ features...")
static_feats = compute_static_bq_features(bq_df)
print(f"    {len(static_feats)} tickers x {len(FEATURE_NAMES_BQ)} static features")

print(f"\n    Computing ticker lists...")
oracle_tickers = compute_oracle(bq_df, TOP_K)
filtered_tickers = select_filtered_tickers(bq_df, TOP_K)
tradeable_tickers = sorted(set(all_configs.keys()))
print(f"    ORACLE (top {TOP_K} by val_b): {', '.join(oracle_tickers)} [B]")
print(f"    FILTERED (top {TOP_K} rule-based): {', '.join(filtered_tickers)}")
print(f"    Tradeable (have configs): {len(tradeable_tickers)}")

print(f"\n    Loading OHLCV for {n_tickers} tickers...")
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
print(f"    Loaded {len(ticker_data)} tickers with OHLCV")

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

print(f"\n    Precomputing daily probabilities...")
daily_probs_cache = {}
for ticker, cfg in all_configs.items():
    if ticker not in ticker_data:
        continue
    probs = compute_daily_probs(ticker, cfg, csv_scores_cache)
    if probs:
        daily_probs_cache[ticker] = probs
print(f"    {len(daily_probs_cache)} tickers with daily probs")

_release_memory()


# -- PHASE 1+2: Walk-Forward --
weeks = generate_weeks(TEST_START, TEST_END)
trading_days_list = pd.bdate_range(TEST_START, TEST_END).date

print(f"\n{'='*W}")
print(f"  PHASE 1-2: Walk-Forward Simulation ({len(weeks)} weeks, burn-in={BURN_IN_WEEKS})")
print(f"{'='*W}")

STRAT_NAMES = ['ML_DYNAMIC', 'FILTERED', 'ORACLE']
strategies = {s: {'trades': [], 'weekly_tickers': []} for s in STRAT_NAMES}

paper_history = defaultdict(list)
paper_pnl_by_week = {}
weekly_features_history = []
weekly_importances = []
weekly_model_info = []

print(f"\n  {'Wk':>3} {'Period':<15} | {'Strategy':<14} | {'Tk':>3} | {'Trd':>4} {'PnL':>8}")
print(f"  {'-'*65}")

for week_idx, (wk_start, wk_end) in enumerate(weeks):
    week_days = [d for d in trading_days_list if wk_start <= d <= wk_end]
    if not week_days:
        continue

    # -- Compute features every week (including burn-in) --
    feat_df = build_weekly_features(
        tradeable_tickers, static_feats, daily_probs_cache,
        all_configs, paper_history, wk_start)
    weekly_features_history.append((week_idx, feat_df))

    # -- Ticker selection for this week --
    if week_idx < BURN_IN_WEEKS:
        ml_active = [t for t in filtered_tickers
                     if t in all_configs and t in ticker_data]
        print(f"  {'':>3} {'':15}   [BURN-IN W{week_idx+1}] using FILTERED: {','.join(ml_active)}")
    else:
        X_train, y_train = build_panel_dataset(
            weekly_features_history[:-1],
            paper_pnl_by_week,
            tradeable_tickers)

        if X_train is not None and len(X_train) >= 10:
            selected, pred_df, best_model, importances, metrics, cv_rho = \
                train_and_predict(X_train, y_train,
                                  feat_df.values.astype(np.float64),
                                  feat_df.index.tolist(), TOP_K)

            # -- Fix 3: Sticky incumbents -- retain top STICKY_RETAIN with positive paper PnL --
            prev_tickers = []
            if strategies['ML_DYNAMIC']['weekly_tickers']:
                prev_tickers = strategies['ML_DYNAMIC']['weekly_tickers'][-1]['tickers']
            if prev_tickers and STICKY_RETAIN > 0:
                incumbent_scores = []
                for tk in prev_tickers:
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
                new_from_ml = [t for t in selected
                              if t not in retained and t in all_configs and t in ticker_data]
                ml_active = retained + new_from_ml[:n_new]
                sticky_msg = f" [sticky:{','.join(retained)}]" if retained else ""
            else:
                ml_active = [t for t in selected
                             if t in all_configs and t in ticker_data]
                sticky_msg = ""

            weekly_importances.append(importances)
            train_rho_best = metrics[best_model]['train_rho']
            weekly_model_info.append({
                'week': week_idx + 1, 'model': best_model,
                'rho': cv_rho, 'train_rho': train_rho_best,
                'n_train': len(X_train), 'metrics': metrics,
            })

            oracle_overlap = len(set(ml_active) & set(oracle_tickers))
            top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
            feat_str = ', '.join(f'{k}={v:.2f}' for k, v in top_feats)
            model_strs = ' | '.join(
                f"{n}:cv={m['rho']:+.3f}/tr={m['train_rho']:+.3f}"
                for n, m in metrics.items())
            print(f"  {'':>3} {'':15}   [ML W{week_idx+1}] Train: {len(X_train)} obs | "
                  f"Best: {best_model} CV_rho={cv_rho:+.3f} train_rho={train_rho_best:+.3f} "
                  f"gap={train_rho_best-cv_rho:.3f}")
            print(f"  {'':>3} {'':15}   [ML W{week_idx+1}] Models: {model_strs}")
            print(f"  {'':>3} {'':15}   [ML W{week_idx+1}] Top feats: {feat_str}")
            print(f"  {'':>3} {'':15}   [ML W{week_idx+1}] Selected: {','.join(ml_active)} | "
                  f"Oracle overlap: {oracle_overlap}/{TOP_K}{sticky_msg}")
        else:
            ml_active = [t for t in filtered_tickers
                         if t in all_configs and t in ticker_data]
            print(f"  {'':>3} {'':15}   [ML W{week_idx+1}] "
                  f"Insufficient training data ({len(X_train) if X_train is not None else 0} obs), "
                  f"using FILTERED")

    oracle_active = [t for t in oracle_tickers
                     if t in all_configs and t in ticker_data]
    filtered_active = [t for t in filtered_tickers
                       if t in all_configs and t in ticker_data]

    strategy_tickers = {
        'ML_DYNAMIC': ml_active,
        'FILTERED':   filtered_active,
        'ORACLE':     oracle_active,
    }

    for sname in STRAT_NAMES:
        strategies[sname]['weekly_tickers'].append({
            'week': week_idx + 1,
            'start': wk_start, 'end': wk_end,
            'tickers': strategy_tickers[sname],
        })

    # -- Paper-trade ALL tickers + real trades --
    week_paper = paper_trade_week(
        tradeable_tickers, week_days, all_configs,
        ticker_data, daily_probs_cache)

    week_pnl_map = {}
    n_paper_traded = 0
    paper_pnls_all = []
    for ticker, trades in week_paper.items():
        paper_history[ticker].extend(trades)
        if trades:
            n_paper_traded += 1
            avg = np.mean([t['pnl'] for t in trades])
            week_pnl_map[ticker] = avg
            paper_pnls_all.extend([t['pnl'] for t in trades])
    paper_pnl_by_week[week_idx] = week_pnl_map

    paper_avg = np.mean(paper_pnls_all) if paper_pnls_all else 0
    paper_best_tk = max(week_pnl_map.items(), key=lambda x: x[1])[0] \
        if week_pnl_map else '-'
    paper_best_val = max(week_pnl_map.values()) if week_pnl_map else 0
    print(f"  {'':>3} {'':15}   [PAPER W{week_idx+1}] "
          f"{n_paper_traded}/{len(tradeable_tickers)} traded, "
          f"avg {paper_avg:+.2%}, best {paper_best_tk} {paper_best_val:+.2%}")

    # -- Assign trades to strategies --
    for sname in STRAT_NAMES:
        for ticker in strategy_tickers[sname]:
            if ticker in week_paper:
                for trade in week_paper[ticker]:
                    t = trade.copy()
                    t['strategy'] = sname
                    t['week'] = week_idx + 1
                    strategies[sname]['trades'].append(t)

    # -- Print week summary --
    for si, sname in enumerate(STRAT_NAMES):
        n_tk = len(strategy_tickers[sname])
        wt = [t for t in strategies[sname]['trades'] if t['week'] == week_idx + 1]
        n_trades = len(wt)
        week_pnl = sum(t['pnl'] for t in wt)

        period_str = f"{wk_start.strftime('%m/%d')}-{wk_end.strftime('%m/%d')}" if si == 0 else ""
        wk_str = f"W{week_idx+1}" if si == 0 else ""
        bias = " [B]" if sname == 'ORACLE' else ""
        print(f"  {wk_str:>3} {period_str:<15} | {sname+bias:<14} | {n_tk:>3} | "
              f"{n_trades:>4} {week_pnl:>+7.1%}")

    if week_idx < len(weeks) - 1:
        print(f"  {'':>3} {'':15} | {'':14} | {'':>3} |")


# -- PHASE 3: Results --
print(f"\n{'='*W}")
print(f"  PHASE 3: Results")
print(f"{'='*W}")

half1_end_dt = pd.Timestamp(HALF1_END).date()
half2_start_dt = pd.Timestamp(HALF2_START).date()

print(f"\n  {'Strategy':<14} | {'-- FULL --':>25} | "
      f"{'-- HALF1 --':>18} | {'-- HALF2 --':>18}")
print(f"  {'':14} | {'N':>4} {'Tot':>7} {'Avg':>7} {'WR':>5} {'DD':>7} | "
      f"{'N':>4} {'Tot':>7} {'Avg':>7} {'WR':>5} | "
      f"{'N':>4} {'Tot':>7} {'Avg':>7} {'WR':>5}")
print(f"  {'-'*105}")

strategy_summaries = {}
for sname in STRAT_NAMES:
    all_t = strategies[sname]['trades']
    h1_t = [t for t in all_t if t['date'] <= half1_end_dt]
    h2_t = [t for t in all_t if t['date'] >= half2_start_dt]
    f_sum = summarize_trades(all_t, f"{sname}/FULL")
    h1_sum = summarize_trades(h1_t, f"{sname}/HALF1")
    h2_sum = summarize_trades(h2_t, f"{sname}/HALF2")
    strategy_summaries[sname] = {'FULL': f_sum, 'HALF1': h1_sum, 'HALF2': h2_sum}
    bias = " [B]" if sname == "ORACLE" else ""
    print(f"  {sname:<14} | {f_sum['n']:>4} {f_sum['total']:>+6.1%} "
          f"{f_sum['avg']:>+6.2%} {f_sum['wr']:>4.0%} {f_sum['dd']:>+6.1%} | "
          f"{h1_sum['n']:>4} {h1_sum['total']:>+6.1%} "
          f"{h1_sum['avg']:>+6.2%} {h1_sum['wr']:>4.0%} | "
          f"{h2_sum['n']:>4} {h2_sum['total']:>+6.1%} "
          f"{h2_sum['avg']:>+6.2%} {h2_sum['wr']:>4.0%}{bias}")


# -- Per-ticker (ML_DYNAMIC) --
print(f"\n  Per-Ticker (ML_DYNAMIC):")
ml_trades = strategies['ML_DYNAMIC']['trades']
per_tk = defaultdict(list)
for t in ml_trades:
    per_tk[t['ticker']].append(t)

print(f"  {'Ticker':<7} | {'N':>4} {'Avg':>7} {'WR':>5} | "
      f"{'H1_avg':>7} {'H2_avg':>7} {'Status':>8} | {'Paper':>7}")
print(f"  {'-'*70}")
for ticker in sorted(per_tk.keys()):
    trades = per_tk[ticker]
    h1 = [t for t in trades if t['date'] <= half1_end_dt]
    h2 = [t for t in trades if t['date'] >= half2_start_dt]
    avg = np.mean([t['pnl'] for t in trades])
    wr = np.mean([1 if t['pnl'] > 0 else 0 for t in trades])
    h1_avg = np.mean([t['pnl'] for t in h1]) if h1 else 0
    h2_avg = np.mean([t['pnl'] for t in h2]) if h2 else 0
    status = "HOLDS" if h1_avg > 0 and h2_avg > 0 else ("DECAYED" if h1_avg > 0 else "WEAK")
    paper_all = paper_history.get(ticker, [])
    paper_avg = np.mean([t['pnl'] for t in paper_all]) if paper_all else 0
    print(f"  {ticker:<7} | {len(trades):>4} {avg:>+6.2%} {wr:>4.0%} | "
          f"{h1_avg:>+6.2%} {h2_avg:>+6.2%} {status:>8} | {paper_avg:>+6.2%}")


# -- Weekly ticker selection --
print(f"\n  ML_DYNAMIC: Weekly Ticker Selection:")
for wk_info in strategies['ML_DYNAMIC']['weekly_tickers']:
    tks = sorted(wk_info['tickers'])
    prev_idx = wk_info['week'] - 2
    delta = ""
    if prev_idx >= 0:
        prev_tks = set(strategies['ML_DYNAMIC']['weekly_tickers'][prev_idx]['tickers'])
        added = sorted(set(tks) - prev_tks)
        removed = sorted(prev_tks - set(tks))
        if added:
            delta += f"  [+{','.join(added)}]"
        if removed:
            delta += f"  [-{','.join(removed)}]"
    print(f"    W{wk_info['week']:>2} ({wk_info['start'].strftime('%m/%d')}-"
          f"{wk_info['end'].strftime('%m/%d')}): {', '.join(tks)}{delta}")


# -- Monthly --
print(f"\n  Monthly Comparison:")
print(f"  {'Month':<8} | {'-- ML_DYNAMIC --':>18} | {'-- FILTERED ----':>18} | "
      f"{'-- ORACLE [B] --':>18}")
print(f"  {'':8} | {'N':>3} {'Tot':>7} {'Avg':>7} | {'N':>3} {'Tot':>7} {'Avg':>7} | "
      f"{'N':>3} {'Tot':>7} {'Avg':>7}")
print(f"  {'-'*80}")

months = sorted(set(
    t['date'].strftime('%Y-%m')
    for s in strategies.values() for t in s['trades']
))
for month in months:
    parts = {}
    for sname in STRAT_NAMES:
        mt = [t for t in strategies[sname]['trades']
              if t['date'].strftime('%Y-%m') == month]
        if mt:
            parts[sname] = {'n': len(mt), 'tot': sum(t['pnl'] for t in mt),
                           'avg': np.mean([t['pnl'] for t in mt])}
        else:
            parts[sname] = {'n': 0, 'tot': 0, 'avg': 0}
    ml = parts['ML_DYNAMIC']
    fi = parts['FILTERED']
    oc = parts['ORACLE']
    print(f"  {month:<8} | {ml['n']:>3} {ml['tot']:>+6.1%} {ml['avg']:>+6.2%} | "
          f"{fi['n']:>3} {fi['tot']:>+6.1%} {fi['avg']:>+6.2%} | "
          f"{oc['n']:>3} {oc['tot']:>+6.1%} {oc['avg']:>+6.2%}")


# -- Ticker overlap analysis --
print(f"\n  Ticker Overlap (ML_DYNAMIC vs ORACLE by week):")
for wk_info in strategies['ML_DYNAMIC']['weekly_tickers']:
    w = wk_info['week']
    ml_set = set(wk_info['tickers'])
    oc_wk = [wk for wk in strategies['ORACLE']['weekly_tickers'] if wk['week'] == w]
    oc_set = set(oc_wk[0]['tickers']) if oc_wk else set()
    overlap = ml_set & oc_set
    ml_only = sorted(ml_set - oc_set)
    oc_only = sorted(oc_set - ml_set)
    print(f"    W{w:>2}: {len(overlap)}/{TOP_K} overlap  "
          f"ML-only:[{','.join(ml_only) if ml_only else '-'}]  "
          f"OC-only:[{','.join(oc_only) if oc_only else '-'}]")


# -- Feature importance stability --
if weekly_importances:
    print(f"\n  Feature Importance Stability (averaged across {len(weekly_importances)} weeks):")
    avg_imp = defaultdict(list)
    for imp_dict in weekly_importances:
        for feat, val in imp_dict.items():
            avg_imp[feat].append(val)
    imp_table = []
    for feat in ALL_FEATURE_NAMES:
        vals = avg_imp.get(feat, [0])
        imp_table.append({
            'feature': feat,
            'avg_imp': np.mean(vals),
            'std_imp': np.std(vals),
            'max_imp': np.max(vals),
        })
    imp_table.sort(key=lambda x: x['avg_imp'], reverse=True)
    print(f"  {'Feature':<22} | {'Avg':>6} {'Std':>6} {'Max':>6} | {'Category':>10}")
    print(f"  {'-'*65}")
    for row in imp_table[:15]:
        f = row['feature']
        cat = ('OHLCV' if f in FEATURE_NAMES_OHLCV
               else 'SIGNAL' if f in FEATURE_NAMES_SIGNAL
               else 'PAPER' if f in FEATURE_NAMES_PAPER
               else 'BQ')
        print(f"  {f:<22} | {row['avg_imp']:>5.3f} {row['std_imp']:>5.3f} "
              f"{row['max_imp']:>5.3f} | {cat:>10}")


# -- Model selection log (shows overfitting gap) --
if weekly_model_info:
    print(f"\n  Model Selection Log (CV rho = generalization, train rho = memorization, gap = overfit):")
    print(f"  {'Week':>4} | {'Model':>6} | {'CV_rho':>7} | {'Tr_rho':>7} | {'Gap':>6} | {'N_train':>8} | {'Diagnosis':>15}")
    print(f"  {'-'*80}")
    for info in weekly_model_info:
        cv_r = info['rho']
        tr_r = info.get('train_rho', cv_r)
        gap = tr_r - cv_r
        if gap > 0.5:
            diag = 'OVERFIT_SEVERE'
        elif gap > 0.3:
            diag = 'OVERFIT_HIGH'
        elif gap > 0.15:
            diag = 'OVERFIT_MILD'
        elif cv_r < 0:
            diag = 'NO_SIGNAL'
        else:
            diag = 'OK'
        print(f"  W{info['week']:>3} | {info['model']:>6} | "
              f"{cv_r:>+6.3f} | {tr_r:>+6.3f} | {gap:>5.3f} | "
              f"{info['n_train']:>8} | {diag:>15}")
    avg_cv = np.mean([i['rho'] for i in weekly_model_info])
    avg_tr = np.mean([i.get('train_rho', i['rho']) for i in weekly_model_info])
    print(f"  {'AVG':>4} | {'':>6} | {avg_cv:>+6.3f} | {avg_tr:>+6.3f} | "
          f"{avg_tr-avg_cv:>5.3f} | {'':>8} |")
    gbm_wins = sum(1 for i in weekly_model_info if i['model'] == 'GBM')
    ridge_wins = sum(1 for i in weekly_model_info if i['model'] == 'Ridge')
    rf_wins = sum(1 for i in weekly_model_info if i['model'] == 'RF')
    print(f"\n  Model wins: GBM={gbm_wins} Ridge={ridge_wins} RF={rf_wins} "
          f"(Ridge/RF winning = CV is working)")


# -- Paper-trade top tickers (ground truth) --
print(f"\n  Paper-Trade Ground Truth (ALL tickers, full period):")
paper_summary = []
for ticker in tradeable_tickers:
    trades = paper_history.get(ticker, [])
    if not trades:
        continue
    pnls = [t['pnl'] for t in trades]
    h1 = [t['pnl'] for t in trades if t['date'] <= half1_end_dt]
    h2 = [t['pnl'] for t in trades if t['date'] >= half2_start_dt]
    paper_summary.append({
        'ticker': ticker, 'n': len(pnls),
        'avg': np.mean(pnls), 'wr': np.mean(np.array(pnls) > 0),
        'h1_avg': np.mean(h1) if h1 else 0,
        'h2_avg': np.mean(h2) if h2 else 0,
        'in_oracle': ticker in oracle_tickers,
        'in_ml': ticker in set(t for wk in strategies['ML_DYNAMIC']['weekly_tickers']
                               for t in wk['tickers']),
    })
paper_summary.sort(key=lambda x: x['avg'], reverse=True)
print(f"  {'Ticker':<7} | {'N':>4} {'Avg':>7} {'WR':>5} | "
      f"{'H1':>7} {'H2':>7} | {'OC':>3} {'ML':>3}")
print(f"  {'-'*60}")
for row in paper_summary[:20]:
    oc = ' Y ' if row['in_oracle'] else ' - '
    ml = ' Y ' if row['in_ml'] else ' - '
    print(f"  {row['ticker']:<7} | {row['n']:>4} {row['avg']:>+6.2%} "
          f"{row['wr']:>4.0%} | {row['h1_avg']:>+6.2%} {row['h2_avg']:>+6.2%} | "
          f"{oc} {ml}")
if len(paper_summary) > 20:
    print(f"  ... ({len(paper_summary) - 20} more)")


# -- June/July diagnostic drill-down --
print(f"\n  JUNE/JULY DIAGNOSTIC DRILL-DOWN:")
print(f"  {'='*75}")
for month_name, month_prefix in [('JUNE', '2026-06'), ('JULY', '2026-07')]:
    print(f"\n  [{month_name}] Paper-trade PnL by ticker (ground truth for ALL tickers):")
    month_paper = []
    for ticker in tradeable_tickers:
        trades = [t for t in paper_history.get(ticker, [])
                  if t['date'].strftime('%Y-%m') == month_prefix]
        if trades:
            pnls = [t['pnl'] for t in trades]
            month_paper.append({
                'ticker': ticker, 'n': len(trades),
                'avg': np.mean(pnls), 'total': sum(pnls),
                'wr': np.mean(np.array(pnls) > 0),
                'best': max(pnls), 'worst': min(pnls),
                'in_oracle': ticker in oracle_tickers,
            })
    if month_paper:
        month_paper.sort(key=lambda x: x['avg'], reverse=True)
        winners = [r for r in month_paper if r['avg'] > 0]
        losers = [r for r in month_paper if r['avg'] <= 0]
        all_pnls = [r['avg'] for r in month_paper]
        print(f"  Universe: {len(month_paper)} tickers traded | "
              f"Winners: {len(winners)} | Losers: {len(losers)} | "
              f"Universe avg: {np.mean(all_pnls):+.2%}")
        print(f"  {'Ticker':<7} | {'N':>3} {'Avg':>7} {'Tot':>7} {'WR':>5} | "
              f"{'Best':>7} {'Worst':>7} | {'OC':>3}")
        print(f"  {'-'*65}")
        for row in month_paper[:10]:
            oc = ' Y ' if row['in_oracle'] else ' - '
            print(f"  {row['ticker']:<7} | {row['n']:>3} {row['avg']:>+6.2%} "
                  f"{row['total']:>+6.1%} {row['wr']:>4.0%} | "
                  f"{row['best']:>+6.2%} {row['worst']:>+6.2%} | {oc}")
        if len(month_paper) > 10:
            print(f"  ... bottom 5:")
            for row in month_paper[-5:]:
                oc = ' Y ' if row['in_oracle'] else ' - '
                print(f"  {row['ticker']:<7} | {row['n']:>3} {row['avg']:>+6.2%} "
                      f"{row['total']:>+6.1%} {row['wr']:>4.0%} | "
                      f"{row['best']:>+6.2%} {row['worst']:>+6.2%} | {oc}")

        for sname in STRAT_NAMES:
            strat_trades = [t for t in strategies[sname]['trades']
                           if t['date'].strftime('%Y-%m') == month_prefix]
            if strat_trades:
                s_avg = np.mean([t['pnl'] for t in strat_trades])
                s_wr = np.mean([1 if t['pnl'] > 0 else 0 for t in strat_trades])
                s_tickers = sorted(set(t['ticker'] for t in strat_trades))
                bias = " [B]" if sname == 'ORACLE' else ""
                print(f"  [{month_name}] {sname}{bias}: {len(strat_trades)}t, "
                      f"avg {s_avg:+.2%}, WR {s_wr:.0%}, tickers: {','.join(s_tickers)}")
    else:
        print(f"  [{month_name}] No trades.")

    oracle_month = [r for r in month_paper if r['in_oracle']]
    non_oracle_month = [r for r in month_paper if not r['in_oracle']]
    if oracle_month:
        oc_avg = np.mean([r['avg'] for r in oracle_month])
        print(f"  [{month_name}] ORACLE tickers avg: {oc_avg:+.2%} "
              f"({len(oracle_month)} tickers)")
    if non_oracle_month:
        noc_avg = np.mean([r['avg'] for r in non_oracle_month])
        print(f"  [{month_name}] Non-ORACLE tickers avg: {noc_avg:+.2%} "
              f"({len(non_oracle_month)} tickers)")


# -- Model rigor summary --
if weekly_model_info:
    print(f"\n  MODEL RIGOR ASSESSMENT:")
    print(f"  {'='*75}")
    cv_rhos = [i['rho'] for i in weekly_model_info]
    tr_rhos = [i.get('train_rho', i['rho']) for i in weekly_model_info]
    gaps = [t - c for t, c in zip(tr_rhos, cv_rhos)]
    positive_cv = sum(1 for r in cv_rhos if r > 0)
    print(f"  CV rho: mean={np.mean(cv_rhos):+.3f}, "
          f"median={np.median(cv_rhos):+.3f}, "
          f"std={np.std(cv_rhos):.3f}, "
          f"positive={positive_cv}/{len(cv_rhos)} weeks")
    print(f"  Train rho: mean={np.mean(tr_rhos):+.3f}, "
          f"median={np.median(tr_rhos):+.3f}")
    print(f"  Overfit gap: mean={np.mean(gaps):.3f}, "
          f"max={np.max(gaps):.3f}, "
          f"min={np.min(gaps):.3f}")
    if np.mean(cv_rhos) < 0.05:
        print(f"  VERDICT: MODEL HAS NO REAL SIGNAL (mean CV rho < 0.05)")
    elif np.mean(gaps) > 0.3:
        print(f"  VERDICT: MODEL IS SEVERELY OVERFITTING (mean gap > 0.3)")
    elif np.mean(cv_rhos) > 0.1 and np.mean(gaps) < 0.2:
        print(f"  VERDICT: MODEL SHOWS REAL SIGNAL (CV rho > 0.1, gap < 0.2)")
    else:
        print(f"  VERDICT: MODEL IS MARGINAL (some signal but high overfit)")


# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle(f'SHORT Backtest: Dynamic ML Walk-Forward  --  {TEST_START} to {TEST_END}\n'
             f'ML: panel data (ticker x week) | Target: paper-trade PnL | '
             f'{len(ALL_FEATURE_NAMES)} features ({N_DYNAMIC_FEATURES} dynamic)',
             fontsize=12, fontweight='bold')

colors = {'ML_DYNAMIC': '#2563eb', 'FILTERED': '#16a34a', 'ORACLE': '#dc2626'}

# Panel 1: Equity curves
ax = axes[0, 0]
for sname in STRAT_NAMES:
    trades = sorted(strategies[sname]['trades'], key=lambda t: t['date'])
    if not trades:
        continue
    dates_arr = [t['date'] for t in trades]
    cum_pnl = np.cumsum([t['pnl'] for t in trades])
    lw = 2.2 if sname == 'ML_DYNAMIC' else 1.5
    ls = '-' if sname != 'ORACLE' else ':'
    label = f"{sname} ({len(trades)}t)" + (" [B]" if sname == 'ORACLE' else "")
    ax.plot(dates_arr, cum_pnl * 100, label=label, color=colors[sname],
            linewidth=lw, linestyle=ls)
ax.axhline(0, color='#555', linewidth=0.5, linestyle='--')
ax.axvline(half2_start_dt, color='#555', linewidth=0.5, linestyle=':', alpha=0.5)
ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('Equity Curve')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Weekly PnL bars
ax = axes[0, 1]
bar_width = 0.25
week_labels = [f"W{i+1}" for i in range(len(weeks))]
x_pos = np.arange(len(weeks))
for si, sname in enumerate(STRAT_NAMES):
    weekly_pnls = []
    for wi in range(len(weeks)):
        wt = [t for t in strategies[sname]['trades'] if t['week'] == wi + 1]
        weekly_pnls.append(sum(t['pnl'] for t in wt) * 100 if wt else 0)
    offset = (si - 1) * bar_width
    ax.bar(x_pos + offset, weekly_pnls, bar_width, label=sname,
           color=colors[sname], alpha=0.8)
ax.axhline(0, color='#555', linewidth=0.5)
ax.set_xlabel('Week')
ax.set_ylabel('Week Total PnL (%)')
ax.set_title('Weekly PnL')
ax.set_xticks(x_pos)
ax.set_xticklabels(week_labels, fontsize=5, rotation=45)
ax.legend(fontsize=7)
ax.grid(True, axis='y', alpha=0.3)

# Panel 3: ML_DYNAMIC ticker selection heatmap
ax = axes[0, 2]
ml_wk = strategies['ML_DYNAMIC']['weekly_tickers']
all_ml_tks = sorted(set(t for wk in ml_wk for t in wk['tickers']))
if all_ml_tks and ml_wk:
    heatmap_data = np.zeros((len(all_ml_tks), len(ml_wk)))
    for wi, wk in enumerate(ml_wk):
        for ti, tk in enumerate(all_ml_tks):
            if tk in wk['tickers']:
                heatmap_data[ti, wi] = 1.0
    ax.imshow(heatmap_data, aspect='auto', cmap='Blues', interpolation='nearest',
              vmin=0, vmax=1)
    ax.set_yticks(range(len(all_ml_tks)))
    ax.set_yticklabels(all_ml_tks, fontsize=7)
    ax.set_xticks(range(len(ml_wk)))
    ax.set_xticklabels([f"W{wk['week']}" for wk in ml_wk], fontsize=6, rotation=45)
    ax.set_title('ML_DYNAMIC: Ticker Selection')
    ax.set_xlabel('Week')
else:
    ax.text(0.5, 0.5, 'No data', ha='center', va='center')

# Panel 4: Feature importance bars (averaged)
ax = axes[1, 0]
if weekly_importances:
    avg_imp_vals = {}
    for feat in ALL_FEATURE_NAMES:
        vals = [d.get(feat, 0) for d in weekly_importances]
        avg_imp_vals[feat] = np.mean(vals)
    sorted_feats = sorted(avg_imp_vals.items(), key=lambda x: x[1], reverse=True)[:12]
    feat_names = [f[0] for f in sorted_feats]
    feat_vals = [f[1] for f in sorted_feats]
    feat_colors = []
    for f in feat_names:
        if f in FEATURE_NAMES_OHLCV:
            feat_colors.append('#2563eb')
        elif f in FEATURE_NAMES_SIGNAL:
            feat_colors.append('#7c3aed')
        elif f in FEATURE_NAMES_PAPER:
            feat_colors.append('#16a34a')
        else:
            feat_colors.append('#d97706')
    y_pos = np.arange(len(feat_names))
    ax.barh(y_pos, feat_vals, color=feat_colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Avg Importance')
    ax.set_title('Feature Importance (avg across weeks)')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2563eb', label='OHLCV'),
        Patch(facecolor='#7c3aed', label='Signal'),
        Patch(facecolor='#16a34a', label='Paper'),
        Patch(facecolor='#d97706', label='BQ'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right')
else:
    ax.text(0.5, 0.5, 'No importance data', ha='center', va='center')

# Panel 5: Model rho over time (CV vs Train — overfit gap)
ax = axes[1, 1]
if weekly_model_info:
    wk_nums = [info['week'] for info in weekly_model_info]
    cv_rho_vals = [info['rho'] for info in weekly_model_info]
    tr_rho_vals = [info.get('train_rho', info['rho']) for info in weekly_model_info]
    n_trains = [info['n_train'] for info in weekly_model_info]
    ax.plot(wk_nums, cv_rho_vals, 'o-', color='#2563eb', linewidth=2, markersize=5,
            label='CV rho (generalization)')
    ax.plot(wk_nums, tr_rho_vals, 's--', color='#dc2626', linewidth=1.5, markersize=4,
            label='Train rho (memorization)')
    ax.fill_between(wk_nums, cv_rho_vals, tr_rho_vals,
                    alpha=0.15, color='#dc2626', label='Overfit gap')
    ax.axhline(0, color='#555', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Week')
    ax.set_ylabel('Spearman rho')
    ax.set_title('Model Quality: CV vs Train rho')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.bar(wk_nums, n_trains, alpha=0.12, color='#16a34a', width=0.5, label='N_train')
    ax2.set_ylabel('Training samples', color='#16a34a')
    ax2.legend(fontsize=7, loc='upper left')
else:
    ax.text(0.5, 0.5, 'No model data', ha='center', va='center')

# Panel 6: Per-ticker avg PnL comparison
ax = axes[1, 2]
ml_by_tk = defaultdict(list)
oc_by_tk = defaultdict(list)
for t in strategies['ML_DYNAMIC']['trades']:
    ml_by_tk[t['ticker']].append(t)
for t in strategies['ORACLE']['trades']:
    oc_by_tk[t['ticker']].append(t)
all_tks = sorted(set(list(ml_by_tk.keys()) + list(oc_by_tk.keys())))
x = np.arange(len(all_tks))
ml_avgs = [np.mean([t['pnl'] for t in ml_by_tk[tk]]) * 100
           if ml_by_tk.get(tk) else 0 for tk in all_tks]
oc_avgs = [np.mean([t['pnl'] for t in oc_by_tk[tk]]) * 100
           if oc_by_tk.get(tk) else 0 for tk in all_tks]
bw = 0.35
ax.bar(x - bw/2, ml_avgs, bw, label='ML_DYNAMIC', color='#2563eb', alpha=0.85)
ax.bar(x + bw/2, oc_avgs, bw, label='ORACLE [B]', color='#dc2626', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(all_tks, fontsize=7, rotation=45)
ax.set_ylabel('Avg PnL per Trade (%)')
ax.set_title('Per-Ticker Avg PnL')
ax.legend(fontsize=8)
ax.axhline(0, color='gray', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
chart_path = '/tmp/backtest_short_compare.png'
fig.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Chart saved: {chart_path}")


# -- Final summary --
ml_f = strategy_summaries['ML_DYNAMIC']['FULL']
fi_f = strategy_summaries['FILTERED']['FULL']
oc_f = strategy_summaries['ORACLE']['FULL']
ml_h2 = strategy_summaries['ML_DYNAMIC']['HALF2']
oc_h2 = strategy_summaries['ORACLE']['HALF2']

n_changes = 0
for wi in range(1, len(strategies['ML_DYNAMIC']['weekly_tickers'])):
    prev = set(strategies['ML_DYNAMIC']['weekly_tickers'][wi-1]['tickers'])
    curr = set(strategies['ML_DYNAMIC']['weekly_tickers'][wi]['tickers'])
    n_changes += len(prev.symmetric_difference(curr))

print(f"\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")
print(f"  ML_DYNAMIC:   {ml_f['n']}t  {ml_f['total']:+.1%} total  "
      f"{ml_f['avg']:+.2%} avg  {ml_f['wr']:.0%} WR  DD {ml_f['dd']:+.1%}")
print(f"  FILTERED:     {fi_f['n']}t  {fi_f['total']:+.1%} total  "
      f"{fi_f['avg']:+.2%} avg  {fi_f['wr']:.0%} WR  DD {fi_f['dd']:+.1%}")
print(f"  ORACLE [B]:   {oc_f['n']}t  {oc_f['total']:+.1%} total  "
      f"{oc_f['avg']:+.2%} avg  {oc_f['wr']:.0%} WR  DD {oc_f['dd']:+.1%}")
print(f"  Gap (ML-OC):  {ml_f['avg']-oc_f['avg']:+.2%} avg PnL")
print(f"  Gap (ML-FI):  {ml_f['avg']-fi_f['avg']:+.2%} avg PnL")
print(f"  ML stability: {n_changes} ticker changes across {len(weeks)} weeks")
print(f"  H2 adapt:     ML {ml_h2['avg']:+.2%} vs ORACLE {oc_h2['avg']:+.2%}")
if weekly_model_info:
    final_cv = weekly_model_info[-1]['rho']
    final_tr = weekly_model_info[-1].get('train_rho', final_cv)
    final_n = weekly_model_info[-1]['n_train']
    print(f"  Final model:  {weekly_model_info[-1]['model']} "
          f"CV_rho={final_cv:+.3f} train_rho={final_tr:+.3f} "
          f"gap={final_tr-final_cv:.3f} ({final_n} training obs)")
print(f"{'='*W}")

_release_memory()
