"""
analyze_oracle_reverse.py — Reverse-Engineer ORACLE SHORT Ticker Selection (v2)
=================================================================================
v2: TEST ORACLE = top-8 by actual Jun-Jul TOTAL PnL (avg_win >= 1% gate).
    Optimizes for total PnL, not per-trade avg. Shows full-period backtest.
    Val_b ORACLE kept for comparison only.

12 analysis sections + 4x3 chart panel.
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
from scipy.stats import spearmanr, skew, kurtosis, mannwhitneyu

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

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1')
BQ_ALLOWED_MODES  = ('static', 'atr_sl', 'grace')

TOP_K = 8
FILTERED_VA_OPT_RATIO = 2.0
FILTERED_MIN_CONFIGS = 5
MIN_SEEDS = 3
OHLCV_LOOKBACK_DAYS = 45

CHART_PATH = '/tmp/analyze_oracle_reverse.png'

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110


# ==============================================================================
# REUSED HELPERS (from backtest_short_compare.py)
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

def compute_vb_oracle(bq_df, top_k):
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = bq_df[bq_df['is_ensemble'] != True].copy()
    vb = indiv.groupby('ticker').agg(vb_median=('val_b_avg_pnl', 'median'))
    return vb.nlargest(top_k, 'vb_median').index.tolist()

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
# OHLCV FEATURES
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
# NEW FUNCTIONS
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


def compute_weekly_aggregates(all_trades, weeks):
    weekly_agg = defaultdict(dict)
    for ticker, trades in all_trades.items():
        for wi, (wk_start, wk_end) in enumerate(weeks):
            wt = [t for t in trades if wk_start <= t['date'] <= wk_end]
            if wt:
                pnls = [t['pnl'] for t in wt]
                weekly_agg[ticker][wi] = {
                    'pnl_mean': np.mean(pnls),
                    'wr': np.mean(np.array(pnls) > 0),
                    'n': len(pnls),
                    'pnl_std': np.std(pnls) if len(pnls) > 1 else 0,
                    'pnl_list': pnls,
                }
    return weekly_agg


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


def compute_candidate_targets(all_trades, weekly_agg, weeks, daily_probs_cache,
                              configs, tradeable_tickers):
    half1_end = pd.Timestamp(HALF1_END).date()
    half2_start = pd.Timestamp(HALF2_START).date()

    weekly_intra_range = {}
    for wi, (wk_start, wk_end) in enumerate(weeks):
        ranges = []
        for ticker in tradeable_tickers:
            feat = compute_ohlcv_features(ticker, wk_start)
            ranges.append(feat['intraday_range_5d'])
        weekly_intra_range[wi] = np.percentile(ranges, 75) if ranges else 0

    rows = []
    for ticker in tradeable_tickers:
        trades = all_trades.get(ticker, [])
        if not trades:
            rows.append({'ticker': ticker, **{t: 0.0 for t in [
                'weekly_pnl_mean', 'weekly_wr', 'weekly_sharpe', 'binary_positive',
                'consistency_score', 'win_magnitude_ratio', 'regime_cond_wr',
                'signal_adj_wr', 'h2_pnl_mean', 'tail_ratio', 'h2_total_pnl',
                'full_total_pnl']}})
            continue

        pnls = np.array([t['pnl'] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        wagg = weekly_agg.get(ticker, {})
        wk_pnls = [wagg[wi]['pnl_mean'] for wi in sorted(wagg.keys())]
        wk_wrs = [wagg[wi]['wr'] for wi in sorted(wagg.keys())]
        wk_stds = [wagg[wi]['pnl_std'] for wi in sorted(wagg.keys())]

        weekly_pnl_mean = np.mean(wk_pnls) if wk_pnls else 0
        weekly_wr = np.mean(wk_wrs) if wk_wrs else 0
        weekly_sharpes = [p / max(s, 1e-6) for p, s in zip(wk_pnls, wk_stds)]
        weekly_sharpe = np.mean(weekly_sharpes) if weekly_sharpes else 0
        binary_positive = float(np.mean(pnls > 0))
        consistency_score = np.mean([1 if p > 0 else 0 for p in wk_pnls]) if wk_pnls else 0
        wmr = np.mean(wins) / max(abs(np.mean(losses)), 1e-6) if len(wins) > 0 and len(losses) > 0 else (2.0 if len(wins) > 0 else 0.0)

        high_vol_wrs = []
        for wi in sorted(wagg.keys()):
            feat = compute_ohlcv_features(ticker, weeks[wi][0])
            if feat['intraday_range_5d'] >= weekly_intra_range.get(wi, 0):
                high_vol_wrs.append(wagg[wi]['wr'])
        regime_cond_wr = np.mean(high_vol_wrs) if high_vol_wrs else 0

        probs = daily_probs_cache.get(ticker, {})
        thresh = configs[ticker]['prob_thresh'] if ticker in configs else 0.5
        trading_days_all = pd.bdate_range(TEST_START, TEST_END).date
        n_signal_days = sum(1 for d in trading_days_all
                           if probs.get(d, 0) >= thresh)
        signal_freq = n_signal_days / max(len(trading_days_all), 1)
        signal_adj_wr = binary_positive * signal_freq

        h2_trades = [t['pnl'] for t in trades if t['date'] >= half2_start]
        h2_pnl_mean = np.mean(h2_trades) if h2_trades else 0
        h2_total_pnl = sum(h2_trades)
        full_total_pnl = sum(pnls)

        p25 = np.percentile(pnls, 25)
        p75 = np.percentile(pnls, 75)
        tail_ratio = p75 / max(abs(p25), 1e-6)

        rows.append({
            'ticker': ticker,
            'weekly_pnl_mean': weekly_pnl_mean,
            'weekly_wr': weekly_wr,
            'weekly_sharpe': weekly_sharpe,
            'binary_positive': binary_positive,
            'consistency_score': consistency_score,
            'win_magnitude_ratio': wmr,
            'regime_cond_wr': regime_cond_wr,
            'signal_adj_wr': signal_adj_wr,
            'h2_pnl_mean': h2_pnl_mean,
            'tail_ratio': tail_ratio,
            'h2_total_pnl': h2_total_pnl,
            'full_total_pnl': full_total_pnl,
        })
    return pd.DataFrame(rows).set_index('ticker')


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  ORACLE REVERSE-ENGINEERING v2 — TEST ORACLE (Total PnL)")
print(f"  Top-{TOP_K} by Jun-Jul TOTAL PnL | avg_win >= {MIN_AVG_WIN:.0%}")
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
print(f"    {len(all_configs)} tickers have valid configs")

print(f"\n    Computing val_b oracle (for comparison)...")
vb_oracle_tickers = compute_vb_oracle(bq_df, TOP_K)
vb_oracle_set = set(vb_oracle_tickers)
vb_ranks = compute_vb_ranks(bq_df)
print(f"    Val_b ORACLE: {', '.join(vb_oracle_tickers)}")

print(f"\n    Computing static BQ features...")
static_feats = compute_static_bq_features(bq_df)
print(f"    {len(static_feats)} tickers x {len(static_feats.columns)} structural features")

tradeable_tickers = sorted(set(all_configs.keys()))

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

print(f"\n    Paper-trading ALL tickers for full period...")
trading_days = pd.bdate_range(TEST_START, TEST_END).date
all_trades = paper_trade_all_tickers(
    tradeable_tickers, trading_days, all_configs, ticker_data, daily_probs_cache)
n_traded = sum(1 for t in all_trades.values() if len(t) > 0)
n_total_trades = sum(len(t) for t in all_trades.values())
print(f"    {n_traded}/{len(tradeable_tickers)} tickers traded, {n_total_trades} total trades")

weeks = generate_weeks(TEST_START, TEST_END)
weekly_agg = compute_weekly_aggregates(all_trades, weeks)

half1_end_dt = pd.Timestamp(HALF1_END).date()
half2_start_dt = pd.Timestamp(HALF2_START).date()

print(f"\n    Computing TEST ORACLE (top-{TOP_K} by Jun-Jul total PnL, avg_win >= {MIN_AVG_WIN:.0%})...")
test_oracle_list, all_ticker_h2_stats = compute_test_oracle(all_trades, TOP_K, MIN_AVG_WIN)
oracle_tickers = [t['ticker'] for t in test_oracle_list]
oracle_set = set(oracle_tickers)
non_oracle_tickers = [t for t in tradeable_tickers if t not in oracle_set]

test_ranks = pd.Series({t['ticker']: t['h2_total_pnl'] for t in all_ticker_h2_stats})

print(f"    TEST ORACLE: {', '.join(oracle_tickers)}")
overlap = oracle_set & vb_oracle_set
print(f"    Overlap with val_b ORACLE: {len(overlap)} tickers ({', '.join(sorted(overlap)) if overlap else 'NONE'})")

_release_memory()


# ==============================================================================
# SECTION 0A: TEST ORACLE Definition
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 0A: TEST ORACLE — Ranked by Jun-Jul TOTAL PnL")
print(f"{'='*W}")

print(f"\n  ALL tickers ranked by H2 total PnL (avg_win >= {MIN_AVG_WIN:.0%} gate):")
print(f"  {'Rk':>3} {'Ticker':<7} | {'H2_Total':>9} {'H2_Trades':>9} {'H2_Avg':>7} "
      f"{'H2_WR':>5} {'Avg_Win':>8} | {'VB_OC':>5} {'TEST':>5}")
print(f"  {'-'*75}")
for i, ts in enumerate(all_ticker_h2_stats):
    t = ts['ticker']
    vb = ' Y ' if t in vb_oracle_set else ' - '
    te = ' Y ' if t in oracle_set else ' - '
    print(f"  {i+1:>3} {t:<7} | {ts['h2_total_pnl']:>+8.2%} {ts['h2_n']:>9} "
          f"{ts['h2_avg']:>+6.2%} {ts['h2_wr']:>4.0%} {ts['avg_win']:>+7.2%} | {vb:>5} {te:>5}")

print(f"\n  VAL_B ORACLE (original):  {', '.join(vb_oracle_tickers)}")
print(f"  TEST ORACLE (total PnL):  {', '.join(oracle_tickers)}")
only_vb = sorted(vb_oracle_set - oracle_set)
only_test = sorted(oracle_set - vb_oracle_set)
print(f"  Only in val_b: {', '.join(only_vb) if only_vb else 'none'}")
print(f"  Only in TEST:  {', '.join(only_test) if only_test else 'none'}")
print(f"  Overlap:       {', '.join(sorted(overlap)) if overlap else 'none'}")


# ==============================================================================
# SECTION 0B: TEST ORACLE Full-Period Backtest
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 0B: TEST ORACLE — Full-Period Backtest (Apr-Jul)")
print(f"{'='*W}")

print(f"\n  Per-ticker performance:")
print(f"  {'Ticker':<7} | {'Trades':>6} {'Total PnL':>10} {'Avg PnL':>8} {'WR':>5} | "
      f"{'H1_Avg':>7} {'H2_Avg':>7} {'H2_Total':>9} | {'Avg_Win':>8}")
print(f"  {'-'*85}")

portfolio_trades = []
for ticker in oracle_tickers:
    trades = all_trades.get(ticker, [])
    if not trades:
        print(f"  {ticker:<7} | {'--':>6} {'--':>10} {'--':>8} {'--':>5} | "
              f"{'--':>7} {'--':>7} {'--':>9} | {'--':>8}")
        continue
    pnls = np.array([t['pnl'] for t in trades])
    wins = pnls[pnls > 0]
    h1 = [t['pnl'] for t in trades if t['date'] <= half1_end_dt]
    h2 = [t['pnl'] for t in trades if t['date'] >= half2_start_dt]
    portfolio_trades.extend(trades)
    print(f"  {ticker:<7} | {len(trades):>6} {sum(pnls):>+9.2%} {np.mean(pnls):>+7.2%} "
          f"{np.mean(pnls>0):>4.0%} | "
          f"{np.mean(h1) if h1 else 0:>+6.2%} {np.mean(h2) if h2 else 0:>+6.2%} "
          f"{sum(h2):>+8.2%} | {np.mean(wins) if len(wins)>0 else 0:>+7.2%}")

port_pnls = np.array([t['pnl'] for t in portfolio_trades])
port_wins = port_pnls[port_pnls > 0]
port_h1 = [t['pnl'] for t in portfolio_trades if t['date'] <= half1_end_dt]
port_h2 = [t['pnl'] for t in portfolio_trades if t['date'] >= half2_start_dt]
print(f"  {'-'*85}")
print(f"  {'TOTAL':<7} | {len(port_pnls):>6} {sum(port_pnls):>+9.2%} "
      f"{np.mean(port_pnls):>+7.2%} {np.mean(port_pnls>0):>4.0%} | "
      f"{np.mean(port_h1) if port_h1 else 0:>+6.2%} "
      f"{np.mean(port_h2) if port_h2 else 0:>+6.2%} "
      f"{sum(port_h2):>+8.2%} | {np.mean(port_wins) if len(port_wins)>0 else 0:>+7.2%}")

vb_port_trades = []
for ticker in vb_oracle_tickers:
    vb_port_trades.extend(all_trades.get(ticker, []))
vb_pnls = np.array([t['pnl'] for t in vb_port_trades]) if vb_port_trades else np.array([0])
vb_h2 = [t['pnl'] for t in vb_port_trades if t['date'] >= half2_start_dt]

all_pnls = np.array([t['pnl'] for tk in tradeable_tickers for t in all_trades.get(tk, [])])
all_h2 = [t['pnl'] for tk in tradeable_tickers for t in all_trades.get(tk, []) if t['date'] >= half2_start_dt]

print(f"\n  COMPARISON:")
print(f"  {'Strategy':<18} | {'Trades':>6} {'Total PnL':>10} {'Avg PnL':>8} {'WR':>5} | "
      f"{'H2_Total':>9} {'H2_Avg':>8}")
print(f"  {'-'*70}")
print(f"  {'TEST ORACLE':<18} | {len(port_pnls):>6} {sum(port_pnls):>+9.2%} "
      f"{np.mean(port_pnls):>+7.2%} {np.mean(port_pnls>0):>4.0%} | "
      f"{sum(port_h2):>+8.2%} {np.mean(port_h2) if port_h2 else 0:>+7.2%}")
print(f"  {'Val_b ORACLE':<18} | {len(vb_pnls):>6} {sum(vb_pnls):>+9.2%} "
      f"{np.mean(vb_pnls):>+7.2%} {np.mean(vb_pnls>0):>4.0%} | "
      f"{sum(vb_h2):>+8.2%} {np.mean(vb_h2) if vb_h2 else 0:>+7.2%}")
print(f"  {'ALL UNIVERSE':<18} | {len(all_pnls):>6} {sum(all_pnls):>+9.2%} "
      f"{np.mean(all_pnls):>+7.2%} {np.mean(all_pnls>0):>4.0%} | "
      f"{sum(all_h2):>+8.2%} {np.mean(all_h2) if all_h2 else 0:>+7.2%}")

print(f"\n  Monthly breakdown (TEST ORACLE):")
for month_name, m_start, m_end in [('APR', '2026-04-01', '2026-04-30'),
                                    ('MAY', '2026-05-01', '2026-05-31'),
                                    ('JUN', '2026-06-01', '2026-06-30'),
                                    ('JUL', '2026-07-01', '2026-07-23')]:
    ms = pd.Timestamp(m_start).date()
    me = pd.Timestamp(m_end).date()
    mt = [t['pnl'] for t in portfolio_trades if ms <= t['date'] <= me]
    mw = [p for p in mt if p > 0]
    if mt:
        print(f"  {month_name}: {len(mt)} trades, total {sum(mt):+.2%}, avg {np.mean(mt):+.2%}, "
              f"WR {np.mean(np.array(mt)>0):.0%}, avg_win {np.mean(mw) if mw else 0:+.2%}")
    else:
        print(f"  {month_name}: 0 trades")

print(f"\n  Weekly breakdown (TEST ORACLE):")
print(f"  {'Week':<14} | {'N':>3} {'Total':>7} {'Avg':>7} {'WR':>5} | {'Cum_Total':>10}")
print(f"  {'-'*55}")
cum_total = 0
for wi, (wk_start, wk_end) in enumerate(weeks):
    wt = [t['pnl'] for t in portfolio_trades if wk_start <= t['date'] <= wk_end]
    if wt:
        cum_total += sum(wt)
        period = f"W{wi+1} {wk_start.strftime('%m/%d')}-{wk_end.strftime('%m/%d')}"
        marker = ' <--' if wk_start >= half2_start_dt else ''
        print(f"  {period:<14} | {len(wt):>3} {sum(wt):>+6.2%} {np.mean(wt):>+6.2%} "
              f"{np.mean(np.array(wt)>0):>4.0%} | {cum_total:>+9.2%}{marker}")


# ==============================================================================
# SECTION 1: PnL Distribution — Mean vs Tails vs Consistency
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 1: PnL Distribution — TEST ORACLE vs Non-ORACLE")
print(f"{'='*W}")

def _group_stats(tickers, label):
    pnls = []
    for t in tickers:
        pnls.extend([tr['pnl'] for tr in all_trades.get(t, [])])
    if not pnls:
        return {'label': label, 'n': 0}
    pnls = np.array(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    return {
        'label': label, 'n': len(pnls),
        'mean': np.mean(pnls), 'median': np.median(pnls),
        'std': np.std(pnls), 'skew': float(skew(pnls)),
        'kurt': float(kurtosis(pnls)),
        'p10': np.percentile(pnls, 10), 'p25': np.percentile(pnls, 25),
        'p75': np.percentile(pnls, 75), 'p90': np.percentile(pnls, 90),
        'wr': float(np.mean(pnls > 0)),
        'avg_win': float(np.mean(wins)) if len(wins) > 0 else 0,
        'avg_loss': float(np.mean(losses)) if len(losses) > 0 else 0,
        'wmr': float(np.mean(wins) / max(abs(np.mean(losses)), 1e-6)) if len(wins) > 0 and len(losses) > 0 else 0,
        'tail_ratio': float(np.percentile(pnls, 90) / max(abs(np.percentile(pnls, 10)), 1e-6)),
        'total_pnl': float(np.sum(pnls)),
    }

oc_stats = _group_stats(oracle_tickers, 'TEST_ORACLE')
noc_stats = _group_stats(non_oracle_tickers, 'Non-ORACLE')

print(f"\n  {'Metric':<25} | {'TEST_OC':>12} | {'Non-OC':>12} | {'Delta':>10}")
print(f"  {'-'*65}")
for metric in ['n', 'total_pnl', 'mean', 'median', 'std', 'skew', 'kurt', 'wr',
               'avg_win', 'avg_loss', 'wmr', 'p10', 'p25', 'p75', 'p90', 'tail_ratio']:
    ov = oc_stats.get(metric, 0)
    nv = noc_stats.get(metric, 0)
    delta = ov - nv if isinstance(ov, float) else 0
    if metric in ('n',):
        print(f"  {metric:<25} | {ov:>12} | {nv:>12} | {'':>10}")
    elif metric in ('total_pnl',):
        print(f"  {metric:<25} | {ov:>+11.2%} | {nv:>+11.2%} | {delta:>+9.2%}")
    elif metric in ('mean', 'median', 'std', 'avg_win', 'avg_loss', 'p10', 'p25', 'p75', 'p90'):
        print(f"  {metric:<25} | {ov:>+11.2%} | {nv:>+11.2%} | {delta:>+9.2%}")
    elif metric in ('wr',):
        print(f"  {metric:<25} | {ov:>11.0%} | {nv:>11.0%} | {delta:>+9.0%}")
    else:
        print(f"  {metric:<25} | {ov:>11.2f} | {nv:>11.2f} | {delta:>+9.2f}")


# ==============================================================================
# SECTION 2: Win Rate Stability Across Weeks
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 2: Win Rate Stability Across Weeks")
print(f"{'='*W}")

wr_stability = []
for ticker in tradeable_tickers:
    wagg = weekly_agg.get(ticker, {})
    wk_wrs = [wagg[wi]['wr'] for wi in sorted(wagg.keys()) if wagg[wi]['n'] > 0]
    if not wk_wrs:
        continue
    n_weeks = len(wk_wrs)
    mean_wr = np.mean(wk_wrs)
    std_wr = np.std(wk_wrs) if n_weeks > 1 else 0
    cv = std_wr / max(mean_wr, 1e-6)
    wks_above_50 = sum(1 for w in wk_wrs if w >= 0.5) / n_weeks
    max_lose = 0
    curr_lose = 0
    for w in wk_wrs:
        if w < 0.5:
            curr_lose += 1
            max_lose = max(max_lose, curr_lose)
        else:
            curr_lose = 0
    trades = all_trades.get(ticker, [])
    total_pnl = sum(t['pnl'] for t in trades)
    wr_stability.append({
        'ticker': ticker, 'n_weeks': n_weeks, 'mean_wr': mean_wr,
        'std_wr': std_wr, 'cv': cv, 'wks_above_50': wks_above_50,
        'max_lose_streak': max_lose, 'n_trades': len(trades),
        'total_pnl': total_pnl,
        'is_oracle': ticker in oracle_set,
    })

wr_df = pd.DataFrame(wr_stability).sort_values('total_pnl', ascending=False)
print(f"\n  {'Ticker':<7} | {'Trd':>3} {'Tot_PnL':>8} {'Mean_WR':>7} {'Wks>=50%':>8} | {'OC':>3}")
print(f"  {'-'*50}")
for _, r in wr_df.head(20).iterrows():
    oc = ' Y ' if r['is_oracle'] else ' - '
    print(f"  {r['ticker']:<7} | {r['n_trades']:>3} {r['total_pnl']:>+7.2%} "
          f"{r['mean_wr']:>6.0%} {r['wks_above_50']:>7.0%} | {oc}")

oc_wr = wr_df[wr_df['is_oracle']]
noc_wr = wr_df[~wr_df['is_oracle']]
print(f"\n  GROUP SUMMARY:")
print(f"  {'Group':<15} | {'Avg_Trd':>7} {'Avg_Total':>9} {'Mean_WR':>7} {'Wks>=50%':>8}")
print(f"  TEST OC ({len(oc_wr)})   | {oc_wr['n_trades'].mean():>7.1f} "
      f"{oc_wr['total_pnl'].mean():>+8.2%} {oc_wr['mean_wr'].mean():>6.0%} "
      f"{oc_wr['wks_above_50'].mean():>7.0%}")
print(f"  Non-OC ({len(noc_wr)})  | {noc_wr['n_trades'].mean():>7.1f} "
      f"{noc_wr['total_pnl'].mean():>+8.2%} {noc_wr['mean_wr'].mean():>6.0%} "
      f"{noc_wr['wks_above_50'].mean():>7.0%}")


# ==============================================================================
# SECTION 3: Regime Sensitivity — HALF1 vs HALF2
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 3: Regime Sensitivity — HALF1 vs HALF2")
print(f"{'='*W}")

regime_data = []
for ticker in tradeable_tickers:
    trades = all_trades.get(ticker, [])
    if not trades:
        continue
    h1 = [t['pnl'] for t in trades if t['date'] <= half1_end_dt]
    h2 = [t['pnl'] for t in trades if t['date'] >= half2_start_dt]
    h1_avg = np.mean(h1) if h1 else 0
    h2_avg = np.mean(h2) if h2 else 0
    h1_wr = np.mean(np.array(h1) > 0) if h1 else 0
    h2_wr = np.mean(np.array(h2) > 0) if h2 else 0
    regime_data.append({
        'ticker': ticker, 'h1_avg': h1_avg, 'h2_avg': h2_avg,
        'h1_wr': h1_wr, 'h2_wr': h2_wr,
        'h1_total': sum(h1), 'h2_total': sum(h2),
        'h1_n': len(h1), 'h2_n': len(h2),
        'dpnl': h2_avg - h1_avg, 'dwr': h2_wr - h1_wr,
        'improved': h2_avg > h1_avg,
        'is_oracle': ticker in oracle_set,
    })

regime_df = pd.DataFrame(regime_data).sort_values('h2_total', ascending=False)
print(f"\n  {'Ticker':<7} | {'H1_n':>4} {'H1_total':>9} | {'H2_n':>4} {'H2_total':>9} {'H2_avg':>7} | {'OC':>3}")
print(f"  {'-'*60}")
for _, r in regime_df.iterrows():
    oc = ' Y ' if r['is_oracle'] else ' - '
    print(f"  {r['ticker']:<7} | {r['h1_n']:>4} {r['h1_total']:>+8.2%} | "
          f"{r['h2_n']:>4} {r['h2_total']:>+8.2%} {r['h2_avg']:>+6.2%} | {oc}")

oc_reg = regime_df[regime_df['is_oracle']]
noc_reg = regime_df[~regime_df['is_oracle']]
print(f"\n  GROUP:")
print(f"  {'Group':<15} | {'H1_total':>9} | {'H2_total':>9} {'H2_avg':>7} | {'%Improved':>9}")
print(f"  TEST OC ({len(oc_reg)})   | {oc_reg['h1_total'].mean():>+8.2%} | "
      f"{oc_reg['h2_total'].mean():>+8.2%} {oc_reg['h2_avg'].mean():>+6.2%} | "
      f"{oc_reg['improved'].mean():>8.0%}")
print(f"  Non-OC ({len(noc_reg)})  | {noc_reg['h1_total'].mean():>+8.2%} | "
      f"{noc_reg['h2_total'].mean():>+8.2%} {noc_reg['h2_avg'].mean():>+6.2%} | "
      f"{noc_reg['improved'].mean():>8.0%}")


# ==============================================================================
# SECTION 4: Signal Quality — Frequency × Conditional WR
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 4: Signal Quality — Frequency × Conditional WR")
print(f"{'='*W}")

signal_data = []
for ticker in tradeable_tickers:
    trades = all_trades.get(ticker, [])
    probs = daily_probs_cache.get(ticker, {})
    thresh = all_configs[ticker]['prob_thresh'] if ticker in all_configs else 0.5
    n_signal = sum(1 for d in trading_days if probs.get(d, 0) >= thresh)
    freq = n_signal / max(len(trading_days), 1)
    cond_wr = np.mean(np.array([t['pnl'] for t in trades]) > 0) if trades else 0
    total_pnl = sum(t['pnl'] for t in trades)
    signal_data.append({
        'ticker': ticker, 'freq': freq, 'cond_wr': cond_wr,
        'quality': freq * cond_wr, 'n_trades': len(trades),
        'total_pnl': total_pnl,
        'is_oracle': ticker in oracle_set,
    })

sig_df = pd.DataFrame(signal_data).sort_values('total_pnl', ascending=False)
print(f"\n  {'Ticker':<7} | {'Freq':>5} {'WR':>5} {'Quality':>7} {'Trades':>6} {'Total':>8} | {'OC':>3}")
print(f"  {'-'*55}")
for _, r in sig_df.head(15).iterrows():
    oc = ' Y ' if r['is_oracle'] else ' - '
    print(f"  {r['ticker']:<7} | {r['freq']:>4.0%} {r['cond_wr']:>4.0%} "
          f"{r['quality']:>6.2f} {r['n_trades']:>6} {r['total_pnl']:>+7.2%} | {oc}")


# ==============================================================================
# SECTION 5: Config Robustness — BQ Structural Features
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 5: Config Robustness — BQ Structural Features")
print(f"{'='*W}")

bq_feats = ['n_configs', 'has_both_alphas', 'cross_alpha_agree', 'agree_18_best',
            'opt_wr_median', 'pct_static', 'pct_atr_sl', 'pct_grace',
            'pct_limit', 'va_median', 'va_pos_rate', 'va_opt_ratio_med']

print(f"\n  FEATURE -> TEST RANK CORRELATION (H2 total PnL):")
print(f"  {'Feature':<22} | {'vs test_rank':>12} {'p':>8} | {'vs val_b':>12} {'p':>8}")
print(f"  {'-'*70}")
common = sorted(set(static_feats.index) & set(test_ranks.index) & set(tradeable_tickers))
common_vb = sorted(set(static_feats.index) & set(vb_ranks.index) & set(tradeable_tickers))
for feat in bq_feats:
    if feat not in static_feats.columns:
        continue
    fvals = [static_feats.at[t, feat] for t in common]
    tvals = [test_ranks.at[t] for t in common]
    rho_t, p_t = spearmanr(fvals, tvals)
    fvals_vb = [static_feats.at[t, feat] for t in common_vb]
    vvals = [vb_ranks.at[t] for t in common_vb]
    rho_v, p_v = spearmanr(fvals_vb, vvals)
    m_t = ' ***' if p_t < 0.01 else ' **' if p_t < 0.05 else ' *' if p_t < 0.1 else ''
    m_v = ' ***' if p_v < 0.01 else ' **' if p_v < 0.05 else ' *' if p_v < 0.1 else ''
    print(f"  {feat:<22} | {rho_t:>+11.3f} {p_t:>7.4f}{m_t} | {rho_v:>+11.3f} {p_v:>7.4f}{m_v}")


# ==============================================================================
# SECTION 6: TARGET EXPLORATION — Systematic Search for Best ML Target
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 6: TARGET EXPLORATION — Correlated with H2 Total PnL")
print(f"{'='*W}")

print(f"\n    Computing 12 candidate targets for {len(tradeable_tickers)} tickers...")
targets_df = compute_candidate_targets(
    all_trades, weekly_agg, weeks, daily_probs_cache,
    all_configs, tradeable_tickers)

TARGET_NAMES = [
    'weekly_pnl_mean', 'weekly_wr', 'weekly_sharpe', 'binary_positive',
    'consistency_score', 'win_magnitude_ratio', 'regime_cond_wr',
    'signal_adj_wr', 'h2_pnl_mean', 'tail_ratio',
    'h2_total_pnl', 'full_total_pnl',
]

common_tgt = sorted(set(targets_df.index) & set(test_ranks.index))
common_tgt_vb = sorted(set(targets_df.index) & set(vb_ranks.index))

print(f"\n  {'Target':<25} | {'rho_test':>8} {'p':>7} | {'rho_vb':>8} {'p':>7} | "
      f"{'OC_mean':>10} {'Non-OC':>10} {'Cohen_d':>8}")
print(f"  {'-'*100}")

target_results = []
for tname in TARGET_NAMES:
    tvals = [targets_df.at[t, tname] for t in common_tgt]
    test_v = [test_ranks.at[t] for t in common_tgt]
    rho_t, p_t = spearmanr(tvals, test_v)

    tvals_vb = [targets_df.at[t, tname] for t in common_tgt_vb]
    vb_v = [vb_ranks.at[t] for t in common_tgt_vb]
    rho_v, p_v = spearmanr(tvals_vb, vb_v)

    oc_vals = [targets_df.at[t, tname] for t in common_tgt if t in oracle_set]
    noc_vals = [targets_df.at[t, tname] for t in common_tgt if t not in oracle_set]
    oc_mean = np.mean(oc_vals) if oc_vals else 0
    noc_mean = np.mean(noc_vals) if noc_vals else 0
    pooled_std = np.sqrt((np.var(oc_vals) + np.var(noc_vals)) / 2) if oc_vals and noc_vals else 1
    cohens_d = (oc_mean - noc_mean) / max(pooled_std, 1e-6)

    target_results.append({
        'name': tname, 'rho': rho_t, 'p': p_t,
        'rho_vb': rho_v, 'p_vb': p_v,
        'oc_mean': oc_mean, 'noc_mean': noc_mean, 'cohens_d': cohens_d,
    })
    m_t = '***' if p_t < 0.01 else '**' if p_t < 0.05 else '*' if p_t < 0.1 else ''
    oc_fmt = f"{oc_mean:>+9.2%}" if abs(oc_mean) < 100 else f"{oc_mean:>+9.1f}"
    noc_fmt = f"{noc_mean:>+9.2%}" if abs(noc_mean) < 100 else f"{noc_mean:>+9.1f}"
    print(f"  {tname:<25} | {rho_t:>+7.3f} {p_t:>6.4f} | {rho_v:>+7.3f} {p_v:>6.4f} | "
          f"{oc_fmt} {noc_fmt} {cohens_d:>+7.2f} {m_t}")

best_target = max(target_results, key=lambda x: x['rho'])
print(f"\n  BEST TARGET vs test_ranks: {best_target['name']} "
      f"(rho={best_target['rho']:+.3f}, p={best_target['p']:.4f}, "
      f"Cohen's d={best_target['cohens_d']:+.2f})")

current_ml = next(t for t in target_results if t['name'] == 'weekly_pnl_mean')
print(f"  CURRENT ML (weekly_pnl_mean): rho_test={current_ml['rho']:+.3f}, rho_vb={current_ml['rho_vb']:+.3f}")


# ==============================================================================
# SECTION 7: June/July Deep Dive
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 7: June/July — TEST ORACLE vs Non-ORACLE")
print(f"{'='*W}")

print(f"\n  Weekly breakdown:")
print(f"  {'Week':<14} | {'--- TEST ORACLE ---':>28} | {'--- Non-ORACLE ---':>28}")
print(f"  {'':14} | {'N':>3} {'PnL':>7} {'WR':>5} {'Total':>7} | "
      f"{'N':>3} {'PnL':>7} {'WR':>5} {'Total':>7}")
print(f"  {'-'*80}")

for wi, (wk_start, wk_end) in enumerate(weeks):
    oc_pnls, noc_pnls = [], []
    for ticker in tradeable_tickers:
        wagg_tk = weekly_agg.get(ticker, {})
        if wi not in wagg_tk:
            continue
        pnls = wagg_tk[wi]['pnl_list']
        if ticker in oracle_set:
            oc_pnls.extend(pnls)
        else:
            noc_pnls.extend(pnls)

    oc_avg = np.mean(oc_pnls) if oc_pnls else 0
    oc_wr_w = np.mean(np.array(oc_pnls) > 0) if oc_pnls else 0
    noc_avg = np.mean(noc_pnls) if noc_pnls else 0
    noc_wr_w = np.mean(np.array(noc_pnls) > 0) if noc_pnls else 0

    period = f"W{wi+1} {wk_start.strftime('%m/%d')}-{wk_end.strftime('%m/%d')}"
    marker = ' <--' if wk_start >= half2_start_dt else ''
    print(f"  {period:<14} | {len(oc_pnls):>3} {oc_avg:>+6.2%} {oc_wr_w:>4.0%} "
          f"{sum(oc_pnls):>+6.2%} | "
          f"{len(noc_pnls):>3} {noc_avg:>+6.2%} {noc_wr_w:>4.0%} "
          f"{sum(noc_pnls):>+6.2%}{marker}")

print(f"\n  TEST ORACLE ticker detail in June/July:")
for month_name, m_start, m_end in [('JUNE', '2026-06-01', '2026-06-30'),
                                     ('JULY', '2026-07-01', '2026-07-23')]:
    ms = pd.Timestamp(m_start).date()
    me = pd.Timestamp(m_end).date()
    print(f"\n  [{month_name}] TEST ORACLE tickers:")
    for ticker in oracle_tickers:
        trades = [t for t in all_trades.get(ticker, []) if ms <= t['date'] <= me]
        if trades:
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            print(f"    {ticker:<7}: {len(trades)} trades, total {sum(pnls):+.2%}, "
                  f"avg {np.mean(pnls):+.2%}, WR {np.mean(np.array(pnls)>0):.0%}, "
                  f"avg_win {np.mean(wins) if wins else 0:+.2%}")
        else:
            print(f"    {ticker:<7}: 0 trades")


# ==============================================================================
# SECTION 8: Feature Comparison at Regime Boundary
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 8: Feature Comparison at Regime Boundary (June 1)")
print(f"{'='*W}")

boundary_date = pd.Timestamp('2026-06-01').date()
all_feature_names = FEATURE_NAMES_OHLCV + FEATURE_NAMES_SIGNAL

feature_at_boundary = {}
for ticker in tradeable_tickers:
    ohlcv_feat = compute_ohlcv_features(ticker, boundary_date)
    probs = daily_probs_cache.get(ticker, {})
    thresh = all_configs[ticker]['prob_thresh'] if ticker in all_configs else 0.5
    sig_feat = compute_signal_features(probs, thresh, boundary_date)
    feature_at_boundary[ticker] = {**ohlcv_feat, **sig_feat}

feat_comparison = []
for fname in all_feature_names:
    oc_vals = [feature_at_boundary[t][fname] for t in tradeable_tickers if t in oracle_set and t in feature_at_boundary]
    noc_vals = [feature_at_boundary[t][fname] for t in tradeable_tickers if t not in oracle_set and t in feature_at_boundary]
    if not oc_vals or not noc_vals:
        continue
    oc_m = np.mean(oc_vals)
    noc_m = np.mean(noc_vals)
    pooled = np.sqrt((np.var(oc_vals) + np.var(noc_vals)) / 2)
    d = (oc_m - noc_m) / max(pooled, 1e-9)
    feat_comparison.append({'feature': fname, 'oc_mean': oc_m, 'noc_mean': noc_m, 'cohens_d': d})

feat_comparison.sort(key=lambda x: abs(x['cohens_d']), reverse=True)
print(f"\n  {'Feature':<22} | {'OC_mean':>10} {'Non-OC':>10} {'Cohen_d':>8} {'Direction':>10}")
print(f"  {'-'*70}")
for r in feat_comparison:
    direction = 'OC higher' if r['cohens_d'] > 0 else 'OC lower'
    print(f"  {r['feature']:<22} | {r['oc_mean']:>+9.4f} {r['noc_mean']:>+9.4f} "
          f"{r['cohens_d']:>+7.2f} {direction:>10}")


# ==============================================================================
# SECTION 9: Feature × Target Correlation Matrix
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 9: Feature × Target Correlation Matrix")
print(f"{'='*W}")

bq_feat_names = [f for f in bq_feats if f in static_feats.columns]
all_analysis_feats = all_feature_names + bq_feat_names

feat_target_corr = {}
for fname in all_analysis_feats:
    feat_target_corr[fname] = {}
    for tname in TARGET_NAMES:
        fvals = []
        tvals = []
        for ticker in common_tgt:
            if fname in all_feature_names:
                fvals.append(feature_at_boundary.get(ticker, {}).get(fname, 0))
            else:
                fvals.append(static_feats.at[ticker, fname] if ticker in static_feats.index else 0)
            tvals.append(targets_df.at[ticker, tname])
        rho, _ = spearmanr(fvals, tvals)
        feat_target_corr[fname][tname] = rho if not np.isnan(rho) else 0

best_tgt_name = best_target['name']
feat_by_best = sorted(all_analysis_feats,
                       key=lambda f: abs(feat_target_corr.get(f, {}).get(best_tgt_name, 0)),
                       reverse=True)

short_targets = [best_tgt_name, 'h2_total_pnl', 'weekly_pnl_mean', 'consistency_score', 'full_total_pnl']
short_targets = [t for t in short_targets if t in TARGET_NAMES][:5]

print(f"\n  Sorted by correlation with BEST TARGET ({best_tgt_name}):")
print(f"  {'Feature':<22} | " + ' | '.join(f"{t[:10]:>10}" for t in short_targets))
print(f"  {'-'*(25 + 13*len(short_targets))}")
for fname in feat_by_best[:15]:
    vals = ' | '.join(f"{feat_target_corr[fname].get(t, 0):>+9.3f}" for t in short_targets)
    print(f"  {fname:<22} | {vals}")


# ==============================================================================
# SECTION 10: ORACLE Membership Prediction — Simple Rules
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 10: TEST ORACLE Membership — Simple Rules")
print(f"{'='*W}")

rule_feats = []
for fname in all_analysis_feats:
    fvals = []
    labels = []
    for ticker in common_tgt:
        if fname in all_feature_names:
            fvals.append(feature_at_boundary.get(ticker, {}).get(fname, 0))
        else:
            fvals.append(static_feats.at[ticker, fname] if ticker in static_feats.index else 0)
        labels.append(1 if ticker in oracle_set else 0)

    fvals = np.array(fvals)
    labels = np.array(labels)
    if np.std(fvals) < 1e-9:
        continue

    best_j = 0
    best_thresh = fvals[0]
    best_prec = 0
    best_recall = 0
    sorted_fvals = np.sort(np.unique(fvals))
    for thresh in sorted_fvals:
        pred_pos = labels[fvals >= thresh]
        pred_neg = labels[fvals < thresh]
        tp = pred_pos.sum()
        fp = len(pred_pos) - tp
        fn = pred_neg.sum()
        tn = len(pred_neg) - fn
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thresh = thresh
            best_prec = tp / max(tp + fp, 1)
            best_recall = sens

    concordant = 0
    discordant = 0
    for i in range(len(fvals)):
        for j in range(len(fvals)):
            if labels[i] > labels[j]:
                if fvals[i] > fvals[j]: concordant += 1
                elif fvals[i] < fvals[j]: discordant += 1
    n_pairs = max(concordant + discordant, 1)
    auc_proxy = concordant / n_pairs

    rule_feats.append({
        'feature': fname, 'auc': auc_proxy, 'thresh': best_thresh,
        'precision': best_prec, 'recall': best_recall, 'youden_j': best_j,
    })

rule_feats.sort(key=lambda x: x['auc'], reverse=True)
print(f"\n  {'Feature':<22} | {'AUC':>5} {'Thresh':>8} {'Prec':>5} {'Recall':>6} {'J':>5}")
print(f"  {'-'*60}")
for r in rule_feats[:12]:
    print(f"  {r['feature']:<22} | {r['auc']:>5.3f} {r['thresh']:>8.4f} "
          f"{r['precision']:>5.2f} {r['recall']:>5.2f} {r['youden_j']:>5.2f}")

captured = 0
fps = 0
if len(rule_feats) >= 3:
    top3 = rule_feats[:3]
    for ticker in common_tgt:
        passes = True
        for rule in top3:
            fname = rule['feature']
            if fname in all_feature_names:
                val = feature_at_boundary.get(ticker, {}).get(fname, 0)
            else:
                val = static_feats.at[ticker, fname] if ticker in static_feats.index else 0
            if val < rule['thresh']:
                passes = False
                break
        if passes:
            if ticker in oracle_set:
                captured += 1
            else:
                fps += 1
    print(f"\n  3-FEATURE AND RULE:")
    for r in top3:
        print(f"    {r['feature']} >= {r['thresh']:.4f}")
    print(f"  Captures: {captured}/{len(oracle_set)} TEST ORACLE tickers, "
          f"{fps} false positives")
    total_pos = captured + fps
    prec = captured / max(total_pos, 1)
    print(f"  Precision: {prec:.0%}  Recall: {captured/len(oracle_set):.0%}")


# ==============================================================================
# SECTION 11: Temporal Stability
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 11: Temporal Stability — Does TEST ORACLE Hold?")
print(f"{'='*W}")

WINDOW = 4
print(f"\n  {'Window':<18} | {'Top-8 in window':<45} | {'Jaccard':>7} {'Rank_rho':>9}")
print(f"  {'-'*85}")

for start_wi in range(0, len(weeks) - WINDOW + 1):
    end_wi = start_wi + WINDOW - 1
    ticker_window_pnl = {}
    for ticker in tradeable_tickers:
        wagg_tk = weekly_agg.get(ticker, {})
        wpnls = []
        for wi in range(start_wi, end_wi + 1):
            if wi in wagg_tk:
                wpnls.extend(wagg_tk[wi]['pnl_list'])
        if wpnls:
            ticker_window_pnl[ticker] = sum(wpnls)

    if not ticker_window_pnl:
        continue
    top8_window = sorted(ticker_window_pnl, key=ticker_window_pnl.get, reverse=True)[:TOP_K]
    top8_set = set(top8_window)
    jaccard = len(top8_set & oracle_set) / max(len(top8_set | oracle_set), 1)

    common_rk = sorted(set(ticker_window_pnl.keys()) & set(test_ranks.index))
    if len(common_rk) > 5:
        rk_rho, _ = spearmanr(
            [ticker_window_pnl[t] for t in common_rk],
            [test_ranks[t] for t in common_rk])
    else:
        rk_rho = 0

    ws = weeks[start_wi][0].strftime('%m/%d')
    we = weeks[end_wi][1].strftime('%m/%d')
    window_label = f"W{start_wi+1}-W{end_wi+1} ({ws}-{we})"
    print(f"  {window_label:<18} | {','.join(top8_window):<45} | "
          f"{jaccard:>6.2f} {rk_rho:>+8.3f}")


# ==============================================================================
# SECTION 12: RECOMMENDATION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 12: RECOMMENDATION")
print(f"{'='*W}")

print(f"\n  1. TEST ORACLE (top-{TOP_K} by Jun-Jul total PnL):")
for ts in test_oracle_list:
    print(f"     {ts['ticker']:<7}: {ts['h2_n']} trades, total {ts['h2_total_pnl']:+.2%}, "
          f"avg {ts['h2_avg']:+.2%}, WR {ts['h2_wr']:.0%}, avg_win {ts['avg_win']:+.2%}")

print(f"\n  2. FULL-PERIOD PERFORMANCE:")
print(f"     TEST ORACLE:  {len(port_pnls)} trades, total {sum(port_pnls):+.2%}, "
      f"avg {np.mean(port_pnls):+.2%}, WR {np.mean(port_pnls>0):.0%}")
print(f"     Val_b ORACLE: {len(vb_pnls)} trades, total {sum(vb_pnls):+.2%}, "
      f"avg {np.mean(vb_pnls):+.2%}, WR {np.mean(vb_pnls>0):.0%}")
print(f"     Universe:     {len(all_pnls)} trades, total {sum(all_pnls):+.2%}, "
      f"avg {np.mean(all_pnls):+.2%}, WR {np.mean(all_pnls>0):.0%}")

print(f"\n  3. BEST TARGET: {best_target['name']}")
print(f"     rho vs test_ranks: {best_target['rho']:+.3f} (p={best_target['p']:.4f})")
print(f"     rho vs val_b:      {best_target['rho_vb']:+.3f}")
print(f"     Cohen's d:         {best_target['cohens_d']:+.2f}")

print(f"\n  4. TOP FEATURES correlated with best target ({best_tgt_name}):")
for i, fname in enumerate(feat_by_best[:5]):
    rho_val = feat_target_corr.get(fname, {}).get(best_tgt_name, 0)
    print(f"     {i+1}. {fname} (rho={rho_val:+.3f})")

if len(rule_feats) >= 3:
    print(f"\n  5. FALLBACK RULE (non-ML):")
    for r in rule_feats[:3]:
        print(f"     {r['feature']} >= {r['thresh']:.4f} (AUC={r['auc']:.3f})")
    print(f"     Captures: {captured}/{len(oracle_set)} TEST ORACLE, {fps} FP, "
          f"Precision: {prec:.0%}")

print(f"\n{'='*W}")


# ==============================================================================
# CHARTS (4x3)
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(4, 3, figsize=(24, 28))
fig.suptitle(f'TEST ORACLE Reverse-Engineering — SHORT Ticker Selection\n'
             f'{TEST_START} to {TEST_END} | TEST ORACLE: {",".join(oracle_tickers)}\n'
             f'(Top-{TOP_K} by Jun-Jul total PnL, avg_win >= {MIN_AVG_WIN:.0%})',
             fontsize=13, fontweight='bold')

OC_COLOR = '#dc2626'
NOC_COLOR = '#2563eb'
VB_COLOR = '#9ca3af'

# Panel (0,0): PnL Distribution
ax = axes[0, 0]
oc_pnls_ch = [t['pnl'] for tk in oracle_tickers for t in all_trades.get(tk, [])]
noc_pnls_ch = [t['pnl'] for tk in non_oracle_tickers for t in all_trades.get(tk, [])]
if oc_pnls_ch and noc_pnls_ch:
    bins = np.linspace(-0.15, 0.15, 40)
    ax.hist(oc_pnls_ch, bins=bins, alpha=0.6, color=OC_COLOR, label=f'TEST OC ({len(oc_pnls_ch)}t)', density=True)
    ax.hist(noc_pnls_ch, bins=bins, alpha=0.4, color=NOC_COLOR, label=f'Non-OC ({len(noc_pnls_ch)}t)', density=True)
    ax.axvline(np.mean(oc_pnls_ch), color=OC_COLOR, linestyle='--', linewidth=2)
    ax.axvline(np.mean(noc_pnls_ch), color=NOC_COLOR, linestyle='--', linewidth=2)
ax.set_title('PnL Distribution')
ax.set_xlabel('Trade PnL')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel (0,1): Total PnL per ticker
ax = axes[0, 1]
tpnl_data = []
for ticker in tradeable_tickers:
    trades = all_trades.get(ticker, [])
    if not trades: continue
    tpnl_data.append({'ticker': ticker, 'total': sum(t['pnl'] for t in trades),
                      'is_oc': ticker in oracle_set})
tpnl_data.sort(key=lambda x: x['total'], reverse=True)
if tpnl_data:
    y_pos = np.arange(len(tpnl_data))
    colors_tp = [OC_COLOR if d['is_oc'] else NOC_COLOR for d in tpnl_data]
    ax.barh(y_pos, [d['total']*100 for d in tpnl_data], color=colors_tp, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([d['ticker'] for d in tpnl_data], fontsize=6)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.invert_yaxis()
ax.set_title('Total PnL per Ticker (red=TEST ORACLE)')
ax.set_xlabel('Total PnL (%)')
ax.grid(True, alpha=0.3)

# Panel (0,2): Trade count vs total PnL scatter
ax = axes[0, 2]
for ticker in tradeable_tickers:
    trades = all_trades.get(ticker, [])
    if not trades: continue
    total = sum(t['pnl'] for t in trades)
    n = len(trades)
    color = OC_COLOR if ticker in oracle_set else NOC_COLOR
    alpha = 0.9 if ticker in oracle_set else 0.4
    size = 80 if ticker in oracle_set else 30
    ax.scatter(n, total*100, c=color, s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
    if ticker in oracle_set:
        ax.annotate(ticker, (n, total*100), fontsize=7, ha='left')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('Number of Trades')
ax.set_ylabel('Total PnL (%)')
ax.set_title('Trades vs Total PnL')
ax.grid(True, alpha=0.3)

# Panel (1,0): H1 vs H2 Total PnL Scatter
ax = axes[1, 0]
for _, r in regime_df.iterrows():
    color = OC_COLOR if r['is_oracle'] else NOC_COLOR
    alpha = 0.9 if r['is_oracle'] else 0.4
    size = 80 if r['is_oracle'] else 30
    ax.scatter(r['h1_total']*100, r['h2_total']*100, c=color, s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
    if r['is_oracle']:
        ax.annotate(r['ticker'], (r['h1_total']*100, r['h2_total']*100), fontsize=7, ha='left')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('H1 Total PnL % (Apr-May)')
ax.set_ylabel('H2 Total PnL % (Jun-Jul)')
ax.set_title('Regime Shift: H1 vs H2 Total PnL')
ax.grid(True, alpha=0.3)

# Panel (1,1): Weekly PnL: TEST ORACLE vs val_b ORACLE vs non-OC
ax = axes[1, 1]
oc_weekly_pnl = []
vb_weekly_pnl = []
noc_weekly_pnl = []
week_dates = []
for wi, (wk_start, wk_end) in enumerate(weeks):
    week_dates.append(wk_start)
    oc_p, vb_p, noc_p = [], [], []
    for ticker in tradeable_tickers:
        wagg_tk = weekly_agg.get(ticker, {})
        if wi in wagg_tk:
            pnl_sum = sum(wagg_tk[wi]['pnl_list'])
            if ticker in oracle_set:
                oc_p.append(pnl_sum)
            elif ticker in vb_oracle_set:
                vb_p.append(pnl_sum)
            else:
                noc_p.append(pnl_sum)
            if ticker in vb_oracle_set and ticker not in oracle_set:
                pass
    oc_weekly_pnl.append(sum(oc_p))
    vb_weekly_pnl.append(sum(vb_p) + sum([sum(wagg_tk[wi]['pnl_list'])
        for tk in vb_oracle_tickers
        if tk in weekly_agg and wi in weekly_agg.get(tk, {})
        for wagg_tk in [weekly_agg[tk]]]))
    noc_weekly_pnl.append(sum(noc_p))

vb_weekly_pnl_clean = []
for wi, (wk_start, wk_end) in enumerate(weeks):
    s = 0
    for tk in vb_oracle_tickers:
        wagg_tk = weekly_agg.get(tk, {})
        if wi in wagg_tk:
            s += sum(wagg_tk[wi]['pnl_list'])
    vb_weekly_pnl_clean.append(s)

ax.plot(week_dates, np.array([sum(
    [sum(weekly_agg.get(tk, {}).get(wi, {}).get('pnl_list', []))
     for tk in oracle_tickers]
) for wi in range(len(weeks))])*100,
    'o-', color=OC_COLOR, linewidth=2, markersize=5, label='TEST ORACLE')
ax.plot(week_dates, np.array(vb_weekly_pnl_clean)*100,
    's--', color=VB_COLOR, linewidth=1.5, markersize=4, label='Val_b ORACLE')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(half2_start_dt, color='gray', linewidth=1, linestyle=':', alpha=0.7)
ax.set_ylabel('Weekly Total PnL (%)')
ax.set_title('Weekly Total PnL: TEST vs Val_b ORACLE')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.grid(True, alpha=0.3)

# Panel (1,2): Cumulative PnL
ax = axes[1, 2]
sorted_port = sorted(portfolio_trades, key=lambda t: t['date'])
cum_pnl = np.cumsum([t['pnl'] for t in sorted_port]) * 100
trade_dates = [t['date'] for t in sorted_port]
if cum_pnl.size > 0:
    ax.plot(trade_dates, cum_pnl, '-', color=OC_COLOR, linewidth=2, label='TEST ORACLE')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(half2_start_dt, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax.fill_between(trade_dates, cum_pnl, 0, alpha=0.1, color=OC_COLOR)
ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('TEST ORACLE Equity Curve')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.grid(True, alpha=0.3)

# Panel (2,0): Target rho bar chart (vs test_ranks)
ax = axes[2, 0]
target_results_sorted = sorted(target_results, key=lambda x: x['rho'], reverse=True)
y_pos = np.arange(len(target_results_sorted))
rho_vals = [t['rho'] for t in target_results_sorted]
bar_colors = ['#16a34a' if r > 0 else '#dc2626' for r in rho_vals]
ax.barh(y_pos, rho_vals, color=bar_colors, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([t['name'] for t in target_results_sorted], fontsize=8)
current_idx = next((i for i, t in enumerate(target_results_sorted) if t['name'] == 'weekly_pnl_mean'), 0)
ax.barh(current_idx, target_results_sorted[current_idx]['rho'], color='#f59e0b',
        alpha=0.9, edgecolor='black', linewidth=2)
ax.invert_yaxis()
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Spearman rho vs H2 total PnL')
ax.set_title('Target Candidates (yellow = current ML)')
ax.grid(True, alpha=0.3)

# Panel (2,1): Best target vs test_ranks scatter
ax = axes[2, 1]
for ticker in common_tgt:
    color = OC_COLOR if ticker in oracle_set else NOC_COLOR
    alpha = 0.9 if ticker in oracle_set else 0.4
    size = 80 if ticker in oracle_set else 30
    ax.scatter(targets_df.at[ticker, best_tgt_name], test_ranks[ticker]*100,
               c=color, s=size, alpha=alpha, edgecolors='black', linewidth=0.5)
    if ticker in oracle_set:
        ax.annotate(ticker, (targets_df.at[ticker, best_tgt_name], test_ranks[ticker]*100),
                    fontsize=7, ha='left')
ax.set_xlabel(f'{best_tgt_name}')
ax.set_ylabel('H2 Total PnL (%)')
ax.set_title(f'Best Target vs H2 Total (rho={best_target["rho"]:+.3f})')
ax.grid(True, alpha=0.3)

# Panel (2,2): Feature importance for best target
ax = axes[2, 2]
top_feats_best = feat_by_best[:12]
y_pos = np.arange(len(top_feats_best))
rho_vals_f = [feat_target_corr.get(f, {}).get(best_tgt_name, 0) for f in top_feats_best]
feat_colors = []
for f in top_feats_best:
    if f in FEATURE_NAMES_OHLCV: feat_colors.append('#2563eb')
    elif f in FEATURE_NAMES_SIGNAL: feat_colors.append('#7c3aed')
    elif f in bq_feat_names: feat_colors.append('#d97706')
    else: feat_colors.append('#6b7280')
ax.barh(y_pos, rho_vals_f, color=feat_colors, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_feats_best, fontsize=8)
ax.invert_yaxis()
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel(f'Spearman rho with {best_tgt_name}')
ax.set_title(f'Features for Best Target')
from matplotlib.patches import Patch
legend_feats = [
    Patch(facecolor='#2563eb', label='OHLCV'),
    Patch(facecolor='#7c3aed', label='Signal'),
    Patch(facecolor='#d97706', label='BQ'),
]
ax.legend(handles=legend_feats, fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel (3,0): BQ feature comparison
ax = axes[3, 0]
bq_compare_feats = ['n_configs', 'cross_alpha_agree', 'agree_18_best', 'opt_wr_median',
                     'va_pos_rate', 'va_opt_ratio_med']
x_pos = np.arange(len(bq_compare_feats))
oc_bq = [static_feats.loc[[t for t in oracle_tickers if t in static_feats.index], f].mean()
          for f in bq_compare_feats]
noc_bq = [static_feats.loc[[t for t in non_oracle_tickers if t in static_feats.index], f].mean()
           for f in bq_compare_feats]
bw = 0.35
ax.bar(x_pos - bw/2, oc_bq, bw, label='TEST ORACLE', color=OC_COLOR, alpha=0.8)
ax.bar(x_pos + bw/2, noc_bq, bw, label='Non-ORACLE', color=NOC_COLOR, alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(bq_compare_feats, fontsize=7, rotation=30)
ax.set_title('BQ Structural Features')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel (3,1): ORACLE stability over time
ax = axes[3, 1]
jaccard_vals = []
jaccard_dates = []
for start_wi in range(0, len(weeks) - WINDOW + 1):
    end_wi = start_wi + WINDOW - 1
    ticker_window_pnl = {}
    for ticker in tradeable_tickers:
        wagg_tk = weekly_agg.get(ticker, {})
        wpnls = []
        for wi in range(start_wi, end_wi + 1):
            if wi in wagg_tk:
                wpnls.extend(wagg_tk[wi]['pnl_list'])
        if wpnls:
            ticker_window_pnl[ticker] = sum(wpnls)
    if ticker_window_pnl:
        top8 = set(sorted(ticker_window_pnl, key=ticker_window_pnl.get, reverse=True)[:TOP_K])
        jaccard_vals.append(len(top8 & oracle_set) / max(len(top8 | oracle_set), 1))
        jaccard_dates.append(weeks[start_wi][0])

if jaccard_vals:
    ax.plot(jaccard_dates, jaccard_vals, 'o-', color=OC_COLOR, linewidth=2, markersize=6)
    ax.axhline(np.mean(jaccard_vals), color='gray', linestyle='--', linewidth=1,
               label=f'Mean: {np.mean(jaccard_vals):.2f}')
    ax.axvline(half2_start_dt, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax.set_ylabel('Jaccard Overlap with TEST ORACLE')
    ax.set_title('TEST ORACLE Stability (rolling 4-week)')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# Panel (3,2): Feature x Target Heatmap
ax = axes[3, 2]
heatmap_feats = feat_by_best[:10]
heatmap_targets = short_targets
hm_data = np.zeros((len(heatmap_feats), len(heatmap_targets)))
for i, fname in enumerate(heatmap_feats):
    for j, tname in enumerate(heatmap_targets):
        hm_data[i, j] = feat_target_corr.get(fname, {}).get(tname, 0)
im = ax.imshow(hm_data, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax.set_xticks(range(len(heatmap_targets)))
ax.set_xticklabels([t[:12] for t in heatmap_targets], fontsize=7, rotation=30)
ax.set_yticks(range(len(heatmap_feats)))
ax.set_yticklabels(heatmap_feats, fontsize=7)
ax.set_title('Feature × Target Correlation')
fig.colorbar(im, ax=ax, shrink=0.8, label='Spearman rho')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(CHART_PATH, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Chart saved: {CHART_PATH}")

_release_memory()
