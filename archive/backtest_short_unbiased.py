"""
backtest_short_unbiased.py — Walk-Forward UNBIASED Backtest (SHORT)
===================================================================
Weekly rebalancing: Ridge + ALL (49 features), backward-looking target ONLY.
Zero future information at any point — model learns from accumulating trades.

At each Monday rebalance:
  1. Compute backward stats: per-ticker total PnL using only past trades
  2. Define "backward oracle" using same rule as final oracle (total PnL + avg_win >= 1%)
  3. Train Ridge on backward mean PnL (blended with BQ prior for sparse tickers)
  4. Score all tickers, select top-8

Key output: PROGRESSION TABLE showing model selection vs final oracle overlap
over time — does the model converge to picking the right tickers?

4 strategies: MODEL (walk-forward) / ALL / ORACLE (ceiling) / CURRENT
6 analysis sections + 2×3 charts. CSV scoring only. Colab-ready.
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
BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1')
BQ_ALLOWED_MODES  = ('static', 'atr_sl', 'grace')

TOP_K = 8
FILTERED_VA_OPT_RATIO = 2.0
MIN_SEEDS = 3
OHLCV_LOOKBACK_DAYS = 45

BURN_IN_WEEKS = 3
DYNAMIC_TARGET_BLEND = 0.5
MIN_TRADES_FOR_BLEND = 2

CURRENT_TICKERS = ['VOYG', 'NTLA', 'CLF', 'SMR']

CHART_PATH = '/tmp/backtest_short_unbiased.png'

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
    return ticker_stats[:top_k]


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
# FEATURE SETS
# ==============================================================================
BQ_FEATURE_NAMES = [
    'n_configs', 'has_both_alphas', 'cross_alpha_agree', 'agree_18_best',
    'opt_wr_median', 'pct_static', 'pct_atr_sl', 'pct_grace',
    'pct_limit', 'va_median', 'va_pos_rate', 'va_opt_ratio_med',
]

ALL_FEATURE_NAMES = FEATURE_NAMES_OHLCV + FEATURE_NAMES_SIGNAL + BQ_FEATURE_NAMES + FEATURE_NAMES_MICRO


# ==============================================================================
# WALK-FORWARD FUNCTIONS
# ==============================================================================
def compute_backward_oracle(all_trades, as_of_date, top_k=TOP_K, min_avg_win=MIN_AVG_WIN):
    """Same rule as compute_test_oracle but using only trades BEFORE as_of_date."""
    ticker_stats = []
    for ticker, trades in all_trades.items():
        past = [t['pnl'] for t in trades if t['date'] < as_of_date]
        if not past:
            continue
        wins = [p for p in past if p > 0]
        avg_win = np.mean(wins) if wins else 0
        if avg_win < min_avg_win:
            continue
        ticker_stats.append({
            'ticker': ticker,
            'total_pnl': sum(past),
            'n': len(past),
            'avg': np.mean(past),
            'avg_win': avg_win,
        })
    ticker_stats.sort(key=lambda x: x['total_pnl'], reverse=True)
    return [t['ticker'] for t in ticker_stats[:top_k]]

def compute_backward_target(all_trades, bq_target, tickers, as_of_date,
                             blend=DYNAMIC_TARGET_BLEND, min_trades=MIN_TRADES_FOR_BLEND):
    """Backward-looking target: blend of actual mean PnL + BQ prior."""
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

def compute_all_features_at_date(tickers, as_of_date, static_bq_feats,
                                  daily_probs_cache, all_configs, feature_cache):
    """Compute 49 features for all tickers at as_of_date (cached)."""
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

def train_and_select(tickers, features, y_dict, top_k=TOP_K):
    """Train Ridge on features/target, return top-K tickers by predicted score."""
    valid = [t for t in tickers if t in features and t in y_dict]
    if len(valid) < top_k:
        return valid[:top_k], {}

    X = np.array([[features[t].get(f, 0) for f in ALL_FEATURE_NAMES]
                  for t in valid], dtype=np.float64)
    y = np.array([y_dict[t] for t in valid], dtype=np.float64)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_s, y)
    preds = model.predict(X_s)

    pred_dict = dict(zip(valid, preds))
    ranking = sorted(zip(valid, preds), key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in ranking[:top_k]]
    return selected, pred_dict


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  WALK-FORWARD UNBIASED BACKTEST — SHORT")
print(f"  Ridge + ALL (49 features) | Weekly rebalancing")
print(f"  Target: backward-looking PnL (zero future information)")
print(f"  Burn-in: {BURN_IN_WEEKS} weeks (BQ prior) → model takes over")
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
print(f"    FINAL ORACLE (H2, forward-looking ceiling): {', '.join(oracle_tickers)}")
for ts in test_oracle_list:
    print(f"      {ts['ticker']:<7} H2: {ts['h2_total_pnl']:>+7.2%} ({ts['h2_n']} trades, WR {ts['h2_wr']:.0%})")

_release_memory()


# -- PHASE 1: Walk-Forward --
print(f"\n  PHASE 1: Walk-Forward (weekly rebalancing, {len(trading_days)} trading days)...")

valid_tradeable = [t for t in tradeable_tickers
                   if t in ticker_data and t in daily_probs_cache]
print(f"    {len(valid_tradeable)} tradeable tickers")

feature_cache = {}
burn_in_end = trading_days[min(BURN_IN_WEEKS * 5, len(trading_days) - 1)]
print(f"    Burn-in ends: {burn_in_end}")

model_daily_active = {}
rebalance_log = []
current_selection = list(valid_tradeable)
n_rebalances = 0

for day_idx, day in enumerate(trading_days):
    is_rebal = (day.weekday() == 0) or (day == trading_days[0])

    if is_rebal:
        backward_oracle = compute_backward_oracle(all_trades, day, TOP_K, MIN_AVG_WIN)
        backward_oracle_set = set(backward_oracle)

        if day <= burn_in_end:
            y_dict = {t: bq_target.get(t, 0) for t in valid_tradeable}
            source = 'burn_in'
        else:
            y_dict = compute_backward_target(
                all_trades, bq_target, valid_tradeable, day)
            source = 'model'

        features = compute_all_features_at_date(
            valid_tradeable, day, static_bq_feats,
            daily_probs_cache, all_configs, feature_cache)

        selected, pred_dict = train_and_select(valid_tradeable, features, y_dict)
        prev_set = set(current_selection) if current_selection else set()
        current_selection = selected
        n_rebalances += 1

        n_total_past = sum(len([t for t in all_trades.get(tk, []) if t['date'] < day])
                          for tk in valid_tradeable)

        rebalance_log.append({
            'date': day,
            'week': n_rebalances,
            'tickers': selected.copy(),
            'backward_oracle': backward_oracle.copy(),
            'overlap_final': len(set(selected) & oracle_set),
            'overlap_backward': len(set(selected) & backward_oracle_set),
            'backward_vs_final': len(backward_oracle_set & oracle_set),
            'n_backward_oracle': len(backward_oracle),
            'changes': len(set(selected).symmetric_difference(prev_set)),
            'source': source,
            'n_trades_so_far': n_total_past,
        })

        if day_idx % 5 == 0 or n_rebalances <= 5 or source == 'model':
            ov_f = rebalance_log[-1]['overlap_final']
            ov_b = rebalance_log[-1]['backward_vs_final']
            n_bo = rebalance_log[-1]['n_backward_oracle']
            print(f"    Wk{n_rebalances:>2} {str(day):<12} [{source:<7}] "
                  f"Model→Final: {ov_f}/{TOP_K}  Back→Final: {ov_b}/{n_bo:<2}  "
                  f"Trades so far: {n_total_past}")

    model_daily_active[day] = set(current_selection)

print(f"\n    {n_rebalances} rebalances completed")

_release_memory()


# -- Collect strategy trades --
print(f"\n  Collecting strategy trades...")

current_set = set(CURRENT_TICKERS)
strat_trades = {
    'MODEL': [], 'ALL': [], 'ORACLE': [], 'CURRENT': [],
}

for ticker, trades in all_trades.items():
    for trade in trades:
        d = trade['date']
        strat_trades['ALL'].append(trade)
        if ticker in model_daily_active.get(d, set()):
            strat_trades['MODEL'].append(trade)
        if ticker in oracle_set:
            strat_trades['ORACLE'].append(trade)
        if ticker in current_set:
            strat_trades['CURRENT'].append(trade)

for sn in strat_trades:
    print(f"    {sn:<8}: {len(strat_trades[sn])} trades")


# ==============================================================================
# SECTION 1: Strategy Comparison
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 1: Strategy Comparison (UNBIASED)")
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

STRAT_NAMES = ['MODEL', 'ALL', 'ORACLE', 'CURRENT']

print(f"\n  {'Strategy':<10} | {'─── H2 (Jun-Jul) ──────────────────────':>42} | {'─── FULL (Apr-Jul) ────────────────────':>42}")
print(f"  {'':10} | {'N':>4} {'Total':>8} {'Avg':>8} {'WR':>5} {'Sharpe':>7} | {'N':>4} {'Total':>8} {'Avg':>8} {'WR':>5} {'Sharpe':>7}")
print(f"  {'-'*100}")

h2_stats = {}
full_stats = {}
for sn in STRAT_NAMES:
    h2 = compute_bt_stats(strat_trades[sn], half2_start_d, test_end_d)
    full = compute_bt_stats(strat_trades[sn], test_start_d, test_end_d)
    h2_stats[sn] = h2
    full_stats[sn] = full
    tag = ' [ceiling]' if sn == 'ORACLE' else (' [UNBIASED]' if sn == 'MODEL' else '')
    print(f"  {sn:<10} | {h2['n']:>4} {h2['total']:>+7.1%} {h2['avg']:>+7.2%} {h2['wr']:>4.0%} {h2['sharpe']:>+6.2f} "
          f"| {full['n']:>4} {full['total']:>+7.1%} {full['avg']:>+7.2%} {full['wr']:>4.0%} {full['sharpe']:>+6.2f}{tag}")

model_vs_all = h2_stats['MODEL']['total'] - h2_stats['ALL']['total']
model_vs_current = h2_stats['MODEL']['total'] - h2_stats['CURRENT']['total']
oracle_ceiling = h2_stats['ORACLE']['total']
model_captured = h2_stats['MODEL']['total'] / oracle_ceiling * 100 if oracle_ceiling > 0 else 0

print(f"\n  MODEL vs ALL:     {model_vs_all:>+7.1%}")
print(f"  MODEL vs CURRENT: {model_vs_current:>+7.1%}")
print(f"  MODEL captures {model_captured:.0f}% of ORACLE ceiling")


# ==============================================================================
# SECTION 2: Walk-Forward Progression (THE KEY TABLE)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 2: Walk-Forward Progression — Does Model Learn?")
print(f"{'='*W}")

print(f"\n  {'Wk':>3} {'Date':<12} {'Src':<7} | {'Mod→Fin':>7} {'Bk→Fin':>7} {'Mod→Bk':>7} | {'#Trades':>7} {'Chg':>3} | Model Selection")
print(f"  {'-'*105}")

for r in rebalance_log:
    bo_str = f"{r['backward_vs_final']}/{r['n_backward_oracle']}" if r['n_backward_oracle'] > 0 else "  -/-"
    tickers_str = ','.join(r['tickers'][:8])
    oracle_matches = [t for t in r['tickers'] if t in oracle_set]
    oracle_mark = f" [{','.join(oracle_matches)}]" if oracle_matches else ""
    print(f"  {r['week']:>3} {str(r['date']):<12} {r['source']:<7} | "
          f"{r['overlap_final']:>3}/{TOP_K}   {bo_str:>5}   {r['overlap_backward']:>3}/{TOP_K} | "
          f"{r['n_trades_so_far']:>7} {r['changes']:>3} | {tickers_str}")


# ==============================================================================
# SECTION 3: Backward Oracle Convergence
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 3: Backward Oracle Convergence")
print(f"{'='*W}")

model_entries = [r for r in rebalance_log if r['source'] == 'model']
if model_entries:
    first_model = model_entries[0]
    last_model = model_entries[-1]
    print(f"\n  First model rebalance (Wk{first_model['week']}, {first_model['date']}):")
    print(f"    Backward oracle: {', '.join(first_model['backward_oracle'][:8])}")
    print(f"    Backward→Final overlap: {first_model['backward_vs_final']}/{first_model['n_backward_oracle']}")
    print(f"    Model→Final overlap: {first_model['overlap_final']}/{TOP_K}")
    print(f"\n  Last model rebalance (Wk{last_model['week']}, {last_model['date']}):")
    print(f"    Backward oracle: {', '.join(last_model['backward_oracle'][:8])}")
    print(f"    Backward→Final overlap: {last_model['backward_vs_final']}/{last_model['n_backward_oracle']}")
    print(f"    Model→Final overlap: {last_model['overlap_final']}/{TOP_K}")

    half2_rebalances = [r for r in model_entries if r['date'] >= half2_start_d]
    if half2_rebalances:
        avg_overlap = float(np.mean([r['overlap_final'] for r in half2_rebalances]))
        avg_backward = float(np.mean([r['backward_vs_final'] for r in half2_rebalances]))
        print(f"\n  H2 average overlap:")
        print(f"    Model→Final: {avg_overlap:.1f}/{TOP_K} ({avg_overlap/TOP_K*100:.0f}%)")
        print(f"    Backward→Final: {avg_backward:.1f}/{TOP_K}")

    convergence_week = None
    for r in rebalance_log:
        if r['backward_vs_final'] >= TOP_K * 0.75 and r['n_backward_oracle'] >= TOP_K:
            convergence_week = r['week']
            break
    if convergence_week:
        print(f"\n  Backward oracle converges to ≥75% of final oracle at Week {convergence_week}")
    else:
        print(f"\n  Backward oracle never reaches 75% overlap with final oracle")


# ==============================================================================
# SECTION 4: Ticker Stability
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 4: Ticker Selection Stability")
print(f"{'='*W}")

h2_days = [d for d in trading_days if d >= half2_start_d]
n_h2_days = len(h2_days)

ticker_presence = defaultdict(int)
for d in h2_days:
    for t in model_daily_active.get(d, set()):
        ticker_presence[t] += 1

sorted_presence = sorted(ticker_presence.items(), key=lambda x: x[1], reverse=True)

print(f"\n  {'Ticker':<7} {'H2 Days':>8} {'Pct':>5} | {'Oracle':>7} | {'H2 PnL':>8} {'H2 WR':>6}")
print(f"  {'-'*55}")
for ticker, n_days in sorted_presence[:15]:
    pct = n_days / n_h2_days * 100
    oc = '  Y  ' if ticker in oracle_set else '  -  '
    h2_trades = [t for t in all_trades.get(ticker, []) if t['date'] >= half2_start_d]
    h2_pnls = [t['pnl'] for t in h2_trades]
    h2_total = sum(h2_pnls) if h2_pnls else 0
    h2_wr = float(np.mean(np.array(h2_pnls) > 0)) if h2_pnls else 0
    print(f"  {ticker:<7} {n_days:>8} {pct:>4.0f}% | {oc} | {h2_total:>+7.2%} {h2_wr:>5.0%}")

core_tickers = set()
all_model_sets = [set(r['tickers']) for r in rebalance_log if r['source'] == 'model']
if all_model_sets:
    core_tickers = all_model_sets[0]
    for s in all_model_sets[1:]:
        core_tickers = core_tickers & s
    print(f"\n  Core tickers (in EVERY model rebalance): {', '.join(sorted(core_tickers)) if core_tickers else '(none)'}")
    print(f"  Core overlap with ORACLE: {len(core_tickers & oracle_set)}/{len(core_tickers)}")


# ==============================================================================
# SECTION 5: Monthly Breakdown
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 5: Monthly PnL Breakdown")
print(f"{'='*W}")

months_in_range = sorted(set(t['date'].month for sn in STRAT_NAMES for t in strat_trades[sn]))
month_names = {4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul'}

print(f"\n  {'Strategy':<10} |", end='')
for m in months_in_range:
    print(f" {month_names.get(m, str(m)):>8}", end='')
print(f" | {'Total':>8}")
print(f"  {'-'*60}")

for sn in STRAT_NAMES:
    monthly = defaultdict(list)
    for t in strat_trades[sn]:
        monthly[t['date'].month].append(t['pnl'])
    print(f"  {sn:<10} |", end='')
    running = 0
    for m in months_in_range:
        total = sum(monthly.get(m, []))
        running += total
        print(f" {total:>+7.1%}", end='')
    print(f" | {running:>+7.1%}")


# ==============================================================================
# SECTION 6: Recommendation
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 6: Verdict — Is Model Selection Worth It?")
print(f"{'='*W}")

model_h2 = h2_stats['MODEL']
all_h2 = h2_stats['ALL']
oracle_h2 = h2_stats['ORACLE']
current_h2 = h2_stats['CURRENT']

print(f"\n  1. BIAS CHECK:")
print(f"     This test uses ZERO future information.")
print(f"     At each Monday rebalance, the model sees only backward trades + current features.")
print(f"     Target: expanding mean PnL (blended with BQ prior for sparse tickers).")
print(f"     Backward oracle rule: same as final oracle (total PnL + avg_win >= {MIN_AVG_WIN:.0%}).")

print(f"\n  2. DOES MODEL SELECTION ADD VALUE?")
print(f"     MODEL H2: {model_h2['total']:>+7.1%} ({model_h2['n']} trades, avg {model_h2['avg']:>+.2%})")
print(f"     ALL   H2: {all_h2['total']:>+7.1%} ({all_h2['n']} trades, avg {all_h2['avg']:>+.2%})")
print(f"     Delta:    {model_h2['total'] - all_h2['total']:>+7.1%}")
if model_h2['total'] > all_h2['total']:
    print(f"     ✓ MODEL BEATS ALL — selection adds value")
elif model_h2['avg'] > all_h2['avg']:
    print(f"     ~ MODEL has better avg PnL but fewer trades — net effect mixed")
else:
    print(f"     ✗ MODEL does NOT beat ALL — selection doesn't help yet")

print(f"\n  3. MODEL vs CURRENT PORTFOLIO:")
print(f"     MODEL   H2: {model_h2['total']:>+7.1%}")
print(f"     CURRENT H2: {current_h2['total']:>+7.1%}")
print(f"     Delta:      {model_h2['total'] - current_h2['total']:>+7.1%}")
if model_h2['total'] > current_h2['total']:
    print(f"     ✓ MODEL BEATS CURRENT — worth switching")
else:
    print(f"     ✗ CURRENT is already better")

print(f"\n  4. ORACLE CEILING:")
print(f"     ORACLE H2: {oracle_h2['total']:>+7.1%}")
print(f"     MODEL captures {model_captured:.0f}% of ceiling")

if model_entries:
    final_overlap = model_entries[-1]['overlap_final']
    print(f"\n  5. CONVERGENCE:")
    print(f"     Final rebalance: Model picks {final_overlap}/{TOP_K} oracle tickers")
    if final_overlap >= 6:
        print(f"     ✓ STRONG convergence — model learned the right tickers")
    elif final_overlap >= 4:
        print(f"     ~ MODERATE convergence — model captures some signal")
    else:
        print(f"     ✗ WEAK convergence — model hasn't locked onto oracle tickers")

print(f"\n  6. DEPLOYMENT READINESS:")
deploy_ready = (model_h2['total'] > all_h2['total'] and model_h2['total'] > current_h2['total']
                and model_h2['avg'] > 0.005)
if deploy_ready:
    print(f"     ✓ READY — model adds value over both ALL and CURRENT, avg > 0.5%")
    print(f"     Recommended: weekly Monday rebalance at 6 AM with Ridge + ALL features")
else:
    print(f"     △ NOT YET — gather more data or test different model/features")
    if model_h2['total'] > all_h2['total']:
        print(f"       Model beats ALL but hasn't passed all criteria")
    print(f"       Consider: more burn-in, or running paper-trade Aug forward")

print(f"\n{'='*W}")


# ==============================================================================
# CHARTS (2×3)
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle(f'Walk-Forward UNBIASED Backtest — SHORT\n'
             f'Ridge + ALL (49 features) | Weekly rebalance | '
             f'Burn-in: {BURN_IN_WEEKS}wk | Test: {TEST_START} to {TEST_END}',
             fontsize=13, fontweight='bold')

STRAT_COLORS = {
    'MODEL': '#dc2626', 'ALL': '#6b7280',
    'ORACLE': '#16a34a', 'CURRENT': '#a3a3a3',
}

# Panel (0,0): Equity curves
ax = axes[0, 0]
for sn in STRAT_NAMES:
    trades = sorted(strat_trades[sn], key=lambda t: t['date'])
    if not trades:
        continue
    cum_pnl = np.cumsum([t['pnl'] for t in trades])
    dates_plot = [t['date'] for t in trades]
    lw = 2.5 if sn == 'MODEL' else (1.5 if sn == 'ORACLE' else 1.0)
    ls = '--' if sn in ('ORACLE', 'CURRENT') else '-'
    ax.plot(dates_plot, cum_pnl * 100, color=STRAT_COLORS[sn],
            linewidth=lw, linestyle=ls, label=sn, alpha=0.9)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(half2_start_d, color='gray', linewidth=0.5, linestyle=':', label='H2 start')
ax.axvline(burn_in_end, color='orange', linewidth=0.5, linestyle=':', alpha=0.5, label='Burn-in end')
ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('Equity Curves (MODEL is fully unbiased)')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel (0,1): H2 total PnL bar chart
ax = axes[0, 1]
bar_vals = [h2_stats[s]['total'] * 100 for s in STRAT_NAMES]
bar_colors = [STRAT_COLORS[s] for s in STRAT_NAMES]
bars = ax.bar(range(len(STRAT_NAMES)), bar_vals, color=bar_colors, alpha=0.85,
              edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(STRAT_NAMES)))
ax.set_xticklabels(STRAT_NAMES, fontsize=10)
ax.set_ylabel('H2 Total PnL (%)')
ax.set_title('H2 (Jun-Jul) Total PnL by Strategy')
ax.axhline(0, color='gray', linewidth=0.5)
for i, v in enumerate(bar_vals):
    ax.text(i, v + (1 if v >= 0 else -3), f'{v:+.0f}%', ha='center', fontsize=9, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Panel (0,2): Model→Oracle overlap progression
ax = axes[0, 2]
rebal_dates = [r['date'] for r in rebalance_log]
model_final_overlap = [r['overlap_final'] for r in rebalance_log]
back_final_overlap = [r['backward_vs_final'] for r in rebalance_log]
ax.plot(rebal_dates, model_final_overlap, 'o-', color='#dc2626',
        linewidth=2, markersize=5, label=f'Model→Final Oracle (/{TOP_K})')
ax.plot(rebal_dates, back_final_overlap, 's--', color='#2563eb',
        linewidth=1.5, markersize=4, label=f'Backward→Final Oracle')
ax.axhline(TOP_K, color='#16a34a', linewidth=1, linestyle=':', alpha=0.5, label=f'Perfect ({TOP_K})')
ax.axvline(half2_start_d, color='gray', linewidth=0.5, linestyle=':')
ax.axvline(burn_in_end, color='orange', linewidth=0.5, linestyle=':', alpha=0.5)
ax.set_ylabel('Overlap count')
ax.set_title('Convergence: Does Model Learn the Right Tickers?')
ax.set_ylim(-0.5, TOP_K + 0.5)
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel (1,0): Ticker selection heatmap (weeks in H2)
ax = axes[1, 0]
h2_rebalances = [r for r in rebalance_log if r['date'] >= half2_start_d]
all_selected = set()
for r in h2_rebalances:
    all_selected.update(r['tickers'])
all_selected = sorted(all_selected)
if h2_rebalances and all_selected:
    heatmap = np.zeros((len(all_selected), len(h2_rebalances)))
    for j, r in enumerate(h2_rebalances):
        for i, ticker in enumerate(all_selected):
            if ticker in r['tickers']:
                heatmap[i, j] = 1
            if ticker in oracle_set:
                heatmap[i, j] += 0.5
    im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1.5)
    ax.set_xticks(range(len(h2_rebalances)))
    ax.set_xticklabels([f"Wk{r['week']}" for r in h2_rebalances], fontsize=6, rotation=45)
    ax.set_yticks(range(len(all_selected)))
    labels = [f'{t} {"*" if t in oracle_set else ""}' for t in all_selected]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title('H2 Ticker Selection (green=selected+oracle)')
else:
    ax.text(0.5, 0.5, 'No H2 rebalances', ha='center', va='center')
    ax.set_title('H2 Ticker Selection')

# Panel (1,1): Monthly PnL grouped bars
ax = axes[1, 1]
n_months = len(months_in_range)
bar_width = 0.2
x_months = np.arange(n_months)
for i, sn in enumerate(STRAT_NAMES):
    monthly_totals = []
    monthly = defaultdict(list)
    for t in strat_trades[sn]:
        monthly[t['date'].month].append(t['pnl'])
    for m in months_in_range:
        monthly_totals.append(sum(monthly.get(m, [])) * 100)
    ax.bar(x_months + i * bar_width, monthly_totals, bar_width,
           label=sn, color=STRAT_COLORS[sn], alpha=0.85)
ax.set_xticks(x_months + 1.5 * bar_width)
ax.set_xticklabels([month_names.get(m, str(m)) for m in months_in_range])
ax.set_ylabel('Total PnL (%)')
ax.set_title('Monthly PnL by Strategy')
ax.axhline(0, color='gray', linewidth=0.5)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# Panel (1,2): Rolling 2-week Sharpe: MODEL vs ALL vs ORACLE
ax = axes[1, 2]
roll_window = 10
for sn in ['MODEL', 'ALL', 'ORACLE']:
    daily_pnl = defaultdict(float)
    for t in strat_trades[sn]:
        daily_pnl[t['date']] += t['pnl']
    if not daily_pnl:
        continue
    sorted_dates = sorted(daily_pnl.keys())
    pnl_series = pd.Series([daily_pnl.get(d, 0) for d in sorted_dates], index=sorted_dates)
    rolling_mean = pnl_series.rolling(roll_window, min_periods=3).mean()
    rolling_std = pnl_series.rolling(roll_window, min_periods=3).std()
    rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan)
    lw = 2.5 if sn == 'MODEL' else 1.0
    ls = '--' if sn == 'ORACLE' else '-'
    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
            color=STRAT_COLORS[sn], linewidth=lw, linestyle=ls, label=sn, alpha=0.8)
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(half2_start_d, color='gray', linewidth=0.5, linestyle=':')
ax.set_ylabel('Rolling 10-day Sharpe')
ax.set_title('Rolling Sharpe: MODEL vs ALL')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(CHART_PATH, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Chart saved: {CHART_PATH}")

_release_memory()
