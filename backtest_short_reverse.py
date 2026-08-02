"""
backtest_short_reverse.py — Reverse-Engineered SHORT Config Selection
=====================================================================
Analyzes ALL tickers in BQ, finds golden configs from test data,
applies data-driven filters, and validates with time-split backtesting.

Three ticker selection strategies:
  A) ORACLE  [BIASED]   — top 10 by val_b positive rate (uses test data)
  B) FILTERED [UNBIASED] — val_a-only filters (no test data)
  C) CURRENT [BASELINE]  — original 4 tickers

Two config selection methods per list:
  1) Ensemble — best is_ensemble=True row by val_a
  2) Filtered individual — exclude limit_entry + tight overfit ratio

Time-split validation:
  FULL:  Apr 1 – Jul 23
  HALF1: Apr 1 – May 31  (in-sample)
  HALF2: Jun 1 – Jul 23  (out-of-sample)

GPU NOT required. Colab-ready (no if __name__).
"""

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
RESULTS_DIR    = "Algos/Results_all_v2"
INFERENCE_DIR  = "Data/Inference_v2"

FULL_START = '2026-04-01'
FULL_END   = '2026-07-23'
HALF1_END  = '2026-05-31'
HALF2_START = '2026-06-01'

ENTRY_MIN     = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.002
STARTING_CAPITAL = 200_000

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1')

CURRENT_TICKERS = ['VOYG', 'NTLA', 'CLF', 'SMR']
ORACLE_TOP_N = 10
MAX_CSV_TICKERS = 20

FILTERED_VA_OPT_RATIO = 2.0
FILTERED_MIN_CONFIGS = 5
FILTERED_VA_POSITIVE_RATE = 0.50
FILTERED_VA_FLOOR = 0.01

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110


# ==============================================================================
# DATA LOADING (reused from backtest_short_csv.py)
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
# SHORT TRADE SIMULATION (reused from backtest_short_csv.py)
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

def _row_to_cfg(ticker, row, source):
    exp = row['experiment'] if isinstance(row, pd.Series) else row.get('experiment', '')
    base_exp = _base_experiment(exp)
    return {
        'ticker': ticker,
        'experiment': exp,
        '_base_experiment': base_exp,
        'alpha': base_exp.split('_')[0],
        'source': source,
        'tpsl_mode': row['tpsl_mode'],
        'entry_minute': _safe_int(row.get('entry_minute', 375), 375),
        'prob_thresh': _safe_float(row['prob_thresh']),
        'tp_pct': _safe_float(row.get('tp_pct', 0)),
        'sl_pct': _safe_float(row.get('sl_pct', 0)),
        'sl_mult': _safe_float(row.get('sl_mult', 0)),
        'grace_minutes': _safe_int(row.get('grace_minutes', 0)),
        'val_pnl': _safe_float(row.get('val_a_avg_pnl', row.get('mean_va_pnl', 0))),
        'test_pnl': _safe_float(row.get('val_b_avg_pnl', row.get('vb_pnl', 0))),
    }

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
# TIME-SPLIT BACKTEST
# ==============================================================================
def run_csv_backtest_split(configs, csv_scores_cache, ticker_data,
                           start_date, end_date, label):
    """Run SHORT backtest with explicit date range."""
    test_start_dt = pd.Timestamp(start_date).date()
    test_end_dt   = pd.Timestamp(end_date).date()

    daily_probs = {}
    for ticker, cfg in sorted(configs.items()):
        if ticker not in ticker_data:
            continue
        base_exp = cfg['_base_experiment']
        entry_min = cfg['entry_minute']
        seed_scores = csv_scores_cache.get((ticker, base_exp))
        if not seed_scores:
            continue
        all_daily = {}
        for seed_id, sdf in seed_scores.items():
            bar_mins = sdf.index.hour * 60 + sdf.index.minute
            entry_bars = sdf[bar_mins == entry_min]
            for dt, row in entry_bars.iterrows():
                d = dt.date()
                if d not in all_daily:
                    all_daily[d] = []
                all_daily[d].append(float(row['Prob_Spike']))
        daily_probs[ticker] = {d: np.mean(ps) for d, ps in all_daily.items()}

    all_trades = []
    per_ticker_trades = defaultdict(list)

    for ticker in sorted(configs.keys()):
        if ticker not in ticker_data or ticker not in daily_probs:
            continue
        cfg = configs[ticker]
        ohlcv = ticker_data[ticker]['ohlcv']
        probs = daily_probs[ticker]
        entry_min = cfg['entry_minute']
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

            trade = {
                'date': d_date, 'ticker': ticker, 'pnl': pnl,
                'prob': prob, 'close': float(prices[idx]),
                'mode': mode, 'tp': cfg['tp_pct'],
                'sl': cfg['sl_pct'] if mode != 'atr_sl' else cfg['sl_mult'],
            }
            all_trades.append(trade)
            per_ticker_trades[ticker].append(trade)

    return all_trades, per_ticker_trades


# ==============================================================================
# PHASE 1: BQ DISCOVERY — ALL TICKERS
# ==============================================================================
def query_all_bq():
    """Pull all survived configs across all tickers (ensemble + individual)."""
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


def compute_ticker_rankings(bq_df):
    """Rank tickers by val_b performance. Returns DataFrame sorted by vb_pos_rate."""
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = bq_df[bq_df['is_ensemble'] != True].copy()

    rows = []
    for ticker, grp in indiv.groupby('ticker'):
        n = len(grp)
        vb_pos = (grp['val_b_avg_pnl'] > 0).mean()
        vb_med = grp['val_b_avg_pnl'].median()
        vb_best = grp['val_b_avg_pnl'].max()
        va_med = grp['val_a_avg_pnl'].median()
        va_opt_ratio_med = (grp['val_a_avg_pnl'] / grp['opt_avg_pnl']).median()
        n_limit = (grp['tpsl_mode'] == 'limit_entry').sum()
        best_row = grp.loc[grp['val_b_avg_pnl'].idxmax()]
        rows.append({
            'ticker': ticker, 'n_configs': n,
            'vb_pos_rate': vb_pos, 'vb_median': vb_med, 'vb_best': vb_best,
            'va_median': va_med, 'va_opt_ratio_med': va_opt_ratio_med,
            'pct_limit': n_limit / n,
            'best_vb_mode': best_row['tpsl_mode'],
            'best_vb_prob': best_row['prob_thresh'],
        })
    rankings = pd.DataFrame(rows).sort_values('vb_pos_rate', ascending=False)
    return rankings


# ==============================================================================
# PHASE 2: TICKER LIST CONSTRUCTION
# ==============================================================================
def build_ticker_lists(rankings, bq_df):
    """Build ORACLE, FILTERED, and CURRENT ticker lists."""

    oracle = rankings.head(ORACLE_TOP_N)['ticker'].tolist()

    indiv = bq_df[(bq_df['is_ensemble'] == False) | (bq_df['is_ensemble'] != True)].copy()
    if indiv.empty:
        indiv = bq_df.copy()
    filt = indiv[indiv['tpsl_mode'] != 'limit_entry'].copy()
    filt = filt[filt['val_a_avg_pnl'] / filt['opt_avg_pnl'] <= FILTERED_VA_OPT_RATIO]

    filtered_tickers = []
    for ticker, grp in filt.groupby('ticker'):
        if len(grp) < FILTERED_MIN_CONFIGS:
            continue
        va_pos_rate = (grp['val_a_avg_pnl'] >= FILTERED_VA_FLOOR).mean()
        if va_pos_rate < FILTERED_VA_POSITIVE_RATE:
            continue
        filtered_tickers.append({
            'ticker': ticker,
            'va_median': grp['val_a_avg_pnl'].median(),
            'n_filt': len(grp),
            'va_pos_rate': va_pos_rate,
        })

    filt_df = pd.DataFrame(filtered_tickers)
    if not filt_df.empty:
        filt_df = filt_df.sort_values('va_median', ascending=False)
        filtered = filt_df.head(ORACLE_TOP_N)['ticker'].tolist()
    else:
        filtered = []

    return {
        'ORACLE': oracle,
        'FILTERED': filtered,
        'CURRENT': CURRENT_TICKERS,
    }


# ==============================================================================
# PHASE 3: CONFIG SELECTION
# ==============================================================================
def select_configs_ensemble(bq_df, ticker_list):
    """Best ensemble row per ticker by val_a_avg_pnl."""
    ens = bq_df[bq_df['is_ensemble'] == True].copy()
    if ens.empty:
        ens = bq_df[bq_df['seed'] == -1].copy()
    ens = ens[ens['tpsl_mode'].isin(['static', 'atr_sl', 'grace'])]
    ens = ens[ens['val_a_avg_pnl'] / ens['opt_avg_pnl'] <= FILTERED_VA_OPT_RATIO]

    configs = {}
    for ticker in ticker_list:
        tk = ens[ens['ticker'] == ticker]
        if tk.empty:
            continue
        best = tk.loc[tk['val_a_avg_pnl'].idxmax()]
        configs[ticker] = _row_to_cfg(ticker, best, 'ensemble')
    return configs


def select_configs_filtered(bq_df, ticker_list):
    """Filtered individual: exclude limit_entry, tight overfit ratio, best mean val_a."""
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = bq_df[bq_df['is_ensemble'] != True].copy()
    indiv = indiv[indiv['tpsl_mode'].isin(['static', 'atr_sl', 'grace'])]
    indiv = indiv[indiv['val_a_avg_pnl'] / indiv['opt_avg_pnl'] <= FILTERED_VA_OPT_RATIO]

    grp_keys = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
                'prob_thresh', 'tp_pct', 'sl_pct', 'sl_mult', 'grace_minutes']
    present_keys = [k for k in grp_keys if k in indiv.columns]
    for k in present_keys:
        if indiv[k].isna().any():
            indiv[k] = indiv[k].fillna(0)

    configs = {}
    for ticker in ticker_list:
        tk = indiv[indiv['ticker'] == ticker]
        if tk.empty:
            continue
        sa = tk.groupby(present_keys).agg(
            n_seeds=('seed', 'nunique'),
            mean_va_pnl=('val_a_avg_pnl', 'mean'),
        ).reset_index()
        if sa.empty:
            continue
        best = sa.loc[sa['mean_va_pnl'].idxmax()]
        cfg = _row_to_cfg(ticker, best, 'filtered')
        cfg['n_seeds'] = int(best['n_seeds'])
        cfg['val_pnl'] = _safe_float(best['mean_va_pnl'])
        configs[ticker] = cfg
    return configs


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT MODEL — REVERSE-ENGINEERED CONFIG SELECTION")
print(f"  FULL: {FULL_START} → {FULL_END}")
print(f"  HALF1: {FULL_START} → {HALF1_END}  |  HALF2: {HALF2_START} → {FULL_END}")
print(f"  ENTRY: 6:15 PST ({ENTRY_MIN} min)  |  EXIT: 1:00 PM ({EXIT_TIME_MIN} min)")
print(f"  SLIPPAGE: {SLIPPAGE_PCT:.2%} each way  |  CAPITAL: ${STARTING_CAPITAL:,}")
print(f"{'='*W}")


# ── PHASE 1: BQ Discovery ──
print(f"\n  PHASE 1: BQ Discovery — querying ALL tickers...")
bq_df = query_all_bq()
n_tickers = bq_df['ticker'].nunique()
n_indiv = len(bq_df[bq_df['is_ensemble'] == False])
n_ens = len(bq_df[bq_df['is_ensemble'] == True])
print(f"    {len(bq_df)} total rows: {n_indiv} individual + {n_ens} ensemble across {n_tickers} tickers")

rankings = compute_ticker_rankings(bq_df)

print(f"\n  [BIASED] Full Ticker Ranking by val_b Positive Rate:")
print(f"  {'Ticker':<7} {'N':>5} {'VB+%':>6} {'VB_med':>8} {'VB_best':>8} "
      f"{'VA_med':>8} {'VA/OPT':>7} {'%Limit':>7} {'Best_Mode':<10}")
print(f"  {'─'*75}")
for _, r in rankings.iterrows():
    print(f"  {r['ticker']:<7} {r['n_configs']:>5} {r['vb_pos_rate']:>5.0%} "
          f"{r['vb_median']:>+7.2%} {r['vb_best']:>+7.2%} "
          f"{r['va_median']:>+7.2%} {r['va_opt_ratio_med']:>6.1f}x "
          f"{r['pct_limit']:>6.0%} {r['best_vb_mode']:<10}")


# ── PHASE 2: Ticker Lists ──
print(f"\n  PHASE 2: Ticker Selection Strategies")
ticker_lists = build_ticker_lists(rankings, bq_df)

for name, tickers in ticker_lists.items():
    bias = {'ORACLE': '[BIASED — uses val_b]',
            'FILTERED': '[UNBIASED — val_a only]',
            'CURRENT': '[BASELINE]'}[name]
    print(f"    {name} {bias}: {', '.join(tickers)} ({len(tickers)} tickers)")

all_tickers_union = sorted(set(
    ticker_lists['ORACLE'] + ticker_lists['FILTERED'] + ticker_lists['CURRENT']
))

if len(all_tickers_union) > MAX_CSV_TICKERS:
    keep = set(CURRENT_TICKERS)
    budget = MAX_CSV_TICKERS - len(keep)
    for t in ticker_lists['ORACLE'] + ticker_lists['FILTERED']:
        if len(keep) >= MAX_CSV_TICKERS:
            break
        keep.add(t)
    all_tickers_union = sorted(keep)
    ticker_lists['ORACLE'] = [t for t in ticker_lists['ORACLE'] if t in keep]
    ticker_lists['FILTERED'] = [t for t in ticker_lists['FILTERED'] if t in keep]
    print(f"    ⚠ Trimmed to {MAX_CSV_TICKERS} tickers for GCS bandwidth")

overlap_of = set(ticker_lists['ORACLE']) & set(ticker_lists['FILTERED'])
print(f"    ORACLE∩FILTERED = {{{', '.join(sorted(overlap_of))}}} ({len(overlap_of)} overlap)")
print(f"    Union for loading: {len(all_tickers_union)} tickers")


# ── PHASE 3: Config Selection ──
print(f"\n  PHASE 3: Config Selection")
strategy_configs = {}
for list_name, tickers in ticker_lists.items():
    for method_name, select_fn in [('Ensemble', select_configs_ensemble),
                                    ('Filtered', select_configs_filtered)]:
        key = f"{list_name}×{method_name}"
        cfgs = select_fn(bq_df, tickers)
        strategy_configs[key] = cfgs
        tks = sorted(cfgs.keys())
        print(f"    {key}: {len(cfgs)} tickers — {', '.join(tks)}")
        for tk in tks:
            c = cfgs[tk]
            print(f"      {tk}: {c['tpsl_mode']} prob>{c['prob_thresh']:.2f} "
                  f"TP={c['tp_pct']:.0%} SL={c['sl_pct']:.0%} slM={c['sl_mult']:.1f} "
                  f"va={c['val_pnl']:+.2%}")


# ── Load OHLCV data (shared) ──
print(f"\n  Loading OHLCV + ATR for {len(all_tickers_union)} tickers...")
ticker_data = {}
for ticker in all_tickers_union:
    df = load_master(ticker)
    if df is None:
        continue
    df = heal_data(df)
    ohlcv = precompute_ohlcv_arrays(df, FULL_START, FULL_END)
    if ohlcv is None:
        print(f"    {ticker}: no data in test period")
        continue
    atr_pct = compute_daily_atr_pct(df)
    atr_vals = atr_pct.reindex(df.index).fillna(0.05)
    s = pd.to_datetime(FULL_START).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(FULL_END).tz_localize('America/Los_Angeles') + timedelta(days=1)
    idx = df.index
    if str(idx.tz) != 'America/Los_Angeles':
        idx = idx.tz_convert('America/Los_Angeles')
    atr_cut = atr_vals[(idx >= s) & (idx < e)]
    ohlcv['atr_pct'] = atr_cut.values.astype(np.float64)
    ticker_data[ticker] = {'ohlcv': ohlcv, 'df': df}
    n_days = len(np.unique(ohlcv['dates']))
    print(f"    {ticker}: {n_days} trading days")
    del df
    gc.collect()


# ── Load CSV scores ──
print(f"\n  Loading CSV scores from GCS...")
csv_scores_cache = {}
all_exp_pairs = set()
for key, cfgs in strategy_configs.items():
    for ticker, cfg in cfgs.items():
        all_exp_pairs.add((ticker, cfg['_base_experiment']))

for ticker, base_exp in sorted(all_exp_pairs):
    if ticker not in ticker_data:
        continue
    if (ticker, base_exp) in csv_scores_cache:
        continue
    seed_scores = load_seed_scores(ticker, base_exp)
    if seed_scores:
        csv_scores_cache[(ticker, base_exp)] = seed_scores
        print(f"    {ticker}/{base_exp}: {len(seed_scores)} seeds")
    else:
        print(f"    {ticker}/{base_exp}: no CSV scores")
    gc.collect()


# ── PHASE 4: Run Backtests ──
print(f"\n  PHASE 4: Running backtests (6 strategies × 3 periods = 18 runs)...")
periods = [
    ('FULL',  FULL_START,  FULL_END),
    ('HALF1', FULL_START,  HALF1_END),
    ('HALF2', HALF2_START, FULL_END),
]

results = {}
for strat_key, cfgs in sorted(strategy_configs.items()):
    if not cfgs:
        for p_name, _, _ in periods:
            results[(strat_key, p_name)] = summarize_trades([], f"{strat_key}/{p_name}")
        continue
    for p_name, p_start, p_end in periods:
        np.random.seed(42)
        label = f"{strat_key}/{p_name}"
        trades, per_tk = run_csv_backtest_split(
            cfgs, csv_scores_cache, ticker_data, p_start, p_end, label)
        results[(strat_key, p_name)] = summarize_trades(trades, label)
        results[(strat_key, p_name)]['per_ticker'] = per_tk


# ── PHASE 5: Comparison ──
print(f"\n{'='*W}")
print(f"  PHASE 5: STRATEGY COMPARISON — SIDE-BY-SIDE")
print(f"{'='*W}")

strat_keys = sorted(strategy_configs.keys())

print(f"\n  {'Strategy':<22} │ {'── FULL (Apr-Jul) ──────':} │ {'── HALF1 (Apr-May) ─────':} │ {'── HALF2 (Jun-Jul) ─────':}")
print(f"  {'':22} │ {'N':>4} {'Tot':>7} {'Avg':>7} {'WR':>5} │ {'N':>4} {'Tot':>7} {'Avg':>7} {'WR':>5} │ {'N':>4} {'Tot':>7} {'Avg':>7} {'WR':>5}")
print(f"  {'─'*105}")

for sk in strat_keys:
    f = results[(sk, 'FULL')]
    h1 = results[(sk, 'HALF1')]
    h2 = results[(sk, 'HALF2')]
    print(f"  {sk:<22} │ {f['n']:>4} {f['total']:>+6.1%} {f['avg']:>+6.2%} {f['wr']:>4.0%} │ "
          f"{h1['n']:>4} {h1['total']:>+6.1%} {h1['avg']:>+6.2%} {h1['wr']:>4.0%} │ "
          f"{h2['n']:>4} {h2['total']:>+6.1%} {h2['avg']:>+6.2%} {h2['wr']:>4.0%}")

# ── Key question: does edge survive HALF2? ──
print(f"\n  ── KEY QUESTION: Does the edge survive out-of-sample (HALF2)? ──")
for sk in strat_keys:
    h1 = results[(sk, 'HALF1')]
    h2 = results[(sk, 'HALF2')]
    if h1['avg'] != 0:
        decay = h2['avg'] / h1['avg'] if h1['avg'] > 0 else float('nan')
        decay_str = f"{decay:+.0%}" if not np.isnan(decay) else "N/A"
    else:
        decay_str = "N/A"
    bias_tag = "[BIASED]" if "ORACLE" in sk else ("[UNBIASED]" if "FILTERED" in sk else "[BASELINE]")
    print(f"    {sk:<22} {bias_tag:<12} HALF1={h1['avg']:>+.2%} → HALF2={h2['avg']:>+.2%}  decay={decay_str}")

# ── Per-ticker for best strategy ──
best_sk = max(strat_keys, key=lambda sk: results[(sk, 'FULL')]['avg'])
print(f"\n  ── Per-Ticker Breakdown (best FULL strategy: {best_sk}) ──")
print(f"  {'Ticker':<7} │ {'FULL_Avg':>8} {'HALF1_Avg':>10} {'HALF2_Avg':>10} {'H2/H1':>7} {'Status':>8}")
print(f"  {'─'*60}")

per_tk_full = results.get((best_sk, 'FULL'), {}).get('per_ticker', {})
per_tk_h1 = results.get((best_sk, 'HALF1'), {}).get('per_ticker', {})
per_tk_h2 = results.get((best_sk, 'HALF2'), {}).get('per_ticker', {})

all_bt_tickers = sorted(set(list(per_tk_full.keys()) + list(per_tk_h1.keys()) + list(per_tk_h2.keys())))
for ticker in all_bt_tickers:
    f_trades = per_tk_full.get(ticker, [])
    h1_trades = per_tk_h1.get(ticker, [])
    h2_trades = per_tk_h2.get(ticker, [])
    f_avg = np.mean([t['pnl'] for t in f_trades]) if f_trades else 0
    h1_avg = np.mean([t['pnl'] for t in h1_trades]) if h1_trades else 0
    h2_avg = np.mean([t['pnl'] for t in h2_trades]) if h2_trades else 0
    if h1_avg > 0 and h2_avg > 0:
        status = "HOLDS"
    elif h1_avg > 0 and h2_avg <= 0:
        status = "DECAYED"
    elif h1_avg <= 0:
        status = "WEAK"
    else:
        status = "?"
    ratio = f"{h2_avg/h1_avg:.0%}" if h1_avg != 0 else "N/A"
    print(f"  {ticker:<7} │ {f_avg:>+7.2%} {h1_avg:>+9.2%} {h2_avg:>+9.2%} {ratio:>7} {status:>8}")


# ── Trade logs for best strategy ──
best_trades_full = results[(best_sk, 'FULL')].get('trades_df', pd.DataFrame())
if not best_trades_full.empty:
    print(f"\n  {best_sk} FULL trades ({len(best_trades_full)}):")
    for _, t in best_trades_full.sort_values('date').iterrows():
        print(f"    {t['date']}  {t['ticker']:<7}  pnl={t['pnl']:+.2%}  "
              f"prob={t['prob']:.3f}  close=${t['close']:.2f}")


# ── PHASE 6: Charts ──
print(f"\n  Generating comparison charts...")

STRAT_COLORS = {
    'ORACLE×Ensemble': '#1565C0', 'ORACLE×Filtered': '#42A5F5',
    'FILTERED×Ensemble': '#2E7D32', 'FILTERED×Filtered': '#66BB6A',
    'CURRENT×Ensemble': '#616161', 'CURRENT×Filtered': '#9E9E9E',
}
STRAT_LS = {k: ('-' if 'Ensemble' in k else '--') for k in STRAT_COLORS}

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Chart 1: Cumulative PnL — FULL period, all strategies
ax1 = axes[0, 0]
for sk in strat_keys:
    r = results[(sk, 'FULL')]
    if r['n'] == 0:
        continue
    df_t = r['trades_df'].sort_values('date')
    cum = df_t['pnl'].cumsum() * 100
    ax1.plot(pd.to_datetime(df_t['date'].values), cum.values,
             color=STRAT_COLORS.get(sk, '#666'), linestyle=STRAT_LS.get(sk, '-'),
             linewidth=2, label=sk)
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax1.set_ylabel('Cumulative PnL (%)')
ax1.set_title('All Strategies — FULL Period', fontsize=13, fontweight='bold')
ax1.legend(fontsize=7, loc='best')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Chart 2: HALF1 vs HALF2 avg PnL — grouped bar
ax2 = axes[0, 1]
x = np.arange(len(strat_keys))
bar_w = 0.35
h1_avgs = [results[(sk, 'HALF1')]['avg'] * 100 for sk in strat_keys]
h2_avgs = [results[(sk, 'HALF2')]['avg'] * 100 for sk in strat_keys]
ax2.bar(x - bar_w/2, h1_avgs, bar_w, label='HALF1 (Apr-May)', color='#1565C0', alpha=0.8)
ax2.bar(x + bar_w/2, h2_avgs, bar_w, label='HALF2 (Jun-Jul)', color='#E65100', alpha=0.8)
short_labels = [sk.replace('×', '\n') for sk in strat_keys]
ax2.set_xticks(x)
ax2.set_xticklabels(short_labels, fontsize=7)
ax2.set_ylabel('Avg PnL per Trade (%)')
ax2.set_title('In-Sample (HALF1) vs Out-of-Sample (HALF2)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.axhline(y=0, color='gray', linewidth=0.5)
ax2.grid(True, alpha=0.3, axis='y')

# Chart 3: Per-ticker cumulative PnL — best strategy, FULL period
ax3 = axes[1, 0]
TICKER_COLORS = {
    'WOLF': '#D32F2F', 'MP': '#1976D2', 'CCJ': '#388E3C', 'MVIS': '#7B1FA2',
    'SMR': '#F57C00', 'VOYG': '#0097A7', 'OKLO': '#5D4037', 'FCEL': '#C2185B',
    'KEEL': '#303F9F', 'NTLA': '#AFB42B', 'CLF': '#00796B', 'BE': '#E64A19',
    'MU': '#512DA8', 'APLD': '#0288D1', 'HUT': '#689F38',
}
for ticker in sorted(per_tk_full.keys()):
    trades = per_tk_full[ticker]
    if not trades:
        continue
    df_t = pd.DataFrame(trades).sort_values('date')
    cum = df_t['pnl'].cumsum() * 100
    ax3.plot(pd.to_datetime(df_t['date'].values), cum.values,
             label=ticker, color=TICKER_COLORS.get(ticker, '#666'), linewidth=2)
ax3.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax3.set_ylabel('Cumulative PnL (%)')
ax3.set_title(f'Per-Ticker ({best_sk})', fontsize=13, fontweight='bold')
ax3.legend(fontsize=8, loc='best', ncol=2)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Chart 4: Ticker-level FULL avg PnL — horizontal bar
ax4 = axes[1, 1]
if all_bt_tickers:
    tk_avgs = []
    for tk in all_bt_tickers:
        trades = per_tk_full.get(tk, [])
        tk_avgs.append(np.mean([t['pnl'] for t in trades]) * 100 if trades else 0)
    colors = ['#2E7D32' if v > 0 else '#C62828' for v in tk_avgs]
    y_pos = np.arange(len(all_bt_tickers))
    ax4.barh(y_pos, tk_avgs, color=colors, alpha=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(all_bt_tickers, fontsize=9)
    ax4.set_xlabel('Avg PnL per Trade (%)')
    ax4.set_title(f'Per-Ticker Avg PnL ({best_sk})', fontsize=13, fontweight='bold')
    ax4.axvline(x=0, color='gray', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='x')

fig.suptitle(f'SHORT Model — Reverse-Engineered Config Selection\n'
             f'FULL: {FULL_START}→{FULL_END}  |  HALF1: →{HALF1_END}  |  HALF2: {HALF2_START}→',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/tmp/backtest_short_reverse.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Chart saved: /tmp/backtest_short_reverse.png")


# ── Summary ──
print(f"\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")
best_full = max(strat_keys, key=lambda sk: results[(sk, 'FULL')]['avg'])
best_h2 = max(strat_keys, key=lambda sk: results[(sk, 'HALF2')]['avg'])
r_full = results[(best_full, 'FULL')]
r_h2 = results[(best_h2, 'HALF2')]
print(f"  Best FULL strategy:  {best_full}  — {r_full['n']} trades, Avg={r_full['avg']:+.2%}, WR={r_full['wr']:.0%}")
print(f"  Best HALF2 strategy: {best_h2}  — {r_h2['n']} trades, Avg={r_h2['avg']:+.2%}, WR={r_h2['wr']:.0%}")
if best_full == best_h2:
    print(f"  ✓ Same strategy wins both — edge likely structural")
else:
    print(f"  ✗ Different winners — FULL strategy may be overfit")
print(f"{'='*W}")
