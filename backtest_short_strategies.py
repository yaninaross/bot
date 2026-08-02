# %% ── Comprehensive SHORT Strategy Evaluation ────────────────────────────────
#
# Tests ALL ways to combine 12 walkforward folds (400-405 α=2.0, 410-415 α=1.0)
# into trading strategies on TEST period (Apr-Jul 2026).
#
# Strategies: single-fold, fold-group ensemble, voting, ticker-gated
# ~25 strategies evaluated on same data. No optimization — pure OOS.

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
BUCKET_NAME   = "time-series-bucket-ys"
RESULTS_DIR   = "Algos/Results_all_v2"
INFERENCE_DIR = "Data/Inference_v2"

TEST_START = '2026-04-01'
TEST_END   = '2026-07-28'
ENTRY_MIN  = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.002

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

FOLDS_A2 = [400, 401, 402, 403, 404, 405]
FOLDS_A1 = [410, 411, 412, 413, 414, 415]
ALL_FOLDS = FOLDS_A2 + FOLDS_A1

FOLD_EXPERIMENTS = {f: f'{f}_XSECTOR' for f in ALL_FOLDS}
FOLD_MODEL_VERSIONS = {f: f'robust_wf_v12_short_{f}' for f in ALL_FOLDS}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_ALLOWED_MODES = ('static', 'atr_sl', 'grace')

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120


# ==============================================================================
# DATA LOADING — OHLCV
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
# DATA LOADING — SEED SCORES
# ==============================================================================
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


def execute_trade(ohlcv, entry_idx, cfg):
    mode = cfg['tpsl_mode']
    if mode == 'atr_sl':
        atr_arr = ohlcv.get('atr_pct', np.full(len(ohlcv['prices']), 0.05))
        pnl, td, filled = run_short_trade_atr_sl(
            ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'],
            entry_idx, cfg['tp_pct'], float(atr_arr[entry_idx]), cfg['sl_mult'], EXIT_TIME_MIN)
    elif mode == 'grace':
        pnl, td, filled = run_short_trade_grace(
            ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'],
            entry_idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['grace_minutes'])
    else:
        pnl, td, filled = run_short_trade_static(
            ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'],
            entry_idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)
    if filled:
        pnl -= 2 * SLIPPAGE_PCT
    return pnl, td, filled


# ==============================================================================
# BQ CONFIG LOADING
# ==============================================================================
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


def load_all_bq_configs():
    mv_str = ", ".join(f"'{FOLD_MODEL_VERSIONS[f]}'" for f in ALL_FOLDS)
    modes_str = ", ".join(f"'{m}'" for m in BQ_ALLOWED_MODES)
    q = f"""
    SELECT ticker, experiment, tpsl_mode, model_version,
           prob_thresh, tp_pct, sl_pct, sl_mult, grace_minutes,
           val_a_avg_pnl, val_a_win_rate, val_a_count,
           val_b_avg_pnl, val_b_count,
           opt_avg_pnl, is_ensemble
    FROM `{BQ_TABLE}`
    WHERE model_version IN ({mv_str})
      AND is_ensemble = TRUE
      AND survived_a = TRUE
      AND val_a_count >= 2
      AND val_a_avg_pnl > 0
      AND tpsl_mode IN ({modes_str})
    """
    df = bq_client.query(q).to_dataframe()
    df['fold'] = df['model_version'].str.extract(r'_(\d+)$').astype(int)

    configs = {}
    for _, row in df.iterrows():
        ticker = row['ticker']
        fold = row['fold']
        key = (ticker, fold)
        if key not in configs or row['val_a_avg_pnl'] > configs[key]['val_pnl']:
            configs[key] = {
                'ticker': ticker,
                'fold': fold,
                'experiment': row['experiment'],
                'tpsl_mode': row['tpsl_mode'],
                'prob_thresh': _safe_float(row['prob_thresh']),
                'tp_pct': _safe_float(row['tp_pct']),
                'sl_pct': _safe_float(row['sl_pct']),
                'sl_mult': _safe_float(row['sl_mult']),
                'grace_minutes': _safe_int(row['grace_minutes']),
                'val_pnl': _safe_float(row['val_a_avg_pnl']),
                'val_b_pnl': _safe_float(row['val_b_avg_pnl']),
            }
    return configs


def best_config_from_folds(ticker, fold_list, bq_configs):
    best = None
    for f in fold_list:
        cfg = bq_configs.get((ticker, f))
        if cfg and (best is None or cfg['val_pnl'] > best['val_pnl']):
            best = cfg
    return best


# ==============================================================================
# TRADE SUMMARIZATION
# ==============================================================================
def summarize_trades(trades_list, label):
    if not trades_list:
        return {'label': label, 'n_tickers': 0, 'n': 0, 'total': 0.0,
                'avg': 0.0, 'wr': 0.0, 'dd': 0.0, 'sharpe': 0.0, 'monthly': {}}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    cum = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    monthly = df.groupby('month').agg(
        n=('pnl', 'count'), total=('pnl', 'sum'),
        avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
    ).to_dict('index')
    pnl_arr = df['pnl'].values
    sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(252) if len(pnl_arr) > 1 and pnl_arr.std() > 0 else 0.0
    return {
        'label': label,
        'n_tickers': df['ticker'].nunique(),
        'n': len(df), 'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': max_dd, 'sharpe': sharpe,
        'monthly': monthly,
        'trades_df': df,
    }


# ==============================================================================
# STRATEGY RUNNER — core simulation loop
# ==============================================================================
def run_strategy(strategy_fn, label, ticker_data, daily_probs, bq_configs):
    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt   = pd.Timestamp(TEST_END).date()
    all_trades = []

    for ticker in sorted(ticker_data.keys()):
        result = strategy_fn(ticker, daily_probs, bq_configs)
        if result is None:
            continue
        trade_dates, cfg = result
        if not trade_dates or cfg is None:
            continue

        ohlcv = ticker_data[ticker]['ohlcv']
        if ohlcv is None:
            continue

        entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]
        last_traded_date = -1

        for idx in entry_indices:
            d_int = int(ohlcv['dates'][idx])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()

            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            if d_int == last_traded_date:
                continue
            if d_date not in trade_dates:
                continue

            pnl, trade_date, filled = execute_trade(ohlcv, idx, cfg)
            if not filled:
                continue
            last_traded_date = d_int

            all_trades.append({
                'date': d_date, 'ticker': ticker, 'pnl': pnl,
                'mode': cfg['tpsl_mode'], 'tp': cfg['tp_pct'],
                'sl': cfg['sl_pct'] if cfg['tpsl_mode'] != 'atr_sl' else cfg['sl_mult'],
                'source_fold': cfg.get('fold', -1),
            })

    return summarize_trades(all_trades, label)


# ==============================================================================
# STRATEGY DEFINITIONS
# ==============================================================================

# ── A. Single-fold strategies ─────────────────────────────────────────────────
def make_single_fold_strategy(fold):
    def strategy(ticker, daily_probs, bq_configs):
        cfg = bq_configs.get((ticker, fold))
        if cfg is None:
            return None
        probs = daily_probs.get(ticker, {}).get(fold)
        if not probs:
            return None
        trade_dates = {d for d, p in probs.items() if p >= cfg['prob_thresh']}
        return trade_dates, cfg
    return strategy


# ── B. Fold-group ensemble strategies ─────────────────────────────────────────
def make_ensemble_strategy(fold_list, label):
    def strategy(ticker, daily_probs, bq_configs):
        cfg = best_config_from_folds(ticker, fold_list, bq_configs)
        if cfg is None:
            return None
        available = []
        for f in fold_list:
            fp = daily_probs.get(ticker, {}).get(f)
            if fp:
                available.append(fp)
        if not available:
            return None
        all_dates = set()
        for fp in available:
            all_dates |= set(fp.keys())
        avg_probs = {}
        for d in all_dates:
            vals = [fp[d] for fp in available if d in fp]
            if vals:
                avg_probs[d] = np.mean(vals)
        trade_dates = {d for d, p in avg_probs.items() if p >= cfg['prob_thresh']}
        return trade_dates, cfg
    return strategy


def make_best_k_strategy(k):
    def strategy(ticker, daily_probs, bq_configs):
        fold_vals = []
        for f in ALL_FOLDS:
            cfg = bq_configs.get((ticker, f))
            if cfg and daily_probs.get(ticker, {}).get(f):
                fold_vals.append((cfg['val_pnl'], f))
        if len(fold_vals) < k:
            return None
        fold_vals.sort(reverse=True)
        best_folds = [f for _, f in fold_vals[:k]]
        cfg = best_config_from_folds(ticker, best_folds, bq_configs)
        if cfg is None:
            return None
        available = [daily_probs[ticker][f] for f in best_folds if f in daily_probs.get(ticker, {})]
        if not available:
            return None
        all_dates = set()
        for fp in available:
            all_dates |= set(fp.keys())
        avg_probs = {}
        for d in all_dates:
            vals = [fp[d] for fp in available if d in fp]
            if vals:
                avg_probs[d] = np.mean(vals)
        trade_dates = {d for d, p in avg_probs.items() if p >= cfg['prob_thresh']}
        return trade_dates, cfg
    return strategy


# ── C. Voting strategies ──────────────────────────────────────────────────────
def make_vote_strategy(fold_list, min_votes, label):
    def strategy(ticker, daily_probs, bq_configs):
        fold_cfgs = []
        for f in fold_list:
            cfg = bq_configs.get((ticker, f))
            fp = daily_probs.get(ticker, {}).get(f)
            if cfg and fp:
                fold_cfgs.append((f, cfg, fp))
        if len(fold_cfgs) < min_votes:
            return None
        all_dates = set()
        for _, _, fp in fold_cfgs:
            all_dates |= set(fp.keys())
        trade_dates = set()
        for d in all_dates:
            votes = 0
            best_val = -999
            best_cfg = None
            for f, cfg, fp in fold_cfgs:
                if d in fp and fp[d] >= cfg['prob_thresh']:
                    votes += 1
                    if cfg['val_pnl'] > best_val:
                        best_val = cfg['val_pnl']
                        best_cfg = cfg
            if votes >= min_votes:
                trade_dates.add(d)
        if not trade_dates:
            return None
        best_overall = max(fold_cfgs, key=lambda x: x[1]['val_pnl'])
        return trade_dates, best_overall[1]
    return strategy


def cross_alpha_strategy(ticker, daily_probs, bq_configs):
    a2_cfgs = []
    a1_cfgs = []
    for f in FOLDS_A2:
        cfg = bq_configs.get((ticker, f))
        fp = daily_probs.get(ticker, {}).get(f)
        if cfg and fp:
            a2_cfgs.append((f, cfg, fp))
    for f in FOLDS_A1:
        cfg = bq_configs.get((ticker, f))
        fp = daily_probs.get(ticker, {}).get(f)
        if cfg and fp:
            a1_cfgs.append((f, cfg, fp))
    if not a2_cfgs or not a1_cfgs:
        return None
    all_dates = set()
    for _, _, fp in a2_cfgs + a1_cfgs:
        all_dates |= set(fp.keys())
    trade_dates = set()
    for d in all_dates:
        a2_fire = any(fp.get(d, 0) >= cfg['prob_thresh'] for _, cfg, fp in a2_cfgs)
        a1_fire = any(fp.get(d, 0) >= cfg['prob_thresh'] for _, cfg, fp in a1_cfgs)
        if a2_fire and a1_fire:
            trade_dates.add(d)
    if not trade_dates:
        return None
    all_cfgs = a2_cfgs + a1_cfgs
    best = max(all_cfgs, key=lambda x: x[1]['val_pnl'])
    return trade_dates, best[1]


# ── D. Ticker-gated strategies ────────────────────────────────────────────────
def make_gated_strategy(min_folds_val_pos, fold_set=None):
    if fold_set is None:
        fold_set = ALL_FOLDS
    def strategy(ticker, daily_probs, bq_configs):
        n_val_pos = sum(1 for f in fold_set if (ticker, f) in bq_configs)
        if n_val_pos < min_folds_val_pos:
            return None
        cfg = best_config_from_folds(ticker, fold_set, bq_configs)
        if cfg is None:
            return None
        best_fold = cfg['fold']
        probs = daily_probs.get(ticker, {}).get(best_fold)
        if not probs:
            return None
        trade_dates = {d for d, p in probs.items() if p >= cfg['prob_thresh']}
        return trade_dates, cfg
    return strategy


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  COMPREHENSIVE SHORT STRATEGY EVALUATION")
print(f"  TEST: {TEST_START} → {TEST_END}")
print(f"  ENTRY: {ENTRY_MIN} min | EXIT: {EXIT_TIME_MIN} min | SLIPPAGE: {SLIPPAGE_PCT:.2%} each way")
print(f"  FOLDS: {len(ALL_FOLDS)} (α=2.0: {FOLDS_A2}, α=1.0: {FOLDS_A1})")
print(f"{'='*W}\n")

# ── Step 1: Load BQ configs ──────────────────────────────────────────────────
print("Step 1: Loading BQ configs...")
bq_configs = load_all_bq_configs()
all_tickers = sorted(set(cfg['ticker'] for cfg in bq_configs.values()))
print(f"  {len(bq_configs)} ticker-fold configs for {len(all_tickers)} tickers")

fold_coverage = defaultdict(set)
for (t, f) in bq_configs:
    fold_coverage[t].add(f)
for n in [1, 2, 3, 4, 6]:
    ct = sum(1 for t, folds in fold_coverage.items() if len(folds) >= n)
    print(f"  >= {n} folds: {ct} tickers")

# ── Step 2: Load OHLCV ──────────────────────────────────────────────────────
print(f"\nStep 2: Loading OHLCV for {len(all_tickers)} tickers...")
ticker_data = {}
for i, ticker in enumerate(all_tickers):
    df = load_master(ticker)
    if df is None:
        continue
    df = heal_data(df)
    ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
    if ohlcv is None:
        continue
    atr_pct = compute_daily_atr_pct(df)
    ohlcv['atr_pct'] = atr_pct.reindex(df.index).values[:len(ohlcv['prices'])]
    if len(ohlcv['atr_pct']) < len(ohlcv['prices']):
        ohlcv['atr_pct'] = np.pad(ohlcv['atr_pct'],
            (0, len(ohlcv['prices']) - len(ohlcv['atr_pct'])), constant_values=0.05)
    ticker_data[ticker] = {'ohlcv': ohlcv}
    if (i + 1) % 10 == 0:
        print(f"    {i+1}/{len(all_tickers)} loaded")
        _release_memory()

print(f"  {len(ticker_data)} tickers with OHLCV data")

# ── Step 3: Load seed scores for all folds ───────────────────────────────────
print(f"\nStep 3: Loading seed scores for all {len(ALL_FOLDS)} folds...")
daily_probs = defaultdict(dict)
total_loaded = 0

for fold in ALL_FOLDS:
    exp = FOLD_EXPERIMENTS[fold]
    fold_tickers = [t for t in all_tickers if (t, fold) in bq_configs]
    n_loaded = 0
    for ticker in fold_tickers:
        if ticker not in ticker_data:
            continue
        seeds = load_seed_scores(ticker, exp)
        if not seeds:
            continue
        dp = compute_daily_prob(seeds, ENTRY_MIN)
        if dp:
            daily_probs[ticker][fold] = dp
            n_loaded += 1
        del seeds
    total_loaded += n_loaded
    print(f"  Fold {fold} ({exp}): {n_loaded}/{len(fold_tickers)} tickers loaded")
    _release_memory()

print(f"  Total: {total_loaded} ticker-fold probability sets loaded")

# ── Step 4: Define all strategies ────────────────────────────────────────────
print(f"\nStep 4: Running strategies...\n")

strategies = []

# A. Single-fold strategies
for fold in ALL_FOLDS:
    alpha = '2.0' if fold < 410 else '1.0'
    strategies.append((f'FOLD_{fold}', make_single_fold_strategy(fold)))

# B. Fold-group ensembles
strategies.append(('EARLY_A2 (400-402)', make_ensemble_strategy([400,401,402], 'EARLY_A2')))
strategies.append(('LATE_A2 (403-405)',  make_ensemble_strategy([403,404,405], 'LATE_A2')))
strategies.append(('ALL_A2 (400-405)',   make_ensemble_strategy(FOLDS_A2, 'ALL_A2')))
strategies.append(('EARLY_A1 (410-412)', make_ensemble_strategy([410,411,412], 'EARLY_A1')))
strategies.append(('LATE_A1 (413-415)',  make_ensemble_strategy([413,414,415], 'LATE_A1')))
strategies.append(('ALL_A1 (410-415)',   make_ensemble_strategy(FOLDS_A1, 'ALL_A1')))
strategies.append(('ALL_12',            make_ensemble_strategy(ALL_FOLDS, 'ALL_12')))
strategies.append(('BEST_3',            make_best_k_strategy(3)))
strategies.append(('BEST_5',            make_best_k_strategy(5)))

# C. Voting strategies
strategies.append(('VOTE_2of6_A2', make_vote_strategy(FOLDS_A2, 2, 'VOTE_2of6_A2')))
strategies.append(('VOTE_3of6_A2', make_vote_strategy(FOLDS_A2, 3, 'VOTE_3of6_A2')))
strategies.append(('VOTE_2of6_A1', make_vote_strategy(FOLDS_A1, 2, 'VOTE_2of6_A1')))
strategies.append(('VOTE_3of6_A1', make_vote_strategy(FOLDS_A1, 3, 'VOTE_3of6_A1')))
strategies.append(('CROSS_ALPHA',  cross_alpha_strategy))

# D. Ticker-gated (best-fold config, gated by N folds having VAL+)
strategies.append(('GATE_2of12', make_gated_strategy(2)))
strategies.append(('GATE_3of12', make_gated_strategy(3)))
strategies.append(('GATE_4of12', make_gated_strategy(4)))
strategies.append(('GATE_6of12', make_gated_strategy(6)))
strategies.append(('GATE_2of6_A2', make_gated_strategy(2, FOLDS_A2)))
strategies.append(('GATE_3of6_A2', make_gated_strategy(3, FOLDS_A2)))
strategies.append(('GATE_2of6_A1', make_gated_strategy(2, FOLDS_A1)))
strategies.append(('GATE_3of6_A1', make_gated_strategy(3, FOLDS_A1)))

# ── Step 5: Execute all strategies ───────────────────────────────────────────
results = []
for label, fn in strategies:
    r = run_strategy(fn, label, ticker_data, daily_probs, bq_configs)
    results.append(r)
    if r['n'] > 0:
        print(f"  {label:25s}  tickers={r['n_tickers']:3d}  trades={r['n']:4d}  "
              f"avg={r['avg']:+.2%}  WR={r['wr']:.0%}  total={r['total']:+.1%}  "
              f"DD={r['dd']:+.2%}  sharpe={r['sharpe']:.2f}")
    else:
        print(f"  {label:25s}  (no trades)")

# ── Step 6: Summary table sorted by total PnL ───────────────────────────────
print(f"\n{'='*W}")
print(f"  RESULTS — SORTED BY TOTAL PnL")
print(f"{'='*W}")

results_sorted = sorted(results, key=lambda r: r['total'], reverse=True)

print(f"\n  {'Strategy':25s}  {'Tickers':>7}  {'Trades':>6}  {'Avg_PnL':>8}  {'WR':>5}  "
      f"{'Total_PnL':>10}  {'Max_DD':>7}  {'Sharpe':>7}  {'PnL/mo':>7}")
print(f"  {'─'*25}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*5}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*7}")

for r in results_sorted:
    if r['n'] == 0:
        print(f"  {r['label']:25s}  {'—':>7}  {'—':>6}  {'—':>8}  {'—':>5}  {'—':>10}  {'—':>7}  {'—':>7}  {'—':>7}")
        continue
    months = len(r['monthly']) if r['monthly'] else 1
    pnl_mo = r['total'] / months
    print(f"  {r['label']:25s}  {r['n_tickers']:>7}  {r['n']:>6}  {r['avg']:>+7.2%}  {r['wr']:>4.0%}  "
          f"{r['total']:>+9.1%}  {r['dd']:>+6.2%}  {r['sharpe']:>7.2f}  {pnl_mo:>+6.1%}")

# ── Step 7: Top strategies — per-ticker breakdown ────────────────────────────
print(f"\n{'='*W}")
print(f"  TOP 5 STRATEGIES — PER-TICKER BREAKDOWN")
print(f"{'='*W}")

top5 = [r for r in results_sorted if r['n'] > 0][:5]

for r in top5:
    df = r.get('trades_df')
    if df is None or df.empty:
        continue
    print(f"\n  ── {r['label']} ({'─'*60}")
    tk_stats = df.groupby('ticker').agg(
        n=('pnl', 'count'), total=('pnl', 'sum'),
        avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
    ).sort_values('total', ascending=False)

    print(f"  {'Ticker':>8}  {'Trades':>6}  {'Avg_PnL':>8}  {'WR':>5}  {'Total':>8}  {'Source':>8}")
    for t, row in tk_stats.iterrows():
        t_trades = df[df['ticker'] == t]
        src = int(t_trades['source_fold'].mode().iloc[0]) if 'source_fold' in t_trades.columns else '?'
        print(f"  {t:>8}  {int(row['n']):>6}  {row['avg']:>+7.2%}  {row['wr']:>4.0%}  "
              f"{row['total']:>+7.1%}  fold_{src}")

# ── Step 8: Monthly PnL for top 3 ───────────────────────────────────────────
print(f"\n{'='*W}")
print(f"  TOP 3 STRATEGIES — MONTHLY PnL")
print(f"{'='*W}")

top3 = [r for r in results_sorted if r['n'] > 0][:3]

all_months = sorted(set(m for r in top3 for m in r['monthly'].keys()))
header = "  {:25s}".format("Strategy") + "".join(f"  {m:>10}" for m in all_months) + "   TOTAL"
print(f"\n{header}")
print(f"  {'─'*25}" + "  ─────────" * len(all_months) + "  ─────────")

for r in top3:
    row = f"  {r['label']:25s}"
    for m in all_months:
        if m in r['monthly']:
            mi = r['monthly'][m]
            row += f"  {mi['total']:>+9.1%}"
        else:
            row += f"  {'·':>10}"
    row += f"  {r['total']:>+8.1%}"
    print(row)

# ── Step 9: Strategy type comparison ─────────────────────────────────────────
print(f"\n{'='*W}")
print(f"  STRATEGY TYPE COMPARISON — BEST OF EACH CATEGORY")
print(f"{'='*W}\n")

categories = {
    'Single Fold': [r for r in results if r['label'].startswith('FOLD_') and r['n'] > 0],
    'Ensemble':    [r for r in results if any(r['label'].startswith(p) for p in ['EARLY','LATE','ALL_','BEST_']) and r['n'] > 0],
    'Voting':      [r for r in results if any(r['label'].startswith(p) for p in ['VOTE_','CROSS_']) and r['n'] > 0],
    'Gated':       [r for r in results if r['label'].startswith('GATE_') and r['n'] > 0],
}

for cat_name, cat_results in categories.items():
    if not cat_results:
        print(f"  {cat_name}: (no viable strategies)")
        continue
    best = max(cat_results, key=lambda r: r['total'])
    months = len(best['monthly']) if best['monthly'] else 1
    print(f"  {cat_name:15s}  BEST: {best['label']:25s}  "
          f"tickers={best['n_tickers']}  trades={best['n']}  "
          f"avg={best['avg']:+.2%}  WR={best['wr']:.0%}  total={best['total']:+.1%}  "
          f"sharpe={best['sharpe']:.2f}")

print(f"\n{'='*W}")
print(f"  DONE — {len(strategies)} strategies evaluated on TEST {TEST_START} → {TEST_END}")
print(f"{'='*W}")
