"""
backtest_naive.py — Model Baseline Edge Diagnostic
===================================================
Tests the model's raw predictive power WITHOUT parameter optimization.

Uses FIXED, conservative parameters:
  - entry_time = 6:20 PST (fixed)
  - prob_thresh = swept [0.30, 0.40, 0.50, 0.60, 0.70] to show signal curve
  - TP = 2× ATR_14 (adaptive to vol, NOT optimized)
  - SL = 1× ATR_14 (adaptive to vol, NOT optimized)
  - No regime filter, no volume filter

Runs on ALL out-of-sample data (Oct 2025 → May 2026) for every experiment-
ticker pair that has seed score files.

Purpose: establish the model's TRUE edge before the WF optimizer touches it.
If naive avg PnL > 1.8%, the model works — the optimizer is the problem.
If naive avg PnL < 1.8%, the model itself needs improvement.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import io
import gc
import pytz
from datetime import timedelta
from google.cloud import storage, bigquery as bq
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(**kwargs):
        def wrapper(fn):
            return fn
        return wrapper

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"
TICKER_FILE_MAP = {'BITF': 'KEEL'}
TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}
PST = pytz.timezone('America/Los_Angeles')

EVAL_START = '2025-10-02'
EVAL_END = '2026-05-21'

ENTRY_MINUTE = 380    # 6:20 PST
EXIT_TIME_MIN = 780   # 13:00 PST

TP_ATR_MULT = 2.0     # TP = 2 × ATR_14
SL_ATR_MULT = 1.0     # SL = 1 × ATR_14
TP_FLOOR = 0.03       # minimum 3% TP
TP_CAP = 0.40         # maximum 40% TP
SL_FLOOR = 0.02       # minimum 2% SL
SL_CAP = 0.20         # maximum 20% SL

PROB_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

W = 130

# ==============================================================================
# EXPERIMENTS — all experiments with seed score files
# ==============================================================================
EXPERIMENTS = [
    '105_DEFENSE', '106_DEFENSE', '152_XSECTOR',
]

# ==============================================================================
# SIMULATION ENGINE (Numba) — ATR-based TP/SL
# ==============================================================================
@jit(nopython=True)
def run_sim_atr_naive(entry_indices, prices, highs, lows, date_ints, time_mins,
                      atr_pct_arr, tp_mult, sl_mult, exit_time,
                      tp_floor, tp_cap, sl_floor, sl_cap):
    n_entries = len(entry_indices)
    n_prices = len(prices)
    trades = np.empty(n_entries, dtype=np.float64)
    tp_used = np.empty(n_entries, dtype=np.float64)
    sl_used = np.empty(n_entries, dtype=np.float64)

    trade_count = 0
    current_entry_ptr = 0
    last_traded_date = -1

    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1:
            break

        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1
            continue

        entry_price = prices[idx]
        atr = atr_pct_arr[idx]
        if atr != atr or atr <= 0:
            atr = 0.05

        tp = min(max(atr * tp_mult, tp_floor), tp_cap)
        sl = min(max(atr * sl_mult, sl_floor), sl_cap)
        target_price = entry_price * (1 + tp)
        stop_price = entry_price * (1 - sl)

        outcome = 0.0
        trade_done = False
        exit_idx = idx

        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k - 1] - entry_price) / entry_price
                exit_idx = k
                trade_done = True
                break
            if lows[k] <= stop_price:
                outcome = -sl
                exit_idx = k
                trade_done = True
                break
            if highs[k] >= target_price:
                outcome = tp
                exit_idx = k
                trade_done = True
                break
            if time_mins[k] >= exit_time:
                outcome = (prices[k] - entry_price) / entry_price
                exit_idx = k
                trade_done = True
                break

        if not trade_done:
            outcome = (prices[n_prices - 1] - entry_price) / entry_price
            exit_idx = n_prices

        trades[trade_count] = outcome
        tp_used[trade_count] = tp
        sl_used[trade_count] = sl
        trade_count += 1
        last_traded_date = entry_date

        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx:
                break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return trades[:trade_count], tp_used[:trade_count], sl_used[:trade_count]


# ==============================================================================
# DATA LOADING
# ==============================================================================
gcs_client = storage.Client()
bq_client = bq.Client()

_ohlcv_cache = {}

def load_ohlcv(ticker):
    if ticker in _ohlcv_cache:
        return _ohlcv_cache[ticker]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{BUCKET_NAME}/Data/Inference_v2/{clean}{suffix}.parquet"
        try:
            df = pd.read_parquet(uri)
            df = df.reset_index()
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if dt_cols:
                df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
            else:
                num_col = next((c for c in df.columns
                               if c in ('Timestamp', 'Date', 'Time')), df.columns[0])
                df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
            df = df.set_index('Datetime').sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df.index = df.index.tz_convert(PST)
            _ohlcv_cache[ticker] = df
            return df
        except Exception:
            continue
    _ohlcv_cache[ticker] = None
    return None


def compute_daily_atr_series(df_ohlcv, period=14):
    df_d = df_ohlcv.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    prev_close = df_d['Close'].shift(1)
    tr = pd.concat([
        df_d['High'] - df_d['Low'],
        (df_d['High'] - prev_close).abs(),
        (df_d['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    atr_pct = (atr / df_d['Close']).fillna(method='ffill')
    atr_1m = atr_pct.reindex(df_ohlcv.index, method='ffill')
    return atr_1m


def load_seed_scores(ticker, experiment):
    clean = TICKER_FILE_MAP.get(ticker.replace("I:", "").replace("X:", ""),
                                ticker.replace("I:", "").replace("X:", ""))
    results_dir = f"Algos/Results_all_v2/Universal_Exp{experiment}"
    prefix = f"{results_dir}/{clean}/Seed_"
    blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=prefix))

    seed_series = {}
    for blob in blobs:
        name = blob.name.split('/')[-1]
        if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
            continue
        seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
        df = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(PST)
        seed_series[seed_id] = df['Prob_Spike'].sort_index()
    return seed_series


# ==============================================================================
# PRECOMPUTE ARRAYS
# ==============================================================================
def precompute_arrays(df_ohlcv, start_date, end_date, atr_pct_series):
    eval_start = pd.to_datetime(start_date).tz_localize(PST)
    eval_end = pd.to_datetime(end_date).tz_localize(PST) + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= eval_start) & (df_ohlcv.index < eval_end)]
    if len(df) == 0:
        return None

    dates = (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64)
    times = (df.index.hour * 60 + df.index.minute).values

    full_indices = np.searchsorted(atr_pct_series.index.asi8, df.index.asi8)
    atr_vals = atr_pct_series.values[np.clip(full_indices, 0, len(atr_pct_series) - 1)]

    return {
        'index': df.index,
        'prices': df['Close'].values.copy(),
        'highs': df['High'].values.copy(),
        'lows': df['Low'].values.copy(),
        'dates': dates,
        'times': times,
        'atr_pct': np.nan_to_num(atr_vals, nan=0.05).astype(np.float64),
        'months': max(1.0, (df.index[-1] - df.index[0]).days / 30.44),
    }


# ==============================================================================
# FIND ALL EXPERIMENT → TICKER PAIRS WITH SEED SCORES
# ==============================================================================
print(f"\n{'='*W}")
print(f"  NAIVE BACKTEST — Model Baseline Edge Diagnostic")
print(f"  Entry: 6:20 PST (fixed) | TP: {TP_ATR_MULT}×ATR | SL: {SL_ATR_MULT}×ATR")
print(f"  Eval period: {EVAL_START} → {EVAL_END}")
print(f"  No regime filter, no volume filter, no parameter optimization")
print(f"{'='*W}")

print("\nScanning for experiment-ticker pairs with seed scores...")

exp_ticker_pairs = []
for exp in EXPERIMENTS:
    prefix = f"Algos/Results_all_v2/Universal_Exp{exp}/"
    tickers_found = set()
    all_blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=prefix))
    for b in all_blobs:
        parts = b.name.replace(prefix, '').split('/')
        if len(parts) >= 2 and parts[1].startswith('Seed_'):
            t = TICKER_RENAME.get(parts[0], parts[0])
            if t not in EXCLUDE_TICKERS:
                tickers_found.add(t)

    for t in sorted(tickers_found):
        exp_ticker_pairs.append((exp, t))
    if tickers_found:
        print(f"  {exp}: {len(tickers_found)} tickers")

print(f"  Found {len(exp_ticker_pairs)} experiment-ticker pairs across {len(EXPERIMENTS)} experiments")

# ==============================================================================
# RUN NAIVE BACKTEST
# ==============================================================================
all_results = []
n_total = len(exp_ticker_pairs)

exp_groups = {}
for exp, ticker in exp_ticker_pairs:
    exp_groups.setdefault(exp, []).append(ticker)

pair_idx = 0
for exp, tickers in exp_groups.items():
    print(f"\n  Experiment {exp}: {len(tickers)} tickers")
    ticker_cache = {}

    for ticker in tickers:
        pair_idx += 1
        if pair_idx % 10 == 0:
            print(f"    [{pair_idx}/{n_total}] {exp}/{ticker}...")

        df_ohlcv = load_ohlcv(ticker)
        if df_ohlcv is None:
            continue

        if ticker not in ticker_cache:
            atr_series = compute_daily_atr_series(df_ohlcv)
            arrays = precompute_arrays(df_ohlcv, EVAL_START, EVAL_END, atr_series)
            ticker_cache[ticker] = arrays
        else:
            arrays = ticker_cache[ticker]

        if arrays is None:
            continue

        seed_scores = load_seed_scores(ticker, exp)
        if not seed_scores:
            continue

        for seed_id, seed_prob in seed_scores.items():
            common = arrays['index'].intersection(seed_prob.index)
            if len(common) == 0:
                continue

            probs = np.zeros(len(arrays['index']), dtype=np.float64)
            common_locs = arrays['index'].get_indexer(common)
            probs[common_locs] = seed_prob.loc[common].values

            for prob_thresh in PROB_THRESHOLDS:
                mask = (probs >= prob_thresh) & (arrays['times'] == ENTRY_MINUTE)
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    continue

                pnl, tp_used, sl_used = run_sim_atr_naive(
                    indices, arrays['prices'], arrays['highs'], arrays['lows'],
                    arrays['dates'], arrays['times'], arrays['atr_pct'],
                    TP_ATR_MULT, SL_ATR_MULT, EXIT_TIME_MIN,
                    TP_FLOOR, TP_CAP, SL_FLOOR, SL_CAP
                )

                if len(pnl) == 0:
                    continue

                all_results.append({
                    'experiment': exp,
                    'ticker': ticker,
                    'seed': seed_id,
                    'prob_thresh': prob_thresh,
                    'n_trades': len(pnl),
                    'avg_pnl': float(np.mean(pnl)),
                    'win_rate': float(np.mean(pnl > 0)),
                    'total_pnl': float(np.sum(pnl)),
                    'avg_tp': float(np.mean(tp_used)),
                    'avg_sl': float(np.mean(sl_used)),
                    'months': arrays['months'],
                })

        del seed_scores; gc.collect()

    _ohlcv_cache.clear()
    del ticker_cache; gc.collect()
    print(f"    {exp} done — {len(all_results)} results so far")

df = pd.DataFrame(all_results)

# ==============================================================================
# RESULTS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 1: AGGREGATE RESULTS BY PROB THRESHOLD")
print(f"{'='*W}")
print(f"  {'Prob':>5} {'Configs':>8} {'Tickers':>8} {'Avg_N':>6} {'Avg_PnL':>9} {'Avg_WR':>7} "
      f"{'Avg_TP':>7} {'Avg_SL':>7} {'Deploy%':>8}")
print(f"  {'─'*(W-2)}")
for pt in PROB_THRESHOLDS:
    sub = df[df['prob_thresh'] == pt]
    if sub.empty:
        continue
    deploy_pct = (sub['avg_pnl'] >= 0.018).mean()
    print(f"  {pt:>5.2f} {len(sub):>8} {sub['ticker'].nunique():>8} {sub['n_trades'].mean():>5.1f} "
          f"{sub['avg_pnl'].mean():>+8.4f} {sub['win_rate'].mean():>6.3f} "
          f"{sub['avg_tp'].mean():>6.1%} {sub['avg_sl'].mean():>6.1%} "
          f"{deploy_pct:>7.1%}")


print(f"\n\n{'='*W}")
print(f"  TABLE 2: BEST SEED PER TICKER (prob>=0.50, highest avg_pnl)")
print(f"{'='*W}")
sub50 = df[df['prob_thresh'] == 0.50].copy()
if not sub50.empty:
    best_per_ticker = sub50.sort_values('avg_pnl', ascending=False).drop_duplicates('ticker', keep='first')
    best_per_ticker = best_per_ticker.sort_values('avg_pnl', ascending=False)

    print(f"  {'Ticker':<7} {'Exp':<16} {'Seed':>6} {'N':>4} {'Avg_PnL':>9} {'WR':>6} "
          f"{'Total':>8} {'TP':>5} {'SL':>5} {'Deploy':>7}")
    print(f"  {'─'*(W-2)}")
    deploy_count = 0
    for _, r in best_per_ticker.iterrows():
        deploy = '***' if r['avg_pnl'] >= 0.018 else ''
        if r['avg_pnl'] >= 0.018:
            deploy_count += 1
        print(f"  {r['ticker']:<7} {r['experiment']:<16} {int(r['seed']):>6} {r['n_trades']:>4} "
              f"{r['avg_pnl']:>+8.4f} {r['win_rate']:>5.3f} {r['total_pnl']:>+7.2%} "
              f"{r['avg_tp']:>4.1%} {r['avg_sl']:>4.1%} {deploy:>7}")

    print(f"\n  Deployable (>= 1.8%): {deploy_count}/{len(best_per_ticker)}")
    if deploy_count > 0:
        deployable = best_per_ticker[best_per_ticker['avg_pnl'] >= 0.018]
        wtd = (deployable['avg_pnl'] * deployable['n_trades']).sum() / deployable['n_trades'].sum()
        print(f"  Trade-weighted avg PnL of deployable: {wtd:+.4f}")
        print(f"  Avg trades per ticker: {deployable['n_trades'].mean():.1f}")


print(f"\n\n{'='*W}")
print(f"  TABLE 3: CONSENSUS VIEW — Per ticker, how many seeds have avg_pnl > 0?")
print(f"{'='*W}")
sub50 = df[df['prob_thresh'] == 0.50].copy()
if not sub50.empty:
    consensus = sub50.groupby('ticker').agg(
        n_seeds=('seed', 'nunique'),
        positive_seeds=('avg_pnl', lambda x: (x > 0).sum()),
        deploy_seeds=('avg_pnl', lambda x: (x >= 0.018).sum()),
        mean_pnl=('avg_pnl', 'mean'),
        mean_wr=('win_rate', 'mean'),
        mean_n=('n_trades', 'mean'),
    ).sort_values('mean_pnl', ascending=False)

    print(f"  {'Ticker':<7} {'Seeds':>6} {'PnL>0':>6} {'>1.8%':>6} {'Mean_PnL':>9} {'Mean_WR':>8} {'Mean_N':>7}")
    print(f"  {'─'*(W-2)}")
    for ticker, r in consensus.iterrows():
        marker = ' ***' if r['deploy_seeds'] >= 8 else ''
        print(f"  {ticker:<7} {r['n_seeds']:>6} {r['positive_seeds']:>6} {r['deploy_seeds']:>6} "
              f"{r['mean_pnl']:>+8.4f} {r['mean_wr']:>7.3f} {r['mean_n']:>6.1f}{marker}")


print(f"\n\n{'='*W}")
print(f"  TABLE 4: EXPERIMENT COMPARISON (prob>=0.50)")
print(f"{'='*W}")
sub50 = df[df['prob_thresh'] == 0.50].copy()
if not sub50.empty:
    exp_summary = sub50.groupby('experiment').agg(
        n_tickers=('ticker', 'nunique'),
        n_configs=('seed', 'count'),
        mean_pnl=('avg_pnl', 'mean'),
        mean_wr=('win_rate', 'mean'),
        mean_n=('n_trades', 'mean'),
        deploy_pct=('avg_pnl', lambda x: (x >= 0.018).mean()),
    ).sort_values('mean_pnl', ascending=False)

    print(f"  {'Experiment':<16} {'Tickers':>8} {'Configs':>8} {'Mean_PnL':>9} {'Mean_WR':>8} "
          f"{'Mean_N':>7} {'Deploy%':>8}")
    print(f"  {'─'*(W-2)}")
    for exp, r in exp_summary.iterrows():
        print(f"  {exp:<16} {r['n_tickers']:>8} {r['n_configs']:>8} {r['mean_pnl']:>+8.4f} "
              f"{r['mean_wr']:>7.3f} {r['mean_n']:>6.1f} {r['deploy_pct']:>7.1%}")

print(f"\n{'='*W}")
print(f"  DONE — {len(all_results)} total configs evaluated")
print(f"{'='*W}")
