# %% ── SHORT: TP Sweep + Thresh Sweep + Asset Dip Analysis ────────────────────
#
# Three questions:
#   1. What TP% maximizes PnL? (sweep 2-15% with SL=3%)
#   2. What prob_thresh maximizes PnL? (sweep 0.50-0.90)
#   3. Do these assets even dip enough for SHORT? (unconditional dip analysis)
#
# Uses ALL BQ ensemble configs, ALL folds, ALL tickers.
# No val_b/TEST in training. Ensemble-only.
#
# GPU NOT required. Colab-ready.

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

TEST_START  = '2026-04-01'
TEST_END    = '2026-07-28'
HALF1_END   = '2026-05-31'
HALF2_START = '2026-06-01'

ENTRY_MIN     = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.002

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

FOLDS_A2 = [400, 401, 402, 403, 404, 405]
FOLDS_A1 = [410, 411, 412, 413, 414, 415]
ALL_FOLDS = FOLDS_A2 + FOLDS_A1

FOLD_EXPERIMENTS    = {f: f'{f}_XSECTOR' for f in ALL_FOLDS}
FOLD_MODEL_VERSIONS = {f: f'robust_wf_v12_short_{f}' for f in ALL_FOLDS}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_ALLOWED_MODES = ('static', 'atr_sl', 'grace')

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120

# Sweep ranges
TP_RANGE = [round(x/100, 2) for x in range(2, 16)]       # 2% to 15%
SL_FIXED = 0.03                                            # optimal from prior analysis
THRESH_RANGE = [round(x/100, 2) for x in range(45, 91, 5)]  # 0.45 to 0.90
DIP_BUCKETS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]       # for asset analysis


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


def execute_trade_fixed(ohlcv, entry_idx, tp_pct, sl_pct):
    prices    = ohlcv['prices']
    highs     = ohlcv['highs']
    lows      = ohlcv['lows']
    date_ints = ohlcv['dates']
    time_mins = ohlcv['times']
    pnl, td, filled = run_short_trade_static(
        prices, highs, lows, date_ints, time_mins,
        entry_idx, tp_pct, sl_pct, EXIT_TIME_MIN)
    if filled:
        pnl -= 2 * SLIPPAGE_PCT
    return pnl, td, filled


# ==============================================================================
# MEASURE MAX DIP FOR A DAY (unconditional — no signal needed)
# ==============================================================================
def measure_day_dip(ohlcv, entry_idx):
    """From entry bar to EOD, measure the max intraday dip (price drop from entry)."""
    prices = ohlcv['prices']
    lows = ohlcv['lows']
    highs = ohlcv['highs']
    date_ints = ohlcv['dates']
    time_mins = ohlcv['times']
    n = len(prices)
    if entry_idx >= n - 1:
        return None

    entry_price = prices[entry_idx]
    ed = int(date_ints[entry_idx])

    max_dip = 0.0
    max_bounce = 0.0
    eod_pnl = 0.0

    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            eod_pnl = (entry_price - prices[k-1]) / entry_price
            break
        bar_low_pct = (entry_price - lows[k]) / entry_price
        bar_high_pct = (highs[k] - entry_price) / entry_price
        if bar_low_pct > max_dip:
            max_dip = bar_low_pct
        if bar_high_pct > max_bounce:
            max_bounce = bar_high_pct
        if time_mins[k] >= EXIT_TIME_MIN:
            eod_pnl = (entry_price - prices[k]) / entry_price
            break
    else:
        eod_pnl = (entry_price - prices[n-1]) / entry_price

    return {
        'max_dip': max_dip,
        'max_bounce': max_bounce,
        'eod_pnl': eod_pnl,
    }


# ==============================================================================
# PHASE 1: LOAD BQ + OHLCV + SCORES FOR ALL TICKERS
# ==============================================================================
np.random.seed(42)

print(f"{'='*W}")
print(f"  SHORT: TP SWEEP + THRESH SWEEP + ASSET DIP ANALYSIS")
print(f"  TEST: {TEST_START} → {TEST_END}")
print(f"  TP range: {TP_RANGE[0]*100:.0f}% – {TP_RANGE[-1]*100:.0f}%  |  SL fixed: {SL_FIXED*100:.0f}%")
print(f"  Thresh range: {THRESH_RANGE[0]:.2f} – {THRESH_RANGE[-1]:.2f}")
print(f"{'='*W}")

# BQ: load ALL ensemble configs (for discovering tickers + best fold per ticker)
print(f"\n  Loading BQ configs...")
mv_str = ", ".join(f"'{FOLD_MODEL_VERSIONS[f]}'" for f in ALL_FOLDS)
modes_str = ", ".join(f"'{m}'" for m in BQ_ALLOWED_MODES)

q = f"""
SELECT ticker, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, sl_mult, grace_minutes,
       opt_avg_pnl, val_a_avg_pnl,
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
            'entry_minute', 'opt_avg_pnl', 'val_a_avg_pnl']:
    bq_df[col] = bq_df[col].apply(lambda x: _safe_float(x, 0.0))

all_tickers = sorted(bq_df['ticker'].unique())
print(f"    BQ: {len(bq_df)} configs, {len(all_tickers)} tickers, {bq_df['fold'].nunique()} folds")

# For TP/Thresh sweep: assign best fold per ticker (by highest val_a across all configs)
best_fold_per_ticker = {}
for ticker in all_tickers:
    t_df = bq_df[bq_df['ticker'] == ticker]
    if len(t_df) == 0:
        continue
    best_row = t_df.loc[t_df['val_a_avg_pnl'].idxmax()]
    best_fold_per_ticker[ticker] = int(best_row['fold'])

# Load OHLCV for all tickers
print(f"\n  Loading OHLCV for {len(all_tickers)} tickers...")
ticker_data = {}
for ticker in all_tickers:
    df = load_master(ticker)
    if df is None:
        continue
    df = heal_data(df)
    ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
    if ohlcv is None:
        del df
        _master_cache.pop(ticker, None)
        continue
    ticker_data[ticker] = ohlcv
    del df
    _master_cache.pop(ticker, None)
_release_memory()
print(f"    OHLCV loaded: {len(ticker_data)} tickers")

# Load seed scores for all tickers (need all folds)
print(f"\n  Loading seed scores...")
daily_probs = {}
needed_folds = set(best_fold_per_ticker.values())
for fold in sorted(needed_folds):
    exp = FOLD_EXPERIMENTS[fold]
    fold_tickers = [t for t, f in best_fold_per_ticker.items()
                    if f == fold and t in ticker_data]
    n_loaded = 0
    for ticker in fold_tickers:
        seeds = load_seed_scores(ticker, exp)
        if not seeds:
            continue
        dp = compute_daily_prob(seeds, ENTRY_MIN)
        if dp:
            daily_probs[ticker] = dp
            n_loaded += 1
        del seeds
    print(f"      Fold {fold}: {n_loaded} tickers")
    _release_memory()

tickers_with_data = sorted(set(ticker_data.keys()) & set(daily_probs.keys()))
print(f"    Total: {len(tickers_with_data)} tickers with OHLCV + probabilities")


# ==============================================================================
# PHASE 2: UNCONDITIONAL ASSET DIP ANALYSIS (the big question)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 2: UNCONDITIONAL ASSET DIP ANALYSIS")
print(f"  For EVERY trading day: how much does the price actually dip from 6:15am?")
print(f"  This answers: do these assets even have SHORT-worthy intraday moves?")
print(f"{'='*W}")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

asset_stats = []
asset_daily_dips = {}  # ticker -> list of (date, max_dip, max_bounce, eod_pnl)

for ticker in sorted(ticker_data.keys()):
    ohlcv = ticker_data[ticker]
    entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]
    last_td = -1
    daily_records = []

    for idx in entry_indices:
        d_int = int(ohlcv['dates'][idx])
        if d_int == last_td:
            continue
        d_str = str(d_int)
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < test_start_dt or d_date > test_end_dt:
            continue

        result = measure_day_dip(ohlcv, idx)
        if result is not None:
            is_signal = False
            prob_val = 0.0
            if ticker in daily_probs:
                prob_val = daily_probs[ticker].get(d_date, 0.0)
                is_signal = prob_val >= 0.50
            daily_records.append({
                'date': d_date,
                'max_dip': result['max_dip'],
                'max_bounce': result['max_bounce'],
                'eod_pnl': result['eod_pnl'],
                'is_signal': is_signal,
                'prob': prob_val,
                'is_h2': d_date >= h2_start_dt,
            })
            last_td = d_int

    if not daily_records:
        continue

    asset_daily_dips[ticker] = daily_records
    dips = np.array([r['max_dip'] for r in daily_records])
    bounces = np.array([r['max_bounce'] for r in daily_records])
    eod_pnls = np.array([r['eod_pnl'] for r in daily_records])
    signal_dips = np.array([r['max_dip'] for r in daily_records if r['is_signal']])
    non_signal_dips = np.array([r['max_dip'] for r in daily_records if not r['is_signal']])

    bucket_counts = {b: (dips >= b).sum() for b in DIP_BUCKETS}

    stat = {
        'ticker': ticker,
        'n_days': len(dips),
        'avg_dip': dips.mean(),
        'med_dip': np.median(dips),
        'p75_dip': np.percentile(dips, 75),
        'p90_dip': np.percentile(dips, 90),
        'max_dip': dips.max(),
        'avg_bounce': bounces.mean(),
        'avg_eod_short_pnl': eod_pnls.mean(),
        'n_signal': len(signal_dips),
        'signal_avg_dip': signal_dips.mean() if len(signal_dips) > 0 else 0,
        'signal_med_dip': np.median(signal_dips) if len(signal_dips) > 0 else 0,
        'nonsignal_avg_dip': non_signal_dips.mean() if len(non_signal_dips) > 0 else 0,
        'signal_lift': (signal_dips.mean() / max(non_signal_dips.mean(), 0.001) - 1.0)
                       if len(signal_dips) > 0 and len(non_signal_dips) > 0 else 0,
    }
    for b in DIP_BUCKETS:
        stat[f'dip_ge_{int(b*100)}pct'] = bucket_counts[b]
        stat[f'dip_ge_{int(b*100)}pct_rate'] = bucket_counts[b] / len(dips)
    asset_stats.append(stat)

adf = pd.DataFrame(asset_stats)
adf = adf.sort_values('avg_dip', ascending=False)

# 2A: Which assets actually dip?
print(f"\n  2A: ASSET DIP RANKINGS (all {len(adf)} tickers, all trading days)")
print(f"\n  {'Ticker':<8} │ {'Days':>4} │ {'AvgDip':>6} {'MedDip':>6} {'P75':>5} {'P90':>5} {'Max':>5} │ "
      f"{'≥3%':>4} {'≥5%':>4} {'≥8%':>4} {'≥10%':>4} │ {'AvgBnc':>6} {'EODshrt':>7}")
print(f"  {'─'*8}─┼─{'─'*4}─┼─{'─'*6}─{'─'*6}─{'─'*5}─{'─'*5}─{'─'*5}─┼─"
      f"{'─'*4}─{'─'*4}─{'─'*4}─{'─'*4}─┼─{'─'*6}─{'─'*7}")

for _, row in adf.iterrows():
    print(f"  {row['ticker']:<8} │ {row['n_days']:>4} │ "
          f"{row['avg_dip']*100:>5.1f}% {row['med_dip']*100:>5.1f}% "
          f"{row['p75_dip']*100:>4.1f} {row['p90_dip']*100:>4.1f} {row['max_dip']*100:>4.1f} │ "
          f"{row['dip_ge_3pct_rate']*100:>3.0f}% {row['dip_ge_5pct_rate']*100:>3.0f}% "
          f"{row['dip_ge_8pct_rate']*100:>3.0f}% {row['dip_ge_10pct_rate']*100:>3.0f}% │ "
          f"{row['avg_bounce']*100:>5.1f}% {row['avg_eod_short_pnl']*100:>+6.2f}%")

# Summary stats
top_dippers = adf[adf['avg_dip'] >= 0.03]
weak_dippers = adf[adf['avg_dip'] < 0.03]
print(f"\n  SUMMARY:")
print(f"    Assets with avg dip >= 3%: {len(top_dippers)} ({', '.join(top_dippers['ticker'].tolist())})")
print(f"    Assets with avg dip < 3%:  {len(weak_dippers)} ({', '.join(weak_dippers['ticker'].tolist())})")
print(f"    Portfolio avg dip: {adf['avg_dip'].mean()*100:.2f}%")
print(f"    Portfolio P90 dip: {adf['p90_dip'].mean()*100:.2f}%")

# 2B: Signal quality — does the model fire on bigger-dip days?
print(f"\n\n  2B: SIGNAL QUALITY — Does the model predict dips correctly?")
print(f"  Comparing max_dip on signal days (prob >= 0.50) vs non-signal days")
print(f"\n  {'Ticker':<8} │ {'Sig Days':>8} {'Non-sig':>7} │ {'Sig AvgDip':>10} {'Non AvgDip':>10} │ "
      f"{'Lift':>6} {'Signal adds edge?':>17}")
print(f"  {'─'*8}─┼─{'─'*8}─{'─'*7}─┼─{'─'*10}─{'─'*10}─┼─{'─'*6}─{'─'*17}")

signal_quality = adf[adf['n_signal'] > 0].sort_values('signal_lift', ascending=False)
n_positive_lift = 0
for _, row in signal_quality.iterrows():
    lift = row['signal_lift']
    edge = "YES ✓" if lift > 0.05 else ("marginal" if lift > 0 else "NO ✗")
    if lift > 0.05:
        n_positive_lift += 1
    print(f"  {row['ticker']:<8} │ {row['n_signal']:>8} {row['n_days'] - row['n_signal']:>7} │ "
          f"{row['signal_avg_dip']*100:>9.2f}% {row['nonsignal_avg_dip']*100:>9.2f}% │ "
          f"{lift*100:>+5.0f}% {edge:>17}")

print(f"\n  Signal has positive lift (>5%): {n_positive_lift}/{len(signal_quality)} tickers")
print(f"  Avg lift across tickers: {signal_quality['signal_lift'].mean()*100:>+.0f}%")


# 2C: Realistic TP achievability per asset
print(f"\n\n  2C: TP ACHIEVABILITY — What % of days can each TP be hit?")
print(f"\n  {'Ticker':<8} │ {'TP=2%':>5} {'TP=3%':>5} {'TP=5%':>5} {'TP=8%':>5} {'TP=10%':>5} {'TP=15%':>5} │ {'Best TP for 30%+ hit rate':>25}")
print(f"  {'─'*8}─┼─{'─'*5}─{'─'*5}─{'─'*5}─{'─'*5}─{'─'*5}─{'─'*5}─┼─{'─'*25}")

for _, row in adf.iterrows():
    ticker = row['ticker']
    if ticker not in asset_daily_dips:
        continue
    dips = np.array([r['max_dip'] for r in asset_daily_dips[ticker]])
    rates = {}
    for tp_val in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        rates[tp_val] = (dips >= tp_val).mean()

    best_tp = 'NONE'
    for tp_val in [0.15, 0.10, 0.08, 0.05, 0.03, 0.02]:
        if rates[tp_val] >= 0.30:
            best_tp = f'{tp_val*100:.0f}%'
            break

    print(f"  {ticker:<8} │ {rates[0.02]*100:>4.0f}% {rates[0.03]*100:>4.0f}% {rates[0.05]*100:>4.0f}% "
          f"{rates[0.08]*100:>4.0f}% {rates[0.10]*100:>4.0f}% {rates[0.15]*100:>4.0f}% │ {best_tp:>25}")


# ==============================================================================
# PHASE 3: TP SWEEP (fixed SL=3%, all tickers, prob_thresh from BQ best config)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 3: TP SWEEP — Fixed SL={SL_FIXED*100:.0f}%, static mode")
print(f"  Uses prob_thresh from BQ best config per ticker")
print(f"{'='*W}")

tp_sweep_results = []

for test_tp in TP_RANGE:
    all_pnls = []
    h2_pnls = []
    trade_tickers = set()

    for ticker in tickers_with_data:
        if ticker not in best_fold_per_ticker:
            continue
        ohlcv = ticker_data[ticker]
        probs = daily_probs[ticker]

        # Use best config's prob_thresh for this ticker
        t_df = bq_df[bq_df['ticker'] == ticker]
        if len(t_df) == 0:
            continue
        best_thresh = t_df.loc[t_df['val_a_avg_pnl'].idxmax(), 'prob_thresh']
        if best_thresh <= 0:
            best_thresh = 0.55

        entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]
        last_td = -1

        for idx in entry_indices:
            d_int = int(ohlcv['dates'][idx])
            if d_int == last_td:
                continue
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            prob_val = probs.get(d_date, 0.0)
            if prob_val < best_thresh:
                continue

            pnl, td, filled = execute_trade_fixed(ohlcv, idx, test_tp, SL_FIXED)
            if filled:
                all_pnls.append(pnl)
                trade_tickers.add(ticker)
                if d_date >= h2_start_dt:
                    h2_pnls.append(pnl)
                last_td = d_int

    pnls = np.array(all_pnls) if all_pnls else np.array([0.0])
    h2 = np.array(h2_pnls) if h2_pnls else np.array([0.0])

    tp_sweep_results.append({
        'tp_pct': test_tp,
        'n_trades': len(all_pnls),
        'n_tickers': len(trade_tickers),
        'avg_pnl': pnls.mean(),
        'wr': (pnls > 0).mean(),
        'total_pnl': pnls.sum(),
        'h2_n': len(h2_pnls),
        'h2_avg': h2.mean() if len(h2_pnls) > 0 else 0,
        'h2_total': h2.sum() if len(h2_pnls) > 0 else 0,
    })

print(f"\n  {'TP%':>5} │ {'Trades':>6} {'Tickers':>7} │ {'Avg PnL':>8} {'WR':>5} {'Total%':>8} │ "
      f"{'H2 Trd':>6} {'H2 Avg':>7} {'H2 Tot':>8} │ {'Deploy?':>7}")
print(f"  {'─'*5}─┼─{'─'*6}─{'─'*7}─┼─{'─'*8}─{'─'*5}─{'─'*8}─┼─{'─'*6}─{'─'*7}─{'─'*8}─┼─{'─'*7}")

best_tp_by_avg = max(tp_sweep_results, key=lambda x: x['avg_pnl'])
for r in tp_sweep_results:
    deploy = "YES" if r['avg_pnl'] >= 0.018 else ""
    marker = " ◄" if r['tp_pct'] == best_tp_by_avg['tp_pct'] else ""
    print(f"  {r['tp_pct']*100:>4.0f}% │ {r['n_trades']:>6} {r['n_tickers']:>7} │ "
          f"{r['avg_pnl']*100:>+7.2f}% {r['wr']*100:>4.0f}% {r['total_pnl']*100:>+7.1f} │ "
          f"{r['h2_n']:>6} {r['h2_avg']*100:>+6.2f} {r['h2_total']*100:>+7.1f} │ {deploy:>7}{marker}")

print(f"\n  Best TP: {best_tp_by_avg['tp_pct']*100:.0f}% → avg PnL {best_tp_by_avg['avg_pnl']*100:>+.2f}%")


# ==============================================================================
# PHASE 4: PROB_THRESH SWEEP (fixed TP=best from sweep, SL=3%)
# ==============================================================================
best_tp = best_tp_by_avg['tp_pct']
print(f"\n\n{'='*W}")
print(f"  PHASE 4: PROB_THRESH SWEEP — TP={best_tp*100:.0f}%, SL={SL_FIXED*100:.0f}%")
print(f"  Testing signal confidence thresholds")
print(f"{'='*W}")

thresh_sweep_results = []

for test_thresh in THRESH_RANGE:
    all_pnls = []
    h2_pnls = []
    trade_tickers = set()
    per_ticker_pnls = defaultdict(list)

    for ticker in tickers_with_data:
        ohlcv = ticker_data[ticker]
        probs = daily_probs[ticker]

        entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]
        last_td = -1

        for idx in entry_indices:
            d_int = int(ohlcv['dates'][idx])
            if d_int == last_td:
                continue
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            prob_val = probs.get(d_date, 0.0)
            if prob_val < test_thresh:
                continue

            pnl, td, filled = execute_trade_fixed(ohlcv, idx, best_tp, SL_FIXED)
            if filled:
                all_pnls.append(pnl)
                trade_tickers.add(ticker)
                per_ticker_pnls[ticker].append(pnl)
                if d_date >= h2_start_dt:
                    h2_pnls.append(pnl)
                last_td = d_int

    pnls = np.array(all_pnls) if all_pnls else np.array([0.0])
    h2 = np.array(h2_pnls) if h2_pnls else np.array([0.0])

    # Per-ticker avg
    ticker_avgs = {t: np.mean(ps) for t, ps in per_ticker_pnls.items()}
    n_deploy = sum(1 for v in ticker_avgs.values() if v >= 0.018)
    avg_ticker_pnl = np.mean(list(ticker_avgs.values())) if ticker_avgs else 0

    thresh_sweep_results.append({
        'thresh': test_thresh,
        'n_trades': len(all_pnls),
        'n_tickers': len(trade_tickers),
        'avg_pnl': pnls.mean(),
        'wr': (pnls > 0).mean(),
        'total_pnl': pnls.sum(),
        'h2_n': len(h2_pnls),
        'h2_avg': h2.mean() if len(h2_pnls) > 0 else 0,
        'n_deploy': n_deploy,
        'avg_ticker_pnl': avg_ticker_pnl,
    })

print(f"\n  {'Thresh':>6} │ {'Trades':>6} {'Tickers':>7} │ {'Avg PnL':>8} {'WR':>5} {'Total%':>8} │ "
      f"{'H2_Avg':>7} │ {'≥1.8%':>5} {'TkrAvg':>7} │ {'Trades/Tkr':>10}")
print(f"  {'─'*6}─┼─{'─'*6}─{'─'*7}─┼─{'─'*8}─{'─'*5}─{'─'*8}─┼─{'─'*7}─┼─{'─'*5}─{'─'*7}─┼─{'─'*10}")

best_thresh_by_avg = max(thresh_sweep_results, key=lambda x: x['avg_pnl'])
for r in thresh_sweep_results:
    trades_per_tkr = r['n_trades'] / max(r['n_tickers'], 1)
    marker = " ◄" if r['thresh'] == best_thresh_by_avg['thresh'] else ""
    print(f"  {r['thresh']:>5.2f} │ {r['n_trades']:>6} {r['n_tickers']:>7} │ "
          f"{r['avg_pnl']*100:>+7.2f}% {r['wr']*100:>4.0f}% {r['total_pnl']*100:>+7.1f} │ "
          f"{r['h2_avg']*100:>+6.2f} │ {r['n_deploy']:>5} {r['avg_ticker_pnl']*100:>+6.2f} │ "
          f"{trades_per_tkr:>9.1f}{marker}")

print(f"\n  Best thresh: {best_thresh_by_avg['thresh']:.2f} → avg PnL {best_thresh_by_avg['avg_pnl']*100:>+.2f}%")


# ==============================================================================
# PHASE 5: TP × THRESH 2D GRID (find the sweet spot)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 5: TP × THRESH 2D GRID — SL={SL_FIXED*100:.0f}% static")
print(f"  Looking for the sweet spot")
print(f"{'='*W}")

# Focused grid around promising region
TP_GRID = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
THRESH_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

grid_results = []

for test_tp in TP_GRID:
    for test_thresh in THRESH_GRID:
        all_pnls = []
        h2_pnls = []
        trade_tickers = set()

        for ticker in tickers_with_data:
            ohlcv = ticker_data[ticker]
            probs = daily_probs[ticker]

            entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]
            last_td = -1

            for idx in entry_indices:
                d_int = int(ohlcv['dates'][idx])
                if d_int == last_td:
                    continue
                d_str = str(d_int)
                d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
                if d_date < test_start_dt or d_date > test_end_dt:
                    continue
                prob_val = probs.get(d_date, 0.0)
                if prob_val < test_thresh:
                    continue

                pnl, td, filled = execute_trade_fixed(ohlcv, idx, test_tp, SL_FIXED)
                if filled:
                    all_pnls.append(pnl)
                    trade_tickers.add(ticker)
                    if d_date >= h2_start_dt:
                        h2_pnls.append(pnl)
                    last_td = d_int

        pnls = np.array(all_pnls) if all_pnls else np.array([0.0])
        h2 = np.array(h2_pnls) if h2_pnls else np.array([0.0])

        grid_results.append({
            'tp': test_tp, 'thresh': test_thresh,
            'n': len(all_pnls), 'avg': pnls.mean(),
            'wr': (pnls > 0).mean(), 'total': pnls.sum(),
            'h2_avg': h2.mean() if len(h2_pnls) > 0 else 0,
            'h2_n': len(h2_pnls),
            'n_tickers': len(trade_tickers),
        })

# Print as heatmap
print(f"\n  AVG PnL (%) — rows=TP, cols=Thresh:")
print(f"  {'TP↓ Th→':>8}", end='')
for th in THRESH_GRID:
    print(f" {th:>7.2f}", end='')
print()
print(f"  {'─'*8}", end='')
for _ in THRESH_GRID:
    print(f" {'─'*7}", end='')
print()

best_grid = max(grid_results, key=lambda x: x['avg'])
for tp in TP_GRID:
    print(f"  {tp*100:>6.0f}%", end=' ')
    for th in THRESH_GRID:
        r = next(x for x in grid_results if x['tp'] == tp and x['thresh'] == th)
        marker = '◄' if r == best_grid else ' '
        print(f" {r['avg']*100:>+5.2f}{marker}", end='')
    print()

print(f"\n  TRADE COUNT — rows=TP, cols=Thresh:")
print(f"  {'TP↓ Th→':>8}", end='')
for th in THRESH_GRID:
    print(f" {th:>7.2f}", end='')
print()
for tp in TP_GRID:
    print(f"  {tp*100:>6.0f}%", end=' ')
    for th in THRESH_GRID:
        r = next(x for x in grid_results if x['tp'] == tp and x['thresh'] == th)
        print(f" {r['n']:>7}", end='')
    print()

print(f"\n  H2 AVG PnL (%) — rows=TP, cols=Thresh:")
print(f"  {'TP↓ Th→':>8}", end='')
for th in THRESH_GRID:
    print(f" {th:>7.2f}", end='')
print()
for tp in TP_GRID:
    print(f"  {tp*100:>6.0f}%", end=' ')
    for th in THRESH_GRID:
        r = next(x for x in grid_results if x['tp'] == tp and x['thresh'] == th)
        print(f" {r['h2_avg']*100:>+5.2f} ", end='')
    print()

print(f"\n  BEST CELL: TP={best_grid['tp']*100:.0f}%, Thresh={best_grid['thresh']:.2f} "
      f"→ avg {best_grid['avg']*100:>+.2f}%, {best_grid['n']} trades, "
      f"H2 avg {best_grid['h2_avg']*100:>+.2f}%")


# ==============================================================================
# PHASE 6: PER-TICKER RESULTS AT BEST TP × THRESH
# ==============================================================================
best_tp_final = best_grid['tp']
best_thresh_final = best_grid['thresh']

print(f"\n\n{'='*W}")
print(f"  PHASE 6: PER-TICKER RESULTS — TP={best_tp_final*100:.0f}% SL={SL_FIXED*100:.0f}% Thresh={best_thresh_final:.2f}")
print(f"{'='*W}")

per_ticker_results = []
for ticker in tickers_with_data:
    ohlcv = ticker_data[ticker]
    probs = daily_probs[ticker]

    all_pnls = []
    h2_pnls = []
    entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]
    last_td = -1

    for idx in entry_indices:
        d_int = int(ohlcv['dates'][idx])
        if d_int == last_td:
            continue
        d_str = str(d_int)
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < test_start_dt or d_date > test_end_dt:
            continue
        prob_val = probs.get(d_date, 0.0)
        if prob_val < best_thresh_final:
            continue

        pnl, td, filled = execute_trade_fixed(ohlcv, idx, best_tp_final, SL_FIXED)
        if filled:
            all_pnls.append(pnl)
            if d_date >= h2_start_dt:
                h2_pnls.append(pnl)
            last_td = d_int

    if not all_pnls:
        continue

    pnls = np.array(all_pnls)
    h2 = np.array(h2_pnls) if h2_pnls else np.array([0.0])

    # Get the unconditional dip stats for context
    dip_stat = next((a for a in asset_stats if a['ticker'] == ticker), None)
    avg_dip = dip_stat['avg_dip'] if dip_stat else 0

    per_ticker_results.append({
        'ticker': ticker,
        'n_trades': len(all_pnls),
        'avg_pnl': pnls.mean(),
        'wr': (pnls > 0).mean(),
        'total_pnl': pnls.sum(),
        'h2_n': len(h2_pnls),
        'h2_avg': h2.mean() if len(h2_pnls) > 0 else 0,
        'avg_dip': avg_dip,
        'deploy': pnls.mean() >= 0.018,
    })

per_ticker_results.sort(key=lambda x: -x['avg_pnl'])

print(f"\n  {'Ticker':<8} │ {'Trades':>6} {'Avg PnL':>8} {'WR':>5} {'Total':>7} │ "
      f"{'H2 Trd':>6} {'H2 Avg':>7} │ {'AvgDip':>6} │ {'Deploy':>6}")
print(f"  {'─'*8}─┼─{'─'*6}─{'─'*8}─{'─'*5}─{'─'*7}─┼─{'─'*6}─{'─'*7}─┼─{'─'*6}─┼─{'─'*6}")

n_deployable = 0
for r in per_ticker_results:
    deploy_str = "YES" if r['deploy'] else ""
    if r['deploy']:
        n_deployable += 1
    print(f"  {r['ticker']:<8} │ {r['n_trades']:>6} {r['avg_pnl']*100:>+7.2f}% {r['wr']*100:>4.0f}% "
          f"{r['total_pnl']*100:>+6.1f} │ {r['h2_n']:>6} {r['h2_avg']*100:>+6.2f} │ "
          f"{r['avg_dip']*100:>5.1f}% │ {deploy_str:>6}")

print(f"\n  Deployable (≥1.8% avg): {n_deployable}/{len(per_ticker_results)} tickers")
if n_deployable > 0:
    deploy_tkrs = [r for r in per_ticker_results if r['deploy']]
    deploy_avg = np.mean([r['avg_pnl'] for r in deploy_tkrs])
    deploy_trades = sum(r['n_trades'] for r in deploy_tkrs)
    print(f"  Deploy portfolio: {deploy_trades} trades, avg {deploy_avg*100:>+.2f}%")
    print(f"  Tickers: {', '.join(r['ticker'] for r in deploy_tkrs)}")


# ==============================================================================
# PHASE 7: CORRELATION — DIP SIZE vs SHORT PnL
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 7: CORRELATION — Does asset dip size predict SHORT profitability?")
print(f"{'='*W}")

# Merge dip stats with per-ticker PnL results
corr_data = []
for r in per_ticker_results:
    dip_stat = next((a for a in asset_stats if a['ticker'] == r['ticker']), None)
    if dip_stat:
        corr_data.append({
            'ticker': r['ticker'],
            'avg_dip': dip_stat['avg_dip'],
            'p90_dip': dip_stat['p90_dip'],
            'signal_avg_dip': dip_stat['signal_avg_dip'],
            'signal_lift': dip_stat['signal_lift'],
            'avg_pnl': r['avg_pnl'],
            'wr': r['wr'],
            'n_trades': r['n_trades'],
        })

if len(corr_data) >= 5:
    cdf = pd.DataFrame(corr_data)
    from scipy import stats as sp_stats

    for x_col, x_label in [('avg_dip', 'Avg Daily Dip'),
                            ('p90_dip', 'P90 Daily Dip'),
                            ('signal_avg_dip', 'Signal-Day Avg Dip'),
                            ('signal_lift', 'Signal Lift')]:
        r_val, p_val = sp_stats.pearsonr(cdf[x_col], cdf['avg_pnl'])
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        print(f"  {x_label:<25} vs Avg PnL: r={r_val:>+.3f}  p={p_val:.3f} {sig}")

    print(f"\n  Interpretation:")
    r_dip, _ = sp_stats.pearsonr(cdf['avg_dip'], cdf['avg_pnl'])
    r_lift, _ = sp_stats.pearsonr(cdf['signal_lift'], cdf['avg_pnl'])
    if r_dip > 0.3:
        print(f"    ✓ STRONG: bigger-dipping assets → better SHORT PnL (r={r_dip:+.2f})")
        print(f"      → Focus on high-dip assets, remove weak dippers")
    elif r_dip > 0.1:
        print(f"    ~ MODERATE: some relationship between dip size and SHORT PnL (r={r_dip:+.2f})")
    else:
        print(f"    ✗ WEAK: dip size doesn't predict SHORT PnL (r={r_dip:+.2f})")
        print(f"      → Asset selection by dip alone won't fix SHORT")

    if r_lift > 0.3:
        print(f"    ✓ Signal lift matters (r={r_lift:+.2f}) — model quality varies by ticker")
    elif r_lift < -0.1:
        print(f"    ✗ Signal lift NEGATIVELY correlated (r={r_lift:+.2f}) — model may be anti-predicting dips")


# ==============================================================================
# PHASE 8: SUMMARY + RECOMMENDATIONS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SUMMARY + RECOMMENDATIONS")
print(f"{'='*W}")

print(f"\n  1. ASSET DIP ANALYSIS:")
print(f"     • {len(adf)} tickers analyzed over {adf['n_days'].mean():.0f} avg trading days")
print(f"     • Portfolio avg dip from entry: {adf['avg_dip'].mean()*100:.2f}%")
print(f"     • Portfolio P90 dip: {adf['p90_dip'].mean()*100:.2f}%")
print(f"     • Assets with avg dip ≥ 5%: {len(adf[adf['avg_dip'] >= 0.05])}")
print(f"     • Assets with avg dip ≥ 3%: {len(adf[adf['avg_dip'] >= 0.03])}")
print(f"     • Assets with avg dip < 2%: {len(adf[adf['avg_dip'] < 0.02])}")

print(f"\n  2. TP SWEEP RESULTS (SL={SL_FIXED*100:.0f}%):")
print(f"     • Best TP: {best_tp_by_avg['tp_pct']*100:.0f}% → avg PnL {best_tp_by_avg['avg_pnl']*100:>+.2f}%")
print(f"     • Hits 1.8% deploy threshold: {'YES' if best_tp_by_avg['avg_pnl'] >= 0.018 else 'NO'}")

print(f"\n  3. THRESH SWEEP RESULTS (TP={best_tp*100:.0f}%, SL={SL_FIXED*100:.0f}%):")
print(f"     • Best thresh: {best_thresh_by_avg['thresh']:.2f} → avg PnL {best_thresh_by_avg['avg_pnl']*100:>+.2f}%")

print(f"\n  4. 2D GRID BEST:")
print(f"     • TP={best_grid['tp']*100:.0f}%, Thresh={best_grid['thresh']:.2f} → avg {best_grid['avg']*100:>+.2f}%")
print(f"     • {best_grid['n']} trades, {best_grid['n_tickers']} tickers, H2 avg {best_grid['h2_avg']*100:>+.2f}%")
if best_grid['avg'] >= 0.018:
    print(f"     ★ HITS 1.8% DEPLOY THRESHOLD!")
else:
    gap = 0.018 - best_grid['avg']
    print(f"     • Gap to deploy: {gap*100:.2f}% avg PnL needed")

print(f"\n  5. DEPLOYABLE TICKERS: {n_deployable}")
if n_deployable > 0:
    for r in per_ticker_results:
        if r['deploy']:
            print(f"     • {r['ticker']}: {r['n_trades']} trades, avg {r['avg_pnl']*100:>+.2f}%, "
                  f"WR {r['wr']*100:.0f}%, avg_dip {r['avg_dip']*100:.1f}%")

print(f"\n{'='*W}")
print(f"  DONE")
print(f"{'='*W}")
