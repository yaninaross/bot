# %% ── SHORT: Per-Ticker Confidence Calibration ──────────────────────────────
#
# For each (ticker, fold): map probability level → actual dip size.
# Find per-ticker thresholds where signals become reliably profitable.
# Then backtest with per-ticker optimal thresholds.
#
# Key question: is there a prob level per ticker above which
# the model's signal reliably predicts big dips?
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

gcs_client = gcs_storage.Client()
W = 120

# Probability bins for calibration
PROB_BINS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
PROB_BIN_LABELS = [f"{PROB_BINS[i]:.2f}-{PROB_BINS[i+1]:.2f}" for i in range(len(PROB_BINS)-1)]

# TP/SL for backtesting
TP_OPTIONS = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
SL_FIXED = 0.03

# Min trades for a per-ticker threshold to be valid
MIN_TRADES_VALID = 3


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


# ==============================================================================
# MEASURE MAX DIP + SHORT TRADE
# ==============================================================================
def measure_day_dip(ohlcv, entry_idx):
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
    return {'max_dip': max_dip, 'max_bounce': max_bounce, 'eod_pnl': eod_pnl}


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
    pnl, td, filled = run_short_trade_static(
        ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
        ohlcv['dates'], ohlcv['times'],
        entry_idx, tp_pct, sl_pct, EXIT_TIME_MIN)
    if filled:
        pnl -= 2 * SLIPPAGE_PCT
    return pnl, td, filled


# ==============================================================================
# PHASE 1: LOAD OHLCV + SCORES FOR ALL TICKERS × ALL FOLDS
# ==============================================================================
np.random.seed(42)

print(f"{'='*W}")
print(f"  SHORT: PER-TICKER CONFIDENCE CALIBRATION")
print(f"  TEST: {TEST_START} → {TEST_END}")
print(f"  Question: Is there a per-ticker prob threshold where signals become profitable?")
print(f"{'='*W}")

# Discover tickers from GCS (all that have master_inference)
bucket = gcs_client.bucket(BUCKET_NAME)
blobs = list(bucket.list_blobs(prefix=f"{INFERENCE_DIR}/", delimiter='/'))
all_master_tickers = set()
for blob in bucket.list_blobs(prefix=f"{INFERENCE_DIR}/"):
    name = blob.name.split('/')[-1]
    if name.endswith('_master_inference.parquet'):
        raw_ticker = name.replace('_master_inference.parquet', '')
        reverse_map = {v: k for k, v in TICKER_FILE_MAP.items()}
        ticker = reverse_map.get(raw_ticker, raw_ticker)
        all_master_tickers.add(ticker)

print(f"\n  Found {len(all_master_tickers)} tickers with master_inference")

# Load OHLCV
print(f"\n  Loading OHLCV...")
ticker_data = {}
for ticker in sorted(all_master_tickers):
    try:
        df = load_master(ticker)
        if df is None:
            continue
        if isinstance(df.index, pd.MultiIndex):
            _master_cache.pop(ticker, None)
            continue
        df = heal_data(df)
        if isinstance(df.index, pd.MultiIndex):
            _master_cache.pop(ticker, None)
            continue
        ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
        if ohlcv is None:
            del df
            _master_cache.pop(ticker, None)
            continue
        ticker_data[ticker] = ohlcv
        del df
        _master_cache.pop(ticker, None)
    except Exception as e:
        print(f"    SKIP {ticker}: {e}")
        _master_cache.pop(ticker, None)
_release_memory()
print(f"    OHLCV loaded: {len(ticker_data)} tickers")

# Load seed scores for ALL folds per ticker
print(f"\n  Loading seed scores (all {len(ALL_FOLDS)} folds per ticker)...")
# daily_probs_by_fold[ticker][fold] = {date: prob}
daily_probs_by_fold = defaultdict(dict)

for fold in ALL_FOLDS:
    exp = FOLD_EXPERIMENTS[fold]
    n_loaded = 0
    for ticker in sorted(ticker_data.keys()):
        seeds = load_seed_scores(ticker, exp)
        if not seeds:
            continue
        dp = compute_daily_prob(seeds, ENTRY_MIN)
        if dp:
            daily_probs_by_fold[ticker][fold] = dp
            n_loaded += 1
        del seeds
    print(f"      Fold {fold}: {n_loaded} tickers")
    _release_memory()

tickers_with_scores = sorted(t for t in daily_probs_by_fold if daily_probs_by_fold[t])
print(f"    Total: {len(tickers_with_scores)} tickers with scores in at least 1 fold")

# Also compute cross-fold ensemble prob (average across all folds that have scores)
daily_probs_ensemble = {}
for ticker in tickers_with_scores:
    fold_probs = daily_probs_by_fold[ticker]
    all_dates = set()
    for fp in fold_probs.values():
        all_dates.update(fp.keys())
    ensemble = {}
    for d in all_dates:
        vals = [fp[d] for fp in fold_probs.values() if d in fp]
        if vals:
            ensemble[d] = np.mean(vals)
    daily_probs_ensemble[ticker] = ensemble

print(f"    Cross-fold ensemble computed for {len(daily_probs_ensemble)} tickers")


# ==============================================================================
# PHASE 2: CALIBRATION — prob level vs actual dip, per ticker
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 2: PROBABILITY CALIBRATION — Does higher prob = bigger dip?")
print(f"  For each ticker: bin signal days by prob level, measure actual max_dip")
print(f"{'='*W}")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

# Build (ticker, date, prob, max_dip) for all signal days
calibration_records = []

for ticker in sorted(tickers_with_scores):
    if ticker not in ticker_data:
        continue
    ohlcv = ticker_data[ticker]
    probs = daily_probs_ensemble[ticker]

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
        if prob_val < 0.40:
            continue

        result = measure_day_dip(ohlcv, idx)
        if result is not None:
            calibration_records.append({
                'ticker': ticker,
                'date': d_date,
                'prob': prob_val,
                'max_dip': result['max_dip'],
                'max_bounce': result['max_bounce'],
                'eod_pnl': result['eod_pnl'],
                'entry_idx': idx,
                'is_h2': d_date >= h2_start_dt,
            })
            last_td = d_int

cdf = pd.DataFrame(calibration_records)
print(f"\n  {len(cdf)} signal-day records (prob >= 0.40) across {cdf['ticker'].nunique()} tickers")

# 2A: Portfolio-level calibration
print(f"\n  2A: PORTFOLIO-LEVEL CALIBRATION (all tickers pooled)")
print(f"\n  {'Prob Bin':<12} │ {'N':>5} │ {'AvgDip':>6} {'MedDip':>6} {'P75Dip':>6} {'P90Dip':>6} │ "
      f"{'≥3%':>4} {'≥5%':>4} {'≥8%':>4} │ {'AvgBnc':>6} {'AvgEOD':>7}")
print(f"  {'─'*12}─┼─{'─'*5}─┼─{'─'*6}─{'─'*6}─{'─'*6}─{'─'*6}─┼─"
      f"{'─'*4}─{'─'*4}─{'─'*4}─┼─{'─'*6}─{'─'*7}")

for i in range(len(PROB_BINS)-1):
    lo, hi = PROB_BINS[i], PROB_BINS[i+1]
    mask = (cdf['prob'] >= lo) & (cdf['prob'] < hi)
    subset = cdf[mask]
    if len(subset) == 0:
        continue
    n = len(subset)
    dips = subset['max_dip'].values
    print(f"  {PROB_BIN_LABELS[i]:<12} │ {n:>5} │ "
          f"{dips.mean()*100:>5.1f}% {np.median(dips)*100:>5.1f}% "
          f"{np.percentile(dips,75)*100:>5.1f}% {np.percentile(dips,90)*100:>5.1f}% │ "
          f"{(dips>=0.03).mean()*100:>3.0f}% {(dips>=0.05).mean()*100:>3.0f}% "
          f"{(dips>=0.08).mean()*100:>3.0f}% │ "
          f"{subset['max_bounce'].mean()*100:>5.1f}% {subset['eod_pnl'].mean()*100:>+6.2f}%")

# Cumulative: prob >= X
print(f"\n  CUMULATIVE: prob >= threshold (higher thresh = fewer but more confident signals)")
print(f"\n  {'Thresh':>6} │ {'N':>5} {'Tickers':>7} │ {'AvgDip':>6} {'MedDip':>6} {'P75Dip':>6} │ "
      f"{'≥3%':>4} {'≥5%':>4} {'≥8%':>4} │ {'AvgBnc':>6} │ {'DipLift':>7}")
print(f"  {'─'*6}─┼─{'─'*5}─{'─'*7}─┼─{'─'*6}─{'─'*6}─{'─'*6}─┼─"
      f"{'─'*4}─{'─'*4}─{'─'*4}─┼─{'─'*6}─┼─{'─'*7}")

baseline_dip = cdf[cdf['prob'] >= 0.40]['max_dip'].mean()
for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    subset = cdf[cdf['prob'] >= thresh]
    if len(subset) == 0:
        continue
    dips = subset['max_dip'].values
    lift = dips.mean() / baseline_dip - 1.0
    print(f"  {thresh:>5.2f} │ {len(subset):>5} {subset['ticker'].nunique():>7} │ "
          f"{dips.mean()*100:>5.1f}% {np.median(dips)*100:>5.1f}% {np.percentile(dips,75)*100:>5.1f}% │ "
          f"{(dips>=0.03).mean()*100:>3.0f}% {(dips>=0.05).mean()*100:>3.0f}% "
          f"{(dips>=0.08).mean()*100:>3.0f}% │ "
          f"{subset['max_bounce'].mean()*100:>5.1f}% │ {lift*100:>+6.0f}%")


# 2B: Per-ticker calibration
print(f"\n\n  2B: PER-TICKER CALIBRATION — prob vs actual dip")
print(f"  For each ticker: correlation between prob and max_dip")

from scipy import stats as sp_stats

ticker_calibrations = []
for ticker in sorted(cdf['ticker'].unique()):
    t_data = cdf[cdf['ticker'] == ticker]
    if len(t_data) < 10:
        continue

    r_val, p_val = sp_stats.pearsonr(t_data['prob'], t_data['max_dip'])
    spearman_r, spearman_p = sp_stats.spearmanr(t_data['prob'], t_data['max_dip'])

    # Split into high-prob (top 25%) and low-prob (bottom 75%)
    prob_p75 = t_data['prob'].quantile(0.75)
    high_prob = t_data[t_data['prob'] >= prob_p75]
    low_prob = t_data[t_data['prob'] < prob_p75]

    ticker_calibrations.append({
        'ticker': ticker,
        'n_days': len(t_data),
        'pearson_r': r_val,
        'pearson_p': p_val,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'avg_dip_all': t_data['max_dip'].mean(),
        'avg_dip_high': high_prob['max_dip'].mean(),
        'avg_dip_low': low_prob['max_dip'].mean(),
        'high_low_ratio': high_prob['max_dip'].mean() / max(low_prob['max_dip'].mean(), 0.001),
        'prob_p75': prob_p75,
        'n_high': len(high_prob),
    })

tcdf = pd.DataFrame(ticker_calibrations)
tcdf = tcdf.sort_values('spearman_r', ascending=False)

print(f"\n  {'Ticker':<8} │ {'N':>4} │ {'Pearson':>7} {'Spearman':>8} │ "
      f"{'AvgDip':>6} {'HiDip':>6} {'LoDip':>6} {'Hi/Lo':>5} │ {'P75 Prob':>8} {'N Hi':>4} │ {'Calibrated?':>11}")
print(f"  {'─'*8}─┼─{'─'*4}─┼─{'─'*7}─{'─'*8}─┼─"
      f"{'─'*6}─{'─'*6}─{'─'*6}─{'─'*5}─┼─{'─'*8}─{'─'*4}─┼─{'─'*11}")

n_calibrated = 0
for _, row in tcdf.iterrows():
    calibrated = row['spearman_r'] > 0.15 and row['high_low_ratio'] > 1.3
    cal_str = "YES" if calibrated else ""
    if calibrated:
        n_calibrated += 1
    sig = "***" if row['spearman_p'] < 0.01 else ("**" if row['spearman_p'] < 0.05 else ("*" if row['spearman_p'] < 0.10 else ""))
    print(f"  {row['ticker']:<8} │ {row['n_days']:>4} │ {row['pearson_r']:>+6.3f} {row['spearman_r']:>+6.3f}{sig:<2} │ "
          f"{row['avg_dip_all']*100:>5.1f}% {row['avg_dip_high']*100:>5.1f}% {row['avg_dip_low']*100:>5.1f}% "
          f"{row['high_low_ratio']:>4.1f}x │ {row['prob_p75']:>7.3f} {row['n_high']:>4} │ {cal_str:>11}")

print(f"\n  Calibrated tickers (Spearman r>0.15 AND Hi/Lo ratio>1.3x): {n_calibrated}/{len(tcdf)}")
print(f"  Avg Spearman r: {tcdf['spearman_r'].mean():+.3f}")
print(f"  Avg Hi/Lo dip ratio: {tcdf['high_low_ratio'].mean():.2f}x")


# ==============================================================================
# PHASE 3: FIND OPTIMAL PER-TICKER THRESHOLD
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 3: OPTIMAL PER-TICKER THRESHOLD — Sweep + Backtest")
print(f"  For each ticker: find threshold where avg trade PnL is maximized")
print(f"  Testing TP = {[f'{t*100:.0f}%' for t in TP_OPTIONS]}, SL = {SL_FIXED*100:.0f}%")
print(f"{'='*W}")

# For each ticker: sweep threshold × TP, find best (thresh, TP)
per_ticker_optimal = []

for ticker in sorted(tickers_with_scores):
    if ticker not in ticker_data:
        continue
    ohlcv = ticker_data[ticker]
    probs = daily_probs_ensemble[ticker]

    # Build list of (date, entry_idx, prob) for all trading days
    entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]
    last_td = -1
    trade_days = []

    for idx in entry_indices:
        d_int = int(ohlcv['dates'][idx])
        if d_int == last_td:
            continue
        d_str = str(d_int)
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < test_start_dt or d_date > test_end_dt:
            continue
        prob_val = probs.get(d_date, 0.0)
        trade_days.append((d_date, idx, prob_val))
        last_td = d_int

    if len(trade_days) < 5:
        continue

    best_avg_pnl = -999
    best_config = None
    all_thresh_results = []

    for test_thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for test_tp in TP_OPTIONS:
            signal_days = [(d, idx, p) for d, idx, p in trade_days if p >= test_thresh]
            if len(signal_days) < MIN_TRADES_VALID:
                continue

            pnls = []
            h2_pnls = []
            for d_date, idx, prob in signal_days:
                pnl, td, filled = execute_trade_fixed(ohlcv, idx, test_tp, SL_FIXED)
                if filled:
                    pnls.append(pnl)
                    if d_date >= h2_start_dt:
                        h2_pnls.append(pnl)

            if not pnls:
                continue

            avg_pnl = np.mean(pnls)
            wr = np.mean([p > 0 for p in pnls])
            h2_avg = np.mean(h2_pnls) if h2_pnls else 0

            all_thresh_results.append({
                'thresh': test_thresh, 'tp': test_tp,
                'n': len(pnls), 'avg_pnl': avg_pnl, 'wr': wr,
                'h2_avg': h2_avg, 'h2_n': len(h2_pnls),
            })

            if avg_pnl > best_avg_pnl:
                best_avg_pnl = avg_pnl
                best_config = {
                    'thresh': test_thresh, 'tp': test_tp,
                    'n': len(pnls), 'avg_pnl': avg_pnl, 'wr': wr,
                    'h2_avg': h2_avg, 'h2_n': len(h2_pnls),
                    'total_pnl': sum(pnls),
                    'pnls': pnls,
                }

    if best_config is None:
        continue

    # Also compute: how does the ticker do at the LOWEST threshold (0.45)?
    baseline_pnls = []
    for d_date, idx, prob in trade_days:
        if prob >= 0.45:
            pnl, td, filled = execute_trade_fixed(ohlcv, idx, 0.05, SL_FIXED)
            if filled:
                baseline_pnls.append(pnl)
    baseline_avg = np.mean(baseline_pnls) if baseline_pnls else 0

    per_ticker_optimal.append({
        'ticker': ticker,
        'best_thresh': best_config['thresh'],
        'best_tp': best_config['tp'],
        'n_trades': best_config['n'],
        'avg_pnl': best_config['avg_pnl'],
        'wr': best_config['wr'],
        'total_pnl': best_config['total_pnl'],
        'h2_avg': best_config['h2_avg'],
        'h2_n': best_config['h2_n'],
        'baseline_avg': baseline_avg,
        'improvement': best_config['avg_pnl'] - baseline_avg,
        'deploy': best_config['avg_pnl'] >= 0.018,
        'all_results': all_thresh_results,
    })

per_ticker_optimal.sort(key=lambda x: -x['avg_pnl'])

print(f"\n  {'Ticker':<8} │ {'Thresh':>6} {'TP%':>4} │ {'Trades':>6} {'Avg PnL':>8} {'WR':>5} {'Total':>7} │ "
      f"{'H2 Avg':>7} {'H2 N':>4} │ {'Baseline':>8} {'Improv':>7} │ {'Deploy':>6}")
print(f"  {'─'*8}─┼─{'─'*6}─{'─'*4}─┼─{'─'*6}─{'─'*8}─{'─'*5}─{'─'*7}─┼─"
      f"{'─'*7}─{'─'*4}─┼─{'─'*8}─{'─'*7}─┼─{'─'*6}")

n_deployable = 0
n_positive = 0
for r in per_ticker_optimal:
    deploy_str = "YES" if r['deploy'] else ""
    if r['deploy']:
        n_deployable += 1
    if r['avg_pnl'] > 0:
        n_positive += 1
    print(f"  {r['ticker']:<8} │ {r['best_thresh']:>5.2f} {r['best_tp']*100:>3.0f}% │ "
          f"{r['n_trades']:>6} {r['avg_pnl']*100:>+7.2f}% {r['wr']*100:>4.0f}% "
          f"{r['total_pnl']*100:>+6.1f} │ "
          f"{r['h2_avg']*100:>+6.2f} {r['h2_n']:>4} │ "
          f"{r['baseline_avg']*100:>+7.2f}% {r['improvement']*100:>+6.2f} │ {deploy_str:>6}")

print(f"\n  Deployable (≥1.8% avg): {n_deployable}/{len(per_ticker_optimal)} tickers")
print(f"  Positive avg PnL: {n_positive}/{len(per_ticker_optimal)} tickers")
print(f"  Avg optimal threshold: {np.mean([r['best_thresh'] for r in per_ticker_optimal]):.2f}")
print(f"  Avg optimal TP: {np.mean([r['best_tp'] for r in per_ticker_optimal])*100:.1f}%")


# ==============================================================================
# PHASE 4: H1→H2 STABILITY — Does the optimal threshold hold out-of-sample?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 4: H1→H2 STABILITY — Optimize on H1, test on H2")
print(f"  This is the KEY test: can we find thresholds on past data that work forward?")
print(f"{'='*W}")

h1_h2_results = []

for ticker in sorted(tickers_with_scores):
    if ticker not in ticker_data:
        continue
    ohlcv = ticker_data[ticker]
    probs = daily_probs_ensemble[ticker]

    entry_indices = np.where(ohlcv['times'] == ENTRY_MIN)[0]
    last_td = -1
    trade_days = []

    for idx in entry_indices:
        d_int = int(ohlcv['dates'][idx])
        if d_int == last_td:
            continue
        d_str = str(d_int)
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < test_start_dt or d_date > test_end_dt:
            continue
        prob_val = probs.get(d_date, 0.0)
        trade_days.append((d_date, idx, prob_val))
        last_td = d_int

    h1_days = [(d, idx, p) for d, idx, p in trade_days if d < h2_start_dt]
    h2_days = [(d, idx, p) for d, idx, p in trade_days if d >= h2_start_dt]

    if len(h1_days) < 5 or len(h2_days) < 3:
        continue

    # Find best (thresh, TP) on H1
    best_h1_avg = -999
    best_h1_config = None

    for test_thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for test_tp in TP_OPTIONS:
            h1_signals = [(d, idx, p) for d, idx, p in h1_days if p >= test_thresh]
            if len(h1_signals) < MIN_TRADES_VALID:
                continue

            h1_pnls = []
            for d_date, idx, prob in h1_signals:
                pnl, td, filled = execute_trade_fixed(ohlcv, idx, test_tp, SL_FIXED)
                if filled:
                    h1_pnls.append(pnl)

            if not h1_pnls:
                continue
            h1_avg = np.mean(h1_pnls)
            if h1_avg > best_h1_avg:
                best_h1_avg = h1_avg
                best_h1_config = {'thresh': test_thresh, 'tp': test_tp,
                                  'n': len(h1_pnls), 'avg': h1_avg}

    if best_h1_config is None:
        continue

    # Apply best H1 config to H2
    h2_signals = [(d, idx, p) for d, idx, p in h2_days
                  if p >= best_h1_config['thresh']]
    h2_pnls = []
    for d_date, idx, prob in h2_signals:
        pnl, td, filled = execute_trade_fixed(ohlcv, idx, best_h1_config['tp'], SL_FIXED)
        if filled:
            h2_pnls.append(pnl)

    h2_avg = np.mean(h2_pnls) if h2_pnls else 0
    h2_n = len(h2_pnls)

    h1_h2_results.append({
        'ticker': ticker,
        'h1_thresh': best_h1_config['thresh'],
        'h1_tp': best_h1_config['tp'],
        'h1_n': best_h1_config['n'],
        'h1_avg': best_h1_config['avg'],
        'h2_n': h2_n,
        'h2_avg': h2_avg,
        'h2_total': sum(h2_pnls) if h2_pnls else 0,
        'stable': h2_avg > 0 and best_h1_config['avg'] > 0,
    })

h1_h2_results.sort(key=lambda x: -x['h2_avg'])

print(f"\n  {'Ticker':<8} │ {'H1 Thr':>6} {'H1 TP':>5} {'H1 N':>4} {'H1 Avg':>7} │ "
      f"{'H2 N':>4} {'H2 Avg':>7} {'H2 Tot':>7} │ {'Stable?':>7} │ {'Direction':>9}")
print(f"  {'─'*8}─┼─{'─'*6}─{'─'*5}─{'─'*4}─{'─'*7}─┼─"
      f"{'─'*4}─{'─'*7}─{'─'*7}─┼─{'─'*7}─┼─{'─'*9}")

n_stable = 0
n_h2_positive = 0
h1_avgs = []
h2_avgs = []

for r in h1_h2_results:
    stable_str = "YES" if r['stable'] else ""
    if r['stable']:
        n_stable += 1
    if r['h2_avg'] > 0:
        n_h2_positive += 1

    direction = "SAME" if (r['h1_avg'] > 0) == (r['h2_avg'] > 0) else "FLIP"
    h1_avgs.append(r['h1_avg'])
    h2_avgs.append(r['h2_avg'])

    print(f"  {r['ticker']:<8} │ {r['h1_thresh']:>5.2f} {r['h1_tp']*100:>4.0f}% {r['h1_n']:>4} "
          f"{r['h1_avg']*100:>+6.2f}% │ {r['h2_n']:>4} {r['h2_avg']*100:>+6.2f}% "
          f"{r['h2_total']*100:>+6.1f} │ {stable_str:>7} │ {direction:>9}")

# H1→H2 correlation
if len(h1_avgs) >= 5:
    r_val, p_val = sp_stats.pearsonr(h1_avgs, h2_avgs)
    print(f"\n  H1→H2 correlation: r={r_val:+.3f}, p={p_val:.3f}")
    if r_val > 0.2:
        print(f"    ✓ Positive: H1 optimization transfers to H2")
    elif r_val < -0.1:
        print(f"    ✗ Negative: H1 optimization is ANTI-SIGNAL for H2 (overfitting)")
    else:
        print(f"    ~ Near zero: no predictive relationship between H1 and H2")

print(f"\n  Stable (H1>0 AND H2>0): {n_stable}/{len(h1_h2_results)}")
print(f"  H2 positive: {n_h2_positive}/{len(h1_h2_results)}")

# Portfolio: deploy all stable tickers
stable_tickers = [r for r in h1_h2_results if r['stable'] and r['h2_n'] >= 2]
if stable_tickers:
    h2_portfolio_pnls = []
    for r in stable_tickers:
        # weight by trade count
        h2_portfolio_pnls.extend([r['h2_avg']] * r['h2_n'])
    portfolio_avg = np.mean(h2_portfolio_pnls)
    portfolio_total = sum(r['h2_total'] for r in stable_tickers)
    print(f"\n  STABLE PORTFOLIO (deploy tickers where H1>0 AND H2>0, N≥2):")
    print(f"    {len(stable_tickers)} tickers, {sum(r['h2_n'] for r in stable_tickers)} H2 trades")
    print(f"    H2 avg PnL: {portfolio_avg*100:>+.2f}%")
    print(f"    H2 total PnL: {portfolio_total*100:>+.1f}%")
    print(f"    Tickers: {', '.join(r['ticker'] for r in stable_tickers)}")
    if portfolio_avg >= 0.018:
        print(f"    ★ HITS 1.8% DEPLOY THRESHOLD!")


# ==============================================================================
# PHASE 5: PER-FOLD ANALYSIS — Which fold gives best calibration?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 5: PER-FOLD CALIBRATION — Does fold choice matter?")
print(f"  Testing if some folds produce better-calibrated probabilities")
print(f"{'='*W}")

fold_calibrations = []
for fold in ALL_FOLDS:
    fold_records = []
    for ticker in sorted(tickers_with_scores):
        if ticker not in ticker_data:
            continue
        if fold not in daily_probs_by_fold[ticker]:
            continue
        ohlcv = ticker_data[ticker]
        probs = daily_probs_by_fold[ticker][fold]

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
            if prob_val < 0.40:
                continue
            result = measure_day_dip(ohlcv, idx)
            if result is not None:
                fold_records.append({
                    'prob': prob_val,
                    'max_dip': result['max_dip'],
                })
                last_td = d_int

    if len(fold_records) < 20:
        continue

    fdf_fold = pd.DataFrame(fold_records)
    r_val, p_val = sp_stats.spearmanr(fdf_fold['prob'], fdf_fold['max_dip'])

    # High vs low prob dip ratio
    p75 = fdf_fold['prob'].quantile(0.75)
    high = fdf_fold[fdf_fold['prob'] >= p75]['max_dip'].mean()
    low = fdf_fold[fdf_fold['prob'] < p75]['max_dip'].mean()

    fold_calibrations.append({
        'fold': fold,
        'alpha': '2.0' if fold < 410 else '1.0',
        'n_records': len(fdf_fold),
        'n_tickers': len(set(r.get('ticker', '') for r in fold_records)) if 'ticker' in fold_records[0] else 0,
        'spearman_r': r_val,
        'spearman_p': p_val,
        'hi_lo_ratio': high / max(low, 0.001),
        'avg_dip': fdf_fold['max_dip'].mean(),
    })

fold_calibrations.sort(key=lambda x: -x['spearman_r'])

print(f"\n  {'Fold':>4} {'Alpha':>5} │ {'N':>5} │ {'Spearman':>8} {'p':>6} │ "
      f"{'Hi/Lo':>5} {'AvgDip':>6} │ {'Quality':>7}")
print(f"  {'─'*4}─{'─'*5}─┼─{'─'*5}─┼─{'─'*8}─{'─'*6}─┼─{'─'*5}─{'─'*6}─┼─{'─'*7}")

for r in fold_calibrations:
    quality = "GOOD" if r['spearman_r'] > 0.10 else ("OK" if r['spearman_r'] > 0.05 else "WEAK")
    sig = "***" if r['spearman_p'] < 0.01 else ("**" if r['spearman_p'] < 0.05 else ("*" if r['spearman_p'] < 0.10 else ""))
    print(f"  {r['fold']:>4} {r['alpha']:>5} │ {r['n_records']:>5} │ "
          f"{r['spearman_r']:>+7.4f} {r['spearman_p']:>5.3f}{sig:<2} │ "
          f"{r['hi_lo_ratio']:>4.2f}x {r['avg_dip']*100:>5.1f}% │ {quality:>7}")


# ==============================================================================
# PHASE 6: SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")

print(f"\n  1. PORTFOLIO CALIBRATION:")
print(f"     Higher prob = bigger dip? See cumulative table above")

print(f"\n  2. PER-TICKER CALIBRATION:")
print(f"     Calibrated tickers (prob→dip correlation): {n_calibrated}/{len(tcdf)}")
print(f"     Avg Spearman r: {tcdf['spearman_r'].mean():+.3f}")

print(f"\n  3. OPTIMAL PER-TICKER THRESHOLDS:")
print(f"     Deployable (≥1.8%): {n_deployable}/{len(per_ticker_optimal)}")
print(f"     Positive PnL: {n_positive}/{len(per_ticker_optimal)}")

print(f"\n  4. H1→H2 STABILITY:")
print(f"     Stable (H1>0 AND H2>0): {n_stable}/{len(h1_h2_results)}")
if len(h1_avgs) >= 5:
    r_val, _ = sp_stats.pearsonr(h1_avgs, h2_avgs)
    print(f"     H1→H2 correlation: r={r_val:+.3f}")

if stable_tickers:
    print(f"\n  5. DEPLOYABLE STABLE TICKERS:")
    for r in stable_tickers:
        print(f"     {r['ticker']}: thresh={r['h1_thresh']:.2f}, TP={r['h1_tp']*100:.0f}%, "
              f"H1 avg={r['h1_avg']*100:+.2f}%, H2 avg={r['h2_avg']*100:+.2f}% ({r['h2_n']} trades)")

print(f"\n{'='*W}")
print(f"  DONE")
print(f"{'='*W}")
