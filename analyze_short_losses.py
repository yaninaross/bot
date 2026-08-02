# %% ── SHORT Trade Loss Forensics — C2_WIDE_TP_FILTER ────────────────────────
#
# Runs forensic_trade() on every C2 signal day to trace exactly where
# money is lost: SL triggered prematurely? Signal wrong? TP too tight?
#
# Outputs:
#   /tmp/analyze_short_losses.csv  — per-trade forensic detail
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

OUTPUT_CSV = '/tmp/analyze_short_losses.csv'


# ==============================================================================
# DATA LOADING (copied from oracle_vs_strategies.py)
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
# SHORT TRADE SIMULATION (for "what if" re-simulation)
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


def execute_trade(ohlcv, entry_idx, mode, tp_pct, sl_pct, sl_mult, grace_min):
    prices    = ohlcv['prices']
    highs     = ohlcv['highs']
    lows      = ohlcv['lows']
    date_ints = ohlcv['dates']
    time_mins = ohlcv['times']
    if mode == 'atr_sl':
        atr_arr = ohlcv.get('atr_pct', np.full(len(prices), 0.05))
        pnl, td, filled = run_short_trade_atr_sl(
            prices, highs, lows, date_ints, time_mins,
            entry_idx, tp_pct, float(atr_arr[entry_idx]), sl_mult, EXIT_TIME_MIN)
    elif mode == 'grace':
        pnl, td, filled = run_short_trade_grace(
            prices, highs, lows, date_ints, time_mins,
            entry_idx, tp_pct, sl_pct, EXIT_TIME_MIN, grace_min)
    else:
        pnl, td, filled = run_short_trade_static(
            prices, highs, lows, date_ints, time_mins,
            entry_idx, tp_pct, sl_pct, EXIT_TIME_MIN)
    if filled:
        pnl -= 2 * SLIPPAGE_PCT
    return pnl, td, filled


# ==============================================================================
# FORENSIC TRADE (extended with bounce_bar)
# ==============================================================================
def forensic_trade(ohlcv, entry_idx, tp_pct, sl_pct, mode, sl_mult, grace_min):
    prices = ohlcv['prices']
    highs = ohlcv['highs']
    lows = ohlcv['lows']
    date_ints = ohlcv['dates']
    time_mins = ohlcv['times']
    n = len(prices)
    if entry_idx >= n - 1:
        return None
    entry_price = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    et = time_mins[entry_idx]

    actual_sl = sl_pct
    if mode == 'atr_sl':
        atr_arr = ohlcv.get('atr_pct', np.full(n, 0.05))
        atr = float(atr_arr[entry_idx]) if atr_arr[entry_idx] > 0 else 0.05
        actual_sl = min(max(atr * sl_mult, 0.01), 0.25)

    max_dip = 0.0
    max_bounce = 0.0
    max_bounce_before_dip = 0.0
    running_max_bounce = 0.0
    exit_reason = 'TIME'
    exit_pnl = 0.0
    exit_bar = 0
    dip_bar = 0
    bounce_bar = 0

    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            exit_pnl = (entry_price - prices[k-1]) / entry_price
            exit_reason = 'EOD'
            exit_bar = k - 1 - entry_idx
            break

        bar_low_pct = (entry_price - lows[k]) / entry_price
        bar_high_pct = (highs[k] - entry_price) / entry_price

        if bar_low_pct > max_dip:
            max_dip = bar_low_pct
            max_bounce_before_dip = running_max_bounce
            dip_bar = k - entry_idx
        if bar_high_pct > running_max_bounce:
            running_max_bounce = bar_high_pct
        if bar_high_pct > max_bounce:
            max_bounce = bar_high_pct
            bounce_bar = k - entry_idx

        target_price = entry_price * (1 - tp_pct)
        stop_price = entry_price * (1 + actual_sl)
        sl_active = True
        if mode == 'grace':
            sl_active = (time_mins[k] - et) >= grace_min

        tp_hit = lows[k] <= target_price
        sl_hit = sl_active and (highs[k] >= stop_price)

        if sl_hit and tp_hit:
            exit_reason = 'TP+SL'
            exit_pnl = tp_pct if np.random.random() < 0.5 else -actual_sl
            exit_bar = k - entry_idx
            break
        if sl_hit:
            exit_reason = 'SL'
            exit_pnl = -actual_sl
            exit_bar = k - entry_idx
            break
        if tp_hit:
            exit_reason = 'TP'
            exit_pnl = tp_pct
            exit_bar = k - entry_idx
            break
        if time_mins[k] >= EXIT_TIME_MIN:
            exit_reason = 'TIME'
            exit_pnl = (entry_price - prices[k]) / entry_price
            exit_bar = k - entry_idx
            break
    else:
        exit_pnl = (entry_price - prices[n-1]) / entry_price
        exit_reason = 'LAST'
        exit_bar = n - 1 - entry_idx

    exit_pnl -= 2 * SLIPPAGE_PCT

    return {
        'max_dip': max_dip,
        'max_bounce': max_bounce,
        'bounce_before_dip': max_bounce_before_dip,
        'exit_reason': exit_reason,
        'exit_pnl': exit_pnl,
        'tp_pct': tp_pct,
        'sl_pct': actual_sl,
        'money_left': max_dip - (exit_pnl + 2*SLIPPAGE_PCT) if exit_pnl >= 0 else max_dip,
        'dip_bar': dip_bar,
        'exit_bar': exit_bar,
        'bounce_bar': bounce_bar,
    }


# ==============================================================================
# PHASE 1: LOAD C2_WIDE_TP_FILTER CONFIGS
# ==============================================================================
np.random.seed(42)

print(f"{'='*W}")
print(f"  SHORT TRADE LOSS FORENSICS — C2_WIDE_TP_FILTER")
print(f"  TEST: {TEST_START} → {TEST_END}")
print(f"  Question: Is SL triggering prematurely on correct signals?")
print(f"{'='*W}")

print(f"\n  PHASE 1: Loading C2 configs from BQ...")
mv_str = ", ".join(f"'{FOLD_MODEL_VERSIONS[f]}'" for f in ALL_FOLDS)
modes_str = ", ".join(f"'{m}'" for m in BQ_ALLOWED_MODES)

q = f"""
SELECT ticker, experiment, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, sl_mult,
       grace_minutes,
       opt_avg_pnl, opt_win_rate,
       val_a_avg_pnl, val_a_win_rate,
       is_ensemble, model_version,
       vol_filter_pct, pre_range_filter_pct
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

print(f"    BQ: {len(bq_df)} ensemble rows, {bq_df['ticker'].nunique()} tickers")

# C2: filter TP >= 10%, pick best val_a per ticker
c2_configs = {}
for ticker in sorted(bq_df['ticker'].unique()):
    t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['tp_pct'] >= 0.10)]
    if len(t_df) == 0:
        continue
    best_idx = t_df['val_a_avg_pnl'].idxmax()
    row = bq_df.loc[best_idx]
    if row['val_a_avg_pnl'] <= 0:
        continue
    c2_configs[ticker] = {
        'fold': int(row['fold']),
        'tpsl_mode': row['tpsl_mode'],
        'prob_thresh': row['prob_thresh'],
        'tp_pct': row['tp_pct'],
        'sl_pct': row['sl_pct'],
        'sl_mult': row['sl_mult'],
        'grace_minutes': int(row['grace_minutes']),
        'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else ENTRY_MIN,
    }

print(f"    C2_WIDE_TP_FILTER: {len(c2_configs)} tickers selected")
print(f"\n    {'Ticker':<8} │ {'Fold':>4} {'Mode':<7} {'TP%':>5} {'SL%':>5} {'Thresh':>6}")
print(f"    {'─'*8}─┼─{'─'*4}─{'─'*7}─{'─'*5}─{'─'*5}─{'─'*6}")
for ticker, cfg in sorted(c2_configs.items()):
    print(f"    {ticker:<8} │ {cfg['fold']:>4} {cfg['tpsl_mode']:<7} "
          f"{cfg['tp_pct']*100:>4.1f} {cfg['sl_pct']*100:>4.1f} {cfg['prob_thresh']:>5.3f}")

# Load OHLCV
print(f"\n    Loading OHLCV...")
ticker_data = {}
for i, ticker in enumerate(sorted(c2_configs.keys())):
    df = load_master(ticker)
    if df is None:
        continue
    df = heal_data(df)
    ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
    if ohlcv is None:
        del df
        _master_cache.pop(ticker, None)
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
    ticker_data[ticker] = ohlcv
    del df
    _master_cache.pop(ticker, None)
_release_memory()
print(f"    OHLCV: {len(ticker_data)} tickers")

# Load seed scores
print(f"\n    Loading seed scores...")
daily_probs = {}
needed_folds = set(cfg['fold'] for cfg in c2_configs.values())
for fold in sorted(needed_folds):
    exp = FOLD_EXPERIMENTS[fold]
    fold_tickers = [t for t, cfg in c2_configs.items()
                    if cfg['fold'] == fold and t in ticker_data]
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
print(f"    Total: {len(daily_probs)} tickers with probabilities")


# ==============================================================================
# PHASE 2: RUN FORENSIC TRADE ON EVERY SIGNAL DAY
# ==============================================================================
print(f"\n  PHASE 2: Running forensic analysis on every signal day...")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

all_forensics = []
for ticker, cfg in sorted(c2_configs.items()):
    if ticker not in ticker_data or ticker not in daily_probs:
        continue
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
        if prob_val < cfg['prob_thresh']:
            continue

        result = forensic_trade(
            ohlcv, idx, cfg['tp_pct'], cfg['sl_pct'],
            cfg['tpsl_mode'], cfg['sl_mult'], cfg['grace_minutes'])

        if result is not None:
            result['ticker'] = ticker
            result['date'] = d_date
            result['prob'] = prob_val
            result['mode'] = cfg['tpsl_mode']
            result['entry_idx'] = idx
            result['entry_price'] = float(ohlcv['prices'][idx])
            result['is_h2'] = d_date >= h2_start_dt
            all_forensics.append(result)
            last_td = d_int

print(f"    {len(all_forensics)} trades analyzed")

fdf = pd.DataFrame(all_forensics)
if len(fdf) == 0:
    print("    NO TRADES — exiting")
    sys.exit(0)

n_total = len(fdf)
n_h2 = fdf['is_h2'].sum()
print(f"    Full period: {n_total} trades | H2: {n_h2} trades")
print(f"    Tickers with trades: {fdf['ticker'].nunique()}")


# ==============================================================================
# PHASE 3A: EXIT REASON BREAKDOWN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  3A: EXIT REASON BREAKDOWN")
print(f"{'='*W}")

print(f"\n  {'Reason':<8} │ {'Count':>5} {'%':>5} │ {'Avg PnL':>8} {'Avg MaxDip':>10} {'Avg Bounce':>10} │ {'Avg Left':>8}")
print(f"  {'─'*8}─┼─{'─'*5}─{'─'*5}─┼─{'─'*8}─{'─'*10}─{'─'*10}─┼─{'─'*8}")

for reason in ['SL', 'TP', 'TIME', 'EOD', 'TP+SL', 'LAST']:
    subset = fdf[fdf['exit_reason'] == reason]
    if len(subset) == 0:
        continue
    n = len(subset)
    pct = n / n_total * 100
    avg_pnl = subset['exit_pnl'].mean() * 100
    avg_dip = subset['max_dip'].mean() * 100
    avg_bnc = subset['max_bounce'].mean() * 100
    avg_left = subset['money_left'].mean() * 100
    print(f"  {reason:<8} │ {n:>5} {pct:>4.0f}% │ {avg_pnl:>+7.2f}% {avg_dip:>9.2f}% {avg_bnc:>9.2f}% │ {avg_left:>7.2f}%")

print(f"\n  TOTAL    │ {n_total:>5}       │ {fdf['exit_pnl'].mean()*100:>+7.2f}% "
      f"{fdf['max_dip'].mean()*100:>9.2f}% {fdf['max_bounce'].mean()*100:>9.2f}% │ "
      f"{fdf['money_left'].mean()*100:>7.2f}%")


# ==============================================================================
# PHASE 3B: SL TRADE CLASSIFICATION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  3B: SL TRADE CLASSIFICATION — Was the signal right?")
print(f"{'='*W}")

sl_trades = fdf[fdf['exit_reason'].isin(['SL', 'TP+SL'])].copy()
n_sl = len(sl_trades)
print(f"\n  {n_sl} SL-triggered trades out of {n_total} total ({n_sl/n_total*100:.0f}%)")

if n_sl > 0:
    sl_trades['sl_class'] = 'SL_CORRECT'
    sl_trades.loc[sl_trades['max_dip'] > sl_trades['sl_pct'], 'sl_class'] = 'SL_PREMATURE'
    sl_trades.loc[sl_trades['max_dip'] > sl_trades['tp_pct'], 'sl_class'] = 'SL_MISSED_TP'

    print(f"\n  Classification:")
    print(f"  {'Class':<15} │ {'Count':>5} {'%':>5} │ {'Avg MaxDip':>10} {'Avg Bounce':>10} {'Avg SL Used':>10} │ Meaning")
    print(f"  {'─'*15}─┼─{'─'*5}─{'─'*5}─┼─{'─'*10}─{'─'*10}─{'─'*10}─┼─{'─'*40}")

    for cls, meaning in [
        ('SL_CORRECT',   'Signal wrong — price never dropped enough'),
        ('SL_PREMATURE', 'Signal right — SL too tight, price dropped past SL'),
        ('SL_MISSED_TP', 'Would have been a WINNER — price dropped past TP!'),
    ]:
        subset = sl_trades[sl_trades['sl_class'] == cls]
        n = len(subset)
        if n == 0:
            continue
        pct = n / n_sl * 100
        avg_dip = subset['max_dip'].mean() * 100
        avg_bnc = subset['max_bounce'].mean() * 100
        avg_sl = subset['sl_pct'].mean() * 100
        print(f"  {cls:<15} │ {n:>5} {pct:>4.0f}% │ {avg_dip:>9.2f}% {avg_bnc:>9.2f}% {avg_sl:>9.2f}% │ {meaning}")

    missed_tp = sl_trades[sl_trades['sl_class'] == 'SL_MISSED_TP']
    if len(missed_tp) > 0:
        print(f"\n  ══ SL_MISSED_TP — These {len(missed_tp)} trades WOULD have hit TP with wider SL ══")
        print(f"     Avg max_dip:    {missed_tp['max_dip'].mean()*100:>+6.2f}%")
        print(f"     Avg TP config:  {missed_tp['tp_pct'].mean()*100:>6.2f}%")
        print(f"     Avg SL config:  {missed_tp['sl_pct'].mean()*100:>6.2f}%")
        print(f"     Avg bounce:     {missed_tp['max_bounce'].mean()*100:>6.2f}%")
        print(f"     Avg bnc before dip: {missed_tp['bounce_before_dip'].mean()*100:>6.2f}%")
        print(f"     Min SL needed:  {missed_tp['max_bounce'].quantile(0.50)*100:>6.2f}% (P50)")
        print(f"                     {missed_tp['max_bounce'].quantile(0.75)*100:>6.2f}% (P75)")
        print(f"                     {missed_tp['max_bounce'].quantile(0.90)*100:>6.2f}% (P90)")
        money_lost = missed_tp['tp_pct'].sum()
        print(f"     Total PnL lost: {money_lost*100:>+6.1f}% (these would be TP winners)")


# ==============================================================================
# PHASE 3C: PER-TICKER SL ANALYSIS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  3C: PER-TICKER SL ANALYSIS")
print(f"{'='*W}")

print(f"\n  {'Ticker':<8} │ {'N':>3} {'SL':>3} {'%SL':>4} │ {'Missed':>6} {'%Missed':>7} │ "
      f"{'AvgBnc':>6} {'BncP75':>6} │ {'AvgDip':>6} {'AvgPnL':>6} │ {'MoneyLeft':>9}")
print(f"  {'─'*8}─┼─{'─'*3}─{'─'*3}─{'─'*4}─┼─{'─'*6}─{'─'*7}─┼─{'─'*6}─{'─'*6}─┼─{'─'*6}─{'─'*6}─┼─{'─'*9}")

ticker_rows = []
for ticker, grp in fdf.groupby('ticker'):
    n = len(grp)
    sl_grp = grp[grp['exit_reason'].isin(['SL', 'TP+SL'])]
    n_sl = len(sl_grp)
    pct_sl = n_sl / n * 100 if n > 0 else 0

    missed = sl_grp[sl_grp['max_dip'] > sl_grp['tp_pct']] if n_sl > 0 else pd.DataFrame()
    n_missed = len(missed)
    pct_missed = n_missed / n * 100 if n > 0 else 0

    avg_bnc = sl_grp['max_bounce'].mean() * 100 if n_sl > 0 else 0
    bnc_p75 = sl_grp['max_bounce'].quantile(0.75) * 100 if n_sl > 0 else 0
    avg_dip = grp['max_dip'].mean() * 100
    avg_pnl = grp['exit_pnl'].mean() * 100
    money_left = grp['money_left'].mean() * 100

    ticker_rows.append((ticker, n, n_sl, pct_sl, n_missed, pct_missed,
                         avg_bnc, bnc_p75, avg_dip, avg_pnl, money_left))

ticker_rows.sort(key=lambda x: -x[10])
for (ticker, n, n_sl, pct_sl, n_missed, pct_missed,
     avg_bnc, bnc_p75, avg_dip, avg_pnl, money_left) in ticker_rows:
    print(f"  {ticker:<8} │ {n:>3} {n_sl:>3} {pct_sl:>3.0f}% │ {n_missed:>6} {pct_missed:>6.0f}% │ "
          f"{avg_bnc:>5.1f}% {bnc_p75:>5.1f}% │ {avg_dip:>5.1f}% {avg_pnl:>+5.1f}% │ {money_left:>8.1f}%")


# ==============================================================================
# PHASE 3D: SL THRESHOLD SWEEP
# ==============================================================================
print(f"\n{'='*W}")
print(f"  3D: SL THRESHOLD SWEEP — What SL would maximize PnL?")
print(f"{'='*W}")
print(f"  For each SL%, estimate PnL if we had used that SL instead.")
print(f"  Logic: if max_bounce < test_SL → trade survives → gets TP (if max_dip >= TP) or TIME pnl")

print(f"\n  {'SL%':>5} │ {'Survive':>7} {'%':>5} │ {'Avg PnL':>8} {'WR':>5} {'Total%':>8} │ {'H2_Avg':>7} {'H2_Tot':>8} │ {'vs Current':>10}")
print(f"  {'─'*5}─┼─{'─'*7}─{'─'*5}─┼─{'─'*8}─{'─'*5}─{'─'*8}─┼─{'─'*7}─{'─'*8}─┼─{'─'*10}")

current_avg = fdf['exit_pnl'].mean()
best_sl_pct = 0
best_avg = -999

for test_sl_int in range(3, 21):
    test_sl = test_sl_int / 100.0

    simulated_pnls = []
    h2_pnls = []

    for _, row in fdf.iterrows():
        orig_reason = row['exit_reason']
        orig_pnl = row['exit_pnl']
        tp = row['tp_pct']

        if orig_reason in ['SL', 'TP+SL']:
            if row['max_bounce'] < test_sl:
                if row['max_dip'] >= tp:
                    new_pnl = tp - 2 * SLIPPAGE_PCT
                else:
                    new_pnl = orig_pnl + row['sl_pct'] - test_sl
                    if new_pnl < -test_sl:
                        new_pnl = -test_sl - 2 * SLIPPAGE_PCT
            else:
                new_pnl = -test_sl - 2 * SLIPPAGE_PCT
        elif orig_reason == 'TP':
            new_pnl = orig_pnl
        else:
            new_pnl = orig_pnl

        simulated_pnls.append(new_pnl)
        if row['is_h2']:
            h2_pnls.append(new_pnl)

    pnls = np.array(simulated_pnls)
    h2 = np.array(h2_pnls) if h2_pnls else np.array([0.0])
    n = len(pnls)
    avg = float(pnls.mean())
    wr = float((pnls > 0).mean())
    total = float(pnls.sum())
    h2_avg = float(h2.mean()) if len(h2) > 0 else 0.0
    h2_tot = float(h2.sum()) if len(h2) > 0 else 0.0
    survive = (pnls > -test_sl - 2*SLIPPAGE_PCT - 0.001).sum()

    delta = avg - current_avg
    marker = ' ◄ BEST' if avg > best_avg else ''
    if avg > best_avg:
        best_avg = avg
        best_sl_pct = test_sl

    print(f"  {test_sl_int:>4}% │ {survive:>7} {survive/n*100:>4.0f}% │ "
          f"{avg*100:>+7.2f}% {wr*100:>4.0f}% {total*100:>+7.1f} │ "
          f"{h2_avg*100:>+6.2f} {h2_tot*100:>+7.1f} │ {delta*100:>+9.2f}%{marker}")

print(f"\n  CURRENT SL (mixed): avg PnL = {current_avg*100:>+.2f}%")
print(f"  OPTIMAL SL ({best_sl_pct*100:.0f}%):  avg PnL = {best_avg*100:>+.2f}% (Δ = {(best_avg-current_avg)*100:>+.2f}%)")


# ==============================================================================
# PHASE 3E: TIMING ANALYSIS — bounce vs dip
# ==============================================================================
print(f"\n{'='*W}")
print(f"  3E: TIMING ANALYSIS — When does the bounce/dip happen?")
print(f"{'='*W}")

print(f"\n  All trades (bars = minutes after entry):")
print(f"    Max bounce occurs at: P25={fdf['bounce_bar'].quantile(0.25):.0f} "
      f"P50={fdf['bounce_bar'].quantile(0.50):.0f} "
      f"P75={fdf['bounce_bar'].quantile(0.75):.0f} "
      f"P90={fdf['bounce_bar'].quantile(0.90):.0f} min")
print(f"    Max dip occurs at:    P25={fdf['dip_bar'].quantile(0.25):.0f} "
      f"P50={fdf['dip_bar'].quantile(0.50):.0f} "
      f"P75={fdf['dip_bar'].quantile(0.75):.0f} "
      f"P90={fdf['dip_bar'].quantile(0.90):.0f} min")

bounce_first = (fdf['bounce_bar'] < fdf['dip_bar']).sum()
dip_first = (fdf['bounce_bar'] > fdf['dip_bar']).sum()
same = (fdf['bounce_bar'] == fdf['dip_bar']).sum()
print(f"\n    Bounce comes BEFORE dip: {bounce_first} trades ({bounce_first/n_total*100:.0f}%)")
print(f"    Dip comes BEFORE bounce: {dip_first} trades ({dip_first/n_total*100:.0f}%)")
print(f"    Same bar:                {same} trades ({same/n_total*100:.0f}%)")

sl_only = fdf[fdf['exit_reason'] == 'SL']
if len(sl_only) > 0:
    print(f"\n  SL trades only ({len(sl_only)}):")
    print(f"    Bounce (=SL trigger) at: P25={sl_only['bounce_bar'].quantile(0.25):.0f} "
          f"P50={sl_only['bounce_bar'].quantile(0.50):.0f} "
          f"P75={sl_only['bounce_bar'].quantile(0.75):.0f} min")
    print(f"    Max dip after entry at:  P25={sl_only['dip_bar'].quantile(0.25):.0f} "
          f"P50={sl_only['dip_bar'].quantile(0.50):.0f} "
          f"P75={sl_only['dip_bar'].quantile(0.75):.0f} min")

    early_bounce = (sl_only['exit_bar'] <= 30).sum()
    late_bounce = (sl_only['exit_bar'] > 30).sum()
    print(f"\n    SL triggered within 30 min: {early_bounce} ({early_bounce/len(sl_only)*100:.0f}%)")
    print(f"    SL triggered after 30 min:  {late_bounce} ({late_bounce/len(sl_only)*100:.0f}%)")

    grace_analysis = []
    for grace in [10, 15, 20, 30, 45, 60, 90]:
        would_survive = (sl_only['exit_bar'] > grace).sum()
        saved_missed = sl_only[(sl_only['exit_bar'] <= grace) &
                               (sl_only['max_dip'] > sl_only['tp_pct'])]
        grace_analysis.append((grace, would_survive, len(saved_missed)))

    print(f"\n    Grace period analysis (for SL trades):")
    print(f"    {'Grace':>6} │ {'Would Survive':>13} │ {'Missed TP Saved':>15}")
    print(f"    {'─'*6}─┼─{'─'*13}─┼─{'─'*15}")
    for grace, survive, saved in grace_analysis:
        print(f"    {grace:>5}m │ {survive:>13} │ {saved:>15}")


# ==============================================================================
# PHASE 3F: WORST LOSSES — trades with most money left on table
# ==============================================================================
print(f"\n{'='*W}")
print(f"  3F: WORST LOSSES — Biggest missed opportunities (SL killed correct signals)")
print(f"{'='*W}")

missed = fdf[(fdf['exit_reason'].isin(['SL', 'TP+SL'])) & (fdf['max_dip'] > fdf['tp_pct'])].copy()
missed = missed.sort_values('money_left', ascending=False)

print(f"\n  {len(missed)} SL trades where max_dip > TP (the signal was RIGHT)")
if len(missed) > 0:
    print(f"\n  {'Ticker':<8} {'Date':<12} {'Mode':<7} │ {'TP%':>5} {'SL%':>5} │ "
          f"{'MaxDip':>6} {'Bounce':>6} {'BncB4':>6} │ {'ExitPnL':>7} {'Left':>6} │ {'SL needed':>9}")
    print(f"  {'─'*8}─{'─'*12}─{'─'*7}─┼─{'─'*5}─{'─'*5}─┼─{'─'*6}─{'─'*6}─{'─'*6}─┼─{'─'*7}─{'─'*6}─┼─{'─'*9}")

    for _, row in missed.head(30).iterrows():
        sl_needed = row['max_bounce'] * 100
        print(f"  {row['ticker']:<8} {str(row['date']):<12} {row['mode']:<7} │ "
              f"{row['tp_pct']*100:>4.1f} {row['sl_pct']*100:>4.1f} │ "
              f"{row['max_dip']*100:>5.1f} {row['max_bounce']*100:>5.1f} {row['bounce_before_dip']*100:>5.1f} │ "
              f"{row['exit_pnl']*100:>+6.2f} {row['money_left']*100:>5.1f} │ {sl_needed:>8.1f}%")


# ==============================================================================
# PHASE 3G: "WHAT IF" RE-SIMULATION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  3G: 'WHAT IF' RE-SIMULATION — Using optimal SL + grace mode")
print(f"{'='*W}")

scenarios = [
    ('CURRENT', None, None, None),
    (f'SL={best_sl_pct*100:.0f}% (sweep best)', best_sl_pct, None, None),
    ('SL=15% static', 0.15, 'static', None),
    ('SL=15% grace 30m', 0.15, 'grace', 30),
    ('SL=15% grace 60m', 0.15, 'grace', 60),
    ('SL=20% static', 0.20, 'static', None),
    ('SL=20% grace 30m', 0.20, 'grace', 30),
    ('NO SL (TP + TIME only)', 0.99, 'static', None),
]

print(f"\n  {'Scenario':<30} │ {'Trades':>6} {'Avg%':>7} {'WR':>5} {'Total%':>8} │ "
      f"{'H2_Trd':>6} {'H2_Avg':>7} {'H2_Tot':>8}")
print(f"  {'─'*30}─┼─{'─'*6}─{'─'*7}─{'─'*5}─{'─'*8}─┼─{'─'*6}─{'─'*7}─{'─'*8}")

for scenario_name, override_sl, override_mode, override_grace in scenarios:
    all_pnls = []
    h2_pnls = []

    for ticker, cfg in sorted(c2_configs.items()):
        if ticker not in ticker_data or ticker not in daily_probs:
            continue
        ohlcv = ticker_data[ticker]
        probs = daily_probs[ticker]

        use_sl = override_sl if override_sl is not None else cfg['sl_pct']
        use_mode = override_mode if override_mode is not None else cfg['tpsl_mode']
        use_grace = override_grace if override_grace is not None else cfg['grace_minutes']
        use_tp = cfg['tp_pct']
        use_sl_mult = cfg['sl_mult']

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
            if prob_val < cfg['prob_thresh']:
                continue

            pnl, td, filled = execute_trade(
                ohlcv, idx, use_mode, use_tp, use_sl, use_sl_mult, use_grace)
            if filled:
                all_pnls.append(pnl)
                if d_date >= h2_start_dt:
                    h2_pnls.append(pnl)
                last_td = d_int

    pnls = np.array(all_pnls) if all_pnls else np.array([0.0])
    h2 = np.array(h2_pnls) if h2_pnls else np.array([0.0])

    print(f"  {scenario_name:<30} │ {len(all_pnls):>6} {pnls.mean()*100:>+6.2f} {(pnls>0).mean()*100:>4.0f}% "
          f"{pnls.sum()*100:>+7.1f} │ {len(h2_pnls):>6} {h2.mean()*100:>+6.2f} {h2.sum()*100:>+7.1f}")


# ==============================================================================
# PHASE 4: SUMMARY + RECOMMENDATION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SUMMARY + DIAGNOSIS")
print(f"{'='*W}")

n_sl_total = (fdf['exit_reason'].isin(['SL', 'TP+SL'])).sum()
n_tp_total = (fdf['exit_reason'] == 'TP').sum()
n_time_total = (fdf['exit_reason'].isin(['TIME', 'EOD'])).sum()

sl_correct = fdf[(fdf['exit_reason'].isin(['SL', 'TP+SL'])) & (fdf['max_dip'] < fdf['sl_pct'])]
sl_premature = fdf[(fdf['exit_reason'].isin(['SL', 'TP+SL'])) & (fdf['max_dip'] >= fdf['sl_pct'])]
sl_missed_tp = fdf[(fdf['exit_reason'].isin(['SL', 'TP+SL'])) & (fdf['max_dip'] >= fdf['tp_pct'])]

print(f"\n  {n_total} total trades analyzed (C2_WIDE_TP_FILTER, {fdf['ticker'].nunique()} tickers)")
print(f"  Current avg PnL: {fdf['exit_pnl'].mean()*100:>+.2f}%")
print(f"  Current total PnL: {fdf['exit_pnl'].sum()*100:>+.1f}%")
print(f"\n  Exit distribution:")
print(f"    TP:   {n_tp_total:>4} ({n_tp_total/n_total*100:.0f}%) — signal right, config captured it")
print(f"    SL:   {n_sl_total:>4} ({n_sl_total/n_total*100:.0f}%) — stopped out")
print(f"    TIME: {n_time_total:>4} ({n_time_total/n_total*100:.0f}%) — ran out of time")
print(f"\n  SL breakdown:")
print(f"    SL correct (max_dip < SL):    {len(sl_correct):>4} ({len(sl_correct)/max(n_sl_total,1)*100:.0f}% of SL trades) — signal was wrong")
print(f"    SL premature (dip > SL):      {len(sl_premature):>4} ({len(sl_premature)/max(n_sl_total,1)*100:.0f}% of SL trades) — signal right, SL too tight")
print(f"    SL missed TP (dip > TP):      {len(sl_missed_tp):>4} ({len(sl_missed_tp)/max(n_sl_total,1)*100:.0f}% of SL trades) — would have been WINNERS")

if len(sl_missed_tp) > 0:
    recovered_pnl = sl_missed_tp['tp_pct'].sum() - sl_missed_tp['exit_pnl'].sum()
    print(f"\n  ▸ If we recovered all {len(sl_missed_tp)} missed-TP trades: +{recovered_pnl*100:.1f}% additional PnL")
    new_avg = (fdf['exit_pnl'].sum() + recovered_pnl) / n_total
    print(f"  ▸ New portfolio avg: {new_avg*100:>+.2f}% (from {fdf['exit_pnl'].mean()*100:>+.2f}%)")

print(f"\n  Optimal SL from sweep: {best_sl_pct*100:.0f}%")
print(f"  Avg bounce before dip (all trades): {fdf['bounce_before_dip'].mean()*100:.2f}%")
print(f"  Avg bounce before dip (SL trades):  {sl_trades['bounce_before_dip'].mean()*100:.2f}%" if n_sl > 0 else "")

print(f"\n{'='*W}")
print(f"  DONE")
print(f"{'='*W}")

# Save forensic detail
fdf.to_csv(OUTPUT_CSV, index=False)
print(f"\n  Saved: {OUTPUT_CSV} ({len(fdf)} rows)")
