# %% ── SHORT Group Comparison — 3 Ticker Sets × 4 Config Strategies (CSV-Scored) ──
#
# Question: Does a wider ticker set drive higher TOTAL PnL despite lower avg?
#
# Groups:
#   RULE_1   (6):  BE, IREN, OKLO, ONDS, RKLB, WOLF — deploy-ready (p75_dip + calibration_r gate)
#   STABLE_13(13): MVIS, MXL, KTOS, JOBY, MP, LUNR, SMR, FCEL, VOYG, WULF, IREN, ONDS, OKLO — H1>0 AND H2>0
#   ORACLE_10(10): AVAV, BE, MU, MVIS, MSTR, MP, CCJ, WOLF, OKLO, CEG — top val_b positive rate
#
# Strategies:
#   M01_BEST_VA  — best val_a across all folds
#   M06_FOLD403  — best val_a within fold 403
#   FOLD_414     — best val_a within fold 414
#   ORACLE       — best actual H2 config (BIASED ceiling)
#
# CSV-scored, fold-specific probabilities. GPU NOT required.

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
    try: auth.authenticate_user()
    except Exception: pass

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
RESULTS_DIR   = "Algos/Results_all_v2"
INFERENCE_DIR = "Data/Inference_v2"

TEST_START  = '2026-04-01'
TEST_END    = '2026-07-28'
HALF2_START = '2026-06-01'

ENTRY_MIN     = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.002
DEPLOY_FLOOR  = 0.018
SL_FIXED      = 0.03

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

# ── Ticker groups ────────────────────────────────────────────────────────────
RULE_1 = ['BE', 'IREN', 'OKLO', 'ONDS', 'RKLB', 'WOLF']

STABLE_13 = ['MVIS', 'MXL', 'KTOS', 'JOBY', 'MP', 'LUNR', 'SMR',
             'FCEL', 'VOYG', 'WULF', 'IREN', 'ONDS', 'OKLO']

ORACLE_10 = ['AVAV', 'BE', 'MU', 'MVIS', 'MSTR', 'MP', 'CCJ',
             'WOLF', 'OKLO', 'CEG']

GROUPS = {
    'RULE_1':    RULE_1,
    'STABLE_13': STABLE_13,
    'ORACLE_10': ORACLE_10,
}

ALL_TICKERS = sorted(set(RULE_1 + STABLE_13 + ORACLE_10))

FOLDS_A2 = [400, 401, 402, 403, 404, 405]
FOLDS_A1 = [410, 411, 412, 413, 414, 415]
ALL_FOLDS = FOLDS_A2 + FOLDS_A1

FOLD_EXPERIMENTS    = {f: f'{f}_XSECTOR' for f in ALL_FOLDS}
FOLD_MODEL_VERSIONS = {f: f'robust_wf_v12_short_{f}' for f in ALL_FOLDS}

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_ALLOWED_MODES = ('static', 'atr_sl', 'grace')

MIN_H2_TRADES = 3

FIXED_UNIVERSAL_CFG = {
    'fold': None,
    'tpsl_mode': 'static',
    'prob_thresh': 0.50,
    'tp_pct': 0.05,
    'sl_pct': SL_FIXED,
    'sl_mult': 0.0,
    'grace_minutes': 0,
    'entry_minute': ENTRY_MIN,
    'val_a_avg_pnl': 0.0,
    'opt_avg_pnl': 0.0,
    'config_id': 'fixed_universal',
}

STRATEGY_NAMES = ['M01_BEST_VA', 'M06_FOLD403', 'FOLD_414', 'ORACLE']

STRATEGY_COLORS = {
    'M01_BEST_VA': '#1565C0',
    'M06_FOLD403': '#E65100',
    'FOLD_414':    '#2E7D32',
    'ORACLE':      '#9C27B0',
}

GROUP_COLORS = {
    'RULE_1':    '#E91E63',
    'STABLE_13': '#2196F3',
    'ORACLE_10': '#FF9800',
}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120


# ==============================================================================
# DATA LOADING (copied from backtest_short_strategies.py)
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
# CONFIG SELECTION METHODS
# ==============================================================================
def _bq_row_to_cfg(row):
    return {
        'fold': int(row['fold']),
        'tpsl_mode': row['tpsl_mode'],
        'prob_thresh': row['prob_thresh'],
        'tp_pct': row['tp_pct'],
        'sl_pct': row['sl_pct'],
        'sl_mult': row['sl_mult'],
        'grace_minutes': int(row['grace_minutes']),
        'entry_minute': int(row['entry_minute']) if row['entry_minute'] > 0 else ENTRY_MIN,
        'val_a_avg_pnl': row['val_a_avg_pnl'],
        'opt_avg_pnl': row['opt_avg_pnl'],
        'config_id': f"f{int(row['fold'])}_{row['tpsl_mode']}_t{row['prob_thresh']:.2f}_tp{row['tp_pct']:.2f}",
    }


def _pick_best(t_df, col='val_a_avg_pnl'):
    if len(t_df) == 0:
        return None
    best_idx = t_df[col].idxmax()
    return _bq_row_to_cfg(t_df.loc[best_idx])


def select_m01_best_va(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[bq_df['ticker'] == ticker]
        if len(t_df) == 0:
            continue
        cfg = _pick_best(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


def select_single_fold(bq_df, tickers, fold):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'] == fold)]
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker]
        cfg = _pick_best(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ==============================================================================
# BACKTEST + SUMMARY
# ==============================================================================
def run_csv_backtest(configs, daily_probs_cache, ticker_data, label):
    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt   = pd.Timestamp(TEST_END).date()
    h2_start_dt   = pd.Timestamp(HALF2_START).date()

    all_trades = []
    per_ticker_trades = defaultdict(list)

    for ticker in sorted(configs.keys()):
        if ticker not in ticker_data or ticker not in daily_probs_cache:
            continue
        cfg = configs[ticker]
        ohlcv = ticker_data[ticker]
        probs = daily_probs_cache[ticker]
        entry_min = cfg.get('entry_minute', ENTRY_MIN)
        prob_thresh = cfg['prob_thresh']
        mode = cfg['tpsl_mode']

        prices = ohlcv['prices']
        dates  = ohlcv['dates']
        times  = ohlcv['times']

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

            pnl, trade_date, filled = execute_trade(
                ohlcv, idx, mode, cfg['tp_pct'], cfg['sl_pct'],
                cfg['sl_mult'], cfg['grace_minutes'])

            if not filled:
                continue
            last_traded_date = d_int

            half = 'H2' if d_date >= h2_start_dt else 'H1'
            trade = {
                'date': d_date, 'ticker': ticker, 'pnl': pnl,
                'prob': prob, 'close': float(prices[idx]),
                'mode': mode, 'tp': cfg['tp_pct'],
                'sl': cfg['sl_pct'] if mode != 'atr_sl' else cfg['sl_mult'],
                'half': half,
            }
            all_trades.append(trade)
            per_ticker_trades[ticker].append(trade)

    return all_trades, per_ticker_trades


def summarize_trades(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0, 'monthly': {}, 'trades_df': pd.DataFrame(),
                'h1_n': 0, 'h1_avg': 0, 'h1_total': 0,
                'h2_n': 0, 'h2_avg': 0, 'h2_total': 0, 'h2_wr': 0}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    cum = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    h1 = df[df['half'] == 'H1']
    h2 = df[df['half'] == 'H2']
    return {
        'label': label,
        'n': len(df), 'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': max_dd,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
        'trades_df': df,
        'h1_n': len(h1), 'h1_avg': h1['pnl'].mean() if len(h1) > 0 else 0,
        'h1_total': h1['pnl'].sum() if len(h1) > 0 else 0,
        'h2_n': len(h2), 'h2_avg': h2['pnl'].mean() if len(h2) > 0 else 0,
        'h2_total': h2['pnl'].sum() if len(h2) > 0 else 0,
        'h2_wr': (h2['pnl'] > 0).mean() if len(h2) > 0 else 0,
    }


# ==============================================================================
# MAIN
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT GROUP COMPARISON — 3 Ticker Sets x 4 Strategies")
print(f"  Question: Does wider ticker set drive higher TOTAL PnL?")
print(f"  TEST: {TEST_START} -> {TEST_END}  |  H2: {HALF2_START} -> {TEST_END}")
print(f"{'='*W}")
print(f"  RULE_1   ({len(RULE_1):>2}): {', '.join(sorted(RULE_1))}")
print(f"  STABLE_13({len(STABLE_13):>2}): {', '.join(sorted(STABLE_13))}")
print(f"  ORACLE_10({len(ORACLE_10):>2}): {', '.join(sorted(ORACLE_10))}")
print(f"  Union: {len(ALL_TICKERS)} unique tickers")


# ── Step 1: Load BQ configs ──────────────────────────────────────────────────
print(f"\n  STEP 1: Loading BQ configs (all 12 folds, ensemble)...")

mv_str = ", ".join(f"'{FOLD_MODEL_VERSIONS[f]}'" for f in ALL_FOLDS)
modes_str = ", ".join(f"'{m}'" for m in BQ_ALLOWED_MODES)

q = f"""
SELECT ticker, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, sl_mult, grace_minutes,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
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
            'entry_minute', 'opt_avg_pnl', 'opt_win_rate', 'opt_count',
            'val_a_avg_pnl', 'val_a_win_rate', 'val_a_count']:
    bq_df[col] = bq_df[col].apply(lambda x: _safe_float(x, 0.0))

bq_df['va_opt_ratio'] = bq_df['val_a_avg_pnl'] / bq_df['opt_avg_pnl'].clip(lower=0.001)

bq_tickers = sorted(bq_df['ticker'].unique())
print(f"    BQ: {len(bq_df)} configs, {len(bq_tickers)} tickers, {bq_df['fold'].nunique()} folds")

for gname, gtickers in GROUPS.items():
    found = [t for t in gtickers if t in bq_tickers]
    missing = [t for t in gtickers if t not in bq_tickers]
    print(f"    {gname}: {len(found)}/{len(gtickers)} in BQ" +
          (f"  (missing: {', '.join(missing)})" if missing else ""))


# ── Step 2: Load OHLCV data for ALL union tickers ────────────────────────────
print(f"\n  STEP 2: Loading OHLCV + ATR for {len(ALL_TICKERS)} tickers...")
ticker_data = {}

for ticker in ALL_TICKERS:
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
            print(f"    {ticker}: no data in test period")
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
        n_days = len(np.unique(ohlcv['dates']))
        print(f"    {ticker}: {n_days} trading days")
        del df
        _master_cache.pop(ticker, None)
    except Exception as ex:
        print(f"    {ticker}: OHLCV load failed -- {ex}")
        _master_cache.pop(ticker, None)
    gc.collect()

_release_memory()
print(f"    OHLCV ready: {len(ticker_data)} tickers")


# ── Step 3: Load fold-specific CSV scores ────────────────────────────────────
print(f"\n  STEP 3: Loading CSV seed scores (all folds x all tickers)...")

all_fold_probs = defaultdict(dict)
for fold in sorted(ALL_FOLDS):
    exp = FOLD_EXPERIMENTS[fold]
    n_loaded = 0
    for ticker in ALL_TICKERS:
        if ticker not in ticker_data:
            continue
        if fold in all_fold_probs[ticker]:
            continue
        seeds = load_seed_scores(ticker, exp)
        if not seeds:
            continue
        dp = compute_daily_prob(seeds, ENTRY_MIN)
        if dp:
            all_fold_probs[ticker][fold] = dp
            n_loaded += 1
        del seeds
    if n_loaded > 0:
        print(f"    Fold {fold}: {n_loaded} tickers loaded")
    _release_memory()

ensemble_probs = {}
for ticker in ticker_data:
    fold_dps = all_fold_probs.get(ticker, {})
    if not fold_dps:
        continue
    all_dates = set()
    for dp in fold_dps.values():
        all_dates.update(dp.keys())
    ep = {}
    for d in all_dates:
        vals = [dp[d] for dp in fold_dps.values() if d in dp]
        if vals:
            ep[d] = np.mean(vals)
    ensemble_probs[ticker] = ep

tickers_ready = sorted(set(ticker_data.keys()) & set(ensemble_probs.keys()))
print(f"\n    Ready: {len(tickers_ready)} tickers with OHLCV + probs")

for gname, gtickers in GROUPS.items():
    ready = [t for t in gtickers if t in tickers_ready]
    print(f"    {gname}: {len(ready)}/{len(gtickers)} ready ({', '.join(ready)})")


# ── Step 4: Config selection + backtests — ALL groups x ALL strategies ───────
print(f"\n  STEP 4: Running 3 groups x 4 strategies = 12 backtests...")

results_matrix = {}

for gname, gtickers in GROUPS.items():
    group_tickers = [t for t in gtickers if t in tickers_ready]
    if not group_tickers:
        print(f"    {gname}: SKIP — no tickers ready")
        continue

    print(f"\n  ── {gname} ({len(group_tickers)} tickers: {', '.join(group_tickers)}) ──")

    # Select configs for each strategy
    cfgs_m01    = select_m01_best_va(bq_df, group_tickers)
    cfgs_m06    = select_single_fold(bq_df, group_tickers, 403)
    cfgs_f414   = select_single_fold(bq_df, group_tickers, 414)

    # Oracle: exhaustive best-H2 per ticker
    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt   = pd.Timestamp(TEST_END).date()
    h2_start_dt   = pd.Timestamp(HALF2_START).date()

    oracle_cfgs = {}
    for ticker in group_tickers:
        t_df = bq_df[bq_df['ticker'] == ticker]
        if len(t_df) == 0:
            continue
        ohlcv = ticker_data[ticker]
        best_h2_avg = -999
        best_cfg = None

        configs_list = [_bq_row_to_cfg(row) for _, row in t_df.iterrows()]
        configs_list.append(FIXED_UNIVERSAL_CFG.copy())

        for cfg in configs_list:
            fold = cfg['fold']
            if fold is not None and fold in all_fold_probs.get(ticker, {}):
                dp = all_fold_probs[ticker][fold]
            elif ticker in ensemble_probs:
                dp = ensemble_probs[ticker]
            else:
                continue

            em = cfg['entry_minute'] if cfg['entry_minute'] > 0 else ENTRY_MIN
            thresh = cfg['prob_thresh']
            entry_indices = np.where(ohlcv['times'] == em)[0]
            h2_trades_list = []
            last_td = -1

            for idx in entry_indices:
                d_int = int(ohlcv['dates'][idx])
                if d_int == last_td:
                    continue
                d_str = str(d_int)
                d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
                if d_date < h2_start_dt or d_date > test_end_dt:
                    last_td = d_int
                    continue
                prob = dp.get(d_date, 0.0)
                if prob < thresh:
                    last_td = d_int
                    continue
                pnl, td, filled = execute_trade(
                    ohlcv, idx, cfg['tpsl_mode'],
                    cfg['tp_pct'], cfg['sl_pct'], cfg['sl_mult'],
                    cfg['grace_minutes'])
                if filled:
                    h2_trades_list.append(pnl)
                last_td = d_int

            if len(h2_trades_list) >= MIN_H2_TRADES:
                h2_avg = np.mean(h2_trades_list)
                if h2_avg > best_h2_avg:
                    best_h2_avg = h2_avg
                    best_cfg = cfg.copy()

        if best_cfg is not None:
            oracle_cfgs[ticker] = best_cfg

    strats_cfgs = {
        'M01_BEST_VA': cfgs_m01,
        'M06_FOLD403': cfgs_m06,
        'FOLD_414':    cfgs_f414,
        'ORACLE':      oracle_cfgs,
    }

    for sname in STRATEGY_NAMES:
        np.random.seed(42)
        cfgs = strats_cfgs[sname]
        if not cfgs:
            print(f"    {sname}: no configs -> SKIP")
            results_matrix[(gname, sname)] = summarize_trades([], f"{gname}_{sname}")
            continue

        dp_map = {}
        for ticker, cfg in cfgs.items():
            fold = cfg['fold']
            if fold is not None and fold in all_fold_probs.get(ticker, {}):
                dp_map[ticker] = all_fold_probs[ticker][fold]
            elif ticker in ensemble_probs:
                dp_map[ticker] = ensemble_probs[ticker]

        trades, per_tk = run_csv_backtest(cfgs, dp_map, ticker_data, f"{gname}_{sname}")
        port = summarize_trades(trades, f"{gname}_{sname}")
        port['per_ticker'] = per_tk
        port['configs'] = cfgs
        port['n_tickers_active'] = len(per_tk)
        results_matrix[(gname, sname)] = port

        print(f"    {sname}: {port['n_tickers_active']} tickers, {port['n']} trades, "
              f"avg={port['avg']:+.2%}, total={port['total']:+.1%}")


# ==============================================================================
# TABLE 1: GROUP x STRATEGY MATRIX — Full Period
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: FULL PERIOD ({TEST_START} -> {TEST_END})")
print(f"{'='*W}")

print(f"\n  {'Group':<12} {'Strategy':<14} {'Tickers':>7} {'Trades':>7} "
      f"{'Avg PnL':>8} {'Total':>9} {'WR':>5} {'MaxDD':>7}")
print(f"  {'─'*72}")

for gname in GROUPS:
    for sname in STRATEGY_NAMES:
        p = results_matrix.get((gname, sname))
        if p is None or p['n'] == 0:
            print(f"  {gname:<12} {sname:<14} {'—':>7} {'—':>7} "
                  f"{'—':>8} {'—':>9} {'—':>5} {'—':>7}")
            continue
        biased = '*' if sname == 'ORACLE' else ' '
        print(f"  {gname:<12} {sname:<14} {p.get('n_tickers_active',0):>7} {p['n']:>7} "
              f"{p['avg']:>+7.2%}{biased} {p['total']:>+8.1%} {p['wr']:>5.0%} {p['dd']:>+7.2%}")
    print(f"  {'─'*72}")


# ==============================================================================
# TABLE 2: H2-ONLY COMPARISON (more relevant for deploy)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 2: H2-ONLY ({HALF2_START} -> {TEST_END})")
print(f"{'='*W}")

print(f"\n  {'Group':<12} {'Strategy':<14} {'H2 Trd':>7} {'H2 Avg':>8} "
      f"{'H2 Total':>9} {'H2 WR':>6}  {'H1 Avg':>7} {'Edge':>5}")
print(f"  {'─'*72}")

for gname in GROUPS:
    for sname in STRATEGY_NAMES:
        p = results_matrix.get((gname, sname))
        if p is None or p['n'] == 0:
            continue
        edge = 'UP' if p['h2_avg'] > p['h1_avg'] else ('DOWN' if p['h2_avg'] < p['h1_avg'] else 'FLAT')
        biased = '*' if sname == 'ORACLE' else ' '
        print(f"  {gname:<12} {sname:<14} {p['h2_n']:>7} {p['h2_avg']:>+7.2%}{biased} "
              f"{p['h2_total']:>+8.1%} {p['h2_wr']:>5.0%}  "
              f"{p['h1_avg']:>+7.2%} {edge:>5}")
    print(f"  {'─'*72}")


# ==============================================================================
# TABLE 3: TOTAL PnL COMPARISON — The Key Question
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: TOTAL PnL MATRIX — Does wider = more total PnL?")
print(f"{'='*W}")

print(f"\n  Full Period Total PnL (sum of all individual trade PnLs):")
hdr = f"  {'Group':<12} |"
for sname in STRATEGY_NAMES:
    hdr += f" {sname:>14} |"
print(hdr)
print(f"  {'─'*72}")

for gname in GROUPS:
    row = f"  {gname:<12} |"
    for sname in STRATEGY_NAMES:
        p = results_matrix.get((gname, sname))
        if p is None or p['n'] == 0:
            row += f" {'—':>14} |"
        else:
            biased = '*' if sname == 'ORACLE' else ' '
            row += f" {p['total']:>+12.1%}{biased} |"
    print(row)

print(f"\n  H2-Only Total PnL:")
hdr = f"  {'Group':<12} |"
for sname in STRATEGY_NAMES:
    hdr += f" {sname:>14} |"
print(hdr)
print(f"  {'─'*72}")

for gname in GROUPS:
    row = f"  {gname:<12} |"
    for sname in STRATEGY_NAMES:
        p = results_matrix.get((gname, sname))
        if p is None or p['n'] == 0:
            row += f" {'—':>14} |"
        else:
            biased = '*' if sname == 'ORACLE' else ' '
            row += f" {p['h2_total']:>+12.1%}{biased} |"
    print(row)

print(f"\n  Trade Count:")
hdr = f"  {'Group':<12} |"
for sname in STRATEGY_NAMES:
    hdr += f" {sname:>14} |"
print(hdr)
print(f"  {'─'*72}")

for gname in GROUPS:
    row = f"  {gname:<12} |"
    for sname in STRATEGY_NAMES:
        p = results_matrix.get((gname, sname))
        if p is None or p['n'] == 0:
            row += f" {'—':>14} |"
        else:
            row += f" {p['n']:>14} |"
    print(row)


# ==============================================================================
# TABLE 4: PER-TICKER DETAIL for best non-oracle strategy
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 4: PER-TICKER BREAKDOWN (best non-oracle strategy per group)")
print(f"{'='*W}")

for gname in GROUPS:
    best_sname = None
    best_total = -999
    for sname in STRATEGY_NAMES:
        if sname == 'ORACLE':
            continue
        p = results_matrix.get((gname, sname))
        if p and p['total'] > best_total:
            best_total = p['total']
            best_sname = sname

    if best_sname is None:
        continue

    p = results_matrix[(gname, best_sname)]
    per_tk = p.get('per_ticker', {})
    cfgs = p.get('configs', {})

    print(f"\n  {gname} — Best: {best_sname} (total={p['total']:+.1%}, avg={p['avg']:+.2%})")
    print(f"  {'Ticker':<7} {'Trades':>6} {'Avg':>7} {'Total':>8} {'WR':>5} "
          f"{'Fold':>5} {'Mode':<8} {'TP':>4} {'SL':>4} {'Thr':>5}")
    print(f"  {'─'*70}")

    for ticker in sorted(set(list(per_tk.keys()) + list(cfgs.keys()))):
        trades = per_tk.get(ticker, [])
        cfg = cfgs.get(ticker)
        n = len(trades)
        avg_pnl = np.mean([t['pnl'] for t in trades]) if n > 0 else 0
        total_pnl = sum(t['pnl'] for t in trades) if n > 0 else 0
        wr = np.mean([1 for t in trades if t['pnl'] > 0]) / n * 100 if n > 0 else 0
        fold_s = f"{cfg['fold']}" if cfg and cfg['fold'] else '—'
        mode_s = cfg['tpsl_mode'] if cfg else '—'
        tp_s = f"{cfg['tp_pct']*100:.0f}%" if cfg else '—'
        sl_s = f"{cfg['sl_pct']*100:.0f}%" if cfg and cfg['tpsl_mode'] != 'atr_sl' else 'ATR'
        thr_s = f"{cfg['prob_thresh']:.2f}" if cfg else '—'
        print(f"  {ticker:<7} {n:>6} {avg_pnl:>+6.2%} {total_pnl:>+7.1%} {wr:>4.0f}% "
              f"{fold_s:>5} {mode_s:<8} {tp_s:>4} {sl_s:>4} {thr_s:>5}")


# ==============================================================================
# CHARTS (3x2 layout)
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(3, 2, figsize=(22, 20))

# ── Chart 1: Total PnL bar chart — Group x Strategy ──
ax1 = axes[0, 0]
x = np.arange(len(STRATEGY_NAMES))
bar_w = 0.22
for i, gname in enumerate(GROUPS):
    totals = []
    for sname in STRATEGY_NAMES:
        p = results_matrix.get((gname, sname))
        totals.append(p['total'] * 100 if p and p['n'] > 0 else 0)
    offset = (i - len(GROUPS)/2 + 0.5) * bar_w
    ax1.bar(x + offset, totals, bar_w, label=f"{gname} ({len(GROUPS[gname])})",
            color=GROUP_COLORS[gname], alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(STRATEGY_NAMES, fontsize=10, rotation=10)
ax1.set_ylabel('Total PnL (%)')
ax1.set_title('Total PnL — Group x Strategy (Full Period)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.axhline(y=0, color='gray', linewidth=0.5)
ax1.grid(True, alpha=0.3, axis='y')

# ── Chart 2: H2 Total PnL ──
ax2 = axes[0, 1]
for i, gname in enumerate(GROUPS):
    h2_totals = []
    for sname in STRATEGY_NAMES:
        p = results_matrix.get((gname, sname))
        h2_totals.append(p['h2_total'] * 100 if p and p['n'] > 0 else 0)
    offset = (i - len(GROUPS)/2 + 0.5) * bar_w
    ax2.bar(x + offset, h2_totals, bar_w, label=f"{gname} ({len(GROUPS[gname])})",
            color=GROUP_COLORS[gname], alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(STRATEGY_NAMES, fontsize=10, rotation=10)
ax2.set_ylabel('H2 Total PnL (%)')
ax2.set_title('H2 Total PnL — Group x Strategy', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.axhline(y=0, color='gray', linewidth=0.5)
ax2.grid(True, alpha=0.3, axis='y')

# ── Chart 3: Avg PnL vs Trade Count scatter ──
ax3 = axes[1, 0]
for gname in GROUPS:
    for sname in STRATEGY_NAMES:
        if sname == 'ORACLE':
            continue
        p = results_matrix.get((gname, sname))
        if p is None or p['n'] == 0:
            continue
        ax3.scatter(p['n'], p['avg'] * 100, s=150,
                    color=GROUP_COLORS[gname], marker='o', alpha=0.8,
                    edgecolors='black', linewidth=0.5)
        ax3.annotate(f"{gname[:3]}_{sname[:4]}", (p['n'], p['avg'] * 100),
                     fontsize=7, ha='left', va='bottom')
ax3.axhline(y=DEPLOY_FLOOR * 100, color='red', linewidth=1, linestyle='--', alpha=0.7,
            label=f'Deploy ({DEPLOY_FLOOR:.1%})')
ax3.axhline(y=0, color='gray', linewidth=0.5)
ax3.set_xlabel('Trade Count')
ax3.set_ylabel('Avg PnL (%)')
ax3.set_title('Avg PnL vs Trade Count (non-oracle)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── Chart 4: Cumulative PnL curves — best strategy per group ──
ax4 = axes[1, 1]
for gname in GROUPS:
    best_sname = None
    best_total = -999
    for sname in STRATEGY_NAMES:
        if sname == 'ORACLE':
            continue
        p = results_matrix.get((gname, sname))
        if p and p['total'] > best_total:
            best_total = p['total']
            best_sname = sname
    if best_sname is None:
        continue
    p = results_matrix[(gname, best_sname)]
    if p['n'] == 0:
        continue
    df_t = p['trades_df'].sort_values('date')
    cum = df_t['pnl'].cumsum() * 100
    ax4.plot(pd.to_datetime(df_t['date'].values), cum.values,
             color=GROUP_COLORS[gname], linewidth=2.5,
             label=f"{gname} ({best_sname})")
ax4.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax4.axvline(x=pd.Timestamp(HALF2_START), color='red', linewidth=1,
            linestyle=':', alpha=0.5, label='H2 start')
ax4.set_ylabel('Cumulative PnL (%)')
ax4.set_title('Cumulative PnL — Best Strategy per Group', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# ── Chart 5: Per-ticker avg PnL for FOLD_414 across groups ──
ax5 = axes[2, 0]
all_tickers_414 = set()
for gname in GROUPS:
    p = results_matrix.get((gname, 'FOLD_414'))
    if p and 'per_ticker' in p:
        all_tickers_414.update(p['per_ticker'].keys())
tickers_sorted = sorted(all_tickers_414)
if tickers_sorted:
    x5 = np.arange(len(tickers_sorted))
    for i, gname in enumerate(GROUPS):
        p = results_matrix.get((gname, 'FOLD_414'))
        per_tk = p.get('per_ticker', {}) if p else {}
        avgs = []
        for tk in tickers_sorted:
            trades = per_tk.get(tk, [])
            avgs.append(np.mean([t['pnl'] for t in trades]) * 100 if trades else 0)
        offset = (i - len(GROUPS)/2 + 0.5) * bar_w
        ax5.bar(x5 + offset, avgs, bar_w, label=gname, color=GROUP_COLORS[gname], alpha=0.85)
    ax5.axhline(y=DEPLOY_FLOOR * 100, color='red', linewidth=1, linestyle='--', alpha=0.7)
    ax5.axhline(y=0, color='gray', linewidth=0.5)
    ax5.set_xticks(x5)
    ax5.set_xticklabels(tickers_sorted, fontsize=8, rotation=45, ha='right')
    ax5.set_ylabel('Avg PnL (%)')
    ax5.set_title('FOLD_414: Per-Ticker Avg PnL', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

# ── Chart 6: H2 avg PnL heatmap — Group x Strategy ──
ax6 = axes[2, 1]
group_names = list(GROUPS.keys())
h2_data = np.zeros((len(group_names), len(STRATEGY_NAMES)))
for i, gname in enumerate(group_names):
    for j, sname in enumerate(STRATEGY_NAMES):
        p = results_matrix.get((gname, sname))
        h2_data[i, j] = p['h2_avg'] * 100 if p and p['h2_n'] > 0 else 0

vmax = max(abs(h2_data.min()), abs(h2_data.max()))
if vmax == 0:
    vmax = 1
im = ax6.imshow(h2_data, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
ax6.set_yticks(range(len(group_names)))
ax6.set_yticklabels(group_names, fontsize=11)
ax6.set_xticks(range(len(STRATEGY_NAMES)))
ax6.set_xticklabels(STRATEGY_NAMES, fontsize=10)
for i in range(len(group_names)):
    for j in range(len(STRATEGY_NAMES)):
        p = results_matrix.get((group_names[i], STRATEGY_NAMES[j]))
        n_trades = p['h2_n'] if p else 0
        ax6.text(j, i, f'{h2_data[i,j]:+.1f}%\n({n_trades}tr)',
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 color='white' if abs(h2_data[i,j]) > 1.5 else 'black')
fig.colorbar(im, ax=ax6, label='H2 Avg PnL (%)', shrink=0.8)
ax6.set_title('H2 Avg PnL Heatmap (trades in parens)', fontsize=13, fontweight='bold')

fig.suptitle(f'SHORT Group Comparison — Does Wider = More Total PnL?\n'
             f'{TEST_START} -> {TEST_END}  |  Deploy threshold: {DEPLOY_FLOOR:.1%}',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/tmp/backtest_short_groups.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Chart saved: /tmp/backtest_short_groups.png")


# ==============================================================================
# VERDICT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

for gname in GROUPS:
    print(f"\n  {gname} ({len(GROUPS[gname])} tickers):")
    for sname in STRATEGY_NAMES:
        p = results_matrix.get((gname, sname))
        if p is None or p['n'] == 0:
            print(f"    {sname:<14}: no trades")
            continue
        biased = ' [BIASED]' if sname == 'ORACLE' else ''
        deploy = 'DEPLOY-READY' if p['avg'] >= DEPLOY_FLOOR else 'BELOW THRESH'
        h2_deploy = 'H2-READY' if p['h2_avg'] >= DEPLOY_FLOOR else 'H2-BELOW'
        print(f"    {sname:<14}: {p['n']:>4} trades, avg={p['avg']:>+.2%}, "
              f"total={p['total']:>+.1%}, H2avg={p['h2_avg']:>+.2%} "
              f"-> {deploy} / {h2_deploy}{biased}")

print(f"\n  KEY QUESTION: Does wider ticker set = more total PnL?")
for sname in ['M01_BEST_VA', 'M06_FOLD403', 'FOLD_414']:
    totals = {}
    for gname in GROUPS:
        p = results_matrix.get((gname, sname))
        if p and p['n'] > 0:
            totals[gname] = p['total']
    if totals:
        best_group = max(totals, key=totals.get)
        print(f"  {sname}: Best total = {best_group} ({totals[best_group]:+.1%})")

print(f"\n  LONG S3 reference: +2.23% avg")
print(f"  Deploy threshold:  {DEPLOY_FLOOR:.1%}")
print(f"{'='*W}")
