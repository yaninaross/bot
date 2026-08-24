# %% ── SHORT Strategy Backtest — Top 3 Config Selection Methods (CSV-Scored) ──
#
# Compares 3 top config selection methods + oracle ceiling from Phase B:
#   M01: Best val_a (highest val_a_avg_pnl per ticker, any fold)
#   M06: Fold-403 (best val_a within fold 403 only)
#   M07: Alpha-2 folds (best val_a within folds 400-405)
#   ORACLE: Best actual H2 config per ticker (BIASED ceiling)
#
# Uses CSV-scored probabilities (not live TF). Fold-specific scoring.
# Tickers: RULE_1 (6 tickers from Phase A asset selection).
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
BUCKET_NAME   = "time-series-bucket-ys"
RESULTS_DIR   = "Algos/Results_all_v2"
INFERENCE_DIR = "Data/Inference_v2"

TEST_START  = '2026-04-01'
TEST_END    = '2026-07-28'
HALF2_START = '2026-06-01'

ENTRY_MIN     = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.002
STARTING_CAPITAL = 200_000
DEPLOY_FLOOR  = 0.018
SL_FIXED      = 0.03

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

SHORT_TICKERS = ['BE', 'IREN', 'OKLO', 'ONDS', 'RKLB', 'WOLF']

FOLDS_A2 = [400, 401, 402, 403, 404, 405]
FOLDS_A1 = [410, 411, 412, 413, 414, 415]
ALL_FOLDS = FOLDS_A2 + FOLDS_A1

FOLD_EXPERIMENTS    = {f: f'{f}_XSECTOR' for f in ALL_FOLDS}
FOLD_MODEL_VERSIONS = {f: f'robust_wf_v12_short_{f}' for f in ALL_FOLDS}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
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

STRATEGY_COLORS = {
    'M01_BEST_VA':      '#1565C0',
    'M06_FOLD403':      '#E65100',
    'M07_ALPHA2':       '#2E7D32',
    'ORACLE':           '#9C27B0',
}

TICKER_COLORS = {
    'BE':   '#2196F3', 'IREN': '#4CAF50', 'OKLO': '#FF9800',
    'ONDS': '#E91E63', 'RKLB': '#9C27B0', 'WOLF': '#00BCD4',
}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110


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
# SUMMARIZE RESULTS
# ==============================================================================
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
# BACKTEST FUNCTION — CSV-scored with fold-specific probs
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


# ==============================================================================
# CONFIG SELECTION METHODS (from select_short_config.py Phase 4)
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


def select_m06_fold403(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'] == 403)]
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker]
        cfg = _pick_best(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


def select_m07_alpha2(bq_df, tickers):
    selections = {}
    for ticker in tickers:
        t_df = bq_df[(bq_df['ticker'] == ticker) & (bq_df['fold'].isin(FOLDS_A2))]
        if len(t_df) == 0:
            t_df = bq_df[bq_df['ticker'] == ticker]
        cfg = _pick_best(t_df, 'val_a_avg_pnl')
        if cfg:
            selections[ticker] = cfg
    return selections


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT STRATEGY BACKTEST — Top 3 Config Methods + Oracle")
print(f"  TICKERS: {', '.join(sorted(SHORT_TICKERS))} (RULE_1)")
print(f"  TEST: {TEST_START} → {TEST_END}  |  H2: {HALF2_START} → {TEST_END}")
print(f"  ENTRY: 6:15 PST ({ENTRY_MIN} min)  |  EXIT: 1:00 PM ({EXIT_TIME_MIN} min)")
print(f"  SLIPPAGE: {SLIPPAGE_PCT:.2%} each way  |  CAPITAL: ${STARTING_CAPITAL:,}")
print(f"{'='*W}")


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
print(f"    RULE_1 in BQ: {[t for t in SHORT_TICKERS if t in bq_tickers]}")


# ── Step 2: Run config selection methods ─────────────────────────────────────
print(f"\n  STEP 2: Selecting configs with 3 methods...")

configs_m01 = select_m01_best_va(bq_df, SHORT_TICKERS)
configs_m06 = select_m06_fold403(bq_df, SHORT_TICKERS)
configs_m07 = select_m07_alpha2(bq_df, SHORT_TICKERS)

for label, cfgs in [('M01_BEST_VA', configs_m01), ('M06_FOLD403', configs_m06),
                     ('M07_ALPHA2', configs_m07)]:
    tks = sorted(cfgs.keys())
    print(f"    {label}: {len(tks)} tickers — {', '.join(tks)}")
    for tk in tks:
        c = cfgs[tk]
        print(f"      {tk}: f{c['fold']}_{c['tpsl_mode']} TP={c['tp_pct']:.0%} "
              f"SL={c['sl_pct']:.0%} thresh={c['prob_thresh']:.2f} va={c['val_a_avg_pnl']:+.2%}")


# ── Step 3: Load OHLCV data ─────────────────────────────────────────────────
print(f"\n  STEP 3: Loading OHLCV + ATR for {len(SHORT_TICKERS)} tickers...")
ticker_data = {}

for ticker in SHORT_TICKERS:
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
        print(f"    {ticker}: {n_days} trading days, ATR_pct={atr_cut.median():.4f}")
        del df
        _master_cache.pop(ticker, None)
    except Exception as ex:
        print(f"    {ticker}: OHLCV load failed -- {ex}")
        _master_cache.pop(ticker, None)
    gc.collect()

_release_memory()


# ── Step 4: Load fold-specific CSV scores ────────────────────────────────────
print(f"\n  STEP 4: Loading CSV seed scores (fold-specific)...")

all_strategy_configs = {
    'M01_BEST_VA': configs_m01,
    'M06_FOLD403': configs_m06,
    'M07_ALPHA2':  configs_m07,
}

needed_folds = set()
for cfgs in all_strategy_configs.values():
    for ticker, cfg in cfgs.items():
        if cfg['fold'] is not None:
            needed_folds.add(cfg['fold'])
needed_folds.update(ALL_FOLDS)

all_fold_probs = defaultdict(dict)
for fold in sorted(needed_folds):
    exp = FOLD_EXPERIMENTS[fold]
    n_loaded = 0
    for ticker in SHORT_TICKERS:
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


# ── Step 5: Build per-strategy daily probs ───────────────────────────────────
print(f"\n  STEP 5: Building per-strategy daily probability maps...")

strategy_probs = {}
for strat_name, cfgs in all_strategy_configs.items():
    dp_map = {}
    for ticker, cfg in cfgs.items():
        if ticker not in ticker_data:
            continue
        fold = cfg['fold']
        if fold is not None and fold in all_fold_probs.get(ticker, {}):
            dp_map[ticker] = all_fold_probs[ticker][fold]
        elif ticker in ensemble_probs:
            dp_map[ticker] = ensemble_probs[ticker]
    strategy_probs[strat_name] = dp_map
    print(f"    {strat_name}: {len(dp_map)} tickers with probs")


# ── Step 6: Exhaustive oracle backtest ───────────────────────────────────────
print(f"\n  STEP 6: Running exhaustive oracle backtest (all configs x all tickers)...")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()

all_ticker_configs = {}
for ticker in SHORT_TICKERS:
    t_df = bq_df[bq_df['ticker'] == ticker]
    if len(t_df) == 0:
        continue
    configs_list = []
    for _, bq_row in t_df.iterrows():
        configs_list.append({
            'fold': int(bq_row['fold']),
            'tpsl_mode': bq_row['tpsl_mode'],
            'prob_thresh': bq_row['prob_thresh'],
            'tp_pct': bq_row['tp_pct'],
            'sl_pct': bq_row['sl_pct'],
            'sl_mult': bq_row['sl_mult'],
            'grace_minutes': int(bq_row['grace_minutes']),
            'entry_minute': int(bq_row['entry_minute']) if bq_row['entry_minute'] > 0 else ENTRY_MIN,
            'config_id': f"f{int(bq_row['fold'])}_{bq_row['tpsl_mode']}_t{bq_row['prob_thresh']:.2f}_tp{bq_row['tp_pct']:.2f}",
        })
    configs_list.append(FIXED_UNIVERSAL_CFG.copy())
    all_ticker_configs[ticker] = configs_list

oracle_configs = {}
total_bt = 0

for ticker in tickers_ready:
    if ticker not in all_ticker_configs:
        continue
    ohlcv = ticker_data[ticker]
    best_h2_avg = -999
    best_cfg = None

    for cfg in all_ticker_configs[ticker]:
        if cfg['fold'] is not None:
            dp = all_fold_probs.get(ticker, {}).get(cfg['fold'], {})
            if not dp:
                dp = ensemble_probs.get(ticker, {})
        else:
            dp = ensemble_probs.get(ticker, {})
        if not dp:
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
            if d_date < test_start_dt or d_date > test_end_dt:
                last_td = d_int
                continue
            if d_date < h2_start_dt:
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

        total_bt += 1
        if len(h2_trades_list) >= MIN_H2_TRADES:
            h2_avg = np.mean(h2_trades_list)
            if h2_avg > best_h2_avg:
                best_h2_avg = h2_avg
                best_cfg = cfg.copy()

    if best_cfg is not None:
        oracle_configs[ticker] = best_cfg
        print(f"    {ticker}: oracle = f{best_cfg['fold']}_{best_cfg['tpsl_mode']} "
              f"TP={best_cfg['tp_pct']:.0%} thresh={best_cfg['prob_thresh']:.2f} -> H2 avg={best_h2_avg:+.2%}")

print(f"    Total configs backtested: {total_bt}")
print(f"    Oracle tickers: {len(oracle_configs)}")

all_strategy_configs['ORACLE'] = oracle_configs

oracle_dp_map = {}
for ticker, cfg in oracle_configs.items():
    fold = cfg['fold']
    if fold is not None and fold in all_fold_probs.get(ticker, {}):
        oracle_dp_map[ticker] = all_fold_probs[ticker][fold]
    elif ticker in ensemble_probs:
        oracle_dp_map[ticker] = ensemble_probs[ticker]
strategy_probs['ORACLE'] = oracle_dp_map


# ── Step 7: Run backtests ───────────────────────────────────────────────────
print(f"\n  STEP 7: Running backtests for all 4 strategies...")

strategy_results = {}
strategy_per_tk = {}

for strat_name in ['M01_BEST_VA', 'M06_FOLD403', 'M07_ALPHA2', 'ORACLE']:
    np.random.seed(42)
    cfgs = all_strategy_configs[strat_name]
    probs = strategy_probs[strat_name]
    trades, per_tk = run_csv_backtest(cfgs, probs, ticker_data, strat_name)
    port = summarize_trades(trades, strat_name)
    strategy_results[strat_name] = port
    strategy_per_tk[strat_name] = per_tk
    print(f"    {strat_name}: {port['n']} trades, avg={port['avg']:+.2%}, "
          f"wr={port['wr']:.0%}, total={port['total']:+.1%}")


# ==============================================================================
# RESULTS TABLES
# ==============================================================================
strat_order = ['ORACLE', 'M01_BEST_VA', 'M07_ALPHA2', 'M06_FOLD403']

# ── Table 1: Portfolio Summary ──
print(f"\n{'='*W}")
print(f"  PORTFOLIO SUMMARY — All Strategies ({TEST_START} -> {TEST_END})")
print(f"{'='*W}")

print(f"\n  {'Strategy':<16}  {'Trades':>6}  {'Total':>8}  {'Avg':>7}  {'WR':>5}  {'MaxDD':>7}  {'Deploy':>6}")
print(f"  {'─'*62}")

oracle_avg = strategy_results['ORACLE']['avg'] if strategy_results['ORACLE']['avg'] != 0 else 1
for sn in strat_order:
    p = strategy_results[sn]
    deploy = 'YES' if p['avg'] >= DEPLOY_FLOOR else 'NO'
    print(f"  {sn:<16}  {p['n']:>6}  {p['total']:>+8.1%}  "
          f"{p['avg']:>+7.2%}  {p['wr']:>5.0%}  {p['dd']:>+7.2%}  {deploy:>6}")


# ── Table 2: Config Details ──
print(f"\n  Config Details per Ticker:")
print(f"  {'Ticker':<7} | {'Strategy':<16} | {'Fold':>4} {'Mode':<8} {'Prob>':>5} {'TP':>5} {'SL':>5} {'ValA':>7}")
print(f"  {'─'*70}")
for ticker in sorted(SHORT_TICKERS):
    for sn in strat_order:
        cfg = all_strategy_configs[sn].get(ticker)
        if cfg:
            fold_s = f"{cfg['fold']}" if cfg['fold'] else '—'
            print(f"  {ticker:<7} | {sn:<16} | {fold_s:>4} {cfg['tpsl_mode']:<8} "
                  f"{cfg['prob_thresh']:>5.2f} {cfg['tp_pct']:>4.0%} "
                  f"{cfg['sl_pct']:>4.0%} {cfg.get('val_a_avg_pnl',0):>+7.2%}")
        else:
            print(f"  {ticker:<7} | {sn:<16} |  {'— no config —':>40}")
    print(f"  {'─'*70}")


# ── Table 3: Per-Ticker Results ──
print(f"\n  Per-Ticker Results:")
hdr = '  {:7} |'.format('Ticker')
for sn in strat_order:
    hdr += f" {sn[:12]:>12} |"
print(hdr)
sub = '  {:7} |'.format('')
for _ in strat_order:
    sub += f" {'N':>3} {'Avg':>6} {'WR':>4} |"
print(sub)
print(f"  {'─'*75}")

for ticker in sorted(SHORT_TICKERS):
    row = f"  {ticker:<7} |"
    for sn in strat_order:
        trades = strategy_per_tk[sn].get(ticker, [])
        n = len(trades)
        avg = np.mean([t['pnl'] for t in trades]) if n > 0 else 0
        wr = np.mean([1 for t in trades if t['pnl'] > 0]) / n * 100 if n > 0 else 0
        row += f" {n:>3} {avg:>+5.2%} {wr:>3.0f}% |"
    print(row)


# ── Table 4: Monthly Breakdown ──
all_months = set()
for p in strategy_results.values():
    all_months.update(p['monthly'].keys())
all_months = sorted(all_months)

if all_months:
    print(f"\n  Monthly Breakdown:")
    hdr = f"  {'Month':<8} |"
    for sn in strat_order:
        hdr += f" {'--- ' + sn[:8] + ' ---':>20} |"
    print(hdr)
    sub = f"  {'':8} |"
    for _ in strat_order:
        sub += f" {'N':>3} {'Total':>7} {'Avg':>7} |"
    print(sub)
    print(f"  {'─'*90}")

    for month in all_months:
        row = f"  {month:<8} |"
        for sn in strat_order:
            m = strategy_results[sn]['monthly'].get(month, {'n': 0, 'total': 0, 'avg': 0})
            row += f" {m['n']:>3} {m['total']:>+7.1%} {m['avg']:>+7.2%} |"
        print(row)


# ── Table 5: H1 vs H2 Split ──
print(f"\n  H1 vs H2 Split (H1=Apr-May, H2=Jun-Jul):")
print(f"  {'Strategy':<16} | {'-- H1 ------------------':} | {'-- H2 ------------------':} | {'Edge':>5}")
print(f"  {'':16} | {'N':>4} {'Avg':>7} {'WR':>5} {'Tot':>7} | {'N':>4} {'Avg':>7} {'WR':>5} {'Tot':>7} |")
print(f"  {'─'*80}")

for sn in strat_order:
    trades_df = strategy_results[sn]['trades_df']
    if len(trades_df) == 0:
        print(f"  {sn:<16} | {'— no trades —':>40}")
        continue
    h1 = trades_df[trades_df['half'] == 'H1']
    h2 = trades_df[trades_df['half'] == 'H2']

    h1_n = len(h1); h1_avg = h1['pnl'].mean() if h1_n > 0 else 0
    h1_wr = (h1['pnl'] > 0).mean() if h1_n > 0 else 0
    h1_tot = h1['pnl'].sum() if h1_n > 0 else 0

    h2_n = len(h2); h2_avg = h2['pnl'].mean() if h2_n > 0 else 0
    h2_wr = (h2['pnl'] > 0).mean() if h2_n > 0 else 0
    h2_tot = h2['pnl'].sum() if h2_n > 0 else 0

    edge = 'UP' if h2_avg > h1_avg else ('DOWN' if h2_avg < h1_avg else 'FLAT')
    print(f"  {sn:<16} | {h1_n:>4} {h1_avg:>+7.2%} {h1_wr:>5.0%} {h1_tot:>+7.1%} | "
          f"{h2_n:>4} {h2_avg:>+7.2%} {h2_wr:>5.0%} {h2_tot:>+7.1%} | {edge:>5}")


# ==============================================================================
# CHARTS (3x2 layout)
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(3, 2, figsize=(20, 18))

# ── Chart 1: Cumulative PnL — all strategies ──
ax1 = axes[0, 0]
for sn in strat_order:
    p = strategy_results[sn]
    if p['n'] == 0:
        continue
    df_t = p['trades_df'].sort_values('date')
    cum = df_t['pnl'].cumsum() * 100
    ls = '--' if sn == 'ORACLE' else '-'
    lw = 1.5 if sn == 'ORACLE' else 2.5
    ax1.plot(pd.to_datetime(df_t['date'].values), cum.values,
             color=STRATEGY_COLORS[sn], linewidth=lw, linestyle=ls, label=sn)
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax1.axvline(x=pd.Timestamp(HALF2_START), color='red', linewidth=1, linestyle=':', alpha=0.5, label='H2 start')
ax1.set_ylabel('Cumulative PnL (%)')
ax1.set_title('Cumulative PnL — All Strategies', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# ── Chart 2: Equity curves ──
ax2 = axes[0, 1]
for sn in strat_order:
    p = strategy_results[sn]
    if p['n'] == 0:
        continue
    df_s = p['trades_df'].sort_values('date')
    n_tk = max(len(strategy_per_tk[sn]), 1)
    eq = STARTING_CAPITAL
    eq_vals, eq_dates = [], []
    for _, row in df_s.iterrows():
        eq += (eq / n_tk) * row['pnl']
        eq_vals.append(eq)
        eq_dates.append(row['date'])
    ls = '--' if sn == 'ORACLE' else '-'
    lw = 1.5 if sn == 'ORACLE' else 2.5
    ax2.plot(pd.to_datetime(eq_dates), np.array(eq_vals) / 1000,
             color=STRATEGY_COLORS[sn], linewidth=lw, linestyle=ls, label=sn)
ax2.axhline(y=STARTING_CAPITAL / 1000, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax2.set_ylabel('Equity ($K)')
ax2.set_title(f'Equity Curves (${STARTING_CAPITAL/1000:.0f}K start)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# ── Chart 3: Per-ticker avg PnL — grouped bar ──
ax3 = axes[1, 0]
tickers_sorted = sorted(SHORT_TICKERS)
x = np.arange(len(tickers_sorted))
bar_w = 0.18
for i, sn in enumerate(strat_order):
    avgs = []
    for tk in tickers_sorted:
        trades = strategy_per_tk[sn].get(tk, [])
        avgs.append(np.mean([t['pnl'] for t in trades]) * 100 if trades else 0)
    offset = (i - len(strat_order)/2 + 0.5) * bar_w
    ax3.bar(x + offset, avgs, bar_w, label=sn, color=STRATEGY_COLORS[sn], alpha=0.85)
ax3.axhline(y=DEPLOY_FLOOR * 100, color='red', linewidth=1, linestyle='--', alpha=0.7, label=f'Deploy ({DEPLOY_FLOOR:.1%})')
ax3.set_xticks(x)
ax3.set_xticklabels(tickers_sorted, fontsize=11)
ax3.set_ylabel('Avg PnL per Trade (%)')
ax3.set_title('Per-Ticker Avg PnL: Strategy Comparison', fontsize=13, fontweight='bold')
ax3.legend(fontsize=8, loc='best')
ax3.axhline(y=0, color='gray', linewidth=0.5)
ax3.grid(True, alpha=0.3, axis='y')

# ── Chart 4: Per-ticker cumulative PnL (best non-oracle strategy) ──
ax4 = axes[1, 1]
best_non_oracle = max([sn for sn in strat_order if sn != 'ORACLE'],
                      key=lambda sn: strategy_results[sn]['avg'])
for ticker in tickers_sorted:
    trades = strategy_per_tk[best_non_oracle].get(ticker, [])
    if not trades:
        continue
    df_t = pd.DataFrame(trades).sort_values('date')
    cum = df_t['pnl'].cumsum() * 100
    ax4.plot(pd.to_datetime(df_t['date'].values), cum.values,
             label=ticker, color=TICKER_COLORS.get(ticker, '#666'), linewidth=2)
ax4.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax4.axvline(x=pd.Timestamp(HALF2_START), color='red', linewidth=1, linestyle=':', alpha=0.5)
ax4.set_ylabel('Cumulative PnL (%)')
ax4.set_title(f'Per-Ticker ({best_non_oracle})', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9, loc='best')
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# ── Chart 5: Monthly PnL heatmap ──
ax5 = axes[2, 0]
if all_months:
    month_data = np.zeros((len(strat_order), len(all_months)))
    for i, sn in enumerate(strat_order):
        for j, month in enumerate(all_months):
            m = strategy_results[sn]['monthly'].get(month, {'avg': 0})
            month_data[i, j] = m['avg'] * 100

    vmax = max(abs(month_data.min()), abs(month_data.max()))
    if vmax == 0:
        vmax = 1
    im = ax5.imshow(month_data, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax5.set_yticks(range(len(strat_order)))
    ax5.set_yticklabels(strat_order, fontsize=10)
    ax5.set_xticks(range(len(all_months)))
    ax5.set_xticklabels(all_months, fontsize=10)
    for i in range(len(strat_order)):
        for j in range(len(all_months)):
            ax5.text(j, i, f'{month_data[i,j]:+.1f}%', ha='center', va='center',
                     fontsize=10, fontweight='bold',
                     color='white' if abs(month_data[i,j]) > 2 else 'black')
    fig.colorbar(im, ax=ax5, label='Avg PnL (%)', shrink=0.8)
    ax5.set_title('Monthly Avg PnL Heatmap', fontsize=13, fontweight='bold')
else:
    ax5.text(0.5, 0.5, 'No monthly data', ha='center', va='center', fontsize=14)
    ax5.set_title('Monthly Avg PnL Heatmap', fontsize=13, fontweight='bold')

# ── Chart 6: H1 vs H2 comparison ──
ax6 = axes[2, 1]
x = np.arange(len(strat_order))
bar_w = 0.35
h1_avgs = []
h2_avgs = []
for sn in strat_order:
    trades_df = strategy_results[sn]['trades_df']
    if len(trades_df) == 0:
        h1_avgs.append(0)
        h2_avgs.append(0)
        continue
    h1 = trades_df[trades_df['half'] == 'H1']
    h2 = trades_df[trades_df['half'] == 'H2']
    h1_avgs.append(h1['pnl'].mean() * 100 if len(h1) > 0 else 0)
    h2_avgs.append(h2['pnl'].mean() * 100 if len(h2) > 0 else 0)

ax6.bar(x - bar_w/2, h1_avgs, bar_w, label='H1 (Apr-May)', color='#90CAF9', alpha=0.85)
ax6.bar(x + bar_w/2, h2_avgs, bar_w, label='H2 (Jun-Jul)', color='#1565C0', alpha=0.85)
ax6.axhline(y=DEPLOY_FLOOR * 100, color='red', linewidth=1, linestyle='--', alpha=0.7)
ax6.set_xticks(x)
ax6.set_xticklabels(strat_order, fontsize=10, rotation=15)
ax6.set_ylabel('Avg PnL per Trade (%)')
ax6.set_title('H1 vs H2 — Edge Persistence', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
ax6.axhline(y=0, color='gray', linewidth=0.5)
ax6.grid(True, alpha=0.3, axis='y')

fig.suptitle(f'SHORT Strategy Backtest — RULE_1 ({len(SHORT_TICKERS)} tickers)\n'
             f'{TEST_START} -> {TEST_END}  |  Deploy threshold: {DEPLOY_FLOOR:.1%}',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/tmp/backtest_short_strategies.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Chart saved: /tmp/backtest_short_strategies.png")


# ==============================================================================
# TRADE LOG — best non-oracle strategy
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TRADE LOG — {best_non_oracle}")
print(f"{'='*W}")

best_port = strategy_results[best_non_oracle]
if best_port['n'] > 0:
    df_log = best_port['trades_df'].sort_values('date')
    for _, t in df_log.iterrows():
        print(f"    {t['date']}  {t['ticker']:<7}  pnl={t['pnl']:+.2%}  "
              f"prob={t['prob']:.3f}  close=${t['close']:.2f}  "
              f"mode={t['mode']}  tp={t['tp']:.0%}  {t['half']}")


# ==============================================================================
# FINAL VERDICT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

for sn in strat_order:
    p = strategy_results[sn]
    biased = ' [BIASED]' if sn == 'ORACLE' else ''
    deploy = 'DEPLOY-READY' if p['avg'] >= DEPLOY_FLOOR else 'BELOW THRESHOLD'
    capture = f" ({p['avg']/oracle_avg*100:.0f}% oracle)" if sn != 'ORACLE' else ''
    n_tk = len(strategy_per_tk[sn])
    print(f"  {sn:<16}: {p['n']} trades, avg={p['avg']:+.2%}, wr={p['wr']:.0%}, "
          f"{n_tk} tickers -> {deploy}{capture}{biased}")

print(f"\n  LONG S3 reference: +2.23% avg")
print(f"  Deploy threshold:  {DEPLOY_FLOOR:.1%}")
print(f"{'='*W}")
