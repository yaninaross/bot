"""
backtest_short_csv.py — CSV-Scored SHORT Model Backtest (Exp 400/410)
======================================================================
Compares two config selection strategies:
  A) ENSEMBLE — best ensemble BQ row (is_ensemble=True, 15-seed averaged)
  B) BEST PER-SEED — best per-seed config group (is_ensemble=False, MIN_SEEDS agreement)

Both use all 15 seeds' averaged CSV probabilities for scoring.
Only the hyperparameters (prob_thresh, tp, sl, mode) differ.

Tickers from deploy selection: VOYG, NTLA, CLF, SMR
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
BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")
RESULTS_DIR    = "Algos/Results_all_v2"
INFERENCE_DIR  = "Data/Inference_v2"

TEST_START = '2026-04-01'
TEST_END   = '2026-07-23'
ENTRY_MIN  = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.002
STARTING_CAPITAL = 200_000

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

SHORT_TICKERS = ['VOYG', 'NTLA', 'CLF', 'SMR']

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1')
BQ_ALLOWED_MODES  = ('static', 'atr_sl', 'grace')
MIN_BQ_TRADES = 3
MIN_SEEDS_PER_GROUP = 1
MAX_VA_OPT_RATIO = 3.0

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
# SHORT TRADE SIMULATION
# ==============================================================================
def run_short_trade_static(prices, highs, lows, date_ints, time_mins,
                           entry_idx, tp, sl, exit_time):
    """Static TP/SL — SHORT: TP when price drops, SL when price rises."""
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False

    entry_price  = prices[entry_idx]
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
    """Fixed TP + ATR-based SL — SHORT."""
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
    """Grace period on SL — SHORT: ignore SL for first N minutes."""
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False

    entry_price  = prices[entry_idx]
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
# BQ CONFIG LOADING
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


def _load_bq_rows(is_ensemble_only=False):
    """Shared BQ query + filtering for both strategies."""
    versions_sql = ', '.join(f"'{v}'" for v in BQ_MODEL_VERSIONS)
    tickers_sql  = ', '.join(f"'{t}'" for t in SHORT_TICKERS)
    ens_clause = "AND is_ensemble = TRUE" if is_ensemble_only else ""
    q = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           grace_minutes,
           val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
           val_b_avg_pnl, opt_avg_pnl, is_ensemble, model_version
    FROM `{BQ_TABLE}`
    WHERE model_version IN ({versions_sql})
      {ens_clause}
      AND survived_a = TRUE
      AND val_a_count >= {MIN_BQ_TRADES}
      AND val_a_avg_pnl > 0
      AND ticker IN ({tickers_sql})
    """
    df = bq_client.query(q).to_dataframe()
    if df.empty:
        return df
    df = df[df['opt_avg_pnl'] > 0].copy()
    df = df[df['val_a_avg_pnl'] / df['opt_avg_pnl'] <= MAX_VA_OPT_RATIO]
    df = df[df['tpsl_mode'].isin(BQ_ALLOWED_MODES)]
    return df


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


def load_configs_ensemble():
    """Strategy A: best ensemble BQ row per ticker (is_ensemble=True)."""
    df = _load_bq_rows(is_ensemble_only=True)
    if df.empty:
        print("    No ensemble rows found")
        return {}
    print(f"    {len(df)} ensemble rows after filters")

    configs = {}
    for ticker in SHORT_TICKERS:
        tk = df[df['ticker'] == ticker].copy()
        if tk.empty:
            print(f"    {ticker}: no ensemble rows")
            continue
        best = tk.loc[tk['val_a_avg_pnl'].idxmax()]
        cfg = _row_to_cfg(ticker, best, 'ensemble')
        configs[ticker] = cfg
        print(f"    {ticker}: exp={cfg['experiment']} {cfg['tpsl_mode']} "
              f"prob>{cfg['prob_thresh']:.2f} TP={cfg['tp_pct']:.0%} SL={cfg['sl_pct']:.0%} "
              f"sl_m={cfg['sl_mult']:.1f} val={cfg['val_pnl']:.2%}")
    return configs


def load_configs_best_seed():
    """Strategy B: best per-seed config group per ticker (grouped, MIN_SEEDS filter)."""
    df = _load_bq_rows(is_ensemble_only=False)
    if df.empty:
        print("    No BQ rows found")
        return {}
    indiv = df[df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = df[df['is_ensemble'] != True].copy()
    print(f"    {len(indiv)} per-seed rows after filters")

    grp_keys = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
                'prob_thresh', 'tp_pct', 'sl_pct', 'sl_mult', 'grace_minutes']
    present_keys = [k for k in grp_keys if k in indiv.columns]
    for k in present_keys:
        if indiv[k].isna().any():
            indiv[k] = indiv[k].fillna(0)

    sa = indiv.groupby(present_keys).agg(
        n_seeds=('seed', 'nunique'),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        vb_pnl=('val_b_avg_pnl', 'mean'),
    ).reset_index()
    sa = sa[sa['n_seeds'] >= MIN_SEEDS_PER_GROUP]
    print(f"    {len(sa)} groups with >={MIN_SEEDS_PER_GROUP} seeds")

    configs = {}
    for ticker in SHORT_TICKERS:
        tk = sa[sa['ticker'] == ticker].copy()
        if tk.empty:
            print(f"    {ticker}: no qualifying seed groups")
            continue
        best = tk.loc[tk['mean_va_pnl'].idxmax()]
        cfg = _row_to_cfg(ticker, best, 'best_seed')
        cfg['n_seeds'] = int(best['n_seeds'])
        cfg['val_pnl'] = _safe_float(best['mean_va_pnl'])
        cfg['test_pnl'] = _safe_float(best['vb_pnl'])
        configs[ticker] = cfg
        print(f"    {ticker}: exp={cfg['experiment']} {cfg['tpsl_mode']} "
              f"prob>{cfg['prob_thresh']:.2f} TP={cfg['tp_pct']:.0%} SL={cfg['sl_pct']:.0%} "
              f"sl_m={cfg['sl_mult']:.1f} val={cfg['val_pnl']:.2%} ({cfg['n_seeds']} seeds)")
    return configs


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
# BACKTEST FUNCTION — runs trade simulation with given configs
# ==============================================================================
def run_csv_backtest(configs, csv_scores_cache, ticker_data, label):
    """Run SHORT backtest with given configs. Returns (all_trades, per_ticker_trades)."""
    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt   = pd.Timestamp(TEST_END).date()

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
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT MODEL BACKTEST — Ensemble vs Best-Seed Config Comparison")
print(f"  TEST: {TEST_START} → {TEST_END}")
print(f"  TICKERS: {', '.join(sorted(SHORT_TICKERS))}")
print(f"  ENTRY: 6:15 PST ({ENTRY_MIN} min)  |  EXIT: 1:00 PM ({EXIT_TIME_MIN} min)")
print(f"  SLIPPAGE: {SLIPPAGE_PCT:.2%} each way  |  CAPITAL: ${STARTING_CAPITAL:,}")
print(f"{'='*W}")


# ── Step 1: Load BQ configs — both strategies ──
print(f"\n  STEP 1A: Loading ENSEMBLE configs...")
configs_ens = load_configs_ensemble()
print(f"  → {len(configs_ens)} tickers (ensemble)")

print(f"\n  STEP 1B: Loading BEST PER-SEED configs...")
configs_seed = load_configs_best_seed()
print(f"  → {len(configs_seed)} tickers (best_seed)")

all_tickers = sorted(set(list(configs_ens.keys()) + list(configs_seed.keys())))
if not all_tickers:
    raise RuntimeError("No deployable configs found for either strategy")


# ── Step 2: Load OHLCV data (shared) ──
print(f"\n  STEP 2: Loading OHLCV + ATR data...")
ticker_data = {}
for ticker in all_tickers:
    df = load_master(ticker)
    if df is None:
        continue
    df = heal_data(df)
    ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
    if ohlcv is None:
        print(f"    {ticker}: no data in test period")
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

    ticker_data[ticker] = {'ohlcv': ohlcv, 'df': df}
    n_days = len(np.unique(ohlcv['dates']))
    print(f"    {ticker}: {n_days} trading days, ATR_pct={atr_cut.median():.4f}")
    del df
    gc.collect()


# ── Step 3: Load CSV scores for all experiment × ticker combos ──
print(f"\n  STEP 3: Loading CSV scores from GCS...")
csv_scores_cache = {}
all_exp_ticker_pairs = set()
for cfgs in [configs_ens, configs_seed]:
    for ticker, cfg in cfgs.items():
        all_exp_ticker_pairs.add((ticker, cfg['_base_experiment']))

for ticker, base_exp in sorted(all_exp_ticker_pairs):
    if ticker not in ticker_data:
        continue
    seed_scores = load_seed_scores(ticker, base_exp)
    if seed_scores:
        csv_scores_cache[(ticker, base_exp)] = seed_scores
        print(f"    {ticker}/{base_exp}: {len(seed_scores)} seeds loaded")
    else:
        print(f"    {ticker}/{base_exp}: no CSV scores")
    gc.collect()


# ── Step 4: Run both backtests ──
print(f"\n  STEP 4A: Running ENSEMBLE backtest...")
np.random.seed(42)
trades_ens, per_tk_ens = run_csv_backtest(configs_ens, csv_scores_cache, ticker_data, 'Ensemble')

print(f"\n  STEP 4B: Running BEST PER-SEED backtest...")
np.random.seed(42)
trades_seed, per_tk_seed = run_csv_backtest(configs_seed, csv_scores_cache, ticker_data, 'Best-Seed')


# ── Step 5: Results comparison ──
port_ens  = summarize_trades(trades_ens, 'Ensemble')
port_seed = summarize_trades(trades_seed, 'Best-Seed')

print(f"\n{'='*W}")
print(f"  COMPARISON — Ensemble vs Best Per-Seed Config ({TEST_START} → {TEST_END})")
print(f"{'='*W}")

print(f"\n  {'Strategy':<12}  {'Trades':>6}  {'Total':>8}  {'Avg':>7}  {'WR':>5}  {'MaxDD':>7}")
print(f"  {'─'*55}")
for p in [port_ens, port_seed]:
    print(f"  {p['label']:<12}  {p['n']:>6}  {p['total']:>+8.1%}  "
          f"{p['avg']:>+7.2%}  {p['wr']:>5.0%}  {p['dd']:>+7.2%}")

# ── Config comparison ──
print(f"\n  Config Comparison:")
print(f"  {'Ticker':<7} │ {'── Ensemble ─────────────────────────':} │ {'── Best-Seed ────────────────────────':}")
print(f"  {'':7} │ {'α':>4} {'Mode':<8} {'Prob>':>6} {'TP':>5} {'SL':>5} {'slM':>4} │ "
      f"{'α':>4} {'Mode':<8} {'Prob>':>6} {'TP':>5} {'SL':>5} {'slM':>4} {'#S':>3}")
print(f"  {'─'*95}")
for ticker in sorted(all_tickers):
    e = configs_ens.get(ticker)
    s = configs_seed.get(ticker)
    e_str = (f"{e['alpha']:>4} {e['tpsl_mode']:<8} {e['prob_thresh']:>6.2f} "
             f"{e['tp_pct']:>4.0%} {e['sl_pct']:>4.0%} {e['sl_mult']:>4.1f}") if e else f"{'—':>40}"
    s_str = (f"{s['alpha']:>4} {s['tpsl_mode']:<8} {s['prob_thresh']:>6.2f} "
             f"{s['tp_pct']:>4.0%} {s['sl_pct']:>4.0%} {s['sl_mult']:>4.1f} "
             f"{s.get('n_seeds',''):>3}") if s else f"{'—':>44}"
    same = " =" if (e and s and e['prob_thresh'] == s['prob_thresh']
                    and e['tp_pct'] == s['tp_pct'] and e['sl_pct'] == s['sl_pct']
                    and e['tpsl_mode'] == s['tpsl_mode']
                    and e['alpha'] == s['alpha']) else ""
    print(f"  {ticker:<7} │ {e_str} │ {s_str}{same}")

# ── Per-ticker results comparison ──
print(f"\n  Per-Ticker Results:")
print(f"  {'Ticker':<7} │ {'── Ensemble ──────────────':} │ {'── Best-Seed ─────────────':}")
print(f"  {'':7} │ {'N':>4}  {'Total':>8}  {'Avg':>7}  {'WR':>5} │ "
      f"{'N':>4}  {'Total':>8}  {'Avg':>7}  {'WR':>5} │ {'Winner':>7}")
print(f"  {'─'*85}")
for ticker in sorted(all_tickers):
    e_trades = per_tk_ens.get(ticker, [])
    s_trades = per_tk_seed.get(ticker, [])
    e_n = len(e_trades)
    s_n = len(s_trades)
    e_total = sum(t['pnl'] for t in e_trades) if e_n > 0 else 0
    s_total = sum(t['pnl'] for t in s_trades) if s_n > 0 else 0
    e_avg = e_total / e_n if e_n > 0 else 0
    s_avg = s_total / s_n if s_n > 0 else 0
    e_wr = sum(1 for t in e_trades if t['pnl'] > 0) / e_n if e_n > 0 else 0
    s_wr = sum(1 for t in s_trades if t['pnl'] > 0) / s_n if s_n > 0 else 0
    winner = 'ENS' if e_avg > s_avg else ('SEED' if s_avg > e_avg else 'TIE')
    print(f"  {ticker:<7} │ {e_n:>4}  {e_total:>+8.1%}  {e_avg:>+7.2%}  {e_wr:>5.0%} │ "
          f"{s_n:>4}  {s_total:>+8.1%}  {s_avg:>+7.2%}  {s_wr:>5.0%} │ {winner:>7}")

# ── Monthly comparison ──
all_months = sorted(set(list(port_ens['monthly'].keys()) + list(port_seed['monthly'].keys())))
if all_months:
    print(f"\n  Monthly Comparison:")
    print(f"  {'Month':<8} │ {'── Ensemble ──────────':} │ {'── Best-Seed ─────────':}")
    print(f"  {'':8} │ {'N':>4}  {'Total':>8}  {'Avg':>7} │ {'N':>4}  {'Total':>8}  {'Avg':>7}")
    print(f"  {'─'*65}")
    for month in all_months:
        e_m = port_ens['monthly'].get(month, {'n': 0, 'total': 0, 'avg': 0})
        s_m = port_seed['monthly'].get(month, {'n': 0, 'total': 0, 'avg': 0})
        print(f"  {month:<8} │ {e_m['n']:>4}  {e_m['total']:>+8.1%}  {e_m['avg']:>+7.2%} │ "
              f"{s_m['n']:>4}  {s_m['total']:>+8.1%}  {s_m['avg']:>+7.2%}")


# ── Step 6: Charts ──
best_label = 'Ensemble' if port_ens['avg'] >= port_seed['avg'] else 'Best-Seed'
best_trades = trades_ens if best_label == 'Ensemble' else trades_seed
best_port = port_ens if best_label == 'Ensemble' else port_seed
best_per_tk = per_tk_ens if best_label == 'Ensemble' else per_tk_seed
best_configs = configs_ens if best_label == 'Ensemble' else configs_seed

if trades_ens or trades_seed:
    print(f"\n  Generating comparison charts...")

    TICKER_COLORS = {
        'SMR': '#2196F3', 'VOYG': '#4CAF50', 'NTLA': '#FF9800',
        'CLF': '#E91E63', 'MVIS': '#9C27B0', 'SOUN': '#00BCD4',
        'MARA': '#795548',
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # ── Chart 1: Cumulative PnL comparison ──
    ax1 = axes[0, 0]
    for trades, label, color, ls in [(trades_ens, 'Ensemble', '#1565C0', '-'),
                                      (trades_seed, 'Best-Seed', '#E65100', '--')]:
        if not trades:
            continue
        df_t = pd.DataFrame(trades).sort_values('date')
        cum = df_t['pnl'].cumsum() * 100
        ax1.plot(pd.to_datetime(df_t['date'].values), cum.values,
                 color=color, linewidth=2.5, linestyle=ls, label=label)
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax1.set_ylabel('Cumulative PnL (%)')
    ax1.set_title('Ensemble vs Best-Seed — Cumulative PnL', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # ── Chart 2: Per-Ticker Cumulative PnL (winning strategy) ──
    ax2 = axes[0, 1]
    for ticker in sorted(best_per_tk.keys()):
        trades = best_per_tk[ticker]
        if not trades:
            continue
        df_t = pd.DataFrame(trades).sort_values('date')
        cum = df_t['pnl'].cumsum() * 100
        ax2.plot(pd.to_datetime(df_t['date'].values), cum.values,
                 label=ticker, color=TICKER_COLORS.get(ticker, '#666'), linewidth=2)
    ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax2.set_ylabel('Cumulative PnL (%)')
    ax2.set_title(f'Per-Ticker ({best_label})', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # ── Chart 3: Equity curves comparison ──
    ax3 = axes[1, 0]
    for trades, per_tk, label, color, ls in [
        (trades_ens, per_tk_ens, 'Ensemble', '#1565C0', '-'),
        (trades_seed, per_tk_seed, 'Best-Seed', '#E65100', '--')]:
        if not trades:
            continue
        df_s = pd.DataFrame(trades).sort_values('date')
        n_tk = max(len(per_tk), 1)
        eq = STARTING_CAPITAL
        eq_vals, eq_dates = [], []
        for _, row in df_s.iterrows():
            eq += (eq / n_tk) * row['pnl']
            eq_vals.append(eq)
            eq_dates.append(row['date'])
        ax3.plot(pd.to_datetime(eq_dates), np.array(eq_vals) / 1000,
                 color=color, linewidth=2.5, linestyle=ls, label=label)
    ax3.axhline(y=STARTING_CAPITAL / 1000, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax3.set_ylabel('Equity ($K)')
    ax3.set_title(f'Equity Comparison (${STARTING_CAPITAL/1000:.0f}K start)',
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # ── Chart 4: Per-ticker avg PnL comparison (grouped bar) ──
    ax4 = axes[1, 1]
    x = np.arange(len(all_tickers))
    bar_w = 0.35
    ens_avgs = [np.mean([t['pnl'] for t in per_tk_ens.get(tk, [])]) * 100
                if per_tk_ens.get(tk) else 0 for tk in all_tickers]
    seed_avgs = [np.mean([t['pnl'] for t in per_tk_seed.get(tk, [])]) * 100
                 if per_tk_seed.get(tk) else 0 for tk in all_tickers]
    ax4.bar(x - bar_w/2, ens_avgs, bar_w, label='Ensemble', color='#1565C0', alpha=0.85)
    ax4.bar(x + bar_w/2, seed_avgs, bar_w, label='Best-Seed', color='#E65100', alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(all_tickers, fontsize=11)
    ax4.set_ylabel('Avg PnL per Trade (%)')
    ax4.set_title('Per-Ticker Avg PnL: Ensemble vs Best-Seed', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.axhline(y=0, color='gray', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'SHORT Model — Ensemble vs Best-Seed  |  {TEST_START} → {TEST_END}\n'
                 f'ENS: {port_ens["n"]}t Avg={port_ens["avg"]:+.2%}  |  '
                 f'SEED: {port_seed["n"]}t Avg={port_seed["avg"]:+.2%}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/tmp/backtest_short_csv.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Chart saved: /tmp/backtest_short_csv.png")


# ── Trade logs ──
for label, trades in [('Ensemble', trades_ens), ('Best-Seed', trades_seed)]:
    if trades:
        print(f"\n  {label} trades ({len(trades)}):")
        df_log = pd.DataFrame(trades).sort_values('date')
        for _, t in df_log.iterrows():
            print(f"    {t['date']}  {t['ticker']:<7}  pnl={t['pnl']:+.2%}  "
                  f"prob={t['prob']:.3f}  close=${t['close']:.2f}")

winner = 'Ensemble' if port_ens['avg'] >= port_seed['avg'] else 'Best-Seed'
print(f"\n{'='*W}")
print(f"  WINNER: {winner}")
print(f"  Ensemble:  {port_ens['n']} trades, Avg={port_ens['avg']:+.2%}, WR={port_ens['wr']:.0%}")
print(f"  Best-Seed: {port_seed['n']} trades, Avg={port_seed['avg']:+.2%}, WR={port_seed['wr']:.0%}")
print(f"{'='*W}")
