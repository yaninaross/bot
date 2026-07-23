"""
backtest_short_csv.py — CSV-Scored SHORT Model Backtest (Exp 400/410)
======================================================================
Backtests SHORT/dump model using pre-computed CSV scores from GCS.
Top 6 tickers from cross-alpha selection:
  SMR, MVIS, NTLA, SOUN, MP, BBAI

Loads configs from BQ (robust_wf_v12_short / robust_wf_v12_short_a1).
Supports static and atr_sl modes (SHORT trade direction).
Generates: cumulative PnL, equity curves, monthly breakdown, per-ticker charts.

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

TEST_START = '2026-04-01'
TEST_END   = '2026-07-22'
ENTRY_MIN  = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.002
STARTING_CAPITAL = 200_000

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

SHORT_TICKERS = {
    'SMR':  {'alpha': '400', 'experiment': '400_XSECTOR'},
    'MVIS': {'alpha': '400', 'experiment': '400_XSECTOR'},
    'NTLA': {'alpha': '400', 'experiment': '400_XSECTOR'},
    'SOUN': {'alpha': '400', 'experiment': '400_XSECTOR'},
    'BBAI': {'alpha': '400', 'experiment': '400_XSECTOR'},
    'MP':   {'alpha': '410', 'experiment': '410_XSECTOR'},
}

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1')
BQ_ALLOWED_MODES  = ('static', 'atr_sl', 'grace')
MIN_SEEDS    = 3
MIN_BQ_TRADES = 3
DEPLOY_THRESHOLD = 0.018
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
def load_short_configs():
    versions_sql = ', '.join(f"'{v}'" for v in BQ_MODEL_VERSIONS)
    tickers_sql  = ', '.join(f"'{t}'" for t in SHORT_TICKERS)
    q = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           grace_minutes,
           val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
           val_b_avg_pnl, opt_avg_pnl, is_ensemble, model_version
    FROM `{BQ_TABLE}`
    WHERE model_version IN ({versions_sql})
      AND survived_a = TRUE
      AND val_a_count >= {MIN_BQ_TRADES}
      AND val_a_avg_pnl > 0
      AND ticker IN ({tickers_sql})
    """
    df = bq_client.query(q).to_dataframe()
    if df.empty:
        raise RuntimeError("No BQ rows found for short model tickers")
    print(f"    {len(df)} raw BQ rows loaded")

    df = df[df['opt_avg_pnl'] > 0].copy()
    df = df[df['val_a_avg_pnl'] / df['opt_avg_pnl'] <= MAX_VA_OPT_RATIO]
    print(f"    {len(df)} after overfit filter")

    modes_sql = ', '.join(f"'{m}'" for m in BQ_ALLOWED_MODES)
    df = df[df['tpsl_mode'].isin(BQ_ALLOWED_MODES)]
    print(f"    {len(df)} after mode filter ({', '.join(BQ_ALLOWED_MODES)})")

    configs = {}
    for ticker, info in SHORT_TICKERS.items():
        exp_base = info['experiment']
        alpha = info['alpha']

        mv = f'robust_wf_v12_short_a1' if alpha == '410' else 'robust_wf_v12_short'
        tk_df = df[(df['ticker'] == ticker)].copy()

        if tk_df.empty:
            print(f"    {ticker}: no configs found")
            continue

        ens = tk_df[tk_df['is_ensemble'] == True].copy()
        indiv = tk_df[tk_df['is_ensemble'] == False].copy()

        best_row = None
        source = None

        if not ens.empty:
            ens['score'] = (ens['val_a_avg_pnl'] ** 2)
            best_ens = ens.loc[ens['score'].idxmax()]
            best_row = best_ens
            source = 'ensemble'

        if not indiv.empty:
            grp_keys = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
                        'prob_thresh', 'tp_pct', 'sl_pct', 'tp_mult', 'sl_mult',
                        'grace_minutes']
            sa = indiv.groupby([k for k in grp_keys if k in indiv.columns]).agg(
                n_seeds=('seed', 'nunique'),
                mean_va_pnl=('val_a_avg_pnl', 'mean'),
                mean_va_wr=('val_a_win_rate', 'mean'),
                vb_pnl=('val_b_avg_pnl', 'mean'),
            ).reset_index()
            sa = sa[sa['n_seeds'] >= MIN_SEEDS]
            if not sa.empty:
                sa['score'] = (sa['mean_va_pnl'] ** 2)
                best_sa = sa.loc[sa['score'].idxmax()]
                if best_row is None or best_sa['score'] > best_row.get('score', 0):
                    best_row = best_sa
                    source = 'per_seed'

        if best_row is None:
            print(f"    {ticker}: no qualifying config")
            continue

        mode = best_row['tpsl_mode']
        cfg = {
            'ticker': ticker,
            'experiment': exp_base,
            'alpha': alpha,
            'source': source,
            'tpsl_mode': mode,
            'entry_minute': int(best_row['entry_minute']),
            'prob_thresh': float(best_row['prob_thresh']),
            'tp_pct': float(best_row['tp_pct']),
            'sl_pct': float(best_row['sl_pct']),
            'sl_mult': float(best_row.get('sl_mult', 0) or 0),
            'grace_minutes': int(best_row.get('grace_minutes', 0) or 0),
            'val_pnl': float(best_row.get('mean_va_pnl', best_row.get('val_a_avg_pnl', 0))),
            'test_pnl': float(best_row.get('vb_pnl', best_row.get('val_b_avg_pnl', 0))),
        }
        configs[ticker] = cfg
        print(f"    {ticker}: α={alpha} {mode} prob>{cfg['prob_thresh']:.2f} "
              f"TP={cfg['tp_pct']:.0%} SL={cfg['sl_pct']:.0%} sl_m={cfg['sl_mult']:.1f} "
              f"({source})")

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
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT MODEL BACKTEST — CSV Scores (Exp 400/410)")
print(f"  TEST: {TEST_START} → {TEST_END}")
print(f"  TICKERS: {', '.join(sorted(SHORT_TICKERS.keys()))}")
print(f"  ENTRY: 6:15 PST ({ENTRY_MIN} min)  |  EXIT: 1:00 PM ({EXIT_TIME_MIN} min)")
print(f"  SLIPPAGE: {SLIPPAGE_PCT:.2%} each way  |  CAPITAL: ${STARTING_CAPITAL:,}")
print(f"{'='*W}")


# ── Step 1: Load BQ configs ──
print(f"\n  STEP 1: Loading BQ configs for short model...")
deploy_configs = load_short_configs()
if not deploy_configs:
    raise RuntimeError("No deployable configs found")
print(f"  → {len(deploy_configs)} tickers configured")


# ── Step 2: Load OHLCV data ──
print(f"\n  STEP 2: Loading OHLCV + ATR data...")
ticker_data = {}
for ticker in sorted(deploy_configs.keys()):
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


# ── Step 3: Load CSV scores ──
print(f"\n  STEP 3: Loading CSV scores from GCS...")
daily_probs = {}
for ticker, cfg in sorted(deploy_configs.items()):
    if ticker not in ticker_data:
        continue
    exp_base = cfg['experiment']
    entry_min = cfg['entry_minute']

    seed_scores = load_seed_scores(ticker, exp_base)
    if not seed_scores:
        print(f"    {ticker}: no CSV scores for {exp_base}")
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

    avg_probs = {d: np.mean(ps) for d, ps in all_daily.items()}
    daily_probs[ticker] = avg_probs
    n_days_scored = len(avg_probs)
    print(f"    {ticker}: {len(seed_scores)} seeds, {n_days_scored} days scored, "
          f"mean prob={np.mean(list(avg_probs.values())):.3f}")
    del seed_scores
    gc.collect()


# ── Step 4: Simulate SHORT trades ──
print(f"\n  STEP 4: Simulating SHORT trades...")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()

all_trades = []
per_ticker_trades = defaultdict(list)
n_scored = 0
n_signals = 0
n_filled = 0

for ticker in sorted(deploy_configs.keys()):
    if ticker not in ticker_data or ticker not in daily_probs:
        continue
    cfg = deploy_configs[ticker]
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
        n_scored += 1

        if prob < prob_thresh:
            continue
        n_signals += 1

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
        n_filled += 1

        pnl -= 2 * SLIPPAGE_PCT
        last_traded_date = d_int

        trade = {
            'date': d_date, 'ticker': ticker, 'pnl': pnl,
            'prob': prob, 'close': float(prices[idx]),
            'mode': mode, 'tp': cfg['tp_pct'],
            'sl': cfg['sl_pct'] if mode == 'static' else cfg['sl_mult'],
        }
        all_trades.append(trade)
        per_ticker_trades[ticker].append(trade)

    tk_n = len(per_ticker_trades[ticker])
    if tk_n > 0:
        tk_pnl = sum(t['pnl'] for t in per_ticker_trades[ticker])
        print(f"    {ticker}: {tk_n} trades, total={tk_pnl:+.1%}, "
              f"avg={tk_pnl/tk_n:+.2%}")


# ── Step 5: Results ──
print(f"\n{'='*W}")
print(f"  RESULTS — SHORT MODEL BACKTEST ({TEST_START} → {TEST_END})")
print(f"{'='*W}")
print(f"  Scored: {n_scored}  |  Signals: {n_signals}  |  Filled: {n_filled}")

portfolio = summarize_trades(all_trades, 'Portfolio')

print(f"\n  {'Strategy':<12}  {'Trades':>6}  {'Total':>8}  {'Avg':>7}  {'WR':>5}  {'MaxDD':>7}")
print(f"  {'─'*55}")
print(f"  {'Portfolio':<12}  {portfolio['n']:>6}  {portfolio['total']:>+8.1%}  "
      f"{portfolio['avg']:>+7.2%}  {portfolio['wr']:>5.0%}  {portfolio['dd']:>+7.2%}")

print(f"\n  Monthly Breakdown:")
print(f"  {'Month':<8}  {'N':>4}  {'Total':>8}  {'Avg':>7}  {'WR':>5}")
print(f"  {'─'*40}")
for month in sorted(portfolio['monthly'].keys()):
    m = portfolio['monthly'][month]
    print(f"  {month:<8}  {m['n']:>4}  {m['total']:>+8.1%}  {m['avg']:>+7.2%}  {m['wr']:>5.0%}")

print(f"\n  Per-Ticker:")
print(f"  {'Ticker':<7} {'Mode':<8} {'α':>4} {'N':>4}  {'Total':>8}  {'Avg':>7}  {'WR':>5}  {'Prob>':>6}")
print(f"  {'─'*60}")
ticker_stats = {}
for ticker in sorted(per_ticker_trades.keys()):
    trades = per_ticker_trades[ticker]
    if not trades:
        continue
    cfg = deploy_configs[ticker]
    df_t = pd.DataFrame(trades)
    total = df_t['pnl'].sum()
    avg = df_t['pnl'].mean()
    wr = (df_t['pnl'] > 0).mean()
    ticker_stats[ticker] = {'n': len(trades), 'total': total, 'avg': avg, 'wr': wr}
    print(f"  {ticker:<7} {cfg['tpsl_mode']:<8} {cfg['alpha']:>4} {len(trades):>4}  "
          f"{total:>+8.1%}  {avg:>+7.2%}  {wr:>5.0%}  {cfg['prob_thresh']:>6.2f}")


# ── Step 6: Charts ──
if not all_trades:
    print("\n  No trades — skipping charts.")
else:
    print(f"\n  Generating charts...")

    TICKER_COLORS = {
        'SMR': '#2196F3', 'MVIS': '#4CAF50', 'NTLA': '#FF9800',
        'SOUN': '#E91E63', 'BBAI': '#9C27B0', 'MP': '#00BCD4',
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # ── Chart 1: Cumulative PnL (%) — Portfolio ──
    ax1 = axes[0, 0]
    df_all = pd.DataFrame(all_trades).sort_values('date')
    cum_pnl = df_all['pnl'].cumsum() * 100
    dates_all = pd.to_datetime(df_all['date'].values)
    ax1.plot(dates_all, cum_pnl.values, color='#333333', linewidth=2.5, label='Portfolio')
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax1.fill_between(dates_all, cum_pnl.values, 0,
                     where=cum_pnl.values >= 0, alpha=0.1, color='green')
    ax1.fill_between(dates_all, cum_pnl.values, 0,
                     where=cum_pnl.values < 0, alpha=0.1, color='red')
    ax1.set_ylabel('Cumulative PnL (%)')
    ax1.set_title('SHORT Portfolio — Cumulative PnL', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # ── Chart 2: Per-Ticker Cumulative PnL ──
    ax2 = axes[0, 1]
    for ticker in sorted(per_ticker_trades.keys()):
        trades = per_ticker_trades[ticker]
        if not trades:
            continue
        df_t = pd.DataFrame(trades).sort_values('date')
        cum = df_t['pnl'].cumsum() * 100
        ax2.plot(pd.to_datetime(df_t['date'].values), cum.values,
                 label=ticker, color=TICKER_COLORS.get(ticker, '#666'),
                 linewidth=2)
    ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax2.set_ylabel('Cumulative PnL (%)')
    ax2.set_title('Per-Ticker Cumulative PnL', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # ── Chart 3: Equity Curve ($) ──
    ax3 = axes[1, 0]
    df_sorted = pd.DataFrame(all_trades).sort_values('date')
    n_tickers = len(per_ticker_trades)
    equity = STARTING_CAPITAL
    equity_vals = []
    equity_dates = []
    for _, row in df_sorted.iterrows():
        pos_size = equity / max(n_tickers, 1)
        equity += pos_size * row['pnl']
        equity_vals.append(equity)
        equity_dates.append(row['date'])
    ax3.plot(pd.to_datetime(equity_dates), np.array(equity_vals) / 1000,
             color='#333333', linewidth=2.5)
    ax3.axhline(y=STARTING_CAPITAL / 1000, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax3.set_ylabel('Equity ($K)')
    ax3.set_title(f'Portfolio Equity (${STARTING_CAPITAL/1000:.0f}K start, {n_tickers} tickers)',
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # ── Chart 4: Monthly Avg PnL per Ticker (grouped bar) ──
    ax4 = axes[1, 1]
    months = sorted(portfolio['monthly'].keys())
    tickers_with_trades = sorted(per_ticker_trades.keys())
    x = np.arange(len(months))
    bar_width = 0.8 / max(len(tickers_with_trades), 1)

    for i, ticker in enumerate(tickers_with_trades):
        df_t = pd.DataFrame(per_ticker_trades[ticker])
        df_t['month'] = pd.to_datetime(df_t['date']).dt.strftime('%Y-%m')
        monthly_avg = df_t.groupby('month')['pnl'].mean()
        vals = [monthly_avg.get(m, 0) * 100 for m in months]
        ax4.bar(x + i * bar_width, vals, bar_width,
                label=ticker, color=TICKER_COLORS.get(ticker, '#666'), alpha=0.85)

    mlabels = [pd.Timestamp(m + '-01').strftime('%b') for m in months]
    ax4.set_xticks(x + bar_width * len(tickers_with_trades) / 2)
    ax4.set_xticklabels(mlabels, fontsize=11)
    ax4.set_ylabel('Avg PnL per Trade (%)')
    ax4.set_title('Monthly Avg PnL by Ticker', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=8, ncol=2)
    ax4.axhline(y=0, color='gray', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'SHORT Model Backtest — {TEST_START} → {TEST_END}  |  '
                 f'{portfolio["n"]} trades, Avg={portfolio["avg"]:+.2%}, WR={portfolio["wr"]:.0%}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('/tmp/backtest_short_csv.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Chart saved: /tmp/backtest_short_csv.png")


# ── Trade log ──
if all_trades:
    print(f"\n  All trades ({len(all_trades)}):")
    df_log = pd.DataFrame(all_trades).sort_values('date')
    for _, t in df_log.iterrows():
        print(f"    {t['date']}  {t['ticker']:<7}  pnl={t['pnl']:+.2%}  "
              f"prob={t['prob']:.3f}  close=${t['close']:.2f}")

print(f"\n{'='*W}")
print(f"  DONE — {len(all_trades)} trades across {len(per_ticker_trades)} tickers")
print(f"{'='*W}")
