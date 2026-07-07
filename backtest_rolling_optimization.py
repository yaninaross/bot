"""
backtest_rolling_optimization.py — Rolling WF Optimization Backtest
====================================================================
Tests whether weekly re-optimization on recent data would have improved
June 2026 performance vs the current static optimization approach.

Simulates:
  - Every REOPT_FREQ trading days, re-run a simplified optimizer on the
    last LOOKBACK_DAYS of data to pick best (threshold, TP, SL) per ticker
  - Deploy those configs for the next REOPT_FREQ days
  - Compare across lookback windows: 21, 42, 63 trading days (~1, 2, 3 months)
  - Baseline: static configs (best from full Apr 2025 - Mar 2026 window)

Step 0: Check score CSV date coverage for June
Step 1: Load score CSVs + OHLCV for all tickers
Step 2: Pre-compute daily trade signals + intraday price paths
Step 3: Rolling optimization simulation
Step 4: Results comparison

Colab-ready (no if __name__). Uses limit_entry mode only (S3 strategy).
"""

import os, io, gc, sys, subprocess, warnings, ctypes
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import product
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "google-cloud-storage", "pyarrow", "gcsfs", "--quiet"])
    from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"
RESULTS_DIR = "Algos/Results_all_v2"
OHLCV_DIR = "Data/Inference"
INFERENCE_DIR = "Data/Inference_v2"

PST = pytz.timezone('America/Los_Angeles')
SLIPPAGE_PCT = 0.0015
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

SCORE_EXPERIMENTS = [152, 160, 175, 185, 200]
TEACHERS_13 = ['BBAI','RCAT','LUNR','ASTS','ACHR','MVIS','WULF','CIFR',
               'IREN','KEEL','HUT','CLSK','APLD']
TEACHERS_31 = ['CIFR','IREN','KEEL','HUT','APLD','WULF','RIOT','CLSK',
               'BTBT','ALAB','MARA','RCAT','ASTS','LUNR','BBAI','RKLB',
               'SPIR','BKSY','MNTS','PLUG','BE','FCEL','FLNC','OKLO',
               'SMR','MVIS','BGDE','RDW','PL','VOYG','SMLR']
EXP_TEACHERS = {
    152: TEACHERS_13, 160: TEACHERS_13, 175: TEACHERS_13,
    185: TEACHERS_31, 200: TEACHERS_31,
}
DEPLOY_TICKERS = ['HUT','FLNC','CIFR','CLSK','KEEL','RCAT','RIOT','APLD','WULF']
EXCLUDE_TICKERS = {'PL', 'SMLR'}

# ── Rolling optimization config ──
DATA_START = '2025-12-01'
TEST_START = '2026-04-01'
TEST_END   = '2026-06-29'

LOOKBACK_OPTIONS = [21, 42, 63]
REOPT_FREQ = 5

# ── Optimizer grid (limit_entry mode, static TP/SL) ──
PROB_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
TP_GRID   = [0.06, 0.10, 0.16, 0.22]
SL_GRID   = [0.03, 0.06, 0.10, 0.14]
ENTRY_TIMES = ['06:00', '06:10', '06:20']

EXIT_TIME_MIN = 780
MIN_OPT_TRADES = 3
MIN_OPT_PNL = 0.005

# ── Family agreement (S3 strategy) ──
EXP_FAMILIES = {152: 'F152', 160: 'F160', 175: 'F175', 185: 'F185', 200: 'F200'}
MIN_FAMILY_AGREE = 3

gcs_client = gcs_storage.Client()


# ==============================================================================
# HELPERS
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df


def load_ohlcv(ticker):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for dir_path in [INFERENCE_DIR, OHLCV_DIR]:
        for suffix in ['_master_inference.parquet', '_1m_full_historical_data.parquet',
                        '_ohlcv_1min_pst.parquet']:
            uri = f"gs://{BUCKET_NAME}/{dir_path}/{clean}{suffix}"
            try:
                df = pd.read_parquet(uri)
                df = df.reset_index()
                dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                if dt_cols:
                    df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
                else:
                    df['Datetime'] = pd.to_datetime(df[df.columns[0]], unit='ms', utc=True)
                df = df.set_index('Datetime').sort_index()
                df = df[~df.index.duplicated(keep='first')]
                std_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
                lower_map = {k.lower():v for k,v in std_map.items()}
                df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
                start = pd.Timestamp(DATA_START, tz='UTC') - pd.Timedelta(days=5)
                end = pd.Timestamp(TEST_END, tz='UTC') + pd.Timedelta(days=2)
                df = df[(df.index >= start) & (df.index <= end)]
                df = heal_data(df)
                df.index = df.index.tz_convert(PST)
                return df
            except Exception:
                continue
    return None


def load_seed_scores(ticker, experiment):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    prefix = f"{RESULTS_DIR}/Universal_Exp{experiment}_XSECTOR/{clean}/Seed_"
    blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=prefix))
    results = {}
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
        results[seed_id] = d
    return results


def simulate_trade(ohlcv_day, entry_min, tp_pct, sl_pct, exit_min=EXIT_TIME_MIN):
    """Simulate a single trade on 1-min bars. Returns PnL as fraction or None."""
    if ohlcv_day is None or ohlcv_day.empty:
        return None
    pst_mins = ohlcv_day.index.hour * 60 + ohlcv_day.index.minute
    entry_mask = pst_mins >= entry_min
    after = ohlcv_day[entry_mask]
    if len(after) < 5 or 'Close' not in after.columns:
        return None

    entry_price = after.iloc[0]['Close']
    if entry_price <= 0:
        return None
    target_price = entry_price * (1 + tp_pct)
    stop_price = entry_price * (1 - sl_pct)

    for i in range(1, len(after)):
        bar = after.iloc[i]
        bar_min = after.index[i].hour * 60 + after.index[i].minute
        high = bar.get('High', bar['Close'])
        low = bar.get('Low', bar['Close'])

        sl_hit = low <= stop_price
        tp_hit = high >= target_price

        if sl_hit and tp_hit:
            return tp_pct if np.random.random() > 0.5 else -sl_pct
        if sl_hit:
            return -sl_pct
        if tp_hit:
            return tp_pct
        if bar_min >= exit_min:
            return (bar['Close'] - entry_price) / entry_price

    last_close = after.iloc[-1]['Close']
    return (last_close - entry_price) / entry_price


# ==============================================================================
# STEP 0: CHECK SCORE CSV COVERAGE
# ==============================================================================
W = 120
print("=" * W)
print(f"  ROLLING OPTIMIZATION BACKTEST")
print("=" * W)

print(f"\n  STEP 0: Checking score CSV date coverage for June 2026...")
coverage = {}
for exp in SCORE_EXPERIMENTS:
    test_ticker = DEPLOY_TICKERS[0]
    seeds = load_seed_scores(test_ticker, exp)
    if seeds:
        first_seed = list(seeds.values())[0]
        min_d = first_seed.index.min().date()
        max_d = first_seed.index.max().date()
        covers_june = max_d >= pd.Timestamp('2026-06-01').date()
        coverage[exp] = {'min': min_d, 'max': max_d, 'n_seeds': len(seeds),
                         'covers_june': covers_june}
        status = "✓ COVERS JUNE" if covers_june else "✗ MISSING JUNE"
        print(f"    Exp {exp}: {min_d} → {max_d}  ({len(seeds)} seeds)  {status}")
    else:
        coverage[exp] = None
        print(f"    Exp {exp}: NO DATA for {test_ticker}")

exps_missing_june = [e for e, c in coverage.items() if c and not c['covers_june']]
exps_with_june = [e for e, c in coverage.items() if c and c['covers_june']]

if exps_missing_june:
    print(f"\n  ⚠ Experiments {exps_missing_june} do NOT cover June.")
    print(f"    You need to re-score these models through June before proceeding.")
    print(f"    Only experiments {exps_with_june} can be used for June analysis.")

if not exps_with_june:
    print(f"\n  ✗ NO experiments cover June. Cannot run rolling optimization backtest.")
    print(f"    Action: Re-score models 152/160/175/185/200 through June 2026.")
    raise SystemExit("No June score data available.")

ACTIVE_EXPERIMENTS = exps_with_june
print(f"\n  Using experiments: {ACTIVE_EXPERIMENTS}")

_release_memory()


# ==============================================================================
# STEP 1: LOAD ALL DATA
# ==============================================================================
print(f"\n  STEP 1: Loading OHLCV + score CSVs...")

all_tickers = sorted(set(DEPLOY_TICKERS))
for exp in ACTIVE_EXPERIMENTS:
    for t in EXP_TEACHERS.get(exp, []):
        if t in DEPLOY_TICKERS and t not in EXCLUDE_TICKERS:
            all_tickers.append(t)
all_tickers = sorted(set(all_tickers))

ohlcv_cache = {}
for ticker in all_tickers:
    df = load_ohlcv(ticker)
    if df is not None:
        ohlcv_cache[ticker] = df
        print(f"    {ticker}: {len(df):>8,} OHLCV bars")
    else:
        print(f"    {ticker}: MISSING OHLCV")

scores_cache = {}
for exp in ACTIVE_EXPERIMENTS:
    for ticker in all_tickers:
        if ticker not in EXP_TEACHERS.get(exp, []):
            continue
        seeds = load_seed_scores(ticker, exp)
        if seeds:
            scores_cache[(ticker, exp)] = seeds
    print(f"    Exp {exp}: loaded {sum(1 for k in scores_cache if k[1]==exp)} tickers")

_release_memory()


# ==============================================================================
# STEP 2: PRE-COMPUTE DAILY SIGNALS AND TRADE OUTCOMES
# ==============================================================================
print(f"\n  STEP 2: Pre-computing daily signals + trade outcomes...")

# ── 2a: Compute avg daily probs per (ticker, exp) ──
daily_probs = {}
all_score_dates_by_ticker = {}

for (ticker, exp), seeds_dict in scores_cache.items():
    avg_probs = None
    n_seeds = 0
    for seed_id, sdf in seeds_dict.items():
        if 'Prob_Spike' not in sdf.columns:
            col = [c for c in sdf.columns if 'prob' in c.lower() or 'spike' in c.lower()]
            if not col:
                continue
            sdf = sdf.rename(columns={col[0]: 'Prob_Spike'})

        daily_prob = sdf.groupby(sdf.index.date)['Prob_Spike'].max()
        if avg_probs is None:
            avg_probs = daily_prob.copy()
        else:
            avg_probs = avg_probs.add(daily_prob, fill_value=0)
        n_seeds += 1

    if n_seeds == 0 or avg_probs is None:
        continue
    avg_probs = avg_probs / n_seeds
    daily_probs[(ticker, exp)] = avg_probs
    if ticker not in all_score_dates_by_ticker:
        all_score_dates_by_ticker[ticker] = set()
    all_score_dates_by_ticker[ticker].update(avg_probs.index)

print(f"    Computed probs for {len(daily_probs)} (ticker, exp) pairs")

# ── 2b: Pre-group OHLCV by date (once per ticker) ──
ohlcv_by_date = {}
for ticker in ohlcv_cache:
    ohlcv = ohlcv_cache[ticker]
    grouped = {d: g for d, g in ohlcv.groupby(ohlcv.index.date)}
    ohlcv_by_date[ticker] = grouped
print(f"    Grouped OHLCV for {len(ohlcv_by_date)} tickers")

# ── 2c: Compute trade outcomes per (ticker, entry, TP, SL) — NOT per experiment ──
# Trade outcome only depends on price path, not on which model signaled
ticker_trades = {}
n_sims = 0

for ticker in ohlcv_by_date:
    if ticker not in all_score_dates_by_ticker:
        continue
    dates = all_score_dates_by_ticker[ticker]
    grouped = ohlcv_by_date[ticker]

    for entry_time_str in ENTRY_TIMES:
        eh, em = map(int, entry_time_str.split(':'))
        entry_min = eh * 60 + em

        for tp in TP_GRID:
            for sl in SL_GRID:
                if sl > tp:
                    continue
                outcomes = {}
                for trade_date in dates:
                    day_bars = grouped.get(trade_date)
                    if day_bars is None or day_bars.empty:
                        continue
                    pnl = simulate_trade(day_bars, entry_min, tp, sl)
                    if pnl is not None:
                        outcomes[trade_date] = pnl - SLIPPAGE_PCT
                        n_sims += 1
                if outcomes:
                    ticker_trades[(ticker, entry_time_str, tp, sl)] = outcomes

    print(f"    {ticker}: {n_sims:,} sims done")

# ── 2d: Expand to daily_trades keyed by (ticker, exp, entry, TP, SL) for Step 3 ──
daily_trades = {}
for (ticker, exp) in daily_probs:
    for entry_time_str in ENTRY_TIMES:
        for tp in TP_GRID:
            for sl in SL_GRID:
                if sl > tp:
                    continue
                key_t = (ticker, entry_time_str, tp, sl)
                if key_t in ticker_trades:
                    daily_trades[(ticker, exp, entry_time_str, tp, sl)] = ticker_trades[key_t]

n_combos = len(daily_trades)
print(f"    Pre-computed {n_combos} (ticker, exp, entry, TP, SL) outcome series")
del ticker_trades, ohlcv_by_date
_release_memory()


# ==============================================================================
# STEP 3: ROLLING OPTIMIZATION SIMULATION
# ==============================================================================
print(f"\n  STEP 3: Running rolling optimization simulation...")
print(f"    Test period: {TEST_START} → {TEST_END}")
print(f"    Lookback options: {LOOKBACK_OPTIONS} trading days")
print(f"    Re-optimization frequency: every {REOPT_FREQ} trading days")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt = pd.Timestamp(TEST_END).date()

all_trading_dates = set()
for outcomes in daily_trades.values():
    all_trading_dates.update(outcomes.keys())
all_trading_dates = sorted([d for d in all_trading_dates if d >= test_start_dt and d <= test_end_dt])

if not all_trading_dates:
    print("  No trading dates found in test range.")
    raise SystemExit()

all_available_dates = sorted(set().union(*(set(o.keys()) for o in daily_trades.values())))

print(f"    {len(all_trading_dates)} trading days in test range")
print(f"    {len(all_available_dates)} total dates with data")


def optimize_for_window(opt_dates, ticker):
    """Find best (entry_time, prob_thresh, tp, sl) for ticker on opt_dates."""
    best_pnl = -999
    best_config = None

    for entry_time_str in ENTRY_TIMES:
        for tp in TP_GRID:
            for sl in SL_GRID:
                if sl > tp:
                    continue
                for prob_thresh in PROB_GRID:
                    total_pnl = 0
                    n_trades = 0

                    for exp in ACTIVE_EXPERIMENTS:
                        key = (ticker, exp, entry_time_str, tp, sl)
                        if key not in daily_trades:
                            continue
                        prob_key = (ticker, exp)
                        if prob_key not in daily_probs:
                            continue

                        outcomes = daily_trades[key]
                        probs = daily_probs[prob_key]

                        for d in opt_dates:
                            if d in probs and d in outcomes:
                                if probs[d] >= prob_thresh:
                                    total_pnl += outcomes[d]
                                    n_trades += 1

                    if n_trades >= MIN_OPT_TRADES:
                        avg = total_pnl / n_trades
                        if avg > best_pnl and avg > MIN_OPT_PNL:
                            best_pnl = avg
                            best_config = {
                                'entry_time': entry_time_str,
                                'prob_thresh': prob_thresh,
                                'tp': tp, 'sl': sl,
                                'avg_pnl': avg, 'n_trades': n_trades,
                            }
    return best_config


def apply_config_for_dates(config, ticker, test_dates):
    """Apply a config for a set of test dates. Returns list of daily PnL."""
    if config is None:
        return []
    trades = []
    entry_time_str = config['entry_time']
    tp = config['tp']
    sl = config['sl']
    prob_thresh = config['prob_thresh']

    for d in test_dates:
        signals = 0
        total_families = 0
        pnl_sum = 0
        pnl_count = 0

        for exp in ACTIVE_EXPERIMENTS:
            prob_key = (ticker, exp)
            trade_key = (ticker, exp, entry_time_str, tp, sl)
            if prob_key not in daily_probs or trade_key not in daily_trades:
                continue
            total_families += 1
            if d in daily_probs[prob_key] and daily_probs[prob_key][d] >= prob_thresh:
                signals += 1
                if d in daily_trades[trade_key]:
                    pnl_sum += daily_trades[trade_key][d]
                    pnl_count += 1

        if total_families >= 2 and signals >= min(MIN_FAMILY_AGREE, (total_families // 2 + 1)):
            if pnl_count > 0:
                trades.append({'date': d, 'ticker': ticker,
                               'pnl': pnl_sum / pnl_count, 'n_exps': signals})
    return trades


# ── Run static baseline (optimize on full pre-test window) ──
print(f"\n  Computing STATIC baseline (full window optimization)...")
static_opt_start = pd.Timestamp(DATA_START).date()
static_opt_end = (pd.Timestamp(TEST_START) - pd.Timedelta(days=1)).date()
static_opt_dates = [d for d in all_available_dates
                    if d >= static_opt_start and d <= static_opt_end]

static_configs = {}
for ticker in all_tickers:
    cfg = optimize_for_window(static_opt_dates, ticker)
    static_configs[ticker] = cfg
    if cfg:
        print(f"    {ticker}: prob>{cfg['prob_thresh']:.2f} TP={cfg['tp']:.0%} SL={cfg['sl']:.0%} "
              f"entry={cfg['entry_time']} avgPnL={cfg['avg_pnl']:.2%} ({cfg['n_trades']} trades)")
    else:
        print(f"    {ticker}: no viable config")

static_trades = []
for ticker in all_tickers:
    static_trades.extend(apply_config_for_dates(static_configs[ticker], ticker, all_trading_dates))

print(f"    Static baseline: {len(static_trades)} trades in test period")


# ── Run rolling optimization for each lookback ──
results_by_lookback = {}

for lookback in LOOKBACK_OPTIONS:
    label = f"{lookback}d"
    print(f"\n  Computing ROLLING {label} lookback...")

    all_rolling_trades = []
    reopt_log = []

    reopt_indices = list(range(0, len(all_trading_dates), REOPT_FREQ))

    for ri in reopt_indices:
        reopt_date = all_trading_dates[ri]
        test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
        test_dates = all_trading_dates[ri:test_end_idx]

        opt_end = reopt_date - timedelta(days=1)
        opt_start_approx = reopt_date - timedelta(days=int(lookback * 1.5))
        opt_dates = [d for d in all_available_dates
                     if d >= opt_start_approx and d <= opt_end]
        opt_dates = opt_dates[-lookback:] if len(opt_dates) > lookback else opt_dates

        if len(opt_dates) < 10:
            continue

        week_configs = {}
        for ticker in all_tickers:
            cfg = optimize_for_window(opt_dates, ticker)
            week_configs[ticker] = cfg

        n_active = sum(1 for c in week_configs.values() if c is not None)
        reopt_log.append({
            'reopt_date': reopt_date,
            'opt_window': f"{opt_dates[0]}→{opt_dates[-1]}",
            'n_opt_days': len(opt_dates),
            'n_active_tickers': n_active,
            'test_dates': f"{test_dates[0]}→{test_dates[-1]}",
        })

        for ticker in all_tickers:
            week_trades = apply_config_for_dates(week_configs[ticker], ticker, test_dates)
            all_rolling_trades.extend(week_trades)

    results_by_lookback[lookback] = {
        'trades': all_rolling_trades,
        'reopt_log': reopt_log,
    }
    print(f"    {label}: {len(all_rolling_trades)} trades, "
          f"{len(reopt_log)} re-optimization cycles")

_release_memory()


# ==============================================================================
# STEP 4: RESULTS COMPARISON
# ==============================================================================
print()
print("=" * W)
print(f"  STEP 4: RESULTS COMPARISON")
print("=" * W)


def summarize_trades(trades_list, label):
    if not trades_list:
        return {'label': label, 'n_trades': 0}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')

    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()
    win_rate = (df['pnl'] > 0).mean()
    n_trades = len(df)

    monthly = df.groupby('month').agg(
        n=('pnl', 'count'),
        total=('pnl', 'sum'),
        avg=('pnl', 'mean'),
        wr=('pnl', lambda x: (x > 0).mean()),
    ).to_dict('index')

    cum_pnl = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum_pnl - cum_pnl.cummax()).min()

    return {
        'label': label, 'n_trades': n_trades, 'total_pnl': total_pnl,
        'avg_pnl': avg_pnl, 'win_rate': win_rate, 'max_dd': max_dd,
        'monthly': monthly,
    }


summaries = [summarize_trades(static_trades, 'STATIC (full window)')]
for lookback in LOOKBACK_OPTIONS:
    trades = results_by_lookback[lookback]['trades']
    summaries.append(summarize_trades(trades, f'ROLLING {lookback}d'))


# ── TABLE A: Overall comparison ──
print(f"\n  TABLE A: OVERALL COMPARISON")
print(f"  {'Strategy':<25} {'Trades':>7} {'TotalPnL':>10} {'AvgPnL':>10} {'WinRate':>8} {'MaxDD':>8}")
print(f"  {'─'*25} {'─'*7} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

for s in summaries:
    if s['n_trades'] == 0:
        print(f"  {s['label']:<25} {'0':>7} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>8}")
        continue
    print(f"  {s['label']:<25} {s['n_trades']:>7} {s['total_pnl']:>+10.2%} "
          f"{s['avg_pnl']:>10.2%} {s['win_rate']:>8.1%} {s['max_dd']:>8.2%}")


# ── TABLE B: Monthly breakdown ──
print(f"\n  TABLE B: MONTHLY BREAKDOWN")
months_in_range = sorted(set(pd.to_datetime(d).strftime('%Y-%m') for d in all_trading_dates))

print(f"  {'Strategy':<25}", end="")
for m in months_in_range:
    print(f" {m:>18}", end="")
print()
print(f"  {'─'*25}", end="")
for _ in months_in_range:
    print(f" {'─'*18}", end="")
print()

for s in summaries:
    print(f"  {s['label']:<25}", end="")
    if s['n_trades'] == 0:
        for _ in months_in_range:
            print(f" {'N/A':>18}", end="")
        print()
        continue
    for m in months_in_range:
        if m in s.get('monthly', {}):
            ms = s['monthly'][m]
            print(f" {ms['avg']:>+6.2%}({ms['n']:>2}t,{ms['wr']:>4.0%})", end="")
        else:
            print(f" {'—':>18}", end="")
    print()


# ── TABLE C: Re-optimization log (rolling) ──
print(f"\n  TABLE C: RE-OPTIMIZATION LOG")

for lookback in LOOKBACK_OPTIONS:
    log = results_by_lookback[lookback]['reopt_log']
    if not log:
        continue
    print(f"\n  ROLLING {lookback}d:")
    print(f"  {'ReoptDate':>12} {'OptWindow':>25} {'OptDays':>8} {'Active':>7} {'TestDates':>25}")
    print(f"  {'─'*12} {'─'*25} {'─'*8} {'─'*7} {'─'*25}")
    for entry in log:
        print(f"  {str(entry['reopt_date']):>12} {entry['opt_window']:>25} "
              f"{entry['n_opt_days']:>8} {entry['n_active_tickers']:>7} {entry['test_dates']:>25}")


# ── TABLE D: Weekly PnL timeline ──
print(f"\n  TABLE D: WEEKLY PnL TIMELINE")
print(f"  {'Week':>12}", end="")
print(f"  {'STATIC':>12}", end="")
for lb in LOOKBACK_OPTIONS:
    print(f"  {f'ROLL_{lb}d':>12}", end="")
print()

static_df = pd.DataFrame(static_trades) if static_trades else pd.DataFrame()
rolling_dfs = {}
for lb in LOOKBACK_OPTIONS:
    t = results_by_lookback[lb]['trades']
    rolling_dfs[lb] = pd.DataFrame(t) if t else pd.DataFrame()

week_starts = all_trading_dates[::REOPT_FREQ]
for ws in week_starts:
    we = min(ws + timedelta(days=6), test_end_dt)
    print(f"  {str(ws):>12}", end="")

    if not static_df.empty:
        wk = static_df[(static_df['date'] >= ws) & (static_df['date'] <= we)]
        if len(wk) > 0:
            print(f"  {wk['pnl'].sum():>+12.2%}", end="")
        else:
            print(f"  {'—':>12}", end="")
    else:
        print(f"  {'—':>12}", end="")

    for lb in LOOKBACK_OPTIONS:
        rdf = rolling_dfs[lb]
        if not rdf.empty:
            wk = rdf[(rdf['date'] >= ws) & (rdf['date'] <= we)]
            if len(wk) > 0:
                print(f"  {wk['pnl'].sum():>+12.2%}", end="")
            else:
                print(f"  {'—':>12}", end="")
        else:
            print(f"  {'—':>12}", end="")
    print()


# ── DIAGNOSIS ──
print()
print("=" * W)
print(f"  DIAGNOSIS")
print("=" * W)

if summaries[0]['n_trades'] > 0:
    static_total = summaries[0]['total_pnl']
    print(f"\n  Static baseline: {static_total:+.2%} total PnL over test period")

    for s in summaries[1:]:
        if s['n_trades'] > 0:
            delta = s['total_pnl'] - static_total
            pct_improve = (delta / abs(static_total)) * 100 if static_total != 0 else 0
            print(f"  {s['label']}: {s['total_pnl']:+.2%} total "
                  f"({delta:+.2%} vs static, {pct_improve:+.0f}% improvement)")

    best = max(summaries, key=lambda s: s.get('total_pnl', -999) if s['n_trades'] > 0 else -999)
    print(f"\n  Best strategy: {best['label']}")

    if best['label'].startswith('ROLLING'):
        june_pnl = best.get('monthly', {}).get('2026-06', {})
        if june_pnl:
            print(f"  June performance: {june_pnl['avg']:+.2%} avg PnL, "
                  f"{june_pnl['n']} trades, {june_pnl['wr']:.0%} win rate")
            if june_pnl['avg'] > 0:
                print(f"  → Rolling optimization SOLVED June losses")
            elif june_pnl['avg'] > -0.01:
                print(f"  → Rolling optimization REDUCED June losses (near breakeven)")
            else:
                print(f"  → Rolling optimization REDUCED but did NOT eliminate June losses")
                print(f"    The adaptation lag means first 1-2 weeks of regime change still hurt")
    else:
        print(f"  → Static optimization was actually best — rolling adds noise without benefit")
        print(f"    This suggests the regime change is too abrupt for backward-looking adaptation")

print()
print("=" * W)
print(f"  Analysis complete.")
print("=" * W)
