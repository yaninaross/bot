"""
backtest_weighted_optimization.py — Weighted Optimization Backtest (3 Approaches)
==================================================================================
Tests three smarter alternatives to pure rolling window optimization:

  1. EXPANDING + EXPONENTIAL DECAY
     Keep all historical data but weight recent trades exponentially higher.
     Half-lives: 15, 30, 60 trading days.

  2. BLENDED RANK SCORING
     Score each config on both the full window and a short recent window,
     then combine ranks. Blend weights: 70/30, 60/40, 50/50.

  3. REGIME-GATED STATIC
     Keep static configs but pause trading a ticker when its trailing
     avg PnL drops below a threshold. Gate lookbacks: 10, 21 days.

Baseline: STATIC (full pre-test window, no adaptation).

Steps 0-2 are identical to backtest_rolling_optimization.py.
Colab-ready (no if __name__). Uses limit_entry mode only (S3 strategy).
"""

import os, io, gc, sys, subprocess, warnings, ctypes, math
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

# ── Test period ──
DATA_START = '2025-12-01'
TEST_START = '2026-04-01'
TEST_END   = '2026-06-29'
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

# ── Approach 1: Expanding + Exponential Decay ──
DECAY_HALF_LIVES = [15, 30, 60]

# ── Approach 2: Blended Rank ──
BLEND_WEIGHTS = [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]
BLEND_RECENT_DAYS = 21

# ── Approach 3: Regime Gate ──
GATE_LOOKBACKS = [10, 21]
GATE_THRESHOLDS = [0.0, -0.01]

gcs_client = gcs_storage.Client()


# ==============================================================================
# HELPERS (same as backtest_rolling_optimization.py)
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


def apply_config_for_dates(config, ticker, test_dates, daily_probs, daily_trades,
                           active_experiments, min_family_agree=MIN_FAMILY_AGREE):
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

        for exp in active_experiments:
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

        if total_families >= 2 and signals >= min(min_family_agree, (total_families // 2 + 1)):
            if pnl_count > 0:
                trades.append({'date': d, 'ticker': ticker,
                               'pnl': pnl_sum / pnl_count, 'n_exps': signals})
    return trades


# ==============================================================================
# STEP 0: CHECK SCORE CSV COVERAGE
# ==============================================================================
W = 120
print("=" * W)
print(f"  WEIGHTED OPTIMIZATION BACKTEST — 3 Approaches")
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

exps_with_june = [e for e, c in coverage.items() if c and c['covers_june']]

if not exps_with_june:
    print(f"\n  ✗ NO experiments cover June. Cannot run backtest.")
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

# ── 2c: Compute trade outcomes per (ticker, entry, TP, SL) ──
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

# ── 2d: Expand to daily_trades keyed by (ticker, exp, entry, TP, SL) ──
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

print(f"    Pre-computed {len(daily_trades)} (ticker, exp, entry, TP, SL) outcome series")
del ticker_trades, ohlcv_by_date
_release_memory()


# ==============================================================================
# STEP 3: OPTIMIZATION STRATEGIES
# ==============================================================================
print(f"\n  STEP 3: Running optimization strategies...")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt = pd.Timestamp(TEST_END).date()

all_trading_dates = set()
for outcomes in daily_trades.values():
    all_trading_dates.update(outcomes.keys())
all_trading_dates = sorted([d for d in all_trading_dates if d >= test_start_dt and d <= test_end_dt])

all_available_dates = sorted(set().union(*(set(o.keys()) for o in daily_trades.values())))

print(f"    Test period: {TEST_START} → {TEST_END}")
print(f"    {len(all_trading_dates)} trading days in test range")
print(f"    {len(all_available_dates)} total dates with data")
print(f"    Re-optimization frequency: every {REOPT_FREQ} trading days")

LN2 = math.log(2)


# ── Build config list (shared across all optimizers) ──
CONFIG_LIST = []
for entry_time_str in ENTRY_TIMES:
    for tp in TP_GRID:
        for sl in SL_GRID:
            if sl > tp:
                continue
            for prob_thresh in PROB_GRID:
                CONFIG_LIST.append((entry_time_str, tp, sl, prob_thresh))


def _eval_config_on_dates(ticker, entry_time_str, tp, sl, prob_thresh, opt_dates, weights=None):
    """Evaluate a single config on given dates. Returns (weighted_pnl, weighted_n)."""
    w_pnl = 0.0
    w_n = 0.0

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
                    w = weights[d] if weights else 1.0
                    w_pnl += w * outcomes[d]
                    w_n += w

    return w_pnl, w_n


def optimize_static(opt_dates, ticker):
    """Original static optimizer — uniform weights on all dates."""
    best_pnl = -999
    best_config = None

    for entry_time_str, tp, sl, prob_thresh in CONFIG_LIST:
        w_pnl, w_n = _eval_config_on_dates(ticker, entry_time_str, tp, sl, prob_thresh, opt_dates)
        if w_n >= MIN_OPT_TRADES:
            avg = w_pnl / w_n
            if avg > best_pnl and avg > MIN_OPT_PNL:
                best_pnl = avg
                best_config = {
                    'entry_time': entry_time_str, 'prob_thresh': prob_thresh,
                    'tp': tp, 'sl': sl, 'avg_pnl': avg, 'n_trades': int(w_n),
                }
    return best_config


# ──────────────────────────────────────────────────────────────────────────────
# STATIC BASELINE
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n  Computing STATIC baseline...")
static_opt_start = pd.Timestamp(DATA_START).date()
static_opt_end = (pd.Timestamp(TEST_START) - pd.Timedelta(days=1)).date()
static_opt_dates = [d for d in all_available_dates
                    if d >= static_opt_start and d <= static_opt_end]

static_configs = {}
for ticker in all_tickers:
    cfg = optimize_static(static_opt_dates, ticker)
    static_configs[ticker] = cfg
    if cfg:
        print(f"    {ticker}: prob>{cfg['prob_thresh']:.2f} TP={cfg['tp']:.0%} SL={cfg['sl']:.0%} "
              f"entry={cfg['entry_time']} avgPnL={cfg['avg_pnl']:.2%} ({cfg['n_trades']} trades)")

static_trades = []
for ticker in all_tickers:
    static_trades.extend(apply_config_for_dates(
        static_configs[ticker], ticker, all_trading_dates,
        daily_probs, daily_trades, ACTIVE_EXPERIMENTS))
print(f"    Static baseline: {len(static_trades)} trades")

reopt_indices = list(range(0, len(all_trading_dates), REOPT_FREQ))


# ──────────────────────────────────────────────────────────────────────────────
# APPROACH 1: EXPANDING WINDOW + EXPONENTIAL DECAY
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n  APPROACH 1: Expanding window + exponential decay...")
results_decay = {}

for half_life in DECAY_HALF_LIVES:
    label = f"DECAY hl={half_life}d"
    print(f"    {label}...")
    decay_rate = LN2 / half_life

    all_decay_trades = []
    reopt_log = []

    for ri in reopt_indices:
        reopt_date = all_trading_dates[ri]
        test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
        test_dates = all_trading_dates[ri:test_end_idx]

        opt_end = reopt_date - timedelta(days=1)
        opt_dates = [d for d in all_available_dates
                     if d >= static_opt_start and d <= opt_end]

        if len(opt_dates) < 10:
            continue

        weights = {}
        for d in opt_dates:
            days_ago = (reopt_date - d).days
            weights[d] = math.exp(-decay_rate * days_ago)

        week_configs = {}
        for ticker in all_tickers:
            best_pnl = -999
            best_config = None

            for entry_time_str, tp, sl, prob_thresh in CONFIG_LIST:
                w_pnl, w_n = _eval_config_on_dates(
                    ticker, entry_time_str, tp, sl, prob_thresh, opt_dates, weights)
                if w_n >= MIN_OPT_TRADES:
                    avg = w_pnl / w_n
                    if avg > best_pnl and avg > MIN_OPT_PNL:
                        best_pnl = avg
                        best_config = {
                            'entry_time': entry_time_str, 'prob_thresh': prob_thresh,
                            'tp': tp, 'sl': sl, 'avg_pnl': avg, 'n_trades': w_n,
                        }
            week_configs[ticker] = best_config

        n_active = sum(1 for c in week_configs.values() if c is not None)
        reopt_log.append({
            'reopt_date': reopt_date,
            'n_opt_days': len(opt_dates),
            'n_active_tickers': n_active,
        })

        for ticker in all_tickers:
            week_trades = apply_config_for_dates(
                week_configs[ticker], ticker, test_dates,
                daily_probs, daily_trades, ACTIVE_EXPERIMENTS)
            all_decay_trades.extend(week_trades)

    results_decay[half_life] = {
        'trades': all_decay_trades,
        'reopt_log': reopt_log,
    }
    print(f"      {len(all_decay_trades)} trades, {len(reopt_log)} reopt cycles")

_release_memory()


# ──────────────────────────────────────────────────────────────────────────────
# APPROACH 2: BLENDED RANK SCORING
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n  APPROACH 2: Blended rank scoring...")
results_blend = {}

for w_full, w_recent in BLEND_WEIGHTS:
    label = f"BLEND {int(w_full*100)}/{int(w_recent*100)}"
    print(f"    {label}...")

    all_blend_trades = []
    reopt_log = []

    for ri in reopt_indices:
        reopt_date = all_trading_dates[ri]
        test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
        test_dates = all_trading_dates[ri:test_end_idx]

        opt_end = reopt_date - timedelta(days=1)

        full_dates = [d for d in all_available_dates
                      if d >= static_opt_start and d <= opt_end]
        recent_approx = reopt_date - timedelta(days=int(BLEND_RECENT_DAYS * 1.5))
        recent_dates = [d for d in all_available_dates
                        if d >= recent_approx and d <= opt_end]
        recent_dates = recent_dates[-BLEND_RECENT_DAYS:] if len(recent_dates) > BLEND_RECENT_DAYS else recent_dates

        if len(full_dates) < 10 or len(recent_dates) < 5:
            continue

        week_configs = {}
        for ticker in all_tickers:
            scored_configs = []

            for entry_time_str, tp, sl, prob_thresh in CONFIG_LIST:
                pnl_full, n_full = _eval_config_on_dates(
                    ticker, entry_time_str, tp, sl, prob_thresh, full_dates)
                pnl_recent, n_recent = _eval_config_on_dates(
                    ticker, entry_time_str, tp, sl, prob_thresh, recent_dates)

                if n_full >= MIN_OPT_TRADES and n_recent >= 2:
                    avg_full = pnl_full / n_full
                    avg_recent = pnl_recent / n_recent
                    if avg_full > MIN_OPT_PNL:
                        scored_configs.append({
                            'entry_time': entry_time_str, 'prob_thresh': prob_thresh,
                            'tp': tp, 'sl': sl,
                            'avg_full': avg_full, 'avg_recent': avg_recent,
                            'n_full': int(n_full), 'n_recent': int(n_recent),
                        })

            if not scored_configs:
                week_configs[ticker] = None
                continue

            scored_configs.sort(key=lambda x: -x['avg_full'])
            for rank, c in enumerate(scored_configs):
                c['rank_full'] = rank

            scored_configs.sort(key=lambda x: -x['avg_recent'])
            for rank, c in enumerate(scored_configs):
                c['rank_recent'] = rank

            for c in scored_configs:
                c['blended_rank'] = w_full * c['rank_full'] + w_recent * c['rank_recent']

            best = min(scored_configs, key=lambda x: x['blended_rank'])
            week_configs[ticker] = {
                'entry_time': best['entry_time'], 'prob_thresh': best['prob_thresh'],
                'tp': best['tp'], 'sl': best['sl'],
                'avg_pnl': best['avg_full'], 'n_trades': best['n_full'],
            }

        n_active = sum(1 for c in week_configs.values() if c is not None)
        reopt_log.append({
            'reopt_date': reopt_date,
            'n_opt_days_full': len(full_dates),
            'n_opt_days_recent': len(recent_dates),
            'n_active_tickers': n_active,
        })

        for ticker in all_tickers:
            week_trades = apply_config_for_dates(
                week_configs[ticker], ticker, test_dates,
                daily_probs, daily_trades, ACTIVE_EXPERIMENTS)
            all_blend_trades.extend(week_trades)

    results_blend[(w_full, w_recent)] = {
        'trades': all_blend_trades,
        'reopt_log': reopt_log,
    }
    print(f"      {len(all_blend_trades)} trades, {len(reopt_log)} reopt cycles")

_release_memory()


# ──────────────────────────────────────────────────────────────────────────────
# APPROACH 3: REGIME-GATED STATIC
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n  APPROACH 3: Regime-gated static...")
results_gate = {}

# Pre-compute all static trades per ticker with dates for gate evaluation
static_trades_by_ticker = defaultdict(list)
for t in static_trades:
    static_trades_by_ticker[t['ticker']].append(t)

for gate_lookback in GATE_LOOKBACKS:
    for gate_thresh in GATE_THRESHOLDS:
        label = f"GATE lb={gate_lookback}d thr={gate_thresh:.0%}"
        print(f"    {label}...")

        gated_trades = []
        gate_log = defaultdict(list)

        for ri in reopt_indices:
            reopt_date = all_trading_dates[ri]
            test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
            test_dates = all_trading_dates[ri:test_end_idx]

            for ticker in all_tickers:
                if static_configs[ticker] is None:
                    continue

                prior_trades = [t for t in static_trades_by_ticker[ticker]
                                if t['date'] < reopt_date]
                recent_trades = prior_trades[-gate_lookback:]

                if len(recent_trades) >= 2:
                    recent_avg = np.mean([t['pnl'] for t in recent_trades])
                    is_active = recent_avg >= gate_thresh
                else:
                    is_active = True

                gate_log[ticker].append({
                    'date': reopt_date,
                    'active': is_active,
                    'recent_avg': np.mean([t['pnl'] for t in recent_trades]) if recent_trades else None,
                    'n_recent': len(recent_trades),
                })

                if is_active:
                    week_trades = apply_config_for_dates(
                        static_configs[ticker], ticker, test_dates,
                        daily_probs, daily_trades, ACTIVE_EXPERIMENTS)
                    gated_trades.extend(week_trades)

        results_gate[(gate_lookback, gate_thresh)] = {
            'trades': gated_trades,
            'gate_log': dict(gate_log),
        }
        print(f"      {len(gated_trades)} trades")

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

    monthly = df.groupby('month').agg(
        n=('pnl', 'count'),
        total=('pnl', 'sum'),
        avg=('pnl', 'mean'),
        wr=('pnl', lambda x: (x > 0).mean()),
    ).to_dict('index')

    cum_pnl = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum_pnl - cum_pnl.cummax()).min()

    return {
        'label': label, 'n_trades': len(df), 'total_pnl': total_pnl,
        'avg_pnl': avg_pnl, 'win_rate': win_rate, 'max_dd': max_dd,
        'monthly': monthly, 'trades_df': df,
    }


# Build all summaries
all_summaries = []
all_summaries.append(summarize_trades(static_trades, 'STATIC'))

for hl in DECAY_HALF_LIVES:
    all_summaries.append(summarize_trades(results_decay[hl]['trades'], f'DECAY hl={hl}d'))

for w_f, w_r in BLEND_WEIGHTS:
    all_summaries.append(summarize_trades(
        results_blend[(w_f, w_r)]['trades'], f'BLEND {int(w_f*100)}/{int(w_r*100)}'))

for gl in GATE_LOOKBACKS:
    for gt in GATE_THRESHOLDS:
        all_summaries.append(summarize_trades(
            results_gate[(gl, gt)]['trades'], f'GATE lb={gl} thr={gt:.0%}'))


# ── TABLE A: Overall comparison ──
print(f"\n  TABLE A: OVERALL COMPARISON")
print(f"  {'Strategy':<25} {'Trades':>7} {'TotalPnL':>10} {'AvgPnL':>10} {'WinRate':>8} {'MaxDD':>8}")
print(f"  {'─'*25} {'─'*7} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

for s in all_summaries:
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

for s in all_summaries:
    print(f"  {s['label']:<25}", end="")
    if s['n_trades'] == 0:
        for _ in months_in_range:
            print(f" {'—':>18}", end="")
        print()
        continue
    for m in months_in_range:
        if m in s.get('monthly', {}):
            ms = s['monthly'][m]
            print(f" {ms['avg']:>+6.2%}({ms['n']:>2}t,{ms['wr']:>4.0%})", end="")
        else:
            print(f" {'—':>18}", end="")
    print()


# ── TABLE C: Per-ticker comparison ──
print(f"\n  TABLE C: PER-TICKER COMPARISON (top strategies vs static)")

# Pick top 3 strategies by total PnL (excluding static)
ranked = sorted([s for s in all_summaries if s['n_trades'] > 0],
                key=lambda x: x['total_pnl'], reverse=True)
top_labels = [s['label'] for s in ranked[:4]]

print(f"  {'Ticker':<7}", end="")
for lbl in top_labels:
    print(f"  {lbl:<22}", end="")
print()
print(f"  {'─'*7}", end="")
for _ in top_labels:
    print(f"  {'─'*22}", end="")
print()

for ticker in all_tickers:
    print(f"  {ticker:<7}", end="")
    for lbl in top_labels:
        s = next((x for x in all_summaries if x['label'] == lbl), None)
        if s and s['n_trades'] > 0 and 'trades_df' in s:
            tdf = s['trades_df']
            ttrades = tdf[tdf['ticker'] == ticker]
            if len(ttrades) > 0:
                tavg = ttrades['pnl'].mean()
                twr = (ttrades['pnl'] > 0).mean()
                print(f"  {tavg:>+6.2%} {len(ttrades):>2}t {twr:>4.0%}wr     ", end="")
            else:
                print(f"  {'no trades':<22}", end="")
        else:
            print(f"  {'—':<22}", end="")
    print()


# ── TABLE D: Weekly PnL timeline (top strategies) ──
print(f"\n  TABLE D: WEEKLY PnL TIMELINE (top strategies)")
display_labels = top_labels[:5]

print(f"  {'Week':>12}", end="")
for lbl in display_labels:
    print(f"  {lbl:>14}", end="")
print()

week_starts = all_trading_dates[::REOPT_FREQ]
for ws in week_starts:
    we = min(ws + timedelta(days=6), test_end_dt)
    print(f"  {str(ws):>12}", end="")

    for lbl in display_labels:
        s = next((x for x in all_summaries if x['label'] == lbl), None)
        if s and s['n_trades'] > 0 and 'trades_df' in s:
            wk = s['trades_df'][(s['trades_df']['date'] >= ws) & (s['trades_df']['date'] <= we)]
            if len(wk) > 0:
                print(f"  {wk['pnl'].sum():>+14.2%}", end="")
            else:
                print(f"  {'—':>14}", end="")
        else:
            print(f"  {'—':>14}", end="")
    print()


# ── TABLE E: Gate log (Approach 3) ──
print(f"\n  TABLE E: REGIME GATE LOG")

for gl in GATE_LOOKBACKS:
    for gt in GATE_THRESHOLDS:
        key = (gl, gt)
        if key not in results_gate:
            continue
        gate_log = results_gate[key].get('gate_log', {})
        if not gate_log:
            continue
        label = f"GATE lb={gl} thr={gt:.0%}"
        print(f"\n  {label}:")
        print(f"  {'Ticker':<7} {'Paused':>8}  Pause periods")
        print(f"  {'─'*7} {'─'*8}  {'─'*60}")

        for ticker in all_tickers:
            if ticker not in gate_log:
                continue
            entries = gate_log[ticker]
            paused_dates = [e for e in entries if not e['active']]
            n_paused = len(paused_dates)
            pause_periods = []
            in_pause = False
            pause_start = None
            for e in entries:
                if not e['active'] and not in_pause:
                    in_pause = True
                    pause_start = e['date']
                elif e['active'] and in_pause:
                    in_pause = False
                    pause_periods.append(f"{pause_start}→{e['date']}")
            if in_pause:
                pause_periods.append(f"{pause_start}→END")
            periods_str = ", ".join(pause_periods) if pause_periods else "never paused"
            print(f"  {ticker:<7} {n_paused:>6}/{len(entries)}  {periods_str}")


# ── DIAGNOSIS ──
print()
print("=" * W)
print(f"  DIAGNOSIS")
print("=" * W)

if all_summaries[0]['n_trades'] > 0:
    static_total = all_summaries[0]['total_pnl']
    static_avg = all_summaries[0]['avg_pnl']
    print(f"\n  Static baseline: {static_total:+.2%} total PnL, {static_avg:+.2%} avg")

    print(f"\n  vs Static delta:")
    for s in all_summaries[1:]:
        if s['n_trades'] == 0:
            continue
        delta = s['total_pnl'] - static_total
        print(f"    {s['label']:<25} {s['total_pnl']:>+8.2%} total  ({delta:>+7.2%} vs static)")

    best = max(all_summaries, key=lambda x: x.get('total_pnl', -999) if x['n_trades'] > 0 else -999)
    print(f"\n  Best overall: {best['label']} ({best['total_pnl']:+.2%})")

    # June-specific analysis
    print(f"\n  June-only comparison:")
    for s in all_summaries:
        if s['n_trades'] == 0:
            continue
        june = s.get('monthly', {}).get('2026-06')
        if june:
            print(f"    {s['label']:<25} {june['avg']:>+7.2%} avg  {june['n']:>2} trades  {june['wr']:>5.0%} WR")

    # Find best June performer
    june_best = max(
        [s for s in all_summaries if s['n_trades'] > 0 and '2026-06' in s.get('monthly', {})],
        key=lambda x: x['monthly']['2026-06']['avg'],
        default=None
    )
    if june_best:
        june_static = all_summaries[0].get('monthly', {}).get('2026-06', {})
        june_best_m = june_best['monthly']['2026-06']
        print(f"\n  Best June: {june_best['label']}")
        if june_static:
            delta = june_best_m['avg'] - june_static.get('avg', 0)
            print(f"    June improvement vs static: {delta:>+.2%} avg PnL")

    # Approach-level verdict
    decay_best = max(
        [s for s in all_summaries if s['label'].startswith('DECAY') and s['n_trades'] > 0],
        key=lambda x: x['total_pnl'], default=None)
    blend_best = max(
        [s for s in all_summaries if s['label'].startswith('BLEND') and s['n_trades'] > 0],
        key=lambda x: x['total_pnl'], default=None)
    gate_best = max(
        [s for s in all_summaries if s['label'].startswith('GATE') and s['n_trades'] > 0],
        key=lambda x: x['total_pnl'], default=None)

    print(f"\n  Best per approach:")
    if decay_best:
        print(f"    Exp. Decay:   {decay_best['label']:<25} {decay_best['total_pnl']:>+8.2%}")
    if blend_best:
        print(f"    Blended Rank: {blend_best['label']:<25} {blend_best['total_pnl']:>+8.2%}")
    if gate_best:
        print(f"    Regime Gate:  {gate_best['label']:<25} {gate_best['total_pnl']:>+8.2%}")

print()
print("=" * W)
print(f"  Analysis complete.")
print("=" * W)
