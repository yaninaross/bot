"""
compare_230_vs_180_200.py — Exp 240 vs 230 vs 180 vs 200 Head-to-Head Comparison
==================================================================================
Compares signal quality (AUC, precision) and backtest PnL for:
  180 = 31-teacher production model (S3 STATIC baseline)
  200 = CNN-Transformer hybrid, 31 teachers
  230 = Curated 20-teacher, equalized weighting, no augmentation
  240 = Regime-awareness features (28 new), same 31 teachers as 180

All trade simulations use 180's optimized limit_entry configs.
Same TP/SL/dip — only the probability signal differs.

Sections:
  A. Signal Quality — per-ticker AUC + precision at 0.50/0.60/0.70
  B. Backtest PnL — limit_entry trades using 180's BQ configs
  C. Monthly Breakdown — Apr/May/Jun/Jul 2026
  D. Per-Ticker Winner Analysis
  E. Charts — cumulative PnL + AUC scatter

Colab-ready (no if __name__).
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

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("  WARNING: sklearn not available — AUC will be skipped")


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"
RESULTS_DIR = "Algos/Results_scored_20260703"
RESULTS_FALLBACK = "Algos/Results_all_v2"
INFERENCE_DIR = "Data/Inference_v2"

TEST_START = '2026-04-01'
TEST_END   = '2026-07-21'
ENTRY_MIN  = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT = 0.0015
DEPLOY_THRESHOLD = 0.018

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

EXPERIMENTS = {
    180: 'Exp 180 (31T)',
    200: 'Exp 200 (CNN-TX)',
    230: 'Exp 230 (20T curated)',
    240: 'Exp 240 (regime)',
}

TEACHERS_230 = [
    'AEHR', 'CRDO', 'MXL', 'WOLF', 'KEEL', 'LUNR', 'MNTS', 'SPIR',
    'FCEL', 'NEOV', 'NNE', 'SMR', 'NBIS', 'ONDS', 'SLS',
    'BGDE', 'PLUG', 'RDW', 'UCTT', 'VOYG',
]

TEACHERS_240 = [
    'CIFR', 'IREN', 'KEEL', 'HUT', 'APLD', 'WULF',
    'RIOT', 'CLSK', 'BTBT', 'ALAB', 'MARA',
    'RCAT', 'ASTS', 'LUNR', 'BBAI', 'RKLB', 'SPIR', 'BKSY', 'MNTS',
    'PLUG', 'BE', 'FCEL', 'FLNC',
    'OKLO', 'SMR', 'MVIS',
    'BGDE', 'RDW', 'PL', 'VOYG',
]

EXP_TEACHERS = {230: TEACHERS_230, 240: TEACHERS_240}

MIN_SEEDS = 3
MAX_VA_OPT_RATIO = 3.0

GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']

gcs_client = gcs_storage.Client()
bq_client = bigquery.Client()
W = 120


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
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                      for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        df = df[df.index.notna()].sort_index()
        cutoff = pd.Timestamp(TEST_START, tz='UTC') - pd.Timedelta(days=5)
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
    for results_dir in [RESULTS_FALLBACK, RESULTS_DIR]:
        prefix = f"{results_dir}/Universal_Exp{experiment_base}/{clean}/Seed_"
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            continue
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
            if seed_id in all_results:
                old = all_results[seed_id]
                combined = pd.concat([old, d])
                combined = combined[~combined.index.duplicated(keep='last')]
                all_results[seed_id] = combined.sort_index()
            else:
                all_results[seed_id] = d
    return all_results


# ==============================================================================
# TRADE SIMULATION — limit_entry mode
# ==============================================================================
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


def run_single_trade_limit_entry(prices, highs, lows, date_ints, time_mins,
                                  entry_idx, tp, sl, exit_time, dip_pct, max_wait):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    sp = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    et = time_mins[entry_idx]
    lp = sp * (1.0 - dip_pct)
    fill_idx = -1
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            break
        if time_mins[k] - et > max_wait:
            break
        if lows[k] <= lp:
            fill_idx = k
            break
    if fill_idx < 0:
        return 0.0, ed, False
    ep = lp
    tgt = ep * (1.0 + tp)
    stp = ep * (1.0 - sl)
    for k in range(fill_idx + 1, n):
        if date_ints[k] != ed:
            return (prices[k - 1] - ep) / ep, ed, True
        sl_h = lows[k] <= stp
        tp_h = highs[k] >= tgt
        if sl_h and tp_h:
            return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_h:
            return -sl, ed, True
        if tp_h:
            return tp, ed, True
        if time_mins[k] >= exit_time:
            return (prices[k] - ep) / ep, ed, True
    return (prices[n - 1] - ep) / ep, ed, True


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
print("=" * W)
print(f"  EXP 240 vs 230 vs 180 vs 200 — HEAD-TO-HEAD COMPARISON")
print(f"  Test period: {TEST_START} → {TEST_END}")
print(f"  Signal source: GCS seed CSVs | Trade config: 180 BQ limit_entry")
print("=" * W)


# ── Step 1: Load BQ configs from 180 family ──
print(f"\n  STEP 1: Loading 180-family BQ configs (limit_entry, robust_wf_v12)...")
q = """
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult, grace_minutes,
       dip_pct, max_wait_minutes, vol_filter_pct,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, opt_avg_pnl, is_ensemble
FROM `ys-time-series.scoring_logs.wf_optimization_results`
WHERE model_version IN ('robust_wf_v12', 'robust_wf_v12_stitched')
  AND survived_a = TRUE
  AND val_a_count >= 3
  AND val_a_avg_pnl > 0
  AND experiment LIKE '18%_XSECTOR_LIMIT'
"""
raw_df = bq_client.query(q).to_dataframe()
print(f"    {len(raw_df)} raw BQ rows loaded (180 family limit_entry)")

raw_df = raw_df[raw_df['opt_avg_pnl'] > 0].copy()
raw_df = raw_df[raw_df['val_a_avg_pnl'] / raw_df['opt_avg_pnl'] <= MAX_VA_OPT_RATIO]
print(f"    {len(raw_df)} after overfit filter")

indiv = raw_df[raw_df['is_ensemble'] == False].copy()
for k in GRP_KEYS:
    if k not in indiv.columns:
        indiv[k] = 0
    else:
        indiv[k] = indiv[k].fillna(0)

sa = indiv.groupby(GRP_KEYS).agg(
    n_seeds=('seed', 'nunique'),
    n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= DEPLOY_THRESHOLD).sum()),
    mean_va_pnl=('val_a_avg_pnl', 'mean'),
    mean_va_wr=('val_a_win_rate', 'mean'),
    mean_va_count=('val_a_count', 'mean'),
    mean_va_freq=('val_a_freq', 'mean'),
    vb_pnl=('val_b_avg_pnl', 'mean'),
).reset_index()
sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)
sa = sa[sa['n_seeds'] >= MIN_SEEDS].copy()
sa['score'] = (sa['mean_va_pnl'] ** 2) * sa['agree_18']

bq_configs = {}
for ticker in sa['ticker'].unique():
    tk_rows = sa[sa['ticker'] == ticker].copy()
    if tk_rows.empty:
        continue
    best = tk_rows.loc[tk_rows['score'].idxmax()]
    if best['mean_va_pnl'] < DEPLOY_THRESHOLD:
        continue
    bq_configs[ticker] = {
        'prob_thresh': float(best['prob_thresh']),
        'tp': float(best['tp_pct']),
        'sl': float(best['sl_pct']),
        'avg_pnl': float(best['mean_va_pnl']),
        'experiment': best['experiment'],
        'dip_pct': float(best['dip_pct']),
        'max_wait_minutes': float(best['max_wait_minutes']),
        'entry_minute': int(best['entry_minute']) if pd.notna(best['entry_minute']) else ENTRY_MIN,
    }

print(f"    {len(bq_configs)} tickers with qualifying limit_entry configs")

eval_tickers = sorted(bq_configs.keys())
print(f"    Eval tickers: {len(eval_tickers)}")

del raw_df, indiv, sa
_release_memory()


# ── Step 2: Load OHLCV ──
print(f"\n  STEP 2: Loading OHLCV data...")

ohlcv_arrays = {}
for ticker in eval_tickers:
    df = load_master(ticker)
    if df is None:
        continue
    df = heal_data(df.sort_index())
    ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
    if ohlcv is not None:
        ohlcv_arrays[ticker] = ohlcv
    del df; gc.collect()

print(f"    {len(ohlcv_arrays)} tickers with OHLCV data")
_release_memory()


# ── Step 3: Load seed CSVs for all experiments ──
print(f"\n  STEP 3: Loading seed CSVs from GCS ({len(EXPERIMENTS)} experiments × {len(eval_tickers)} tickers)...")

daily_probs = {}      # (ticker, exp_base) → {date: avg_prob}
daily_targets = {}    # (ticker, exp_base) → {date: target}
scores_loaded = defaultdict(int)

for exp_base in EXPERIMENTS:
    n_loaded = 0
    for ticker in eval_tickers:
        seeds_dict = load_seed_scores(ticker, f"{exp_base}_XSECTOR")
        if not seeds_dict:
            continue

        cfg = bq_configs.get(ticker)
        entry_min = cfg['entry_minute'] if cfg else ENTRY_MIN

        avg_probs = None
        avg_targets = None
        n_seeds = 0
        for seed_id, sdf in seeds_dict.items():
            if 'Prob_Spike' not in sdf.columns:
                col = [c for c in sdf.columns if 'prob' in c.lower() or 'spike' in c.lower()]
                if not col:
                    continue
                sdf = sdf.rename(columns={col[0]: 'Prob_Spike'})

            bar_mins = sdf.index.hour * 60 + sdf.index.minute
            entry_bars = sdf[bar_mins == entry_min]
            if entry_bars.empty:
                continue
            daily_prob = entry_bars.groupby(entry_bars.index.date)['Prob_Spike'].first()

            if 'Target' in sdf.columns:
                daily_tgt = entry_bars.groupby(entry_bars.index.date)['Target'].first()
                if avg_targets is None:
                    avg_targets = daily_tgt.copy()
                else:
                    avg_targets = avg_targets.combine_first(daily_tgt)

            if avg_probs is None:
                avg_probs = daily_prob.copy()
            else:
                avg_probs = avg_probs.add(daily_prob, fill_value=0)
            n_seeds += 1

        if n_seeds > 0 and avg_probs is not None:
            daily_probs[(ticker, exp_base)] = (avg_probs / n_seeds).to_dict()
            if avg_targets is not None:
                daily_targets[(ticker, exp_base)] = avg_targets.to_dict()
            n_loaded += 1
            scores_loaded[exp_base] += 1

    print(f"    Exp {exp_base}: {n_loaded}/{len(eval_tickers)} tickers loaded")

_release_memory()


# ── Step 4: Pre-simulate trades (limit_entry) ──
print(f"\n  STEP 4: Pre-simulating trades (limit_entry mode)...")

daily_trades = {}
test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt = pd.Timestamp(TEST_END).date()
all_trading_dates = set()

for ticker in eval_tickers:
    cfg = bq_configs.get(ticker)
    if cfg is None:
        continue
    ohlcv = ohlcv_arrays.get(ticker)
    if ohlcv is None:
        continue
    tp, sl = cfg['tp'], cfg['sl']
    dip_pct = cfg['dip_pct']
    max_wait = cfg['max_wait_minutes']
    entry_min = cfg.get('entry_minute', ENTRY_MIN)
    outcomes = {}
    unique_dates = np.unique(ohlcv['dates'])
    for d_int in unique_dates:
        d_str = str(int(d_int))
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < test_start_dt or d_date > test_end_dt:
            continue
        mask = (ohlcv['dates'] == d_int) & (ohlcv['times'] >= entry_min)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        entry_idx = idxs[0]
        if ohlcv['prices'][entry_idx] <= 0:
            continue
        pnl, _, valid = run_single_trade_limit_entry(
            ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
            ohlcv['dates'], ohlcv['times'],
            entry_idx, tp, sl, EXIT_TIME_MIN, dip_pct, max_wait)
        if valid:
            outcomes[d_date] = pnl - SLIPPAGE_PCT
    if outcomes:
        daily_trades[ticker] = outcomes
        all_trading_dates.update(outcomes.keys())

all_trading_dates = sorted(all_trading_dates)
print(f"    {len(daily_trades)} tickers with fills, {len(all_trading_dates)} test days")

del ohlcv_arrays
_release_memory()


# ==============================================================================
# SECTION A: SIGNAL QUALITY — AUC + PRECISION
# ==============================================================================
print()
print("=" * W)
print(f"  A. SIGNAL QUALITY — AUC + PRECISION AT ENTRY BAR")
print("=" * W)

auc_results = {}   # (ticker, exp) → auc
prec_results = {}  # (ticker, exp, thresh) → precision

for ticker in eval_tickers:
    for exp_base in EXPERIMENTS:
        pk = (ticker, exp_base)
        if pk not in daily_probs or pk not in daily_targets:
            continue

        probs = daily_probs[pk]
        targets = daily_targets[pk]
        common_dates = sorted(set(probs.keys()) & set(targets.keys()))
        if len(common_dates) < 10:
            continue

        y_true = np.array([int(targets[d]) for d in common_dates])
        y_prob = np.array([probs[d] for d in common_dates])

        if HAS_SKLEARN and len(np.unique(y_true)) == 2:
            try:
                auc_results[(ticker, exp_base)] = roc_auc_score(y_true, y_prob)
            except Exception:
                pass

        for thresh in [0.50, 0.60, 0.70]:
            pred = (y_prob >= thresh).astype(int)
            n_pred = pred.sum()
            if n_pred > 0:
                prec = (pred * y_true).sum() / n_pred
                prec_results[(ticker, exp_base, thresh)] = prec

EXP_LIST = [180, 200, 230, 240]

print(f"\n  {'Ticker':<7} │ {'AUC-180':>7} {'AUC-200':>7} {'AUC-230':>7} {'AUC-240':>7} │ "
      f"{'P60-180':>7} {'P60-240':>7} │ {'Δ(240-180)':>10} │ {'Best':>4}")
print(f"  {'─'*7} │ {'─'*7} {'─'*7} {'─'*7} {'─'*7} │ {'─'*7} {'─'*7} │ {'─'*10} │ {'─'*4}")

ticker_auc_deltas = {}  # exp → [(ticker, delta_vs_180)]
for exp in [230, 240]:
    ticker_auc_deltas[exp] = []

for ticker in eval_tickers:
    aucs = {}
    precs = {}
    for exp in EXP_LIST:
        aucs[exp] = auc_results.get((ticker, exp))
        precs[exp] = prec_results.get((ticker, exp, 0.60))

    if aucs[180] is None and aucs[240] is None:
        continue

    def _fmt_auc(v):
        return f"{v:.3f}" if v is not None else "  —  "
    def _fmt_prec(v):
        return f"{v:.1%}" if v is not None else "  —  "

    delta = ""
    best = "—"
    if aucs[240] is not None and aucs[180] is not None:
        d = aucs[240] - aucs[180]
        delta = f"{d:+.3f}"
        ticker_auc_deltas[240].append((ticker, d))
    if aucs[230] is not None and aucs[180] is not None:
        ticker_auc_deltas[230].append((ticker, aucs[230] - aucs[180]))

    valid = {e: a for e, a in aucs.items() if a is not None}
    if valid:
        best = str(max(valid, key=valid.get))

    print(f"  {ticker:<7} │ {_fmt_auc(aucs[180]):>7} {_fmt_auc(aucs[200]):>7} "
          f"{_fmt_auc(aucs[230]):>7} {_fmt_auc(aucs[240]):>7} │ {_fmt_prec(precs[180]):>7} "
          f"{_fmt_prec(precs[240]):>7} │ {delta:>10} │ {best:>4}")

# Summary
print(f"\n  {'SUMMARY':<7} │", end="")
for exp in EXP_LIST:
    exp_aucs = [v for (t, e), v in auc_results.items() if e == exp]
    if exp_aucs:
        print(f" {exp}:med={np.median(exp_aucs):.3f}", end="")
    else:
        print(f" {exp}:—", end="")
print()

for cmp_exp in [230, 240]:
    deltas = ticker_auc_deltas[cmp_exp]
    if deltas:
        n_better = sum(1 for _, d in deltas if d > 0)
        n_worse = sum(1 for _, d in deltas if d < 0)
        avg_delta = np.mean([d for _, d in deltas])
        print(f"  AUC {cmp_exp} vs 180: {n_better} better, {n_worse} worse | "
              f"avg Δ = {avg_delta:+.4f}")


# ==============================================================================
# SECTION B: BACKTEST PNL COMPARISON
# ==============================================================================
print()
print("=" * W)
print(f"  B. BACKTEST PNL — LIMIT_ENTRY (180 configs, {len(EXPERIMENTS)} experiments' signals)")
print("=" * W)

exp_trades = {exp: [] for exp in EXPERIMENTS}

for ticker in eval_tickers:
    cfg = bq_configs.get(ticker)
    if cfg is None or ticker not in daily_trades:
        continue
    for exp_base in EXPERIMENTS:
        pk = (ticker, exp_base)
        if pk not in daily_probs:
            continue
        probs = daily_probs[pk]
        prob_thresh = cfg['prob_thresh']
        for d in all_trading_dates:
            if d in probs and probs[d] >= prob_thresh:
                if d in daily_trades[ticker]:
                    exp_trades[exp_base].append({
                        'date': d, 'ticker': ticker,
                        'pnl': daily_trades[ticker][d],
                        'month': pd.Timestamp(d).strftime('%Y-%m'),
                    })

# Per-experiment portfolio stats
print(f"\n  {'Experiment':<22} │ {'Tickers':>7} {'Trades':>6} {'AvgPnL':>8} "
      f"{'TotalPnL':>9} {'WR':>5} │ {'MaxDD':>7}")
print(f"  {'─'*22} │ {'─'*7} {'─'*6} {'─'*8} {'─'*9} {'─'*5} │ {'─'*7}")

exp_stats = {}
for exp_base, label in EXPERIMENTS.items():
    trades = exp_trades[exp_base]
    if not trades:
        print(f"  {label:<22} │ {'—':>7} {'—':>6} {'—':>8} {'—':>9} {'—':>5} │ {'—':>7}")
        continue
    df = pd.DataFrame(trades)
    filt = df.groupby('ticker').filter(lambda x: x['pnl'].mean() >= DEPLOY_THRESHOLD)
    if filt.empty:
        filt = df
    n_tickers = filt['ticker'].nunique()
    cum = filt.sort_values('date')['pnl'].cumsum()
    dd = (cum - cum.cummax()).min()
    exp_stats[exp_base] = {
        'n_tickers': n_tickers, 'n_trades': len(filt),
        'avg': filt['pnl'].mean(), 'total': filt['pnl'].sum(),
        'wr': (filt['pnl'] > 0).mean(), 'dd': dd,
    }
    print(f"  {label:<22} │ {n_tickers:>7} {len(filt):>6} {filt['pnl'].mean():>+7.2%} "
          f"{filt['pnl'].sum():>+8.1%} {(filt['pnl'] > 0).mean():>4.0%} │ {dd:>+6.1%}")

# Per-ticker PnL comparison
print(f"\n  {'Ticker':<7} │ {'PnL-180':>8} {'PnL-200':>8} {'PnL-230':>8} {'PnL-240':>8} │ "
      f"{'Δ(240-180)':>10} {'Trades':>6} │ {'Winner':>6}")
print(f"  {'─'*7} │ {'─'*8} {'─'*8} {'─'*8} {'─'*8} │ {'─'*10} {'─'*6} │ {'─'*6}")

ticker_pnl = {}  # ticker → {exp: avg_pnl}
for ticker in eval_tickers:
    tp = {}
    tc = {}
    for exp_base in EXPERIMENTS:
        tt = [t for t in exp_trades[exp_base] if t['ticker'] == ticker]
        if tt:
            tp[exp_base] = np.mean([t['pnl'] for t in tt])
            tc[exp_base] = len(tt)
        else:
            tp[exp_base] = None
            tc[exp_base] = 0
    if tp[180] is None and tp[240] is None:
        continue
    ticker_pnl[ticker] = tp

    def _fp(v):
        return f"{v:+7.2%}" if v is not None else "    —   "

    delta = ""
    winner = "—"
    valid = {e: p for e, p in tp.items() if p is not None}
    if valid:
        winner = str(max(valid, key=valid.get))
    if tp[240] is not None and tp[180] is not None:
        delta = f"{(tp[240] - tp[180]):+.2%}"
    n_trades = max(tc.values()) if tc else 0

    print(f"  {ticker:<7} │ {_fp(tp[180]):>8} {_fp(tp[200]):>8} {_fp(tp[230]):>8} {_fp(tp[240]):>8} │ "
          f"{delta:>10} {n_trades:>6} │ {winner:>6}")


# ==============================================================================
# SECTION C: MONTHLY BREAKDOWN
# ==============================================================================
print()
print("=" * W)
print(f"  C. MONTHLY BREAKDOWN")
print("=" * W)

months = ['2026-04', '2026-05', '2026-06', '2026-07']
mlabels = ['Apr', 'May', 'Jun', 'Jul']

print(f"\n  {'Experiment':<22} │", end="")
for ml in mlabels:
    print(f" {ml+' Avg':>8} {ml+' #T':>5}", end="")
print()
print(f"  {'─'*22} │", end="")
for _ in mlabels:
    print(f" {'─'*8} {'─'*5}", end="")
print()

for exp_base, label in EXPERIMENTS.items():
    trades = exp_trades[exp_base]
    if not trades:
        continue
    df = pd.DataFrame(trades)
    filt = df.groupby('ticker').filter(lambda x: x['pnl'].mean() >= DEPLOY_THRESHOLD)
    if filt.empty:
        filt = df
    print(f"  {label:<22} │", end="")
    for m in months:
        mdf = filt[filt['month'] == m]
        if len(mdf) > 0:
            print(f" {mdf['pnl'].mean():>+7.2%} {len(mdf):>5}", end="")
        else:
            print(f" {'—':>8} {'—':>5}", end="")
    print()


# ==============================================================================
# SECTION D: PER-TICKER WINNER ANALYSIS
# ==============================================================================
print()
print("=" * W)
print(f"  D. PER-TICKER WINNER ANALYSIS")
print("=" * W)

wins = {exp: 0 for exp in EXPERIMENTS}

for ticker, tp in ticker_pnl.items():
    valid = {e: p for e, p in tp.items() if p is not None}
    if not valid:
        continue
    best_exp = max(valid, key=valid.get)
    wins[best_exp] += 1

print(f"\n  Overall: {len(ticker_pnl)} tickers compared")
for exp_base, label in EXPERIMENTS.items():
    pct = wins[exp_base] / len(ticker_pnl) * 100 if ticker_pnl else 0
    print(f"    {label}: {wins[exp_base]} wins ({pct:.0f}%)")

# Teacher-ticker analysis for 230 and 240
for cmp_exp, teacher_list in EXP_TEACHERS.items():
    teacher_in_eval = [t for t in teacher_list if t in ticker_pnl]
    if not teacher_in_eval:
        continue
    print(f"\n  {cmp_exp}'s Teachers ({len(teacher_in_eval)} in eval set) — PnL deltas ({cmp_exp} - 180):")
    teacher_deltas = []
    for ticker in sorted(teacher_in_eval):
        tp = ticker_pnl[ticker]
        if tp.get(cmp_exp) is not None and tp.get(180) is not None:
            d = tp[cmp_exp] - tp[180]
            teacher_deltas.append(d)
            tag = "↑" if d > 0.001 else ("↓" if d < -0.001 else "~")
            print(f"    {ticker:<7} {d:+.2%} {tag}")
    if teacher_deltas:
        print(f"    Mean Δ: {np.mean(teacher_deltas):+.3%}  "
              f"({sum(1 for d in teacher_deltas if d > 0)}/{len(teacher_deltas)} improved)")


# ==============================================================================
# SECTION E: CHARTS
# ==============================================================================
print(f"\n  Generating charts...")


def build_equity_curve(trades_list):
    if not trades_list:
        return pd.Series(dtype=float)
    df = pd.DataFrame(trades_list)
    filt = df.groupby('ticker').filter(lambda x: x['pnl'].mean() >= DEPLOY_THRESHOLD)
    if filt.empty:
        filt = df
    filt['date'] = pd.to_datetime(filt['date'])
    daily = filt.groupby('date')['pnl'].mean()
    return daily.sort_index().cumsum()


colors = {180: '#1f77b4', 200: '#ff7f0e', 230: '#2ca02c', 240: '#d62728'}
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: Cumulative PnL
ax1 = axes[0]
for exp_base, label in EXPERIMENTS.items():
    curve = build_equity_curve(exp_trades[exp_base])
    if curve.empty:
        continue
    final = curve.iloc[-1]
    ax1.plot(curve.index, curve.values,
             label=f"{label} ({final:+.1%})",
             color=colors[exp_base], linewidth=2)

june1 = pd.Timestamp('2026-06-01')
july1 = pd.Timestamp('2026-07-01')
ax1.axvline(june1, color='gray', linestyle=':', alpha=0.7)
ax1.axvline(july1, color='gray', linestyle=':', alpha=0.7)
ax1.set_title('Cumulative PnL (deploy threshold filtered)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cumulative Avg PnL')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.5)

# Chart 2: AUC scatter — 240 vs 180
ax2 = axes[1]
if HAS_SKLEARN:
    scatter_x, scatter_y, scatter_labels = [], [], []
    for ticker in eval_tickers:
        a180 = auc_results.get((ticker, 180))
        a240 = auc_results.get((ticker, 240))
        if a180 is not None and a240 is not None:
            scatter_x.append(a180)
            scatter_y.append(a240)
            scatter_labels.append(ticker)

    if scatter_x:
        is_teacher = [t in TEACHERS_240 for t in scatter_labels]
        xs = np.array(scatter_x)
        ys = np.array(scatter_y)
        mask_t = np.array(is_teacher)
        ax2.scatter(xs[~mask_t], ys[~mask_t], c='#1f77b4', alpha=0.6, s=40, label='Non-teacher')
        ax2.scatter(xs[mask_t], ys[mask_t], c='#d62728', alpha=0.8, s=60, marker='D', label='240 Teacher')

        mn = min(min(scatter_x), min(scatter_y)) - 0.02
        mx = max(max(scatter_x), max(scatter_y)) + 0.02
        ax2.plot([mn, mx], [mn, mx], 'k--', alpha=0.4, linewidth=1)
        ax2.set_xlim(mn, mx)
        ax2.set_ylim(mn, mx)

        for i, t in enumerate(scatter_labels):
            if abs(scatter_y[i] - scatter_x[i]) > 0.03:
                ax2.annotate(t, (scatter_x[i], scatter_y[i]), fontsize=7, alpha=0.7)

    ax2.set_xlabel('AUC — Exp 180')
    ax2.set_ylabel('AUC — Exp 240')
    ax2.set_title('AUC: 240 vs 180 (above diagonal = 240 wins)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'sklearn not available\nAUC scatter skipped',
             ha='center', va='center', fontsize=14, transform=ax2.transAxes)

plt.tight_layout()
plt.savefig('/tmp/compare_240_cumulative.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Charts saved: /tmp/compare_240_cumulative.png")


# ==============================================================================
# FINAL VERDICT
# ==============================================================================
print()
print("=" * W)
print(f"  VERDICT")
print("=" * W)

for cmp_exp in [230, 240]:
    deltas = ticker_auc_deltas.get(cmp_exp, [])
    if deltas:
        n_auc_better = sum(1 for _, d in deltas if d > 0)
        print(f"  Signal: {cmp_exp} wins AUC on {n_auc_better}/{len(deltas)} tickers "
              f"(avg Δ = {np.mean([d for _, d in deltas]):+.4f})")

if exp_stats:
    for exp_base, label in EXPERIMENTS.items():
        s = exp_stats.get(exp_base)
        if s:
            print(f"  {label}: {s['n_trades']} trades, avg={s['avg']:+.2%}, "
                  f"WR={s['wr']:.0%}, total={s['total']:+.1%}")

for cmp_exp in [230, 240]:
    if cmp_exp in exp_stats and 180 in exp_stats:
        d_avg = exp_stats[cmp_exp]['avg'] - exp_stats[180]['avg']
        d_wr = exp_stats[cmp_exp]['wr'] - exp_stats[180]['wr']
        verdict = "IMPROVEMENT" if d_avg > 0 else "DEGRADATION"
        print(f"\n  ★ {cmp_exp} vs 180: ΔAvgPnL = {d_avg:+.2%}, ΔWR = {d_wr:+.1%} → {verdict}")

print()
print(f"  NOTE: Trade simulation uses 180's BQ configs.")
print(f"  Run WF optimizer on 240 for definitive comparison with optimized TP/SL.")
print("=" * W)
