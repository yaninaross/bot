"""
backtest_composite_csv.py — Composite Scoring Method Comparison (CSV-Scored)
===========================================================================
Tests 5 composite signal gating methods × 2 variants (STATIC + ADAPTIVE):

Methods:
  1. Integer family agreement (agree=1,2,3)
  2. Percentage family agreement (pct=50%,60%,75%)
  3. Seed-level agreement (seed=40%,50%,60%,75%)
  4. Precision-weighted vote (threshold=50%)
  5. Teacher-aware voting (pct=60%)

Variants:
  STATIC: Fixed prob_thresh from BQ config
  ADAPTIVE: Dynamic prob_thresh adjusted by trailing precision

Scores from pre-computed CSV files on GCS. Limit_entry trade simulation.
Selective portfolio filter: only tickers with avg PnL >= 1.8%.
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


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
RESULTS_DIR = "Algos/Results_scored_20260703"
RESULTS_FALLBACK = "Algos/Results_all_v2"
INFERENCE_DIR = "Data/Inference_v2"

TEST_START = '2026-04-01'
TEST_END   = '2026-07-11'
ENTRY_MIN  = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT = 0.0015
STARTING_CAPITAL = 200_000
DEPLOY_THRESHOLD = 0.018
REOPT_FREQ = 5

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

ALL_TICKERS = []  # populated dynamically from BQ after sa_limit is built

MIN_SEEDS = 3
MIN_TRADES_BQ = 3
MAX_VA_OPT_RATIO = 3.0

GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']

TRAIN_TEACHERS_13 = ['BBAI','RCAT','LUNR','ASTS','ACHR','MVIS',
                     'WULF','CIFR','IREN','KEEL','HUT','CLSK','APLD']
TEACHERS_31 = ['CIFR','IREN','KEEL','HUT','APLD','WULF','RIOT','CLSK',
               'BTBT','ALAB','MARA','RCAT','ASTS','LUNR','BBAI','RKLB',
               'SPIR','BKSY','MNTS','PLUG','BE','FCEL','FLNC','OKLO',
               'SMR','MVIS','BGDE','RDW','PL','VOYG','SMLR']
TEACHERS_47 = TEACHERS_31 + [
    'AMD','NVDA','TSLA','PLTR','SOUN','HOOD','SMCI','VRT','NBIS',
    'KTOS','AVAV','ACHR','JOBY','NNDM','CCJ','CEG']

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


def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    if 200 <= num <= 205: return '200'
    if 220 <= num <= 225: return '220'
    return 'other'


def get_teacher_set(exp_name):
    base = exp_name.split('_')[0]
    if not base.isdigit():
        return set()
    num = int(base)
    if num in (152, 160):
        return set(TRAIN_TEACHERS_13)
    if 170 <= num <= 175:
        return set(TRAIN_TEACHERS_13)
    if 180 <= num <= 185:
        return set(TEACHERS_31)
    if 200 <= num <= 205:
        return set(TEACHERS_31)
    if 220 <= num <= 225:
        return set(TEACHERS_47)
    return set()


# ==============================================================================
# COMPOSITE GATE FUNCTIONS
# ==============================================================================
def gate_family_agree(config, ticker, test_dates, daily_probs, daily_trades,
                      active_exps, min_agree=3, min_present=2):
    if config is None:
        return []
    prob_thresh = config['prob_thresh']
    exp_by_fam = defaultdict(list)
    for exp in active_exps:
        fam = assign_family(exp)
        if fam != 'other':
            exp_by_fam[fam].append(exp)
    trades = []
    for d in test_dates:
        fam_present = 0
        fam_signal = 0
        for fam, exps in exp_by_fam.items():
            has_data = False
            has_signal = False
            for exp in exps:
                pk = (ticker, exp)
                if pk in daily_probs and d in daily_probs[pk]:
                    has_data = True
                    if daily_probs[pk][d] >= prob_thresh:
                        has_signal = True
            if has_data:
                fam_present += 1
            if has_signal:
                fam_signal += 1
        if fam_present >= min_present and \
           fam_signal >= min(min_agree, (fam_present // 2 + 1)):
            if ticker in daily_trades and d in daily_trades[ticker]:
                trades.append({'date': d, 'ticker': ticker,
                               'pnl': daily_trades[ticker][d],
                               'n_exps': fam_signal})
    return trades


def gate_pct_family(config, ticker, test_dates, daily_probs, daily_trades,
                    active_exps, pct_thresh=0.60, min_present=2):
    if config is None:
        return []
    prob_thresh = config['prob_thresh']
    exp_by_fam = defaultdict(list)
    for exp in active_exps:
        fam = assign_family(exp)
        if fam != 'other':
            exp_by_fam[fam].append(exp)
    trades = []
    for d in test_dates:
        fam_present = 0
        fam_signal = 0
        for fam, exps in exp_by_fam.items():
            has_data = False
            has_signal = False
            for exp in exps:
                pk = (ticker, exp)
                if pk in daily_probs and d in daily_probs[pk]:
                    has_data = True
                    if daily_probs[pk][d] >= prob_thresh:
                        has_signal = True
            if has_data:
                fam_present += 1
            if has_signal:
                fam_signal += 1
        if fam_present >= min_present and fam_present > 0 and \
           fam_signal / fam_present >= pct_thresh:
            if ticker in daily_trades and d in daily_trades[ticker]:
                trades.append({'date': d, 'ticker': ticker,
                               'pnl': daily_trades[ticker][d],
                               'n_exps': fam_signal})
    return trades


def gate_seed_agree(config, ticker, test_dates, daily_seed_probs, daily_trades,
                    active_exps, seed_pct=0.60, min_seeds=15):
    if config is None:
        return []
    prob_thresh = config['prob_thresh']
    seed_keys = [(t, exp, sid) for (t, exp, sid) in daily_seed_probs
                 if t == ticker and exp in active_exps]
    if not seed_keys:
        return []
    trades = []
    for d in test_dates:
        total_seeds = 0
        agree_seeds = 0
        for key in seed_keys:
            sprobs = daily_seed_probs[key]
            if d in sprobs:
                total_seeds += 1
                if sprobs[d] >= prob_thresh:
                    agree_seeds += 1
        if total_seeds >= min_seeds and \
           agree_seeds / total_seeds >= seed_pct:
            if ticker in daily_trades and d in daily_trades[ticker]:
                trades.append({'date': d, 'ticker': ticker,
                               'pnl': daily_trades[ticker][d],
                               'n_exps': agree_seeds})
    return trades


def gate_teacher_aware(config, ticker, test_dates, daily_probs, daily_trades,
                       active_exps, pct_thresh=0.60, min_present=1):
    if config is None:
        return []
    prob_thresh = config['prob_thresh']
    relevant_exps = [e for e in active_exps if ticker in get_teacher_set(e)]
    if not relevant_exps:
        relevant_exps = active_exps
    exp_by_fam = defaultdict(list)
    for exp in relevant_exps:
        fam = assign_family(exp)
        if fam != 'other':
            exp_by_fam[fam].append(exp)
    trades = []
    for d in test_dates:
        fam_present = 0
        fam_signal = 0
        for fam, exps in exp_by_fam.items():
            has_data = False
            has_signal = False
            for exp in exps:
                pk = (ticker, exp)
                if pk in daily_probs and d in daily_probs[pk]:
                    has_data = True
                    if daily_probs[pk][d] >= prob_thresh:
                        has_signal = True
            if has_data:
                fam_present += 1
            if has_signal:
                fam_signal += 1
        if fam_present >= min_present and fam_present > 0 and \
           fam_signal / fam_present >= pct_thresh:
            if ticker in daily_trades and d in daily_trades[ticker]:
                trades.append({'date': d, 'ticker': ticker,
                               'pnl': daily_trades[ticker][d],
                               'n_exps': fam_signal})
    return trades


def gate_weighted_vote(config, ticker, test_dates, daily_probs, daily_trades,
                       active_exps, exp_precision, vote_thresh=0.50):
    if config is None:
        return []
    prob_thresh = config['prob_thresh']
    trades = []
    for d in test_dates:
        total_weight = 0
        agree_weight = 0
        for exp in active_exps:
            pk = (ticker, exp)
            if pk not in daily_probs:
                continue
            w = exp_precision.get((ticker, exp), 0.5)
            total_weight += w
            if d in daily_probs[pk] and daily_probs[pk][d] >= prob_thresh:
                agree_weight += w
        if total_weight > 0 and agree_weight / total_weight > vote_thresh:
            if ticker in daily_trades and d in daily_trades[ticker]:
                trades.append({'date': d, 'ticker': ticker,
                               'pnl': daily_trades[ticker][d],
                               'n_exps': 0})
    return trades


# ==============================================================================
# PORTFOLIO ANALYSIS
# ==============================================================================
def selective_portfolio(trades_list, label, min_avg=DEPLOY_THRESHOLD):
    if not trades_list:
        return None
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    per_ticker = []
    for ticker in sorted(df['ticker'].unique()):
        tt = df[df['ticker'] == ticker]
        june = tt[tt['month'] == '2026-06']
        avg = tt['pnl'].mean()
        if avg < min_avg:
            continue
        per_ticker.append({
            'ticker': ticker, 'trades': len(tt), 'avg': avg,
            'total': tt['pnl'].sum(), 'wr': (tt['pnl'] > 0).mean(),
            'j_n': len(june),
            'j_avg': june['pnl'].mean() if len(june) else 0,
            'j_wr': (june['pnl'] > 0).mean() if len(june) else 0,
        })
    if not per_ticker:
        return None
    keep = set(r['ticker'] for r in per_ticker)
    filt = df[df['ticker'].isin(keep)]
    cum = filt.sort_values('date')['pnl'].cumsum()
    june_all = filt[filt['month'] == '2026-06']
    return {
        'label': label, 'per_ticker': per_ticker,
        'n': len(filt), 'total': filt['pnl'].sum(),
        'avg': filt['pnl'].mean(), 'dd': (cum - cum.cummax()).min(),
        'wr': (filt['pnl'] > 0).mean(),
        'j_n': len(june_all),
        'j_avg': june_all['pnl'].mean() if len(june_all) else 0,
        'j_total': june_all['pnl'].sum() if len(june_all) else 0,
        'n_tickers': len(per_ticker),
    }


def display_portfolio(ps, benchmark_avg=None):
    if ps is None:
        print(f"    (no tickers survive {DEPLOY_THRESHOLD:.1%} threshold)")
        return
    tag = ""
    if benchmark_avg is not None:
        if ps['avg'] >= benchmark_avg and ps.get('j_avg', 0) > 0:
            tag = " ★ BEATS BENCHMARK + JUNE POSITIVE"
        elif ps['avg'] >= benchmark_avg:
            tag = " ✓ beats benchmark avg"
    print(f"    Portfolio: {ps['n_tickers']} tickers, {ps['n']} trades, "
          f"avg={ps['avg']:+.2%}, total={ps['total']:+.1%}, "
          f"WR={ps['wr']:.0%}, MaxDD={ps['dd']:+.1%}")
    print(f"    June: {ps['j_n']} trades, avg={ps.get('j_avg',0):+.2%}, "
          f"total={ps.get('j_total',0):+.1%}{tag}")
    print(f"    {'Ticker':<7} {'Trades':>6} {'AvgPnL':>8} {'TotPnL':>9} {'WR':>6}"
          f" │ {'JunT':>5} {'JunAvg':>8} {'JunWR':>6}")
    print(f"    {'─'*7} {'─'*6} {'─'*8} {'─'*9} {'─'*6}"
          f" │ {'─'*5} {'─'*8} {'─'*6}")
    for t in sorted(ps['per_ticker'], key=lambda x: x['avg'], reverse=True):
        jm = " ✓" if t['j_avg'] > 0 and t['j_n'] > 0 else ""
        print(f"    {t['ticker']:<7} {t['trades']:>6} {t['avg']:>+7.2%} "
              f"{t['total']:>+8.1%} {t['wr']:>5.0%}"
              f" │ {t['j_n']:>5} {t['j_avg']:>+7.2%} {t['j_wr']:>5.0%}{jm}")


def run_adaptive(gate_func, gate_args, all_tickers, bq_configs,
                 all_trading_dates, reopt_freq=REOPT_FREQ):
    reopt_indices = list(range(0, len(all_trading_dates), reopt_freq))
    all_trades = []
    for ticker in all_tickers:
        cfg = bq_configs.get(ticker)
        if cfg is None:
            continue
        base_thresh = cfg['prob_thresh']
        current_thresh = base_thresh
        recent_results = []
        for ri in reopt_indices:
            test_end_idx = min(ri + reopt_freq, len(all_trading_dates))
            test_dates = all_trading_dates[ri:test_end_idx]
            if len(recent_results) >= 3:
                last10 = recent_results[-10:]
                precision = sum(1 for r in last10 if r > 0) / len(last10)
                if precision < 0.40:
                    current_thresh = min(0.90, current_thresh + 0.10)
                elif precision > 0.60:
                    current_thresh = max(0.30, current_thresh - 0.10)
            adj_config = dict(cfg)
            adj_config['prob_thresh'] = current_thresh
            wt = gate_func(adj_config, ticker, test_dates, *gate_args)
            all_trades.extend(wt)
            recent_results.extend([t['pnl'] for t in wt])
    return all_trades


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
print("=" * W)
print(f"  COMPOSITE SCORING METHOD COMPARISON — CSV-Scored, limit_entry")
print("=" * W)


# ── Step 1: Load BQ configs ──
print(f"\n  STEP 1: Loading BQ configs (limit_entry, individual seeds)...")
q = """
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult, grace_minutes,
       dip_pct, max_wait_minutes, vol_filter_pct,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, opt_avg_pnl, is_ensemble
FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
WHERE model_version IN ('robust_wf_v12', 'robust_wf_v12_stitched')
  AND survived_a = TRUE
  AND val_a_count >= 3
  AND val_a_avg_pnl > 0
"""
raw_df = bq_client.query(q).to_dataframe()
print(f"    {len(raw_df)} raw BQ rows loaded")

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
sa_limit = sa[sa['tpsl_mode'] == 'limit_entry'].copy()
print(f"    {len(sa_limit)} limit_entry seed groups (n_seeds >= {MIN_SEEDS})")

sa_limit['score'] = (sa_limit['mean_va_pnl'] ** 2) * sa_limit['agree_18']

# S3 family agreement for benchmark
sa_limit['exp_fam'] = sa_limit['experiment'].apply(assign_family)
sa_fam = sa_limit[sa_limit['exp_fam'] != 'other'].copy()
best_per_fam = sa_fam.loc[sa_fam.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
best_per_fam = best_per_fam[best_per_fam['mean_va_pnl'] >= DEPLOY_THRESHOLD]

agreement = {}
for tk in best_per_fam['ticker'].unique():
    tk_rows = best_per_fam[best_per_fam['ticker'] == tk]
    n_fams = len(tk_rows)
    n_pos = int((tk_rows['vb_pnl'] > 0).sum())
    agreement[tk] = {
        'n_fams': n_fams, 'n_pos': n_pos,
        'majority_pos': n_pos > n_fams / 2 and n_fams >= 2,
    }

s3_tickers = sorted([tk for tk, ag in agreement.items() if ag['majority_pos']])
print(f"    Family agreement: {len(s3_tickers)} tickers pass S3 gate")
print(f"    S3 tickers: {', '.join(s3_tickers)}")

# S3 family agreement WITHOUT 200 family (for TABLE E comparison)
sa_fam_no200 = sa_fam[sa_fam['exp_fam'] != '200'].copy()
best_per_fam_no200 = sa_fam_no200.loc[sa_fam_no200.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
best_per_fam_no200 = best_per_fam_no200[best_per_fam_no200['mean_va_pnl'] >= DEPLOY_THRESHOLD]

agreement_no200 = {}
for tk in best_per_fam_no200['ticker'].unique():
    tk_rows = best_per_fam_no200[best_per_fam_no200['ticker'] == tk]
    n_fams = len(tk_rows)
    n_pos = int((tk_rows['vb_pnl'] > 0).sum())
    agreement_no200[tk] = {'n_fams': n_fams, 'n_pos': n_pos,
                           'majority_pos': n_pos > n_fams / 2 and n_fams >= 2}

s3_tickers_no200 = sorted([tk for tk, ag in agreement_no200.items() if ag['majority_pos']])
print(f"    S3 without 200: {len(s3_tickers_no200)} tickers: {', '.join(s3_tickers_no200)}")

# Derive ALL_TICKERS dynamically: any ticker with a qualifying limit_entry config
_qualifying = sa_limit[sa_limit['mean_va_pnl'] >= DEPLOY_THRESHOLD]['ticker'].unique()
ALL_TICKERS = sorted(set(_qualifying) | set(s3_tickers) | set(s3_tickers_no200))
print(f"    Dynamic ALL_TICKERS: {len(ALL_TICKERS)} tickers")

# Best config per ticker (for trade sim params)
bq_configs = {}
for ticker in ALL_TICKERS:
    tk_rows = sa_limit[sa_limit['ticker'] == ticker].copy()
    if tk_rows.empty:
        bq_configs[ticker] = None
        continue
    tk_rows['score'] = (tk_rows['mean_va_pnl'] ** 2) * tk_rows['agree_18']
    best = tk_rows.loc[tk_rows['score'].idxmax()]
    if best['mean_va_pnl'] < DEPLOY_THRESHOLD:
        bq_configs[ticker] = None
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

# Determine needed (ticker, experiment) combos
needed_combos = set()
for _, row in sa_limit[sa_limit['ticker'].isin(ALL_TICKERS)].iterrows():
    needed_combos.add((row['ticker'], row['experiment']))
for tk, cfg in bq_configs.items():
    if cfg is not None:
        needed_combos.add((tk, cfg['experiment']))

all_bq_experiments = sorted(set(exp for _, exp in needed_combos))
ACTIVE_EXPS = all_bq_experiments
print(f"    {len(all_bq_experiments)} experiments to load: {', '.join(all_bq_experiments)}")

for ticker in ALL_TICKERS:
    cfg = bq_configs.get(ticker)
    if cfg is None:
        print(f"    {ticker}: no qualifying limit_entry config")
        continue
    s3_flag = "S3" if ticker in s3_tickers else "  "
    print(f"    {s3_flag} {ticker}: prob>{cfg['prob_thresh']:.2f} TP={cfg['tp']:.0%} "
          f"SL={cfg['sl']:.0%} dip={cfg['dip_pct']:.1%} wait={cfg['max_wait_minutes']:.0f}m "
          f"avgPnL={cfg['avg_pnl']:.2%}")

del raw_df, indiv, sa, sa_limit, sa_fam, best_per_fam
_release_memory()


# ── Step 2: Load OHLCV ──
print(f"\n  STEP 2: Loading OHLCV data...")

ohlcv_arrays = {}
for ticker in ALL_TICKERS:
    df = load_master(ticker)
    if df is None:
        print(f"    {ticker}: FAILED to load")
        continue
    df = heal_data(df.sort_index())
    ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
    if ohlcv is not None:
        ohlcv_arrays[ticker] = ohlcv
    n_days = len(np.unique(ohlcv['dates'])) if ohlcv is not None else 0
    print(f"    {ticker}: {len(df):>8,} bars, {n_days} test days")
    del df; gc.collect()

_release_memory()


# ── Step 3: Load CSV scores ──
print(f"\n  STEP 3: Loading CSV scores from GCS...")

daily_probs = {}
daily_seed_probs = {}
scores_cache = {}

for bq_exp in all_bq_experiments:
    exp_base = bq_exp.replace('_LIMIT', '')
    tickers_for_exp = [tk for tk in ALL_TICKERS if (tk, bq_exp) in needed_combos]
    if not tickers_for_exp:
        continue
    n_loaded = 0
    for ticker in tickers_for_exp:
        cache_key = (ticker, exp_base)
        if cache_key not in scores_cache:
            seeds_dict = load_seed_scores(ticker, exp_base)
            scores_cache[cache_key] = seeds_dict
        else:
            seeds_dict = scores_cache[cache_key]

        if not seeds_dict:
            continue

        cfg = bq_configs.get(ticker)
        entry_min = cfg['entry_minute'] if cfg else ENTRY_MIN

        avg_probs = None
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

            daily_seed_probs[(ticker, bq_exp, seed_id)] = daily_prob.to_dict()
            if avg_probs is None:
                avg_probs = daily_prob.copy()
            else:
                avg_probs = avg_probs.add(daily_prob, fill_value=0)
            n_seeds += 1

        if n_seeds > 0 and avg_probs is not None:
            daily_probs[(ticker, bq_exp)] = (avg_probs / n_seeds).to_dict()
            n_loaded += 1

    print(f"    {bq_exp} -> Exp{exp_base}: loaded {n_loaded}/{len(tickers_for_exp)} tickers")

del scores_cache
_release_memory()

print(f"    Total: {len(daily_probs)} daily_probs, {len(daily_seed_probs)} seed_probs")


# ── Step 4: Pre-simulate trades (limit_entry) ──
print(f"\n  STEP 4: Pre-simulating trades (limit_entry mode)...")

daily_trades = {}
test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt = pd.Timestamp(TEST_END).date()
all_trading_dates = set()

for ticker in ALL_TICKERS:
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
    n_unfilled = 0
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
        else:
            n_unfilled += 1
    if outcomes:
        daily_trades[ticker] = outcomes
        all_trading_dates.update(outcomes.keys())
    n_total = len(outcomes) + n_unfilled
    print(f"    {ticker}: {len(outcomes)} filled / {n_total} days "
          f"(TP={tp:.0%} SL={sl:.0%} dip={dip_pct:.1%} wait={max_wait:.0f}m)")

all_trading_dates = sorted(all_trading_dates)
print(f"    {len(all_trading_dates)} test days with at least one fill")

del ohlcv_arrays
_release_memory()


# ── Step 5: Compute precision weights ──
print(f"\n  STEP 5: Computing precision weights for weighted vote...")

exp_precision = {}
pre_test_dates = [d for d in all_trading_dates if d < test_start_dt]

for ticker in ALL_TICKERS:
    cfg = bq_configs.get(ticker)
    if cfg is None:
        continue
    for exp in ACTIVE_EXPS:
        pk = (ticker, exp)
        if pk not in daily_probs:
            continue
        probs = daily_probs[pk]
        outcomes = daily_trades.get(ticker, {})
        n_sig, n_prof = 0, 0
        for d in all_trading_dates:
            if d in probs and probs[d] >= cfg['prob_thresh'] and d in outcomes:
                n_sig += 1
                if outcomes[d] > 0:
                    n_prof += 1
        exp_precision[(ticker, exp)] = n_prof / n_sig if n_sig >= 3 else 0.5

print(f"    Computed precision for {len(exp_precision)} (ticker, exp) pairs")


# ==============================================================================
# STEP 6: COMPOSITE SWEEP
# ==============================================================================
print()
print("=" * W)
print(f"  STEP 6: COMPOSITE SWEEP — 5 methods × 2 variants")
print("=" * W)


# ── TABLE A: S3 BENCHMARK ──
print(f"\n  TABLE A: S3 BENCHMARK — deployed tickers, BQ family agreement")

bench_trades = []
for ticker in s3_tickers:
    cfg = bq_configs.get(ticker)
    if cfg is None:
        continue
    exp = cfg['experiment']
    for d in all_trading_dates:
        pk = (ticker, exp)
        if pk in daily_probs and d in daily_probs[pk]:
            if daily_probs[pk][d] >= cfg['prob_thresh']:
                if ticker in daily_trades and d in daily_trades[ticker]:
                    bench_trades.append({'date': d, 'ticker': ticker,
                                         'pnl': daily_trades[ticker][d]})

bench_ps = selective_portfolio(bench_trades, 'S3 BENCHMARK', min_avg=-999)
display_portfolio(bench_ps)
BENCHMARK_AVG = bench_ps['avg'] if bench_ps and bench_ps['n'] > 0 else 0.025
print(f"\n    >>> BENCHMARK: {BENCHMARK_AVG:+.2%} avg PnL to beat")


# ── TABLE B: STATIC + composite gates (selective) ──
print()
print("=" * W)
print(f"  TABLE B: STATIC STRATEGY + COMPOSITE GATES (selective portfolios)")
print("=" * W)

all_sweep = []

# 1. Integer family agreement
for min_agree in [1, 2, 3]:
    trades = []
    for ticker in ALL_TICKERS:
        trades.extend(gate_family_agree(
            bq_configs.get(ticker), ticker, all_trading_dates,
            daily_probs, daily_trades, ACTIVE_EXPS,
            min_agree=min_agree, min_present=2))
    label = f"STATIC fam_agree={min_agree}"
    ps = selective_portfolio(trades, label)
    all_sweep.append(ps)
    print(f"\n  ── {label} ({len(trades)} raw trades) ──")
    display_portfolio(ps, BENCHMARK_AVG)

# 2. Percentage family agreement
for pct in [0.50, 0.60, 0.75]:
    trades = []
    for ticker in ALL_TICKERS:
        trades.extend(gate_pct_family(
            bq_configs.get(ticker), ticker, all_trading_dates,
            daily_probs, daily_trades, ACTIVE_EXPS,
            pct_thresh=pct, min_present=2))
    label = f"STATIC pct_fam={pct:.0%}"
    ps = selective_portfolio(trades, label)
    all_sweep.append(ps)
    print(f"\n  ── {label} ({len(trades)} raw trades) ──")
    display_portfolio(ps, BENCHMARK_AVG)

# 3. Seed-level agreement
for seed_pct in [0.40, 0.50, 0.60, 0.75]:
    trades = []
    for ticker in ALL_TICKERS:
        trades.extend(gate_seed_agree(
            bq_configs.get(ticker), ticker, all_trading_dates,
            daily_seed_probs, daily_trades, ACTIVE_EXPS,
            seed_pct=seed_pct, min_seeds=15))
    label = f"STATIC seed={seed_pct:.0%}/15"
    ps = selective_portfolio(trades, label)
    all_sweep.append(ps)
    print(f"\n  ── {label} ({len(trades)} raw trades) ──")
    display_portfolio(ps, BENCHMARK_AVG)

# 4. Precision-weighted vote
trades = []
for ticker in ALL_TICKERS:
    trades.extend(gate_weighted_vote(
        bq_configs.get(ticker), ticker, all_trading_dates,
        daily_probs, daily_trades, ACTIVE_EXPS,
        exp_precision, vote_thresh=0.50))
label = "STATIC weighted>50%"
ps = selective_portfolio(trades, label)
all_sweep.append(ps)
print(f"\n  ── {label} ({len(trades)} raw trades) ──")
display_portfolio(ps, BENCHMARK_AVG)

# 5. Teacher-aware voting
for pct in [0.60]:
    trades = []
    for ticker in ALL_TICKERS:
        trades.extend(gate_teacher_aware(
            bq_configs.get(ticker), ticker, all_trading_dates,
            daily_probs, daily_trades, ACTIVE_EXPS,
            pct_thresh=pct, min_present=1))
    label = f"STATIC teacher={pct:.0%}"
    ps = selective_portfolio(trades, label)
    all_sweep.append(ps)
    print(f"\n  ── {label} ({len(trades)} raw trades) ──")
    display_portfolio(ps, BENCHMARK_AVG)


# ── TABLE C: ADAPTIVE + composite gates ──
print()
print("=" * W)
print(f"  TABLE C: ADAPTIVE THRESHOLD + COMPOSITE GATES (selective portfolios)")
print("=" * W)
print(f"  Dynamic prob_thresh: +0.10 when trailing precision < 40%, -0.10 when > 60%")

# 1. Adaptive family agreement
for min_agree in [1, 2, 3]:
    trades = run_adaptive(
        gate_family_agree,
        (daily_probs, daily_trades, ACTIVE_EXPS, min_agree, 2),
        ALL_TICKERS, bq_configs, all_trading_dates)
    label = f"ADAPTIVE fam_agree={min_agree}"
    ps = selective_portfolio(trades, label)
    all_sweep.append(ps)
    print(f"\n  ── {label} ({len(trades)} raw trades) ──")
    display_portfolio(ps, BENCHMARK_AVG)

# 2. Adaptive percentage family
for pct in [0.50, 0.60, 0.75]:
    trades = run_adaptive(
        gate_pct_family,
        (daily_probs, daily_trades, ACTIVE_EXPS, pct, 2),
        ALL_TICKERS, bq_configs, all_trading_dates)
    label = f"ADAPTIVE pct_fam={pct:.0%}"
    ps = selective_portfolio(trades, label)
    all_sweep.append(ps)
    print(f"\n  ── {label} ({len(trades)} raw trades) ──")
    display_portfolio(ps, BENCHMARK_AVG)

# 3. Adaptive seed agreement
for seed_pct in [0.40, 0.50, 0.60, 0.75]:
    trades = run_adaptive(
        gate_seed_agree,
        (daily_seed_probs, daily_trades, ACTIVE_EXPS, seed_pct, 15),
        ALL_TICKERS, bq_configs, all_trading_dates)
    label = f"ADAPTIVE seed={seed_pct:.0%}/15"
    ps = selective_portfolio(trades, label)
    all_sweep.append(ps)
    print(f"\n  ── {label} ({len(trades)} raw trades) ──")
    display_portfolio(ps, BENCHMARK_AVG)

# 4. Adaptive weighted vote
trades = run_adaptive(
    gate_weighted_vote,
    (daily_probs, daily_trades, ACTIVE_EXPS, exp_precision, 0.50),
    ALL_TICKERS, bq_configs, all_trading_dates)
label = "ADAPTIVE weighted>50%"
ps = selective_portfolio(trades, label)
all_sweep.append(ps)
print(f"\n  ── {label} ({len(trades)} raw trades) ──")
display_portfolio(ps, BENCHMARK_AVG)

# 5. Adaptive teacher-aware
trades = run_adaptive(
    gate_teacher_aware,
    (daily_probs, daily_trades, ACTIVE_EXPS, 0.60, 1),
    ALL_TICKERS, bq_configs, all_trading_dates)
label = "ADAPTIVE teacher=60%"
ps = selective_portfolio(trades, label)
all_sweep.append(ps)
print(f"\n  ── {label} ({len(trades)} raw trades) ──")
display_portfolio(ps, BENCHMARK_AVG)


# ── TABLE D: WINNERS SUMMARY ──
print()
print("=" * W)
print(f"  TABLE D: METHODS THAT BEAT S3 BENCHMARK ({BENCHMARK_AVG:+.2%} avg)")
print("=" * W)

valid = [ps for ps in all_sweep if ps is not None]
winners = [ps for ps in valid if ps['avg'] >= BENCHMARK_AVG]
winners.sort(key=lambda x: x['avg'], reverse=True)

if not winners:
    print(f"\n  No methods beat benchmark. Showing top 5 by avg PnL:")
    winners = sorted(valid, key=lambda x: x['avg'], reverse=True)[:5]

print(f"\n  {'Method':<28} │ {'#Tk':>3} {'Trd':>4} {'AvgPnL':>7} {'TotPnL':>8} "
      f"{'WR':>4} {'MaxDD':>7} │ {'JAvg':>6} {'JTot':>7} {'JunOK':>5} │ Tickers")
print(f"  {'─'*28} │ {'─'*3} {'─'*4} {'─'*7} {'─'*8} "
      f"{'─'*4} {'─'*7} │ {'─'*6} {'─'*7} {'─'*5} │ {'─'*30}")

for ps in winners:
    tickers = [t['ticker'] for t in sorted(ps['per_ticker'],
               key=lambda x: x['avg'], reverse=True)]
    tstr = ','.join(tickers[:8])
    if len(tickers) > 8:
        tstr += f"+{len(tickers)-8}"
    jok = "✓" if ps.get('j_avg', 0) > 0 else "✗"
    beat = "★" if ps['avg'] >= BENCHMARK_AVG and ps.get('j_avg', 0) > 0 else \
           "✓" if ps['avg'] >= BENCHMARK_AVG else " "
    print(f" {beat}{ps['label']:<28} │ {ps['n_tickers']:>3} {ps['n']:>4} "
          f"{ps['avg']:>+6.2%} {ps['total']:>+7.1%} "
          f"{ps['wr']:>3.0%} {ps['dd']:>+6.1%} │ "
          f"{ps.get('j_avg',0):>+5.2%} {ps.get('j_total',0):>+6.1%} {jok:>5} │ {tstr}")


# ── BEST RESULT ──
best = None
for ps in valid:
    if ps['avg'] >= BENCHMARK_AVG and ps.get('j_avg', 0) > 0:
        if best is None or ps['avg'] > best['avg']:
            best = ps

if best is None:
    best_no_june = max(valid, key=lambda x: x['avg']) if valid else None
    if best_no_june:
        print(f"\n  No method beats benchmark AND is June-positive.")
        print(f"  Best avg overall: {best_no_june['label']} at {best_no_june['avg']:+.2%}")
else:
    print(f"\n  >>> BEST: {best['label']}")
    print(f"      {best['avg']:+.2%} avg PnL ({best['avg']/BENCHMARK_AVG - 1:+.0%} vs benchmark)")
    print(f"      June: {best.get('j_avg',0):+.2%} avg, {best.get('j_n',0)} trades")
    print(f"      Tickers: {', '.join(t['ticker'] for t in sorted(best['per_ticker'], key=lambda x: x['avg'], reverse=True))}")

# ── TABLE E: 200 FAMILY IMPACT — S3 + WEIGHTED>50% ──
print()
print("=" * W)
print(f"  TABLE E: 200 FAMILY IMPACT — S3 benchmark vs STATIC weighted>50%")
print("=" * W)

EXPS_WITHOUT_200 = [e for e in ACTIVE_EXPS if assign_family(e) != '200']
print(f"  Experiments: {len(ACTIVE_EXPS)} total  |  {len(EXPS_WITHOUT_200)} without 200 family")
print(f"  200 family exps: {', '.join(e for e in ACTIVE_EXPS if assign_family(e) == '200')}")

# — Section A: S3 with vs without 200 —
print(f"\n  ── A. S3 BENCHMARK: with vs without 200 family ──")

# S3 WITH 200 — reuse bench_trades from TABLE A
print(f"\n  S3 WITH 200 ({len(s3_tickers)} tickers: {', '.join(s3_tickers)})")
s3_w200_ps = selective_portfolio(bench_trades, 'S3 w/ 200', min_avg=-999)
display_portfolio(s3_w200_ps)

# S3 WITHOUT 200
bench_trades_no200 = []
for ticker in s3_tickers_no200:
    cfg = bq_configs.get(ticker)
    if cfg is None:
        continue
    exp = cfg['experiment']
    for d in all_trading_dates:
        pk = (ticker, exp)
        if pk in daily_probs and d in daily_probs[pk]:
            if daily_probs[pk][d] >= cfg['prob_thresh']:
                if ticker in daily_trades and d in daily_trades[ticker]:
                    bench_trades_no200.append({'date': d, 'ticker': ticker,
                                               'pnl': daily_trades[ticker][d]})

print(f"\n  S3 WITHOUT 200 ({len(s3_tickers_no200)} tickers: {', '.join(s3_tickers_no200)})")
s3_wo200_ps = selective_portfolio(bench_trades_no200, 'S3 w/o 200', min_avg=-999)
display_portfolio(s3_wo200_ps)

if s3_w200_ps and s3_wo200_ps:
    d_avg = s3_w200_ps['avg'] - s3_wo200_ps['avg']
    d_jun = s3_w200_ps.get('j_avg', 0) - s3_wo200_ps.get('j_avg', 0)
    d_tot = s3_w200_ps['total'] - s3_wo200_ps['total']
    w_tk = set(t['ticker'] for t in s3_w200_ps['per_ticker'])
    wo_tk = set(t['ticker'] for t in s3_wo200_ps['per_ticker'])
    added = w_tk - wo_tk
    removed = wo_tk - w_tk
    print(f"\n  S3 200-IMPACT: ΔAvg={d_avg:+.2%}  ΔJuneAvg={d_jun:+.2%}  ΔTotalPnL={d_tot:+.1%}")
    if added:
        print(f"    200 adds tickers: {', '.join(sorted(added))}")
    if removed:
        print(f"    200 drops tickers: {', '.join(sorted(removed))}")
    if not added and not removed:
        print(f"    Same ticker set (gate change only)")

# — Section B: weighted>50% with vs without 200 —
print(f"\n  ── B. STATIC weighted>50%: with vs without 200 family ──")

# weighted WITH 200
wt_trades_w200 = [t for tk in ALL_TICKERS
                  for t in gate_weighted_vote(bq_configs.get(tk), tk, all_trading_dates,
                                              daily_probs, daily_trades, ACTIVE_EXPS,
                                              exp_precision, vote_thresh=0.50)]
print(f"\n  WEIGHTED>50% WITH 200")
wt_w200_ps = selective_portfolio(wt_trades_w200, 'weighted>50% w/ 200')
display_portfolio(wt_w200_ps, benchmark_avg=BENCHMARK_AVG)

# weighted WITHOUT 200
wt_trades_wo200 = [t for tk in ALL_TICKERS
                   for t in gate_weighted_vote(bq_configs.get(tk), tk, all_trading_dates,
                                               daily_probs, daily_trades, EXPS_WITHOUT_200,
                                               exp_precision, vote_thresh=0.50)]
print(f"\n  WEIGHTED>50% WITHOUT 200")
wt_wo200_ps = selective_portfolio(wt_trades_wo200, 'weighted>50% w/o 200')
display_portfolio(wt_wo200_ps, benchmark_avg=BENCHMARK_AVG)

if wt_w200_ps and wt_wo200_ps:
    d_avg = wt_w200_ps['avg'] - wt_wo200_ps['avg']
    d_jun = wt_w200_ps.get('j_avg', 0) - wt_wo200_ps.get('j_avg', 0)
    d_tot = wt_w200_ps['total'] - wt_wo200_ps['total']
    w_tk = set(t['ticker'] for t in wt_w200_ps['per_ticker'])
    wo_tk = set(t['ticker'] for t in wt_wo200_ps['per_ticker'])
    added = w_tk - wo_tk
    removed = wo_tk - w_tk
    print(f"\n  WEIGHTED 200-IMPACT: ΔAvg={d_avg:+.2%}  ΔJuneAvg={d_jun:+.2%}  ΔTotalPnL={d_tot:+.1%}")
    if added:
        print(f"    200 adds tickers: {', '.join(sorted(added))}")
    if removed:
        print(f"    200 drops tickers: {', '.join(sorted(removed))}")
    if not added and not removed:
        print(f"    Same ticker set")

# — Section C: Combined comparison table —
print(f"\n  ── C. COMBINED 200-IMPACT SUMMARY ──")
print(f"\n  {'Strategy':<22} │ {'Tickers':>7} {'Trades':>6} {'AvgPnL':>8} {'TotalPnL':>9} "
      f"│ {'JunAvg':>8} {'JunTot':>8} {'JunN':>5} │ {'WR':>4}")
print(f"  {'─'*22} │ {'─'*7} {'─'*6} {'─'*8} {'─'*9} │ {'─'*8} {'─'*8} {'─'*5} │ {'─'*4}")

summary_rows = [
    ('S3 w/ 200', s3_w200_ps),
    ('S3 w/o 200', s3_wo200_ps),
    ('weighted>50% w/ 200', wt_w200_ps),
    ('weighted>50% w/o 200', wt_wo200_ps),
]

for label, ps in summary_rows:
    if ps is None:
        print(f"  {label:<22} │ {'—':>7} {'—':>6} {'—':>8} {'—':>9} │ {'—':>8} {'—':>8} {'—':>5} │ {'—':>4}")
        continue
    print(f"  {label:<22} │ {ps['n_tickers']:>7} {ps['n']:>6} {ps['avg']:>+7.2%} {ps['total']:>+8.1%} "
          f"│ {ps.get('j_avg',0):>+7.2%} {ps.get('j_total',0):>+7.1%} {ps.get('j_n',0):>5} │ {ps['wr']:>3.0%}")

# ── CUMULATIVE PNL CHARTS ──
print(f"\n  Generating cumulative PnL charts...")

def build_equity_curve(trades_list):
    if not trades_list:
        return pd.Series(dtype=float)
    df = pd.DataFrame(trades_list)
    df['date'] = pd.to_datetime(df['date'])
    daily = df.groupby('date')['pnl'].mean()
    return daily.sort_index().cumsum()

curves = {}
curves['S3 w/ 200'] = build_equity_curve(bench_trades)
curves['S3 w/o 200'] = build_equity_curve(bench_trades_no200)
curves['weighted>50% w/ 200'] = build_equity_curve(wt_trades_w200)
curves['weighted>50% w/o 200'] = build_equity_curve(wt_trades_wo200)

colors = {'S3 w/ 200': '#1f77b4', 'S3 w/o 200': '#aec7e8',
          'weighted>50% w/ 200': '#d62728', 'weighted>50% w/o 200': '#ff9896'}
styles = {'S3 w/ 200': '-', 'S3 w/o 200': '--',
          'weighted>50% w/ 200': '-', 'weighted>50% w/o 200': '--'}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 2]})

june1 = pd.Timestamp('2026-06-01')

# Top: full period
for label, curve in curves.items():
    if curve.empty:
        continue
    final = curve.iloc[-1] if len(curve) else 0
    ax1.plot(curve.index, curve.values, label=f"{label} ({final:+.1%})",
             color=colors[label], linestyle=styles[label], linewidth=2)

ax1.axvline(june1, color='gray', linestyle=':', alpha=0.7, label='June 1')
ax1.set_title('Cumulative PnL: S3 vs weighted>50% — 200 Family Impact', fontsize=13, fontweight='bold')
ax1.set_ylabel('Cumulative Avg PnL')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.5)

# Bottom: June only
for label, curve in curves.items():
    if curve.empty:
        continue
    june_curve = curve[curve.index >= june1]
    if june_curve.empty:
        continue
    june_offset = june_curve - june_curve.iloc[0]
    ax2.plot(june_offset.index, june_offset.values, label=label,
             color=colors[label], linestyle=styles[label], linewidth=2)

ax2.set_title('June 2026 Only — Cumulative PnL (rebased to 0)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Avg PnL (June)')
ax2.set_xlabel('Date')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('/tmp/composite_200_impact.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Chart saved: /tmp/composite_200_impact.png")

print()
print("=" * W)
print(f"  Analysis complete. {len(all_sweep)} composite methods + 200-impact evaluated.")
print(f"  Scores from: {RESULTS_DIR} + {RESULTS_FALLBACK}")
print("=" * W)
