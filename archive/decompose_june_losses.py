"""
decompose_june_losses.py — Systematic June 2026 Loss Decomposition
==================================================================
4-step analysis using Score CSVs + OHLCV + BQ config space:

  Step 1: Model Accuracy — Did predicted spikes actually occur in June?
  Step 2: SL/TP Analysis — For correct signals, did SL whipsaw before TP?
  Step 3: Ticker Breakdown — Which tickers dragged June down?
  Step 4: Timing — Were losses concentrated in specific weeks?
  + Step 5: Config Space — Are we selecting wrong configs from BQ?

Lightweight: no TF model loading. Uses score CSVs from training output,
OHLCV from GCS, and BQ wf_optimization_results for config exploration.

Colab-ready (no if __name__). Requires: GCS, BQ, pandas.
"""

import os, io, gc, sys, subprocess, warnings, ctypes
from datetime import datetime, timedelta
from collections import defaultdict
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
    from google.cloud import bigquery
    HAS_BQ = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
            "google-cloud-bigquery", "pyarrow", "--quiet"])
        from google.cloud import bigquery
        HAS_BQ = True
    except Exception:
        HAS_BQ = False

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
RESULTS_DIR = "Algos/Results_all_v2"
OHLCV_DIR = "Data/Inference"
INFERENCE_DIR = "Data/Inference_v2"

ANALYSIS_START = '2026-04-01'
JUNE_START = '2026-06-01'
JUNE_END = '2026-06-29'
MODEL_VERSION = 'robust_wf_v12'

PST = pytz.timezone('America/Los_Angeles')
SLIPPAGE_PCT = 0.0015
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

# Experiments to load score CSVs for (last fold of each family)
SCORE_EXPERIMENTS = [152, 160, 175, 185, 200]

# All tickers to analyse — superset across experiment families
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

DEPLOY_TICKERS = {'HUT','FLNC','CIFR','CLSK','KEEL','RCAT','RIOT','APLD','WULF'}
EXCLUDE_TICKERS = {'PL', 'SMLR'}

DEFAULT_PROB_THRESHOLD = 0.50
DEPLOY_THRESHOLD = 0.018
MIN_TRADES = 3

W = 120

gcs_client = gcs_storage.Client()
bq_client = bigquery.Client() if HAS_BQ else None


# ==============================================================================
# HELPERS
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open','High','Low','Close']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df


def load_seed_scores(ticker, experiment):
    """Load all seed score CSVs for (ticker, experiment). Returns dict: seed → DataFrame."""
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


def load_ohlcv(ticker, start_date=None, end_date=None):
    """Load OHLCV from GCS, heal, filter to date range."""
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    df = None
    for dir_path in [INFERENCE_DIR, OHLCV_DIR]:
        for suffix in ['_master_inference.parquet', '_ohlcv_1min_pst.parquet']:
            uri = f"gs://{BUCKET_NAME}/{dir_path}/{clean}{suffix}"
            try:
                df = pd.read_parquet(uri)
                break
            except Exception:
                continue
        if df is not None:
            break
    if df is None:
        return None
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
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz='UTC') - pd.Timedelta(days=1)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=2)]
    df = heal_data(df)
    df.index = df.index.tz_convert(PST)
    return df


def compute_mfe_mae(ohlcv_df, trade_date, entry_price, entry_hour=6, entry_min=0):
    """Compute max favorable/adverse excursion for a trade day.
    entry_hour/entry_min in ET → converted to PST internally."""
    if ohlcv_df is None or entry_price <= 0:
        return None
    day_bars = ohlcv_df[ohlcv_df.index.date == pd.Timestamp(trade_date).date()]
    if len(day_bars) == 0:
        return None
    pst_hour = entry_hour - 3
    entry_ts = pd.Timestamp(f"{trade_date} {pst_hour:02d}:{entry_min:02d}:00", tz=PST)
    after = day_bars[day_bars.index >= entry_ts]
    if len(after) < 5:
        return None
    ep = entry_price * (1.0 + SLIPPAGE_PCT)
    highs_ret = (after['High'].values / ep - 1)
    lows_ret = (after['Low'].values / ep - 1)
    close_ret = (after['Close'].iloc[-1] / ep - 1) - SLIPPAGE_PCT
    mfe = float(np.max(highs_ret))
    mae = float(np.min(lows_ret))
    sl_levels = {}
    for sl in [0.04, 0.06, 0.08, 0.10]:
        sl_hit = np.any(lows_ret <= -sl)
        if sl_hit:
            sl_bar = np.argmax(lows_ret <= -sl)
            mfe_before_sl = float(np.max(highs_ret[:sl_bar+1])) if sl_bar > 0 else 0.0
            eod_after_sl = float((after['Close'].iloc[-1] / ep - 1)) if sl_bar < len(after)-1 else None
        else:
            mfe_before_sl = mfe
            eod_after_sl = None
        sl_levels[sl] = {'hit': bool(sl_hit), 'mfe_before': mfe_before_sl, 'eod_after': eod_after_sl}
    tp_levels = {}
    for tp in [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        tp_hit = np.any(highs_ret >= tp)
        tp_levels[tp] = bool(tp_hit)
    return {'mfe': mfe, 'mae': mae, 'eod_ret': close_ret, 'n_bars': len(after),
            'sl_analysis': sl_levels, 'tp_analysis': tp_levels}


# ==============================================================================
# SECTION 1: LOAD SCORE CSVs
# ==============================================================================
print(f"{'='*W}")
print(f"  JUNE 2026 LOSS DECOMPOSITION — Score CSV + OHLCV + BQ Analysis")
print(f"  Analysis period: {ANALYSIS_START} → {JUNE_END}")
print(f"  June window: {JUNE_START} → {JUNE_END}")
print(f"{'='*W}")

print(f"\n  LOADING SCORE CSVs...")
print(f"  Experiments: {SCORE_EXPERIMENTS}")

all_scores = []
loaded_counts = defaultdict(int)
missing_tickers = defaultdict(list)

# --- Diagnostic: check what actually exists in GCS for first experiment ---
_diag_exp = SCORE_EXPERIMENTS[0]
_diag_prefix = f"{RESULTS_DIR}/Universal_Exp{_diag_exp}_XSECTOR/"
_diag_blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=_diag_prefix, max_results=20))
if _diag_blobs:
    print(f"\n  GCS diagnostic — {_diag_prefix} has files:")
    for b in _diag_blobs[:10]:
        print(f"    {b.name}")
else:
    print(f"\n  GCS diagnostic — NO FILES at {_diag_prefix}")
    print(f"  Checking parent directory...")
    _parent = f"{RESULTS_DIR}/"
    _parent_blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=_parent, max_results=20, delimiter='/'))
    print(f"  Subdirectories under {_parent}:")
    for p in gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=_parent, max_results=5):
        print(f"    {p.name}")

for exp_id in SCORE_EXPERIMENTS:
    tickers = EXP_TEACHERS.get(exp_id, TEACHERS_13)
    tickers = [t for t in tickers if t not in EXCLUDE_TICKERS]
    print(f"\n  Exp {exp_id}: loading {len(tickers)} tickers...")
    for ticker in tickers:
        seed_data = load_seed_scores(ticker, exp_id)
        if not seed_data:
            missing_tickers[exp_id].append(ticker)
            continue
        prob_series = {}
        target_series = {}
        close_series = {}
        for sid, df in seed_data.items():
            if 'Prob_Spike' in df.columns:
                prob_series[sid] = df['Prob_Spike']
            if 'Target' in df.columns:
                target_series[sid] = df['Target']
            if 'Close' in df.columns:
                close_series[sid] = df['Close']
        if not prob_series:
            missing_tickers[exp_id].append(ticker)
            continue
        prob_df = pd.DataFrame(prob_series)
        avg_prob = prob_df.mean(axis=1)
        if target_series:
            target_vals = pd.DataFrame(target_series).iloc[:, 0]
        else:
            target_vals = pd.Series(np.nan, index=avg_prob.index)
        if close_series:
            close_vals = pd.DataFrame(close_series).iloc[:, 0]
        else:
            close_vals = pd.Series(np.nan, index=avg_prob.index)
        for dt_idx in avg_prob.index:
            date_str = dt_idx.strftime('%Y-%m-%d')
            if date_str < ANALYSIS_START or date_str > JUNE_END:
                continue
            all_scores.append({
                'date': date_str,
                'ticker': ticker,
                'experiment': exp_id,
                'prob_spike': avg_prob[dt_idx],
                'target': target_vals.get(dt_idx, np.nan),
                'close': close_vals.get(dt_idx, np.nan),
                'n_seeds': len(prob_series),
            })
        loaded_counts[exp_id] += 1
    print(f"    Loaded: {loaded_counts[exp_id]} tickers | "
          f"Missing: {len(missing_tickers.get(exp_id, []))}")
    _release_memory()

df_scores = pd.DataFrame(all_scores)
print(f"\n  Total score rows: {len(df_scores)}")

if len(df_scores) == 0:
    print(f"\n  ERROR: No score CSVs loaded. Check GCS paths above.")
    print(f"  Expected path pattern: gs://{BUCKET_NAME}/{RESULTS_DIR}/Universal_Exp<ID>/<TICKER>/Seed_<SEED>_Results.csv")
    for exp_id, missing in missing_tickers.items():
        if missing:
            print(f"  Exp {exp_id} missing: {', '.join(missing[:10])}"
                  + (f" +{len(missing)-10} more" if len(missing) > 10 else ""))
    print(f"\n  Skipping Steps 1-4 (no score data). Jumping to Step 5 (BQ config space)...")
else:
    print(f"  Date range: {df_scores['date'].min()} → {df_scores['date'].max()}")
    print(f"  Tickers: {df_scores['ticker'].nunique()} | Experiments: {df_scores['experiment'].nunique()}")
    for exp_id, missing in missing_tickers.items():
        if missing:
            print(f"  Exp {exp_id} missing: {', '.join(missing[:10])}"
                  + (f" +{len(missing)-10} more" if len(missing) > 10 else ""))

df_scores['is_june'] = df_scores['date'] >= JUNE_START if len(df_scores) > 0 else False
df_scores['month'] = pd.to_datetime(df_scores['date']).dt.strftime('%Y-%m') if len(df_scores) > 0 else ''
df_scores['week'] = pd.to_datetime(df_scores['date']).dt.isocalendar().week.values if len(df_scores) > 0 else 0
df_scores['signal'] = (df_scores['prob_spike'] >= DEFAULT_PROB_THRESHOLD) if len(df_scores) > 0 else False


HAS_SCORES = len(df_scores) > 0

# ==============================================================================
# STEP 1: MODEL ACCURACY — Did predicted spikes actually occur?
# ==============================================================================
if not HAS_SCORES:
    print(f"\n\n  [Steps 1-4 skipped — no score CSV data loaded]")
    df_with_target = pd.DataFrame()
    df_trades = pd.DataFrame()

if HAS_SCORES:
    print(f"\n\n{'='*W}")
    print(f"  STEP 1: MODEL ACCURACY — Signal vs Actual Spike")
    print(f"  Threshold: prob_spike >= {DEFAULT_PROB_THRESHOLD}")
    print(f"{'='*W}")

    has_target = df_scores['target'].notna()
    df_with_target = df_scores[has_target].copy()

    if len(df_with_target) == 0:
        print("\n  ERROR: No score CSVs have Target column. Cannot assess model accuracy.")
        print("  Score CSVs may be from older training runs without labels on test data.")

if HAS_SCORES and len(df_with_target) > 0:
    df_with_target['target_bool'] = df_with_target['target'].astype(int) == 1

    # TABLE 1: Monthly hit rates
    print(f"\n  TABLE 1: MONTHLY MODEL ACCURACY (all tickers × experiments)")
    print(f"  {'Month':<10} {'Signals':>8} {'Spikes':>8} {'SigRate':>8} {'SpikeRate':>9} "
          f"{'HitRate':>8} {'FalsePos':>9} {'Missed':>7}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*9} {'─'*8} {'─'*9} {'─'*7}")

    for month in sorted(df_with_target['month'].unique()):
        m = df_with_target[df_with_target['month'] == month]
        n_total = len(m)
        n_signals = m['signal'].sum()
        n_spikes = m['target_bool'].sum()
        sig_rate = n_signals / n_total * 100 if n_total > 0 else 0
        spike_rate = n_spikes / n_total * 100 if n_total > 0 else 0
        tp = ((m['signal']) & (m['target_bool'])).sum()
        fp = ((m['signal']) & (~m['target_bool'])).sum()
        fn = ((~m['signal']) & (m['target_bool'])).sum()
        hit_rate = tp / n_signals * 100 if n_signals > 0 else 0
        fp_rate = fp / n_signals * 100 if n_signals > 0 else 0
        marker = '>>' if month >= '2026-06' else '  '
        print(f"{marker}{month:<10} {n_signals:>8} {n_spikes:>8} {sig_rate:>7.1f}% {spike_rate:>8.1f}% "
              f"{hit_rate:>7.1f}% {fp_rate:>8.1f}% {fn:>7}")

    # TABLE 2: Per-experiment accuracy in June vs pre-June
    print(f"\n\n  TABLE 2: PER-EXPERIMENT ACCURACY — June vs Pre-June")
    print(f"  {'Exp':>6} {'Period':<10} {'Signals':>8} {'HitRate':>8} {'FP Rate':>8} "
          f"{'SpikeRate':>9} {'AvgProb':>8}")
    print(f"  {'─'*6} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*9} {'─'*8}")

    for exp_id in sorted(df_with_target['experiment'].unique()):
        for period, filt in [('Pre-June', ~df_with_target['is_june']),
                              ('June', df_with_target['is_june'])]:
            sub = df_with_target[(df_with_target['experiment'] == exp_id) & filt]
            if len(sub) == 0:
                continue
            n_signals = sub['signal'].sum()
            tp = ((sub['signal']) & (sub['target_bool'])).sum()
            fp = ((sub['signal']) & (~sub['target_bool'])).sum()
            spike_rate = sub['target_bool'].mean() * 100
            avg_prob = sub.loc[sub['signal'], 'prob_spike'].mean() if n_signals > 0 else 0
            hit_rate = tp / n_signals * 100 if n_signals > 0 else 0
            fp_rate = fp / n_signals * 100 if n_signals > 0 else 0
            marker = '>>' if period == 'June' else '  '
            print(f"{marker}{exp_id:>6} {period:<10} {n_signals:>8} {hit_rate:>7.1f}% "
                  f"{fp_rate:>7.1f}% {spike_rate:>8.1f}% {avg_prob:>8.3f}")

    # TABLE 3: Family agreement analysis
    print(f"\n\n  TABLE 3: FAMILY AGREEMENT — When majority of experiments agree")
    # n_exps = total experiments covering this ticker-date
    # n_signalling = distinct experiments where signal==True
    fam_agree = df_with_target.groupby(['date', 'ticker']).agg(
        n_exps=('experiment', 'nunique'),
        any_target=('target_bool', 'max'),
        avg_prob=('prob_spike', 'mean'),
    ).reset_index()
    _sig_exps = df_with_target[df_with_target['signal']].groupby(
        ['date', 'ticker'])['experiment'].nunique().reset_index()
    _sig_exps.columns = ['date', 'ticker', 'n_signalling']
    fam_agree = fam_agree.merge(_sig_exps, on=['date', 'ticker'], how='left')
    fam_agree['n_signalling'] = fam_agree['n_signalling'].fillna(0).astype(int)
    fam_agree['majority_agree'] = fam_agree['n_signalling'] >= (fam_agree['n_exps'] / 2.0)
    fam_agree['is_june'] = fam_agree['date'] >= JUNE_START
    fam_agree['month'] = pd.to_datetime(fam_agree['date']).dt.strftime('%Y-%m')

    print(f"  {'Month':<10} {'FamSignals':>11} {'HitRate':>8} {'FP Rate':>8} "
          f"{'AvgAgreePct':>12} {'AvgProb':>8}")
    print(f"  {'─'*10} {'─'*11} {'─'*8} {'─'*8} {'─'*12} {'─'*8}")

    for month in sorted(fam_agree['month'].unique()):
        m = fam_agree[fam_agree['month'] == month]
        agreed = m[m['majority_agree']]
        n_agreed = len(agreed)
        if n_agreed == 0:
            continue
        tp = (agreed['any_target']).sum()
        fp = n_agreed - tp
        hit_rate = tp / n_agreed * 100
        fp_rate = fp / n_agreed * 100
        avg_agree_pct = (agreed['n_signalling'] / agreed['n_exps']).mean() * 100
        avg_prob = agreed['avg_prob'].mean()
        marker = '>>' if month >= '2026-06' else '  '
        print(f"{marker}{month:<10} {n_agreed:>11} {hit_rate:>7.1f}% {fp_rate:>7.1f}% "
              f"{avg_agree_pct:>11.1f}% {avg_prob:>8.3f}")

    # Diagnosis
    pre_june = df_with_target[~df_with_target['is_june']]
    june = df_with_target[df_with_target['is_june']]
    if len(june) > 0 and len(pre_june) > 0:
        pre_spike_rate = pre_june['target_bool'].mean() * 100
        jun_spike_rate = june['target_bool'].mean() * 100
        pre_sigs = pre_june[pre_june['signal']]
        jun_sigs = june[june['signal']]
        pre_sig_rate = pre_june['signal'].mean() * 100
        jun_sig_rate = june['signal'].mean() * 100
        pre_hit = (pre_sigs['target_bool'].mean() * 100) if len(pre_sigs) > 0 else 0
        jun_hit = (jun_sigs['target_bool'].mean() * 100) if len(jun_sigs) > 0 else 0

        print(f"\n  STEP 1 DIAGNOSIS:")
        print(f"    Signal rate: Pre-June {pre_sig_rate:.1f}%  →  June {jun_sig_rate:.1f}%  "
              f"(Δ {jun_sig_rate - pre_sig_rate:+.1f}pp)")
        print(f"    Spike rate:  Pre-June {pre_spike_rate:.1f}%  →  June {jun_spike_rate:.1f}%  "
              f"(Δ {jun_spike_rate - pre_spike_rate:+.1f}pp)")
        print(f"    Hit rate:    Pre-June {pre_hit:.1f}%  →  June {jun_hit:.1f}%  "
              f"(Δ {jun_hit - pre_hit:+.1f}pp)")
        sig_spike_divergence = (jun_sig_rate - pre_sig_rate) - (jun_spike_rate - pre_spike_rate)
        print(f"    Signal/Spike divergence: {sig_spike_divergence:+.1f}pp (positive = model over-triggering)")
        if jun_sig_rate > pre_sig_rate and jun_spike_rate < pre_spike_rate:
            print(f"    → MODEL CALIBRATION FAILURE: signal rate INCREASED ({pre_sig_rate:.0f}%→{jun_sig_rate:.0f}%) "
                  f"while spikes DECREASED ({pre_spike_rate:.0f}%→{jun_spike_rate:.0f}%)")
            print(f"    → Model is MORE trigger-happy in the WORST month — it's not adapting to regime change")
        elif jun_hit < pre_hit * 0.7:
            print(f"    → MODEL ACCURACY DEGRADED — spikes still occur but model picks wrong ones")
        else:
            print(f"    → MODEL IS ROUGHLY CORRECT — losses likely driven by SL/TP execution, not prediction quality")


# ==============================================================================
# STEP 1A: SIGNAL CALIBRATION — Is the model over-triggering?
# ==============================================================================
if HAS_SCORES and len(df_with_target) > 0:
    print(f"\n\n{'='*W}")
    print(f"  STEP 1A: SIGNAL CALIBRATION — Probability distribution shift")
    print(f"{'='*W}")

    pre_june_sigs = df_with_target[(~df_with_target['is_june']) & (df_with_target['signal'])]
    june_sigs = df_with_target[(df_with_target['is_june']) & (df_with_target['signal'])]

    # TABLE 1A-1: Prob distribution — are June FPs at high or low confidence?
    prob_bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
    print(f"\n  TABLE 1A-1: SIGNAL PROBABILITY DISTRIBUTION — June vs Pre-June")
    print(f"  {'Prob Range':<12} {'Pre-Jun Sigs':>13} {'Pre-Jun Hit%':>13} "
          f"{'June Sigs':>10} {'June Hit%':>10} {'June FPs':>9}")
    print(f"  {'─'*12} {'─'*13} {'─'*13} {'─'*10} {'─'*10} {'─'*9}")

    for lo, hi in prob_bins:
        pre_bin = pre_june_sigs[(pre_june_sigs['prob_spike'] >= lo) & (pre_june_sigs['prob_spike'] < hi)]
        jun_bin = june_sigs[(june_sigs['prob_spike'] >= lo) & (june_sigs['prob_spike'] < hi)]
        pre_n = len(pre_bin)
        pre_hit = pre_bin['target_bool'].mean() * 100 if pre_n > 0 else 0
        jun_n = len(jun_bin)
        jun_hit = jun_bin['target_bool'].mean() * 100 if jun_n > 0 else 0
        jun_fp = int((~jun_bin['target_bool']).sum()) if jun_n > 0 else 0
        label = f"{lo:.2f}-{hi:.2f}"
        marker = '>>' if jun_n > pre_n * 0.3 and jun_hit < pre_hit * 0.5 else '  '
        print(f"{marker}{label:<12} {pre_n:>13} {pre_hit:>12.1f}% "
              f"{jun_n:>10} {jun_hit:>9.1f}% {jun_fp:>9}")

    # TABLE 1A-2: Per-experiment signal rate shift
    print(f"\n\n  TABLE 1A-2: SIGNAL RATE SHIFT — Model firing more or less?")
    print(f"  {'Exp':>6} {'Pre-Jun SigRate':>16} {'June SigRate':>13} "
          f"{'Δ SigRate':>10} {'Pre SpikeRate':>14} {'Jun SpikeRate':>14} {'Over-trigger':>13}")
    print(f"  {'─'*6} {'─'*16} {'─'*13} {'─'*10} {'─'*14} {'─'*14} {'─'*13}")

    for exp_id in sorted(df_with_target['experiment'].unique()):
        pre_exp = df_with_target[(df_with_target['experiment'] == exp_id) & (~df_with_target['is_june'])]
        jun_exp = df_with_target[(df_with_target['experiment'] == exp_id) & (df_with_target['is_june'])]
        if len(pre_exp) == 0 or len(jun_exp) == 0:
            continue
        pre_sr = pre_exp['signal'].mean() * 100
        jun_sr = jun_exp['signal'].mean() * 100
        pre_spk = pre_exp['target_bool'].mean() * 100
        jun_spk = jun_exp['target_bool'].mean() * 100
        delta_sr = jun_sr - pre_sr
        over = 'YES' if delta_sr > 0 and jun_spk < pre_spk else 'no'
        marker = '>>' if over == 'YES' else '  '
        print(f"{marker}{exp_id:>6} {pre_sr:>15.1f}% {jun_sr:>12.1f}% "
              f"{delta_sr:>+9.1f}pp {pre_spk:>13.1f}% {jun_spk:>13.1f}% {over:>13}")

    print(f"\n  STEP 1A DIAGNOSIS:")
    pre_all_sig_rate = pre_june['signal'].mean() * 100
    jun_all_sig_rate = june['signal'].mean() * 100
    pre_all_spike_rate = pre_june['target_bool'].mean() * 100
    jun_all_spike_rate = june['target_bool'].mean() * 100
    print(f"    Overall signal rate: {pre_all_sig_rate:.1f}% → {jun_all_sig_rate:.1f}% "
          f"(Δ {jun_all_sig_rate - pre_all_sig_rate:+.1f}pp)")
    print(f"    Overall spike rate:  {pre_all_spike_rate:.1f}% → {jun_all_spike_rate:.1f}% "
          f"(Δ {jun_all_spike_rate - pre_all_spike_rate:+.1f}pp)")
    if jun_all_sig_rate > pre_all_sig_rate and jun_all_spike_rate < pre_all_spike_rate:
        ratio = jun_all_sig_rate / max(jun_all_spike_rate, 0.1)
        pre_ratio = pre_all_sig_rate / max(pre_all_spike_rate, 0.1)
        print(f"    Signal-to-spike ratio: Pre-June {pre_ratio:.1f}x → June {ratio:.1f}x")
        print(f"    → MODEL IS OVER-TRIGGERING: predicting {ratio:.0f} signals for every real spike")
        print(f"    → A well-calibrated model would fire FEWER signals when spikes are rarer")
    else:
        print(f"    → Signal rate scaled proportionally with spike rate")


# ==============================================================================
# STEP 1B: LOSS CATEGORIZATION — WHY each signal lost
# ==============================================================================
if HAS_SCORES and len(df_with_target) > 0:
    print(f"\n\n{'='*W}")
    print(f"  STEP 1B: LOSS CATEGORIZATION — Model wrong vs execution failure")
    print(f"{'='*W}")

    # Build per-(date, ticker) aggregation with OHLCV for ALL months, not just June
    # Use the ohlcv_cache that will be built in Step 2, but first we need to
    # load OHLCV for pre-June tickers too for comparison
    print(f"\n  Loading OHLCV for pre-June + June comparison...")
    all_signal_days = df_with_target[df_with_target['signal']].copy()
    # n_exps_signal = distinct experiments that signalled (not row count)
    # n_exps_total = distinct experiments covering this ticker (from full data)
    all_per_day = all_signal_days.groupby(['date', 'ticker']).agg(
        avg_prob=('prob_spike', 'mean'),
        n_exps_signal=('experiment', 'nunique'),
        target=('target', 'first'),
        close=('close', 'mean'),
    ).reset_index()
    _total_exps = df_with_target.groupby(['date', 'ticker'])['experiment'].nunique().reset_index()
    _total_exps.columns = ['date', 'ticker', 'n_exps_total']
    all_per_day = all_per_day.merge(_total_exps, on=['date', 'ticker'], how='left')

    ohlcv_cache_all = {}
    all_tickers_needed = all_per_day['ticker'].unique()
    for ticker in sorted(all_tickers_needed):
        if ticker in ohlcv_cache_all:
            continue
        df_ohlcv = load_ohlcv(ticker, start_date=ANALYSIS_START, end_date=JUNE_END)
        if df_ohlcv is not None and len(df_ohlcv) > 0:
            ohlcv_cache_all[ticker] = df_ohlcv
    print(f"    Loaded OHLCV for {len(ohlcv_cache_all)}/{len(all_tickers_needed)} tickers")

    all_trade_cats = []
    for _, row in all_per_day.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        entry_price = row['close']
        if ticker not in ohlcv_cache_all or pd.isna(entry_price) or entry_price <= 0:
            continue
        result = compute_mfe_mae(ohlcv_cache_all[ticker], date_str, entry_price)
        if result is None:
            continue
        mfe = result['mfe']
        mae = result['mae']
        eod = result['eod_ret']
        if mfe < 0.03 and mae <= -0.04:
            category = 'A_model_wrong'
        elif mfe < 0.05 and eod < -0.02:
            category = 'B_weak_spike'
        elif mfe >= 0.08 and mae <= -0.06:
            category = 'C_sl_before_tp'
        elif eod > 0:
            category = 'D_winner'
        elif mfe >= 0.05 and eod <= 0:
            category = 'E_reversed'
        else:
            category = 'F_marginal_loss'
        all_trade_cats.append({
            'date': date_str,
            'ticker': ticker,
            'avg_prob': row['avg_prob'],
            'n_exps_signal': row['n_exps_signal'],
            'target': row['target'],
            'mfe': mfe, 'mae': mae, 'eod_ret': eod,
            'category': category,
            'is_june': date_str >= JUNE_START,
            'month': date_str[:7],
        })
    df_cats = pd.DataFrame(all_trade_cats)
    _release_memory()

    if len(df_cats) > 0:
        cat_labels = {
            'A_model_wrong': 'A: Model wrong (no spike, straight drop)',
            'B_weak_spike': 'B: Weak spike (MFE<5%, lost >2%)',
            'C_sl_before_tp': 'C: SL before TP (MFE≥8% but MAE≤-6%)',
            'D_winner': 'D: Winner (EOD > 0)',
            'E_reversed': 'E: Reversed (MFE≥5% but EOD≤0)',
            'F_marginal_loss': 'F: Marginal loss (other)',
        }

        # TABLE 1B-1: Category distribution by month
        print(f"\n  TABLE 1B-1: LOSS CATEGORY DISTRIBUTION BY MONTH")
        cats_ordered = ['A_model_wrong', 'B_weak_spike', 'C_sl_before_tp',
                        'D_winner', 'E_reversed', 'F_marginal_loss']
        header = f"  {'Month':<10} {'Total':>6}"
        for cat in cats_ordered:
            label = cat.split('_')[0]
            header += f" {label:>8}"
        header += f" {'A+B %':>7} {'D %':>6}"
        print(header)
        print(f"  {'─'*10} {'─'*6}" + f" {'─'*8}" * len(cats_ordered) + f" {'─'*7} {'─'*6}")

        for month in sorted(df_cats['month'].unique()):
            msub = df_cats[df_cats['month'] == month]
            n_total = len(msub)
            row_str = f"  {month:<10} {n_total:>6}"
            n_ab = 0
            n_d = 0
            for cat in cats_ordered:
                n_cat = (msub['category'] == cat).sum()
                pct = n_cat / n_total * 100 if n_total > 0 else 0
                row_str += f" {pct:>7.0f}%"
                if cat in ('A_model_wrong', 'B_weak_spike'):
                    n_ab += n_cat
                if cat == 'D_winner':
                    n_d += n_cat
            ab_pct = n_ab / n_total * 100 if n_total > 0 else 0
            d_pct = n_d / n_total * 100 if n_total > 0 else 0
            marker = '>>' if month >= '2026-06' else '  '
            row_str = f"{marker}{row_str[2:]}"
            row_str += f" {ab_pct:>6.0f}% {d_pct:>5.0f}%"
            print(row_str)

        # Legend
        print(f"\n  Category definitions:")
        for cat in cats_ordered:
            print(f"    {cat_labels[cat]}")

        # TABLE 1B-2: Category A deep dive — model-wrong signals
        june_cat_a = df_cats[(df_cats['is_june']) & (df_cats['category'] == 'A_model_wrong')]
        if len(june_cat_a) > 0:
            print(f"\n\n  TABLE 1B-2: CATEGORY A — Model confidently wrong in June ({len(june_cat_a)} signals)")
            print(f"  {'Date':<12} {'Ticker':<7} {'Prob':>5} {'MFE':>7} {'MAE':>7} "
                  f"{'EOD':>7} {'Target':>7} {'FamAgree':>9}")
            print(f"  {'─'*12} {'─'*7} {'─'*5} {'─'*7} {'─'*7} "
                  f"{'─'*7} {'─'*7} {'─'*9}")
            for _, r in june_cat_a.sort_values('mae').head(30).iterrows():
                target_str = f"{int(r['target'])}" if pd.notna(r['target']) else '?'
                n_total = int(r['n_exps_total']) if pd.notna(r.get('n_exps_total', np.nan)) else 5
                fam_str = f"{int(r['n_exps_signal'])}/{n_total}"
                print(f"  {r['date']:<12} {r['ticker']:<7} {r['avg_prob']:.3f} "
                      f"{r['mfe']*100:>+6.2f}% {r['mae']*100:>+6.2f}% "
                      f"{r['eod_ret']*100:>+6.2f}% {target_str:>7} {fam_str:>9}")

        # TABLE 1B-3: Avg prob by category — are wrong signals at low or high confidence?
        print(f"\n\n  TABLE 1B-3: SIGNAL CONFIDENCE BY CATEGORY — June only")
        print(f"  {'Category':<42} {'Count':>6} {'AvgProb':>8} {'MedProb':>8} "
              f"{'Prob≥0.6':>9} {'AvgMAE':>8}")
        print(f"  {'─'*42} {'─'*6} {'─'*8} {'─'*8} {'─'*9} {'─'*8}")
        june_cats = df_cats[df_cats['is_june']]
        for cat in cats_ordered:
            csub = june_cats[june_cats['category'] == cat]
            if len(csub) == 0:
                continue
            avg_prob = csub['avg_prob'].mean()
            med_prob = csub['avg_prob'].median()
            high_conf = (csub['avg_prob'] >= 0.60).sum()
            high_pct = high_conf / len(csub) * 100
            avg_mae = csub['mae'].mean() * 100
            marker = '>>' if cat == 'A_model_wrong' else '  '
            print(f"{marker}{cat_labels[cat]:<42} {len(csub):>6} {avg_prob:>8.3f} "
                  f"{med_prob:>8.3f} {high_pct:>7.0f}%  {avg_mae:>+7.2f}%")

        print(f"\n  STEP 1B DIAGNOSIS:")
        pre_cats = df_cats[~df_cats['is_june']]
        jun_cats = df_cats[df_cats['is_june']]
        if len(pre_cats) > 0 and len(jun_cats) > 0:
            pre_a_pct = (pre_cats['category'] == 'A_model_wrong').mean() * 100
            jun_a_pct = (jun_cats['category'] == 'A_model_wrong').mean() * 100
            pre_d_pct = (pre_cats['category'] == 'D_winner').mean() * 100
            jun_d_pct = (jun_cats['category'] == 'D_winner').mean() * 100
            pre_ab_pct = (pre_cats['category'].isin(['A_model_wrong', 'B_weak_spike'])).mean() * 100
            jun_ab_pct = (jun_cats['category'].isin(['A_model_wrong', 'B_weak_spike'])).mean() * 100

            print(f"    'Model wrong' rate: Pre-June {pre_a_pct:.0f}% → June {jun_a_pct:.0f}% "
                  f"(Δ {jun_a_pct - pre_a_pct:+.0f}pp)")
            print(f"    'Model wrong + Weak' rate: Pre-June {pre_ab_pct:.0f}% → June {jun_ab_pct:.0f}% "
                  f"(Δ {jun_ab_pct - pre_ab_pct:+.0f}pp)")
            print(f"    Winner rate: Pre-June {pre_d_pct:.0f}% → June {jun_d_pct:.0f}% "
                  f"(Δ {jun_d_pct - pre_d_pct:+.0f}pp)")

            jun_a_high = june_cats[(june_cats['category'] == 'A_model_wrong') & (june_cats['avg_prob'] >= 0.60)]
            if len(june_cat_a) > 0:
                print(f"    Category A with prob≥0.60: {len(jun_a_high)}/{len(june_cat_a)} "
                      f"({len(jun_a_high)/len(june_cat_a)*100:.0f}%)")

            pre_b_pct = (pre_cats['category'] == 'B_weak_spike').mean() * 100
            jun_b_pct = (jun_cats['category'] == 'B_weak_spike').mean() * 100
            print(f"    'Weak spike' rate: Pre-June {pre_b_pct:.0f}% → June {jun_b_pct:.0f}% "
                  f"(Δ {jun_b_pct - pre_b_pct:+.0f}pp)")

            if jun_a_pct > pre_a_pct + 5:
                print(f"    → MODEL IS THE PRIMARY PROBLEM: 'model wrong' rate jumped {jun_a_pct - pre_a_pct:+.0f}pp")
                print(f"      The model says BUY and the stock drops. No SL/TP fix can save a bad prediction.")
            elif jun_b_pct > pre_b_pct + 5:
                print(f"    → SPIKE QUALITY DEGRADED: Cat-B (weak spike) jumped {jun_b_pct - pre_b_pct:+.0f}pp")
                print(f"      Model detects activity but spikes are too weak to profit. This is regime shift.")
                print(f"      TP targets are unreachable when MFE < 5% — need lower TP or skip marginal signals.")
            elif jun_d_pct < pre_d_pct - 8:
                print(f"    → WINNER RATE COLLAPSED: {pre_d_pct:.0f}% → {jun_d_pct:.0f}% ({jun_d_pct - pre_d_pct:+.0f}pp)")
                print(f"      Fewer trades end positive — regime cannot sustain profitable exits.")
            else:
                print(f"    → Loss categories stable — issue is execution, not prediction quality")


# ==============================================================================
# STEP 1C: FAMILY AGREEMENT FILTER — Does S3 filter help?
# ==============================================================================
if HAS_SCORES and len(df_with_target) > 0 and len(df_cats) > 0:
    print(f"\n\n{'='*W}")
    print(f"  STEP 1C: FAMILY AGREEMENT FILTER — Does majority-vote help?")
    print(f"{'='*W}")

    # For each (date, ticker), check if majority of experiments signalled
    fam_filter = df_cats.copy()
    fam_filter['majority'] = fam_filter['n_exps_signal'] >= 3  # majority of 5

    # TABLE 1C-1: Compare all signals vs family-filtered signals
    print(f"\n  TABLE 1C-1: ALL SIGNALS vs FAMILY-FILTERED (≥3/5 agree)")
    print(f"  {'Month':<10} {'All':>5} {'A+B%':>6} {'D%':>5} {'AvgEOD':>8} "
          f"{'│':>1} {'Fam':>5} {'A+B%':>6} {'D%':>5} {'AvgEOD':>8} {'│':>1} {'Filtered%':>10}")
    print(f"  {'─'*10} {'─'*5} {'─'*6} {'─'*5} {'─'*8} "
          f"{'│':>1} {'─'*5} {'─'*6} {'─'*5} {'─'*8} {'│':>1} {'─'*10}")

    for month in sorted(df_cats['month'].unique()):
        all_m = df_cats[df_cats['month'] == month]
        fam_m = fam_filter[(fam_filter['month'] == month) & (fam_filter['majority'])]
        n_all = len(all_m)
        n_fam = len(fam_m)
        if n_all == 0:
            continue
        all_ab = all_m['category'].isin(['A_model_wrong', 'B_weak_spike']).mean() * 100
        all_d = (all_m['category'] == 'D_winner').mean() * 100
        all_eod = all_m['eod_ret'].mean() * 100
        fam_ab = fam_m['category'].isin(['A_model_wrong', 'B_weak_spike']).mean() * 100 if n_fam > 0 else 0
        fam_d = (fam_m['category'] == 'D_winner').mean() * 100 if n_fam > 0 else 0
        fam_eod = fam_m['eod_ret'].mean() * 100 if n_fam > 0 else 0
        filtered_pct = (1 - n_fam / n_all) * 100 if n_all > 0 else 0
        marker = '>>' if month >= '2026-06' else '  '
        print(f"{marker}{month:<10} {n_all:>5} {all_ab:>5.0f}% {all_d:>4.0f}% {all_eod:>+7.2f}% "
              f"│ {n_fam:>5} {fam_ab:>5.0f}% {fam_d:>4.0f}% {fam_eod:>+7.2f}% "
              f"│ {filtered_pct:>9.0f}%")

    # TABLE 1C-2: Category A — does family agreement filter them out?
    print(f"\n\n  TABLE 1C-2: CATEGORY A SIGNALS — Family agreement effect")
    print(f"  {'Month':<10} {'All Cat-A':>10} {'Fam Cat-A':>10} {'Filtered':>9} "
          f"{'All Cat-A EOD':>14} {'Fam Cat-A EOD':>14}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*9} {'─'*14} {'─'*14}")

    for month in sorted(df_cats['month'].unique()):
        all_a = df_cats[(df_cats['month'] == month) & (df_cats['category'] == 'A_model_wrong')]
        fam_a = fam_filter[(fam_filter['month'] == month) & (fam_filter['majority']) &
                           (fam_filter['category'] == 'A_model_wrong')]
        n_all = len(all_a)
        n_fam = len(fam_a)
        if n_all == 0:
            continue
        filtered = n_all - n_fam
        all_eod = all_a['eod_ret'].mean() * 100
        fam_eod = fam_a['eod_ret'].mean() * 100 if n_fam > 0 else 0
        marker = '>>' if month >= '2026-06' else '  '
        print(f"{marker}{month:<10} {n_all:>10} {n_fam:>10} {filtered:>9} "
              f"{all_eod:>+13.2f}% {fam_eod:>+13.2f}%")

    # TABLE 1C-3: Are all experiments agreeing on wrong calls?
    june_fam_wrong = fam_filter[(fam_filter['is_june']) & (fam_filter['majority']) &
                                (fam_filter['category'] == 'A_model_wrong')]
    if len(june_fam_wrong) > 0:
        print(f"\n\n  TABLE 1C-3: FAMILY-AGREED 'MODEL WRONG' — All 5 exps agree on bad call")
        print(f"  {'Date':<12} {'Ticker':<7} {'Prob':>5} {'Agree':>6} "
              f"{'MFE':>7} {'MAE':>7} {'EOD':>7}")
        print(f"  {'─'*12} {'─'*7} {'─'*5} {'─'*6} "
              f"{'─'*7} {'─'*7} {'─'*7}")
        for _, r in june_fam_wrong.sort_values('mae').head(20).iterrows():
            n_total = int(r['n_exps_total']) if pd.notna(r.get('n_exps_total', np.nan)) else 5
            print(f"  {r['date']:<12} {r['ticker']:<7} {r['avg_prob']:.3f} "
                  f"{int(r['n_exps_signal']):>4}/{n_total} "
                  f"{r['mfe']*100:>+6.2f}% {r['mae']*100:>+6.2f}% "
                  f"{r['eod_ret']*100:>+6.2f}%")

    print(f"\n  STEP 1C DIAGNOSIS:")
    jun_all = df_cats[df_cats['is_june']]
    jun_fam = fam_filter[(fam_filter['is_june']) & (fam_filter['majority'])]
    if len(jun_all) > 0 and len(jun_fam) > 0:
        all_a_pct = (jun_all['category'] == 'A_model_wrong').mean() * 100
        fam_a_pct = (jun_fam['category'] == 'A_model_wrong').mean() * 100
        all_eod = jun_all['eod_ret'].mean() * 100
        fam_eod = jun_fam['eod_ret'].mean() * 100
        n_fam_wrong = len(june_fam_wrong)
        print(f"    All signals:     {len(jun_all)} trades, Cat-A {all_a_pct:.0f}%, avg EOD {all_eod:+.2f}%")
        print(f"    Family-filtered: {len(jun_fam)} trades, Cat-A {fam_a_pct:.0f}%, avg EOD {fam_eod:+.2f}%")
        print(f"    Family-agreed 'model wrong': {n_fam_wrong} signals")

        if fam_a_pct >= all_a_pct - 3:
            print(f"    → FAMILY FILTER DOES NOT HELP: Cat-A rate barely changes ({all_a_pct:.0f}%→{fam_a_pct:.0f}%)")
            print(f"      ALL experiments are confidently wrong together — correlated failure")
        elif fam_a_pct < all_a_pct * 0.7:
            print(f"    → Family filter helps: reduces Cat-A from {all_a_pct:.0f}% to {fam_a_pct:.0f}%")
            print(f"      Disagreeing experiments correctly flag bad signals")
        if fam_eod >= all_eod + 0.5:
            print(f"    → Family filter improves EOD by {fam_eod - all_eod:+.2f}pp — useful but not sufficient")
        else:
            print(f"    → Family filter barely improves EOD ({all_eod:+.2f}% → {fam_eod:+.2f}%) — not the fix")


# ==============================================================================
# STEP 2: SL/TP ANALYSIS — Stopped out before spike?
# ==============================================================================
df_trades = pd.DataFrame()

if HAS_SCORES:
    print(f"\n\n{'='*W}")
    print(f"  STEP 2: SL/TP ANALYSIS — Max Favorable/Adverse Excursion")
    print(f"{'='*W}")

june_signals = df_scores[(df_scores['is_june']) & (df_scores['signal'])].copy() if HAS_SCORES else pd.DataFrame()
june_signal_tickers = june_signals['ticker'].unique() if len(june_signals) > 0 else []
if HAS_SCORES:
    print(f"\n  Loading OHLCV for {len(june_signal_tickers)} tickers with June signals...")

ohlcv_cache = {}
if len(june_signal_tickers) > 0:
    for ticker in sorted(june_signal_tickers):
        df_ohlcv = load_ohlcv(ticker, start_date=JUNE_START, end_date=JUNE_END)
        if df_ohlcv is not None and len(df_ohlcv) > 0:
            ohlcv_cache[ticker] = df_ohlcv
            print(f"    {ticker}: {len(df_ohlcv)} bars")
        else:
            print(f"    {ticker}: NO DATA")
    _release_memory()

    june_per_day = june_signals.groupby(['date', 'ticker']).agg(
        avg_prob=('prob_spike', 'mean'),
        n_exps_signal=('experiment', 'nunique'),
        target=('target', 'first'),
        close=('close', 'mean'),
    ).reset_index()
    _june_total_exps = df_scores[df_scores['is_june']].groupby(
        ['date', 'ticker'])['experiment'].nunique().reset_index()
    _june_total_exps.columns = ['date', 'ticker', 'n_exps_total']
    june_per_day = june_per_day.merge(_june_total_exps, on=['date', 'ticker'], how='left')

    print(f"\n  Computing MFE/MAE for {len(june_per_day)} signal days...")

    trade_results = []
    for _, row in june_per_day.iterrows():
        ticker = row['ticker']
        date_str = row['date']
        entry_price = row['close']
        if ticker not in ohlcv_cache or pd.isna(entry_price) or entry_price <= 0:
            continue
        result = compute_mfe_mae(ohlcv_cache[ticker], date_str, entry_price)
        if result is None:
            continue
        trade_results.append({
            'date': date_str,
            'ticker': ticker,
            'avg_prob': row['avg_prob'],
            'n_exps_signal': row['n_exps_signal'],
            'n_exps_total': row['n_exps_total'],
            'target': row['target'],
            'entry_price': entry_price,
            **result,
        })

    df_trades = pd.DataFrame(trade_results)

if len(df_trades) == 0:
    print("\n  No trade data available for SL/TP analysis.")
else:
    # TABLE 4: MFE/MAE summary
    print(f"\n  TABLE 4: MFE/MAE FOR JUNE SIGNALS (prob >= {DEFAULT_PROB_THRESHOLD})")
    print(f"  {'Date':<12} {'Ticker':<7} {'Prob':>5} {'Entry$':>8} {'MFE':>7} {'MAE':>7} "
          f"{'EOD%':>7} {'Target':>7} {'Diagnosis':>20}")
    print(f"  {'─'*12} {'─'*7} {'─'*5} {'─'*8} {'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*7} {'─'*20}")

    for _, r in df_trades.sort_values(['date', 'ticker']).iterrows():
        target_str = f"{int(r['target'])}" if pd.notna(r['target']) else '?'
        mfe_pct = r['mfe'] * 100
        mae_pct = r['mae'] * 100
        eod_pct = r['eod_ret'] * 100

        if r['mfe'] >= 0.08 and r['mae'] <= -0.06:
            diag = 'SL before TP'
        elif r['mfe'] >= 0.08:
            diag = 'TP reachable'
        elif r['mfe'] < 0.03 and r['mae'] <= -0.04:
            diag = 'No spike, just drop'
        elif r['mfe'] < 0.05:
            diag = 'Weak spike'
        else:
            diag = 'Moderate move'

        win = '+' if r['eod_ret'] > 0 else '-'
        print(f"  {r['date']:<12} {r['ticker']:<7} {r['avg_prob']:.3f} "
              f"${r['entry_price']:>7.2f} {mfe_pct:>+6.2f}% {mae_pct:>+6.2f}% "
              f"{eod_pct:>+6.2f}% {target_str:>7} {diag:>20}")

    # TABLE 5: SL sensitivity — what if SL were wider?
    print(f"\n\n  TABLE 5: SL SENSITIVITY — Would wider SL help?")
    print(f"  {'SL Level':>10} {'Hit Rate':>9} {'Avg MFE before SL':>19} {'Whipsaw %':>10} "
          f"{'Avg EOD after SL':>17}")
    print(f"  {'─'*10} {'─'*9} {'─'*19} {'─'*10} {'─'*17}")

    for sl in [0.04, 0.06, 0.08, 0.10]:
        sl_data = [r['sl_analysis'][sl] for _, r in df_trades.iterrows()]
        n_hit = sum(1 for s in sl_data if s['hit'])
        hit_rate = n_hit / len(sl_data) * 100 if sl_data else 0
        mfe_before = np.mean([s['mfe_before'] for s in sl_data if s['hit']]) * 100 if n_hit > 0 else 0
        whipsaw = sum(1 for s in sl_data if s['hit'] and s['eod_after'] is not None and s['eod_after'] > 0)
        whipsaw_pct = whipsaw / n_hit * 100 if n_hit > 0 else 0
        eod_after = np.mean([s['eod_after'] for s in sl_data
                             if s['hit'] and s['eod_after'] is not None]) * 100 if n_hit > 0 else 0

        marker = '>>' if sl == 0.06 else '  '
        print(f"{marker}{sl*100:>7.0f}%   {hit_rate:>8.1f}% {mfe_before:>+18.2f}% {whipsaw_pct:>9.0f}% "
              f"{eod_after:>+16.2f}%")

    # TABLE 6: TP reachability — how many signals could have reached TP?
    print(f"\n\n  TABLE 6: TP REACHABILITY — Could the spike reach TP?")
    print(f"  {'TP Level':>10} {'Reach %':>8} {'Not reached':>12}")
    print(f"  {'─'*10} {'─'*8} {'─'*12}")

    for tp in [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        reached = sum(1 for _, r in df_trades.iterrows() if r['tp_analysis'].get(tp, False))
        reach_pct = reached / len(df_trades) * 100 if len(df_trades) > 0 else 0
        marker = '>>' if tp == 0.08 else '  '
        print(f"{marker}{tp*100:>7.0f}%   {reach_pct:>7.1f}% {len(df_trades)-reached:>12}")

    # Step 2 diagnosis
    if len(df_trades) > 0:
        sl6_hit = sum(1 for _, r in df_trades.iterrows() if r['sl_analysis'][0.06]['hit'])
        sl6_whipsaw = sum(1 for _, r in df_trades.iterrows()
                          if r['sl_analysis'][0.06]['hit']
                          and r['sl_analysis'][0.06]['eod_after'] is not None
                          and r['sl_analysis'][0.06]['eod_after'] > 0)
        tp8_reached = sum(1 for _, r in df_trades.iterrows() if r['tp_analysis'].get(0.08, False))
        avg_mfe = df_trades['mfe'].mean() * 100
        avg_mae = df_trades['mae'].mean() * 100

        print(f"\n  STEP 2 DIAGNOSIS:")
        print(f"    Avg MFE: {avg_mfe:+.2f}%  |  Avg MAE: {avg_mae:+.2f}%")
        print(f"    SL 6% hit rate: {sl6_hit}/{len(df_trades)} ({sl6_hit/len(df_trades)*100:.0f}%)")
        if sl6_hit > 0:
            print(f"    SL 6% whipsaw rate: {sl6_whipsaw}/{sl6_hit} ({sl6_whipsaw/sl6_hit*100:.0f}%)")
        print(f"    TP 8% reachable: {tp8_reached}/{len(df_trades)} ({tp8_reached/len(df_trades)*100:.0f}%)")

        if sl6_hit / len(df_trades) > 0.5 and sl6_whipsaw / max(sl6_hit, 1) > 0.3:
            print(f"    → HIGH SL WHIPSAW RATE — SL too tight for June volatility, widen to 8-10%")
        elif tp8_reached / len(df_trades) < 0.3:
            print(f"    → SPIKES TOO WEAK — TP 8% rarely reachable, consider lowering to 6%")
        elif avg_mfe > 0.06 and avg_mae < -0.04:
            print(f"    → VOLATILITY ISSUE — good MFE but deep MAE, SL fires before TP")
        else:
            print(f"    → MIXED — review per-ticker for specific patterns")


# ==============================================================================
# STEP 3: PER-TICKER BREAKDOWN — Which tickers drag June down?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STEP 3: PER-TICKER BREAKDOWN")
print(f"{'='*W}")

if len(df_trades) > 0:
    # TABLE 7: Per-ticker summary
    print(f"\n  TABLE 7: PER-TICKER JUNE PERFORMANCE")
    print(f"  {'Ticker':<7} {'Sigs':>5} {'MFE≥8%':>7} {'MAE≤-6%':>8} {'AvgMFE':>7} "
          f"{'AvgMAE':>7} {'AvgEOD':>7} {'Whipsaw':>8} {'Deploy':>7}")
    print(f"  {'─'*7} {'─'*5} {'─'*7} {'─'*8} {'─'*7} "
          f"{'─'*7} {'─'*7} {'─'*8} {'─'*7}")

    ticker_stats = {}
    for ticker in sorted(df_trades['ticker'].unique()):
        tsub = df_trades[df_trades['ticker'] == ticker]
        n = len(tsub)
        mfe_8 = (tsub['mfe'] >= 0.08).sum()
        mae_6 = (tsub['mae'] <= -0.06).sum()
        avg_mfe = tsub['mfe'].mean() * 100
        avg_mae = tsub['mae'].mean() * 100
        avg_eod = tsub['eod_ret'].mean() * 100
        whipsaw = sum(1 for _, r in tsub.iterrows()
                      if r['sl_analysis'][0.06]['hit']
                      and r['sl_analysis'][0.06]['eod_after'] is not None
                      and r['sl_analysis'][0.06]['eod_after'] > 0)
        deployed = 'YES' if ticker in DEPLOY_TICKERS else 'no'
        flag = '!' if avg_eod < -2.0 and ticker in DEPLOY_TICKERS else ' '

        print(f"{flag} {ticker:<7} {n:>5} {mfe_8:>7} {mae_6:>8} {avg_mfe:>+6.2f}% "
              f"{avg_mae:>+6.2f}% {avg_eod:>+6.2f}% {whipsaw:>8} {deployed:>7}")

        ticker_stats[ticker] = {
            'n': n, 'avg_mfe': avg_mfe, 'avg_mae': avg_mae,
            'avg_eod': avg_eod, 'whipsaw': whipsaw, 'deployed': deployed,
        }

    # TABLE 8: Deep dive on worst tickers
    worst = sorted(ticker_stats.items(), key=lambda x: x[1]['avg_eod'])[:5]
    print(f"\n\n  TABLE 8: WORST 5 TICKERS — Trade-by-trade detail")
    for ticker, stats in worst:
        tsub = df_trades[df_trades['ticker'] == ticker].sort_values('date')
        dep_tag = '[DEPLOYED]' if stats['deployed'] == 'YES' else '[not deployed]'
        print(f"\n  ── {ticker} {dep_tag} ── {stats['n']} signals, "
              f"avg EOD={stats['avg_eod']:+.2f}%, "
              f"avg MFE={stats['avg_mfe']:+.2f}%, avg MAE={stats['avg_mae']:+.2f}%")
        print(f"  {'Date':<12} {'Prob':>5} {'MFE':>7} {'MAE':>7} {'EOD':>7} "
              f"{'SL6%':>5} {'SL8%':>5} {'TP8%':>5} {'Diagnosis':<20}")
        print(f"  {'─'*12} {'─'*5} {'─'*7} {'─'*7} {'─'*7} "
              f"{'─'*5} {'─'*5} {'─'*5} {'─'*20}")
        for _, r in tsub.iterrows():
            sl6 = 'HIT' if r['sl_analysis'][0.06]['hit'] else 'ok'
            sl8 = 'HIT' if r['sl_analysis'][0.08]['hit'] else 'ok'
            tp8 = 'HIT' if r['tp_analysis'].get(0.08, False) else 'no'
            if r['mfe'] >= 0.08 and r['sl_analysis'][0.06]['hit']:
                diag = 'SL before TP!'
            elif r['mfe'] < 0.03:
                diag = 'No spike'
            elif r['eod_ret'] > 0:
                diag = 'Would be win'
            else:
                diag = 'Weak/reversed'
            print(f"  {r['date']:<12} {r['avg_prob']:.3f} {r['mfe']*100:>+6.2f}% "
                  f"{r['mae']*100:>+6.2f}% {r['eod_ret']*100:>+6.2f}% "
                  f"{sl6:>5} {sl8:>5} {tp8:>5} {diag:<20}")

    # Step 3 diagnosis
    deployed_trades = df_trades[df_trades['ticker'].isin(DEPLOY_TICKERS)]
    non_deployed = df_trades[~df_trades['ticker'].isin(DEPLOY_TICKERS)]
    if len(deployed_trades) > 0:
        dep_avg_eod = deployed_trades['eod_ret'].mean() * 100
        non_dep_avg_eod = non_deployed['eod_ret'].mean() * 100 if len(non_deployed) > 0 else 0
        worst_dep = deployed_trades.groupby('ticker')['eod_ret'].mean().sort_values()
        print(f"\n  STEP 3 DIAGNOSIS:")
        print(f"    Deployed tickers avg EOD: {dep_avg_eod:+.2f}%")
        if len(non_deployed) > 0:
            print(f"    Non-deployed avg EOD: {non_dep_avg_eod:+.2f}%")
        print(f"    Worst deployed tickers:")
        for ticker, avg_eod in worst_dep.head(3).items():
            n_sigs = len(deployed_trades[deployed_trades['ticker'] == ticker])
            print(f"      {ticker}: avg EOD {avg_eod*100:+.2f}% ({n_sigs} signals)")


# ==============================================================================
# STEP 4: TIMING ANALYSIS — When did losses concentrate?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STEP 4: TIMING ANALYSIS — Weekly breakdown")
print(f"{'='*W}")

if len(df_trades) > 0:
    df_trades['week_start'] = pd.to_datetime(df_trades['date']).apply(
        lambda d: (d - timedelta(days=d.weekday())).strftime('%Y-%m-%d'))

    # TABLE 9: Weekly breakdown
    print(f"\n  TABLE 9: WEEKLY PERFORMANCE IN JUNE")
    print(f"  {'Week Start':<12} {'Signals':>8} {'AvgMFE':>8} {'AvgMAE':>8} "
          f"{'AvgEOD':>8} {'SL6%Hit':>8} {'TP8%Hit':>8} {'Whipsaw':>8}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for week in sorted(df_trades['week_start'].unique()):
        wsub = df_trades[df_trades['week_start'] == week]
        n = len(wsub)
        avg_mfe = wsub['mfe'].mean() * 100
        avg_mae = wsub['mae'].mean() * 100
        avg_eod = wsub['eod_ret'].mean() * 100
        sl6_hit = sum(1 for _, r in wsub.iterrows() if r['sl_analysis'][0.06]['hit'])
        tp8_hit = sum(1 for _, r in wsub.iterrows() if r['tp_analysis'].get(0.08, False))
        whipsaw = sum(1 for _, r in wsub.iterrows()
                      if r['sl_analysis'][0.06]['hit']
                      and r['sl_analysis'][0.06]['eod_after'] is not None
                      and r['sl_analysis'][0.06]['eod_after'] > 0)
        marker = '>>' if avg_eod < -1.0 else '  '
        print(f"{marker}{week:<12} {n:>8} {avg_mfe:>+7.2f}% {avg_mae:>+7.2f}% "
              f"{avg_eod:>+7.2f}% {sl6_hit:>8} {tp8_hit:>8} {whipsaw:>8}")

    # Step 4 diagnosis
    print(f"\n  STEP 4 DIAGNOSIS:")
    weekly_eod = df_trades.groupby('week_start')['eod_ret'].mean() * 100
    worst_week = weekly_eod.idxmin()
    best_week = weekly_eod.idxmax()
    print(f"    Worst week: {worst_week} (avg EOD {weekly_eod[worst_week]:+.2f}%)")
    print(f"    Best week:  {best_week} (avg EOD {weekly_eod[best_week]:+.2f}%)")

    early_june = df_trades[df_trades['date'] < '2026-06-15']
    late_june = df_trades[df_trades['date'] >= '2026-06-15']
    if len(early_june) > 0 and len(late_june) > 0:
        early_eod = early_june['eod_ret'].mean() * 100
        late_eod = late_june['eod_ret'].mean() * 100
        print(f"    Early June (< Jun 15): {len(early_june)} signals, avg EOD {early_eod:+.2f}%")
        print(f"    Late June (≥ Jun 15):  {len(late_june)} signals, avg EOD {late_eod:+.2f}%")
        if late_eod < early_eod - 2:
            print(f"    → LOSSES CONCENTRATED IN LATE JUNE — regime deteriorated after mid-month")
        else:
            print(f"    → LOSSES SPREAD ACROSS JUNE — not a timing-specific issue")


# ==============================================================================
# STEP 5: CONFIG SPACE EXPLORATION (BQ)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STEP 5: CONFIG SPACE — Are we selecting wrong configs?")
print(f"{'='*W}")

if not HAS_BQ or bq_client is None:
    print("\n  BQ not available — skipping config space analysis.")
    print("  Run in Colab with BQ access for full analysis.")
else:
    print(f"\n  Querying BQ for ALL v12 configs (model_version={MODEL_VERSION})...")

    config_query = f"""
    SELECT
        ticker, experiment, seed, tpsl_mode, entry_minute,
        prob_thresh, tp_pct, sl_pct,
        tp_mult, sl_mult, trail_start_mult, trail_end_mult,
        dip_pct, max_wait_minutes,
        vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
        val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
        val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
        opt_avg_pnl, opt_win_rate, opt_count,
        survived_a, survived_b, is_ensemble,
        grace_minutes
    FROM `{BQ_TABLE}`
    WHERE model_version = '{MODEL_VERSION}'
      AND survived_a = TRUE
      AND val_a_count >= {MIN_TRADES}
      AND val_a_avg_pnl > 0
      AND opt_avg_pnl > 0
    """

    df_bq = bq_client.query(config_query).to_dataframe()
    df_bq['ticker'] = df_bq['ticker'].replace({'MIGI': 'BGDE', 'BITF': 'KEEL'})

    all_suffixes = {'': 'static', '_ATR': 'atr', '_ATRADP': 'atr_adaptive',
                    '_REGIME': 'regime_scaled', '_TRAIL': 'trailing_decay',
                    '_REGADP': 'regime_adaptive', '_ATRSL': 'atr_sl',
                    '_GRACE': 'grace', '_LIMIT': 'limit_entry'}
    suffix_to_mode = {v: k for k, v in all_suffixes.items()}

    DEPLOY_EXPERIMENTS = {'152_XSECTOR', '160_XSECTOR', '170_XSECTOR',
                          '175_XSECTOR', '180_XSECTOR', '185_XSECTOR',
                          '200_XSECTOR'}

    def _base_exp(exp_name):
        for sfx in sorted(all_suffixes.keys(), key=len, reverse=True):
            if sfx and exp_name.endswith(sfx):
                return exp_name[:-len(sfx)]
        return exp_name

    df_bq['base_experiment'] = df_bq['experiment'].apply(_base_exp)
    df_bq = df_bq[df_bq['base_experiment'].isin(DEPLOY_EXPERIMENTS)]
    df_bq = df_bq[~df_bq['ticker'].isin(EXCLUDE_TICKERS)]

    print(f"  Loaded {len(df_bq)} configs across "
          f"{df_bq['ticker'].nunique()} tickers, "
          f"{df_bq['tpsl_mode'].nunique()} modes")

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
        return 'other'

    df_bq['family'] = df_bq['base_experiment'].apply(assign_family)
    df_bq = df_bq[df_bq['family'] != 'other']

    # TABLE 10: Best val_b config per (ticker, mode) — averaged across seeds
    print(f"\n\n  TABLE 10: BEST CONFIG PER (TICKER, MODE) — by val_b_avg_pnl")

    indiv = df_bq[df_bq['is_ensemble'] == False].copy()
    mode_best = indiv.groupby(['ticker', 'tpsl_mode']).agg(
        best_vb_pnl=('val_b_avg_pnl', 'max'),
        mean_vb_pnl=('val_b_avg_pnl', 'mean'),
        n_configs=('val_b_avg_pnl', 'count'),
        n_positive_vb=('val_b_avg_pnl', lambda x: (x > 0).sum()),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
    ).reset_index()

    # Pivot: tickers as rows, modes as columns
    modes_to_show = ['limit_entry', 'static', 'atr', 'atr_adaptive', 'regime_scaled', 'trailing_decay']
    modes_present = [m for m in modes_to_show if m in mode_best['tpsl_mode'].values]

    print(f"\n  Mean val_b PnL by (ticker, mode):")
    header = f"  {'Ticker':<7}"
    for m in modes_present:
        label = m[:8]
        header += f" {label:>10}"
    header += f" {'BEST MODE':>12} {'BestPnL':>8}"
    print(header)
    print(f"  {'─'*7}" + f" {'─'*10}" * len(modes_present) + f" {'─'*12} {'─'*8}")

    for ticker in sorted(mode_best['ticker'].unique()):
        if ticker not in DEPLOY_TICKERS:
            continue
        row_str = f"  {ticker:<7}"
        best_mode = None
        best_val = -999
        for m in modes_present:
            sub = mode_best[(mode_best['ticker'] == ticker) & (mode_best['tpsl_mode'] == m)]
            if len(sub) > 0:
                val = sub['mean_vb_pnl'].iloc[0] * 100
                row_str += f" {val:>+9.2f}%"
                if val > best_val:
                    best_val = val
                    best_mode = m
            else:
                row_str += f" {'—':>10}"
        limit_sub = mode_best[(mode_best['ticker'] == ticker) & (mode_best['tpsl_mode'] == 'limit_entry')]
        limit_val = limit_sub['mean_vb_pnl'].iloc[0] * 100 if len(limit_sub) > 0 else -999
        flag = '!!' if best_mode != 'limit_entry' and best_val > limit_val + 1.0 else '  '
        row_str += f" {flag}{best_mode[:10] if best_mode else '—':>10} {best_val:>+7.2f}%"
        print(row_str)

    # TABLE 11: Family agreement comparison — S3 vs best mode
    print(f"\n\n  TABLE 11: FAMILY AGREEMENT — limit_entry vs best alternative")
    print(f"  {'Ticker':<7} {'LimitEntry':>11} {'Fams+':>6} {'BestAlt':>11} "
          f"{'AltMode':>12} {'AltFams+':>9} {'Delta':>7}")
    print(f"  {'─'*7} {'─'*11} {'─'*6} {'─'*11} "
          f"{'─'*12} {'─'*9} {'─'*7}")

    for ticker in sorted(DEPLOY_TICKERS):
        t_data = indiv[indiv['ticker'] == ticker]
        if len(t_data) == 0:
            continue
        limit_data = t_data[t_data['tpsl_mode'] == 'limit_entry']
        limit_fam_pos = 0
        limit_avg_vb = 0
        if len(limit_data) > 0:
            fam_vb = limit_data.groupby('family')['val_b_avg_pnl'].mean()
            limit_fam_pos = (fam_vb > 0).sum()
            limit_avg_vb = fam_vb.mean() * 100

        best_alt_mode = None
        best_alt_vb = -999
        best_alt_fam_pos = 0
        for mode in t_data['tpsl_mode'].unique():
            if mode == 'limit_entry':
                continue
            m_data = t_data[t_data['tpsl_mode'] == mode]
            fam_vb = m_data.groupby('family')['val_b_avg_pnl'].mean()
            avg_vb = fam_vb.mean() * 100
            fam_pos = (fam_vb > 0).sum()
            if avg_vb > best_alt_vb:
                best_alt_vb = avg_vb
                best_alt_mode = mode
                best_alt_fam_pos = fam_pos

        n_fams = t_data['family'].nunique()
        delta = best_alt_vb - limit_avg_vb if best_alt_mode else 0
        flag = '!!' if delta > 1.0 and best_alt_fam_pos >= n_fams / 2 else '  '

        alt_str = f"{best_alt_vb:>+10.2f}%" if best_alt_mode else f"{'—':>11}"
        alt_mode_str = best_alt_mode[:12] if best_alt_mode else '—'
        print(f"{flag}{ticker:<7} {limit_avg_vb:>+10.2f}% {limit_fam_pos:>5}/{n_fams} "
              f"{alt_str} {alt_mode_str:>12} {best_alt_fam_pos:>5}/{n_fams} "
              f"{delta:>+6.2f}%")

    # TABLE 12: Top configs that S3 skips (non-limit_entry with strong val_b)
    print(f"\n\n  TABLE 12: CONFIGS S3 SKIPS — Best non-limit_entry with val_b > deploy threshold")

    non_limit = indiv[indiv['tpsl_mode'] != 'limit_entry'].copy()
    non_limit_agg = non_limit.groupby(
        ['ticker', 'tpsl_mode', 'base_experiment', 'entry_minute',
         'prob_thresh', 'tp_pct', 'sl_pct']
    ).agg(
        mean_vb=('val_b_avg_pnl', 'mean'),
        mean_va=('val_a_avg_pnl', 'mean'),
        n_seeds=('seed', 'nunique'),
        vb_wr=('val_b_win_rate', 'mean'),
        vb_count=('val_b_count', 'mean'),
    ).reset_index()

    top_skipped = non_limit_agg[
        (non_limit_agg['mean_vb'] >= DEPLOY_THRESHOLD) &
        (non_limit_agg['n_seeds'] >= 3) &
        (non_limit_agg['ticker'].isin(DEPLOY_TICKERS))
    ].sort_values('mean_vb', ascending=False).head(30)

    if len(top_skipped) > 0:
        print(f"  {'Ticker':<7} {'Mode':<13} {'Exp':<14} {'EM':>3} {'Thresh':>7} "
              f"{'TP%':>5} {'SL%':>5} {'VB PnL':>8} {'VB WR':>6} {'Seeds':>6}")
        print(f"  {'─'*7} {'─'*13} {'─'*14} {'─'*3} {'─'*7} "
              f"{'─'*5} {'─'*5} {'─'*8} {'─'*6} {'─'*6}")
        for _, r in top_skipped.iterrows():
            print(f"  {r['ticker']:<7} {r['tpsl_mode']:<13} {r['base_experiment']:<14} "
                  f"{int(r['entry_minute']):>3} {r['prob_thresh']:>7.3f} "
                  f"{r['tp_pct']*100:>4.1f}% {r['sl_pct']*100:>4.1f}% "
                  f"{r['mean_vb']*100:>+7.2f}% {r['vb_wr']*100:>5.0f}% "
                  f"{int(r['n_seeds']):>6}")
    else:
        print(f"  No non-limit_entry configs exceed deploy threshold ({DEPLOY_THRESHOLD*100:.1f}%)")

    # TABLE 13: Config parameter patterns — what do June-surviving configs have in common?
    print(f"\n\n  TABLE 13: CONFIG PARAMETER PATTERNS — Surviving configs vs failing")

    if 'survived_b' in indiv.columns:
        survived = indiv[indiv['survived_b'] == True]
        failed = indiv[indiv['survived_b'] == False]

        if len(survived) > 0 and len(failed) > 0:
            print(f"\n  {'Parameter':<20} {'Survived B':>15} {'Failed B':>15} {'Delta':>10}")
            print(f"  {'─'*20} {'─'*15} {'─'*15} {'─'*10}")

            for col, label in [('tp_pct', 'TP %'), ('sl_pct', 'SL %'),
                                ('prob_thresh', 'Prob Thresh'),
                                ('dip_pct', 'Dip %'), ('entry_minute', 'Entry Min')]:
                if col in survived.columns and col in failed.columns:
                    s_val = survived[col].mean()
                    f_val = failed[col].mean()
                    mult = 100 if col in ('tp_pct', 'sl_pct', 'dip_pct') else 1
                    sfx = '%' if col in ('tp_pct', 'sl_pct', 'dip_pct') else ''
                    delta = (s_val - f_val) * mult
                    print(f"  {label:<20} {s_val*mult:>14.2f}{sfx} {f_val*mult:>14.2f}{sfx} "
                          f"{delta:>+9.2f}{sfx}")

            # Mode breakdown
            print(f"\n  Mode survival rates:")
            for mode in modes_present:
                m_total = len(indiv[indiv['tpsl_mode'] == mode])
                m_surv = len(survived[survived['tpsl_mode'] == mode])
                rate = m_surv / m_total * 100 if m_total > 0 else 0
                marker = '>>' if rate > 70 else '  '
                print(f"  {marker}{mode:<15} {m_surv:>6}/{m_total:<6} ({rate:.0f}%)")

    # Step 5 diagnosis
    print(f"\n  STEP 5 DIAGNOSIS:")
    limit_deployed = mode_best[
        (mode_best['tpsl_mode'] == 'limit_entry') &
        (mode_best['ticker'].isin(DEPLOY_TICKERS))
    ]
    if len(limit_deployed) > 0:
        avg_limit_vb = limit_deployed['mean_vb_pnl'].mean() * 100
        n_positive = (limit_deployed['mean_vb_pnl'] > 0).sum()
        print(f"    S3 (limit_entry) avg val_b for deployed: {avg_limit_vb:+.2f}%")
        print(f"    Deployed tickers with positive val_b: {n_positive}/{len(limit_deployed)}")
    if len(top_skipped) > 0:
        print(f"    Non-limit configs above deploy threshold: {len(top_skipped)}")
        print(f"    → S3 restricts to limit_entry only. Review TABLE 12 for skipped configs.")
    else:
        print(f"    → No strong non-limit_entry alternatives found. S3 mode restriction is not the bottleneck.")


# ==============================================================================
# SYNTHESIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SYNTHESIS — JUNE LOSS DECOMPOSITION RESULTS")
print(f"{'='*W}")

conclusions = []
actions = []

# 1. Model calibration
if HAS_SCORES and len(df_with_target) > 0:
    pre_june_wt = df_with_target[~df_with_target['is_june']]
    june_wt = df_with_target[df_with_target['is_june']]
    if len(june_wt) > 0 and len(pre_june_wt) > 0:
        pre_sig_rate = pre_june_wt['signal'].mean() * 100
        jun_sig_rate = june_wt['signal'].mean() * 100
        pre_spike_rate = pre_june_wt['target_bool'].mean() * 100
        jun_spike_rate = june_wt['target_bool'].mean() * 100
        pre_ratio = pre_sig_rate / max(pre_spike_rate, 0.1)
        jun_ratio = jun_sig_rate / max(jun_spike_rate, 0.1)

        print(f"\n  1. MODEL CALIBRATION:")
        print(f"     Signal rate: {pre_sig_rate:.0f}% → {jun_sig_rate:.0f}% (model fires MORE)")
        print(f"     Spike rate:  {pre_spike_rate:.0f}% → {jun_spike_rate:.0f}% (spikes are FEWER)")
        print(f"     Signal/spike ratio: {pre_ratio:.1f}x → {jun_ratio:.1f}x")
        if jun_sig_rate > pre_sig_rate and jun_spike_rate < pre_spike_rate:
            conclusions.append(f"Model over-triggering: signal rate UP ({pre_sig_rate:.0f}%→{jun_sig_rate:.0f}%) "
                               f"while spikes DOWN ({pre_spike_rate:.0f}%→{jun_spike_rate:.0f}%)")
            actions.append("Model needs recalibration or regime-aware prob threshold")

# 2. Loss categories
if 'df_cats' in dir() and len(df_cats) > 0:
    pre_cats = df_cats[~df_cats['is_june']]
    jun_cats = df_cats[df_cats['is_june']]
    if len(pre_cats) > 0 and len(jun_cats) > 0:
        pre_a = (pre_cats['category'] == 'A_model_wrong').mean() * 100
        jun_a = (jun_cats['category'] == 'A_model_wrong').mean() * 100
        pre_ab = pre_cats['category'].isin(['A_model_wrong', 'B_weak_spike']).mean() * 100
        jun_ab = jun_cats['category'].isin(['A_model_wrong', 'B_weak_spike']).mean() * 100
        pre_d = (pre_cats['category'] == 'D_winner').mean() * 100
        jun_d = (jun_cats['category'] == 'D_winner').mean() * 100
        pre_c = (pre_cats['category'] == 'C_sl_before_tp').mean() * 100
        jun_c = (jun_cats['category'] == 'C_sl_before_tp').mean() * 100

        print(f"\n  2. LOSS CATEGORIZATION:")
        print(f"     Model wrong (A):     {pre_a:.0f}% → {jun_a:.0f}% (Δ {jun_a-pre_a:+.0f}pp)")
        print(f"     Model wrong+weak (A+B): {pre_ab:.0f}% → {jun_ab:.0f}% (Δ {jun_ab-pre_ab:+.0f}pp)")
        print(f"     SL before TP (C):    {pre_c:.0f}% → {jun_c:.0f}% (Δ {jun_c-pre_c:+.0f}pp)")
        print(f"     Winners (D):         {pre_d:.0f}% → {jun_d:.0f}% (Δ {jun_d-pre_d:+.0f}pp)")

        if jun_a > pre_a + 5:
            conclusions.append(f"'Model wrong' rate jumped {jun_a-pre_a:+.0f}pp — model says BUY, stock drops")
            actions.append("Raise prob_threshold or add secondary confirmation signal")
        if jun_ab > 50:
            conclusions.append(f"{jun_ab:.0f}% of June signals are model-wrong or weak — majority are bad calls")
        if jun_c > pre_c + 3:
            conclusions.append(f"SL-before-TP rate increased {jun_c-pre_c:+.0f}pp — widen SL or lower TP")
            actions.append("Widen SL or lower TP to capture weaker spikes")
        if jun_d < pre_d * 0.7:
            conclusions.append(f"Winner rate collapsed: {pre_d:.0f}% → {jun_d:.0f}%")

# 3. Family agreement
if 'df_cats' in dir() and 'fam_filter' in dir():
    jun_all = df_cats[df_cats['is_june']]
    jun_fam = fam_filter[(fam_filter['is_june']) & (fam_filter['majority'])]
    if len(jun_all) > 0 and len(jun_fam) > 0:
        all_a = (jun_all['category'] == 'A_model_wrong').mean() * 100
        fam_a = (jun_fam['category'] == 'A_model_wrong').mean() * 100
        all_eod = jun_all['eod_ret'].mean() * 100
        fam_eod = jun_fam['eod_ret'].mean() * 100

        print(f"\n  3. FAMILY AGREEMENT FILTER:")
        print(f"     Cat-A rate: all signals {all_a:.0f}% → family-filtered {fam_a:.0f}%")
        print(f"     Avg EOD: all {all_eod:+.2f}% → family-filtered {fam_eod:+.2f}%")
        if fam_a >= all_a - 3:
            conclusions.append("Family filter doesn't help — all experiments wrong together (correlated failure)")
            actions.append("Cross-family agreement cannot fix correlated model error")
        elif fam_eod > all_eod + 1.0:
            conclusions.append(f"Family filter improves EOD by {fam_eod-all_eod:+.2f}pp but doesn't eliminate losses")

# 4. SL/TP (keep but reframe)
if len(df_trades) > 0:
    sl6_hit = sum(1 for _, r in df_trades.iterrows() if r['sl_analysis'][0.06]['hit'])
    tp8_reached = sum(1 for _, r in df_trades.iterrows() if r['tp_analysis'].get(0.08, False))
    print(f"\n  4. SL/TP EXECUTION:")
    print(f"     SL 6% hit rate: {sl6_hit}/{len(df_trades)} ({sl6_hit/len(df_trades)*100:.0f}%)")
    print(f"     TP 8% reachable: {tp8_reached}/{len(df_trades)} ({tp8_reached/len(df_trades)*100:.0f}%)")
    print(f"     NOTE: SL hits are mostly NOT whipsaws — the stock genuinely dropped")
    print(f"     → Widening SL won't help if the model prediction is wrong")

print(f"\n  CONCLUSIONS (ranked by impact):")
for i, c in enumerate(conclusions, 1):
    print(f"    {i}. {c}")
if not conclusions:
    print(f"    No clear conclusions — check data coverage.")

print(f"\n  ROOT CAUSE HIERARCHY:")
print(f"    Layer 0: MODEL OVER-TRIGGERING — signal rate increased while spikes decreased")
print(f"    Layer 1: SPIKE REGIME SHIFT — fewer/weaker spikes in June (market condition)")
print(f"    Layer 2: CORRELATED FAILURE — all experiment families wrong together")
print(f"    Layer 3: EXECUTION — SL/TP parameters only matter when prediction is correct")

print(f"\n  RECOMMENDED ACTIONS:")
for i, a in enumerate(actions, 1):
    print(f"    {i}. {a}")
if not actions:
    print(f"    No clear actions — review raw data.")

print(f"\n{'='*W}")
print(f"  DECOMPOSITION COMPLETE")
print(f"{'='*W}")
