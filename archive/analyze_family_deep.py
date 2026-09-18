"""
analyze_family_deep.py — Root Cause Diagnosis: Regime Change vs Small-N
========================================================================
Quantitative diagnosis of WHY OPT→TEST degradation occurs.
Two hypotheses:
  H1: Regime change — spike rates decaying, model can't adapt
  H2: Small-N validation — optimizer picks flukes with <10 trades

Experiments designed to test these:
  152 (Oct 2025 cutoff) — fresh model, 7 months stale at TEST
  160 (Mar 2025 cutoff) — stale model, 13 months stale at TEST
  170-175 (walk-forward) — 6 folds, progressively more recent training data

Tables:
  1: Regime Decay — spike rate trends from actual data
  2: Walk-Forward Fold Analysis — does recent training help?
  3: Staleness Test — 152 vs 160 paired comparison
  4: N-Sensitivity — does trade count predict degradation?
  5: Seed Agreement — how many seeds agree on direction?
  6: Degradation Decomposition — frequency vs win rate vs magnitude
  7: Bootstrap Small-N Simulation — expected noise at observed N
  8: Diagnosis Summary — quantitative root cause attribution

Run in Colab: paste as cell, runs immediately.
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import bigquery

# ==============================================================================
# CONFIG
# ==============================================================================
BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"
SPIKE_THRESHOLD = 0.10

MODEL_VERSION = 'robust_wf_v12'

CLEAN_EXP_BASES = ('152_XSECTOR', '160_XSECTOR',
                   '170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                   '173_XSECTOR', '174_XSECTOR', '175_XSECTOR',
                   '180_XSECTOR', '181_XSECTOR', '182_XSECTOR',
                   '183_XSECTOR', '184_XSECTOR', '185_XSECTOR')

FOLD_MAP = {
    '170_XSECTOR': {'cutoff': '2025-03-01', 'label': 'F1 (Mar25)', 'order': 1},
    '171_XSECTOR': {'cutoff': '2025-05-01', 'label': 'F2 (May25)', 'order': 2},
    '172_XSECTOR': {'cutoff': '2025-07-01', 'label': 'F3 (Jul25)', 'order': 3},
    '173_XSECTOR': {'cutoff': '2025-09-01', 'label': 'F4 (Sep25)', 'order': 4},
    '174_XSECTOR': {'cutoff': '2025-11-01', 'label': 'F5 (Nov25)', 'order': 5},
    '175_XSECTOR': {'cutoff': '2026-01-01', 'label': 'F6 (Jan26)', 'order': 6},
}

FOLD_MAP_180 = {
    '180_XSECTOR': {'cutoff': '2025-08-01', 'label': 'F1 (Aug25)', 'order': 1},
    '181_XSECTOR': {'cutoff': '2025-09-01', 'label': 'F2 (Sep25)', 'order': 2},
    '182_XSECTOR': {'cutoff': '2025-10-01', 'label': 'F3 (Oct25)', 'order': 3},
    '183_XSECTOR': {'cutoff': '2025-11-01', 'label': 'F4 (Nov25)', 'order': 4},
    '184_XSECTOR': {'cutoff': '2025-12-01', 'label': 'F5 (Dec25)', 'order': 5},
    '185_XSECTOR': {'cutoff': '2026-01-01', 'label': 'F6 (Jan26)', 'order': 6},
}

REGIME_CHECK_TICKERS = [
    'LUNR', 'BBAI', 'WULF', 'CIFR', 'IREN', 'CLSK', 'HUT', 'MARA', 'RIOT',
    'RCAT', 'ASTS', 'ACHR', 'MVIS', 'KEEL', 'APLD',
]

TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}

_TPSL_SUFFIXES = ('_ATRADP', '_REGADP', '_TRAIL', '_REGIME', '_ATRSL', '_ATR',
                  '_GRACE', '_LIMIT')
def _base_exp(e):
    for sfx in _TPSL_SUFFIXES:
        if e.endswith(sfx):
            return e[:-len(sfx)]
    return e

def _exp_family(exp_name):
    base = _base_exp(exp_name)
    if 'XSECTOR' not in base:
        return None
    num_str = base.split('_')[0]
    if not num_str.isdigit():
        return None
    num = int(num_str)
    if num == 152: return 'late_cutoff'
    if num == 160: return 'early_cutoff'
    if 170 <= num <= 175: return 'walkforward_170'
    if 180 <= num <= 185: return 'walkforward_180'
    return None

FAMILIES = ['late_cutoff', 'early_cutoff', 'walkforward_170', 'walkforward_180']
FAMILY_LABELS = {'late_cutoff': '152 (Oct25)', 'early_cutoff': '160 (Mar25)',
                 'walkforward_170': '170 WF', 'walkforward_180': '180 WF'}

W = 120
np.random.seed(42)

# ==============================================================================
# LOAD BQ DATA
# ==============================================================================
print(f"{'='*W}")
print(f"  ROOT CAUSE DIAGNOSIS — Regime Change vs Small-N")
print(f"  H1: Spike regime is decaying → model predictions degrade over time")
print(f"  H2: Too few trades in OPT → optimizer selects flukes")
print(f"{'='*W}")

client = bigquery.Client()
query = f"""
SELECT experiment, ticker, seed, tpsl_mode, is_ensemble, run_id,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
       survived_a, survived_b, model_version, run_timestamp,
       prob_thresh, tp_pct, sl_pct, dip_pct, max_wait_minutes,
       grace_minutes, vol_filter_pct
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}'
ORDER BY experiment, ticker, seed
"""
df_all = client.query(query).to_dataframe()

if df_all.empty:
    print("\n  No data in BQ table.")
    raise SystemExit(0)

df_all['ticker'] = df_all['ticker'].replace(TICKER_RENAME)
df_all = df_all[~df_all['ticker'].isin(EXCLUDE_TICKERS)]
df_all = df_all[df_all['experiment'].apply(
    lambda e: any(e == b or e.startswith(b + '_') for b in CLEAN_EXP_BASES))]

if 'model_version' not in df_all.columns:
    df_all['model_version'] = 'unknown'
df_all['model_version'] = df_all['model_version'].fillna('unknown')

if 'tpsl_mode' not in df_all.columns:
    df_all['tpsl_mode'] = 'static'
df_all['tpsl_mode'] = df_all['tpsl_mode'].fillna('static').replace('', 'static')

df_all['survived_a'] = df_all['survived_a'].fillna(False).astype(bool)
df_all['survived_b'] = df_all['survived_b'].fillna(False).astype(bool)
for col in ['val_a_avg_pnl', 'val_b_avg_pnl', 'val_a_win_rate', 'val_b_win_rate',
            'val_a_freq', 'val_b_freq']:
    if col in df_all.columns:
        df_all[col] = df_all[col].fillna(0.0)
for col in ['val_a_count', 'val_b_count']:
    if col in df_all.columns:
        df_all[col] = df_all[col].fillna(0).astype(int)

if 'is_ensemble' not in df_all.columns:
    df_all['is_ensemble'] = False
df_all['is_ensemble'] = df_all['is_ensemble'].fillna(False).astype(bool)
df_all['run_timestamp'] = pd.to_datetime(df_all['run_timestamp'])

df = df_all[df_all['model_version'] == MODEL_VERSION].copy()
if df.empty:
    print(f"\n  No {MODEL_VERSION} results found. Trying any version...")
    df = df_all.copy()

# Dedup: keep latest run per (experiment, ticker, seed)
_latest = (df.groupby(['experiment', 'ticker', 'run_id'])['run_timestamp']
           .max().reset_index()
           .sort_values('run_timestamp')
           .groupby(['experiment', 'ticker']).last()
           .reset_index()[['experiment', 'ticker', 'run_id']])
df = df.merge(_latest, on=['experiment', 'ticker', 'run_id'], how='inner')

df = df[~df['is_ensemble']].copy()
df['base_experiment'] = df['experiment'].apply(_base_exp)
df['family'] = df['experiment'].apply(_exp_family)
df = df[df['family'].notna()].copy()

families_present = [f for f in FAMILIES if f in df['family'].unique()]
all_tickers = sorted(df['ticker'].unique())

print(f"\n  Loaded {len(df)} seed rows across {len(all_tickers)} tickers")
print(f"  Families: {[FAMILY_LABELS[f] for f in families_present]}")
print(f"  Model version filter: {MODEL_VERSION}")

# ==============================================================================
# TABLE 1: REGIME DECAY — Direct Spike Rate Measurement
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: REGIME DECAY — Monthly Spike Rate Trends")
print(f"  Loading master_inference data from GCS to measure actual spike frequency")
print(f"{'='*W}")

regime_data = {}
for ticker in REGIME_CHECK_TICKERS:
    clean = TICKER_RENAME.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        tdf = pd.read_parquet(uri, columns=['Target', 'Valid_Row'])
        tdf.index = pd.to_datetime(tdf.index, utc=True)
        valid = tdf[tdf['Valid_Row'] == 1]
        monthly = valid.resample('ME').agg(
            n_bars=('Target', 'count'),
            n_spikes=('Target', 'sum'),
        )
        monthly['spike_rate'] = monthly['n_spikes'] / monthly['n_bars']
        monthly = monthly[monthly['n_bars'] >= 100]
        regime_data[ticker] = monthly
        print(f"  {ticker}: {len(monthly)} months loaded")
    except Exception as e:
        print(f"  {ticker}: FAILED ({e})")

if regime_data:
    start_month = '2024-01'
    print(f"\n  Monthly spike rates (Valid_Row==1 bars where Target==1, from {start_month}):")
    all_months = sorted(set(
        m.strftime('%Y-%m')
        for df_m in regime_data.values()
        for m in df_m.index
        if m.strftime('%Y-%m') >= start_month
    ))

    hdr = f"  {'Ticker':<7}"
    for m in all_months:
        hdr += f" {m[2:]:>6}"
    hdr += f"  {'Slope':>7} {'p-val':>6} {'Trend':>8}"
    print(hdr)
    print(f"  {'─'*7}" + f" {'─'*6}" * len(all_months) + f"  {'─'*7} {'─'*6} {'─'*8}")

    slopes = []
    for ticker in REGIME_CHECK_TICKERS:
        if ticker not in regime_data:
            continue
        md = regime_data[ticker]
        line = f"  {ticker:<7}"
        rates = []
        month_indices = []
        for i, m in enumerate(all_months):
            ts_candidates = md.index[md.index.strftime('%Y-%m') == m]
            if len(ts_candidates) > 0:
                rate = float(md.loc[ts_candidates[0], 'spike_rate'])
                line += f" {rate:>5.1%}"
                rates.append(rate)
                month_indices.append(i)
            else:
                line += f"  {'—':>4}"

        if len(rates) >= 6:
            slope, intercept, r_val, p_val, std_err = sp_stats.linregress(month_indices, rates)
            slope_pct = slope * 100
            trend = "DECAY" if slope < -0.001 and p_val < 0.10 else (
                "RISING" if slope > 0.001 and p_val < 0.10 else "FLAT")
            line += f"  {slope_pct:>+6.2f}% {p_val:>5.3f} {trend:>8}"
            slopes.append({'ticker': ticker, 'slope': slope, 'p_val': p_val, 'trend': trend})
        else:
            line += f"  {'—':>7} {'—':>6} {'—':>8}"
        print(line)

    if slopes:
        n_decay = sum(1 for s in slopes if s['trend'] == 'DECAY')
        n_flat = sum(1 for s in slopes if s['trend'] == 'FLAT')
        n_rising = sum(1 for s in slopes if s['trend'] == 'RISING')
        avg_slope = np.mean([s['slope'] for s in slopes]) * 100
        print(f"\n  VERDICT: {n_decay}/{len(slopes)} tickers show significant decay, "
              f"{n_flat} flat, {n_rising} rising")
        print(f"  Avg slope: {avg_slope:+.3f}%/month")
        if n_decay > len(slopes) * 0.5:
            t1_verdict = "SUPPORTS H1 (regime decay)"
            t1_score = min(5, n_decay)
        elif n_decay > len(slopes) * 0.25:
            t1_verdict = "WEAKLY SUPPORTS H1"
            t1_score = 2
        else:
            t1_verdict = "AGAINST H1 (spikes are stable)"
            t1_score = 0
        print(f"  → {t1_verdict}")
else:
    t1_verdict = "INCONCLUSIVE (no data loaded)"
    t1_score = 0
    slopes = []

# ==============================================================================
# TABLE 2: WALK-FORWARD FOLD ANALYSIS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 2: WALK-FORWARD FOLD ANALYSIS — Does Recent Training Help?")
print(f"  Each fold trained on progressively more recent data")
print(f"  If later folds perform better → regime change IS happening")
print(f"{'='*W}")

for wf_label, wf_fold_map in [('170-175 WF', FOLD_MAP), ('180-185 WF', FOLD_MAP_180)]:
    fold_exps = [e for e in sorted(wf_fold_map.keys()) if e in df['base_experiment'].values]
    if not fold_exps:
        print(f"\n  {wf_label}: No fold data found.")
        continue
    print(f"\n  {wf_label}:")
    hdr2 = f"  {'Fold':<14} {'Cutoff':<12} {'Avg TEST':>9} {'Med TEST':>9} {'Avg WR':>7} {'Avg N':>6} {'Surv%':>6} {'Seeds':>6}"
    print(hdr2)
    print(f"  {'─'*14} {'─'*12} {'─'*9} {'─'*9} {'─'*7} {'─'*6} {'─'*6} {'─'*6}")

    fold_test_pnls = []
    fold_orders = []

    for fexp in fold_exps:
        fi = wf_fold_map[fexp]
        fd = df[df['base_experiment'] == fexp]
        surv = fd[fd['survived_a']]

        if surv.empty:
            print(f"  {fi['label']:<14} {fi['cutoff']:<12}   (no survived seeds)")
            continue

        test_pnls = surv['val_b_avg_pnl'].values
        avg_test = np.mean(test_pnls)
        med_test = np.median(test_pnls)
        avg_wr = surv['val_b_win_rate'].mean()
        avg_n = surv['val_b_count'].mean()
        surv_rate = len(surv) / len(fd) if len(fd) > 0 else 0
        n_seeds = fd['seed'].nunique()

        fold_test_pnls.append(avg_test)
        fold_orders.append(fi['order'])

        print(f"  {fi['label']:<14} {fi['cutoff']:<12} {avg_test:>8.2%} {med_test:>8.2%} "
              f"{avg_wr:>6.0%} {avg_n:>5.1f} {surv_rate:>5.0%} {n_seeds:>6}")

    if len(fold_test_pnls) >= 3:
        slope, _, r_val, p_val, _ = sp_stats.linregress(fold_orders, fold_test_pnls)
        rho, rho_p = sp_stats.spearmanr(fold_orders, fold_test_pnls)
        print(f"\n  {wf_label} fold performance trend:")
        print(f"    Linear slope: {slope*100:+.2f}%/fold  (p={p_val:.3f})")
        print(f"    Spearman rho: {rho:+.2f}  (p={rho_p:.3f})")
        if slope > 0 and p_val < 0.15:
            print(f"  → SUPPORTS H1 — later folds (more recent training) perform better")
        elif slope < 0 and p_val < 0.15:
            print(f"  → AGAINST H1 — earlier folds perform better (overfitting to recent noise?)")
        else:
            print(f"  → INCONCLUSIVE — no clear trend across folds")
    else:
        print(f"\n  {wf_label}: too few folds with data for trend analysis")

t2_verdict = "See per-family fold analysis above"
t2_score = 2

# ==============================================================================
# TABLE 3: STALENESS TEST — 152 vs 160
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: STALENESS TEST — 152 (Oct25 cutoff) vs 160 (Mar25 cutoff)")
print(f"  Same architecture, different training recency. TEST on same Apr-May window.")
print(f"  If 152 >> 160 → model freshness matters → regime shift")
print(f"{'='*W}")

fam_152 = df[df['family'] == 'late_cutoff'] if 'late_cutoff' in families_present else pd.DataFrame()
fam_160 = df[df['family'] == 'early_cutoff'] if 'early_cutoff' in families_present else pd.DataFrame()

if not fam_152.empty and not fam_160.empty:
    hdr3 = (f"  {'Ticker':<7} | {'152 TEST':>8} {'152 WR':>7} {'152 N':>5} "
            f"| {'160 TEST':>8} {'160 WR':>7} {'160 N':>5} | {'Diff':>7} {'Winner':>7}")
    print(hdr3)
    print(f"  {'─'*7} ─ {'─'*8} {'─'*7} {'─'*5} ─ {'─'*8} {'─'*7} {'─'*5} ─ {'─'*7} {'─'*7}")

    paired_diffs = []
    common_tickers = sorted(set(fam_152['ticker'].unique()) & set(fam_160['ticker'].unique()))

    for ticker in common_tickers:
        s152 = fam_152[(fam_152['ticker'] == ticker) & fam_152['survived_a']]
        s160 = fam_160[(fam_160['ticker'] == ticker) & fam_160['survived_a']]

        avg_152 = s152['val_b_avg_pnl'].mean() if not s152.empty else np.nan
        avg_160 = s160['val_b_avg_pnl'].mean() if not s160.empty else np.nan
        wr_152 = s152['val_b_win_rate'].mean() if not s152.empty else np.nan
        wr_160 = s160['val_b_win_rate'].mean() if not s160.empty else np.nan
        n_152 = int(s152['val_b_count'].mean()) if not s152.empty else 0
        n_160 = int(s160['val_b_count'].mean()) if not s160.empty else 0

        if np.isnan(avg_152) and np.isnan(avg_160):
            continue

        diff = (avg_152 if not np.isnan(avg_152) else 0) - (avg_160 if not np.isnan(avg_160) else 0)
        winner = "152" if diff > 0.001 else ("160" if diff < -0.001 else "TIE")

        if not np.isnan(avg_152) and not np.isnan(avg_160):
            paired_diffs.append(diff)

        line = f"  {ticker:<7}"
        line += f" | {avg_152:>7.1%} {wr_152:>6.0%} {n_152:>5}" if not np.isnan(avg_152) else f" | {'—':>8} {'—':>7} {'—':>5}"
        line += f" | {avg_160:>7.1%} {wr_160:>6.0%} {n_160:>5}" if not np.isnan(avg_160) else f" | {'—':>8} {'—':>7} {'—':>5}"
        line += f" | {diff:>+6.1%} {winner:>7}"
        print(line)

    if len(paired_diffs) >= 5:
        paired_diffs = np.array(paired_diffs)
        n_152_wins = np.sum(paired_diffs > 0.001)
        n_160_wins = np.sum(paired_diffs < -0.001)
        try:
            stat, p_val = sp_stats.wilcoxon(paired_diffs)
        except ValueError:
            stat, p_val = 0, 1.0
        _binom_fn = getattr(sp_stats, 'binomtest', None) or getattr(sp_stats, 'binom_test', None)
        sign_test_p = _binom_fn(n_152_wins, n_152_wins + n_160_wins, 0.5) if (n_152_wins + n_160_wins) > 0 else 1.0
        if hasattr(sign_test_p, 'pvalue'):
            sign_test_p = sign_test_p.pvalue

        print(f"\n  Paired comparison ({len(paired_diffs)} tickers):")
        print(f"    152 wins: {n_152_wins}  |  160 wins: {n_160_wins}  |  ties: {len(paired_diffs) - n_152_wins - n_160_wins}")
        print(f"    Mean diff (152 - 160): {np.mean(paired_diffs):+.2%}")
        print(f"    Wilcoxon p-value: {p_val:.4f}")
        print(f"    Sign test p-value: {sign_test_p:.4f}")

        if np.mean(paired_diffs) > 0.005 and p_val < 0.10:
            t3_verdict = "SUPPORTS H1 — fresher model (152) significantly better"
            t3_score = 4 if p_val < 0.05 else 3
        elif abs(np.mean(paired_diffs)) < 0.005:
            t3_verdict = "AGAINST H1 — stale (160) and fresh (152) perform similarly"
            t3_score = 0
        else:
            t3_verdict = "INCONCLUSIVE — trend but not significant"
            t3_score = 1
        print(f"  → {t3_verdict}")
    else:
        t3_verdict = "INCONCLUSIVE (too few paired tickers)"
        t3_score = 1
        print(f"  → {t3_verdict}")
else:
    t3_verdict = "NO DATA (need both 152 and 160 in BQ)"
    t3_score = 0
    print(f"  Missing one or both families in BQ data.")

# ==============================================================================
# TABLE 4: N-SENSITIVITY — Does Trade Count Predict Degradation?
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 4: N-SENSITIVITY — Does Trade Count Predict Degradation?")
print(f"  If degradation inversely correlates with N → small N causes overfitting")
print(f"{'='*W}")

surv_df = df[df['survived_a'] & (df['val_a_count'] > 0) & (df['val_b_count'] > 0)].copy()
surv_df['degradation'] = 1 - surv_df['val_b_avg_pnl'] / surv_df['val_a_avg_pnl'].clip(lower=0.001)

if len(surv_df) >= 10:
    n_bins = [(3, 5, '3-5'), (6, 10, '6-10'), (11, 20, '11-20'), (21, 999, '21+')]
    print(f"\n  {'N range':<8} {'Seeds':>6} {'Avg Degrad':>11} {'Med Degrad':>11} {'%>100%':>7} {'%<0%':>7} {'Avg OPT':>8} {'Avg TEST':>8}")
    print(f"  {'─'*8} {'─'*6} {'─'*11} {'─'*11} {'─'*7} {'─'*7} {'─'*8} {'─'*8}")

    for lo, hi, label in n_bins:
        bin_df = surv_df[(surv_df['val_a_count'] >= lo) & (surv_df['val_a_count'] <= hi)]
        if bin_df.empty:
            continue
        dg = bin_df['degradation'].values
        dg_clipped = np.clip(dg, -5, 5)
        pct_over100 = np.mean(dg > 1.0) * 100
        pct_neg = np.mean(dg < 0) * 100
        avg_opt = bin_df['val_a_avg_pnl'].mean()
        avg_test = bin_df['val_b_avg_pnl'].mean()
        print(f"  {label:<8} {len(bin_df):>6} {np.mean(dg_clipped):>10.0%} "
              f"{np.median(dg_clipped):>10.0%} {pct_over100:>6.0f}% {pct_neg:>6.0f}% "
              f"{avg_opt:>7.1%} {avg_test:>7.1%}")

    rho, rho_p = sp_stats.spearmanr(surv_df['val_a_count'], surv_df['degradation'])
    print(f"\n  Spearman correlation (N vs degradation): rho={rho:+.3f}  p={rho_p:.4f}")
    print(f"  (Negative rho = more trades → less degradation = small N IS the problem)")

    if rho < -0.15 and rho_p < 0.05:
        t4_verdict = "SUPPORTS H2 — more trades → less degradation"
        t4_score = 4 if rho < -0.25 else 3
    elif rho < -0.10 and rho_p < 0.15:
        t4_verdict = "WEAKLY SUPPORTS H2"
        t4_score = 2
    elif abs(rho) < 0.10:
        t4_verdict = "AGAINST H2 — degradation is independent of trade count"
        t4_score = 0
    else:
        t4_verdict = "INCONCLUSIVE"
        t4_score = 1
    print(f"  → {t4_verdict}")

    print(f"\n  Per-family breakdown:")
    for fam in families_present:
        fdf = surv_df[surv_df['family'] == fam]
        if len(fdf) >= 5:
            r, p = sp_stats.spearmanr(fdf['val_a_count'], fdf['degradation'])
            print(f"    {FAMILY_LABELS[fam]:<15}: rho={r:+.3f}  p={p:.3f}  "
                  f"(avg N={fdf['val_a_count'].mean():.1f}, avg degrad={fdf['degradation'].median():.0%})")
else:
    t4_verdict = "INCONCLUSIVE (too few survived seeds)"
    t4_score = 0
    print(f"  Only {len(surv_df)} survived seeds — insufficient for analysis.")

# ==============================================================================
# TABLE 5: SEED AGREEMENT ANALYSIS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 5: SEED AGREEMENT — How many seeds agree on TEST direction?")
print(f"  High agreement (>80%) = robust signal. Near 50% = noise / overfitting.")
print(f"{'='*W}")

hdr5 = f"  {'Ticker':<7}"
for fam in families_present:
    fl = FAMILY_LABELS[fam][:6]
    hdr5 += f" | {fl+' +':>5} {fl+' -':>5} {fl+' %':>5} {'p':>6}"
hdr5 += f" | {'Diagnosis':>12}"
print(hdr5)
sep5 = f"  {'─'*7}"
for _ in families_present:
    sep5 += f" ─ {'─'*5} {'─'*5} {'─'*5} {'─'*6}"
sep5 += f" ─ {'─'*12}"
print(sep5)

agreement_rates = []
noise_tickers = 0
signal_tickers = 0

for ticker in all_tickers:
    td = df[df['ticker'] == ticker]
    line = f"  {ticker:<7}"
    ticker_max_agree = 0

    for fam in families_present:
        fd = td[(td['family'] == fam) & td['survived_a']]
        if fd.empty:
            line += f" | {'—':>5} {'—':>5} {'—':>5} {'—':>6}"
            continue

        n_pos = int((fd['val_b_avg_pnl'] > 0).sum())
        n_neg = int((fd['val_b_avg_pnl'] <= 0).sum())
        n_total = n_pos + n_neg
        agree_pct = max(n_pos, n_neg) / n_total if n_total > 0 else 0
        _btest = getattr(sp_stats, 'binomtest', None) or getattr(sp_stats, 'binom_test', None)
        p_binom = _btest(max(n_pos, n_neg), n_total, 0.5) if n_total > 0 else 1.0
        if hasattr(p_binom, 'pvalue'):
            p_binom = p_binom.pvalue
        ticker_max_agree = max(ticker_max_agree, agree_pct)

        line += f" | {n_pos:>5} {n_neg:>5} {agree_pct:>4.0%} {p_binom:>5.3f}"
        agreement_rates.append(agree_pct)

    if ticker_max_agree >= 0.80:
        diagnosis = "SIGNAL"
        signal_tickers += 1
    elif ticker_max_agree >= 0.60:
        diagnosis = "WEAK"
        signal_tickers += 1
    else:
        diagnosis = "NOISE"
        noise_tickers += 1

    line += f" | {diagnosis:>12}"
    print(line)

print(f"\n  Summary: {signal_tickers} tickers with signal (≥60% agreement), "
      f"{noise_tickers} tickers with noise (<60%)")
if agreement_rates:
    avg_agree = np.mean(agreement_rates)
    print(f"  Average agreement rate: {avg_agree:.0%}")
    if avg_agree < 0.65:
        t5_verdict = "SUPPORTS H2 — seeds mostly disagree, optimizer fitting noise"
        t5_score = 4
    elif avg_agree < 0.75:
        t5_verdict = "WEAKLY SUPPORTS H2 — moderate seed disagreement"
        t5_score = 2
    else:
        t5_verdict = "AGAINST H2 — seeds largely agree, signal is robust"
        t5_score = 0
    print(f"  → {t5_verdict}")
else:
    t5_verdict = "INCONCLUSIVE"
    t5_score = 0

# ==============================================================================
# TABLE 6: DEGRADATION DECOMPOSITION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 6: DEGRADATION DECOMPOSITION — What breaks between OPT and TEST?")
print(f"  Frequency decay → regime (model can't find spikes)")
print(f"  Win rate decay → overfitting (model signals but wrong)")
print(f"  Magnitude shift → market microstructure change")
print(f"{'='*W}")

hdr6 = f"  {'Family':<15} {'Freq Ratio':>10} {'WR Ratio':>10} {'Mag Ratio':>10} {'Dominant':>12}"
print(hdr6)
print(f"  {'─'*15} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")

decomp_results = {}
for fam in families_present:
    fd = surv_df[surv_df['family'] == fam]
    if fd.empty:
        continue

    freq_ratio = fd['val_b_freq'].mean() / fd['val_a_freq'].mean() if fd['val_a_freq'].mean() > 0 else 1.0

    wr_opt = fd['val_a_win_rate'].mean()
    wr_test = fd['val_b_win_rate'].mean()
    wr_ratio = wr_test / wr_opt if wr_opt > 0 else 1.0

    pnl_per_trade_opt = fd['val_a_avg_pnl'].mean()
    pnl_per_trade_test = fd['val_b_avg_pnl'].mean()
    if wr_opt > 0 and wr_test > 0 and pnl_per_trade_opt != 0:
        mag_ratio = (pnl_per_trade_test / wr_test) / (pnl_per_trade_opt / wr_opt)
    else:
        mag_ratio = 1.0

    deviations = {
        'Frequency': abs(1 - freq_ratio),
        'Win Rate': abs(1 - wr_ratio),
        'Magnitude': abs(1 - mag_ratio),
    }
    dominant = max(deviations, key=deviations.get)
    decomp_results[fam] = {'freq': freq_ratio, 'wr': wr_ratio, 'mag': mag_ratio, 'dominant': dominant}

    print(f"  {FAMILY_LABELS[fam]:<15} {freq_ratio:>9.2f} {wr_ratio:>9.2f} {mag_ratio:>9.2f} {dominant:>12}")

print(f"\n  Interpretation:")
print(f"    Ratio = 1.0: no change.  <1.0: TEST worse.  >1.0: TEST better.")
if decomp_results:
    dominants = [v['dominant'] for v in decomp_results.values()]
    freq_dom = dominants.count('Frequency')
    wr_dom = dominants.count('Win Rate')
    mag_dom = dominants.count('Magnitude')
    if freq_dom > wr_dom:
        t6_verdict = "Frequency decay dominates → SUPPORTS H1 (regime: fewer spikes to find)"
        t6_score_h1 = 3
        t6_score_h2 = 0
    elif wr_dom > freq_dom:
        t6_verdict = "Win rate decay dominates → SUPPORTS H2 (overfitting: wrong signals)"
        t6_score_h1 = 0
        t6_score_h2 = 3
    else:
        t6_verdict = "Mixed — no single dominant factor"
        t6_score_h1 = 1
        t6_score_h2 = 1
    print(f"  → {t6_verdict}")
else:
    t6_verdict = "INCONCLUSIVE"
    t6_score_h1 = 0
    t6_score_h2 = 0

# ==============================================================================
# TABLE 7: BOOTSTRAP SMALL-N SIMULATION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 7: BOOTSTRAP SMALL-N SIMULATION")
print(f"  How much noise does small N introduce? Simulating 10,000 draws.")
print(f"{'='*W}")

true_wr = 0.70
true_avg_win = 0.035
true_avg_loss = -0.015
n_sims = 10_000
n_values = [3, 5, 8, 10, 15, 20, 30, 50]

print(f"  Assumed true: WR={true_wr:.0%}, avg_win={true_avg_win:.1%}, avg_loss={true_avg_loss:.1%}")
print(f"  True avg PnL = {true_wr * true_avg_win + (1-true_wr) * true_avg_loss:.2%}")
print()
print(f"  {'N':>4} {'Avg PnL':>8} {'Std PnL':>8} {'CoV':>6} {'P(>1.8%)':>9} {'P(<0%)':>8} {'P(neg WR)':>10}")
print(f"  {'─'*4} {'─'*8} {'─'*8} {'─'*6} {'─'*9} {'─'*8} {'─'*10}")

sim_covs = {}
for n in n_values:
    sim_pnls = []
    sim_wrs = []
    for _ in range(n_sims):
        wins = np.random.binomial(n, true_wr)
        losses = n - wins
        if n == 0:
            continue
        pnl_trades = np.concatenate([
            np.random.normal(true_avg_win, 0.01, wins),
            np.random.normal(true_avg_loss, 0.008, losses),
        ])
        np.random.shuffle(pnl_trades)
        sim_pnls.append(np.mean(pnl_trades))
        sim_wrs.append(wins / n)

    sim_pnls = np.array(sim_pnls)
    sim_wrs = np.array(sim_wrs)
    avg = np.mean(sim_pnls)
    std = np.std(sim_pnls)
    cov = std / abs(avg) if abs(avg) > 1e-6 else 99.9
    p_above = np.mean(sim_pnls > 0.018)
    p_neg = np.mean(sim_pnls < 0)
    p_neg_wr = np.mean(sim_wrs < 0.5)
    sim_covs[n] = cov

    print(f"  {n:>4} {avg:>7.2%} {std:>7.2%} {cov:>5.1f} {p_above:>8.0%} {p_neg:>7.0%} {p_neg_wr:>9.0%}")

if surv_df is not None and len(surv_df) > 0:
    actual_n = surv_df['val_a_count'].median()
    actual_cov_data = []
    for (ticker, fam), grp in surv_df.groupby(['ticker', 'family']):
        if len(grp) >= 3:
            test_pnls = grp['val_b_avg_pnl'].values
            mean_pnl = np.mean(test_pnls)
            if abs(mean_pnl) > 1e-6:
                actual_cov_data.append(np.std(test_pnls) / abs(mean_pnl))

    if actual_cov_data:
        actual_cov = np.median(actual_cov_data)
        closest_n = min(sim_covs, key=lambda n: abs(sim_covs[n] - actual_cov))
        print(f"\n  Actual observed median CoV across seeds: {actual_cov:.2f}")
        print(f"  Actual median OPT trade count: {actual_n:.0f}")
        print(f"  Simulated CoV at N={actual_n:.0f}: {sim_covs.get(int(actual_n), sim_covs[closest_n]):.2f}")
        print(f"  Closest simulated match: N={closest_n} (CoV={sim_covs[closest_n]:.2f})")

        if actual_cov > sim_covs.get(int(actual_n), sim_covs[closest_n]) * 0.8:
            t7_verdict = "SUPPORTS H2 — observed variance matches small-N noise expectations"
            t7_score = 4
        else:
            t7_verdict = "AGAINST H2 — variance lower than expected from N alone (other factors at play)"
            t7_score = 0
        print(f"  → {t7_verdict}")
    else:
        t7_verdict = "INCONCLUSIVE"
        t7_score = 0
else:
    t7_verdict = "INCONCLUSIVE"
    t7_score = 0

# ==============================================================================
# TABLE 8: DIAGNOSIS SUMMARY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 8: DIAGNOSIS SUMMARY — Root Cause Attribution")
print(f"{'='*W}")

h1_evidence = {
    'T1 Spike rate trends': (t1_score, t1_verdict),
    'T2 WF fold trend': (t2_score, t2_verdict),
    'T3 Staleness test': (t3_score, t3_verdict),
    'T6 Freq decay (decomp)': (t6_score_h1 if 't6_score_h1' in dir() else 0,
                                t6_verdict if 't6_verdict' in dir() else 'N/A'),
}

h2_evidence = {
    'T4 N-sensitivity': (t4_score, t4_verdict),
    'T5 Seed agreement': (t5_score, t5_verdict),
    'T7 Bootstrap match': (t7_score, t7_verdict),
    'T6 WR decay (decomp)': (t6_score_h2 if 't6_score_h2' in dir() else 0,
                              t6_verdict if 't6_verdict' in dir() else 'N/A'),
}

print(f"\n  H1: REGIME CHANGE (spike rates decaying, model can't adapt)")
print(f"  {'Test':<25} {'Score':>5} {'Verdict'}")
print(f"  {'─'*25} {'─'*5} {'─'*60}")
h1_total = 0
for test_name, (score, verdict) in h1_evidence.items():
    h1_total += score
    bar = '*' * score + '.' * (5 - score)
    print(f"  {test_name:<25} [{bar}] {verdict}")

print(f"\n  H2: SMALL-N VALIDATION NOISE (optimizer picks flukes)")
print(f"  {'Test':<25} {'Score':>5} {'Verdict'}")
print(f"  {'─'*25} {'─'*5} {'─'*60}")
h2_total = 0
for test_name, (score, verdict) in h2_evidence.items():
    h2_total += score
    bar = '*' * score + '.' * (5 - score)
    print(f"  {test_name:<25} [{bar}] {verdict}")

h1_max = len(h1_evidence) * 5
h2_max = len(h2_evidence) * 5

print(f"\n  {'─'*W}")
print(f"  H1 (Regime Change):    {h1_total:>2}/{h1_max}  ({h1_total/h1_max:.0%})")
print(f"  H2 (Small-N Noise):    {h2_total:>2}/{h2_max}  ({h2_total/h2_max:.0%})")

if h1_total > h2_total * 1.5:
    conclusion = ("REGIME CHANGE is the dominant root cause. "
                  "Recommended: shorter retraining cycles (walk-forward), "
                  "lower spike thresholds to capture 6-8% moves.")
elif h2_total > h1_total * 1.5:
    conclusion = ("SMALL-N NOISE is the dominant root cause. "
                  "Recommended: aggregate across more seeds, "
                  "require higher min trade counts in WF optimizer, "
                  "consider ensemble prob thresholds.")
elif h1_total > 0 and h2_total > 0:
    conclusion = ("BOTH FACTORS CONTRIBUTE. Regime is shifting AND optimizer N is too small. "
                  "Recommended: walk-forward retraining (addresses H1) with "
                  "minimum 10+ OPT trades and ensemble seeds (addresses H2).")
else:
    conclusion = ("INCONCLUSIVE — insufficient data to diagnose. "
                  "Run WF optimizer on v2 models and re-analyze.")

print(f"\n  CONCLUSION: {conclusion}")
print(f"\n{'='*W}")
