"""
analyze_predictive_decay.py — Model Freshness & Retrain Cadence Analysis
=========================================================================
Measures how quickly prediction quality degrades after the training cutoff
using raw scoring CSVs from exp 160 and 170-175 walkforward folds.

Answers two questions:
  1. How often should models be retrained?
  2. What's the optimal validation-test split?

Tables:
  1: Decay Curve — Precision vs weeks since cutoff (per fold)
  2: Cross-Fold Comparison — Stale vs fresh model at same calendar period
  3: Half-Life — Weeks until precision drops 50% from initial
  4: PnL Decay — Simulated trading returns by time bucket
  5: Retrain Cadence Recommendation
  6: Optimal Val-Test Split

Run in Colab: paste as cell, runs immediately.
"""

import gc
import ctypes
import io
import pandas as pd
import numpy as np
from scipy import stats as sp_stats

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import storage

def _release_memory():
    gc.collect()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"
SPIKE_THRESHOLD = 0.10

FOLDS = {
    '160_XSECTOR': {'cutoff': '2025-03-01', 'seeds': list(range(16001, 16016)),
                     'family': '160', 'label': '160 (Mar25)', 'order': 0},
    '170_XSECTOR': {'cutoff': '2025-03-01', 'seeds': list(range(17001, 17016)),
                     'family': '170', 'label': 'F1 (Mar25)', 'order': 1},
    '171_XSECTOR': {'cutoff': '2025-05-01', 'seeds': list(range(17101, 17116)),
                     'family': '170', 'label': 'F2 (May25)', 'order': 2},
    '172_XSECTOR': {'cutoff': '2025-07-01', 'seeds': list(range(17201, 17216)),
                     'family': '170', 'label': 'F3 (Jul25)', 'order': 3},
    '173_XSECTOR': {'cutoff': '2025-09-01', 'seeds': list(range(17301, 17316)),
                     'family': '170', 'label': 'F4 (Sep25)', 'order': 4},
    '174_XSECTOR': {'cutoff': '2025-11-01', 'seeds': list(range(17401, 17416)),
                     'family': '170', 'label': 'F5 (Nov25)', 'order': 5},
    '175_XSECTOR': {'cutoff': '2026-01-01', 'seeds': list(range(17501, 17516)),
                     'family': '170', 'label': 'F6 (Jan26)', 'order': 6},
}

ANALYSIS_TICKERS = [
    'LUNR', 'BBAI', 'WULF', 'CIFR', 'IREN', 'CLSK', 'HUT', 'MARA', 'RIOT',
    'RCAT', 'ASTS', 'KEEL', 'APLD', 'PLUG', 'FCEL', 'BE', 'BTBT', 'ALAB',
]

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

PROB_THRESHOLD = 0.50
PROB_THRESHOLDS = [0.30, 0.50, 0.70]
BUCKET_DAYS = 14
MAX_WEEKS = 30

W = 120

# ==============================================================================
# DATA LOADING
# ==============================================================================
gcs_client = storage.Client()
gcs_bucket = gcs_client.bucket(BUCKET_NAME)

def load_fold_ensemble(fold_id, tickers):
    info = FOLDS[fold_id]
    cutoff = pd.Timestamp(info['cutoff']).tz_localize('UTC')
    results_dir = f"Algos/Results_all_v2/Universal_Exp{fold_id}"

    all_rows = []
    for ticker in tickers:
        clean = TICKER_FILE_MAP.get(ticker, ticker)
        prefix = f"{results_dir}/{clean}/Seed_"
        blobs = list(gcs_bucket.list_blobs(prefix=prefix))

        seed_dfs = []
        for blob in blobs:
            name = blob.name.split('/')[-1]
            if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
                continue
            seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
            if seed_id not in info['seeds']:
                continue
            try:
                df = pd.read_csv(io.BytesIO(blob.download_as_bytes()),
                                 index_col=0, parse_dates=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                seed_dfs.append(df[['Prob_Spike', 'Target']].rename(
                    columns={'Prob_Spike': f'p_{seed_id}'}))
            except Exception:
                continue

        if not seed_dfs:
            continue

        merged = seed_dfs[0]
        for sdf in seed_dfs[1:]:
            merged = merged.join(sdf[[c for c in sdf.columns if c.startswith('p_')]],
                                 how='outer')

        prob_cols = [c for c in merged.columns if c.startswith('p_')]
        merged['Prob_Ensemble'] = merged[prob_cols].mean(axis=1)
        merged['Target'] = merged['Target'].fillna(0).astype(int)
        merged['Ticker'] = ticker
        merged['N_Seeds'] = merged[prob_cols].notna().sum(axis=1)

        post_cutoff = merged[merged.index > cutoff].copy()
        if post_cutoff.empty:
            continue

        post_cutoff['Days_Since'] = (post_cutoff.index - cutoff).days
        post_cutoff['Bucket'] = post_cutoff['Days_Since'] // BUCKET_DAYS

        all_rows.append(post_cutoff[['Prob_Ensemble', 'Target', 'Ticker',
                                      'N_Seeds', 'Days_Since', 'Bucket']])
        del merged, seed_dfs
        _release_memory()

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows)
    del all_rows
    _release_memory()
    return result


# ==============================================================================
# LOAD ALL FOLDS
# ==============================================================================
print(f"{'='*W}")
print(f"  PREDICTIVE POWER DECAY ANALYSIS")
print(f"  Folds: {len(FOLDS)}  Tickers: {len(ANALYSIS_TICKERS)}  Bucket: {BUCKET_DAYS} days")
print(f"{'='*W}")

fold_data = {}
for fold_id, info in FOLDS.items():
    print(f"\n  Loading {info['label']} ({fold_id})...")
    df = load_fold_ensemble(fold_id, ANALYSIS_TICKERS)
    if df.empty:
        print(f"    SKIP — no data found")
        continue
    n_tickers = df['Ticker'].nunique()
    n_bars = len(df)
    max_days = df['Days_Since'].max()
    print(f"    {n_tickers} tickers, {n_bars:,} bars, {max_days} days of OOS data")
    fold_data[fold_id] = df
    _release_memory()

if not fold_data:
    print("\n  No data loaded. Check GCS paths.")
    raise SystemExit(0)

print(f"\n  Loaded {len(fold_data)} folds successfully.")


# ==============================================================================
# TABLE 1: DECAY CURVE — Precision vs Weeks Since Cutoff
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: PRECISION DECAY CURVE")
print(f"  Ensemble precision (prob > {PROB_THRESHOLD:.0%}) at each {BUCKET_DAYS}-day bucket")
print(f"{'='*W}")

fold_labels = sorted(fold_data.keys(), key=lambda f: FOLDS[f]['order'])
decay_curves = {}

hdr = f"  {'Bucket':>6} {'Days':>6}"
for fid in fold_labels:
    hdr += f" {FOLDS[fid]['label']:>12}"
print(hdr)
print(f"  {'─'*6} {'─'*6}" + f" {'─'*12}" * len(fold_labels))

for bucket_idx in range(MAX_WEEKS):
    line = f"  {bucket_idx:>6} {bucket_idx * BUCKET_DAYS:>5}d"
    has_data = False
    for fid in fold_labels:
        df = fold_data[fid]
        bkt = df[df['Bucket'] == bucket_idx]
        signals = bkt[bkt['Prob_Ensemble'] > PROB_THRESHOLD]
        if len(signals) >= 5:
            prec = signals['Target'].mean()
            n_sig = len(signals)
            line += f" {prec:>5.0%} ({n_sig:>4})"
            if fid not in decay_curves:
                decay_curves[fid] = []
            decay_curves[fid].append({'bucket': bucket_idx, 'days': bucket_idx * BUCKET_DAYS,
                                       'precision': prec, 'n_signals': n_sig})
            has_data = True
        else:
            line += f" {'—':>12}"
    if has_data:
        print(line)

print(f"\n  Decay Rate (linear regression of precision vs bucket):")
print(f"  {'Fold':<14} {'Slope/bucket':>12} {'R²':>6} {'p-val':>7} {'Trend':>8}")
print(f"  {'─'*14} {'─'*12} {'─'*6} {'─'*7} {'─'*8}")

fold_slopes = {}
for fid in fold_labels:
    if fid not in decay_curves or len(decay_curves[fid]) < 3:
        continue
    dc = decay_curves[fid]
    x = np.array([d['bucket'] for d in dc])
    y = np.array([d['precision'] for d in dc])
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(x, y)
    trend = ("DECAY" if slope < -0.005 and p_val < 0.15 else
             "RISING" if slope > 0.005 and p_val < 0.15 else "STABLE")
    fold_slopes[fid] = {'slope': slope, 'intercept': intercept, 'r2': r_val**2,
                         'p_val': p_val, 'trend': trend}
    print(f"  {FOLDS[fid]['label']:<14} {slope*100:>+10.2f}pp {r_val**2:>5.3f} {p_val:>6.3f} {trend:>8}")


# ==============================================================================
# TABLE 2: CROSS-FOLD COMPARISON — Stale vs Fresh at Same Calendar Period
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 2: STALE vs FRESH MODEL — Same Calendar Period")
print(f"  For overlapping calendar months, compare a 'stale' model (8+ weeks")
print(f"  since cutoff) against a 'fresh' model (0-4 weeks since cutoff)")
print(f"{'='*W}")

wf_folds = [f for f in fold_labels if FOLDS[f]['family'] == '170']

comparisons = []
if len(wf_folds) >= 2:
    print(f"\n  {'Calendar':>10} {'Stale Fold':<14} {'Stale Prec':>10} {'Fresh Fold':<14} "
          f"{'Fresh Prec':>10} {'Delta':>7} {'Winner':>7}")
    print(f"  {'─'*10} {'─'*14} {'─'*10} {'─'*14} {'─'*10} {'─'*7} {'─'*7}")

    for i in range(len(wf_folds) - 1):
        stale_fid = wf_folds[i]
        fresh_fid = wf_folds[i + 1]

        stale_cutoff = pd.Timestamp(FOLDS[stale_fid]['cutoff']).tz_localize('UTC')
        fresh_cutoff = pd.Timestamp(FOLDS[fresh_fid]['cutoff']).tz_localize('UTC')

        overlap_start = fresh_cutoff
        overlap_end = fresh_cutoff + pd.Timedelta(days=BUCKET_DAYS * 2)

        if stale_fid not in fold_data or fresh_fid not in fold_data:
            continue

        stale_df = fold_data[stale_fid]
        fresh_df = fold_data[fresh_fid]

        stale_overlap = stale_df[(stale_df.index >= overlap_start) &
                                  (stale_df.index < overlap_end)]
        fresh_overlap = fresh_df[(fresh_df.index >= overlap_start) &
                                  (fresh_df.index < overlap_end)]

        stale_signals = stale_overlap[stale_overlap['Prob_Ensemble'] > PROB_THRESHOLD]
        fresh_signals = fresh_overlap[fresh_overlap['Prob_Ensemble'] > PROB_THRESHOLD]

        if len(stale_signals) < 3 or len(fresh_signals) < 3:
            continue

        stale_prec = stale_signals['Target'].mean()
        fresh_prec = fresh_signals['Target'].mean()
        delta = fresh_prec - stale_prec
        winner = "FRESH" if delta > 0.02 else ("STALE" if delta < -0.02 else "TIE")

        cal_label = overlap_start.strftime('%Y-%m')
        stale_weeks = (overlap_start - stale_cutoff).days // 7
        fresh_weeks = 0

        comparisons.append({'stale_prec': stale_prec, 'fresh_prec': fresh_prec,
                            'delta': delta, 'winner': winner})

        print(f"  {cal_label:>10} {FOLDS[stale_fid]['label']:<14} "
              f"{stale_prec:>5.0%} (w{stale_weeks:>2}) "
              f"{FOLDS[fresh_fid]['label']:<14} {fresh_prec:>5.0%} (w0)  "
              f"{delta:>+6.0%} {winner:>7}")

    if comparisons:
        n_fresh = sum(1 for c in comparisons if c['winner'] == 'FRESH')
        n_stale = sum(1 for c in comparisons if c['winner'] == 'STALE')
        n_tie = sum(1 for c in comparisons if c['winner'] == 'TIE')
        avg_delta = np.mean([c['delta'] for c in comparisons])
        print(f"\n  VERDICT: Fresh wins {n_fresh}/{len(comparisons)}, "
              f"Stale wins {n_stale}, Ties {n_tie}")
        print(f"  Avg precision advantage for fresh: {avg_delta:+.1%}")
        if n_fresh > n_stale:
            print(f"  → Model staleness IS the dominant factor — retraining helps")
            staleness_dominant = True
        else:
            print(f"  → Regime change may dominate — retraining alone may not help")
            staleness_dominant = False
    else:
        print(f"  Not enough overlapping signals to compare.")
        staleness_dominant = None
else:
    print(f"  Need at least 2 walkforward folds for comparison.")
    staleness_dominant = None


# ==============================================================================
# TABLE 3: HALF-LIFE ANALYSIS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: HALF-LIFE — Weeks Until Precision Drops 50%")
print(f"{'='*W}")

half_lives = {}
initial_precs = {}

for fid in fold_labels:
    if fid not in decay_curves or len(decay_curves[fid]) < 3:
        continue
    dc = decay_curves[fid]
    initial = dc[0]['precision']
    initial_precs[fid] = initial
    half_target = initial * 0.5
    half_life = None

    for d in dc[1:]:
        if d['precision'] <= half_target:
            half_life = d['days']
            break

    half_lives[fid] = half_life

print(f"\n  {'Fold':<14} {'Initial Prec':>12} {'Half-Life':>10} {'Status':>10}")
print(f"  {'─'*14} {'─'*12} {'─'*10} {'─'*10}")

for fid in fold_labels:
    if fid not in initial_precs:
        continue
    init = initial_precs[fid]
    hl = half_lives.get(fid)
    if hl is not None:
        status = f"{hl}d"
        print(f"  {FOLDS[fid]['label']:<14} {init:>11.0%} {hl:>8}d  {'MEASURED':>10}")
    else:
        last_bucket = decay_curves[fid][-1]['days'] if fid in decay_curves else 0
        last_prec = decay_curves[fid][-1]['precision'] if fid in decay_curves else 0
        print(f"  {FOLDS[fid]['label']:<14} {init:>11.0%} {'>'+str(last_bucket)+'d':>9}  "
              f"{'STILL OK':>10} (last={last_prec:.0%})")

measured_hls = [v for v in half_lives.values() if v is not None]
if measured_hls:
    avg_hl = np.mean(measured_hls)
    med_hl = np.median(measured_hls)
    print(f"\n  Average half-life: {avg_hl:.0f} days ({avg_hl/7:.1f} weeks)")
    print(f"  Median half-life:  {med_hl:.0f} days ({med_hl/7:.1f} weeks)")
else:
    avg_hl = None
    print(f"\n  No half-life measured — precision may not drop 50% within available data")

# Per-ticker half-life
print(f"\n  Per-Ticker Decay Speed (using fold with most data):")
print(f"  {'Ticker':<7} {'Init Prec':>9} {'Final Prec':>10} {'Decay%':>7} {'Category':>10}")
print(f"  {'─'*7} {'─'*9} {'─'*10} {'─'*7} {'─'*10}")

longest_fold = max(fold_data.keys(), key=lambda f: len(fold_data[f]))
lf_df = fold_data[longest_fold]
lf_cutoff = pd.Timestamp(FOLDS[longest_fold]['cutoff']).tz_localize('UTC')

for ticker in sorted(ANALYSIS_TICKERS):
    tdf = lf_df[lf_df['Ticker'] == ticker]
    if tdf.empty:
        continue
    early = tdf[tdf['Bucket'] <= 2]
    late = tdf[tdf['Bucket'] >= (tdf['Bucket'].max() - 2)]

    e_sig = early[early['Prob_Ensemble'] > PROB_THRESHOLD]
    l_sig = late[late['Prob_Ensemble'] > PROB_THRESHOLD]

    if len(e_sig) < 3 or len(l_sig) < 3:
        continue

    e_prec = e_sig['Target'].mean()
    l_prec = l_sig['Target'].mean()
    decay_pct = (1 - l_prec / e_prec) * 100 if e_prec > 0 else 0

    cat = ("FAST" if decay_pct > 50 else
           "MODERATE" if decay_pct > 25 else
           "SLOW" if decay_pct > 0 else "STABLE")

    print(f"  {ticker:<7} {e_prec:>8.0%} {l_prec:>9.0%} {decay_pct:>+6.0f}% {cat:>10}")


# ==============================================================================
# TABLE 4: PnL DECAY — Simulated Returns by Time Bucket
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 4: SIGNAL QUALITY DECAY — Hit Rate & Avg Return by Time Bucket")
print(f"  Using ensemble prob > {PROB_THRESHOLD:.0%} signals, Target=1 = spike happened")
print(f"{'='*W}")

for fid in fold_labels:
    if fid not in fold_data:
        continue
    df = fold_data[fid]
    info = FOLDS[fid]

    buckets = sorted(df['Bucket'].unique())
    buckets = [b for b in buckets if b <= MAX_WEEKS]

    print(f"\n  {info['label']} (cutoff {info['cutoff']}):")
    print(f"  {'Bucket':>6} {'Days':>5} {'Signals':>8} {'Hits':>5} {'Prec':>6} "
          f"{'Sig Rate':>8} {'Bars':>6}")
    print(f"  {'─'*6} {'─'*5} {'─'*8} {'─'*5} {'─'*6} {'─'*8} {'─'*6}")

    for b in buckets:
        bkt = df[df['Bucket'] == b]
        signals = bkt[bkt['Prob_Ensemble'] > PROB_THRESHOLD]
        n_sig = len(signals)
        n_bars = len(bkt)
        if n_sig < 2:
            continue
        hits = signals['Target'].sum()
        prec = signals['Target'].mean()
        sig_rate = n_sig / n_bars if n_bars > 0 else 0

        print(f"  {b:>6} {b*BUCKET_DAYS:>4}d {n_sig:>8} {int(hits):>5} {prec:>5.0%} "
              f"{sig_rate:>7.1%} {n_bars:>6}")


# ==============================================================================
# TABLE 5: RETRAIN CADENCE RECOMMENDATION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 5: RETRAIN CADENCE RECOMMENDATION")
print(f"{'='*W}")

if measured_hls:
    hl_weeks = avg_hl / 7
    if hl_weeks < 4:
        cadence = "2-3 weeks"
        cadence_days = 14
    elif hl_weeks < 8:
        cadence = "monthly (4 weeks)"
        cadence_days = 28
    elif hl_weeks < 16:
        cadence = "every 2 months (8 weeks)"
        cadence_days = 56
    else:
        cadence = "quarterly (12 weeks)"
        cadence_days = 84

    print(f"\n  Average model half-life: {avg_hl:.0f} days ({hl_weeks:.1f} weeks)")
    print(f"  Recommended retrain cadence: {cadence}")
    print(f"")
    print(f"  Reasoning:")
    print(f"    - Model precision drops 50% at ~{avg_hl:.0f} days post-cutoff")
    print(f"    - Deploy window should be BEFORE half-life point")
    print(f"    - Retrain at ~{cadence_days} days ensures model stays above 50% effectiveness")
else:
    print(f"\n  Could not measure half-life from available data.")
    print(f"  Examining decay slopes instead...")

    if fold_slopes:
        avg_slope = np.mean([s['slope'] for s in fold_slopes.values()])
        if abs(avg_slope) < 0.005:
            print(f"  Avg decay slope: {avg_slope*100:+.2f}pp/bucket — MINIMAL decay")
            print(f"  Recommended: retrain every 2-3 months (model stays predictive)")
            cadence_days = 70
        else:
            est_hl = -0.5 / avg_slope * BUCKET_DAYS if avg_slope < 0 else 999
            print(f"  Avg decay slope: {avg_slope*100:+.2f}pp/bucket")
            print(f"  Estimated half-life from slope: ~{est_hl:.0f} days")
            if est_hl < 30:
                cadence = "2-3 weeks"
                cadence_days = 14
            elif est_hl < 60:
                cadence = "monthly (4 weeks)"
                cadence_days = 28
            elif est_hl < 120:
                cadence = "every 2 months"
                cadence_days = 56
            else:
                cadence = "quarterly"
                cadence_days = 84
            print(f"  Recommended retrain cadence: {cadence}")
    else:
        print(f"  No decay slopes computed. Need more fold data.")
        cadence_days = 56

if staleness_dominant is True:
    print(f"\n  Cross-fold analysis confirms: model staleness IS the primary factor.")
    print(f"  Retraining on fresh data will improve prediction quality.")
elif staleness_dominant is False:
    print(f"\n  WARNING: Cross-fold analysis suggests regime change may dominate.")
    print(f"  Retraining helps but won't fully solve decay. Consider:")
    print(f"    - Adaptive thresholds that account for market regime")
    print(f"    - Regime-aware model features (VIX, correlation structure)")


# ==============================================================================
# TABLE 6: OPTIMAL VAL-TEST SPLIT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 6: OPTIMAL VALIDATION-TEST SPLIT")
print(f"{'='*W}")

MIN_TRADES_TARGET = 3

val_period = cadence_days // 2
test_period = cadence_days // 2
total_forward = val_period + test_period

print(f"\n  Based on retrain cadence of {cadence_days} days:")
print(f"")
print(f"  ┌──────────────────────────────────────────────────┐")
print(f"  │  TRAINING DATA    │  VAL ({val_period}d)  │  TEST ({test_period}d)  │  DEPLOY  │")
print(f"  │  (all history)    │  optimize  │  holdout   │  live    │")
print(f"  └──────────────────────────────────────────────────┘")
print(f"")
print(f"  Validation period: {val_period} days ({val_period // 7} weeks)")
print(f"    - Used for: WF optimization (grid search over prob_thresh, TP/SL, etc.)")
print(f"    - Model is fresh here — optimization results trustworthy")
print(f"")
print(f"  Test period: {test_period} days ({test_period // 7} weeks)")
print(f"    - Used for: Holdout verification (survived_a → survived_b)")
print(f"    - Model is {val_period}-{total_forward} days old — still within useful range")
print(f"")
print(f"  Total forward window: {total_forward} days ({total_forward // 7} weeks)")
print(f"    - Must be <= model half-life to ensure test results are meaningful")
print(f"")

if measured_hls:
    if total_forward > avg_hl:
        print(f"  WARNING: Total forward ({total_forward}d) exceeds half-life ({avg_hl:.0f}d)")
        print(f"  Consider shorter periods:")
        alt_val = int(avg_hl * 0.35)
        alt_test = int(avg_hl * 0.35)
        print(f"    Alternative: val={alt_val}d, test={alt_test}d, total={alt_val+alt_test}d")
    else:
        print(f"  Total forward ({total_forward}d) is within half-life ({avg_hl:.0f}d) — OK")

print(f"\n  Trade frequency check (target: {MIN_TRADES_TARGET} trades per ticker in val):")

for fid in fold_labels[:3]:
    if fid not in fold_data:
        continue
    df = fold_data[fid]
    val_buckets = val_period // BUCKET_DAYS
    val_df = df[df['Bucket'] < val_buckets]
    signals_per_ticker = val_df[val_df['Prob_Ensemble'] > PROB_THRESHOLD].groupby('Ticker').size()
    if signals_per_ticker.empty:
        continue
    med_trades = signals_per_ticker.median()
    pct_enough = (signals_per_ticker >= MIN_TRADES_TARGET).mean() * 100
    print(f"    {FOLDS[fid]['label']}: median {med_trades:.0f} signals/ticker in {val_period}d, "
          f"{pct_enough:.0f}% have >= {MIN_TRADES_TARGET}")


# ==============================================================================
# MULTI-THRESHOLD COMPARISON
# ==============================================================================
print(f"\n{'='*W}")
print(f"  APPENDIX: DECAY AT DIFFERENT PROBABILITY THRESHOLDS")
print(f"{'='*W}")

for thresh in PROB_THRESHOLDS:
    print(f"\n  Threshold: {thresh:.0%}")
    print(f"  {'Fold':<14} {'Week 0-1':>8} {'Week 4-5':>8} {'Week 8-9':>8} {'Week 12+':>8}")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for fid in fold_labels:
        if fid not in fold_data:
            continue
        df = fold_data[fid]

        windows = [
            (0, 2, 'w0-1'),
            (2, 4, 'w4-5'),
            (4, 6, 'w8-9'),
            (6, 99, 'w12+'),
        ]

        line = f"  {FOLDS[fid]['label']:<14}"
        for bkt_lo, bkt_hi, _ in windows:
            wdf = df[(df['Bucket'] >= bkt_lo) & (df['Bucket'] < bkt_hi)]
            sig = wdf[wdf['Prob_Ensemble'] > thresh]
            if len(sig) >= 3:
                prec = sig['Target'].mean()
                line += f" {prec:>5.0%}({len(sig):>2})"
            else:
                line += f" {'—':>8}"
        print(line)


# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SUMMARY — KEY FINDINGS")
print(f"{'='*W}")

print(f"\n  1. RETRAIN CADENCE")
if measured_hls:
    print(f"     Model half-life: ~{avg_hl:.0f} days ({avg_hl/7:.1f} weeks)")
    print(f"     Recommended: retrain every {cadence_days} days")
else:
    print(f"     Half-life not directly measured; using slope-based estimate")
    print(f"     Recommended: retrain every {cadence_days} days")

if staleness_dominant is True:
    print(f"     Fresh models consistently outperform stale ones at same calendar date")
elif staleness_dominant is False:
    print(f"     WARNING: Market regime change may also be a factor")

print(f"\n  2. OPTIMAL VAL-TEST SPLIT")
print(f"     Validation: {val_period} days ({val_period // 7} weeks)")
print(f"     Test:       {test_period} days ({test_period // 7} weeks)")
print(f"     Total forward window: {total_forward} days ({total_forward // 7} weeks)")

print(f"\n  3. COMPARISON: 160 (13 teachers) vs 170 (31 teachers)")
if '160_XSECTOR' in fold_slopes and any(f.startswith('17') for f in fold_slopes):
    s160 = fold_slopes.get('160_XSECTOR', {}).get('slope', 0)
    wf_slopes = [fold_slopes[f]['slope'] for f in fold_slopes if f.startswith('17')]
    s170_avg = np.mean(wf_slopes) if wf_slopes else 0
    print(f"     160 decay rate: {s160*100:+.2f}pp/bucket")
    print(f"     170 avg decay rate: {s170_avg*100:+.2f}pp/bucket")
    if abs(s160) > abs(s170_avg) * 1.2:
        print(f"     → 170 (more teachers) decays SLOWER — more data helps longevity")
    elif abs(s170_avg) > abs(s160) * 1.2:
        print(f"     → 160 decays slower — teacher count alone doesn't determine longevity")
    else:
        print(f"     → Similar decay rates — teacher count doesn't strongly affect longevity")
else:
    print(f"     Insufficient data to compare")

print(f"\n{'='*W}")
