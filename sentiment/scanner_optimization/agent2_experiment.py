# %% Agent 2: Expansion + Logging Experiment
# Research question: How to reliably surface 3 qualified LLM candidates every morning?
# Uses existing v2_features_full.csv (316 LLM trades, 4 dates)
# Experiments: B1 gap threshold, B2 catalyst inclusion, B3 scanner separation, B4 n_runs
# Output: AGENT_2_EXPANSION_RESULTS.md + agent2_*.png

import os, sys, json, warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

TRADING_DATES = ['2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11']
N_DATES = len(TRADING_DATES)

_W = 130


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def trim_mean(arr, prop=0.1):
    s = np.sort(arr)
    n = len(s)
    k = max(1, int(n * prop))
    if 2 * k >= n:
        return float(np.mean(s))
    return float(np.mean(s[k:-k]))


def full_metrics(pnl_list, n_dates=N_DATES):
    pnl = np.array([x for x in pnl_list if x is not None and np.isfinite(x)])
    n = len(pnl)
    empty = {'n': n, 'trades_per_day': round(n / n_dates, 1) if n_dates else 0}
    if n < 3:
        empty['note'] = 'insufficient'
        for k in ['avg_pnl_pct', 'median_pnl_pct', 'trimmed_mean_pct', 'std_pct',
                  'sharpe', 'win_rate', 'profit_factor', 'max_loss_pct', 'max_gain_pct',
                  'ci_95_low', 'ci_95_high', 'ci_confirmed', 'total_pnl_pct']:
            empty[k] = None
        if n > 0:
            empty['avg_pnl_pct'] = round(float(np.mean(pnl)), 3)
            empty['total_pnl_pct'] = round(float(np.sum(pnl)), 2)
        return empty

    avg = float(np.mean(pnl))
    med = float(np.median(pnl))
    std = float(np.std(pnl, ddof=1))
    se = std / np.sqrt(n)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    pf = float(winners.sum() / abs(losers.sum())) if losers.sum() != 0 else (
        float('inf') if len(winners) > 0 else 0.0)
    ci_lo = avg - 1.96 * se
    ci_hi = avg + 1.96 * se

    return {
        'n': n,
        'trades_per_day': round(n / n_dates, 1) if n_dates else 0,
        'avg_pnl_pct': round(avg, 3),
        'median_pnl_pct': round(med, 3),
        'trimmed_mean_pct': round(trim_mean(pnl, 0.1), 3),
        'std_pct': round(std, 3),
        'sharpe': round(avg / std, 3) if std > 0 else 0,
        'win_rate': round(float((pnl > 0).mean()), 3),
        'profit_factor': round(pf, 2) if pf < 100 else float('inf'),
        'max_loss_pct': round(float(np.min(pnl)), 3),
        'max_gain_pct': round(float(np.max(pnl)), 3),
        'ci_95_low': round(ci_lo, 3),
        'ci_95_high': round(ci_hi, 3),
        'ci_confirmed': bool(ci_lo > 0),
        'total_pnl_pct': round(float(np.sum(pnl)), 2),
    }


def metrics_for_subset(df, pnl_col, n_dates=N_DATES):
    vals = df[pnl_col].dropna().tolist()
    return full_metrics(vals, n_dates)


def leave_one_out_check(pnl_list):
    if len(pnl_list) < 5:
        return {'stable': None, 'reason': 'too few trades'}
    arr = np.array(pnl_list)
    base_avg = np.mean(arr)
    loo_avgs = []
    for i in range(len(arr)):
        loo = np.delete(arr, i)
        loo_avgs.append(np.mean(loo))
    loo_avgs = np.array(loo_avgs)
    min_loo = float(np.min(loo_avgs))
    max_loo = float(np.max(loo_avgs))
    sign_flip = (min_loo < 0 < base_avg) or (max_loo > 0 > base_avg)
    return {
        'stable': not sign_flip,
        'base_avg': round(base_avg, 3),
        'min_loo_avg': round(min_loo, 3),
        'max_loo_avg': round(max_loo, 3),
        'max_influence_pct': round(abs(max_loo - min_loo) / abs(base_avg) * 100, 1) if base_avg != 0 else None,
        'sign_flip': sign_flip,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  AGENT 2: EXPANSION + LOGGING EXPERIMENT")
print(f"  Data: v2_features_full.csv (316 LLM trades, 4 dates)")
print(f"{'='*_W}")

df = pd.read_csv(os.path.join(RESULTS_DIR, 'v2_features_full.csv'))
print(f"\n  Loaded {len(df)} trades")
print(f"  Dates: {sorted(df['eval_date'].unique())}")
print(f"  Directions: {df['direction'].value_counts().to_dict()}")
print(f"  Scanners: {df['scanner'].value_counts().to_dict()}")

# Normalize catalyst_type
df['catalyst_type_lower'] = df['catalyst_type'].fillna('unknown').str.lower()

# Flag columns
df['is_offering'] = df['catalyst_type_lower'].str.contains('offering', na=False)
df['is_deal_flag'] = df['catalyst_type_lower'].str.contains('deal|acqui|merger|buyout|takeover', na=False, regex=True)
df['is_macro_flag'] = df['catalyst_type_lower'].str.contains('macro', na=False)
df['is_breaking_flag'] = df['catalyst_type_lower'].str.contains('breaking', na=False)
df['is_regulatory'] = df['catalyst_type_lower'].str.contains('regulat', na=False)
df['is_insider'] = df['catalyst_type_lower'].str.contains('insider', na=False)
df['is_analyst'] = df['catalyst_type_lower'].str.contains('analyst', na=False)
df['is_premarket_move'] = df['catalyst_type_lower'].str.contains('premarket', na=False)

# Count per-date per-direction
for dt in TRADING_DATES:
    sub = df[df['eval_date'] == dt]
    n_short = (sub['direction'] == 'SHORT').sum()
    n_long = (sub['direction'] == 'LONG').sum()
    n_t2 = sub['pnl_t2_close'].notna().sum()
    print(f"    {dt}: {len(sub)} total ({n_short} SHORT, {n_long} LONG), {n_t2} with T+2 data")


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: Scanner Silence Analysis
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  DIAGNOSTIC: Scanner Silence Analysis")
print(f"{'='*_W}")

silence_diagnosis = {}

for dt in TRADING_DATES:
    sub = df[df['eval_date'] == dt]
    premarket = sub[sub['scanner'] == 'premarket']
    discovery = sub[sub['scanner'] == 'discovery']

    short_sub = sub[sub['direction'] == 'SHORT']
    short_gap_up = short_sub[short_sub['gap_pct'] > 0]
    short_gap_3 = short_gap_up[short_gap_up['abs_gap_pct'] >= 3.0]

    has_t2 = sub['pnl_t2_close'].notna().sum()

    # Catalyst breakdown for SHORT gap-ups
    cat_counts = {}
    if len(short_gap_up) > 0:
        for cat in short_gap_up['catalyst_type_lower'].unique():
            cat_counts[cat] = int((short_gap_up['catalyst_type_lower'] == cat).sum())

    diag = {
        'total': len(sub),
        'premarket': len(premarket),
        'discovery': len(discovery),
        'SHORT': len(short_sub),
        'SHORT_gap_up': len(short_gap_up),
        'SHORT_gap_up_gte3': len(short_gap_3),
        'has_t2_data': has_t2,
        'catalyst_counts': cat_counts,
    }
    silence_diagnosis[dt] = diag

    t2_flag = f"  *** NO T+2 DATA ***" if has_t2 == 0 else f"  ({has_t2} with T+2)"
    print(f"\n  {dt}:")
    print(f"    Total LLM signals:     {len(sub):>4} (PM: {len(premarket)}, Disc: {len(discovery)})")
    print(f"    SHORT signals:         {len(short_sub):>4}")
    print(f"    SHORT gap-up (any):    {len(short_gap_up):>4}")
    print(f"    SHORT gap-up >= 3%:    {len(short_gap_3):>4}")
    print(f"    Has T+2 outcome data:  {has_t2:>4}{t2_flag}")
    if cat_counts:
        print(f"    Catalysts (SHORT gap-up): {cat_counts}")

# Determine cause
total_signals = sum(d['total'] for d in silence_diagnosis.values())
dates_with_t2 = [dt for dt, d in silence_diagnosis.items() if d['has_t2_data'] > 0]
dates_without_t2 = [dt for dt, d in silence_diagnosis.items() if d['has_t2_data'] == 0]

print(f"\n  DIAGNOSIS:")
print(f"    Total LLM signals across 4 days: {total_signals}")
print(f"    Average per day: {total_signals/N_DATES:.0f}")
print(f"    Dates WITH T+2 data: {dates_with_t2}")
print(f"    Dates WITHOUT T+2 data: {dates_without_t2}")

if dates_without_t2:
    n_wo = sum(silence_diagnosis[dt]['total'] for dt in dates_without_t2)
    print(f"\n    CAUSE: T+2 data unavailable for {len(dates_without_t2)} dates")
    print(f"    Those dates have {n_wo} signals that WOULD produce trades in live")
    print(f"    Scanner IS running on all 4 days — this is a data availability issue, not operational")
    print(f"    Effective sample: only {len(dates_with_t2)} dates have measurable T+2 outcomes")

N_DATES_WITH_T2 = len(dates_with_t2)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B1: Gap Threshold Sweep
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  B1: Gap Threshold Sweep (LLM SHORT Universe)")
print(f"{'='*_W}")

# Filter to SHORT direction with positive gaps (gap-up shorts = fade)
short_df = df[(df['direction'] == 'SHORT') & (df['gap_pct'] > 0)].copy()
print(f"\n  SHORT gap-up universe: {len(short_df)} trades")
print(f"  With T+2 data: {short_df['pnl_t2_close'].notna().sum()}")

gap_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]
b1_results = []

# Apply baseline catalyst exclusions (earnings, deal, macro)
short_filtered = short_df[
    ~short_df['is_earnings'] &
    ~short_df['is_deal_flag'] &
    ~short_df['is_macro_flag']
].copy()
print(f"  After catalyst exclusion (earnings/deal/macro): {len(short_filtered)}")

print(f"\n  {'Threshold':>12} {'n_total':>8} {'n_t2':>6} {'t/d':>6} "
      f"{'avg%':>8} {'med%':>8} {'WR':>6} {'Sharpe':>7} {'CI_lo':>8} {'CI_hi':>8} {'CI_OK':>6}")
print(f"  {'─'*100}")

for thresh in gap_thresholds:
    subset = short_filtered[short_filtered['abs_gap_pct'] >= thresh]
    n_total = len(subset)
    pnl_vals = subset['pnl_t2_close'].dropna().tolist()
    n_t2 = len(pnl_vals)

    if n_t2 >= 3:
        m = full_metrics(pnl_vals, N_DATES_WITH_T2)
        loo = leave_one_out_check(pnl_vals)
    else:
        m = full_metrics(pnl_vals, N_DATES_WITH_T2)
        loo = {'stable': None}

    result = {
        'threshold_pct': thresh,
        'n_total': n_total,
        'n_with_t2': n_t2,
        'trades_per_day_all': round(n_total / N_DATES, 1),
        'trades_per_day_t2': round(n_t2 / N_DATES_WITH_T2, 1) if N_DATES_WITH_T2 > 0 else 0,
        **m,
        'loo_stable': loo.get('stable'),
    }
    b1_results.append(result)

    avg_s = f"{m['avg_pnl_pct']:>+8.2f}" if m['avg_pnl_pct'] is not None else f"{'—':>8}"
    med_s = f"{m['median_pnl_pct']:>+8.2f}" if m.get('median_pnl_pct') is not None else f"{'—':>8}"
    wr_s = f"{m['win_rate']*100:>5.0f}%" if m.get('win_rate') is not None else f"{'—':>6}"
    sh_s = f"{m['sharpe']:>7.3f}" if m.get('sharpe') is not None else f"{'—':>7}"
    ci_lo_s = f"{m['ci_95_low']:>+8.2f}" if m.get('ci_95_low') is not None else f"{'—':>8}"
    ci_hi_s = f"{m['ci_95_high']:>+8.2f}" if m.get('ci_95_high') is not None else f"{'—':>8}"
    ci_ok_s = f"{'YES' if m.get('ci_confirmed') else 'NO':>6}" if m.get('ci_confirmed') is not None else f"{'—':>6}"

    tpd = round(n_total / N_DATES, 1)
    print(f"  {thresh:>10.1f}% {n_total:>8} {n_t2:>6} {tpd:>6.1f} "
          f"{avg_s} {med_s} {wr_s} {sh_s} {ci_lo_s} {ci_hi_s} {ci_ok_s}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B2: Catalyst Inclusion Expansion
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  B2: Catalyst Inclusion Expansion (LLM SHORT gap-up >= 2%)")
print(f"{'='*_W}")

# Base universe: SHORT gap-up >= 2%
b2_base = short_df[short_df['abs_gap_pct'] >= 2.0].copy()
print(f"\n  Base (SHORT gap-up >= 2%): {len(b2_base)} trades, {b2_base['pnl_t2_close'].notna().sum()} with T+2")

# Catalyst breakdown
print(f"\n  Per-catalyst T+2 performance:")
print(f"  {'Catalyst':>20} {'n':>5} {'n_t2':>5} {'avg%':>8} {'med%':>8} {'WR':>6} {'Sharpe':>7}")
print(f"  {'─'*70}")

catalyst_results = {}
for cat in sorted(b2_base['catalyst_type_lower'].unique()):
    cat_sub = b2_base[b2_base['catalyst_type_lower'] == cat]
    pnl = cat_sub['pnl_t2_close'].dropna().tolist()
    n_t2 = len(pnl)
    n_total = len(cat_sub)

    if n_t2 >= 3:
        m = full_metrics(pnl, N_DATES_WITH_T2)
        avg_s = f"{m['avg_pnl_pct']:>+8.2f}"
        med_s = f"{m['median_pnl_pct']:>+8.2f}"
        wr_s = f"{m['win_rate']*100:>5.0f}%"
        sh_s = f"{m['sharpe']:>7.3f}"
    elif n_t2 > 0:
        avg_v = np.mean(pnl)
        m = {'avg_pnl_pct': round(avg_v, 3), 'n': n_t2, 'note': 'insufficient'}
        avg_s = f"{avg_v:>+8.2f}"
        med_s = f"{'—':>8}"
        wr_s = f"{'—':>6}"
        sh_s = f"{'—':>7}"
    else:
        m = {'n': 0, 'note': 'no T+2 data'}
        avg_s = f"{'—':>8}"
        med_s = f"{'—':>8}"
        wr_s = f"{'—':>6}"
        sh_s = f"{'—':>7}"

    catalyst_results[cat] = {'n_total': n_total, 'n_t2': n_t2, **m}
    print(f"  {cat:>20} {n_total:>5} {n_t2:>5} {avg_s} {med_s} {wr_s} {sh_s}")

# Test different catalyst inclusion sets
catalyst_sets = {
    'current': {'exclude': ['earnings', 'deal', 'macro', 'breaking']},
    'add_breaking': {'exclude': ['earnings', 'deal', 'macro']},
    'add_macro': {'exclude': ['earnings', 'deal']},
    'minimal_earnings_only': {'exclude': ['earnings']},
    'no_exclusions': {'exclude': []},
    'curated_best': {'exclude': ['earnings', 'deal', 'breaking']},
}

print(f"\n  Catalyst Set Comparison (SHORT gap-up >= 2%, T+2 hold):")
print(f"  {'Set':>25} {'n':>5} {'n_t2':>5} {'t/d':>6} "
      f"{'avg%':>8} {'med%':>8} {'WR':>6} {'Sharpe':>7} {'CI_lo':>8} {'CI_OK':>6}")
print(f"  {'─'*100}")

b2_set_results = {}
for set_name, config in catalyst_sets.items():
    exclude_list = config['exclude']
    mask = pd.Series(True, index=b2_base.index)

    for excl in exclude_list:
        if excl == 'earnings':
            mask &= ~b2_base['is_earnings']
        elif excl == 'deal':
            mask &= ~b2_base['is_deal_flag']
        elif excl == 'macro':
            mask &= ~b2_base['is_macro_flag']
        elif excl == 'breaking':
            mask &= ~b2_base['is_breaking_flag']

    subset = b2_base[mask]
    pnl = subset['pnl_t2_close'].dropna().tolist()
    n_t2 = len(pnl)
    n_total = len(subset)
    m = full_metrics(pnl, N_DATES_WITH_T2)
    b2_set_results[set_name] = {'n_total': n_total, 'n_t2': n_t2, **m, 'exclude': exclude_list}

    avg_s = f"{m['avg_pnl_pct']:>+8.2f}" if m.get('avg_pnl_pct') is not None else f"{'—':>8}"
    med_s = f"{m['median_pnl_pct']:>+8.2f}" if m.get('median_pnl_pct') is not None else f"{'—':>8}"
    wr_s = f"{m['win_rate']*100:>5.0f}%" if m.get('win_rate') is not None else f"{'—':>6}"
    sh_s = f"{m['sharpe']:>7.3f}" if m.get('sharpe') is not None else f"{'—':>7}"
    ci_lo_s = f"{m['ci_95_low']:>+8.2f}" if m.get('ci_95_low') is not None else f"{'—':>8}"
    ci_ok_s = f"{'YES' if m.get('ci_confirmed') else 'NO':>6}" if m.get('ci_confirmed') is not None else f"{'—':>6}"
    tpd = round(n_total / N_DATES, 1)

    print(f"  {set_name:>25} {n_total:>5} {n_t2:>5} {tpd:>6.1f} "
          f"{avg_s} {med_s} {wr_s} {sh_s} {ci_lo_s} {ci_ok_s}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B3: Discovery vs Premarket Scanner Separation
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  B3: Discovery vs Premarket Scanner Performance")
print(f"{'='*_W}")

# Use best catalyst set from B2 (current = exclude earnings/deal/macro/breaking)
b3_base = short_filtered[short_filtered['abs_gap_pct'] >= 2.0].copy()

b3_results = {}
for source in ['premarket', 'discovery']:
    sub = b3_base[b3_base['scanner'] == source]
    n_total = len(sub)
    pnl = sub['pnl_t2_close'].dropna().tolist()
    n_t2 = len(pnl)
    m = full_metrics(pnl, N_DATES_WITH_T2)

    # Per-date breakdown
    daily = {}
    for dt in TRADING_DATES:
        day_sub = sub[sub['eval_date'] == dt]
        day_pnl = day_sub['pnl_t2_close'].dropna().tolist()
        daily[dt] = {
            'n_total': len(day_sub),
            'n_t2': len(day_pnl),
            'avg_pnl': round(np.mean(day_pnl), 3) if day_pnl else None,
            'tickers': list(day_sub['ticker'].values),
        }

    b3_results[source] = {'n_total': n_total, 'n_t2': n_t2, **m, 'daily': daily}

    avg_s = f"{m['avg_pnl_pct']:>+8.2f}" if m.get('avg_pnl_pct') is not None else "—"
    med_s = f"{m['median_pnl_pct']:>+8.2f}" if m.get('median_pnl_pct') is not None else "—"
    wr_s = f"{m['win_rate']*100:.0f}%" if m.get('win_rate') is not None else "—"
    sh_s = f"{m['sharpe']:.3f}" if m.get('sharpe') is not None else "—"
    ci_s = f"[{m['ci_95_low']:+.2f}, {m['ci_95_high']:+.2f}]" if m.get('ci_95_low') is not None else "—"

    print(f"\n  {source.upper():>12}: {n_total} total, {n_t2} with T+2")
    print(f"    avg={avg_s}%  med={med_s}%  WR={wr_s}  Sharpe={sh_s}  CI={ci_s}")
    for dt in TRADING_DATES:
        d = daily[dt]
        tks = ', '.join(d['tickers'][:5])
        avg_d = f"{d['avg_pnl']:+.2f}%" if d['avg_pnl'] is not None else "—"
        print(f"    {dt}: {d['n_total']} signals, {d['n_t2']} T+2, avg={avg_d}  [{tks}]")


# Also test: combined premarket + discovery (no split)
print(f"\n  COMBINED (for reference):")
combined_pnl = b3_base['pnl_t2_close'].dropna().tolist()
m_combined = full_metrics(combined_pnl, N_DATES_WITH_T2)
if m_combined.get('avg_pnl_pct') is not None:
    print(f"    n={m_combined['n']}, avg={m_combined['avg_pnl_pct']:+.2f}%, "
          f"Sharpe={m_combined.get('sharpe', 0):.3f}, "
          f"CI=[{m_combined.get('ci_95_low', 0):+.2f}, {m_combined.get('ci_95_high', 0):+.2f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B4: n_runs Consistency Signal
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  B4: n_runs Consistency Signal")
print(f"{'='*_W}")

b4_base = short_filtered[short_filtered['abs_gap_pct'] >= 2.0].copy()
print(f"\n  Base: {len(b4_base)} SHORT gap-up >= 2% (catalyst filtered)")

# n_runs distribution
print(f"\n  n_runs distribution:")
for nr in sorted(b4_base['n_runs'].unique()):
    n = (b4_base['n_runs'] == nr).sum()
    n_t2 = b4_base[b4_base['n_runs'] == nr]['pnl_t2_close'].notna().sum()
    pnl = b4_base[b4_base['n_runs'] == nr]['pnl_t2_close'].dropna().tolist()
    avg = f"{np.mean(pnl):+.2f}%" if pnl else "—"
    print(f"    n_runs={int(nr)}: {n} trades ({n_t2} T+2), avg={avg}")

# Threshold sweep
b4_results = {}
nruns_thresholds = [1, 2, 3, 4, 5]

print(f"\n  n_runs threshold sweep (cumulative >=):")
print(f"  {'n_runs>=':>10} {'n':>5} {'n_t2':>5} {'t/d':>6} "
      f"{'avg%':>8} {'med%':>8} {'WR':>6} {'Sharpe':>7} {'CI_lo':>8} {'CI_OK':>6}")
print(f"  {'─'*80}")

for thresh in nruns_thresholds:
    subset = b4_base[b4_base['n_runs'] >= thresh]
    pnl = subset['pnl_t2_close'].dropna().tolist()
    m = full_metrics(pnl, N_DATES_WITH_T2)
    b4_results[thresh] = {'n_total': len(subset), 'n_t2': len(pnl), **m}

    avg_s = f"{m['avg_pnl_pct']:>+8.2f}" if m.get('avg_pnl_pct') is not None else f"{'—':>8}"
    med_s = f"{m['median_pnl_pct']:>+8.2f}" if m.get('median_pnl_pct') is not None else f"{'—':>8}"
    wr_s = f"{m['win_rate']*100:>5.0f}%" if m.get('win_rate') is not None else f"{'—':>6}"
    sh_s = f"{m['sharpe']:>7.3f}" if m.get('sharpe') is not None else f"{'—':>7}"
    ci_lo_s = f"{m['ci_95_low']:>+8.2f}" if m.get('ci_95_low') is not None else f"{'—':>8}"
    ci_ok_s = f"{'YES' if m.get('ci_confirmed') else 'NO':>6}" if m.get('ci_confirmed') is not None else f"{'—':>6}"
    tpd = round(len(subset) / N_DATES, 1)

    print(f"  {thresh:>10} {len(subset):>5} {len(pnl):>5} {tpd:>6.1f} "
          f"{avg_s} {med_s} {wr_s} {sh_s} {ci_lo_s} {ci_ok_s}")

# Test: ranking by n_runs × gap_pct (top N per day)
print(f"\n  Ranking by n_runs × |gap_pct| — top N per day:")
b4_base['rank_score'] = b4_base['n_runs'] * b4_base['abs_gap_pct']

for top_n in [1, 2, 3, 4, 5]:
    selected = []
    for dt in TRADING_DATES:
        day = b4_base[b4_base['eval_date'] == dt].sort_values('rank_score', ascending=False)
        selected.append(day.head(top_n))
    sel_df = pd.concat(selected)
    pnl = sel_df['pnl_t2_close'].dropna().tolist()
    if pnl:
        m = full_metrics(pnl, N_DATES_WITH_T2)
        avg_s = f"{m['avg_pnl_pct']:+.2f}" if m.get('avg_pnl_pct') is not None else "—"
        wr_s = f"{m['win_rate']*100:.0f}%" if m.get('win_rate') is not None else "—"
        tickers = list(sel_df[['eval_date', 'ticker', 'pnl_t2_close']].itertuples(index=False, name=None))
        print(f"    Top {top_n}/day: n={m['n']} with T+2, avg={avg_s}, WR={wr_s}")
    else:
        print(f"    Top {top_n}/day: no T+2 data available")


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL: Multi-hold Analysis for LLM SHORT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  ADDITIONAL: Multi-Hold Analysis for LLM SHORT")
print(f"{'='*_W}")

hold_cols = {
    '30m': 'pnl_30m_nosl',
    '60m': 'pnl_60m_nosl',
    '120m': 'pnl_120m_nosl',
    'EOD': 'pnl_eod_nosl',
    'T+1': 'pnl_t1_close',
    'T+2': 'pnl_t2_close',
    'T+3': 'pnl_t3_close',
}

multi_hold_base = short_filtered[short_filtered['abs_gap_pct'] >= 2.0]
print(f"\n  Base: {len(multi_hold_base)} SHORT gap-up >= 2% (catalyst filtered)")
print(f"\n  {'Hold':>8} {'n':>5} {'avg%':>8} {'med%':>8} {'WR':>6} {'Sharpe':>7} {'CI_lo':>8} {'CI_OK':>6}")
print(f"  {'─'*65}")

multi_hold_results = {}
for hold_name, col in hold_cols.items():
    if col not in multi_hold_base.columns:
        continue
    pnl = multi_hold_base[col].dropna().tolist()
    nd = N_DATES_WITH_T2 if 'T+' in hold_name else N_DATES
    m = full_metrics(pnl, nd)
    multi_hold_results[hold_name] = m

    avg_s = f"{m['avg_pnl_pct']:>+8.2f}" if m.get('avg_pnl_pct') is not None else f"{'—':>8}"
    med_s = f"{m['median_pnl_pct']:>+8.2f}" if m.get('median_pnl_pct') is not None else f"{'—':>8}"
    wr_s = f"{m['win_rate']*100:>5.0f}%" if m.get('win_rate') is not None else f"{'—':>6}"
    sh_s = f"{m['sharpe']:>7.3f}" if m.get('sharpe') is not None else f"{'—':>7}"
    ci_lo_s = f"{m['ci_95_low']:>+8.2f}" if m.get('ci_95_low') is not None else f"{'—':>8}"
    ci_ok_s = f"{'YES' if m.get('ci_confirmed') else 'NO':>6}" if m.get('ci_confirmed') is not None else f"{'—':>6}"

    print(f"  {hold_name:>8} {m['n']:>5} {avg_s} {med_s} {wr_s} {sh_s} {ci_lo_s} {ci_ok_s}")


# ═══════════════════════════════════════════════════════════════════════════════
# TIER SYSTEM SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  TIER SYSTEM SIMULATION — 3 Trades/Day Target")
print(f"{'='*_W}")

TARGET_PER_DAY = 3

# Define tier criteria
def classify_tier(row):
    gap_ok = row['abs_gap_pct'] >= 2.0
    gap_ok_loose = row['abs_gap_pct'] >= 1.5
    is_short_gap_up = row['direction'] == 'SHORT' and row['gap_pct'] > 0
    float_ok = True  # no float filter in v2 data, assume OK
    pm_ok = True
    if pd.notna(row.get('pm_retrace')):
        pm_ok = row['pm_retrace'] <= 0.8
    pm_sweet = False
    if pd.notna(row.get('pm_retrace')):
        pm_sweet = 0.2 <= row['pm_retrace'] <= 0.8

    no_earnings = not row.get('is_earnings', False)
    no_deal = not row.get('is_deal_flag', False)
    no_breaking = not row.get('is_breaking_flag', False)
    no_macro = not row.get('is_macro_flag', False)

    is_offering = row.get('is_offering', False)
    is_regulatory = row.get('is_regulatory', False)

    if not is_short_gap_up:
        return None

    # Tier 1: best catalysts, tighter filters
    if gap_ok and (is_offering or is_regulatory) and pm_sweet and no_earnings and no_deal:
        return 1

    # Tier 2: broader catalysts, standard filters
    if gap_ok and no_earnings and no_deal and no_breaking and pm_ok:
        return 2

    # Tier 3: loosest filters
    if gap_ok_loose and no_earnings and no_deal:
        return 3

    return None


tier_simulation = {}
tier_all_trades = []

for dt in TRADING_DATES:
    day_df = df[df['eval_date'] == dt].copy()
    day_df['tier'] = day_df.apply(classify_tier, axis=1)
    day_qualified = day_df[day_df['tier'].notna()].copy()

    # Sort: Tier 1 first, then 2, then 3; within tier by n_runs * gap_pct
    day_qualified['sort_key'] = day_qualified['tier'] * 1000 - day_qualified['n_runs'] * day_qualified['abs_gap_pct']
    day_qualified = day_qualified.sort_values('sort_key')

    selected = day_qualified.head(TARGET_PER_DAY)

    tier_counts = {1: 0, 2: 0, 3: 0}
    for _, row in selected.iterrows():
        t = int(row['tier'])
        tier_counts[t] = tier_counts.get(t, 0) + 1

    available_counts = {1: int((day_qualified['tier'] == 1).sum()),
                        2: int((day_qualified['tier'] == 2).sum()),
                        3: int((day_qualified['tier'] == 3).sum())}

    tier_simulation[dt] = {
        'total_qualified': len(day_qualified),
        'available': available_counts,
        'selected': tier_counts,
        'n_selected': len(selected),
        'trades': []
    }

    print(f"\n  {dt}: {len(day_qualified)} qualified "
          f"(T1:{available_counts[1]}, T2:{available_counts[2]}, T3:{available_counts[3]})")
    print(f"    Selected {len(selected)} trades:")

    for _, row in selected.iterrows():
        t2 = f"{row['pnl_t2_close']:+.2f}%" if pd.notna(row['pnl_t2_close']) else "no T+2"
        trade_info = {
            'ticker': row['ticker'],
            'tier': int(row['tier']),
            'gap_pct': round(row['gap_pct'], 2),
            'catalyst': row['catalyst_type_lower'],
            'n_runs': int(row['n_runs']),
            'pnl_t2': round(row['pnl_t2_close'], 2) if pd.notna(row['pnl_t2_close']) else None,
            'scanner': row['scanner'],
        }
        tier_simulation[dt]['trades'].append(trade_info)
        tier_all_trades.append(trade_info)
        print(f"      T{int(row['tier'])}: {row['ticker']:<6} gap={row['gap_pct']:+.1f}%  "
              f"cat={row['catalyst_type_lower']:<20} n_runs={int(row['n_runs'])}  T+2={t2}")

# Overall tier simulation metrics
print(f"\n  TIER SIMULATION SUMMARY:")
all_t2 = [t['pnl_t2'] for t in tier_all_trades if t['pnl_t2'] is not None]
n_selected_total = sum(s['n_selected'] for s in tier_simulation.values())
print(f"    Total selected: {n_selected_total} ({n_selected_total/N_DATES:.1f}/day)")
if all_t2:
    print(f"    With T+2 data: {len(all_t2)}")
    print(f"    Avg T+2 PnL: {np.mean(all_t2):+.2f}%")
    print(f"    Median T+2 PnL: {np.median(all_t2):+.2f}%")
    print(f"    Win rate: {(np.array(all_t2) > 0).mean()*100:.0f}%")

per_tier_pnl = defaultdict(list)
for t in tier_all_trades:
    if t['pnl_t2'] is not None:
        per_tier_pnl[t['tier']].append(t['pnl_t2'])

print(f"\n    Per-tier performance:")
for tier in [1, 2, 3]:
    pnl = per_tier_pnl.get(tier, [])
    if pnl:
        print(f"      Tier {tier}: n={len(pnl)}, avg={np.mean(pnl):+.2f}%, "
              f"WR={100*(np.array(pnl)>0).mean():.0f}%")
    else:
        print(f"      Tier {tier}: no T+2 data")


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SCHEMA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  LOGGING SCHEMA VALIDATION")
print(f"{'='*_W}")

TRADE_LOG_FIELDS = [
    ('log_date', 'str', True),
    ('ticker', 'str', True),
    ('gap_date', 'str', True),
    ('source_scanner', 'str', True),
    ('scanner_run_time', 'str', False),
    ('signal_rank', 'int', True),
    ('gap_pct', 'float', True),
    ('gap_direction', 'str', True),
    ('gap_bucket', 'str', True),
    ('open_price', 'float', True),
    ('prior_close', 'float', True),
    ('llm_direction', 'str', True),
    ('llm_confidence', 'float', True),
    ('llm_expected_move', 'float', True),
    ('llm_catalyst_type', 'str', True),
    ('llm_catalyst_summary', 'str', True),
    ('n_runs', 'int', True),
    ('direction_consistency', 'float', True),
    ('pm_high', 'float', True),
    ('pm_last', 'float', True),
    ('pm_retrace_pct', 'float', True),
    ('pm_retrace_bucket', 'str', True),
    ('pm_vol_ratio', 'float', True),
    ('pm_volume_profile', 'str', False),
    ('float_shares', 'float', False),
    ('float_excluded', 'bool', True),
    ('pm_retrace_excluded', 'bool', True),
    ('catalyst_excluded', 'bool', True),
    ('breaking_excluded', 'bool', True),
    ('passed_all_filters', 'bool', True),
    ('orb_high', 'float', True),
    ('orb_low', 'float', True),
    ('orb_width_pct', 'float', True),
    ('orb_width_atr_multiple', 'float', False),
    ('orb_confirmed_short', 'bool', True),
    ('orb_confirm_bar', 'int', True),
    ('entry_type', 'str', True),
    ('entry_bar', 'int', True),
    ('entry_time', 'str', True),
    ('entry_price', 'float', True),
    ('theoretical_local_high', 'float', True),
    ('slippage_pct', 'float', False),
    ('score_orb_width', 'float', True),
    ('score_pm_retrace', 'float', True),
    ('score_gap_magnitude', 'float', True),
    ('score_offering_bonus', 'float', True),
    ('composite_score', 'float', True),
    ('composite_rank_in_day', 'int', True),
    ('spy_direction_at_entry', 'str', True),
    ('spy_intraday_pct_at_entry', 'float', True),
    ('sector_etf', 'str', False),
    ('sector_direction_at_entry', 'str', False),
    ('pnl_30m', 'float', True),
    ('pnl_60m', 'float', True),
    ('pnl_eod', 'float', True),
    ('pnl_t1', 'float', True),
    ('pnl_t2', 'float', True),
    ('actual_hold', 'str', False),
    ('actual_pnl', 'float', False),
    ('exit_reason', 'str', False),
    ('borrow_rate_pct', 'float', False),
    ('did_orb_confirm', 'bool', True),
    ('notes', 'str', False),
]

# Map log fields to v2_features_full columns
field_mapping = {
    'ticker': 'ticker',
    'gap_date': 'eval_date',
    'source_scanner': 'scanner',
    'gap_pct': 'gap_pct',
    'open_price': 'entry_price',
    'llm_direction': 'direction',
    'llm_confidence': 'confidence',
    'llm_expected_move': 'expected_move_pct',
    'llm_catalyst_type': 'catalyst_type',
    'llm_catalyst_summary': 'catalyst',
    'n_runs': 'n_runs',
    'direction_consistency': 'stability_score',
    'pm_retrace_pct': 'pm_retrace',
    'pm_vol_ratio': 'log_vol_ratio',
    'orb_high': 'or_high',
    'orb_low': 'or_low',
    'entry_price': 'entry_price',
    'entry_time': 'entry_time_et',
    'pnl_30m': 'pnl_30m_nosl',
    'pnl_60m': 'pnl_60m_nosl',
    'pnl_eod': 'pnl_eod_nosl',
    'pnl_t1': 'pnl_t1_close',
    'pnl_t2': 'pnl_t2_close',
}

computable = []
needs_new_data = []
needs_live_only = []

for field_name, field_type, from_existing in TRADE_LOG_FIELDS:
    if field_name in field_mapping:
        src = field_mapping[field_name]
        if src in df.columns:
            n_valid = df[src].notna().sum()
            computable.append((field_name, src, n_valid))
        else:
            needs_new_data.append((field_name, f'mapped to {src} but column missing'))
    elif from_existing:
        # Derivable from existing data
        if field_name in ['log_date', 'gap_direction', 'gap_bucket', 'signal_rank',
                          'pm_retrace_bucket', 'float_excluded', 'pm_retrace_excluded',
                          'catalyst_excluded', 'breaking_excluded', 'passed_all_filters',
                          'orb_width_pct', 'orb_confirmed_short', 'orb_confirm_bar',
                          'entry_type', 'entry_bar', 'theoretical_local_high',
                          'score_orb_width', 'score_pm_retrace', 'score_gap_magnitude',
                          'score_offering_bonus', 'composite_score', 'composite_rank_in_day',
                          'spy_direction_at_entry', 'spy_intraday_pct_at_entry',
                          'did_orb_confirm']:
            computable.append((field_name, 'derivable from existing data', None))
        elif field_name in ['pm_high', 'pm_last', 'prior_close']:
            computable.append((field_name, 'from minute bars / daily data', None))
        else:
            needs_new_data.append((field_name, 'not in existing data'))
    else:
        needs_live_only.append((field_name, 'live trading only'))

print(f"\n  Log schema: {len(TRADE_LOG_FIELDS)} fields total")
print(f"  Computable from backtest data: {len(computable)}")
print(f"  Needs new data pulls in live: {len(needs_new_data)}")
print(f"  Live trading only: {len(needs_live_only)}")

if needs_new_data:
    print(f"\n  FIELDS NEEDING NEW DATA IN LIVE:")
    for name, reason in needs_new_data:
        print(f"    {name}: {reason}")

if needs_live_only:
    print(f"\n  LIVE-ONLY FIELDS (expected):")
    for name, reason in needs_live_only:
        print(f"    {name}: {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  Generating charts...")

# ── Chart 1: Gap Threshold Sweep ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Agent 2 — B1: Gap Threshold Sweep (LLM SHORT, T+2)\n'
             f'Catalyst excluded: earnings, deal, macro',
             fontsize=12, fontweight='bold')

thresholds = [r['threshold_pct'] for r in b1_results]
avgs = [r.get('avg_pnl_pct', 0) or 0 for r in b1_results]
ns = [r.get('n_with_t2', 0) for r in b1_results]
tpds = [r.get('trades_per_day_all', 0) for r in b1_results]

# Avg PnL
ax = axes[0]
colors = ['#27ae60' if a > 0 else '#e74c3c' for a in avgs]
ax.bar(range(len(thresholds)), avgs, color=colors, alpha=0.8, edgecolor='white')
ax.set_xticks(range(len(thresholds)))
ax.set_xticklabels([f'{t}%' for t in thresholds], fontsize=8)
ax.set_ylabel('Avg T+2 PnL (%)')
ax.set_xlabel('Min Gap Threshold')
ax.axhline(0, color='#333', lw=0.7)
ax.set_title('Avg PnL by Gap Threshold')
for i, (a, n) in enumerate(zip(avgs, ns)):
    ax.text(i, a + 0.1, f'n={n}', ha='center', fontsize=7)

# Trades per day
ax = axes[1]
ax.bar(range(len(thresholds)), tpds, color='#3498db', alpha=0.8, edgecolor='white')
ax.set_xticks(range(len(thresholds)))
ax.set_xticklabels([f'{t}%' for t in thresholds], fontsize=8)
ax.set_ylabel('Trades / Day (all dates)')
ax.set_xlabel('Min Gap Threshold')
ax.axhline(3, color='#e67e22', ls='--', lw=1.5, label='Target: 3/day')
ax.set_title('Trade Frequency')
ax.legend(fontsize=8)

# Sharpe
sharpes = [r.get('sharpe', 0) or 0 for r in b1_results]
ax = axes[2]
ax.plot(range(len(thresholds)), sharpes, 'o-', color='#8e44ad', markersize=8, linewidth=2)
ax.set_xticks(range(len(thresholds)))
ax.set_xticklabels([f'{t}%' for t in thresholds], fontsize=8)
ax.set_ylabel('Sharpe')
ax.set_xlabel('Min Gap Threshold')
ax.axhline(0, color='#333', lw=0.7)
ax.set_title('Sharpe by Gap Threshold')

plt.tight_layout(rect=[0, 0, 1, 0.92])
chart1_path = os.path.join(BASE_DIR, 'agent2_gap_threshold.png')
plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Chart 1: {chart1_path}")

# ── Chart 2: Catalyst Comparison ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Agent 2 — B2: Catalyst Inclusion Sets', fontsize=12, fontweight='bold')

set_names = list(b2_set_results.keys())
set_avgs = [b2_set_results[s].get('avg_pnl_pct', 0) or 0 for s in set_names]
set_ns = [b2_set_results[s].get('n_total', 0) for s in set_names]
set_sharpes = [b2_set_results[s].get('sharpe', 0) or 0 for s in set_names]

ax = axes[0]
colors = ['#27ae60' if a > 0 else '#e74c3c' for a in set_avgs]
bars = ax.barh(range(len(set_names)), set_avgs, color=colors, alpha=0.8)
ax.set_yticks(range(len(set_names)))
ax.set_yticklabels(set_names, fontsize=9)
ax.set_xlabel('Avg T+2 PnL (%)')
ax.axvline(0, color='#333', lw=0.7)
ax.set_title('Avg PnL')
for i, (a, n) in enumerate(zip(set_avgs, set_ns)):
    ax.text(a + 0.1, i, f'n={n}', va='center', fontsize=8)

ax = axes[1]
colors = ['#3498db' if s > 0 else '#e74c3c' for s in set_sharpes]
ax.barh(range(len(set_names)), set_sharpes, color=colors, alpha=0.8)
ax.set_yticks(range(len(set_names)))
ax.set_yticklabels(set_names, fontsize=9)
ax.set_xlabel('Sharpe')
ax.axvline(0, color='#333', lw=0.7)
ax.set_title('Sharpe')

plt.tight_layout(rect=[0, 0, 1, 0.92])
chart2_path = os.path.join(BASE_DIR, 'agent2_catalyst_sets.png')
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Chart 2: {chart2_path}")

# ── Chart 3: Scanner Comparison + n_runs ──────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Agent 2 — B3: Scanner Split + B4: n_runs', fontsize=12, fontweight='bold')

# Scanner comparison
ax = axes[0]
scanner_names = list(b3_results.keys())
scanner_avgs = [b3_results[s].get('avg_pnl_pct', 0) or 0 for s in scanner_names]
scanner_ns = [b3_results[s].get('n_t2', 0) for s in scanner_names]
colors = ['#27ae60' if a > 0 else '#e74c3c' for a in scanner_avgs]
bars = ax.bar(scanner_names, scanner_avgs, color=colors, alpha=0.8, edgecolor='white')
for i, (a, n) in enumerate(zip(scanner_avgs, scanner_ns)):
    ax.text(i, a + 0.1, f'n={n}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(0, color='#333', lw=0.7)
ax.set_ylabel('Avg T+2 PnL (%)')
ax.set_title('B3: Premarket vs Discovery Scanner')

# n_runs
ax = axes[1]
nruns = list(b4_results.keys())
nruns_avgs = [b4_results[nr].get('avg_pnl_pct', 0) or 0 for nr in nruns]
nruns_ns = [b4_results[nr].get('n_t2', 0) for nr in nruns]
colors = ['#27ae60' if a > 0 else '#e74c3c' for a in nruns_avgs]
bars = ax.bar([f'>={nr}' for nr in nruns], nruns_avgs, color=colors, alpha=0.8, edgecolor='white')
for i, (a, n) in enumerate(zip(nruns_avgs, nruns_ns)):
    ax.text(i, a + 0.1, f'n={n}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(0, color='#333', lw=0.7)
ax.set_ylabel('Avg T+2 PnL (%)')
ax.set_xlabel('n_runs Threshold')
ax.set_title('B4: n_runs Consistency Signal')

plt.tight_layout(rect=[0, 0, 1, 0.92])
chart3_path = os.path.join(BASE_DIR, 'agent2_scanner_nruns.png')
plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Chart 3: {chart3_path}")

# ── Chart 4: Tier System Daily ────────────────────────────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.suptitle('Agent 2 — Tier System Simulation (3 trades/day target)',
             fontsize=12, fontweight='bold')

tier_colors = {1: '#27ae60', 2: '#3498db', 3: '#e67e22'}
x = np.arange(N_DATES)
bottom = np.zeros(N_DATES)

for tier in [1, 2, 3]:
    heights = []
    for dt in TRADING_DATES:
        sim = tier_simulation[dt]
        heights.append(sim['selected'].get(tier, 0))
    ax.bar(x, heights, bottom=bottom, label=f'Tier {tier}', color=tier_colors[tier],
           alpha=0.8, edgecolor='white')
    bottom += np.array(heights)

ax.set_xticks(x)
ax.set_xticklabels(TRADING_DATES, fontsize=10)
ax.set_ylabel('# Trades Selected')
ax.axhline(TARGET_PER_DAY, color='#e74c3c', ls='--', lw=2, label=f'Target ({TARGET_PER_DAY}/day)')
ax.legend()
ax.set_title('Daily Trade Selection by Tier')

# Add trade annotations
for i, dt in enumerate(TRADING_DATES):
    trades = tier_simulation[dt]['trades']
    labels = [f"{t['ticker']} ({t['pnl_t2']:+.1f}%)" if t['pnl_t2'] is not None
              else f"{t['ticker']} (no T+2)" for t in trades]
    label_text = '\n'.join(labels)
    ax.text(i, bottom[i] + 0.1, label_text, ha='center', va='bottom', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.93])
chart4_path = os.path.join(BASE_DIR, 'agent2_tier_simulation.png')
plt.savefig(chart4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Chart 4: {chart4_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

results_output = {
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'silence_diagnosis': silence_diagnosis,
    'b1_gap_threshold': b1_results,
    'b2_catalyst_sets': {k: {kk: vv for kk, vv in v.items() if kk != 'exclude'}
                         for k, v in b2_set_results.items()},
    'b2_per_catalyst': catalyst_results,
    'b3_scanner_split': {k: {kk: vv for kk, vv in v.items() if kk != 'daily'}
                         for k, v in b3_results.items()},
    'b4_nruns': {str(k): v for k, v in b4_results.items()},
    'multi_hold': multi_hold_results,
    'tier_simulation': tier_simulation,
}

results_path = os.path.join(RESULTS_DIR, 'agent2_experiment_results.json')
with open(results_path, 'w') as f:
    json.dump(results_output, f, indent=2, default=str)
print(f"\n  Results saved: {results_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE MARKDOWN REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def fmt(v, decimals=2):
    if v is None:
        return '—'
    if isinstance(v, bool):
        return 'YES' if v else 'NO'
    if isinstance(v, float):
        return f'{v:+.{decimals}f}' if v != 0 else f'{v:.{decimals}f}'
    return str(v)

def fmt_wr(r):
    wr = r.get('win_rate')
    if wr is None:
        return '—'
    return f'{wr*100:.0f}%'

report_lines = []
report_lines.append(f"# Agent 2: Expansion + Logging — Experiment Results")
report_lines.append(f"")
report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"Dataset: v2_features_full.csv — 316 LLM trades, 4 dates (Sep 8-11, 2026)")
report_lines.append(f"")

# 1. Scanner Silence
report_lines.append(f"## 1. Scanner Silence Diagnosis")
report_lines.append(f"")
report_lines.append(f"| Date | Total | Premarket | Discovery | SHORT gap-up | SHORT >=3% | Has T+2 |")
report_lines.append(f"|------|-------|-----------|-----------|-------------|-----------|---------|")
for dt in TRADING_DATES:
    d = silence_diagnosis[dt]
    t2 = f"{d['has_t2_data']}" if d['has_t2_data'] > 0 else "**0**"
    report_lines.append(f"| {dt} | {d['total']} | {d['premarket']} | {d['discovery']} | "
                        f"{d['SHORT_gap_up']} | {d['SHORT_gap_up_gte3']} | {t2} |")
report_lines.append(f"")
report_lines.append(f"**DIAGNOSIS**: Scanner runs on ALL 4 days ({total_signals} total signals, "
                     f"{total_signals/N_DATES:.0f}/day avg). ")
if dates_without_t2:
    report_lines.append(f"Days {', '.join(dates_without_t2)} have 0 T+2 outcome data — "
                        f"this is a **data availability issue**, not scanner silence. ")
    report_lines.append(f"In live trading, Agent 2 would have signals every day. "
                        f"Backtest T+2 metrics use only {N_DATES_WITH_T2} dates with complete data.")
report_lines.append(f"")

# 2. Gap Threshold
report_lines.append(f"## 2. Gap Threshold Sweep (B1)")
report_lines.append(f"")
report_lines.append(f"Universe: LLM SHORT gap-up, catalyst-filtered (excl earnings/deal/macro)")
report_lines.append(f"")
report_lines.append(f"| Threshold | n_total | n_T+2 | t/d | Avg% | Med% | WR | Sharpe | CI_lo | CI_hi | CI>0 |")
report_lines.append(f"|-----------|---------|-------|-----|------|------|-----|--------|-------|-------|------|")
for r in b1_results:
    report_lines.append(f"| {r['threshold_pct']}% | {r['n_total']} | {r['n_with_t2']} | "
                        f"{r['trades_per_day_all']} | {fmt(r.get('avg_pnl_pct'))} | "
                        f"{fmt(r.get('median_pnl_pct'))} | {fmt_wr(r)} | "
                        f"{fmt(r.get('sharpe'), 3)} | {fmt(r.get('ci_95_low'))} | "
                        f"{fmt(r.get('ci_95_high'))} | {fmt(r.get('ci_confirmed'))} |")
report_lines.append(f"")

# 3. Catalyst Inclusion
report_lines.append(f"## 3. Catalyst Inclusion Expansion (B2)")
report_lines.append(f"")
report_lines.append(f"### Per-Catalyst Breakdown (SHORT gap-up >= 2%)")
report_lines.append(f"")
report_lines.append(f"| Catalyst | n | n_T+2 | Avg% | WR | Sharpe |")
report_lines.append(f"|----------|---|-------|------|-----|--------|")
for cat, r in sorted(catalyst_results.items(), key=lambda x: -(x[1].get('avg_pnl_pct') or -999)):
    report_lines.append(f"| {cat} | {r['n_total']} | {r['n_t2']} | "
                        f"{fmt(r.get('avg_pnl_pct'))} | {fmt_wr(r)} | "
                        f"{fmt(r.get('sharpe'), 3)} |")
report_lines.append(f"")

report_lines.append(f"### Catalyst Set Comparison")
report_lines.append(f"")
report_lines.append(f"| Set | Excludes | n | n_T+2 | t/d | Avg% | Sharpe | CI>0 |")
report_lines.append(f"|-----|----------|---|-------|-----|------|--------|------|")
for name, r in b2_set_results.items():
    excl = ', '.join(r.get('exclude', []))
    report_lines.append(f"| {name} | {excl} | {r['n_total']} | {r['n_t2']} | "
                        f"{round(r['n_total']/N_DATES, 1)} | {fmt(r.get('avg_pnl_pct'))} | "
                        f"{fmt(r.get('sharpe'), 3)} | {fmt(r.get('ci_confirmed'))} |")
report_lines.append(f"")

# 4. Scanner Split
report_lines.append(f"## 4. Discovery vs Premarket Scanner (B3)")
report_lines.append(f"")
report_lines.append(f"| Scanner | n | n_T+2 | Avg% | Med% | WR | Sharpe | CI_lo | CI_hi |")
report_lines.append(f"|---------|---|-------|------|------|-----|--------|-------|-------|")
for src, r in b3_results.items():
    report_lines.append(f"| {src} | {r['n_total']} | {r['n_t2']} | "
                        f"{fmt(r.get('avg_pnl_pct'))} | {fmt(r.get('median_pnl_pct'))} | "
                        f"{fmt_wr(r)} | "
                        f"{fmt(r.get('sharpe'), 3)} | {fmt(r.get('ci_95_low'))} | "
                        f"{fmt(r.get('ci_95_high'))} |")
report_lines.append(f"")

# 5. n_runs
report_lines.append(f"## 5. n_runs Signal Analysis (B4)")
report_lines.append(f"")
report_lines.append(f"| n_runs >= | n | n_T+2 | t/d | Avg% | WR | Sharpe | CI>0 |")
report_lines.append(f"|-----------|---|-------|-----|------|-----|--------|------|")
for nr, r in sorted(b4_results.items()):
    report_lines.append(f"| {nr} | {r['n_total']} | {r['n_t2']} | "
                        f"{round(r['n_total']/N_DATES, 1)} | {fmt(r.get('avg_pnl_pct'))} | "
                        f"{fmt_wr(r)} | "
                        f"{fmt(r.get('sharpe'), 3)} | {fmt(r.get('ci_confirmed'))} |")
report_lines.append(f"")

# 6. Tier System
report_lines.append(f"## 6. Tier System Simulation")
report_lines.append(f"")
report_lines.append(f"### Per-Day Selection (target: {TARGET_PER_DAY}/day)")
report_lines.append(f"")
for dt in TRADING_DATES:
    sim = tier_simulation[dt]
    report_lines.append(f"**{dt}** — {sim['total_qualified']} qualified "
                        f"(T1:{sim['available'][1]}, T2:{sim['available'][2]}, T3:{sim['available'][3]})")
    report_lines.append(f"")
    report_lines.append(f"| # | Tier | Ticker | Gap% | Catalyst | n_runs | T+2 PnL |")
    report_lines.append(f"|---|------|--------|------|----------|--------|---------|")
    for i, t in enumerate(sim['trades'], 1):
        t2 = f"{t['pnl_t2']:+.2f}%" if t['pnl_t2'] is not None else "no data"
        report_lines.append(f"| {i} | T{t['tier']} | {t['ticker']} | {t['gap_pct']:+.1f}% | "
                            f"{t['catalyst']} | {t['n_runs']} | {t2} |")
    report_lines.append(f"")

# Aggregate
report_lines.append(f"### Aggregate Tier Performance")
report_lines.append(f"")
if all_t2:
    report_lines.append(f"- Total selected: {n_selected_total} ({n_selected_total/N_DATES:.1f}/day)")
    report_lines.append(f"- With T+2 data: {len(all_t2)}")
    report_lines.append(f"- Avg T+2 PnL: {np.mean(all_t2):+.2f}%")
    report_lines.append(f"- Median T+2 PnL: {np.median(all_t2):+.2f}%")
    report_lines.append(f"- Win rate: {(np.array(all_t2) > 0).mean()*100:.0f}%")
report_lines.append(f"")

# 7. Final Spec
report_lines.append(f"## 7. FINAL AGENT 2 SPECIFICATION")
report_lines.append(f"")
report_lines.append(f"### Tier Criteria")
report_lines.append(f"")
report_lines.append(f"**Tier 1** (highest priority):")
report_lines.append(f"- LLM-flagged SHORT, gap-up >= 2%")
report_lines.append(f"- Catalyst: offering OR regulatory")
report_lines.append(f"- PM retrace: 0.2-0.8 (partial retrace sweet spot)")
report_lines.append(f"- Exclude: earnings, deal")
report_lines.append(f"")
report_lines.append(f"**Tier 2** (fill to 3/day):")
report_lines.append(f"- LLM-flagged SHORT, gap-up >= 2%")
report_lines.append(f"- Any catalyst except earnings, deal, breaking")
report_lines.append(f"- PM retrace < 0.8")
report_lines.append(f"")
report_lines.append(f"**Tier 3** (fallback only if T1+T2 < 3):")
report_lines.append(f"- LLM-flagged SHORT, gap-up >= 1.5%")
report_lines.append(f"- Exclude: earnings, deal")
report_lines.append(f"- Rank by n_runs × |gap_pct|")
report_lines.append(f"")

# 8. Logging
report_lines.append(f"## 8. Logging Schema Validation")
report_lines.append(f"")
report_lines.append(f"- Total fields: {len(TRADE_LOG_FIELDS)}")
report_lines.append(f"- Computable from backtest data: {len(computable)}")
report_lines.append(f"- Needs new data in live: {len(needs_new_data)}")
report_lines.append(f"- Live-only fields: {len(needs_live_only)}")
report_lines.append(f"")
if needs_new_data:
    report_lines.append(f"### Fields Needing New Data Pulls")
    for name, reason in needs_new_data:
        report_lines.append(f"- `{name}`: {reason}")
    report_lines.append(f"")
if needs_live_only:
    report_lines.append(f"### Live-Only Fields (expected)")
    for name, reason in needs_live_only:
        report_lines.append(f"- `{name}`: {reason}")
    report_lines.append(f"")

report_lines.append(f"## Multi-Hold Analysis")
report_lines.append(f"")
report_lines.append(f"| Hold | n | Avg% | Med% | WR | Sharpe | CI_lo | CI>0 |")
report_lines.append(f"|------|---|------|------|-----|--------|-------|------|")
for hold, m in multi_hold_results.items():
    report_lines.append(f"| {hold} | {m['n']} | {fmt(m.get('avg_pnl_pct'))} | "
                        f"{fmt(m.get('median_pnl_pct'))} | {fmt_wr(m)} | "
                        f"{fmt(m.get('sharpe'), 3)} | {fmt(m.get('ci_95_low'))} | "
                        f"{fmt(m.get('ci_confirmed'))} |")
report_lines.append(f"")

report_path = os.path.join(BASE_DIR, 'AGENT_2_EXPANSION_RESULTS.md')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"  Report saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  AGENT 2 EXPERIMENT COMPLETE")
print(f"{'='*_W}")
print(f"  Output files:")
print(f"    {results_path}")
print(f"    {report_path}")
print(f"    {chart1_path}")
print(f"    {chart2_path}")
print(f"    {chart3_path}")
print(f"    {chart4_path}")
print(f"{'='*_W}")
