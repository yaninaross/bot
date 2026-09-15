# %% EXP GROUP D: ORB (S8) + Ticker History (S9) + Earnings (S10)
# Loads from v2_features_full.csv + v2_daily_data.json

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

try:
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    OUTPUT_DIR = os.getcwd()
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
ALL_DATES = ['2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11']
FOLD_A = ['2026-09-08', '2026-09-09']
FOLD_B = ['2026-09-10', '2026-09-11']
_W = 100

df = pd.read_csv(f'{RESULTS_DIR}/v2_features_full.csv')
daily_data = {}
if os.path.exists(f'{RESULTS_DIR}/v2_daily_data.json'):
    with open(f'{RESULTS_DIR}/v2_daily_data.json') as f: daily_data = json.load(f)
print(f"Loaded {len(df)} trades, {len(daily_data)} tickers daily data")

def robust_stats(values):
    v = pd.Series(values).dropna(); n = len(v)
    if n == 0: return {'n': 0, 'avg': np.nan, 'med': np.nan, 'trim': np.nan, 'se': np.nan, 'wr': np.nan,
                        'loo_min': np.nan, 'loo_max': np.nan}
    avg = float(v.mean()); med = float(v.median()); se = float(v.std() / np.sqrt(n)) if n > 1 else 0.0
    wr = float((v > 0).sum() / n * 100)
    trim = float(v.sort_values().iloc[max(1, int(n*0.1)):-max(1, int(n*0.1))].mean()) if n >= 10 else avg
    loo_min, loo_max = avg, avg
    if n < 50 and n > 1:
        loo_means = [(v.sum() - val) / (n - 1) for val in v]
        loo_min = min(loo_means); loo_max = max(loo_means)
    return {'n': n, 'avg': avg, 'med': med, 'trim': trim, 'se': se, 'wr': wr,
            'loo_min': loo_min, 'loo_max': loo_max}

def fold_robust(values, dates):
    v = pd.Series(values).dropna(); d = pd.Series(dates).loc[v.index]
    sa = robust_stats(v[d.isin(FOLD_A)]); sb = robust_stats(v[d.isin(FOLD_B)])
    cv = (sa['n'] > 0 and sb['n'] > 0 and sa['avg'] > 0 and sb['avg'] > 0)
    return {'all': robust_stats(v), 'A': sa, 'B': sb, 'cv': cv}

# ==============================================================================
# EXP S8: OPENING RANGE BREAKOUT
# ==============================================================================
print(f"\n{'='*_W}")
print(f"  EXP S8: OPENING RANGE BREAKOUT")
print(f"  Direction from price, catalyst qualifies which names.")
print(f"{'='*_W}")

df_orb = df[df['orb_dir'].notna()].copy()
df_no_orb = df[df['orb_dir'].isna()].copy()

print(f"\n  ORB triggered: {len(df_orb)}/{len(df)} trades")
print(f"  ORB direction matches LLM: {(df_orb['orb_dir'] == df_orb['direction']).sum()}/{len(df_orb)}")

if len(df_orb) > 0:
    orb_s = robust_stats(df_orb['orb_pnl_60m'])
    print(f"\n  ORB PnL @60m (OR as natural stop):")
    print(f"    All ORB:    n={orb_s['n']}  avg={orb_s['avg']:+.3f}%  med={orb_s['med']:+.3f}%  WR={orb_s['wr']:.0f}%")

    llm_on_orb = robust_stats(df_orb['pnl_60m_nosl'])
    print(f"    LLM dir:    n={llm_on_orb['n']}  avg={llm_on_orb['avg']:+.3f}%  med={llm_on_orb['med']:+.3f}%  WR={llm_on_orb['wr']:.0f}%")

    agree = df_orb[df_orb['orb_dir'] == df_orb['direction']]
    disagree = df_orb[df_orb['orb_dir'] != df_orb['direction']]
    if len(agree) > 0:
        s_ag = robust_stats(agree['pnl_60m_nosl'])
        print(f"    ORB agrees:    n={s_ag['n']}  avg={s_ag['avg']:+.3f}%  WR={s_ag['wr']:.0f}%")
    if len(disagree) > 0:
        s_dis = robust_stats(disagree['pnl_60m_nosl'])
        print(f"    ORB disagrees: n={s_dis['n']}  avg={s_dis['avg']:+.3f}%  WR={s_dis['wr']:.0f}%")

    # ORB direction by itself vs LLM direction
    print(f"\n  ORB as standalone system (direction from price, not LLM):")
    for d in ['LONG', 'SHORT']:
        sub = df_orb[df_orb['orb_dir'] == d]
        if len(sub) > 0:
            s = robust_stats(sub['orb_pnl_60m'])
            print(f"    ORB {d}: n={s['n']}  avg={s['avg']:+.3f}%  med={s['med']:+.3f}%  WR={s['wr']:.0f}%")

    # ORB with LLM as qualifier: only trade ORB when it agrees with LLM
    if len(agree) > 0:
        s_agree_orb = robust_stats(agree['orb_pnl_60m'])
        print(f"\n  ORB + LLM agreement (ORB direction, LLM qualifier):")
        print(f"    n={s_agree_orb['n']}  avg={s_agree_orb['avg']:+.3f}%  med={s_agree_orb['med']:+.3f}%  WR={s_agree_orb['wr']:.0f}%")

    # Chart
    fig_s8, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig_s8.suptitle('EXP S8: Opening Range Breakout', fontsize=13, fontweight='bold')
    ax = axes[0]
    agree_pnl = agree['pnl_60m_nosl'].dropna() if len(agree) > 0 else pd.Series()
    disagree_pnl = disagree['pnl_60m_nosl'].dropna() if len(disagree) > 0 else pd.Series()
    bins_orb = np.linspace(-10, 10, 25)
    if len(agree_pnl) > 0:
        ax.hist(agree_pnl, bins=bins_orb, alpha=0.7, color='#2ecc71', label=f'ORB=LLM (n={len(agree_pnl)})')
    if len(disagree_pnl) > 0:
        ax.hist(disagree_pnl, bins=bins_orb, alpha=0.5, color='#e74c3c', label=f'ORB≠LLM (n={len(disagree_pnl)})')
    ax.axvline(x=0, color='black', lw=0.8); ax.set_xlabel('PnL @60m %')
    ax.set_title('LLM PnL: ORB agrees vs disagrees'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    orb_times = df_orb['orb_entry_time'].dropna()
    if len(orb_times) > 0:
        ax.hist(orb_times, bins=20, color='#3498db', alpha=0.7)
        ax.set_xlabel('Minutes after open'); ax.set_ylabel('Count')
        ax.set_title('ORB breakout timing'); ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig_s8.savefig(f'{OUTPUT_DIR}/v2_s8_orb.png', dpi=150, bbox_inches='tight'); plt.show()

# ==============================================================================
# EXP S9: TICKER REACTION HISTORY
# ==============================================================================
print(f"\n\n{'='*_W}")
print(f"  EXP S9: TICKER REACTION HISTORY")
print(f"  Per-ticker gap continuation vs fade tendency")
print(f"{'='*_W}")

all_tickers = sorted(df['ticker'].unique())
ticker_history = {}
for tk in all_tickers:
    tk_d = daily_data.get(tk, {})
    dates_sorted = sorted(tk_d.keys())
    if len(dates_sorted) < 10: continue
    gaps = []
    for i in range(1, len(dates_sorted)):
        d = dates_sorted[i]; prev_d = dates_sorted[i - 1]
        if d in ALL_DATES: continue
        prev_c = tk_d[prev_d]['close']; today_o = tk_d[d]['open']; today_c = tk_d[d]['close']
        if prev_c <= 0 or today_o <= 0: continue
        gap = (today_o - prev_c) / prev_c * 100
        if abs(gap) < 2.0: continue
        continuation = (today_c > today_o) if gap > 0 else (today_c < today_o)
        gaps.append({'gap': gap, 'continuation': continuation,
                     'day_ret': (today_c - today_o) / today_o * 100})
    if len(gaps) >= 3:
        cont_rate = sum(1 for g in gaps if g['continuation']) / len(gaps)
        ticker_history[tk] = {'n_gaps': len(gaps), 'continuation_rate': cont_rate,
                              'avg_fade_return': np.mean([-g['day_ret'] * np.sign(g['gap']) for g in gaps])}

print(f"\n  Computed gap history for {len(ticker_history)}/{len(all_tickers)} tickers")

df['hist_cont_rate'] = df['ticker'].map(lambda t: ticker_history.get(t, {}).get('continuation_rate', np.nan))
df['hist_n_gaps'] = df['ticker'].map(lambda t: ticker_history.get(t, {}).get('n_gaps', 0))

has_hist = df[df['hist_cont_rate'].notna()]
if len(has_hist) > 10:
    faders = has_hist[has_hist['hist_cont_rate'] < 0.4]
    runners = has_hist[has_hist['hist_cont_rate'] > 0.6]
    print(f"\n  Tickers that historically FADE gaps (cont_rate < 40%):")
    if len(faders) > 0:
        s = robust_stats(faders['pnl_60m_nosl'])
        print(f"    n={s['n']}  avg={s['avg']:+.3f}%  med={s['med']:+.3f}%  WR={s['wr']:.0f}%")
        if len(faders) < 20:
            print(f"    Tickers: {', '.join(sorted(faders['ticker'].unique()))}")
    print(f"  Tickers that historically RUN (cont_rate > 60%):")
    if len(runners) > 0:
        s = robust_stats(runners['pnl_60m_nosl'])
        print(f"    n={s['n']}  avg={s['avg']:+.3f}%  med={s['med']:+.3f}%  WR={s['wr']:.0f}%")

    r_hist = has_hist['hist_cont_rate'].corr(has_hist['pnl_60m_nosl'])
    print(f"\n  Correlation: hist_cont_rate vs PnL @60m: r={r_hist:+.3f}")

    # Interaction: SHORT on historical faders
    short_faders = faders[faders['direction'] == 'SHORT']
    if len(short_faders) > 0:
        s = robust_stats(short_faders['pnl_60m_nosl'])
        print(f"\n  SHORT on historical faders: n={s['n']}  avg={s['avg']:+.3f}%  med={s['med']:+.3f}%  WR={s['wr']:.0f}%")

# ==============================================================================
# EXP S10: EARNINGS MODULE
# ==============================================================================
print(f"\n\n{'='*_W}")
print(f"  EXP S10: EARNINGS MODULE")
print(f"{'='*_W}")

df_earn = df[df['is_earnings'] == True].copy()
df_non_earn = df[df['is_earnings'] == False].copy()

print(f"\n  Earnings trades: {len(df_earn)}, Non-earnings: {len(df_non_earn)}")

if len(df_earn) > 0:
    for lbl, sub in [('Earnings', df_earn), ('Non-earnings', df_non_earn)]:
        s = robust_stats(sub['pnl_60m_nosl'])
        print(f"    {lbl:<20} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  WR={s['wr']:.0f}%")

    beats = df_earn[df_earn['cat_beat'] & ~df_earn['cat_miss']]
    misses = df_earn[df_earn['cat_miss'] & ~df_earn['cat_beat']]
    ambig = df_earn[~df_earn['cat_beat'] & ~df_earn['cat_miss']]

    print(f"\n  Beat / Miss / Ambiguous:")
    for lbl, sub in [('Beat', beats), ('Miss', misses), ('Ambiguous', ambig)]:
        if len(sub) > 0:
            s = robust_stats(sub['pnl_60m_nosl'])
            print(f"    {lbl:<20} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  WR={s['wr']:.0f}%")

    has_rule = df_earn[df_earn['earnings_rule_dir'].notna()]
    if len(has_rule) > 0:
        agrees = has_rule[has_rule['earnings_rule_dir'] == has_rule['direction']]
        disagrees = has_rule[has_rule['earnings_rule_dir'] != has_rule['direction']]
        print(f"\n  Earnings rule direction vs LLM:")
        print(f"    Rule agrees with LLM: {len(agrees)}/{len(has_rule)}")
        for lbl, sub in [('Agrees', agrees), ('Disagrees', disagrees)]:
            if len(sub) > 0:
                s = robust_stats(sub['pnl_60m_nosl'])
                print(f"      {lbl}: n={s['n']}  avg={s['avg']:+.3f}%  WR={s['wr']:.0f}%")

    h_cols = ['pnl_60m_nosl', 'pnl_eod_nosl'] + [f'pnl_t{i}_close' for i in range(1, 4)]
    h_labels = ['60m', 'EOD', 'T+1', 'T+2', 'T+3']
    print(f"\n  Earnings by horizon:")
    for lbl, col in zip(h_labels, h_cols):
        if col in df_earn.columns:
            vals = df_earn[col].dropna()
            if len(vals) > 0:
                s = robust_stats(vals)
                print(f"    {lbl:<8} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%")

    # Earnings: direction breakdown
    print(f"\n  Earnings x Direction:")
    for d in ['LONG', 'SHORT']:
        sub = df_earn[df_earn['direction'] == d]
        if len(sub) > 0:
            s = robust_stats(sub['pnl_60m_nosl'])
            print(f"    {d}: n={s['n']}  avg={s['avg']:+.3f}%  med={s['med']:+.3f}%  WR={s['wr']:.0f}%")

    # Chart
    fig_s10, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig_s10.suptitle('EXP S10: Earnings Module', fontsize=13, fontweight='bold')

    ax = axes[0]
    cats = ['Beat', 'Miss', 'Ambiguous', 'Non-earnings']
    subs = [beats, misses, ambig, df_non_earn]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
    avgs = [robust_stats(s['pnl_60m_nosl'])['avg'] for s in subs]
    meds = [robust_stats(s['pnl_60m_nosl'])['med'] for s in subs]
    x_e = range(len(cats))
    ax.bar([x - 0.15 for x in x_e], avgs, 0.3, label='Mean', color=[c for c in colors], alpha=0.8)
    ax.bar([x + 0.15 for x in x_e], meds, 0.3, label='Median', color=[c for c in colors], alpha=0.4,
           edgecolor=[c for c in colors], lw=1.5)
    ax.set_xticks(x_e); ax.set_xticklabels(cats)
    ax.axhline(y=0, color='black', lw=0.8); ax.set_ylabel('PnL @60m %')
    ax.set_title('Earnings Categories'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    h_avgs_e = [robust_stats(df_earn[c])['avg'] for c in h_cols if c in df_earn.columns]
    h_avgs_ne = [robust_stats(df_non_earn[c])['avg'] for c in h_cols if c in df_non_earn.columns]
    h_labels_avail = [l for l, c in zip(h_labels, h_cols) if c in df.columns]
    x_h = range(len(h_labels_avail))
    ax.plot(x_h[:len(h_avgs_e)], h_avgs_e, 'o-', color='#e74c3c', lw=2, label='Earnings')
    ax.plot(x_h[:len(h_avgs_ne)], h_avgs_ne, 's--', color='#95a5a6', lw=1.5, label='Non-earnings')
    ax.set_xticks(x_h); ax.set_xticklabels(h_labels_avail)
    ax.axhline(y=0, color='black', lw=0.8); ax.set_ylabel('Avg PnL %')
    ax.set_title('Earnings vs Non-earnings by Horizon'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig_s10.savefig(f'{OUTPUT_DIR}/v2_s10_earnings.png', dpi=150, bbox_inches='tight'); plt.show()

# ==============================================================================
# SUMMARY DASHBOARD
# ==============================================================================
print(f"\n\n{'='*_W}")
print(f"  GROUP D SUMMARY")
print(f"{'='*_W}")

findings = []
if len(df_orb) > 0:
    orb_avg = df_orb['orb_pnl_60m'].dropna().mean()
    findings.append(f"S8: ORB avg = {orb_avg:+.3f}% (n={len(df_orb)})")
    if len(agree) > 0:
        findings.append(f"S8: ORB+LLM agree avg = {robust_stats(agree['pnl_60m_nosl'])['avg']:+.3f}% (n={len(agree)})")
if len(has_hist) > 10:
    findings.append(f"S9: hist_cont_rate → PnL r={r_hist:+.3f}")
if len(df_earn) > 0:
    findings.append(f"S10: Earnings avg = {robust_stats(df_earn['pnl_60m_nosl'])['avg']:+.3f}% (n={len(df_earn)})")

for f in findings:
    print(f"    {f}")

print(f"\n  Group D DONE.")
