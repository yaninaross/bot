# %% EXP GROUP A: Stop Design (S1) + Horizon Extension (S2) + Version + Robustness
# Loads from v2_features_full.csv — run v2_data_prep.py first.

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
FOLD_A = ['2026-09-08', '2026-09-09']
FOLD_B = ['2026-09-10', '2026-09-11']
_W = 100

df = pd.read_csv(f'{RESULTS_DIR}/v2_features_full.csv')
print(f"Loaded {len(df)} trades")

# ── Robustness utilities ──
def robust_stats(values):
    v = pd.Series(values).dropna()
    n = len(v)
    if n == 0:
        return {'n': 0, 'avg': np.nan, 'med': np.nan, 'trim': np.nan,
                'se': np.nan, 'wr': np.nan, 'loo_min': np.nan, 'loo_max': np.nan}
    avg = float(v.mean()); med = float(v.median())
    se = float(v.std() / np.sqrt(n)) if n > 1 else 0.0
    wr = float((v > 0).sum() / n * 100)
    if n >= 10:
        k = max(1, int(n * 0.1)); trim = float(v.sort_values().iloc[k:-k].mean())
    else: trim = avg
    loo_min, loo_max = avg, avg
    if n < 50 and n > 1:
        loo_means = [(v.sum() - val) / (n - 1) for val in v]
        loo_min = min(loo_means); loo_max = max(loo_means)
    return {'n': n, 'avg': avg, 'med': med, 'trim': trim,
            'se': se, 'wr': wr, 'loo_min': loo_min, 'loo_max': loo_max}

def fold_robust(values, dates):
    v = pd.Series(values).dropna()
    d = pd.Series(dates).loc[v.index]
    sa = robust_stats(v[d.isin(FOLD_A)])
    sb = robust_stats(v[d.isin(FOLD_B)])
    cv = (sa['n'] > 0 and sb['n'] > 0 and sa['avg'] > 0 and sb['avg'] > 0)
    return {'all': robust_stats(v), 'A': sa, 'B': sb, 'cv': cv}

def print_robust_row(label, rs, fold_a_rs=None, fold_b_rs=None, cv=None):
    s = rs
    loo = f" LOO=[{s['loo_min']:+.2f},{s['loo_max']:+.2f}]" if s['n'] < 50 and s['n'] > 1 else ''
    fa = f"A={fold_a_rs['avg']:+.3f}%({fold_a_rs['n']})" if fold_a_rs and fold_a_rs['n'] > 0 else ''
    fb = f"B={fold_b_rs['avg']:+.3f}%({fold_b_rs['n']})" if fold_b_rs and fold_b_rs['n'] > 0 else ''
    print(f"    {label:<40} n={s['n']:>4}  avg={s['avg']:+.3f}%  med={s['med']:+.3f}%  "
          f"trim={s['trim']:+.3f}%  WR={s['wr']:.0f}%  SE={s['se']:.3f}  CV:{'YES' if cv else 'no '}  {fa}  {fb}{loo}")

# ==============================================================================
# EXP S1: STOP DESIGN — MAE/MFE Analysis
# ==============================================================================
print(f"\n{'='*_W}")
print(f"  EXP S1: STOP DESIGN — MAE/MFE ANALYSIS")
print(f"{'='*_W}")

winners = df[df['pnl_60m_nosl'] > 0]
losers = df[df['pnl_60m_nosl'] <= 0]

print(f"\n  MAE/MFE Summary (no SL, 60m horizon):")
print(f"  {'':─<{_W-4}}")
print(f"  {'':30} {'Winners':>12} {'Losers':>12} {'All':>12}")
print(f"  {'':─<{_W-4}}")
for stat, col in [('n', None), ('Avg MAE %', 'mae'), ('Med MAE %', 'mae'),
                   ('P80 MAE %', 'mae'), ('Avg MFE %', 'mfe'), ('Med MFE %', 'mfe')]:
    if col is None:
        print(f"  {stat:<30} {len(winners):>12} {len(losers):>12} {len(df):>12}")
    elif 'P80' in stat:
        w_v = winners[col].quantile(0.2); l_v = losers[col].quantile(0.2); a_v = df[col].quantile(0.2)
        print(f"  {stat:<30} {w_v:>+11.2f}% {l_v:>+11.2f}% {a_v:>+11.2f}%")
    elif 'Med' in stat:
        print(f"  {stat:<30} {winners[col].median():>+11.2f}% {losers[col].median():>+11.2f}% {df[col].median():>+11.2f}%")
    else:
        print(f"  {stat:<30} {winners[col].mean():>+11.2f}% {losers[col].mean():>+11.2f}% {df[col].mean():>+11.2f}%")

w_mae_p80 = abs(winners['mae'].quantile(0.2))
print(f"\n  Optimal stop (80th pct of winners' MAE): {w_mae_p80:.2f}%")
print(f"  In ATR units: {w_mae_p80 / df['atr_pct'].median():.2f} × ATR")

for d in ['LONG', 'SHORT']:
    w_d = winners[winners['direction'] == d]
    l_d = losers[losers['direction'] == d]
    if len(w_d) > 0:
        p80 = abs(w_d['mae'].quantile(0.2))
        print(f"    {d}: winners MAE P80 = {p80:.2f}%, n_win={len(w_d)}, n_lose={len(l_d)}")

# Stop grid
print(f"\n  Stop Grid Results @60m:")
print(f"  {'':─<{_W-4}}")
print(f"  {'Stop Level':<25} {'n':>5} {'Avg':>8} {'Med':>8} {'Trim':>8} {'WR':>6} {'SE':>6} {'CV':>4} {'A':>8} {'B':>8}")
print(f"  {'':─<{_W-4}}")

sl_cols = {}
for sl in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    col = f'pnl_60m_sl{sl}'
    if col in df.columns: sl_cols[f'Fixed {sl}%'] = col
for k in [0.25, 0.5, 0.75, 1.0]:
    col = f'pnl_60m_atr{k}'
    if col in df.columns: sl_cols[f'ATR×{k}'] = col
sl_cols['No SL'] = 'pnl_60m_nosl'
for sl in [2.5, 4.0, 5.0]:
    col = f'pnl_60m_t5_sl{sl}'
    if col in df.columns: sl_cols[f'T+5 Fixed {sl}%'] = col
for k in [0.5, 0.75]:
    col = f'pnl_60m_t5_atr{k}'
    if col in df.columns: sl_cols[f'T+5 ATR×{k}'] = col

sl_summary = []
for label, col in sl_cols.items():
    vals = df[col].dropna()
    if len(vals) == 0: continue
    fr = fold_robust(df[col], df['eval_date'])
    s = fr['all']
    print(f"  {label:<25} {s['n']:>5} {s['avg']:>+7.3f}% {s['med']:>+7.3f}% {s['trim']:>+7.3f}% "
          f"{s['wr']:>5.0f}% {s['se']:>.3f} {'YES' if fr['cv'] else 'no ':>3} "
          f"{fr['A']['avg']:>+7.3f}% {fr['B']['avg']:>+7.3f}%")
    sl_summary.append({'label': label, 'col': col, **fr})

# SHORT-only stop grid
df_short = df[df['direction'] == 'SHORT']
print(f"\n  SHORT-only Stop Grid @60m (n={len(df_short)}):")
for label, col in sl_cols.items():
    if 'T+5' in label: continue
    vals = df_short[col].dropna()
    if len(vals) == 0: continue
    fr = fold_robust(df_short[col], df_short['eval_date'])
    s = fr['all']
    print(f"  {label:<25} {s['n']:>5} {s['avg']:>+7.3f}% {s['med']:>+7.3f}% {s['trim']:>+7.3f}% "
          f"{s['wr']:>5.0f}% {'YES' if fr['cv'] else 'no ':>3}")

# Chart
fig_s1, axes = plt.subplots(2, 2, figsize=(16, 12))
fig_s1.suptitle('EXP S1: Stop Design — MAE/MFE Analysis', fontsize=14, fontweight='bold')

ax = axes[0, 0]
bins_mae = np.linspace(-15, 0, 40)
ax.hist(winners['mae'], bins=bins_mae, alpha=0.7, color='#2ecc71', label=f'Winners (n={len(winners)})')
ax.hist(losers['mae'], bins=bins_mae, alpha=0.5, color='#e74c3c', label=f'Losers (n={len(losers)})')
ax.axvline(x=-w_mae_p80, color='blue', lw=2, ls='--', label=f'P80 stop = {w_mae_p80:.1f}%')
ax.axvline(x=-2.5, color='orange', lw=1.5, ls=':', label='Current SL=2.5%')
ax.set_xlabel('Max Adverse Excursion %'); ax.set_title('MAE: Winners vs Losers')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.scatter(df['mae'], df['mfe'], c=['#2ecc71' if p > 0 else '#e74c3c' for p in df['pnl_60m_nosl']], alpha=0.5, s=30)
ax.set_xlabel('MAE %'); ax.set_ylabel('MFE %'); ax.set_title('MAE vs MFE (green=winner)')
ax.axvline(x=-2.5, color='orange', lw=1, ls=':'); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
sl_labels = [l for l in sl_cols.keys() if 'T+5' not in l]
sl_avgs = [df[sl_cols[l]].dropna().mean() for l in sl_labels]
sl_meds = [df[sl_cols[l]].dropna().median() for l in sl_labels]
x_sl = range(len(sl_labels))
ax.bar([x - 0.15 for x in x_sl], sl_avgs, 0.3, label='Mean', color='#3498db', alpha=0.8)
ax.bar([x + 0.15 for x in x_sl], sl_meds, 0.3, label='Median', color='#9b59b6', alpha=0.7)
ax.set_xticks(x_sl); ax.set_xticklabels(sl_labels, rotation=45, fontsize=7)
ax.axhline(y=0, color='black', lw=0.8); ax.set_ylabel('PnL @60m %')
ax.set_title('Stop Level Comparison'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
for d, c in [('LONG', '#27ae60'), ('SHORT', '#c0392b')]:
    d_df = df[df['direction'] == d]
    d_w = d_df[d_df['pnl_60m_nosl'] > 0]
    if len(d_w) > 0:
        mae_sorted = np.sort(d_w['mae'].values)
        pcts = np.arange(1, len(mae_sorted) + 1) / len(mae_sorted) * 100
        ax.plot(mae_sorted, pcts, color=c, lw=2, label=f'{d} winners (n={len(d_w)})')
ax.axhline(y=80, color='gray', ls='--', lw=0.8, label='80th percentile')
ax.set_xlabel('MAE %'); ax.set_ylabel('Cumulative % of winners')
ax.set_title('Winner MAE CDF — where to set stop'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_s1.savefig(f'{OUTPUT_DIR}/v2_s1_stop_design.png', dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# EXP S2: HORIZON EXTENSION
# ==============================================================================
print(f"\n\n{'='*_W}")
print(f"  EXP S2: HORIZON EXTENSION — Is this a scalp or a swing signal?")
print(f"{'='*_W}")

horizon_cols = ['pnl_30m_nosl', 'pnl_60m_nosl', 'pnl_120m_nosl', 'pnl_eod_nosl']
horizon_labels = ['30m', '60m', '2h', 'EOD']
for i in range(1, 6):
    col = f'pnl_t{i}_close'
    if col in df.columns and df[col].notna().sum() > 10:
        horizon_cols.append(col); horizon_labels.append(f'T+{i}')

print(f"\n  PnL by Horizon (no SL, all signals):")
print(f"  {'':─<{_W-4}}")
print(f"  {'Horizon':<10} {'n':>5} {'Avg':>8} {'Med':>8} {'Trim':>8} {'WR':>6} {'SE':>6} {'CV':>4}")
print(f"  {'':─<{_W-4}}")
for lbl, col in zip(horizon_labels, horizon_cols):
    fr = fold_robust(df[col], df['eval_date'])
    s = fr['all']
    print(f"  {lbl:<10} {s['n']:>5} {s['avg']:>+7.3f}% {s['med']:>+7.3f}% {s['trim']:>+7.3f}% "
          f"{s['wr']:>5.0f}% {s['se']:>.3f} {'YES' if fr['cv'] else 'no ':>3}")

feat_cols = ['remaining_move_pct', 'expected_move_pct', 'log_vol_ratio', 'gap_pct', 'confidence', 'atr_gap']
print(f"\n  Feature Correlation vs Horizon (Pearson r):")
print(f"  {'':─<{_W-4}}")
hdr = f"  {'Feature':<22}" + "".join(f" {l:>8}" for l in horizon_labels)
print(hdr); print(f"  {'':─<{_W-4}}")
corr_data = {}
for feat in feat_cols:
    if feat not in df.columns: continue
    corrs = []
    for col in horizon_cols:
        v = df[[feat, col]].dropna()
        corrs.append(v[feat].corr(v[col]) if len(v) > 5 else np.nan)
    corr_data[feat] = corrs
    row_str = f"  {feat:<22}" + "".join(f" {c:>+7.3f}%" if pd.notna(c) else " " * 9 for c in corrs)
    print(row_str)

df_s = df[df['direction'] == 'SHORT']
print(f"\n  SHORT-only by Horizon (no SL, n={len(df_s)}):")
for lbl, col in zip(horizon_labels, horizon_cols):
    vals = df_s[col].dropna()
    if len(vals) > 0:
        fr = fold_robust(df_s[col], df_s['eval_date']); s = fr['all']
        print(f"    {lbl:<10} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  WR={s['wr']:.0f}%  CV:{'YES' if fr['cv'] else 'no'}")

fig_s2, axes = plt.subplots(1, 3, figsize=(18, 5))
fig_s2.suptitle('EXP S2: Horizon Extension', fontsize=13, fontweight='bold')
ax = axes[0]
avgs = [df[c].dropna().mean() for c in horizon_cols]; meds = [df[c].dropna().median() for c in horizon_cols]
x_h = range(len(horizon_labels))
ax.plot(x_h, avgs, 'o-', color='#3498db', lw=2, label='Mean'); ax.plot(x_h, meds, 's--', color='#9b59b6', lw=1.5, label='Median')
ax.set_xticks(x_h); ax.set_xticklabels(horizon_labels); ax.axhline(y=0, color='black', lw=0.8)
ax.set_ylabel('PnL %'); ax.set_title('All signals by horizon'); ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[1]
for d, c in [('LONG', '#27ae60'), ('SHORT', '#c0392b')]:
    d_df = df[df['direction'] == d]; avgs_d = [d_df[col].dropna().mean() for col in horizon_cols]
    ax.plot(x_h, avgs_d, 'o-', color=c, lw=2, label=d)
ax.set_xticks(x_h); ax.set_xticklabels(horizon_labels); ax.axhline(y=0, color='black', lw=0.8)
ax.set_ylabel('Avg PnL %'); ax.set_title('By direction'); ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[2]
for feat in ['remaining_move_pct', 'expected_move_pct', 'log_vol_ratio']:
    if feat in corr_data: ax.plot(x_h[:len(corr_data[feat])], corr_data[feat], 'o-', lw=1.5, label=feat)
ax.set_xticks(x_h); ax.set_xticklabels(horizon_labels); ax.axhline(y=0, color='black', lw=0.8)
ax.set_ylabel('Correlation (r)'); ax.set_title('Feature correlation vs horizon'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.92])
fig_s2.savefig(f'{OUTPUT_DIR}/v2_s2_horizon.png', dpi=150, bbox_inches='tight'); plt.show()

# ==============================================================================
# VERSION BREAKDOWN
# ==============================================================================
print(f"\n\n{'='*_W}")
print(f"  VERSION BREAKDOWN — Is fold-CV actually version-CV?")
print(f"{'='*_W}")
versions = df['scanner_version'].unique()
print(f"\n  Versions: {sorted(versions)}")
print(f"\n  PnL by scanner version (no SL, 60m):")
for ver in sorted(versions):
    sub = df[df['scanner_version'] == ver]
    if len(sub) < 3: continue
    s = robust_stats(sub['pnl_60m_nosl']); dates = sorted(sub['eval_date'].unique())
    print(f"    {ver:<12} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  WR={s['wr']:.0f}%  dates={','.join(d[-5:] for d in dates)}")

print(f"\n  Direction x Version:")
for ver in sorted(versions):
    for d in ['LONG', 'SHORT']:
        sub = df[(df['scanner_version'] == ver) & (df['direction'] == d)]
        if len(sub) < 3: continue
        s = robust_stats(sub['pnl_60m_nosl'])
        print(f"    {ver:<12} {d:<6} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  WR={s['wr']:.0f}%")

# ==============================================================================
# ROBUSTNESS RE-EVALUATION
# ==============================================================================
print(f"\n\n{'='*_W}")
print(f"  ROBUSTNESS RE-EVALUATION — V1 Findings with median/trimmed/LOO")
print(f"{'='*_W}")

good_cats = ['offering', 'regulatory', 'squeeze', 'secondary']
df['good_cat'] = df['catalyst_type'].str.lower().apply(lambda x: any(c in str(x) for c in good_cats))
short_gc = df[(df['direction'] == 'SHORT') & df['good_cat']]

print(f"\n  SHORT + good catalyst, no SL @60m:")
print_robust_row('SHORT+goodcat noSL', robust_stats(short_gc['pnl_60m_nosl']),
                 robust_stats(short_gc[short_gc['eval_date'].isin(FOLD_A)]['pnl_60m_nosl']),
                 robust_stats(short_gc[short_gc['eval_date'].isin(FOLD_B)]['pnl_60m_nosl']),
                 fold_robust(short_gc['pnl_60m_nosl'], short_gc['eval_date'])['cv'])

if len(short_gc) > 1:
    vals = short_gc['pnl_60m_nosl'].dropna().sort_values()
    top_trade = vals.iloc[-1]; without_top = vals.iloc[:-1]
    s_wt = robust_stats(without_top)
    print(f"    WITHOUT top trade ({top_trade:+.1f}%): avg={s_wt['avg']:+.3f}%  med={s_wt['med']:+.3f}%  n={s_wt['n']}")

df_short_all = df[df['direction'] == 'SHORT']
print(f"\n  SHORT only, no SL @60m:")
print_robust_row('SHORT noSL', robust_stats(df_short_all['pnl_60m_nosl']),
                 robust_stats(df_short_all[df_short_all['eval_date'].isin(FOLD_A)]['pnl_60m_nosl']),
                 robust_stats(df_short_all[df_short_all['eval_date'].isin(FOLD_B)]['pnl_60m_nosl']),
                 fold_robust(df_short_all['pnl_60m_nosl'], df_short_all['eval_date'])['cv'])

print(f"\n  All signals baseline @60m (SL=2.5%):")
col25 = 'pnl_60m_sl2.5' if 'pnl_60m_sl2.5' in df.columns else 'pnl_60m_sl2_5'
if col25 not in df.columns:
    for c in df.columns:
        if 'pnl_60m_sl2' in c: col25 = c; break
if col25 in df.columns:
    print_robust_row('All SL=2.5%', robust_stats(df[col25]),
                     robust_stats(df[df['eval_date'].isin(FOLD_A)][col25]),
                     robust_stats(df[df['eval_date'].isin(FOLD_B)][col25]),
                     fold_robust(df[col25], df['eval_date'])['cv'])

print(f"\n  All signals baseline @60m (no SL):")
print_robust_row('All noSL', robust_stats(df['pnl_60m_nosl']),
                 robust_stats(df[df['eval_date'].isin(FOLD_A)]['pnl_60m_nosl']),
                 robust_stats(df[df['eval_date'].isin(FOLD_B)]['pnl_60m_nosl']),
                 fold_robust(df['pnl_60m_nosl'], df['eval_date'])['cv'])

# Save summary
results = {'experiment': 'group_a_stops_horizon',
           'best_stop': max(sl_summary, key=lambda x: x['all']['avg'])['label'] if sl_summary else 'N/A'}
with open(f'{RESULTS_DIR}/v2_group_a_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Group A results saved. DONE.")
