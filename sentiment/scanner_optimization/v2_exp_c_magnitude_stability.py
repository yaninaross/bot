# %% EXP GROUP C: Magnitude (S5) + Stability (S6) + PM Shape (S7)
# Loads from v2_features_full.csv — pure DataFrame operations.

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
    from scipy import stats as _scipy_stats
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "--quiet"])
    from scipy import stats as _scipy_stats

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

def robust_stats(values):
    v = pd.Series(values).dropna(); n = len(v)
    if n == 0: return {'n': 0, 'avg': np.nan, 'med': np.nan, 'trim': np.nan, 'se': np.nan, 'wr': np.nan}
    avg = float(v.mean()); med = float(v.median()); se = float(v.std() / np.sqrt(n)) if n > 1 else 0.0
    wr = float((v > 0).sum() / n * 100)
    trim = float(v.sort_values().iloc[max(1, int(n*0.1)):-max(1, int(n*0.1))].mean()) if n >= 10 else avg
    return {'n': n, 'avg': avg, 'med': med, 'trim': trim, 'se': se, 'wr': wr}

def fold_robust(values, dates):
    v = pd.Series(values).dropna(); d = pd.Series(dates).loc[v.index]
    sa = robust_stats(v[d.isin(FOLD_A)]); sb = robust_stats(v[d.isin(FOLD_B)])
    cv = (sa['n'] > 0 and sb['n'] > 0 and sa['avg'] > 0 and sb['avg'] > 0)
    return {'all': robust_stats(v), 'A': sa, 'B': sb, 'cv': cv}

# ==============================================================================
# EXP S5: MAGNITUDE WITHOUT DIRECTION
# ==============================================================================
print(f"\n{'='*_W}")
print(f"  EXP S5: MAGNITUDE WITHOUT DIRECTION")
print(f"  Does |expected_move| predict |realized move|?")
print(f"{'='*_W}")

df['abs_pnl_60m'] = df['pnl_60m_nosl'].abs()
df['abs_expected'] = df['expected_move_pct'].abs()
df['abs_remaining'] = df['remaining_move_pct'].abs()
df['total_excursion'] = df['mfe'] + df['mae'].abs()

mag_pairs = [
    ('|expected_move|', 'abs_expected', '|realized_60m|', 'abs_pnl_60m'),
    ('|expected_move|', 'abs_expected', 'MFE+|MAE|', 'total_excursion'),
    ('|remaining_move|', 'abs_remaining', '|realized_60m|', 'abs_pnl_60m'),
    ('log_vol_ratio', 'log_vol_ratio', '|realized_60m|', 'abs_pnl_60m'),
    ('|gap_pct|', 'abs_gap_pct', '|realized_60m|', 'abs_pnl_60m'),
    ('confidence', 'confidence', '|realized_60m|', 'abs_pnl_60m'),
    ('atr_gap', 'atr_gap', '|realized_60m|', 'abs_pnl_60m'),
]

print(f"\n  Magnitude correlations (n={len(df)}):")
print(f"  {'':─<{_W-4}}")
print(f"  {'Predictor':<22} {'Target':<18} {'r':>7} {'p-value':>10}")
print(f"  {'':─<{_W-4}}")
for pred_lbl, pred_col, tgt_lbl, tgt_col in mag_pairs:
    v = df[[pred_col, tgt_col]].dropna()
    if len(v) > 5:
        r, p = _scipy_stats.pearsonr(v[pred_col], v[tgt_col])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {pred_lbl:<22} {tgt_lbl:<18} {r:>+6.3f} {p:>9.4f} {sig}")

# Tercile analysis: does high |expected_move| mean high realized volatility?
df['exp_tercile'] = pd.qcut(df['abs_expected'].rank(method='first'), 3, labels=['low', 'mid', 'high'])
print(f"\n  |Expected Move| Tercile Analysis:")
print(f"  {'Tercile':<10} {'n':>5} {'Avg |PnL|':>10} {'Avg MFE':>9} {'Avg |MAE|':>10} {'Avg Excursion':>14}")
for t in ['low', 'mid', 'high']:
    sub = df[df['exp_tercile'] == t]
    print(f"  {t:<10} {len(sub):>5} {sub['abs_pnl_60m'].mean():>+9.3f}% {sub['mfe'].mean():>+8.3f}% "
          f"{sub['mae'].abs().mean():>+9.3f}% {sub['total_excursion'].mean():>13.3f}%")

fig_s5, axes = plt.subplots(1, 3, figsize=(18, 5))
fig_s5.suptitle('EXP S5: Magnitude Without Direction', fontsize=13, fontweight='bold')

ax = axes[0]
v = df[['abs_expected', 'abs_pnl_60m']].dropna()
ax.scatter(v['abs_expected'], v['abs_pnl_60m'], alpha=0.4, s=30, c='#3498db')
if len(v) > 5:
    z = np.polyfit(v['abs_expected'], v['abs_pnl_60m'], 1)
    ax.plot([0, v['abs_expected'].max()], [z[1], z[1] + z[0] * v['abs_expected'].max()],
            'r--', lw=1.5, label=f'r={v["abs_expected"].corr(v["abs_pnl_60m"]):+.3f}')
ax.set_xlabel('|Expected Move| %'); ax.set_ylabel('|Realized Move @60m| %')
ax.set_title('Magnitude prediction'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
v2 = df[['abs_expected', 'total_excursion']].dropna()
ax.scatter(v2['abs_expected'], v2['total_excursion'], alpha=0.4, s=30, c='#9b59b6')
if len(v2) > 5:
    z2 = np.polyfit(v2['abs_expected'], v2['total_excursion'], 1)
    ax.plot([0, v2['abs_expected'].max()], [z2[1], z2[1] + z2[0] * v2['abs_expected'].max()],
            'r--', lw=1.5, label=f'r={v2["abs_expected"].corr(v2["total_excursion"]):+.3f}')
ax.set_xlabel('|Expected Move| %'); ax.set_ylabel('MFE + |MAE| %')
ax.set_title('Total excursion prediction'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
tercile_data = df.groupby('exp_tercile').agg(
    avg_abs_pnl=('abs_pnl_60m', 'mean'), avg_mfe=('mfe', 'mean'),
    avg_mae_abs=('mae', lambda x: x.abs().mean()), n=('abs_pnl_60m', 'count')).reindex(['low', 'mid', 'high'])
x_t = range(3)
ax.bar([x - 0.15 for x in x_t], tercile_data['avg_abs_pnl'], 0.3, label='|PnL|', color='#3498db')
ax.bar([x + 0.15 for x in x_t], tercile_data['avg_mfe'], 0.3, label='MFE', color='#2ecc71')
ax.set_xticks(x_t); ax.set_xticklabels(['Low |exp|', 'Mid', 'High |exp|'])
ax.set_ylabel('Avg %'); ax.set_title('Tercile breakdown'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig_s5.savefig(f'{OUTPUT_DIR}/v2_s5_magnitude.png', dpi=150, bbox_inches='tight'); plt.show()

# ==============================================================================
# EXP S6: STABILITY ACROSS RUNS
# ==============================================================================
print(f"\n\n{'='*_W}")
print(f"  EXP S6: STABILITY ACROSS RUNS")
print(f"{'='*_W}")

stable = df[df['stability_score'] >= 0.8]; unstable = df[df['stability_score'] < 0.8]
flipped = df[df['direction_flipped'] == True]; not_flipped = df[df['direction_flipped'] == False]

print(f"\n  Stability breakdown:")
for lbl, sub in [('Stable (score>=0.8)', stable), ('Unstable (score<0.8)', unstable),
                  ('Direction flipped', flipped), ('Not flipped', not_flipped)]:
    fr = fold_robust(sub['pnl_60m_nosl'], sub['eval_date']); s = fr['all']
    if s['n'] > 0:
        print(f"    {lbl:<25} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  "
              f"WR={s['wr']:.0f}%  CV:{'YES' if fr['cv'] else 'no'}")

for n_min in [2, 3, 4]:
    sub = df[df['n_runs'] >= n_min]
    if len(sub) > 0:
        fr = fold_robust(sub['pnl_60m_nosl'], sub['eval_date']); s = fr['all']
        print(f"    n_runs >= {n_min:<20} n={s['n']:>4}  avg={s['avg']:>+.3f}%  "
              f"med={s['med']:>+.3f}%  CV:{'YES' if fr['cv'] else 'no'}")

# Interaction: stability x direction
print(f"\n  Stability x Direction:")
for d in ['LONG', 'SHORT']:
    for lbl, cond in [('stable', df['stability_score'] >= 0.8), ('unstable', df['stability_score'] < 0.8)]:
        sub = df[(df['direction'] == d) & cond]
        if len(sub) > 3:
            s = robust_stats(sub['pnl_60m_nosl'])
            print(f"    {d} {lbl:<12} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  WR={s['wr']:.0f}%")

# ==============================================================================
# EXP S7: PREMARKET PATH SHAPE
# ==============================================================================
print(f"\n\n{'='*_W}")
print(f"  EXP S7: PREMARKET PATH SHAPE")
print(f"{'='*_W}")

df['retrace_bin'] = pd.cut(df['pm_retrace'], bins=[0, 0.2, 0.5, 0.8, 1.0],
                            labels=['held', 'mild', 'moderate', 'full_retrace'],
                            include_lowest=True)

print(f"\n  PM Retrace vs PnL @60m (no SL):")
for cat in ['held', 'mild', 'moderate', 'full_retrace']:
    sub = df[df['retrace_bin'] == cat]
    if len(sub) < 3: continue
    s = robust_stats(sub['pnl_60m_nosl'])
    print(f"    {cat:<20} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  WR={s['wr']:.0f}%")

df['pm_vol_late_bin'] = pd.cut(df['pm_vol_late_ratio'],
                                bins=[0, 0.3, 0.5, 0.7, 1.0],
                                labels=['early_vol', 'spread', 'late_heavy', 'very_late'],
                                include_lowest=True)
print(f"\n  Late PM Volume Ratio vs PnL:")
for cat in ['early_vol', 'spread', 'late_heavy', 'very_late']:
    sub = df[df['pm_vol_late_bin'] == cat]
    if len(sub) < 3: continue
    s = robust_stats(sub['pnl_60m_nosl'])
    print(f"    {cat:<20} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  WR={s['wr']:.0f}%")

pm_feats = ['pm_retrace', 'pm_time_since_high', 'pm_vol_late_ratio', 'pm_vol']
print(f"\n  PM feature correlations with PnL:")
for feat in pm_feats:
    v = df[[feat, 'pnl_60m_nosl']].dropna()
    if len(v) > 5:
        r, p = _scipy_stats.pearsonr(v[feat], v['pnl_60m_nosl'])
        print(f"    {feat:<25} r={r:+.3f}  p={p:.4f}")

# PM retrace x direction
print(f"\n  PM Retrace x Direction:")
for d in ['LONG', 'SHORT']:
    for cat in ['held', 'mild', 'moderate', 'full_retrace']:
        sub = df[(df['direction'] == d) & (df['retrace_bin'] == cat)]
        if len(sub) < 3: continue
        s = robust_stats(sub['pnl_60m_nosl'])
        print(f"    {d} {cat:<20} n={s['n']:>4}  avg={s['avg']:>+.3f}%  WR={s['wr']:.0f}%")

fig_s7, axes = plt.subplots(1, 2, figsize=(14, 5))
fig_s7.suptitle('EXP S7: Premarket Path Shape', fontsize=13, fontweight='bold')

ax = axes[0]
for cat, color in [('held', '#27ae60'), ('mild', '#3498db'), ('moderate', '#f39c12'), ('full_retrace', '#e74c3c')]:
    sub = df[df['retrace_bin'] == cat]
    if len(sub) > 2:
        ax.scatter(sub['pm_retrace'], sub['pnl_60m_nosl'], alpha=0.5, s=30, color=color, label=f'{cat} (n={len(sub)})')
ax.axhline(y=0, color='black', lw=0.8); ax.set_xlabel('PM Retrace')
ax.set_ylabel('PnL @60m %'); ax.set_title('Retrace vs PnL'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(df['pm_vol_late_ratio'], df['pnl_60m_nosl'], alpha=0.4, s=30, c='#9b59b6')
ax.axhline(y=0, color='black', lw=0.8); ax.set_xlabel('Late PM Vol Ratio')
ax.set_ylabel('PnL @60m %'); ax.set_title('Late Volume vs PnL'); ax.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.92])
fig_s7.savefig(f'{OUTPUT_DIR}/v2_s7_pm_shape.png', dpi=150, bbox_inches='tight'); plt.show()

print(f"\n  Group C results saved. DONE.")
