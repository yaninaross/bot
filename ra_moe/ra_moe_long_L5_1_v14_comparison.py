# %% ── Compare LONG L5-1 v14 vs v1 Optimizer Results ────────────────────────────
#
# Pulls both model_versions from BQ and compares per-ticker:
#   - TEST PnL change, win rate change, trade count change
#   - Which tickers gained/lost deployability
#   - Which modes (static/grace/limit_entry/trailing) won
#   - Overfit ratio comparison
#   - v14-only features: exit time, stability, overfit gate
#
# No data files needed — everything from BQ.
# Colab-ready.

import pandas as pd
import numpy as np
from google.cloud import bigquery as bq

try:
    from google.colab import auth
    _IN_COLAB = True
    auth.authenticate_user()
except (ImportError, Exception):
    _IN_COLAB = False

import matplotlib
if not _IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIG
# ==============================================================================
FOLD_ID = 7

V1_MODEL_VERSION  = f'spike_transformer_long_L5_1_fold{FOLD_ID}'
V14_MODEL_VERSION = f'spike_transformer_long_L5_1_v14_fold{FOLD_ID}'

BQ_TABLE_ID = "ys-time-series.scoring_logs.wf_optimization_results"
DEPLOY_PNL_THRESHOLD = 0.018

CHART_DIR = '/tmp/ra_moe_long_L5_1_models'

W = 100

# ==============================================================================
# LOAD FROM BQ
# ==============================================================================
client = bq.Client()

query = f"""
SELECT
    ticker, tpsl_mode, model_version,
    prob_thresh, tp_pct, sl_pct,
    opt_avg_pnl, opt_win_rate, opt_count,
    val_a_avg_pnl, val_a_win_rate, val_a_count, survived_a,
    val_b_avg_pnl, val_b_win_rate, val_b_count, survived_b,
    experiment,
    grace_minutes, dip_pct, max_wait_minutes,
    exit_time_min_used, trail_trigger_pct, trail_pct_pct,
    neighbor_stability, overfit_ratio
FROM `{BQ_TABLE_ID}`
WHERE model_version IN ('{V1_MODEL_VERSION}', '{V14_MODEL_VERSION}')
ORDER BY ticker, tpsl_mode
"""

print(f"Loading BQ results for fold {FOLD_ID}...")
df_all = client.query(query).to_dataframe()
print(f"  Total rows: {len(df_all)}")

df_v1  = df_all[df_all['model_version'] == V1_MODEL_VERSION].copy()
df_v14 = df_all[df_all['model_version'] == V14_MODEL_VERSION].copy()
print(f"  v1:  {len(df_v1)} rows, {df_v1['ticker'].nunique()} tickers")
print(f"  v14: {len(df_v14)} rows, {df_v14['ticker'].nunique()} tickers")

if len(df_v1) == 0:
    print(f"\n  ERROR: No v1 results found for {V1_MODEL_VERSION}")
    print(f"  Run the original optimizer first.")
    raise SystemExit(1)

if len(df_v14) == 0:
    print(f"\n  ERROR: No v14 results found for {V14_MODEL_VERSION}")
    print(f"  Run the v14 optimizer first.")
    raise SystemExit(1)


# ==============================================================================
# BEST CONFIG PER TICKER (each version picks its best mode)
# ==============================================================================
def best_per_ticker(df, label):
    """For each ticker, pick the mode with highest TEST PnL (survived_b=True preferred)."""
    rows = []
    for ticker in df['ticker'].unique():
        td = df[df['ticker'] == ticker]
        survived = td[td['survived_b'] == True]
        pool = survived if len(survived) > 0 else td
        best = pool.loc[pool['val_b_avg_pnl'].idxmax()]
        rows.append(best)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out['version'] = label
    return out

best_v1  = best_per_ticker(df_v1, 'v1')
best_v14 = best_per_ticker(df_v14, 'v14')

print(f"\n  Best per ticker: v1={len(best_v1)}, v14={len(best_v14)}")


# ==============================================================================
# MERGE & COMPARE
# ==============================================================================
merged = best_v1[['ticker', 'tpsl_mode', 'prob_thresh', 'tp_pct', 'sl_pct',
                   'opt_avg_pnl', 'val_b_avg_pnl', 'val_b_win_rate', 'val_b_count',
                   'survived_b']].rename(columns={
    'tpsl_mode': 'v1_mode', 'prob_thresh': 'v1_prob',
    'tp_pct': 'v1_tp', 'sl_pct': 'v1_sl',
    'opt_avg_pnl': 'v1_opt_pnl',
    'val_b_avg_pnl': 'v1_test_pnl', 'val_b_win_rate': 'v1_test_wr',
    'val_b_count': 'v1_test_n', 'survived_b': 'v1_survived'
}).merge(
    best_v14[['ticker', 'tpsl_mode', 'prob_thresh', 'tp_pct', 'sl_pct',
              'opt_avg_pnl', 'val_b_avg_pnl', 'val_b_win_rate', 'val_b_count',
              'survived_b', 'exit_time_min_used', 'neighbor_stability',
              'overfit_ratio']].rename(columns={
        'tpsl_mode': 'v14_mode', 'prob_thresh': 'v14_prob',
        'tp_pct': 'v14_tp', 'sl_pct': 'v14_sl',
        'opt_avg_pnl': 'v14_opt_pnl',
        'val_b_avg_pnl': 'v14_test_pnl', 'val_b_win_rate': 'v14_test_wr',
        'val_b_count': 'v14_test_n', 'survived_b': 'v14_survived',
        'exit_time_min_used': 'v14_exit', 'neighbor_stability': 'v14_stab',
        'overfit_ratio': 'v14_ovf'
    }),
    on='ticker', how='outer'
)

merged['both_present'] = merged['v1_test_pnl'].notna() & merged['v14_test_pnl'].notna()
merged['v1_only']  = merged['v1_test_pnl'].notna() & merged['v14_test_pnl'].isna()
merged['v14_only'] = merged['v1_test_pnl'].isna() & merged['v14_test_pnl'].notna()

merged['pnl_delta'] = np.where(
    merged['both_present'],
    merged['v14_test_pnl'] - merged['v1_test_pnl'],
    np.nan
)
merged['mode_changed'] = merged['v1_mode'] != merged['v14_mode']
merged['config_changed'] = (
    (merged['v1_tp'] != merged['v14_tp']) |
    (merged['v1_sl'] != merged['v14_sl']) |
    (merged['v1_prob'] != merged['v14_prob']) |
    merged['mode_changed']
)

# Deployability
merged['v1_deploy']  = merged['v1_survived'].fillna(False) & (merged['v1_test_pnl'].fillna(0) >= DEPLOY_PNL_THRESHOLD)
merged['v14_deploy'] = merged['v14_survived'].fillna(False) & (merged['v14_test_pnl'].fillna(0) >= DEPLOY_PNL_THRESHOLD)


# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  COMPARISON: v1 vs v14 — LONG L5-1 Fold {FOLD_ID}")
print(f"{'='*W}")

both = merged[merged['both_present']]
n_both = len(both)
n_v1_only = merged['v1_only'].sum()
n_v14_only = merged['v14_only'].sum()
n_total = len(merged)

n_improved = (both['pnl_delta'] > 0).sum()
n_degraded = (both['pnl_delta'] < 0).sum()
n_same = (both['pnl_delta'] == 0).sum()
n_config_changed = both['config_changed'].sum()
n_mode_changed = both['mode_changed'].sum()

v1_deploy_n  = merged['v1_deploy'].sum()
v14_deploy_n = merged['v14_deploy'].sum()
gained_deploy = ((~merged['v1_deploy']) & merged['v14_deploy']).sum()
lost_deploy   = (merged['v1_deploy'] & (~merged['v14_deploy'])).sum()

avg_v1_pnl  = both['v1_test_pnl'].mean()
avg_v14_pnl = both['v14_test_pnl'].mean()

v1_ovf  = both['v1_opt_pnl'].mean()
v14_ovf = both['v14_opt_pnl'].mean()
v1_overfit  = 1.0 - (avg_v1_pnl / v1_ovf) if v1_ovf > 0 else float('nan')
v14_overfit = 1.0 - (avg_v14_pnl / v14_ovf) if v14_ovf > 0 else float('nan')

print(f"\n  Total tickers:     {n_total}")
print(f"  In BOTH versions:  {n_both}")
print(f"  v1-only (missing): {n_v1_only}")
print(f"  v14-only (new):    {n_v14_only}")
print(f"\n  Among {n_both} shared tickers:")
print(f"  Improved (v14>v1): {n_improved}  ({n_improved/n_both:.0%})")
print(f"  Degraded (v14<v1): {n_degraded}  ({n_degraded/n_both:.0%})")
print(f"  Unchanged:         {n_same}")
print(f"  Config changed:    {n_config_changed}")
print(f"  Mode changed:      {n_mode_changed}")
print(f"\n  Avg TEST PnL (shared tickers only):")
print(f"    v1:  {avg_v1_pnl:+.2%}")
print(f"    v14: {avg_v14_pnl:+.2%}  (delta: {avg_v14_pnl - avg_v1_pnl:+.2%})")
print(f"\n  Aggregate overfit ratio:")
print(f"    v1:  {v1_overfit:.2f}")
print(f"    v14: {v14_overfit:.2f}")
print(f"\n  Deployable tickers (TEST >= {DEPLOY_PNL_THRESHOLD:.1%}):")
print(f"    v1:  {v1_deploy_n}")
print(f"    v14: {v14_deploy_n}  (gained: {gained_deploy}, lost: {lost_deploy})")

# v14 mode distribution
print(f"\n  v14 mode distribution:")
for mode, count in best_v14['tpsl_mode'].value_counts().items():
    print(f"    {mode:<15} {count}")

# v14 exit time distribution
if 'exit_time_min_used' in best_v14.columns:
    et_counts = best_v14['exit_time_min_used'].dropna().astype(int).value_counts().sort_index()
    if len(et_counts) > 0:
        print(f"\n  v14 exit time distribution:")
        for et, count in et_counts.items():
            print(f"    {et//60}:{et%60:02d}  {count} tickers")


# ==============================================================================
# PER-TICKER TABLE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PER-TICKER COMPARISON (sorted by PnL delta)")
print(f"{'='*W}")

merged_both = merged[merged['both_present']].sort_values('pnl_delta', ascending=False)
merged_v1only = merged[merged['v1_only']].sort_values('v1_test_pnl', ascending=False)
merged_v14only = merged[merged['v14_only']].sort_values('v14_test_pnl', ascending=False)

print(f"\n  {'Ticker':<7} {'v1 Mode':<14} {'v14 Mode':<14} "
      f"{'v1 TEST':>8} {'v14 TEST':>9} {'Delta':>7} "
      f"{'v14 Exit':>9} {'v14 Ovf':>8} {'v14 Stab':>9} {'Status':>10}")
print(f"  {'─'*105}")

for _, r in merged_both.iterrows():
    v1_pnl = r['v1_test_pnl']
    v14_pnl = r['v14_test_pnl']
    delta = r['pnl_delta']

    v1_mode = str(r.get('v1_mode', 'N/A') or 'N/A')
    v14_mode = str(r.get('v14_mode', 'N/A') or 'N/A')

    et = r.get('v14_exit')
    et_str = f"{int(et)//60}:{int(et)%60:02d}" if pd.notna(et) else 'N/A'
    ovf = r.get('v14_ovf', 0) or 0
    stab = r.get('v14_stab', 0) or 0

    if r['v14_deploy'] and not r['v1_deploy']:
        status = "GAINED"
    elif r['v1_deploy'] and not r['v14_deploy']:
        status = "LOST"
    elif r['v14_deploy'] and r['v1_deploy']:
        status = "BOTH"
    else:
        status = ""

    arrow = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")

    print(f"  {r['ticker']:<7} {v1_mode:<14} {v14_mode:<14} "
          f"{v1_pnl:>+7.2%} {v14_pnl:>+8.2%} {delta:>+6.2%}{arrow} "
          f"{et_str:>9} {ovf:>7.2f} {stab:>8.2f} {status:>10}")

if len(merged_v1only) > 0:
    print(f"\n  ── v1-ONLY tickers (missing from v14) ──")
    for _, r in merged_v1only.iterrows():
        v1_pnl = r['v1_test_pnl']
        v1_mode = str(r.get('v1_mode', 'N/A') or 'N/A')
        deploy = "DEPLOY" if r['v1_deploy'] else ""
        print(f"  {r['ticker']:<7} {v1_mode:<14} {'---':<14} {v1_pnl:>+7.2%} {'---':>9} {'---':>7}  "
              f"{'---':>9} {'---':>8} {'---':>9} {deploy:>10}")

if len(merged_v14only) > 0:
    print(f"\n  ── v14-ONLY tickers (new in v14) ──")
    for _, r in merged_v14only.iterrows():
        v14_pnl = r['v14_test_pnl']
        v14_mode = str(r.get('v14_mode', 'N/A') or 'N/A')
        et = r.get('v14_exit')
        et_str = f"{int(et)//60}:{int(et)%60:02d}" if pd.notna(et) else 'N/A'
        deploy = "NEW" if r['v14_deploy'] else ""
        print(f"  {r['ticker']:<7} {'---':<14} {v14_mode:<14} {'---':>8} {v14_pnl:>+8.2%} {'---':>7}  "
              f"{et_str:>9} {'---':>8} {'---':>9} {deploy:>10}")


# ==============================================================================
# TICKERS GAINED / LOST DEPLOYABILITY
# ==============================================================================
gained = merged[merged['v14_deploy'] & ~merged['v1_deploy']]
lost   = merged[merged['v1_deploy'] & ~merged['v14_deploy']]

if len(gained) > 0:
    print(f"\n\n  GAINED deployability (v14 only):")
    for _, r in gained.iterrows():
        print(f"    {r['ticker']:<7} v14: {r['v14_test_pnl']:+.2%} ({r['v14_mode']})")

if len(lost) > 0:
    lost_missing = lost[lost['v14_only'] | lost['v14_test_pnl'].isna()]
    lost_present = lost[lost['both_present']]
    if len(lost_missing) > 0:
        print(f"\n  LOST deployability (missing from v14 entirely):")
        for _, r in lost_missing.iterrows():
            print(f"    {r['ticker']:<7} v1: {r['v1_test_pnl']:+.2%} ({r['v1_mode']})  -> NOT IN v14")
    if len(lost_present) > 0:
        print(f"\n  LOST deployability (worse config in v14):")
        for _, r in lost_present.iterrows():
            v14_pnl = r['v14_test_pnl']
            reason = ""
            ovf = r.get('v14_ovf', 0) or 0
            stab = r.get('v14_stab', 0) or 0
            if ovf > 0.60 and v14_pnl < DEPLOY_PNL_THRESHOLD:
                reason = " (overfit gate)"
            elif stab < 0.30:
                reason = " (stability gate)"
            else:
                reason = " (different config)"
            print(f"    {r['ticker']:<7} v1: {r['v1_test_pnl']:+.2%} -> v14: {v14_pnl:+.2%}{reason}")


# ==============================================================================
# v14-SPECIFIC: configs using new features
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  v14 NEW FEATURES IN USE")
print(f"{'='*W}")

# 3:1 ratio tickers (exclude trailing — trailing uses TP=inf)
non_trailing = best_v14[best_v14['tpsl_mode'] != 'trailing']
ratio_3to1 = non_trailing[non_trailing.apply(
    lambda r: (r['tp_pct'] / r['sl_pct'] >= 2.9) if r['sl_pct'] > 0 else False, axis=1)]
print(f"\n  3:1 TP:SL ratio selected (static/grace/limit_entry only): {len(ratio_3to1)} tickers")
for _, r in ratio_3to1.iterrows():
    print(f"    {r['ticker']:<7} TP={r['tp_pct']:.0%} SL={r['sl_pct']:.0%} ({r['tpsl_mode']})  "
          f"TEST={r['val_b_avg_pnl']:+.2%}")

# Trailing mode tickers
trailing = best_v14[best_v14['tpsl_mode'] == 'trailing']
print(f"\n  Trailing stop selected: {len(trailing)} tickers")
for _, r in trailing.iterrows():
    print(f"    {r['ticker']:<7} trig={r.get('trail_trigger_pct',0):.1%} "
          f"trail={r.get('trail_pct_pct',0):.1%}  TEST={r['val_b_avg_pnl']:+.2%}")

# Non-default exit time
if 'exit_time_min_used' in best_v14.columns:
    non_default_exit = best_v14[best_v14['exit_time_min_used'].fillna(780) != 780]
    print(f"\n  Non-default exit time: {len(non_default_exit)} tickers")
    for _, r in non_default_exit.iterrows():
        et = int(r['exit_time_min_used'])
        print(f"    {r['ticker']:<7} exit={et//60}:{et%60:02d}  TEST={r['val_b_avg_pnl']:+.2%}")

# Overfit-rejected: configs where overfit gate was the ACTUAL cause of rejection
# Gate fires when: overfit_ratio > 0.60 AND val_b < DEPLOY_PNL_THRESHOLD AND val_b > 0
# (configs with val_b <= 0 would fail survived_b regardless of the gate)
if 'overfit_ratio' in df_v14.columns:
    ovf_truly_rejected = df_v14[
        (df_v14['overfit_ratio'].fillna(0) > 0.60) &
        (df_v14['val_b_avg_pnl'] > 0) &
        (df_v14['val_b_avg_pnl'] < DEPLOY_PNL_THRESHOLD) &
        (df_v14['survived_b'] == False)
    ]
    ovf_would_fail = df_v14[
        (df_v14['overfit_ratio'].fillna(0) > 0.60) &
        (df_v14['val_b_avg_pnl'] <= 0)
    ]
    print(f"\n  Overfit gate rejected (positive TEST but gate killed): {len(ovf_truly_rejected)} configs")
    for _, r in ovf_truly_rejected.iterrows():
        print(f"    {r['ticker']:<7} {r['tpsl_mode']:<10} ovf={r['overfit_ratio']:.2f}  "
              f"OPT={r['opt_avg_pnl']:+.2%} TEST={r['val_b_avg_pnl']:+.2%}")
    print(f"  High overfit + negative TEST (would fail anyway): {len(ovf_would_fail)} configs")


# ==============================================================================
# CHARTS
# ==============================================================================
import os
os.makedirs(CHART_DIR, exist_ok=True)

# ── Chart 1: Side-by-side TEST PnL (shared tickers only) ──────────────────
chart_df = merged_both.copy()
fig, ax = plt.subplots(figsize=(14, max(6, len(chart_df) * 0.4)))

tickers_sorted = chart_df['ticker'].values
y = np.arange(len(tickers_sorted))
bar_height = 0.35

v1_vals  = chart_df['v1_test_pnl'].values * 100
v14_vals = chart_df['v14_test_pnl'].values * 100

bars1 = ax.barh(y + bar_height/2, v1_vals, bar_height, label='v1', color='steelblue', alpha=0.7)
bars2 = ax.barh(y - bar_height/2, v14_vals, bar_height, label='v14', color='darkorange', alpha=0.7)

ax.set_yticks(y)
ax.set_yticklabels(tickers_sorted, fontsize=7)
ax.set_xlabel('TEST Avg PnL (%)')
ax.set_title(f'LONG L5-1 Fold {FOLD_ID}: v1 vs v14 TEST PnL ({n_both} shared tickers)\n'
             f'v1 avg: {avg_v1_pnl:+.2%}  |  v14 avg: {avg_v14_pnl:+.2%}  |  '
             f'delta: {avg_v14_pnl - avg_v1_pnl:+.2%}  |  '
             f'{n_v1_only} v1-only, {n_v14_only} v14-only (not shown)')
ax.axvline(0, color='black', linewidth=0.8)
ax.axvline(DEPLOY_PNL_THRESHOLD * 100, color='green', linewidth=0.8, linestyle='--', alpha=0.5)
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()

chart1_path = os.path.join(CHART_DIR, f'long_L5_1_fold{FOLD_ID}_v1_vs_v14.png')
plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart 1 saved: {chart1_path}")
if _IN_COLAB:
    plt.show()
plt.close()

# ── Chart 2: PnL Delta histogram (shared tickers only) ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

deltas = both['pnl_delta'].dropna().values * 100
colors = ['green' if d > 0 else 'red' for d in deltas]
axes[0].bar(range(len(deltas)), sorted(deltas, reverse=True), color=colors, alpha=0.7)
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].set_xlabel('Ticker (ranked)')
axes[0].set_ylabel('PnL Delta (%)')
axes[0].set_title(f'v14 - v1 PnL Delta ({n_both} shared tickers)\n'
                  f'{n_improved} improved, {n_degraded} degraded')
axes[0].grid(axis='y', alpha=0.3)

# Scatter: v1 vs v14 TEST PnL (shared only)
axes[1].scatter(both['v1_test_pnl'] * 100,
                both['v14_test_pnl'] * 100,
                c='steelblue', alpha=0.6, s=40)
lim = max(abs(both['v1_test_pnl'].max()),
          abs(both['v14_test_pnl'].max()),
          abs(both['v1_test_pnl'].min()),
          abs(both['v14_test_pnl'].min())) * 100 + 1
axes[1].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label='y=x')
axes[1].axhline(DEPLOY_PNL_THRESHOLD * 100, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
axes[1].axvline(DEPLOY_PNL_THRESHOLD * 100, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
axes[1].set_xlabel('v1 TEST PnL (%)')
axes[1].set_ylabel('v14 TEST PnL (%)')
axes[1].set_title(f'v1 vs v14 TEST PnL Scatter ({n_both} shared)\nAbove diagonal = v14 better')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_aspect('equal')

plt.tight_layout()
chart2_path = os.path.join(CHART_DIR, f'long_L5_1_fold{FOLD_ID}_v14_delta.png')
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
print(f"  Chart 2 saved: {chart2_path}")
if _IN_COLAB:
    plt.show()
plt.close()

# ── Chart 3: Mode distribution comparison ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for idx, (df_best, label) in enumerate([(best_v1, 'v1'), (best_v14, 'v14')]):
    mode_counts = df_best['tpsl_mode'].value_counts()
    axes[idx].bar(mode_counts.index, mode_counts.values,
                  color=['steelblue', 'darkorange', 'green', 'purple'][:len(mode_counts)])
    axes[idx].set_title(f'{label} — Best Mode per Ticker')
    axes[idx].set_ylabel('# Tickers')
    for i, (mode, count) in enumerate(mode_counts.items()):
        axes[idx].text(i, count + 0.3, str(count), ha='center', fontsize=10)

plt.tight_layout()
chart3_path = os.path.join(CHART_DIR, f'long_L5_1_fold{FOLD_ID}_mode_dist.png')
plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
print(f"  Chart 3 saved: {chart3_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# VERDICT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

pnl_better = avg_v14_pnl > avg_v1_pnl
deploy_better = v14_deploy_n >= v1_deploy_n
overfit_better = v14_overfit < v1_overfit if not (np.isnan(v1_overfit) or np.isnan(v14_overfit)) else False

print(f"\n  Based on {n_both} shared tickers (excludes {n_v1_only} v1-only, {n_v14_only} v14-only):")

if pnl_better and deploy_better:
    print(f"  v14 WINS: higher avg PnL ({avg_v14_pnl:+.2%} vs {avg_v1_pnl:+.2%}) "
          f"and more deployable tickers ({v14_deploy_n} vs {v1_deploy_n})")
elif pnl_better:
    print(f"  v14 MIXED: higher avg PnL ({avg_v14_pnl:+.2%} vs {avg_v1_pnl:+.2%}) "
          f"but fewer deployable tickers ({v14_deploy_n} vs {v1_deploy_n})")
elif deploy_better:
    print(f"  v14 MIXED: more deployable tickers ({v14_deploy_n} vs {v1_deploy_n}) "
          f"but lower avg PnL ({avg_v14_pnl:+.2%} vs {avg_v1_pnl:+.2%})")
else:
    print(f"  v1 WINS: higher avg PnL ({avg_v1_pnl:+.2%} vs {avg_v14_pnl:+.2%}) "
          f"and more deployable tickers ({v1_deploy_n} vs {v14_deploy_n})")

if overfit_better:
    print(f"  v14 has lower overfit ({v14_overfit:.2f} vs {v1_overfit:.2f})")

if n_v1_only > 0:
    n_v1_only_deploy = merged[merged['v1_only'] & merged['v1_deploy']].shape[0]
    print(f"\n  WARNING: {n_v1_only} tickers in v1 but missing from v14 "
          f"({n_v1_only_deploy} were deployable)")

if lost_deploy > 0:
    print(f"  WARNING: {lost_deploy} tickers lost deployability total")

print(f"\n{'='*W}")
print(f"  DONE")
print(f"{'='*W}")
