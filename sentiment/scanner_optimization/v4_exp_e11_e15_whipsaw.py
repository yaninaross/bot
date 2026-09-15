"""
V4 Experiments E11, E15, E16, E17, E18, E19 — Whipsaw Analysis & Entry Timing
===============================================================================
E11: Whipsaw Characterization (descriptive)
E15: Whipsaw Intensity vs Trade Outcome (correlation)
E16: Entry Timing Sweep — Minute-Level Resolution (SHORT focus)
E17: Whipsaw Exhaustion Signal — Dynamic Entry
E18: Entry at Local Price Extremes
E19: Entry Offset Stratified by Opening Volatility
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr, trim_mean

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = '/home/admin_yaninaso_altostrat_com/bot-RAG/sentiment/scanner_optimization'
RESULTS = f'{BASE}/results'
CSV_PATH = f'{RESULTS}/v2_features_full.csv'
BARS_PATH = f'{RESULTS}/v3_minute_bars.json'

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
with open(BARS_PATH) as f:
    bars_data = json.load(f)
ticker_bars = bars_data['ticker_bars']

print(f"  Trades: {len(df)} ({(df.direction=='LONG').sum()} LONG, {(df.direction=='SHORT').sum()} SHORT)")
print(f"  Minute bar keys: {len(ticker_bars)}")
print(f"  pnl_t2_close non-null: {df.pnl_t2_close.notna().sum()}")
print()

LOW_N_THRESH = 20

# ── Helpers ────────────────────────────────────────────────────────────────────
def trimmed_mean(x, pct=0.05):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return np.nan
    return trim_mean(x, pct)

def dist_stats(arr, label=""):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return {'n': 0}
    return {
        'n': n,
        'mean': np.mean(arr),
        'trimmed_mean': trimmed_mean(arr),
        'median': np.median(arr),
        'P25': np.percentile(arr, 25),
        'P75': np.percentile(arr, 75),
        'P90': np.percentile(arr, 90),
        'std': np.std(arr),
    }

def fmt_stats(d, name=""):
    if d['n'] == 0:
        return f"  {name}: NO DATA"
    flag = " [LOW-N]" if d['n'] < LOW_N_THRESH else ""
    return (f"  {name} (n={d['n']}{flag}): mean={d['mean']:.3f}, "
            f"trim_mean={d['trimmed_mean']:.3f}, median={d['median']:.3f}, "
            f"P25={d['P25']:.3f}, P75={d['P75']:.3f}, P90={d['P90']:.3f}")

def get_market_bars(trade_key, start='09:30', end='10:30'):
    """Get sorted minute bars for a trade within a time range."""
    if trade_key not in ticker_bars:
        return []
    bars = ticker_bars[trade_key]
    result = []
    for t in sorted(bars.keys()):
        if start <= t <= end:
            b = bars[t]
            result.append({'time': t, 'o': b['o'], 'h': b['h'],
                           'l': b['l'], 'c': b['c'], 'v': b['v']})
    return result

def minutes_from_open(time_str):
    """Minutes from 09:30."""
    h, m = map(int, time_str.split(':'))
    return (h - 9) * 60 + (m - 30)


# ══════════════════════════════════════════════════════════════════════════════
# E11: WHIPSAW CHARACTERIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("E11: WHIPSAW CHARACTERIZATION")
print("=" * 80)

e11_records = []

for idx, row in df.iterrows():
    key = f"{row['ticker']}__{row['eval_date']}"
    bars = get_market_bars(key, '09:30', '10:30')
    if len(bars) < 10:
        continue

    or_high = row['or_high']
    or_low = row['or_low']
    or_mid = (or_high + or_low) / 2.0
    entry_price = row['entry_price']

    # 1. Range crosses in first 30 min (09:30-09:59)
    bars_30m = [b for b in bars if b['time'] <= '09:59']
    crosses = 0
    if len(bars_30m) >= 2:
        prev_side = 1 if bars_30m[0]['c'] >= or_mid else -1
        for b in bars_30m[1:]:
            curr_side = 1 if b['c'] >= or_mid else -1
            if curr_side != prev_side:
                crosses += 1
            prev_side = curr_side

    # 2. Max excursion above OR ceiling (first 60 min)
    bars_60m = [b for b in bars if b['time'] <= '10:29']
    max_above = 0.0
    for b in bars_60m:
        exc = (b['c'] - or_high) / entry_price * 100
        if exc > max_above:
            max_above = exc

    # 3. Max excursion below OR floor (first 60 min)
    max_below = 0.0
    for b in bars_60m:
        exc = (or_low - b['c']) / entry_price * 100
        if exc > max_below:
            max_below = exc

    # 4. Time to first committed directional move (5 consecutive same-side closes)
    time_to_commit = np.nan
    if len(bars_60m) >= 5:
        sides = [1 if b['c'] >= or_mid else -1 for b in bars_60m]
        for i in range(len(sides) - 4):
            if all(s == sides[i] for s in sides[i:i+5]):
                time_to_commit = minutes_from_open(bars_60m[i+4]['time'])
                break

    e11_records.append({
        'ticker': row['ticker'],
        'eval_date': row['eval_date'],
        'direction': row['direction'],
        'sign': row['sign'],
        'gap_pct': row['gap_pct'],
        'abs_gap_pct': row['abs_gap_pct'],
        'atr_pct': row['atr_pct'],
        'entry_price': entry_price,
        'or_high': or_high,
        'or_low': or_low,
        'or_mid': or_mid,
        'or_width': (or_high - or_low) / entry_price * 100,
        'range_crosses': crosses,
        'max_excursion_above': max_above,
        'max_excursion_below': max_below,
        'time_to_commitment': time_to_commit,
        'pnl_60m_nosl': row['pnl_60m_nosl'],
        'pnl_eod_nosl': row['pnl_eod_nosl'],
        'pnl_t2_close': row['pnl_t2_close'],
    })

e11 = pd.DataFrame(e11_records)
print(f"\nTrades with minute bars: {len(e11)}")

# Distribution statistics
print("\n--- Overall Distribution Statistics ---")
for metric in ['range_crosses', 'max_excursion_above', 'max_excursion_below', 'time_to_commitment']:
    print(fmt_stats(dist_stats(e11[metric]), metric))

# By direction
print("\n--- By Direction ---")
for direction in ['LONG', 'SHORT']:
    sub = e11[e11.direction == direction]
    print(f"\n  {direction} (n={len(sub)}):")
    for metric in ['range_crosses', 'max_excursion_above', 'max_excursion_below', 'time_to_commitment']:
        print(fmt_stats(dist_stats(sub[metric]), f"    {metric}"))

# By gap_pct quartile
print("\n--- By Gap Quartile ---")
e11['gap_q'] = pd.qcut(e11['abs_gap_pct'], 4, labels=['Q1_small', 'Q2', 'Q3', 'Q4_large'])
for q in ['Q1_small', 'Q2', 'Q3', 'Q4_large']:
    sub = e11[e11.gap_q == q]
    flag = " [LOW-N]" if len(sub) < LOW_N_THRESH else ""
    print(f"\n  {q} (n={len(sub)}{flag}):")
    for metric in ['range_crosses', 'max_excursion_above', 'max_excursion_below', 'time_to_commitment']:
        s = dist_stats(sub[metric])
        if s['n'] > 0:
            print(f"    {metric}: mean={s['mean']:.3f}, median={s['median']:.3f}")

# Summary table
print("\n--- Summary Table ---")
summary_rows = []
for direction in ['ALL', 'LONG', 'SHORT']:
    sub = e11 if direction == 'ALL' else e11[e11.direction == direction]
    row_data = {'direction': direction, 'n': len(sub)}
    for m in ['range_crosses', 'max_excursion_above', 'max_excursion_below', 'time_to_commitment']:
        s = dist_stats(sub[m])
        row_data[f'{m}_mean'] = s.get('mean', np.nan)
        row_data[f'{m}_median'] = s.get('median', np.nan)
    summary_rows.append(row_data)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False, float_format='%.3f'))

# ── E11 Charts ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('E11: Whipsaw Characterization', fontsize=14, fontweight='bold')

# (1) Histogram of range crosses
ax = axes[0, 0]
ax.hist(e11['range_crosses'], bins=range(0, int(e11['range_crosses'].max()) + 2),
        edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(e11['range_crosses'].mean(), color='red', linestyle='--',
           label=f"mean={e11['range_crosses'].mean():.1f}")
ax.axvline(e11['range_crosses'].median(), color='orange', linestyle='--',
           label=f"median={e11['range_crosses'].median():.1f}")
ax.set_xlabel('Range Crosses (30 min)')
ax.set_ylabel('Count')
ax.set_title('OR Midpoint Crosses in First 30 Min')
ax.legend(fontsize=8)

# (2) Histogram of max excursion above ceiling
ax = axes[0, 1]
data = e11['max_excursion_above']
ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='coral')
ax.axvline(data.mean(), color='red', linestyle='--', label=f"mean={data.mean():.2f}%")
ax.set_xlabel('Max Excursion Above OR Ceiling (%)')
ax.set_ylabel('Count')
ax.set_title('Max Excursion Above Opening Range')
ax.legend(fontsize=8)

# (3) Histogram of max excursion below floor
ax = axes[1, 0]
data = e11['max_excursion_below']
ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='mediumseagreen')
ax.axvline(data.mean(), color='red', linestyle='--', label=f"mean={data.mean():.2f}%")
ax.set_xlabel('Max Excursion Below OR Floor (%)')
ax.set_ylabel('Count')
ax.set_title('Max Excursion Below Opening Range')
ax.legend(fontsize=8)

# (4) Histogram of time to commitment
ax = axes[1, 1]
data = e11['time_to_commitment'].dropna()
ax.hist(data, bins=range(0, 65, 5), edgecolor='black', alpha=0.7, color='mediumpurple')
ax.axvline(data.mean(), color='red', linestyle='--', label=f"mean={data.mean():.1f}min")
ax.axvline(data.median(), color='orange', linestyle='--', label=f"median={data.median():.1f}min")
pct_never = e11['time_to_commitment'].isna().mean() * 100
ax.set_xlabel('Minutes from 09:30')
ax.set_ylabel('Count')
ax.set_title(f'Time to Commitment (5 consec. same-side)\n{pct_never:.0f}% never commit in 60 min')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{BASE}/v4_e11_whipsaw_characterization.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: v4_e11_whipsaw_characterization.png")

# Additional chart: direction breakdown + gap quartile
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('E11: Whipsaw by Direction & Gap Size', fontsize=13, fontweight='bold')

# (5) Range crosses by direction
ax = axes[0]
for i, direction in enumerate(['LONG', 'SHORT']):
    sub = e11[e11.direction == direction]
    ax.hist(sub['range_crosses'], bins=range(0, int(e11['range_crosses'].max()) + 2),
            alpha=0.5, label=f"{direction} (mean={sub['range_crosses'].mean():.1f})",
            color=['steelblue', 'coral'][i], edgecolor='black')
ax.set_xlabel('Range Crosses (30 min)')
ax.set_ylabel('Count')
ax.set_title('Range Crosses by Direction')
ax.legend()

# (6) Time to commitment by gap quartile
ax = axes[1]
gap_labels = ['Q1_small', 'Q2', 'Q3', 'Q4_large']
commit_data = []
commit_labels = []
for q in gap_labels:
    sub = e11[e11.gap_q == q]['time_to_commitment'].dropna()
    if len(sub) > 0:
        commit_data.append(sub.values)
        commit_labels.append(f"{q}\n(n={len(sub)})")
ax.boxplot(commit_data, labels=commit_labels)
ax.set_ylabel('Minutes from 09:30')
ax.set_title('Time to Commitment by Gap Quartile')

plt.tight_layout()
plt.savefig(f'{BASE}/v4_e11_whipsaw_by_direction_gap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: v4_e11_whipsaw_by_direction_gap.png")


# ══════════════════════════════════════════════════════════════════════════════
# E15: WHIPSAW INTENSITY VS TRADE OUTCOME
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("E15: WHIPSAW INTENSITY VS TRADE OUTCOME")
print("=" * 80)

whipsaw_metrics = ['range_crosses', 'max_excursion_above', 'max_excursion_below', 'time_to_commitment']
pnl_cols = ['pnl_60m_nosl', 'pnl_eod_nosl', 'pnl_t2_close']
pnl_labels = ['PnL 60m', 'PnL EOD', 'PnL T+2']

# 1. Correlation matrix
print("\n--- Pearson Correlations ---")
pearson_matrix = np.full((len(whipsaw_metrics), len(pnl_cols)), np.nan)
pearson_pvals = np.full_like(pearson_matrix, np.nan)
spearman_matrix = np.full_like(pearson_matrix, np.nan)
spearman_pvals = np.full_like(pearson_matrix, np.nan)

for i, wm in enumerate(whipsaw_metrics):
    for j, pc in enumerate(pnl_cols):
        mask = e11[[wm, pc]].notna().all(axis=1)
        x = e11.loc[mask, wm].values
        y = e11.loc[mask, pc].values
        n = len(x)
        if n >= 10:
            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y)
            pearson_matrix[i, j] = r_p
            pearson_pvals[i, j] = p_p
            spearman_matrix[i, j] = r_s
            spearman_pvals[i, j] = p_s

print(f"\n{'Metric':<25} | {'PnL 60m':>15} | {'PnL EOD':>15} | {'PnL T+2':>15}")
print("-" * 80)
for i, wm in enumerate(whipsaw_metrics):
    parts = []
    for j in range(len(pnl_cols)):
        if np.isnan(pearson_matrix[i, j]):
            parts.append(f"{'N/A':>15}")
        else:
            star = '*' if pearson_pvals[i, j] < 0.05 else ''
            parts.append(f"{pearson_matrix[i,j]:>7.3f} (p={pearson_pvals[i,j]:.3f}){star:>1}")
    print(f"{wm:<25} | {' | '.join(parts)}")

print(f"\n--- Spearman Correlations ---")
print(f"{'Metric':<25} | {'PnL 60m':>15} | {'PnL EOD':>15} | {'PnL T+2':>15}")
print("-" * 80)
for i, wm in enumerate(whipsaw_metrics):
    parts = []
    for j in range(len(pnl_cols)):
        if np.isnan(spearman_matrix[i, j]):
            parts.append(f"{'N/A':>15}")
        else:
            star = '*' if spearman_pvals[i, j] < 0.05 else ''
            parts.append(f"{spearman_matrix[i,j]:>7.3f} (p={spearman_pvals[i,j]:.3f}){star:>1}")
    print(f"{wm:<25} | {' | '.join(parts)}")

# 2. Bucket analysis (tertiles by range_crosses)
print("\n--- Whipsaw Bucket Analysis (Tertiles by Range Crosses) ---")
try:
    e11['whipsaw_tertile'] = pd.qcut(e11['range_crosses'], 3, labels=['Low', 'Medium', 'High'], duplicates='drop')
except ValueError:
    # If too few unique values for tertiles, use manual buckets
    p33 = e11['range_crosses'].quantile(0.33)
    p66 = e11['range_crosses'].quantile(0.66)
    e11['whipsaw_tertile'] = pd.cut(e11['range_crosses'], bins=[-1, p33, p66, 100],
                                     labels=['Low', 'Medium', 'High'])

print(f"\n{'Tertile':<10} | {'n':>5} | {'Avg PnL 60m':>12} | {'Avg PnL EOD':>12} | {'Avg PnL T+2':>12} | {'Trim PnL T+2':>13}")
print("-" * 80)
bucket_results = {}
for tertile in ['Low', 'Medium', 'High']:
    sub = e11[e11.whipsaw_tertile == tertile]
    n = len(sub)
    avg_60m = sub['pnl_60m_nosl'].mean()
    avg_eod = sub['pnl_eod_nosl'].mean()
    avg_t2 = sub['pnl_t2_close'].mean()
    trim_t2 = trimmed_mean(sub['pnl_t2_close'].dropna())
    n_t2 = sub['pnl_t2_close'].notna().sum()
    flag = " [LOW-N]" if n_t2 < LOW_N_THRESH else ""
    bucket_results[tertile] = {'n': n, 'n_t2': n_t2, 'avg_60m': avg_60m,
                                'avg_eod': avg_eod, 'avg_t2': avg_t2, 'trim_t2': trim_t2}
    print(f"{tertile:<10} | {n:>5} | {avg_60m:>12.3f} | {avg_eod:>12.3f} | "
          f"{avg_t2:>12.3f} | {trim_t2:>13.3f}{flag}")

# 3. Direction-specific analysis
print("\n--- Direction-Specific Bucket Analysis ---")
for direction in ['LONG', 'SHORT']:
    sub_dir = e11[e11.direction == direction]
    print(f"\n  {direction} (n={len(sub_dir)}):")
    print(f"  {'Tertile':<10} | {'n':>5} | {'Avg 60m':>10} | {'Avg EOD':>10} | {'Avg T+2':>10}")
    print(f"  " + "-" * 60)
    for tertile in ['Low', 'Medium', 'High']:
        sub = sub_dir[sub_dir.whipsaw_tertile == tertile]
        n = len(sub)
        avg_60m = sub['pnl_60m_nosl'].mean()
        avg_eod = sub['pnl_eod_nosl'].mean()
        avg_t2 = sub['pnl_t2_close'].mean()
        flag = " [LOW-N]" if n < LOW_N_THRESH else ""
        print(f"  {tertile:<10} | {n:>5} | {avg_60m:>10.3f} | {avg_eod:>10.3f} | {avg_t2:>10.3f}{flag}")

# 4. OLS Regression: PnL_T2 ~ whipsaw metrics
print("\n--- OLS Regression: pnl_t2_close ~ whipsaw metrics ---")
reg_mask = e11[['pnl_t2_close'] + whipsaw_metrics].notna().all(axis=1)
reg_data = e11[reg_mask].copy()
print(f"  Regression sample: n={len(reg_data)}")

if len(reg_data) >= 15:
    X_cols = whipsaw_metrics
    X = reg_data[X_cols].values
    y = reg_data['pnl_t2_close'].values

    # Add constant
    X_with_const = np.column_stack([np.ones(len(X)), X])
    try:
        betas, residuals, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
        y_pred = X_with_const @ betas
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n_obs = len(y)
        k = len(betas) - 1
        adj_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - k - 1) if n_obs > k + 1 else 0

        # Standard errors
        mse = ss_res / (n_obs - k - 1) if n_obs > k + 1 else 1e-10
        cov = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        se = np.sqrt(np.diag(cov))
        t_stats = betas / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_obs - k - 1))

        print(f"\n  R-squared: {r_squared:.4f}")
        print(f"  Adj R-squared: {adj_r_squared:.4f}")
        print(f"\n  {'Variable':<25} | {'Coeff':>10} | {'SE':>10} | {'t-stat':>8} | {'p-value':>8}")
        print(f"  " + "-" * 75)
        names = ['intercept'] + X_cols
        for i, name in enumerate(names):
            star = ' *' if p_values[i] < 0.05 else '  '
            print(f"  {name:<25} | {betas[i]:>10.4f} | {se[i]:>10.4f} | {t_stats[i]:>8.3f} | {p_values[i]:>8.4f}{star}")
    except np.linalg.LinAlgError:
        print("  Regression failed (singular matrix)")
        r_squared = np.nan
        betas = []
        p_values = []
else:
    print("  Insufficient data for regression")
    r_squared = np.nan
    betas = []
    p_values = []

# 5. Key finding
print("\n--- KEY FINDING ---")
# Check if range_crosses correlates with T+2
mask_t2 = e11[['range_crosses', 'pnl_t2_close']].notna().all(axis=1)
if mask_t2.sum() >= 10:
    r, p = spearmanr(e11.loc[mask_t2, 'range_crosses'], e11.loc[mask_t2, 'pnl_t2_close'])
    if p < 0.05:
        direction_word = "worse" if r < 0 else "better"
        print(f"  High whipsaw predicts {direction_word} T+2 outcome (Spearman r={r:.3f}, p={p:.3f})")
    else:
        print(f"  No significant correlation between whipsaw and T+2 outcome (Spearman r={r:.3f}, p={p:.3f})")
else:
    print("  Insufficient T+2 data to determine whipsaw-outcome relationship")

# ── E15 Charts ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('E15: Whipsaw Intensity vs Trade Outcome', fontsize=14, fontweight='bold')

# (1) Scatter: range_crosses vs PnL T+2
ax = axes[0, 0]
mask_t2 = e11['pnl_t2_close'].notna()
for direction, color, marker in [('LONG', 'steelblue', 'o'), ('SHORT', 'coral', 's')]:
    sub = e11[(e11.direction == direction) & mask_t2]
    ax.scatter(sub['range_crosses'], sub['pnl_t2_close'], alpha=0.5,
               color=color, marker=marker, label=direction, s=40)
# Add regression line (all)
all_sub = e11[mask_t2]
if len(all_sub) > 5:
    z = np.polyfit(all_sub['range_crosses'], all_sub['pnl_t2_close'], 1)
    x_line = np.linspace(all_sub['range_crosses'].min(), all_sub['range_crosses'].max(), 50)
    ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.7, linewidth=1.5,
            label=f"slope={z[0]:.2f}")
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Range Crosses (30 min)')
ax.set_ylabel('PnL T+2 (%)')
ax.set_title('Range Crosses vs T+2 PnL')
ax.legend(fontsize=8)

# (2) Bar chart: whipsaw tertile -> avg PnL at each horizon
ax = axes[0, 1]
tertiles = ['Low', 'Medium', 'High']
x_pos = np.arange(len(tertiles))
width = 0.25
colors_bar = ['steelblue', 'mediumseagreen', 'coral']
for i, (pc, label) in enumerate(zip(pnl_cols, pnl_labels)):
    vals = []
    for t in tertiles:
        sub = e11[e11.whipsaw_tertile == t]
        vals.append(sub[pc].mean())
    bars_plot = ax.bar(x_pos + i * width, vals, width, label=label, color=colors_bar[i], alpha=0.8)
ax.set_xticks(x_pos + width)
ax.set_xticklabels(tertiles)
ax.set_ylabel('Avg PnL (%)')
ax.set_title('PnL by Whipsaw Tertile')
ax.legend(fontsize=8)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

# (3) Heatmap: whipsaw metric x PnL horizon -> correlation
ax = axes[1, 0]
# Use spearman for heatmap
hm_data = np.full((len(whipsaw_metrics), len(pnl_cols)), np.nan)
for i, wm in enumerate(whipsaw_metrics):
    for j, pc in enumerate(pnl_cols):
        mask = e11[[wm, pc]].notna().all(axis=1)
        if mask.sum() >= 10:
            r_s, _ = spearmanr(e11.loc[mask, wm], e11.loc[mask, pc])
            hm_data[i, j] = r_s

im = ax.imshow(hm_data, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
ax.set_xticks(range(len(pnl_labels)))
ax.set_xticklabels(pnl_labels, fontsize=9)
ax.set_yticks(range(len(whipsaw_metrics)))
ax.set_yticklabels([m.replace('_', '\n') for m in whipsaw_metrics], fontsize=8)
for i in range(len(whipsaw_metrics)):
    for j in range(len(pnl_cols)):
        if not np.isnan(hm_data[i, j]):
            ax.text(j, i, f"{hm_data[i,j]:.2f}", ha='center', va='center', fontsize=9,
                    color='white' if abs(hm_data[i,j]) > 0.15 else 'black')
ax.set_title('Spearman Correlation Heatmap')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# (4) Scatter: time_to_commitment vs PnL T+2
ax = axes[1, 1]
mask_commit = e11[['time_to_commitment', 'pnl_t2_close']].notna().all(axis=1)
for direction, color, marker in [('LONG', 'steelblue', 'o'), ('SHORT', 'coral', 's')]:
    sub = e11[(e11.direction == direction) & mask_commit]
    ax.scatter(sub['time_to_commitment'], sub['pnl_t2_close'], alpha=0.5,
               color=color, marker=marker, label=direction, s=40)
if mask_commit.sum() > 5:
    sub_all = e11[mask_commit]
    z = np.polyfit(sub_all['time_to_commitment'], sub_all['pnl_t2_close'], 1)
    x_line = np.linspace(sub_all['time_to_commitment'].min(), sub_all['time_to_commitment'].max(), 50)
    ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.7, linewidth=1.5,
            label=f"slope={z[0]:.3f}")
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Minutes to Commitment')
ax.set_ylabel('PnL T+2 (%)')
ax.set_title('Time to Commitment vs T+2 PnL')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{BASE}/v4_e15_whipsaw_vs_outcome.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: v4_e15_whipsaw_vs_outcome.png")


# ══════════════════════════════════════════════════════════════════════════════
# E16: ENTRY TIMING SWEEP — MINUTE-LEVEL RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("E16: ENTRY TIMING SWEEP (SHORT trades)")
print("=" * 80)

entry_times = ['09:30', '09:32', '09:34', '09:36', '09:38', '09:40',
               '09:45', '09:50', '09:55', '10:00', '10:15', '10:30']

# For SHORT: original pnl_t2_close = (entry_price - exit_t2) / entry_price * 100
# So exit_t2 = entry_price * (1 - pnl_t2_close/100)
# New PnL at different entry: new_pnl = (new_price - exit_t2) / new_price * 100

short_trades = e11[(e11.direction == 'SHORT') & (e11.pnl_t2_close.notna())].copy()
long_trades = e11[(e11.direction == 'LONG') & (e11.pnl_t2_close.notna())].copy()
print(f"\nSHORT trades with T+2 data: {len(short_trades)}")
print(f"LONG trades with T+2 data: {len(long_trades)}")

# Also compute for EOD (available for all)
short_all = e11[e11.direction == 'SHORT'].copy()
long_all = e11[e11.direction == 'LONG'].copy()

def compute_exit_price(row, pnl_col):
    """Compute the exit price from entry price and sign-adjusted PnL."""
    # PnL is sign-adjusted: positive = profitable
    # For SHORT: pnl = (entry - exit) / entry * 100 => exit = entry * (1 - pnl/100)
    # For LONG:  pnl = (exit - entry) / entry * 100 => exit = entry * (1 + pnl/100)
    if row['direction'] == 'SHORT':
        return row['entry_price'] * (1 - row[pnl_col] / 100)
    else:
        return row['entry_price'] * (1 + row[pnl_col] / 100)

def compute_pnl_from_exit(new_entry, exit_price, direction):
    """Compute sign-adjusted PnL given new entry and exit prices."""
    if direction == 'SHORT':
        return (new_entry - exit_price) / new_entry * 100
    else:
        return (exit_price - new_entry) / new_entry * 100

# Sweep results
e16_results = {'SHORT_t2': {}, 'SHORT_eod': {}, 'LONG_t2': {}, 'LONG_eod': {}}

for entry_t in entry_times:
    for label, trades_sub, pnl_col in [
        ('SHORT_t2', short_trades, 'pnl_t2_close'),
        ('SHORT_eod', short_all, 'pnl_eod_nosl'),
        ('LONG_t2', long_trades, 'pnl_t2_close'),
        ('LONG_eod', long_all, 'pnl_eod_nosl'),
    ]:
        pnls = []
        for _, row in trades_sub.iterrows():
            key = f"{row['ticker']}__{row['eval_date']}"
            bars = ticker_bars.get(key, {})
            if entry_t not in bars:
                continue
            exit_price = compute_exit_price(row, pnl_col)
            new_entry = bars[entry_t]['c']
            new_pnl = compute_pnl_from_exit(new_entry, exit_price, row['direction'])
            pnls.append(new_pnl)

        e16_results[label][entry_t] = {
            'n': len(pnls),
            'mean': np.mean(pnls) if pnls else np.nan,
            'trimmed_mean': trimmed_mean(pnls) if len(pnls) >= 4 else np.nan,
            'median': np.median(pnls) if pnls else np.nan,
            'std': np.std(pnls) if pnls else np.nan,
        }

# Print SHORT T+2 sweep
print("\n--- SHORT Entry Timing Sweep: T+2 PnL ---")
flag_note = " [LOW-N]" if len(short_trades) < LOW_N_THRESH else ""
print(f"  Base: 09:30 entry, n={len(short_trades)}{flag_note}")
print(f"\n  {'Entry':>7} | {'n':>4} | {'Mean PnL':>10} | {'Trim Mean':>10} | {'Median':>10} | {'vs 09:30':>10}")
print(f"  " + "-" * 65)
baseline_t2 = e16_results['SHORT_t2'].get('09:30', {}).get('mean', np.nan)
for et in entry_times:
    r = e16_results['SHORT_t2'][et]
    delta = r['mean'] - baseline_t2 if not np.isnan(r['mean']) else np.nan
    flag = " [LOW-N]" if r['n'] < LOW_N_THRESH else ""
    print(f"  {et:>7} | {r['n']:>4} | {r['mean']:>10.3f} | {r['trimmed_mean']:>10.3f} | "
          f"{r['median']:>10.3f} | {delta:>+10.3f}{flag}")

# Print SHORT EOD sweep
print("\n--- SHORT Entry Timing Sweep: EOD PnL ---")
print(f"  Base: 09:30 entry, n={len(short_all)}")
print(f"\n  {'Entry':>7} | {'n':>4} | {'Mean PnL':>10} | {'Trim Mean':>10} | {'Median':>10} | {'vs 09:30':>10}")
print(f"  " + "-" * 65)
baseline_eod = e16_results['SHORT_eod'].get('09:30', {}).get('mean', np.nan)
for et in entry_times:
    r = e16_results['SHORT_eod'][et]
    delta = r['mean'] - baseline_eod if not np.isnan(r['mean']) else np.nan
    print(f"  {et:>7} | {r['n']:>4} | {r['mean']:>10.3f} | {r['trimmed_mean']:>10.3f} | "
          f"{r['median']:>10.3f} | {delta:>+10.3f}")

# Print LONG T+2 sweep
print("\n--- LONG Entry Timing Sweep: T+2 PnL ---")
print(f"\n  {'Entry':>7} | {'n':>4} | {'Mean PnL':>10} | {'Trim Mean':>10} | {'vs 09:30':>10}")
print(f"  " + "-" * 55)
baseline_long_t2 = e16_results['LONG_t2'].get('09:30', {}).get('mean', np.nan)
for et in entry_times:
    r = e16_results['LONG_t2'][et]
    delta = r['mean'] - baseline_long_t2 if not np.isnan(r['mean']) else np.nan
    flag = " [LOW-N]" if r['n'] < LOW_N_THRESH else ""
    print(f"  {et:>7} | {r['n']:>4} | {r['mean']:>10.3f} | {r['trimmed_mean']:>10.3f} | {delta:>+10.3f}{flag}")


# ══════════════════════════════════════════════════════════════════════════════
# E17: WHIPSAW EXHAUSTION SIGNAL — DYNAMIC ENTRY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("E17: WHIPSAW EXHAUSTION SIGNAL (SHORT trades)")
print("=" * 80)

e17_records = []

for _, row in short_all.iterrows():
    key = f"{row['ticker']}__{row['eval_date']}"
    bars_list = get_market_bars(key, '09:30', '10:30')
    if len(bars_list) < 15:
        continue

    entry_price = row['entry_price']
    or_high = row['or_high']
    or_low = row['or_low']
    or_mid = (or_high + or_low) / 2.0

    # Compute exit prices for EOD and T+2
    exit_eod = compute_exit_price(row, 'pnl_eod_nosl')
    has_t2 = pd.notna(row['pnl_t2_close'])
    exit_t2 = compute_exit_price(row, 'pnl_t2_close') if has_t2 else np.nan

    # Signal A: Volatility collapse
    # 2-min rolling realized vol (std of returns)
    closes = [b['c'] for b in bars_list]
    times = [b['time'] for b in bars_list]
    returns = [abs(closes[i] - closes[i-1]) / closes[i-1] * 100 for i in range(1, len(closes))]

    # Rolling 2-bar vol
    rolling_vol = []
    for i in range(1, len(returns)):
        rolling_vol.append(np.std(returns[max(0,i-1):i+1]))

    # Find peak vol in first 15 min
    first_15_bars = min(15, len(rolling_vol))
    if first_15_bars > 0:
        peak_vol = max(rolling_vol[:first_15_bars])
    else:
        peak_vol = 0

    vol_exhaust_time = None
    vol_exhaust_price = None
    if peak_vol > 0:
        for i in range(first_15_bars, len(rolling_vol)):
            if rolling_vol[i] < 0.5 * peak_vol:
                # +2 because: returns starts at index 1, rolling_vol starts at index 1 of returns
                bar_idx = i + 2
                if bar_idx < len(bars_list):
                    vol_exhaust_time = times[bar_idx]
                    vol_exhaust_price = closes[bar_idx]
                break

    # Signal B: Range stabilization (crossing rate drops to 0 for 3 consecutive minutes)
    sides = [1 if c >= or_mid else -1 for c in closes]
    range_stab_time = None
    range_stab_price = None
    # For each minute, count if there's a cross
    cross_at = [0] * len(sides)
    for i in range(1, len(sides)):
        cross_at[i] = 1 if sides[i] != sides[i-1] else 0

    # Find 3 consecutive 0s (no crosses) after at least 5 bars
    for i in range(5, len(cross_at) - 2):
        if cross_at[i] == 0 and cross_at[i+1] == 0 and cross_at[i+2] == 0:
            if i + 2 < len(bars_list):
                range_stab_time = times[i+2]
                range_stab_price = closes[i+2]
            break

    rec = {
        'ticker': row['ticker'],
        'eval_date': row['eval_date'],
        'entry_price_0930': entry_price,
        'exit_eod': exit_eod,
        'exit_t2': exit_t2,
        'has_t2': has_t2,
        'pnl_eod_fixed': row['pnl_eod_nosl'],
        'pnl_t2_fixed': row['pnl_t2_close'],
    }

    # Vol collapse entry
    if vol_exhaust_time and vol_exhaust_price:
        rec['vol_exhaust_time'] = vol_exhaust_time
        rec['vol_exhaust_minutes'] = minutes_from_open(vol_exhaust_time)
        rec['pnl_eod_vol'] = compute_pnl_from_exit(vol_exhaust_price, exit_eod, 'SHORT')
        rec['pnl_t2_vol'] = compute_pnl_from_exit(vol_exhaust_price, exit_t2, 'SHORT') if has_t2 else np.nan
    else:
        rec['vol_exhaust_time'] = None
        rec['vol_exhaust_minutes'] = np.nan
        rec['pnl_eod_vol'] = np.nan
        rec['pnl_t2_vol'] = np.nan

    # Range stabilization entry
    if range_stab_time and range_stab_price:
        rec['range_stab_time'] = range_stab_time
        rec['range_stab_minutes'] = minutes_from_open(range_stab_time)
        rec['pnl_eod_stab'] = compute_pnl_from_exit(range_stab_price, exit_eod, 'SHORT')
        rec['pnl_t2_stab'] = compute_pnl_from_exit(range_stab_price, exit_t2, 'SHORT') if has_t2 else np.nan
    else:
        rec['range_stab_time'] = None
        rec['range_stab_minutes'] = np.nan
        rec['pnl_eod_stab'] = np.nan
        rec['pnl_t2_stab'] = np.nan

    e17_records.append(rec)

e17 = pd.DataFrame(e17_records)
print(f"\nSHORT trades analyzed: {len(e17)}")

# Compare strategies
print("\n--- Entry Strategy Comparison (SHORT, EOD PnL) ---")
strategies = [
    ('Fixed 09:30', 'pnl_eod_fixed'),
    ('Vol Collapse', 'pnl_eod_vol'),
    ('Range Stab', 'pnl_eod_stab'),
]
print(f"\n  {'Strategy':<20} | {'n':>5} | {'Mean PnL':>10} | {'Trim Mean':>10} | {'Median':>10}")
print(f"  " + "-" * 65)
for name, col in strategies:
    valid = e17[col].notna()
    n = valid.sum()
    flag = " [LOW-N]" if n < LOW_N_THRESH else ""
    if n > 0:
        print(f"  {name:<20} | {n:>5} | {e17.loc[valid, col].mean():>10.3f} | "
              f"{trimmed_mean(e17.loc[valid, col]):>10.3f} | {e17.loc[valid, col].median():>10.3f}{flag}")

print("\n--- Entry Strategy Comparison (SHORT, T+2 PnL) ---")
strategies_t2 = [
    ('Fixed 09:30', 'pnl_t2_fixed'),
    ('Vol Collapse', 'pnl_t2_vol'),
    ('Range Stab', 'pnl_t2_stab'),
]
print(f"\n  {'Strategy':<20} | {'n':>5} | {'Mean PnL':>10} | {'Trim Mean':>10} | {'Median':>10}")
print(f"  " + "-" * 65)
for name, col in strategies_t2:
    valid = e17[col].notna()
    n = valid.sum()
    flag = " [LOW-N]" if n < LOW_N_THRESH else ""
    if n > 0:
        print(f"  {name:<20} | {n:>5} | {e17.loc[valid, col].mean():>10.3f} | "
              f"{trimmed_mean(e17.loc[valid, col]):>10.3f} | {e17.loc[valid, col].median():>10.3f}{flag}")

# Timing distribution for dynamic signals
print("\n--- Dynamic Entry Timing Distribution ---")
for signal_name, col in [('Vol Collapse', 'vol_exhaust_minutes'), ('Range Stab', 'range_stab_minutes')]:
    valid = e17[col].notna()
    if valid.sum() > 0:
        vals = e17.loc[valid, col]
        fire_rate = valid.mean() * 100
        print(f"\n  {signal_name}: fires on {fire_rate:.0f}% of trades (n={valid.sum()})")
        print(f"    Mean time: {vals.mean():.1f} min, Median: {vals.median():.1f} min, "
              f"P25: {vals.quantile(0.25):.1f}, P75: {vals.quantile(0.75):.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# E18: ENTRY AT LOCAL PRICE EXTREMES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("E18: ENTRY AT LOCAL PRICE EXTREMES (SHORT trades)")
print("=" * 80)

e18_records = []

for _, row in short_all.iterrows():
    key = f"{row['ticker']}__{row['eval_date']}"
    bars_list = get_market_bars(key, '09:30', '10:00')
    if len(bars_list) < 10:
        continue

    entry_price = row['entry_price']
    or_high = row['or_high']
    or_low = row['or_low']
    or_mid = (or_high + or_low) / 2.0

    exit_eod = compute_exit_price(row, 'pnl_eod_nosl')
    has_t2 = pd.notna(row['pnl_t2_close'])
    exit_t2 = compute_exit_price(row, 'pnl_t2_close') if has_t2 else np.nan

    closes = [b['c'] for b in bars_list]
    highs = [b['h'] for b in bars_list]
    times = [b['time'] for b in bars_list]

    # Rule 1: First local high (window=3)
    # Local max: higher than prev 3 and next 3 (using close prices)
    rule1_price = None
    rule1_time = None
    window = 3
    for i in range(window, len(closes) - window):
        neighborhood = closes[i-window:i+window+1]
        if closes[i] == max(neighborhood):
            rule1_price = closes[i]
            rule1_time = times[i]
            break

    # Rule 2: Highest high in whipsaw window (09:30 to 09:59)
    rule2_price = max(highs) if highs else None
    rule2_idx = highs.index(max(highs)) if highs else None
    rule2_time = times[rule2_idx] if rule2_idx is not None else None

    # Rule 3: Last local high before commitment below OR
    # Find when price commits below OR (5 consecutive closes below or_mid)
    commit_idx = None
    sides = [1 if c >= or_mid else -1 for c in closes]
    for i in range(len(sides) - 4):
        if all(s == -1 for s in sides[i:i+5]):
            commit_idx = i
            break

    rule3_price = None
    rule3_time = None
    if commit_idx is not None and commit_idx > window:
        # Find last local high before commit_idx
        for i in range(min(commit_idx, len(closes) - window) - 1, window - 1, -1):
            neighborhood = closes[max(0, i-window):min(len(closes), i+window+1)]
            if closes[i] == max(neighborhood):
                rule3_price = closes[i]
                rule3_time = times[i]
                break

    rec = {
        'ticker': row['ticker'],
        'eval_date': row['eval_date'],
        'entry_price_0930': entry_price,
        'exit_eod': exit_eod,
        'exit_t2': exit_t2,
        'has_t2': has_t2,
        'pnl_eod_baseline': row['pnl_eod_nosl'],
        'pnl_t2_baseline': row['pnl_t2_close'],
    }

    # Rule 1
    if rule1_price:
        rec['rule1_price'] = rule1_price
        rec['rule1_time'] = rule1_time
        rec['rule1_pnl_eod'] = compute_pnl_from_exit(rule1_price, exit_eod, 'SHORT')
        rec['rule1_pnl_t2'] = compute_pnl_from_exit(rule1_price, exit_t2, 'SHORT') if has_t2 else np.nan
        rec['rule1_improvement_eod'] = rec['rule1_pnl_eod'] - row['pnl_eod_nosl']
    else:
        rec['rule1_price'] = np.nan
        rec['rule1_time'] = None
        rec['rule1_pnl_eod'] = np.nan
        rec['rule1_pnl_t2'] = np.nan
        rec['rule1_improvement_eod'] = np.nan

    # Rule 2 (theoretical max)
    if rule2_price:
        rec['rule2_price'] = rule2_price
        rec['rule2_time'] = rule2_time
        rec['rule2_pnl_eod'] = compute_pnl_from_exit(rule2_price, exit_eod, 'SHORT')
        rec['rule2_pnl_t2'] = compute_pnl_from_exit(rule2_price, exit_t2, 'SHORT') if has_t2 else np.nan
        rec['rule2_improvement_eod'] = rec['rule2_pnl_eod'] - row['pnl_eod_nosl']
    else:
        rec['rule2_price'] = np.nan
        rec['rule2_time'] = None
        rec['rule2_pnl_eod'] = np.nan
        rec['rule2_pnl_t2'] = np.nan
        rec['rule2_improvement_eod'] = np.nan

    # Rule 3
    if rule3_price:
        rec['rule3_price'] = rule3_price
        rec['rule3_time'] = rule3_time
        rec['rule3_pnl_eod'] = compute_pnl_from_exit(rule3_price, exit_eod, 'SHORT')
        rec['rule3_pnl_t2'] = compute_pnl_from_exit(rule3_price, exit_t2, 'SHORT') if has_t2 else np.nan
        rec['rule3_improvement_eod'] = rec['rule3_pnl_eod'] - row['pnl_eod_nosl']
    else:
        rec['rule3_price'] = np.nan
        rec['rule3_time'] = None
        rec['rule3_pnl_eod'] = np.nan
        rec['rule3_pnl_t2'] = np.nan
        rec['rule3_improvement_eod'] = np.nan

    e18_records.append(rec)

e18 = pd.DataFrame(e18_records)
print(f"\nSHORT trades analyzed: {len(e18)}")

# Comparison table
print("\n--- Entry Rule Comparison (SHORT, EOD PnL) ---")
rules = [
    ('Baseline (09:30 close)', 'pnl_eod_baseline'),
    ('Rule 1: First local high', 'rule1_pnl_eod'),
    ('Rule 2: Highest high (theo max)', 'rule2_pnl_eod'),
    ('Rule 3: Last high before commit', 'rule3_pnl_eod'),
]
print(f"\n  {'Rule':<35} | {'n':>5} | {'Mean':>10} | {'Trim':>10} | {'Median':>10} | {'vs Base':>10}")
print(f"  " + "-" * 90)
base_eod = e18['pnl_eod_baseline'].mean()
for name, col in rules:
    valid = e18[col].notna()
    n = valid.sum()
    flag = " [LOW-N]" if n < LOW_N_THRESH else ""
    if n > 0:
        m = e18.loc[valid, col].mean()
        print(f"  {name:<35} | {n:>5} | {m:>10.3f} | "
              f"{trimmed_mean(e18.loc[valid, col]):>10.3f} | {e18.loc[valid, col].median():>10.3f} | "
              f"{m - base_eod:>+10.3f}{flag}")

# T+2 comparison
print("\n--- Entry Rule Comparison (SHORT, T+2 PnL) ---")
rules_t2 = [
    ('Baseline (09:30 close)', 'pnl_t2_baseline'),
    ('Rule 1: First local high', 'rule1_pnl_t2'),
    ('Rule 2: Highest high (theo max)', 'rule2_pnl_t2'),
    ('Rule 3: Last high before commit', 'rule3_pnl_t2'),
]
print(f"\n  {'Rule':<35} | {'n':>5} | {'Mean':>10} | {'Trim':>10} | {'Median':>10}")
print(f"  " + "-" * 80)
for name, col in rules_t2:
    valid = e18[col].notna()
    n = valid.sum()
    flag = " [LOW-N]" if n < LOW_N_THRESH else ""
    if n > 0:
        print(f"  {name:<35} | {n:>5} | {e18.loc[valid, col].mean():>10.3f} | "
              f"{trimmed_mean(e18.loc[valid, col]):>10.3f} | {e18.loc[valid, col].median():>10.3f}{flag}")

# Improvement stats
print("\n--- Rule Improvement Over Baseline (EOD, %-points) ---")
for rule_name, col in [('Rule 1', 'rule1_improvement_eod'), ('Rule 2', 'rule2_improvement_eod'),
                         ('Rule 3', 'rule3_improvement_eod')]:
    valid = e18[col].notna()
    if valid.sum() > 0:
        vals = e18.loc[valid, col]
        print(f"  {rule_name}: mean improvement = {vals.mean():+.3f}%, median = {vals.median():+.3f}%, "
              f"pct positive = {(vals > 0).mean()*100:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
# E19: ENTRY OFFSET STRATIFIED BY OPENING VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("E19: ENTRY OFFSET STRATIFIED BY OPENING VOLATILITY (SHORT trades)")
print("=" * 80)

# Compute OR width as ATR multiple for SHORT trades
short_for_e19 = short_all.copy()
short_for_e19['or_width_pct'] = (short_for_e19['or_high'] - short_for_e19['or_low']) / short_for_e19['entry_price'] * 100
short_for_e19['or_atr_ratio'] = short_for_e19['or_width_pct'] / short_for_e19['atr_pct']

print(f"\nOR width as ATR multiple — mean: {short_for_e19['or_atr_ratio'].mean():.3f}, "
      f"median: {short_for_e19['or_atr_ratio'].median():.3f}")

# Bucket by OR width
short_for_e19['vol_bucket'] = pd.cut(
    short_for_e19['or_atr_ratio'],
    bins=[-np.inf, 0.5, 1.0, np.inf],
    labels=['Tight (<0.5x)', 'Normal (0.5-1x)', 'Wide (>1x)']
)

print(f"\nBucket distribution:")
for b in ['Tight (<0.5x)', 'Normal (0.5-1x)', 'Wide (>1x)']:
    n = (short_for_e19['vol_bucket'] == b).sum()
    flag = " [LOW-N]" if n < LOW_N_THRESH else ""
    print(f"  {b}: n={n}{flag}")

# Run timing sweep per bucket
print("\n--- Timing Sweep by Opening Volatility Bucket (EOD PnL) ---")
e19_results = {}
for bucket in ['Tight (<0.5x)', 'Normal (0.5-1x)', 'Wide (>1x)']:
    sub = short_for_e19[short_for_e19['vol_bucket'] == bucket]
    e19_results[bucket] = {}
    for entry_t in entry_times:
        pnls = []
        for _, row in sub.iterrows():
            key = f"{row['ticker']}__{row['eval_date']}"
            bars = ticker_bars.get(key, {})
            if entry_t not in bars:
                continue
            exit_price = compute_exit_price(row, 'pnl_eod_nosl')
            new_entry = bars[entry_t]['c']
            new_pnl = compute_pnl_from_exit(new_entry, exit_price, 'SHORT')
            pnls.append(new_pnl)
        e19_results[bucket][entry_t] = {
            'n': len(pnls),
            'mean': np.mean(pnls) if pnls else np.nan,
            'trimmed_mean': trimmed_mean(pnls) if len(pnls) >= 4 else np.nan,
        }

for bucket in ['Tight (<0.5x)', 'Normal (0.5-1x)', 'Wide (>1x)']:
    sub_n = (short_for_e19['vol_bucket'] == bucket).sum()
    flag = " [LOW-N]" if sub_n < LOW_N_THRESH else ""
    print(f"\n  {bucket} (n={sub_n}{flag}):")
    baseline = e19_results[bucket].get('09:30', {}).get('mean', np.nan)
    best_et = None
    best_pnl = -999
    print(f"    {'Entry':>7} | {'n':>4} | {'Mean':>10} | {'Trim':>10} | {'vs 09:30':>10}")
    print(f"    " + "-" * 50)
    for et in entry_times:
        r = e19_results[bucket][et]
        delta = r['mean'] - baseline if not np.isnan(r['mean']) else np.nan
        print(f"    {et:>7} | {r['n']:>4} | {r['mean']:>10.3f} | {r['trimmed_mean']:>10.3f} | {delta:>+10.3f}")
        if r['mean'] > best_pnl and r['n'] >= 5:
            best_pnl = r['mean']
            best_et = et
    if best_et:
        print(f"    ==> Optimal entry: {best_et} (mean PnL = {best_pnl:.3f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED CHARTS: E16 + E17 + E18 + E19
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('E16-E19: Entry Timing Analysis (SHORT Trades)', fontsize=14, fontweight='bold')

# E16: Timing sweep curve (EOD)
ax = axes[0, 0]
times_minutes = [minutes_from_open(t) for t in entry_times]
short_eod_means = [e16_results['SHORT_eod'][t]['mean'] for t in entry_times]
short_eod_trims = [e16_results['SHORT_eod'][t]['trimmed_mean'] for t in entry_times]
ax.plot(times_minutes, short_eod_means, 'o-', color='coral', label='Mean EOD', linewidth=2)
ax.plot(times_minutes, short_eod_trims, 's--', color='darkred', label='Trim Mean EOD', linewidth=1.5)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
# Mark the peak
peak_idx = np.nanargmax(short_eod_means)
ax.annotate(f'Peak: {entry_times[peak_idx]}\n{short_eod_means[peak_idx]:.2f}%',
            xy=(times_minutes[peak_idx], short_eod_means[peak_idx]),
            xytext=(times_minutes[peak_idx]+3, short_eod_means[peak_idx]+0.3),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=8)
ax.set_xlabel('Minutes from 09:30')
ax.set_ylabel('PnL (%)')
ax.set_title('E16: SHORT Entry Timing Sweep (EOD)')
ax.legend(fontsize=8)
ax.set_xticks(times_minutes)
ax.set_xticklabels(entry_times, rotation=45, fontsize=7)

# E16: Timing sweep curve (T+2)
ax = axes[0, 1]
if any(not np.isnan(e16_results['SHORT_t2'][t]['mean']) for t in entry_times):
    t2_means = [e16_results['SHORT_t2'][t]['mean'] for t in entry_times]
    t2_trims = [e16_results['SHORT_t2'][t]['trimmed_mean'] for t in entry_times]
    ax.plot(times_minutes, t2_means, 'o-', color='steelblue', label='Mean T+2', linewidth=2)
    ax.plot(times_minutes, t2_trims, 's--', color='navy', label='Trim Mean T+2', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    peak_idx = np.nanargmax(t2_means)
    ax.annotate(f'Peak: {entry_times[peak_idx]}\n{t2_means[peak_idx]:.2f}%',
                xy=(times_minutes[peak_idx], t2_means[peak_idx]),
                xytext=(times_minutes[peak_idx]+3, t2_means[peak_idx]+0.3),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=8)
ax.set_xlabel('Minutes from 09:30')
ax.set_ylabel('PnL (%)')
ax.set_title('E16: SHORT Entry Timing Sweep (T+2)')
ax.legend(fontsize=8)
ax.set_xticks(times_minutes)
ax.set_xticklabels(entry_times, rotation=45, fontsize=7)
n_t2 = e16_results['SHORT_t2']['09:30']['n']
ax.text(0.02, 0.98, f'n={n_t2}', transform=ax.transAxes, fontsize=8, va='top')

# E17: Dynamic entry comparison (EOD)
ax = axes[0, 2]
strat_names = ['Fixed\n09:30', 'Vol\nCollapse', 'Range\nStab']
strat_vals_eod = [
    e17['pnl_eod_fixed'].mean(),
    e17['pnl_eod_vol'].dropna().mean() if e17['pnl_eod_vol'].notna().sum() > 0 else 0,
    e17['pnl_eod_stab'].dropna().mean() if e17['pnl_eod_stab'].notna().sum() > 0 else 0,
]
strat_n = [
    len(e17),
    e17['pnl_eod_vol'].notna().sum(),
    e17['pnl_eod_stab'].notna().sum(),
]
colors_strat = ['steelblue', 'coral', 'mediumseagreen']
bars_ax = ax.bar(strat_names, strat_vals_eod, color=colors_strat, alpha=0.8, edgecolor='black')
for i, (b, n) in enumerate(zip(bars_ax, strat_n)):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
            f'n={n}', ha='center', fontsize=8)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_ylabel('Avg PnL EOD (%)')
ax.set_title('E17: Dynamic Entry Signals (EOD)')

# E18: Entry rule comparison (EOD)
ax = axes[1, 0]
rule_names = ['Baseline\n09:30', 'Rule 1\nFirst High', 'Rule 2\nMax High', 'Rule 3\nLast High']
rule_cols_eod = ['pnl_eod_baseline', 'rule1_pnl_eod', 'rule2_pnl_eod', 'rule3_pnl_eod']
rule_vals = []
rule_ns = []
for col in rule_cols_eod:
    valid = e18[col].notna()
    rule_vals.append(e18.loc[valid, col].mean() if valid.sum() > 0 else 0)
    rule_ns.append(valid.sum())
colors_rule = ['steelblue', 'coral', 'gold', 'mediumseagreen']
bars_ax = ax.bar(rule_names, rule_vals, color=colors_rule, alpha=0.8, edgecolor='black')
for i, (b, n) in enumerate(zip(bars_ax, rule_ns)):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
            f'n={n}', ha='center', fontsize=8)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_ylabel('Avg PnL EOD (%)')
ax.set_title('E18: Entry at Price Extremes (EOD)')

# E19: Timing by vol bucket
ax = axes[1, 1]
for bucket, color, marker in [('Tight (<0.5x)', 'steelblue', 'o'),
                                ('Normal (0.5-1x)', 'mediumseagreen', 's'),
                                ('Wide (>1x)', 'coral', '^')]:
    means = [e19_results[bucket][t]['mean'] for t in entry_times]
    n_bucket = (short_for_e19['vol_bucket'] == bucket).sum()
    ax.plot(times_minutes, means, f'{marker}-', color=color,
            label=f'{bucket} (n={n_bucket})', linewidth=1.5, markersize=5)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Minutes from 09:30')
ax.set_ylabel('PnL EOD (%)')
ax.set_title('E19: Timing by Opening Volatility')
ax.legend(fontsize=7)
ax.set_xticks(times_minutes)
ax.set_xticklabels(entry_times, rotation=45, fontsize=7)

# Summary annotation
ax = axes[1, 2]
ax.axis('off')
# Find optimal entries
opt_short_eod = entry_times[np.nanargmax(short_eod_means)]
opt_short_eod_pnl = max(short_eod_means)
base_short_eod_pnl = e16_results['SHORT_eod']['09:30']['mean']

summary_text = "KEY FINDINGS\n" + "=" * 30 + "\n\n"
summary_text += f"E16 Optimal SHORT entry (EOD):\n  {opt_short_eod} ({opt_short_eod_pnl:+.2f}%)\n"
summary_text += f"  vs 09:30 baseline: {base_short_eod_pnl:+.2f}%\n"
summary_text += f"  Improvement: {opt_short_eod_pnl - base_short_eod_pnl:+.2f}%\n\n"

# E17 best
strat_best = ['Fixed 09:30', 'Vol Collapse', 'Range Stab']
best_idx = np.nanargmax(strat_vals_eod)
summary_text += f"E17 Best dynamic signal (EOD):\n  {strat_best[best_idx]} ({strat_vals_eod[best_idx]:+.2f}%)\n\n"

# E18 best rule
best_rule_idx = np.nanargmax(rule_vals)
summary_text += f"E18 Best entry rule (EOD):\n  {rule_names[best_rule_idx].replace(chr(10),' ')} ({rule_vals[best_rule_idx]:+.2f}%)\n"
summary_text += f"  Theo max (Rule 2): {rule_vals[2]:+.2f}%\n\n"

# E19
for bucket in ['Tight (<0.5x)', 'Normal (0.5-1x)', 'Wide (>1x)']:
    bucket_means = [e19_results[bucket][t]['mean'] for t in entry_times]
    valid_means = [(m, t) for m, t in zip(bucket_means, entry_times) if not np.isnan(m)]
    if valid_means:
        best_m, best_t = max(valid_means, key=lambda x: x[0])
        summary_text += f"E19 {bucket}: opt={best_t}\n"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{BASE}/v4_e16_e19_entry_timing.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: v4_e16_e19_entry_timing.png")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS JSON
# ══════════════════════════════════════════════════════════════════════════════
results_json = {
    'E11_whipsaw_characterization': {
        'n_trades': len(e11),
        'overall': {m: dist_stats(e11[m]) for m in ['range_crosses', 'max_excursion_above',
                                                      'max_excursion_below', 'time_to_commitment']},
        'by_direction': {},
    },
    'E15_whipsaw_vs_outcome': {
        'pearson_correlations': {},
        'spearman_correlations': {},
        'bucket_analysis': {},
        'regression': {},
    },
    'E16_entry_timing_sweep': {
        'SHORT_eod': {},
        'SHORT_t2': {},
        'LONG_eod': {},
        'LONG_t2': {},
    },
    'E17_dynamic_entry': {},
    'E18_price_extremes': {},
    'E19_volatility_stratified': {},
}

# E11
for direction in ['LONG', 'SHORT']:
    sub = e11[e11.direction == direction]
    results_json['E11_whipsaw_characterization']['by_direction'][direction] = {
        m: dist_stats(sub[m]) for m in ['range_crosses', 'max_excursion_above',
                                         'max_excursion_below', 'time_to_commitment']
    }

# E15 correlations
for i, wm in enumerate(whipsaw_metrics):
    results_json['E15_whipsaw_vs_outcome']['pearson_correlations'][wm] = {}
    results_json['E15_whipsaw_vs_outcome']['spearman_correlations'][wm] = {}
    for j, pc in enumerate(pnl_cols):
        val_p = float(pearson_matrix[i, j]) if not np.isnan(pearson_matrix[i, j]) else None
        val_s = float(spearman_matrix[i, j]) if not np.isnan(spearman_matrix[i, j]) else None
        results_json['E15_whipsaw_vs_outcome']['pearson_correlations'][wm][pc] = val_p
        results_json['E15_whipsaw_vs_outcome']['spearman_correlations'][wm][pc] = val_s

# E15 buckets
for t in ['Low', 'Medium', 'High']:
    if t in bucket_results:
        br = bucket_results[t]
        results_json['E15_whipsaw_vs_outcome']['bucket_analysis'][t] = {
            k: (float(v) if not (isinstance(v, float) and np.isnan(v)) else None) for k, v in br.items()
        }

# E15 regression
if betas is not None and len(betas) > 0:
    results_json['E15_whipsaw_vs_outcome']['regression'] = {
        'r_squared': float(r_squared) if not np.isnan(r_squared) else None,
        'coefficients': {name: float(b) for name, b in zip(['intercept'] + whipsaw_metrics, betas)},
        'p_values': {name: float(p) for name, p in zip(['intercept'] + whipsaw_metrics, p_values)},
    }

# E16
for label in ['SHORT_eod', 'SHORT_t2', 'LONG_eod', 'LONG_t2']:
    for et in entry_times:
        r = e16_results[label][et]
        results_json['E16_entry_timing_sweep'][label][et] = {
            k: (float(v) if not (isinstance(v, float) and np.isnan(v)) else None) for k, v in r.items()
        }

# E17
for strat_name, col_eod, col_t2 in [('fixed_0930', 'pnl_eod_fixed', 'pnl_t2_fixed'),
                                      ('vol_collapse', 'pnl_eod_vol', 'pnl_t2_vol'),
                                      ('range_stab', 'pnl_eod_stab', 'pnl_t2_stab')]:
    valid_eod = e17[col_eod].notna()
    valid_t2 = e17[col_t2].notna()
    results_json['E17_dynamic_entry'][strat_name] = {
        'n_eod': int(valid_eod.sum()),
        'mean_eod': float(e17.loc[valid_eod, col_eod].mean()) if valid_eod.sum() > 0 else None,
        'n_t2': int(valid_t2.sum()),
        'mean_t2': float(e17.loc[valid_t2, col_t2].mean()) if valid_t2.sum() > 0 else None,
    }

# E18
for rule_name, col_eod, col_t2 in [('baseline', 'pnl_eod_baseline', 'pnl_t2_baseline'),
                                     ('rule1_first_local_high', 'rule1_pnl_eod', 'rule1_pnl_t2'),
                                     ('rule2_highest_high', 'rule2_pnl_eod', 'rule2_pnl_t2'),
                                     ('rule3_last_high_commit', 'rule3_pnl_eod', 'rule3_pnl_t2')]:
    valid_eod = e18[col_eod].notna()
    valid_t2 = e18[col_t2].notna()
    results_json['E18_price_extremes'][rule_name] = {
        'n_eod': int(valid_eod.sum()),
        'mean_eod': float(e18.loc[valid_eod, col_eod].mean()) if valid_eod.sum() > 0 else None,
        'n_t2': int(valid_t2.sum()),
        'mean_t2': float(e18.loc[valid_t2, col_t2].mean()) if valid_t2.sum() > 0 else None,
    }

# E19
for bucket in ['Tight (<0.5x)', 'Normal (0.5-1x)', 'Wide (>1x)']:
    results_json['E19_volatility_stratified'][bucket] = {}
    for et in entry_times:
        r = e19_results[bucket][et]
        results_json['E19_volatility_stratified'][bucket][et] = {
            k: (float(v) if not (isinstance(v, float) and np.isnan(v)) else None) for k, v in r.items()
        }

# Convert numpy types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

results_json = convert_numpy(results_json)

with open(f'{RESULTS}/v4_e11_e15_results.json', 'w') as f:
    json.dump(results_json, f, indent=2, default=str)
print(f"\nSaved: results/v4_e11_e15_results.json")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"""
E11 — Whipsaw Characterization:
  Range crosses (30 min): mean={e11['range_crosses'].mean():.1f}, median={e11['range_crosses'].median():.0f}
  Max excursion above OR: mean={e11['max_excursion_above'].mean():.2f}%
  Max excursion below OR: mean={e11['max_excursion_below'].mean():.2f}%
  Time to commitment: mean={e11['time_to_commitment'].dropna().mean():.1f} min, never={e11['time_to_commitment'].isna().mean()*100:.0f}%

E15 — Whipsaw vs Outcome:
  Regression R-squared: {f'{r_squared:.4f}' if not np.isnan(r_squared) else 'N/A'}
  Bucket spread (EOD): Low={bucket_results.get('Low',{}).get('avg_eod',0):.2f}%, High={bucket_results.get('High',{}).get('avg_eod',0):.2f}%

E16 — Entry Timing Sweep (SHORT EOD):
  Baseline 09:30: {base_short_eod_pnl:.2f}%
  Optimal entry: {opt_short_eod} ({opt_short_eod_pnl:.2f}%)
  Improvement: {opt_short_eod_pnl - base_short_eod_pnl:+.2f}%

E17 — Dynamic Entry (SHORT EOD):
  Fixed 09:30: {strat_vals_eod[0]:.2f}%
  Vol Collapse: {strat_vals_eod[1]:.2f}% (fires {e17['pnl_eod_vol'].notna().mean()*100:.0f}%)
  Range Stab:   {strat_vals_eod[2]:.2f}% (fires {e17['pnl_eod_stab'].notna().mean()*100:.0f}%)

E18 — Price Extremes (SHORT EOD):
  Baseline:     {rule_vals[0]:.2f}%
  Rule 1 (first high):  {rule_vals[1]:.2f}%
  Rule 2 (max high):    {rule_vals[2]:.2f}%  (theoretical max)
  Rule 3 (last pre-commit): {rule_vals[3]:.2f}%

E19 — Vol-Stratified Optimal Entry:""")

for bucket in ['Tight (<0.5x)', 'Normal (0.5-1x)', 'Wide (>1x)']:
    bucket_means = [e19_results[bucket][t]['mean'] for t in entry_times]
    valid_means = [(m, t) for m, t in zip(bucket_means, entry_times) if not np.isnan(m)]
    if valid_means:
        best_m, best_t = max(valid_means, key=lambda x: x[0])
        n_bucket = (short_for_e19['vol_bucket'] == bucket).sum()
        flag = " [LOW-N]" if n_bucket < LOW_N_THRESH else ""
        print(f"  {bucket}: optimal={best_t} ({best_m:+.2f}%), n={n_bucket}{flag}")

print("\nDone.")
