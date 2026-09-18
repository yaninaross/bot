"""
V4 Scanner Optimization — E16 (Entry Timing Sweep) & E19 (Entry Offset Stratified by Opening Volatility)
==========================================================================================================
E16: For every SHORT trade, simulate entering at every minute from 09:30 to 10:30 and measure T+2 outcome.
E19: Stratify by opening range volatility and find per-bucket optimal entry time.
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import trim_mean, sem

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = 'os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../') + '/sentiment/scanner_optimization'
RESULTS = os.path.join(BASE, 'results')

CSV_PATH = os.path.join(RESULTS, 'v2_features_full.csv')
MINUTE_PATH = os.path.join(RESULTS, 'v3_minute_bars.json')
DAILY_PATH = os.path.join(RESULTS, 'v2_daily_data.json')

FIG_E16 = os.path.join(BASE, 'v4_e16_timing_sweep.png')
FIG_E19 = os.path.join(BASE, 'v4_e19_volatility_stratified.png')
JSON_OUT = os.path.join(RESULTS, 'v4_e16_e19_results.json')

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
with open(MINUTE_PATH) as f:
    minute_data = json.load(f)
with open(DAILY_PATH) as f:
    daily_data = json.load(f)

ticker_bars = minute_data['ticker_bars']

shorts = df[df['direction'] == 'SHORT'].copy()
print(f"Total SHORT trades: {len(shorts)}")

# ── Entry times to sweep ──────────────────────────────────────────────────────
ENTRY_TIMES = [
    '09:30', '09:32', '09:34', '09:36', '09:38', '09:40',
    '09:42', '09:44', '09:46', '09:48', '09:50', '09:52',
    '09:54', '09:56', '09:58', '10:00', '10:05', '10:10',
    '10:15', '10:20', '10:25', '10:30',
]


def time_to_minutes(t):
    """Convert HH:MM to minutes since midnight."""
    h, m = t.split(':')
    return int(h) * 60 + int(m)


ENTRY_MINUTES = [time_to_minutes(t) for t in ENTRY_TIMES]


def get_entry_price(bars_dict, target_time):
    """Get close of minute bar at target_time. If missing, use next available within 5 min."""
    target_min = time_to_minutes(target_time)
    # Try exact time first
    if target_time in bars_dict:
        return bars_dict[target_time]['c']
    # Try next available within 5 minutes
    for offset in range(1, 6):
        candidate_min = target_min + offset
        h, m = divmod(candidate_min, 60)
        candidate_time = f'{h:02d}:{m:02d}'
        if candidate_time in bars_dict:
            return bars_dict[candidate_time]['c']
    return None


def get_t2_close(ticker, eval_date):
    """Get T+2 close from daily data."""
    if ticker not in daily_data:
        return None
    dates = sorted(daily_data[ticker].keys())
    if eval_date not in dates:
        return None
    idx = dates.index(eval_date)
    if idx + 2 >= len(dates):
        return None
    t2_date = dates[idx + 2]
    return daily_data[ticker][t2_date]['close']


# ── E16: Build PnL matrix ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("E16: ENTRY TIMING SWEEP (SHORT T+2)")
print("=" * 80)

# For each trade and each entry time, compute PnL
pnl_matrix = {}  # {entry_time: [pnl values]}
for et in ENTRY_TIMES:
    pnl_matrix[et] = []

trade_details = []  # Store per-trade info for E19

for _, row in shorts.iterrows():
    ticker = row['ticker']
    eval_date = row['eval_date']
    key = f"{ticker}__{eval_date}"

    if key not in ticker_bars:
        continue

    bars = ticker_bars[key]
    t2_close = get_t2_close(ticker, eval_date)
    if t2_close is None:
        continue

    trade_info = {
        'ticker': ticker,
        'eval_date': eval_date,
        'or_high': row['or_high'],
        'or_low': row['or_low'],
        'atr_pct': row['atr_pct'],
        'original_entry_price': row['entry_price'],
        't2_close': t2_close,
        'pnls': {},
    }

    for et in ENTRY_TIMES:
        entry_price = get_entry_price(bars, et)
        if entry_price is None or entry_price <= 0:
            continue
        # SHORT PnL: (entry - exit) / entry * 100
        pnl = (entry_price - t2_close) / entry_price * 100
        pnl_matrix[et].append(pnl)
        trade_info['pnls'][et] = pnl

    if trade_info['pnls']:
        trade_details.append(trade_info)

print(f"Trades with minute bars + T+2 data: {len(trade_details)}")


def compute_metrics(pnl_list):
    """Compute standard metrics for a PnL list."""
    arr = np.array(pnl_list)
    n = len(arr)
    if n == 0:
        return {'n': 0, 'avg': np.nan, 'trimmed_mean': np.nan, 'median': np.nan,
                'win_rate': np.nan, 'sharpe': np.nan, 'std': np.nan, 'se': np.nan,
                'low_n': True}
    avg = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else np.nan
    return {
        'n': n,
        'avg': avg,
        'trimmed_mean': trim_mean(arr, proportiontocut=0.05) if n >= 10 else avg,
        'median': np.median(arr),
        'win_rate': np.mean(arr > 0) * 100,
        'sharpe': avg / std if std and std > 0 else np.nan,
        'std': std,
        'se': sem(arr) if n > 1 else np.nan,
        'low_n': n < 20,
    }


# Compute metrics for each entry time
e16_results = {}
for et in ENTRY_TIMES:
    e16_results[et] = compute_metrics(pnl_matrix[et])

# Print table
print(f"\n{'Entry Time':<12} {'N':>5} {'Avg PnL%':>10} {'Trim Mean':>10} {'Median':>10} "
      f"{'Win Rate':>10} {'Sharpe':>8} {'Std':>8} {'Flag':>6}")
print("-" * 90)
for et in ENTRY_TIMES:
    m = e16_results[et]
    flag = 'LOW-N' if m['low_n'] else ''
    print(f"{et:<12} {m['n']:>5d} {m['avg']:>10.3f} {m['trimmed_mean']:>10.3f} "
          f"{m['median']:>10.3f} {m['win_rate']:>9.1f}% {m['sharpe']:>8.3f} "
          f"{m['std']:>8.3f} {flag:>6}")

# Find optimal
valid_results = {et: m for et, m in e16_results.items() if not m['low_n']}
if valid_results:
    best_avg = max(valid_results.items(), key=lambda x: x[1]['avg'])
    best_trim = max(valid_results.items(), key=lambda x: x[1]['trimmed_mean'])
    best_sharpe = max(valid_results.items(), key=lambda x: x[1]['sharpe'] if not np.isnan(x[1]['sharpe']) else -999)

    print(f"\nOptimal entry time (avg PnL):      {best_avg[0]}  ({best_avg[1]['avg']:.3f}%)")
    print(f"Optimal entry time (trimmed mean): {best_trim[0]}  ({best_trim[1]['trimmed_mean']:.3f}%)")
    print(f"Optimal entry time (Sharpe):       {best_sharpe[0]}  ({best_sharpe[1]['sharpe']:.3f})")

    baseline = e16_results['09:30']
    print(f"\nBaseline (09:30): avg={baseline['avg']:.3f}%, trim={baseline['trimmed_mean']:.3f}%, "
          f"WR={baseline['win_rate']:.1f}%, Sharpe={baseline['sharpe']:.3f}")
    print(f"Improvement (avg):    {best_avg[1]['avg'] - baseline['avg']:+.3f}%")
    print(f"Improvement (trim):   {best_trim[1]['trimmed_mean'] - baseline['trimmed_mean']:+.3f}%")
    print(f"Improvement (Sharpe): {best_sharpe[1]['sharpe'] - baseline['sharpe']:+.3f}")

# ── E16 Charts ─────────────────────────────────────────────────────────────────
print("\nGenerating E16 charts...")

fig, axes = plt.subplots(3, 2, figsize=(18, 20))
fig.suptitle('E16: SHORT Entry Timing Sweep (T+2 Outcome)', fontsize=16, fontweight='bold', y=0.98)

x_labels = ENTRY_TIMES
x_pos = range(len(ENTRY_TIMES))
avgs = [e16_results[et]['avg'] for et in ENTRY_TIMES]
trims = [e16_results[et]['trimmed_mean'] for et in ENTRY_TIMES]
wrs = [e16_results[et]['win_rate'] for et in ENTRY_TIMES]
sharpes = [e16_results[et]['sharpe'] for et in ENTRY_TIMES]
ses = [e16_results[et]['se'] for et in ENTRY_TIMES]
stds = [e16_results[et]['std'] for et in ENTRY_TIMES]

# Panel 1: Avg PnL with std error band
ax = axes[0, 0]
ax.plot(x_pos, avgs, 'b-o', markersize=5, linewidth=2, label='Avg PnL')
upper = [a + s for a, s in zip(avgs, ses)]
lower = [a - s for a, s in zip(avgs, ses)]
ax.fill_between(x_pos, lower, upper, alpha=0.2, color='blue')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
peak_idx = np.nanargmax(avgs)
ax.annotate(f'Peak: {ENTRY_TIMES[peak_idx]}\n{avgs[peak_idx]:.2f}%',
            xy=(peak_idx, avgs[peak_idx]), xytext=(peak_idx + 2, avgs[peak_idx] + 1),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Avg T+2 PnL (%)')
ax.set_title('1. Average T+2 PnL by Entry Time (+/- 1 SE)')
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 2: Trimmed Mean
ax = axes[0, 1]
ax.plot(x_pos, trims, 'g-s', markersize=5, linewidth=2, label='Trimmed Mean (5-95%)')
peak_idx_t = np.nanargmax(trims)
ax.annotate(f'Peak: {ENTRY_TIMES[peak_idx_t]}\n{trims[peak_idx_t]:.2f}%',
            xy=(peak_idx_t, trims[peak_idx_t]), xytext=(peak_idx_t + 2, trims[peak_idx_t] + 1),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red', fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Trimmed Mean T+2 PnL (%)')
ax.set_title('2. Trimmed Mean T+2 PnL by Entry Time')
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 3: Win Rate
ax = axes[1, 0]
ax.plot(x_pos, wrs, 'r-^', markersize=5, linewidth=2, label='Win Rate')
peak_idx_w = np.nanargmax(wrs)
ax.annotate(f'Peak: {ENTRY_TIMES[peak_idx_w]}\n{wrs[peak_idx_w]:.1f}%',
            xy=(peak_idx_w, wrs[peak_idx_w]), xytext=(peak_idx_w + 2, wrs[peak_idx_w] + 2),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red', fontweight='bold')
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Win Rate (%)')
ax.set_title('3. Win Rate by Entry Time')
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 4: Sharpe
ax = axes[1, 1]
ax.plot(x_pos, sharpes, 'm-D', markersize=5, linewidth=2, label='Sharpe')
peak_idx_s = np.nanargmax([s if not np.isnan(s) else -999 for s in sharpes])
ax.annotate(f'Peak: {ENTRY_TIMES[peak_idx_s]}\n{sharpes[peak_idx_s]:.3f}',
            xy=(peak_idx_s, sharpes[peak_idx_s]), xytext=(peak_idx_s + 2, sharpes[peak_idx_s] + 0.02),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red', fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Sharpe Ratio')
ax.set_title('4. Sharpe Ratio by Entry Time')
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 5: Heatmap — PnL distribution by entry time
ax = axes[2, 0]
# Build histogram matrix
pnl_bins = np.linspace(-20, 20, 41)
hist_matrix = np.zeros((len(pnl_bins) - 1, len(ENTRY_TIMES)))
for i, et in enumerate(ENTRY_TIMES):
    if pnl_matrix[et]:
        counts, _ = np.histogram(pnl_matrix[et], bins=pnl_bins)
        hist_matrix[:, i] = counts
im = ax.imshow(hist_matrix, aspect='auto', cmap='YlOrRd', origin='lower',
               extent=[-0.5, len(ENTRY_TIMES) - 0.5, pnl_bins[0], pnl_bins[-1]])
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('T+2 PnL (%)')
ax.set_title('5. PnL Distribution Heatmap by Entry Time')
ax.axhline(y=0, color='white', linestyle='--', alpha=0.7, linewidth=1)
plt.colorbar(im, ax=ax, label='Count')

# Panel 6: Best 3 vs baseline
ax = axes[2, 1]
# Sort by avg PnL, pick top 3 with n >= 20
sorted_times = sorted([(et, e16_results[et]) for et in ENTRY_TIMES if not e16_results[et]['low_n']],
                       key=lambda x: x[1]['avg'], reverse=True)
top3 = sorted_times[:3]
compare_times = ['09:30'] + [t[0] for t in top3 if t[0] != '09:30']
compare_times = list(dict.fromkeys(compare_times))[:4]  # deduplicate, keep order

bar_x = range(len(compare_times))
bar_avgs = [e16_results[t]['avg'] for t in compare_times]
bar_trims = [e16_results[t]['trimmed_mean'] for t in compare_times]
bar_width = 0.35
bars1 = ax.bar([x - bar_width/2 for x in bar_x], bar_avgs, bar_width, label='Avg PnL', color='steelblue')
bars2 = ax.bar([x + bar_width/2 for x in bar_x], bar_trims, bar_width, label='Trimmed Mean', color='coral')
ax.set_xticks(bar_x)
ax.set_xticklabels(compare_times, fontsize=10)
ax.set_ylabel('T+2 PnL (%)')
ax.set_title('6. Best Entry Times vs 09:30 Baseline')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIG_E16, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_E16}")


# ══════════════════════════════════════════════════════════════════════════════
# E19: Entry Offset Stratified by Opening Volatility
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("E19: ENTRY OFFSET STRATIFIED BY OPENING VOLATILITY")
print("=" * 80)

# Compute or_width_atr for each trade
for td in trade_details:
    or_width = td['or_high'] - td['or_low']
    entry_price = td['original_entry_price']
    atr_pct = td['atr_pct']
    if atr_pct > 0 and entry_price > 0:
        atr_dollar = entry_price * atr_pct / 100
        td['or_width'] = or_width
        td['or_width_atr'] = or_width / atr_dollar
    else:
        td['or_width'] = or_width
        td['or_width_atr'] = np.nan

# Stratify into buckets
buckets = {
    'Tight (< 0.5 ATR)': [],
    'Normal (0.5-1.0 ATR)': [],
    'Wide (>= 1.0 ATR)': [],
}
bucket_keys = list(buckets.keys())

for td in trade_details:
    owr = td.get('or_width_atr', np.nan)
    if np.isnan(owr):
        continue
    if owr < 0.5:
        buckets['Tight (< 0.5 ATR)'].append(td)
    elif owr < 1.0:
        buckets['Normal (0.5-1.0 ATR)'].append(td)
    else:
        buckets['Wide (>= 1.0 ATR)'].append(td)

print(f"\nBucket sizes:")
for bname, trades in buckets.items():
    print(f"  {bname}: {len(trades)} trades")

# Compute metrics per (bucket, entry_time)
e19_results = {}
for bname in bucket_keys:
    e19_results[bname] = {}
    for et in ENTRY_TIMES:
        pnls = [td['pnls'][et] for td in buckets[bname] if et in td['pnls']]
        e19_results[bname][et] = compute_metrics(pnls)

# Print tables per bucket
for bname in bucket_keys:
    print(f"\n--- {bname} (n={len(buckets[bname])}) ---")
    print(f"{'Entry Time':<12} {'N':>5} {'Avg PnL%':>10} {'Trim Mean':>10} {'Win Rate':>10} {'Flag':>6}")
    print("-" * 55)
    for et in ENTRY_TIMES:
        m = e19_results[bname][et]
        if m['n'] == 0:
            continue
        flag = 'LOW-N' if m['low_n'] else ''
        print(f"{et:<12} {m['n']:>5d} {m['avg']:>10.3f} {m['trimmed_mean']:>10.3f} "
              f"{m['win_rate']:>9.1f}% {flag:>6}")

# Find optimal per bucket
print("\n\nOptimal entry times per bucket:")
print(f"{'Bucket':<25} {'Best Time':>10} {'Avg PnL':>10} {'Trim Mean':>10} {'vs 09:30':>10}")
print("-" * 70)
e19_optimal = {}
for bname in bucket_keys:
    # Use trimmed mean for robustness; filter for n >= 10 (relaxed from 20 for buckets)
    valid = {et: m for et, m in e19_results[bname].items() if m['n'] >= 10}
    if not valid:
        valid = {et: m for et, m in e19_results[bname].items() if m['n'] >= 5}
    if valid:
        best_et = max(valid.items(), key=lambda x: x[1]['trimmed_mean'])
        baseline_m = e19_results[bname].get('09:30', {'trimmed_mean': np.nan, 'avg': np.nan})
        improvement = best_et[1]['trimmed_mean'] - baseline_m['trimmed_mean'] if not np.isnan(baseline_m['trimmed_mean']) else np.nan
        e19_optimal[bname] = {
            'optimal_time': best_et[0],
            'avg_pnl': best_et[1]['avg'],
            'trimmed_mean': best_et[1]['trimmed_mean'],
            'n': best_et[1]['n'],
            'improvement_vs_0930': improvement,
        }
        print(f"{bname:<25} {best_et[0]:>10} {best_et[1]['avg']:>10.3f} "
              f"{best_et[1]['trimmed_mean']:>10.3f} {improvement:>+10.3f}")
    else:
        print(f"{bname:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

# Key question answer
print("\nKey question: Does optimal entry time shift later for wider opening ranges?")
for bname in bucket_keys:
    if bname in e19_optimal:
        print(f"  {bname}: optimal at {e19_optimal[bname]['optimal_time']}")

# ── E19 Charts ─────────────────────────────────────────────────────────────────
print("\nGenerating E19 charts...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('E19: Entry Offset Stratified by Opening Volatility (SHORT T+2)',
             fontsize=16, fontweight='bold', y=0.98)

bucket_colors = {'Tight (< 0.5 ATR)': '#2196F3', 'Normal (0.5-1.0 ATR)': '#FF9800', 'Wide (>= 1.0 ATR)': '#F44336'}

# Panel 1: Overlaid avg PnL
ax = axes[0, 0]
for bname in bucket_keys:
    y_vals = [e19_results[bname][et]['avg'] for et in ENTRY_TIMES]
    ax.plot(x_pos, y_vals, '-o', markersize=4, linewidth=2, color=bucket_colors[bname],
            label=f'{bname} (n={len(buckets[bname])})')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Avg T+2 PnL (%)')
ax.set_title('1. Average T+2 PnL by Entry Time per Bucket')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Overlaid trimmed mean
ax = axes[0, 1]
for bname in bucket_keys:
    y_vals = [e19_results[bname][et]['trimmed_mean'] for et in ENTRY_TIMES]
    ax.plot(x_pos, y_vals, '-s', markersize=4, linewidth=2, color=bucket_colors[bname],
            label=f'{bname} (n={len(buckets[bname])})')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Trimmed Mean T+2 PnL (%)')
ax.set_title('2. Trimmed Mean T+2 PnL by Entry Time per Bucket')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Optimal entry time + avg PnL per bucket
ax = axes[1, 0]
bar_buckets = [b for b in bucket_keys if b in e19_optimal]
bar_x = range(len(bar_buckets))
bar_pnls = [e19_optimal[b]['trimmed_mean'] for b in bar_buckets]
bar_cols = [bucket_colors[b] for b in bar_buckets]
bars = ax.bar(bar_x, bar_pnls, color=bar_cols, edgecolor='black', linewidth=0.5)
ax.set_xticks(bar_x)
labels_with_time = [f"{b}\nOptimal: {e19_optimal[b]['optimal_time']}" for b in bar_buckets]
ax.set_xticklabels(labels_with_time, fontsize=9)
ax.set_ylabel('Trimmed Mean T+2 PnL at Optimal Time (%)')
ax.set_title('3. Optimal Entry Time & PnL per Volatility Bucket')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')
for bar, pnl in zip(bars, bar_pnls):
    ax.annotate(f'{pnl:.2f}%', xy=(bar.get_x() + bar.get_width()/2, pnl),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

# Panel 4: Scatter of or_width_atr vs optimal entry delay (per-trade best)
ax = axes[1, 1]
# For each trade, find the entry time that gives the best PnL
trade_owrs = []
trade_best_delays = []
for td in trade_details:
    owr = td.get('or_width_atr', np.nan)
    if np.isnan(owr) or not td['pnls']:
        continue
    # Find best entry time for this trade
    best_et = max(td['pnls'].items(), key=lambda x: x[1])
    best_time = best_et[0]
    delay_min = time_to_minutes(best_time) - time_to_minutes('09:30')
    trade_owrs.append(owr)
    trade_best_delays.append(delay_min)

ax.scatter(trade_owrs, trade_best_delays, alpha=0.4, s=30, c='steelblue', edgecolors='navy', linewidths=0.3)
# Add bucket medians
for bname in bar_buckets:
    bucket_owrs = [td.get('or_width_atr', np.nan) for td in buckets[bname]]
    bucket_owrs = [x for x in bucket_owrs if not np.isnan(x)]
    if bucket_owrs:
        med_owr = np.median(bucket_owrs)
        opt_time = e19_optimal[bname]['optimal_time']
        opt_delay = time_to_minutes(opt_time) - time_to_minutes('09:30')
        ax.scatter([med_owr], [opt_delay], s=200, marker='*', color=bucket_colors[bname],
                   edgecolors='black', linewidths=1, zorder=5, label=f'{bname}: {opt_time}')
ax.set_xlabel('Opening Range Width (fraction of ATR)')
ax.set_ylabel('Best Entry Delay from 09:30 (minutes)')
ax.set_title('4. OR Width vs Optimal Entry Delay')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
# Add trend line
if len(trade_owrs) > 5:
    z = np.polyfit(trade_owrs, trade_best_delays, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(trade_owrs), max(trade_owrs), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2,
            label=f'Trend (slope={z[0]:.1f} min/ATR)')
    ax.legend(fontsize=8, loc='upper left')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIG_E19, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_E19}")


# ── Save results JSON ─────────────────────────────────────────────────────────
output = {
    'experiment': 'V4 E16+E19: Entry Timing Sweep + Volatility Stratification',
    'total_short_trades': len(shorts),
    'trades_with_data': len(trade_details),
    'e16_timing_sweep': {},
    'e16_optimal': {},
    'e19_volatility_stratified': {},
    'e19_optimal': {},
}

for et in ENTRY_TIMES:
    m = e16_results[et]
    output['e16_timing_sweep'][et] = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                                       for k, v in m.items()}

if valid_results:
    output['e16_optimal'] = {
        'by_avg': {'time': best_avg[0], 'avg_pnl': best_avg[1]['avg']},
        'by_trimmed_mean': {'time': best_trim[0], 'trimmed_mean': best_trim[1]['trimmed_mean']},
        'by_sharpe': {'time': best_sharpe[0], 'sharpe': best_sharpe[1]['sharpe']},
        'baseline_0930': {
            'avg_pnl': baseline['avg'],
            'trimmed_mean': baseline['trimmed_mean'],
            'sharpe': baseline['sharpe'],
        },
    }

for bname in bucket_keys:
    output['e19_volatility_stratified'][bname] = {
        'n_trades': len(buckets[bname]),
        'timing_sweep': {},
    }
    for et in ENTRY_TIMES:
        m = e19_results[bname][et]
        output['e19_volatility_stratified'][bname]['timing_sweep'][et] = {
            k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in m.items()
        }

for bname, opt in e19_optimal.items():
    output['e19_optimal'][bname] = {
        k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in opt.items()
    }

with open(JSON_OUT, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved results: {JSON_OUT}")

print("\n" + "=" * 80)
print("DONE — E16 + E19 complete")
print("=" * 80)
