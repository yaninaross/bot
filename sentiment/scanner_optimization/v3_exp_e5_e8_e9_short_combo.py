"""
V3 Scanner Optimization: Experiments E5, E8, E9
================================================
E5: SHORT Swing Entry Timing (market open vs 10:30 vs ORB break vs scanner entry)
E8: PM Path + ORB Combination (stacking filters for selection)
E9: Sympathy Tickers (lead vs follow within catalyst clusters)

Data: v2_features_full.csv (316 trades), v3_minute_bars.json, v2_daily_data.json
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
BASE = Path('/home/admin_yaninaso_altostrat_com/bot-RAG/sentiment/scanner_optimization')
RESULTS = BASE / 'results'
CSV_PATH = RESULTS / 'v2_features_full.csv'
MINUTE_PATH = RESULTS / 'v3_minute_bars.json'
DAILY_PATH = RESULTS / 'v2_daily_data.json'

LOW_N_THRESHOLD = 20


# ─── Load data ────────────────────────────────────────────────────────────────
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_csv(CSV_PATH)
print(f"CSV: {len(df)} trades ({(df.direction=='SHORT').sum()} SHORT, {(df.direction=='LONG').sum()} LONG)")

with open(MINUTE_PATH) as f:
    minute_data = json.load(f)
ticker_bars = minute_data['ticker_bars']
spy_bars = minute_data.get('spy_bars', {})
print(f"Minute bars: {len(ticker_bars)} ticker-dates")

with open(DAILY_PATH) as f:
    daily_data = json.load(f)
print(f"Daily data: {len(daily_data)} tickers")


# ─── Helper functions ─────────────────────────────────────────────────────────

def trimmed_mean(arr, pct=0.05):
    """Trimmed mean at 5-95%."""
    arr = np.array([x for x in arr if np.isfinite(x)])
    if len(arr) < 3:
        return np.nan
    return float(sp_stats.trim_mean(arr, pct))


def sharpe(arr):
    """Sharpe ratio (mean / std)."""
    arr = np.array([x for x in arr if np.isfinite(x)])
    if len(arr) < 3 or np.std(arr) == 0:
        return np.nan
    return float(np.mean(arr) / np.std(arr))


def metrics_dict(arr, label=""):
    """Compute standard metrics for a PnL array."""
    arr = np.array([x for x in arr if np.isfinite(x)])
    n = len(arr)
    if n == 0:
        return {'label': label, 'n': 0, 'avg': np.nan, 'trimmed': np.nan,
                'median': np.nan, 'win_rate': np.nan, 'sharpe': np.nan, 'low_n': True}
    return {
        'label': label,
        'n': n,
        'avg': float(np.mean(arr)),
        'trimmed': trimmed_mean(arr),
        'median': float(np.median(arr)),
        'win_rate': float(np.sum(arr > 0) / n * 100),
        'sharpe': sharpe(arr),
        'low_n': n < LOW_N_THRESHOLD,
    }


def print_metrics_table(metrics_list, title=""):
    """Print a formatted table of metrics."""
    if title:
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")
    header = f"{'Strategy':<35} {'n':>5} {'Avg%':>8} {'Trim%':>8} {'Med%':>8} {'WR%':>6} {'Sharpe':>7} {'Flag':>5}"
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        flag = " *" if m.get('low_n') else ""
        print(f"{m['label']:<35} {m['n']:>5} {m['avg']:>8.2f} {m['trimmed']:>8.2f} "
              f"{m['median']:>8.2f} {m['win_rate']:>6.1f} {m['sharpe']:>7.3f}{flag}")
    if any(m.get('low_n') for m in metrics_list):
        print("  * = low-n (< 20), interpret with caution")


def categorize_pm_retrace(val):
    """Categorize numeric pm_retrace into held/partial/full."""
    if pd.isna(val):
        return 'unknown'
    if val <= 0.15:
        return 'held'
    elif val >= 0.85:
        return 'full'
    else:
        return 'partial'


def get_t2_close(ticker, eval_date, daily_data):
    """Get closing price 2 trading days after eval_date from daily data."""
    if ticker not in daily_data:
        return np.nan
    dates = sorted(daily_data[ticker].keys())
    if eval_date not in dates:
        return np.nan
    idx = dates.index(eval_date)
    if idx + 2 >= len(dates):
        return np.nan
    t2_date = dates[idx + 2]
    return daily_data[ticker][t2_date]['close']


def get_price_at_time(ticker, eval_date, time_str, ticker_bars):
    """Get open price at a specific time from minute bars."""
    key = f"{ticker}__{eval_date}"
    if key not in ticker_bars:
        return np.nan
    bars = ticker_bars[key]
    if time_str in bars:
        return bars[time_str]['o']
    # Try nearby times (+/- 1 min)
    h, m = map(int, time_str.split(':'))
    for delta in [1, -1, 2, -2]:
        nm = m + delta
        nh = h
        if nm >= 60:
            nm -= 60
            nh += 1
        elif nm < 0:
            nm += 60
            nh -= 1
        alt = f"{nh:02d}:{nm:02d}"
        if alt in bars:
            return bars[alt]['o']
    return np.nan


def find_orb_break_below(ticker, eval_date, or_low, ticker_bars):
    """Find the first minute bar that closes below or_low after 09:30.
    Returns (time_str, close_price) or (None, NaN)."""
    key = f"{ticker}__{eval_date}"
    if key not in ticker_bars or pd.isna(or_low):
        return None, np.nan
    bars = ticker_bars[key]
    # Sort times, filter to after 09:35 (after opening range forms)
    sorted_times = sorted(bars.keys())
    for t in sorted_times:
        if t < '09:35':
            continue
        if t > '16:00':
            break
        bar = bars[t]
        if bar['c'] < or_low:
            return t, bar['c']
    return None, np.nan


# ─── Categorize pm_retrace ───────────────────────────────────────────────────
df['pm_retrace_cat'] = df['pm_retrace'].apply(categorize_pm_retrace)
df['orb_agrees'] = df['orb_dir'] == df['direction']

print(f"\npm_retrace categories: {df['pm_retrace_cat'].value_counts().to_dict()}")
print(f"ORB agrees: {df['orb_agrees'].value_counts().to_dict()}")


# =============================================================================
# E5: SHORT Swing Entry Timing
# =============================================================================
print("\n" + "=" * 80)
print("E5: SHORT SWING ENTRY TIMING")
print("=" * 80)

short_df = df[df.direction == 'SHORT'].copy()
print(f"\nSHORT trades: {len(short_df)}")

# Strategy 1: Market open (09:30) entry
# Strategy 2: 10:30 AM entry
# Strategy 3: First ORB break below or_low
# Strategy 4: Scanner entry (baseline) = pnl_t2_close

e5_results = []

for idx, row in short_df.iterrows():
    ticker = row['ticker']
    eval_date = row['eval_date']
    t2_close = get_t2_close(ticker, eval_date, daily_data)

    # Strategy 1: 09:30 entry
    price_0930 = get_price_at_time(ticker, eval_date, '09:30', ticker_bars)
    if np.isfinite(price_0930) and np.isfinite(t2_close) and price_0930 > 0:
        pnl_0930 = (price_0930 - t2_close) / price_0930 * 100  # SHORT: profit when price falls
    else:
        pnl_0930 = np.nan

    # Strategy 2: 10:30 entry
    price_1030 = get_price_at_time(ticker, eval_date, '10:30', ticker_bars)
    if np.isfinite(price_1030) and np.isfinite(t2_close) and price_1030 > 0:
        pnl_1030 = (price_1030 - t2_close) / price_1030 * 100
    else:
        pnl_1030 = np.nan

    # Strategy 3: ORB break below
    orb_time, orb_price = find_orb_break_below(ticker, eval_date, row.get('or_low'), ticker_bars)
    if orb_price is not np.nan and np.isfinite(orb_price) and np.isfinite(t2_close) and orb_price > 0:
        pnl_orb = (orb_price - t2_close) / orb_price * 100
    else:
        pnl_orb = np.nan

    # Strategy 4: Scanner baseline
    pnl_scanner = row['pnl_t2_close']

    e5_results.append({
        'ticker': ticker,
        'eval_date': eval_date,
        'entry_price': row['entry_price'],
        'gap_pct': row['gap_pct'],
        'price_0930': price_0930,
        'price_1030': price_1030,
        'orb_break_price': orb_price if orb_price is not np.nan else np.nan,
        'orb_break_time': orb_time,
        't2_close': t2_close,
        'pnl_0930': pnl_0930,
        'pnl_1030': pnl_1030,
        'pnl_orb_break': pnl_orb,
        'pnl_scanner': pnl_scanner,
    })

e5_df = pd.DataFrame(e5_results)

# Compute metrics for each strategy
strategies = {
    'Market Open (09:30)': 'pnl_0930',
    '10:30 AM Entry': 'pnl_1030',
    'ORB Break Below': 'pnl_orb_break',
    'Scanner Entry (baseline)': 'pnl_scanner',
}

e5_metrics = []
for label, col in strategies.items():
    arr = e5_df[col].dropna().values
    m = metrics_dict(arr, label)
    e5_metrics.append(m)

print_metrics_table(e5_metrics, "SHORT Entry Timing -> T+2 Close PnL")

# Show ORB break timing distribution
orb_times = e5_df['orb_break_time'].dropna()
print(f"\nORB break below or_low occurred in {len(orb_times)}/{len(e5_df)} SHORT trades")
if len(orb_times) > 0:
    print(f"  Earliest break: {orb_times.min()}, Latest: {orb_times.max()}")
    # Time distribution buckets
    time_buckets = {'09:35-09:59': 0, '10:00-10:29': 0, '10:30-10:59': 0,
                    '11:00-11:59': 0, '12:00-13:59': 0, '14:00-16:00': 0}
    for t in orb_times:
        if t < '10:00':
            time_buckets['09:35-09:59'] += 1
        elif t < '10:30':
            time_buckets['10:00-10:29'] += 1
        elif t < '11:00':
            time_buckets['10:30-10:59'] += 1
        elif t < '12:00':
            time_buckets['11:00-11:59'] += 1
        elif t < '14:00':
            time_buckets['12:00-13:59'] += 1
        else:
            time_buckets['14:00-16:00'] += 1
    print(f"  Time distribution: {time_buckets}")


# ─── E5 Charts ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('E5: SHORT Swing Entry Timing (T+2 PnL)', fontsize=14, fontweight='bold')

# Chart 1: Bar chart of avg T+2 PnL
labels = [m['label'] for m in e5_metrics]
avgs = [m['avg'] for m in e5_metrics]
ns = [m['n'] for m in e5_metrics]
colors = ['#4e79a7', '#59a14f', '#e15759', '#76b7b2']
bars = axes[0].bar(range(len(labels)), avgs, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_xticks(range(len(labels)))
axes[0].set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], rotation=25, ha='right', fontsize=8)
axes[0].set_ylabel('Avg T+2 PnL (%)')
axes[0].set_title('Average PnL by Entry Strategy')
axes[0].axhline(y=0, color='black', linewidth=0.5)
for bar, avg in zip(bars, avgs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{avg:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Chart 2: Box plots
box_data = []
box_labels = []
for label, col in strategies.items():
    arr = e5_df[col].dropna().values
    if len(arr) > 0:
        box_data.append(arr)
        box_labels.append(label.replace(' (baseline)', '\n(baseline)'))
bp = axes[1].boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=True,
                     flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_ylabel('T+2 PnL (%)')
axes[1].set_title('PnL Distribution by Strategy')
axes[1].axhline(y=0, color='red', linewidth=0.5, linestyle='--')
axes[1].tick_params(axis='x', rotation=20, labelsize=7)

# Chart 3: Cumulative PnL curves
for label, col in strategies.items():
    arr = e5_df[col].dropna().values
    if len(arr) > 0:
        # Sort by eval_date for cumulative
        sub = e5_df[['eval_date', col]].dropna().sort_values('eval_date')
        cum = sub[col].cumsum().values
        axes[2].plot(range(len(cum)), cum, label=label, linewidth=1.5)
axes[2].set_xlabel('Trade #')
axes[2].set_ylabel('Cumulative PnL (%)')
axes[2].set_title('Cumulative PnL')
axes[2].legend(fontsize=7, loc='best')
axes[2].axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(str(BASE / 'v3_e5_short_timing.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: v3_e5_short_timing.png")


# =============================================================================
# E8: PM Path + ORB Combination
# =============================================================================
print("\n" + "=" * 80)
print("E8: PM PATH + ORB COMBINATION")
print("=" * 80)

# ─── Analysis matrix: pm_retrace_cat x orb_agrees ─────────────────────────────
print("\n--- Full dataset: pm_retrace x orb_agrees -> Avg PnL 60m ---")

matrix_data = {}
for pm_cat in ['held', 'partial', 'full']:
    matrix_data[pm_cat] = {}
    for orb_ag in [True, False]:
        mask = (df['pm_retrace_cat'] == pm_cat) & (df['orb_agrees'] == orb_ag)
        sub = df[mask]
        pnl = sub['pnl_60m_nosl'].values
        m = metrics_dict(pnl, f"PM={pm_cat}, ORB={'agree' if orb_ag else 'disagree'}")
        matrix_data[pm_cat][orb_ag] = m

# Print as grid
print(f"\n{'':>20} | {'ORB Agrees':>35} | {'ORB Disagrees':>35}")
print(f"{'':>20} | {'avg':>8} {'wr%':>6} {'n':>5} {'trim':>8} | {'avg':>8} {'wr%':>6} {'n':>5} {'trim':>8}")
print("-" * 95)
for pm_cat in ['held', 'partial', 'full']:
    a = matrix_data[pm_cat][True]
    d = matrix_data[pm_cat][False]
    flag_a = " *" if a['low_n'] else ""
    flag_d = " *" if d['low_n'] else ""
    print(f"PM {pm_cat:>15} | {a['avg']:>8.2f} {a['win_rate']:>6.1f} {a['n']:>5}{flag_a:>3} {a['trimmed']:>8.2f} | "
          f"{d['avg']:>8.2f} {d['win_rate']:>6.1f} {d['n']:>5}{flag_d:>3} {d['trimmed']:>8.2f}")

# Interaction effect
print("\n--- Interaction Effect Analysis ---")
# Expected PnL from individual effects
all_pnl = df['pnl_60m_nosl'].mean()
for pm_cat in ['held', 'partial', 'full']:
    pm_mask = df['pm_retrace_cat'] == pm_cat
    pm_effect = df[pm_mask]['pnl_60m_nosl'].mean() - all_pnl
    orb_mask = df['orb_agrees'] == True
    orb_effect = df[orb_mask]['pnl_60m_nosl'].mean() - all_pnl
    expected_combo = all_pnl + pm_effect + orb_effect
    actual_combo = matrix_data[pm_cat][True]['avg']
    interaction = actual_combo - expected_combo
    n_combo = matrix_data[pm_cat][True]['n']
    flag = " (LOW N)" if n_combo < LOW_N_THRESHOLD else ""
    print(f"  PM={pm_cat:>8}: expected={expected_combo:>6.2f}%, actual={actual_combo:>6.2f}%, "
          f"interaction={interaction:>+6.2f}%, n={n_combo}{flag}")

# ─── Specific combo tests ─────────────────────────────────────────────────────
print("\n--- Specific Filter Combos ---")

e8_combos = []

# Combo 1: SHORT + ORB agrees + PM held -> T+2
mask = (df.direction == 'SHORT') & (df.orb_agrees) & (df.pm_retrace_cat == 'held')
arr = df[mask]['pnl_t2_close'].dropna().values
e8_combos.append(metrics_dict(arr, 'SHORT+ORB_agree+PM_held -> T+2'))

# Combo 2: SHORT + ORB agrees + PM full -> T+2
mask = (df.direction == 'SHORT') & (df.orb_agrees) & (df.pm_retrace_cat == 'full')
arr = df[mask]['pnl_t2_close'].dropna().values
e8_combos.append(metrics_dict(arr, 'SHORT+ORB_agree+PM_full -> T+2'))

# Combo 3: LONG + ORB agrees + PM held -> 60m
mask = (df.direction == 'LONG') & (df.orb_agrees) & (df.pm_retrace_cat == 'held')
arr = df[mask]['pnl_60m_nosl'].dropna().values
e8_combos.append(metrics_dict(arr, 'LONG+ORB_agree+PM_held -> 60m'))

# Combo 4: All stacked: direction + ORB + PM + n_runs >= 4
mask_long = (df.direction == 'LONG') & (df.orb_agrees) & (df.pm_retrace_cat == 'held') & (df.n_runs >= 4)
mask_short = (df.direction == 'SHORT') & (df.orb_agrees) & (df.pm_retrace_cat == 'held') & (df.n_runs >= 4)
arr_l = df[mask_long]['pnl_60m_nosl'].dropna().values
arr_s = df[mask_short]['pnl_t2_close'].dropna().values
e8_combos.append(metrics_dict(arr_l, 'LONG+ORB+PM_held+n_runs>=4 -> 60m'))
e8_combos.append(metrics_dict(arr_s, 'SHORT+ORB+PM_held+n_runs>=4 -> T+2'))

print_metrics_table(e8_combos, "Specific Filter Combos")

# ─── Filter stacking analysis ────────────────────────────────────────────────
print("\n--- Filter Stacking (marginal improvement) ---")

# LONG stacking at 60m
print("\n  LONG side (PnL @ 60m):")
stacking_long = []
# No filter
arr = df[df.direction == 'LONG']['pnl_60m_nosl'].dropna().values
stacking_long.append(metrics_dict(arr, 'LONG (no filter)'))
# +PM held
mask = (df.direction == 'LONG') & (df.pm_retrace_cat == 'held')
arr = df[mask]['pnl_60m_nosl'].dropna().values
stacking_long.append(metrics_dict(arr, '+ PM held'))
# +ORB agrees
mask = (df.direction == 'LONG') & (df.orb_agrees)
arr = df[mask]['pnl_60m_nosl'].dropna().values
stacking_long.append(metrics_dict(arr, '+ ORB agrees'))
# +PM held + ORB agrees
mask = (df.direction == 'LONG') & (df.pm_retrace_cat == 'held') & (df.orb_agrees)
arr = df[mask]['pnl_60m_nosl'].dropna().values
stacking_long.append(metrics_dict(arr, '+ PM held + ORB agrees'))
# +PM held + ORB agrees + n_runs >= 4
mask = (df.direction == 'LONG') & (df.pm_retrace_cat == 'held') & (df.orb_agrees) & (df.n_runs >= 4)
arr = df[mask]['pnl_60m_nosl'].dropna().values
stacking_long.append(metrics_dict(arr, '+ PM held + ORB + n_runs>=4'))
print_metrics_table(stacking_long)

# SHORT stacking at T+2
print("\n  SHORT side (PnL @ T+2 close):")
stacking_short = []
arr = df[df.direction == 'SHORT']['pnl_t2_close'].dropna().values
stacking_short.append(metrics_dict(arr, 'SHORT (no filter)'))
mask = (df.direction == 'SHORT') & (df.pm_retrace_cat == 'held')
arr = df[mask]['pnl_t2_close'].dropna().values
stacking_short.append(metrics_dict(arr, '+ PM held'))
mask = (df.direction == 'SHORT') & (df.orb_agrees)
arr = df[mask]['pnl_t2_close'].dropna().values
stacking_short.append(metrics_dict(arr, '+ ORB agrees'))
mask = (df.direction == 'SHORT') & (df.pm_retrace_cat == 'held') & (df.orb_agrees)
arr = df[mask]['pnl_t2_close'].dropna().values
stacking_short.append(metrics_dict(arr, '+ PM held + ORB agrees'))
mask = (df.direction == 'SHORT') & (df.pm_retrace_cat == 'held') & (df.orb_agrees) & (df.n_runs >= 4)
arr = df[mask]['pnl_t2_close'].dropna().values
stacking_short.append(metrics_dict(arr, '+ PM held + ORB + n_runs>=4'))
print_metrics_table(stacking_short)


# ─── E8 Charts ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('E8: PM Path + ORB Combination', fontsize=14, fontweight='bold')

# Chart 1: Heatmap pm_retrace x orb_agrees -> avg PnL 60m
pm_cats = ['held', 'partial', 'full']
orb_labels = ['ORB Agrees', 'ORB Disagrees']
heatmap_data = np.zeros((3, 2))
heatmap_n = np.zeros((3, 2), dtype=int)
for i, pm in enumerate(pm_cats):
    for j, orb_ag in enumerate([True, False]):
        heatmap_data[i, j] = matrix_data[pm][orb_ag]['avg']
        heatmap_n[i, j] = matrix_data[pm][orb_ag]['n']

im = axes[0, 0].imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
axes[0, 0].set_xticks(range(2))
axes[0, 0].set_xticklabels(orb_labels)
axes[0, 0].set_yticks(range(3))
axes[0, 0].set_yticklabels([f"PM {c}" for c in pm_cats])
for i in range(3):
    for j in range(2):
        flag = "*" if heatmap_n[i, j] < LOW_N_THRESHOLD else ""
        axes[0, 0].text(j, i, f"{heatmap_data[i,j]:.2f}%\nn={heatmap_n[i,j]}{flag}",
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white' if abs(heatmap_data[i,j]) > 1.5 else 'black')
fig.colorbar(im, ax=axes[0, 0], label='Avg PnL 60m (%)')
axes[0, 0].set_title('PM Retrace x ORB -> Avg PnL 60m')

# Chart 2: LONG filter stacking bar chart
labels_l = [m['label'] for m in stacking_long]
avgs_l = [m['avg'] for m in stacking_long]
ns_l = [m['n'] for m in stacking_long]
colors_stack = ['#bab0ac', '#4e79a7', '#59a14f', '#e15759', '#f28e2b']
bars_l = axes[0, 1].barh(range(len(labels_l)), avgs_l, color=colors_stack, edgecolor='black', linewidth=0.5)
axes[0, 1].set_yticks(range(len(labels_l)))
axes[0, 1].set_yticklabels([f"{l} (n={n})" for l, n in zip(labels_l, ns_l)], fontsize=8)
for bar, avg in zip(bars_l, avgs_l):
    x_pos = max(avg, 0) + 0.05
    axes[0, 1].text(x_pos, bar.get_y() + bar.get_height()/2, f'{avg:.2f}%',
                   va='center', fontsize=9, fontweight='bold')
axes[0, 1].set_xlabel('Avg PnL 60m (%)')
axes[0, 1].set_title('LONG: Filter Stacking (60m PnL)')
axes[0, 1].axvline(x=0, color='black', linewidth=0.5)
axes[0, 1].invert_yaxis()

# Chart 3: SHORT filter stacking bar chart
labels_s = [m['label'] for m in stacking_short]
avgs_s = [m['avg'] for m in stacking_short]
ns_s = [m['n'] for m in stacking_short]
bars_s = axes[1, 0].barh(range(len(labels_s)), avgs_s, color=colors_stack, edgecolor='black', linewidth=0.5)
axes[1, 0].set_yticks(range(len(labels_s)))
axes[1, 0].set_yticklabels([f"{l} (n={n})" for l, n in zip(labels_s, ns_s)], fontsize=8)
for bar, avg in zip(bars_s, avgs_s):
    x_pos = max(avg, 0) + 0.05
    axes[1, 0].text(x_pos, bar.get_y() + bar.get_height()/2, f'{avg:.2f}%',
                   va='center', fontsize=9, fontweight='bold')
axes[1, 0].set_xlabel('Avg PnL T+2 close (%)')
axes[1, 0].set_title('SHORT: Filter Stacking (T+2 PnL)')
axes[1, 0].axvline(x=0, color='black', linewidth=0.5)
axes[1, 0].invert_yaxis()

# Chart 4: SHORT combos bar chart at T+2
combo_labels = [m['label'] for m in e8_combos[:3]]
combo_avgs = [m['avg'] for m in e8_combos[:3]]
combo_ns = [m['n'] for m in e8_combos[:3]]
combo_colors = ['#4e79a7', '#e15759', '#59a14f']
bars_c = axes[1, 1].bar(range(len(combo_labels)), combo_avgs, color=combo_colors, edgecolor='black', linewidth=0.5)
axes[1, 1].set_xticks(range(len(combo_labels)))
axes[1, 1].set_xticklabels([f"{l}\n(n={n})" for l, n in zip(combo_labels, combo_ns)],
                           rotation=20, ha='right', fontsize=6)
for bar, avg in zip(bars_c, combo_avgs):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{avg:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1, 1].set_ylabel('Avg PnL (%)')
axes[1, 1].set_title('Specific Filter Combos')
axes[1, 1].axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(str(BASE / 'v3_e8_pm_orb_combo.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: v3_e8_pm_orb_combo.png")


# =============================================================================
# E9: Sympathy Tickers
# =============================================================================
print("\n" + "=" * 80)
print("E9: SYMPATHY TICKERS")
print("=" * 80)

# ─── Cluster by (eval_date, catalyst_type) ───────────────────────────────────
# Exclude 'unknown' catalysts and earnings (too generic)
cluster_df = df[~df.catalyst_type.isin(['unknown', 'earnings'])].copy()
cluster_df['cluster_key'] = cluster_df['eval_date'] + '_' + cluster_df['catalyst_type']

cluster_counts = cluster_df.groupby('cluster_key').size()
multi_clusters = cluster_counts[cluster_counts >= 2]

print(f"\nTotal cluster keys (date+catalyst): {len(cluster_counts)}")
print(f"Clusters with 2+ tickers: {len(multi_clusters)}")
print(f"Trades in multi-ticker clusters: {cluster_df[cluster_df.cluster_key.isin(multi_clusters.index)].shape[0]}")

if len(multi_clusters) > 0:
    print(f"\nCluster size distribution:")
    size_dist = multi_clusters.value_counts().sort_index()
    for sz, cnt in size_dist.items():
        print(f"  Size {sz}: {cnt} clusters")

    # Identify lead vs follow within each cluster
    lead_pnls = {'60m': [], 'eod': [], 't2': []}
    follow_pnls = {'60m': [], 'eod': [], 't2': []}
    cluster_details = []

    for ck in multi_clusters.index:
        sub = cluster_df[cluster_df.cluster_key == ck].copy()
        # Lead = largest absolute gap
        sub['abs_gap'] = sub['gap_pct'].abs()
        sub = sub.sort_values('abs_gap', ascending=False)

        lead_row = sub.iloc[0]
        follow_rows = sub.iloc[1:]

        lead_pnls['60m'].append(lead_row['pnl_60m_nosl'])
        lead_pnls['eod'].append(lead_row['pnl_eod_nosl'])
        lead_pnls['t2'].append(lead_row['pnl_t2_close'])

        for _, fr in follow_rows.iterrows():
            follow_pnls['60m'].append(fr['pnl_60m_nosl'])
            follow_pnls['eod'].append(fr['pnl_eod_nosl'])
            follow_pnls['t2'].append(fr['pnl_t2_close'])

        cluster_details.append({
            'cluster': ck,
            'size': len(sub),
            'lead_ticker': lead_row['ticker'],
            'lead_gap': lead_row['gap_pct'],
            'lead_dir': lead_row['direction'],
            'follow_tickers': list(follow_rows['ticker']),
            'follow_gaps': list(follow_rows['gap_pct']),
            'lead_pnl_60m': lead_row['pnl_60m_nosl'],
            'lead_pnl_eod': lead_row['pnl_eod_nosl'],
            'follow_avg_pnl_60m': follow_rows['pnl_60m_nosl'].mean(),
            'follow_avg_pnl_eod': follow_rows['pnl_eod_nosl'].mean(),
        })

    print(f"\n--- Lead vs Follow PnL Comparison ---")
    for horizon in ['60m', 'eod', 't2']:
        lead_arr = np.array([x for x in lead_pnls[horizon] if np.isfinite(x)])
        follow_arr = np.array([x for x in follow_pnls[horizon] if np.isfinite(x)])
        print(f"\n  Horizon: {horizon}")
        print(f"    Lead  (n={len(lead_arr)}): avg={np.mean(lead_arr):.2f}%, "
              f"trim={trimmed_mean(lead_arr):.2f}%, median={np.median(lead_arr):.2f}%, "
              f"wr={np.sum(lead_arr>0)/len(lead_arr)*100:.1f}%"
              f"{'  [LOW N]' if len(lead_arr) < LOW_N_THRESHOLD else ''}")
        print(f"    Follow (n={len(follow_arr)}): avg={np.mean(follow_arr):.2f}%, "
              f"trim={trimmed_mean(follow_arr):.2f}%, median={np.median(follow_arr):.2f}%, "
              f"wr={np.sum(follow_arr>0)/len(follow_arr)*100:.1f}%"
              f"{'  [LOW N]' if len(follow_arr) < LOW_N_THRESHOLD else ''}")

    # Gap size vs PnL within clusters
    print(f"\n--- Gap Size Quartiles vs PnL within Clusters ---")
    cluster_trades = cluster_df[cluster_df.cluster_key.isin(multi_clusters.index)].copy()
    # Rank within cluster by gap size
    cluster_trades['gap_rank'] = cluster_trades.groupby('cluster_key')['abs_gap_pct'].rank(ascending=False)
    cluster_trades['is_lead'] = cluster_trades['gap_rank'] == 1

    # Split all cluster trades into gap quartiles
    cluster_trades['gap_q'] = pd.qcut(cluster_trades['abs_gap_pct'], q=4, labels=['Q1(small)', 'Q2', 'Q3', 'Q4(large)'],
                                       duplicates='drop')
    gap_q_metrics = []
    for q in cluster_trades['gap_q'].unique():
        if pd.isna(q):
            continue
        sub = cluster_trades[cluster_trades['gap_q'] == q]
        m60 = metrics_dict(sub['pnl_60m_nosl'].dropna().values, f"Gap {q}")
        gap_q_metrics.append(m60)
    if gap_q_metrics:
        print_metrics_table(gap_q_metrics, "Gap Quartile -> PnL 60m (within clusters)")

    # Show cluster details
    print(f"\n--- Top Clusters (by size) ---")
    cd_df = pd.DataFrame(cluster_details).sort_values('size', ascending=False)
    for _, row in cd_df.head(10).iterrows():
        print(f"  {row['cluster']:>30} | size={row['size']} | lead={row['lead_ticker']} "
              f"(gap={row['lead_gap']:+.1f}%, {row['lead_dir']}) | "
              f"lead PnL_60m={row['lead_pnl_60m']:.2f}% | follow avg={row['follow_avg_pnl_60m']:.2f}%")

    # ─── Alternative approach: daily data cross-gaps ──────────────────────────
    print(f"\n--- Alternative: Cross-gap analysis using daily data ---")
    # For each eval_date, find ALL tickers that gapped > 2% in same direction
    unique_dates = df['eval_date'].unique()
    cross_gap_groups = []

    for date in unique_dates:
        day_trades = df[df.eval_date == date]
        for direction in ['LONG', 'SHORT']:
            dir_trades = day_trades[day_trades.direction == direction]
            if len(dir_trades) >= 2:
                cross_gap_groups.append({
                    'date': date,
                    'direction': direction,
                    'n_tickers': len(dir_trades),
                    'tickers': list(dir_trades['ticker']),
                    'gaps': list(dir_trades['gap_pct']),
                    'avg_pnl_60m': dir_trades['pnl_60m_nosl'].mean(),
                    'avg_pnl_eod': dir_trades['pnl_eod_nosl'].mean(),
                })

    print(f"  Same-date same-direction groups (n>=2): {len(cross_gap_groups)}")
    if cross_gap_groups:
        cg_df = pd.DataFrame(cross_gap_groups)
        print(f"  Avg group size: {cg_df['n_tickers'].mean():.1f}")
        print(f"  Total trades in groups: {cg_df['n_tickers'].sum()}")

else:
    print("\nNo multi-ticker clusters found -- sympathy analysis not possible with this data.")
    cluster_details = []
    lead_pnls = {'60m': [], 'eod': [], 't2': []}
    follow_pnls = {'60m': [], 'eod': [], 't2': []}


# ─── E9 Charts ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('E9: Sympathy Tickers Analysis', fontsize=14, fontweight='bold')

if len(multi_clusters) > 0 and len(lead_pnls['60m']) > 0:
    # Chart 1: Lead vs Follow PnL
    horizons = ['60m', 'eod', 't2']
    lead_avgs = [np.nanmean(lead_pnls[h]) for h in horizons]
    follow_avgs = [np.nanmean(follow_pnls[h]) for h in horizons]
    x = np.arange(len(horizons))
    w = 0.35
    bars1 = axes[0].bar(x - w/2, lead_avgs, w, label='Lead (largest gap)', color='#4e79a7', edgecolor='black', linewidth=0.5)
    bars2 = axes[0].bar(x + w/2, follow_avgs, w, label='Follow (smaller gap)', color='#e15759', edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x)
    lead_ns = [len([v for v in lead_pnls[h] if np.isfinite(v)]) for h in horizons]
    follow_ns = [len([v for v in follow_pnls[h] if np.isfinite(v)]) for h in horizons]
    axes[0].set_xticklabels([f"{h}\nLead n={ln}, Follow n={fn}" for h, ln, fn in zip(horizons, lead_ns, follow_ns)],
                            fontsize=8)
    axes[0].set_ylabel('Avg PnL (%)')
    axes[0].set_title('Lead vs Follow Ticker PnL')
    axes[0].legend()
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    for bar_set, avgs_set in [(bars1, lead_avgs), (bars2, follow_avgs)]:
        for bar, avg in zip(bar_set, avgs_set):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.1 if avg >= 0 else -0.3),
                        f'{avg:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Chart 2: Gap size vs PnL scatterplot within clusters
    if len(cluster_trades) > 0:
        lead_mask = cluster_trades['is_lead']
        axes[1].scatter(cluster_trades[lead_mask]['abs_gap_pct'],
                       cluster_trades[lead_mask]['pnl_60m_nosl'],
                       alpha=0.6, c='#4e79a7', label='Lead', s=50, edgecolors='black', linewidth=0.3)
        axes[1].scatter(cluster_trades[~lead_mask]['abs_gap_pct'],
                       cluster_trades[~lead_mask]['pnl_60m_nosl'],
                       alpha=0.6, c='#e15759', label='Follow', s=50, edgecolors='black', linewidth=0.3)
        axes[1].set_xlabel('Absolute Gap %')
        axes[1].set_ylabel('PnL 60m (%)')
        axes[1].set_title('Gap Size vs PnL (within clusters)')
        axes[1].legend()
        axes[1].axhline(y=0, color='black', linewidth=0.5)
        # Add correlation
        all_gaps = cluster_trades['abs_gap_pct'].values
        all_pnls = cluster_trades['pnl_60m_nosl'].values
        valid = np.isfinite(all_gaps) & np.isfinite(all_pnls)
        if valid.sum() > 5:
            r, p = sp_stats.pearsonr(all_gaps[valid], all_pnls[valid])
            axes[1].text(0.05, 0.95, f'r={r:.3f}, p={p:.3f}', transform=axes[1].transAxes,
                        fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
else:
    axes[0].text(0.5, 0.5, 'Insufficient cluster data', ha='center', va='center', transform=axes[0].transAxes)
    axes[1].text(0.5, 0.5, 'Insufficient cluster data', ha='center', va='center', transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig(str(BASE / 'v3_e9_sympathy.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: v3_e9_sympathy.png")


# =============================================================================
# Save combined results JSON
# =============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Convert numpy types for JSON serialization
def to_json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    elif isinstance(obj, np.ndarray):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


combined_results = {
    'experiment_date': '2026-09-12',
    'data_source': 'v2_features_full.csv (316 trades)',
    'E5_short_timing': {
        'description': 'SHORT swing entry timing comparison -> T+2 PnL',
        'n_short_trades': len(short_df),
        'strategies': to_json_safe(e5_metrics),
        'orb_break_rate': len(orb_times) / len(e5_df) if len(e5_df) > 0 else 0,
    },
    'E8_pm_orb_combo': {
        'description': 'PM retrace + ORB agreement stacking analysis',
        'heatmap': to_json_safe({
            pm: {str(orb): matrix_data[pm][orb] for orb in [True, False]}
            for pm in ['held', 'partial', 'full']
        }),
        'specific_combos': to_json_safe(e8_combos),
        'long_stacking': to_json_safe(stacking_long),
        'short_stacking': to_json_safe(stacking_short),
    },
    'E9_sympathy': {
        'description': 'Sympathy ticker analysis (lead vs follow within catalyst clusters)',
        'n_multi_clusters': len(multi_clusters) if len(multi_clusters) > 0 else 0,
        'lead_vs_follow': to_json_safe({
            h: {
                'lead_avg': float(np.nanmean(lead_pnls[h])) if lead_pnls[h] else None,
                'follow_avg': float(np.nanmean(follow_pnls[h])) if follow_pnls[h] else None,
                'lead_n': len([v for v in lead_pnls[h] if np.isfinite(v)]),
                'follow_n': len([v for v in follow_pnls[h] if np.isfinite(v)]),
            }
            for h in ['60m', 'eod', 't2']
        }),
        'cluster_details': to_json_safe(cluster_details[:20]),  # Top 20
    },
}

output_path = RESULTS / 'v3_e5_e8_e9_results.json'
with open(output_path, 'w') as f:
    json.dump(combined_results, f, indent=2, default=str)
print(f"Saved: {output_path}")

print("\n" + "=" * 80)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 80)
print(f"\nOutput files:")
print(f"  - v3_e5_short_timing.png")
print(f"  - v3_e8_pm_orb_combo.png")
print(f"  - v3_e9_sympathy.png")
print(f"  - results/v3_e5_e8_e9_results.json")
