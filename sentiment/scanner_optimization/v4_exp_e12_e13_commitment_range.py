"""
V4 Scanner Optimization: Experiments E12 and E13
=================================================
E12: Commitment Signal — N Consecutive Candles below OR low (SHORT filter)
E13: Range Expansion Filter — ORB width / ATR as signal quality predictor

Data: v2_features_full.csv (316 trades), v3_minute_bars.json, v2_daily_data.json
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats
from pathlib import Path

warnings.filterwarnings('ignore')

# --- Config -------------------------------------------------------------------
BASE = Path('os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../') + '/sentiment/scanner_optimization')
RESULTS = BASE / 'results'
CSV_PATH = RESULTS / 'v2_features_full.csv'
MINUTE_PATH = RESULTS / 'v3_minute_bars.json'
DAILY_PATH = RESULTS / 'v2_daily_data.json'

LOW_N_THRESHOLD = 20


# --- Load data ----------------------------------------------------------------
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_csv(CSV_PATH)
print(f"CSV: {len(df)} trades ({(df.direction=='SHORT').sum()} SHORT, "
      f"{(df.direction=='LONG').sum()} LONG)")

with open(MINUTE_PATH) as f:
    minute_data = json.load(f)
ticker_bars = minute_data['ticker_bars']
spy_bars = minute_data.get('spy_bars', {})
print(f"Minute bars: {len(ticker_bars)} ticker-dates")

with open(DAILY_PATH) as f:
    daily_data = json.load(f)
print(f"Daily data: {len(daily_data)} tickers")


# --- Helper functions ---------------------------------------------------------

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
        print(f"\n{'~' * 70}")
        print(f"  {title}")
        print(f"{'~' * 70}")
    header = (f"{'Strategy':<35} {'n':>5} {'Avg%':>8} {'Trim%':>8} "
              f"{'Med%':>8} {'WR%':>6} {'Sharpe':>7} {'Flag':>5}")
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        flag = " *" if m.get('low_n') else ""
        avg = m['avg'] if np.isfinite(m.get('avg', np.nan)) else 0
        trim = m['trimmed'] if np.isfinite(m.get('trimmed', np.nan)) else 0
        med = m['median'] if np.isfinite(m.get('median', np.nan)) else 0
        wr = m['win_rate'] if np.isfinite(m.get('win_rate', np.nan)) else 0
        sh = m['sharpe'] if np.isfinite(m.get('sharpe', np.nan)) else 0
        print(f"{m['label']:<35} {m['n']:>5} {avg:>8.2f} {trim:>8.2f} "
              f"{med:>8.2f} {wr:>6.1f} {sh:>7.3f}{flag}")
    if any(m.get('low_n') for m in metrics_list):
        print("  * = low-n (< 20), interpret with caution")


def get_t2_close(ticker, eval_date):
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


def get_open_930(ticker, eval_date):
    """Get the open price at 09:30 from minute bars."""
    key = f"{ticker}__{eval_date}"
    if key not in ticker_bars:
        return np.nan
    bars = ticker_bars[key]
    if '09:30' in bars:
        return bars['09:30']['o']
    return np.nan


def time_to_minutes(t_str):
    """Convert 'HH:MM' to minutes from midnight."""
    h, m = map(int, t_str.split(':'))
    return h * 60 + m


# =============================================================================
# E12: COMMITMENT SIGNAL -- N CONSECUTIVE CANDLES
# =============================================================================
print("\n" + "=" * 80)
print("E12: COMMITMENT SIGNAL -- N CONSECUTIVE CANDLES")
print("=" * 80)

short_df = df[df.direction == 'SHORT'].copy()
print(f"\nSHORT trades: {len(short_df)}")

# For each SHORT trade, find commitment confirmation for various N
N_VALUES = [1, 2, 3, 5, 7, 10]

# Build results per trade
commitment_results = []

for _, row in short_df.iterrows():
    ticker = row['ticker']
    eval_date = row['eval_date']
    or_low = row['or_low']
    or_high = row['or_high']

    # Get minute bars
    key = f"{ticker}__{eval_date}"
    if key not in ticker_bars or pd.isna(or_low):
        continue

    bars = ticker_bars[key]

    # Get T+2 close from daily data
    t2_close = get_t2_close(ticker, eval_date)
    if np.isnan(t2_close):
        continue

    # Get 09:30 open for baseline
    open_930 = get_open_930(ticker, eval_date)
    if np.isnan(open_930):
        continue

    # Sorted times from 09:35 onward (after ORB forms)
    sorted_times = sorted([t for t in bars.keys()
                           if '09:35' <= t <= '16:00'])

    # Build sequence of closes-below-or_low
    closes_below = []
    for t in sorted_times:
        closes_below.append((t, bars[t]['c'], bars[t]['c'] < or_low))

    trade_entry = {
        'ticker': ticker,
        'eval_date': eval_date,
        'or_low': or_low,
        'or_high': or_high,
        'open_930': open_930,
        't2_close': t2_close,
        # Baseline: open entry SHORT PnL
        'pnl_open': (open_930 - t2_close) / open_930 * 100,
    }

    # For each N, find first N consecutive closes below or_low
    for n_val in N_VALUES:
        consecutive = 0
        found_time = None
        found_price = None
        for t, close, is_below in closes_below:
            if is_below:
                consecutive += 1
                if consecutive >= n_val:
                    found_time = t
                    found_price = close
                    break
            else:
                consecutive = 0

        if found_time is not None:
            delay_minutes = time_to_minutes(found_time) - time_to_minutes('09:30')
            pnl = (found_price - t2_close) / found_price * 100
            trade_entry[f'n{n_val}_time'] = found_time
            trade_entry[f'n{n_val}_price'] = found_price
            trade_entry[f'n{n_val}_pnl'] = pnl
            trade_entry[f'n{n_val}_delay'] = delay_minutes
            trade_entry[f'n{n_val}_signal'] = True
        else:
            trade_entry[f'n{n_val}_time'] = None
            trade_entry[f'n{n_val}_price'] = np.nan
            trade_entry[f'n{n_val}_pnl'] = np.nan
            trade_entry[f'n{n_val}_delay'] = np.nan
            trade_entry[f'n{n_val}_signal'] = False

    commitment_results.append(trade_entry)

cdf = pd.DataFrame(commitment_results)
print(f"SHORT trades with minute bars + daily data: {len(cdf)}")

# --- Compute metrics for each N ---
print("\n--- COMMITMENT SIGNAL RESULTS ---")

all_e12_metrics = []

# Baseline: 09:30 open entry
baseline_pnl = cdf['pnl_open'].dropna().values
m_baseline = metrics_dict(baseline_pnl, 'Open entry (09:30)')
all_e12_metrics.append(m_baseline)

for n_val in N_VALUES:
    signal_mask = cdf[f'n{n_val}_signal']
    pnl_col = cdf.loc[signal_mask, f'n{n_val}_pnl'].dropna().values
    delay_col = cdf.loc[signal_mask, f'n{n_val}_delay'].dropna().values

    m = metrics_dict(pnl_col, f'N={n_val} consecutive')
    m['avg_delay'] = float(np.mean(delay_col)) if len(delay_col) > 0 else np.nan
    all_e12_metrics.append(m)

print_metrics_table(all_e12_metrics, "E12: SHORT T+2 PnL by Commitment N")

# Print delay table
print(f"\n{'~' * 50}")
print(f"  Entry Delay (minutes after 09:30)")
print(f"{'~' * 50}")
print(f"{'Strategy':<25} {'n':>5} {'AvgDelay':>10} {'MedDelay':>10}")
print("-" * 55)
for n_val in N_VALUES:
    signal_mask = cdf[f'n{n_val}_signal']
    delays = cdf.loc[signal_mask, f'n{n_val}_delay'].dropna().values
    n_trades = signal_mask.sum()
    avg_d = np.mean(delays) if len(delays) > 0 else np.nan
    med_d = np.median(delays) if len(delays) > 0 else np.nan
    print(f"N={n_val:<22} {n_trades:>5} {avg_d:>10.1f} {med_d:>10.1f}")

# Signal capture rate
print(f"\n{'~' * 50}")
print(f"  Signal Capture Rate")
print(f"{'~' * 50}")
total = len(cdf)
for n_val in N_VALUES:
    n_sig = cdf[f'n{n_val}_signal'].sum()
    print(f"N={n_val}: {n_sig}/{total} trades get signal ({n_sig/total*100:.1f}%)")

# --- E12 key question answer ---
print(f"\n{'~' * 70}")
print(f"  KEY FINDING: Does commitment filter beat 09:30 open?")
print(f"{'~' * 70}")
base_avg = m_baseline['avg']
for m in all_e12_metrics[1:]:
    delta = m['avg'] - base_avg if np.isfinite(m['avg']) else np.nan
    delta_str = f"{delta:+.2f}pp" if np.isfinite(delta) else "N/A"
    flag = " [LOW-N]" if m.get('low_n') else ""
    print(f"  {m['label']}: avg {m['avg']:.2f}% vs open {base_avg:.2f}% "
          f"(delta {delta_str}, delay ~{m.get('avg_delay',0):.0f}min){flag}")


# =============================================================================
# E12 CHARTS
# =============================================================================
print("\n--- Generating E12 charts ---")
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

# Panel 1: Line chart N -> avg T+2 PnL with trade count on secondary
ax1 = fig.add_subplot(gs[0, 0])
n_labels = ['Open'] + [str(n) for n in N_VALUES]
avg_pnls = [all_e12_metrics[0]['avg']]
pnl_stds = [np.std(baseline_pnl) / np.sqrt(len(baseline_pnl))]
n_trades = [all_e12_metrics[0]['n']]

for i, n_val in enumerate(N_VALUES):
    m = all_e12_metrics[i + 1]
    avg_pnls.append(m['avg'] if np.isfinite(m['avg']) else 0)
    signal_mask = cdf[f'n{n_val}_signal']
    pnl_arr = cdf.loc[signal_mask, f'n{n_val}_pnl'].dropna().values
    se = np.std(pnl_arr) / np.sqrt(len(pnl_arr)) if len(pnl_arr) > 1 else 0
    pnl_stds.append(se)
    n_trades.append(m['n'])

x = np.arange(len(n_labels))
color_pnl = '#2196F3'
color_n = '#FF9800'

ax1.errorbar(x, avg_pnls, yerr=pnl_stds, marker='o', color=color_pnl,
             linewidth=2, markersize=8, capsize=4, label='Avg T+2 PnL')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(n_labels)
ax1.set_xlabel('N (Consecutive Candles)')
ax1.set_ylabel('Avg T+2 PnL (%)', color=color_pnl)
ax1.tick_params(axis='y', labelcolor=color_pnl)
ax1.set_title('E12: Commitment Signal -- PnL vs Trade Count')

ax1b = ax1.twinx()
ax1b.bar(x, n_trades, alpha=0.25, color=color_n, width=0.4, label='Trade count')
ax1b.set_ylabel('Trade Count', color=color_n)
ax1b.tick_params(axis='y', labelcolor=color_n)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

# Panel 2: Bar chart N -> avg entry delay
ax2 = fig.add_subplot(gs[0, 1])
delays_avg = []
delays_med = []
for n_val in N_VALUES:
    signal_mask = cdf[f'n{n_val}_signal']
    d = cdf.loc[signal_mask, f'n{n_val}_delay'].dropna().values
    delays_avg.append(np.mean(d) if len(d) > 0 else 0)
    delays_med.append(np.median(d) if len(d) > 0 else 0)

x_n = np.arange(len(N_VALUES))
width = 0.35
ax2.bar(x_n - width/2, delays_avg, width, color='#E91E63', alpha=0.8, label='Mean delay')
ax2.bar(x_n + width/2, delays_med, width, color='#9C27B0', alpha=0.8, label='Median delay')
ax2.set_xticks(x_n)
ax2.set_xticklabels([f'N={n}' for n in N_VALUES])
ax2.set_ylabel('Entry Delay (minutes after 09:30)')
ax2.set_title('E12: Entry Delay by N')
ax2.legend()
for i, (a, m) in enumerate(zip(delays_avg, delays_med)):
    ax2.text(i - width/2, a + 0.5, f'{a:.0f}', ha='center', fontsize=8)
    ax2.text(i + width/2, m + 0.5, f'{m:.0f}', ha='center', fontsize=8)

# Panel 3: Table visualization
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')
table_data = []
col_labels = ['Strategy', 'n', 'Avg%', 'Trim%', 'Med%', 'WR%', 'Sharpe', 'Delay']
for i, m in enumerate(all_e12_metrics):
    delay = f"{m.get('avg_delay', 0):.0f}" if np.isfinite(m.get('avg_delay', 0)) else '-'
    if i == 0:
        delay = '0'
    row = [m['label'],
           str(m['n']),
           f"{m['avg']:.2f}" if np.isfinite(m['avg']) else '-',
           f"{m['trimmed']:.2f}" if np.isfinite(m['trimmed']) else '-',
           f"{m['median']:.2f}" if np.isfinite(m['median']) else '-',
           f"{m['win_rate']:.1f}" if np.isfinite(m['win_rate']) else '-',
           f"{m['sharpe']:.3f}" if np.isfinite(m['sharpe']) else '-',
           delay]
    table_data.append(row)

table = ax3.table(cellText=table_data, colLabels=col_labels, loc='center',
                  cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.6)
# Color header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#333333')
    table[0, j].set_text_props(color='white', fontweight='bold')
# Color best avg row
if len(all_e12_metrics) > 1:
    valid_metrics = [(i, m) for i, m in enumerate(all_e12_metrics) if np.isfinite(m['avg'])]
    if valid_metrics:
        best_idx = max(valid_metrics, key=lambda x: x[1]['avg'])[0]
        for j in range(len(col_labels)):
            table[best_idx + 1, j].set_facecolor('#E8F5E9')
ax3.set_title('E12: Summary Table', fontweight='bold', pad=20)

# Panel 4: Cumulative PnL curves for selected N values
ax4 = fig.add_subplot(gs[1, 1])
# Baseline open entry
sorted_trades = cdf.sort_values(['eval_date', 'ticker'])
cum_open = np.cumsum(sorted_trades['pnl_open'].fillna(0).values)
ax4.plot(range(len(cum_open)), cum_open, label='Open (09:30)', linewidth=2,
         color='black', linestyle='--')

plot_ns = [1, 3, 5]
colors_cum = ['#F44336', '#2196F3', '#4CAF50']
for n_val, color in zip(plot_ns, colors_cum):
    # Trades that received a signal
    mask = sorted_trades[f'n{n_val}_signal']
    pnls = sorted_trades.loc[mask, f'n{n_val}_pnl'].fillna(0).values
    cum = np.cumsum(pnls)
    ax4.plot(range(len(cum)), cum, label=f'N={n_val} (n={mask.sum()})',
             linewidth=1.5, color=color)

ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax4.set_xlabel('Trade #')
ax4.set_ylabel('Cumulative PnL (%)')
ax4.set_title('E12: Cumulative T+2 PnL')
ax4.legend(fontsize=9)

fig.suptitle('E12: Commitment Signal -- N Consecutive Candles Below OR Low (SHORT)',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig(BASE / 'v4_e12_commitment_signal.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {BASE / 'v4_e12_commitment_signal.png'}")


# =============================================================================
# E13: RANGE EXPANSION FILTER
# =============================================================================
print("\n" + "=" * 80)
print("E13: RANGE EXPANSION FILTER")
print("=" * 80)

# Compute ORB width metrics for all trades
df['or_width_pct'] = (df['or_high'] - df['or_low']) / df['entry_price'] * 100
df['or_width_atr'] = np.where(
    df['atr_pct'] > 0,
    (df['or_high'] - df['or_low']) / (df['entry_price'] * df['atr_pct'] / 100),
    np.nan
)

# Add ORB agreement column
df['orb_agrees'] = df['orb_dir'] == df['direction']

# Filter to trades with valid ORB width
valid_mask = df['or_width_atr'].notna() & np.isfinite(df['or_width_atr'])
edf = df[valid_mask].copy()
print(f"Trades with valid ORB width: {len(edf)}/{len(df)}")

# Distribution summary
print(f"\nor_width_atr distribution:")
print(f"  Mean:   {edf['or_width_atr'].mean():.3f}")
print(f"  Median: {edf['or_width_atr'].median():.3f}")
print(f"  Std:    {edf['or_width_atr'].std():.3f}")
print(f"  Min:    {edf['or_width_atr'].min():.3f}")
print(f"  Max:    {edf['or_width_atr'].max():.3f}")
for pct in [10, 25, 50, 75, 90]:
    print(f"  P{pct}:   {edf['or_width_atr'].quantile(pct/100):.3f}")

# --- Split analysis: tight / normal / wide ---
def bucket_orb(val):
    if val < 0.5:
        return 'tight'
    elif val < 1.0:
        return 'normal'
    else:
        return 'wide'

edf['range_bucket'] = edf['or_width_atr'].apply(bucket_orb)

print(f"\nRange buckets:")
for b in ['tight', 'normal', 'wide']:
    n = (edf['range_bucket'] == b).sum()
    print(f"  {b}: {n} trades ({n/len(edf)*100:.1f}%)")

# --- E13a: PnL by range bucket ---
print("\n--- E13a: PnL by Range Bucket ---")
horizons = ['pnl_30m_nosl', 'pnl_60m_nosl', 'pnl_120m_nosl', 'pnl_eod_nosl']
horizon_labels = ['30m', '60m', '120m', 'EOD']

# Also add T+2 if available
if 'pnl_t2_close' in edf.columns:
    horizons.append('pnl_t2_close')
    horizon_labels.append('T+2')

for bucket in ['tight', 'normal', 'wide']:
    bdata = edf[edf['range_bucket'] == bucket]
    n = len(bdata)
    flag = " [LOW-N]" if n < LOW_N_THRESHOLD else ""
    print(f"\n  {bucket.upper()} (n={n}){flag}:")
    for hz, lbl in zip(horizons, horizon_labels):
        vals = bdata[hz].dropna().values
        if len(vals) == 0:
            continue
        avg = np.mean(vals)
        trim = trimmed_mean(vals)
        med = np.median(vals)
        wr = np.sum(vals > 0) / len(vals) * 100
        print(f"    {lbl:>5}: avg={avg:>7.2f}%  trim={trim:>7.2f}%  "
              f"med={med:>7.2f}%  WR={wr:>5.1f}%  n={len(vals)}")

# Full metrics table for 60m
print("\n")
metrics_60m = []
for bucket in ['tight', 'normal', 'wide']:
    bdata = edf[edf['range_bucket'] == bucket]
    vals = bdata['pnl_60m_nosl'].dropna().values
    m = metrics_dict(vals, f'{bucket} (OR/ATR)')
    metrics_60m.append(m)
all_vals = edf['pnl_60m_nosl'].dropna().values
metrics_60m.append(metrics_dict(all_vals, 'ALL'))
print_metrics_table(metrics_60m, "E13a: 60m PnL by ORB Width Bucket")

# --- Also test 1.5x ATR threshold ---
print("\n--- E13a_alt: 1.5x ATR threshold ---")
def bucket_orb_1_5(val):
    if val < 0.5:
        return 'tight'
    elif val < 1.5:
        return 'normal'
    else:
        return 'wide'

edf['range_bucket_1_5'] = edf['or_width_atr'].apply(bucket_orb_1_5)
for bucket in ['tight', 'normal', 'wide']:
    bdata = edf[edf['range_bucket_1_5'] == bucket]
    vals = bdata['pnl_60m_nosl'].dropna().values
    n = len(vals)
    flag = " [LOW-N]" if n < LOW_N_THRESHOLD else ""
    avg = np.mean(vals) if len(vals) > 0 else np.nan
    trim = trimmed_mean(vals)
    wr = np.sum(vals > 0) / len(vals) * 100 if len(vals) > 0 else np.nan
    print(f"  {bucket:>7} (n={n:>3}){flag}: avg={avg:>7.2f}%  trim={trim:>7.2f}%  WR={wr:>5.1f}%")


# --- E13b: ORB agree vs disagree by range bucket ---
print("\n--- E13b: ORB Signal Quality by Range Bucket ---")

for bucket in ['tight', 'normal', 'wide']:
    bdata = edf[edf['range_bucket'] == bucket]
    n = len(bdata)
    flag = " [LOW-N]" if n < LOW_N_THRESHOLD else ""
    agree = bdata[bdata['orb_agrees']]
    disagree = bdata[~bdata['orb_agrees']]

    a_pnl = agree['pnl_60m_nosl'].dropna().values
    d_pnl = disagree['pnl_60m_nosl'].dropna().values

    a_avg = np.mean(a_pnl) if len(a_pnl) > 0 else np.nan
    d_avg = np.mean(d_pnl) if len(d_pnl) > 0 else np.nan
    spread = a_avg - d_avg if np.isfinite(a_avg) and np.isfinite(d_avg) else np.nan

    print(f"\n  {bucket.upper()} (n={n}){flag}:")
    print(f"    ORB agree   (n={len(a_pnl):>3}): avg 60m = {a_avg:>7.2f}%")
    print(f"    ORB disagree(n={len(d_pnl):>3}): avg 60m = {d_avg:>7.2f}%")
    spread_str = f"{spread:>+7.2f}pp" if np.isfinite(spread) else "N/A"
    print(f"    Spread: {spread_str}")

# Hypothesis test: is spread larger for tight than wide?
tight_agree = edf[(edf['range_bucket'] == 'tight') & edf['orb_agrees']]['pnl_60m_nosl'].dropna().values
tight_disagree = edf[(edf['range_bucket'] == 'tight') & ~edf['orb_agrees']]['pnl_60m_nosl'].dropna().values
wide_agree = edf[(edf['range_bucket'] == 'wide') & edf['orb_agrees']]['pnl_60m_nosl'].dropna().values
wide_disagree = edf[(edf['range_bucket'] == 'wide') & ~edf['orb_agrees']]['pnl_60m_nosl'].dropna().values

tight_spread = np.mean(tight_agree) - np.mean(tight_disagree) if len(tight_agree) > 0 and len(tight_disagree) > 0 else np.nan
wide_spread = np.mean(wide_agree) - np.mean(wide_disagree) if len(wide_agree) > 0 and len(wide_disagree) > 0 else np.nan

print(f"\n  Hypothesis test: tight spread vs wide spread")
print(f"    Tight ORB spread: {tight_spread:+.2f}pp" if np.isfinite(tight_spread) else "    Tight ORB spread: N/A")
print(f"    Wide ORB spread:  {wide_spread:+.2f}pp" if np.isfinite(wide_spread) else "    Wide ORB spread:  N/A")
if np.isfinite(tight_spread) and np.isfinite(wide_spread):
    if tight_spread > wide_spread:
        print(f"    --> Tight range has LARGER agree/disagree spread ({tight_spread - wide_spread:+.2f}pp)")
        print(f"        ORB direction is a better signal for tight-range stocks")
    else:
        print(f"    --> Wide range has LARGER agree/disagree spread ({wide_spread - tight_spread:+.2f}pp)")
        print(f"        ORB direction is a better signal for wide-range stocks")


# --- E13c: Direction interaction ---
print("\n--- E13c: Range Bucket x Direction ---")

for direction in ['LONG', 'SHORT']:
    ddata = edf[edf['direction'] == direction]
    print(f"\n  {direction}:")
    dir_metrics = []
    for bucket in ['tight', 'normal', 'wide']:
        bdata = ddata[ddata['range_bucket'] == bucket]
        vals = bdata['pnl_60m_nosl'].dropna().values
        m = metrics_dict(vals, f'{bucket}')
        dir_metrics.append(m)
    print_metrics_table(dir_metrics, f"  {direction}: 60m PnL by Range Bucket")


# --- E13d: Continuous analysis ---
print("\n--- E13d: Continuous Analysis ---")
valid_60m = edf[edf['pnl_60m_nosl'].notna()].copy()

# Correlation: or_width_atr vs pnl_60m
corr_pnl = np.corrcoef(valid_60m['or_width_atr'], valid_60m['pnl_60m_nosl'])[0, 1]
print(f"  Correlation or_width_atr vs pnl_60m: r = {corr_pnl:.4f}")

# Correlation: or_width_atr vs |pnl_60m| (volatility)
corr_abs = np.corrcoef(valid_60m['or_width_atr'],
                       np.abs(valid_60m['pnl_60m_nosl']))[0, 1]
print(f"  Correlation or_width_atr vs |pnl_60m|: r = {corr_abs:.4f}")

# Regression
slope, intercept, r_val, p_val, se = sp_stats.linregress(
    valid_60m['or_width_atr'], valid_60m['pnl_60m_nosl'])
print(f"  Regression: PnL_60m = {slope:.3f} * or_width_atr + {intercept:.3f}")
print(f"    R-squared = {r_val**2:.4f}, p-value = {p_val:.4f}")

# By direction
for direction in ['LONG', 'SHORT']:
    dd = valid_60m[valid_60m['direction'] == direction]
    if len(dd) > 5:
        r = np.corrcoef(dd['or_width_atr'], dd['pnl_60m_nosl'])[0, 1]
        sl, ic, rv, pv, _ = sp_stats.linregress(dd['or_width_atr'], dd['pnl_60m_nosl'])
        print(f"  {direction}: r = {r:.4f}, slope = {sl:.3f}, p = {pv:.4f}")


# =============================================================================
# E13 CHARTS
# =============================================================================
print("\n--- Generating E13 charts ---")
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

# Panel 1: Bar chart tight/normal/wide -> avg PnL at 60m with error bars
ax1 = fig.add_subplot(gs[0, 0])
buckets = ['tight', 'normal', 'wide']
bucket_colors = ['#4CAF50', '#FF9800', '#F44336']
avgs = []
ses = []
ns = []
for bucket in buckets:
    vals = edf[edf['range_bucket'] == bucket]['pnl_60m_nosl'].dropna().values
    avgs.append(np.mean(vals) if len(vals) > 0 else 0)
    ses.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
    ns.append(len(vals))

x = np.arange(len(buckets))
bars = ax1.bar(x, avgs, yerr=ses, color=bucket_colors, alpha=0.85,
               capsize=5, edgecolor='black', linewidth=0.5)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{b}\n(n={n})' for b, n in zip(buckets, ns)])
ax1.set_ylabel('Avg 60m PnL (%)')
ax1.set_title('E13: ORB Width Bucket vs 60m PnL')
# Add value labels
for i, (avg, n) in enumerate(zip(avgs, ns)):
    flag = " *" if n < LOW_N_THRESHOLD else ""
    ax1.text(i, avg + ses[i] + 0.05, f'{avg:.2f}%{flag}', ha='center', fontsize=10,
             fontweight='bold')

# Panel 2: Grouped bar — tight/normal/wide x ORB_agree/disagree -> avg PnL
ax2 = fig.add_subplot(gs[0, 1])
agree_avgs = []
disagree_avgs = []
agree_ns = []
disagree_ns = []
for bucket in buckets:
    bdata = edf[edf['range_bucket'] == bucket]
    a_vals = bdata[bdata['orb_agrees']]['pnl_60m_nosl'].dropna().values
    d_vals = bdata[~bdata['orb_agrees']]['pnl_60m_nosl'].dropna().values
    agree_avgs.append(np.mean(a_vals) if len(a_vals) > 0 else 0)
    disagree_avgs.append(np.mean(d_vals) if len(d_vals) > 0 else 0)
    agree_ns.append(len(a_vals))
    disagree_ns.append(len(d_vals))

width = 0.35
ax2.bar(x - width/2, agree_avgs, width, color='#2196F3', alpha=0.85,
        label='ORB Agrees', edgecolor='black', linewidth=0.5)
ax2.bar(x + width/2, disagree_avgs, width, color='#E91E63', alpha=0.85,
        label='ORB Disagrees', edgecolor='black', linewidth=0.5)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(buckets)
ax2.set_ylabel('Avg 60m PnL (%)')
ax2.set_title('E13: ORB Agree/Disagree by Range Bucket')
ax2.legend()
# Add n labels
for i in range(len(buckets)):
    yoff_a = 0.1 if agree_avgs[i] >= 0 else -0.3
    yoff_d = 0.1 if disagree_avgs[i] >= 0 else -0.3
    ax2.text(i - width/2, agree_avgs[i] + yoff_a, f'n={agree_ns[i]}',
             ha='center', fontsize=8)
    ax2.text(i + width/2, disagree_avgs[i] + yoff_d, f'n={disagree_ns[i]}',
             ha='center', fontsize=8)

# Panel 3: Scatter or_width_atr vs pnl_60m with regression
ax3 = fig.add_subplot(gs[1, 0])
for direction, color, marker in [('LONG', '#2196F3', 'o'), ('SHORT', '#F44336', 's')]:
    dd = valid_60m[valid_60m['direction'] == direction]
    ax3.scatter(dd['or_width_atr'], dd['pnl_60m_nosl'], c=color,
                marker=marker, alpha=0.4, s=30, label=direction)

# Overall regression line
x_reg = np.linspace(valid_60m['or_width_atr'].min(), valid_60m['or_width_atr'].max(), 100)
y_reg = slope * x_reg + intercept
ax3.plot(x_reg, y_reg, 'k--', linewidth=2,
         label=f'Reg: {slope:.2f}x + {intercept:.2f} (r={r_val:.3f})')
ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax3.set_xlabel('OR Width / ATR')
ax3.set_ylabel('60m PnL (%)')
ax3.set_title(f'E13: ORB Width vs 60m PnL (r={corr_pnl:.3f})')
ax3.legend(fontsize=9)

# Panel 4: Heatmap range_bucket x direction -> avg PnL
ax4 = fig.add_subplot(gs[1, 1])
heatmap_data = np.zeros((2, 3))
heatmap_labels = np.empty((2, 3), dtype=object)
directions = ['LONG', 'SHORT']
for i, direction in enumerate(directions):
    for j, bucket in enumerate(buckets):
        vals = edf[(edf['direction'] == direction) &
                   (edf['range_bucket'] == bucket)]['pnl_60m_nosl'].dropna().values
        avg = np.mean(vals) if len(vals) > 0 else 0
        n = len(vals)
        heatmap_data[i, j] = avg
        flag = "*" if n < LOW_N_THRESHOLD else ""
        heatmap_labels[i, j] = f'{avg:.2f}%\n(n={n}){flag}'

# Use a diverging colormap centered at zero
vmax = max(abs(heatmap_data.min()), abs(heatmap_data.max()))
im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto',
                vmin=-vmax, vmax=vmax)
ax4.set_xticks(range(3))
ax4.set_xticklabels(buckets)
ax4.set_yticks(range(2))
ax4.set_yticklabels(directions)
ax4.set_title('E13: Avg 60m PnL Heatmap')
plt.colorbar(im, ax=ax4, label='Avg PnL (%)')
for i in range(2):
    for j in range(3):
        ax4.text(j, i, heatmap_labels[i, j], ha='center', va='center',
                 fontsize=10, fontweight='bold')

fig.suptitle('E13: Range Expansion Filter -- ORB Width / ATR',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig(BASE / 'v4_e13_range_expansion.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {BASE / 'v4_e13_range_expansion.png'}")


# =============================================================================
# SAVE RESULTS JSON
# =============================================================================
print("\n--- Saving results JSON ---")

results = {
    'experiment': 'V4 E12+E13',
    'date': '2026-09-12',
    'E12_commitment_signal': {
        'description': 'N consecutive candles below OR low for SHORT entry',
        'short_trades_analyzed': len(cdf),
        'strategies': {}
    },
    'E13_range_expansion': {
        'description': 'ORB width / ATR as signal quality predictor',
        'trades_analyzed': len(edf),
        'or_width_atr_stats': {
            'mean': float(edf['or_width_atr'].mean()),
            'median': float(edf['or_width_atr'].median()),
            'std': float(edf['or_width_atr'].std()),
        },
        'buckets': {},
        'continuous': {
            'corr_pnl_60m': float(corr_pnl),
            'corr_abs_pnl_60m': float(corr_abs),
            'regression_slope': float(slope),
            'regression_intercept': float(intercept),
            'regression_r_squared': float(r_val**2),
            'regression_p_value': float(p_val),
        },
    }
}

# E12 strategies
for m in all_e12_metrics:
    key = m['label'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    entry = {k: (float(v) if isinstance(v, (np.floating, float)) and np.isfinite(v) else
                 (int(v) if isinstance(v, (int, np.integer)) else
                  (bool(v) if isinstance(v, (bool, np.bool_)) else str(v))))
             for k, v in m.items()}
    results['E12_commitment_signal']['strategies'][key] = entry

# E13 buckets
for bucket in ['tight', 'normal', 'wide']:
    bdata = edf[edf['range_bucket'] == bucket]
    bucket_entry = {'n': int(len(bdata))}
    for hz, lbl in zip(horizons, horizon_labels):
        vals = bdata[hz].dropna().values
        if len(vals) > 0:
            bucket_entry[f'avg_{lbl}'] = float(np.mean(vals))
            bucket_entry[f'trimmed_{lbl}'] = float(trimmed_mean(vals))
            bucket_entry[f'win_rate_{lbl}'] = float(np.sum(vals > 0) / len(vals) * 100)
    # ORB agree/disagree
    a_vals = bdata[bdata['orb_agrees']]['pnl_60m_nosl'].dropna().values
    d_vals = bdata[~bdata['orb_agrees']]['pnl_60m_nosl'].dropna().values
    bucket_entry['orb_agree_avg_60m'] = float(np.mean(a_vals)) if len(a_vals) > 0 else None
    bucket_entry['orb_disagree_avg_60m'] = float(np.mean(d_vals)) if len(d_vals) > 0 else None
    bucket_entry['orb_agree_n'] = int(len(a_vals))
    bucket_entry['orb_disagree_n'] = int(len(d_vals))
    results['E13_range_expansion']['buckets'][bucket] = bucket_entry

# Direction interaction
results['E13_range_expansion']['by_direction'] = {}
for direction in ['LONG', 'SHORT']:
    dir_entry = {}
    for bucket in ['tight', 'normal', 'wide']:
        vals = edf[(edf['direction'] == direction) &
                   (edf['range_bucket'] == bucket)]['pnl_60m_nosl'].dropna().values
        dir_entry[bucket] = {
            'n': int(len(vals)),
            'avg_60m': float(np.mean(vals)) if len(vals) > 0 else None,
            'trimmed_60m': float(trimmed_mean(vals)) if len(vals) > 0 else None,
        }
    results['E13_range_expansion']['by_direction'][direction] = dir_entry

with open(RESULTS / 'v4_e12_e13_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved: {RESULTS / 'v4_e12_e13_results.json'}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("\nE12: Commitment Signal")
print("-" * 50)
base_avg = all_e12_metrics[0]['avg']
best_m = max(all_e12_metrics, key=lambda m: m['avg'] if np.isfinite(m['avg']) else -999)
print(f"  Baseline (09:30 open): {base_avg:.2f}% avg T+2 PnL")
print(f"  Best strategy: {best_m['label']} = {best_m['avg']:.2f}% avg")
# Check if any N beats open
beaten = [m for m in all_e12_metrics[1:]
          if np.isfinite(m['avg']) and m['avg'] > base_avg and not m.get('low_n')]
if beaten:
    print(f"  --> {len(beaten)} commitment levels beat 09:30 open")
    for m in beaten:
        delta = m['avg'] - base_avg
        print(f"      {m['label']}: +{delta:.2f}pp vs open, "
              f"delay ~{m.get('avg_delay',0):.0f}min, n={m['n']}")
else:
    print(f"  --> NO commitment level robustly beats 09:30 open entry")
    print(f"      Commitment filter costs entry quality without filtering whipsaws")

print(f"\nE13: Range Expansion Filter")
print("-" * 50)
for bucket in ['tight', 'normal', 'wide']:
    vals = edf[edf['range_bucket'] == bucket]['pnl_60m_nosl'].dropna().values
    n = len(vals)
    flag = " [LOW-N]" if n < LOW_N_THRESHOLD else ""
    avg = np.mean(vals) if len(vals) > 0 else np.nan
    print(f"  {bucket:>7} (OR/ATR): avg 60m = {avg:>7.2f}%, n={n}{flag}")
print(f"  Correlation or_width_atr vs pnl_60m: r = {corr_pnl:.4f}")
if abs(corr_pnl) < 0.1:
    print(f"  --> WEAK/NO correlation: ORB width does not predict PnL magnitude")
elif corr_pnl > 0.1:
    print(f"  --> POSITIVE correlation: wider ORB associated with higher PnL")
else:
    print(f"  --> NEGATIVE correlation: tighter ORB associated with higher PnL")
print(f"  Correlation or_width_atr vs |pnl_60m|: r = {corr_abs:.4f}")
if corr_abs > 0.1:
    print(f"  --> Wider ORB does predict more volatility (higher |PnL|)")
else:
    print(f"  --> ORB width does NOT strongly predict PnL volatility")

print(f"\n{'=' * 80}")
print("DONE")
print(f"{'=' * 80}")
