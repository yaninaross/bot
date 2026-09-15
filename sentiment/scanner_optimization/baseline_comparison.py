# %% Baseline Comparison: Old Scanner vs New Recipe (V1-V4 findings)
# Head-to-head on the same 316-trade dataset.

import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

_W = 80
print(f"\n{'='*_W}")
print(f"  BASELINE COMPARISON — Old Scanner vs New Recipe")
print(f"{'='*_W}")

# ── Load data ────────────────────────────────────────────────────────────────
print("\n  Loading data...")
df = pd.read_csv(os.path.join(RESULTS_DIR, 'v2_features_full.csv'))
with open(os.path.join(RESULTS_DIR, 'v3_minute_bars.json')) as f:
    mb_raw = json.load(f)
ticker_bars = mb_raw['ticker_bars']

with open(os.path.join(RESULTS_DIR, 'v2_daily_data.json')) as f:
    daily_data = json.load(f)

print(f"  Loaded {len(df)} trades, {len(ticker_bars)} minute bar sets, {len(daily_data)} tickers daily")

# ── Helper functions ─────────────────────────────────────────────────────────

def get_sorted_bars(ticker, date):
    key = f"{ticker}__{date}"
    bm = ticker_bars.get(key)
    if not bm:
        return None
    market_bars = [(t, b) for t, b in sorted(bm.items()) if '09:30' <= t < '16:00']
    return market_bars

def get_t2_close(ticker, date):
    td = daily_data.get(ticker, {})
    if not td:
        return None
    dates = sorted(td.keys())
    if date not in dates:
        return None
    idx = dates.index(date)
    if idx + 2 >= len(dates):
        return None
    return td[dates[idx + 2]]['close']

def find_first_local_high(bars, window=3, max_idx=15):
    if len(bars) < 2 * window + 1:
        return None
    for i in range(window, min(len(bars) - window, max_idx)):
        center_high = bars[i][1]['h']
        is_peak = True
        for j in range(1, window + 1):
            if bars[i - j][1]['h'] >= center_high or bars[i + j][1]['h'] >= center_high:
                is_peak = False
                break
        if is_peak:
            confirm_idx = i + window
            if confirm_idx < len(bars):
                return {
                    'peak_time': bars[i][0],
                    'peak_price': center_high,
                    'confirm_time': bars[confirm_idx][0],
                    'entry_price': bars[confirm_idx][1]['c'],
                    'delay_mins': confirm_idx
                }
    return None

def compute_orb(bars, window_mins=5):
    orb_bars = bars[:window_mins]
    if not orb_bars:
        return None, None
    orb_high = max(b[1]['h'] for b in orb_bars)
    orb_low = min(b[1]['l'] for b in orb_bars)
    return orb_high, orb_low

def orb_confirms_short(bars, orb_low, check_window=25):
    for t, b in bars[5:5 + check_window]:
        if b['c'] < orb_low:
            return True
    return False

def metrics(pnl_arr, label=''):
    pnl = np.array(pnl_arr)
    n = len(pnl)
    if n == 0:
        return {}
    avg = np.mean(pnl)
    med = np.median(pnl)
    std = np.std(pnl, ddof=1) if n > 1 else 0
    se = std / np.sqrt(n) if n > 1 else 0
    sharpe = avg / std if std > 0 else 0
    wr = np.mean(pnl > 0)
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = wins / losses if losses > 0 else np.inf
    tm = stats.trim_mean(pnl, 0.05) if n >= 20 else avg
    return {
        'n': n, 'avg': avg, 'median': med, 'trimmed_mean': tm,
        'std': std, 'se': se, 'sharpe': sharpe, 'win_rate': wr,
        'profit_factor': pf, 'max_loss': np.min(pnl), 'max_gain': np.max(pnl),
        'ci95_lo': avg - 1.96 * se, 'ci95_hi': avg + 1.96 * se,
        'total_pnl': np.sum(pnl)
    }

# ══════════════════════════════════════════════════════════════════════════════
# OLD BASELINE: LLM-flagged, both directions, 09:30 entry, 60m hold, 2.5% SL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*_W}")
print(f"  Computing OLD BASELINE...")
print(f"  Rules: LLM-flagged | both dirs | 09:30 entry | 60m hold | 2.5% SL | equal weight")
print(f"{'─'*_W}")

old_trades = []
for _, row in df.iterrows():
    tk, dt = row['ticker'], row['eval_date']
    sign = row['sign']  # +1 LONG, -1 SHORT
    bars = get_sorted_bars(tk, dt)
    if not bars or len(bars) < 60:
        continue

    entry_price = bars[0][1]['o']
    if entry_price <= 0:
        continue

    sl_pct = 0.025
    stop_price = entry_price * (1 - sl_pct) if sign == 1 else entry_price * (1 + sl_pct)

    exit_price = None
    stopped = False
    stop_time = None
    for i, (t, b) in enumerate(bars[:60]):
        if sign == 1 and b['l'] <= stop_price:
            exit_price = stop_price
            stopped = True
            stop_time = t
            break
        elif sign == -1 and b['h'] >= stop_price:
            exit_price = stop_price
            stopped = True
            stop_time = t
            break

    if not stopped:
        exit_price = bars[min(59, len(bars) - 1)][1]['c']

    pnl = sign * (exit_price - entry_price) / entry_price * 100

    old_trades.append({
        'ticker': tk, 'date': dt, 'direction': 'LONG' if sign == 1 else 'SHORT',
        'entry_price': entry_price, 'exit_price': exit_price,
        'pnl': pnl, 'stopped': stopped, 'weight': 1.0,
        'catalyst': row['catalyst_type'], 'gap_pct': row['gap_pct'],
        'is_earnings': row['is_earnings']
    })

df_old = pd.DataFrame(old_trades)
print(f"  Old baseline: {len(df_old)} trades ({(df_old['stopped']).sum():.0f} stopped)")

# ══════════════════════════════════════════════════════════════════════════════
# NEW BASELINE V1: SHORT only, filters, local high entry, T+2, gap-weighted
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*_W}")
print(f"  Computing NEW BASELINE V1...")
print(f"  Rules: SHORT only | excl earnings/deal/macro | ORB confirms SHORT")
print(f"         first local high entry (fallback 09:45) | T+2 hold | gap-weighted | no SL")
print(f"{'─'*_W}")

new_trades = []
excluded = {'not_short': 0, 'bad_catalyst': 0, 'no_bars': 0, 'no_orb_confirm': 0,
            'no_t2': 0, 'no_entry': 0}

day_median_gap = df[df['sign'] == -1].groupby('eval_date')['abs_gap_pct'].median().to_dict()

for _, row in df.iterrows():
    tk, dt = row['ticker'], row['eval_date']

    if row['sign'] != -1:
        excluded['not_short'] += 1
        continue

    cat = str(row['catalyst_type']).lower()
    if any(x in cat for x in ['earnings', 'deal', 'macro']):
        excluded['bad_catalyst'] += 1
        continue

    bars = get_sorted_bars(tk, dt)
    if not bars or len(bars) < 20:
        excluded['no_bars'] += 1
        continue

    orb_high, orb_low = compute_orb(bars, 5)
    if orb_high is None:
        excluded['no_bars'] += 1
        continue

    if not orb_confirms_short(bars, orb_low, 25):
        excluded['no_orb_confirm'] += 1
        continue

    t2_close = get_t2_close(tk, dt)
    if t2_close is None:
        excluded['no_t2'] += 1
        continue

    local_high = find_first_local_high(bars, window=3, max_idx=15)
    if local_high:
        entry_price = local_high['entry_price']
        entry_type = 'local_high'
        entry_time = local_high['confirm_time']
        delay = local_high['delay_mins']
    else:
        fb_idx = min(15, len(bars) - 1)
        entry_price = bars[fb_idx][1]['c']
        entry_type = 'fallback_0945'
        entry_time = bars[fb_idx][0]
        delay = 15

    if entry_price <= 0:
        excluded['no_entry'] += 1
        continue

    pnl = (entry_price - t2_close) / entry_price * 100

    gap = abs(row['gap_pct'])
    med_gap = day_median_gap.get(dt, gap)
    weight = gap / med_gap if med_gap > 0 else 1.0

    or_width = orb_high - orb_low
    atr_dollars = entry_price * row['atr_pct'] / 100 if row['atr_pct'] > 0 else or_width
    or_width_atr = or_width / atr_dollars if atr_dollars > 0 else 0
    orb_bucket = 'tight' if or_width_atr < 0.5 else ('normal' if or_width_atr < 1.0 else 'wide')

    new_trades.append({
        'ticker': tk, 'date': dt, 'direction': 'SHORT',
        'entry_price': entry_price, 'exit_price': t2_close,
        'entry_type': entry_type, 'entry_time': entry_time,
        'delay_mins': delay, 'pnl': pnl, 'weight': weight,
        'weighted_pnl': pnl * weight,
        'catalyst': row['catalyst_type'], 'gap_pct': row['gap_pct'],
        'orb_bucket': orb_bucket, 'or_width_atr': or_width_atr,
        'stopped': False, 'is_earnings': row['is_earnings']
    })

df_new = pd.DataFrame(new_trades)
print(f"  New baseline V1: {len(df_new)} trades")
print(f"  Excluded: {json.dumps(excluded, indent=4)}")

# ══════════════════════════════════════════════════════════════════════════════
# HEAD-TO-HEAD COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*_W}")
print(f"  HEAD-TO-HEAD COMPARISON")
print(f"{'='*_W}")

m_old = metrics(df_old['pnl'].values)
m_new_ew = metrics(df_new['pnl'].values)
m_new_wt = metrics(df_new['weighted_pnl'].values)

rows = [
    ('N trades',        'm', m_old['n'],        m_new_ew['n'],      m_new_wt['n']),
    ('Avg PnL (%)',     'p', m_old['avg'],       m_new_ew['avg'],    m_new_wt['avg']),
    ('Median PnL (%)',  'p', m_old['median'],    m_new_ew['median'], m_new_wt['median']),
    ('Trimmed Mean (%)', 'p', m_old['trimmed_mean'], m_new_ew['trimmed_mean'], m_new_wt['trimmed_mean']),
    ('Std Dev (%)',     'p', m_old['std'],       m_new_ew['std'],    m_new_wt['std']),
    ('Sharpe',          'f', m_old['sharpe'],    m_new_ew['sharpe'], m_new_wt['sharpe']),
    ('Win Rate',        'r', m_old['win_rate'],  m_new_ew['win_rate'], m_new_wt['win_rate']),
    ('Profit Factor',   'f', m_old['profit_factor'], m_new_ew['profit_factor'], m_new_wt['profit_factor']),
    ('Max Loss (%)',    'p', m_old['max_loss'],  m_new_ew['max_loss'], m_new_wt['max_loss']),
    ('Max Gain (%)',    'p', m_old['max_gain'],  m_new_ew['max_gain'], m_new_wt['max_gain']),
    ('95% CI Low (%)',  'p', m_old['ci95_lo'],   m_new_ew['ci95_lo'], m_new_wt['ci95_lo']),
    ('95% CI High (%)', 'p', m_old['ci95_hi'],  m_new_ew['ci95_hi'], m_new_wt['ci95_hi']),
    ('Total PnL (%)',   'p', m_old['total_pnl'], m_new_ew['total_pnl'], m_new_wt['total_pnl']),
]

def _fmt(v, t):
    if t == 'p': return f"{v:+.3f}"
    if t == 'f': return f"{v:.3f}"
    if t == 'r': return f"{v:.1%}"
    return f"{int(v)}"

hdr = f"  {'METRIC':<22} {'OLD BASELINE':>14} {'NEW (EW)':>14} {'NEW (GAP-WT)':>14} {'DELTA (WT)':>12}"
print(hdr)
print(f"  {'─'*76}")
for label, t, v_old, v_new_ew, v_new_wt in rows:
    delta = v_new_wt - v_old if isinstance(v_old, (int, float)) else ''
    d_str = _fmt(delta, t) if delta != '' else ''
    print(f"  {label:<22} {_fmt(v_old, t):>14} {_fmt(v_new_ew, t):>14} {_fmt(v_new_wt, t):>14} {d_str:>12}")

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*_W}")
print(f"  STATISTICAL SIGNIFICANCE")
print(f"{'─'*_W}")

t_stat, p_welch = stats.ttest_ind(df_new['pnl'].values, df_old['pnl'].values, equal_var=False)
u_stat, p_mw = stats.mannwhitneyu(df_new['pnl'].values, df_old['pnl'].values, alternative='greater')

sig_welch = 'SIGNIFICANT' if p_welch < 0.05 else 'not significant'
sig_mw = 'SIGNIFICANT' if p_mw < 0.05 else 'not significant'

print(f"  Welch's t-test:   t = {t_stat:.3f}, p = {p_welch:.4f}  ({sig_welch})")
print(f"  Mann-Whitney U:   U = {u_stat:.0f}, p = {p_mw:.4f}  ({sig_mw})")
print(f"\n  Note: n={len(df_old)} vs n={len(df_new)} over 4 days.")
print(f"  Significance is directional — live validation required.")

# Also test weighted PnL
t_stat_w, p_welch_w = stats.ttest_ind(df_new['weighted_pnl'].values, df_old['pnl'].values, equal_var=False)
u_stat_w, p_mw_w = stats.mannwhitneyu(df_new['weighted_pnl'].values, df_old['pnl'].values, alternative='greater')
sig_w = 'SIGNIFICANT' if p_welch_w < 0.05 else 'not significant'
sig_mw_w = 'SIGNIFICANT' if p_mw_w < 0.05 else 'not significant'
print(f"\n  Gap-weighted vs Old:")
print(f"  Welch's t-test:   t = {t_stat_w:.3f}, p = {p_welch_w:.4f}  ({sig_w})")
print(f"  Mann-Whitney U:   U = {u_stat_w:.0f}, p = {p_mw_w:.4f}  ({sig_mw_w})")

# ══════════════════════════════════════════════════════════════════════════════
# BREAKDOWNS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*_W}")
print(f"  BREAKDOWN: NEW BASELINE V1")
print(f"{'─'*_W}")

print(f"\n  By Catalyst:")
for cat, g in df_new.groupby('catalyst'):
    m = metrics(g['pnl'].values)
    flag = ' (LOW-N)' if m['n'] < 20 else ''
    print(f"    {cat:<25} n={m['n']:>3}  avg={m['avg']:+.2f}%  WR={m['win_rate']:.0%}{flag}")

print(f"\n  By ORB Bucket:")
for bucket, g in df_new.groupby('orb_bucket'):
    m = metrics(g['pnl'].values)
    flag = ' (LOW-N)' if m['n'] < 20 else ''
    print(f"    {bucket:<12} n={m['n']:>3}  avg={m['avg']:+.2f}%  WR={m['win_rate']:.0%}  Sharpe={m['sharpe']:.3f}{flag}")

print(f"\n  By Entry Type:")
for etype, g in df_new.groupby('entry_type'):
    m = metrics(g['pnl'].values)
    avg_delay = g['delay_mins'].mean()
    print(f"    {etype:<18} n={m['n']:>3}  avg={m['avg']:+.2f}%  WR={m['win_rate']:.0%}  delay={avg_delay:.1f}min")

print(f"\n  By Date:")
print(f"    {'Date':<12} {'OLD n':>6} {'OLD avg':>10} {'NEW n':>6} {'NEW avg':>10} {'NEW WR':>8}")
for dt in sorted(df['eval_date'].unique()):
    g_old = df_old[df_old['date'] == dt]
    g_new = df_new[df_new['date'] == dt]
    o_avg = g_old['pnl'].mean() if len(g_old) > 0 else 0
    n_avg = g_new['pnl'].mean() if len(g_new) > 0 else 0
    n_wr = (g_new['pnl'] > 0).mean() if len(g_new) > 0 else 0
    print(f"    {dt:<12} {len(g_old):>6} {o_avg:>+10.2f}% {len(g_new):>6} {n_avg:>+10.2f}% {n_wr:>7.0%}")

print(f"\n  Old Baseline — Direction Breakdown:")
for d, g in df_old.groupby('direction'):
    m = metrics(g['pnl'].values)
    print(f"    {d:<8} n={m['n']:>3}  avg={m['avg']:+.3f}%  WR={m['win_rate']:.0%}  stopped={g['stopped'].sum():.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE EXPANSION CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*_W}")
print(f"  UNIVERSE EXPANSION CONTEXT (from V3-E4)")
print(f"{'─'*_W}")
print(f"  Full gapper universe (>3% gap): 1,810 tickers across 4 dates")
print(f"  LLM-flagged subset: 316 tickers (17.5%)")
print(f"  Universe gap-fade avg PnL: +0.326% (open→close)")
print(f"  LLM-flagged avg PnL (60m): +0.014%")
print(f"  LLM does NOT add selection value.")
print(f"  NOTE: New baseline tested on 316 LLM-flagged subset only")
print(f"  (minute bars not available for full universe).")
print(f"  Live system should draw from full gapper universe.")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Old Baseline vs New Baseline V1 — Head-to-Head Comparison',
             fontsize=14, fontweight='bold', y=0.98)

# 1. PnL Distribution
ax = axes[0, 0]
bins = np.linspace(-15, 25, 40)
ax.hist(df_old['pnl'], bins=bins, alpha=0.55, color='#e74c3c', label=f'Old (n={len(df_old)})', density=True)
ax.hist(df_new['pnl'], bins=bins, alpha=0.55, color='#27ae60', label=f'New V1 (n={len(df_new)})', density=True)
ax.axvline(df_old['pnl'].mean(), color='#e74c3c', ls='--', lw=2, label=f"Old avg: {df_old['pnl'].mean():+.2f}%")
ax.axvline(df_new['pnl'].mean(), color='#27ae60', ls='--', lw=2, label=f"New avg: {df_new['pnl'].mean():+.2f}%")
ax.set_xlabel('PnL (%)')
ax.set_ylabel('Density')
ax.set_title('PnL Distribution')
ax.legend(fontsize=8)

# 2. Cumulative PnL
ax = axes[0, 1]
old_cum = df_old.sort_values('date')['pnl'].cumsum().values
new_cum = df_new.sort_values('date')['pnl'].cumsum().values
ax.plot(range(len(old_cum)), old_cum, color='#e74c3c', lw=2,
        label=f'Old (total: {old_cum[-1]:+.1f}%)')
ax.plot(range(len(new_cum)), new_cum, color='#27ae60', lw=2,
        label=f'New V1 (total: {new_cum[-1]:+.1f}%)')
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('Trade #')
ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('Cumulative PnL (Equal Weight)')
ax.legend(fontsize=9)

# 3. Win Rate by Date
ax = axes[0, 2]
dates = sorted(df['eval_date'].unique())
x = np.arange(len(dates))
old_wr = [df_old[df_old['date']==d]['pnl'].apply(lambda v: v>0).mean() if len(df_old[df_old['date']==d]) > 0 else 0 for d in dates]
new_wr = [(df_new[df_new['date']==d]['pnl'] > 0).mean() if len(df_new[df_new['date']==d]) > 0 else 0 for d in dates]
old_n = [len(df_old[df_old['date']==d]) for d in dates]
new_n = [len(df_new[df_new['date']==d]) for d in dates]
bars1 = ax.bar(x - 0.2, old_wr, 0.35, color='#e74c3c', alpha=0.8, label='Old')
bars2 = ax.bar(x + 0.2, new_wr, 0.35, color='#27ae60', alpha=0.8, label='New V1')
for i, (b, n) in enumerate(zip(bars1, old_n)):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f'n={n}', ha='center', fontsize=7)
for i, (b, n) in enumerate(zip(bars2, new_n)):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f'n={n}', ha='center', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([d[-5:] for d in dates])
ax.set_ylabel('Win Rate')
ax.set_title('Win Rate by Date')
ax.legend(fontsize=9)

# 4. Metric Comparison Bars
ax = axes[1, 0]
metric_labels = ['Avg PnL', 'Median', 'Trimmed\nMean', 'Sharpe\n(×10)']
old_vals = [m_old['avg'], m_old['median'], m_old['trimmed_mean'], m_old['sharpe'] * 10]
new_vals = [m_new_ew['avg'], m_new_ew['median'], m_new_ew['trimmed_mean'], m_new_ew['sharpe'] * 10]
x = np.arange(len(metric_labels))
ax.bar(x - 0.2, old_vals, 0.35, color='#e74c3c', alpha=0.8, label='Old')
ax.bar(x + 0.2, new_vals, 0.35, color='#27ae60', alpha=0.8, label='New V1')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.axhline(0, color='gray', lw=0.5)
ax.set_title('Key Metrics Comparison')
ax.legend(fontsize=9)

# 5. New V1 by ORB Bucket
ax = axes[1, 1]
orb_order = ['tight', 'normal', 'wide']
orb_data = []
for bucket in orb_order:
    g = df_new[df_new['orb_bucket'] == bucket]
    if len(g) > 0:
        orb_data.append((bucket, g['pnl'].mean(), len(g)))
    else:
        orb_data.append((bucket, 0, 0))
bx = np.arange(len(orb_data))
colors = ['#3498db', '#2ecc71', '#e67e22']
bars = ax.bar(bx, [d[1] for d in orb_data], color=colors, alpha=0.8)
for i, (_, avg, n) in enumerate(orb_data):
    flag = ' (LOW-N)' if n < 20 else ''
    ax.text(i, avg + 0.1, f'n={n}{flag}', ha='center', fontsize=8)
ax.set_xticks(bx)
ax.set_xticklabels([d[0] for d in orb_data])
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylabel('Avg PnL (%)')
ax.set_title('New V1: PnL by ORB Width')

# 6. Entry Type Comparison
ax = axes[1, 2]
entry_data = []
for etype in ['local_high', 'fallback_0945']:
    g = df_new[df_new['entry_type'] == etype]
    if len(g) > 0:
        entry_data.append((etype.replace('_', '\n'), g['pnl'].mean(), len(g), g['delay_mins'].mean()))
bx = np.arange(len(entry_data))
bars = ax.bar(bx, [d[1] for d in entry_data], color=['#27ae60', '#95a5a6'], alpha=0.8)
for i, (_, avg, n, delay) in enumerate(entry_data):
    ax.text(i, avg + 0.1, f'n={n}, {delay:.0f}min', ha='center', fontsize=8)
ax.set_xticks(bx)
ax.set_xticklabels([d[0] for d in entry_data])
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylabel('Avg PnL (%)')
ax.set_title('New V1: PnL by Entry Type')

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = os.path.join(BASE_DIR, 'baseline_comparison.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Chart saved: {chart_path}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE FROZEN RESULTS
# ══════════════════════════════════════════════════════════════════════════════
df_old.to_csv(os.path.join(RESULTS_DIR, 'baseline_old_frozen.csv'), index=False)
df_new.to_csv(os.path.join(RESULTS_DIR, 'baseline_v1_frozen.csv'), index=False)

results = {
    'frozen_date': '2026-09-12',
    'description': 'Frozen reference metrics for live trading comparison',
    'old_baseline': {
        'rules': 'LLM-flagged, both dirs, 09:30 entry, 60m hold, 2.5% SL, equal weight',
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in m_old.items()}
    },
    'new_baseline_v1_equal_weight': {
        'rules': 'SHORT only, excl earnings/deal/macro, ORB confirms, local high entry, T+2, no SL',
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in m_new_ew.items()}
    },
    'new_baseline_v1_gap_weighted': {
        'rules': 'Same as equal weight + gap_pct sizing',
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in m_new_wt.items()}
    },
    'significance': {
        'welch_t': float(t_stat), 'welch_p': float(p_welch),
        'mw_u': float(u_stat), 'mw_p': float(p_mw),
        'welch_t_weighted': float(t_stat_w), 'welch_p_weighted': float(p_welch_w),
    },
    'excluded_counts': excluded
}

with open(os.path.join(RESULTS_DIR, 'baseline_comparison_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Frozen baselines saved:")
print(f"    results/baseline_old_frozen.csv")
print(f"    results/baseline_v1_frozen.csv")
print(f"    results/baseline_comparison_results.json")
print(f"\n{'='*_W}")
print(f"  DONE")
print(f"{'='*_W}")
