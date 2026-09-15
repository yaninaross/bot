# %% Baseline Comparison V2 — All dates, multiple horizons
# Uses CSV pnl columns + minute bars for local high entry.

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

_W = 90
print(f"\n{'='*_W}")
print(f"  BASELINE COMPARISON V2 — All Dates, Multiple Horizons")
print(f"{'='*_W}")

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(RESULTS_DIR, 'v2_features_full.csv'))
with open(os.path.join(RESULTS_DIR, 'v3_minute_bars.json')) as f:
    mb_raw = json.load(f)
ticker_bars = mb_raw['ticker_bars']

print(f"  Loaded {len(df)} trades, {len(ticker_bars)} minute bar sets")
print(f"  Dates: {sorted(df['eval_date'].unique())}")

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_sorted_bars(ticker, date):
    key = f"{ticker}__{date}"
    bm = ticker_bars.get(key)
    if not bm:
        return None
    return [(t, b) for t, b in sorted(bm.items()) if '09:30' <= t < '16:00']

def find_first_local_high(bars, window=3, max_idx=15):
    if len(bars) < 2 * window + 1:
        return None
    for i in range(window, min(len(bars) - window, max_idx)):
        center_high = bars[i][1]['h']
        is_peak = all(bars[i-j][1]['h'] < center_high and bars[i+j][1]['h'] < center_high
                      for j in range(1, window + 1))
        if is_peak:
            ci = i + window
            if ci < len(bars):
                return {'entry_price': bars[ci][1]['c'], 'delay': ci,
                        'peak_time': bars[i][0], 'confirm_time': bars[ci][0]}
    return None

def compute_orb_confirms_short(bars):
    if len(bars) < 30:
        return False, None, None
    orb_high = max(b[1]['h'] for b in bars[:5])
    orb_low = min(b[1]['l'] for b in bars[:5])
    for t, b in bars[5:30]:
        if b['c'] < orb_low:
            return True, orb_high, orb_low
    return False, orb_high, orb_low

def compute_pnl_from_entry(entry_price, bars, horizon_mins):
    if entry_price <= 0 or len(bars) < horizon_mins:
        return None
    exit_bar = bars[min(horizon_mins - 1, len(bars) - 1)]
    exit_price = exit_bar[1]['c']
    return (entry_price - exit_price) / entry_price * 100  # SHORT

def compute_pnl_eod(entry_price, bars):
    if entry_price <= 0 or not bars:
        return None
    exit_price = bars[-1][1]['c']
    return (entry_price - exit_price) / entry_price * 100  # SHORT

def metrics(pnl):
    pnl = np.array([x for x in pnl if x is not None and np.isfinite(x)])
    n = len(pnl)
    if n == 0:
        return {'n': 0}
    avg = np.mean(pnl)
    std = np.std(pnl, ddof=1) if n > 1 else 0
    se = std / np.sqrt(n) if n > 1 else 0
    return {
        'n': n, 'avg': avg, 'median': np.median(pnl),
        'trimmed': stats.trim_mean(pnl, 0.05) if n >= 20 else avg,
        'std': std, 'sharpe': avg / std if std > 0 else 0,
        'wr': np.mean(pnl > 0), 'se': se,
        'ci95_lo': avg - 1.96 * se, 'ci95_hi': avg + 1.96 * se,
        'total': np.sum(pnl), 'max_loss': np.min(pnl), 'max_gain': np.max(pnl),
        'pf': pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()) if (pnl < 0).any() else np.inf
    }

# ══════════════════════════════════════════════════════════════════════════════
# BUILD TRADE-LEVEL DATA
# ══════════════════════════════════════════════════════════════════════════════

old_trades = []
new_trades = []
excluded = {'not_short': 0, 'bad_catalyst': 0, 'no_bars': 0, 'no_orb_confirm': 0}

for _, row in df.iterrows():
    tk, dt = row['ticker'], str(row['eval_date'])
    sign = row['sign']
    bars = get_sorted_bars(tk, dt)

    # ── Old baseline: all trades, 09:30 entry ──
    if bars and len(bars) >= 60:
        ep_old = bars[0][1]['o']
        if ep_old > 0:
            sl_price = ep_old * (1 - 0.025) if sign == 1 else ep_old * (1 + 0.025)
            stopped = False
            stop_pnl = None
            for i, (t, b) in enumerate(bars[:60]):
                if (sign == 1 and b['l'] <= sl_price) or (sign == -1 and b['h'] >= sl_price):
                    stop_pnl = sign * (sl_price - ep_old) / ep_old * 100
                    stopped = True
                    break
            pnl_60m_sl = stop_pnl if stopped else sign * (bars[min(59, len(bars)-1)][1]['c'] - ep_old) / ep_old * 100
            pnl_60m_nosl = sign * (bars[min(59, len(bars)-1)][1]['c'] - ep_old) / ep_old * 100
            pnl_eod = sign * (bars[-1][1]['c'] - ep_old) / ep_old * 100

            old_trades.append({
                'ticker': tk, 'date': dt, 'direction': 'LONG' if sign == 1 else 'SHORT',
                'pnl_60m_sl': pnl_60m_sl, 'pnl_60m_nosl': pnl_60m_nosl,
                'pnl_eod': pnl_eod, 'stopped': stopped,
                'pnl_t1': row['pnl_t1_close'] if pd.notna(row['pnl_t1_close']) else None,
                'pnl_t2': row['pnl_t2_close'] if pd.notna(row['pnl_t2_close']) else None,
                'catalyst': row['catalyst_type'], 'gap_pct': row['gap_pct'],
            })

    # ── New baseline: SHORT only, filtered, local high entry ──
    if sign != -1:
        excluded['not_short'] += 1
        continue

    cat = str(row['catalyst_type']).lower()
    if any(x in cat for x in ['earnings', 'deal', 'macro']):
        excluded['bad_catalyst'] += 1
        continue

    if not bars or len(bars) < 20:
        excluded['no_bars'] += 1
        continue

    confirms, orb_high, orb_low = compute_orb_confirms_short(bars)
    if not confirms:
        excluded['no_orb_confirm'] += 1
        continue

    # Entry: first local high or fallback 09:45
    lh = find_first_local_high(bars, window=3, max_idx=15)
    if lh:
        ep_new = lh['entry_price']
        entry_type = 'local_high'
        delay = lh['delay']
        entry_time = lh['confirm_time']
    else:
        fb_idx = min(15, len(bars) - 1)
        ep_new = bars[fb_idx][1]['c']
        entry_type = 'fallback_0945'
        delay = 15
        entry_time = bars[fb_idx][0]

    if ep_new <= 0:
        continue

    # Compute PnL at multiple horizons from new entry price
    pnl_30m = compute_pnl_from_entry(ep_new, bars, 30) if len(bars) >= 30 else None
    pnl_60m = compute_pnl_from_entry(ep_new, bars, 60) if len(bars) >= 60 else None
    pnl_120m = compute_pnl_from_entry(ep_new, bars, 120) if len(bars) >= 120 else None
    pnl_eod_new = compute_pnl_eod(ep_new, bars)

    # T+1 and T+2: scale from CSV (these use scanner entry, adjust for new entry)
    # Simpler: use the CSV close prices directly with our entry
    ep_scanner = row['entry_price']
    ratio = ep_new / ep_scanner if ep_scanner > 0 else 1.0

    # For T+1/T+2 we need the actual close price, derive from CSV PnL
    # pnl_t1_close = sign * (close_t1 - ep_scanner) / ep_scanner * 100
    # For SHORT (sign=-1): pnl = -(close_t1 - ep) / ep * 100 = (ep - close_t1) / ep * 100
    # So close_t1 = ep * (1 - pnl/100)
    pnl_t1 = None
    pnl_t2 = None
    if pd.notna(row['pnl_t1_close']):
        close_t1 = ep_scanner * (1 - row['pnl_t1_close'] / 100)
        pnl_t1 = (ep_new - close_t1) / ep_new * 100
    if pd.notna(row['pnl_t2_close']):
        close_t2 = ep_scanner * (1 - row['pnl_t2_close'] / 100)
        pnl_t2 = (ep_new - close_t2) / ep_new * 100

    or_width = orb_high - orb_low
    atr_d = ep_new * row['atr_pct'] / 100 if row['atr_pct'] > 0 else or_width
    or_atr = or_width / atr_d if atr_d > 0 else 0
    orb_bucket = 'tight' if or_atr < 0.5 else ('normal' if or_atr < 1.0 else 'wide')

    gap = abs(row['gap_pct'])
    day_med = df[(df['eval_date']==dt) & (df['sign']==-1)]['abs_gap_pct'].median()
    weight = gap / day_med if day_med > 0 else 1.0

    new_trades.append({
        'ticker': tk, 'date': dt, 'direction': 'SHORT',
        'entry_type': entry_type, 'entry_time': entry_time, 'delay_mins': delay,
        'entry_price': ep_new, 'gap_pct': row['gap_pct'],
        'catalyst': row['catalyst_type'], 'orb_bucket': orb_bucket,
        'pnl_30m': pnl_30m, 'pnl_60m': pnl_60m, 'pnl_120m': pnl_120m,
        'pnl_eod': pnl_eod_new, 'pnl_t1': pnl_t1, 'pnl_t2': pnl_t2,
        'weight': weight, 'or_width_atr': or_atr,
    })

df_old = pd.DataFrame(old_trades)
df_new = pd.DataFrame(new_trades)

print(f"\n  Old baseline: {len(df_old)} trades ({df_old['stopped'].sum():.0f} stopped at 2.5% SL)")
print(f"  New V1: {len(df_new)} trades")
print(f"  Excluded: {json.dumps(excluded)}")

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-HORIZON COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

horizons = [
    ('60m (2.5% SL vs no SL)', 'pnl_60m_sl', 'pnl_60m'),
    ('60m (both no SL)',        'pnl_60m_nosl', 'pnl_60m'),
    ('EOD',                     'pnl_eod', 'pnl_eod'),
    ('T+1 close',              'pnl_t1', 'pnl_t1'),
    ('T+2 close',              'pnl_t2', 'pnl_t2'),
]

print(f"\n{'='*_W}")
print(f"  MULTI-HORIZON HEAD-TO-HEAD")
print(f"{'='*_W}")
print(f"\n  {'Horizon':<28} {'OLD avg':>10} {'OLD n':>6} {'NEW avg':>10} {'NEW n':>6} {'Delta':>10} {'NEW WR':>8} {'NEW Sh':>8} {'p-val':>8}")
print(f"  {'─'*94}")

horizon_results = {}
for label, old_col, new_col in horizons:
    old_pnl = df_old[old_col].dropna().values
    new_pnl = df_new[new_col].dropna().values

    m_o = metrics(old_pnl)
    m_n = metrics(new_pnl)

    if m_o['n'] > 0 and m_n['n'] > 0:
        _, p_val = stats.mannwhitneyu(new_pnl, old_pnl, alternative='greater') if m_n['n'] >= 5 else (0, 1.0)
        delta = m_n['avg'] - m_o['avg']
        p_str = f"{p_val:.4f}" if p_val < 1 else "—"
        print(f"  {label:<28} {m_o['avg']:>+10.3f} {m_o['n']:>6} {m_n['avg']:>+10.3f} {m_n['n']:>6} {delta:>+10.3f} {m_n['wr']:>7.0%} {m_n['sharpe']:>8.3f} {p_str:>8}")
    else:
        print(f"  {label:<28} {'—':>10} {m_o['n']:>6} {'—':>10} {m_n['n']:>6}")

    horizon_results[label] = {'old': m_o, 'new': m_n}

# ══════════════════════════════════════════════════════════════════════════════
# PER-DATE BREAKDOWN (60m, all dates)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*_W}")
print(f"  PER-DATE BREAKDOWN (60m horizon)")
print(f"{'─'*_W}")
print(f"\n  {'Date':<14} {'OLD n':>6} {'OLD avg':>10} {'OLD WR':>8} {'NEW n':>6} {'NEW avg':>10} {'NEW WR':>8} {'Delta':>10}")
print(f"  {'─'*78}")

for dt in sorted(df['eval_date'].unique()):
    g_o = df_old[df_old['date'] == dt]
    g_n = df_new[df_new['date'] == dt]
    o_avg = g_o['pnl_60m_sl'].mean() if len(g_o) > 0 else 0
    o_wr = (g_o['pnl_60m_sl'] > 0).mean() if len(g_o) > 0 else 0
    n_avg = g_n['pnl_60m'].mean() if len(g_n) > 0 else 0
    n_wr = (g_n['pnl_60m'] > 0).mean() if len(g_n) > 0 else 0
    delta = n_avg - o_avg if len(g_n) > 0 else 0
    print(f"  {dt:<14} {len(g_o):>6} {o_avg:>+10.3f} {o_wr:>7.0%} {len(g_n):>6} {n_avg:>+10.3f} {n_wr:>7.0%} {delta:>+10.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# DETAILED NEW V1 BREAKDOWNS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*_W}")
print(f"  NEW V1 DETAILED BREAKDOWNS (60m)")
print(f"{'─'*_W}")

print(f"\n  By Entry Type:")
for etype, g in df_new.groupby('entry_type'):
    m = metrics(g['pnl_60m'].dropna())
    if m['n'] > 0:
        print(f"    {etype:<18} n={m['n']:>3}  avg={m['avg']:+.2f}%  WR={m['wr']:.0%}  delay={g['delay_mins'].mean():.1f}min")

print(f"\n  By Catalyst:")
for cat, g in sorted(df_new.groupby('catalyst'), key=lambda x: -x[1]['pnl_60m'].dropna().mean() if len(x[1]['pnl_60m'].dropna()) > 0 else 0):
    m = metrics(g['pnl_60m'].dropna())
    if m['n'] > 0:
        flag = ' (LOW-N)' if m['n'] < 20 else ''
        print(f"    {cat:<25} n={m['n']:>3}  avg={m['avg']:+.2f}%  WR={m['wr']:.0%}{flag}")

print(f"\n  By ORB Width:")
for bucket in ['tight', 'normal', 'wide']:
    g = df_new[df_new['orb_bucket'] == bucket]
    m = metrics(g['pnl_60m'].dropna())
    if m['n'] > 0:
        flag = ' (LOW-N)' if m['n'] < 20 else ''
        print(f"    {bucket:<12} n={m['n']:>3}  avg={m['avg']:+.2f}%  WR={m['wr']:.0%}  Sharpe={m['sharpe']:.3f}{flag}")

print(f"\n  By Date (all available horizons):")
for dt in sorted(df_new['date'].unique()):
    g = df_new[df_new['date'] == dt]
    print(f"\n    {dt}  (n={len(g)}):")
    for col, label in [('pnl_60m','60m'), ('pnl_eod','EOD'), ('pnl_t1','T+1'), ('pnl_t2','T+2')]:
        vals = g[col].dropna()
        if len(vals) > 0:
            print(f"      {label:<6} avg={vals.mean():+.2f}%  WR={(vals>0).mean():.0%}  n={len(vals)}")

# ══════════════════════════════════════════════════════════════════════════════
# OLD BASELINE DIRECTION BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*_W}")
print(f"  OLD BASELINE DIRECTION BREAKDOWN (60m, 2.5% SL)")
print(f"{'─'*_W}")
for d, g in df_old.groupby('direction'):
    m = metrics(g['pnl_60m_sl'].dropna())
    print(f"  {d:<8} n={m['n']:>3}  avg={m['avg']:+.3f}%  WR={m['wr']:.0%}  stopped={g['stopped'].sum():.0f} ({g['stopped'].mean():.0%})")

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS (60m — all dates)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*_W}")
print(f"  STATISTICAL SIGNIFICANCE (60m, all dates)")
print(f"{'─'*_W}")

old_60 = df_old['pnl_60m_sl'].dropna().values
new_60 = df_new['pnl_60m'].dropna().values

t_s, p_w = stats.ttest_ind(new_60, old_60, equal_var=False)
u_s, p_m = stats.mannwhitneyu(new_60, old_60, alternative='greater')
print(f"  Welch's t-test:   t = {t_s:.3f}, p = {p_w:.4f}  ({'SIGNIFICANT' if p_w < 0.05 else 'not significant'})")
print(f"  Mann-Whitney U:   U = {u_s:.0f}, p = {p_m:.4f}  ({'SIGNIFICANT' if p_m < 0.05 else 'not significant'})")
print(f"  n_old={len(old_60)}, n_new={len(new_60)}, dates=4")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Old Baseline vs New V1 — All Dates, Multiple Horizons', fontsize=14, fontweight='bold', y=0.98)

# 1. PnL distributions at 60m
ax = axes[0, 0]
bins = np.linspace(-10, 20, 35)
ax.hist(df_old['pnl_60m_sl'].dropna(), bins=bins, alpha=0.55, color='#e74c3c', label=f"Old (n={len(df_old)}, 2.5%SL)", density=True)
ax.hist(df_new['pnl_60m'].dropna(), bins=bins, alpha=0.55, color='#27ae60', label=f"New V1 (n={len(df_new)})", density=True)
ax.axvline(df_old['pnl_60m_sl'].mean(), color='#e74c3c', ls='--', lw=2)
ax.axvline(df_new['pnl_60m'].dropna().mean(), color='#27ae60', ls='--', lw=2)
ax.set_xlabel('PnL (%)')
ax.set_title('PnL Distribution (60m)')
ax.legend(fontsize=8)

# 2. Multi-horizon comparison bars
ax = axes[0, 1]
h_labels = ['60m', 'EOD', 'T+1', 'T+2']
h_old_cols = ['pnl_60m_sl', 'pnl_eod', 'pnl_t1', 'pnl_t2']
h_new_cols = ['pnl_60m', 'pnl_eod', 'pnl_t1', 'pnl_t2']
old_avgs = [df_old[c].dropna().mean() if df_old[c].dropna().shape[0] > 0 else 0 for c in h_old_cols]
new_avgs = [df_new[c].dropna().mean() if df_new[c].dropna().shape[0] > 0 else 0 for c in h_new_cols]
old_ns = [df_old[c].dropna().shape[0] for c in h_old_cols]
new_ns = [df_new[c].dropna().shape[0] for c in h_new_cols]
x = np.arange(len(h_labels))
b1 = ax.bar(x - 0.2, old_avgs, 0.35, color='#e74c3c', alpha=0.8, label='Old')
b2 = ax.bar(x + 0.2, new_avgs, 0.35, color='#27ae60', alpha=0.8, label='New V1')
for i in range(len(h_labels)):
    ax.text(x[i]-0.2, max(old_avgs[i]+0.1, 0.1), f'n={old_ns[i]}', ha='center', fontsize=7)
    ax.text(x[i]+0.2, max(new_avgs[i]+0.1, 0.1), f'n={new_ns[i]}', ha='center', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(h_labels)
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylabel('Avg PnL (%)')
ax.set_title('Avg PnL by Horizon')
ax.legend(fontsize=9)

# 3. Win rate by date (60m)
ax = axes[0, 2]
dates = sorted(df['eval_date'].unique())
x = np.arange(len(dates))
old_wr = [(df_old[df_old['date']==d]['pnl_60m_sl']>0).mean() if len(df_old[df_old['date']==d])>0 else 0 for d in dates]
new_wr_vals = []
new_n_vals = []
for d in dates:
    g = df_new[df_new['date']==d]['pnl_60m'].dropna()
    new_wr_vals.append((g>0).mean() if len(g)>0 else 0)
    new_n_vals.append(len(g))
b1 = ax.bar(x - 0.2, old_wr, 0.35, color='#e74c3c', alpha=0.8, label='Old')
b2 = ax.bar(x + 0.2, new_wr_vals, 0.35, color='#27ae60', alpha=0.8, label='New V1')
for i in range(len(dates)):
    ax.text(x[i]+0.2, new_wr_vals[i]+0.02, f'n={new_n_vals[i]}', ha='center', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([d[-5:] for d in dates])
ax.set_ylabel('Win Rate')
ax.set_title('Win Rate by Date (60m)')
ax.legend(fontsize=9)

# 4. Cumulative PnL (60m)
ax = axes[1, 0]
old_cum = df_old.sort_values('date')['pnl_60m_sl'].cumsum().values
new_cum = df_new.sort_values('date')['pnl_60m'].dropna().cumsum().values
ax.plot(range(len(old_cum)), old_cum, color='#e74c3c', lw=2, label=f'Old (total: {old_cum[-1]:+.1f}%)')
ax.plot(range(len(new_cum)), new_cum, color='#27ae60', lw=2, label=f'New V1 (total: {new_cum[-1]:+.1f}%)')
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('Trade #')
ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('Cumulative PnL (60m)')
ax.legend(fontsize=9)

# 5. New V1: PnL by catalyst
ax = axes[1, 1]
cat_avgs = df_new.groupby('catalyst')['pnl_60m'].mean().dropna().sort_values()
colors = ['#e74c3c' if v < 0 else '#27ae60' for v in cat_avgs]
cat_avgs.plot(kind='barh', ax=ax, color=colors, alpha=0.8)
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel('Avg PnL 60m (%)')
ax.set_title('New V1: PnL by Catalyst')

# 6. Entry type + ORB bucket
ax = axes[1, 2]
groups = []
for etype in ['local_high', 'fallback_0945']:
    g = df_new[df_new['entry_type'] == etype]['pnl_60m'].dropna()
    if len(g) > 0:
        groups.append((etype.replace('_','\n'), g.mean(), len(g)))
for bucket in ['tight', 'normal', 'wide']:
    g = df_new[df_new['orb_bucket'] == bucket]['pnl_60m'].dropna()
    if len(g) > 0:
        groups.append((f'ORB\n{bucket}', g.mean(), len(g)))
bx = np.arange(len(groups))
colors = ['#27ae60', '#95a5a6', '#3498db', '#2ecc71', '#e67e22'][:len(groups)]
bars = ax.bar(bx, [g[1] for g in groups], color=colors, alpha=0.8)
for i, (_, avg, n) in enumerate(groups):
    flag = '*' if n < 20 else ''
    ax.text(i, avg + 0.05, f'n={n}{flag}', ha='center', fontsize=7)
ax.set_xticks(bx)
ax.set_xticklabels([g[0] for g in groups], fontsize=8)
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylabel('Avg PnL (%)')
ax.set_title('New V1: By Entry Type & ORB Width')

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = os.path.join(BASE_DIR, 'baseline_comparison_v2.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Chart saved: {chart_path}")

# ── Save results ─────────────────────────────────────────────────────────────
df_old.to_csv(os.path.join(RESULTS_DIR, 'baseline_old_frozen_v2.csv'), index=False)
df_new.to_csv(os.path.join(RESULTS_DIR, 'baseline_v1_frozen_v2.csv'), index=False)

save_data = {
    'frozen_date': '2026-09-12',
    'note': 'T+2 unavailable for Sep 10-11 (weekend). 60m and EOD cover all 4 dates.',
    'excluded': excluded,
    'old_baseline': {'n': len(df_old), 'rules': 'LLM-flagged, both dirs, 09:30, 60m, 2.5% SL'},
    'new_baseline': {'n': len(df_new), 'rules': 'SHORT, excl earnings/deal/macro, ORB confirms, local high, no SL'},
}
for label, res in horizon_results.items():
    key = label.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')
    for side in ['old', 'new']:
        save_data[f'{key}_{side}'] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                       for k, v in res[side].items()}

with open(os.path.join(RESULTS_DIR, 'baseline_comparison_v2_results.json'), 'w') as f:
    json.dump(save_data, f, indent=2, default=str)

print(f"\n  Saved frozen baselines + results JSON")
print(f"\n{'='*_W}")
print(f"  DONE")
print(f"{'='*_W}")
