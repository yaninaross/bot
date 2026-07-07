"""
compare_500_vs_spike.py — Head-to-head: Exp 500 Morning Swing vs 180-185 Spike
================================================================================
Compares WF optimization results from BQ for:
  - Exp 500 (morning swing, 3.5% target, 75-min window)
  - Exp 180-185 (spike, 10% target, 6.5-hour window) — production model family

Three analyses:
  A. Per-ticker PnL comparison on overlapping tickers
  B. Capital efficiency (PnL / hours held)
  C. Seed stability comparison

BQ-only, no backtesting. Re-run as optimizer completes more tickers.
"""

from google.cloud import bigquery
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

bq = bigquery.Client()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
W = 130
BQ_TABLE = 'ys-time-series.scoring_logs.wf_optimization_results'

SWING_MODEL_VERSION = 'robust_wf_v12_swing'
SPIKE_MODEL_VERSION = 'robust_wf_v12'

SWING_HOLD_HOURS = 1.25   # 75 min (6:15 → 7:30)
SPIKE_HOLD_HOURS = 6.75   # 6h45m (6:15 → 1:00 PM)

TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPY', 'IWM'}

SPIKE_FAMILY = '180'  # 180-185 family only (production model, fully trained)

def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if 180 <= num <= 185: return '180'
    return 'other'


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD SWING (500) RESULTS — ensemble + individual seeds
# ══════════════════════════════════════════════════════════════════════════════
print(f"{'═'*W}")
print(f"  EXP 500 MORNING SWING vs EXP 180-185 SPIKE — HEAD-TO-HEAD")
print(f"{'═'*W}")

q_swing_ens = f'''
SELECT ticker, tpsl_mode, experiment,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       dip_pct, max_wait_minutes, grace_minutes,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       survived_a, survived_b, n_seeds_avg
FROM `{BQ_TABLE}`
WHERE model_version = '{SWING_MODEL_VERSION}'
  AND is_ensemble = TRUE
'''
swing_ens = bq.query(q_swing_ens).to_dataframe()
swing_ens['ticker'] = swing_ens['ticker'].replace(TICKER_RENAME)
swing_ens = swing_ens[~swing_ens['ticker'].isin(EXCLUDE_TICKERS)]

q_swing_seeds = f'''
SELECT ticker, tpsl_mode, seed,
       val_a_avg_pnl, val_b_avg_pnl, survived_a
FROM `{BQ_TABLE}`
WHERE model_version = '{SWING_MODEL_VERSION}'
  AND is_ensemble = FALSE
  AND survived_a = TRUE
'''
swing_seeds = bq.query(q_swing_seeds).to_dataframe()
swing_seeds['ticker'] = swing_seeds['ticker'].replace(TICKER_RENAME)
swing_seeds = swing_seeds[~swing_seeds['ticker'].isin(EXCLUDE_TICKERS)]

print(f"\n  Swing (500): {len(swing_ens)} ensemble rows, "
      f"{swing_ens['ticker'].nunique()} tickers, "
      f"{len(swing_seeds)} seed rows")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD SPIKE RESULTS — ensemble across all families
# ══════════════════════════════════════════════════════════════════════════════
q_spike_ens = f'''
SELECT ticker, tpsl_mode, experiment,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       dip_pct, max_wait_minutes, grace_minutes,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       survived_a, survived_b, n_seeds_avg
FROM `{BQ_TABLE}`
WHERE model_version = '{SPIKE_MODEL_VERSION}'
  AND is_ensemble = TRUE
  AND CAST(SPLIT(experiment, '_')[OFFSET(0)] AS INT64) BETWEEN 180 AND 185
'''
spike_ens = bq.query(q_spike_ens).to_dataframe()
spike_ens['ticker'] = spike_ens['ticker'].replace(TICKER_RENAME)
spike_ens = spike_ens[~spike_ens['ticker'].isin(EXCLUDE_TICKERS)]
spike_ens['exp_fam'] = spike_ens['experiment'].apply(assign_family)
spike_ens = spike_ens[spike_ens['exp_fam'] != 'other']

q_spike_seeds = f'''
SELECT ticker, tpsl_mode, experiment, seed,
       val_a_avg_pnl, val_b_avg_pnl, survived_a
FROM `{BQ_TABLE}`
WHERE model_version = '{SPIKE_MODEL_VERSION}'
  AND is_ensemble = FALSE
  AND survived_a = TRUE
  AND CAST(SPLIT(experiment, '_')[OFFSET(0)] AS INT64) BETWEEN 180 AND 185
'''
spike_seeds = bq.query(q_spike_seeds).to_dataframe()
spike_seeds['ticker'] = spike_seeds['ticker'].replace(TICKER_RENAME)
spike_seeds = spike_seeds[~spike_seeds['ticker'].isin(EXCLUDE_TICKERS)]
spike_seeds['exp_fam'] = spike_seeds['experiment'].apply(assign_family)

print(f"  Spike (180-185): {len(spike_ens)} ensemble rows, "
      f"{spike_ens['ticker'].nunique()} tickers, "
      f"{len(spike_seeds)} seed rows")
print(f"  Spike families: {sorted(spike_ens['exp_fam'].unique())}")

overlap_tickers = sorted(set(swing_ens['ticker'].unique()) &
                         set(spike_ens['ticker'].unique()))
print(f"  Overlapping tickers: {len(overlap_tickers)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BEST CONFIG PER TICKER — each model's best ensemble
# ══════════════════════════════════════════════════════════════════════════════
def best_ensemble_per_ticker(df_ens, label):
    survived = df_ens[df_ens['survived_a'] == True].copy()
    if survived.empty:
        return pd.DataFrame()
    best = survived.loc[survived.groupby('ticker')['val_a_avg_pnl'].idxmax()]
    return best.set_index('ticker')

swing_best = best_ensemble_per_ticker(swing_ens, 'swing')
spike_best = best_ensemble_per_ticker(spike_ens, 'spike')

# For spike, also get best per-family for family agreement analysis
spike_best_by_fam = {}
for fam in spike_ens['exp_fam'].unique():
    fam_ens = spike_ens[spike_ens['exp_fam'] == fam]
    spike_best_by_fam[fam] = best_ensemble_per_ticker(fam_ens, f'spike_{fam}')


# ══════════════════════════════════════════════════════════════════════════════
# A. PER-TICKER HEAD-TO-HEAD
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  A. PER-TICKER HEAD-TO-HEAD — Best Ensemble (limit_entry preferred)")
print(f"  Swing TEST: May 16 - Jul 1  |  Spike TEST: May 16 - Jun 17")
print(f"{'═'*W}")

comparison = []
for tk in overlap_tickers:
    sw = swing_best.loc[tk] if tk in swing_best.index else None
    sp = spike_best.loc[tk] if tk in spike_best.index else None

    row = {'ticker': tk}
    if sw is not None:
        row['sw_mode'] = sw['tpsl_mode']
        row['sw_va_pnl'] = sw['val_a_avg_pnl']
        row['sw_va_cnt'] = int(sw['val_a_count'])
        row['sw_va_wr'] = sw['val_a_win_rate']
        row['sw_vb_pnl'] = sw['val_b_avg_pnl']
        row['sw_vb_cnt'] = int(sw['val_b_count'])
        row['sw_vb_wr'] = sw['val_b_win_rate']
        row['sw_prob'] = sw['prob_thresh']
        tp_str = f"{sw['tp_pct']:.0%}" if sw['tp_pct'] > 0 else f"x{sw.get('tp_mult',0):.1f}"
        sl_str = f"{sw['sl_pct']:.0%}" if sw['sl_pct'] > 0 else f"x{sw.get('sl_mult',0):.1f}"
        row['sw_tpsl'] = f"{tp_str}/{sl_str}"
    else:
        row['sw_mode'] = '—'
        row['sw_va_pnl'] = None
        row['sw_va_cnt'] = 0
        row['sw_vb_pnl'] = None
        row['sw_vb_cnt'] = 0

    if sp is not None:
        row['sp_mode'] = sp['tpsl_mode']
        row['sp_va_pnl'] = sp['val_a_avg_pnl']
        row['sp_va_cnt'] = int(sp['val_a_count'])
        row['sp_va_wr'] = sp['val_a_win_rate']
        row['sp_vb_pnl'] = sp['val_b_avg_pnl']
        row['sp_vb_cnt'] = int(sp['val_b_count'])
        row['sp_vb_wr'] = sp['val_b_win_rate']
        row['sp_prob'] = sp['prob_thresh']
        row['sp_tpsl'] = f"{sp['tp_pct']:.0%}/{sp['sl_pct']:.0%}"
    else:
        row['sp_mode'] = '—'
        row['sp_va_pnl'] = None
        row['sp_va_cnt'] = 0
        row['sp_vb_pnl'] = None
        row['sp_vb_cnt'] = 0

    comparison.append(row)

cdf = pd.DataFrame(comparison)

print(f"\n  {'Ticker':<7} │ {'── SWING 500 (75 min) ──────────────────':} │ {'── SPIKE (6.75 hr) ─────────────────────':} │ {'Winner':}")
print(f"  {'':7} │ {'Mode':<14} {'VA_PnL':>7} {'VA_N':>4} {'VB_PnL':>7} {'VB_N':>4} │ {'Mode':<14} {'VA_PnL':>7} {'VA_N':>4} {'VB_PnL':>7} {'VB_N':>4} │")
print(f"  {'─'*7} │ {'─'*14} {'─'*7} {'─'*4} {'─'*7} {'─'*4} │ {'─'*14} {'─'*7} {'─'*4} {'─'*7} {'─'*4} │ {'─'*8}")

swing_wins = 0
spike_wins = 0
ties = 0

for _, r in cdf.iterrows():
    sw_va = f"{r['sw_va_pnl']:+.2%}" if r['sw_va_pnl'] is not None else '    —'
    sw_vb = f"{r['sw_vb_pnl']:+.2%}" if r['sw_vb_pnl'] is not None else '    —'
    sp_va = f"{r['sp_va_pnl']:+.2%}" if r['sp_va_pnl'] is not None else '    —'
    sp_vb = f"{r['sp_vb_pnl']:+.2%}" if r['sp_vb_pnl'] is not None else '    —'

    # Winner by TEST PnL (val_b), prefer one with data
    sw_vb_val = r['sw_vb_pnl'] if r['sw_vb_pnl'] is not None and r['sw_vb_cnt'] > 0 else -999
    sp_vb_val = r['sp_vb_pnl'] if r['sp_vb_pnl'] is not None and r['sp_vb_cnt'] > 0 else -999
    if sw_vb_val == -999 and sp_vb_val == -999:
        winner = '—'
        ties += 1
    elif sw_vb_val > sp_vb_val:
        winner = 'SWING'
        swing_wins += 1
    elif sp_vb_val > sw_vb_val:
        winner = 'SPIKE'
        spike_wins += 1
    else:
        winner = 'TIE'
        ties += 1

    print(f"  {r['ticker']:<7} │ {r['sw_mode']:<14} {sw_va:>7} {r['sw_va_cnt']:>4} {sw_vb:>7} {r['sw_vb_cnt']:>4}"
          f" │ {r['sp_mode']:<14} {sp_va:>7} {r['sp_va_cnt']:>4} {sp_vb:>7} {r['sp_vb_cnt']:>4}"
          f" │ {winner}")

print(f"\n  Score: SWING {swing_wins} — SPIKE {spike_wins} — TIE/N/A {ties}")


# ══════════════════════════════════════════════════════════════════════════════
# B. CAPITAL EFFICIENCY — PnL per hour of capital deployed
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  B. CAPITAL EFFICIENCY — PnL per hour of capital held")
print(f"  Swing holds ~{SWING_HOLD_HOURS:.2f}h, Spike holds ~{SPIKE_HOLD_HOURS:.2f}h")
print(f"  Capital-adj = avg_pnl / hold_hours → annualized by ×252×6.5h")
print(f"{'═'*W}")

print(f"\n  {'Ticker':<7} │ {'─── SWING ──────────────────':} │ {'─── SPIKE ──────────────────':} │ {'Adj Winner':}")
print(f"  {'':7} │ {'VA_PnL':>7} {'PnL/hr':>8} {'Ann%':>7} │ {'VA_PnL':>7} {'PnL/hr':>8} {'Ann%':>7} │")
print(f"  {'─'*7} │ {'─'*7} {'─'*8} {'─'*7} │ {'─'*7} {'─'*8} {'─'*7} │ {'─'*10}")

sw_adj_wins = 0
sp_adj_wins = 0

for _, r in cdf.iterrows():
    sw_pnl_hr = (r['sw_va_pnl'] / SWING_HOLD_HOURS) if r['sw_va_pnl'] is not None else None
    sp_pnl_hr = (r['sp_va_pnl'] / SPIKE_HOLD_HOURS) if r['sp_va_pnl'] is not None else None

    sw_ann = sw_pnl_hr * 252 * 6.5 if sw_pnl_hr is not None else None
    sp_ann = sp_pnl_hr * 252 * 6.5 if sp_pnl_hr is not None else None

    sw_va = f"{r['sw_va_pnl']:+.2%}" if r['sw_va_pnl'] is not None else '    —'
    sw_hr = f"{sw_pnl_hr:+.3%}" if sw_pnl_hr is not None else '     —'
    sw_an = f"{sw_ann:+.0%}" if sw_ann is not None else '    —'
    sp_va = f"{r['sp_va_pnl']:+.2%}" if r['sp_va_pnl'] is not None else '    —'
    sp_hr = f"{sp_pnl_hr:+.3%}" if sp_pnl_hr is not None else '     —'
    sp_an = f"{sp_ann:+.0%}" if sp_ann is not None else '    —'

    sw_val = sw_pnl_hr if sw_pnl_hr is not None else -999
    sp_val = sp_pnl_hr if sp_pnl_hr is not None else -999
    if sw_val > sp_val and sw_val > 0:
        adj_winner = 'SWING'
        sw_adj_wins += 1
    elif sp_val > sw_val and sp_val > 0:
        adj_winner = 'SPIKE'
        sp_adj_wins += 1
    else:
        adj_winner = '—'

    print(f"  {r['ticker']:<7} │ {sw_va:>7} {sw_hr:>8} {sw_an:>7}"
          f" │ {sp_va:>7} {sp_hr:>8} {sp_an:>7}"
          f" │ {adj_winner}")

print(f"\n  Capital-adjusted score: SWING {sw_adj_wins} — SPIKE {sp_adj_wins}")


# ══════════════════════════════════════════════════════════════════════════════
# C. SEED STABILITY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  C. SEED STABILITY — surviving seeds per ticker")
print(f"{'═'*W}")

print(f"\n  {'Ticker':<7} │ {'── SWING 500 ──────────':} │ {'── SPIKE (best family) ─────────────':}")
print(f"  {'':7} │ {'Seeds':>5} {'Modes':>5} {'VA>1%':>5} │ {'Seeds':>5} {'Modes':>5} {'VA>1%':>5} {'Family':>8}")
print(f"  {'─'*7} │ {'─'*5} {'─'*5} {'─'*5} │ {'─'*5} {'─'*5} {'─'*5} {'─'*8}")

for tk in overlap_tickers:
    # Swing seed stats
    sw_tk = swing_seeds[swing_seeds['ticker'] == tk]
    sw_n_seeds = sw_tk['seed'].nunique()
    sw_n_modes = sw_tk['tpsl_mode'].nunique()
    sw_n_above = len(sw_tk[sw_tk['val_a_avg_pnl'] >= 0.01])

    # Spike: find best family by number of surviving seeds
    best_fam = None
    best_sp_seeds = 0
    best_sp_modes = 0
    best_sp_above = 0
    for fam in spike_seeds['exp_fam'].unique():
        sp_fam = spike_seeds[(spike_seeds['ticker'] == tk) & (spike_seeds['exp_fam'] == fam)]
        n = sp_fam['seed'].nunique()
        if n > best_sp_seeds:
            best_sp_seeds = n
            best_sp_modes = sp_fam['tpsl_mode'].nunique()
            best_sp_above = len(sp_fam[sp_fam['val_a_avg_pnl'] >= 0.01])
            best_fam = fam

    fam_str = best_fam or '—'
    print(f"  {tk:<7} │ {sw_n_seeds:>5} {sw_n_modes:>5} {sw_n_above:>5}"
          f" │ {best_sp_seeds:>5} {best_sp_modes:>5} {best_sp_above:>5} {fam_str:>8}")


# ══════════════════════════════════════════════════════════════════════════════
# D. AGGREGATE PORTFOLIO COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  D. AGGREGATE PORTFOLIO — total TEST PnL across qualifying tickers")
print(f"{'═'*W}")

# Swing portfolio: tickers with survived_a ensemble
sw_portfolio = swing_best[swing_best.index.isin(overlap_tickers)].copy()
sw_qual = sw_portfolio[sw_portfolio['val_a_avg_pnl'] > 0]

# Spike portfolio: tickers with survived_a ensemble
sp_portfolio = spike_best[spike_best.index.isin(overlap_tickers)].copy()
sp_qual = sp_portfolio[sp_portfolio['val_a_avg_pnl'] > 0]

print(f"\n  ── SWING 500 Portfolio ──")
if not sw_qual.empty:
    sw_total_vb = (sw_qual['val_b_avg_pnl'] * sw_qual['val_b_count']).sum()
    sw_total_trades = sw_qual['val_b_count'].sum()
    sw_avg_vb = sw_total_vb / sw_total_trades if sw_total_trades > 0 else 0
    sw_va_avg = sw_qual['val_a_avg_pnl'].mean()
    print(f"  Qualifying tickers: {len(sw_qual)}")
    print(f"  VAL avg PnL (mean across tickers): {sw_va_avg:+.2%}")
    print(f"  TEST total trades: {int(sw_total_trades)}")
    print(f"  TEST avg PnL (trade-weighted): {sw_avg_vb:+.2%}")
    print(f"  TEST total PnL: {sw_total_vb:+.2%}")
    print(f"  Capital-adj TEST PnL/hr: {sw_avg_vb/SWING_HOLD_HOURS:+.3%}")
else:
    print(f"  No qualifying tickers")

print(f"\n  ── SPIKE Portfolio ──")
if not sp_qual.empty:
    sp_total_vb = (sp_qual['val_b_avg_pnl'] * sp_qual['val_b_count']).sum()
    sp_total_trades = sp_qual['val_b_count'].sum()
    sp_avg_vb = sp_total_vb / sp_total_trades if sp_total_trades > 0 else 0
    sp_va_avg = sp_qual['val_a_avg_pnl'].mean()
    print(f"  Qualifying tickers: {len(sp_qual)}")
    print(f"  VAL avg PnL (mean across tickers): {sp_va_avg:+.2%}")
    print(f"  TEST total trades: {int(sp_total_trades)}")
    print(f"  TEST avg PnL (trade-weighted): {sp_avg_vb:+.2%}")
    print(f"  TEST total PnL: {sp_total_vb:+.2%}")
    print(f"  Capital-adj TEST PnL/hr: {sp_avg_vb/SPIKE_HOLD_HOURS:+.3%}")
else:
    print(f"  No qualifying tickers")


# ══════════════════════════════════════════════════════════════════════════════
# E. COMPLEMENTARITY — can they run together?
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  E. COMPLEMENTARITY — trading windows and combined potential")
print(f"{'═'*W}")

print(f"""
  Swing 500:  Entry 6:15 AM → Exit 7:30 AM  (75 min)
  Spike:      Entry 6:15 AM → Exit 1:00 PM  (6h 45m)

  These are NOT mutually exclusive:
  - Swing exits by 7:30 AM, freeing capital
  - Spike can trade from 6:15 AM onward
  - On days where BOTH fire: swing runs first (7:30 exit),
    then spike could run in the remaining 5.5 hours

  Combined portfolio approach:
  - Morning session (6:15-7:30): swing trades only
  - Rest of day (7:30-1:00 PM): spike trades only
  - Capital is recycled — no conflict
""")

# Combined: sum of both portfolios' TEST PnL
if not sw_qual.empty and not sp_qual.empty:
    combined_total = sw_total_vb + sp_total_vb
    combined_trades = sw_total_trades + sp_total_trades
    combined_avg = combined_total / combined_trades if combined_trades > 0 else 0

    print(f"  ── Combined Portfolio (if both deployed) ──")
    print(f"  Total TEST trades: {int(combined_trades)} "
          f"(swing: {int(sw_total_trades)}, spike: {int(sp_total_trades)})")
    print(f"  Total TEST PnL: {combined_total:+.2%} "
          f"(swing: {sw_total_vb:+.2%}, spike: {sp_total_vb:+.2%})")
    print(f"  Avg PnL per trade: {combined_avg:+.2%}")


# ══════════════════════════════════════════════════════════════════════════════
# F. PER-FOLD BREAKDOWN — 180-185 TEST PnL per fold vs Swing
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  F. PER-FOLD SPIKE TEST PnL — 180-185 folds on overlapping tickers")
print(f"  (Shows fold-level consistency for the production spike model)")
print(f"{'═'*W}")

fold_ids = ['180', '181', '182', '183', '184', '185']
fold_best = {}
for fid in fold_ids:
    fid_ens = spike_ens[spike_ens['experiment'].apply(lambda e: e.split('_')[0] == fid)]
    fid_surv = fid_ens[fid_ens['survived_a'] == True]
    if not fid_surv.empty:
        fold_best[fid] = fid_surv.loc[fid_surv.groupby('ticker')['val_a_avg_pnl'].idxmax()].set_index('ticker')

active_folds = sorted(fold_best.keys())
print(f"\n  {'Ticker':<7} │ {'Swing':>7} │", end='')
for fid in active_folds:
    print(f" {fid:>6}", end='')
print(f" │ {'#Fold':>5} {'#Pos':>4} {'Cons':>4}")
print(f"  {'─'*7} │ {'─'*7} │{'─'*7*len(active_folds)} │ {'─'*5} {'─'*4} {'─'*4}")

for tk in overlap_tickers:
    sw_vb = swing_best.loc[tk, 'val_b_avg_pnl'] if tk in swing_best.index and swing_best.loc[tk, 'val_b_count'] > 0 else None
    sw_str = f"{sw_vb:+.1%}" if sw_vb is not None else '    —'
    print(f"  {tk:<7} │ {sw_str:>7} │", end='')
    n_folds = 0
    n_pos = 0
    for fid in active_folds:
        fb = fold_best.get(fid)
        if fb is not None and tk in fb.index:
            vb = fb.loc[tk, 'val_b_avg_pnl']
            n_folds += 1
            if vb > 0:
                n_pos += 1
            mark = f"{vb:+.1%}" if fb.loc[tk, 'val_b_count'] > 0 else '  N/A'
            print(f" {mark:>6}", end='')
        else:
            print(f" {'·':>6}", end='')
    consistency = '✓' if n_pos > n_folds / 2 and n_folds >= 2 else '✗'
    print(f" │ {n_folds:>5} {n_pos:>4} {consistency:>4}")


# ══════════════════════════════════════════════════════════════════════════════
# G. SWING-ONLY TICKERS — tickers where swing works but spike doesn't
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  G. UNIQUE VALUE — tickers where one model works but not the other")
print(f"{'═'*W}")

swing_only_tickers = set(swing_ens['ticker'].unique()) - set(spike_ens['ticker'].unique())
spike_only_tickers = set(spike_ens['ticker'].unique()) - set(swing_ens['ticker'].unique())

swing_qual_tickers = set(swing_best[swing_best['val_a_avg_pnl'] > 0.005].index)
spike_qual_tickers = set(spike_best[spike_best['val_a_avg_pnl'] > 0.005].index)

swing_unique_edge = swing_qual_tickers - spike_qual_tickers
spike_unique_edge = spike_qual_tickers - swing_qual_tickers
both_edge = swing_qual_tickers & spike_qual_tickers

print(f"\n  Swing has edge, spike doesn't: {sorted(swing_unique_edge) if swing_unique_edge else 'none'}")
print(f"  Spike has edge, swing doesn't: {sorted(spike_unique_edge) if spike_unique_edge else '(too many to list)'}")
print(f"  Both have edge: {sorted(both_edge) if both_edge else 'none'}")
print(f"\n  Swing-only tickers (not scored by spike): {sorted(swing_only_tickers) if swing_only_tickers else 'none'}")
print(f"  Spike-only tickers (not scored by swing): {len(spike_only_tickers)} tickers")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  SUMMARY")
print(f"{'═'*W}")

print(f"""
  Raw PnL winner:          {'SWING' if swing_wins > spike_wins else 'SPIKE' if spike_wins > swing_wins else 'TIE'} ({swing_wins} vs {spike_wins} on {len(overlap_tickers)} tickers)
  Capital-adj winner:      {'SWING' if sw_adj_wins > sp_adj_wins else 'SPIKE' if sp_adj_wins > sw_adj_wins else 'TIE'} ({sw_adj_wins} vs {sp_adj_wins})
  Swing qualifying:        {len(swing_qual_tickers)} tickers with VA > 0.5%
  Spike qualifying:        {len(spike_qual_tickers)} tickers with VA > 0.5%
  Both have edge:          {len(both_edge)} tickers
  Trading windows:         NON-OVERLAPPING (can coexist)

  NOTE: Swing optimizer is still running ({swing_ens['ticker'].nunique()} of ~69 tickers done).
  Re-run this script as more tickers complete for updated comparison.
""")
print(f"{'═'*W}")
