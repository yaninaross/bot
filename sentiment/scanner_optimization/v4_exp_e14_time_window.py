"""
V4 Scanner Optimization: Experiment E14 — Time-Based Commitment Window
======================================================================
Instead of candle-counting for commitment, enter at 09:30 and evaluate at
fixed checkpoint times whether the trade is on the right side of the ORB.

- SHORT: hold if price < or_low (below range), exit if price > or_high
- LONG:  hold if price > or_high (above range), exit if price < or_low
- Ambiguous: price inside [or_low, or_high]

Checkpoints: 09:40, 09:45, 09:50, 09:55, 10:00, 10:15, 10:30

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
from collections import defaultdict

warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
BASE = Path('/home/admin_yaninaso_altostrat_com/bot-RAG/sentiment/scanner_optimization')
RESULTS = BASE / 'results'
CSV_PATH = RESULTS / 'v2_features_full.csv'
MINUTE_PATH = RESULTS / 'v3_minute_bars.json'
DAILY_PATH = RESULTS / 'v2_daily_data.json'

LOW_N_THRESHOLD = 20
CHECKPOINTS = ['09:40', '09:45', '09:50', '09:55', '10:00', '10:15', '10:30']

# ─── Load data ────────────────────────────────────────────────────────────────
print("=" * 80)
print("E14: TIME-BASED COMMITMENT WINDOW")
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
print()

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


def win_rate(arr):
    """Win rate: fraction > 0."""
    arr = np.array([x for x in arr if np.isfinite(x)])
    if len(arr) == 0:
        return np.nan
    return float(np.sum(arr > 0) / len(arr) * 100)


def low_n_flag(n):
    return " [LOW-N]" if n < LOW_N_THRESHOLD else ""


def get_bar_close(bars, time_str):
    """Get close price at a specific minute bar time."""
    if time_str in bars:
        return bars[time_str]['c']
    return None


def get_bar_open(bars, time_str):
    """Get open price at a specific minute bar time."""
    if time_str in bars:
        return bars[time_str]['o']
    return None


def get_t2_close(ticker, eval_date):
    """Get T+2 close price from daily data."""
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


def get_eod_close(ticker, eval_date):
    """Get end-of-day close from daily data."""
    if ticker not in daily_data:
        return None
    if eval_date not in daily_data[ticker]:
        return None
    return daily_data[ticker][eval_date]['close']


def compute_pnl(entry_price, exit_price, direction):
    """Compute PnL percentage, sign-adjusted so positive = profitable."""
    if direction == 'SHORT':
        return (entry_price - exit_price) / entry_price * 100
    else:  # LONG
        return (exit_price - entry_price) / entry_price * 100


# ─── Build trade-level data ──────────────────────────────────────────────────
print("Building trade-level minute bar data...")
print()

trades = []
skipped = 0

for _, row in df.iterrows():
    ticker = row['ticker']
    eval_date = row['eval_date']
    direction = row['direction']
    sign = row['sign']
    or_high = row['or_high']
    or_low = row['or_low']

    key = f"{ticker}__{eval_date}"
    if key not in ticker_bars:
        skipped += 1
        continue

    bars = ticker_bars[key]

    # Need 09:30 bar for entry
    if '09:30' not in bars:
        skipped += 1
        continue

    entry_price = bars['09:30']['o']

    # Check we have at least some checkpoint bars
    has_any_checkpoint = any(t in bars for t in CHECKPOINTS)
    if not has_any_checkpoint:
        skipped += 1
        continue

    # Get checkpoint prices
    checkpoint_prices = {}
    for t in CHECKPOINTS:
        cp = get_bar_close(bars, t)
        if cp is not None:
            checkpoint_prices[t] = cp

    # Get T+2 close for swing hold
    t2_close = get_t2_close(ticker, eval_date)
    t2_pnl = compute_pnl(entry_price, t2_close, direction) if t2_close else None

    # Get EOD close from minute bars (use 15:59 or last bar before 16:00)
    eod_price = None
    for t in ['15:59', '15:58', '15:57', '15:56', '15:55']:
        if t in bars:
            eod_price = bars[t]['c']
            break
    eod_pnl = compute_pnl(entry_price, eod_price, direction) if eod_price else None

    # Get 60m price (10:30 close)
    price_60m = get_bar_close(bars, '10:30')
    pnl_60m = compute_pnl(entry_price, price_60m, direction) if price_60m else None

    # Also use CSV PnL values as fallback
    csv_pnl_60m = row['pnl_60m_nosl'] if pd.notna(row['pnl_60m_nosl']) else None
    csv_pnl_eod = row['pnl_eod_nosl'] if pd.notna(row['pnl_eod_nosl']) else None
    csv_pnl_t2 = row['pnl_t2_close'] if pd.notna(row['pnl_t2_close']) else None

    trade = {
        'ticker': ticker,
        'eval_date': eval_date,
        'direction': direction,
        'sign': sign,
        'entry_price': entry_price,
        'or_high': or_high,
        'or_low': or_low,
        'or_mid': (or_high + or_low) / 2,
        'checkpoint_prices': checkpoint_prices,
        't2_pnl': t2_pnl if t2_pnl is not None else csv_pnl_t2,
        'eod_pnl': eod_pnl if eod_pnl is not None else csv_pnl_eod,
        'pnl_60m': pnl_60m if pnl_60m is not None else csv_pnl_60m,
        'csv_pnl_60m': csv_pnl_60m,
        'csv_pnl_eod': csv_pnl_eod,
        'csv_pnl_t2': csv_pnl_t2,
        'is_earnings': row['is_earnings'],
        'catalyst_type': row.get('catalyst_type', ''),
        'bars': bars,
    }
    trades.append(trade)

print(f"Trades with minute bar data: {len(trades)} (skipped {skipped})")
short_trades = [t for t in trades if t['direction'] == 'SHORT']
long_trades = [t for t in trades if t['direction'] == 'LONG']
print(f"  SHORT: {len(short_trades)}, LONG: {len(long_trades)}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SHORT-only analysis (T+2 hold outcome)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_checkpoint(trade_list, t_check, hold_pnl_key='t2_pnl',
                       direction_filter=None):
    """
    For a set of trades at a given checkpoint time:
    - Classify each trade as hold/exit/ambiguous based on price vs OR
    - Compute PnL for each bucket

    Returns dict with results.
    """
    hold_trades = []
    exit_trades = []
    ambig_trades = []

    for t in trade_list:
        if direction_filter and t['direction'] != direction_filter:
            continue

        if t_check not in t['checkpoint_prices']:
            continue  # Skip if no bar at this checkpoint

        cp = t['checkpoint_prices'][t_check]
        entry = t['entry_price']
        or_high = t['or_high']
        or_low = t['or_low']
        direction = t['direction']

        # Compute exit PnL (take the checkpoint price)
        exit_pnl = compute_pnl(entry, cp, direction)

        # Hold PnL (T+2 or 60m or EOD depending on key)
        h_pnl = t.get(hold_pnl_key)

        # Classify based on direction
        if direction == 'SHORT':
            if cp < or_low:
                # Price below opening range -> SHORT is working -> HOLD
                hold_trades.append({
                    'trade': t, 'exit_pnl': exit_pnl, 'hold_pnl': h_pnl,
                    'zone': 'hold'
                })
            elif cp > or_high:
                # Price above opening range -> SHORT is failing -> EXIT
                exit_trades.append({
                    'trade': t, 'exit_pnl': exit_pnl, 'hold_pnl': h_pnl,
                    'zone': 'exit'
                })
            else:
                # Inside range -> ambiguous
                ambig_trades.append({
                    'trade': t, 'exit_pnl': exit_pnl, 'hold_pnl': h_pnl,
                    'zone': 'ambiguous'
                })
        else:  # LONG
            if cp > or_high:
                hold_trades.append({
                    'trade': t, 'exit_pnl': exit_pnl, 'hold_pnl': h_pnl,
                    'zone': 'hold'
                })
            elif cp < or_low:
                exit_trades.append({
                    'trade': t, 'exit_pnl': exit_pnl, 'hold_pnl': h_pnl,
                    'zone': 'exit'
                })
            else:
                ambig_trades.append({
                    'trade': t, 'exit_pnl': exit_pnl, 'hold_pnl': h_pnl,
                    'zone': 'ambiguous'
                })

    # Compute metrics
    def bucket_metrics(bucket, name):
        exit_pnls = [b['exit_pnl'] for b in bucket if b['exit_pnl'] is not None]
        hold_pnls = [b['hold_pnl'] for b in bucket if b['hold_pnl'] is not None]
        return {
            'name': name,
            'n': len(bucket),
            'n_with_hold_pnl': len(hold_pnls),
            'exit_pnl_avg': np.mean(exit_pnls) if exit_pnls else np.nan,
            'exit_pnl_trimmed': trimmed_mean(exit_pnls) if exit_pnls else np.nan,
            'hold_pnl_avg': np.mean(hold_pnls) if hold_pnls else np.nan,
            'hold_pnl_trimmed': trimmed_mean(hold_pnls) if hold_pnls else np.nan,
            'hold_pnl_median': np.median(hold_pnls) if hold_pnls else np.nan,
            'hold_win_rate': win_rate(hold_pnls),
            'hold_sharpe': sharpe(hold_pnls),
            'hold_pnls': hold_pnls,
            'exit_pnls': exit_pnls,
        }

    total_n = len(hold_trades) + len(exit_trades) + len(ambig_trades)

    # Combined portfolio: held trades get their hold PnL, exited get their exit PnL
    # Ambiguous: test both hold and exit
    combined_hold_pnls = []
    combined_exit_pnls = []
    for b in hold_trades:
        if b['hold_pnl'] is not None:
            combined_hold_pnls.append(b['hold_pnl'])
    for b in exit_trades:
        combined_exit_pnls.append(b['exit_pnl'])

    # Portfolio A: ambiguous = HOLD
    portfolio_a = combined_hold_pnls.copy()
    for b in ambig_trades:
        if b['hold_pnl'] is not None:
            portfolio_a.append(b['hold_pnl'])
    portfolio_a.extend(combined_exit_pnls)

    # Portfolio B: ambiguous = EXIT
    portfolio_b = combined_hold_pnls.copy()
    for b in ambig_trades:
        portfolio_b.append(b['exit_pnl'])
    portfolio_b.extend(combined_exit_pnls)

    return {
        't_check': t_check,
        'total_n': total_n,
        'hold': bucket_metrics(hold_trades, 'hold'),
        'exit': bucket_metrics(exit_trades, 'exit'),
        'ambiguous': bucket_metrics(ambig_trades, 'ambiguous'),
        'portfolio_a_avg': np.mean(portfolio_a) if portfolio_a else np.nan,
        'portfolio_a_trimmed': trimmed_mean(portfolio_a) if portfolio_a else np.nan,
        'portfolio_a_sharpe': sharpe(portfolio_a),
        'portfolio_a_n': len(portfolio_a),
        'portfolio_b_avg': np.mean(portfolio_b) if portfolio_b else np.nan,
        'portfolio_b_trimmed': trimmed_mean(portfolio_b) if portfolio_b else np.nan,
        'portfolio_b_sharpe': sharpe(portfolio_b),
        'portfolio_b_n': len(portfolio_b),
    }


# ─── SHORT-only analysis ─────────────────────────────────────────────────────
print("=" * 80)
print("SECTION 1: SHORT TRADES — Time-Based Commitment Window (T+2 Hold)")
print("=" * 80)

# Baseline: all SHORT trades, hold to T+2
short_t2_pnls = [t['t2_pnl'] for t in short_trades if t['t2_pnl'] is not None]
short_60m_pnls = [t['pnl_60m'] for t in short_trades if t['pnl_60m'] is not None]
short_eod_pnls = [t['eod_pnl'] for t in short_trades if t['eod_pnl'] is not None]

print(f"\nBASELINE — All SHORT trades entered at 09:30:")
print(f"  T+2 hold:  n={len(short_t2_pnls)}{low_n_flag(len(short_t2_pnls))}, "
      f"avg={np.mean(short_t2_pnls):.2f}%, trimmed={trimmed_mean(short_t2_pnls):.2f}%, "
      f"WR={win_rate(short_t2_pnls):.1f}%, Sharpe={sharpe(short_t2_pnls):.3f}")
print(f"  60m hold:  n={len(short_60m_pnls)}{low_n_flag(len(short_60m_pnls))}, "
      f"avg={np.mean(short_60m_pnls):.2f}%, trimmed={trimmed_mean(short_60m_pnls):.2f}%, "
      f"WR={win_rate(short_60m_pnls):.1f}%, Sharpe={sharpe(short_60m_pnls):.3f}")
print(f"  EOD hold:  n={len(short_eod_pnls)}{low_n_flag(len(short_eod_pnls))}, "
      f"avg={np.mean(short_eod_pnls):.2f}%, trimmed={trimmed_mean(short_eod_pnls):.2f}%, "
      f"WR={win_rate(short_eod_pnls):.1f}%, Sharpe={sharpe(short_eod_pnls):.3f}")

# Run each checkpoint
print(f"\n{'─'*80}")
print(f"{'T_check':<8} {'n_hold':<8} {'n_exit':<8} {'n_ambig':<8} "
      f"{'Hold Avg%':>10} {'Hold Trim':>10} {'Hold WR':>8} {'Hold Shp':>9} "
      f"{'Exit Avg%':>10} {'Port A%':>8} {'Port B%':>8}")
print(f"{'─'*80}")

short_results = {}
for t_check in CHECKPOINTS:
    res = analyze_checkpoint(short_trades, t_check, hold_pnl_key='t2_pnl',
                             direction_filter='SHORT')
    short_results[t_check] = res

    h = res['hold']
    e = res['exit']
    a = res['ambiguous']

    ln_h = low_n_flag(h['n_with_hold_pnl'])
    ln_e = low_n_flag(e['n'])

    print(f"{t_check:<8} {h['n']:<8} {e['n']:<8} {a['n']:<8} "
          f"{h['hold_pnl_avg']:>10.2f} {h['hold_pnl_trimmed']:>10.2f} "
          f"{h['hold_win_rate']:>7.1f}% {h['hold_sharpe']:>9.3f} "
          f"{e['exit_pnl_avg']:>10.2f} "
          f"{res['portfolio_a_avg']:>8.2f} {res['portfolio_b_avg']:>8.2f}"
          f"{ln_h}{ln_e}")

# Also run with 60m hold as the hold PnL (more data)
print(f"\n{'─'*80}")
print("SHORT — Using 60m hold PnL (larger n):")
print(f"{'─'*80}")
print(f"{'T_check':<8} {'n_hold':<8} {'n_exit':<8} {'n_ambig':<8} "
      f"{'Hold Avg%':>10} {'Hold Trim':>10} {'Hold WR':>8} {'Hold Shp':>9} "
      f"{'Exit Avg%':>10} {'Port A%':>8} {'Port B%':>8}")
print(f"{'─'*80}")

short_results_60m = {}
for t_check in CHECKPOINTS:
    res = analyze_checkpoint(short_trades, t_check, hold_pnl_key='pnl_60m',
                             direction_filter='SHORT')
    short_results_60m[t_check] = res

    h = res['hold']
    e = res['exit']
    a = res['ambiguous']
    ln_h = low_n_flag(h['n_with_hold_pnl'])

    print(f"{t_check:<8} {h['n']:<8} {e['n']:<8} {a['n']:<8} "
          f"{h['hold_pnl_avg']:>10.2f} {h['hold_pnl_trimmed']:>10.2f} "
          f"{h['hold_win_rate']:>7.1f}% {h['hold_sharpe']:>9.3f} "
          f"{e['exit_pnl_avg']:>10.2f} "
          f"{res['portfolio_a_avg']:>8.2f} {res['portfolio_b_avg']:>8.2f}"
          f"{ln_h}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ALL trades analysis (direction-aware)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("SECTION 2: ALL TRADES — Direction-Aware Commitment Window")
print(f"{'='*80}")

# Baseline
all_60m = [t['pnl_60m'] for t in trades if t['pnl_60m'] is not None]
all_eod = [t['eod_pnl'] for t in trades if t['eod_pnl'] is not None]

print(f"\nBASELINE — All {len(trades)} trades:")
print(f"  60m:  n={len(all_60m)}, avg={np.mean(all_60m):.2f}%, "
      f"trimmed={trimmed_mean(all_60m):.2f}%, WR={win_rate(all_60m):.1f}%, "
      f"Sharpe={sharpe(all_60m):.3f}")
print(f"  EOD:  n={len(all_eod)}, avg={np.mean(all_eod):.2f}%, "
      f"trimmed={trimmed_mean(all_eod):.2f}%, WR={win_rate(all_eod):.1f}%, "
      f"Sharpe={sharpe(all_eod):.3f}")

# 60m hold
print(f"\n{'─'*80}")
print("ALL TRADES — 60m hold PnL:")
print(f"{'─'*80}")
print(f"{'T_check':<8} {'n_hold':<8} {'n_exit':<8} {'n_ambig':<8} "
      f"{'Hold Avg%':>10} {'Hold Trim':>10} {'Hold WR':>8} {'Hold Shp':>9} "
      f"{'Exit Avg%':>10} {'Port A%':>8} {'Port B%':>8}")
print(f"{'─'*80}")

all_results_60m = {}
for t_check in CHECKPOINTS:
    res = analyze_checkpoint(trades, t_check, hold_pnl_key='pnl_60m')
    all_results_60m[t_check] = res

    h = res['hold']
    e = res['exit']
    a = res['ambiguous']
    ln_h = low_n_flag(h['n_with_hold_pnl'])

    print(f"{t_check:<8} {h['n']:<8} {e['n']:<8} {a['n']:<8} "
          f"{h['hold_pnl_avg']:>10.2f} {h['hold_pnl_trimmed']:>10.2f} "
          f"{h['hold_win_rate']:>7.1f}% {h['hold_sharpe']:>9.3f} "
          f"{e['exit_pnl_avg']:>10.2f} "
          f"{res['portfolio_a_avg']:>8.2f} {res['portfolio_b_avg']:>8.2f}"
          f"{ln_h}")

# EOD hold
print(f"\n{'─'*80}")
print("ALL TRADES — EOD hold PnL:")
print(f"{'─'*80}")
print(f"{'T_check':<8} {'n_hold':<8} {'n_exit':<8} {'n_ambig':<8} "
      f"{'Hold Avg%':>10} {'Hold Trim':>10} {'Hold WR':>8} {'Hold Shp':>9} "
      f"{'Exit Avg%':>10} {'Port A%':>8} {'Port B%':>8}")
print(f"{'─'*80}")

all_results_eod = {}
for t_check in CHECKPOINTS:
    res = analyze_checkpoint(trades, t_check, hold_pnl_key='eod_pnl')
    all_results_eod[t_check] = res

    h = res['hold']
    e = res['exit']
    a = res['ambiguous']
    ln_h = low_n_flag(h['n_with_hold_pnl'])

    print(f"{t_check:<8} {h['n']:<8} {e['n']:<8} {a['n']:<8} "
          f"{h['hold_pnl_avg']:>10.2f} {h['hold_pnl_trimmed']:>10.2f} "
          f"{h['hold_win_rate']:>7.1f}% {h['hold_sharpe']:>9.3f} "
          f"{e['exit_pnl_avg']:>10.2f} "
          f"{res['portfolio_a_avg']:>8.2f} {res['portfolio_b_avg']:>8.2f}"
          f"{ln_h}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Ambiguous Zone Analysis
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("SECTION 3: AMBIGUOUS ZONE ANALYSIS")
print(f"{'='*80}")
print("\nWhat fraction of trades are inside the OR range at each checkpoint?")
print(f"{'─'*80}")
print(f"{'T_check':<8} {'SHORT':<20} {'LONG':<20} {'ALL':<20}")
print(f"{'':8} {'n_ambig/total':>12} {'%':>6} {'n_ambig/total':>12} {'%':>6} "
      f"{'n_ambig/total':>12} {'%':>6}")
print(f"{'─'*80}")

for t_check in CHECKPOINTS:
    # SHORT
    res_s = analyze_checkpoint(short_trades, t_check, hold_pnl_key='pnl_60m',
                                direction_filter='SHORT')
    # LONG
    res_l = analyze_checkpoint(long_trades, t_check, hold_pnl_key='pnl_60m',
                                direction_filter='LONG')
    # ALL
    res_a = all_results_60m[t_check]

    def fmt_ambig(res):
        total = res['total_n']
        amb = res['ambiguous']['n']
        pct = amb / total * 100 if total > 0 else 0
        return f"{amb}/{total}", f"{pct:.0f}%"

    s_r, s_p = fmt_ambig(res_s)
    l_r, l_p = fmt_ambig(res_l)
    a_r, a_p = fmt_ambig(res_a)
    print(f"{t_check:<8} {s_r:>12} {s_p:>6} {l_r:>12} {l_p:>6} {a_r:>12} {a_p:>6}")

# Ambiguous zone PnL
print(f"\nAmbiguous zone trades — what happens if you hold them?")
print(f"{'─'*80}")
print(f"{'T_check':<8} {'n_ambig':>8} {'Hold 60m avg':>13} {'Hold EOD avg':>13} "
      f"{'Exit avg':>9}")
print(f"{'─'*80}")

for t_check in CHECKPOINTS:
    res_60m = all_results_60m[t_check]
    res_eod = all_results_eod[t_check]
    a60 = res_60m['ambiguous']
    aeod = res_eod['ambiguous']

    print(f"{t_check:<8} {a60['n']:>8} {a60['hold_pnl_avg']:>13.2f} "
          f"{aeod['hold_pnl_avg']:>13.2f} {a60['exit_pnl_avg']:>9.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Optimal Commitment Time
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("SECTION 4: OPTIMAL COMMITMENT TIME")
print(f"{'='*80}")
print("\nFor each trade, find the earliest minute the price is outside OR on the")
print("correct side for 5+ consecutive minutes.")

# Generate minute-by-minute times from 09:30 to 10:30
def time_range(start_h, start_m, end_h, end_m):
    """Generate HH:MM strings."""
    times = []
    h, m = start_h, start_m
    while (h, m) <= (end_h, end_m):
        times.append(f"{h:02d}:{m:02d}")
        m += 1
        if m >= 60:
            m = 0
            h += 1
    return times

all_minutes = time_range(9, 30, 10, 30)

commitment_times = []
commitment_times_short = []

for t in trades:
    direction = t['direction']
    or_high = t['or_high']
    or_low = t['or_low']
    bars = t['bars']

    # Track consecutive minutes on correct side
    streak = 0
    commit_time = None

    for minute in all_minutes:
        if minute not in bars:
            streak = 0
            continue

        price = bars[minute]['c']

        if direction == 'SHORT':
            on_correct_side = price < or_low
        else:
            on_correct_side = price > or_high

        if on_correct_side:
            streak += 1
            if streak >= 5 and commit_time is None:
                # Committed 5 minutes ago
                commit_time = minute
        else:
            streak = 0

    if commit_time is not None:
        commitment_times.append(commit_time)
        if direction == 'SHORT':
            commitment_times_short.append(commit_time)

n_never = len(trades) - len(commitment_times)
n_never_short = len(short_trades) - len(commitment_times_short)

print(f"\nALL trades: {len(commitment_times)}/{len(trades)} commit within 09:30-10:30 "
      f"({n_never} never commit)")
print(f"SHORT only: {len(commitment_times_short)}/{len(short_trades)} commit "
      f"({n_never_short} never commit)")

# Distribution of commitment times
if commitment_times:
    # Convert to minutes since 09:30
    def time_to_min(t):
        h, m = map(int, t.split(':'))
        return (h - 9) * 60 + m - 30

    ct_mins = [time_to_min(t) for t in commitment_times]
    ct_mins_short = [time_to_min(t) for t in commitment_times_short]

    print(f"\nCommitment time distribution (minutes after 09:30):")
    print(f"  ALL:   median={np.median(ct_mins):.0f}min, "
          f"mean={np.mean(ct_mins):.1f}min, "
          f"25th={np.percentile(ct_mins, 25):.0f}min, "
          f"75th={np.percentile(ct_mins, 75):.0f}min")
    if ct_mins_short:
        print(f"  SHORT: median={np.median(ct_mins_short):.0f}min, "
              f"mean={np.mean(ct_mins_short):.1f}min, "
              f"25th={np.percentile(ct_mins_short, 25):.0f}min, "
              f"75th={np.percentile(ct_mins_short, 75):.0f}min")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Detailed breakdown by checkpoint (SHORT with 60m — best data)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("SECTION 5: DETAILED CHECKPOINT ANALYSIS — SHORT 60m")
print(f"{'='*80}")

for t_check in CHECKPOINTS:
    res = short_results_60m[t_check]
    h = res['hold']
    e = res['exit']
    a = res['ambiguous']

    print(f"\n--- T_check = {t_check} ---")
    print(f"  HOLD (price < or_low, SHORT working):")
    print(f"    n={h['n']}{low_n_flag(h['n'])}, "
          f"60m avg={h['hold_pnl_avg']:.2f}%, trimmed={h['hold_pnl_trimmed']:.2f}%, "
          f"WR={h['hold_win_rate']:.1f}%, Sharpe={h['hold_sharpe']:.3f}")
    print(f"    At-checkpoint avg={h['exit_pnl_avg']:.2f}%")

    print(f"  EXIT (price > or_high, SHORT failing):")
    print(f"    n={e['n']}{low_n_flag(e['n'])}, "
          f"exit PnL avg={e['exit_pnl_avg']:.2f}%, trimmed={e['exit_pnl_trimmed']:.2f}%")
    if e['hold_pnls']:
        print(f"    (if held: 60m avg={e['hold_pnl_avg']:.2f}%, "
              f"WR={e['hold_win_rate']:.1f}%) <- what we avoided")

    print(f"  AMBIGUOUS (inside OR range):")
    print(f"    n={a['n']}{low_n_flag(a['n'])}")
    if a['hold_pnls']:
        print(f"    if held: 60m avg={a['hold_pnl_avg']:.2f}%, "
              f"WR={a['hold_win_rate']:.1f}%")

    print(f"  PORTFOLIO A (ambig=hold): avg={res['portfolio_a_avg']:.2f}%, "
          f"Sharpe={res['portfolio_a_sharpe']:.3f}, n={res['portfolio_a_n']}")
    print(f"  PORTFOLIO B (ambig=exit): avg={res['portfolio_b_avg']:.2f}%, "
          f"Sharpe={res['portfolio_b_sharpe']:.3f}, n={res['portfolio_b_n']}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Summary and Key Findings
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("SECTION 6: SUMMARY & KEY FINDINGS")
print(f"{'='*80}")

# Find best checkpoint for SHORT (60m)
best_t_check_short = None
best_port_avg_short = -999
for t_check in CHECKPOINTS:
    res = short_results_60m[t_check]
    h = res['hold']
    if h['n'] >= LOW_N_THRESHOLD:
        if h['hold_pnl_avg'] > best_port_avg_short:
            best_port_avg_short = h['hold_pnl_avg']
            best_t_check_short = t_check

# Find best checkpoint for ALL (60m)
best_t_check_all = None
best_port_avg_all = -999
for t_check in CHECKPOINTS:
    res = all_results_60m[t_check]
    if res['portfolio_a_avg'] > best_port_avg_all:
        best_port_avg_all = res['portfolio_a_avg']
        best_t_check_all = t_check

print(f"\nBest checkpoint for SHORT held trades (60m): {best_t_check_short} "
      f"(avg={best_port_avg_short:.2f}%)")
print(f"Best checkpoint for ALL portfolio A (60m): {best_t_check_all} "
      f"(avg={best_port_avg_all:.2f}%)")

short_baseline_60m = np.mean(short_60m_pnls) if short_60m_pnls else np.nan
all_baseline_60m = np.mean(all_60m) if all_60m else np.nan

print(f"\nSHORT baseline (all trades, 60m): {short_baseline_60m:.2f}%")
print(f"ALL baseline (all trades, 60m): {all_baseline_60m:.2f}%")

if best_t_check_short:
    res = short_results_60m[best_t_check_short]
    h = res['hold']
    e = res['exit']
    delta = h['hold_pnl_avg'] - short_baseline_60m
    print(f"\nSHORT: {best_t_check_short} filter -> hold trades avg "
          f"{h['hold_pnl_avg']:.2f}% vs baseline {short_baseline_60m:.2f}% "
          f"(delta={delta:+.2f}%)")
    print(f"  But n drops from {len(short_60m_pnls)} to {h['n_with_hold_pnl']}")
    print(f"  Exited trades ({e['n']}) had avg exit PnL = {e['exit_pnl_avg']:.2f}%")
    if e['hold_pnls']:
        print(f"  If those exited trades had been held: avg 60m = {e['hold_pnl_avg']:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("GENERATING CHARTS...")
print(f"{'='*80}")

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('E14: Time-Based Commitment Window', fontsize=16, fontweight='bold')

# Chart 1: SHORT held trades avg PnL by checkpoint (T+2 and 60m)
ax1 = axes[0, 0]
tc_labels = CHECKPOINTS
hold_avgs_t2 = [short_results[t]['hold']['hold_pnl_avg'] for t in CHECKPOINTS]
hold_avgs_60m = [short_results_60m[t]['hold']['hold_pnl_avg'] for t in CHECKPOINTS]
hold_ns_60m = [short_results_60m[t]['hold']['n'] for t in CHECKPOINTS]

ax1.plot(tc_labels, hold_avgs_60m, 'b-o', label='Hold 60m PnL', linewidth=2)
ax1.plot(tc_labels, hold_avgs_t2, 'r--s', label='Hold T+2 PnL', linewidth=2)
ax1.axhline(y=short_baseline_60m, color='blue', linestyle=':', alpha=0.5,
            label=f'Baseline 60m ({short_baseline_60m:.2f}%)')
if short_t2_pnls:
    ax1.axhline(y=np.mean(short_t2_pnls), color='red', linestyle=':', alpha=0.5,
                label=f'Baseline T+2 ({np.mean(short_t2_pnls):.2f}%)')
ax1.set_xlabel('Checkpoint Time')
ax1.set_ylabel('Avg PnL (%)')
ax1.set_title('SHORT: Held Trades Avg PnL by Checkpoint')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Secondary axis for n
ax1b = ax1.twinx()
ax1b.bar(tc_labels, hold_ns_60m, alpha=0.15, color='gray', label='n (60m)')
ax1b.set_ylabel('n (held trades)', color='gray')
ax1b.tick_params(axis='y', labelcolor='gray')

# Chart 2: SHORT combined portfolio PnL
ax2 = axes[0, 1]
port_a = [short_results_60m[t]['portfolio_a_avg'] for t in CHECKPOINTS]
port_b = [short_results_60m[t]['portfolio_b_avg'] for t in CHECKPOINTS]

ax2.plot(tc_labels, port_a, 'g-o', label='Portfolio A (ambig=hold)', linewidth=2)
ax2.plot(tc_labels, port_b, 'm-s', label='Portfolio B (ambig=exit)', linewidth=2)
ax2.axhline(y=short_baseline_60m, color='gray', linestyle='--', alpha=0.5,
            label=f'Baseline ({short_baseline_60m:.2f}%)')
ax2.set_xlabel('Checkpoint Time')
ax2.set_ylabel('Combined Portfolio Avg PnL (%)')
ax2.set_title('SHORT: Portfolio PnL (held T+2 + exited at checkpoint)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Chart 3: Stacked bar — n_hold, n_exit, n_ambiguous
ax3 = axes[1, 0]
n_holds = [short_results_60m[t]['hold']['n'] for t in CHECKPOINTS]
n_exits = [short_results_60m[t]['exit']['n'] for t in CHECKPOINTS]
n_ambigs = [short_results_60m[t]['ambiguous']['n'] for t in CHECKPOINTS]
x = np.arange(len(CHECKPOINTS))
width = 0.6

ax3.bar(x, n_holds, width, label='Hold (correct side)', color='#2ecc71')
ax3.bar(x, n_ambigs, width, bottom=n_holds, label='Ambiguous (inside OR)',
        color='#f39c12')
ax3.bar(x, n_exits, width, bottom=[h+a for h, a in zip(n_holds, n_ambigs)],
        label='Exit (wrong side)', color='#e74c3c')
ax3.set_xticks(x)
ax3.set_xticklabels(CHECKPOINTS)
ax3.set_xlabel('Checkpoint Time')
ax3.set_ylabel('Number of Trades')
ax3.set_title('SHORT: Trade Classification by Checkpoint')
ax3.legend(fontsize=8)

# Chart 4: Sharpe of held trades
ax4 = axes[1, 1]
sharpes_60m = [short_results_60m[t]['hold']['hold_sharpe'] for t in CHECKPOINTS]
sharpes_port_a = [short_results_60m[t]['portfolio_a_sharpe'] for t in CHECKPOINTS]

ax4.plot(tc_labels, sharpes_60m, 'b-o', label='Held trades Sharpe (60m)', linewidth=2)
ax4.plot(tc_labels, sharpes_port_a, 'g--s', label='Portfolio A Sharpe', linewidth=2)
if short_60m_pnls:
    ax4.axhline(y=sharpe(short_60m_pnls), color='gray', linestyle='--', alpha=0.5,
                label=f'Baseline Sharpe ({sharpe(short_60m_pnls):.3f})')
ax4.set_xlabel('Checkpoint Time')
ax4.set_ylabel('Sharpe Ratio')
ax4.set_title('SHORT: Sharpe by Checkpoint')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Chart 5: Baseline vs best checkpoint (bar comparison)
ax5 = axes[2, 0]
categories = ['Baseline\n(all SHORT)', f'Best T_check\n({best_t_check_short})\nHeld Only',
              f'Best T_check\n({best_t_check_short})\nPortfolio A']
if best_t_check_short:
    best_res = short_results_60m[best_t_check_short]
    vals = [short_baseline_60m,
            best_res['hold']['hold_pnl_avg'],
            best_res['portfolio_a_avg']]
    ns = [len(short_60m_pnls),
          best_res['hold']['n_with_hold_pnl'],
          best_res['portfolio_a_n']]
    colors = ['#3498db', '#2ecc71', '#27ae60']
    bars = ax5.bar(categories, vals, color=colors)
    for bar, n in zip(bars, ns):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                 f'n={n}', ha='center', va='bottom', fontsize=9)
ax5.set_ylabel('Avg PnL (%)')
ax5.set_title('SHORT 60m: Baseline vs Best Checkpoint')
ax5.grid(True, alpha=0.3, axis='y')

# Chart 6: Commitment time histogram
ax6 = axes[2, 1]
if commitment_times:
    ct_mins_all = [time_to_min(t) for t in commitment_times]
    ct_mins_s = [time_to_min(t) for t in commitment_times_short]
    bins = np.arange(0, 62, 2)
    ax6.hist(ct_mins_all, bins=bins, alpha=0.5, label=f'All (n={len(ct_mins_all)})',
             color='#3498db', edgecolor='black')
    if ct_mins_s:
        ax6.hist(ct_mins_s, bins=bins, alpha=0.5, label=f'SHORT (n={len(ct_mins_s)})',
                 color='#e74c3c', edgecolor='black')
    ax6.set_xlabel('Minutes After 09:30')
    ax6.set_ylabel('Number of Trades')
    ax6.set_title('Distribution of Optimal Commitment Times\n(5+ consecutive minutes outside OR)')
    ax6.legend(fontsize=8)

    # Add checkpoint markers
    for cp in CHECKPOINTS:
        cp_min = time_to_min(cp)
        ax6.axvline(x=cp_min, color='gray', linestyle=':', alpha=0.4)
else:
    ax6.text(0.5, 0.5, 'No commitment data', ha='center', va='center',
             transform=ax6.transAxes)

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = BASE / 'v4_e14_time_window.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved chart: {chart_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL CHART: ALL trades analysis
# ═══════════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('E14: Time-Based Commitment — ALL Trades (Direction-Aware)',
              fontsize=14, fontweight='bold')

# Chart A: ALL trades held PnL by checkpoint
ax_a = axes2[0, 0]
hold_avgs_all_60m = [all_results_60m[t]['hold']['hold_pnl_avg'] for t in CHECKPOINTS]
hold_avgs_all_eod = [all_results_eod[t]['hold']['hold_pnl_avg'] for t in CHECKPOINTS]
hold_ns_all = [all_results_60m[t]['hold']['n'] for t in CHECKPOINTS]

ax_a.plot(tc_labels, hold_avgs_all_60m, 'b-o', label='Hold 60m', linewidth=2)
ax_a.plot(tc_labels, hold_avgs_all_eod, 'r--s', label='Hold EOD', linewidth=2)
ax_a.axhline(y=all_baseline_60m, color='blue', linestyle=':', alpha=0.5,
             label=f'Baseline 60m ({all_baseline_60m:.2f}%)')
ax_a.set_xlabel('Checkpoint Time')
ax_a.set_ylabel('Avg PnL (%)')
ax_a.set_title('ALL: Held Trades Avg PnL')
ax_a.legend(fontsize=8)
ax_a.grid(True, alpha=0.3)
ax_ab = ax_a.twinx()
ax_ab.bar(tc_labels, hold_ns_all, alpha=0.15, color='gray')
ax_ab.set_ylabel('n', color='gray')

# Chart B: ALL portfolio PnL
ax_b = axes2[0, 1]
port_a_all = [all_results_60m[t]['portfolio_a_avg'] for t in CHECKPOINTS]
port_b_all = [all_results_60m[t]['portfolio_b_avg'] for t in CHECKPOINTS]
ax_b.plot(tc_labels, port_a_all, 'g-o', label='Portfolio A (ambig=hold)', linewidth=2)
ax_b.plot(tc_labels, port_b_all, 'm-s', label='Portfolio B (ambig=exit)', linewidth=2)
ax_b.axhline(y=all_baseline_60m, color='gray', linestyle='--', alpha=0.5,
             label=f'Baseline ({all_baseline_60m:.2f}%)')
ax_b.set_xlabel('Checkpoint Time')
ax_b.set_ylabel('Combined Portfolio Avg PnL (%)')
ax_b.set_title('ALL: Portfolio PnL')
ax_b.legend(fontsize=8)
ax_b.grid(True, alpha=0.3)

# Chart C: Stacked bar ALL
ax_c = axes2[1, 0]
n_h_a = [all_results_60m[t]['hold']['n'] for t in CHECKPOINTS]
n_e_a = [all_results_60m[t]['exit']['n'] for t in CHECKPOINTS]
n_a_a = [all_results_60m[t]['ambiguous']['n'] for t in CHECKPOINTS]
ax_c.bar(x, n_h_a, width, label='Hold', color='#2ecc71')
ax_c.bar(x, n_a_a, width, bottom=n_h_a, label='Ambiguous', color='#f39c12')
ax_c.bar(x, n_e_a, width, bottom=[h+a for h, a in zip(n_h_a, n_a_a)],
         label='Exit', color='#e74c3c')
ax_c.set_xticks(x)
ax_c.set_xticklabels(CHECKPOINTS)
ax_c.set_xlabel('Checkpoint Time')
ax_c.set_ylabel('Number of Trades')
ax_c.set_title('ALL: Trade Classification by Checkpoint')
ax_c.legend(fontsize=8)

# Chart D: Sharpe ALL
ax_d = axes2[1, 1]
sharpes_all_60m = [all_results_60m[t]['hold']['hold_sharpe'] for t in CHECKPOINTS]
sharpes_port_a_all = [all_results_60m[t]['portfolio_a_sharpe'] for t in CHECKPOINTS]
ax_d.plot(tc_labels, sharpes_all_60m, 'b-o', label='Held trades Sharpe', linewidth=2)
ax_d.plot(tc_labels, sharpes_port_a_all, 'g--s', label='Portfolio A Sharpe', linewidth=2)
ax_d.axhline(y=sharpe(all_60m), color='gray', linestyle='--', alpha=0.5,
             label=f'Baseline ({sharpe(all_60m):.3f})')
ax_d.set_xlabel('Checkpoint Time')
ax_d.set_ylabel('Sharpe Ratio')
ax_d.set_title('ALL: Sharpe by Checkpoint')
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart2_path = BASE / 'v4_e14_time_window_all.png'
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved chart: {chart2_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS JSON
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_results(results_dict):
    """Convert results to JSON-serializable format."""
    out = {}
    for t_check, res in results_dict.items():
        r = {
            't_check': t_check,
            'total_n': res['total_n'],
            'hold_n': res['hold']['n'],
            'hold_n_with_pnl': res['hold']['n_with_hold_pnl'],
            'hold_avg': res['hold']['hold_pnl_avg'],
            'hold_trimmed': res['hold']['hold_pnl_trimmed'],
            'hold_win_rate': res['hold']['hold_win_rate'],
            'hold_sharpe': res['hold']['hold_sharpe'],
            'exit_n': res['exit']['n'],
            'exit_avg': res['exit']['exit_pnl_avg'],
            'exit_trimmed': res['exit']['exit_pnl_trimmed'],
            'ambig_n': res['ambiguous']['n'],
            'ambig_hold_avg': res['ambiguous']['hold_pnl_avg'],
            'portfolio_a_avg': res['portfolio_a_avg'],
            'portfolio_a_sharpe': res['portfolio_a_sharpe'],
            'portfolio_a_n': res['portfolio_a_n'],
            'portfolio_b_avg': res['portfolio_b_avg'],
            'portfolio_b_sharpe': res['portfolio_b_sharpe'],
            'portfolio_b_n': res['portfolio_b_n'],
        }
        # Replace NaN with None for JSON
        for k, v in r.items():
            if isinstance(v, float) and np.isnan(v):
                r[k] = None
        out[t_check] = r
    return out


results_json = {
    'experiment': 'E14_time_based_commitment_window',
    'description': 'Enter at 09:30, evaluate at checkpoint if price is on correct side of OR',
    'baselines': {
        'short_60m': {
            'n': len(short_60m_pnls),
            'avg': float(np.mean(short_60m_pnls)) if short_60m_pnls else None,
            'trimmed': float(trimmed_mean(short_60m_pnls)) if short_60m_pnls else None,
            'win_rate': float(win_rate(short_60m_pnls)) if short_60m_pnls else None,
            'sharpe': float(sharpe(short_60m_pnls)) if short_60m_pnls else None,
        },
        'short_t2': {
            'n': len(short_t2_pnls),
            'avg': float(np.mean(short_t2_pnls)) if short_t2_pnls else None,
            'trimmed': float(trimmed_mean(short_t2_pnls)) if short_t2_pnls else None,
            'win_rate': float(win_rate(short_t2_pnls)) if short_t2_pnls else None,
            'sharpe': float(sharpe(short_t2_pnls)) if short_t2_pnls else None,
        },
        'all_60m': {
            'n': len(all_60m),
            'avg': float(np.mean(all_60m)) if all_60m else None,
            'trimmed': float(trimmed_mean(all_60m)) if all_60m else None,
            'win_rate': float(win_rate(all_60m)) if all_60m else None,
            'sharpe': float(sharpe(all_60m)) if all_60m else None,
        },
    },
    'short_t2_results': serialize_results(short_results),
    'short_60m_results': serialize_results(short_results_60m),
    'all_60m_results': serialize_results(all_results_60m),
    'all_eod_results': serialize_results(all_results_eod),
    'commitment_time_distribution': {
        'all_n_committed': len(commitment_times),
        'all_n_total': len(trades),
        'short_n_committed': len(commitment_times_short),
        'short_n_total': len(short_trades),
        'all_median_min': float(np.median(ct_mins_all)) if ct_mins_all else None,
        'all_mean_min': float(np.mean(ct_mins_all)) if ct_mins_all else None,
        'short_median_min': float(np.median(ct_mins_s)) if ct_mins_s else None,
        'short_mean_min': float(np.mean(ct_mins_s)) if ct_mins_s else None,
    },
    'best_checkpoint': {
        'short_60m': best_t_check_short,
        'all_60m': best_t_check_all,
    },
}

json_path = RESULTS / 'v4_e14_results.json'
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"Saved results: {json_path}")

print(f"\n{'='*80}")
print("E14 COMPLETE")
print(f"{'='*80}")
