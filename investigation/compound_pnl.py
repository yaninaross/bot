"""Compound daily PnL for Project X winners — standalone."""
import sys, os

# Load only the functions/constants from run_investigation.py, stop before main loop
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_investigation.py')
with open(script_path) as f:
    lines = f.readlines()

# Find where the main loop starts (line 1375 area — the print statement)
cutoff = None
for i, line in enumerate(lines):
    if 'PROJECT X — SEMI-AUTONOMOUS STRATEGY INVESTIGATION' in line:
        cutoff = i
        break
if cutoff is None:
    print("ERROR: couldn't find main loop start"); sys.exit(1)

code = ''.join(lines[:cutoff])
exec(compile(code, script_path, 'exec'), globals())

from collections import defaultdict

STARTING_CAPITAL = 100_000

HYPOTHESES_BY_ID = {h['id']: h for h in HYPOTHESES}

def compound_equity(trades, label="", starting=STARTING_CAPITAL):
    by_date = defaultdict(list)
    for t in trades:
        by_date[t['scan_date']].append(t['pnl'])

    sorted_dates = sorted(by_date.keys())
    equity = starting
    curve = []

    for dt in sorted_dates:
        pnls = by_date[dt]
        n = len(pnls)
        pos_size = equity / n
        day_dollar = sum(pos_size * (p / 100) for p in pnls)
        day_pct = (day_dollar / equity) * 100 if equity > 0 else 0
        equity += day_dollar
        curve.append((dt, equity, day_dollar, day_pct, n))

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  Period: {sorted_dates[0]} -> {sorted_dates[-1]}")
    print(f"  Starting capital: ${starting:,.0f}")
    print(f"{'='*72}")
    print(f"  {'Date':>12}  {'Equity':>14}  {'Day P&L':>12}  {'Day %':>8}  {'#':>3}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*12}  {'-'*8}  {'-'*3}")

    for dt, eq, dpnl, dpct, n in curve:
        marker = " ***" if dpct < -3 else ""
        print(f"  {dt:>12}  ${eq:>13,.2f}  ${dpnl:>+11,.2f}  {dpct:>+7.2f}%  {n:>3}{marker}")

    final = curve[-1][1] if curve else starting
    total_ret = (final - starting) / starting * 100
    n_days_with_trades = len(sorted_dates)
    n_trades = sum(len(by_date[d]) for d in sorted_dates)

    peak = starting
    max_dd = 0
    for _, eq, _, _, _ in curve:
        if eq > peak: peak = eq
        dd = (eq - peak) / peak * 100
        if dd < max_dd: max_dd = dd

    print(f"\n  Final equity:    ${final:,.2f}")
    print(f"  Total return:    {total_ret:+.1f}%")
    print(f"  Dollar profit:   ${final - starting:+,.2f}")
    print(f"  Max drawdown:    {max_dd:.2f}%")
    print(f"  Days w/ trades:  {n_days_with_trades}")
    print(f"  Total trades:    {n_trades}")
    print(f"  Avg trades/day:  {n_trades/n_days_with_trades:.1f}")
    print(f"{'='*72}")

    return curve


print("=" * 72)
print("  COMPOUND DAILY P&L — PROJECT X WINNERS")
print("  100% capital deployed daily, equal-weight, compounded")
print("=" * 72)

# --- AUG-OCT PERIOD ---
print("\nLoading Aug-Oct data...")
data = run_data_pipeline(EVAL_START, EVAL_END)

h006 = HYPOTHESES_BY_ID['H006']
# H037 was auto-generated — define inline
h037 = {
    'id': 'H037', 'desc': 'SHORT gap_up gap>=3% 17:00',
    'direction': 'SHORT', 'catalyst': 'gap_up',
    'entry': 'local_high', 'exit': '17:00',
    'filters': {'gap_min': 0.03, 'pattern': 'none'},
}

trades_006 = run_backtest(h006, data)
trades_037 = run_backtest(h037, data)

compound_equity(trades_006, "H006: SHORT sustained_drop LH→17:00 (Aug 1 - Oct 6, 2026)")
compound_equity(trades_037, "H037: SHORT gap>=3% LH→17:00 (Aug 1 - Oct 6, 2026)")

# --- FULL PERIOD ---
print("\n\nLoading full period data (Sep 2025 - Oct 2026)...")
full_data = run_data_pipeline(FULL_START, EVAL_END)

full_006 = run_backtest(h006, full_data)
full_037 = run_backtest(h037, full_data)

compound_equity(full_006, "H006: FULL PERIOD (Sep 15, 2025 - Oct 6, 2026)")
compound_equity(full_037, "H037: FULL PERIOD (Sep 15, 2025 - Oct 6, 2026)")
