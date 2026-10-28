#!/usr/bin/env python3
"""Master runner: build pipeline once, run all 6 investigation scripts sequentially."""
import sys, os, time

os.environ.setdefault("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

scripts = [
    'agent1_eval_sl_skip.py',
    'agent1_diagnostic_monthly.py',
    'agent1_regime_analysis.py',
    'agent1_time_decay_analysis.py',
    'agent1_gap_tier_analysis.py',
    'agent1_universe_refresh.py',
    'agent1_entry_pattern_analysis.py',
    'agent1_alt_signals.py',
]

t0 = time.time()
for i, script in enumerate(scripts):
    path = os.path.join(BASE, script)
    if not os.path.exists(path):
        print(f"\n  SKIP {script} — file not found")
        continue
    print(f"\n{'#'*80}")
    print(f"  [{i+1}/{len(scripts)}] RUNNING: {script}")
    print(f"{'#'*80}")
    t1 = time.time()
    try:
        exec(open(path).read())
    except Exception as e:
        print(f"\n  ERROR in {script}: {e}")
        import traceback; traceback.print_exc()
    print(f"\n  {script} finished in {time.time()-t1:.0f}s")

print(f"\n\n{'='*80}")
print(f"  ALL DONE — {time.time()-t0:.0f}s total")
print(f"{'='*80}")
