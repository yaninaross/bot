# Scanner Optimization Lab

Systematic backtesting of scanner filter/selection strategies against Sep 8-11 2026 data.
Goal: find strategies that produce >2% avg PnL per trade.

## Approach
1. Pull all 169+ trades from BQ with full metadata
2. Test filter strategies (catalyst, direction, scanner, timing, gate removal)
3. Rank by avg PnL, win rate, and trade count (minimum viable frequency)
4. Cross-validate Sep 8-9 vs Sep 10-11
5. Report top strategies with full stats

## Files
- `optimization_harness.py` — main backtesting engine
- `results/` — per-strategy JSON results
- `OPTIMIZATION_REPORT.md` — final findings
