# Agent 1 — STRATEGY LOCK

## Locked Config: E2 Benchmark

This is the production strategy definition. Any deviation requires
ablation evidence showing improvement.

| Parameter       | Value                                          |
|-----------------|------------------------------------------------|
| Direction       | SHORT                                          |
| Universe        | Full US equity (Polygon grouped daily)         |
| Gap filter      | gap_pct >= 3% vs AH reference price            |
| Asset type      | CS only (Polygon reference `type == 'CS'`)     |
| Pattern         | sustained_drop only (bar 60 = 10:30 AM ET)     |
|                 | drop_30m < -3% AND recovery <= 50%             |
| Float filter    | SOFT — hardcoded 217-ticker dict only           |
|                 | Dict miss = INCLUDE (pass through)             |
|                 | Dict hit + float < 10M = EXCLUDE               |
| Gap calc        | gap = (open - ah_ref) / ah_ref; ah_ref = last   |
|                 | AH price (16:00-20:00 ET) on prior trading day  |
|                 | Falls back to prior close if no AH data         |
| Gap erosion     | skip if >50% of gap consumed by SCAN_TIME_ET    |
|                 | gap_consumed = (open - low) / (open - ref_price)|
| TOP_N           | 5 (top 5 by gap_pct descending)                |
| Ranking         | gap_pct descending (vs AH ref, no PM adj)       |
| PM reranking    | OFF                                            |
| Entry           | local_high: 3-bar window, scan bars 3-11       |
|                 | Fallback: bar 15 close (9:45 AM ET)            |
| Stop loss       | -50% disaster cap (entry × 1.50 for SHORT)     |
|                 | Human sets in TOS — not automated               |
| Exit            | 17:00 ET (1h after close, after-hours)         |
| Skip rule       | NONE — eval showed no threshold improves $      |
| Gap max         | None                                           |
| PNL cap         | None (SL=-50% is the disaster cap)             |
| Capital         | $100K/day, non-compounding                     |
| PnL formula     | (entry_price - exit_price) / entry_price × 100 |

## E2 Reference Numbers

| Metric        | Value      |
|---------------|------------|
| Trades        | 160        |
| Avg PnL       | +14.71%    |
| Win Rate      | 79%        |
| Sharpe        | 0.594      |
| Profit Factor | 6.26       |
| Total $ (NC)  | $+2,350,000|
| Period        | 2026-01-02 to 2026-09-11 (174 trading days) |

## Ablation Results (2026-09-18 run)

### Q1: Is E2 reproducible?
**YES.** Reproduced at +13.64% avg (166 trades), within 1.07pp of
+14.71% reference. Sharpe 0.547 vs 0.594. $2,264,952 vs $2,350,000.

### Q2: What is the costliest deviation?
**Hard float (A): -9.02pp.** Full ranking:

| Rank | Variant | Δ Avg PnL | Sharpe | Max Loss | $ NC |
|------|---------|-----------|--------|----------|------|
| 1 | A: Hard float | -9.02pp | 0.445 | -40.4% | $634K |
| 2 | E: TOP_N=5 | -6.70pp | 0.189 | -779.2% | $1,203K |
| 3 | B: PM reranking | -5.45pp | 0.123 | -779.2% | $1,352K |
| 4 | C: Exit EOD | -2.05pp | 0.509 | -86.2% | $1,924K |
| 5 | F: Skip >15% | -1.69pp | 0.493 | -132.6% | $1,674K |
| — | **Baseline** | **—** | **0.547** | **-132.6%** | **$2,265K** |
| 6 | G: PNL_CAP=-50% | +0.51pp | **0.624** | -50.0% | $2,350K |
| 7 | D: T+2 hold | +3.83pp | 0.385 | -273.3% | $2,901K |

### Q3: Best config WITH a PNL cap?
**PNL_CAP=-50% is a free upgrade.** Avg +14.16% (vs +13.64% base),
Sharpe improves 0.547→0.624, PF 5.47→6.58, max loss capped at -50%.
Total $ nearly identical ($2,350K vs $2,265K). Risk-adjusted, G
dominates the baseline.

### Key Insight: D (T+2) is a trap
T+2 shows +3.83pp higher average but Sharpe collapses 0.547→0.385
and max loss balloons to -273%. The overnight risk is not compensated.

## Divergences from CLAUDE.md "Frozen Baseline"

The CLAUDE.md CS+SD+GUARDS spec and this lock are now mostly aligned
after v1.3-1.5 updates. Remaining differences:

| Parameter     | E2 (this lock)   | CLAUDE.md Guards  |
|---------------|------------------|-------------------|
| Float miss    | INCLUDE          | HARD EXCLUDE      |
| Gap max       | None             | <= 75%            |
| Skip rule     | None             | >15% drop = skip  |
| Exit          | 17:00 ET (AH)   | T+2 close         |
| PNL cap       | None             | Not specified      |

## Change Log

| Version | Date       | Change                          | Backtest Result |
|---------|------------|---------------------------------|-----------------|
| 1.0     | 2026-09-18 | Initial E2 lock                 | +14.71% ref     |
| 1.1     | 2026-09-18 | Add -10% SL, remove skip rule   | SL: +$303K (FLAWED — post-hoc cap, not real intraday exit) |
| 1.2     | 2026-09-18 | SL -10%→-50% after real eval    | Real intraday SL eval: -10% costs -$382K (kills 21 recoveries). -50% adds +$53K (saves 3 disasters, kills 1). Sharpe 0.435→0.510. |
| 1.3     | 2026-09-21 | AH freshness filter             | Gap calc vs last AH price, not prior close. Catches stale M&A gaps (ACVA: daily +44.6% but fresh +0.6% → excluded). |
| 1.4     | 2026-09-21 | Gap erosion filter (50%)        | Skip if >50% of gap consumed by scan time. Catches faded gaps (TANH: 55% consumed → excluded). |
| 1.5     | 2026-09-21 | TOP_N 1→5, GAP_MIN 5%→3%       | Wider candidate pool; E2-A time gate removed (runs at any time). |
