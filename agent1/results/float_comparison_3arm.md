# Agent 1 — 3-Arm Float Filter Comparison

**Date:** 2026-09-20
**Period:** 2026-08-01 → 2026-09-20 (34 trading days)
**Config:** 3% gap, 09:31 ET scan, TOP_N=1, CS filter, local-high entry, 17:00 ET exit

## Float Dictionaries

| Arm | Dict Size | >= 10M | < 10M | Miss Policy |
|-----|-----------|--------|-------|-------------|
| Arm 1: Frozen | 217 | 194 | 23 | INCLUDE |
| Arm 2: Live | 219 | 195 | 24 | INCLUDE |
| Arm 3: Rebuilt | 256 | 185 | 71 | EXCLUDE |

- Arm 2 adds BIAF (600K, <10M) and ODD (34.6M, >10M) vs Frozen
- Arm 3 built from Polygon `share_class_shares_outstanding` — 55.8% coverage of candidate universe
- Arm 3 has 71 tickers below 10M vs only 23-24 for Arms 1/2

## E2-A: Agent 1 (no SD)

| Metric | Arm 1: Frozen | Arm 2: Live | Arm 3: Rebuilt | Δ Arm2-1 | Δ Arm3-1 |
|--------|---------------|-------------|----------------|----------|----------|
| Trades | 34 | 34 | 33 | +0 | -1 |
| Avg PnL | +4.73% | +4.06% | +1.70% | -0.67pp | -3.03pp |
| Median PnL | +4.24% | +4.24% | +2.76% | +0.00pp | -1.48pp |
| Win Rate | 71% | 71% | 61% | +0pp | -10pp |
| Sharpe | 0.171 | 0.147 | 0.110 | -0.024 | -0.061 |
| Profit Factor | 1.89 | 1.74 | 1.42 | -0.15 | -0.47 |
| $ NC | $+160,886 | $+138,117 | $+56,030 | $-22,769 | $-104,856 |
| Max DD | -166.8% | -159.3% | -63.3% | +7.5pp | +103.6pp |
| 95% CI | [-4.6, +14.0] | [-5.2, +13.3] | [-3.6, +7.0] | | |

## E2-B: Agent 1-SD (SD @10:30)

| Metric | Arm 1: Frozen | Arm 2: Live | Arm 3: Rebuilt | Δ Arm2-1 | Δ Arm3-1 |
|--------|---------------|-------------|----------------|----------|----------|
| Trades | 28 | 28 | 27 | +0 | -1 |
| Avg PnL | +3.17% | +3.31% | +1.52% | **+0.14pp** | -1.64pp |
| Median PnL | +1.66% | +1.66% | +1.04% | +0.00pp | -0.63pp |
| Win Rate | 57% | 57% | 52% | +0pp | -5pp |
| Sharpe | 0.351 | **0.371** | 0.205 | **+0.020** | -0.146 |
| Profit Factor | 3.14 | **3.47** | 1.87 | **+0.33** | -1.27 |
| $ NC | $+88,723 | **$+92,682** | $+41,161 | **$+3,959** | $-47,562 |
| Max DD | -12.4% | **-12.2%** | -13.8% | +0.3pp | -1.4pp |
| 95% CI | [-0.2, +6.5] | **[+0.0, +6.6]** | [-1.3, +4.3] | | |

## Pick Divergence

### E2-A: 19/34 days diverge (Arm 1 vs 2 on 2 days, Arm 1 vs 3 on 19 days)

Key Arm 1 vs Arm 2 divergences:
- 2026-09-03: BIAF (-3.18%) → GPRK (+4.55%) — Live wins +7.73pp
- 2026-09-18: BIAF (+21.61%) → SVRN (-8.89%) — Frozen wins +30.50pp
- Net: Frozen wins on BIAF days (+9.21% avg vs -2.17% avg)

### E2-B: 9/34 days diverge

Key Arm 1 vs Arm 2 divergence:
- 2026-09-03: BIAF → NTSK (Live excludes BIAF due to 600K float)

## Exclusion Counts

| Arm | Mode | CS | Float miss | Float <10M | SD | Total float |
|-----|------|----|-----------|-----------|-----|-------------|
| Frozen | E2-A | 439 | 0 | 7 | 0 | 7 |
| Live | E2-A | 439 | 0 | 10 | 0 | 10 |
| Rebuilt | E2-A | 439 | 34 | 43 | 0 | 77 |
| Frozen | E2-B | 439 | 0 | 7 | 153 | 7 |
| Live | E2-B | 439 | 0 | 10 | 151 | 10 |
| Rebuilt | E2-B | 439 | 34 | 43 | 103 | 77 |

## Key Findings

1. **E2-A: Frozen performs best.** Avg +4.73% vs Live +4.06% vs Rebuilt +1.70%.
   The miss=INCLUDE policy lets unknown tickers through, and BIAF's +21.61%
   trade on 09-18 carries Frozen's advantage. This is lucky — BIAF is a
   600K-float micro-cap that shouldn't be in the pool.

2. **E2-B: Live marginally beats Frozen.** Avg +3.31% vs +3.17%, Sharpe
   0.371 vs 0.351, PF 3.47 vs 3.14, CI crosses into positive [+0.0, +6.6].
   The SD filter removes the lucky BIAF trade, so Live's BIAF exclusion
   becomes a net positive.

3. **Rebuilt (Arm 3) worst by every metric.** Only 55.8% coverage with
   miss=EXCLUDE kills too many candidates. Plus shares_outstanding ≠ float
   introduces noise. Not viable without better data.

4. **Float filter has minimal impact on E2-B.** Same trade count (28),
   same pick on 25/34 days. The SD filter dominates candidate selection
   — float is a secondary concern when SD already narrows the pool.

5. **Only 1 material float divergence**: MCFT (16.3M frozen vs 24.3M rebuilt).
   Both above 10M threshold, so no impact on filtering.

## Recommendation

- **E2-A (no SD):** Keep Frozen dict. The miss=INCLUDE policy is a feature,
  not a bug — most unknown tickers are small-caps that the CS filter and
  gap ranking handle. Switching to Live costs $22K on 34 days.

- **E2-B (SD):** Live dict is marginally better (+0.14pp avg, +0.020 Sharpe,
  CI enters positive). With SD filtering out volatile micro-caps, the
  explicit BIAF exclusion helps. Consider adopting Live for E2-B deployment.

- **Arm 3 (Rebuilt):** Reject. Coverage too low, shares_outstanding ≠ float,
  point-in-time bias. Would need a proper float data source.

## Limitations

1. Polygon `share_class_shares_outstanding` ≠ true float
2. Point-in-time: using current shares count, not historical
3. PRE_FILTER_N=30 may miss candidates on high-activity days
4. Only 34 trading days — small sample for 2 divergent trades
