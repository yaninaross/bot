# Agent 1 — SD Ablation: 3-Way Comparison

**Date:** 2026-09-20
**Period:** 2026-08-01 → 2026-09-20 (34 trading days)
**Scan:** 09:31 ET | Exit: 17:00 ET | TOP_N=1 | GAP >= 3% | CS only | SOFT float

## Variants

| Variant | SD Required | Entry Method | Entry Window |
|---------|-------------|--------------|--------------|
| A: Baseline | No | local_high bars 1-12, fb 15 | 09:31-09:45 ET |
| B: SD+Direct | Yes (bar 59) | bar 59 close directly | 10:29 ET |
| C: SD+LH | Yes (bar 59) | local_high bars 60-72, fb 75 | 10:30-10:45 ET |

## 3-Way Comparison

| Metric | A: Baseline | B: SD+Direct | C: SD+LH |
|--------|-------------|--------------|-----------|
| Trades | 34 | 29 | 28 |
| Avg PnL | +4.73% | +3.18% | +3.33% |
| Median PnL | +4.24% | +2.29% | +1.99% |
| Trimmed Mean | +6.70% | +2.01% | +2.28% |
| Win Rate | 71% | 66% | 61% |
| Sharpe | 0.171 | 0.335 | 0.371 |
| Profit Factor | 1.89 | 2.78 | 3.34 |
| $ NC | $+160,886 | $+92,159 | $+93,270 |
| Max DD | -166.8% | -14.8% | -12.4% |
| 95% CI | [-4.6, +14.0] | [-0.3, +6.6] | [+0.0, +6.7] |

## Entry Type Breakdown

| Variant | local_high | fallback | direct |
|---------|-----------|----------|--------|
| A: Baseline | n=24 avg=+11.5% Sh=0.67 | n=10 avg=-11.4% Sh=-0.28 | — |
| B: SD+Direct | — | — | n=29 avg=+3.2% Sh=0.34 |
| C: SD+LH | n=24 avg=+4.1% Sh=0.43 | n=4 avg=-1.3% Sh=-0.74 | — |

## PnL by Horizon

### A: Baseline (no SD)
| Horizon | n | Avg | Median | WR | Sharpe |
|---------|---|-----|--------|----|----|
| EOD | 34 | +4.73% | +4.24% | 71% | 0.171 |
| T+1 | 33 | +4.00% | +3.05% | 55% | 0.092 |
| T+2 | 32 | +2.21% | +3.86% | 66% | 0.041 |

### B: SD+Direct @10:30
| Horizon | n | Avg | Median | WR | Sharpe |
|---------|---|-----|--------|----|----|
| EOD | 29 | +3.18% | +2.29% | 66% | 0.335 |
| T+1 | 29 | +3.73% | +0.53% | 55% | 0.233 |
| T+2 | 28 | +5.90% | +3.99% | 64% | 0.373 |

### C: SD+LH post-10:30
| Horizon | n | Avg | Median | WR | Sharpe |
|---------|---|-----|--------|----|----|
| EOD | 28 | +3.33% | +1.99% | 61% | 0.371 |
| T+1 | 28 | +4.00% | +1.33% | 54% | 0.252 |
| T+2 | 27 | +6.10% | +6.38% | 67% | 0.378 |

## Pattern Breakdown (Baseline, no SD)

| Pattern | n | Avg PnL | WR | Sharpe |
|---------|---|---------|----|----|
| sustained_drop | 13 | +9.98% | 77% | 0.571 |
| drop_then_recover | 12 | +1.05% | 67% | 0.024 |
| mild_drop | 2 | +13.54% | 100% | 2.210 |
| flat_or_bounce | 6 | -2.66% | 50% | -0.412 |
| no_data | 1 | +7.42% | 100% | — |

## Key Findings

1. **SD filter fixes risk, not return**: Baseline has highest avg PnL (+4.73%) but
   catastrophic Max DD (-166.8%) from fallback entries on non-SD patterns.
   SD variants cut DD to -12% to -15% at the cost of ~1.5pp avg PnL.

2. **Sharpe doubles with SD**: 0.171 → 0.335/0.371. The CI narrows dramatically —
   baseline CI includes negative territory, SD variants are near-positive.

3. **Late entry costs ~7pp per LH trade**: Baseline LH avg +11.5% vs SD+LH avg +4.1%.
   You're shorting after the stock has already fallen 3%+ and held — the easy part
   of the move is captured before you enter.

4. **SD+LH slightly beats SD+Direct on risk**: Sharpe 0.371 vs 0.335, PF 3.34 vs 2.78,
   DD -12.4% vs -14.8%. The local_high confirmation post-10:30 adds marginal value
   over raw bar-59 entry.

5. **T+2 edge survives for SD variants**: Both SD variants improve from EOD to T+2
   (+3.2%→+5.9% for Direct, +3.3%→+6.1% for LH). Baseline *degrades* (+4.7%→+2.2%).
   SD selects stocks with genuine continued downside.

6. **Fallback entries remain toxic in SD+LH**: n=4, avg -1.29%, 0% WR.
   Consider removing fallback for the SD variant entirely (LH-only).

7. **Baseline fallback entries are the blowup**: n=10, avg -11.4%, Sharpe -0.28.
   These 10 trades account for essentially all the drawdown.

## Recommendation

**SD+LH (C) is the best risk-adjusted variant** — highest Sharpe, lowest DD, positive CI.
But the entry is 1 hour later and avg PnL is ~1.5pp lower than baseline.

Next tests to consider:
- Remove fallback from SD+LH (LH-only, no bar-75 fallback)
- Test T+2 hold for SD variants (the edge appears to grow)
- Widen to TOP_N=2 or 3 with SD filter (SD trades are higher quality)
