# Agent 1 — Edge Deterioration Investigation (RESULTS)

## Overview

**Date:** 2026-09-19
**Period:** Jan 2 – Sep 11, 2026 (174 trading days)
**Baseline:** E2 benchmark — 166 trades, +13.52% avg PnL, 77% WR, Sharpe 0.542, $+2,244K
**Config:** N=1, CS+SD, SOFT float, exit 17:00, SL=-50% (v1.2)

**Problem:** Performance deteriorates sharply in Q3 (Jul-Sep): avg PnL drops -42%, Sharpe 0.593→0.420.

---

## E2 Baseline (Full Pipeline, 166 trades)

| Metric       | Value       |
|--------------|-------------|
| Trades       | 166         |
| Avg PnL      | +13.52%     |
| Win Rate     | 77%         |
| Sharpe       | 0.542       |
| Total $ (NC) | $+2,243,565 |
| Winners      | 128 (+$2,770,909) |
| Losers       | 38 (-$506,448)    |
| Big losers (>-10%) | 86% of total losses |

---

## Monthly Diagnostic

| Month | n  | Avg PnL  | WR  | Sharpe | Total $  |
|-------|----|----------|-----|--------|----------|
| Jan   | 20 | +15.57%  | 90% | 0.907  | $+311K   |
| Feb   | 19 | +13.41%  | 74% | 0.853  | $+255K   |
| Mar   | 22 | +13.73%  | 73% | 0.511  | $+302K   |
| Apr   | 19 | +15.44%  | 84% | 0.881  | $+293K   |
| May   | 20 | +15.84%  | 70% | 0.361  | $+317K   |
| Jun   | 20 | +17.76%  | 90% | 0.674  | $+355K   |
| Jul   | 21 | +10.72%  | 62% | 0.379  | $+225K   |
| Aug   | 18 | +9.51%   | 83% | 0.675  | $+171K   |
| Sep   | 7  | +1.99%   | 57% | 0.219  | $+14K    |

**Not a smooth decline — sharp drop in Jul, worse in Sep.** No negative months, but Sep is near zero.

---

## Agent A — Regime Detector

### Regime Performance

| Regime | n   | Avg PnL | Sharpe | Total $     |
|--------|-----|---------|--------|-------------|
| Bull   | 26  | +13.03% | 0.347  | $+338,848   |
| Flat   | 135 | +13.60% | 0.622  | $+1,836,573 |
| Bear   | 5   | +17.81% | 0.637  | $+89,040    |

### H1 vs H2

| Period  | n   | Avg PnL  | Med PnL  | WR  | Sharpe | Total $     |
|---------|-----|----------|----------|-----|--------|-------------|
| Jan-Jun | 120 | +15.45%  | +13.16%  | 80% | 0.593  | $+1,854,284 |
| Jul-Sep | 46  | +8.92%   | +3.77%   | 70% | 0.420  | $+410,177   |

**PnL degradation: -42.3%.** VIX dropped (19.4→16.2), SPY barely changed (+1.5%→+1.2%).

### Best Regime Filters (IS=Jan-Jun, OOS=Jul-Sep)

| Filter         | IS n | IS Sharpe | OOS n | OOS Avg   | OOS Skip% |
|----------------|------|-----------|-------|-----------|-----------|
| (no filter)    | 120  | 0.593     | 46    | +8.92%    | 0%        |
| SPY_20d < +3%  | 82   | 0.686     | 36    | +9.28%    | 25%       |
| Sector_5d < -1%| 33   | 0.983     | 5     | +5.41%    | 86%       |
| VIX > 18       | 64   | 0.468     | 6     | +0.16%    | 86%       |
| SPY < 50 SMA   | 36   | 0.660     | 8     | +2.31%    | 82%       |

**VERDICT: No single regime filter explains deterioration.** SPY_20d < +3% is most practical (keeps 75% of days, +9.28% OOS) but doesn't meaningfully improve Q3.

---

## Agent B — Time of Day Decay

### Exit Time Sweep

| Exit   | H1 Avg  | H1 Sharpe | H2 Avg  | H2 Sharpe |
|--------|---------|-----------|---------|-----------|
| 10:00  | +4.18%  | 0.568     | +3.29%  | 0.589     |
| 10:30  | +8.99%  | 0.806     | +4.95%  | 0.547     |
| 11:00  | +10.13% | 0.799     | +6.89%  | **0.700** |
| 12:00  | +12.72% | **0.843** | +7.15%  | 0.539     |
| 13:00  | +13.62% | 0.737     | +6.43%  | 0.438     |
| 14:00  | +12.30% | 0.625     | +8.74%  | 0.494     |
| 15:00  | +11.98% | 0.530     | +8.38%  | 0.455     |
| 16:00  | +12.94% | 0.551     | +8.25%  | 0.403     |
| 17:00  | +15.45% | 0.593     | +8.92%  | 0.420     |
| T+1    | +6.52%  | 0.049     | +11.52% | 0.479     |
| T+2    | +17.53% | 0.344     | +16.54% | 0.614     |

**Optimal exit STABLE at T+2** for both periods (Δ only -0.99%).
**Best risk-adjusted in H2: 11:00** (Sharpe 0.700).
**Fade exhaustion 1h earlier in H2:** 13:00 vs 14:00.

### Monthly Heatmap (Avg PnL)

| Exit   | Jan    | Feb    | Mar    | Apr    | May    | Jun    | Jul    | Aug    | Sep    |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| 11:00  | +10.3% | +9.8%  | +10.6% | +10.9% | +8.9%  | +10.3% | +9.7%  | +5.8%  | +1.2%  |
| 17:00  | +15.6% | +13.4% | +14.5% | +15.4% | +15.8% | +17.9% | +10.7% | +9.5%  | +2.0%  |
| T+2    | +13.0% | +22.4% | -1.1%  | +22.1% | +28.7% | +22.4% | +24.3% | +11.2% | +7.1%  |

**Sep is weak at every exit time.** The edge itself is weaker, not the exit timing.

**VERDICT: NO SHIFT in optimal exit.** Not causal.

---

## Agent C — Gap Size Regime Shift

### Gap Tier Performance by Period

| Tier       | Q1 n | Q1 Avg  | Q1 Sharpe | Q2 n | Q2 Avg  | Q2 Sharpe | Q3 n | Q3 Avg  | Q3 Sharpe |
|------------|------|---------|-----------|------|---------|-----------|------|---------|-----------|
| T4 (5-8%)  | 0    | —       | —         | 0    | —       | —         | 3    | +2.84%  | 0.729     |
| T5 (8-15%) | 4    | -2.24%  | -0.189    | 2    | +3.29%  | 0.435     | 7    | +0.55%  | 0.205     |
| T6 (15-30%)| 14   | +6.58%  | 0.562     | 11   | +8.78%  | 0.931     | 17   | +5.24%  | 0.301     |
| T7 (30%+)  | 43   | +18.26% | 0.827     | 46   | +18.74% | 0.545     | 19   | +16.24% | 0.593     |
| **TOTAL**  | 61   | +14.23% | 0.693     | 59   | +16.36% | 0.529     | 46   | +8.92%  | 0.420     |

### Q3 Deterioration by Tier

| Tier        | Q1+Q2 Avg | Q1+Q2 n | Q3 Avg | Q3 n | Delta  | Verdict    |
|-------------|-----------|---------|--------|------|--------|------------|
| T4 (5-8%)   | +0.00%    | 0       | +2.84% | 3    | +0.00% | STABLE     |
| T5 (8-15%)  | -0.40%    | 6       | +0.55% | 7    | +0.95% | STABLE     |
| T6 (15-30%) | +7.55%    | 25      | +5.24% | 17   | -2.30% | **WEAKENED** |
| T7 (30%+)   | +18.51%   | 89      | +16.24%| 19   | -2.26% | **WEAKENED** |

**T7 dominates all PnL** (89 trades in Q1+Q2 at +18.51%), but **T7 count collapsed in Q3** (89→19, -79%).
T6 weakened slightly (-2.30%) but shifted from 25 to 17 trades.

### Opportunity Set (All Gap-Ups ≥ 5%)

| Period | Avg Gap | Gap-Ups/Day | T4%   | T5%   | T6%  | T7%  |
|--------|---------|-------------|-------|-------|------|------|
| Q1     | 151.7%  | 81.8        | 60.6% | 28.3% | 7.7% | 3.4% |
| Q2     | 103.4%  | 107.0       | 61.2% | 29.3% | 6.9% | 2.7% |
| Q3     | 10.0%   | 107.0       | 59.2% | 30.3% | 8.3% | 2.1% |

**Avg gap size collapsed: 151.7% (Q1) → 10.0% (Q3).** The massive outlier gaps that drove T7 are disappearing. Tier distribution is relatively stable, but with fewer extreme gaps, the N=1 pick is increasingly a T5/T6 stock instead of T7.

### Q3 Worst Trades

| Date       | Ticker | Gap%  | Tier | PnL%   | $ PnL   |
|------------|--------|-------|------|--------|---------|
| 2026-07-27 | GOSS   | 27.5% | T6   | -50.68%| -$50,680|
| 2026-07-23 | LGCL   | 67.1% | T7   | -23.24%| -$23,239|
| 2026-07-17 | SDOT   | 54.0% | T7   | -15.98%| -$15,982|
| 2026-09-09 | ODD    | 34.4% | T7   | -10.93%| -$10,928|

GOSS -50.68% is the single biggest drag on Q3. That's the SL=-50% cap in action.

**VERDICT: T7 count collapse is the primary driver.** The strategy depends on 30%+ gap-ups, and those are 79% rarer in Q3. When forced to trade smaller gaps, PnL drops.

---

## Agent D — New Universe Discovery

### Universe Comparison

| Metric           | Value     |
|------------------|-----------|
| Original _FLOAT  | 217       |
| NEW_FLOAT        | 217       |
| Overlap          | 17 (7.8%) |
| Jaccard          | 0.041     |
| Turnover         | 92.2%     |

### Performance by Universe Group

| Group     | n  | Avg PnL  | WR  | Sharpe | CI Low  | CI High  |
|-----------|----|----------|-----|--------|---------|----------|
| BOTH      | 2  | -0.57%   | 50% | -0.539 | -2.03%  | +0.89%   |
| ORIG_ONLY | 3  | +0.35%   | 67% | 0.302  | -0.96%  | +1.66%   |
| **NEW_ONLY** | **73** | **+19.11%** | **85%** | **0.632** | +12.18% | +26.04% |
| NEITHER   | 88 | +9.64%   | 72% | 0.501  | +5.63%  | +13.66%  |

**NEW_ONLY massively outperforms: +19.11% avg, 85% WR, Sharpe 0.632.** These are tickers in the new optimal universe but NOT in the original _FLOAT. The strategy is already picking them (because SOFT float = dict miss → INCLUDE) but they would be excluded under HARD float.

### Monthly Breakdown — NEW_ONLY vs NEITHER

| Month | NEW_ONLY n | NEW_ONLY Avg | NEITHER n | NEITHER Avg |
|-------|------------|--------------|-----------|-------------|
| Jan   | 8          | +20.56%      | 12        | +12.25%     |
| Feb   | 11         | +18.45%      | 8         | +6.48%      |
| Mar   | 11         | +13.05%      | 10        | +15.75%     |
| Apr   | 4          | +21.01%      | 14        | +15.02%     |
| May   | 10         | +23.09%      | 10        | +8.58%      |
| Jun   | 13         | +16.15%      | 7         | +20.75%     |
| Jul   | 9          | **+27.81%**  | 10        | **-2.50%**  |
| Aug   | 7          | +15.55%      | 10        | +6.22%      |
| Sep   | 0          | N/A          | 7         | +1.99%      |

**Jul is the critical month.** NEW_ONLY: +27.81%, NEITHER: -2.50%. The edge in Jul comes entirely from new-universe tickers. Sep had zero NEW_ONLY trades — all 7 were NEITHER tickers at +1.99%.

**VERDICT: Universe is massively stale (92% turnover).** The tickers driving profits are NOT in the original _FLOAT dict. SOFT float mode is saving us (dict miss = INCLUDE), but a HARD float would destroy the strategy.

---

## Agent E — Entry Pattern Shift

### Period Comparison

| Period | n  | LH  | FB  | FB%  | LH Mag  | LH PnL  | FB PnL  | Avg PnL | Sharpe |
|--------|----|-----|-----|------|---------|---------|---------|---------|--------|
| Q1     | 61 | 31  | 30  | 49%  | +1.79%  | +16.97% | +12.00% | +14.53% | 0.705  |
| Q2     | 59 | 37  | 22  | 37%  | +2.75%  | +12.72% | +22.62% | +16.41% | 0.531  |
| Q3     | 46 | 29  | 17  | 37%  | +1.08%  | +11.20% | +5.02%  | +8.92%  | 0.420  |

Both entry types degraded uniformly. LH magnitude collapsed (Q2 +2.75% → Q3 +1.08%).

### Window Sweep (Q3 only)

| Window | Avg PnL | Sharpe | vs Current |
|--------|---------|--------|------------|
| w=2    | +9.97%  | 0.474  | **+1.06%** |
| w=3    | +8.92%  | 0.420  | (current)  |
| w=4    | +8.54%  | 0.409  | -0.38%     |
| w=5    | +8.22%  | 0.385  | -0.70%     |
| FB_ONLY| +8.03%  | 0.381  | -0.89%     |

w=2 adds +1.06% — marginal improvement.

**VERDICT: Entry mechanics NOT causal.** Both types degraded equally. The weaker edge is upstream.

---

## Agent F — Alternative Signals

### Full Period (166 trades)

| Ranking              | Avg PnL | WR  | Sharpe | Total $     | vs Baseline  |
|----------------------|---------|-----|--------|-------------|--------------|
| **Baseline (gap_pct)** | **+13.52%** | **77%** | **0.542** | **$+2,243,565** | —            |
| R1: Vol Spike        | +8.55%  | 80% | 0.127  | $+1,418,535 | -$825,030    |
| R2: Idio Gap         | +13.37% | 77% | 0.536  | $+2,219,727 | -$23,838     |
| R3: PM Retrace       | +6.94%  | 78% | 0.107  | $+1,144,290 | -$1,099,275  |
| R4: Price Ext        | +3.45%  | 75% | 0.053  | $+572,702   | -$1,670,863  |
| R5: Combined         | +6.27%  | 77% | 0.093  | $+1,041,552 | -$1,202,013  |

**No alternative beats gap_pct over the full period.** R2 (idiosyncratic gap) is nearly identical (-$24K).

### Jul-Sep Only (46 trades)

| Ranking              | Avg PnL | WR  | Sharpe | Total $   | vs Baseline |
|----------------------|---------|-----|--------|-----------|-------------|
| **Baseline (gap_pct)** | **+8.92%** | **70%** | **0.420** | **$+410,177** | —           |
| R1: Vol Spike        | +9.05%  | 74% | 0.440  | $+416,087 | +$5,910     |
| R2: Idio Gap         | +8.98%  | 70% | 0.423  | $+412,935 | +$2,758     |
| R3: PM Retrace       | +8.69%  | 73% | **0.643** | $+391,097 | -$19,080    |
| R4: Price Ext        | **+10.56%** | **78%** | 0.527  | **$+485,963** | **+$75,786**|
| R5: Combined         | +9.49%  | 76% | 0.461  | $+436,433 | +$26,256    |

**R4 (price extension) wins in Jul-Sep:** +10.56% vs +8.92% baseline (+$76K). Picks the stock most extended above its 20-day high — captures overextension better than raw gap size.

R3 (PM retrace) has highest Sharpe (0.643) but lower dollar amount.

### Overlap Analysis

| Ranking       | Same Pick | Diff Pick | Diff Better | Diff Worse | Avg Gain When Diff |
|---------------|-----------|-----------|-------------|------------|-------------------|
| R1: Vol Spike | 105       | 61        | 32          | 29         | -13.53%           |
| R2: Idio Gap  | 164       | 2         | 1           | 1          | -11.92%           |
| R3: PM Retrace| 60        | 105       | 43          | 62         | -10.46%           |
| R4: Price Ext | 85        | 81        | 26          | 55         | -20.63%           |

R2 picks the same stock 99% of the time — it's gap_pct with a small sector adjustment. R4 diverges 49% of the time but when it diverges, baseline wins more often (-20.63% avg on divergent days). The Jul-Sep improvement from R4 is real but comes from a minority of picks.

**VERDICT: No alternative signal dominates.** Gap_pct baseline is optimal over full period. R4 (price extension) shows a modest Jul-Sep edge (+$76K) but divergent picks lose on average — not robust enough to switch.

---

## SYNTHESIS — Root Cause & Action Items

### What Caused the Q3 Deterioration

| Rank | Factor | Evidence | Impact |
|------|--------|----------|--------|
| **1** | **T7 gap count collapse** | 89 trades Q1+Q2 → 19 in Q3 (-79%). Avg gap: 151.7% → 10.0% | PRIMARY — fewer extreme gaps = fewer high-PnL trades |
| **2** | **VIX compression** | 19.4 → 16.2 | CONTRIBUTING — lower vol = smaller moves = smaller fades |
| 3 | Universe staleness | 92% turnover, but SOFT float saves us | NOT DIRECT — would only matter under HARD float |
| 4 | Entry pattern | Both types declined equally | NOT CAUSAL |
| 5 | Exit timing | Stable at T+2 | NOT CAUSAL |
| 6 | Alternative signals | No signal beats gap_pct | NOT CAUSAL |

### The Bottom Line

**The edge didn't break — the opportunity set shrank.** Q1-Q2 had abundant 30%+ gap-ups (89 T7 trades at +18.5% avg). Q3 has far fewer extreme gaps (19 T7 trades). When the strategy is forced to trade smaller gaps (T5/T6), PnL is lower (+0.55% T5, +5.24% T6 vs +16.24% T7). This is structural — the strategy needs volatility to produce extreme gap-ups.

### Actionable Next Steps

| Priority | Action | Expected Impact | Risk |
|----------|--------|----------------|------|
| **NONE** | **Accept variance** | Q3 is still profitable (+8.92%) | None — this is the correct response |
| LOW | Switch to w=2 peak window | +1.06% on Q3 data | Marginal, may not persist |
| MONITOR | Track T7 count monthly | Early warning if opportunity set continues shrinking | None |
| DO NOT | Add regime filter | Kills 25-86% of trading days for marginal Sharpe gain | Massive opportunity cost |
| DO NOT | Switch to alt signals | Gap_pct dominates; R4 wins Jul-Sep but loses on divergent picks | Worse full-period performance |
| DO NOT | Switch to HARD float | Would exclude the +19.11% avg NEW_ONLY trades | Catastrophic |

---

## What Was Established Before This Investigation

- **SL=-10% destroys value** — kills 21 recoveries, costs -$382K
- **SL=-50% is a free upgrade** — saves 3 disasters, +$53K, Sharpe 0.435→0.510
- **No skip rule helps** — every drop bucket and gap-consumed level makes money
- **Config is locked** at v1.2: N=1, CS+SD, SOFT float, exit 17:00, SL=-50%
