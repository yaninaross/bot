# E2-A Entry & Trade Mechanics Investigation Results

**Date:** 2026-10-04
**Period:** 2026-01-02 → 2026-10-06 | Break: 2026-08-01
**Config:** TOP_N=5, CS only, SOFT float, AH fresh, erosion, exit 17:00 ET, $100K/day NC
**Runtime:** 19.7 min (Colab) | 191 trading days | 10,544/10,901 bar pairs

---

## Baseline

| Metric | Pre-Aug | Post-Aug | Delta |
|--------|---------|----------|-------|
| Trades | 671 | 153 | — |
| Avg PnL | +4.46% | +2.18% | -2.28pp |
| Sharpe | 0.196 | 0.178 | — |
| Win Rate | 62% | 57% | -5pp |

---

## Exp 1 — Local High Detection Audit

**Purpose:** Audit fire type (clean/late/fallback) and entry bar distributions to understand where the post-Aug degradation originates.

### Fire Type Distribution

| Type | Pre N | Pre Avg | Post N | Post Avg | Delta |
|------|-------|---------|--------|----------|-------|
| clean | 289 | +4.72% | 62 | +0.99% | -3.73pp |
| late | 150 | +1.20% | 36 | +7.78% | +6.58pp |
| fallback | 232 | +6.23% | 55 | -0.16% | -6.38pp |

### Entry Bar Distribution (local_high fires only)

| Peak Bar | Pre N | Pre Avg | Post N | Post Avg |
|----------|-------|---------|--------|----------|
| bar 3 | 53 | +1.71% | 12 | +1.42% |
| bar 4 | 53 | +0.27% | 11 | +1.62% |
| bar 5 | 53 | +5.30% | 15 | +1.88% |
| bar 6 | 75 | +8.11% | 12 | -2.28% |
| bar 7 | 55 | +6.74% | 12 | +2.12% |
| bar 8 | 44 | +1.28% | 13 | +6.09% |
| bar 9 | 35 | +2.85% | 6 | +6.23% |
| bar 10 | 23 | +2.59% | 9 | +10.65% |
| bar 11 | 48 | -0.73% | 8 | +8.47% |

### Subsequent PnL from Entry Point

| Type@Offset | Pre N | Pre Avg | Post N | Post Avg |
|-------------|-------|---------|--------|----------|
| clean@+5 | 285 | +0.17% | 61 | -0.23% |
| clean@+10 | 285 | +0.19% | 59 | +0.31% |
| clean@+15 | 283 | +0.93% | 59 | +0.02% |
| late@+5 | 149 | +0.36% | 36 | +0.58% |
| late@+10 | 148 | -0.44% | 36 | +1.41% |
| late@+15 | 148 | -0.47% | 36 | +2.40% |
| fallback@+5 | 231 | +0.61% | 54 | +0.16% |
| fallback@+10 | 229 | +1.18% | 54 | -0.06% |
| fallback@+15 | 228 | +1.74% | 53 | -0.52% |

### Entry vs Open Price by Fire Type

| Type | Pre | Post |
|------|-----|------|
| clean | +2.27% | +0.46% |
| late | +1.46% | +1.33% |
| fallback | -2.01% | -1.47% |

**Key Finding:** Fallback entries are the primary source of degradation (-6.38pp). Late fires actually *improved* post-Aug (+6.58pp). Clean fires degraded moderately (-3.73pp). The fallback mechanism is the single biggest liability in the compressed regime.

---

## Exp 2 — Opening Range Breakout (ORB) Entry

**Purpose:** Test ORB (break below N-bar range low) as alternative entry vs local_high + fallback.

| ORB Range | Pre Avg | Pre N | Post Avg | Post N | Delta | vs Baseline Delta |
|-----------|---------|-------|----------|--------|-------|-------------------|
| range=1 | +3.94% | 468 | +2.18% | 106 | -1.76pp | +0.52pp |
| range=2 | +4.30% | 427 | +1.71% | 100 | -2.58pp | -0.30pp |
| range=3 | +4.22% | 398 | +1.51% | 94 | -2.70pp | -0.42pp |
| range=5 | +4.38% | 353 | +1.04% | 85 | -3.33pp | -1.05pp |

**Key Finding:** ORB range=1 narrows the gap by +0.52pp while maintaining post-Aug avg at +2.18%. Higher ranges degrade. ORB-1 is a viable alternative entry but the improvement is modest.

---

## Exp 3 — VWAP-Relative Entry

**Purpose:** Test VWAP cross (price breaks below VWAP) as entry signal.

| Entry Method | Pre Avg | Pre N | Post Avg | Post N | Delta |
|--------------|---------|-------|----------|--------|-------|
| local_high | +3.52% | 439 | +3.48% | 98 | -0.04pp |
| fallback | +6.23% | 232 | -0.16% | 55 | -6.38pp |
| vwap_cross | +4.47% | 583 | +2.45% | 132 | -2.03pp |

**Key Finding:** VWAP cross narrows the gap by +0.25pp and improves post-Aug to +2.45%. More importantly, the comparison confirms local_high *alone* (no fallback) has near-zero degradation (-0.04pp). The fallback is the entire problem.

---

## Exp 4 — Volume-Weighted Entry Quality

**Purpose:** Test whether entry bar volume quality predicts subsequent PnL.

Entry bar vol / first-5-bar vol quartiles: Q25=0.049, Q50=0.089, Q75=0.153

| Vol Bucket | Pre N | Pre Avg | Post N | Post Avg |
|------------|-------|---------|--------|----------|
| low | 162 | +1.61% | 47 | -0.80% |
| mid | 345 | +4.15% | 65 | +1.39% |
| high | 164 | +7.92% | 41 | +6.83% |

Volume ratio by entry type:
- local_high: Pre mean=0.173 | Post mean=0.182
- fallback: Pre mean=0.165 | Post mean=0.125

**Key Finding:** High-volume entries massively outperform: +7.92% pre / +6.83% post with only -1.09pp degradation. Low-volume entries go negative post-Aug (-0.80%). Volume at entry is a strong quality signal. Fallback entries have lower volume ratios post-Aug (0.125 vs 0.165).

---

## Exp 5 — Fallback Timing Grid

**Purpose:** Test fallback bar timing from FB=3 to FB=19, plus no-fallback (LH only).

| Config | Pre N | Pre Avg | Post N | Post Avg | Delta | FB Pre N | FB Pre Avg | FB Post N | FB Post Avg |
|--------|-------|---------|--------|----------|-------|----------|------------|-----------|-------------|
| FB=3 | 677 | +4.68% | 156 | +2.44% | -2.24pp | 238 | +6.82% | 58 | +0.67% |
| FB=5 | 677 | +4.25% | 156 | +2.34% | -1.91pp | 238 | +5.61% | 58 | +0.40% |
| FB=7 | 677 | +4.16% | 156 | +2.13% | -2.03pp | 238 | +5.35% | 58 | -0.15% |
| FB=9 | 674 | +4.11% | 154 | +2.14% | -1.97pp | 235 | +5.21% | 56 | -0.22% |
| FB=11 | 673 | +4.22% | 153 | +2.28% | -1.94pp | 234 | +5.54% | 55 | +0.15% |
| FB=13 | 672 | +4.40% | 153 | +2.13% | -2.28pp | 233 | +6.06% | 55 | -0.30% |
| FB=15 | 671 | +4.46% | 153 | +2.18% | -2.28pp | 232 | +6.23% | 55 | -0.16% |
| FB=19 | 670 | +4.32% | 152 | +2.06% | -2.26pp | 231 | +5.85% | 54 | -0.53% |
| **LH only** | — | **+3.52%** | — | **+3.48%** | **-0.04pp** | — | — | — | — |

**Key Finding:** No-fallback (LH only) has virtually zero degradation (-0.04pp) but sacrifices ~28% of trades. The best fallback timing is FB=5 (-1.91pp delta) but all fallbacks degrade post-Aug. This confirms: **removing fallback entirely is the single most effective regime-adaptation lever.**

---

## Exp 6 — Gap Freshness at Entry vs at Scan

**Purpose:** Check if gap remaining at entry time (vs scan time) predicts outcomes.

Gap remaining at entry (entry_gap / scan_gap):
- Pre: mean=1.103, med=1.006, <0.30=30/671, <0.50=60/671
- Post: mean=0.974, med=0.993, <0.30=18/153, <0.50=29/153

| Gate | Pre N | Pre Avg | Post N | Post Avg | Delta |
|------|-------|---------|--------|----------|-------|
| >=30% remaining | 641 | +4.53% | 135 | +2.33% | -2.21pp |
| <30% remaining | 30 | +2.80% | 18 | +1.05% | -1.75pp |
| >=50% remaining | 611 | +4.21% | 124 | +2.43% | -1.78pp |
| <50% remaining | 60 | +6.98% | 29 | +1.08% | -5.90pp |
| >=70% remaining | 545 | +4.14% | 115 | +2.69% | -1.45pp |
| <70% remaining | 126 | +5.83% | 38 | +0.63% | -5.19pp |

Correlations:
- Pre gap_remaining→PnL: r=-0.025 (p=0.529)
- Post gap_remaining→PnL: r=+0.118 (p=0.158)

**Key Finding:** The >=70% gate narrows delta to -1.45pp and lifts post-Aug to +2.69%. Trades where gap has eroded significantly by entry time (<50% remaining) collapse -5.90pp post-Aug. Gap freshness at entry is a useful quality filter.

---

## Exp 7 — Intraday Volatility Entry Gate

**Purpose:** Test minimum first-5-bar volatility as entry gate.

| Vol Gate | Pre N | Pre Avg | Pre WR | Post N | Post Avg | Post WR | Delta |
|----------|-------|---------|--------|--------|----------|---------|-------|
| >=0.50% | 668 | +4.48% | 62% | 150 | +2.22% | 57% | -2.26pp |
| >=0.75% | 665 | +4.50% | 62% | 149 | +2.24% | 58% | -2.26pp |
| >=1.00% | 663 | +4.48% | 62% | 149 | +2.24% | 58% | -2.24pp |
| >=1.25% | 661 | +4.49% | 62% | 148 | +2.23% | 57% | -2.26pp |
| >=1.50% | 657 | +4.54% | 62% | 147 | +2.27% | 58% | -2.27pp |
| >=1.75% | 656 | +4.55% | 62% | 145 | +2.07% | 57% | -2.47pp |
| >=2.00% | 652 | +4.56% | 62% | 143 | +2.12% | 58% | -2.44pp |

First-5-bar vol distribution:
- Pre: mean=10.57%, med=8.44%
- Post: mean=8.02%, med=6.63%

**Key Finding:** Intraday vol gate has negligible effect — nearly all trades pass even aggressive gates. The post-Aug vol compression (8.02% vs 10.57%) is real but the gate doesn't separate good from bad trades. Not actionable.

---

## Exp 8 — Exit Optimization

**Purpose:** Test earlier fixed exits and trailing stops from MFE.

### 8a — Fixed Exit Time Grid

| Exit Time | Pre N | Pre Avg | Pre WR | Post N | Post Avg | Post WR | Delta |
|-----------|-------|---------|--------|--------|----------|---------|-------|
| 10:00 | 621 | +0.94% | 58% | 140 | +0.09% | 51% | -0.84pp |
| 10:30 | 646 | +1.27% | 58% | 147 | +0.70% | 59% | -0.57pp |
| 11:00 | 653 | +1.48% | 58% | 148 | +1.60% | 63% | +0.12pp |
| 11:30 | 659 | +1.32% | 58% | 150 | +1.93% | 61% | +0.62pp |
| 12:00 | 660 | +2.05% | 59% | 150 | +1.89% | 61% | -0.15pp |
| 13:00 | 663 | +2.82% | 59% | 151 | +1.51% | 60% | -1.31pp |
| 14:00 | 668 | +3.08% | 59% | 151 | +1.51% | 55% | -1.57pp |
| 15:00 | 670 | +3.20% | 60% | 153 | +1.61% | 55% | -1.59pp |
| 16:00 | 671 | +3.58% | 61% | 153 | +2.18% | 56% | -1.40pp |
| 17:00 | 671 | +4.46% | 62% | 153 | +2.18% | 57% | -2.28pp |

### 8b — Trailing Stop from MFE

| Retrace | Pre Avg | Pre N | Pre Stopped | Post Avg | Post N | Post Stopped | Delta |
|---------|---------|-------|-------------|----------|--------|--------------|-------|
| 25% | +1.07% | 671 | 653 | +0.75% | 153 | 150 | -0.32pp |
| 33% | +0.94% | 671 | 653 | +0.65% | 153 | 150 | -0.29pp |
| 50% | +0.67% | 671 | 653 | +0.45% | 153 | 150 | -0.22pp |

**Key Finding:** 11:00-11:30 exits are regime-neutral — delta narrows to +0.12pp / +0.62pp respectively. But absolute PnL drops dramatically (+1.48% / +1.32% pre vs +4.46% baseline). The afternoon session adds ~3pp to pre-Aug PnL. Trailing stops reduce both PnL and degradation equally — not useful. **11:30 exit is the best delta-narrower but costs ~3pp of pre-Aug avg.**

---

## Exp 9 — Sustained Drop Filter as Entry Condition

**Purpose:** Test early SD detection (bar 5/6/7) as entry gate instead of bar-60 classification.

| SD Confirm | Pre Avg | Pre N | Post Avg | Post N | Delta | vs Baseline |
|------------|---------|-------|----------|--------|-------|-------------|
| bar 5 | +6.45% | 162 | +3.04% | 38 | -3.41pp | -1.13pp |
| bar 6 | +5.50% | 177 | +1.27% | 42 | -4.23pp | -1.95pp |
| bar 7 | +5.96% | 182 | +1.60% | 35 | -4.36pp | -2.08pp |
| bar 60 (base) | +12.46% | 235 | +6.11% | 50 | — | — |

**Key Finding:** Early SD detection *widens* the gap vs baseline (all deltas worse than -2.28pp). The SD filter selects for high-conviction drops but these degrade more post-Aug, not less. SD is a high-alpha pre-Aug signal but not regime-resistant.

---

## Exp 10 — Re-Rank Quality Audit

**Purpose:** Compare pre-market rank (no re-rank) vs current re-rank at 09:31.

| Method | Pre Avg | Pre N | Post Avg | Post N | Delta |
|--------|---------|-------|----------|--------|-------|
| Pre-market (no re-rank) | +4.95% | 377 | +1.77% | 34 | -3.18pp |
| With re-rank at 09:31 | +4.46% | 671 | +2.18% | 153 | -2.28pp |

**Key Finding:** Re-rank helps post-Aug (+2.18% vs +1.77%) and narrows the delta (-2.28pp vs -3.18pp). The 09:31 re-rank is correct and should be kept.

---

## Exp 11 — Partial Entry (3-Tranche Scale-In)

**Purpose:** Test 3-tranche scale-in: LH/bar3 + 1% drop + VWAP cross.

| Method | Pre Avg | Pre N | Post Avg | Post N | Delta | vs Baseline |
|--------|---------|-------|----------|--------|-------|-------------|
| 3-tranche | +4.30% | 672 | +2.31% | 153 | -1.99pp | +0.29pp |
| baseline | +4.46% | 671 | +2.18% | 153 | -2.28pp | — |

Tranche fill distribution: 1 tranche=38, 2 tranches=282, 3 tranches=505

**Key Finding:** Scale-in narrows the gap by +0.29pp and slightly improves post-Aug (+2.31% vs +2.18%). Most trades (75%) fill all 3 tranches. Modest improvement, adds complexity.

---

## Exp 12 — SPY Intraday Direction Filter

**Purpose:** Test filtering by SPY direction (first 30min / first 15min) at entry.

Distribution: SPY down/flat=396, SPY up=428

### SPY 30-min Direction

| SPY Direction | Pre Avg | Pre N | Post Avg | Post N | Delta |
|---------------|---------|-------|----------|--------|-------|
| SPY down | +4.34% | 316 | +3.41% | 80 | -0.93pp |
| SPY up | +4.56% | 355 | +0.83% | 73 | -3.73pp |

### SPY 15-min Direction

| SPY Direction | Pre Avg | Pre N | Post Avg | Post N | Delta |
|---------------|---------|-------|----------|--------|-------|
| SPY 15m down | +4.15% | 326 | +2.96% | 70 | -1.18pp |
| SPY 15m up | +4.75% | 345 | +1.51% | 83 | -3.24pp |

**Key Finding:** SPY-down filter is powerful: delta narrows from -2.28pp to -0.93pp, post-Aug lifts to +3.41%. SPY-up trades collapse post-Aug (+0.83%). But this filter removes ~48% of trades. **SPY direction is the strongest regime-adaptation signal found in this investigation.**

---

## Exp 13 — ATR-Normalized Position Sizing

**Purpose:** Test ATR14-based position sizing vs fixed $100K.

742/824 trades have ATR14 data.

| Period | ATR Avg Pos | ATR% | ATR Total $ | Fixed Total $ |
|--------|-------------|------|-------------|---------------|
| Pre | $30,846 | 22.08% | +$625,455 | +$540,243 |
| Post | $39,808 | 9.18% | +$123,371 | +$69,293 |

### ATR% Quartiles: Q25=4.46%, Q50=7.05%, Q75=12.86%

| ATR Bucket | Pre N | Pre Avg | Post N | Post Avg |
|------------|-------|---------|--------|----------|
| low_atr | 128 | +2.57% | 58 | +3.60% |
| mid_atr | 298 | +3.04% | 72 | -0.02% |
| high_atr | 165 | +8.89% | 21 | +6.62% |

**Key Finding:** ATR sizing boosts dollar PnL significantly (post-Aug: $123K vs $69K) by sizing up on lower-vol names. Low-ATR stocks actually perform better post-Aug (+3.60% vs +2.57% pre). Mid-ATR collapses to -0.02% post-Aug. High-ATR maintains edge but has only 21 post-Aug trades.

---

## Exp 14 — Gap Bucket Analysis

**Purpose:** Analyze performance by gap size buckets.

| Gap Bucket | Pre N | Pre Avg | Pre WR | Post N | Post Avg | Post WR | Delta |
|------------|-------|---------|--------|--------|----------|---------|-------|
| <15% | 240 | +1.47% | 56% | 88 | +1.45% | 60% | -0.02pp |
| 15-30% | 236 | +1.92% | 61% | 38 | +0.47% | 50% | -1.45pp |
| >30% | 195 | +11.20% | 71% | 27 | +6.95% | 56% | -4.25pp |

### Entry Type by Gap Bucket

| Gap × Entry | Pre Avg | Pre N | Post Avg | Post N |
|-------------|---------|-------|----------|--------|
| <15% local_high | +1.62% | 165 | +2.22% | 53 |
| <15% fallback | +1.13% | 75 | +0.28% | 35 |
| 15-30% local_high | +0.74% | 147 | +1.83% | 28 |
| 15-30% fallback | +3.87% | 89 | -3.32% | 10 |
| >30% local_high | +9.21% | 127 | +10.16% | 17 |
| >30% fallback | +14.92% | 68 | +1.50% | 10 |

### Fallback PnL Collapse by Gap Bucket

| Gap Bucket | Pre FB Avg | Pre FB N | Post FB Avg | Post FB N | Delta |
|------------|------------|----------|-------------|-----------|-------|
| <15% | +1.13% | 75 | +0.28% | 35 | -0.86pp |
| 15-30% | +3.87% | 89 | -3.32% | 10 | -7.19pp |
| >30% | +14.92% | 68 | +1.50% | 10 | -13.43pp |

**Key Finding:** Small gaps (<15%) are regime-proof (-0.02pp delta). Large gaps (>30%) have the highest alpha but the worst degradation (-4.25pp). The fallback collapse is catastrophic in 15-30% gaps (-7.19pp) and >30% gaps (-13.43pp). **Local_high entries in <15% and >30% buckets actually improve post-Aug.**

---

## Exp 15 — Stop Loss Calibration

**Purpose:** Test fixed % and ATR-normalized stop losses.

### 15a — Fixed % Stop Loss

| SL | Pre Avg | Pre N | Pre Stopped | Post Avg | Post N | Post Stopped | Delta |
|----|---------|-------|-------------|----------|--------|--------------|-------|
| 2% | +0.33% | 671 | 538 | +0.07% | 153 | 105 | -0.26pp |
| 3% | +0.71% | 671 | 482 | -0.10% | 153 | 92 | -0.81pp |
| 5% | +0.66% | 671 | 394 | +0.68% | 153 | 64 | +0.03pp |
| 7% | +1.18% | 671 | 313 | +0.40% | 153 | 50 | -0.78pp |
| 10% | +1.68% | 671 | 235 | +0.94% | 153 | 32 | -0.74pp |

### 15b — ATR-Normalized Stop Loss

| SL | Pre Avg | Pre N | Pre Stopped | Post Avg | Post N | Post Stopped | Delta |
|----|---------|-------|-------------|----------|--------|--------------|-------|
| 0.25×ATR | +0.94% | 591 | 454 | +0.11% | 151 | 114 | -0.83pp |
| 0.50×ATR | +1.89% | 591 | 359 | +0.36% | 151 | 83 | -1.54pp |
| 0.75×ATR | +2.16% | 591 | 289 | +0.16% | 151 | 68 | -2.00pp |
| 1.00×ATR | +2.22% | 591 | 252 | +0.55% | 151 | 56 | -1.67pp |

MFE/MAE reference:
- Pre: MFE mean=+14.11%, MAE mean=+15.27%
- Post: MFE mean=+8.40%, MAE mean=+8.10%

**Key Finding:** SL=5% is the only delta-neutral option (+0.03pp) but it caps avg PnL at +0.66%/+0.68%. All stop losses destroy absolute PnL because the strategy's edge comes from large tail moves that require riding through drawdowns. **Stop losses are incompatible with this strategy's return profile.** The no-SL approach remains correct.

---

## Summary Table

| Exp | Name | Best Pre | Best Post | Delta | Narrows Gap? | Improves Post? |
|-----|------|----------|-----------|-------|:------------:|:--------------:|
| 1 | Local High Audit | — | — | — | — | — |
| 2 | ORB Entry | +3.94% | +2.18% | -1.76pp | YES | YES |
| 3 | VWAP Entry | +4.47% | +2.45% | -2.03pp | YES | YES |
| 4 | Volume Entry Quality | +7.92% | +6.83% | -1.09pp | YES | YES |
| 5 | Fallback Timing Grid | +3.52% | +3.48% | -0.04pp | YES | YES |
| 6 | Gap at Entry Gate | +4.14% | +2.69% | -1.45pp | YES | YES |
| 7 | Intraday Vol Gate | +4.54% | +2.27% | -2.27pp | YES | YES |
| 8 | Exit Optimization | — | — | — | — | — |
| 9 | SD Filter Entry | +6.45% | +3.04% | -3.41pp | NO | YES |
| 10 | Re-rank Quality | +4.95% | +1.77% | -3.18pp | NO | NO |
| 11 | Partial Entry | +4.30% | +2.31% | -1.99pp | YES | YES |
| 12 | SPY Direction Filter | +4.34% | +3.41% | -0.93pp | YES | YES |
| 13 | ATR Position Sizing | — | — | — | — | — |
| 14 | Gap Bucket Analysis | +11.20% | +6.95% | -4.25pp | NO | YES |
| 15 | Stop Loss Calibration | — | — | — | — | — |

---

## Top Actionable Findings (ranked by delta-narrowing impact)

### Tier 1 — High Impact (delta < -1.0pp)

1. **Remove fallback (Exp 5):** LH-only delta = -0.04pp (vs baseline -2.28pp). Sacrifices 28% of trades but eliminates virtually all post-Aug degradation. Post-Aug avg = +3.48%.

2. **SPY-down filter (Exp 12):** Delta = -0.93pp, post-Aug = +3.41%. Strongest single filter but removes ~48% of trades.

3. **Volume quality gate (Exp 4):** High-vol entries: delta = -1.09pp, post-Aug = +6.83%. Requires volume-at-entry threshold.

4. **Gap freshness at entry >=70% (Exp 6):** Delta = -1.45pp, post-Aug = +2.69%. Easy to implement.

### Tier 2 — Moderate Impact

5. **ORB range=1 entry (Exp 2):** Delta = -1.76pp, post-Aug = +2.18%. Alternative entry mechanism.

6. **VWAP cross entry (Exp 3):** Delta = -2.03pp, post-Aug = +2.45%. Higher fill rate than LH-only.

7. **3-tranche scale-in (Exp 11):** Delta = -1.99pp, post-Aug = +2.31%. Modest improvement.

### Tier 3 — Informational / Not Actionable

8. **11:30 exit (Exp 8):** Best delta (+0.62pp) but caps absolute PnL at +1.32%.
9. **ATR sizing (Exp 13):** Boosts dollar PnL but doesn't change % PnL.
10. **Intraday vol gate (Exp 7):** No separation power.
11. **SD filter (Exp 9):** Widens gap.
12. **Stop losses (Exp 15):** Incompatible with strategy's tail-move profile.

### Cross-Experiment Insight

The consistent thread across all 15 experiments: **the fallback mechanism is the single largest source of post-Aug degradation**. Removing it (Exp 5), improving entry quality (Exp 4), requiring gap freshness at entry (Exp 6), and filtering by SPY direction (Exp 12) all converge on the same conclusion — the compressed post-Aug regime punishes low-conviction entries that worked in the high-vol pre-Aug period.

The optimal regime-adapted configuration would combine:
- LH-only (no fallback): -0.04pp delta
- SPY-down filter: lifts post-Aug from +2.18% to +3.41%
- Volume quality gate: lifts post-Aug to +6.83% for high-vol entries
- Gap freshness >=70%: additional quality gate

Trade-off: this combination would reduce trade count significantly (estimated ~30-40% of current volume) but would deliver nearly regime-neutral performance.
