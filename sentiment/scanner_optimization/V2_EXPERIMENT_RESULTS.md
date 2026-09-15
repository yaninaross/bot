# Scanner V2 Experiment Results — 10 New Directions

**Date:** 2026-09-11  
**Trades:** 316 (Fold A: 117, Fold B: 199)  
**Versions:** V2, V3, V3.5, V3.6, V3.7, V4.0  
**CAVEAT:** Fold-CV = version-CV (A=V2-V3, B=V3.5-V4)

---

## Top-Level Findings (ranked by impact)

| # | Finding | Evidence | Actionable? |
|---|---------|----------|-------------|
| 1 | **SHORT signals are swing signals, not scalps** | SHORT PnL: 30m +0.40% → T+2 +4.42% (84% WR, n=50) | YES — test multi-day holds |
| 2 | **LLM adds no selection value over price screen** | Non-flagged gappers: +5.68% vs flagged +2.78% (SHORT, 60m) | YES — rebuild universe from price/volume |
| 3 | **ORB+LLM agreement is strongest filter** | +1.30% avg, 61% WR (n=148) vs disagree -1.53% | YES — use as entry qualifier |
| 4 | **LLM predicts volatility, not direction** | \|expected_move\| → \|realized\| r=+0.28 (p<0.0001) | YES — use for sizing/watchlist |
| 5 | **Earnings trades are net negative** | -0.61% avg, 43% WR, worsens to -2.38% at T+3 | YES — exclude from system |
| 6 | **Every stop level destroys value** | Best SL (1.5%): -0.09%, No SL: +0.01% | INVESTIGATE — tail risk vs. PnL |
| 7 | **gap_pct is strongest feature** | r=+0.56 with \|realized move\|, stable across horizons | Use for ranking |

---

## EXP S1: Stop Design

**Optimal stop from MAE analysis:** 1.98% (80th pct of winners' MAE), = 0.36 × ATR.

But every stop level is net negative:

| Stop Level | n | Avg PnL | Med | Trim | WR | CV |
|-----------|---|---------|-----|------|-----|-----|
| Fixed 1.5% | 272 | -0.089% | -0.381% | -0.403% | 40% | no |
| Fixed 2.5% | 272 | -0.184% | -0.153% | -0.427% | 44% | no |
| Fixed 5.0% | 272 | -0.210% | -0.030% | -0.273% | 48% | no |
| ATR×0.5 | 272 | -0.300% | -0.082% | -0.230% | 46% | no |
| ATR×0.75 | 272 | -0.243% | -0.025% | -0.135% | 48% | no |
| **No SL** | 272 | **+0.014%** | +0.065% | +0.038% | 51% | no |

**MAE/MFE Summary:**
- Winners MAE P80: LONG -1.73%, SHORT -2.23%
- Winners avg MFE: +5.42% vs losers +1.18%
- Losers avg MAE: -5.08% vs winners -1.94%

**Interpretation:** Stops kill winners more than they protect from losers. The signal has fat-tailed winners that need room to breathe. However "no SL" on 4 days doesn't prove tail safety.

---

## EXP S2: Horizon Extension

**This is the most important result.** SHORT signals get dramatically better at longer horizons:

| Horizon | All Avg | All WR | SHORT Avg | SHORT WR | SHORT CV |
|---------|---------|--------|-----------|----------|----------|
| 30m | +0.16% | 54% | **+0.40%** | 56% | **YES** |
| 60m | +0.01% | 51% | +0.31% | 52% | YES |
| 2h | +0.23% | 53% | **+0.81%** | 56% | **YES** |
| EOD | +0.26% | 50% | +0.63% | 56% | no |
| T+1 | -0.04% | 50% | **+1.97%** | 69% | no |
| T+2 | -0.32% | 49% | **+4.42%** | 84% | no |
| T+3 | -0.52% | 46% | **+4.27%** | 73% | no |

**Feature correlation vs horizon:**
- `gap_pct` strongest across all horizons: r=+0.41 to +0.46
- `log_vol_ratio` improves at longer horizons: +0.15 (60m) → +0.35 (T+1)
- `remaining_move_pct` becomes increasingly anti-correlated: -0.12 (60m) → **-0.35** (T+2)

**CRITICAL CAVEATS:** T+2/T+3 don't cross-validate and n drops to 50/44. These could be regime artifacts (Sep 8-11 was a specific market period). The remaining_move_pct anti-correlation at longer horizons suggests the LLM's expected move estimate may be actively harmful for swing trades.

---

## EXP S3: Counterfactual Universe

**The most important question about the project: does the LLM add value over a price screen?**

- Total gappers (>3% gap): **3,628** across 4 days
- Scanner flagged: **93** (2.6% of gappers)
- Non-flagged: 3,535

| Group | n | Avg PnL | Med | WR |
|-------|---|---------|-----|-----|
| Flagged: all signals | 272 | +0.014% | +0.065% | 51% |
| Flagged: SHORT gap>3% | 25 | +2.779% | +0.554% | 64% |
| **Non-flagged: gap-fade** | **60** | **+4.914%** | **+1.699%** | **60%** |
| **Non-flagged: SHORT gap-ups** | **37** | **+5.679%** | **+1.802%** | **65%** |

**Gap-fade is profitable market-wide:** Gap-up stocks fade +1.86% on average (n=1,728).

**Conclusion:** The LLM's selection provides no edge — non-flagged gappers outperform flagged ones. The gap-fade effect is a market-wide phenomenon that doesn't require LLM classification. The scanner's value (if any) is in catalyst classification, not ticker selection.

---

## EXP S4: Cross-Sectional Ranking

Daily tercile long/short (dollar-neutral, removes day effect):

| Feature | Avg L/S Return | Consistent? |
|---------|---------------|-------------|
| remaining_move_pct | **-0.944%** | Mixed (anti-signal) |
| expected_move_pct | -0.352% | Mixed |
| log_vol_ratio | +0.223% | Mixed |
| gap_pct | +0.355% | Mixed |
| abs_gap_pct | +0.389% | Mixed |

No feature produces consistent daily cross-sectional returns. `remaining_move_pct` is the worst — bottom tercile outperforms top in 3 of 4 days.

---

## EXP S5: Magnitude Without Direction

**LLM predicts volatility, not direction.**

| Predictor | Target | r | p-value |
|-----------|--------|---|---------|
| \|gap_pct\| | \|realized_60m\| | **+0.560** | <0.0001 *** |
| atr_gap | \|realized_60m\| | **+0.456** | <0.0001 *** |
| log_vol_ratio | \|realized_60m\| | **+0.346** | <0.0001 *** |
| \|expected_move\| | \|realized_60m\| | **+0.281** | <0.0001 *** |
| \|expected_move\| | MFE+\|MAE\| | **+0.250** | <0.0001 *** |
| confidence | \|realized_60m\| | +0.054 | 0.3792 |
| \|remaining_move\| | \|realized_60m\| | +0.039 | 0.5167 |

**Tercile analysis:** High |expected_move| tercile has 2.5x the |PnL| and 2x the total excursion of the low tercile. The LLM correctly identifies which stocks will move the most — it just can't tell you which way.

**Confidence and remaining_move_pct are worthless** as predictors.

---

## EXP S6: Stability Across Runs

| Filter | n | Avg PnL | Med | CV |
|--------|---|---------|-----|-----|
| Stable (score>=0.8) | 215 | 0.000% | +0.020% | no |
| Unstable (<0.8) | 57 | +0.068% | +0.197% | YES |
| n_runs >= 3 | 78 | +0.276% | +0.322% | **YES** |
| n_runs >= 4 | 37 | **+0.571%** | +0.261% | **YES** |

**Interesting:** High-consistency signals (n_runs >= 3-4) perform better and cross-validate. But "stable" (same direction) isn't better than "unstable" — suggesting the scanner reinforces its initial pick across runs rather than converging on truth.

---

## EXP S7: Premarket Path Shape

| PM Retrace | n | Avg PnL | WR |
|-----------|---|---------|-----|
| Held (<0.2) | 80 | **+0.673%** | 55% |
| Mild (0.2-0.5) | 74 | -0.464% | 42% |
| Moderate (0.5-0.8) | 60 | -0.410% | 52% |
| Full retrace (>0.8) | 58 | +0.154% | 57% |

**PM volume** is the only significant correlate: r=+0.198 (p=0.001).

**Direction interaction:**
- SHORT + held: +1.030%, 59% WR
- SHORT + full retrace: +0.900%, 70% WR

---

## EXP S8: Opening Range Breakout

ORB triggered on 315/316 trades.

| Condition | n | Avg PnL | WR |
|----------|---|---------|-----|
| All ORB (price direction) | 276 | +0.282% | 44% |
| LLM direction (same trades) | 271 | +0.013% | 51% |
| **ORB agrees with LLM** | **148** | **+1.296%** | **61%** |
| ORB disagrees with LLM | 123 | -1.530% | 39% |

**The 2.8% spread between agreement and disagreement is the strongest filter found in this dataset.** When ORB (pure price) and LLM agree on direction, the trade works. When they disagree, it loses.

This reframes the LLM's role: it's not a directional predictor — it's a qualifier that confirms or denies a price-based signal.

---

## EXP S9: Ticker Reaction History

Weak signal. Historical continuation rate vs PnL: r=+0.065 (not significant).

| Group | n | Avg PnL | WR |
|-------|---|---------|-----|
| Faders (cont_rate<40%) | 64 | -0.108% | 48% |
| Runners (cont_rate>60%) | 43 | +0.466% | 53% |

Not actionable at this sample size.

---

## EXP S10: Earnings Module

**Earnings trades are net negative and should be excluded.**

| Group | n | Avg PnL | Med | WR |
|-------|---|---------|-----|-----|
| **Earnings** | **77** | **-0.613%** | -0.284% | 43% |
| Non-earnings | 195 | +0.262% | +0.177% | 54% |
| Beat | 15 | -1.304% | -0.697% | 40% |
| Miss | 6 | -1.397% | -0.910% | 33% |

Earnings trades worsen at longer horizons: 60m -0.61% → T+3 -2.38%.

Even when the LLM's rule direction (beat→LONG, miss→SHORT) agrees with the LLM's direction call: -1.597% avg. Earnings gap dynamics are fundamentally different from other catalysts.

---

## Version Breakdown

| Version | n | Avg PnL | Dates |
|---------|---|---------|-------|
| V2 | 65 | +0.502% | 09-08 |
| V3 | 43 | -0.102% | 09-08, 09-09 |
| V3.5 | 29 | -0.057% | 09-10 |
| V4.0 | 110 | -0.156% | 09-11 |

V2 drives most of Fold A's outperformance. This confirms the fold-CV = version-CV confound — Fold A's positive results may be about the V2 candidate generator, not a stable signal.

---

## Robustness Re-evaluation

| Strategy | n | Avg | Med | Trim | SE | LOO Range | CV |
|----------|---|-----|-----|------|----|-----------|----|
| SHORT+goodcat noSL | 48 | +1.067% | +0.417% | +0.632% | 1.099 | [+0.34, +1.61] | YES |
| (without top trade) | 47 | +0.337% | +0.351% | — | — | — | — |
| SHORT noSL | 137 | +0.307% | +0.081% | +0.136% | 0.511 | — | YES |
| All noSL | 272 | +0.014% | +0.065% | +0.038% | 0.299 | — | no |
| All SL=2.5% | 272 | -0.184% | -0.153% | -0.427% | 0.125 | — | no |

SHORT+goodcat drops from +1.067% to +0.337% when the top trade (+35.4%) is removed. LOO range [+0.34, +1.61] — fragile.

---

## What This Means for the System

1. **Universe:** Build from price/volume (gappers >3%, high PM volume), not LLM selection. The counterfactual proves this.

2. **Direction:** Use ORB (price break of 5-min range) for direction. LLM agrees → trade. LLM disagrees → skip.

3. **Holding period:** SHORT signals are swing signals. Test 2h-EOD holds with wider stops.

4. **LLM role:** Catalyst classifier + volatility forecaster. Not a directional predictor.

5. **Exclusions:** Remove earnings trades. They are net negative across every cut.

6. **Stops:** Current 2.5% SL costs ~0.2%/trade. Wider stops or no-SL outperform. Test ATR-scaled trailing stops.

7. **Features that matter:** `gap_pct` (r=+0.56), `log_vol_ratio` (r=+0.35), `|expected_move|` (r=+0.28). `confidence`, `remaining_move_pct`, and direction are worthless.
