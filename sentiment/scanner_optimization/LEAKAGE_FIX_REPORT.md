# Scanner Optimization — Leakage Fix Report

**Date:** 2026-09-11 (post-fix re-run)
**Original report:** `OPTIMIZATION_REPORT.md` (pre-fix, numbers are inflated)
**Data:** 258-278 trades across Sep 8-11 2026
**Cross-validation:** Fold A (Sep 8-9, V2-V3.3) vs Fold B (Sep 10-11, V3.8-V4)

---

## What Happened

The original creative_experiments.py had **four sources of data leakage** that inflated the top findings. After fixing all four, the picture changes dramatically.

---

## Leakage #1: Entry-Time Features in ML (EXP 1)

**Bug:** `first_5min_ret`, `first_5min_range`, and `first_5min_confirms` were used as ML features. These measure price movement relative to `entry_price` — the same price used to compute the PnL target. A stock that moved +2% in the first 5 minutes after entry already has +2% of "free" PnL baked into the 60m target. GBM learns to select these trades mechanically.

**Evidence:** `first_5min_ret` (16.1%) + `first_5min_confirms` (10.7%) + `first_5min_range` (8.2%) = **35% of GBM importance** — far more than any legitimate feature.

**Fix:** Removed `first_5min_ret`, `first_5min_range`, and `first_5min_confirms_int` from `FEATURE_COLS`.

**Impact:**

| Feature | Before (leaky) | After (clean) |
|---------|----------------|---------------|
| #1 feature | first_5min_ret (16.1%) | log_vol_ratio (20.6%) |
| #2 feature | log_vol_ratio (12.8%) | gap_pct (16.4%) |
| #3 feature | remaining_move_pct (12.4%) | remaining_move_pct (11.8%) |
| #4 feature | first_5min_confirms (10.7%) | atr_gap (9.8%) |
| #5 feature | gap_pct (8.9%) | expected_move_pct (9.5%) |
| confidence | ~0% | ~0% (still noise) |

All ML strategies (GBM, RF, Logistic) are now **negative on the test fold** at every threshold. The ML "edge" was entirely first-5-min leakage.

---

## Leakage #2: 5-Min Confirmation PnL (EXP 6)

**Bug:** The "5-min confirmation" filter checked whether price moved in the signal direction during the first 5 minutes — but PnL was still measured from the original `entry_price` at t=0. So the confirmation move itself was included in the PnL. A trade where price confirmed +0.5% in 5 minutes already had +0.5% of "free" PnL embedded in the 60m target. The filter was selecting trades that had already moved favorably, not trades that would *continue* to move.

**Fix:** Rebased entry to t+5. Created `entry_price_5m = close of bar at entry + 5 minutes`. All EXP 6 PnL is now computed from this rebased price forward (column `pnl_5m_60m_sl2_5`). Stop-loss is also checked from the rebased price.

**Impact:**

| Strategy | Before (leaky) | After (rebased) | Verdict |
|----------|----------------|-----------------|---------|
| 5min confirms direction | +0.55%, CV:YES | **+0.04%**, CV:no | DEAD |
| 5min strong >0.3% | +0.79%, CV:YES | **+0.06%**, CV:no | DEAD |
| 5min + good catalyst | +0.78%, CV:YES | **+0.50%**, CV:no (A=-0.52%, B=+0.85%) | DEAD |
| 5min + SHORT | — | **+0.09%**, CV:no | DEAD |
| 5min contrarian | -1.03% | **-0.22%** | Still negative but smaller |
| 5min confirms no SL | — | **+0.06%**, CV:no | DEAD |

The original +0.79% headline was almost entirely the confirmation move itself plus stop-loss exclusion. Rebased, confirmation adds essentially nothing.

The confirm+goodcat combo (+0.50%) is interesting but fails CV (Fold A is -0.52%).

---

## Leakage #3: NLP Token Selection (EXP 5)

**Bug:** Bearish/bullish tokens were mined from *all* data, then the same data was split into folds for "cross-validation." The token sets were fit on the test fold. This is the NLP equivalent of fitting a model on test data.

**Fix:** Rewrote with fold-local token mining. `_mine_tokens(df_train, lift_thresh, min_n=5)` extracts tokens from training fold only. Test fold uses the training fold's token set blindly. Tests three lift thresholds (5, 10, 15) for both `no_bearish` and `has_bullish` modes.

**Impact:**

| Strategy | Before (leaky) | After (fold-local) | CV | Note |
|----------|----------------|--------------------|----|------|
| no_bearish | +0.51% (n=91) | **+0.49%** (n=44) | YES | n=37 in A, n=7 in B |
| no_bearish lift>10 | — | **+0.21%** (n=104) | YES | A=+0.36%, B=+0.09% |
| no_bearish lift>15 | — | **-0.17%** (n=164) | no | — |
| has_bullish (all) | — | all negative | no | — |

The no_bearish filter survived but at reduced strength and with wildly different token sets per fold (Fold A: "earnings", "beat", "guidance"; Fold B: "dilution", "august", "concerns"). With n=7 in Fold B for lift>5, the +0.49% result is not statistically robust.

---

## Leakage #4: Anti-Selection Inversion (EXP 3)

**Bug (methodological):** `pnl_flipped = -pnl` doesn't properly simulate the inverse trade's stop-loss behavior. A LONG that gets stopped out at -2.5% doesn't mean a SHORT would have earned +2.5% — the SHORT has its own stop-loss path. Additionally, with Fold A = V2-V3.3 and Fold B = V3.8-V4, fold-CV is also version-CV.

**Fix:** Renamed to `DIAG_flip_selected_direction` and demoted to diagnostic. The result (+0.44%, 67.9% WR) is evidence that the scanner's selection mechanism is anti-predictive — not a tradeable strategy.

---

## What Survives After Cleanup

### Cross-Validated Winners (14 strategies, ranked by avg PnL):

| # | Experiment | Strategy | n | Avg PnL | WR | SE | CV | A | B |
|---|-----------|----------|---|---------|----|----|----|----|---|
| 1 | EXP 8 | SHORT good cat, no SL @60m | 45 | **+1.68%** | 57.8% | 1.01 | YES | +4.62% | +0.94% |
| 2 | EXP 8 | SHORT good cat, no SL @30m | 45 | **+0.74%** | 62.2% | 0.61 | YES | +0.71% | +0.75% |
| 3 | EXP 8 | SHORT only, no SL @60m | 129 | **+0.67%** | 52.7% | 0.49 | YES | +0.88% | +0.57% |
| 4 | EXP 5 | NLP no_bearish lift>5 | 44 | **+0.49%** | 54.5% | 0.36 | YES | +0.30% | +1.48% |
| 5 | EXP 3 | DIAG flip selected direction | 56 | **+0.44%** | 67.9% | 0.28 | YES | +0.62% | +0.25% |
| 6 | EXP 8 | SHORT only, no SL @30m | 140 | **+0.40%** | 55.7% | 0.29 | YES | +0.21% | +0.49% |
| 7 | EXP 8 | All signals, no SL @60m | 258 | **+0.25%** | 52.3% | 0.29 | YES | +0.54% | +0.06% |
| 8 | EXP 5 | NLP no_bearish lift>10 | 104 | **+0.21%** | 45.2% | 0.24 | YES | +0.36% | +0.09% |
| 9 | EXP 8 | All signals, no SL @30m | 278 | **+0.16%** | 54.0% | 0.20 | YES | +0.22% | +0.12% |
| 10 | EXP 5 | NLP short catalyst text | 64 | **+0.14%** | 46.9% | 0.29 | YES | +0.18% | +0.09% |
| 11 | EXP 3 | ANTI gated only | 43 | **+0.13%** | 55.8% | 0.19 | YES | +0.48% | +0.03% |
| 12 | EXP 4 | Gap fade >5% SHORT | 20 | **+0.25%** | 45.0% | — | YES | — | — |

### What's Dead After Cleanup:
- **5-min confirmation** (EXP 6): ALL variants fail CV after rebasing entry to t+5
- **ML classifiers** (EXP 1): ALL negative on test fold without leaky features
- **Ensemble micro-signals** (EXP 7): No threshold cross-validates
- **has_bullish NLP**: All negative

---

## Clean GBM Feature Importance

| Rank | Feature | Importance | Interpretation |
|------|---------|-----------|----------------|
| 1 | **log_vol_ratio** | 20.6% | Premarket volume vs avg daily — strongest clean signal |
| 2 | **gap_pct** | 16.4% | Gap size at open |
| 3 | **remaining_move_pct** | 11.8% | Residual from LLM's expected move |
| 4 | **atr_gap** | 9.8% | ATR-normalized gap |
| 5 | **expected_move_pct** | 9.5% | LLM's expected move magnitude |
| 6 | **catalyst_len** | 5.9% | Length of catalyst text |
| 7 | **abs_gap_pct** | 5.1% | Absolute gap |
| 8 | **cat_breaking** | 4.6% | Breaking news flag |
| 9 | **kw_analyst** | 3.8% | Analyst action keyword |
| 10 | **gap_aligns_int** | 2.9% | Gap aligns with direction |
| — | **confidence** | ~0% | Confirmed worthless |
| — | **direction_int** | ~0.9% | Near zero |

---

## Baseline Measurements (for reference)

| Baseline | n | Avg PnL | WR | CV |
|----------|---|---------|----|----|
| All signals (SL=2.5%) | 258 | -0.14% | 45.3% | no |
| Selected only | 56 | -0.44% | 32.1% | no |
| SHORT only | 129 | -0.10% | 44.2% | no |
| LONG only | 129 | -0.19% | 46.5% | no |

---

## Key Takeaways

1. **The 5-min confirmation finding was leakage.** The +0.79% headline and 1.58pp "spread" were almost entirely the confirmation move itself being counted in PnL. After rebasing, confirmation adds ~0.04% — nothing.

2. **ML had no real edge.** GBM's 27% importance on first_5min features was mechanical correlation. Without them, no ML strategy works. The value of ML here is in **feature importance ranking**, not prediction.

3. **SHORT + no SL is the real finding.** The +1.68% (n=45) for SHORT + good catalyst without stop loss is the strongest surviving signal. The stop-loss is destroying edge by triggering during the initial momentum phase of gap-up stocks.

4. **NLP survives but barely.** The no_bearish filter at +0.49% passes CV but with n=7 in one fold. At lift>10 (+0.21%, n=104) it's more robust but marginal.

5. **The scanner is a reliable contra-indicator.** The DIAG flip result (+0.44%) confirms the LLM consistently gets direction wrong on selected trades. This is diagnostic, not a strategy.

6. **Fold-CV is also version-CV.** Fold A = V2-V3.3, Fold B = V3.8-V4. Any fold-level difference could be version-driven.

---

## Updated Priority List

1. **P0:** Fix harness leakage (DONE — this report)
2. **P0:** ATR-scaled stops (Fable #8) — "no SL" is not a finding, the answer is wider stops
3. **P1:** Re-run NLP with more data to get robust fold sizes
4. **P1:** Log `log_vol_ratio` and `gap_pct` in BQ for ongoing validation
5. **P2:** Restructure LLM: classify catalyst → look up base rate → use price for direction
6. **META:** Get more than 4 days of data before drawing conclusions

## User's Vision for the Surviving System

> "Universe from price (#10), LLM classifies catalyst, direction from residual sign with an overshoot regime, entry after 5-minute confirmation, short-side prior, ATR stop."

The 5-min confirmation survives as an **entry timing mechanism** (avoid opening-bar stop-outs) even though it's dead as a **PnL filter**. The distinction matters: entering at t+5 with an ATR-scaled stop may still beat entering at t+0 with a fixed stop — but that's an SL experiment, not a confirmation experiment.

---

## Fixes Applied to `creative_experiments.py`

1. Removed `first_5min_ret`, `first_5min_range`, `first_5min_confirms_int` from ML features
2. Added rebased t+5 entry price computation with full PnL re-simulation
3. All EXP 6 strategies now use rebased PnL (`pnl_5m_60m_sl2_5`)
4. Leaky originals kept for comparison, labeled `LEAKY_*`
5. NLP token mining rewritten with fold-local `_mine_tokens()` function
6. EXP 3 renamed from `ANTI_` to `DIAG_` prefix
7. Assertion added: post-entry features must only gate into rebased columns
8. `first_5min_confirms` removed from ensemble micro-signal scoring

---

## Files

| File | Description |
|------|-------------|
| `creative_experiments.py` | Fixed harness (all 4 leakage sources corrected) |
| `LEAKAGE_FIX_REPORT.md` | This report — post-fix results and analysis |
| `OPTIMIZATION_REPORT.md` | Original report (PRE-FIX — numbers are inflated) |
| `results/creative_experiments_results.json` | Updated results JSON |
| `results/creative_features_full.csv` | Updated feature data with rebased PnL columns |

---

*Generated: 2026-09-11. All numbers are from the post-leakage-fix re-run.*
*Leakage identified by user review of the original harness code.*
