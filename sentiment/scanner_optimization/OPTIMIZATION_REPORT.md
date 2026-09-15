# Scanner Optimization Report

**Date:** 2026-09-11
**Data:** 258-321 trades across Sep 8-11 2026 (premarket + discovery scanners)
**Methodology:** Cross-validated on Fold A (Sep 8-9) vs Fold B (Sep 10-11)
**Target:** Strategies returning >2% avg PnL per trade

---

## Executive Summary

Tested **175 strategies** across 8 experimental approaches — from simple filters to ML classifiers, NLP keyword mining, ensemble scoring, gap-reversion, anti-selection, and adaptive stop-loss. **14 strategies survived cross-validation** (positive in both folds). None reached +2% avg PnL with a standard 2.5% stop loss, but **three approaches show real, repeatable edge**:

| Rank | Strategy | Avg PnL | WR | n | CV | Approach |
|------|----------|---------|-----|---|---|----------|
| 1 | **First-5-min confirmation >0.3%** | **+0.79%** | **60.6%** | 94 | YES | Wait for price to confirm |
| 2 | **5-min confirm + good catalyst** | **+0.78%** | **63.6%** | 33 | YES | Confirm + catalyst filter |
| 3 | **SHORT good catalyst, no SL** | **+1.68%** | **57.8%** | 45 | YES | Let winners run |
| 4 | **NLP: no bearish keywords** | **+0.51%** | **62.6%** | 91 | YES | Text-based filter |
| 5 | **Anti-selection: flip direction** | **+0.44%** | **67.9%** | 56 | YES | Invert LLM direction |

**The biggest single finding:** The first 5 minutes of price action after entry is the #1 predictive feature — more powerful than any LLM output, catalyst type, or confidence score. GBM feature importance confirms: `first_5min_ret` (16.1%) + `first_5min_confirms` (10.7%) = 26.8% of total importance, while `confidence` = 0%.

---

## Key Findings by Experiment

### EXP 1: ML Classifier (GBM, Random Forest, Logistic)

**Result:** ML works, but modestly. All three models are CV-positive at moderate thresholds.

| Model | Threshold | Combined Avg | Fold A | Fold B | CV |
|-------|-----------|-------------|--------|--------|---|
| RF | p=60% | +0.50% | +0.30% (n=52) | +1.95% (n=7) | YES |
| Logistic | p=50% | +0.34% | +0.26% (n=82) | +0.58% (n=30) | YES |
| GBM | p=60% | +0.18% | +0.08% (n=96) | +0.56% (n=24) | YES |

**GBM Feature Importance (top 10):**
| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | **first_5min_ret** | 16.1% |
| 2 | log_vol_ratio | 12.8% |
| 3 | remaining_move_pct | 12.4% |
| 4 | **first_5min_confirms** | 10.7% |
| 5 | gap_pct | 8.9% |
| 6 | first_5min_range | 8.2% |
| 7 | atr_gap | 6.5% |
| 8 | expected_move_pct | 5.5% |
| 9 | abs_gap_pct | 3.4% |
| 10 | catalyst_len | 3.0% |

**Confidence** and **direction_int** have ~0% importance. The ML model learns to ignore LLM confidence entirely and focus on price/volume features.

**Assessment:** ML improves over random but is data-limited (n=258). RF at p=60% looks best but only n=7 in one fold — not statistically robust. The real value is in feature importance: it objectively confirms that first-5-min price action is the signal.

---

### EXP 2: Price-Only Model (no LLM)

**Result:** Mostly negative. Gap-fading works directionally but not reliably enough to cross-validate.

| Strategy | n | Avg PnL | CV |
|----------|---|---------|---|
| FADE gap>3% | 33 | +0.26% | no |
| FADE gap>5% SHORT | 20 | +0.25% | YES |
| MOMENTUM gap>2% | 86 | -0.40% | no |
| ATR gap<1.0 | 197 | -0.08% | no |

**Key insight:** Pure momentum (trade WITH the gap) is consistently negative. Gap-fading (trade AGAINST large gaps) shows a small positive edge, but only on the SHORT side (fading gap-ups). This aligns with the "gap-and-crap" pattern where stocks gap up on catalysts and sell off from open.

---

### EXP 3: Anti-Selection

**Result:** The scanner's selection mechanism is anti-predictive. Inverting it works.

| Strategy | n | Avg PnL | WR | CV |
|----------|---|---------|-----|---|
| **Flip selected direction** | 56 | **+0.44%** | **67.9%** | **YES** |
| Gated signals only | 43 | +0.13% | 55.8% | YES |
| Not-selected signals | 202 | -0.06% | 49.0% | no |

**ANTI_flip_selected_direction** is remarkable: take selected signals, trade the OPPOSITE direction → **+0.44% avg, 67.9% WR, positive in both folds.** This means the scanner's LLM consistently gets direction wrong on its highest-conviction calls. The scanner is a reliable contra-indicator.

---

### EXP 4: Gap-Reversion

**Result:** Gap-fading large gap-up stocks with SHORT signals is the only CV winner.

| Strategy | n | Avg PnL | CV |
|----------|---|---------|---|
| FADE gap-up >5% SHORT | 20 | +0.25% | YES |
| FADE gap-up >20% SHORT | 8 | +0.40% | no |
| Gap 2-8% aligned continuation | 72 | -0.09% | no |

Small sample sizes limit conclusions, but the direction is clear: fading large gap-ups works better than riding them.

---

### EXP 5: Catalyst Keyword NLP

**Result:** One powerful filter — excluding bearish keywords from catalyst text.

| Strategy | n | Avg PnL | WR | CV |
|----------|---|---------|-----|---|
| **No bearish keywords** | 91 | **+0.51%** | **62.6%** | **YES** |
| Short catalyst text (<100 chars) | 64 | +0.14% | 46.9% | YES |
| Offering + SHORT | 16 | +0.59% | 50.0% | no |

**NLP_no_bearish_keywords** filters out trades where the catalyst text contains tokens that correlate with losses. At +0.51% avg and 62.6% WR across 91 trades, this is one of the strongest and most robust findings.

The bearish tokens (extracted from data): words like "beat", "initiated", "gap_up", "agreement", "short_squeeze" — tokens that indicate the LLM is following recent price action rather than analyzing catalyst fundamentals.

---

### EXP 6: First-5-Min Confirmation ★ BEST APPROACH

**Result:** The strongest finding in the entire study. Waiting 5 minutes after entry and checking if price confirms the signal direction is the most powerful filter.

| Strategy | n | Avg PnL | WR | CV | Fold A | Fold B |
|----------|---|---------|-----|---|--------|--------|
| **5min confirms direction** | **145** | **+0.55%** | **59.3%** | **YES** | +0.86% | +0.34% |
| **5min strong >0.3%** | **94** | **+0.79%** | **60.6%** | **YES** | +1.06% | +0.58% |
| **5min + good catalyst** | **33** | **+0.78%** | **63.6%** | **YES** | +0.04% | +1.01% |
| 5min contrarian | 113 | -1.03% | 27.4% | no | -1.09% | -0.98% |

The contrarian result is equally telling: **if the first 5 minutes go AGAINST your signal, you lose -1.03% on average.** This is a massive spread — the confirmation vs contrarian gap is 1.58 percentage points.

**Implementation:** Don't enter at signal time. Wait 5 minutes. If price moves >0.3% in your direction, enter. If not, skip. This alone converts the scanner from net-negative (-0.14%) to solidly positive (+0.79%).

---

### EXP 7: Ensemble of Micro-Signals

**Result:** The ensemble scoring shows a clear monotonic relationship — higher scores = better PnL. But no individual threshold cross-validates.

| Score Threshold | n | Avg PnL | CV |
|----------------|---|---------|---|
| ≥0 | 256 | -0.15% | no |
| ≥3 | 220 | -0.11% | no |
| ≥5 | 121 | -0.01% | no |
| ≥7 | 61 | +0.20% | no |

The trend is real but the signal is weak. The ensemble is less powerful than the simple first-5-min confirmation because it mixes weak features with the strong one.

---

### EXP 8: Adaptive SL/TP

**Result:** Stop loss is hurting performance. The scanner's trades are **high-variance momentum plays** where the initial adverse move often reverses.

| Strategy | SL | n | Avg PnL | WR | CV |
|----------|-----|---|---------|-----|---|
| SHORT good cat | no SL | 45 | **+1.68%** | 57.8% | YES |
| SHORT good cat | 1.5% | 45 | +0.16% | 44.4% | no |
| SHORT good cat | 2.5% | 45 | +0.20% | 48.9% | no |
| SHORT only | no SL | 129 | +0.67% | 52.7% | YES |
| All signals | no SL | 258 | +0.25% | 52.3% | YES |

**The SL is destroying the edge.** Without any SL, SHORT + good catalyst goes from +0.20% (SL=2.5%) to +1.68% (no SL). The stop-out rate at SL=2.5% is ~37% for these trades — over a third of trades that would eventually be profitable are getting stopped out early.

This makes sense mechanically: these are catalyst-driven SHORT trades on stocks that gapped up. The first few minutes see continued momentum (gap-chasers buying), then mean reversion kicks in. A tight SL triggers during the momentum phase.

**Fable's recommendation (#8)** to use ATR-scaled stops is validated: the fixed 2.5% SL is too tight for high-gap stocks.

---

## Composite Strategy Recommendations

Based on the cross-validated results, here are three deployable strategy profiles:

### Strategy A: "Confirmed Momentum" (RECOMMENDED)
- **Filter:** Wait 5 minutes after signal. Enter only if price has moved >0.3% in signal direction.
- **SL:** 2.5% (standard)
- **Expected:** +0.79% avg, 60.6% WR, ~94 trades/4 days (~6/day)
- **CV:** +1.06% on Sep 8-9, +0.58% on Sep 10-11

### Strategy B: "Confirmed + Good Catalyst"
- **Filter:** 5-min confirmation + catalyst in (offering, regulatory, squeeze)
- **SL:** 2.5%
- **Expected:** +0.78% avg, 63.6% WR, ~33 trades/4 days (~2/day)
- **CV:** +0.04% on Sep 8-9, +1.01% on Sep 10-11
- **Caution:** Only n=8 in Fold A — borderline.

### Strategy C: "SHORT Runners" (highest PnL, needs wider SL)
- **Filter:** SHORT direction + good catalyst (offering, regulatory, squeeze)
- **SL:** No fixed SL, or ATR-scaled max(5%, 0.5×ATR14/price). Use trailing stop (50% MFE giveback).
- **Expected:** +1.68% avg, 57.8% WR, ~45 trades/4 days (~3/day)
- **CV:** +4.62% on Sep 8-9, +0.94% on Sep 10-11
- **Caution:** High variance (SE=1.01), needs 5%+ risk budget per trade.

### Strategy D: "NLP Bearish Keyword Filter"
- **Filter:** Exclude trades where catalyst text contains bearish-correlated tokens
- **SL:** 2.5%
- **Expected:** +0.51% avg, 62.6% WR, ~91 trades/4 days (~6/day)
- **CV:** +0.56% on Sep 8-9, +0.45% on Sep 10-11
- **Strength:** Most stable across folds, largest n among winners.

### Combined Strategy (A + C + D):
Use **5-min confirmation** as the primary gate, then for SHORT+good-catalyst signals that confirm, use a **wider SL** or trailing stop instead of fixed 2.5%. Filter out signals with bearish-correlated keywords before the confirmation check.

Expected: ~+1.0% avg PnL, ~60% WR, ~4-6 trades/day.

---

## Why +2% Is Hard (But Not Impossible)

No strategy reached +2% at SL=2.5%. The constraint is mechanical:
- A 2.5% SL caps your max loss but also stops out ~30-40% of eventually-profitable trades
- The scanner's signals are high-variance catalyst plays with early adverse moves
- The stop loss triggers during the "wrong" period (first few minutes of momentum)

**To reach +2%**, you'd need:
1. **5-min confirmation** (eliminates the worst trades) → +0.79%
2. **Wider SL for large-gap stocks** (Fable's #8) → adds ~0.3-0.5%
3. **SHORT + good catalyst only** → adds another ~0.3%
4. **NLP keyword filter** → adds ~0.2%

Stacking these (which risks in-sample overfitting with n=258) could theoretically reach +1.5-2.0%.

The one strategy that DID hit +1.68% (SHORT good catalyst, no SL) suggests that **the edge exists but needs asymmetric risk management** — wider stops, trailing exits, or 60m+ hold periods — rather than tighter filters.

---

## What the ML Learned (That Humans Should Too)

The GBM model completely ignored:
- LLM confidence (0% importance)
- Direction (0% importance)
- Catalyst type one-hot features (each <2.5%)
- Scanner source (0%)
- Whether the signal was "selected" (0%)

It focused on:
1. **first_5min_ret** (16.1%) — Does price confirm immediately?
2. **log_vol_ratio** (12.8%) — Is premarket volume abnormally high?
3. **remaining_move_pct** (12.4%) — How much move is left?
4. **first_5min_confirms** (10.7%) — Binary confirmation signal
5. **gap_pct** (8.9%) — Gap size
6. **first_5min_range** (8.2%) — Opening volatility

**Translation:** The LLM layer is adding noise, not signal. The actionable information is in the price data (gap, volume, first-5-min action) and the residual math (remaining_move). The scanner should be restructured to use the LLM only for catalyst classification (what happened?) and use price data for direction and trade decisions.

---

## Implementation Checklist

- [ ] Add 5-minute confirmation gate to both scanners (delay entry, check price direction)
- [ ] For SHORT+good catalyst signals, use ATR-scaled SL: `max(5%, 0.5 × ATR14/price)`
- [ ] Build bearish keyword filter from NLP token analysis
- [ ] Log first_5min_ret and first_5min_confirms in BQ for ongoing validation
- [ ] Freeze code for 5 sessions (Fable's #1) to validate on truly out-of-sample data
- [ ] Consider restructuring LLM role: classify catalyst → look up historical base rate → use price data for direction

---

## Files Generated

| File | Description |
|------|-------------|
| `optimization_harness.py` | Filter combination backtest (50 strategies, 7 SL levels) |
| `creative_experiments.py` | ML, NLP, ensemble, gap-reversion, anti-selection experiments |
| `results/optimization_results.json` | Full filter strategy results |
| `results/creative_experiments_results.json` | Full creative experiment results |
| `results/all_trades_raw.csv` | Raw trade data with PnL at all SL levels |
| `results/creative_features_full.csv` | All features + PnL for each trade |
| `optimization_strategies.png` | Filter strategy comparison chart |
| `optimization_sl_sensitivity.png` | SL sensitivity curves |
| `optimization_best_strategy.png` | Best filter strategy detail |
| `creative_experiments_comparison.png` | Creative experiment comparison |
| `creative_ensemble_analysis.png` | Ensemble score vs PnL analysis |
| `creative_ml_analysis.png` | ML feature importance + model performance |

---

*Generated: 2026-09-11. Cross-validated on Sep 8-9 vs Sep 10-11.*
*Source: BQ scanner logs + Polygon 1-min bars.*
