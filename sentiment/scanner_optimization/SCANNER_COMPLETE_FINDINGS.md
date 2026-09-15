# Scanner Optimization — Complete Findings

**Period:** Sep 8-11 2026 (4 trading days)  
**Trades:** 258-316 (depending on filter/dedup method)  
**Versions:** V2, V3, V3.3, V3.5, V3.6, V3.7, V4.0  
**Cross-validation:** Fold A (Sep 8-9) vs Fold B (Sep 10-11)  
**CAVEAT:** Fold-CV = version-CV — Fold A is V2-V3.3, Fold B is V3.5-V4

This document consolidates all findings from two investigation rounds:
- **V1** (optimization_harness + creative_experiments): 175 filter strategies, 4 leakage sources found and fixed
- **V2** (scanner_experiments_v2): 10 new experiments testing fundamentally different questions

---

## Executive Summary

The LLM scanner is **not a directional predictor**. It predicts volatility magnitude (|expected_move| → |realized| r=+0.28) and classifies catalysts, but its direction calls and selection decisions add no value over a simple price screen.

**What works:**
1. Gap-fade is a market-wide phenomenon (+4.91% avg for gap>3% SHORT)
2. ORB + LLM agreement is the strongest filter found (+1.30%, 61% WR)
3. SHORT signals are swing signals, not scalps (T+2: +4.42%, 84% WR)
4. HIGH consistency signals (n_runs>=4) cross-validate (+0.571%)

**What doesn't work:**
- LLM direction calls (confidence ~0% importance)
- LLM selection (non-flagged gappers outperform flagged ones)
- Any fixed stop-loss (all levels net negative)
- Earnings trades (-0.61%, worsens with horizon)
- ML classifiers on scanner features (all negative after leakage fix)
- 5-min confirmation as PnL filter (was leakage)

---

## Part 1: V1 Investigation — Leakage Discovery

### 4 Leakage Sources Found and Fixed

1. **ML features (EXP 1):** `first_5min_ret/range/confirms` measured from entry_price — same price used for PnL. 27% of GBM importance was mechanical correlation.
2. **5-min confirmation (EXP 6):** PnL from t=0 but filter checked t+5. The +0.5% confirmation move was double-counted. Rebased: +0.79% → +0.04%.
3. **NLP tokens (EXP 5):** Tokens mined from all data, then "cross-validated" on same data. Fixed with fold-local mining.
4. **Anti-selection (EXP 3):** `pnl_flipped = -pnl` doesn't simulate inverse trade's SL behavior. Demoted to diagnostic.

### V1 Surviving Strategies (post-fix)

| # | Strategy | Avg PnL | WR | n | CV | Note |
|---|----------|---------|-----|---|---|------|
| 1 | SHORT good catalyst, no SL @60m | **+1.68%** | 57.8% | 45 | YES | SE=1.01, high variance |
| 2 | SHORT only, no SL @60m | **+0.67%** | 52.7% | 129 | YES | Most robust |
| 3 | NLP no_bearish lift>5 | **+0.49%** | 54.5% | 44 | YES | n=7 in Fold B |
| 4 | DIAG flip selected direction | **+0.44%** | 67.9% | 56 | YES | Diagnostic only |
| 5 | All signals, no SL @60m | **+0.25%** | 52.3% | 258 | YES | Baseline improves without SL |

### What Died After Fix

- **5-min confirmation:** ALL variants fail CV after rebasing
- **ML classifiers:** ALL negative on test fold without leaky features
- **Ensemble micro-signals:** No threshold cross-validates
- **has_bullish NLP:** All negative

### Clean GBM Feature Importance

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | log_vol_ratio | 20.6% |
| 2 | gap_pct | 16.4% |
| 3 | remaining_move_pct | 11.8% |
| 4 | atr_gap | 9.8% |
| 5 | expected_move_pct | 9.5% |
| — | confidence | ~0% |
| — | direction_int | ~0.9% |

---

## Part 2: V2 Experiments — 10 New Directions

### S1: Stop Design (MAE/MFE Analysis)

**Optimal stop from MAE analysis:** 1.98% (80th pct of winners' MAE) = 0.36 x ATR.

Every stop level is net negative:

| Stop Level | n | Avg PnL | Med | WR |
|-----------|---|---------|-----|-----|
| Fixed 1.5% | 272 | -0.089% | -0.381% | 40% |
| Fixed 2.5% | 272 | -0.184% | -0.153% | 44% |
| Fixed 5.0% | 272 | -0.210% | -0.030% | 48% |
| ATR x 0.5 | 272 | -0.300% | -0.082% | 46% |
| ATR x 0.75 | 272 | -0.243% | -0.025% | 48% |
| **No SL** | 272 | **+0.014%** | +0.065% | 51% |

Winners MAE P80: LONG -1.73%, SHORT -2.23%.  
Winners avg MFE: +5.42% vs losers +1.18%.  
Losers avg MAE: -5.08% vs winners -1.94%.

**Interpretation:** Stops kill winners more than they protect from losers. Fat-tailed winners need room. But "no SL" on 4 days doesn't prove tail safety — ATR trailing stops are the real answer.

### S2: Horizon Extension (MOST IMPORTANT RESULT)

SHORT signals get dramatically better at longer horizons:

| Horizon | All Avg | All WR | SHORT Avg | SHORT WR | SHORT CV |
|---------|---------|--------|-----------|----------|----------|
| 30m | +0.16% | 54% | **+0.40%** | 56% | **YES** |
| 60m | +0.01% | 51% | +0.31% | 52% | YES |
| 2h | +0.23% | 53% | **+0.81%** | 56% | **YES** |
| EOD | +0.26% | 50% | +0.63% | 56% | no |
| T+1 | -0.04% | 50% | **+1.97%** | 69% | no |
| T+2 | -0.32% | 49% | **+4.42%** | 84% | no |
| T+3 | -0.52% | 46% | **+4.27%** | 73% | no |

Feature correlation vs horizon:
- `gap_pct` strongest across all: r=+0.41 to +0.46
- `log_vol_ratio` improves: +0.15 (60m) → +0.35 (T+1)
- `remaining_move_pct` becomes anti-correlated: -0.12 (60m) → **-0.35** (T+2)

**CAVEATS:** T+2/T+3 don't cross-validate. n drops to 50/44. Could be regime artifact. remaining_move_pct anti-correlation suggests LLM expected move may be actively harmful for swing trades.

### S3: Counterfactual Universe (MOST IMPORTANT QUESTION)

**Does the LLM add value over a price screen?**

- Total gappers (>3% gap) across 4 days: **3,628**
- Scanner flagged: **93** (2.6% of gappers)
- Non-flagged: 3,535

| Group | n | Avg PnL | Med | WR |
|-------|---|---------|-----|-----|
| Flagged: all signals | 272 | +0.014% | +0.065% | 51% |
| Flagged: SHORT gap>3% | 25 | +2.779% | +0.554% | 64% |
| **Non-flagged: gap-fade** | **60** | **+4.914%** | **+1.699%** | **60%** |
| **Non-flagged: SHORT gap-ups** | **37** | **+5.679%** | **+1.802%** | **65%** |

Gap-fade is profitable market-wide: gap-up stocks fade +1.86% on average (n=1,728).

**Conclusion:** LLM selection provides NO edge. Non-flagged gappers outperform flagged ones. The gap-fade effect is market-wide and doesn't require LLM classification. Scanner's value (if any) is in catalyst classification, not ticker selection.

### S4: Cross-Sectional Ranking

Daily tercile long/short (dollar-neutral, removes day effect):

| Feature | Avg L/S Return | Consistent? |
|---------|---------------|-------------|
| remaining_move_pct | **-0.944%** | Mixed (anti-signal) |
| expected_move_pct | -0.352% | Mixed |
| log_vol_ratio | +0.223% | Mixed |
| gap_pct | +0.355% | Mixed |
| abs_gap_pct | +0.389% | Mixed |

No feature produces consistent daily cross-sectional returns. `remaining_move_pct` is the worst — bottom tercile outperforms top in 3/4 days.

### S5: Magnitude Without Direction

**LLM predicts volatility, not direction.**

| Predictor | Target | r | p-value |
|-----------|--------|---|---------|
| |gap_pct| | |realized_60m| | **+0.560** | <0.0001 |
| atr_gap | |realized_60m| | **+0.456** | <0.0001 |
| log_vol_ratio | |realized_60m| | **+0.346** | <0.0001 |
| |expected_move| | |realized_60m| | **+0.281** | <0.0001 |
| |expected_move| | MFE+|MAE| | **+0.250** | <0.0001 |
| confidence | |realized_60m| | +0.054 | 0.3792 |
| |remaining_move| | |realized_60m| | +0.039 | 0.5167 |

High |expected_move| tercile: 2.5x the |PnL| and 2x the total excursion of low tercile. LLM correctly identifies which stocks will move most — just can't tell which way.

**Confidence and remaining_move_pct are worthless** as predictors.

### S6: Stability Across Runs

| Filter | n | Avg PnL | Med | CV |
|--------|---|---------|-----|-----|
| Stable (score>=0.8) | 215 | 0.000% | +0.020% | no |
| Unstable (<0.8) | 57 | +0.068% | +0.197% | YES |
| n_runs >= 3 | 78 | +0.276% | +0.322% | **YES** |
| n_runs >= 4 | 37 | **+0.571%** | +0.261% | **YES** |

High-consistency signals (n_runs >= 3-4) perform better and cross-validate. But "stable" (same direction) isn't better than "unstable" — the scanner reinforces its initial pick across runs rather than converging on truth.

### S7: Premarket Path Shape

| PM Retrace | n | Avg PnL | WR |
|-----------|---|---------|-----|
| Held (<0.2) | 80 | **+0.673%** | 55% |
| Mild (0.2-0.5) | 74 | -0.464% | 42% |
| Moderate (0.5-0.8) | 60 | -0.410% | 52% |
| Full retrace (>0.8) | 58 | +0.154% | 57% |

PM volume: only significant correlate (r=+0.198, p=0.001).

Direction interaction:
- SHORT + held: +1.030%, 59% WR
- SHORT + full retrace: +0.900%, 70% WR

### S8: Opening Range Breakout (STRONGEST FILTER FOUND)

ORB triggered on 315/316 trades.

| Condition | n | Avg PnL | WR |
|----------|---|---------|-----|
| All ORB (price direction) | 276 | +0.282% | 44% |
| LLM direction (same trades) | 271 | +0.013% | 51% |
| **ORB agrees with LLM** | **148** | **+1.296%** | **61%** |
| ORB disagrees with LLM | 123 | -1.530% | 39% |

**The 2.8% spread between agreement and disagreement is the strongest filter found.** When ORB (pure price) and LLM agree on direction, the trade works. When they disagree, it loses.

This reframes the LLM's role: not a directional predictor — a **qualifier** that confirms or denies a price-based signal.

### S9: Ticker Reaction History

Weak signal. Historical continuation rate vs PnL: r=+0.065 (not significant).

| Group | n | Avg PnL | WR |
|-------|---|---------|-----|
| Faders (cont_rate<40%) | 64 | -0.108% | 48% |
| Runners (cont_rate>60%) | 43 | +0.466% | 53% |

Not actionable at this sample size.

### S10: Earnings Module

**Earnings trades are net negative and should be excluded.**

| Group | n | Avg PnL | Med | WR |
|-------|---|---------|-----|-----|
| **Earnings** | **77** | **-0.613%** | -0.284% | 43% |
| Non-earnings | 195 | +0.262% | +0.177% | 54% |
| Beat | 15 | -1.304% | -0.697% | 40% |
| Miss | 6 | -1.397% | -0.910% | 33% |

Worsens at longer horizons: 60m -0.61% → T+3 -2.38%.

Even when LLM's rule direction (beat→LONG, miss→SHORT) agrees with LLM's direction call: -1.597% avg. Earnings gap dynamics are fundamentally different from other catalysts.

### Version Breakdown

| Version | n | Avg PnL | Dates |
|---------|---|---------|-------|
| V2 | 65 | +0.502% | 09-08 |
| V3 | 43 | -0.102% | 09-08, 09-09 |
| V3.5 | 29 | -0.057% | 09-10 |
| V4.0 | 110 | -0.156% | 09-11 |

V2 drives most of Fold A's outperformance. Confirms fold-CV = version-CV confound.

### Robustness Re-evaluation

| Strategy | n | Avg | Med | Trim | SE | LOO Range | CV |
|----------|---|-----|-----|------|----|-----------|----|
| SHORT+goodcat noSL | 48 | +1.067% | +0.417% | +0.632% | 1.099 | [+0.34, +1.61] | YES |
| (without top trade) | 47 | +0.337% | +0.351% | — | — | — | — |
| SHORT noSL | 137 | +0.307% | +0.081% | +0.136% | 0.511 | — | YES |
| All noSL | 272 | +0.014% | +0.065% | +0.038% | 0.299 | — | no |

SHORT+goodcat drops from +1.067% to +0.337% when the top trade (+35.4%) is removed. LOO range [+0.34, +1.61] — fragile.

---

## Part 3: System Redesign Recommendations

Based on both V1 and V2 findings, the scanner system should be fundamentally restructured:

### 1. Universe Construction
**Current:** LLM flags ~93 tickers across 4 days (2.6% of gappers).  
**Proposed:** Price/volume screen — all stocks gapping >3% with elevated PM volume. The counterfactual proves the LLM adds no selection value.

### 2. Direction Signal
**Current:** LLM direction call (LONG/SHORT) — proven worthless (confidence ~0%, direction_int ~0.9% importance).  
**Proposed:** Opening Range Breakout (5-min range break). When ORB and LLM agree → trade. When they disagree → skip. The 2.8% spread makes this the strongest filter.

### 3. Holding Period
**Current:** 60-minute evaluation horizon with 2.5% SL.  
**Proposed:** SHORT signals are swing signals (2h to T+2). LONG signals should be scalps (30m) or skipped entirely. The 2.5% SL costs ~0.2%/trade; wider stops or trailing ATR-based stops needed.

### 4. LLM Role
**Current:** Directional predictor selecting trades.  
**Proposed:** Catalyst classifier + volatility forecaster. |expected_move| → |realized| r=+0.28 is real signal. Use for position sizing, watchlist ranking, not direction.

### 5. Exclusions
- **Earnings trades:** -0.61% avg, worsens to -2.38% at T+3. Exclude completely.
- **Unknown/empty catalyst:** Consider exclusion (Fable #4 filter).

### 6. Features That Matter vs Don't

**Valuable:**
- `gap_pct` (r=+0.56 with |realized move|) — strongest feature
- `log_vol_ratio` (r=+0.35) — premarket volume is #1 clean ML signal
- `|expected_move|` (r=+0.28) — magnitude predictor
- `atr_gap` (r=+0.46) — ATR-normalized gap
- `n_runs` (consistency filter, n>=4 cross-validates)

**Worthless:**
- `confidence` (~0% importance)
- `remaining_move_pct` (anti-signal at longer horizons, r=-0.35 at T+2)
- `direction` (near-zero importance)
- `selected` (non-selected outperforms selected in multiple cuts)

---

## Part 4: Outstanding Questions

1. **Multi-day SHORT holds:** T+2 +4.42% is striking but doesn't cross-validate and n=50. Need more data and proper walk-forward validation.

2. **ORB+LLM strategy:** The 2.8% agreement/disagreement spread needs backtesting as a standalone strategy with proper risk management.

3. **Data sufficiency:** All conclusions are from 4 trading days. The META requirement is to freeze code and validate on truly out-of-sample data before any system changes.

4. **ATR trailing stops:** The MAE analysis shows winners need room. ATR-based trailing stops (not fixed stops) are the P0 risk management experiment.

5. **Second-order plays:** Sympathy tickers (related_tickers from LLM) are untested — slower, less crowded, potentially higher edge.

---

## Artifacts Index

### Reports
| File | Description |
|------|-------------|
| `SCANNER_COMPLETE_FINDINGS.md` | This consolidated report |
| `V2_EXPERIMENT_RESULTS.md` | V2-only detailed results |
| `LEAKAGE_FIX_REPORT.md` | V1 post-fix analysis |
| `OPTIMIZATION_REPORT.md` | V1 original (PRE-FIX, inflated numbers) |

### Scripts
| File | Description |
|------|-------------|
| `scanner_experiments_v2.py` | Monolithic V2 script (all 10 experiments, 1684 lines) |
| `v2_data_prep.py` | Data fetch + feature computation |
| `v2_exp_a_stops_horizon.py` | S1 (Stop Design) + S2 (Horizon Extension) |
| `v2_exp_b_counterfactual.py` | S3 (Counterfactual) + S4 (Cross-Sectional) |
| `v2_exp_c_magnitude_stability.py` | S5 (Magnitude) + S6 (Stability) + S7 (PM Shape) |
| `v2_exp_d_orb_ticker_earnings.py` | S8 (ORB) + S9 (Ticker) + S10 (Earnings) |
| `creative_experiments.py` | V1 fixed harness (all 4 leakage sources corrected) |
| `optimization_harness.py` | V1 original: 175 strategies x 7 SL levels x 2 horizons |

### Data
| File | Description |
|------|-------------|
| `results/v2_features_full.csv` | 316 trades with all computed features |
| `results/v2_daily_data.json` | Daily bars for 266 tickers |
| `results/v2_spy_daily.json` | SPY daily bars for beta adjustment |
| `results/all_trades_raw.csv` | V1 raw trade data |
| `results/creative_features_full.csv` | V1 feature data with rebased PnL |

---

*Generated: 2026-09-11. Consolidated from V1 (optimization_harness, creative_experiments) and V2 (scanner_experiments_v2) investigation rounds.*
