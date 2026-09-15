# Scanner Optimization — Experiment Changelog

All experiments conducted on 316 trades across 4 dates (2026-09-08 to 2026-09-11), premarket + discovery scanners.

---

## V1 — Initial Optimization (2026-09-11)

**Goal:** Find profitable trading strategies from scanner signals.  
**Report:** `OPTIMIZATION_REPORT.md`, `LEAKAGE_FIX_REPORT.md`  
**Scripts:** `creative_experiments.py`

### V1a: 175 Strategies Tested (pre-leakage-fix)

| # | Experiment | Method | Result | CV? |
|---|-----------|--------|--------|-----|
| EXP 1 | ML Classifier (GBM/RF/Logistic) | Train on LLM features, predict PnL | GBM/RF positive at moderate thresholds | YES (inflated) |
| EXP 2 | Catalyst-Based Filters | Filter by catalyst_type (good vs bad) | SHORT + good catalyst +1.68% | YES |
| EXP 3 | Confidence Thresholds | Filter by LLM confidence > X | Confidence = 0% importance, useless | NO |
| EXP 4 | NLP Keyword Mining | Extract bullish/bearish keywords from catalyst text | "No bearish keywords" filter: +0.51% | YES |
| EXP 5 | Ensemble Scoring | Combine ML + filters into composite | Marginal improvement over single filters | YES |
| EXP 6 | 5-Min Confirmation | Wait for price to confirm direction in first 5 min | +0.79%, 60.6% WR (LEAKED) | YES (inflated) |
| EXP 7 | Gap Reversion | Fade all gaps regardless of LLM direction | Anti-selection: flip direction +0.44% | YES |
| EXP 8 | Adaptive Stop-Loss | Vary SL by volatility/gap size | No SL beats all SL levels | YES |

### V1b: Leakage Fix — 4 Sources Found

| # | Leakage Source | Impact | Fix |
|---|---------------|--------|-----|
| L1 | ML features using entry_price (first_5min_ret/range/confirms) | 35% of GBM importance was leaky features | Removed from FEATURE_COLS |
| L2 | 5-min confirmation double-counting t+5 move in PnL | +0.79% → near zero after fix | Re-simulated PnL from t+5 forward |
| L3 | NLP tokens mined on all data, not fold-local | Token selection biased by test fold | Retrained per-fold |
| L4 | Anti-selection using -pnl without SL re-simulation | PnL flip without re-running stop-loss | Re-simulated with correct direction |

### V1 Surviving Strategies (post-fix)

| Strategy | Avg PnL | WR | n | CV |
|----------|---------|-----|---|---|
| SHORT + good catalyst, no SL @60m | +1.68% | 57.8% | 45 | YES |
| SHORT only, no SL @60m | +0.67% | 52.7% | 129 | YES |
| All signals, no SL @60m | +0.25% | 52.3% | 258 | YES |

---

## V2 — 10 New Directions (2026-09-11)

**Goal:** Test deeper questions about signal structure, horizons, and feature value.  
**Report:** `V2_EXPERIMENT_RESULTS.md`  
**Scripts:** `v2_data_prep.py`, `v2_exp_a_stops_horizon.py`, `v2_exp_b_counterfactual.py`, `v2_exp_c_magnitude_stability.py`, `v2_exp_d_orb_ticker_earnings.py`

| # | Experiment | Question | Result | Actionable? |
|---|-----------|----------|--------|-------------|
| S1 | Stop Design | MAE-optimal stop level? | Best SL (1.5%): -0.09%, No SL: +0.01%. Every stop net negative. | NO — no stop works |
| S2 | Horizon Extension | Does edge grow at longer horizons? | **SHORT T+2: +4.42%, 84% WR (n=50).** Monotonically improves 30m→T+2. CAVEAT: doesn't CV at multi-day. | YES — test multi-day holds |
| S3 | Counterfactual Universe | Does LLM select better than random gappers? | 3,628 gappers >3%, scanner flagged 93 (2.6%). Non-flagged gap-fade: +4.91%. **LLM adds no selection value.** | YES — rebuild universe |
| S4 | Cross-Sectional Ranking | Can features rank trades within a day? | No consistent signal. remaining_move_pct is anti-signal (-0.94% L/S). | NO |
| S5 | Magnitude (no direction) | Does |expected_move| predict |realized|? | |gap_pct|→|realized| r=+0.56. |expected_move|→|realized| r=+0.28. Confidence worthless. | YES — use gap for sizing |
| S6 | Stability (n_runs) | Does consistency across versions predict quality? | n_runs>=4: +0.571%, CV=YES (n=37). | YES — use as filter/sizer |
| S7 | PM Shape | Does premarket retrace pattern matter? | "Held" PM retrace best (+0.673%, 55% WR). pm_vol significant r=+0.198. | MAYBE |
| S8 | ORB Direction | Does opening range breakout direction matter? | **ORB+LLM agree: +1.30%, 61% WR (n=148) vs disagree -1.53%. Strongest filter.** | YES — use as entry qualifier |
| S9 | Ticker History | Does past scanner accuracy predict future? | Weak signal r=+0.065. | NO |
| S10 | Earnings | Are earnings trades different? | -0.61% avg, worsens to -2.38% at T+3. Even beat+LONG: -1.60%. | YES — exclude |

### V2 Key Insight

**The LLM is not a stock picker — it's a catalyst classifier and volatility forecaster.** The real edge is gap-fade (SHORT gappers), filtered by ORB agreement, with no stop-loss.

---

## V3 — Filters, Sizing, and Universe Design (2026-09-12)

**Goal:** Test specific filter combinations, position sizing approaches, and pure price-screen alternatives.  
**Report:** `V3_EXPERIMENT_RESULTS.md`  
**Scripts:** `v3_data_prep.py`, `v3_exp_e1_e2_stops_orb.py`, `v3_exp_e3_e6_e10_catalysts_sizing.py`, `v3_exp_e4_e7_pricescreen_regime.py`, `v3_exp_e5_e8_e9_short_combo.py`

| # | Experiment | Question | Result | Actionable? |
|---|-----------|----------|--------|-------------|
| E1 | ATR Trailing Stops | Do trailing stops from HWM help? | Every (k, horizon) combo negative vs no-SL. Best k=1.5 30m: -0.06% vs no-SL +0.16%. Hybrid trail+T2 for SHORT: +1.48% but driven entirely by T+2 hold. | NO — stops still destroy value |
| E2 | ORB Refinement | Better ORB timeframe/threshold? | **2min ORB: largest spread +2.99%** (agree +1.04% vs disagree -1.95%, n=44/26). Current 5min moderate +1.12%. Volume filter too aggressive. | MAYBE — 2min worth testing, LOW-N |
| E3 | Catalyst Breakdown | Which catalyst types work? | Good: offering (+2.30%), unknown (+1.22%). Bad: deal (-1.55%), macro (-0.97%), earnings (-0.74%). Good-catalyst-only: **+0.92% at 60m** (n=120). | YES — exclude earnings + bad types |
| E4 | Pure Price Screen | Can price/volume screen replace LLM? | 1,810 gappers >3% vs 316 LLM-flagged. Universe gap-fade: +0.326% vs LLM: +0.014%. **Non-earnings + ORB confirms: +1.548%, 64% WR (n=111).** | YES — best filter combo found |
| E5 | SHORT Entry Timing | Open vs 10:30 vs ORB break entry? | **Market open (09:30): +4.98% T+2, 85.7% WR.** 10:30: +4.34%. ORB break: +3.52%. Waiting reduces returns. | YES — enter at open |
| E6 | n_runs Sizing | Weight by consistency count? | n_runs>=4: +0.57% at 60m. 2.0x weighting: Sharpe 0.71→1.12. Hurts at T+2. | YES (intraday only) |
| E7 | Market Regime | Does SPY trend affect performance? | Trending down: SHORT +0.329%, LONG -0.648%. r=-0.884 SPY vs PnL but n=4 (not significant). | NOT YET — need more data |
| E8 | PM+ORB Combo | Do PM retrace and ORB stack? | ORB dominates. SHORT+ORB agrees: **+6.45% T+2, Sharpe 0.881** (n=28). PM adds nothing on top. | YES — ORB is sufficient |
| E9 | Sympathy Tickers | Trade related tickers with smaller gaps? | Follow tickers cluster around breakeven. Lead has edge but extreme variance. | NO — inconclusive |
| E10 | Magnitude Sizing | Size by |expected_move| or |gap_pct|? | expected_move sizing HURTS (r=-0.16). **Gap-pct sizing: Sharpe 0.05→0.75.** SHORT gap-sizing: Sharpe 0.81→1.78. | YES — size by gap, not expected_move |

### V3 Key Insight

**Non-earnings + ORB confirms direction = +1.548%, 64% WR** is the single strongest filter combination. Gap-pct is a far better sizing signal than the LLM's expected_move.

---

## V4 — Whipsaw & Entry Timing Deep Dive (2026-09-12)

**Goal:** Characterize whipsaw behavior and optimize SHORT entry timing.  
**Report:** `V4_EXPERIMENT_RESULTS.md`  
**Scripts:** `v4_exp_e11_e15_whipsaw.py`, `v4_exp_e12_e13_commitment_range.py`, `v4_exp_e14_time_window.py`, `v4_exp_e16_e19_timing_sweep.py`, `v4_exp_e17_e18_dynamic_entry.py`

| # | Experiment | Question | Result | Actionable? |
|---|-----------|----------|--------|-------------|
| E11 | Whipsaw Characterization | How much whipsaw in our universe? | Median 3 OR-midpoint crosses in 30 min. Commitment in 6 min median (mean 8.7). SHORT slightly slower (9.4 min). Only 1% never commit. | Descriptive |
| E12 | N Consecutive Candles | Does requiring N closes below OR improve SHORT? | **N=2: +5.36% T+2, Sharpe 0.755** (vs open 0.421). Small PnL delta (+0.38pp) but best Sharpe. Higher N adds delay without proportional gain. | MAYBE — Sharpe improvement |
| E13 | Range Expansion Filter | Are tight ORBs cleaner breaks? | **Hypothesis INVERTED:** Wide ORBs (>1x ATR) have best PnL (+2.14% 60m). ORB signal strongest on wide-range stocks (spread +6.05pp vs +1.99pp tight). r=+0.32 width vs PnL. | YES — prefer wide ORBs |
| E14 | Time-Based Window | Check direction at fixed time, exit if wrong? | Perfect classifier: HOLD +4.89% (100% WR) vs EXIT -5.18% (0% WR) at 10:30. But **40-60% stay in ambiguous zone** (inside OR). Portfolio benefit marginal. | NO — ambiguous zone too large |
| E15 | Whipsaw vs Outcome | Does whipsaw intensity predict T+2? | **No correlation** (r=-0.107, p=0.253). Only predictor: max excursion below OR floor (r=+0.285, p=0.002). R²=0.08. Whipsaw is noise. | NO — don't filter on whipsaw |
| E16 | Entry Timing Sweep | Minute-by-minute optimal SHORT entry? | **09:30 is optimal for T+2.** PnL monotonically declines with delay (+5.34% at 09:30 → +4.07% at 10:30). Sharpe improves slightly late due to variance compression. | YES — enter at open |
| E17 | Dynamic Exhaustion | Does vol-collapse or range-stabilization signal work? | Vol-collapse 0.3x: +5.55%, Sharpe 0.750, 20.5 min delay. Range stabilization: worse than baseline. | MAYBE — small edge |
| E18 | Local Price Extremes | Enter SHORT at local highs during whipsaw? | **Rule 1 (first local high): +6.10%, 88.4% WR, 86% capture rate.** +1.68pp over baseline. Captures 69% of theoretical ceiling. Rule 3 better but only 36% capture. | **YES — best entry rule** |
| E19 | Volatility-Stratified | Does optimal entry vary by OR width? | **Hypothesis inverted:** Wide OR → enter immediately (fade is fast). Tight OR → waiting helps (10:25 optimal). Normal/Wide buckets LOW-N. | YES — condition on OR width |

### V4 Key Insight

**E18 Rule 1 (first local high) is the single best entry improvement found.** Wait for price to peak above 3 neighboring bars (~7-8 min after open), then enter SHORT. Lifts T+2 from +4.42% to +6.10%. Whipsaw is noise, not signal — don't avoid it, exploit it by entering at the bounce peak.

---

## V5 — Strategy Backtest (2026-09-13)

**Goal:** Formalize V1–V4 findings into 4 concrete strategies with multi-hold, multi-layer architecture. Run on full gapper universe (9,628 tickers) with finite-capital portfolio PnL.  
**Report:** `BACKTEST_RESULTS_V1.md` (also pushed to `github.com/yaninaross/test4`)  
**Scripts:** `strategy_backtest.py`  
**Data:** 9,628 gappers × 4 dates, 372 LLM-flagged, 9,327 minute bars fetched via Polygon, 217 tickers with float data

### 4 Strategies Tested

| ID | Universe | Gap | Entry | ORB | Raw Trades |
|----|----------|-----|-------|-----|------------|
| BASELINE_V1 | LLM + ORB | > 3% | Local high | Required | 8 |
| B2_NO_ORB | LLM only | > 3% | Local high | No | 24 |
| B3_EOD_ENTRY | LLM + ORB | > 3% | EOD close | Required | 6 |
| B1_WIDE_SHORT | All > $5 | > 1.5% | Local high | No | 2,398 |

### 6 Feature Layers

L0 (raw) → L1 (float < 10M exclude) → L2 (PM retrace > 0.8 exclude) → L3 (sweet spot 0.2–0.5) → L4 (breaking catalyst exclude) → L5 (L1 + L2 + L4 combined)

### 3 Hold Periods

30m (bar[29] close), 60m (bar[59] close), T+2 (2-day close)

### Master Table: 57 Cells (4 strategies × ~6 layers × 3 holds)

★ **Live Trading Candidates (CI > 0 AND t/d >= 1.0):**

| Strategy | Layer | Hold | n | avg% | Sharpe | WR | CI range | Compound% |
|----------|-------|------|---|------|--------|----|----------|-----------|
| B1_WIDE_SHORT | L5_combined | T+2 | 1,013 | +3.34% | 0.484 | 73% | +2.91 to +3.76 | +6.60% |
| B1_WIDE_SHORT | L1_float | T+2 | 1,016 | +3.33% | 0.483 | 73% | +2.90 to +3.75 | +6.59% |
| B1_WIDE_SHORT | L0_raw | T+2 | 1,017 | +3.28% | 0.462 | 73% | +2.84 to +3.71 | +6.50% |
| B1_WIDE_SHORT | L3_sweet | T+2 | 162 | +2.98% | 0.407 | 64% | +1.85 to +4.10 | +5.48% |

**High-alpha but unconfirmed (need more data):**

| Strategy | Layer | Hold | n | avg% | Sharpe | WR | Issue |
|----------|-------|------|---|------|--------|----|-------|
| BASELINE_V1 | L5 | T+2 | 3 | +18.11% | 0.695 | 100% | n=3, 1 day only |
| B2_NO_ORB | L5 | T+2 | 7 | +10.67% | 0.633 | 86% | CI spans zero |
| B2_NO_ORB | L5 | 30m | 13 | +3.94% | 0.396 | 46% | CI spans zero |
| BASELINE_V1 | L4 | T+2 | 4 | +12.66% | 0.530 | 75% | n=4 |

### V5 Key Findings

1. **B1_WIDE_SHORT T+2 is the ONLY CI-confirmed edge** — shorting every gap-up > 1.5% (> $5) and holding 2 days yields +3.28% avg, 73% WR, over 1,017 trades. The CI is tight (+2.84 to +3.71) due to massive sample size.

2. **30m and 60m are DEAD** — no strategy at any layer shows CI > 0 at intraday horizons. B1 intraday is actively negative (-0.20% at 30m, -0.23% at 60m). The SHORT gap-fade is exclusively a swing trade.

3. **Float filter adds value for LLM strategies** — removing low-float tickers improved B2 30m from +0.76% to +2.30% (5 removed trades had avg PnL of -5.06%). Barely affects B1 because float data only covers 217 of ~9,600 tickers.

4. **Feature layers don't transfer from LLM to broad universe** — L1/L2/L5 barely move B1 numbers because the filters were calibrated on the 253-ticker LLM universe. L3 sweet spot dramatically cuts B1 trade count (2,398 → 395) without improving quality.

5. **B3 Sharpe degrades monotonically** — from 0.599 (L0) to 0.251 (L5) as each filter removes profitable trades from n=6.

6. **Composite score shows monotonic signal for B1 T+2** — Q1 (+2.27%) → Q2 (+3.28%) → Q3 (+3.52%) → Q4 (+4.29%). Spread +2.02%, ratio 1.89x (just under 2.0x validation threshold). Using Q3+Q4 only (~505 trades, 126/day) yields +3.90% avg.

7. **LLM strategies fire on only 2 of 4 days** — BASELINE/B2/B3 only trade on Sep 8 and Sep 11. Portfolio is silent on Days 2 and 3.

### V5 Data Limitations

- 4 trading days only (Sep 8–11)
- T+2 data for only 2 entry days (Sep 10/11 blocked by weekend)
- Float data covers 217 of ~9,600 tickers
- No transaction costs, slippage, or borrow costs modeled

---

## Cumulative Findings Across All Versions

### What Works (Confirmed Across Multiple Experiments)

| Finding | First Found | Confirmed By | Strength |
|---------|-----------|--------------|----------|
| SHORT gap-fade is the primary edge | V1 (EXP 2) | V2-S2, V3-E5, V4-E16, **V5-B1** | **CI CONFIRMED (n=1,017)** |
| No stop-loss design helps | V1 (EXP 8) | V2-S1, V3-E1 | Strong (3 rounds of testing) |
| ORB direction is the best filter | V2 (S8) | V3-E4, V3-E8, V4-E14 | Strong (n=111-148) |
| Earnings exclusion helps | V2 (S10) | V3-E3 | Moderate (n=44) |
| LLM adds no selection value | V2 (S3) | V3-E4 | Strong (1,810 vs 316 comparison) |
| Gap-pct best sizing signal | V2 (S5) | V3-E10 | Strong (r=+0.40-0.56) |
| Enter SHORT at open (or first local high) | V3 (E5) | V4-E16, V4-E18, V5 | Strong (monotonic decline) |
| Intraday holds (30m/60m) are dead | V5 | All 4 strategies × all layers | **CI CONFIRMED negative** |
| Float filter improves LLM strategies | V5 | B2 L0→L1: +0.76→+2.30% | Moderate (n=19-24) |
| Wide ORBs are best trades | V4 (E13) | V4-E19 | Moderate (LOW-N on wide bucket) |

### What Doesn't Work (Confirmed Dead Ends)

| Finding | Tested In | Conclusion |
|---------|----------|------------|
| Any stop-loss (fixed, ATR, trailing, hybrid) | V1, V2-S1, V3-E1 | Destroys value. Winners need room. |
| LLM confidence as signal | V1 (EXP 3), V2-S5 | ~0% importance, completely useless |
| LLM direction prediction | V2-S3, V3-E4 | No better than random gappers |
| remaining_move_pct | V2-S4 | Anti-signal (-0.94% L/S) |
| expected_move for sizing | V3-E10 | Negative correlation with PnL (r=-0.16) |
| Cross-sectional ranking | V2-S4 | No consistent signal |
| Ticker history | V2-S9 | Weak (r=+0.065), not actionable |
| Sympathy tickers | V3-E9 | Follow tickers = breakeven |
| Whipsaw intensity as filter | V4-E15 | No correlation with outcome |
| PM retrace on top of ORB | V3-E8 | Redundant — ORB dominates |
| Time-based commitment window | V4-E14 | Ambiguous zone (40-60%) kills benefit |

### Evolution of the Trading Recipe

**V1:** SHORT + good catalyst, no SL → +1.68%, 57.8% WR  
**V2:** Add ORB agreement, extend to T+2 → +4.42%, 84% WR  
**V3:** Add gap-sizing, enter at open, exclude earnings → +4.98%, 85.7% WR  
**V4:** Enter at first local high → +6.10%, 88.4% WR  
**V5:** Formalized backtest: B1_WIDE_SHORT T+2 **CI CONFIRMED** +3.28%, 73% WR (n=1,017). Intraday dead.  

### Final Recipe (V4)

```
UNIVERSE:   All gappers > 3% (not LLM-selected)
            Exclude earnings
            Exclude bad catalysts (deal, macro) if available
FILTER:     ORB agrees with gap-fade direction (5min)
            Prefer wide ORB stocks (> 0.5x ATR)
DIRECTION:  SHORT (fade the gap)
ENTRY:      First local high after open (3-bar window, ~7-8 min)
            Fallback: market open if no local high by 09:45
HOLD:       T+2 for SHORT swing
SIZING:     Weight by |gap_pct| / median(|gap_pct|)
            Upweight n_runs >= 4 (1.5-2.0x) for intraday only
STOP-LOSS:  None
```

---

## Experiment Count Summary

| Version | Experiments | Scripts | Charts | Key Discovery |
|---------|------------|---------|--------|---------------|
| V1a | 8 (175 strategies) | 1 | 6 pages | 5-min confirmation (leaked) |
| V1b | 4 leakage fixes | 1 | — | SHORT + good catalyst survives |
| V2 | 10 (S1-S10) | 5 | 10 | SHORT T+2 = +4.42%, ORB filter |
| V3 | 10 (E1-E10) | 5 | 10 | No-earnings+ORB = +1.548%, gap sizing |
| V4 | 9 (E11-E19) | 5 | 12 | First local high entry = +6.10% |
| V5 | 4 strategies × 6 layers × 3 holds (57 cells) | 1 | 4 | B1_WIDE T+2 CI CONFIRMED +3.28% |
| **Total** | **41 experiments + 57-cell backtest** | **18 scripts** | **42+ charts** | |

---

## File Index

| File | Contents |
|------|----------|
| `OPTIMIZATION_REPORT.md` | V1a: 175 strategies, 8 experiments (pre-fix numbers) |
| `LEAKAGE_FIX_REPORT.md` | V1b: 4 leakage sources identified and fixed |
| `V2_EXPERIMENT_RESULTS.md` | V2: 10 experiments (S1-S10), horizon/counterfactual/ORB |
| `V3_EXPERIMENT_RESULTS.md` | V3: 10 experiments (E1-E10), filters/sizing/universe |
| `V4_EXPERIMENT_RESULTS.md` | V4: 9 experiments (E11-E19), whipsaw/entry timing |
| `SCANNER_COMPLETE_FINDINGS.md` | V1+V2 consolidated findings |
| `BACKTEST_RESULTS_V1.md` | V5: 4-strategy backtest, master table, composite analysis, portfolio PnL |
| `strategy_backtest.py` | V5: Monolithic backtest script (1493 lines), runs as Colab cell |
| `FEATURE_INVESTIGATION_RESULTS.md` | 21-feature investigation summary (F01–F21) |
| `FEATURE_INVESTIGATION_DETAILED.md` | Expanded feature investigation with full quartile tables |
| `EXPERIMENT_CHANGELOG.md` | This file — full experiment history |
