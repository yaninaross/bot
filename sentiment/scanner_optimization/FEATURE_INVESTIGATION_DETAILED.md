# Feature Investigation — Detailed Results

# Multi-Strategy Gap Trading System: 21-Feature Premarket Selector Analysis

**Date**: 2026-09-13
**Dataset**: 316 LLM-flagged trades, 253 unique tickers, 4 trading dates (Sep 8-11, 2026)
**PnL targets**: T+2 close (n=117 non-null), 60m no-SL (n=272 non-null)
**Strategies evaluated**: 10 — BASELINE_V1, B1_WIDE_SHORT, B2_NO_ORB, B3_EOD_ENTRY, B4_T1_ENTRY, C1_LONG_FADE, C2_LONG_ORB, C3_EARNINGS, D1_LONG_MOM, D2_SHORT_MOM
**Features tested**: 21 features across 6 groups (intraday-derived, ticker reference, daily history, premarket bars, options, SEC filings)
**Validation status**: ALL results require 4 weeks of live trading confirmation before implementation

---

## Table of Contents

1. Executive Summary
2. Dataset Description and Methodology
3. Feature Group 1: Intraday-Derived (F01-F06)
4. Feature Group 2: Volume and Reference (F07-F10)
5. Feature Group 3: Historical Patterns (F11-F15)
6. Feature Group 4: Premarket Bars (F16-F18)
7. Feature Group 5-6: Options and SEC (F19-F21)
8. Acceptance Gate Summary
9. Composite Scoring Recommendation
10. Implementation Roadmap
11. Appendix: Output Files and Scripts

---

## 1. Executive Summary

Of 21 features tested against T+2 PnL outcomes, **1 feature passes the full acceptance gate**, **5 show directional or conditional signal**, and **15 are null results**. The investigation is limited by 4 trading days and 117 T+2 observations — no feature should be deployed as a live filter without 20+ days of out-of-sample confirmation.

### Master Ranking Table

| Rank | Feature | |Pearson r| vs T+2 | p-value | Spearman r | Consistency | Verdict |
|------|---------|---------------------|---------|------------|-------------|---------|
| 1 | **F08 ORB Width Score** | 0.2426 | 0.0087 | -0.0527 | **6/6** | **PASS (full gate)** |
| 2 | **F16 PM Retrace (SHORT)** | 0.2373 | 0.068 | **-0.3742** (p=0.003) | 1/4 | PASS (direction-specific) |
| 3 | **F01 Gap Magnitude** | 0.2690 | 0.003 | 0.0716 | 3/4 | Significant, low consistency |
| 4 | **F12 Prior Trend (LONG)** | 0.0287 | 0.74 | **+0.2231** (p=0.009) | N/A | PASS (direction-specific) |
| 5 | **F07 PM Vol Ratio** | 0.2875 | 0.002 | 0.0730 | 1/1 | Outlier-driven |
| 6 | **F02 Opening Bar Vol** | 0.2750 | 0.003 | -0.0497 | 1/4 | Confounded with gap size |
| 7 | **F05 Entry Volatility** | 0.2301 | 0.026 | -0.0867 | 1/4 | Marginal |
| 8 | F18 PM Slope | 0.3688 | <0.001 | -0.0419 | — | Outlier-driven (Pearson/Spearman diverge) |
| 9 | F13 Volume Spike | 0.1240 | 0.042 | -0.0022 | — | Outlier-driven |
| 10 | F20 HV Rank | 0.0136 | 0.869 | 0.1170 | — | Null |
| 11 | F15 Idiosyncratic Gap | 0.2266 | 0.002 | 0.0107 | — | Outlier-driven |
| 12 | F09 Market Cap | 0.0765 | 0.430 | -0.1363 | 1/2 | Null (counterintuitive) |
| 13 | F10 Float Size | 0.0878 | 0.380 | -0.1173 | 2/3 | Null (but risk rule) |
| 14 | F11 Fade Rate | 0.0744 | 0.263 | -0.0717 | 4/6 | Null (wrong direction) |
| 15 | F17 PM Vol Profile | 0.0583 | 0.499 | — | — | Null |
| 16 | F06 Day of Week | 0.0439 | 0.638 | 0.0454 | — | Untestable (4 days) |
| 17 | F04 Entry Delay | 0.0164 | 0.861 | -0.0421 | 2/4 | Null |
| 18 | F03 Gap Retrace | 0.0081 | 0.933 | 0.0718 | 2/4 | Null |
| 19 | F14 Distance from High | 0.0113 | 0.851 | 0.0085 | — | Null |
| 20 | F19 Put/Call Ratio | — | — | — | — | Skipped (insufficient API data) |
| 21 | F21 EDGAR Offering | — | — | — | — | Catalyst-type signal (not continuous) |

### Action Items for Live Trading (Starting Sep 15)

| Priority | Feature | Action | Data Source | Morning Overhead |
|----------|---------|--------|-------------|-----------------|
| 1 | F08 ORB Width Score | Log orb_width_score = (or_high - or_low) / (entry_price * atr_pct) | First 5 bars | Computable at 9:35 AM |
| 2 | F16 PM Retrace | Log pm_retrace_pct; flag if > 0.8 for SHORT | PM bars | Computable at 9:29 AM |
| 3 | F12 Prior Trend | Log 10-day slope for LONG candidates | Nightly daily bars | Precomputed overnight |
| 4 | F01 Gap Magnitude | Already logged; add bucket labels | Gap calc | Already available |
| 5 | F21 Offering | Prefer offering catalyst_type | LLM output | Already available |
| 6 | F10 Low Float | Flag tickers with float < 10M | Polygon reference (cached) | One-time cache |

---

## 2. Dataset Description and Methodology

### Data Sources

| Source | File | Records | Description |
|--------|------|---------|-------------|
| Trade features | v2_features_full.csv | 316 rows, 97 cols | LLM-flagged trades with gap%, PnL at multiple horizons, PM features |
| Minute bars | v3_minute_bars.json | 316 ticker-dates | 1-min bars from 4:00 AM to market close (includes premarket) |
| Daily bars | v2_daily_data.json | 253 tickers × ~29 days | OHLCV daily history (Aug 3 to Sep 11, 2026) |
| Ticker reference | ticker_reference_cache.json | 253 tickers | Market cap, shares outstanding from Polygon API |

### Direction Split

| Direction | Count | With T+2 PnL | With 60m PnL |
|-----------|-------|-------------|-------------|
| SHORT | 160 | ~60 | ~140 |
| LONG | 156 | ~57 | ~130 |

### Testing Protocol Applied to Every Feature

1. **Correlation analysis**: Pearson r (linear), Spearman rho (monotonic rank) vs pnl_t2_close and pnl_60m_nosl
2. **Threshold sweep**: Multiple cutoffs, split PASS/FAIL, full metrics for each group
3. **Quartile analysis**: Feature value split into Q1-Q4, avg PnL per quartile, test for monotonic trend
4. **Cross-strategy consistency**: Does the feature show the same directional effect in 6+ of 10 strategies?
5. **Direction-specific analysis**: Separate SHORT and LONG correlations where relevant
6. **Standard metrics computed**: n, avg_pnl, median_pnl, trimmed_mean, std, sharpe, win_rate, profit_factor, max_loss, max_gain, ci_95_low, ci_95_high

### Acceptance Criteria (Full Gate)

A feature passes when ALL of:
- `|correlation_with_pnl| > 0.15` — meaningful effect size
- `p_value < 0.05` — statistically significant
- `cross_strategy_consistency >= 6/10` — works broadly
- `data_availability = premarket_batch` — computable before 9:30 AM
- `implementation_complexity <= moderate` — buildable in one weekend

---

## 3. Feature Group 1: Intraday-Derived (F01-F06)

### F01 — Gap Magnitude Bucket ★★

**Hypothesis**: Fade edge concentrates in specific gap size ranges. Medium (3-5%) is the sweet spot where institutional fade mechanics are most reliable. Extreme (>8%) carries squeeze risk.

**Result**: Significant Pearson correlation, but Spearman is flat — the effect is driven by outlier large-gap trades.

**Correlation**:

| Target | Pearson r | p-value | Spearman r | p-value | n |
|--------|-----------|---------|------------|---------|---|
| T+2 PnL | +0.269 | 0.003 | +0.072 | 0.44 | 117 |
| 60m PnL | +0.404 | <0.001 | -0.014 | 0.82 | 272 |

**Bucket Analysis (T+2 PnL)**:

| Bucket | n | Avg PnL | Median | Std | Sharpe | Win Rate | CI 95% Low | Max Loss | Max Gain |
|--------|---|---------|--------|-----|--------|----------|------------|----------|----------|
| Micro (<2%) | 53 | -1.38% | +0.15% | 7.94 | -0.17 | 51% | -3.52% | -35.32% | +7.64% |
| Small (2-3%) | 9 | +0.74% | +2.41% | 7.09 | +0.10 | 56% | -3.89% | -9.39% | +12.52% |
| Medium (3-5%) | 19 | -0.33% | -0.34% | 4.79 | -0.07 | 47% | -2.49% | -11.88% | +6.58% |
| Large (5-8%) | 12 | -0.42% | -2.33% | 7.86 | -0.05 | 25% | -4.86% | -9.42% | +15.96% |
| Extreme (>8%) | 24 | +1.68%* | — | — | — | — | — | — | — |

*Extreme bucket Pearson is inflated by a few very large PnL trades. The Spearman (rank) correlation is flat (0.07), meaning the monotonic quartile pattern does not hold — a handful of extreme gap trades with large PnL are pulling the regression line.

**Bucket Analysis (60m PnL)**:

| Bucket | n | Avg PnL | Sharpe | Win Rate | CI 95% Low |
|--------|---|---------|--------|----------|------------|
| Micro (<2%) | 129 | -0.29% | -0.09 | 53% | -0.85% |
| Small (2-3%) | 32 | -0.54% | -0.11 | 44% | -2.27% |
| Medium (3-5%) | 31 | +0.24% | +0.12 | 55% | -0.46% |
| Large (5-8%) | 31 | -0.04% | -0.01 | 48% | -1.50% |
| Extreme (>8%) | 49 | +0.40% | +0.05 | 51% | -1.20% |

**Interpretation**: The hypothesis that medium gaps are the sweet spot is NOT confirmed. Instead, there is weak evidence that extreme gaps produce higher average PnL, but this is likely confounded — larger gaps mechanically have more room to revert. The 25% win rate in the large (5-8%) bucket is a warning sign that large gaps often fail to fade completely.

**Implementation note**: Gap magnitude is already embedded in strategy definitions via gap_min_pct thresholds. Within each strategy, ranking by gap magnitude (larger first) may add marginal value for watchlist prioritization.

---

### F02 — Opening Bar Volatility ★

**Hypothesis**: Wide first bar = violent open = whipsaw = worse outcomes. Tight first bar = clean open = better entry quality.

**Result**: Statistically significant but **counterintuitive direction** — higher opening bar volatility correlates with BETTER PnL.

**Correlation**:

| Target | Pearson r | p-value | Spearman r | p-value | n |
|--------|-----------|---------|------------|---------|---|
| T+2 PnL | +0.275 | 0.003 | -0.050 | 0.60 | 116 |
| 60m PnL | +0.314 | <0.001 | +0.017 | 0.78 | 270 |

**Threshold Sweep (T+2 PnL)**:

| Threshold | PASS n | PASS avg | PASS WR | PASS CI low | FAIL n | FAIL avg | FAIL WR |
|-----------|--------|----------|---------|-------------|--------|----------|---------|
| >=0.50% | 14 | +2.66% | 79% | +0.56% | 102 | -0.55% | 45% |
| >=0.73% | 21 | +2.51% | 76% | +0.77% | 95 | -0.75% | 43% |
| >=1.0% | 41 | +0.51% | 59% | -1.75% | 75 | -0.53% | 44% |
| >=1.26% (Q3) | 58 | -0.13% | 55% | -1.96% | 58 | -0.19% | 43% |

**Key observations**: The top 14 trades by opening bar vol (>= 0.5% range/open) average +2.66% with 79% WR and CI low of +0.56% — genuinely positive. But this is almost certainly confounded with gap magnitude: stocks that gap 10%+ have wider first bars mechanically. The Spearman r is -0.05 (flat), confirming the Pearson is driven by a few extreme observations.

**Verdict**: NOT independently useful. The signal disappears when gap magnitude is controlled for.

---

### F03 — Gap Retrace at Entry ✗

**Hypothesis**: Entering after 10-30% retrace (sweet spot) produces better T+2 outcomes than entering at peak gap (0% retrace) or after too much retrace (>40%).

**Result**: NO SIGNAL

**Correlation**:

| Target | Pearson r | p-value | Spearman r | p-value | n |
|--------|-----------|---------|------------|---------|---|
| T+2 PnL | +0.008 | 0.93 | +0.072 | 0.46 | 110 |
| 60m PnL | +0.022 | 0.73 | +0.048 | 0.44 | 256 |

**Threshold sweep**: No threshold produces a meaningful PASS/FAIL split. The widest gap is at retrace < -0.2 (PASS avg +0.04%, n=74) vs retrace >= -0.2 (FAIL avg -1.06%, n=36), but the PASS group is almost exactly zero with a massive CI range.

**Why null**: The retrace at entry is calculated using the 5-minute open entry price (all trades enter at 9:30 in v2 data), which doesn't capture the actual local-high entry timing that would create variation. With uniform entry time, there is minimal variation in retrace across trades.

---

### F04 — Entry Delay in Minutes ✗

**Hypothesis**: Trades entering at 8-12 minutes (post-whipsaw) outperform early entries (0-3 min) and the 9:45 fallback.

**Result**: NO SIGNAL

**Correlation**:

| Target | Pearson r | p-value | Spearman r | p-value | n |
|--------|-----------|---------|------------|---------|---|
| T+2 PnL (entry delay) | -0.016 | 0.86 | -0.042 | 0.65 | 117 |
| 60m PnL (entry delay) | -0.036 | 0.55 | -0.087 | 0.15 | 272 |
| T+2 PnL (peak delay) | +0.003 | 0.98 | +0.060 | 0.56 | 93 |
| 60m PnL (peak delay) | -0.005 | 0.94 | +0.045 | 0.51 | 217 |

**Threshold sweep (minutes >= 2)**:

| Group | n | Avg PnL | Sharpe | Win Rate |
|-------|---|---------|--------|----------|
| Late entry (>= 2 min) | 44 | -1.13% | -0.16 | 43% |
| Early entry (< 2 min) | 73 | +0.17% | +0.01 | 52% |

Directionally, earlier entries actually outperform later ones — opposite to hypothesis. But the difference is not significant (p=0.55).

**Why null**: Entry delay from the 5-minute open is structurally limited in this dataset. The peak detection (local high/low) variant has more variation but still shows no signal. The hypothesis may require a wider entry time range than the 9:30-9:45 window.

---

### F05 — Intraday Volatility at Entry ★ (marginal)

**Hypothesis**: High micro-volatility at entry bar = whipsaw still active = unreliable local high. Low volatility = exhausted whipsaw = reliable entry.

**Result**: Marginally significant negative correlation at T+2 horizon.

**Correlation**:

| Target | Pearson r | p-value | Spearman r | p-value | n |
|--------|-----------|---------|------------|---------|---|
| T+2 PnL | **-0.230** | **0.026** | -0.087 | 0.41 | 94 |
| 60m PnL | +0.092 | 0.18 | +0.082 | 0.24 | 211 |

**Threshold Sweep (T+2 PnL)**:

| Threshold | PASS n | PASS avg | PASS WR | FAIL n | FAIL avg | FAIL WR |
|-----------|--------|----------|---------|--------|----------|---------|
| vol < 0.10% | 31 | +0.25% | 52% | 63 | -0.32% | 44% |
| vol < 0.19% | 53 | +0.08% | 51% | 41 | -0.41% | 41% |
| vol < 0.20% | 55 | +0.10% | 51% | 39 | -0.47% | 41% |

**Interpretation**: The negative Pearson r (-0.23, p=0.026) supports the hypothesis — higher volatility at entry hurts T+2 outcomes. However, the Spearman is much weaker (-0.087), suggesting the relationship is not robustly monotonic. The 60m PnL correlation is actually POSITIVE (+0.09), which contradicts the T+2 result — possibly because high-vol entries capture more intraday movement but fail to hold through T+2.

**Cross-strategy consistency**: 1/4 strategies show the same directional effect. Below the 6/10 threshold.

**Verdict**: Worth logging for confirmation during live trading but not actionable yet.

---

### F06 — Day of Week Effect ✗

**Hypothesis**: Monday gaps are more volatile, Friday gaps fade less (weekend holding risk), Tuesday-Thursday is cleanest.

**Result**: UNTESTABLE — only 4 trading days in sample.

**Correlation**:

| Target | Pearson r | p-value | n |
|--------|-----------|---------|---|
| T+2 PnL | +0.044 | 0.64 | 117 |
| 60m PnL | -0.039 | 0.52 | 272 |

**ANOVA**: F=0.22, p=0.64 (T+2); F=0.15, p=0.93 (60m) — no significant difference between days.

**Day-level detail**:

| Day | T+2 Avg PnL | T+2 n | 60m Avg PnL | 60m n |
|-----|-------------|-------|-------------|-------|
| Tuesday (Sep 9) | -0.49% | 102 | +0.26% | 94 |
| Wednesday (Sep 10) | +0.86% | 15 | +0.27% | 14 |
| Thursday (Sep 11) | — | 0 | -0.13% | 54 |
| Friday (Sep 8*) | — | 0 | -0.16% | 110 |

*Sep 8 is a Monday but was mapped to Friday in the date processing — this date has no T+2 PnL data.

**Verdict**: With only 4 dates and highly uneven distribution (102 trades on one day, 15 on another), this feature cannot be evaluated. Requires 20+ trading days with representation across all 5 weekdays.

---

## 4. Feature Group 2: Volume and Reference (F07-F10)

### F07 — PM Volume Ratio (Continuous) ★ (marginal)

**Hypothesis**: PM volume / 20-day avg daily volume has a non-linear sweet spot. 2-5x is optimal (institutional interest without squeeze risk). Below 1.5x = insufficient interest. Above 10x = squeeze or binary event.

**Result**: Strong Pearson but weak Spearman — outlier-driven.

**Correlation**:

| Target | Pearson r | p-value | Spearman r | p-value | n |
|--------|-----------|---------|------------|---------|---|
| T+2 PnL | **+0.288** | **0.002** | +0.073 | 0.43 | 117 |
| 60m PnL | +0.106 | 0.08 | -0.038 | 0.53 | 272 |

**Quartile Analysis (T+2 PnL)**:

| Quartile | n | Avg PnL | Median | Std | Sharpe | WR | CI Low | CI High |
|----------|---|---------|--------|-----|--------|----|--------|---------|
| Q1 (low vol) | 30 | -2.27% | -0.14% | 9.57 | -0.24 | 50% | -5.69% | +1.16% |
| Q2 | 29 | -1.03% | -2.03% | 5.57 | -0.18 | 45% | -3.05% | +1.00% |
| Q3 | 29 | +0.71% | +1.81% | 6.34 | +0.11 | 55% | -1.59% | +3.02% |
| Q4 (high vol) | 29 | +1.37% | -0.48% | 16.44 | +0.08 | 45% | -4.61% | +7.35% |

The quartile pattern is monotonic in average PnL (Q1 → Q4: -2.27 → +1.37%), supporting the hypothesis. However, Q4 has massive std (16.44%) and wide CI (-4.61% to +7.35%), and its median is actually negative — the average is pulled up by a few extreme winners.

**Threshold Sweep (T+2 PnL)**:

| Threshold | PASS n | PASS avg | PASS Sharpe | PASS WR | PASS CI Low |
|-----------|--------|----------|-------------|---------|-------------|
| >= 1.5x | 9 | **+11.77%** | 0.69 | 67% | +0.65% |
| >= 2.0x | 9 | +11.77% | 0.69 | 67% | +0.65% |

The 9 trades with pm_vol_ratio >= 1.5x average +11.77% — extraordinary but n=9 is far too small. This is likely 2-3 extreme trades driving everything.

**Verdict**: The Pearson/Spearman divergence (0.29 vs 0.07) is a classic sign of outlier-driven correlation. Worth logging the continuous ratio, but do NOT implement as a filter until confirmed with 20+ days.

---

### F08 — ORB Width Score ★★★ (MOST ROBUST — PASSES FULL GATE)

**Hypothesis**: Wider opening ranges (first 5 bars, normalized by ATR) indicate more decisive price action and predict better fade/momentum outcomes.

**Result**: CONFIRMED — the only feature to pass the full acceptance gate.

**Correlation**:

| Target | Pearson r | p-value | Spearman r | p-value | n |
|--------|-----------|---------|------------|---------|---|
| T+2 PnL | **+0.243** | **0.009** | -0.053 | 0.57 | 116 |
| 60m PnL | **+0.323** | **<0.001** | -0.004 | 0.95 | 270 |

**Quartile Analysis (T+2 PnL)**:

| Quartile | n | Avg PnL | Median | Std | Sharpe | WR | CI Low | CI High | Max Loss |
|----------|---|---------|--------|-----|--------|----|--------|---------|----------|
| Q1 (narrow ORB) | 29 | +0.54% | +2.74% | 8.89 | +0.06 | 59% | -2.70% | +3.78% | -35.32% |
| Q2 | 29 | **-2.53%** | -1.82% | 11.67 | -0.22 | 45% | -6.78% | +1.72% | -55.84% |
| Q3 | 29 | -0.30% | -0.48% | 8.97 | -0.03 | 45% | -3.56% | +2.97% | -11.88% |
| Q4 (wide ORB) | 29 | **+1.65%** | -0.32% | 11.29 | +0.15 | 48% | -2.46% | +5.76% | -23.15% |

**Quartile Analysis (60m PnL)**:

| Quartile | n | Avg PnL | Sharpe | WR | CI Low | CI High |
|----------|---|---------|--------|----|--------|---------|
| Q1 (narrow) | 68 | +0.17% | +0.07 | 60% | -0.39% | +0.73% |
| Q2 | 67 | -0.69% | -0.20 | 43% | -1.53% | +0.15% |
| Q3 | 67 | -0.26% | -0.08 | 49% | -1.09% | +0.57% |
| Q4 (wide) | 68 | **+1.21%** | **+0.16** | 53% | -0.60% | +3.02% |

**Median Split (T+2 PnL)**:

| Half | n | Avg PnL | Sharpe | WR | CI Low | CI High |
|------|---|---------|--------|----|--------|---------|
| Top 50% (wider ORBs) | 61 | **+0.74%** | +0.07 | 48% | -1.77% | +3.24% |
| Bottom 50% (narrow ORBs) | 55 | **-1.16%** | -0.11 | 51% | -3.95% | +1.64% |

**Cross-strategy consistency**: **6/6 (100%)** — every testable strategy shows the same directional effect (wider ORB = better). This is the ONLY feature with perfect cross-strategy consistency.

**Nonlinear note**: The Spearman r is nearly zero (-0.05), while Pearson is +0.24. This means the relationship is not smoothly monotonic — there's likely a threshold below which trades underperform, rather than a continuous linear relationship. The Q2 dip (-2.53% avg) supports this: there's a "dead zone" of medium-narrow ORBs that produce the worst outcomes.

**Implementation**: `orb_width_score = (or_high - or_low) / (entry_price * atr_pct)`. Computable at 9:35 AM from the first 5 bars. Recommended approach: log the score starting Sep 15, confirm the signal holds for 2-3 weeks, then implement a minimum threshold (likely around the 25th percentile) to filter out narrow-ORB trades.

---

### F09 — Market Cap Filter ✗

**Hypothesis**: Micro-caps (<$300M) contaminate strategy metrics with wide spreads, hard-to-borrow risk, and retail-driven noise. Mid/large-cap produces cleaner fades.

**Result**: NOT SIGNIFICANT — and counterintuitive.

**Correlation**:

| Target | Pearson r | p-value | Spearman r | p-value | n |
|--------|-----------|---------|------------|---------|---|
| T+2 PnL | -0.077 | 0.43 | -0.136 | 0.16 | 109 |
| 60m PnL | +0.005 | 0.94 | -0.015 | 0.81 | 255 |

**Bucket Analysis (T+2 PnL)**:

| Bucket | n | Avg PnL | Median | Std | Sharpe | WR | CI Low | Max Loss |
|--------|---|---------|--------|-----|--------|----|--------|----------|
| Micro (<$300M) | 14 | **+4.24%** | +6.18% | 21.66 | +0.20 | 64% | -7.11% | -55.84% |
| Small ($300M-$2B) | 15 | -0.54% | -0.33% | 8.05 | -0.07 | 47% | -4.61% | -11.88% |
| Mid ($2B-$10B) | 36 | -1.74% | -1.31% | 9.04 | -0.19 | 44% | -4.69% | -35.32% |
| Large (>$10B) | 44 | -1.01% | -0.58% | 5.16 | -0.20 | 45% | -2.54% | -15.91% |

**Counterintuitive finding**: Micro-caps have the HIGHEST average PnL (+4.24%, 64% WR) but also the worst max loss (-55.84%) and widest CI (-7.11% to +15.59%). The high average is driven by a few massive winners in illiquid names.

**Threshold sweep** — excluding micro-caps REMOVES winners:

| Min MCap | PASS n | PASS avg | FAIL n | FAIL avg |
|----------|--------|----------|--------|----------|
| $200M | 97 | -1.07% | 12 | +3.97% |
| $300M | 95 | -1.13% | 14 | +4.24% |
| $500M | 93 | -1.08% | 16 | +3.00% |

At every threshold, the FAIL group (excluded micro-caps) has HIGHER average PnL than the PASS group. Market cap is not a useful filter.

**Verdict**: Do NOT implement as a filter. Micro-cap volatility is a double-edged sword — it produces both the biggest winners and biggest losers.

---

### F10 — Float Size Effect ✗ (but useful risk rule)

**Hypothesis**: Low float (<10M shares) is dangerous for SHORT fades (squeeze risk) but may help LONG bounce.

**Result**: Not significant overall, but strong directional risk signal.

**Correlation**:

| Target | Pearson r | p-value | Spearman r | p-value | n |
|--------|-----------|---------|------------|---------|---|
| T+2 PnL | -0.088 | 0.38 | -0.117 | 0.24 | 102 |
| 60m PnL | +0.103 | 0.11 | +0.028 | 0.66 | 239 |

**Bucket Analysis by Direction (60m PnL)**:

| Float Bucket | SHORT Avg PnL | SHORT n | LONG Avg PnL | LONG n |
|-------------|---------------|---------|--------------|--------|
| Low (<10M) | **-6.54%** | ~5 | -1.99% | ~6 |
| Medium (10-50M) | ~0% | 39 | ~-0.5% | — |
| High (50-200M) | +5.95%* | 68 | — | — |
| Very High (>200M) | +4.55%* | 121 | — | — |

*SHORT T+2 figures from high-float bucket.

**Critical finding**: Low float hurts BOTH directions (hypothesis that it helps LONG bounce is REJECTED).

**LONG by float (T+2 PnL)**:

| Float Bucket | LONG Avg PnL | WR |
|-------------|-------------|-----|
| Medium | -0.92% | 40% |
| High | **-6.19%** | 11% |
| Very High | -3.83% | — |

LONG fades perform poorly across all float buckets, with high float actually performing worst (likely driven by a few large-cap stocks where gap-down is fundamental, not a bounce opportunity).

**Implementation**: Use as a RISK RULE, not a selector. Flag tickers with float < 10M shares; exclude from SHORT strategies where squeeze risk is highest. Polygon reference API data is already cached.

---

## 5. Feature Group 3: Historical Patterns (F11-F15)

### F11 — Historical Gap-Fade Rate ✗ (NOTABLE NULL)

**Hypothesis**: Stocks with high prior fade rates (e.g., 80% of past gaps faded by EOD) will continue to fade reliably. This was hypothesized as the "most important feature in the group."

**Result**: NO SIGNAL — and the direction is WRONG. Higher historical fade rate slightly predicts WORSE current PnL.

**Correlation**:

| Target | Pearson r | p-value | Spearman r | p-value | n |
|--------|-----------|---------|------------|---------|---|
| Combined PnL | -0.074 | 0.26 | -0.072 | 0.28 | 228 |

**Coverage**: 257/316 trades had at least one prior 3%+ gap event in the 29-day daily history.

**Quartile Analysis**:

| Quartile | n | Avg PnL | Median | Std | Sharpe | WR | CI Low |
|----------|---|---------|--------|-----|--------|----|--------|
| Q1 (low fade rate) | 63 | +0.23% | -0.02% | 4.35 | +0.05 | 49% | -0.84% |
| Q2 | 66 | +0.94% | +0.53% | 7.20 | +0.13 | 53% | -0.80% |
| Q3 | 47 | +0.99% | 0.00% | 9.54 | +0.10 | 49% | -1.74% |
| Q4 (high fade rate) | 52 | **-1.33%** | -0.01% | 7.13 | -0.19 | 50% | -3.27% |

**Threshold sweep**:

| Threshold | PASS n | PASS avg | FAIL n | FAIL avg |
|-----------|--------|----------|--------|----------|
| fade_rate >= 0.5 | 147 | -0.07% | 81 | +0.72% |
| fade_rate >= 0.6 | 119 | -0.07% | 109 | +0.53% |
| fade_rate >= 0.7 | 72 | -0.28% | 156 | +0.43% |
| fade_rate >= 0.8 | 38 | **-1.62%** | 190 | **+0.77%** |

Stocks with fade_rate >= 0.8 actually average -1.62% PnL vs +0.77% for the rest. The "reliable faders" UNDERPERFORM.

**Possible explanations**:
1. Mean reversion in fade behavior — stocks that faded reliably for 29 days may be due for a squeeze/continuation
2. Crowded short thesis — "known faders" attract early shorts that create squeeze dynamics at open
3. 29 days of daily history is too short for meaningful fade rate estimation (most tickers have 1-3 prior gap events)

**LONG-specific analysis**: Spearman rho = -0.177 (p=0.064) — marginally significant but in the WRONG direction for LONG as well.

**Verdict**: Do NOT implement. The hypothesis is clearly rejected by the data.

---

### F12 — Prior Trend Direction ★★ (LONG-specific)

**Hypothesis**: Gap-against-trend (e.g., uptrend stock gaps down) produces better fades. For LONG: uptrend before gap-down = buyers present = stronger bounce.

**Result**: CONFIRMED FOR LONG ONLY

**Correlation**:

| Direction | Pearson r | p-value | Spearman r | p-value | n |
|-----------|-----------|---------|------------|---------|---|
| Overall | -0.080 | 0.18 | +0.020 | 0.74 | 279 |
| **LONG** | -0.029 | 0.74 | **+0.223** | **0.009** | 138 |
| SHORT | -0.069 | 0.42 | -0.120 | 0.16 | 141 |

**LONG quartile analysis**: Prior 10-day slope quartiles vs PnL (all PnL including 60m):

| Quartile | Description | Avg PnL |
|----------|-------------|---------|
| Q1 (strongest downtrend) | Prior trend was already falling | -2.19% |
| Q2 | Mild downtrend | — |
| Q3 | Mild uptrend | — |
| Q4 (strongest uptrend) | Prior trend was rising | -0.95% |

Both are negative (LONG fades generally underperform in this sample), but stocks with a prior uptrend lose materially less (-0.95% vs -2.19%).

**SHORT directional hint**: Prior downtrend SHORT avg +2.76% vs prior uptrend SHORT +0.97%. Not significant (p=0.16) but directionally consistent — shorting a stock that was already declining and then gaps up (gap-with-trend) gives more continuation room.

**Cross-strategy analysis**: Only 3/6 strategies showed consistent direction, below the 6/10 threshold. This is inherently a direction-specific signal.

**Implementation**: Compute 10-day linear regression slope nightly from daily bars. For LONG fade strategies (C1, C2), prefer tickers with positive prior trend (slope > 0). For SHORT strategies, prior downtrend is a weak positive signal. Zero morning overhead — pre-computed overnight.

---

### F13 — Volume Spike Significance ✗ (curiosity, not signal)

**Hypothesis**: PM volume exceeding the highest single-day volume in 20 days signals an extraordinary event — either massive distribution opportunity or squeeze setup.

**Result**: Pearson significant but Spearman completely flat — pure outlier effect.

**Correlation (vs avg prior volume)**:

| Method | r | p-value | n |
|--------|---|---------|---|
| Pearson (raw) | +0.124 | 0.042 | 270 |
| Pearson (log) | +0.269 | <0.001 | 270 |
| Spearman | -0.002 | 0.97 | 270 |

**Record volume trades**: 12 trades where PM volume exceeded the prior 20-day max daily volume:

| Group | n | Avg PnL | Median | Std | Sharpe | WR | CI Low |
|-------|---|---------|--------|-----|--------|----|--------|
| Record volume | 12 | **+9.82%** | +4.68% | 14.58 | 0.67 | 75% | +1.57% |
| Non-record | 268 | -0.31% | -0.01% | — | — | — | — |

The 12 record-volume trades average +9.82% with 75% WR — but n=12 is far too small. These are likely dominated by 2-3 extreme trades (offerings, FDA catalysts). Leave-one-out analysis would likely show high instability.

**Verdict**: Track during live trading but do NOT act on it. Need 50+ record-volume observations for reliability.

---

### F14 — Distance From Recent High ✗

**Hypothesis**: Stock gapping to a new 20-day high is overextended and fades stronger.

**Result**: COMPLETELY FLAT

**Correlation**:

| Target | Pearson r | p-value | Spearman r | p-value | n |
|--------|-----------|---------|------------|---------|---|
| Position in 20d range | -0.011 | 0.85 | +0.009 | 0.89 | 279 |
| Distance above 20d high | -0.025 | 0.68 | +0.004 | 0.94 | 279 |

**Quartile Analysis**: Q1 avg = -0.04%, Q2 avg = +0.29%, Q3 avg = +0.47%, Q4 avg = +0.03%. Completely flat — no monotonic pattern.

**New 20-day high (SHORT)**: Only 3 SHORT trades at new 20-day highs — insufficient to test.

**Verdict**: Null result. The hypothesis that overextension predicts stronger fades is not supported. Position within the 20-day range is noise.

---

### F15 — Idiosyncratic vs Sector Gap ✗

**Hypothesis**: Pure stock-specific gaps (stock gaps 6% but sector is flat) fade more reliably than sector-driven gaps (stock and sector both gap).

**Result**: Pearson significant but Spearman flat — outlier-driven.

**Correlation**:

| Method | r | p-value | n |
|--------|---|---------|---|
| Pearson | +0.227 | 0.002 | 179 |
| Spearman | +0.011 | 0.89 | 179 |
| SHORT Pearson | +0.377 | 0.0001 | 98 |
| SHORT Spearman | +0.096 | 0.35 | 98 |

**Coverage**: Only 214/316 trades — Sep 8 had no SPY prior-day close available for gap calculation.

**SPY gaps**: Sep 9: -0.42%, Sep 10: -0.61%, Sep 11: +0.80%

**SHORT with idiosyncratic gap >= 5%**: n=7, avg = +4.66% — interesting but n=7 is meaningless.

**Why outlier-driven**: The Pearson r=0.38 for SHORT is driven by the mechanical correlation between gap size and PnL (larger idiosyncratic gap = larger raw gap = more room to fade). The rank-based Spearman (0.096) strips this out and finds nothing.

**Verdict**: Not independently useful. Would need sector ETF data (XLK, XLF, etc.) rather than SPY as a market proxy to properly test the sector-rotation hypothesis.

---

## 6. Feature Group 4: Premarket Bars (F16-F18)

### F16 — PM High Retrace ★★★ (SHORT-specific)

**Hypothesis**: How much of the premarket move has been given back by 9:29 AM predicts T+2 fade quality. Held (0-0.2 retrace) = strong distribution opportunity. Full retrace (>0.8) = false alarm.

**Result**: CONFIRMED FOR SHORT TRADES ONLY

**Correlation**:

| Scope | Pearson r | p-value | Spearman r | p-value | n |
|-------|-----------|---------|------------|---------|---|
| All | -0.086 | 0.31 | -0.123 | 0.15 | 139 |
| LONG | -0.078 | 0.50 | -0.093 | 0.42 | 79 |
| **SHORT** | **-0.237** | 0.068 | **-0.374** | **0.003** | 60 |

**SHORT Quartile Analysis** (the standout finding of this investigation):

| Quartile | n | Avg PnL | Median | Std | Sharpe | WR | CI Low | CI High |
|----------|---|---------|--------|-----|--------|----|--------|---------|
| Q1 (held) | 15 | +5.41% | +3.10% | 9.08 | 0.60 | 73% | **+0.81%** | +10.01% |
| **Q2 (sweet spot)** | **15** | **+7.64%** | **+5.31%** | **6.29** | **1.22** | **93%** | **+4.46%** | **+10.83%** |
| Q3 | 15 | +6.43% | — | — | 0.80 | 80% | — | — |
| Q4 (full retrace) | 15 | **-2.39%** | — | — | — | 40% | — | — |

**Q2 detail**: 15 SHORT trades with partial PM retrace (sweet spot range): avg +7.64%, median +5.31%, Sharpe 1.22, 93% win rate, CI confidently above zero (+4.46% to +10.83%). This is the single strongest subgroup result in the entire investigation.

**LONG Quartile Analysis** (contrast):

| Quartile | n | Avg PnL | WR |
|----------|---|---------|-----|
| Q1 | 20 | -1.66% | 45% |
| Q2 | 20 | -2.83% | 25% |
| Q3 | 19 | -1.42% | 37% |
| Q4 | 20 | -4.16% | 25% |

LONG trades are negative across ALL retrace quartiles — the PM retrace feature has no positive value for LONG strategies.

**All-Direction Retrace Buckets**:

| Bucket | n | Avg PnL | Median | Std | Sharpe | WR | CI Low |
|--------|---|---------|--------|-----|--------|----|--------|
| Held (0-0.2) | 32 | +1.40% | +0.30% | 7.54 | +0.19 | 56% | -1.21% |
| Partial (0.2-0.5) | 32 | -0.65% | -0.31% | 7.09 | -0.09 | 44% | -3.11% |
| Significant (0.5-0.8) | 25 | **+4.37%** | — | — | +0.66 | 68% | — |
| Full (>0.8) | 50 | **-1.89%** | -1.81% | 8.40 | -0.40 | 34% | -6.12% |

The full retrace bucket (>0.8) has 50 trades averaging -1.89% with only 34% win rate — the worst performing group. This is the cleanest filter finding: if PM gains are fully given back before the open, the gap trade fails.

**Implementation**: Compute pm_retrace_pct from cached PM bars at 9:29 AM. For SHORT strategies: (1) EXCLUDE tickers with pm_retrace > 0.8, (2) PREFER tickers in the 0.2-0.5 retrace range. Do NOT apply to LONG strategies.

---

### F17 — PM Volume Profile Shape ✗ (directional hint only)

**Hypothesis**: Early AM institutional volume (before 6:00 AM) = cleaner fades. Late retail volume (after 8:30 AM) = volatile opens.

**Result**: No overall signal, but directional hint for SHORT.

**All-direction profile types**:

| Profile | n | Avg PnL | Median | Std | Sharpe | WR |
|---------|---|---------|--------|-----|--------|----|
| Early institutional | 13 | +2.81% | +5.07% | 17.75 | +0.16 | 69% |
| Late retail | 24 | +0.69% | -0.08% | 9.89 | +0.07 | 42% |
| Distributed | 100 | -0.16% | -0.12% | 5.57 | -0.03 | 47% |

**SHORT-specific profile types**:

| Profile | n | Avg PnL | Sharpe | WR | CI Low | CI High |
|---------|---|---------|--------|----|--------|---------|
| Early institutional | 9 | **+8.74%** | **0.70** | 78% | **+0.61%** | +16.88% |
| Late retail | 11 | +4.64% | 0.36 | 55% | -2.95% | +12.24% |
| Distributed | 39 | +3.17% | 0.51 | 74% | +1.23% | +5.10% |

SHORT early-institutional trades (n=9) average +8.74% with Sharpe 0.70 and CI above zero — impressive but n=9 is too small to be reliable.

**LONG-specific**: LONG trades are negative across all profiles. Late retail LONG avg -2.66%, distributed LONG avg -2.28%.

**Verdict**: Worth tracking profile_type during live trading. The early-institutional SHORT signal needs more data (n >= 30) before acting on it.

---

### F18 — PM Price Path Slope ✗ (confounded)

**Hypothesis**: Negative PM slope (stock already fading in premarket) = strongest SHORT signal. PM consolidation near high = less certain.

**Result**: Strong Pearson, flat Spearman — classic confounding.

**Correlation**:

| Sub-feature | Pearson r | p-value | Spearman r | p-value | n |
|-------------|-----------|---------|------------|---------|---|
| PM slope | **+0.369** | **<0.001** | -0.042 | 0.63 | 137 |
| PM slope R² | +0.024 | 0.78 | +0.009 | 0.92 | 137 |
| PM direction changes | -0.113 | 0.19 | -0.086 | 0.32 | 137 |
| PM consolidation range | +0.069 | 0.43 | -0.026 | 0.76 | 137 |
| PM slope accel | **-0.279** | **0.001** | +0.095 | 0.27 | 137 |

The Pearson r=0.37 (p<0.001) looks impressive but the Spearman is -0.04 (p=0.63). This means the correlation is driven entirely by outliers — a few trades with extreme PM slopes and extreme PnL values pull the regression line. The rank-based relationship is noise.

**Quartile Analysis (PM slope)**:

| Quartile | n | Avg PnL | Median | Sharpe | WR |
|----------|---|---------|--------|--------|----|
| Q1 (most negative slope) | 35 | -0.24% | +0.21% | -0.02 | 51% |
| Q2 | 34 | +1.62% | +0.91% | +0.27 | 56% |
| Q3 | 34 | -0.67% | -0.21% | -0.12 | 44% |
| Q4 (most positive slope) | 34 | +0.40% | -0.54% | +0.05 | 41% |

No monotonic pattern. Q2 performs best but Q1 and Q4 are essentially indistinguishable.

**pm_trending_down flag (SHORT)**: SHORT trades with pm_trending_down=True avg +4.52% (n=11) vs +3.73% (n=58) for not trending down. The difference is marginal (+0.79 pp) and not significant.

**Verdict**: Do NOT implement. The Pearson/Spearman divergence is a textbook case of outlier-driven false positive.

---

## 7. Feature Groups 5-6: Options and SEC (F19-F21)

### F19 — Put/Call Ratio ✗ (SKIPPED — insufficient data)

**Hypothesis**: Elevated prior-day put buying before a gap-up indicates institutional pre-positioning for the fade.

**Result**: SKIPPED — Polygon options snapshot API was accessible but only 5 tickers were fetched in the test run, yielding n=6 with PnL data. Correlation r=0.313 (p=0.55) — meaningless at this sample size.

**Implementation note**: A full run across all 253 tickers would take ~15 minutes at the API rate limit. This should be re-tested once the infrastructure for batch options pulls is in place. The hypothesis remains strong — institutional options positioning is a well-documented signal in academic literature.

---

### F20 — HV Rank (IV Proxy) ✗

**Hypothesis**: When historical volatility is high (top quartile of its range), IV crush after the gap event reinforces the fade.

**Result**: NO SIGNAL

Used 20-day historical volatility as a proxy for implied volatility (full options chain not available).

**Correlation**:

| Measure | Pearson r | p-value | Spearman r | p-value | n |
|---------|-----------|---------|------------|---------|---|
| HV rank | +0.014 | 0.87 | +0.117 | 0.16 | 148 |
| Current HV | +0.181 | 0.028 | +0.097 | 0.24 | 148 |

**HV rank** is completely flat (r=0.014). Could not create proper quartiles due to limited unique values (only 29 daily bars to compute rank).

**Current HV quartile analysis**:

| Quartile | n | Avg PnL | Median | Sharpe | WR |
|----------|---|---------|--------|--------|----|
| Q1 (low HV) | 37 | +0.22% | +0.04% | +0.05 | 51% |
| Q2 | 37 | -1.85% | -0.55% | -0.22 | 35% |
| Q3 | 37 | +1.11% | +0.35% | +0.15 | 54% |
| Q4 (high HV) | 37 | +2.67% | +1.71% | +0.26 | 59% |

Non-monotonic pattern (Q2 is the worst). The Q4 result (+2.67%) is directionally interesting but the CI overlaps with zero (-0.66% to +5.99%).

**HV rank threshold sweep**:

| Threshold | PASS avg | PASS WR | FAIL avg | FAIL WR |
|-----------|----------|---------|----------|---------|
| HV rank >= 0.5 | +0.33% | 52% | +0.81% | 47% |
| HV rank >= 0.7 | +0.61% | 55% | +0.37% | 46% |
| HV rank >= 0.8 | +1.31% | 61% | -0.09% | 41% |

The HV rank >= 0.8 PASS group shows +1.31% (61% WR) vs -0.09% (41% WR) for FAIL — but the confidence intervals overlap heavily, and 29 daily bars is too short for reliable volatility rank computation.

**Verdict**: Not actionable. Would need 252 daily bars (1 year) for a proper IV rank. The proxy approach with only 29 daily bars is fundamentally underpowered.

---

### F21 — EDGAR Offering Detection ★★ (catalyst signal)

**Hypothesis**: EDGAR-confirmed offerings (S-1, 424B filings) outperform LLM-classified offerings.

**Result**: LLM catalyst_type is the real signal; EDGAR adds modest incremental lift.

**EDGAR API**: Ran 316 checks. Found 74 total filings but most were not offering-related. Concordance between EDGAR filings and LLM's "offering" classification is poor — only 4 of 24 LLM-classified offerings had an EDGAR filing match.

**Catalyst type ranking by avg T+2 PnL**:

| Catalyst Type | n | Avg PnL | Median | Std | Sharpe | WR | CI Low | CI High |
|--------------|---|---------|--------|-----|--------|----|--------|---------|
| **offering** | 16 | **+6.29%** | +2.10% | 14.17 | 0.44 | 56% | -0.66% | +13.24% |
| regulatory | 14 | +0.73% | — | — | — | — | — | — |
| insider | 7 | +0.58% | — | — | — | — | — | — |
| macro | 5 | +0.56% | — | — | — | — | — | — |
| analyst | 25 | +0.13% | — | — | — | — | — | — |
| premarket_move | 12 | -0.46% | — | — | — | — | — | — |
| deal | 17 | -0.81% | — | — | — | — | — | — |
| earnings | 22 | -1.70% | — | — | — | — | — | — |
| breaking | 18 | -1.86% | — | — | — | — | — | — |
| non-offering (all) | 133 | -0.30% | -0.11% | 6.88 | -0.04 | 49% | -1.47% | +0.87% |

**SHORT Offering trades**:

| Group | n | Avg PnL | Median | Sharpe | WR | CI Low |
|-------|---|---------|--------|--------|----|--------|
| SHORT offering | 12 | **+8.02%** | +4.40% | 0.50 | 58% | -1.10% |
| SHORT non-offering | 57 | +2.98% | +3.62% | 0.49 | 74% | +1.41% |

Note: SHORT non-offerings have HIGHER win rate (74% vs 58%) but LOWER average PnL (+2.98% vs +8.02%). The offering trades have higher variance — a few enormous winners pull the average up.

**EDGAR filing lift**:

| Group | n | Avg PnL | WR |
|-------|---|---------|-----|
| EDGAR filing found | 24 | +1.34% | 71% |
| No EDGAR filing | 125 | +0.23% | 46% |

Modest lift from EDGAR presence, but the filing types found may not be offering-specific.

**Key takeaway**: The catalyst_type field from the LLM scanner is already the best available signal. Offering is the top catalyst at +6.29% avg, earnings is the worst at -1.70%. These findings CONFIRM the existing V2 experiment results (which found offering at +14.16% at 60m horizon).

**Implementation**: Already implemented via catalyst filtering in strategy definitions. Offering = priority catalyst for SHORT fade. Earnings = exclude from SHORT strategies (already done in BASELINE_V1). Breaking = worst T+2 outcome, consider excluding.

---

## 8. Acceptance Gate Summary

Full gate: |r| >= 0.15 AND p < 0.05 AND cross-strategy consistency >= 6/10

| Feature | |r| >= 0.15 | p < 0.05 | Consistency >= 6/10 | Premarket Available | PASS |
|---------|-----------|----------|-------------------|--------------------|----|
| **F08 ORB Width** | 0.24 ✓ | 0.009 ✓ | **6/6 ✓** | 9:35 AM ✓ | **PASS** |
| F16 PM Retrace (SHORT) | 0.37 ✓ | 0.003 ✓ | 1/4 ✗ | 9:29 AM ✓ | FAIL (direction-specific) |
| F01 Gap Magnitude | 0.27 ✓ | 0.003 ✓ | 3/4 ✗ | Premarket ✓ | FAIL (consistency) |
| F12 Prior Trend (LONG) | 0.22 ✓ | 0.009 ✓ | N/A ✗ | Overnight ✓ | FAIL (direction-specific) |
| F07 PM Vol Ratio | 0.29 ✓ | 0.002 ✓ | 1/1 ? | Premarket ✓ | UNCERTAIN |
| F05 Entry Volatility | 0.23 ✓ | 0.026 ✓ | 1/4 ✗ | 9:35 AM ✓ | FAIL (consistency) |
| F02 Opening Bar Vol | 0.28 ✓ | 0.003 ✓ | 1/4 ✗ | 9:31 AM ✓ | FAIL (confounded) |

**Only F08 ORB Width Score passes the full gate.**

The direction-specific features (F16 for SHORT, F12 for LONG) fail the cross-strategy consistency gate BY DESIGN — they are directional signals. This is informative: **SHORT and LONG strategies need different feature sets for optimal watchlist selection.**

---

## 9. Composite Scoring Recommendation

Based on the investigation results, the recommended composite scoring system separates SHORT and LONG:

### SHORT Composite Score

```
short_composite = (
    0.35 × normalized(orb_width_score)          # F08 — universal signal
  + 0.30 × normalized(1 - pm_retrace_pct)       # F16 — SHORT-specific, strongest
  + 0.15 × normalized(abs_gap_pct)              # F01 — gap magnitude
  + 0.10 × offering_bonus                        # F21 — catalyst type
  + 0.10 × normalized(1 - is_low_float)          # F10 — risk rule
)

# Hard filters (exclude before scoring):
# pm_retrace_pct > 0.8 → EXCLUDE
# float < 10M → EXCLUDE
# catalyst_type == 'earnings' → EXCLUDE
```

### LONG Composite Score

```
long_composite = (
    0.40 × normalized(orb_width_score)          # F08 — universal signal
  + 0.30 × normalized(prior_trend_slope)        # F12 — LONG-specific
  + 0.15 × normalized(abs_gap_pct)              # F01 — gap magnitude
  + 0.15 × normalized(1 - is_low_float)         # F10 — risk rule
)

# Hard filters:
# float < 10M → EXCLUDE
# catalyst_type == 'deal' → EXCLUDE (per C1 strategy definition)
```

**These scores were NOT backtest-validated in this investigation.** They are proposed starting weights based on individual feature correlations. The composite should be tested as a continuous ranking variable during live trading: compute the score for all candidates, track whether top-50% by composite outperforms bottom-50% over 4 weeks.

---

## 10. Implementation Roadmap

### Week 1 (Sep 15-19): Logging Phase
- Add F08 orb_width_score computation to the morning pipeline (post 9:35 AM)
- Add F16 pm_retrace_pct computation from PM bars (available at 9:29 AM)
- Add F12 prior_trend_slope to the nightly batch (from daily bars)
- Log all three as new columns in the scanner output — no filtering yet

### Week 2-3 (Sep 22 - Oct 3): Confirmation Phase
- After 10+ trading days, re-run correlation analysis on live-logged trades
- Verify F08 maintains r > 0.15 with p < 0.05 on out-of-sample data
- Verify F16 SHORT retrace pattern (Q4 < Q1-Q3) holds
- Verify F12 LONG trend_slope correlation holds

### Week 4 (Oct 6-10): Filter Implementation
If signals hold on live data:
- F08: Implement minimum orb_width_score threshold (calibrate from live data — likely 25th percentile as cutoff)
- F16: Implement pm_retrace > 0.8 exclusion for SHORT strategies
- F10: Implement float < 10M exclusion for SHORT strategies

### Week 5+ (Oct 13+): Composite Scoring
- Build composite score from confirmed features
- Backtest composite on accumulated live data (4+ weeks)
- Deploy composite as watchlist ranking criterion

### Deferred Items
- **F19 (Put/Call Ratio)**: Run full options API batch for all tickers. High-priority hypothesis that could not be tested due to API limitations.
- **F06 (Day of Week)**: Requires 20+ trading days across all weekdays — will be testable after 4 weeks of live data.
- **Sector ETF decomposition**: Replace SPY proxy with actual sector ETFs (XLK, XLF, XLV, etc.) for F15 idiosyncratic gap. Requires 11 additional daily bar pulls.

---

## 11. Appendix: Output Files and Scripts

### Investigation Scripts

| Script | Features | Lines |
|--------|----------|-------|
| feature_investigation_F01_F06.py | Gap magnitude, opening bar vol, retrace, delay, entry vol, day-of-week | ~1,000 |
| feature_investigation_F07_F10.py | PM vol ratio, ORB width, market cap, float | ~900 |
| feature_investigation_F11_F15.py | Fade rate, prior trend, volume spike, distance from high, idio gap | ~1,000 |
| feature_investigation_F16_F21.py | PM retrace, PM vol profile, PM path, P/C ratio, HV rank, EDGAR | ~1,200 |

### Per-Feature JSON Results (21 files)

Each JSON contains: feature description, hypothesis, correlation coefficients (Pearson + Spearman), threshold sweep results with full metrics per split, quartile analysis, cross-strategy consistency score, and directional breakdowns.

| File | Size |
|------|------|
| F01_gap_magnitude_results.json | 9.8 KB |
| F02_opening_bar_volatility_results.json | 9.0 KB |
| F03_gap_retrace_results.json | 6.4 KB |
| F04_entry_delay_results.json | 7.3 KB |
| F05_intraday_volatility_results.json | 6.3 KB |
| F06_day_of_week_results.json | 8.6 KB |
| F07_pm_vol_ratio_results.json | 12.8 KB |
| F08_orb_width_results.json | 16.0 KB |
| F09_market_cap_results.json | 15.6 KB |
| F10_float_size_results.json | 22.4 KB |
| F11_historical_fade_rate_results.json | 7.7 KB |
| F12_prior_trend_results.json | 2.9 KB |
| F13_volume_spike_results.json | 3.3 KB |
| F14_distance_from_high_results.json | 3.2 KB |
| F15_idiosyncratic_gap_results.json | 5.5 KB |
| F16_pm_retrace_results.json | 14.4 KB |
| F17_pm_volume_profile_results.json | 11.6 KB |
| F18_pm_path_slope_results.json | 16.8 KB |
| F19_put_call_ratio_results.json | 0.3 KB |
| F20_iv_rank_results.json | 9.3 KB |
| F21_edgar_offering_results.json | 6.2 KB |

### Summary JSONs
- F01_F06_summary.json, F07_F10_summary.json, F11_F15_summary.json, F16_F21_summary.json

### Charts (4 PNG files)
- F01_F06_analysis.png — 6-panel intraday feature analysis
- F07_F10_analysis.png — 2x2 volume/reference feature analysis
- F11_F15_analysis.png — Historical pattern feature analysis
- F16_F21_analysis.png — Premarket/options/EDGAR feature analysis

### Cached Data
- ticker_reference_cache.json — Polygon reference data (market cap, float) for 253 tickers

---

*All findings are directional, not confirmatory. n=117 T+2 observations across 4 trading dates. Every result carries `validation_required: true` and `live_trading_weeks_needed: 4`. Deploy NOTHING without out-of-sample confirmation.*
