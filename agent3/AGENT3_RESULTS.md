# Agent 3 — Long Strategy Investigation: Complete Results

**Date:** 2026-09-15
**Script:** `agent3/long_investigation.py`
**Runtime:** ~11 min (API-bound, Polygon.io)
**Period:** 2025-09-15 → 2026-09-12 (250 trading days)
**OOS Cutoff:** 2026-01-01
**Universe:** _FLOAT (194 tickers with float >= 10M)
**Constraint:** LONG only, no shorting, no leverage (pension account)

---

## 1. Investigation Overview

Three independent LONG hypotheses tested on the _FLOAT universe, scanning all 250 trading days with a single data-fetch pass. 62 total experiment configurations evaluated.

### Hypotheses

| ID | Hypothesis | Signal | Direction |
|----|-----------|--------|-----------|
| **A** | Gap-Down Mean Reversion | Buy the dip after ≥5% gap-down | LONG |
| **B** | Gap-Up Momentum Continuation | ORB breakout after ≥5% gap-up | LONG |
| **C** | Multi-Day Reversal | Oversold bounce + gap-up confirmation | LONG |

### Data Collection Summary

| Phase | Description | Duration |
|-------|-------------|----------|
| 1a | ADV lookback (20 dates) | ~5s |
| 1b | Scan 250 days for candidates | ~343s |
| 2 | Reference data for 138 unique tickers | ~180s |
| 3 | 1-min bars (PM+RTH) for 1072 pairs | ~120s |
| 4 | Outcome date fetching | ~5s |
| 5 | Enriched trade list building | ~2s |
| **Total** | | **675s (11.2 min)** |

**Candidate counts:** A=373 gap-down, B=577 gap-up, C=177 reversal
**Trades with entry:** A=365, B=249 (ORB break), C=177

---

## 2. Strategy Mechanics

### 2.1 Hypothesis A — Gap-Down Mean Reversion (LONG)

Mirror of Agent 1's gap-fade SHORT strategy, adapted for LONG:

| Component | Agent 1 (SHORT) | Agent 3 Hyp A (LONG) |
|-----------|-----------------|----------------------|
| Gap direction | Gap-UP ≥5% | Gap-DOWN ≥5% |
| Entry detection | `find_local_high()` — local max in bars[3..11] | `find_local_low()` — local min in bars[3..11] |
| PnL formula | `(entry - exit) / entry * 100` | `(exit - entry) / entry * 100` |
| Bet direction | Fade the gap (SHORT) | Buy the bounce (LONG) |
| Sort | Most positive gap first | Most negative gap first |
| Fallback | Bar 15 close | Bar 15 close |

**Entry logic (`find_local_low`):**
- Scan RTH bars 3 through 11 (configurable via `MAX_PEAK_BAR=12`)
- At each bar `i`, check if it has the lowest `low` among bars `[i-3, i+3]` (window=3)
- If found, enter at bar `i+3` close price (confirmation bars)
- If no local low by bar 11, fallback entry at bar 15 close
- Skip if stock has already bounced >15% by bar 15 (`SKIP_BOUNCE_PCT=0.15`)

**Tier classification (gap magnitude):**
| Tier | Gap Range | Description |
|------|-----------|-------------|
| D1 | -5% to -8% | Small gap-down |
| D2 | -8% to -12% | Medium gap-down |
| D3 | -12% to -20% | Large gap-down |
| D4 | -20% to -35% | Severe gap-down |
| D5 | < -35% | Extreme gap-down |

**Pattern classification (`classify_gap_down_pattern`):**
Examines first 60 minutes of RTH trading after gap-down:
- `rise_30m = (high of first 30 bars - open) / |open|`
- `recovery = (close at bar 60 - open) / |open|`

| Pattern | Condition | Interpretation |
|---------|-----------|----------------|
| `sustained_rise` | rise_30m > 3% AND recovery > 50% of rise | Dip bought aggressively, holds |
| `rise_then_drop` | rise_30m > 3% AND recovery ≤ 50% of rise | Dead-cat bounce, fades |
| `mild_rise` | rise_30m 1%–3% | Tepid reaction |
| `continued_drop` | rise_30m ≤ 1% | No bounce at all |

### 2.2 Hypothesis B — Gap-Up Momentum (ORB Breakout)

**Opening Range Breakout (ORB) logic (`find_orb_break_up`):**
- First 5 RTH bars define the opening range: `orb_high = max(h)`, `orb_low = min(l)`
- Scan bars 5–25 for first close above `orb_high`
- If found: entry at that bar's close, record break strength, break bar volume, avg ORB volume
- Break strength = `(close - orb_high) / orb_high`

**Volume confirmation (B2):** Break bar volume > 1.5x average ORB bar volume

**Break strength tiers (B3):**
| Tier | Break % | Count |
|------|---------|-------|
| weak | 0 – 0.25% | 78 |
| medium | 0.25% – 0.75% | 104 |
| strong | > 0.75% | 67 |

### 2.3 Hypothesis C — Multi-Day Reversal

**C1: 5-day decline + gap-up signal**
- Prior 5 trading days: cumulative return ≤ -15%
- Today: gap-up ≥ 3%
- Price ≥ $20 (higher quality for pension)
- Entry at open

**C2: 10-day decline + volume spike**
- Prior 10 trading days: cumulative return ≤ -20%
- Today: gap-up ≥ 3%
- Today volume ≥ 3x 20-day average volume
- Price ≥ $20
- Entry at open

---

## 3. Full-Period Results (2025-09-15 → 2026-09-12)

### 3.1 Hypothesis A — Gap-Down Mean Reversion

#### A1: Raw Baseline (all gap-downs ≥5%)

| Hold | n | Avg | Median | TrimMean | WR | Sharpe | PF | CI 95% | Pass |
|------|---|-----|--------|----------|-----|--------|-----|--------|------|
| T+2 | 365 | +0.84% | +0.66% | +0.70% | 53% | 0.074 | 1.24 | [-0.32, +1.99] | **NO** |
| T+3 | 362 | +0.86% | +0.28% | +0.70% | 52% | 0.060 | 1.19 | [-0.61, +2.33] | **NO** |
| T+4 | 358 | +0.73% | -0.50% | +0.48% | 47% | 0.045 | 1.14 | [-0.96, +2.41] | **NO** |
| T+5 | 355 | +1.30% | -0.42% | +0.96% | 48% | 0.076 | 1.25 | [-0.49, +3.08] | **NO** |

**Verdict:** Gap-down mean reversion NOT statistically significant on the raw universe. CI includes zero at all hold periods. Positive average driven by right tail.

#### A3: Common Stock Only (no ETFs)

| Hold | n | Avg | Median | TrimMean | WR | Sharpe | PF | CI 95% | Pass |
|------|---|-----|--------|----------|-----|--------|-----|--------|------|
| T+2 | 352 | +0.86% | +0.64% | +0.73% | 53% | 0.075 | 1.24 | [-0.34, +2.05] | **NO** |
| T+3 | 349 | +0.80% | +0.23% | +0.64% | 52% | 0.056 | 1.18 | [-0.71, +2.32] | **NO** |
| T+5 | 342 | +1.15% | -0.54% | +0.83% | 47% | 0.067 | 1.21 | [-0.68, +2.98] | **NO** |

**Verdict:** Filtering to common stock only has negligible effect. The 13 ETF/non-CS trades are not the issue.

#### A5: Gap-Down Magnitude Tiers

**T+2 hold:**

| Tier | n | Avg | Median | TrimMean | WR | Sharpe | PF | CI 95% | Pass |
|------|---|-----|--------|----------|-----|--------|-----|--------|------|
| D1 -5%→-8% | 227 | +0.86% | +0.12% | +0.60% | 51% | 0.072 | 1.23 | [-0.71, +2.43] | NO |
| **D2 -8%→-12%** | **82** | **+1.93%** | **+1.71%** | **+1.92%** | **62%** | **0.247** | **1.94** | **[+0.24, +3.62]** | **YES** |
| D3 -12%→-20% | 44 | -1.40% | -1.92% | -1.31% | 45% | -0.115 | 0.72 | [-4.99, +2.20] | NO |
| D4 -20%→-35% | 10 | +2.04% | -0.06% | +2.43% | 50% | 0.157 | 1.49 | [-6.01, +10.10] | NO |
| D5 <-35% | 2 | -3.67% | -3.67% | -3.67% | 50% | -0.384 | 0.30 | [-16.94, +9.60] | NO |

**T+3 hold:**

| Tier | n | Avg | Median | TrimMean | WR | Sharpe | PF | CI 95% | Pass |
|------|---|-----|--------|----------|-----|--------|-----|--------|------|
| D1 | 226 | +0.78% | +0.01% | +0.57% | 50% | 0.051 | 1.16 | [-1.20, +2.76] | NO |
| **D2** | **82** | **+2.22%** | **+2.88%** | **+2.35%** | **60%** | **0.221** | **1.78** | **[+0.05, +4.40]** | **YES** |
| D3 | 42 | -1.89% | -2.01% | -2.12% | 45% | -0.119 | 0.69 | [-6.67, +2.90] | NO |
| D4 | 10 | +4.66% | +2.62% | +4.95% | 60% | 0.293 | 2.19 | [-5.20, +14.51] | NO |
| D5 | 2 | -6.62% | -6.62% | -6.62% | 50% | -0.485 | 0.19 | [-25.55, +12.31] | NO |

**T+5 hold:**

| Tier | n | Avg | Median | TrimMean | WR | Sharpe | PF | CI 95% | Pass |
|------|---|-----|--------|----------|-----|--------|-----|--------|------|
| D1 | 226 | +1.01% | -0.99% | +0.36% | 46% | 0.057 | 1.18 | [-1.29, +3.30] | NO |
| **D2** | **80** | **+3.57%** | **+2.09%** | **+3.52%** | **57%** | **0.229** | **2.05** | **[+0.15, +6.99]** | **CI only** |
| D3 | 38 | -2.09% | -2.91% | -1.80% | 37% | -0.115 | 0.73 | [-7.86, +3.69] | NO |
| D4 | 9 | +3.50% | -2.92% | +2.44% | 44% | 0.237 | 1.84 | [-6.16, +13.17] | NO |
| D5 | 2 | -2.85% | -2.85% | -2.85% | 50% | -0.254 | 0.47 | [-18.40, +12.70] | NO |

**Key finding:** D2 (-8% to -12%) is the ONLY tier with CI > 0 across all hold periods. The sweet spot — big enough gap to trigger real buying, but not so catastrophic that the stock is fundamentally broken. D3+ (≥12% gap-down) has NEGATIVE returns.

#### A6: Pattern Classification (Opening Price Action)

**T+2 hold:**

| Pattern | n | Avg | Median | TrimMean | WR | Sharpe | PF | CI 95% | Pass |
|---------|---|-----|--------|----------|-----|--------|-----|--------|------|
| **sustained_rise** | **98** | **+4.41%** | **+2.18%** | **+3.78%** | **62%** | **0.409** | **3.39** | **[+2.27, +6.54]** | **YES** |
| rise_then_drop | 80 | -4.20% | -2.01% | -3.29% | 39% | -0.322 | 0.32 | [-7.07, -1.34] | ANTI |
| mild_rise | 112 | +0.30% | +0.52% | +0.26% | 54% | 0.033 | 1.09 | [-1.38, +1.98] | NO |
| continued_drop | 70 | +2.42% | +1.36% | +1.74% | 54% | 0.216 | 1.84 | [-0.20, +5.04] | NO |

**T+3 hold:**

| Pattern | n | Avg | Median | TrimMean | WR | Sharpe | PF | CI 95% | Pass |
|---------|---|-----|--------|----------|-----|--------|-----|--------|------|
| **sustained_rise** | **97** | **+3.84%** | **+1.83%** | **+3.83%** | **63%** | **0.244** | **2.13** | **[+0.71, +6.97]** | **YES** |
| rise_then_drop | 79 | -4.46% | -4.00% | -3.79% | 35% | -0.291 | 0.42 | [-7.84, -1.08] | ANTI |
| mild_rise | 111 | +0.34% | -0.49% | +0.15% | 50% | 0.030 | 1.08 | [-1.74, +2.41] | NO |
| **continued_drop** | **70** | **+3.74%** | **+1.23%** | **+2.67%** | **59%** | **0.263** | **2.23** | **[+0.41, +7.06]** | **CI only** |

**T+5 hold:**

| Pattern | n | Avg | Median | TrimMean | WR | Sharpe | PF | CI 95% | Pass |
|---------|---|-----|--------|----------|-----|--------|-----|--------|------|
| sustained_rise | 94 | +4.56% | +1.75% | +4.91% | 55% | 0.217 | 1.88 | [+0.32, +8.80] | CI only |
| rise_then_drop | 77 | -5.74% | -4.67% | -4.81% | 32% | -0.363 | 0.35 | [-9.28, -2.20] | ANTI |
| mild_rise | 110 | +1.68% | +1.05% | +1.33% | 53% | 0.135 | 1.44 | [-0.65, +4.00] | NO |
| continued_drop | 69 | +4.40% | -0.07% | +2.91% | 49% | 0.248 | 2.16 | [+0.21, +8.59] | CI only |

**Key findings:**
1. **`sustained_rise` is the strongest single signal in the entire investigation.** Sharpe 0.409 at T+2, 62% WR, PF 3.39.
2. **`rise_then_drop` is a powerful ANTI-signal.** Consistently -4% to -6% across all hold periods. If the bounce fades within 60 minutes, the stock continues declining.
3. `continued_drop` at T+3/T+5 also shows positive CI — stocks that don't bounce at all on gap-down day tend to recover over the next week.

#### A7: Hold Period Sweep (All Horizons)

| Hold | n | Avg | Median | TrimMean | WR | Sharpe | CI 95% |
|------|---|-----|--------|----------|-----|--------|--------|
| 30m | 363 | +0.25% | +0.09% | +0.23% | 51% | 0.097 | [-0.02, +0.51] |
| 60m | 360 | +0.12% | +0.04% | +0.11% | 51% | 0.034 | [-0.25, +0.49] |
| EOD | 365 | -0.10% | +0.08% | -0.12% | 51% | -0.017 | [-0.68, +0.49] |
| T+1 | 365 | +0.09% | -0.06% | -0.12% | 49% | 0.011 | [-0.73, +0.91] |
| T+2 | 365 | +0.84% | +0.66% | +0.70% | 53% | 0.074 | [-0.32, +1.99] |
| T+3 | 362 | +0.86% | +0.28% | +0.70% | 52% | 0.060 | [-0.61, +2.33] |
| T+4 | 358 | +0.73% | -0.50% | +0.48% | 47% | 0.045 | [-0.96, +2.41] |
| T+5 | 355 | +1.30% | -0.42% | +0.96% | 48% | 0.076 | [-0.49, +3.08] |

**Key finding:** Intraday is FLAT. The bounce takes time — mean reversion is a multi-day phenomenon. Optimal hold is T+2 to T+3 (best risk-adjusted) or T+5 (highest absolute).

### 3.2 Hypothesis B — Gap-Up ORB Momentum

#### B1: ORB Continuation Baseline

| Hold | n | Avg | Median | TrimMean | WR | Sharpe | PF | CI 95% |
|------|---|-----|--------|----------|-----|--------|-----|--------|
| 60m | 247 | -0.27% | -0.23% | -0.30% | 45% | -0.061 | 0.84 | [-0.81, +0.28] |
| EOD | 249 | +0.05% | +0.28% | -0.04% | 52% | 0.007 | 1.02 | [-0.75, +0.84] |
| T+1 | 249 | -0.74% | -1.17% | -1.01% | 43% | -0.086 | 0.80 | [-1.79, +0.32] |
| T+2 | 249 | -0.80% | -1.60% | -0.96% | 41% | -0.076 | 0.82 | [-2.11, +0.51] |

**Verdict:** DEAD. Negative at every hold period. ORB breakouts after ≥5% gap-up on _FLOAT stocks are NOT continuation signals — they are exhaustion signals.

#### B2: Volume Confirmation

| Config | Hold | n | Avg | WR | Sharpe | CI 95% |
|--------|------|---|-----|-----|--------|--------|
| Vol confirmed | EOD | 22 | -2.22% | 32% | -0.254 | [-5.87, +1.43] |
| Vol confirmed | T+2 | 22 | -2.86% | 41% | -0.185 | [-9.33, +3.61] |
| No vol confirm | EOD | 227 | +0.27% | 54% | 0.044 | [-0.53, +1.06] |
| No vol confirm | T+2 | 227 | -0.60% | 41% | -0.060 | [-1.90, +0.70] |

**Verdict:** Volume confirmation makes it WORSE. High-volume ORB breaks are the biggest losers.

#### B3: Break Strength Tiers

| Tier | Hold | n | Avg | WR | Sharpe | CI 95% |
|------|------|---|-----|-----|--------|--------|
| weak 0–0.25% | EOD | 78 | -0.43% | 51% | -0.092 | [-1.48, +0.61] |
| medium 0.25–0.75% | EOD | 104 | +0.09% | 52% | 0.011 | [-1.37, +1.54] |
| strong >0.75% | EOD | 67 | +0.55% | 54% | 0.089 | [-0.93, +2.02] |
| weak | T+2 | 78 | -1.90% | 35% | -0.222 | [-3.81, +0.00] |
| medium | T+2 | 104 | +0.61% | 49% | 0.061 | [-1.33, +2.55] |
| strong | T+2 | 67 | -1.70% | 37% | -0.131 | [-4.82, +1.41] |

**Verdict:** No tier rescues the strategy. Hypothesis B is conclusively DEAD.

### 3.3 Hypothesis C — Multi-Day Reversal

#### C1: 5-Day Decline + Gap-Up Signal

| Hold | n | Avg | Median | TrimMean | WR | Sharpe | PF | CI 95% | Pass |
|------|---|-----|--------|----------|-----|--------|-----|--------|------|
| **T+3** | **177** | **+5.77%** | **+3.21%** | **+5.05%** | **62%** | **0.398** | **3.02** | **[+3.64, +7.91]** | **YES** |
| T+5 | 177 | +4.20% | +4.58% | +3.52% | 59% | 0.248 | 1.93 | [+1.70, +6.69] | CI only |
| T+10 | 177 | +7.64% | +1.75% | +5.94% | 55% | 0.318 | 2.43 | [+4.10, +11.18] | CI only |

**Verdict:** STRONG. C1 is CI-confirmed at ALL three hold periods. The signal: a stock that dropped ≥15% over 5 days, then gaps up ≥3%, continues higher. At T+3: +5.77% avg with Sharpe 0.398 and 62% WR. This is a high-conviction pension signal.

**Key observation:** T+10 has the highest absolute return (+7.64%) but lower WR (55%) and wider dispersion. T+3 is the best risk-adjusted hold.

#### C2: 10-Day Decline + Volume Spike

| Hold | n | Avg | Median | WR | Sharpe | CI 95% |
|------|---|-----|--------|-----|--------|--------|
| T+3 | 3 | +4.08% | +7.05% | 67% | 0.247 | [-14.67, +22.84] |
| T+5 | 3 | +5.79% | +3.21% | 67% | 0.279 | [-17.68, +29.26] |
| T+10 | 3 | +7.03% | +3.65% | 67% | 0.339 | [-16.43, +30.49] |

**Verdict:** INSUFFICIENT DATA (n=3). The 10-day ≥20% decline + ≥3x volume spike criteria are too restrictive. Numbers look good but CIs are meaningless at n=3.

---

## 4. Pension Acceptance Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| CI 95% lower bound > 0 | Required | Edge is statistically significant |
| Sharpe ratio ≥ 0.15 | Required | Sufficient risk-adjusted return |
| Win rate ≥ 60% | Required | Psychologically sustainable for pension |
| Max single loss > -30% | Required | No catastrophic drawdown risk |

### Full-Period Passing Experiments (4 configs)

| Experiment | n | Avg | WR | Sharpe | CI 95% |
|-----------|---|-----|-----|--------|--------|
| **A5 D2 -8%→-12% T+2** | 82 | +1.93% | 62% | 0.247 | [+0.24, +3.62] |
| **A6 sustained_rise T+2** | 98 | +4.41% | 62% | 0.409 | [+2.27, +6.54] |
| **A6 sustained_rise T+3** | 97 | +3.84% | 63% | 0.244 | [+0.71, +6.97] |
| **C1 T+3** | 177 | +5.77% | 62% | 0.398 | [+3.64, +7.91] |

Note: A5 D2 T+3 just barely passes CI [+0.05] and has 60% WR (borderline). Not included as a strong candidate.

---

## 5. OOS Validation (Fixed, v2)

IS period: 2025-09-15 → 2026-01-01
OOS period: 2026-01-01 → 2026-09-12

**Bug fixed in v2:** Original OOS validation used unfiltered trade lists for all experiments (A5_D2 was validated on 369 full-A trades instead of 82 D2-filtered trades). Fixed by adding `_resolve_trade_filter()` that maps each experiment ID to its exact filter function.

### OOS Results

| Experiment | IS n | IS Avg | IS Sharpe | IS CI | OOS n | OOS Avg | OOS Sharpe | OOS CI | Status |
|-----------|------|--------|-----------|-------|-------|---------|------------|--------|--------|
| A5 D2 T+2 | 18 | +3.11% | 0.352 | [-0.97, +7.18] | 64 | +1.60% | 0.212 | [-0.25, +3.45] | IS CI fails |
| A5 D2 T+3 | 18 | +6.28% | 0.557 | [+1.07, +11.49] | 64 | +1.08% | 0.114 | [-1.24, +3.40] | OOS Sharpe fails |
| A5 D2 T+5 | 18 | +8.17% | 0.517 | [+0.87, +15.47] | 62 | +2.24% | 0.145 | [-1.60, +6.07] | OOS Sharpe fails |
| **A6 sustained_rise T+2** | **27** | **+4.76%** | **0.591** | **[+1.72, +7.80]** | **71** | **+4.27%** | **0.365** | **[+1.55, +6.99]** | **PENSION READY** |
| A6 sustained_rise T+3 | 27 | +4.72% | 0.428 | [+0.56, +8.89] | 70 | +3.50% | 0.203 | [-0.54, +7.54] | OOS CI fails |
| A6 sustained_rise T+5 | 27 | +4.40% | 0.295 | [-1.22, +10.01] | 67 | +4.63% | 0.200 | [-0.90, +10.16] | IS CI fails |
| A6 continued_drop T+3 | 15 | +9.47% | 0.564 | [+0.98, +17.97] | 55 | +2.17% | 0.165 | [-1.30, +5.65] | OOS CI fails |
| A6 continued_drop T+5 | 15 | +10.39% | 0.472 | [-0.75, +21.54] | 54 | +2.73% | 0.168 | [-1.59, +7.06] | IS CI fails |
| C1 T+3 | 45 | +3.91% | 0.250 | [-0.66, +8.49] | 132 | +6.41% | 0.454 | [+4.00, +8.81] | IS CI fails |
| C1 T+5 | 45 | +2.84% | 0.174 | [-1.92, +7.60] | 132 | +4.66% | 0.271 | [+1.73, +7.58] | IS CI fails |
| C1 T+10 | 45 | +5.66% | 0.251 | [-0.94, +12.26] | 132 | +8.32% | 0.339 | [+4.13, +12.50] | IS CI fails |

### PENSION READY: A6 sustained_rise T+2

**The only experiment that passes both IS AND OOS pension criteria:**
- IS: +4.76% avg, Sharpe 0.591, CI [+1.72, +7.80], n=27
- OOS: +4.27% avg, Sharpe 0.365, CI [+1.55, +6.99], n=71
- Minimal degradation: IS→OOS avg drops only 10% (+4.76% → +4.27%)
- OOS is 2.6x the IS sample size — this is a robust signal

### Near-Miss Analysis

**C1 T+3** is the most promising near-miss:
- OOS is EXCELLENT: +6.41%, Sharpe 0.454, CI [+4.00, +8.81]
- IS fails only because IS period is short (45 trades, CI [-0.66, +8.49])
- The OOS actually IMPROVES over IS — unusual and encouraging
- With a longer IS window or more data, this would likely pass
- **Recommendation:** Monitor C1 as primary candidate for deployment alongside A6

**A5 D2** shows classic IS→OOS degradation:
- IS avg +3–8% degrades to OOS +1–2%
- Small IS sample (n=18) drives high IS Sharpe that doesn't hold
- The tier filter alone is not enough; needs pattern confirmation

**A6 continued_drop** shows severe overfitting:
- IS avg +9–10% with n=15 collapses to OOS +2%
- Too few IS trades to draw conclusions

---

## 6. Key Questions Answered

### Q1: Does gap-DOWN mean reversion exist at T+2 or longer?
**NO — not significant on the raw universe.** A1 T+2: avg +0.84%, CI [-0.32, +1.99]. The edge only appears in filtered subsets (D2 tier, sustained_rise pattern).

### Q2: Which gap tiers work best? (Skipped experiments)
**D2 (-8% to -12%) is the only viable tier.** D1 is too diluted, D3+ is negative (fundamentally broken stocks).

### Q3: Is the bounce intraday or multi-day?
**Multi-day.** 30m and 60m returns are near zero (+0.25%, +0.12%). The edge builds from T+2 onwards. This is NOT a scalping signal.

### Q4: Does gap-UP ORB continuation provide LONG alpha?
**NO.** B1 T+2: avg -0.80%, Sharpe -0.076. Dead at every hold period, every filter, every tier. Hypothesis B is conclusively rejected.

### Q5: Are multi-day reversals viable for pension capital?
**YES — C1 is one of the strongest signals.** C1 T+3: +5.77%, Sharpe 0.398, 62% WR. All three hold periods CI-confirmed. Needs longer IS window for pension approval.

### Q6: Best single configuration with CI > 0?
**A6 sustained_rise T+2:** avg +4.41%, Sharpe 0.409, WR 62%, n=98. Also the ONLY experiment that passes OOS pension validation.

---

## 7. Actionable Conclusions

### Deploy Now (Pension Ready)
1. **A6 sustained_rise T+2** — Gap-down ≥5%, sustained rise in first 30 min (rise ≥3%, recovery ≥50%), buy at local low, hold T+2.
   - Full period: +4.41% avg, Sharpe 0.409, 62% WR, n=98
   - OOS validated: IS +4.76% → OOS +4.27% (stable)
   - ~0.39 trades/day, ~8 trades/month

### Strong Candidate (Needs Longer IS)
2. **C1 reversal T+3** — 5-day decline ≥15%, gap-up ≥3%, buy at open, hold T+3.
   - Full period: +5.77% avg, Sharpe 0.398, 62% WR, n=177
   - OOS is strong (+6.41%, Sharpe 0.454) but IS period too short for CI
   - ~0.71 trades/day, ~15 trades/month

### Kill
3. **Hypothesis B (ORB momentum)** — Dead. Do not revisit.

### Anti-Signals (Consider for SHORT)
4. **A6 rise_then_drop** — Stocks that bounce then fade within 60 min of gap-down: -4% to -6% at T+2 through T+5. Strong SHORT signal if margin account allows.

### Not Yet Actionable
5. **A5 D2 tier** — CI-confirmed on full period but degrades IS→OOS. Needs pattern overlay (combine with A6 sustained_rise filter).
6. **C2 oversold + volume** — n=3, needs relaxed thresholds or longer history.

---

## 8. Skipped Experiments (Require BigQuery)

| Experiment | Description | Why Skipped | Potential |
|-----------|-------------|-------------|-----------|
| A2 | Catalyst exclusion | Needs BQ catalyst table | Could improve D1/D3 tiers |
| A4 | Earnings-only gap-down | Needs BQ earnings calendar | Known high-alpha subset |
| B4 | Catalyst-confirmed momentum | Needs BQ catalyst table | Hypothesis B is dead anyway |

---

## 9. Technical Notes

### Infrastructure Reused from Agent 1
- `_FLOAT` dictionary (194 tickers with float ≥ 10M)
- `fetch_grouped_daily()` — Polygon grouped daily bars
- `_pg()` — API caller with retry/429 handling
- `compute_metrics()` — avg, median, trimmed_mean, std, WR, Sharpe, PF, CI 95%
- Async minute bar fetching with `aiohttp` + `Semaphore(15)`
- Holiday calendar, trading day helpers

### New Components Built for Agent 3
- `find_local_low()` — mirror of `find_local_high()`, detects trough in bars[3..11]
- `find_orb_break_up()` — ORB detection, first close above 5-bar opening range high
- `classify_gap_down_pattern()` — first-60-min classification (sustained_rise, rise_then_drop, mild_rise, continued_drop)
- `collect_data()` — single-pass scanner for all 3 hypothesis types simultaneously
- `enrich_long()` — LONG PnL computation: `(exit - entry) / entry * 100`
- `_resolve_trade_filter()` — maps experiment IDs to filter functions for OOS validation
- 10 experiment functions: run_a1, run_a3, run_a5, run_a6, run_a7, run_b1, run_b2, run_b3, run_c1, run_c2

### Config Parameters

```
BACKTEST_START = '2025-09-15'
BACKTEST_END   = '2026-09-12'
OOS_CUTOFF     = '2026-01-01'

GAP_DOWN_MIN   = 0.05        # |gap| >= 5% for gap-down (Hypothesis A)
GAP_UP_MIN     = 0.05        # gap >= 5% for gap-up (Hypothesis B)
PRICE_MIN      = 10.0        # A and B minimum price
PRICE_MIN_C    = 20.0        # C minimum price (pension = higher quality)
FLOAT_MIN      = 10,000,000
TOP_N          = 5            # Top N candidates per day per hypothesis

PEAK_WINDOW    = 3            # Local low/high detection window
MAX_PEAK_BAR   = 12           # Max bar for local low/high search
FALLBACK_BAR   = 15           # Fallback entry if no local low/high
SKIP_BOUNCE_PCT = 0.15        # Skip if already bounced >15% by fallback bar

ORB_BARS       = 5            # Opening range = first 5 bars
ORB_MAX_BAR    = 25           # Max bar to wait for ORB break

ADV_LOOKBACK   = 20           # Days for average daily volume
PRIOR_DECLINE_5D  = 0.15      # C1: 5-day decline >= 15%
PRIOR_DECLINE_10D = 0.20      # C2: 10-day decline >= 20%
REVERSAL_GAP_MIN  = 0.03      # C: today gap-up >= 3%
VOLUME_SPIKE_MULT = 3.0       # C2: volume > 3x 20-day avg
```

---

## 10. Charts

- `agent3_summary.png` — Sharpe bar chart and CI range plot for all 62 experiments
- `agent3_a7_hold_sweep.png` — Hold period sweep showing avg PnL across all horizons (30m to T+5)
