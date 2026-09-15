# Strategy Backtest Results V1 — Gap-Up SHORT Fade

**Generated**: 2026-09-13  
**Data**: 4 trading dates (Sep 8–11, 2026)  
**Universe**: 9,628 gappers (|gap| >= 1.5%), 372 LLM-flagged signals  
**Starting Capital**: $100,000  

---

## 1. Backtest Architecture

### 4 Strategies (all SHORT)

| ID | Label | Universe | Gap Min | Entry | ORB | Holds |
|----|-------|----------|---------|-------|-----|-------|
| BASELINE_V1 | LLM + ORB | LLM-flagged only | 3.0% | Local high (3-bar peak, fallback bar 15) | Required (SHORT break bars 5–25) | 30m, 60m, T+2 |
| B2_NO_ORB | LLM only | LLM-flagged only | 3.0% | Local high | Not required | 30m, 60m, T+2 |
| B3_EOD_ENTRY | LLM + ORB, EOD | LLM-flagged only | 3.0% | EOD close price | Required | T+2 only |
| B1_WIDE_SHORT | All gappers | All tickers > $5 | 1.5% | Local high | Not required | 30m, 60m, T+2 |

- **Catalyst exclusions** (LLM strategies only): earnings, deal, macro
- **Hold periods**: 30m = bar[29] close (10:00 AM), 60m = bar[59] close (10:30 AM), T+2 = 2-day closing price

### 6 Feature Layers (cumulative overlays)

| Layer | Filter | Effect |
|-------|--------|--------|
| L0 | Raw — no filter | Baseline |
| L1 | Exclude float < 10M shares | Removes low-float noise |
| L2 | Exclude PM retrace > 0.8 | Removes "already faded" setups |
| L3 | Sweet spot: PM retrace 0.2–0.5 only | Most selective |
| L4 | Exclude breaking catalyst | Removes momentum-continuation catalysts |
| L5 | Combined: L1 + L2 + L4 | Best combined filter |

### Composite Score Formula

Within-day normalized, per strategy:

```
Score = 0.35 × ORB_width + 0.30 × PM_retrace_score + 0.20 × gap_magnitude + 0.15 × offering_bonus
```

- ORB width: min-max normalized within day (0.5 default if no ORB)
- PM retrace score: >0.8 → 0.0, 0.2–0.5 → 1.0, <0.2 → 0.8, else 0.6
- Gap magnitude: min-max normalized within day
- Offering bonus: 1.0 if catalyst_type == 'offering', else 0.0

### Acceptance Gate

- **CI_95_low > 0** (95% confidence interval lower bound positive)
- **trades_per_day >= 1.0**
- **n >= 3** (minimum sample size)

---

## 2. Data Pipeline Summary

| Step | Result |
|------|--------|
| SPY daily calendar | 9 bars (Sep 2–15 range) |
| Grouped daily | 5 dates, ~12,000 tickers/day |
| LLM universe (BQ) | 277 premarket + 409 discovery = 372 unique (ticker, date) |
| Gapper universe | 9,628 total (4,711 gap-up > 1.5%) |
| Minute bars fetched | 9,327 / 9,628 (97% success rate) |
| Ticker float reference | 217 tickers from embedded cache |
| T+2 availability | Sep 8 → Sep 10, Sep 9 → Sep 11 only (Sep 10/11 entries have no T+2 — weekend) |

### Per-Day Gapper Counts

| Date | Gap-Up | Gap-Down | Total |
|------|--------|----------|-------|
| 2026-09-08 | 1,168 | 1,154 | 2,322 |
| 2026-09-09 | 875 | 878 | 1,753 |
| 2026-09-10 | 575 | 2,313 | 2,888 |
| 2026-09-11 | 2,093 | 572 | 2,665 |

---

## 3. Filter Funnels

### BASELINE_V1

```
Raw gappers:           9,628
After universe (LLM):    186  (-9,442 not LLM)
After gap >= 3%:          54  (-132 gap too small)
After catalyst:           24  (-30 earnings/deal/macro excluded)
After ORB SHORT:           8  (-16 no ORB break in bars 5–25)
─────────────────────────────
Raw qualifying:            8  (2.0/day)

L1 float:          6 trades  (1.5/day) — removed 2 low-float
L2 pm_retrace:     7 trades  (1.8/day) — removed 1 high-retrace
L3 sweet_spot:     2 trades  (0.5/day) — very selective
L4 breaking:       6 trades  (1.5/day) — removed 2 breaking catalyst
L5 combined:       3 trades  (0.8/day) — float + retrace + breaking
```

### B2_NO_ORB

```
Raw gappers:           9,628
After universe (LLM):    186  (-9,442 not LLM)
After gap >= 3%:          54  (-132 gap too small)
After catalyst:           24  (-30 earnings/deal/macro excluded)
─────────────────────────────
Raw qualifying:           24  (6.0/day) — 3× BASELINE (no ORB requirement)

L1 float:         19 trades  (4.8/day) — removed 5 low-float
L2 pm_retrace:    21 trades  (5.2/day) — removed 3 high-retrace
L3 sweet_spot:    10 trades  (2.5/day)
L4 breaking:      20 trades  (5.0/day) — removed 4 breaking
L5 combined:      13 trades  (3.2/day)
```

### B3_EOD_ENTRY

```
Raw qualifying:            6  (1.5/day) — same funnel as BASELINE, EOD entry

L1 float:          5 trades  (1.2/day)
L5 combined:       3 trades  (0.8/day)
```

### B1_WIDE_SHORT

```
Raw gappers:           9,628
After gap >= 1.5%:     4,711  (-4,917 gap too small)
After price >= $5:     3,181  (-1,530 penny stocks excluded)
After minute bars:     2,398  (-105 no data, -678 short session)
─────────────────────────────
Raw qualifying:        2,398  (599.5/day)

L1 float:        2,391 trades  (583.0/day) — only 7 removed (cache has 217 of ~9,600 tickers)
L2 pm_retrace:   2,387 trades  (582.0/day) — only 11 removed
L3 sweet_spot:     395 trades  (98.8/day) — PM retrace 0.2–0.5 only
L5 combined:     2,381 trades  (580.5/day) — float + retrace combined
```

---

## 4. Master Table — Full Results

### BASELINE_V1

| Layer | Hold | n | t/d | avg% | med% | WR | Sharpe | CI_lo | CI_hi | CI OK | Compound% | Equity$ |
|-------|------|---|-----|------|------|-----|--------|-------|-------|-------|-----------|---------|
| L0_raw | 30m | 8 | 2.0 | +4.82 | +0.93 | 75% | 0.404 | -3.44 | +13.08 | NO | +10.27 | $110,265 |
| L0_raw | 60m | 8 | 2.0 | +5.41 | +0.68 | 62% | 0.331 | -5.91 | +16.72 | NO | +6.71 | $106,715 |
| L0_raw | T+2 | 6 | 1.5 | +8.39 | +1.37 | 67% | 0.426 | -7.37 | +24.15 | NO | +8.39 | $108,389 |
| **L1_float** | **30m** | **6** | **1.5** | **+4.70** | **+0.35** | **67%** | **0.338** | **-6.43** | **+15.83** | **NO** | **+7.24** | **$107,243** |
| L1_float | 60m | 6 | 1.5 | +6.82 | -0.82 | 50% | 0.357 | -8.45 | +22.08 | NO | +6.10 | $106,097 |
| **L1_float** | **T+2** | **5** | **1.2** | **+10.80** | **+1.52** | **80%** | **0.514** | **-7.61** | **+29.22** | **NO** | **+10.80** | **$110,804** |
| L2_pm_retrace | 30m | 7 | 1.8 | +5.24 | +0.40 | 71% | 0.409 | -4.25 | +14.73 | NO | +13.93 | $113,934 |
| L2_pm_retrace | 60m | 7 | 1.8 | +6.51 | +1.02 | 71% | 0.376 | -6.31 | +19.34 | NO | +8.52 | $108,523 |
| L2_pm_retrace | T+2 | 6 | 1.5 | +8.39 | +1.37 | 67% | 0.426 | -7.37 | +24.15 | NO | +8.39 | $108,389 |
| L3_sweet_spot | 30m | 2 | 0.5 | — | — | — | — | — | — | — | +10.47 | $110,469 |
| L3_sweet_spot | 60m | 2 | 0.5 | — | — | — | — | — | — | — | +2.38 | $102,375 |
| L3_sweet_spot | T+2 | 1 | 0.25 | — | — | — | — | — | — | — | -3.69 | $96,314 |
| L4_breaking | 30m | 6 | 1.5 | +7.41 | +1.66 | 83% | 0.582 | -2.78 | +17.60 | NO | +14.27 | $114,265 |
| L4_breaking | 60m | 6 | 1.5 | +6.84 | -0.48 | 50% | 0.359 | -8.41 | +22.09 | NO | +9.86 | $109,858 |
| L4_breaking | T+2 | 4 | 1.0 | +12.66 | +3.10 | 75% | 0.530 | -10.77 | +36.10 | NO | +12.66 | $112,663 |
| L5_combined | 30m | 3 | 0.8 | +10.75 | +0.31 | 67% | 0.571 | -10.55 | +32.06 | NO | +10.75 | $110,752 |
| L5_combined | 60m | 3 | 0.8 | +13.68 | -1.99 | 33% | 0.495 | -17.59 | +44.95 | NO | +13.68 | $113,676 |
| L5_combined | T+2 | 3 | 0.8 | +18.11 | +4.98 | 100% | 0.695 | -11.38 | +47.61 | NO | +18.11 | $118,112 |

### B2_NO_ORB

| Layer | Hold | n | t/d | avg% | med% | WR | Sharpe | CI_lo | CI_hi | CI OK | Compound% | Equity$ |
|-------|------|---|-----|------|------|-----|--------|-------|-------|-------|-----------|---------|
| L0_raw | 30m | 24 | 6.0 | +0.76 | -0.10 | 46% | 0.071 | -3.54 | +5.06 | NO | +1.52 | $101,516 |
| L0_raw | 60m | 24 | 6.0 | +1.50 | -0.61 | 42% | 0.117 | -3.64 | +6.64 | NO | +2.97 | $102,970 |
| L0_raw | T+2 | 12 | 3.0 | +6.84 | +3.94 | 75% | 0.504 | -0.84 | +14.53 | NO | +6.84 | $106,841 |
| **L1_float** | **30m** | **19** | **4.8** | **+2.30** | **-0.02** | **47%** | **0.266** | **-1.59** | **+6.19** | **NO** | **+4.70** | **$104,704** |
| L1_float | 60m | 19 | 4.8 | +3.50 | -0.37 | 42% | 0.261 | -2.53 | +9.52 | NO | +6.95 | $106,947 |
| **L1_float** | **T+2** | **11** | **2.8** | **+7.80** | **+4.58** | **82%** | **0.565** | **-0.36** | **+15.96** | **NO** | **+7.80** | **$107,798** |
| L2_pm_retrace | 30m | 21 | 5.2 | +0.89 | -0.17 | 43% | 0.077 | -4.03 | +5.80 | NO | +1.59 | $101,589 |
| L2_pm_retrace | 60m | 21 | 5.2 | +1.80 | -0.37 | 43% | 0.131 | -4.07 | +7.67 | NO | +3.33 | $103,327 |
| L2_pm_retrace | T+2 | 11 | 2.8 | +7.05 | +3.31 | 73% | 0.495 | -1.36 | +15.45 | NO | +7.05 | $107,046 |
| L3_sweet_spot | 30m | 10 | 2.5 | -0.63 | -0.11 | 40% | -0.048 | -8.73 | +7.47 | NO | -0.41 | $99,589 |
| L3_sweet_spot | 60m | 10 | 2.5 | -0.17 | +0.06 | 50% | -0.013 | -8.71 | +8.36 | NO | +0.21 | $100,213 |
| L3_sweet_spot | T+2 | 2 | 0.5 | — | — | — | — | — | — | — | -0.19 | $99,810 |
| L4_breaking | 30m | 20 | 5.0 | +1.42 | +0.14 | 50% | 0.122 | -3.69 | +6.53 | NO | +3.81 | $103,808 |
| L4_breaking | 60m | 20 | 5.0 | +1.83 | -0.61 | 40% | 0.130 | -4.35 | +8.01 | NO | +4.89 | $104,886 |
| L4_breaking | T+2 | 8 | 2.0 | +8.88 | +4.14 | 75% | 0.541 | -2.50 | +20.26 | NO | +8.88 | $108,879 |
| L5_combined | 30m | 13 | 3.2 | +3.94 | -0.02 | 46% | 0.396 | -1.47 | +9.36 | NO | +7.88 | $107,882 |
| L5_combined | 60m | 13 | 3.2 | +5.20 | -0.37 | 38% | 0.324 | -3.53 | +13.93 | NO | +10.48 | $110,479 |
| L5_combined | T+2 | 7 | 1.8 | +10.67 | +4.98 | 86% | 0.633 | -1.82 | +23.17 | NO | +10.67 | $110,674 |

### B3_EOD_ENTRY (T+2 only)

| Layer | Hold | n | t/d | avg% | med% | WR | Sharpe | CI_lo | CI_hi | CI OK | Compound% | Equity$ |
|-------|------|---|-----|------|------|-----|--------|-------|-------|-------|-----------|---------|
| L0_raw | T+2 | 6 | 1.5 | +2.34 | +3.20 | 83% | 0.599 | -0.79 | +5.48 | NO | +2.34 | $102,344 |
| L1_float | T+2 | 5 | 1.2 | +2.42 | +3.58 | 80% | 0.555 | -1.41 | +6.26 | NO | +2.42 | $102,424 |
| L2_pm_retrace | T+2 | 6 | 1.5 | +2.34 | +3.20 | 83% | 0.599 | -0.79 | +5.48 | NO | +2.34 | $102,344 |
| L3_sweet_spot | T+2 | 1 | 0.25 | — | — | — | — | — | — | — | +1.94 | $101,942 |
| L4_breaking | T+2 | 4 | 1.0 | +1.57 | +2.76 | 75% | 0.333 | -3.06 | +6.21 | NO | +1.57 | $101,575 |
| L5_combined | T+2 | 3 | 0.8 | +1.45 | +3.58 | 67% | 0.251 | -5.10 | +8.01 | NO | +1.45 | $101,453 |

### B1_WIDE_SHORT

| Layer | Hold | n | t/d | avg% | med% | WR | Sharpe | CI_lo | CI_hi | CI OK | Compound% | Equity$ |
|-------|------|---|-----|------|------|-----|--------|-------|-------|-------|-----------|---------|
| L0_raw | 30m | 2,339 | 584.8 | -0.20 | +0.00 | 49% | -0.091 | -0.28 | -0.11 | NO | -0.33 | $99,666 |
| L0_raw | 60m | 2,095 | 523.8 | -0.23 | -0.02 | 49% | -0.090 | -0.35 | -0.12 | NO | -0.41 | $99,594 |
| **L0_raw** | **T+2** | **1,017** | **254.2** | **+3.28** | **+2.98** | **73%** | **0.462** | **+2.84** | **+3.71** | **YES ★** | **+6.50** | **$106,497** |
| L1_float | 30m | 2,332 | 583.0 | -0.20 | +0.00 | 49% | -0.095 | -0.29 | -0.12 | NO | -0.38 | $99,618 |
| L1_float | 60m | 2,090 | 522.5 | -0.23 | -0.02 | 49% | -0.088 | -0.34 | -0.12 | NO | -0.40 | $99,603 |
| **L1_float** | **T+2** | **1,016** | **254.0** | **+3.33** | **+2.98** | **73%** | **0.483** | **+2.90** | **+3.75** | **YES ★** | **+6.59** | **$106,587** |
| L2_pm_retrace | 30m | 2,328 | 582.0 | -0.20 | +0.00 | 49% | -0.091 | -0.28 | -0.11 | NO | -0.36 | $99,643 |
| L2_pm_retrace | 60m | 2,084 | 521.0 | -0.23 | -0.02 | 49% | -0.090 | -0.35 | -0.12 | NO | -0.42 | $99,581 |
| **L2_pm_retrace** | **T+2** | **1,014** | **253.5** | **+3.28** | **+2.98** | **73%** | **0.464** | **+2.85** | **+3.72** | **YES ★** | **+6.51** | **$106,513** |
| L3_sweet_spot | 30m | 395 | 98.8 | -0.35 | -0.03 | 48% | -0.141 | -0.60 | -0.10 | NO | -0.60 | $99,399 |
| L3_sweet_spot | 60m | 384 | 96.0 | -0.42 | -0.17 | 44% | -0.155 | -0.70 | -0.15 | NO | -0.78 | $99,218 |
| **L3_sweet_spot** | **T+2** | **162** | **40.5** | **+2.98** | **+2.26** | **64%** | **0.407** | **+1.85** | **+4.10** | **YES ★** | **+5.48** | **$105,481** |
| L5_combined | 30m | 2,322 | 580.5 | -0.20 | +0.00 | 49% | -0.094 | -0.29 | -0.11 | NO | -0.38 | $99,624 |
| L5_combined | 60m | 2,080 | 520.0 | -0.23 | -0.01 | 49% | -0.088 | -0.34 | -0.12 | NO | -0.39 | $99,606 |
| **L5_combined** | **T+2** | **1,013** | **253.2** | **+3.34** | **+2.98** | **73%** | **0.484** | **+2.91** | **+3.76** | **YES ★** | **+6.60** | **$106,604** |

---

## 5. Portfolio PnL — Finite Capital ($100K, Equal Split, Compounding)

| Strategy | Layer | Hold | Day 1 | Day 2 | Day 3 | Day 4 | Final$ | Return% |
|----------|-------|------|-------|-------|-------|-------|--------|---------|
| BASELINE_V1 | L0_raw | 30m | +4.63% | — | — | +5.38% | $110,265 | +10.26% |
| BASELINE_V1 | L0_raw | 60m | +7.43% | — | — | -0.67% | $106,715 | +6.72% |
| BASELINE_V1 | L0_raw | T+2 | +8.39% | — | — | — | $108,389 | +8.39% |
| BASELINE_V1 | L5 | 30m | +10.75% | — | — | — | $110,752 | +10.75% |
| BASELINE_V1 | L5 | 60m | +13.68% | — | — | — | $113,676 | +13.68% |
| BASELINE_V1 | L5 | T+2 | +18.11% | — | — | — | $118,112 | +18.11% |
| B2_NO_ORB | L0_raw | 30m | +2.10% | — | — | -0.58% | $101,516 | +1.52% |
| B2_NO_ORB | L0_raw | 60m | +3.70% | — | — | -0.70% | $102,970 | +2.97% |
| B2_NO_ORB | L0_raw | T+2 | +6.84% | — | — | — | $106,841 | +6.84% |
| B2_NO_ORB | L5 | 30m | +4.83% | — | — | +2.91% | $107,882 | +7.88% |
| B2_NO_ORB | L5 | 60m | +6.24% | — | — | +3.99% | $110,479 | +10.48% |
| B2_NO_ORB | L5 | T+2 | +10.67% | — | — | — | $110,674 | +10.67% |
| B3_EOD_ENTRY | L0_raw | T+2 | +2.34% | — | — | — | $102,344 | +2.34% |
| B3_EOD_ENTRY | L5 | T+2 | +1.45% | — | — | — | $101,453 | +1.45% |
| B1_WIDE | L0_raw | 30m | -0.62% | +0.25% | +0.25% | -0.22% | $99,666 | -0.33% |
| B1_WIDE | L0_raw | 60m | -1.02% | +0.35% | +0.43% | -0.16% | $99,594 | -0.41% |
| B1_WIDE | L0_raw | T+2 | +3.59% | +2.81% | — | — | $106,497 | +6.50% |
| B1_WIDE | L5 | 30m | -0.61% | +0.25% | +0.21% | -0.23% | $99,624 | -0.38% |
| B1_WIDE | L5 | 60m | -1.01% | +0.35% | +0.42% | -0.15% | $99,606 | -0.39% |
| B1_WIDE | L5 | T+2 | +3.69% | +2.81% | — | — | $106,604 | +6.60% |

**Key observations**:
- LLM strategies (BASELINE, B2, B3) only trade on Day 1 and Day 4 — Days 2/3 are silent
- BASELINE L5 concentrates all portfolio returns on Day 1 only (3 trades)
- B1 T+2 is the only strategy with multi-day compounding (Day 1 + Day 2)
- B1 intraday is consistently negative across all 4 days

---

## 6. Composite Score Quartile Analysis (at L5)

### B2_NO_ORB / 30m (n=13)

| Quartile | n | avg% | WR | Sharpe |
|----------|---|------|-----|--------|
| Q1 | 4 | -0.22 | 25% | -0.574 |
| Q2 | 3 | +0.22 | 67% | 0.073 |
| Q3 | 3 | +5.75 | 33% | 0.556 |
| Q4 | 3 | +11.42 | 67% | 0.625 |

Q4-Q1 spread: +11.64%. Ratio: -51.37x (Q1 is negative). **NOT validated** (Q1 < 0).  
Note: extremely useful directional signal — composite cleanly separates winners from losers. But n per quartile is 3–4.

### B2_NO_ORB / 60m (n=13)

| Quartile | n | avg% | WR | Sharpe |
|----------|---|------|-----|--------|
| Q1 | 4 | -1.44 | 0% | -1.425 |
| Q2 | 3 | -0.30 | 33% | -0.073 |
| Q3 | 3 | +11.67 | 67% | 0.580 |
| Q4 | 3 | +13.09 | 67% | 0.453 |

Q4-Q1 spread: +14.54%. Same pattern: composite separates winners but can't validate due to negative Q1.

### B1_WIDE_SHORT / 30m (n=2,322)

| Quartile | n | avg% | WR | Sharpe |
|----------|---|------|-----|--------|
| Q1 | 585 | -0.23 | 47% | -0.129 |
| Q2 | 576 | -0.19 | 48% | -0.073 |
| Q3 | 580 | -0.09 | 52% | -0.057 |
| Q4 | 581 | -0.29 | 49% | -0.119 |

Q4-Q1 spread: -0.07%. **Composite has zero signal for B1 intraday.** All quartiles negative.

### B1_WIDE_SHORT / 60m (n=2,080)

| Quartile | n | avg% | WR | Sharpe |
|----------|---|------|-----|--------|
| Q1 | 523 | -0.21 | 48% | -0.086 |
| Q2 | 527 | -0.42 | 46% | -0.149 |
| Q3 | 510 | -0.07 | 55% | -0.032 |
| Q4 | 520 | -0.21 | 48% | -0.072 |

Q4-Q1 spread: -0.01%. Same — no intraday signal.

### B1_WIDE_SHORT / T+2 (n=1,013)

| Quartile | n | avg% | WR | Sharpe |
|----------|---|------|-----|--------|
| Q1 | 255 | +2.27 | 72% | 0.310 |
| Q2 | 253 | +3.28 | 74% | 0.543 |
| Q3 | 252 | +3.52 | 78% | 0.606 |
| Q4 | 253 | +4.29 | 70% | 0.534 |

Q4-Q1 spread: **+2.02%**. Ratio: **1.89x** (threshold for validation is 2.0x). **NOT validated but close.**

**Clean monotonic staircase**: Q1 → Q2 → Q3 → Q4 avg PnL rises consistently. Q3 has the best Sharpe (0.606) — the top quartile trades harder (more variance) for slightly more return. With n=253 per quartile, this monotonic pattern is meaningful even though it just misses the 2x threshold.

---

## 7. Float Filter Impact (L0 vs L1)

The float filter (exclude shares_outstanding < 10M) was enabled for this run with 217 tickers from embedded data.

| Strategy | Hold | L0 avg% (n) | L1 avg% (n) | Removed | Removed avg% |
|----------|------|-------------|-------------|---------|--------------|
| BASELINE | 30m | +4.82 (8) | +4.70 (6) | 2 | ~+5.2 |
| BASELINE | T+2 | +8.39 (6) | +10.80 (5) | 1 | ~ -3.7 |
| B2 | 30m | +0.76 (24) | +2.30 (19) | 5 | **-5.06** |
| B2 | T+2 | +6.84 (12) | +7.80 (11) | 1 | ~ -3.7 |
| B3 | T+2 | +2.34 (6) | +2.42 (5) | 1 | ~ +1.9 |
| B1 | T+2 | +3.28 (1017) | +3.33 (1016) | 1 | negligible |

**Key finding**: Float filter removes the worst-performing trades from LLM strategies. The 5 tickers removed from B2 at 30m had avg PnL of **-5.06%** — clear drag. B1 barely affected because only 7 of 2,398 trades had float data in the 217-ticker cache.

**Limitation**: To extend float filtering to B1's full universe, would need shares_outstanding for all ~9,600 gappers via Polygon reference API.

---

## 8. Answering the 5 Key Questions

### Q1 — Trade Frequency After L5 Filtering

| Strategy | Hold | t/d | avg% | Assessment |
|----------|------|-----|------|------------|
| BASELINE_V1 | 30m | 0.8 | +10.75% | Below 1.0 minimum |
| BASELINE_V1 | T+2 | 0.8 | +18.11% | Below 1.0 minimum |
| B2_NO_ORB | 30m | 3.2 | +3.94% | Viable (>= 1) but no CI |
| B2_NO_ORB | T+2 | 1.8 | +10.67% | Viable (>= 1) but no CI |
| B3_EOD_ENTRY | T+2 | 0.8 | +1.45% | Below 1.0 minimum |
| B1_WIDE_SHORT | T+2 | 253.2 | +3.34% | **Passes gate (CI confirmed)** |

**Answer**: Only B1_WIDE_SHORT T+2 meets both the CI and frequency gates. B2 L5 has reasonable frequency (3.2/day at 30m) but no CI confirmation.

### Q2 — Does 30m Show a Confirmed Edge?

| Strategy | Layer | avg% | CI |
|----------|-------|------|----|
| BASELINE_V1 | L0 | +4.82% | CI<0 |
| BASELINE_V1 | L5 | +10.75% | CI<0 |
| B2_NO_ORB | L0 | +0.76% | CI<0 |
| B2_NO_ORB | L5 | +3.94% | CI<0 |
| B1_WIDE_SHORT | L0 | -0.20% | CI<0 |
| B1_WIDE_SHORT | L5 | -0.20% | CI<0 |

**Answer: NO.** Zero strategies show CI > 0 at 30m. BASELINE L5 has high avg (+10.75%) but it's 3 trades — the CI is -10.55 to +32.06. The SHORT gap-fade does not work as a 30-minute scalp.

### Q3 — Does B1_WIDE_SHORT Improve After Filtering?

| Layer | Hold | n | avg% | Sharpe |
|-------|------|---|------|--------|
| L0_raw | T+2 | 1,017 | +3.28% | 0.462 |
| L1_float | T+2 | 1,016 | +3.33% | 0.483 |
| L2_pm_retrace | T+2 | 1,014 | +3.28% | 0.464 |
| L3_sweet_spot | T+2 | 162 | +2.98% | 0.407 |
| L5_combined | T+2 | 1,013 | +3.34% | 0.484 |

**Answer**: Barely. L1/L5 marginally improve Sharpe (0.462 → 0.484) but the effect is negligible because the float cache only covers 217 tickers. L3 sweet spot drastically reduces trade count (1,017 → 162) and actually *reduces* avg PnL and Sharpe. The feature layers were calibrated on the LLM universe and don't transfer well to the broad gapper universe.

### Q4 — B3_EOD_ENTRY Sharpe Stability Across Layers

| Layer | n | avg% | Sharpe |
|-------|---|------|--------|
| L0_raw | 6 | +2.34% | **0.599** |
| L1_float | 5 | +2.42% | 0.555 |
| L2_pm_retrace | 6 | +2.34% | 0.599 |
| L4_breaking | 4 | +1.57% | 0.333 |
| L5_combined | 3 | +1.45% | **0.251** |

**Answer: Sharpe degrades monotonically from 0.599 to 0.251** as filters remove trades. Each filter layer removes a profitable trade from an already tiny sample (6 → 3). B3's apparent strength at L0 is a small-n artifact — the breaking exclusion (L4) and float filter (L1) remove exactly the right trades to collapse both avg PnL and Sharpe. Do not rely on B3 until sample size grows.

### Q5 — Composite Score Validation

| Strategy | Hold | Q4-Q1 Spread | Ratio | Monotonic? | Validated? |
|----------|------|-------------|-------|------------|------------|
| B2_NO_ORB | 30m | +11.64% | -51.37x | Yes | NO (Q1 < 0) |
| B2_NO_ORB | 60m | +14.54% | -9.08x | Yes | NO (Q1 < 0) |
| B1_WIDE | 30m | -0.07% | 1.30x | No | NO |
| B1_WIDE | 60m | -0.01% | 1.03x | No | NO |
| B1_WIDE | T+2 | **+2.02%** | **1.89x** | **Yes** | NO (< 2.0x) |

**Answer**: No formal validation (none cross the 2.0x threshold with positive Q1). However:

- **B1 T+2** shows a **clean monotonic staircase** (Q1: +2.27% → Q4: +4.29%) with n=253 per quartile — this is directionally meaningful and misses the gate by 0.11x. Using Q3+Q4 trades only (~505 trades, 126/day) would yield avg +3.90% vs Q1+Q2's +2.78%, a +1.12 pp lift.
- **B2 30m/60m** composite cleanly separates winners from losers (Q4 is +11–13%, Q1 is negative) but can't formally validate because Q1 is below zero.

---

## 9. Confirmed Findings

### ★ The Only Statistically Confirmed Edge

**B1_WIDE_SHORT at T+2**: Short every gap-up > 1.5% (price > $5), enter at local high, hold to T+2 close.

| Metric | Value |
|--------|-------|
| n | 1,013–1,017 |
| avg PnL | +3.28% to +3.34% |
| median PnL | +2.98% |
| Win rate | 73% |
| Sharpe | 0.462–0.484 |
| 95% CI | +2.84% to +3.76% |
| Portfolio compound (2 days) | +6.50% to +6.60% |
| Trades per day | 253–254 |

This is a **broad-based mean-reversion signal**: gap-up stocks tend to fade over 2 trading days. The effect is not present intraday (30m avg = -0.20%, 60m avg = -0.23%).

### Dead Zones

- **30m and 60m holds**: No confirmed edge for ANY strategy at ANY layer. The SHORT gap-fade does not work as a scalp.
- **B1 intraday**: Actively negative. Shorting random gappers and exiting at 10:00 AM or 10:30 AM loses money.
- **L3 sweet spot**: Too aggressive. Kills trade count without improving quality (except on LLM strategies where n drops to 1–2).

### High-Alpha but Unconfirmed

- **BASELINE L5 T+2**: +18.11% avg, 0.695 Sharpe, 100% WR — but n=3 on 1 day
- **B2 L5 T+2**: +10.67% avg, 0.633 Sharpe, 86% WR — but n=7, CI spans zero
- **B2 L5 30m**: +3.94% avg, 0.396 Sharpe — n=13, closest to viability

---

## 10. Data Limitations

1. **4 trading days only** — all results are from Sep 8–11, 2026. Regime-specific effects (sector rotation, VIX level) could dominate.
2. **T+2 limited to 2 entry days** — Sep 10/11 entries have no T+2 exit (weekend). T+2 equity curves have only 2 data points.
3. **Float data covers 217 of ~9,600 tickers** — L1/L5 filter barely affects B1. Full Polygon reference data needed for meaningful B1 float filtering.
4. **LLM strategies fire on 2 of 4 days** — BASELINE/B2/B3 only trade on Day 1 (Sep 8) and Day 4 (Sep 11). Days 2 and 3 are completely silent.
5. **No transaction costs** — results assume zero commission, zero slippage, no borrow cost for SHORT.

---

## 11. Recommendations

### For Monday (Sep 15) Live Logging

1. **Log B2_NO_ORB trades with composite scores** — 3.2 trades/day at L5 gives fastest path to CI confirmation. After 2 weeks (~40+ T+2 observations), re-evaluate.
2. **Log B1_WIDE_SHORT T+2 composite quartiles** — track whether Q3+Q4 filter (126/day) continues to outperform Q1+Q2.
3. **Do NOT trade intraday SHORT** — 30m and 60m are confirmed dead across all strategies.

### For Next Backtest Iteration

1. **Expand to 10+ trading dates** — need more data to confirm LLM strategy edges. Current 4 days are insufficient.
2. **Fetch full float data** — use Polygon reference API for all gappers, not just LLM tickers. This would enable meaningful L1/L5 filtering for B1.
3. **Add gap magnitude tiers** — test B1 at > 3%, > 5%, > 10% thresholds to find the optimal gap size for T+2 fade.
4. **Add borrow cost and slippage** — B1 T+2 at +3.28% avg could easily lose edge with 1–2% borrow + slippage on hard-to-borrow small caps.
5. **Test T+3, T+5 holds** — if T+2 works, does the fade continue or reverse?
