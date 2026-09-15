# V3 Experiment Results — Scanner Optimization

**Date:** 2026-09-12  
**Dataset:** 316 trades across 4 dates (2026-09-08 to 2026-09-11)  
**Data source:** v2_features_full.csv + Polygon minute/daily bars  

---

## Executive Summary

10 experiments across 3 priority tiers. The emerging picture: **the LLM is not a stock picker — it's a catalyst classifier**. The real edge is gap-fade (SHORT gappers at open), filtered by ORB agreement, sized by gap magnitude, with no stop-loss.

### Top 5 Actionable Findings

1. **Enter SHORT at market open (09:30)** — best timing at +4.98% T+2, 85.7% WR. Waiting for 10:30 or ORB break reduces returns (E5)
2. **Size by |gap_pct|, not expected_move** — gap-sizing: Sharpe 0.05→0.75. Expected-move sizing actively hurts (r=-0.16) (E10)
3. **Non-earnings + ORB confirms = +1.548%, 64% WR** — strongest filter combo across all experiments (E4)
4. **LLM adds NO selection value** — 1,810 gappers >3% outperform the 316 LLM-flagged (+0.326% vs +0.014%) (E4)
5. **Every stop-loss design destroys value** — trailing stops same conclusion as fixed stops. "No stop" is the finding (E1)

---

## P0 Experiments (Must Run Before Live)

### E1: ATR Trailing Stop Design

**Goal:** Test trailing stops from high-water-mark at k × ATR.

**Method:** Bar-by-bar simulation on 316 trades using minute bars. Trail from HWM (LONG) or LWM (SHORT). k = 0.5, 0.75, 1.0, 1.5 × ATR. Horizons: 30m, 60m, 120m, EOD.

**Results — ATR Trailing Stops (All Trades):**

| k | 30m | 60m | 120m | EOD |
|---|-----|-----|------|-----|
| 0.5 | -0.13% | -0.38% | -0.43% | -0.47% |
| 0.75 | -0.11% | -0.31% | -0.42% | -0.50% |
| 1.0 | -0.07% | -0.23% | -0.36% | -0.42% |
| 1.5 | -0.06% | -0.15% | -0.28% | -0.38% |
| No SL | +0.16% | +0.01% | -0.09% | -0.20% |

Every trailing config underperforms no-SL. Stop trigger rates: 6% (k=1.5, 30m) to 56% (k=0.5, EOD).

**E1b — Hybrid (trail 30m then hold to T+2, SHORT only):**

| k | Avg PnL | Trimmed | WR | Stopped in 30m |
|---|---------|---------|-----|----------------|
| 0.5 | +0.65% | +0.68% | 59.8% | 15 |
| 0.75 | +0.95% | +1.10% | 60.8% | 12 |
| 1.0 | +1.16% | +1.31% | 62.7% | 10 |
| 1.5 | +1.48% | +1.57% | 63.7% | 7 |
| No stop, pure T+2 | +4.42% | +4.87% | 84.0% | 0 |

The hybrid is positive but dilutes the T+2 signal. The stop removes winners — the T+2 hold effect is the entire edge.

**Conclusion:** No stop-loss design works. This is consistent across V1 (fixed SL), V2 (fixed + ATR SL), and V3 (trailing + hybrid). Fat-tailed winners need room.

### E2: ORB Filter Refinement

**Goal:** Test 2/5/10/15-min ORB timeframes with breakout thresholds and volume filters.

**Results — ORB Agree/Disagree Spread (60m PnL, threshold=0.0):**

| ORB Window | Agree | n | Disagree | n | Spread |
|------------|-------|---|----------|---|--------|
| 2min | +1.04% | 44 | -1.95% | 26 | **+2.99%** |
| 5min (current) | +1.51% | 26 | +0.39% | 17 | +1.12% |
| 10min | weak across the board | | | | |
| 15min | good agree PnL but all LOW-N | | | | |

**With breakout thresholds (2min ORB):**

| Threshold | Agree | n | Disagree | n | Spread |
|-----------|-------|---|----------|---|--------|
| 0.0% | +1.04% | 44 | -1.95% | 26 | +2.99% |
| 0.1% | varies | | | | |
| 0.25% | varies | | | | |
| 0.5% | +3.50% | 10 | -4.24% | 11 | +7.74% (LOW-N) |

Trimmed mean confirms 2min spreads are robust (+2.32% to +5.08%).

Volume filter (>1.5x avg) decimates sample sizes — not useful.

**Conclusion:** 2min ORB may be better than current 5min (larger spread), but many cells are LOW-N (<20). Worth testing live but needs more data to confirm.

### E3: Earnings Exclusion + Catalyst Breakdown

**Goal:** Rank catalyst types by performance.

**Results — Catalyst Type Performance (60m, no SL):**

| Catalyst Type | n | Avg PnL | Trim Mean | WR | Direction |
|---------------|---|---------|-----------|-----|-----------|
| offering | 24 | **+2.30%** | +2.21% | 62.5% | LONG+SHORT |
| unknown | 48 | +1.22% | +1.06% | 54.2% | mixed |
| SEC lawsuit | 5 | +2.18% | — | 60.0% | LOW-N |
| premarket_move | 20 | +0.45% | +0.52% | 55.0% | mixed |
| analyst | 32 | +0.24% | +0.38% | 53.1% | mixed |
| macro | 28 | -0.97% | -0.82% | 42.9% | mixed |
| deal | 22 | **-1.55%** | -1.38% | 36.4% | mixed |
| earnings | 44 | **-0.74%** | -0.66% | 43.2% | mixed |

**Earnings exclusion impact:**

| Metric | With Earnings | Without Earnings | Delta |
|--------|--------------|------------------|-------|
| 60m PnL | +0.01% | +0.25% | +0.24% |
| T+2 PnL | -0.14% | +0.30% | +0.44% |
| WR 60m | 51.1% | 53.2% | +2.1pp |

**Good-catalyst-only filter** (offering, unknown, analyst, premarket_move, SEC lawsuit): **+0.92% at 60m** (n=120/272). Strongest catalyst filter but cuts universe by 56%.

**Conclusion:** Exclude earnings (confirmed V2 finding). Exclude deal + macro. Good-catalyst filter powerful but reduces sample significantly.

---

## P1 Experiments (High Value)

### E4: Pure Price Screen vs LLM Selection

**Goal:** Does a simple price/volume screen match or beat LLM-based selection?

**Full universe (Polygon grouped daily, gap >3%):**

| Metric | Full Universe | LLM-Flagged |
|--------|-------------|-------------|
| Tickers | 1,810 | 316 |
| Gap-fade PnL | **+0.326%** | +0.014% |
| Median PnL | +0.112% | +0.065% |
| Win rate | 50.9% | 51.1% |

LLM does NOT add selection value. The raw universe outperforms.

**Gap direction matters:** Gap-up fade (SHORT): +0.807%, 57.9% WR. Gap-down fade (LONG): -0.114%.

**Best filter combo (LLM-flagged subset):**

| Filter | Avg PnL | WR | n |
|--------|---------|-----|---|
| All trades | +0.014% | 51.1% | 316 |
| Non-earnings | +0.25% | 53.2% | 272 |
| Non-earnings + ORB confirms | **+1.548%** | **64.0%** | 111 |

**Conclusion:** Replace LLM universe with price screen (gappers >3%). Use ORB for direction. Exclude earnings. LLM adds value only as catalyst classifier, not selector.

### E5: SHORT Swing Entry Timing

**Goal:** Test different entry times for SHORT trades, measuring T+2 outcome.

**Results (SHORT trades, T+2 PnL):**

| Entry Strategy | n | Avg PnL | Trimmed | WR | Sharpe |
|---------------|---|---------|---------|-----|--------|
| **Market open (09:30)** | 49 | **+4.98%** | **+5.24%** | **85.7%** | 0.421 |
| 10:30 AM | 49 | +4.34% | +4.59% | 85.7% | 0.487 |
| ORB break below | 35 | +3.52% | +4.48% | 82.9% | 0.285 |
| Scanner entry (baseline) | 50 | +4.42% | +4.87% | 84.0% | 0.370 |

ORB break below or_low occurs in first 25 minutes (09:35-09:59) for 76/110 trades where it happens.

**Conclusion:** Enter SHORT immediately at open. Waiting for a "better" entry point actually reduces returns. The gap-fade SHORT is strongest when entered early.

### E6: n_runs Position Sizing

**Goal:** Weight trades by consistency count (n_runs).

**n_runs buckets:**

| Bucket | n | Avg 60m | Avg EOD |
|--------|---|---------|---------|
| n_runs <= 2 | ~120 | -0.09% | -0.30% |
| n_runs == 3 | ~120 | +0.10% | +0.05% |
| n_runs >= 4 | ~37 | **+0.57%** | **+1.14%** |

**Weighting comparison (60m):**

| Strategy | Avg PnL | Sharpe |
|----------|---------|--------|
| Equal weight | +0.01% | 0.71 |
| 1.5x for n>=4 | +0.12% | 1.02 |
| 2.0x for n>=4 | +0.16% | **1.12** |

Caveat: at T+2 horizon, weighting hurts (-0.52%) due to n_runs=3 bucket underperforming at longer horizons.

**Conclusion:** n_runs weighting improves intraday Sharpe. Don't apply to swing (T+2) trades.

---

## P2 Experiments (Exploratory)

### E7: Market Regime Filter

**Goal:** Test whether SPY trend affects scanner performance.

**Results (n=4 dates — VERY LOW-N):**

| Regime | Dates | Avg 60m | SHORT | LONG |
|--------|-------|---------|-------|------|
| Trending up | 1 | +0.260% | — | — |
| Trending down | 3 | -0.116% | +0.329% | -0.648% |

- SPY intraday vs trade PnL: r=-0.884, p=0.116 (suggestive but not significant)
- SHORT alpha: +0.296% (meaningful). LONG alpha: -0.221% (negative)
- Alpha decays at longer horizons

**Conclusion:** Suggestive that trending-down markets hurt LONG but help SHORT. Need far more data (4 dates is insufficient for regime conclusions).

### E8: PM Path + ORB Combination

**Goal:** Do PM retrace and ORB filters stack?

**Key combos (60m PnL):**

| Combo | Avg PnL | WR | n |
|-------|---------|-----|---|
| No filter | +0.01% | 51% | 316 |
| ORB agrees only | +0.84% | 57% | ~150 |
| PM partial + ORB agrees | **+1.77%** | 60% | 88 |
| PM held + ORB agrees | below additive prediction | | |

**SHORT + ORB agrees at T+2:** **+6.45%, Sharpe 0.881** (n=28)

PM "held" + ORB is redundant — interaction is -0.87% (the two filters partially overlap). Stacking PM on top of ORB narrows sample without improving returns.

**Conclusion:** ORB agreement is the dominant filter. PM retrace adds minimal marginal value. Keep it simple: direction + ORB agrees.

### E9: Sympathy Tickers

**Goal:** Can related tickers (smaller gaps in same catalyst group) be traded?

28 multi-ticker clusters found (same date + catalyst type), covering 234/316 trades.

| Position | 60m PnL | WR | n |
|----------|---------|-----|---|
| Lead (largest gap) | +1.64% | 40% | varies |
| Follow (smaller gap) | -0.23% | 52% | 84 |

Lead ticker has higher mean but extreme variance (outlier-driven). Follow tickers cluster around breakeven.

**Conclusion:** Inconclusive. Data doesn't support sympathy trading without richer cross-ticker linkage data (sector, sub-industry, supply chain).

### E10: Magnitude Position Sizing

**Goal:** Size positions by predicted or realized volatility measures.

**Expected-move sizing (LLM's expected_move_pct):**
- Correlation with realized PnL: **r = -0.16, p = 0.008** (NEGATIVE)
- Larger LLM-predicted moves → worse PnL
- Q1 (smallest expected move): +0.73%. Q4 (largest): -0.67%
- **Expected-move weighting HURTS:** -0.61% delta at 60m

**Gap-pct sizing (|gap_pct|):**
- Correlation with |realized PnL|: **r = +0.40, p < 0.0001** (POSITIVE)
- Gap-weighted Sharpe: 0.05 → **0.75** at 60m
- SHORT gap-sizing: Sharpe 0.81 → **1.78**

**Conclusion:** Size by |gap_pct|. Never size by expected_move_pct — it's an anti-signal for PnL magnitude.

---

## Combined Findings: V1 + V2 + V3

### What the LLM Actually Does

| Capability | Evidence | Use It? |
|-----------|----------|---------|
| Stock selection | No value vs random gappers (V2-S3, V3-E4) | NO |
| Direction prediction | ORB disagree trades lose -1.53% (V2-S8) | NO |
| Catalyst classification | Earnings filter saves +0.44% T+2 (V3-E3) | YES |
| Volatility forecasting | |expected_move| → |realized| r=+0.28 (V2-S5) | MAYBE (gap_pct is better) |

### Emerging Live Trading Recipe

```
UNIVERSE:   All gappers > 3% (Polygon grouped daily)
            Exclude earnings
            Exclude bad catalysts (deal, macro) if LLM data available
FILTER:     ORB agrees with gap-fade direction (2min or 5min)
DIRECTION:  SHORT (fade the gap)
ENTRY:      Market open (09:30) — don't wait
HOLD:       T+2 for SHORT swing
SIZING:     Weight by |gap_pct| / median(|gap_pct|)
            Optionally: upweight n_runs >= 4 (1.5-2.0x) for intraday only
STOP-LOSS:  None — every design tested destroys value
```

**Expected performance (from data):**
- SHORT + ORB agrees: +6.45% avg T+2, Sharpe 0.881 (n=28)
- Non-earnings + ORB confirms: +1.548% avg 60m, 64% WR (n=111)
- Gap-weighted SHORT: Sharpe 1.78 at 60m

### Critical Caveats

1. **Only 4 days of data.** These findings need validation on more dates.
2. **SHORT T+2 n=28-50.** High returns but small sample.
3. **No transaction costs modeled.** Gap-fade at open may have wide spreads.
4. **No market-cap filter applied.** Small-cap gappers may be untradeable.
5. **Regime analysis inconclusive** at n=4 dates.

---

## Artifacts

All in `sentiment/scanner_optimization/`:

**Scripts:**
- `v3_data_prep.py` — Minute bar cache (316 trades + SPY)
- `v3_exp_e1_e2_stops_orb.py` — E1 (ATR Trailing Stops) + E2 (ORB Refinement)
- `v3_exp_e3_e6_e10_catalysts_sizing.py` — E3 (Catalysts) + E6 (n_runs) + E10 (Magnitude)
- `v3_exp_e4_e7_pricescreen_regime.py` — E4 (Price Screen) + E7 (Regime)
- `v3_exp_e5_e8_e9_short_combo.py` — E5 (SHORT Timing) + E8 (PM+ORB) + E9 (Sympathy)

**Charts:**
- `v3_e1_trailing_stops.png` — ATR trailing stop heatmaps and comparisons
- `v3_e2_orb_refinement.png` — ORB timeframe/threshold analysis
- `v3_e3_catalysts.png` — Catalyst type breakdown and earnings exclusion
- `v3_e4_pricescreen.png` — Pure price screen vs LLM comparison
- `v3_e5_short_timing.png` — SHORT entry timing strategies
- `v3_e6_nruns_sizing.png` — n_runs position sizing Sharpe comparison
- `v3_e7_regime.png` — Market regime analysis
- `v3_e8_pm_orb_combo.png` — PM retrace + ORB combination heatmap
- `v3_e9_sympathy.png` — Sympathy ticker cluster analysis
- `v3_e10_magnitude_sizing.png` — Magnitude sizing scatter and Sharpe

**Data:**
- `results/v3_minute_bars.json` — Cached minute bars (9.3 MB)
- `results/v3_e1_e2_results.json` — E1+E2 metrics
- `results/v3_e3_e6_e10_results.json` — E3+E6+E10 metrics
- `results/v3_e4_e7_results.json` — E4+E7 metrics
- `results/v3_e5_e8_e9_results.json` — E5+E8+E9 metrics
