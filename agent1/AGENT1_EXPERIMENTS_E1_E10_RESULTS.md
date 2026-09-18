# Agent 1 — Experiments E1-E10 Results (v2)

**Date:** 2026-09-14 (v2 re-run with fixes)  
**Backtest:** 2026-01-02 → 2026-09-12 (174 trading days)  
**Universe:** Hardcoded `_FLOAT` (217 tickers)  
**Strategy:** SHORT top 5 gap-ups ≥ 5%, local high entry, T+2 baseline hold  
**Wide pool:** 2277 trades with entry (PRE_FILTER_N=15, 2610 candidates scanned)  
**Top-5 deployed:** 798 trades (used for E1-E9)  
**Runtime:** 1602s (26.7 min) — LLM sector labeling was 67% of runtime (1081s)

---

## Fixes Applied (v2 vs v1)

1. **Wider pre-filter (PRE_FILTER_N=15):** Scan now selects top 15 per day (not top 5). E1-E9 analyze the top-5 "deployed" subset; E10 re-ranks from the full 15.
2. **LLM sector labeling:** Gemini 2.5 Flash labels tickers without SIC codes. Coverage: SIC=399 + LLM=341 = 740 labeled (93%), only 58 unknown (7%). Previously ~50% unknown.
3. **E10 re-ranking works:** Pool of 2259 trades, avg 13.1/day, 171/173 days with >5 candidates. Real differentiation between weight combos.

## T4 Tier: Confirmed Empty (Expected)

All 798 top-5 trades are T5 (>8%). When selecting the top 5 gap-ups from ~3,605 tickers, the biggest 5 always exceed 8%. This is structural — not a bug. T4 (5-8%) gaps exist but never rank in the top 5.

---

## Baseline

| Metric | Value |
|--------|-------|
| Trades | 793 (with T+2 outcome) |
| Avg PnL | +8.84% |
| Median | +7.61% |
| Trimmed Mean | +10.85% |
| Win Rate | 66% |
| Sharpe | 0.235 |
| Profit Factor | 2.23 |
| 95% CI | [+6.22%, +11.46%] ✓ |

---

## E1: Hold Period Sweep

**Result:** Longer holds monotonically improve Sharpe. T+5 is optimal.

| Hold | n | Avg | Median | TrimMean | WR | Sharpe | PF | CI 95% | LOO |
|------|---|-----|--------|----------|----|--------|----|--------|-----|
| 30m | 774 | +0.91% | +0.72% | +1.26% | 59% | 0.097 | 1.38 | [+0.25,+1.57] ✓ | YES |
| 60m | 759 | +1.32% | +1.10% | +1.73% | 58% | 0.094 | 1.39 | [+0.32,+2.32] ✓ | YES |
| 2h | 730 | +2.63% | +1.87% | +2.91% | 62% | 0.169 | 1.74 | [+1.51,+3.76] ✓ | YES |
| EOD | 798 | +2.42% | +2.36% | +4.09% | 60% | 0.073 | 1.40 | [+0.12,+4.73] ✓ | YES |
| T+1 | 797 | +3.68% | +4.88% | +7.62% | 63% | 0.059 | 1.39 | [-0.68,+8.04] ✗ | YES |
| **T+2** | **793** | **+8.84%** | **+7.61%** | **+10.85%** | **66%** | **0.235** | **2.23** | **[+6.22,+11.46] ✓** | **YES** |
| T+3 | 791 | +9.64% | +9.49% | +12.50% | 69% | 0.216 | 2.19 | [+6.53,+12.74] ✓ | YES |
| T+4 | 786 | +11.19% | +11.87% | +14.21% | 69% | 0.253 | 2.35 | [+8.09,+14.28] ✓ | YES |
| **T+5** | **783** | **+13.15%** | **+13.17%** | **+16.18%** | **69%** | **0.298** | **2.62** | **[+10.06,+16.24] ✓** | **YES** |

**Key:** Sharpe 0.235 (T+2) → 0.298 (T+5). Edge is real at all horizons except T+1. EOD is a local minimum (Sharpe 0.073) — profits dip intraday then build over multiple days.

**Caution:** T+5 increases capital lockup. Sharpe improvement comes from higher avg PnL, not tighter distribution.

---

## E2: ETF Exclusion Filter

**Result:** Excluding ETFs improves Sharpe by +0.029. ETFs avg +1.82% vs equities +10.85%.

| Filter | n | Avg | Median | WR | Sharpe | PF | CI 95% |
|--------|---|-----|--------|----|--------|----|--------|
| no_filter | 793 | +8.84% | +7.61% | 66% | 0.235 | 2.23 | [+6.22,+11.46] ✓ |
| **exclude_etf** | **615** | **+10.87%** | **+8.86%** | **69%** | **0.264** | **2.48** | **[+7.62,+14.13] ✓** |
| equities_only | 554 | +10.85% | +10.38% | 70% | 0.254 | 2.38 | [+7.29,+14.41] ✓ |

### ETF vs Equity Head-to-Head

| Group | n | Avg T+2 | WR | Sharpe |
|-------|---|---------|------|--------|
| ETFs | 178 | +1.82% | 58% | 0.093 |
| **Equities** | **554** | **+10.85%** | **70%** | **0.254** |

**Key:** ETFs are 22% of trades but drag heavily. Worst: BEX (-18.11%), UXRP (-18.07%), XRPT (-16.78%). `exclude_etf` is best — `equities_only` drops 61 more trades for slightly lower Sharpe.

---

## E3: Liquidity Filter (ADV)

**Result:** Counterintuitive — lowest ADV quartile has BEST performance.

| Threshold | n | Avg | WR | Sharpe | PF | TPD |
|-----------|---|-----|----|--------|----|-----|
| no_filter | 793 | +8.84% | 66% | 0.235 | 2.23 | 4.6 |
| adv_$500K | 658 | +7.46% | 64% | 0.195 | 1.99 | 3.8 |
| adv_$1M | 586 | +7.16% | 65% | 0.187 | 1.95 | 3.4 |
| adv_$5M | 424 | +6.29% | 62% | 0.195 | 1.92 | 2.5 |
| adv_$10M | 337 | +5.77% | 61% | 0.191 | 1.87 | 2.1 |
| adv_$25M | 227 | +5.03% | 62% | 0.157 | 1.71 | 1.7 |
| adv_$50M | 140 | +8.14% | 65% | 0.374 | 2.91 | 1.4 |
| adv_$100M | 76 | +9.95% | 68% | 0.428 | 3.35 | 1.3 |

### ADV Quartile Performance

| Quartile | n | Avg | WR | Sharpe |
|----------|---|-----|----|--------|
| **Q1 (lowest)** | **198** | **+13.70%** | **72%** | **0.387** |
| Q2 | 198 | +10.15% | 70% | 0.211 |
| Q3 | 198 | +5.07% | 60% | 0.141 |
| Q4 (highest) | 199 | +6.46% | 63% | 0.230 |

**Correlation log10(ADV) vs T+2 PnL:** r=-0.0873

**Key:** Bimodal — Q1 (smallest) is best (Sharpe 0.387), Q4 ($100M+) also strong (0.428 but n=76). Middle thresholds ($1M-$25M) degrade. **Do not apply minimum ADV filter.**

---

## E4: Asset Type Stratification

| Type | n | Avg | WR | Sharpe | PF | Tickers |
|------|---|-----|----|--------|----|---------|
| **CS** | **534** | **+11.47%** | **70%** | **0.297** | **2.59** | **361** |
| ETF_LEV | 177 | +2.31% | 58% | 0.115 | 1.34 | 120 |
| UNK | 45 | +11.60% | 53% | 0.514 | 7.99 | 39 |
| ADRC | 20 | -5.66% | 70% | -0.054 | 0.77 | 16 |
| ETF | 9 | +1.49% | 67% | 0.142 | 1.53 | 6 |
| ETS | 3 | -19.11% | 0% | -12.618 | 0 | 3 |

**Key:** CS is the core (Sharpe 0.297, n=534). Leveraged ETFs drag most. ADRC negative avg despite 70% WR — extreme outlier losses.

---

## E5: Price Band Analysis

**Result:** Higher-priced stocks ($200+) show dramatically better fade performance.

### Price Bands

| Band | n | Avg | WR | Sharpe | PF |
|------|---|-----|----|--------|----|
| $10-20 | 309 | +8.03% | 67% | 0.223 | 2.03 |
| $20-50 | 281 | +6.13% | 65% | 0.153 | 1.80 |
| $50-100 | 94 | +12.27% | 62% | 0.368 | 3.29 |
| $100-200 | 55 | +4.06% | 60% | 0.093 | 1.41 |
| **$200-500** | **30** | **+31.39%** | **83%** | **1.004** | **32.64** |
| **$500+** | **24** | **+20.28%** | **83%** | **0.761** | **8.06** |

### Minimum Price Floors

| Floor | n | Avg | WR | Sharpe |
|-------|---|-----|----|--------|
| min $10 | 793 | +8.84% | 66% | 0.235 |
| min $15 | 604 | +9.53% | 67% | 0.260 |
| **min $50** | **203** | **+13.82%** | **67%** | **0.381** |

**Key:** $200-500 is extraordinary (Sharpe 1.004, PF 32.64) but n=30. $50+ floor is practical: Sharpe 0.381, n=203. The $100-200 dip (Sharpe 0.093) is a "dead zone."

---

## E6: Volume Spike / Squeeze Proxy

**Note:** Short interest endpoint unavailable (requires Polygon Business plan). Using gap-day volume/ADV ratio.

| Vol Ratio | n | Avg | WR | Sharpe | PF |
|-----------|---|-----|----|--------|----|
| **vol <1.5x** | **63** | **+13.87%** | **76%** | **0.656** | **6.24** |
| vol 1.5-3x | 86 | +6.51% | 70% | 0.192 | 2.00 |
| vol 3-5x | 98 | +8.25% | 64% | 0.411 | 3.16 |
| vol 5-10x | 167 | +6.15% | 62% | 0.305 | 2.38 |
| vol >10x | 379 | +9.87% | 66% | 0.204 | 1.97 |

**Correlation log10(vol_ratio) vs T+2 PnL:** r=+0.1026

**Key:** Low volume spikes (<1.5x ADV) = best fade (Sharpe 0.656). "Quiet" gap-ups without retail frenzy. Actionable but n=63 is modest.

---

## E7: Intraday Pattern at Open — STRONGEST SIGNAL

**Result:** sustained_drop is the dominant predictor. drop_then_recover is noise.

| Pattern | n | Avg T+2 | Median | WR | Sharpe | PF | EOD Avg |
|---------|---|---------|--------|----|--------|----|---------|
| **sustained_drop** | **319** | **+16.90%** | **+16.47%** | **80%** | **0.465** | **4.35** | **+10.73%** |
| flat_or_bounce | 92 | +5.84% | +0.00% | 49% | 0.290 | 3.03 | +2.42% |
| mild_drop | 113 | +4.19% | +1.78% | 55% | 0.179 | 1.68 | -7.10% |
| drop_then_recover | 230 | +1.82% | +6.65% | 60% | 0.037 | 1.14 | -4.38% |

### Alternative Entry (drop_then_recover only)

| Entry | n | Avg | Sharpe |
|-------|---|-----|--------|
| local_high (standard) | 230 | +1.82% | 0.037 |
| 60m bounce entry | 230 | +6.95% | 0.147 |

**Key:** sustained_drop = Sharpe 0.465, 80% WR, avg +16.90%. Already profitable by EOD (+10.73%). drop_then_recover is essentially zero-edge (Sharpe 0.037). **Filter to sustained_drop only** nearly doubles Sharpe (0.235 → 0.465) while keeping 319 trades (40% of pool).

---

## E8: Gap Freshness — SECOND STRONGEST SIGNAL

**Result:** Strong monotonic relationship. Stale gaps fade much harder.

| Freshness | n | Avg | WR | Sharpe | PF | Avg Mins Pre-Open |
|-----------|---|-----|----|--------|----|-------------------|
| very_fresh (≤30m) | 169 | +2.71% | 58% | 0.056 | 1.24 | 12m |
| fresh (30m-2h) | 270 | +6.75% | 67% | 0.187 | 1.91 | 73m |
| moderate (2h-5h) | 203 | +12.55% | 69% | 0.346 | 3.06 | 193m |
| **stale (>5h)** | **132** | **+15.25%** | **70%** | **0.564** | **4.99** | **324m** |
| no_pm | 19 | +9.01% | 84% | 0.656 | 5.44 | N/A |

**Correlation mins_before_open vs T+2 PnL:** r=+0.1219 (positive = staler = better)

**Key:** Stale gaps (PM high >5h before open) = Sharpe 0.564, avg +15.25%. The gap is "exhausted" — early PM buyers are underwater by open, creating selling pressure. Very fresh gaps (≤30m) have momentum still intact, Sharpe only 0.056. **Exclude very_fresh** at minimum.

---

## E9: Sector Concentration (Updated with LLM Sectors)

### Sector Source Breakdown

| Source | Count | % |
|--------|-------|---|
| SIC | 399 | 50% |
| **LLM (Gemini)** | **341** | **43%** |
| None | 58 | 7% |

### Idiosyncratic vs Sector-Wide — REVERSED FROM v1

| Group | n | Avg | Median | WR | Sharpe | LOO |
|-------|---|-----|--------|----|--------|-----|
| Idiosyncratic (≤2) | 667 | +8.98% | +8.08% | 67% | 0.227 | YES |
| **Sector-wide (≥3)** | **126** | **+8.11%** | **+5.71%** | **65%** | **0.323** | **YES** |

**v1 had:** Idiosyncratic 0.334 > Sector-wide 0.137. **v2 reverses:** Sector-wide 0.323 > Idiosyncratic 0.227.

**Why reversed:** v1 had 50% "Unknown" sector, which inflated the "idiosyncratic" count (unknown tickers couldn't cluster). With proper LLM labeling, sector clustering is more accurate, and sector-wide gaps actually show tighter distributions.

### Top Sectors (n ≥ 10)

| Sector | n | Avg | WR | Sharpe | Source |
|--------|---|-----|----|--------|--------|
| Healthcare | 30 | +22.99% | 77% | 0.880 | mostly LLM |
| Transportation | 15 | +19.14% | 87% | 0.966 | SIC |
| Agriculture | 10 | +28.97% | 80% | 0.827 | SIC |
| Construction | 12 | +17.16% | 75% | 0.687 | SIC |
| Mining | 17 | +12.13% | 88% | 0.444 | SIC |
| Services | 100 | +9.03% | 68% | 0.372 | SIC |
| Finance | 77 | +8.93% | 69% | 0.292 | mixed |
| Retail | 32 | +13.13% | 75% | 0.276 | SIC |
| Crypto/Blockchain | 29 | +4.57% | 69% | 0.209 | LLM |
| Manufacturing | 201 | +6.26% | 63% | 0.176 | SIC |
| Technology | 133 | +4.28% | 63% | 0.080 | mixed |

**Key:** Technology (Sharpe 0.080) and Manufacturing (0.176) are the weakest large sectors. Healthcare (0.880) and Transportation (0.966) are strongest but small n. The E9 reversal means **sector-wide gaps are NOT worse** — the prior finding was a data quality artifact.

---

## E10: Combined Signal Score — PURE GAP WINS

**Pool:** 2259 trades across 173 days, avg 13.1/day, 171 days with >5 candidates.  
**Method:** Re-rank from top 15 → top 5 per day using weighted feature scores.

| Combo | n | Avg | Median | WR | Sharpe | Δ vs pure_gap |
|-------|---|-----|--------|----|--------|---------------|
| **pure_gap** | **863** | **+8.41%** | **+7.46%** | **66%** | **0.230** | **baseline** |
| balanced | 863 | +7.83% | +6.41% | 68% | 0.191 | -0.040 |
| quality | 863 | +7.47% | +6.07% | 67% | 0.181 | -0.049 |
| gap_adv | 863 | +6.24% | +5.26% | 64% | 0.170 | -0.060 |
| pattern_heavy | 863 | +7.13% | +6.49% | 68% | 0.153 | -0.077 |
| anti_squeeze | 863 | +6.73% | +6.20% | 67% | 0.139 | -0.091 |

**Key finding: Every weighted alternative DEGRADES performance.** pure_gap (rank by biggest gap %) is optimal. Adding pattern/freshness/quality features to the ranking pulls in lower-gap stocks whose improved signal quality doesn't compensate for the gap-size loss. **Don't add complexity to the daily selection ranking.**

Note: The features (sustained_drop, stale, etc.) are valuable as **post-selection filters** (E7, E8), not as **selection weights** (E10). Filter first, then pick the biggest gaps from what remains.

---

## Actionable Recommendations for Live Deployment

### Tier 1: Deploy Now (High confidence, large n)

| Change | Sharpe Δ | Trades | Implementation |
|--------|----------|--------|----------------|
| Exclude ETFs | +0.029 | 615 | Check `asset_type` in Polygon ref |
| Keep pure gap ranking | +0.000 | — | Don't change selection logic |

### Tier 2: Strong Signal (Test in paper trading)

| Change | Sharpe | n | Implementation |
|--------|--------|---|----------------|
| Filter to sustained_drop only | 0.465 | 319 | Classify first 60min pattern at open |
| Exclude very_fresh PM gaps (≤30m) | ~0.30+ | ~630 | Track PM high formation time |
| Price floor $50+ | 0.381 | 203 | Simple price check |
| Extend hold to T+5 | 0.298 | 783 | Capital efficiency tradeoff |

### Tier 3: Monitor (Small n, needs validation)

| Signal | Sharpe | n | Concern |
|--------|--------|---|---------|
| $200-500 price band | 1.004 | 30 | Possibly name-specific |
| Vol <1.5x ADV | 0.656 | 63 | Needs borrow cost overlay |
| no_pm (no premarket bars) | 0.656 | 19 | Too small to act on |

### Deployed Configuration (validated via E2 grid + E3b + Universe Float)

```
DIRECTION = SHORT
FILTER = CS + sustained_drop (CS+SD)
RANK BY = pure gap size (biggest gap wins)
TOP_N = 1 (single biggest gap each day)
EXIT = 17:00 ET same day (1h after close, after-hours)
FLOAT FILTER = SOFT (hardcoded dict; unknown tickers pass through)
```

**Performance:** 160 trades, +14.71% avg, Sharpe 0.594, 79% WR, PF 6.26, $+2.35M on $100K pool.

### Don't Do (validated by experiments)

- Don't add minimum ADV filter (Q1 lowest is best — E3)
- Don't add multi-feature ranking weights (E10 proves pure gap is optimal)
- Don't prefer idiosyncratic gaps (v2 reversed this — sector-wide is actually slightly better)
- Don't use hard float filter (kills $1.64M in profit — Universe Float Comparison)
- Don't hold overnight T+1 (dead zone, Sharpe 0.027-0.074 — E2 Part 2)
- Don't use PM retrace filter (costs -$378K — PM Retrace Comparison)

---

---

## E2 Grid: N×Hold + Capital Efficiency + Intraday Exit (2026-09-15)

**Script:** `agent1_e2_nhold_grid.py` — CS+SD filter, 174 trading days, 497 CS+SD trades across 160 active days.

### Part 1 — Raw Grid: N=1 T+5 best (Sharpe 0.604, +26.11%, 84% WR, PF 6.09)
### Part 2 — Capital Efficiency: N=1 EOD best pool$ ($1.98M, Sharpe 0.533)
### Part 3 — Intraday Exit: **N=1 exit@17:00 AH = $+2.34M** (best overall)
- Sharpe peaks 11:00-12:00 (0.788-0.790) — most reliable window
- AH spike 16:00→17:00 — biggest single-hour jump
- T+1 dead zone (Sharpe 0.027-0.074) — overnight destroys edge

---

## E3b: Gap Size Impact (2026-09-15)

**Config:** CS+SD, N=1, exit@17:00, 160 trades.

| Bucket | n | Avg PnL | Sharpe | WR | Pool $ | % of Total |
|--------|---|---------|--------|-----|--------|------------|
| 5-8% | 1 | +0.00% | — | — | $0 | 0% |
| 8-12% | 3 | -4.82% | -0.783 | 0% | -$14K | -0.6% |
| 12-20% | 17 | +4.52% | 0.498 | 71% | +$77K | 3.3% |
| 20-30% | 30 | +9.19% | **0.875** | 83% | +$276K | 11.8% |
| 30-50% | 37 | +14.67% | 0.718 | 78% | +$543K | 23.2% |
| 50%+ | 72 | **+20.20%** | 0.631 | 82% | +$1.45M | **62.3%** |

**Verdict:** Keep biggest-gap ranking. 50%+ gaps = 62% of all dollars. Non-monotonic but positive correlation (r=+0.247).

---

## Universe Float Comparison: Soft vs Hard (2026-09-15)

**Script:** `agent1_universe_float.py` — CS+SD, N=1, exit@17:00.

| Metric | SOFT_FLOAT (current) | HARD_FLOAT (Polygon ≥10M) | Delta |
|--------|---------------------|--------------------------|-------|
| Trades | 160 | 126 | -34 |
| Avg PnL | +14.71% | +5.67% | **-9.04%** |
| Sharpe | 0.594 | 0.478 | -0.116 |
| WR | 79% | 70% | -9% |
| Pool $ | $+2.35M | $+715K | **-$1.64M** |
| Max Loss | -132.6% | -40.4% | +92.2% better |

**SOFT-only trades (removed by hard filter):** n=94, avg +20.39%, Sharpe 0.706, WR 86% — the BEST trades.
**HARD-only trades (replacements):** n=60, avg +4.63%, Sharpe 0.514, WR 72% — much weaker.

**Verdict:** KEEP soft float. Small-float stocks ARE the edge. Hard float kills $1.64M profit.

---

## Remaining Limitations

1. **T4 structurally empty:** Top-5 selection always yields >8% gaps. To study T4, would need to explicitly target 5-8% range with separate scan.
2. **No short interest data:** Polygon Business plan required. Vol ratio is a weak proxy.
3. **LLM sector accuracy:** 1521 tickers classified by Gemini — spot-check accuracy is good but not validated systematically.
4. **Filter interactions untested:** Individual filters show improvement, but stacking multiple filters may have non-additive effects or create survivorship bias.
5. **NumPy warnings:** "Mean of empty slice" for `no_pm` group with `nanm` — cosmetic only.
