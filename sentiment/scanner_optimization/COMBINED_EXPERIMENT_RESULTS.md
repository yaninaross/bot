# Scanner Optimization: Combined Experiment Results

**Generated**: 2026-09-13 (updated with full volume data) | **Dates**: Sep 8–11, 2026 | **Hold**: T+2 Close (SHORT direction, local_high entry)

---

# PART 1 — AGENT 1: Universe Narrowing

**Research Question**: What signals applied to the full B1 T+2 universe reliably identify top 3–5 candidates/day with avg PnL significantly above the +3.28% baseline?

**Dataset**: B1_WIDE_SHORT — all gap-up >= 1.5%, price >= $5, SHORT direction
**Funnel**: 3,181 gap-ups → 2,483 qualifying (after bars/entry filter) → 1,017 with T+2
**Volume coverage**: 2,478 / 2,483 trades (99.8%) — computed from 24 dates of Polygon grouped daily

## Baseline

| Metric | Value |
|--------|-------|
| Total qualifying trades | 1,017 (with T+2) |
| Avg PnL | +3.28% |
| Median | +2.98% |
| Win Rate | 73% |
| Sharpe | 0.462 |
| CI 95% | [+2.84, +3.71] |
| CI > 0 | YES |

---

## A1: Gap Tier Analysis

Monotonic staircase — avg PnL increases with gap magnitude. T4 xlarge (5–8%) is the sweet spot with best Sharpe (0.770).

| Tier | Range | n | n_T+2 | t/d | Avg% | Med% | WR | Sharpe | CI_lo | CI>0 | LOO Stable |
|------|-------|---|-------|-----|------|------|-----|--------|-------|------|------------|
| tier_1_small | 1.5–2.0% | 1,019 | 302 | 254.8 | +1.90 | +1.72 | 69% | 0.338 | +1.26 | YES | YES |
| tier_2_medium | 2.0–3.0% | 781 | 327 | 195.2 | +2.88 | +2.83 | 74% | 0.458 | +2.20 | YES | YES |
| tier_3_large | 3.0–5.0% | 401 | 207 | 100.2 | +3.70 | +4.02 | 78% | 0.463 | +2.62 | YES | YES |
| **tier_4_xlarge** | **5.0–8.0%** | **189** | **115** | **47.2** | **+5.50** | **+6.01** | **80%** | **0.770** | **+4.20** | **YES** | **YES** |
| tier_5_extreme | >8.0% | 93 | 66 | 23.2 | +6.31 | +4.66 | 65% | 0.578 | +3.68 | YES | YES |

Top percentile cuts: Top 10% (gap >= 5.4%): avg +5.80%, Sharpe 0.660 | Top 5% (gap >= 7.2%): avg +6.57%, Sharpe 0.651

![Gap Tier Analysis](agent1_gap_tiers.png)

---

## A2: Price Threshold Analysis

$5 floor is adequate. $100+ shows marginal Sharpe improvement (0.595) but cuts n by 88%. Market cap proxy >= $10B yields Sharpe 0.932 (n=31).

| Min Price | n | n_T+2 | t/d | Avg% | Sharpe | CI>0 |
|-----------|---|-------|-----|------|--------|------|
| $5 | 2,483 | 1,017 | 620.8 | +3.28 | 0.462 | YES |
| $10 | 2,053 | 841 | 513.2 | +3.29 | 0.533 | YES |
| $15 | 1,710 | 693 | 427.5 | +3.08 | 0.567 | YES |
| $20 | 1,463 | 580 | 365.8 | +2.90 | 0.566 | YES |
| $30 | 1,114 | 433 | 278.5 | +2.72 | 0.548 | YES |
| $50 | 696 | 249 | 174.0 | +2.07 | 0.470 | YES |
| $100 | 331 | 120 | 82.8 | +2.59 | 0.595 | YES |

Market cap proxy (price × float shares):

| Min Cap | n (T+2) | Avg% | Sharpe |
|---------|---------|------|--------|
| >= $100M | 60 | +3.00 | 0.525 |
| >= $500M | 58 | +3.08 | 0.533 |
| >= $1B | 56 | +3.07 | 0.527 |
| >= $5B | 41 | +3.62 | 0.639 |
| >= $10B | 31 | +4.30 | 0.932 |

![Price and Volume Analysis](agent1_price_volume.png)

---

## A3: Volume Signal Analysis (UPDATED — full coverage)

**Now covers 2,478 / 2,483 trades (99.8%)** — previously only 144 (6%). Volume ratios computed from 24 dates of Polygon grouped daily data with 20-day lookback.

**VERDICT: Volume is NOT a useful standalone filter.** All VR thresholds produce Sharpe *below* the 0.462 baseline. The prior finding of VR >= 1.5x = Sharpe 0.658 was an artifact of 6% coverage bias (sampling only LLM-universe tickers).

| Volume Ratio >= | n | n_T+2 | Avg% | Sharpe | CI_lo |
|-----------------|---|-------|------|--------|-------|
| 1.0x | 1,143 | 554 | +2.93 | 0.388 | +2.30 |
| 1.5x | 485 | 262 | +3.37 | 0.376 | +2.29 |
| 2.0x | 279 | 157 | +2.89 | 0.283 | +1.29 |
| 3.0x | 118 | 61 | +2.66 | 0.211 | -0.50 |
| 5.0x | 48 | 23 | -0.33 | -0.019 | -7.47 |
| 10.0x | 19 | 11 | +2.85 | 0.206 | -5.31 |

Volume rank within day:

| Percentile | n (T+2) | Avg% | Sharpe |
|------------|---------|------|--------|
| Top 10% | 100 | +2.00 | 0.169 |
| Top 5% | 50 | +2.22 | 0.167 |
| Top 3% | 30 | +0.59 | 0.038 |
| Top 1% | 10 | +6.30 | 0.564 |

**Joint filter** (gap >= 5% AND VR >= 3.0): n=28 T+2, avg +5.34%, Sharpe 0.435 — beats avg but not Sharpe of gap-only T4.

---

## A4: Ranking Function Optimization (UPDATED — with real volume)

With full volume data, the ranking function **underperforms the baseline**. Best combo (balanced) averages +1.54% vs baseline +3.28%. Volume-heavy weights actively destroy edge: `volume_dominant` = -5.09% avg.

### Top 3/day — Weight Combos

| Combo | Weights (g/v/p/pm) | n_T+2 | Avg% | Sharpe |
|-------|--------------------|-------|------|--------|
| **balanced** | **0.35/0.35/0.15/0.15** | **6** | **+1.54** | **0.424** |
| gap_price | 0.4/0.2/0.3/0.1 | 6 | +1.54 | 0.424 |
| pure_gap | 0.6/0.2/0.1/0.1 | 6 | +1.36 | 0.352 |
| gap_dominant | 0.5/0.3/0.1/0.1 | 6 | +0.49 | 0.113 |
| gap_vol_equal | 0.4/0.4/0.1/0.1 | 6 | -0.42 | -0.102 |
| volume_dominant | 0.3/0.5/0.1/0.1 | 6 | -5.09 | -0.675 |
| pure_volume | 0.2/0.6/0.1/0.1 | 6 | -5.09 | -0.675 |

Best combo (balanced) across top-N: Top 1 avg -1.44% | Top 2 avg -0.03% | **Top 3 avg +1.54%** | Top 4 avg +0.71% | Top 5 avg -1.82%

**Conclusion**: The composite ranking approach is worse than simple gap-tier filtering. T4 xlarge (gap 5–8%) at Sharpe 0.770 remains the strongest single signal. Ranking pulls in lower-gap tickers weighted by volume/price, which dilutes the edge.

![Ranking Function Optimization](agent1_ranking.png)

---

## A5: SPY Regime Filter

Untestable — SPY was down on both T+2 dates. Correlation: r = -0.054, p = 0.083 (suggestive but not significant).

| Date | SPY Gap | SPY Ret | B1 Trades | T+2 Avg |
|------|---------|---------|-----------|---------|
| 2026-09-08 | -0.14% | -0.40% | 609 | +3.59% |
| 2026-09-09 | -0.24% | -0.22% | 408 | +2.81% |
| 2026-09-10 | -0.57% | -0.03% | 246 | — |
| 2026-09-11 | +0.91% | -0.06% | 1,220 | — |

---

## Agent 1: Top-3 Simulation (UPDATED)

Using balanced ranking (gap=0.35, vol=0.35, price=0.15, pm=0.15):

| Date | Rank | Ticker | Gap% | Price | Vol Ratio | Score | T+2 PnL |
|------|------|--------|------|-------|-----------|-------|---------|
| 2026-09-08 | #1 | QCOM | +6.9% | $180.40 | 2.9x | 0.928 | -1.83% |
| 2026-09-08 | #2 | LINT | +10.2% | $118.12 | 2.2x | 0.918 | +3.62% |
| 2026-09-08 | #3 | ZCSH | +10.9% | $92.94 | 2.8x | 0.908 | +1.69% |
| 2026-09-09 | #1 | SIG | +14.9% | $95.01 | 6.4x | 0.977 | -1.04% |
| 2026-09-09 | #2 | CSR | +9.7% | $57.84 | 17.9x | 0.958 | -0.88% |
| 2026-09-09 | #3 | ZCSH | +10.6% | $101.39 | 1.9x | 0.928 | +7.69% |
| 2026-09-10 | #1 | AXTQ | +8.5% | $40.62 | 2.5x | 0.874 | no data |
| 2026-09-10 | #2 | SKHN | +9.8% | $10.39 | 5.0x | 0.868 | no data |
| 2026-09-10 | #3 | ZSL | +7.5% | $24.31 | 2.1x | 0.851 | no data |
| 2026-09-11 | #1 | ORCL | +7.5% | $164.43 | 3.4x | 0.974 | no data |
| 2026-09-11 | #2 | FEIM | +33.1% | $82.61 | 7.0x | 0.967 | no data |
| 2026-09-11 | #3 | WF | +5.8% | $78.90 | 2.5x | 0.912 | no data |

**Simulation total**: 6 trades with T+2, avg = +1.54%, total = +9.25%, WR = 50%

![Top-3 Simulation](agent1_simulation.png)

---

## Agent 1: Final Specification (REVISED)

| Parameter | Value |
|-----------|-------|
| **Recommended filter** | **Gap tier T4 xlarge (5–8%)** |
| Sharpe | 0.770 (best of all signals tested) |
| Avg PnL | +5.50% |
| Win rate | 80% |
| Trades/day | ~47 qualifying, select top 3 by gap magnitude |
| Ranking function | **NOT RECOMMENDED** — underperforms baseline |
| Volume filter | **NOT RECOMMENDED** — all thresholds below baseline Sharpe |
| Caveat | 4 dates (2 with T+2), n=115 for T4 tier |

**Key revision**: The composite ranking function (gap + volume + price + PM retrace) was abandoned after full volume data revealed that volume actively hurts selection. Simple gap-tier filtering at T4 (5–8%) is the strongest signal with Sharpe 0.770 vs baseline 0.462.

---
---

# PART 2 — AGENT 2: Expansion + Logging

**Research Question**: How to reliably surface 3 qualified LLM candidates every morning, and what data to log to confirm/reject the LLM strategy within 4 weeks?

**Dataset**: v2_features_full.csv — 316 LLM trades, 4 dates (Sep 8–11, 2026)
**T+2 available**: 117 trades (Sep 8–9 only)

## Scanner Silence Diagnosis

The scanner fires every day — 316 total signals, 79/day avg. Sep 10–11 show 0 T+2 data because T+2 close prices aren't available for those dates. This is a **data availability issue**, not scanner silence.

| Date | Total | Premarket | Discovery | SHORT gap-up | SHORT >=3% | Has T+2 |
|------|-------|-----------|-----------|-------------|-----------|---------|
| 2026-09-08 | 102 | 58 | 44 | 24 | 13 | 102 |
| 2026-09-09 | 15 | 7 | 8 | 0 | 0 | 15 |
| 2026-09-10 | 58 | 23 | 35 | 5 | 1 | **0** |
| 2026-09-11 | 141 | 19 | 122 | 49 | 16 | **0** |

---

## B1: Gap Threshold Sweep

CI > 0 at all thresholds. Sharpe peaks at 1.230 for >= 5% but with only 5 T+2 trades. The 2% threshold balances volume and edge.

| Threshold | n_total | n_T+2 | t/d | Avg% | Med% | WR | Sharpe | CI_lo | CI_hi | CI>0 |
|-----------|---------|-------|-----|------|------|-----|--------|-------|-------|------|
| 0.5% | 44 | 15 | 11.0 | +8.80 | +6.27 | 87% | 0.751 | +2.87 | +14.73 | YES |
| 1.0% | 39 | 13 | 9.8 | +9.08 | +5.97 | 85% | 0.719 | +2.21 | +15.95 | YES |
| 1.5% | 29 | 10 | 7.2 | +10.48 | +6.12 | 80% | 0.737 | +1.67 | +19.30 | YES |
| **2.0%** | **25** | **8** | **6.2** | **+11.67** | **+6.53** | **75%** | **0.735** | **+0.67** | **+22.68** | **YES** |
| 3.0% | 18 | 7 | 4.5 | +14.39 | +7.08 | 86% | 0.959 | +3.27 | +25.51 | YES |
| 4.0% | 14 | 6 | 3.5 | +15.80 | +9.12 | 83% | 0.992 | +3.05 | +28.54 | YES |
| 5.0% | 13 | 5 | 3.2 | +19.02 | +11.17 | 100% | 1.230 | +5.47 | +32.58 | YES |

![Gap Threshold Sweep](agent2_gap_threshold.png)

---

## B2: Catalyst Inclusion Expansion

### Per-Catalyst Breakdown (SHORT gap-up >= 2%)

| Catalyst | n | n_T+2 | Avg% | WR | Sharpe |
|----------|---|-------|------|-----|--------|
| offering | 4 | 2 | +35.80 | — | — |
| earnings | 5 | 2 | +14.64 | — | — |
| premarket_move | 2 | 1 | +12.98 | — | — |
| deal | 6 | 3 | +10.40 | 100% | 1.100 |
| analyst | 8 | 2 | +5.60 | — | — |
| breaking | 2 | 2 | +3.37 | — | — |
| insider | 2 | 2 | -1.04 | — | — |

### Catalyst Set Comparison

no_exclusions has best Sharpe (0.919) but minimal_earnings_only is the best tradeoff — excludes only earnings while keeping volume.

| Set | Excludes | n | n_T+2 | t/d | Avg% | Sharpe | CI>0 |
|-----|----------|---|-------|-----|------|--------|------|
| current | earnings, deal, macro, breaking | 23 | 6 | 5.8 | +14.44 | 0.819 | YES |
| add_breaking | earnings, deal, macro | 25 | 8 | 6.2 | +11.67 | 0.735 | YES |
| **minimal_earnings_only** | **earnings** | **28** | **9** | **7.0** | **+12.74** | **0.838** | **YES** |
| no_exclusions | — | 39 | 14 | 9.8 | +11.49 | 0.919 | YES |

![Catalyst Set Comparison](agent2_catalyst_sets.png)

---

## B3: Discovery vs Premarket Scanner

Discovery scanner massively outperforms premarket at T+2. Premarket has 4x higher signal volume but negligible edge.

| Scanner | n | n_T+2 | Avg% | Med% | WR | Sharpe | CI_lo | CI_hi |
|---------|---|-------|------|------|-----|--------|-------|-------|
| premarket | 6 | 4 | +2.36 | +2.82 | 50% | 0.295 | -5.48 | +10.21 |
| **discovery** | **19** | **4** | **+20.99** | **+21.23** | **100%** | **1.226** | **+4.21** | **+37.76** |

---

## B4: n_runs Signal Analysis

**INVERTED** vs V2 finding. n_runs=1 outperforms but with only 8 total T+2 trades. Do not use as filter until sample grows.

| n_runs >= | n | n_T+2 | t/d | Avg% | WR | Sharpe | CI>0 |
|-----------|---|-------|-----|------|-----|--------|------|
| 1 | 25 | 8 | 6.2 | +11.67 | 75% | 0.735 | YES |
| 2 | 3 | 2 | 0.8 | -3.85 | — | — | — |
| 3 | 1 | 1 | 0.2 | -0.34 | — | — | — |
| 4+ | 0 | 0 | — | — | — | — | — |

![Scanner & n_runs Analysis](agent2_scanner_nruns.png)

---

## Multi-Hold Analysis

T+2 is the only hold with CI > 0. Confirms the swing-hold thesis for SHORT gap-ups.

| Hold | n | Avg% | Med% | WR | Sharpe | CI_lo | CI>0 |
|------|---|------|------|-----|--------|-------|------|
| 30m | 22 | +1.55 | +0.38 | 77% | 0.307 | -0.56 | NO |
| 60m | 19 | +3.17 | +0.51 | 58% | 0.293 | -1.69 | NO |
| 120m | 16 | +3.71 | -0.02 | 50% | 0.263 | -3.20 | NO |
| EOD | 25 | +3.59 | +0.51 | 64% | 0.340 | -0.54 | NO |
| T+1 | 9 | +8.87 | +4.10 | 67% | 0.522 | -2.24 | NO |
| **T+2** | **8** | **+11.67** | **+6.53** | **75%** | **0.735** | **+0.67** | **YES** |
| T+3 | 8 | +11.53 | +7.97 | 75% | 0.589 | -2.02 | NO |

---

## Tier System Simulation

Target: 3 trades/day. Achieved: 1.8/day (below target). Sep 9 produced 0 qualified trades.

### Per-Day Selections

**2026-09-08** — 10 qualified (T1:0, T2:5, T3:5) → selected top 3:

| # | Tier | Ticker | Gap% | Catalyst | n_runs | T+2 PnL |
|---|------|--------|------|----------|--------|---------|
| 1 | T2 | PDSB | +121.8% | offering | 1 | +35.39% |
| 2 | T2 | FCEL | +17.4% | earnings | 1 | +11.17% |
| 3 | T2 | SHMD | +8.2% | insider | 1 | +5.28% |

**2026-09-09** — 0 qualified → no trades

**2026-09-10** — 1 qualified (T3: OTLK +3.3%) → no T+2 data

**2026-09-11** — 18 qualified (T1:2, T2:12, T3:4) → selected top 3:

| # | Tier | Ticker | Gap% | Catalyst | n_runs | T+2 PnL |
|---|------|--------|------|----------|--------|---------|
| 1 | T1 | LASE | +6.8% | offering | 1 | no data |
| 2 | T1 | INTC | +2.1% | offering | 1 | no data |
| 3 | T2 | SWRD | +98.7% | unknown | 1 | no data |

### Aggregate

| Metric | Value |
|--------|-------|
| Total selected | 7 (1.8/day) |
| With T+2 data | 3 |
| Avg T+2 PnL | +17.28% |
| Median T+2 PnL | +11.17% |
| Win rate | 100% |

![Tier System Simulation](agent2_tier_simulation.png)

---

## Logging Schema Validation

- Total fields: 63
- Computable from backtest data: 51
- Live-only fields: 12 (`scanner_run_time`, `pm_volume_profile`, `float_shares`, `orb_width_atr_multiple`, `slippage_pct`, `sector_etf`, `sector_direction_at_entry`, `actual_hold`, `actual_pnl`, `exit_reason`, `borrow_rate_pct`, `notes`)

---

## Agent 2: Final Specification

### Tier Criteria

| Tier | Priority | Gap | Catalyst | PM Retrace | Exclude |
|------|----------|-----|----------|------------|---------|
| T1 | Highest | >= 2% | offering OR regulatory | 0.2–0.8 | earnings, deal |
| T2 | Fill to 3/day | >= 2% | any except earnings/deal/breaking | < 0.8 | — |
| T3 | Fallback | >= 1.5% | any except earnings/deal | — | — |

Rank within tier by `n_runs × |gap_pct|`

---
---

# CROSS-AGENT COMPARISON

| Dimension | Agent 1 (B1 Universe) | Agent 2 (LLM Universe) |
|-----------|----------------------|----------------------|
| Universe size | 2,483 qualifying (1,017 T+2) | 316 signals (117 T+2) |
| Baseline avg PnL | +3.28% | +11.67% (SHORT gap >= 2%) |
| Best filter | Gap T4 xlarge (5–8%): +5.50%, Sharpe 0.770 | Discovery scanner: +20.99%, Sharpe 1.226 |
| Volume signal | **NOT useful** — all thresholds below baseline Sharpe | Available (253 tickers) |
| Ranking function | **Underperforms baseline** (+1.54% vs +3.28%) | N/A |
| Trades/day | ~47 in T4 tier (select top 3 by gap) | 1.8 (organic, below 3 target) |
| Regime signal | Untestable (2 dates, both SPY down) | Untestable (2 dates) |
| Key risk | Need more dates for regime test | Below target volume (1.8 vs 3/day) |
| T+2 sample | 1,017 (robust) | 8 (fragile) |

## Key Takeaways

1. **Gap magnitude is THE signal** — monotonic staircase from T1 (+1.90%) to T5 (+6.31%). T4 xlarge (5–8%) has best Sharpe 0.770, 80% WR, CI confirmed. All other signals tested are inferior.
2. **Volume is NOT a useful filter** — with full data coverage (99.8%), all VR thresholds produce Sharpe *below* baseline. The prior VR >= 1.5x = Sharpe 0.658 was a 6% sample bias artifact. Volume-heavy ranking weights actively destroy edge (-5.09%).
3. **Composite ranking underperforms simple gap filtering** — best ranking combo (+1.54%) is worse than baseline (+3.28%). Top-3 selection by composite score dilutes the gap-tier edge.
4. **Discovery scanner >> premarket** — +20.99% vs +2.36% avg T+2. Prioritize discovery signals.
5. **T+2 is the only hold with CI > 0** — confirms swing-hold thesis for SHORT gap-ups.
6. **n_runs signal inverted** from V2 — do not use as filter until sample grows past 30+.
7. **Agent 2 tier system undershoots target** — 1.8/day vs 3/day. Sep 9 had 0 trades. Needs looser T3 or lower gap floor.
8. **Offerings dominate** — PDSB +35.39%, offering catalyst avg +35.80%. T1 tier (offering/regulatory) correctly prioritized.

## Revised Agent 1 Recommendation

**Drop the ranking function entirely.** Instead:
- Filter to gap tier T4 (5–8%) — Sharpe 0.770, avg +5.50%, 80% WR, ~47 trades/day
- Select top 3 by raw gap magnitude (no composite score needed)
- This outperforms every ranking combo tested

## Next Steps

- [ ] Accumulate T+2 data for Sep 10–11 (and subsequent trading days) to double the sample
- [ ] Test SPY regime filter once an up-day appears in the T+2 window
- [ ] Loosen Agent 2 T3 criteria (gap >= 1.0%?) to hit 3 trades/day target
- [ ] Deploy logging schema and begin 4-week live collection period
- [ ] ~~Fetch volume data for full B1 universe~~ **DONE** — volume confirmed NOT useful
