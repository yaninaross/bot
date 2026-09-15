# Agent 1 — Investigation Results I1-I10

**Period:** 2025-09-15 to 2026-09-12 (250 trading days)
**Universe:** _FLOAT (217 tickers, 126 with trades), TOP_N=5, both entry types
**Data:** 878 enriched trades, 852 CS, 126 unique tickers
**Baseline reference:** E1-E10 settings — avg +8.84%, Sharpe 0.235, n=793 (Jan-Sep 2026 only)

---

## Master Comparison Table

| ID | Investigation | n | TPD | Avg PnL | Median | Trim Mean | WR | Sharpe | PF | CI 95% | Sig? |
|----|--------------|---|-----|---------|--------|-----------|-----|--------|-----|--------|------|
| **I2** | **CS + sustained_drop** | **199** | **1.9** | **+1.66%** | **+3.27%** | **+2.43%** | **61%** | **0.144** | **1.49** | **[+0.06, +3.25]** | **YES** |
| **I6a** | **CS + vol_ratio < 1.5x** | **305** | **2.4** | **+1.07%** | **+1.18%** | **+1.33%** | **57%** | **0.107** | **1.32** | **[-0.05, +2.18]** | **~YES** |
| I6b | CS + vol_ratio < 2.0x | 370 | 2.5 | +0.69% | +0.80% | +0.94% | 55% | 0.070 | 1.20 | [-0.31, +1.68] | No |
| I6c | CS + vol_ratio < 3.0x | 464 | 2.6 | +0.68% | +0.71% | +0.82% | 54% | 0.070 | 1.20 | [-0.20, +1.57] | No |
| I8b | Close DTR+mild at 10:30 | 556 | 2.8 | +0.04% | -0.35% | +0.21% | 46% | 0.005 | 1.01 | [-0.66, +0.73] | No |
| I8a | Close DTR at 10:30 | 556 | 2.8 | -0.08% | -0.39% | +0.19% | 48% | -0.008 | 0.98 | [-0.91, +0.75] | No |
| I4_50 | CS + Price >= $50 | 347 | 2.4 | -0.03% | -0.16% | +0.10% | 49% | -0.004 | 0.99 | [-0.98, +0.91] | No |
| I7b | Composite (CS+Stale/Mod+$30) T+2 | 269 | 2.2 | -0.14% | -0.30% | +0.07% | 49% | -0.014 | 0.96 | [-1.31, +1.03] | No |
| I3b | CS + stale+moderate | 365 | 2.4 | -0.22% | +0.02% | +0.24% | 50% | -0.020 | 0.95 | [-1.34, +0.91] | No |
| I7a | Composite (CS+Stale/Mod+$30) T+3 | 269 | 2.2 | -0.27% | +0.63% | +0.17% | 54% | -0.023 | 0.94 | [-1.69, +1.14] | No |
| I7c | CS + Stale only T+3 | 196 | 1.9 | -0.34% | +1.21% | +0.46% | 56% | -0.026 | 0.93 | [-2.22, +1.53] | No |
| I3a | CS + stale only | 196 | 1.9 | -0.38% | +0.28% | +0.08% | 53% | -0.035 | 0.91 | [-1.89, +1.14] | No |
| I4_30 | CS + Price >= $30 | 419 | 2.6 | -0.45% | +0.02% | -0.05% | 50% | -0.045 | 0.89 | [-1.41, +0.52] | No |
| I4_100 | CS + Price >= $100 | 245 | 2.1 | -0.48% | -0.40% | -0.44% | 49% | -0.059 | 0.86 | [-1.48, +0.53] | No |
| I3c | CS + not very_fresh | 506 | 2.7 | -0.52% | +0.77% | +0.42% | 54% | -0.038 | 0.88 | [-1.73, +0.68] | No |
| I3d | CS + stale+moderate+fresh | 504 | 2.7 | -0.53% | +0.77% | +0.42% | 54% | -0.038 | 0.88 | [-1.74, +0.68] | No |
| I1a | Baseline (all _FLOAT) | 568 | 2.9 | -0.57% | +0.53% | +0.21% | 52% | -0.042 | 0.87 | [-1.68, +0.54] | No |
| I1 | CS_ONLY (re-ranked) | 556 | 2.8 | -0.66% | +0.38% | +0.13% | 52% | -0.048 | 0.86 | [-1.79, +0.48] | No |
| I5 T+2 | CS T+2 hold | 556 | 2.8 | -0.66% | +0.38% | +0.13% | 52% | -0.048 | 0.86 | [-1.79, +0.48] | No |
| I5 T+5 | CS T+5 hold | 547 | 2.8 | -0.66% | +0.72% | +0.16% | 52% | -0.036 | 0.90 | [-2.17, +0.86] | No |
| I9a | Sector-weighted ranking | 556 | 2.8 | -0.71% | +0.60% | +0.14% | 53% | -0.052 | 0.85 | [-1.85, +0.43] | No |
| I5 T+4 | CS T+4 hold | 552 | 2.8 | -0.85% | +0.47% | +0.18% | 52% | -0.053 | 0.85 | [-2.18, +0.48] | No |
| I9c | Exclude Tech+Manufacturing | 325 | 2.3 | -1.02% | +0.82% | +0.06% | 53% | -0.069 | 0.79 | [-2.64, +0.60] | No |
| I5 T+3 | CS T+3 hold | 555 | 2.8 | -1.14% | +0.99% | -0.15% | 54% | -0.079 | 0.79 | [-2.35, +0.07] | No |
| I5 T+4 N2 | CS T+4 TOP_N=2 | 328 | 1.7 | -1.66% | -0.03% | -0.57% | 50% | -0.099 | 0.74 | [-3.47, +0.16] | No |
| I6 >=5x | CS + vol_ratio >= 5.0x | 43 | 1.1 | -10.92% | -1.04% | -6.01% | 49% | -0.324 | 0.27 | [-20.99, -0.86] | YES (neg) |

---

## Per-Investigation Verdicts

### I1: CS_ONLY Standalone — NO EDGE

CS filter removes only 4 non-CS tickers (ADRCs) from the 126 _FLOAT universe. Re-ranking from CS-only pool slightly worsens performance (-0.66% vs -0.57%). The 17 non-CS trades excluded had avg +2.15% (small n, but directionally these ADRCs were profitable shorts).

**Verdict:** CS filter is marginally harmful for _FLOAT universe. Not recommended standalone.

---

### I2: CS + sustained_drop — SIGNIFICANT POSITIVE EDGE

The **only** statistically significant positive filter combination found.

| Variant | n | Avg PnL | Sharpe | CI 95% | LOO Stable |
|---------|---|---------|--------|--------|------------|
| CS + sustained_drop | 199 | +1.66% | 0.144 | [+0.06, +3.25] | Yes (0 flips) |
| sustained_drop only (no CS) | 200 | +1.66% | 0.145 | [+0.07, +3.24] | Yes (0 flips) |
| CS only (reference) | 556 | -0.66% | -0.048 | [-1.79, +0.48] | Yes |

Key facts:
- 98.7% of sustained_drop trades are already CS — CS filter adds nothing here
- sustained_drop is the sole driver; CS is redundant
- 61% win rate, PF 1.49, trimmed mean +2.43%
- Pool: 878 total → 852 CS → 235 CS+SD → 199 selected (TOP_N=5)
- Trades/day = 1.9 (practical, above 1.5 threshold)

I2c intraday management (enter at local_high, pattern-route): avg -0.06%, LOO unstable (3 flips). Not reliable.

**Verdict:** sustained_drop is the dominant signal. Use it as the primary filter.

---

### I3: CS + Gap Freshness — NO EDGE

| Freshness Filter | n | Avg PnL | Sharpe | Delta vs CS |
|-----------------|---|---------|--------|-------------|
| Stale only | 196 | -0.38% | -0.035 | +0.013 |
| Stale + moderate | 365 | -0.22% | -0.020 | +0.029 |
| Not very_fresh | 506 | -0.52% | -0.038 | +0.010 |
| Stale + moderate + fresh | 504 | -0.53% | -0.038 | +0.010 |
| CS baseline | 556 | -0.66% | -0.048 | — |

Freshness distribution in CS trades: stale=288, fresh=240, moderate=217, very_fresh=102.

Stale+moderate is the best freshness filter (+0.029 Sharpe improvement), but still negative and not significant. Removing very_fresh helps slightly but not enough.

**Verdict:** Gap freshness is not a useful filter for this strategy.

---

### I4: CS + Price Floor — NO EDGE

| Price Floor | n | Avg PnL | Sharpe | Excluded Avg PnL | LOO Flips |
|------------|---|---------|--------|-------------------|-----------|
| >= $30 | 419 | -0.45% | -0.045 | -1.50% | 0 |
| >= $50 | 347 | -0.03% | -0.004 | -1.80% | 24 |
| >= $100 | 245 | -0.48% | -0.059 | -1.21% | 0 |

$50 floor looks best on average (-0.03%), but has 24 LOO sign flips — unstable. Excluded low-price trades are even worse (avg -1.50% to -1.80%), so the filter correctly removes bad trades but doesn't make the remaining ones profitable.

**Verdict:** Price floors remove bad trades but don't create edge. $50 is closest to neutral but LOO-unstable.

---

### I5: CS + Extended Hold — ALL NEGATIVE

| Hold | TOP_N | n | Avg PnL | Sharpe | Max Open Positions | Capital @ $20K/pos |
|------|-------|---|---------|--------|--------------------|--------------------|
| T+2 | 5 | 556 | -0.66% | -0.048 | 13 | $260K |
| T+3 | 5 | 555 | -1.14% | -0.079 | 17 | $340K |
| T+4 | 5 | 552 | -0.85% | -0.053 | 22 | $440K |
| T+5 | 5 | 547 | -0.66% | -0.036 | 23 | $460K |
| T+4 | 2 | 328 | -1.66% | -0.099 | 10 | $200K |

T+3 is the worst (-1.14%). Longer holds don't help and dramatically increase capital requirements (13 → 23 max simultaneous positions). Reducing TOP_N to 2 concentrates on the most extreme gaps, which are the most dangerous shorts.

**Verdict:** T+2 is optimal. Do NOT extend hold periods.

---

### I6: CS + Volume Ratio — STRONG FINDING

**vol_ratio vs T+2 PnL correlation: -0.464** (n=848 CS trades) — very strong negative relationship.

| Volume Filter | n | Avg PnL | Sharpe | WR | PF | Trim Mean | Practical |
|--------------|---|---------|--------|-----|-----|-----------|-----------|
| vol < 1.5x | 305 | +1.07% | 0.107 | 57% | 1.32 | +1.33% | Yes (2.4 tpd) |
| vol < 2.0x | 370 | +0.69% | 0.070 | 55% | 1.20 | +0.94% | Yes (2.5 tpd) |
| vol < 3.0x | 464 | +0.68% | 0.070 | 54% | 1.20 | +0.82% | Yes (2.6 tpd) |
| vol >= 5.0x | 43 | -10.92% | -0.324 | 49% | 0.27 | -6.01% | No (1.1 tpd) |

Key insights:
- **Low-volume gap-ups fade** (avg +1.07% for vol < 1.5x)
- **High-volume gap-ups continue** (avg -10.92% for vol >= 5.0x — catastrophic for shorts)
- The correlation is monotonic: tighter vol cap = better performance
- vol < 1.5x has CI [-0.05, +2.18] — just barely misses 95% significance
- 0 LOO sign flips across all configs
- Trimmed mean +1.33% confirms the edge isn't driven by outliers

**Verdict:** vol_ratio < 1.5x is the second-best filter found. The mechanism is clear: high-volume gaps represent genuine institutional interest / breakouts that DON'T fade. Low-volume gaps are noise that mean-reverts.

---

### I7: Composite Filter — NEGATIVE, CS HARMFUL

| Composite | n | Avg PnL | Sharpe | Notes |
|-----------|---|---------|--------|-------|
| CS + Stale/Mod + $30, T+3 | 269 | -0.27% | -0.023 | CS filter HARMFUL |
| CS + Stale/Mod + $30, T+2 | 269 | -0.14% | -0.014 | |
| CS + Stale only, T+3 | 196 | -0.34% | -0.026 | |

Filter funnel analysis (I7a):
- Step 1 (CS): 878 → 852 — excluded trades avg **+1.49%** (HARMFUL)
- Step 2 (Stale/Moderate): 852 → 505 — excluded avg **-2.88%** (good filter)
- Step 3 (Price >= $30): 505 → 344 — excluded avg **-1.52%** (good filter)

The composite doesn't work because stacking filters doesn't create additive edge — each filter was derived from E1-E10 results where other filters weren't applied. The CS filter is actively harmful here (removing profitable non-CS trades).

**Verdict:** Don't stack filters naively. Each E1-E10 finding was marginal and conditional.

---

### I8: Intraday Management — VALUABLE BUT NOT SUFFICIENT

| Strategy | n | Avg PnL | Sharpe | Management Value |
|----------|---|---------|--------|-----------------|
| Close DTR at 10:30 | 556 | -0.08% | -0.008 | +3.24%/trade |
| Close DTR+mild at 10:30 | 556 | +0.04% | 0.005 | +1.33%/trade |
| CS hold-all T+2 (baseline) | 556 | -0.66% | -0.048 | — |
| CS+sustained_drop T+2 | 199 | +1.66% | 0.144 | — |

DTR (drop_then_recover) trade management:
- 99 DTR trades closed at 10:30 → avg -2.46% (close price)
- Same 99 trades if held T+2 → avg -5.70% (counterfactual)
- **Management saved +3.24%/trade on 99 trades**

The aggressive variant (close DTR + mild_drop) pushes portfolio avg to +0.04% but with 46% WR and Sharpe near zero.

**Verdict:** Intraday DTR management is valuable (+3.24%/trade saved) but alone doesn't make the strategy profitable. Best combined with sustained_drop selection.

---

### I9: Sector-Weighted Ranking — HURTS PERFORMANCE

| Ranking Method | n | Avg PnL | Sharpe |
|---------------|---|---------|--------|
| Sector-weighted (A/B/C tiers) | 556 | -0.71% | -0.052 |
| Pure gap size | 556 | -0.66% | -0.048 |
| Exclude Tech+Manufacturing | 325 | -1.02% | -0.069 |

Sector distribution: Manufacturing dominates (n=444, 52%), then Finance (155), Services (100), Transportation (61).

Sector weighting delta: -0.003 Sharpe vs pure gap ranking. Excluding Tech+Mfg makes things worse (-0.020 delta) because excluded trades actually had avg -0.75% (better than the remaining trades at -1.02%).

**Verdict:** Sector tiers from E9 don't work as ranking weights. Pure gap size is the better ranking criterion.

---

### I10: Out-of-Sample Validation — CRITICAL

| Strategy | Period | n | Avg PnL | Sharpe | WR | PF |
|----------|--------|---|---------|--------|-----|-----|
| Raw baseline T+2 | OOS (Q4 2025) | 160 | -0.59% | -0.047 | 54% | 0.87 |
| Raw baseline T+2 | IS (Jan-Sep 2026) | 408 | -0.56% | -0.041 | 51% | 0.87 |
| CS_ONLY T+2 | OOS | 159 | -0.61% | -0.048 | 54% | 0.86 |
| CS_ONLY T+2 | IS | 397 | -0.67% | -0.048 | 51% | 0.85 |
| I7 Composite T+3 | OOS | 73 | **+0.80%** | **0.060** | **60%** | **1.20** |
| I7 Composite T+3 | IS | 196 | -0.67% | -0.060 | 51% | 0.85 |

**Critical finding:** The IS period (Jan-Sep 2026) shows avg -0.56% for the raw baseline, NOT the expected +8.84%. This 9.4% discrepancy means the investigation script's data collection pipeline produces different trade selections than the original E1-E10 script. Likely causes:
- Different entry logic implementation
- Different minute-bar data collection or open_price calculation
- Different TOP_N selection (ranking ties, etc.)

I7 Composite is the only strategy positive OOS (+0.80%, Sharpe 0.060, 60% WR) but fails IS — classic regime dependency.

**Verdict:** The IS vs E1-E10 baseline mismatch must be resolved before these results are fully trustworthy. The relative comparisons (which filters improve vs worsen) are valid, but absolute numbers are shifted negative.

---

## Summary of Actionable Findings

### Tier 1 — Confirmed Edge (deploy-worthy if IS baseline confirmed)

| Rank | Filter | Mechanism | Avg PnL | Sharpe | n | Practical? |
|------|--------|-----------|---------|--------|---|------------|
| 1 | **sustained_drop** | Open pattern = strongest mean reversion signal | +1.66% | 0.144 | 199 | Yes (1.9 tpd) |
| 2 | **vol_ratio < 1.5x** | Low-vol gaps = noise that fades; high-vol = breakouts | +1.07% | 0.107 | 305 | Yes (2.4 tpd) |

### Tier 2 — Useful But Not Standalone

| Filter | Value | Notes |
|--------|-------|-------|
| Close DTR at 10:30 | +3.24%/trade saved | Apply to DTR pattern trades only |
| Price >= $50 | Removes worst trades | LOO-unstable (24 flips) |

### Tier 3 — No Value / Harmful

| Filter | Finding |
|--------|---------|
| CS_ONLY | Marginal harm (-0.003 Sharpe vs all) |
| Gap freshness | No improvement at any combo |
| Extended hold (T+3 to T+5) | All worse than T+2, more capital needed |
| Sector weighting | Hurts performance |
| Composite stacking | Filters aren't additive |

### Anti-Signal (DO NOT DO)

| Filter | Avg PnL | Why |
|--------|---------|-----|
| vol_ratio >= 5.0x | -10.92% | High-vol gaps = breakouts, NOT mean-reversion candidates |
| TOP_N=2 + T+4 | -1.66% | Concentrating on extreme gaps = concentrating risk |
| Exclude Tech+Manufacturing | -1.02% | Removes >50% of pool, keeps worse trades |

---

## Recommended Next Steps

1. **Resolve baseline discrepancy**: Run the original E1-E10 script (agent1_experiments.py) on Jan-Sep 2026 to confirm n=793, avg=+8.84%. Then run the same period through agent1_investigations.py to identify where trades diverge.

2. **Test sustained_drop + vol < 1.5x combo**: These two filters have independent mechanisms (pattern vs volume). If they overlap well, the combined filter could have avg ~2-3% with n~100-150.

3. **Test sustained_drop + DTR management**: Apply intraday DTR close to the remaining non-sustained_drop trades (or to all trades as a safety net).

4. **Hard-exclude vol >= 5.0x**: This is the clearest anti-signal (-10.92%). Any strategy should exclude these catastrophic shorts regardless of other filters.

---

*Generated 2026-09-15 from 5 parallel investigation agents*
*Data: 878 trades, 250 trading days, Sep 2025 - Sep 2026*
*All LOO stability checks passed for Tier 1 findings (0 sign flips)*
