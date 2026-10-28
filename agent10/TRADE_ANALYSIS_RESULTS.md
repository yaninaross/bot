# Agent 10 — RXRX Trade Analysis Results (Tier 1 + Tier 2)

**Date:** 2026-09-26
**Dev period:** 2021-04-16 to 2026-03-31 (1,245 bars)
**Holdout:** 2026-04-01 onward (LOCKED, not spent)
**Position size:** $10,000 per trade
**Costs:** 10 bps half-spread + 7.5 bps slippage + $0.005/share commission + 5% annual borrow (SHORT)

---

## Phase 1 Verdict

| Test | Result | Implication |
|------|--------|-------------|
| Spectral significance (1000 AR(1) surrogates) | **NOT SIGNIFICANT** (best: 82.1th pctile) | ~30d cycle hypothesis REJECTED |
| Hurst exponent (DFA) | **0.457** (< 0.5) | Mean-reverting confirmed |
| Variance Ratio (lag 20) | 0.813 (z = -0.16, p = 0.873) | Directional MR, not stat. sig. |
| OPEX clustering | p = 0.160 | Not calendar-driven |
| Month-end clustering | p = 0.923 | Not calendar-driven |

**Gate:** MR strategies (S1-S7 except S3) UNLOCKED. Cycle phase (S3) BLOCKED.

---

## Sweep Summary

| Metric | Count |
|--------|-------|
| Total trials logged | 674 |
| Strategies with n >= 5 | 607 |
| Survivors (>= 2% avg PnL, n >= 10) | 64 |
| Pass null test (95th pctile) | 25 |
| Pass DSR > 0.5 (corrected for 674 trials) | 5 |
| Pass LOEO (all episode drops) | 10/10 tested |

---

## Top 30 Strategies (sorted by avg PnL, n >= 10)

| Strategy | N | Avg PnL | Sharpe | WR | PF | MaxDD | Hold | Null% | DSR | LOEO |
|----------|---|---------|--------|----|----|-------|------|-------|-----|------|
| S5_OLS_ARKG_k2.0_zexit0.0 | 25 | +6.80% | +1.96 | 60% | 2.94 | -22.0% | 13.3d | 99.8 | — | YES |
| S5_OLS_ARKG_k1.5_zexit0.0 | 54 | +5.92% | +1.58 | 61% | 2.38 | -43.0% | 13.1d | 100 | — | YES |
| S5_OLS_ARKG_k1.5_zexit0.5 | 57 | +5.62% | +1.94 | 67% | 2.48 | -36.5% | 10.3d | 100 | — | YES |
| S11_N60_b0.03_short | 10 | +4.49% | +4.08 | 50% | 3.22 | -13.2% | 3.6d | 93.1 | — | YES |
| S5_OLS_ARKG_k2.0_zexit0.5 | 26 | +4.48% | +1.77 | 62% | 2.23 | -24.5% | 10.8d | 97.7 | — | YES |
| S5_sscore_t1.5_short_h5 | 15 | +4.34% | +2.45 | 67% | 2.36 | -19.5% | 5.8d | 95.2 | — | YES |
| **S5_OLS_XBI_k2.0_short_h5** | **20** | **+4.27%** | **+6.48** | **80%** | **3.54** | **-13.3%** | **2.2d** | **98.1** | **1.00** | **YES** |
| S5_OLS_ARKG_k1.5_zexit1.0 | 68 | +4.04% | +2.28 | 66% | 2.29 | -37.3% | 6.2d | 99.8 | — | YES |
| S11_N40_b0.03_short_h1 | 13 | +3.96% | +8.25 | 62% | 4.77 | -4.3% | 1.5d | 93.4 | — | — |
| S5_OLS_ARKG_k2.0_zexit1.0 | 26 | +3.95% | +1.97 | 65% | 2.19 | -22.0% | 8.4d | 95.5 | — | — |
| S11_N40_b0.03_short_h2 | 13 | +3.81% | +5.37 | 54% | 3.26 | -5.7% | 2.1d | 92.5 | — | — |
| S11_N40_b0.03_short_h3 | 13 | +3.71% | +4.46 | 46% | 2.91 | -7.4% | 2.1d | 92.0 | — | — |
| **S5_OLS_XBI_k2.0_short_h2** | **24** | **+3.70%** | **+6.49** | **75%** | **3.17** | **-13.3%** | **1.8d** | **97.4** | **1.00** | — |
| S5_OLS_XBI_k2.5_short_h5 | 11 | +3.59% | +4.26 | 73% | 2.60 | -13.3% | 2.6d | 90.0 | — | — |
| S11_N20_b0.05_short_h3 | 24 | +3.54% | +4.84 | 50% | 2.86 | -9.6% | 2.2d | 96.4 | — | — |
| S13_bb0.05_long_h5 | 12 | +3.41% | +2.63 | 67% | 2.65 | -9.8% | 6.1d | 89.4 | — | — |
| S5_sscore_t1.5_short_h3 | 20 | +3.38% | +3.18 | 65% | 2.44 | -18.3% | 3.6d | 94.1 | — | — |
| S5_sscore_t1.5_short_h2 | 21 | +3.21% | +3.93 | 67% | 2.48 | -21.2% | 2.6d | 94.1 | — | — |
| S5_confirmed_k2.0_short_h5 | 15 | +3.07% | +3.32 | 73% | 2.47 | -15.9% | 3.9d | 89.4 | 0.29 | — |
| S5_OLS_XBI_k2.5_short_h3 | 11 | +2.94% | +3.93 | 73% | 2.31 | -13.3% | 2.3d | 86.5 | — | — |
| **S5_OLS_ARKG_k1.5_short_h5** | **52** | **+2.87%** | **+3.44** | **69%** | **2.27** | **-23.6%** | **2.9d** | **99.3** | — | — |
| S5_OLS_IWM_k2.5_short_h5 | 10 | +2.78% | +2.96 | 70% | 2.00 | -12.5% | 2.7d | 82.8 | — | — |
| S11_N60_b0.03_short_h2 | 10 | +2.77% | +3.60 | 50% | 2.37 | -13.2% | 2.6d | 82.7 | — | — |
| S11_N60_b0.03_short_h1 | 11 | +2.67% | +6.24 | 64% | 3.72 | -4.0% | 1.5d | 84.3 | — | — |
| S5_OLS_XBI_k2.0_short_h3 | 24 | +2.64% | +5.26 | 75% | 2.70 | -13.3% | 2.2d | 93.2 | 0.55 | — |
| S5_OLS_IWM_k2.0_short_h5 | 18 | +2.60% | +3.55 | 72% | 2.15 | -12.5% | 2.8d | 90.3 | — | — |
| S5_OLS_IWM_k2.0_short_h2 | 22 | +2.55% | +5.37 | 68% | 2.41 | -10.1% | 1.8d | 91.3 | 0.91 | — |
| S5_OLS_ARKG_k2.0_short_h5 | 25 | +2.49% | +2.35 | 60% | 1.73 | -23.6% | 3.3d | 88.9 | — | — |
| S5_OLS_IWM_k2.0_short_h3 | 21 | +2.48% | +4.65 | 71% | 2.34 | -10.3% | 2.3d | 89.6 | 0.54 | — |
| S5_filtered_k1.5_noEv_allVol_h5 | 51 | +2.43% | +2.22 | 63% | 1.83 | -38.7% | 3.0d | 94.0 | — | — |

---

## Strategies Passing ALL Three Gates

Strategies that pass null (>=95th pctile) + DSR > 0.5 + LOEO survival:

| Strategy | N | Avg PnL | Sharpe | Null% | DSR | LOEO |
|----------|---|---------|--------|-------|-----|------|
| **S5_OLS_XBI_k2.0_short_h5** | 20 | +4.27% | +6.48 | 98.1 | 1.00 | YES (min avg +3.24%) |
| **S5_OLS_XBI_k2.0_short_h2** | 24 | +3.70% | +6.49 | 97.4 | 1.00 | not tested |

Note: The zexit family (top raw PnL) produced DSR = NaN because the DSR formula diverges when observed Sharpe < expected max SR under the Bonferroni correction for 674 trials. These strategies pass null convincingly but cannot be validated via DSR. The XBI SHORT family is the cleanest signal.

---

## Long/Short Split (key strategies)

| Strategy | N Long | Avg Long | N Short | Avg Short |
|----------|--------|----------|---------|-----------|
| S5_OLS_ARKG_k2.0_zexit0.0 | 9 | +7.64% | 16 | +6.33% |
| S5_OLS_ARKG_k1.5_zexit0.0 | 25 | +6.10% | 29 | +5.78% |
| S5_OLS_ARKG_k1.5_zexit0.5 | 26 | +4.08% | 31 | +6.91% |
| S5_OLS_XBI_k2.0_short_h5 | 0 | — | 20 | +4.27% |
| S5_sscore_t1.5_short_h5 | 0 | — | 15 | +4.34% |
| S5_OLS_ARKG_k1.5_short_h5 | 0 | — | 52 | +2.87% |
| S11_N60_b0.03_short | 0 | — | 10 | +4.49% |

**Finding:** SHORT side dominates. Pure SHORT strategies have higher Sharpe ratios and cleaner risk profiles. LONG works in zexit family but the edge is weaker and more concentrated.

---

## Pre/Post 2024 Regime Split

| Strategy | N Pre-2024 | Avg Pre-2024 | N Post-2024 | Avg Post-2024 |
|----------|------------|--------------|-------------|---------------|
| S5_OLS_ARKG_k2.0_zexit0.0 | 10 | +5.11% | 15 | +7.93% |
| S5_OLS_ARKG_k1.5_zexit0.5 | 23 | +4.22% | 34 | +6.57% |
| S5_OLS_XBI_k2.0_short_h5 | 6 | +4.01% | 14 | +4.38% |
| S5_sscore_t1.5_short_h5 | 4 | +4.43% | 11 | +4.31% |
| S5_OLS_ARKG_k2.0_zexit0.5 | 10 | +0.30% | 16 | +7.09% |
| S11_N60_b0.03_short | 8 | +4.66% | 2 | +3.79% |

**Finding:** Edge IMPROVES in 2024+ for most S5 variants. The relative value signal got stronger as RXRX diverged more from its ETF peers. S11 is sample-dependent (8 of 10 trades pre-2024).

---

## Episode Analysis (LOEO) — Top Survivors

All 10 tested survivors **pass LOEO** (edge survives dropping any single episode):

| Strategy | Trades | Episodes | LOEO Min Avg | LOEO Min Sharpe |
|----------|--------|----------|--------------|-----------------|
| S5_OLS_ARKG_k2.0_zexit0.0 | 25 | 16 | +5.52% | +1.71 |
| S5_OLS_ARKG_k1.5_zexit0.0 | 54 | 23 | +4.85% | +1.31 |
| S5_OLS_ARKG_k1.5_zexit0.5 | 57 | 28 | +4.92% | +1.78 |
| S11_N60_b0.03_short | 10 | 5 | +2.59% | +2.80 |
| S5_OLS_ARKG_k2.0_zexit0.5 | 26 | 16 | +3.80% | +1.51 |
| S5_sscore_t1.5_short | 15 | 12 | +3.06% | +1.84 |
| S5_OLS_XBI_k2.0_short_h5 | 20 | 12 | +3.24% | +4.84 |
| S5_OLS_ARKG_k1.5_zexit1.0 | 68 | 32 | +3.62% | +1.99 |

No strategy's edge is driven by a single episode.

---

## Near-Survivors (1-2% avg PnL, n >= 15) — Top 10

| Strategy | N | Avg PnL | Sharpe |
|----------|---|---------|--------|
| S11_N20_b0.03_short_h3 | 16 | +1.96% | +2.81 |
| S5_OLS_ARKG_k1.5_long_h5 | 46 | +1.94% | +1.78 |
| S11_N20_b0.03_short_h2 | 16 | +1.94% | +2.85 |
| S5_confirmed_k1.5_short_h3 | 46 | +1.92% | +2.47 |
| S5_asym_L2.0_S1.5_comb_h5 | 67 | +1.91% | +2.01 |
| S5_OLS_IWM_k2.0_combined_h5 | 36 | +1.86% | +1.74 |
| S11_N20_b0.1_short_h5 | 38 | +1.85% | +2.22 |
| S10_vwap40_z2.0_short_h5 | 32 | +1.84% | +1.36 |
| S5_confirmed_k1.5_short_h2 | 51 | +1.82% | +2.66 |
| S11_N40_b0.1_short_h1 | 36 | +1.81% | +3.23 |

---

## Strategy Family Summary

### Tier 1: S5 Refinements

| Variant | Best Config | Avg PnL | Sharpe | Verdict |
|---------|-------------|---------|--------|---------|
| OLS vs ARKG (vanilla) | k=1.5 short h5 | +2.87% | +3.44 | Baseline S5; 52 trades |
| OLS vs XBI | k=2.0 short h5 | +4.27% | +6.48 | **BEST DSR; fewer trades but cleaner** |
| OLS vs IWM | k=2.0 short h2 | +2.55% | +5.37 | Passes DSR; broader market beta |
| Multi-factor (ARKG+XBI) | k=1.5 comb h5 | +1.54% | +1.62 | Below target; over-hedged |
| Multi-factor (ALL) | k=1.5 comb h5 | +1.28% | +1.32 | Below target; over-hedged |
| Kalman filter beta | z40 k1.5 short h5 | +2.05% | +2.10 | Near target; smoother but no lift |
| Avellaneda-Lee s-score | t1.5 short h5 | +4.34% | +2.45 | Strong; HL filter has no effect |
| Signal-based exit (zexit) | k1.5 zexit=0.5 | +5.62% | +1.94 | **HIGHEST raw PnL; long holds** |
| Entry confirmation | k2.0 short h5 | +3.07% | +3.32 | Works but fewer trades |
| Asymmetric thresholds | L2.0/S1.5 h5 | +1.91% | +2.01 | Near-miss; no lift over symmetric |
| Noise filter (no events) | k1.5 noEv h5 | +2.43% | +2.22 | Marginal; events data too sparse |

### Tier 2: New Families

| Family | Best Config | Avg PnL | Sharpe | Verdict |
|--------|-------------|---------|--------|---------|
| S8 (event shock fade) | gap3.0 short h3 | +1.12% | +0.87 | FAIL |
| S9 (gap fade) | gap1-3 short h1 | +0.42% | +0.31 | FAIL |
| S10 (anchored VWAP) | vwap40 z2.0 short h5 | +1.84% | +1.36 | Near-miss |
| S11 (relaxed range fade) | N60 b0.03 short h3 | +4.49% | +4.08 | **PASS but n=10** |
| S13 (vol compression) | bb0.05 long h5 | +3.41% | +2.63 | n=12; too few |

---

## Key Findings

1. **S5 relative value vs XBI is the cleanest signal.** OLS_XBI k=2.0 SHORT passes null (98th), DSR = 1.0, LOEO survives. 20 trades, +4.27% avg, Sharpe 6.48. The edge is SHORT-only (RXRX overvalued vs biotech ETF -> short RXRX, z-score reverts).

2. **Signal-based exits (z-cross) amplify PnL dramatically.** Letting winners run until z crosses back through zero or 0.5 instead of fixed hold produces +5-7% avg PnL. Downside: longer hold periods (10-13 days avg), deeper drawdowns (-22 to -43%), and the annualized Sharpe is lower because returns are lumped.

3. **SHORT side dominates.** Across all families, SHORT variants outperform LONG. This aligns with RXRX's structural downtrend — the stock mean-reverts DOWN more reliably than UP.

4. **Edge IMPROVES in 2024+.** S5 variants show stronger performance in the recent regime, likely because RXRX has diverged more from its ETF peers, creating larger and more reliable residual extensions.

5. **S11 (relaxed range fade) is a surprising find** with +4.49% avg, but only 10 trades. Needs more data to be deployable.

6. **Multi-factor hedging hurts.** Adding more hedge factors (XBI + IWM alongside ARKG) reduces signal strength by over-hedging the idiosyncratic component.

7. **Tier 2 new families (S8, S9) failed.** Gap fade and event shock fade show no edge on RXRX daily data. The signal lives in the residual spread, not in raw price mechanics.

---

## User's Hypothesis Test

> "If I had to bet, the edge lives at the intersection of items 4 and 7: fading residual extensions that aren't news-driven, after confirmation."

**Partially confirmed.** The noise filter (exclude events + require low volume) produces +2.43% avg PnL, and the entry confirmation variant produces +3.07%. These are positive, but the raw unfiltered S5 actually does better. The events data is too sparse (only 3 hardcoded events) to draw conclusions about news-driven filtering. The confirmation filter adds value on the SHORT side.

The edge does live in "fading residual extensions" — S5 relative value is the dominant signal class. But the filter story needs better event data to validate.

---

## Known Issues

1. **DSR = NaN for zexit family:** The DSR formula uses expected max SR under Bonferroni for 674 trials (~3.61). Strategies with annualized Sharpe below this produce negative t-stats. The zexit strategies have Sharpe 1.6-2.3 (annualized over long hold periods), which falls below the haircut.

2. **s-score half-life filter has no effect:** All three hl_max values (20, 30, 50) produce identical trade lists for t=1.5. The OU half-life is always within bounds when the s-score is extreme enough to trigger.

3. **Event data is minimal:** Only 3 hardcoded events. The noise filter test needs earnings dates, pipeline readouts, and financing announcements to be meaningful.

4. **S11 sample size:** 10-13 trades is insufficient for confidence. This family should be monitored OOS before committing capital.

---

## Next Steps (Tier 3)

1. **Combine S5_XBI_SHORT + S5_ARKG_zexit as portfolio** — decorrelated signal sources
2. **Run Tier 3 portfolio construction** — equal weight vs. ROAC, correlation matrix
3. **Phase 4 generalization** — test "short biotech stock when z-score vs ETF > 2" on peer stocks (BEAM, CRSP, NTLA, EDIT, VERV)
4. **Enrich event data** — scrape earnings dates and pipeline catalysts before testing noise filter
5. **Holdout evaluation** — spend holdout ONLY on final combined strategy

---

## Files

| File | Description |
|------|-------------|
| `agent10/results/tier1_tier2_trial_log.csv` | All 674 trials with full metrics |
| `agent10/results/tier1_tier2_summary.csv` | 607 strategies (n>=5) sorted by avg PnL |
| `agent10/results/tier1_heatmap_s5_arkg.png` | S5 ARKG parameter heatmap (k x hold) |
| `agent10/results/tier1_heatmap_s5_kalman.png` | S5 Kalman parameter heatmap |
| `agent10/results/tier1_tier2_equity_curves.png` | Top 8 equity curves |
| `agent10/results/phase1_results.json` | Phase 1 gate decision |
| `agent10/results/p1_*.png` | Phase 1 diagnostic plots (6 files) |
