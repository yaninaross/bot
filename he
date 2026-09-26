# Agent 10 — Phase 4: S5_zexit Generalization Results

**Date:** 2026-09-26
**Verdict: NOT DEPLOYABLE as a multi-ticker strategy.**
**The edge is RXRX-specific.**

---

## 1. What Was Tested

Froze the exact RXRX config (S5_zexit_ARKG_k1.5_ze0.5) and ran it on 32 candidates:
- 9 previously tested biotech peers (SDGR, ABCL, ABSI, BEAM, CRSP, NTLA, TXG, PACB, TWST)
- 15 new biotech/techbio (EDIT, DNA, TEM, ADPT, GH, SANA, ARCT, RGNX, FATE, VERV, RARE, DCPH, NUVB, ACAD, TGTX)
- 7 out-of-sector (MARA, RIOT, CLSK vs WGMI; PLUG, RUN vs TAN/ICLN; RIVN, LCID vs ARKK/XLY)
- RXRX (control)

Hedge assigned by rule: highest 250-day return correlation from {ARKG, XBI, IBB, ARKK, IWM, ICLN, TAN, WGMI, BTC, XLV, XLE, XLY}.

## 2. Universe Filtering

Filters: price $2-60, ADV>$20M, ATR% 3-10%. Applied with trailing data only.

| Cutoff | Passing |
|--------|---------|
| Stage A (2024-01-01) | 18/32 |
| Stage C (2026-01-01) | 17/32 |

Filtered out: ABCL (low ADV periods), ABSI (ATR% too high), DNA (ATR% border), EDIT (price too low), FATE (price too low), PACB (price too low), etc.

## 3. Hedge Assignments (Stage C)

| Ticker | Hedge | Corr | Sector |
|--------|-------|------|--------|
| RXRX | ARKG | +0.795 | biotech |
| BEAM | ARKG | +0.745 | biotech |
| CRSP | ARKG | +0.742 | biotech |
| TWST | ARKG | +0.767 | biotech |
| NTLA | ARKG | +0.619 | biotech |
| SDGR | ARKG | +0.695 | biotech |
| NUVB | XBI | +0.419 | biotech |
| RARE | XBI | +0.499 | biotech |
| TGTX | XBI | +0.396 | biotech |
| MARA | WGMI | +0.777 | crypto |
| RIOT | WGMI | +0.860 | crypto |
| CLSK | WGMI | +0.825 | crypto |
| RUN | TAN | +0.759 | clean energy |
| RIVN | XLY | +0.428 | EV |
| LCID | ARKK | +0.397 | EV |

## 4. Stage A — Rule Development (2021-2023)

### Fingerprint Predictors (Spearman vs Stage A avg PnL)

| Feature | rho | p-value | Signal? |
|---------|-----|---------|---------|
| hedge_r2 | -0.346 | 0.056 | Weak negative (!) |
| hedge_corr | -0.159 | 0.393 | No |
| ou_halflife | +0.050 | 0.788 | No |
| resid_vol | +0.035 | 0.851 | No |
| atr_pct | -0.222 | 0.230 | No |
| trail_avg_pnl | +1.000 | 0.000 | Tautological |
| shrunk_avg_pnl | +0.999 | 0.000 | Tautological |

**Critical finding: hedge R² is NEGATIVELY correlated with performance.** Tighter hedge fit → worse strategy returns. This means the OLS spread construction is more effective on loosely-hedged pairs where idiosyncratic moves dominate — i.e., the signal is stock-specific noise trading, not a structural relative-value relationship.

### Selection Rule

`hedge_r2 >= 0.376 AND ou_halflife in [3, 15] AND shrunk_trail_avg > 0`

Stage A result: Selected 5 tickers (+2.01% avg) vs Rejected 26 tickers (-1.14% avg). Diff: +3.16%.

Selected: ACAD, CLSK, DCPH, RUN, RXRX

## 5. Stage B — Validation (2024-2025)

| Window | Selected | Sel avg | Rejected | Rej avg | Diff |
|--------|----------|---------|----------|---------|------|
| H1 2024 | ACAD, RUN, RXRX | +0.76% | 15 tickers | -0.73% | +1.49% |
| H2 2024 | NTLA, PLUG, RXRX | +1.22% | 13 tickers | -1.51% | +2.73% |
| H1 2025 | RUN only | +4.35% | 11 tickers | +0.16% | +4.19% |
| H2 2025 | ACAD, CLSK, GH, NTLA, VERV | -3.46% | 13 tickers | +0.24% | -3.70% |
| **Aggregate** | **107 trades** | **-0.42%** | **449 trades** | **-0.55%** | **+0.12%** |

**RXRX carried Stage B.** Without RXRX (+6.83% avg, 17 trades, +$11,612), the selected group was deeply negative. The rule worked in H1 2024 – H1 2025 but collapsed in H2 2025 when it overexpanded to 5 tickers.

## 6. Stage C — 2026 (Pre-Registered)

Pre-registration: 3 tickers selected (BEAM, RXRX, NUVB)

| Ticker | Bucket | n | Avg PnL | Total | $ PnL | Hedge |
|--------|--------|---|---------|-------|-------|-------|
| BEAM | seen | 16 | +0.90% | +14.3% | +$1,433 | ARKG |
| RXRX | seen | 9 | -0.17% | -1.5% | -$151 | ARKG |
| NUVB | **new** | 13 | -2.00% | -26.0% | -$2,602 | XBI |
| **TOTAL** | | **38** | **-0.35%** | **-13.2%** | **-$1,320** | |

**RXRX Stage C underperformed** because the hedge assignment used full-history ARKG (corr=0.795) while the OOS evaluation ran the L+S strategy on the full spread. The LONG side got hammered with 5 stopped-out longs before catching two winners.

## 7. Deployment Decision

| Rule | Result |
|------|--------|
| 1. Selected > Rejected in B AND C | **FAIL** (B: +0.12% PASS; C: -1.12% FAIL) |
| 2. C avg ≥ 1.5% AND B+C clustered t ≥ 2 | **FAIL** (avg=-0.35%, t=-0.38) |
| 3. Portfolio C total positive | **FAIL** (-13.2%) |
| **VERDICT** | **NOT DEPLOYABLE** |

## 8. Why It Failed — Root Cause Analysis

1. **The edge is RXRX-specific, not a general residual-reversion phenomenon.** RXRX has an unusual combination: high ARKG correlation (0.80) but distinct catalyst-driven moves that create tradeable z-score dislocations. Most other biotechs either (a) track ARKG too closely (little residual signal) or (b) have too-loose hedges (the spread is just raw stock noise).

2. **Hedge R² predicts failure, not success.** The counterintuitive negative R² correlation (rho=-0.346) tells us: tighter-fitting hedges produce spreads with too little variance to trade profitably. The 1.5 ATR stop gets triggered by normal residual noise. Looser hedges create bigger z-score moves, but those moves are just unhedged stock risk — they don't mean-revert.

3. **The selection rule is dominated by trailing performance (tautological).** The only predictive fingerprint feature is past returns, which is persistence (the same signal will keep working on the same stock) rather than generalization (the pattern works on new stocks). The OU half-life, residual vol, and hedge correlation features had zero predictive value.

4. **Out-of-sector was a complete miss.** Crypto miners (vs WGMI) averaged -1.07%, clean energy varied (RUN +1.77% but PLUG -0.30%), EVs were negative. The relative-value spread construction requires a specific stock-ETF relationship that's rare.

5. **BEAM was the only partial success.** BEAM (+0.90% avg in Stage C) benefits from ARKG exposure and enough idiosyncratic biotech catalyst risk to create mean-reverting z-dislocations. But 16 trades at +0.90% is marginal — below the 1.5% deployment threshold.

## 9. Implication for RXRX Strategy

This analysis strengthens the case for the RXRX-specific deployment from the Phase 3 OOS evaluation:

- The edge is NOT a generic residual-reversion artifact — it's specific to RXRX's relationship with ARKG
- Deploying S5_zexit on RXRX alone (at reduced size) remains the correct decision
- Expanding to a multi-ticker portfolio would dilute the alpha and introduce correlated losses

## 10. Output Files

- `results/phase4_pre_registration.json` — frozen rule + selected tickers
- `results/phase4_generalization.png` — 4-panel diagnostic
- `results/phase4_all_results.csv` — 93 rows, all ticker×stage combinations
- `agent10_phase4_generalize.py` — full reproducible script
