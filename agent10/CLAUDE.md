# Agent 10 — Cyclical Mean Reversion (RXRX)

## What This Agent Does
Investigates whether RXRX exhibits a statistically real, recurring
oscillation (~25-35 trading day period) with stable support/resistance.
If confirmed, designs and stress-tests a robust long/short strategy
with 1-5 day holding period. Tests generalization to other stocks
with the same statistical fingerprint.

Pure statistical / technical. No ML model training (except meta-labeling
filter in S7). Completely independent of bot-RAG spike transformer
and Agent 1/3 systems.

## Ground Rules (Non-Negotiable)
1. NO LOOKAHEAD. Signals on close of bar t, execute at open of t+1.
   Also report MOC variant.
2. FINAL HOLDOUT LOCKED. Most recent 6 months (2026-04-01 onward).
   Do not load, plot, or evaluate until final step.
3. LOG EVERY TRIAL. All variants + parameters in trials table.
   Count feeds multiple-testing corrections in Phase 3.
4. REALISTIC COSTS. Half-spread + 5-10 bps slippage + per-share
   commission + short borrow (3-10% annualized sensitivity).
5. PREFER SIMPLICITY. 2-param strategy on stable plateau beats
   6-param strategy on sharp peak.

## Holdout Cutoff
HOLDOUT_START = '2026-04-01'
Everything before this is available for development/validation.
Everything from this date onward is LOCKED until final evaluation.

## Phase Structure
| Phase | Description                    | Script                    | Status  |
|-------|--------------------------------|---------------------------|---------|
| P0    | Data acquisition               | agent10_phase0_data.py    | PENDING |
| P1    | Cycle characterization (gate)  | agent10_phase1_cycle.py   | PENDING |
| P2    | Strategy families S0-S7        | agent10_phase2_strategies.py | PENDING |
| P3    | Validation protocol            | agent10_phase3_validation.py | PENDING |
| P4    | Generalization to other stocks | agent10_phase4_generalize.py | PENDING |
| P5    | Deliverables & final report    | agent10_phase5_report.py  | PENDING |

## Data Sources
- Polygon API: daily + intraday OHLCV for RXRX + context series
- Context: ARKG, XBI, IWM, SPY, VIX (^VIX), BTC (X:BTCUSD)
- ARK trades: public CSV from ark-funds.com
- Short interest: FINRA bi-weekly (if available via Polygon)
- Events: earnings dates, pipeline readouts, financing, M&A

## Strategy Families
| ID | Name              | Description                                | Gate    |
|----|-------------------|--------------------------------------------|---------|
| S0 | Baseline          | User's discretionary rules (TO BE DEFINED) | None    |
| S1 | Z-score reversion | Z on rolling mean/std, ±ARKG-hedged        | MR only |
| S2 | Range fade        | S/R from rolling low/high or KDE clusters  | MR only |
| S3 | Cycle phase       | Hilbert phase on bandpass cycle             | P1 gate |
| S4 | Short-horizon rev | RSI(2), IBS, ConsecutiveCloses, ConnorsRSI  | MR only |
| S5 | Relative value    | RXRX vs ARKG/basket, Kalman hedge ratio    | MR only |
| S6 | Regime-gated comp | Best of S1/S2/S4 + Hurst/HMM regime filter | MR only |
| S7 | Meta-labeling     | GBM filter on best rule-based strategy     | MR only |

## Key Metrics (report for every strategy)
Trade count (min 30 OOS), win rate, avg win/loss in R-multiples,
expectancy, profit factor, CAGR, Sharpe, Sortino, max DD, Calmar,
exposure %, avg holding days, cost drag.

## Baseline S0
**STATUS: AWAITING USER INPUT**
User needs to provide their current discretionary rules for RXRX:
- How they identify levels
- Entry trigger
- Exit rules
- Stop loss rules
