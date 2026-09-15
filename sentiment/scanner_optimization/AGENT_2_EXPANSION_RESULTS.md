# Agent 2: Expansion + Logging — Experiment Results

Generated: 2026-09-13 21:51:17
Dataset: v2_features_full.csv — 316 LLM trades, 4 dates (Sep 8-11, 2026)

## 1. Scanner Silence Diagnosis

| Date | Total | Premarket | Discovery | SHORT gap-up | SHORT >=3% | Has T+2 |
|------|-------|-----------|-----------|-------------|-----------|---------|
| 2026-09-08 | 102 | 58 | 44 | 24 | 13 | 102 |
| 2026-09-09 | 15 | 7 | 8 | 0 | 0 | 15 |
| 2026-09-10 | 58 | 23 | 35 | 5 | 1 | **0** |
| 2026-09-11 | 141 | 19 | 122 | 49 | 16 | **0** |

**DIAGNOSIS**: Scanner runs on ALL 4 days (316 total signals, 79/day avg). 
Days 2026-09-10, 2026-09-11 have 0 T+2 outcome data — this is a **data availability issue**, not scanner silence. 
In live trading, Agent 2 would have signals every day. Backtest T+2 metrics use only 2 dates with complete data.

## 2. Gap Threshold Sweep (B1)

Universe: LLM SHORT gap-up, catalyst-filtered (excl earnings/deal/macro)

| Threshold | n_total | n_T+2 | t/d | Avg% | Med% | WR | Sharpe | CI_lo | CI_hi | CI>0 |
|-----------|---------|-------|-----|------|------|-----|--------|-------|-------|------|
| 0.5% | 44 | 15 | 11.0 | +8.80 | +6.27 | 87% | +0.751 | +2.87 | +14.73 | YES |
| 1.0% | 39 | 13 | 9.8 | +9.08 | +5.97 | 85% | +0.719 | +2.21 | +15.95 | YES |
| 1.5% | 29 | 10 | 7.2 | +10.48 | +6.12 | 80% | +0.737 | +1.67 | +19.30 | YES |
| 2.0% | 25 | 8 | 6.2 | +11.67 | +6.53 | 75% | +0.735 | +0.67 | +22.68 | YES |
| 2.5% | 20 | 8 | 5.0 | +11.67 | +6.53 | 75% | +0.735 | +0.67 | +22.68 | YES |
| 3.0% | 18 | 7 | 4.5 | +14.39 | +7.08 | 86% | +0.959 | +3.27 | +25.51 | YES |
| 4.0% | 14 | 6 | 3.5 | +15.80 | +9.12 | 83% | +0.992 | +3.05 | +28.54 | YES |
| 5.0% | 13 | 5 | 3.2 | +19.02 | +11.17 | 100% | +1.230 | +5.47 | +32.58 | YES |
| 8.0% | 6 | 5 | 1.5 | +19.02 | +11.17 | 100% | +1.230 | +5.47 | +32.58 | YES |

## 3. Catalyst Inclusion Expansion (B2)

### Per-Catalyst Breakdown (SHORT gap-up >= 2%)

| Catalyst | n | n_T+2 | Avg% | WR | Sharpe |
|----------|---|-------|------|-----|--------|
| offering | 4 | 2 | +35.80 | — | — |
| earnings | 5 | 2 | +14.64 | — | — |
| premarket_move | 2 | 1 | +12.98 | — | — |
| deal | 6 | 3 | +10.40 | 100% | +1.100 |
| analyst | 8 | 2 | +5.60 | — | — |
| breaking | 2 | 2 | +3.37 | — | — |
| insider | 2 | 2 | -1.04 | — | — |
| bankruptcy | 1 | 0 | — | — | — |
| regulatory | 5 | 0 | — | — | — |
| unknown | 4 | 0 | — | — | — |

### Catalyst Set Comparison

| Set | Excludes | n | n_T+2 | t/d | Avg% | Sharpe | CI>0 |
|-----|----------|---|-------|-----|------|--------|------|
| current | earnings, deal, macro, breaking | 23 | 6 | 5.8 | +14.44 | +0.819 | YES |
| add_breaking | earnings, deal, macro | 25 | 8 | 6.2 | +11.67 | +0.735 | YES |
| add_macro | earnings, deal | 25 | 8 | 6.2 | +11.67 | +0.735 | YES |
| minimal_earnings_only | earnings | 28 | 9 | 7.0 | +12.74 | +0.838 | YES |
| no_exclusions |  | 39 | 14 | 9.8 | +11.49 | +0.919 | YES |
| curated_best | earnings, deal, breaking | 23 | 6 | 5.8 | +14.44 | +0.819 | YES |

## 4. Discovery vs Premarket Scanner (B3)

| Scanner | n | n_T+2 | Avg% | Med% | WR | Sharpe | CI_lo | CI_hi |
|---------|---|-------|------|------|-----|--------|-------|-------|
| premarket | 6 | 4 | +2.36 | +2.82 | 50% | +0.295 | -5.48 | +10.21 |
| discovery | 19 | 4 | +20.99 | +21.23 | 100% | +1.226 | +4.21 | +37.76 |

## 5. n_runs Signal Analysis (B4)

| n_runs >= | n | n_T+2 | t/d | Avg% | WR | Sharpe | CI>0 |
|-----------|---|-------|-----|------|-----|--------|------|
| 1 | 25 | 8 | 6.2 | +11.67 | 75% | +0.735 | YES |
| 2 | 3 | 2 | 0.8 | -3.85 | — | — | — |
| 3 | 1 | 1 | 0.2 | -0.34 | — | — | — |
| 4 | 0 | 0 | 0.0 | — | — | — | — |
| 5 | 0 | 0 | 0.0 | — | — | — | — |

## 6. Tier System Simulation

### Per-Day Selection (target: 3/day)

**2026-09-08** — 10 qualified (T1:0, T2:5, T3:5)

| # | Tier | Ticker | Gap% | Catalyst | n_runs | T+2 PnL |
|---|------|--------|------|----------|--------|---------|
| 1 | T2 | PDSB | +121.8% | offering | 1 | +35.39% |
| 2 | T2 | FCEL | +17.4% | earnings | 1 | +11.17% |
| 3 | T2 | SHMD | +8.2% | insider | 1 | +5.28% |

**2026-09-09** — 0 qualified (T1:0, T2:0, T3:0)

| # | Tier | Ticker | Gap% | Catalyst | n_runs | T+2 PnL |
|---|------|--------|------|----------|--------|---------|

**2026-09-10** — 1 qualified (T1:0, T2:0, T3:1)

| # | Tier | Ticker | Gap% | Catalyst | n_runs | T+2 PnL |
|---|------|--------|------|----------|--------|---------|
| 1 | T3 | OTLK | +3.3% | regulatory | 2 | no data |

**2026-09-11** — 18 qualified (T1:2, T2:12, T3:4)

| # | Tier | Ticker | Gap% | Catalyst | n_runs | T+2 PnL |
|---|------|--------|------|----------|--------|---------|
| 1 | T1 | LASE | +6.8% | offering | 1 | no data |
| 2 | T1 | INTC | +2.1% | offering | 1 | no data |
| 3 | T2 | SWRD | +98.7% | unknown | 1 | no data |

### Aggregate Tier Performance

- Total selected: 7 (1.8/day)
- With T+2 data: 3
- Avg T+2 PnL: +17.28%
- Median T+2 PnL: +11.17%
- Win rate: 100%

## 7. FINAL AGENT 2 SPECIFICATION

### Tier Criteria

**Tier 1** (highest priority):
- LLM-flagged SHORT, gap-up >= 2%
- Catalyst: offering OR regulatory
- PM retrace: 0.2-0.8 (partial retrace sweet spot)
- Exclude: earnings, deal

**Tier 2** (fill to 3/day):
- LLM-flagged SHORT, gap-up >= 2%
- Any catalyst except earnings, deal, breaking
- PM retrace < 0.8

**Tier 3** (fallback only if T1+T2 < 3):
- LLM-flagged SHORT, gap-up >= 1.5%
- Exclude: earnings, deal
- Rank by n_runs × |gap_pct|

## 8. Logging Schema Validation

- Total fields: 63
- Computable from backtest data: 51
- Needs new data in live: 0
- Live-only fields: 12

### Live-Only Fields (expected)
- `scanner_run_time`: live trading only
- `pm_volume_profile`: live trading only
- `float_shares`: live trading only
- `orb_width_atr_multiple`: live trading only
- `slippage_pct`: live trading only
- `sector_etf`: live trading only
- `sector_direction_at_entry`: live trading only
- `actual_hold`: live trading only
- `actual_pnl`: live trading only
- `exit_reason`: live trading only
- `borrow_rate_pct`: live trading only
- `notes`: live trading only

## Multi-Hold Analysis

| Hold | n | Avg% | Med% | WR | Sharpe | CI_lo | CI>0 |
|------|---|------|------|-----|--------|-------|------|
| 30m | 22 | +1.55 | +0.38 | 77% | +0.307 | -0.56 | NO |
| 60m | 19 | +3.17 | +0.51 | 58% | +0.293 | -1.69 | NO |
| 120m | 16 | +3.71 | -0.02 | 50% | +0.263 | -3.20 | NO |
| EOD | 25 | +3.59 | +0.51 | 64% | +0.340 | -0.54 | NO |
| T+1 | 9 | +8.87 | +4.10 | 67% | +0.522 | -2.24 | NO |
| T+2 | 8 | +11.67 | +6.53 | 75% | +0.735 | +0.67 | YES |
| T+3 | 8 | +11.53 | +7.97 | 75% | +0.589 | -2.02 | NO |
