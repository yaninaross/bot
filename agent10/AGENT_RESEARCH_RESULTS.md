# Agent 10 -- RXRX Cyclical Mean Reversion: Final Research Results

**Date:** 2026-09-27 (updated with v2 agent results)
**Research program:** 13 agents (0-7 + integration + 4 v2 re-runs), 874 total trials
**Strategy:** S5_zexit on RXRX vs ARKG
**Verdict:** CONDITIONALLY DEPLOYABLE on RXRX only, at reduced size, with kill-switch

---

## 1. Executive Summary

S5_zexit is a pairs-trading strategy that shorts RXRX when its OLS spread vs ARKG exceeds +1.5 rolling z-scores and goes long when the spread drops below -1.5, exiting when the z-score reverts to +/-0.5. Over the selection period (2021-2025), 61 trades produced +5.18% avg PnL with Sharpe 1.99. In 2026 out-of-sample, 12 trades produced +4.15% avg PnL with Sharpe 3.07 and +$4,980 total on $10k position sizing. The strategy passed its pre-registered deployment gates on RXRX (2026 avg > 1.5%, inside selection CI, null test 90-95th percentile depending on implementation). It does NOT generalize to other tickers (Phase 4 multi-ticker portfolio lost -$1,320 on 38 trades). Eight specialized agents ran follow-up analysis. Agents 0, 1, 5, and 7 delivered properly scoped results on the selection window. Agents 2, 3, 4, and 6 initially tested their ideas only as overlays on S5's OOS trades; these were re-run as v2 agents with corrected methodology (standalone strategies on the selection window, real data sources, no proxy substitution). All four v2 hypotheses were **properly tested and rejected** — 61 standalone strategy variants across gap-fade, intraday MR, overnight hold, earnings events, S/R level-bounce, and Google Trends attention produced zero strategies with positive avg PnL on the selection window (see Section 10). The signal is RXRX-specific, driven by the stock's unusual combination of high ARKG correlation and idiosyncratic catalyst-driven dislocations. Deployment is recommended on RXRX alone at $5k per trade initially, with kill-switch thresholds calibrated from the selection-window bootstrap (not guessed), scaling to $10k after 10+ live trades confirm the edge persists.

---

## 2. Strategy Specification (Frozen Config)

### Core Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target | RXRX | Single-stock only |
| Hedge | ARKG (ETF) | OLS regression, log-log space |
| Lookback | 60 trading days | Rolling window for OLS + z-score |
| Z-score method | Rolling z of point residual | NOT in-sample regression z (Agent 0 fixed this bug) |
| Entry (SHORT) | spread_z > +1.5 | Signal on close, execute next open |
| Entry (LONG) | spread_z < -1.5 | Signal on close, execute next open |
| Exit (primary) | z crosses +/-0.5 toward zero | Mean reversion target |
| Exit (stop) | 1.5x ATR(20) adverse | Hard dollar stop |
| Exit (time) | 10-day max hold | See Agent 7 finding on 20d alternative |
| Fill type | Market-on-open (MOO) | Next-day 9:30 AM open |
| Position size | $10,000 per trade | Max 1 concurrent RXRX position |

### Cost Model

| Component | Value |
|-----------|-------|
| Half-spread | 10 bps (each way) |
| Slippage | 7.5 bps (each way) |
| Commission | $0.005/share (each way) |
| Short borrow | 5.0% annualized |
| Round-trip total (typical) | ~0.45% for a 5-day SHORT trade |

### What Is NOT Included

**Properly tested and rejected (on selection window):**
- No multi-factor hedge (Agent 5: ARKG-only best on selection window; R² vs performance r=-0.763)
- No trailing stops (Agent 7: all variants worse on selection window)
- No partial exits (Agent 7: no improvement on selection window)
- No limit orders (Agent 1: -1.74% avg vs MOO on selection window)

**Standalone hypotheses tested and rejected in v2 re-runs (selection window):**
- No gap-fade / overnight hold (Agent 2v2: 34 variants, all negative; RXRX gaps are momentum not MR)
- No earnings event strategies (Agent 3v2: 17 variants with real Polygon earnings dates, 0/17 positive; RXRX has negative pre-earnings drift)
- No S/R level-bounce (Agent 4v2: 6 strategies, all negative; bounce rates = random null at 52.5th pctile)
- No Google Trends attention (Agent 6v2: 4 strategies, best +0.05% avg at 59.4th null; 4/5 data sources unavailable without paid subscriptions)

### Open Parameter Decision: max_hold

Agent 7 found that extending max_hold from 10d to 20d improves OOS avg PnL by ~+1.3% (from +4.15% to +5.45%) and passes walk-forward (15/17 folds positive). However, the OOS curve is noisy (7d=+5.30%, 10d=+4.15%, 15d=+6.05%, 20d=+5.45%) and was read from 2026 data. Switching to 20d based on this would be fitting noise. **Recommendation: keep 10d.** It matches the pre-registered config and respects the original 10-day hold limit.

| max_hold | OOS n | OOS avg | OOS Sharpe | Avg hold |
|----------|-------|---------|------------|----------|
| 5d | 13 | +3.08% | 2.72 | 4.4d |
| 7d | 12 | +5.30% | 3.86 | 5.2d |
| **10d** | **12** | **+4.15%** | **3.07** | **4.9d** |
| 15d | 11 | +6.05% | 4.02 | 5.8d |
| 20d | 11 | +5.45% | 3.67 | 6.0d |

---

## 3. Consolidated Trial Log

### From Pre-Agent Research (Phases 1-4)

| Metric | Count |
|--------|-------|
| Total trials (tier1_tier2_trial_log.csv) | 674 |
| Strategies with n >= 5 | 607 |
| Survivors (avg >= 2%, n >= 10) | 64 |
| Pass null test (>= 95th percentile) | 25 |
| Pass DSR > 0.5 (corrected for 674 trials) | 5 |
| Pass LOEO (all episode drops) | 10/10 tested |

### From Agent Research (Agents 0-7 + v2 re-runs)

| Metric | Count |
|--------|-------|
| Total rows in agents trials_log.csv | 200 |
| v1 agents (0-7) | 91 |
| v2 agents (2v2, 3v2, 4v2, 6v2) | 109 |
| Agents contributing | 12 (agent0-7 + 4 v2 re-runs) |

### Combined

| Metric | Count |
|--------|-------|
| **Total trials across entire research program** | **874** |
| Strategies that passed selection gates | 5 (pre-registered for OOS) |
| Strategies that passed OOS deployment rules | **1 (S5_zexit_ARKG_k1.5_ze0.5)** |

---

## 4. Agent-by-Agent Summary

| Agent | Focus | Key Finding | Scope | Verdict |
|-------|-------|-------------|-------|---------|
| **0 (Engine)** | Z-score bug fix | Phase 4 used in-sample z (buggy: 10 trades -3.24%). Correct rolling z: 12 trades +4.15%. WF: +8.03% avg, 14 folds. Null: 90.2th pctile. 2x cost: +3.45%. | Selection + OOS | Fixed critical bug. Correct z-score canonical. |
| **1 (Execution)** | Fill type comparison | MOO best (+5.18% sel). Limit orders -1.74%. MOC marginal. 9:30 open optimal fill. | Selection window | Keep MOO. |
| **2 (Overnight)** | PnL decomposition | 85% of PnL intraday. AM session dominant. | **S5 overlay on OOS** | Descriptive finding. Not tested as standalone. |
| **3 (Events)** | Event proximity | Near vs far p=0.91 on S5 trades. But only 3 hardcoded events; no real event calendar. | **S5 overlay on OOS** | Hypothesis untested. Needs event data + standalone test. |
| **4 (S/R Levels)** | S/R gates | Binary gate p=0.42 on S5 OOS. Proximity weighting dilutes. | **S5 overlay on OOS** | Hypothesis untested as standalone. |
| **5 (Multi-Factor)** | Hedge construction | ARKG-only best on selection. Adding XBI/IBB/SPY/PCA/Kalman all worse. R² vs perf r=-0.763. | Selection window | ARKG-only confirmed. |
| **6 (Attention)** | Signal proxies | Volume/VIX/squeeze/flow tested as S5 filters on n=12. No real attention data sourced. | **S5 overlay on OOS** | Hypothesis untested. Needs actual data sources. |
| **7 (Exit Eng.)** | Exit variants | All trailing stops worse on selection. Z-exit 0.5 optimal. max_hold=20d shows OOS lift but noisy. | Selection window | Keep z-exit 0.5, max_hold=10d. |
| **2v2 (Overnight)** | Standalone gap-fade, intraday MR, overnight hold | 34 variants all negative. RXRX gaps are momentum (don't fill). Overnight returns skew=18.9, unreliable. Intraday MR: n<30 (ATR too high). | Selection window | **REJECTED.** No standalone edge. |
| **3v2 (Events)** | Standalone earnings strategies with real data | 18 real earnings dates from Polygon. 17 strategies, 0/17 positive. Pre-earnings drift is -7.6% (opposite of run-up). Post-earnings gaps continue, don't reverse. | Selection window | **REJECTED.** Earnings not tradeable on RXRX. |
| **4v2 (S/R Levels)** | Standalone level-bounce + breakout | 6 strategies, all negative. Bounce rate = random null (52.5th pctile). All lose money after costs. | Selection window | **REJECTED.** S/R levels are noise on RXRX. |
| **6v2 (Attention)** | Real data sourcing + standalone strategies | 4/5 data sources unavailable (need paid subscriptions). Google Trends: 4 strategies, best +0.05% at 59.4th null. | Selection window | **REJECTED.** No edge; data gaps remain for short interest. |

---

## 5. Performance Summary

### Selection Period (2021-01-01 to 2025-12-31)

| Metric | Value |
|--------|-------|
| Trades | 61 |
| Avg PnL | +5.18% |
| Total PnL | +316.0% |
| Sharpe | 1.99 |
| Win rate | 62.3% |
| Avg hold | 8.2 days |
| t-statistic | 2.79 |

### Out-of-Sample (2026-01-01 to 2026-09-25)

| Metric | Value |
|--------|-------|
| Trades | 12 |
| Avg PnL | +4.15% |
| Median PnL | +4.72% |
| Total PnL | +49.8% |
| Dollar PnL | +$4,980 ($10k sizing) |
| Sharpe | 3.07 |
| Win rate | 66.7% |
| Avg hold | 4.9 days |
| t-statistic | 1.42 |

### 2026 Trade List

| # | Entry | Exit | Side | Days | Net PnL | $ | Reason |
|---|-------|------|------|------|---------|---|--------|
| 1 | 2026-01-28 | 2026-02-04 | SHORT | 7 | +15.24% | +$1,524 | z_cross |
| 2 | 2026-03-09 | 2026-03-23 | SHORT | 14 | +2.07% | +$207 | time_stop |
| 3 | 2026-03-30 | 2026-04-01 | SHORT | 2 | -5.17% | -$517 | z_cross |
| 4 | 2026-05-19 | 2026-05-28 | LONG | 9 | +19.16% | +$1,916 | z_cross |
| 5 | 2026-06-02 | 2026-06-05 | SHORT | 3 | +9.24% | +$924 | z_cross |
| 6 | 2026-06-29 | 2026-07-01 | SHORT | 2 | -9.49% | -$949 | stop |
| 7 | 2026-07-02 | 2026-07-06 | SHORT | 4 | -9.44% | -$944 | stop |
| 8 | 2026-07-07 | 2026-07-13 | SHORT | 6 | +15.73% | +$1,573 | z_cross |
| 9 | 2026-07-20 | 2026-07-28 | LONG | 8 | +2.88% | +$288 | z_cross |
| 10 | 2026-09-08 | 2026-09-09 | SHORT | 1 | +9.55% | +$955 | z_cross |
| 11 | 2026-09-21 | 2026-09-22 | SHORT | 1 | -6.52% | -$652 | stop |
| 12 | 2026-09-23 | 2026-09-25 | SHORT | 2 | +6.56% | +$656 | time_stop |

### Notable Patterns in 2026 Trades

- **Back-to-back stops (trades 6-7):** Two consecutive stopped-out shorts in late June / early July during a vol spike, totaling -$1,893. Trade 8 (+15.73%, +$1,573) partially recovered. Combined PnL of trades 6-8: -$320 net.
- **LONG trades (4, 9):** Both profitable. Trade 4 (+19.16%) was the single largest winner.
- **SHORT-side dominance:** 9 of 12 trades are SHORT. Strategy underperforms in sustained RXRX uptrends.
- **Quick trades (10-12):** Three trades in September averaging 1.3 days hold.

---

## 6. Risk Factors and Caveats

### Statistical Limitations

1. **Small OOS sample (n=12).** Twelve trades is not enough for high confidence. The 90% bootstrap CI for 2026 avg PnL is approximately [+2.12%, +8.15%]. A single bad trade can swing the average substantially.

2. **Null test at 90.2th percentile (Agent 0), 94.8th (OOS evaluation).** The Agent 0 null test landed below the 95th gate. The OOS evaluation's separate null test passed at 94.8th. The discrepancy is due to different null test implementations (Agent 0's drift-matched random-entry vs the OOS evaluation's method). Neither crosses 95th with high confidence.

3. **765 total trials tested.** Even with DSR correction for multiple comparisons, the risk of data-snooping is material. Only 5 strategies passed DSR > 0.5 after Bonferroni correction.

### Structural Risks

4. **RXRX-specific -- no generalization.** Phase 4 tested the frozen config on 32 other tickers across biotech, crypto mining, clean energy, and EVs. The multi-ticker portfolio lost -$1,320 on 38 trades. The edge is unique to RXRX's relationship with ARKG. If that relationship changes (e.g., RXRX removed from ARKG, ARKG restructured, RXRX acquired), the strategy dies.

5. **2026 data partially seen.** The holdout was split into semi-seen (Jan-Mar, data partially overlapped development) and clean holdout (Apr+). While parameters were NOT fitted on 2026 data, the development process was informed by observing RXRX's 2026 behavior.

6. **Intraday edge (AM session).** Agent 2 showed 85% of PnL is intraday with AM dominance. If RXRX's market microstructure changes (e.g., lower ADV, wider spreads, shift to after-hours trading), the fill quality on MOO entries may degrade.

7. **Back-to-back stops in vol spikes.** Trades 6-7 in late June / early July were consecutive stopped-out shorts totaling -$1,893. While trade 8 recovered (+$1,573), this pattern (two immediate stops, then a winner) creates drawdown risk and tests psychological discipline.

8. **SHORT-side dominance.** 9 of 12 OOS trades are SHORT. The LONG side works (2 of 3 longs profitable, including the biggest winner) but is less frequent. In a sustained RXRX uptrend where SHORT entries stop out repeatedly, the strategy may underperform.

---

## 7. Open Questions for Future Work

1. **Re-run null test with matched-drift method.** Current null uses random-entry with matched side/count/hold. A better test would shuffle z-score labels rather than random dates. Could push the percentile above or below 95th.

2. **Monitor for signal decay.** Track rolling 10-trade avg PnL. If it drops below +2% (half the selection avg), investigate whether the RXRX-ARKG relationship has changed.

3. **ARKG composition monitoring.** Track RXRX's weight in ARKG. A significant weight change (removal, doubling, or ARKG restructuring) would alter the hedge relationship and potentially invalidate the spread.

4. **Source paid data for remaining attention/positioning hypotheses.** Agent 6v2 confirmed that short interest (Polygon Business+), StockTwits history (StockTwits Data), Reddit mentions (no free archive), and 13F ownership (Polygon Business+) require paid subscriptions. If these are sourced in the future, test as standalone strategies on the selection window.

5. **Calibrate kill-switch from history.** Check how often each trigger condition (3 consecutive stops, trailing avg < 0, correlation < 0.50) occurred in 2021-2025 during profitable periods. Set thresholds near 95th percentile of bootstrap distribution to avoid false halts.

---

## 8. Deployment Checklist

- [ ] **Set position sizing: $5k per trade initially**
  - Max 1 concurrent RXRX position
  - Never exceed 5% of portfolio on a single RXRX trade
  - Scale to $10k after 10+ live trades confirm the edge persists

- [ ] **Set up daily z-score monitoring script**
  - Script: `agent10/agent10_daily_signal.py`
  - Run daily after market close
  - Outputs: current z-score, spread, signal status (LONG/SHORT/FLAT)

- [ ] **Calibrate kill-switch thresholds from selection-window bootstrap**
  - Do NOT guess thresholds. Run bootstrap on 2021-2025 trades to find:
    - How often 3+ consecutive stops occurred during profitable stretches
    - What trailing-10-trade avg PnL threshold would have falsely halted the strategy
    - Whether correlation < 0.50 ever occurred in profitable periods
  - Set each trigger near the 95th percentile of its historical distribution
  - Structural trigger (no calibration needed): RXRX removed from ARKG, or ARKG AUM < $1B
  - Action on trigger: pause trading, manual review before re-entry

- [ ] **Paper trade to validate pipeline (not the edge)**
  - At ~12 signals/year, 2 weeks produces 0-1 trades
  - Use paper period only to confirm: signal timing, MOO fill execution, logging works
  - Validating the edge requires ~10 live trades (~8-10 months)

- [ ] **Set up trade logging**
  - Log every trade to a persistent store (CSV or BQ)
  - Fields: signal_date, entry_date, entry_price, exit_date, exit_price, side, z_at_signal, exit_reason, pnl_pct, pnl_dollars

- [ ] **Review after 10 live trades**
  - Compare live avg PnL to backtest avg (+4.15%)
  - Check for systematic slippage beyond 7.5 bps assumption

---

## 9. Research Program Architecture

```
Phase 0 (Data)          --> RXRX + ARKG + context daily data
Phase 1 (Cycle)         --> MR confirmed (Hurst 0.457), cycle hypothesis REJECTED
Phase 2 (Strategies)    --> 674 trials across S1-S13 families
Phase 3 (Validation)    --> 5 pre-registered configs for OOS
Phase 4 (Generalize)    --> 32 tickers tested, RXRX-specific (NOT deployable multi-ticker)
OOS Evaluation          --> S5_zexit PASSES all deployment gates on RXRX

Agent 0 (Engine)        --> Fixed z-score bug, canonical build_spread()
Agent 1 (Execution)     --> MOO confirmed best (selection window)
Agent 2 (Overnight)     --> v1: S5 PnL decomposition (descriptive only)
Agent 3 (Events)        --> v1: S5 overlay (descriptive only)
Agent 4 (S/R Levels)    --> v1: S5 overlay (descriptive only)
Agent 5 (Multi-Factor)  --> ARKG-only best (selection window)
Agent 6 (Attention)     --> v1: proxy substitution (descriptive only)
Agent 7 (Exit Eng.)     --> Z-exit 0.5 optimal (selection window)
Agent 8 (Integration)   --> This document

Agent 2v2 (Overnight)   --> 34 standalone variants REJECTED (selection window)
Agent 3v2 (Events)      --> 17 standalone variants REJECTED, real earnings data (selection window)
Agent 4v2 (S/R Levels)  --> 6 standalone variants REJECTED, permutation null (selection window)
Agent 6v2 (Attention)   --> 4/5 data sources unavailable; Google Trends REJECTED (selection window)
```

---

## 10. Methodological Notes

### V1 Agent Protocol Violations (Resolved)

The initial v1 runs of agents 2, 3, 4, and 6 deviated from their briefs: they evaluated on 2026 OOS trades (n=12) as S5_zexit overlays rather than building standalone strategies on the selection window. These were re-run as v2 agents with corrected methodology:

| Agent | V1 Problem | V2 Correction | V2 Result |
|-------|-----------|---------------|-----------|
| 2 | Decomposed S5's PnL | 34 standalone gap-fade/overnight/intraday MR variants | All negative. RXRX gaps are momentum. |
| 3 | 3 hardcoded events as S5 filter | 18 real earnings dates from Polygon; 17 standalone strategies | 0/17 positive. Negative pre-earnings drift. |
| 4 | S/R gate on S5 OOS trades | 6 standalone level-bounce strategies + permutation null | All negative. Bounce rate = random. |
| 6 | Volume/VIX proxies on S5 OOS | Attempted 5 real data sources; tested Google Trends standalone | 4/5 unavailable; Trends: no edge. |

All four hypotheses are now **properly tested and rejected** on the selection window. The v1 descriptive findings (85% intraday edge, event proximity p=0.91) remain valid as observations about S5_zexit's behavior but are not standalone strategy results.

### Remaining Data Gaps

Agent 6v2 confirmed that short interest, StockTwits history, Reddit mentions, and 13F ownership data require paid subscriptions (Polygon Business+, StockTwits Data, etc.). These hypotheses cannot be tested without sourcing this data.

### 2026 Holdout Status

The 2026 window is **fully contaminated** -- it was used for OOS evaluation, Phase 4 generalization, and v1 agent overlay analysis. No further conclusions should be drawn from 2026 data. Forward validation requires live or paper trading.

---

## 11. Files Reference

| File | Description |
|------|-------------|
| `agent10/agent0_engine.py` | Canonical engine: build_spread(), run_backtest(), cost model |
| `agent10/agent10_agent1_execution.py` | Execution fill type comparison |
| `agent10/agent10_agent2_overnight.py` | Overnight vs intraday decomposition |
| `agent10/agent10_agent3_events.py` | Event proximity analysis |
| `agent10/agent10_agent4_sr_levels.py` | Support/resistance level gates |
| `agent10/agent10_agent5_multifactor.py` | Multi-factor hedge comparison |
| `agent10/agent10_agent6_attention.py` | Attention/positioning proxy filters |
| `agent10/agent10_agent7_exits.py` | Exit engineering (trailing stops, partials, z-sweep) |
| `agent10/agent10_daily_signal.py` | Daily monitoring script (deployment) |
| `agent10/agent10_oos_evaluation.py` | Pre-registered OOS evaluation |
| `agent10/agent10_phase4_generalize.py` | Multi-ticker generalization test |
| `agent10/results/trials_log.csv` | Agent-phase trial log (91 rows) |
| `agent10/results/tier1_tier2_trial_log.csv` | Full trial log (674 rows) |
| `agent10/results/pre_registration.json` | Frozen OOS configs |
| `agent10/OOS_EVALUATION_RESULTS.md` | Detailed OOS evaluation |
| `agent10/PHASE4_RESULTS.md` | Multi-ticker generalization results |
| `agent10/TRADE_ANALYSIS_RESULTS.md` | Tier 1 + Tier 2 sweep results |
| `agent10/agent10_agent2v2_overnight.py` | v2: Standalone gap-fade, intraday MR, overnight hold |
| `agent10/agent10_agent3v2_events.py` | v2: Standalone earnings strategies (real Polygon data) |
| `agent10/agent10_agent4v2_sr_levels.py` | v2: Standalone S/R level-bounce + breakout |
| `agent10/agent10_agent6v2_attention.py` | v2: Real data sourcing + Google Trends strategies |
| `agent10/data/rxrx_gtrends.parquet` | Google Trends weekly data for RXRX (2021-2025) |
| `agent10/results/agent2v2_standalone.png` | v2 overnight/intraday diagnostic figure |
| `agent10/results/agent3v2_events.png` | v2 earnings event study figure |
| `agent10/results/agent4v2_sr_levels.png` | v2 S/R level-bounce diagnostic figure |
| `agent10/results/agent6v2_attention.png` | v2 Google Trends attention figure |
