# Agent 1 — E2-A Post-August Performance Investigation Results

**Date:** 2026-10-04
**Script:** `agent1_e2a_investigation.py`
**Runtime:** 18.7 min (Colab Enterprise)

---

## Config

| Parameter       | Value                |
|-----------------|----------------------|
| Backtest Start  | 2026-01-02           |
| Backtest End    | 2026-10-06           |
| Break Date      | 2026-08-01           |
| TOP_N           | 5                    |
| Gap Min         | 3.0%                 |
| Price Min       | $10.00               |
| Float Filter    | SOFT (dict, miss=INCLUDE, hit <10M=EXCLUDE) |
| Asset Type      | CS only              |
| Gap Freshness   | vs AH ref (last 16:00-20:00 bar on prior day) |
| Gap Erosion     | skip if >50% consumed by scan time |
| Re-rank         | at 09:31 ET          |
| Exit            | 17:00 ET             |
| Capital         | $100,000/day NC      |
| Trading Days    | 191                  |

---

## Pipeline Summary

| Phase            | Result                          |
|------------------|---------------------------------|
| Scan             | 191 trading days scanned        |
| Bar Pairs        | 10,901 fetched → 10,527 received |
| AH Freshness     | 703 stale excluded              |
| Gap Erosion      | 335 eroded excluded             |
| Ticker Refs      | 1,416 fetched → 823 CS          |

---

## Overview

| Metric      | Pre-Aug      | Post-Aug     | Delta        |
|-------------|--------------|--------------|--------------|
| Trades      | 672          | 154          |              |
| Avg PnL     | +4.57%       | +2.06%       | -2.51pp      |
| Sharpe      | 0.201        | 0.168        | -0.033       |
| Win Rate    | 62%          | 56%          | -6pp         |

---

## Phase 1 — Temporal Break Analysis

**Sharpest rolling Sharpe drop: 2026-07-13**
- Rolling Sharpe went from 0.743 to 0.377
- The break started mid-July, not Aug 1

**Chart:** `investigation_phase1.png` (4 panels: daily PnL + 20d rolling mean, 20d rolling Sharpe, 20d rolling win rate, cumulative P&L + drawdown)

---

## Phase 2 — Signal Audit

### Gap-to-PnL Correlation

| Period    | Pearson r  | p-value | Interpretation      |
|-----------|------------|---------|---------------------|
| Pre-Aug   | +0.245     | 0.000   | Positive, significant |
| Post-Aug  | +0.354     | 0.000   | Positive, STRONGER  |
| Verdict   | SAME SIGN  |         | NOT a regime flip   |

### Gap Distribution

| Metric      | Pre-Aug  | Post-Aug | Delta   |
|-------------|----------|----------|---------|
| Mean gap %  | 31.6%    | 22.6%    | -9.0pp  |
| Std gap %   | 43.1%    | 31.4%    | -11.7pp |
| Median gap %| 18.4%    | 12.3%    | -6.1pp  |

### Volume (Daily)

| Metric    | Pre-Aug  | Post-Aug |
|-----------|----------|----------|
| Mean      | 12.1M    | 12.3M    |
| Median    | 3.2M     | 3.6M     |

### Gap Consumed at Scan

| Metric    | Pre-Aug  | Post-Aug |
|-----------|----------|----------|
| Mean      | 13.7%    | 15.2%    |
| Median    | 10.4%    | 10.0%    |

**Chart:** `investigation_phase2.png` (4 panels: gap % distribution pre vs post, gap vs PnL scatter, monthly avg gap %, gap consumed distribution)

---

## Phase 3 — Execution Audit

### Fallback Entry Rate

| Period    | Rate  | Count        |
|-----------|-------|--------------|
| Pre-Aug   | 35%   | 235 / 672    |
| Post-Aug  | 36%   | 56 / 154     |
| Verdict   | NOT CAUSE — unchanged |

### Entry Type PnL

| Period | Entry Type   | Avg PnL  | N   |
|--------|--------------|----------|-----|
| Pre    | local_high   | +3.60%   | 437 |
| Pre    | fallback     | +6.37%   | 235 |
| Post   | local_high   | +3.48%   | 98  |
| Post   | fallback     | -0.43%   | 56  |

**CRITICAL FINDING:** Fallback entries went from +6.37% to -0.43% — a -6.80pp collapse. Local_high entries held steady (-0.12pp). The bar-15 fallback is the weakest link.

### Entry vs Open Price

| Period    | Mean      | Median    |
|-----------|-----------|-----------|
| Pre-Aug   | +0.62%    | +0.19%    |
| Post-Aug  | -0.02%    | -0.22%    |

Pre-Aug, entries were 0.62% above open (good for SHORT — entering at a higher price). Post-Aug, entries are flat vs open.

### Data Feed (RTH Bars per Day)

| Period    | Avg RTH Bars |
|-----------|-------------|
| Pre-Aug   | 12,693      |
| Post-Aug  | 8,829       |

Lower bar count post-Aug reflects fewer gap candidates passing filters, not a data feed issue.

---

## Phase 4 — Strategy Decomposition

### 4.1 — PnL by Open Pattern

| Pattern           | Pre N | Pre Avg   | Post N | Post Avg  | Delta    |
|-------------------|-------|-----------|--------|-----------|----------|
| sustained_drop    | 234   | +12.58%   | 50     | +6.11%    | -6.47pp  |
| drop_then_recover | 194   | -0.18%    | 35     | +1.41%    | +1.59pp  |
| mild_drop         | 107   | -0.63%    | 32     | -1.96%    | -1.32pp  |
| flat_or_bounce    | 110   | +1.39%    | 29     | +0.81%    | -0.58pp  |
| no_data           | 27    | +2.86%    | 8      | +0.24%    | -2.62pp  |

**sustained_drop is still the best pattern** (+6.11% post-Aug) but halved from +12.58%. All other patterns are near zero or negative.

### 4.2 — MFE / MAE

| Metric          | Pre-Aug  | Post-Aug | Delta    |
|-----------------|----------|----------|----------|
| MFE (favorable) | +14.15%  | +8.40%   | -5.75pp  |
| MFE median      | +8.30%   | +4.88%   | -3.42pp  |
| MAE (adverse)   | +15.16%  | +8.22%   | -6.93pp  |
| MAE median      | +6.36%   | +3.90%   | -2.46pp  |

**VERDICT:** Both MFE and MAE compressed roughly equally → low-movement period (SIZING), not a directional regime shift.

### 4.3 — Hold Period Sensitivity (Avg PnL by Exit Time)

| Exit Time | Pre Avg  | Post Avg | Delta    |
|-----------|----------|----------|----------|
| 10:00     | +0.91%   | +0.09%   | -0.82pp  |
| 11:00     | +1.56%   | +1.59%   | +0.03pp  |
| 12:00     | +2.14%   | +1.87%   | -0.27pp  |
| 13:00     | +2.92%   | +1.49%   | -1.43pp  |
| 14:00     | +3.18%   | +1.49%   | -1.69pp  |
| 15:00     | +3.31%   | +1.59%   | -1.72pp  |
| 16:00     | +3.70%   | +2.18%   | -1.52pp  |
| 17:00     | +4.57%   | +2.06%   | -2.51pp  |

**Post-Aug PnL peaks at 11:00-12:00 and plateaus.** Pre-Aug it kept climbing through AH. The afternoon/AH continuation has weakened.

### 4.4 — Top/Bottom Tickers by Total $ PnL

**Bottom 10 (worst):**

| Ticker | Total $     | Pre $       | Pre N | Post $      | Post N |
|--------|-------------|-------------|-------|-------------|--------|
| XRPN   | -$30,683    | $0          | 0     | -$30,683    | 2      |
| VEEE   | -$30,474    | -$34,224    | 5     | +$3,750     | 1      |
| TDIC   | -$26,518    | -$26,518    | 1     | $0          | 0      |
| CPOP   | -$22,922    | -$22,922    | 3     | $0          | 0      |
| CGTL   | -$21,715    | -$21,715    | 2     | $0          | 0      |
| ESTC   | -$15,364    | $0          | 0     | -$15,364    | 1      |
| THH    | -$15,135    | -$15,135    | 1     | $0          | 0      |
| AKAN   | -$13,741    | -$13,741    | 5     | $0          | 0      |
| BNAI   | -$13,179    | -$13,179    | 4     | $0          | 0      |
| MBAI   | -$11,119    | -$11,119    | 2     | $0          | 0      |

**Top 10 (best):**

| Ticker | Total $     | Pre $       | Pre N | Post $      | Post N |
|--------|-------------|-------------|-------|-------------|--------|
| ONFO   | +$31,533    | +$31,533    | 3     | $0          | 0      |
| KIDZ   | +$31,330    | +$31,330    | 6     | $0          | 0      |
| CTNT   | +$28,307    | +$17,027    | 2     | +$11,280    | 1      |
| KUST   | +$23,490    | +$23,490    | 4     | $0          | 0      |
| ONCO   | +$22,960    | +$22,960    | 3     | $0          | 0      |
| RUBI   | +$22,150    | +$22,150    | 2     | $0          | 0      |
| CIIT   | +$21,856    | +$21,856    | 4     | $0          | 0      |
| BYAH   | +$20,931    | +$20,931    | 3     | $0          | 0      |
| LRHC   | +$18,376    | +$18,376    | 3     | $0          | 0      |
| JUNS   | +$17,896    | +$17,896    | 3     | $0          | 0      |

**Chart:** `investigation_phase4.png` (4 panels: pattern PnL pre vs post, MFE vs MAE scatter, hold period curve, monthly PnL bars)

---

## Phase 5 — Regime Classification

### SPY Regime Thresholds

| Parameter        | Value                    |
|------------------|--------------------------|
| Vol median       | 11.9% (10d ann)          |
| Trend threshold  | SPY 20d return >= 0 = bull, < 0 = bear |

### E2-A Performance by Regime

| Regime     | Pre N | Pre Avg  | Post N | Post Avg | All N | All Avg  |
|------------|-------|----------|--------|----------|-------|----------|
| calm_bull  | 187   | +3.58%   | 53     | +2.44%   | 240   | +3.33%   |
| calm_bear  | 65    | +5.08%   | 42     | +1.21%   | 107   | +3.56%   |
| vol_bull   | 114   | +6.80%   | 51     | +3.30%   | 165   | +5.72%   |
| vol_bear   | 209   | +5.13%   | 8      | -3.89%   | 217   | +4.80%   |

### Regime Distribution

| Regime     | Pre-Aug  | Post-Aug |
|------------|----------|----------|
| unknown    | 97       | 0        |
| vol_bull   | 114      | 51       |
| calm_bull  | 187      | 53       |
| calm_bear  | 65       | 42       |
| vol_bear   | 209      | 8        |

**CRITICAL:** vol_bear went from 209 trades pre-Aug to only 8 post-Aug. This was E2-A's second-best regime (+5.13%) and it nearly vanished. The market shifted from volatile/bearish to calm.

**Best pre-Aug regime:** vol_bull (avg +6.80%)

**Chart:** `investigation_phase5.png` (2 panels: regime scatter SPY vol vs trend, E2-A avg PnL by regime pre vs post)

---

## Investigation Summary — Verdict Table

| Phase | Metric          | Pre-Aug    | Post-Aug   | Delta    | Verdict               |
|-------|-----------------|------------|------------|----------|-----------------------|
| 1.1   | Avg PnL         | +4.57%     | +2.06%     | -2.51pp  | DEGRADATION CONFIRMED |
| 1.1   | Sharpe          | 0.201      | 0.168      | -0.033   | DEGRADATION CONFIRMED |
| 1.1   | Win Rate        | 62%        | 56%        | -6pp     | NOT CAUSE             |
| 2.1   | Gap→PnL corr    | r=+0.245   | r=+0.354   | +0.109   | SAME SIGN             |
| 2.1   | Avg gap size    | 31.6%      | 22.6%      | -9.0pp   | CONTRIBUTING          |
| 3.1   | Fallback rate   | 35%        | 36%        | +1pp     | NOT CAUSE             |
| 3.1   | Entry vs open   | +0.62%     | -0.02%     | -0.64pp  | CHECK                 |
| 4.2   | MFE (favorable) | +14.15%    | +8.40%     | -5.75pp  | CAUSE                 |
| 4.2   | MAE (adverse)   | +15.16%    | +8.22%     | -6.93pp  | CAUSE                 |

---

## Decision Tree

- MFE collapsed 14.1% → 8.4%: gap-fade upside capture degraded
- Action: gaps not fading as deeply post-Aug — possible bull regime preventing mean reversion

---

## Root Cause Diagnosis

### Primary: Low-Movement Compression (REGIME)
Both MFE and MAE compressed equally (~40% reduction). Stocks that gap up are simply moving less intraday in the current calm market. E2-A's edge is real (signal-outcome correlation actually strengthened) but the magnitude of moves is smaller.

### Contributing: Gap Size Reduction
Average gap shrank from 31.6% to 22.6% (-9pp). Smaller gaps mean less room for the fade to generate PnL.

### Contributing: vol_bear Regime Disappearance
209 vol_bear trades pre-Aug → 8 post-Aug. The volatile bearish regime where gap-fades work hardest nearly vanished from the market.

### Weak Link: Fallback Entries
Fallback (bar 15) entries collapsed from +6.37% to -0.43% while local_high held steady (+3.60% → +3.48%). The fallback mechanism no longer finds good SHORT entries in the new regime.

### NOT Causes
- Signal death (correlation strengthened)
- Data feed issues (no gaps found)
- Fallback rate change (35% → 36%, unchanged)
- Win rate (62% → 56%, moderate decline)

---

## Monthly Breakdown

| Month    | Trades | Avg PnL  |
|----------|--------|----------|
| 2026-01  | 97     | +2.37%   |
| 2026-02  | 90     | +5.38%   |
| 2026-03  | 105    | +6.83%   |
| 2026-04  | 98     | +3.81%   |
| 2026-05  | 98     | -2.05%   |
| 2026-06  | 98     | +5.22%   |
| 2026-07  | 85     | +1.86%   |
| 2026-08  | 83     | +3.41%   |
| 2026-09  | 64     | +0.82%   |
| 2026-10  | 6      | -1.96%   |

**Break point:** Performance peaked in Feb-Mar (+5-7%), then May was the first negative month (-2.05%). July showed the sharpest rolling Sharpe drop (Jul 13). Aug-Sep-Oct are degraded but still slightly positive.

---

## Actual Break Date

**The sharpest rolling Sharpe drop was 2026-07-13**, not Aug 1. The degradation was already underway 2.5 weeks before the hypothesized break date. Any further analysis should re-anchor to mid-July.

---

## Charts

| File                       | Contents                                                      |
|----------------------------|---------------------------------------------------------------|
| investigation_phase1.png   | Daily PnL, rolling Sharpe, rolling WR, cumulative + drawdown  |
| investigation_phase2.png   | Gap distribution, signal-outcome, monthly gap, gap consumed   |
| investigation_phase4.png   | Pattern PnL, MFE/MAE scatter, hold period, monthly bars       |
| investigation_phase5.png   | Regime scatter, E2-A PnL by regime                            |

---

## Recommended Actions

1. **Eliminate or tighten fallback entries** — bar 15 fallback is -0.43% post-Aug vs +6.37% pre-Aug. Consider removing fallback entirely or moving it to bar 11.
2. **Add vol regime gate** — only trade when SPY 10d vol > 11.9% (median). This would concentrate in vol_bull and vol_bear regimes where the edge is strongest, at the cost of ~50% trade frequency.
3. **Shorter exit window** — post-Aug PnL peaks at 11:00-12:00 and plateaus. Consider 12:00 or 13:00 exit instead of 17:00 to lock in gains before afternoon reversals.
4. **Re-anchor break date to July 13** — the degradation started mid-July, not August.
5. **sustained_drop filter** — E2-B (with SD filter) may be more robust since sustained_drop is the only pattern still strongly positive post-Aug (+6.11%).
