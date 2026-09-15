# Scanner V4 Root Cause Analysis — Sep 10-11, 2026

**Scope:** 169 directional trades from premarket_scanner + discovery_scanner (V3.5-V4.0)
**Dates:** 2026-09-10 (58 trades), 2026-09-11 (111 trades)
**Methodology:** 2.5% SL checked bar-by-bar, equal-weight allocation, per-signal entry times, alpha vs SPY

---

## Executive Summary

The V4 scanner system is **net negative**: -0.16% avg at 30m, -0.19% at 60m across 169 trades. Selected trades (-0.22% avg) underperform unselected (-0.18%) and even gated trades (-0.03%). The selection mechanism is worse than random.

However, there is a **profitable core** hidden inside the noise. Filtering to only SHORT + offering/regulatory/squeeze catalysts would flip total PnL from **-24.9% to +23.9%** across 61 remaining trades.

### The 5 Root Causes (ranked by $ impact)

| # | Root Cause | Impact | Fix Complexity |
|---|-----------|--------|----------------|
| 1 | **Toxic catalyst types** (analyst/earnings/deal/unknown) | -47.3% total PnL drag | Low: filter rules |
| 2 | **LONG signals have negative alpha** (-0.39% vs SHORT +0.43%) | -29.3% from LONG trades | Low: direction weighting |
| 3 | **NEUTRAL bottleneck** blocks 47% of in-universe movers | 44 missed big movers | High: LLM prompt tuning |
| 4 | **Cross-scanner dedup missing** (APUR selected LONG+SHORT same day) | Active contradictions | Medium: dedup logic |
| 5 | **V4 EV metrics not predictive** (r=+0.07 with PnL) | No improvement over V3 | High: rethink residual math |

---

## 1. False Positive Analysis — What Selected Trades Lose Money

**Source:** 36 selected trades, 24 losers (67% FP rate), 12 winners (33%)

### FP Categories (corrected, by primary root cause)

| Category | Count | Avg Loss | Total Impact | Key Tickers |
|----------|-------|----------|--------------|-------------|
| EXHAUSTED_MOVE | 11 | -0.73% | -8.04% | COIN, EXEL, KR, OPTX, XOM, TEN, INTC, TSLA, SCCO, RDDT, HBM |
| WRONG_DIRECTION | 6 | -1.73% | -10.38% | RXT, AENT, ETN, COIN, VST, CORZ |
| BAD_CATALYST | 2 | -1.49% | -2.97% | CVX (stale oil news), MRNA (premarket_move) |
| DIR_REMAINING_CONTRADICTION | 2 | -1.78% | -3.57% | HOOD, ONDS |
| GAP_AND_CRAP | 1 | -2.50% | -2.50% | RH (earnings gap already priced) |
| NON_EVENT | 1 | -0.49% | -0.49% | TSLA (NHTSA audit immaterial for $1.2T cap) |
| NOISE | 2 | -0.39% | -0.77% | APA, IP (within normal variance) |

### Critical Findings

1. **EXHAUSTED_MOVE is #1 by count (11/24)**: The `move_is_exhausted` flag fires on 72% of selected trades but is NEVER checked in `_is_selected()`. The flag itself has weak predictiveness (exhausted -0.19% vs not-exhausted -0.14%), suggesting it needs recalibration before use.

2. **WRONG_DIRECTION is #1 by severity (-1.73% avg)**: Scanner follows premarket price direction over catalyst direction. On bullish catalysts (deals, upgrades, approvals), this consistently fails. AENT was shorted despite +31.9% gap on earnings beat — stopped out at minute 0.

3. **Direction-remaining contradiction is a strong negative signal**: -0.89% avg vs -0.01% for non-contradictions. The V4 `dir_residual_conflict` penalty exists but is too weak (50% EV penalty) for extreme-EV signals like ACVA (EV=22.43) and AENT (EV=17.70).

4. **Repeat tickers degrade**: MRNA, TSLA, XOM selected both days — all performed worse on Sep 11 with staler catalysts.

5. **Confidence is anti-predictive**: Conf=2 avg +0.33%, conf=5 avg -0.72%. r = -0.076 overall.

---

## 2. False Negative Analysis — What the Scanner Misses

**Source:** 906 stocks moved >5% across Sep 10-11

### Coverage Funnel

| Stage | Count | % of Total |
|-------|-------|------------|
| Total >5% movers | 906 | 100% |
| In scanner universe | 93 | 10.3% |
| Directional (not NEUTRAL) | 49 | 5.4% |
| Correct direction | 30 | 3.3% |
| Not gated | 24 | 2.6% |
| **Selected (CAUGHT)** | **7** | **0.8%** |

### The 3 Alpha Leaks

**Leak 1: NEUTRAL bottleneck (47% of in-universe, 44 tickers)**
Scanner saw 44 big movers but classified them all NEUTRAL — couldn't determine direction. These include:
- TNON +117% (scanned 3x, always NEUTRAL — then +69% next day, still NEUTRAL with Rank=0)
- AEO -14% (earnings, 4 scans, always NEUTRAL)
- VRT -5.6% (9 scans, always NEUTRAL)
- WOLF -7.7% (8 scans, always NEUTRAL)

The LLM defaults to NEUTRAL when it can't find strong directional evidence, even when the stock is already making the move. This is a fundamental coverage cap.

**Leak 2: Wrong direction (20% of in-universe, 19 tickers)**
Scanner picked the opposite direction from the actual move. Worst cases on Sep 11:
- ACVA +44.4%: acquisition target, scanner called SHORT (should be LONG on acquiree)
- COLA +26.9%: deal catalyst, scanner called SHORT
- AENT +26.6%: earnings beat +31.9% gap, scanner called SHORT
- ORCL -5.4%: earnings beat, scanner called LONG conf=7 (highest confidence = worst outcome)

**Leak 3: Not selected (18% of in-universe, 17 tickers)**
Correct direction, no gate, but Rank=0, Conf=0. These went through Stage 2 analysis but got zero confidence:
- HPE +9.4% (earnings beat, raised outlook — Rank=0)
- SMR -13.1% (UBS downgrade to Sell — Rank=0)
- COO -14.7% (securities fraud investigation — Rank=0)

### Sep 11 Was Particularly Bad
- Only 1 of 41 in-universe movers caught (OKLO)
- 12 wrong-direction calls (most calling SHORT on LONG movers)
- The discovery scanner's Stage 1 (Gemini) appears to default to SHORT when uncertain

---

## 3. Gate & Selection Quality Audit

### Selection Is Worse Than Random

| Category | N | Avg PnL@60m | WR@60m |
|----------|---|-------------|--------|
| **SELECTED** | **28** | **-0.32%** | **32%** |
| GATED | 48 | -0.03% | 52% |
| NOT_SELECTED | 93 | -0.17% | 42% |

Gates cost 0.29%/trade — they block signals that are no worse (and often better) than selected signals.

### Gate Reason Scorecard

**Good gates (correctly blocking losers):**
| Gate | N Blocked | Avg PnL if Traded | Verdict |
|------|-----------|-------------------|---------|
| self_contradiction: no remaining edge | 2 | -1.51% | KEEP |
| PRE_EARNINGS | 2 | -1.34% | KEEP |
| LOW_CONF (3) | 9 | -0.31% | KEEP |

**Bad gates (blocking winners):**
| Gate | N Blocked | Avg PnL if Traded | Verdict |
|------|-----------|-------------------|---------|
| premarket_direction_mismatch (various) | 3 | +1.26% avg | REMOVE |
| LOW_LIQUIDITY | 2 | +2.13% avg | REMOVE |
| self_contradiction: immaterial | 3 | +0.40% | REMOVE |
| EARNINGS_GAP_DOWN | 1 | +1.06% | REMOVE |
| PENNY_STOCK | 1 | +1.24% | REMOVE |

**Total opportunity cost from bad gates**: +19.12% across 17 trades blocked.

### V4 Metrics — None Are Predictive

| Metric | Correlation with PnL@60m | N | Assessment |
|--------|--------------------------|---|------------|
| EV | r = +0.070 | 25 | No signal |
| Residual Mid | r = -0.038 | 25 | No signal |
| Confidence | r = -0.076 | 73 | Weak anti-signal |
| Informed Score | r = +0.013 | 25 | No signal |

- EV shows no monotonic relationship: EV 0.25-0.5 has 83% WR but EV 2.0-5.0 has 0% WR
- Residual direction is NOT more accurate than LLM direction (both ~56%)
- Tape discount fired only once — no data
- The ONE good V4 innovation: `dir_residual_conflict` penalty (penalized avg -1.01%) — but 50% penalty is insufficient

---

## 4. Catalyst Pattern Analysis

### Golden Signals (WR >= 55%, avg PnL > 0, n >= 3)

| Profile | n | Avg PnL | WR | Stop Rate |
|---------|---|---------|-----|-----------|
| deal + SHORT + discovery + open_5min | 3 | +1.81% | 100% | 0% |
| squeeze + LONG + discovery | 5 | +1.48% | 60% | 40% |
| regulatory + SHORT + discovery + mid_morning | 4 | +1.15% | 75% | 0% |
| regulatory + SHORT + discovery + open_5min | 12 | +0.43% | 58% | 25% |
| analyst + LONG + discovery + mid_morning | 6 | +0.42% | 67% | 0% |

### Toxic Signals (WR < 40%, avg PnL < -0.3%, n >= 3)

| Profile | n | Avg PnL | WR | Total PnL |
|---------|---|---------|-----|-----------|
| analyst + SHORT + premarket + open_5min | 5 | -2.35% | 0% | -11.8% |
| earnings + LONG (any) | 11 | -1.14% | 18% | -12.6% |
| deal + LONG (any) | 7 | -1.11% | 14% | -7.8% |
| analyst + SHORT (any) | 10 | -1.09% | 30% | -10.9% |
| unknown + SHORT + discovery + open_5min | 10 | -0.99% | 20% | -9.9% |

### Catalyst Direction Edge

| Catalyst | LONG PnL | SHORT PnL | Preferred Direction |
|----------|----------|-----------|---------------------|
| offering | -2.50% | +0.92% | SHORT only |
| deal | -1.11% | -0.10% | SHORT only |
| earnings | -1.14% | +0.14% | SHORT only |
| analyst | +0.12% | -1.09% | LONG only |
| regulatory | +0.10% | +0.40% | SHORT preferred |

### Keyword Signals in Catalyst Text

| Keyword | n | Avg PnL | WR | Signal |
|---------|---|---------|-----|--------|
| "offering" | 14 | +1.08% | 50% | BULLISH |
| "FDA" | 13 | +0.41% | 54% | BULLISH |
| "initiated" | 5 | -1.52% | 20% | BEARISH |
| "short_squeeze" | 6 | -1.13% | 33% | BEARISH |
| "gap_up" | 13 | -0.64% | 31% | BEARISH |
| "beat" | 11 | -0.51% | 36% | BEARISH |

---

## 5. Cross-Scanner Dedup — P0 Bug

### The Problem
Same ticker selected as BOTH LONG and SHORT on the same day. These are active contradictions, not theoretical.

**Sep 10: 18 tickers with directional conflicts, including:**
- RXT: selected LONG (premarket) AND SHORT (discovery) simultaneously. 3 selections total.
- ORCL: selected LONG conf=7 at 17:42, was SHORT conf=4 at 09:13
- TSLA: SHORT at 07:47, LONG at 09:13, SHORT at 10:03

**Sep 11: 9 tickers with directional conflicts, including:**
- APUR: selected LONG at 06:09 AND SHORT at 07:42 (opposite directions, 93 minutes apart)
- TSLA: selected SHORT 3x, then LONG 1x
- OKLO: selected LONG (discovery) AND SHORT (premarket)
- ACVA: selected SHORT 2x, then LONG appeared 3x later

### Impact
Holding both LONG and SHORT on the same ticker zeroes out. You pay spread and commissions twice for no net exposure. This is the #1 bug to fix — it's a logic error, not a judgment call.

---

## 6. V4 vs V3 Comparison

| Metric | V3.x (Sep 10) | V4.0 (Sep 11) | Delta |
|--------|---------------|---------------|-------|
| Selected avg PnL@60m | -0.19% | -0.44% | -0.25% worse |
| Selected WR | 21% | 43% | +22pp better |
| SL Hit Rate | 21% | 36% | +15pp worse |
| Contradictions | 14 (6 selected) | 6 (3 selected) | Improved |

V4's EV-based selection produces a higher win rate but also more stop-outs — the high-EV signals have extreme gaps (ACVA +44.6%, AENT +33.4%) that lead to violent reversals. The residual math amplifies extreme gaps into extreme EVs, but extreme gaps are where the move is already done.

---

## 7. Consolidated Fix Roadmap

### Ship Immediately (P0)

| Fix | Description | Expected Impact | Effort |
|-----|-------------|-----------------|--------|
| **Cross-scanner dedup** | Per ticker per day: keep only highest-EV direction, block opposing | Eliminates zero-exposure trades | 1 day |
| **Direction-remaining hard block** | If direction opposes remaining_move sign, block (don't just penalize) | +6.21% over 2 days from blocking 5 losers | 2 hours |
| **Toxic catalyst filter** | Gate: analyst+SHORT, earnings+LONG, deal+LONG, unknown, bankruptcy | Flips avg PnL from -0.15% to +0.34% | 2 hours |

### Ship This Week (P1)

| Fix | Description | Expected Impact | Effort |
|-----|-------------|-----------------|--------|
| **SHORT preference** | Weight SHORT allocation 2:1 over LONG | SHORT alpha +0.43% vs LONG -0.39% | 1 hour |
| **Remove bad gates** | Drop: premarket_direction_mismatch, LOW_LIQUIDITY, PENNY_STOCK, EARNINGS_GAP_DOWN | Recovers +19.12% in blocked winners | 2 hours |
| **Mega-cap materiality filter** | >$100B market cap: require expected_move >= 3% or catalyst in [earnings, offering, M&A] | Would block TSLA, CVX, XOM non-events | 4 hours |
| **Repeat-ticker suppression** | If ticker selected yesterday with same catalyst, require new article or higher remaining_move | Blocks stale MRNA/XOM/TSLA Day 2 trades | 4 hours |

### Needs Investigation (P2)

| Fix | Description | Expected Impact | Effort |
|-----|-------------|-----------------|--------|
| **NEUTRAL reduction** | Tune LLM prompt to be less conservative on directional calls | 44 missed big movers stuck NEUTRAL | Multi-day |
| **Exhaustion flag recalibration** | Current flag fires 72% of the time — useless. Recalibrate threshold or tie to gap size | Could improve if flag becomes discriminating | 1-2 days |
| **EV formula rethink** | Current residual math: r = +0.07 with PnL. Consider: (a) cap EV at 5, (b) use gap-adjusted residual, (c) try alternative scoring | V4 EV adds complexity without improving outcomes | Multi-day |
| **Acquisition target detection** | ACVA +44% shorted because scanner couldn't distinguish acquirer from acquiree | Would prevent worst wrong-direction errors on M&A | 1 day |
| **Gap-and-crap detector** | When |gap| > |expected_move × 0.5|, flag as HIGH_RISK | Would catch RXT, AENT, RH | 4 hours |

---

## 8. Simulation: What If We Applied P0+P1 Fixes?

Starting from the 169 trades:

1. **Cross-scanner dedup**: Remove contradictory same-day same-ticker trades (~12 trades)
2. **Direction-remaining block**: Remove 7 trades with contradiction (saves +6.21%)
3. **Toxic catalyst filter**: Remove analyst+SHORT, earnings+LONG, deal+LONG, unknown, bankruptcy (~108 trades)
4. **Remove bad gates**: Re-admit ~17 wrongly-gated trades (+19.12%)
5. **SHORT preference**: 2:1 allocation weight

**Estimated remaining trades**: ~65-70
**Estimated avg PnL**: +0.35% to +0.45% (from -0.15%)
**Estimated WR**: ~49% (from 41%)

The scanner has a profitable signal buried under noise. The P0 fixes are essentially "stop hurting yourself" — removing self-contradictions, toxic catalysts, and wrong-direction trades that the scanner's own data already flags.

---

## 9. Reference: Per-Agent Output Files

| Agent | Output | Lines | Key Contribution |
|-------|--------|-------|------------------|
| FP Analysis | `/tmp/scanner_fp_analysis.md` | 998 | Root-caused all 24 FPs with corrected categories |
| FN Analysis | `/tmp/scanner_fn_analysis.md` | 408 | Full coverage funnel, 93 in-universe misses detailed |
| Gate Audit | `/tmp/scanner_gate_audit.md` | 557 | Gate scorecard, V4 metric correlations, P0 bug status |
| Catalyst Patterns | `/tmp/scanner_catalyst_patterns.md` | 671 | Golden/toxic profiles, combined filter simulation |

---

*Generated: 2026-09-11. Data: premarket_scanner + discovery_scanner BQ logs, Polygon 1-min bars.*
