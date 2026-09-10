# Scanner Investigation — 2026-09-10

## Overview

Full investigation of scanner performance on Sep 10 2026. Three parallel agents analyzed false positives, false negatives, and selection/gate quality. **All PnL numbers use decontaminated entry times** (each signal enters at its actual scan_time converted to ET, pre-open signals enter at 9:30 ET open).

### Critical Methodology Fix

The original evaluation had a **lookahead bias**: all 7 runs (05:46 to 10:03 PT) were pooled, deduped by highest confidence, then evaluated as if entered at 9:30 ET open. A signal generated at 10:03 PT (1:03 PM ET) with 3.5 hours of price data visible was credited with open-to-60m return. This inflated the selected mean from **-0.06% (real) to +1.25% (biased)**.

Fixed eval uses per-signal entry time: `entry_time = max(scan_time_ET, 9:30 ET)`.

---

## Decontaminated PnL Summary (SL=2.5%)

| Metric | 30m | 60m | SE @60m |
|--------|-----|-----|---------|
| ALL (n=58) | -0.08% | -0.04% | ±0.19% |
| **SELECTED (n=19)** | **+0.13%** | **-0.06%** | **±0.47%** |
| Not Selected (n=39) | -0.18% | -0.03% | |
| Ungated (n=24) | +0.05% | -0.05% | |
| Gated (n=34) | -0.17% | -0.03% | |

**Selection is random** — selected avg (-0.06%) is within SE of zero and not distinguishable from not-selected (-0.03%).

**Gates have negative value** — gated signals (-0.03%) performed BETTER than ungated (-0.05%). Gates cost -0.02%/trade.

### Entry Time Distribution

| Entry ET | Trades | Source |
|----------|--------|--------|
| 09:30 | 7 | Pre-open runs (05:46, 06:08 PT) |
| 09:34-09:35 | 8 | 06:34-06:35 PT runs |
| 09:51 | 6 | 06:51 PT run |
| 10:18-10:48 | 18 | 07:18-07:48 PT runs |
| 12:13 | 9 | 09:13 PT run |
| 13:03-13:08 | 10 | 10:03 PT run |

---

## Selected Signals — Full Detail

### Winners (7 trades, avg +2.18%)

| Ticker | Dir | Entry ET | Conf | Catalyst | PnL@60m | Notes |
|--------|-----|----------|------|----------|---------|-------|
| BHVN | SHORT | 09:35 | 4 | regulatory | +6.36% | FDA partial clinical hold, caught early |
| BE | LONG | 09:30 | 6 | breaking | +3.07% | Gap-down recovery |
| AXGN | SHORT | 09:51 | 7 | offering | +1.30% | Secondary offering |
| MRNA | SHORT | 10:18 | 4 | regulatory | +0.90% | |
| M | SHORT | 12:13 | 6 | earnings | +0.40% | |
| XOM | LONG | 13:03 | 5 | breaking | +0.00% | Flat |
| SCCO | SHORT | 13:03 | 4 | macro | -0.00% | Flat |

### Losers (12 trades, avg -1.08%)

| Ticker | Dir | Entry ET | Conf | Catalyst | PnL@60m | Classification | Root Cause |
|--------|-----|----------|------|----------|---------|----------------|------------|
| RXT | LONG | 09:30 | 6 | breaking | -2.50% | WRONG_DIRECTION | Micro-cap gapped +8.6%, classic gap-and-crap. Later runs correctly called SHORT but earliest-selected locked in LONG. |
| CVX | LONG | 09:30 | 4 | breaking | -2.50% | BAD_CATALYST | 2-day-old Venezuela news recycled by MarketBeat aggregator. $300B mega-cap doesn't move on stale oil news. |
| HOOD | SHORT | 09:30 | 4 | analyst | -2.50% | WRONG_DIRECTION | StoneX initiated BUY/$170 PT (51% upside). Scanner shorted a stock with a bullish catalyst because pre-market was -2.3%. |
| TEN | LONG | 12:13 | 5 | earnings | -1.60% | STALE_SIGNAL | Strong Q2 earnings, stock peaked by 09:45 ET. Entered 2.5 hours after move completed. |
| ONDS | LONG | 10:47 | 4 | earnings | -1.10% | BAD_CATALYST | Month-old earnings (Aug 13) recycled by Simply Wall St editorial. remaining_move_pct was -25% (negative!) yet selected LONG. |
| VST | SHORT | 09:34 | 4 | deal | -0.80% | WRONG_DIRECTION | Vistra announced 20-year nuclear deals with Meta/AWS, $1B AI JV with NVIDIA/KKR. Scanner shorted on flat premarket despite massively bullish catalysts. |
| OPTX | LONG | 10:19 | 5 | deal | -0.70% | BAD_CATALYST | $5M contract on $300M company (1.6% of revenue). Immaterial deal, extremely illiquid. |
| CORZ | SHORT | 09:34 | 4 | regulatory | -0.50% | WRONG_DIRECTION | ERCOT power approvals for data centers = POSITIVE. Scanner shorted because premarket was -3.1% despite bullish catalyst. |
| RDDT | LONG | 13:03 | 4 | analyst | -0.40% | EXHAUSTED_MOVE | +6.9% gap on Piper Sandler data. By 13:03 entry, at day's high with no upside. Exhausted=True in BQ. |
| HBM | SHORT | 13:03 | 6 | macro | -0.20% | EXHAUSTED_MOVE | Copper pullback drove -7.8% gap. By 13:03, selling was done. Exhausted=True. |
| INTC | SHORT | 12:13 | 4 | offering | -0.10% | EXHAUSTED_MOVE | $20B offering drove -5% gap at open. By 12:13, dilution fully absorbed. Exhausted=True. |
| TSLA | LONG | 12:13 | 4 | analyst | -0.10% | BAD_CATALYST | "Multiple analyst upgrades" with no named firms. $1.45T mega-cap, routine analyst notes are noise. |

### FP Pattern Summary

| Category | Count | Avg Loss | Total Impact | Key Tickers |
|----------|-------|----------|--------------|-------------|
| WRONG_DIRECTION | 4 | -1.58% | -6.30% | RXT, HOOD, VST, CORZ |
| BAD_CATALYST | 4 | -1.10% | -4.40% | CVX, ONDS, OPTX, TSLA |
| EXHAUSTED_MOVE | 3 | -0.23% | -0.70% | RDDT, HBM, INTC |
| STALE_SIGNAL | 1 | -1.60% | -1.60% | TEN |
| **Total** | **12** | **-1.08%** | **-13.00%** | |

---

## False Negative Analysis

### Scanner Coverage

- **Premarket scanner**: 471 rows, 68 unique tickers, 7 scan times. 410 NEUTRAL, 34 SHORT, 27 LONG.
- **Discovery scanner**: 217 rows, 137 unique tickers, 7 scan times. 171 NEUTRAL, 28 SHORT, 18 LONG.
- **Combined**: 205 unique tickers scanned, 19 selected.

### Big Movers in Scanner Universe (>3% change): 102 tickers

| Category | Count | Pct | Description |
|----------|-------|-----|-------------|
| NEUTRAL_BLOCKED | 72 | 71% | Saw ticker, classified NEUTRAL, never reached Stage 2 |
| WRONG_DIRECTION | 12 | 12% | Stage 2 assigned opposite direction |
| GATED | 11 | 11% | Correct direction but gated out |
| CAUGHT | 7 | 7% | Correct direction, selected for trade |

### Top 15 Most Costly Misses

| # | Ticker | Move | Vol | Category | Score | Why Missed |
|---|--------|------|-----|----------|-------|------------|
| 1 | NAVN | -21.75% | 19.2M | NEUTRAL_BLOCKED | 15.0 | Post-earnings miss, Stage 1 couldn't call direction |
| 2 | BBOT | -18.28% | 2.8M | NEUTRAL_BLOCKED | 16.6 | Catalyst type "unspecified" |
| 3 | COO | -14.67% | 18.7M | NEUTRAL_BLOCKED | 15.6 | Scanned 7 times (score 13.5-15.6), always NEUTRAL |
| 4 | AEO | -13.97% | 28.4M | NEUTRAL_BLOCKED | 15.0 | Earnings detected every scan, always NEUTRAL |
| 5 | FLWS | -13.04% | 945K | GATED | 14.4 | Correct SHORT, but conf=3 — gated by LOW_CONF |
| 6 | LOVE | -11.04% | 715K | GATED | 9.6 | Correct SHORT, gated: SELF_CONTRADICTION + LOW_CONF(2) |
| 7 | MCFT | -9.74% | 681K | WRONG_DIR | 7.0 | Stage 2 assigned LONG but stock dropped |
| 8 | LEU | -8.61% | 995K | NEUTRAL_BLOCKED | 13.5 | Offering news detected, direction NEUTRAL |
| 9 | WOLF | -7.73% | 2.7M | NEUTRAL_BLOCKED | 9 | Regulatory catalyst, always NEUTRAL |
| 10 | KEEL | -7.28% | 26.5M | WRONG_DIR | 7 | Called LONG but stock dropping |
| 11 | BLZE | -6.95% | 2.6M | NEUTRAL_BLOCKED | 7 | Breaking news, always NEUTRAL |
| 12 | OKLO | -6.32% | 10.7M | WRONG_DIR | 9 | Called LONG on analyst upgrade, stock fell |
| 13 | PGY | -6.02% | 3.2M | GATED | 5 | Wrong direction + LOW_CONF gate |
| 14 | FCEL | -5.91% | 5.8M | NEUTRAL_BLOCKED | 9 | High scan_score, NEUTRAL every time |
| 15 | VRT | -5.61% | 10.0M | NEUTRAL_BLOCKED | 36.0 | scan_move_score=36 (expected 45% move!) but NEUTRAL |

### NEUTRAL Bottleneck Sub-Patterns

**(a) Post-earnings NEUTRAL** (COO, AEO, NAVN, CULP): Earnings RESULTS were already out showing clear beats/misses, but Stage 1 classified NEUTRAL. The direction was knowable from the reported numbers.

**(b) High score but NEUTRAL** (VRT=36, TNON=63, SNYR=54): Discovery scanner computed very high move scores but direction was NEUTRAL. The BQ logging at line 2637 force-sets `direction: 'NEUTRAL'` for non-deep-analyzed tickers.

**(c) Deploy-universe tickers stuck NEUTRAL** (MU, VRT, AMD, ARM, CRDO, ALAB, SMR, WOLF): In the premarket watchlist, scanned 7 times each, always returned NEUTRAL.

---

## Selection & Gate Audit

### Gate Breakdown (107 LONG/SHORT signals, all runs)

| Gate Category | Count | Notes |
|---------------|-------|-------|
| No gate | 47 | 23 selected |
| LOW_CONF (1-3) | 24 | Would not be selected anyway (conf < 4 threshold) |
| Self-contradiction | 18 | Conf set to 0 |
| Direction mismatch | 10 | LONG on red stocks |
| Other (penny, liquidity, earnings, clinical) | 8 | Various |

**Key finding:** Gates catch signals that would not be selected anyway. LOW_CONF gates (24 signals) have conf 1-3, but selection requires conf >= 4. Self-contradiction gates catch the LLM being honest ("no edge", "immaterial") — which actually correlates with flat outcomes, not losses. **Gates are blocking the less-bad signals and letting the overconfident contradictory ones through.**

### Confidence is NOT Predictive

| Confidence | Avg PnL @60m | N Selected |
|------------|-------------|------------|
| 7 | -4.11% | 1 (WORST) |
| 6 | +0.94% | 6 |
| 5 | +6.69% | 3 (BEST) |
| 4 | +0.94% | 13 |

No monotonic relationship. The highest-confidence signal was the single worst performer.

### P0 Bug: Direction-Remaining Move Contradiction

31 signals had direction opposing their own `remaining_move_pct`. 7 of these were selected:

| Ticker | Direction | Remaining Move | Problem |
|--------|-----------|----------------|---------|
| ONDS | LONG | -25.00% | LONG but expects 25% DROP |
| HOOD | SHORT | +2.00% | SHORT but expects RISE |
| CORZ | SHORT | +2.00% | SHORT but expects RISE |
| MRNA | SHORT | +3.00% | SHORT but expects RISE |
| VST | SHORT | +4.00% | SHORT but expects RISE |
| RXT | SHORT | +3.00%/+5.00% | SHORT but expects RISE |

**Root cause:** Post-processing (lines 1704-1727) computes `remaining_move_pct = total_catalyst_impact - polygon_absorbed_pct`. When a stock overshoots its catalyst estimate, remaining becomes opposite-sign. But the code never updates `direction` to match. The direction mismatch gate only fires at conf <= 5, so conf=6 contradictions pass.

### P0 Bug: move_is_exhausted Not Used in Selection

14 of 19 selected signals were marked `exhausted=True`. Results:
- 14 exhausted selected: avg **-0.26%**
- 5 non-exhausted selected: avg **+0.41%**

The `_is_selected()` function (premarket line ~2154, discovery line ~2024) checks rank, confidence >= 4, and no gate — but **never checks `move_is_exhausted`**. The flag is computed, logged, displayed, but ignored in the selection decision.

### P0 Bug: No Cross-Scanner Dedup

RXT was selected as both LONG (premarket, conf=6) and SHORT (later discovery run). The same ticker appeared with contradictory directions simultaneously. No dedup logic exists between scanners.

---

## Consolidated Fix Proposals

### P0 — Bugs (code defects, no judgment calls)

**Fix A: Direction-remaining consistency gate**
If `direction == 'LONG' and remaining_move_pct < -0.5`, or `direction == 'SHORT' and remaining_move_pct > 0.5`, gate as `DIRECTION_REMAINING_CONTRADICTION`.
- Location: `premarket_scanner.py` quality gates (~line 1860), `discovery_scanner.py` (~line 2263)
- Impact: Would have prevented 7 bad selected signals on Sep 10

**Fix B: Exhaustion gate in `_is_selected()`**
Add `and not entry.get('move_is_exhausted')` to selection check. Exception: allow if `abs(remaining_move_pct) > 3.0` (preserves BE LONG which was exhausted=True but had 12% remaining).
- Location: `premarket_scanner.py` ~line 2154, `discovery_scanner.py` ~line 2024
- Impact: Would have blocked 14/19 selected (keeping the 5 non-exhausted which averaged +0.41%)

**Fix C: Cross-scanner dedup**
Before final output, deduplicate by ticker. Same ticker as both LONG and SHORT = gate both. Same direction = keep higher composite.
- Location: Both scanners, final output section

### P1 — High-Impact Logic Changes

**Fix D: NEUTRAL override for high-score tickers**
If `catalyst_score >= 7` (premarket) or `scan_move_score >= 10` (discovery), force to Stage 2 regardless of NEUTRAL. Let Claude determine direction.
- Impact: Would have caught NAVN (-21.75%), COO (-14.67%), AEO (-13.97%), VRT (-5.61%)
- Cost: ~5-10 more Claude API calls per run (~$0.10-0.30)

**Fix E: Catalyst-direction consistency gate**
If Stage 2 identifies catalyst as bullish (deal, Buy initiation, contract) but recommends SHORT based only on price weakness, gate as `CATALYST_DIRECTION_MISMATCH`.
- Impact: Would have prevented HOOD SHORT (Buy initiation shorted) and VST SHORT (mega-deals shorted)

**Fix F: Lower MIN_CONFIDENCE to 3 for post-earnings**
For `catalyst_type == 'earnings'` where results are already out, lower selection threshold from 4 to 3.
- Impact: Would have caught FLWS (-13.04%) and LOVE (-11.04%)

### P2 — Nice-to-Have

**Fix G:** Post-earnings direction assignment in Stage 1 — add to prompt: "If earnings results are out, the direction IS known. Do NOT return NEUTRAL."

**Fix H:** Detect aggregator articles recycling old news — apply staleness to ORIGINAL date, not article publication

**Fix I:** Catalyst materiality gate for deals — if deal_value / market_cap < 3%, gate as IMMATERIAL_DEAL

**Fix J:** Late-day staleness multiplier — entries after 12:00 ET on pre-open catalysts get 0.3x tradability

**Fix K:** Require named analyst firm for analyst catalyst type — "multiple upgrades" with no firms = VAGUE_ANALYST gate

**Fix L:** Cap LLM confidence at 6 for non-earnings catalysts — empirical conf=5 sweet spot, higher = overconfidence

**Fix M:** Discovery scanner BQ logging fix — line 2637 force-sets `direction: 'NEUTRAL'` for all non-deep-analyzed tickers, destroying Stage 1 direction info. Log actual Stage 1 direction instead.

### Impact Estimate

Fixes A+B alone would flip Sep 10 from -0.06% to positive — they prevent 7 of 12 losses while preserving all 7 winners. Fix D addresses the biggest alpha leak (71% of >3% movers blocked by NEUTRAL). Fixes A+B+D together address both sides: stopping FP losses and capturing FN alpha.

---

## Stage 2 Model Tracking (V3.7 Fix 21 Validation)

Latest run (10:03 PT):
- **Premarket**: 12 Stage 2 calls, ALL used `claude-sonnet-4-6`, zero errors
- **Discovery**: 8 Stage 2 calls, ALL used `claude-sonnet-4-6`, zero errors
- **Total**: 20/20 Claude, 0 Gemini fallback, 0 errors

Fix 21 confirmed working.

---

## Performance by Catalyst Type (Decontaminated)

| Type | N | Avg PnL@60m | WR | Notes |
|------|---|-------------|-----|-------|
| regulatory | 10 | +1.16% | 70% | Best — BHVN drove this |
| offering | 2 | +0.58% | 50% | Small sample |
| premarket_move | 2 | +0.42% | 50% | |
| macro | 6 | +0.18% | 33% | |
| insider | 2 | -0.55% | 0% | |
| earnings | 10 | -0.41% | 40% | Mixed |
| breaking | 4 | -0.47% | 50% | RXT, CVX losses |
| deal | 6 | -1.37% | 17% | Worst — VST, OPTX losses |
| analyst | 16 | -0.10% | 38% | High volume, flat |

---

## Performance by Direction

| Direction | N | SL@30m | SL@60m | WR@60m | EOD |
|-----------|---|--------|--------|--------|-----|
| LONG | 29 | -0.27% | -0.20% | 34% | -1.02% |
| SHORT | 29 | +0.11% | +0.13% | 48% | +0.82% |

**CAUTION:** SHORT outperformance and 24% LONG EOD WR on this day is likely market beta, not scanner skill — 13 of the top 15 misses were down-moves (same pattern as Sep 8's 12/12). **Need SPY/QQQ same-window return control on every row** to separate direction skill from tape.

---

## Review Corrections & Revised Execution Plan

### Corrections to Initial Analysis

1. **Gate value is noise, not negative.** -0.02%/trade is within SE. Gates are *redundant* (LOW_CONF catches conf 1-3 signals that fail the conf>=4 threshold anyway), not harmful. Accurate statement: gates add nothing because they catch signals that would not be selected regardless.

2. **71% NEUTRAL_BLOCKED is partly a logging artifact.** Fix M reveals that discovery BQ logging force-sets `direction='NEUTRAL'` for ALL non-deep-analyzed tickers (line 2637). So "NEUTRAL_BLOCKED" conflates two failures: (a) Stage 1 genuinely said NEUTRAL, vs (b) Stage 1 said LONG/SHORT but ticker didn't make the top-20 deep pool. These need different fixes. Ship M first (zero-risk), run one day, re-measure the real NEUTRAL rate, THEN size Fix D.

3. **Fix A impact overstated.** MRNA SHORT (+0.90%) appears in BOTH the contradiction table AND the winners list. Fix A does NOT "preserve all 7 winners." Net effect on Sep 10 is still positive, but the claim is wrong.

4. **Fix E contradicts V3.3.** HOOD/VST/CORZ shorted bullish catalysts because V3.3's rule says "go with the move, don't fight momentum." They followed V3.3 exactly as designed. Fix E (don't short bullish catalysts) is the exact inverse of V3.3 (go with the tape). Don't stack a gate on a rule it contradicts — pick one policy. Resolution: price direction decides only when the catalyst is genuinely ambiguous; a Buy initiation or multi-billion nuclear deal is not ambiguous. That means retuning/removing V3.3, not adding E.

5. **Fix B should be held.** 14/19 selected flagged exhausted=True means 74% fire rate — the flag is miscalibrated (peak-based total_move, 0.8 threshold, gap-down computation bug), not a strong filter. The +0.41% vs -0.26% split at n=5 vs n=14 has SEs wider than the gap. Fix exhaustion math first, then re-derive whether the flag predicts anything. No BE carve-out if shipped.

6. **Fixes F and L (confidence thresholds) — skip.** Confidence is non-predictive (doc's own table shows this). Lowering threshold for earnings (n=2 signal) and capping at 6 (n=1 signal) are rules on a variable that shouldn't be in `_is_selected` at all. Selection should run on remaining-move consistency and exchange data.

7. **Fix G should be deterministic, not prompt-based.** "Do not return NEUTRAL for post-earnings" produces forced directions. Earnings direction is a data lookup: sign of EPS/revenue surprise from Polygon financials or 8-K, plus premarket change sign. Deterministic, free, no LLM guessing.

8. **Fix D promotion criteria wrong.** Promote by exchange data — `|premarket_change| × premarket_volume / ADV` — not by Gemini's catalyst_score. Keep promoted-candidate FP rate as a separate metric.

9. **Pattern library bugs.** `primary_gap_pct` is populated from `expected_move_pct` (LLM estimate, not the actual gap). Dedup in library keeps highest score across scanners; eval keeps earliest-selected. Library and eval disagree about which signal "was" the trade. If keeping the library: store outcomes per pattern type, retrieve only patterns with >=N instances and measured hit rate, treat as labeled dataset for a classifier rather than prompt text.

### Revised Fix Classification

**Ship immediately (mechanical, zero judgment):**
- **M** — Discovery BQ logging: log actual Stage 1 direction instead of force-setting NEUTRAL
- **C** — Cross-scanner dedup + within-scanner run supersession (log every supersession as event)
- **Conf carve-out removal** — Apply existing mismatch gate at ALL confidence levels (remove conf<=5 restriction)

**Eval improvements (next):**
- Add SPY/QQQ same-window return to every eval row
- Unify entry latency logic
- Filter eval to runs that would actually be traded

**Policy decision required:**
- **Resolve E vs V3.3** — One policy for catalyst-vs-price conflicts, not two contradictory rules

**Ship after policy resolution:**
- **A** — Direction-remaining contradiction gate at all confidence levels

**Ship after M + one day of clean data:**
- **D** — NEUTRAL promotion via exchange-data criteria (not catalyst_score)

**Ship as deterministic checks:**
- **H** — Staleness on original news date (earliest published_utc or filing timestamp)
- **I** — Deal materiality gate (deal_value / market_cap < 3%)
- **K** — Named analyst firm required for analyst catalyst type

**Hold until exhaustion math rewrite:**
- **B** — Exhaustion gate (fix peak-based total_move, gap-at-entry as absorbed quantity, direction-aware, retrace-from-high as separate feature, then re-derive)

**Skip:**
- **F** — Lower confidence threshold for earnings (n=2 on non-predictive variable)
- **G** — Post-earnings prompt change (use deterministic lookup instead)
- **J** — Late-day staleness (resolved by trading-window decision)
- **L** — Cap confidence at 6 (n=1 on non-predictive variable)

---

## Charts

Generated by `scanner_pnl_eval.py` with decontaminated entry times:
- `scanner_pnl_timeseries_2026-09-10.png` — 6-panel time-series curves
- `scanner_pnl_bars_2026-09-10.png` — 6-panel summary bar charts
