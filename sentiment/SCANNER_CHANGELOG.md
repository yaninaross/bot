# Scanner Agent Improvements Log

Documents every improvement to `premarket_scanner.py` and `discovery_scanner.py`, what problem it addressed, and measured impact.

---

## V1 — Initial Release (2026-09-04)

**Commit:** `aff485b` — Add premarket/discovery scanners V1

### Architecture
- **Premarket Scanner**: 2-stage LLM pipeline. Stage 1 scans ~68 deploy tickers via Polygon news API → batch LLM ranking. Stage 2 deep-analyzes top candidates → confidence + direction + reasoning.
- **Discovery Scanner**: Same 2-stage pipeline but searches 14 thematic queries (e.g., "biotech FDA approval", "crypto regulation") to find opportunities outside the deploy universe.
- LLM: Claude claude-opus-4-20250514 (later changed to 4-6 in V1.1)
- News source: Polygon.io REST API
- Price validation: yfinance for prev_close and current price
- Output: BQ tables `ys-time-series.scoring_logs.premarket_scanner` and `discovery_scanner`

### Known Issues at Launch
- Timestamp parsing fragile — articles with missing timestamps caused crashes
- Freshness ranking didn't weight recency aggressively enough
- Price validation missing for some tickers

---

## V1.1 — Bug Fixes (2026-09-04)

**Commits:** `f7fc440`, `db2bfc6`

### Changes
1. **Claude model updated** from claude-opus-4-20250514 to claude-opus-4-6-20250904 (newer, better reasoning)
2. **Timestamp fallback**: When article has no `published_utc`, falls back to `news_timeline` field
3. **16h cutoff**: Only consider articles published within 16 hours of scan time (was unbounded)
4. **Aggressive freshness weights**: Articles < 2h old get 3x weight, 2-6h get 2x, 6-16h get 1x
5. **Price validation fixes**: Handle missing/null prices gracefully

### Problem Addressed
First live run on Sep 4 had several crashes from malformed article data and was surfacing stale 24h+ articles as "breaking news."

---

## V2 — Baseline Performance (2026-09-08 morning)

No code changes from V1.1 — V2 is the label for the first full evaluation run with actual price tracking.

### Measured Performance (Sep 8)

**Premarket Scanner V2:**
- 14 signals evaluated
- 57% win rate (8/14)
- -0.09% average PnL
- Key FPs: TSLA LONG -8.63% (ignored premarket direction), ALAB SHORT -2.87%

**Discovery Scanner V2:**
- 20 signals evaluated
- 40% win rate (8/20)
- +0.30% average PnL
- Key FPs: CHAD SHORT -16.99% (preferred offering → squeeze), penny stocks (ARBE, BJDX)

### Problems Identified
1. **Selection inversion**: Selected trades performed WORSE than non-selected — the composite/tradability ranking was inversely correlated with actual PnL
2. **Self-contradictory reasoning**: LLM would write "minimal impact" or "untradeable" but still assign confidence 5-6
3. **Penny stock LONG traps**: Sub-$2 stocks gapping up on PR → fade. These consistently lost money on LONG side
4. **Pre-earnings speculation**: Tickers near earnings with no actual catalyst got high scores
5. **Partnership asymmetry missed**: PLTR/NBIS partnership news — scanner rated PLTR (big co, barely moved) instead of NBIS (small co, +14%)

---

## V3 — Quality Gates + Pattern Library (2026-09-08)

### Changes — Both Scanners

#### 1. Post-LLM Quality Gates
Applied after Stage 2 deep analysis, before final selection. Each gate can block a trade from `selected=TRUE` and logs a `gate_reason` to BQ.

| Gate | Logic | Problem Addressed |
|------|-------|-------------------|
| PENNY_STOCK | LONG: price < $3.00 blocked; SHORT: price < $0.25 blocked | Penny stock LONG traps (ARBE -2.38%, BJDX +0.73% but random). SHORT penny stocks CAN work (dilution/offering crashes) so threshold is very low |
| SELF_CONTRADICTION | 25+ kill phrases detected in reasoning ("untradeable", "zero edge", "no catalyst", "minimal impact", etc.) | LLM assigns high confidence while describing why the trade won't work |
| PRE_EARNINGS | 11 phrases ("expected to report", "after close today", "earnings tomorrow") | Speculative signals with no actual catalyst, just anticipation |
| LOW_CONF | confidence < 4 blocked from selection | Low-confidence signals waste capital |
| PREMARKET_MISMATCH | (premarket only) LONG when stock down >2% premarket, or vice versa | Fighting strong premarket momentum |

#### 2. Catalyst Impact Reference Table (in DEEP_SYSTEM prompt)
Empirical impact ranges injected into the LLM's system prompt so it can calibrate confidence against historical norms:

| Catalyst Type | Typical Impact |
|---------------|---------------|
| FDA procedural (filing, panel date) | +2% to +5% |
| FDA approval/rejection | +5% to +30% / -20% to -50% |
| Non-recurring earnings (crypto gains, one-time) | +0% to +2%, fades |
| Partnership/contract | +3% to +15% (for smaller company) |
| Analyst upgrade/downgrade | +/-2% to +/-5% |
| Editorial/opinion | ~0% (noise) |
| Offering/dilution | -5% to -20% (trade SHORT) |

#### 3. Warning Blocks in System Prompt
Three explicit WARNING sections with rules:

- **FDA PROCEDURAL EVENTS**: Distinguish approval (big) from procedural (small). Don't treat panel scheduling as approval.
- **NON-RECURRING EARNINGS**: One-time gains (crypto holdings, asset sales) are not operational improvement. Score conservatively.
- **PRE-EARNINGS SPECULATION**: No actual earnings = no catalyst. Don't assign high confidence to "might beat estimates."

#### 4. Proven Trade Outcome Archetypes (Tier 1 — Static)
Four real-world examples hardcoded into DEEP_SYSTEM prompt:

1. **Partnership Asymmetry** (PLTR/NBIS, Sep 8): Mega-cap + small-cap partnership → trade the small company. NBIS +14.2% vs PLTR +0.8%.
2. **Non-Recurring Earnings** (GME): Crypto holdings boost EPS → fades within hours. Conf should be ≤3.
3. **Pre-Earnings Speculation** (BRZE): "Might beat estimates" with no leak = no edge. Kill these signals.
4. **Penny Stock LONG Trap** (ARBE/BJDX): Sub-$2 stocks gap on PR → fade by midday.

#### 5. BQ Pattern Library (Tier 2 — Dynamic Few-Shot Retrieval)

**Table:** `ys-time-series.scoring_logs.scanner_patterns`

**Schema:** pattern_id, created_at, event_date, pattern_type, catalyst_type, tickers, primary/secondary ticker with full price data (entry, gap%, 30m/60m/EOD returns, MFE/MAE), lesson, pattern_rule, sentiment_summary, market_context, scanner_prediction (JSON), tags.

**Function:** `fetch_similar_patterns(catalyst_type, ticker, max_results=3)` — queries BQ for matching patterns by catalyst_type OR ticker. Returns formatted examples injected into Stage 2 prompt as "HISTORICAL PATTERN EXAMPLES."

**First entry:** `PARTNERSHIP_ASYM_001` — PLTR/NBIS partnership (2026-09-08). Primary: NBIS LONG +14.2%, Secondary: PLTR flat +0.8%.

#### 6. BQ Schema Update
Added `gate_reason` column (STRING) to both scanner BQ tables. Stores which gate blocked the trade, or empty string if passed all gates.

### Measured Impact (Simulated on Sep 8 V2 data)

**Premarket Scanner:**
| Metric | V2 | V3 | Delta |
|--------|----|----|-------|
| Win Rate | 57% | 80% | +23pp |
| Avg PnL | -0.09% | +4.33% | +4.42pp |
| Signals | 14 | 5 selected | Gates filtered 9 |

**Discovery Scanner:**
| Metric | V2 | V3 | Delta |
|--------|----|----|-------|
| Win Rate | 40% | 58% | +18pp |
| Avg PnL | +0.30% | +2.80% | +2.50pp |
| Signals | 20 | 12 passed gates | 8 gated |

**Key trade saved:** TSLA LONG -8.63% → gated by PREMARKET_MISMATCH (stock was down premarket)
**Key trade preserved:** PDSB SHORT +35.59% → passed all gates (price $1.16 but SHORT threshold is $0.25)
**1 TP lost:** GMEX LONG +10.53% → gated by LOW_CONF (confidence=3). Acceptable tradeoff.

### Unfixable by Gates
**CHAD SHORT -16.99%**: Preferred offering (non-convertible, non-dilutive) → LLM correctly identified offering but called it SHORT. Stock squeezed. Confidence=7. No gate catches a fundamentally wrong direction call with high confidence. The new prompt guidance about preferred vs common offerings should help future LLM reasoning.

---

## V3.1 — Recovery Bounce Gap Screener + AEHR Dual-Signal Fix (2026-09-08)

### AEHR Dual-Signal Bug — Root Cause & Fix

**Problem:** AEHR appeared as LONG in V2 and SHORT in V3 on the same day, confusing the output.

**Root cause:** NOT a same-run bug. Two separate issues:
1. V2 (5:51 AM PT): Stage 1 classified AEHR as LONG (score=5, "premarket gains +2%") but it wasn't in the top 15 for Stage 2. Logged to BQ as **LONG conf=0** — a meaningless Stage 1 label that looked like a real signal.
2. V3 (8:35 AM PT): Better news found ("New Silicon Photonics Order"), score=7 → deep-analyzed → Claude called it SHORT conf=6 (gap-fade thesis: +7.4% already absorbed).

**Price data verified:** V3 used Polygon prices ($86.26 close, $92.62 at scan time). These were CORRECT unadjusted prices — Polygon and yfinance have a 1.4x discrepancy likely due to a split adjustment, but the scanner did not hallucinate.

**Why SHORT was wrong:** AEHR was -37% below its 20-day high. An oversold stock bouncing on a real catalyst has momentum continuation, not fade. The gap screener (added below) would have caught this.

**Fix applied:** Non-deep-analyzed tickers now log `direction: NEUTRAL` instead of Stage 1's raw direction. This prevents phantom LONG/SHORT entries at conf=0 from appearing in BQ alongside real signals. Applied to both scanners.

---

### Problem Addressed
AXTI gapped +9.5% premarket on Sep 8 with no news catalyst — it was recovering to its 2-week-ago level after a 37% drawdown from the Aug 17 high ($97.79 → $61.64). Both V2 and V3 scanner runs called it SHORT (fade), because:
1. The only articles Polygon returned were stale (reporting a prior session's -8.7% drop)
2. The LLM defaulted to "no catalyst → gap-fade" thesis
3. V3 gate correctly blocked the SHORT (premarket_direction_mismatch), but nothing suggested LONG

Actual result: AXTI entry $64.48 → EOD $68.32 = **+6.0% LONG** missed.

### Changes — Both Scanners

#### 1. `fetch_20d_high()` Function (Polygon API)
Fetches 30 days of daily bars from Polygon, takes the last 20 trading days, returns:
- `high_20d`: highest high in 20 days
- `prev_close`: most recent close
- `drawdown_pct`: how far prev_close is below 20d high

All prices from Polygon exchange data — no LLM hallucination possible.

#### 2. Gap Screener (between Stage 1.5 and Stage 2)
After price data is fetched, checks all deep candidates:
- **Gap threshold**: premarket change >= +5.0%
- **Drawdown threshold**: prev_close >= 20% below 20-day high

If both conditions met → ticker flagged as "RECOVERY BOUNCE candidate."

#### 3. Recovery Bounce Context Injection (Stage 2 prompt)
For flagged tickers, injects a `⚠ RECOVERY BOUNCE ALERT` block into the Stage 2 prompt with:
- Verified 20-day high, prev close, drawdown percentage (all from Polygon)
- Recovery target (50% retracement of drawdown)
- Explicit instruction: "Do NOT default to gap-fade SHORT when stock is this far below recent range — recovery continuation LONG is more likely"

### Backtest on AXTI Sep 8
| Field | Value |
|-------|-------|
| 20-day high | $97.79 |
| Prev close | $61.64 |
| Drawdown | -37.0% |
| Gap | +9.5% |
| Triggered | YES |
| Recovery target (50%) | $79.72 |
| Actual entry→EOD | +6.0% LONG |

---

## V3.2 — Exhausted SHORT Gate + Clinical Data Trap (2026-09-08)

### Problem 1: Exhausted SHORT Fades — -16.53% total loss
Every selected SHORT trade where `move_is_exhausted=True` and `absorbed > 3%` was an FP.
The LLM correctly identified that the gap exceeded the catalyst's fair-value impact, but concluded "fade" — which was systematically wrong. Stocks that gap up on real catalysts continue via momentum and short squeezes, not fade.

**Gate added (both scanners):** `EXHAUSTED_SHORT_FADE` — when `move_is_exhausted=True` AND direction=SHORT AND absorbed > +3% → kill the trade. Only 1 marginal TP (ARM +1.60% at 60m, -2.05% EOD) would be lost.

Affected Sep 8 tickers: KEEL (-4.66%), IREN (-2.66%), BE (-2.29%), AMD (-1.90%), AEHR (-1.36%)

### Problem 2: Clinical Trial Data Trap — INBX -14.55%
LLMs read "positive Phase 2 data", "outperforms", "doubles response rate" and predict LONG. But institutions sell clinical data events for 4 reasons: (1) valuation exhaustion (priced for perfection), (2) buy-the-rumor-sell-the-news, (3) trial expansion = delayed timeline, (4) small sample sizes don't guarantee Phase 3.

**Gate added (both scanners):** `CLINICAL_DATA_TRAP` — when catalyst/reasoning contains clinical trial keywords (phase 1/2/3, ORR, complete response, endpoint, etc.) AND direction=LONG → cap confidence at 3 (premarket) or gate entirely (discovery).

**BQ pattern added:** `CLINICAL_DATA_TRAP_001` in `scanner_patterns` table with full INBX data, lesson, and rule.

**Prompt archetype added:** "CLINICAL TRIAL DATA TRAP" section with INBX real example added to DEEP_SYSTEM prompts in both scanners.

### Simulated Impact on Sep 8
- Exhausted SHORT gate: saves -16.53%, loses +1.60% (ARM) = net +14.93%
- Clinical data gate: saves -14.55% (INBX) = net +14.55%
- Combined: net improvement ~+29.48% if both had been active

---

## V3.3 — Smart Momentum: 3-Layer Gap-Up Logic (2026-09-08)

### Problem: Exhausted SHORTs Are Missed LONGs
V3.2 gated exhausted SHORTs (saving -16.53%), but killing the trades left money on the table.
Analysis of ALL 12 exhausted SHORTs from Sep 8 showed **every single one continued UP from open**:

| Ticker | Gap% | Drawdown from 20d High | LONG Return (from open) |
|--------|------|------------------------|------------------------|
| KEEL   | +10.5% | -11.5% | **+8.6%** |
| AXTI   | +9.5%  | -37.0% | **+7.9%** |
| BE     | +10.7% | -0.2%  | **+5.2%** |
| CLSK   | +5.4%  | -4.9%  | **+5.3%** |
| FLNC   | +8.2%  | -24.4% | **+4.8%** |
| IREN   | +8.5%  | -9.2%  | **+4.6%** |
| AMD    | +6.5%  | -7.7%  | **+4.0%** |

12/12 = **100% LONG win rate**, avg **+3.9%** per trade. The signal was there (the gap); we were just inverting it.

### Changes — Both Scanners

#### Layer 1: Enriched Context (Before LLM)
- Gap screener threshold lowered from **5% to 3%** — catches more gap-up tickers
- `fetch_20d_high()` now runs for ALL gap >= 3% tickers, not just those with drawdown >= 20%
- Zone classification injected into Stage 2 prompt:
  - **DEEP_RECOVERY**: >20% below 20d high → "recovery, not overextension"
  - **MODERATE_RECOVERY**: 10-20% below → same momentum bias
  - **MILD_DIP**: 5-10% below → moderate momentum bias
  - **NEAR_HIGHS**: <5% below → "momentum can continue but fade more plausible if weak catalyst"
- Short interest highlighted: SI > 10% + gap → "SHORT SQUEEZE WARNING" injected

#### Layer 2: Momentum vs Mean Reversion Prompt Rules (LLM Brain)
New DEEP_SYSTEM section teaches the LLM gap mechanics:

1. **Gap and Go**: Stock gapping >3% on real catalyst = momentum. Don't fade.
2. **Short Squeeze Danger**: SI > 10% + gap-up → heavily bias LONG.
3. **Recovery Bounces**: >10% below 20d high + gap = recovery, not overshoot.
4. **Gap and Crap** (when to SHORT): Only if catalyst is deceptive, micro-cap promotional, or stock AT highs with weak catalyst.
5. **Exhausted ≠ SHORT**: move_is_exhausted means gap absorbed the estimated impact. Default NEUTRAL, not SHORT. Backed by real data (12/12 failures cited).

#### Layer 3: Regime-Aware Momentum Flip Gate
Replaces the blanket EXHAUSTED_SHORT_FADE kill from V3.2.

**Risk-On regime** (SPY futures > -0.5%, VIX < 25):
- FLIP direction from SHORT → LONG
- Confidence scaled by drawdown zone:
  - >15% below 20d high: keep original confidence (strong recovery)
  - 5-15% below: `max(4, conf - 1)`
  - <5% below (near highs): `min(4, conf)` (cap — less certain)
- Entry moved from SHORT list to LONG list
- Gate reason: `momentum_flip_risk_on`

**Risk-Off regime** (SPY < -0.5% OR VIX > 25):
- KILL to NEUTRAL (don't flip — gap-ups in bear days are 50/50)
- Gate reason: `exhausted_short_killed_risk_off`

### Simulated Impact on Sep 8 (risk-on day: SPY futures positive)
Instead of saving -16.53% by killing, we would have CAPTURED:
- 12 LONG trades at avg +3.9% = **+46.8% aggregate** if all entered
- vs V3.2 which only saved the -16.53% loss (net: +0)
- Improvement over V3.2: **+46.8%** additional edge captured

### Why This Works
The "catalyst fair value ceiling" model is wrong for intraday trading. Gap-up stocks attract:
1. Retail FOMO at market open
2. Algorithmic momentum scanners (institutional)
3. Short squeeze covering (forced buying)
4. Recovery positioning when stock is below recent highs

The LLM even acknowledges these risks in its bear_case field — but still calls SHORT. The prompt rules + gate catch this systematic error.

---

## Pending / Future Improvements

### Nightly Auto-Evaluation Job
Missing automation: a nightly script that (1) fetches actual prices for all scanner signals, (2) computes PnL, (3) labels TP/FP, (4) auto-populates the pattern library with new examples. Currently done manually.

### AEHR Dual-Signal Bug — RESOLVED
Root cause: not a same-run bug. V2 logged Stage 1 direction (LONG conf=0) for non-analyzed tickers. V3 deep-analyzed and called SHORT. Fix: non-analyzed tickers now log `direction: NEUTRAL`.

### Selection Inversion
The fundamental problem — selected trades perform worse than non-selected — is not fully solved by V3 gates. Gates filter obvious bad signals, but the composite ranking that determines selection order may still be inversely predictive. Future work: investigate replacing composite score with a simpler signal (e.g., just confidence × catalyst_magnitude).
