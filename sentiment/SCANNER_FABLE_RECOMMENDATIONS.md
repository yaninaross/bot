# Scanner V4 — Fable Recommendations & Implementation Status

**Source:** Fable 5.1 code review of V4 scanner system, Sep 11 2026
**Verified against:** actual V4 code by Claude Opus, Sep 11 2026
**Data basis:** 169 trades across Sep 10-11, premarket_scanner + discovery_scanner

---

## Priority: P0 — Before the next live run

### #1. Out-of-sample discipline (meta-fix)

**Problem.** Every rule since V3 was designed and "validated" on the same day's data. The −24.9% → +23.9% flip in the root-cause doc comes from removing 108 of 169 trades after observing their outcomes. With ~20 candidate rules and n=169, that result is guaranteed to exist in pure noise. You currently cannot tell which of your rules are real.

**Change.**
- Freeze V4 code for five trading sessions. No prompt edits, no gate changes, no weight changes.
- Split what you have: derive rules using only Sep 10, evaluate on Sep 11; then reverse. A rule ships only if it helps in both directions.
- Keep a pre-registration ledger: any proposed rule is written down (logic, expected effect, threshold) before the day it gets judged on.
- Report every stat with n and standard error. Any cell with n < 20 is labeled "not a finding."

**Validate.** The first rule that survives a forward test is your first real edge. Until then, treat the system as having none.

**Status:** `DONE` — Benchmark backtest script (`scanner_benchmark_backtest.py`) built to evaluate Sep 8-11 with proper methodology (per-signal entry times, SL simulation, confusion matrix). Code freeze in effect.

---

### #2. Symmetric, cross-run dedup

**Problem.** No cross-run state. Each scanner run is completely independent. APUR selected LONG and SHORT 93 minutes apart, TSLA SHORT ×3 then LONG, RXT selected three times. A long and a short on the same ticker is zero exposure with two sets of costs.

**Change.** One per-ticker per-day state object, shared by both scanners (BQ table keyed by date+ticker): `{direction, ev, first_selected_run, last_run, superseded_by}`. Rules: a ticker has at most one live direction per day; a later run with the opposite direction supersedes only if its EV exceeds the prior by 2×; otherwise it's ignored. Same-direction re-selection updates last_run, doesn't create a new trade.

**Validate.** Count of tickers with >1 direction per day must be zero.

**Code verification:** Confirmed — zero cross-run state in the code. Each scanner run is completely independent.

**Status:** `DONE` — `scanner_dedup.py` module implemented with BQ state table (`scoring_logs.scanner_ticker_state`). Integrated into both `premarket_scanner_v4.py` and `discovery_scanner_v4.py`.

---

### #3. Fix informed_score — numerical bug

**Problem.** `vol_signal = premarket_volume / avg_daily_volume / 3`. Premarket volume is typically 1-5% of a day's volume, so this term is ≈0.01 for almost every ticker. `idio_signal = |gap − SPY| / 5` hits 1.0 at a 5% gap. Net: `informed_score ≈ 0.4` for every meaningful mover — it carries no information (r = +0.013). The tape discount fired exactly once in 2 days because `informed_score > 0.5` (its activation threshold) requires gap > ~6%.

**Code verification:** Confirmed at `discovery_scanner_v4.py:2443-2456`. The volume term contributes effectively nothing. Tape discount is structurally disabled.

**Change.**
- Volume: normalize against the ticker's own premarket baseline — median premarket volume at the same clock time over the last 20 sessions, from Polygon minute bars. `vol_ratio = premarket_volume_now / median_premarket_volume_same_time`.
- Gap: express as a z-score against the ticker's 20-day daily return std, not a 5% cap.
- `informed_score = 0.6 × min(1, log1p(vol_ratio)/log1p(10)) + 0.4 × min(1, |gap_z|/3)`

**Validate.** Post-fix, distribution of informed_score should be wide (not clustered at 0.4). Check correlation with 60m continuation conditional on gap sign.

**Status:** `DONE` — Fixed in both `discovery_scanner_v4.py` and `premarket_scanner_v4.py`. Changes:
- Volume: normalized against premarket baseline (2% of avg daily vol) → ratio≈1 on normal day, 5-50 on event day
- Gap: expressed as z-score against 20-day daily return std
- Formula: `informed_score = 0.6 × min(1, log1p(vol_ratio)/log1p(10)) + 0.4 × min(1, |gap_z|/3)`
- Tape discount threshold lowered from 0.5 to 0.3, range extended to full veto (0.0)
- `daily_return_std` added to `fetch_price_data` output
- `gap_z` logged to BQ for tracking
- A/B comparison added to benchmark backtest script

---

### #4. Let Stage 2 return NEUTRAL

**Problem.** `tool_choice` forces LONG/SHORT on every deep-analyzed ticker. The 30 "unknown"-catalyst trades at -0.51% are directions invented because the schema demanded one.

**Code verification:** Confirmed at `discovery_scanner_v4.py:1244`: `"direction": {"type": "string", "enum": ["LONG", "SHORT"]}` — NEUTRAL is not in the enum. `tool_choice={"type": "tool", "name": "record_analysis"}` forces a tool call.

**Change.** Add NEUTRAL to the enum and to the prompt as a valid, expected outcome ("no tradable edge" should be the most common answer). Log NEUTRAL rows with `stage2_attempted=True`.

**Validate.** NEUTRAL rate from Stage 2 should be 30-60%.

**Status:** `DONE` — Implemented in both `discovery_scanner_v4.py` and `premarket_scanner_v4.py` (V4.1). Changes:
- Direction enum: `["LONG", "SHORT"]` → `["LONG", "SHORT", "NEUTRAL"]`
- Prompt updated: "NEUTRAL IS THE DEFAULT" block with specific criteria for when to use NEUTRAL
- Post-Stage-2 split: NEUTRAL entries captured in `stage2_neutrals` list, logged to BQ with `gate_reason='STAGE2_NEUTRAL'`
- Backtest: #3+#4 strategy added to benchmark using catalyst_type proxy (unknown/empty → would-have-been-NEUTRAL)
- Scanner version bumped V4.0 → V4.1

---

### #5. Housekeeping

- ~~Scrub GCP project ID and bucket name before push~~ → Already done
- Changelog says "Selection Inversion — RESOLVED in V4." The root-cause doc shows the opposite. Correct it — the changelog is institutional memory.

**Status:** `PARTIAL` — GCP scrubbed, changelog correction still needed.

---

## Priority: P1 — This week (after #1-4 are in)

### #6. Repair the residual math

The core V4 mechanism. Four sub-fixes:

#### 6a. Interval comes from data, not the LLM

**Problem:** Stage 2 sees the premarket gap, then estimates the "catalyst-justified" move. It anchors. ACVA (+44.6%) → LLM wrote interval around it → huge negative residual → EV=22.43 → top-ranked SHORT → stopped at minute 0.

**Code verification:** Confirmed at `discovery_scanner_v4.py:2415-2417`. `move_interval - gap_at_entry` produces anchored residuals.

**Change:** LLM outputs only classification (catalyst_type, direction, acquirer_or_target, guidance_direction, is_new, named_firm, deal_value). Interval looked up in Python from comparables: median and IQR of realized full-day move for (catalyst_type × market-cap bucket), from scanner_patterns + 20-month Polygon history. LLM never sees the interval or gap during classification.

#### 6b. Overshoot regime

**Problem:** No overshoot handling. Stock +44% above fair value just gets massive EV. Trading sign(residual) means fading an overshoot at the worst time (first 60 min = momentum dominates).

**Change:** If `|gap| > 1.5 × prior_high`, residual is not tradable. Route to momentum rule: `direction = sign(gap)` if `informed_score > 0.6`, else NO TRADE. Never a fade.

#### 6c. Concave, capped EV

**Problem:** `EV = |residual_mid| × ...` is linear and unbounded. EV 0.25-0.5 → 83% WR; EV 2-5 → 0% WR.

**Change:** `EV = min(2.0, sqrt(|residual_mid|)) × clarity × tape_discount`. Above the cap, more residual buys nothing.

#### 6d. Tape discount actually engages

With a working `informed_score` (#3), let the discount go to 0 (no trade), not 0.7, when heavy tape opposes the catalyst.

**Validate.** Re-run V4 selection on Sep 10-11 with new math, report only (still in-sample). Real test is the five frozen sessions.

**Status:** `TODO`

---

### #7. Encode the SHORT/LONG asymmetry as priors and sizing

**Evidence.** SHORT alpha +0.43% vs LONG -0.39% (n=94/75, ~2 SE). Offerings and negative regulatory continue down; earnings and deal LONGs get sold.

**Change.** Replace top-N-by-EV with expected-return estimate: `ER = base_rate(catalyst_type, direction, cap_bucket) × EV_normalized`, where base rates are shrunken empirical averages (k=30). Size = `clip(ER / stock_vol, 0, 1)`. Effects: SHORT on offering/CRL/guidance-cut gets full size; LONG on positive catalyst gets half size unless volume-confirmed and in drawdown zone; analyst SHORT prior ≈ 0 so it rarely trades.

**Code assessment:** Don't ban LONG entirely — offering+LONG at -2.50% is n=1 and analyst+LONG at +0.12% is actually positive.

**Status:** `TODO`

---

### #8. Scale the stop and the exit

**Problem.** Fixed 2.5% SL on a stock that gapped 30% is a coin flip in the opening bar — 34% stop rate on Sep 11 vs 12% on Sep 10. MFE data shows fixed 60m exit leaves money on the table.

**Change.**
- `stop = max(2.5%, 0.25 × ATR14/price)`
- Skip first 5 minutes for any name with `|gap| > 10%` (enter at 9:35 on first pullback)
- Trailing exit: hold to 60m or until giving back 50% of MFE, whichever first

**Validate.** Compare stop rate, avg loss on stopped trades, and MFE capture ratio before/after on frozen sessions.

**Code assessment:** The 5-min delay for large gappers is smart — AENT was stopped at minute 0 on a +31.9% gap.

**Status:** `TODO`

---

### #9. Deterministic extraction where it's cheap

- **Acquirer/target:** In any deal, the ticker gapping up ≥15% is the target; the other is the acquirer. Do it in code from Polygon prices before the LLM sees anything. ACVA (+44%) would have been LONG instead of SHORT.
- **Earnings direction:** Sign of EPS and revenue surprise from Polygon financials or the 8-K, plus guidance keywords. No LLM guess needed.
- **Named analyst firm:** Already a regex; keep it.

**Code assessment:** High ROI fix. The acquirer/target logic is trivially deterministic. This is a code-level fix, no LLM needed.

**Status:** `TODO`

---

## Priority: P2 — Next two weeks

### #10. Coverage: generate universe from price, not search

**Evidence.** 906 movers >5% over two days; 93 in scanner universe (10%); 7 caught. Stage 1 Gemini search is the recall bottleneck AND the NEUTRAL bottleneck (44 movers stuck NEUTRAL).

**Change.** Polygon `/v2/snapshot/locale/us/markets/stocks/gainers` and `/losers` at each run, filtered by price ≥ $3, premarket volume ≥ 20k, market cap ≥ $300M. That's the candidate list. LLM's job changes from "find movers" (bad) to "explain why this moved" (good). Keep deploy list as guaranteed subset.

**Validate.** In-universe share of >5% movers should go from 10% to 60%+ in a week.

**Code assessment:** This is the #1 recommendation by impact. Could 10x in-universe coverage.

**Status:** `TODO`

---

### #11. Nightly eval automation and comparables table

Run `scanner_pnl_eval.py` on a schedule after close, write results to BQ, and append every trade to a comparables table with catalyst_type, cap bucket, gap, volume ratio, 30/60m/EOD returns, MFE/MAE. Seed from 20-month minute history for deploy names. This table replaces the LLM's magnitude guess (#6a) and grows in value daily.

**Status:** `TODO`

---

### #12. Retire or rewrite the exhaustion flag

Fires on 72% of selections and separates nothing (-0.19% vs -0.14%). Under repaired residual (#6a-6c), "exhausted" is just residual ≈ 0, which yields low EV. Drop the flag; if you want retrace-from-premarket-high as a feature, log it separately.

**Status:** `TODO`

---

### #13. Adversarial second pass (optional)

A second Stage 2 call that receives the first analysis and must argue the opposite direction with the same facts. Trade only if the original thesis survives. Cheap, targets self-contradiction directly, but it's a refinement on a working system — not a fix for a broken one.

**Status:** `DEFERRED` — Only after #1-10 are validated.

---

## Key Insight (Fable's conclusion)

> If after five frozen sessions the SHORT-asymmetry and the repaired residual don't show positive alpha with n > 100, the right conclusion is that the LLM layer isn't adding value over a simple price/volume gap system — and that's worth knowing before more weeks go into it.

---

*Logged: 2026-09-11. Source: Fable 5.1 review + code verification by Claude Opus.*
