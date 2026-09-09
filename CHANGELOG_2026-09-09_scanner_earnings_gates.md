# Changelog: Scanner Earnings Gates, Quality Gate Fixes & SEC EDGAR Timestamps (2026-09-09)

## Summary

Six changes across both `discovery_scanner.py` and `premarket_scanner.py` to fix
earnings direction bias, self-contradiction false kills, exhaustion threshold, and
add volume-based sell exhaustion as an independent lever.

Triggered by Sep 9 PnL analysis: TTAN -17.2% (LONG bounce predicted on guidance cut),
CRDO +3.52% and JOBY +3.98% killed by "tier 4/5" false positive, RKLB +4.21% killed
by aggressive exhaustion threshold.

---

## Fix 1: Stage 2 prompt — replace bounce bias with "guidance is everything" rule

**Files:** `discovery_scanner.py` (~L299-420), `premarket_scanner.py` (~L259-397)

**Root cause:** DEEP_SYSTEM prompt instructed Claude that a stock down 13% pre-market
"may be OVER" and the remaining move "could be a BOUNCE (LONG)." This caused Claude to
predict LONG bounce on TTAN after -20.3% AH drop despite guidance cut → -17.2% loss.

**Fix:**
- Removed bounce-bias framing ("bearish move may be OVER", "could be a BOUNCE")
- Added "GUIDANCE IS EVERYTHING" rule: earnings beat + guidance cut = TWO-WAVE SELLOFF
- Added TTAN as real example (beat EPS/revenue, cut Q3 guidance → AH -20.3%, open -12.4%)
- `total_catalyst_impact_pct` must match CATALYST direction, not bounce thesis
- Guidance matrix: beat+raise=LONG, beat+unchanged=cautious LONG, beat+cut=SHORT

---

## Fix 2: Remaining move formula — sign-mismatch bug

**Files:** `discovery_scanner.py` (~L1575), `premarket_scanner.py` (~L1554)

**Root cause:** `computed_remaining = claude_total - polygon_absorbed` produced +60.39%
when Claude predicted +30% (LONG bounce) but Polygon showed -30.39% (actual drop).
Opposite signs made subtraction additive instead of reductive.

**Fix:**
- Added `same_direction` check before subtraction
- When signs differ: `computed_remaining = claude_total` (no subtraction)
- Capped result at `abs(claude_total)` in both directions
- Opposite direction → `move_is_exhausted = True`

---

## Fix 3: Earnings gate (new Gate 6/4)

**Files:** `discovery_scanner.py` (~L2073), `premarket_scanner.py` (~L1772)

**What:** New Stage 3 gate kills earnings signals where stock gap-down >5% premarket.

**Logic:**
- `catalyst_type == 'earnings'` AND `change_pct < -5.0` → KILL (gap-down on earnings
  means market correctly priced the catalyst; continuation at open is the norm)
- `change_pct > +5.0` AND `direction == 'LONG'` → PASS (gap-up LONG = earnings beat
  continuation, e.g. SIG +19.1% gap → +3.69% PnL)
- `abs(change_pct) > 5.0` AND not gap-up LONG → KILL (e.g. gap-up SHORT = fading a
  real earnings beat, which fails)

**Impact:** TTAN, BRZE, CASY killed (correct). SIG preserved (+3.69%).

---

## Fix 4a: Self-contradiction phrases — remove "tier 4/5"

**Files:** `discovery_scanner.py` (~L1963), `premarket_scanner.py` (~L1662)

**Root cause:** `SELF_CONTRADICTION_PHRASES` included `'tier 4'`, `'tier 5'`,
`'tier 4/5'`. Claude uses catalyst tiers (1-5) as part of structured analysis —
saying "tier 4 catalyst" is honest quality rating, not self-contradiction. The
confidence gate (Gate 4, min conf = 4) already handles low-quality catalysts.

**Impact:** CRDO (+3.52%) and JOBY (+3.98%) recovered from false kills.

---

## Fix 4b: Exhaustion threshold — require minimum absolute remaining

**Files:** `discovery_scanner.py` (~L918, ~L1574), `premarket_scanner.py` (~L894, ~L1552)

**Root cause:** `is_exhausted = absorbed_frac > 0.80` flagged stocks as exhausted
when >80% of the move was absorbed, regardless of how much absolute remaining existed.
RKLB had 5.16% remaining (tradeable) but was flagged because 84% was absorbed.

**Fix:**
- `calculate_absorption`: `is_exhausted = (absorbed_frac > 0.80 AND abs(remaining) < 0.03) OR ...`
- Python math exhaustion: added `abs(computed_remaining) < 3.0` as additional condition
- A stock with 5% remaining is NOT exhausted even if 84% absorbed

**Impact:** RKLB (+4.21%) recovered from false exhaustion flag.

---

## Fix 5: Sell exhaustion volume lever (new Gate 7/5)

**Files:** `discovery_scanner.py` (~L2092), `premarket_scanner.py` (~L1797)

**What:** New independent Polygon-data-only gate for earnings gap-downs between -2%
and -5% (below the earnings gate threshold). Uses volume to detect whether
institutional selling is exhausted.

**Formula:**
```
sell_exhaustion_ratio = PM_volume / (avg_daily_volume × volume_multiplier)
```

**Volume multiplier** scales with gap magnitude:
| Gap size | Multiplier | Rationale |
|----------|-----------|-----------|
| 2-5%     | 2x        | Minor miss, moderate volume spike |
| 5-10%    | 3x        | Significant miss |
| 10-20%   | 5x        | Major miss / guidance cut |
| 20%+     | 8x        | Catastrophic (e.g. TTAN) |

**Gate logic:** If `ratio < 0.5` AND `direction == 'LONG'` → KILL. Institutions
haven't finished selling (PM volume is less than half of expected total day volume).

**Data source:** Added `avg_daily_volume` to `fetch_price_data()` in both scanners
(10-day average from Polygon daily bars). Falls back to 0 on yfinance path.

---

## Fix 6: Thematic query fixes (discovery scanner only)

**File:** `discovery_scanner.py` (~L99, ~L141, ~L152)

- Added `YESTERDAY_STR` constant
- `afterhours_movers` query: `tonight {TODAY_STR}` → `last night {YESTERDAY_STR}`
  (old query returned 0 results — "tonight" is wrong timing for morning scan)
- Added new `earnings_surprise` query targeting mid-cap earnings movers
  (old `earnings_today` was dominated by mega-cap coverage, missed ODD +18.3%,
  SIG +19.6%, TBBK -21.4%)
- Total thematic queries: 14 → 15

---

## Net PnL impact (Sep 9 signals)

| Ticker | Old result | New result | PnL | Effect |
|--------|-----------|------------|-----|--------|
| CRDO | KILLED (tier 4) | PASS | +3.52% | Recovered winner |
| JOBY | KILLED (tier 5) | PASS | +3.98% | Recovered winner |
| RKLB | EXHAUSTED | PASS | +4.21% | Recovered winner |
| SIG | PASS | PASS | +3.69% | Preserved winner |
| TTAN | PASS (LONG bounce) | KILLED | -17.20% | Blocked loser |
| BRZE | PASS | KILLED | -3.69% | Blocked loser |
| CASY | PASS | KILLED | +0.13% | Blocked (near zero) |

---

## Fix 7: Overnight AH freshness filter bug

**Files:** `discovery_scanner.py` (~L987), `premarket_scanner.py` (~L964)

**Root cause:** `verify_and_correct_event_timing` scans Polygon 5-min bars back
to the previous day. For AH earnings (e.g., ODD surging 33% after-hours on Sep 8),
Polygon detects the spike at ~4:30 PM ET yesterday. The function then overwrites
`news_time` to this AH timestamp. The downstream `_is_fresh` filter computes
age = ~20 hours > `MAX_NEWS_AGE_HOURS` (16) and **kills the signal as stale**.

AH earnings spikes from last night are absolutely fresh for today's trading.

**Fix:**
- Added overnight AH detection: `event_dt.date() < today AND hour >= 16 AND age <= 20h`
- When event is overnight AH: skip the news_time correction, preserve article time
- Non-AH events from previous days still get corrected (truly stale signals)

**Impact (discovery scanner Sep 9):**
| Ticker | Was | Fix | PnL | MFE |
|--------|-----|-----|-----|-----|
| ODD | FILTERED (AH stale) | PASS → top LONG | +3.37% | +3.90% |
| TTAN | FILTERED (AH stale) | PASS → top SHORT | +2.50% | +2.67% |
| JILL | FILTERED (AH stale) | PASS | +1.34% | +2.11% |
| SIG | FILTERED (AH stale) | PASS | +0.66% | +1.48% |
| NCNO | FILTERED (AH stale) | PASS | +0.53% | +0.64% |

With fix, projected top-5 LONG by tradability: ODD (5.76), PDYN (7.36),
ASO (1.57), GILT (5.04), SIG (2.42) → avg PnL significantly higher than
the +0.30% achieved with only ASO/CSR/SUNB selected.

---

## Fix 8: SEC EDGAR authoritative timestamps for earnings/deal catalysts

**Files:** `discovery_scanner.py` (~L965-1097), `premarket_scanner.py` (~L943-1050)

**Root cause:** `news_time` relied on LLM-extracted article publish timestamps. For
earnings catalysts, the LLM frequently hallucinated: COO's Sep 9 earnings were
released at 4:34 PM ET (SEC EDGAR: `acceptanceDateTime: 2026-09-09T16:34:31.000Z`),
but the LLM reported `05:45 PM ET` — the scan time, not the earnings time.

Wrong timestamps propagate into freshness multipliers (0.02x–1.5x of composite
score), silently distorting the ranking.

**Three approaches tried and evaluated:**

| Approach | Problem | Verdict |
|----------|---------|---------|
| Polygon spike time | Any event causes >3% move; earthquake, macro, etc. | Approximate, not authoritative |
| Gemini web search for earnings time | LLM searching articles returns article publish time, not event time | Same unreliability |
| SEC EDGAR 8-K filing acceptance time | Structured government data, exact UTC timestamp | Authoritative ✓ |

**Implementation — `fetch_sec_filing_time(ticker, filing_date)`:**
1. Get CIK from Polygon: `GET /v3/reference/tickers/{ticker}` → `results.cik`
2. Query SEC EDGAR: `GET data.sec.gov/submissions/CIK{cik_padded}.json`
3. Find `form == '8-K'` on target date → extract `acceptanceDateTime`
4. Convert UTC → ET

**Updated `verify_and_correct_event_timing()`:**
- SEC 8-K lookup runs FIRST for catalyst types `{earnings, deal, regulatory, offering}`
- Also checks `YESTERDAY_STR` for after-hours earnings from prior day
- Source tracked: `entry['news_time_source']` = `'sec_edgar'` / `'polygon_spike'` / `'llm'`
- Polygon absorption data always fetched regardless of timestamp source

**Verified:** COO → `2026-09-09T16:34:31.000Z` = 4:34 PM ET ✓

---

## Fix 9: Catalyst-type weights

**Files:** `discovery_scanner.py` (~L2210-2240), `premarket_scanner.py` (~L2210-2240)

**Rationale:** PnL analysis (Sep 3-9) showed massive variance by catalyst type.
Applied empirical weights to composite score after freshness multiplier.

**Weights (from backtest):**

| Type | Weight | Data |
|------|--------|------|
| macro | 1.2 | n=3, +2.9% avg, 67% WR |
| breaking | 1.2 | n=12, +2.7% avg, 67% WR |
| insider | 1.2 | n=5, +1.6%, 100% WR |
| offering | 1.0 | n=8, 0.0%, 50% WR |
| analyst | 1.0 | n=15, 0.0%, 47% WR |
| premarket_move | 0.5 | n=9, -0.3%, 11% WR |
| earnings | 0.7 | n=24, -0.5%, 38% WR |
| deal | 0.6 | n=25, -0.8%, 20% WR |
| regulatory | 0.4 | n=10, -1.5%, 0% WR |

Tagged: RE-EVALUATE as data accumulates.

---

## Fix 10: Mismatch index

**Files:** `discovery_scanner.py` (~L2245-2265), `premarket_scanner.py` (~L2245-2265)

**Formula:** `mismatch_index = gap_pct / confidence`

High mismatch (big gap, low LLM confidence) = retail FOMO on weak catalyst.

| Mismatch | Direction | Mult | Why |
|----------|-----------|------|-----|
| > 2.0 | LONG | 0.5x | FOMO gap, likely fades |
| > 2.0 | SHORT | 1.5x | Fade thesis correctly aligned |
| < 0.5 | LONG | 1.3x | Small gap, strong catalyst = room to run |
| other | any | 1.0x | Neutral |

Logged to BQ as `mismatch_index` for backtesting.

---

## Fix 11: Selected/Not-Selected output labeling

**Files:** Both scanners — `_is_selected()`, `_print_summary_table()`, `_print_detail_cards()`

**Problem:** Output was a flat ranked list — user couldn't quickly distinguish
selected signals (that would be traded) from analyzed-but-rejected signals.

**Changes:**
- New `_is_selected()`: `rank <= top_n AND conf >= 4 AND NOT gate_reason`
- Summary table splits into `═══ SELECTED (N) ═══` and `─── NOT SELECTED (N) ───`
- Detail cards show `✅ SELECTED` or `⛔ NOT SELECTED` tag
- All entries shown (not just top N) for debugging visibility
- BQ `selected` flag fixed in premarket to include `AND NOT _gate_reason`

---

## Fix 12: YESTERDAY_STR constant in premarket scanner

**File:** `premarket_scanner.py` (L105)

**Problem:** Discovery scanner had `YESTERDAY_STR` for overnight AH earnings
lookups and SEC EDGAR yesterday checks. Premarket scanner was missing it.

**Fix:** Added `YESTERDAY_STR = (NOW_PT - timedelta(days=1)).strftime('%Y-%m-%d')`
