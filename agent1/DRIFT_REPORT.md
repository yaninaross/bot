# Agent 1 — DRIFT REPORT

## Reference: E2 Strategy Lock vs Live Scanner (`agent1.py`)

**Date:** 2026-09-21 (updated)
**Lock:** `STRATEGY_LOCK.md` v1.5 (E2 benchmark + AH freshness + gap erosion)
**Live:** `agent1.py` (production scanner, E2-A any time / E2-B 7:30 PST)

---

## Parameter-by-Parameter Comparison

| # | Parameter       | Strategy Lock (v1.5)             | agent1.py (live)                | Status   |
|---|-----------------|----------------------------------|---------------------------------|----------|
| 1 | Direction       | SHORT                            | SHORT (implied by gap-fade)     | MATCH    |
| 2 | Gap filter      | >= 3%                            | GAP_MIN = 0.03                  | MATCH    |
| 3 | Gap calc        | vs AH ref price (last 16:00-20:00 ET) | AH freshness filter in step 6a | MATCH |
| 4 | Gap erosion     | skip if >50% consumed by scan    | GAP_CONSUMED_MAX = 0.50 in step 6b | MATCH |
| 5 | Price min       | $10                              | PRICE_MIN = 10.0                | MATCH    |
| 6 | Float threshold | 10M shares                       | FLOAT_MIN_SHARES = 10M          | MATCH    |
| 7 | Float mode      | SOFT (dict only, miss=INCLUDE)   | SOFT (dict, miss=INCLUDE)       | MATCH    |
| 8 | Asset type      | CS only (Polygon ref)            | CS via `_is_common_stock()` API | MATCH    |
| 9 | TOP_N           | **5**                            | **5**                           | MATCH    |
| 10| Pattern filter  | **sustained_drop** (bar 60)      | **E2-B only** (not in E2-A)    | **BY DESIGN** |
| 11| Entry logic     | local_high (3-bar, bars 3-11)    | Implemented (E2-A bars 3-11, E2-B bars 60-72) | MATCH |
| 12| Exit            | **17:00 ET** (after-hours)       | **NOT IMPLEMENTED** (scanner only) | **DRIFT** |
| 13| PM reranking    | OFF (use daily open gap)         | OFF (not in pipeline)           | MATCH    |
| 14| Skip rule       | None                             | None                            | MATCH    |
| 15| Gap max         | None                             | None                            | MATCH    |
| 16| PNL cap         | None                             | None (not applicable to scanner)| MATCH    |
| 17| Ranking         | gap_pct descending (vs AH ref)   | gap_pct descending (vs AH ref)  | MATCH    |

## Drift Summary

**15 MATCH, 1 BY DESIGN, 1 DRIFT** (PNL cap doesn't apply to scanner)

### Resolved (since v1.0 drift report)

- **TOP_N:** Was 3 vs lock's 1. Lock updated to 5; live scanner now 5. MATCH.
- **sustained_drop:** Now implemented as E2-B variant (runs at 7:30 PST).
  E2-A intentionally runs without SD — this is the dual-variant design.
- **Entry logic:** Now implemented. E2-A: local_high bars 3-11, fallback 15.
  E2-B: local_high bars 60-72, fallback 75.
- **Gap calc:** Was vs prior close. Now vs AH reference price (last 16:00-20:00 ET
  bar on prior trading day). Catches stale M&A gaps (ACVA, FEIM, BWET).
- **Gap erosion:** New filter. Skips stocks where >50% of gap consumed by
  scan time (intraday low vs open). Catches faded gaps (TANH).

### Remaining Drifts

#### DRIFT 1: Exit logic NOT IMPLEMENTED
- **Location:** Not present in `agent1.py`
- **Strategy Lock:** Exit at 17:00 ET (after-hours, 1h post close)
- **Impact:** Human trader manages exit. E2 data shows the AH hour
  (16:00→17:00) is the biggest single-hour PnL jump.
- **Severity:** LOW — scanner is a screener, human executes.
  But human must be briefed on 17:00 exit.

## Scanner Pipeline (agent1.py)

For reference, the live pipeline is:

```
E2-A: runs at any time (no time gate)
  1. fetch_grouped_daily()     → all US equities, prior close
  2. fetch_snapshot_all()      → current/recent prices (live) or grouped daily (historical)
  3. compute_gaps()            → gap_pct >= 3%, price >= $10
  4. apply_float_filter()      → SOFT dict (217 tickers), miss=INCLUDE
  5. get_ah_reference_prices() → last AH price on prior trading day
     → recompute gap vs AH ref, exclude if fresh gap < 3% (stale)
  6. fetch_minute_bars()       → scan-day bars for top candidates
     → re-rank by gap at SCAN_TIME_ET (09:31)
     → skip if gap_consumed > 50% (erosion)
  7. select_top_n()            → CS check via API, top 5 by gap
  8. persist_scan_pool()       → save top 30 for E2-B reuse
  → output watchlist

E2-B: runs at 7:30 PST (10:30 ET) or later
  1. load_scan_pool()          → E2-A's top 30
  2. fetch_minute_bars()       → need 60+ bars for SD classification
  3. classify_open_pattern()   → sustained_drop only
  4. CS check + entry logic    → local_high bars 60-72, fallback 75
  → output watchlist
```

## Recommendations (DO NOT IMPLEMENT — informational only)

1. **Brief trader on 17:00 ET exit** — the AH hour is load-bearing

These recommendations are for discussion only.
