# Agent 1 — Live Scanner vs Backtest Divergence Audit

**Date:** 2026-09-20
**Files compared:**
- Live: `agent1.py` (production scanner, runs 6:15 AM PST)
- Backtest: `agent1_baseline_bt.py` (E2 benchmark config)

## Matching Parameters (identical)

| Parameter | Live | Backtest |
|---|---|---|
| TOP_N | 1 | 1 |
| Price min | $10 | $10 |
| Float min | 10M shares | 10M shares |
| Float policy | SOFT (dict only, miss=INCLUDE) | SOFT (dict only, miss=INCLUDE) |
| Float dict | 217 tickers, same values | 217 tickers, same values |
| Excluded tickers | 31 leveraged/inverse ETFs | Same set |
| Excluded suffixes | W, WS, U, R | W, WS, U, R |
| CS filter | Yes (Polygon reference API) | Yes (Polygon reference API) |
| Entry logic | local_high (w=3, max bar 12, fallback 15) | local_high (w=3, max bar 12, fallback 15) |
| PM reranking | OFF | OFF |

## Divergences

### D1: Gap threshold — RESOLVED
| | Live | Backtest |
|---|---|---|
| Value | `GAP_MIN = 0.03` (3%) | `gap_min = 0.03` (3% wide net) |

Both now use 3% minimum. Previously live used 5%.

### D1b: Gap calculation method — NEW divergence (LOW impact)
| | Live | Backtest |
|---|---|---|
| Value | Gap vs AH reference price (last 16:00-20:00 ET bar) | Gap vs prior close |

Live scanner now computes gap vs the last after-hours price on the prior
trading day, filtering out stale M&A gaps (ACVA, FEIM, BWET). The backtest
still uses prior close. This means live may exclude some candidates the BT
would trade, and gap percentages differ slightly for stocks with AH activity.

### D1c: Gap erosion filter — NEW divergence (LOW impact)
| | Live | Backtest |
|---|---|---|
| Value | Skip if >50% gap consumed by scan time | No erosion check |

Live scanner skips stocks where the gap is actively fading before entry
(intraday low consumed >50% of gap by SCAN_TIME_ET). BT does not have this.

### D2: Pattern filter (sustained_drop) — PARTIALLY RESOLVED

| | Live | Backtest |
|---|---|---|
| Value | E2-A: no pattern; E2-B: sustained_drop required | **Required** — `sustained_drop` only |

E2-B now implements sustained_drop at 7:30 PST (10:30 ET). E2-A still runs
without pattern filter — this is by design (dual-variant architecture).
BT trades correspond to the E2-B variant.

### D3: PnL entry reference price — HIGH impact

| | Live | Backtest |
|---|---|---|
| Value | `open_price` (gap open) | `entry_price` (local_high close) |

Live fill mode (`run_fill_outcomes()`) computes PnL as:
`(open_price - daily_close) / open_price`

Backtest computes PnL as:
`(entry_price - exit_price) / entry_price`

The local_high entry is typically lower than the open (stock has started
fading), so these can differ by several percentage points. Live performance
numbers are not directly comparable to backtest results.

### D4: Exit price — MEDIUM impact

| | Live | Backtest |
|---|---|---|
| Value | Daily close (grouped daily, ~16:00 ET) | 17:00 ET (after-hours bar close) |

Backtest iterates through RTH + after-hours bars and takes the last close at
or before 17:00 ET. Live fill mode uses Polygon grouped daily close, which is
the regular session close (~16:00 ET). The 1 hour of after-hours trading is
captured in BT but not in live tracking.

### D5: Stop loss application — LOW impact

| | Live | Backtest |
|---|---|---|
| Value | -50% disaster cap (manual in TOS) | Not applied in PnL calc |

Both are consistent in that the SL is manual (set in TOS by the trader).
The BT results include uncapped tails (max loss in ablation was -132.6%
baseline, -273.3% for T+2). The STRATEGY_LOCK variant G (PNL_CAP=-50%)
showed Sharpe improvement 0.547→0.624.

## Summary

| Divergence | Severity | Effect |
|---|---|---|
| D2: Pattern filter | **PARTIALLY RESOLVED** | E2-B has SD; E2-A by design does not |
| D3: PnL uses open_price not entry_price | **HIGH** | Live performance tracking is wrong |
| D1: Gap threshold | **RESOLVED** | Both use 3% |
| D1b: Gap calc (AH ref vs prior close) | LOW | Live more conservative (excludes stale gaps) |
| D1c: Gap erosion filter | LOW | Live more conservative (excludes faded gaps) |
| D4: Exit at daily close vs 17:00 AH | MEDIUM | Live misses AH fade capture |
| D5: SL not in BT PnL | LOW | Consistent — both manual |

---

# Part 2: Agent 1 v2 (`agent1_v2.py` vs `agent1_v2_backtest.py`) Audit & Resolution

**Date:** 2026-10-28
**Files Compared & Aligned:**
- Live / Single-Day Scanner: `agent1/agent1_v2.py`
- Multi-Month Backtest: `agent1/agent1_v2_backtest.py`
- Comparison Backtest: `agent1/agent1_v2_bt_comparison.py`

## 1. Seven Pipeline Divergences / Leakage Bugs Resolved (2026-10-28)

| # | Bug / Divergence | Prior State (Before 2026-10-28 Fix) | Resolution Implemented (100% Aligned) |
|---|---|---|---|
| **V2-D1** | **Float Cache Time-Travel Bias** | `agent1_v2.py` merged growing `data/float_cache.json` (`1,182` tickers populated by `v1`) on top of `_FLOAT_DATA` (`217` tickers), while `agent1_v2_backtest.py` used only `_FLOAT_DATA`. | Removed `float_cache.json` loading from `agent1_v2.py`. Both live and backtest strictly use the frozen 217-ticker `_FLOAT_DATA` dict (`<10M` = EXCLUDE, miss = INCLUDE). |
| **V2-D2** | **`PRE_FILTER_N = 30` Slicing Before `CS` Filtering** | Both scripts sliced `candidates[:30]` **before** filtering out leveraged single-stock ETFs/ETNs (`VRTL`, `SKHX`, `HYNX`, etc.) and non-CS tickers. On `2026-10-28`, 17 of 26 non-stale Top-30 slots (`65.4%`) were wasted on 2x/3x ETFs, truncating valid CS gappers. | Both `agent1_v2.py` and `agent1_v2_backtest.py` now filter `_EXCLUDE`, `_LEV_ETF`, suffix `W/WS/U/R`, and cached non-CS tickers in `data/ticker_type_cache.json` **before** slicing `candidates[:30]`, and persist newly discovered Polygon `type` lookups to `ticker_type_cache.json`. |
| **V2-D3** | **Array-Index vs. Clock-Timestamp Drift in `find_local_high`** | `find_local_high` checked array indices `3..11` assuming 1 bar per minute. On thin stocks with missing zero-volume bars, index `11 + 3` triggered as late as `10:08 ET` (`MB`, `RDAC`, `TBN`). | Enforced strict clock-timestamp bounds in `find_local_high`: `peak_bar["time_str"] <= "09:41"` and `entry_bar["time_str"] <= "09:44"`. |
| **V2-D4** | **Entry-Time Gap Erosion Blind Spot (`09:32–09:44 ET`)** | `gap_consumed <= 0.50` was only checked at `09:31 ET`. Stocks that flushed `>50%` of their gap between `09:32` and `09:36 ET` before forming a low-level bounce still triggered short entries after the gap was gone. | Added `ah_ref_price` check inside `find_local_high`: requires `(open_price - entry_price) / (open_price - ah_ref_price) <= 0.50` at the confirmation bar close. |
| **V2-D5** | **`raw_open_gap` Compression Bypass** | `VRT` (`2026-10-28`) opened at `+16.51%` vs prior close (`T6` institutional earnings zone), which compressed to `+12.05%` vs `ah_ref_price`, bypassing `15–30%` gap exclusion. | `check_s6`, `check_s1`, and `check_s12` now enforce gap-band exclusions on **both** `gap_pct` (vs AH ref) and `raw_open_gap` (vs prior close). |
| **V2-D6** | **Hardcoded `capital / 3` Sizing in Backtest** | `agent1_v2_backtest.py` divided daily capital by `3` (`MAX_SLOTS`) for all strategies regardless of `strat["optimal_n"]`. | Fixed `agent1_v2_backtest.py` to allocate `capital / strat["optimal_n"]` per slot (`N=3` for `S6`/`S1`, `N=2` for `S12`). |
| **V2-D7** | **Unconditional BigQuery Write in `agent1_v2.py`** | `log_scan_to_bq()` ran unconditionally at the bottom of `agent1_v2.py`, even when `DRY_RUN = True`. | Guarded `log_scan_to_bq()` with `if not DRY_RUN:`. |

## 2. Live Execution vs. 17:00 ET Backtest Reconciliation (`Oct 13–28, 2026`)

| Metric | Live Executed Trades (`Oct 13–28`, excl. RXRX) | Same 24 Trades Held to `17:00 ET` (BT Exit) | Old 5-Strategy Combined BT (`Oct 13–28`) | **New v2.1 `S6` (`Oct 13–28`)** | **New v2.1 `S1` (`Oct 13–28`)** |
|---|---:|---:|---:|---:|---:|
| **Trades** | `24` | `24` | `21` | **`10`** | **`11`** |
| **Win Rate** | `41.7%` (`10W / 14L`) | `33.3%` (`8W / 16L`) | `47.6%` (`10W / 11L`) | **`80.0%` (`8W / 2L`)** | **`81.8%` (`9W / 2L`)** |
| **Avg PnL / Trade** | **`+0.82%`** | **`+0.09%`** | **`+0.87%`** | **`+4.13%`** | **`+3.86%`** |
| **Total Sum PnL %** | **`+19.65%`** | **`+2.15%`** | **`+18.37%`** | **`+41.34%`** | **`+42.44%`** |
| **Realized / Sim Dollar PnL** | **`+$4,159.41`** | **`-$8,402.82`** (at live sizes) | `+$5,680` (`$100k` pool) | **`+$14,475` (`$100k` pool)** | **`+$14,893` (`$100k` pool)** |

