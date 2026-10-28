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
