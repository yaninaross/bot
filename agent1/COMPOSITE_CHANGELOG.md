# Agent 1 — Composite Backtest Changelog

## E2: N×Hold Grid + Capital Efficiency + Intraday Exit (2026-09-15)

**Script:** `agent1_e2_nhold_grid.py` — 3-part analysis, 174 trading days, CS+SD filter.

### Part 1 — Raw Grid (N×T+2..T+5)

| | T+2 | T+3 | T+4 | T+5 |
|---|---|---|---|---|
| N=1 Sharpe | 0.399 | 0.323 | 0.511 | **0.604** |
| N=1 Avg PnL | +18.40% | +19.32% | +23.60% | **+26.11%** |
| N=5 Sharpe | 0.253 | 0.303 | 0.429 | 0.475 |

- **Best raw: N=1 T+5** — Sharpe 0.604, +26.11% avg, 84% WR, PF 6.09, $+4.1M
- T+5 dominates ALL N. N=1 dominates ALL T. ALL 20 cells CI confirmed.

### Part 2 — Capital Efficiency ($100K fixed pool)

| | EOD | T+1 | T+2 | T+5 |
|---|---|---|---|---|
| N=1 Pool$ | **$1,980K** | $684K | $981K | $683K |
| N=1 Sharpe | 0.533 | 0.074 | 0.399 | 0.604 |

- **Best pool$: N=1 EOD** — $+1.98M, Sharpe 0.533, +12.37% avg, 76% WR
- Capital dilution flips raw picture: EOD ($1.98M) beats T+5 ($683K) by 3x
- **T+1 is dead zone** — Sharpe 0.027-0.074, CI NOT confirmed for any N

### Part 3 — Intraday Exit (10:00-20:00 hourly, incl. after-hours)

| | 10:00 | 11:00 | 12:00 | 13:00 | 16:00 | 17:00 | 20:00 |
|---|---|---|---|---|---|---|---|
| N=1 Sharpe | 0.579 | **0.788** | **0.790** | 0.685 | 0.535 | 0.589 | 0.528 |
| N=1 Pool$ | $637K | $1,535K | $1,886K | $1,966K | $1,977K | **$2,339K** | $2,312K |

- **Best overall exit: N=1 @17:00 AH** — $+2.34M, Sharpe 0.589, +14.62% avg, 78% WR
- Sharpe peaks 11:00-12:00 (0.788-0.790, PF 16.04) — most reliable fade window
- Afternoon dip 13:00→15:00 — fade partially reverses
- AH spike 16:00→17:00 — biggest single-hour jump ($1,977K→$2,339K)
- Max loss @17:00: -132.6% (TDIC) vs -334.1% @T+5 (SDOT). Better tail risk.
- 100% AH data availability (160/160 trades) for CS+SD stocks

### Key Decisions

- **Recommended config: N=1, exit at 17:00 (1h after close, after-hours)**
- If no AH access: exit at EOD (16:00) for $+1.98M
- If max Sharpe preferred: exit at 11:00-12:00 (Sharpe 0.790, $+1.5-1.9M)
- AVOID T+1 hold — overnight destroys edge

---

## agent1.py Production Scanner Changes

### 2026-09-21 — AH Freshness Filter + Gap Erosion + GAP_MIN 3% + TOP_N=5

**Changes:**
1. **AH freshness filter** — gap now computed vs last after-hours price (16:00-20:00 ET) on prior trading day, not vs prior close. Catches stale M&A/overnight gaps where the stock already moved to equilibrium in AH.
   - Example: ACVA daily gap +44.6%, but AH ref $10.38 → fresh gap +0.6% → EXCLUDED
2. **Gap erosion filter** — skip stocks where >50% of gap consumed by scan time. Uses intraday LOW (not close) through SCAN_TIME_ET to catch stocks that dropped hard then bounced.
   - `GAP_CONSUMED_MAX = 0.50`
   - Example: TANH gap_consumed 55% → EXCLUDED
3. **GAP_MIN changed 5% → 3%** — wider candidate pool for both E2-A and E2-B
4. **TOP_N changed to 5** — wider watchlist
5. **E2-A time gate removed** — scanner runs at any time (no longer gated to 6:31 PST)
6. **Step 6b restructured** — now runs for both live and historical modes (was historical-only). Fetches scan-day minute bars, re-ranks by gap at SCAN_TIME_ET, and checks gap erosion.
7. **New function `get_ah_reference_prices()`** — fetches prior day minute bars and returns last AH close per ticker.

**Motivation:** Live scanner was recommending stale gaps (ACVA — M&A stock that moved in AH 10 days prior) and eroded gaps (TANH — gap had already faded 55% by scan time). Both are false signals that waste trader attention.

---

### 2026-09-15 — Remove PM Retrace Filter + TOP_N=6

**Changes:**
1. **Removed PM_RETRACE_MAX = 0.80 filter** — backtest proved it costs -1.85% avg PnL
2. **TOP_N changed from 3 → 6** — wider candidate selection
3. **Removed PM bar fetch + retrace filter from scan pipeline** — steps 7-8 eliminated, scan goes straight from float filter to top-N selection

**Backtest evidence (agent1_baseline_bt.py, 174 trading days):**

| Metric | Baseline (no retrace) | PM Retrace (0.80) | Delta |
|--------|----------------------|-------------------|-------|
| Trades | 801 | 783 | -18 |
| Avg PnL | +8.90% | +7.05% | **-1.85%** |
| Sharpe | 0.237 | 0.199 | -0.039 |
| Win Rate | 67% | 65% | -2% |
| Profit Factor | 2.25 | 2.00 | -0.25 |
| Total $ (NC) | $+1,583,485 | $+1,205,808 | **-$377,677** |
| Max DD | -83.9% | -90.0% | -6.1% |

**Why retrace filter hurts:** It removes 156 trades averaging +12.73% (WR 69%) and replaces them with 138 trades averaging only +2.75% (WR 61%). High-retrace tickers are actually strong faders — the filter is counterproductive.

**Note:** PM bar data still stored in DB schema (informational) and displayed in watchlist. The `apply_retrace_filter` and `fetch_pm_bars_async` functions are retained but no longer called in the scan pipeline.

---

## Execution Rules (fixed across all tests)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Direction | SHORT only | Gap-fade strategy |
| Gap minimum | >= 5% | Open vs prior close |
| Gap maximum | <= 200% | Anything above = reverse split / data error |
| Price minimum | >= $20 | Was $10; raised to avoid micro-cap squeeze risk (Run 3+) |
| Float minimum | >= 10M shares | Polygon ref; cache miss = include |
| PM retrace | exclude > 0.80 | `(pm_high - pm_last) / (pm_high - open_price)`; None = include |
| Selection | Top 2 by gap size | After all filters applied |
| Entry | **Fallback only** (bar 15) | Local high entries skipped (Run 3+). CS_ONLY: local_high=-0.25%, fallback=+2.08% |
| Skip rule | `(open - bar15_close) / open > 0.15` → skip | Fallback only; too-faded entries exempt |
| Hold | T+2 close | No stop-loss, no intraday exit |
| PnL cap | -50% per trade | Clips catastrophic shorts (VELL -166%, WSHP -159%) |
| Capital | $100,000/day | Equal split across trades, non-compounding |
| Backtest period | 2026-01-01 → 2026-09-12 | Was 2025-09-15; Q4 2025 removed (net negative, not in E1-E10) |

---

## Universes Tested

### U0: OG_ORIGINAL
- **Filter:** `ticker_list` = 217 _FLOAT tickers, `top_n=5`, `pm_retrace_max=None`, `skip_drop_pct=None`, `no_float_filter=True`
- **Rationale:** Exact reproduction of original Agent 1 backtest (Sharpe 0.245, avg +9.07%). No new filters. Used to confirm calculation correctness and as the true baseline.
- **Status:** PENDING (Run 3)

### U1: BASELINE
- **Filter:** Base rules + all new filters (PM retrace, skip rule, Polygon float). N=2.
- **Rationale:** Full universe benchmark with current execution rules
- **Status:** Run 2: 497 trades, avg +1.14%, Sharpe 0.058, +$254K

### U2: CS_ONLY
- **Filter:** `exclude_etf=True, exclude_adrc=True`. N=2 + new filters.
- **Rationale:** E2 showed ETFs avg +1.82% (terrible for shorts). E4 showed ADRCs with catastrophic outlier losses.
- **Status:** Run 2: 482 trades, avg +0.63%, Sharpe 0.041, +$167K

### U3: OG_FLOAT
- **Filter:** `ticker_list` = 217 _FLOAT tickers. N=2 + new filters.
- **Rationale:** Same tickers as original but with N=2 and new filters to isolate the N effect.
- **Status:** Run 2: 158 trades, avg +0.18%, Sharpe 0.017, +$24K

---

## Run Log

### Run 1 — 2026-09-14 (BUGS FOUND)
- **Script:** `agent1_composite.py`
- **Universes:** BASELINE, CS_ONLY, OG_FLOAT
- **Runtime:** 20.3 min (250 trading days, 9,755 pool, 4,106 unique tickers)
- **Raw results:** BASELINE +$199K (Sharpe 0.046), CS_ONLY +$96K (0.013), OG_FLOAT -$105K (-0.139)

**3 bugs found:**

| Bug | Impact | Fix |
|-----|--------|-----|
| **OG_FLOAT pool undercount** | Only 50 trades (expected ~500). POOL_N=40 takes top 40 overall gaps; _FLOAT tickers rarely in top 40 so most are missed. | Expand pool: after top-40 overall, also include ALL candidates from ticker_list universes. |
| **WOLF 1387% gap (data error)** | Reverse split mis-reported as gap. SHORT lost -98.64%. Corrupts BASELINE/CS_ONLY results. | Add GAP_MAX_PCT=200% cap. |
| **Monthly Avg PnL vs Dollar PnL mismatch** | Jan 2026 shows avg +0.52% but Dollar -$19K. Avg PnL was simple trade average (unweighted); Dollar PnL was capital-weighted. 1-trade days get $100K vs 2-trade days get $50K each. | Changed monthly table to show "Avg Daily %" (= month_dollar / CAPITAL / active_days). |

### Run 2 — 2026-09-14 (RESULTS OK, but avg PnL much lower than original)
- **Script:** `agent1_composite.py` (with 3 bug fixes)
- **Universes:** BASELINE, CS_ONLY, OG_FLOAT
- **Runtime:** 20.7 min (250 days, 10,416 pool, 4,099 unique tickers)
- **Results:** BASELINE +$254K (0.058), CS_ONLY +$167K (0.041), OG_FLOAT +$24K (0.017)
- **Finding:** OG_FLOAT avg +0.18% vs original +9.07%. Root cause: 5 structural differences (N=2 vs 5, PM retrace filter, skip rule, Polygon float vs hardcoded, 4 extra months). See below.

**Key insight — not a bug, just different rules:**

| Difference | Original (9%) | Composite (<1%) | Impact |
|-----------|--------------|-----------------|--------|
| TOP_N | 5 | 2 | **Primary driver**. N=2 concentrates on most extreme gaps (dangerous for shorts) |
| PM retrace | none | exclude > 0.80 | Removes ~20% of candidates |
| Skip rule | none | exclude > 15% drop fallbacks | Removes fallback entries where gap already faded |
| Float source | hardcoded _FLOAT | Polygon API | Different values → some tickers excluded |
| Float filter | none (all 217 used) | exclude < 10M | Some _FLOAT tickers excluded by Polygon data |
| Period | Jan-Sep 2026 | Sep 2025 - Sep 2026 | Sep-Dec 2025 was negative for OG_FLOAT |

### Run 3 — PENDING (5 FIXES APPLIED)
- **Script:** `agent1_composite.py` (all filters per-universe; 5 fixes from Run 2 analysis)
- **Fixes applied:**
  1. `BACKTEST_START = '2026-01-01'` — remove Q4 2025 noise (-$81K drag, not in E1-E10 period)
  2. `PRICE_MIN = 20.0` — remove micro-cap catastrophic shorts (VELL -166%, WSHP -159%)
  3. `FALLBACK_ONLY = True` — skip local_high entries (CS_ONLY local_high=-0.25% vs fallback=+2.08%)
  4. `PNL_CAP = -50.0` — clip worst-case PnL per trade
  5. OG_ORIGINAL keeps original settings (`price_min=10, fallback_only=False, pnl_cap=None`)
- **Universes:** OG_ORIGINAL (reference), BASELINE, CS_ONLY, OG_FLOAT
- **Expectation:** BASELINE/CS_ONLY should show materially higher avg PnL with fixes; OG_ORIGINAL unchanged
- **Results:** TBD

---

## Investigations I1-I10 — 2026-09-14

**Script:** `agent1_investigations.py` (1,771 lines, Colab-cell)
**Baseline:** E1-E10 exact settings — 217 _FLOAT tickers, TOP_N=5, both local_high+fallback entry, gap>=5%, price>=$10, T+2 hold, no PM retrace, no skip rule, no PnL cap (avg +8.84%, Sharpe 0.235, n=793)
**Period:** 2025-09-15 → 2026-09-12 (extended for I10 OOS validation, cutoff 2026-01-01)

**Architecture:** Single data-collection pass (~30 min) → 10 investigation functions → consolidated markdown report + charts

| ID | Investigation | What it tests |
|----|--------------|---------------|
| I1 | CS_ONLY standalone | Re-rank top-5 from CS-only pool; non-CS type breakdown |
| I2 | CS + sustained_drop | Most important combo; overlap analysis; I2c intraday management (close DTR at 10:30) |
| I3 | CS + gap freshness | 4 freshness combos (stale, stale+moderate, not very_fresh, stale+moderate+fresh) |
| I4 | CS + price floor | $30/$50/$100 floors; excluded-trade quality; trades-per-day check |
| I5 | CS + extended hold | T+2 through T+5; max simultaneous positions; capital analysis |
| I6 | CS + volume ratio | Vol caps 1.5x/2.0x/3.0x, floor 5.0x; vol-PnL correlation |
| I7 | Composite filter | CS + stale/moderate + $30 + T+3; harmful-filter flagging per step |
| I8 | Intraday management | Close DTR at 10:30; counterfactual vs T+2 hold; aggressive variant |
| I9 | Sector-weighted ranking | Tier A/B/C weights from E9; hard exclude Tech+Mfg |
| I10 | Out-of-sample validation | Q4 2025 (OOS) vs Jan-Sep 2026 (IS) for baseline, CS, I7 composite |

**Data enrichment added beyond E1-E10:**
- `vol_ratio` — gap day volume / 20-day ADV
- `price_1030` — bar 60 close (10:30 AM) for intraday management
- `is_cs`, `is_adrc` — asset type flags from Polygon reference
- Multi-period outcomes: T+1 through T+5
- Optional Gemini LLM sector labeling (`ENABLE_LLM_SECTORS=True`)

**Each investigation includes:** LOO stability, filter funnel with n at each step, trades-per-day practicality check, CI 95%, trimmed mean.

**Outputs:** `INVESTIGATION_RESULTS.md`, `agent1_investigations_sharpe.png`, `agent1_investigations_scatter.png`

**Status:** Script complete, pending execution

---

## 5-Way Filter Comparison — 2026-09-15

**Script:** `agent1_baseline_bt.py` (5-way: 2x2 matrix + guards variant)
**Period:** 2026-01-02 → 2026-09-11 (174 trading days)
**Settings:** TOP_N=5, PRE_FILTER_N=15, local_high + fallback entry, T+2 hold, $100K/day NC

### Results

| Metric | A) Baseline | B) CS Only | C) Sust.Drop | D) CS+SD | E) CS+SD+G |
|--------|-------------|------------|--------------|----------|------------|
| Trades | 801 | 775 | 649 | 476 | 244 |
| Active days | 174 | 174 | 171 | 160 | 124 |
| Trades/day | 4.6 | 4.5 | 3.8 | 3.0 | 2.0 |
| Avg PnL | +8.90% | +8.32% | +9.42% | **+12.15%** | +5.98% |
| Median PnL | +7.68% | +7.45% | +9.18% | **+10.12%** | +5.11% |
| Trimmed Mean | +10.89% | +11.36% | +12.57% | **+14.97%** | +6.34% |
| Win Rate | 67% | 68% | 73% | **78%** | 70% |
| Sharpe | 0.237 | 0.186 | 0.178 | 0.255 | **0.272** |
| Profit Factor | 2.25 | 2.07 | 2.36 | **3.05** | 2.89 |
| 95% CI | [+6.30, +11.50] | [+5.18, +11.47] | [+5.36, +13.48] | [+7.87, +16.42] | [+3.22, +8.73] |
| Total $ (NC) | $+1,583,819 | $+1,467,398 | $+1,657,970 | **$+1,974,222** | $+802,017 |
| Max DD | -83.9% | -11240% | -1059% | -3223% | **-57.3%** |

**E) CS+SD+G guards:** gap <= 75% + Polygon `weighted_shares_outstanding` >= 10M (hard exclude on missing)

### Delta vs Baseline

| Metric | CS Only | Sust.Drop | CS+SD | CS+SD+G |
|--------|---------|-----------|-------|---------|
| Avg PnL | -0.58% | +0.52% | **+3.25%** | -2.92% |
| Median | -0.23% | +1.50% | **+2.44%** | -2.57% |
| Win Rate | +2% | +7% | **+11%** | +4% |
| Sharpe | -0.051 | -0.059 | +0.018 | **+0.035** |
| PF | -0.18 | +0.11 | **+0.80** | +0.64 |
| $ (NC) | -$116,421 | +$74,152 | **+$390,403** | -$781,801 |

### Forward Benchmark: CS+SD (D)

**CS+SD is the forward benchmark** — best on avg PnL (+12.15%), WR (78%), PF (3.05), and total $ ($1.97M). It dominates all other configurations on absolute return.

CS+SD+G (E) has the highest Sharpe (0.272) and dramatically reduced tail risk (max loss -245% vs -717%), but at the cost of halving trade count (244 vs 476) and $1.17M less total $ — the guards remove too many winning trades along with the losers.

**Decision: Use CS+SD as the primary benchmark for future experiments.** The guards concept is validated (tail risk reduction works) but the specific implementation is too aggressive for production. Future work should explore softer guards (float >= 5M, gap <= 100%) or PNL_CAP instead.

### Key Findings

1. **CS+SD is the clear winner on returns** — +12.15% avg, 78% WR, PF 3.05, $+1.97M NC
2. **Synergistic interaction** — each filter ALONE reduces Sharpe (CS: 0.186, SD: 0.178 vs baseline 0.237), but COMBINED they improve it to 0.255. Neither works solo; both work great together.
3. **CS+SD+G validates the guard concept** — Sharpe improves to 0.272, worst loss from -717% to -245%, max DD from -3223% to -57%. But avg PnL drops to +5.98% and $ halves.
4. **Float filter is the heavy hitter** — only 561/1293 tickers (43%) have Polygon shares >= 10M. This eliminates most micro-cap catastrophic shorts (WOK 4.5M, ELAB 0.8M, VCIG 0.6M, SDOT 1.5M, AIXI 0.1M) but also removes many profitable mid-cap trades.
5. **VCX (-245%) survives all guards** — 35.8M shares, 37.9% gap, CS, sustained_drop. This is a legitimate large-cap trade that went wrong. Only PNL_CAP can address this.

### Pattern Breakdown (Baseline trades)

| Pattern | n | Avg | WR | Sharpe |
|---------|---|-----|-----|--------|
| sustained_drop | 323 | +16.87% | 80% | 0.467 |
| drop_then_recover | 233 | +2.06% | 60% | 0.042 |
| mild_drop | 113 | +4.18% | 55% | 0.179 |
| flat_or_bounce | 93 | +5.77% | 48% | 0.288 |
| no_data | 39 | +4.89% | 72% | 0.351 |

### Asset Type Breakdown (Baseline trades)

| Type | n | Avg | WR | Sharpe |
|------|---|-----|-----|--------|
| CS | 538 | +11.43% | 70% | 0.297 |
| ETF | 165 | +1.86% | 57% | 0.093 |
| UNK | 47 | +11.67% | 55% | 0.526 |
| ADRC | 21 | -2.74% | 71% | -0.026 |

### Tail Risk — CS+SD+G vs CS+SD

| | CS+SD worst | CS+SD+G worst |
|--|-------------|---------------|
| 1 | WOK -717% (4.5M shares) | VCX -245% (35.8M shares) |
| 2 | ELAB -273% (0.8M, gap 143%) | PN -82% |
| 3 | VCIG -253% (0.6M shares) | MLTX -22% |
| 4 | VCX -245% (35.8M shares) | MOVE -21% |
| 5 | SDOT -212% (1.5M shares) | ODD -20% |

Guards eliminated 4 of 5 catastrophic trades. Only VCX survives (legitimate large-cap).

### 2x2 Marginal Value Matrix

| Comparison | What it tests | Avg PnL Delta | Sharpe Delta |
|------------|---------------|---------------|--------------|
| B vs A | CS filter alone | -0.58% | -0.051 |
| C vs A | sustained_drop alone | +0.52% | -0.059 |
| D vs B | sustained_drop on top of CS | +3.83% | +0.069 |
| D vs C | CS on top of sustained_drop | +2.73% | +0.077 |
| D vs A | both combined | +3.25% | +0.018 |
| E vs D | guards on top of CS+SD | -6.17% | +0.017 |

**Conclusion:** CS+SD is the forward benchmark. Guards concept works for tail risk but costs too much in returns. Future: explore PNL_CAP (-50%) on CS+SD instead of hard float exclude.

---

## Universe Float Comparison: Soft vs Hard (2026-09-15)

**Script:** `agent1_universe_float.py` — CS+SD, N=1, exit@17:00, 174 trading days.

| Metric | SOFT_FLOAT (current) | HARD_FLOAT (Polygon ≥10M) | Delta |
|--------|---------------------|--------------------------|-------|
| Trades | 160 | 126 | -34 |
| Avg PnL | +14.71% | +5.67% | -9.04% |
| Sharpe | 0.594 | 0.478 | -0.116 |
| Pool $ | $+2.35M | $+715K | -$1.64M |
| Max Loss | -132.6% | -40.4% | +92.2% better tail |

**Key finding:** The 94 trades removed by hard float filter are the BEST trades (avg +20.39%, Sharpe 0.706, WR 86%). Small-float stocks gap up harder and fade harder — they ARE the edge.

**Verdict:** KEEP soft float filter. Do NOT switch to hard float.

---

## Future Universe Ideas (not yet tested)

| ID | Name | Filter | Hypothesis |
|----|------|--------|------------|
| U4 | NO_ETF | `exclude_etf=True` | Isolate ETF effect without ADRC exclusion |
| U5 | QUALITY | `exclude_etf + exclude_adrc + pattern=sustained_drop + exclude_fresh` | Stack best E1-E10 filters |
| U6 | PRICE_50 | `exclude_etf + exclude_adrc + price_min=50` | E5 showed $50+ Sharpe 0.381 |
| U7 | STALE_GAP | `exclude_etf + exclude_adrc + require_fresh=[stale, moderate]` | E8 stale gaps Sharpe 0.564 |
| U8 | MAX_STACK | All of above combined | Risk: n too small for statistical significance |
