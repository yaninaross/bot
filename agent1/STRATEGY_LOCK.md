# Agent 1 — STRATEGY LOCK

## Locked Config: E2 Benchmark

This is the production strategy definition. Any deviation requires
ablation evidence showing improvement.

| Parameter       | Value                                          |
|-----------------|------------------------------------------------|
| Direction       | SHORT                                          |
| Universe        | Full US equity (Polygon grouped daily)         |
| Gap filter      | gap_pct >= 3% vs AH reference price            |
| Asset type      | CS only (Polygon reference `type == 'CS'`)     |
| Pattern         | sustained_drop only (bar 60 = 10:30 AM ET)     |
|                 | drop_30m < -3% AND recovery <= 50%             |
| Float filter    | SOFT — hardcoded 217-ticker dict only           |
|                 | Dict miss = INCLUDE (pass through)             |
|                 | Dict hit + float < 10M = EXCLUDE               |
| Gap calc        | gap = (open - ah_ref) / ah_ref; ah_ref = last   |
|                 | AH price (16:00-20:00 ET) on prior trading day  |
|                 | Falls back to prior close if no AH data         |
| Gap erosion     | skip if >50% of gap consumed by SCAN_TIME_ET    |
|                 | gap_consumed = (open - low) / (open - ref_price)|
| TOP_N           | 5 (top 5 by gap_pct descending)                |
| Ranking         | gap_pct descending (vs AH ref, no PM adj)       |
| PM reranking    | OFF                                            |
| Entry           | local_high: 3-bar window, scan bars 3-11       |
|                 | Fallback: bar 15 close (9:45 AM ET)            |
| Stop loss       | -50% disaster cap (entry × 1.50 for SHORT)     |
|                 | Human sets in TOS — not automated               |
| Exit            | 17:00 ET (1h after close, after-hours)         |
| Skip rule       | NONE — eval showed no threshold improves $      |
| Gap max         | None                                           |
| PNL cap         | None (SL=-50% is the disaster cap)             |
| Capital         | $100K/day, non-compounding                     |
| PnL formula     | (entry_price - exit_price) / entry_price × 100 |

## E2 Reference Numbers

| Metric        | Value      |
|---------------|------------|
| Trades        | 160        |
| Avg PnL       | +14.71%    |
| Win Rate      | 79%        |
| Sharpe        | 0.594      |
| Profit Factor | 6.26       |
| Total $ (NC)  | $+2,350,000|
| Period        | 2026-01-02 to 2026-09-11 (174 trading days) |

## Ablation Results (2026-09-18 run)

### Q1: Is E2 reproducible?
**YES.** Reproduced at +13.64% avg (166 trades), within 1.07pp of
+14.71% reference. Sharpe 0.547 vs 0.594. $2,264,952 vs $2,350,000.

### Q2: What is the costliest deviation?
**Hard float (A): -9.02pp.** Full ranking:

| Rank | Variant | Δ Avg PnL | Sharpe | Max Loss | $ NC |
|------|---------|-----------|--------|----------|------|
| 1 | A: Hard float | -9.02pp | 0.445 | -40.4% | $634K |
| 2 | E: TOP_N=5 | -6.70pp | 0.189 | -779.2% | $1,203K |
| 3 | B: PM reranking | -5.45pp | 0.123 | -779.2% | $1,352K |
| 4 | C: Exit EOD | -2.05pp | 0.509 | -86.2% | $1,924K |
| 5 | F: Skip >15% | -1.69pp | 0.493 | -132.6% | $1,674K |
| — | **Baseline** | **—** | **0.547** | **-132.6%** | **$2,265K** |
| 6 | G: PNL_CAP=-50% | +0.51pp | **0.624** | -50.0% | $2,350K |
| 7 | D: T+2 hold | +3.83pp | 0.385 | -273.3% | $2,901K |

### Q3: Best config WITH a PNL cap?
**PNL_CAP=-50% is a free upgrade.** Avg +14.16% (vs +13.64% base),
Sharpe improves 0.547→0.624, PF 5.47→6.58, max loss capped at -50%.
Total $ nearly identical ($2,350K vs $2,265K). Risk-adjusted, G
dominates the baseline.

### Key Insight: D (T+2) is a trap
T+2 shows +3.83pp higher average but Sharpe collapses 0.547→0.385
and max loss balloons to -273%. The overnight risk is not compensated.

## Divergences from CLAUDE.md "Frozen Baseline"

The CLAUDE.md CS+SD+GUARDS spec and this lock are now mostly aligned
after v1.3-1.5 updates. Remaining differences:

| Parameter     | E2 (this lock)   | CLAUDE.md Guards  |
|---------------|------------------|-------------------|
| Float miss    | INCLUDE          | HARD EXCLUDE      |
| Gap max       | None             | <= 75%            |
| Skip rule     | None             | >15% drop = skip  |
| Exit          | 17:00 ET (AH)   | T+2 close         |
| PNL cap       | None             | Not specified      |

---

## Locked Config: Agent 1 v2.4 (`agent1_v2.py` — `Deployed C2 (Tier1→Tier2)`, `Combo C1`, `Combo C3` Hybrid Suite)

Updated on **2026-11-06** after the 12-Month Walk-Forward Audit (`2026-11-02`), Priority Tier Ranking (`Tier 1 → Tier 2`, `2026-11-04`), the 2.33-Year (`28` Months: `2024-07-01 → 2026-11-06`, `592` Trading Days) Multi-Horizon Equal-Weight Portfolio Backtest, and the 2.33-Year Google News RSS + SEC EDGAR Point-in-Time Catalyst Study (`98.8%` verified catalyst coverage across `929` `C2` trades).

### Base Pipeline & Hybrid Execution Rules (Shared Across `C1`, `C2`, `C3`)

| Parameter | Value |
|---|---|
| Direction | SHORT |
| Universe | Full US equity (Polygon grouped daily + live snapshot) |
| Pre-filter hygiene & SPAC filter | Exclude known ETFs, leveraged ETNs, warrants/units/rights (`W/WS/U/R`, `.WS`, `.WT`), cached non-CS in `ticker_type_cache.json`, and **SPAC / Blank-Check shells (`ACQUISITION`, `BLANK CHECK`, `SPAC` in company name)** **BEFORE** slicing `PRE_FILTER_N = 30` |
| Asset type | Common Stock (`type == 'CS'`) verified via Polygon reference + persistent `ticker_type_cache.json` |
| Price bounds | `$10.00 <= open_price <= $100.00` (`PRICE_MIN = 10.0`, `PRICE_MAX = 100.0`) |
| Float filter | Frozen 217-ticker `_FLOAT` dict (`< 10M` shares = EXCLUDE, dict miss = INCLUDE) |
| Gap filter | `gap_pct >= 3.0%` vs last AH price (`16:00–20:00 ET`) on prior trading day |
| Priority Tier Ranking (`USE_PRIORITY_TIER_RANKING = True`) | Candidates sorted by **`(priority_tier, gap_pct)` descending**:<br>• **Tier 1 (`Priority 1`):** `gap_pct >= 30%` OR (`gap_pct < 15%` AND `open_price <= $25.00` AND `prior_dvol < $25M`).<br>• **Tier 2 (`Priority 0`):** `gap_pct < 15%` AND (`open_price > $25.00` OR `prior_dvol >= $25M`), used only as fallback when `< 3` Tier 1 candidates qualify. |
| Scan-time erosion | Skip if `gap_consumed > 0.50` by `06:31 PST` (`09:31 ET`, using intraday low vs `ah_ref_price`) |
| Entry-time erosion | Skip if `entry_price` has already consumed `> 50%` of the gap above `ah_ref_price` |
| Moderate-Gap Guard (`<15%` gap) | `MAX_MISSING_BARS_MOD_GAP = 1` (`<= 1` missing 1-min bar before entry) AND `MAX_OPEN_RUNUP_MOD_GAP = 0.07` (`< +7.0%` post-open run-up above `open_price` at entry) |
| Liquidity gate | `>= 2` bars and `>= 5,000` shares cumulative volume in `06:30–06:31 PST` (`09:30–09:31 ET`) |
| Morning Entry #1 (`bar0_flush`) | **Bar 0 Opening Flush (`06:33 PST` / `09:33 ET`):** Bar 0 (`06:30 PST`) is red (`c <= o`), its high holds unbroken through bars 1–3 (`06:31–06:33 PST`), drop below open `>= 1.0%`, and `>= 35%` of gap remains unfilled |
| Morning Entry #2 (`local_high`) | **3-Bar Local High (`06:36–06:44 PST` / `09:36–09:44 ET`):** 3-bar lookback + 3-bar confirmation across `bars 3..11` (`peak_bar <= 06:41 PST`, `entry_bar <= 06:44 PST`), requiring `MIN_PULLBACK_PCT = 0.5%`, with **`continue_on_failed_pb = True`** |
| Midday Add-On #1 (`bounce_reshort`, `_B2`) | **Midday Bounce Re-Short (`06:55–10:35 PST` / `09:55–13:35 ET`, `C1/C2/C3`):** If `< 3` slots used and an open morning short hits `>= +3.0%` MFE drop then bounces `>= +1.5%` off its low to form a confirmed 3-bar local high (`>= 0.5%` pullback) while still `>= +0.5%` in profit, short 1 idle slot (`exit 11:00 PST`) |
| Midday Add-On #2 (`winner_1100`, `_W11`) | **08:00 PST Winner Pyramid (`08:00 PST` / `11:00 ET`, `C3` only):** If `< 3` slots used at `08:00 PST` and an open morning short is up `>= +3.0%` with `<= 40%` bounce off its morning low, short 1 idle slot (`exit 11:00 PST`) |
| Intraday Profit Lock & Disaster Stop | **Trailing Profit Lock (`+5.0%` MFE / `45%` Giveback / `>= +2.0%` Floor)** + **Hard Intraday Stop Loss (`-10.0%`)** |
| Equal-Weight Position Sizing (`active_cap50`) | `min(1 / n_active_slots, 0.50)` — **50.0% per trade** when 1–2 slots trigger; **33.3% per trade** when 3 slots trigger |
| 3-Tier RTH Exit Schedule (PST) | • **`08:30 PST` (`11:30 ET`):** `$50+ local_high` morning shorts (`C1/C2/C3`) & `<15%` moderate-gap morning shorts in `C2`.<br>• **`09:30 PST` (`12:30 ET`):** `<$50 local_high` morning shorts (`C1/C3`, and `>=30%` gaps in `C2`).<br>• **`11:00 PST` (`14:00 ET` — Hard Flat 2h Before Close):** All `bar0_flush`, `_B2` bounce re-shorts, and `_W11` pyramids. |

### Multi-Horizon Equal-Weight Portfolio Backtest (`Active-Slot min(1/k, 50%)` Across All Strategies through `2026-11-06`)

| Strategy | Full 2.33 Years (`28M: 2024-07-01 → 2026-11-06`, `592d`) | Last 12 Months (`2025-11-03 → 2026-11-06`, `255d`) | Last 4 Months (`2026-07-01 → 2026-11-06`, `91d`) | Last 2 Months (`2026-09-01 → 2026-11-06`, `48d`) | Last 2 Weeks (`Oct 23 → Nov 06, 2026`, `11d`) |
|---|---|---|---|---|---|
| **`1. Deployed C2 (Tier1→Tier2)` (LOCKED #1)** | `937t` (`426/592d` = **`72%`**) \| `61.8%` WR \| **`+1.62%/d` all-day** \| **`+962.3%` port** \| **`$195.6M`** \| **Sh `3.07` (#1)** | `460t` (`200/255d` = **`78%`**) \| `67.4%` WR \| **`+3.03%/d` all-day** \| **`+771.9%` port** \| **`$69.7M` (#1)** \| **Sh `5.03` (#1)** | `166t` (`71/91d` = **`78%`**) \| `72.3%` WR \| `+4.41%/d` \| `+401.3%` port \| `$3.11M` \| Sh `6.22` | `84t` (`37/48d` = **`77%`**) \| `69.0%` WR \| `+1.92%/d` \| `+91.9%` port \| `$232,678` \| Sh `5.32` | `24t` (`11/11d` = **`100%`**) \| `70.8%` WR \| **`+1.35%/d`** \| **`+14.8%` port** \| **`$115,359`** \| DD `-4.0%` \| **Sh `7.15`** |
| **`2. C2 + Catalyst Filter`** *(Google News + SEC)* | `877t` (`406/592d` = `68%`) \| `62.8%` WR \| **`+1.68%/d` all-day** \| **`+998.3%` port** \| **`$288.0M`** \| **Sh `3.20`** | `433t` (`190/255d` = `75%`) \| `67.9%` WR \| `+3.01%/d` all-day \| `+767.1%` port \| `$64.9M` \| Sh `4.93` | `153t` (`67/91d` = `74%`) \| **`74.5%` WR** \| **`+4.55%/d`** \| **`+414.2%` port** \| **`$3.48M`** \| **Sh `6.33`** | `79t` (`34/48d` = `71%`) \| `69.6%` WR \| `+1.96%/d` \| `+94.3%` port \| `$238,260` \| **Sh `5.49`** | `22t` (`10/11d` = `91%`) \| **`72.7%` WR** \| **`+1.60%/d`** \| **`+17.6%` port** \| **`$118,717`** \| DD `-4.0%` \| **Sh `9.58`** |
| **`3. Combo C1 (Skip 12–30% + B2)` (LOCKED #2)** | `757t` (`362/592d` = `61%`) \| `60.5%` WR \| `+1.33%/d` all-day \| `+791.5%` port \| `$31.1M` \| Sh `2.45` | `393t` (`176/255d` = `69%`) \| `66.2%` WR \| `+2.74%/d` all-day \| `+698.0%` port \| `$32.6M` \| Sh `4.48` | `158t` (`69/91d` = `76%`) \| `74.1%` WR \| `+4.46%/d` \| `+406.2%` port \| `$3.23M` \| Sh `6.23` | `78t` (`35/48d` = `73%`) \| **`73.1%` WR** \| **`+1.99%/d`** \| **`+95.5%` port** \| **`$239,562`** \| Sh `5.32` | `22t` (`10/11d` = `91%`) \| `68.2%` WR \| `+1.06%/d` \| `+11.6%` port \| `$111,453` \| DD `-8.4%` \| Sh `4.31` |
| **`4. Combo C3 (Skip 12–30% + B2 + Pyr)` (LOCKED #3)** | `817t` (`362/592d` = `61%`) \| `60.2%` WR \| `+1.03%/d` all-day \| `+610.7%` port \| `$6.4M` \| Sh `2.02` | `425t` (`176/255d` = `69%`) \| `66.1%` WR \| `+2.31%/d` all-day \| `+589.7%` port \| `$12.8M` \| Sh `4.08` | `174t` (`69/91d` = `76%`) \| `74.7%` WR \| `+3.90%/d` \| `+354.7%` port \| `$2.13M` \| Sh `6.03` | `86t` (`35/48d` = `73%`) \| `72.1%` WR \| `+1.62%/d` \| `+78.0%` port \| `$205,502` \| Sh `5.13` | `23t` (`10/11d` = `91%`) \| `65.2%` WR \| `+0.95%/d` \| `+10.5%` port \| `$110,187` \| DD `-8.4%` \| Sh `3.91` |
| *`5. C2 Strict (<=$25 Hard Cap)`* | *`519t` (`252/592d` = `42%`) \| `64.4%` WR \| `+1.60%/d` \| `+951.6%` port \| `$102.6M` \| Sh `2.72`* | *`310t` (`144/255d` = `56%`) \| `68.1%` WR \| `+3.06%/d` \| `+780.3%` port \| `$64.2M` \| Sh `4.69`* | *`105t` (`50/91d` = `55%`) \| `73.3%` WR \| `+4.41%/d` \| `+401.2%` port \| `$2.88M` \| Sh `5.73`* | *`40t` (`20/48d` = `42%`) \| `65.0%` WR \| `+1.54%/d` \| `+73.7%` port \| `$192,455` \| Sh `4.00`* | *`7t` (`3/11d` = `27%`) \| `85.7%` WR \| `+1.73%/d` \| `+19.0%` port \| `$120,132` \| Sh `8.51`* |
| *`6. Baseline S1 (Skip 15–30%, No B2)`* | *`550t` (`385/592d` = `65%`) \| `59.6%` WR \| `+1.27%/d` \| `+755.8%` port \| `$45.4M` \| Sh `2.90`* | *`269t` (`182/255d` = `71%`) \| `65.1%` WR \| `+2.22%/d` \| `+567.0%` port \| `$13.3M` \| Sh `4.56`* | *`91t` (`63/91d` = `69%`) \| `69.2%` WR \| `+3.06%/d` \| `+278.3%` port \| `$1.24M` \| Sh `6.28`* | *`49t` (`34/48d` = `71%`) \| `61.2%` WR \| `+0.95%/d` \| `+45.8%` port \| `$151,370` \| Sh `3.48`* | *`17t` (`11/11d` = `100%`) \| `70.6%` WR \| `+1.16%/d` \| `+12.7%` port \| `$113,025` \| Sh `6.42`* |
| *`7. Baseline S6 (Skip 12–30%, No B2)`* | *`433t` (`324/592d` = `55%`) \| `58.4%` WR \| `+0.93%/d` \| `+553.7%` port \| `$6.8M` \| Sh `2.23`* | *`219t` (`155/255d` = `61%`) \| `64.8%` WR \| `+1.96%/d` \| `+498.5%` port \| `$6.9M` \| Sh `4.09`* | *`83t` (`61/91d` = `67%`) \| `66.3%` WR \| `+3.10%/d` \| `+281.9%` port \| `$1.27M` \| Sh `6.25`* | *`40t` (`32/48d` = `67%`) \| `57.5%` WR \| `+0.95%/d` \| `+45.8%` port \| `$151,200` \| Sh `3.43`* | *`13t` (`10/11d` = `91%`) \| `61.5%` WR \| `+0.70%/d` \| `+7.7%` port \| `$107,186` \| Sh `2.92`* |

---

## Change Log

| Version | Date | Change | Backtest Result |
|---|---|---|---|
| 1.0 | 2026-09-18 | Initial E2 lock | +14.71% ref |
| 1.1 | 2026-09-18 | Add -10% SL, remove skip rule | SL: +$303K (FLAWED — post-hoc cap, not real intraday exit) |
| 1.2 | 2026-09-18 | SL -10%→-50% after real eval | Real intraday SL eval: -10% costs -$382K (kills 21 recoveries). -50% adds +$53K (saves 3 disasters, kills 1). Sharpe 0.435→0.510. |
| 1.3 | 2026-09-21 | AH freshness filter | Gap calc vs last AH price, not prior close. Catches stale M&A gaps (ACVA: daily +44.6% but fresh +0.6% → excluded). |
| 1.4 | 2026-09-21 | Gap erosion filter (50%) | Skip if >50% of gap consumed by scan time. Catches faded gaps (TANH: 55% consumed → excluded). |
| 1.5 | 2026-09-21 | TOP_N 1→5, GAP_MIN 5%→3% | Wider candidate pool; E2-A time gate removed (runs at any time). |
| 2.0 | 2026-10-13 | Initial `agent1_v2.py` launch | 5-strategy suite (`C5, C3+C5, C7, C9, C3`), local-high only (no fallback), 17:00 ET exit. |
| 2.1 | 2026-10-28 | Leak-free pipeline + `S6/S1/S12` | Fixed 7 live-vs-BT pipeline divergences (`_FLOAT_DATA` freeze, pre-slice CS filtering, timestamp bounds in `find_local_high`, entry-time gap erosion, dual `raw_open_gap` check). Retired `C5/C7/C9` and adopted `S6`, `S1`, `S12`. |
| 2.2 | 2026-10-29 | Hybrid `Combo C1/C2/C3` + `CLMB` Guard + PST Playbook | Added moderate-gap quality guard (`<=1` missing bar & `<7%` post-open runup when `gap < 15%`), Bar 0 Opening Flush (`06:33 PST`), Continue-Loop on shallow PB, Midday Bounce Re-Short (`_B2`, `06:55–10:35 PST`), 08:00 PST Winner Pyramid (`_W11`), `08:30 PST` moderate-gap exit (`C2`), `active_cap50` sizing, and interactive PST Operator Playbook. |
| 2.3 | 2026-11-02 | 12M Walk-Forward Audit + 3-Tier RTH Exits + SPAC Filter + Profit Lock | Replaced `14:00 PST` AH exit with **3-Tier RTH Exit Schedule (`08:30 PST` `$50+ local_high` & `C2 <15%`, `09:30 PST` `<$50 local_high`, `11:00 PST` `bar0_flush` / `_B2` / `_W11`)**, added SPAC/Warrant filter (`ACQUISITION / BLANK CHECK / SPAC`), and added Intraday Trailing Profit Lock (`+5% MFE / 45% Giveback / >=+2% Floor`) + `-10%` Hard Stop Loss. |
| 2.4 | 2026-11-06 | Priority Tier Ranking (`Tier 1 → Tier 2`), 2.33Y Equal-Weight Portfolio Backtest & Google News + SEC Catalyst Study | Replaced hard `<=$25` moderate-gap cap with **Priority Tier Ranking (`Tier 1: >=30% gap or <15% retail <=$25 & <$25M dvol` ranked ahead of `Tier 2: <15% inst/mid-cap`)**, preserving `72%` active-day deployment over 2.33 years (`+962.3%` non-compounded portfolio return, `$195.6M` compounded, Sharpe `3.07` vs `2.45` for `C1`). Verified 2.33Y Google News RSS + SEC EDGAR Catalyst Filter (`+998.3%` portfolio return, `$288.0M` compounded, Sharpe `3.20`). |
| 2.5 | 2026-11-09 | Point-in-Time Unadjusted Price Audit (`$10–$100 Unadj`) & Upgraded `T&Q` (`+8 Orthogonal Features`) Benchmark on `142` `T&Q` Stocks | Audited all `142` `T&Q` stocks for reverse-split penny-stock illusions (`26` stocks with `8x–60,000x` reverse splits). On the clean point-in-time `$10–$100 Unadj` `T&Q` subset (`89` stocks, `506` trading days), `Deployed C2 + Catalyst` achieves `+40.6%` (`$138,210`, `0.69` Sharpe), while `M7 Deep GBDT ($10–$100 Unadj): Upgraded (+8 Orthogonal Features)` achieves **`+63.9%` (`$164,654` Compounded Equity, `63.0%` WR across `316` trades, `-29.0%` Max DD, `0.85` Sharpe)** over 24M and **`+55.3%` (`$159,797`, `66.2%` WR, `1.33` Sharpe)** over the Last 12 Months (`+1.6%` / `100%` WR in Last 2 Weeks). |
| 2.6 | 2026-11-10 | `157`-Stock `T&Q` Expansion, 4-Column Sub-Minute Tick Audit (`Odd_Lot_OFI - Block_OFI`) & **3-Pillar Synergy Architecture (`SYNERGY 2` & `SYNERGY 4A`)** | Expanded `T&Q` dataset to `157` stocks (`139` with 16-col `Microstructure_Advanced`) and proved that `Odd_Lot_OFI - Block_OFI` achieves `r = +0.112 (p=0.036*)` and `+1.77%/tr` `Q3-Q1` spread. Built the **3-Pillar Synergy Architecture** combining 3-bar C2 structural mechanics, Pre-Open Catalyst veto, and domain-trained walk-forward `T&Q` ML (`M7+M12+M4+ORTH_Z` trained on `2,619` clean setups): **`SYNERGY 2` (Gatekeeper Mode)** vetoes `37` toxic morning trades (`-19.9%` sum) and `22` bearish `_B2` bounces (`-22.0%` sum), lifting avg trade `+1.55% → +2.16%`, 24M equity `$309,939 → $365,672`, and cutting Max DD `-38.1% → -30.8%` (`1.57` Sharpe); **`SYNERGY 4A` (Full 3-Pillar + Non-Diluting Kelly 35% Upgraded `T&Q` Slot-Fill)** reaches **`$437,129` (`+175.8%` Tot PnL, `60.8%` WR, `1.59` Sharpe)** over 24M OOS (`+$127,190` ahead of `Deployed C2 + Catalyst`), **`$238,228` (`+102.2%`, `1.79` Sharpe)** over Last 12M, **`88.9%` Day WR (`1.89` Sharpe)** over Last 2M, and **`+2.3%` (`$102,291`, `0.0%` DD, `4.79` Sharpe)** in Last 2W. |

