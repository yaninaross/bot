# Agent 1 — SHORT Gap-Fade Strategy

## What This Agent Does
Pure technical SHORT strategy. No ML, no transformers, no LLM.
Runs at 6:10 AM PST. Scans full US equity universe for gap-up
stocks. Shorts the fade. Holds T+2. Completely independent of
the bot-RAG system in the parent directory.

## Frozen Baseline: CS+SD+GUARDS
**DO NOT change these parameters without explicit instruction.**
Changing any of these invalidates all prior backtest comparisons.

  Universe:    _FLOAT (217 tickers, float_cache.json)
  Gap filter:  gap_pct >= 3% (gap-UP vs AH reference price)
  Gap calc:    gap = (open - ah_ref) / ah_ref, where ah_ref is
               the last after-hours price (16:00-20:00 ET) on
               the prior trading day. Falls back to prior close
               if no AH data. Filters stale M&A/overnight gaps.
  Gap erosion: if >50% of gap consumed by scan time, SKIP.
               gap_consumed = (open - low_to_scan) / (open - ref)
               Uses intraday LOW (not close) through SCAN_TIME_ET.
  Asset type:  CS only — exclude ETF, ETN, ADRC, FUND, PFD, UNIT
  Pattern:     sustained_drop (classified at bar 60 = 10:30 AM ET)
               rise_30m < -3% AND recovery_at_bar60 < 50%
  Entry:       local high (3-bar lookback + 3-bar confirmation)
               bar i is local high if:
                 high[i] > high[i-1], high[i-2], high[i-3]
                 AND high[i] > high[i+1], high[i+2], high[i+3]
               entry_price = close[i+3]
               scan bars index 3 through 11 (9:33-9:41 AM ET)
  Fallback:    if no local high by bar 11 → enter at close[15]
               (9:45 AM ET bar close)
  Guards:      price >= $10.00
               float >= 10M shares (HARD EXCLUDE on cache miss —
               if ticker not in float_cache, fetch from Polygon
               reference; if fetch fails, EXCLUDE)
               gap_pct <= 75% (exclude extreme gaps)
  Skip rule:   if (open - bar15_close) / open > 0.15 → SKIP
               (stock down >15% from open by 9:45 AM ET)
  Hold:        T+2 close (2 trading days after entry, skip weekends)
  TOP_N:       5 (select top 5 by gap_pct descending)
  Stop loss:   None
  PnL:         (entry_price - t2_close) / entry_price * 100
               Positive = profit (stock fell)
  Reference:   avg +12.09%, WR 78%, Sharpe 0.254,
               CI [+7.82, +16.36], n=476

## Known Bugs — Already Fixed — Do NOT Re-introduce
1. Float filter: cache miss was INCLUDE → MUST BE EXCLUDE
2. pm_retrace denominator: use (pm_high - pm_open),
   NOT (pm_high - regular_session_open)
3. FALLBACK_ONLY was True globally → must be False for baseline
4. OOS validation: must use actual filtered subset trades,
   NOT baseline trades for all variants
5. Gap consumed at entry: two separate skip logics exist —
   the 15% absolute drop AND the gap-fill % check.
   Do not merge them.
6. Gap vs prior close missed stale gaps: M&A stocks (e.g. ACVA)
   that moved in AH showed huge daily gap but zero fresh gap.
   Fixed: gap now computed vs last AH price on prior trading day.
7. Gap erosion not checked: stocks like TANH where gap already
   faded by scan time were recommended. Fixed: skip if >50% of
   gap consumed (intraday low vs open) by SCAN_TIME_ET.

## New Skip Logic Being Tested (E-GAP-FILL)
In addition to the 15% absolute skip rule, test:
  gap_consumed = (open_price - entry_price) / (open_price - prior_close)
  skip if gap_consumed > threshold (test 0.50, 0.60, 0.70, 0.75, 0.80)
This catches stocks like HELP/ONON where the gap was nearly
fully consumed before/at entry even though absolute drop < 15%.

## Experiment Queue (run in order, one per session)
| ID       | Description                    | Status   | Results file |
|----------|--------------------------------|----------|--------------|
| E1       | CS+SD+GUARDS verify            | COMPLETE | results/E1   |
| E2       | N×Hold grid (N=1..5, T=2..5)  | PENDING  |              |
| E3       | Hold curve (30m to T+10)      | PENDING  |              |
| E4       | Sector excl Tech+Manufacturing | PENDING  |              |
| E5       | SD × TOP_N interaction        | PENDING  |              |
| E6       | Prior day pattern proxy       | PENDING  |              |
| E-GAP-FILL | Gap consumed skip filter    | PENDING  |              |

## Output Format (required for every experiment)
Every experiment must write a JSON file to results/ containing:
  n, tpd, avg_pnl, median_pnl, trimmed_mean, std,
  win_rate, sharpe, profit_factor, max_loss, max_gain,
  ci_95_low, ci_95_high, ci_confirmed,
  is_n, is_avg, is_sharpe, is_ci_low,
  oos_n, oos_avg, oos_sharpe, oos_ci_low,
  degradation_pct,
  worst_5_trades, best_5_trades,
  monthly_breakdown

## Session Rules
- One experiment per session. Complete it, write results, stop.
- Never modify baseline files — create new files for experiments
- Always verify guards are working: print n_excluded_by_float,
  n_excluded_by_price, n_excluded_by_gap_max at start of each run
- If PAVS, WOK, ELAB, VCIG appear in results → guards not working
- Print entry type breakdown (local_high vs fallback) for every run

## Agent 1 v2.4 — Multi-Strategy Hybrid SHORT Scanner (`agent1_v2.py`)
**Status: ACTIVE — Upgraded 2026-11-06 (`Deployed C2 (Tier1→Tier2)`, `Combo C1`, `Combo C3` + 3-Tier RTH Exits + Priority Tier Ranking)**

Three high-grossing hybrid multi-trigger SHORT strategies sharing a strictly
leak-free, causal base pipeline:
- **Base Universe & Guards:** Full US equity universe (`gap_pct >= 3.0%` vs
  prior AH reference price, `$10.00 <= open_price <= $100.00`, frozen 217-ticker
  `_FLOAT` dict where `< 10M` float = EXCLUDE and dict miss = INCLUDE,
  Common Stock `type == 'CS'` only).
- **Pre-Filter Hygiene & SPAC/Warrant Filter (Updated 2026-11-02):** Known non-CS
  ETFs, leveraged ETNs, warrants/units/rights (`W/WS/U/R`, `.WS`, `.WT`), cached
  non-CS tickers in `ticker_type_cache.json`, and **SPAC / Blank-Check shells**
  (`ACQUISITION`, `BLANK CHECK`, `SPAC` in company name) are filtered **BEFORE**
  slicing `PRE_FILTER_N = 30`.
- **Priority Tier Ranking (`USE_PRIORITY_TIER_RANKING = True`, Added 2026-11-04):**
  Instead of hard-dropping `$25–$100` or `>=$25M` prior-day dollar volume stocks
  on `<15%` moderate gaps (which left 38% of days idle), candidates are ranked
  by **`(priority_tier, gap_pct)` descending**:
  1. **Tier 1 (`Priority 1` — Highest Priority):** Parabolic gaps (`gap_pct >= 30%`)
     AND retail moderate gaps (`gap_pct < 15%` with `open_price <= $25.00` AND
     `prior_dvol < $25M`).
  2. **Tier 2 (`Priority 0` — Fallback Fill Only):** Institutional/mid-cap
     moderate gaps (`gap_pct < 15%` with `open_price > $25.00` OR `prior_dvol >= $25M`),
     used only when `< 3` Tier 1 candidates qualify on that day.
- **Gap Freshness & Erosion (Fixed 2026-10-28):**
  1. `gap_pct` recomputed vs last AH bar (`16:00–20:00 ET`) on prior day (`>= 3%`).
  2. Scan-time erosion (`06:30–06:31 PST` / `09:30–09:31 ET`): skip if
     `gap_consumed > 0.50` using intraday low vs `ah_ref_price`.
  3. Entry-time erosion (`06:30 PST–entry_bar`): skip if `entry_price` has already
     consumed `> 50%` of the gap above `ah_ref_price`.
- **Moderate-Gap Quality Guard (Added 2026-10-29):**
  When `gap_pct < 15%`, requires:
  1. `MAX_MISSING_BARS_MOD_GAP = 1` (`<= 1` missing 1-min bar prior to entry), AND
  2. `MAX_OPEN_RUNUP_MOD_GAP = 0.07` (post-open run-up `(entry_price - open_price) / open_price < +7.0%`).
- **Morning Entry Window (`06:33–06:44 PST` / `09:33–09:44 ET`, `find_local_high`):**
  1. **Bar 0 Opening Flush (`bar0_flush` @ `06:33 PST`):** If bar 0 (`06:30 PST`)
     is red (`close <= open`), its high holds unbroken across bars 1–3 (`06:31–06:33 PST`),
     and bar 3 closes `>= 1.0%` below open while `>= 35%` of the gap remains
     unfilled, enter short at `06:33 PST`.
  2. **3-Bar Local High (`local_high` @ `06:36–06:44 PST`):** 3-bar lookback +
     3-bar confirmation across `bars 3..11` (`peak_bar <= 06:41 PST`,
     `entry_bar <= 06:44 PST`), requiring `MIN_PULLBACK_PCT = 0.5%`.
  3. **Continue-Loop on Shallow Pullback (`continue_on_failed_pb = True`):** If an
     early local high has `< 0.5%` pullback, continue scanning later bars up to
     `06:41 PST` for the true exhaustion peak.
- **Intraday Add-On Legs Into Idle Slots (`n_active_slots < 3`):**
  1. **Midday Bounce Re-Short (`_B2`, `06:55–10:35 PST` / `09:55–13:35 ET`, `C1/C2/C3`):**
     If a confirmed morning short drops `>= +3.0%` MFE, then bounces `>= +1.5%` off
     its intraday low to form a 3-bar local high (`>= 0.5%` pullback) while the
     primary trade is still `>= +0.5%` in profit, deploy 1 idle slot into a
     secondary short leg (`_B2`, exit `11:00 PST`).
  2. **08:00 PST Winner Pyramid (`_W11`, `08:00 PST` / `11:00 ET`, `C3` only):**
     If an idle slot still remains at `08:00 PST` and an open morning short is
     up `>= +3.0%` while trading within `<= 40%` of its morning low, deploy 1
     idle slot into a pyramid short leg (`_W11`, exit `11:00 PST`).
- **Intraday Profit Lock & Disaster Stop (Added 2026-11-02):**
  1. **Trailing Profit Lock (`+5.0%` MFE / `45%` Giveback / `>= +2.0%` Floor):** Once a
     short reaches `>= +5.0%` unrealized gain, cover if it retraces `>= 45%` of peak
     MFE while still locking in `>= +2.0%`.
  2. **Hard Intraday Stop Loss (`-10.0%`):** Cover immediately if a trade moves
     `-10.0%` against entry intraday.
- **Equal-Weight Active Slot Sizing (`active_cap50`):** Allocates `min(1 / n_active_slots, 50%)`
  per confirmed trade (**`50.0%` each** when 1–2 slots trigger; **`33.3%` each** when all 3 slots trigger).
- **3-Tier RTH Exit Schedule (`08:30 / 09:30 / 11:00 PST`, Upgraded 2026-11-02):**
  - **`08:30 PST` (`11:30 ET`):** Cover `$50+` higher-priced `local_high` morning spikes (`C1/C2/C3`) AND all `<15%` moderate-gap morning shorts in `C2`.
  - **`09:30 PST` (`12:30 ET`):** Cover `<$50` `local_high` morning spikes (`C1/C3`, plus `>=30%` gaps in `C2`).
  - **`11:00 PST` (`14:00 ET` — Hard Flat 2h Before Close):** Cover all `bar0_flush` opening flushes, `_B2` midday bounce re-shorts, and `_W11` `08:00 PST` pyramids. Never hold into the `11:30–13:00 PST` late-day squeeze window or After-Hours.

### Multi-Horizon Equal-Weight Portfolio Backtest (`Active-Slot min(1/k, 50%)` through `2026-11-06`, `$100K` Start)

| Strategy | Full 2.33 Years (`28M: Jul 2024–Nov 2026`, `592d`) | Last 12 Months (`Nov 2025–Nov 2026`, `255d`) | Last 4 Months (`Jul–Nov 2026`, `91d`) | Last 2 Months (`Sep–Nov 2026`, `48d`) | Last 2 Weeks (`Oct 23–Nov 06`, `11d`) |
|---|---|---|---|---|---|
| **`1. Deployed C2 (Tier1→Tier2)`** | `937t` (`72%` act) \| `61.8%` WR \| **`+1.62%/d`** \| **`+962.3%` port** \| **`$195.6M`** \| **Sh `3.07` (#1)** | `460t` (`78%` act) \| `67.4%` WR \| **`+3.03%/d`** \| **`+771.9%` port** \| **`$69.7M` (#1)** \| **Sh `5.03` (#1)** | `166t` (`78%` act) \| `72.3%` WR \| `+4.41%/d` \| `+401.3%` port \| `$3.11M` \| Sh `6.22` | `84t` (`77%` act) \| `69.0%` WR \| `+1.92%/d` \| `+91.9%` port \| `$232,678` \| Sh `5.32` | `24t` (`100%` act) \| `70.8%` WR \| **`+1.35%/d`** \| **`+14.8%` port** \| **`$115,359`** \| **Sh `7.15`** |
| **`2. C2 + Catalyst Filter`** *(Google News + SEC)* | `877t` (`68%` act) \| `62.8%` WR \| **`+1.68%/d`** \| **`+998.3%` port** \| **`$288.0M`** \| **Sh `3.20`** | `433t` (`75%` act) \| `67.9%` WR \| `+3.01%/d` \| `+767.1%` port \| `$64.9M` \| Sh `4.93` | `153t` (`74%` act) \| **`74.5%` WR** \| **`+4.55%/d`** \| **`+414.2%` port** \| **`$3.48M`** \| **Sh `6.33`** | `79t` (`71%` act) \| `69.6%` WR \| `+1.96%/d` \| `+94.3%` port \| `$238,260` \| **Sh `5.49`** | `22t` (`91%` act) \| **`72.7%` WR** \| **`+1.60%/d`** \| **`+17.6%` port** \| **`$118,717`** \| **Sh `9.58`** |
| **`3. Combo C1 (Skip 12–30% + B2)`** | `757t` (`61%` act) \| `60.5%` WR \| `+1.33%/d` \| `+791.5%` port \| `$31.1M` \| Sh `2.45` | `393t` (`69%` act) \| `66.2%` WR \| `+2.74%/d` \| `+698.0%` port \| `$32.6M` \| Sh `4.48` | `158t` (`76%` act) \| `74.1%` WR \| `+4.46%/d` \| `+406.2%` port \| `$3.23M` \| Sh `6.23` | `78t` (`73%` act) \| **`73.1%` WR** \| **`+1.99%/d`** \| **`+95.5%` port** \| **`$239,562`** \| Sh `5.32` | `22t` (`91%` act) \| `68.2%` WR \| `+1.06%/d` \| `+11.6%` port \| `$111,453` \| Sh `4.31` |
| **`4. Combo C3 (Skip 12–30% + B2 + Pyr)`** | `817t` (`61%` act) \| `60.2%` WR \| `+1.03%/d` \| `+610.7%` port \| `$6.4M` \| Sh `2.02` | `425t` (`69%` act) \| `66.1%` WR \| `+2.31%/d` \| `+589.7%` port \| `$12.8M` \| Sh `4.08` | `174t` (`76%` act) \| `74.7%` WR \| `+3.90%/d` \| `+354.7%` port \| `$2.13M` \| Sh `6.03` | `86t` (`73%` act) \| `72.1%` WR \| `+1.62%/d` \| `+78.0%` port \| `$205,502` \| Sh `5.13` | `23t` (`91%` act) \| `65.2%` WR \| `+0.95%/d` \| `+10.5%` port \| `$110,187` \| Sh `3.91` |
| **`5. C2 Strict (<=$25 Hard Cap)`** | `519t` (`42%` act) \| **`64.4%` WR** \| `+1.60%/d` \| `+951.6%` port \| `$102.6M` \| Sh `2.72` | `310t` (`56%` act) \| **`68.1%` WR** \| **`+3.06%/d`** \| **`+780.3%` port** \| `$64.2M` \| Sh `4.69` | `105t` (`55%` act) \| `73.3%` WR \| `+4.41%/d` \| `+401.2%` port \| `$2.88M` \| Sh `5.73` | `40t` (`42%` act) \| `65.0%` WR \| `+1.54%/d` \| `+73.7%` port \| `$192,455` \| Sh `4.00` | `7t` (`27%` act) \| `85.7%` WR \| `+1.73%/d` \| `+19.0%` port \| `$120,132` \| Sh `8.51` |

### Daily Multi-Trigger Run Schedule (PST)
- **`06:31–06:32 PST` (Once at `06:31:15 PST`):** Locks top 3 gap candidates per strategy (`C1`, `C2`, `C3`) using `Tier 1 → Tier 2` priority ranking. Do not short yet.
- **`06:33–06:44 PST` (Every 1–2 min):** Confirms `Bar0 Flush` (`06:33 PST`) and `3-Bar Local High` shorts (`06:36–06:44 PST`). Unconfirmed candidates at `06:44:30 PST` are permanently `SKIPPED`.
- **`06:55–10:35 PST` (Every 3–5 min, ONLY if 1–2 morning shorts triggered):** Watches open morning shorts for `_B2` Midday Bounce Re-Short into idle slots.
- **`08:00 PST` (Once at `08:00:15 PST`, ONLY if `C3` has an idle slot):** Checks `_W11` Winner Pyramid.
- **`08:30 PST` (Action Checkpoint #1):** Cover `$50+` `local_high` morning shorts AND `C2` morning shorts with `gap < 15%`.
- **`09:30 PST` (Action Checkpoint #2):** Cover `<$50` `local_high` morning shorts (`C1/C3` and `>=30%` gaps in `C2`).
- **`11:00 PST` (Action Checkpoint #3 — Hard Flat RTH Exit):** Cover all remaining open short positions (`bar0_flush`, `_B2` bounce re-shorts, `_W11` pyramids) 2 hours before market close.

## File Structure
  agent1.py                                main scanner (live trading, E2-A + E2-B)
  agent1_v2.py                             multi-strategy hybrid scanner (C1, C2, C3; upgraded 2026-11-06 to v2.4)
  agent1_v2_backtest.py                    multi-horizon backtest + 6-panel revenue dashboard
  agent1_v2_backtest.png                   6-panel cumulative PnL & compounded revenue chart
  all_portfolio_strategies_comparison.png  4-panel 2.33Y / 12M / 4M / 2M equal-weight portfolio comparison
  fundamental_catalyst_analysis.png        4-panel 2.33Y Google News RSS + SEC EDGAR catalyst study chart
  cyclicality_and_monotonicity_28m.png     2.33Y quarterly/monthly cyclicality & monotonicity chart
  agent1_12m_walkforward_audit.png         12-month walk-forward audit chart
  agent1_v2_last_2_weeks.png               last 2 weeks equal-weight portfolio chart
  agent1_v2_bt_comparison.py               comparison backtest script
  agent1_backtest.py                       v1 backtest framework
  agent1_trade_review.py                   N sweep for strategies (backtest)
  agent1_feature_investigation.py          feature/filter ranking analysis
  experiments/                             one new .py file per experiment
  results/                                 JSON results, one per experiment
  data/agent1.db                           SQLite trade log
  data/float_cache.json                    {ticker: float_shares} — used by v1 only (v2 uses frozen _FLOAT_DATA)
  data/ticker_type_cache.json              {ticker: asset_type} — persistent Polygon CS cache for v2

### Point-in-Time Unadjusted Price Guard & 3-Pillar `T&Q` Synergy Architecture (Updated 2026-11-10)
- When benchmarking `Deployed C2` against `Trades & Quotes (T&Q)` ML models on the `157`-stock `T&Q` subset (`139` with 16-column `Microstructure_Advanced`), always enforce **point-in-time unadjusted open price (`10.0 <= unadj_open_px <= 100.0`, `split_factor <= 1.25`)** and restore unadjusted percentage spreads (`tq_spread_pct * split_factor`). Without this guard, `29` reverse-split penny stocks (`VIVK 60,000x`, `SDOT 2,000x`, `AZI 500x`, `BIAF 450x`, `VEEE 370x`, `BIYA 250x`, `AKAN 140.6x`, `MASK 100x`, `EHGO 80x`, `WETO 80x`, `VRAX 35x`, `OPTT/RVSN/IXHL 30x`) contaminate both `build_trades_v2` and the ML training/test folds.
- **Why Standalone 1-Bar `T&Q` ML Lagged Rule-Based C2 & How the 3-Pillar Synergy Architecture (`deep_synergy_forensics_and_architectures.py`) Solves It:**
  1. **Domain-Pool Transfer (`N = 2,619` vs. `N = 189`):** Walk-forward `T&Q` models (`M7` Monotonic GBDT, `M12` Deep-18 Ensemble, `M4` Huber-Ridge, and `ORTH_Z`) are trained on the full `2,619`-setup clean `$10–$100` domain pool (`1,000+` training setups/fold) and applied out-of-sample (`GRAND_SYN`) to gate, size, and augment 3-bar confirmed `Deployed C2 + Catalyst` setups.
  2. **Out-of-Sample Forensic Separation on `189` `Deployed C2 + Catalyst` Trades:**
     - **Toxic Morning Order-Flow Veto (`M7 < -0.30` and `M12 < -0.30`):** Identifies **`37` toxic trades (`-0.54%/trade` avg, `-19.9%` sum)**.
     - **Bearish-Flow `_B2` Midday Bounce Veto (`is_b2` and `GRAND_SYN < -0.10`):** Identifies **`22` losing bounce re-shorts (`-1.00%/trade` avg, `-22.0%` sum)** while keeping **`46` `T&Q`-confirmed `_B2` bounces (`+2.15%/trade` avg, `+99.0%` sum)**.
  3. **Multi-Horizon Walk-Forward OOS Results Across `157` `T&Q` Stocks (`$10–$100 Unadj`, `$100K` Compounded, `tq_3pillar_synergy_backtest.png`):**
     - **Baseline `Deployed C2 + Catalyst`:** `189t` (`20%` act) | `57.7%` WR | `+1.55%` avg | `+135.1%` Tot PnL | **`$309,939` (24M)** | **`$182,823` (12M)** | `-38.1%` DD | `1.37` Sharpe (`1.42` 12M).
     - **`SYNERGY 2` (`C2 + Morning & _B2 T&Q Veto` — High-Precision Gatekeeper):** `146t` (`16%` act) | `58.2%` WR | **`+2.16%` avg (`+39%` lift)** | `+150.2%` Tot PnL | **`$365,672` (24M, `+$55.7K`)** | **`-30.8%` DD (`-7.3%` lower DD)** | **`1.57` Sharpe**.
     - **`SYNERGY 4A` (`Full 3-Pillar: SYNERGY 2 + Unlocked 15–30% Gaps + Non-Diluting Kelly 35% Upgraded T&Q Slot-Fill` — #1 Overall Winner):** **`401t` (`51%` act)** | **`60.8%` WR** | **`+175.8%` Tot PnL** | **`$437,129` (24M, `+$127,190` / `+41%` ahead of C2, Sharpe `1.59` #1)** | **`$238,228` (12M, `+102.2%`, Sharpe `1.79` #1 vs `1.42` for C2)** | **`88.9%` Day WR (`8W–1L`, Sharpe `1.89`) in Last 2M** | **`+2.3%` (`$102,291`, `0.0%` DD, Sharpe `4.79`) in Last 2W**.

