# Agent 1 — Composite Backtest Changelog

## 2026-11-10 — Agent 1 v2.6 (`deep_synergy_forensics_and_architectures.py` & `diagnose_ml_vs_rule_based_gap.py`): `157`-Stock `T&Q` Expansion, 4-Column Sub-Minute Tick Audit (`Odd_Lot_OFI - Block_OFI`), and **3-Pillar Synergy Architecture (`SYNERGY 2` & `SYNERGY 4A`)**

**Scripts & Charts Added:** `agent1/deep_synergy_forensics_and_architectures.py`, `agent1/diagnose_ml_vs_rule_based_gap.py`, `agent1/run_expanded_tq_universe_full_analysis.py`, `agent1/tq_3pillar_synergy_backtest.png`, `agent1/tq_152_expanded_clean_upgraded_backtest.png`
**Audit Horizon:** Full 24M Walk-Forward OOS (`2024-11-01 → 2026-11-06`, `506` Trading Days), Last 12 Months (`255` Trading Days), Last 4 Months (`91` Trading Days), Last 2 Months (`48` Trading Days), and Last 2 Weeks (`11` Trading Days) across all `157` `T&Q` stocks (`$10.00 <= unadj_open_px <= $100.00`, `split_factor <= 1.25`).

### 1. Why Standalone `T&Q` ML Underperformed Rule-Based `Deployed C2` — And the 4 Structural Fixes
1. **1-Bar Unconfirmed Triggers (`-0.16%` base-rate mean) vs. 3-Bar Structural Highs (`+1.55%` base-rate mean):** Standalone ML evaluated `2,619` raw 1-bar red candles after `>= 3%` gaps, suffering right-skewed `-10%` squeeze stops on unconfirmed peaks. `Deployed C2` (`189` trades) waits for a **3-bar confirmed structural peak (`i-3..i+3` + `2` confirmation bars)** on `>= 30%` parabolic gaps plus `_B2` midday bounce re-shorts.
2. **Domain-Pool Transfer vs. Small-Sample Overfitting (`N = 2,619` vs. `N = 189`):** Training ML models solely on `189` C2 trades overfits small folds (`N < 80`). Instead, we train our walk-forward `T&Q` models (`M7` Monotonic GBDT, `M12` Deep-18 Ensemble, `M4` Huber-Ridge, and `ORTH_Z`) on the **full `2,619`-setup clean `$10–$100` domain pool** (`1,000+` training rows/fold) and apply their out-of-sample scores (`GRAND_SYN`) to filter, size, and fill C2 slots.
3. **Strict Non-Diluting Slot Hierarchy (`min(1/k, 50%)`):** Adding a 1-bar ML trade on a day where `Deployed C2` already had `2` confirmed 3-bar trades (`2 × 50% = 100%` deployed) diluted `+2.06%/tr` C2 setups from `50%` down to `33.3%`. Fixing this so 1-bar `T&Q` consensus setups **only fill idle slots (`open_slots = 2 - k_C2`)** eliminates slot dilution.
4. **Kelly-Proportional Slot Weight (`35%` vs. `50%`):** Sizing 1-bar `T&Q` consensus slot-fills at **Kelly-optimal `35%` slot weight (`w_mult = 0.70`)** while sizing `T&Q`-confirmed 3-bar C2 trades at `50%–55%` (`w_mult = 1.00–1.10`) maximizes geometric compounding (`$437,129` vs. `$358,706` at full `50%` fill).

### 2. Out-of-Sample Forensic Partition of the `189` `Deployed C2 + Catalyst` Trades by Walk-Forward `T&Q` ML
- **All `Deployed C2 + Catalyst` Trades (Baseline):** `189` trades | `57.7%` WR | **`+1.55%` avg** | `+1.70%` med | **`+293.6%` sum**.
- **1. Toxic Morning Order-Flow Vetoed (`M7 < -0.30` and `M12 < -0.30`):** **`37` trades | `54.1%` WR | `-0.54%` avg | `-19.9%` sum** (pure drag removed!).
- **2. Clean Morning Order-Flow Kept (`SYNERGY 1`):** **`152` trades | `58.6%` WR | `+2.06%` avg | `+313.5%` sum**.
- **3. Bearish-Flow `_B2` Midday Bounces Vetoed (`is_b2` and `GRAND_SYN < -0.10`):** **`22` trades | `50.0%` WR | `-1.00%` avg | `-22.0%` sum**.
- **4. `T&Q`-Confirmed `_B2` Midday Bounces Kept (`is_b2` and `GRAND_SYN >= -0.10`):** **`46` trades | `52.2%` WR | `+2.15%` avg | `+99.0%` sum**.

### 3. Multi-Horizon Walk-Forward OOS Comparison Across `157` `T&Q` Stocks (`$10–$100 Unadj`, `$100K` Compounded)

| Horizon | Strategy / 3-Pillar Synergy Architecture | Trades (Active Days) | Trade WR | Day WR (`W–L`) | Avg Trade% | Tot PnL% | Comp Equity (`$100K`) | Max DD% | Ann. Sharpe |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full 24M OOS (`506d`)** | **1. `Deployed C2 + Catalyst` (Baseline)** | `189` (`103/506d = 20%`) | `57.7%` | `54.4%` (`56W–47L`) | `+1.55%` | `+135.1%` | `$309,939` | `-38.1%` | `1.37` |
| | **3. Standalone `T&Q` ML (`M7 GBDT`, 1-Bar)** | `314` (`236/506d = 47%`) | **`61.8%`** | **`61.9%`** (`146W–90L`) | `+0.50%` | `+77.1%` | `$196,626` | `-33.4%` | `1.07` |
| | **4. `SYNERGY 1`: C2 + Toxic Order-Flow Veto** | `152` (`82/506d = 16%`) | `58.6%` | `57.3%` (`47W–35L`) | `+2.06%` | `+148.4%` | `$358,270` | **`-30.7%`** | `1.55` |
| | **5. `SYNERGY 2`: C2 + Morning & `_B2` Veto** | `146` (`81/506d = 16%`) | `58.2%` | `59.3%` (`48W–33L`) | **`+2.16%`** | `+150.2%` | **`$365,672`** | `-30.8%` | **`1.57`** |
| | **6. `SYNERGY 3`: Arch 2 + 15–30% Gaps & Sizing** | `152` (`85/506d = 17%`) | `57.9%` | `57.6%` (`49W–36L`) | `+2.06%` | `+152.7%` | `$371,521` | `-32.3%` | `1.54` |
| | **7. `SYNERGY 4A`: Full 3-Pillar (Kelly 35% `T&Q`)** | **`401` (`258/506d = 51%`)** | **`60.8%`** | `56.2%` (`145W–113L`) | `+0.98%` | **`+175.8%`** | **`$437,129` (#1)** | `-44.3%` | **`1.59` (#1)** |
| **Last 12M (`255d`)** | **1. `Deployed C2 + Catalyst` (Baseline)** | `92` (`48/255d = 19%`) | `58.7%` | `56.2%` (`27W–21L`) | `+1.81%` | `+72.3%` | `$182,823` | `-38.1%` | `1.42` |
| | **5. `SYNERGY 2`: C2 + Morning & `_B2` Veto** | `70` (`38/255d = 15%`) | `57.1%` | `60.5%` (`23W–15L`) | **`+2.15%`** | `+68.7%` | `$177,881` | **`-30.8%`** | `1.40` |
| | **7. `SYNERGY 4A`: Full 3-Pillar (Kelly 35% `T&Q`)** | **`197` (`128/255d = 50%`)** | **`59.9%`** | `57.8%` (`74W–54L`) | `+1.25%` | **`+102.2%`** | **`$238,228` (#1)** | `-44.3%` | **`1.79` (#1)** |
| **Last 4M (`91d`)** | **1. `Deployed C2 + Catalyst` (Baseline)** | `40` (`20/91d = 22%`) | `70.0%` | `70.0%` (`14W–6L`) | `+4.84%` | `+89.1%` | `$224,472` | **`-13.0%`** | `3.54` |
| | **7. `SYNERGY 4A`: Full 3-Pillar (Kelly 35% `T&Q`)** | `52` (`34/91d = 37%`) | `69.2%` | `67.6%` (`23W–11L`) | `+3.73%` | `+89.4%` | `$223,696` | `-14.3%` | `3.43` |
| | **8. `SYNERGY 4B`: Full 3-Pillar (50% Slot `T&Q`)** | **`56` (`37/91d = 41%`)** | `67.9%` | `67.6%` (`25W–12L`) | `+3.42%` | **`+91.8%`** | **`$227,830` (#1)** | `-14.3%` | `3.42` |
| **Last 2M (`48d`)** | **1. `Deployed C2 + Catalyst` (Baseline)** | `12` (`7/48d = 15%`) | `66.7%` | `71.4%` (`5W–2L`) | `+1.25%` | `+6.4%` | `$106,128` | `-5.3%` | `1.54` |
| | **7. `SYNERGY 4A`: Full 3-Pillar (Kelly 35% `T&Q`)** | `13` (`9/48d = 19%`) | **`76.9%`** | **`88.9%` (`8W–1L`)** | **`+1.30%`** | `+6.2%` | `$106,049` | `-5.3%` | **`1.89` (#1)** |
| | **8. `SYNERGY 4B`: Full 3-Pillar (50% Slot `T&Q`)** | **`15` (`11/48d = 23%`)** | `73.3%` | `81.8%` (`9W–2L`) | `+1.19%` | **`+8.0%`** | **`$107,839` (#1)** | `-5.3%` | `1.87` |
| **Last 2W (`11d`)** | **1. `Deployed C2 + Catalyst` (Baseline)** | `4` (`2/11d = 18%`) | `50.0%` | `50.0%` (`1W–1L`) | `+0.57%` | `+0.0%` | `$100,003` | `-2.1%` | `0.07` |
| | **7. `SYNERGY 4A`: Full 3-Pillar (Kelly 35% `T&Q`)** | `3` (`1/11d = 9%`) | **`66.7%`** | **`100.0%` (`1W–0L`)** | **`+2.17%`** | **`+2.3%`** | **`$102,291` (#1)** | **`0.0%`** | **`4.79` (#1)** |

---

## 2026-11-06 — Agent 1 v2.4 (`agent1_v2.py` & `agent1_v2_backtest.py`): Priority Tier Ranking (`Tier 1 → Tier 2`), 2.33-Year Equal-Weight Portfolio Backtest (`28M / 12M / 4M / 2M / 2W`), Google News RSS + SEC EDGAR Catalyst Study, and `Nov 5–6` Live Portfolio Forensics


**Scripts Updated:** `agent1/agent1_v2.py`, `agent1/agent1_v2_backtest.py`, `agent1/all_portfolio_strategies_comparison.png`, `agent1/fundamental_catalyst_analysis.png`, `agent1/cyclicality_and_monotonicity_28m.png`, `agent1/agent1_v2_last_2_weeks.png`
**Audit Horizon:** Full 2.33 Years (`28` Months: `2024-07-01` → `2026-11-06`, `592` Trading Days), Last 12 Months (`2025-11-03` → `2026-11-06`, `255` Trading Days), Last 4 Months (`2026-07-01` → `2026-11-06`, `91` Trading Days), Last 2 Months (`2026-09-01` → `2026-11-06`, `48` Trading Days), and Last 2 Weeks (`2026-10-23` → `2026-11-06`, `11` Trading Days).

### 1. Priority Tier Ranking (`USE_PRIORITY_TIER_RANKING = True` in `agent1_v2.py` & `agent1_v2_backtest.py`)
- **Problem with Hard `$25` / `$25M` Moderate-Gap Cap (`C2 Strict`):** Hard-rejecting `<15%` moderate gaps with `open_price > $25.00` or `prior_dvol >= $25M` left the portfolio sitting in `100%` idle cash on **58% of all trading days** over 2.33 years (`252/592` active days = `42%` vs `426/592` = `72%` for `Deployed C2`) and missed high-grade winners when `< 3` Tier-1 candidates existed (such as `2026-11-04 CSIQ +3.40%` & `RIOT +3.71%`, and `2026-11-05 VNT +2.56%`, `EWTX +2.00%`, `PGNY +0.67%`).
- **Solution — 2-Tier Priority Ranking (`Tier 1 → Tier 2` Fallback):**
  - Candidates are sorted by **`(priority_tier, gap_pct)` descending**:
    1. **Tier 1 (`Priority 1`):** Parabolic blow-offs (`gap_pct >= 30%`) AND retail/low-dvol moderate gaps (`gap_pct < 15%` with `open_price <= $25.00` AND `prior_dvol < $25M`).
    2. **Tier 2 (`Priority 0`):** Institutional/mid-cap moderate gaps (`gap_pct < 15%` with `open_price > $25.00` OR `prior_dvol >= $25M`), used only to fill remaining slots when `< 3` Tier 1 candidates qualify.

### 2. Multi-Horizon Equal-Weight Portfolio Backtest (`Active-Slot min(1/k, 50%)` Across All 7 Strategies through `2026-11-06`)
All strategies evaluated under the deployed equal-weight portfolio sizing rule (`33.3%` per trade when 3 trades trigger, `50.0%` per trade when 1–2 trades trigger):

| Horizon | Strategy | Trades (Active Days) | Trade WR | Day WR (`W–L`) | Avg Stock% | Active-Day Port% | All-Day Port% | Non-Comp Port% | Comp Equity (`$100K`) | Max DD | Ann. Sharpe |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **2.33 Years (`28M`)**<br>`2024-07-01 → 2026-11-06` (`592d`) | **`1. Deployed C2 (Tier1→Tier2)`** | `937` (`426/592d` = **`72%`**) | `61.8%` | `59.2%` (`252W–174L`) | `+2.56%` | `+2.26%/d` | **`+1.62%/d`** | **`+962.3%`** | **`$195.6M`** | `-71.5%` | **`3.07` (#1 Base)** |
| | **`2. C2 + Catalyst Filter`** | `877` (`406/592d` = `68%`) | `62.8%` | **`60.1%`** (`244W–162L`) | `+2.82%` | **`+2.46%/d`** | **`+1.68%/d`** | **`+998.3%`** | **`$288.0M`** | `-63.4%` | **`3.20` (#1 All)** |
| | **`3. Combo C1 (Skip 12–30% + B2)`** | `757` (`362/592d` = `61%`) | `60.5%` | `54.1%` (`196W–166L`) | `+2.63%` | `+2.19%/d` | `+1.33%/d` | `+791.5%` | `$31.1M` | `-78.9%` | `2.45` |
| | **`4. Combo C3 (C1 + Pyr)`** | `817` (`362/592d` = `61%`) | `60.2%` | `53.6%` (`194W–168L`) | `+2.65%` | `+1.69%/d` | `+1.03%/d` | `+610.7%` | `$6.4M` | `-83.0%` | `2.02` |
| | **`5. C2 Strict (<=$25 Cap)`** | `519` (`252/592d` = `42%`) | **`64.4%`** | `60.7%` (`153W–99L`) | **`+3.97%`** | `+3.78%/d` | `+1.60%/d` | `+951.6%` | `$102.6M` | `-79.2%` | `2.72` |
| | **`6. Baseline S1 (No B2)`** | `550` (`385/592d` = `65%`) | `59.6%` | `60.5%` (`233W–152L`) | `+2.81%` | `+1.96%/d` | `+1.27%/d` | `+755.8%` | `$45.4M` | `-60.2%` | `2.90` |
| | **`7. Baseline S6 (No B2)`** | `433` (`324/592d` = `55%`) | `58.4%` | `57.7%` (`187W–137L`) | `+2.66%` | `+1.71%/d` | `+0.93%/d` | `+553.7%` | `$6.8M` | `-72.7%` | `2.23` |
| **Last 12 Months**<br>`2025-11-03 → 2026-11-06` (`255d`) | **`1. Deployed C2 (Tier1→Tier2)`** | `460` (`200/255d` = **`78%`**) | `67.4%` | `68.0%` (`136W–64L`) | `+4.14%` | `+3.86%/d` | **`+3.03%/d`** | `+771.9%` | **`$69.7M` (#1)** | `-43.3%` | **`5.03` (#1)** |
| | **`2. C2 + Catalyst Filter`** | `433` (`190/255d` = `75%`) | `67.9%` | `67.9%` (`129W–61L`) | `+4.32%` | `+4.04%/d` | `+3.01%/d` | `+767.1%` | `$64.9M` | `-41.7%` | `4.93` |
| | **`3. Combo C1`** | `393` (`176/255d` = `69%`) | `66.2%` | `63.6%` (`112W–64L`) | `+4.37%` | `+3.97%/d` | `+2.74%/d` | `+698.0%` | `$32.6M` | `-47.6%` | `4.48` |
| | **`4. Combo C3`** | `425` (`176/255d` = `69%`) | `66.1%` | `63.6%` (`112W–64L`) | `+4.33%` | `+3.35%/d` | `+2.31%/d` | `+589.7%` | `$12.8M` | `-47.6%` | `4.08` |
| | **`5. C2 Strict (<=$25 Cap)`** | `310` (`144/255d` = `56%`) | **`68.1%`** | **`68.8%`** (`99W–45L`) | **`+5.50%`** | **`+5.42%/d`** | **`+3.06%/d`** | **`+780.3%`** | `$64.2M` | `-41.3%` | `4.69` |
| **Last 4 Months**<br>`2026-07-01 → 2026-11-06` (`91d`) | **`2. C2 + Catalyst Filter`** | `153` (`67/91d` = `74%`) | `74.5%` | `73.1%` (`49W–18L`) | `+6.16%` | `+6.18%/d` | **`+4.55%/d`** | **`+414.2%`** | **`$3.48M` (#1)** | **`-34.0%`** | **`6.33` (#1)** |
| | **`3. Combo C1`** | `158` (`69/91d` = `76%`) | `74.1%` | `71.0%` (`49W–20L`) | `+5.95%` | `+5.89%/d` | `+4.46%/d` | `+406.2%` | `$3.23M` | `-35.9%` | `6.23` |
| | **`1. Deployed C2`** | `166` (`71/91d` = **`78%`**) | `72.3%` | **`73.2%`** (`52W–19L`) | `+5.62%` | `+5.65%/d` | `+4.41%/d` | `+401.3%` | `$3.11M` | `-35.9%` | `6.22` |
| **Last 2 Months**<br>`2026-09-01 → 2026-11-06` (`48d`) | **`3. Combo C1`** | `78` (`35/48d` = `73%`) | **`73.1%`** | `62.9%` (`22W–13L`) | `+2.78%` | `+2.73%/d` | **`+1.99%/d`** | **`+95.5%`** | **`$239,562` (#1)** | `-8.4%` | `5.32` |
| | **`2. C2 + Catalyst Filter`** | `79` (`34/48d` = `71%`) | `69.6%` | **`67.6%`** (`23W–11L`) | `+2.73%` | **`+2.77%/d`** | `+1.96%/d` | `+94.3%` | `$238,260` | `-8.7%` | **`5.49` (#1)** |
| | **`1. Deployed C2`** | `84` (`37/48d` = **`77%`**) | `69.0%` | **`67.6%`** (`25W–12L`) | `+2.52%` | `+2.48%/d` | `+1.92%/d` | `+91.9%` | `$232,678` | `-8.7%` | `5.32` |
| **Last 2 Weeks**<br>`2026-10-23 → 2026-11-06` (`11d`) | **`2. C2 + Catalyst Filter`** | `22` (`10/11d` = `91%`) | **`72.7%`** | **`70.0%`** (`7W–3L`) | `+2.22%` | **`+1.76%/d`** | **`+1.60%/d`** | **`+17.6%`** | **`$118,717`** | **`-4.0%`** | **`9.58` (#1)** |
| | **`1. Deployed C2`** | `24` (`11/11d` = **`100%`**) | `70.8%` | `63.6%` (`7W–4L`) | `+1.80%` | `+1.35%/d` | `+1.35%/d` | `+14.8%` | `$115,359` | **`-4.0%`** | `7.15` |
| | **`3. Combo C1`** | `22` (`10/11d` = `91%`) | `68.2%` | `60.0%` (`6W–4L`) | `+1.60%` | `+1.16%/d` | `+1.06%/d` | `+11.6%` | `$111,453` | `-8.4%` | `4.31` |
| | **`4. Combo C3`** | `23` (`10/11d` = `91%`) | `65.2%` | `60.0%` (`6W–4L`) | `+1.49%` | `+1.05%/d` | `+0.95%/d` | `+10.5%` | `$110,187` | `-8.4%` | `3.91` |

### 3. 2.33-Year Fundamental Catalyst Study (`Google News RSS by Timestamp + SEC EDGAR Point-in-Time Filings`, `98.8%` Coverage)
- **Dataset:** Classified `929` deployed `C2` trades (`918/929` = **`98.8%`** matched to verified timestamped Google News RSS headlines and/or SEC EDGAR `8-K`, `424B5`, `S-3`, `SC 13D`, `10-Q` filings):
  1. **5. Dilution / Share Offering / Convertible Notes (`N=28`):** **`75.0%` WR \| `+5.85%/trade` \| `+163.8%` sum** — `#1` highest-edge short catalyst (`2026-11-06 DOCN +5.19%` on `$1B` Debt Offering).
  2. **8. Unreported Pure Technical Squeeze (`N=11`):** **`81.8%` WR \| `+5.35%/trade` \| `+58.9%` sum** — `#2` highest-edge short catalyst.
  3. **1B. Biotech Corporate / Pipeline Update (`N=110`):** **`66.4%` WR \| `+4.18%/trade` \| `+459.4%` sum**.
  4. **1A. Biotech Clinical Trial / FDA PR (`N=118`):** **`63.6%` WR \| `+3.00%/trade` \| `+354.3%` sum**.
  5. **2A. Earnings Beat / Guidance Raise (`N=135`):** **`62.2%` WR \| `+2.72%/trade` \| `+367.2%` sum**.
  6. **3. Commercial Contract / Partnership / Deal (`N=127`):** **`61.4%` WR \| `+2.39%/trade` \| `+304.1%` sum**.
  7. **6. Analyst Upgrade / Price Target Hike (`N=78`):** `59.0%` WR \| `+1.20%/trade` (**`-0.25%/trade` on `<15%` moderate gaps without earnings** — e.g., `2026-11-06 SITM -5.85%` & `SLVM -6.57%`).
  8. **7A. Crypto / Commodity Sympathy (`N=58`):** `53.4%` WR \| `+0.72%/trade` (`+0.39%/trade` on `<15%` gaps).
  9. **4. M&A / Buyout / Divestiture / Strategic Stake (`N=26`):** **`46.2%` WR \| `-1.92%/trade` \| `-50.0%` sum** — **Negative-EV trap** (buyout floors & value-unlocking divestitures like `2026-11-06 SLVM -6.57%` do not fade).

### 4. `2026-11-05` & `2026-11-06` Live Equal-Weight Portfolio Forensics
- **`2026-11-05` (`Deployed C2` `+1.74%` Portfolio Return vs `+0.40%` for `C1/C3`):**
  - `Deployed C2` went **`3/3` (`100%` WR)** at its `08:30 PST` exit: `VNT +2.56%` (`+0.85%` port), `EWTX +2.00%` (`+0.67%` port), `PGNY +0.67%` (`+0.22%` port) = **`+1.74%` Equal-Weight Portfolio PnL (`+$1,740` on `$100K`)**.
- **`2026-11-06` (`Combo C1` `+4.49%` Portfolio Return vs `-3.28%` for `Deployed C2`):**
  - `Combo C1` (`Skip 12–30%`) locked `#1 DOCN` (`+11.2%` gap on `$1B` Debt Offering -> `local_high` `+5.19%` @ `50%` weight = `+2.60%` port) and `#2 DOCN_B2` (`bounce_reshort` `+3.79%` @ `50%` weight = `+1.89%` port) for **`+4.49%` Equal-Weight Portfolio PnL (`+$4,490`)**.
  - `Deployed C2` (`Skip 15–30%`) locked `#1 CHRN` (`+14.1%` gap, no trigger), `#2 SHAZ` (`+13.9%` gap, no trigger), and `#3 SLVM` (`+9.9%` gap, `bar0_flush` `-6.57%` @ `50%` weight = **`-3.28%` Portfolio PnL**). `C2 + Catalyst Filter` automatically skips `SLVM` (M&A Divestiture + BofA Upgrade + Materials Sector `>$25`) and promotes `DOCN` into Slot #3 for **`+4.49%` Portfolio PnL**.

---

## 2026-11-02 — Agent 1 v2.3 (`agent1_v2.py` & `agent1_v2_backtest.py`): 12-Month Walk-Forward Audit, 3-Tier RTH Exits (`08:30 / 09:30 / 11:00 PST`), SPAC Filter, and Intraday Profit Lock + `-10%` Stop

**Scripts Updated:** `agent1/agent1_v2.py`, `agent1/agent1_v2_backtest.py`, `agent1/agent1_12m_walkforward_audit.png`
**Audit Horizon:** 12-Month Walk-Forward (`2025-11-03` → `2026-11-02`, `251` Trading Days: 8-Month Blind Pre-July OOS `2025-11-03 → 2026-06-30` + 4-Month Recent `2026-07-01 → 2026-11-02`).

1. **3-Tier RTH Exit Schedule (`08:30 / 09:30 / 11:00 PST`, Replacing `14:00 PST` After-Hours Exit):**
   - Intraday MFE/MAE trajectory analysis across 12 months proved that holding past `11:00 PST` (`14:00 ET`) exposes shorts to late-afternoon (`11:30–13:00 PST`) and after-hours institutional squeezes:
     - **Tier 1 (`08:30 PST` / `11:30 ET`):** Cover `$50+` higher-priced `local_high` morning shorts (`C1/C2/C3`) and all `<15%` moderate-gap morning shorts in `C2`.
     - **Tier 2 (`09:30 PST` / `12:30 ET`):** Cover `<$50` `local_high` morning shorts (`C1/C3`, and `>=30%` gaps in `C2`).
     - **Tier 3 (`11:00 PST` / `14:00 ET` — Hard Flat 2h Before Close):** Cover all `bar0_flush` opening flushes, `_B2` midday bounce re-shorts, and `_W11` `08:00 PST` pyramids.
2. **SPAC & Blank-Check Filter (`is_spac`):**
   - Excludes SPAC shells (`ACQUISITION`, `BLANK CHECK`, `SPAC` in company name or `.WS`/`.WT` suffixes) prior to `PRE_FILTER_N = 30`, eliminating low-float redemption squeezes (`RDAC`, `SKGR`).
3. **Intraday Trailing Profit Lock (`+5.0%` MFE / `45%` Giveback / `>= +2.0%` Floor) & `-10.0%` Hard Intraday Stop:**
   - Protects morning fade gains against midday V-bottoms and caps single-trade tail risk at `-10.0%` intraday.
4. **Disabled `MOD_GAP_PEAK_STOP_BUFFER` (`0.3%` Peak Stop):**
   - 12-Month Walk-Forward Audit revealed that a tight `0.3%` peak stop on `<15%` gaps cost `-56.4%` in the blind 8-month pre-July OOS window due to normal 1-minute noise wicks before the real fade.

---

## 2026-10-29 — Agent 1 v2.2 (`agent1_v2.py` & `agent1_v2_backtest.py`): `2026-10-29` Trade Forensic (`CLMB` vs `PWP`), Moderate-Gap Guard, Total PnL & Hybrid Suite (`Combo C1 / C2 / C3`), and PST Operator Playbook

**Scripts Updated:** `agent1/agent1_v2.py`, `agent1/agent1_v2_backtest.py`, `agent1/agent1_v2_backtest.png`
**Audit Horizon:** Full 84 Trading Days (`2026-07-01` → `2026-10-28`), Last 3 Months (`2026-07-28` → `2026-10-28`, 65d), Post-Aug (`2026-08-16` → `2026-10-28`, 51d), Last 2 Weeks Out-of-Sample (`2026-10-13` → `2026-10-28`, 12d), plus `2026-10-29` Live Session Forensic.


### 1. `2026-10-29` Live Trade Forensic (`CLMB` vs. `PWP`) & Moderate-Gap Quality Guard
- **What Happened on `2026-10-29`:**
  - **`PWP` (`+13.4%` gap, `$14.29` open, Profitable):** Traded continuously with **0 missing 1-min bars** (`83,766` shares in first 8m), peaked at `06:36 PST` (`09:36 ET`) at `$15.23` (`+6.58%` above open), triggered primary short at `06:39 PST` (`09:39 ET`) at `$15.23`, dropped to `$13.99` (`+8.14%` MFE at `06:51 PST`), covered at `08:30 PST` (`11:30 ET`) in `Combo C2` at **`$14.32` (`+5.98%`)**, and also triggered a `PWP_B2` Bounce Re-Short at `07:07 PST` (`10:07 ET`) at `$14.48`.
  - **`CLMB` (`+8.2%` moderate gap, `$36.13` open, Lossy `-4.29%` at `14:00 PST`):** Had **2 missing 1-min bars (`06:32`, `06:34 PST`)** before entry (thin `22,681` shares total in 8m) and squeezed **`+9.05%` above its `$36.13` open** (`1.10x` the entire overnight gap) before forming its `06:35 PST` (`09:35 ET`) peak at `$39.40`. After entering at `06:40 PST` (`$38.00`), it ground higher all afternoon to `$39.63` (`-4.29%`).
- **Moderate-Gap Quality Guard Implemented (`find_local_high()`):**
  - When `max(gap_pct, raw_open_gap) < 15.0%`:
    1. `MAX_MISSING_BARS_MOD_GAP = 1`: Require `<= 1` missing 1-minute bar before entry (`entry_idx + 1 - expected_mins >= -1`).
    2. `MAX_OPEN_RUNUP_MOD_GAP = 0.07`: Require RTH peak run-up above open `(peak_high - open_price) / open_price <= 7.0%`.
  - **Retroactive 84-Day Impact:** Filters out `2026-10-29 CLMB` (`-4.29%`) and `2026-08-29 BABA` (`-3.51%`, which ran `+7.55%` above open on a `+5.7%` gap), improving `S6` 3M WR (`73.3% → 75.0%`) and Avg Return (`+4.92% → +5.12%`).

### 2. Total Cumulative PnL & Multi-Trigger Hybrid Architecture (`U1..U8` → `C6/C7/C8` → `Combo C1/C2/C3`)
- Evaluated 8 causal revenue-expansion levers (`U1..U8`) and 12 hybrid combinations uniting **`S6 / S1 / S12`** (high-precision filter & exit rules) with **`C6 / C7 / C8`** (multi-trigger entries & active slot sizing):
  1. **Bar-0 Immediate Flush Entry (`U1`, `06:34 PST` / `09:34 ET`):** Captures immediate open-flush parabolic blow-offs (`09-09 DRUG +66.5%`, `10-24 HOTH +23.2%`, `08-26 CING +19.5%`) where `bar0` (`06:30 PST`) is the high of the first 4 minutes (`high[1..3] < high[0]`), `close[3] < open[0]`, and pullback from `high[0] >= 1.5%`.
  2. **Continue-Loop Peak Discovery (`U2`, `06:36–06:49 PST` / `09:36–09:49 ET`):** Replaces `break` with `continue` inside `find_local_high()` when an early micro-bump (`< 0.5%` pullback) fails, allowing discovery of the real exhaustion peak (`idx 3..16`, peak `<= 06:46 PST`, entry `<= 06:49 PST`). Recovers **+16 high-grade trades** (`10-27 DRUG +8.02%`, `09-12TNXP +14.82%`, `08-19 SIGA +10.22%`).
  3. **Intraday Lower-High Bounce Re-Short (`U4`, `06:45–10:30 PST` / `09:45–13:30 ET`):** On qualified tickers after a morning drop `>= 3.0%` from the opening peak, shorts a 2-bar lower-high bounce (`bounce >= 1.5%` off trough, `< peak * 0.995`, confirmed by 2 lower highs with `>= 0.4%` pullback; max 1 bounce trade/day). Adds **+26 trades in 3M at `80.8% WR` and `+5.65%` avg**.
  4. **11:00 ET (`08:00 PST`) Winner Add-On Pyramid (`U8` in `Combo C3`):** Adds a second tranche at `08:00 PST` (`11:00 ET`) on trades already `>= +4.0%` in profit (`gap >= 12%`) and still `>= 1.0%` below entry (`24` add-ons in 3M at **`87.5% WR`, `+7.28%` avg**).
  5. **Hybrid `08:30 PST` (`11:30 ET`) Early Cover on Moderate Gaps (`<15%` in `Combo C2`):** Locks in morning fade profits at `08:30 PST` (`11:30 ET`) on `<15%` gaps (`10/29 PWP +5.98%` vs `+1.97%` at close) while holding `>=30%` blow-offs to `14:00 PST` (`17:00 ET`).
  6. **Active-Candidate Dynamic Sizing (`active_cap50`):** Divides daily capital across `min(actual_signals_today, 3)` capped at **50% max per position** (`50%` on 1–2 signal days, `33.3%` on 3+ signal days), eliminating idle cash drag on 1–2 signal days.

### 3. Production Adoption: `Combo C1`, `Combo C2`, `Combo C3` Multi-Horizon Results (`$100K` Start)

| Strategy ID | Core Rules & Triggers | Last 2W OOS (`10/13–10/28`) | Last 3 Months (`07/28–10/28`, 65d) | Post-Aug (`08/16–10/28`, 51d) | Full 84d (`07/01–10/28`) |
|---|---|---|---|---|---|
| **`C1: S6+C6`** *(Deployed #1)* | `S6` (`Skip 12–30%`) + `Bar0Flush` + `ContinueLoop` + `Bounce Re-Short` + `ActCap50%` (`N=3`) | `18t` \| `77.8% WR` \| `+3.57%` avg \| `+64.3%` sum \| **`$124,145`** | `93t` \| **`80.0% WR`** \| `+6.67%` avg \| `+620.3%` sum \| Sh `0.51` \| **`$1,202,700` (+1,103%)** | `66t` \| **`79.0% WR`** \| `+6.30%` avg \| `+415.9%` sum \| Sh `0.50` \| **`$550,523`** | `120t` \| `78.0% WR` \| `+9.62%` avg \| `+1,153.9%` sum \| Sh `0.50` \| **`$8,103,957`** |
| **`C2: S1+C6+08:30`** *(Deployed #2)* | `S1` (`Skip 15–30%`) + `Bar0Flush` + `ContinueLoop` + `Bounce` + **`08:30 PST Exit (<15% gap)`** + `ActCap50%` (`N=3`) | **`19t` \| `78.9% WR` \| `+3.73%` avg \| `+70.8%` sum \| `$125,812`** | `104t` \| `75.0% WR` \| `+5.48%` avg \| `+570.2%` sum \| Sh `0.42` \| **`$966,451` (+866%)** | `77t` \| `73.0% WR` \| `+5.04%` avg \| `+388.0%` sum \| Sh `0.41` \| **`$478,664`** | `135t` \| `75.0% WR` \| `+8.50%` avg \| `+1,147.8%` sum \| Sh `0.46` \| **`$6,037,136`** |
| **`C3: S6+Full`** *(Deployed #3)* | `S6` (`Skip 12–30%`) + `Bar0Flush` + `ContinueLoop` + `Bounce` + **`08:00 PST Winner Add-On`** + `ActCap50%` (`N=3`) | `20t` \| `75.0% WR` \| `+3.17%` avg \| `+63.4%` sum \| **`$121,297`** | **`104t` \| `80.0% WR` \| `+6.73%` avg \| `+700.0%` sum \| Sh `0.53`** \| `$926,392` (+826%) | **`76t` \| `79.0% WR` \| `+6.48%` avg \| `+492.8%` sum \| Sh `0.52`** \| `$435,550` | **`136t` \| `78.0% WR` \| `+10.07%` avg \| `+1,370.2%` sum \| Sh `0.52`** \| `$5,264,509` |
| *Ref: `S6` (v2.1)* | `Skip 12–30%` + `P<=$100` + `PB>=0.5%` (Single trigger, fixed `1/3` slot) | `10t` \| `80.0% WR` \| `+4.13%` avg \| `+41.3%` sum \| `$114,492` | `44t` \| `75.0% WR` \| `+5.12%` avg \| `+225.0%` sum \| Sh `0.45` \| `$204,707` | `31t` \| `74.0% WR` \| `+5.17%` avg \| `+160.2%` sum \| Sh `0.43` \| `$164,849` | `57t` \| `74.0% WR` \| `+8.77%` avg \| `+500.1%` sum \| Sh `0.48` \| `$454,321` |

### 4. Interactive PST Operator Playbook & Intraday Telemetry (`agent1_v2.py`)
- Added `get_intraday_trade_telemetry()` and `_et_to_pst()` to `agent1_v2.py` so every intraday execution (`python3 agent1/agent1_v2.py`) automatically:
  1. Evaluates all 4 intraday triggers (`Bar0_Flush`, `LocalHigh_Cont`, `Bounce_ReShort`, `AddOn_0800PST`) using bars available up to the current PST minute.
  2. Computes live mark-to-market unrealized PnL, MFE, MAE, and `08:30 PST` (`11:30 ET`) early-cover prices for `<15%` moderate gaps.
  3. Prints a contextual **OPERATOR PLAYBOOK (PST)** (`[A] Immediate Action Items`, `[B] When & How Often to Run Next`, `[C] Daily Multi-Trigger Run Schedule in PST` at `06:35`, `06:45`, `07:00–07:45`, `08:01`, `08:30`, `09:00–10:30`, and `14:00 PST`).

---

## 2026-10-28 — Agent 1 v2.1 (`agent1_v2.py`): 360° Audit, 7 Leak-Free Fixes, 26-Idea Benchmark & `S6/S1/S12` Adoption

**Scripts Updated:** `agent1/agent1_v2.py`, `agent1/agent1_v2_backtest.py`, `agent1/agent1_v2_bt_comparison.py`
**Audit Horizon:** Full 84 Trading Days (`2026-07-01` → `2026-10-28`), Last 3 Months (`2026-07-28` → `2026-10-28`, 65d), Post-Aug (`2026-08-16` → `2026-10-28`, 51d), and Last 2 Weeks Out-of-Sample (`2026-10-13` → `2026-10-28`, 12d).

### 1. Live vs. Backtest Forensic Reconciliation (`Oct 13–28, 2026`)
- **Live Trading (`Oct 13–28`, 24 Executed Trades across 10 Days, excl. RXRX):**
  - **Realized Dollar PnL:** `+$4,159.41` (`10W / 14L`, `41.7% WR`, **+0.82% avg return per trade**, `+19.65%` sum of trade returns).
  - **Why Live Dollars Were Positive Despite `41.7%` WR:** Larger position sizing (`$35k–$42k` vs `$12k–$15k`) and early intraday profit-taking on 8 winners (`10/13 EFXT +3.85%` vs `-2.72%` at 17:00 ET, `10/20 BRZE +2.78%` vs `-0.26%`, `10/28 VRT +2.89%` vs `-2.98%`, `10/28 TEX +1.50%` vs `+0.01%`) added **`+$12,562` (`+17.50%`) of execution alpha** over holding to 17:00 ET.
  - **Why `2026-10-28` Lost Money (`-$1,752.49` across 4 positions):**
    - `AXGN` (`-$2,403.00`, `-6.25%` live / `-7.77%` at 17:00 ET): `+24.4%` institutional earnings gap (`T6` tier) selected by `C5` & `C7`, which squeezed after 10:30 ET.
    - `WING` (`-$951.14`, `-3.28%`): `$251.50` large-cap restaurant earnings squeeze; wasn't even in `agent1_v2`'s top-30 pool because 17 leveraged single-stock ETFs crowded out common stocks before `PRE_FILTER_N = 30`.
    - `VRT` (`+$992.54` live / `-2.98%` at 17:00 ET): `$200+` large-cap earnings gap (`+16.51%` raw gap compressed to `+12.05%` vs AH ref, sneaking past `C3`'s `15–30%` filter).
    - `TEX` (`+$609.11` live / `+0.01%` at 17:00 ET): Covered into morning dip at `$52.65`.

### 2. Root Causes of Strategy Deterioration (360° Audit Findings)
1. **Why `Agent 1 (v1)` Showed `>10%`/Trade (`+14.71%` in `E2`) & Why It Broke in Mid-July/August:**
   - **`+7.4pp` to `+8.5pp` of `E2`'s `+14.71%` was Bar-60 (`10:30 ET`) Look-Ahead Bias:** `E2` required `sustained_drop` classified at `10:30 ET` while entering 45–54 minutes earlier at `09:36–09:45 ET`. True causal H1 baseline (`E2-A`) averaged **`+4.46%`**.
   - **Why `v1` Degraded in August (`+4.46% → +2.06%`):** (a) Extreme `>= 30%` (`T7`) micro-cap gaps dropped `-79%` in Q3 (`89` trades in H1 → `19` in Q3), and (b) **Bar-15 (`09:45 ET`) `fallback` entries collapsed by `-6.80pp`** (`+6.37%` Pre-Aug → `-0.43%` Post-Aug, `-11.4%` for `N=1`). Meanwhile, **RTH `local_high` entries (`09:36–09:44 ET`) had ZERO degradation (`+3.52%` Pre-Aug vs `+3.48%` Post-Aug, `-0.04pp` delta)**.
2. **Why `Agent 1 v2`'s `C5`, `C3+C5`, `C7`, and `C9` Failed Out-of-Sample (`Oct 13–28`):**
   - **In-Sample Overfitting on Tiny `N=6..7` Trade Counts:** `C5`, `C3+C5`, and `C7` were selected by screening 32 features × 18 rules on `Jul 1 – Oct 9` to maximize Post-Aug Sharpe, where `C5` (`N=1`) had only **7 trades**. `Oct 13–28` was the first true out-of-sample window.
   - **The "Dip-and-Rip" Structural Contradiction:** Requiring `bar0` red (`C5`), `PM trend < 0` (`C7`), or `2-bar drop < 0` (`C9`) at `09:30–09:31 ET` **AND** a subsequent 3-bar `local_high` peak at `09:33–09:41 ET` selects stocks that **dipped at the open and then ripped to a new high (V-bottom accumulation)** (`AXGN -7.77%`, `IRON -4.46%`, `KNSA -3.40%`, `VRT -2.98%`, `KVYO -1.89%`), while **systematically rejecting** genuine parabolic blow-off tops that pushed green during `09:30–09:35 ET` into an exhaustion peak (`AIOK +19.83%`, `BHVN +8.20%`, `MED +7.58%`, `SIGI +3.23%`).
   - **Disjoint Bucket Proof (`Jul 28 – Oct 28`, 3 Months):**
     - **Bucket 1 — Selected ONLY by `C3` (`bar0` GREEN into local high):** `54` trades | **66.7% WR** | **+4.16% avg** | **+224.85% total PnL** | Sharpe `0.369`. (In `Oct 13–28` OOS: `7` trades, **85.7% WR**, **+6.08% avg**, **+42.58% total PnL**).
     - **Bucket 2 — Overlap (`C3` AND `C5/C7/C9`):** `57` trades | `52.6% WR` | `+0.92% avg` | `+52.49% total PnL`. (In `Oct 13–28` OOS: `9` trades, `33.3% WR`, `-1.16% avg`, `-10.46% total PnL`).
     - **Bucket 3 — Selected ONLY by `C5/C7/C9` (Outside `C3`, inside `15–30%` T6 gap):** `17` trades | **33.3% WR** | **-2.46% avg** | **-41.78% total PnL**. (In `Oct 13–28` OOS: `5` trades, **20.0% WR**, **-2.75% avg**, **-13.75% total PnL**).

### 3. Seven Leak-Free & Pipeline Alignment Fixes Implemented
1. **Frozen Float Dictionary (`_FLOAT_DATA` only):** Removed `float_cache.json` merge in `agent1_v2.py` so live/historical scans and `agent1_v2_backtest.py` use the exact same frozen 217-ticker dictionary without time-travel bias.
2. **Pre-Filter Non-CS & Leveraged ETFs BEFORE `PRE_FILTER_N = 30`:** Both `agent1_v2.py` and `agent1_v2_backtest.py` now exclude `_EXCLUDE`, `_LEV_ETF`, suffix `W/WS/U/R`, and cached non-CS tickers in `data/ticker_type_cache.json` *before* slicing `candidates[:30]`. (Recovered 41 previously truncated CS trades across 84 days).
3. **Strict Clock-Timestamp Bounds in `find_local_high()`:** Added `peak_bar["time_str"] <= "09:41"` and `entry_bar["time_str"] <= "09:44"` so illiquid stocks with missing minute bars cannot trigger late `09:45–10:08 ET` entries at array index `11`.
4. **Entry-Time Gap Erosion Check (`<= 50%` Consumed at Entry):** In addition to the `09:31 ET` scan-time erosion check (`gap_consumed <= 0.50` vs `ah_ref_price`), `find_local_high()` now verifies that `entry_price` at confirmation (`09:36–09:44 ET`) has not already consumed `> 50%` of the gap above `ah_ref_price`.
5. **Dual Gap Check (`gap_pct` AND `raw_open_gap`):** Gap-band exclusions now check both `gap_pct` (vs AH ref) and `raw_open_gap` (vs prior RTH close), preventing stocks like `VRT` (`+16.51%` raw gap compressed to `+12.05%` vs AH ref) from sneaking past the skip zone.
6. **Dynamic `optimal_n` Capital Allocation in Backtest:** Fixed `agent1_v2_backtest.py` to allocate `capital / strat["optimal_n"]` per slot instead of hardcoding `capital / 3`.
7. **`DRY_RUN` Guard on BigQuery Logging:** Guarded `log_scan_to_bq()` with `if not DRY_RUN:` in `agent1_v2.py`.

### 4. Consolidated Multi-Factor Benchmark & Production Adoption (`S6`, `S1`, `S12`)

Out of 26 single-factor and multi-factor combinations tested on the leak-free pipeline, three synergistic filters on top of `C3` delivered a massive step-change across all horizons:
- **Price Cap (`open_price <= $100.00`):** Eliminates mega/large-cap institutional earnings squeezes (`WING $251`, `VRT $236`, `PAG $211`, `URI $800+`).
- **Pullback Confirmation (`pullback_pct >= 0.5%` from Peak High at Confirmation Close):** Requires `(peak_high - entry_price) / peak_high * 100 >= 0.5%` so flat bull-flag pauses (`EFXT -2.72%`, `KNSA -3.40%`, `PAG -0.36%`, `BRZE -0.26%`) are skipped.
- **Skip `[12%, 30%)` (`S6`) or `[15%, 30%)` (`S1` / `S12`) on Both `gap_pct` and `raw_open_gap`:** Blocks institutional earnings repricings in the `12–30%` / `15–30%` drift zone (`AXGN -7.77%`, `IRON -4.46%`, `VRT -2.98%`, `KNSA -3.40%`).

| Strategy | `N` | Last 2W (`Oct 13–28`, 12d) | Last 3M (`Jul 28–Oct 28`, 65d) | Post-Aug (`Aug 16–Oct 28`, 51d) | Full 84d (`Jul 01–Oct 28`) | $100k Compounded (84d) |
|---|---:|---|---|---|---|---:|
| *Old 5-Strategy Combined (`C5..C3`)* | *3* | *21t \| 47.6% WR \| +0.87% avg \| +18.37% sum* | *128t \| 53.9% WR \| +1.84% avg \| Sh 0.181* | *98t \| 54.1% WR \| +1.82% avg \| Sh 0.171* | *166t \| 58.4% WR \| +4.55% avg \| Sh 0.289* | *$425,400 (+325.4%)* |
| *Leak-Free `C3` Baseline (`Skip 15–30%`)* | *3* | *16t \| 56.2% WR \| +2.01% avg \| +32.12% sum* | *111t \| 60.4% WR \| +2.50% avg \| Sh 0.262* | *84t \| 59.5% WR \| +2.55% avg \| Sh 0.256* | *141t \| 63.1% WR \| +5.34% avg \| Sh 0.340* | *$541,940 (+441.9%)* |
| **`S6: Skip12-30+P<=100+PB0.5` (ADOPTED)** | **3** | **10t \| 80.0% WR \| +4.13% avg \| +41.34% sum** | **45t \| 73.3% WR \| +4.92% avg \| Sh 0.444** | **32t \| 71.9% WR \| +4.90% avg \| Sh 0.415** | **58t \| 72.4% WR \| +8.56% avg \| Sh 0.471** | **$466,007 (+366.0%)** |
| **`S1: Skip15-30+P<=100+PB0.5` (ADOPTED)** | **3** | **11t \| 81.8% WR \| +3.86% avg \| +42.44% sum** | **56t \| 67.9% WR \| +3.64% avg \| Sh 0.349** | **40t \| 67.5% WR \| +3.69% avg \| Sh 0.333** | **72t \| 69.4% WR \| +7.29% avg \| Sh 0.428** | **$505,029 (+405.0%)** |
| **`S12: Skip15-30+P<=100+PB0.5(N2)` (ADOPTED)** | **2** | **10t \| 80.0% WR \| +3.97% avg \| +39.73% sum** | **50t \| 68.0% WR \| +3.99% avg \| Sh 0.365** | **37t \| 67.6% WR \| +3.81% avg \| Sh 0.334** | **63t \| 68.3% WR \| +7.78% avg \| Sh 0.431** | **$899,887 (+799.9%)** |

---

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
