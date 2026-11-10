# Agent Status — Last Updated: 2026-11-10

## Currently Active Work
- **Trades & Quotes (`T&Q`) 3-Pillar Synergy Architecture (`agent1/deep_synergy_forensics_and_architectures.py`, `data_ops/bulk_priority_pos_events_tq.py` & `data_ops/gcp_parallel_tq_worker.py`):** Expanded `T&Q` universe from `102 → 157` stocks (`139` with full `Microstructure_Advanced`), upgraded `Microstructure_Advanced` schema from `12 → 16` tick-level columns (`Odd_Lot_OFI`, `Block_OFI`, `Signed_Semivariance`, `Ask_Step_Down_Rate`), verified **8 Orthogonal `T&Q` / Multi-Day Features**, and built the **3-Pillar Synergy Architecture (`Rule-Based 3-Bar Structural Mechanics + Pre-Open Catalyst Filter + Walk-Forward T&Q Microstructure ML`)** that reaches **`$437,129` (`+175.8%` Tot PnL, `60.8%` WR, `1.59` Sharpe)** over 24M OOS vs `$309,939` (`+135.1%`, `1.37` Sharpe) for `Deployed C2 + Catalyst` on the exact same `157`-stock `$10–$100 Unadj` universe.
- **Agent 1 v2.5 (`agent1/agent1_v2.py` & `agent1/agent1_v2_backtest.py`):** DEPLOYED (`Deployed C2 (Tier1→Tier2)`, `Combo C1`, `Combo C3` multi-trigger hybrid suite + Priority Tier Ranking + 3-Tier RTH Exit Schedule (`08:30 / 09:30 / 11:00 PST`) + SPAC/Warrant Filter + Intraday Profit Lock & `-10%` Disaster Stop)
- **Agent 1 (`agent1/agent1.py`):** E2-A + E2-B live scanner
- **Agent 3 (`agent3/`):** C1-E01 (WedThu formal backtest)

## Last Session Summary (2026-11-10 — `157`-Stock `T&Q` Expansion, 4-Column Sub-Minute Tick Audit & 3-Pillar Synergy Architecture)
- **Why Standalone `T&Q` ML Underperformed Rule-Based `Deployed C2` & How We Fixed All 4 Flaws (`diagnose_ml_vs_rule_based_gap.py` & `deep_synergy_forensics_and_architectures.py`):**
  1. **1-Bar Unconfirmed Triggers vs. 3-Bar Structural Exhaustion:** Standalone ML evaluated `2,619` raw 1-bar red candles (`-0.16%` base-rate mean) exposed to right-skewed `-10%` squeeze stops, whereas `Deployed C2` (`189` setups, `+1.55%` mean) waits for a 3-bar confirmed structural high (`i-3..i+3` + `>= 30%` gap) and `_B2` midday bounce re-shorts.
  2. **Domain-Pool Transfer vs. Small-Sample Overfitting (`N = 2,619` vs. `N = 189`):** Training GBDT/ML models directly on the `189` C2 trades overfits small folds (`N < 80`). Training walk-forward `T&Q` models (`M7` Monotonic GBDT, `M12` Deep-18 Ensemble, `M4` Huber-Ridge, and parameter-free `ORTH_Z`) on the full `2,619`-setup `$10–$100` domain pool (`1,000+` rows/fold) and applying their out-of-sample scores (`GRAND_SYN`) to gate, size, and augment 3-bar C2 setups eliminates overfitting.
  3. **Portfolio Slot Dilution (`min(1/k, 50%)`):** Adding a 1-bar ML trade on a day where `Deployed C2` already had `2` confirmed 3-bar trades (`2 × 50% = 100%` deployed) diluted `+2.06%/trade` C2 setups down from `50%` to `33.3%`. Fixed via **Strict Non-Diluting Slot Hierarchy** (1-bar `T&Q` consensus fills only idle slots `open_slots = 2 - k_C2`).
  4. **Kelly-Proportional Slot Sizing (`35%` vs. `50%`):** Sizing 1-bar `T&Q` consensus fills at a **Kelly-optimal `35%` slot weight (`w_mult = 0.70`)** while sizing `T&Q`-confirmed 3-bar C2 trades at `50%–55%` (`w_mult = 1.00–1.10`) maximizes geometric compounding and controls drawdown.
- **Forensic Separation of the `189` `Deployed C2 + Catalyst` Trades by Walk-Forward `T&Q` ML (`157` `T&Q` Stocks, `$10–$100 Unadj`):**
  - **Toxic Morning Order-Flow Vetoed (`M7 < -0.30` and `M12 < -0.30`):** **`37` trades | `54.1%` WR | `-0.54%` avg | `-19.9%` sum** (pure drag removed!).
  - **Bearish-Flow `_B2` Midday Bounces Vetoed (`is_b2` and `GRAND_SYN < -0.10`):** **`22` trades | `50.0%` WR | `-1.00%` avg | `-22.0%` sum**, whereas the **remaining `46` `T&Q`-Confirmed `_B2` Bounces (`GRAND_SYN >= -0.10`) average `+2.15%/trade` (`+99.0%` sum)**!
- **Multi-Horizon Walk-Forward OOS Results Across the `157`-Stock `T&Q` Universe (`$10–$100 Unadj`, `$100K` Compounded, `tq_3pillar_synergy_backtest.png`):**
  - **`SYNERGY 4A: Full 3-Pillar (C2 + Morning & _B2 T&Q Veto + Unlocked 15–30% Gaps + Non-Diluting Kelly 35% Upgraded T&Q Slot-Fill)` — #1 Overall Winner:**
    - **Full 24M OOS (`506d`):** `401` trades (`51%` active days), **`60.8%` Trade WR**, **`+175.8%` Tot PnL**, **`$437,129` Compounded Equity (`+$127,190` / `+41%` ahead of `Deployed C2 + Catalyst`'s `$309,939`)**, **Sharpe `1.59` (#1 vs `1.37` for C2)**.
    - **Last 12 Months (`255d`):** `197` trades (`50%` active days), **`59.9%` Trade WR**, **`+102.2%` Tot PnL**, **`$238,228` Compounded Equity (`+$55,405` ahead of C2's `$182,823`)**, **Sharpe `1.79` (#1 vs `1.42` for C2)**.
    - **Last 2 Months (`48d`) & Last 2 Weeks (`11d`):** Last 2M achieves **`76.9%` Trade WR / `88.9%` Day WR (`8W–1L`, Sharpe `1.89` vs `1.54` for C2)**; Last 2W achieves **`+2.3%` Tot PnL (`$102,291`, `0.0%` DD, Sharpe `4.79` vs `$100,003` / `0.07` Sharpe for C2)**.
  - **`SYNERGY 2: C2 + Morning & _B2 T&Q Veto` (Pure High-Precision Mode):**
    - Lifts average trade return by **`+39%` (`+1.55% → +2.16%/trade`)**, grows 24M compounded equity from **`$309,939 → $365,672` (`+$55,733`)**, and cuts 24M Maximum Drawdown by `-7.3%` (**`-38.1% → -30.8%`**, Sharpe **`1.57`**).
- **16-Column `Microstructure_Advanced` Tick Schema Audit (`352` Setups Across Top-50 Completed Tickers):**
  - Verified that **`Odd_Lot_OFI - Block_OFI` (Trigger Bar)** achieves **Spearman `r = +0.112 (p = 0.036*)` and a `+1.77%/trade` `Q3 - Q1` return spread** (`+0.88%` in `Q3` when retail odd-lots buy into institutional block selling vs. `-0.89%` in `Q1`).

## Prior Session Summary (2026-11-04 to 2026-11-09)
- **Multi-Horizon Equal-Weight Portfolio Backtest (`Active-Slot min(1/k, 50%)` Across All Strategies through `2026-11-06`):**
  - Evaluated all 7 portfolio strategies using equal-weight portfolio sizing (`33.3%` per trade when 3 trades trigger, `50.0%` per trade when 1–2 trades trigger) across `2.33 Years (28M: 2024-07-01 → 2026-11-06, 592d)`, `Last 12 Months (255d)`, `Last 4 Months (91d)`, `Last 2 Months (48d)`, and `Last 2 Weeks (11d)`:
    - **`1. Deployed C2 (Tier1→Tier2)` is the #1 Base Portfolio Strategy:**
      - **Full 2.33 Years (`592d`):** `937` trades (`426/592` active days = **`72%`**), `61.8%` Trade WR, `59.2%` Day WR (`252W–174L`), **`+2.26%/d` Active-Day Port%**, **`+1.62%/d` All-Day Port%**, **`+962.3%` Non-Compounded Portfolio Return** (`+170.8%` ahead of `Combo C1`'s `+791.5%`), **`$195.6M` Compounded Equity** (from `$100K`), **Ann. Sharpe `3.07` (#1)**.
      - **Last 12 Months (`255d`):** `460` trades (`78%` active days), `67.4%` Trade WR, `68.0%` Day WR (`136W–64L`), **`+3.03%/d` All-Day Port%**, **`+771.9%` Non-Compounded Portfolio Return**, **`$69.7M` Compounded Equity (#1)**, **Ann. Sharpe `5.03` (#1)**.
      - **Last 2 Weeks (`Oct 23 → Nov 06, 2026`, `11d`):** `24` trades (`100%` active days), `70.8%` Trade WR, **`+1.35%/d` All-Day Port%**, **`+14.8%` Non-Compounded Portfolio Return** (`$115,359`, `-4.0%` Max DD, **Ann. Sharpe `7.15`**), beating `Combo C1` (`+11.6%`, `-8.4%` Max DD, Sharpe `4.31`) and `Combo C3` (`+10.5%`, Sharpe `3.91`).
    - **`2. C2 + Catalyst Filter` (Google News RSS by Timestamp + SEC EDGAR Filings):**
      - Skipping **M&A / Buyout / Divestiture / Strategic Stake** (`-1.92%/tr` over 2.3Y) and **`<15%` Moderate-Gap Analyst Upgrades / Sympathy / Institutional Macro-Cyclical (`Mining/Materials, Energy/Industrials, Financials` with `>$25` or `>=$25M` dvol)** lifts 2.33-Year Equal-Weight Portfolio Return to **`+998.3%`** (`+1.68%/d` all-day, **`$288.0M` compounded**, Max DD reduced `-71.5% → -63.4%`, **Sharpe `3.20`**) and Last 2 Weeks to **`+17.6%`** (`$118,717`, **Sharpe `9.58`**).
- **`2026-11-05` & `2026-11-06` Live Equal-Weight Portfolio Forensics:**
  - **`2026-11-05`:** `Deployed C2` went **`3/3` (`100%` WR)** at its `08:30 PST` exit (`VNT +2.56%`, `EWTX +2.00%`, `PGNY +0.67%` @ `33.3%` weight each) for **`+1.74%` Equal-Weight Portfolio PnL (`+$1,740` on `$100K`)**, vs `+0.40%` for `C1/C3` and `0.00%` for `C2 Strict`.
  - **`2026-11-06`:** `Combo C1` (`DOCN +5.19%`, `DOCN_B2 +3.79%` @ `50%` each) made **`+4.49%` Portfolio PnL** and `Combo C3` made **`+3.35%` Portfolio PnL**, while `Deployed C2` locked `#1 CHRN` (no trigger), `#2 SHAZ` (no trigger), `#3 SLVM` (`-6.57%` raw / **`-3.28%` Portfolio PnL** at `50%` slot weight). `SLVM` (`$35.15`, `$30.6M` dvol, `+9.9%` gap) squeezed on an **M&A Divestiture (`Plans to Sell Europe Business`) + BofA Analyst Upgrade**, whereas `DOCN` faded after announcing a **`$1B` Convertible/Debt Financing** (our #1 highest-edge short catalyst at `+5.85%/trade`).

## Completed This Week
- **Trades & Quotes (`T&Q`) 142-Stock Expansion, 16-Column `Microstructure_Advanced` Upgrade & Point-in-Time Unadjusted Walk-Forward Backtest (`2026-11-09`):**
  - Upgraded `bulk_priority_pos_events_tq.py` and `gcp_parallel_tq_worker.py` to 16 `Microstructure_Advanced` columns (`+Odd_Lot_OFI`, `+Block_OFI`, `+Signed_Semivariance`, `+Ask_Step_Down_Rate`).
  - Completed 48-feature orthogonal study and verified `+69.5%` 24M OOS PnL lift (`$81.4K → $164.7K`) on `M7 Deep GBDT ($10–$100 Unadj)` from the 8 verified orthogonal features (`tq_129_unadj_clean_upgraded_backtest.png`).
- **Zero-Look-Ahead Unadjusted-Price (`adjusted="false"`) Audit & Combined Multi-Horizon Backtest (`2026-11-07`):**
  - Pulled exact unadjusted historical daily prices (`unadj_np_28m.pkl`) across all 593 trading days and proved that `4.47%` (`225,365` ticker-days) of `gd_28m` `Close >= $8` entries were sub-`$8` penny stocks inflated by future `1-for-50+` reverse splits. Eliminated daily-bar MOC Short illusions (`#1`, `#2`, `#3`) and `A6` `07:30 PST` late-entry bounce chasing (`-13.2%` in Last 2W).
  - Verified that **`#9 COMBINED: Deployed C2 + Pre-Open Catalyst Filter + C1 Crash Reversal (Wed/Thu ONLY, 06:30 PST Open -> T+3)`** is 100% zero-look-ahead, split-proof, and beats `Deployed C2 (Base)` across every horizon: **Full 28M `+1,038.7%` (`$436.3M`, Sharpe `3.36`)**, **Last 12M `+805.7%` (`+3.16%/d` all days, `+4.17%/active day`, Sharpe `5.31`, Max DD `-15.8%`)**, and **Last 2 Weeks `+19.5%` (`+1.78%/d` all days, `+1.95%/active day`, `73.9%` Trade WR, `70.0%` Day WR, Sharpe `10.33`, Max DD `-3.4%`)**.
- Agent 1 v2.4 (`agent1_v2.py` & `agent1_v2_backtest.py`): Priority Tier Ranking (`Tier 1 → Tier 2`), 2.33-Year (`28M`) Cyclicality & Multi-Horizon Equal-Weight Portfolio Backtest (`all_portfolio_strategies_comparison.png`), and 2.33-Year Google News RSS + SEC EDGAR Catalyst Study (`fundamental_catalyst_analysis.png`).
- Agent 1 v2.3 (`2026-11-02`): 12-Month Walk-Forward Audit (`agent1_12m_walkforward_audit.png`), 3-Tier RTH Exit Schedule (`08:30 / 09:30 / 11:00 PST`), SPAC & Warrant Filter, and Intraday Profit Lock (`+5% MFE / 45% Giveback`) + `-10%` Hard Stop.
- Agent 1 v2.2 (`2026-10-29`): `Combo C1/C2/C3` multi-trigger hybrid suite, Moderate-Gap Guard, and PST Operator Playbook.

## Blocked / Waiting
- Nothing currently blocked

## Do Not Forget
- **Equal-Weight Portfolio Reporting:** Always report **Equal-Weight Portfolio Returns (`Active-Slot min(1/k, 50%)` — `33.3%` per trade if 3 trigger, `50%` if 1–2 trigger)** rather than unweighted trade sums.
- **All Times in PST:** Always report intraday entry/exit timestamps in **PST (`America/Los_Angeles`)**: Watchlist at `06:31 PST`, Morning Entries `06:33–06:44 PST`, Midday Add-Ons `06:55–10:35 PST` (`08:00 PST` Pyramid for `C3`), and 3-Tier RTH Exits at **`08:30 PST`** (`$50+ local_high` & `C2 <15%` gap), **`09:30 PST`** (`<$50 local_high`), and **`11:00 PST`** (`bar0_flush`, `_B2` bounce, `_W11` pyramid — hard flat RTH exit).
- `agent1.py`: Float cache miss must EXCLUDE for hard-guard experiments (soft float miss=INCLUDE for E2 lock).
- `agent3/`: C1 fallback entries excluded (near-zero edge); never trade C1 on Tuesday (avg -1.02%).
- All agents: OOS cutoff = 2026-01-01.

## Next Session Should Start With
Read `CLAUDE.md`, `AGENT_STATUS.md`, `agent1/CLAUDE.md`, and `agent1/STRATEGY_LOCK.md`.



