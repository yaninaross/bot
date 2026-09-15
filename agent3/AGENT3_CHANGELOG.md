# Agent 3 — Changelog

## 2026-09-15 — Phase 2 Investigation (v3 OOS fix)

### v3 — Phase 2 OOS Validation Bug Fix
Three bugs in `agent3_phase2.py` OOS validation fixed:
1. **S2 bounce_score filter**: `_resolve_p2_filter()` used `p33`/`p66` with `dir()` scope check that always returned False inside lambda, causing all S2 variants to map to `lambda t: True`. Fix: stored thresholds as module-level `_bs_p33`/`_bs_p66`.
   - Before: S2_low/mid/high all showed IS n=27, OOS n=70 (identical = unfiltered)
   - After: S2_low IS n=7, S2_mid IS n=4, S2_high IS n=9 (correct per-tercile split)
2. **S4 PnL key resolution**: `_resolve_pnl_key()` found `T+2` before checking S4a_OPEN/S4c_1030, so all S4 variants used `pnl_t2` instead of entry-specific keys. Fix: check S4 entry variants FIRST.
   - Before: S4a_OPEN IS avg=+4.76% (same as S4b = baseline)
   - After: S4a_OPEN IS avg=+8.54% (correctly using pnl_open_t2)
3. **C1-S6 lookback filter missing**: No matching conditions for C1S6 experiment IDs (e.g. `C1S6_T+3_C1_3d_-10%`), causing all to fall through to `lambda t: True`. Fix: added 5 lookback filter conditions.
   - Before: all C1S6 showed IS n=45, OOS n=131 (identical = full C1)
   - After: C1S6_3d IS n=24, C1S6_15d IS n=9 (correct per-lookback splits)

**Corrected results (8 pension-ready):**
- S2_T+2_S2_mid: IS +7.07%/1.137 Sharpe, OOS +5.45%/0.647 Sharpe
- S3_T+2_S3b_vol<2x: IS +6.35%/0.744, OOS +4.57%/0.373
- S4_T+2_S4a_OPEN: IS +8.54%/1.059, OOS +7.57%/0.626 (⚠ lookahead)
- S4_T+2_S4b_LOCAL_LOW: IS +4.76%/0.591, OOS +4.34%/0.368 (= Phase 1 A6_SR)
- S4_T+3_S4a_OPEN: pension ready (⚠ lookahead)
- S6_T+2_S6_CS_only / S6_all: = Phase 1 A6_SR baseline (trivially same)
- **N5C1_T+3_WedThu**: IS +7.22%/0.565, OOS +13.13%/0.848 — STRONGEST

### Phase 2 Investigation
- Created `agent3_phase2.py` — Phase 2 subsegmentation of A6 sustained_rise and C1 reversal
- 99 experiments across 4 groups (G1: A6 S1-S6, G2: C1 S1-S6, G3: Combos, G4: New Ideas)
- Loads Phase 1 cache (phase1_cache.pkl) for fast rerun
- Key findings: N5C1_T+3_WedThu is star discovery; bull regime kills both strategies; A6+C1 uncorrelated

## 2026-09-15 — Initial Investigation (v1 + v2 OOS fix)

### v2 — OOS Validation Bug Fix
- **Bug:** `run_oos_validation()` used unfiltered trade lists for all experiments.
  - A5_D2, A6_sustained_rise, etc. were all validated on the full 365-trade A list instead of their filtered subsets (82, 98 trades respectively).
  - All A-prefix experiments showed identical IS/OOS counts (n=94/275), confirming the trades were not filtered.
- **Fix:** Added `_resolve_trade_filter(exp_id)` function that maps each experiment ID to its exact filter function:
  - `A5_D1..D5` → gap_pct range filters
  - `A6_sustained_rise` → `open_pattern == 'sustained_rise'`
  - `A6_rise_then_drop` → `open_pattern == 'rise_then_drop'`
  - `A6_mild_rise` → `open_pattern == 'mild_rise'`
  - `A6_continued_drop` → `open_pattern == 'continued_drop'`
  - `A3_*` → `is_cs == True`
  - `B2_Vol confirmed` → break_bar_vol > 1.5x avg_orb_vol
  - `B3_*` → break_pct tier filters
  - `C1_*` → `hyp_c_type.startswith('C1')`
  - `C2_*` → `'C2' in hyp_c_type`
  - `A1, A7, B1` → no filter (full trade list)
- **Result:** Corrected OOS validation reveals:
  - **A6 sustained_rise T+2 is PENSION READY** (IS: +4.76%/0.591 Sharpe, OOS: +4.27%/0.365 Sharpe)
  - A5 D2 degrades IS→OOS (IS +3-8% → OOS +1-2%)
  - C1 T+3 has excellent OOS but IS CI crosses zero (short IS period)

### v1 — Initial Investigation
- Created `long_investigation.py` — complete Colab-friendly script (~900 lines)
- Three independent LONG hypotheses tested:
  - **A: Gap-Down Mean Reversion** — mirror of Agent 1's SHORT fade
  - **B: Gap-Up Momentum (ORB)** — opening range breakout continuation
  - **C: Multi-Day Reversal** — oversold bounce + gap-up confirmation
- 62 experiment configurations across 250 trading days (2025-09-15 → 2026-09-12)
- Universe: _FLOAT (194 tickers with float ≥ 10M)
- Constraint: LONG only, no shorting, no leverage (pension account)

**New components built:**
- `find_local_low()` — mirror of Agent 1's `find_local_high()`
- `find_orb_break_up()` — ORB breakout detection
- `classify_gap_down_pattern()` — first-60-min pattern classification
- `collect_data()` — single-pass scanner for all 3 hypotheses
- `enrich_long()` — LONG PnL computation
- 10 experiment functions: run_a1, run_a3, run_a5, run_a6, run_a7, run_b1, run_b2, run_b3, run_c1, run_c2
- `run_oos_validation()` — IS/OOS split with experiment-specific filtering
- `plot_summary()` — Sharpe + CI charts for all experiments

**Key findings:**
- Hypothesis B (ORB momentum) is DEAD — negative at all configs
- A6 sustained_rise T+2 is best config: +4.41%, Sharpe 0.409, 62% WR
- C1 reversal T+3 is strongest absolute: +5.77%, Sharpe 0.398, 62% WR
- rise_then_drop is a strong ANTI-signal (-4% to -6%)
- Only A6 sustained_rise T+2 passes full OOS pension validation

**Skipped experiments (require BigQuery):**
- A2: Catalyst exclusion
- A4: Earnings-only gap-down
- B4: Catalyst-confirmed momentum (hypothesis B is dead anyway)

### Files
| File | Description |
|------|-------------|
| `long_investigation.py` | Main investigation script (~900 lines) |
| `long_investigation_output.txt` | Full console output from v2 run |
| `agent3_summary.png` | Sharpe + CI summary chart |
| `agent3_a7_hold_sweep.png` | Hold period sweep chart |
| `AGENT3_RESULTS.md` | Detailed results document (this companion) |
| `AGENT3_CHANGELOG.md` | This file |
