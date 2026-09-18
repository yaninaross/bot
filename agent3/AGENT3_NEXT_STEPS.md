# Agent 3 — Complete Baseline and Backtest Specification

## PART 1: BASELINES

These are the frozen reference implementations. Every subsequent experiment is measured against them.

### BASELINE-A6 — Gap-Down Sustained Rise

```
NAME:           BASELINE-A6
DIRECTION:      LONG
UNIVERSE:       _FLOAT tickers only
                (194 tickers with float >= 10M shares)

GAP FILTER:     gap_pct <= -0.05
                (stock gapped DOWN 5% or more from prior close)
                gap_pct = (open_price - prior_close) / prior_close

ASSET FILTER:   CS type only (common stock)
                Exclude: ETF, ETN, ETV, ETS, ADRC, FUND, PFD

PRICE FILTER:   open_price >= $10.00

PATTERN FILTER: sustained_rise == True
                Classified at bar 60 (10:30 AM ET)
                Using first 60 minute bars:
                  rise_30m = (max_high_bars[0:30] - open) / abs(open)
                  recovery = (close_bar60 - open) / (max_high_bars[0:30] - open)
                  sustained_rise = rise_30m > 0.03 AND recovery > 0.50

ENTRY:          First local low detection
                A bar at index i is a local low if:
                  low[i] < low[i-1], low[i-2], low[i-3]
                  AND low[i] < low[i+1], low[i+2], low[i+3]
                Entry price = close of bar i+3 (confirmation bar)
                Scan bars index 3 through 11 (9:33-9:41 AM ET)
                FALLBACK: if no local low by bar 11,
                          enter at close of bar 15 (9:45 AM ET)
                SKIP:     if price at bar 15 is UP more than 15%
                          from open — bounce already happened

HOLD:           T+1 close  [UPDATED from T+2 per E1/E1b analysis]
                (regular session closing price
                 1 trading day after entry date,
                 skip weekends and holidays)
                E1 result: T+1 Sharpe 0.600 vs T+2 0.458 (+31%)
                E1b result: pre-market 08:00 peak Sharpe 0.703, WR 79%
                            but 16:00 close used for execution safety

STOP LOSS:      None

SELECTION:      Top 5 by abs(gap_pct) descending
                (largest gap-down candidates first)

PNL FORMULA:    (t1_close - entry_price) / entry_price * 100

CAPITAL MODEL:  $100,000 per day, equal split across active trades,
                non-compounding
```

### BASELINE-C1 — Multi-Day Reversal

```
NAME:           BASELINE-C1
DIRECTION:      LONG
UNIVERSE:       _FLOAT tickers only

PRIOR DECLINE:  5-day cumulative return <= -0.15
                prior_5d_return = (prior_close - close_5d_ago) / close_5d_ago
                Must use actual trading day closes (skip weekends/holidays)

GAP FILTER:     gap_pct >= 0.03
                (stock gapped UP 3% or more today)
                This is the reversal signal after the prior decline

PRICE FILTER:   open_price >= $20.00
                (higher floor than A6 — pension quality)

ASSET FILTER:   CS type only

ENTRY:          Market open price (9:30 AM ET)
                entry_price = open_price
                No local low detection — enter immediately at open
                No skip rule for C1

HOLD:           T+3 close
                (regular session closing price
                 3 trading days after entry date,
                 skip weekends and holidays)

STOP LOSS:      None

SELECTION:      All qualifying trades that day
                (no top-N limit — take every qualifying trade)

PNL FORMULA:    (t3_close - entry_price) / entry_price * 100

CAPITAL MODEL:  $100,000 per day, equal split across active trades,
                non-compounding
```

---

## EXPERIMENT RESULTS: E1 Hold Period + E1b Exit Hour (2026-09-15)

### E1: Hold Period Grid

A6 N×Hold (5×6 = 30 cells):
- T+1 dominates ALL N values (Sharpe peak at T+1 for every row)
- Best cell: N=2 T+1 — Sharpe 0.600, +5.29%, WR 67%, n=57, IS/OOS stable (+4.88/+5.44%)
- Baseline T+2 was suboptimal (Sharpe 0.458) → UPDATED to T+1
- Sharpe degrades monotonically past T+1: 0.600 → 0.540 → 0.399 → 0.326 → 0.331
- Pool-adjusted $: EOD best ($150K, 1 pool) but lower Sharpe (0.508). T+1 = $130K, 2 pools.

C1 Hold (6 cells):
- T+3 baseline CONFIRMED optimal (Sharpe 0.394, +5.56%, WR 62%, n=173)
- Clean inverted-U: EOD(0.227) → T+1(0.342) → T+2(0.384) → T+3(0.394) → T+4(0.326) → T+5(0.227)
- Pool $ best at T+2 ($73K vs $69K T+3) — marginal, keep T+3 for Sharpe

### E1b: T+1 Exit Hour (A6 only)

N × 18 exit hours (04:00-20:00, pre-market + RTH + after-hours):
- Pre-market peak: N=4 @08:00 — Sharpe 0.703, +4.29%, WR 79%, n=62
  IS Sharpe 0.743 → OOS 0.676 (stable)
- All N values best at 06:00-08:00 pre-market
- WR 75-79% in pre-market vs 58-67% in RTH — overnight bounce very reliable
- RTH open dip at 09:30 (~-0.10 Sharpe) — T+1 open volatility hurts
- Second peak at 16:00 close: Sharpe 0.56-0.60, higher avg PnL (+4.7-5.3%), better Pool $
- After-hours 17:00-20:00 flat vs 16:00 — no incremental edge

Decision: Use 16:00 close for execution safety. Pre-market 08:00 is informational upside
(+0.10 Sharpe, +14% WR) but requires pre-market execution capability.

Scripts: `agent3_e1_hold_grid.py`, `agent3_e1b_exit_hour.py`

---

## EXPERIMENT RESULTS: E2 Universe Comparison (2026-09-15)

8 universes tested with A6 T+1 and C1 T+3. _FLOAT is baseline (194 tickers).
U1: Full CS (hard float), U2: Large cap ≥$1B, U3: Mid cap $300M-$10B,
U4: Russell 1000 ≥$3B, U5: High ADV ≥$10M/day, U6: CS ex biotech, U7: _FLOAT+guarded.

A6 results (Sharpe): _FLOAT 0.542, U3 0.332, U7 0.306, U2 0.222, U4 0.197, U1 0.176, U6 0.176, U5 0.100.
C1 results (Sharpe): _FLOAT 0.391, U4 0.388, U7 0.385, U2 0.361, U3 0.289, U1 0.168, U6 0.168, U5 0.013.

Conclusion: _FLOAT optimal for A6. U4 nearly matches for C1 (350 trades vs 176, CI_lo 3.74 vs 3.41).
Expanding beyond 194-ticker _FLOAT consistently dilutes quality.

Script: `agent3_e2_universe.py`

---

## EXPERIMENT RESULTS: E3 Strategy × Universe Grid (2026-09-16)

Tested ALL Phase 1 strategies (24 variants: A1-A7, B1-B3, C1-C2) across 7 universes:
  _FLOAT (194 hardcoded), M1_200 (top 200 by avg opening vol), M1_500 (top 500),
  M2_15 (gap freq ≥15%), M3_60 (mean-reversion rate ≥60%), M5_200 (top 200 composite),
  M5_500 (top 500 composite).

Universe construction: 93-day lookback (2025-05-01 → 2025-09-12), 4,301 CS tickers scored.
M1/M2/M3/M5 scores computed from grouped daily data. M5 composite = 0.35×M1 + 0.25×M2 + 0.25×M3 + 0.15×ADV.

Universe overlap with _FLOAT:
  M1_200: 5/194 (3%), M1_500: 19/194 (10%), M2_15: 44/194 (23%),
  M3_60: 43/194 (22%), M5_200: 40/194 (21%), M5_500: 67/194 (35%).
  Union: 1,510 tickers. Total: A=3,004 B=3,257 C=1,125 candidates.

### KEY FINDING: _FLOAT wins EVERY strategy. No M-universe beats _FLOAT on any metric.

Sharpe grid (top strategies):
  A6sr T+1: _FLOAT 0.439★ | M1_200 0.252 | M1_500 0.164 | M2_15 0.157 | M3_60 0.209 | M5_200 0.177 | M5_500 0.209
  A6sr T+2: _FLOAT 0.364~ | M1_200 0.179 | M1_500 0.098 | M2_15 0.099 | M3_60 0.122 | M5_200 0.135 | M5_500 0.071
  C1 T+3:   _FLOAT 0.373~ | M1_200 -0.244 | M1_500 -0.251 | M2_15 -0.245 | M3_60 -0.208 | M5_200 0.027 | M5_500 0.079
  C1 T+5:   _FLOAT 0.216  | all others negative

Only 3 cells out of 168 pass CI>0 AND Sharpe≥0.30 — ALL are _FLOAT:
  A6sr T+1 _FLOAT: Sharpe 0.439, +4.22%, n=116, CI [+2.47, +5.97] — PENSION
  C1 T+3 _FLOAT:   Sharpe 0.373, +5.61%, n=207, CI [+3.56, +7.65] — NEAR MISS
  A6sr T+2 _FLOAT:  Sharpe 0.364, +4.16%, n=116, CI [+2.08, +6.25] — NEAR MISS

Avg PnL destruction in expanded universes:
  A1 T+5: _FLOAT +0.97% vs M2_15 -11.41% — expanded universe actively harmful
  C1 T+3: _FLOAT +5.61% vs M1_200 -9.91% — gap-magnitude selection picks losers

Why M-universes fail:
  M1/M2 top tickers (CWD 25%, HCTI 13%, SBET 10%) are ultra-small-cap with huge gaps but no reversion.
  M3 selects on 3-7 events — massive sampling noise (100% reversion on 3 trades is meaningless).
  M5 composite inherits M1/M2 contamination.
  Hypothesis B (ORB momentum) stays DEAD across all universes.

OOS validation (top 3):
  A6sr T+1 _FLOAT: IS 0.587 → OOS 0.402, +4.37%/+4.17% — ★ PENSION READY
  A6sr T+2 _FLOAT: IS 0.374 → OOS 0.365, +3.18%/+4.54% — ★ PENSION READY
  C1 T+3 _FLOAT:   IS 0.174 → OOS 0.448, +2.71%/+6.60% — ✗ (IS CI fails)

### CONCLUSION: UNIVERSE INVESTIGATION CLOSED.
_FLOAT is the correct universe. Systematic reconstruction (opening vol, gap frequency,
mean-reversion rate, composite) cannot replicate what _FLOAT captured. The 194-ticker set
has implicit quality characteristics beyond raw volatility metrics. All future experiments
should use _FLOAT only. No further universe experiments planned.

Script: `agent3_e3_strategy_universe.py`

---

## PART 2: METRICS (Apply to Every Experiment)

For each experiment report ALL of the following:

```
BASIC METRICS:
  n                  total qualifying trades
  tpd                trades per day (n / trading_days)
  avg_pnl            mean PnL %
  median_pnl         median PnL %
  trimmed_mean       10% trimmed mean (drop top/bottom 5%)
  std                standard deviation of PnL
  win_rate           % trades with PnL > 0
  sharpe             avg_pnl / std
  profit_factor      sum(winners) / abs(sum(losers))
  max_loss           worst single trade PnL %
  max_gain           best single trade PnL %
  ci_95_low          avg - 1.96 * (std / sqrt(n))
  ci_95_high         avg + 1.96 * (std / sqrt(n))
  ci_confirmed       True if ci_95_low > 0

OOS SPLIT at 2026-01-01:
  is_n               n in IS period
  is_avg             avg PnL in IS period
  is_sharpe          Sharpe in IS period
  is_ci_low          95% CI lower bound in IS
  oos_n              n in OOS period
  oos_avg            avg PnL in OOS period
  oos_sharpe         Sharpe in OOS period
  oos_ci_low         95% CI lower bound in OOS
  degradation_pct    (oos_avg - is_avg) / abs(is_avg) * 100

STATUS CLASSIFICATION:
  PENSION READY:  ci_confirmed=True
                  AND oos_ci_low > 0
                  AND oos_avg >= 3.0%
                  AND oos_sharpe >= 0.15
                  AND win_rate >= 58%
                  AND degradation_pct > -50%

  NEAR MISS:      ci_confirmed=True
                  AND oos_avg >= 3.0%
                  AND oos_sharpe >= 0.15
                  AND is_ci_low fails due to is_n < 50

  DEGRADED:       oos_avg < 60% of is_avg

  DEAD:           oos_avg < 3.0% OR oos_ci_low < 0

ADDITIONAL OUTPUT:
  worst_5_trades     (date, ticker, gap_pct, entry_type, pnl)
  best_5_trades      (date, ticker, gap_pct, entry_type, pnl)
  monthly_breakdown  (month, n, avg_pnl, dollar_pnl)
```

---

## PART 3: A6 EXPERIMENTS (in order)

Run each against BASELINE-A6. Change only the specified parameter.

### A6-E01: Volume Filter
```
CHANGE:         Add volume filter
FILTER:         vol_ratio = today_volume / adv_20d
                EXCLUDE if vol_ratio >= 2.0
                (keep only low-volume bounce days)
                If ADV data not available: include (do not exclude)
ALL ELSE:       Identical to BASELINE-A6
HYPOTHESIS:     Low-volume bounces are cleaner institutional accumulation.
                High-volume bounces attract sellers.
REFERENCE:      S3_T+2_S3b_vol<2x — PENSION READY in Phase 2
EXPECTED:       n drops from 97 to ~66, avg improves ~+0.7pp
```

### A6-E02: Regime Filter — Exclude Bull
```
CHANGE:         Add market regime filter
FILTER:         spy_5d_return = (spy_close_today - spy_close_5d_ago) / spy_close_5d_ago
                EXCLUDE if spy_5d_return > 0.02 (SPY up >2% over 5 days)
ALL ELSE:       Identical to BASELINE-A6
HYPOTHESIS:     A6 dies in bull markets (-0.26% avg when SPY >+2%).
                Flat and bear regimes carry the edge.
REFERENCE:      N1A6_T+2_flat — SPY flat gives +4.61% avg
EXPECTED:       Small n reduction, meaningful Sharpe improvement
```

### A6-E03: Prior Trend Filter — Exclude Prior Downtrend
```
CHANGE:         Add prior trend filter
COMPUTATION:    prior_5d_return = return of the stock itself
                over the 5 trading days BEFORE the gap-down day
                prior_5d = (close_5d_before_gap - close_10d_before_gap)
                           / close_10d_before_gap
FILTER:         EXCLUDE if prior_5d_return < -0.05
                (stock was already in downtrend before gap-down)
ALL ELSE:       Identical to BASELINE-A6
HYPOTHESIS:     Gap-downs during prior uptrend = overreaction.
                Gap-downs during prior downtrend = continuation.
REFERENCE:      N2_prior_down (<-5%): avg +1.41%, fails CI
                N2_prior_up (>+5%): avg +5.12%, CI confirmed
EXPECTED:       n drops ~20%, avg improves ~+0.7pp
```

### A6-E04: Gap Tier — Small (D1 Only)
```
CHANGE:         Restrict to small gap-down tier
FILTER:         -0.08 <= gap_pct <= -0.05
                (gap-down between 5% and 8%)
ALL ELSE:       Identical to BASELINE-A6
HYPOTHESIS:     Test whether edge concentrates in specific gap range.
REFERENCE:      S1_D1: avg +4.46%, n=61, CI confirmed
EXPECTED:       n=61 (63% of baseline), avg similar to baseline
```

### A6-E05: Gap Tier — Medium (D2 Only)
```
CHANGE:         Restrict to medium gap-down tier
FILTER:         -0.12 <= gap_pct < -0.08
                (gap-down between 8% and 12%)
ALL ELSE:       Identical to BASELINE-A6
HYPOTHESIS:     Larger gaps have more mean-reversion room.
REFERENCE:      S1_D2: avg +5.02%, Sharpe 0.769, n=19
                BUT: degrades OOS (-71%) due to tiny IS n
EXPECTED:       Small n (~19), higher per-trade avg, OOS uncertain
```

### A6-E06: Hold T+3
```
CHANGE:         Extend hold period to T+3
ALL ELSE:       Identical to BASELINE-A6
HYPOTHESIS:     Does the bounce continue building through day 3?
REFERENCE:      Phase 1 A6_SR_T+3: avg +3.84%, Sharpe 0.244
                Full-period CI confirmed, OOS fails
EXPECTED:       Slightly lower avg than T+2 at 60m but tests persistence
```

### A6-E07: Day of Week — Monday Only
```
CHANGE:         Restrict to Monday entries
FILTER:         entry_date.weekday() == 0
ALL ELSE:       Identical to BASELINE-A6
HYPOTHESIS:     A6 on Mondays: avg +7.80%, Sharpe 0.810 in Phase 2
                (opposite of C1 which is worst on Mondays)
REFERENCE:      S5_Monday: n=11, avg +7.80%, CI confirmed
EXPECTED:       Very small n (~11), high avg, IS CI uncertain
```

### A6-E08: Day of Week — Tuesday Only
```
CHANGE:         Restrict to Tuesday entries
FILTER:         entry_date.weekday() == 1
ALL ELSE:       Identical to BASELINE-A6
REFERENCE:      S5_Tuesday: n=19, avg +5.14%, but degrades OOS
EXPECTED:       Small n, moderate avg
```

### A6-E09: Day of Week — Wed/Thu Only
```
CHANGE:         Restrict to Wednesday or Thursday entries
FILTER:         entry_date.weekday() in [2, 3]
ALL ELSE:       Identical to BASELINE-A6
REFERENCE:      S5_WedThu: n=45, avg +4.98%, CI confirmed, NEAR MISS OOS
EXPECTED:       ~45 trades, avg near baseline, tests OOS consistency
```

### A6-E10: Bounce Score Filter — Middle Tercile
```
CHANGE:         Add bounce quality score filter
COMPUTATION:    rise_proxy = (max_high_bars[0:30] - open) / abs(open)
                recovery = (close_bar60 - open) / (max_high_bars[0:30] - open)
                bounce_score = (rise_proxy / abs(gap_pct)) * recovery
                Compute P33 and P66 thresholds from the training data
FILTER:         Keep only trades where bounce_score >= P33
                AND bounce_score <= P66 (middle tercile)
ALL ELSE:       Identical to BASELINE-A6
HYPOTHESIS:     Mid-bounce is the sweet spot — not too weak, not too strong.
REFERENCE:      S2_mid: avg +5.72%, Sharpe 0.715, n=24
WARNING:        IS n=4 is dangerously thin — treat result with caution
EXPECTED:       n=24, higher avg, OOS uncertain due to thin IS
```

### A6-E11: Combined Best (Vol + Regime)
```
CHANGE:         Stack two confirmed filters simultaneously
FILTERS:        vol_ratio < 2.0 (from E01)
                AND spy_5d_return <= 0.02 (from E02)
ALL ELSE:       Identical to BASELINE-A6
HYPOTHESIS:     Two independent filters stack additively.
EXPECTED:       n drops further (~40-50 trades), avg improves
                This is the first composite test for A6
```

### A6-E12: Combined Best + Prior Trend
```
CHANGE:         Add prior trend filter to E11
FILTERS:        vol_ratio < 2.0
                AND spy_5d_return <= 0.02
                AND prior_stock_5d_return > -0.05
ALL ELSE:       Identical to BASELINE-A6
HYPOTHESIS:     Three filters combined — find the clean core A6 signal.
EXPECTED:       n ~30-40, tests whether filters are additive or redundant
```

---

## PART 4: C1 EXPERIMENTS (in order)

Run each against BASELINE-C1. Change only the specified parameter.

### C1-E01: Day of Week — Wednesday/Thursday Only
```
CHANGE:         Restrict to Wednesday or Thursday entries
FILTER:         entry_date.weekday() in [2, 3]
ALL ELSE:       Identical to BASELINE-C1
HYPOTHESIS:     The entire C1 edge concentrates in Wed/Thu entries.
                Mon: +0.51%, Tue: -1.02%, Wed/Thu: +11.21%, Fri: +5.41%
REFERENCE:      N5C1_WedThu — PENSION READY, OOS +13.13%, Sharpe 0.848
EXPECTED:       n=77 (44% of baseline), avg dramatically higher
STATUS:         This is the highest-priority C1 experiment
```

### C1-E02: Day of Week — Monday Only
```
CHANGE:         Monday entries only
FILTER:         entry_date.weekday() == 0
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      N5C1_Monday: n=46, avg +0.51% — essentially dead
EXPECTED:       Near-zero or negative avg — confirms Monday is worst
```

### C1-E03: Day of Week — Friday Only
```
CHANGE:         Friday entries only
FILTER:         entry_date.weekday() == 4
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      N5C1_Friday: n=29, avg +5.41%, CI borderline
EXPECTED:       Moderate performance, OOS uncertain
```

### C1-E04: Decline Threshold — 20%
```
CHANGE:         Tighten prior 5-day decline requirement
FILTER:         prior_5d_return <= -0.20 (was -0.15)
ALL ELSE:       Identical to BASELINE-C1
HYPOTHESIS:     More oversold = stronger reversal signal.
REFERENCE:      C1S1_20%: avg +9.25%, Sharpe 0.548, n=77
                OOS: +10.42%, Sharpe 0.626, CI [+6.10, +14.74]
EXPECTED:       n=77 (44% of baseline), avg +3.5pp improvement
```

### C1-E05: Decline Threshold — 25%
```
CHANGE:         Further tighten prior decline
FILTER:         prior_5d_return <= -0.25
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      C1S1_25%: avg +9.74%, Sharpe 0.505, n=29
EXPECTED:       n=29, highest per-trade avg, OOS uncertain (thin IS)
```

### C1-E06: Gap-Up Threshold — 5%
```
CHANGE:         Require larger gap-up on reversal day
FILTER:         gap_pct >= 0.05 (was 0.03)
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      C1S2_gap5: avg +5.81%, Sharpe 0.378, n=90
EXPECTED:       n=90, similar avg to baseline, tests monotonic relationship
```

### C1-E07: Gap-Up Threshold — 8%
```
CHANGE:         Require strong gap-up on reversal day
FILTER:         gap_pct >= 0.08
ALL ELSE:       Identical to BASELINE-C1
HYPOTHESIS:     Gap-up >=8% = institutional accumulation signal.
REFERENCE:      C1S2_gap8: avg +9.98%, Sharpe 0.572, n=35
                OOS: +10.83%, Sharpe 0.689, CI [+5.01, +16.66]
EXPECTED:       n=35, avg nearly doubles vs baseline
```

### C1-E08: Gap-Up Threshold — 10%
```
CHANGE:         Require very strong gap-up
FILTER:         gap_pct >= 0.10
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      C1S2_gap10: avg +11.78%, Sharpe 0.585, n=24
                OOS: +13.92%, Sharpe 0.752, CI [+5.12, +22.72]
EXPECTED:       n=24, highest avg in gap threshold sweep
```

### C1-E09: Gap-Up Threshold — 12% (D3 Signal)
```
CHANGE:         Require extreme gap-up on reversal day
FILTER:         gap_pct >= 0.12
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      COMBO3_D3_12%+: avg +16.85%, Sharpe 0.863, n=11
                OOS: +13.78%, Sharpe 0.864
WARNING:        n=11 total, n=8 OOS — too thin for confirmation
EXPECTED:       Very small n, extraordinary avg — log every occurrence
```

### C1-E10: Hold T+2
```
CHANGE:         Shorten hold period
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      C1S4_T+2: avg +4.97%, Sharpe 0.377
EXPECTED:       Lower avg than T+3, tests if T+3 is genuinely optimal
```

### C1-E11: Hold T+4
```
CHANGE:         Extend hold period by 1 day
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      C1S4_T+4: avg +5.08%, Sharpe 0.338
EXPECTED:       Similar to T+3, tests persistence of reversal
```

### C1-E12: Hold T+5
```
CHANGE:         Extend hold to 5 days
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      C1S4_T+5: avg +4.24%, Sharpe 0.250
EXPECTED:       Lower than T+3, decay begins
```

### C1-E13: Hold T+10
```
CHANGE:         Hold 10 trading days
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      C1S4_T+10: avg +7.70%, Sharpe 0.320, WR 55%
                Surprise spike but lower WR — driven by tail winners
EXPECTED:       Higher avg than T+3 in some configs, investigate structure
```

### C1-E14: Lookback 3 Days
```
CHANGE:         Shorten lookback window for prior decline
FILTER:         prior_3d_return <= -0.10
                (3-day decline of 10%+ instead of 5-day 15%)
ALL ELSE:       Identical to BASELINE-C1 (still gap-up >=3% today)
REFERENCE:      C1S6_3d_-10%: avg +7.15%, Sharpe 0.498, n=120
                OOS: same as baseline (shares IS constraint)
HYPOTHESIS:     3-day sharp drops may be sharper overreactions.
EXPECTED:       n=120, higher avg than baseline, more frequent trades
```

### C1-E15: Lookback 7 Days
```
CHANGE:         7-day lookback, -18% threshold
FILTER:         prior_7d_return <= -0.18
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      C1S6_7d_-18%: avg +6.11%, Sharpe 0.411, n=108
EXPECTED:       Between 3d and 5d baselines in n and avg
```

### C1-E16: Lookback 15 Days
```
CHANGE:         15-day lookback, -25% threshold
FILTER:         prior_15d_return <= -0.25
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      C1S6_15d_-25%: avg +8.19%, Sharpe 0.525, n=59
EXPECTED:       Fewer but better-quality trades, deeply oversold
```

### C1-E17: Regime Filter — Exclude Bull
```
CHANGE:         Add SPY regime filter
FILTER:         spy_5d_return <= 0.02 (exclude bull SPY)
ALL ELSE:       Identical to BASELINE-C1
REFERENCE:      N1C1_flat: avg +6.50%, Sharpe 0.423, n=138
                Bull: avg +1.28%, Sharpe 0.133 — kills the edge
EXPECTED:       n reduction ~10%, avg improvement ~+0.7pp
```

### C1-E18: CS Only (Already Baseline — Confirm)
```
CHANGE:         Explicit CS-only filter (confirm all _FLOAT C1 are CS)
REFERENCE:      C1S5: 173/176 CS, 3 ADRC — essentially 100% CS already
EXPECTED:       Identical to baseline — confirms no action needed
```

---

## PART 5: COMBINATION EXPERIMENTS

Run after individual experiments complete.

### COMBO-E01: Best A6 + Best C1 Combined Portfolio
```
STRATEGY A:     BASELINE-A6 + E01 (vol<2x) + E02 (regime not bull)
STRATEGY C:     BASELINE-C1 + E01 (WedThu) + E04 (decline >=20%) + E07 (gap >=8%)

PORTFOLIO:      Run both simultaneously
                Each morning: check A6 candidates AND C1 candidates
                Trade both independently, capital split equally per active trade
                Log ticker+date overlap (expected: 0)
                Log date overlap (days when both fire)

METRICS:        Everything from standard metrics PLUS:
                combined_n, combined_tpd
                correlation_daily_pnl (are the daily returns correlated?)
                max_simultaneous_positions
                monthly_breakdown for combined portfolio
REFERENCE:      Phase 2 COMBO baseline: n=273, Sharpe 0.399, +$544K/13mo
EXPECTED:       Similar Sharpe to individual strategies,
                improved consistency from diversification
```

### COMBO-E02: C1 WedThu + Decline 20% + Gap 8%
```
CHANGE:         Stack three C1 improvements simultaneously
FILTERS:        entry day in [Wed, Thu]
                prior_5d_return <= -0.20
                gap_pct >= 0.08
ALL ELSE:       Identical to BASELINE-C1
HYPOTHESIS:     These three filters are each independently validated.
                Do they stack additively or are they correlated?
EXPECTED:       n probably 15-25 trades over full period
                Very high per-trade avg if filters are independent
                IS will fail (too few) — OOS is the key metric
```

### COMBO-E03: A6 WedThu vs A6 MonTue
```
CHANGE:         Compare A6 by day subset back to back
FILTER A:       entry day in [Wed, Thu] (A6 is WORSE on these days per S5)
FILTER B:       entry day in [Mon, Tue] (A6 is BETTER on Mon/Tue)
ALL ELSE:       Identical to BASELINE-A6
HYPOTHESIS:     A6 and C1 have COMPLEMENTARY day-of-week profiles.
                If confirmed: run A6 on Mon/Tue, C1 on Wed/Thu.
EXPECTED:       A6 Mon/Tue outperforms A6 Wed/Thu
                Perfect portfolio structure if confirmed
```

---

## PART 6: HOLD PERIOD CURVE (Run on Both Final Configs)

```
EXPERIMENT:     Full PnL curve from intraday to T+20
APPLY TO:       (1) BASELINE-A6 best config after experiments complete
                (2) BASELINE-C1 best config after experiments complete

HOLDS TO TEST:
  30m   = bar 30 close
  60m   = bar 60 close
  EOD   = T+0 regular session close
  T+1   = T+1 close
  T+2   = T+2 close
  T+3   = T+3 close
  T+4   = T+4 close
  T+5   = T+5 close
  T+7   = T+7 close
  T+10  = T+10 close
  T+15  = T+15 close
  T+20  = T+20 close

REPORT:         avg_pnl at each hold period
                Sharpe at each hold period
                win_rate at each hold period
                Plot: avg_pnl curve with 95% CI shading

PURPOSE:        Find natural exhaustion point of the bounce/reversal.
                Where does the edge peak?
                Where does it start decaying?
                Confirms optimal hold period empirically.
```

---

## PART 7: EXECUTION ORDER

**Week 1 — Baselines and highest-priority experiments:**
```
  BASELINE-A6          (reference point)
  BASELINE-C1          (reference point)
  C1-E01 WedThu        (highest priority — confirmed pension-ready)
  A6-E01 Vol<2x        (confirmed pension-ready in Phase 2)
  A6-E02 Regime        (confirmed kills edge in bull)
  C1-E04 Decline 20%   (strong OOS, near-miss due to IS n)
  C1-E07 Gap 8%        (strong OOS, monotonic relationship)
```

**Week 2 — Secondary experiments:**
```
  A6-E03 Prior trend
  A6-E11 Combined vol+regime
  C1-E02 Monday (confirm it's dead)
  C1-E08 Gap 10%
  C1-E14 Lookback 3d
  C1-E17 Regime filter
  C1-E10 Hold T+2 (confirm T+3 optimal)
  C1-E11 Hold T+4
  HOLD PERIOD CURVE for both baselines
```

**Week 3 — Combos and edge cases:**
```
  COMBO-E01 Best A6 + Best C1 portfolio
  COMBO-E02 C1 WedThu + Dec20 + Gap8
  COMBO-E03 A6 MonTue vs WedThu
  C1-E16 Lookback 15d
  C1-E09 Gap 12% (D3 signal)
  A6-E07 Monday only
  A6-E12 Triple filter combo
```

---

## PART 8: OUTPUT FORMAT

For each experiment, produce:
```
  experiment_id        (e.g. "A6-E01")
  description          (one line)
  change_from_baseline (what was modified)
  n, tpd, avg, median, wr, sharpe, pf, ci_low, ci_high, ci_confirmed
  is_n, is_avg, is_sharpe, is_ci_low
  oos_n, oos_avg, oos_sharpe, oos_ci_low
  degradation_pct
  status               (PENSION READY / NEAR MISS / DEGRADED / DEAD)
  worst_5_trades
  best_5_trades
  monthly_breakdown
```

Master summary table at end:
```
  All experiments side by side
  Sorted by: oos_avg descending
  Flag PENSION READY in bold
  Flag NEAR MISS with diamond
  Flag DEGRADED and DEAD with X
```
