# Agent 3 — LONG Pension Strategies

## What This Agent Does
Long-only strategies for pension account (no shorting, no leverage).
Two independent strategies running simultaneously. Pure technical,
no ML. Completely independent of bot-RAG system.

## Universe
_FLOAT (217 tickers, same float_cache.json as Agent 1).
This universe was selected for high opening volatility — this
behavioral characteristic is WHY the strategies work.
Do NOT expand to full US equity unless explicitly asked.
Previous tests confirmed _FLOAT outperforms all expanded universes.

## Strategy A6 — Frozen Baseline
  Direction:   LONG
  Gap:         gap_pct <= -5% (gap-DOWN from prior close)
  Asset:       CS only
  Price:       >= $10.00
  Float:       >= 10M shares (hard exclude on cache miss)
  Pattern:     sustained_rise (classified at bar 60 = 10:30 AM ET)
               rise_30m > 3% AND recovery_at_bar60 > 50%
               rise_30m = (max_high_bars[0:30] - open) / abs(open)
               recovery = (close_bar60 - open) / (max_high_bars[0:30] - open)
  Entry:       LOCAL LOW ONLY — NO FALLBACK
               local low at bar i if:
                 low[i] < low[i-1], low[i-2], low[i-3]
                 AND low[i] < low[i+1], low[i+2], low[i+3]
               entry_price = close[i+3]
               scan bars 3-11
               FALLBACK EXCLUDED: fallback entries avg +0.26%,
               near zero — do not include them
  Skip:        if price at 9:45 UP > 15% from open → SKIP
  Hold:        T+2 close
  PnL:         (t2_close - entry_price) / entry_price * 100
  Reference:   local_low only: avg +4.98%, WR 65%, Sharpe 0.400

## Strategy C1 — Frozen Baseline
  Direction:   LONG
  Prior 5d:    cumulative return <= -15%
               (5 actual trading days, skip weekends/holidays)
  Gap today:   gap_pct >= 3% (gap-UP = reversal signal)
  Asset:       CS only
  Price:       >= $20.00
  Float:       >= 10M shares (hard exclude on cache miss)
  Gap cap:     gap_pct <= 50% (PAVS gapping 127% is not reversal)
  Entry:       Market open price (9:30 AM ET open_price)
               No local low detection — enter at open immediately
               No skip rule
  Hold:        T+3 close (3 trading days after entry)
  PnL:         (t3_close - entry_price) / entry_price * 100
  Reference:   avg +5.79%, WR 61%, Sharpe 0.398, n=176

## Star Finding — Deploy Now
  N5C1_WedThu: C1 entries on Wednesday/Thursday ONLY
  Full period: avg +11.21%, WR 75%, Sharpe 0.755
  IS:          avg +7.22%, CI [+2.21, +12.23] ✓
  OOS:         avg +13.13%, Sharpe 0.848, CI [+8.92, +17.34] ✓
  OOS improves over IS (+82%) — rare signal of robustness
  Monday C1: +0.51% (near zero)
  Tuesday C1: -1.02% (negative) — never trade on Tue

## Known Bugs Fixed — Do NOT Re-introduce
1. PAVS, DBGI, GRI: micro-caps causing -60% to -72% losses
   Fix: price >= $20 + float hard exclude + gap_max <= 50%
   If these tickers appear in results → guards not working
2. A6 fallback entries: avg +0.26%, excluded from baseline
   Never re-add fallback to A6
3. OOS validation bug (v3 fix): S2/S4/S6 variants were returning
   baseline trades not filtered subsets — always verify
   is_n and oos_n differ from baseline when filters applied
4. S4a_OPEN entry: lookahead bias — cannot know sustained_rise
   pattern at 9:30 AM open. Do not use as live strategy.
5. C1 T+1 management rule: costs -0.63%/trade — never implement

## Experiment Queue (run in order, one per session)
| ID        | Description               | Status   | Results file    |
|-----------|---------------------------|----------|-----------------|
| A6-E01    | Vol filter (< 2x ADV)     | PENDING  |                 |
| A6-E02    | Regime excl bull SPY      | PENDING  |                 |
| A6-E03    | Prior trend filter        | PENDING  |                 |
| C1-E01    | WedThu formal backtest    | PENDING  |                 |
| C1-E04    | Decline threshold 20%     | PENDING  |                 |
| C1-E07    | Gap-up threshold 8%       | PENDING  |                 |
| C1-E14    | Lookback 3d (-10%)        | PENDING  |                 |
| COMBO-E01 | A6 + C1 combined portfolio| PENDING  |                 |

## Output Format (same as Agent 1 — required every run)
  n, tpd, avg_pnl, median_pnl, trimmed_mean, std,
  win_rate, sharpe, profit_factor, max_loss, max_gain,
  ci_95_low, ci_95_high, ci_confirmed,
  is_n, is_avg, is_sharpe, is_ci_low,
  oos_n, oos_avg, oos_sharpe, oos_ci_low,
  degradation_pct,
  worst_5_trades, best_5_trades (flag if PAVS/DBGI appear),
  monthly_breakdown

## Session Rules
- One experiment per session. Write results, then stop.
- Never modify baseline files — create new files for experiments
- OOS cutoff: 2026-01-01 (never change)
- Always verify at start: print n_excluded_by_float,
  n_excluded_by_price, n_excluded_by_gap_max
- A6 and C1 are INDEPENDENT — run and report separately
  before reporting combined portfolio metrics

## File Structure
  agent3.py              main script (live trading)
  agent3_backtest.py     backtest framework
  experiments/           one new .py file per experiment
  results/               JSON results, one per experiment
  data/agent3.db         SQLite trade log
  data/float_cache.json  shared with Agent 1
