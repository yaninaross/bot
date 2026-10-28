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

## Agent 1 v2 — Multi-Strategy Scanner (agent1_v2.py)
**Status: ACTIVE — runs alongside agent1.py**

Five filter strategies from feature investigation, all sharing
the same base pipeline (gap >= 3%, $10+, float 10M+, CS only,
AH ref price, gap erosion, local high bars 3-11, exit 17:00 ET).

| ID    | Filter                          | N | PostSh | Avg      | Worst  |
|-------|---------------------------------|---|--------|----------|--------|
| C5    | PM retrace >= 8% + bar0 red     | 1 | 0.948  | +14.64%  | -3.12% |
| C3+C5 | C5 + skip T6 gap (15-30%)       | 1 | 0.849  | +14.34%  | -3.12% |
| C7    | PM close < PM open (after 8:30) | 1 | 0.801  | +12.87%  | -3.12% |
| C9    | bar1 close < bar0 open          | 2 | 0.628  | +8.33%   | -3.18% |
| C3    | Skip T6 gap tier only           | 3 | 0.343  | +3.55%   | -24.89%|

Run at 9:33 ET (6:33 PST). Outputs per-strategy picks with
PRIMARY/BACKUP labels, PM signal metrics, cross-strategy
agreement table, and action summary.

For N=1 strategies (C5, C3+C5, C7), shows N=2 backup pick.
No fallback entry — local high or skip.
No stop loss for C5/C7 (natural tail protection).

## File Structure
  agent1.py              main scanner (live trading, E2-A + E2-B)
  agent1_v2.py           multi-strategy scanner (5 filters, 2026-10-13)
  agent1_backtest.py     backtest framework
  agent1_trade_review.py N sweep for top 5 strategies (backtest)
  agent1_feature_investigation.py  feature/filter ranking analysis
  experiments/           one new .py file per experiment
  results/               JSON results, one per experiment
  data/agent1.db         SQLite trade log
  data/float_cache.json  {ticker: float_shares} — grows over time
