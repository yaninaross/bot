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
  Gap filter:  gap_pct >= 5% (gap-UP from prior close)
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

## File Structure
  agent1.py              main scanner (live trading, 6:10 AM PST)
  agent1_backtest.py     backtest framework
  experiments/           one new .py file per experiment
  results/               JSON results, one per experiment
  data/agent1.db         SQLite trade log
  data/float_cache.json  {ticker: float_shares} — grows over time
