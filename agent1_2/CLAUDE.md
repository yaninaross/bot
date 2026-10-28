# Agent 1_2 — Project X SHORT Scanner (H006 + DEEP4 Proximity)

## What This Agent Does
Live SHORT scanner derived from Project X investigation winner H006.
Scans gap-up stocks, waits for sustained_drop pattern (bar 60),
enters at local_high within 3% of running intraday high, exits 17:00 ET.
Same-day intraday only — no overnight holds.

## Strategy Parameters (FROZEN)
  Universe:    _FLOAT_SET (tickers with float >= 5M shares)
  Gap filter:  gap_pct >= 2%
  Price filter: >= $5.00
  Gap cap:     <= 75%
  Asset type:  CS only (verified via Polygon reference API)
  Pattern:     sustained_drop (classified at bar 60 = 10:30 AM ET)
               rise_30m < -3% AND recovery_at_bar60 < 50%
  Entry:       local_high (3-bar lookback + 3-bar confirmation)
               MUST be within 3% of running intraday high (DEEP4)
               Bars 3 through 72 (no fallback — quality only)
  Exit:        17:00 ET (2:00 PM PT) — after-hours close
  Position:    $100,000 per signal, equal weight
  Top N:       5 signals max per day

## Backtest Results (Aug-Oct 2026)
  avg +2.25%  |  Sharpe 0.749  |  WR 93%  |  PF 14.14  |  n=28
  All 14 months positive (Sep 2025 - Oct 2026)

## Two-Phase Scan
  Phase A (6:31 PT): Gap scan, float/price filter, CS verify
  Phase B (7:30 PT): Bar 60 pattern check, local_high + proximity

## Modes
  MODE = 'scan'    Morning scanner (default)
  MODE = 'fill'    Fill outcomes after 2 PM PT for open signals
  MODE = 'report'  Performance summary

## Output Format
  Matches the user's required signal format:
  - Ticker, entry price, signal time
  - Gap %, drop from open, proximity to running high
  - Position size + shares
  - Exit reminder at 2:00 PM PT
  - Watched candidates with reject reasons

## File Structure
  agent1_2.py          main scanner
  data/agent1_2.db     SQLite signal log
  logs/agent1_2.log    debug log

## Key Differences from Agent 1 (E2-A)
  1. Lower thresholds: gap>=2% (vs 3%), price>=$5 (vs $10), float>=5M (vs 10M)
  2. Same-day exit at 17:00 ET (vs T+2 close)
  3. Proximity filter: entry must be within 3% of running high
  4. No fallback entry — quality signals only
  5. Independent of agent1 — does not share DB or scan pool
