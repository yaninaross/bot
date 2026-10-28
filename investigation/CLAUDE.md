# Project X — Semi-Autonomous Strategy Investigation

## What This Is
Automated hypothesis-driven strategy search across LONG, SHORT,
and combination strategies for US equities. Goal: beat the current
SHORT baseline in the Aug-Oct 2026 compressed regime.

## Baseline to Beat (Aug-Oct 2026)
  E2-A SHORT gap-fade: avg +2.28%, Sharpe 0.185, n=154

## Hard Pass Criteria (ALL must be met)
  1. avg_pnl  > 2.0%
  2. sharpe   > 0.20
  3. n        >= 15
  4. ci_95_lo > 0.0
  5. win_rate >= 50%
  6. max_dd   < 35%
  7. trades/week <= 15
  8. avg_hold >= 1 hour
  9. min_position >= $5,000

## Guardrails
  G1. Never tune a threshold after seeing its result
  G2. Never test the same parameter combination twice
  G3. Stop at 3 winners OR 50 hypotheses
  G4. n < 5 or runtime error = INVALID
  G5. One backtest at a time
  G6. Write every result to results/ before next
  G7. Human-tradeable: clear signal, clear exit, max 15/week

## File Structure
  hypothesis_space.py   20 seed hypotheses + pass criteria
  run_investigation.py  Universal backtest engine + loop
  results/              JSON per hypothesis
  winners/              Markdown deployment specs for winners

## Evaluation Period
  Primary: 2026-08-01 -> 2026-10-06
  Full validation: 2025-09-15 -> 2026-10-06 (only after Aug-Oct pass)

## Data Sources
  Polygon API: grouped daily, 1-min bars, ticker references
  Universe: _FLOAT dict (217 tickers, float >= 5M)
  Infrastructure: adapted from agent1/agent3 proven pipelines

## Independence
  Completely independent of bot-RAG (System 1), agent1, agent3, agent10.
  Shares only the Polygon API key and float dictionary.
