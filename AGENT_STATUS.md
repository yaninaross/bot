# Agent Status — Last Updated: 2026-09-18

## Currently Active Work
- Agent 1: E2 (N×Hold grid) — NEXT TO RUN
- Agent 3: C1-E01 (WedThu formal backtest) — NEXT TO RUN

## Last Session Summary
- Confirmed CS+SD+GUARDS baseline working
- Identified gap-fill skip logic as new experiment (E-GAP-FILL)
- HELP and ONON: stocks where gap was consumed before entry
  These should be caught by gap_consumed > 0.75 filter

## Completed This Week
- Agent 1 E1: CS+SD+GUARDS guards verified working
- Agent 3 baseline: confirmed PAVS/DBGI guards working

## Blocked / Waiting
- Nothing currently blocked

## Do Not Forget
- Agent 1: float cache miss must EXCLUDE (not include)
- Agent 3: C1 fallback entries excluded (near-zero edge)
- Agent 3: Never trade C1 on Tuesday (avg -1.02%)
- Both agents: OOS cutoff = 2026-01-01

## Next Session Should Start With
Read CLAUDE.md, then read this file.
Then run: [experiment name here]
