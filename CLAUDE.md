# bot-RAG Trading System — Navigation Guide

## ⚠️ FIVE SEPARATE SYSTEMS — READ BEFORE TOUCHING ANYTHING

### System 1: bot-RAG (Spike Transformer ML)
**Status: PRODUCTION — DO NOT MODIFY WITHOUT EXPLICIT INSTRUCTION**
Directories: pipeline/, scoring/, ra_moe/, training/, optimization/,
             backtesting/, analysis/, data_ops/, experiments/,
             moirai/, scanner/, sentiment/, archive/
Purpose: Intraday spike prediction using DeStationary Attention +
         FiLM conditioning transformer. Runs daily at 6:15 AM PST.
If asked to work on this system: read MODEL_EVALUATION_GUIDE.md
and EXPERIMENT_LOG.md first.

### System 2: Agent 1 + Agent 3 (Pure Technical Gap Strategies)
**Status: ACTIVE DEVELOPMENT**
Directories: agent1/, agent3/
Purpose: Gap-fade SHORT (Agent 1) and LONG pension strategies
         (Agent 3). Pure price mechanics, no ML, no transformers.
If asked to work on these: read agent1/CLAUDE.md or agent3/CLAUDE.md

### System 3: Agent 10 (Cyclical Mean Reversion — RXRX)
**Status: ACTIVE DEVELOPMENT**
Directory: agent10/
Purpose: Investigate RXRX ~25-35d oscillation cycle. Design/validate
         long/short mean-reversion strategies with 1-5 day hold.
         Test generalization to other stocks with same fingerprint.
If asked to work on this: read agent10/CLAUDE.md

### System 4: Project X (Semi-Autonomous Strategy Investigation)
**Status: ACTIVE DEVELOPMENT**
Directory: investigation/
Purpose: Automated hypothesis-driven search for new trading strategies
         that beat the E2-A SHORT baseline in Aug-Oct 2026.
         LONG, SHORT, any catalyst, any hold period.
         Max 50 hypotheses, 3 winners, human-tradeable pace (<=15/week).
If asked to work on this: read investigation/CLAUDE.md

### System 5: Agent 1_2 (Project X Live SHORT Scanner)
**Status: ACTIVE — LIVE SCORING**
Directory: agent1_2/
Purpose: Live SHORT scanner from Project X winner H006 + DEEP4.
         Gap-up + sustained_drop + local_high within 3% proximity.
         Same-day exit at 17:00 ET. Runs morning at 7:30 PT.
If asked to work on this: read agent1_2/CLAUDE.md

## Global Rules
- Never confuse Systems 1-5 — they are completely separate
- Never import or reference System 1 code from agent1/, agent1_2/, agent3/, agent10/, or investigation/
- API keys: POLYGON_API_KEY in .env (never hardcode)
- GCS bucket and BQ project: in .env (never hardcode)
- OOS cutoff for all backtests: 2026-01-01 (never change)
- Float cache miss = EXCLUDE (applies to agent1 and agent3 only)

## Infrastructure (System 1 — do not change)
- GCS: gs://<GCS_BUCKET>/Data/Inference_v2/, Algos/RA_MoE/
- BQ: <GCP_PROJECT>.scoring_logs.wf_optimization_results
- Scripts run as Colab cells (no if __name__, config vars at top)
- Seeds: 15 per experiment (seed = exp*100 + 01-15)

## Active Work Priority
1. agent1_2/ — Project X live SHORT scorer (daily)
2. investigation/ — Project X strategy search
3. agent10/ — RXRX cyclical mean reversion investigation
4. agent1/ — SHORT gap-fade experiments
5. agent3/ — LONG pension strategy experiments
6. Everything else is reference only
