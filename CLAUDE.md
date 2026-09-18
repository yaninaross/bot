# bot-RAG Trading System — Navigation Guide

## ⚠️ TWO SEPARATE SYSTEMS — READ BEFORE TOUCHING ANYTHING

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

## Global Rules
- Never confuse System 1 and System 2 — they are completely separate
- Never import or reference System 1 code from agent1/ or agent3/
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
1. agent1/ — SHORT gap-fade experiments
2. agent3/ — LONG pension strategy experiments
3. Everything else is reference only
