# MOIRAI-Large v2 — Foundation Model Consensus Filter

**Direction:** LONG
**Status:** PENDING (consensus filter on S3)
**Dates:** Started 2026-Q2
**Parent:** Independent investigation

## Hypothesis
MOIRAI-Large (Salesforce foundation model) can provide a consensus signal to filter S3 predictions, improving precision without losing too many trades.

## Key Findings (Post-Mortem)

1. **Family correlation rho = +0.93/+0.97** — MOIRAI and S3 see nearly the same signal
2. **87% of LONG edge lost to binarization** — MOIRAI's continuous output is more informative than its binary prediction
3. **Max achievable uplift: +0.41%/trade** — standalone marginal
4. **Standalone validation: r=+0.55**, VAL_A +0.70%
5. **6 ranked failure causes** identified in 5-agent investigation
6. **1.2 effective information units** — not enough independent signal for standalone use

## Decision
**PENDING** — Consensus filter path is the only viable use. Standalone deployment rejected.

## Scripts
- Setup: `moirai/moirai_setup_and_inference.py`, `moirai/moirai_setup_and_inference_v2.py`
- Optimizer: `moirai/moirai_wf_optimizer_v1.py`, `moirai/moirai_wf_optimizer_v2.py`
