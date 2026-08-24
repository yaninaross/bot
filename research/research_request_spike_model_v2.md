# Research Request: Next-Generation Architectures and Strategies for Intraday Spike Prediction on Individual Equities

## 1. Research Goal

Propose novel model architectures, alpha sources, and monetization strategies that can **improve upon or replace** an existing Transformer-based intraday spike prediction system deployed on individual US equities. The system currently trades ~100 tickers across crypto-adjacent and defense sectors, predicting binary spike probabilities at the pre-market/early session, and entering directional trades (LONG or SHORT) with fixed TP/SL brackets during the first hour of regular trading.

The research must go beyond incremental hyperparameter tuning. We seek **fundamentally different approaches** — whether architectural innovations, new feature spaces, alternative prediction targets, ensemble paradigms, or entirely new monetization frameworks for the existing signal. All proposals must be grounded in recent literature (2024–2026) and evaluated for practical viability in a retail-scale quantitative trading operation (single-GPU training, Google Colab execution, 1-minute bar data).

## 2. Current System Description (Baseline to Beat)

### 2.1 Architecture: Spike Transformer

The current production model is a custom Transformer Encoder with the following components:

- **Series Stationarization Layer**: Instance normalization (subtract mean, divide by std per sample) applied to input sequences before attention. Statistics (mean, std) are preserved and re-injected via De-Stationary Attention.
- **De-Stationary Attention**: Multi-head self-attention modified to re-incorporate the removed non-stationary statistics. Learnable tau (bias) and sigma (scaling) projections from the stationarization statistics modulate attention scores and output values respectively. This addresses the over-stationarization problem identified in Non-stationary Transformers (Liu et al., 2022).
- **FiLM Conditioning (Feature-wise Linear Modulation)**: A 2-layer MLP processes 22 regime features (VIX level, SPY return, TLT return — all computed as 30-minute medians anchored to the per-ticker entry bar) and produces per-channel gamma/beta modulations applied to the pooled Transformer output. This allows the model to shift its behavior based on macro regime without contaminating the temporal attention.
- **Attention Pooling**: Learnable query vector attends over the full sequence output to produce a fixed-length representation (replacing mean pooling or CLS tokens).
- **Asymmetric Focal Loss (ASL)**: The training objective uses asymmetric gamma values (gamma_neg=6.0, gamma_pos=0.1) with a hard probability clipping margin (clip=0.08) on negative samples. This addresses extreme class imbalance (~3-5% positive rate for spikes) by heavily down-weighting easy negatives while preserving gradient signal from positives.
- **Multi-Horizon Head**: Single sigmoid head outputs probabilities for 3 horizons simultaneously (15m, 30m, 60m spike), though deployment typically uses a single horizon selected during walk-forward optimization.

**Hyperparameters**: d_model=64, n_heads=4, d_ff=128, n_blocks=2, dropout=0.1, FiLM_hidden=64, time_steps=128, batch_size=512.

### 2.2 Data Pipeline

- **Input Features**: 1-minute OHLCV bars (Open, High, Low, Close, Volume + derived features). 128-timestep lookback window. N_asset_features varies by experiment (typically 40-53 columns after feature engineering).
- **Regime Features**: 22 features derived from VIX, SPY, and TLT — 30-minute rolling medians of returns and levels, anchored to each ticker's specific entry minute.
- **Target**: Binary spike label — whether price moves ≥ X% within 15/30/60 minutes of entry. Threshold varies by experiment (typically 3.5-8% for LONG, 6-16% TP for SHORT).
- **Training Data**: Walk-forward folds with ~2 years of training data per fold. 15 seeds per experiment. Sector-aware sampling.
- **Ticker Universe**: ~64-99 tickers across crypto-adjacent (MARA, RIOT, CLSK, COIN, MSTR, etc.), defense (LMT, RTX, NOC, GD, KTOS, RKLB, etc.), energy, biotech, and high-beta tech.

### 2.3 Trading Execution

- **Score Time**: Minute 375 (6:15 AM Pacific / 9:15 AM Eastern) — pre-market scoring.
- **Entry Time**: Minute 385 (6:25 AM Pacific / 9:25 AM Eastern) — 5 minutes before market open.
- **Exit Modes**: 5 modes optimized per-ticker via walk-forward grid search:
  - Static: Fixed TP/SL at entry
  - Grace: SL delayed by N minutes
  - Limit Entry: Wait for a rally/dip of X% before filling, then apply TP/SL
  - Vol-Scaled: TP/SL scaled by 20-day trailing realized volatility ratio
  - Ensemble: Blended 15m/30m/60m scores with weighted or max aggregation
- **Slippage Model**: 0.15% each way (0.30% round-trip)
- **Deploy Threshold**: Ticker/config must achieve ≥ 1.8% average PnL per trade on out-of-sample test period
- **Trade Frequency Target**: ~2 trades per month per ticker minimum

### 2.4 Performance Characteristics

- **LONG**: Best deployed config achieves ~2.2% avg PnL per trade using NEWEST_FIRST model selection (6 models across L5-1 and ASL6 families). However, LONG performance has degraded post-July 2026 due to regime shift (volatility +125%, sector rotation away from crypto/tech).
- **SHORT**: Hybrid deployment across 3 walk-forward folds (F5+F6+F7) on 30 tickers, achieving ~2.19% avg PnL per trade. SHORT has been more stable through the regime shift.
- **Win Rate**: Typically 55-77% depending on ticker and TP/SL configuration.
- **Key Finding**: The S3 static strategy (simple threshold + fixed TP/SL) consistently beats all ML-based ticker selection, dynamic sizing, and ensemble methods in live walk-forward validation. Simplicity is the regularizer.

### 2.5 What Has Been Tried and Failed (31 Attempts)

The following improvement categories have all been tested and produced **zero wins** against the L5-1 v1 baseline:

**Architecture/Training (P3-P4):**
- Multi-head regression (predict magnitude, not just probability) — worse calibration
- Morning swing target (different label definition) — no edge
- Deeper models (more blocks, larger d_model) — overfit
- v14 optimizer (new grid search methodology) — failed to beat v1
- Different loss functions — ASL already optimal for this imbalance ratio

**Signal Processing (P1-P2):**
- A/B/C experiments (3 architectural variants) — all failed to beat L5-1 baseline
- Ensemble across seeds — marginal at best, often worse
- ML-based ticker selection (Lasso, Random Forest on WF features) — S3 static beats ML oracle
- Confidence calibration — aggregate calibrated but per-ticker fails (r=+0.10)
- Dynamic TP/SL (7 modes including trailing stops) — no improvement over static

**External Models:**
- MOIRAI (foundation model) — family rho +0.93/+0.97 with existing model (too correlated), only +0.41%/trade max. Binarization destroys 87% of LONG edge.
- Exp-200 CNN-Transformer hybrid — underperformed Spike Transformer
- Exp-500 swing model (CNN+MLP, 3.5% target) — VAL→TEST -2.94%, not viable

**Key Lesson**: Every "textbook" ML improvement (deeper models, better features, fancier ensembles, calibration layers) has failed. The current architecture appears to extract the maximum learnable signal from 1-min OHLCV + regime features. Improvement likely requires **fundamentally different inputs, targets, or problem formulations** — not incremental model tweaks.

## 3. Research Directions to Explore

We seek comprehensive analysis across **five research pillars**. For each, propose 2-4 concrete, implementable ideas with literature backing, expected edge, and practical requirements.

### 3.1 Novel Architectures for Sparse Event Prediction

The spike prediction task has extreme class imbalance (3-5% positive rate) and requires capturing both local microstructure patterns and longer-range temporal dependencies in 1-minute bar data. Explore:

- **State Space Models (Mamba, S4, S5)**: Do structured state space models offer advantages over attention for long-range dependencies in financial time series? How do they handle the stationarity problem?
- **Graph Neural Networks for Cross-Asset Modeling**: Can modeling the ticker universe as a graph (with edges based on sector, correlation, or lead-lag relationships) improve spike prediction by propagating information across related assets?
- **Diffusion Models for Financial Time Series**: Can denoising diffusion probabilistic models generate better uncertainty estimates or serve as data augmentation for rare spike events?
- **Mixture of Experts (MoE) with Regime Routing**: Can a sparse MoE architecture learn to route different market regimes to specialized expert networks, improving per-regime accuracy?
- **Temporal Fusion Transformers (TFT)**: The current model uses a custom Transformer + FiLM. Would a proper TFT architecture (with gating, variable selection, and interpretable attention) offer improvements?
- **Kolmogorov-Arnold Networks (KAN)**: Recent alternatives to MLPs that use learnable activation functions on edges. Any evidence for financial time series?
- **xLSTM / Modern RNN variants**: Do the 2024-era extended LSTM architectures compete with Transformers on this task class?

### 3.2 New Feature Spaces and Alpha Sources

The current model uses only 1-minute OHLCV bars + 3 regime indicators. What additional data could provide orthogonal signal?

- **Options Flow and Implied Volatility Surface**: Can pre-market options activity (unusual volume, IV skew changes, put/call ratio) predict intraday spikes? What data sources are accessible at retail scale?
- **Order Book Microstructure**: Even without L2 data, can derived features (bid-ask spread dynamics, trade-to-quote ratio, volume-weighted price pressure) from 1-minute bars approximate microstructure signals?
- **Cross-Asset Lead-Lag**: Bitcoin/Ethereum leading crypto stocks, sector ETFs leading individual stocks, futures leading cash — can these temporal relationships be systematically exploited?
- **Alternative Data at Retail Scale**: Social sentiment (Reddit/Twitter velocity), short interest changes, dark pool activity (FINRA data), SEC filing signals — which have demonstrated alpha for intraday spike prediction specifically?
- **Learned Features via Contrastive Learning**: Can self-supervised pre-training (e.g., TS2Vec, TNC, or SimCLR adapted for time series) learn better representations from unlabeled 1-minute data than hand-crafted features?
- **Multi-Resolution Input**: Simultaneously feeding 1-min, 5-min, and 15-min bars (or wavelet decompositions) to capture patterns at multiple time scales.

### 3.3 Alternative Prediction Targets and Problem Formulations

The current system predicts binary spike probability. Are there better ways to frame the prediction problem?

- **Quantile Regression / Distributional Forecasting**: Instead of P(spike), predict the full return distribution (e.g., 10th, 25th, 50th, 75th, 90th percentiles). This enables dynamic TP/SL setting and Kelly-based sizing without a separate walk-forward optimizer.
- **Survival Analysis / Time-to-Event**: Model the time until a spike occurs (if at all) within the trading window. This could improve entry timing beyond the fixed-minute approach.
- **Ordinal / Multi-Class Targets**: Instead of binary spike/no-spike, predict severity tiers (no move, small move, medium spike, large spike). This preserves ordinal structure lost in binarization.
- **Reinforcement Learning for Execution**: Keep the Transformer as signal generator but replace the grid-search TP/SL optimizer with an RL agent that learns optimal entry timing, position sizing, and exit rules jointly.
- **Conformal Prediction for Uncertainty**: Can conformal prediction provide calibrated prediction intervals that are more useful than point probabilities for trading decisions?

### 3.4 Portfolio-Level and Cross-Sectional Strategies

The current system treats each ticker independently. Are there portfolio-level approaches that could improve overall returns?

- **Cross-Sectional Momentum / Mean-Reversion**: Instead of absolute spike probability per ticker, rank tickers by relative signal strength and trade the top/bottom N. Does relative ranking outperform absolute thresholds?
- **Risk Parity / Volatility Targeting**: Can portfolio-level volatility targeting improve risk-adjusted returns vs. per-ticker fixed TP/SL?
- **Correlation-Aware Position Sizing**: When multiple tickers signal simultaneously, should position sizes be reduced to account for correlation? How much concentration risk exists in the current approach?
- **Market-Neutral Construction**: Can a long-short portfolio of spike signals (long the highest-probability LONG spikes, short the highest-probability SHORT spikes) produce market-neutral alpha?

### 3.5 New Monetization and Trade Structures

Same signal, different trade implementation:

- **Options Overlays**: Can spike probability signals be monetized more efficiently through options (buying OTM calls/puts, selling straddles on low-probability days)?
- **Intraday Scaling (Pyramiding)**: If the model produces scores throughout the day, can positions be scaled into rather than entered all-at-once?
- **Multi-Day Holding**: The current system is strictly intraday. Is there evidence that spike signals have predictive power beyond the same day (overnight holds, 2-3 day swings)?
- **Signal Selling / Factor Construction**: Could the spike probability be packaged as a tradeable factor or signal service rather than directly traded?

## 4. Evaluation Criteria for Proposed Ideas

Each proposed idea should be evaluated on:

1. **Expected Edge Over Baseline**: Quantitative estimate of potential improvement (% PnL, Sharpe ratio, etc.) with literature-backed evidence.
2. **Implementation Complexity**: Can it be built in TensorFlow/Keras, trained on a single GPU (Colab T4/A100), and scored in <30 seconds per ticker?
3. **Data Requirements**: What additional data is needed? Is it available at retail scale and reasonable cost?
4. **Robustness to Regime Change**: How would this approach handle the type of volatility regime shift observed in mid-2026 (VIX +125%, sector rotation)?
5. **Complementarity**: Does it provide orthogonal signal to the existing Spike Transformer, or would it be redundant?
6. **Time to Production**: Estimated weeks/months from concept to deployable system.

## 5. Constraints and Practical Requirements

- **Framework**: TensorFlow/Keras (existing codebase). PyTorch acceptable for research but production must be TF.
- **Compute**: Google Colab (T4 or A100 GPU). Training must complete within 4-8 hours per fold.
- **Data**: Primary source is 1-minute OHLCV bars from a proprietary GCS data lake. Additional data must be acquirable at retail cost (<$500/month).
- **Execution**: Trades are entered at specific minutes during pre-market/early session. No sub-minute execution capability. No direct market access — broker API only.
- **Universe**: ~100 individual US equities, primarily high-beta small/mid-cap stocks. NOT index ETFs or futures.
- **Regime**: The model must work across different volatility regimes. The current system shows degradation during low-volatility or trending regimes.
- **Simplicity Premium**: Based on empirical evidence (31 failed improvements), simpler approaches that introduce genuinely new information tend to outperform complex approaches that reprocess existing information more cleverly. Proposals should respect this finding.

## 6. Deliverables

For each research direction, provide:

1. A ranked list of 2-4 concrete strategy/architecture proposals
2. Literature review with key papers and their relevance
3. Comparative analysis of proposals (pros, cons, expected edge, implementation difficulty)
4. A synthesis identifying the single highest-priority idea to prototype first
5. Unexpected connections or cross-pollinations between ideas that could yield novel approaches
6. A recommended 3-phase implementation roadmap (quick wins → medium-term → moonshots)
