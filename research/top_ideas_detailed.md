# Top Candidate Strategies — Detailed Descriptions

From the Comparative Analysis of Transformer-Based Intraday Trading Strategies (2026-08-17).

---

## Idea 1: Hawkes-Calibrated Opening Momentum (HCOM) — RECOMMENDED

**Category:** Transformer-based Algorithmic Trading > Market Microstructure and Execution > Transformer Encoder with VSN, Hawkes Intensity, Isotonic Calibration, and Friction-Adjusted Half-Kelly Sizing

### Idea Proposal

The Hawkes-Calibrated Opening Momentum (HCOM) strategy is a sophisticated intraday system that leverages a Transformer Encoder with a Gaussian Likelihood head to trade the SPDR S&P 500 ETF (SPY) during the high-volatility window from 9:30 AM to 10:30 AM EST. By integrating market microstructure and E-mini futures trade intensity (Hawkes process) captured up to 9:28 AM, the model generates calibrated return and volatility forecasts to trigger Market-on-Open (MOO) entries. The strategy optimizes risk-adjusted returns through a continuous Half-Kelly position sizing framework and dynamic, volatility-adaptive stop-losses, specifically designed to capture alpha from institutional price discovery while minimizing execution friction through batch auctions.

### Description

#### 1. Financial Instrument and Time Horizon

- **Instrument:** SPDR S&P 500 ETF (SPY). SPY is selected for its high liquidity and its role as the primary vehicle for institutional macro-hedging at the market open, which is designed to ensure that signals can be executed at scale with minimal market impact.
- **Horizon:** A strict 60-minute holding period. Positions are initiated at the 09:30 AM EST Opening Cross and liquidated at 10:30 AM EST. This window targets the peak of the intraday U-shaped volatility profile, exiting as the information advantage of the opening signal typically dissipates into mid-day mean reversion.

#### 2. Signal Generation and Architecture

The HCOM strategy utilizes a Transformer Encoder architecture equipped with a Variable Selection Network (VSN) and a Gaussian Likelihood head.

**Input Pipeline (Cutoff at 09:28:00 AM EST):** To respect the submission cutoff constraints for Market-on-Open (MOO) orders on NYSE Arca, the model ingests data up to 09:28:00 AM. Inputs include:

- **Market Microstructure:** 1-minute OHLCV sequences and overnight gaps.
- **Informed Flow (Hawkes Intensity):** The trade arrival intensity of E-mini S&P 500 futures (ES) between 08:35 AM and 09:28 AM. Commencing data ingestion at 08:35 AM strictly excludes the 08:30 AM macroeconomic data releases (e.g., CPI, Non-Farm Payrolls). This is intended to prevent the Hawkes process from misclassifying deterministic public news spikes as endogenous informed flow, aiming to ensure intensity accurately reflects structural price discovery rather than scheduled noise.

**Predictive Output:** The model outputs the predicted mean return (mu) and the predicted standard deviation (sigma) for the upcoming 60-minute window.

**Calibration:** The raw mu is calibrated using Monotonic Isotonic Regression against a validation set. This corrects for the tendency of deep learning models to be overconfident in financial time series while preserving the signal's magnitude for position sizing.

#### 3. Execution Logic (Entry and Exit)

The HCOM system is designed to minimize the "triple-cost hurdle" of commissions, spreads, and slippage by utilizing batch auctions.

**Entry Rule:** Submit a Market-on-Open (MOO) order to NYSE Arca at 09:28:30 AM EST.
- Long Entry: If calibrated mu > 0.03% (3 basis points).
- Short Entry: If calibrated mu < -0.03%.
- Neutral: If |mu| <= 0.03%, no trade is executed. This "Null Zone" represents a 3x safety margin over realized friction.

**Exit Rules:**
- **Primary Exit:** A market order is executed at exactly 10:30:00 AM EST.
- **Risk Boundaries:** Stop-Loss (SL) at entry price -/+ 1.5 x sigma_60m and Take-Profit (TP) at entry price +/- 3.0 x sigma_60m. By using the model's predicted 60-minute standard deviation rather than a fixed ATR, the risk bounds dynamically adapt to the expected volatility of that specific session.

#### 4. Position Sizing: Continuous Mean-Variance Optimal

|f*| = lambda x (|mu| - c) / sigma^2

- |mu| - c: The absolute calibrated expected return minus the friction cost. All inputs must be strictly evaluated as decimal representations.
- sigma^2: The predicted variance of decimal returns over the 60-minute horizon.
- lambda: A conservative Half-Kelly multiplier (0.5) to account for parameter uncertainty.

**Constraints:**
- Gross Leverage Cap: 4.0x account equity.
- Portfolio Risk Cap: Distance to 1.5x sigma_60m SL must not exceed 1.0% of total portfolio equity.

#### 5. Transaction Costs and Frictions

Total institutional friction budget: approximately 1.0 basis point (bps) per round trip.
- Entry Slippage: 0 bps (Opening Cross auction)
- Commissions: 0.10 bps ($0.005 per share)
- Regulatory Fees: 0.21 bps (SEC Section 31, FINRA TAF)
- Exit Slippage: 0.7 bps (crossing typical SPY spread at 10:30 AM)

#### 6. Scientific Rationale

The HCOM strategy addresses the "Causality Paradox" inherent in market-open ML models. By stopping data ingestion at 09:28 AM, it allows for computationally feasible inference and regulatory compliance for MOO submission. The integration of the Hawkes process is based on microstructural evidence that futures trade-clustering is a leading indicator of cash-market volatility.

---

## Idea 2: Transformer-Encoded QQQ Opening Auction Strategy

**Category:** Transformer-based Trading Strategies > Intraday Algorithmic Trading > QQQ Transformer, Isotonic Calibration, and AR-Based Risk Management

### Idea Proposal

This intraday strategy targets the Invesco QQQ Trust (QQQ) during the first hour of market trade (09:30 AM to 10:30 AM EST) to capitalize on high opening volatility and institutional liquidity. It utilizes a Transformer Encoder calibrated via Isotonic Regression to generate long/short signals based on pre-market futures, 1-minute OHLCV bars, and Nasdaq Early Order Imbalance Indicator data ingested up to 09:26 AM.

### Description

#### 1. Financial Instrument and Time Horizon

- **Instrument:** Invesco QQQ Trust (QQQ).
- **Time Horizon:** 60-minute intraday holding period (09:30 AM to 10:30 AM EST).
- **Rationale:** QQQ's high absolute share price mathematically dilutes fixed, per-share institutional commissions into fractional basis points, establishing an asymmetrical cost advantage. Its technology-heavy composition exhibits a pronounced U-shaped volatility profile at the market open.

#### 2. Signal Generation and Calibration

The core predictive engine is a Transformer Encoder trained as a binary classifier to forecast whether the 60-minute directional return will exceed a defined transaction-cost hurdle.

- **Data Ingestion:** System ingests 1-minute OHLCV data up to the closed 09:26 AM bar. Enriched with pre-market futures volatility and tick-level Nasdaq EOII data (09:25 AM to 09:27 AM).
- **Probability Calibration:** Isotonic Regression calibration layer trained on out-of-sample data to correct classification overconfidence.
- **Activation Hurdle:** Trade only authorized if calibrated probability exceeds statistical confidence threshold.

#### 3. Execution Mechanics

- **MOO Entry:** Transmitted at 09:27:45 AM EST. Under Nasdaq rules, new MOO orders permitted until 09:28:00 AM.
- **TWAP Exit:** 30-second TWAP initiated at 10:29:30 AM to reduce tail-risk slippage.
- **Friction:** ~2.0 bps round-trip.

#### 4. Risk Management

- **Gap-Adjusted Volatility:** Average Range (AR) of 09:30-10:30 candle (High minus Low). Standard ATR rejected because it incorporates overnight gaps.
- **Stop-Loss:** 1.0 x 60-minute AR from entry.
- **Take-Profit:** 1.5 x 60-minute AR.
- **Sizing:** 0.5% of total account equity per trade. Hard leverage cap of 2.0x.

---

## Idea 3: Calibrated Nasdaq Opening Alpha (C-NOA)

**Category:** Transformer-based Trading Strategies > Intraday Algorithmic Trading > QQQ Transformer-MOO Strategy with Platt Calibration and ATR-60m Risk Management

### Idea Proposal

The C-NOA strategy utilizes a Transformer model to generate directional signals for QQQ's first hour of trading. By processing multi-modal data—including Net Order Imbalance and Hawkes volatility—and applying Platt Scaling for probability calibration, the strategy identifies high-conviction trades executed via MOO orders. Risk is managed through a strict 60-minute temporal exit and price-based stops scaled to intraday ATR, while capital allocation is optimized using a fractional Kelly criterion capped at 2.0x leverage.

### Description

#### Signal and Calibration
- **Platt Scaling** applied to transform overconfident softmax outputs into calibrated empirical probabilities.
- **Confidence Threshold:** P(win) > 0.56 for Longs, P(win) < 0.44 for Shorts.

#### Execution
- **MOO order** transmitted at 09:27:59 AM EST (1-second buffer before 09:28 lockout).
- **9:28:00 Lockout:** Complies with Nasdaq Rule 4702(b)(8)(A).

#### Risk Management
- **ATR-60m based stops:** SL at 1.2x ATR-60m, TP at 1.8x ATR-60m.
- Uses 20-period 60-minute ATR (typically $1.50-$2.50) instead of daily ATR (~$8.00).
- **Temporal Exit:** Market order at 10:30 AM if neither SL nor TP triggered.

#### Position Sizing
- Fractional Kelly: f = 0.4 x (P(win) - (1 - P(win)) / (W/L))
- Volatility equalization across regimes.
- Leverage cap: 2.0x.

#### Frictions
- Total: 6.5 bps round-trip.
- Entry: 0 bps (MOO auction).
- Exit: 4 bps (continuous market at 10:30 AM).
- Buffer: 2.39 bps conservative margin.

---

## Idea 4: SPY Transformer with VSN, Hawkes, and Continuous Half-Kelly

**Category:** Intraday Algorithmic Trading > SPY Transformer Encoder with Gaussian Likelihood and RTH-ATR Risk Management

### Idea Proposal

This strategy employs a Transformer Encoder architecture with a Gaussian Likelihood head to generate 60-minute intraday long/short signals for SPY at the market open. The model processes pre-market microstructure, Hawkes intensity of E-mini futures trades, and OHLCV data through a Variable Selection Network to output calibrated return and volatility forecasts. Execution is optimized via MOO orders.

### Description

#### Signal Generation
- **Inputs (at 9:25 AM EST):**
  - Market Microstructure: Overnight Gap, Pre-market Volume Ratio, 1-min OHLCV (8:00-9:25 AM)
  - Hawkes intensity of ES futures (8:30-9:25 AM) — captures clustering of informed orders
  - Variable Selection Network (VSN) for feature weighting
- **Output:** Predicted mu and sigma for 9:30-10:30 AM period
- **Calibration:** Platt Scaling or Isotonic Regression

#### Execution
- **Entry:** MOO order at 9:28 AM EST
  - Long if mu > 0.05% (5 bps hurdle)
  - Short if mu < -0.05%
  - No trade if |mu| <= 0.05%
- **Exit:** Market order at 10:30 AM EST
- **Risk Boundaries:** SL at 2.5x RTH-ATR, TP at 4.0x RTH-ATR (using RTH first-hour ATR, not pre-market)

#### Position Sizing
- f* = lambda x mu / sigma^2 (Half-Kelly, lambda=0.5)
- Distance to SL <= 1.0% of portfolio equity
- Order size <= 2% of projected Opening Cross volume

#### Frictions
- Total: ~1.5-2.0 bps round-trip
- Entry: 0 bps (Opening Cross)
- Commissions: 0.15 bps ($0.0035/share at ~$500)
- Exit: 1.0-1.5 bps

#### Key Design Choices
- **Continuous Kelly vs Binary Kelly:** Treats 60-min return as continuous random variable, not win/loss bet
- **RTH-ATR:** Pre-market vol << RTH vol; using opening-hour ATR for realistic stop placement
- **Transformer Encoder only:** No decoder needed for single-horizon task (more parsimonious)
- **Hawkes Process:** Distinguishes random pre-market drift from sustained directional moves
- **MOO Efficiency:** Deploys capital without "walking the book"
