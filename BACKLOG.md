# Backlog — Ideas to Test

Prioritized queue of experiments and improvements. Move items to EXPERIMENT_LOG.md once tested.

**Priority:** P0 = ready to run (scripts exist), P1 = needs one prep step, P2 = needs design

---

## P0 — Ready to Run

These have scripts written and data built. Just need a Colab GPU session.

### 1. X-ASSET optimizer (SHORT)
- **Hypothesis:** BTC/ETH lead-lag features improve SHORT predictions for crypto-correlated tickers
- **Evidence:** Backtest shows +0.42% avg PnL lift, deployable tickers 20 to 29
- **Script:** `ra_moe/ra_moe_short_L5_2_xasset_optimizer.py`
- **Effort:** ~2-3 hrs GPU
- **Why run:** First positive ML experiment in 31 attempts. Optimizer determines deploy configs.

### 2. Multi-Res v2 optimizer (SHORT)
- **Hypothesis:** Dual 1-min + 15-min Transformer backbone captures longer-horizon context
- **Evidence:** v1 overfit (50K samples), v2 uses 150K samples to fix. Training complete.
- **Script:** `ra_moe/ra_moe_short_L5_2_multires_v2_optimizer.py`
- **Effort:** ~2-3 hrs GPU
- **Why run:** +1.29% PnL lift in early tests. Independent signal from X-ASSET.

### 3. LONG trade prioritization backtest
- **Hypothesis:** M_AGREEMENT improves LONG daily PnL (like it did for SHORT: +3.20%)
- **Evidence:** SHORT pattern suggests multi-model agreement filters weak signals
- **Script:** `backtesting/backtest_st_long_prioritization.py`
- **Effort:** ~30 min GPU
- **Why run:** LONG currently runs ALL_TRADES at +1.88%. Prioritization could push higher.

---

## P1 — One Prep Step Needed

### 4. X-ASSET for LONG
- **Hypothesis:** BTC/ETH lead-lag features help LONG too (worked for SHORT)
- **Prep:** Adapt `ra_moe_short_L5_2_xasset_data_prep.py` for LONG direction
- **Effort:** ~1 hr script adaptation + ~4 hrs training + ~2 hrs optimizer
- **Why try:** If it works for SHORT, same external signal should help LONG crypto tickers

### 5. Multi-Res for LONG
- **Hypothesis:** 15-min backbone helps LONG predictions too
- **Prep:** Adapt `ra_moe_short_L5_2_multires_v2_data_prep.py` for LONG
- **Effort:** ~1 hr script adaptation + ~4 hrs training + ~2 hrs optimizer
- **Why try:** Captures overnight drift patterns that 1-min alone misses

### 6. MOIRAI consensus filter on S3
- **Hypothesis:** MOIRAI-Large as ensemble vote improves S3 precision
- **Evidence:** r=+0.55 standalone, family rho +0.93 (correlated but not redundant)
- **Prep:** Build consensus scoring logic into daily_score_s3.py
- **Effort:** ~2 hrs scripting + ~1 hr backtest
- **Why try:** Low-risk filter addition. Standalone rejected, but consensus path untested.

### 7. Fold 8 training (both directions)
- **Hypothesis:** Newer training data captures current regime better
- **Prep:** Define F8 date boundaries (train through Jul 2026, val Aug, test Sep)
- **Effort:** ~8 hrs training per direction
- **Why try:** F5/F6/F7 trained on data through Jun 2026. Market regime shifted in July.

---

## P2 — Needs Design

### 8. Cross-asset features beyond crypto
- **Hypothesis:** Other lead-lag relationships exist (e.g., sector ETFs lead sector stocks)
- **Design needed:** Which assets lead which tickers? Correlation analysis first.
- **Effort:** ~2 hrs analysis + ~4 hrs data prep + training
- **Why consider:** X-ASSET proved external signal works. Sector ETFs (XLF, XLE, XBI) may lead their stocks.

### 9. Adaptive position sizing
- **Hypothesis:** Vary position size by model confidence and agreement spread
- **Design needed:** Sizing function, backtest framework that supports variable sizing
- **Effort:** ~4 hrs backtesting framework + ~2 hrs analysis
- **Why consider:** Currently equal-weight. High-agreement low-spread signals likely have higher edge.

### 10. Intraday regime switching
- **Hypothesis:** First 30 min of trading has different dynamics than 30-60 min window
- **Design needed:** Separate models or conditional scoring per time window
- **Effort:** ~8 hrs data analysis + training
- **Why consider:** 6:15 >> 6:45 suggests signal decays. Understanding why could unlock later windows.

### 11. Overnight gap features
- **Hypothesis:** Pre-market gap size and direction predict intraday spike probability
- **Design needed:** Feature engineering from overnight OHLCV
- **Effort:** ~2 hrs feature eng + ~4 hrs training
- **Why consider:** Gaps often drive opening hour volatility. Currently not captured.

---

## Parked (low priority or blocked)

### Exp 240-245 (28 regime features)
- **Status:** Training scripts exist, never validated
- **Blocked by:** LONG ceiling finding — L5-1 v1 is unbeatable for LONG architecture
- **Revisit when:** If X-ASSET proves that wider features help LONG too

### Exp 425/435 (10AM SHORT)
- **Status:** Optimizers created, not run
- **Blocked by:** 6:15 AM timing dominance. 10AM entry has less edge.
- **Revisit when:** If intraday regime study (#10) finds a second window

---

## Graveyard (do not revisit)

These categories have been exhaustively tested and failed. See EXPERIMENT_LOG.md for details.

- **Architecture changes** — 31 ideas, 0 wins. L5-1 v1 is the ceiling.
- **Training tweaks** — Curriculum learning, label smoothing, augmentation, different losses
- **Optimizer variants** — v14, temporal decay, vol-scaled TP/SL, multi-horizon ensemble
- **6:45 AM entry** — 61-95% PnL decay vs 6:15 for both directions
- **ML ticker selection** — S3 static beats Lasso, random forest, and ORACLE
- **MOIRAI standalone** — Max +0.41%/trade, 87% edge lost to binarization
