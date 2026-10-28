# Agent 1 — Baseline Backtest Results

## Run: 2026-09-18

**Script:** `agent1_baseline_bt.py`
**Period:** 2026-01-02 to 2026-09-18 (179 trading days)
**Runtime:** 697s (11.6 min)

## Strategy Config

| Parameter | Value |
|-----------|-------|
| Direction | SHORT |
| Filter | CS + sustained_drop + float >= 10M (SOFT) |
| Ranking | Pure gap size (biggest gap wins) |
| TOP_N | 1 |
| Entry | local_high (window=3, max_bar=12) or fallback bar 15 |
| Exit | 17:00 ET (1h after close, after-hours) |
| PNL_CAP | -50% |
| Gap threshold | >= 5% (PM-reranked at 09:20 ET) |
| Price min | $10 |
| PM_MIN_BARS | 5 |
| Capital | $100K/day, non-compounding |

## Results

| Metric | Value |
|--------|-------|
| Trades | 155 |
| Active days | 155 |
| Trades/day | 1.0 |
| **Avg PnL** | **+8.06%** |
| Median PnL | +4.22% |
| Trimmed Mean | +7.62% |
| **Win Rate** | **74%** |
| **Sharpe** | **0.426** |
| Profit Factor | 4.17 |
| 95% CI | [+5.09, +11.04] |
| Total $ (NC) | $+1,249,914 |
| Max DD | -61.0% |

### Entry Type Breakdown

| Type | n | Avg PnL | WR | Sharpe |
|------|---|---------|-----|--------|
| local_high | 85 | +9.37% | 74% | 0.433 |
| fallback | 70 | +6.48% | 73% | 0.432 |

### PnL by Horizon

| Horizon | n | Avg | Median | WR | Sharpe |
|---------|---|-----|--------|-----|--------|
| EOD (17:00) | 155 | +8.06% | +4.22% | 74% | 0.426 |
| T+1 | 155 | -5.22% | +6.44% | 70% | -0.034 |
| T+2 | 154 | +4.60% | +7.63% | 71% | 0.066 |

## Gap vs E2 Benchmark

E2 deployed config (from `AGENT1_EXPERIMENTS_E1_E10_RESULTS.md`):
- 160 trades, +14.71% avg PnL, Sharpe 0.594, 79% WR, PF 6.26, $+2.35M

**Current backtest vs E2 benchmark:**

| Metric | E2 Benchmark | Current BT | Delta |
|--------|-------------|------------|-------|
| Trades | 160 | 155 | -5 |
| Avg PnL | +14.71% | +8.06% | -6.65% |
| Win Rate | 79% | 74% | -5pp |
| Sharpe | 0.594 | 0.426 | -0.168 |
| Profit Factor | 6.26 | 4.17 | -2.09 |
| Total $ | $+2.35M | $+1.25M | -$1.10M |

## What's Different from E2

The E2 benchmark was run from `agent1_e2_nhold_grid.py` / `agent1_universe_float.py` over a different date range and may have used different data:

### Potential sources of gap:

1. **Date range mismatch** — E2 may have used a different start date or included different market conditions. The current run includes Jul-Sep 2026 which has known regime deterioration.

2. **PM reranking** — E2 may not have used premarket bar reranking (PM_MIN_BARS=5 filter). This script uses 09:20 ET PM price to recompute gaps, which changes the ranking vs using daily open. The PM reranking also filters out thinly-traded tickers (585 skipped).

3. **API float filter** — E2 used SOFT float (hardcoded dict only, unknown=pass). Current backtest adds API `share_class_shares_outstanding` fallback for tickers not in dict (198 filtered). This is more restrictive.

4. **After-hours bar availability** — Some tickers may not have AH bars, causing fallback to daily close for PnL calculation.

5. **T+1 dead zone confirmed** — T+1 avg -5.22% confirms E2 finding that overnight hold destroys the edge.

## Improvement Ideas

- [ ] Run without PM reranking (use daily open gap) to isolate its impact
- [ ] Run without API float filter (`use_api_float=False`) to match E2 SOFT filter
- [ ] Compare trade lists between E2 and current BT to find specific divergences
- [ ] Check if E2 date range excluded the Jul-Sep regime shift period
- [ ] Test EXIT_TIME sweep (11:00-17:00) to confirm E2's Sharpe peak at 11:00-12:00
- [ ] Add monthly/weekly PnL charts for regime analysis
