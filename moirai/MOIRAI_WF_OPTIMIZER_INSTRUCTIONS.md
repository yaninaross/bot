# MOIRAI Walk-Forward Optimizer Instructions

## What This Script Does

Takes the **frozen** LightGBM model's predictions on **May 1 - Jun 30, 2026** data and grid-searches for the best combination of:
- **Signal threshold** — minimum LightGBM predicted max return to enter a trade
- **TP** — take profit exit (% from entry)
- **SL** — stop loss exit (% from entry)
- **max_hold_bars** — maximum holding period (minutes) before forced exit

The output is a single frozen config (threshold + TP + SL + max_hold_bars) that gets evaluated on July in the final step.

## Prerequisites

This script requires the outputs from the LightGBM training phase (see `MOIRAI_FEATURE_EXTRACTION_AND_MODEL_TRAINING.md`):

1. **Frozen LightGBM model** at `/tmp/moirai_lgbm_model.pkl`
2. **Scored OPT data** at `/tmp/moirai_opt_scored.csv` — May-Jun predictions with `predicted_max_return` column
3. **Scored TEST data** at `/tmp/moirai_test_scored.csv` — Jul predictions with `predicted_max_return` column

The LightGBM model was trained on Mar-Apr only. May-Jun and Jul predictions are fully out-of-sample.

---

## 3-Phase Timeline (Zero Leakage)

```
        LGB_TRAIN         WF_OPT            TEST
       +----------+   +--------------+   +----------+
       | Mar - Apr |   |  May - Jun   |   | Jul 2026 |
       +----------+   +--------------+   +----------+
LGB:    trains here     frozen, predicts   frozen, predicts
WF OPT:                 searches TP/SL/
                        thresh (2 months)
TEST:                                      evaluate
                                           (touch nothing)
```

| Period | Dates | Duration | Purpose |
|--------|-------|----------|---------|
| LGB_TRAIN | Mar 1 - Apr 30 | 2 months | Train LightGBM (DONE — see other doc) |
| WF_OPT | May 1 - Jun 30 | 2 months | This script: optimize TP/SL/threshold/hold_time |
| TEST | Jul 1 - Jul 31 | 1 month | Evaluate with everything frozen |

---

## Input Data

The optimizer needs one DataFrame with these columns, covering **May 1 - Jun 30, 2026 only**:

| Column | Type | Source |
|--------|------|--------|
| `datetime` | datetime | Prediction timestamp (minute-level) |
| `ticker` | str | Asset symbol |
| `predicted_max_return` | float | Frozen LightGBM output (model trained on Mar-Apr only) |
| `forward_close` | float[60] | 60 subsequent 1-minute close prices after entry |
| `forward_high` | float[60] | 60 subsequent 1-minute high prices after entry |
| `forward_low` | float[60] | 60 subsequent 1-minute low prices after entry |
| `entry_price` | float | Close price at prediction bar |

The forward bars come from the master_inference parquets in GCS (`gs://<GCS_BUCKET>/Data/Inference_v2/{TICKER}_master_inference.parquet`). Join on ticker + datetime.

---

## Trade Simulation (Numba)

Copy the Numba trade simulator pattern from existing optimizers. The `max_bars` parameter controls holding period.

```python
@numba.njit
def simulate_one_trade(forward_high, forward_low, forward_close,
                       entry_price, tp_pct, sl_pct, max_bars):
    tp_price = entry_price * (1.0 + tp_pct / 100.0)
    sl_price = entry_price * (1.0 - sl_pct / 100.0)

    for bar in range(max_bars):
        # Check SL first (conservative — assume worst case hit first)
        if forward_low[bar] <= sl_price:
            return -sl_pct, bar + 1, -1   # pnl, bars_held, exit_type (SL)
        if forward_high[bar] >= tp_price:
            return +tp_pct, bar + 1, +1   # pnl, bars_held, exit_type (TP)

    # Timeout at max_hold_bars (NOT bar 60)
    timeout_pnl = (forward_close[max_bars - 1] - entry_price) / entry_price * 100.0
    return timeout_pnl, max_bars, 0       # pnl, bars_held, exit_type (timeout)
```

**Important:** Check SL before TP within each bar. This is the conservative convention used across all existing optimizers. If both TP and SL could be hit in the same bar, assume SL hit first.

**Important:** Forward bars arrays are always 60 elements long. The `max_bars` parameter just controls how far into that array the simulator looks. A config with `max_hold_bars=15` exits at bar 15 if neither TP nor SL hit.

For the full grid, vectorize across all trades:

```python
@numba.njit
def simulate_all_trades(forward_highs, forward_lows, forward_closes,
                        entry_prices, tp_pct, sl_pct, max_bars):
    n = len(entry_prices)
    pnls = np.empty(n, dtype=np.float64)
    bars_held = np.empty(n, dtype=np.int32)
    exit_types = np.empty(n, dtype=np.int32)

    for i in range(n):
        pnls[i], bars_held[i], exit_types[i] = simulate_one_trade(
            forward_highs[i], forward_lows[i], forward_closes[i],
            entry_prices[i], tp_pct, sl_pct, max_bars
        )
    return pnls, bars_held, exit_types
```

---

## Optimizer Grid

```python
GRID = {
    'signal_threshold': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    'tp_pct':           [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
    'sl_pct':           [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    'max_hold_bars':    [15, 30, 45, 60],
}
# Total: 9 x 8 x 7 x 4 = 2,016 raw combos
# After SL <= TP filter: ~1,000 valid combos
```

**Constraint:** Reject any combo where `sl_pct > tp_pct`. This prevents asymmetric risk configs where you lose more on losers than you win on winners.

---

## Scoring Function

```python
MIN_TRADES_PER_MONTH = 2.0

def composite_score(pnls, months_in_period=2.0):
    n_trades = len(pnls)
    trades_per_month = n_trades / months_in_period

    if trades_per_month < MIN_TRADES_PER_MONTH:
        return -999.0

    avg_pnl = np.mean(pnls)
    win_rate = np.sum(pnls > 0) / n_trades * 100.0

    score = (avg_pnl * 0.6
             + (win_rate - 50) * 0.02
             + min(trades_per_month, 8) * 0.05)
    return score
```

The `months_in_period = 2.0` for the May-Jun optimization window.

---

## Main Optimization Loop

```python
results = []

for threshold in GRID['signal_threshold']:
    # Filter trades above threshold
    mask = df['predicted_max_return'] >= threshold
    if mask.sum() < MIN_TRADES_PER_MONTH * 2:
        continue

    subset_highs  = forward_highs[mask]
    subset_lows   = forward_lows[mask]
    subset_closes = forward_closes[mask]
    subset_entries = entry_prices[mask]

    for tp in GRID['tp_pct']:
        for sl in GRID['sl_pct']:
            if sl > tp:
                continue
            for max_bars in GRID['max_hold_bars']:
                pnls, bars, exits = simulate_all_trades(
                    subset_highs, subset_lows, subset_closes,
                    subset_entries, tp, sl, max_bars
                )

                score = composite_score(pnls, months_in_period=2.0)

                results.append({
                    'threshold': threshold,
                    'tp_pct': tp,
                    'sl_pct': sl,
                    'max_hold_bars': max_bars,
                    'avg_pnl': np.mean(pnls),
                    'win_rate': np.sum(pnls > 0) / len(pnls) * 100,
                    'n_trades': len(pnls),
                    'trades_per_month': len(pnls) / 2.0,
                    'tp_hits': np.sum(exits == 1),
                    'sl_hits': np.sum(exits == -1),
                    'timeouts': np.sum(exits == 0),
                    'avg_bars_held': np.mean(bars),
                    'composite': score,
                })

results_df = pd.DataFrame(results).sort_values('composite', ascending=False)
```

---

## Two Modes: Pooled and Per-Ticker

**Mode 1 -- Pooled (required):** One threshold/TP/SL/max_hold_bars for all tickers. Run the loop above on all tickers combined. This is the primary result.

**Mode 2 -- Per-ticker (if data allows):** For each ticker with >= 30 predictions in May-Jun, run the grid independently. Only report tickers where the best config has trades_per_month >= 2.

Report both. If per-ticker is consistently better, the signal is ticker-specific. If pooled wins, the signal generalizes.

---

## Output Tables

### Table 1: Top 20 Pooled Configs

```
Rank | Threshold | TP   | SL   | MaxBars | AvgPnL | WR    | Trades/Mo | AvgBarsHeld | Composite
   1 |      1.50 | 3.0  | 1.5  |      30 | +2.10% | 68.0% |       4.2 |        18.3 |     1.61
   2 |      1.25 | 2.5  | 1.5  |      45 | +1.85% | 65.0% |       5.1 |        22.7 |     1.48
 ...
```

### Table 2: Threshold Sensitivity (best TP/SL/MaxBars at each threshold)

```
Threshold | BestTP | BestSL | BestBars | AvgPnL | WR    | Trades/Mo | N_trades
     0.50 |    2.0 |    1.0 |       30 | +0.80% | 55.0% |      12.3 |       25
     0.75 |    2.5 |    1.5 |       45 | +1.10% | 58.0% |       9.1 |       18
     1.00 |    3.0 |    1.5 |       30 | +1.45% | 62.0% |       6.5 |       13
 ...
```

This table answers: "is higher threshold always better, or does it peak and decline?"

### Table 3: Hold Time Sensitivity (best config at each max_hold_bars)

```
MaxBars | BestThresh | BestTP | BestSL | AvgPnL | WR    | Trades/Mo | Timeout%
     15 |       2.00 |   2.5  |   1.0  | +1.60% | 72.0% |       3.1 |    12.0%
     30 |       1.50 |   3.0  |   1.5  | +2.10% | 68.0% |       4.2 |     8.0%
     45 |       1.25 |   4.0  |   2.0  | +1.90% | 64.0% |       5.5 |    15.0%
     60 |       1.00 |   5.0  |   2.5  | +1.70% | 60.0% |       6.8 |    22.0%
```

This answers a key question: does the MOIRAI signal produce fast scalps or slow grinds? If `MaxBars=15` wins, the signal is about early momentum. If `MaxBars=60` wins, it's about persistence.

### Table 4: Per-Ticker Breakdown (best pooled config)

```
Ticker | N_trades | AvgPnL | WR    | Best_trade | Worst_trade
ACHR   |       12 | +2.50% | 75.0% |     +5.20% |      -1.50%
RKLB   |        8 | +1.80% | 62.5% |     +4.10% |      -2.00%
 ...
```

### Table 5: Pooled vs Per-Ticker Summary

```
Mode       | AvgPnL | WR    | Trades/Mo | N_tickers
Pooled     | +1.85% | 65.0% |       5.1 |        35
Per-Ticker | +2.10% | 68.0% |       3.8 |        22
```

---

## Charts (6 charts, matplotlib)

Save all charts to `/tmp/moirai_wf_optimizer.png` as a single 3x2 figure.

### Chart 1: Grid Heatmap

TP (x-axis) x SL (y-axis), color = composite score, at the best threshold and best max_hold_bars. Use `plt.imshow` or `sns.heatmap`. Annotate cells with avg_pnl values.

### Chart 2: Sensitivity (2x1 subplot)

- **Top:** Threshold sensitivity — x = signal_threshold, two lines: avg_pnl (left y-axis, blue) and trades_per_month (right y-axis, orange). Shows the selectivity tradeoff.
- **Bottom:** Hold time sensitivity — x = max_hold_bars (15/30/45/60), y = avg_pnl for the best config at each hold time. Bar chart.

### Chart 3: Equity Curve

Cumulative PnL on May-Jun using the #1 config, trades sorted chronologically. Include a horizontal line at y=0.

### Chart 4: Trade Distribution

Histogram of per-trade PnL for the #1 config. Bin width = 0.5%. Color TP hits green, SL hits red, timeouts gray.

### Chart 5: Per-Ticker PnL Bars

Horizontal bar chart, one bar per ticker, sorted by avg_pnl descending. Green bars for positive, red for negative.

### Chart 6: OPT Stability Scatter

Each of the ~1,000 valid configs as a dot. x = trades_per_month, y = avg_pnl. Color by composite score. Shows whether high-PnL configs have enough trades to be reliable.

---

## Final Blind Test (July)

After selecting the best config from the OPT period, run it on the TEST period with everything frozen.

```python
# Load frozen config from OPT
FROZEN_CONFIG = {
    'signal_threshold': best_threshold,   # from OPT
    'tp_pct': best_tp,                    # from OPT
    'sl_pct': best_sl,                    # from OPT
    'max_hold_bars': best_max_bars,       # from OPT
}

# Filter TEST data
test_mask = test_df['predicted_max_return'] >= FROZEN_CONFIG['signal_threshold']
test_trades = test_df[test_mask]

# Simulate trades
pnls, bars, exits = simulate_all_trades(
    test_highs[test_mask], test_lows[test_mask], test_closes[test_mask],
    test_entries[test_mask],
    FROZEN_CONFIG['tp_pct'], FROZEN_CONFIG['sl_pct'], FROZEN_CONFIG['max_hold_bars']
)

# Report
test_avg_pnl = np.mean(pnls)
test_wr = np.sum(pnls > 0) / len(pnls) * 100
test_tpm = len(pnls) / 1.0  # 1 month
opt_avg_pnl = results_df.iloc[0]['avg_pnl']
degradation = 1 - (test_avg_pnl / opt_avg_pnl) if opt_avg_pnl > 0 else float('inf')

print(f"OPT avg PnL:  {opt_avg_pnl:+.2f}%")
print(f"TEST avg PnL: {test_avg_pnl:+.2f}%")
print(f"Degradation:  {degradation:.0%}")
print(f"TEST WR:      {test_wr:.0f}%")
print(f"TEST trades:  {len(pnls)} ({test_tpm:.1f}/month)")
```

### TEST Output Tables

**Table 6: Frozen Config TEST Results**

```
Metric          | OPT (May-Jun)  | TEST (Jul)    | Degradation
Avg PnL         | +2.10%         | +1.65%        | 21%
Win Rate        | 68%            | 63%           | -
Trades/Month    | 4.2            | 3.8           | -
Total PnL       | +8.82%         | +6.27%        | -
```

**Table 7: Per-Ticker TEST Breakdown**

```
Ticker | N_trades | AvgPnL | WR    | Best  | Worst
ACHR   |        5 | +2.10% | 80.0% | +4.1% | -1.5%
 ...
```

### TEST Charts

Save to `/tmp/moirai_wf_test_results.png` as 2x1:

1. **Equity curve** — cumulative PnL on July, best config
2. **Per-ticker PnL bars** — horizontal bar chart for July only

---

## BQ Persistence

Write all results (not just top 20) to BigQuery:

```python
table_id = '<GCP_PROJECT>.scoring_logs.wf_optimization_results'

for _, row in results_df.iterrows():
    bq_row = {
        'model_version': 'MOIRAI_LGB_v1',
        'experiment_family': 'moirai',
        'signal_type': 'predicted_max_return',
        'ticker': 'POOLED',           # or ticker name for per-ticker mode
        'tp_pct': row['tp_pct'],
        'sl_pct': row['sl_pct'],
        'threshold': row['threshold'],
        'max_hold_bars': int(row['max_hold_bars']),
        'avg_pnl': row['avg_pnl'],
        'win_rate': row['win_rate'],
        'n_trades': int(row['n_trades']),
        'trades_per_month': row['trades_per_month'],
        'avg_bars_held': row['avg_bars_held'],
        'composite_score': row['composite'],
        'period': 'OPT',
        'period_start': '2026-05-01',
        'period_end': '2026-06-30',
        'run_timestamp': datetime.utcnow().isoformat(),
    }
```

---

## Frozen Config Output

The optimizer exports exactly **4 frozen values**:

```python
FROZEN_CONFIG = {
    'signal_threshold': 1.50,   # example
    'tp_pct': 3.0,              # example
    'sl_pct': 1.5,              # example
    'max_hold_bars': 30,        # example
}
```

These 4 numbers plus the frozen LightGBM model from Phase 1 are the complete inputs to the July blind test. Nothing else gets tuned.

---

## Decision Criteria

| Metric | Threshold | Action |
|--------|-----------|--------|
| TEST avg_pnl > 1.8% AND degradation < 40% | Deploy | Add to daily scoring pipeline |
| TEST avg_pnl 0.5-1.8% | Gate mode | Use as S3 filter, not standalone |
| TEST avg_pnl < 0.5% OR degradation > 60% | Archive | Signal doesn't survive optimization |

The 1.8% threshold matches S3's deploy bar. MOIRAI must clear the same standard.

---

## Script Format

Single self-contained `.py` file. Config variables at the top. No `if __name__`. Colab-friendly. Print all tables with clear headers. Save charts to `/tmp/`. Write to BQ with skip-if-exists check on `(model_version, threshold, tp_pct, sl_pct, max_hold_bars, period)`.
