# MOIRAI SHORT Walk-Forward Optimizer Instructions

## What This Script Does

Takes the **frozen** SHORT LightGBM model's predictions on **May 1 - Jun 30, 2026** data and grid-searches for the best SHORT trade configuration:
- **Signal threshold** — minimum LightGBM predicted max drawdown to enter a SHORT trade
- **TP** — take profit exit: price drops this % below entry
- **SL** — stop loss exit: price rises this % above entry
- **max_hold_bars** — maximum holding period (minutes) before forced exit

The output is a single frozen config (threshold + TP + SL + max_hold_bars) evaluated on July.

## Prerequisites

Outputs from the SHORT LightGBM training phase (see `MOIRAI_SHORT_FEATURE_EXTRACTION_AND_MODEL_TRAINING.md`):

1. **Frozen LightGBM model** at `/tmp/moirai_lgbm_short_model.pkl`
2. **Scored OPT data** at `/tmp/moirai_short_opt_scored.csv` — May-Jun predictions with `predicted_max_drawdown` column
3. **Scored TEST data** at `/tmp/moirai_short_test_scored.csv` — Jul predictions with `predicted_max_drawdown` column

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
| WF_OPT | May 1 - Jun 30 | 2 months | This script: optimize SHORT TP/SL/threshold/hold_time |
| TEST | Jul 1 - Jul 31 | 1 month | Evaluate with everything frozen |

---

## Input Data

The optimizer needs one DataFrame with these columns, covering **May 1 - Jun 30, 2026 only**:

| Column | Type | Source |
|--------|------|--------|
| `datetime` | datetime | Prediction timestamp (minute-level) |
| `ticker` | str | Asset symbol |
| `predicted_max_drawdown` | float | Frozen LightGBM output (model trained on Mar-Apr only) |
| `forward_close` | float[60] | 60 subsequent 1-minute close prices after entry |
| `forward_high` | float[60] | 60 subsequent 1-minute high prices after entry |
| `forward_low` | float[60] | 60 subsequent 1-minute low prices after entry |
| `entry_price` | float | Close price at prediction bar |

The forward bars come from the master_inference parquets in GCS (`gs://<GCS_BUCKET>/Data/Inference_v2/{TICKER}_master_inference.parquet`). Join on ticker + datetime.

### Entry Time Filtering

Filter predictions to **entry_minute = 375** (9:30 AM ET), matching the existing SHORT Phase A/B models. Only predictions at this specific minute enter the optimizer.

```python
# Filter to SHORT entry time
df_pst = df.copy()
df_pst.index = df_pst.index.tz_convert('America/Los_Angeles')
entry_min = df_pst.index.hour * 60 + df_pst.index.minute
opt_df = df_pst[entry_min == 375]  # 9:30 entry
```

---

## SHORT Trade Simulation (Numba)

**Key difference from LONG:** Entry is a SHORT sell. TP is BELOW entry (price drops = profit). SL is ABOVE entry (price rises = loss). Check SL (High) before TP (Low) within each bar — conservative convention.

```python
@numba.njit
def simulate_one_short_trade(forward_high, forward_low, forward_close,
                              entry_price, tp_pct, sl_pct, max_bars):
    tp_price = entry_price * (1.0 - tp_pct / 100.0)   # TP below entry
    sl_price = entry_price * (1.0 + sl_pct / 100.0)   # SL above entry

    for bar in range(max_bars):
        # Check SL first — HIGH hits SL for SHORT (price rising against us)
        if forward_high[bar] >= sl_price:
            return -sl_pct, bar + 1, -1   # pnl, bars_held, exit_type (SL)
        # Check TP — LOW hits TP for SHORT (price dropping in our favor)
        if forward_low[bar] <= tp_price:
            return +tp_pct, bar + 1, +1   # pnl, bars_held, exit_type (TP)

    # Timeout at max_hold_bars
    timeout_pnl = (entry_price - forward_close[max_bars - 1]) / entry_price * 100.0
    return timeout_pnl, max_bars, 0       # pnl, bars_held, exit_type (timeout)
```

**Important:** SL checked before TP within each bar. This is the conservative convention — if both could be hit in the same bar, assume SL hit first (worst case for us).

**Important:** Forward arrays are always 60 elements. `max_bars` just controls how far into the array the simulator looks.

Vectorized version:

```python
@numba.njit
def simulate_all_short_trades(forward_highs, forward_lows, forward_closes,
                               entry_prices, tp_pct, sl_pct, max_bars):
    n = len(entry_prices)
    pnls = np.empty(n, dtype=np.float64)
    bars_held = np.empty(n, dtype=np.int32)
    exit_types = np.empty(n, dtype=np.int32)

    for i in range(n):
        pnls[i], bars_held[i], exit_types[i] = simulate_one_short_trade(
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
    'tp_pct':           [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    'sl_pct':           [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    'max_hold_bars':    [15, 30, 45, 60],
}
# Total: 9 x 8 x 7 x 4 = 2,016 raw combos
# After SL <= TP filter: ~1,000 valid combos
```

**Constraint:** Reject any combo where `sl_pct > tp_pct`. Same risk-management rule as LONG.

**Note:** These TP/SL values are appropriate for the 60-minute MOIRAI window. The existing full-day SHORT models use larger TP/SL (6-16%) because they can hold until market close. MOIRAI exits at max_hold_bars.

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
    mask = df['predicted_max_drawdown'] >= threshold
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
                pnls, bars, exits = simulate_all_short_trades(
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

**Mode 1 -- Pooled (required):** One threshold/TP/SL/max_hold_bars for all SHORT tickers. Run the loop above on all tickers combined. This is the primary result.

**Mode 2 -- Per-ticker (if data allows):** For each ticker with >= 30 predictions in May-Jun, run the grid independently. Only report tickers where the best config has trades_per_month >= 2.

Report both. If per-ticker is consistently better, the SHORT signal is ticker-specific. If pooled wins, the signal generalizes.

---

## Output Tables

### Table 1: Top 20 Pooled SHORT Configs

```
Rank | Threshold | TP   | SL   | MaxBars | AvgPnL | WR    | Trades/Mo | AvgBarsHeld | Composite
   1 |      1.50 | 3.0  | 1.5  |      30 | +1.80% | 64.0% |       3.5 |        16.2 |     1.38
   2 |      1.25 | 2.5  | 1.5  |      45 | +1.55% | 62.0% |       4.8 |        21.1 |     1.25
 ...
```

### Table 2: Threshold Sensitivity (best TP/SL/MaxBars at each threshold)

```
Threshold | BestTP | BestSL | BestBars | AvgPnL | WR    | Trades/Mo | N_trades
     0.50 |    2.0 |    1.0 |       30 | +0.60% | 53.0% |      10.8 |       22
     0.75 |    2.5 |    1.5 |       45 | +0.90% | 56.0% |       7.5 |       15
     1.00 |    3.0 |    1.5 |       30 | +1.25% | 60.0% |       5.2 |       10
 ...
```

### Table 3: Hold Time Sensitivity

```
MaxBars | BestThresh | BestTP | BestSL | AvgPnL | WR    | Trades/Mo | Timeout%
     15 |       2.00 |   2.0  |   1.0  | +1.40% | 68.0% |       2.5 |    15.0%
     30 |       1.50 |   3.0  |   1.5  | +1.80% | 64.0% |       3.5 |    10.0%
     45 |       1.25 |   4.0  |   2.0  | +1.60% | 60.0% |       4.8 |    18.0%
     60 |       1.00 |   5.0  |   2.5  | +1.40% | 57.0% |       6.2 |    25.0%
```

Key SHORT question: do dumps happen fast (favor short max_hold_bars) or slow (favor longer hold)?

### Table 4: Per-Ticker Breakdown (best pooled config)

```
Ticker | N_trades | AvgPnL | WR    | Best_trade | Worst_trade
BE     |        8 | +2.30% | 75.0% |     +3.00% |      -1.50%
OKLO   |        6 | +1.90% | 66.7% |     +4.00% |      -2.00%
 ...
```

### Table 5: Pooled vs Per-Ticker Summary

```
Mode       | AvgPnL | WR    | Trades/Mo | N_tickers
Pooled     | +1.55% | 62.0% |       4.8 |        30
Per-Ticker | +1.80% | 65.0% |       3.2 |        18
```

### Table 6: LONG vs SHORT MOIRAI Comparison

```
Direction | Pooled AvgPnL | D10 Actual | Spearman r | Deploy tickers
LONG      | +X.XX%        | +X.XX%     | +0.XXX     | XX
SHORT     | +X.XX%        | +X.XX%     | +0.XXX     | XX
```

---

## Charts (6 charts + 2 test charts, matplotlib)

Save all charts to `/tmp/moirai_short_wf_optimizer.png` as a single 3x2 figure.

### Chart 1: Grid Heatmap

TP (x-axis) x SL (y-axis), color = composite score, at the best threshold and best max_hold_bars. Annotate cells with avg_pnl values.

### Chart 2: Sensitivity (2x1 subplot)

- **Top:** Threshold sensitivity — x = signal_threshold, two lines: avg_pnl (left y-axis, blue) and trades_per_month (right y-axis, orange).
- **Bottom:** Hold time sensitivity — x = max_hold_bars (15/30/45/60), y = avg_pnl for the best config at each hold time. Bar chart.

### Chart 3: Equity Curve (SHORT)

Cumulative PnL on May-Jun using the #1 SHORT config, trades sorted chronologically. Include a horizontal line at y=0.

### Chart 4: Trade Distribution

Histogram of per-trade PnL for the #1 config. Bin width = 0.5%. Color TP hits green, SL hits red, timeouts gray.

### Chart 5: Per-Ticker PnL Bars

Horizontal bar chart, one bar per ticker, sorted by avg_pnl descending. Green bars for positive, red for negative. Mark RULE_1 tickers (BE, IREN, OKLO, ONDS, RKLB, WOLF).

### Chart 6: OPT Stability Scatter

Each valid config as a dot. x = trades_per_month, y = avg_pnl. Color by composite score.

---

## Final Blind Test (July)

After selecting the best config from OPT, run it on TEST with everything frozen.

```python
FROZEN_CONFIG = {
    'signal_threshold': best_threshold,
    'tp_pct': best_tp,
    'sl_pct': best_sl,
    'max_hold_bars': best_max_bars,
}

test_mask = test_df['predicted_max_drawdown'] >= FROZEN_CONFIG['signal_threshold']
test_trades = test_df[test_mask]

pnls, bars, exits = simulate_all_short_trades(
    test_highs[test_mask], test_lows[test_mask], test_closes[test_mask],
    test_entries[test_mask],
    FROZEN_CONFIG['tp_pct'], FROZEN_CONFIG['sl_pct'], FROZEN_CONFIG['max_hold_bars']
)

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

**Table 7: Frozen Config TEST Results**

```
Metric          | OPT (May-Jun)  | TEST (Jul)    | Degradation
Avg PnL         | +1.80%         | +1.35%        | 25%
Win Rate        | 64%            | 60%           | -
Trades/Month    | 3.5            | 3.0           | -
Total PnL       | +6.30%         | +4.05%        | -
```

### TEST Charts

Save to `/tmp/moirai_short_wf_test_results.png` as 2x1:

1. **Equity curve** — cumulative PnL on July, best config
2. **Per-ticker PnL bars** — horizontal bar chart for July only

---

## BQ Persistence

Write all results to BigQuery:

```python
table_id = '<GCP_PROJECT>.scoring_logs.wf_optimization_results'

for _, row in results_df.iterrows():
    bq_row = {
        'model_version': 'MOIRAI_LGB_SHORT_v1',
        'experiment_family': 'moirai_short',
        'signal_type': 'predicted_max_drawdown',
        'ticker': 'POOLED',
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
FROZEN_SHORT_CONFIG = {
    'signal_threshold': 1.50,   # example
    'tp_pct': 3.0,              # example
    'sl_pct': 1.5,              # example
    'max_hold_bars': 30,        # example
}
```

These 4 numbers plus the frozen SHORT LightGBM model from Phase 1 are the complete inputs to the July blind test.

---

## Decision Criteria

| Metric | Threshold | Action |
|--------|-----------|--------|
| TEST avg_pnl > 1.0% AND degradation < 40% | Deploy | Add to daily SHORT scoring pipeline |
| TEST avg_pnl 0.5-1.0% | Gate mode | Use as SHORT filter alongside existing models |
| TEST avg_pnl < 0.5% OR degradation > 60% | Archive | SHORT signal doesn't survive optimization |

The 1.0% threshold matches the existing SHORT deploy bar (lower than LONG's 1.8% because SHORT trades are inherently more volatile with squeeze risk).

---

## Comparison: LONG vs SHORT Optimizer

| Aspect | LONG Optimizer | SHORT Optimizer |
|--------|---------------|-----------------|
| Trade direction | BUY at entry | SELL SHORT at entry |
| TP check | High >= tp_price (price rises) | Low <= tp_price (price drops) |
| SL check | Low <= sl_price (price drops) | High >= sl_price (price rises) |
| SL before TP | Yes (conservative) | Yes (conservative) |
| Timeout PnL | (close - entry) / entry | (entry - close) / entry |
| Signal | predicted_max_return | predicted_max_drawdown |
| Deploy threshold | 1.8% avg PnL | 1.0% avg PnL |
| model_version | MOIRAI_LGB_v1 | MOIRAI_LGB_SHORT_v1 |
| Entry minute | TBD by LONG optimizer | 375 (9:30 AM) |
| Grid | Same structure | Same structure |

---

## Script Format

Single self-contained `.py` file. Config variables at the top. No `if __name__`. Colab-friendly. Print all tables with clear headers. Save charts to `/tmp/`. Write to BQ with skip-if-exists check on `(model_version, threshold, tp_pct, sl_pct, max_hold_bars, period)`.
