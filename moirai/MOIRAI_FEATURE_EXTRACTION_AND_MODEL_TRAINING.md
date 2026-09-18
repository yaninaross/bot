# MOIRAI Path Feature Extraction & LightGBM Model Training

## Objective

Build a LightGBM regression model that predicts **actual max forward return** (60 minutes, from real market data) using structural features extracted from MOIRAI's 100 sample paths. The model's predicted value becomes the trading signal — higher predicted max return = higher conviction trade.

## Target Variable

**`actual_max_return_60m`** — the maximum of `(High[t+1:t+60] - Close[t]) / Close[t]` computed from real 1-minute OHLCV data. This is NOT from the paths — it's ground truth from the market.

---

## Complete Feature List (69 features)

### Group 1: Spike Probability (6 features)

| # | Feature | Definition |
|---|---------|-----------|
| 1 | `spike_prob_2pct` | Fraction of 100 paths where max return >= 2% |
| 2 | `spike_prob_3pct` | Fraction of 100 paths where max return >= 3% |
| 3 | `spike_prob_4pct` | Fraction of 100 paths where max return >= 4% |
| 4 | `spike_prob_6pct` | Fraction of 100 paths where max return >= 6% |
| 5 | `spike_prob_8pct` | Fraction of 100 paths where max return >= 8% |
| 6 | `spike_prob_10pct` | Fraction of 100 paths where max return >= 10% |

### Group 2: Peak Magnitude (8 features)

| # | Feature | Definition |
|---|---------|-----------|
| 7 | `expected_max_return` | Mean of per-path maximum returns |
| 8 | `p75_max` | 75th percentile of per-path maximum returns |
| 9 | `p90_max_return` | 90th percentile of per-path maximum returns |
| 10 | `p95_max_return` | 95th percentile of per-path maximum returns |
| 11 | `mean_return_top_20_paths` | Average max return of the 20 best paths (top quintile) |
| 12 | `mean_return_top_10_paths` | Average max return of the 10 best paths (top decile) |
| 13 | `p25_min` | 25th percentile of per-path minimum returns |
| 14 | `spread_top20_vs_bottom20` | Mean max of top 20 paths minus mean max of bottom 20 paths |

### Group 3: Timing (12 features)

| # | Feature | Definition |
|---|---------|-----------|
| 15 | `median_time_to_peak` | Median minute (0-59) when paths hit their max return |
| 16 | `pct_peak_in_first_5min` | Fraction of paths peaking in bars 0-4 |
| 17 | `pct_peak_in_first_15min` | Fraction of paths peaking in bars 0-14 |
| 18 | `pct_paths_new_high_last_15min` | Fraction of paths making their all-time high in bars 45-59 |
| 19 | `pct_paths_new_high_first_15min` | Fraction peaking in bars 0-14 (early spike) |
| 20 | `early_vs_late_peak_ratio` | (# paths peaking in bars 0-29) / (# paths peaking in bars 30-59) |
| 21 | `median_time_to_tp_3pct` | Median minute when paths first touch +3% (NaN -> 60 if never) |
| 22 | `median_time_to_tp_6pct` | Median minute when paths first touch +6% (NaN -> 60 if never) |
| 64 | `pct_tp_hit_under_10min_2pct` | Fraction of paths hitting +2% within bars 0-9 |
| 65 | `pct_tp_hit_under_10min_3pct` | Fraction of paths hitting +3% within bars 0-9 |
| 66 | `pct_tp_hit_under_10min_6pct` | Fraction of paths hitting +6% within bars 0-9 |
| 67 | `median_time_to_tp_2pct` | Median bar when paths first touch +2% (never = 60) |

### Group 4: Velocity & Momentum (8 features)

| # | Feature | Definition |
|---|---------|-----------|
| 23 | `mean_slope_first_5bars` | Average return across paths at bar 5 (early direction) |
| 24 | `mean_slope_first_10bars` | Average return across paths at bar 10 |
| 25 | `pct_positive_at_bar_5` | Fraction of paths with positive return at bar 5 |
| 26 | `pct_positive_at_bar_10` | Fraction of paths with positive return at bar 10 |
| 27 | `mean_return_bar30_to_bar60` | Average return in the second half (continuation vs reversal) |
| 28 | `mean_return_bar45_to_bar60` | Average return in last 15 minutes |
| 29 | `pct_paths_higher_at_t60_than_t30` | Fraction still climbing in second half |
| 30 | `acceleration` | Mean return bars 30-59 minus mean return bars 0-29 (positive = accelerating) |

### Group 5: Order of Operations (6 features)

| # | Feature | Definition |
|---|---------|-----------|
| 31 | `pct_peak_before_trough` | Fraction of paths where max return occurs at an earlier bar than min return |
| 32 | `mean_drawdown_before_peak` | Average max drawdown experienced BEFORE each path hits its peak |
| 33 | `mean_bars_to_first_dip_1pct` | Average bar index when a path first dips below -1% (never = 60) |
| 34 | `pct_paths_3pct_with_dd_under_1pct` | Fraction hitting +3% without ever losing 1% (clean moves) |
| 35 | `pct_paths_6pct_with_dd_under_2pct` | Fraction hitting +6% without ever losing 2% |
| 36 | `reward_risk_ratio` | Mean of (per-path max return / abs(per-path min return)) across paths |

### Group 6: Persistence & Dwell Time (7 features)

| # | Feature | Definition |
|---|---------|-----------|
| 37 | `pct_time_above_2pct` | Average fraction of 60 bars each path spends above +2% return |
| 38 | `pct_time_above_4pct` | Average fraction of 60 bars each path spends above +4% return |
| 39 | `pct_paths_sustained_above_2pct` | Fraction of paths that go above +2% AND stay above +2% through bar 59 |
| 40 | `pct_paths_sustained_above_4pct` | Same at +4% |
| 41 | `max_consecutive_positive_bars` | Mean longest streak of consecutive positive returns across paths |
| 42 | `mean_terminal_return` | Mean of all paths' final bar (t=59) return |
| 43 | `pct_paths_positive_terminal` | Fraction of paths with positive return at bar 59 |

### Group 7: TP/SL Sequencing (6 features)

| # | Feature | Definition |
|---|---------|-----------|
| 44 | `pct_tp_before_sl_2_1` | Fraction of paths hitting +2% before hitting -1% |
| 45 | `pct_tp_before_sl_3_1_5` | Fraction hitting +3% before -1.5% |
| 46 | `pct_tp_before_sl_4_2` | Fraction hitting +4% before -2% |
| 47 | `pct_tp_before_sl_6_3` | Fraction hitting +6% before -3% |
| 48 | `pct_tp_before_sl_8_4` | Fraction hitting +8% before -4% |
| 49 | `conditional_peak_after_surviving_30min` | Of paths still positive at bar 30, average peak in bars 30-59 |

### Group 8: Clustering (6 features)

| # | Feature | Definition |
|---|---------|-----------|
| 50 | `n_clusters` | Optimal k from k-means (k=2,3,4) on terminal returns, chosen by silhouette score |
| 51 | `largest_cluster_pct` | Fraction of paths in the dominant cluster |
| 52 | `largest_cluster_mean_ret` | Mean terminal return of the dominant cluster |
| 53 | `cluster_separation` | Distance between the two most separated cluster centers |
| 54 | `second_cluster_mean_ret` | Mean terminal return of the second largest cluster |
| 55 | `bimodality_coefficient` | Sarle's bimodality coefficient: (skew^2 + 1) / (kurtosis + 3) |

### Group 9: Distribution Shape (8 features)

| # | Feature | Definition |
|---|---------|-----------|
| 56 | `path_skew` | Skewness of terminal returns across 100 paths |
| 57 | `path_std` | Std dev of terminal returns |
| 58 | `kurtosis_terminal` | Kurtosis of terminal returns |
| 59 | `path_entropy` | Shannon entropy of terminal returns binned into 20 equal-width bins |
| 60 | `right_tail_mass` | Fraction of terminal returns above mean + 2*std |
| 61 | `cvar_95_upside` | Mean return of the top 5 paths (expected value of best outcomes) |
| 62 | `cvar_95_downside` | Mean return of the bottom 5 paths (worst case) |
| 63 | `upside_downside_ratio` | cvar_95_upside / abs(cvar_95_downside) |

### Group 10: Time-to-TP Variance (2 features)

| # | Feature | Definition |
|---|---------|-----------|
| 68 | `std_time_to_tp_3pct` | Std dev of time-to-TP at 3% across paths. Low = consistent explosive move, high = noisy grind |
| 69 | `std_time_to_tp_6pct` | Same at 6% |

---

## Feature Extraction Implementation

### Input

One .npy file per prediction, shape `(100, 60)` — each row is a path, each column is a minute. Values are **prices** (not returns). Convert to returns relative to the first bar:

```python
returns[i,t] = (paths[i,t] - paths[i,0]) / paths[i,0]
```

### Per-path derived arrays (compute once, reuse)

```python
per_path_max     = returns.max(axis=1)          # shape (100,)
per_path_min     = returns.min(axis=1)          # shape (100,)
per_path_argmax  = returns.argmax(axis=1)       # shape (100,)
per_path_argmin  = returns.argmin(axis=1)       # shape (100,)
terminal_returns = returns[:, -1]               # shape (100,)
```

### Time-to-TP helper

```python
def time_to_tp(returns, threshold):
    """returns: shape (100, 60). Returns array of shape (100,) with bar index of first TP hit."""
    hit_mask = returns >= threshold                    # (100, 60) bool
    first_hit = np.argmax(hit_mask, axis=1)            # (100,)
    never_hit = ~hit_mask.any(axis=1)                  # (100,) bool
    first_hit[never_hit] = 60                          # convention: never hit = 60
    return first_hit

ttt_2pct = time_to_tp(returns, 0.02)
ttt_3pct = time_to_tp(returns, 0.03)
ttt_6pct = time_to_tp(returns, 0.06)
```

### Clustering

Use `sklearn.cluster.KMeans` on `terminal_returns.reshape(-1,1)`. Test k=2,3,4. Pick k with highest `silhouette_score`. If all silhouette scores < 0.3, set `n_clusters=1`.

### Handling NaN/Inf

Clip all features to [-100, 100]. Replace NaN with 0. Replace Inf with the clip bound.

### Output

One CSV per ticker: `{TICKER}_moirai_features.csv` with Datetime index + 69 feature columns + `actual_max_return_60m` target column.

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
| LGB_TRAIN | Mar 1 - Apr 30 | 2 months | Train LightGBM on features -> actual max return |
| WF_OPT | May 1 - Jun 30 | 2 months | Optimize TP/SL/threshold on frozen LightGBM predictions |
| TEST | Jul 1 - Jul 31 | 1 month | Evaluate with everything frozen |

### LightGBM Internal Validation

Do NOT use May-Jun for LightGBM early stopping. Split within the training window:

| LGB split | Dates | Purpose |
|-----------|-------|---------|
| Train | Mar 1 - Apr 15 | Fit trees |
| Val | Apr 16 - Apr 30 | Early stopping only |

Then retrain on full Mar-Apr with the best iteration count from that val split. This keeps May-Jun completely blind to LightGBM.

---

## Model Training

### Step 1: Data Assembly

Concatenate all per-ticker CSVs into one master dataframe. Add a `ticker` column. Total rows expected: ~5,000-15,000 (89 tickers x 60-170 predictions each).

### Step 2: Train/Test Split

Split by date, NOT random. This is time series — no future leakage.

- **Train:** Mar 1 - Apr 30, 2026
- **Held out (do not touch):** May 1 onwards

### Step 3: Correlation Diagnostic (run before model training)

```python
import scipy.stats as stats

train_df = master[master['date'] <= '2026-04-30']
feature_cols = [list of 69 features]

correlations = {}
for col in feature_cols:
    r, p = stats.spearmanr(train_df[col], train_df['actual_max_return_60m'], nan_policy='omit')
    correlations[col] = {'spearman_r': r, 'p_value': p}

corr_df = pd.DataFrame(correlations).T.sort_values('spearman_r', ascending=False)
print(corr_df.head(20))
print(corr_df.tail(10))
```

**Report:** Top 20 and bottom 10 features by Spearman r. If no feature has |r| > 0.10, the paths contain very weak information about actual outcomes — flag this but proceed with the model.

### Step 4: Decile Monotonicity Check

For the top 10 features by |Spearman r|:

```python
for col in top_10_features:
    train_df['decile'] = pd.qcut(train_df[col], 10, labels=False, duplicates='drop')
    decile_stats = train_df.groupby('decile')['actual_max_return_60m'].mean()

    values = decile_stats.values
    mono_pairs = sum(1 for i in range(len(values)-1) if values[i+1] >= values[i])
    total_pairs = len(values) - 1
    print(f"{col}: {mono_pairs}/{total_pairs} monotonic ({'GOOD' if mono_pairs >= 7 else 'WEAK'})")
    print(f"  D1={values[0]:.3f}%  D10={values[-1]:.3f}%  lift={values[-1]/max(values[0], 0.001):.1f}x")
```

**Report:** Feature name, monotonicity score, D1 vs D10 avg return, lift ratio.

### Step 5: LightGBM Model Training

```python
import lightgbm as lgb

feature_cols = [list of 69 features]

# Internal train/val split for early stopping
train_internal = train_df[train_df['date'] <= '2026-04-15']
val_internal   = train_df[train_df['date'] > '2026-04-15']

X_ti = train_internal[feature_cols]
y_ti = train_internal['actual_max_return_60m']
X_vi = val_internal[feature_cols]
y_vi = val_internal['actual_max_return_60m']

model_es = lgb.LGBMRegressor(
    n_estimators=1000,
    max_depth=5,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)

model_es.fit(
    X_ti, y_ti,
    eval_set=[(X_vi, y_vi)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)

best_iter = model_es.best_iteration_
print(f"Best iteration from early stopping: {best_iter}")

# Retrain on full Mar-Apr with the best iteration count
X_train_full = train_df[feature_cols]
y_train_full = train_df['actual_max_return_60m']

model_final = lgb.LGBMRegressor(
    n_estimators=best_iter,
    max_depth=5,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)

model_final.fit(X_train_full, y_train_full)
```

**Freeze this model. Save to disk. Do not retrain.**

### Step 6: Score May-Jun for the WF Optimizer

```python
import joblib

# Load frozen model
model = joblib.load('/tmp/moirai_lgbm_model.pkl')

# Score May-Jun data (the WF_OPT period)
opt_df = master[(master['date'] >= '2026-05-01') & (master['date'] <= '2026-06-30')]
opt_df['predicted_max_return'] = model.predict(opt_df[feature_cols])

# Score Jul data (the TEST period)
test_df = master[(master['date'] >= '2026-07-01') & (master['date'] <= '2026-07-31')]
test_df['predicted_max_return'] = model.predict(test_df[feature_cols])

# Save both for the optimizer
opt_df.to_csv('/tmp/moirai_opt_scored.csv', index=True)
test_df.to_csv('/tmp/moirai_test_scored.csv', index=True)
joblib.dump(model, '/tmp/moirai_lgbm_model.pkl')
```

### Step 7: Model Evaluation Diagnostics

Run these on the WF_OPT period (May-Jun) as a sanity check. These predictions are fully out-of-sample for the LightGBM model.

**7A: Overall correlation**

```python
y_pred = opt_df['predicted_max_return']
y_actual = opt_df['actual_max_return_60m']
r, p = stats.spearmanr(y_pred, y_actual)
print(f"Spearman correlation (predicted vs actual): {r:.4f}  p={p:.4e}")
```

| Result | Interpretation |
|--------|---------------|
| r > 0.30 | Strong — viable standalone strategy |
| r = 0.15-0.30 | Moderate — useful as secondary signal |
| r < 0.15 | Weak — paths don't carry enough info |

**7B: Decile analysis**

```python
opt_df['decile'] = pd.qcut(opt_df['predicted_max_return'], 10, labels=False, duplicates='drop')

print(f"{'Decile':>8}  {'N':>5}  {'AvgPredicted':>14}  {'AvgActual':>11}  {'ActualMedian':>13}")
for d in range(10):
    mask = opt_df['decile'] == d
    n = mask.sum()
    pred_avg = opt_df.loc[mask, 'predicted_max_return'].mean()
    act_avg  = opt_df.loc[mask, 'actual_max_return_60m'].mean()
    act_med  = opt_df.loc[mask, 'actual_max_return_60m'].median()
    print(f"  D{d+1:>2}     {n:>5}  {pred_avg:>+13.3f}%  {act_avg:>+10.3f}%  {act_med:>+12.3f}%")
```

**7C: Feature importance**

```python
importance = pd.Series(model.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=False)
print("\nTop 20 features by importance:")
for feat, imp in importance.head(20).items():
    print(f"  {feat:<45} {imp:>6}")
```

**7D: Per-ticker breakdown**

```python
for ticker in opt_df['ticker'].unique():
    t_mask = opt_df['ticker'] == ticker
    if t_mask.sum() < 10:
        continue
    r_t, _ = stats.spearmanr(opt_df.loc[t_mask, 'predicted_max_return'],
                              opt_df.loc[t_mask, 'actual_max_return_60m'])
    print(f"  {ticker:<8}  r={r_t:>+.3f}  n={t_mask.sum()}")
```

### Step 8: Charts (4 charts)

1. **Feature importance bar chart** — top 20 features, horizontal bars
2. **Decile analysis** — actual max return per decile of predicted max return (bar chart). Include monotonicity score in title.
3. **Scatter plot** — predicted vs actual max return, color by ticker, regression line with R^2
4. **Per-ticker Spearman correlation bar chart** — sorted descending, green > 0.20, blue > 0, red < 0

Save to `/tmp/moirai_lgbm_diagnostics.png` as 2x2 figure.

---

## Decision Criteria

After running the full investigation, report these three numbers:

1. **May-Jun Spearman r** (predicted vs actual max return, fully out-of-sample)
2. **D10 actual avg return** (what the top decile actually delivers)
3. **Number of tickers with per-ticker r > 0.20**

| Outcome | Action |
|---------|--------|
| r > 0.30 AND D10 > 2.0% AND > 5 tickers with r > 0.20 | Proceed to WF Optimizer (see MOIRAI_WF_OPTIMIZER_INSTRUCTIONS.md) |
| r = 0.15-0.30 AND D10 > 1.0% | Use as secondary filter in S3 pipeline |
| r < 0.15 OR D10 < 0.5% | MOIRAI paths lack tradeable information — archive |

---

## File Locations

| What | Path |
|------|------|
| Raw .npy paths | TBD (wherever batch job saves them) |
| Per-ticker feature CSVs | `gs://<GCS_BUCKET>/Algos/MOIRAI/features/` |
| Master training CSV | `/tmp/moirai_features_master.csv` |
| Frozen LightGBM model | `/tmp/moirai_lgbm_model.pkl` |
| Scored OPT data (May-Jun) | `/tmp/moirai_opt_scored.csv` |
| Scored TEST data (Jul) | `/tmp/moirai_test_scored.csv` |
| Market data | `gs://<GCS_BUCKET>/Data/Inference_v2/{TICKER}_master_inference.parquet` |
| Diagnostics chart | `/tmp/moirai_lgbm_diagnostics.png` |
