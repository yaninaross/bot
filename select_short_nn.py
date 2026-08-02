"""
select_short_nn.py — Neural Network Config-Level Selection for SHORT
====================================================================
Trains on ~2900 per-config backtest results (from oracle_true_ceiling.py).
Features: unbiased BQ stats + hyperparams (~24 features per config).
Target: bt_full_avg (CSV backtest PnL ground truth).
Validation: Leave-one-TICKER-out CV (no leakage).

Models: MLP neural net (2 sizes), Ridge, GradientBoosting
Output: ticker rankings, config selection, strategy comparison

Input:  /tmp/oracle_per_config_results.csv
Output: /tmp/select_short_nn.png, tables
GPU NOT required. Colab-ready (no if __name__).
"""

import os, gc, sys, warnings, ctypes
import numpy as np, pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
except ImportError:
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

if _IN_COLAB:
    try:
        auth.authenticate_user()
    except Exception:
        pass


# ==============================================================================
# CONFIG
# ==============================================================================
INPUT_CSV  = '/tmp/oracle_per_config_results.csv'
OUTPUT_CHART = '/tmp/select_short_nn.png'

CURRENT_TICKERS = ['VOYG', 'NTLA', 'CLF', 'SMR']
TOP_K_VALUES = [6, 8, 10]
FILTERED_VA_OPT_RATIO = 2.0
W = 110

MODELS = {
    'MLP_64_32': (MLPRegressor, dict(
        hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
        max_iter=1000, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, random_state=42, alpha=0.001)),
    'MLP_128_64': (MLPRegressor, dict(
        hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
        max_iter=1000, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, random_state=42, alpha=0.001)),
    'Ridge': (Ridge, dict(alpha=1.0)),
    'GradBoosting': (GradientBoostingRegressor, dict(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)),
}

FEATURE_COLS = [
    'prob_thresh', 'tp_pct', 'sl_pct', 'sl_mult', 'grace_minutes',
    'entry_minute',
    'is_static', 'is_atr_sl', 'is_grace', 'is_limit_entry', 'is_ensemble',
    'is_exp_410', 'is_exp_401',
    'vol_filter_pct', 'pre_range_filter_pct',
    'opt_avg_pnl', 'opt_win_rate', 'opt_count', 'opt_freq',
    'val_a_avg_pnl', 'val_a_win_rate', 'val_a_count', 'val_a_freq',
    'va_opt_ratio', 'tp_sl_ratio',
]

TARGET = 'bt_full_avg'


# ==============================================================================
# PHASE 1: LOAD DATA + FEATURES
# ==============================================================================
print("=" * W)
print("  SHORT MODEL — NEURAL NETWORK CONFIG-LEVEL SELECTION")
print("  Training on per-config backtest results (~2900 rows)")
print("  Leave-one-TICKER-out CV, 4 models")
print("=" * W)

print(f"\n  PHASE 1: Loading data + feature engineering...")
df = pd.read_csv(INPUT_CSV)

df['is_ensemble'] = df['is_ensemble'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
df['is_limit_entry'] = df['is_limit_entry'].astype(int)

df['is_static'] = (df['tpsl_mode'] == 'static').astype(int)
df['is_atr_sl'] = (df['tpsl_mode'] == 'atr_sl').astype(int)
df['is_grace']  = (df['tpsl_mode'] == 'grace').astype(int)

df['is_exp_410'] = df['experiment'].str.startswith('410').astype(int)
df['is_exp_401'] = df['experiment'].str.startswith('401').astype(int)

df['tp_sl_ratio'] = df['tp_pct'] / df['sl_pct'].clip(lower=0.01)

for col in FEATURE_COLS:
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].fillna(0).astype(float)

df[TARGET] = df[TARGET].fillna(0).astype(float)

n_tickers = df['ticker'].nunique()
n_configs = len(df)
n_with_trades = (df['bt_full_n'] > 0).sum()

print(f"    {n_configs} configs across {n_tickers} tickers")
print(f"    {n_with_trades} configs with ≥1 trade ({100*n_with_trades/n_configs:.0f}%)")
print(f"    Target ({TARGET}): min={df[TARGET].min()*100:+.2f}%, med={df[TARGET].median()*100:+.2f}%, max={df[TARGET].max()*100:+.2f}%")
print(f"    Positive targets: {(df[TARGET] > 0).sum()}/{n_configs} ({100*(df[TARGET] > 0).mean():.0f}%)")
print(f"    {len(FEATURE_COLS)} features (all unbiased)")

print(f"\n  ── FEATURES ──")
for i, col in enumerate(FEATURE_COLS, 1):
    vals = df[col]
    print(f"    {i:2d}. {col:<25} min={vals.min():>10.3f}  med={vals.median():>10.3f}  max={vals.max():>10.3f}")

X = df[FEATURE_COLS].values
y = df[TARGET].values
tickers = sorted(df['ticker'].unique())


# ==============================================================================
# PHASE 2: LOO-TICKER CV
# ==============================================================================
print(f"\n  PHASE 2: LOO-Ticker CV ({n_tickers} folds × {len(MODELS)} models)...")

model_predictions = {}

for model_name, (model_cls, model_params) in MODELS.items():
    predictions = np.zeros(len(df))
    ticker_col = df['ticker'].values

    for i, test_ticker in enumerate(tickers):
        train_mask = ticker_col != test_ticker
        test_mask  = ticker_col == test_ticker

        X_train, y_train = X[train_mask], y[train_mask]
        X_test = X[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = model_cls(**model_params)
        model.fit(X_train_s, y_train)
        predictions[test_mask] = model.predict(X_test_s)

    df[f'pred_{model_name}'] = predictions
    model_predictions[model_name] = predictions

    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mae = np.mean(np.abs(y - predictions))
    rho, p = sp_stats.spearmanr(y, predictions)
    print(f"    {model_name:<15} R²={r2:+.3f}  MAE={mae:.4f}  ρ={rho:+.3f} (p={p:.4f})")

gc.collect()


# ==============================================================================
# PHASE 3: PER-CONFIG EVALUATION
# ==============================================================================
print(f"\n  PHASE 3: Per-config evaluation...")

with_trades = df[df['bt_full_n'] > 0]
print(f"    Metrics on configs WITH trades ({len(with_trades)} rows):")

print(f"\n  {'Model':<15} │ {'R²':>7} {'MAE':>8} {'RMSE':>8} {'ρ':>7} {'p-val':>8} │ {'Pos%':>5} {'Neg%':>5}")
print(f"  {'─'*15}─┼─{'─'*7}─{'─'*8}─{'─'*8}─{'─'*7}─{'─'*8}─┼─{'─'*5}─{'─'*5}")

model_metrics = {}
for model_name in MODELS:
    pred = with_trades[f'pred_{model_name}'].values
    actual = with_trades[TARGET].values

    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    rho, p = sp_stats.spearmanr(actual, pred)

    true_pos = set(np.where(actual > 0)[0])
    pred_pos = set(np.where(pred > 0)[0])
    pos_prec = len(true_pos & pred_pos) / len(pred_pos) if pred_pos else 0

    true_neg = set(np.where(actual <= 0)[0])
    pred_neg = set(np.where(pred <= 0)[0])
    neg_prec = len(true_neg & pred_neg) / len(pred_neg) if pred_neg else 0

    model_metrics[model_name] = {'r2': r2, 'mae': mae, 'rmse': rmse, 'rho': rho, 'p': p}
    print(f"  {model_name:<15} │ {r2:+6.3f} {mae:7.4f} {rmse:7.4f} {rho:+6.3f} {p:7.4f} │ {pos_prec*100:4.0f}% {neg_prec*100:4.0f}%")

best_model = max(model_metrics, key=lambda m: model_metrics[m]['rho'])
print(f"\n    Best model (by ρ): {best_model}")


# ==============================================================================
# PHASE 4: TICKER-LEVEL AGGREGATION
# ==============================================================================
print(f"\n  PHASE 4: Ticker-level aggregation...")

actual_ticker = df.groupby('ticker').agg(
    actual_full=('bt_full_avg', 'max'),
    actual_h2=('bt_h2_avg', 'max'),
    n_configs=('bt_full_avg', 'count'),
).reset_index()

for model_name in MODELS:
    pred_col = f'pred_{model_name}'
    agg = df.groupby('ticker').agg(
        pred_max=(pred_col, 'max'),
        pred_mean=(pred_col, 'mean'),
        pred_median=(pred_col, 'median'),
    ).reset_index()
    actual_ticker[f'{model_name}_max'] = agg.set_index('ticker').reindex(actual_ticker['ticker'])['pred_max'].values
    actual_ticker[f'{model_name}_mean'] = agg.set_index('ticker').reindex(actual_ticker['ticker'])['pred_mean'].values

oracle_ranking = actual_ticker.sort_values('actual_full', ascending=False)['ticker'].tolist()
oracle_top10 = oracle_ranking[:10]

print(f"\n  ── TICKER RANKING CORRELATION (predicted vs actual) ──")
print(f"  {'Model':<15} │ {'max→ρ':>8} {'top10':>6} │ {'mean→ρ':>8} {'top10':>6}")
print(f"  {'─'*15}─┼─{'─'*8}─{'─'*6}─┼─{'─'*8}─{'─'*6}")

best_ticker_model = None
best_ticker_rho = -1
best_ticker_agg = None

for model_name in MODELS:
    for agg_method in ['max', 'mean']:
        col = f'{model_name}_{agg_method}'
        rho_tk, _ = sp_stats.spearmanr(actual_ticker['actual_full'], actual_ticker[col])

        predicted_ranking = actual_ticker.sort_values(col, ascending=False)['ticker'].tolist()
        pred_top10 = predicted_ranking[:10]
        overlap = len(set(pred_top10) & set(oracle_top10))

        if agg_method == 'max':
            max_rho = rho_tk
            max_overlap = overlap
        else:
            mean_rho = rho_tk
            mean_overlap = overlap

        if rho_tk > best_ticker_rho:
            best_ticker_rho = rho_tk
            best_ticker_model = model_name
            best_ticker_agg = agg_method

    print(f"  {model_name:<15} │ {max_rho:+7.3f} {max_overlap:4d}/10 │ {mean_rho:+7.3f} {mean_overlap:4d}/10")

print(f"\n    Best ticker ranker: {best_ticker_model} ({best_ticker_agg}), ρ={best_ticker_rho:+.3f}")


# ==============================================================================
# PHASE 5: STRATEGY COMPARISON
# ==============================================================================
print(f"\n  PHASE 5: Strategy comparison...")

def get_ensemble_config(tk_df):
    ens = tk_df[(tk_df['is_ensemble'] == 1) &
                (tk_df['tpsl_mode'].isin(['static', 'atr_sl', 'grace'])) &
                (tk_df['va_opt_ratio'] <= FILTERED_VA_OPT_RATIO)]
    if ens.empty:
        return None
    return ens.loc[ens['val_a_avg_pnl'].idxmax()]


def compute_strategy(selected_tickers, config_method, model_name=None):
    per_tk = []
    for ticker in selected_tickers:
        tk = df[df['ticker'] == ticker]
        if tk.empty:
            continue

        if config_method == 'oracle':
            cfg = tk.loc[tk['bt_full_avg'].idxmax()]
        elif config_method == 'model':
            cfg = tk.loc[tk[f'pred_{model_name}'].idxmax()]
        elif config_method == 'ensemble':
            cfg = get_ensemble_config(tk)
            if cfg is None:
                continue
        else:
            continue

        per_tk.append({
            'ticker': ticker,
            'full_avg': cfg['bt_full_avg'], 'full_wr': cfg['bt_full_wr'],
            'full_n': cfg['bt_full_n'], 'full_total': cfg['bt_full_total'],
            'h1_avg': cfg['bt_h1_avg'], 'h1_n': cfg['bt_h1_n'], 'h1_wr': cfg['bt_h1_wr'],
            'h2_avg': cfg['bt_h2_avg'], 'h2_n': cfg['bt_h2_n'], 'h2_wr': cfg['bt_h2_wr'],
            'mode': cfg['tpsl_mode'], 'tp': cfg['tp_pct'], 'sl': cfg['sl_pct'],
        })

    if not per_tk:
        return None, []

    rdf = pd.DataFrame(per_tk)
    total_n = int(rdf['full_n'].sum())
    total_h1 = int(rdf['h1_n'].sum())
    total_h2 = int(rdf['h2_n'].sum())

    result = {
        'n_tickers': len(rdf),
        'full_n': total_n,
        'full_avg': (rdf['full_avg'] * rdf['full_n']).sum() / total_n if total_n > 0 else 0,
        'full_total': rdf['full_total'].sum(),
        'full_wr': (rdf['full_wr'] * rdf['full_n']).sum() / total_n if total_n > 0 else 0,
        'h1_n': total_h1,
        'h1_avg': (rdf['h1_avg'] * rdf['h1_n']).sum() / total_h1 if total_h1 > 0 else 0,
        'h1_wr': (rdf['h1_wr'] * rdf['h1_n']).sum() / total_h1 if total_h1 > 0 else 0,
        'h2_n': total_h2,
        'h2_avg': (rdf['h2_avg'] * rdf['h2_n']).sum() / total_h2 if total_h2 > 0 else 0,
        'h2_wr': (rdf['h2_wr'] * rdf['h2_n']).sum() / total_h2 if total_h2 > 0 else 0,
    }
    return result, per_tk

pred_col_best = f'{best_ticker_model}_{best_ticker_agg}'
model_ranking = actual_ticker.sort_values(pred_col_best, ascending=False)['ticker'].tolist()

strategies = {}
strategies_detail = {}

for k in TOP_K_VALUES:
    model_top_k = model_ranking[:k]

    key = f'{best_ticker_model}_TOP{k}_MODEL'
    s, d = compute_strategy(model_top_k, 'model', model_name=best_ticker_model)
    if s: strategies[key] = s; strategies_detail[key] = d

    key = f'{best_ticker_model}_TOP{k}_ENS'
    s, d = compute_strategy(model_top_k, 'ensemble')
    if s: strategies[key] = s; strategies_detail[key] = d

for k in TOP_K_VALUES:
    key = f'ORACLE_TOP{k}'
    s, d = compute_strategy(oracle_ranking[:k], 'oracle')
    if s: strategies[key] = s; strategies_detail[key] = d

    key = f'ORACLE_TOP{k}_ENS'
    s, d = compute_strategy(oracle_ranking[:k], 'ensemble')
    if s: strategies[key] = s; strategies_detail[key] = d

s, d = compute_strategy(CURRENT_TICKERS, 'ensemble')
if s: strategies['CURRENT'] = s; strategies_detail['CURRENT'] = d

# ── Ticker lists ──
print(f"\n  ── TICKER LISTS ──")
for k in TOP_K_VALUES:
    model_top_k = model_ranking[:k]
    overlap = len(set(model_top_k) & set(oracle_ranking[:k]))
    print(f"    {best_ticker_model}_TOP{k:<3} ({overlap}/{k} in ORACLE): {', '.join(model_top_k)}")
print(f"    ORACLE_TOP10         : {', '.join(oracle_ranking[:10])}")
print(f"    CURRENT              : {', '.join(CURRENT_TICKERS)}")

# ── Strategy table ──
print(f"\n  ── STRATEGY COMPARISON ──")
print(f"  {'Strategy':<30}│{'── FULL ──────────':>18} │{'── HALF1 ─────────':>18} │{'── HALF2 ─────────':>18}")
print(f"  {'':30}│{'N':>5} {'Avg':>8} {'WR':>5} │{'N':>5} {'Avg':>8} {'WR':>5} │{'N':>5} {'Avg':>8} {'WR':>5}")
print(f"  {'─'*30}┼{'─'*19}┼{'─'*19}┼{'─'*19}")

display_order = []
for k in TOP_K_VALUES:
    display_order.append(f'ORACLE_TOP{k}')
    display_order.append(f'ORACLE_TOP{k}_ENS')
    display_order.append(f'{best_ticker_model}_TOP{k}_MODEL')
    display_order.append(f'{best_ticker_model}_TOP{k}_ENS')
display_order.append('CURRENT')

for key in display_order:
    if key not in strategies:
        continue
    s = strategies[key]
    tag = ''
    if 'ORACLE' in key and '_ENS' not in key:
        tag = ' [BIASED]'
    elif 'CURRENT' in key:
        tag = ' [BASELINE]'

    print(f"  {key:<30}│{s['full_n']:4d} {s['full_avg']*100:+7.2f}% {s['full_wr']*100:4.0f}%"
          f" │{s['h1_n']:4d} {s['h1_avg']*100:+7.2f}% {s['h1_wr']*100:4.0f}%"
          f" │{s['h2_n']:4d} {s['h2_avg']*100:+7.2f}% {s['h2_wr']*100:4.0f}%{tag}")


# ── Per-ticker detail for best model strategy ──
best_k = TOP_K_VALUES[-1]
detail_key = f'{best_ticker_model}_TOP{best_k}_MODEL'
if detail_key in strategies_detail:
    print(f"\n  ── PER-TICKER DETAIL: {detail_key} ──")
    print(f"  {'Ticker':<8}│{'FULL':>8} {'WR':>5} {'N':>4} │{'H1':>8} │{'H2':>8} │ Config")
    print(f"  {'─'*8}┼{'─'*18}─┼{'─'*9}┼{'─'*9}┼{'─'*20}")
    for row in sorted(strategies_detail[detail_key], key=lambda r: -r['full_avg']):
        sl_str = f"sl={row['sl']:.3f}" if row['mode'] != 'atr_sl' else f"sm={row['sl']:.1f}"
        cfg = f"{row['mode'][:6]} tp={row['tp']:.3f} {sl_str}"
        print(f"  {row['ticker']:<8}│{row['full_avg']*100:+7.2f}% {row['full_wr']*100:4.0f}% {int(row['full_n']):3d}"
              f" │{row['h1_avg']*100:+7.2f}% │{row['h2_avg']*100:+7.2f}% │ {cfg}")


# ==============================================================================
# PHASE 6: FEATURE IMPORTANCE
# ==============================================================================
print(f"\n  PHASE 6: Feature importance...")

scaler_full = StandardScaler()
X_full_s = scaler_full.fit_transform(X)

if 'GradBoosting' in MODELS:
    gb_model = GradientBoostingRegressor(**MODELS['GradBoosting'][1])
    gb_model.fit(X_full_s, y)
    fi = pd.Series(gb_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n  ── Feature Importance (GradBoosting, full data) ──")
    for feat, imp in fi.head(15).items():
        bar = '█' * int(imp * 80)
        print(f"    {feat:<25} {imp:.4f}  {bar}")

if best_ticker_model.startswith('MLP'):
    mlp_full = MLPRegressor(**MODELS[best_ticker_model][1])
    mlp_full.fit(X_full_s, y)

    perm_importance = np.zeros(len(FEATURE_COLS))
    base_pred = mlp_full.predict(X_full_s)
    base_mse = np.mean((y - base_pred) ** 2)
    rng = np.random.RandomState(42)
    for j in range(len(FEATURE_COLS)):
        X_perm = X_full_s.copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])
        perm_pred = mlp_full.predict(X_perm)
        perm_mse = np.mean((y - perm_pred) ** 2)
        perm_importance[j] = perm_mse - base_mse

    fi_mlp = pd.Series(perm_importance, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n  ── Permutation Importance ({best_ticker_model}, full data) ──")
    for feat, imp in fi_mlp.head(15).items():
        scaled = imp / fi_mlp.max() if fi_mlp.max() > 0 else 0
        bar = '█' * int(scaled * 40)
        print(f"    {feat:<25} {imp:.6f}  {bar}")


# ==============================================================================
# PHASE 7: COMPARISON vs TICKER-LEVEL APPROACH
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PHASE 7: CONFIG-LEVEL vs TICKER-LEVEL COMPARISON")
print(f"{'='*W}")

print(f"""
  Previous approach (select_short_model.py):
    53 samples (ticker-aggregated), 30 features
    Best model: DecisionTree ρ=+0.461, Top10prec=50%

  This approach (per-config):
    {n_configs} samples, {len(FEATURE_COLS)} features
    Best model: {best_ticker_model} ρ={best_ticker_rho:+.3f}

  Advantages of per-config:
    ✓ {n_configs}x more training samples vs 53
    ✓ Model learns config-level patterns (which hyperparams work)
    ✓ Can select BOTH tickers AND configs
    ✓ Neural network viable at this scale
""")


# ==============================================================================
# PHASE 8: CHARTS
# ==============================================================================
print(f"  Generating charts...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SHORT — Neural Network Config-Level Selection', fontsize=14, fontweight='bold')

# Panel 1: Predicted vs Actual PnL (per-config scatter, best model)
ax = axes[0, 0]
pred_best = df[f'pred_{best_ticker_model}'].values * 100
actual_vals = y * 100
ax.scatter(actual_vals, pred_best, alpha=0.15, s=8, c='steelblue')
lim_val = max(abs(np.percentile(actual_vals, [2, 98])).max(),
              abs(np.percentile(pred_best, [2, 98])).max()) + 0.5
ax.plot([-lim_val, lim_val], [-lim_val, lim_val], 'r--', alpha=0.4)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Actual PnL (%)')
ax.set_ylabel('Predicted PnL (%)')
rho_all = model_metrics.get(best_ticker_model, {}).get('rho', 0)
ax.set_title(f'{best_ticker_model} LOO Predictions (ρ={rho_all:+.3f}, N={len(y)})')
ax.set_xlim(-lim_val, lim_val)
ax.set_ylim(-lim_val, lim_val)

# Panel 2: Ticker ranking — predicted vs actual
ax = axes[0, 1]
actual_rank = actual_ticker.sort_values('actual_full', ascending=False).reset_index(drop=True)
actual_rank['rank'] = range(1, len(actual_rank) + 1)

pred_sorted = actual_ticker.sort_values(pred_col_best, ascending=False).reset_index(drop=True)
pred_sorted['pred_rank'] = range(1, len(pred_sorted) + 1)

merged = actual_rank.merge(pred_sorted[['ticker', 'pred_rank']], on='ticker')
colors_scatter = ['#2E7D32' if t in oracle_top10 else '#90A4AE' for t in merged['ticker']]
ax.scatter(merged['rank'], merged['pred_rank'], c=colors_scatter, s=40, alpha=0.8)
ax.plot([1, n_tickers], [1, n_tickers], 'r--', alpha=0.3)
for _, row_m in merged[merged['rank'] <= 10].iterrows():
    ax.annotate(row_m['ticker'], (row_m['rank'], row_m['pred_rank']),
                fontsize=6, alpha=0.8)
ax.set_xlabel('Actual Rank (1=best)')
ax.set_ylabel('Predicted Rank')
ax.set_title(f'Ticker Ranking (ρ={best_ticker_rho:+.3f})')
ax.invert_yaxis()
ax.invert_xaxis()

# Panel 3: Strategy comparison bars (FULL period)
ax = axes[1, 0]
bar_keys = []
bar_vals = []
bar_colors = []
for k in TOP_K_VALUES:
    okey = f'ORACLE_TOP{k}'
    mkey = f'{best_ticker_model}_TOP{k}_MODEL'
    ekey = f'{best_ticker_model}_TOP{k}_ENS'
    if okey in strategies:
        bar_keys.append(f'ORC_{k}')
        bar_vals.append(strategies[okey]['full_avg'] * 100)
        bar_colors.append('#1565C0')
    if mkey in strategies:
        bar_keys.append(f'NN_{k}_M')
        bar_vals.append(strategies[mkey]['full_avg'] * 100)
        bar_colors.append('#2E7D32')
    if ekey in strategies:
        bar_keys.append(f'NN_{k}_E')
        bar_vals.append(strategies[ekey]['full_avg'] * 100)
        bar_colors.append('#66BB6A')

if 'CURRENT' in strategies:
    bar_keys.append('CURR')
    bar_vals.append(strategies['CURRENT']['full_avg'] * 100)
    bar_colors.append('#C62828')

x_pos = np.arange(len(bar_keys))
ax.bar(x_pos, bar_vals, color=bar_colors, alpha=0.8)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_keys, rotation=45, fontsize=8)
ax.set_ylabel('Avg PnL (%)')
ax.set_title('Strategy Comparison (FULL period)')

# Panel 4: Feature importance
ax = axes[1, 1]
if 'GradBoosting' in MODELS:
    top_feats = fi.head(12)
    y_pos = np.arange(len(top_feats))
    ax.barh(y_pos, top_feats.values, color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (GradBoosting)')

plt.tight_layout()
plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches='tight')
print(f"  Chart saved: {OUTPUT_CHART}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  FINAL SUMMARY")
print(f"{'='*W}")

best_strat_key = f'{best_ticker_model}_TOP10_MODEL'
oracle_strat_key = 'ORACLE_TOP10'
current_key = 'CURRENT'

s_best = strategies.get(best_strat_key, {})
s_oracle = strategies.get(oracle_strat_key, {})
s_current = strategies.get(current_key, {})

print(f"\n  Best NN model: {best_ticker_model}")
print(f"  Per-config ρ:  {model_metrics.get(best_ticker_model, {}).get('rho', 0):+.3f}")
print(f"  Ticker ρ:      {best_ticker_rho:+.3f} (aggregation: {best_ticker_agg})")

if s_best and s_oracle and s_current:
    print(f"\n  ORACLE_TOP10:    {s_oracle.get('full_avg',0)*100:+.2f}% FULL  {s_oracle.get('h2_avg',0)*100:+.2f}% H2  [biased ceiling]")
    print(f"  MODEL_TOP10:     {s_best.get('full_avg',0)*100:+.2f}% FULL  {s_best.get('h2_avg',0)*100:+.2f}% H2  [unbiased NN]")
    print(f"  CURRENT:         {s_current.get('full_avg',0)*100:+.2f}% FULL  {s_current.get('h2_avg',0)*100:+.2f}% H2  [baseline]")
    if s_current.get('full_avg', 0) != 0:
        improvement = (s_best.get('full_avg', 0) - s_current.get('full_avg', 0)) / abs(s_current['full_avg']) * 100
        print(f"  MODEL vs CURRENT: {improvement:+.0f}% improvement")
    gap = (s_oracle.get('full_avg', 0) - s_best.get('full_avg', 0))
    print(f"  MODEL vs ORACLE gap: {gap*100:+.2f}%")

print(f"\n  Selected tickers ({best_ticker_model}_TOP10):")
model_top10 = model_ranking[:10]
for i, tk in enumerate(model_top10, 1):
    in_oracle = "✓" if tk in oracle_top10 else " "
    tk_row = actual_ticker[actual_ticker['ticker'] == tk].iloc[0]
    print(f"    {i:2d}. {tk:<8} pred={tk_row[pred_col_best]*100:+.2f}%  actual={tk_row['actual_full']*100:+.2f}%  {in_oracle}")

print(f"\n{'='*W}")
