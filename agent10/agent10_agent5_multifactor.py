# %% Agent 5 — Multi-Factor Residual Spread Construction
#
# Tests whether multi-factor hedging (ARKG + XBI, ARKG + IBB, ARKG + SPY,
# PCA, or Kalman filter) produces a better mean-reversion signal for RXRX
# than the single-factor ARKG-only OLS baseline.
#
# Imports canonical engine functions from agent0_engine.py.

import os, sys, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure dependencies
for pkg in ["statsmodels", "scikit-learn"]:
    try:
        __import__(pkg.replace("-", "_").split("-")[0])
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                               "--quiet", "--break-system-packages"])

import statsmodels.api as sm
from sklearn.decomposition import PCA

# ── Import canonical engine ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent if "__file__" in dir() else Path("agent10")))
from agent0_engine import (
    build_spread, run_backtest, compute_stats, compute_costs,
    log_trial, s5_zexit_signal, s5_zexit_exit, add_features,
    daily_mtm_equity, compute_sharpe, walk_forward,
    DATA_DIR, OUT_DIR,
)

# ── Config ───────────────────────────────────────────────────────
LOOKBACK = 60
K_ENTRY = 1.5
Z_EXIT = 0.5
MAX_HOLD = 10
PCA_WINDOW = 250
KALMAN_DELTA = 1e-5
SEL_CUTOFF = "2026-01-01"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 72)
print("  AGENT 5: MULTI-FACTOR RESIDUAL SPREAD CONSTRUCTION")
print("=" * 72)

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════

rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)
rxrx = add_features(rxrx)

ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])

arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)
xbi = ctx[ctx["ticker"] == "XBI"].sort_values("date").reset_index(drop=True)
iwm = ctx[ctx["ticker"] == "IWM"].sort_values("date").reset_index(drop=True)
spy = ctx[ctx["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)

ibb = pd.read_parquet(DATA_DIR / "p4_IBB.parquet")
ibb["date"] = pd.to_datetime(ibb["date"])
ibb = ibb.sort_values("date").reset_index(drop=True)

print(f"  RXRX: {len(rxrx)} bars, {rxrx['date'].min().date()} to {rxrx['date'].max().date()}")
print(f"  ARKG: {len(arkg)} | XBI: {len(xbi)} | IBB: {len(ibb)} | IWM: {len(iwm)} | SPY: {len(spy)}")


# ══════════════════════════════════════════════════════════════════
# 2. MULTI-FACTOR SPREAD BUILDERS
# ══════════════════════════════════════════════════════════════════

def build_multifactor_spread(target_df, hedge_dfs, hedge_names, lookback=60):
    """OLS regression of log(target) on multiple log(hedge) factors.

    Returns DataFrame with columns: date, close, open, high, low, volume,
    log_price, spread, spread_z, hedge_beta_*, r_squared, atr20.
    """
    t_df = target_df[["date", "close", "open", "high", "low", "volume"]].copy()
    t_df["log_price"] = np.log(t_df["close"].clip(lower=0.01))

    high, low, prev_close = t_df["high"], t_df["low"], t_df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    t_df["atr20"] = tr.rolling(20).mean()

    merged = t_df.copy()
    for hdf, hname in zip(hedge_dfs, hedge_names):
        merged = merged.merge(
            hdf[["date", "close"]].rename(columns={"close": f"{hname}_close"}),
            on="date", how="inner"
        )
    merged = merged.sort_values("date").reset_index(drop=True)

    # Compute log prices for all hedges
    for hname in hedge_names:
        merged[f"{hname}_log"] = np.log(merged[f"{hname}_close"].clip(lower=0.01))

    merged["target_log"] = merged["log_price"]
    merged["spread"] = np.nan
    merged["r_squared"] = np.nan
    for hname in hedge_names:
        merged[f"beta_{hname}"] = np.nan

    for i in range(lookback, len(merged)):
        y = merged["target_log"].iloc[i-lookback:i+1].values
        x_cols = [merged[f"{hname}_log"].iloc[i-lookback:i+1].values for hname in hedge_names]
        x = sm.add_constant(np.column_stack(x_cols))
        try:
            model = sm.OLS(y, x).fit()
            merged.iloc[i, merged.columns.get_loc("spread")] = y[-1] - model.predict(x)[-1]
            merged.iloc[i, merged.columns.get_loc("r_squared")] = model.rsquared
            for k, hname in enumerate(hedge_names):
                merged.iloc[i, merged.columns.get_loc(f"beta_{hname}")] = model.params[k+1]
        except Exception:
            pass

    # Rolling z-score of spread (canonical method)
    spread = merged["spread"]
    merged["spread_z"] = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()

    return merged


def build_pca_spread(target_df, hedge_dfs, hedge_names, lookback=60, pca_window=250, n_components=2):
    """PCA-based spread: regress log(target) on first N components of hedge log prices.

    PCA is fit on trailing pca_window days (no lookahead).
    """
    t_df = target_df[["date", "close", "open", "high", "low", "volume"]].copy()
    t_df["log_price"] = np.log(t_df["close"].clip(lower=0.01))

    high, low, prev_close = t_df["high"], t_df["low"], t_df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    t_df["atr20"] = tr.rolling(20).mean()

    merged = t_df.copy()
    for hdf, hname in zip(hedge_dfs, hedge_names):
        merged = merged.merge(
            hdf[["date", "close"]].rename(columns={"close": f"{hname}_close"}),
            on="date", how="inner"
        )
    merged = merged.sort_values("date").reset_index(drop=True)

    for hname in hedge_names:
        merged[f"{hname}_log"] = np.log(merged[f"{hname}_close"].clip(lower=0.01))

    merged["target_log"] = merged["log_price"]
    merged["spread"] = np.nan
    merged["r_squared"] = np.nan
    merged["spread_z"] = np.nan

    # Need at least pca_window + lookback bars
    min_start = max(pca_window, lookback)

    for i in range(min_start, len(merged)):
        # Fit PCA on trailing pca_window of hedge log returns
        pca_start = max(0, i - pca_window + 1)
        hedge_logs = np.column_stack([
            merged[f"{hname}_log"].iloc[pca_start:i+1].values for hname in hedge_names
        ])
        if len(hedge_logs) < n_components + 1:
            continue

        try:
            pca = PCA(n_components=n_components)
            pc_scores = pca.fit_transform(hedge_logs)  # (window, n_components)

            # For the OLS regression, use last lookback+1 rows of PC scores
            ols_start = max(0, len(pc_scores) - lookback - 1)
            y_window = merged["target_log"].iloc[pca_start:i+1].values[ols_start:]
            x_window = sm.add_constant(pc_scores[ols_start:])

            model = sm.OLS(y_window, x_window).fit()
            merged.iloc[i, merged.columns.get_loc("spread")] = y_window[-1] - model.predict(x_window)[-1]
            merged.iloc[i, merged.columns.get_loc("r_squared")] = model.rsquared
        except Exception:
            pass

    # Rolling z-score of spread
    spread = merged["spread"]
    merged["spread_z"] = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()

    return merged


def build_kalman_spread(target_df, hedge_df, hedge_name="ARKG", lookback=60, delta=1e-5):
    """Kalman filter hedge ratio (single factor, online estimation).

    State: [alpha, beta] (intercept and slope)
    Transition: identity (random walk with covariance delta*I)
    Observation: y_t = alpha + beta * x_t + noise
    """
    t_df = target_df[["date", "close", "open", "high", "low", "volume"]].copy()
    t_df["log_price"] = np.log(t_df["close"].clip(lower=0.01))

    high, low, prev_close = t_df["high"], t_df["low"], t_df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    t_df["atr20"] = tr.rolling(20).mean()

    merged = t_df.merge(
        hedge_df[["date", "close"]].rename(columns={"close": "hedge_close"}),
        on="date", how="inner"
    ).sort_values("date").reset_index(drop=True)

    merged["hedge_log"] = np.log(merged["hedge_close"].clip(lower=0.01))
    merged["target_log"] = merged["log_price"]

    n = len(merged)
    merged["spread"] = np.nan
    merged["spread_z"] = np.nan
    merged["hedge_beta"] = np.nan
    merged["r_squared"] = np.nan

    # Initialize Kalman state from first OLS window
    if n < lookback:
        return merged

    y_init = merged["target_log"].iloc[:lookback].values
    x_init = sm.add_constant(merged["hedge_log"].iloc[:lookback].values)
    init_model = sm.OLS(y_init, x_init).fit()
    obs_var = np.var(init_model.resid)

    # State vector: [alpha, beta]
    theta = np.array(init_model.params)  # [intercept, slope]
    P = np.eye(2) * 1.0  # state covariance
    Q = np.eye(2) * delta  # process noise
    R = obs_var  # observation noise (scalar)

    for i in range(len(merged)):
        y_i = merged["target_log"].iloc[i]
        x_i = np.array([1.0, merged["hedge_log"].iloc[i]])

        # Predict step
        theta_pred = theta  # random walk: no change
        P_pred = P + Q

        # Update step
        y_hat = x_i @ theta_pred
        resid = y_i - y_hat
        S = x_i @ P_pred @ x_i + R  # innovation variance
        K = P_pred @ x_i / S  # Kalman gain

        theta = theta_pred + K * resid
        P = P_pred - np.outer(K, x_i) @ P_pred

        # Store
        merged.iloc[i, merged.columns.get_loc("spread")] = resid
        merged.iloc[i, merged.columns.get_loc("hedge_beta")] = theta[1]

    # Rolling z-score of Kalman residual
    spread = merged["spread"]
    merged["spread_z"] = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()

    # Rolling R-squared approximation: 1 - var(resid) / var(target_log) over lookback
    for i in range(lookback, len(merged)):
        y_win = merged["target_log"].iloc[i-lookback:i+1].values
        r_win = merged["spread"].iloc[i-lookback:i+1].values
        if np.isnan(r_win).any():
            continue
        var_y = np.var(y_win)
        var_r = np.var(r_win)
        if var_y > 0:
            merged.iloc[i, merged.columns.get_loc("r_squared")] = 1 - var_r / var_y

    return merged


# ══════════════════════════════════════════════════════════════════
# 3. BUILD ALL VARIANTS
# ══════════════════════════════════════════════════════════════════

print("\n  Building spread variants...")

variants = {}

# A: ARKG only (baseline — use canonical build_spread)
print("    A: ARKG only (baseline)...", end=" ", flush=True)
variants["A_ARKG"] = build_spread(rxrx, arkg, "ARKG", LOOKBACK)
print(f"done ({len(variants['A_ARKG'])} bars)")

# B: ARKG + XBI
print("    B: ARKG + XBI...", end=" ", flush=True)
variants["B_ARKG_XBI"] = build_multifactor_spread(
    rxrx, [arkg, xbi], ["ARKG", "XBI"], LOOKBACK)
print(f"done ({len(variants['B_ARKG_XBI'])} bars)")

# C: ARKG + IBB
print("    C: ARKG + IBB...", end=" ", flush=True)
variants["C_ARKG_IBB"] = build_multifactor_spread(
    rxrx, [arkg, ibb], ["ARKG", "IBB"], LOOKBACK)
print(f"done ({len(variants['C_ARKG_IBB'])} bars)")

# D: ARKG + SPY
print("    D: ARKG + SPY...", end=" ", flush=True)
variants["D_ARKG_SPY"] = build_multifactor_spread(
    rxrx, [arkg, spy], ["ARKG", "SPY"], LOOKBACK)
print(f"done ({len(variants['D_ARKG_SPY'])} bars)")

# E: PCA first 2 components of {ARKG, XBI, IBB, IWM, SPY}
print("    E: PCA(2) of ARKG+XBI+IBB+IWM+SPY...", end=" ", flush=True)
variants["E_PCA2"] = build_pca_spread(
    rxrx, [arkg, xbi, ibb, iwm, spy], ["ARKG", "XBI", "IBB", "IWM", "SPY"],
    lookback=LOOKBACK, pca_window=PCA_WINDOW, n_components=2)
print(f"done ({len(variants['E_PCA2'])} bars)")

# F: Kalman filter (ARKG only)
print("    F: Kalman filter (ARKG)...", end=" ", flush=True)
variants["F_Kalman"] = build_kalman_spread(
    rxrx, arkg, "ARKG", lookback=LOOKBACK, delta=KALMAN_DELTA)
print(f"done ({len(variants['F_Kalman'])} bars)")


# ══════════════════════════════════════════════════════════════════
# 4. RUN S5_ZEXIT ON ALL VARIANTS
# ══════════════════════════════════════════════════════════════════

print("\n  Running S5_zexit backtest on all variants...")

def sig_fn(row, i, df):
    return s5_zexit_signal(row, i, df, k_entry=K_ENTRY)

def exit_fn(sig, row, j, df):
    return s5_zexit_exit(sig, row, j, df, z_exit_thresh=Z_EXIT)

all_trades = {}
all_stats_sel = {}
all_stats_oos = {}

for vname, spread_df in variants.items():
    trades = run_backtest(spread_df, sig_fn, exit_fn, max_hold=MAX_HOLD)
    all_trades[vname] = trades

    sel = trades[trades["entry_date"] < SEL_CUTOFF]
    oos = trades[trades["entry_date"] >= SEL_CUTOFF]

    sel_stats = compute_stats(sel, label=f"{vname} SEL")
    oos_stats = compute_stats(oos, label=f"{vname} OOS")

    all_stats_sel[vname] = sel_stats
    all_stats_oos[vname] = oos_stats

    print(f"    {vname:15s}  SEL: n={sel_stats['n']:3d}  avg={sel_stats['avg_pnl']:+.3f}%  "
          f"Sharpe={sel_stats['sharpe']:5.2f}  |  OOS: n={oos_stats['n']:3d}  "
          f"avg={oos_stats['avg_pnl']:+.3f}%  Sharpe={oos_stats['sharpe']:5.2f}")


# ══════════════════════════════════════════════════════════════════
# 5. JACCARD SIMILARITY vs BASELINE A
# ══════════════════════════════════════════════════════════════════

print("\n  Computing Jaccard similarity of trade dates vs baseline A...")

baseline_trades = all_trades["A_ARKG"]
baseline_dates_sel = set(baseline_trades[baseline_trades["entry_date"] < SEL_CUTOFF]["entry_date"].tolist())
baseline_dates_oos = set(baseline_trades[baseline_trades["entry_date"] >= SEL_CUTOFF]["entry_date"].tolist())

jaccard_sel = {}
jaccard_oos = {}

for vname, trades in all_trades.items():
    sel_dates = set(trades[trades["entry_date"] < SEL_CUTOFF]["entry_date"].tolist())
    oos_dates = set(trades[trades["entry_date"] >= SEL_CUTOFF]["entry_date"].tolist())

    # Jaccard = |intersection| / |union|
    if len(baseline_dates_sel | sel_dates) > 0:
        jaccard_sel[vname] = len(baseline_dates_sel & sel_dates) / len(baseline_dates_sel | sel_dates)
    else:
        jaccard_sel[vname] = 0.0

    if len(baseline_dates_oos | oos_dates) > 0:
        jaccard_oos[vname] = len(baseline_dates_oos & oos_dates) / len(baseline_dates_oos | oos_dates)
    else:
        jaccard_oos[vname] = 0.0


# ══════════════════════════════════════════════════════════════════
# 6. HEDGE R-SQUARED ANALYSIS
# ══════════════════════════════════════════════════════════════════

print("\n  Computing rolling R-squared statistics...")

r2_means = {}
for vname, spread_df in variants.items():
    r2 = spread_df["r_squared"].dropna()
    r2_mean = r2.mean() if len(r2) > 0 else 0.0
    r2_means[vname] = r2_mean

# Check R-squared vs strategy performance correlation
r2_vals = [r2_means[v] for v in variants]
avg_pnl_vals = [all_stats_sel[v]["avg_pnl"] for v in variants]
if len(r2_vals) > 2:
    from scipy.stats import pearsonr
    r2_pnl_corr, r2_pnl_pval = pearsonr(r2_vals, avg_pnl_vals)
else:
    r2_pnl_corr, r2_pnl_pval = 0.0, 1.0


# ══════════════════════════════════════════════════════════════════
# 7. COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  COMPARISON TABLE — SELECTION WINDOW (before 2026-01-01)")
print("=" * 72)
print(f"  {'Variant':15s} {'N':>4s} {'AvgPnL':>8s} {'Sharpe':>7s} {'WR%':>6s} "
      f"{'PF':>6s} {'R2':>6s} {'Jaccard':>8s} {'TotalPnL':>10s}")
print("  " + "-" * 70)
for vname in variants:
    s = all_stats_sel[vname]
    print(f"  {vname:15s} {s['n']:4d} {s['avg_pnl']:+7.3f}% {s['sharpe']:6.2f} "
          f"{s['wr']:5.1f} {s['pf']:5.2f} {r2_means[vname]:5.3f} "
          f"{jaccard_sel[vname]:7.3f} {s['total_pnl_pct']:+9.2f}%")

print(f"\n  R2 vs avg_pnl correlation: r={r2_pnl_corr:+.3f} (p={r2_pnl_pval:.3f})")

print("\n" + "=" * 72)
print("  COMPARISON TABLE — 2026 OOS (>= 2026-01-01)")
print("=" * 72)
print(f"  {'Variant':15s} {'N':>4s} {'AvgPnL':>8s} {'Sharpe':>7s} {'WR%':>6s} "
      f"{'PF':>6s} {'R2_oos':>7s} {'Jaccard':>8s} {'TotalPnL':>10s}")
print("  " + "-" * 70)
for vname in variants:
    s = all_stats_oos[vname]
    # R2 mean for OOS period only
    oos_spread = variants[vname][variants[vname]["date"] >= pd.Timestamp(SEL_CUTOFF)]
    r2_oos = oos_spread["r_squared"].dropna().mean() if len(oos_spread) > 0 else 0.0
    print(f"  {vname:15s} {s['n']:4d} {s['avg_pnl']:+7.3f}% {s['sharpe']:6.2f} "
          f"{s['wr']:5.1f} {s['pf']:5.2f} {r2_oos:6.3f} "
          f"{jaccard_oos[vname]:7.3f} {s['total_pnl_pct']:+9.2f}%")


# ══════════════════════════════════════════════════════════════════
# 8. IDENTIFY BEST VARIANT + WALK-FORWARD (if beats baseline)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  WALK-FORWARD VALIDATION")
print("=" * 72)

baseline_sel_avg = all_stats_sel["A_ARKG"]["avg_pnl"]
best_variant = None
best_improvement = 0

for vname in variants:
    if vname == "A_ARKG":
        continue
    improvement = all_stats_sel[vname]["avg_pnl"] - baseline_sel_avg
    if improvement > best_improvement and all_stats_sel[vname]["n"] >= 10:
        best_improvement = improvement
        best_variant = vname

if best_variant and best_improvement > 0.05:
    print(f"\n  Best non-baseline variant: {best_variant}")
    print(f"  Improvement over A_ARKG: {best_improvement:+.3f}% avg PnL")
    print(f"\n  Running walk-forward (12m train / 3m test) on {best_variant}...")

    sel_spread = variants[best_variant][variants[best_variant]["date"] < SEL_CUTOFF].copy()
    wf = walk_forward(sel_spread, sig_fn, exit_fn, train_months=12, test_months=3)
    if not wf.empty:
        print(f"\n  Walk-forward results for {best_variant}:")
        for _, w in wf.iterrows():
            print(f"    {str(w['test_start'].date()):>12s} - {str(w['test_end'].date()):>12s}: "
                  f"n={w['n']:>3.0f}  avg={w['avg_pnl']:+.3f}%")
        wf_avg = wf["avg_pnl"].mean()
        wf_positive = (wf["avg_pnl"] > 0).sum()
        wf_total = len(wf)
        print(f"\n  WF average: {wf_avg:+.3f}%  ({wf_positive}/{wf_total} positive folds)")
        print(f"  Verdict: {'STABLE' if wf_avg > 0 and wf_positive >= wf_total * 0.5 else 'UNSTABLE'}")
    else:
        print("  No walk-forward folds (insufficient data)")

    # Also run WF on baseline for comparison
    print(f"\n  Walk-forward on baseline A_ARKG for comparison...")
    sel_spread_a = variants["A_ARKG"][variants["A_ARKG"]["date"] < SEL_CUTOFF].copy()
    wf_a = walk_forward(sel_spread_a, sig_fn, exit_fn, train_months=12, test_months=3)
    if not wf_a.empty:
        for _, w in wf_a.iterrows():
            print(f"    {str(w['test_start'].date()):>12s} - {str(w['test_end'].date()):>12s}: "
                  f"n={w['n']:>3.0f}  avg={w['avg_pnl']:+.3f}%")
        wf_a_avg = wf_a["avg_pnl"].mean()
        print(f"\n  WF average (baseline): {wf_a_avg:+.3f}%")
else:
    print(f"\n  No variant beats baseline by > 0.05% avg PnL in selection window.")
    print(f"  Baseline A_ARKG avg PnL: {baseline_sel_avg:+.3f}%")
    if best_variant:
        print(f"  Closest challenger: {best_variant} ({best_improvement:+.3f}%)")
    print("  Skipping walk-forward.")

    # Still run WF on baseline for reference
    print(f"\n  Walk-forward on baseline A_ARKG for reference...")
    sel_spread_a = variants["A_ARKG"][variants["A_ARKG"]["date"] < SEL_CUTOFF].copy()
    wf_a = walk_forward(sel_spread_a, sig_fn, exit_fn, train_months=12, test_months=3)
    if not wf_a.empty:
        for _, w in wf_a.iterrows():
            print(f"    {str(w['test_start'].date()):>12s} - {str(w['test_end'].date()):>12s}: "
                  f"n={w['n']:>3.0f}  avg={w['avg_pnl']:+.3f}%")
        wf_a_avg = wf_a["avg_pnl"].mean()
        print(f"    WF average: {wf_a_avg:+.3f}%")


# ══════════════════════════════════════════════════════════════════
# 9. KALMAN vs OLS COMPARISON (Variant F vs A)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  KALMAN vs OLS COMPARISON")
print("=" * 72)

# Compare z-scores where both are valid
kalman_df = variants["F_Kalman"]
ols_df = variants["A_ARKG"]

# Merge on date
kv = kalman_df[["date", "spread_z", "hedge_beta"]].rename(
    columns={"spread_z": "z_kalman", "hedge_beta": "beta_kalman"})
ov = ols_df[["date", "spread_z", "hedge_beta"]].rename(
    columns={"spread_z": "z_ols", "hedge_beta": "beta_ols"})
comparison = kv.merge(ov, on="date", how="inner").dropna()

if len(comparison) > 0:
    z_corr = comparison["z_kalman"].corr(comparison["z_ols"])
    beta_corr = comparison["beta_kalman"].corr(comparison["beta_ols"])
    z_diff_mean = (comparison["z_kalman"] - comparison["z_ols"]).mean()
    z_diff_std = (comparison["z_kalman"] - comparison["z_ols"]).std()
    print(f"  Z-score correlation (Kalman vs OLS):  r={z_corr:.4f}")
    print(f"  Beta correlation (Kalman vs OLS):     r={beta_corr:.4f}")
    print(f"  Z-score difference: mean={z_diff_mean:+.4f}, std={z_diff_std:.4f}")
    print(f"  Kalman beta range: [{comparison['beta_kalman'].min():.3f}, {comparison['beta_kalman'].max():.3f}]")
    print(f"  OLS beta range:    [{comparison['beta_ols'].min():.3f}, {comparison['beta_ols'].max():.3f}]")
else:
    print("  WARNING: No overlapping valid z-scores for Kalman vs OLS comparison")


# ══════════════════════════════════════════════════════════════════
# 10. LOG ALL TRIALS
# ══════════════════════════════════════════════════════════════════

print("\n  Logging trials...")
for vname in variants:
    for period, stats_dict in [("sel", all_stats_sel), ("oos", all_stats_oos)]:
        params = {
            "variant": vname,
            "lookback": LOOKBACK,
            "k_entry": K_ENTRY,
            "z_exit": Z_EXIT,
            "max_hold": MAX_HOLD,
            "period": period,
            "r2_mean": round(r2_means[vname], 4),
        }
        log_trial("agent5", f"multifactor_{vname}_{period}", params, stats_dict[vname])
print(f"  Logged {len(variants) * 2} trials.")


# ══════════════════════════════════════════════════════════════════
# 11. MATPLOTLIB FIGURE
# ══════════════════════════════════════════════════════════════════

print("\n  Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Agent 5: Multi-Factor Residual Spread Comparison", fontsize=14, fontweight="bold")

colors = {
    "A_ARKG": "#1f77b4",
    "B_ARKG_XBI": "#ff7f0e",
    "C_ARKG_IBB": "#2ca02c",
    "D_ARKG_SPY": "#d62728",
    "E_PCA2": "#9467bd",
    "F_Kalman": "#8c564b",
}

# Panel (a): Equity curves — Selection window
ax = axes[0, 0]
ax.set_title("(a) Equity Curves — Selection Window (pre-2026)")
for vname in variants:
    trades = all_trades[vname]
    sel_trades = trades[trades["entry_date"] < SEL_CUTOFF]
    if sel_trades.empty:
        continue
    cum_pnl = sel_trades["pnl_net_pct"].cumsum().values
    ax.plot(range(len(cum_pnl)), cum_pnl, label=f"{vname} (n={len(sel_trades)})",
            color=colors.get(vname, "gray"), linewidth=1.5)
ax.set_xlabel("Trade #")
ax.set_ylabel("Cumulative PnL (%)")
ax.legend(fontsize=8)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.grid(True, alpha=0.3)

# Panel (b): Equity curves — 2026 OOS
ax = axes[0, 1]
ax.set_title("(b) Equity Curves — 2026 OOS")
for vname in variants:
    trades = all_trades[vname]
    oos_trades = trades[trades["entry_date"] >= SEL_CUTOFF]
    if oos_trades.empty:
        continue
    cum_pnl = oos_trades["pnl_net_pct"].cumsum().values
    ax.plot(range(len(cum_pnl)), cum_pnl, label=f"{vname} (n={len(oos_trades)})",
            color=colors.get(vname, "gray"), linewidth=1.5)
ax.set_xlabel("Trade #")
ax.set_ylabel("Cumulative PnL (%)")
ax.legend(fontsize=8)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.grid(True, alpha=0.3)

# Panel (c): Rolling R-squared comparison
ax = axes[1, 0]
ax.set_title("(c) Rolling 60-day R-squared")
for vname in variants:
    spread_df = variants[vname]
    r2_series = spread_df.set_index("date")["r_squared"].dropna()
    if len(r2_series) > 0:
        ax.plot(r2_series.index, r2_series.values, label=vname,
                color=colors.get(vname, "gray"), linewidth=1.0, alpha=0.8)
ax.set_ylabel("R-squared")
ax.legend(fontsize=8)
ax.axvline(pd.Timestamp(SEL_CUTOFF), color="red", linestyle="--", alpha=0.5, label="OOS cutoff")
ax.grid(True, alpha=0.3)

# Panel (d): Z-score time series (last 250 days)
ax = axes[1, 1]
ax.set_title("(d) Spread Z-Score — Last 250 Trading Days")
for vname in variants:
    spread_df = variants[vname]
    z_series = spread_df.set_index("date")["spread_z"].dropna()
    if len(z_series) > 250:
        z_tail = z_series.iloc[-250:]
    else:
        z_tail = z_series
    if len(z_tail) > 0:
        ax.plot(z_tail.index, z_tail.values, label=vname,
                color=colors.get(vname, "gray"), linewidth=1.0, alpha=0.7)
ax.axhline(K_ENTRY, color="gray", linestyle=":", alpha=0.5)
ax.axhline(-K_ENTRY, color="gray", linestyle=":", alpha=0.5)
ax.axhline(Z_EXIT, color="gray", linestyle="--", alpha=0.3)
ax.axhline(-Z_EXIT, color="gray", linestyle="--", alpha=0.3)
ax.set_ylabel("Z-score")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = OUT_DIR / "agent5_multifactor.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Saved figure: {fig_path}")
plt.close()


# ══════════════════════════════════════════════════════════════════
# 12. CONCLUSION
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  CONCLUSION")
print("=" * 72)

# Find best variant overall
best_sel = max(all_stats_sel, key=lambda v: all_stats_sel[v]["avg_pnl"])
best_oos = max(all_stats_oos, key=lambda v: all_stats_oos[v]["avg_pnl"])

baseline_sel = all_stats_sel["A_ARKG"]
baseline_oos = all_stats_oos["A_ARKG"]

any_beats_baseline_sel = any(
    all_stats_sel[v]["avg_pnl"] > baseline_sel["avg_pnl"] + 0.05
    and all_stats_sel[v]["n"] >= 10
    for v in variants if v != "A_ARKG"
)
any_beats_baseline_oos = any(
    all_stats_oos[v]["avg_pnl"] > baseline_oos["avg_pnl"] + 0.05
    and all_stats_oos[v]["n"] >= 5
    for v in variants if v != "A_ARKG"
)

print(f"""
  SELECTION WINDOW:
    Best variant: {best_sel} (avg PnL {all_stats_sel[best_sel]['avg_pnl']:+.3f}%, n={all_stats_sel[best_sel]['n']})
    Baseline A_ARKG: avg PnL {baseline_sel['avg_pnl']:+.3f}%, n={baseline_sel['n']}
    Any variant beats baseline by >0.05%: {'YES' if any_beats_baseline_sel else 'NO'}

  2026 OOS:
    Best variant: {best_oos} (avg PnL {all_stats_oos[best_oos]['avg_pnl']:+.3f}%, n={all_stats_oos[best_oos]['n']})
    Baseline A_ARKG: avg PnL {baseline_oos['avg_pnl']:+.3f}%, n={baseline_oos['n']}
    Any variant beats baseline by >0.05%: {'YES' if any_beats_baseline_oos else 'NO'}

  R2 vs PERFORMANCE:
    Correlation (selection): r={r2_pnl_corr:+.3f} (p={r2_pnl_pval:.3f})
    Phase 4 finding was R2 NEGATIVELY correlated with performance.
    Multi-factor R2 {'CONFIRMS' if r2_pnl_corr < 0 else 'CONTRADICTS'} the Phase 4 finding.

  KALMAN vs OLS:
    Z-score correlation: {z_corr:.3f}
    Trade overlap (Jaccard): SEL={jaccard_sel.get('F_Kalman', 0):.3f}, OOS={jaccard_oos.get('F_Kalman', 0):.3f}
    Kalman {'IMPROVES' if all_stats_sel['F_Kalman']['avg_pnl'] > baseline_sel['avg_pnl'] else 'DOES NOT IMPROVE'} on static OLS.

  VERDICT: {'Multi-factor hedging provides meaningful improvement — consider adopting ' + best_sel if any_beats_baseline_sel and any_beats_baseline_oos else 'ARKG-only baseline remains the best choice. Multi-factor hedging adds complexity without consistent improvement.'}
""")

print("  AGENT 5 COMPLETE")
