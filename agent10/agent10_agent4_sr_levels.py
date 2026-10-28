# %% Agent 4 — Price-Level Support/Resistance Gate for S5_zexit
#
# Tests whether aligning z-score signals with nearby S/R levels
# (swing highs/lows + volume-profile HVNs) improves S5_zexit.
#
# Output: agent10/results/agent4_sr_levels.png
#         Trials logged to shared trials_log

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Import canonical engine ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent if "__file__" in dir() else Path("agent10")))
from agent0_engine import (
    build_spread, run_backtest, compute_stats, compute_costs,
    log_trial, add_features, s5_zexit_signal, s5_zexit_exit,
    DATA_DIR, OUT_DIR,
)

# ── Config ───────────────────────────────────────────────────────
SEL_END = "2026-01-01"           # OOS starts here
SWING_WINDOWS = [5, 10, 20]      # bars each side for swing detection
VP_LOOKBACK = 120                 # trailing days for volume profile
VP_BIN_SIZE = 0.50                # price bin width for volume profile
VP_TOP_N = 3                      # number of HVN levels to keep
ATR_PROXIMITY = 1.0               # within 1 ATR counts as "near" S/R
ALPHAS = [0.5, 1.0, 2.0]         # proximity weighting multipliers
N_PERMUTATIONS = 1000             # shuffles for permutation test
K_ENTRY = 1.5                     # z-score entry threshold
Z_EXIT = 0.5                      # z-score exit threshold
MAX_HOLD = 10
SEED = 42

print("=" * 70)
print("  AGENT 4: SUPPORT/RESISTANCE GATE FOR S5_ZEXIT")
print("=" * 70)

# ── Load data ────────────────────────────────────────────────────
rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)

ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])
arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)

rxrx = add_features(rxrx)
print(f"  RXRX: {len(rxrx)} bars, {rxrx['date'].min().date()} to {rxrx['date'].max().date()}")
print(f"  ARKG: {len(arkg)} bars")

# ── Build canonical spread ───────────────────────────────────────
print("\n  Building canonical spread (OLS, lookback=60)...")
spread_df = build_spread(rxrx, arkg, "ARKG", 60)
print(f"  Spread: {len(spread_df)} bars after merge")


# ══════════════════════════════════════════════════════════════════
# 1. SWING HIGH/LOW DETECTION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 1: SWING HIGH/LOW DETECTION")
print("=" * 70)

def detect_swing_highs(prices_high, window):
    """Swing high: bar whose high is the max of the surrounding +-window bars."""
    n = len(prices_high)
    is_swing = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        local_slice = prices_high[i - window : i + window + 1]
        if prices_high[i] == local_slice.max() and np.sum(local_slice == local_slice.max()) == 1:
            is_swing[i] = True
    return is_swing

def detect_swing_lows(prices_low, window):
    """Swing low: bar whose low is the min of the surrounding +-window bars."""
    n = len(prices_low)
    is_swing = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        local_slice = prices_low[i - window : i + window + 1]
        if prices_low[i] == local_slice.min() and np.sum(local_slice == local_slice.min()) == 1:
            is_swing[i] = True
    return is_swing

# Detect swings for each window size on spread_df
swing_data = {}
for w in SWING_WINDOWS:
    highs_mask = detect_swing_highs(spread_df["high"].values, w)
    lows_mask = detect_swing_lows(spread_df["low"].values, w)
    swing_data[w] = {
        "highs": spread_df.loc[highs_mask, ["date", "high"]].rename(columns={"high": "level"}),
        "lows": spread_df.loc[lows_mask, ["date", "low"]].rename(columns={"low": "level"}),
    }
    n_h = highs_mask.sum()
    n_l = lows_mask.sum()
    print(f"  Window N={w:>2d}: {n_h:>3d} swing highs, {n_l:>3d} swing lows")


# ══════════════════════════════════════════════════════════════════
# 2. VOLUME-WEIGHTED PRICE CLUSTERS (TRAILING 120-DAY)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 2: VOLUME PROFILE (trailing 120-day, $0.50 bins)")
print("=" * 70)

def compute_volume_profile_hvn(df, bar_idx, lookback, bin_size, top_n):
    """Compute volume profile on trailing window, return top-N HVN price levels."""
    start_idx = max(0, bar_idx - lookback)
    window = df.iloc[start_idx : bar_idx + 1]
    if len(window) < 20:
        return []

    prices = window["close"].values
    volumes = window["volume"].values
    price_min = np.floor(prices.min() / bin_size) * bin_size
    price_max = np.ceil(prices.max() / bin_size) * bin_size
    bins = np.arange(price_min, price_max + bin_size, bin_size)

    if len(bins) < 2:
        return []

    vol_profile = np.zeros(len(bins) - 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for p, v in zip(prices, volumes):
        idx = int((p - price_min) / bin_size)
        idx = min(idx, len(vol_profile) - 1)
        idx = max(idx, 0)
        vol_profile[idx] += v

    # Top-N by volume
    if len(vol_profile) == 0:
        return []
    top_indices = np.argsort(vol_profile)[-top_n:]
    hvn_levels = [(bin_centers[i], vol_profile[i]) for i in sorted(top_indices) if vol_profile[i] > 0]
    return hvn_levels

# Precompute HVN levels for every bar (trailing, no lookahead)
print("  Computing trailing volume profiles for every bar...")
hvn_per_bar = []
for i in range(len(spread_df)):
    hvn = compute_volume_profile_hvn(spread_df, i, VP_LOOKBACK, VP_BIN_SIZE, VP_TOP_N)
    hvn_per_bar.append(hvn)

# Show sample HVN levels at a few key dates
sample_dates = [
    spread_df[spread_df["date"] < SEL_END].index[-1],  # last sel bar
    len(spread_df) // 2,
]
for idx in sample_dates:
    d = spread_df["date"].iloc[idx].date()
    levels = hvn_per_bar[idx]
    level_str = ", ".join([f"${lv:.2f} ({vol/1e6:.1f}M)" for lv, vol in levels])
    print(f"  HVN at {d}: {level_str}")


# ══════════════════════════════════════════════════════════════════
# 3. S/R LEVEL AGGREGATION (for each bar, trailing only)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 3: AGGREGATE S/R LEVELS PER BAR")
print("=" * 70)

def get_sr_levels_at_bar(bar_idx, spread_df, swing_data, hvn_per_bar, swing_window=10):
    """Return dict with 'support' and 'resistance' price levels using only trailing data.

    Support: swing lows below current price + HVN below
    Resistance: swing highs above current price + HVN above
    Only swing points at least swing_window bars before bar_idx are used (to confirm the swing).
    """
    current_price = spread_df["close"].iloc[bar_idx]
    current_date = spread_df["date"].iloc[bar_idx]

    supports = []
    resistances = []

    # Swing levels (use the N=10 window as primary, require confirmation)
    for w in SWING_WINDOWS:
        # Swing highs confirmed by bar_idx - w (fully in the past)
        confirmed_cutoff = bar_idx - w
        if confirmed_cutoff < 0:
            continue

        sh = swing_data[w]["highs"]
        sh_past = sh[sh.index <= confirmed_cutoff]
        for _, row in sh_past.iterrows():
            lv = row["level"]
            if lv > current_price:
                resistances.append(lv)
            else:
                supports.append(lv)

        sl = swing_data[w]["lows"]
        sl_past = sl[sl.index <= confirmed_cutoff]
        for _, row in sl_past.iterrows():
            lv = row["level"]
            if lv < current_price:
                supports.append(lv)
            else:
                resistances.append(lv)

    # HVN levels
    for lv, vol in hvn_per_bar[bar_idx]:
        if lv < current_price:
            supports.append(lv)
        elif lv > current_price:
            resistances.append(lv)
        else:
            # At current price: both
            supports.append(lv)
            resistances.append(lv)

    # Deduplicate and sort
    supports = sorted(set(supports), reverse=True)   # closest first
    resistances = sorted(set(resistances))             # closest first

    return {"support": supports, "resistance": resistances}


# Precompute S/R for every bar
print("  Precomputing S/R levels for every bar...")
sr_per_bar = []
for i in range(len(spread_df)):
    sr = get_sr_levels_at_bar(i, spread_df, swing_data, hvn_per_bar)
    sr_per_bar.append(sr)

# Count how many bars have S/R levels
has_support = sum(1 for sr in sr_per_bar if len(sr["support"]) > 0)
has_resist = sum(1 for sr in sr_per_bar if len(sr["resistance"]) > 0)
print(f"  Bars with support levels: {has_support}/{len(spread_df)}")
print(f"  Bars with resistance levels: {has_resist}/{len(spread_df)}")


# ══════════════════════════════════════════════════════════════════
# 4. BASELINE S5_zexit (ungated)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 4: BASELINE S5_ZEXIT (UNGATED)")
print("=" * 70)

def sig_fn_baseline(row, i, df):
    return s5_zexit_signal(row, i, df, k_entry=K_ENTRY)

def exit_fn_baseline(sig, row, j, df):
    return s5_zexit_exit(sig, row, j, df, z_exit_thresh=Z_EXIT)

trades_baseline = run_backtest(spread_df, sig_fn_baseline, exit_fn_baseline,
                                max_hold=MAX_HOLD)

sel_baseline = trades_baseline[trades_baseline["entry_date"] < SEL_END]
oos_baseline = trades_baseline[trades_baseline["entry_date"] >= SEL_END]

stats_sel_base = compute_stats(sel_baseline, label="Baseline SEL")
stats_oos_base = compute_stats(oos_baseline, label="Baseline OOS")

print(f"  Selection: n={stats_sel_base['n']}, avg_pnl={stats_sel_base['avg_pnl']:+.3f}%, "
      f"sharpe={stats_sel_base['sharpe']}, t={stats_sel_base['t_stat']:.2f}")
print(f"  OOS 2026:  n={stats_oos_base['n']}, avg_pnl={stats_oos_base['avg_pnl']:+.3f}%, "
      f"sharpe={stats_oos_base['sharpe']}, t={stats_oos_base['t_stat']:.2f}")


# ══════════════════════════════════════════════════════════════════
# 5. S/R GATED S5_ZEXIT (binary gate)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 5: S/R GATED S5_ZEXIT")
print("=" * 70)

def is_near_support(bar_idx, spread_df, sr_per_bar, atr_mult=ATR_PROXIMITY):
    """Check if close is within atr_mult * ATR of a support level."""
    price = spread_df["close"].iloc[bar_idx]
    atr = spread_df["atr20"].iloc[bar_idx]
    if pd.isna(atr) or atr <= 0:
        return False
    for lv in sr_per_bar[bar_idx]["support"]:
        if abs(price - lv) <= atr_mult * atr:
            return True
    return False

def is_near_resistance(bar_idx, spread_df, sr_per_bar, atr_mult=ATR_PROXIMITY):
    """Check if close is within atr_mult * ATR of a resistance level."""
    price = spread_df["close"].iloc[bar_idx]
    atr = spread_df["atr20"].iloc[bar_idx]
    if pd.isna(atr) or atr <= 0:
        return False
    for lv in sr_per_bar[bar_idx]["resistance"]:
        if abs(price - lv) <= atr_mult * atr:
            return True
    return False

def sig_fn_sr_gated(row, i, df):
    """Only take LONG signals near support, SHORT signals near resistance."""
    base = s5_zexit_signal(row, i, df, k_entry=K_ENTRY)
    if base is None:
        return None

    if base["side"] == "long" and is_near_support(i, df, sr_per_bar):
        return base
    elif base["side"] == "short" and is_near_resistance(i, df, sr_per_bar):
        return base
    return None

trades_gated = run_backtest(spread_df, sig_fn_sr_gated, exit_fn_baseline,
                             max_hold=MAX_HOLD)

sel_gated = trades_gated[trades_gated["entry_date"] < SEL_END]
oos_gated = trades_gated[trades_gated["entry_date"] >= SEL_END]

stats_sel_gated = compute_stats(sel_gated, label="SR-Gated SEL")
stats_oos_gated = compute_stats(oos_gated, label="SR-Gated OOS")

print(f"  Gated Selection: n={stats_sel_gated['n']}, avg_pnl={stats_sel_gated['avg_pnl']:+.3f}%, "
      f"sharpe={stats_sel_gated['sharpe']}, t={stats_sel_gated['t_stat']:.2f}")
print(f"  Gated OOS 2026:  n={stats_oos_gated['n']}, avg_pnl={stats_oos_gated['avg_pnl']:+.3f}%, "
      f"sharpe={stats_oos_gated['sharpe']}, t={stats_oos_gated['t_stat']:.2f}")

# Which trades were filtered out?
if not trades_baseline.empty and not trades_gated.empty:
    baseline_entries = set(trades_baseline["entry_date"].tolist())
    gated_entries = set(trades_gated["entry_date"].tolist())
    filtered_out = baseline_entries - gated_entries
    print(f"\n  Trades filtered by S/R gate: {len(filtered_out)}")
    if filtered_out:
        filtered_trades = trades_baseline[trades_baseline["entry_date"].isin(filtered_out)]
        filtered_avg = filtered_trades["pnl_net_pct"].mean()
        print(f"  Avg PnL of filtered trades: {filtered_avg:+.3f}%")
        print(f"  (Positive = gate removed winners; Negative = gate removed losers)")


# ══════════════════════════════════════════════════════════════════
# 6. PROXIMITY-WEIGHTED SIGNAL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 6: PROXIMITY-WEIGHTED SIGNAL (alpha sweep)")
print("=" * 70)

def compute_proximity_score(bar_idx, side, spread_df, sr_per_bar):
    """Compute proximity score [0, 1] based on distance to nearest relevant S/R level.
    Score = 1 when right at the level, decays linearly to 0 at 2*ATR distance.
    """
    price = spread_df["close"].iloc[bar_idx]
    atr = spread_df["atr20"].iloc[bar_idx]
    if pd.isna(atr) or atr <= 0:
        return 0.0

    levels = sr_per_bar[bar_idx]["support"] if side == "long" else sr_per_bar[bar_idx]["resistance"]
    if not levels:
        return 0.0

    # Distance to nearest relevant level
    distances = [abs(price - lv) for lv in levels]
    min_dist = min(distances)

    # Linear decay: 1 at level, 0 at 2*ATR
    max_dist = 2.0 * atr
    score = max(0.0, 1.0 - min_dist / max_dist)
    return score

# Precompute proximity scores for all bars
proximity_long = np.zeros(len(spread_df))
proximity_short = np.zeros(len(spread_df))
for i in range(len(spread_df)):
    proximity_long[i] = compute_proximity_score(i, "long", spread_df, sr_per_bar)
    proximity_short[i] = compute_proximity_score(i, "short", spread_df, sr_per_bar)

# Store in spread_df for access during backtest
spread_df["prox_long"] = proximity_long
spread_df["prox_short"] = proximity_short

alpha_results = {}
for alpha in ALPHAS:
    def make_sig_fn(a):
        def sig_fn_prox(row, i, df):
            """Proximity-weighted entry: effective threshold varies by S/R distance.

            Without S/R proximity (prox=0), threshold is raised to K_ENTRY * (1 + alpha*0.5),
            requiring stronger z to enter. Near S/R (prox=1), threshold drops to
            K_ENTRY * (1 + alpha*0.5) / (1 + alpha*1.0) = K_ENTRY * (1+0.5a)/(1+a).

            This creates a differential: signals near S/R can enter with lower z,
            while signals far from S/R need higher z.
            """
            z = row.get("spread_z", np.nan)
            if pd.isna(z):
                return None
            atr = row.get("atr20", np.nan)
            if pd.isna(atr) or atr <= 0:
                return None
            price = row["close"]
            abs_z = abs(z)

            # Determine side from z direction
            if z > 0:
                side = "short"
                prox = row.get("prox_short", 0.0)
            elif z < 0:
                side = "long"
                prox = row.get("prox_long", 0.0)
            else:
                return None

            # Adjusted threshold: higher baseline, reduced by proximity
            k_raised = K_ENTRY * (1 + a * 0.5)   # raised baseline
            k_effective = k_raised / (1 + a * prox)  # proximity discount

            if abs_z < k_effective:
                return None

            if side == "short":
                return {"side": "short", "stop": price + 1.5 * atr, "target": None}
            else:
                return {"side": "long", "stop": price - 1.5 * atr, "target": None}
        return sig_fn_prox

    sig_fn_prox = make_sig_fn(alpha)
    trades_prox = run_backtest(spread_df, sig_fn_prox, exit_fn_baseline, max_hold=MAX_HOLD)

    sel_prox = trades_prox[trades_prox["entry_date"] < SEL_END]
    oos_prox = trades_prox[trades_prox["entry_date"] >= SEL_END]

    stats_sel_p = compute_stats(sel_prox, label=f"Prox alpha={alpha} SEL")
    stats_oos_p = compute_stats(oos_prox, label=f"Prox alpha={alpha} OOS")

    alpha_results[alpha] = {
        "trades_all": trades_prox,
        "trades_sel": sel_prox,
        "trades_oos": oos_prox,
        "stats_sel": stats_sel_p,
        "stats_oos": stats_oos_p,
    }

    print(f"  alpha={alpha:.1f}  SEL: n={stats_sel_p['n']:>3d}, avg={stats_sel_p['avg_pnl']:+.3f}%, "
          f"sharpe={stats_sel_p['sharpe']:>5.2f}  |  "
          f"OOS: n={stats_oos_p['n']:>3d}, avg={stats_oos_p['avg_pnl']:+.3f}%, "
          f"sharpe={stats_oos_p['sharpe']:>5.2f}")


# ══════════════════════════════════════════════════════════════════
# 7. PERMUTATION TEST
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 7: PERMUTATION TEST (1000 shuffles)")
print("=" * 70)

def permutation_test_sr_gate(trades_baseline, trades_gated, n_perms, seed=SEED):
    """Permutation test: is the S/R gating providing statistically significant lift?

    Null hypothesis: S/R alignment is irrelevant. The avg PnL of any random subset
    of baseline trades (same size as gated) would be the same.

    Shuffle: randomly select len(gated) trades from baseline, compute avg PnL.
    p-value: fraction of permuted avg PnLs >= real gated avg PnL.
    """
    if trades_gated.empty or trades_baseline.empty:
        return 1.0, np.array([])

    real_avg = trades_gated["pnl_net_pct"].mean()
    n_gated = len(trades_gated)
    n_base = len(trades_baseline)

    if n_gated >= n_base:
        return 1.0, np.array([])

    rng = np.random.RandomState(seed)
    perm_avgs = np.zeros(n_perms)
    base_pnls = trades_baseline["pnl_net_pct"].values

    for p in range(n_perms):
        subset = rng.choice(base_pnls, size=n_gated, replace=False)
        perm_avgs[p] = np.mean(subset)

    p_value = np.mean(perm_avgs >= real_avg)
    return p_value, perm_avgs

# Test S/R gate (full sample: selection period)
print("\n  S/R Gate permutation (Selection period):")
p_gate_sel, perm_gate_sel = permutation_test_sr_gate(sel_baseline, sel_gated, N_PERMUTATIONS)
if not sel_gated.empty:
    print(f"    Real gated avg PnL: {sel_gated['pnl_net_pct'].mean():+.3f}%")
    print(f"    Null mean:          {perm_gate_sel.mean():+.3f}%")
    print(f"    p-value:            {p_gate_sel:.4f}  "
          f"{'SIGNIFICANT' if p_gate_sel < 0.05 else 'NOT SIGNIFICANT'}")
else:
    print(f"    No gated trades in selection period.")

# Test S/R gate (OOS)
print("\n  S/R Gate permutation (OOS 2026):")
p_gate_oos, perm_gate_oos = permutation_test_sr_gate(oos_baseline, oos_gated, N_PERMUTATIONS)
if not oos_gated.empty:
    print(f"    Real gated avg PnL: {oos_gated['pnl_net_pct'].mean():+.3f}%")
    print(f"    Null mean:          {perm_gate_oos.mean():+.3f}%")
    print(f"    p-value:            {p_gate_oos:.4f}  "
          f"{'SIGNIFICANT' if p_gate_oos < 0.05 else 'NOT SIGNIFICANT'}")
else:
    print(f"    No gated trades in OOS period.")

# Test each alpha variant (OOS)
alpha_pvals = {}
alpha_perm_dists = {}
for alpha in ALPHAS:
    ar = alpha_results[alpha]
    p_val, perm_dist = permutation_test_sr_gate(
        oos_baseline, ar["trades_oos"], N_PERMUTATIONS, seed=SEED + int(alpha * 100)
    )
    alpha_pvals[alpha] = p_val
    alpha_perm_dists[alpha] = perm_dist
    real_avg_a = ar["trades_oos"]["pnl_net_pct"].mean() if not ar["trades_oos"].empty else 0
    print(f"\n  Proximity alpha={alpha:.1f} permutation (OOS):")
    print(f"    Real avg PnL: {real_avg_a:+.3f}%, p-value: {p_val:.4f}  "
          f"{'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'}")


# ══════════════════════════════════════════════════════════════════
# 8. LOG TRIALS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 8: LOG TRIALS")
print("=" * 70)

log_trial("agent4", "S5_zexit_baseline",
          {"k": K_ENTRY, "ze": Z_EXIT, "max_hold": MAX_HOLD},
          stats_oos_base)

log_trial("agent4", "S5_zexit_SR_gated",
          {"k": K_ENTRY, "ze": Z_EXIT, "max_hold": MAX_HOLD,
           "atr_proximity": ATR_PROXIMITY, "swing_windows": SWING_WINDOWS,
           "vp_lookback": VP_LOOKBACK, "p_value_sel": round(p_gate_sel, 4),
           "p_value_oos": round(p_gate_oos, 4)},
          stats_oos_gated)

for alpha in ALPHAS:
    ar = alpha_results[alpha]
    log_trial("agent4", f"S5_zexit_prox_alpha{alpha}",
              {"k": K_ENTRY, "ze": Z_EXIT, "max_hold": MAX_HOLD,
               "alpha": alpha, "p_value_oos": round(alpha_pvals[alpha], 4)},
              ar["stats_oos"])

print("  Logged 5 trials to shared trials_log.")


# ══════════════════════════════════════════════════════════════════
# 9. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SUMMARY: GATED VS UNGATED COMPARISON")
print("=" * 70)

rows = [
    ("Baseline (ungated)", stats_sel_base, stats_oos_base, "-"),
    ("S/R Gate (binary)", stats_sel_gated, stats_oos_gated, f"{p_gate_oos:.4f}"),
]
for alpha in ALPHAS:
    ar = alpha_results[alpha]
    rows.append((f"Prox alpha={alpha}", ar["stats_sel"], ar["stats_oos"],
                 f"{alpha_pvals[alpha]:.4f}"))

print(f"\n  {'Variant':<25s} | {'SEL n':>5s} {'SEL avg':>8s} {'SEL SR':>7s} | "
      f"{'OOS n':>5s} {'OOS avg':>8s} {'OOS SR':>7s} | {'p-val':>7s}")
print("  " + "-" * 85)
for name, sel_s, oos_s, pv in rows:
    print(f"  {name:<25s} | {sel_s['n']:>5d} {sel_s['avg_pnl']:>+7.3f}% {sel_s['sharpe']:>6.2f} | "
          f"{oos_s['n']:>5d} {oos_s['avg_pnl']:>+7.3f}% {oos_s['sharpe']:>6.2f} | {pv:>7s}")


# ══════════════════════════════════════════════════════════════════
# 10. MATPLOTLIB FIGURE (4 panels)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 10: GENERATING FIGURE")
print("=" * 70)

fig = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# ── Panel A: RXRX price with S/R levels ─────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

# Plot price
ax_a.plot(spread_df["date"], spread_df["close"], color="#333333", linewidth=0.8,
          label="RXRX Close", zorder=2)

# Overlay swing highs/lows (N=10 as primary, using selection period only for display)
sel_mask = spread_df["date"] < SEL_END
sh10 = swing_data[10]["highs"]
sl10 = swing_data[10]["lows"]
sh_sel = sh10[sh10.index.isin(spread_df[sel_mask].index)]
sl_sel = sl10[sl10.index.isin(spread_df[sel_mask].index)]

ax_a.scatter(spread_df.loc[sh_sel.index, "date"], sh_sel["level"],
             marker="v", color="#D32F2F", s=18, alpha=0.7, label="Swing High (R)", zorder=3)
ax_a.scatter(spread_df.loc[sl_sel.index, "date"], sl_sel["level"],
             marker="^", color="#388E3C", s=18, alpha=0.7, label="Swing Low (S)", zorder=3)

# Show HVN levels at a few dates as horizontal bands
# Snapshot at end of selection period
snap_idx = spread_df[sel_mask].index[-1]
snap_hvn = hvn_per_bar[snap_idx]
for lv, vol in snap_hvn:
    ax_a.axhline(y=lv, color="#1565C0", linewidth=1.2, linestyle="--", alpha=0.5)
    ax_a.text(spread_df["date"].iloc[-1], lv, f"HVN ${lv:.1f}", fontsize=7,
              color="#1565C0", ha="right", va="bottom")

# Mark SEL/OOS split
oos_start_date = pd.Timestamp(SEL_END)
ax_a.axvline(x=oos_start_date, color="#FF6F00", linestyle=":", linewidth=1.5, label="OOS Start")
ax_a.set_title("A) RXRX Price with S/R Levels", fontsize=11, fontweight="bold")
ax_a.set_ylabel("Price ($)")
ax_a.legend(fontsize=7, loc="upper left")
ax_a.grid(True, alpha=0.3)

# ── Panel B: Volume Profile (horizontal histogram) ──────────────
ax_b = fig.add_subplot(gs[0, 1])

# Compute volume profile on full selection period
sel_data = spread_df[sel_mask]
prices_sel = sel_data["close"].values
volumes_sel = sel_data["volume"].values
price_min = np.floor(prices_sel.min() / VP_BIN_SIZE) * VP_BIN_SIZE
price_max = np.ceil(prices_sel.max() / VP_BIN_SIZE) * VP_BIN_SIZE
bins = np.arange(price_min, price_max + VP_BIN_SIZE, VP_BIN_SIZE)

vol_profile = np.zeros(len(bins) - 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

for p, v in zip(prices_sel, volumes_sel):
    idx = int((p - price_min) / VP_BIN_SIZE)
    idx = min(idx, len(vol_profile) - 1)
    idx = max(idx, 0)
    vol_profile[idx] += v

# Horizontal bar chart
colors = ["#1565C0" if v < np.percentile(vol_profile[vol_profile > 0], 80) else "#D32F2F"
          for v in vol_profile]
ax_b.barh(bin_centers, vol_profile / 1e6, height=VP_BIN_SIZE * 0.9, color=colors, alpha=0.7)
ax_b.set_xlabel("Volume (millions)")
ax_b.set_ylabel("Price ($)")
ax_b.set_title("B) Volume Profile (Selection Period)", fontsize=11, fontweight="bold")

# Mark HVN levels
for lv, vol in snap_hvn:
    ax_b.axhline(y=lv, color="#FF6F00", linewidth=1.5, linestyle="-", alpha=0.8)
    ax_b.text(vol_profile.max() / 1e6 * 0.95, lv + 0.3, f"HVN ${lv:.1f}",
              fontsize=8, color="#FF6F00", ha="right", fontweight="bold")
ax_b.grid(True, alpha=0.3)

# ── Panel C: PnL comparison bar chart ───────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

labels_c = ["Baseline", "S/R Gate"]
for alpha in ALPHAS:
    labels_c.append(f"Prox a={alpha}")

sel_pnls = [stats_sel_base["avg_pnl"], stats_sel_gated["avg_pnl"]]
oos_pnls = [stats_oos_base["avg_pnl"], stats_oos_gated["avg_pnl"]]
for alpha in ALPHAS:
    sel_pnls.append(alpha_results[alpha]["stats_sel"]["avg_pnl"])
    oos_pnls.append(alpha_results[alpha]["stats_oos"]["avg_pnl"])

x = np.arange(len(labels_c))
width = 0.35

bars_sel = ax_c.bar(x - width / 2, sel_pnls, width, label="Selection", color="#1565C0", alpha=0.8)
bars_oos = ax_c.bar(x + width / 2, oos_pnls, width, label="OOS 2026", color="#D32F2F", alpha=0.8)

ax_c.set_xticks(x)
ax_c.set_xticklabels(labels_c, rotation=30, ha="right", fontsize=9)
ax_c.set_ylabel("Avg PnL (%)")
ax_c.set_title("C) Avg PnL: Gated vs Ungated", fontsize=11, fontweight="bold")
ax_c.axhline(y=0, color="black", linewidth=0.8)
ax_c.legend(fontsize=9)
ax_c.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar in bars_sel:
    h = bar.get_height()
    ax_c.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
              f"{h:+.2f}", ha="center", va="bottom", fontsize=7)
for bar in bars_oos:
    h = bar.get_height()
    ax_c.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
              f"{h:+.2f}", ha="center", va="bottom", fontsize=7)

# ── Panel D: Permutation test distribution ───────────────────────
ax_d = fig.add_subplot(gs[1, 1])

# Plot the S/R gate permutation distribution (OOS)
if len(perm_gate_oos) > 0 and not oos_gated.empty:
    ax_d.hist(perm_gate_oos, bins=50, color="#90CAF9", edgecolor="#1565C0",
              alpha=0.7, label="Null distribution")
    real_val = oos_gated["pnl_net_pct"].mean()
    ax_d.axvline(x=real_val, color="#D32F2F", linewidth=2.5, linestyle="-",
                 label=f"S/R Gate avg = {real_val:+.3f}%")
    ax_d.axvline(x=np.percentile(perm_gate_oos, 95), color="#FF6F00",
                 linewidth=1.5, linestyle="--", label="95th percentile")
    ax_d.set_xlabel("Avg PnL (%)")
    ax_d.set_ylabel("Count")
    ax_d.set_title(f"D) Permutation Test: S/R Gate OOS (p={p_gate_oos:.4f})",
                   fontsize=11, fontweight="bold")
    ax_d.legend(fontsize=8)
elif len(perm_gate_sel) > 0 and not sel_gated.empty:
    # Fallback: show selection period permutation
    ax_d.hist(perm_gate_sel, bins=50, color="#90CAF9", edgecolor="#1565C0",
              alpha=0.7, label="Null distribution")
    real_val = sel_gated["pnl_net_pct"].mean()
    ax_d.axvline(x=real_val, color="#D32F2F", linewidth=2.5, linestyle="-",
                 label=f"S/R Gate avg = {real_val:+.3f}%")
    ax_d.axvline(x=np.percentile(perm_gate_sel, 95), color="#FF6F00",
                 linewidth=1.5, linestyle="--", label="95th percentile")
    ax_d.set_xlabel("Avg PnL (%)")
    ax_d.set_ylabel("Count")
    ax_d.set_title(f"D) Permutation Test: S/R Gate SEL (p={p_gate_sel:.4f})",
                   fontsize=11, fontweight="bold")
    ax_d.legend(fontsize=8)
else:
    ax_d.text(0.5, 0.5, "No gated trades for\npermutation test",
              ha="center", va="center", transform=ax_d.transAxes, fontsize=14)
    ax_d.set_title("D) Permutation Test (N/A)", fontsize=11, fontweight="bold")

ax_d.grid(True, alpha=0.3)

fig.suptitle("Agent 4: Support/Resistance Gate for S5_zexit (RXRX)",
             fontsize=14, fontweight="bold", y=0.98)

out_path = OUT_DIR / "agent4_sr_levels.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Figure saved: {out_path}")


# ══════════════════════════════════════════════════════════════════
# 11. S/R LEVELS TABLE (sampled windows)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  S/R LEVELS TABLE (sampled windows)")
print("=" * 70)

# Show S/R at quarterly intervals in selection period
sel_dates = spread_df[sel_mask]["date"]
quarterly_indices = sel_dates.iloc[::63].index.tolist()  # ~quarterly
if sel_dates.index[-1] not in quarterly_indices:
    quarterly_indices.append(sel_dates.index[-1])

print(f"\n  {'Date':>12s} | {'Price':>7s} | {'Top Supports':>35s} | {'Top Resistances':>35s} | {'HVN Levels':>30s}")
print("  " + "-" * 130)
for idx in quarterly_indices[-8:]:  # last 8 snapshots
    d = spread_df["date"].iloc[idx].date()
    p = spread_df["close"].iloc[idx]
    sr = sr_per_bar[idx]
    top_s = sr["support"][:3]
    top_r = sr["resistance"][:3]
    hvn = hvn_per_bar[idx]
    s_str = ", ".join([f"${lv:.2f}" for lv in top_s]) if top_s else "none"
    r_str = ", ".join([f"${lv:.2f}" for lv in top_r]) if top_r else "none"
    h_str = ", ".join([f"${lv:.1f}" for lv, _ in hvn]) if hvn else "none"
    print(f"  {str(d):>12s} | ${p:>6.2f} | {s_str:>35s} | {r_str:>35s} | {h_str:>30s}")


# ══════════════════════════════════════════════════════════════════
# 12. CONCLUSION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  CONCLUSION")
print("=" * 70)

# Determine if S/R gating helps
gate_improves_sel = stats_sel_gated["avg_pnl"] > stats_sel_base["avg_pnl"]
gate_improves_oos = stats_oos_gated["avg_pnl"] > stats_oos_base["avg_pnl"]
gate_significant = p_gate_oos < 0.05 if not oos_gated.empty else False

# Best proximity alpha
best_alpha = None
best_alpha_oos_avg = stats_oos_base["avg_pnl"]
for alpha in ALPHAS:
    ar = alpha_results[alpha]
    if ar["stats_oos"]["avg_pnl"] > best_alpha_oos_avg and alpha_pvals[alpha] < 0.10:
        best_alpha = alpha
        best_alpha_oos_avg = ar["stats_oos"]["avg_pnl"]

print(f"""
  S/R BINARY GATE:
    Selection: {'IMPROVES' if gate_improves_sel else 'DEGRADES'} avg PnL
      Baseline: {stats_sel_base['avg_pnl']:+.3f}% (n={stats_sel_base['n']})
      Gated:    {stats_sel_gated['avg_pnl']:+.3f}% (n={stats_sel_gated['n']})
    OOS 2026:  {'IMPROVES' if gate_improves_oos else 'DEGRADES'} avg PnL
      Baseline: {stats_oos_base['avg_pnl']:+.3f}% (n={stats_oos_base['n']})
      Gated:    {stats_oos_gated['avg_pnl']:+.3f}% (n={stats_oos_gated['n']})
    Permutation p-value (OOS): {p_gate_oos:.4f} — {'SIGNIFICANT' if gate_significant else 'NOT SIGNIFICANT'}

  PROXIMITY WEIGHTING:
    Best alpha (OOS, p<0.10): {f'alpha={best_alpha}' if best_alpha else 'NONE — no alpha improves OOS'}

  VERDICT: {'S/R gating provides statistically significant improvement to S5_zexit.' if gate_significant else 'S/R gating does NOT provide statistically significant improvement to S5_zexit. The z-score signal alone captures the mean-reversion edge; price-level alignment adds noise or reduces trade count without meaningful lift.'}
""")

print("  AGENT 4 COMPLETE")
