"""
V3 Scanner Optimization — Experiments E3, E6, E10
==================================================
E3: Earnings Exclusion + Catalyst Type Breakdown
E6: n_runs Position Sizing
E10: Magnitude Position Sizing (expected_move_pct and gap_pct)

Data: results/v2_features_full.csv (316 trades)
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/home/admin_yaninaso_altostrat_com/bot-RAG/sentiment/scanner_optimization")
DATA = BASE / "results" / "v2_features_full.csv"
OUT_JSON = BASE / "results" / "v3_e3_e6_e10_results.json"

PNL_COLS = {
    "30m": "pnl_30m_nosl",
    "60m": "pnl_60m_nosl",
    "EOD": "pnl_eod_nosl",
    "T+1": "pnl_t1_close",
    "T+2": "pnl_t2_close",
}

LOW_N = 20  # flag slices below this


# ── helpers ───────────────────────────────────────────────────────────────────
def tmean(arr, prop=0.05):
    """Trimmed mean (5-95%) — drops NaN first."""
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 3:
        return np.nan
    return stats.trim_mean(a, prop)


def safe_sharpe(arr):
    """Mean / std, annualised by sqrt(252). Returns NaN if std==0 or n<5."""
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 5 or np.std(a, ddof=1) == 0:
        return np.nan
    return (np.mean(a) / np.std(a, ddof=1)) * np.sqrt(252)


def pnl_stats(series, label=""):
    """Return a dict of standard PnL stats for a series."""
    s = series.dropna()
    n = len(s)
    flag = " [LOW-N]" if n < LOW_N else ""
    return {
        "n": n,
        "flag": flag,
        "mean": round(s.mean(), 4) if n else np.nan,
        "trimmed_mean": round(tmean(s.values), 4) if n >= 3 else np.nan,
        "median": round(s.median(), 4) if n else np.nan,
        "std": round(s.std(), 4) if n > 1 else np.nan,
        "win_rate": round((s > 0).mean() * 100, 1) if n else np.nan,
        "sharpe": round(safe_sharpe(s.values), 2) if n >= 5 else np.nan,
    }


def fmt_row(name, d, width=22):
    """Format one row for a text table."""
    flag = d.get("flag", "")
    return (
        f"  {name:<{width}} n={d['n']:>4}{flag:>8}  "
        f"mean={d['mean']:>+8.3f}%  trim={d['trimmed_mean']:>+8.3f}%  "
        f"med={d['median']:>+8.3f}%  std={d['std']:>7.3f}  "
        f"WR={d['win_rate']:>5.1f}%  Sharpe={d['sharpe']:>6.2f}"
    )


# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"Loaded {len(df)} trades from {DATA.name}")
print(f"  LONG: {(df.sign == 1).sum()}  SHORT: {(df.sign == -1).sum()}")
print(f"  PnL 60m valid: {df.pnl_60m_nosl.notna().sum()}")
print()

results = {}  # will be dumped to JSON at the end


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E3: EARNINGS EXCLUSION + CATALYST TYPE BREAKDOWN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 90)
print("E3: EARNINGS EXCLUSION + CATALYST TYPE BREAKDOWN")
print("=" * 90)

# ── 1. Catalyst type stats ────────────────────────────────────────────────────
print("\n── 1. Catalyst type performance (all directions, sorted by 60m mean) ──")
cat_stats = {}
for cat, grp in df.groupby("catalyst_type"):
    row = {"catalyst_type": cat}
    for hz_label, col in PNL_COLS.items():
        s = grp[col].dropna()
        row[f"n_{hz_label}"] = len(s)
        row[f"mean_{hz_label}"] = round(s.mean(), 4) if len(s) else np.nan
    row.update(pnl_stats(grp["pnl_60m_nosl"], cat))
    cat_stats[cat] = row

# Sort by 60m mean descending
sorted_cats = sorted(cat_stats.values(), key=lambda x: x.get("mean_60m", -999) if not np.isnan(x.get("mean_60m", np.nan)) else -999, reverse=True)

header = f"  {'catalyst_type':<18} {'n':>4}  {'30m':>8}  {'60m':>8}  {'EOD':>8}  {'T+1':>8}  {'T+2':>8}  {'trim60':>8}  {'WR60':>6}  {'med60':>8}"
print(header)
print("  " + "-" * len(header))
for r in sorted_cats:
    flag = " *" if r["n"] < LOW_N else ""
    print(
        f"  {r['catalyst_type']:<18} {r['n']:>4}{flag:>2}"
        f"  {r.get('mean_30m', np.nan):>+8.3f}"
        f"  {r.get('mean_60m', np.nan):>+8.3f}"
        f"  {r.get('mean_EOD', np.nan):>+8.3f}"
        f"  {r.get('mean_T+1', np.nan):>+8.3f}"
        f"  {r.get('mean_T+2', np.nan):>+8.3f}"
        f"  {r['trimmed_mean']:>+8.3f}"
        f"  {r['win_rate']:>5.1f}%"
        f"  {r['median']:>+8.3f}"
    )

# ── 2. LONG vs SHORT within each catalyst ────────────────────────────────────
print("\n── 2. Catalyst × direction breakdown (60m PnL) ──")
dir_cat = {}
for (cat, sign_val), grp in df.groupby(["catalyst_type", "sign"]):
    direction = "LONG" if sign_val == 1 else "SHORT"
    s = grp["pnl_60m_nosl"].dropna()
    key = f"{cat}|{direction}"
    d = pnl_stats(s, key)
    dir_cat[key] = d
    print(fmt_row(f"{cat} ({direction})", d, width=30))

# ── 3. Earnings sub-analysis ─────────────────────────────────────────────────
print("\n── 3. Earnings sub-analysis ──")
earn = df[df.is_earnings == True]
non_earn = df[df.is_earnings == False]

earn_results = {}
for label, subset in [("ALL earnings", earn), ("ALL non-earnings", non_earn)]:
    row = {}
    for hz_label, col in PNL_COLS.items():
        s = subset[col].dropna()
        row[hz_label] = {"n": len(s), "mean": round(s.mean(), 4) if len(s) else np.nan,
                         "trimmed_mean": round(tmean(s.values), 4) if len(s) >= 3 else np.nan}
    earn_results[label] = row
    print(f"  {label}:")
    for hz in ["30m", "60m", "EOD", "T+1", "T+2"]:
        r = row[hz]
        flag = " [LOW-N]" if r["n"] < LOW_N else ""
        print(f"    {hz}: n={r['n']:>4}{flag}  mean={r['mean']:>+8.3f}%  trim={r['trimmed_mean']:>+8.3f}%")

# Earnings by direction
print("\n  Earnings by direction:")
for sign_val, dir_label in [(1, "LONG"), (-1, "SHORT")]:
    sub = earn[earn.sign == sign_val]
    s60 = sub["pnl_60m_nosl"].dropna()
    d = pnl_stats(s60, f"earnings {dir_label}")
    earn_results[f"earnings_{dir_label}"] = d
    print(fmt_row(f"  Earnings {dir_label}", d, width=22))

    sub_ne = non_earn[non_earn.sign == sign_val]
    s60_ne = sub_ne["pnl_60m_nosl"].dropna()
    d_ne = pnl_stats(s60_ne, f"non-earnings {dir_label}")
    earn_results[f"non_earnings_{dir_label}"] = d_ne
    print(fmt_row(f"  Non-earn {dir_label}", d_ne, width=22))

# PnL improvement from excluding earnings
print("\n  PnL improvement from excluding earnings trades:")
for hz_label, col in PNL_COLS.items():
    all_mean = df[col].dropna().mean()
    ne_mean = non_earn[col].dropna().mean()
    n_all = df[col].notna().sum()
    n_ne = non_earn[col].notna().sum()
    delta = ne_mean - all_mean
    print(f"    {hz_label}: all={all_mean:>+7.3f}% (n={n_all})  excl-earn={ne_mean:>+7.3f}% (n={n_ne})  delta={delta:>+7.3f}%")

# ── 4. Rank catalyst types ───────────────────────────────────────────────────
print("\n── 4. Catalyst ranking (60m) ──")
good_cats = [r["catalyst_type"] for r in sorted_cats if r.get("mean_60m", -999) > 0 and not np.isnan(r.get("mean_60m", np.nan))]
bad_cats = [r["catalyst_type"] for r in sorted_cats if r.get("mean_60m", 1) <= 0 and not np.isnan(r.get("mean_60m", np.nan))]
print(f"  GOOD catalysts (avg 60m > 0): {good_cats}")
print(f"  BAD  catalysts (avg 60m <= 0): {bad_cats}")

# ── 5. Good-catalyst filter ──────────────────────────────────────────────────
print("\n── 5. Good-catalyst-only filter ──")
df_good = df[df.catalyst_type.isin(good_cats)]
for hz_label, col in PNL_COLS.items():
    s_all = df[col].dropna()
    s_good = df_good[col].dropna()
    s_ne = non_earn[col].dropna()
    print(
        f"  {hz_label}: ALL={s_all.mean():>+7.3f}% (n={len(s_all)})  "
        f"good-cat={s_good.mean():>+7.3f}% (n={len(s_good)})  "
        f"excl-earn={s_ne.mean():>+7.3f}% (n={len(s_ne)})"
    )

# ── E3 results for JSON ──────────────────────────────────────────────────────
results["E3"] = {
    "catalyst_ranking_60m": [
        {"catalyst_type": r["catalyst_type"], "n": r["n"],
         "mean_60m": r.get("mean_60m"), "trimmed_mean_60m": r.get("trimmed_mean"),
         "win_rate_60m": r.get("win_rate")}
        for r in sorted_cats
    ],
    "good_catalysts": good_cats,
    "bad_catalysts": bad_cats,
    "earnings_impact": {
        hz: {
            "all_mean": round(df[col].dropna().mean(), 4),
            "excl_earnings_mean": round(non_earn[col].dropna().mean(), 4),
            "delta": round(non_earn[col].dropna().mean() - df[col].dropna().mean(), 4),
        }
        for hz, col in PNL_COLS.items()
        if df[col].notna().sum() > 0
    },
}

# ── E3 charts ─────────────────────────────────────────────────────────────────
fig_e3, axes_e3 = plt.subplots(2, 2, figsize=(16, 14))
fig_e3.suptitle("E3: Earnings Exclusion + Catalyst Breakdown", fontsize=15, fontweight="bold")

# Chart 1: Horizontal bar — catalyst → avg 60m PnL
ax = axes_e3[0, 0]
cat_names = [r["catalyst_type"] for r in sorted_cats if not np.isnan(r.get("mean_60m", np.nan))]
cat_means = [r["mean_60m"] for r in sorted_cats if not np.isnan(r.get("mean_60m", np.nan))]
cat_ns = [r["n"] for r in sorted_cats if not np.isnan(r.get("mean_60m", np.nan))]
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in cat_means]
bars = ax.barh(range(len(cat_names)), cat_means, color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(cat_names)))
ax.set_yticklabels([f"{n} (n={c})" for n, c in zip(cat_names, cat_ns)], fontsize=8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Avg 60m PnL (%)")
ax.set_title("Catalyst Type: Avg 60m PnL")
ax.invert_yaxis()

# Chart 2: Catalyst × direction heatmap
ax = axes_e3[0, 1]
# Build pivot table
pivot_data = []
all_cats_sorted = [r["catalyst_type"] for r in sorted_cats]
for cat in all_cats_sorted:
    for sign_val, dir_label in [(1, "LONG"), (-1, "SHORT")]:
        sub = df[(df.catalyst_type == cat) & (df.sign == sign_val)]
        s = sub["pnl_60m_nosl"].dropna()
        pivot_data.append({"catalyst": cat, "direction": dir_label,
                           "pnl": s.mean() if len(s) > 0 else np.nan,
                           "n": len(s)})
pivot_df = pd.DataFrame(pivot_data)
pivot_mat = pivot_df.pivot(index="catalyst", columns="direction", values="pnl")
pivot_mat = pivot_mat.reindex(all_cats_sorted)
n_mat = pivot_df.pivot(index="catalyst", columns="direction", values="n")
n_mat = n_mat.reindex(all_cats_sorted)

im = ax.imshow(pivot_mat.values, cmap="RdYlGn", aspect="auto",
               vmin=-5, vmax=5)
ax.set_xticks([0, 1])
ax.set_xticklabels(pivot_mat.columns)
ax.set_yticks(range(len(pivot_mat)))
ax.set_yticklabels(pivot_mat.index, fontsize=7)
# Annotate with values
for i in range(len(pivot_mat)):
    for j in range(2):
        val = pivot_mat.values[i, j]
        n = n_mat.values[i, j]
        if not np.isnan(val):
            color = "white" if abs(val) > 3 else "black"
            ax.text(j, i, f"{val:+.1f}\nn={int(n)}", ha="center", va="center",
                    fontsize=6, color=color)
fig_e3.colorbar(im, ax=ax, shrink=0.6, label="Avg 60m PnL (%)")
ax.set_title("Catalyst x Direction Heatmap")

# Chart 3: Earnings vs non-earnings with error bars
ax = axes_e3[1, 0]
horizons_plot = ["30m", "60m", "EOD"]
x_pos = np.arange(len(horizons_plot))
width = 0.35

earn_means = []
earn_sems = []
ne_means = []
ne_sems = []
for hz in horizons_plot:
    col = PNL_COLS[hz]
    e_s = earn[col].dropna()
    n_s = non_earn[col].dropna()
    earn_means.append(e_s.mean() if len(e_s) else 0)
    earn_sems.append(e_s.sem() if len(e_s) > 1 else 0)
    ne_means.append(n_s.mean() if len(n_s) else 0)
    ne_sems.append(n_s.sem() if len(n_s) > 1 else 0)

ax.bar(x_pos - width / 2, earn_means, width, yerr=earn_sems, label=f"Earnings (n={len(earn)})",
       color="#e74c3c", alpha=0.8, capsize=4)
ax.bar(x_pos + width / 2, ne_means, width, yerr=ne_sems, label=f"Non-earnings (n={len(non_earn)})",
       color="#2ecc71", alpha=0.8, capsize=4)
ax.set_xticks(x_pos)
ax.set_xticklabels(horizons_plot)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Avg PnL (%)")
ax.set_title("Earnings vs Non-Earnings")
ax.legend(fontsize=8)

# Chart 4: Cumulative PnL — all vs good-cat vs earnings-excluded
ax = axes_e3[1, 1]
# Use EOD since it has 0 nulls for cleaner cumulative
for label, subset, color, ls in [
    ("All trades", df, "#3498db", "-"),
    ("Good-catalyst only", df_good, "#2ecc71", "--"),
    ("Earnings-excluded", non_earn, "#e67e22", "-."),
]:
    s = subset.sort_values("eval_date")["pnl_eod_nosl"].values
    ax.plot(np.arange(len(s)), np.cumsum(s), label=f"{label} (n={len(s)})", color=color, ls=ls, linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Trade #")
ax.set_ylabel("Cumulative PnL (%)")
ax.set_title("Cumulative EOD PnL: Filters")
ax.legend(fontsize=8)

fig_e3.tight_layout(rect=[0, 0, 1, 0.96])
e3_path = BASE / "v3_e3_catalysts.png"
fig_e3.savefig(e3_path, dpi=150, bbox_inches="tight")
plt.close(fig_e3)
print(f"\n  Saved: {e3_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E6: n_runs POSITION SIZING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("E6: n_runs POSITION SIZING")
print("=" * 90)

# n_runs distribution
print("\n── n_runs distribution ──")
print(df.n_runs.value_counts().sort_index().to_string())

# Assign buckets and weights
def nruns_bucket(n):
    if n <= 2:
        return "<=2"
    elif n == 3:
        return "3"
    else:
        return ">=4"

df["nruns_bucket"] = df.n_runs.apply(nruns_bucket)

WEIGHT_CONSERVATIVE = {"<=2": 0.5, "3": 1.0, ">=4": 1.5}
WEIGHT_AGGRESSIVE = {"<=2": 0.5, "3": 1.0, ">=4": 2.0}

# ── Bucket breakdown ─────────────────────────────────────────────────────────
print("\n── n_runs bucket performance ──")
bucket_results = {}
for bucket in ["<=2", "3", ">=4"]:
    sub = df[df.nruns_bucket == bucket]
    print(f"\n  Bucket: n_runs {bucket} (n={len(sub)})")
    row = {"n": len(sub)}
    for hz_label, col in PNL_COLS.items():
        s = sub[col].dropna()
        d = pnl_stats(s)
        row[hz_label] = d
        flag = d["flag"]
        print(f"    {hz_label}: n={d['n']:>4}{flag}  mean={d['mean']:>+7.3f}%  trim={d['trimmed_mean']:>+7.3f}%  WR={d['win_rate']:>5.1f}%")
    bucket_results[bucket] = row


def weighted_portfolio(df_in, weight_map, col):
    """Compute weighted avg PnL: sum(w*pnl)/sum(w), and weighted Sharpe."""
    sub = df_in[[col, "nruns_bucket"]].dropna(subset=[col]).copy()
    sub["w"] = sub["nruns_bucket"].map(weight_map)
    sub["wpnl"] = sub["w"] * sub[col]
    n = len(sub)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sharpe": np.nan}
    ew_mean = sub[col].mean()
    w_mean = sub["wpnl"].sum() / sub["w"].sum()
    # Sharpe: treat each weighted PnL as a return observation
    w_returns = sub["wpnl"].values / sub["w"].mean()  # normalize
    sh = safe_sharpe(w_returns)
    ew_sh = safe_sharpe(sub[col].values)
    return {
        "n": n,
        "ew_mean": round(ew_mean, 4),
        "w_mean": round(w_mean, 4),
        "delta": round(w_mean - ew_mean, 4),
        "ew_sharpe": round(ew_sh, 2) if not np.isnan(ew_sh) else np.nan,
        "w_sharpe": round(sh, 2) if not np.isnan(sh) else np.nan,
        "ew_wr": round((sub[col] > 0).mean() * 100, 1),
        "w_wr": round((sub["wpnl"] > 0).mean() * 100, 1),
    }


# ── Portfolio comparisons ────────────────────────────────────────────────────
print("\n── Portfolio: equal-weight vs n_runs-weighted ──")
e6_portfolio = {}
for scope_label, scope_df in [("ALL", df), ("LONG", df[df.sign == 1]), ("SHORT", df[df.sign == -1])]:
    print(f"\n  {scope_label}:")
    scope_res = {}
    for hz_label, col in [("60m", "pnl_60m_nosl"), ("EOD", "pnl_eod_nosl"), ("T+2", "pnl_t2_close")]:
        cons = weighted_portfolio(scope_df, WEIGHT_CONSERVATIVE, col)
        aggr = weighted_portfolio(scope_df, WEIGHT_AGGRESSIVE, col)
        scope_res[hz_label] = {"conservative_1.5x": cons, "aggressive_2.0x": aggr}
        print(
            f"    {hz_label}: EW={cons['ew_mean']:>+7.3f}% (Sh={cons['ew_sharpe']})  "
            f"1.5x={cons['w_mean']:>+7.3f}% (Sh={cons['w_sharpe']}, d={cons['delta']:>+6.3f})  "
            f"2.0x={aggr['w_mean']:>+7.3f}% (Sh={aggr['w_sharpe']}, d={aggr['delta']:>+6.3f})"
        )
    e6_portfolio[scope_label] = scope_res

results["E6"] = {
    "bucket_stats": {
        bucket: {hz: {k: v for k, v in d.items() if k != "flag"} for hz, d in row.items() if hz != "n"}
        for bucket, row in bucket_results.items()
    },
    "portfolio": e6_portfolio,
}

# ── E6 charts ─────────────────────────────────────────────────────────────────
fig_e6, axes_e6 = plt.subplots(2, 2, figsize=(16, 12))
fig_e6.suptitle("E6: n_runs Position Sizing", fontsize=15, fontweight="bold")

# Chart 1: n_runs bucket → avg PnL + n trades (dual axis)
ax = axes_e6[0, 0]
buckets = ["<=2", "3", ">=4"]
bucket_means_60 = []
bucket_ns = []
for b in buckets:
    sub = df[df.nruns_bucket == b]
    s = sub["pnl_60m_nosl"].dropna()
    bucket_means_60.append(s.mean() if len(s) else 0)
    bucket_ns.append(len(sub))

x = np.arange(len(buckets))
colors_b = ["#e74c3c" if v < 0 else "#2ecc71" for v in bucket_means_60]
bars = ax.bar(x, bucket_means_60, color=colors_b, alpha=0.8, width=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"n_runs {b}" for b in buckets])
ax.set_ylabel("Avg 60m PnL (%)")
ax.axhline(0, color="black", linewidth=0.8)

ax2 = ax.twinx()
ax2.plot(x, bucket_ns, "ko-", markersize=8)
ax2.set_ylabel("# trades")
for i, n in enumerate(bucket_ns):
    ax2.annotate(f"n={n}", (i, n), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.set_title("n_runs Bucket: Avg 60m PnL & Trade Count")

# Chart 2: Sharpe comparison bar
ax = axes_e6[0, 1]
# Sharpe at 60m for ALL scope
sharpes_60 = []
labels_sh = ["Equal-weight", "1.5x weighted", "2.0x weighted"]
for wm_label in ["ew_sharpe", "w_sharpe"]:
    pass  # will build directly

sub_60 = df[["pnl_60m_nosl", "nruns_bucket"]].dropna(subset=["pnl_60m_nosl"]).copy()
ew_sh = safe_sharpe(sub_60["pnl_60m_nosl"].values)

sub_60["w_cons"] = sub_60["nruns_bucket"].map(WEIGHT_CONSERVATIVE)
sub_60["w_aggr"] = sub_60["nruns_bucket"].map(WEIGHT_AGGRESSIVE)
cons_ret = (sub_60["w_cons"] * sub_60["pnl_60m_nosl"]).values / sub_60["w_cons"].mean()
aggr_ret = (sub_60["w_aggr"] * sub_60["pnl_60m_nosl"]).values / sub_60["w_aggr"].mean()
cons_sh = safe_sharpe(cons_ret)
aggr_sh = safe_sharpe(aggr_ret)

sharpe_vals = [ew_sh, cons_sh, aggr_sh]
bar_colors = ["#3498db", "#2ecc71", "#e67e22"]
ax.bar(range(3), sharpe_vals, color=bar_colors, alpha=0.8, width=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels(labels_sh, fontsize=9)
ax.set_ylabel("Sharpe Ratio (annualized)")
for i, v in enumerate(sharpe_vals):
    if not np.isnan(v):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Sharpe: EW vs n_runs Weighted (60m)")

# Chart 3: Cumulative PnL — EW vs weighted (EOD, no nulls)
ax = axes_e6[1, 0]
sorted_df = df.sort_values("eval_date").copy()
sorted_df["w_cons"] = sorted_df["nruns_bucket"].map(WEIGHT_CONSERVATIVE)
sorted_df["w_aggr"] = sorted_df["nruns_bucket"].map(WEIGHT_AGGRESSIVE)

ew_cum = np.cumsum(sorted_df["pnl_eod_nosl"].values)
cons_cum = np.cumsum((sorted_df["w_cons"] * sorted_df["pnl_eod_nosl"]).values) / sorted_df["w_cons"].mean()
aggr_cum = np.cumsum((sorted_df["w_aggr"] * sorted_df["pnl_eod_nosl"]).values) / sorted_df["w_aggr"].mean()

ax.plot(ew_cum, label="Equal-weight", color="#3498db", linewidth=1.5)
ax.plot(cons_cum, label="1.5x weighted", color="#2ecc71", linewidth=1.5, ls="--")
ax.plot(aggr_cum, label="2.0x weighted", color="#e67e22", linewidth=1.5, ls="-.")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Trade #")
ax.set_ylabel("Cumulative PnL (%)")
ax.set_title("Cumulative EOD PnL: EW vs n_runs Weighted")
ax.legend(fontsize=9)

# Chart 4: n_runs distribution histogram
ax = axes_e6[1, 1]
n_vals = df.n_runs.value_counts().sort_index()
ax.bar(n_vals.index, n_vals.values, color="#3498db", alpha=0.8, edgecolor="white")
ax.set_xlabel("n_runs")
ax.set_ylabel("# trades")
ax.set_title("n_runs Distribution")
for i, (idx, val) in enumerate(n_vals.items()):
    ax.text(idx, val + 1, str(val), ha="center", fontsize=9)

fig_e6.tight_layout(rect=[0, 0, 1, 0.96])
e6_path = BASE / "v3_e6_nruns_sizing.png"
fig_e6.savefig(e6_path, dpi=150, bbox_inches="tight")
plt.close(fig_e6)
print(f"\n  Saved: {e6_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E10: MAGNITUDE POSITION SIZING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("E10: MAGNITUDE POSITION SIZING")
print("=" * 90)

# Compute weights
df["abs_expected_move"] = df["expected_move_pct"].abs()
df["abs_gap"] = df["gap_pct"].abs()

# Median for normalization (exclude zeros for expected_move to avoid skew)
med_exp = df.loc[df.abs_expected_move > 0, "abs_expected_move"].median()
med_gap = df.loc[df.abs_gap > 0, "abs_gap"].median()
print(f"\n  Median |expected_move_pct| (non-zero): {med_exp:.3f}")
print(f"  Median |gap_pct| (non-zero): {med_gap:.3f}")

# Weight = magnitude / median (clip at 0.2 floor, 5.0 cap)
df["w_exp"] = (df["abs_expected_move"] / med_exp).clip(0.2, 5.0)
df["w_gap"] = (df["abs_gap"] / med_gap).clip(0.2, 5.0)
# For rows with 0 expected_move, set weight to floor
df.loc[df.abs_expected_move == 0, "w_exp"] = 0.2
df.loc[df.abs_gap == 0, "w_gap"] = 0.2

print(f"  w_exp stats: mean={df.w_exp.mean():.3f}  median={df.w_exp.median():.3f}  min={df.w_exp.min():.3f}  max={df.w_exp.max():.3f}")
print(f"  w_gap stats: mean={df.w_gap.mean():.3f}  median={df.w_gap.median():.3f}  min={df.w_gap.min():.3f}  max={df.w_gap.max():.3f}")


def magnitude_portfolio(df_in, weight_col, pnl_col):
    """Compute magnitude-weighted portfolio stats."""
    sub = df_in[[pnl_col, weight_col]].dropna(subset=[pnl_col]).copy()
    n = len(sub)
    if n == 0:
        return {"n": 0}
    ew_mean = sub[pnl_col].mean()
    w_mean = (sub[weight_col] * sub[pnl_col]).sum() / sub[weight_col].sum()
    # Sharpe
    w_ret = (sub[weight_col] * sub[pnl_col]).values / sub[weight_col].mean()
    ew_sh = safe_sharpe(sub[pnl_col].values)
    w_sh = safe_sharpe(w_ret)
    return {
        "n": n,
        "ew_mean": round(ew_mean, 4),
        "w_mean": round(w_mean, 4),
        "delta": round(w_mean - ew_mean, 4),
        "ew_sharpe": round(ew_sh, 2) if not np.isnan(ew_sh) else np.nan,
        "w_sharpe": round(w_sh, 2) if not np.isnan(w_sh) else np.nan,
    }


# ── Portfolio comparison ──────────────────────────────────────────────────────
print("\n── Portfolio: EW vs expected_move-weighted vs gap-weighted ──")
e10_portfolio = {}
for scope_label, scope_df in [("ALL", df), ("LONG", df[df.sign == 1]), ("SHORT", df[df.sign == -1])]:
    print(f"\n  {scope_label}:")
    scope_res = {}
    for hz_label, col in [("60m", "pnl_60m_nosl"), ("EOD", "pnl_eod_nosl"), ("T+2", "pnl_t2_close")]:
        exp_r = magnitude_portfolio(scope_df, "w_exp", col)
        gap_r = magnitude_portfolio(scope_df, "w_gap", col)
        scope_res[hz_label] = {"expected_move": exp_r, "gap": gap_r}
        print(
            f"    {hz_label}: EW={exp_r['ew_mean']:>+7.3f}% (Sh={exp_r['ew_sharpe']})  "
            f"exp_move={exp_r['w_mean']:>+7.3f}% (Sh={exp_r['w_sharpe']}, d={exp_r['delta']:>+6.3f})  "
            f"gap={gap_r['w_mean']:>+7.3f}% (Sh={gap_r['w_sharpe']}, d={gap_r['delta']:>+6.3f})"
        )
    e10_portfolio[scope_label] = scope_res

# ── Correlation analysis ─────────────────────────────────────────────────────
print("\n── Correlation: magnitude vs realized PnL ──")
valid_60 = df[["abs_expected_move", "abs_gap", "pnl_60m_nosl", "pnl_eod_nosl"]].dropna(subset=["pnl_60m_nosl"])
r_exp_60, p_exp_60 = stats.pearsonr(valid_60["abs_expected_move"], valid_60["pnl_60m_nosl"])
r_gap_60, p_gap_60 = stats.pearsonr(valid_60["abs_gap"], valid_60["pnl_60m_nosl"])
r_exp_abs_60, p_exp_abs_60 = stats.pearsonr(valid_60["abs_expected_move"], valid_60["pnl_60m_nosl"].abs())
r_gap_abs_60, p_gap_abs_60 = stats.pearsonr(valid_60["abs_gap"], valid_60["pnl_60m_nosl"].abs())

print(f"  |expected_move| vs PnL_60m:      r={r_exp_60:+.4f}  p={p_exp_60:.4f}")
print(f"  |gap_pct|       vs PnL_60m:      r={r_gap_60:+.4f}  p={p_gap_60:.4f}")
print(f"  |expected_move| vs |PnL_60m|:    r={r_exp_abs_60:+.4f}  p={p_exp_abs_60:.4f}")
print(f"  |gap_pct|       vs |PnL_60m|:    r={r_gap_abs_60:+.4f}  p={p_gap_abs_60:.4f}")

# ── Quartile analysis ────────────────────────────────────────────────────────
print("\n── Quartile analysis: |expected_move_pct| quartiles ──")
valid_60["exp_q"] = pd.qcut(valid_60["abs_expected_move"], 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"], duplicates="drop")
for q, grp in valid_60.groupby("exp_q", observed=True):
    s = grp["pnl_60m_nosl"]
    flag = " [LOW-N]" if len(s) < LOW_N else ""
    print(f"  {q}: n={len(s):>4}{flag}  mean={s.mean():>+7.3f}%  trim={tmean(s.values):>+7.3f}%  WR={100*(s>0).mean():>5.1f}%")

print("\n── Quartile analysis: |gap_pct| quartiles ──")
valid_60["gap_q"] = pd.qcut(valid_60["abs_gap"], 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"], duplicates="drop")
for q, grp in valid_60.groupby("gap_q", observed=True):
    s = grp["pnl_60m_nosl"]
    flag = " [LOW-N]" if len(s) < LOW_N else ""
    print(f"  {q}: n={len(s):>4}{flag}  mean={s.mean():>+7.3f}%  trim={tmean(s.values):>+7.3f}%  WR={100*(s>0).mean():>5.1f}%")

# Store results
results["E10"] = {
    "correlation": {
        "exp_move_vs_pnl60": {"r": round(r_exp_60, 4), "p": round(p_exp_60, 4)},
        "gap_vs_pnl60": {"r": round(r_gap_60, 4), "p": round(p_gap_60, 4)},
        "exp_move_vs_abs_pnl60": {"r": round(r_exp_abs_60, 4), "p": round(p_exp_abs_60, 4)},
        "gap_vs_abs_pnl60": {"r": round(r_gap_abs_60, 4), "p": round(p_gap_abs_60, 4)},
    },
    "portfolio": e10_portfolio,
    "median_expected_move": round(med_exp, 4),
    "median_gap": round(med_gap, 4),
}

# ── E10 charts ────────────────────────────────────────────────────────────────
fig_e10, axes_e10 = plt.subplots(2, 2, figsize=(16, 12))
fig_e10.suptitle("E10: Magnitude Position Sizing", fontsize=15, fontweight="bold")

# Chart 1: Scatter — |expected_move| vs PnL 60m with regression
ax = axes_e10[0, 0]
v = valid_60.copy()
ax.scatter(v["abs_expected_move"], v["pnl_60m_nosl"], alpha=0.35, s=20, c="#3498db", edgecolors="white", linewidth=0.3)
# Regression line
slope, intercept, _, _, _ = stats.linregress(v["abs_expected_move"], v["pnl_60m_nosl"])
x_line = np.linspace(0, v["abs_expected_move"].quantile(0.95), 50)
ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, label=f"r={r_exp_60:+.3f}")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("|expected_move_pct|")
ax.set_ylabel("PnL 60m (%)")
ax.set_title("|Expected Move| vs 60m PnL")
ax.set_xlim(0, v["abs_expected_move"].quantile(0.95))
ax.set_ylim(v["pnl_60m_nosl"].quantile(0.02), v["pnl_60m_nosl"].quantile(0.98))
ax.legend()

# Chart 2: Scatter — |gap_pct| vs PnL 60m with regression
ax = axes_e10[0, 1]
ax.scatter(v["abs_gap"], v["pnl_60m_nosl"], alpha=0.35, s=20, c="#e67e22", edgecolors="white", linewidth=0.3)
slope_g, intercept_g, _, _, _ = stats.linregress(v["abs_gap"], v["pnl_60m_nosl"])
x_line_g = np.linspace(0, v["abs_gap"].quantile(0.95), 50)
ax.plot(x_line_g, slope_g * x_line_g + intercept_g, "r-", linewidth=2, label=f"r={r_gap_60:+.3f}")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("|gap_pct|")
ax.set_ylabel("PnL 60m (%)")
ax.set_title("|Gap %| vs 60m PnL")
ax.set_xlim(0, v["abs_gap"].quantile(0.95))
ax.set_ylim(v["pnl_60m_nosl"].quantile(0.02), v["pnl_60m_nosl"].quantile(0.98))
ax.legend()

# Chart 3: Sharpe comparison bar
ax = axes_e10[1, 0]
# 60m Sharpe: EW, exp-weighted, gap-weighted
r_all_60 = e10_portfolio["ALL"]["60m"]
sharpes_10 = [
    r_all_60["expected_move"]["ew_sharpe"],
    r_all_60["expected_move"]["w_sharpe"],
    r_all_60["gap"]["w_sharpe"],
]
labels_10 = ["Equal-weight", "Expected-move\nweighted", "Gap-pct\nweighted"]
colors_10 = ["#3498db", "#2ecc71", "#e67e22"]
ax.bar(range(3), sharpes_10, color=colors_10, alpha=0.8, width=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels(labels_10, fontsize=9)
ax.set_ylabel("Sharpe Ratio (annualized)")
for i, v_sh in enumerate(sharpes_10):
    if not np.isnan(v_sh):
        ax.text(i, v_sh + 0.05 * np.sign(v_sh), f"{v_sh:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Sharpe: EW vs Magnitude Weighted (60m)")
ax.axhline(0, color="black", linewidth=0.8)

# Chart 4: Quartile PnL bars (expected_move quartiles)
ax = axes_e10[1, 1]
q_labels = []
q_means = []
q_ns = []
for q, grp in valid_60.groupby("exp_q", observed=True):
    s = grp["pnl_60m_nosl"]
    q_labels.append(str(q))
    q_means.append(s.mean())
    q_ns.append(len(s))

colors_q = ["#2ecc71" if m > 0 else "#e74c3c" for m in q_means]
bars_q = ax.bar(range(len(q_labels)), q_means, color=colors_q, alpha=0.8, width=0.5)
ax.set_xticks(range(len(q_labels)))
ax.set_xticklabels(q_labels, fontsize=9)
ax.set_ylabel("Avg 60m PnL (%)")
ax.set_title("|Expected Move| Quartile: Avg 60m PnL")
ax.axhline(0, color="black", linewidth=0.8)
for i, (m, n) in enumerate(zip(q_means, q_ns)):
    ax.text(i, m + 0.05 * (1 if m >= 0 else -1), f"n={n}\n{m:+.2f}%", ha="center", fontsize=8, va="bottom" if m >= 0 else "top")

fig_e10.tight_layout(rect=[0, 0, 1, 0.96])
e10_path = BASE / "v3_e10_magnitude_sizing.png"
fig_e10.savefig(e10_path, dpi=150, bbox_inches="tight")
plt.close(fig_e10)
print(f"\n  Saved: {e10_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMBINED SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 90)
print("COMBINED SUMMARY")
print("=" * 90)

print("\nE3 — Catalyst findings:")
print(f"  Good catalysts: {good_cats}")
print(f"  Bad catalysts: {bad_cats}")
e3_delta_60 = results["E3"]["earnings_impact"].get("60m", {})
print(f"  Earnings exclusion delta (60m): {e3_delta_60.get('delta', 'N/A'):+.3f}%")

print("\nE6 — n_runs sizing:")
e6_all_60 = e6_portfolio.get("ALL", {}).get("60m", {})
cons_d = e6_all_60.get("conservative_1.5x", {}).get("delta", "N/A")
aggr_d = e6_all_60.get("aggressive_2.0x", {}).get("delta", "N/A")
print(f"  1.5x weighting delta (60m): {cons_d:+.3f}%")
print(f"  2.0x weighting delta (60m): {aggr_d:+.3f}%")

print("\nE10 — Magnitude sizing:")
e10_all_60 = e10_portfolio.get("ALL", {}).get("60m", {})
exp_d = e10_all_60.get("expected_move", {}).get("delta", "N/A")
gap_d = e10_all_60.get("gap", {}).get("delta", "N/A")
print(f"  Expected-move weighting delta (60m): {exp_d:+.3f}%")
print(f"  Gap-pct weighting delta (60m): {gap_d:+.3f}%")
print(f"  Correlation |exp_move| vs PnL_60m: r={r_exp_60:+.4f}")
print(f"  Correlation |gap_pct|  vs PnL_60m: r={r_gap_60:+.4f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
# Clean NaN for JSON serialization
def clean_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(i) for i in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

with open(OUT_JSON, "w") as f:
    json.dump(clean_nan(results), f, indent=2)
print(f"\nSaved: {OUT_JSON}")
print("\nDone.")
