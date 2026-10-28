# %% Agent 6 — Attention & Positioning Proxies for S5_zexit
#
# Investigates whether volume-based attention signals, short-squeeze proxies,
# VIX regimes, and sector flow indicators improve S5_zexit performance on RXRX.
#
# All proxies use trailing data only (no lookahead).
# Selection window: before 2026-01-01. OOS: 2026-01-01 onward.

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

for pkg in ["matplotlib", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                               "--quiet", "--break-system-packages"])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as scipy_stats

# ── Import shared engine ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent if "__file__" in dir() else Path("agent10")))
from agent0_engine import (build_spread, run_backtest, compute_stats,
                           compute_costs, log_trial, add_features,
                           s5_zexit_signal, s5_zexit_exit, DATA_DIR, OUT_DIR)

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
SEL_END = "2026-01-01"          # selection / OOS boundary
K_ENTRY = 1.5                   # z-score entry threshold
Z_EXIT = 0.5                    # z-score exit threshold
MAX_HOLD = 10                   # max holding days
LOOKBACK = 60                   # OLS spread lookback

# Attention proxy thresholds
VOL_Z_HIGH = 1.0                # volume z-score threshold for "high attention"
REL_VOL_HIGH = 2.0              # relative volume threshold
DVA_HIGH = 1.3                  # dollar volume acceleration threshold

# Short squeeze proxy thresholds
ATR_COMPRESS = 0.8              # ATR contraction ratio threshold
BEAR_CANDLE_HIGH = 12           # bearish candle count threshold (out of 20)

# VIX percentile-based regimes (adaptive to data)
VIX_LOW_PCT = 33                # below this percentile = low
VIX_HIGH_PCT = 66               # above this percentile = high

# Sector flow
RS_LOOKBACK = 20                # relative strength trailing window

print("=" * 70)
print("  AGENT 6: ATTENTION & POSITIONING PROXIES")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════
rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)

ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])

arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)
xbi = ctx[ctx["ticker"] == "XBI"].sort_values("date").reset_index(drop=True)

vix_raw = pd.read_parquet(DATA_DIR / "vix_daily.parquet")
vix_raw["date"] = pd.to_datetime(vix_raw["date"])
vix_raw = vix_raw.sort_values("date").reset_index(drop=True)

rxrx = add_features(rxrx)

print(f"\n  Data loaded:")
print(f"    RXRX:  {len(rxrx)} bars, {rxrx['date'].min().date()} to {rxrx['date'].max().date()}")
print(f"    ARKG:  {len(arkg)} bars")
print(f"    VIX:   {len(vix_raw)} bars")

# ══════════════════════════════════════════════════════════════════
# 2. BUILD SPREAD & RUN BASELINE S5_zexit
# ══════════════════════════════════════════════════════════════════
print("\n  Building OLS spread (RXRX vs ARKG)...")
spread = build_spread(rxrx, arkg, "ARKG", LOOKBACK)

def sig_fn(row, i, df):
    return s5_zexit_signal(row, i, df, k_entry=K_ENTRY)

def exit_fn(sig, row, j, df):
    return s5_zexit_exit(sig, row, j, df, z_exit_thresh=Z_EXIT)

trades_all = run_backtest(spread, sig_fn, exit_fn, max_hold=MAX_HOLD)

trades_sel = trades_all[trades_all["entry_date"] < SEL_END].copy()
trades_oos = trades_all[trades_all["entry_date"] >= SEL_END].copy()

baseline_sel = compute_stats(trades_sel, label="Baseline SEL")
baseline_oos = compute_stats(trades_oos, label="Baseline OOS")

print(f"\n  Baseline S5_zexit:")
print(f"    Selection: n={baseline_sel['n']}, avg_pnl={baseline_sel['avg_pnl']:+.3f}%, "
      f"t={baseline_sel['t_stat']:.2f}")
print(f"    OOS 2026:  n={baseline_oos['n']}, avg_pnl={baseline_oos['avg_pnl']:+.3f}%, "
      f"t={baseline_oos['t_stat']:.2f}")

# ══════════════════════════════════════════════════════════════════
# 3. COMPUTE ATTENTION & POSITIONING PROXIES
# ══════════════════════════════════════════════════════════════════
print("\n  Computing proxies...")

# --- 3a. Volume-based attention ---
rxrx["vol_ma20"] = rxrx["volume"].rolling(20).mean()
rxrx["vol_std20"] = rxrx["volume"].rolling(20).std()
rxrx["vol_z"] = (rxrx["volume"] - rxrx["vol_ma20"]) / rxrx["vol_std20"]
rxrx["rel_vol"] = rxrx["volume"] / rxrx["vol_ma20"]
rxrx["dollar_vol"] = rxrx["close"] * rxrx["volume"]
rxrx["dva_5_20"] = rxrx["dollar_vol"].rolling(5).mean() / rxrx["dollar_vol"].rolling(20).mean()

# --- 3b. Short squeeze proxy ---
rxrx["bearish_candle"] = (rxrx["close"] < rxrx["open"]).astype(int)
rxrx["bear_count_20"] = rxrx["bearish_candle"].rolling(20).sum()

# ATR contraction
high, low, prev_close = rxrx["high"], rxrx["low"], rxrx["close"].shift(1)
tr = pd.concat([high - low, (high - prev_close).abs(),
                (low - prev_close).abs()], axis=1).max(axis=1)
rxrx["atr20_local"] = tr.rolling(20).mean()
rxrx["atr60_local"] = tr.rolling(60).mean()
rxrx["atr_ratio"] = rxrx["atr20_local"] / rxrx["atr60_local"]

# Volume decline proxy (days to cover approximation)
rxrx["vol_ma5"] = rxrx["volume"].rolling(5).mean()
rxrx["vol_decline"] = rxrx["vol_ma5"] / rxrx["vol_ma20"]  # <1 means declining volume

# Composite short-squeeze score: elevated bear candles + low volume + ATR compression
# Normalize each component to [0,1] range using trailing 252d percentile rank
for col in ["bear_count_20", "atr_ratio"]:
    rxrx[f"{col}_pctrank"] = rxrx[col].rolling(252, min_periods=60).apply(
        lambda x: scipy_stats.percentileofscore(x[:-1], x[-1]) / 100 if len(x) > 1 else 0.5,
        raw=True
    )

# Invert vol_decline (low volume = harder to cover = higher squeeze potential)
rxrx["vol_decline_inv_pctrank"] = rxrx["vol_decline"].rolling(252, min_periods=60).apply(
    lambda x: 1 - scipy_stats.percentileofscore(x[:-1], x[-1]) / 100 if len(x) > 1 else 0.5,
    raw=True
)

# Invert ATR ratio (lower = more compressed = higher squeeze potential)
rxrx["atr_compress_pctrank"] = rxrx["atr_ratio"].rolling(252, min_periods=60).apply(
    lambda x: 1 - scipy_stats.percentileofscore(x[:-1], x[-1]) / 100 if len(x) > 1 else 0.5,
    raw=True
)

rxrx["squeeze_score"] = (rxrx["bear_count_20_pctrank"] +
                          rxrx["vol_decline_inv_pctrank"] +
                          rxrx["atr_compress_pctrank"]) / 3

# --- 3c. VIX regime ---
# Use adaptive thresholds based on trailing VIX distribution
vix_series = vix_raw[["date", "close"]].rename(columns={"close": "vix_close"})

# --- 3d. Sector relative strength ---
arkg_rs = arkg[["date", "close"]].rename(columns={"close": "arkg_close"})

# Merge proxies onto spread dataframe
spread_aug = spread.copy()
rxrx_proxies = rxrx[["date", "vol_z", "rel_vol", "dva_5_20",
                      "bear_count_20", "atr_ratio", "vol_decline",
                      "squeeze_score"]].copy()
spread_aug = spread_aug.merge(rxrx_proxies, on="date", how="left")
spread_aug = spread_aug.merge(vix_series, on="date", how="left")
spread_aug = spread_aug.merge(arkg_rs, on="date", how="left")

# Compute RXRX trailing 20d return
spread_aug["rxrx_ret_20"] = spread_aug["close"].pct_change(RS_LOOKBACK)
spread_aug["arkg_ret_20"] = spread_aug["arkg_close"].pct_change(RS_LOOKBACK)
spread_aug["relative_strength"] = spread_aug["rxrx_ret_20"] - spread_aug["arkg_ret_20"]

print("  Proxies computed.")

# ══════════════════════════════════════════════════════════════════
# 4. ATTACH PROXY VALUES TO EACH TRADE (at signal date)
# ══════════════════════════════════════════════════════════════════
proxy_cols = ["vol_z", "rel_vol", "dva_5_20", "bear_count_20",
              "atr_ratio", "vol_decline", "squeeze_score",
              "vix_close", "relative_strength"]

spread_proxy_lookup = spread_aug[["date"] + proxy_cols].copy()

trades_all_aug = trades_all.merge(
    spread_proxy_lookup, left_on="signal_date", right_on="date", how="left"
).drop(columns=["date"], errors="ignore")

trades_sel_aug = trades_all_aug[trades_all_aug["entry_date"] < SEL_END].copy()
trades_oos_aug = trades_all_aug[trades_all_aug["entry_date"] >= SEL_END].copy()

# ══════════════════════════════════════════════════════════════════
# 5. ANALYSIS HELPER
# ══════════════════════════════════════════════════════════════════

def split_and_compare(trades_df, col, threshold, label, high_label="high",
                      low_label="low", above_is_high=True):
    """Split trades by proxy threshold and compare performance."""
    valid = trades_df.dropna(subset=[col])
    if above_is_high:
        high = valid[valid[col] >= threshold]
        low = valid[valid[col] < threshold]
    else:
        high = valid[valid[col] <= threshold]
        low = valid[valid[col] > threshold]

    n_high = len(high)
    n_low = len(low)
    avg_high = high["pnl_net_pct"].mean() if n_high > 0 else np.nan
    avg_low = low["pnl_net_pct"].mean() if n_low > 0 else np.nan
    diff = (avg_high - avg_low) if (n_high > 0 and n_low > 0) else np.nan

    # t-test
    if n_high >= 2 and n_low >= 2:
        t, p = scipy_stats.ttest_ind(high["pnl_net_pct"], low["pnl_net_pct"],
                                     equal_var=False)
    else:
        t, p = np.nan, np.nan

    return {
        "proxy": label, "threshold": threshold,
        f"n_{high_label}": n_high, f"avg_pnl_{high_label}": round(avg_high, 3) if not np.isnan(avg_high) else np.nan,
        f"n_{low_label}": n_low, f"avg_pnl_{low_label}": round(avg_low, 3) if not np.isnan(avg_low) else np.nan,
        "diff": round(diff, 3) if not np.isnan(diff) else np.nan,
        "t_stat": round(t, 2) if not np.isnan(t) else np.nan,
        "p_val": round(p, 3) if not np.isnan(p) else np.nan,
    }


def vix_regime_split(trades_df, vix_col="vix_close"):
    """Split trades by VIX tercile regimes."""
    valid = trades_df.dropna(subset=[vix_col])
    if len(valid) == 0:
        return pd.DataFrame()

    # Use 33rd and 66th percentile of VIX values in trades
    all_vix = valid[vix_col]
    t1 = all_vix.quantile(0.33)
    t2 = all_vix.quantile(0.66)

    low = valid[valid[vix_col] <= t1]
    med = valid[(valid[vix_col] > t1) & (valid[vix_col] <= t2)]
    high = valid[valid[vix_col] > t2]

    results = []
    for lbl, sub, thresh_desc in [("VIX_low", low, f"<={t1:.1f}"),
                                   ("VIX_med", med, f"{t1:.1f}-{t2:.1f}"),
                                   ("VIX_high", high, f">{t2:.1f}")]:
        n = len(sub)
        avg = sub["pnl_net_pct"].mean() if n > 0 else np.nan
        wr = (sub["pnl_net_pct"] > 0).mean() * 100 if n > 0 else np.nan
        results.append({
            "regime": lbl, "vix_range": thresh_desc, "n": n,
            "avg_pnl": round(avg, 3) if not np.isnan(avg) else np.nan,
            "win_rate": round(wr, 1) if not np.isnan(wr) else np.nan,
            "total_pnl": round(sub["pnl_net_pct"].sum(), 2) if n > 0 else 0,
        })

    return pd.DataFrame(results), t1, t2


# ══════════════════════════════════════════════════════════════════
# 6. RUN ALL PROXY SPLITS — SELECTION WINDOW
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PROXY ANALYSIS — SELECTION WINDOW (before 2026-01-01)")
print("=" * 70)

results_sel = []

# 6a. Volume-based attention
print("\n  --- Volume-Based Attention ---")
for col, thresh, label in [("vol_z", VOL_Z_HIGH, "Volume Z-score"),
                             ("rel_vol", REL_VOL_HIGH, "Relative Volume"),
                             ("dva_5_20", DVA_HIGH, "Dollar Vol Acceleration")]:
    r = split_and_compare(trades_sel_aug, col, thresh, label)
    results_sel.append(r)
    print(f"    {label} (>={thresh}): "
          f"HIGH n={r['n_high']}, avg={r['avg_pnl_high']}, "
          f"LOW n={r['n_low']}, avg={r['avg_pnl_low']}, "
          f"diff={r['diff']}, t={r['t_stat']}")

# 6b. Short squeeze proxy
print("\n  --- Short Squeeze Proxies ---")
r = split_and_compare(trades_sel_aug, "bear_count_20", BEAR_CANDLE_HIGH,
                      "Bearish Candle Count (20d)")
results_sel.append(r)
print(f"    Bear count >=12: "
      f"HIGH n={r['n_high']}, avg={r['avg_pnl_high']}, "
      f"LOW n={r['n_low']}, avg={r['avg_pnl_low']}, "
      f"diff={r['diff']}, t={r['t_stat']}")

r = split_and_compare(trades_sel_aug, "atr_ratio", ATR_COMPRESS,
                      "ATR Compression (20/60)", above_is_high=False)
results_sel.append(r)
print(f"    ATR ratio <=0.8: "
      f"HIGH n={r['n_high']}, avg={r['avg_pnl_high']}, "
      f"LOW n={r['n_low']}, avg={r['avg_pnl_low']}, "
      f"diff={r['diff']}, t={r['t_stat']}")

# Composite squeeze score — use median as threshold
squeeze_median = trades_sel_aug["squeeze_score"].median()
r = split_and_compare(trades_sel_aug, "squeeze_score", squeeze_median,
                      f"Squeeze Score (median={squeeze_median:.2f})")
results_sel.append(r)
print(f"    Squeeze score >={squeeze_median:.2f}: "
      f"HIGH n={r['n_high']}, avg={r['avg_pnl_high']}, "
      f"LOW n={r['n_low']}, avg={r['avg_pnl_low']}, "
      f"diff={r['diff']}, t={r['t_stat']}")

# 6c. VIX regime
print("\n  --- VIX Regime ---")
vix_df_sel, vix_t1_sel, vix_t2_sel = vix_regime_split(trades_sel_aug)
if not vix_df_sel.empty:
    for _, row in vix_df_sel.iterrows():
        print(f"    {row['regime']} ({row['vix_range']}): "
              f"n={row['n']}, avg_pnl={row['avg_pnl']}, wr={row['win_rate']}%")

# VIX-gated variant: use upper tercile threshold
vix_gate_thresh_sel = vix_t2_sel
r_vix_gated = split_and_compare(trades_sel_aug, "vix_close", vix_gate_thresh_sel,
                                f"VIX-gated (>{vix_gate_thresh_sel:.1f})")
results_sel.append(r_vix_gated)
print(f"    VIX-gated >{vix_gate_thresh_sel:.1f}: "
      f"HIGH n={r_vix_gated['n_high']}, avg={r_vix_gated['avg_pnl_high']}, "
      f"LOW n={r_vix_gated['n_low']}, avg={r_vix_gated['avg_pnl_low']}")

# 6d. Sector relative strength
print("\n  --- Sector Relative Strength (RXRX vs ARKG 20d) ---")
rs_median = trades_sel_aug["relative_strength"].median()
# "Laggard" = negative RS (RXRX underperforming ARKG)
r_rs = split_and_compare(trades_sel_aug, "relative_strength", 0,
                         "Sector RS (laggard vs leader)",
                         high_label="leader", low_label="laggard",
                         above_is_high=True)
results_sel.append(r_rs)
print(f"    Laggard (RS<0): n={r_rs['n_laggard']}, avg={r_rs['avg_pnl_laggard']}")
print(f"    Leader  (RS>0): n={r_rs['n_leader']}, avg={r_rs['avg_pnl_leader']}")
print(f"    diff={r_rs['diff']}, t={r_rs['t_stat']}")

# ══════════════════════════════════════════════════════════════════
# 7. OOS VALIDATION (2026+)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PROXY ANALYSIS — OOS (2026-01-01 onward)")
print("=" * 70)

results_oos = []

# Volume attention
print("\n  --- Volume-Based Attention (OOS) ---")
for col, thresh, label in [("vol_z", VOL_Z_HIGH, "Volume Z-score"),
                             ("rel_vol", REL_VOL_HIGH, "Relative Volume"),
                             ("dva_5_20", DVA_HIGH, "Dollar Vol Acceleration")]:
    r = split_and_compare(trades_oos_aug, col, thresh, label)
    results_oos.append(r)
    print(f"    {label}: HIGH n={r['n_high']}, avg={r['avg_pnl_high']}, "
          f"LOW n={r['n_low']}, avg={r['avg_pnl_low']}, "
          f"diff={r['diff']}, t={r['t_stat']}")

# Short squeeze
print("\n  --- Short Squeeze (OOS) ---")
r = split_and_compare(trades_oos_aug, "squeeze_score", squeeze_median,
                      f"Squeeze Score (median={squeeze_median:.2f})")
results_oos.append(r)
print(f"    Squeeze score: HIGH n={r['n_high']}, avg={r['avg_pnl_high']}, "
      f"LOW n={r['n_low']}, avg={r['avg_pnl_low']}, "
      f"diff={r['diff']}, t={r['t_stat']}")

# VIX regime OOS
print("\n  --- VIX Regime (OOS) ---")
vix_df_oos, vix_t1_oos, vix_t2_oos = vix_regime_split(trades_oos_aug)
if not vix_df_oos.empty:
    for _, row in vix_df_oos.iterrows():
        print(f"    {row['regime']} ({row['vix_range']}): "
              f"n={row['n']}, avg_pnl={row['avg_pnl']}, wr={row['win_rate']}%")

# Sector RS OOS
print("\n  --- Sector RS (OOS) ---")
r_rs_oos = split_and_compare(trades_oos_aug, "relative_strength", 0,
                             "Sector RS (laggard vs leader)",
                             high_label="leader", low_label="laggard",
                             above_is_high=True)
results_oos.append(r_rs_oos)
print(f"    Laggard (RS<0): n={r_rs_oos['n_laggard']}, avg={r_rs_oos['avg_pnl_laggard']}")
print(f"    Leader  (RS>0): n={r_rs_oos['n_leader']}, avg={r_rs_oos['avg_pnl_leader']}")
print(f"    diff={r_rs_oos['diff']}, t={r_rs_oos['t_stat']}")

# ══════════════════════════════════════════════════════════════════
# 8. GATED BACKTEST VARIANTS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  GATED BACKTEST VARIANTS")
print("=" * 70)

# --- 8a. Volume-gated: only trade when vol_z > 1 ---
def make_vol_gated_signal(vol_z_lookup):
    """Signal only when volume z-score > threshold at signal bar."""
    def vol_gated_sig(row, i, df):
        sig = s5_zexit_signal(row, i, df, k_entry=K_ENTRY)
        if sig is None:
            return None
        dt = row.get("date", None)
        if dt is not None and dt in vol_z_lookup:
            if vol_z_lookup[dt] >= VOL_Z_HIGH:
                return sig
        return None
    return vol_gated_sig

# Build lookup
vol_z_lookup = dict(zip(rxrx["date"], rxrx["vol_z"]))

# Merge vol_z onto spread
spread_with_volz = spread.copy()
spread_with_volz = spread_with_volz.merge(
    rxrx[["date", "vol_z", "vix_close" if "vix_close" in rxrx.columns else "vol_z"]].rename(columns={}),
    on="date", how="left", suffixes=("", "_rxrx")
)

# Re-run with gating on spread_aug which has proxies
# Actually, let's build proper gated signals using spread_aug

# Add vol_z to spread for signal function access
spread_vol = spread_aug.copy()

def vol_gated_sig_fn(row, i, df):
    sig = s5_zexit_signal(row, i, df, k_entry=K_ENTRY)
    if sig is None:
        return None
    vz = row.get("vol_z", np.nan)
    if pd.notna(vz) and vz >= VOL_Z_HIGH:
        return sig
    return None

trades_vol_gated = run_backtest(spread_vol, vol_gated_sig_fn, exit_fn, max_hold=MAX_HOLD)
sel_vg = trades_vol_gated[trades_vol_gated["entry_date"] < SEL_END]
oos_vg = trades_vol_gated[trades_vol_gated["entry_date"] >= SEL_END]
stats_vg_sel = compute_stats(sel_vg, label="Vol-gated SEL")
stats_vg_oos = compute_stats(oos_vg, label="Vol-gated OOS")
print(f"\n  Volume-gated (vol_z >= {VOL_Z_HIGH}):")
print(f"    SEL: n={stats_vg_sel['n']}, avg_pnl={stats_vg_sel['avg_pnl']:+.3f}%, "
      f"t={stats_vg_sel['t_stat']:.2f}")
print(f"    OOS: n={stats_vg_oos['n']}, avg_pnl={stats_vg_oos['avg_pnl']:+.3f}%, "
      f"t={stats_vg_oos['t_stat']:.2f}")

# --- 8b. VIX-gated: only trade when VIX > upper tercile ---
def vix_gated_sig_fn(row, i, df):
    sig = s5_zexit_signal(row, i, df, k_entry=K_ENTRY)
    if sig is None:
        return None
    vix_val = row.get("vix_close", np.nan)
    if pd.notna(vix_val) and vix_val >= vix_gate_thresh_sel:
        return sig
    return None

trades_vix_gated = run_backtest(spread_vol, vix_gated_sig_fn, exit_fn, max_hold=MAX_HOLD)
sel_vxg = trades_vix_gated[trades_vix_gated["entry_date"] < SEL_END]
oos_vxg = trades_vix_gated[trades_vix_gated["entry_date"] >= SEL_END]
stats_vxg_sel = compute_stats(sel_vxg, label="VIX-gated SEL")
stats_vxg_oos = compute_stats(oos_vxg, label="VIX-gated OOS")
print(f"\n  VIX-gated (vix >= {vix_gate_thresh_sel:.1f}):")
print(f"    SEL: n={stats_vxg_sel['n']}, avg_pnl={stats_vxg_sel['avg_pnl']:+.3f}%, "
      f"t={stats_vxg_sel['t_stat']:.2f}")
print(f"    OOS: n={stats_vxg_oos['n']}, avg_pnl={stats_vxg_oos['avg_pnl']:+.3f}%, "
      f"t={stats_vxg_oos['t_stat']:.2f}")

# --- 8c. Laggard-only: only trade when RXRX underperforming ARKG ---
def laggard_sig_fn(row, i, df):
    sig = s5_zexit_signal(row, i, df, k_entry=K_ENTRY)
    if sig is None:
        return None
    rs = row.get("relative_strength", np.nan)
    if pd.notna(rs) and rs < 0:
        return sig
    return None

trades_laggard = run_backtest(spread_vol, laggard_sig_fn, exit_fn, max_hold=MAX_HOLD)
sel_lag = trades_laggard[trades_laggard["entry_date"] < SEL_END]
oos_lag = trades_laggard[trades_laggard["entry_date"] >= SEL_END]
stats_lag_sel = compute_stats(sel_lag, label="Laggard-only SEL")
stats_lag_oos = compute_stats(oos_lag, label="Laggard-only OOS")
print(f"\n  Laggard-only (RS < 0):")
print(f"    SEL: n={stats_lag_sel['n']}, avg_pnl={stats_lag_sel['avg_pnl']:+.3f}%, "
      f"t={stats_lag_sel['t_stat']:.2f}")
print(f"    OOS: n={stats_lag_oos['n']}, avg_pnl={stats_lag_oos['avg_pnl']:+.3f}%, "
      f"t={stats_lag_oos['t_stat']:.2f}")

# --- 8d. Squeeze-gated: only trade when squeeze score > median ---
def squeeze_sig_fn(row, i, df):
    sig = s5_zexit_signal(row, i, df, k_entry=K_ENTRY)
    if sig is None:
        return None
    sq = row.get("squeeze_score", np.nan)
    if pd.notna(sq) and sq >= squeeze_median:
        return sig
    return None

trades_squeeze = run_backtest(spread_vol, squeeze_sig_fn, exit_fn, max_hold=MAX_HOLD)
sel_sq = trades_squeeze[trades_squeeze["entry_date"] < SEL_END]
oos_sq = trades_squeeze[trades_squeeze["entry_date"] >= SEL_END]
stats_sq_sel = compute_stats(sel_sq, label="Squeeze-gated SEL")
stats_sq_oos = compute_stats(oos_sq, label="Squeeze-gated OOS")
print(f"\n  Squeeze-gated (score >= {squeeze_median:.2f}):")
print(f"    SEL: n={stats_sq_sel['n']}, avg_pnl={stats_sq_sel['avg_pnl']:+.3f}%, "
      f"t={stats_sq_sel['t_stat']:.2f}")
print(f"    OOS: n={stats_sq_oos['n']}, avg_pnl={stats_sq_oos['avg_pnl']:+.3f}%, "
      f"t={stats_sq_oos['t_stat']:.2f}")

# ══════════════════════════════════════════════════════════════════
# 9. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  GATED VARIANTS SUMMARY")
print("=" * 70)

summary_rows = [
    ("Baseline S5_zexit", baseline_sel["n"], baseline_sel["avg_pnl"],
     baseline_sel["t_stat"], baseline_oos["n"], baseline_oos["avg_pnl"],
     baseline_oos["t_stat"]),
    (f"Vol-gated (vz>={VOL_Z_HIGH})", stats_vg_sel["n"], stats_vg_sel["avg_pnl"],
     stats_vg_sel["t_stat"], stats_vg_oos["n"], stats_vg_oos["avg_pnl"],
     stats_vg_oos["t_stat"]),
    (f"VIX-gated (>{vix_gate_thresh_sel:.0f})", stats_vxg_sel["n"],
     stats_vxg_sel["avg_pnl"], stats_vxg_sel["t_stat"],
     stats_vxg_oos["n"], stats_vxg_oos["avg_pnl"], stats_vxg_oos["t_stat"]),
    ("Laggard-only (RS<0)", stats_lag_sel["n"], stats_lag_sel["avg_pnl"],
     stats_lag_sel["t_stat"], stats_lag_oos["n"], stats_lag_oos["avg_pnl"],
     stats_lag_oos["t_stat"]),
    (f"Squeeze-gated (>{squeeze_median:.2f})", stats_sq_sel["n"],
     stats_sq_sel["avg_pnl"], stats_sq_sel["t_stat"],
     stats_sq_oos["n"], stats_sq_oos["avg_pnl"], stats_sq_oos["t_stat"]),
]

print(f"\n  {'Variant':<30s} {'SEL_n':>6s} {'SEL_avg':>9s} {'SEL_t':>7s} "
      f"{'OOS_n':>6s} {'OOS_avg':>9s} {'OOS_t':>7s}")
print("  " + "-" * 78)
for row in summary_rows:
    name, sn, sa, st, on, oa, ot = row
    print(f"  {name:<30s} {sn:>6d} {sa:>+8.3f}% {st:>6.2f} "
          f"{on:>6d} {oa:>+8.3f}% {ot:>6.2f}")

# ══════════════════════════════════════════════════════════════════
# 10. LOG TRIALS
# ══════════════════════════════════════════════════════════════════
print("\n  Logging trials...")

for label, stats in [("baseline", baseline_oos),
                     ("vol_gated", stats_vg_oos),
                     ("vix_gated", stats_vxg_oos),
                     ("laggard_only", stats_lag_oos),
                     ("squeeze_gated", stats_sq_oos)]:
    log_trial("agent6", f"S5_zexit_{label}",
              {"k": K_ENTRY, "ze": Z_EXIT, "max_hold": MAX_HOLD,
               "gate": label},
              stats)

# ══════════════════════════════════════════════════════════════════
# 11. MATPLOTLIB FIGURE
# ══════════════════════════════════════════════════════════════════
print("\n  Generating figure...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Agent 6 — Attention & Positioning Proxies for S5_zexit (RXRX)",
             fontsize=14, fontweight="bold", y=0.98)

# --- Panel A: Volume attention proxy time series with signal overlay ---
ax = axes[0, 0]
mask_plot = spread_aug["date"].notna() & spread_aug["vol_z"].notna()
plot_data = spread_aug[mask_plot].copy()

ax.plot(plot_data["date"], plot_data["vol_z"], color="steelblue",
        alpha=0.5, linewidth=0.7, label="Volume z-score")
ax.axhline(y=VOL_Z_HIGH, color="red", linestyle="--", alpha=0.6,
           linewidth=1, label=f"High attention (z>{VOL_Z_HIGH})")
ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

# Overlay trade signal dates
if not trades_all.empty:
    for _, t in trades_all.iterrows():
        sig_date = t["signal_date"]
        match = plot_data[plot_data["date"] == sig_date]
        if len(match) > 0:
            vz_val = match["vol_z"].iloc[0]
            color = "green" if t["pnl_net_pct"] > 0 else "red"
            marker = "^" if t["side"] == "long" else "v"
            ax.scatter(sig_date, vz_val, c=color, marker=marker, s=40,
                       zorder=5, edgecolors="black", linewidth=0.5)

ax.set_title("(a) Volume Z-Score with Trade Signals", fontsize=11)
ax.set_ylabel("Volume Z-Score")
ax.legend(fontsize=8, loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=30, labelsize=8)
# Mark selection/OOS boundary
ax.axvline(x=pd.Timestamp(SEL_END), color="orange", linestyle="-",
           linewidth=1.5, alpha=0.7, label="OOS start")

# --- Panel B: PnL by attention bucket (high vs low vol attention) ---
ax = axes[0, 1]

valid_sel = trades_sel_aug.dropna(subset=["vol_z"])
valid_oos = trades_oos_aug.dropna(subset=["vol_z"])

categories = []
avgs = []
errors = []
colors_bar = []
n_labels = []

for period_label, df in [("SEL", valid_sel), ("OOS", valid_oos)]:
    for bucket, mask_fn, clr in [("Low", lambda x: x < VOL_Z_HIGH, "#4a90d9"),
                                  ("High", lambda x: x >= VOL_Z_HIGH, "#d94a4a")]:
        sub = df[mask_fn(df["vol_z"])]
        n = len(sub)
        avg = sub["pnl_net_pct"].mean() if n > 0 else 0
        se = sub["pnl_net_pct"].std() / np.sqrt(n) if n > 1 else 0
        categories.append(f"{period_label}\n{bucket}\n(n={n})")
        avgs.append(avg)
        errors.append(se)
        colors_bar.append(clr)

x_pos = np.arange(len(categories))
bars = ax.bar(x_pos, avgs, yerr=errors, capsize=4, color=colors_bar,
              edgecolor="black", linewidth=0.5, alpha=0.8)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=8)
ax.set_ylabel("Avg Net PnL (%)")
ax.set_title(f"(b) PnL by Volume Attention (z>={VOL_Z_HIGH} = High)", fontsize=11)

# --- Panel C: PnL by VIX regime ---
ax = axes[1, 0]

vix_all_results = []
for period_label, df in [("SEL", trades_sel_aug), ("OOS", trades_oos_aug)]:
    vix_result, _, _ = vix_regime_split(df)
    if not vix_result.empty:
        vix_result["period"] = period_label
        vix_all_results.append(vix_result)

if vix_all_results:
    vix_combined = pd.concat(vix_all_results, ignore_index=True)
    vix_cats = []
    vix_avgs = []
    vix_colors = []

    for _, row in vix_combined.iterrows():
        vix_cats.append(f"{row['period']}\n{row['regime']}\n(n={row['n']})")
        vix_avgs.append(row["avg_pnl"] if not pd.isna(row["avg_pnl"]) else 0)
        if "low" in row["regime"]:
            vix_colors.append("#66b3ff")
        elif "med" in row["regime"]:
            vix_colors.append("#ffd700")
        else:
            vix_colors.append("#ff6666")

    x_vix = np.arange(len(vix_cats))
    ax.bar(x_vix, vix_avgs, color=vix_colors, edgecolor="black",
           linewidth=0.5, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x_vix)
    ax.set_xticklabels(vix_cats, fontsize=7)

ax.set_ylabel("Avg Net PnL (%)")
ax.set_title("(c) PnL by VIX Regime (percentile-based terciles)", fontsize=11)

# --- Panel D: PnL by sector relative strength ---
ax = axes[1, 1]

rs_cats = []
rs_avgs = []
rs_errors = []
rs_colors = []

for period_label, df in [("SEL", trades_sel_aug), ("OOS", trades_oos_aug)]:
    valid = df.dropna(subset=["relative_strength"])
    for bucket, mask_fn, clr in [("Laggard\n(RS<0)", lambda x: x < 0, "#ff9999"),
                                  ("Leader\n(RS>0)", lambda x: x >= 0, "#99ff99")]:
        sub = valid[mask_fn(valid["relative_strength"])]
        n = len(sub)
        avg = sub["pnl_net_pct"].mean() if n > 0 else 0
        se = sub["pnl_net_pct"].std() / np.sqrt(n) if n > 1 else 0
        rs_cats.append(f"{period_label}\n{bucket}\n(n={n})")
        rs_avgs.append(avg)
        rs_errors.append(se)
        rs_colors.append(clr)

x_rs = np.arange(len(rs_cats))
ax.bar(x_rs, rs_avgs, yerr=rs_errors, capsize=4, color=rs_colors,
       edgecolor="black", linewidth=0.5, alpha=0.8)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xticks(x_rs)
ax.set_xticklabels(rs_cats, fontsize=7)
ax.set_ylabel("Avg Net PnL (%)")
ax.set_title("(d) PnL by Sector Relative Strength (RXRX vs ARKG 20d)", fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = OUT_DIR / "agent6_attention.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure saved: {fig_path}")

# ══════════════════════════════════════════════════════════════════
# 12. CONCLUSION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  CONCLUSION")
print("=" * 70)

# Determine which proxies show meaningful signal
findings = []

# Check each gated variant vs baseline
gated_variants = [
    ("Vol-gated", stats_vg_sel, stats_vg_oos),
    ("VIX-gated", stats_vxg_sel, stats_vxg_oos),
    ("Laggard-only", stats_lag_sel, stats_lag_oos),
    ("Squeeze-gated", stats_sq_sel, stats_sq_oos),
]

for name, sel_stats, oos_stats in gated_variants:
    sel_better = sel_stats["avg_pnl"] > baseline_sel["avg_pnl"]
    oos_better = oos_stats["avg_pnl"] > baseline_oos["avg_pnl"]
    sel_sufficient_n = sel_stats["n"] >= 10
    oos_sufficient_n = oos_stats["n"] >= 3

    if sel_better and oos_better and sel_sufficient_n:
        findings.append(f"  + {name}: IMPROVES avg_pnl in BOTH selection and OOS "
                        f"(SEL: {sel_stats['avg_pnl']:+.3f}% vs baseline {baseline_sel['avg_pnl']:+.3f}%, "
                        f"OOS: {oos_stats['avg_pnl']:+.3f}% vs baseline {baseline_oos['avg_pnl']:+.3f}%)")
    elif sel_better and not oos_better and sel_sufficient_n:
        findings.append(f"  ~ {name}: Improves selection but NOT OOS — possible overfit "
                        f"(SEL: {sel_stats['avg_pnl']:+.3f}%, OOS: {oos_stats['avg_pnl']:+.3f}%)")
    elif not sel_better:
        findings.append(f"  - {name}: No improvement even in selection "
                        f"(SEL: {sel_stats['avg_pnl']:+.3f}% vs baseline {baseline_sel['avg_pnl']:+.3f}%)")
    elif not sel_sufficient_n:
        findings.append(f"  ? {name}: Insufficient trades in selection (n={sel_stats['n']})")

print("\n  Proxy evaluation (+ = helps, ~ = ambiguous, - = hurts, ? = insufficient):\n")
for f in findings:
    print(f)

# Check proxy split t-stats from selection
print("\n  Selection window proxy split significance:")
for r in results_sel:
    name = r["proxy"]
    t = r.get("t_stat", np.nan)
    p = r.get("p_val", np.nan)
    sig = "***" if (not np.isnan(p) and p < 0.01) else \
          "**" if (not np.isnan(p) and p < 0.05) else \
          "*" if (not np.isnan(p) and p < 0.10) else ""
    t_str = f"{t:+.2f}" if not np.isnan(t) else "  N/A"
    p_str = f"{p:.3f}" if not np.isnan(p) else "N/A"
    print(f"    {name:<35s} t={t_str:>7s}  p={p_str:>6s}  {sig}")

# Final verdict
print(f"""
  FINAL VERDICT:
  - Baseline S5_zexit: SEL avg={baseline_sel['avg_pnl']:+.3f}% (n={baseline_sel['n']}),
    OOS avg={baseline_oos['avg_pnl']:+.3f}% (n={baseline_oos['n']})
  - None of the attention/positioning proxies should be adopted unless they
    show BOTH selection AND OOS improvement with t-stat > 1.5.
  - With only {baseline_oos['n']} OOS trades, proxy splits create very small
    subgroups — any apparent difference is likely noise.
  - RECOMMENDATION: Keep ungated S5_zexit as the primary signal.
    Log proxy values for future analysis as more OOS data accumulates.
""")

print("  AGENT 6 COMPLETE")
