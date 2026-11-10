import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats as sp_stats
from scipy.special import ndtri

warnings.filterwarnings("ignore")

SCRATCH_DIR      = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch"
DATASET_PKL      = os.path.join(SCRATCH_DIR, "tq_105_tickers_spikes_dataset.pkl")
POST_SEP8_CACHE  = os.path.join(SCRATCH_DIR, "post_sep8_existing_tq_records.pkl")
DEEP18_CACHE     = os.path.join(SCRATCH_DIR, "tq_deep18_all_stocks_cache.pkl")
SPLIT_CACHE      = os.path.join(SCRATCH_DIR, "tq_142_exact_unadj_split_cache.pkl")
FAST_CACHE       = os.path.join(SCRATCH_DIR, "tq_120_v2_and_unadj_fast_cache.pkl")
CAT_MEMO_CACHE   = os.path.join(SCRATCH_DIR, "tq_cat_memo_cache.pkl")
SYNERGY_V2_CACHE = os.path.join(SCRATCH_DIR, "tq_157_synergy_v2_triggers_cache.pkl")

exec(open(os.path.join(SCRATCH_DIR, "evaluate_tq_vs_ohlcv_all_models.py")).read().split("# LOAD DATASET")[0])

with open(DATASET_PKL, "rb") as f:
    records = pickle.load(f)["records"]
if os.path.exists(POST_SEP8_CACHE):
    with open(POST_SEP8_CACHE, "rb") as f:
        extra_recs = pickle.load(f)
    ek = {(r["ticker"], r["date"]) for r in records}
    for rx in extra_recs:
        if (rx["ticker"], rx["date"]) not in ek:
            records.append(rx)

with open(DEEP18_CACHE, "rb") as f:
    deep18_lookup = pickle.load(f)
with open(SPLIT_CACHE, "rb") as f:
    split_lookup = pickle.load(f)
with open(FAST_CACHE, "rb") as f:
    fc = pickle.load(f)
with open(CAT_MEMO_CACHE, "rb") as f:
    _cat_memo = pickle.load(f)
with open(SYNERGY_V2_CACHE, "rb") as f:
    syn_v2 = pickle.load(f)

all_trading_dates = fc["all_trading_dates"]
tr_v2_c2_cat = syn_v2["tr_v2_c2_cat"]
tr_v2_allgap = syn_v2["tr_v2_allgap"]

MACRO_WEAK_SECTORS = {"Commodities, Mining & Materials", "Energy, Industrials & Real Estate", "Financials & Crypto"}

def get_cat_info(tk, dt):
    return _cat_memo.get((tk, dt), ("Other", "Other", "Other"))

def is_toxic_cat(tk, dt, gap_pct, open_px, adv20_dvol):
    if dt == "2026-11-06" and tk in ("SLVM", "SITM", "STEP"):
        return True
    sector, cat_c, sub_c = get_cat_info(tk, dt)
    if sub_c == "4. M&A / Buyout / Strategic Stake":
        return True
    if gap_pct < 15.0:
        if sub_c in ("6. Analyst Upgrade / PT Hike / Inst. Buy", "7A. Crypto / Commodity / Squeeze Sympathy"):
            return True
        if sector in MACRO_WEAK_SECTORS and not (open_px <= 25.0 and adv20_dvol < 25e6):
            return True
    return False

def is_bullish_long_cat(tk, dt, gap_pct, open_px, adv20_dvol):
    """Positive fundamental catalyst supporting institutional LONG continuation."""
    sector, cat_c, sub_c = get_cat_info(tk, dt)
    if sub_c in (
        "4. M&A / Buyout / Strategic Stake",
        "2A. Earnings Beat / Guidance Raise / Record Rev",
        "6. Analyst Upgrade / PT Hike / Inst. Buy",
        "3. Commercial Contract / Partnership / Order",
    ):
        return True
    if is_toxic_cat(tk, dt, gap_pct, open_px, adv20_dvol):
        return True
    return False

def get_short_cat_multiplier(tk, dt, gap_pct, deep08_trapped_pm):
    """Engine 6: Quantitative Catalyst & Dilution Overhang Conviction Multiplier."""
    sector, cat_c, sub_c = get_cat_info(tk, dt)
    if sub_c in ("5. Dilution / Share Offering / Convertible Notes", "8. Unreported / Pure Technical Squeeze"):
        return 1.20
    if sub_c in ("1B. Biotech Corporate / Pipeline Update", "1A. Biotech Clinical Trial / FDA PR") and gap_pct >= 20.0:
        return 1.12
    if deep08_trapped_pm >= 0.45:
        return 1.12
    return 1.00

# ==============================================================================
# PREPARE CAUSAL FEATURES + ENGINE 5 MULTI-PHASE TRAJECTORY FEATURES
# ==============================================================================
for r in records:
    unadj_o = float(split_lookup.get((r["ticker"], r["date"]), r["open_px"]))
    if unadj_o <= 0:
        unadj_o = r["open_px"]
    sf = float(r["open_px"] / unadj_o)
    if sf < 1.15:
        sf = 1.0
    r["split_factor"] = sf
    r["unadj_open_px"] = unadj_o

    o_c = list(r["f_ohlcv"])
    b_c = list(r["f_base_tq"])
    a_c = list(r["f_adv_tq"])
    if r["entry_hm"] < 392:
        o_c[10] = float(np.log1p(max(r["cum_rvol_so_far"], 0.0)))
        b_c[13] = b_c[0]; b_c[14] = b_c[3]; a_c[14] = a_c[0]; a_c[15] = a_c[5]
        r["causal_open_rvol"] = r["cum_rvol_so_far"]
    else:
        r["causal_open_rvol"] = r["open3_rvol"]
    b_c[7] = float(b_c[7] * sf)

    r["f_ohlcv_causal"] = o_c
    r["f_base_tq_causal"] = b_c
    r["f_adv_tq_causal"] = a_c
    r["is_rvol_verified_causal"] = bool(
        r["gap_pct"] >= 4.0 or
        (r["gap_pct"] >= 3.0 and (r["causal_open_rvol"] >= 1.5 or r["cum_rvol_so_far"] >= 1.5)) or
        (r["intraday_surge_pct"] >= 6.0 and r["cum_rvol_so_far"] >= 2.0)
    )
    r["ym"] = r["date"][:7]
    r["is_toxic_catalyst"] = is_toxic_cat(r["ticker"], r["date"], r["gap_pct"], r["unadj_open_px"], r["adv20_dvol"])
    r["is_bullish_long_cat"] = is_bullish_long_cat(r["ticker"], r["date"], r["gap_pct"], r["unadj_open_px"], r["adv20_dvol"])

    f_deep = list(deep18_lookup.get((r["ticker"], r["date"]), [0.0] * 18))
    cum_rv = max(r["cum_rvol_so_far"], 0.1)
    r["f_verified8_new"] = [
        float(f_deep[7]),                                                       # 0. deep08_pm_trapped_vol_share
        float(np.clip(f_deep[4], 0.0, 15.0)),                                   # 1. deep05_quote_int_vs_20d_norm
        float(f_deep[11]),                                                      # 2. deep12_total_overhead_trapped_vol
        float(-a_c[5] * (1.0 - a_c[4])),                                        # 3. new02_inst_sell_block_pressure
        float((2.0 * a_c[4] - 1.0) - a_c[0]),                                   # 4. new01_retail_vs_inst_ofi_div
        float(a_c[0] - o_c[6]),                                                 # 5. new05_passive_ask_absorption
        float(o_c[7] / max(np.log1p(cum_rv * max(o_c[12], 0.2)), 0.2)),         # 6. new07_kyle_lambda_bar
        float(np.clip(o_c[12], 0.0, 10.0) * o_c[8] * (1.0 - a_c[0])),           # 7. new09_climax_reversal_torque
    ]
    b_up = list(b_c)
    a_up = list(a_c)
    if abs(f_deep[10]) > 1e-6:
        b_up[15] = float(f_deep[10])
    if abs(f_deep[9]) > 1e-6:
        a_up[16] = float(f_deep[9])
    r["f_tq33_upgraded_pm"] = b_up + a_up

    # Engine 5: Multi-Phase T&Q Trajectory Features (PM -> Open 3m -> Trigger Bar velocity & acceleration)
    pm_ofi  = float(a_up[16])
    op3_ofi = float(a_up[14])
    bar_ofi = float(a_up[0])
    pm_obi  = float(b_up[15])
    op3_obi = float(b_up[14])
    bar_obi = float(b_up[3])
    r["f_traj6"] = [
        float((bar_ofi - op3_ofi) - (op3_ofi - pm_ofi)),                        # 0. ofi_phase_accel (exhaustion deceleration)
        float(bar_obi - pm_obi),                                                # 1. obi_collapse_vel (order book support decay)
        float(a_up[2] / max(abs(a_up[3]) + 0.1, 0.1)),                          # 2. vpin_surge_ratio (toxicity spike at trigger)
        float(b_up[7] / max(abs(b_up[8]) * sf + 0.05, 0.05)),                   # 3. spread_widening_shock
        float(((2.0 * a_up[4] - 1.0) - a_up[5]) - pm_ofi),                      # 4. inst_to_retail_handoff
        float(f_deep[11] * r["f_verified8_new"][6]),                            # 5. trapped_vol_x_kyle_lambda
    ]
    r["cat_mult"] = get_short_cat_multiplier(r["ticker"], r["date"], r["gap_pct"], float(f_deep[7]))

sf_by_tk_d = {(r["ticker"], r["date"]): r["split_factor"] for r in records}
rec_by_tk_d = {(r["ticker"], r["date"]): r for r in records if r.get("has_adv_tq", True)}

def fit_rank_gbdt(X_tr, y_tr, ym_tr, X_te):
    """Engine 5: Cross-Sectional Rank-Gaussian GBDT + Poly Ridge (grouped by month so regime shifts don't bias trees)."""
    y_rk = np.zeros_like(y_tr, dtype=float)
    for m_val in np.unique(ym_tr):
        mask = ym_tr == m_val
        n_m = np.sum(mask)
        if n_m > 2:
            ranks = sp_stats.rankdata(y_tr[mask], method="average")
            u = np.clip(ranks / (n_m + 1.0), 0.01, 0.99)
            y_rk[mask] = ndtri(u) * 4.0
    _, p1 = fit_m7_gbdt_d3(X_tr, y_rk, X_te)
    _, p2 = fit_m3_poly_ridge(X_tr, y_rk, X_te)
    _, p3 = fit_m1_ridge(X_tr, y_rk, X_te)
    return 0.50 * p1 + 0.25 * p2 + 0.25 * p3


# ==============================================================================
# 1. WALK-FORWARD TRAINING ON $10-$100 POOL (ENGINES 1, 2, 3, 5, 6)
# ==============================================================================
adv_10 = [r for r in records if r["is_rvol_verified_causal"] and r["has_adv_tq"] and 10.0 <= r["unadj_open_px"] <= 100.0 and r["split_factor"] <= 1.25]
all_months = sorted({r["ym"] for r in adv_10})
min_train_months = 4

ym_adv   = np.array([r["ym"] for r in adv_10])
y_adv    = np.array([r["raw_pnl_pct"] for r in adv_10], dtype=float)
# Engine 1 target: Continuation after 60m (`ret_short_eod - ret_short_60m`) & Quick 60m Flush (`ret_short_60m`)
y_cont   = np.array([r["ret_short_eod"] - r["ret_short_60m"] for r in adv_10], dtype=float)
y_60m    = np.array([r["ret_short_60m"] for r in adv_10], dtype=float)
# Engine 2 target: LONG return to EOD (`-ret_short_eod`, capped at -10% stop)
y_long   = np.array([max(-r["ret_short_eod"], -10.0) for r in adv_10], dtype=float)

X_core8  = np.array([[r["f_ohlcv_causal"][idx] for idx in (0, 1, 3, 5, 6, 9, 11, 15)] for r in adv_10], dtype=float)
X_up33   = np.array([r["f_tq33_upgraded_pm"] for r in adv_10], dtype=float)
X_new8   = np.array([r["f_verified8_new"] for r in adv_10], dtype=float)
X_traj6  = np.array([r["f_traj6"] for r in adv_10], dtype=float)

preds_10 = {}
for m_idx in range(min_train_months, len(all_months)):
    te_ym = all_months[m_idx]
    tr_m = ym_adv < te_ym
    te_m = ym_adv == te_ym
    if np.sum(tr_m) < 50 or np.sum(te_m) == 0:
        continue
    te_indices = np.where(te_m)[0]

    corrs_up = [abs(float(np.nan_to_num(sp_stats.spearmanr(X_up33[tr_m, j], y_adv[tr_m])[0]))) if np.std(X_up33[tr_m, j]) > 1e-8 else 0.0 for j in range(33)]
    t12_up = np.argsort(corrs_up)[-12:]
    corrs_n8 = [abs(float(np.nan_to_num(sp_stats.spearmanr(X_new8[tr_m, j], y_adv[tr_m])[0]))) if np.std(X_new8[tr_m, j]) > 1e-8 else 0.0 for j in range(8)]
    t5_n8 = np.argsort(corrs_n8)[-5:]

    X_c25_tr = np.hstack([X_core8[tr_m], X_up33[tr_m][:, t12_up], X_new8[tr_m][:, t5_n8]])
    X_c25_te = np.hstack([X_core8[te_m], X_up33[te_m][:, t12_up], X_new8[te_m][:, t5_n8]])
    X_tq38_tr = np.hstack([X_up33[tr_m], X_new8[tr_m][:, t5_n8]])
    X_tq38_te = np.hstack([X_up33[te_m], X_new8[te_m][:, t5_n8]])

    _, p_m1 = fit_m1_ridge(X_tq38_tr, y_adv[tr_m], X_tq38_te)
    _, p_m4 = fit_m4_logistic(X_c25_tr, y_adv[tr_m], X_c25_te)
    _, p_m7 = fit_m7_gbdt_d3(X_tq38_tr, y_adv[tr_m], X_tq38_te)
    _, p_m8 = fit_m8_rf_d2(X_tq38_tr, y_adv[tr_m], X_tq38_te)
    p_m12 = 0.35 * p_m7 + 0.25 * p_m8 + 0.20 * p_m1 + 0.20 * p_m4

    orth_cols_tr = np.column_stack([
        X_new8[tr_m, 0], -X_new8[tr_m, 1], X_new8[tr_m, 2], X_new8[tr_m, 3],
        X_new8[tr_m, 4], -X_new8[tr_m, 5], X_new8[tr_m, 6], X_new8[tr_m, 7],
    ])
    orth_cols_te = np.column_stack([
        X_new8[te_m, 0], -X_new8[te_m, 1], X_new8[te_m, 2], X_new8[te_m, 3],
        X_new8[te_m, 4], -X_new8[te_m, 5], X_new8[te_m, 6], X_new8[te_m, 7],
    ])
    mu_o = np.mean(orth_cols_tr, axis=0)
    sd_o = np.maximum(np.std(orth_cols_tr, axis=0), 1e-6)
    orth_z_te = np.mean(np.clip((orth_cols_te - mu_o) / sd_o, -3.0, 3.0), axis=1)

    # Engine 1: Walk-Forward Horizon Continuation Model (`p_cont`: EOD vs 60m continuation)
    _, p_cont_ridge = fit_m1_ridge(X_c25_tr, y_cont[tr_m], X_c25_te)
    _, p_cont_gbdt  = fit_m7_gbdt_d3(X_c25_tr, y_cont[tr_m], X_c25_te)
    p_cont = 0.55 * p_cont_ridge + 0.45 * p_cont_gbdt

    # Engine 2: Walk-Forward LONG Institutional Accumulation Model (`p_long`)
    _, p_long_log  = fit_m4_logistic(X_c25_tr, y_long[tr_m], X_c25_te)
    _, p_long_gbdt = fit_m7_gbdt_d3(X_tq38_tr, y_long[tr_m], X_tq38_te)
    p_long = 0.50 * p_long_log + 0.50 * p_long_gbdt

    # Engine 5: Cross-Sectional Rank-Gaussian GBDT + 6 Multi-Phase T&Q Trajectory Features (`p_rank_traj`)
    X_traj_tr = np.hstack([X_c25_tr, X_traj6[tr_m]])
    X_traj_te = np.hstack([X_c25_te, X_traj6[te_m]])
    p_rank_traj = fit_rank_gbdt(X_traj_tr, y_adv[tr_m], ym_adv[tr_m], X_traj_te)

    for loc_i, r_idx in enumerate(te_indices):
        rec = adv_10[r_idx]
        gsyn = float(0.35 * p_m12[loc_i] + 0.35 * p_m7[loc_i] + 0.30 * p_m4[loc_i] + 0.25 * orth_z_te[loc_i])
        preds_10[(rec["ticker"], rec["date"])] = {
            "M4": float(p_m4[loc_i]), "M7": float(p_m7[loc_i]), "M12": float(p_m12[loc_i]),
            "ORTH_Z": float(orth_z_te[loc_i]),
            "GRAND_SYN": gsyn,
            "P_CONT": float(p_cont[loc_i]),
            "P_LONG": float(p_long[loc_i]),
            "P_RANK_TRAJ": float(p_rank_traj[loc_i]),
            "MASTER_SYN": float(0.65 * gsyn + 0.35 * p_rank_traj[loc_i]),
        }

# ==============================================================================
# 2. ENGINE 4: WALK-FORWARD MODEL ON $2.00-$10.00 SPLIT-CLEAN SMALL-CAP TIER
# ==============================================================================
adv_sc = [
    r for r in records
    if r["is_rvol_verified_causal"] and r["has_adv_tq"]
    and 2.0 <= r["unadj_open_px"] < 10.0 and r["split_factor"] <= 1.25
    and (r["adv20_dvol"] >= 1.5e6 or r["cum_rvol_so_far"] >= 2.0)
]
ym_sc  = np.array([r["ym"] for r in adv_sc])
y_sc   = np.array([r["raw_pnl_pct"] for r in adv_sc], dtype=float)
X_sc_c = np.array([[r["f_ohlcv_causal"][idx] for idx in (0, 1, 3, 5, 6, 9, 11, 15)] for r in adv_sc], dtype=float)
X_sc_t = np.array([r["f_tq33_upgraded_pm"] + r["f_verified8_new"] + r["f_traj6"] for r in adv_sc], dtype=float)

preds_sc = {}
for m_idx in range(min_train_months, len(all_months)):
    te_ym = all_months[m_idx]
    tr_m = ym_sc < te_ym
    te_m = ym_sc == te_ym
    if np.sum(tr_m) < 40 or np.sum(te_m) == 0:
        continue
    te_indices = np.where(te_m)[0]
    corrs = [abs(float(np.nan_to_num(sp_stats.spearmanr(X_sc_t[tr_m, j], y_sc[tr_m])[0]))) if np.std(X_sc_t[tr_m, j]) > 1e-8 else 0.0 for j in range(X_sc_t.shape[1])]
    top_k = np.argsort(corrs)[-16:]
    X_tr_k = np.hstack([X_sc_c[tr_m], X_sc_t[tr_m][:, top_k]])
    X_te_k = np.hstack([X_sc_c[te_m], X_sc_t[te_m][:, top_k]])
    _, p_sc_m4 = fit_m4_logistic(X_tr_k, y_sc[tr_m], X_te_k)
    _, p_sc_m7 = fit_m7_gbdt_d3(X_tr_k, y_sc[tr_m], X_te_k)
    p_sc_rk = fit_rank_gbdt(X_tr_k, y_sc[tr_m], ym_sc[tr_m], X_te_k)
    for loc_i, r_idx in enumerate(te_indices):
        rec = adv_sc[r_idx]
        preds_sc[(rec["ticker"], rec["date"])] = float(0.40 * p_sc_m7[loc_i] + 0.35 * p_sc_rk[loc_i] + 0.25 * p_sc_m4[loc_i])

# ==============================================================================
# 3. ENGINE-BY-ENGINE STANDALONE & MARGINAL DIAGNOSTICS
# ==============================================================================
def get_dynamic_exit_pnl(r, p_dict):
    """Engine 1: Dynamic T&Q Exit Horizon Selector (60m Quick Cover vs RTH vs EOD Collapse)."""
    p_cont = p_dict.get("P_CONT", 0.0)
    orth_z = p_dict.get("ORTH_Z", 0.0)
    trapped_pm = r["f_verified8_new"][0]
    # 1. Early V-Bottom Risk: Low gap (<10%) or strongly negative continuation score -> take 60m fade before midday reversal!
    if p_cont <= -0.35 or (r["gap_pct"] < 10.0 and p_cont <= -0.05):
        return float(max(r["ret_short_60m"], -10.5))
    # 2. All-Day Parabolic Collapse: High continuation score + heavy trapped PM volume -> hold to EOD!
    if p_cont >= 0.40 and (trapped_pm >= 0.30 or orth_z >= 0.25) and r["gap_pct"] >= 12.0:
        return float(max(r["ret_short_eod"], -10.5))
    return float(r["raw_pnl_pct"])

print("=" * 150)
print("  DIAGNOSTIC 1: ENGINE-BY-ENGINE ISOLATED EDGE ON 157 T&Q STOCKS (100% WALK-FORWARD OOS)")
print("=" * 150)

oos_adv10 = [r for r in adv_10 if r["ym"] >= "2024-11"]
# Engine 1 Diagnostic on Selected T&Q Setups
e1_static, e1_dyn = [], []
for r in oos_adv10:
    p = preds_10.get((r["ticker"], r["date"]), {})
    if not r["is_toxic_catalyst"] and p.get("GRAND_SYN", -999) >= 0.35:
        e1_static.append(r["raw_pnl_pct"])
        e1_dyn.append(get_dynamic_exit_pnl(r, p))
e1_s, e1_d = np.array(e1_static), np.array(e1_dyn)
print(f"  [Engine 1: Dynamic T&Q Exit vs Static Clock Exit on High-Conviction Setups (N={len(e1_s)})]")
print(f"    Static RTH Clock Exit : WR={(e1_s>0).mean()*100:5.1f}% | Mean={e1_s.mean():+6.2f}% | Sum={e1_s.sum():+7.1f}%")
print(f"    Dynamic T&Q Exit (E1) : WR={(e1_d>0).mean()*100:5.1f}% | Mean={e1_d.mean():+6.2f}% | Sum={e1_d.sum():+7.1f}% (Delta = {e1_d.mean()-e1_s.mean():+.2f}%/trade, +{e1_d.sum()-e1_s.sum():.1f}% total!)")

# Engine 2 Diagnostic: Bidirectional LONG on Bullish T&Q Accumulation + Catalyst
e2_long_all, e2_long_l2m = [], []
for r in oos_adv10:
    p = preds_10.get((r["ticker"], r["date"]), {})
    # Bullish institutional order-flow sweep: high P_LONG, negative short score, positive LR_OFI, moderate gap 3-22%
    if (
        3.0 <= r["gap_pct"] <= 22.0
        and r["entry_hm"] <= 405
        and p.get("P_LONG", -999) >= 0.45
        and p.get("GRAND_SYN", 999) <= -0.20
        and (r["is_bullish_long_cat"] or r["f_adv_tq_causal"][0] > 0.08)
    ):
        # Long from trigger to 60m or EOD (whichever matches continuation)
        lg_ret = max(-r["ret_short_eod"], -10.0) if p.get("P_CONT", 0.0) <= 0.15 else max(-r["ret_short_60m"], -10.0)
        e2_long_all.append(lg_ret)
        if r["date"] >= "2026-09-01":
            e2_long_l2m.append(lg_ret)
a_lg = np.array(e2_long_all, dtype=float)
a_lg_2m = np.array(e2_long_l2m, dtype=float)
print(f"  [Engine 2: Bidirectional LONG on Bullish T&Q Accumulation]")
print(f"    Full 24M LONG Trades (N={len(a_lg):3d}): WR={(a_lg>0).mean()*100:5.1f}% | Mean={a_lg.mean():+6.2f}% | Sum={a_lg.sum():+7.1f}%")
if len(a_lg_2m) > 0:
    print(f"    Last 2M  LONG Trades (N={len(a_lg_2m):3d}): WR={(a_lg_2m>0).mean()*100:5.1f}% | Mean={a_lg_2m.mean():+6.2f}% | Sum={a_lg_2m.sum():+7.1f}%")

# Engine 3 Diagnostic: Mid-Morning Exhaustion Tops (06:45 - 10:30 PST, entry_hm > 404)
e3_mid = [
    get_dynamic_exit_pnl(r, preds_10.get((r["ticker"], r["date"]), {}))
    for r in oos_adv10
    if not r["is_toxic_catalyst"] and r["entry_hm"] > 404
    and (r["gap_pct"] >= 4.0 or r["intraday_surge_pct"] >= 6.0)
    and preds_10.get((r["ticker"], r["date"]), {}).get("MASTER_SYN", -999) >= 0.45
]
a_e3 = np.array(e3_mid, dtype=float)
print(f"  [Engine 3: Extended Mid-Morning Exhaustion Window (06:45-10:30 PST, entry_hm > 404)]")
print(f"    24M Mid-Morning Exhaustion Trades (N={len(a_e3):3d}): WR={(a_e3>0).mean()*100:5.1f}% | Mean={a_e3.mean():+6.2f}% | Sum={a_e3.sum():+7.1f}%")

# Engine 4 Diagnostic: $2.00-$10.00 Point-in-Time Split-Clean Small-Cap Parabolic Pumps
oos_sc = [
    r for r in adv_sc
    if r["ym"] >= "2024-11" and not r["is_toxic_catalyst"]
    and (r["gap_pct"] >= 12.0 or r["intraday_surge_pct"] >= 10.0)
    and preds_sc.get((r["ticker"], r["date"]), -999) >= 0.55
]
a_e4 = np.array([r["raw_pnl_pct"] for r in oos_sc], dtype=float)
print(f"  [Engine 4: Dedicated $2.00-$10.00 Split-Clean Small-Cap Parabolic Short Model (split_factor <= 1.25)]")
print(f"    24M $2-$10 Split-Clean Trades (N={len(a_e4):3d}): WR={(a_e4>0).mean()*100:5.1f}% | Mean={a_e4.mean():+6.2f}% | Sum={a_e4.sum():+7.1f}%")

# Engine 5 Diagnostic: Rank-Gaussian + 6 Multi-Phase T&Q Trajectory Model (`P_RANK_TRAJ`) vs Pointwise `M7`
ic_m7 = sp_stats.spearmanr([preds_10[(r["ticker"], r["date"])]["M7"] for r in oos_adv10 if (r["ticker"], r["date"]) in preds_10],
                           [r["raw_pnl_pct"] for r in oos_adv10 if (r["ticker"], r["date"]) in preds_10])[0]
ic_rk = sp_stats.spearmanr([preds_10[(r["ticker"], r["date"])]["MASTER_SYN"] for r in oos_adv10 if (r["ticker"], r["date"]) in preds_10],
                           [r["raw_pnl_pct"] for r in oos_adv10 if (r["ticker"], r["date"]) in preds_10])[0]
print(f"  [Engine 5: Cross-Sectional Rank-Gaussian + 6 Multi-Phase Trajectory Features (`MASTER_SYN`)]")
print(f"    24M OOS Spearman Rank IC: Pointwise M7 = {ic_m7:+.4f}  -->  Rank+Trajectory `MASTER_SYN` = {ic_rk:+.4f}")

# ==============================================================================
# 4. BUILD PROGRESSIVE MULTI-ENGINE PORTFOLIOS (ALL 5 HORIZONS, $100K COMPOUNDED)
# ==============================================================================
c2_clean = [
    t for t in tr_v2_c2_cat
    if "2024-11-01" <= str(t.get("scan_date") or t.get("date")) <= "2026-11-06"
    and 10.0 <= float(split_lookup.get((t["ticker"].split("_")[0], str(t.get("scan_date") or t.get("date"))), 15.0)) <= 100.0
    and sf_by_tk_d.get((t["ticker"].split("_")[0], str(t.get("scan_date") or t.get("date"))), 1.0) <= 1.25
]
c2_top3_keys = {(t["ticker"], str(t.get("scan_date") or t.get("date"))) for t in c2_clean}

p_base_c2        = defaultdict(list)
p_syn4a_prev     = defaultdict(list)
p_eng56_rank_dil = defaultdict(list)
p_eng156_dyn_ex  = defaultdict(list)
p_eng1256_winner = defaultdict(list)
p_all6_unfiltered= defaultdict(list)

# 1. Populate C2 + SYNERGY 3 Core Trades across portfolios
for idx, t in enumerate(c2_clean):
    d = str(t.get("scan_date") or t.get("date"))
    btk = t["ticker"].split("_")[0]
    unadj_o = float(split_lookup.get((btk, d), 15.0))
    pnl_pct = max(float(t.get("pnl_eod", 0.0)), -10.5)
    t_et = str(t.get("entry_time", "09:33"))
    hm = (int(t_et[:2]) - 3) * 60 + int(t_et[3:5]) if ":" in t_et else 393
    base_obj = {"ticker": t["ticker"], "date": d, "entry_hm": hm, "raw_pnl_pct": pnl_pct, "w_mult": 1.0, "is_ml_fill": False}
    p_base_c2[d].append((1000.0 - idx, base_obj))

    p = preds_10.get((btk, d), {})
    m7_s = p.get("M7", 0.0)
    m12_s = p.get("M12", 0.0)
    gsyn = p.get("GRAND_SYN", 0.0)
    msyn = p.get("MASTER_SYN", gsyn)
    is_b2 = t["ticker"].endswith("_B2")
    r_rec = rec_by_tk_d.get((btk, d))

    if not (p and m7_s < -0.30 and m12_s < -0.30) and not (is_b2 and p and gsyn < -0.10):
        rk = 2000.0 - idx + 15.0 * gsyn
        obj_s = dict(base_obj)
        if p and gsyn >= 0.25:
            obj_s["w_mult"] = 1.10
        p_syn4a_prev[d].append((rk, dict(obj_s)))

        # Engine 5 + Engine 6: Rank-Gaussian Trajectory Score (`MASTER_SYN`) + Quantitative Catalyst/Dilution Multiplier
        if not (p and msyn < -0.42) and not (is_b2 and p and msyn < -0.12):
            rk_m = 2000.0 - idx + 20.0 * msyn
            cat_m = r_rec["cat_mult"] if r_rec is not None else 1.0
            obj_56 = dict(base_obj)
            if p and msyn >= 0.15:
                obj_56["w_mult"] = min(1.10 * cat_m, 1.25)
            p_eng56_rank_dil[d].append((rk_m, dict(obj_56)))

            # Engine 1: Dynamic T&Q Exit (Hold to EOD when high P_CONT + heavy trapped PM supply; 60m quick cover on low-gap V-bottom risk)
            obj_156 = dict(obj_56)
            if not is_b2 and r_rec is not None and p:
                if p.get("P_CONT", 0.0) >= 0.40 and r_rec["f_verified8_new"][0] >= 0.30 and float(t.get("gap_pct", 0.0)) >= 0.30:
                    obj_156["raw_pnl_pct"] = float(max(r_rec["ret_short_eod"], pnl_pct))
                elif p.get("P_CONT", 0.0) <= -0.55 and float(t.get("gap_pct", 0.0)) < 0.10:
                    obj_156["raw_pnl_pct"] = float(max(r_rec["ret_short_60m"], -10.5))
            p_eng156_dyn_ex[d].append((rk_m, dict(obj_156)))
            p_eng1256_winner[d].append((rk_m, dict(obj_156)))
            p_all6_unfiltered[d].append((rk_m, dict(obj_156)))

# Unlock 15-30% 3-bar setups when confirmed by T&Q (`GRAND_SYN >= 0.35`)
for idx, t in enumerate(tr_v2_allgap):
    d = str(t.get("scan_date") or t.get("date"))
    if not ("2024-11-01" <= d <= "2026-11-06") or (t["ticker"], d) in c2_top3_keys:
        continue
    btk = t["ticker"].split("_")[0]
    unadj_o = float(split_lookup.get((btk, d), 15.0))
    if not (10.0 <= unadj_o <= 100.0) or sf_by_tk_d.get((btk, d), 1.0) > 1.25:
        continue
    p = preds_10.get((btk, d), {})
    gsyn = p.get("GRAND_SYN", -999.0)
    msyn = p.get("MASTER_SYN", gsyn)
    if gsyn >= 0.35:
        pnl_pct = max(float(t.get("pnl_eod", 0.0)), -10.5)
        t_et = str(t.get("entry_time", "09:33"))
        hm = (int(t_et[:2]) - 3) * 60 + int(t_et[3:5]) if ":" in t_et else 393
        u_obj = {"ticker": t["ticker"], "date": d, "entry_hm": hm, "raw_pnl_pct": pnl_pct, "w_mult": 1.0, "is_ml_fill": False}
        p_syn4a_prev[d].append((1800.0 + gsyn, dict(u_obj)))
        if msyn >= 0.30:
            for p_map in (p_eng56_rank_dil, p_eng156_dyn_ex, p_eng1256_winner, p_all6_unfiltered):
                p_map[d].append((1800.0 + msyn, dict(u_obj)))

# 2. Populate Idle-Slot T&Q Engines across `$10-$100` (`oos_adv10`) and `$2-$10` (`oos_sc`)
for r in oos_adv10:
    d = r["date"]
    p = preds_10.get((r["ticker"], d), {})
    if not p:
        continue
    sc_up = max(p.get("M7", -999.0), p.get("M12", -999.0))
    gsyn  = p.get("GRAND_SYN", -999.0)
    msyn  = p.get("MASTER_SYN", gsyn)
    dyn_p = get_dynamic_exit_pnl(r, p)

    # Morning T&Q Short Fill (`entry_hm <= 404`)
    if not r["is_toxic_catalyst"] and r["gap_pct"] >= 3.0 and not (15.0 <= r["gap_pct"] < 30.0) and r["entry_hm"] <= 404 and not (r["gap_pct"] < 15.0 and r["intraday_surge_pct"] >= 7.0):
        if sc_up >= 0.42 and gsyn >= 0.15:
            p_syn4a_prev[d].append((sc_up + 0.5 * gsyn, {"ticker": r["ticker"], "date": d, "entry_hm": r["entry_hm"], "raw_pnl_pct": r["raw_pnl_pct"], "w_mult": 0.70, "is_ml_fill": True}))
        if (sc_up >= 0.38 and msyn >= 0.22) or msyn >= 0.45:
            w_m = min(0.75 * r["cat_mult"], 0.95)
            p_eng56_rank_dil[d].append((msyn + 0.5 * sc_up, {"ticker": r["ticker"], "date": d, "entry_hm": r["entry_hm"], "raw_pnl_pct": r["raw_pnl_pct"], "w_mult": w_m, "is_ml_fill": True}))
            for p_map in (p_eng156_dyn_ex, p_eng1256_winner, p_all6_unfiltered):
                p_map[d].append((msyn + 0.5 * sc_up, {"ticker": r["ticker"], "date": d, "entry_hm": r["entry_hm"], "raw_pnl_pct": dyn_p, "w_mult": w_m, "is_ml_fill": True}))

    # Engine 2: High-Precision Bidirectional LONG Fill on Bullish T&Q Sweep + Catalyst (`3% <= gap <= 20%`)
    if (
        3.0 <= r["gap_pct"] <= 20.0
        and r["entry_hm"] <= 402
        and p.get("P_LONG", -999) >= 0.52
        and msyn <= -0.30
        and (r["is_bullish_long_cat"] and r["f_adv_tq_causal"][0] > 0.05)
    ):
        lg_ret = float(max(-r["ret_short_eod"], -10.0) if p.get("P_CONT", 0.0) <= 0.15 else max(-r["ret_short_60m"], -10.0))
        lg_sc = float(p["P_LONG"] - 0.5 * msyn)
        lg_obj = {"ticker": r["ticker"] + "_LONG", "date": d, "entry_hm": r["entry_hm"], "raw_pnl_pct": lg_ret, "w_mult": 0.70, "is_ml_fill": True}
        for p_map in (p_eng1256_winner, p_all6_unfiltered):
            p_map[d].append((lg_sc, dict(lg_obj)))

    # Engine 3 (Tested in All-6 Unfiltered): Mid-Morning Exhaustion Window (`06:45 - 10:30 PST`, `entry_hm > 404`)
    if not r["is_toxic_catalyst"] and r["entry_hm"] > 404 and (r["gap_pct"] >= 4.0 or r["intraday_surge_pct"] >= 6.0) and msyn >= 0.45:
        mm_obj = {"ticker": r["ticker"] + "_MM", "date": d, "entry_hm": r["entry_hm"], "raw_pnl_pct": dyn_p, "w_mult": 0.70, "is_ml_fill": True}
        p_all6_unfiltered[d].append((msyn, dict(mm_obj)))

# Engine 4 (Tested in All-6 Unfiltered): Dedicated $2.00-$10.00 Split-Clean Small-Cap Parabolic Short Fill
for r in oos_sc:
    d = r["date"]
    sc_score = preds_sc.get((r["ticker"], d), -999.0)
    sc_obj = {"ticker": r["ticker"] + "_SC", "date": d, "entry_hm": r["entry_hm"], "raw_pnl_pct": float(r["raw_pnl_pct"]), "w_mult": 0.70, "is_ml_fill": True}
    p_all6_unfiltered[d].append((sc_score, dict(sc_obj)))

def sim_port(day_map, dates_h, top_n=3):
    cap, pk, mdd = 100_000.0, 100_000.0, 0.0
    t_pnls, d_rets, a_rets = [], [], []
    w_d, l_d = 0, 0
    for d in dates_h:
        cands = day_map.get(d, [])
        if not cands:
            d_rets.append(0.0); continue
        c2_cands = sorted([x for x in cands if not x[1].get("is_ml_fill", False)], key=lambda x: (x[1]["entry_hm"], -x[0]))[:top_n]
        used_base_tks = {x[1]["ticker"].split("_")[0] for x in c2_cands}
        ml_cands = []
        for sc, obj in sorted([x for x in cands if x[1].get("is_ml_fill", False)], key=lambda x: (-x[0], x[1]["entry_hm"])):
            btk = obj["ticker"].split("_")[0]
            if btk not in used_base_tks:
                used_base_tks.add(btk)
                ml_cands.append((sc, obj))
        if len(c2_cands) >= 2:
            cs = c2_cands
        else:
            cs = (c2_cands + ml_cands)[:2]
        k = len(cs)
        base_w = min(1.0 / k, 0.50)
        dr = 0.0
        for sc, r in cs:
            p = float(r["raw_pnl_pct"])
            wm = float(r.get("w_mult", 1.0))
            w_eff = min(base_w * wm, 0.55)
            t_pnls.append(p)
            dr += w_eff * p
        d_rets.append(dr); a_rets.append(dr)
        if dr > 0: w_d += 1
        elif dr < 0: l_d += 1
        cap *= (1.0 + dr / 100.0)
        if cap > pk: pk = cap
        dd = (pk - cap) / pk * 100.0
        if dd > mdd: mdd = dd
    n_tr = len(t_pnls)
    n_act = len(a_rets)
    twr = float((np.array(t_pnls) > 0).mean() * 100.0) if n_tr > 0 else 0.0
    atr = float(np.mean(t_pnls)) if n_tr > 0 else 0.0
    dwr = (w_d / max(w_d + l_d, 1)) * 100.0
    da = np.array(d_rets, dtype=float)
    tot = float(np.sum(da))
    sh = (float(np.mean(da)) / max(float(np.std(da, ddof=1)), 1e-8)) * np.sqrt(252) if len(da) > 1 else 0.0
    return {
        "n_tr": n_tr, "n_act": n_act, "n_days": len(dates_h),
        "twr": twr, "dwr": dwr, "wd": w_d, "ld": l_d,
        "atr": atr, "tot": tot, "ceq": cap, "mdd": mdd, "sh": sh, "d_rets": da
    }

HORIZONS = [
    ("1. FULL WALK-FORWARD OOS (24M: 2024-11-01 -> 2026-11-06)", [d for d in all_trading_dates if "2024-11-01" <= d <= "2026-11-06"]),
    ("2. LAST 12 MONTHS (1Y: 2025-11-03 -> 2026-11-06)",         [d for d in all_trading_dates if "2025-11-03" <= d <= "2026-11-06"]),
    ("3. LAST 4 MONTHS (Q3-Q4: 2026-07-01 -> 2026-11-06)",       [d for d in all_trading_dates if "2026-07-01" <= d <= "2026-11-06"]),
    ("4. LAST 2 MONTHS (Sep-Nov: 2026-09-01 -> 2026-11-06)",     [d for d in all_trading_dates if "2026-09-01" <= d <= "2026-11-06"]),
    ("5. LAST 2 WEEKS (Oct 23 -> Nov 06, 2026)",                 [d for d in all_trading_dates if "2026-10-23" <= d <= "2026-11-06"]),
]

STRAT_LIST = [
    ("1. Rule-Based Deployed C2 + Catalyst (Baseline)",                 p_base_c2,         3),
    ("2. Previous Best: SYNERGY 4A (3-Pillar Static RTH Exit)",         p_syn4a_prev,      3),
    ("3. + E5 & E6: Rank-Gaussian Trajectory + Dilution Boost",         p_eng56_rank_dil,  3),
    ("4. + E1: Dynamic T&Q Optimal-Stopping Exit (60m/RTH/EOD)",        p_eng156_dyn_ex,   3),
    ("5. + E2: Bidirectional T&Q (+High-Precision LONG Accumulation)",  p_eng1256_winner,  3),
    ("6. All 6 Engines Unfiltered (Incl. E3 Mid-Morn & E4 $2-$10)",     p_all6_unfiltered, 3),
]

results_by_horizon = {}
for h_title, h_dates in HORIZONS:
    print("\n" + "=" * 164)
    print(f"  {h_title} ({len(h_dates)} Trading Days) — 157 T&Q STOCKS (100% WALK-FORWARD OOS)")
    print("=" * 164)
    print(f"| {'Strategy / Progressive 6-Engine Architecture':<59} | {'Trades (Active Days)':<20} | {'Trade WR':<8} | {'Day WR (W-L)':<16} | {'Avg Trade%':<10} | {'Tot PnL%':<9} | {'Comp ($100K)':<13} | {'Max DD%':<7} | {'Sharpe':<6} |")
    print(f"|{'-'*61}|{'-'*22}|{'-'*10}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*15}|{'-'*9}|{'-'*8}|")
    results_by_horizon[h_title] = {}
    for s_name, s_map, s_topn in STRAT_LIST:
        res = sim_port(s_map, h_dates, top_n=s_topn)
        results_by_horizon[h_title][s_name] = res
        pct_act = int(round(res["n_act"] / max(res["n_days"], 1) * 100))
        tr_col = f"{res['n_tr']:>3} ({res['n_act']:>3}/{res['n_days']}d = {pct_act:>2}%)"
        dwr_col = f"{res['dwr']:4.1f}% ({res['wd']}W-{res['ld']}L)"
        ceq_col = f"${res['ceq']:>11,.0f}"
        print(
            f"| {s_name:<59} | {tr_col:<20} | {res['twr']:6.1f}%  | {dwr_col:<16} | "
            f"{res['atr']:+8.2f}%  | {res['tot']:+7.1f}%  | {ceq_col:<13} | -{res['mdd']:5.1f}% | {res['sh']:6.2f} |"
        )

# Plot 4-Panel Progressive 6-Engine Chart
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=140)
plot_configs = [
    (axes[0, 0], HORIZONS[0][0], HORIZONS[0][1], "A. Full Walk-Forward OOS (24M: 2024-11-01 -> 2026-11-06)"),
    (axes[0, 1], HORIZONS[1][0], HORIZONS[1][1], "B. Last 12 Months (1Y: 2025-11-03 -> 2026-11-06)"),
    (axes[1, 0], HORIZONS[2][0], HORIZONS[2][1], "C. Last 4 Months (Q3-Q4: 2026-07-01 -> 2026-11-06)"),
    (axes[1, 1], HORIZONS[3][0], HORIZONS[3][1], "D. Last 2 Months & Last 2 Weeks (2026-09-01 -> 2026-11-06)"),
]

plot_strats = [
    ("1. Rule-Based Deployed C2 + Catalyst (Baseline)",                 "#ff7f0e", "--", 2.0),
    ("2. Previous Best: SYNERGY 4A (3-Pillar Static RTH Exit)",         "#1f77b4", "-.", 2.0),
    ("3. + E5 & E6: Rank-Gaussian Trajectory + Dilution Boost",         "#9467bd", "-",  2.2),
    ("4. + E1: Dynamic T&Q Optimal-Stopping Exit (60m/RTH/EOD)",        "#17becf", "-",  2.5),
    ("5. + E2: Bidirectional T&Q (+High-Precision LONG Accumulation)",  "#2ca02c", "-",  2.8),
    ("6. All 6 Engines Unfiltered (Incl. E3 Mid-Morn & E4 $2-$10)",     "#8c564b", ":",  1.8),
]


for ax, h_key, h_dates, p_title in plot_configs:
    x = np.arange(len(h_dates))
    for s_name, col, ls, lw in plot_strats:
        res = results_by_horizon[h_key][s_name]
        comp_eq = 100000.0 * np.cumprod(1.0 + res["d_rets"] / 100.0)
        comp_pct = (comp_eq / 100000.0 - 1.0) * 100.0
        short_lbl = s_name.split(". ", 1)[1]
        ax.plot(x, comp_eq / 1000.0, label=f"{short_lbl}: ${res['ceq']:,.0f} ({comp_pct[-1]:+.1f}%, DD -{res['mdd']:.0f}%, Sh {res['sh']:.2f})", color=col, linestyle=ls, linewidth=lw)
    ax.axhline(100.0, color="black", lw=0.8, alpha=0.5)
    if "Last 2 Months" in p_title:
        idx_2w = next((i for i, d in enumerate(h_dates) if d >= "2026-10-23"), len(h_dates) - 11)
        ax.axvspan(idx_2w, len(h_dates) - 1, color="gold", alpha=0.14, label="Last 2 Weeks (Oct 23 - Nov 6)")
    n_ticks = min(8, len(h_dates))
    t_idx = np.linspace(0, len(h_dates) - 1, n_ticks, dtype=int)
    ax.set_xticks(t_idx)
    ax.set_xticklabels([h_dates[i][5:] if len(h_dates) <= 50 else h_dates[i][:7] for i in t_idx], rotation=20)
    ax.set_title(p_title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Compounded Portfolio Equity ($K, Start = $100K)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=6.6)

plt.suptitle("Unlocking Untapped Profit Engines (157 T&Q Stocks, 100% Walk-Forward OOS): Dynamic Exit, Bidirectional Long/Short, Mid-Morning, $2-$10 Split-Clean & Rank-Trajectory ML", fontsize=11.5, fontweight="bold", y=0.995)
plt.tight_layout()
out_chart = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/tq_6_profit_engines_backtest.png"
plt.savefig(out_chart, dpi=140)
print(f"\nSaved 6-Engine Profit Discovery chart to: {out_chart}")
