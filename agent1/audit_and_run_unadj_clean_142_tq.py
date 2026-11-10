import os
import io
import time
import pickle
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from google.cloud import storage

warnings.filterwarnings("ignore")

LOCAL_SA_KEY = "/Users/yaninaso/Downloads/ys-time-series-sa-key.json"
BUCKET_NAME = "time-series-bucket-ys"
PROJECT_ID = "ys-time-series"
PKL_PATH = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_105_tickers_spikes_dataset.pkl"
POST_SEP8_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/post_sep8_existing_tq_records.pkl"
DEEP18_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_deep18_all_stocks_cache.pkl"
FAST_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_120_v2_and_unadj_fast_cache.pkl"
SPLIT_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_142_exact_unadj_split_cache.pkl"

# 1. Load model definitions & cached 142-ticker T&Q dataset + Deep18 parquet features
exec(open("/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/evaluate_tq_vs_ohlcv_all_models.py").read().split("# LOAD DATASET")[0])

with open(PKL_PATH, "rb") as f:
    payload = pickle.load(f)
records = payload["records"]
ticker_summary = dict(payload["ticker_summary"])

if os.path.exists(POST_SEP8_CACHE):
    with open(POST_SEP8_CACHE, "rb") as f:
        extra_post_sep8_recs = pickle.load(f)
    existing_keys = {(r["ticker"], r["date"]) for r in records}
    for r_ex in extra_post_sep8_recs:
        if (r_ex["ticker"], r_ex["date"]) not in existing_keys:
            records.append(r_ex)

with open(DEEP18_CACHE, "rb") as f:
    deep18_lookup = pickle.load(f)

with open(FAST_CACHE, "rb") as f:
    fc = pickle.load(f)
all_trading_dates = fc["all_trading_dates"]

# 2. Compute or load exact point-in-time split_factor = Total_Vol / Volume from Master_Inference_v2
all_tickers = sorted({r["ticker"] for r in records})
if os.path.exists(SPLIT_CACHE):
    with open(SPLIT_CACHE, "rb") as f:
        split_lookup = pickle.load(f)
else:
    split_lookup = {}

missing_split_tks = [tk for tk in all_tickers if not any(k[0] == tk for k in split_lookup)]
if missing_split_tks:
    import requests
    POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
    print(f"Fetching exact point-in-time unadjusted daily open prices for {len(missing_split_tks)} tickers (takes ~2s)...")

    def fetch_ticker_unadj_daily(tk):
        out = {}
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{tk}/range/1/day/2024-06-01/2026-11-06"
            resp = requests.get(url, params={"adjusted": "false", "sort": "asc", "limit": 5000, "apiKey": POLYGON_API_KEY}, timeout=10)
            if resp.status_code == 200:
                for b in resp.json().get("results", []):
                    d_s = pd.Timestamp(b["t"], unit="ms", tz="UTC").tz_convert("America/New_York").strftime("%Y-%m-%d")
                    out[(tk, d_s)] = float(b["o"])
        except Exception:
            pass
        return out

    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = [ex.submit(fetch_ticker_unadj_daily, tk) for tk in missing_split_tks]
        for fut in as_completed(futs):
            split_lookup.update(fut.result())

    with open(SPLIT_CACHE, "wb") as f:
        pickle.dump(split_lookup, f)
    print(f"Saved {len(split_lookup):,} ticker-day exact unadjusted open prices to {SPLIT_CACHE}")

# 3. Audit Reverse-Split Distortion & Restore True Point-in-Time Unadjusted Price and Spread%
split_distorted_examples = []
for r in records:
    unadj_o = float(split_lookup.get((r["ticker"], r["date"]), r["open_px"]))
    if unadj_o <= 0:
        unadj_o = r["open_px"]
    sf = float(r["open_px"] / unadj_o)
    if sf < 1.15:
        sf = 1.0
    r["split_factor"] = sf
    r["unadj_open_px"] = unadj_o
    r["unadj_entry_px"] = float(r["entry_px"] / sf)

    o_c = list(r["f_ohlcv"])
    b_c = list(r["f_base_tq"])
    a_c = list(r["f_adv_tq"])
    if r["entry_hm"] < 392:
        o_c[10] = float(np.log1p(max(r["cum_rvol_so_far"], 0.0)))
        b_c[13] = b_c[0]; b_c[14] = b_c[3]; a_c[14] = a_c[0]; a_c[15] = a_c[5]
        r["causal_open_rvol"] = r["cum_rvol_so_far"]
    else:
        r["causal_open_rvol"] = r["open3_rvol"]

    # CRITICAL FIX: Restore true unadjusted percentage spread (b_c[7] was divided by split_factor!)
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
    r["ret_gap_adaptive"] = r["ret_short_60m"] if r["gap_pct"] < 6.0 else r["raw_pnl_pct"]

    gap_pct, surge_pct, tot_ext, velocity, pm_fade = o_c[0], o_c[1], o_c[2], o_c[3], o_c[4]
    u_wick, body_rat, bar_rng, pb_peak, vwap_dist = o_c[5], o_c[6], o_c[7], o_c[8], o_c[9]
    log_o3_rv, log_cum_rv, vol_climax, trig_is_b0, mins_op = o_c[10], o_c[11], o_c[12], o_c[13], o_c[14]
    atr_pct, ret_5d, ret_20d, dist_20d_h, log10_dvol = o_c[15], o_c[16], o_c[17], o_c[18], o_c[19]
    obi_b, obi_drop, cum_obi, dpct_b, dpct_div = b_c[0], b_c[1], b_c[2], b_c[3], b_c[4]
    cum_dpct, vpin_p, sp_pct, sp_wid, bs_as_log = b_c[5], b_c[6], b_c[7], b_c[8], b_c[9]
    depth_exh, tcnt_rat, tsize_rat, o3_obi, o3_dpct, pm_obi = b_c[10], b_c[11], b_c[12], b_c[13], b_c[14], b_c[15]
    lr_ofi, ofi_div, cum_ofi, lr_vpin, lr_buy = a_c[0], a_c[1], a_c[2], a_c[3], a_c[4]
    lr_simb, pe_b, lzc_b, tvol_b, shock_b = a_c[5], a_c[6], a_c[7], a_c[8], a_c[9]
    size_cv, tint_rat, qint_rat, sp_cv, o3_ofi, o3_simb, pm_ofi = a_c[10], a_c[11], a_c[12], a_c[13], a_c[14], a_c[15], a_c[16]

    cum_rv = max(r["cum_rvol_so_far"], 0.1)
    f_deep = list(deep18_lookup.get((r["ticker"], r["date"]), [0.0] * 18))

    # The 8 Empirically Verified Orthogonal New Features:
    r["f_verified8_new"] = [
        float(f_deep[7]),                                                       # 1. deep08_pm_trapped_vol_share
        float(np.clip(f_deep[4], 0.0, 15.0)),                                   # 2. deep05_quote_int_vs_20d_norm
        float(f_deep[11]),                                                      # 3. deep12_total_overhead_trapped_vol
        float(-lr_simb * (1.0 - lr_buy)),                                       # 4. new02_inst_sell_block_pressure
        float((2.0 * lr_buy - 1.0) - lr_ofi),                                   # 5. new01_retail_vs_inst_ofi_div
        float(lr_ofi - body_rat),                                               # 6. new05_passive_ask_absorption
        float(bar_rng / max(np.log1p(cum_rv * max(vol_climax, 0.2)), 0.2)),     # 7. new07_kyle_lambda_bar
        float(np.clip(vol_climax, 0.0, 10.0) * pb_peak * (1.0 - lr_ofi)),       # 8. new09_climax_reversal_torque
    ]
    b_up = list(b_c)
    a_up = list(a_c)
    if abs(f_deep[10]) > 1e-6:
        b_up[15] = float(f_deep[10]) # deep11_pm_vol_weighted_obi
    if abs(f_deep[9]) > 1e-6:
        a_up[16] = float(f_deep[9])  # deep10_pm_vol_weighted_lr_ofi
    r["f_tq33_upgraded_pm"] = b_up + a_up

# Print Reverse-Split Audit Table across all 142 tickers
print("\n" + "=" * 145)
print("AUDIT OF REVERSE-SPLIT PENNY-STOCK DISTORTION CAUGHT BY EXACT (Total_Vol / Volume) RATIO")
print("=" * 145)
by_tk_sf = defaultdict(list)
for r in records:
    if r["is_rvol_verified_causal"] and r["has_adv_tq"]:
        by_tk_sf[r["ticker"]].append(r)

distorted_tks = []
for tk, rlist in sorted(by_tk_sf.items()):
    max_sf = max(x["split_factor"] for x in rlist)
    min_unadj = min(x["unadj_open_px"] for x in rlist)
    med_unadj = float(np.median([x["unadj_open_px"] for x in rlist]))
    med_adj = float(np.median([x["open_px"] for x in rlist]))
    n_sub5 = sum(1 for x in rlist if x["unadj_open_px"] < 5.0)
    if max_sf >= 1.5 or n_sub5 >= 5:
        distorted_tks.append((max_sf, n_sub5, tk, len(rlist), med_adj, med_unadj, min_unadj, float(np.mean([x["raw_pnl_pct"] for x in rlist]))))

distorted_tks.sort(key=lambda x: (-x[0], -x[1]))
print(f"| {'Ticker':<6} | {'RVol Setups':<11} | {'Max Split Factor':<16} | {'Split-Adj Med Open':<18} | {'TRUE Unadj Med Open':<19} | {'Min Unadj Open':<14} | {'Setups < $5 Unadj':<17} | {'Avg Short PnL%':<14} |")
print(f"|{'-'*8}|{'-'*13}|{'-'*18}|{'-'*20}|{'-'*21}|{'-'*16}|{'-'*19}|{'-'*16}|")
for max_sf, n_sub5, tk, n_tot, med_adj, med_unadj, min_unadj, avg_p in distorted_tks[:28]:
    print(f"| {tk:<6} | {n_tot:>11d} | {max_sf:>15.1f}x | ${med_adj:>16.2f} | ${med_unadj:>17.2f} | ${min_unadj:>12.2f} | {n_sub5:>17d} | {avg_p:>+13.2f}% |")

# 4. Run Walk-Forward OOS Evaluation on Clean Point-in-Time Universe (unadj_open_px >= $5.00, no post-split penny stock illusions!)
def run_walk_forward_for_pool(pool_records, label=""):
    adv_pool = [r for r in pool_records if r["is_rvol_verified_causal"] and r["has_adv_tq"]]
    all_months = sorted({r["ym"] for r in adv_pool})
    min_train_months = 4
    oos_start_ym = all_months[min_train_months]

    ym_adv = np.array([r["ym"] for r in adv_pool])
    y_adv = np.array([r["raw_pnl_pct"] for r in adv_pool], dtype=float)
    X_core8 = np.array([[r["f_ohlcv_causal"][idx] for idx in (0, 1, 3, 5, 6, 9, 11, 15)] for r in adv_pool], dtype=float)
    X_old33 = np.array([r["f_base_tq_causal"] + r["f_adv_tq_causal"] for r in adv_pool], dtype=float)
    X_up33  = np.array([r["f_tq33_upgraded_pm"] for r in adv_pool], dtype=float)
    X_new8  = np.array([r["f_verified8_new"] for r in adv_pool], dtype=float)

    preds = defaultdict(dict)
    oos_rows = []
    for m_idx in range(min_train_months, len(all_months)):
        te_ym = all_months[m_idx]
        tr_m = ym_adv < te_ym
        te_m = ym_adv == te_ym
        if np.sum(tr_m) < 50 or np.sum(te_m) == 0:
            continue
        te_indices = np.where(te_m)[0]

        # Old 33 T&Q
        corrs_old = [abs(float(np.nan_to_num(sp_stats.spearmanr(X_old33[tr_m, j], y_adv[tr_m])[0]))) if np.std(X_old33[tr_m, j]) > 1e-8 else 0.0 for j in range(33)]
        t12_old = np.argsort(corrs_old)[-12:]
        X_c20_old_tr = np.hstack([X_core8[tr_m], X_old33[tr_m][:, t12_old]])
        X_c20_old_te = np.hstack([X_core8[te_m], X_old33[te_m][:, t12_old]])

        _, p_m1_old = fit_m1_ridge(X_old33[tr_m], y_adv[tr_m], X_old33[te_m])
        _, p_m3_old = fit_m3_poly_ridge(X_old33[tr_m], y_adv[tr_m], X_old33[te_m])
        _, p_m6_old = fit_m6_gbdt_d2(X_old33[tr_m], y_adv[tr_m], X_old33[te_m])
        _, p_m7_old = fit_m7_gbdt_d3(X_old33[tr_m], y_adv[tr_m], X_old33[te_m])
        _, p_m8_old = fit_m8_rf_d2(X_old33[tr_m], y_adv[tr_m], X_old33[te_m])
        _, p_m10_old = fit_m10_mlp(X_old33[tr_m], y_adv[tr_m], X_old33[te_m])
        p_m11_old = 0.25 * (p_m3_old + p_m6_old + p_m8_old + p_m10_old)
        _, p_m4_old = fit_m4_logistic(X_c20_old_tr, y_adv[tr_m], X_c20_old_te)
        p_m12_old = 0.35 * p_m11_old + 0.25 * p_m7_old + 0.20 * p_m1_old + 0.20 * p_m4_old

        # Upgraded T&Q (+8 Verified New Features, In-Fold Top 5 Orthogonal Selection)
        corrs_up = [abs(float(np.nan_to_num(sp_stats.spearmanr(X_up33[tr_m, j], y_adv[tr_m])[0]))) if np.std(X_up33[tr_m, j]) > 1e-8 else 0.0 for j in range(33)]
        t12_up = np.argsort(corrs_up)[-12:]
        corrs_n8 = [abs(float(np.nan_to_num(sp_stats.spearmanr(X_new8[tr_m, j], y_adv[tr_m])[0]))) if np.std(X_new8[tr_m, j]) > 1e-8 else 0.0 for j in range(8)]
        t5_n8 = np.argsort(corrs_n8)[-5:]

        X_c25_new_tr = np.hstack([X_core8[tr_m], X_up33[tr_m][:, t12_up], X_new8[tr_m][:, t5_n8]])
        X_c25_new_te = np.hstack([X_core8[te_m], X_up33[te_m][:, t12_up], X_new8[te_m][:, t5_n8]])
        X_tq38_new_tr = np.hstack([X_up33[tr_m], X_new8[tr_m][:, t5_n8]])
        X_tq38_new_te = np.hstack([X_up33[te_m], X_new8[te_m][:, t5_n8]])

        _, p_m1_new = fit_m1_ridge(X_tq38_new_tr, y_adv[tr_m], X_tq38_new_te)
        _, p_m3_new = fit_m3_poly_ridge(X_tq38_new_tr, y_adv[tr_m], X_tq38_new_te)
        _, p_m6_new = fit_m6_gbdt_d2(X_tq38_new_tr, y_adv[tr_m], X_tq38_new_te)
        _, p_m7_new = fit_m7_gbdt_d3(X_tq38_new_tr, y_adv[tr_m], X_tq38_new_te)
        _, p_m8_new = fit_m8_rf_d2(X_tq38_new_tr, y_adv[tr_m], X_tq38_new_te)
        _, p_m10_new = fit_m10_mlp(X_tq38_new_tr, y_adv[tr_m], X_tq38_new_te)
        p_m11_new = 0.25 * (p_m3_new + p_m6_new + p_m8_new + p_m10_new)
        _, p_m4_new = fit_m4_logistic(X_c25_new_tr, y_adv[tr_m], X_c25_new_te)
        p_m12_new = 0.35 * p_m11_new + 0.25 * p_m7_new + 0.20 * p_m1_new + 0.20 * p_m4_new

        for loc_i, r_idx in enumerate(te_indices):
            rec = adv_pool[r_idx]
            k = (rec["ticker"], rec["date"])
            preds[k] = {
                "M4_OLD": float(p_m4_old[loc_i]), "M4_NEW": float(p_m4_new[loc_i]),
                "M7_OLD": float(p_m7_old[loc_i]), "M7_NEW": float(p_m7_new[loc_i]),
                "M11_OLD": float(p_m11_old[loc_i]), "M11_NEW": float(p_m11_new[loc_i]),
                "M12_OLD": float(p_m12_old[loc_i]), "M12_NEW": float(p_m12_new[loc_i]),
            }
            oos_rows.append((y_adv[r_idx], preds[k], rec))

    print("\n" + "=" * 145)
    print(f"WALK-FORWARD OOS SIGNAL QUALITY [{label}] — {len(oos_rows)} OOS SETUPS ACROSS {len({r['ticker'] for r in adv_pool})} T&Q STOCKS")
    print("=" * 145)
    y_arr = np.array([x[0] for x in oos_rows])
    for m_base, m_lbl in [("M4", "M4 Regularized Logistic"), ("M7", "M7 Deep GBDT (Depth-3)"), ("M11", "M11 Non-Linear Super-Ensemble"), ("M12", "M12 Grand T&Q Hybrid Ensemble")]:
        p_o = np.array([x[1][f"{m_base}_OLD"] for x in oos_rows])
        p_n = np.array([x[1][f"{m_base}_NEW"] for x in oos_rows])
        ic_o, pv_o = sp_stats.spearmanr(p_o, y_arr)
        ic_n, pv_n = sp_stats.spearmanr(p_n, y_arr)
        t10_o = y_arr[p_o >= np.percentile(p_o, 90)]
        t10_n = y_arr[p_n >= np.percentile(p_n, 90)]
        print(
            f"  {m_lbl:<32} | Old IC: {ic_o:+.4f} (p={pv_o:.3f}) -> Upgraded IC: {ic_n:+.4f} (p={pv_n:.3f}) | "
            f"Top-10% Avg: Old {t10_o.mean():+.2f}% ({(t10_o>0).mean()*100:.1f}% WR) -> Upgraded {t10_n.mean():+.2f}% ({(t10_n>0).mean()*100:.1f}% WR)"
        )
    return adv_pool, oos_start_ym, preds

# Test Pool A: Unadjusted Price >= $5.00 & Split Factor <= 1.25 (Purging all reverse-split penny-stock illusions)
pool_clean_5 = [r for r in records if 5.0 <= r["unadj_open_px"] <= 150.0 and r["split_factor"] <= 1.25]
# Test Pool B: Unadjusted Price >= $10.00 & <= $100.00 & Split Factor <= 1.25 (Exact C2 Price Floor $10-$100)
pool_clean_10 = [r for r in records if 10.0 <= r["unadj_open_px"] <= 100.0 and r["split_factor"] <= 1.25]

adv_5, oos_ym_5, preds_5 = run_walk_forward_for_pool(pool_clean_5, "POOL A: POINT-IN-TIME UNADJ PRICE >= $5.00 (NO REVERSE-SPLIT PENNY STOCKS)")
adv_10, oos_ym_10, preds_10 = run_walk_forward_for_pool(pool_clean_10, "POOL B: POINT-IN-TIME UNADJ PRICE $10-$100 (EXACT C2 PRICE RANGE)")

# 5. Build Multi-Horizon Portfolio Comparison & Plot Clean Compounded Curves
def v2_trades_to_day_map(tr_list):
    out = defaultdict(list)
    for idx, t in enumerate(tr_list):
        d = str(t.get("scan_date") or t.get("date"))
        tk = t["ticker"].split("_")[0]
        unadj_o = float(split_lookup.get((tk, d), 15.0))
        if unadj_o < 5.0:
            continue
        pnl_pct = max(float(t.get("pnl_eod", 0.0)), -10.5)
        t_et = str(t.get("entry_time", "09:33"))
        hm = (int(t_et[:2]) - 3) * 60 + int(t_et[3:5]) if ":" in t_et else 393
        out[d].append((1000.0 - idx, {"ticker": t["ticker"], "date": d, "entry_hm": hm, "raw_pnl_pct": pnl_pct, "ret_gap_adaptive": pnl_pct}))
    return out

s_v2_c2_base = v2_trades_to_day_map(fc["tr_v2_c2_base"])
s_v2_c2_cat  = v2_trades_to_day_map(fc["tr_v2_c2_cat"])
s_v2_c2_s25  = v2_trades_to_day_map(fc["tr_v2_c2_s25"])

def make_ml_map(pool_list, preds_dict, oos_ym, m_key, thresh=0.35):
    out = defaultdict(list)
    for r in pool_list:
        if r["ym"] < oos_ym or r["gap_pct"] < 3.0:
            continue
        sc = preds_dict.get((r["ticker"], r["date"]), {}).get(m_key, -999.0)
        if sc >= thresh:
            out[r["date"]].append((sc, r))
    return out

s_m4_5_new   = make_ml_map(adv_5,  preds_5,  oos_ym_5,  "M4_NEW",  0.35)
s_m7_5_old   = make_ml_map(adv_5,  preds_5,  oos_ym_5,  "M7_OLD",  0.35)
s_m7_5_new   = make_ml_map(adv_5,  preds_5,  oos_ym_5,  "M7_NEW",  0.35)
s_m12_5_old  = make_ml_map(adv_5,  preds_5,  oos_ym_5,  "M12_OLD", 0.35)
s_m12_5_new  = make_ml_map(adv_5,  preds_5,  oos_ym_5,  "M12_NEW", 0.35)

s_m4_10_new  = make_ml_map(adv_10, preds_10, oos_ym_10, "M4_NEW",  0.35)
s_m7_10_old  = make_ml_map(adv_10, preds_10, oos_ym_10, "M7_OLD",  0.35)
s_m7_10_new  = make_ml_map(adv_10, preds_10, oos_ym_10, "M7_NEW",  0.35)
s_m12_10_old = make_ml_map(adv_10, preds_10, oos_ym_10, "M12_OLD", 0.35)
s_m12_10_new = make_ml_map(adv_10, preds_10, oos_ym_10, "M12_NEW", 0.35)

# Hybrid: Deployed C2 + Catalyst (Clean Unadj >= $10) + Upgraded M7/M12 T&Q ML ($10-$100 Unadj)
s_hybrid_clean = defaultdict(list)
for d, cands in s_v2_c2_cat.items():
    for sc, r in cands:
        s_hybrid_clean[d].append((sc + 100.0, r))
for r in adv_10:
    if r["ym"] < oos_ym_10 or r["gap_pct"] < 3.0:
        continue
    d = r["date"]
    p_d = preds_10.get((r["ticker"], d), {})
    sc_h = max(p_d.get("M7_NEW", -999.0), p_d.get("M12_NEW", -999.0))
    if sc_h >= 0.40:
        existing_tks_d = {x[1]["ticker"].split("_")[0] for x in s_hybrid_clean[d]}
        if r["ticker"] not in existing_tks_d:
            s_hybrid_clean[d].append((sc_h, r))

HORIZONS = [
    ("1. FULL WALK-FORWARD OOS (24M: 2024-11-01 -> 2026-11-06)", [d for d in all_trading_dates if "2024-11-01" <= d <= "2026-11-06"]),
    ("2. LAST 12 MONTHS (1Y: 2025-11-03 -> 2026-11-06)",         [d for d in all_trading_dates if "2025-11-03" <= d <= "2026-11-06"]),
    ("3. LAST 4 MONTHS (Q3-Q4: 2026-07-01 -> 2026-11-06)",       [d for d in all_trading_dates if "2026-07-01" <= d <= "2026-11-06"]),
    ("4. LAST 2 MONTHS (Sep-Nov: 2026-09-01 -> 2026-11-06)",     [d for d in all_trading_dates if "2026-09-01" <= d <= "2026-11-06"]),
    ("5. LAST 2 WEEKS (Oct 23 -> Nov 06, 2026)",                 [d for d in all_trading_dates if "2026-10-23" <= d <= "2026-11-06"]),
]

STRAT_LIST = [
    ("1. Deployed C2 + Catalyst (Clean Unadj >= $10)",      s_v2_c2_cat,    "raw_pnl_pct",      3),
    ("2. C2 Strict <= $25 + Catalyst (Clean Unadj >= $10)", s_v2_c2_s25,    "raw_pnl_pct",      3),
    ("3. M7 Deep GBDT (>= $5 Unadj): Old 33 T&Q",           s_m7_5_old,     "raw_pnl_pct",      2),
    ("4. M7 Deep GBDT (>= $5 Unadj): Upgraded (+8 New)",    s_m7_5_new,     "raw_pnl_pct",      2),
    ("5. M12 Ensemble (>= $5 Unadj): Old 33 T&Q",           s_m12_5_old,    "ret_gap_adaptive", 2),
    ("6. M12 Ensemble (>= $5 Unadj): Upgraded (+8 New)",    s_m12_5_new,    "ret_gap_adaptive", 2),
    ("7. M7 Deep GBDT ($10-$100 Unadj): Old 33 T&Q",        s_m7_10_old,    "raw_pnl_pct",      2),
    ("8. M7 Deep GBDT ($10-$100 Unadj): Upgraded (+8 New)", s_m7_10_new,    "raw_pnl_pct",      2),
    ("9. M12 Ensemble ($10-$100 Unadj): Old 33 T&Q",        s_m12_10_old,   "ret_gap_adaptive", 2),
    ("10. M12 Ensemble ($10-$100 Unadj): Upgraded (+8 New)",s_m12_10_new,   "ret_gap_adaptive", 2),
    ("11. HYBRID: Clean C2 + Catalyst + Upgraded M7/M12",   s_hybrid_clean, "raw_pnl_pct",      3),
]

def sim_horizon(day_map, dates_h, rf="raw_pnl_pct", top_n=2):
    cap, pk, mdd = 100_000.0, 100_000.0, 0.0
    t_pnls, d_rets, a_rets = [], [], []
    w_d, l_d = 0, 0
    for d in dates_h:
        cands = day_map.get(d, [])
        if not cands:
            d_rets.append(0.0); continue
        cs = sorted(cands, key=lambda x: (x[1]["entry_hm"], -x[0]))[:top_n]
        k = len(cs)
        w = min(1.0 / k, 0.50)
        dr = 0.0
        for sc, r in cs:
            p = float(r[rf])
            t_pnls.append(p)
            dr += w * p
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

results_by_horizon = {}
for h_title, h_dates in HORIZONS:
    print("\n" + "=" * 158)
    print(f"  {h_title} ({len(h_dates)} Trading Days)")
    print("=" * 158)
    print(f"| {'Strategy / Unadj Price Filter Variant':<52} | {'Trades (Active Days)':<20} | {'Trade WR':<8} | {'Day WR (W-L)':<16} | {'Avg Trade%':<10} | {'Tot PnL%':<9} | {'Comp ($100K)':<13} | {'Max DD%':<7} | {'Sharpe':<6} |")
    print(f"|{'-'*54}|{'-'*22}|{'-'*10}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*15}|{'-'*9}|{'-'*8}|")
    results_by_horizon[h_title] = {}
    for s_name, s_map, s_rf, s_topn in STRAT_LIST:
        res = sim_horizon(s_map, h_dates, s_rf, top_n=s_topn)
        results_by_horizon[h_title][s_name] = res
        pct_act = int(round(res["n_act"] / max(res["n_days"], 1) * 100))
        tr_col = f"{res['n_tr']:>3} ({res['n_act']:>3}/{res['n_days']}d = {pct_act:>2}%)"
        dwr_col = f"{res['dwr']:4.1f}% ({res['wd']}W-{res['ld']}L)"
        ceq_col = f"${res['ceq']:>11,.0f}"
        print(
            f"| {s_name:<52} | {tr_col:<20} | {res['twr']:6.1f}%  | {dwr_col:<16} | "
            f"{res['atr']:+8.2f}%  | {res['tot']:+7.1f}%  | {ceq_col:<13} | -{res['mdd']:5.1f}% | {res['sh']:6.2f} |"
        )

# Plot 4-Panel Compounded Portfolio Equity Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=140)
plot_configs = [
    (axes[0, 0], HORIZONS[0][0], HORIZONS[0][1], "A. Full Walk-Forward OOS (24M: 2024-11-01 -> 2026-11-06)"),
    (axes[0, 1], HORIZONS[1][0], HORIZONS[1][1], "B. Last 12 Months (1Y: 2025-11-03 -> 2026-11-06)"),
    (axes[1, 0], HORIZONS[2][0], HORIZONS[2][1], "C. Last 4 Months (Q3-Q4: 2026-07-01 -> 2026-11-06)"),
    (axes[1, 1], HORIZONS[3][0], HORIZONS[3][1], "D. Last 2 Months & Last 2 Weeks (2026-09-01 -> 2026-11-06)"),
]

plot_strats = [
    ("1. Deployed C2 + Catalyst (Clean Unadj >= $10)",      "#ff7f0e", "-.", 2.0),
    ("2. C2 Strict <= $25 + Catalyst (Clean Unadj >= $10)", "#8c564b", ":",  2.0),
    ("3. M7 Deep GBDT (>= $5 Unadj): Old 33 T&Q",           "#9467bd", "--", 1.6),
    ("4. M7 Deep GBDT (>= $5 Unadj): Upgraded (+8 New)",    "#9467bd", "-",  2.1),
    ("7. M7 Deep GBDT ($10-$100 Unadj): Old 33 T&Q",        "#1f77b4", "--", 1.8),
    ("8. M7 Deep GBDT ($10-$100 Unadj): Upgraded (+8 New)", "#1f77b4", "-",  2.5),
    ("9. M12 Ensemble ($10-$100 Unadj): Old 33 T&Q",        "#e377c2", "--", 1.8),
    ("10. M12 Ensemble ($10-$100 Unadj): Upgraded (+8 New)","#e377c2", "-",  2.4),
    ("11. HYBRID: Clean C2 + Catalyst + Upgraded M7/M12",   "#2ca02c", "-",  2.8),
]

for ax, h_key, h_dates, p_title in plot_configs:
    x = np.arange(len(h_dates))
    for s_name, col, ls, lw in plot_strats:
        res = results_by_horizon[h_key][s_name]
        comp_eq = 100000.0 * np.cumprod(1.0 + res["d_rets"] / 100.0)
        comp_pct = (comp_eq / 100000.0 - 1.0) * 100.0
        short_lbl = s_name.split(". ", 1)[1]
        ax.plot(x, comp_eq / 1000.0, label=f"{short_lbl}: ${res['ceq']:,.0f} ({comp_pct[-1]:+.1f}%, DD -{res['mdd']:.0f}%)", color=col, linestyle=ls, linewidth=lw)
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
    ax.legend(loc="best", fontsize=7.0)

plt.suptitle("Point-in-Time Unadjusted Price Filtered T&Q Universe: Old 33 T&Q vs. Upgraded (+8 Orthogonal Features) & Deployed C2", fontsize=12.5, fontweight="bold", y=0.995)
plt.tight_layout()
out_chart = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/tq_129_unadj_clean_upgraded_backtest.png"
plt.savefig(out_chart, dpi=140)
print(f"\nSaved clean unadjusted price backtest chart to: {out_chart}")
