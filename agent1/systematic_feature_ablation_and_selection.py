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

client = storage.Client.from_service_account_json(LOCAL_SA_KEY, project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

# 1. Load model definitions & check if any new stocks finished on GCS (e.g. the 21 additional stocks #19 -> #39!)
exec(open("/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/evaluate_tq_vs_ohlcv_all_models.py").read().split("# LOAD DATASET")[0])
exec(open("/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/update_and_run_tq_backtest.py").read().split("if new_tickers_to_extract:")[0])

if new_tickers_to_extract:
    print(f"Extracting {len(new_tickers_to_extract)} newly completed tickers from GCS: {new_tickers_to_extract}...")
    records = [r for r in records if r["ticker"] not in set(new_tickers_to_extract)]
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(extract_one_ticker, tk): tk for tk in new_tickers_to_extract}
        for fut in as_completed(futs):
            tk = futs[fut]
            recs_tk, inf_tk = fut.result()
            records.extend(recs_tk)
            ticker_summary[tk] = inf_tk
            print(f"  + {tk:<6}: {len(recs_tk):>3} positive spikes ({inf_tk['n_adv_tq_days']} Adv T&Q days)")
    with open(PKL_PATH, "wb") as f:
        pickle.dump({"records": records, "ticker_summary": sorted(ticker_summary.items())}, f)

if os.path.exists(POST_SEP8_CACHE):
    with open(POST_SEP8_CACHE, "rb") as f:
        extra_post_sep8_recs = pickle.load(f)
    existing_keys = {(r["ticker"], r["date"]) for r in records}
    for r_ex in extra_post_sep8_recs:
        if (r_ex["ticker"], r_ex["date"]) not in existing_keys:
            records.append(r_ex)

records = [r for r in records if r["open_px"] <= 300.0]

# 2. Prepare causal features + 30 cross-modal features
NEW30_NAMES = [
    "new01_retail_vs_inst_ofi_div",     "new02_inst_sell_block_pressure",   "new03_retail_climax_exhaustion",
    "new04_odd_lot_dispersion_buy",     "new05_passive_ask_absorption",     "new06_wick_flow_rejection",
    "new07_kyle_lambda_bar",            "new08_amihud_hollow_surge",        "new09_climax_reversal_torque",
    "new10_microprice_down_pressure",   "new11_spread_shock_x_obi_drop",    "new12_liquidity_vacuum_index",
    "new13_quote_stuffing_per_vol",     "new14_spread_instability_stress",  "new15_open_to_trig_ofi_decay",
    "new16_open_to_trig_obi_decay",     "new17_open_to_trig_simb_shift",    "new18_vwap_ext_vs_cum_ofi_div",
    "new19_rvol_frontload_decay",       "new20_pm_to_trig_ofi_reversal",    "new21_entropy_complexity_prod",
    "new22_algo_sell_sweep_intensity",  "new23_roll_bid_ask_bounce_est",    "new24_churn_tickvol_to_range",
    "new25_signed_vpin_short_tox",      "new26_gap_in_atr_units",           "new27_total_ext_in_atr_units",
    "new28_smallcap_illiquidity_score", "new29_vwap_rubberband_atr",        "new30_pm_rejection_x_gap",
]

DEEP18_NAMES = [
    "deep01_trade_size_vs_20d_norm",     "deep02_spread_vs_20d_norm",         "deep03_trade_cnt_vs_20d_norm",
    "deep04_tick_vol_vs_20d_norm",       "deep05_quote_int_vs_20d_norm",      "deep06_open_vs_pm_vwap_pct",
    "deep07_entry_vs_pm_vwap_pct",       "deep08_pm_trapped_vol_share",       "deep09_pm_peak_mins_before_open",
    "deep10_pm_vol_weighted_lr_ofi",     "deep11_pm_vol_weighted_obi",        "deep12_total_overhead_trapped_vol",
    "deep13_peak_to_trig_ofi_drop",      "deep14_pe_entropy_jump_vs_morn",    "deep15_lzc_complexity_jump",
    "deep16_down_vs_up_kyle_lambda",     "deep17_prev_day_last_hr_lr_ofi",    "deep18_prev_day_last_hr_obi",
]

for r in records:
    o_c = list(r["f_ohlcv"])
    b_c = list(r["f_base_tq"])
    a_c = list(r["f_adv_tq"])
    if r["entry_hm"] < 392:
        o_c[10] = float(np.log1p(max(r["cum_rvol_so_far"], 0.0)))
        b_c[13] = b_c[0]; b_c[14] = b_c[3]; a_c[14] = a_c[0]; a_c[15] = a_c[5]
        r["causal_open_rvol"] = r["cum_rvol_so_far"]
    else:
        r["causal_open_rvol"] = r["open3_rvol"]
    r["f_ohlcv_causal"] = o_c; r["f_base_tq_causal"] = b_c; r["f_adv_tq_causal"] = a_c
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
    r["f_new30"] = [
        float((2.0 * lr_buy - 1.0) - lr_ofi),
        float(-lr_simb * (1.0 - lr_buy)),
        float(np.clip(tcnt_rat / max(tsize_rat, 0.15), 0.0, 25.0)),
        float(np.clip(size_cv, 0.0, 10.0) * (2.0 * lr_buy - 1.0)),
        float(lr_ofi - body_rat),
        float(u_wick * (1.0 - lr_ofi)),
        float(bar_rng / max(np.log1p(cum_rv * max(vol_climax, 0.2)), 0.2)),
        float(surge_pct / max(cum_rv, 0.25)),
        float(np.clip(vol_climax, 0.0, 10.0) * pb_peak * (1.0 - lr_ofi)),
        float(-obi_b * sp_pct),
        float(-obi_drop * np.clip(sp_wid, 0.2, 10.0)),
        float(np.clip(sp_pct * np.clip(sp_wid, 0.2, 10.0) / max(depth_exh, 0.2), 0.0, 15.0)),
        float(np.clip(qint_rat / max(vol_climax, 0.2), 0.0, 20.0)),
        float(np.clip(sp_cv, 0.0, 5.0) * sp_pct),
        float(o3_ofi - lr_ofi),
        float(o3_obi - obi_b),
        float(o3_simb - lr_simb),
        float(vwap_dist - 5.0 * cum_ofi),
        float(log_o3_rv - log_cum_rv),
        float(pm_ofi - lr_ofi),
        float(pe_b * lzc_b),
        float((1.0 - pe_b) * (-lr_ofi)),
        float(-shock_b * tvol_b),
        float(tvol_b / max(bar_rng, 0.15)),
        float(lr_vpin * (-lr_ofi)),
        float(gap_pct / max(atr_pct, 0.5)),
        float(tot_ext / max(atr_pct, 0.5)),
        float(((50.0 - np.clip(r["open_px"], 2.0, 50.0)) / 50.0) * (9.0 - np.clip(log10_dvol, 5.0, 9.0))),
        float(vwap_dist / max(atr_pct, 0.5)),
        float(-pm_fade * np.log1p(max(gap_pct, 0.0))),
    ]

# 3. Extract or load Deep18 Multi-Day / Pre-Market / Trapped-Volume features across ALL T&Q stocks
if os.path.exists(DEEP18_CACHE):
    with open(DEEP18_CACHE, "rb") as f:
        deep18_lookup = pickle.load(f)
else:
    deep18_lookup = {}

recs_by_tk = defaultdict(list)
for r in records:
    if r["is_rvol_verified_causal"] and r["has_adv_tq"] and (r["ticker"], r["date"]) not in deep18_lookup:
        recs_by_tk[r["ticker"]].append(r)

if recs_by_tk:
    print(f"Extracting Deep18 Multi-Day & Pre-Market Parquet Features for {len(recs_by_tk)} T&Q tickers...")
    exec(open("/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/study_deep_parquet_and_tick_features.py").read().split("deep_recs = []")[0].split("top_tks = ")[-1].split("\n", 2)[-1])
    with ThreadPoolExecutor(max_workers=14) as ex:
        futs = {ex.submit(process_ticker_deep_features, tk): tk for tk in recs_by_tk.keys()}
        for fut in as_completed(futs):
            for row in fut.result():
                deep18_lookup[(row["ticker"], row["date"])] = row["deep18"]
    with open(DEEP18_CACHE, "wb") as f:
        pickle.dump(deep18_lookup, f)
    print(f"Cached Deep18 features for {len(deep18_lookup)} total RVol-verified spike setups!")

for r in records:
    r["f_deep18"] = deep18_lookup.get((r["ticker"], r["date"]), [0.0] * 18)

# 4. Controlled Walk-Forward Marginal Lift Study across all 48 New Features (30 Cross-Modal + 18 Deep Parquet)
adv_pool = [r for r in records if r["is_rvol_verified_causal"] and r["has_adv_tq"]]
all_months = sorted({r["ym"] for r in adv_pool})
min_train_months = 4
oos_start_ym = all_months[min_train_months]

ym_adv = np.array([r["ym"] for r in adv_pool])
y_adv = np.array([r["raw_pnl_pct"] for r in adv_pool], dtype=float)
X_core8 = np.array([[r["f_ohlcv_causal"][idx] for idx in (0, 1, 3, 5, 6, 9, 11, 15)] for r in adv_pool], dtype=float)
X_old33 = np.array([r["f_base_tq_causal"] + r["f_adv_tq_causal"] for r in adv_pool], dtype=float)
X_cand48 = np.array([r["f_new30"] + r["f_deep18"] for r in adv_pool], dtype=float)
CAND48_NAMES = NEW30_NAMES + DEEP18_NAMES

print("\n" + "=" * 155)
print(f"SECTION 1: FULL-UNIVERSE UNIVARIATE & INCREMENTAL ORTHOGONAL SIGNAL AUDIT ({len(adv_pool)} SETUPS ACROSS {len({r['ticker'] for r in adv_pool})} T&Q STOCKS)")
print("=" * 155)
print(f"| {'Rk':<3} | {'Candidate Feature (48 Total)':<34} | {'Spearman r (p-val)':<20} | {'Q1 Avg (WR)':<18} | {'Q3 Avg (WR)':<18} | {'Q3-Q1 Spread':<13} | {'Partial r | Core8':<17} |")
print(f"|{'-'*5}|{'-'*36}|{'-'*22}|{'-'*20}|{'-'*20}|{'-'*15}|{'-'*19}|")

# Compute residual of each candidate feature after linear projection onto X_core8 (to measure true incremental signal beyond OHLCV!)
Z_c8 = (X_core8 - X_core8.mean(axis=0)) / np.maximum(X_core8.std(axis=0), 1e-8)
Z_c8_b = np.column_stack([np.ones(len(Z_c8)), Z_c8])
P_c8 = Z_c8_b @ np.linalg.pinv(Z_c8_b.T @ Z_c8_b) @ Z_c8_b.T

u_rows = []
for j, fname in enumerate(CAND48_NAMES):
    col = X_cand48[:, j]
    col_clean = np.nan_to_num(col, 0.0)
    if np.std(col_clean) < 1e-8:
        continue
    sr, sp = sp_stats.spearmanr(col_clean, y_adv)
    # Residualize against Core 8 OHLCV
    res_col = col_clean - (P_c8 @ col_clean)
    pr, pp = sp_stats.spearmanr(res_col, y_adv)
    q33, q67 = np.percentile(col_clean, [33.3, 66.7])
    q1_m, q3_m = col_clean <= q33, col_clean >= q67
    q1_wr, q1_avg = (y_adv[q1_m] > 0).mean() * 100.0, y_adv[q1_m].mean()
    q3_wr, q3_avg = (y_adv[q3_m] > 0).mean() * 100.0, y_adv[q3_m].mean()
    u_rows.append((max(abs(sr), abs(pr)), j, fname, sr, sp, q1_avg, q1_wr, q3_avg, q3_wr, q3_avg - q1_avg, pr, pp))

u_rows.sort(key=lambda x: -x[0])
for rk, (_, j, fname, sr, sp, q1a, q1w, q3a, q3w, spr, pr, pp) in enumerate(u_rows[:25], 1):
    s_sig = "**" if sp < 0.01 else ("*" if sp < 0.05 else "  ")
    p_sig = "**" if pp < 0.01 else ("*" if pp < 0.05 else "  ")
    print(
        f"| {rk:>2}  | {fname:<34} | {sr:+6.3f} (p={sp:.3f}){s_sig}  | {q1a:+6.2f}% ({q1w:4.1f}%) | "
        f"{q3a:+6.2f}% ({q3w:4.1f}%) | {spr:+6.2f}%       | {pr:+6.3f} (p={pp:.3f}){p_sig} |"
    )

# 5. Controlled Marginal OOS Walk-Forward Test:
# Keep Base 20 features FIXED (Core 8 OHLCV + In-Fold Top 12 of Old33 T&Q), and test adding:
#   (a) Each top candidate feature individually (+1 feature: 20f -> 21f)
#   (b) Replace unweighted pm_obi/pm_ofi with Volume-Weighted PM OBI/OFI + Add Top Orthogonal Parquet Features
print("\n" + "=" * 155)
print("SECTION 2: CONTROLLED MARGINAL WALK-FORWARD OOS ABLATION (KEEPING BASE 20 FEATURES FIXED + ADDING NEW FEATURES WITHOUT DISPLACEMENT)")
print("=" * 155)

top_cand_indices = [row[1] for row in u_rows[:15]]

# Precompute fold-by-fold base matrices so testing 15 candidates + combinations takes < 5 seconds!
folds = []
for m_idx in range(min_train_months, len(all_months)):
    te_ym = all_months[m_idx]
    tr_m = ym_adv < te_ym
    te_m = ym_adv == te_ym
    if np.sum(tr_m) < 50 or np.sum(te_m) == 0:
        continue
    te_indices = np.where(te_m)[0]
    corrs_tr = [abs(float(np.nan_to_num(sp_stats.spearmanr(X_old33[tr_m, j], y_adv[tr_m])[0]))) if np.std(X_old33[tr_m, j]) > 1e-8 else 0.0 for j in range(33)]
    top12_idx = np.argsort(corrs_tr)[-12:]
    X_b20_tr = np.hstack([X_core8[tr_m], X_old33[tr_m][:, top12_idx]])
    X_b20_te = np.hstack([X_core8[te_m], X_old33[te_m][:, top12_idx]])
    folds.append((te_ym, tr_m, te_m, te_indices, X_b20_tr, X_b20_te))

with open(FAST_CACHE, "rb") as f:
    all_trading_dates = pickle.load(f)["all_trading_dates"]
oos_dates = [d for d in all_trading_dates if "2024-11-01" <= d <= "2026-11-06"]
last12_dates = [d for d in all_trading_dates if "2025-11-03" <= d <= "2026-11-06"]
last4_dates = [d for d in all_trading_dates if "2026-07-01" <= d <= "2026-11-06"]

def eval_feature_addition(extra_cols_idx, use_m7=True):
    pred_m4 = {}
    pred_m7 = {}
    pred_ens = {}
    oos_y, oos_p4, oos_p7 = [], [], []
    for te_ym, tr_m, te_m, te_indices, X_b20_tr, X_b20_te in folds:
        if extra_cols_idx:
            X_tr = np.hstack([X_b20_tr, X_cand48[tr_m][:, extra_cols_idx]])
            X_te = np.hstack([X_b20_te, X_cand48[te_m][:, extra_cols_idx]])
            X_tq_tr = np.hstack([X_old33[tr_m], X_cand48[tr_m][:, extra_cols_idx]])
            X_tq_te = np.hstack([X_old33[te_m], X_cand48[te_m][:, extra_cols_idx]])
        else:
            X_tr, X_te = X_b20_tr, X_b20_te
            X_tq_tr, X_tq_te = X_old33[tr_m], X_old33[te_m]

        _, p4 = fit_m4_logistic(X_tr, y_adv[tr_m], X_te)
        if use_m7:
            _, p7 = fit_m7_gbdt_d3(X_tq_tr, y_adv[tr_m], X_tq_te)
        else:
            p7 = p4
        for loc_i, r_idx in enumerate(te_indices):
            rec = adv_pool[r_idx]
            k = (rec["ticker"], rec["date"])
            pred_m4[k] = float(p4[loc_i])
            pred_m7[k] = float(p7[loc_i])
            pred_ens[k] = 0.50 * float(p4[loc_i]) + 0.50 * float(p7[loc_i])
            oos_y.append(y_adv[r_idx])
            oos_p4.append(float(p4[loc_i]))
            oos_p7.append(float(p7[loc_i]))

    ic_m4 = float(sp_stats.spearmanr(oos_p4, oos_y)[0])
    ic_m7 = float(sp_stats.spearmanr(oos_p7, oos_y)[0])
    ic_ens = float(sp_stats.spearmanr(0.5*np.array(oos_p4)+0.5*np.array(oos_p7), oos_y)[0])
    return pred_m4, pred_m7, pred_ens, ic_m4, ic_m7, ic_ens

def sim_port(pred_dict, dates_h, thresh=0.35):
    day_map = defaultdict(list)
    for r in adv_pool:
        if r["ym"] < oos_start_ym or r["gap_pct"] < 3.0:
            continue
        sc = pred_dict.get((r["ticker"], r["date"]), -999.0)
        if sc >= thresh:
            day_map[r["date"]].append((sc, r))
    cap, pk, mdd = 100_000.0, 100_000.0, 0.0
    t_pnls, d_rets = [], []
    for d in dates_h:
        cands = day_map.get(d, [])
        if not cands:
            d_rets.append(0.0); continue
        cs = sorted(cands, key=lambda x: (x[1]["entry_hm"], -x[0]))[:2]
        w = min(1.0 / len(cs), 0.50)
        dr = sum(w * float(r["raw_pnl_pct"]) for _, r in cs)
        t_pnls.extend([float(r["raw_pnl_pct"]) for _, r in cs])
        d_rets.append(dr)
        cap *= (1.0 + dr / 100.0)
        if cap > pk: pk = cap
        dd = (pk - cap) / pk * 100.0
        if dd > mdd: mdd = dd
    da = np.array(d_rets)
    twr = (np.array(t_pnls) > 0).mean() * 100.0 if t_pnls else 0.0
    sh = (da.mean() / max(da.std(ddof=1), 1e-8)) * np.sqrt(252)
    return len(t_pnls), twr, float(da.sum()), cap, mdd, sh

p4_0, p7_0, pe_0, ic4_0, ic7_0, ice_0 = eval_feature_addition([], use_m7=True)
n4_0, wr4_0, tot4_0, eq4_0, dd4_0, sh4_0 = sim_port(p4_0, oos_dates)
n7_0, wr7_0, tot7_0, eq7_0, dd7_0, sh7_0 = sim_port(p7_0, oos_dates)
ne_0, wre_0, tote_0, eqe_0, dde_0, she_0 = sim_port(pe_0, oos_dates)

print(f"| {'Feature Addition to Fixed Base 20 (Core8 + Top12 T&Q)':<45} | {'OOS IC (M4 / M7 / Ens)':<24} | {'M4 Logistic Comp Eq (Tot% / Sh)':<33} | {'M7 Deep GBDT Comp Eq (Tot% / Sh)':<33} |")
print(f"|{'-'*47}|{'-'*26}|{'-'*35}|{'-'*35}|")
print(
    f"| {'0. BASELINE (Existing 33 T&Q Only, 0 New)':<45} | {ic4_0:+.3f} / {ic7_0:+.3f} / {ice_0:+.3f}  | "
    f"${eq4_0:>9,.0f} ({tot4_0:+5.1f}%, Sh {sh4_0:+.2f})   | ${eq7_0:>9,.0f} ({tot7_0:+5.1f}%, Sh {sh7_0:+.2f})   |"
)

improving_m4 = []
improving_m7 = []
for idx_c in top_cand_indices:
    fname = CAND48_NAMES[idx_c]
    p4_c, p7_c, pe_c, ic4_c, ic7_c, ice_c = eval_feature_addition([idx_c], use_m7=True)
    n4_c, wr4_c, tot4_c, eq4_c, dd4_c, sh4_c = sim_port(p4_c, oos_dates)
    n7_c, wr7_c, tot7_c, eq7_c, dd7_c, sh7_c = sim_port(p7_c, oos_dates)
    if sh4_c > sh4_0 or ic4_c > ic4_0 + 0.001:
        improving_m4.append(idx_c)
    if sh7_c > sh7_0 or ic7_c > ic7_0 + 0.001:
        improving_m7.append(idx_c)
    m4_flag = "★" if eq4_c > eq4_0 else " "
    m7_flag = "★" if eq7_c > eq7_0 else " "
    print(
        f"| + {fname:<43} | {ic4_c:+.3f} / {ic7_c:+.3f} / {ice_c:+.3f}  | "
        f"${eq4_c:>9,.0f} ({tot4_c:+5.1f}%, Sh {sh4_c:+.2f}){m4_flag} | ${eq7_c:>9,.0f} ({tot7_c:+5.1f}%, Sh {sh7_c:+.2f}){m7_flag} |"
    )

# Test combining the top M4-improving features and top M7-improving features
print(f"|{'-'*47}|{'-'*26}|{'-'*35}|{'-'*35}|")
for combo_lbl, combo_indices in [
    ("COMBO A: Top M4-Improving New Features", improving_m4[:4]),
    ("COMBO B: Top M7-Improving New Features", improving_m7[:4]),
    ("COMBO C: Vol-Weighted PM (deep10+11) + PrevEOD (deep17)", [30+9, 30+10, 30+16]),
    ("COMBO D: KyleLambda (new07) + Torque (new09) + Deep17", [6, 8, 30+16]),
]:
    if not combo_indices:
        continue
    p4_c, p7_c, pe_c, ic4_c, ic7_c, ice_c = eval_feature_addition(combo_indices, use_m7=True)
    n4_c, wr4_c, tot4_c, eq4_c, dd4_c, sh4_c = sim_port(p4_c, oos_dates)
    n7_c, wr7_c, tot7_c, eq7_c, dd7_c, sh7_c = sim_port(p7_c, oos_dates)
    print(
        f"| {combo_lbl:<45} | {ic4_c:+.3f} / {ic7_c:+.3f} / {ice_c:+.3f}  | "
        f"${eq4_c:>9,.0f} ({tot4_c:+5.1f}%, Sh {sh4_c:+.2f})   | ${eq7_c:>9,.0f} ({tot7_c:+5.1f}%, Sh {sh7_c:+.2f})   |"
    )
