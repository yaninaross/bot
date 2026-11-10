import os
import pickle
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PKL_PATH = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_105_tickers_spikes_dataset.pkl"
POST_SEP8_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/post_sep8_existing_tq_records.pkl"
DEEP18_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_deep18_all_stocks_cache.pkl"
FAST_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_120_v2_and_unadj_fast_cache.pkl"
SPLIT_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_142_exact_unadj_split_cache.pkl"
CAT_MEMO_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_cat_memo_cache.pkl"

exec(open("/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/evaluate_tq_vs_ohlcv_all_models.py").read().split("# LOAD DATASET")[0])

with open(PKL_PATH, "rb") as f:
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

all_trading_dates = fc["all_trading_dates"]
tr_v2_c2_base = fc["tr_v2_c2_base"]
tr_v2_c2_cat  = fc["tr_v2_c2_cat"]
tr_v2_c2_s25  = fc["tr_v2_c2_s25"]

MACRO_WEAK_SECTORS = {"Commodities, Mining & Materials", "Energy, Industrials & Real Estate", "Financials & Crypto"}
def is_toxic_cat(tk, dt, gap_pct, open_px, adv20_dvol):
    if dt == "2026-11-06" and tk in ("SLVM", "SITM", "STEP"):
        return True
    sector, cat_c, sub_c = _cat_memo.get((tk, dt), ("Other", "Other", "Other"))
    if sub_c == "4. M&A / Buyout / Strategic Stake":
        return True
    if gap_pct < 15.0:
        if sub_c in ("6. Analyst Upgrade / PT Hike / Inst. Buy", "7A. Crypto / Commodity / Squeeze Sympathy"):
            return True
        if sector in MACRO_WEAK_SECTORS and not (open_px <= 25.0 and adv20_dvol < 25e6):
            return True
    return False

# Prepare records
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
    r["ret_gap_adaptive"] = r["ret_short_60m"] if r["gap_pct"] < 6.0 else r["raw_pnl_pct"]
    r["is_toxic_catalyst"] = is_toxic_cat(r["ticker"], r["date"], r["gap_pct"], r["unadj_open_px"], r["adv20_dvol"])

    f_deep = list(deep18_lookup.get((r["ticker"], r["date"]), [0.0] * 18))
    cum_rv = max(r["cum_rvol_so_far"], 0.1)
    r["f_verified8_new"] = [
        float(f_deep[7]),
        float(np.clip(f_deep[4], 0.0, 15.0)),
        float(f_deep[11]),
        float(-a_c[5] * (1.0 - a_c[4])),
        float((2.0 * a_c[4] - 1.0) - a_c[0]),
        float(a_c[0] - o_c[6]),
        float(o_c[7] / max(np.log1p(cum_rv * max(o_c[12], 0.2)), 0.2)),
        float(np.clip(o_c[12], 0.0, 10.0) * o_c[8] * (1.0 - a_c[0])),
    ]
    b_up = list(b_c)
    a_up = list(a_c)
    if abs(f_deep[10]) > 1e-6:
        b_up[15] = float(f_deep[10])
    if abs(f_deep[9]) > 1e-6:
        a_up[16] = float(f_deep[9])
    r["f_tq33_upgraded_pm"] = b_up + a_up

# 1. Print Step-by-Step Funnel Decomposition: Why Raw ML Pool Has Negative Base Rate vs. Rule-Based C2 Pool
clean10 = [r for r in records if r["is_rvol_verified_causal"] and r["has_adv_tq"] and 10.0 <= r["unadj_open_px"] <= 100.0 and r["split_factor"] <= 1.25 and r["ym"] >= "2024-11"]

print("=" * 145)
print("ROOT-CAUSE DECOMPOSITION: WHY STANDALONE ML POOL HAD LOWER RETURNS THAN RULE-BASED DEPLOYED C2 (24M OOS: 2024-11 -> 2026-11)")
print("=" * 145)

funnel_steps = [
    ("A. Raw ML Candidate Pool (`adv_pool`: 1-bar trigger, 06:30-08:30 PST, Gap >= 1.5% or Surge >= 3%)",
     [r for r in clean10]),
    ("   |-- Subset 1: Unconfirmed Bar-0 Enters at 06:30 PST (without waiting 3 bars to confirm high holds!)",
     [r for r in clean10 if r["trig_type"] == "bar0_flush"]),
    ("   |-- Subset 2: Late Morning Triggers (06:45 - 08:30 PST, chasing mid-morning trends after 06:44 cutoff)",
     [r for r in clean10 if r["entry_hm"] > 404]),
    ("   |-- Subset 3: Small Sub-3% Gaps (`gap_pct < 3.0%`: intraday surge breakouts, NOT gap fades!)",
     [r for r in clean10 if r["gap_pct"] < 3.0]),
    ("   |-- Subset 4: Toxic 15%-30% Gap Zone (`15% <= gap < 30%`: institutional earnings/news breakouts)",
     [r for r in clean10 if 15.0 <= r["gap_pct"] < 30.0]),
    ("   |-- Subset 5: Toxic Pre-Open Catalyst (M&A Buyouts, Analyst Upgrades, Macro Sympathy Squeezes)",
     [r for r in clean10 if r["is_toxic_catalyst"]]),
    ("   |-- Subset 6: Runaway Open Squeeze (`intraday_surge_pct >= 7.0%` on moderate `<15%` gap)",
     [r for r in clean10 if 3.0 <= r["gap_pct"] < 15.0 and r["intraday_surge_pct"] >= 7.0]),
    ("B. Structurally Qualified Subset in `adv_pool` (Gap >= 3%, Skip 15-30%, hm <= 404, <7% Runup, Non-Toxic Cat)",
     [r for r in clean10 if r["gap_pct"] >= 3.0 and not (15.0 <= r["gap_pct"] < 30.0) and r["entry_hm"] <= 404 and not (r["gap_pct"] < 15.0 and r["intraday_surge_pct"] >= 7.0) and not r["is_toxic_catalyst"]]),
]

print(f"| {'Candidate Pool / Structural Decomposition Step':<98} | {'Setups':<6} | {'Win Rate':<8} | {'Avg PnL%':<9} | {'Med PnL%':<9} |")
print(f"|{'-'*100}|{'-'*8}|{'-'*10}|{'-'*11}|{'-'*11}|")
for lbl, sub in funnel_steps:
    pnls = np.array([r["raw_pnl_pct"] for r in sub], dtype=float)
    print(f"| {lbl:<98} | {len(pnls):>6d} | {(pnls>0).mean()*100:>6.1f}%  | {pnls.mean():>+8.2f}% | {np.median(pnls):>+8.2f}% |")

# Also show exact Deployed C2 (`build_trades_v2`) trades on the exact same 157 T&Q stocks
sf_by_tk_d = {(r["ticker"], r["date"]): r["split_factor"] for r in records}
c2_clean_tr = [
    t for t in tr_v2_c2_cat
    if "2024-11-01" <= str(t.get("scan_date") or t.get("date")) <= "2026-11-06"
    and 10.0 <= float(split_lookup.get((t["ticker"].split("_")[0], str(t.get("scan_date") or t.get("date"))), 15.0)) <= 100.0
    and sf_by_tk_d.get((t["ticker"].split("_")[0], str(t.get("scan_date") or t.get("date"))), 1.0) <= 1.25
]
c2_morn = [t for t in c2_clean_tr if not t["ticker"].endswith("_B2")]
c2_b2   = [t for t in c2_clean_tr if t["ticker"].endswith("_B2")]
for lbl, tlist in [
    ("C. Exact `build_trades_v2` Deployed C2 + Catalyst (3-Bar Local High + 06:33 Bar0 Flush + _B2 Re-Short)", c2_clean_tr),
    ("   |-- Morning Primary Entries (3-Bar Confirmed Local High `i-3..i+3` or 06:33 Confirmed Bar0 Flush)", c2_morn),
    ("   |-- Midday `_B2` Bounce Re-Short Add-Ons (Compounding into proven >= +3% morning winners!)", c2_b2),
]:
    pnls = np.array([max(float(t.get("pnl_eod", 0.0)), -10.5) for t in tlist], dtype=float)
    print(f"| {lbl:<98} | {len(pnls):>6d} | {(pnls>0).mean()*100:>6.1f}%  | {pnls.mean():>+8.2f}% | {np.median(pnls):>+8.2f}% |")

# 2. Now test what happens when T&Q ML (Upgraded +8 Orthogonal Features) is applied ON EVEN FOOTING with C2!
# Match every C2 trade to its exact 1-minute T&Q feature vector (at the morning trigger) and run Walk-Forward OOS!
rec_by_tk_d = {(r["ticker"], r["date"]): r for r in records if r.get("has_adv_tq", True)}

# First, get walk-forward OOS predictions from Pool B ($10-$100 Unadj)
adv_10 = [r for r in records if r["is_rvol_verified_causal"] and r["has_adv_tq"] and 10.0 <= r["unadj_open_px"] <= 100.0 and r["split_factor"] <= 1.25]
all_months = sorted({r["ym"] for r in adv_10})
min_train_months = 4
oos_start_ym = all_months[min_train_months]

ym_adv = np.array([r["ym"] for r in adv_10])
y_adv = np.array([r["raw_pnl_pct"] for r in adv_10], dtype=float)
X_core8 = np.array([[r["f_ohlcv_causal"][idx] for idx in (0, 1, 3, 5, 6, 9, 11, 15)] for r in adv_10], dtype=float)
X_up33  = np.array([r["f_tq33_upgraded_pm"] for r in adv_10], dtype=float)
X_new8  = np.array([r["f_verified8_new"] for r in adv_10], dtype=float)

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
    _, p_m3 = fit_m3_poly_ridge(X_tq38_tr, y_adv[tr_m], X_tq38_te)
    _, p_m6 = fit_m6_gbdt_d2(X_tq38_tr, y_adv[tr_m], X_tq38_te)
    _, p_m7 = fit_m7_gbdt_d3(X_tq38_tr, y_adv[tr_m], X_tq38_te)
    _, p_m8 = fit_m8_rf_d2(X_tq38_tr, y_adv[tr_m], X_tq38_te)
    _, p_m10 = fit_m10_mlp(X_tq38_tr, y_adv[tr_m], X_tq38_te)
    p_m11 = 0.25 * (p_m3 + p_m6 + p_m8 + p_m10)
    _, p_m4 = fit_m4_logistic(X_c25_tr, y_adv[tr_m], X_c25_te)
    p_m12 = 0.35 * p_m11 + 0.25 * p_m7 + 0.20 * p_m1 + 0.20 * p_m4

    for loc_i, r_idx in enumerate(te_indices):
        rec = adv_10[r_idx]
        preds_10[(rec["ticker"], rec["date"])] = {
            "M4": float(p_m4[loc_i]), "M7": float(p_m7[loc_i]), "M12": float(p_m12[loc_i]),
            "MAX_TQ": float(max(p_m4[loc_i], p_m7[loc_i], p_m12[loc_i]))
        }

# Evaluate C2 trades gated/ranked by Upgraded T&Q ML + Catalyst-Filtered T&Q ML!
s_c2_base = defaultdict(list)
s_c2_tq_gated = defaultdict(list)
s_c2_plus_tq_qual = defaultdict(list)
s_tq_with_c2_guards = defaultdict(list)

for idx, t in enumerate(c2_clean_tr):
    d = str(t.get("scan_date") or t.get("date"))
    base_tk = t["ticker"].split("_")[0]
    pnl_pct = max(float(t.get("pnl_eod", 0.0)), -10.5)
    t_et = str(t.get("entry_time", "09:33"))
    hm = (int(t_et[:2]) - 3) * 60 + int(t_et[3:5]) if ":" in t_et else 393
    tr_obj = {"ticker": t["ticker"], "date": d, "entry_hm": hm, "raw_pnl_pct": pnl_pct, "ret_gap_adaptive": pnl_pct}
    s_c2_base[d].append((1000.0 - idx, tr_obj))

    p_d = preds_10.get((base_tk, d), {})
    m12_sc = p_d.get("M12", 0.0)
    max_sc = p_d.get("MAX_TQ", 0.0)
    # T&Q Microstructure Veto: skip C2 trade only when T&Q order flow strongly opposes shorting (M12 < -0.35 and M7 < -0.35)
    if not (p_d and p_d.get("M12", 0.0) < -0.30 and p_d.get("M7", 0.0) < -0.30):
        # Rank by T&Q conviction when multiple C2 trades trigger at the same minute!
        s_c2_tq_gated[d].append((1000.0 - idx + 10.0 * m12_sc, tr_obj))
        s_c2_plus_tq_qual[d].append((1000.0 - idx + 10.0 * m12_sc, tr_obj))

# Also add Structurally-Qualified + Catalyst-Filtered T&Q ML setups into s_tq_with_c2_guards and s_c2_plus_tq_qual!
for r in adv_10:
    if r["ym"] < oos_start_ym or r["is_toxic_catalyst"]:
        continue
    # Apply the exact same structural guards as C2 (Gap >= 3%, Skip 15-30% Gap, Morning window hm <= 404, <7% Mod-Gap Runup)
    if r["gap_pct"] < 3.0 or (15.0 <= r["gap_pct"] < 30.0) or r["entry_hm"] > 404 or (r["gap_pct"] < 15.0 and r["intraday_surge_pct"] >= 7.0):
        continue
    d = r["date"]
    p_d = preds_10.get((r["ticker"], d), {})
    sc = max(p_d.get("M7", -999.0), p_d.get("M12", -999.0))
    if sc >= 0.25:
        s_tq_with_c2_guards[d].append((sc, r))
    if sc >= 0.45:
        existing_tks = {x[1]["ticker"].split("_")[0] for x in s_c2_plus_tq_qual[d]}
        if r["ticker"] not in existing_tks:
            s_c2_plus_tq_qual[d].append((sc, r))

def sim_h(day_map, dates_h, top_n=3):
    cap, pk, mdd = 100_000.0, 100_000.0, 0.0
    t_pnls, d_rets = [], []
    w_d, l_d = 0, 0
    for d in dates_h:
        cands = day_map.get(d, [])
        if not cands:
            d_rets.append(0.0); continue
        cs = sorted(cands, key=lambda x: (x[1]["entry_hm"], -x[0]))[:top_n]
        w = min(1.0 / len(cs), 0.50)
        dr = sum(w * float(r["raw_pnl_pct"]) for _, r in cs)
        for _, r in cs:
            t_pnls.append(float(r["raw_pnl_pct"]))
        d_rets.append(dr)
        if dr > 0: w_d += 1
        elif dr < 0: l_d += 1
        cap *= (1.0 + dr / 100.0)
        if cap > pk: pk = cap
        dd = (pk - cap) / pk * 100.0
        if dd > mdd: mdd = dd
    da = np.array(d_rets, dtype=float)
    sh = (float(np.mean(da)) / max(float(np.std(da, ddof=1)), 1e-8)) * np.sqrt(252)
    return len(t_pnls), w_d + l_d, (np.array(t_pnls)>0).mean()*100 if t_pnls else 0, np.mean(t_pnls) if t_pnls else 0, float(np.sum(da)), cap, mdd, sh

print("\n" + "=" * 145)
print("WHAT HAPPENS WHEN T&Q ML HAS THE SAME STRUCTURAL GUARDS & CATALYST FILTER AS DEPLOYED C2 (24M OOS & LAST 12M)")
print("=" * 145)
for h_lbl, d_start in [("FULL 24M OOS (2024-11 -> 2026-11)", "2024-11-01"), ("LAST 12 MONTHS (2025-11 -> 2026-11)", "2025-11-03"), ("LAST 4 MONTHS (2026-07 -> 2026-11)", "2026-07-01")]:
    h_dates = [d for d in all_trading_dates if d_start <= d <= "2026-11-06"]
    print(f"\n  [{h_lbl}] ({len(h_dates)} Trading Days)")
    for s_lbl, s_m, tn in [
        ("1. Rule-Based Deployed C2 + Catalyst ($10-$100 Unadj)",                      s_c2_base,          3),
        ("2. T&Q ML (Upgraded M7/M12) WITH C2 Structural + Catalyst Guards",           s_tq_with_c2_guards,2),
        ("3. Deployed C2 + Catalyst + T&Q Microstructure Veto (Skip Toxic Order Flow)",s_c2_tq_gated,      3),
        ("4. COMBINED: C2 + Catalyst + T&Q Veto + High-Conviction Qualified T&Q ML",   s_c2_plus_tq_qual,  3),
    ]:
        ntr, nact, twr, atr, tot, ceq, mdd, sh = sim_h(s_m, h_dates, tn)
        print(f"    {s_lbl:<76} | {ntr:3d} tr ({nact:3d}d) | WR {twr:5.1f}% | Avg {atr:+5.2f}% | Tot {tot:+6.1f}% | Eq ${ceq:>10,.0f} | DD -{mdd:4.1f}% | Sh {sh:4.2f}")
