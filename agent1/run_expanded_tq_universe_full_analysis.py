import os
import io
import time
import pickle
import warnings
import requests
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
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")

PKL_PATH = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_105_tickers_spikes_dataset.pkl"
POST_SEP8_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/post_sep8_existing_tq_records.pkl"
DEEP18_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_deep18_all_stocks_cache.pkl"
FAST_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_120_v2_and_unadj_fast_cache.pkl"
SPLIT_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_142_exact_unadj_split_cache.pkl"
TICK4_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_tick4_new_cols_cache.pkl"
CAT_MEMO_CACHE = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/tq_cat_memo_cache.pkl"

CONTEXT_SKIP = {"SPY", "VIX", "TLT", "IWM", "SMH", "BTCUSD", "ETHUSD", "ITA", "XLE", "BITF", "MIGI"}
ADV_MICRO_16_COLS = [
    "LR_OFI", "LR_VPIN", "LR_Buy_Ratio", "LR_Size_Imbalance",
    "PE", "LZC", "Tick_Volatility", "Shock_Persist",
    "Trade_Size_CV", "Trade_Intensity", "Quote_Intensity", "Spread_CV",
    "Odd_Lot_OFI", "Block_OFI", "Signed_Semivariance", "Ask_Step_Down_Rate"
]

client = storage.Client.from_service_account_json(LOCAL_SA_KEY, project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

# 1. Load model definitions & existing caches
exec(open("/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/evaluate_tq_vs_ohlcv_all_models.py").read().split("# LOAD DATASET")[0])
exec(open("/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/update_and_run_tq_backtest.py").read().split("if new_tickers_to_extract:")[0])

with open(DEEP18_CACHE, "rb") as f:
    deep18_lookup = pickle.load(f)

if os.path.exists(TICK4_CACHE):
    with open(TICK4_CACHE, "rb") as f:
        tick4_lookup = pickle.load(f)
else:
    tick4_lookup = {}

# Check which tickers on GCS are new OR were updated with the 16-col schema
print("=" * 145)
print(f"1. GCS INVENTORY CHECK: {len(adv_blobs)} Adv Micro Parquets & {len(inf_blobs)} Master_Inference_v2 Parquets")
print(f"   Newly Completed Tickers Since Last Extraction ({len(new_tickers_to_extract)}): {new_tickers_to_extract}")
print("=" * 145)

_orig_download_parquet = download_parquet
_pq_mem_cache = {}
def download_parquet_cached(blob):
    bname = blob.name
    if bname not in _pq_mem_cache:
        _pq_mem_cache[bname] = _orig_download_parquet(blob)
    return _pq_mem_cache[bname].copy()
download_parquet = download_parquet_cached

def extract_ticker_all_in_one(tk):
    """Downloads Master_Inference_v2 & Microstructure_Advanced ONCE for `tk` and extracts:
       1) spike_records (with 20 OHLCV + 16 Base T&Q + 17 Adv T&Q)
       2) deep18_map ((tk, date) -> 18 multi-day/pre-market/trapped-vol features)
       3) tick4_map ((tk, date) -> [Odd_Lot_OFI, Block_OFI, Signed_Semivariance, Ask_Step_Down_Rate] if 16-col schema present)
    """
    recs_tk, inf_tk = extract_one_ticker(tk)
    deep_map = {}
    t4_map = {}
    has_16_cols = False
    if not recs_tk:
        _pq_mem_cache.pop(inf_blobs[tk].name, None)
        if tk in adv_blobs:
            _pq_mem_cache.pop(adv_blobs[tk].name, None)
        return tk, recs_tk, inf_tk, deep_map, t4_map, has_16_cols

    try:
        df_inf = download_parquet(inf_blobs[tk])
        df_adv = download_parquet(adv_blobs[tk])
        _pq_mem_cache.pop(inf_blobs[tk].name, None)
        _pq_mem_cache.pop(adv_blobs[tk].name, None)
        cutoff = pd.Timestamp("2024-06-01", tz="UTC")
        df_inf = df_inf[df_inf.index >= cutoff]
        df_adv = df_adv[df_adv.index >= cutoff]
        has_16_cols = all(c in df_adv.columns for c in ("Odd_Lot_OFI", "Block_OFI", "Signed_Semivariance", "Ask_Step_Down_Rate"))
        acols = [c for c in ADV_MICRO_16_COLS if c in df_adv.columns]
        overlap = [c for c in acols if c in df_inf.columns]
        if overlap:
            df_inf = df_inf.drop(columns=overlap)
        df = df_inf.join(df_adv[acols], how="left")
        df.index = df.index.tz_convert("America/Los_Angeles")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df["date_str"] = df.index.strftime("%Y-%m-%d")
        df["hm"] = df.index.hour * 60 + df.index.minute

        day_groups = {}
        daily_micro = []
        for d_s, grp in df.groupby("date_str"):
            rth = grp[(grp["hm"] >= 390) & (grp["hm"] < 780)]
            if len(rth) < 30:
                continue
            day_groups[d_s] = grp
            tvol = rth["Total_Vol"].fillna(rth["Volume"]).fillna(0.0).values
            tcnt = np.maximum(rth["Trade_Count"].fillna(1.0).values, 1.0)
            avg_sz = float(np.sum(tvol) / np.sum(tcnt)) if np.sum(tcnt) > 0 else 100.0
            mean_sp_pct = float((rth["Spread"].fillna(0.0) / np.maximum(rth["Close"], 0.1) * 100.0).mean())
            mean_tcnt = float(tcnt.mean())
            mean_tvolat = float(rth["Tick_Volatility"].fillna(0.0).mean() * 100.0) if "Tick_Volatility" in rth else 0.1
            mean_qint = float(rth["Quote_Intensity"].fillna(1.0).mean()) if "Quote_Intensity" in rth else 1.0
            last_hr = rth[rth["hm"] >= 720]
            eod_ofi = float(last_hr["LR_OFI"].fillna(0.0).mean()) if len(last_hr) > 0 and "LR_OFI" in last_hr else 0.0
            eod_obi = float(last_hr["OBI"].fillna(0.0).mean()) if len(last_hr) > 0 else 0.0
            daily_micro.append({
                "date": d_s, "avg_sz": avg_sz, "sp_pct": mean_sp_pct,
                "tcnt": mean_tcnt, "tvolat": mean_tvolat, "qint": mean_qint,
                "eod_ofi": eod_ofi, "eod_obi": eod_obi
            })

        dm_df = pd.DataFrame(daily_micro).sort_values("date").reset_index(drop=True)
        date_to_idx = {d: i for i, d in enumerate(dm_df["date"])}

        for r in recs_tk:
            d_s = r["date"]
            i_d = date_to_idx.get(d_s)
            if i_d is None or i_d < 20 or d_s not in day_groups:
                continue
            grp = day_groups[d_s]
            rth = grp[(grp["hm"] >= 390) & (grp["hm"] < 780)]
            pm = grp[(grp["hm"] >= 60) & (grp["hm"] < 390) & (grp["Volume"] > 0)]
            hist20 = dm_df.iloc[i_d - 20:i_d]
            base20_sz = max(float(hist20["avg_sz"].mean()), 1.0)
            base20_sp = max(float(hist20["sp_pct"].mean()), 1e-4)
            base20_tcnt = max(float(hist20["tcnt"].mean()), 1.0)
            base20_tvolat = max(float(hist20["tvolat"].mean()), 1e-4)
            base20_qint = max(float(hist20["qint"].mean()), 0.1)
            prev_eod_ofi = float(dm_df.iloc[i_d - 1]["eod_ofi"])
            prev_eod_obi = float(dm_df.iloc[i_d - 1]["eod_obi"])

            eh = r["entry_hm"]
            rth_c = rth[rth["hm"] <= eh]
            if len(rth_c) == 0:
                continue
            entry_px = r["entry_px"]
            curr_o = r["open_px"]

            tvol_c = rth_c["Total_Vol"].fillna(rth_c["Volume"]).fillna(0.0).values
            tcnt_c = np.maximum(rth_c["Trade_Count"].fillna(1.0).values, 1.0)
            curr_avg_sz = float(np.sum(tvol_c) / np.sum(tcnt_c))
            f01 = float(np.log1p(curr_avg_sz) - np.log1p(base20_sz))
            f02 = float((float(rth_c["Spread"].fillna(0.0).iloc[-1] / max(entry_px, 0.1) * 100.0)) / base20_sp)
            f03 = float(np.log1p(float(tcnt_c.mean()) / base20_tcnt))
            f04 = float((float(rth_c["Tick_Volatility"].fillna(0.0).iloc[-1] * 100.0)) / base20_tvolat)
            f05 = float(float(rth_c["Quote_Intensity"].fillna(1.0).iloc[-1]) / base20_qint)

            if len(pm) >= 3 and pm["Volume"].sum() > 0:
                pm_v = pm["Volume"].values.astype(float)
                pm_c = pm["Close"].values.astype(float)
                pm_h = pm["High"].values.astype(float)
                pm_vwap = float(np.sum(pm_c * pm_v) / np.sum(pm_v))
                f06 = float((curr_o / pm_vwap - 1.0) * 100.0)
                f07 = float((entry_px / pm_vwap - 1.0) * 100.0)
                f08 = float(np.sum(pm_v[pm_c > entry_px]) / np.sum(pm_v))
                f09 = float(390 - int(pm["hm"].values[int(np.argmax(pm_h))]))
                f10 = float(np.sum(pm["LR_OFI"].fillna(0.0).values * pm_v) / np.sum(pm_v)) if "LR_OFI" in pm else 0.0
                f11 = float(np.sum(pm["OBI"].fillna(0.0).values * pm_v) / np.sum(pm_v))
            else:
                f06, f07, f08, f09, f10, f11 = 0.0, 0.0, 0.0, 30.0, 0.0, 0.0

            rth_v = rth_c["Volume"].values.astype(float)
            rth_close = rth_c["Close"].values.astype(float)
            rth_open = rth_c["Open"].values.astype(float)
            rth_high = rth_c["High"].values.astype(float)
            rth_low = rth_c["Low"].values.astype(float)
            rth_ofi = rth_c["LR_OFI"].fillna(0.0).values.astype(float) if "LR_OFI" in rth_c else np.zeros(len(rth_c))
            rth_pe = rth_c["PE"].fillna(0.8).values.astype(float) if "PE" in rth_c else np.full(len(rth_c), 0.8)
            rth_lzc = rth_c["LZC"].fillna(0.8).values.astype(float) if "LZC" in rth_c else np.full(len(rth_c), 0.8)

            tot_morn_v = float(np.sum(rth_v) + (float(pm["Volume"].sum()) if len(pm) > 0 else 0.0))
            trapped_morn_v = float(np.sum(rth_v[rth_close > entry_px]) + (float(pm["Volume"].values[pm["Close"].values > entry_px].sum()) if len(pm) > 0 else 0.0))
            f12 = float(trapped_morn_v / max(tot_morn_v, 1.0))

            pk_i = int(np.argmax(rth_high))
            f13 = float(rth_ofi[pk_i] - rth_ofi[-1])
            f14 = float(rth_pe[-1] - np.mean(rth_pe[:-1])) if len(rth_pe) > 1 else 0.0
            f15 = float(rth_lzc[-1] - np.mean(rth_lzc[:-1])) if len(rth_lzc) > 1 else 0.0

            bar_rets = (rth_close - rth_open) / np.maximum(rth_open, 0.1) * 100.0
            up_m, dn_m = bar_rets > 0.05, bar_rets < -0.05
            up_lam = float(np.mean(np.abs(bar_rets[up_m]) / np.log1p(np.maximum(rth_v[up_m], 1.0)))) if np.any(up_m) else 0.01
            dn_lam = float(np.mean(np.abs(bar_rets[dn_m]) / np.log1p(np.maximum(rth_v[dn_m], 1.0)))) if np.any(dn_m) else 0.01
            f16 = float(np.clip(dn_lam / max(up_lam, 1e-4), 0.1, 10.0))

            deep_map[(tk, d_s)] = [
                f01, f02, f03, f04, f05, f06, f07, f08, f09,
                f10, f11, f12, f13, f14, f15, f16, prev_eod_ofi, prev_eod_obi
            ]
            if has_16_cols:
                t4_map[(tk, d_s)] = [
                    float(rth_c["Odd_Lot_OFI"].fillna(0.0).iloc[-1]),
                    float(rth_c["Block_OFI"].fillna(0.0).iloc[-1]),
                    float(rth_c["Signed_Semivariance"].fillna(0.0).iloc[-1]),
                    float(rth_c["Ask_Step_Down_Rate"].fillna(0.0).iloc[-1]),
                    float(rth_c["Odd_Lot_OFI"].fillna(0.0).mean()),
                    float(rth_c["Block_OFI"].fillna(0.0).mean()),
                    float(rth_c["Signed_Semivariance"].fillna(0.0).mean()),
                    float(rth_c["Ask_Step_Down_Rate"].fillna(0.0).mean()),
                ]
    except Exception as e:
        print(f"  [Warn] {tk}: {e}")
    return tk, recs_tk, inf_tk, deep_map, t4_map, has_16_cols

if new_tickers_to_extract:
    print(f"Extracting {len(new_tickers_to_extract)} newly completed tickers in parallel...")
    records = [r for r in records if r["ticker"] not in set(new_tickers_to_extract)]
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(extract_ticker_all_in_one, tk): tk for tk in new_tickers_to_extract}
        for fut in as_completed(futs):
            tk, recs_tk, inf_tk, d_map, t4_map, has_16 = fut.result()
            records.extend(recs_tk)
            ticker_summary[tk] = inf_tk
            deep18_lookup.update(d_map)
            tick4_lookup.update(t4_map)
            print(f"  + {tk:<6}: {len(recs_tk):>3} positive spikes | {inf_tk['n_adv_tq_days']} Adv T&Q days | 16-Col Schema: {has_16}")

    with open(PKL_PATH, "wb") as f:
        pickle.dump({"records": records, "ticker_summary": sorted(ticker_summary.items())}, f)
    with open(DEEP18_CACHE, "wb") as f:
        pickle.dump(deep18_lookup, f)
    with open(TICK4_CACHE, "wb") as f:
        pickle.dump(tick4_lookup, f)

# Also check if SERV (which was rebuilt with 16 columns) is in tick4_lookup
for tk_16 in ["RGTI", "CSIQ", "TPET", "HOLO", "SGML"]:
    if tk_16 in adv_blobs and not any(k[0] == tk_16 for k in tick4_lookup):
        _, _, _, d_m, t4_m, _ = extract_ticker_all_in_one(tk_16)
        deep18_lookup.update(d_m)
        tick4_lookup.update(t4_m)
with open(TICK4_CACHE, "wb") as f:
    pickle.dump(tick4_lookup, f)

if os.path.exists(POST_SEP8_CACHE):
    with open(POST_SEP8_CACHE, "rb") as f:
        extra_post_sep8_recs = pickle.load(f)
    existing_keys = {(r["ticker"], r["date"]) for r in records}
    for r_ex in extra_post_sep8_recs:
        if (r_ex["ticker"], r_ex["date"]) not in existing_keys:
            records.append(r_ex)

# 2. Update Point-in-Time Unadjusted Daily Open Price Cache for any newly added tickers
if os.path.exists(SPLIT_CACHE):
    with open(SPLIT_CACHE, "rb") as f:
        split_lookup = pickle.load(f)
else:
    split_lookup = {}

all_tickers = sorted({r["ticker"] for r in records})
missing_split_tks = [tk for tk in all_tickers if not any(k[0] == tk for k in split_lookup)]
if missing_split_tks:
    print(f"Fetching exact unadjusted daily open prices for {len(missing_split_tks)} new tickers: {missing_split_tks}...")
    def fetch_ticker_unadj_daily(tk):
        out = {}
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{tk}/range/1/day/2024-06-01/2026-11-09"
            resp = requests.get(url, params={"adjusted": "false", "sort": "asc", "limit": 5000, "apiKey": POLYGON_API_KEY}, timeout=10)
            if resp.status_code == 200:
                for b in resp.json().get("results", []):
                    d_s = pd.Timestamp(b["t"], unit="ms", tz="UTC").tz_convert("America/New_York").strftime("%Y-%m-%d")
                    out[(tk, d_s)] = float(b["o"])
        except Exception:
            pass
        return out

    with ThreadPoolExecutor(max_workers=16) as ex:
        for fut in as_completed([ex.submit(fetch_ticker_unadj_daily, tk) for tk in missing_split_tks]):
            split_lookup.update(fut.result())
    with open(SPLIT_CACHE, "wb") as f:
        pickle.dump(split_lookup, f)

# 3. Load or Recompute Rule-Based Deployed C2 / C2+Cat / C2 Strict <= $25 / C1 on the FULL Expanded T&Q Stock Universe!
with open(FAST_CACHE, "rb") as f:
    fc = pickle.load(f)
all_trading_dates = fc["all_trading_dates"]

avail_tq_tickers = {r["ticker"] for r in records if r.get("has_adv_tq", True)}
if len(fc.get("tr_v2_c2_cat", [])) >= 200:
    tr_v2_c2_base = fc["tr_v2_c2_base"]
    tr_v2_c2_cat  = fc["tr_v2_c2_cat"]
    tr_v2_c2_s25  = fc["tr_v2_c2_s25"]
    tr_v2_c1_base = fc["tr_v2_c1_base"]
    print(f"\nLoaded cached Deployed C2 / C2+Catalyst / C2 Strict trades across all {len(avail_tq_tickers)} T&Q stocks.")
else:
    print(f"\nRecomputing exact Deployed C2 / C2+Catalyst / C2 Strict trades across all {len(avail_tq_tickers)} T&Q stocks...")
    bt_path = "/Users/yaninaso/.gemini/jetski/scratch/bot/agent1/agent1_v2_backtest.py"
    with open(bt_path, "r") as f:
        code = f.read().split("# ═══════════════════════════════════════════════════════════════════════════════\n# RUN")[0]
    ns = {"__file__": bt_path}
    exec(code, ns)

    cache_28m = "/Users/yaninaso/.gemini/jetski/scratch/bot/agent1/.v2_pipeline_cache_28m_20240701_20261102.pkl"
    with open(cache_28m, "rb") as f:
        cs_pool_by_day, all_bars, dates_28m, gd_28m, ref_28m = pickle.load(f)
    ns["_gd_cache"].update(gd_28m)
    ns["_ticker_ref"].update(ref_28m)

    for extra_c in [
        "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/today_20261102_cs30_cache.pkl",
        "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/nov3_to_nov5_cs30_cache.pkl",
        "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/scratch/nov6_cs30_cache.pkl",
    ]:
        if os.path.exists(extra_c):
            with open(extra_c, "rb") as f:
                obj = pickle.load(f)
            if len(obj) == 2:
                pool_x, bars_x = obj
                all_bars.update(bars_x)
                dt_x = "2026-11-02" if "1102" in extra_c else "2026-11-06"
                cs_pool_by_day[dt_x] = pool_x
            elif len(obj) == 4:
                p35, b35, g35, r35 = obj
                all_bars.update(b35); cs_pool_by_day.update(p35); ns["_gd_cache"].update(g35); ns["_ticker_ref"].update(r35)

    all_dates = sorted([d for d in cs_pool_by_day.keys() if "2024-07-01" <= d <= "2026-11-06"])
    for dt in all_dates:
        prior_str = ns["_prior_td"](dt).strftime("%Y-%m-%d")
        p_gd = ns["_gd_cache"].get(prior_str, {})
        for c in cs_pool_by_day.get(dt, []):
            tk = c["ticker"]
            if "prior_dvol" not in c or c["prior_dvol"] is None:
                pv = p_gd.get(tk, {}).get("v", 0)
                c["prior_dvol"] = pv * c.get("prior_close", 0)
            if not c.get("name"):
                c["name"] = (ns["_ticker_ref"].get(tk, {}) or {}).get("name", "")
            c["is_spac"] = any(w in (c["name"] or "").upper() for w in ("ACQUISITION", "BLANK CHECK", "SPAC"))

    with open(CAT_MEMO_CACHE, "rb") as f:
        _cat_memo = pickle.load(f)

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

    tq_pool_base = {dt: [c for c in pool if c["ticker"] in avail_tq_tickers] for dt, pool in cs_pool_by_day.items()}
    tq_pool_cat  = {
        dt: [c for c in pool if c["ticker"] in avail_tq_tickers and not is_toxic_cat(c["ticker"], dt, c.get("gap_pct", 0)*100.0, c.get("open_price", 0), c.get("prior_dvol", 0))]
        for dt, pool in cs_pool_by_day.items()
    }
    tq_pool_s25  = {
        dt: [c for c in pool if (c.get("gap_pct", 0) >= 0.30 or (c.get("open_price", 0) <= 25.0 and c.get("prior_dvol", 0) < 25e6))]
        for dt, pool in tq_pool_cat.items()
    }

    c2_cfg = next(s for s in ns["STRATEGIES"] if s["name"].startswith("Combo C2"))
    c1_cfg = next(s for s in ns["STRATEGIES"] if s["name"].startswith("Combo C1"))

    tr_v2_c2_base = ns["build_trades_v2"](tq_pool_base, all_bars, all_dates, c2_cfg["id"], 3, strat_cfg=c2_cfg)
    tr_v2_c2_cat  = ns["build_trades_v2"](tq_pool_cat,  all_bars, all_dates, c2_cfg["id"], 3, strat_cfg=c2_cfg)
    tr_v2_c2_s25  = ns["build_trades_v2"](tq_pool_s25,  all_bars, all_dates, c2_cfg["id"], 3, strat_cfg=c2_cfg)
    tr_v2_c1_base = ns["build_trades_v2"](tq_pool_base, all_bars, all_dates, c1_cfg["id"], 3, strat_cfg=c1_cfg)

    fc["n_tq_tickers"] = len(avail_tq_tickers)
    fc["tr_v2_c2_base"] = tr_v2_c2_base
    fc["tr_v2_c2_cat"]  = tr_v2_c2_cat
    fc["tr_v2_c2_s25"]  = tr_v2_c2_s25
    fc["tr_v2_c1_base"] = tr_v2_c1_base
    with open(FAST_CACHE, "wb") as f:
        pickle.dump(fc, f)

# 4. Prepare Features & Print Summary of Newly Added Tickers
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

    # Restore true unadjusted percentage spread
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

    # 6 Pure Orthogonal T&Q / Overhead-Supply Features (for pure T&Q tree/ensemble models M7/M11/M12)
    # + 2 Cross-Modal Climax/Illiquidity Features (for linear/logistic Core8+T&Q model M4)
    r["f_verified8_new"] = [
        float(f_deep[7]),                                                       # 0. deep08_pm_trapped_vol_share
        float(np.clip(f_deep[4], 0.0, 15.0)),                                   # 1. deep05_quote_int_vs_20d_norm
        float(f_deep[11]),                                                      # 2. deep12_total_overhead_trapped_vol
        float(-lr_simb * (1.0 - lr_buy)),                                       # 3. new02_inst_sell_block_pressure
        float((2.0 * lr_buy - 1.0) - lr_ofi),                                   # 4. new01_retail_vs_inst_ofi_div
        float(lr_ofi - body_rat),                                               # 5. new05_passive_ask_absorption
        float(bar_rng / max(np.log1p(cum_rv * max(vol_climax, 0.2)), 0.2)),     # 6. new07_kyle_lambda_bar
        float(np.clip(vol_climax, 0.0, 10.0) * pb_peak * (1.0 - lr_ofi)),       # 7. new09_climax_reversal_torque
    ]
    b_up = list(b_c)
    a_up = list(a_c)
    if abs(f_deep[10]) > 1e-6:
        b_up[15] = float(f_deep[10])
    if abs(f_deep[9]) > 1e-6:
        a_up[16] = float(f_deep[9])
    r["f_tq33_upgraded_pm"] = b_up + a_up

# Print Breakdown of the Newly Downloaded Stocks (#40 -> #50 + any GCP stocks)
RECENT_BATCH = ["RGTI", "USGO", "WIMI", "TENX", "PPCB", "STAI", "CSIQ", "QUBT", "TPET", "HOLO", "SGML", "AMIX", "CAPS", "MOVE", "VRAX", "WETO"]
print("\n" + "=" * 145)
print(f"2. PROFILE OF NEWLY DOWNLOADED STOCKS ({len(RECENT_BATCH)} Tickers: Ranks #40-#50 + Overnight GCP Completions)")
print("=" * 145)
print(f"| {'Ticker':<6} | {'Total Spikes':<12} | {'RVol Setups':<11} | {'Max Split Fac':<13} | {'Split-Adj Med Open':<18} | {'TRUE Unadj Med Open':<19} | {'Setups >= $5 Unadj':<18} | {'Setups >= $10 Unadj':<19} | {'16-Col Tick Schema':<18} |")
print(f"|{'-'*8}|{'-'*14}|{'-'*13}|{'-'*15}|{'-'*20}|{'-'*21}|{'-'*20}|{'-'*21}|{'-'*20}|")
for tk in RECENT_BATCH:
    all_tk = [r for r in records if r["ticker"] == tk and r["has_adv_tq"]]
    rv_tk = [r for r in all_tk if r["is_rvol_verified_causal"]]
    if not rv_tk:
        print(f"| {tk:<6} | {len(all_tk):>12d} | {0:>11d} | {'--':>13} | {'--':>18} | {'--':>19} | {0:>18d} | {0:>19d} | {'No':>18} |")
        continue
    msf = max(r["split_factor"] for r in rv_tk)
    m_adj = float(np.median([r["open_px"] for r in rv_tk]))
    m_unadj = float(np.median([r["unadj_open_px"] for r in rv_tk]))
    n_ge5 = sum(1 for r in rv_tk if 5.0 <= r["unadj_open_px"] <= 150.0 and r["split_factor"] <= 1.25)
    n_ge10 = sum(1 for r in rv_tk if 10.0 <= r["unadj_open_px"] <= 100.0 and r["split_factor"] <= 1.25)
    has16 = "YES (16 cols)" if any(k[0] == tk for k in tick4_lookup) else "12 cols (GCP)"
    print(f"| {tk:<6} | {len(all_tk):>12d} | {len(rv_tk):>11d} | {msf:>12.1f}x | ${m_adj:>16.2f} | ${m_unadj:>17.2f} | {n_ge5:>18d} | {n_ge10:>19d} | {has16:>18} |")

# 5. Evaluate the 4 New Sub-Minute Tick-Level Columns on the 16-Column Subset (SERV, RGTI, CSIQ, QUBT, TPET, HOLO, SGML)
tick4_recs = [r for r in records if r["is_rvol_verified_causal"] and (r["ticker"], r["date"]) in tick4_lookup and r["open_px"] <= 300.0]
if len(tick4_recs) >= 30:
    print("\n" + "=" * 145)
    print(f"3. EARLY SIGNAL AUDIT OF THE 4 NEW SUB-MINUTE TICK COLUMNS ACROSS {len(tick4_recs)} RVOL SETUPS ({sorted({r['ticker'] for r in tick4_recs})})")
    print("=" * 145)
    y_t4 = np.array([r["raw_pnl_pct"] for r in tick4_recs], dtype=float)
    X_t4 = np.array([tick4_lookup[(r["ticker"], r["date"])] for r in tick4_recs], dtype=float)
    T4_NAMES = [
        "1. Odd_Lot_OFI (Trigger Bar)", "2. Block_OFI (Trigger Bar)", "3. Signed_Semivariance (Trigger Bar)", "4. Ask_Step_Down_Rate (Trigger Bar)",
        "5. Odd_Lot_OFI (Morning Mean)", "6. Block_OFI (Morning Mean)", "7. Signed_Semivariance (Morning Mean)", "8. Ask_Step_Down_Rate (Morning Mean)",
    ]
    div_trig = X_t4[:, 0] - X_t4[:, 1]
    div_morn = X_t4[:, 4] - X_t4[:, 5]
    X_t4_ext = np.column_stack([X_t4, div_trig, div_morn])
    T4_NAMES.extend(["9. Odd_Lot_minus_Block_OFI (Trigger)", "10. Odd_Lot_minus_Block_OFI (Morning)"])
    for j, tname in enumerate(T4_NAMES):
        col = np.nan_to_num(X_t4_ext[:, j], 0.0)
        if np.std(col) < 1e-8:
            continue
        sr, sp = sp_stats.spearmanr(col, y_t4)
        q33, q67 = np.percentile(col, [33.3, 66.7])
        q1a, q3a = y_t4[col <= q33].mean(), y_t4[col >= q67].mean()
        q1w, q3w = (y_t4[col <= q33] > 0).mean() * 100.0, (y_t4[col >= q67] > 0).mean() * 100.0
        print(f"  {tname:<38} | Spearman r = {sr:+6.3f} (p={sp:.3f}) | Q1: {q1a:+5.2f}% ({q1w:4.1f}% WR) -> Q3: {q3a:+5.2f}% ({q3w:4.1f}% WR) | Q3-Q1 = {q3a-q1a:+5.2f}%")

# 6. Run Walk-Forward OOS Evaluation on the Full Expanded T&Q Universe
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

        # Upgraded T&Q:
        # 1) Keep exact 33-col dimensionality for tree/MLP models (by replacing the 3 lowest-IC in-fold T&Q cols with the Top-3 Pure Orthogonal T&Q features [0..5]) so random-subspace split geometry is preserved!
        corrs_up = [abs(float(np.nan_to_num(sp_stats.spearmanr(X_up33[tr_m, j], y_adv[tr_m])[0]))) if np.std(X_up33[tr_m, j]) > 1e-8 else 0.0 for j in range(33)]
        t12_up = np.argsort(corrs_up)[-12:]
        keep30_up = np.sort(np.argsort(corrs_up)[-30:])
        corrs_pure6 = [abs(float(np.nan_to_num(sp_stats.spearmanr(X_new8[tr_m, j], y_adv[tr_m])[0]))) if np.std(X_new8[tr_m, j]) > 1e-8 else 0.0 for j in range(6)]
        t3_pure = np.argsort(corrs_pure6)[-3:]
        corrs_n8 = [abs(float(np.nan_to_num(sp_stats.spearmanr(X_new8[tr_m, j], y_adv[tr_m])[0]))) if np.std(X_new8[tr_m, j]) > 1e-8 else 0.0 for j in range(8)]
        t4_n8 = np.argsort(corrs_n8)[-4:]

        X_c24_new_tr = np.hstack([X_core8[tr_m], X_up33[tr_m][:, t12_up], X_new8[tr_m][:, t4_n8]])
        X_c24_new_te = np.hstack([X_core8[te_m], X_up33[te_m][:, t12_up], X_new8[te_m][:, t4_n8]])
        X_tq33_new_tr = np.hstack([X_up33[tr_m][:, keep30_up], X_new8[tr_m][:, t3_pure]])
        X_tq33_new_te = np.hstack([X_up33[te_m][:, keep30_up], X_new8[te_m][:, t3_pure]])

        _, p_m1_new = fit_m1_ridge(X_tq33_new_tr, y_adv[tr_m], X_tq33_new_te)
        _, p_m3_new = fit_m3_poly_ridge(X_tq33_new_tr, y_adv[tr_m], X_tq33_new_te)
        _, p_m6_new = fit_m6_gbdt_d2(X_tq33_new_tr, y_adv[tr_m], X_tq33_new_te)
        _, p_m7_new = fit_m7_gbdt_d3(X_tq33_new_tr, y_adv[tr_m], X_tq33_new_te)
        _, p_m8_new = fit_m8_rf_d2(X_tq33_new_tr, y_adv[tr_m], X_tq33_new_te)
        _, p_m10_new = fit_m10_mlp(X_tq33_new_tr, y_adv[tr_m], X_tq33_new_te)
        p_m11_new = 0.25 * (p_m3_new + p_m6_new + p_m8_new + p_m10_new)
        _, p_m4_new = fit_m4_logistic(X_c24_new_tr, y_adv[tr_m], X_c24_new_te)
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

pool_clean_5  = [r for r in records if 5.0 <= r["unadj_open_px"] <= 150.0 and r["split_factor"] <= 1.25]
pool_clean_10 = [r for r in records if 10.0 <= r["unadj_open_px"] <= 100.0 and r["split_factor"] <= 1.25]

adv_5,  oos_ym_5,  preds_5  = run_walk_forward_for_pool(pool_clean_5,  "POOL A: POINT-IN-TIME UNADJ PRICE >= $5.00 (EXPANDED T&Q UNIVERSE)")
adv_10, oos_ym_10, preds_10 = run_walk_forward_for_pool(pool_clean_10, "POOL B: POINT-IN-TIME UNADJ PRICE $10-$100 (EXACT C2 PRICE RANGE, EXPANDED T&Q)")

# Build lookup of split_factor by (tk, d) for exact C2 unadjusted filtering
sf_by_tk_d = {(r["ticker"], r["date"]): r["split_factor"] for r in records}

# 7. Build Multi-Horizon Portfolio Comparison & Plot Clean Compounded Curves
def v2_trades_to_day_map(tr_list, min_unadj_px=10.0, max_unadj_px=100.0, check_split=True):
    out = defaultdict(list)
    for idx, t in enumerate(tr_list):
        d = str(t.get("scan_date") or t.get("date"))
        tk = t["ticker"].split("_")[0]
        unadj_o = float(split_lookup.get((tk, d), 15.0))
        if not (min_unadj_px <= unadj_o <= max_unadj_px):
            continue
        if check_split and sf_by_tk_d.get((tk, d), 1.0) > 1.25:
            continue
        pnl_pct = max(float(t.get("pnl_eod", 0.0)), -10.5)
        t_et = str(t.get("entry_time", "09:33"))
        hm = (int(t_et[:2]) - 3) * 60 + int(t_et[3:5]) if ":" in t_et else 393
        out[d].append((1000.0 - idx, {"ticker": t["ticker"], "date": d, "entry_hm": hm, "raw_pnl_pct": pnl_pct, "ret_gap_adaptive": pnl_pct}))
    return out

s_v2_c2_cat_raw = v2_trades_to_day_map(tr_v2_c2_cat, min_unadj_px=5.0,  max_unadj_px=150.0, check_split=True)
s_v2_c2_cat     = v2_trades_to_day_map(tr_v2_c2_cat, min_unadj_px=10.0, max_unadj_px=100.0, check_split=True)
s_v2_c2_s25     = v2_trades_to_day_map(tr_v2_c2_s25, min_unadj_px=10.0, max_unadj_px=25.0,  check_split=True)

def make_ml_map(pool_list, preds_dict, oos_ym, m_key, thresh=0.35):
    out = defaultdict(list)
    for r in pool_list:
        if r["ym"] < oos_ym or r["gap_pct"] < 3.0:
            continue
        sc = preds_dict.get((r["ticker"], r["date"]), {}).get(m_key, -999.0)
        if sc >= thresh:
            out[r["date"]].append((sc, r))
    return out

s_m7_5_old   = make_ml_map(adv_5,  preds_5,  oos_ym_5,  "M7_OLD",  0.35)
s_m7_5_new   = make_ml_map(adv_5,  preds_5,  oos_ym_5,  "M7_NEW",  0.35)
s_m12_5_old  = make_ml_map(adv_5,  preds_5,  oos_ym_5,  "M12_OLD", 0.35)
s_m12_5_new  = make_ml_map(adv_5,  preds_5,  oos_ym_5,  "M12_NEW", 0.35)

s_m7_10_old  = make_ml_map(adv_10, preds_10, oos_ym_10, "M7_OLD",  0.35)
s_m7_10_new  = make_ml_map(adv_10, preds_10, oos_ym_10, "M7_NEW",  0.35)
s_m12_10_old = make_ml_map(adv_10, preds_10, oos_ym_10, "M12_OLD", 0.35)
s_m12_10_new = make_ml_map(adv_10, preds_10, oos_ym_10, "M12_NEW", 0.35)

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
    ("0. Deployed C2 + Catalyst (Unfiltered Split-Adj)",    s_v2_c2_cat_raw,"raw_pnl_pct",      3),
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
    print(f"  {h_title} ({len(h_dates)} Trading Days) — EXPANDED {len(avail_tq_tickers)}-STOCK T&Q UNIVERSE")
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

plt.suptitle(f"Expanded T&Q Universe ({len(avail_tq_tickers)} Stocks, Clean Unadjusted Price): Old 33 T&Q vs. Upgraded (+8 Orthogonal Features) & Deployed C2", fontsize=12.2, fontweight="bold", y=0.995)
plt.tight_layout()
out_chart = "/Users/yaninaso/.gemini/jetski/brain/0f9a25b3-6507-46a8-8b90-15067a1734a1/tq_152_expanded_clean_upgraded_backtest.png"
plt.savefig(out_chart, dpi=140)
print(f"\nSaved expanded T&Q universe chart to: {out_chart}")
