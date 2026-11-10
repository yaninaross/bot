# %% ── Standalone GCP Parallel T&Q Downloader + Single-Pass Lee-Ready Builder ──
#
# Designed to run in parallel with the local Mac downloader without collisions:
#   - Auto-fetches the prioritized ticker manifest from gs://time-series-bucket-ys/Data/Config/priority_missing_tq_tickers.json
#   - Starts at Rank #16 -> #150 (while the Mac works on Ranks #1 -> #15), then refreshes
#     the 26 existing tickers (Sep 9 - Nov 9, 2026), then sweeps Ranks #15 -> #1 in reverse.
#   - Uses GCS distributed lock blobs (Data/Locks/{TICKER}.lock) so two workers never process
#     the same ticker simultaneously.
#   - SINGLE-PASS IN-MEMORY ENGINE: Computes 1-min Base Microstructure + 12 Lee-Ready Advanced
#     Microstructure features in RAM as each day's T&Q downloads from Polygon, eliminating
#     the second GCS read-back pass!
#
# Works directly inside a GCP Notebook cell OR from a GCP VM terminal:
#   python3 -u gcp_parallel_tq_worker.py

import os
import io
import gc
import sys
import json
import time
import ctypes
import warnings
import requests
import pytz
import numpy as np
import pandas as pd
from math import log2, factorial
from collections import Counter
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

try:
    from polygon import RESTClient
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "polygon-api-client", "google-cloud-storage", "pyarrow", "--quiet"])
    from polygon import RESTClient

from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except Exception:
    pass

# ==============================================================================
# CONFIGURATION
# ==============================================================================
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
BUCKET_NAME     = os.environ.get("GCS_BUCKET", "time-series-bucket-ys")
PROJECT_ID      = os.environ.get("GCP_PROJECT", "ys-time-series")

GCS_RAW_DIR       = "Data/TradesQuotes"
GCS_OHLCV_DIR     = "Data/Inference"
GCS_MICRO_DIR     = "Data/Inference/Microstructure"
GCS_ADV_MICRO_DIR = "Data/Inference/Microstructure_Advanced"
GCS_MASTER_DIR    = "Data/Inference_v2"
GCS_LOCK_DIR      = "Data/Locks"

HISTORY_START      = "2024-06-01"
PE_ORDER           = 5
MIN_TRADES_PER_BAR = 6
POLYGON_SLEEP      = 0.02

# Default concurrency for an 8-16 vCPU, 32GB RAM GCP instance
DAY_WORKERS = int(os.environ.get("TQ_WORKERS", "16"))
START_RANK  = int(os.environ.get("START_RANK", "16"))
END_RANK    = int(os.environ.get("END_RANK", "150"))

ADVANCED_MICRO_COLS = [
    "LR_OFI", "LR_VPIN", "LR_Buy_Ratio", "LR_Size_Imbalance",
    "PE", "LZC", "Tick_Volatility", "Shock_Persist",
    "Trade_Size_CV", "Trade_Intensity", "Quote_Intensity", "Spread_CV",
    "Odd_Lot_OFI", "Block_OFI", "Signed_Semivariance", "Ask_Step_Down_Rate",
]

def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

LOCAL_SA_KEY = "/Users/yaninaso/Downloads/ys-time-series-sa-key.json"

def _init_gcs_client():
    if os.path.exists(LOCAL_SA_KEY):
        return gcs_storage.Client.from_service_account_json(LOCAL_SA_KEY, project=PROJECT_ID)
    if "--sa" in sys.argv or os.environ.get("USE_SERVICE_ACCOUNT", "") == "1":
        from google.auth import compute_engine
        creds = compute_engine.Credentials()
        return gcs_storage.Client(credentials=creds, project=PROJECT_ID)
    return gcs_storage.Client(project=PROJECT_ID)

gcs_client = _init_gcs_client()
bucket = gcs_client.bucket(BUCKET_NAME)
_last_refresh = time.time()

def _refresh_bucket():
    global gcs_client, bucket, _last_refresh
    if time.time() - _last_refresh > 1800:
        gcs_client = _init_gcs_client()
        bucket = gcs_client.bucket(BUCKET_NAME)
        _last_refresh = time.time()

# ==============================================================================
# DISTRIBUTED GCS LOCK / ACTIVE-CHECK (PREVENTS COLLISION WITH MAC WORKER)
# ==============================================================================
def try_claim_ticker(ticker):
    """Returns True if we can claim this ticker, or False if already complete or locked by another worker."""
    adv_blob = bucket.blob(f"{GCS_ADV_MICRO_DIR}/{ticker}_1m_advanced_micro.parquet")
    master_blob = bucket.blob(f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet")
    if adv_blob.exists() and master_blob.exists():
        adv_blob.reload()
        if adv_blob.updated and (datetime.now(pytz.UTC) - adv_blob.updated).total_seconds() < 86400 * 5:
            return False

    lock_blob = bucket.blob(f"{GCS_LOCK_DIR}/{ticker}.lock")
    if lock_blob.exists():
        lock_blob.reload()
        if lock_blob.updated and (datetime.now(pytz.UTC) - lock_blob.updated).total_seconds() < 2700:
            return False

    # Also check if another machine uploaded a raw T&Q file for this ticker in the last 5 minutes
    recent_blobs = list(bucket.list_blobs(prefix=f"{GCS_RAW_DIR}/{ticker}/", max_results=10))
    for b in recent_blobs:
        if b.updated and (datetime.now(pytz.UTC) - b.updated).total_seconds() < 300:
            return False

    lock_blob.upload_from_string(f"claimed_by_gcp_{datetime.now(pytz.UTC).isoformat()}")
    return True

def release_ticker_lock(ticker):
    try:
        lock_blob = bucket.blob(f"{GCS_LOCK_DIR}/{ticker}.lock")
        if lock_blob.exists():
            lock_blob.delete()
    except Exception:
        pass

# ==============================================================================
# 1-MINUTE OHLCV DOWNLOADER
# ==============================================================================
def ensure_ohlcv_parquet(ticker, start_date=HISTORY_START):
    ohlcv_path = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
    blob = bucket.blob(ohlcv_path)
    if blob.exists():
        blob.reload()
        if blob.updated and (datetime.now(pytz.UTC) - blob.updated).total_seconds() < 86400 * 3:
            return True

    end_date = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )
    all_frames = []
    while url:
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
        else:
            break

        if "results" in data and data.get("resultsCount", 0) > 0:
            all_frames.append(pd.DataFrame(data["results"]))
            url = data["next_url"] + f"&apiKey={POLYGON_API_KEY}" if "next_url" in data else None
        else:
            url = None

    if not all_frames:
        print(f"  [{ticker:6s}] WARNING: No 1m OHLCV returned from Polygon", flush=True)
        return False

    df = pd.concat(all_frames, ignore_index=True).rename(columns={
        "t": "Timestamp", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
    })
    df["Timestamp"] = df["Timestamp"] / 1000.0
    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    df = df.set_index("Datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")][["Timestamp", "Open", "High", "Low", "Close", "Volume"]]

    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow")
    buf.seek(0)
    blob.upload_from_file(buf)
    print(f"  [{ticker:6s}] Saved 1m OHLCV: {len(df):,} rows ({df.index.min().strftime('%Y-%m-%d')} -> {df.index.max().strftime('%Y-%m-%d')})", flush=True)
    del df, all_frames
    _release_memory()
    return True

# ==============================================================================
# SINGLE-PASS IN-MEMORY MICROSTRUCTURE COMPUTATION + T&Q DOWNLOAD
# ==============================================================================
def normalize_raw_schema(df, kind):
    if df is None or df.empty:
        return pd.DataFrame()
    for ts_col in ("ns", "sip_timestamp", "participant_timestamp"):
        if ts_col in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={ts_col: "timestamp"})
            break
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    col = df["timestamp"]
    if pd.api.types.is_integer_dtype(col):
        df["timestamp"] = pd.to_datetime(col, unit="ns", utc=True)
    elif pd.api.types.is_datetime64_any_dtype(col):
        if col.dt.tz is None:
            df["timestamp"] = col.dt.tz_localize("UTC")
    else:
        df["timestamp"] = pd.to_datetime(col, utc=True)
    if kind == "quotes":
        df = df.rename(columns={"bid": "bid_price", "ask": "ask_price", "bsize": "bid_size", "asize": "ask_size"})
    return df

def compute_micro_from_dfs(dt, dq):
    if dt is None or dq is None or dt.empty or dq.empty:
        return None, None
    try:
        # 1. Base Microstructure (1-min)
        dt_idx = dt.set_index("timestamp").sort_index()
        direction = np.sign(dt_idx["price"].diff())
        direction = direction.mask(direction == 0).ffill().fillna(0)
        dt_idx["Volume_Delta"] = dt_idx["size"] * direction
        t_1m = dt_idx.resample("1min").agg(
            {"size": "sum", "Volume_Delta": "sum", "price": "count"}
        ).rename(columns={"size": "Total_Vol", "price": "Trade_Count"}).fillna(0)

        dq_idx = dq.set_index("timestamp").sort_index()
        dq_idx["Spread"] = dq_idx["ask_price"] - dq_idx["bid_price"]
        dq_idx["OBI"] = (dq_idx["bid_size"] - dq_idx["ask_size"]) / (dq_idx["bid_size"] + dq_idx["ask_size"] + 1e-9)
        q_1m = dq_idx.resample("1min").agg(
            {"Spread": "mean", "OBI": "mean", "bid_size": "sum", "ask_size": "sum"}
        ).rename(columns={"bid_size": "bsize", "ask_size": "asize"}).ffill()

        base_df = pd.concat([q_1m, t_1m], axis=1).ffill().fillna(0)
        base_df["Delta_Pct"] = base_df["Volume_Delta"] / (base_df["Total_Vol"] + 1e-9)

        # 2. Advanced Lee-Ready Microstructure (1-min)
        trades_s = dt.sort_values("timestamp").copy()
        quotes_s = dq[["timestamp", "bid_price", "ask_price"]].sort_values("timestamp").copy()
        signed = pd.merge_asof(trades_s, quotes_s, on="timestamp", direction="backward")
        mid = (signed["bid_price"] + signed["ask_price"]) / 2.0
        signed["sign"] = np.where(signed["price"] > mid, 1.0, np.where(signed["price"] < mid, -1.0, 0.0))
        z_m = signed["sign"] == 0.0
        if z_m.any():
            tsign = np.sign(signed["price"].diff())
            signed.loc[z_m, "sign"] = tsign[z_m]
        signed["sign"] = signed["sign"].replace(0.0, np.nan).ffill().fillna(1.0)
        signed["tick_return"] = signed["price"].pct_change().fillna(0.0)
        signed["bar"] = signed["timestamp"].dt.floor("1min")
        dq_bar = dq.assign(bar=dq["timestamp"].dt.floor("1min"))
        q_grouped = {ts: grp for ts, grp in dq_bar.groupby("bar")}

        adv_rows = []
        max_ent = log2(factorial(PE_ORDER))
        for bar_ts, t_grp in signed.groupby("bar"):
            q_grp = q_grouped.get(bar_ts)
            n = len(t_grp)
            if n < MIN_TRADES_PER_BAR:
                row = {c: np.nan for c in ADVANCED_MICRO_COLS}
            else:
                signs = t_grp["sign"].values
                sizes = t_grp["size"].values.astype(np.float64)
                trets = t_grp["tick_return"].values
                bm, sm = signs > 0, signs < 0
                nb, ns_c = bm.sum(), sm.sum()
                bvol, svol = sizes[bm].sum(), sizes[sm].sum()
                tvol = bvol + svol
                lr_ofi = (bvol - svol) / (tvol + 1e-9)
                lr_vpin = abs(bvol - svol) / (tvol + 1e-9)
                lr_buy = nb / (nb + ns_c + 1e-9)
                abs_b = sizes[bm].mean() if nb > 0 else 0.0
                abs_s = sizes[sm].mean() if ns_c > 0 else 0.0
                lr_simb = (abs_b - abs_s) / (abs_b + abs_s + 1e-9)

                if n >= PE_ORDER + 1:
                    pats = [tuple(np.argsort(np.argsort(signs[i:i + PE_ORDER]))) for i in range(n - PE_ORDER + 1)]
                    cnts = Counter(pats)
                    probs = np.array([c / len(pats) for c in cnts.values()])
                    pe = float(-np.sum(probs * np.log2(probs + 1e-12)) / max_ent)
                else:
                    pe = np.nan

                s_bin = "".join(str(int(b)) for b in (signs > 0))
                subs, c_lz, st_i = set(), 0, 0
                for i_c in range(n):
                    sub = s_bin[st_i:i_c + 1]
                    if sub not in subs:
                        subs.add(sub); c_lz += 1; st_i = i_c + 1
                if st_i < n:
                    c_lz += 1
                lzc = float(c_lz / (n / log2(n))) if n > 1 else np.nan

                tvolat = float(np.std(trets[1:])) if n > 2 else 0.0
                spers = 0.0
                if n > 3:
                    r_sub = trets[1:]
                    if len(r_sub) > 2 and np.std(r_sub) > 1e-10:
                        x, y = r_sub[:-1], r_sub[1:]
                        sx, sy = x.std(), y.std()
                        if sx > 1e-10 and sy > 1e-10:
                            spers = float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))

                tscv = float(np.std(sizes) / (np.mean(sizes) + 1e-9))
                ts_rng = (t_grp["timestamp"].max() - t_grp["timestamp"].min()).total_seconds()
                tint = float(n / (ts_rng + 1e-9)) if ts_rng > 0 else float(n)
                qint = float(len(q_grp)) if q_grp is not None else 0.0
                spcv = 0.0
                ask_step_dn_rate = 0.0
                if q_grp is not None and len(q_grp) > 1:
                    ask_p = q_grp["ask_price"].values.astype(np.float64)
                    bid_p = q_grp["bid_price"].values.astype(np.float64)
                    sps = ask_p - bid_p
                    if np.mean(sps) > 1e-8:
                        spcv = float(np.std(sps) / np.mean(sps))
                    if len(q_grp) > 2:
                        da = np.diff(ask_p)
                        db = np.diff(bid_p)
                        n_adn = float(np.sum(da < -1e-6))
                        n_bup = float(np.sum(db > 1e-6))
                        if (n_adn + n_bup) > 0:
                            ask_step_dn_rate = float((n_adn - n_bup) / (n_adn + n_bup))

                # 1. Odd_Lot_OFI (retail prints < 100 shares) & 2. Block_OFI (institutional prints >= 500 shares or >= $25K)
                prices_bar = t_grp["price"].values.astype(np.float64)
                odd_m = sizes < 100.0
                if odd_m.any():
                    obv, osv = sizes[odd_m & bm].sum(), sizes[odd_m & sm].sum()
                    odd_lot_ofi = float((obv - osv) / (obv + osv + 1e-9))
                else:
                    odd_lot_ofi = 0.0

                blk_m = (sizes >= 500.0) | ((sizes * prices_bar) >= 25000.0)
                if blk_m.any():
                    bbv, bsv = sizes[blk_m & bm].sum(), sizes[blk_m & sm].sum()
                    block_ofi = float((bbv - bsv) / (bbv + bsv + 1e-9))
                else:
                    block_ofi = 0.0

                # 3. Signed_Semivariance ((RV_up - RV_dn) / (RV_up + RV_dn))
                signed_semivar = 0.0
                if n > 2:
                    rt_sub = trets[1:]
                    rv_u = float(np.sum(rt_sub[rt_sub > 0.0] ** 2))
                    rv_d = float(np.sum(rt_sub[rt_sub < 0.0] ** 2))
                    if (rv_u + rv_d) > 1e-12:
                        signed_semivar = float((rv_u - rv_d) / (rv_u + rv_d))

                row = {
                    "LR_OFI": lr_ofi, "LR_VPIN": lr_vpin, "LR_Buy_Ratio": lr_buy,
                    "LR_Size_Imbalance": lr_simb, "PE": pe, "LZC": lzc,
                    "Tick_Volatility": tvolat, "Shock_Persist": spers,
                    "Trade_Size_CV": tscv, "Trade_Intensity": tint,
                    "Quote_Intensity": qint, "Spread_CV": spcv,
                    "Odd_Lot_OFI": odd_lot_ofi, "Block_OFI": block_ofi,
                    "Signed_Semivariance": signed_semivar, "Ask_Step_Down_Rate": ask_step_dn_rate,
                }
            row["timestamp"] = bar_ts
            adv_rows.append(row)

        adv_df = pd.DataFrame(adv_rows).set_index("timestamp").sort_index() if adv_rows else None
        return base_df, adv_df
    except Exception:
        return None, None

def scan_existing_tq_dates(ticker):
    prefix = f"{GCS_RAW_DIR}/{ticker}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    t_dates, q_dates = set(), set()
    for b in blobs:
        fname = os.path.basename(b.name)
        if fname.startswith(f"{ticker}_trades_") and fname.endswith(".parquet"):
            t_dates.add(fname.replace(f"{ticker}_trades_", "").replace(".parquet", ""))
        elif fname.startswith(f"{ticker}_quotes_") and fname.endswith(".parquet"):
            q_dates.add(fname.replace(f"{ticker}_quotes_", "").replace(".parquet", ""))
    return t_dates & q_dates

def process_one_day_single_pass(ticker, date_str, is_cached):
    """If cached on GCS, reads from GCS; otherwise downloads from Polygon and saves to GCS.
    In both cases, computes 1-min Base + Lee-Ready Adv Microstructure in RAM in a single pass!"""
    t_path = f"{GCS_RAW_DIR}/{ticker}/{ticker}_trades_{date_str}.parquet"
    q_path = f"{GCS_RAW_DIR}/{ticker}/{ticker}_quotes_{date_str}.parquet"
    try:
        if is_cached:
            t_blob = bucket.blob(t_path)
            q_blob = bucket.blob(q_path)
            if not t_blob.exists() or not q_blob.exists():
                return "empty", None, None
            dt = normalize_raw_schema(pd.read_parquet(io.BytesIO(t_blob.download_as_bytes())), "trades")
            dq = normalize_raw_schema(pd.read_parquet(io.BytesIO(q_blob.download_as_bytes())), "quotes")
            if dt.empty or dq.empty:
                return "empty", None, None
            b_df, a_df = compute_micro_from_dfs(dt, dq)
            return "cached", b_df, a_df
        else:
            poly = RESTClient(POLYGON_API_KEY)
            day_start = pd.Timestamp(date_str, tz="UTC")
            day_end = day_start + timedelta(days=1)
            trades = [
                {"timestamp": pd.to_datetime(t.sip_timestamp, unit="ns", utc=True), "price": t.price, "size": t.size}
                for t in poly.list_trades(ticker, timestamp_gte=day_start, timestamp_lte=day_end, limit=50000)
            ]
            quotes = [
                {"timestamp": pd.to_datetime(q.sip_timestamp, unit="ns", utc=True),
                 "ask_price": q.ask_price, "ask_size": q.ask_size,
                 "bid_price": q.bid_price, "bid_size": q.bid_size}
                for q in poly.list_quotes(ticker, timestamp_gte=day_start, timestamp_lte=day_end, limit=50000)
            ]
            if POLYGON_SLEEP > 0:
                time.sleep(POLYGON_SLEEP)
            dt = pd.DataFrame(trades) if trades else pd.DataFrame()
            dq = pd.DataFrame(quotes) if quotes else pd.DataFrame()
            if dt.empty and dq.empty:
                return "empty", None, None
            if not dt.empty:
                buf = io.BytesIO()
                dt.to_parquet(buf, engine="pyarrow")
                buf.seek(0)
                bucket.blob(t_path).upload_from_file(buf)
            if not dq.empty:
                buf = io.BytesIO()
                dq.to_parquet(buf, engine="pyarrow")
                buf.seek(0)
                bucket.blob(q_path).upload_from_file(buf)
            b_df, a_df = compute_micro_from_dfs(dt, dq)
            return "ok", b_df, a_df
    except Exception as e:
        return f"err:{e}", None, None

def run_full_ticker_single_pass(ticker, target_dates):
    existing = scan_existing_tq_dates(ticker)
    print(f"  [{ticker:6s}] Processing {len(target_dates)} days ({len(existing)} cached, {len(set(target_dates) - existing)} to download) with {DAY_WORKERS} workers...", flush=True)
    dl_ok, cached_ok, empty_cnt, err_cnt = 0, 0, 0, 0
    done_cnt = 0
    base_chunks, adv_chunks = [], []

    with ThreadPoolExecutor(max_workers=DAY_WORKERS) as pool:
        futs = {pool.submit(process_one_day_single_pass, ticker, d, d in existing): d for d in target_dates}
        for fut in as_completed(futs):
            status, b_df, a_df = fut.result()
            done_cnt += 1
            if status == "ok":
                dl_ok += 1
            elif status == "cached":
                cached_ok += 1
            elif status == "empty":
                empty_cnt += 1
            else:
                err_cnt += 1
            if b_df is not None and not b_df.empty:
                base_chunks.append(b_df)
            if a_df is not None and not a_df.empty:
                adv_chunks.append(a_df)
            if done_cnt % 100 == 0 or done_cnt == len(target_dates):
                print(f"  [{ticker:6s}]   {done_cnt}/{len(target_dates)} days -> +{dl_ok} new, {cached_ok} cached, {empty_cnt} empty, {err_cnt} err", flush=True)

    if base_chunks:
        df_base = pd.concat(base_chunks).sort_index()
        df_base = df_base[~df_base.index.duplicated(keep="last")]
        buf = io.BytesIO()
        df_base.to_parquet(buf, engine="pyarrow")
        buf.seek(0)
        bucket.blob(f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet").upload_from_file(buf)

        ohlcv_blob = bucket.blob(f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet")
        if ohlcv_blob.exists():
            df_ohlcv = pd.read_parquet(io.BytesIO(ohlcv_blob.download_as_bytes()))
            if df_ohlcv.index.tz is None:
                df_ohlcv.index = df_ohlcv.index.tz_localize("UTC")
            df_master = df_ohlcv.join(df_base, how="left")
            for c in ["Open", "High", "Low", "Close"]:
                if c in df_master.columns:
                    df_master[c] = df_master[c].ffill()
            for c in ["Volume", "Total_Vol", "Volume_Delta", "Trade_Count", "Delta_Pct", "Spread", "OBI", "bsize", "asize"]:
                if c in df_master.columns:
                    df_master[c] = df_master[c].fillna(0)
            buf_m = io.BytesIO()
            df_master.to_parquet(buf_m, engine="pyarrow")
            buf_m.seek(0)
            bucket.blob(f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet").upload_from_file(buf_m)
            print(f"  [{ticker:6s}] Saved Master_Inference_v2 ({len(df_master):,} rows)", flush=True)
            del df_ohlcv, df_master

    if adv_chunks:
        df_adv = pd.concat(adv_chunks).sort_index()
        df_adv = df_adv[~df_adv.index.duplicated(keep="last")]
        buf_a = io.BytesIO()
        df_adv.to_parquet(buf_a, engine="pyarrow")
        buf_a.seek(0)
        bucket.blob(f"{GCS_ADV_MICRO_DIR}/{ticker}_1m_advanced_micro.parquet").upload_from_file(buf_a)
        print(f"  [{ticker:6s}] Saved Microstructure_Advanced ({len(df_adv):,} rows)", flush=True)

    del base_chunks, adv_chunks
    _release_memory()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
print("Loading prioritized manifest from gs://time-series-bucket-ys/Data/Config/priority_missing_tq_tickers.json...", flush=True)
manifest_bytes = bucket.blob("Data/Config/priority_missing_tq_tickers.json").download_as_bytes()
manifest_data = json.loads(manifest_bytes.decode("utf-8"))

missing_list = manifest_data["missing_prioritized"]
existing_refresh_list = manifest_data["existing_needing_post_sep8_refresh"]

# Build non-colliding queue:
# 1) Ranks START_RANK (16) -> END_RANK (150) forward
# 2) Existing 26 tickers needing Sep 9 - Nov 9 refresh
# 3) Ranks 15 -> 1 in reverse (skipping any ticker already finished or locked by the Mac)
queue = (
    missing_list[max(0, START_RANK - 1):END_RANK] +
    existing_refresh_list +
    list(reversed(missing_list[:max(0, START_RANK - 1)]))
)

full_bdays = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(HISTORY_START, datetime.now(pytz.UTC).strftime("%Y-%m-%d"))]

print("=" * 95, flush=True)
print(f"GCP PARALLEL SINGLE-PASS T&Q WORKER  |  Workers: {DAY_WORKERS}  |  Queue: {len(queue)} Tickers (Ranks #{START_RANK}->#{END_RANK} first)", flush=True)
print("=" * 95, flush=True)

for idx, item in enumerate(queue, 1):
    _refresh_bucket()
    tk = item["ticker"]
    if not try_claim_ticker(tk):
        print(f"[{idx}/{len(queue)}] Skipping {tk} (Already completed or actively locked by Mac worker)", flush=True)
        continue
    try:
        t0 = time.time()
        print(f"\n[{idx}/{len(queue)}] Claimed {tk} (Rank #{item.get('priority_rank', 'Refresh')}, Score {item.get('priority_score', 0):.1f})...", flush=True)
        ensure_ohlcv_parquet(tk, HISTORY_START)
        run_full_ticker_single_pass(tk, full_bdays)
        print(f"  [{tk:6s}] Completed in {(time.time() - t0)/60:.1f} min!", flush=True)
    finally:
        release_ticker_lock(tk)
