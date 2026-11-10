# %% ── Prioritized Positive-Events T&Q Downloader + Full Microstructure Builder ──
#
# Downloads missing Polygon Trades & Quotes (T&Q), 1-minute OHLCV, Base Microstructure,
# Master_Inference_v2, and 12-feature Advanced Lee-Ready Microstructure for the
# prioritized missing tickers from the 28-month Positive-Spike / Combo C2 backtest.
#
# Supports two download modes:
#   1. --mode=events (DEFAULT, ~35x faster):
#      Downloads raw T&Q strictly for [prior_trading_day, event_date] around every
#      positive gap/spike event in the backtest, plus full 1-min OHLCV.
#      - Batch 1 (Top 50 tickers):  only 1,265 ticker-days (vs ~30,000 full history days)
#      - Batch 1+2 (Top 150):       only 2,527 ticker-days (covers 418 C2 trades!)
#      - Batch 1+2+3 (Top 300):     only 3,847 ticker-days (covers 641 C2 trades = 73%!)
#   2. --mode=full:
#      Downloads all business days from 2024-06-01 to present for each selected ticker.
#
# Usage Examples:
#   # Fast Event-Window Mode for Top 50 Priority Missing Tickers (Batch 1):
#   python bulk_priority_pos_events_tq.py --top=50 --mode=events
#
#   # Fast Event-Window Mode for Top 150 Priority Missing Tickers (Batch 1+2) on VM:
#   python bulk_priority_pos_events_tq.py --top=150 --mode=events --sa
#
#   # Also refresh the 51 already-downloaded tickers for Sep 9 -> Nov 6, 2026 events:
#   python bulk_priority_pos_events_tq.py --refresh-existing --top=50 --mode=events --sa
#
#   # Specific tickers only:
#   python bulk_priority_pos_events_tq.py SDOT DUO SERV ASST OPTT EHGO --mode=events

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

# ==============================================================================
# CONFIGURATION & CLI FLAGS
# ==============================================================================
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
BUCKET_NAME     = os.environ.get("GCS_BUCKET", "time-series-bucket-ys")
PROJECT_ID      = os.environ.get("GCP_PROJECT", "ys-time-series")

GCS_RAW_DIR       = "Data/TradesQuotes"
GCS_OHLCV_DIR     = "Data/Inference"
GCS_MICRO_DIR     = "Data/Inference/Microstructure"
GCS_ADV_MICRO_DIR = "Data/Inference/Microstructure_Advanced"
GCS_MASTER_DIR    = "Data/Inference_v2"

MANIFEST_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "priority_missing_tq_tickers.json")

HISTORY_START = "2024-06-01"
PE_ORDER = 5
MIN_TRADES_PER_BAR = 6
POLYGON_SLEEP = 0.05
DAY_WORKERS = 8

ADVANCED_MICRO_COLS = [
    "LR_OFI", "LR_VPIN", "LR_Buy_Ratio", "LR_Size_Imbalance",
    "PE", "LZC", "Tick_Volatility", "Shock_Persist",
    "Trade_Size_CV", "Trade_Intensity", "Quote_Intensity", "Spread_CV",
    "Odd_Lot_OFI", "Block_OFI", "Signed_Semivariance", "Ask_Step_Down_Rate",
]

USE_SERVICE_ACCOUNT = "--sa" in sys.argv or os.environ.get("USE_SERVICE_ACCOUNT", "") == "1"
REFRESH_EXISTING    = "--refresh-existing" in sys.argv
DRY_RUN             = "--dry-run" in sys.argv

MODE = "events"
TOP_N = 50
START_RANK = 1
END_RANK = None
CUSTOM_TICKERS = []

for arg in sys.argv[1:]:
    if arg.startswith("--mode="):
        MODE = arg.split("=", 1)[1].strip().lower()
    elif arg.startswith("--top="):
        val = arg.split("=", 1)[1].strip().lower()
        TOP_N = 99999 if val == "all" else int(val)
    elif arg.startswith("--start-rank="):
        START_RANK = int(arg.split("=", 1)[1].strip())
    elif arg.startswith("--end-rank="):
        END_RANK = int(arg.split("=", 1)[1].strip())
    elif arg == "--all":
        TOP_N = 99999
    elif arg.startswith("--workers="):
        DAY_WORKERS = int(arg.split("=", 1)[1].strip())
    elif not arg.startswith("-"):
        CUSTOM_TICKERS.append(arg.upper())

def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

LOCAL_SA_KEY = "/Users/yaninaso/Downloads/ys-time-series-sa-key.json"

def _init_gcs_client():
    if os.path.exists(LOCAL_SA_KEY):
        client = gcs_storage.Client.from_service_account_json(LOCAL_SA_KEY, project=PROJECT_ID)
        print(f"  Auth: Service Account JSON Key ({LOCAL_SA_KEY}) — never expires")
    elif USE_SERVICE_ACCOUNT:
        from google.auth import compute_engine
        creds = compute_engine.Credentials()
        client = gcs_storage.Client(credentials=creds, project=PROJECT_ID)
        print("  Auth: VM compute service account (never expires)")
    else:
        client = gcs_storage.Client(project=PROJECT_ID)
        print("  Auth: Application Default Credentials")
    return client

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
# STEP 1: ENSURE 1-MINUTE OHLCV PARQUET EXISTS ON GCS
# ==============================================================================
def ensure_ohlcv_parquet(ticker, start_date=HISTORY_START):
    ohlcv_path = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
    blob = bucket.blob(ohlcv_path)
    if blob.exists() and not REFRESH_EXISTING:
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
                    time.sleep(15 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            break

        if "results" in data and data.get("resultsCount", 0) > 0:
            all_frames.append(pd.DataFrame(data["results"]))
            url = data["next_url"] + f"&apiKey={POLYGON_API_KEY}" if "next_url" in data else None
        else:
            url = None

    if not all_frames:
        print(f"  [{ticker:6s}] WARNING: No 1m OHLCV returned from Polygon")
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
    print(f"  [{ticker:6s}] Saved 1m OHLCV: {len(df):,} rows ({df.index.min().strftime('%Y-%m-%d')} -> {df.index.max().strftime('%Y-%m-%d')})")
    del df, all_frames
    _release_memory()
    return True

# ==============================================================================
# STEP 2: DOWNLOAD MISSING RAW TRADES & QUOTES FOR TARGET DATES
# ==============================================================================
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

def _download_one_tq_day(ticker, date_str):
    poly = RESTClient(POLYGON_API_KEY)
    day_start = pd.Timestamp(date_str, tz="UTC")
    day_end = day_start + timedelta(days=1)
    try:
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
        time.sleep(POLYGON_SLEEP)
        dt = pd.DataFrame(trades) if trades else pd.DataFrame()
        dq = pd.DataFrame(quotes) if quotes else pd.DataFrame()
        if dt.empty and dq.empty:
            return "empty", date_str
        if not dt.empty:
            buf = io.BytesIO()
            dt.to_parquet(buf, engine="pyarrow")
            buf.seek(0)
            bucket.blob(f"{GCS_RAW_DIR}/{ticker}/{ticker}_trades_{date_str}.parquet").upload_from_file(buf)
        if not dq.empty:
            buf = io.BytesIO()
            dq.to_parquet(buf, engine="pyarrow")
            buf.seek(0)
            bucket.blob(f"{GCS_RAW_DIR}/{ticker}/{ticker}_quotes_{date_str}.parquet").upload_from_file(buf)
        return "ok", date_str
    except Exception as e:
        return f"err:{e}", date_str

def download_tq_for_ticker(ticker, target_dates):
    existing = scan_existing_tq_dates(ticker)
    missing = sorted(set(target_dates) - existing)
    if not missing:
        print(f"  [{ticker:6s}] All {len(target_dates)} target T&Q days already cached on GCS")
        return 0, len(existing)

    print(f"  [{ticker:6s}] Downloading {len(missing)} missing T&Q days (cached: {len(existing)})...")
    dl_ok, dl_empty, dl_err = 0, 0, 0
    done_cnt = 0
    with ThreadPoolExecutor(max_workers=DAY_WORKERS) as pool:
        futs = {pool.submit(_download_one_tq_day, ticker, d): d for d in missing}
        for fut in as_completed(futs):
            status, _ = fut.result()
            done_cnt += 1
            if status == "ok":
                dl_ok += 1
            elif status == "empty":
                dl_empty += 1
            else:
                dl_err += 1
            if done_cnt % 50 == 0 or done_cnt == len(missing):
                print(f"  [{ticker:6s}]   {done_cnt}/{len(missing)} days -> +{dl_ok} ok, {dl_empty} empty, {dl_err} err", flush=True)
    print(f"  [{ticker:6s}] T&Q Download finished: +{dl_ok} ok, {dl_empty} empty, {dl_err} err")
    return dl_ok, len(existing) + dl_ok

# ==============================================================================
# STEP 3 & 4: BUILD BASE MICRO, MASTER_INFERENCE_V2, AND LEE-READY ADV MICRO
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

def process_day_base_and_adv_micro(ticker, date_str):
    t_blob = bucket.blob(f"{GCS_RAW_DIR}/{ticker}/{ticker}_trades_{date_str}.parquet")
    q_blob = bucket.blob(f"{GCS_RAW_DIR}/{ticker}/{ticker}_quotes_{date_str}.parquet")
    if not t_blob.exists() or not q_blob.exists():
        return None, None
    try:
        dt = normalize_raw_schema(pd.read_parquet(io.BytesIO(t_blob.download_as_bytes())), "trades")
        dq = normalize_raw_schema(pd.read_parquet(io.BytesIO(q_blob.download_as_bytes())), "quotes")
        if dt.empty or dq.empty:
            return None, None

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
        dq["bar"] = dq["timestamp"].dt.floor("1min")
        q_grouped = {ts: grp for ts, grp in dq.groupby("bar")}

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

def rebuild_all_micro_and_master(ticker):
    avail_dates = sorted(scan_existing_tq_dates(ticker))
    if not avail_dates:
        return
    base_chunks, adv_chunks = [], []
    with ThreadPoolExecutor(max_workers=DAY_WORKERS) as pool:
        futs = {pool.submit(process_day_base_and_adv_micro, ticker, d): d for d in avail_dates}
        for fut in as_completed(futs):
            b_df, a_df = fut.result()
            if b_df is not None and not b_df.empty:
                base_chunks.append(b_df)
            if a_df is not None and not a_df.empty:
                adv_chunks.append(a_df)

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
            print(f"  [{ticker:6s}] Saved Master_Inference_v2 ({len(df_master):,} rows)")
            del df_ohlcv, df_master

    if adv_chunks:
        df_adv = pd.concat(adv_chunks).sort_index()
        df_adv = df_adv[~df_adv.index.duplicated(keep="last")]
        buf_a = io.BytesIO()
        df_adv.to_parquet(buf_a, engine="pyarrow")
        buf_a.seek(0)
        bucket.blob(f"{GCS_ADV_MICRO_DIR}/{ticker}_1m_advanced_micro.parquet").upload_from_file(buf_a)
        print(f"  [{ticker:6s}] Saved Microstructure_Advanced ({len(df_adv):,} rows, {len(avail_dates)} T&Q days)")
    _release_memory()

# ==============================================================================
# MAIN EXECUTION LOOP
# ==============================================================================
if not os.path.exists(MANIFEST_JSON):
    print("  Downloading priority_missing_tq_tickers.json from gs://time-series-bucket-ys/Data/Config/...")
    bucket.blob("Data/Config/priority_missing_tq_tickers.json").download_to_filename(MANIFEST_JSON)

with open(MANIFEST_JSON, "r") as f:
    manifest_data = json.load(f)

missing_list = manifest_data["missing_prioritized"]
existing_refresh_list = manifest_data["existing_needing_post_sep8_refresh"]

queue = []
if CUSTOM_TICKERS:
    lookup = {r["ticker"]: r for r in missing_list + existing_refresh_list}
    for tk in CUSTOM_TICKERS:
        if tk in lookup:
            queue.append(lookup[tk])
        else:
            queue.append({"ticker": tk, "tq_window_dates": [], "priority_rank": 0, "priority_score": 0.0})
else:
    if REFRESH_EXISTING:
        queue.extend(existing_refresh_list)
    end_idx = END_RANK if END_RANK is not None else (START_RANK - 1 + TOP_N)
    queue.extend(missing_list[max(0, START_RANK - 1):end_idx])

tot_days_est = sum(len(item.get("tq_window_dates", [])) for item in queue)
print("=" * 95)
print(f"PRIORITIZED POSITIVE-EVENTS T&Q PIPELINE  |  Mode: {MODE.upper()}  |  Tickers: {len(queue)}  |  Target Event+Prior Days: {tot_days_est}")
print("=" * 95)

if DRY_RUN:
    for idx, item in enumerate(queue[:30], 1):
        print(f"  {idx:>3}. {item['ticker']:<6} | Score: {item.get('priority_score', 0):5.1f} | Target T&Q Days: {len(item.get('tq_window_dates', [])):>3}")
    sys.exit(0)

full_bdays = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(HISTORY_START, datetime.now(pytz.UTC).strftime("%Y-%m-%d"))]

for idx, item in enumerate(queue, 1):
    _refresh_bucket()
    tk = item["ticker"]
    if bucket.blob(f"{GCS_ADV_MICRO_DIR}/{tk}_1m_advanced_micro.parquet").exists() and bucket.blob(f"{GCS_MASTER_DIR}/{tk}_master_inference.parquet").exists() and not REFRESH_EXISTING:
        print(f"[{idx}/{len(queue)}] {tk} already has Master_Inference_v2 & Microstructure_Advanced on GCS — skipping!", flush=True)
        continue
    lock_blob = bucket.blob(f"Data/Locks/{tk}.lock")
    if lock_blob.exists():
        lock_blob.reload()
        if lock_blob.updated and (datetime.now(pytz.UTC) - lock_blob.updated).total_seconds() < 2700:
            print(f"[{idx}/{len(queue)}] {tk} is actively locked by GCP worker ({lock_blob.updated}) — skipping!", flush=True)
            continue
    lock_blob.upload_from_string(f"claimed_by_mac_{datetime.now(pytz.UTC).isoformat()}")
    try:
        target_dates = item.get("tq_window_dates", []) if (MODE == "events" and item.get("tq_window_dates")) else full_bdays
        print(f"\n[{idx}/{len(queue)}] Processing {tk} (Rank #{item.get('priority_rank', '-')}, Score {item.get('priority_score', 0):.1f}, {len(target_dates)} target T&Q days)...", flush=True)
        ensure_ohlcv_parquet(tk, HISTORY_START)
        dl_new, tot_cached = download_tq_for_ticker(tk, target_dates)
        if dl_new > 0 or not bucket.blob(f"{GCS_MASTER_DIR}/{tk}_master_inference.parquet").exists() or not bucket.blob(f"{GCS_ADV_MICRO_DIR}/{tk}_1m_advanced_micro.parquet").exists():
            rebuild_all_micro_and_master(tk)
    finally:
        try:
            if lock_blob.exists():
                lock_blob.delete()
        except Exception:
            pass

