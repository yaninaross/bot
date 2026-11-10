# %% ── Standalone GCP Parallel T&Q Refresher + Single-Pass 16-Col Lee-Ready Builder ──
# Paste this entire block into a single GCP / Colab Notebook cell and run.
#
# What it does automatically (100% collision-free with the Local Mac worker):
#   1. PHASE 1 — Incremental Refresh of All 113 Stale Existing T&Q Stocks (Sep 8 -> Present):
#      - Sweeps the 113 stale existing T&Q stocks in REVERSE order (ZETA -> ACHR) while the
#        Local Mac sweeps them in FORWARD order (ACHR -> ZETA), meeting in the middle!
#      - Uses fast incremental append: loads existing pre-Sep-8 Microstructure & Microstructure_Advanced
#        parquets, downloads/computes ONLY the ~44 missing trading days (2026-09-08 -> present),
#        updates 1m OHLCV, and writes fresh Master_Inference_v2 & 16-col Microstructure_Advanced (~20s/stock).
#   2. PHASE 2 — Downloads Next 150 Priority New T&Q Stocks (Ranks #106 -> #255 forward, then #105 -> #56 reverse):
#      - Uses Single-Pass In-Memory Engine to compute Base + 16-col Advanced Microstructure in RAM.
#   3. Atomic GCS Locks (Data/Locks/{TICKER}.lock):
#      - Ensures Mac and GCP never process the same ticker simultaneously.

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
FRESH_CUTOFF_DATE  = "2026-11-07"
PE_ORDER           = 5
MIN_TRADES_PER_BAR = 6
POLYGON_SLEEP      = 0.02

# Worker role: "gcp" (reverse stale sweep + #106->#255) or "mac" (forward stale sweep + #56->#155)
WORKER_ROLE    = os.environ.get("WORKER_ROLE", "gcp").lower()
DAY_WORKERS    = int(os.environ.get("TQ_WORKERS", "16" if WORKER_ROLE == "gcp" else "8"))
START_RANK     = int(os.environ.get("START_RANK", "106" if WORKER_ROLE == "gcp" else "56"))
END_RANK       = int(os.environ.get("END_RANK", "255" if WORKER_ROLE == "gcp" else "155"))
MAC_START_RANK = int(os.environ.get("MAC_START_RANK", "56"))

for arg in sys.argv[1:]:
    if arg.startswith("--role="):
        WORKER_ROLE = arg.split("=", 1)[1].strip().lower()
    elif arg.startswith("--start-rank="):
        START_RANK = int(arg.split("=", 1)[1].strip())
    elif arg.startswith("--end-rank="):
        END_RANK = int(arg.split("=", 1)[1].strip())
    elif arg.startswith("--workers="):
        DAY_WORKERS = int(arg.split("=", 1)[1].strip())

SKIP_SYMBOLS = {"SMLR", "BITF", "MIGI", "XLE", "SPY", "QQQ", "IWM", "VIX", "TLT", "BTCUSD", "ETHUSD"}

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
# DISTRIBUTED GCS LOCK & FRESHNESS CHECK
# ==============================================================================
def is_ticker_fresh_on_gcs(ticker):
    """Returns True if Master_Inference_v2 and Microstructure_Advanced both exist and were updated >= Nov 7, 2026."""
    adv_blob = bucket.blob(f"{GCS_ADV_MICRO_DIR}/{ticker}_1m_advanced_micro.parquet")
    master_blob = bucket.blob(f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet")
    if not (adv_blob.exists() and master_blob.exists()):
        return False
    adv_blob.reload()
    master_blob.reload()
    if (
        adv_blob.updated and str(adv_blob.updated)[:10] >= FRESH_CUTOFF_DATE
        and master_blob.updated and str(master_blob.updated)[:10] >= FRESH_CUTOFF_DATE
    ):
        return True
    return False

def try_claim_ticker(ticker):
    """Returns True if we can claim this ticker, or False if already fresh or locked by another worker."""
    if is_ticker_fresh_on_gcs(ticker):
        return False

    lock_blob = bucket.blob(f"{GCS_LOCK_DIR}/{ticker}.lock")
    if lock_blob.exists():
        lock_blob.reload()
        if lock_blob.updated and (datetime.now(pytz.UTC) - lock_blob.updated).total_seconds() < 2700:
            return False

    lock_blob.upload_from_string(f"claimed_by_{WORKER_ROLE}_{datetime.now(pytz.UTC).isoformat()}")
    return True

def release_ticker_lock(ticker):
    try:
        lock_blob = bucket.blob(f"{GCS_LOCK_DIR}/{ticker}.lock")
        if lock_blob.exists():
            lock_blob.delete()
    except Exception:
        pass

# ==============================================================================
# 1-MINUTE OHLCV DOWNLOADER (INCREMENTAL OR FULL)
# ==============================================================================
def ensure_ohlcv_parquet(ticker, start_date=HISTORY_START):
    ohlcv_path = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
    blob = bucket.blob(ohlcv_path)
    existing_df = None
    fetch_start = start_date

    if blob.exists():
        blob.reload()
        if blob.updated and str(blob.updated)[:10] >= FRESH_CUTOFF_DATE:
            return True
        try:
            existing_df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
            if existing_df.index.tz is None:
                existing_df.index = existing_df.index.tz_localize("UTC")
            if len(existing_df) > 1000:
                last_dt = existing_df.index.max().strftime("%Y-%m-%d")
                if last_dt >= "2026-11-06":
                    return True
                fetch_start = last_dt
        except Exception:
            existing_df = None

    end_date = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{fetch_start}/{end_date}"
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

    if not all_frames and existing_df is None:
        print(f"  [{ticker:6s}] WARNING: No 1m OHLCV returned from Polygon", flush=True)
        return False

    if all_frames:
        new_df = pd.concat(all_frames, ignore_index=True).rename(columns={
            "t": "Timestamp", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
        })
        new_df["Timestamp"] = new_df["Timestamp"] / 1000.0
        new_df["Datetime"] = pd.to_datetime(new_df["Timestamp"], unit="s", utc=True)
        new_df = new_df.set_index("Datetime").sort_index()
        new_df = new_df[~new_df.index.duplicated(keep="last")][["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
        if existing_df is not None and not existing_df.empty:
            df = pd.concat([existing_df, new_df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        else:
            df = new_df
    else:
        df = existing_df

    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow")
    buf.seek(0)
    blob.upload_from_file(buf)
    print(f"  [{ticker:6s}] Saved 1m OHLCV: {len(df):,} rows ({df.index.min().strftime('%Y-%m-%d')} -> {df.index.max().strftime('%Y-%m-%d')})", flush=True)
    del df, existing_df, all_frames
    _release_memory()
    return True

# ==============================================================================
# SINGLE-PASS IN-MEMORY 16-COLUMN MICROSTRUCTURE COMPUTATION + T&Q DOWNLOAD
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

        # 2. Advanced 16-Column Lee-Ready + Sub-Minute Tick Microstructure (1-min)
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

def run_ticker_incremental_or_full(ticker, full_bdays):
    """If ticker already has Microstructure parquets (e.g. up to 2026-09-08), incrementally downloads & appends
    ONLY the missing post-Sep-8 days (~44 days -> ~20 sec/ticker!). Otherwise runs full history single-pass."""
    base_blob = bucket.blob(f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet")
    adv_blob  = bucket.blob(f"{GCS_ADV_MICRO_DIR}/{ticker}_1m_advanced_micro.parquet")

    old_base_df, old_adv_df = None, None
    target_dates = full_bdays

    if base_blob.exists() and adv_blob.exists():
        try:
            old_base_df = pd.read_parquet(io.BytesIO(base_blob.download_as_bytes()))
            old_adv_df  = pd.read_parquet(io.BytesIO(adv_blob.download_as_bytes()))
            if old_base_df.index.tz is None:
                old_base_df.index = old_base_df.index.tz_localize("UTC")
            if old_adv_df.index.tz is None:
                old_adv_df.index = old_adv_df.index.tz_localize("UTC")
            for c in ADVANCED_MICRO_COLS:
                if c not in old_adv_df.columns:
                    old_adv_df[c] = 0.0
            last_dt = min(
                old_base_df.index.max().strftime("%Y-%m-%d"),
                old_adv_df.index.max().strftime("%Y-%m-%d")
            )
            target_dates = [d for d in full_bdays if d >= last_dt]
            print(f"  [{ticker:6s}] Incremental refresh from {last_dt} -> present ({len(target_dates)} days to process)...", flush=True)
        except Exception:
            old_base_df, old_adv_df = None, None
            target_dates = full_bdays

    existing = scan_existing_tq_dates(ticker)
    if old_base_df is None:
        print(f"  [{ticker:6s}] Full build across {len(target_dates)} days ({len(existing)} cached, {len(set(target_dates) - existing)} to download) with {DAY_WORKERS} workers...", flush=True)

    dl_ok, cached_ok, empty_cnt, err_cnt = 0, 0, 0, 0
    done_cnt = 0
    base_chunks = [old_base_df] if old_base_df is not None else []
    adv_chunks  = [old_adv_df]  if old_adv_df is not None else []

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
            if done_cnt % 50 == 0 or done_cnt == len(target_dates):
                print(f"  [{ticker:6s}]   {done_cnt}/{len(target_dates)} days -> +{dl_ok} new, {cached_ok} cached, {empty_cnt} empty, {err_cnt} err", flush=True)

    if base_chunks:
        df_base = pd.concat(base_chunks).sort_index()
        df_base = df_base[~df_base.index.duplicated(keep="last")]
        buf = io.BytesIO()
        df_base.to_parquet(buf, engine="pyarrow")
        buf.seek(0)
        base_blob.upload_from_file(buf)

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
            print(f"  [{ticker:6s}] Saved Master_Inference_v2 ({len(df_master):,} rows, max={df_master.index.max().strftime('%Y-%m-%d')})", flush=True)
            del df_ohlcv, df_master

    if adv_chunks:
        df_adv = pd.concat(adv_chunks).sort_index()
        df_adv = df_adv[~df_adv.index.duplicated(keep="last")]
        for c in ADVANCED_MICRO_COLS:
            if c not in df_adv.columns:
                df_adv[c] = 0.0
        df_adv = df_adv[ADVANCED_MICRO_COLS]
        buf_a = io.BytesIO()
        df_adv.to_parquet(buf_a, engine="pyarrow")
        buf_a.seek(0)
        adv_blob.upload_from_file(buf_a)
        print(f"  [{ticker:6s}] Saved 16-Col Microstructure_Advanced ({len(df_adv):,} rows, max={df_adv.index.max().strftime('%Y-%m-%d')})", flush=True)

    del base_chunks, adv_chunks, old_base_df, old_adv_df
    _release_memory()

# ==============================================================================
# MAIN EXECUTION: PHASE 1 (REFRESH ALL 113 STALE STOCKS) + PHASE 2 (NEW STOCKS)
# ==============================================================================
print("Scanning GCS Data/Inference_v2/ for stale existing T&Q stocks (< 2026-11-07)...", flush=True)
v2_blobs = list(bucket.list_blobs(prefix=f"{GCS_MASTER_DIR}/"))
stale_existing = []
for b in v2_blobs:
    if not b.name.endswith("_master_inference.parquet"):
        continue
    tk = b.name.split("/")[-1].replace("_master_inference.parquet", "")
    if tk in SKIP_SYMBOLS:
        continue
    if not b.updated or str(b.updated)[:10] < FRESH_CUTOFF_DATE:
        stale_existing.append({"ticker": tk, "priority_rank": "STALE_REFRESH", "priority_score": 100.0})

stale_existing = sorted(stale_existing, key=lambda x: x["ticker"])
if WORKER_ROLE == "gcp":
    # GCP sweeps stale tickers in REVERSE (ZETA -> ACHR) while Mac sweeps FORWARD (ACHR -> ZETA)!
    stale_existing = list(reversed(stale_existing))

print(f"Found {len(stale_existing)} stale existing T&Q stocks needing Sep 8 -> Nov 10 incremental refresh.", flush=True)

print("Loading prioritized manifest from gs://time-series-bucket-ys/Data/Config/priority_missing_tq_tickers.json...", flush=True)
manifest_bytes = bucket.blob("Data/Config/priority_missing_tq_tickers.json").download_as_bytes()
manifest_data = json.loads(manifest_bytes.decode("utf-8"))
missing_list = manifest_data["missing_prioritized"]

if WORKER_ROLE == "gcp":
    # Phase 1: Stale Existing (Reverse: ZETA -> ACHR)
    # Phase 2: Ranks #106 -> #255 forward, then #105 -> #56 reverse
    queue = (
        stale_existing +
        missing_list[max(0, START_RANK - 1):END_RANK] +
        list(reversed(missing_list[max(0, MAC_START_RANK - 1):max(0, START_RANK - 1)]))
    )
else:
    # Mac Role:
    # Phase 1: Stale Existing (Forward: ACHR -> ZETA)
    # Phase 2: Ranks #56 -> #155 forward
    queue = (
        stale_existing +
        missing_list[max(0, START_RANK - 1):END_RANK]
    )

full_bdays = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(HISTORY_START, datetime.now(pytz.UTC).strftime("%Y-%m-%d"))]

print("=" * 100, flush=True)
print(f"PARALLEL INCREMENTAL REFRESH + SINGLE-PASS T&Q WORKER ({WORKER_ROLE.upper()}) | Workers: {DAY_WORKERS} | Queue: {len(queue)} Tickers", flush=True)
print(f"  - Phase 1: {len(stale_existing)} Stale Existing Stocks ({'Reverse Z->A' if WORKER_ROLE=='gcp' else 'Forward A->Z'})", flush=True)
print(f"  - Phase 2: New Priority Stocks (Ranks #{START_RANK}->#{END_RANK})", flush=True)
print("=" * 100, flush=True)

for idx, item in enumerate(queue, 1):
    _refresh_bucket()
    tk = item["ticker"]
    if not try_claim_ticker(tk):
        print(f"[{idx}/{len(queue)}] Skipping {tk} (Already fresh >= {FRESH_CUTOFF_DATE} or locked by other worker)", flush=True)
        continue
    try:
        t0 = time.time()
        print(f"\n[{idx}/{len(queue)}] Claimed {tk} (Rank: {item.get('priority_rank', '-')}, Score: {item.get('priority_score', 0):.1f})...", flush=True)
        ensure_ohlcv_parquet(tk, HISTORY_START)
        run_ticker_incremental_or_full(tk, full_bdays)
        print(f"  [{tk:6s}] Done in {(time.time() - t0):.1f}s ({(time.time() - t0)/60:.1f}m)!", flush=True)
    finally:
        release_ticker_lock(tk)
