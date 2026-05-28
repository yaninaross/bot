"""
rebuild_master_all.py — Full Rebuild of All master_inference Parquets (v2)
==========================================================================
Reads existing raw OHLCV + microstructure parquets from GCS and rebuilds
every master_inference.parquet with heal_data applied to the ENTIRE file.

Writes to Data/Inference_v2/ to distinguish from old (corrupted) files
where Spread/OBI/bsize/asize were ffilled instead of zero-filled.

Does NOT fetch from Polygon — uses whatever raw data is already in GCS.
"""

import sys
import pandas as pd
from datetime import datetime
import pytz
import warnings
from google.cloud import storage as gcs_storage

warnings.filterwarnings('ignore')

FORCE_REBUILD = '--force' in sys.argv

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TARGET_TICKERS = [
    # --- Exp-97/110-113/197 (Crypto Cluster) ---
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'MSTR', 'COIN',
    'KEEL', 'HUT', 'CIFR', 'BTBT', 'BGDE', 'HOOD', 'CORZ',
    'BITF', 'MIGI',  # legacy names kept for old experiment compatibility
    'SMLR',          # delisted ~Jan 2026 but needed for training history

    # --- Exp-110/111 Tech Inference Targets ---
    'SMCI', 'TSLA', 'VRT', 'AMD', 'NVDA',

    # --- Exp-100/103/104 (Defense Sector) ---
    'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS', 'LUNR', 'SPIR',
    'JOBY', 'NNDM', 'MVIS', 'RCAT', 'ACHR',

    # --- Exp-150 (Energy / AI Power) ---
    'VST', 'CEG', 'GEV', 'BE', 'CCJ', 'NEE', 'DUK',

    # --- Exp-152/160/170/200 (Cross-Sector) ---
    'ALAB', 'FCEL', 'FLNC', 'PLUG', 'MP', 'POWL', 'OKLO', 'SMR',

    # --- Exp-153 (Space / Dual-Use) ---
    'PL', 'RDW', 'BKSY', 'MNTS', 'VOYG',

    # --- Defense Primes ---
    'GD', 'LHX', 'LMT', 'NOC', 'RTX', 'GE',
]

CONTEXT_ASSETS = ['ITA', 'SMH', 'MOD', 'I:VIX', 'TLT', 'IWM', 'X:BTCUSD', 'SPY', 'XLE']

BUCKET_NAME     = "time-series-bucket-ys"
GCS_OHLCV_DIR   = f"gs://{BUCKET_NAME}/Data/Inference"
GCS_MICRO_DIR   = f"gs://{BUCKET_NAME}/Data/Inference/Microstructure"
GCS_OUTPUT_DIR  = f"gs://{BUCKET_NAME}/Data/Inference_v2"

# ==============================================================================
# HEAL — resample to 1-min continuous + ffill/zero-fill
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df

# ==============================================================================
# MAIN
# ==============================================================================
print("rebuild_master_all.py — Full Master Inference Rebuild (v2)")
print(f"   Run time: {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print(f"   Output:   {GCS_OUTPUT_DIR}")

ALL_ASSETS = (
    [(t, False) for t in list(set(TARGET_TICKERS))] +
    [(t, True)  for t in list(set(CONTEXT_ASSETS))]
)

cli_tickers = [a.upper() for a in sys.argv[1:] if not a.startswith('-') and not a.endswith('.json')]
if cli_tickers:
    ALL_ASSETS = [(t, ctx) for t, ctx in ALL_ASSETS
                  if t.upper() in cli_tickers or t.replace("I:","").replace("X:","").upper() in cli_tickers]
    print(f"   Filtered to: {[t for t,_ in ALL_ASSETS]}")
else:
    print(f"   Rebuilding ALL {len(ALL_ASSETS)} assets")

gcs_client = gcs_storage.Client()
gcs_bucket = gcs_client.bucket(BUCKET_NAME)

success, failed, skipped = 0, 0, 0

for ticker, is_context in ALL_ASSETS:
    print(f"\n{'─'*52}")
    print(f"{'[context]' if is_context else '[target]'} {ticker}")
    clean_ticker = ticker.replace("I:", "").replace("X:", "")

    ohlcv_uri  = f"{GCS_OHLCV_DIR}/{clean_ticker}_1m_full_historical_data.parquet"
    micro_uri  = f"{GCS_MICRO_DIR}/{clean_ticker}_1m_microstructure.parquet"
    master_uri = f"{GCS_OUTPUT_DIR}/{clean_ticker}_master_inference.parquet"

    # ── Skip if master is newer than sources ────────────────────────────
    if not FORCE_REBUILD:
        try:
            master_blob = gcs_bucket.blob(f"Data/Inference_v2/{clean_ticker}_master_inference.parquet")
            ohlcv_blob  = gcs_bucket.blob(f"Data/Inference/{clean_ticker}_1m_full_historical_data.parquet")
            micro_blob  = gcs_bucket.blob(f"Data/Inference/Microstructure/{clean_ticker}_1m_microstructure.parquet")
            if master_blob.exists():
                master_blob.reload()
                master_ts = master_blob.updated
                needs_rebuild = False
                if ohlcv_blob.exists():
                    ohlcv_blob.reload()
                    if ohlcv_blob.updated > master_ts:
                        needs_rebuild = True
                if not is_context and micro_blob.exists():
                    micro_blob.reload()
                    if micro_blob.updated > master_ts:
                        needs_rebuild = True
                if not needs_rebuild:
                    print(f"   ✅ master_inference up to date — skipping")
                    skipped += 1
                    continue
        except Exception:
            pass

    # ── Load raw OHLCV ──────────────────────────────────────────────────
    try:
        df_ohlcv = pd.read_parquet(ohlcv_uri)
        if df_ohlcv.index.tz is None:
            df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
        print(f"   OHLCV: {len(df_ohlcv):,} rows  "
              f"({df_ohlcv.index.min().date()} to {df_ohlcv.index.max().date()})")
    except Exception as e:
        print(f"   OHLCV not found — skipping ({e})")
        skipped += 1
        continue

    # ── Load micro (target assets only) ─────────────────────────────────
    df_micro = pd.DataFrame()
    if not is_context:
        try:
            df_micro = pd.read_parquet(micro_uri)
            if df_micro.index.tz is None:
                df_micro.index = df_micro.index.tz_localize('UTC')
            print(f"   Micro: {len(df_micro):,} rows  "
                  f"({df_micro.index.min().date()} to {df_micro.index.max().date()})")
        except Exception:
            print(f"   Micro not found — master will have NaN micro columns")

    # ── Join ────────────────────────────────────────────────────────────
    if not is_context and not df_micro.empty:
        df_unified = df_ohlcv.join(df_micro, how='left')
    else:
        df_unified = df_ohlcv.copy()

    raw_rows = len(df_unified)

    # ── Heal ENTIRE dataset ─────────────────────────────────────────────
    print(f"   Healing entire dataset ({raw_rows:,} raw rows)...")
    try:
        df_unified = heal_data(df_unified)
        df_unified = df_unified[df_unified.index.notna()].sort_index()
        df_unified.to_parquet(master_uri, engine='pyarrow')
        print(f"   master_inference saved: {len(df_unified):,} rows "
              f"(expanded from {raw_rows:,})")
        success += 1
    except Exception as e:
        print(f"   FAILED: {e}")
        failed += 1

print(f"\n{'='*52}")
print(f"DONE — {success} rebuilt, {skipped} skipped (up to date / no OHLCV), {failed} failed")
print(f"   Use --force to rebuild all assets regardless of timestamps")
print("="*52)
