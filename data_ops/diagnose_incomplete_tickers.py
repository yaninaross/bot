"""
diagnose_incomplete_tickers.py — Check why tickers fail the 5% micro quality gate
==================================================================================
For each ticker, checks:
  1. OHLCV file: exists? date range? row count?
  2. Micro file: exists? date range? Volume_Delta nonzero%?
  3. Master_inference file: exists? date range? Volume_Delta nonzero%?
  4. Raw T&Q files in GCS: how many days of data?

Reports the root cause and recommended action for each.
"""

import io
import gc
import warnings
import pandas as pd
from google.cloud import storage

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"
GCS_OHLCV_DIR  = "Data/Inference"
GCS_MICRO_DIR  = "Data/Inference/Microstructure"
GCS_MASTER_DIR = "Data/Inference_v2"
GCS_RAW_DIR    = "Data/Raw"

TICKERS = ['FCEL', 'FCX', 'MOD', 'MU']

QUALITY_THRESHOLD = 5.0

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# ==============================================================================
# HELPERS
# ==============================================================================
def check_blob(path):
    blob = bucket.blob(path)
    return blob.exists()

def load_parquet(path):
    blob = bucket.blob(path)
    if not blob.exists():
        return None
    df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
    if df.index.tz is None and hasattr(df.index, 'tz_localize'):
        try:
            df.index = df.index.tz_localize('UTC')
        except Exception:
            pass
    return df

def count_raw_files(ticker, data_type):
    prefix = f"{GCS_RAW_DIR}/{ticker}/{data_type}/"
    blobs = list(bucket.list_blobs(prefix=prefix, max_results=5000))
    return len(blobs)

def vd_stats(df):
    if df is None or 'Volume_Delta' not in df.columns:
        return 0, 0, 0.0
    total = len(df)
    nz = (df['Volume_Delta'] != 0).sum()
    pct = nz / total * 100 if total > 0 else 0.0
    return total, int(nz), pct

# ==============================================================================
# MAIN
# ==============================================================================
print("=" * 80)
print("  DIAGNOSE INCOMPLETE TICKERS")
print("=" * 80)

results = []

for ticker in TICKERS:
    print(f"\n{'─' * 80}")
    print(f"  {ticker}")
    print(f"{'─' * 80}")

    ohlcv_path = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
    micro_path = f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet"
    master_path = f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet"

    # ── 1. OHLCV ──
    print(f"\n  [1] OHLCV: {ohlcv_path}")
    df_ohlcv = load_parquet(ohlcv_path)
    if df_ohlcv is not None:
        print(f"      Rows: {len(df_ohlcv):,}")
        print(f"      Range: {df_ohlcv.index.min()} → {df_ohlcv.index.max()}")
        ohlcv_start_year = df_ohlcv.index.min().year
        ohlcv_rows = len(df_ohlcv)
    else:
        print(f"      NOT FOUND")
        ohlcv_start_year = None
        ohlcv_rows = 0
    del df_ohlcv; gc.collect()

    # ── 2. Micro ──
    print(f"\n  [2] Micro: {micro_path}")
    df_micro = load_parquet(micro_path)
    if df_micro is not None:
        total_m, nz_m, pct_m = vd_stats(df_micro)
        print(f"      Rows: {total_m:,}")
        print(f"      Range: {df_micro.index.min()} → {df_micro.index.max()}")
        print(f"      Volume_Delta nonzero: {nz_m:,} / {total_m:,} = {pct_m:.1f}%")
        micro_exists = True
        micro_start_year = df_micro.index.min().year
    else:
        print(f"      NOT FOUND")
        micro_exists = False
        micro_start_year = None
        pct_m = 0
    del df_micro; gc.collect()

    # ── 3. Master inference ──
    print(f"\n  [3] Master: {master_path}")
    df_master = load_parquet(master_path)
    if df_master is not None:
        total_ma, nz_ma, pct_ma = vd_stats(df_master)
        print(f"      Rows: {total_ma:,}")
        print(f"      Range: {df_master.index.min()} → {df_master.index.max()}")
        print(f"      Volume_Delta nonzero: {nz_ma:,} / {total_ma:,} = {pct_ma:.1f}%")
        gate = "PASS" if pct_ma >= QUALITY_THRESHOLD else "FAIL"
        print(f"      5% gate: {gate}")
        master_exists = True
    else:
        print(f"      NOT FOUND")
        master_exists = False
        pct_ma = 0
    del df_master; gc.collect()

    # ── 4. Raw T&Q files ──
    print(f"\n  [4] Raw data in GCS:")
    n_trades = count_raw_files(ticker, "Trades")
    n_quotes = count_raw_files(ticker, "Quotes")
    print(f"      Trade files: {n_trades}")
    print(f"      Quote files: {n_quotes}")

    # ── Diagnosis ──
    print(f"\n  DIAGNOSIS:")
    if not master_exists:
        action = "BUILD_MASTER"
        if not micro_exists and n_trades == 0:
            print(f"    No master, no micro, no raw T&Q → ticker not in data pipeline")
            action = "NO_DATA"
        elif not micro_exists and n_trades > 0:
            print(f"    No micro but {n_trades} trade files exist → rebuild micro from raw, then build master")
            action = "REBUILD_MICRO_AND_MASTER"
        elif micro_exists:
            print(f"    Micro exists but no master → build master_inference")
            action = "BUILD_MASTER"
    elif pct_ma < QUALITY_THRESHOLD:
        if not micro_exists:
            if n_trades > 0:
                print(f"    Master exists but no micro ({n_trades} raw T&Q files) → rebuild micro, then rebuild master")
                action = "REBUILD_MICRO_AND_MASTER"
            else:
                print(f"    Master exists but no micro and no raw T&Q → can't fix, ticker lacks T&Q data")
                action = "NO_MICRO_DATA"
        elif pct_m >= 30:
            if ohlcv_start_year and ohlcv_start_year < 2020:
                print(f"    OHLCV bloat: starts {ohlcv_start_year}, micro healthy ({pct_m:.0f}% VD) → trim OHLCV to 2022+")
                action = "TRIM_OHLCV"
            else:
                print(f"    Micro healthy ({pct_m:.0f}%) but master diluted ({pct_ma:.1f}%) → rebuild master with trim")
                action = "TRIM_OHLCV"
        else:
            print(f"    Micro itself is low quality ({pct_m:.1f}% VD) → raw T&Q issue, needs investigation")
            action = "LOW_QUALITY_MICRO"
    else:
        print(f"    Master passes 5% gate ({pct_ma:.1f}%) — should be scoring fine")
        action = "OK"

    results.append({
        'ticker': ticker,
        'ohlcv_rows': ohlcv_rows,
        'ohlcv_start': ohlcv_start_year,
        'micro_exists': micro_exists,
        'micro_vd_pct': pct_m,
        'master_vd_pct': pct_ma,
        'raw_trades': n_trades,
        'raw_quotes': n_quotes,
        'action': action,
    })

# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n\n{'=' * 80}")
print(f"  SUMMARY")
print(f"{'=' * 80}")
print(f"\n  {'Ticker':<8} {'OHLCV':>10} {'Start':>6} {'Micro?':>7} {'MicroVD%':>9} {'MasterVD%':>10} {'T&Q':>5} {'Action'}")
print(f"  {'─'*8} {'─'*10} {'─'*6} {'─'*7} {'─'*9} {'─'*10} {'─'*5} {'─'*25}")
for r in results:
    print(f"  {r['ticker']:<8} {r['ohlcv_rows']:>10,} {str(r['ohlcv_start'] or '-'):>6} "
          f"{'YES' if r['micro_exists'] else 'NO':>7} {r['micro_vd_pct']:>8.1f}% {r['master_vd_pct']:>9.1f}% "
          f"{r['raw_trades']:>5} {r['action']}")

# ── Action groups ──
trim_tickers = [r['ticker'] for r in results if r['action'] == 'TRIM_OHLCV']
rebuild_tickers = [r['ticker'] for r in results if r['action'] == 'REBUILD_MICRO_AND_MASTER']
no_data_tickers = [r['ticker'] for r in results if r['action'] in ('NO_DATA', 'NO_MICRO_DATA')]

print(f"\n  ACTION PLAN:")
if trim_tickers:
    print(f"    1. TRIM_OHLCV (rebuild_master_trimmed.py): {trim_tickers}")
    print(f"       → Add to RUN_TICKERS and run")
if rebuild_tickers:
    print(f"    2. REBUILD_MICRO+MASTER: {rebuild_tickers}")
    print(f"       → Run rebuild_micro_incremental.py then rebuild_master_trimmed.py")
if no_data_tickers:
    print(f"    3. NO FIX AVAILABLE: {no_data_tickers}")
    print(f"       → These tickers have no raw T&Q data — can't build micro")

print(f"\n{'=' * 80}")
