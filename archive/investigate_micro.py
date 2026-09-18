"""
investigate_micro.py — Check if microstructure data exists in master_inference
===============================================================================
Investigates why MICRO_COLS (Spread, OBI, Trade_Count, etc.) showed as zeros
in the cluster analysis. Checks:
  1. Raw column names in the parquet (before any renaming)
  2. Whether micro columns exist but under different names
  3. Whether micro data was merged into master_inference or lives separately
  4. Sample values from micro columns if they exist
  5. Checks the separate micro parquet files

Usage:  python investigate_micro.py
"""

import io
import pandas as pd
from google.cloud import storage

BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference"
TEST_TICKERS = ['RIOT', 'CLSK', 'IREN']

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

print("=" * 80)
print("  MICROSTRUCTURE DATA INVESTIGATION")
print("=" * 80)

# ==============================================================================
# 1. CHECK master_inference PARQUET — raw columns before any renaming
# ==============================================================================
for ticker in TEST_TICKERS:
    print(f"\n{'─' * 60}")
    print(f"  {ticker}_master_inference.parquet")
    print(f"{'─' * 60}")

    blob = bucket.blob(f"{INFERENCE_DIR}/{ticker}_master_inference.parquet")
    if not blob.exists():
        print(f"   FILE NOT FOUND")
        continue

    df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))

    print(f"\n   Shape: {df.shape}")
    print(f"   Index: {df.index.name} (dtype={df.index.dtype})")
    print(f"   Date range: {df.index.min()} → {df.index.max()}")
    print(f"\n   ALL COLUMNS ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"      [{i:2d}] {col!r:30s}  dtype={df[col].dtype}")

    # Check for micro-like columns (fuzzy match)
    print(f"\n   MICRO COLUMN SEARCH:")
    micro_lower = {m.lower(): m for m in MICRO_COLS}
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if col_clean in micro_lower:
            vals = df[col]
            nonzero = (vals != 0).sum()
            notnull = vals.notna().sum()
            sample = vals[vals != 0].head(5).tolist() if nonzero > 0 else []
            print(f"      FOUND: {col!r} → maps to {micro_lower[col_clean]}")
            print(f"         nonzero: {nonzero:,} / {len(df):,} ({nonzero/len(df)*100:.1f}%)")
            print(f"         notnull: {notnull:,} / {len(df):,}")
            print(f"         sample nonzero values: {sample}")
        elif any(m.lower() in col_clean for m in MICRO_COLS):
            print(f"      PARTIAL MATCH: {col!r}")
            vals = df[col]
            nonzero = (vals != 0).sum() if pd.api.types.is_numeric_dtype(vals) else 'N/A'
            print(f"         nonzero: {nonzero}")

    # Check which columns have actual data
    print(f"\n   COLUMNS WITH DATA (nonzero count):")
    for col in df.columns:
        try:
            nz = (df[col] != 0).sum()
            nn = df[col].notna().sum()
            pct = nz / len(df) * 100
            if pct > 0.1:
                print(f"      {str(col):25s}  nonzero={nz:>10,} ({pct:5.1f}%)  notnull={nn:>10,}")
        except:
            pass

# ==============================================================================
# 2. CHECK for separate MICRO parquet files
# ==============================================================================
print(f"\n\n{'=' * 80}")
print(f"  CHECKING FOR SEPARATE MICRO FILES")
print(f"{'=' * 80}")

for ticker in TEST_TICKERS:
    print(f"\n   {ticker}:")
    patterns = [
        f"{INFERENCE_DIR}/{ticker}_micro.parquet",
        f"{INFERENCE_DIR}/{ticker}_micro_data.parquet",
        f"{INFERENCE_DIR}/{ticker}_1m_micro.parquet",
        f"Data/Micro/{ticker}_micro.parquet",
        f"Data/Micro/{ticker}.parquet",
    ]
    for path in patterns:
        blob = bucket.blob(path)
        exists = blob.exists()
        print(f"      {path}: {'EXISTS' if exists else 'not found'}")
        if exists:
            df_micro = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
            print(f"         Shape: {df_micro.shape}")
            print(f"         Columns: {list(df_micro.columns)}")
            print(f"         Date range: {df_micro.index.min()} → {df_micro.index.max()}")
            sample = df_micro.head(3)
            print(f"         Sample:\n{sample.to_string()}")

# ==============================================================================
# 3. LIST all files in Data/Inference to find micro-related files
# ==============================================================================
print(f"\n\n{'=' * 80}")
print(f"  ALL FILES IN {INFERENCE_DIR}/ (looking for micro patterns)")
print(f"{'=' * 80}")

blobs = list(bucket.list_blobs(prefix=INFERENCE_DIR + "/"))
print(f"\n   Total files: {len(blobs)}")

micro_files = [b.name for b in blobs if 'micro' in b.name.lower()]
print(f"   Files containing 'micro': {len(micro_files)}")
for f in sorted(micro_files):
    print(f"      {f}")

# Also check for any non-master, non-scaling files
other_files = [b.name for b in blobs
               if 'master_inference' not in b.name
               and 'scaling_stats' not in b.name
               and not b.name.endswith('/')]
print(f"\n   Non-master, non-scaling files: {len(other_files)}")
for f in sorted(other_files):
    print(f"      {f}")

# ==============================================================================
# 4. CHECK daily_micro_all.py output location
# ==============================================================================
print(f"\n\n{'=' * 80}")
print(f"  CHECKING daily_micro_all.py OUTPUT PATHS")
print(f"{'=' * 80}")

# Check Data/Micro/ directory
micro_blobs = list(bucket.list_blobs(prefix="Data/Micro"))
print(f"\n   Files in Data/Micro/: {len(micro_blobs)}")
for b in sorted(micro_blobs, key=lambda x: x.name)[:20]:
    print(f"      {b.name}  ({b.size:,} bytes)")

# Check if micro data goes into a different bucket path
for prefix in ["Data/micro", "data/micro", "Data/Microstructure", "Micro"]:
    blobs_check = list(bucket.list_blobs(prefix=prefix, max_results=5))
    if blobs_check:
        print(f"\n   Found files under {prefix}/:")
        for b in blobs_check:
            print(f"      {b.name}")

print(f"\n\n{'=' * 80}")
print(f"  INVESTIGATION COMPLETE")
print(f"{'=' * 80}")
