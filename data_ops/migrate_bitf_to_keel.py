"""
migrate_bitf_to_keel.py — Copy & rename all BITF data files to KEEL in GCS
===========================================================================
BITF (Bitfarms Ltd) was renamed to KEEL on Nasdaq/TSX effective 2026-04-06.
The Exp-97 model was trained on BITF (embedding index 9). Going forward,
Polygon serves data under KEEL. This script copies all historical BITF files
to KEEL so the pipeline can use KEEL as the single ticker going forward.

What it copies:
  1. Raw T&Q:  Data/TradesQuotes/BITF/BITF_{trades|quotes}_{date}.parquet
            →  Data/TradesQuotes/KEEL/KEEL_{trades|quotes}_{date}.parquet
  2. OHLCV:    Data/Inference/BITF_1m_full_historical_data.parquet
            →  Data/Inference/KEEL_1m_full_historical_data.parquet

Does NOT copy derived files (micro, master_inference) — those will be
rebuilt from the copied raw files.

Usage:
  python migrate_bitf_to_keel.py          # dry run — list what would be copied
  python migrate_bitf_to_keel.py --copy   # actually copy files
"""

import sys
from google.cloud import storage

BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
SRC_TICKER  = "BITF"
DST_TICKER  = "KEEL"


def main(copy=False):
    dry_run = not copy and "--copy" not in sys.argv
    if dry_run:
        print("🔍 DRY RUN — listing files that would be copied")
        print("   Run with --copy to actually copy them\n")
    else:
        print(f"📋 COPY MODE — copying {SRC_TICKER} → {DST_TICKER}\n")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    copied, skipped, errors = 0, 0, 0

    # ── 1. Raw T&Q files ────────────────────────────────────────────────────
    print("=" * 65)
    print(f"📁 Raw T&Q: Data/TradesQuotes/{SRC_TICKER}/ → Data/TradesQuotes/{DST_TICKER}/")
    print("=" * 65)

    src_prefix = f"Data/TradesQuotes/{SRC_TICKER}/"
    src_blobs = list(client.list_blobs(BUCKET_NAME, prefix=src_prefix))

    if not src_blobs:
        print(f"   ⚠️  No files found under {src_prefix}")
    else:
        print(f"   Found {len(src_blobs)} files in {SRC_TICKER} folder\n")

    for blob in src_blobs:
        src_name = blob.name
        fname = src_name.split("/")[-1]

        # Only copy properly named trade/quote files
        if not fname.endswith(".parquet"):
            continue
        if not (fname.startswith(f"{SRC_TICKER}_trades_") or
                fname.startswith(f"{SRC_TICKER}_quotes_")):
            print(f"   ⏭️  Skipping non-standard file: {fname}")
            continue

        # Rename: BITF_trades_2024-01-15.parquet → KEEL_trades_2024-01-15.parquet
        dst_fname = fname.replace(f"{SRC_TICKER}_", f"{DST_TICKER}_", 1)
        dst_name  = f"Data/TradesQuotes/{DST_TICKER}/{dst_fname}"

        # Skip if destination already has real data (>2KB)
        dst_blob = bucket.blob(dst_name)
        if dst_blob.exists():
            dst_blob.reload()
            if dst_blob.size > 2000:
                skipped += 1
                continue

        if dry_run:
            print(f"   📄 {fname} → {dst_fname}")
            copied += 1
        else:
            try:
                bucket.copy_blob(blob, bucket, dst_name)
                copied += 1
                if copied % 50 == 0:
                    print(f"   ⏳ Copied {copied} files...")
            except Exception as e:
                print(f"   ❌ Failed to copy {fname}: {e}")
                errors += 1

    print(f"\n   T&Q: {'would copy' if dry_run else 'copied'} {copied}, "
          f"skipped {skipped} (already exist), {errors} errors")

    # ── 2. OHLCV file ───────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"📁 OHLCV: Data/Inference/")
    print("=" * 65)

    ohlcv_src = f"Data/Inference/{SRC_TICKER}_1m_full_historical_data.parquet"
    ohlcv_dst = f"Data/Inference/{DST_TICKER}_1m_full_historical_data.parquet"

    src_blob = bucket.blob(ohlcv_src)
    dst_blob = bucket.blob(ohlcv_dst)

    if not src_blob.exists():
        print(f"   ⚠️  Source not found: {ohlcv_src}")
    elif dst_blob.exists():
        print(f"   ⏭️  Already exists: {ohlcv_dst}")
    elif dry_run:
        print(f"   📄 {ohlcv_src}")
        print(f"    → {ohlcv_dst}")
    else:
        try:
            bucket.copy_blob(src_blob, bucket, ohlcv_dst)
            print(f"   ✅ Copied: {ohlcv_dst}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    if dry_run:
        print(f"⚠️  DRY RUN complete. Run with --copy to execute.")
    else:
        print(f"✅ Migration complete: {SRC_TICKER} → {DST_TICKER}")
        print(f"   Next steps:")
        print(f"   1. Remove BITF from daily_micro_all.py TARGET_TICKERS")
        print(f"   2. Rebuild microstructure: run daily_micro_all.py")
        print(f"   3. Rebuild master_inference: run daily_data_all.py")
    print("=" * 65)


# ==============================================================================
# RUN CONFIG — edit before running the cell
# ==============================================================================
COPY_MODE = True   # False = dry run, True = actually copy

main(copy=COPY_MODE)
