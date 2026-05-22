"""
delete_derived_parquets.py — Clean up derived parquet files
============================================================
Finds and deletes microstructure + master_inference parquets
so they can be rebuilt from fresh raw OHLCV and T&Q data.

Keeps:
  - Raw OHLCV:  Data/Inference/{ticker}_1m_full_historical_data.parquet
  - Raw T&Q:    Data/TradesQuotes/{ticker}/{ticker}_trades_{date}.parquet
                Data/TradesQuotes/{ticker}/{ticker}_quotes_{date}.parquet

Deletes:
  - Micro:      Data/Inference/Microstructure/{ticker}_1m_microstructure.parquet
  - Master:     Data/Inference/{ticker}_master_inference.parquet

Usage:
  python delete_derived_parquets.py          # dry run — list only
  python delete_derived_parquets.py --delete  # actually delete
"""

import sys
from google.cloud import storage

BUCKET_NAME = "time-series-bucket-ys"

ALL_TQ_ASSETS = [
    'ACHR', 'AMD', 'APLD', 'ASTS', 'AVAV', 'BBAI', 'BITF', 'BTBT',
    'CIFR', 'CLSK', 'COIN', 'CORZ', 'HOOD', 'HUT', 'IREN', 'IWM',
    'JOBY', 'KEEL', 'KTOS', 'LUNR', 'MARA', 'MIGI', 'MSTR', 'MVIS', 'NNDM',
    'NVDA', 'PLTR', 'RCAT', 'RIOT', 'RKLB', 'SMLR', 'SMCI', 'SMH',
    'SPIR', 'TLT', 'TSLA', 'VRT', 'WULF',
]

# Context assets that also have derived files
CONTEXT_ASSETS = ['ITA', 'MOD', 'SPY', 'VIX', 'BTCUSD']

ALL_ASSETS = sorted(set(ALL_TQ_ASSETS + CONTEXT_ASSETS))

def main():
    dry_run = "--delete" not in sys.argv
    if dry_run:
        print("🔍 DRY RUN — listing files that would be deleted")
        print("   Run with --delete to actually remove them\n")
    else:
        print("🗑️  DELETE MODE — files will be permanently removed\n")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    micro_found, master_found = [], []

    # --- Scan for microstructure parquets ---
    print("=" * 60)
    print("📁 Microstructure parquets (Data/Inference/Microstructure/)")
    print("=" * 60)
    for ticker in ALL_ASSETS:
        blob_path = f"Data/Inference/Microstructure/{ticker}_1m_microstructure.parquet"
        blob = bucket.blob(blob_path)
        if blob.exists():
            micro_found.append(blob)
            print(f"   {'🗑️' if not dry_run else '📄'}  {blob_path}")

    # --- Scan for master_inference parquets ---
    print(f"\n{'=' * 60}")
    print("📁 Master inference parquets (Data/Inference/)")
    print("=" * 60)
    for ticker in ALL_ASSETS:
        blob_path = f"Data/Inference/{ticker}_master_inference.parquet"
        blob = bucket.blob(blob_path)
        if blob.exists():
            master_found.append(blob)
            print(f"   {'🗑️' if not dry_run else '📄'}  {blob_path}")

    # --- Summary ---
    total = len(micro_found) + len(master_found)
    print(f"\n{'=' * 60}")
    print(f"   Microstructure files: {len(micro_found)}")
    print(f"   Master inference files: {len(master_found)}")
    print(f"   Total: {total}")
    print("=" * 60)

    if total == 0:
        print("\n✅ Nothing to delete.")
        return

    if dry_run:
        print(f"\n⚠️  Run with --delete to remove these {total} files")
        return

    # --- Delete ---
    print(f"\n🗑️  Deleting {total} files...")
    deleted = 0
    for blob in micro_found + master_found:
        try:
            blob.delete()
            deleted += 1
        except Exception as e:
            print(f"   ❌ Failed to delete {blob.name}: {e}")

    print(f"\n✅ Deleted {deleted}/{total} files.")


if __name__ == "__main__":
    main()
