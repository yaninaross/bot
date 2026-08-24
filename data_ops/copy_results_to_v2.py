"""
copy_results_to_v2.py — Copy scoring results from Results_all/ to Results_all_v2/
==================================================================================
Two modes:
  1. CLEANUP mode (CLEANUP_FIRST = True): deletes ALL files in Results_all_v2/ for
     the listed experiments, then copies only the specified seeds.
  2. Copy-only mode (CLEANUP_FIRST = False): copies specified seeds, skips existing.

Seed filtering ensures only newly-trained v2 results enter the v2 folder.

Run in Colab: paste as cell, runs immediately.
"""

from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

BUCKET_NAME = "time-series-bucket-ys"

# ── CONFIG ──────────────────────────────────────────────────────────────────
CLEANUP_FIRST = False

EXPERIMENTS_TO_COPY = {
    '152_XSECTOR': list(range(15201, 15216)),   # all 15 seeds
}

EXPERIMENTS_TO_CLEANUP = []
# ── END CONFIG ──────────────────────────────────────────────────────────────

SRC_PREFIX = "Algos/Results_all/Universal_Exp"
DST_PREFIX = "Algos/Results_all_v2/Universal_Exp"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# ── STEP 1: Cleanup v2 folder ──────────────────────────────────────────────
if CLEANUP_FIRST:
    print("=== CLEANUP: deleting old files from Results_all_v2/ ===\n")
    for exp_id in EXPERIMENTS_TO_CLEANUP:
        dst_dir = f"{DST_PREFIX}{exp_id}/"
        blobs = list(bucket.list_blobs(prefix=dst_dir))
        if not blobs:
            print(f"  {exp_id}: nothing to delete")
            continue
        for blob in blobs:
            blob.delete()
        print(f"  {exp_id}: deleted {len(blobs)} files")
    print()

# ── STEP 2: Copy only specified seeds ──────────────────────────────────────
print("=== COPY: copying v2 seeds from Results_all/ to Results_all_v2/ ===\n")

total_copied = 0
total_skipped = 0
total_filtered = 0

for exp_id, valid_seeds in EXPERIMENTS_TO_COPY.items():
    seed_strs = {str(s) for s in valid_seeds}
    src_dir = f"{SRC_PREFIX}{exp_id}/"
    dst_dir = f"{DST_PREFIX}{exp_id}/"

    blobs = list(bucket.list_blobs(prefix=src_dir))
    csvs = [b for b in blobs if b.name.endswith('_Results.csv')]

    if not csvs:
        print(f"  {exp_id}: no results found in {src_dir}")
        continue

    copied = 0
    skipped = 0
    filtered = 0
    for blob in csvs:
        fname = blob.name.split('/')[-1]  # e.g. LUNR_17001_Results.csv
        parts = fname.replace('_Results.csv', '').split('_')
        seed = parts[-1] if len(parts) >= 2 else ''

        if seed not in seed_strs:
            filtered += 1
            continue

        dst_name = blob.name.replace(SRC_PREFIX, DST_PREFIX)
        dst_blob = bucket.blob(dst_name)
        if dst_blob.exists():
            skipped += 1
            continue
        bucket.copy_blob(blob, bucket, dst_name)
        copied += 1

    print(f"  {exp_id}: {copied} copied, {skipped} existed, {filtered} filtered out "
          f"(seeds {min(valid_seeds)}-{max(valid_seeds)})")
    total_copied += copied
    total_skipped += skipped
    total_filtered += filtered

print(f"\nDone: {total_copied} copied, {total_skipped} skipped, {total_filtered} filtered out")
