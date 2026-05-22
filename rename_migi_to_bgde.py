"""Rename MIGI → BGDE on GCS (3 files: OHLCV, micro, master_inference)."""
from google.cloud import storage

client = storage.Client()
bucket = client.bucket("time-series-bucket-ys")

renames = [
    ("Data/Inference/MIGI_1m_full_historical_data.parquet",
     "Data/Inference/BGDE_1m_full_historical_data.parquet"),
    ("Data/Inference/Microstructure/MIGI_1m_microstructure.parquet",
     "Data/Inference/Microstructure/BGDE_1m_microstructure.parquet"),
    ("Data/Inference/MIGI_master_inference.parquet",
     "Data/Inference/BGDE_master_inference.parquet"),
]

for old_path, new_path in renames:
    old_blob = bucket.blob(old_path)
    if not old_blob.exists():
        print(f"  SKIP — {old_path} not found")
        continue
    bucket.copy_blob(old_blob, bucket, new_path)
    old_blob.delete()
    print(f"  DONE — {old_path} → {new_path}")

print("\nAll MIGI → BGDE renames complete.")
