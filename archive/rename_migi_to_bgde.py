"""Rename MIGI → BGDE on GCS (3 files: OHLCV, micro, master_inference)."""
from google.cloud import storage

client = storage.Client()
bucket = client.bucket(os.environ.get("GCS_BUCKET", ""))

renames = [
    ("Data/Inference_v2/MIGI_1m_full_historical_data.parquet",
     "Data/Inference_v2/BGDE_1m_full_historical_data.parquet"),
    ("Data/Inference_v2/Microstructure/MIGI_1m_microstructure.parquet",
     "Data/Inference_v2/Microstructure/BGDE_1m_microstructure.parquet"),
    ("Data/Inference_v2/MIGI_master_inference.parquet",
     "Data/Inference_v2/BGDE_master_inference.parquet"),
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
