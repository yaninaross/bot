"""
rebuild_2_delete_downstream.py — Delete master_inference + scoring CSVs
========================================================================
Run AFTER rebuild_1a and rebuild_1b have both completed.

  1. DELETE MASTER — remove master_inference so daily_ohlcv scripts re-stitch
  2. DELETE SCORING CSVs — remove corrupted model predictions (all 12 experiments)

After this completes:
  1. Run daily_ohlcv_1.py + daily_ohlcv_2.py    → rebuild master_inference
  2. Re-run training scripts (scoring only)      → re-score all experiments
  3. Re-run WF optimizers (v9/v10)               → re-optimize with clean data
  4. Re-run backtests + re-evaluate deploy configs
"""

from google.cloud import storage as gcs_storage

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"

MICRO_TICKERS = [
    'ACHR', 'ALAB', 'APLD', 'ASTS', 'BE',
    'BTBT', 'CIFR', 'CLSK', 'FCEL', 'IREN',
    'JOBY', 'KEEL', 'LUNR', 'OKLO', 'PL',
    'PLUG', 'RCAT', 'RDW', 'RIOT', 'RKLB',
    'SMCI', 'WULF',
]

CONTEXT_ASSETS = ['VIX', 'SPY', 'TLT', 'IWM']

CORRUPTED_EXPERIMENTS = [
    'Universal_Exp152_XSECTOR',
    'Universal_Exp160_XSECTOR',
    'Universal_Exp170_XSECTOR',
    'Universal_Exp171_XSECTOR',
    'Universal_Exp172_XSECTOR',
    'Universal_Exp173_XSECTOR',
    'Universal_Exp174_XSECTOR',
    'Universal_Exp175_XSECTOR',
    'Universal_Exp180_XSECTOR',
    'Universal_Exp185_XSECTOR',
    'Universal_Exp185_MAG_CLS',
    'Universal_Exp185_MAG_REG',
]

gcs_client = gcs_storage.Client()
bucket     = gcs_client.bucket(BUCKET_NAME)

# ==============================================================================
# STEP 1: DELETE master_inference files
# ==============================================================================
print("=" * 60)
print("STEP 1: DELETE master_inference files for scored + context tickers")
print("=" * 60)

master_deleted = 0
for ticker in MICRO_TICKERS:
    clean_ticker = ticker.replace("I:", "").replace("X:", "")
    blob = bucket.blob(f"Data/Inference_v2/{clean_ticker}_master_inference.parquet")
    if blob.exists():
        blob.delete()
        master_deleted += 1
        print(f"  🗑️  Deleted {clean_ticker}_master_inference.parquet")

for ctx in CONTEXT_ASSETS:
    blob = bucket.blob(f"Data/Inference_v2/{ctx}_master_inference.parquet")
    if blob.exists():
        blob.delete()
        master_deleted += 1
        print(f"  🗑️  Deleted {ctx}_master_inference.parquet (context)")

print(f"\n  Deleted {master_deleted} master_inference files.")

# ==============================================================================
# STEP 2: DELETE corrupted scoring CSVs
# ==============================================================================
print(f"\n{'=' * 60}")
print("STEP 2: DELETE corrupted scoring CSVs (all experiments)")
print("=" * 60)

csv_deleted = 0
for exp in CORRUPTED_EXPERIMENTS:
    prefix = f"Algos/Results_all_v2/{exp}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    csvs = [b for b in blobs if b.name.endswith('.csv')]
    for b in csvs:
        b.delete()
        csv_deleted += 1
    print(f"  🗑️  {exp}: deleted {len(csvs)} scoring CSVs")

print(f"\n  Deleted {csv_deleted} scoring CSVs across {len(CORRUPTED_EXPERIMENTS)} experiments.")

print(f"\n{'=' * 60}")
print("CLEANUP COMPLETE")
print("=" * 60)
print("""
Next steps (in order):
  1. Run daily_ohlcv_1.py + daily_ohlcv_2.py    → rebuild master_inference
  2. Re-run training scripts (scoring only):
     - exp_180_walkforward_train.py              → folds 170-175, 180
     - exp_185_train.py                          → fold 185
     - exp 152/160 training scripts              → folds 152, 160
  3. Re-run WF optimizers:
     - optimize_wf_robust.py (v9)
     - optimize_wf_v10.py
  4. Re-run backtests + re-evaluate deploy configs
""")
