"""
check_v2_artifacts.py — Audit what's in v2 folders for models and results
=========================================================================
Run in Colab to see exactly what exists in Results_all_v2/ and Saved_models_v2/
for each experiment.
"""

from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

BUCKET_NAME = "time-series-bucket-ys"
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

EXPERIMENTS = ['152_XSECTOR', '160_XSECTOR', '170_XSECTOR', '171_XSECTOR',
               '172_XSECTOR', '173_XSECTOR', '174_XSECTOR', '175_XSECTOR']

print("=" * 70)
print("V2 ARTIFACT AUDIT")
print("=" * 70)

# ── MODELS (Saved_models_v2) ───────────────────────────────────────────────
print("\n── MODELS (Saved_models_v2/) ──\n")
model_prefix = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction/"
all_models = list(bucket.list_blobs(prefix=model_prefix))
model_names = [b.name.split('/')[-1] for b in all_models if b.name.endswith('.h5')]

for exp_id in EXPERIMENTS:
    exp_models = sorted([m for m in model_names if f'M{exp_id}' in m])
    seeds = [m.split('_S')[1].split('_')[0] for m in exp_models]
    if exp_models:
        print(f"  {exp_id}: {len(exp_models)} models  seeds=[{', '.join(seeds)}]")
    else:
        print(f"  {exp_id}: 0 models")

# ── RESULTS v2 (Results_all_v2) ───────────────────────────────────────────
print("\n── RESULTS CSVs (Results_all_v2/) ──\n")
for exp_id in EXPERIMENTS:
    prefix = f"Algos/Results_all_v2/Universal_Exp{exp_id}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    csvs = [b.name for b in blobs if b.name.endswith('_Results.csv')]
    if not csvs:
        print(f"  {exp_id}: 0 CSVs")
        continue
    seeds = sorted(set(b.split('Seed_')[1].split('_')[0] for b in csvs if 'Seed_' in b))
    tickers = sorted(set(b.split('/')[-2] for b in csvs))
    print(f"  {exp_id}: {len(csvs)} CSVs  seeds=[{', '.join(seeds)}]  tickers={len(tickers)}")

# ── RESULTS v1 (Results_all — old path, for comparison) ──────────────────
print("\n── RESULTS CSVs (Results_all/ — old path) ──\n")
for exp_id in EXPERIMENTS:
    prefix = f"Algos/Results_all/Universal_Exp{exp_id}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    csvs = [b.name for b in blobs if b.name.endswith('_Results.csv')]
    if not csvs:
        print(f"  {exp_id}: 0 CSVs")
        continue
    seeds = sorted(set(b.split('Seed_')[1].split('_')[0] for b in csvs if 'Seed_' in b))
    print(f"  {exp_id}: {len(csvs)} CSVs  seeds=[{', '.join(seeds)}]")

# ── STATS (scaling_stats JSON in Inference_v2) ───────────────────────────
print("\n── SCALING STATS (Data/Inference_v2/) ──\n")
for exp_id in EXPERIMENTS:
    blob_path = f"Data/Inference_v2/scaling_stats_experiment_{exp_id}.json"
    exists = bucket.blob(blob_path).exists()
    print(f"  {exp_id}: {'EXISTS' if exists else 'MISSING'}")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
