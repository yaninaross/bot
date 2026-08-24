"""
audit_scored_vs_micro.py — Detect which scored CSVs were created from incomplete master files
==============================================================================================
For each ticker with scored CSVs, compares:
  1. Scored CSV creation time (GCS blob metadata)
  2. Microstructure file last-updated time
  3. Master_inference file last-updated time

Logic:
  - If micro file was updated AFTER the CSV was created → CSV was scored with stale/incomplete data
  - If micro file doesn't exist → CSV was definitely scored without micro
  - If micro file was updated BEFORE the CSV → CSV was scored with current micro (OK)

Also checks current master quality (Volume_Delta non-zero %) for cross-reference.
"""

import pandas as pd
from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

client = storage.Client()
bucket = client.bucket("time-series-bucket-ys")

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
CONTEXT_SKIP = {'SPY', 'VIX', 'TLT', 'IWM', 'SMH', 'BTCUSD', 'ITA', 'XLE'}

EXPERIMENTS = {
    '160_XSECTOR': {
        'results_prefix': 'Algos/Results_all_v2/Universal_Exp160_XSECTOR',
        'seeds': list(range(16001, 16016)),
    },
    '152_XSECTOR': {
        'results_prefix': 'Algos/Results_all_v2/Universal_Exp152_XSECTOR',
        'seeds': list(range(15201, 15216)),
    },
}

MICRO_DIR = 'Data/Inference/Microstructure'
MASTER_DIR = 'Data/Inference_v2'
MICRO_CHECK_COL = 'Volume_Delta'
MICRO_MIN_PCT = 5.0

# ==============================================================================
# Discover all tickers from master files
# ==============================================================================
print("Scanning GCS for master_inference files...")
master_blobs = list(bucket.list_blobs(prefix=f'{MASTER_DIR}/'))
all_tickers = []
for b in master_blobs:
    if b.name.endswith('_master_inference.parquet'):
        ticker = b.name.replace(f'{MASTER_DIR}/', '').replace('_master_inference.parquet', '')
        if ticker and ticker not in CONTEXT_SKIP:
            all_tickers.append(ticker)
all_tickers = sorted(set(all_tickers))
print(f"Found {len(all_tickers)} tickers\n")

# ==============================================================================
# For each ticker: get timestamps and quality
# ==============================================================================
results = []

for ticker in all_tickers:
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)

    # --- Master file timestamp ---
    master_blob = bucket.blob(f"{MASTER_DIR}/{file_ticker}_master_inference.parquet")
    master_blob.reload()
    master_updated = master_blob.updated

    # --- Micro file timestamp ---
    micro_blob = bucket.blob(f"{MICRO_DIR}/{file_ticker}_1m_microstructure.parquet")
    micro_exists = micro_blob.exists()
    micro_updated = None
    if micro_exists:
        micro_blob.reload()
        micro_updated = micro_blob.updated

    # --- Current micro quality in master ---
    uri = f"gs://time-series-bucket-ys/{MASTER_DIR}/{file_ticker}_master_inference.parquet"
    try:
        df = pd.read_parquet(uri, columns=[MICRO_CHECK_COL])
        vd_pct = (df[MICRO_CHECK_COL] != 0).sum() / len(df) * 100 if len(df) > 0 else 0
    except Exception:
        vd_pct = 0

    has_real_micro = vd_pct >= MICRO_MIN_PCT

    # --- Scored CSV timestamps (check first seed that exists for each experiment) ---
    for exp_id, exp_cfg in EXPERIMENTS.items():
        csv_blob = None
        csv_created = None

        for seed in exp_cfg['seeds']:
            b = bucket.blob(f"{exp_cfg['results_prefix']}/{ticker}/Seed_{seed}_Results.csv")
            if b.exists():
                b.reload()
                csv_blob = b
                csv_created = b.time_created
                break

        if csv_created is None:
            continue

        # --- Diagnosis ---
        if not micro_exists:
            diagnosis = "BAD — no micro file exists"
        elif micro_updated > csv_created:
            diagnosis = f"STALE — micro updated {micro_updated.strftime('%Y-%m-%d %H:%M')} AFTER csv created {csv_created.strftime('%Y-%m-%d %H:%M')}"
        else:
            diagnosis = "OK — csv scored after micro was ready"

        if not has_real_micro:
            master_status = f"INCOMPLETE (VD={vd_pct:.1f}%)"
        else:
            master_status = f"COMPLETE (VD={vd_pct:.1f}%)"

        results.append({
            'Ticker': ticker,
            'Experiment': exp_id,
            'CSV_Created': csv_created,
            'Micro_Updated': micro_updated,
            'Master_Updated': master_updated,
            'Master_Status': master_status,
            'Diagnosis': diagnosis,
            'VD_Pct': round(vd_pct, 1),
        })

        print(f"  {ticker:8s} {exp_id:14s}  csv={csv_created.strftime('%m-%d %H:%M')}  "
              f"micro={'NONE' if not micro_exists else micro_updated.strftime('%m-%d %H:%M'):>11s}  "
              f"master={master_updated.strftime('%m-%d %H:%M')}  "
              f"VD={vd_pct:5.1f}%  → {diagnosis.split('—')[0].strip()}")

# ==============================================================================
# Summary
# ==============================================================================
df_r = pd.DataFrame(results)

if df_r.empty:
    print("\nNo scored CSVs found.")
else:
    print(f"\n{'='*80}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*80}")

    for exp_id in EXPERIMENTS:
        exp_df = df_r[df_r['Experiment'] == exp_id]
        if exp_df.empty:
            continue

        print(f"\n--- {exp_id} ---")

        stale = exp_df[exp_df['Diagnosis'].str.startswith('STALE')]
        bad = exp_df[exp_df['Diagnosis'].str.startswith('BAD')]
        ok = exp_df[exp_df['Diagnosis'].str.startswith('OK')]

        print(f"  OK (scored with current micro):  {len(ok)}")
        print(f"  STALE (micro updated after CSV): {len(stale)}")
        print(f"  BAD (no micro file at all):      {len(bad)}")

        if len(stale) > 0:
            print(f"\n  STALE tickers (NEED RESCORE):")
            for _, r in stale.iterrows():
                print(f"    {r['Ticker']:8s}  csv={r['CSV_Created'].strftime('%Y-%m-%d %H:%M')}  "
                      f"micro={r['Micro_Updated'].strftime('%Y-%m-%d %H:%M')}  {r['Master_Status']}")

        if len(bad) > 0:
            print(f"\n  BAD tickers (no micro file — need rebuild_micro first):")
            for _, r in bad.iterrows():
                print(f"    {r['Ticker']:8s}  {r['Master_Status']}")

    print(f"\n{'='*80}")
    total_rescore = len(df_r[df_r['Diagnosis'].str.startswith('STALE')]) + \
                    len(df_r[df_r['Diagnosis'].str.startswith('BAD')])
    print(f"TOTAL needing rescore: {total_rescore} / {len(df_r)} ticker-experiment pairs")
    print(f"{'='*80}")
