"""
check_training_artifacts.py — Check whether Exp-110 and Exp-104 training completed.
Checks GCS for models, stats JSON, and result CSVs for each seed × ticker.
"""

from google.cloud import storage

BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
MODEL_PATH  = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR   = "Data/Inference"

EXPERIMENTS = {
    "110_MICRO": {
        "seeds": list(range(11001, 11016)),
        "scoring_pool": [
            'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
            'COIN', 'BITF', 'HUT', 'CIFR', 'BTBT', 'HOOD',
            'SMCI', 'NVDA', 'VRT', 'AMD', 'PLTR', 'TSLA',
            'SMH', 'TLT', 'IWM', 'VIX',
        ],
    },
    "104_DEFENSE": {
        "seeds": list(range(10401, 10416)),
        "scoring_pool": [
            'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS', 'LUNR', 'SPIR',
            'JOBY', 'NNDM', 'MVIS',
            'GD', 'LMT', 'NOC', 'RTX', 'LHX', 'GE', 'RCAT', 'ACHR',
        ],
    },
}


def blob_exists(bucket, path):
    return bucket.blob(path).exists()


def check_experiment(bucket, model_id, cfg):
    seeds = cfg["seeds"]
    tickers = cfg["scoring_pool"]
    results_base = f"Algos/Results_all_v2/Universal_Exp{model_id}"

    print(f"\n{'='*65}")
    print(f"  Experiment {model_id}")
    print(f"{'='*65}")

    # --- Stats JSON ---
    stats_path = f"{STATS_DIR}/scaling_stats_experiment_{model_id}.json"
    stats_ok = blob_exists(bucket, stats_path)
    print(f"\n  Stats JSON: {'FOUND' if stats_ok else 'MISSING'}")
    print(f"    {stats_path}")

    # --- Models ---
    print(f"\n  Models (expect {len(seeds)}):")
    models_found = []
    models_missing = []
    for seed in seeds:
        p = f"{MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
        if blob_exists(bucket, p):
            models_found.append(seed)
        else:
            models_missing.append(seed)

    print(f"    Found:   {len(models_found)}/{len(seeds)}", end="")
    if models_found:
        print(f"  [{models_found[0]}..{models_found[-1]}]")
    else:
        print()
    if models_missing:
        print(f"    Missing: {models_missing}")

    # --- Result CSVs ---
    total_expected = len(seeds) * len(tickers)
    print(f"\n  Result CSVs (expect {len(seeds)} seeds × {len(tickers)} tickers = {total_expected}):")
    found_count = 0
    missing_by_seed = {}
    for seed in seeds:
        missing_tickers = []
        for ticker in tickers:
            p = f"{results_base}/{ticker}/Seed_{seed}_Results.csv"
            if blob_exists(bucket, p):
                found_count += 1
            else:
                missing_tickers.append(ticker)
        if missing_tickers:
            missing_by_seed[seed] = missing_tickers

    print(f"    Found: {found_count}/{total_expected}")

    if not missing_by_seed:
        print(f"    All CSVs present.")
    else:
        completed_seeds = [s for s in seeds if s not in missing_by_seed]
        partial_seeds = {s: t for s, t in missing_by_seed.items()
                         if len(t) < len(tickers)}
        empty_seeds = [s for s, t in missing_by_seed.items()
                       if len(t) == len(tickers)]

        if completed_seeds:
            print(f"    Fully complete seeds: {completed_seeds}")
        if partial_seeds:
            for s, t in partial_seeds.items():
                print(f"    Seed {s}: missing {len(t)} tickers — {t}")
        if empty_seeds:
            print(f"    No results at all:   {empty_seeds}")

    # --- Summary ---
    complete = stats_ok and not models_missing and not missing_by_seed
    if complete:
        print(f"\n  >>> TRAINING COMPLETE <<<")
    else:
        last_done = models_found[-1] if models_found else None
        if last_done and last_done < seeds[-1]:
            print(f"\n  >>> INCOMPLETE — last finished seed: {last_done}, "
                  f"stopped before {last_done + 1} <<<")
        elif not models_found:
            print(f"\n  >>> NOT STARTED (no models found) <<<")
        else:
            print(f"\n  >>> PARTIALLY COMPLETE — check missing items above <<<")


def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    for model_id, cfg in EXPERIMENTS.items():
        check_experiment(bucket, model_id, cfg)

    print()


if __name__ == "__main__":
    main()
