"""
score_exp110_batch.py — Batch re-scoring for Exp-110 (all seeds, all tickers)
==============================================================================
Standalone script for scoring historical data with saved Exp-110 models.
Runs on CPU — no GPU required.

Produces result CSVs in the same format as the training pipeline,
compatible with optimize_exp110_v2.py.

Usage:
    python score_exp110_batch.py                            # Score Jul 2025 - Apr 2026
    python score_exp110_batch.py --start 2026-01-01         # Score Jan 2026 onwards
    python score_exp110_batch.py --tickers RIOT CLSK IREN   # Score specific tickers
    python score_exp110_batch.py --seeds 11001 11005        # Score specific seeds
"""

import os
import sys
import json
import argparse
import gc
import tempfile
import time

import numpy as np
import pandas as pd
import pytz

try:
    import pandas_ta as ta
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

import tensorflow as tf
from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils_110 import (
    load_master, heal_data, build_market_context, engineer_features,
    apply_scaling, build_model, build_context_cols,
    ASSET_COLS, TIME_STEPS, PST,
)

# ==============================================================================
# CONFIG — must match exp_110.py exactly
# ==============================================================================
MODEL_ID = "110_MICRO"
SEEDS = list(range(11001, 11016))

TRAIN_TEACHERS = [
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
    'COIN', 'BITF', 'HUT', 'CIFR', 'BTBT', 'HOOD',
]

SCORING_POOL = TRAIN_TEACHERS + [
    'SMCI', 'NVDA', 'VRT', 'AMD', 'PLTR', 'TSLA',
    'SMH', 'TLT', 'IWM', 'VIX',
]

CONTEXT_ASSETS = ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD']
HAS_SMH_RSI    = True

TICKER_START_DATES = {'CORZ': '2024-01-25'}
DATA_START_DATE    = "2022-01-01"

BUCKET_NAME      = "time-series-bucket-ys"
INFERENCE_DIR    = "Data/Inference"
GCS_MODEL_PATH   = "Algos/to_deploy/Saved_models/UNIVERSAL/Direction"
GCS_RESULTS_PATH = f"Algos/Results_all/Universal_Exp{MODEL_ID}"
STATS_PATH       = f"Data/Inference/scaling_stats_experiment_{MODEL_ID}.json"

SPIKE_THRESHOLD = 0.10

ASSET_INDEX = {t: i for i, t in enumerate(TRAIN_TEACHERS)}
N_ASSETS    = len(TRAIN_TEACHERS) + 1
OTHER_IDX   = len(TRAIN_TEACHERS)

CONTEXT_COLS = build_context_cols(CONTEXT_ASSETS, HAS_SMH_RSI)
ALL_COLS     = ASSET_COLS + CONTEXT_COLS

SCORE_BATCH_SIZE = 4096


# ==============================================================================
# VALID_ROW + TARGET (matches training's add_targets)
# ==============================================================================
def add_scoring_metadata(df):
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute

    df['Valid_Row'] = ((mins >= 360) & (mins <= 390)).astype(int)

    is_safe = (df_pst.hour < 13) | ((df_pst.hour == 13) & (df_pst.minute == 0))
    is_real = df['Volume'] > 0
    df['Safe_High'] = np.where(is_safe & is_real, df['High'], -1.0)
    df['Future_Max'] = df.groupby(df.index.date)['Safe_High'].transform(
        lambda x: x[::-1].cummax()[::-1])
    df['Target'] = (df['Future_Max'] > (df['Close'] * (1 + SPIKE_THRESHOLD))).astype(int)
    df.drop(columns=['Safe_High', 'Future_Max'], inplace=True, errors='ignore')
    return df


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Batch scoring for Exp-110')
    parser.add_argument('--start', default='2025-07-01',
                        help='Score start date (default: 2025-07-01)')
    parser.add_argument('--end', default='2026-04-18',
                        help='Score end date (default: 2026-04-18)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Specific seeds to score (default: all 15)')
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Specific tickers to score (default: full SCORING_POOL)')
    parser.add_argument('--check-only', action='store_true',
                        help='Check existing CSV date ranges without scoring')
    args = parser.parse_args()

    seeds   = args.seeds or SEEDS
    tickers = args.tickers or SCORING_POOL
    score_start = pd.to_datetime(args.start).tz_localize('UTC')
    score_end   = pd.to_datetime(args.end).tz_localize('UTC') + pd.Timedelta(days=1)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # ------------------------------------------------------------------
    # CHECK-ONLY MODE: inspect existing CSVs
    # ------------------------------------------------------------------
    if args.check_only:
        import io
        print("Checking existing result CSVs...")
        for ticker in tickers[:5]:
            path = f"{GCS_RESULTS_PATH}/{ticker}/Seed_{seeds[0]}_Results.csv"
            blob = bucket.blob(path)
            if not blob.exists():
                print(f"  {ticker}/Seed_{seeds[0]}: NOT FOUND")
                continue
            df = pd.read_csv(io.BytesIO(blob.download_as_bytes()),
                             index_col=0, parse_dates=True)
            print(f"  {ticker}/Seed_{seeds[0]}: {len(df)} rows, "
                  f"{df.index.min()} -> {df.index.max()}")
        return

    print("=" * 80)
    print(f"  EXP-110 BATCH SCORING")
    print(f"  Model: {MODEL_ID}  |  Seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)})")
    print(f"  Tickers: {len(tickers)}  |  Range: {args.start} -> {args.end}")
    print(f"  GPU: {'YES' if tf.config.list_physical_devices('GPU') else 'NO (CPU)'}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. LOAD SCALING STATS
    # ------------------------------------------------------------------
    print("\n[1/4] Loading scaling stats from GCS...")
    stats = json.loads(bucket.blob(STATS_PATH).download_as_bytes())
    print(f"   {len(stats)} feature stats loaded")

    # ------------------------------------------------------------------
    # 2. BUILD MARKET CONTEXT
    # ------------------------------------------------------------------
    print("\n[2/4] Building market context (full history)...")
    t0 = time.time()
    market_df = build_market_context(
        CONTEXT_ASSETS, HAS_SMH_RSI, BUCKET_NAME, INFERENCE_DIR, tail_days=None)
    print(f"   {len(market_df):,} rows  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # 3. PREPARE TICKER DATA
    # ------------------------------------------------------------------
    print(f"\n[3/4] Preparing {len(tickers)} tickers...")
    ticker_data = {}
    for ticker in tickers:
        t0 = time.time()
        df = load_master(ticker, BUCKET_NAME, INFERENCE_DIR,
                         tail_days=None, add_micro_zeros=True)
        if df is None:
            print(f"   {ticker:<6} FAILED to load")
            continue

        df = df.sort_index()
        ticker_start = TICKER_START_DATES.get(ticker, DATA_START_DATE)
        df = df.loc[pd.to_datetime(ticker_start).tz_localize('UTC'):]

        df = heal_data(df)
        df = engineer_features(df, market_df)
        df = add_scoring_metadata(df)
        df = apply_scaling(df, stats, ALL_COLS)

        in_range = (df.index >= score_start) & (df.index < score_end)
        valid    = df['Valid_Row'].values == 1
        score_indices = np.where(in_range & valid)[0]
        score_indices = score_indices[score_indices >= TIME_STEPS]

        if len(score_indices) == 0:
            print(f"   {ticker:<6} no valid bars in [{args.start}, {args.end}]")
            del df; gc.collect()
            continue

        keep_cols = list(ASSET_COLS) + list(CONTEXT_COLS) + ['Valid_Row', 'Target', 'Close']
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]

        ticker_data[ticker] = {'df': df, 'idx': score_indices}
        dt = time.time() - t0
        print(f"   {ticker:<6} {len(score_indices):>5} bars  "
              f"[{df.index[score_indices[0]].strftime('%Y-%m-%d')} -> "
              f"{df.index[score_indices[-1]].strftime('%Y-%m-%d')}]  ({dt:.1f}s)")
        gc.collect()

    del market_df
    gc.collect()

    if not ticker_data:
        print("\nNo tickers with valid data. Exiting.")
        return

    # ------------------------------------------------------------------
    # 4. SCORE (outer: seed, inner: ticker)
    # ------------------------------------------------------------------
    print(f"\n[4/4] Scoring {len(ticker_data)} tickers × {len(seeds)} seeds "
          f"= {len(ticker_data)*len(seeds)} CSVs")

    total_csvs = 0
    total_bars = 0
    t_start = time.time()

    for seed_i, seed in enumerate(seeds):
        t_seed = time.time()
        print(f"\n   ── SEED {seed} ({seed_i+1}/{len(seeds)}) ──")

        model_blob = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{MODEL_ID}.h5"
        local_path = os.path.join(tempfile.gettempdir(),
                                  f"UNIVERSAL_S{seed}_M{MODEL_ID}.h5")

        if not os.path.exists(local_path):
            print(f"      Downloading model...", end=" ", flush=True)
            bucket.blob(model_blob).download_to_filename(local_path)
            print("done")

        tf.keras.backend.clear_session()
        model = build_model(len(ASSET_COLS), len(CONTEXT_COLS), N_ASSETS)
        model.load_weights(local_path)

        for ticker, tdata in ticker_data.items():
            df    = tdata['df']
            idx_v = tdata['idx']

            XA = df[ASSET_COLS].values.astype(np.float32)
            XC = df[CONTEXT_COLS].values.astype(np.float32)
            asset_id = np.int32(ASSET_INDEX.get(ticker, OTHER_IDX))

            probs = []
            for i in range(0, len(idx_v), SCORE_BATCH_SIZE):
                batch_idx = idx_v[i:i + SCORE_BATCH_SIZE]
                XA_b  = np.stack([XA[j - TIME_STEPS:j] for j in batch_idx])
                XC_b  = np.stack([XC[j - 1]            for j in batch_idx])
                XID_b = np.full((len(batch_idx), 1), asset_id, dtype=np.int32)
                p = model.predict(
                    {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
                    verbose=0, batch_size=len(batch_idx),
                ).flatten()
                probs.extend(p.tolist())

            results_df = pd.DataFrame({
                'Target':     df['Target'].values[idx_v],
                'Prob_Spike': probs,
                'Prediction': (np.array(probs) > 0.5).astype(int),
                'Close':      df['Close'].values[idx_v],
                'Ticker':     ticker,
            }, index=df.index[idx_v])

            gcs_path = f"{GCS_RESULTS_PATH}/{ticker}/Seed_{seed}_Results.csv"
            bucket.blob(gcs_path).upload_from_string(
                results_df.to_csv(), content_type='text/csv')

            total_csvs += 1
            total_bars += len(probs)
            prob_arr = np.array(probs)
            print(f"      {ticker:<6} {len(probs):>5} bars  "
                  f"p50={np.median(prob_arr):.3f}  "
                  f"p95={np.percentile(prob_arr,95):.3f}  "
                  f"max={prob_arr.max():.3f}")

        dt_seed = time.time() - t_seed
        print(f"      Seed {seed} done in {dt_seed:.0f}s")

        del model
        gc.collect()

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"  COMPLETE")
    print(f"  {total_csvs} CSVs uploaded  |  {total_bars:,} total bars scored")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    print(f"  GCS: gs://{BUCKET_NAME}/{GCS_RESULTS_PATH}/{{ticker}}/Seed_{{seed}}_Results.csv")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
