# %% ── RA-MoE Data Loader Test ────────────────────────────────────────────────
#
# Run this as a Colab cell or standalone script to verify data loading works.
# Tests: GCS connectivity, ticker discovery, feature engineering, regime
# features, forward returns, windowing, and full fold dataset build.
#
# Set MAX_TICKERS to a small number for fast testing (e.g. 5).
# Set to None for full universe (~88 tickers, slower).
#
# GPU NOT required. Colab-ready.

import os, sys, time
import numpy as np, pandas as pd

# Add parent dir to path so ra_moe module is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ra_moe.data_loader import (
    load_master, heal_data, load_context_assets, engineer_features,
    compute_regime_features, compute_forward_returns, get_valid_mask,
    build_windows, discover_tickers, build_fold_datasets, make_tf_dataset,
    WALK_FORWARD_FOLDS, ALL_FEATURE_COLS, REGIME_COLS, TARGET_HORIZONS,
    TIME_STEPS, CONTEXT_ASSETS, clear_cache,
)

# ==============================================================================
# CONFIG
# ==============================================================================
MAX_TICKERS = 5          # set to None for full universe
TEST_FOLD   = 0          # which fold to test (0-3)
BATCH_SIZE  = 256

W = 100
print(f"{'=' * W}")
print(f"  RA-MoE DATA LOADER TEST")
print(f"  Window: T={TIME_STEPS} bars (1-min)")
print(f"  Target horizons: {TARGET_HORIZONS}")
print(f"  Regime features: {len(REGIME_COLS)}")
print(f"  Asset features: {len(ALL_FEATURE_COLS)}")
print(f"  Max tickers: {MAX_TICKERS or 'ALL'}")
print(f"{'=' * W}")


# ==============================================================================
# TEST 1: Discover tickers
# ==============================================================================
print(f"\n── TEST 1: Discover tradable tickers ──")
t0 = time.time()
all_tickers = discover_tickers()
print(f"  Found {len(all_tickers)} tradable tickers ({time.time()-t0:.1f}s)")
print(f"  First 10: {all_tickers[:10]}")
print(f"  Last 10:  {all_tickers[-10:]}")

tickers_to_use = all_tickers[:MAX_TICKERS] if MAX_TICKERS else all_tickers
print(f"  Using: {tickers_to_use}")


# ==============================================================================
# TEST 2: Load one ticker + context
# ==============================================================================
print(f"\n── TEST 2: Load single ticker + context ──")
test_ticker = tickers_to_use[0]
t0 = time.time()
df = load_master(test_ticker)
print(f"  {test_ticker}: {df.shape} ({df.index.min().date()} → {df.index.max().date()}) "
      f"[{time.time()-t0:.1f}s]")
print(f"  Columns: {list(df.columns)}")

t0 = time.time()
df = heal_data(df)
print(f"  After heal_data: {df.shape} [{time.time()-t0:.1f}s]")

t0 = time.time()
market_df = load_context_assets()
print(f"  Context assets loaded: {market_df.shape} [{time.time()-t0:.1f}s]")
print(f"  Context columns: {list(market_df.columns)}")


# ==============================================================================
# TEST 3: Feature engineering
# ==============================================================================
print(f"\n── TEST 3: Feature engineering ──")
t0 = time.time()
df = engineer_features(df, market_df)
print(f"  After engineer_features: {df.shape} [{time.time()-t0:.1f}s]")

missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
present = [c for c in ALL_FEATURE_COLS if c in df.columns]
print(f"  Feature cols present: {len(present)}/{len(ALL_FEATURE_COLS)}")
if missing:
    print(f"  ⚠ MISSING: {missing}")
else:
    print(f"  ✓ All feature columns present")

# Check for NaN/Inf
feat_df = df[ALL_FEATURE_COLS]
nan_counts = feat_df.isna().sum()
nan_cols = nan_counts[nan_counts > 0]
if len(nan_cols) > 0:
    print(f"  ⚠ NaN columns: {dict(nan_cols)}")
else:
    print(f"  ✓ No NaN values in features")

print(f"\n  Feature statistics (last 1000 bars):")
print(feat_df.tail(1000).describe().round(4).to_string())


# ==============================================================================
# TEST 4: Regime features
# ==============================================================================
print(f"\n── TEST 4: Regime features ──")
t0 = time.time()
regime_df = compute_regime_features(market_df)
print(f"  Regime features: {regime_df.shape} [{time.time()-t0:.1f}s]")
print(f"  Columns: {list(regime_df.columns)}")

missing_r = [c for c in REGIME_COLS if c not in regime_df.columns]
if missing_r:
    print(f"  ⚠ MISSING: {missing_r}")
else:
    print(f"  ✓ All regime columns present")

print(f"\n  Regime statistics (last 1000 bars):")
print(regime_df.tail(1000).describe().round(4).to_string())


# ==============================================================================
# TEST 5: Forward returns (target)
# ==============================================================================
print(f"\n── TEST 5: Forward returns ──")
t0 = time.time()
targets = compute_forward_returns(df)
print(f"  Targets: {targets.shape} [{time.time()-t0:.1f}s]")
print(f"  Columns: {list(targets.columns)}")

print(f"\n  Target statistics (last 5000 valid bars):")
valid = get_valid_mask(df)
valid_targets = targets[valid].dropna().tail(5000)
print(valid_targets.describe().round(6).to_string())

for h in TARGET_HORIZONS:
    col = f'fwd_ret_{h}'
    vals = valid_targets[col].dropna()
    pct_pos = (vals > 0).mean() * 100
    pct_neg = (vals < 0).mean() * 100
    print(f"  {col}: {pct_pos:.1f}% positive, {pct_neg:.1f}% negative, "
          f"mean={vals.mean()*100:.3f}%, std={vals.std()*100:.3f}%")


# ==============================================================================
# TEST 6: Windowing
# ==============================================================================
print(f"\n── TEST 6: Build windows (single ticker) ──")
t0 = time.time()
feat_slice = df[ALL_FEATURE_COLS].tail(5000)
regime_slice = regime_df.reindex(feat_slice.index).fillna(0)
target_slice = targets.reindex(feat_slice.index)
valid_slice = valid.reindex(feat_slice.index, fill_value=False)

arrays = build_windows(feat_slice, regime_slice, target_slice, valid_slice,
                        TIME_STEPS, asset_id=0)
if arrays is not None:
    X_seq, X_regime, X_id, Y, dates = arrays
    print(f"  X_seq:    {X_seq.shape}  (N, T, F_asset)")
    print(f"  X_regime: {X_regime.shape}  (N, F_regime)")
    print(f"  X_id:     {X_id.shape}  (N, 1)")
    print(f"  Y:        {Y.shape}  (N, n_horizons)")
    print(f"  dates:    {dates.shape}  [{time.time()-t0:.1f}s]")
    print(f"  Date range: {dates[0]} → {dates[-1]}")
    print(f"  Y NaN count: {np.isnan(Y).sum()}")
    print(f"  ✓ Windowing works")
else:
    print(f"  ⚠ No windows produced!")


# ==============================================================================
# TEST 7: Full fold dataset build
# ==============================================================================
print(f"\n── TEST 7: Full fold dataset build ──")
clear_cache()

fold = WALK_FORWARD_FOLDS[TEST_FOLD]
print(f"  Building fold {fold['fold']}...")
t0 = time.time()

result = build_fold_datasets(fold, tickers=tickers_to_use, verbose=True)
elapsed = time.time() - t0
print(f"  Total build time: {elapsed:.1f}s")

for split in ['train', 'val', 'test']:
    if result[split] is not None:
        n = result[split]['X_seq'].shape[0]
        print(f"\n  {split}:")
        print(f"    Samples:  {n:,}")
        print(f"    X_seq:    {result[split]['X_seq'].shape}")
        print(f"    X_regime: {result[split]['X_regime'].shape}")
        print(f"    X_id:     {result[split]['X_id'].shape}  "
              f"(unique: {np.unique(result[split]['X_id']).tolist()})")
        print(f"    Y:        {result[split]['Y'].shape}")
        print(f"    Y stats:  mean={result[split]['Y'].mean(axis=0)}, "
              f"std={result[split]['Y'].std(axis=0)}")
        if len(result[split]['dates']) > 0:
            print(f"    Dates:    {result[split]['dates'][0]} → {result[split]['dates'][-1]}")
    else:
        print(f"\n  {split}: EMPTY")

print(f"\n  Scaling stats: {len(result['scaling_stats'])} features")
print(f"  Regime stats:  {len(result['regime_stats'])} features")
print(f"  Ticker index:  {result['ticker_index']}")


# ==============================================================================
# TEST 8: TF Dataset
# ==============================================================================
print(f"\n── TEST 8: tf.data.Dataset ──")
if result['train'] is not None:
    ds = make_tf_dataset(result['train'], batch_size=BATCH_SIZE, shuffle=True)
    for batch_inputs, batch_y in ds.take(1):
        print(f"  Batch seq_input:    {batch_inputs['seq_input'].shape}")
        print(f"  Batch regime_input: {batch_inputs['regime_input'].shape}")
        print(f"  Batch asset_id:     {batch_inputs['asset_id'].shape}")
        print(f"  Batch Y:            {batch_y.shape}")
        print(f"  Y sample (first 3): {batch_y[:3].numpy()}")
        print(f"  ✓ tf.data.Dataset works")
else:
    print(f"  ⚠ No training data to build dataset from")


# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  DATA LOADER TEST SUMMARY")
print(f"{'=' * W}")
print(f"  Tickers discovered:     {len(all_tickers)}")
print(f"  Tickers tested:         {len(tickers_to_use)}")
print(f"  Asset feature cols:     {len(ALL_FEATURE_COLS)}")
print(f"  Regime feature cols:    {len(REGIME_COLS)}")
print(f"  Target horizons:        {TARGET_HORIZONS}")
print(f"  Window size:            {TIME_STEPS}")
print(f"  Walk-forward folds:     {len(WALK_FORWARD_FOLDS)}")
if result['train'] is not None:
    print(f"  Train samples (fold {fold['fold']}): {result['train']['X_seq'].shape[0]:,}")
if result['val'] is not None:
    print(f"  Val samples (fold {fold['fold']}):   {result['val']['X_seq'].shape[0]:,}")
if result['test'] is not None:
    print(f"  Test samples (fold {fold['fold']}):  {result['test']['X_seq'].shape[0]:,}")
print(f"  Total time:             {elapsed:.0f}s")
print(f"{'=' * W}")
