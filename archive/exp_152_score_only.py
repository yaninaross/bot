"""
exp_152_score_only.py — Score ALL assets with master_inference using exp-152 models
====================================================================================
Loads trained models from GCS, scores all tickers that have master_inference files.
Skips any (seed, ticker) pair that already has a CSV in GCS.

Ticker-first scoring: loads each ticker's data ONCE, scores all 15 seeds against it.
This is 15x more memory-efficient than seed-first scoring.
"""

import sys
import subprocess
import os
import json
import gc

import numpy as np
import pandas as pd
import pytz

SCRIPT_VERSION = "152.3-score-all"
print(f"Script version: {SCRIPT_VERSION}")
print("Scoring all assets for Experiment 152...")

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]", "pandas_ta",
                           "google-cloud-storage", "pyarrow", "gcsfs", "--quiet"])
    import tensorflow as tf

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2
from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass
client = storage.Client()
bucket = client.bucket(os.environ.get("GCS_BUCKET", ""))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"GPU ACTIVE: {len(gpus)} GPU(s)")

# ==============================================================================
# CONFIG — must match exp_152.py exactly
# ==============================================================================
MODEL_ID = "152_XSECTOR"
VERIFY_EXISTING = False  # no CSVs exist yet for 152 — nothing to verify
SEEDS_TO_SCORE = [15201, 15202, 15203, 15204, 15205, 15206, 15207,
                  15208, 15209, 15210, 15211, 15212, 15213, 15214, 15215]

TRAIN_TEACHERS = [
    'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
    'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
]

SCORING_POOL = []  # auto-discovered from GCS master_inference files

CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]
ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS += ['Time_Pos']
ALL_COLS = ASSET_COLS + CONTEXT_COLS

TIME_STEPS = 240
EMBED_DIM  = 16
PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
SPIKE_THRESHOLD = 0.10

BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
GCS_RESULTS_PATH = f"Algos/Results_all_v2/Universal_Exp{MODEL_ID}"
STATS_DIR = "Data/Inference_v2"

DATA_START_DATE   = "2022-01-01"
TRAIN_CUTOFF_DATE = "2025-10-01"
aware_cutoff = pd.Timestamp(TRAIN_CUTOFF_DATE).tz_localize('UTC')

ASSET_INDEX = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
N_ASSETS    = len(TRAIN_TEACHERS) + 1
OTHER_IDX   = len(TRAIN_TEACHERS)

LOCAL_MODEL_DIR = "Models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)


# ==============================================================================
# DATA LOADING — identical to exp_152.py
# ==============================================================================
def load_master(ticker):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    try:
        df = pd.read_parquet(uri)
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            num_col = next((c for c in df.columns
                           if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                          df.columns[0])
            df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
        df = df.set_index('Datetime')
        df = df[~df.index.duplicated(keep='first')]
        std_map = {
            'Total_vol': 'Total_Vol', 'Trade_count': 'Trade_Count',
            'Volume_delta': 'Volume_Delta', 'Delta_pct': 'Delta_Pct',
            'Obi': 'OBI', 'Bsize': 'bsize', 'Asize': 'asize',
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume',
        }
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                      for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        for c in MICRO_COLS:
            if c not in df.columns:
                df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        return df
    except Exception as e:
        print(f"   Failed to load {ticker}: {e}")
        return None


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# MARKET CONTEXT
# ==============================================================================
def build_market_context():
    print("   Building market context...")
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker)
        if df is None:
            continue
        df = df.sort_index()
        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c; gc.collect()
    if not frames:
        raise RuntimeError("No context assets loaded.")
    return pd.concat(frames, axis=1).ffill().fillna(0)


# ==============================================================================
# FEATURE ENGINEERING — identical to exp_152.py
# ==============================================================================
def engineer_features(df, market_df):
    daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
        df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c
    df['PM_High'] = df.groupby(df.index.date)['High'].cummax()
    df['Dist_to_PM_High'] = (df['Close'] - df['PM_High']) / (df['PM_High'] + 1e-9)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_60'] = ta.rsi(df['Close'], length=60)
    vol_real = df['Volume'].where(df['Volume'] > 0)
    df['RVOL'] = df['Volume'] / vol_real.rolling(50, min_periods=5).mean().ffill().clip(lower=1e-9)
    real_mask = df['Volume'] > 0
    atr_raw = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_14'] = atr_raw.where(real_mask).ffill().fillna(0)
    df['Compression'] = (df['High'] - df['Low']) / (df['ATR_14'] + 1e-9)
    for h in ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)
    df = df.join(market_df, how='left').ffill().fillna(0)
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def apply_scaling(df, stats):
    df_s = df.copy()
    for col in ALL_COLS:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


def add_targets(df):
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    is_safe = (df_pst.hour < 13) | ((df_pst.hour == 13) & (df_pst.minute == 0))
    is_real = df['Volume'] > 0
    df['Safe_High'] = np.where(is_safe & is_real, df['High'], -1.0)
    df['Future_Max'] = df.groupby(df.index.date)['Safe_High'].transform(
        lambda x: x.shift(-1).fillna(-1.0).iloc[::-1].cummax().iloc[::-1])
    df['Target'] = (df['Future_Max'] > (df['Close'] * (1 + SPIKE_THRESHOLD))).astype(int)
    df['Valid_Row'] = ((mins >= 360) & (mins <= 390) & (df['Volume'] > 0)).astype(int)
    keep_cols = ALL_COLS + ['Target', 'Valid_Row',
                            'Open', 'High', 'Low', 'Close', 'Volume']
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def engineer_asset(asset_name, market_df):
    df = load_master(asset_name)
    if df is None:
        return None
    df = df.sort_index()
    df = df.loc[pd.to_datetime(DATA_START_DATE).tz_localize('UTC'):]
    df = heal_data(df)
    df = engineer_features(df, market_df)
    df = add_targets(df)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ==============================================================================
# MODEL ARCHITECTURE — identical to exp_152.py
# ==============================================================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.4):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])


def build_universal_model():
    in_a  = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c  = Input(shape=(len(CONTEXT_COLS),),          name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,          name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Embedding(N_ASSETS, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


# ==============================================================================
# STEP 1: Discover tickers + check micro data quality
# ==============================================================================
CONTEXT_SKIP = {'SPY', 'VIX', 'TLT', 'IWM', 'SMH', 'BTCUSD', 'ITA', 'XLE'}
MICRO_CHECK_COL = 'Volume_Delta'
MICRO_MIN_PCT = 5.0

def check_micro_quality(ticker):
    """Return (has_micro, master_updated) for a ticker's master file."""
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return False, None
    blob.reload()
    master_updated = blob.updated
    uri = f"gs://{BUCKET_NAME}/{blob_path}"
    try:
        df = pd.read_parquet(uri, columns=[MICRO_CHECK_COL])
        if df.empty:
            return False, master_updated
        nonzero_pct = (df[MICRO_CHECK_COL] != 0).sum() / len(df) * 100
        return nonzero_pct >= MICRO_MIN_PCT, master_updated
    except Exception:
        return False, master_updated

def delete_scored_csvs(ticker, seeds):
    """Delete all result CSVs for a ticker."""
    deleted = 0
    for seed in seeds:
        blob = bucket.blob(f"{GCS_RESULTS_PATH}/{ticker}/Seed_{seed}_Results.csv")
        if blob.exists():
            blob.delete()
            deleted += 1
    return deleted

print(f"\nDiscovering tickers with master_inference files...")
master_blobs = list(bucket.list_blobs(prefix='Data/Inference_v2/'))
for b in master_blobs:
    if b.name.endswith('_master_inference.parquet'):
        ticker = b.name.replace('Data/Inference_v2/', '').replace('_master_inference.parquet', '')
        if ticker and ticker not in CONTEXT_SKIP:
            SCORING_POOL.append(ticker)
SCORING_POOL = sorted(set(SCORING_POOL))
print(f"  Found {len(SCORING_POOL)} master files")

# ==============================================================================
# STEP 2: Verify which models exist in GCS + download
# ==============================================================================
print(f"\nChecking GCS for trained models...")
available_seeds = []
for seed in SEEDS_TO_SCORE:
    blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{MODEL_ID}.h5"
    local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{MODEL_ID}.h5")
    if bucket.blob(blob_path).exists():
        available_seeds.append(seed)
        if not os.path.exists(local_model_path):
            bucket.blob(blob_path).download_to_filename(local_model_path)
            print(f"  S{seed}: downloaded")
        else:
            print(f"  S{seed}: cached locally")
    else:
        print(f"  S{seed}: MISSING — skipping")

if not available_seeds:
    print("No models found in GCS. Run exp_152.py Phase 1 first.")
    raise SystemExit

print(f"\n{len(available_seeds)} models available for scoring")

# ==============================================================================
# STEP 3: Quality gate — check micro, compare timestamps, build work list
# ==============================================================================
print(f"\nChecking micro data quality for {len(SCORING_POOL)} tickers...")
work_by_ticker = {}
verify_tickers = []
total_deleted_incomplete = 0
total_deleted_stale = 0
total_skipped = 0
complete_tickers = []
incomplete_tickers = []

for ticker in SCORING_POOL:
    has_micro, master_updated = check_micro_quality(ticker)

    if has_micro:
        complete_tickers.append(ticker)
        existing_seeds = []
        missing_seeds = []
        stale_seeds = []
        for seed in available_seeds:
            csv_blob = bucket.blob(f"{GCS_RESULTS_PATH}/{ticker}/Seed_{seed}_Results.csv")
            if csv_blob.exists():
                csv_blob.reload()
                if master_updated and csv_blob.updated < master_updated:
                    stale_seeds.append(seed)
                else:
                    existing_seeds.append(seed)
            else:
                missing_seeds.append(seed)
        needs_scoring = missing_seeds + stale_seeds
        if stale_seeds:
            n_del = delete_scored_csvs(ticker, stale_seeds)
            total_deleted_stale += n_del
        if needs_scoring:
            work_by_ticker[ticker] = needs_scoring
            print(f"  {ticker:8s} COMPLETE — {len(needs_scoring)} to score"
                  f" ({len(stale_seeds)} stale, {len(missing_seeds)} missing"
                  f", {len(existing_seeds)} fresh)")
        elif VERIFY_EXISTING:
            verify_tickers.append(ticker)
            print(f"  {ticker:8s} COMPLETE — {len(existing_seeds)} CSVs, queued for verify")
        else:
            total_skipped += len(existing_seeds)
            print(f"  {ticker:8s} COMPLETE — {len(existing_seeds)} CSVs fresh, skipping")
    else:
        incomplete_tickers.append(ticker)
        n_del = delete_scored_csvs(ticker, available_seeds)
        total_deleted_incomplete += n_del
        if n_del:
            print(f"  {ticker:8s} INCOMPLETE — deleted {n_del} bad CSVs, skipping")
        else:
            print(f"  {ticker:8s} INCOMPLETE — not scored yet, skipping")

total_to_score = sum(len(s) for s in work_by_ticker.values())
print(f"\n  Complete:    {len(complete_tickers)} tickers"
      f" ({total_skipped} fresh, {len(verify_tickers)} to verify)")
print(f"  To score:    {len(work_by_ticker)} tickers, {total_to_score} pairs")
print(f"  Deleted:     {total_deleted_stale} stale + {total_deleted_incomplete} incomplete")
print(f"  Incomplete:  {len(incomplete_tickers)} tickers — no micro data")

if total_to_score == 0 and not verify_tickers:
    print("\nAll complete tickers already scored! Nothing to do.")
    raise SystemExit

# ==============================================================================
# STEP 4: Load scaling stats + market context (one-time)
# ==============================================================================
stats_path = f"{STATS_DIR}/scaling_stats_experiment_{MODEL_ID}.json"
stats = json.loads(bucket.blob(stats_path).download_as_bytes())
print(f"Scaling stats loaded: {len(stats)} features")

market_df = build_market_context()
print(f"Market context: {len(market_df)} rows")

# ==============================================================================
# STEP 5: Verify existing CSVs (spot-check 1 seed per ticker)
# ==============================================================================
if verify_tickers:
    verify_seed = available_seeds[0]
    print(f"\nVerifying {len(verify_tickers)} tickers against Seed {verify_seed}...")

    tf.keras.backend.clear_session()
    vmodel = build_universal_model()
    vmodel.load_weights(os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{verify_seed}_M{MODEL_ID}.h5"))

    verified_ok = 0
    verified_bad = 0

    for vi, vt in enumerate(verify_tickers):
        df_v = engineer_asset(vt, market_df)
        if df_v is None:
            print(f"  [{vi+1}/{len(verify_tickers)}] {vt:8s} SKIP (no data)")
            continue
        df_vs = apply_scaling(df_v, stats)
        df_vt = df_vs[df_vs.index > aware_cutoff]
        idx_v = np.where(df_vt['Valid_Row'].values)[0]
        idx_v = idx_v[idx_v >= TIME_STEPS]

        if len(idx_v) == 0:
            print(f"  [{vi+1}/{len(verify_tickers)}] {vt:8s} SKIP (0 valid rows)")
            del df_v, df_vs, df_vt; gc.collect()
            continue

        XA = df_vt[ASSET_COLS].values
        XC = df_vt[CONTEXT_COLS].values
        aid = np.int32(ASSET_INDEX.get(vt, OTHER_IDX))

        probs_new = []
        for i in range(0, len(idx_v), 4096):
            bi = idx_v[i:i+4096]
            XA_b = np.array([XA[j - TIME_STEPS:j] for j in bi])
            XC_b = np.array([XC[j - 1] for j in bi])
            XID_b = np.full((len(bi), 1), aid, dtype=np.int32)
            p = vmodel.predict([XA_b, XC_b, XID_b], verbose=0).flatten()
            probs_new.extend(p)
        probs_new = np.array(probs_new, dtype=np.float32)

        csv_uri = f"gs://{BUCKET_NAME}/{GCS_RESULTS_PATH}/{vt}/Seed_{verify_seed}_Results.csv"
        try:
            existing = pd.read_csv(csv_uri, index_col=0)
            probs_old = existing['Prob_Spike'].values.astype(np.float32)
        except Exception as e:
            print(f"  [{vi+1}/{len(verify_tickers)}] {vt:8s} FAILED (read error: {e}) — will rescore")
            verified_bad += 1
            delete_scored_csvs(vt, available_seeds)
            work_by_ticker[vt] = list(available_seeds)
            del df_v, df_vs, df_vt; gc.collect()
            continue

        if len(probs_new) != len(probs_old):
            print(f"  [{vi+1}/{len(verify_tickers)}] {vt:8s} FAILED rows {len(probs_old)}→{len(probs_new)} — will rescore")
            verified_bad += 1
            delete_scored_csvs(vt, available_seeds)
            work_by_ticker[vt] = list(available_seeds)
        else:
            max_diff = float(np.max(np.abs(probs_new - probs_old)))
            if max_diff < 0.02:
                print(f"  [{vi+1}/{len(verify_tickers)}] {vt:8s} VERIFIED  max_diff={max_diff:.6f}")
                verified_ok += 1
                total_skipped += len(available_seeds)
            else:
                print(f"  [{vi+1}/{len(verify_tickers)}] {vt:8s} FAILED  max_diff={max_diff:.4f} — will rescore")
                verified_bad += 1
                delete_scored_csvs(vt, available_seeds)
                work_by_ticker[vt] = list(available_seeds)

        del df_v, df_vs, df_vt, XA, XC; gc.collect()

    del vmodel; tf.keras.backend.clear_session(); gc.collect()
    print(f"\n  Verified OK: {verified_ok}")
    print(f"  Failed:      {verified_bad} — added to scoring queue")

    total_to_score = sum(len(s) for s in work_by_ticker.values())
    if total_to_score == 0:
        print("\nAll tickers verified — nothing to rescore!")
        raise SystemExit

# ==============================================================================
# STEP 6: Score — ticker-first (load data once, score all seeds)
# ==============================================================================
total_scored = 0
tickers_done = 0

for score_ticker, seeds_needed in work_by_ticker.items():
    tickers_done += 1
    print(f"\n{'─'*60}")
    print(f"[{tickers_done}/{len(work_by_ticker)}] {score_ticker} — {len(seeds_needed)} seeds")

    df_target = engineer_asset(score_ticker, market_df)
    if df_target is None:
        print("   SKIP (no data)")
        continue
    df_target_scaled = apply_scaling(df_target, stats)

    test_mask = df_target_scaled.index > aware_cutoff
    df_test = df_target_scaled[test_mask]

    idx_v = np.where(df_test['Valid_Row'].values)[0]
    idx_v = idx_v[idx_v >= TIME_STEPS]
    if len(idx_v) == 0:
        print(f"   SKIP (0 valid rows post-cutoff)")
        del df_target, df_target_scaled, df_test; gc.collect()
        continue

    XA_v = df_test[ASSET_COLS].values
    XC_v = df_test[CONTEXT_COLS].values
    asset_id_val = np.int32(ASSET_INDEX.get(score_ticker, OTHER_IDX))

    for seed in seeds_needed:
        tf.keras.backend.clear_session()
        local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{MODEL_ID}.h5")

        model = build_universal_model()
        try:
            model.load_weights(local_model_path)
        except Exception as e:
            print(f"   S{seed}: weights failed ({e}), skipping")
            continue

        probs = []
        for i in range(0, len(idx_v), 4096):
            batch_idx = idx_v[i:i+4096]
            XA_b  = np.array([XA_v[j - TIME_STEPS:j] for j in batch_idx])
            XC_b  = np.array([XC_v[j - 1] for j in batch_idx])
            XID_b = np.full((len(batch_idx), 1), asset_id_val, dtype=np.int32)
            p = model.predict([XA_b, XC_b, XID_b], verbose=0).flatten()
            probs.extend(p)

        results_df = pd.DataFrame({
            'Target':     df_test['Target'].values[idx_v],
            'Prob_Spike': probs,
            'Prediction': (np.array(probs) > 0.5).astype(int),
            'Close':      df_test['Close'].values[idx_v],
            'Ticker':     score_ticker
        }, index=df_test.index[idx_v])

        gcs_csv_path = f"{GCS_RESULTS_PATH}/{score_ticker}/Seed_{seed}_Results.csv"
        csv_bytes = results_df.to_csv().encode()
        bucket.blob(gcs_csv_path).upload_from_string(csv_bytes, content_type='text/csv')

        total_scored += 1
        print(f"   S{seed}: {len(idx_v)} rows → {score_ticker}")

        del model, probs, results_df
        gc.collect()

    del df_target, df_target_scaled, df_test, XA_v, XC_v
    gc.collect()

print(f"\n{'='*60}")
print(f"SCORING COMPLETE: {total_scored} scored, {total_skipped} already fresh")
print(f"{'='*60}")
