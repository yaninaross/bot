"""
score_missing_180_185.py — Score 26 missing tickers across folds 180-185
=========================================================================
These tickers (large-cap, defense, energy, infra) were added to SCORING_POOL
after training completed. Models and scaling stats already exist on GCS.
This script downloads each fold's model, scores only unscored tickers, and
uploads results.

No training. No teacher data loading. Loads scaling stats from GCS.
Run in Colab with GPU for numba JIT / TF inference.
"""

import sys
import subprocess
import os
import json
import numpy as np
import pandas as pd
import pytz
import gc
import ctypes

def _release_memory():
    gc.collect()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ==============================================================================
# ENVIRONMENT SYNC & AUTH
# ==============================================================================
print("Scoring missing tickers for folds 180-185...")

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
from tensorflow.keras import mixed_precision
from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass
client = storage.Client()
bucket = client.bucket("time-series-bucket-ys")

# ==============================================================================
# GPU SETUP
# ==============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    mixed_precision.set_global_policy('mixed_float16')
    print(f"GPU ACTIVE: {len(gpus)} GPU(s) detected, Mixed Precision Enabled")

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Multi-GPU: MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
else:
    strategy = tf.distribute.get_strategy()
    print(f"Single-device strategy")

# ==============================================================================
# COLUMN DEFINITIONS (must match training script exactly)
# ==============================================================================
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ROC_HORIZONS = [5, 30, 60, 120]

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

TIME_STEPS = 240
EMBED_DIM  = 16
PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

SPIKE_THRESHOLD   = 0.10
SAMPLE_WEIGHT_CAP = 5.0

BUCKET_NAME   = "time-series-bucket-ys"
INFERENCE_DIR = "Data/Inference_v2"
OHLCV_DIR     = "Data/Inference"

LOCAL_RESULTS_DIR = "Results"
LOCAL_MODEL_DIR   = "Models"
os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR      = "Data/Inference_v2"

DATA_START_DATE  = "2022-01-01"
TRAIN_START_DATE = "2022-02-01"

CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS += ['Time_Pos']
ALL_COLS = ASSET_COLS + CONTEXT_COLS

# ==============================================================================
# FOLD CONFIGS
# ==============================================================================
ALL_FOLDS = [
    {'exp_id': '180_XSECTOR', 'cutoff': '2025-08-01', 'seed_base': 18000},
    {'exp_id': '181_XSECTOR', 'cutoff': '2025-09-01', 'seed_base': 18100},
    {'exp_id': '182_XSECTOR', 'cutoff': '2025-10-01', 'seed_base': 18200},
    {'exp_id': '183_XSECTOR', 'cutoff': '2025-11-01', 'seed_base': 18300},
    {'exp_id': '184_XSECTOR', 'cutoff': '2025-12-01', 'seed_base': 18400},
    {'exp_id': '185_XSECTOR', 'cutoff': '2026-01-01', 'seed_base': 18500},
]
SEEDS_PER_FOLD = 15

# Set which folds to score (empty = all 6)
FOLD_RANGE = []
FOLDS_TO_SCORE = [ALL_FOLDS[i] for i in FOLD_RANGE] if FOLD_RANGE else ALL_FOLDS

# ==============================================================================
# TEACHER LIST (needed for asset embedding index — must match training)
# ==============================================================================
TRAIN_TEACHERS = [
    'CIFR', 'IREN', 'KEEL', 'HUT', 'APLD', 'WULF',
    'RIOT', 'CLSK', 'BTBT', 'ALAB', 'MARA',
    'RCAT', 'ASTS', 'LUNR', 'BBAI', 'RKLB', 'SPIR', 'BKSY', 'MNTS',
    'PLUG', 'BE', 'FCEL', 'FLNC',
    'OKLO', 'SMR',
    'MVIS',
    'BGDE', 'RDW', 'PL', 'VOYG',
    'SMLR',
]
ASSET_INDEX = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
N_ASSETS    = len(TRAIN_TEACHERS) + 1
OTHER_IDX   = len(TRAIN_TEACHERS)

# ==============================================================================
# TICKERS TO SCORE — the 26 missing from all 180-185 folds
# ==============================================================================
MISSING_TICKERS = [
    # Large-cap / tech
    'AMD', 'NVDA', 'TSLA', 'PLTR', 'SOUN', 'VRT', 'VST',
    # Defense
    'GD', 'GE', 'GEV', 'LHX', 'LMT', 'NOC', 'RTX',
    # Energy / resources
    'BHP', 'CAT', 'CCJ', 'CEG', 'COP', 'CVX', 'XOM',
    # Infra / utilities
    'DUK', 'EQIX', 'ETN', 'LIN', 'NEE',
]

print(f"\nConfig: {len(TRAIN_TEACHERS)} teachers, {N_ASSETS} embeddings")
print(f"Folds: {[f['exp_id'] for f in FOLDS_TO_SCORE]}")
print(f"Tickers to score: {len(MISSING_TICKERS)}")
print(f"Seeds per fold: {SEEDS_PER_FOLD}")
print(f"Max work: {len(FOLDS_TO_SCORE)} folds × {len(MISSING_TICKERS)} tickers × {SEEDS_PER_FOLD} seeds "
      f"= {len(FOLDS_TO_SCORE) * len(MISSING_TICKERS) * SEEDS_PER_FOLD} CSVs")

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_master(ticker, tail_days=None, add_micro_zeros=True):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)

    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    fallback_uri = f"gs://{BUCKET_NAME}/{OHLCV_DIR}/{file_ticker}_1m_full_historical_data.parquet"

    df = None
    for attempt_uri in [uri, fallback_uri]:
        try:
            df = pd.read_parquet(attempt_uri)
            if attempt_uri == fallback_uri:
                print(f"      {ticker}: loaded from OHLCV fallback (no master_inference)")
            break
        except Exception:
            continue

    if df is None:
        print(f"   Failed to load {ticker}: no master_inference or OHLCV found")
        return None

    try:
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
        if add_micro_zeros:
            for c in MICRO_COLS:
                if c not in df.columns:
                    df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        if tail_days is not None:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
            df = df[df.index >= cutoff]
            if df.empty:
                return None
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
            print(f"      Could not load context asset {ticker}")
            continue
        df = df.sort_index()
        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
        _release_memory()
    if not frames:
        raise RuntimeError("No context assets loaded.")
    return pd.concat(frames, axis=1).ffill().fillna(0)


# ==============================================================================
# FEATURE ENGINEERING
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
    close_safe = df['Close'].replace(0, np.nan)
    df['Spike_Mag'] = ((df['Future_Max'] / close_safe) - 1).clip(lower=0).fillna(0)
    df['Sample_Weight'] = np.where(
        df['Target'] == 0, 1.0,
        np.clip(df['Spike_Mag'] / SPIKE_THRESHOLD, 1.0, SAMPLE_WEIGHT_CAP))
    keep_cols = ALL_COLS + ['Target', 'Valid_Row', 'Sample_Weight',
                            'Open', 'High', 'Low', 'Close', 'Volume']
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def engineer_asset(asset_name, market_df):
    df = load_master(asset_name, tail_days=None, add_micro_zeros=True)
    if df is None:
        return None
    df = df.sort_index()
    df = df.loc[pd.to_datetime(DATA_START_DATE).tz_localize('UTC'):]
    df = heal_data(df)
    df = engineer_features(df, market_df)
    df = add_targets(df)
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


# ==============================================================================
# MODEL ARCHITECTURE (must match training exactly)
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
# PRE-SCAN — check what actually needs scoring
# ==============================================================================
print(f"\n{'='*70}")
print(f"PRE-SCAN: Checking existing scored CSVs on GCS...")
print(f"{'='*70}")

total_needed = 0
fold_work = {}

for fold_cfg in FOLDS_TO_SCORE:
    fid = fold_cfg['exp_id']
    seeds = [fold_cfg['seed_base'] + 1 + i for i in range(SEEDS_PER_FOLD)]
    gcs_res = f"Algos/Results_all_v2/Universal_Exp{fid}"

    n_models = 0
    missing_pairs = []
    for s in seeds:
        if not bucket.blob(f"{GCS_MODEL_PATH}/UNIVERSAL_S{s}_M{fid}.h5").exists():
            continue
        n_models += 1
        for t in MISSING_TICKERS:
            if not bucket.blob(f"{gcs_res}/{t}/Seed_{s}_Results.csv").exists():
                missing_pairs.append((t, s))

    fold_work[fid] = missing_pairs
    total_needed += len(missing_pairs)
    print(f"  {fid}: {n_models}/{SEEDS_PER_FOLD} models, {len(missing_pairs)} ticker-seed CSVs to generate")

if total_needed == 0:
    print(f"\nAll tickers already scored across all folds. Nothing to do.")
else:
    print(f"\nTotal work: {total_needed} CSVs to generate")
    print(f"Proceeding with data load and scoring...\n")

    # ==============================================================================
    # BUILD MARKET CONTEXT (once)
    # ==============================================================================
    market_df = build_market_context()

    # ==============================================================================
    # SCORE EACH FOLD — engineer one ticker at a time to avoid OOM
    # ==============================================================================
    total_uploaded = 0

    for fold_cfg in FOLDS_TO_SCORE:
        FOLD_EXP_ID = fold_cfg['exp_id']
        FOLD_CUTOFF = fold_cfg['cutoff']
        FOLD_SEEDS  = [fold_cfg['seed_base'] + 1 + i for i in range(SEEDS_PER_FOLD)]
        aware_cutoff = pd.to_datetime(FOLD_CUTOFF).tz_localize('UTC')
        gcs_results_path = f"Algos/Results_all_v2/Universal_Exp{FOLD_EXP_ID}"

        missing_pairs = fold_work.get(FOLD_EXP_ID, [])
        if not missing_pairs:
            print(f"\n  [{FOLD_EXP_ID}] Nothing to score — skipping")
            continue

        missing_tickers_this_fold = sorted(set(t for t, s in missing_pairs))
        missing_seeds_this_fold = sorted(set(s for t, s in missing_pairs))
        missing_set = set(missing_pairs)

        print(f"\n{'='*70}")
        print(f"FOLD: {FOLD_EXP_ID}  cutoff={FOLD_CUTOFF}")
        print(f"  {len(missing_pairs)} CSVs: {len(missing_tickers_this_fold)} tickers × "
              f"up to {len(missing_seeds_this_fold)} seeds")
        print(f"{'='*70}")

        # --- Load scaling stats from GCS ---
        stats_blob_path = f"{STATS_DIR}/scaling_stats_experiment_{FOLD_EXP_ID}.json"
        stats_blob = bucket.blob(stats_blob_path)
        if not stats_blob.exists():
            print(f"  ERROR: scaling stats not found at {stats_blob_path} — skipping fold")
            continue
        stats = json.loads(stats_blob.download_as_text())
        print(f"  Loaded scaling stats from GCS ({len(stats)} features)")

        # --- Download models for this fold ---
        print(f"  Downloading models...")
        available_seeds = []
        for seed in missing_seeds_this_fold:
            local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5")
            gcs_model_blob = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5"
            if not os.path.exists(local_model_path):
                if bucket.blob(gcs_model_blob).exists():
                    bucket.blob(gcs_model_blob).download_to_filename(local_model_path)
                    available_seeds.append(seed)
                else:
                    print(f"    S{seed}: model not found on GCS — skipping")
            else:
                available_seeds.append(seed)
        print(f"  {len(available_seeds)}/{len(missing_seeds_this_fold)} models ready")

        # --- Score ticker-by-ticker (engineer, score all seeds, free) ---
        fold_uploaded = 0
        for t_idx, ticker in enumerate(missing_tickers_this_fold, 1):
            seeds_for_ticker = [s for s in available_seeds if (ticker, s) in missing_set]
            if not seeds_for_ticker:
                continue

            print(f"\n   [{t_idx}/{len(missing_tickers_this_fold)}] {ticker} ({len(seeds_for_ticker)} seeds)...")

            df_eng = engineer_asset(ticker, market_df)
            if df_eng is None:
                print(f"      FAILED to load — skipping")
                continue

            df_scaled = apply_scaling(df_eng, stats)
            del df_eng
            _release_memory()

            test_mask = df_scaled.index > aware_cutoff
            df_test   = df_scaled[test_mask]
            del df_scaled
            _release_memory()

            idx_v = np.where(df_test['Valid_Row'].values)[0]
            idx_v = idx_v[idx_v >= TIME_STEPS]
            if len(idx_v) == 0:
                print(f"      No valid rows after cutoff — skipping")
                del df_test
                _release_memory()
                continue

            XA_v = df_test[ASSET_COLS].values
            XC_v = df_test[CONTEXT_COLS].values
            asset_id_val = np.int32(ASSET_INDEX.get(ticker, OTHER_IDX))
            print(f"      {len(idx_v)} valid rows, asset_id={asset_id_val} (OTHER)")

            for seed in seeds_for_ticker:
                local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5")
                if not os.path.exists(local_model_path):
                    continue

                tf.keras.backend.clear_session()
                with strategy.scope():
                    model = build_universal_model()
                    model.compile(optimizer='adam', loss='binary_crossentropy')
                model.load_weights(local_model_path)

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
                    'Ticker':     ticker
                }, index=df_test.index[idx_v])

                local_csv_name = f"Seed_{seed}_{ticker}_Results.csv"
                local_csv_path = os.path.join(LOCAL_RESULTS_DIR, local_csv_name)
                results_df.to_csv(local_csv_path)

                gcs_csv_path = f"{gcs_results_path}/{ticker}/Seed_{seed}_Results.csv"
                bucket.blob(gcs_csv_path).upload_from_filename(local_csv_path)
                print(f"      S{seed} → {gcs_csv_path}")
                fold_uploaded += 1

                del model
                _release_memory()

            del df_test, XA_v, XC_v
            _release_memory()

        total_uploaded += fold_uploaded
        print(f"\n  Fold {FOLD_EXP_ID} complete: {fold_uploaded} CSVs uploaded")

        # Clean up local model files for this fold
        for seed in available_seeds:
            local_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5")
            if os.path.exists(local_path):
                os.remove(local_path)
        _release_memory()

    del market_df
    _release_memory()

    print(f"\n{'='*70}")
    print(f"ALL FOLDS COMPLETE — {total_uploaded} CSVs uploaded")
    print(f"{'='*70}")
