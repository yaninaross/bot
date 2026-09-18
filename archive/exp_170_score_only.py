"""
exp_170_score_only.py — Score ALL assets for walk-forward folds 170-175
=======================================================================
Loads trained models from GCS (Saved_models_v2), scores SCORING_POOL
tickers one at a time, uploads result CSVs to Results_all_v2.

All 6 folds (170-175), 15 seeds each = 90 models.
Full ticker universe (~70 tickers with master_inference data).

Skips any (seed, ticker) that already has a CSV in GCS.
Aggressive gc.collect() between tickers to stay under RAM limits.

Set FOLD_RANGE to control which folds to score in this notebook.
Run multiple notebooks with different FOLD_RANGE for parallel scoring.
"""

import sys
import subprocess
import os
import json
import gc

import numpy as np
import pandas as pd
import pytz

SCRIPT_VERSION = "170.2-score-only"
print(f"Script version: {SCRIPT_VERSION}")
print("Scoring-only for Walk-Forward folds 170-175 (all assets)...")

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
# CONFIG
# ==============================================================================
ALL_FOLDS = [
    {'exp_id': '170_XSECTOR', 'cutoff': '2025-03-01', 'seeds': [17001+i for i in range(15)]},
    {'exp_id': '171_XSECTOR', 'cutoff': '2025-05-01', 'seeds': [17101+i for i in range(15)]},
    {'exp_id': '172_XSECTOR', 'cutoff': '2025-07-01', 'seeds': [17201+i for i in range(15)]},
    {'exp_id': '173_XSECTOR', 'cutoff': '2025-09-01', 'seeds': [17301+i for i in range(15)]},
    {'exp_id': '174_XSECTOR', 'cutoff': '2025-11-01', 'seeds': [17401+i for i in range(15)]},
    {'exp_id': '175_XSECTOR', 'cutoff': '2026-01-01', 'seeds': [17501+i for i in range(15)]},
]

# Set which folds to score in this notebook (e.g. [0,1,2] for folds 170-172)
# Leave empty to score all 6
FOLD_RANGE = []

FOLDS_TO_SCORE = [ALL_FOLDS[i] for i in FOLD_RANGE] if FOLD_RANGE else ALL_FOLDS

TRAIN_TEACHERS = [
    'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
    'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
]
SCORING_POOL = TRAIN_TEACHERS + [
    # Original 170 scoring pool
    'MARA', 'RIOT', 'MSTR', 'COIN', 'SMCI', 'RKLB', 'BE',
    'FCEL', 'FLNC', 'PLUG', 'MP', 'POWL', 'MOD', 'ALAB',
    'OKLO', 'SMR',
    # Expanded — all tickers with master_inference data
    'BGDE', 'RDW', 'PL', 'VOYG', 'SPIR', 'BKSY', 'MNTS',
    'JOBY', 'ACHR', 'HOOD', 'NNDM', 'KTOS', 'CORZ', 'NBIS',
    'AVAV', 'BTBT', 'IREN',
    # Large-cap / tech
    'AMD', 'NVDA', 'TSLA', 'PLTR', 'SOUN', 'VRT', 'VST', 'MU',
    # Defense
    'GD', 'GE', 'GEV', 'LHX', 'LMT', 'NOC', 'RTX',
    # Energy / resources
    'BHP', 'CAT', 'CCJ', 'CEG', 'COP', 'CVX', 'XOM', 'FCX',
    # Infra / utilities
    'DUK', 'EQIX', 'ETN', 'LIN', 'NEE', 'NBIS',
]
SCORING_POOL = sorted(set(SCORING_POOL))

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
STATS_DIR = "Data/Inference_v2"
DATA_START_DATE = "2022-01-01"

ASSET_INDEX = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
N_ASSETS    = len(TRAIN_TEACHERS) + 1
OTHER_IDX   = len(TRAIN_TEACHERS)

LOCAL_MODEL_DIR = "Models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)


# ==============================================================================
# DATA LOADING
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
# MODEL ARCHITECTURE
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
# LOAD MARKET CONTEXT (one-time)
# ==============================================================================
market_df = build_market_context()
print(f"Market context: {len(market_df)} rows")

# ==============================================================================
# FOLD LOOP — ticker-cached strategy
# ==============================================================================
# For each fold: load & engineer all tickers ONCE, cache the scaled arrays,
# then iterate seeds (predict-only, ~3s per ticker). ~10x faster than
# reloading per seed.
#
grand_total_scored = 0
grand_total_skipped = 0

for fold in FOLDS_TO_SCORE:
    fold_exp_id = fold['exp_id']
    fold_cutoff = fold['cutoff']
    fold_seeds  = fold['seeds']
    aware_cutoff = pd.Timestamp(fold_cutoff).tz_localize('UTC')
    gcs_results_path = f"Algos/Results_all_v2/Universal_Exp{fold_exp_id}"

    print(f"\n{'='*70}")
    print(f"FOLD: {fold_exp_id}  cutoff={fold_cutoff}  seeds={fold_seeds[0]}-{fold_seeds[-1]}")
    print(f"{'='*70}")

    # --- Check which models exist ---
    available_seeds = []
    for seed in fold_seeds:
        blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{fold_exp_id}.h5"
        if bucket.blob(blob_path).exists():
            available_seeds.append(seed)
    missing = len(fold_seeds) - len(available_seeds)
    print(f"  Models: {len(available_seeds)}/{len(fold_seeds)} available"
          + (f" ({missing} missing)" if missing else ""))

    if not available_seeds:
        print(f"  No models found for {fold_exp_id}. Skipping fold.")
        continue

    # --- Check existing CSVs ---
    existing_csvs = set()
    for seed in available_seeds:
        for ticker in SCORING_POOL:
            csv_path = f"{gcs_results_path}/{ticker}/Seed_{seed}_Results.csv"
            if bucket.blob(csv_path).exists():
                existing_csvs.add((seed, ticker))
    to_score = len(available_seeds) * len(SCORING_POOL) - len(existing_csvs)
    print(f"  Existing CSVs: {len(existing_csvs)}, remaining: {to_score}")

    if to_score == 0:
        print(f"  All scoring complete for {fold_exp_id}. Skipping.")
        grand_total_skipped += len(existing_csvs)
        continue

    # --- Load scaling stats for this fold ---
    stats_path = f"{STATS_DIR}/scaling_stats_experiment_{fold_exp_id}.json"
    stats = json.loads(bucket.blob(stats_path).download_as_bytes())
    print(f"  Scaling stats loaded: {len(stats)} features")

    # --- Pre-download all models for this fold ---
    print(f"\n  Pre-downloading {len(available_seeds)} models...")
    for seed in available_seeds:
        local_model_path = os.path.join(LOCAL_MODEL_DIR,
                                        f"UNIVERSAL_S{seed}_M{fold_exp_id}.h5")
        if not os.path.exists(local_model_path):
            blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{fold_exp_id}.h5"
            bucket.blob(blob_path).download_to_filename(local_model_path)
            print(f"    S{seed}: downloaded")
        else:
            print(f"    S{seed}: already local")

    # --- Score ticker-first: load each ticker ONCE, score with all seeds ---
    # Only 1 ticker + 1 model in RAM at a time. Each ticker loaded once per fold.
    fold_scored = 0
    fold_skipped = 0

    for t_idx, score_ticker in enumerate(SCORING_POOL):
        seeds_needed = [s for s in available_seeds
                        if (s, score_ticker) not in existing_csvs]
        seeds_skip = len(available_seeds) - len(seeds_needed)
        fold_skipped += seeds_skip

        if not seeds_needed:
            print(f"\n  [{t_idx+1}/{len(SCORING_POOL)}] {score_ticker}: "
                  f"all seeds done, skipping")
            continue

        print(f"\n  [{t_idx+1}/{len(SCORING_POOL)}] {score_ticker}: "
              f"{len(seeds_needed)} seeds to score, {seeds_skip} skipped")

        df_target = engineer_asset(score_ticker, market_df)
        if df_target is None:
            print(f"    SKIP (no data)")
            continue
        df_target_scaled = apply_scaling(df_target, stats)

        test_mask = df_target_scaled.index > aware_cutoff
        df_test = df_target_scaled[test_mask]

        idx_v = np.where(df_test['Valid_Row'].values)[0]
        idx_v = idx_v[idx_v >= TIME_STEPS]
        if len(idx_v) == 0:
            print(f"    SKIP (0 valid rows after cutoff)")
            del df_target, df_target_scaled, df_test; gc.collect()
            continue

        XA_v = df_test[ASSET_COLS].values
        XC_v = df_test[CONTEXT_COLS].values
        asset_id_val = np.int32(ASSET_INDEX.get(score_ticker, OTHER_IDX))
        targets = df_test['Target'].values[idx_v]
        closes = df_test['Close'].values[idx_v]
        test_index = df_test.index[idx_v]

        # Pre-build input batches (same for all seeds)
        XA_batches = []
        XC_batches = []
        XID_batches = []
        for i in range(0, len(idx_v), 4096):
            batch_idx = idx_v[i:i+4096]
            XA_batches.append(np.array([XA_v[j - TIME_STEPS:j] for j in batch_idx]))
            XC_batches.append(np.array([XC_v[j - 1] for j in batch_idx]))
            XID_batches.append(np.full((len(batch_idx), 1), asset_id_val, dtype=np.int32))

        del df_target, df_target_scaled, df_test, XA_v, XC_v
        gc.collect()

        model = build_universal_model()

        for seed in seeds_needed:
            local_model_path = os.path.join(LOCAL_MODEL_DIR,
                                            f"UNIVERSAL_S{seed}_M{fold_exp_id}.h5")
            try:
                model.load_weights(local_model_path)
            except Exception as e:
                print(f"    S{seed}: could not load weights ({e}), skipping")
                continue

            probs = []
            for XA_b, XC_b, XID_b in zip(XA_batches, XC_batches, XID_batches):
                p = model.predict([XA_b, XC_b, XID_b], verbose=0).flatten()
                probs.extend(p)

            results_df = pd.DataFrame({
                'Target':     targets,
                'Prob_Spike': probs,
                'Prediction': (np.array(probs) > 0.5).astype(int),
                'Close':      closes,
                'Ticker':     score_ticker
            }, index=test_index)

            gcs_csv_path = f"{gcs_results_path}/{score_ticker}/Seed_{seed}_Results.csv"
            csv_bytes = results_df.to_csv().encode()
            bucket.blob(gcs_csv_path).upload_from_string(csv_bytes,
                                                          content_type='text/csv')
            fold_scored += 1
            print(f"    S{seed}: {len(idx_v)} rows → uploaded")

            del results_df, probs

        del model; gc.collect()

        del XA_batches, XC_batches, XID_batches, targets, closes, test_index
        gc.collect()
        print(f"    {score_ticker} done. Fold progress: {fold_scored} scored")

    grand_total_scored += fold_scored
    grand_total_skipped += fold_skipped
    print(f"\n  FOLD {fold_exp_id} COMPLETE: {fold_scored} scored, {fold_skipped} skipped")

print(f"\n{'='*70}")
print(f"ALL FOLDS COMPLETE: {grand_total_scored} scored, {grand_total_skipped} skipped")
print(f"{'='*70}")
