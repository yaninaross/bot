"""
Exp Magnitude — Spike Magnitude Prediction (Classification & Regression)
========================================================================
Two model types using the same transformer architecture as exp_185:

  MODEL_TYPE = 'classification'
    5-class: <2%, 2-4%, 4-7%, 7-10%, 10%+
    Loss: Sparse categorical focal loss

  MODEL_TYPE = 'regression'
    Continuous Spike_Mag prediction (0 to ~0.30)
    Loss: Huber (delta=0.05) with sample weighting

Both train on ALL valid bars (unconditional), 15 seeds, fold 185 cutoff.
Uses existing data + features from exp_185_train.py.

Usage: Set MODEL_TYPE at top, paste into Colab, run.
"""

import sys, subprocess, os, json, random
import numpy as np, pandas as pd, pytz
from datetime import datetime
import gc, ctypes

def _release_memory():
    gc.collect(); gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

SCRIPT_VERSION = "MAG.1"
print(f"Script version: {SCRIPT_VERSION}")

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
bucket = client.bucket(os.environ.get("GCS_BUCKET", ""))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    mixed_precision.set_global_policy('mixed_float16')
    print(f"GPU ACTIVE: {len(gpus)} GPU(s), Mixed Precision Enabled")

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()
NUM_REPLICAS = strategy.num_replicas_in_sync

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_TYPE = 'classification'  # 'classification' or 'regression'

MAG_CLASSES = [0.02, 0.04, 0.07, 0.10]
N_CLASSES = len(MAG_CLASSES) + 1

FOLD_EXP_ID = '185_XSECTOR'
FOLD_CUTOFF = '2026-01-01'
FOLD_SEEDS = [18501 + i for i in range(15)]

MAG_EXP_ID = f'185_MAG_{"CLS" if MODEL_TYPE == "classification" else "REG"}'

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]
ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS += ['Time_Pos']
ALL_COLS = ASSET_COLS + CONTEXT_COLS

TIME_STEPS = 240; EMBED_DIM = 16
PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

TRAIN_TEACHERS = [
    'CIFR', 'IREN', 'KEEL', 'HUT', 'APLD', 'WULF', 'RIOT', 'CLSK', 'BTBT', 'ALAB', 'MARA',
    'RCAT', 'ASTS', 'LUNR', 'BBAI', 'RKLB', 'SPIR', 'BKSY', 'MNTS', 'PLUG', 'BE', 'FCEL', 'FLNC',
    'OKLO', 'SMR', 'MVIS', 'BGDE', 'RDW', 'PL', 'VOYG', 'SMLR',
]
SCORING_POOL = [t for t in TRAIN_TEACHERS if t != 'SMLR'] + [
    'JOBY', 'MP', 'ACHR', 'SMCI', 'COIN', 'MSTR', 'HOOD',
    'MOD', 'POWL', 'NNDM', 'KTOS', 'CORZ', 'NBIS', 'AVAV',
]

DATA_START_DATE = "2022-01-01"
TRAIN_START_DATE = "2022-02-01"
SPIKE_THRESHOLD = 0.10
SAMPLE_WEIGHT_CAP = 5.0

BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"
OHLCV_DIR = "Data/Inference"
GCS_MODEL_PATH = f"Algos/to_deploy/Saved_models_v2/UNIVERSAL/Magnitude"
STATS_DIR = "Data/Inference_v2"

LOCAL_RESULTS_DIR = "Results_Mag"
LOCAL_MODEL_DIR = "Models_Mag"
os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

PER_REPLICA_BATCH = 256
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH * NUM_REPLICAS

ASSET_INDEX = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
N_ASSETS = len(TRAIN_TEACHERS) + 1
OTHER_IDX = len(TRAIN_TEACHERS)

print(f"\nConfig: {MODEL_TYPE} | {len(TRAIN_TEACHERS)} teachers | {N_ASSETS} embeddings | "
      f"batch={GLOBAL_BATCH_SIZE}")
if MODEL_TYPE == 'classification':
    print(f"Classes: {N_CLASSES} — boundaries at {[f'{c*100:.0f}%' for c in MAG_CLASSES]}")
print(f"Fold: {MAG_EXP_ID}  cutoff={FOLD_CUTOFF}  seeds={FOLD_SEEDS[0]}-{FOLD_SEEDS[-1]}")

# ==============================================================================
# DATA LOADING (reused from exp_185_train.py)
# ==============================================================================
def load_master(ticker, tail_days=None):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    for uri in [f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{file_ticker}_master_inference.parquet",
                f"gs://{BUCKET_NAME}/{OHLCV_DIR}/{file_ticker}_1m_full_historical_data.parquet"]:
        try:
            df = pd.read_parquet(uri)
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if dt_cols:
                df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
            else:
                num_col = next((c for c in df.columns if c in ('Datetime','Date','Time','Timestamp','Level_0')), df.columns[0])
                df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
            df = df.set_index('Datetime')
            df = df[~df.index.duplicated(keep='first')]
            std_map = {'Total_vol':'Total_Vol','Trade_count':'Trade_Count','Volume_delta':'Volume_Delta',
                       'Delta_pct':'Delta_Pct','Obi':'OBI','Bsize':'bsize','Asize':'asize',
                       'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
            lower_map = {k.lower():v for k,v in std_map.items()}
            df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
            for c in MICRO_COLS:
                if c not in df.columns: df[c] = 0.0
            df = df[df.index.notna()].sort_index()
            if tail_days:
                cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
                df = df[df.index >= cutoff]
            return df
        except Exception:
            continue
    return None

def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open','High','Low','Close']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume','Total_Vol','Volume_Delta','Trade_Count','Delta_Pct','Spread','OBI','bsize','asize']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df

def build_market_context():
    print("   Building market context...")
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker)
        if df is None: continue
        df = heal_data(df.sort_index())
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c; _release_memory()
    return pd.concat(frames, axis=1).ffill().fillna(0)

def engineer_features(df, market_df):
    daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(df.index, method='ffill')
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
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def apply_scaling(df, stats, inplace=False):
    df_s = df if inplace else df.copy()
    for col in ALL_COLS:
        if col in ('Time_Pos','Bar_Is_Real'): continue
        if col in stats:
            df_s[col] = ((df_s[col]-stats[col]['med'])/stats[col]['iqr']).clip(-10,10).astype('float32')
    return df_s

# ==============================================================================
# TARGET COMPUTATION — Magnitude
# ==============================================================================
def add_targets_magnitude(df):
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    is_safe = (df_pst.hour < 13) | ((df_pst.hour == 13) & (df_pst.minute == 0))
    is_real = df['Volume'] > 0
    df['Safe_High'] = np.where(is_safe & is_real, df['High'], -1.0)
    df['Future_Max'] = df.groupby(df.index.date)['Safe_High'].transform(
        lambda x: x.shift(-1).fillna(-1.0).iloc[::-1].cummax().iloc[::-1])
    close_safe = df['Close'].replace(0, np.nan)
    df['Spike_Mag'] = ((df['Future_Max'] / close_safe) - 1).clip(lower=0).fillna(0)
    df['Valid_Row'] = ((mins >= 360) & (mins <= 390) & (df['Volume'] > 0)).astype(int)

    if MODEL_TYPE == 'classification':
        mag = df['Spike_Mag'].values
        classes = np.zeros(len(mag), dtype=np.int32)
        for i, boundary in enumerate(MAG_CLASSES):
            classes[mag >= boundary] = i + 1
        df['Mag_Class'] = classes
        df['Target'] = classes

    else:
        df['Target'] = df['Spike_Mag'].astype(np.float32)

    df['Sample_Weight'] = np.where(
        df['Spike_Mag'] < 0.02, 1.0,
        np.clip(df['Spike_Mag'] / 0.05, 1.0, SAMPLE_WEIGHT_CAP))

    keep_cols = ALL_COLS + ['Target', 'Valid_Row', 'Sample_Weight', 'Spike_Mag',
                            'Open', 'High', 'Low', 'Close', 'Volume']
    if MODEL_TYPE == 'classification':
        keep_cols.append('Mag_Class')
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df

def engineer_asset(asset_name, market_df):
    df = load_master(asset_name)
    if df is None: return None
    df = df.sort_index()
    df = df.loc[pd.to_datetime(DATA_START_DATE).tz_localize('UTC'):]
    df = heal_data(df)
    df = engineer_features(df, market_df)
    df = add_targets_magnitude(df)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

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

def build_magnitude_model():
    in_a = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c = Input(shape=(len(CONTEXT_COLS),), name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32, name='asset_id')

    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Flatten()(Embedding(N_ASSETS, EMBED_DIM, name='asset_embed')(in_id))
    e = Dense(32, activation='relu')(e)
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    z = Dropout(0.5)(z)

    if MODEL_TYPE == 'classification':
        out = Dense(N_CLASSES, activation='softmax', dtype='float32', name='mag_class')(z)
    else:
        out = Dense(1, activation='relu', dtype='float32', name='mag_value')(z)

    return Model(inputs=[in_a, in_c, in_id], outputs=out)

# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================
def sparse_focal_loss(gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=tf.math.log(y_pred))
        probs = tf.reduce_max(y_pred * tf.one_hot(y_true, N_CLASSES), axis=-1)
        focal_weight = tf.pow(1.0 - probs, gamma)
        return tf.reduce_mean(focal_weight * ce)
    return loss

# ==============================================================================
# DATASET BUILDER
# ==============================================================================
def make_magnitude_ds(df_scaled, tickers, start_dt, end_dt, steps, batch_size,
                      use_weights=True):
    ticker_arrays = []
    for ticker in tickers:
        mask = df_scaled['Ticker'] == ticker
        if start_dt is not None: mask &= df_scaled.index > start_dt
        if end_dt is not None: mask &= df_scaled.index <= end_dt
        df_t = df_scaled[mask]
        if len(df_t) < steps: continue

        XA = df_t[ASSET_COLS].values.astype('float32')
        XC = df_t[CONTEXT_COLS].values.astype('float32')
        Y = df_t['Target'].values
        SW = df_t['Sample_Weight'].values.astype('float32')
        vi = np.where(df_t['Valid_Row'].values)[0]
        vi = vi[vi >= steps]
        if len(vi) == 0: continue

        asset_id_val = np.int32(ASSET_INDEX.get(ticker, OTHER_IDX))
        ticker_arrays.append((XA, XC, Y, SW, vi, asset_id_val))

        if MODEL_TYPE == 'classification':
            class_dist = [f"c{c}={np.sum(Y[vi]==c)}" for c in range(N_CLASSES)]
            print(f"   [{ticker}] valid={len(vi)}  {' '.join(class_dist)}")
        else:
            mag_vals = Y[vi]
            print(f"   [{ticker}] valid={len(vi)}  mag_mean={mag_vals.mean():.4f}  "
                  f"mag_max={mag_vals.max():.4f}  pct>2%={(mag_vals>0.02).mean():.1%}")

    if not ticker_arrays:
        return None, 0

    combined = np.array(
        [(t, int(i)) for t, (_, _, _, _, idx, _) in enumerate(ticker_arrays) for i in idx],
        dtype=np.int64)

    if MODEL_TYPE == 'classification':
        target_spec = tf.TensorSpec((), tf.int32)
    else:
        target_spec = tf.TensorSpec((), tf.float32)

    def gen():
        while True:
            perm = np.random.permutation(len(combined))
            for p in perm:
                t_idx, i = combined[p]
                XA, XC, Y, SW, _, asset_id_val = ticker_arrays[t_idx]
                w = SW[i] if use_weights else np.float32(1.0)
                target = np.int32(Y[i]) if MODEL_TYPE == 'classification' else np.float32(Y[i])
                yield (
                    {'asset_in': XA[i-steps:i],
                     'context_in': XC[i-1],
                     'asset_id': np.array([asset_id_val], dtype=np.int32)},
                    target, w)

    ds = (
        tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                {'asset_in': tf.TensorSpec((steps, len(ASSET_COLS)), tf.float32),
                 'context_in': tf.TensorSpec((len(CONTEXT_COLS),), tf.float32),
                 'asset_id': tf.TensorSpec((1,), tf.int32)},
                target_spec,
                tf.TensorSpec((), tf.float32)))
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE))
    return ds, len(combined) // batch_size

# ==============================================================================
# EXECUTION
# ==============================================================================
market_df = build_market_context()

print("\nLoading all teacher data...")
stats_dfs = []
for ticker in TRAIN_TEACHERS:
    print(f"   Loading {ticker}...")
    df_temp = engineer_asset(ticker, market_df)
    if df_temp is not None:
        df_temp = df_temp.loc[TRAIN_START_DATE:]
        df_temp['Ticker'] = ticker
        stats_dfs.append(df_temp)
    _release_memory()

df_all_raw = pd.concat(stats_dfs)
del stats_dfs; _release_memory()
df_all_raw.sort_index(inplace=True)
_release_memory()
print(f"\nData loaded: {len(df_all_raw):,} rows")

aware_cutoff = pd.to_datetime(FOLD_CUTOFF).tz_localize('UTC')

# --- Magnitude distribution ---
vr_mask = (df_all_raw['Valid_Row'] == 1) & (df_all_raw.index <= aware_cutoff)
mag_vals = df_all_raw.loc[vr_mask, 'Spike_Mag']
print(f"\nSpike magnitude distribution (train, valid rows only):")
print(f"   Total bars: {len(mag_vals):,}")
print(f"   No move (<2%):  {(mag_vals < 0.02).sum():,} ({(mag_vals < 0.02).mean():.1%})")
for i, b in enumerate(MAG_CLASSES):
    upper = MAG_CLASSES[i+1] if i+1 < len(MAG_CLASSES) else 999
    mask = (mag_vals >= b) & (mag_vals < upper)
    print(f"   {b*100:.0f}%-{upper*100:.0f}%: {mask.sum():,} ({mask.mean():.1%})")
print(f"   10%+:   {(mag_vals >= 0.10).sum():,} ({(mag_vals >= 0.10).mean():.1%})")

# --- Scaling stats (reuse from 185) ---
print(f"\nLoading scaling stats from {FOLD_EXP_ID}...")
stats_blob = bucket.blob(f"{STATS_DIR}/scaling_stats_experiment_{FOLD_EXP_ID}.json")
stats = json.loads(stats_blob.download_as_bytes())

print(f"\nScaling data in-place...")
df_scaled_all = apply_scaling(df_all_raw, stats, inplace=True)
del df_all_raw; _release_memory()

gcs_results_path = f"Algos/Results_all_v2/Universal_Exp{MAG_EXP_ID}"

# ==============================================================================
# TRAINING — 15 seeds
# ==============================================================================
for seed in FOLD_SEEDS:
    print(f"\n{'='*70}")
    print(f"SEED {seed} — {MODEL_TYPE.upper()} MODEL")
    print(f"{'='*70}")

    tf.keras.backend.clear_session()
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

    local_model_path = os.path.join(LOCAL_MODEL_DIR, f"MAG_{MODEL_TYPE[0].upper()}_S{seed}_M{FOLD_EXP_ID}.h5")
    gcs_model_blob = f"{GCS_MODEL_PATH}/MAG_{MODEL_TYPE[0].upper()}_S{seed}_M{FOLD_EXP_ID}.h5"

    if bucket.blob(gcs_model_blob).exists():
        print(f"   Model already trained — downloading for scoring...")
        bucket.blob(gcs_model_blob).download_to_filename(local_model_path)
    else:
        print(f"   Building training dataset...")
        ds_tr, st_tr = make_magnitude_ds(
            df_scaled_all, TRAIN_TEACHERS,
            start_dt=None, end_dt=aware_cutoff,
            steps=TIME_STEPS, batch_size=GLOBAL_BATCH_SIZE, use_weights=True)

        print(f"   Building validation dataset...")
        ds_v, st_v = make_magnitude_ds(
            df_scaled_all, TRAIN_TEACHERS,
            start_dt=aware_cutoff, end_dt=None,
            steps=TIME_STEPS, batch_size=GLOBAL_BATCH_SIZE, use_weights=False)

        if ds_tr is None or st_tr == 0:
            print(f"   No valid training samples. Skipping.")
            continue

        print(f"   Steps/epoch: {st_tr}  Val steps: {st_v}")

        with strategy.scope():
            model = build_magnitude_model()
            try:
                optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
            except AttributeError:
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

            if MODEL_TYPE == 'classification':
                model.compile(
                    optimizer=optimizer,
                    loss=sparse_focal_loss(gamma=2.0),
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.SparseCategoricalAccuracy(name='cat_acc'),
                        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc'),
                    ])
            else:
                model.compile(
                    optimizer=optimizer,
                    loss=tf.keras.losses.Huber(delta=0.05),
                    metrics=[
                        'mae',
                        tf.keras.metrics.MeanSquaredError(name='mse'),
                        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                        tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
                    ])

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                local_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=6, restore_best_weights=True, mode='min', verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, mode='min', verbose=1),
        ]

        model.fit(
            ds_tr, validation_data=ds_v,
            epochs=50, steps_per_epoch=st_tr, validation_steps=st_v,
            callbacks=callbacks, verbose=1)

        if os.path.exists(local_model_path):
            bucket.blob(gcs_model_blob).upload_from_filename(local_model_path)
            print(f"   UPLOADED: {gcs_model_blob}")

    # ------------------------------------------------------------------
    # SCORING
    # ------------------------------------------------------------------
    print(f"\nSCORING SEED {seed}...")
    with strategy.scope():
        model = build_magnitude_model()
        if MODEL_TYPE == 'classification':
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        else:
            model.compile(optimizer='adam', loss='mse')
    model.load_weights(local_model_path)

    for score_ticker in SCORING_POOL:
        gcs_csv_check = f"{gcs_results_path}/{score_ticker}/Seed_{seed}_Results.csv"
        if bucket.blob(gcs_csv_check).exists():
            continue
        print(f"   Scoring {score_ticker}...")

        df_target = engineer_asset(score_ticker, market_df)
        if df_target is None: continue
        df_target_scaled = apply_scaling(df_target, stats)

        test_mask = df_target_scaled.index > aware_cutoff
        df_test = df_target_scaled[test_mask]
        idx_v = np.where(df_test['Valid_Row'].values)[0]
        idx_v = idx_v[idx_v >= TIME_STEPS]
        if len(idx_v) == 0: continue

        XA_v = df_test[ASSET_COLS].values
        XC_v = df_test[CONTEXT_COLS].values
        asset_id_val = np.int32(ASSET_INDEX.get(score_ticker, OTHER_IDX))

        preds = []
        for i in range(0, len(idx_v), 4096):
            batch_idx = idx_v[i:i+4096]
            XA_b = np.array([XA_v[j-TIME_STEPS:j] for j in batch_idx])
            XC_b = np.array([XC_v[j-1] for j in batch_idx])
            XID_b = np.full((len(batch_idx), 1), asset_id_val, dtype=np.int32)
            p = model.predict([XA_b, XC_b, XID_b], verbose=0)
            preds.append(p)

        preds = np.concatenate(preds, axis=0)

        if MODEL_TYPE == 'classification':
            pred_class = np.argmax(preds, axis=1)
            results_df = pd.DataFrame({
                'Actual_Mag': df_test['Spike_Mag'].values[idx_v],
                'Pred_Class': pred_class,
                'Prob_C0': preds[:, 0], 'Prob_C1': preds[:, 1],
                'Prob_C2': preds[:, 2], 'Prob_C3': preds[:, 3],
                'Prob_C4': preds[:, 4],
                'Close': df_test['Close'].values[idx_v],
                'Ticker': score_ticker
            }, index=df_test.index[idx_v])
        else:
            pred_mag = preds.flatten()
            results_df = pd.DataFrame({
                'Actual_Mag': df_test['Spike_Mag'].values[idx_v],
                'Pred_Mag': pred_mag,
                'Close': df_test['Close'].values[idx_v],
                'Ticker': score_ticker
            }, index=df_test.index[idx_v])

        local_csv = os.path.join(LOCAL_RESULTS_DIR, f"Seed_{seed}_{score_ticker}_Results.csv")
        results_df.to_csv(local_csv)
        gcs_csv_path = f"{gcs_results_path}/{score_ticker}/Seed_{seed}_Results.csv"
        bucket.blob(gcs_csv_path).upload_from_filename(local_csv)
        print(f"      Uploaded: {gcs_csv_path}")

        del df_target, df_target_scaled, df_test; _release_memory()

    del model; _release_memory()
    print(f"   Seed {seed} complete.")

del df_scaled_all; _release_memory()

print(f"\n{'='*70}")
print(f"MAGNITUDE TRAINING COMPLETE — {MAG_EXP_ID}, {len(FOLD_SEEDS)} seeds")
print(f"{'='*70}")
