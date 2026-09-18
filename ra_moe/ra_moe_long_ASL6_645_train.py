# %% ── Spike Transformer LONG ASL6-645: Training ──────────────────────────────
#
# ASL6-645: 6:45 AM score / 7:00 AM entry variant with entry-offset targets.
# Uses SHORT-style ASL loss (gamma_neg=6.0, clip=0.08) on LONG data.
#
# Architecture: identical to all ST variants (Non-Stationary Transformer +
# Attention Pooling + FiLM Conditioning + sigmoid head).
# FiLM input is 22 features (16 original + 6 sector-aware).
# N_REGIME_FEATURES auto-detected from meta.npz.
#
# Loss: ASL with gamma_neg=6.0, clip=0.08 (SHORT-style, harder neg penalty)
#
# GPU recommended. Colab-ready. All artifacts persisted to GCS.
# Input:  gs://{GCS_BUCKET}/Algos/RA_MoE/long_ASL6_645_data/fold_{5,6,7}_{train,val,test}.npz
# Output: gs://{GCS_BUCKET}/Algos/RA_MoE/long_ASL6_645_models/fold_{5,6,7}/weights.*

import os, time, warnings, gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from google.cloud import storage

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
    auth.authenticate_user()
except ImportError:
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIG
# ==============================================================================
OUTPUT_DIR = '/tmp/ra_moe_long_ASL6_645_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = '/tmp/ra_moe_long_ASL6_645_data'
os.makedirs(DATA_DIR, exist_ok=True)

GCS_BUCKET       = os.environ.get("GCS_BUCKET", "")
GCS_DATA_PREFIX  = 'Algos/RA_MoE/long_ASL6_645_data'
GCS_MODEL_PREFIX = 'Algos/RA_MoE/long_ASL6_645_models'

def _gcs_bucket():
    return storage.Client().bucket(GCS_BUCKET)

def download_from_gcs(gcs_blob_path, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _gcs_bucket().blob(gcs_blob_path).download_to_filename(local_path)

def upload_to_gcs(local_path, prefix):
    fname = os.path.basename(local_path)
    blob_path = f'{prefix}/{fname}'
    _gcs_bucket().blob(blob_path).upload_from_filename(local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"      -> gs://{GCS_BUCKET}/{blob_path} ({size_mb:.1f} MB)")

def upload_dir_to_gcs(local_dir, prefix):
    bucket = _gcs_bucket()
    count = 0
    for f in os.listdir(local_dir):
        fp = os.path.join(local_dir, f)
        if os.path.isfile(fp):
            bucket.blob(f'{prefix}/{f}').upload_from_filename(fp)
            count += 1
    print(f"      -> gs://{GCS_BUCKET}/{prefix}/ ({count} files)")

def ensure_local(filename, gcs_prefix=GCS_DATA_PREFIX):
    local_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    gcs_path = f'{gcs_prefix}/{filename}'
    print(f"    Downloading {filename} from GCS...")
    download_from_gcs(gcs_path, local_path)
    return local_path

# -- Architecture (identical to all ST variants) --
D_MODEL              = 64
N_HEADS              = 4
D_FF                 = 128
N_TRANSFORMER_BLOCKS = 2
DROPOUT              = 0.1
FILM_HIDDEN          = 64

N_ASSET_FEATURES     = None
N_REGIME_FEATURES    = None
TIME_STEPS           = 128
TARGET_HORIZONS      = [15, 30, 60]
N_HORIZONS           = len(TARGET_HORIZONS)

# -- Loss (SHORT-style: harder neg penalty) --
ASL_GAMMA_NEG        = 6.0
ASL_GAMMA_POS        = 0.1
ASL_CLIP             = 0.08

# -- Training --
EPOCHS               = 60
BATCH_SIZE           = 256
LEARNING_RATE        = 1e-3
MIN_LR               = 1e-5
PATIENCE             = 10
LR_PATIENCE          = 3

MAX_TRAIN_SAMPLES    = None
FOLDS_TO_TRAIN       = [5]  # Change to [6] or [7] to train other folds

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
W = 100


# ==============================================================================
# CUSTOM LAYERS (must match all ST variants exactly for weight compatibility)
# ==============================================================================

class SeriesStationarization(layers.Layer):
    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        std  = tf.math.reduce_std(x, axis=1, keepdims=True) + 1e-5
        x_norm = (x - mean) / std
        stats = tf.concat([mean[:, 0, :], std[:, 0, :]], axis=-1)
        return x_norm, stats


class DeStationaryAttention(layers.Layer):
    def __init__(self, d_model, n_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.scale     = tf.sqrt(tf.cast(self.d_head, tf.float32))

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.wo = layers.Dense(d_model)
        self.attn_drop = layers.Dropout(dropout)

        self.tau_proj   = layers.Dense(n_heads)
        self.sigma_proj = layers.Dense(d_model)

    def call(self, x, stats, training=False):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        H, D = self.n_heads, self.d_head

        Q = tf.transpose(tf.reshape(self.wq(x), (B, T, H, D)), (0, 2, 1, 3))
        K = tf.transpose(tf.reshape(self.wk(x), (B, T, H, D)), (0, 2, 1, 3))
        V = tf.transpose(tf.reshape(self.wv(x), (B, T, H, D)), (0, 2, 1, 3))

        scores = tf.matmul(Q, K, transpose_b=True) / self.scale

        tau = tf.reshape(self.tau_proj(stats), (B, H, 1, 1))
        scores = scores + tau

        attn = self.attn_drop(tf.nn.softmax(scores, axis=-1), training=training)
        out  = tf.matmul(attn, V)
        out  = tf.reshape(tf.transpose(out, (0, 2, 1, 3)), (B, T, self.d_model))

        sigma = tf.nn.softplus(self.sigma_proj(stats))
        out   = out * sigma[:, tf.newaxis, :]

        return self.wo(out)


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn  = DeStationaryAttention(d_model, n_heads, dropout)
        self.ffn   = tf.keras.Sequential([
            layers.Dense(d_ff, activation='gelu'),
            layers.Dense(d_model),
            layers.Dropout(dropout)])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop  = layers.Dropout(dropout)

    def call(self, x, stats, training=False):
        x = x + self.drop(
            self.attn(self.norm1(x), stats, training=training), training=training)
        x = x + self.ffn(self.norm2(x), training=training)
        return x


class TransformerBackbone(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len,
                 dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stationarize = SeriesStationarization()
        self.input_proj   = layers.Dense(d_model)
        self.pos_embed    = self.add_weight(
            name='pos_embed', shape=(1, seq_len, d_model),
            initializer='glorot_normal', trainable=True)
        self.input_drop   = layers.Dropout(dropout)
        self.blocks       = [TransformerBlock(d_model, n_heads, d_ff, dropout,
                                              name=f'block_{i}')
                             for i in range(n_blocks)]
        self.final_norm   = layers.LayerNormalization()

    def call(self, x, training=False):
        x_norm, stats = self.stationarize(x)
        h = self.input_proj(x_norm) + self.pos_embed
        h = self.input_drop(h, training=training)
        for block in self.blocks:
            h = block(h, stats, training=training)
        return self.final_norm(h)


class AttentionPooling(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.query = self.add_weight(
            name='pool_query', shape=(1, 1, d_model),
            initializer='glorot_normal', trainable=True)
        self.scale = tf.sqrt(tf.cast(d_model, tf.float32))

    def call(self, x, training=False):
        scores = tf.matmul(self.query, x, transpose_b=True) / self.scale
        weights = tf.nn.softmax(scores, axis=-1)
        pooled = tf.matmul(weights, x)
        return pooled[:, 0, :]


class FiLMConditioner(layers.Layer):
    def __init__(self, d_model, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu'),
        ])
        self.gamma_proj = layers.Dense(d_model)
        self.beta_proj  = layers.Dense(d_model)

    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        gamma = 1.0 + self.gamma_proj(h)
        beta  = self.beta_proj(h)
        return gamma * x + beta


# ==============================================================================
# MODEL
# ==============================================================================

class SpikeTransformer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone  = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_TRANSFORMER_BLOCKS,
            TIME_STEPS, DROPOUT, name='backbone')
        self.attn_pool = AttentionPooling(D_MODEL, name='attn_pool')
        self.film      = FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
        self.drop      = layers.Dropout(DROPOUT)
        self.head      = layers.Dense(N_HORIZONS, activation='sigmoid',
                                       name='spike_head')

    def call(self, inputs, training=False):
        seq    = inputs['seq_input']
        regime = inputs['regime_input']

        h = self.backbone(seq, training=training)
        h = self.attn_pool(h, training=training)
        h = self.film(h, regime, training=training)
        h = self.drop(h, training=training)
        return self.head(h)

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            preds = self(x, training=True)
            loss = asymmetric_focal_loss(y, preds, sample_weight)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables),
                self.trainable_variables))
        avg_pred = tf.reduce_mean(preds, axis=0)
        return {'loss': loss,
                'avg_pred_15': avg_pred[0],
                'avg_pred_30': avg_pred[1],
                'avg_pred_60': avg_pred[2]}

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        preds = self(x, training=False)
        loss = asymmetric_focal_loss(y, preds, sample_weight)
        avg_pred = tf.reduce_mean(preds, axis=0)
        return {'loss': loss,
                'avg_pred_15': avg_pred[0],
                'avg_pred_30': avg_pred[1],
                'avg_pred_60': avg_pred[2]}


# ==============================================================================
# LOSS
# ==============================================================================

def asymmetric_focal_loss(y_true, y_pred, sample_weight=None):
    p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    pos_loss = -y_true * tf.pow(1.0 - p, ASL_GAMMA_POS) * tf.math.log(p)

    p_m = tf.maximum(p - ASL_CLIP, 0.0)
    neg_loss = -(1.0 - y_true) * tf.pow(p_m, ASL_GAMMA_NEG) * tf.math.log(
        tf.clip_by_value(1.0 - p_m, 1e-7, 1.0))

    per_sample = tf.reduce_mean(pos_loss + neg_loss, axis=-1)

    if sample_weight is not None:
        per_sample = per_sample * sample_weight

    return tf.reduce_mean(per_sample)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_fold(fold_id, split, max_samples=None):
    filename = f'fold_{fold_id}_{split}.npz'
    path = ensure_local(filename)
    data = np.load(path)
    X_seq = data['X_seq']
    X_reg = data['X_regime']
    Y     = data['Y']
    W_sample = data['W_sample'] if 'W_sample' in data else np.ones_like(Y)
    regime_ids = data['regime_ids'] if 'regime_ids' in data else None
    data.close()
    if max_samples and len(Y) > max_samples:
        idx = np.random.default_rng(SEED).choice(len(Y), max_samples, replace=False)
        X_seq, X_reg, Y, W_sample = X_seq[idx], X_reg[idx], Y[idx], W_sample[idx]
        if regime_ids is not None:
            regime_ids = regime_ids[idx]
    W_avg = W_sample.mean(axis=1).astype(np.float32)
    return X_seq, X_reg, Y, W_avg, regime_ids


def make_dataset(X_seq, X_regime, Y, W_sample, batch_size, shuffle=True):
    inputs = {'seq_input': X_seq, 'regime_input': X_regime}
    ds = tf.data.Dataset.from_tensor_slices((inputs, Y, W_sample))
    if shuffle:
        ds = ds.shuffle(min(len(Y), 50000))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# MAIN
# ==============================================================================

print(f"{'=' * W}")
print(f"  SPIKE TRANSFORMER LONG ASL6-645: TRAINING (ENTRY-OFFSET + SHORT-STYLE LOSS)")
print(f"{'=' * W}")
print(f"  Architecture:")
print(f"    Backbone:          Non-Stationary Transformer")
print(f"    d_model:           {D_MODEL}")
print(f"    Heads:             {N_HEADS}")
print(f"    FFN dim:           {D_FF}")
print(f"    Blocks:            {N_TRANSFORMER_BLOCKS}")
print(f"    Pooling:           Attention (learnable query)")
print(f"    Conditioning:      FiLM (hidden={FILM_HIDDEN})")
print(f"    Output:            {N_HORIZONS} horizons x sigmoid (binary spike)")
print(f"  Loss:")
print(f"    Type:              Asymmetric Focal Loss (SHORT-style)")
print(f"    gamma_neg:         {ASL_GAMMA_NEG}")
print(f"    gamma_pos:         {ASL_GAMMA_POS}")
print(f"    clip:              {ASL_CLIP}")
print(f"  Training:")
print(f"    Epochs:            {EPOCHS}")
print(f"    Batch size:        {BATCH_SIZE}")
print(f"    Learning rate:     {LEARNING_RATE} -> {MIN_LR}")
print(f"    Early stop:        patience {PATIENCE}")
print(f"    Max train samples: {MAX_TRAIN_SAMPLES or 'ALL'}")
print(f"    Folds:             {FOLDS_TO_TRAIN}")
print(f"  Data:                {DATA_DIR}")
print(f"  Output:              {OUTPUT_DIR}")
print(f"  NOTE: Entry-offset targets (score 6:45, entry 7:00 PST)")
print(f"{'=' * W}")

# Verify data exists
print(f"\n  Checking data availability...")
bucket = _gcs_bucket()
for fold_id in FOLDS_TO_TRAIN:
    for split in ['train', 'val', 'test']:
        fname = f'fold_{fold_id}_{split}.npz'
        local_ok = os.path.exists(os.path.join(DATA_DIR, fname))
        gcs_ok = bucket.blob(f'{GCS_DATA_PREFIX}/{fname}').exists()
        if not local_ok and not gcs_ok:
            raise FileNotFoundError(
                f"Missing: {fname} (not in /tmp or GCS)\n"
                f"Run ra_moe_long_ASL6_645_data_prep.py first.")
meta_ok = os.path.exists(os.path.join(DATA_DIR, 'meta.npz')) or \
          bucket.blob(f'{GCS_DATA_PREFIX}/meta.npz').exists()
if not meta_ok:
    raise FileNotFoundError("Missing: meta.npz")
print(f"  All data files available (local or GCS)")

# Auto-detect feature dimensions
_probe_path = ensure_local(f'fold_{FOLDS_TO_TRAIN[0]}_val.npz')
_probe = np.load(_probe_path)
N_ASSET_FEATURES = _probe['X_seq'].shape[2]
N_REGIME_FEATURES = _probe['X_regime'].shape[1]
_probe.close()
print(f"  Auto-detected: N_ASSET_FEATURES={N_ASSET_FEATURES}, "
      f"N_REGIME_FEATURES={N_REGIME_FEATURES}")

# GPU check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  GPU: {gpus[0].name}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        print(f"  (memory growth already set — TF initialized by prior cell)")
else:
    print(f"  GPU: NONE — CPU training will be slow (~10x)")


# -- Train each fold --
all_histories = {}
all_test_results = {}
t_total = time.time()

for fold_id in FOLDS_TO_TRAIN:
    print(f"\n{'─' * W}")
    print(f"  FOLD {fold_id}")
    print(f"{'─' * W}")

    # -- Load data --
    print(f"  Loading data...")
    t0 = time.time()
    X_seq_tr, X_reg_tr, Y_tr, W_tr, _ = load_fold(fold_id, 'train', MAX_TRAIN_SAMPLES)
    n_train = len(Y_tr)
    spike_rates_tr = Y_tr.mean(axis=0)
    print(f"    Train spike rates: " + ", ".join(
        [f"{h}m={spike_rates_tr[i]*100:.1f}%"
         for i, h in enumerate(TARGET_HORIZONS)]))
    train_ds = make_dataset(X_seq_tr, X_reg_tr, Y_tr, W_tr, BATCH_SIZE, shuffle=True)
    del X_seq_tr, X_reg_tr, Y_tr, W_tr
    gc.collect()

    X_seq_v, X_reg_v, Y_v, W_v, _ = load_fold(fold_id, 'val', max_samples=100_000)
    n_val = len(Y_v)
    val_ds = make_dataset(X_seq_v, X_reg_v, Y_v, W_v, BATCH_SIZE, shuffle=False)
    del X_seq_v, X_reg_v, Y_v, W_v
    gc.collect()

    print(f"    Train: {n_train:,} samples  |  Val: {n_val:,} samples  ({time.time()-t0:.1f}s)")

    # -- Build model --
    model = SpikeTransformer(name=f'spike_tf_fold_{fold_id}')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    for batch in train_ds.take(1):
        x_b, _, _ = tf.keras.utils.unpack_x_y_sample_weight(batch)
        _ = model(x_b, training=False)

    if fold_id == FOLDS_TO_TRAIN[0]:
        groups = {}
        for v in model.trainable_variables:
            prefix = v.name.split('/')[0]
            groups[prefix] = groups.get(prefix, 0) + v.numpy().size
        total_params = sum(groups.values())
        print(f"\n  Model: {total_params:,} trainable parameters")
        for group, count in sorted(groups.items()):
            print(f"    {group:30s} {count:>8,}")
        print()

    # -- Callbacks --
    fold_dir = os.path.join(OUTPUT_DIR, f'fold_{fold_id}')
    os.makedirs(fold_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=LR_PATIENCE,
            min_lr=MIN_LR, verbose=1),
    ]

    # -- Train --
    print(f"  Training ({n_train // BATCH_SIZE} steps/epoch)...")
    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks, verbose=1)
    train_time = time.time() - t0

    # -- Save weights --
    weights_path = os.path.join(fold_dir, 'model.weights.h5')
    model.save_weights(weights_path)
    upload_to_gcs(weights_path, f'{GCS_MODEL_PREFIX}/fold_{fold_id}')

    # -- Save history --
    hist_dict = {k: np.array(v) for k, v in history.history.items()}
    np.savez(os.path.join(fold_dir, 'history.npz'), **hist_dict)
    upload_to_gcs(os.path.join(fold_dir, 'history.npz'),
                  f'{GCS_MODEL_PREFIX}/fold_{fold_id}')

    best_epoch = int(np.argmin(hist_dict['val_loss']))
    n_epochs_run = len(hist_dict['loss'])
    print(f"\n  Fold {fold_id} training results:")
    print(f"    Epochs run:      {n_epochs_run} (best: {best_epoch + 1})")
    print(f"    Train loss:      {hist_dict['loss'][best_epoch]:.6f}")
    print(f"    Val loss:        {hist_dict['val_loss'][best_epoch]:.6f}")
    print(f"    Training time:   {train_time:.0f}s ({train_time/n_epochs_run:.1f}s/epoch)")
    print(f"    Weights saved:   {weights_path}")

    # -- Quick test evaluation --
    print(f"\n  Test evaluation...")
    test_path = ensure_local(f'fold_{fold_id}_test.npz')
    test_npz = np.load(test_path)
    X_seq_te = test_npz['X_seq']
    X_reg_te = test_npz['X_regime']
    Y_te = test_npz['Y']
    W_te = test_npz['W_sample'].mean(axis=1).astype(np.float32) if 'W_sample' in test_npz else np.ones(len(Y_te), dtype=np.float32)
    test_npz.close()

    max_test = 100_000
    if len(Y_te) > max_test:
        idx = np.random.default_rng(SEED).choice(len(Y_te), max_test, replace=False)
        X_seq_te, X_reg_te, Y_te, W_te = X_seq_te[idx], X_reg_te[idx], Y_te[idx], W_te[idx]

    test_ds = make_dataset(X_seq_te, X_reg_te, Y_te, W_te, BATCH_SIZE, shuffle=False)
    test_res = model.evaluate(test_ds, verbose=0, return_dict=True)
    del test_ds

    print(f"    Test loss:       {test_res['loss']:.6f}")

    preds_te = model.predict(
        {'seq_input': X_seq_te, 'regime_input': X_reg_te},
        batch_size=BATCH_SIZE, verbose=0)

    for hi, h in enumerate(TARGET_HORIZONS):
        p = preds_te[:, hi]
        y = Y_te[:, hi]
        pred_pos = (p >= 0.5).astype(float)
        acc = (pred_pos == y).mean()
        spike_rate = y.mean()
        avg_pred = p.mean()
        print(f"    {h}-min:  acc={acc*100:.1f}%  spike_rate={spike_rate*100:.1f}%  "
              f"avg_pred={avg_pred:.4f}")

    del X_seq_te, X_reg_te, preds_te

    all_histories[fold_id] = hist_dict
    all_test_results[fold_id] = test_res

    del model
    gc.collect()
    tf.keras.backend.clear_session()


# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  TRAINING DIAGNOSTICS")
print(f"{'=' * W}")

n_folds = len(FOLDS_TO_TRAIN)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Spike Transformer LONG ASL6-645 — Training Diagnostics (Entry-Offset + SHORT-Style Loss)',
             fontsize=13, fontweight='bold')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

ax = axes[0, 0]
for i, fid in enumerate(FOLDS_TO_TRAIN):
    h = all_histories[fid]
    ax.plot(h['loss'], color=colors[i % 4], alpha=0.3, linewidth=0.8)
    ax.plot(h['val_loss'], color=colors[i % 4], linewidth=1.5,
            label=f'Fold {fid} val')
    best = np.argmin(h['val_loss'])
    ax.scatter(best, h['val_loss'][best], color=colors[i % 4],
               s=50, zorder=5, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('ASL Loss')
ax.set_title('Asymmetric Focal Loss (train=faint, val=solid)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for i, fid in enumerate(FOLDS_TO_TRAIN):
    h = all_histories[fid]
    for hi, horizon in enumerate(TARGET_HORIZONS):
        key = f'avg_pred_{horizon}'
        val_key = f'val_avg_pred_{horizon}'
        if key in h:
            ls = ['-', '--', ':'][hi]
            ax.plot(h[key], color=colors[i % 4], alpha=0.3,
                    linewidth=0.8, linestyle=ls)
        if val_key in h:
            ax.plot(h[val_key], color=colors[i % 4], linewidth=1.5,
                    linestyle=ls, label=f'F{fid} val {horizon}m')
ax.set_xlabel('Epoch')
ax.set_ylabel('Avg Predicted Probability')
ax.set_title('Avg Spike Predictions Over Training')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for i, fid in enumerate(FOLDS_TO_TRAIN):
    h = all_histories[fid]
    if 'lr' in h:
        ax.plot(h['lr'], color=colors[i % 4], linewidth=1.5,
                label=f'Fold {fid}')
    else:
        n_ep = len(h['loss'])
        ax.axhline(LEARNING_RATE, color=colors[i % 4], linewidth=1.5,
                   label=f'Fold {fid}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule')
ax.legend(fontsize=8)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for i, fid in enumerate(FOLDS_TO_TRAIN):
    h = all_histories[fid]
    gap = np.array(h['val_loss']) - np.array(h['loss'])
    ax.plot(gap, color=colors[i % 4], linewidth=1.5, label=f'Fold {fid}')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Val Loss - Train Loss')
ax.set_title('Generalization Gap (positive = overfitting)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, 'training_diagnostics.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
upload_to_gcs(chart_path, GCS_MODEL_PREFIX)
print(f"  Chart saved -> {chart_path} + GCS")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# SUMMARY
# ==============================================================================
total_elapsed = time.time() - t_total
print(f"\n{'=' * W}")
print(f"  LONG ASL6-645 TRAINING COMPLETE (ENTRY-OFFSET + SHORT-STYLE LOSS)")
print(f"{'=' * W}")
print(f"  Total time:  {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

print(f"\n  Per-fold summary:")
header = (f"  {'Fold':>4}  {'Epochs':>6}  {'Best':>4}  {'TrainLoss':>10}  "
          f"{'ValLoss':>10}  {'TestLoss':>10}")
print(header)
print(f"  {'─' * (len(header) - 2)}")
for fid in FOLDS_TO_TRAIN:
    h = all_histories[fid]
    t_res = all_test_results[fid]
    best = np.argmin(h['val_loss'])
    print(f"  {fid:>4}  {len(h['loss']):>6}  {best+1:>4}  "
          f"{h['loss'][best]:>10.6f}  {h['val_loss'][best]:>10.6f}  "
          f"{t_res['loss']:>10.6f}")

print(f"\n  Local cache:     {OUTPUT_DIR}")
print(f"  GCS:             gs://{GCS_BUCKET}/{GCS_MODEL_PREFIX}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fp = os.path.join(OUTPUT_DIR, f)
    if os.path.isdir(fp):
        files = os.listdir(fp)
        total_mb = sum(os.path.getsize(os.path.join(fp, ff)) / 1e6
                       for ff in files if os.path.isfile(os.path.join(fp, ff)))
        print(f"    {f}/  ({len(files)} files, {total_mb:.1f} MB)")
    else:
        sz = os.path.getsize(fp) / 1e6
        print(f"    {f}  ({sz:.1f} MB)")

print(f"\n  All artifacts persisted to GCS. Safe to restart runtime.")
print(f"  Next: write ASL6-645 optimizer (score at 6:45 AM, entry at 7:00 AM PST)")
print(f"{'=' * W}")
