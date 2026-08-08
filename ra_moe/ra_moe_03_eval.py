# %% ── Spike Transformer 03: Evaluation ───────────────────────────────────────
#
# Evaluates the trained Spike Transformer model on test data from each fold.
#
# Analyses:
#   1. Classification metrics — AUC-ROC, AUC-PR, precision, recall, F1
#   2. Probability calibration — reliability diagram
#   3. Trading edge — binned predictions vs actual max forward returns
#   4. Per-regime classification breakdown
#   5. Per-ticker AUC ranking
#
# GPU recommended. Colab-ready. All artifacts loaded from / saved to GCS.
# Input:  gs://time-series-bucket-ys/Algos/RA_MoE/real_data/
#         gs://time-series-bucket-ys/Algos/RA_MoE/real_models/
# Output: gs://time-series-bucket-ys/Algos/RA_MoE/real_eval/

import os, time, warnings, gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from google.cloud import storage
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             f1_score, precision_score, recall_score)

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
DATA_MODE = 'real'

DATA_DIR   = '/tmp/ra_moe_real_data' if DATA_MODE == 'real' else '/tmp/ra_moe_data'
MODEL_DIR  = '/tmp/ra_moe_models'
EVAL_DIR   = '/tmp/ra_moe_eval'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

GCS_BUCKET = 'time-series-bucket-ys'
if DATA_MODE == 'real':
    GCS_DATA_PREFIX  = 'Algos/RA_MoE/real_data'
    GCS_MODEL_PREFIX = 'Algos/RA_MoE/real_models'
    GCS_EVAL_PREFIX  = 'Algos/RA_MoE/real_eval'
else:
    GCS_DATA_PREFIX  = 'Algos/RA_MoE/data'
    GCS_MODEL_PREFIX = 'Algos/RA_MoE/models'
    GCS_EVAL_PREFIX  = 'Algos/RA_MoE/eval'

def _gcs_bucket():
    return storage.Client().bucket(GCS_BUCKET)

def download_from_gcs(blob_path, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _gcs_bucket().blob(blob_path).download_to_filename(local_path)

def upload_to_gcs(local_path, prefix):
    fname = os.path.basename(local_path)
    blob_path = f'{prefix}/{fname}'
    _gcs_bucket().blob(blob_path).upload_from_filename(local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"    -> gs://{GCS_BUCKET}/{blob_path} ({size_mb:.1f} MB)")

def ensure_data(filename):
    local_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(local_path):
        print(f"    Downloading {filename} from GCS...")
        download_from_gcs(f'{GCS_DATA_PREFIX}/{filename}', local_path)
    return local_path

def ensure_model(fold_id, filename):
    fold_dir = os.path.join(MODEL_DIR, f'fold_{fold_id}')
    os.makedirs(fold_dir, exist_ok=True)
    local_path = os.path.join(fold_dir, filename)
    if not os.path.exists(local_path):
        print(f"    Downloading fold_{fold_id}/{filename} from GCS...")
        download_from_gcs(f'{GCS_MODEL_PREFIX}/fold_{fold_id}/{filename}', local_path)
    return local_path

# ── Architecture (must match training script) ──
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
SPIKE_THRESHOLD      = 0.06

ASL_GAMMA_NEG        = 5.0
ASL_GAMMA_POS        = 0.1
ASL_CLIP             = 0.10

BATCH_SIZE           = 512
FOLDS_TO_EVAL        = [1, 2]

SEED = 42
W = 100


# ==============================================================================
# MODEL DEFINITION (identical to training script)
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
        return {'loss': loss}

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        preds = self(x, training=False)
        loss = asymmetric_focal_loss(y, preds, sample_weight)
        return {'loss': loss}


# ==============================================================================
# MAIN
# ==============================================================================

print(f"{'=' * W}")
print(f"  SPIKE TRANSFORMER 03: EVALUATION")
print(f"{'=' * W}")
print(f"  Folds to evaluate: {FOLDS_TO_EVAL}")
print(f"  Horizons:          {TARGET_HORIZONS}")
print(f"  Spike threshold:   {SPIKE_THRESHOLD*100:.1f}%")
print(f"{'=' * W}")

# Load metadata
meta = np.load(ensure_data('meta.npz'), allow_pickle=True)
regime_schedule = meta['regime_schedule']
regime_names = list(meta['regime_names'])
n_regimes = len(regime_names)
ticker_names = list(meta['ticker_names']) if 'ticker_names' in meta else None

# Auto-detect feature dimensions
_probe = np.load(ensure_data(f'fold_{FOLDS_TO_EVAL[0]}_test.npz'))
N_ASSET_FEATURES = _probe['X_seq'].shape[2]
N_REGIME_FEATURES = _probe['X_regime'].shape[1]
_probe.close()
print(f"  Data mode:         {DATA_MODE}")
print(f"  Asset features:    {N_ASSET_FEATURES}")
print(f"  Regime features:   {N_REGIME_FEATURES}")

# GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  GPU: {gpus[0].name}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print(f"  GPU: NONE")


# ── Evaluate each fold ──
all_fold_results = {}
t_total = time.time()

for fold_id in FOLDS_TO_EVAL:
    print(f"\n{'─' * W}")
    print(f"  FOLD {fold_id}")
    print(f"{'─' * W}")

    # ── Load model ──
    print(f"  Loading model...")
    model = SpikeTransformer(name=f'spike_tf_fold_{fold_id}')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_FEATURES))
    dummy_reg = tf.zeros((1, N_REGIME_FEATURES))
    _ = model({'seq_input': dummy_seq, 'regime_input': dummy_reg}, training=False)

    weights_path = ensure_model(fold_id, 'model.weights.h5')
    model.load_weights(weights_path)
    print(f"    Weights loaded from {weights_path}")

    # ── Load test data ──
    print(f"  Loading test data...")
    test_path = ensure_data(f'fold_{fold_id}_test.npz')
    test_data = np.load(test_path)
    X_seq    = test_data['X_seq']
    X_regime = test_data['X_regime']
    Y_true   = test_data['Y']
    test_days = test_data['days']
    test_ids  = test_data['X_id'].flatten()
    test_regime_ids = test_data['regime_ids'] if 'regime_ids' in test_data else None
    Y_continuous = test_data['Y_continuous'] if 'Y_continuous' in test_data else None

    max_eval = 100_000
    if len(Y_true) > max_eval:
        idx = np.random.default_rng(SEED).choice(len(Y_true), max_eval, replace=False)
        X_seq, X_regime, Y_true = X_seq[idx], X_regime[idx], Y_true[idx]
        test_days, test_ids = test_days[idx], test_ids[idx]
        if test_regime_ids is not None:
            test_regime_ids = test_regime_ids[idx]
        if Y_continuous is not None:
            Y_continuous = Y_continuous[idx]
    test_data.close()

    n_test = len(Y_true)
    print(f"    {n_test:,} test samples")

    # ── Predictions ──
    print(f"\n  ── 1. Classification Metrics ──")
    preds = model.predict(
        {'seq_input': X_seq, 'regime_input': X_regime},
        batch_size=BATCH_SIZE, verbose=0)

    print(f"    {'Horizon':>8}  {'AUC-ROC':>8}  {'AUC-PR':>8}  {'Prec@0.5':>9}  "
          f"{'Rec@0.5':>8}  {'F1@0.5':>7}  {'SpikeRate':>10}  {'AvgPred':>8}")

    horizon_metrics = {}
    for hi, h in enumerate(TARGET_HORIZONS):
        y = Y_true[:, hi]
        p = preds[:, hi]
        pred_bin = (p >= 0.5).astype(float)

        spike_rate = y.mean()
        avg_pred = p.mean()

        if spike_rate > 0 and spike_rate < 1:
            auc_roc = roc_auc_score(y, p)
            auc_pr  = average_precision_score(y, p)
        else:
            auc_roc = float('nan')
            auc_pr  = float('nan')

        prec = precision_score(y, pred_bin, zero_division=0)
        rec  = recall_score(y, pred_bin, zero_division=0)
        f1   = f1_score(y, pred_bin, zero_division=0)

        print(f"    {h:>5}-min  {auc_roc:>8.4f}  {auc_pr:>8.4f}  {prec:>9.4f}  "
              f"{rec:>8.4f}  {f1:>7.4f}  {spike_rate:>10.4f}  {avg_pred:>8.4f}")

        horizon_metrics[h] = {
            'auc_roc': auc_roc, 'auc_pr': auc_pr,
            'precision': prec, 'recall': rec, 'f1': f1,
            'spike_rate': spike_rate, 'avg_pred': avg_pred
        }

    # ── 2. Probability Calibration ──
    print(f"\n  ── 2. Probability Calibration ──")
    N_BINS = 10
    calibration_data = {}

    for hi, h in enumerate(TARGET_HORIZONS):
        y = Y_true[:, hi]
        p = preds[:, hi]
        bin_edges = np.linspace(0, 1, N_BINS + 1)
        bin_centers = []
        bin_actuals = []
        bin_counts = []

        for b in range(N_BINS):
            lo, hi_edge = bin_edges[b], bin_edges[b + 1]
            if b == N_BINS - 1:
                mask = (p >= lo) & (p <= hi_edge)
            else:
                mask = (p >= lo) & (p < hi_edge)
            n_b = mask.sum()
            if n_b > 0:
                bin_centers.append((lo + hi_edge) / 2)
                bin_actuals.append(y[mask].mean())
                bin_counts.append(n_b)

        calibration_data[h] = {
            'centers': np.array(bin_centers),
            'actuals': np.array(bin_actuals),
            'counts': np.array(bin_counts)
        }

        print(f"    {h}-min calibration:")
        for c, a, n_b in zip(bin_centers, bin_actuals, bin_counts):
            gap = a - c
            print(f"      pred={c:.2f}  actual={a:.4f}  n={n_b:>6}  gap={gap:+.4f}")

    # ── 3. Trading Edge Analysis ──
    print(f"\n  ── 3. Trading Edge Analysis ──")
    trading_edge = {}

    for hi, h in enumerate(TARGET_HORIZONS):
        p = preds[:, hi]
        y = Y_true[:, hi]
        y_cont = Y_continuous[:, hi] if Y_continuous is not None else None

        n_deciles = 10
        decile_edges = np.percentile(p, np.linspace(0, 100, n_deciles + 1))
        decile_edges[-1] += 1e-6

        print(f"\n    {h}-min Trading Edge (by predicted probability decile):")
        print(f"    {'Decile':>8}  {'PredRange':>14}  {'N':>6}  {'SpikeRate':>10}  "
              f"{'AvgMaxRet':>10}  {'AvgPred':>8}")

        decile_stats = []
        for di in range(n_deciles):
            lo, hi_edge = decile_edges[di], decile_edges[di + 1]
            mask = (p >= lo) & (p < hi_edge)
            n_d = mask.sum()
            if n_d == 0:
                continue

            spike_rate_d = y[mask].mean()
            avg_pred_d = p[mask].mean()
            avg_ret_d = y_cont[mask].mean() * 100 if y_cont is not None else float('nan')

            label = f'D{di+1}'
            print(f"    {label:>8}  [{lo:.3f}, {hi_edge:.3f})  {n_d:>6}  "
                  f"{spike_rate_d:>10.4f}  {avg_ret_d:>+9.3f}%  {avg_pred_d:>8.4f}")

            decile_stats.append({
                'lo': lo, 'hi': hi_edge, 'n': n_d,
                'spike_rate': spike_rate_d, 'avg_ret': avg_ret_d,
                'avg_pred': avg_pred_d
            })

        trading_edge[h] = decile_stats

        # Monotonicity check
        rates = [d['spike_rate'] for d in decile_stats]
        monotonic_pairs = sum(1 for i in range(len(rates)-1) if rates[i+1] >= rates[i])
        total_pairs = max(len(rates) - 1, 1)
        print(f"    Monotonicity: {monotonic_pairs}/{total_pairs} pairs increasing "
              f"({'GOOD' if monotonic_pairs >= total_pairs * 0.7 else 'WEAK'})")

        if y_cont is not None:
            rets = [d['avg_ret'] for d in decile_stats]
            ret_mono = sum(1 for i in range(len(rets)-1) if rets[i+1] >= rets[i])
            print(f"    Return monotonicity: {ret_mono}/{total_pairs} pairs increasing")

    # ── 4. Per-Regime Analysis ──
    print(f"\n  ── 4. Per-Regime Classification ──")
    regime_results = {}

    if test_regime_ids is not None:
        print(f"    {'Regime':>18}  {'N':>6}  " +
              "  ".join([f'{h}m AUC' for h in TARGET_HORIZONS]) +
              "  " + "  ".join([f'{h}m SR' for h in TARGET_HORIZONS]))

        for rid in range(n_regimes):
            rname = regime_names[rid]
            mask = (test_regime_ids == rid)
            n_r = mask.sum()
            if n_r < 50:
                continue

            aucs = []
            spike_rates = []
            for hi, h in enumerate(TARGET_HORIZONS):
                y_r = Y_true[mask, hi]
                p_r = preds[mask, hi]
                sr = y_r.mean()
                spike_rates.append(sr)
                if sr > 0 and sr < 1:
                    aucs.append(roc_auc_score(y_r, p_r))
                else:
                    aucs.append(float('nan'))

            auc_str = "  ".join([f'{a:.4f}' for a in aucs])
            sr_str  = "  ".join([f'{s:.4f}' for s in spike_rates])
            print(f"    {rname:>18}  {n_r:>6}  {auc_str}  {sr_str}")

            regime_results[rname] = {
                'n': n_r, 'aucs': aucs, 'spike_rates': spike_rates
            }
    else:
        print(f"    No regime labels available")

    # ── 5. Per-Ticker Analysis ──
    print(f"\n  ── 5. Per-Ticker AUC + Return Analysis ──")
    unique_tickers = np.sort(np.unique(test_ids))
    ticker_stats = []

    TICKER_THRESH = 0.5

    for tid in unique_tickers:
        mask = test_ids == tid
        n_t = mask.sum()
        if n_t < 20:
            continue

        aucs = []
        spike_rates = []
        avg_rets_all = []
        avg_rets_top = []
        n_signals = []
        for hi, h in enumerate(TARGET_HORIZONS):
            y_t = Y_true[mask, hi]
            p_t = preds[mask, hi]
            sr = y_t.mean()
            spike_rates.append(sr)
            if sr > 0 and sr < 1 and n_t >= 20:
                aucs.append(roc_auc_score(y_t, p_t))
            else:
                aucs.append(float('nan'))

            if Y_continuous is not None:
                yc_t = Y_continuous[mask, hi]
                avg_rets_all.append(float(yc_t.mean() * 100))
                top_mask = p_t >= TICKER_THRESH
                n_top = top_mask.sum()
                n_signals.append(int(n_top))
                if n_top > 0:
                    avg_rets_top.append(float(yc_t[top_mask].mean() * 100))
                else:
                    avg_rets_top.append(float('nan'))
            else:
                avg_rets_all.append(float('nan'))
                avg_rets_top.append(float('nan'))
                n_signals.append(0)

        tname = ticker_names[tid] if ticker_names is not None and tid < len(ticker_names) else f'T{tid}'
        ticker_stats.append({
            'ticker_id': tid, 'ticker': tname, 'n': n_t,
            'aucs': aucs, 'spike_rates': spike_rates,
            'avg_auc': np.nanmean(aucs),
            'avg_rets_all': avg_rets_all,
            'avg_rets_top': avg_rets_top,
            'n_signals': n_signals,
        })

    ticker_stats.sort(key=lambda x: -x['avg_auc'] if not np.isnan(x['avg_auc']) else -999)

    # AUC table (top 10 + bottom 5)
    print(f"    {'Ticker':>8}  {'N':>5}  " +
          "  ".join([f'{h}m AUC' for h in TARGET_HORIZONS]) +
          f"  {'AvgAUC':>7}")

    for ts in ticker_stats[:10]:
        auc_str = "  ".join([f'{a:.4f}' if not np.isnan(a) else '   N/A'
                             for a in ts['aucs']])
        print(f"    {ts['ticker']:>8}  {ts['n']:>5}  {auc_str}  {ts['avg_auc']:>7.4f}")
    if len(ticker_stats) > 20:
        print(f"    {'...':>8}")
    for ts in ticker_stats[-5:]:
        auc_str = "  ".join([f'{a:.4f}' if not np.isnan(a) else '   N/A'
                             for a in ts['aucs']])
        print(f"    {ts['ticker']:>8}  {ts['n']:>5}  {auc_str}  {ts['avg_auc']:>7.4f}")

    above_50 = sum(1 for ts in ticker_stats if ts['avg_auc'] > 0.5)
    print(f"\n    {above_50}/{len(ticker_stats)} tickers with avg AUC > 0.5")

    # Per-ticker return table (sorted by 30m top-signal return)
    if Y_continuous is not None:
        hi_30 = TARGET_HORIZONS.index(30) if 30 in TARGET_HORIZONS else 0
        ticker_by_ret = sorted(ticker_stats,
                               key=lambda x: -x['avg_rets_top'][hi_30]
                               if not np.isnan(x['avg_rets_top'][hi_30]) else -999)

        print(f"\n    Per-Ticker Avg Max Return (pred >= {TICKER_THRESH}):")
        print(f"    {'Ticker':>8}  {'N':>5}  " +
              "  ".join([f'{h}m Ret%' for h in TARGET_HORIZONS]) +
              "  " + "  ".join([f'{h}m #Sig' for h in TARGET_HORIZONS]) +
              "  " + "  ".join([f'{h}m All%' for h in TARGET_HORIZONS]))

        for ts in ticker_by_ret:
            if all(np.isnan(r) for r in ts['avg_rets_top']):
                continue
            ret_str = "  ".join([f'{r:>+6.2f}%' if not np.isnan(r) else '    N/A'
                                 for r in ts['avg_rets_top']])
            sig_str = "  ".join([f'{n:>6}' for n in ts['n_signals']])
            all_str = "  ".join([f'{r:>+6.2f}%' for r in ts['avg_rets_all']])
            print(f"    {ts['ticker']:>8}  {ts['n']:>5}  {ret_str}  {sig_str}  {all_str}")

        above_thresh = [ts for ts in ticker_stats
                        if not np.isnan(ts['avg_rets_top'][hi_30])
                        and ts['avg_rets_top'][hi_30] >= 1.7]
        print(f"\n    Tickers with 30m top-signal avg return >= 1.7%: "
              f"{len(above_thresh)}/{len(ticker_stats)}")
        if above_thresh:
            above_thresh.sort(key=lambda x: -x['avg_rets_top'][hi_30])
            for ts in above_thresh:
                print(f"      {ts['ticker']:>8}  30m={ts['avg_rets_top'][hi_30]:>+.2f}%  "
                      f"signals={ts['n_signals'][hi_30]}  AUC={ts['avg_auc']:.3f}")

    # ── 6. Threshold Sweep ──
    print(f"\n  ── 6. Threshold Sweep ──")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"    {'Thresh':>7}  {'Horizon':>8}  {'N_pos':>6}  {'Prec':>6}  "
          f"{'Rec':>6}  {'F1':>6}  {'AvgRet':>8}")

    for hi, h in enumerate(TARGET_HORIZONS):
        y = Y_true[:, hi]
        p = preds[:, hi]
        y_cont = Y_continuous[:, hi] if Y_continuous is not None else None

        for thr in thresholds:
            pred_pos = p >= thr
            n_pos = pred_pos.sum()
            if n_pos < 5:
                continue
            prec = y[pred_pos].mean()
            rec = y[pred_pos].sum() / max(y.sum(), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            avg_ret = y_cont[pred_pos].mean() * 100 if y_cont is not None else float('nan')

            print(f"    {thr:>7.2f}  {h:>5}-min  {n_pos:>6}  {prec:>6.3f}  "
                  f"{rec:>6.3f}  {f1:>6.3f}  {avg_ret:>+7.3f}%")
        print()

    # ── Store results ──
    all_fold_results[fold_id] = {
        'preds': preds,
        'Y_true': Y_true,
        'Y_continuous': Y_continuous,
        'test_days': test_days,
        'test_ids': test_ids,
        'test_regime_ids': test_regime_ids,
        'horizon_metrics': horizon_metrics,
        'calibration_data': calibration_data,
        'trading_edge': trading_edge,
        'regime_results': regime_results,
        'ticker_stats': ticker_stats,
    }

    del model, X_seq, X_regime
    gc.collect()
    tf.keras.backend.clear_session()


# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  CHARTS")
print(f"{'=' * W}")

for fold_id in FOLDS_TO_EVAL:
    res = all_fold_results[fold_id]

    # ── Chart 1: Classification overview (2×3) ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Spike Transformer Evaluation — Fold {fold_id}', fontsize=14,
                 fontweight='bold')
    colors_h = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # [0,0] ROC Curves
    ax = axes[0, 0]
    for hi, h in enumerate(TARGET_HORIZONS):
        y = res['Y_true'][:, hi]
        p = res['preds'][:, hi]
        if y.mean() > 0 and y.mean() < 1:
            fpr, tpr, _ = roc_curve(y, p)
            auc_val = res['horizon_metrics'][h]['auc_roc']
            ax.plot(fpr, tpr, color=colors_h[hi], linewidth=1.5,
                    label=f'{h}-min (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,1] PR Curves
    ax = axes[0, 1]
    for hi, h in enumerate(TARGET_HORIZONS):
        y = res['Y_true'][:, hi]
        p = res['preds'][:, hi]
        if y.mean() > 0 and y.mean() < 1:
            prec_arr, rec_arr, _ = precision_recall_curve(y, p)
            auc_pr = res['horizon_metrics'][h]['auc_pr']
            ax.plot(rec_arr, prec_arr, color=colors_h[hi], linewidth=1.5,
                    label=f'{h}-min (AP={auc_pr:.3f})')
            ax.axhline(y.mean(), color=colors_h[hi], linestyle='--',
                       linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,2] Reliability Diagram
    ax = axes[0, 2]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Perfect')
    for hi, h in enumerate(TARGET_HORIZONS):
        cal = res['calibration_data'][h]
        ax.plot(cal['centers'], cal['actuals'], 'o-', color=colors_h[hi],
                linewidth=1.5, markersize=6, label=f'{h}-min')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Actual Spike Rate')
    ax.set_title('Reliability Diagram')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # [1,0] Trading Edge (spike rate by prediction decile)
    ax = axes[1, 0]
    x_pos = np.arange(10)
    width = 0.25
    for hi, h in enumerate(TARGET_HORIZONS):
        te = res['trading_edge'][h]
        rates = [d['spike_rate'] for d in te]
        while len(rates) < 10:
            rates.append(0)
        ax.bar(x_pos + hi * width, rates[:10], width, color=colors_h[hi],
               alpha=0.7, label=f'{h}-min')
    ax.set_xlabel('Prediction Decile (D1=lowest → D10=highest)')
    ax.set_ylabel('Actual Spike Rate')
    ax.set_title('Trading Edge: Spike Rate by Predicted Probability')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f'D{i+1}' for i in range(10)], fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # [1,1] Trading Edge (avg max return by decile)
    ax = axes[1, 1]
    if res['Y_continuous'] is not None:
        for hi, h in enumerate(TARGET_HORIZONS):
            te = res['trading_edge'][h]
            rets = [d['avg_ret'] for d in te]
            while len(rets) < 10:
                rets.append(0)
            ax.bar(x_pos + hi * width, rets[:10], width, color=colors_h[hi],
                   alpha=0.7, label=f'{h}-min')
        ax.axhline(SPIKE_THRESHOLD * 100, color='red', linestyle='--',
                   linewidth=0.8, label=f'Spike threshold ({SPIKE_THRESHOLD*100:.0f}%)')
        ax.set_xlabel('Prediction Decile')
        ax.set_ylabel('Avg Max Forward Return (%)')
        ax.set_title('Trading Edge: Avg Return by Predicted Probability')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels([f'D{i+1}' for i in range(10)], fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No Y_continuous data', ha='center', va='center',
                transform=ax.transAxes)

    # [1,2] Per-Regime AUC
    ax = axes[1, 2]
    if res['regime_results']:
        rnames = list(res['regime_results'].keys())
        x_pos_r = np.arange(len(rnames))
        for hi, h in enumerate(TARGET_HORIZONS):
            aucs_r = [res['regime_results'][r]['aucs'][hi] for r in rnames]
            ax.bar(x_pos_r + hi * 0.25, aucs_r, 0.23, color=colors_h[hi],
                   alpha=0.7, label=f'{h}-min')
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8,
                   label='Random (0.5)')
        ax.set_xticks(x_pos_r + 0.25)
        ax.set_xticklabels(rnames, fontsize=9)
        ax.set_ylabel('AUC-ROC')
        ax.set_title('AUC by Regime')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No regime data', ha='center', va='center',
                transform=ax.transAxes)

    plt.tight_layout()
    chart_path = os.path.join(EVAL_DIR, f'eval_fold_{fold_id}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    upload_to_gcs(chart_path, GCS_EVAL_PREFIX)
    print(f"  Chart saved → {chart_path} + GCS")
    if _IN_COLAB:
        plt.show()
    plt.close()

    # ── Chart 2: Prediction distribution + per-ticker AUC ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'Spike Transformer — Prediction Analysis (Fold {fold_id})',
                 fontsize=13, fontweight='bold')

    # [0] Prediction histogram colored by actual label
    ax = axes[0]
    for hi, h in enumerate(TARGET_HORIZONS):
        y = res['Y_true'][:, hi]
        p = res['preds'][:, hi]
        ax.hist(p[y == 0], bins=50, alpha=0.3, color='blue',
                density=True, label=f'{h}m neg' if hi == 0 else None)
        ax.hist(p[y == 1], bins=50, alpha=0.3, color='red',
                density=True, label=f'{h}m pos' if hi == 0 else None)
    ax.set_xlabel('Predicted Spike Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution by Label')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1] Per-ticker AUC distribution
    ax = axes[1]
    ts_list = res['ticker_stats']
    avg_aucs = [ts['avg_auc'] for ts in ts_list if not np.isnan(ts['avg_auc'])]
    ax.hist(avg_aucs, bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.0, label='Random (0.5)')
    if avg_aucs:
        ax.axvline(np.median(avg_aucs), color='green', linestyle='--',
                   linewidth=1.0, label=f'Median ({np.median(avg_aucs):.3f})')
    ax.set_xlabel('Avg AUC-ROC')
    ax.set_ylabel('# Tickers')
    ax.set_title('Per-Ticker AUC Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [2] Top/bottom tickers
    ax = axes[2]
    n_show = min(15, len(ts_list))
    top_tickers = ts_list[:n_show]
    names = [ts['ticker'] for ts in top_tickers]
    aucs_show = [ts['avg_auc'] for ts in top_tickers]
    bar_colors = ['#2E7D32' if a > 0.55 else '#1565C0' if a > 0.5 else '#C62828'
                  for a in aucs_show]
    ax.barh(names[::-1], aucs_show[::-1], color=bar_colors[::-1], alpha=0.7)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Avg AUC-ROC')
    ax.set_title(f'Top {n_show} Tickers by AUC')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    chart2_path = os.path.join(EVAL_DIR, f'pred_analysis_fold_{fold_id}.png')
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    upload_to_gcs(chart2_path, GCS_EVAL_PREFIX)
    print(f"  Chart saved → {chart2_path} + GCS")
    if _IN_COLAB:
        plt.show()
    plt.close()

    # ── Chart 3: Per-ticker return analysis ──
    if res['Y_continuous'] is not None:
        hi_30 = TARGET_HORIZONS.index(30) if 30 in TARGET_HORIZONS else 0
        ts_with_ret = [ts for ts in ts_list
                       if not np.isnan(ts['avg_rets_top'][hi_30])
                       and ts['n_signals'][hi_30] >= 5]
        ts_with_ret.sort(key=lambda x: -x['avg_rets_top'][hi_30])

        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.suptitle(f'Spike Transformer — Per-Ticker Returns (Fold {fold_id}, '
                     f'pred >= {TICKER_THRESH})', fontsize=13, fontweight='bold')

        # [0] All tickers: 30m top-signal avg return (horizontal bar)
        ax = axes[0]
        if ts_with_ret:
            names_r = [ts['ticker'] for ts in ts_with_ret]
            rets_r = [ts['avg_rets_top'][hi_30] for ts in ts_with_ret]
            bar_colors_r = ['#2E7D32' if r >= 1.7 else '#1565C0' if r >= 0 else '#C62828'
                            for r in rets_r]
            ax.barh(names_r[::-1], rets_r[::-1], color=bar_colors_r[::-1], alpha=0.7)
            ax.axvline(SPIKE_THRESHOLD * 100, color='red', linestyle='--',
                       linewidth=0.8, label=f'Spike thresh ({SPIKE_THRESHOLD*100:.0f}%)')
            ax.axvline(1.7, color='orange', linestyle='--',
                       linewidth=0.8, label='Deploy thresh (1.7%)')
            ax.set_xlabel('Avg Max Return (%)')
            ax.set_title(f'30-min Top-Signal Return by Ticker')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis='x')

        # [1] Scatter: AUC vs avg return (30m)
        ax = axes[1]
        for ts in ts_with_ret:
            auc_val = ts['avg_auc']
            ret_val = ts['avg_rets_top'][hi_30]
            if np.isnan(auc_val):
                continue
            color = '#2E7D32' if ret_val >= 1.7 else '#1565C0' if ret_val >= 0 else '#C62828'
            ax.scatter(auc_val, ret_val, c=color, s=40, alpha=0.7, edgecolors='black',
                       linewidth=0.3)
            if ret_val >= 1.7 or auc_val >= 0.9:
                ax.annotate(ts['ticker'], (auc_val, ret_val), fontsize=6,
                            xytext=(3, 3), textcoords='offset points')
        ax.axhline(1.7, color='orange', linestyle='--', linewidth=0.8)
        ax.axvline(0.5, color='red', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Avg AUC-ROC')
        ax.set_ylabel('30m Avg Max Return (%)')
        ax.set_title('AUC vs Return (per ticker)')
        ax.grid(True, alpha=0.3)

        # [2] Return distribution histogram (30m)
        ax = axes[2]
        all_top_rets = [ts['avg_rets_top'][hi_30] for ts in ts_with_ret]
        if all_top_rets:
            ax.hist(all_top_rets, bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
            ax.axvline(1.7, color='orange', linestyle='--', linewidth=1.0,
                       label=f'Deploy thresh (1.7%)')
            ax.axvline(np.median(all_top_rets), color='green', linestyle='--',
                       linewidth=1.0, label=f'Median ({np.median(all_top_rets):.2f}%)')
            ax.set_xlabel('30m Avg Max Return (%)')
            ax.set_ylabel('# Tickers')
            ax.set_title('Top-Signal Return Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        chart3_path = os.path.join(EVAL_DIR, f'ticker_returns_fold_{fold_id}.png')
        plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
        upload_to_gcs(chart3_path, GCS_EVAL_PREFIX)
        print(f"  Chart saved → {chart3_path} + GCS")
        if _IN_COLAB:
            plt.show()
        plt.close()


# ==============================================================================
# SUMMARY
# ==============================================================================
total_elapsed = time.time() - t_total
print(f"\n{'=' * W}")
print(f"  EVALUATION SUMMARY")
print(f"{'=' * W}")

for fold_id in FOLDS_TO_EVAL:
    res = all_fold_results[fold_id]
    hm = res['horizon_metrics']

    print(f"\n  Fold {fold_id}:")
    for h in TARGET_HORIZONS:
        m = hm[h]
        print(f"    {h}-min:  AUC-ROC={m['auc_roc']:.4f}  AUC-PR={m['auc_pr']:.4f}  "
              f"F1={m['f1']:.4f}  spike_rate={m['spike_rate']*100:.1f}%")

    # Trading edge summary
    for h in TARGET_HORIZONS:
        te = res['trading_edge'][h]
        if len(te) >= 2:
            top_rate = te[-1]['spike_rate']
            bot_rate = te[0]['spike_rate']
            lift = top_rate / max(bot_rate, 1e-6)
            print(f"    {h}-min edge:  D10 spike rate={top_rate:.4f}  "
                  f"D1={bot_rate:.4f}  lift={lift:.1f}x")

    # Per-ticker summary
    ts_list = res['ticker_stats']
    avg_aucs = [ts['avg_auc'] for ts in ts_list if not np.isnan(ts['avg_auc'])]
    if avg_aucs:
        print(f"    Tickers: {len(avg_aucs)} evaluated  "
              f"median AUC={np.median(avg_aucs):.4f}  "
              f">{0.5}: {sum(1 for a in avg_aucs if a > 0.5)}/{len(avg_aucs)}")

print(f"\n  Key success criteria:")
print(f"    1. AUC-ROC > 0.5 across horizons (above random)")
print(f"    2. Monotonic spike rate: D10 >> D1 (trading edge)")
print(f"    3. Avg max return increases with predicted probability")
print(f"    4. Majority of tickers have AUC > 0.5")

print(f"\n  Total time:  {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"  GCS:         gs://{GCS_BUCKET}/{GCS_EVAL_PREFIX}/")
print(f"{'=' * W}")
