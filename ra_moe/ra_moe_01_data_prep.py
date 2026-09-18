# %% ── RA-MoE 01: Data Preparation (Synthetic) ──────────────────────────────
#
# Generates synthetic multivariate 1-min bar data with rotating regimes,
# engineers features, computes regime descriptors and forward returns,
# builds walk-forward fold datasets, and saves to GCS bucket.
#
# 3 synthetic regimes rotate every ~40 trading days (~2 months):
#   - TRENDING:       positive drift, low noise, high autocorrelation
#   - MEAN_REVERTING: zero drift, moderate noise, negative autocorrelation
#   - HIGH_VOL:       negative drift, high noise, fat tails
#
# The regime schedule is deterministic so evaluation can check whether
# the router detects transitions.
#
# No BQ or GPU required. Colab-ready.
# Output: gs://{GCS_BUCKET}/Algos/RA_MoE/data/fold_{1..4}_{train,val,test}.npz
#         gs://{GCS_BUCKET}/Algos/RA_MoE/data/meta.npz

import os, time, warnings, io
import numpy as np
import pandas as pd
from google.cloud import storage

warnings.filterwarnings('ignore')

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
N_TICKERS       = 30
BARS_PER_DAY    = 390       # 6.5 hours × 60 min
TRADING_DAYS    = 500       # ~2 years of trading days (Aug 2024 → Aug 2026)
TIME_STEPS      = 128       # lookback window
TARGET_HORIZONS = [15, 30, 60]
SEED            = 42

# Subsample: take every Nth valid bar as a prediction point.
# 15 = one sample per 15 min. Keeps all 128 bars in the lookback window,
# just fewer prediction points. Cuts memory ~15x.
SUBSAMPLE_EVERY = 15

# Regime rotation period (trading days)
REGIME_PERIOD   = 40        # ~2 months

# Feature dimensions
N_ASSET_FEATURES  = 18      # per-bar features
N_REGIME_FEATURES = 12      # router input
N_CONTEXT_COLS    = 5       # SPY/VIX/TLT/IWM ROCs + Time_Pos (subset)

# Walk-forward folds: each fold uses day indices
# Total 500 days → Train ~350, Val ~30, Test ~30 per fold, expanding
# Fold boundaries at days 350, 380, 410, 440 approximate the monthly schedule
FOLD_CONFIGS = [
    {'fold': 1, 'train_end': 350, 'val_start': 350, 'val_end': 380,
     'test_start': 380, 'test_end': 410},
    {'fold': 2, 'train_end': 380, 'val_start': 380, 'val_end': 410,
     'test_start': 410, 'test_end': 440},
    {'fold': 3, 'train_end': 410, 'val_start': 410, 'val_end': 440,
     'test_start': 440, 'test_end': 470},
    {'fold': 4, 'train_end': 440, 'val_start': 440, 'val_end': 470,
     'test_start': 470, 'test_end': 500},
]

OUTPUT_DIR = '/tmp/ra_moe_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── GCS persistence ──
GCS_BUCKET      = os.environ.get("GCS_BUCKET", "")
GCS_DATA_PREFIX = 'Algos/RA_MoE/data'

def _gcs_bucket():
    return storage.Client().bucket(GCS_BUCKET)

def upload_to_gcs(local_path, prefix=GCS_DATA_PREFIX):
    fname = os.path.basename(local_path)
    blob_path = f'{prefix}/{fname}'
    _gcs_bucket().blob(blob_path).upload_from_filename(local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"      -> gs://{GCS_BUCKET}/{blob_path} ({size_mb:.1f} MB)")

W = 100

# Feature and regime column names (for reference / debugging)
ASSET_COL_NAMES = [
    'Gap_Pct', 'RSI_14', 'RVOL', 'ATR_14', 'Bar_Is_Real',
    'ROC_5m', 'ROC_30m', 'ROC_60m',
    'Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
    'bsize', 'asize', 'Total_Vol',
    'SPY_ROC_5m', 'Time_Pos',
]

REGIME_COL_NAMES = [
    'regime_rvol_15', 'regime_rvol_60', 'regime_rvol_128',
    'regime_trend_60', 'regime_trend_128',
    'regime_autocorr_60',
    'regime_xsect_disp',
    'regime_vix_level', 'regime_vix_velocity',
    'regime_time_sin', 'regime_time_cos',
    'regime_spy_ret_60',
]

TARGET_COL_NAMES = [f'fwd_ret_{h}' for h in TARGET_HORIZONS]


# ==============================================================================
# REGIME DEFINITIONS
# ==============================================================================
REGIMES = {
    0: {'name': 'TRENDING',       'drift': +0.00003, 'vol': 0.001,  'autocorr': +0.05, 'tail': 3.0},
    1: {'name': 'MEAN_REVERTING', 'drift':  0.00000, 'vol': 0.0015, 'autocorr': -0.08, 'tail': 5.0},
    2: {'name': 'HIGH_VOL',       'drift': -0.00002, 'vol': 0.004,  'autocorr': +0.02, 'tail': 2.0},
}


def get_regime_schedule(n_days, period=REGIME_PERIOD):
    """Returns array of regime IDs (0/1/2) per trading day."""
    n_regimes = len(REGIMES)
    schedule = np.zeros(n_days, dtype=np.int32)
    for d in range(n_days):
        cycle = (d // period) % n_regimes
        schedule[d] = cycle
    return schedule


# ==============================================================================
# SYNTHETIC PRICE GENERATION
# ==============================================================================
def generate_prices(n_days, bars_per_day, regime_schedule, rng, ticker_id=0):
    """
    Generate 1-min OHLCV for one synthetic ticker.
    Regime parameters modulate drift, volatility, and autocorrelation.
    Each ticker gets a slight idiosyncratic offset for diversity.
    """
    n_bars = n_days * bars_per_day
    prices = np.zeros(n_bars)
    highs  = np.zeros(n_bars)
    lows   = np.zeros(n_bars)
    volumes = np.zeros(n_bars)

    ticker_drift_offset = (ticker_id - N_TICKERS / 2) * 0.000002
    ticker_vol_scale = 1.0 + (ticker_id % 5) * 0.1

    anchor_price = 50.0 + ticker_id * 2.0
    price = anchor_price
    prev_ret = 0.0
    mean_revert_strength = 0.00001

    for day in range(n_days):
        regime_id = regime_schedule[day]
        regime = REGIMES[regime_id]
        drift = regime['drift'] + ticker_drift_offset
        vol = regime['vol'] * ticker_vol_scale
        ac = regime['autocorr']
        tail_df = regime['tail']

        for bar in range(bars_per_day):
            idx = day * bars_per_day + bar

            log_deviation = np.log(price / anchor_price)
            mr_pull = -mean_revert_strength * log_deviation

            noise = rng.standard_t(tail_df) * vol
            ret = drift + mr_pull + ac * prev_ret + noise
            ret = np.clip(ret, -0.05, 0.05)
            prev_ret = ret

            open_p = price
            close_p = price * (1 + ret)
            close_p = max(close_p, 1.0)
            high_p = max(open_p, close_p) * (1 + abs(rng.normal(0, vol * 0.3)))
            low_p  = min(open_p, close_p) * (1 - abs(rng.normal(0, vol * 0.3)))
            low_p  = max(low_p, 0.5)

            prices[idx] = close_p
            highs[idx]  = high_p
            lows[idx]   = low_p
            volumes[idx] = max(0, rng.normal(1e6, 3e5) * (1 + abs(ret) * 50))

            price = close_p

    opens = np.roll(prices, 1)
    opens[0] = prices[0]

    return opens, highs, lows, prices, volumes


# ==============================================================================
# FEATURE ENGINEERING (synthetic)
# ==============================================================================
def engineer_asset_features(opens, highs, lows, closes, volumes, bars_per_day):
    """Compute N_ASSET_FEATURES per bar from synthetic OHLCV (vectorized)."""
    n = len(closes)
    features = np.zeros((n, N_ASSET_FEATURES), dtype=np.float32)

    close_s = pd.Series(closes)
    prev_day_c = np.roll(closes, bars_per_day)
    prev_day_c[:bars_per_day] = closes[0]
    features[:, 0] = (opens - prev_day_c) / (prev_day_c + 1e-8)

    diffs = close_s.diff()
    gains = diffs.clip(lower=0).rolling(14, min_periods=14).mean()
    losses = (-diffs).clip(lower=0).rolling(14, min_periods=14).mean()
    rs = gains / (losses + 1e-8)
    features[:, 1] = ((100 - 100 / (1 + rs)) / 100.0).fillna(0.5).values

    vol_s = pd.Series(volumes)
    vol_ma = vol_s.rolling(50, min_periods=5).mean().fillna(1e6)
    features[:, 2] = (vol_s / vol_ma.clip(lower=1)).fillna(1).values

    hl = highs - lows
    prev_c = np.roll(closes, 1); prev_c[0] = closes[0]
    hc = np.abs(highs - prev_c)
    lc = np.abs(lows - prev_c)
    tr = np.maximum(hl, np.maximum(hc, lc))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().fillna(0).values
    features[:, 3] = atr / (closes + 1e-8)

    features[:, 4] = 1.0

    for j, h in enumerate([5, 30, 60]):
        shifted = np.roll(closes, h)
        shifted[:h] = closes[0]
        features[:, 5 + j] = (closes - shifted) / (shifted + 1e-8)
        features[:h, 5 + j] = 0.0

    local_rng = np.random.default_rng(abs(hash(closes[:8].tobytes())) % (2**31))
    for col_idx in range(8, 16):
        features[:, col_idx] = local_rng.normal(0, 0.01, n).astype(np.float32)

    features[:, 16] = features[:, 5] * 0.5

    bar_in_day = np.arange(n) % bars_per_day
    features[:, 17] = bar_in_day / bars_per_day

    return features


def compute_regime_features_synthetic(closes_market, regime_schedule,
                                       bars_per_day, all_closes):
    """
    Compute N_REGIME_FEATURES per bar from market-wide data.
    Uses the "market" ticker (index 0) as SPY proxy.
    """
    n = len(closes_market)
    regime = np.zeros((n, N_REGIME_FEATURES), dtype=np.float32)

    returns_market = np.zeros(n)
    returns_market[1:] = np.diff(closes_market) / (closes_market[:-1] + 1e-8)

    # 0-2: Realized vol at 15, 60, 128 windows
    ret_series = pd.Series(returns_market)
    for j, w in enumerate([15, 60, 128]):
        regime[:, j] = ret_series.rolling(w, min_periods=max(w // 2, 1)).std().fillna(0).values

    # 3-4: Trend strength (cum_ret / rvol)
    for j, w in enumerate([60, 128]):
        cum = ret_series.rolling(w, min_periods=max(w // 2, 1)).sum().fillna(0).values
        rvol = np.clip(regime[:, 1 + j], 1e-8, None)  # reuse rvol_60 or rvol_128
        regime[:, 3 + j] = np.clip(cum / rvol, -10, 10)

    # 5: Return autocorrelation (60 bars) — vectorized via rolling cov/var
    ret_lag1 = ret_series.shift(1)
    roll_cov = ret_series.rolling(60, min_periods=30).cov(ret_lag1).fillna(0)
    roll_var = ret_series.rolling(60, min_periods=30).var().clip(lower=1e-16)
    regime[:, 5] = (roll_cov / roll_var).clip(-1, 1).fillna(0).values

    # 6: Cross-sectional dispersion
    if all_closes is not None and all_closes.shape[1] > 1:
        all_rets = np.zeros_like(all_closes)
        all_rets[1:] = np.diff(all_closes, axis=0) / (all_closes[:-1] + 1e-8)
        regime[:, 6] = np.nanstd(all_rets, axis=1)
    else:
        regime[:, 6] = regime[:, 1]

    # 7-8: VIX proxy (use realized vol as VIX stand-in)
    regime[:, 7] = regime[:, 1] * 100  # "VIX level"
    vix_series = pd.Series(regime[:, 7])
    regime[:, 8] = vix_series.diff(15).fillna(0).values  # "VIX velocity"

    # 9-10: Time-of-day sin/cos
    for i in range(n):
        bar_in_day = i % bars_per_day
        t = bar_in_day / bars_per_day
        regime[i, 9]  = np.sin(2 * np.pi * t)
        regime[i, 10] = np.cos(2 * np.pi * t)

    # 11: SPY 60-bar return
    regime[:, 11] = ret_series.rolling(60, min_periods=30).sum().fillna(0).values

    return regime


def compute_forward_returns_synthetic(closes, horizons):
    """Compute forward returns at given horizons."""
    n = len(closes)
    targets = np.full((n, len(horizons)), np.nan, dtype=np.float32)
    for j, h in enumerate(horizons):
        for i in range(n - h):
            targets[i, j] = (closes[i + h] - closes[i]) / (closes[i] + 1e-8)
    return targets


# ==============================================================================
# SCALING
# ==============================================================================
def compute_scaling_stats(data, axis=0):
    """Compute median/IQR stats for robust scaling."""
    med = np.nanmedian(data, axis=axis)
    q75 = np.nanpercentile(data, 75, axis=axis)
    q25 = np.nanpercentile(data, 25, axis=axis)
    iqr = np.maximum(q75 - q25, 1e-8)
    return med, iqr


def apply_scaling(data, med, iqr, skip_cols=None):
    """Apply median/IQR scaling with clipping."""
    scaled = data.copy()
    for j in range(data.shape[-1]):
        if skip_cols and j in skip_cols:
            continue
        scaled[..., j] = np.clip((data[..., j] - med[j]) / iqr[j], -10, 10)
    return scaled


# ==============================================================================
# WINDOWING
# ==============================================================================
def build_windows(features, regime, targets, valid_mask,
                  window_size, asset_id, day_indices, subsample=1):
    """
    Extract (window, regime, target) triples from contiguous arrays.
    valid_mask: boolean array, True for bars eligible as prediction points.
    subsample: take every Nth valid bar (1 = all, 15 = one per 15 min).
    """
    n = len(features)
    max_h = max(TARGET_HORIZONS)

    candidate = np.where(valid_mask)[0]
    candidate = candidate[(candidate >= window_size) & (candidate < n - max_h)]

    has_target = ~np.isnan(targets[candidate]).any(axis=1)
    candidate = candidate[has_target]

    if subsample > 1:
        candidate = candidate[::subsample]

    if len(candidate) == 0:
        return None

    N = len(candidate)
    X_seq    = np.zeros((N, window_size, features.shape[1]), dtype=np.float32)
    X_regime = regime[candidate].astype(np.float32)
    X_id     = np.full((N, 1), asset_id, dtype=np.int32)
    Y        = targets[candidate].astype(np.float32)
    bar_days = day_indices[candidate]

    for j, i in enumerate(candidate):
        X_seq[j] = features[i - window_size:i]

    return X_seq, X_regime, X_id, Y, bar_days


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
print(f"{'=' * W}")
print(f"  RA-MoE 01: SYNTHETIC DATA PREPARATION")
print(f"{'=' * W}")
print(f"  Tickers:         {N_TICKERS}")
print(f"  Trading days:    {TRADING_DAYS} (~{TRADING_DAYS/250:.1f} years)")
print(f"  Bars/day:        {BARS_PER_DAY}")
print(f"  Total bars:      {TRADING_DAYS * BARS_PER_DAY:,} per ticker")
print(f"  Regime period:   {REGIME_PERIOD} days (~{REGIME_PERIOD/20:.1f} months)")
print(f"  Window:          T={TIME_STEPS}")
print(f"  Targets:         {TARGET_HORIZONS} min fwd returns")
print(f"  Folds:           {len(FOLD_CONFIGS)}")
print(f"  Output:          {OUTPUT_DIR}")
print(f"{'=' * W}")

rng = np.random.default_rng(SEED)
t_total = time.time()

# ── Step 1: Generate regime schedule ──
print(f"\n  Step 1: Regime schedule")
regime_schedule = get_regime_schedule(TRADING_DAYS)
for rid, rinfo in REGIMES.items():
    n_days = (regime_schedule == rid).sum()
    print(f"    Regime {rid} ({rinfo['name']:>15}): {n_days} days, "
          f"drift={rinfo['drift']:+.5f}, vol={rinfo['vol']:.4f}, "
          f"autocorr={rinfo['autocorr']:+.2f}")

# ── Step 2: Generate synthetic OHLCV ──
print(f"\n  Step 2: Generating OHLCV for {N_TICKERS} tickers...")
t0 = time.time()
n_bars = TRADING_DAYS * BARS_PER_DAY

# Store OHLCV per ticker (list of tuples, not one giant array)
ticker_ohlcv = []
all_closes_for_regime = np.zeros((n_bars, N_TICKERS), dtype=np.float32)

for t_id in range(N_TICKERS):
    o, h, l, c, v = generate_prices(TRADING_DAYS, BARS_PER_DAY,
                                      regime_schedule, rng, t_id)
    ticker_ohlcv.append((o.astype(np.float32), h.astype(np.float32),
                          l.astype(np.float32), c.astype(np.float32),
                          v.astype(np.float32)))
    all_closes_for_regime[:, t_id] = c

print(f"    Done ({time.time()-t0:.1f}s). Price range: "
      f"[{all_closes_for_regime.min():.1f}, {all_closes_for_regime.max():.1f}]")

day_indices = np.repeat(np.arange(TRADING_DAYS), BARS_PER_DAY)

# ── Step 3: Regime features (market-wide, computed once) ──
print(f"\n  Step 3: Computing regime features...")
t0 = time.time()
regime_features = compute_regime_features_synthetic(
    all_closes_for_regime[:, 0], regime_schedule, BARS_PER_DAY,
    all_closes_for_regime)
print(f"    Regime shape: {regime_features.shape} ({time.time()-t0:.1f}s)")

# ── Step 4: Valid bar mask ──
valid_bars_in_day = np.zeros(BARS_PER_DAY, dtype=bool)
valid_bars_in_day[:360] = True
valid_mask_global = np.tile(valid_bars_in_day, TRADING_DAYS)
print(f"\n  Step 4: Valid bars: {valid_mask_global.sum():,} / {n_bars:,} "
      f"({valid_mask_global.mean()*100:.0f}%)")
print(f"    Subsample: every {SUBSAMPLE_EVERY}th valid bar "
      f"(~{valid_mask_global.sum() // SUBSAMPLE_EVERY:,} per ticker per full period)")

# ── Step 5: Compute scaling stats from first fold's training period ──
# Use a random sample of tickers to estimate stats (memory-efficient)
print(f"\n  Step 5: Computing scaling stats...")
t0 = time.time()
sample_tickers = list(range(0, N_TICKERS, max(1, N_TICKERS // 10)))
feat_samples = []
train_bar_end_f1 = FOLD_CONFIGS[0]['train_end'] * BARS_PER_DAY
for t_id in sample_tickers:
    o, h, l, c, v = ticker_ohlcv[t_id]
    feat = engineer_asset_features(o[:train_bar_end_f1], h[:train_bar_end_f1],
                                    l[:train_bar_end_f1], c[:train_bar_end_f1],
                                    v[:train_bar_end_f1], BARS_PER_DAY)
    feat_samples.append(feat[::50])  # subsample for stats

feat_concat = np.concatenate(feat_samples, axis=0)
feat_med, feat_iqr = compute_scaling_stats(feat_concat)
del feat_samples, feat_concat
skip_cols = {4, 17}

regime_train = regime_features[:train_bar_end_f1]
regime_med, regime_iqr = compute_scaling_stats(regime_train)
print(f"    Done ({time.time()-t0:.1f}s)")

# ── Step 6: Build fold datasets (one ticker at a time, memory-efficient) ──
print(f"\n  Step 6: Building walk-forward fold datasets...")

for fold_cfg in FOLD_CONFIGS:
    fold_id    = fold_cfg['fold']
    train_end  = fold_cfg['train_end']
    val_start  = fold_cfg['val_start']
    val_end    = fold_cfg['val_end']
    test_start = fold_cfg['test_start']
    test_end   = fold_cfg['test_end']

    print(f"\n    ── Fold {fold_id} ──")
    print(f"    Train: day 0→{train_end} | Val: {val_start}→{val_end} | "
          f"Test: {test_start}→{test_end}")

    for label, ds, de in [('Train', 0, train_end), ('Val', val_start, val_end),
                           ('Test', test_start, test_end)]:
        split_regimes = regime_schedule[ds:de]
        counts = {REGIMES[r]['name']: int((split_regimes == r).sum())
                  for r in REGIMES}
        print(f"    {label} regimes: {counts}")

    splits = {
        'train': (0, train_end),
        'val':   (val_start, val_end),
        'test':  (test_start, test_end),
    }

    for split_name, (day_s, day_e) in splits.items():
        bar_e = day_e * BARS_PER_DAY
        lookback_bar_s = max(0, day_s * BARS_PER_DAY - TIME_STEPS - 10)

        split_X_seq, split_X_regime, split_X_id, split_Y = [], [], [], []
        split_days = []

        for t_id in range(N_TICKERS):
            o, h, l, c, v = ticker_ohlcv[t_id]

            feat = engineer_asset_features(
                o[lookback_bar_s:bar_e], h[lookback_bar_s:bar_e],
                l[lookback_bar_s:bar_e], c[lookback_bar_s:bar_e],
                v[lookback_bar_s:bar_e], BARS_PER_DAY)

            tgt = compute_forward_returns_synthetic(
                c[lookback_bar_s:bar_e], TARGET_HORIZONS)

            reg   = regime_features[lookback_bar_s:bar_e, :]
            valid = valid_mask_global[lookback_bar_s:bar_e].copy()
            d_idx = day_indices[lookback_bar_s:bar_e]

            in_split = (d_idx >= day_s) & (d_idx < day_e)
            valid = valid & in_split

            feat_scaled = apply_scaling(feat, feat_med, feat_iqr, skip_cols)
            reg_scaled  = apply_scaling(reg, regime_med, regime_iqr)

            result = build_windows(feat_scaled, reg_scaled, tgt, valid,
                                    TIME_STEPS, t_id, d_idx,
                                    subsample=SUBSAMPLE_EVERY)
            if result is None:
                continue

            xs, xr, xi, y, bd = result
            split_X_seq.append(xs)
            split_X_regime.append(xr)
            split_X_id.append(xi)
            split_Y.append(y)
            split_days.append(bd)

        if not split_X_seq:
            print(f"      {split_name}: NO SAMPLES")
            continue

        X_seq    = np.concatenate(split_X_seq, axis=0)
        X_regime = np.concatenate(split_X_regime, axis=0)
        X_id     = np.concatenate(split_X_id, axis=0)
        Y        = np.concatenate(split_Y, axis=0)
        days_arr = np.concatenate(split_days, axis=0)
        del split_X_seq, split_X_regime, split_X_id, split_Y, split_days

        if split_name == 'train':
            perm = rng.permutation(len(X_seq))
            X_seq    = X_seq[perm]
            X_regime = X_regime[perm]
            X_id     = X_id[perm]
            Y        = Y[perm]
            days_arr = days_arr[perm]

        out_path = os.path.join(OUTPUT_DIR, f'fold_{fold_id}_{split_name}.npz')
        np.savez_compressed(out_path,
                            X_seq=X_seq, X_regime=X_regime, X_id=X_id,
                            Y=Y, days=days_arr)

        upload_to_gcs(out_path)
        print(f"      {split_name}: {X_seq.shape[0]:>8,} samples | "
              f"X_seq={X_seq.shape} X_regime={X_regime.shape} "
              f"Y={Y.shape}")
        del X_seq, X_regime, X_id, Y, days_arr

meta_path = os.path.join(OUTPUT_DIR, 'meta.npz')
np.savez(meta_path,
         n_tickers=N_TICKERS,
         n_asset_features=N_ASSET_FEATURES,
         n_regime_features=N_REGIME_FEATURES,
         n_target_horizons=len(TARGET_HORIZONS),
         target_horizons=TARGET_HORIZONS,
         time_steps=TIME_STEPS,
         regime_schedule=regime_schedule,
         regime_names=np.array([REGIMES[r]['name'] for r in sorted(REGIMES.keys())]),
         fold_count=len(FOLD_CONFIGS),
         asset_col_names=ASSET_COL_NAMES,
         regime_col_names=REGIME_COL_NAMES,
         target_col_names=TARGET_COL_NAMES,
         feat_scaling_med=feat_med,
         feat_scaling_iqr=feat_iqr,
         regime_scaling_med=regime_med,
         regime_scaling_iqr=regime_iqr)
upload_to_gcs(meta_path)
print(f"\n    Metadata saved → {meta_path} + GCS")


# ==============================================================================
# DIAGNOSTICS & CHARTS
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  DIAGNOSTICS")
print(f"{'=' * W}")

# Load back fold 1 train to verify
check = np.load(os.path.join(OUTPUT_DIR, 'fold_1_train.npz'))
print(f"\n  Fold 1 train verification:")
print(f"    X_seq:    {check['X_seq'].shape}  dtype={check['X_seq'].dtype}")
print(f"    X_regime: {check['X_regime'].shape}  dtype={check['X_regime'].dtype}")
print(f"    X_id:     {check['X_id'].shape}  unique={np.unique(check['X_id']).shape[0]} tickers")
print(f"    Y:        {check['Y'].shape}  dtype={check['Y'].dtype}")
print(f"    Y NaN:    {np.isnan(check['Y']).sum()}")
print(f"    Y stats:")
for j, h in enumerate(TARGET_HORIZONS):
    vals = check['Y'][:, j]
    print(f"      fwd_ret_{h}: mean={vals.mean()*100:.4f}%, "
          f"std={vals.std()*100:.4f}%, "
          f"pos={100*(vals>0).mean():.1f}%, "
          f"min={vals.min()*100:.3f}%, max={vals.max()*100:.3f}%")

# Feature stats
print(f"\n    Feature statistics (X_seq last timestep):")
last_step = check['X_seq'][:, -1, :]
for j in range(min(N_ASSET_FEATURES, len(ASSET_COL_NAMES))):
    vals = last_step[:, j]
    print(f"      {ASSET_COL_NAMES[j]:>18}: mean={vals.mean():+.3f}, "
          f"std={vals.std():.3f}, "
          f"[{vals.min():+.2f}, {vals.max():+.2f}]")

# Regime feature stats
print(f"\n    Regime feature statistics:")
for j in range(min(N_REGIME_FEATURES, len(REGIME_COL_NAMES))):
    vals = check['X_regime'][:, j]
    print(f"      {REGIME_COL_NAMES[j]:>22}: mean={vals.mean():+.3f}, "
          f"std={vals.std():.3f}")

# Charts
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('RA-MoE Synthetic Data — Diagnostics', fontsize=14, fontweight='bold')

# [0,0] Price paths with regime coloring
ax = axes[0, 0]
show_tickers = [0, 5, 15, 25]
colors_regime = {0: '#2E7D32', 1: '#1565C0', 2: '#C62828'}
for t_id in show_tickers:
    ax.plot(all_closes_for_regime[::BARS_PER_DAY, t_id], alpha=0.6,
            label=f'Ticker {t_id}', linewidth=0.8)
for d in range(0, TRADING_DAYS, REGIME_PERIOD):
    rid = regime_schedule[d]
    ax.axvspan(d, min(d + REGIME_PERIOD, TRADING_DAYS),
               alpha=0.08, color=colors_regime[rid])
ax.set_xlabel('Trading day')
ax.set_ylabel('Close price')
ax.set_title('Synthetic price paths (colored by regime)')
ax.legend(fontsize=7, loc='upper left')

# [0,1] Regime schedule
ax = axes[0, 1]
for d in range(TRADING_DAYS):
    rid = regime_schedule[d]
    ax.barh(0, 1, left=d, color=colors_regime[rid], height=0.8)
ax.set_yticks([])
ax.set_xlabel('Trading day')
ax.set_title('Regime schedule')
for rid, rinfo in REGIMES.items():
    ax.barh(-1, 0, color=colors_regime[rid], label=rinfo['name'])
ax.legend(fontsize=8)
ax.set_ylim(-0.5, 0.5)

# Fold boundaries
for fc in FOLD_CONFIGS:
    ax.axvline(fc['test_start'], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(fc['test_start'], 0.4, f"F{fc['fold']}",
            fontsize=7, color='red', ha='center')

# [1,0] Target distribution per horizon
ax = axes[1, 0]
for j, h in enumerate(TARGET_HORIZONS):
    vals = check['Y'][:, j] * 100
    ax.hist(vals, bins=100, alpha=0.5, label=f'{h}-min', density=True)
ax.set_xlabel('Forward return (%)')
ax.set_ylabel('Density')
ax.set_title('Target distribution (fold 1 train)')
ax.legend(fontsize=8)
ax.axvline(0, color='black', linewidth=0.5)

# [1,1] Regime features over time (sampled)
ax = axes[1, 1]
sample_regime = regime_features[::BARS_PER_DAY]
for j, name in enumerate(['regime_rvol_60', 'regime_trend_60', 'regime_autocorr_60']):
    idx = REGIME_COL_NAMES.index(name) if name in REGIME_COL_NAMES else j
    vals = sample_regime[:, idx]
    ax.plot(vals, label=name.replace('regime_', ''), alpha=0.7, linewidth=0.8)
ax.set_xlabel('Trading day')
ax.set_title('Regime features over time (daily sample)')
ax.legend(fontsize=7)
for d in range(0, TRADING_DAYS, REGIME_PERIOD):
    rid = regime_schedule[d]
    ax.axvspan(d, min(d + REGIME_PERIOD, TRADING_DAYS),
               alpha=0.06, color=colors_regime[rid])

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, 'data_diagnostics.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
upload_to_gcs(chart_path)
print(f"\n  Chart saved → {chart_path} + GCS")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# SUMMARY
# ==============================================================================
elapsed = time.time() - t_total
print(f"\n{'=' * W}")
print(f"  DATA PREPARATION COMPLETE")
print(f"{'=' * W}")
print(f"  Local cache:        {OUTPUT_DIR}")
print(f"  GCS:                gs://{GCS_BUCKET}/{GCS_DATA_PREFIX}/")
print(f"  Files generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1e6
    print(f"    {f:>35}  {size_mb:>7.1f} MB")
print(f"  Total time:         {elapsed:.0f}s")
print(f"")
print(f"  All artifacts persisted to GCS. Safe to restart runtime.")
print(f"  Next: run ra_moe_02_train.py to train the model on these datasets.")
print(f"{'=' * W}")
