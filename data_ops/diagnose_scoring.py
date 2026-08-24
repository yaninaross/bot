"""
diagnose_scoring.py — Compare live model scoring vs pre-computed scoring CSVs
==============================================================================
For each deploy ticker, loads the pre-computed Prob_Spike from GCS Results CSVs
and runs live TF inference, then compares probabilities side-by-side.

This identifies whether the backtest PnL gap comes from scoring pipeline
differences vs. the optimizer using different data.

Colab-ready (no if __name__).
"""

import os
import io
import gc
import sys
import json
import subprocess
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]", "pandas_ta",
                           "google-cloud-storage", "pyarrow", "gcsfs",
                           "--quiet"])
    import tensorflow as tf

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2

# ==============================================================================
# CONFIG
# ==============================================================================
TEST_START = '2026-04-01'
TEST_END   = '2026-05-24'

BUCKET_NAME    = "time-series-bucket-ys"
INFERENCE_DIR  = "Data/Inference"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR      = "Data/Inference"
RESULTS_DIR    = "Algos/Results_all"

TIME_STEPS  = 240
EMBED_DIM   = 16
PST = pytz.timezone('America/Los_Angeles')

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

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

TRAIN_TEACHERS = [
    'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
    'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
]
ASSET_INDEX = {t: i for i, t in enumerate(TRAIN_TEACHERS)}
N_ASSETS    = len(TRAIN_TEACHERS) + 1
OTHER_IDX   = len(TRAIN_TEACHERS)

DEPLOY_CONFIGS = [
    {'ticker': 'WULF', 'model_id': '160_XSECTOR', 'seed': 16005, 'entry_minute': 385},
    {'ticker': 'BBAI', 'model_id': '160_XSECTOR', 'seed': 16010, 'entry_minute': 385},
    {'ticker': 'ALAB', 'model_id': '160_XSECTOR', 'seed': 16004, 'entry_minute': 385},
    {'ticker': 'PLUG', 'model_id': '160_XSECTOR', 'seed': 16011, 'entry_minute': 375},
    {'ticker': 'LUNR', 'model_id': '152_XSECTOR', 'seed': 15214, 'entry_minute': 375},
]

gcs_client = gcs_storage.Client()

# ==============================================================================
# DATA LOADING (same as backtest_portfolio_v2.py)
# ==============================================================================
_master_cache = {}

def load_master(ticker, start_date=None):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    cache_key = (clean, start_date)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
    try:
        blob_path = f"{INFERENCE_DIR}/{clean}_master_inference.parquet"
        bucket = gcs_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_path)
        raw = blob.download_as_bytes()
        df = pd.read_parquet(io.BytesIO(raw))
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
        if start_date:
            cutoff = pd.Timestamp(start_date, tz='UTC') - pd.Timedelta(days=30)
            df = df[df.index >= cutoff]
        _master_cache[cache_key] = df
        return df
    except Exception as e:
        print(f"    load_master({ticker}): FAILED — {e}")
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
# FEATURE ENGINEERING (same as backtest_portfolio_v2.py)
# ==============================================================================
_market_context_cache = None

def build_market_context(start_date=None):
    global _market_context_cache
    if _market_context_cache is not None:
        return _market_context_cache
    print("  Building market context (SPY, VIX, TLT, IWM)...")
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker, start_date=start_date)
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
    result = pd.concat(frames, axis=1, sort=True).ffill().fillna(0)
    _market_context_cache = result
    print(f"    Market context: {len(result)} rows")
    return result


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
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ==============================================================================
# SCALING
# ==============================================================================
_stats_cache = {}

def load_scaling_stats(model_id):
    if model_id in _stats_cache:
        return _stats_cache[model_id]
    path = f"{STATS_DIR}/scaling_stats_experiment_{model_id}.json"
    blob = gcs_client.bucket(BUCKET_NAME).blob(path)
    raw = json.loads(blob.download_as_bytes())
    _stats_cache[model_id] = raw
    print(f"    Scaling stats loaded: {model_id} ({len(raw)} features)")
    return raw


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
# MODEL ARCHITECTURE (same as backtest_portfolio_v2.py)
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


def build_model(n_context_cols, n_assets, n_asset_cols=None):
    n_ac = n_asset_cols or len(ASSET_COLS)
    in_a  = Input(shape=(TIME_STEPS, n_ac), name='asset_in')
    in_c  = Input(shape=(n_context_cols,), name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32, name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Embedding(n_assets, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


_model_cache = {}

def get_model(model_id, seed):
    key = (model_id, seed)
    if key in _model_cache:
        return _model_cache[key]
    blob_path  = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{model_id}.h5"
    if not os.path.exists(local_path):
        print(f"    Downloading model: S{seed}_M{model_id}")
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
    model = build_model(len(CONTEXT_COLS), N_ASSETS, n_asset_cols=len(ASSET_COLS))
    model.load_weights(local_path)
    _model_cache[key] = model
    return model


# ==============================================================================
# LOAD PRE-COMPUTED SCORING CSV
# ==============================================================================
def load_precomputed_scores(ticker, model_id, seed):
    """Load Prob_Spike from the scoring Results CSV produced during training."""
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    blob_path = f"{RESULTS_DIR}/Universal_Exp{model_id}/{clean}/Seed_{seed}_Results.csv"
    bucket = gcs_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        print(f"    WARNING: No pre-computed CSV for {ticker} {model_id} seed {seed}")
        return None
    df = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(PST)
    return df['Prob_Spike'].sort_index()


# ==============================================================================
# LIVE SCORING — score ALL entry bars (not just above threshold)
# ==============================================================================
def live_score_all_entries(ticker, cfg, df_scaled):
    """Score all entry-minute bars in TEST period. Return Series(date → prob)."""
    entry_minute = cfg['entry_minute']
    model_id = cfg['model_id']
    seed = cfg['seed']
    asset_id = ASSET_INDEX.get(ticker, OTHER_IDX)

    test_start_utc = pd.Timestamp(TEST_START, tz='UTC')
    test_end_utc   = pd.Timestamp(TEST_END, tz='UTC') + pd.Timedelta(days=1)
    test_mask = (df_scaled.index >= test_start_utc) & (df_scaled.index < test_end_utc)
    df_test = df_scaled[test_mask]

    df_test_pst = df_test.index.tz_convert(PST)
    mins = df_test_pst.hour * 60 + df_test_pst.minute
    entry_mask = mins == entry_minute

    entry_positions = []
    entry_timestamps = []
    for i in range(len(df_test)):
        if not entry_mask[i]:
            continue
        global_idx = df_scaled.index.get_loc(df_test.index[i])
        if global_idx < TIME_STEPS:
            continue
        entry_positions.append(global_idx)
        entry_timestamps.append(df_test.index[i])

    if not entry_positions:
        return pd.Series(dtype=float)

    XA_batch = np.stack([
        df_scaled[ASSET_COLS].values[idx - TIME_STEPS:idx]
        for idx in entry_positions
    ])
    XC_batch = np.stack([
        df_scaled[CONTEXT_COLS].values[idx - 1]
        for idx in entry_positions
    ])
    XID_batch = np.full((len(entry_positions), 1), asset_id, dtype=np.int32)

    model = get_model(model_id, seed)
    probs = model.predict(
        {'asset_in': XA_batch, 'context_in': XC_batch, 'asset_id': XID_batch},
        batch_size=64, verbose=0
    ).flatten()

    pst_timestamps = [ts.tz_convert(PST) for ts in entry_timestamps]
    return pd.Series(probs, index=pst_timestamps, name='live_prob')


# ==============================================================================
# MAIN COMPARISON
# ==============================================================================
print("=" * 100)
print("  SCORING DIAGNOSTIC — Live vs Pre-computed Probability Comparison")
print("=" * 100)

market_df = build_market_context(start_date=TEST_START)

all_comparisons = []

for cfg in DEPLOY_CONFIGS:
    ticker = cfg['ticker']
    model_id = cfg['model_id']
    seed = cfg['seed']
    entry_minute = cfg['entry_minute']

    print(f"\n{'─' * 80}")
    print(f"  {ticker}  model={model_id}  seed={seed}  entry={entry_minute}")
    print(f"{'─' * 80}")

    # Load pre-computed scores
    precomp = load_precomputed_scores(ticker, model_id, seed)
    if precomp is None:
        continue

    # Live scoring
    df_raw = load_master(ticker, start_date=TEST_START)
    if df_raw is None:
        continue
    df_raw = df_raw.sort_index()
    df_raw = heal_data(df_raw)
    df_feat = engineer_features(df_raw.copy(), market_df)
    stats = load_scaling_stats(model_id)
    df_scaled = apply_scaling(df_feat, stats)

    live_probs = live_score_all_entries(ticker, cfg, df_scaled)

    if live_probs.empty:
        print("    No live entry bars found.")
        del df_raw, df_feat, df_scaled; gc.collect()
        continue

    # Align: filter pre-computed to entry_minute bars in TEST period
    test_start_pst = pd.Timestamp(TEST_START, tz=PST)
    test_end_pst   = pd.Timestamp(TEST_END, tz=PST) + pd.Timedelta(days=1)
    precomp_test = precomp[(precomp.index >= test_start_pst) & (precomp.index < test_end_pst)]

    precomp_mins = precomp_test.index.hour * 60 + precomp_test.index.minute
    precomp_entry = precomp_test[precomp_mins == entry_minute]

    # Match by date
    live_dates = pd.Series(live_probs.values,
                           index=live_probs.index.date, name='live')
    precomp_dates = pd.Series(precomp_entry.values,
                              index=precomp_entry.index.date, name='precomp')

    comp = pd.DataFrame({'live': live_dates, 'precomp': precomp_dates}).dropna()

    if comp.empty:
        print(f"    No overlapping dates between live ({len(live_probs)}) and precomp ({len(precomp_entry)})")
        print(f"    Live dates:    {sorted(live_probs.index.date)[:5]}...")
        print(f"    Precomp dates: {sorted(precomp_entry.index.date)[:5]}...")
        del df_raw, df_feat, df_scaled; gc.collect()
        continue

    comp['diff'] = comp['live'] - comp['precomp']
    comp['abs_diff'] = comp['diff'].abs()

    # Summary stats
    print(f"\n    Matched days: {len(comp)}")
    print(f"    Live bars (total, no threshold):    {len(live_probs)}")
    print(f"    Pre-computed bars (entry minute):   {len(precomp_entry)}")
    print(f"\n    Probability comparison:")
    print(f"      Mean live prob:     {comp['live'].mean():.4f}")
    print(f"      Mean precomp prob:  {comp['precomp'].mean():.4f}")
    print(f"      Mean diff (L-P):    {comp['diff'].mean():+.4f}")
    print(f"      Max abs diff:       {comp['abs_diff'].max():.4f}")
    print(f"      Correlation:        {comp['live'].corr(comp['precomp']):.4f}")

    # Threshold analysis
    for thresh in [0.30, 0.40, 0.50, 0.60]:
        live_above = (comp['live'] >= thresh).sum()
        pre_above  = (comp['precomp'] >= thresh).sum()
        both_above = ((comp['live'] >= thresh) & (comp['precomp'] >= thresh)).sum()
        print(f"      Thresh {thresh:.2f}: live={live_above:2d} precomp={pre_above:2d} "
              f"both={both_above:2d}  mismatch={abs(live_above-pre_above):2d}")

    # Day-by-day table (show all days, sorted by date)
    print(f"\n    {'Date':>12}  {'Live':>8}  {'Precomp':>8}  {'Diff':>8}  {'Signal?':>10}")
    print(f"    {'─'*52}")
    prob_thresh_map = {'WULF': 0.50, 'BBAI': 0.50, 'ALAB': 0.50, 'PLUG': 0.50, 'LUNR': 0.40}
    pt = prob_thresh_map.get(ticker, 0.50)
    for date in sorted(comp.index):
        row = comp.loc[date]
        l, p = row['live'], row['precomp']
        sig_live = 'L' if l >= pt else '.'
        sig_pre  = 'P' if p >= pt else '.'
        flag = ' *** MISMATCH' if sig_live != sig_pre else ''
        print(f"    {str(date):>12}  {l:8.4f}  {p:8.4f}  {l-p:+8.4f}  {sig_live}/{sig_pre}{flag}")

    comp['ticker'] = ticker
    all_comparisons.append(comp)

    del df_raw, df_feat, df_scaled; gc.collect()


# ==============================================================================
# AGGREGATE SUMMARY
# ==============================================================================
if all_comparisons:
    all_comp = pd.concat(all_comparisons, ignore_index=False)
    print(f"\n\n{'=' * 100}")
    print(f"  AGGREGATE SUMMARY — {len(all_comp)} total matched bars across {len(all_comparisons)} tickers")
    print(f"{'=' * 100}")
    print(f"  Overall correlation:  {all_comp['live'].corr(all_comp['precomp']):.4f}")
    print(f"  Mean abs difference:  {all_comp['abs_diff'].mean():.4f}")
    print(f"  Max abs difference:   {all_comp['abs_diff'].max():.4f}")
    print(f"  Mean live prob:       {all_comp['live'].mean():.4f}")
    print(f"  Mean precomp prob:    {all_comp['precomp'].mean():.4f}")
    print(f"  Systematic bias:      {all_comp['diff'].mean():+.4f} (positive = live scores higher)")

    # Per-ticker summary
    print(f"\n  Per-ticker summary:")
    print(f"  {'Ticker':>8}  {'N':>4}  {'Corr':>6}  {'MeanDiff':>9}  {'MaxAbsDiff':>10}  {'Signal mismatches':>18}")
    print(f"  {'─'*62}")
    for cfg in DEPLOY_CONFIGS:
        t = cfg['ticker']
        pt = 0.40 if t == 'LUNR' else 0.50
        tc = all_comp[all_comp['ticker'] == t]
        if tc.empty:
            continue
        mismatches = ((tc['live'] >= pt) != (tc['precomp'] >= pt)).sum()
        print(f"  {t:>8}  {len(tc):4d}  {tc['live'].corr(tc['precomp']):6.3f}  "
              f"{tc['diff'].mean():+9.4f}  {tc['abs_diff'].max():10.4f}  {mismatches:18d}")

    # Chart: scatter plot live vs precomputed
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    colors = {'WULF': 'red', 'BBAI': 'blue', 'ALAB': 'green', 'PLUG': 'orange', 'LUNR': 'purple'}
    for t in colors:
        tc = all_comp[all_comp['ticker'] == t]
        if not tc.empty:
            ax1.scatter(tc['precomp'], tc['live'], c=colors[t], label=t, alpha=0.7, s=40)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='perfect match')
    ax1.set_xlabel('Pre-computed Prob')
    ax1.set_ylabel('Live Prob')
    ax1.set_title('Live vs Pre-computed Probabilities')
    ax1.legend()
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)

    ax2 = axes[1]
    ax2.hist(all_comp['diff'], bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--')
    ax2.set_xlabel('Difference (Live - Precomp)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Probability Differences (N={len(all_comp)})')

    plt.tight_layout()
    chart_path = '/tmp/scoring_diagnostic.png'
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    print(f"\n  Chart saved: {chart_path}")
    plt.show()

print(f"\n{'=' * 100}")
print("  DIAGNOSTIC COMPLETE")
print(f"{'=' * 100}")
