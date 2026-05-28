"""
time_machine_score.py — Scoring vs Training Prediction Comparator
=================================================================
Re-scores historical timestamps using BOTH the training and scoring data
pipelines, then compares against training-time result CSVs in GCS.

This validates:
  1. Training pipeline fidelity — can we reproduce training predictions exactly?
  2. Scoring pipeline fidelity — does the live scoring pipeline match training?

If training-mode replay is EXACT but scoring-mode diverges, the scoring
pipeline has a data processing mismatch vs training.

Usage:
  python time_machine_score.py 2025-11-15            # both experiments, all tickers, all seeds
  python time_machine_score.py 2025-11-15 MARA KTOS  # specific tickers
  python time_machine_score.py 2025-11-15 --exp 97   # Exp-97 only
  python time_machine_score.py 2025-11-15 --exp 103  # Exp-103 only
  python time_machine_score.py --deployed-only        # only deployed ticker+seed pairs
  python time_machine_score.py 2025-11-15 --deployed-only              # deployed seeds, specific date
  python time_machine_score.py 2025-11-15 --deployed-only --exp 97     # deployed Exp-97 only
"""

import sys
import os
import io
import json
import gc
import warnings
import subprocess

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    BatchNormalization, Concatenate, Embedding, Flatten,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except Exception:
    pass

# ==============================================================================
# CONSTANTS
# ==============================================================================
PST = pytz.timezone('America/Los_Angeles')
BUCKET_NAME = "time-series-bucket-ys"
GCS_INFERENCE_DIR = f"gs://{BUCKET_NAME}/Data/Inference"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
ROC_HORIZONS = [5, 30, 60, 120]
TIME_STEPS = 240
EMBED_DIM = 16
WARMUP_DAYS = 30
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct', 'bsize', 'asize', 'Total_Vol']
ASSET_COLS = ['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression', 'ATR_14', 'Dist_to_PM_High'] + \
             [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS

gcs_client = gcs_storage.Client()

# ==============================================================================
# EXPERIMENT CONFIGS
# ==============================================================================
def _build_context_cols(context_assets, has_smh_rsi):
    cols = []
    for t in context_assets:
        cols.append(f'{t}_Anchor_ROC')
        for h in ROC_HORIZONS:
            cols.append(f'{t}_ROC_{h}m')
    if has_smh_rsi:
        cols.append('SMH_RSI')
    cols.append('Time_Pos')
    return cols


EXP_97 = {
    'name': 'Exp-97 (Crypto)',
    'model_id': '97_MICRO',
    'context_assets': ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD'],
    'train_teachers': [
        'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR', 'COIN',
        'BITF', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'SMLR',
    ],
    'ticker_file_map': {'BITF': 'KEEL'},
    'seeds': list(range(9701, 9716)),
    'has_smh_rsi': True,
    'results_path': 'Algos/Results_all_v2/Universal_Exp97_MICRO',
    'scoring_tickers': [
        'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR', 'COIN',
        'BITF', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'SMLR',
        'SMCI', 'VRT', 'TSLA',
    ],
}
EXP_97['context_cols'] = _build_context_cols(EXP_97['context_assets'], EXP_97['has_smh_rsi'])
EXP_97['asset_index'] = {t: i for i, t in enumerate(EXP_97['train_teachers'])}
EXP_97['asset_index']['KEEL'] = EXP_97['asset_index']['BITF']
EXP_97['n_assets'] = len(EXP_97['train_teachers']) + 1
EXP_97['other_idx'] = len(EXP_97['train_teachers'])
EXP_97['deployed_seeds'] = {
    'MIGI': 9714, 'KEEL': 9712, 'HUT': 9703, 'CIFR': 9712, 'WULF': 9710,
    'BTBT': 9711, 'RIOT': 9702, 'IREN': 9701, 'CLSK': 9707, 'APLD': 9715,
    'MSTR': 9711, 'SMCI': 9706, 'COIN': 9705, 'VRT': 9709,
}

EXP_103 = {
    'name': 'Exp-103 (Defense)',
    'model_id': '103_DEFENSE',
    'context_assets': ['ITA', 'VIX', 'TLT', 'IWM'],
    'train_teachers': [
        'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS', 'LUNR', 'SPIR', 'JOBY', 'NNDM', 'MVIS',
    ],
    'ticker_file_map': {},
    'seeds': list(range(10301, 10316)),
    'has_smh_rsi': False,
    'results_path': 'Algos/Results_all_v2/Universal_Exp103_DEFENSE',
    'scoring_tickers': [
        'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS', 'LUNR', 'SPIR', 'JOBY', 'NNDM', 'MVIS',
        'GD', 'LMT', 'NOC', 'RTX', 'LHX', 'GE', 'RCAT', 'ACHR',
    ],
}
EXP_103['context_cols'] = _build_context_cols(EXP_103['context_assets'], EXP_103['has_smh_rsi'])
EXP_103['asset_index'] = {t: i for i, t in enumerate(EXP_103['train_teachers'])}
EXP_103['n_assets'] = len(EXP_103['train_teachers']) + 1
EXP_103['other_idx'] = len(EXP_103['train_teachers'])
EXP_103['deployed_seeds'] = {
    'LUNR': 10315, 'RCAT': 10306, 'SPIR': 10301, 'ASTS': 10305, 'KTOS': 10301,
    'ACHR': 10305, 'RKLB': 10313, 'JOBY': 10305, 'NNDM': 10308, 'AVAV': 10302,
    'PLTR': 10308, 'MVIS': 10312, 'LMT': 10306,
}


# ==============================================================================
# DATA LOADING
# ==============================================================================
def _load_raw_master(ticker, exp_cfg, target_date):
    """Load raw master_inference parquet with window around target date.
    Returns raw df before heal — caller decides heal mode."""
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = exp_cfg['ticker_file_map'].get(clean, clean)
    uri = f"{GCS_INFERENCE_DIR}/{file_ticker}_master_inference.parquet"

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
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
        }
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        df = df[df.index.notna()].sort_index()

        target_ts = pd.Timestamp(target_date, tz='UTC')
        window_start = target_ts - pd.Timedelta(days=WARMUP_DAYS)
        window_end = target_ts + pd.Timedelta(days=1)
        df = df[(df.index >= window_start) & (df.index <= window_end)]
        return df if not df.empty else None

    except Exception as e:
        print(f"      FAILED to load {ticker}: {e}")
        return None


def _apply_micro_cols(df, ticker, exp_cfg, mode):
    """Apply MICRO_COLS fill based on mode."""
    if mode == 'training':
        for c in MICRO_COLS:
            if c not in df.columns:
                df[c] = 0.0
    else:
        if ticker in exp_cfg['context_assets'] or ticker == 'SPY':
            for c in MICRO_COLS:
                if c not in df.columns:
                    df[c] = 0.0
    return df


def _heal(df, mode):
    """Heal data. training=no resample, scoring=with resample."""
    if mode == 'scoring':
        df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# CONTEXT BUILDER
# ==============================================================================
def build_market_context(exp_cfg, target_date, mode):
    frames = []
    for ticker in exp_cfg['context_assets']:
        df = _load_raw_master(ticker, exp_cfg, target_date)
        if df is None:
            continue
        df = _apply_micro_cols(df, ticker, exp_cfg, mode)
        df = df.sort_index()
        df = _heal(df, mode)

        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        if exp_cfg['has_smh_rsi'] and ticker == 'SMH':
            rsi = ta.rsi(df['Close'], length=14)
            cols['SMH_RSI'] = rsi.fillna(50) / 100.0 if rsi is not None else pd.Series(0.5, index=df.index)

        frames.append(pd.DataFrame(cols, index=df.index))
        del df; gc.collect()

    if not frames:
        raise RuntimeError("No context assets loaded.")
    return pd.concat(frames, axis=1).ffill().fillna(0)


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def engineer_asset(ticker, market_df, exp_cfg, target_date, mode):
    df = _load_raw_master(ticker, exp_cfg, target_date)
    if df is None:
        return None
    df = _apply_micro_cols(df, ticker, exp_cfg, mode)
    df = df.sort_index()
    df = _heal(df, mode)

    daily_c = df['Close'].resample('D').last().shift(1).reindex(df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c
    df['PM_High'] = df.groupby(df.index.date)['High'].cummax()
    df['Dist_to_PM_High'] = (df['Close'] - df['PM_High']) / (df['PM_High'] + 1e-9)

    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_60'] = ta.rsi(df['Close'], length=60)
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(50).mean().clip(lower=1e-9)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Compression'] = (df['High'] - df['Low']) / (df['ATR_14'] + 1e-9)
    for h in ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)

    df = df.join(market_df, how='left').ffill().fillna(0)

    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ==============================================================================
# SCALING + MODEL
# ==============================================================================
def load_stats(model_id):
    blob = gcs_client.bucket(BUCKET_NAME).blob(
        f"Data/Inference_v2/scaling_stats_experiment_{model_id}.json")
    return json.loads(blob.download_as_text())


def apply_scaling(df, stats, context_cols):
    df_s = df.copy()
    for col in ASSET_COLS + context_cols:
        if col == 'Time_Pos':
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med']) / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.4):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])


def build_model(n_assets, n_context_cols):
    in_a = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c = Input(shape=(n_context_cols,), name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32, name='asset_id')

    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(BatchNormalization()(in_c))
    e = Embedding(n_assets, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)

    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


_model_cache = {}

def get_model(seed, exp_cfg):
    key = (seed, exp_cfg['model_id'])
    if key in _model_cache:
        return _model_cache[key]
    blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{exp_cfg['model_id']}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{exp_cfg['model_id']}.h5"
    if not os.path.exists(local_path):
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
    model = build_model(exp_cfg['n_assets'], len(exp_cfg['context_cols']))
    model.load_weights(local_path)
    _model_cache[key] = model
    return model


# ==============================================================================
# TRAINING CSV LOADER
# ==============================================================================
def load_training_csv(ticker, seed, exp_cfg):
    lookup = 'BITF' if ticker == 'KEEL' else ticker
    csv_path = f"{exp_cfg['results_path']}/{lookup}/Seed_{seed}_Results.csv"
    try:
        blob = gcs_client.bucket(BUCKET_NAME).blob(csv_path)
        if not blob.exists():
            return None
        csv_data = blob.download_as_text()
        df = pd.read_csv(io.StringIO(csv_data), index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        return df
    except Exception:
        return None


def find_training_dates(exp_cfg, n_tickers=3):
    """Find dates that have training results. Returns sorted list of dates."""
    all_dates = set()
    checked = 0
    for ticker in exp_cfg['scoring_tickers']:
        if checked >= n_tickers:
            break
        seed = exp_cfg['seeds'][0]
        df = load_training_csv(ticker, seed, exp_cfg)
        if df is not None and not df.empty:
            dates = set(df.index.tz_convert(PST).date)
            all_dates.update(dates)
            checked += 1
    return sorted(all_dates)


# ==============================================================================
# SCORE ONE EXPERIMENT
# ==============================================================================
def score_experiment(exp_cfg, target_date, ticker_filter=None, deployed_only=False):
    context_cols = exp_cfg['context_cols']
    deployed = exp_cfg.get('deployed_seeds', {})

    if deployed_only:
        tickers = ticker_filter or list(deployed.keys())
        tickers = [t for t in tickers if t in deployed]
    else:
        tickers = ticker_filter or exp_cfg['scoring_tickers']
        tickers = [t for t in tickers if t in exp_cfg['scoring_tickers']]

    if not tickers:
        print(f"   No matching tickers for {exp_cfg['name']}")
        return []

    if deployed_only:
        needed_seeds = set(deployed[t] for t in tickers)
        print(f"\n   DEPLOYED-ONLY mode: {len(tickers)} tickers, {len(needed_seeds)} unique seeds")
    else:
        needed_seeds = None

    print(f"\n   Loading scaling stats for {exp_cfg['model_id']}...")
    stats = load_stats(exp_cfg['model_id'])
    print(f"   Loaded {len(stats)} feature stats")

    # Build context in BOTH modes
    print(f"   Building market context (training mode — no resample)...")
    mkt_train = build_market_context(exp_cfg, target_date, mode='training')
    print(f"      {len(mkt_train)} rows")

    print(f"   Building market context (scoring mode — with resample)...")
    mkt_score = build_market_context(exp_cfg, target_date, mode='scoring')
    print(f"      {len(mkt_score)} rows")

    # Pre-load models (only seeds that have training CSVs)
    seeds_to_check = sorted(needed_seeds) if deployed_only else exp_cfg['seeds']
    seeds_with_results = set()
    print(f"   Checking which seeds have training results...")
    sample_ticker = tickers[0]
    for seed in seeds_to_check:
        df_csv = load_training_csv(sample_ticker, seed, exp_cfg)
        if df_csv is not None:
            seeds_with_results.add(seed)
    print(f"   Found {len(seeds_with_results)}/{len(seeds_to_check)} seeds with results")

    if not seeds_with_results:
        print(f"   No training results found — cannot compare")
        return []

    print(f"   Loading {len(seeds_with_results)} models...")
    for seed in sorted(seeds_with_results):
        try:
            get_model(seed, exp_cfg)
        except Exception as e:
            print(f"      Seed {seed}: FAILED ({e})")
            seeds_with_results.discard(seed)

    results = []
    target_d = pd.Timestamp(target_date).date()

    for ticker in tickers:
        asset_id = exp_cfg['asset_index'].get(ticker, exp_cfg['other_idx'])
        ticker_seeds = [deployed[ticker]] if deployed_only else sorted(seeds_with_results)
        ticker_seeds = [s for s in ticker_seeds if s in seeds_with_results]
        print(f"\n   {ticker} (asset_id={asset_id}, seeds={ticker_seeds})")

        # Engineer features in BOTH modes
        df_t = engineer_asset(ticker, mkt_train, exp_cfg, target_date, mode='training')
        df_s = engineer_asset(ticker, mkt_score, exp_cfg, target_date, mode='scoring')

        if df_t is None:
            print(f"      SKIP: no data (training mode)")
            continue

        # Scale
        df_t_scaled = apply_scaling(df_t, stats, context_cols)
        df_s_scaled = apply_scaling(df_s, stats, context_cols) if df_s is not None else None

        # Find Valid_Row bars on target date (minutes 360-390 PST)
        df_pst = df_t_scaled.index.tz_convert(PST)
        mins = df_pst.hour * 60 + df_pst.minute
        valid_mask = (df_pst.date == target_d) & (mins >= 360) & (mins <= 390)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            print(f"      No Valid_Row bars on {target_date}")
            continue

        print(f"      {len(valid_indices)} Valid_Row bars found")

        for seed in ticker_seeds:
            # Load training CSV for this ticker+seed
            df_csv = load_training_csv(ticker, seed, exp_cfg)
            if df_csv is None or 'Prob_Spike' not in df_csv.columns:
                continue

            model = get_model(seed, exp_cfg)

            for idx in valid_indices:
                if idx < TIME_STEPS:
                    continue

                bar_ts = df_t_scaled.index[idx]

                # Find training prob
                train_prob = None
                if bar_ts in df_csv.index:
                    train_prob = float(df_csv.loc[bar_ts, 'Prob_Spike'])
                else:
                    diffs = abs(df_csv.index - bar_ts)
                    if len(diffs) > 0:
                        nearest_i = diffs.argmin()
                        if diffs[nearest_i] <= pd.Timedelta(minutes=1):
                            train_prob = float(df_csv.iloc[nearest_i]['Prob_Spike'])

                if train_prob is None:
                    continue

                # Score in TRAINING mode
                XA = df_t_scaled[ASSET_COLS].values
                XC = df_t_scaled[context_cols].values
                XA_b = XA[idx - TIME_STEPS:idx][np.newaxis]
                XC_b = XC[idx - 1][np.newaxis]
                XID_b = np.array([[asset_id]], dtype=np.int32)

                prob_train = float(model(
                    {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
                    training=False
                ).numpy().flatten()[0])

                # Score in SCORING mode
                prob_score = None
                if df_s_scaled is not None and bar_ts in df_s_scaled.index:
                    s_idx = df_s_scaled.index.get_loc(bar_ts)
                    if s_idx >= TIME_STEPS:
                        XA_s = df_s_scaled[ASSET_COLS].values
                        XC_s = df_s_scaled[context_cols].values
                        XA_sb = XA_s[s_idx - TIME_STEPS:s_idx][np.newaxis]
                        XC_sb = XC_s[s_idx - 1][np.newaxis]
                        prob_score = float(model(
                            {'asset_in': XA_sb, 'context_in': XC_sb, 'asset_id': XID_b},
                            training=False
                        ).numpy().flatten()[0])

                delta_t = abs(prob_train - train_prob)
                delta_s = abs(prob_score - train_prob) if prob_score is not None else None

                if delta_t < 0.0001:
                    match = "EXACT"
                elif delta_t < 0.001:
                    match = "~OK"
                elif delta_t < 0.01:
                    match = "CLOSE"
                else:
                    match = "MISMATCH"

                results.append({
                    'ticker': ticker,
                    'seed': seed,
                    'bar_ts': bar_ts,
                    'train_prob': train_prob,
                    'replay_t': prob_train,
                    'replay_s': prob_score,
                    'delta_t': delta_t,
                    'delta_s': delta_s,
                    'match': match,
                })

    return results


# ==============================================================================
# DASHBOARD
# ==============================================================================
def print_dashboard(results, exp_cfg, target_date):
    W = 110
    print(f"\n{'=' * W}")
    print(f"  TIME MACHINE — {exp_cfg['name']}   Date: {target_date}")
    print(f"{'=' * W}")

    if not results:
        print(f"  NO RESULTS — no matching bars found in training CSVs")
        print(f"{'=' * W}")
        return

    print(f"  {'Ticker':<6}  {'Seed':>5}  {'Bar (PST)':<18}  "
          f"{'Train':>8}  {'Rply(T)':>8}  {'Dlt(T)':>10}  "
          f"{'Rply(S)':>8}  {'Dlt(S)':>10}  {'Match':<10}")
    print(f"  {'─' * 6}  {'─' * 5}  {'─' * 18}  "
          f"{'─' * 8}  {'─' * 8}  {'─' * 10}  "
          f"{'─' * 8}  {'─' * 10}  {'─' * 10}")

    for r in results:
        bar_str = r['bar_ts'].astimezone(PST).strftime('%m-%d %H:%M')
        rs = f"{r['replay_s']:.4f}" if r['replay_s'] is not None else "N/A"
        ds = f"{r['delta_s']:+.6f}" if r['delta_s'] is not None else "N/A"

        print(f"  {r['ticker']:<6}  {r['seed']:>5}  {bar_str:<18}  "
              f"{r['train_prob']:>8.4f}  {r['replay_t']:>8.4f}  {r['delta_t']:>+10.6f}  "
              f"{rs:>8}  {ds:>10}  {r['match']:<10}")

    print(f"{'=' * W}")

    exact = sum(1 for r in results if r['match'] == 'EXACT')
    ok = sum(1 for r in results if r['match'] == '~OK')
    close = sum(1 for r in results if r['match'] == 'CLOSE')
    mismatch = sum(1 for r in results if r['match'] == 'MISMATCH')
    total = len(results)

    print(f"  {total} comparisons: {exact} EXACT, {ok} ~OK (<0.001), "
          f"{close} CLOSE (<0.01), {mismatch} MISMATCH (>=0.01)")

    if mismatch > 0:
        mm = [r for r in results if r['match'] == 'MISMATCH']
        avg_d = np.mean([r['delta_t'] for r in mm])
        max_d = max(r['delta_t'] for r in mm)
        print(f"  MISMATCH(Training): avg={avg_d:.6f}, max={max_d:.6f}")

        sm = [r for r in mm if r['delta_s'] is not None]
        if sm:
            avg_ds = np.mean([r['delta_s'] for r in sm])
            print(f"  MISMATCH(Scoring):  avg={avg_ds:.6f} "
                  f"({'WORSE' if avg_ds > avg_d else 'BETTER'} than training mode)")

    print(f"{'=' * W}")
    print()
    print("  Legend:")
    print("    Train    = Prob_Spike from training-time CSV (ground truth)")
    print("    Rply(T)  = Re-scored with TRAINING pipeline (no resample, unconditional MICRO)")
    print("    Rply(S)  = Re-scored with SCORING pipeline (resample, conditional MICRO)")
    print("    Dlt(T)   = |Rply(T) - Train|  — should be ~0 if data unchanged since training")
    print("    Dlt(S)   = |Rply(S) - Train|  — shows scoring vs training divergence")
    print("    EXACT < 0.0001 | ~OK < 0.001 | CLOSE < 0.01 | MISMATCH >= 0.01")
    print()


# ==============================================================================
# CONFIG — edit these before running
# ==============================================================================
TARGET_DATE    = '2025-11-15'       # date to score, or None to auto-detect
DEPLOYED_ONLY  = True               # True = only deployed ticker+seed pairs
EXP_FILTER     = None               # None = both, '97' = Exp-97 only, '103' = Exp-103 only
TICKER_FILTER  = None               # None = all, or ['MARA', 'KTOS'] etc.

# ==============================================================================
# RUN
# ==============================================================================
experiments = {}
if EXP_FILTER is None or EXP_FILTER == '97':
    experiments['97'] = EXP_97
if EXP_FILTER is None or EXP_FILTER == '103':
    experiments['103'] = EXP_103

if TARGET_DATE is None:
    print("No date specified — auto-detecting from training CSVs...")
    first_exp = list(experiments.values())[0]
    dates = find_training_dates(first_exp)
    if dates:
        TARGET_DATE = str(dates[-30]) if len(dates) > 30 else str(dates[len(dates) // 2])
        print(f"   Auto-selected: {TARGET_DATE}")
        print(f"   Range: {dates[0]} to {dates[-1]} ({len(dates)} dates)")
    else:
        raise RuntimeError("No training result CSVs found.")

print(f"\n{'=' * 70}")
print(f"  TIME MACHINE SCORER")
print(f"  Date: {TARGET_DATE}")
print(f"  Experiments: {', '.join(experiments.keys())}")
print(f"  Mode: {'DEPLOYED-ONLY' if DEPLOYED_ONLY else 'ALL SEEDS'}")
if TICKER_FILTER:
    print(f"  Tickers: {', '.join(TICKER_FILTER)}")
print(f"  Warmup: {WARMUP_DAYS} days")
print(f"  Valid_Row window: minutes 360-390 PST (6:00-6:30 AM)")
print(f"{'=' * 70}")

for exp_key, exp_cfg in experiments.items():
    print(f"\n{'#' * 70}")
    print(f"  {exp_cfg['name']}")
    if DEPLOYED_ONLY:
        ds = exp_cfg.get('deployed_seeds', {})
        print(f"  Deployed: {len(ds)} tickers — {', '.join(f'{t}:{s}' for t,s in ds.items())}")
    print(f"{'#' * 70}")

    results = score_experiment(exp_cfg, TARGET_DATE, TICKER_FILTER, deployed_only=DEPLOYED_ONLY)
    print_dashboard(results, exp_cfg, TARGET_DATE)
