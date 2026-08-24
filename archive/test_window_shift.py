"""
Window Shift Test — Apr 27-28  (SCORING MODE)
==============================================
Same as previous test but now MATCHES SCORING PIPELINE:
  - TAIL_DAYS=7 (only 7 days of data, like daily_score_combined.py)
  - NO Volume=0 skip (scoring checks volume at last bar, not entry bar)

Compares against previous test (full history + volume skip) to isolate
whether shorter data window and synthetic bars cause additional false positives.
"""

import sys
import os
import io
import gc
import json
import pytz
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import subprocess
try:
    import pandas_ta as ta
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--quiet"])
    import pandas_ta as ta

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    BatchNormalization, Concatenate, Embedding, Flatten,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from google.cloud import storage

# ==============================================================================
# CONSTANTS
# ==============================================================================
BUCKET_NAME    = "time-series-bucket-ys"
INFERENCE_DIR  = "Data/Inference"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
TIME_STEPS     = 240
EMBED_DIM      = 16
PST = pytz.timezone('America/Los_Angeles')

TICKER_FILE_MAP = {'BITF': 'KEEL'}

# --- SCORING MODE: match daily_score_combined.py behavior ---
TAIL_DAYS       = 7      # scoring loads only 7 days (was 180 in backtester mode)
SKIP_VOLUME_0   = False  # scoring checks volume at last bar, not entry bar

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]
ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

# --- Test dates ---
TEST_DATES = [
    datetime(2026, 4, 27),
    datetime(2026, 4, 28),
]
SHIFT_OFFSETS = list(range(-20, 16))  # -20 through +15 minutes

# ==============================================================================
# EXPERIMENT DEFINITIONS — EXACT COPY from daily_score_combined.py
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
    'model_id': '97_MICRO',
    'label': 'EXP-97 CRYPTO',
    'train_teachers': [
        'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
        'COIN', 'BITF', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'SMLR',
    ],
    'context_assets': ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD'],
    'has_smh_rsi': True,
    'asset_configs': {
        'RIOT': {
            'seed': 9703, 'prob_threshold': 0.490,
            'tp_pct': 0.06, 'sl_pct': 0.04, 'entry_minute': 385,
            'vix_threshold': 10.0, 'tlt_crash': -0.40, 'spy_ignore': 0.10,
        },
        'IREN': {
            'seed': 9715, 'prob_threshold': 0.515,
            'tp_pct': 0.12, 'sl_pct': 0.10, 'entry_minute': 375,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.04,
        },
        'WULF': {
            'seed': 9715, 'prob_threshold': 0.515,
            'tp_pct': 0.17, 'sl_pct': 0.05, 'entry_minute': 360,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.18,
        },
        'CIFR': {
            'seed': 9714, 'prob_threshold': 0.510,
            'tp_pct': 0.13, 'sl_pct': 0.09, 'entry_minute': 385,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.50,
        },
    },
}

EXP_110 = {
    'model_id': '110_MICRO',
    'label': 'EXP-110 CRYPTO',
    'train_teachers': [
        'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
        'COIN', 'BITF', 'HUT', 'CIFR', 'BTBT', 'HOOD',
    ],
    'context_assets': ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD'],
    'has_smh_rsi': True,
    'asset_configs': {
        'BTBT': {
            'seed': 11006, 'prob_threshold': 0.510,
            'tp_pct': 0.15, 'sl_pct': 0.04, 'entry_minute': 375,
            'vix_threshold': 15.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
        },
    },
}

EXP_112 = {
    'model_id': '112_MICRO',
    'label': 'EXP-112 CRYPTO (8%)',
    'train_teachers': [
        'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
        'COIN', 'BITF', 'HUT', 'CIFR', 'BTBT', 'HOOD',
    ],
    'context_assets': ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD'],
    'has_smh_rsi': True,
    'asset_configs': {
        'APLD': {
            'seed': 11204, 'prob_threshold': 0.510,
            'tp_pct': 0.08, 'sl_pct': 0.07, 'entry_minute': 379,
            'vix_threshold': 10.0, 'tlt_crash': -0.50, 'spy_ignore': 0.00,
        },
        'CLSK': {
            'seed': 11205, 'prob_threshold': 0.495,
            'tp_pct': 0.19, 'sl_pct': 0.04, 'entry_minute': 383,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.10,
        },
        'HUT': {
            'seed': 11206, 'prob_threshold': 0.510,
            'tp_pct': 0.19, 'sl_pct': 0.09, 'entry_minute': 370,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
        },
    },
}

EXP_103 = {
    'model_id': '103_DEFENSE',
    'label': 'EXP-103 DEFENSE',
    'train_teachers': [
        'AVAV', 'KTOS', 'PLTR', 'BBAI',
        'RKLB', 'ASTS', 'LUNR', 'SPIR',
        'JOBY', 'NNDM', 'MVIS',
    ],
    'context_assets': ['ITA', 'VIX', 'TLT', 'IWM'],
    'has_smh_rsi': False,
    'asset_configs': {
        'RCAT': {
            'seed': 10304, 'prob_threshold': 0.525,
            'tp_pct': 0.16, 'sl_pct': 0.14, 'entry_minute': 365,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
        },
        'LUNR': {
            'seed': 10311, 'prob_threshold': 0.465,
            'tp_pct': 0.21, 'sl_pct': 0.05, 'entry_minute': 365,
            'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
        },
        'RKLB': {
            'seed': 10308, 'prob_threshold': 0.510,
            'tp_pct': 0.08, 'sl_pct': 0.04, 'entry_minute': 385,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
        },
        'SPIR': {
            'seed': 10310, 'prob_threshold': 0.310,
            'tp_pct': 0.12, 'sl_pct': 0.03, 'entry_minute': 360,
            'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.10,
        },
        'PLTR': {
            'seed': 10311, 'prob_threshold': 0.465,
            'tp_pct': 0.07, 'sl_pct': 0.06, 'entry_minute': 360,
            'vix_threshold': 10.0, 'tlt_crash': -0.20, 'spy_ignore': 0.10,
        },
        'JOBY': {
            'seed': 10307, 'prob_threshold': 0.510,
            'tp_pct': 0.12, 'sl_pct': 0.08, 'entry_minute': 380,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
        },
        'ASTS': {
            'seed': 10308, 'prob_threshold': 0.510,
            'tp_pct': 0.10, 'sl_pct': 0.06, 'entry_minute': 390,
            'vix_threshold': 15.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
        },
        'KTOS': {
            'seed': 10312, 'prob_threshold': 0.510,
            'tp_pct': 0.05, 'sl_pct': 0.04, 'entry_minute': 385,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
        },
    },
}

for _exp in (EXP_97, EXP_110, EXP_112, EXP_103):
    _exp['context_cols'] = _build_context_cols(
        _exp['context_assets'], _exp['has_smh_rsi'])
    _exp['all_cols'] = ASSET_COLS + _exp['context_cols']
    _idx = {t: i for i, t in enumerate(_exp['train_teachers'])}
    if 'BITF' in _idx:
        _idx['KEEL'] = _idx['BITF']
    _exp['asset_index'] = _idx
    _exp['n_assets'] = len(_exp['train_teachers']) + 1
    _exp['other_idx'] = len(_exp['train_teachers'])
    _exp['stats_path'] = f"Data/Inference_v2/scaling_stats_experiment_{_exp['model_id']}.json"

EXPERIMENTS = [EXP_97, EXP_110, EXP_112, EXP_103]

# ==============================================================================
# GCS + DATA LOADING (same as backtesting_combined.py)
# ==============================================================================
try:
    from google.colab import auth
    auth.authenticate_user()
except Exception:
    pass

gcs_client = storage.Client()
_data_cache = {}


def load_historical_base(ticker):
    if ticker in _data_cache:
        return _data_cache[ticker].copy()
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob = gcs_client.bucket(BUCKET_NAME).blob(
        f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet")
    if not blob.exists():
        print(f"   Missing {file_ticker}_master_inference.parquet")
        return None
    df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
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
    cutoff_dt = pd.Timestamp.now(tz='UTC') - timedelta(days=TAIL_DAYS + 2)
    df = df[df.index >= cutoff_dt]
    _data_cache[ticker] = df
    return df.copy()


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def build_market_context(exp):
    frames = []
    for ticker in exp['context_assets']:
        df = load_historical_base(ticker)
        if df is None or df.empty:
            continue
        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        if exp['has_smh_rsi'] and ticker == 'SMH':
            rsi = ta.rsi(df['Close'], length=14)
            cols['SMH_RSI'] = (rsi.fillna(50) / 100.0
                               if rsi is not None
                               else pd.Series(0.5, index=df.index))
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
        gc.collect()
    if not frames:
        raise RuntimeError(f"No context assets loaded for {exp['label']}.")
    return pd.concat(frames, axis=1).ffill().fillna(0)


def engineer_asset(ticker, market_df):
    df = load_historical_base(ticker)
    if df is None or df.empty:
        return None
    df = heal_data(df)
    daily_c = df['Close'].resample('D').last().shift(1).reindex(
        df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c
    df['PM_High'] = df.groupby(df.index.date)['High'].cummax()
    df['Dist_to_PM_High'] = (df['Close'] - df['PM_High']) / (df['PM_High'] + 1e-9)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_60'] = ta.rsi(df['Close'], length=60)
    df['RVOL']   = df['Volume'] / df['Volume'].rolling(50).mean().clip(lower=1e-9)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Compression'] = (df['High'] - df['Low']) / (df['ATR_14'] + 1e-9)
    for h in ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)
    if market_df is not None and not market_df.empty:
        df = df.join(market_df, how='left').ffill().fillna(0)
    df_pst = df.index.tz_convert(PST)
    df['Time_Pos'] = np.clip((df_pst.hour * 60 + df_pst.minute - 390) / 390, 0.0, 1.0)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def apply_scaling(df, stats, all_cols):
    for col in all_cols:
        if col == 'Time_Pos':
            continue
        if col in stats:
            df[col] = ((df[col] - stats[col]['med'])
                       / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df


# ==============================================================================
# MODEL
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


def build_model(n_context_cols, n_assets):
    in_a  = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c  = Input(shape=(n_context_cols,),             name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,          name='asset_id')
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


def get_model(exp, seed):
    key = (exp['model_id'], seed)
    if key in _model_cache:
        return _model_cache[key]
    model_id = exp['model_id']
    blob_path  = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    local_path = f"/tmp/ws_test_{seed}_{model_id}.h5"
    if not os.path.exists(local_path):
        blob = gcs_client.bucket(BUCKET_NAME).blob(blob_path)
        if not blob.exists():
            print(f"      Model missing: {blob_path}")
            return None
        blob.download_to_filename(local_path)
    tf.get_logger().setLevel('ERROR')
    model = build_model(len(exp['context_cols']), exp['n_assets'])
    model.load_weights(local_path)
    _model_cache[key] = model
    return model


# ==============================================================================
# TRADE SIMULATION (from entry at offset, exit EOD 13:00 PST)
# ==============================================================================
def simulate_trade(df_unscaled, entry_dt_utc, d, tp_pct, sl_pct):
    exit_dt_utc = PST.localize(
        datetime(d.year, d.month, d.day, 13, 0)
    ).astimezone(pytz.utc)

    entry_idx = df_unscaled.index.get_loc(entry_dt_utc)
    entry_price = float(df_unscaled['Close'].iloc[entry_idx])

    df_intraday = df_unscaled.loc[
        entry_dt_utc + timedelta(minutes=1):exit_dt_utc
    ]
    if df_intraday.empty:
        return None

    sl_price = entry_price * (1 - sl_pct)
    tp_price = entry_price * (1 + tp_pct)

    for _, row in df_intraday.iterrows():
        if row['Low'] <= sl_price:
            return {'pnl': -sl_pct * 100, 'reason': 'SL', 'exit': sl_price}
        if row['High'] >= tp_price:
            return {'pnl': tp_pct * 100, 'reason': 'TP', 'exit': tp_price}

    eod_pnl = (df_intraday['Close'].iloc[-1] - entry_price) / entry_price
    return {'pnl': eod_pnl * 100, 'reason': 'EOD', 'exit': float(df_intraday['Close'].iloc[-1])}


# ==============================================================================
# MAIN — WINDOW SHIFT TEST
# ==============================================================================
def main():
    print("=" * 130)
    print("  WINDOW SHIFT TEST — Apr 27-28, 2026  [SCORING MODE, NEGATIVE+POSITIVE OFFSETS]")
    print(f"  TAIL_DAYS={TAIL_DAYS}, SKIP_VOLUME_0={SKIP_VOLUME_0}")
    print(f"  Offsets: {SHIFT_OFFSETS[0]} to {SHIFT_OFFSETS[-1]} minutes from entry_minute")
    print("  Negative = scored BEFORE entry (what live scoring actually did)")
    print("=" * 130)

    all_shift_results = []

    for exp in EXPERIMENTS:
        model_id = exp['model_id']
        label    = exp['label']
        context_cols = exp['context_cols']
        all_cols     = exp['all_cols']

        print(f"\n--- Loading {label} ---")

        stats_blob = gcs_client.bucket(BUCKET_NAME).blob(exp['stats_path'])
        if not stats_blob.exists():
            print(f"   Stats missing: {exp['stats_path']}")
            continue
        stats = json.loads(stats_blob.download_as_string())

        market_df = build_market_context(exp)

        needed_seeds = sorted(set(cfg['seed'] for cfg in exp['asset_configs'].values()))
        for seed in needed_seeds:
            get_model(exp, seed)
        print(f"   Models ready: {needed_seeds}")

        for ticker, config in exp['asset_configs'].items():
            seed = config['seed']
            model = get_model(exp, seed)
            if model is None:
                continue
            asset_id = exp['asset_index'].get(ticker, exp['other_idx'])

            df = engineer_asset(ticker, market_df)
            if df is None or df.empty:
                continue

            df_unscaled = df.copy()
            df_scaled = apply_scaling(df.copy(), stats, all_cols)

            entry_minute = config['entry_minute']
            base_h = entry_minute // 60
            base_m = entry_minute % 60
            thresh = config['prob_threshold']
            tp_pct = config['tp_pct']
            sl_pct = config['sl_pct']

            for test_date in TEST_DATES:
                d = test_date.date() if hasattr(test_date, 'date') else test_date

                probs_at_offset = {}
                trade_at_offset = {}

                for offset in SHIFT_OFFSETS:
                    shifted_minute = entry_minute + offset
                    sh = shifted_minute // 60
                    sm = shifted_minute % 60

                    entry_dt_pst = PST.localize(datetime(d.year, d.month, d.day, sh, sm))
                    entry_dt_utc = entry_dt_pst.astimezone(pytz.utc)

                    if entry_dt_utc not in df_scaled.index:
                        probs_at_offset[offset] = None
                        continue

                    idx = df_scaled.index.get_loc(entry_dt_utc)
                    if idx < TIME_STEPS:
                        probs_at_offset[offset] = None
                        continue

                    if SKIP_VOLUME_0 and df_unscaled['Volume'].iloc[idx] == 0:
                        probs_at_offset[offset] = None
                        continue

                    xa = df_scaled[ASSET_COLS].values[idx - TIME_STEPS:idx].astype('float32')
                    xc = df_scaled[context_cols].values[idx - 1].astype('float32')
                    XA_b  = np.expand_dims(xa, axis=0)
                    XC_b  = np.expand_dims(xc, axis=0)
                    XID_b = np.array([[asset_id]], dtype=np.int32)

                    prob = float(model(
                        {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
                        training=False
                    ).numpy().flatten()[0])

                    probs_at_offset[offset] = prob

                    if prob >= thresh:
                        # Simulate trade entering at the CONFIGURED entry_minute
                        # (trade entry is always at entry_minute regardless of when model fires)
                        base_entry_pst = PST.localize(
                            datetime(d.year, d.month, d.day, base_h, base_m))
                        base_entry_utc = base_entry_pst.astimezone(pytz.utc)
                        if base_entry_utc in df_unscaled.index:
                            trade = simulate_trade(
                                df_unscaled, base_entry_utc, d, tp_pct, sl_pct)
                            trade_at_offset[offset] = trade

                all_shift_results.append({
                    'ticker': ticker,
                    'experiment': model_id,
                    'date': str(d),
                    'threshold': thresh,
                    'entry_minute': entry_minute,
                    'tp_pct': tp_pct,
                    'sl_pct': sl_pct,
                    'probs': probs_at_offset,
                    'trades': trade_at_offset,
                })

    # ==========================================================================
    # REPORT
    # ==========================================================================
    print("\n\n" + "=" * 150)
    print("  RESULTS: Probability at each window offset")
    print("  Offset 0 = exact entry_minute. Negative = scored BEFORE entry (what live scoring did).")
    print("  Positive = scored AFTER entry. * = above threshold (GO).")
    print("=" * 150)

    for d_str in [str(d.date() if hasattr(d, 'date') else d) for d in TEST_DATES]:
        print(f"\n{'='*150}")
        print(f"  DATE: {d_str}")
        print(f"{'='*150}")

        day_results = [r for r in all_shift_results if r['date'] == d_str]

        false_positive_count = 0
        false_positive_pnl = 0.0

        for r in day_results:
            probs = r['probs']
            trades = r['trades']
            thresh = r['threshold']
            em = r['entry_minute']

            go_offsets = [off for off in SHIFT_OFFSETS
                         if probs.get(off) is not None and probs[off] >= thresh]

            base_go = 0 in go_offsets
            neg_go = [o for o in go_offsets if o < 0]
            pos_go = [o for o in go_offsets if o > 0]

            # Classify signal
            if base_go:
                sig_str = "GO@0 (correct)"
            elif neg_go or pos_go:
                offsets_str = ",".join(f"{o:+d}" for o in go_offsets)
                sig_str = f"FALSE POS @ {offsets_str}"
                false_positive_count += 1
            else:
                sig_str = "no signal"

            # Prob at offset 0
            p0 = probs.get(0)
            p0_str = f"{p0:.3f}" if p0 is not None else "N/A"

            # Max prob and where
            valid = {o: p for o, p in probs.items() if p is not None}
            if valid:
                max_off = max(valid, key=valid.get)
                max_p = valid[max_off]
                max_str = f"{max_p:.3f} @{max_off:+d}m"
            else:
                max_str = "N/A"

            # Trade PnL
            trade_str = ""
            if trades:
                for off in sorted(trades.keys()):
                    t = trades[off]
                    if t:
                        trade_str = f"{t['pnl']:+.2f}% ({t['reason']})"
                        if not base_go and off != 0:
                            false_positive_pnl += t['pnl']
                        break

            # BQ live comparison: what shift did live scoring actually use?
            live_shifts = {
                ('ASTS', '2026-04-28'): -11, ('JOBY', '2026-04-28'): -13,
                ('LUNR', '2026-04-28'): +1,  ('RKLB', '2026-04-28'): -19,
                ('IREN', '2026-04-28'): -9,  ('SPIR', '2026-04-28'): -9,
                ('SPIR', '2026-04-27'): -4,
            }
            live_key = (r['ticker'], d_str)
            live_off = live_shifts.get(live_key)
            live_str = ""
            if live_off is not None:
                lp = probs.get(live_off)
                if lp is not None:
                    live_go = "GO" if lp >= thresh else "NO"
                    live_str = f"  live@{live_off:+d}m: {lp:.3f} → {live_go}"

            print(f"  {r['ticker']:<6} {r['experiment']:<13} thr={thresh:.3f}  "
                  f"entry={em}  prob@0={p0_str}  max={max_str}  "
                  f"GO_offsets={go_offsets if go_offsets else 'none'}")
            print(f"         {sig_str:<35} {trade_str:<20}{live_str}")

            # Print probability curve (compact)
            curve = "         curve: "
            for off in SHIFT_OFFSETS:
                p = probs.get(off)
                if p is None:
                    continue
                if off == 0:
                    marker = ">>>" if p >= thresh else "   "
                    curve += f" [{off:+3d}:{p:.3f}{marker}]"
                elif p >= thresh:
                    curve += f" {off:+3d}:{p:.3f}*"
                elif off % 5 == 0:
                    curve += f" {off:+3d}:{p:.3f} "
            print(curve)
            print()

        print(f"  {'-'*100}")
        print(f"  {d_str} Summary: {false_positive_count} false positives, "
              f"total FP PnL: {false_positive_pnl:+.2f}%")

    # ==========================================================================
    # OVERALL SUMMARY
    # ==========================================================================
    print("\n\n" + "=" * 150)
    print("  OVERALL SUMMARY")
    print("=" * 150)

    total_fp = 0
    total_fp_pnl = 0.0
    total_correct_go = 0
    total_correct_pnl = 0.0

    for r in all_shift_results:
        probs = r['probs']
        trades = r['trades']
        thresh = r['threshold']

        base_go = probs.get(0) is not None and probs[0] >= thresh
        shifted_gos = [off for off in SHIFT_OFFSETS if off != 0
                       and probs.get(off) is not None and probs[off] >= thresh]

        if base_go:
            total_correct_go += 1
            if 0 in trades and trades[0]:
                total_correct_pnl += trades[0]['pnl']
        elif shifted_gos:
            total_fp += 1
            for off in sorted(shifted_gos, key=abs):
                if off in trades and trades[off]:
                    total_fp_pnl += trades[off]['pnl']
                    break

    print(f"  Correct signals (GO at offset 0):              {total_correct_go}")
    print(f"  False positives (GO only at non-zero offset):  {total_fp}")
    print(f"  False positive total PnL:                      {total_fp_pnl:+.2f}%")
    print(f"  Correct signal total PnL:                      {total_correct_pnl:+.2f}%")
    print()
    print(f"  Offsets tested: {SHIFT_OFFSETS[0]} to {SHIFT_OFFSETS[-1]} minutes")
    print(f"  Negative offsets = model scored BEFORE entry_minute (live scoring behavior)")
    print(f"  Offset 0 = exact entry_minute bar (what backtester uses)")
    print()

    if total_fp > 0:
        print(f"  CONCLUSION: Window shift caused {total_fp} false GO signals")
        print(f"  that would NOT have fired at the exact entry_minute bar.")
        print(f"  FIX: anchor model input to entry_minute bar, not last available bar.")
    else:
        print(f"  CONCLUSION: No false positives from window shift on these dates.")

    print("=" * 150)


if __name__ == "__main__":
    main()
