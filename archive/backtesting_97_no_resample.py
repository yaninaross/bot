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

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Keep standard pandas_ta per Exp 97 requirements
import subprocess
try:
    import pandas_ta as ta
except ModuleNotFoundError:
    print("Installing pandas_ta...")
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
# BACKTESTER CONFIGURATION (EXP 97)
# ==============================================================================
MODEL_ID = "97_MICRO"

ASSET_CONFIGS = {
    'MIGI': {'seed': 9709, 'prob_threshold': 0.510, 'tp_pct': 0.45, 'sl_pct': 0.05, 'entry_minute': 375, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.50, 'stats': {'win_rate': '55.6%', 'avg_pnl': '15.41%', 'freq': '1.5/mo'}},
    'WULF': {'seed': 9710, 'prob_threshold': 0.510, 'tp_pct': 0.17, 'sl_pct': 0.06, 'entry_minute': 360, 'vix_threshold': 10.0, 'tlt_crash': -0.50, 'spy_ignore': 0.00, 'stats': {'win_rate': '77.8%', 'avg_pnl': '6.57%', 'freq': '1.5/mo'}},
    'BTBT': {'seed': 9711, 'prob_threshold': 0.510, 'tp_pct': 0.15, 'sl_pct': 0.08, 'entry_minute': 360, 'vix_threshold': 15.0, 'tlt_crash': -0.40, 'spy_ignore': 0.30, 'stats': {'win_rate': '88.9%', 'avg_pnl': '6.29%', 'freq': '1.5/mo'}},
    'HUT':  {'seed': 9701, 'prob_threshold': 0.485, 'tp_pct': 0.19, 'sl_pct': 0.06, 'entry_minute': 385, 'vix_threshold': 10.0, 'tlt_crash': -0.16, 'spy_ignore': 0.10, 'stats': {'win_rate': '88.9%', 'avg_pnl': '6.10%', 'freq': '1.5/mo'}},
    'CIFR': {'seed': 9704, 'prob_threshold': 0.515, 'tp_pct': 0.14, 'sl_pct': 0.08, 'entry_minute': 385, 'vix_threshold': 10.0, 'tlt_crash': -0.50, 'spy_ignore': 0.00, 'stats': {'win_rate': '90.9%', 'avg_pnl': '6.05%', 'freq': '1.9/mo'}},
    'RIOT': {'seed': 9711, 'prob_threshold': 0.505, 'tp_pct': 0.06, 'sl_pct': 0.04, 'entry_minute': 380, 'vix_threshold': 10.0, 'tlt_crash': -0.46, 'spy_ignore': 0.00, 'stats': {'win_rate': '100.0%', 'avg_pnl': '6.00%', 'freq': '1.5/mo'}},
    'IREN': {'seed': 9707, 'prob_threshold': 0.510, 'tp_pct': 0.08, 'sl_pct': 0.10, 'entry_minute': 375, 'vix_threshold': 10.0, 'tlt_crash': -0.20, 'spy_ignore': 0.08, 'stats': {'win_rate': '83.3%', 'avg_pnl': '5.71%', 'freq': '2.0/mo'}},
    'KEEL': {'seed': 9714, 'prob_threshold': 0.510, 'tp_pct': 0.11, 'sl_pct': 0.16, 'entry_minute': 385, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.18, 'stats': {'win_rate': '77.8%', 'avg_pnl': '5.50%', 'freq': '1.5/mo'}},
    'CLSK': {'seed': 9701, 'prob_threshold': 0.500, 'tp_pct': 0.19, 'sl_pct': 0.10, 'entry_minute': 365, 'vix_threshold': 10.0, 'tlt_crash': -0.32, 'spy_ignore': 0.30, 'stats': {'win_rate': '88.9%', 'avg_pnl': '5.46%', 'freq': '1.5/mo'}},
    'SMLR': {'seed': 9709, 'prob_threshold': 0.420, 'tp_pct': 0.15, 'sl_pct': 0.06, 'entry_minute': 375, 'vix_threshold': 15.0, 'tlt_crash':  0.00, 'spy_ignore': 0.00, 'stats': {'win_rate': '75.0%', 'avg_pnl': '5.29%', 'freq': '2.3/mo'}},
    'APLD': {'seed': 9705, 'prob_threshold': 0.510, 'tp_pct': 0.10, 'sl_pct': 0.05, 'entry_minute': 390, 'vix_threshold': 10.0, 'tlt_crash': -0.14, 'spy_ignore': 0.00, 'stats': {'win_rate': '88.9%', 'avg_pnl': '5.16%', 'freq': '1.5/mo'}},
    'SMCI': {'seed': 9707, 'prob_threshold': 0.485, 'tp_pct': 0.09, 'sl_pct': 0.02, 'entry_minute': 390, 'vix_threshold': 10.0, 'tlt_crash': -0.30, 'spy_ignore': 0.00, 'stats': {'win_rate': '55.6%', 'avg_pnl': '3.09%', 'freq': '1.5/mo'}},
    'MARA': {'seed': 9703, 'prob_threshold': 0.465, 'tp_pct': 0.18, 'sl_pct': 0.06, 'entry_minute': 385, 'vix_threshold': 10.0, 'tlt_crash': -0.50, 'spy_ignore': 0.20, 'stats': {'win_rate': '55.6%', 'avg_pnl': '2.87%', 'freq': '1.5/mo'}},
    'COIN': {'seed': 9715, 'prob_threshold': 0.470, 'tp_pct': 0.08, 'sl_pct': 0.06, 'entry_minute': 365, 'vix_threshold': 10.0, 'tlt_crash': -0.42, 'spy_ignore': 0.00, 'stats': {'win_rate': '66.7%', 'avg_pnl': '2.83%', 'freq': '2.0/mo'}},
    'CORZ': {'seed': 9705, 'prob_threshold': 0.445, 'tp_pct': 0.13, 'sl_pct': 0.04, 'entry_minute': 370, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.10, 'stats': {'win_rate': '66.7%', 'avg_pnl': '2.55%', 'freq': '1.5/mo'}},
    'HOOD': {'seed': 9706, 'prob_threshold': 0.450, 'tp_pct': 0.10, 'sl_pct': 0.04, 'entry_minute': 385, 'vix_threshold': 10.0, 'tlt_crash': -0.20, 'spy_ignore': 0.10, 'stats': {'win_rate': '77.8%', 'avg_pnl': '2.36%', 'freq': '1.5/mo'}},
    'MSTR': {'seed': 9711, 'prob_threshold': 0.510, 'tp_pct': 0.17, 'sl_pct': 0.08, 'entry_minute': 365, 'vix_threshold': 10.0, 'tlt_crash': -0.30, 'spy_ignore': 0.00, 'stats': {'win_rate': '50.0%', 'avg_pnl': '2.19%', 'freq': '1.7/mo'}},
    'AMD':  {'seed': 9711, 'prob_threshold': 0.465, 'tp_pct': 0.15, 'sl_pct': 0.06, 'entry_minute': 370, 'vix_threshold': 15.0, 'tlt_crash':  0.00, 'spy_ignore': 0.00, 'stats': {'win_rate': '70.0%', 'avg_pnl': '2.03%', 'freq': '1.7/mo'}},
    'TSLA': {'seed': 9704, 'prob_threshold': 0.480, 'tp_pct': 0.06, 'sl_pct': 0.02, 'entry_minute': 360, 'vix_threshold': 10.0, 'tlt_crash': -0.50, 'spy_ignore': 0.30, 'stats': {'win_rate': '69.2%', 'avg_pnl': '1.89%', 'freq': '2.2/mo'}},
    'VRT':  {'seed': 9708, 'prob_threshold': 0.440, 'tp_pct': 0.08, 'sl_pct': 0.08, 'entry_minute': 370, 'vix_threshold': 10.0, 'tlt_crash': -0.50, 'spy_ignore': 0.00, 'stats': {'win_rate': '70.0%', 'avg_pnl': '1.60%', 'freq': '1.7/mo'}},
    'PLTR': {'seed': 9705, 'prob_threshold': 0.430, 'tp_pct': 0.07, 'sl_pct': 0.08, 'entry_minute': 370, 'vix_threshold': 10.0, 'tlt_crash': -0.04, 'spy_ignore': 0.00, 'stats': {'win_rate': '70.0%', 'avg_pnl': '1.52%', 'freq': '1.7/mo'}},
    'SMH':  {'seed': 9705, 'prob_threshold': 0.410, 'tp_pct': 0.10, 'sl_pct': 0.04, 'entry_minute': 365, 'vix_threshold': 20.0, 'tlt_crash': -0.20, 'spy_ignore': 0.00, 'stats': {'win_rate': '70.0%', 'avg_pnl': '0.99%', 'freq': '1.7/mo'}},
    'NVDA': {'seed': 9713, 'prob_threshold': 0.445, 'tp_pct': 0.06, 'sl_pct': 0.03, 'entry_minute': 390, 'vix_threshold': 10.0, 'tlt_crash': -0.46, 'spy_ignore': 0.20, 'stats': {'win_rate': '50.0%', 'avg_pnl': '0.90%', 'freq': '2.0/mo'}},
    'IWM':  {'seed': 9713, 'prob_threshold': 0.430, 'tp_pct': 0.10, 'sl_pct': 0.02, 'entry_minute': 385, 'vix_threshold': 15.0, 'tlt_crash':  0.00, 'spy_ignore': 0.10, 'stats': {'win_rate': '70.0%', 'avg_pnl': '0.74%', 'freq': '1.7/mo'}},
    'TLT':  {'seed': 9714, 'prob_threshold': 0.410, 'tp_pct': 0.10, 'sl_pct': 0.02, 'entry_minute': 365, 'vix_threshold': 15.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '88.9%', 'avg_pnl': '0.28%', 'freq': '1.5/mo'}},
}

# Asset Embedding mapper for in_id
TRAIN_TEACHERS = [
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR', 'COIN',
    'BITF', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'SMLR',
]
ASSET_INDEX = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
ASSET_INDEX['KEEL'] = ASSET_INDEX['BITF']   # BITF renamed to KEEL on 2026-04-06
N_ASSETS    = len(TRAIN_TEACHERS) + 1   # 17
OTHER_IDX   = len(TRAIN_TEACHERS)       # 16
EMBED_DIM   = 16

CONTEXT_ASSETS = ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD']
ROC_HORIZONS   = [5, 30, 60, 120]

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct', 'bsize', 'asize', 'Total_Vol']
ASSET_COLS = ['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression', 'ATR_14', 'Dist_to_PM_High'] + [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS

CONTEXT_COLS = []
for ticker in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{ticker}_Anchor_ROC')
    for h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{ticker}_ROC_{h}m')
CONTEXT_COLS += ['SMH_RSI', 'Time_Pos']
ALL_COLS = ASSET_COLS + CONTEXT_COLS

# Universal Paths
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference"
STATS_FILE_NAME = f"scaling_stats_experiment_{MODEL_ID}.json"

TIME_STEPS = 240
BACKTEST_DAYS = 180
PST = pytz.timezone('America/Los_Angeles')

try:
    from google.colab import auth
    auth.authenticate_user()
except:
    pass

client = storage.Client()

# ==============================================================================
# DATA LOADING & FEATURE ENGINEERING
# ==============================================================================
def load_historical_base(ticker):
    clean_ticker = ticker.replace("I:", "")
    blob = client.bucket(BUCKET_NAME).blob(f"{INFERENCE_DIR}/{clean_ticker}_master_inference.parquet")
    if not blob.exists():
        print(f"   Missing {clean_ticker}_master_inference.parquet in GCS.")
        return None

    df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    df = df.sort_index()

    # Align column capitalization exactly as done in exp_97
    std_map = {
        'Total_vol': 'Total_Vol', 'Trade_count': 'Trade_Count',
        'Volume_delta': 'Volume_Delta', 'Delta_pct': 'Delta_Pct',
        'Obi': 'OBI', 'Bsize': 'bsize', 'Asize': 'asize',
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }
    lower_map = {k.lower(): v for k, v in std_map.items()}
    df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()

    df = df[~df.index.duplicated(keep='first')]

    cutoff_dt = pd.Timestamp.now(tz='UTC') - timedelta(days=BACKTEST_DAYS + 15)
    df = df[df.index >= cutoff_dt]

    for c in MICRO_COLS:
        if c not in df.columns: df[c] = 0.0
    return df

def heal_data(df):
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df

def build_market_context():
    frames = []
    print("Building Macro Context Features...")
    for ticker in CONTEXT_ASSETS:
        df = load_historical_base(ticker)
        if df is None or df.empty: continue

        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        if ticker == 'SMH':
            cols['SMH_RSI'] = ta.rsi(df['Close'], length=14).fillna(50) / 100.0

        frames.append(pd.DataFrame(cols, index=df.index))
        del df; gc.collect()

    return pd.concat(frames, axis=1).ffill().fillna(0)

def engineer_asset(asset_name, market_df):
    df = load_historical_base(asset_name)
    if df is None or df.empty: return None

    df = heal_data(df)
    daily_c = df['Close'].resample('D').last().shift(1).reindex(df.index, method='ffill')
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

# ==============================================================================
# MODEL ARCHITECTURE (Exact match to exp_97)
# ==============================================================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.4):
    x   = LayerNormalization(epsilon=1e-6)(inputs)
    x   = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs])
    x   = LayerNormalization(epsilon=1e-6)(res)
    x   = Dense(ff_dim, activation="relu")(x)
    x   = Dropout(dropout)(x)
    x   = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

def build_model():
    in_a  = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c  = Input(shape=(len(CONTEXT_COLS),),          name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,           name='asset_id')

    # Transformer sequence branch
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))

    # Context branch
    y = Dense(128, activation='relu')(BatchNormalization()(in_c))

    # Asset embedding branch
    e = Embedding(N_ASSETS, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)

    # Fuse all three
    z   = Concatenate()([x, Dropout(0.5)(y), e])
    z   = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)

# ==============================================================================
# THE BACKTESTING ENGINE
# ==============================================================================
def run_backtester():
    print(f"STARTING {BACKTEST_DAYS}-DAY HISTORICAL BACKTEST (EXP {MODEL_ID}) — NO RESAMPLE")
    market_df = build_market_context()

    print("Pre-loading SPY, TLT, and VIX for historical Trap gap calculations...")
    spy_raw = load_historical_base("SPY")
    tlt_raw = load_historical_base("TLT")
    vix_raw = load_historical_base("VIX")
    if spy_raw is not None: spy_raw = spy_raw.tz_convert(PST)
    if tlt_raw is not None: tlt_raw = tlt_raw.tz_convert(PST)
    if vix_raw is not None: vix_raw = vix_raw.tz_convert(PST)

    stats_blob = client.bucket(BUCKET_NAME).blob(f"Data/Inference_v2/{STATS_FILE_NAME}")
    if not stats_blob.exists(): sys.exit(f"Stats missing: {STATS_FILE_NAME}")
    stats = json.loads(stats_blob.download_as_string())

    results_summary = []
    model_cache = {}

    for ticker, config in ASSET_CONFIGS.items():
        print(f"\nTesting {ticker}...")

        seed = config['seed']
        if seed not in model_cache:
            # Assuming Universal Direction models
            model_path = f"Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction/UNIVERSAL_S{seed}_M{MODEL_ID}.h5"
            blob = client.bucket(BUCKET_NAME).blob(model_path)
            if not blob.exists():
                print(f"   Model missing for seed {seed} at {model_path}")
                continue
            local_model = f"temp_{seed}.h5"
            blob.download_to_filename(local_model)

            tf.get_logger().setLevel('ERROR')
            model = build_model()
            model.load_weights(local_model)
            model_cache[seed] = model
            os.remove(local_model)

        model = model_cache[seed]
        asset_id = ASSET_INDEX.get(ticker, OTHER_IDX)

        df = engineer_asset(ticker, market_df)
        if df is None or df.empty:
            print(f"   Could not load data for {ticker}. Check GCS bucket.")
            continue

        # Apply Scaling
        for col in ALL_COLS:
            if col != 'Time_Pos' and col in stats:
                df[col] = ((df[col] - stats[col]['med']) / stats[col]['iqr']).clip(-10, 10).astype('float32')

        entry_minute = config['entry_minute']
        entry_h = entry_minute // 60
        entry_m = entry_minute % 60

        tp_pct = config['tp_pct']
        sl_pct = config['sl_pct']
        avg_pnl_pct = float(config['stats']['avg_pnl'].strip('%')) / 100.0

        cm = {'strict': {'TP':0, 'FP':0, 'FN':0, 'TN':0}, 'avg': {'TP':0, 'FP':0, 'FN':0, 'TN':0}}
        trades_taken = 0
        trades_blocked = 0
        total_pnl = 0.0

        test_start_dt = pd.Timestamp.now(tz=PST) - timedelta(days=BACKTEST_DAYS)
        df_test = df[df.index.tz_convert(PST) >= test_start_dt]
        unique_dates = np.unique(df_test.index.tz_convert(PST).date)

        for d in unique_dates:
            if d.weekday() >= 5: continue  # skip Saturday/Sunday

            entry_dt_pst = PST.localize(datetime(d.year, d.month, d.day, entry_h, entry_m))
            entry_dt_utc = entry_dt_pst.astimezone(pytz.utc)

            if entry_dt_utc not in df.index: continue

            idx = df.index.get_loc(entry_dt_utc)
            if idx < TIME_STEPS: continue

            if pd.isna(df.iloc[idx]['Close']): continue  # skip holiday/NaN bars

            # 1. Trap Logic — trailing 30-min median ending at entry time
            try:
                spy_prev = spy_raw[(spy_raw.index.date < d)]['Close'].iloc[-1]
                tlt_prev = tlt_raw[(tlt_raw.index.date < d)]['Close'].iloc[-1]

                window_start = entry_dt_pst - timedelta(minutes=30)
                vix_bars = vix_raw[(vix_raw.index >= window_start) & (vix_raw.index <= entry_dt_pst)]
                spy_bars = spy_raw[(spy_raw.index >= window_start) & (spy_raw.index <= entry_dt_pst)]
                tlt_bars = tlt_raw[(tlt_raw.index >= window_start) & (tlt_raw.index <= entry_dt_pst)]

                vix_now = float(vix_bars['Close'].median()) if len(vix_bars) else None
                spy_now = float(spy_bars['Close'].median()) if len(spy_bars) else None
                tlt_now = float(tlt_bars['Close'].median()) if len(tlt_bars) else None

                if vix_now is None or spy_now is None or tlt_now is None:
                    is_trap = False
                else:
                    spy_gap = ((spy_now - spy_prev) / spy_prev) * 100
                    tlt_gap = ((tlt_now - tlt_prev) / tlt_prev) * 100

                    vix_bad = vix_now > config['vix_threshold']
                    tlt_bad = tlt_gap < config['tlt_crash']
                    spy_bad = spy_gap > config['spy_ignore']

                    is_trap = vix_bad and tlt_bad and spy_bad
            except Exception:
                is_trap = False

            # 2. Prediction (3 Inputs for Exp 97)
            xa = df[ASSET_COLS].values[idx-TIME_STEPS:idx].astype('float32')
            xc = df[CONTEXT_COLS].values[idx-1].astype('float32')
            XA_b  = np.expand_dims(xa, axis=0)
            XC_b  = np.expand_dims(xc, axis=0)
            XID_b = np.array([[asset_id]], dtype=np.int32)

            score = float(model(
                {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
                training=False
            ).numpy().flatten()[0])

            thresh = config['prob_threshold']
            # 3. Decision Gate
            if score >= thresh:
                if is_trap:
                    trades_blocked += 1
                    signal = 0
                else:
                    signal = 1
            else:
                signal = 0

            entry_price = df['Close'].iloc[idx]

            # 4. Intraday Sim
            exit_dt_utc = PST.localize(datetime(d.year, d.month, d.day, 13, 0)).astimezone(pytz.utc)
            df_intraday = df.loc[entry_dt_utc + timedelta(minutes=1) : exit_dt_utc]

            if df_intraday.empty: continue

            sl_price = entry_price * (1 - sl_pct)
            tp_price_strict = entry_price * (1 + tp_pct)
            tp_price_avg = entry_price * (1 + avg_pnl_pct)

            actual_pnl = 0.0
            hit_strict = 0
            hit_avg = 0
            trade_active = True

            for _, row in df_intraday.iterrows():
                if not trade_active: break

                if row['Low'] <= sl_price:
                    actual_pnl = -sl_pct
                    trade_active = False
                elif row['High'] >= tp_price_strict:
                    hit_strict = 1
                    hit_avg = 1
                    actual_pnl = tp_pct
                    trade_active = False
                elif row['High'] >= tp_price_avg:
                    hit_avg = 1

            if trade_active:
                actual_pnl = (df_intraday['Close'].iloc[-1] - entry_price) / entry_price

            if signal == 1:
                trades_taken += 1
                total_pnl += actual_pnl
                if hit_strict: cm['strict']['TP'] += 1
                else:          cm['strict']['FP'] += 1
                if hit_avg:    cm['avg']['TP'] += 1
                else:          cm['avg']['FP'] += 1
            else:
                if hit_strict: cm['strict']['FN'] += 1
                else:          cm['strict']['TN'] += 1
                if hit_avg:    cm['avg']['FN'] += 1
                else:          cm['avg']['TN'] += 1

        def calc_metrics(c_dict):
            prec = c_dict['TP'] / (c_dict['TP'] + c_dict['FP']) if (c_dict['TP'] + c_dict['FP']) > 0 else 0
            rec = c_dict['TP'] / (c_dict['TP'] + c_dict['FN']) if (c_dict['TP'] + c_dict['FN']) > 0 else 0
            return prec, rec

        prec_strict, rec_strict = calc_metrics(cm['strict'])
        prec_avg, rec_avg = calc_metrics(cm['avg'])
        avg_trade_pnl = total_pnl / trades_taken if trades_taken > 0 else 0
        win_rate = cm['strict']['TP'] / trades_taken if trades_taken > 0 else 0

        results_summary.append({
            'Asset': ticker, 'Trades': trades_taken, 'Blocked': trades_blocked,
            'P_Strict': prec_strict, 'R_Strict': rec_strict,
            'P_Avg': prec_avg, 'R_Avg': rec_avg,
            'WinRate': win_rate, 'TotalPnL': total_pnl, 'AvgPnL': avg_trade_pnl
        })

    # Dashboard
    print("\n" + "="*140)
    print(f"{BACKTEST_DAYS}-DAY BACKTEST RESULTS w/ TRAP FILTER (EXP {MODEL_ID}) — NO RESAMPLE")
    print("="*140)
    print(f"{'ASSET':<6} | {'TRADES':<6} | {'BLOCKED':<7} | {'STRICT TP TARGET':<22} | {'AVG PNL TARGET':<22} | {'ACTUAL PNL':<25}")
    print(f"{'':<6} | {'TAKEN':<6} | {'(TRAPS)':<7} | {'Prec.':<10} {'Recall':<10} | {'Prec.':<10} {'Recall':<10} | {'WinRate':<8} {'Total PnL':<10} {'Avg PnL':<8}")
    print("-" * 140)

    for r in results_summary:
        s_prec = f"{r['P_Strict']:.1%}"
        s_rec  = f"{r['R_Strict']:.1%}"
        a_prec = f"{r['P_Avg']:.1%}"
        a_rec  = f"{r['R_Avg']:.1%}"
        wr     = f"{r['WinRate']:.1%}"
        tot    = f"{r['TotalPnL']:.1%}"
        avg    = f"{r['AvgPnL']:.2%}"

        print(f"{r['Asset']:<6} | {r['Trades']:<6} | {r['Blocked']:<7} | {s_prec:<10} {s_rec:<10} | {a_prec:<10} {a_rec:<10} | {wr:<8} {tot:<10} {avg:<8}")

    print("="*140)

if __name__ == "__main__":
    run_backtester()
