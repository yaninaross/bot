"""
Backtesting — Exp-110 (Crypto miners)
======================================
Self-contained. Mirrors daily_score_exp110.py logic exactly.
Runs retrospective intraday trade simulation for 4 deploy candidates.

Deploys 4 tickers:
  CLSK  seed=11002  entry=06:10  prob>0.450  TP=18%  SL=5%
  IREN  seed=11014  entry=06:25  prob>0.510  TP=10%  SL=10%
  RIOT  seed=11013  entry=06:15  prob>0.650  TP=8%   SL=6%
  APLD  seed=11008  entry=06:05  prob>0.715  TP=20%  SL=12%

Usage:  python backtesting_110.py
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
# BACKTESTER CONFIGURATION (EXP 110)
# ==============================================================================
MODEL_ID = "110_MICRO"

ASSET_CONFIGS = {
    'CLSK': {
        'seed': 11002, 'prob_threshold': 0.450,
        'tp_pct': 0.18, 'sl_pct': 0.05, 'entry_minute': 370,
        'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
        'stats': {'win_rate': '55.6%', 'avg_pnl': '2.85%', 'freq': '2.6/mo'},
    },
    'IREN': {
        'seed': 11014, 'prob_threshold': 0.510,
        'tp_pct': 0.10, 'sl_pct': 0.10, 'entry_minute': 385,
        'vix_threshold': 15.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
        'stats': {'win_rate': '57.1%', 'avg_pnl': '2.54%', 'freq': '2.0/mo'},
    },
    'RIOT': {
        'seed': 11013, 'prob_threshold': 0.650,
        'tp_pct': 0.08, 'sl_pct': 0.06, 'entry_minute': 375,
        'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.24,
        'stats': {'win_rate': '75.0%', 'avg_pnl': '1.91%', 'freq': '1.2/mo'},
    },
    'APLD': {
        'seed': 11008, 'prob_threshold': 0.715,
        'tp_pct': 0.20, 'sl_pct': 0.12, 'entry_minute': 365,
        'vix_threshold': 15.0, 'tlt_crash': 0.00, 'spy_ignore': 0.10,
        'stats': {'win_rate': '75.0%', 'avg_pnl': '1.36%', 'freq': '2.3/mo'},
    },
}

TRAIN_TEACHERS = [
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
    'COIN', 'BITF', 'HUT', 'CIFR', 'BTBT', 'HOOD',
]
ASSET_INDEX = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
ASSET_INDEX['KEEL'] = ASSET_INDEX['BITF']
N_ASSETS  = len(TRAIN_TEACHERS) + 1   # 15
OTHER_IDX = len(TRAIN_TEACHERS)        # 14
EMBED_DIM = 16

TICKER_FILE_MAP = {'BITF': 'KEEL'}

CONTEXT_ASSETS = ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD']
HAS_SMH_RSI    = True
ROC_HORIZONS   = [5, 30, 60, 120]

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
if HAS_SMH_RSI:
    CONTEXT_COLS.append('SMH_RSI')
CONTEXT_COLS.append('Time_Pos')

ALL_COLS = ASSET_COLS + CONTEXT_COLS

BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference"
STATS_FILE_NAME = f"scaling_stats_experiment_{MODEL_ID}.json"

TIME_STEPS    = 240
BACKTEST_DAYS = 150
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
    clean_ticker = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean_ticker, clean_ticker)
    blob = client.bucket(BUCKET_NAME).blob(f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet")
    if not blob.exists():
        print(f"   Missing {file_ticker}_master_inference.parquet in GCS.")
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

    cutoff_dt = pd.Timestamp.now(tz='UTC') - timedelta(days=BACKTEST_DAYS + 15)
    df = df[df.index >= cutoff_dt]
    return df


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def build_market_context():
    frames = []
    print("Building Macro Context Features...")
    for ticker in CONTEXT_ASSETS:
        df = load_historical_base(ticker)
        if df is None or df.empty:
            continue

        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        if HAS_SMH_RSI and ticker == 'SMH':
            rsi = ta.rsi(df['Close'], length=14)
            cols['SMH_RSI'] = (rsi.fillna(50) / 100.0
                               if rsi is not None
                               else pd.Series(0.5, index=df.index))

        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
        gc.collect()

    if not frames:
        raise RuntimeError("No context assets loaded.")
    return pd.concat(frames, axis=1).ffill().fillna(0)


def engineer_asset(asset_name, market_df):
    df = load_historical_base(asset_name)
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


# ==============================================================================
# MODEL ARCHITECTURE (Exact match to Exp-110 training)
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
    in_id = Input(shape=(1,), dtype=tf.int32,          name='asset_id')

    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))

    y = Dense(128, activation='relu')(BatchNormalization()(in_c))

    e = Embedding(N_ASSETS, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)

    z   = Concatenate()([x, Dropout(0.5)(y), e])
    z   = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


# ==============================================================================
# THE BACKTESTING ENGINE
# ==============================================================================
def run_backtester():
    print(f"STARTING {BACKTEST_DAYS}-DAY HISTORICAL BACKTEST (EXP {MODEL_ID})")
    print(f"   Tickers: {list(ASSET_CONFIGS.keys())}")
    market_df = build_market_context()

    print("Pre-loading SPY, TLT, and VIX for historical regime filter...")
    spy_raw = load_historical_base("SPY")
    tlt_raw = load_historical_base("TLT")
    vix_raw = load_historical_base("VIX")
    if spy_raw is not None:
        spy_raw = heal_data(spy_raw)
        spy_raw.index = spy_raw.index.tz_convert(PST)
    if tlt_raw is not None:
        tlt_raw = heal_data(tlt_raw)
        tlt_raw.index = tlt_raw.index.tz_convert(PST)
    if vix_raw is not None:
        vix_raw = heal_data(vix_raw)
        vix_raw.index = vix_raw.index.tz_convert(PST)

    stats_blob = client.bucket(BUCKET_NAME).blob(f"Data/Inference_v2/{STATS_FILE_NAME}")
    if not stats_blob.exists():
        sys.exit(f"Stats missing: {STATS_FILE_NAME}")
    stats = json.loads(stats_blob.download_as_string())

    results_summary = []
    trade_log = []
    model_cache = {}

    for ticker, config in ASSET_CONFIGS.items():
        print(f"\nTesting {ticker}...")

        seed = config['seed']
        if seed not in model_cache:
            model_path = f"Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction/UNIVERSAL_S{seed}_M{MODEL_ID}.h5"
            blob = client.bucket(BUCKET_NAME).blob(model_path)
            if not blob.exists():
                print(f"   Model missing for seed {seed} at {model_path}")
                continue
            local_model = f"/tmp/bt_temp_{seed}.h5"
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
            print(f"   Could not load data for {ticker}.")
            continue

        df_unscaled = df.copy()

        for col in ALL_COLS:
            if col != 'Time_Pos' and col in stats:
                df[col] = ((df[col] - stats[col]['med']) / stats[col]['iqr']).clip(-10, 10).astype('float32')

        entry_minute = config['entry_minute']
        entry_h = entry_minute // 60
        entry_m = entry_minute % 60

        tp_pct = config['tp_pct']
        sl_pct = config['sl_pct']
        avg_pnl_pct = float(config['stats']['avg_pnl'].strip('%')) / 100.0

        cm = {'strict': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
              'avg':    {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}}
        trades_taken = 0
        trades_blocked = 0
        skipped_synthetic = 0
        total_pnl = 0.0
        win_trades = 0
        loss_trades = 0

        test_start_dt = pd.Timestamp.now(tz=PST) - timedelta(days=BACKTEST_DAYS)
        df_test = df[df.index.tz_convert(PST) >= test_start_dt]
        unique_dates = np.unique(df_test.index.tz_convert(PST).date)

        for d in unique_dates:
            if d.weekday() >= 5:
                continue

            entry_dt_pst = PST.localize(datetime(d.year, d.month, d.day, entry_h, entry_m))
            entry_dt_utc = entry_dt_pst.astimezone(pytz.utc)

            if entry_dt_utc not in df.index:
                continue

            idx = df.index.get_loc(entry_dt_utc)
            if idx < TIME_STEPS:
                continue

            if pd.isna(df_unscaled.iloc[idx]['Close']):
                continue

            if df_unscaled['Volume'].iloc[idx] == 0:
                skipped_synthetic += 1
                continue

            # 1. Regime filter — trailing 30-min median
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

            # 2. Prediction
            xa = df[ASSET_COLS].values[idx - TIME_STEPS:idx].astype('float32')
            xc = df[CONTEXT_COLS].values[idx - 1].astype('float32')
            XA_b  = np.expand_dims(xa, axis=0)
            XC_b  = np.expand_dims(xc, axis=0)
            XID_b = np.array([[asset_id]], dtype=np.int32)

            prob = float(model(
                {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
                training=False
            ).numpy().flatten()[0])

            thresh = config['prob_threshold']

            # 3. Decision gate
            if prob >= thresh:
                if is_trap:
                    trades_blocked += 1
                    signal = 0
                else:
                    signal = 1
            else:
                signal = 0

            entry_price = float(df_unscaled['Close'].iloc[idx])

            # 4. Intraday simulation — exit at 1:00 PM PST (market close)
            exit_dt_utc = PST.localize(
                datetime(d.year, d.month, d.day, 13, 0)
            ).astimezone(pytz.utc)
            df_intraday = df_unscaled.loc[
                entry_dt_utc + timedelta(minutes=1):exit_dt_utc
            ]

            if df_intraday.empty:
                continue

            sl_price         = entry_price * (1 - sl_pct)
            tp_price_strict  = entry_price * (1 + tp_pct)
            tp_price_avg     = entry_price * (1 + avg_pnl_pct)

            actual_pnl   = 0.0
            hit_strict   = 0
            hit_avg      = 0
            trade_active = True
            exit_reason  = 'EOD'
            exit_price   = float(df_intraday['Close'].iloc[-1])
            max_favorable = 0.0
            max_adverse   = 0.0

            for _, row in df_intraday.iterrows():
                if not trade_active:
                    break

                bar_pnl_high = (row['High'] - entry_price) / entry_price
                bar_pnl_low  = (row['Low'] - entry_price) / entry_price
                max_favorable = max(max_favorable, bar_pnl_high)
                max_adverse   = min(max_adverse, bar_pnl_low)

                if row['Low'] <= sl_price:
                    actual_pnl   = -sl_pct
                    exit_reason  = 'SL'
                    exit_price   = sl_price
                    trade_active = False
                elif row['High'] >= tp_price_strict:
                    hit_strict   = 1
                    hit_avg      = 1
                    actual_pnl   = tp_pct
                    exit_reason  = 'TP'
                    exit_price   = tp_price_strict
                    trade_active = False
                elif row['High'] >= tp_price_avg:
                    hit_avg = 1

            if trade_active:
                actual_pnl = (df_intraday['Close'].iloc[-1] - entry_price) / entry_price

            trade_record = {
                'ticker': ticker, 'date': str(d), 'prob': round(prob, 4),
                'signal': signal, 'blocked': is_trap,
                'entry_price': round(entry_price, 4),
                'exit_price': round(exit_price, 4),
                'exit_reason': exit_reason,
                'pnl': round(actual_pnl * 100, 2),
                'max_favorable': round(max_favorable * 100, 2),
                'max_adverse': round(max_adverse * 100, 2),
            }
            trade_log.append(trade_record)

            if signal == 1:
                trades_taken += 1
                total_pnl += actual_pnl
                if actual_pnl > 0:
                    win_trades += 1
                else:
                    loss_trades += 1
                if hit_strict:
                    cm['strict']['TP'] += 1
                else:
                    cm['strict']['FP'] += 1
                if hit_avg:
                    cm['avg']['TP'] += 1
                else:
                    cm['avg']['FP'] += 1
            else:
                if hit_strict:
                    cm['strict']['FN'] += 1
                else:
                    cm['strict']['TN'] += 1
                if hit_avg:
                    cm['avg']['FN'] += 1
                else:
                    cm['avg']['TN'] += 1

        def calc_metrics(c_dict):
            prec = c_dict['TP'] / (c_dict['TP'] + c_dict['FP']) if (c_dict['TP'] + c_dict['FP']) > 0 else 0
            rec  = c_dict['TP'] / (c_dict['TP'] + c_dict['FN']) if (c_dict['TP'] + c_dict['FN']) > 0 else 0
            return prec, rec

        prec_strict, rec_strict = calc_metrics(cm['strict'])
        prec_avg, rec_avg       = calc_metrics(cm['avg'])
        avg_trade_pnl = total_pnl / trades_taken if trades_taken > 0 else 0
        win_rate      = win_trades / trades_taken if trades_taken > 0 else 0

        results_summary.append({
            'Asset': ticker, 'Trades': trades_taken, 'Blocked': trades_blocked,
            'Synthetic': skipped_synthetic,
            'P_Strict': prec_strict, 'R_Strict': rec_strict,
            'P_Avg': prec_avg, 'R_Avg': rec_avg,
            'WinRate': win_rate, 'TotalPnL': total_pnl, 'AvgPnL': avg_trade_pnl,
            'Wins': win_trades, 'Losses': loss_trades,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # DASHBOARD
    # ──────────────────────────────────────────────────────────────────────────
    W = 160
    print("\n" + "=" * W)
    print(f"  {BACKTEST_DAYS}-DAY BACKTEST RESULTS w/ REGIME FILTER  (EXP {MODEL_ID})")
    print(f"  OPT: Jul-Dec 2025, TEST: Jan-Apr 2026")
    print("=" * W)
    print(f"  {'ASSET':<6} | {'TRADES':<6} | {'W/L':<7} | {'BLOCKED':<7} | {'SYNTH':<6} | "
          f"{'STRICT TP TARGET':<22} | {'AVG PNL TARGET':<22} | {'ACTUAL PNL':<30}")
    print(f"  {'':<6} | {'TAKEN':<6} | {'':<7} | {'(TRAPS)':<7} | {'SKIP':<6} | "
          f"{'Prec.':<10} {'Recall':<10} | {'Prec.':<10} {'Recall':<10} | "
          f"{'WinRate':<8} {'Total PnL':<10} {'Avg PnL':<8}")
    print("  " + "-" * (W - 2))

    portfolio_pnl = 0.0
    portfolio_trades = 0

    for r in results_summary:
        s_prec = f"{r['P_Strict']:.1%}"
        s_rec  = f"{r['R_Strict']:.1%}"
        a_prec = f"{r['P_Avg']:.1%}"
        a_rec  = f"{r['R_Avg']:.1%}"
        wr     = f"{r['WinRate']:.1%}"
        tot    = f"{r['TotalPnL']:.1%}"
        avg    = f"{r['AvgPnL']:.2%}"
        wl     = f"{r['Wins']}/{r['Losses']}"

        portfolio_pnl    += r['TotalPnL']
        portfolio_trades += r['Trades']

        print(f"  {r['Asset']:<6} | {r['Trades']:<6} | {wl:<7} | {r['Blocked']:<7} | "
              f"{r['Synthetic']:<6} | {s_prec:<10} {s_rec:<10} | {a_prec:<10} {a_rec:<10} | "
              f"{wr:<8} {tot:<10} {avg:<8}")

    print("  " + "-" * (W - 2))
    port_avg = portfolio_pnl / portfolio_trades if portfolio_trades > 0 else 0
    print(f"  {'TOTAL':<6} | {portfolio_trades:<6} | {'':<7} | {'':<7} | {'':<6} | "
          f"{'':<10} {'':<10} | {'':<10} {'':<10} | "
          f"{'':<8} {portfolio_pnl:.1%}{'':>5} {port_avg:.2%}")
    print("=" * W)

    # ──────────────────────────────────────────────────────────────────────────
    # TRADE LOG
    # ──────────────────────────────────────────────────────────────────────────
    signal_trades = [t for t in trade_log if t['signal'] == 1]
    if signal_trades:
        print(f"\n  TRADE LOG ({len(signal_trades)} executed trades)")
        print("  " + "-" * 120)
        print(f"  {'Date':<12} {'Ticker':<6} {'Prob':>6} {'Entry':>9} {'Exit':>9} "
              f"{'Reason':<6} {'PnL%':>7} {'MFE%':>7} {'MAE%':>7} {'Blocked':<7}")
        print("  " + "-" * 120)
        for t in sorted(signal_trades, key=lambda x: x['date']):
            pnl_color = '\033[92m' if t['pnl'] > 0 else '\033[91m'
            print(f"  {t['date']:<12} {t['ticker']:<6} {t['prob']:>6.3f} "
                  f"${t['entry_price']:>8.2f} ${t['exit_price']:>8.2f} "
                  f"{t['exit_reason']:<6} {pnl_color}{t['pnl']:>+6.2f}%\033[0m "
                  f"{t['max_favorable']:>+6.2f}% {t['max_adverse']:>+6.2f}% "
                  f"{'YES' if t['blocked'] else '':<7}")
        print("  " + "-" * 120)

        pnls = [t['pnl'] for t in signal_trades]
        mfes = [t['max_favorable'] for t in signal_trades]
        maes = [t['max_adverse'] for t in signal_trades]
        print(f"\n  Trade Efficiency Stats:")
        print(f"    Avg PnL:  {np.mean(pnls):+.2f}%   Median: {np.median(pnls):+.2f}%")
        print(f"    Avg MFE:  {np.mean(mfes):+.2f}%   (max favorable excursion)")
        print(f"    Avg MAE:  {np.mean(maes):+.2f}%   (max adverse excursion)")
        if np.mean(mfes) > 0:
            capture = np.mean(pnls) / np.mean(mfes) * 100
            print(f"    Capture:  {capture:.1f}%  (PnL / MFE — how much of the move you keep)")

    # ──────────────────────────────────────────────────────────────────────────
    # BLOCKED TRADES ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────
    blocked_signals = [t for t in trade_log if t['signal'] == 0 and t['blocked']]
    if blocked_signals:
        print(f"\n  BLOCKED TRADES ANALYSIS ({len(blocked_signals)} blocked)")
        print("  " + "-" * 80)
        blocked_pnls = [t['pnl'] for t in blocked_signals]
        would_have_won = sum(1 for p in blocked_pnls if p > 0)
        print(f"    Blocked trades that would have won:  {would_have_won}/{len(blocked_signals)}")
        print(f"    Avg PnL if taken: {np.mean(blocked_pnls):+.2f}%")
        print(f"    Filter saved:     {-sum(p for p in blocked_pnls if p < 0):.2f}% total loss avoided")

    print(f"\nDone. {BACKTEST_DAYS}-day backtest complete for Exp-{MODEL_ID}.")


if __name__ == "__main__":
    run_backtester()
