"""
Daily Scoring — Exp-104 (Defense sector)
=========================================
*** DISABLED 2026-04-25: Exp-103/104 needs retraining before redeployment ***

Self-contained script. No external module imports.
Optimized on Jul-Dec 2025, tested Jan-Apr 2026.

Deploys 4 tickers:
  BBAI  seed=10403  entry=06:15  prob>0.575  TP=25%  SL=2%
  LUNR  seed=10403  entry=06:20  prob>0.525  TP=21%  SL=5%
  SPIR  seed=10402  entry=06:30  prob>0.530  TP=15%  SL=6%
  RCAT  seed=10403  entry=06:10  prob>0.525  TP=16%  SL=8%

Run AFTER daily_data_all.py has finished updating GCS.
Usage:  python daily_score_exp104.py
"""

import os
import sys
import json
import gc
import subprocess

import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    BatchNormalization, Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2
from google.cloud import storage as gcs_storage

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None

# ==============================================================================
# CONSTANTS
# ==============================================================================
BUCKET_NAME    = "time-series-bucket-ys"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models/UNIVERSAL/Direction"
BQ_TABLE       = "ys-time-series.scoring_logs.live_predictions"
INFERENCE_DIR  = "Data/Inference"
TAIL_DAYS      = 7
TIME_STEPS     = 240
EMBED_DIM      = 16
PST            = pytz.timezone('America/Los_Angeles')

TICKER_FILE_MAP = {'BITF': 'KEEL'}

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ROC_HORIZONS = [5, 30, 60, 120]

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

MODEL_ID = "104_DEFENSE"
STATS_PATH = f"Data/Inference/scaling_stats_experiment_{MODEL_ID}.json"

TRAIN_TEACHERS = [
    'AVAV', 'KTOS', 'PLTR', 'BBAI',
    'RKLB', 'ASTS', 'LUNR', 'SPIR',
    'JOBY', 'NNDM', 'MVIS',
]
ASSET_INDEX = {t: i for i, t in enumerate(TRAIN_TEACHERS)}
N_ASSETS  = len(TRAIN_TEACHERS) + 1   # 12
OTHER_IDX = len(TRAIN_TEACHERS)        # 11

CONTEXT_ASSETS = ['ITA', 'VIX', 'TLT', 'IWM']

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS.append('Time_Pos')

ALL_COLS = ASSET_COLS + CONTEXT_COLS

ASSET_CONFIGS = {
    'BBAI': {
        'seed': 10403, 'prob_threshold': 0.575,
        'tp_pct': 0.25, 'sl_pct': 0.02, 'entry_minute': 375,
        'vix_threshold': 10.0, 'tlt_crash': -0.50, 'spy_ignore': 0.00,
        'stats': {'win_rate': '50.0%', 'avg_pnl': '3.03%', 'freq': '0.6/mo'},
    },
    'LUNR': {
        'seed': 10403, 'prob_threshold': 0.525,
        'tp_pct': 0.21, 'sl_pct': 0.05, 'entry_minute': 380,
        'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.10,
        'stats': {'win_rate': '50.0%', 'avg_pnl': '2.90%', 'freq': '4.6/mo'},
    },
    'SPIR': {
        'seed': 10402, 'prob_threshold': 0.530,
        'tp_pct': 0.15, 'sl_pct': 0.06, 'entry_minute': 390,
        'vix_threshold': 15.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
        'stats': {'win_rate': '60.0%', 'avg_pnl': '1.85%', 'freq': '4.3/mo'},
    },
    'RCAT': {
        'seed': 10403, 'prob_threshold': 0.525,
        'tp_pct': 0.16, 'sl_pct': 0.08, 'entry_minute': 370,
        'vix_threshold': 10.0, 'tlt_crash': -0.08, 'spy_ignore': 0.00,
        'stats': {'win_rate': '55.6%', 'avg_pnl': '1.79%', 'freq': '2.6/mo'},
    },
}

# ==============================================================================
# GCS CLIENT
# ==============================================================================
gcs_client = gcs_storage.Client()

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_master(ticker, tail_days=None):
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
        if tail_days is not None:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
            df = df[df.index >= cutoff]
            if df.empty:
                print(f"      {ticker}: no recent data — skipping")
                return None
        return df
    except Exception as e:
        print(f"   Failed to load {ticker}: {e}")
        return None


# ==============================================================================
# DATA HEALING
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
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
        df = load_master(ticker, tail_days=TAIL_DAYS)
        if df is None:
            print(f"      Could not load context asset {ticker}")
            continue
        df = df.sort_index()
        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
        gc.collect()
    if not frames:
        raise RuntimeError("No context assets loaded.")
    return pd.concat(frames, axis=1).ffill().fillna(0)


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def engineer_features(df, market_df):
    daily_c = df['Close'].resample('D').last().shift(1).reindex(
        df.index, method='ffill')
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


def engineer_asset(ticker, market_df):
    df = load_master(ticker, tail_days=TAIL_DAYS)
    if df is None:
        return None
    df = df.sort_index()
    df = heal_data(df)
    return engineer_features(df, market_df)


# ==============================================================================
# SCALING
# ==============================================================================
def apply_scaling(df, stats):
    df_s = df.copy()
    for col in ALL_COLS:
        if col == 'Time_Pos':
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


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
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


# ==============================================================================
# MODEL LOADER
# ==============================================================================
_model_cache = {}


def get_model(seed):
    if seed in _model_cache:
        return _model_cache[seed]
    print(f"      Loading model seed {seed} ({MODEL_ID})...")
    blob_path  = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{MODEL_ID}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{MODEL_ID}.h5"
    if not os.path.exists(local_path):
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
    model = build_model()
    model.load_weights(local_path)
    _model_cache[seed] = model
    return model


# ==============================================================================
# SCORING ENGINE
# ==============================================================================
def score_ticker(ticker, cfg, df_scaled):
    if len(df_scaled) == 0:
        return None
    row_idx    = len(df_scaled) - 1
    score_date = df_scaled.index[row_idx].tz_convert(PST).date()
    if row_idx < TIME_STEPS:
        print(f"      {ticker}: not enough history ({row_idx} < {TIME_STEPS})")
        return None
    if 'Volume' in df_scaled.columns and float(df_scaled['Volume'].iloc[row_idx]) == 0:
        print(f"      {ticker}: synthetic bar (Volume=0) — skipping")
        return None
    XA = df_scaled[ASSET_COLS].values
    XC = df_scaled[CONTEXT_COLS].values
    XA_b  = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]
    XC_b  = XC[row_idx - 1][np.newaxis]
    XID_b = np.array([[ASSET_INDEX.get(ticker, OTHER_IDX)]], dtype=np.int32)
    model = get_model(cfg['seed'])
    prob = float(model(
        {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
        training=False
    ).numpy().flatten()[0])
    signal = prob >= cfg['prob_threshold']
    bar_ts = df_scaled.index[row_idx]
    close  = (float(df_scaled['Close'].iloc[row_idx])
              if 'Close' in df_scaled.columns else float('nan'))
    return {
        'ticker': ticker, 'experiment': MODEL_ID, 'seed': cfg['seed'],
        'prob': round(prob, 4), 'threshold': cfg['prob_threshold'],
        'signal': signal, 'tp_pct': cfg['tp_pct'], 'sl_pct': cfg['sl_pct'],
        'entry_minute': cfg['entry_minute'], 'score_date': score_date,
        'bar_ts': bar_ts, 'close': round(close, 4),
    }


# ==============================================================================
# REGIME FILTER
# ==============================================================================
def load_regime_data(score_date, market_df, run_ts):
    window_end   = run_ts.astimezone(PST)
    window_start = window_end - timedelta(minutes=30)

    vix_level = None
    df_vix = load_master('VIX', tail_days=TAIL_DAYS)
    if df_vix is not None:
        df_vix = heal_data(df_vix)
        dpst = df_vix.copy()
        dpst.index = dpst.index.tz_convert(PST)
        bars = dpst[(dpst.index >= window_start) & (dpst.index <= window_end)]
        if len(bars):
            vix_level = float(bars['Close'].median())

    tlt_gap = None
    if 'TLT_Anchor_ROC' in market_df.columns:
        dpst = market_df.copy()
        dpst.index = dpst.index.tz_convert(PST)
        bars = dpst[(dpst.index >= window_start) & (dpst.index <= window_end)]
        if len(bars):
            tlt_gap = float(bars['TLT_Anchor_ROC'].median()) * 100

    spy_gap = None
    df_spy = load_master('SPY', tail_days=TAIL_DAYS)
    if df_spy is not None:
        df_spy = heal_data(df_spy)
        dpst = df_spy.copy()
        dpst.index = dpst.index.tz_convert(PST)
        prev = dpst[dpst.index.date < score_date]
        bars = dpst[(dpst.index >= window_start) & (dpst.index <= window_end)]
        if len(prev) > 0 and len(bars) > 0:
            prev_close  = float(prev['Close'].iloc[-1])
            entry_close = float(bars['Close'].median())
            if prev_close != 0:
                spy_gap = ((entry_close - prev_close) / prev_close) * 100

    return {'vix': vix_level, 'tlt_gap': tlt_gap, 'spy_gap': spy_gap}


def _filter_pass(cfg, regime):
    vix_bad = regime['vix']     is not None and regime['vix']     > cfg['vix_threshold']
    tlt_bad = regime['tlt_gap'] is not None and regime['tlt_gap'] < cfg['tlt_crash']
    spy_bad = regime['spy_gap'] is not None and regime['spy_gap'] > cfg['spy_ignore']
    blocked = vix_bad and tlt_bad and spy_bad
    return not blocked, not vix_bad, not tlt_bad, not spy_bad


# ==============================================================================
# DASHBOARD
# ==============================================================================
def print_dashboard(results, run_ts, regime):
    W       = 155
    pst_str = run_ts.astimezone(PST).strftime('%Y-%m-%d %H:%M PST')
    print()
    print("=" * W)
    print(f"  EXP-104 DAILY SIGNALS   run: {pst_str}")
    print(f"  OPT: Jul-Dec 2025, TEST: Jan-Apr 2026")
    if regime:
        vix_s = f"{regime['vix']:.1f}"       if regime['vix']     is not None else "N/A"
        tlt_s = f"{regime['tlt_gap']:+.2f}%" if regime['tlt_gap'] is not None else "N/A"
        spy_s = f"{regime['spy_gap']:+.2f}%" if regime['spy_gap'] is not None else "N/A"
        print(f"  VIX={vix_s}  TLT={tlt_s}  SPY={spy_s}")
    print("=" * W)
    print(f"  {'Ticker':<6}  {'Seed':>5}  {'SIG':>6}  {'Prob':>6}  {'Thr':>5}  "
          f"{'Entry':>7}  {'TP tgt':>13}  {'SL stop':>13}  {'Close':>8}  "
          f"{'Score bar':<16}  {'VIX<':>5}  {'TLT>':>6}  {'SPY<':>6}  "
          f"{'WR':>6}  {'AvgPnL':>8}  {'Freq':>8}  FILT")
    print("-" * W)
    for r in results:
        cfg = ASSET_CONFIGS.get(r['ticker'], {})
        trade_ok = True
        if regime and cfg:
            trade_ok, _, _, _ = _filter_pass(cfg, regime)
        if r['signal']:
            sig_str = ("\033[92mGO\033[0m     " if trade_ok
                       else "\033[31mBLOCK\033[0m  ")
        else:
            sig_str = "--     "
        stats    = cfg.get('stats', {})
        win_str  = stats.get('win_rate', '--%')
        pnl_str  = stats.get('avg_pnl', '--%')
        freq_str = stats.get('freq', '--/mo')
        close  = r['close']
        tp_str = f"${close*(1+r['tp_pct']):.2f}(+{r['tp_pct']*100:.0f}%)"
        sl_str = f"${close*(1-r['sl_pct']):.2f}(-{r['sl_pct']*100:.0f}%)"
        vix_t = f"{cfg['vix_threshold']:.0f}" if cfg else "--"
        tlt_t = f"{cfg['tlt_crash']:+.2f}"    if cfg else "--"
        spy_t = f"{cfg['spy_ignore']:+.2f}"   if cfg else "--"
        if regime is None:
            filt_str = " --"
        elif trade_ok:
            filt_str = " OK"
        else:
            parts = []
            if regime['vix']     is not None:
                parts.append(f"VIX:{regime['vix']:.1f}>{cfg['vix_threshold']:.0f}")
            if regime['tlt_gap'] is not None:
                parts.append(f"TLT:{regime['tlt_gap']:+.2f}<{cfg['tlt_crash']:+.2f}")
            if regime['spy_gap'] is not None:
                parts.append(f"SPY:{regime['spy_gap']:+.2f}>{cfg['spy_ignore']:+.2f}")
            filt_str = f" BLOCKED {' & '.join(parts)}"
        em        = r['entry_minute']
        entry_str = f"{em // 60}:{em % 60:02d}"
        prob_str  = f"{r['prob']:.3f}"
        bar_str   = r['bar_ts'].astimezone(PST).strftime('%m-%d %H:%M PST')
        print(
            f"  {r['ticker']:<6}  {r['seed']:>5}  "
            f"{sig_str}  {prob_str:>6}  {r['threshold']:>5.3f}  "
            f"{entry_str:>7}  {tp_str:>13}  {sl_str:>13}  {close:>8.2f}  "
            f"{bar_str:<16}  {vix_t:>5}  {tlt_t:>6}  {spy_t:>6}  "
            f"{win_str:>6}  {pnl_str:>8}  {freq_str:>8}{filt_str}"
        )
    print("=" * W)
    signals = [r for r in results if r['signal']]
    go_trades, blocked = [], []
    for r in signals:
        cfg = ASSET_CONFIGS.get(r['ticker'])
        if cfg and regime:
            ok, _, _, _ = _filter_pass(cfg, regime)
            (go_trades if ok else blocked).append(r)
        else:
            go_trades.append(r)
    print(f"  GO signals: {len(signals)}/{len(results)}", end="")
    if signals:
        print(f"  ->  {', '.join(r['ticker'] for r in signals)}", end="")
    if blocked:
        print(f"  |  blocked: {', '.join(r['ticker'] for r in blocked)}", end="")
    print()
    if go_trades:
        print(f"  TRADE: {', '.join(r['ticker'] for r in go_trades)}"
              f"  ({len(go_trades)} position{'s' if len(go_trades) != 1 else ''})")
    print("=" * W)


# ==============================================================================
# BIGQUERY LOGGER
# ==============================================================================
_BQ_SCHEMA_DEFS = [
    ("run_timestamp",  "TIMESTAMP"),
    ("bar_timestamp",  "TIMESTAMP"),
    ("score_date",     "DATE"),
    ("experiment",     "STRING"),
    ("ticker",         "STRING"),
    ("seed",           "INT64"),
    ("prob",           "FLOAT64"),
    ("threshold",      "FLOAT64"),
    ("signal",         "STRING"),
    ("tp_pct",         "FLOAT64"),
    ("sl_pct",         "FLOAT64"),
    ("entry_minute",   "INT64"),
    ("close",          "FLOAT64"),
]


def _ensure_bq_schema(bq_client):
    for col_name, col_type in _BQ_SCHEMA_DEFS:
        sql = (f"ALTER TABLE `{BQ_TABLE}` "
               f"ADD COLUMN IF NOT EXISTS `{col_name}` {col_type}")
        bq_client.query(sql).result()


def log_to_bigquery(results, run_ts):
    if bigquery is None:
        print("   BigQuery not available — skipping")
        return
    if not results:
        return
    bq_client = bigquery.Client()
    _ensure_bq_schema(bq_client)
    rows = []
    for r in results:
        rows.append({
            'run_timestamp': run_ts.isoformat(),
            'bar_timestamp': r['bar_ts'].isoformat(),
            'score_date':    str(r['score_date']),
            'experiment':    r['experiment'],
            'ticker':        r['ticker'],
            'seed':          r['seed'],
            'prob':          r['prob'],
            'threshold':     r['threshold'],
            'signal':        'GO' if r['signal'] else 'NO',
            'tp_pct':        r['tp_pct'],
            'sl_pct':        r['sl_pct'],
            'entry_minute':  r['entry_minute'],
            'close':         r['close'],
        })
    try:
        errors = bq_client.insert_rows_json(BQ_TABLE, rows)
        if errors:
            print(f"   BigQuery errors: {errors}")
        else:
            print(f"   Logged {len(rows)} rows to {BQ_TABLE}")
    except Exception as e:
        print(f"   BigQuery failed: {e}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    run_ts = datetime.now(pytz.utc)
    print(f"Exp-104 Daily Scoring Pipeline")
    print(f"   Run: {run_ts.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"   Model: {MODEL_ID}  |  Tickers: {list(ASSET_CONFIGS.keys())}")
    print(f"   TIME_STEPS={TIME_STEPS}")

    print(f"\n   Loading scaling stats ({MODEL_ID})...")
    try:
        blob  = gcs_client.bucket(BUCKET_NAME).blob(STATS_PATH)
        stats = json.loads(blob.download_as_text())
        print(f"   Loaded {len(stats)} features")
    except Exception as e:
        print(f"   Failed to load stats: {e}")
        return

    market_df = build_market_context()

    print(f"   Pre-loading models...")
    needed_seeds = sorted(set(cfg['seed'] for cfg in ASSET_CONFIGS.values()))
    for seed in needed_seeds:
        try:
            get_model(seed)
        except Exception as e:
            print(f"      Failed to load seed {seed}: {e}")
    print(f"   Models ready: {needed_seeds}")

    print(f"   Scoring...")
    results = []
    for ticker, cfg in ASSET_CONFIGS.items():
        asset_id = ASSET_INDEX.get(ticker, OTHER_IDX)
        print(f"   {ticker}  seed={cfg['seed']}  "
              f"prob>{cfg['prob_threshold']}  id={asset_id}")
        df_raw = engineer_asset(ticker, market_df)
        if df_raw is None:
            print(f"      Skipping {ticker}")
            continue
        df_scaled = apply_scaling(df_raw, stats)
        result = score_ticker(ticker, cfg, df_scaled)
        if result is not None:
            results.append(result)

    regime = None
    if results:
        score_date = results[0]['score_date']
        print(f"\n   Computing regime for {score_date}...")
        regime = load_regime_data(score_date, market_df, run_ts)
        vix_s = f"{regime['vix']:.1f}"       if regime['vix']     is not None else "N/A"
        tlt_s = f"{regime['tlt_gap']:+.2f}%" if regime['tlt_gap'] is not None else "N/A"
        spy_s = f"{regime['spy_gap']:+.2f}%" if regime['spy_gap'] is not None else "N/A"
        print(f"   VIX={vix_s}  TLT={tlt_s}  SPY={spy_s}")

    print_dashboard(results, run_ts, regime)

    print("\nLogging to BigQuery...")
    log_to_bigquery(results, run_ts)
    print("\nDone.")


if __name__ == "__main__":
    main()
