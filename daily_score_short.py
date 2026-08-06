"""
Daily Scoring — SHORT Strategy (FOLD 414)
==========================================
Self-contained. Scores RULE_1 tickers under SHORT fold 414 and prints dashboard.

Strategy: Best val_a ensemble config from fold 414 per ticker.
  - No family agreement gate (single fold).
  - No regime filter (SHORT is a bearish hedge).
  - Modes: static, atr_sl, grace only (no limit_entry).
  - 15-seed ensemble scoring (seeds 41401-41415).

Tickers (RULE_1): BE, IREN, OKLO, ONDS, RKLB, WOLF

Run AFTER daily_stitch_master.py has finished updating GCS.
"""

import os
import sys
import json
import gc
import subprocess

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

try:
    import tf_keras
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tf_keras", "--quiet"])

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

import tensorflow as tf

from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2
from google.cloud import storage as gcs_storage

if tf.config.list_physical_devices('GPU'):
    mixed_precision.set_global_policy('mixed_float16')

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None


# ==============================================================================
# CONSTANTS
# ==============================================================================
BUCKET_NAME    = "time-series-bucket-ys"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
BQ_PREDICTIONS = "ys-time-series.scoring_logs.live_predictions"
BQ_WF_TABLE    = "ys-time-series.scoring_logs.wf_optimization_results"
INFERENCE_DIR  = "Data/Inference_v2"
TAIL_DAYS      = 25
TIME_STEPS     = 240
EMBED_DIM      = 16
N_ASSETS       = 31
OTHER_IDX      = 30
PST            = pytz.timezone('America/Los_Angeles')
MIN_PRICE      = 2.00

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
CONTEXT_COLS.append('Time_Pos')

ALL_COLS = ASSET_COLS + CONTEXT_COLS

# ── SHORT-specific ──────────────────────────────────────────────────────────
SHORT_TICKERS = ['BE', 'IREN', 'OKLO', 'ONDS', 'RKLB', 'WOLF']

SHORT_FOLD       = 414
SHORT_EXPERIMENT = '414_XSECTOR'
SHORT_SEED_LIST  = list(range(41401, 41416))

BQ_MODEL_VERSIONS = ('robust_wf_v12_short', 'robust_wf_v12_short_a1',
                      'robust_wf_v12_short_414')
BQ_ALLOWED_MODES  = ('static', 'atr_sl', 'grace')

SHORT_CONFIGS = {}

gcs_client = gcs_storage.Client()


# ==============================================================================
# BQ CONFIG SELECTION — Best val_a from fold 414
# ==============================================================================
def load_short_configs_from_bq():
    """Load best ensemble config per ticker from fold 414."""
    if bigquery is None:
        print("  ERROR: BigQuery not available")
        return 0

    bq_client = bigquery.Client()
    mv_str = ", ".join(f"'{v}'" for v in BQ_MODEL_VERSIONS)
    modes_str = ", ".join(f"'{m}'" for m in BQ_ALLOWED_MODES)

    query = f"""
    SELECT ticker, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, sl_mult, grace_minutes,
           opt_avg_pnl, val_a_avg_pnl, val_a_win_rate, val_a_count,
           is_ensemble, model_version, seed
    FROM `{BQ_WF_TABLE}`
    WHERE model_version IN ({mv_str})
      AND opt_avg_pnl > 0
      AND tpsl_mode IN ({modes_str})
      AND is_ensemble = TRUE
    ORDER BY ticker, model_version
    """
    df = bq_client.query(query).to_dataframe()

    if len(df) == 0:
        print("  WARNING: No SHORT configs found in BQ")
        return 0

    df['fold'] = df['model_version'].str.extract(r'_(\d+)$').astype(float)
    short_mask = df['model_version'].str.contains('short')
    a1_mask = df['model_version'].str.contains('_a1')
    df.loc[short_mask & ~a1_mask & df['fold'].isna(), 'fold'] = 400.0
    df.loc[a1_mask & df['fold'].isna(), 'fold'] = 410.0

    fold_df = df[df['fold'] == SHORT_FOLD]
    if len(fold_df) == 0:
        fold_df = df
        print(f"  WARNING: No fold {SHORT_FOLD} configs — using all folds")

    n_selected = 0
    for ticker in SHORT_TICKERS:
        t_df = fold_df[fold_df['ticker'] == ticker]
        if len(t_df) == 0:
            t_df = df[df['ticker'] == ticker]
        if len(t_df) == 0:
            continue

        best_idx = t_df['val_a_avg_pnl'].idxmax()
        row = t_df.loc[best_idx]

        def _safe(val, default=0):
            try:
                if pd.isna(val):
                    return default
            except (TypeError, ValueError):
                pass
            return val

        cfg = {
            'prob_threshold': float(row['prob_thresh']),
            'entry_minute': int(_safe(row['entry_minute'], 375)),
            'tpsl_mode': row['tpsl_mode'],
            'tp_pct': float(row['tp_pct']),
            'sl_pct': float(row['sl_pct']),
            'sl_mult': float(_safe(row['sl_mult'], 0)),
            'grace_minutes': int(_safe(row['grace_minutes'], 0)),
            'val_a_avg_pnl': float(_safe(row['val_a_avg_pnl'], 0)),
            'val_a_win_rate': float(_safe(row['val_a_win_rate'], 0)),
            'val_a_count': int(_safe(row['val_a_count'], 0)),
            'model_version': row['model_version'],
        }
        SHORT_CONFIGS[ticker] = cfg
        n_selected += 1

    return n_selected


# ==============================================================================
# DATA LOADING
# ==============================================================================
_master_cache = {}


def load_master(ticker, tail_days=None):
    cache_key = (ticker, tail_days)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
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
                return None
        _master_cache[cache_key] = df
        return df
    except Exception as e:
        print(f"      load_master({ticker}): FAILED — {e}")
        return None


# ==============================================================================
# DATA HEALING
# ==============================================================================
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
# MARKET CONTEXT
# ==============================================================================
def build_market_context():
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker, tail_days=TAIL_DAYS)
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
        del df, daily_c
        gc.collect()
    if not frames:
        raise RuntimeError("No context assets loaded.")
    return pd.concat(frames, axis=1, sort=True).ffill().fillna(0)


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
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
        if col in ('Time_Pos', 'Bar_Is_Real'):
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
    n_ac = len(ASSET_COLS)
    n_cc = len(CONTEXT_COLS)
    in_a  = Input(shape=(TIME_STEPS, n_ac), name='asset_in')
    in_c  = Input(shape=(n_cc,),            name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32, name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
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
    blob_path  = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{SHORT_EXPERIMENT}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{SHORT_EXPERIMENT}.h5"
    blob = gcs_client.bucket(BUCKET_NAME).blob(blob_path)
    if os.path.exists(local_path):
        blob.reload()
        if os.path.getsize(local_path) != blob.size:
            os.remove(local_path)
    if not os.path.exists(local_path):
        blob.download_to_filename(local_path)
    model = build_model()
    model.load_weights(local_path)
    _model_cache[seed] = model
    return model


# ==============================================================================
# DYNAMIC TP/SL
# ==============================================================================
def _daily_atr_pct(df_raw, row_idx, period=14):
    daily = df_raw[['High', 'Low', 'Close']].resample('D').agg(
        {'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    if len(daily) < 2:
        return 0.05
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([
        daily['High'] - daily['Low'],
        (daily['High'] - prev_close).abs(),
        (daily['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(period, min_periods=max(period // 2, 3)).mean()
    atr_pct = atr_series / daily['Close']
    atr_pct = atr_pct.reindex(df_raw.index, method='ffill')
    val = float(atr_pct.iloc[row_idx]) if row_idx < len(atr_pct) else np.nan
    if np.isnan(val) or val <= 0:
        return 0.05
    return val


def compute_dynamic_tpsl(cfg, df_raw, row_idx):
    mode = cfg.get('tpsl_mode', 'static')

    if mode == 'atr_sl':
        atr_pct = _daily_atr_pct(df_raw, row_idx)
        sl = min(max(atr_pct * cfg['sl_mult'], 0.01), 0.25)
        return cfg['tp_pct'], sl, {'atr_pct': round(atr_pct, 6), 'mode': 'atr_sl'}

    elif mode == 'grace':
        return cfg['tp_pct'], cfg['sl_pct'], {
            'mode': 'grace',
            'grace_minutes': cfg.get('grace_minutes', 0),
        }

    return cfg['tp_pct'], cfg['sl_pct'], {'mode': 'static'}


# ==============================================================================
# SCORING ENGINE
# ==============================================================================
def score_short_ticker(ticker, cfg, df_scaled, df_raw):
    if len(df_scaled) == 0:
        print(f"      {ticker}: SKIP — df_scaled is empty")
        return None

    entry_min = cfg['entry_minute']
    today = datetime.now(PST).date()
    target_pst = PST.localize(datetime(today.year, today.month, today.day,
                                       entry_min // 60, entry_min % 60))
    target_utc = target_pst.astimezone(pytz.utc)

    candidates = df_scaled.index[df_scaled.index <= target_utc]
    if len(candidates) == 0:
        print(f"      {ticker}: SKIP — no bars at or before entry {entry_min}")
        return None

    row_idx = df_scaled.index.get_loc(candidates[-1])
    bar_ts_pst = df_scaled.index[row_idx].tz_convert(PST)
    score_date = bar_ts_pst.date()

    if score_date != today:
        print(f"      {ticker}: SKIP — nearest bar date {score_date} != today {today}")
        return None

    bar_minute = bar_ts_pst.hour * 60 + bar_ts_pst.minute
    if bar_minute != entry_min:
        print(f"      {ticker}: SKIP — bar at minute {bar_minute} != entry_minute {entry_min}")
        return None

    if row_idx < TIME_STEPS:
        print(f"      {ticker}: SKIP — not enough history: {row_idx} < {TIME_STEPS}")
        return None

    if 'Volume' in df_scaled.columns and float(df_scaled['Volume'].iloc[row_idx]) == 0:
        print(f"      {ticker}: SKIP — Volume=0 at entry bar (synthetic)")
        return None

    if df_raw is not None and 'Close' in df_raw.columns:
        raw_candidates = df_raw.index[df_raw.index <= df_scaled.index[row_idx]]
        if len(raw_candidates) > 0:
            raw_close = float(df_raw['Close'].iloc[df_raw.index.get_loc(raw_candidates[-1])])
            if raw_close < MIN_PRICE:
                print(f"      {ticker}: SKIP — price ${raw_close:.2f} < ${MIN_PRICE:.2f}")
                return None

    XA = df_scaled[ASSET_COLS].values
    XC = df_scaled[CONTEXT_COLS].values
    XA_b  = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]
    XC_b  = XC[row_idx - 1][np.newaxis]
    XID_b = np.array([[OTHER_IDX]], dtype=np.int32)

    all_probs = []
    for seed in SHORT_SEED_LIST:
        try:
            m = get_model(seed)
            p = float(m(
                {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
                training=False
            ).numpy().flatten()[0])
            all_probs.append(p)
        except Exception:
            continue

    if not all_probs:
        print(f"      {ticker}: SKIP — no ensemble models loaded")
        return None

    prob = float(np.mean(all_probs))
    signal = prob >= cfg['prob_threshold']
    bar_ts = df_scaled.index[row_idx]
    close = float(df_scaled['Close'].iloc[row_idx]) if 'Close' in df_scaled.columns else float('nan')

    raw_row_idx = None
    if df_raw is not None and len(df_raw) > 0:
        raw_candidates = df_raw.index[df_raw.index <= bar_ts]
        if len(raw_candidates) > 0:
            raw_row_idx = df_raw.index.get_loc(raw_candidates[-1])

    tp_pct, sl_pct, tpsl_info = compute_dynamic_tpsl(cfg, df_raw, raw_row_idx) \
        if raw_row_idx is not None else (cfg['tp_pct'], cfg['sl_pct'], {'mode': 'static'})

    return {
        'ticker': ticker,
        'experiment': SHORT_EXPERIMENT,
        'seed': -1,
        'prob': round(prob, 4),
        'threshold': cfg['prob_threshold'],
        'signal': signal,
        'tp_pct': tp_pct,
        'sl_pct': sl_pct,
        'entry_minute': cfg['entry_minute'],
        'score_date': score_date,
        'bar_ts': bar_ts,
        'close': round(close, 4),
        'filter_ok': True,
        'filter_status': 'OK',
        'tpsl_mode': tpsl_info.get('mode', 'static'),
        'atr_pct_entry': tpsl_info.get('atr_pct'),
        'grace_minutes': tpsl_info.get('grace_minutes'),
        'n_seeds': len(all_probs),
        'val_a_avg_pnl': cfg.get('val_a_avg_pnl', 0),
        'val_a_win_rate': cfg.get('val_a_win_rate', 0),
    }


# ==============================================================================
# DASHBOARD
# ==============================================================================
_MODE_LABELS = {'static': 'STATIC', 'atr_sl': 'ATR-SL', 'grace': 'GRACE'}


def print_short_dashboard(results, run_ts):
    W = 120
    pst_str = run_ts.astimezone(PST).strftime('%Y-%m-%d %H:%M PST')

    print()
    print(f"  {'='*W}")
    print(f"  SHORT SIGNALS — FOLD {SHORT_FOLD}   {pst_str}")
    print(f"  Tickers: {', '.join(SHORT_TICKERS)}")
    print(f"  {'='*W}")

    if not results:
        print(f"\n  No tickers scored.")
        print(f"  {'='*W}")
        return

    # ── Signal table ────────────────────────────────────────────────────────
    print()
    print(f"  {'─'*W}")
    print(f"  {'Ticker':<6}  {'SIG':<7}  "
          f"{'Short@':>10}  {'TP@':>10}  {'SL@':>10}  "
          f"{'Prob':>5} / {'Thr':>5}  {'Entry':>5}  "
          f"{'Close':>8}  {'Mode':<7}  {'Seeds':>5}  {'val_a%':>7}")
    print(f"  {'─'*W}")

    for r in results:
        if r['signal']:
            sig_str = "\033[92m GO  \033[0m"
        else:
            sig_str = "  --  "

        close = r['close']
        tp_price = close * (1 - r['tp_pct'])
        sl_price = close * (1 + r['sl_pct'])
        mode_label = _MODE_LABELS.get(r['tpsl_mode'], '??')
        em = r['entry_minute']
        entry_str = f"{em // 60}:{em % 60:02d}"

        print(f"  {r['ticker']:<6}  {sig_str}  "
              f"${close:>8.2f}  ${tp_price:>8.2f}  ${sl_price:>8.2f}  "
              f"{r['prob']:>5.3f} / {r['threshold']:>5.3f}  {entry_str:>5}  "
              f"{close:>8.2f}  {mode_label:<7}  {r['n_seeds']:>5}  "
              f"{r['val_a_avg_pnl']*100:>+6.2f}%")

    # ── Summary ─────────────────────────────────────────────────────────────
    signals   = [r for r in results if r['signal']]
    go_trades = signals

    print()
    print(f"  {'='*W}")
    print(f"  SUMMARY")
    print(f"  {'─'*W}")
    print(f"  Scored:    {len(results)} tickers")
    print(f"  Signals:   {len(signals)}/{len(results)}" +
          (f"  ({', '.join(r['ticker'] for r in signals)})" if signals else ""))

    if go_trades:
        print()
        print(f"  \033[91mSHORT: {', '.join(r['ticker'] for r in go_trades)}"
              f"  ({len(go_trades)} position{'s' if len(go_trades) != 1 else ''})\033[0m")
        print()
        print(f"  {'─'*W}")
        print(f"  SHORT ORDERS")
        print(f"  {'─'*W}")
        print(f"  {'Ticker':<6}  {'SHORT @':>10}  {'TP target':>10}  "
              f"{'SL stop':>10}  {'Mode':<7}  {'Exit by':>10}")
        print(f"  {'─'*W}")

        for r in go_trades:
            c = r['close']
            tp = c * (1 - r['tp_pct'])
            sl = c * (1 + r['sl_pct'])
            mode_label = _MODE_LABELS.get(r['tpsl_mode'], '??')

            print(f"  {r['ticker']:<6}  "
                  f"\033[91m${c:>8.2f}\033[0m  "
                  f"${tp:>8.2f}  ${sl:>8.2f}  "
                  f"{mode_label:<7}  {'1:00PM PT':>10}")

    # ── Config details ──────────────────────────────────────────────────────
    print()
    print(f"  {'─'*W}")
    print(f"  CONFIG DETAILS (from BQ fold {SHORT_FOLD})")
    print(f"  {'─'*W}")
    print(f"  {'Ticker':<6}  {'Mode':<7}  {'TP%':>5}  {'SL%':>5}  {'Thr':>5}  "
          f"{'val_a%':>7}  {'WR':>5}  {'Seeds':>5}")
    print(f"  {'─'*W}")

    for ticker in SHORT_TICKERS:
        cfg = SHORT_CONFIGS.get(ticker)
        if cfg is None:
            print(f"  {ticker:<6}  — no BQ config —")
            continue
        sl_str = f"{cfg['sl_pct']*100:.1f}" if cfg['tpsl_mode'] != 'atr_sl' else 'ATR'
        print(f"  {ticker:<6}  {_MODE_LABELS.get(cfg['tpsl_mode'], '??'):<7}  "
              f"{cfg['tp_pct']*100:>4.1f}  {sl_str:>5}  {cfg['prob_threshold']:>4.2f}  "
              f"{cfg['val_a_avg_pnl']*100:>+6.2f}%  "
              f"{cfg['val_a_win_rate']*100:>4.0f}%  15")

    print(f"  {'='*W}")


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
    ("filter_status",  "STRING"),
    ("tpsl_mode",      "STRING"),
    ("atr_pct_entry",  "FLOAT64"),
    ("strategy",       "STRING"),
]


def _ensure_bq_schema(bq_client):
    for col_name, col_type in _BQ_SCHEMA_DEFS:
        sql = (f"ALTER TABLE `{BQ_PREDICTIONS}` "
               f"ADD COLUMN IF NOT EXISTS `{col_name}` {col_type}")
        bq_client.query(sql).result()


def log_to_bigquery(results, run_ts):
    if bigquery is None:
        print("   BigQuery not available — skipping")
        return
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
            'filter_status': r.get('filter_status', 'OK'),
            'tpsl_mode':     r.get('tpsl_mode', 'static'),
            'atr_pct_entry': r.get('atr_pct_entry'),
            'strategy':      'SHORT_414',
        })
    if not rows:
        return
    bq_client = bigquery.Client()
    _ensure_bq_schema(bq_client)
    try:
        errors = bq_client.insert_rows_json(BQ_PREDICTIONS, rows)
        if errors:
            print(f"   BigQuery errors: {errors}")
        else:
            print(f"   Logged {len(rows)} rows to {BQ_PREDICTIONS} (strategy=SHORT_414)")
    except Exception as e:
        print(f"   BigQuery failed: {e}")


# ==============================================================================
# STITCH FRESHNESS CHECK
# ==============================================================================
def _check_stitch_freshness():
    blob = gcs_client.bucket(BUCKET_NAME).blob("Data/Inference/pipeline_status.json")
    try:
        status = json.loads(blob.download_as_text())
    except Exception:
        print("   pipeline_status.json not found — stitch freshness unknown")
        return
    stitch = status.get('stitch', {})
    if stitch.get('status') != 'complete':
        print("   daily_stitch_master has not run — master_inference may be stale")
        return
    ts = pd.Timestamp(stitch.get('timestamp', '2000-01-01'))
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    today = pd.Timestamp.now(tz='UTC').normalize()
    if ts.normalize() < today:
        print(f"   Stitch last ran {ts.strftime('%Y-%m-%d %H:%M')} — not today")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    run_ts = datetime.now(pytz.utc)
    pst_str = run_ts.astimezone(PST).strftime('%Y-%m-%d %I:%M %p PST')
    print(f"\n{'='*80}")
    print(f"  SHORT SCORING — FOLD {SHORT_FOLD}   {pst_str}")
    print(f"{'='*80}")

    _check_stitch_freshness()

    now_pst = run_ts.astimezone(PST)
    now_minutes = now_pst.hour * 60 + now_pst.minute

    # ── BQ config selection ─────────────────────────────────────────────────
    n_configs = load_short_configs_from_bq()
    if n_configs == 0:
        print("  No SHORT configs selected from BQ — check wf_optimization_results")
        return

    print(f"\n  BQ configs loaded: {n_configs} tickers")
    for ticker in SHORT_TICKERS:
        cfg = SHORT_CONFIGS.get(ticker)
        if cfg:
            mode = _MODE_LABELS.get(cfg['tpsl_mode'], '??')
            print(f"    {ticker}: {mode} prob>{cfg['prob_threshold']:.2f} "
                  f"TP={cfg['tp_pct']*100:.0f}% val_a={cfg['val_a_avg_pnl']*100:+.2f}%")
        else:
            print(f"    {ticker}: NO CONFIG")

    # ── Load scaling stats ──────────────────────────────────────────────────
    stats_path = f"{INFERENCE_DIR}/scaling_stats_experiment_{SHORT_EXPERIMENT}.json"
    try:
        blob = gcs_client.bucket(BUCKET_NAME).blob(stats_path)
        stats = json.loads(blob.download_as_text())
        print(f"\n  Scaling stats: {len(stats)} features from {SHORT_EXPERIMENT}")
    except Exception as e:
        print(f"  ERROR loading scaling stats: {e}")
        return

    # ── Build market context ────────────────────────────────────────────────
    print(f"  Building market context...")
    market_df = build_market_context()
    print(f"  Market context: {len(market_df):,} bars")

    # ── Pre-load models ─────────────────────────────────────────────────────
    print(f"  Loading {len(SHORT_SEED_LIST)} models for {SHORT_EXPERIMENT}...")
    models_loaded = 0
    for seed in SHORT_SEED_LIST:
        try:
            get_model(seed)
            models_loaded += 1
        except Exception as e:
            print(f"    Seed {seed}: FAILED — {e}")
    print(f"  Models loaded: {models_loaded}/{len(SHORT_SEED_LIST)}")

    if models_loaded == 0:
        print("  ERROR: No models loaded — cannot score")
        return

    # ── Score each ticker ───────────────────────────────────────────────────
    print(f"\n  Scoring {len(SHORT_TICKERS)} tickers...")
    results = []
    for ticker in SHORT_TICKERS:
        cfg = SHORT_CONFIGS.get(ticker)
        if cfg is None:
            print(f"      {ticker}: SKIP — no BQ config")
            continue

        if now_minutes < cfg['entry_minute']:
            print(f"      {ticker}: SKIP — entry_minute {cfg['entry_minute']} > now {now_minutes}")
            continue

        df_raw = engineer_asset(ticker, market_df)
        if df_raw is None:
            print(f"      {ticker}: SKIP — engineer_asset returned None")
            continue

        df_scaled = apply_scaling(df_raw, stats)
        result = score_short_ticker(ticker, cfg, df_scaled, df_raw)
        if result is not None:
            results.append(result)

    print(f"  Scored: {len(results)}/{len(SHORT_TICKERS)}")

    # ── Dashboard + BQ log ──────────────────────────────────────────────────
    print_short_dashboard(results, run_ts)
    log_to_bigquery(results, run_ts)

    _master_cache.clear()
    _model_cache.clear()
    tf.keras.backend.clear_session()
    gc.collect()


main()
