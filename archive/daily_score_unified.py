"""
Unified Daily Scoring — Exp-97 (Crypto) + Exp-103 (Defense)
============================================================
Walk-forward validated configurations only.
OPT: Oct-Dec 2025, VAL: Jan-Mar 2026.

Deploys 6 tickers:
  Exp-97:  RIOT, WULF, IREN
  Exp-103: RCAT, JOBY, SPIR

Feature engineering, scaling, and model architecture are identical
to backtesting_97.py / backtesting_103.py and training scripts.

Run AFTER daily_data_all.py has finished updating GCS.
Usage:  python daily_score_unified.py
"""

import os
import sys
import json
import subprocess

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
    from google.cloud import bigquery
except Exception:
    bigquery = None

# ==============================================================================
# SHARED CONSTANTS
# ==============================================================================
BUCKET_NAME       = os.environ.get("GCS_BUCKET", "")
GCS_INFERENCE_DIR = f"gs://{BUCKET_NAME}/Data/Inference"
GCS_MODEL_PATH    = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
BQ_TABLE          = f"{GCP_PROJECT}.scoring_logs.live_predictions"

TAIL_DAYS  = 7
TIME_STEPS = 240
EMBED_DIM  = 16

ROC_HORIZONS = [5, 30, 60, 120]

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

PST = pytz.timezone('America/Los_Angeles')

# ==============================================================================
# EXP-97 CONFIGURATION (Crypto miners)
# ==============================================================================
EXP97_TRAIN_TEACHERS = [
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR', 'COIN',
    'BITF', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'SMLR',
]
EXP97_ASSET_INDEX = {t: i for i, t in enumerate(EXP97_TRAIN_TEACHERS)}
EXP97_ASSET_INDEX['KEEL'] = EXP97_ASSET_INDEX['BITF']
EXP97_N_ASSETS  = len(EXP97_TRAIN_TEACHERS) + 1   # 17
EXP97_OTHER_IDX = len(EXP97_TRAIN_TEACHERS)        # 16

EXP97_CONTEXT_ASSETS = ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD']

EXP97_CONTEXT_COLS = []
for _t in EXP97_CONTEXT_ASSETS:
    EXP97_CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        EXP97_CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
EXP97_CONTEXT_COLS += ['SMH_RSI', 'Time_Pos']

# Walk-forward validated configs (best seed per ticker from WF summary)
EXP97_ASSET_CONFIGS = {
    'RIOT': {'seed': 9707, 'prob_threshold': 0.510, 'tp_pct': 0.10, 'sl_pct': 0.06,
             'entry_minute': 370, 'vix_threshold': 15.0, 'tlt_crash': -0.30, 'spy_ignore': 0.20,
             'stats': {'win_rate': '100.0%', 'avg_pnl': '4.03%', 'freq': '1.0/mo'}},
    'WULF': {'seed': 9702, 'prob_threshold': 0.510, 'tp_pct': 0.17, 'sl_pct': 0.04,
             'entry_minute': 360, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.22,
             'stats': {'win_rate': '100.0%', 'avg_pnl': '2.62%', 'freq': '0.3/mo'}},
    'IREN': {'seed': 9706, 'prob_threshold': 0.510, 'tp_pct': 0.10, 'sl_pct': 0.10,
             'entry_minute': 375, 'vix_threshold': 15.0, 'tlt_crash': 0.00, 'spy_ignore': 0.10,
             'stats': {'win_rate': '75.0%', 'avg_pnl': '1.89%', 'freq': '1.3/mo'}},
}

# ==============================================================================
# EXP-103 CONFIGURATION (Defense sector)
# ==============================================================================
EXP103_TRAIN_TEACHERS = [
    'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS', 'LUNR', 'SPIR',
    'JOBY', 'NNDM', 'MVIS',
]
EXP103_ASSET_INDEX = {t: i for i, t in enumerate(EXP103_TRAIN_TEACHERS)}
EXP103_N_ASSETS  = len(EXP103_TRAIN_TEACHERS) + 1   # 12
EXP103_OTHER_IDX = len(EXP103_TRAIN_TEACHERS)        # 11

EXP103_CONTEXT_ASSETS = ['ITA', 'VIX', 'TLT', 'IWM']

EXP103_CONTEXT_COLS = []
for _t in EXP103_CONTEXT_ASSETS:
    EXP103_CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        EXP103_CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
EXP103_CONTEXT_COLS += ['Time_Pos']

# Walk-forward validated configs
EXP103_ASSET_CONFIGS = {
    'RCAT': {'seed': 10309, 'prob_threshold': 0.435, 'tp_pct': 0.15, 'sl_pct': 0.14,
             'entry_minute': 370, 'vix_threshold': 15.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
             'stats': {'win_rate': '75.0%', 'avg_pnl': '4.93%', 'freq': '4.0/mo'}},
    'JOBY': {'seed': 10305, 'prob_threshold': 0.410, 'tp_pct': 0.12, 'sl_pct': 0.08,
             'entry_minute': 370, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
             'stats': {'win_rate': '50.0%', 'avg_pnl': '3.42%', 'freq': '1.3/mo'}},
    'SPIR': {'seed': 10315, 'prob_threshold': 0.440, 'tp_pct': 0.10, 'sl_pct': 0.04,
             'entry_minute': 390, 'vix_threshold': 15.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
             'stats': {'win_rate': '66.7%', 'avg_pnl': '1.47%', 'freq': '3.0/mo'}},
}

# ==============================================================================
# EXPERIMENT REGISTRY
# ==============================================================================
EXPERIMENTS = [
    {
        'model_id':       '97_MICRO',
        'label':          'EXP-97 CRYPTO',
        'context_assets': EXP97_CONTEXT_ASSETS,
        'context_cols':   EXP97_CONTEXT_COLS,
        'has_smh_rsi':    True,
        'asset_index':    EXP97_ASSET_INDEX,
        'n_assets':       EXP97_N_ASSETS,
        'other_idx':      EXP97_OTHER_IDX,
        'asset_configs':  EXP97_ASSET_CONFIGS,
        'stats_path':     'Data/Inference_v2/scaling_stats_experiment_97_MICRO.json',
    },
    {
        'model_id':       '103_DEFENSE',
        'label':          'EXP-103 DEFENSE',
        'context_assets': EXP103_CONTEXT_ASSETS,
        'context_cols':   EXP103_CONTEXT_COLS,
        'has_smh_rsi':    False,
        'asset_index':    EXP103_ASSET_INDEX,
        'n_assets':       EXP103_N_ASSETS,
        'other_idx':      EXP103_OTHER_IDX,
        'asset_configs':  EXP103_ASSET_CONFIGS,
        'stats_path':     'Data/Inference_v2/scaling_stats_experiment_103_DEFENSE.json',
    },
]

# ==============================================================================
# GCS HELPERS
# ==============================================================================
gcs_client = gcs_storage.Client()


def _load_master(ticker, context_assets):
    clean = ticker.replace("I:", "").replace("X:", "")
    uri   = f"{GCS_INFERENCE_DIR}/{clean}_master_inference.parquet"
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
        if ticker in context_assets or ticker == 'SPY':
            for c in MICRO_COLS:
                if c not in df.columns:
                    df[c] = 0.0
        df = df[df.index.notna()].sort_index()

        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=TAIL_DAYS)
        df = df[df.index >= cutoff]
        if df.empty:
            print(f"      {ticker}: no recent data — skipping")
            return None
        return df
    except Exception as e:
        print(f"   Failed to load {ticker}: {e}")
        return None


def _heal(df):
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
def build_market_context(context_assets, has_smh_rsi):
    print("   Building market context...")
    frames = []
    for ticker in context_assets:
        df = _load_master(ticker, context_assets)
        if df is None:
            print(f"      Could not load context asset {ticker}")
            continue
        df = _heal(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        if has_smh_rsi and ticker == 'SMH':
            if len(df) >= 14:
                rsi = ta.rsi(df['Close'], length=14)
                cols['SMH_RSI'] = (rsi.fillna(50) / 100.0
                                   if rsi is not None
                                   else pd.Series(0.5, index=df.index))
            else:
                cols['SMH_RSI'] = pd.Series(0.5, index=df.index)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df

    if not frames:
        raise RuntimeError("No context assets loaded.")
    market_df = pd.concat(frames, axis=1).ffill().fillna(0)

    spy_df = _load_master('SPY', context_assets)
    if spy_df is not None:
        spy_df = _heal(spy_df)

    return market_df, spy_df


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def engineer_asset(ticker, market_df, context_assets):
    df = _load_master(ticker, context_assets)
    if df is None:
        return None
    df = _heal(df)

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

    df = df.join(market_df, how='left').ffill().fillna(0)

    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ==============================================================================
# SCALING
# ==============================================================================
def apply_scaling(df, stats, context_cols):
    df_s = df.copy()
    for col in ASSET_COLS + context_cols:
        if col == 'Time_Pos':
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# MODEL ARCHITECTURE (matches training exactly)
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


def build_model(n_context, n_assets):
    in_a  = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c  = Input(shape=(n_context,),                  name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,          name='asset_id')

    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))

    y = Dense(128, activation='relu')(BatchNormalization()(in_c))

    e = Embedding(n_assets, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)

    z   = Concatenate()([x, Dropout(0.5)(y), e])
    z   = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


# ==============================================================================
# MODEL LOADER
# ==============================================================================
_model_cache = {}


def get_model(seed, model_id, n_context, n_assets):
    cache_key = (seed, model_id)
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    print(f"      Loading model seed {seed} ({model_id})...")
    blob_path  = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{model_id}.h5"
    gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
    model = build_model(n_context, n_assets)
    model.load_weights(local_path)
    _model_cache[cache_key] = model
    return model


# ==============================================================================
# SCORING ENGINE
# ==============================================================================
def score_ticker(ticker, cfg, df_scaled, exp):
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

    context_cols = exp['context_cols']
    asset_index  = exp['asset_index']
    other_idx    = exp['other_idx']

    XA = df_scaled[ASSET_COLS].values
    XC = df_scaled[context_cols].values

    XA_b  = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]
    XC_b  = XC[row_idx - 1][np.newaxis]
    XID_b = np.array([[asset_index.get(ticker, other_idx)]], dtype=np.int32)

    model = get_model(cfg['seed'], exp['model_id'],
                      len(context_cols), exp['n_assets'])
    prob = float(model(
        {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
        training=False
    ).numpy().flatten()[0])
    signal = prob >= cfg['prob_threshold']

    bar_ts = df_scaled.index[row_idx]
    close  = (float(df_scaled['Close'].iloc[row_idx])
              if 'Close' in df_scaled.columns else float('nan'))

    return {
        'ticker':       ticker,
        'experiment':   exp['model_id'],
        'seed':         cfg['seed'],
        'prob':         round(prob, 4),
        'threshold':    cfg['prob_threshold'],
        'signal':       signal,
        'tp_pct':       cfg['tp_pct'],
        'sl_pct':       cfg['sl_pct'],
        'entry_minute': cfg['entry_minute'],
        'score_date':   score_date,
        'bar_ts':       bar_ts,
        'close':        round(close, 4),
    }


# ==============================================================================
# REGIME FILTER
# ==============================================================================
def load_regime_data(score_date, market_df, spy_df, run_ts, context_assets):
    window_end   = run_ts.astimezone(PST)
    window_start = window_end - timedelta(minutes=30)

    vix_level = None
    df_vix = _load_master('VIX', context_assets)
    if df_vix is not None:
        df_vix = _heal(df_vix)
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
    if spy_df is not None:
        dpst = spy_df.copy()
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
def _get_config(ticker, experiment):
    for exp in EXPERIMENTS:
        if exp['model_id'] == experiment and ticker in exp['asset_configs']:
            return exp['asset_configs'][ticker]
    return None


def print_dashboard(results, run_ts, regimes):
    W       = 155
    pst_str = run_ts.astimezone(PST).strftime('%Y-%m-%d %H:%M PST')
    print()
    print("=" * W)
    print(f"  UNIFIED DAILY SIGNALS   run: {pst_str}")
    print(f"  Walk-forward validated (OPT Oct-Dec 2025, VAL Jan-Mar 2026)")
    for model_id, regime in regimes.items():
        vix_s = f"{regime['vix']:.1f}"       if regime['vix']     is not None else "N/A"
        tlt_s = f"{regime['tlt_gap']:+.2f}%" if regime['tlt_gap'] is not None else "N/A"
        spy_s = f"{regime['spy_gap']:+.2f}%" if regime['spy_gap'] is not None else "N/A"
        print(f"  [{model_id}]  VIX={vix_s}  TLT={tlt_s}  SPY={spy_s}")
    print("=" * W)
    print(f"  {'Ticker':<6}  {'Exp':<13}  {'Seed':>5}  {'SIG':>6}  {'Prob':>6}  {'Thr':>5}  "
          f"{'Entry':>7}  {'TP tgt':>13}  {'SL stop':>13}  {'Close':>8}  "
          f"{'Score bar':<16}  {'VIX<':>5}  {'TLT>':>6}  {'SPY<':>6}  "
          f"{'WR':>6}  {'AvgPnL':>8}  {'Freq':>8}  FILT")
    print("-" * W)

    for r in results:
        cfg    = _get_config(r['ticker'], r['experiment'])
        regime = regimes.get(r['experiment'])

        trade_ok = True
        if regime is not None and cfg is not None:
            trade_ok, _, _, _ = _filter_pass(cfg, regime)

        if r['signal']:
            sig_str = ("\033[92mGO\033[0m     " if trade_ok
                       else "\033[31mBLOCK\033[0m  ")
        else:
            sig_str = "--     "

        stats    = cfg.get('stats', {}) if cfg else {}
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
            f"  {r['ticker']:<6}  {r['experiment']:<13}  {r['seed']:>5}  "
            f"{sig_str}  {prob_str:>6}  {r['threshold']:>5.3f}  "
            f"{entry_str:>7}  {tp_str:>13}  {sl_str:>13}  {close:>8.2f}  "
            f"{bar_str:<16}  {vix_t:>5}  {tlt_t:>6}  {spy_t:>6}  "
            f"{win_str:>6}  {pnl_str:>8}  {freq_str:>8}{filt_str}"
        )

    print("=" * W)

    signals = [r for r in results if r['signal']]
    go_trades, blocked = [], []
    for r in signals:
        cfg    = _get_config(r['ticker'], r['experiment'])
        regime = regimes.get(r['experiment'])
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
# MAIN PIPELINE
# ==============================================================================
def main():
    run_ts = datetime.now(pytz.utc)
    print(f"Unified Daily Scoring Pipeline")
    print(f"   Run: {run_ts.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"   Tickers: {sum(len(e['asset_configs']) for e in EXPERIMENTS)}")
    print(f"   TIME_STEPS={TIME_STEPS}  ROC_HORIZONS={ROC_HORIZONS}")

    all_results = []
    all_regimes = {}

    for exp in EXPERIMENTS:
        model_id = exp['model_id']
        label    = exp['label']
        print(f"\n{'='*80}")
        print(f"  {label}  ({len(exp['asset_configs'])} tickers)")
        print(f"{'='*80}")

        # 1. Load scaling stats
        print(f"\n   Loading scaling stats ({model_id})...")
        try:
            blob  = gcs_client.bucket(BUCKET_NAME).blob(exp['stats_path'])
            stats = json.loads(blob.download_as_text())
            print(f"   Loaded {len(stats)} features")
        except Exception as e:
            print(f"   Failed to load stats: {e} — skipping experiment")
            continue

        # 2. Build market context + SPY
        market_df, spy_df = build_market_context(
            exp['context_assets'], exp['has_smh_rsi'])

        # 3. Pre-load models
        print(f"   Pre-loading models...")
        needed_seeds = sorted(set(cfg['seed']
                                  for cfg in exp['asset_configs'].values()))
        for seed in needed_seeds:
            try:
                get_model(seed, model_id,
                          len(exp['context_cols']), exp['n_assets'])
            except Exception as e:
                print(f"      Failed to load seed {seed}: {e}")
        print(f"   Models ready: {needed_seeds}")

        # 4. Score each ticker
        print(f"   Scoring...")
        exp_results = []
        for ticker, cfg in exp['asset_configs'].items():
            asset_id = exp['asset_index'].get(ticker, exp['other_idx'])
            print(f"   {ticker}  seed={cfg['seed']}  "
                  f"prob>{cfg['prob_threshold']}  id={asset_id}")

            df_raw = engineer_asset(ticker, market_df, exp['context_assets'])
            if df_raw is None:
                print(f"      Skipping {ticker}")
                continue
            df_scaled = apply_scaling(df_raw, stats, exp['context_cols'])
            result = score_ticker(ticker, cfg, df_scaled, exp)
            if result is not None:
                exp_results.append(result)

        # 5. Compute regime filter
        if exp_results:
            score_date = exp_results[0]['score_date']
            print(f"\n   Computing regime for {score_date}...")
            regime = load_regime_data(
                score_date, market_df, spy_df, run_ts, exp['context_assets'])
            vix_s = f"{regime['vix']:.1f}"       if regime['vix']     is not None else "N/A"
            tlt_s = f"{regime['tlt_gap']:+.2f}%" if regime['tlt_gap'] is not None else "N/A"
            spy_s = f"{regime['spy_gap']:+.2f}%" if regime['spy_gap'] is not None else "N/A"
            print(f"   VIX={vix_s}  TLT={tlt_s}  SPY={spy_s}")
            all_regimes[model_id] = regime

        all_results.extend(exp_results)

    # 6. Combined dashboard
    print_dashboard(all_results, run_ts, all_regimes)

    # 7. Log to BigQuery
    print("\nLogging to BigQuery...")
    log_to_bigquery(all_results, run_ts)

    print("\nDone.")


if __name__ == "__main__":
    main()
