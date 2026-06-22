"""
backtest_portfolio_v3.py — v12 vs v11 Head-to-Head Backtest
============================================================
Loads deploy configs from BQ for BOTH model versions, runs live TF model
inference over the TEST period (Apr-Jun 2026), applies dynamic TP/SL +
regime + volume filters, simulates trades, then compares v12 vs v11 results.

v12 additions over v2:
  - atr_sl mode: fixed TP + ATR-based SL
  - grace mode: SL disabled for first N minutes
  - limit_entry mode: virtual limit buy at dip below entry

Colab-ready (no if __name__). Requires: TF, pandas_ta, numba, GCS, BQ.
"""

import os
import io
import gc
import sys
import json
import subprocess
import warnings
from datetime import datetime, timedelta

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

try:
    import tf_keras
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tf_keras", "--quiet"])

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]", "tf_keras", "pandas_ta",
                           "google-cloud-storage", "pyarrow", "gcsfs",
                           "numba", "--quiet"])
    import tensorflow as tf

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

from numba import jit
from google.cloud import storage as gcs_storage, bigquery

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from tf_keras.models import Model
from tf_keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    BatchNormalization, Concatenate, Embedding, Flatten,
)
from tf_keras.regularizers import l2

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_VERSION_V12 = 'robust_wf_v12'
MODEL_VERSION_V11 = 'robust_wf_v11'
MIN_TRADES = 3
DEPLOY_THRESHOLD = 0.018

TEST_START = '2026-04-01'
TEST_END   = '2026-06-13'

USE_CSV_SCORES = True

# --- Composite selection method ---
# 'blended'        = current wavg_pnl * wavg_wr * log2(1+N) * consistency (baseline)
# 'seed_agreement' = cross-seed agreement: pick configs where most seeds show val_a >= threshold
# 'opt_pnl_x_wr'   = simplest opt-based: opt_avg_pnl * opt_win_rate
COMPOSITE_METHOD = 'seed_agreement'

STARTING_CAPITAL = 200_000
MIN_SIGNALS_PER_DAY = 2

DEPLOY_EXPERIMENTS = {'152_XSECTOR', '160_XSECTOR', '170_XSECTOR',
                      '175_XSECTOR', '180_XSECTOR', '185_XSECTOR'}
EXCLUDE_TICKERS = set()

BUCKET_NAME    = "time-series-bucket-ys"
BQ_TABLE       = "ys-time-series.scoring_logs.wf_optimization_results"
INFERENCE_DIR  = "Data/Inference_v2"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR      = "Data/Inference_v2"

TIME_STEPS  = 240
EMBED_DIM   = 16
EXIT_TIME_MIN = 780
PST = pytz.timezone('America/Los_Angeles')
W = 130

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
SLIPPAGE_PCT = 0.0015
VOL_LOOKBACK_DAYS = 30

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

REGIME_TP_SCALE = np.array([0.70, 0.85, 1.00, 1.30, 1.60])
REGIME_SL_SCALE = np.array([0.60, 0.80, 1.00, 1.30, 1.80])

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
               'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
               'grace': '_GRACE', 'limit_entry': '_LIMIT'}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()


# ==============================================================================
# LOAD DEPLOY CONFIGS FROM BQ
# ==============================================================================
def load_top_configs_from_bq(model_version, min_trades, label=''):
    print(f"  Loading WF results from BQ (model_version={model_version}) [{label}]...")
    query = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_time, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           trail_start_mult, trail_end_mult,
           vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
           val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
           val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
           opt_avg_pnl, opt_win_rate, opt_count,
           is_ensemble, survived_a, survived_b,
           grace_minutes, dip_pct, max_wait_minutes
    FROM `{BQ_TABLE}`
    WHERE model_version = '{model_version}'
      AND survived_a = TRUE
      AND val_a_count >= {min_trades}
      AND val_a_avg_pnl > 0
    """
    df = bq_client.query(query).to_dataframe()
    print(f"    {len(df)} rows loaded from BQ")

    if df.empty:
        print(f"    WARNING: No results found for model_version={model_version}")
        return []

    df['ticker'] = df['ticker'].replace(TICKER_RENAME)

    all_suffixes = list(MODE_SUFFIX.values())
    def _base_exp(exp_name):
        for sfx in sorted(all_suffixes, key=len, reverse=True):
            if sfx and exp_name.endswith(sfx):
                return exp_name[:-len(sfx)]
        return exp_name

    df['base_experiment'] = df['experiment'].apply(_base_exp)
    df = df[df['base_experiment'].isin(DEPLOY_EXPERIMENTS)]
    df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
    print(f"    {len(df)} rows after experiment/ticker filter")

    if df.empty:
        return []

    for col in ['grace_minutes', 'max_wait_minutes', 'vol_filter_pct']:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    for col in ['dip_pct', 'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # --- Overfit filter: reject configs where val_a spiked vs opt ---
    MAX_VA_OPT_RATIO = 3.0
    before_n = len(df)
    df = df[df['opt_avg_pnl'] > 0].copy()
    va_opt_ratio = df['val_a_avg_pnl'] / df['opt_avg_pnl']
    overfit_mask = va_opt_ratio > MAX_VA_OPT_RATIO
    n_overfit = overfit_mask.sum()
    df = df[~overfit_mask]
    print(f"    Overfit filter: {before_n} → {len(df)} configs (dropped {before_n - len(df)}, "
          f"of which {n_overfit} had val_a/opt > {MAX_VA_OPT_RATIO}x)")

    if df.empty:
        return []

    # --- Composite scoring (method selected by COMPOSITE_METHOD config) ---
    print(f"    Composite method: {COMPOSITE_METHOD}")

    if COMPOSITE_METHOD == 'seed_agreement':
        # Group by config params across seeds, measure cross-seed agreement
        # Filter to individual seed rows only (exclude ensemble aggregates)
        df_indiv = df[df['is_ensemble'] == False].copy()
        grp_keys = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
                     'prob_thresh', 'tp_pct', 'sl_pct',
                     'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
                     'grace_minutes', 'dip_pct', 'max_wait_minutes',
                     'vol_filter_pct']
        extra_agg = {}
        for c in ['vix_threshold','tlt_crash','spy_ignore']:
            if c in df_indiv.columns:
                extra_agg[c] = (c, 'first')
        sa = df_indiv.groupby(grp_keys).agg(
            n_seeds=('seed', 'nunique'),
            n_pos_seeds=('val_a_avg_pnl', lambda x: (x >= DEPLOY_THRESHOLD).sum()),
            mean_va_pnl=('val_a_avg_pnl', 'mean'),
            mean_opt_pnl=('opt_avg_pnl', 'mean'),
            mean_va_wr=('val_a_win_rate', 'mean'),
            mean_va_count=('val_a_count', 'mean'),
            mean_va_freq=('val_a_freq', 'mean'),
            mean_opt_wr=('opt_win_rate', 'mean'),
            **extra_agg,
        ).reset_index()
        sa['agree_pct'] = (sa['n_pos_seeds'] / sa['n_seeds']).clip(upper=1.0)
        sa = sa[sa['n_seeds'] >= 3]
        sa['composite'] = sa['mean_va_pnl'] * sa['mean_va_wr'] * sa['agree_pct']
        best = sa.loc[sa.groupby('ticker')['composite'].idxmax()]
        best = best.sort_values('composite', ascending=False)
        print(f"    Seed agreement: {len(sa)} config groups → {len(best)} tickers")

        configs = []
        for _, row in best.iterrows():
            exp = row['experiment']
            mode = row['tpsl_mode']
            mode_sfx = MODE_SUFFIX.get(mode, '')
            base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp
            base_num = int(base_exp.split('_')[0])
            seed_list = list(range(base_num * 100 + 1, base_num * 100 + 16))
            configs.append({
                'ticker': row['ticker'],
                'base_experiment': base_exp,
                'experiment': exp,
                'tpsl_mode': mode,
                'seed': -1,
                'is_ensemble': True,
                'seed_list': seed_list,
                'entry_minute': int(row['entry_minute']),
                'prob_thresh': float(row['prob_thresh']),
                'tp_pct': float(row.get('tp_pct') or 0),
                'sl_pct': float(row.get('sl_pct') or 0),
                'tp_mult': float(row.get('tp_mult') or 3.0),
                'sl_mult': float(row.get('sl_mult') or 1.5),
                'trail_start_mult': float(row.get('trail_start_mult') or 4.0),
                'trail_end_mult': float(row.get('trail_end_mult') or 1.0),
                'vix_threshold': float(row.get('vix_threshold') or 99),
                'tlt_crash': float(row.get('tlt_crash') or -99),
                'spy_ignore': float(row.get('spy_ignore') or 99),
                'vol_filter_pct': int(row.get('vol_filter_pct') or 0),
                'grace_minutes': int(row.get('grace_minutes') or 0),
                'dip_pct': float(row.get('dip_pct') or 0),
                'max_wait_minutes': int(row.get('max_wait_minutes') or 0),
                'model_id': base_exp,
                'stats_id': base_exp,
                'wf_avg_pnl': float(row['mean_va_pnl']),
                'wf_win_rate': float(row['mean_va_wr']),
                'wf_count': int(row['mean_va_count']),
                'wf_freq': float(row.get('mean_va_freq') or 0),
                'wf_composite': float(row['composite']),
                'seed_agree_pct': float(row['agree_pct']),
                'n_seeds': int(row['n_seeds']),
            })

    elif COMPOSITE_METHOD == 'opt_pnl_x_wr':
        df['composite'] = df['opt_avg_pnl'] * df['opt_win_rate']
        best = df.loc[df.groupby('ticker')['composite'].idxmax()]
        best = best.sort_values('composite', ascending=False)

        configs = []
        for _, row in best.iterrows():
            exp = row['experiment']
            mode = row['tpsl_mode']
            mode_sfx = MODE_SUFFIX.get(mode, '')
            base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp
            is_ens = bool(row['is_ensemble'])
            if is_ens:
                base_num = int(base_exp.split('_')[0])
                seed_list = list(range(base_num * 100 + 1, base_num * 100 + 16))
            else:
                seed_list = [int(row['seed'])]
            configs.append({
                'ticker': row['ticker'],
                'base_experiment': base_exp,
                'experiment': exp,
                'tpsl_mode': mode,
                'seed': int(row['seed']),
                'is_ensemble': is_ens,
                'seed_list': seed_list,
                'entry_minute': int(row['entry_minute']),
                'prob_thresh': float(row['prob_thresh']),
                'tp_pct': float(row.get('tp_pct') or 0),
                'sl_pct': float(row.get('sl_pct') or 0),
                'tp_mult': float(row.get('tp_mult') or 3.0),
                'sl_mult': float(row.get('sl_mult') or 1.5),
                'trail_start_mult': float(row.get('trail_start_mult') or 4.0),
                'trail_end_mult': float(row.get('trail_end_mult') or 1.0),
                'vix_threshold': float(row.get('vix_threshold') or 99),
                'tlt_crash': float(row.get('tlt_crash') or -99),
                'spy_ignore': float(row.get('spy_ignore') or 99),
                'vol_filter_pct': int(row.get('vol_filter_pct') or 0),
                'grace_minutes': int(row.get('grace_minutes') or 0),
                'dip_pct': float(row.get('dip_pct') or 0),
                'max_wait_minutes': int(row.get('max_wait_minutes') or 0),
                'model_id': base_exp,
                'stats_id': base_exp,
                'wf_avg_pnl': float(row['val_a_avg_pnl']),
                'wf_win_rate': float(row['val_a_win_rate']),
                'wf_count': int(row['val_a_count']),
                'wf_freq': float(row.get('val_a_freq') or 0),
                'wf_composite': float(row['composite']),
            })

    else:  # 'blended' (default/baseline)
        total_n = df['opt_count'] + df['val_a_count']
        wavg_pnl = (df['opt_avg_pnl'] * df['opt_count'] + df['val_a_avg_pnl'] * df['val_a_count']) / total_n
        wavg_wr = (df['opt_win_rate'] * df['opt_count'] + df['val_a_win_rate'] * df['val_a_count']) / total_n
        consistency = (1.0 - (df['val_a_avg_pnl'] - df['opt_avg_pnl']).abs() / df['opt_avg_pnl'].clip(lower=0.001)).clip(lower=0.2)
        df['composite'] = wavg_pnl * wavg_wr * np.log2(1 + df['val_a_count']) * consistency
        best = df.loc[df.groupby('ticker')['composite'].idxmax()]
        best = best.sort_values('composite', ascending=False)

        configs = []
        for _, row in best.iterrows():
            exp = row['experiment']
            mode = row['tpsl_mode']
            mode_sfx = MODE_SUFFIX.get(mode, '')
            base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp
            is_ens = bool(row['is_ensemble'])
            if is_ens:
                base_num = int(base_exp.split('_')[0])
                seed_list = list(range(base_num * 100 + 1, base_num * 100 + 16))
            else:
                seed_list = [int(row['seed'])]
            configs.append({
                'ticker': row['ticker'],
                'base_experiment': base_exp,
                'experiment': exp,
                'tpsl_mode': mode,
                'seed': int(row['seed']),
                'is_ensemble': is_ens,
                'seed_list': seed_list,
                'entry_minute': int(row['entry_minute']),
                'prob_thresh': float(row['prob_thresh']),
                'tp_pct': float(row.get('tp_pct') or 0),
                'sl_pct': float(row.get('sl_pct') or 0),
                'tp_mult': float(row.get('tp_mult') or 3.0),
                'sl_mult': float(row.get('sl_mult') or 1.5),
                'trail_start_mult': float(row.get('trail_start_mult') or 4.0),
                'trail_end_mult': float(row.get('trail_end_mult') or 1.0),
                'vix_threshold': float(row.get('vix_threshold') or 99),
                'tlt_crash': float(row.get('tlt_crash') or -99),
                'spy_ignore': float(row.get('spy_ignore') or 99),
                'vol_filter_pct': int(row.get('vol_filter_pct') or 0),
                'grace_minutes': int(row.get('grace_minutes') or 0),
                'dip_pct': float(row.get('dip_pct') or 0),
                'max_wait_minutes': int(row.get('max_wait_minutes') or 0),
                'model_id': base_exp,
                'stats_id': base_exp,
                'wf_avg_pnl': float(row['val_a_avg_pnl']),
                'wf_win_rate': float(row['val_a_win_rate']),
                'wf_count': int(row['val_a_count']),
                'wf_freq': float(row.get('val_a_freq') or 0),
                'wf_composite': float(row['composite']),
            })

    print(f"    {len(configs)} tickers selected:")
    for c in configs:
        seed_str = 'ENS' if c['is_ensemble'] else str(c['seed'])
        extra = ''
        if 'seed_agree_pct' in c:
            extra = f" agree={c['seed_agree_pct']:.0%} ({c['n_seeds']}seeds)"
        print(f"      {c['ticker']:<7} {c['experiment']:<25} seed={seed_str:>5} "
              f"mode={c['tpsl_mode']:<14} WF: pnl={c['wf_avg_pnl']:.2%} wr={c['wf_win_rate']:.0%} N={c['wf_count']}{extra}")
    return configs


# ==============================================================================
# DATA LOADING — verbatim from backtest_portfolio_v2.py
# ==============================================================================
_master_cache = {}


def load_master(ticker, start_date=None):
    cache_key = (ticker, start_date)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
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
# FEATURE ENGINEERING — verbatim from v2
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
    if not frames:
        raise RuntimeError("No context assets loaded.")
    result = pd.concat(frames, axis=1, sort=True).ffill().fillna(0)
    _market_context_cache = result
    print(f"    Market context: {len(result)} rows, {result.index[0]} → {result.index[-1]}")
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
# SCALING — verbatim from v2
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
# MODEL — verbatim from v2
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
# ATR — verbatim from v2
# ==============================================================================
def compute_daily_atr_series(df_ohlcv, period=14):
    daily = df_ohlcv.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([
        daily['High'] - daily['Low'],
        (daily['High'] - prev_close).abs(),
        (daily['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    daily_atr = tr.rolling(period).mean()
    atr_pct = daily_atr / daily['Close']
    return atr_pct.reindex(df_ohlcv.index, method='ffill')


def compute_adaptive_atr_series(df_ohlcv, short=5, long=22, base=14):
    daily = df_ohlcv.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([
        daily['High'] - daily['Low'],
        (daily['High'] - prev_close).abs(),
        (daily['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_base  = tr.rolling(base).mean()
    atr_short = tr.rolling(short).mean()
    atr_long  = tr.rolling(long).mean()
    accel = (atr_short / atr_long).clip(0.5, 3.0)
    atr_adaptive = atr_base * accel
    atr_pct = atr_adaptive / daily['Close']
    return atr_pct.reindex(df_ohlcv.index, method='ffill')


# ==============================================================================
# VOLUME FILTER — verbatim from v2
# ==============================================================================
def precompute_vol_baselines(df_ohlcv, lookback_days=VOL_LOOKBACK_DAYS):
    times = (df_ohlcv.index.hour * 60 + df_ohlcv.index.minute).values
    dates = (df_ohlcv.index.year * 10000 + df_ohlcv.index.month * 100
             + df_ohlcv.index.day).values.astype(np.int64)
    volumes = df_ohlcv['Volume'].values

    baselines = {}
    for em in range(358, 393):
        at_em = np.where(times == em)[0]
        for j in range(len(at_em)):
            d_int = int(dates[at_em[j]])
            start_j = max(0, j - lookback_days)
            if j - start_j < 5:
                baselines[(d_int, em)] = 0.0
                continue
            prev_vols = volumes[at_em[start_j:j]]
            baselines[(d_int, em)] = float(np.median(prev_vols))
    return baselines


def check_volume_filter(vol_baselines, d_int, entry_minute, volume, vol_filter_pct):
    if vol_filter_pct <= 0:
        return True
    baseline = vol_baselines.get((d_int, entry_minute), 0.0)
    if baseline <= 0:
        return True
    pct = (volume / baseline) * 100
    return pct >= vol_filter_pct


# ==============================================================================
# BATCH SCORING — verbatim from v2
# ==============================================================================
def score_ticker_period(ticker, cfg, df_scaled, df_raw):
    entry_minute = cfg['entry_minute']
    model_id = cfg['model_id']
    seed = cfg['seed']
    prob_thresh = cfg['prob_thresh']
    asset_id = ASSET_INDEX.get(ticker, OTHER_IDX)

    test_start_utc = pd.Timestamp(TEST_START, tz='UTC')
    test_end_utc   = pd.Timestamp(TEST_END, tz='UTC') + pd.Timedelta(days=1)
    test_mask = (df_scaled.index >= test_start_utc) & (df_scaled.index < test_end_utc)
    df_test = df_scaled[test_mask]

    df_test_pst = df_test.index.tz_convert(PST)
    mins = df_test_pst.hour * 60 + df_test_pst.minute
    entry_mask = mins == entry_minute

    entry_positions = []
    for i in range(len(df_test)):
        if not entry_mask[i]:
            continue
        global_idx = df_scaled.index.get_loc(df_test.index[i])
        if global_idx < TIME_STEPS:
            continue
        if 'Volume' in df_scaled.columns and float(df_scaled['Volume'].iloc[global_idx]) == 0:
            continue
        entry_positions.append(global_idx)

    if not entry_positions:
        return []

    XA_batch = np.stack([
        df_scaled[ASSET_COLS].values[idx - TIME_STEPS:idx]
        for idx in entry_positions
    ])
    XC_batch = np.stack([
        df_scaled[CONTEXT_COLS].values[idx - 1]
        for idx in entry_positions
    ])
    XID_batch = np.full((len(entry_positions), 1), asset_id, dtype=np.int32)

    seed_list = cfg.get('seed_list', [seed])
    all_probs = []
    for s in seed_list:
        try:
            m = get_model(model_id, s)
            p = m.predict(
                {'asset_in': XA_batch, 'context_in': XC_batch, 'asset_id': XID_batch},
                batch_size=64, verbose=0
            ).flatten()
            all_probs.append(p)
        except Exception:
            continue
    if not all_probs:
        return []
    probs = np.mean(all_probs, axis=0)

    results = []
    for i, global_idx in enumerate(entry_positions):
        prob = float(probs[i])
        if prob < prob_thresh:
            continue
        bar_ts = df_scaled.index[global_idx]
        raw_candidates = df_raw.index[df_raw.index <= bar_ts]
        raw_idx = df_raw.index.get_loc(raw_candidates[-1]) if len(raw_candidates) > 0 else None
        results.append({
            'bar_ts': bar_ts,
            'prob': prob,
            'raw_idx': raw_idx,
            'scaled_idx': global_idx,
        })

    return results


# ==============================================================================
# CSV-BASED SCORING — uses same pre-computed scores as the optimizer
# ==============================================================================
_csv_score_cache = {}

def load_seed_scores(ticker, experiment):
    cache_key = (ticker, experiment)
    if cache_key in _csv_score_cache:
        return _csv_score_cache[cache_key]

    clean = TICKER_FILE_MAP.get(ticker, ticker)
    results_dir = f"Algos/Results_all_v2/Universal_Exp{experiment}"
    prefix = f"{results_dir}/{clean}/Seed_"
    blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=prefix))

    seed_series = {}
    for blob in blobs:
        name = blob.name.split('/')[-1]
        if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
            continue
        seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
        df = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(PST)
        seed_series[seed_id] = df['Prob_Spike'].sort_index()

    _csv_score_cache[cache_key] = seed_series
    return seed_series


def score_ticker_from_csv(ticker, cfg, df_ohlcv_pst):
    experiment = cfg['model_id']
    entry_minute = cfg['entry_minute']
    prob_thresh = cfg['prob_thresh']
    is_ensemble = cfg.get('is_ensemble', False)
    seed = cfg['seed']

    all_seeds = load_seed_scores(ticker, experiment)
    if not all_seeds:
        return []

    if is_ensemble:
        seed_list = sorted(all_seeds.keys())
    else:
        seed_list = [seed] if seed in all_seeds else [s for s in all_seeds if s == seed]
        if not seed_list and seed == -1:
            seed_list = sorted(all_seeds.keys())

    idx = df_ohlcv_pst.index
    probs = np.zeros(len(idx), dtype=np.float64)
    n_seeds_used = 0

    for s in seed_list:
        if s not in all_seeds:
            continue
        seed_prob = all_seeds[s]
        common = idx.intersection(seed_prob.index)
        if len(common) == 0:
            continue
        common_locs = idx.get_indexer(common)
        probs[common_locs] += seed_prob.loc[common].values
        n_seeds_used += 1

    if n_seeds_used == 0:
        return []
    probs /= n_seeds_used

    times = (df_ohlcv_pst.index.hour * 60 + df_ohlcv_pst.index.minute).values

    test_start_pst = pd.Timestamp(TEST_START, tz=PST)
    test_end_pst = pd.Timestamp(TEST_END, tz=PST) + pd.Timedelta(days=1)

    results = []
    for i in range(len(idx)):
        if idx[i] < test_start_pst or idx[i] >= test_end_pst:
            continue
        if times[i] != entry_minute:
            continue
        if probs[i] < prob_thresh:
            continue
        results.append({
            'bar_ts': idx[i],
            'prob': float(probs[i]),
            'raw_idx': None,
            'scaled_idx': None,
        })

    return results


# ==============================================================================
# DYNAMIC TP/SL — extended for v12 modes
# ==============================================================================
def compute_dynamic_tpsl(cfg, atr_pct_val, vix_value):
    mode = cfg['tpsl_mode']

    if mode in ('atr', 'atr_adaptive'):
        tp = min(max(atr_pct_val * cfg['tp_mult'], 0.02), 0.60)
        sl = min(max(atr_pct_val * cfg['sl_mult'], 0.01), 0.25)
        return tp, sl

    elif mode == 'regime_scaled':
        regime_idx = 2
        if vix_value is not None:
            if vix_value <= 15: regime_idx = 0
            elif vix_value <= 20: regime_idx = 1
            elif vix_value <= 25: regime_idx = 2
            elif vix_value <= 35: regime_idx = 3
            else: regime_idx = 4
        tp = cfg['tp_pct'] * REGIME_TP_SCALE[regime_idx]
        sl = cfg['sl_pct'] * REGIME_SL_SCALE[regime_idx]
        return tp, sl

    elif mode == 'trailing_decay':
        sl = min(max(atr_pct_val * cfg['sl_mult'], 0.01), 0.25)
        tp = min(max(atr_pct_val * cfg['trail_start_mult'], 0.02), 0.60)
        return tp, sl

    elif mode == 'atr_sl':
        tp = cfg['tp_pct']
        sl = min(max(atr_pct_val * cfg['sl_mult'], 0.01), 0.25)
        return tp, sl

    elif mode in ('grace', 'limit_entry'):
        return cfg['tp_pct'], cfg['sl_pct']

    return cfg['tp_pct'], cfg['sl_pct']


# ==============================================================================
# TRADE SIMULATION — existing from v2
# ==============================================================================
@jit(nopython=True)
def run_single_trade(prices, highs, lows, date_ints, time_mins,
                     entry_idx, tp, sl, exit_time):
    n_prices = len(prices)
    if entry_idx >= n_prices - 1:
        return 0.0, date_ints[entry_idx], False

    entry_price  = prices[entry_idx]
    entry_date   = date_ints[entry_idx]
    target_price = entry_price * (1 + tp)
    stop_price   = entry_price * (1 - sl)

    outcome = 0.0

    for k in range(entry_idx + 1, n_prices):
        if date_ints[k] != entry_date:
            outcome = (prices[k-1] - entry_price) / entry_price
            break
        sl_hit = lows[k] <= stop_price
        tp_hit = highs[k] >= target_price
        if sl_hit and tp_hit:
            outcome = -sl if np.random.random() < 0.5 else tp
            break
        if sl_hit:
            outcome = -sl; break
        if tp_hit:
            outcome = tp; break
        if time_mins[k] >= exit_time:
            outcome = (prices[k] - entry_price) / entry_price
            break
    else:
        outcome = (prices[n_prices-1] - entry_price) / entry_price

    return outcome, entry_date, True


@jit(nopython=True)
def run_single_trade_trailing_decay(prices, highs, lows, date_ints, time_mins,
                                     entry_idx, atr_pct, trail_start_mult,
                                     trail_end_mult, sl_mult, exit_time):
    n_prices = len(prices)
    if entry_idx >= n_prices - 1:
        return 0.0, date_ints[entry_idx], False

    entry_price = prices[entry_idx]
    entry_date  = date_ints[entry_idx]

    if atr_pct != atr_pct or atr_pct <= 0:
        atr_pct = 0.05

    sl = min(max(atr_pct * sl_mult, 0.01), 0.25)
    hard_stop = entry_price * (1 - sl)

    trail_start  = atr_pct * trail_start_mult
    trail_end    = atr_pct * trail_end_mult
    decay_target = trail_start

    high_watermark = entry_price

    outcome = 0.0

    for k in range(entry_idx + 1, n_prices):
        if date_ints[k] != entry_date:
            outcome = (prices[k-1] - entry_price) / entry_price
            break

        if highs[k] > high_watermark:
            high_watermark = highs[k]

        unrealized = (high_watermark - entry_price) / entry_price
        trailing_stop = 0.0
        if unrealized > 0.0 and decay_target > 0.0:
            ratio = unrealized / decay_target
            if ratio > 1.0:
                ratio = 1.0
            cur_trail = trail_start + (trail_end - trail_start) * ratio
            trailing_stop = high_watermark * (1.0 - cur_trail)

        effective_stop = hard_stop
        if trailing_stop > hard_stop:
            effective_stop = trailing_stop

        if lows[k] <= effective_stop:
            if trailing_stop > hard_stop:
                outcome = (trailing_stop - entry_price) / entry_price
            else:
                outcome = -sl
            break

        if time_mins[k] >= exit_time:
            outcome = (prices[k] - entry_price) / entry_price
            break
    else:
        outcome = (prices[n_prices-1] - entry_price) / entry_price

    return outcome, entry_date, True


# ==============================================================================
# v12 NEW TRADE SIMS
# ==============================================================================
@jit(nopython=True)
def run_single_trade_grace(prices, highs, lows, date_ints, time_mins,
                           entry_idx, tp, sl, exit_time, grace_minutes):
    n_prices = len(prices)
    if entry_idx >= n_prices - 1:
        return 0.0, date_ints[entry_idx], False

    entry_price = prices[entry_idx]
    entry_date  = date_ints[entry_idx]
    entry_time  = time_mins[entry_idx]
    target_price = entry_price * (1 + tp)
    stop_price   = entry_price * (1 - sl)

    outcome = 0.0

    for k in range(entry_idx + 1, n_prices):
        if date_ints[k] != entry_date:
            outcome = (prices[k-1] - entry_price) / entry_price
            break

        minutes_elapsed = time_mins[k] - entry_time
        sl_active = minutes_elapsed >= grace_minutes

        tp_hit = highs[k] >= target_price
        sl_hit = lows[k] <= stop_price

        if sl_active and sl_hit and tp_hit:
            outcome = -sl if np.random.random() < 0.5 else tp
            break
        if sl_active and sl_hit:
            outcome = -sl
            break
        if tp_hit:
            outcome = tp
            break
        if time_mins[k] >= exit_time:
            outcome = (prices[k] - entry_price) / entry_price
            break
    else:
        outcome = (prices[n_prices-1] - entry_price) / entry_price

    return outcome, entry_date, True


@jit(nopython=True)
def run_single_trade_limit_entry(prices, highs, lows, date_ints, time_mins,
                                  entry_idx, tp, sl, exit_time,
                                  dip_pct, max_wait_minutes):
    n_prices = len(prices)
    if entry_idx >= n_prices - 1:
        return 0.0, date_ints[entry_idx], False

    signal_price = prices[entry_idx]
    entry_date   = date_ints[entry_idx]
    entry_time   = time_mins[entry_idx]
    limit_price  = signal_price * (1.0 - dip_pct)

    fill_idx = -1
    for k in range(entry_idx + 1, n_prices):
        if date_ints[k] != entry_date:
            break
        minutes_elapsed = time_mins[k] - entry_time
        if minutes_elapsed > max_wait_minutes:
            break
        if lows[k] <= limit_price:
            fill_idx = k
            break

    if fill_idx < 0:
        return 0.0, entry_date, False

    entry_price  = limit_price
    target_price = entry_price * (1.0 + tp)
    stop_price   = entry_price * (1.0 - sl)
    outcome = 0.0

    for k in range(fill_idx + 1, n_prices):
        if date_ints[k] != entry_date:
            outcome = (prices[k-1] - entry_price) / entry_price
            break
        sl_hit = lows[k] <= stop_price
        tp_hit = highs[k] >= target_price
        if sl_hit and tp_hit:
            outcome = -sl if np.random.random() < 0.5 else tp
            break
        if sl_hit:
            outcome = -sl
            break
        if tp_hit:
            outcome = tp
            break
        if time_mins[k] >= exit_time:
            outcome = (prices[k] - entry_price) / entry_price
            break
    else:
        outcome = (prices[n_prices-1] - entry_price) / entry_price

    return outcome, entry_date, True


# ==============================================================================
# REGIME FILTER — verbatim from v2
# ==============================================================================
def preload_regime_data(eval_start, eval_end):
    print("  Loading regime data (VIX, SPY, TLT)...")

    def _load_regime_asset(ticker):
        clean = TICKER_FILE_MAP.get(ticker, ticker)
        uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
        try:
            df = pd.read_parquet(uri)
            df = df.reset_index()
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if dt_cols:
                df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
            else:
                num_col = next((c for c in df.columns
                               if c in ('Datetime', 'Date', 'Time', 'Timestamp')),
                              df.columns[0])
                df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
            df = df.set_index('Datetime').sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df.index = df.index.tz_convert('America/Los_Angeles')
            return df
        except Exception:
            return None

    vix_df = _load_regime_asset('VIX')
    spy_df = _load_regime_asset('SPY')
    tlt_df = _load_regime_asset('TLT')

    spy_daily = spy_df['Close'].resample('D').last().dropna() if spy_df is not None else pd.Series(dtype=float)
    tlt_daily = tlt_df['Close'].resample('D').last().dropna() if tlt_df is not None else pd.Series(dtype=float)

    start = pd.Timestamp(eval_start, tz=PST) - timedelta(days=10)
    end   = pd.Timestamp(eval_end, tz=PST) + timedelta(days=1)

    def _build_lookup(df):
        if df is None: return {}
        sub = df[(df.index >= start) & (df.index <= end)]
        d_ints = (sub.index.year * 10000 + sub.index.month * 100 + sub.index.day).astype(int)
        m_ints = (sub.index.hour * 60 + sub.index.minute).astype(int)
        closes = sub['Close'].values
        return {(int(d_ints[i]), int(m_ints[i])): float(closes[i]) for i in range(len(sub))}

    vix_lk = _build_lookup(vix_df)
    spy_lk = _build_lookup(spy_df)
    tlt_lk = _build_lookup(tlt_df)

    def _trailing_median(lookup, d_int, entry_min, window=30):
        vals = []
        for m in range(entry_min - window + 1, entry_min + 1):
            v = lookup.get((d_int, m))
            if v is not None:
                vals.append(v)
        return float(np.median(vals)) if vals else None

    def _prev_close_map(daily_s):
        m = {}
        for i in range(1, len(daily_s)):
            d_int = int(daily_s.index[i].strftime('%Y%m%d'))
            m[d_int] = float(daily_s.iloc[i - 1])
        return m

    spy_prev = _prev_close_map(spy_daily)
    tlt_prev = _prev_close_map(tlt_daily)

    entry_minutes = list(range(360, 391))
    all_dates = pd.date_range(start, end, freq='D', tz=PST)
    regime_dict = {}
    for d in all_dates:
        d_int = int(d.strftime('%Y%m%d'))
        for em in entry_minutes:
            v_val = _trailing_median(vix_lk, d_int, em) or 15.0
            s_gap = 0.0
            sc = _trailing_median(spy_lk, d_int, em)
            sp = spy_prev.get(d_int)
            if sc is not None and sp is not None and sp != 0:
                s_gap = ((sc - sp) / sp) * 100
            t_gap = 0.0
            tc = _trailing_median(tlt_lk, d_int, em)
            tp_val = tlt_prev.get(d_int)
            if tc is not None and tp_val is not None and tp_val != 0:
                t_gap = ((tc - tp_val) / tp_val) * 100
            regime_dict[(d_int, em)] = np.array([v_val, t_gap, s_gap])

    print(f"    Regime data: {len(regime_dict)} (date, minute) combos")
    return regime_dict


# ==============================================================================
# OHLCV ARRAYS — verbatim from v2
# ==============================================================================
def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date):
    eval_start = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    eval_end   = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= eval_start) & (df_ohlcv.index < eval_end)]
    if len(df) == 0:
        return None
    return {
        'index':   df.index,
        'prices':  df['Close'].values.copy(),
        'highs':   df['High'].values.copy(),
        'lows':    df['Low'].values.copy(),
        'volumes': df['Volume'].values.copy(),
        'dates':   (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64),
        'times':   (df.index.hour * 60 + df.index.minute).values.astype(np.int64),
    }


# ==============================================================================
# MAIN BACKTEST FUNCTION
# ==============================================================================
def run_backtest(configs, label, market_df, regime_dict):
    print(f"\n{'='*W}")
    print(f"  BACKTEST: {label} — {len(configs)} tickers")
    print(f"  TEST Period: {TEST_START} → {TEST_END}")
    print(f"{'='*W}")

    if not configs:
        print(f"  No configs for {label} — skipping")
        return pd.DataFrame()

    all_trades = []

    for cfg in configs:
        ticker = cfg['ticker']
        model_id = cfg['model_id']
        seed = cfg['seed']
        entry_minute = cfg['entry_minute']
        mode = cfg['tpsl_mode']

        df_raw = load_master(ticker, start_date=TEST_START)
        if df_raw is None:
            print(f"  {ticker}: SKIP — no data")
            continue
        df_raw = df_raw.sort_index()
        df_raw = heal_data(df_raw)

        df_ohlcv_pst = df_raw.copy()
        df_ohlcv_pst.index = df_ohlcv_pst.index.tz_convert(PST)

        if USE_CSV_SCORES:
            signals = score_ticker_from_csv(ticker, cfg, df_ohlcv_pst)
            df_scaled = None
        else:
            df_feat = engineer_features(df_raw.copy(), market_df)
            stats = load_scaling_stats(cfg['stats_id'])
            df_scaled = apply_scaling(df_feat, stats)
            signals = score_ticker_period(ticker, cfg, df_scaled, df_raw)

        if not signals:
            src = "CSV" if USE_CSV_SCORES else "LIVE"
            print(f"  {ticker}: No signals above prob {cfg['prob_thresh']:.2f} [{src}]")
            del df_raw; gc.collect()
            continue

        if mode in ('atr', 'trailing_decay', 'atr_sl'):
            atr_series = compute_daily_atr_series(df_ohlcv_pst)
        elif mode == 'atr_adaptive':
            atr_series = compute_adaptive_atr_series(df_ohlcv_pst)
        else:
            atr_series = None

        vol_baselines = precompute_vol_baselines(df_ohlcv_pst)

        ohlcv = precompute_ohlcv_arrays(df_ohlcv_pst, TEST_START, TEST_END)

        if ohlcv is None:
            print(f"  {ticker}: No OHLCV in TEST window")
            del df_raw; gc.collect()
            continue

        trades_this_ticker = []
        filtered_vol = 0
        filtered_regime = 0
        last_traded_date = -1
        unfilled_limit = 0

        for sig in signals:
            bar_ts_pst = sig['bar_ts'].tz_convert(PST) if sig['bar_ts'].tzinfo else sig['bar_ts']
            d_int = bar_ts_pst.year * 10000 + bar_ts_pst.month * 100 + bar_ts_pst.day
            if d_int == last_traded_date:
                continue

            ohlcv_vol_idx = None
            for j in range(len(ohlcv['index'])):
                if ohlcv['dates'][j] == d_int and ohlcv['times'][j] == entry_minute:
                    ohlcv_vol_idx = j
                    break
            if ohlcv_vol_idx is not None:
                vol_at_entry = float(ohlcv['volumes'][ohlcv_vol_idx])
                if not check_volume_filter(vol_baselines, d_int, entry_minute,
                                            vol_at_entry, cfg['vol_filter_pct']):
                    filtered_vol += 1
                    continue

            regime_key = (d_int, entry_minute)
            vix_value = None
            if regime_key in regime_dict:
                vix_value = regime_dict[regime_key][0]

            atr_pct_val = 0.05
            if atr_series is not None:
                bar_ts_pst_ts = pd.Timestamp(bar_ts_pst)
                if bar_ts_pst_ts in atr_series.index:
                    v = float(atr_series.loc[bar_ts_pst_ts])
                    if not np.isnan(v) and v > 0:
                        atr_pct_val = v
                else:
                    prior = atr_series.index[atr_series.index <= bar_ts_pst_ts]
                    if len(prior) > 0:
                        v = float(atr_series.iloc[atr_series.index.get_loc(prior[-1])])
                        if not np.isnan(v) and v > 0:
                            atr_pct_val = v

            tp, sl = compute_dynamic_tpsl(cfg, atr_pct_val, vix_value)

            ohlcv_idx = ohlcv_vol_idx
            if ohlcv_idx is None:
                for j in range(len(ohlcv['index'])):
                    if ohlcv['dates'][j] == d_int and ohlcv['times'][j] == entry_minute:
                        ohlcv_idx = j
                        break

            if ohlcv_idx is None:
                continue

            if regime_key in regime_dict:
                rv = regime_dict[regime_key]
                vix_bad = rv[0] > cfg.get('vix_threshold', 99)
                tlt_bad = rv[1] < cfg.get('tlt_crash', -99)
                spy_bad = rv[2] > cfg.get('spy_ignore', 99)
                if vix_bad and tlt_bad and spy_bad:
                    filtered_regime += 1
                    continue

            if mode == 'trailing_decay':
                pnl, trade_date, valid = run_single_trade_trailing_decay(
                    ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                    ohlcv['dates'], ohlcv['times'],
                    ohlcv_idx, atr_pct_val,
                    cfg['trail_start_mult'], cfg['trail_end_mult'],
                    cfg['sl_mult'], EXIT_TIME_MIN
                )
            elif mode == 'grace':
                pnl, trade_date, valid = run_single_trade_grace(
                    ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                    ohlcv['dates'], ohlcv['times'],
                    ohlcv_idx, tp, sl, EXIT_TIME_MIN,
                    cfg['grace_minutes']
                )
            elif mode == 'limit_entry':
                pnl, trade_date, valid = run_single_trade_limit_entry(
                    ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                    ohlcv['dates'], ohlcv['times'],
                    ohlcv_idx, tp, sl, EXIT_TIME_MIN,
                    cfg['dip_pct'], cfg['max_wait_minutes']
                )
                if not valid:
                    unfilled_limit += 1
                    continue
            else:
                pnl, trade_date, valid = run_single_trade(
                    ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                    ohlcv['dates'], ohlcv['times'],
                    ohlcv_idx, tp, sl, EXIT_TIME_MIN
                )

            if valid:
                pnl = pnl - 2 * SLIPPAGE_PCT
                trades_this_ticker.append({
                    'ticker': ticker,
                    'model_id': model_id,
                    'experiment': cfg['experiment'],
                    'date_int': int(trade_date),
                    'pnl': float(pnl),
                    'prob': sig['prob'],
                    'tp': tp,
                    'sl': sl,
                    'atr_pct': atr_pct_val,
                    'mode': mode,
                })
                last_traded_date = int(trade_date)

        wins = sum(1 for t in trades_this_ticker if t['pnl'] > 0)
        losses = len(trades_this_ticker) - wins
        avg = np.mean([t['pnl'] for t in trades_this_ticker]) if trades_this_ticker else 0
        extra = ""
        if filtered_vol > 0:
            extra += f"  vol_f={filtered_vol}"
        if filtered_regime > 0:
            extra += f"  reg_f={filtered_regime}"
        if unfilled_limit > 0:
            extra += f"  unfilled={unfilled_limit}"
        seed_str = 'ENS' if cfg.get('is_ensemble') else str(seed)
        print(f"  {ticker:<7} {cfg['experiment']:<25} S{seed_str:>5}  "
              f"{len(signals):>3} sig → {len(trades_this_ticker):>3} trades "
              f"({wins}W/{losses}L)  avg {avg:>+6.2%}  {mode}{extra}")

        all_trades.extend(trades_this_ticker)
        del df_raw, df_ohlcv_pst, ohlcv; gc.collect()

    if not all_trades:
        print(f"\n  No trades simulated for {label}.")
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)
    trades_df['date'] = pd.to_datetime(trades_df['date_int'].astype(str), format='%Y%m%d')
    return trades_df


# ==============================================================================
# ANALYTICS HELPERS
# ==============================================================================
def print_per_asset_table(trades_df, label):
    if trades_df.empty:
        print(f"  No trades for {label}")
        return pd.DataFrame()

    months = max(1.0, (trades_df['date'].max() - trades_df['date'].min()).days / 30.44)

    print(f"\n  {'Ticker':<7} {'Experiment':<25} {'Mode':<12} {'Trades':>6} {'W/L':>7} "
          f"{'WR':>6} {'AvgPnL':>8} {'TotPnL':>9} {'Freq/mo':>8}")
    print(f"  {'─'*(W-2)}")

    rows = []
    for ticker in sorted(trades_df['ticker'].unique()):
        td = trades_df[trades_df['ticker'] == ticker]
        n_t = len(td)
        wins = int((td['pnl'] > 0).sum())
        losses = n_t - wins
        wr = wins / n_t if n_t > 0 else 0
        avg = td['pnl'].mean()
        total = td['pnl'].sum()
        freq = n_t / months
        exp = td['experiment'].iloc[0]
        mode = td['mode'].iloc[0]
        rows.append({
            'ticker': ticker, 'experiment': exp, 'mode': mode,
            'trades': n_t, 'wins': wins, 'losses': losses,
            'win_rate': wr, 'avg_pnl': avg, 'total_pnl': total, 'freq': freq,
        })
        print(f"  {ticker:<7} {exp:<25} {mode:<12} {n_t:>6} {wins:>3}/{losses:<3} "
              f"{wr:>5.0%} {avg:>+7.2%} {total:>+8.2%} {freq:>7.1f}")

    return pd.DataFrame(rows)


def portfolio_summary(trades_df, label):
    if trades_df.empty:
        return {}
    total_trades = len(trades_df)
    total_wins = int((trades_df['pnl'] > 0).sum())
    avg_pnl = trades_df['pnl'].mean()
    win_rate = total_wins / total_trades if total_trades > 0 else 0
    total_pnl = trades_df['pnl'].sum()
    n_tickers = trades_df['ticker'].nunique()
    months = max(1.0, (trades_df['date'].max() - trades_df['date'].min()).days / 30.44)
    return {
        'label': label,
        'tickers': n_tickers,
        'trades': total_trades,
        'wins': total_wins,
        'losses': total_trades - total_wins,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl,
        'freq_per_mo': total_trades / months,
    }


# ==============================================================================
# LOAD CONFIGS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  PORTFOLIO BACKTEST V3 — v12 vs v11 Head-to-Head")
print(f"  TEST Period: {TEST_START} → {TEST_END}")
print(f"  Composite: {COMPOSITE_METHOD}")
print(f"  Scoring: {'CSV' if USE_CSV_SCORES else 'LIVE TF'}")
print(f"  v12: model_version = '{MODEL_VERSION_V12}'")
print(f"  v11: model_version = '{MODEL_VERSION_V11}'")
print(f"{'='*W}")

v12_all = load_top_configs_from_bq(MODEL_VERSION_V12, MIN_TRADES, label='v12')
# v11_all = load_top_configs_from_bq(MODEL_VERSION_V11, MIN_TRADES, label='v11')

v12_configs = [c for c in v12_all if c['wf_avg_pnl'] >= DEPLOY_THRESHOLD]
# v11_configs = [c for c in v11_all if c['wf_avg_pnl'] >= DEPLOY_THRESHOLD]
v11_configs = []

print(f"\n  v12: {len(v12_configs)} DEPLOY configs (of {len(v12_all)} total)")
print(f"  v11: {len(v11_configs)} DEPLOY configs (skipped)")

# ==============================================================================
# PRECOMPUTE SHARED DATA
# ==============================================================================
market_df = build_market_context(start_date=TEST_START)
regime_dict = preload_regime_data(TEST_START, TEST_END)

# ==============================================================================
# RUN BACKTESTS
# ==============================================================================
v12_trades = run_backtest(v12_configs, f'v12 ({MODEL_VERSION_V12})', market_df, regime_dict)
v11_trades = run_backtest(v11_configs, f'v11 ({MODEL_VERSION_V11})', market_df, regime_dict)

# ==============================================================================
# TABLE 1: v12 PER-ASSET RESULTS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 1: v12 PER-ASSET RESULTS")
print(f"{'='*W}")
v12_stats = print_per_asset_table(v12_trades, 'v12')

# ==============================================================================
# TABLE 2: v11 PER-ASSET RESULTS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 2: v11 PER-ASSET RESULTS")
print(f"{'='*W}")
v11_stats = print_per_asset_table(v11_trades, 'v11')

# ==============================================================================
# TABLE 3: HEAD-TO-HEAD COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 3: HEAD-TO-HEAD COMPARISON (v12 vs v11)")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'v12 Mode':<12} {'v12 PnL':>8} {'v12 WR':>7} {'v12 N':>5}  │  "
      f"{'v11 Mode':<12} {'v11 PnL':>8} {'v11 WR':>7} {'v11 N':>5}  │  "
      f"{'ΔPnL':>8} {'Winner':>7}")
print(f"  {'─'*(W-2)}")

all_tickers = sorted(set(
    (list(v12_stats['ticker'].values) if not v12_stats.empty else []) +
    (list(v11_stats['ticker'].values) if not v11_stats.empty else [])
))

v12_wins = 0
v11_wins = 0
ties = 0

for ticker in all_tickers:
    v12_row = v12_stats[v12_stats['ticker'] == ticker] if not v12_stats.empty else pd.DataFrame()
    v11_row = v11_stats[v11_stats['ticker'] == ticker] if not v11_stats.empty else pd.DataFrame()

    if not v12_row.empty:
        v12_pnl = float(v12_row['avg_pnl'].iloc[0])
        v12_wr = float(v12_row['win_rate'].iloc[0])
        v12_n = int(v12_row['trades'].iloc[0])
        v12_mode = str(v12_row['mode'].iloc[0])
    else:
        v12_pnl = 0; v12_wr = 0; v12_n = 0; v12_mode = '---'

    if not v11_row.empty:
        v11_pnl = float(v11_row['avg_pnl'].iloc[0])
        v11_wr = float(v11_row['win_rate'].iloc[0])
        v11_n = int(v11_row['trades'].iloc[0])
        v11_mode = str(v11_row['mode'].iloc[0])
    else:
        v11_pnl = 0; v11_wr = 0; v11_n = 0; v11_mode = '---'

    delta = v12_pnl - v11_pnl
    if abs(delta) < 0.001:
        winner = 'TIE'
        ties += 1
    elif delta > 0:
        winner = 'v12'
        v12_wins += 1
    else:
        winner = 'v11'
        v11_wins += 1

    v12_pnl_s = f"{v12_pnl:>+7.2%}" if v12_n > 0 else f"{'---':>8}"
    v12_wr_s = f"{v12_wr:>6.0%}" if v12_n > 0 else f"{'---':>7}"
    v11_pnl_s = f"{v11_pnl:>+7.2%}" if v11_n > 0 else f"{'---':>8}"
    v11_wr_s = f"{v11_wr:>6.0%}" if v11_n > 0 else f"{'---':>7}"

    print(f"  {ticker:<7} {v12_mode:<12} {v12_pnl_s} {v12_wr_s} {v12_n:>5}  │  "
          f"{v11_mode:<12} {v11_pnl_s} {v11_wr_s} {v11_n:>5}  │  "
          f"{delta:>+7.2%} {winner:>7}")

print(f"\n  Score: v12 wins {v12_wins}  |  v11 wins {v11_wins}  |  ties {ties}")

# ==============================================================================
# TABLE 4: PORTFOLIO SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 4: PORTFOLIO SUMMARY")
print(f"{'='*W}")

s12 = portfolio_summary(v12_trades, 'v12')
s11 = portfolio_summary(v11_trades, 'v11')

if s12 and s11:
    print(f"  {'Metric':<20} {'v12':>12} {'v11':>12} {'Delta':>12}")
    print(f"  {'─'*60}")
    print(f"  {'Tickers':<20} {s12['tickers']:>12} {s11['tickers']:>12} {s12['tickers']-s11['tickers']:>+12}")
    print(f"  {'Total trades':<20} {s12['trades']:>12} {s11['trades']:>12} {s12['trades']-s11['trades']:>+12}")
    print(f"  {'Wins':<20} {s12['wins']:>12} {s11['wins']:>12} {s12['wins']-s11['wins']:>+12}")
    print(f"  {'Win rate':<20} {s12['win_rate']:>11.1%} {s11['win_rate']:>11.1%} {s12['win_rate']-s11['win_rate']:>+11.1%}")
    print(f"  {'Avg PnL/trade':<20} {s12['avg_pnl']:>11.2%} {s11['avg_pnl']:>11.2%} {s12['avg_pnl']-s11['avg_pnl']:>+11.2%}")
    print(f"  {'Total PnL':<20} {s12['total_pnl']:>11.2%} {s11['total_pnl']:>11.2%} {s12['total_pnl']-s11['total_pnl']:>+11.2%}")
    print(f"  {'Trades/month':<20} {s12['freq_per_mo']:>11.1f} {s11['freq_per_mo']:>11.1f} {s12['freq_per_mo']-s11['freq_per_mo']:>+11.1f}")
elif s12:
    print(f"  v12: {s12['trades']} trades, avg {s12['avg_pnl']:+.2%}, WR {s12['win_rate']:.0%}")
    print(f"  v11: No trades")
elif s11:
    print(f"  v12: No trades")
    print(f"  v11: {s11['trades']} trades, avg {s11['avg_pnl']:+.2%}, WR {s11['win_rate']:.0%}")

# ==============================================================================
# TABLE 5: v12-ONLY TICKERS (unlocked by new modes)
# ==============================================================================
v12_only = set()
if not v12_stats.empty:
    v12_tickers_set = set(v12_stats['ticker'].values)
    v11_tickers_set = set(v11_stats['ticker'].values) if not v11_stats.empty else set()
    v12_only = v12_tickers_set - v11_tickers_set

if v12_only:
    print(f"\n\n{'='*W}")
    print(f"  TABLE 5: v12-ONLY TICKERS (unlocked by new modes — not in v11)")
    print(f"{'='*W}")
    for ticker in sorted(v12_only):
        row = v12_stats[v12_stats['ticker'] == ticker].iloc[0]
        print(f"  {ticker:<7} {row['mode']:<12} {row['trades']} trades, "
              f"avg {row['avg_pnl']:+.2%}, WR {row['win_rate']:.0%}")

# ==============================================================================
# TABLE 6: MONTHLY BREAKDOWN (v12 vs v11)
# ==============================================================================
def monthly_breakdown(trades_df, label):
    if trades_df.empty:
        return pd.DataFrame()
    trades_df = trades_df.copy()
    trades_df['month'] = trades_df['date'].dt.to_period('M')
    rows = []
    for month in sorted(trades_df['month'].unique()):
        td = trades_df[trades_df['month'] == month]
        n = len(td)
        wins = int((td['pnl'] > 0).sum())
        wr = wins / n if n > 0 else 0
        avg = td['pnl'].mean()
        total = td['pnl'].sum()
        rows.append({
            'month': str(month), 'trades': n, 'wins': wins,
            'losses': n - wins, 'win_rate': wr,
            'avg_pnl': avg, 'total_pnl': total,
        })
    return pd.DataFrame(rows)

print(f"\n\n{'='*W}")
print(f"  TABLE 6: MONTHLY BREAKDOWN (v12 vs v11)")
print(f"{'='*W}")

v12_monthly = monthly_breakdown(v12_trades, 'v12')
v11_monthly = monthly_breakdown(v11_trades, 'v11')

all_months = sorted(set(
    (list(v12_monthly['month'].values) if not v12_monthly.empty else []) +
    (list(v11_monthly['month'].values) if not v11_monthly.empty else [])
))

print(f"\n  {'Month':<10} │ {'v12 N':>5} {'v12 WR':>7} {'v12 Avg':>8} {'v12 Tot':>9}  │  "
      f"{'v11 N':>5} {'v11 WR':>7} {'v11 Avg':>8} {'v11 Tot':>9}  │  "
      f"{'ΔAvg':>8} {'ΔTot':>9}")
print(f"  {'─'*(W-2)}")

for month in all_months:
    v12r = v12_monthly[v12_monthly['month'] == month] if not v12_monthly.empty else pd.DataFrame()
    v11r = v11_monthly[v11_monthly['month'] == month] if not v11_monthly.empty else pd.DataFrame()

    v12_n = int(v12r['trades'].iloc[0]) if not v12r.empty else 0
    v12_wr = float(v12r['win_rate'].iloc[0]) if not v12r.empty else 0
    v12_avg = float(v12r['avg_pnl'].iloc[0]) if not v12r.empty else 0
    v12_tot = float(v12r['total_pnl'].iloc[0]) if not v12r.empty else 0

    v11_n = int(v11r['trades'].iloc[0]) if not v11r.empty else 0
    v11_wr = float(v11r['win_rate'].iloc[0]) if not v11r.empty else 0
    v11_avg = float(v11r['avg_pnl'].iloc[0]) if not v11r.empty else 0
    v11_tot = float(v11r['total_pnl'].iloc[0]) if not v11r.empty else 0

    d_avg = v12_avg - v11_avg
    d_tot = v12_tot - v11_tot

    print(f"  {month:<10} │ {v12_n:>5} {v12_wr:>6.0%} {v12_avg:>+7.2%} {v12_tot:>+8.2%}  │  "
          f"{v11_n:>5} {v11_wr:>6.0%} {v11_avg:>+7.2%} {v11_tot:>+8.2%}  │  "
          f"{d_avg:>+7.2%} {d_tot:>+8.2%}")

# ==============================================================================
# TABLE 7: WEEKLY BREAKDOWN (v12 vs v11)
# ==============================================================================
def weekly_breakdown(trades_df, label):
    if trades_df.empty:
        return pd.DataFrame()
    trades_df = trades_df.copy()
    trades_df['week_start'] = trades_df['date'] - pd.to_timedelta(trades_df['date'].dt.dayofweek, unit='D')
    rows = []
    for ws in sorted(trades_df['week_start'].unique()):
        td = trades_df[trades_df['week_start'] == ws]
        n = len(td)
        wins = int((td['pnl'] > 0).sum())
        wr = wins / n if n > 0 else 0
        avg = td['pnl'].mean()
        total = td['pnl'].sum()
        rows.append({
            'week_start': ws, 'week_label': ws.strftime('%b %d'),
            'trades': n, 'wins': wins, 'losses': n - wins,
            'win_rate': wr, 'avg_pnl': avg, 'total_pnl': total,
        })
    return pd.DataFrame(rows)

print(f"\n\n{'='*W}")
print(f"  TABLE 7: WEEKLY BREAKDOWN (v12 vs v11)")
print(f"{'='*W}")

v12_weekly = weekly_breakdown(v12_trades, 'v12')
v11_weekly = weekly_breakdown(v11_trades, 'v11')

all_weeks = sorted(set(
    (list(v12_weekly['week_label'].values) if not v12_weekly.empty else []) +
    (list(v11_weekly['week_label'].values) if not v11_weekly.empty else [])
), key=lambda w: pd.Timestamp('2026-' + w))

print(f"\n  {'Week':<10} │ {'v12 N':>5} {'v12 WR':>7} {'v12 Avg':>8} {'v12 Tot':>9}  │  "
      f"{'v11 N':>5} {'v11 WR':>7} {'v11 Avg':>8} {'v11 Tot':>9}  │  "
      f"{'ΔAvg':>8} {'Winner':>7}")
print(f"  {'─'*(W-2)}")

for week in all_weeks:
    v12r = v12_weekly[v12_weekly['week_label'] == week] if not v12_weekly.empty else pd.DataFrame()
    v11r = v11_weekly[v11_weekly['week_label'] == week] if not v11_weekly.empty else pd.DataFrame()

    v12_n = int(v12r['trades'].iloc[0]) if not v12r.empty else 0
    v12_wr = float(v12r['win_rate'].iloc[0]) if not v12r.empty else 0
    v12_avg = float(v12r['avg_pnl'].iloc[0]) if not v12r.empty else 0
    v12_tot = float(v12r['total_pnl'].iloc[0]) if not v12r.empty else 0

    v11_n = int(v11r['trades'].iloc[0]) if not v11r.empty else 0
    v11_wr = float(v11r['win_rate'].iloc[0]) if not v11r.empty else 0
    v11_avg = float(v11r['avg_pnl'].iloc[0]) if not v11r.empty else 0
    v11_tot = float(v11r['total_pnl'].iloc[0]) if not v11r.empty else 0

    d_avg = v12_avg - v11_avg
    if abs(d_avg) < 0.001:
        winner = 'TIE'
    elif d_avg > 0:
        winner = 'v12'
    else:
        winner = 'v11'

    print(f"  {week:<10} │ {v12_n:>5} {v12_wr:>6.0%} {v12_avg:>+7.2%} {v12_tot:>+8.2%}  │  "
          f"{v11_n:>5} {v11_wr:>6.0%} {v11_avg:>+7.2%} {v11_tot:>+8.2%}  │  "
          f"{d_avg:>+7.2%} {winner:>7}")

# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 1, figsize=(14, 12))
fig.suptitle(f'v12 vs v11 Head-to-Head Backtest\n'
             f'TEST ({TEST_START} → {TEST_END})', fontsize=14, fontweight='bold')

ax1 = axes[0]
if not v12_stats.empty or not v11_stats.empty:
    comp_data = []
    for ticker in all_tickers:
        v12_p = float(v12_stats[v12_stats['ticker'] == ticker]['avg_pnl'].iloc[0]) * 100 if not v12_stats.empty and ticker in v12_stats['ticker'].values else 0
        v11_p = float(v11_stats[v11_stats['ticker'] == ticker]['avg_pnl'].iloc[0]) * 100 if not v11_stats.empty and ticker in v11_stats['ticker'].values else 0
        comp_data.append({'ticker': ticker, 'v12': v12_p, 'v11': v11_p})
    comp_df = pd.DataFrame(comp_data).sort_values('v12', ascending=True)
    y_pos = np.arange(len(comp_df))
    bar_h = 0.35
    ax1.barh(y_pos - bar_h/2, comp_df['v12'].values, bar_h,
             label='v12', color='#2196F3', alpha=0.8)
    ax1.barh(y_pos + bar_h/2, comp_df['v11'].values, bar_h,
             label='v11', color='#FF9800', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(comp_df['ticker'].values)
    ax1.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    ax1.set_title('Avg PnL per Trade (%) — v12 vs v11', fontsize=12)
    ax1.set_xlabel('Avg PnL (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')

ax2 = axes[1]
for trades_df, label, color in [(v12_trades, 'v12', '#2196F3'), (v11_trades, 'v11', '#FF9800')]:
    if trades_df.empty:
        continue
    daily_pnl = trades_df.groupby('date')['pnl'].mean()
    equity = STARTING_CAPITAL
    eq_points = [{'date': daily_pnl.index[0] - pd.Timedelta(days=1), 'equity': equity}]
    for d, p in daily_pnl.items():
        n_trades = len(trades_df[trades_df['date'] == d])
        alloc = equity / max(n_trades, 1)
        equity += alloc * p * n_trades
        eq_points.append({'date': d, 'equity': equity})
    eq_df = pd.DataFrame(eq_points).set_index('date')
    ax2.plot(eq_df.index, eq_df['equity'] / 1000, label=label, color=color, linewidth=2)

ax2.axhline(y=STARTING_CAPITAL / 1000, color='gray', linewidth=0.5, linestyle='--')
ax2.set_title(f'Cumulative Equity (${STARTING_CAPITAL/1000:.0f}K start)', fontsize=12)
ax2.set_ylabel('Equity ($K)')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax2.tick_params(axis='x', rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = '/tmp/backtest_v3_v12_vs_v11.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Charts saved: {chart_path}")
try:
    from IPython.display import display, Image
    display(Image(filename=chart_path))
except ImportError:
    plt.show()

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FINAL SUMMARY")
print(f"{'='*W}")
if s12:
    print(f"  v12: {s12['trades']} trades across {s12['tickers']} tickers, "
          f"avg {s12['avg_pnl']:+.2%}/trade, {s12['win_rate']:.0%} WR, "
          f"total PnL {s12['total_pnl']:+.2%}")
if s11:
    print(f"  v11: {s11['trades']} trades across {s11['tickers']} tickers, "
          f"avg {s11['avg_pnl']:+.2%}/trade, {s11['win_rate']:.0%} WR, "
          f"total PnL {s11['total_pnl']:+.2%}")
if s12 and s11:
    d_pnl = s12['avg_pnl'] - s11['avg_pnl']
    d_wr = s12['win_rate'] - s11['win_rate']
    verdict = 'v12 WINS' if d_pnl > 0 else 'v11 WINS' if d_pnl < -0.001 else 'TIE'
    print(f"\n  VERDICT: {verdict}")
    print(f"    ΔAvg PnL: {d_pnl:+.2%}  |  ΔWR: {d_wr:+.1%}  |  Head-to-head: v12 {v12_wins} / v11 {v11_wins} / tie {ties}")
if v12_only:
    print(f"    v12 unlocked {len(v12_only)} new tickers: {', '.join(sorted(v12_only))}")
print(f"{'='*W}")
