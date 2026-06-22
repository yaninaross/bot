"""
backtest_composite_compare.py — Seed Agreement vs Opt PnL×WR Head-to-Head
==========================================================================
Runs BOTH composite methods on v12 configs, backtests each with CSV scores,
then compares results side by side. Reuses all functions from backtest_portfolio_v3.

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
MODEL_VERSION = 'robust_wf_v12'
MIN_TRADES = 3
DEPLOY_THRESHOLD = 0.018

TEST_START = '2026-04-01'
TEST_END   = '2026-06-13'

STARTING_CAPITAL = 200_000

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
# LOAD RAW BQ DATA (shared between both methods)
# ==============================================================================
def load_raw_wf_data():
    print(f"  Loading WF results from BQ (model_version={MODEL_VERSION})...")
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
    WHERE model_version = '{MODEL_VERSION}'
      AND survived_a = TRUE
      AND val_a_count >= {MIN_TRADES}
      AND val_a_avg_pnl > 0
    """
    df = bq_client.query(query).to_dataframe()
    print(f"    {len(df)} rows loaded from BQ")

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

    for col in ['grace_minutes', 'max_wait_minutes', 'vol_filter_pct']:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    for col in ['dip_pct', 'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    MAX_VA_OPT_RATIO = 3.0
    before_n = len(df)
    df = df[df['opt_avg_pnl'] > 0].copy()
    va_opt_ratio = df['val_a_avg_pnl'] / df['opt_avg_pnl']
    overfit_mask = va_opt_ratio > MAX_VA_OPT_RATIO
    df = df[~overfit_mask]
    print(f"    Overfit filter: {before_n} -> {len(df)} configs")

    return df


def _row_to_config(row, wf_avg_pnl_col='val_a_avg_pnl', wf_wr_col='val_a_win_rate',
                   wf_count_col='val_a_count', wf_freq_col='val_a_freq',
                   composite_val=0.0, extra_fields=None):
    exp = row['experiment']
    mode = row['tpsl_mode']
    mode_sfx = MODE_SUFFIX.get(mode, '')
    base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp

    is_ens = bool(row.get('is_ensemble', True))
    seed_val = int(row.get('seed', -1))
    if is_ens or seed_val == -1:
        base_num = int(base_exp.split('_')[0])
        seed_list = list(range(base_num * 100 + 1, base_num * 100 + 16))
        is_ens = True
        seed_val = -1
    else:
        seed_list = [seed_val]

    cfg = {
        'ticker': row['ticker'],
        'base_experiment': base_exp,
        'experiment': exp,
        'tpsl_mode': mode,
        'seed': seed_val,
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
        'wf_avg_pnl': float(row[wf_avg_pnl_col]),
        'wf_win_rate': float(row[wf_wr_col]),
        'wf_count': int(row[wf_count_col]),
        'wf_freq': float(row.get(wf_freq_col) or 0),
        'wf_composite': composite_val,
    }
    if extra_fields:
        cfg.update(extra_fields)
    return cfg


# ==============================================================================
# COMPOSITE METHOD A: seed_agreement
# ==============================================================================
def select_seed_agreement(df):
    print(f"\n  --- Method A: seed_agreement ---")
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
    print(f"    {len(sa)} config groups -> {len(best)} tickers")

    configs = []
    for _, row in best.iterrows():
        cfg = _row_to_config(
            row,
            wf_avg_pnl_col='mean_va_pnl',
            wf_wr_col='mean_va_wr',
            wf_count_col='mean_va_count',
            wf_freq_col='mean_va_freq',
            composite_val=float(row['composite']),
            extra_fields={
                'seed': -1, 'is_ensemble': True,
                'seed_agree_pct': float(row['agree_pct']),
                'n_seeds': int(row['n_seeds']),
            }
        )
        configs.append(cfg)

    deployed = [c for c in configs if c['wf_avg_pnl'] >= DEPLOY_THRESHOLD]
    print(f"    {len(deployed)} DEPLOY configs (of {len(configs)} total, threshold={DEPLOY_THRESHOLD:.1%})")
    for c in deployed:
        print(f"      {c['ticker']:<7} {c['experiment']:<25} ENS  "
              f"mode={c['tpsl_mode']:<14} pnl={c['wf_avg_pnl']:.2%} wr={c['wf_win_rate']:.0%} "
              f"N={c['wf_count']} agree={c['seed_agree_pct']:.0%} ({c['n_seeds']}seeds)")
    return deployed


# ==============================================================================
# COMPOSITE METHOD B: opt_pnl_x_wr
# ==============================================================================
def select_opt_pnl_x_wr(df):
    print(f"\n  --- Method B: opt_pnl_x_wr ---")
    d = df.copy()
    d['composite'] = d['opt_avg_pnl'] * d['opt_win_rate']
    best = d.loc[d.groupby('ticker')['composite'].idxmax()]
    best = best.sort_values('composite', ascending=False)

    configs = []
    for _, row in best.iterrows():
        cfg = _row_to_config(row, composite_val=float(row['composite']))
        configs.append(cfg)

    deployed = [c for c in configs if c['wf_avg_pnl'] >= DEPLOY_THRESHOLD]
    print(f"    {len(deployed)} DEPLOY configs (of {len(configs)} total, threshold={DEPLOY_THRESHOLD:.1%})")
    for c in deployed:
        seed_str = 'ENS' if c['is_ensemble'] else str(c['seed'])
        print(f"      {c['ticker']:<7} {c['experiment']:<25} {seed_str:>5}  "
              f"mode={c['tpsl_mode']:<14} pnl={c['wf_avg_pnl']:.2%} wr={c['wf_win_rate']:.0%} N={c['wf_count']}")
    return deployed


# ==============================================================================
# DATA LOADING
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
        print(f"    load_master({ticker}): FAILED -- {e}")
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
# MARKET CONTEXT
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
    print(f"    Market context: {len(result)} rows")
    return result


# ==============================================================================
# CSV-BASED SCORING
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
        seed_list = [seed] if seed in all_seeds else []
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
# DYNAMIC TP/SL
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
# ATR
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
# VOLUME FILTER
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
# TRADE SIMULATION (Numba)
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
            outcome = -sl; break
        if tp_hit:
            outcome = tp; break
        if time_mins[k] >= exit_time:
            outcome = (prices[k] - entry_price) / entry_price
            break
    else:
        outcome = (prices[n_prices-1] - entry_price) / entry_price
    return outcome, entry_date, True


# ==============================================================================
# REGIME FILTER
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
# OHLCV ARRAYS
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
    print(f"  BACKTEST: {label} -- {len(configs)} tickers")
    print(f"  TEST Period: {TEST_START} -> {TEST_END}")
    print(f"{'='*W}")

    if not configs:
        print(f"  No configs for {label} -- skipping")
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
            print(f"  {ticker}: SKIP -- no data")
            continue
        df_raw = df_raw.sort_index()
        df_raw = heal_data(df_raw)

        df_ohlcv_pst = df_raw.copy()
        df_ohlcv_pst.index = df_ohlcv_pst.index.tz_convert(PST)

        signals = score_ticker_from_csv(ticker, cfg, df_ohlcv_pst)

        if not signals:
            print(f"  {ticker}: No signals above prob {cfg['prob_thresh']:.2f}")
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
                    cfg['sl_mult'], EXIT_TIME_MIN)
            elif mode == 'grace':
                pnl, trade_date, valid = run_single_trade_grace(
                    ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                    ohlcv['dates'], ohlcv['times'],
                    ohlcv_idx, tp, sl, EXIT_TIME_MIN, cfg['grace_minutes'])
            elif mode == 'limit_entry':
                pnl, trade_date, valid = run_single_trade_limit_entry(
                    ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                    ohlcv['dates'], ohlcv['times'],
                    ohlcv_idx, tp, sl, EXIT_TIME_MIN,
                    cfg['dip_pct'], cfg['max_wait_minutes'])
                if not valid:
                    unfilled_limit += 1
                    continue
            else:
                pnl, trade_date, valid = run_single_trade(
                    ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                    ohlcv['dates'], ohlcv['times'],
                    ohlcv_idx, tp, sl, EXIT_TIME_MIN)

            if valid:
                pnl = pnl - 2 * SLIPPAGE_PCT
                trades_this_ticker.append({
                    'ticker': ticker,
                    'model_id': model_id,
                    'experiment': cfg['experiment'],
                    'date_int': int(trade_date),
                    'pnl': float(pnl),
                    'prob': sig['prob'],
                    'tp': tp, 'sl': sl,
                    'atr_pct': atr_pct_val,
                    'mode': mode,
                })
                last_traded_date = int(trade_date)

        wins = sum(1 for t in trades_this_ticker if t['pnl'] > 0)
        losses = len(trades_this_ticker) - wins
        avg = np.mean([t['pnl'] for t in trades_this_ticker]) if trades_this_ticker else 0
        extra = ""
        if filtered_vol > 0: extra += f"  vol_f={filtered_vol}"
        if filtered_regime > 0: extra += f"  reg_f={filtered_regime}"
        if unfilled_limit > 0: extra += f"  unfilled={unfilled_limit}"
        seed_str = 'ENS' if cfg.get('is_ensemble') else str(seed)
        print(f"  {ticker:<7} {cfg['experiment']:<25} S{seed_str:>5}  "
              f"{len(signals):>3} sig -> {len(trades_this_ticker):>3} trades "
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
# ANALYTICS
# ==============================================================================
def print_per_asset_table(trades_df, label):
    if trades_df.empty:
        print(f"  No trades for {label}")
        return pd.DataFrame()

    months = max(1.0, (trades_df['date'].max() - trades_df['date'].min()).days / 30.44)

    print(f"\n  {'Ticker':<7} {'Experiment':<25} {'Mode':<14} {'Trades':>6} {'W/L':>7} "
          f"{'WR':>6} {'AvgPnL':>8} {'TotPnL':>9} {'Freq/mo':>8}")
    print(f"  {'-'*(W-2)}")

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
        print(f"  {ticker:<7} {exp:<25} {mode:<14} {n_t:>6} {wins:>3}/{losses:<3} "
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
        'label': label, 'tickers': n_tickers, 'trades': total_trades,
        'wins': total_wins, 'losses': total_trades - total_wins,
        'win_rate': win_rate, 'avg_pnl': avg_pnl, 'total_pnl': total_pnl,
        'freq_per_mo': total_trades / months,
    }


def monthly_breakdown(trades_df):
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
        rows.append({
            'month': str(month), 'trades': n, 'wins': wins,
            'losses': n - wins, 'win_rate': wr,
            'avg_pnl': td['pnl'].mean(), 'total_pnl': td['pnl'].sum(),
        })
    return pd.DataFrame(rows)


# ==============================================================================
# RUN EVERYTHING
# ==============================================================================
print(f"\n{'='*W}")
print(f"  COMPOSITE METHOD COMPARISON BACKTEST")
print(f"  TEST Period: {TEST_START} -> {TEST_END}")
print(f"  Model: {MODEL_VERSION}")
print(f"  Deploy Threshold: {DEPLOY_THRESHOLD:.1%}")
print(f"  Scoring: CSV (pre-computed)")
print(f"{'='*W}")

# --- Load BQ data once ---
raw_df = load_raw_wf_data()

# --- Select configs with both methods ---
configs_A = select_seed_agreement(raw_df)
configs_B = select_opt_pnl_x_wr(raw_df)

# --- Precompute shared data ---
market_df = build_market_context(start_date=TEST_START)
regime_dict = preload_regime_data(TEST_START, TEST_END)

# --- Run backtests ---
trades_A = run_backtest(configs_A, 'A: seed_agreement', market_df, regime_dict)
trades_B = run_backtest(configs_B, 'B: opt_pnl_x_wr', market_df, regime_dict)

# ==============================================================================
# TABLE 1: PER-ASSET RESULTS (side by side)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 1: PER-ASSET RESULTS -- seed_agreement")
print(f"{'='*W}")
stats_A = print_per_asset_table(trades_A, 'seed_agreement')

print(f"\n\n{'='*W}")
print(f"  TABLE 2: PER-ASSET RESULTS -- opt_pnl_x_wr")
print(f"{'='*W}")
stats_B = print_per_asset_table(trades_B, 'opt_pnl_x_wr')

# ==============================================================================
# TABLE 3: HEAD-TO-HEAD COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 3: HEAD-TO-HEAD (seed_agreement vs opt_pnl_x_wr)")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'A Mode':<14} {'A PnL':>8} {'A WR':>6} {'A N':>5}  |  "
      f"{'B Mode':<14} {'B PnL':>8} {'B WR':>6} {'B N':>5}  |  "
      f"{'dPnL':>8} {'Winner':>7}")
print(f"  {'-'*(W-2)}")

all_tickers = sorted(set(
    (list(stats_A['ticker'].values) if not stats_A.empty else []) +
    (list(stats_B['ticker'].values) if not stats_B.empty else [])
))

a_wins = 0; b_wins = 0; ties = 0

for ticker in all_tickers:
    a_row = stats_A[stats_A['ticker'] == ticker] if not stats_A.empty else pd.DataFrame()
    b_row = stats_B[stats_B['ticker'] == ticker] if not stats_B.empty else pd.DataFrame()

    if not a_row.empty:
        a_pnl = float(a_row['avg_pnl'].iloc[0])
        a_wr = float(a_row['win_rate'].iloc[0])
        a_n = int(a_row['trades'].iloc[0])
        a_mode = str(a_row['mode'].iloc[0])
    else:
        a_pnl = 0; a_wr = 0; a_n = 0; a_mode = '---'

    if not b_row.empty:
        b_pnl = float(b_row['avg_pnl'].iloc[0])
        b_wr = float(b_row['win_rate'].iloc[0])
        b_n = int(b_row['trades'].iloc[0])
        b_mode = str(b_row['mode'].iloc[0])
    else:
        b_pnl = 0; b_wr = 0; b_n = 0; b_mode = '---'

    delta = a_pnl - b_pnl
    if abs(delta) < 0.001:
        winner = 'TIE'; ties += 1
    elif delta > 0:
        winner = 'A'; a_wins += 1
    else:
        winner = 'B'; b_wins += 1

    a_pnl_s = f"{a_pnl:>+7.2%}" if a_n > 0 else f"{'---':>8}"
    a_wr_s = f"{a_wr:>5.0%}" if a_n > 0 else f"{'---':>6}"
    b_pnl_s = f"{b_pnl:>+7.2%}" if b_n > 0 else f"{'---':>8}"
    b_wr_s = f"{b_wr:>5.0%}" if b_n > 0 else f"{'---':>6}"

    print(f"  {ticker:<7} {a_mode:<14} {a_pnl_s} {a_wr_s} {a_n:>5}  |  "
          f"{b_mode:<14} {b_pnl_s} {b_wr_s} {b_n:>5}  |  "
          f"{delta:>+7.2%} {winner:>7}")

print(f"\n  Score: A wins {a_wins}  |  B wins {b_wins}  |  ties {ties}")

# ==============================================================================
# TABLE 4: PORTFOLIO SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 4: PORTFOLIO SUMMARY")
print(f"{'='*W}")

sA = portfolio_summary(trades_A, 'seed_agreement')
sB = portfolio_summary(trades_B, 'opt_pnl_x_wr')

print(f"  {'Metric':<20} {'seed_agreement':>16} {'opt_pnl_x_wr':>16} {'Delta':>12}")
print(f"  {'-'*68}")
if sA and sB:
    print(f"  {'Tickers':<20} {sA['tickers']:>16} {sB['tickers']:>16} {sA['tickers']-sB['tickers']:>+12}")
    print(f"  {'Total trades':<20} {sA['trades']:>16} {sB['trades']:>16} {sA['trades']-sB['trades']:>+12}")
    print(f"  {'Wins':<20} {sA['wins']:>16} {sB['wins']:>16} {sA['wins']-sB['wins']:>+12}")
    print(f"  {'Win rate':<20} {sA['win_rate']:>15.1%} {sB['win_rate']:>15.1%} {sA['win_rate']-sB['win_rate']:>+11.1%}")
    print(f"  {'Avg PnL/trade':<20} {sA['avg_pnl']:>15.2%} {sB['avg_pnl']:>15.2%} {sA['avg_pnl']-sB['avg_pnl']:>+11.2%}")
    print(f"  {'Total PnL':<20} {sA['total_pnl']:>15.2%} {sB['total_pnl']:>15.2%} {sA['total_pnl']-sB['total_pnl']:>+11.2%}")
    print(f"  {'Trades/month':<20} {sA['freq_per_mo']:>15.1f} {sB['freq_per_mo']:>15.1f} {sA['freq_per_mo']-sB['freq_per_mo']:>+11.1f}")
elif sA:
    print(f"  A: {sA['trades']} trades, avg {sA['avg_pnl']:+.2%}, WR {sA['win_rate']:.0%}")
    print(f"  B: No trades")
elif sB:
    print(f"  A: No trades")
    print(f"  B: {sB['trades']} trades, avg {sB['avg_pnl']:+.2%}, WR {sB['win_rate']:.0%}")

# ==============================================================================
# TABLE 5: MONTHLY BREAKDOWN
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 5: MONTHLY BREAKDOWN")
print(f"{'='*W}")

monthly_A = monthly_breakdown(trades_A)
monthly_B = monthly_breakdown(trades_B)

all_months = sorted(set(
    (list(monthly_A['month'].values) if not monthly_A.empty else []) +
    (list(monthly_B['month'].values) if not monthly_B.empty else [])
))

print(f"\n  {'Month':<10} | {'A N':>5} {'A WR':>6} {'A Avg':>8} {'A Tot':>9}  |  "
      f"{'B N':>5} {'B WR':>6} {'B Avg':>8} {'B Tot':>9}  |  "
      f"{'dAvg':>8} {'Winner':>7}")
print(f"  {'-'*(W-2)}")

for month in all_months:
    ar = monthly_A[monthly_A['month'] == month] if not monthly_A.empty else pd.DataFrame()
    br = monthly_B[monthly_B['month'] == month] if not monthly_B.empty else pd.DataFrame()

    a_n = int(ar['trades'].iloc[0]) if not ar.empty else 0
    a_wr = float(ar['win_rate'].iloc[0]) if not ar.empty else 0
    a_avg = float(ar['avg_pnl'].iloc[0]) if not ar.empty else 0
    a_tot = float(ar['total_pnl'].iloc[0]) if not ar.empty else 0

    b_n = int(br['trades'].iloc[0]) if not br.empty else 0
    b_wr = float(br['win_rate'].iloc[0]) if not br.empty else 0
    b_avg = float(br['avg_pnl'].iloc[0]) if not br.empty else 0
    b_tot = float(br['total_pnl'].iloc[0]) if not br.empty else 0

    d_avg = a_avg - b_avg
    winner = 'TIE' if abs(d_avg) < 0.001 else ('A' if d_avg > 0 else 'B')

    print(f"  {month:<10} | {a_n:>5} {a_wr:>5.0%} {a_avg:>+7.2%} {a_tot:>+8.2%}  |  "
          f"{b_n:>5} {b_wr:>5.0%} {b_avg:>+7.2%} {b_tot:>+8.2%}  |  "
          f"{d_avg:>+7.2%} {winner:>7}")

# ==============================================================================
# TABLE 6: OVERLAP ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 6: TICKER OVERLAP")
print(f"{'='*W}")

tickers_A = set(stats_A['ticker'].values) if not stats_A.empty else set()
tickers_B = set(stats_B['ticker'].values) if not stats_B.empty else set()
both = tickers_A & tickers_B
a_only = tickers_A - tickers_B
b_only = tickers_B - tickers_A

print(f"  A only ({len(a_only)}): {sorted(a_only)}")
print(f"  B only ({len(b_only)}): {sorted(b_only)}")
print(f"  Both   ({len(both)}): {sorted(both)}")

if both:
    print(f"\n  Shared tickers head-to-head:")
    print(f"  {'Ticker':<7} {'A avg':>8} {'B avg':>8} {'Winner':>7}")
    print(f"  {'-'*35}")
    a_better = 0; b_better = 0
    for t in sorted(both):
        a_p = float(stats_A[stats_A['ticker'] == t]['avg_pnl'].iloc[0])
        b_p = float(stats_B[stats_B['ticker'] == t]['avg_pnl'].iloc[0])
        w = 'A' if a_p > b_p + 0.001 else ('B' if b_p > a_p + 0.001 else 'TIE')
        if w == 'A': a_better += 1
        elif w == 'B': b_better += 1
        print(f"  {t:<7} {a_p:>+7.2%} {b_p:>+7.2%} {w:>7}")
    print(f"\n  On shared tickers: A wins {a_better}, B wins {b_better}")

# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 1, figsize=(14, 12))
fig.suptitle(f'Composite Method Comparison\n'
             f'TEST ({TEST_START} -> {TEST_END})', fontsize=14, fontweight='bold')

ax1 = axes[0]
if not stats_A.empty or not stats_B.empty:
    comp_data = []
    for ticker in all_tickers:
        a_p = float(stats_A[stats_A['ticker'] == ticker]['avg_pnl'].iloc[0]) * 100 if not stats_A.empty and ticker in stats_A['ticker'].values else 0
        b_p = float(stats_B[stats_B['ticker'] == ticker]['avg_pnl'].iloc[0]) * 100 if not stats_B.empty and ticker in stats_B['ticker'].values else 0
        comp_data.append({'ticker': ticker, 'seed_agree': a_p, 'opt_pnl_wr': b_p})
    comp_df = pd.DataFrame(comp_data).sort_values('seed_agree', ascending=True)
    y_pos = np.arange(len(comp_df))
    bar_h = 0.35
    ax1.barh(y_pos - bar_h/2, comp_df['seed_agree'].values, bar_h,
             label='A: seed_agreement', color='#2196F3', alpha=0.8)
    ax1.barh(y_pos + bar_h/2, comp_df['opt_pnl_wr'].values, bar_h,
             label='B: opt_pnl_x_wr', color='#FF9800', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(comp_df['ticker'].values)
    ax1.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    ax1.set_title('Avg PnL per Trade (%)', fontsize=12)
    ax1.set_xlabel('Avg PnL (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')

ax2 = axes[1]
for trades_df, label, color in [
    (trades_A, 'A: seed_agreement', '#2196F3'),
    (trades_B, 'B: opt_pnl_x_wr', '#FF9800'),
]:
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
chart_path = '/tmp/backtest_composite_compare.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Charts saved: {chart_path}")
try:
    from IPython.display import display, Image
    display(Image(filename=chart_path))
except ImportError:
    plt.show()

# ==============================================================================
# FINAL VERDICT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FINAL VERDICT")
print(f"{'='*W}")
if sA:
    print(f"  A (seed_agreement): {sA['trades']} trades across {sA['tickers']} tickers, "
          f"avg {sA['avg_pnl']:+.2%}/trade, {sA['win_rate']:.0%} WR, "
          f"total {sA['total_pnl']:+.2%}")
if sB:
    print(f"  B (opt_pnl_x_wr):   {sB['trades']} trades across {sB['tickers']} tickers, "
          f"avg {sB['avg_pnl']:+.2%}/trade, {sB['win_rate']:.0%} WR, "
          f"total {sB['total_pnl']:+.2%}")
if sA and sB:
    d_pnl = sA['avg_pnl'] - sB['avg_pnl']
    d_tot = sA['total_pnl'] - sB['total_pnl']
    if d_pnl > 0.001:
        verdict = 'A (seed_agreement) WINS on avg PnL'
    elif d_pnl < -0.001:
        verdict = 'B (opt_pnl_x_wr) WINS on avg PnL'
    else:
        verdict = 'TIE on avg PnL'
    print(f"\n  {verdict}")
    print(f"    dAvg: {d_pnl:+.2%}  |  dTotal: {d_tot:+.2%}  |  Head-to-head: A {a_wins} / B {b_wins} / tie {ties}")

    if d_tot > 0:
        print(f"    A also leads on TOTAL PnL ({sA['total_pnl']:+.2%} vs {sB['total_pnl']:+.2%})")
    elif d_tot < 0:
        print(f"    But B leads on TOTAL PnL ({sB['total_pnl']:+.2%} vs {sA['total_pnl']:+.2%}) due to more tickers")
print(f"{'='*W}")
