"""
analyze_tier1_expanded.py — Do 170/185/200 models populate tier-1 strategies?
==============================================================================
Expands DEPLOY_EXPERIMENTS to include 185 + 200, then runs a 3×3 grid:

  Rows: B (limit, seeds≥3), B2 (all modes, seeds≥3), B5 (limit, seeds≥5)
  Cols: original gate (both 152+160), wide gate (any 2 families), open (no gate)

  B_orig   B_wide   B_open
  B2_orig  B2_wide  B2_open
  B5_orig  B5_wide  B5_open

Plus an upfront EXPERIMENT CENSUS showing what's in BQ per family.
CSV-scored, val_a period configs. Colab-ready.
"""

import os
import io
import gc
import sys
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

# ==============================================================================
# CONFIG — expanded experiment pool
# ==============================================================================
MODEL_VERSION = 'robust_wf_v12'
MIN_TRADES = 3
DEPLOY_THRESHOLD = 0.018

TEST_START = '2026-04-01'
TEST_END   = '2026-06-17'

STARTING_CAPITAL = 200_000

DEPLOY_EXPERIMENTS = {
    '152_XSECTOR', '160_XSECTOR', '170_XSECTOR',
    '175_XSECTOR', '180_XSECTOR', '185_XSECTOR', '200_XSECTOR',
}
EXCLUDE_TICKERS = set()

WALKFORWARD_FOLDS = {
    '175_XSECTOR': ['170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                     '173_XSECTOR', '174_XSECTOR', '175_XSECTOR'],
    '185_XSECTOR': ['180_XSECTOR', '181_XSECTOR', '182_XSECTOR',
                     '183_XSECTOR', '184_XSECTOR', '185_XSECTOR'],
}
WALKFORWARD_BASE = set(WALKFORWARD_FOLDS.keys())

BUCKET_NAME    = "time-series-bucket-ys"
BQ_TABLE       = "ys-time-series.scoring_logs.wf_optimization_results"
INFERENCE_DIR  = "Data/Inference_v2"

EXIT_TIME_MIN = 780
PST = pytz.timezone('America/Los_Angeles')
W = 130

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
SLIPPAGE_PCT = 0.0015
VOL_LOOKBACK_DAYS = 30

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

REGIME_TP_SCALE = np.array([0.70, 0.85, 1.00, 1.30, 1.60])
REGIME_SL_SCALE = np.array([0.60, 0.80, 1.00, 1.30, 1.80])

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
               'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
               'grace': '_GRACE', 'limit_entry': '_LIMIT'}

TRAIN_TEACHERS = [
    'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
    'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
]

FAMILY_LABELS = {
    '152': '152 Late',
    '160': '160 Early',
    '170': '170-175 WF',
    '180': '180-185 WF',
    '185': '185 Single',
    '200': '200 CNN-TX',
}

KEYS = ['B_orig', 'B_wide', 'B_open',
        'B2_orig', 'B2_wide', 'B2_open',
        'B5_orig', 'B5_wide', 'B5_open']

METHOD_LABELS = {
    'B_orig':  'B limit|152+160',
    'B_wide':  'B limit|any2fam',
    'B_open':  'B limit|no gate',
    'B2_orig': 'B2 best|152+160',
    'B2_wide': 'B2 best|any2fam',
    'B2_open': 'B2 best|no gate',
    'B5_orig': 'B5 s≥5|152+160',
    'B5_wide': 'B5 s≥5|any2fam',
    'B5_open': 'B5 s≥5|no gate',
}
METHOD_COLORS = {
    'B_orig':  '#E91E63',
    'B_wide':  '#F06292',
    'B_open':  '#F8BBD0',
    'B2_orig': '#9C27B0',
    'B2_wide': '#BA68C8',
    'B2_open': '#CE93D8',
    'B5_orig': '#009688',
    'B5_wide': '#4DB6AC',
    'B5_open': '#80CBC4',
}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()


# ==============================================================================
# LOAD RAW BQ DATA
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
    df = df[va_opt_ratio <= MAX_VA_OPT_RATIO]
    print(f"    Overfit filter: {before_n} -> {len(df)} configs")

    return df


# ==============================================================================
# EXPERIMENT CENSUS — what's in BQ per family
# ==============================================================================
def run_experiment_census(df):
    print(f"\n{'='*W}")
    print(f"  EXPERIMENT CENSUS — what models populate BQ?")
    print(f"{'='*W}")

    df['exp_fam'] = df['base_experiment'].str[:3]
    indiv = df[df['is_ensemble'] == False].copy()

    fams = sorted(df['exp_fam'].unique())
    print(f"\n  {'Family':<12} {'Rows':>7} {'Seeds':>7} {'Tickers':>8} {'Modes':>6} {'limit_entry':>12} {'Survived≥1.8%':>14}")
    print(f"  {'-'*72}")
    for fam in fams:
        fam_df = indiv[indiv['exp_fam'] == fam]
        fam_label = FAMILY_LABELS.get(fam, fam)
        n_rows = len(fam_df)
        n_seeds = fam_df['seed'].nunique()
        n_tickers = fam_df['ticker'].nunique()
        n_modes = fam_df['tpsl_mode'].nunique()
        n_limit = len(fam_df[fam_df['tpsl_mode'] == 'limit_entry'])
        survived = fam_df[fam_df['val_a_avg_pnl'] >= DEPLOY_THRESHOLD]
        n_survived = survived['ticker'].nunique()
        print(f"  {fam_label:<12} {n_rows:>7} {n_seeds:>7} {n_tickers:>8} {n_modes:>6} {n_limit:>12} {n_survived:>14}")

    print(f"\n  TICKERS PER FAMILY (survived_a + pnl≥1.8%):")
    for fam in fams:
        fam_df = indiv[(indiv['exp_fam'] == fam) & (indiv['val_a_avg_pnl'] >= DEPLOY_THRESHOLD)]
        tickers = sorted(fam_df['ticker'].unique())
        fam_label = FAMILY_LABELS.get(fam, fam)
        print(f"    {fam_label:<12} ({len(tickers):>2}): {', '.join(tickers)}")

    print(f"\n  TICKERS WITH limit_entry CONFIGS (seeds≥3, pnl≥1.8%):")
    for fam in fams:
        fam_limit = indiv[(indiv['exp_fam'] == fam) &
                          (indiv['tpsl_mode'] == 'limit_entry') &
                          (indiv['val_a_avg_pnl'] >= DEPLOY_THRESHOLD)]
        if fam_limit.empty:
            fam_label = FAMILY_LABELS.get(fam, fam)
            print(f"    {fam_label:<12}: (none)")
            continue
        grp = fam_limit.groupby('ticker')['seed'].nunique()
        qualified = grp[grp >= 3]
        fam_label = FAMILY_LABELS.get(fam, fam)
        if qualified.empty:
            print(f"    {fam_label:<12}: (none with ≥3 seeds)")
        else:
            parts = [f"{t}({n}s)" for t, n in sorted(qualified.items())]
            print(f"    {fam_label:<12} ({len(qualified):>2}): {', '.join(parts)}")

    print(f"\n  CROSS-FAMILY COVERAGE (which tickers appear in ≥2 families):")
    ticker_fams = {}
    for fam in fams:
        fam_df = indiv[(indiv['exp_fam'] == fam) & (indiv['val_a_avg_pnl'] >= DEPLOY_THRESHOLD)]
        for t in fam_df['ticker'].unique():
            ticker_fams.setdefault(t, set()).add(fam)
    multi = {t: f for t, f in ticker_fams.items() if len(f) >= 2}
    for t in sorted(multi.keys()):
        fam_list = ', '.join(sorted(multi[t]))
        print(f"    {t:<7} → {fam_list} ({len(multi[t])} families)")
    single = {t: f for t, f in ticker_fams.items() if len(f) == 1}
    if single:
        print(f"\n  SINGLE-FAMILY ONLY (would need open gate):")
        for t in sorted(single.keys()):
            fam = list(single[t])[0]
            print(f"    {t:<7} → {FAMILY_LABELS.get(fam, fam)} only")


# ==============================================================================
# ROW TO CONFIG
# ==============================================================================
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

    model_id = base_exp
    stats_id = base_exp
    for wf_base, wf_folds in WALKFORWARD_FOLDS.items():
        if base_exp in wf_folds:
            model_id = wf_base
            stats_id = wf_base
            break

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
        'model_id': model_id,
        'stats_id': stats_id,
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
# SEED GROUP BUILDER
# ==============================================================================
GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']


def _build_seed_groups(df):
    df_indiv = df[df['is_ensemble'] == False].copy()
    extra_agg = {}
    for c in ['vix_threshold','tlt_crash','spy_ignore']:
        if c in df_indiv.columns:
            extra_agg[c] = (c, 'first')
    sa = df_indiv.groupby(GRP_KEYS).agg(
        n_seeds=('seed', 'nunique'),
        n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= 0.018).sum()),
        n_pos_seeds_15=('val_a_avg_pnl', lambda x: (x >= 0.015).sum()),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_opt_pnl=('opt_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        mean_va_count=('val_a_count', 'mean'),
        mean_va_freq=('val_a_freq', 'mean'),
        mean_opt_wr=('opt_win_rate', 'mean'),
        **extra_agg,
    ).reset_index()
    sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)
    sa['agree_15'] = (sa['n_pos_seeds_15'] / sa['n_seeds']).clip(upper=1.0)
    sa['base_experiment'] = sa['experiment'].apply(
        lambda e: next((e[:-len(s)] for s in sorted(MODE_SUFFIX.values(), key=len, reverse=True)
                         if s and e.endswith(s)), e))
    sa['exp_fam'] = sa['base_experiment'].str[:3]
    return sa


def _sa_to_configs(best_df, agree_col='agree_18', tier_label=''):
    configs = []
    for _, row in best_df.iterrows():
        cfg = _row_to_config(
            row,
            wf_avg_pnl_col='mean_va_pnl',
            wf_wr_col='mean_va_wr',
            wf_count_col='mean_va_count',
            wf_freq_col='mean_va_freq',
            composite_val=float(row.get('score', 0)),
            extra_fields={
                'seed': -1, 'is_ensemble': True,
                'seed_agree_pct': float(row.get(agree_col, 0)),
                'n_seeds': int(row['n_seeds']),
                'tier': tier_label,
            }
        )
        configs.append(cfg)
    return configs


# ==============================================================================
# EXPERIMENT GATE FUNCTIONS
# ==============================================================================
def _tickers_both_152_160(df, threshold=DEPLOY_THRESHOLD):
    indiv = df[df['is_ensemble'] == False].copy()
    indiv['exp_fam'] = indiv['experiment'].str[:3]
    survived = indiv[indiv['val_a_avg_pnl'] >= threshold]
    both = set()
    for tk in survived['ticker'].unique():
        fams = set(survived[survived['ticker'] == tk]['exp_fam'].unique())
        if '152' in fams and '160' in fams:
            both.add(tk)
    return both


def _tickers_any_2_families(df, threshold=DEPLOY_THRESHOLD):
    indiv = df[df['is_ensemble'] == False].copy()
    indiv['exp_fam'] = indiv['experiment'].str[:3]
    survived = indiv[indiv['val_a_avg_pnl'] >= threshold]
    qualified = set()
    for tk in survived['ticker'].unique():
        fams = set(survived[survived['ticker'] == tk]['exp_fam'].unique())
        if len(fams) >= 2:
            qualified.add(tk)
    return qualified


# ==============================================================================
# GENERIC SELECTION FUNCTION
# ==============================================================================
def select_variant(sa, label, mode_filter=None, min_seeds=3,
                   ticker_gate=None, tier_label=''):
    print(f"\n  --- {label} ---")
    filt = sa.copy()
    if min_seeds > 0:
        filt = filt[filt['n_seeds'] >= min_seeds]
    if mode_filter:
        filt = filt[filt['tpsl_mode'] == mode_filter]
    if ticker_gate is not None:
        filt = filt[filt['ticker'].isin(ticker_gate)]

    filt = filt.copy()
    filt['score'] = (filt['mean_va_pnl'] ** 2) * filt['agree_18']
    best = filt.loc[filt.groupby('ticker')['score'].idxmax()] if not filt.empty else filt
    best = best[best['mean_va_pnl'] >= DEPLOY_THRESHOLD].copy()
    configs = _sa_to_configs(best, tier_label=tier_label)

    fam_counts = best['exp_fam'].value_counts() if not best.empty else pd.Series(dtype=int)
    mode_counts = best['tpsl_mode'].value_counts() if not best.empty else pd.Series(dtype=int)

    print(f"    {len(configs)} configs")
    for fam in sorted(fam_counts.index):
        fam_label = FAMILY_LABELS.get(fam, fam)
        print(f"      {fam_label}: {fam_counts[fam]} tickers")
    if len(mode_counts) > 1:
        for m in sorted(mode_counts.index):
            print(f"      {m}: {mode_counts[m]} tickers")
    for c in configs:
        print(f"      {c['ticker']:<7} {c['experiment']:<25} {c['tpsl_mode']:<14} "
              f"pnl={c['wf_avg_pnl']:.2%} agree={c.get('seed_agree_pct',0):.0%} "
              f"({c.get('n_seeds',0)}s)")
    return configs


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
# CSV SCORING
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
        d = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
        if d.index.tz is None:
            d.index = d.index.tz_localize('UTC')
        d.index = d.index.tz_convert(PST)
        seed_series[seed_id] = d['Prob_Spike'].sort_index()
    _csv_score_cache[cache_key] = seed_series
    return seed_series


def load_stitched_seed_scores(ticker, fold_experiments):
    cache_key = ('stitched', ticker, tuple(fold_experiments))
    if cache_key in _csv_score_cache:
        return _csv_score_cache[cache_key]
    all_fold_seeds = {}
    for fold_exp in fold_experiments:
        all_fold_seeds[fold_exp] = load_seed_scores(ticker, fold_exp)
    stitched = {}
    for pos in range(15):
        series_list = []
        for fold_exp in fold_experiments:
            fold_seeds = all_fold_seeds[fold_exp]
            sorted_ids = sorted(fold_seeds.keys())
            if pos < len(sorted_ids):
                series_list.append(fold_seeds[sorted_ids[pos]])
        if series_list:
            combined = pd.concat(series_list).sort_index()
            combined = combined[~combined.index.duplicated(keep='last')]
            stitched[pos] = combined
    _csv_score_cache[cache_key] = stitched
    return stitched


def score_ticker_from_csv(ticker, cfg, df_ohlcv_pst):
    experiment = cfg['model_id']
    entry_minute = cfg['entry_minute']
    prob_thresh = cfg['prob_thresh']

    if experiment in WALKFORWARD_FOLDS:
        all_seeds = load_stitched_seed_scores(ticker, WALKFORWARD_FOLDS[experiment])
    else:
        all_seeds = load_seed_scores(ticker, experiment)
    if not all_seeds:
        return []

    seed_list = sorted(all_seeds.keys()) if cfg.get('is_ensemble', True) else (
        [cfg['seed']] if cfg['seed'] in all_seeds else sorted(all_seeds.keys())
    )

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
        results.append({'bar_ts': idx[i], 'prob': float(probs[i])})
    return results


# ==============================================================================
# DYNAMIC TP/SL, ATR, VOLUME FILTER, TRADE SIM
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


def compute_daily_atr_series(df_ohlcv, period=14):
    daily = df_ohlcv.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([daily['High'] - daily['Low'],
                     (daily['High'] - prev_close).abs(),
                     (daily['Low'] - prev_close).abs()], axis=1).max(axis=1)
    atr_pct = tr.rolling(period).mean() / daily['Close']
    return atr_pct.reindex(df_ohlcv.index, method='ffill')


def compute_adaptive_atr_series(df_ohlcv, short=5, long=22, base=14):
    daily = df_ohlcv.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([daily['High'] - daily['Low'],
                     (daily['High'] - prev_close).abs(),
                     (daily['Low'] - prev_close).abs()], axis=1).max(axis=1)
    atr_base = tr.rolling(base).mean()
    accel = (tr.rolling(short).mean() / tr.rolling(long).mean()).clip(0.5, 3.0)
    atr_pct = (atr_base * accel) / daily['Close']
    return atr_pct.reindex(df_ohlcv.index, method='ffill')


def precompute_vol_baselines(df_ohlcv, lookback_days=VOL_LOOKBACK_DAYS):
    times = (df_ohlcv.index.hour * 60 + df_ohlcv.index.minute).values
    dates = (df_ohlcv.index.year * 10000 + df_ohlcv.index.month * 100 + df_ohlcv.index.day).values.astype(np.int64)
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
            baselines[(d_int, em)] = float(np.median(volumes[at_em[start_j:j]]))
    return baselines


def check_volume_filter(vol_baselines, d_int, entry_minute, volume, vol_filter_pct):
    if vol_filter_pct <= 0:
        return True
    baseline = vol_baselines.get((d_int, entry_minute), 0.0)
    if baseline <= 0:
        return True
    return (volume / baseline) * 100 >= vol_filter_pct


@jit(nopython=True)
def run_single_trade(prices, highs, lows, date_ints, time_mins, entry_idx, tp, sl, exit_time):
    n = len(prices)
    if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], False
    ep = prices[entry_idx]; ed = date_ints[entry_idx]
    tgt = ep * (1 + tp); stp = ep * (1 - sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, True
        sl_h = lows[k] <= stp; tp_h = highs[k] >= tgt
        if sl_h and tp_h: return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_h: return -sl, ed, True
        if tp_h: return tp, ed, True
        if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


@jit(nopython=True)
def run_single_trade_trailing_decay(prices, highs, lows, date_ints, time_mins,
                                     entry_idx, atr_pct, trail_start_mult,
                                     trail_end_mult, sl_mult, exit_time):
    n = len(prices)
    if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], False
    ep = prices[entry_idx]; ed = date_ints[entry_idx]
    if atr_pct != atr_pct or atr_pct <= 0: atr_pct = 0.05
    sl = min(max(atr_pct * sl_mult, 0.01), 0.25)
    hard_stop = ep * (1 - sl)
    ts = atr_pct * trail_start_mult; te = atr_pct * trail_end_mult
    hwm = ep
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, True
        if highs[k] > hwm: hwm = highs[k]
        unr = (hwm - ep) / ep; trailing_stop = 0.0
        if unr > 0 and ts > 0:
            r = min(unr / ts, 1.0)
            trailing_stop = hwm * (1.0 - (ts + (te - ts) * r))
        eff = max(hard_stop, trailing_stop)
        if lows[k] <= eff:
            return ((trailing_stop - ep) / ep if trailing_stop > hard_stop else -sl), ed, True
        if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


@jit(nopython=True)
def run_single_trade_grace(prices, highs, lows, date_ints, time_mins,
                           entry_idx, tp, sl, exit_time, grace_minutes):
    n = len(prices)
    if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], False
    ep = prices[entry_idx]; ed = date_ints[entry_idx]; et = time_mins[entry_idx]
    tgt = ep * (1 + tp); stp = ep * (1 - sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, True
        elapsed = time_mins[k] - et; sl_active = elapsed >= grace_minutes
        tp_h = highs[k] >= tgt; sl_h = lows[k] <= stp
        if sl_active and sl_h and tp_h: return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_active and sl_h: return -sl, ed, True
        if tp_h: return tp, ed, True
        if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


@jit(nopython=True)
def run_single_trade_limit_entry(prices, highs, lows, date_ints, time_mins,
                                  entry_idx, tp, sl, exit_time, dip_pct, max_wait):
    n = len(prices)
    if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], False
    sp = prices[entry_idx]; ed = date_ints[entry_idx]; et = time_mins[entry_idx]
    lp = sp * (1.0 - dip_pct)
    fill_idx = -1
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed: break
        if time_mins[k] - et > max_wait: break
        if lows[k] <= lp: fill_idx = k; break
    if fill_idx < 0: return 0.0, ed, False
    ep = lp; tgt = ep * (1.0 + tp); stp = ep * (1.0 - sl)
    for k in range(fill_idx + 1, n):
        if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, True
        sl_h = lows[k] <= stp; tp_h = highs[k] >= tgt
        if sl_h and tp_h: return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_h: return -sl, ed, True
        if tp_h: return tp, ed, True
        if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


# ==============================================================================
# REGIME FILTER
# ==============================================================================
def preload_regime_data(eval_start, eval_end):
    print("  Loading regime data (VIX, SPY, TLT)...")
    def _load(ticker):
        clean = TICKER_FILE_MAP.get(ticker, ticker)
        uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
        try:
            d = pd.read_parquet(uri).reset_index()
            dt_cols = [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]
            if dt_cols: d['Datetime'] = pd.to_datetime(d[dt_cols[0]], utc=True)
            else:
                nc = next((c for c in d.columns if c in ('Datetime','Date','Time','Timestamp')), d.columns[0])
                d['Datetime'] = pd.to_datetime(d[nc], unit='ms', utc=True)
            d = d.set_index('Datetime').sort_index()
            d = d[~d.index.duplicated(keep='first')]
            d.index = d.index.tz_convert('America/Los_Angeles')
            return d
        except Exception:
            return None

    vix_df = _load('VIX'); spy_df = _load('SPY'); tlt_df = _load('TLT')
    spy_daily = spy_df['Close'].resample('D').last().dropna() if spy_df is not None else pd.Series(dtype=float)
    tlt_daily = tlt_df['Close'].resample('D').last().dropna() if tlt_df is not None else pd.Series(dtype=float)

    start = pd.Timestamp(eval_start, tz=PST) - timedelta(days=10)
    end = pd.Timestamp(eval_end, tz=PST) + timedelta(days=1)

    def _lk(d):
        if d is None: return {}
        sub = d[(d.index >= start) & (d.index <= end)]
        di = (sub.index.year*10000 + sub.index.month*100 + sub.index.day).astype(int)
        mi = (sub.index.hour*60 + sub.index.minute).astype(int)
        return {(int(di[i]), int(mi[i])): float(sub['Close'].values[i]) for i in range(len(sub))}

    vix_lk = _lk(vix_df); spy_lk = _lk(spy_df); tlt_lk = _lk(tlt_df)

    def _med(lk, d_int, em, w=30):
        vals = [lk[(d_int, m)] for m in range(em-w+1, em+1) if (d_int, m) in lk]
        return float(np.median(vals)) if vals else None

    def _pc(daily_s):
        m = {}
        for i in range(1, len(daily_s)):
            m[int(daily_s.index[i].strftime('%Y%m%d'))] = float(daily_s.iloc[i-1])
        return m

    spy_prev = _pc(spy_daily); tlt_prev = _pc(tlt_daily)
    regime_dict = {}
    for d in pd.date_range(start, end, freq='D', tz=PST):
        d_int = int(d.strftime('%Y%m%d'))
        for em in range(360, 391):
            v = _med(vix_lk, d_int, em) or 15.0
            sc = _med(spy_lk, d_int, em); sp = spy_prev.get(d_int)
            s_gap = ((sc - sp) / sp * 100) if sc and sp and sp != 0 else 0.0
            tc = _med(tlt_lk, d_int, em); tp = tlt_prev.get(d_int)
            t_gap = ((tc - tp) / tp * 100) if tc and tp and tp != 0 else 0.0
            regime_dict[(d_int, em)] = np.array([v, t_gap, s_gap])
    print(f"    Regime data: {len(regime_dict)} combos")
    return regime_dict


# ==============================================================================
# OHLCV ARRAYS
# ==============================================================================
def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date):
    s = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= s) & (df_ohlcv.index < e)]
    if len(df) == 0: return None
    return {
        'index': df.index, 'prices': df['Close'].values.copy(),
        'highs': df['High'].values.copy(), 'lows': df['Low'].values.copy(),
        'volumes': df['Volume'].values.copy(),
        'dates': (df.index.year*10000 + df.index.month*100 + df.index.day).values.astype(np.int64),
        'times': (df.index.hour*60 + df.index.minute).values.astype(np.int64),
    }


# ==============================================================================
# MAIN BACKTEST
# ==============================================================================
def run_backtest(configs, label, regime_dict):
    print(f"\n{'='*W}")
    print(f"  BACKTEST: {label} -- {len(configs)} tickers")
    print(f"{'='*W}")
    if not configs:
        print(f"  No configs -- skipping"); return pd.DataFrame()

    all_trades = []
    for cfg in configs:
        ticker = cfg['ticker']; mode = cfg['tpsl_mode']; em = cfg['entry_minute']
        tier_tag = cfg.get('tier', '')
        df_raw = load_master(ticker, start_date=TEST_START)
        if df_raw is None: print(f"  {ticker}: SKIP"); continue
        df_raw = heal_data(df_raw.sort_index())
        df_pst = df_raw.copy(); df_pst.index = df_pst.index.tz_convert(PST)

        signals = score_ticker_from_csv(ticker, cfg, df_pst)
        if not signals:
            print(f"  {ticker}: No signals"); del df_raw; gc.collect(); continue

        atr_series = (compute_daily_atr_series(df_pst) if mode in ('atr','trailing_decay','atr_sl')
                      else compute_adaptive_atr_series(df_pst) if mode == 'atr_adaptive' else None)
        vol_baselines = precompute_vol_baselines(df_pst)
        ohlcv = precompute_ohlcv_arrays(df_pst, TEST_START, TEST_END)
        if ohlcv is None: del df_raw; gc.collect(); continue

        trades = []; fv = 0; fr = 0; last_d = -1; uf = 0
        for sig in signals:
            bts = sig['bar_ts'].tz_convert(PST) if sig['bar_ts'].tzinfo else sig['bar_ts']
            d_int = bts.year*10000 + bts.month*100 + bts.day
            if d_int == last_d: continue

            oi = None
            for j in range(len(ohlcv['index'])):
                if ohlcv['dates'][j] == d_int and ohlcv['times'][j] == em:
                    oi = j; break
            if oi is not None:
                if not check_volume_filter(vol_baselines, d_int, em,
                                           float(ohlcv['volumes'][oi]), cfg['vol_filter_pct']):
                    fv += 1; continue

            rk = (d_int, em); vix_val = regime_dict[rk][0] if rk in regime_dict else None
            atr_pct = 0.05
            if atr_series is not None:
                ts = pd.Timestamp(bts)
                if ts in atr_series.index:
                    v = float(atr_series.loc[ts])
                    if not np.isnan(v) and v > 0: atr_pct = v
                else:
                    prior = atr_series.index[atr_series.index <= ts]
                    if len(prior) > 0:
                        v = float(atr_series.iloc[atr_series.index.get_loc(prior[-1])])
                        if not np.isnan(v) and v > 0: atr_pct = v

            tp, sl = compute_dynamic_tpsl(cfg, atr_pct, vix_val)
            if oi is None: continue

            if rk in regime_dict:
                rv = regime_dict[rk]
                if rv[0] > cfg.get('vix_threshold', 99) and rv[1] < cfg.get('tlt_crash', -99) and rv[2] > cfg.get('spy_ignore', 99):
                    fr += 1; continue

            if mode == 'trailing_decay':
                pnl, td, valid = run_single_trade_trailing_decay(ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'], oi, atr_pct, cfg['trail_start_mult'], cfg['trail_end_mult'], cfg['sl_mult'], EXIT_TIME_MIN)
            elif mode == 'grace':
                pnl, td, valid = run_single_trade_grace(ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'], oi, tp, sl, EXIT_TIME_MIN, cfg['grace_minutes'])
            elif mode == 'limit_entry':
                pnl, td, valid = run_single_trade_limit_entry(ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'], oi, tp, sl, EXIT_TIME_MIN, cfg['dip_pct'], cfg['max_wait_minutes'])
                if not valid: uf += 1; continue
            else:
                pnl, td, valid = run_single_trade(ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'], oi, tp, sl, EXIT_TIME_MIN)

            if valid:
                pnl -= 2 * SLIPPAGE_PCT
                trades.append({'ticker': ticker, 'experiment': cfg['experiment'],
                               'date_int': int(td), 'pnl': float(pnl), 'prob': sig['prob'],
                               'tp': tp, 'sl': sl, 'mode': mode, 'tier': tier_tag,
                               'dip_pct': cfg.get('dip_pct', 0), 'max_wait': cfg.get('max_wait_minutes', 0)})
                last_d = int(td)

        w = sum(1 for t in trades if t['pnl'] > 0); l = len(trades) - w
        avg = np.mean([t['pnl'] for t in trades]) if trades else 0
        extra = ""
        if fv > 0: extra += f" vol_f={fv}"
        if fr > 0: extra += f" reg_f={fr}"
        if uf > 0: extra += f" unfilled={uf}"
        print(f"  {ticker:<7} {cfg['experiment']:<25} ENS  {len(signals):>3} sig -> {len(trades):>3} trades "
              f"({w}W/{l}L)  avg {avg:>+6.2%}  {mode}  dip={cfg.get('dip_pct',0):.1%}{extra}")
        all_trades.extend(trades)
        del df_raw, df_pst, ohlcv; gc.collect()

    if not all_trades: print(f"  No trades for {label}"); return pd.DataFrame()
    tdf = pd.DataFrame(all_trades)
    tdf['date'] = pd.to_datetime(tdf['date_int'].astype(str), format='%Y%m%d')
    return tdf


# ==============================================================================
# ANALYTICS
# ==============================================================================
def per_asset_stats(trades_df):
    if trades_df.empty: return pd.DataFrame()
    months = max(1.0, (trades_df['date'].max() - trades_df['date'].min()).days / 30.44)
    rows = []
    for ticker in sorted(trades_df['ticker'].unique()):
        td = trades_df[trades_df['ticker'] == ticker]
        n = len(td); w = int((td['pnl'] > 0).sum())
        rows.append({'ticker': ticker, 'experiment': td['experiment'].iloc[0],
                     'mode': td['mode'].iloc[0], 'tier': td['tier'].iloc[0],
                     'trades': n, 'wins': w, 'losses': n-w,
                     'win_rate': w/n, 'avg_pnl': td['pnl'].mean(),
                     'total_pnl': td['pnl'].sum(), 'freq': n/months})
    return pd.DataFrame(rows)


def portfolio_summary(trades_df, label):
    if trades_df.empty: return {}
    n = len(trades_df); w = int((trades_df['pnl'] > 0).sum())
    months = max(1.0, (trades_df['date'].max() - trades_df['date'].min()).days / 30.44)
    return {'label': label, 'tickers': trades_df['ticker'].nunique(), 'trades': n,
            'wins': w, 'losses': n-w, 'win_rate': w/n, 'avg_pnl': trades_df['pnl'].mean(),
            'total_pnl': trades_df['pnl'].sum(), 'freq_per_mo': n/months}


# ==============================================================================
# RUN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TIER-1 EXPANDED EXPERIMENT ANALYSIS")
print(f"  TEST: {TEST_START} -> {TEST_END}  |  Model: {MODEL_VERSION}  |  Threshold: {DEPLOY_THRESHOLD:.1%}")
print(f"  EXPERIMENTS: {sorted(DEPLOY_EXPERIMENTS)}")
print(f"{'='*W}")

raw_df = load_raw_wf_data()

# --- Experiment census FIRST ---
run_experiment_census(raw_df)

sa = _build_seed_groups(raw_df)

# --- Compute experiment gates ---
gate_both_152_160 = _tickers_both_152_160(raw_df)
gate_any_2_families = _tickers_any_2_families(raw_df)

print(f"\n  EXPERIMENT GATES:")
print(f"    Both 152+160:   {len(gate_both_152_160):>2} tickers — {sorted(gate_both_152_160)}")
print(f"    Any 2 families: {len(gate_any_2_families):>2} tickers — {sorted(gate_any_2_families)}")

# --- Build 3x3 strategy grid ---
# Row 1: B (limit_entry, seeds≥3)
configs = {}
configs['B_orig'] = select_variant(
    sa, "B_orig: limit, seeds≥3, both 152+160",
    mode_filter='limit_entry', min_seeds=3,
    ticker_gate=gate_both_152_160, tier_label='B_orig')

configs['B_wide'] = select_variant(
    sa, "B_wide: limit, seeds≥3, any 2 families",
    mode_filter='limit_entry', min_seeds=3,
    ticker_gate=gate_any_2_families, tier_label='B_wide')

configs['B_open'] = select_variant(
    sa, "B_open: limit, seeds≥3, NO experiment gate",
    mode_filter='limit_entry', min_seeds=3,
    ticker_gate=None, tier_label='B_open')

# Row 2: B2 (all modes, seeds≥3)
configs['B2_orig'] = select_variant(
    sa, "B2_orig: all modes, seeds≥3, both 152+160",
    mode_filter=None, min_seeds=3,
    ticker_gate=gate_both_152_160, tier_label='B2_orig')

configs['B2_wide'] = select_variant(
    sa, "B2_wide: all modes, seeds≥3, any 2 families",
    mode_filter=None, min_seeds=3,
    ticker_gate=gate_any_2_families, tier_label='B2_wide')

configs['B2_open'] = select_variant(
    sa, "B2_open: all modes, seeds≥3, NO experiment gate",
    mode_filter=None, min_seeds=3,
    ticker_gate=None, tier_label='B2_open')

# Row 3: B5 (limit_entry, seeds≥5)
configs['B5_orig'] = select_variant(
    sa, "B5_orig: limit, seeds≥5, both 152+160",
    mode_filter='limit_entry', min_seeds=5,
    ticker_gate=gate_both_152_160, tier_label='B5_orig')

configs['B5_wide'] = select_variant(
    sa, "B5_wide: limit, seeds≥5, any 2 families",
    mode_filter='limit_entry', min_seeds=5,
    ticker_gate=gate_any_2_families, tier_label='B5_wide')

configs['B5_open'] = select_variant(
    sa, "B5_open: limit, seeds≥5, NO experiment gate",
    mode_filter='limit_entry', min_seeds=5,
    ticker_gate=None, tier_label='B5_open')

# --- Run backtests ---
regime_dict = preload_regime_data(TEST_START, TEST_END)

results = {}
stats = {}
summaries = {}

for key in KEYS:
    label = f"{key}: {METHOD_LABELS[key]}"
    trades = run_backtest(configs[key], label, regime_dict)
    results[key] = trades
    stats[key] = per_asset_stats(trades)
    summaries[key] = portfolio_summary(trades, METHOD_LABELS[key])


# ==============================================================================
# 3x3 GRID SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  3×3 STRATEGY GRID — do new experiments populate?")
print(f"{'='*W}")

grid_rows = [('B',  'limit, s≥3'), ('B2', 'best, s≥3'), ('B5', 'limit, s≥5')]
grid_cols = ['orig', 'wide', 'open']
col_labels = ['152+160 only', 'any 2 families', 'no gate']

for metric_name, metric_key, fmt in [
    ('Avg PnL', 'avg_pnl', '+.2%'),
    ('Total PnL', 'total_pnl', '+.1%'),
    ('Tickers', 'tickers', 'd'),
    ('Trades', 'trades', 'd'),
    ('Win Rate', 'win_rate', '.0%'),
]:
    print(f"\n  {metric_name}:")
    print(f"  {'Strategy':<14} {'152+160':>14} {'any 2 fam':>14} {'no gate':>14}")
    print(f"  {'-'*58}")
    for row_key, row_label in grid_rows:
        print(f"  {row_label:<14}", end='')
        for col in grid_cols:
            full_key = f"{row_key}_{col}"
            s = summaries.get(full_key, {})
            if s:
                v = s[metric_key]
                formatted = format(v, fmt)
                print(f" {formatted:>13}", end='')
            else:
                print(f" {'---':>13}", end='')
        print()


# ==============================================================================
# EXPERIMENT FAMILY BREAKDOWN — which models got selected?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  EXPERIMENT FAMILY BREAKDOWN — which models populate each strategy?")
print(f"{'='*W}")

for key in KEYS:
    s = stats[key]
    if s.empty:
        print(f"\n  {key}: (no trades)")
        continue
    fam_summary = {}
    for _, row in s.iterrows():
        exp = row['experiment']
        fam = exp[:3]
        fam_summary.setdefault(fam, []).append(row)
    print(f"\n  {key} ({METHOD_LABELS[key]}):")
    for fam in sorted(fam_summary.keys()):
        fam_label = FAMILY_LABELS.get(fam, fam)
        rows = fam_summary[fam]
        tickers = [r['ticker'] for r in rows]
        avg_pnl = np.mean([r['avg_pnl'] for r in rows])
        n_trades = sum(r['trades'] for r in rows)
        print(f"    {fam_label:<14} {len(tickers)} tickers, {n_trades} trades, avg {avg_pnl:+.2%} — {', '.join(tickers)}")


# ==============================================================================
# NEW TICKERS FROM 170/185/200
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  NEW TICKERS — what do 170/185/200 add vs 152+160 baseline?")
print(f"{'='*W}")

for row_key, row_label in grid_rows:
    orig_key = f"{row_key}_orig"
    orig_tickers = set(stats[orig_key]['ticker'].tolist()) if not stats[orig_key].empty else set()

    for col in ['wide', 'open']:
        full_key = f"{row_key}_{col}"
        s = stats[full_key]
        if s.empty: continue
        new_tickers = set(s['ticker'].tolist()) - orig_tickers
        if not new_tickers:
            continue
        print(f"\n  {full_key} vs {orig_key}:")
        for t in sorted(new_tickers):
            t_row = s[s['ticker'] == t].iloc[0]
            print(f"    + {t:<7} {t_row['experiment']:<25} {t_row['mode']:<14} "
                  f"avg={t_row['avg_pnl']:+.2%} N={t_row['trades']}")


# ==============================================================================
# HEAD-TO-HEAD
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  HEAD-TO-HEAD: avg PnL per ticker")
print(f"{'='*W}")
all_tickers = sorted(set().union(*[set(stats[k]['ticker'].tolist()) for k in stats if not stats[k].empty]))

short_keys = ['B_orig', 'B_wide', 'B_open', 'B2_open', 'B5_open']
short_labels = ['B', 'B_w', 'B_o', 'B2_o', 'B5_o']

print(f"  {'Ticker':<7}", end='')
for lbl in short_labels:
    print(f" {lbl:>7}", end='')
print(f" {'Best':>6}")
print(f"  {'-'*(7 + 8*len(short_keys) + 7)}")

for ticker in all_tickers:
    print(f"  {ticker:<7}", end='')
    best_key = None; best_pnl = -999
    for key, lbl in zip(short_keys, short_labels):
        s = stats[key]
        match = s[s['ticker'] == ticker] if not s.empty else pd.DataFrame()
        if not match.empty:
            v = float(match.iloc[0]['avg_pnl'])
            print(f" {v:>+6.2%}", end='')
            if v > best_pnl: best_pnl = v; best_key = lbl
        else:
            print(f" {'---':>7}", end='')
    print(f" {best_key or '':>6}")


# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n  Generating charts...")
fig, axes = plt.subplots(2, 1, figsize=(18, 16))
fig.suptitle(f'Tier-1 Expanded Analysis — Do 170/185/200 Populate?\n'
             f'TEST ({TEST_START} -> {TEST_END})', fontsize=14, fontweight='bold')

# Chart 1: 3x3 grid bar chart (avg PnL + tickers)
ax1 = axes[0]
x_pos = np.arange(3)
bar_w = 0.25
for i, (row_key, row_label) in enumerate(grid_rows):
    vals = []
    labels_for = []
    for col in grid_cols:
        full_key = f"{row_key}_{col}"
        s = summaries.get(full_key, {})
        vals.append(s.get('avg_pnl', 0) * 100)
        labels_for.append(f"{s.get('tickers', 0)}tk")
    color = {'B': '#E91E63', 'B2': '#9C27B0', 'B5': '#009688'}[row_key]
    bars = ax1.bar(x_pos + (i-1)*bar_w, vals, bar_w, label=row_label, color=color, alpha=0.85)
    for bar, lbl in zip(bars, labels_for):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 lbl, ha='center', va='bottom', fontsize=8)
ax1.set_xticks(x_pos); ax1.set_xticklabels(col_labels)
ax1.axhline(y=DEPLOY_THRESHOLD*100, color='red', linewidth=1, linestyle='--', alpha=0.5, label=f'Threshold {DEPLOY_THRESHOLD:.1%}')
ax1.set_title('3×3 Strategy Grid: Avg PnL per Trade (%)', fontsize=12)
ax1.set_ylabel('Avg PnL (%)'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis='y')

# Chart 2: equity curves (subset for readability)
ax2 = axes[1]
plot_keys = ['B_orig', 'B_wide', 'B_open', 'B2_orig', 'B2_open', 'B5_orig', 'B5_open']
for key in plot_keys:
    tdf = results[key]
    if tdf.empty: continue
    daily = tdf.groupby('date')['pnl'].mean()
    eq = STARTING_CAPITAL
    pts = [{'date': daily.index[0] - pd.Timedelta(days=1), 'equity': eq}]
    for d, p in daily.items():
        n_t = len(tdf[tdf['date'] == d])
        eq += (eq / max(n_t, 1)) * p * n_t
        pts.append({'date': d, 'equity': eq})
    edf = pd.DataFrame(pts).set_index('date')
    ls = '-' if key.endswith('orig') else ('--' if key.endswith('wide') else ':')
    lw = 2.5 if key == 'B_orig' else 1.5
    ax2.plot(edf.index, edf['equity']/1000, label=f'{key}',
             color=METHOD_COLORS[key], linewidth=lw, linestyle=ls)
ax2.axhline(y=STARTING_CAPITAL/1000, color='gray', linewidth=0.5, linestyle='--')
ax2.set_title(f'Cumulative Equity (${STARTING_CAPITAL/1000:.0f}K start)', fontsize=12)
ax2.set_ylabel('Equity ($K)')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax2.tick_params(axis='x', rotation=45); ax2.legend(fontsize=8, ncol=2); ax2.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = '/tmp/tier1_expanded_analysis.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Charts saved: {chart_path}")
try:
    from IPython.display import display, Image
    display(Image(filename=chart_path))
except ImportError:
    plt.show()


# ==============================================================================
# FINAL RANKING
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FINAL RANKING")
print(f"{'='*W}")
ranking = []
for key in KEYS:
    s = summaries[key]
    if s:
        ranking.append({'key': key, 'label': METHOD_LABELS[key], **s})
ranking_df = pd.DataFrame(ranking).sort_values('avg_pnl', ascending=False)
print(f"\n  {'#':>2} {'Method':<22} {'Tickers':>8} {'Trades':>7} {'WR':>6} "
      f"{'AvgPnL':>8} {'TotPnL':>9} {'Freq/mo':>8} {'≥1.8%':>6}")
print(f"  {'-'*86}")
for i, (_, r) in enumerate(ranking_df.iterrows(), 1):
    above = '  Y' if r['avg_pnl'] >= DEPLOY_THRESHOLD else ''
    print(f"  {i:>2} {r['key']:<22} {r['tickers']:>8} {r['trades']:>7} "
          f"{r['win_rate']:>5.0%} {r['avg_pnl']:>+7.2%} {r['total_pnl']:>+8.2%} "
          f"{r['freq_per_mo']:>7.1f}{above}")

best = ranking_df.iloc[0]
print(f"\n  WINNER: {best['key']} — {best['tickers']} tickers, "
      f"avg {best['avg_pnl']:+.2%}/trade, {best['win_rate']:.0%} WR, total {best['total_pnl']:+.2%}")

# --- Key question answer ---
print(f"\n\n{'='*W}")
print(f"  ANSWER: Do 170/185/200 populate tier-1 strategies?")
print(f"{'='*W}")
for key in KEYS:
    s = stats[key]
    if s.empty: continue
    fam_set = set()
    for _, row in s.iterrows():
        fam_set.add(row['experiment'][:3])
    new_fams = fam_set - {'152', '160'}
    if new_fams:
        new_tickers = []
        for _, row in s.iterrows():
            if row['experiment'][:3] in new_fams:
                new_tickers.append(f"{row['ticker']}({row['experiment'][:3]})")
        print(f"  {key:<12} YES — {', '.join(sorted(new_tickers))}")
    else:
        print(f"  {key:<12} No — only 152/160 configs selected")
print(f"{'='*W}")
