"""
backtest_strategy_comparison.py — Live TF Strategy Comparison
=============================================================
Compares 5 strategies with live model inference using limit_entry trade simulation:

1. S3 Deployed (benchmark): BQ family-agreement gate (dynamic ticker selection),
   ONE experiment per ticker, limit_entry trade sim (matches production)
2. Composite seed=50%: All tickers, per-experiment prob_thresh, seed=50%/15 gate
3. S3 + GATE lb=10: S3 + rolling performance gate (lookback 10)
4. S3 + GATE lb=7: S3 + rolling performance gate (lookback 7)
5. Composite + GATE lb=10: seed=50% + rolling performance gate

BQ configs loaded from individual seeds, aggregated into seed groups,
filtered to limit_entry mode. Family agreement gate determines S3 tickers.
Trade sim uses limit_entry: limit order below signal price, unfilled = skipped.
Colab-ready (no if __name__).
"""

import os, io, gc, sys, subprocess, warnings, ctypes
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

try:
    import pandas_ta as _ta_check
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas_ta", "--no-deps", "--quiet",
                           "--break-system-packages"])
    import pandas_ta as ta

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
except ImportError:
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from google.cloud import storage as gcs_storage, bigquery

if _IN_COLAB:
    try:
        auth.authenticate_user()
    except Exception:
        pass

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
RESULTS_DIR = "Algos/Results_scored_20260703"
RESULTS_FALLBACK = "Algos/Results_all_v2"
INFERENCE_DIR = "Data/Inference_v2"

TEST_START = '2026-04-01'
TEST_END   = '2026-07-03'
ENTRY_MIN  = 375
EXIT_TIME_MIN = 780
TIME_STEPS = 240
EMBED_DIM  = 16
SLIPPAGE_PCT = 0.0015
STARTING_CAPITAL = 200_000
DEPLOY_THRESHOLD = 0.018
REOPT_FREQ = 5

PST = pytz.timezone('America/Los_Angeles')
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
CONTEXT_COLS += ['Time_Pos']
ALL_FEATURE_COLS = ASSET_COLS + CONTEXT_COLS

ALL_TICKERS = []  # populated dynamically from BQ after sa_limit is built

MIN_SEEDS = 3
MIN_TRADES_BQ = 3
MAX_VA_OPT_RATIO = 3.0

GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']

TRAIN_TEACHERS_13 = ['BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
                     'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD']
TEACHERS_31 = ['CIFR', 'IREN', 'KEEL', 'HUT', 'APLD', 'WULF', 'RIOT', 'CLSK',
               'BTBT', 'ALAB', 'MARA', 'RCAT', 'ASTS', 'LUNR', 'BBAI', 'RKLB',
               'SPIR', 'BKSY', 'MNTS', 'PLUG', 'BE', 'FCEL', 'FLNC', 'OKLO',
               'SMR', 'MVIS', 'BGDE', 'RDW', 'PL', 'VOYG', 'SMLR']

SCORE_EXPERIMENTS = []  # built dynamically after BQ loading
EXP_CONFIGS = {}       # built dynamically after BQ loading


def _build_exp_config(bq_experiment):
    base = bq_experiment.replace('_LIMIT', '')
    base_num = int(base.split('_')[0])
    if base_num in (152, 160) or 170 <= base_num <= 175:
        teachers = TRAIN_TEACHERS_13
    elif 180 <= base_num <= 185 or base_num == 200:
        teachers = TEACHERS_31
    else:
        return None
    cfg = {
        'train_teachers': teachers,
        'n_assets': len(teachers) + 1,
        'stats_id': base,
        'model_id_override': base,
        'seed_base': base_num,
    }
    if base_num == 200:
        cfg['model_builder'] = 'cnn_transformer'
        cfg['embed_dim'] = 32
    idx = {t: i for i, t in enumerate(teachers)}
    if 'BITF' in idx: idx['KEEL'] = idx['BITF']
    if 'MIGI' in idx: idx['BGDE'] = idx['MIGI']
    cfg['asset_index'] = idx
    cfg['other_idx'] = len(teachers)
    return cfg

gcs_client = gcs_storage.Client()
bq_client = bigquery.Client()
W = 120


# ==============================================================================
# DATA LOADING (OHLCV for trade sim + pre-scored CSVs from GCS)
# ==============================================================================
_master_cache = {}


def load_master(ticker):
    if ticker in _master_cache:
        return _master_cache[ticker]
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
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume',
        }
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                      for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        df = df[df.index.notna()].sort_index()
        cutoff = pd.Timestamp(TEST_START, tz='UTC') - pd.Timedelta(days=5)
        df = df[df.index >= cutoff]
        _master_cache[ticker] = df
        return df
    except Exception as e:
        print(f"    load_master({ticker}): FAILED -- {e}")
        return None


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def load_seed_scores(ticker, experiment_base):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    bucket = gcs_client.bucket(BUCKET_NAME)
    all_results = {}
    for results_dir in [RESULTS_FALLBACK, RESULTS_DIR]:
        prefix = f"{results_dir}/Universal_Exp{experiment_base}/{clean}/Seed_"
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            continue
        for blob in blobs:
            name = blob.name.split('/')[-1]
            if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
                continue
            seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
            d = pd.read_csv(io.BytesIO(blob.download_as_bytes()),
                            index_col=0, parse_dates=True)
            if d.index.tz is None:
                d.index = d.index.tz_localize('UTC')
            d.index = d.index.tz_convert(PST)
            d = d.sort_index()
            if seed_id in all_results:
                old = all_results[seed_id]
                combined = pd.concat([old, d])
                combined = combined[~combined.index.duplicated(keep='last')]
                all_results[seed_id] = combined.sort_index()
            else:
                all_results[seed_id] = d
    return all_results


# ==============================================================================
# TRADE SIMULATION — limit_entry mode
# ==============================================================================
def precompute_ohlcv_arrays(df, start_date, end_date):
    s = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    idx = df.index
    if str(idx.tz) != 'America/Los_Angeles':
        idx = idx.tz_convert('America/Los_Angeles')
    mask = (idx >= s) & (idx < e)
    df_cut = df[mask]
    if len(df_cut) == 0:
        return None
    idx_cut = df_cut.index if str(df_cut.index.tz) == 'America/Los_Angeles' \
        else df_cut.index.tz_convert('America/Los_Angeles')
    return {
        'prices': df_cut['Close'].values.astype(np.float64).copy(),
        'highs':  df_cut['High'].values.astype(np.float64).copy(),
        'lows':   df_cut['Low'].values.astype(np.float64).copy(),
        'dates':  (idx_cut.year * 10000 + idx_cut.month * 100 + idx_cut.day).values.astype(np.int64),
        'times':  (idx_cut.hour * 60 + idx_cut.minute).values.astype(np.int64),
    }


def run_single_trade_limit_entry(prices, highs, lows, date_ints, time_mins,
                                  entry_idx, tp, sl, exit_time, dip_pct, max_wait):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    sp = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    et = time_mins[entry_idx]
    lp = sp * (1.0 - dip_pct)
    fill_idx = -1
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            break
        if time_mins[k] - et > max_wait:
            break
        if lows[k] <= lp:
            fill_idx = k
            break
    if fill_idx < 0:
        return 0.0, ed, False
    ep = lp
    tgt = ep * (1.0 + tp)
    stp = ep * (1.0 - sl)
    for k in range(fill_idx + 1, n):
        if date_ints[k] != ed:
            return (prices[k - 1] - ep) / ep, ed, True
        sl_h = lows[k] <= stp
        tp_h = highs[k] >= tgt
        if sl_h and tp_h:
            return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_h:
            return -sl, ed, True
        if tp_h:
            return tp, ed, True
        if time_mins[k] >= exit_time:
            return (prices[k] - ep) / ep, ed, True
    return (prices[n - 1] - ep) / ep, ed, True


def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    if num == 200: return '200'
    return 'other'


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
print("=" * W)
print(f"  LIVE TF STRATEGY COMPARISON — 5 STRATEGIES × 11 TICKERS")
print("=" * W)


# ── Step 1: Load BQ configs (individual seeds → seed groups → family agreement) ──
print(f"\n  STEP 1: Loading BQ configs (limit_entry, individual seeds)...")
q = """
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult, grace_minutes,
       dip_pct, max_wait_minutes, vol_filter_pct,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, opt_avg_pnl, is_ensemble
FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
WHERE model_version IN ('robust_wf_v12', 'robust_wf_v12_stitched')
  AND survived_a = TRUE
  AND val_a_count >= 3
  AND val_a_avg_pnl > 0
"""
raw_df = bq_client.query(q).to_dataframe()
print(f"    {len(raw_df)} raw BQ rows loaded")

# Overfit filter
raw_df = raw_df[raw_df['opt_avg_pnl'] > 0].copy()
raw_df = raw_df[raw_df['val_a_avg_pnl'] / raw_df['opt_avg_pnl'] <= MAX_VA_OPT_RATIO]
print(f"    {len(raw_df)} after overfit filter")

# Individual seeds only
indiv = raw_df[raw_df['is_ensemble'] == False].copy()
for k in GRP_KEYS:
    if k not in indiv.columns:
        indiv[k] = 0

# Seed group aggregation
sa = indiv.groupby(GRP_KEYS).agg(
    n_seeds=('seed', 'nunique'),
    n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= DEPLOY_THRESHOLD).sum()),
    mean_va_pnl=('val_a_avg_pnl', 'mean'),
    mean_va_wr=('val_a_win_rate', 'mean'),
    mean_va_count=('val_a_count', 'mean'),
    mean_va_freq=('val_a_freq', 'mean'),
    vb_pnl=('val_b_avg_pnl', 'mean'),
    vix_threshold=('val_a_avg_pnl', 'count'),  # placeholder, not used
).reset_index()
sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)
sa.drop(columns=['vix_threshold'], inplace=True)

# Filter: n_seeds >= 3, limit_entry only
sa = sa[sa['n_seeds'] >= MIN_SEEDS].copy()
sa_limit = sa[sa['tpsl_mode'] == 'limit_entry'].copy()
print(f"    {len(sa_limit)} limit_entry seed groups (n_seeds >= {MIN_SEEDS})")

# Score: (mean_va_pnl ** 2) * agree_18
sa_limit['score'] = (sa_limit['mean_va_pnl'] ** 2) * sa_limit['agree_18']

# Family assignment
sa_limit['exp_fam'] = sa_limit['experiment'].apply(assign_family)
sa_fam = sa_limit[sa_limit['exp_fam'] != 'other'].copy()

# Best per (ticker, family), filter mean_va_pnl >= threshold
best_per_fam = sa_fam.loc[sa_fam.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
best_per_fam = best_per_fam[best_per_fam['mean_va_pnl'] >= DEPLOY_THRESHOLD]

# Family agreement gate: majority of families with positive vb_pnl, n_fams >= 2
agreement = {}
for tk in best_per_fam['ticker'].unique():
    tk_rows = best_per_fam[best_per_fam['ticker'] == tk]
    n_fams = len(tk_rows)
    n_pos = int((tk_rows['vb_pnl'] > 0).sum())
    agreement[tk] = {
        'n_fams': n_fams, 'n_pos': n_pos,
        'majority_pos': n_pos > n_fams / 2 and n_fams >= 2,
    }

s3_tickers = sorted([tk for tk, ag in agreement.items() if ag['majority_pos']])
print(f"    Family agreement: {len(s3_tickers)} tickers pass S3 gate")
print(f"    S3 tickers: {', '.join(s3_tickers)}")

# Derive ALL_TICKERS dynamically: any ticker with a qualifying limit_entry config
_qualifying = sa_limit[sa_limit['mean_va_pnl'] >= DEPLOY_THRESHOLD]['ticker'].unique()
ALL_TICKERS = sorted(set(_qualifying) | set(s3_tickers))
print(f"    Dynamic ALL_TICKERS: {len(ALL_TICKERS)} tickers")

# Best config per ticker (from S3-passing tickers) — for S3 experiment selection + trade sim
sa_s3 = sa_limit[sa_limit['ticker'].isin(s3_tickers)].copy()
sa_s3['score'] = (sa_s3['mean_va_pnl'] ** 2) * sa_s3['agree_18']
best_s3 = sa_s3.loc[sa_s3.groupby('ticker')['score'].idxmax()]
best_s3 = best_s3[best_s3['mean_va_pnl'] >= DEPLOY_THRESHOLD]

bq_configs = {}
for _, row in best_s3.iterrows():
    tk = row['ticker']
    bq_configs[tk] = {
        'prob_thresh': float(row['prob_thresh']),
        'tp': float(row['tp_pct']),
        'sl': float(row['sl_pct']),
        'avg_pnl': float(row['mean_va_pnl']),
        'experiment': row['experiment'],
        'dip_pct': float(row['dip_pct']),
        'max_wait_minutes': float(row['max_wait_minutes']),
        'entry_minute': int(row['entry_minute']) if pd.notna(row['entry_minute']) else ENTRY_MIN,
    }

# Per-(ticker, experiment) configs — for composite gates (all tickers, not just S3)
bq_per_exp = {}
sa_all = sa_limit[sa_limit['ticker'].isin(ALL_TICKERS)].copy()
sa_all['score'] = (sa_all['mean_va_pnl'] ** 2) * sa_all['agree_18']
best_per_tk_exp = sa_all.loc[sa_all.groupby(['ticker', 'experiment'])['score'].idxmax()]
best_per_tk_exp = best_per_tk_exp[best_per_tk_exp['mean_va_pnl'] >= DEPLOY_THRESHOLD]

for _, row in best_per_tk_exp.iterrows():
    bq_per_exp[(row['ticker'], row['experiment'])] = {
        'prob_thresh': float(row['prob_thresh']),
        'tp': float(row['tp_pct']),
        'sl': float(row['sl_pct']),
        'avg_pnl': float(row['mean_va_pnl']),
        'dip_pct': float(row['dip_pct']),
        'max_wait_minutes': float(row['max_wait_minutes']),
        'entry_minute': int(row['entry_minute']) if pd.notna(row['entry_minute']) else ENTRY_MIN,
    }
print(f"    {len(bq_per_exp)} per-(ticker, experiment) configs for composites")

# Also add best configs for non-S3 tickers (for composite trade sim)
for ticker in ALL_TICKERS:
    if ticker in bq_configs:
        continue
    tk_rows = sa_limit[sa_limit['ticker'] == ticker].copy()
    if tk_rows.empty:
        bq_configs[ticker] = None
        continue
    tk_rows['score'] = (tk_rows['mean_va_pnl'] ** 2) * tk_rows['agree_18']
    best = tk_rows.loc[tk_rows['score'].idxmax()]
    if best['mean_va_pnl'] < DEPLOY_THRESHOLD:
        bq_configs[ticker] = None
        continue
    bq_configs[ticker] = {
        'prob_thresh': float(best['prob_thresh']),
        'tp': float(best['tp_pct']),
        'sl': float(best['sl_pct']),
        'avg_pnl': float(best['mean_va_pnl']),
        'experiment': best['experiment'],
        'dip_pct': float(best['dip_pct']),
        'max_wait_minutes': float(best['max_wait_minutes']),
        'entry_minute': int(best['entry_minute']) if pd.notna(best['entry_minute']) else ENTRY_MIN,
    }

# Print configs
for ticker in ALL_TICKERS:
    cfg = bq_configs.get(ticker)
    if cfg is None:
        print(f"    {ticker}: no qualifying limit_entry config")
        continue
    s3_flag = "S3" if ticker in s3_tickers else "  "
    n_exps = len([k for k in bq_per_exp if k[0] == ticker])
    print(f"    {s3_flag} {ticker}: prob>{cfg['prob_thresh']:.2f} TP={cfg['tp']:.0%} "
          f"SL={cfg['sl']:.0%} dip={cfg['dip_pct']:.1%} wait={cfg['max_wait_minutes']:.0f}m "
          f"avgPnL={cfg['avg_pnl']:.2%} ({cfg['experiment']}) [{n_exps} exps]")

del raw_df, indiv, sa, sa_limit, sa_fam, best_per_fam, sa_s3, best_s3, sa_all, best_per_tk_exp
_release_memory()

# Build EXP_CONFIGS dynamically from BQ experiments
all_bq_experiments = set()
for (tk, exp) in bq_per_exp:
    all_bq_experiments.add(exp)
for tk, cfg in bq_configs.items():
    if cfg is not None:
        all_bq_experiments.add(cfg['experiment'])

for exp_name in sorted(all_bq_experiments):
    ecfg = _build_exp_config(exp_name)
    if ecfg is not None:
        EXP_CONFIGS[exp_name] = ecfg

SCORE_EXPERIMENTS = sorted(EXP_CONFIGS.keys())
print(f"    {len(SCORE_EXPERIMENTS)} experiments to score: {', '.join(SCORE_EXPERIMENTS)}")


# ── Step 2: Load OHLCV + build features ──
print(f"\n  STEP 2: Loading OHLCV + engineering features...")
market_df = build_market_context()

featured_data = {}
ohlcv_arrays = {}
for ticker in ALL_TICKERS:
    df = load_master(ticker)
    if df is None:
        print(f"    {ticker}: FAILED to load")
        continue
    df = heal_data(df.sort_index())
    ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
    if ohlcv is not None:
        ohlcv_arrays[ticker] = ohlcv
    n_days = len(np.unique(ohlcv['dates'])) if ohlcv is not None else 0
    df_feat = engineer_features(df, market_df)
    featured_data[ticker] = df_feat
    print(f"    {ticker}: {len(df):>8,} bars, {n_days} test days")
    del df; gc.collect()

del market_df
_release_memory()


# ── Step 3: Score tickers × experiments (only BQ-configured combos) ──
needed_combos = set()
for (tk, exp) in bq_per_exp:
    if tk in ALL_TICKERS:
        needed_combos.add((tk, exp))
for tk, cfg in bq_configs.items():
    if cfg is not None and tk in ALL_TICKERS:
        needed_combos.add((tk, cfg['experiment']))

print(f"\n  STEP 3: Live TF scoring ({len(needed_combos)} ticker×experiment combos, "
      f"{len(SCORE_EXPERIMENTS)} experiments)...")

daily_probs = {}
daily_seed_probs = {}

for exp_id in SCORE_EXPERIMENTS:
    ecfg = EXP_CONFIGS[exp_id]
    tickers_for_exp = [tk for tk in ALL_TICKERS
                       if (tk, exp_id) in needed_combos and tk in featured_data]
    if not tickers_for_exp:
        continue
    stats = load_scaling_stats(ecfg['stats_id'])
    n_scored = 0
    for ticker in tickers_for_exp:
        df_scaled = apply_scaling(featured_data[ticker], stats)
        ensemble_probs, per_seed = score_ticker_batch(ticker, df_scaled, exp_id, ecfg)
        if ensemble_probs:
            daily_probs[(ticker, exp_id)] = ensemble_probs
            for seed, sd in per_seed.items():
                daily_seed_probs[(ticker, exp_id, seed)] = sd
            n_scored += 1
        del df_scaled; gc.collect()
    print(f"    {exp_id}: scored {n_scored} tickers")
    _release_memory()

del featured_data
_release_memory()


# ── Step 4: Pre-simulate trades (limit_entry) ──
print(f"\n  STEP 4: Pre-simulating trades (limit_entry mode)...")

daily_trades = {}
test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt = pd.Timestamp(TEST_END).date()
all_trading_dates = set()

for ticker in ALL_TICKERS:
    cfg = bq_configs.get(ticker)
    if cfg is None:
        continue
    ohlcv = ohlcv_arrays.get(ticker)
    if ohlcv is None:
        continue
    tp, sl = cfg['tp'], cfg['sl']
    dip_pct = cfg['dip_pct']
    max_wait = cfg['max_wait_minutes']
    entry_min = cfg.get('entry_minute', ENTRY_MIN)
    outcomes = {}
    n_unfilled = 0
    unique_dates = np.unique(ohlcv['dates'])
    for d_int in unique_dates:
        d_str = str(int(d_int))
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < test_start_dt or d_date > test_end_dt:
            continue
        mask = (ohlcv['dates'] == d_int) & (ohlcv['times'] >= entry_min)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        entry_idx = idxs[0]
        if ohlcv['prices'][entry_idx] <= 0:
            continue
        pnl, _, valid = run_single_trade_limit_entry(
            ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
            ohlcv['dates'], ohlcv['times'],
            entry_idx, tp, sl, EXIT_TIME_MIN, dip_pct, max_wait)
        if valid:
            outcomes[d_date] = pnl - SLIPPAGE_PCT
        else:
            n_unfilled += 1
    if outcomes:
        daily_trades[ticker] = outcomes
        all_trading_dates.update(outcomes.keys())
    n_total = len(outcomes) + n_unfilled
    print(f"    {ticker}: {len(outcomes)} filled / {n_total} days "
          f"(TP={tp:.0%} SL={sl:.0%} dip={dip_pct:.1%} wait={max_wait:.0f}m)")

all_trading_dates = sorted(all_trading_dates)
print(f"    {len(all_trading_dates)} test days with at least one fill")

del ohlcv_arrays
_release_memory()


# ==============================================================================
# STEP 5: COMPOSITE GATES + STRATEGY EXECUTION
# ==============================================================================
print(f"\n  STEP 5: Running 5 strategies...")

ACTIVE_EXPS = SCORE_EXPERIMENTS


def apply_s3_gate(ticker, date):
    """Production S3: score ONE experiment, simple prob threshold."""
    cfg = bq_configs.get(ticker)
    if cfg is None:
        return False
    exp = cfg['experiment']
    probs = daily_probs.get((ticker, exp))
    if probs is None or date not in probs:
        return False
    return probs[date] >= cfg['prob_thresh']


def apply_agree_gate(ticker, date, min_agree=3, min_present=2):
    """Composite agree: each experiment checked against its OWN prob_thresh."""
    signals = 0
    total = 0
    for exp in ACTIVE_EXPS:
        exp_cfg = bq_per_exp.get((ticker, exp))
        if exp_cfg is None:
            continue
        probs = daily_probs.get((ticker, exp))
        if probs is None or date not in probs:
            continue
        total += 1
        if probs[date] >= exp_cfg['prob_thresh']:
            signals += 1
    if total >= min_present and signals >= min(min_agree, (total // 2 + 1)):
        return True
    return False


def apply_seed_gate(ticker, date, seed_pct=0.50, min_seeds=15):
    """Composite seed: each seed checked against its experiment's prob_thresh."""
    total_seeds = 0
    agree_seeds = 0
    for (t, exp, sid), sprobs in daily_seed_probs.items():
        if t != ticker or exp not in ACTIVE_EXPS:
            continue
        exp_cfg = bq_per_exp.get((ticker, exp))
        if exp_cfg is None:
            continue
        if date in sprobs:
            total_seeds += 1
            if sprobs[date] >= exp_cfg['prob_thresh']:
                agree_seeds += 1
    if total_seeds >= min_seeds and agree_seeds / total_seeds >= seed_pct:
        return True
    return False


def get_trade(ticker, date):
    outcomes = daily_trades.get(ticker, {})
    if date in outcomes:
        return outcomes[date]
    return None


def run_static(tickers, gate_func, gate_kwargs, label):
    trades = []
    for d in all_trading_dates:
        for ticker in tickers:
            if gate_func(ticker, d, **gate_kwargs):
                pnl = get_trade(ticker, d)
                if pnl is not None:
                    trades.append({'date': d, 'ticker': ticker, 'pnl': pnl})
    return trades


def run_gate(tickers, gate_func, gate_kwargs, gate_lookback, label):
    trades = []
    reopt_indices = list(range(0, len(all_trading_dates), REOPT_FREQ))
    accumulated = defaultdict(list)

    for ri in reopt_indices:
        test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
        test_dates = all_trading_dates[ri:test_end_idx]

        for ticker in tickers:
            prior = accumulated[ticker]
            recent = prior[-gate_lookback:]
            if len(recent) >= 2:
                if np.mean([t['pnl'] for t in recent]) < 0:
                    continue
            for d in test_dates:
                if gate_func(ticker, d, **gate_kwargs):
                    pnl = get_trade(ticker, d)
                    if pnl is not None:
                        t = {'date': d, 'ticker': ticker, 'pnl': pnl}
                        trades.append(t)
                        accumulated[ticker].append(t)
    return trades


# Run strategies
strategies = {}

# 1. S3 Deployed (benchmark): family-agreement tickers, one experiment, limit_entry trade sim
strategies['S3 Deployed'] = run_static(
    s3_tickers, apply_s3_gate, {},
    'S3 Deployed')
print(f"    [1/5] S3 Deployed: {len(strategies['S3 Deployed'])} trades ({len(s3_tickers)} S3 tickers)")

# 2. Composite seed=50%: per-experiment prob_thresh
strategies['Composite seed=50%'] = run_static(
    ALL_TICKERS, apply_seed_gate, {'seed_pct': 0.50, 'min_seeds': 15},
    'Composite seed=50%')
print(f"    [2/5] Composite seed=50%: {len(strategies['Composite seed=50%'])} trades")

# 3. S3 + GATE lb=10: family-agreement tickers + rolling performance gate
strategies['S3 + GATE lb=10'] = run_gate(
    s3_tickers, apply_s3_gate, {},
    10, 'S3 + GATE lb=10')
print(f"    [3/5] S3 + GATE lb=10: {len(strategies['S3 + GATE lb=10'])} trades")

# 4. S3 + GATE lb=7: family-agreement tickers + rolling performance gate
strategies['S3 + GATE lb=7'] = run_gate(
    s3_tickers, apply_s3_gate, {},
    7, 'S3 + GATE lb=7')
print(f"    [4/5] S3 + GATE lb=7: {len(strategies['S3 + GATE lb=7'])} trades")

# 5. Composite + GATE lb=10: seed=50% + rolling gate
strategies['Composite+GATE'] = run_gate(
    ALL_TICKERS, apply_seed_gate, {'seed_pct': 0.50, 'min_seeds': 15},
    10, 'Composite+GATE')
print(f"    [5/5] Composite+GATE: {len(strategies['Composite+GATE'])} trades")


# ==============================================================================
# STEP 6: RESULTS
# ==============================================================================
STRATEGY_ORDER = ['S3 Deployed', 'Composite seed=50%', 'S3 + GATE lb=10',
                  'S3 + GATE lb=7', 'Composite+GATE']

def compute_stats(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    cum = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    n_months = max(1, df['month'].nunique())
    june = df[df['month'] == '2026-06']

    daily_pnl = df.groupby('date')['pnl'].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() > 0 else 0)

    return {
        'label': label,
        'n': len(df), 'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': max_dd,
        'sharpe': sharpe, 'per_mo': len(df) / n_months,
        'n_tickers': df['ticker'].nunique(),
        'j_n': len(june), 'j_total': june['pnl'].sum() if len(june) else 0,
        'j_avg': june['pnl'].mean() if len(june) else 0,
        'j_wr': (june['pnl'] > 0).mean() if len(june) else 0,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
        'trades_df': df,
    }


stats = {s: compute_stats(strategies[s], s) for s in STRATEGY_ORDER}

print()
print("=" * W)
print(f"  STEP 6: RESULTS")
print("=" * W)


# ── TABLE 1: STRATEGY COMPARISON ──
print(f"\n  TABLE 1: STRATEGY COMPARISON")
print(f"\n  {'Strategy':<22} {'#Tk':>3} {'Trd':>4} {'AvgPnL':>7} {'TotPnL':>8} "
      f"{'WR':>4} {'MaxDD':>7} {'Sharpe':>6} {'T/mo':>5} "
      f"{'JAvg':>6} {'JTrd':>4} {'JWR':>4}")
print(f"  {'─'*22} {'─'*3} {'─'*4} {'─'*7} {'─'*8} "
      f"{'─'*4} {'─'*7} {'─'*6} {'─'*5} "
      f"{'─'*6} {'─'*4} {'─'*4}")
for s in STRATEGY_ORDER:
    st = stats[s]
    if st['n'] == 0:
        print(f"  {s:<22} — no trades")
        continue
    marker = ' ★' if st['avg'] > stats['S3 Deployed'].get('avg', 0) else '  '
    print(f" {marker}{s:<22} {st['n_tickers']:>3} {st['n']:>4} {st['avg']:>+6.2%} "
          f"{st['total']:>+7.1%} {st['wr']:>3.0%} {st['dd']:>+6.1%} "
          f"{st['sharpe']:>5.1f} {st['per_mo']:>5.1f} "
          f"{st['j_avg']:>+5.2%} {st['j_n']:>4} {st['j_wr']:>3.0%}")


# ── TABLE 2: MONTHLY BREAKDOWN ──
months = ['2026-04', '2026-05', '2026-06']
mlabels = ['Apr', 'May', 'Jun']

print(f"\n  TABLE 2: MONTHLY BREAKDOWN")
print(f"\n  {'Strategy':<22}", end="")
for ml in mlabels:
    print(f"  {'':>2}{ml:>20}", end="")
print()
print(f"  {'─'*22}", end="")
for _ in mlabels:
    print(f"  {'─'*22}", end="")
print()

for s in STRATEGY_ORDER:
    st = stats[s]
    if st['n'] == 0:
        continue
    print(f"  {s:<22}", end="")
    for m in months:
        ms = st.get('monthly', {}).get(m)
        if ms:
            print(f"  {ms['avg']:>+5.2%}({ms['n']:>3}t,{ms['wr']:>4.0%}wr)", end="")
        else:
            print(f"  {'—':>22}", end="")
    print()


# ── TABLE 3: PER-TICKER BREAKDOWN ──
print(f"\n  TABLE 3: PER-TICKER BREAKDOWN")
col_w = 18
print(f"\n  {'Ticker':<7}", end="")
for s in STRATEGY_ORDER:
    label = s[:col_w]
    print(f"  {label:<{col_w}}", end="")
print()
print(f"  {'─'*7}", end="")
for _ in STRATEGY_ORDER:
    print(f"  {'─'*col_w}", end="")
print()

for ticker in ALL_TICKERS:
    print(f"  {ticker:<7}", end="")
    for s in STRATEGY_ORDER:
        st = stats[s]
        if st['n'] == 0 or 'trades_df' not in st:
            print(f"  {'—':<{col_w}}", end="")
            continue
        tt = st['trades_df'][st['trades_df']['ticker'] == ticker]
        if len(tt) > 0:
            cell = f"{tt['pnl'].mean():>+5.2%} {len(tt):>2}t {(tt['pnl']>0).mean():>3.0%}w"
            print(f"  {cell:<{col_w}}", end="")
        else:
            print(f"  {'—':<{col_w}}", end="")
    print()


# ── TABLE 4: PORTFOLIO SIMULATION ──
print(f"\n  TABLE 4: PORTFOLIO SIMULATION (${STARTING_CAPITAL:,} starting capital)")

for s in STRATEGY_ORDER:
    st = stats[s]
    if st['n'] == 0:
        continue
    df_t = st['trades_df'].sort_values('date')
    equity = STARTING_CAPITAL
    equity_curve = [{'date': pd.Timestamp(TEST_START).date(), 'equity': equity}]
    peak = equity
    max_dd_pct = 0
    max_dd_dollar = 0

    for _, row in df_t.iterrows():
        alloc = equity / max(1, st['n_tickers'])
        pnl_dollar = alloc * row['pnl']
        equity += pnl_dollar
        equity_curve.append({'date': row['date'], 'equity': equity})
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        if dd < max_dd_pct:
            max_dd_pct = dd
            max_dd_dollar = equity - peak

    total_return = (equity - STARTING_CAPITAL) / STARTING_CAPITAL
    daily_pnl = df_t.groupby('date')['pnl'].sum()
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

    marker = '★' if total_return > (stats['S3 Deployed'].get('total', 0) * 0.9) else ' '
    print(f"\n  {marker} {s}")
    print(f"    Final equity: ${equity:,.0f} ({total_return:+.1%})")
    print(f"    Max drawdown: ${max_dd_dollar:,.0f} ({max_dd_pct:+.1%})")
    print(f"    Sharpe (approx): {sharpe:.2f}")
    print(f"    Trades/month: {st['per_mo']:.1f}")

    strategies[s + '_curve'] = pd.DataFrame(equity_curve)


# ── TABLE 5: JUNE DEEP DIVE ──
print(f"\n  TABLE 5: JUNE DEEP DIVE")
print(f"\n  {'Strategy':<22} {'Trd':>4} {'AvgPnL':>7} {'TotPnL':>8} {'WR':>4}")
print(f"  {'─'*22} {'─'*4} {'─'*7} {'─'*8} {'─'*4}")
for s in STRATEGY_ORDER:
    st = stats[s]
    if st['j_n'] > 0:
        print(f"  {s:<22} {st['j_n']:>4} {st['j_avg']:>+6.2%} "
              f"{st['j_total']:>+7.1%} {st['j_wr']:>3.0%}")
    else:
        print(f"  {s:<22} — no June trades")

print(f"\n  June per-ticker detail:")
for s in STRATEGY_ORDER:
    st = stats[s]
    if st['n'] == 0 or 'trades_df' not in st:
        continue
    june_df = st['trades_df'][st['trades_df']['month'] == '2026-06']
    if june_df.empty:
        continue
    print(f"\n  {s}:")
    for ticker in sorted(june_df['ticker'].unique()):
        tt = june_df[june_df['ticker'] == ticker]
        print(f"    {ticker}: {len(tt)} trades, avg={tt['pnl'].mean():+.2%}, "
              f"WR={(tt['pnl']>0).mean():.0%}")


# ── CHARTS ──
print(f"\n  Generating charts...")

COLORS = {'S3 Deployed': '#2196F3', 'Composite seed=50%': '#4CAF50',
          'S3 + GATE lb=10': '#FF9800', 'S3 + GATE lb=7': '#E91E63',
          'Composite+GATE': '#9C27B0'}
LINESTYLES = {'S3 Deployed': '-', 'Composite seed=50%': '-',
              'S3 + GATE lb=10': '--', 'S3 + GATE lb=7': ':',
              'Composite+GATE': '--'}


# ── CHART 1: Equity Curves ──
fig, ax1 = plt.subplots(figsize=(14, 6))
for s in STRATEGY_ORDER:
    curve_key = s + '_curve'
    if curve_key not in strategies:
        continue
    curve_df = strategies[curve_key]
    dates = pd.to_datetime(curve_df['date'])
    equity = curve_df['equity'] / 1000
    ax1.plot(dates, equity, label=s, color=COLORS.get(s, '#666'),
             linewidth=2.5, linestyle=LINESTYLES.get(s, '-'))

ax1.axhline(y=STARTING_CAPITAL/1000, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax1.axvline(x=pd.Timestamp('2026-06-01'), color='red', linestyle=':', alpha=0.5)
ax1.text(pd.Timestamp('2026-06-02'), ax1.get_ylim()[1] * 0.99, 'Jun 1',
         color='red', fontsize=8, alpha=0.7, va='top')
ax1.set_ylabel('Equity ($K)', fontsize=12)
ax1.set_title('Strategy Comparison — Equity Curves ($200K start)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
plt.savefig('/tmp/chart1_equity_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ── CHART 2: Cumulative PnL % (trade-by-trade) ──
fig, ax2 = plt.subplots(figsize=(14, 6))
for s in STRATEGY_ORDER:
    st = stats[s]
    if st['n'] == 0 or 'trades_df' not in st:
        continue
    df_sorted = st['trades_df'].sort_values('date')
    cum_pnl = df_sorted['pnl'].cumsum() * 100
    dates = pd.to_datetime(df_sorted['date'].values)
    ax2.plot(dates, cum_pnl.values, label=s, color=COLORS.get(s, '#666'),
             linewidth=2, linestyle=LINESTYLES.get(s, '-'))

ax2.axhline(y=0, color='gray', linewidth=0.5)
ax2.axvline(x=pd.Timestamp('2026-06-01'), color='red', linestyle=':', alpha=0.5)
ax2.set_ylabel('Cumulative PnL (%)', fontsize=12)
ax2.set_title('Cumulative PnL — Trade by Trade', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
plt.savefig('/tmp/chart2_cumulative_pnl.png', dpi=150, bbox_inches='tight')
plt.show()


# ── CHART 3: Drawdown ──
fig, ax3 = plt.subplots(figsize=(14, 4))
for s in STRATEGY_ORDER:
    st = stats[s]
    if st['n'] == 0 or 'trades_df' not in st:
        continue
    df_sorted = st['trades_df'].sort_values('date')
    cum_pnl = df_sorted['pnl'].cumsum()
    dd = (cum_pnl - cum_pnl.cummax()) * 100
    dates = pd.to_datetime(df_sorted['date'].values)
    ax3.fill_between(dates, dd.values, 0, alpha=0.15, color=COLORS.get(s, '#666'))
    ax3.plot(dates, dd.values, label=s, color=COLORS.get(s, '#666'),
             linewidth=1.5, linestyle=LINESTYLES.get(s, '-'))

ax3.axvline(x=pd.Timestamp('2026-06-01'), color='red', linestyle=':', alpha=0.5)
ax3.set_ylabel('Drawdown (%)', fontsize=12)
ax3.set_title('Strategy Drawdowns', fontsize=14, fontweight='bold')
ax3.legend(loc='lower left', fontsize=10, framealpha=0.9)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
plt.savefig('/tmp/chart3_drawdowns.png', dpi=150, bbox_inches='tight')
plt.show()


# ── CHART 4: Monthly Avg PnL bar chart ──
fig, ax4 = plt.subplots(figsize=(12, 5))
x = np.arange(len(months))
bar_width = 0.15
for i, s in enumerate(STRATEGY_ORDER):
    st = stats[s]
    if st['n'] == 0:
        continue
    monthly_avgs = []
    for m in months:
        ms = st.get('monthly', {}).get(m)
        monthly_avgs.append(ms['avg'] * 100 if ms else 0)
    bars = ax4.bar(x + i * bar_width, monthly_avgs, bar_width,
                   label=s, color=COLORS.get(s, '#666'), alpha=0.85)
    for bar, val in zip(bars, monthly_avgs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=7)

ax4.set_xticks(x + bar_width * 2)
ax4.set_xticklabels(mlabels, fontsize=12)
ax4.set_ylabel('Avg PnL per Trade (%)', fontsize=12)
ax4.set_title('Monthly Average PnL by Strategy', fontsize=14, fontweight='bold')
ax4.legend(fontsize=9, framealpha=0.9)
ax4.axhline(y=0, color='gray', linewidth=0.5)
ax4.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/tmp/chart4_monthly_avg_pnl.png', dpi=150, bbox_inches='tight')
plt.show()


# ── CHART 5: Per-ticker heatmap ──
active_tickers = [t for t in ALL_TICKERS if any(
    stats[s].get('trades_df') is not None and
    len(stats[s]['trades_df'][stats[s]['trades_df']['ticker'] == t]) > 0
    for s in STRATEGY_ORDER if stats[s]['n'] > 0)]

if active_tickers:
    heatmap_data = np.full((len(active_tickers), len(STRATEGY_ORDER)), np.nan)
    for j, s in enumerate(STRATEGY_ORDER):
        st = stats[s]
        if st['n'] == 0 or 'trades_df' not in st:
            continue
        for i, ticker in enumerate(active_tickers):
            tt = st['trades_df'][st['trades_df']['ticker'] == ticker]
            if len(tt) > 0:
                heatmap_data[i, j] = tt['pnl'].mean() * 100

    fig, ax5 = plt.subplots(figsize=(12, max(4, len(active_tickers) * 0.6)))
    short_labels = ['S3', 'Seed50%', 'S3+G10', 'S3+G7', 'Comp+G']
    im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto',
                    vmin=-3, vmax=3)
    ax5.set_xticks(range(len(STRATEGY_ORDER)))
    ax5.set_xticklabels(short_labels, fontsize=10)
    ax5.set_yticks(range(len(active_tickers)))
    ax5.set_yticklabels(active_tickers, fontsize=10)
    for i in range(len(active_tickers)):
        for j in range(len(STRATEGY_ORDER)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                ax5.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                         fontsize=9, fontweight='bold',
                         color='white' if abs(val) > 2 else 'black')
    plt.colorbar(im, label='Avg PnL (%)', shrink=0.8)
    ax5.set_title('Per-Ticker Avg PnL by Strategy (%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/chart5_ticker_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── CHART 6: Strategy summary bar chart ──
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
metrics_to_plot = [
    ('Total PnL (%)', [stats[s]['total'] * 100 for s in STRATEGY_ORDER if stats[s]['n'] > 0]),
    ('Avg PnL (%)', [stats[s]['avg'] * 100 for s in STRATEGY_ORDER if stats[s]['n'] > 0]),
    ('Max DD (%)', [stats[s]['dd'] * 100 for s in STRATEGY_ORDER if stats[s]['n'] > 0]),
    ('Sharpe', [stats[s]['sharpe'] for s in STRATEGY_ORDER if stats[s]['n'] > 0]),
]
active_strategies = [s for s in STRATEGY_ORDER if stats[s]['n'] > 0]
short_names = ['S3', 'Seed50%', 'S3+G10', 'S3+G7', 'Comp+G'][:len(active_strategies)]

for ax, (metric_name, values) in zip(axes, metrics_to_plot):
    colors_list = [COLORS.get(s, '#666') for s in active_strategies]
    bars = ax.bar(range(len(values)), values, color=colors_list, alpha=0.85)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(short_names, fontsize=8, rotation=30, ha='right')
    ax.set_title(metric_name, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    if 'DD' in metric_name:
        ax.axhline(y=0, color='gray', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.3 if bar.get_height() >= 0 else -1.5),
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

fig.suptitle('Strategy Comparison Summary', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/tmp/chart6_summary.png', dpi=150, bbox_inches='tight')
plt.show()


# ── DIAGNOSIS ──
print()
print("=" * W)
print(f"  DIAGNOSIS")
print("=" * W)

ranked = sorted(
    [(s, stats[s]) for s in STRATEGY_ORDER if stats[s]['n'] > 0],
    key=lambda x: x[1]['total'], reverse=True)

if ranked:
    best = ranked[0]
    print(f"\n  BEST TOTAL PnL: {best[0]}")
    print(f"    {best[1]['total']:+.1%} total, {best[1]['avg']:+.2%} avg, "
          f"{best[1]['wr']:.0%} WR, {best[1]['dd']:+.1%} MaxDD, "
          f"Sharpe={best[1]['sharpe']:.2f}")

    bench = stats['S3 Deployed']
    if bench['n'] > 0:
        print(f"\n  vs BENCHMARK (S3 Deployed):")
        print(f"    ΔTotal: {best[1]['total'] - bench['total']:+.1%}")
        print(f"    ΔAvgPnL: {best[1]['avg'] - bench['avg']:+.2%}")
        print(f"    ΔMaxDD: {best[1]['dd'] - bench['dd']:+.1%}")

best_risk = min(
    [(s, stats[s]) for s in STRATEGY_ORDER if stats[s]['n'] > 0],
    key=lambda x: abs(x[1]['dd']))
print(f"\n  BEST RISK-ADJUSTED: {best_risk[0]}")
print(f"    {best_risk[1]['dd']:+.1%} MaxDD, {best_risk[1]['total']:+.1%} total, "
      f"Sharpe={best_risk[1]['sharpe']:.2f}")

best_june = max(
    [(s, stats[s]) for s in STRATEGY_ORDER if stats[s]['j_n'] > 0],
    key=lambda x: x[1]['j_avg'], default=None)
if best_june:
    print(f"\n  BEST JUNE: {best_june[0]}")
    print(f"    {best_june[1]['j_avg']:+.2%} avg, {best_june[1]['j_n']} trades, "
          f"{best_june[1]['j_wr']:.0%} WR")

print()
print("=" * W)
print(f"  Analysis complete. 5 strategies evaluated with live TF scoring on {len(ALL_TICKERS)} tickers.")
print("=" * W)
