# %% ── OLD vs NEW S3 Strategy Comparison (CSV-Scored) ────────────────────────
#
# Side-by-side backtest of two S3 LONG config selections:
#   OLD: BQ configs from base robust_wf_v12 only (5 families: 152,160,170,180,200)
#   NEW: BQ configs from base + per-fold optimized (robust_wf_v12_17x/18x/20x/24x)
#        7 families: adds 220, 240
#
# Same CSV probabilities, same trade sim, same family agreement gate.
# Only difference: which BQ configs are available for selection.
# CSV-scored — no GPU required. Colab-ready.

import os, io, gc, sys, warnings, ctypes
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

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
    try: auth.authenticate_user()
    except Exception: pass

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")
RESULTS_DIR    = "Algos/Results_scored_20260703"
RESULTS_DIR_V2 = "Algos/Results_all_v2"
INFERENCE_DIR  = "Data/Inference_v2"

TEST_START  = '2026-04-01'
TEST_END    = '2026-08-04'
HALF2_START = '2026-06-01'
ENTRY_MIN   = 375
EXIT_TIME_MIN = 780

SLIPPAGE = 0.0015
STARTING_CAPITAL = 200_000

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

DEPLOY_THRESHOLD = 0.018
MIN_SEEDS  = 3
MAX_VA_OPT = 3.0
EXCLUDE_TICKERS = {'LUNR'}

DEPLOY_EXPERIMENTS = {
    '152_XSECTOR', '160_XSECTOR',
    '170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
    '173_XSECTOR', '174_XSECTOR', '175_XSECTOR',
    '180_XSECTOR', '181_XSECTOR', '182_XSECTOR',
    '183_XSECTOR', '184_XSECTOR', '185_XSECTOR',
    '200_XSECTOR', '201_XSECTOR', '202_XSECTOR',
    '203_XSECTOR', '204_XSECTOR', '205_XSECTOR',
    '220_XSECTOR', '221_XSECTOR', '222_XSECTOR',
    '223_XSECTOR', '224_XSECTOR', '225_XSECTOR',
    '240_XSECTOR', '241_XSECTOR', '242_XSECTOR',
    '243_XSECTOR', '244_XSECTOR', '245_XSECTOR',
}

GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
               'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
               'grace': '_GRACE', 'limit_entry': '_LIMIT'}

VOL_LOOKBACK_DAYS = 30

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120

OLD_COLOR = '#2196F3'
NEW_COLOR = '#E91E63'


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def _safe_float(val, default=0.0):
    try:
        if pd.isna(val): return default
    except (TypeError, ValueError): pass
    try: return float(val)
    except (TypeError, ValueError): return default


def _base_exp(exp_name):
    for sfx in sorted(MODE_SUFFIX.values(), key=len, reverse=True):
        if sfx and exp_name.endswith(sfx):
            return exp_name[:-len(sfx)]
    return exp_name


def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    if 200 <= num <= 205: return '200'
    if 220 <= num <= 225: return '220'
    if 240 <= num <= 245: return '240'
    return 'other'


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
        std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'volume': 'Volume'}
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        df = df[df.index.notna()].sort_index()
        cutoff = pd.Timestamp(TEST_START, tz='UTC') - pd.Timedelta(days=30)
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


# ==============================================================================
# CSV SCORE LOADING
# ==============================================================================
def load_seed_scores(ticker, experiment_base):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    bucket = gcs_client.bucket(BUCKET_NAME)
    all_results = {}
    for results_dir in [RESULTS_DIR_V2, RESULTS_DIR]:
        prefix = f"{results_dir}/Universal_Exp{experiment_base}/{clean}/Seed_"
        blobs = list(bucket.list_blobs(prefix=prefix))
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


def compute_daily_prob(seed_scores, entry_min):
    n_seeds = 0
    avg_probs = None
    for seed_id, sdf in seed_scores.items():
        if 'Prob_Spike' not in sdf.columns:
            col = [c for c in sdf.columns if 'prob' in c.lower() or 'spike' in c.lower()]
            if not col:
                continue
            sdf = sdf.rename(columns={col[0]: 'Prob_Spike'})
        bar_mins = sdf.index.hour * 60 + sdf.index.minute
        entry_bars = sdf[bar_mins == entry_min]
        if entry_bars.empty:
            continue
        daily_prob = entry_bars.groupby(entry_bars.index.date)['Prob_Spike'].first()
        if avg_probs is None:
            avg_probs = daily_prob.copy()
        else:
            avg_probs = avg_probs.add(daily_prob, fill_value=0)
        n_seeds += 1
    if n_seeds > 0 and avg_probs is not None:
        return (avg_probs / n_seeds).to_dict()
    return {}


# ==============================================================================
# TRADE SIMULATION — limit_entry
# ==============================================================================
def run_trade_limit_entry(prices, highs, lows, date_ints, time_mins,
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


# ==============================================================================
# STATS COMPUTATION
# ==============================================================================
def compute_stats(trades_list, label, n_alloc=1):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0, 'sharpe': 0, 'per_mo': 0, 'n_tickers': 0,
                'n_alloc': n_alloc, 'port_return': 0, 'monthly': {},
                'trades_df': pd.DataFrame()}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    n_months = max(1, df['month'].nunique())
    n_pos = max(1, n_alloc)
    df_sorted = df.sort_values('date')
    port_pnl = df_sorted['pnl'] / n_pos
    port_cum = port_pnl.cumsum()
    port_dd = (port_cum - port_cum.cummax()).min()
    port_return = port_cum.iloc[-1] if len(port_cum) > 0 else 0
    daily_pnl = df.groupby('date')['pnl'].sum() / n_pos
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() > 0 else 0)
    return {
        'label': label,
        'n': len(df), 'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': port_dd,
        'sharpe': sharpe, 'per_mo': len(df) / n_months,
        'n_tickers': df['ticker'].nunique(),
        'n_alloc': n_pos, 'port_return': port_return,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
        'trades_df': df,
    }


# ==============================================================================
# S3 SELECTION PIPELINE — reusable for OLD and NEW
# ==============================================================================
def run_s3_selection(raw_df, label):
    """Run S3 family-agreement gate on BQ data. Returns (configs_dict, agreement_dict)."""
    df = raw_df.copy()
    df['ticker'] = df['ticker'].replace(TICKER_FILE_MAP)
    df['base_experiment'] = df['experiment'].apply(_base_exp)
    df = df[df['base_experiment'].isin(DEPLOY_EXPERIMENTS)]
    df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
    print(f"    [{label}] {len(df)} after experiment + ticker filter")

    df = df[df['opt_avg_pnl'] > 0].copy()
    df = df[df['val_a_avg_pnl'] / df['opt_avg_pnl'] <= MAX_VA_OPT]
    print(f"    [{label}] {len(df)} after overfit filter")

    indiv = df[df['is_ensemble'] == False].copy()
    for k in GRP_KEYS:
        if k not in indiv.columns:
            indiv[k] = 0
        else:
            indiv[k] = indiv[k].fillna(0)

    sa = indiv.groupby(GRP_KEYS).agg(
        n_seeds=('seed', 'nunique'),
        n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= DEPLOY_THRESHOLD).sum()),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        mean_va_count=('val_a_count', 'mean'),
        mean_va_freq=('val_a_freq', 'mean'),
        vb_pnl=('val_b_avg_pnl', 'mean'),
    ).reset_index()
    sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)
    sa = sa[sa['n_seeds'] >= MIN_SEEDS].copy()
    sa_limit = sa[sa['tpsl_mode'] == 'limit_entry'].copy()
    print(f"    [{label}] {len(sa_limit)} limit_entry seed groups (n_seeds >= {MIN_SEEDS})")

    sa_limit['score'] = (sa_limit['mean_va_pnl'] ** 2) * sa_limit['agree_18']
    sa_limit['exp_fam'] = sa_limit['experiment'].apply(assign_family)
    sa_fam = sa_limit[sa_limit['exp_fam'] != 'other'].copy()

    best_per_fam = sa_fam.loc[sa_fam.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    best_per_fam = best_per_fam[best_per_fam['mean_va_pnl'] >= DEPLOY_THRESHOLD]

    agreement = {}
    for tk in best_per_fam['ticker'].unique():
        tk_rows = best_per_fam[best_per_fam['ticker'] == tk]
        n_fams = len(tk_rows)
        n_pos = int((tk_rows['vb_pnl'] > 0).sum())
        fam_list = sorted(tk_rows['exp_fam'].unique())
        agreement[tk] = {
            'n_fams': n_fams, 'n_pos': n_pos,
            'majority_pos': n_pos > n_fams / 2 and n_fams >= 2,
            'families': fam_list,
        }

    s3_tickers = sorted([tk for tk, ag in agreement.items() if ag['majority_pos']])
    print(f"    [{label}] Family agreement: {len(s3_tickers)} tickers pass S3 gate")

    sa_s3 = sa_limit[sa_limit['ticker'].isin(s3_tickers)].copy()
    sa_s3['score'] = (sa_s3['mean_va_pnl'] ** 2) * sa_s3['agree_18']
    best_s3 = sa_s3.loc[sa_s3.groupby('ticker')['score'].idxmax()]
    best_s3 = best_s3[best_s3['mean_va_pnl'] >= DEPLOY_THRESHOLD]

    configs = {}
    for _, row in best_s3.iterrows():
        tk = row['ticker']
        configs[tk] = {
            'prob_thresh': float(row['prob_thresh']),
            'tp': float(row['tp_pct']),
            'sl': float(row['sl_pct']),
            'avg_pnl': float(row['mean_va_pnl']),
            'experiment': row['experiment'],
            'dip_pct': float(row['dip_pct']),
            'max_wait_minutes': float(row['max_wait_minutes']),
            'entry_minute': int(row['entry_minute']) if pd.notna(row['entry_minute']) else ENTRY_MIN,
            'agree_18': float(row['agree_18']),
        }

    return configs, agreement


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print("=" * W)
print(f"  OLD vs NEW S3 STRATEGY COMPARISON (CSV-Scored)")
print(f"  OLD: base robust_wf_v12 only")
print(f"  NEW: base + per-fold optimized (17x/18x/20x/24x)")
print(f"  TEST: {TEST_START} -> {TEST_END}")
print(f"  CAPITAL: ${STARTING_CAPITAL:,}  |  EXIT: 1:00 PM PST")
print("=" * W)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load BQ data — OLD and NEW
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  STEP 1: Loading BQ data...")

bq_cols = """ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult, grace_minutes,
       dip_pct, max_wait_minutes, vol_filter_pct,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, opt_avg_pnl, is_ensemble"""

q_old = f"""
SELECT {bq_cols}
FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
WHERE model_version IN ('robust_wf_v12', 'robust_wf_v12_stitched')
  AND survived_a = TRUE
  AND val_a_count >= 3
  AND val_a_avg_pnl > 0
"""

q_new = f"""
SELECT {bq_cols}
FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
WHERE (model_version IN ('robust_wf_v12', 'robust_wf_v12_stitched')
       OR model_version LIKE 'robust_wf_v12_17%'
       OR model_version LIKE 'robust_wf_v12_18%'
       OR model_version LIKE 'robust_wf_v12_20%'
       OR model_version LIKE 'robust_wf_v12_24%')
  AND survived_a = TRUE
  AND val_a_count >= 3
  AND val_a_avg_pnl > 0
"""

raw_old = bq_client.query(q_old).to_dataframe()
raw_new = bq_client.query(q_new).to_dataframe()
print(f"    OLD: {len(raw_old)} raw BQ rows")
print(f"    NEW: {len(raw_new)} raw BQ rows (+{len(raw_new) - len(raw_old)} from per-fold)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Run S3 selection for both
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  STEP 2: S3 selection pipeline...")

print(f"\n  --- OLD S3 (base only) ---")
old_configs, old_agreement = run_s3_selection(raw_old, 'OLD')

print(f"\n  --- NEW S3 (base + per-fold) ---")
new_configs, new_agreement = run_s3_selection(raw_new, 'NEW')

del raw_old, raw_new
_release_memory()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Compare selections
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  STEP 3: Selection comparison...")

old_tickers = sorted(old_configs.keys())
new_tickers = sorted(new_configs.keys())
all_tickers = sorted(set(old_tickers) | set(new_tickers))
common_tickers = sorted(set(old_tickers) & set(new_tickers))
old_only = sorted(set(old_tickers) - set(new_tickers))
new_only = sorted(set(new_tickers) - set(old_tickers))

print(f"\n  TICKER SELECTION:")
print(f"    OLD: {len(old_tickers)} tickers — {', '.join(old_tickers)}")
print(f"    NEW: {len(new_tickers)} tickers — {', '.join(new_tickers)}")
print(f"    Common: {len(common_tickers)}")
if old_only:
    print(f"    OLD only (dropped in NEW): {', '.join(old_only)}")
if new_only:
    print(f"    NEW only (gained): {', '.join(new_only)}")

print(f"\n  CONFIG COMPARISON (common tickers):")
print(f"  {'Ticker':<7} | {'------- OLD --------':} | {'------- NEW --------':} | {'Changed?':}")
print(f"  {'':7} | {'Experiment':<20} {'Prob':>5} {'TP':>5} {'SL':>5} {'Dip':>5} "
      f"| {'Experiment':<20} {'Prob':>5} {'TP':>5} {'SL':>5} {'Dip':>5} |")
print(f"  {'─'*100}")

n_changed = 0
for tk in common_tickers:
    oc = old_configs[tk]
    nc = new_configs[tk]
    same = (oc['experiment'] == nc['experiment'] and
            abs(oc['prob_thresh'] - nc['prob_thresh']) < 0.001 and
            abs(oc['tp'] - nc['tp']) < 0.001 and
            abs(oc['sl'] - nc['sl']) < 0.001 and
            abs(oc['dip_pct'] - nc['dip_pct']) < 0.001)
    flag = '  SAME' if same else '  ** CHANGED **'
    if not same:
        n_changed += 1
    print(f"  {tk:<7} | {oc['experiment']:<20} {oc['prob_thresh']:>5.2f} {oc['tp']:>4.0%} "
          f"{oc['sl']:>4.0%} {oc['dip_pct']:>4.1%} "
          f"| {nc['experiment']:<20} {nc['prob_thresh']:>5.2f} {nc['tp']:>4.0%} "
          f"{nc['sl']:>4.0%} {nc['dip_pct']:>4.1%} |{flag}")

for tk in old_only:
    oc = old_configs[tk]
    print(f"  {tk:<7} | {oc['experiment']:<20} {oc['prob_thresh']:>5.2f} {oc['tp']:>4.0%} "
          f"{oc['sl']:>4.0%} {oc['dip_pct']:>4.1%} "
          f"| {'--- DROPPED ---':<20} {'':>5} {'':>5} {'':>5} {'':>5} |  OLD ONLY")

for tk in new_only:
    nc = new_configs[tk]
    print(f"  {tk:<7} | {'--- N/A ---':<20} {'':>5} {'':>5} {'':>5} {'':>5} "
          f"| {nc['experiment']:<20} {nc['prob_thresh']:>5.2f} {nc['tp']:>4.0%} "
          f"{nc['sl']:>4.0%} {nc['dip_pct']:>4.1%} |  NEW ONLY")

print(f"\n  Summary: {n_changed}/{len(common_tickers)} common tickers changed config, "
      f"{len(old_only)} dropped, {len(new_only)} gained")

# ── Family agreement detail ──
print(f"\n  FAMILY AGREEMENT DETAIL:")
print(f"  {'Ticker':<7} | {'OLD fams':>8} {'OLD +/-':>7} {'OLD vote':>8} "
      f"| {'NEW fams':>8} {'NEW +/-':>7} {'NEW vote':>8} | {'Δ':}")
print(f"  {'─'*80}")
for tk in all_tickers:
    oa = old_agreement.get(tk, {'n_fams': 0, 'n_pos': 0, 'majority_pos': False, 'families': []})
    na = new_agreement.get(tk, {'n_fams': 0, 'n_pos': 0, 'majority_pos': False, 'families': []})
    o_vote = 'PASS' if oa['majority_pos'] else 'FAIL'
    n_vote = 'PASS' if na['majority_pos'] else 'FAIL'
    delta = ''
    if o_vote != n_vote:
        delta = f'  {o_vote}->{n_vote}'
    elif oa['n_fams'] != na['n_fams']:
        delta = f'  +{na["n_fams"] - oa["n_fams"]} fams'
    print(f"  {tk:<7} | {oa['n_fams']:>8} {oa['n_pos']:>3}/{oa['n_fams']:<3} {o_vote:>8} "
          f"| {na['n_fams']:>8} {na['n_pos']:>3}/{na['n_fams']:<3} {n_vote:>8} |{delta}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Load OHLCV
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  STEP 4: Loading OHLCV for {len(all_tickers)} tickers...")

ohlcv_arrays = {}
for ticker in all_tickers:
    try:
        df = load_master(ticker)
        if df is None:
            continue
        if isinstance(df.index, pd.MultiIndex):
            _master_cache.pop(ticker, None)
            continue
        df = heal_data(df)
        if isinstance(df.index, pd.MultiIndex):
            _master_cache.pop(ticker, None)
            continue
        ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
        if ohlcv is None:
            print(f"    {ticker}: no data in test period")
            _master_cache.pop(ticker, None)
            continue
        ohlcv_arrays[ticker] = ohlcv
        n_days = len(np.unique(ohlcv['dates']))
        in_old = 'O' if ticker in old_tickers else ' '
        in_new = 'N' if ticker in new_tickers else ' '
        print(f"    {ticker} [{in_old}{in_new}]: {n_days} trading days")
        del df; _master_cache.pop(ticker, None)
    except Exception as ex:
        print(f"    {ticker}: FAILED -- {ex}")
        _master_cache.pop(ticker, None)
    gc.collect()

_release_memory()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Load CSV scores (once per experiment, shared between OLD and NEW)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  STEP 5: Loading CSV scores...")

all_experiments = set()
for cfg in list(old_configs.values()) + list(new_configs.values()):
    all_experiments.add(cfg['experiment'])
print(f"    {len(all_experiments)} unique experiments to load")

score_cache = {}
for ticker in all_tickers:
    if ticker not in ohlcv_arrays:
        continue
    exps_for_ticker = set()
    if ticker in old_configs:
        exps_for_ticker.add(old_configs[ticker]['experiment'])
    if ticker in new_configs:
        exps_for_ticker.add(new_configs[ticker]['experiment'])

    for exp in exps_for_ticker:
        exp_base = exp.replace('_LIMIT', '')
        cache_key = (ticker, exp_base)
        if cache_key in score_cache:
            continue
        seeds = load_seed_scores(ticker, exp_base)
        if not seeds:
            print(f"    {ticker}/{exp_base}: no CSV scores")
            continue
        entry_min_old = old_configs[ticker]['entry_minute'] if ticker in old_configs else ENTRY_MIN
        entry_min_new = new_configs[ticker]['entry_minute'] if ticker in new_configs else ENTRY_MIN
        entry_min = entry_min_old
        probs = compute_daily_prob(seeds, entry_min)
        if probs:
            score_cache[cache_key] = probs
            print(f"    {ticker} ({exp_base}): {len(seeds)} seeds, {len(probs)} dates")
        if entry_min_new != entry_min_old and ticker in new_configs:
            probs2 = compute_daily_prob(seeds, entry_min_new)
            if probs2:
                score_cache[(ticker, exp_base, entry_min_new)] = probs2
        del seeds; gc.collect()

_release_memory()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Run backtests for OLD and NEW
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  STEP 6: Running backtests...")

test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()

def run_backtest(configs, label):
    tickers = sorted(configs.keys())
    trades_list = []
    for ticker in tickers:
        cfg = configs[ticker]
        ohlcv = ohlcv_arrays.get(ticker)
        if ohlcv is None:
            continue
        exp_base = cfg['experiment'].replace('_LIMIT', '')
        entry_min = cfg.get('entry_minute', ENTRY_MIN)
        probs = score_cache.get((ticker, exp_base))
        if probs is None:
            probs = score_cache.get((ticker, exp_base, entry_min), {})
        if not probs:
            continue
        tp, sl = cfg['tp'], cfg['sl']
        dip_pct = cfg['dip_pct']
        max_wait = cfg['max_wait_minutes']
        n_filled = 0
        n_unfilled = 0
        n_signals = 0

        unique_dates = np.unique(ohlcv['dates'])
        for d_int in unique_dates:
            d_str = str(int(d_int))
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            prob = probs.get(d_date, 0.0)
            if prob < cfg['prob_thresh']:
                continue
            n_signals += 1
            mask = (ohlcv['dates'] == d_int) & (ohlcv['times'] >= entry_min)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            entry_idx = idxs[0]
            if ohlcv['prices'][entry_idx] <= 0:
                continue
            pnl, _, valid = run_trade_limit_entry(
                ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                ohlcv['dates'], ohlcv['times'],
                entry_idx, tp, sl, EXIT_TIME_MIN, dip_pct, max_wait)
            if valid:
                trades_list.append({
                    'date': d_date, 'ticker': ticker, 'pnl': pnl - SLIPPAGE,
                    'prob': prob, 'experiment': cfg['experiment'],
                })
                n_filled += 1
            else:
                n_unfilled += 1

        print(f"    [{label}] {ticker}: {n_filled} filled / {n_signals} signals "
              f"({cfg['experiment']} prob>{cfg['prob_thresh']:.2f})")

    return trades_list

np.random.seed(42)
print(f"\n  --- OLD S3 ---")
old_trades = run_backtest(old_configs, 'OLD')

np.random.seed(42)
print(f"\n  --- NEW S3 ---")
new_trades = run_backtest(new_configs, 'NEW')


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Results
# ══════════════════════════════════════════════════════════════════════════════
stats_old = compute_stats(old_trades, 'OLD S3', len(old_configs))
stats_new = compute_stats(new_trades, 'NEW S3', len(new_configs))

print(f"\n{'='*W}")
print(f"  RESULTS — OLD vs NEW S3 COMPARISON")
print(f"{'='*W}")


# ── TABLE 1: Portfolio comparison ──
print(f"\n  TABLE 1: PORTFOLIO COMPARISON")
print(f"\n  {'Strategy':<12} {'#Tk':>3} {'Trd':>5} {'AvgPnL':>8} {'PortRet':>8} "
      f"{'WR':>5} {'MaxDD':>8} {'Sharpe':>7} {'T/mo':>5} {'RawSum':>8}")
print(f"  {'─'*80}")

for label, st in [('OLD S3', stats_old), ('NEW S3', stats_new)]:
    if st['n'] == 0:
        print(f"  {label:<12} — no trades")
        continue
    print(f"  {label:<12} {st['n_tickers']:>3} {st['n']:>5} {st['avg']:>+7.2%} "
          f"{st['port_return']:>+7.1%} {st['wr']:>4.0%} {st['dd']:>+7.1%} "
          f"{st['sharpe']:>6.2f} {st['per_mo']:>5.1f} {st['total']:>+7.1%}")

if stats_old['n'] > 0 and stats_new['n'] > 0:
    d_avg = stats_new['avg'] - stats_old['avg']
    d_port = stats_new['port_return'] - stats_old['port_return']
    d_dd = stats_new['dd'] - stats_old['dd']
    d_sharpe = stats_new['sharpe'] - stats_old['sharpe']
    print(f"  {'DELTA':<12} {'':>3} {'':>5} {d_avg:>+7.2%} "
          f"{d_port:>+7.1%} {'':>5} {d_dd:>+7.1%} "
          f"{d_sharpe:>+6.2f}")


# ── TABLE 2: Monthly breakdown ──
months = sorted(set(list(stats_old.get('monthly', {}).keys()) +
                     list(stats_new.get('monthly', {}).keys())))

if months:
    print(f"\n  TABLE 2: MONTHLY BREAKDOWN")
    print(f"\n  {'Month':<8} | {'──── OLD S3 ───────────':} | "
          f"{'──── NEW S3 ───────────':} | {'Δ Avg':>7}")
    print(f"  {'':8} | {'Trd':>3} {'AvgPnL':>7} {'WR':>5} {'TotPnL':>8} | "
          f"{'Trd':>3} {'AvgPnL':>7} {'WR':>5} {'TotPnL':>8} |")
    print(f"  {'─'*80}")

    for month in months:
        row = f"  {month:<8} |"
        o_avg = n_avg = 0
        for st in [stats_old, stats_new]:
            m = st.get('monthly', {}).get(month)
            if m:
                row += f" {m['n']:>3} {m['avg']:>+6.2%} {m['wr']:>4.0%} {m['total']:>+7.1%} |"
                if st is stats_old: o_avg = m['avg']
                else: n_avg = m['avg']
            else:
                row += f" {'—':>3} {'—':>7} {'—':>5} {'—':>8} |"
        delta = n_avg - o_avg
        row += f" {delta:>+6.2%}" if (o_avg or n_avg) else ""
        print(row)


# ── TABLE 3: Per-ticker comparison ──
print(f"\n  TABLE 3: PER-TICKER HEAD-TO-HEAD")
print(f"\n  {'Ticker':<7} | {'── OLD ──────────────────────':} | "
      f"{'── NEW ──────────────────────':} | {'Δ Avg':>7}")
print(f"  {'':7} | {'Experiment':<20} {'Trd':>4} {'Avg':>7} {'WR':>5} "
      f"| {'Experiment':<20} {'Trd':>4} {'Avg':>7} {'WR':>5} |")
print(f"  {'─'*100}")

for ticker in all_tickers:
    ot = [t for t in old_trades if t['ticker'] == ticker]
    nt = [t for t in new_trades if t['ticker'] == ticker]
    o_exp = old_configs[ticker]['experiment'] if ticker in old_configs else '—'
    n_exp = new_configs[ticker]['experiment'] if ticker in new_configs else '—'
    o_n = len(ot)
    n_n = len(nt)
    o_avg = np.mean([t['pnl'] for t in ot]) if ot else 0
    n_avg = np.mean([t['pnl'] for t in nt]) if nt else 0
    o_wr = np.mean([t['pnl'] > 0 for t in ot]) if ot else 0
    n_wr = np.mean([t['pnl'] > 0 for t in nt]) if nt else 0
    delta = n_avg - o_avg if (ot or nt) else 0

    o_str = f"{o_exp:<20} {o_n:>4} {o_avg:>+6.2%} {o_wr:>4.0%}" if ot else f"{'— not selected —':<20} {'':>4} {'':>7} {'':>5}"
    n_str = f"{n_exp:<20} {n_n:>4} {n_avg:>+6.2%} {n_wr:>4.0%}" if nt else f"{'— not selected —':<20} {'':>4} {'':>7} {'':>5}"
    d_str = f"{delta:>+6.2%}" if (ot or nt) else ""
    winner = ' *' if delta > 0.005 else (' *' if delta < -0.005 else '')
    print(f"  {ticker:<7} | {o_str} | {n_str} | {d_str}{winner}")


# ── TABLE 4: Verdict ──
print(f"\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

if stats_old['n'] > 0 and stats_new['n'] > 0:
    improvements = sum([
        stats_new['avg'] > stats_old['avg'],
        stats_new['port_return'] > stats_old['port_return'],
        abs(stats_new['dd']) < abs(stats_old['dd']),
        stats_new['sharpe'] > stats_old['sharpe'],
    ])

    print(f"\n  NEW vs OLD:")
    print(f"    AvgPnL:    OLD {stats_old['avg']:+.2%} → NEW {stats_new['avg']:+.2%}  "
          f"({'better' if stats_new['avg'] > stats_old['avg'] else 'worse'})")
    print(f"    PortRet:   OLD {stats_old['port_return']:+.1%} → NEW {stats_new['port_return']:+.1%}  "
          f"({'better' if stats_new['port_return'] > stats_old['port_return'] else 'worse'})")
    print(f"    MaxDD:     OLD {stats_old['dd']:+.1%} → NEW {stats_new['dd']:+.1%}  "
          f"({'shallower' if abs(stats_new['dd']) < abs(stats_old['dd']) else 'deeper'})")
    print(f"    Sharpe:    OLD {stats_old['sharpe']:.2f} → NEW {stats_new['sharpe']:.2f}  "
          f"({'better' if stats_new['sharpe'] > stats_old['sharpe'] else 'worse'})")
    print(f"    Tickers:   OLD {stats_old['n_tickers']} → NEW {stats_new['n_tickers']}  "
          f"({'+' if stats_new['n_tickers'] > stats_old['n_tickers'] else ''}"
          f"{stats_new['n_tickers'] - stats_old['n_tickers']})")
    print(f"    Trades:    OLD {stats_old['n']} → NEW {stats_new['n']}")

    if improvements >= 3:
        print(f"\n  → NEW S3 WINS ({improvements}/4 metrics better)")
        print(f"    Per-fold optimization improves asset selection")
    elif improvements == 2:
        print(f"\n  → MIXED ({improvements}/4) — per-fold optimization has marginal impact")
    else:
        print(f"\n  → OLD S3 WINS ({4-improvements}/4 metrics better)")
        print(f"    Per-fold optimization did NOT improve — stick with base configs")

    # Per-ticker winners
    old_wins = new_wins = ties = 0
    for tk in common_tickers:
        ot = [t for t in old_trades if t['ticker'] == tk]
        nt = [t for t in new_trades if t['ticker'] == tk]
        if not ot and not nt: continue
        o_avg = np.mean([t['pnl'] for t in ot]) if ot else 0
        n_avg = np.mean([t['pnl'] for t in nt]) if nt else 0
        if abs(n_avg - o_avg) < 0.002:
            ties += 1
        elif n_avg > o_avg:
            new_wins += 1
        else:
            old_wins += 1
    print(f"\n  Per-ticker winners (common {len(common_tickers)}): "
          f"OLD={old_wins}, NEW={new_wins}, TIE={ties}")


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS (2x2)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('OLD vs NEW S3 Strategy Comparison', fontsize=16, fontweight='bold', y=0.98)


# ── Chart 1: Equity curves ──
ax1 = axes[0, 0]
for label, trades, n_tk, color in [
    ('OLD', old_trades, len(old_configs), OLD_COLOR),
    ('NEW', new_trades, len(new_configs), NEW_COLOR),
]:
    if not trades:
        continue
    df_t = pd.DataFrame(trades).sort_values('date')
    equity = STARTING_CAPITAL
    curve = [{'date': pd.Timestamp(TEST_START).date(), 'equity': equity}]
    for _, row in df_t.iterrows():
        alloc = equity / max(1, n_tk)
        equity += alloc * row['pnl']
        curve.append({'date': row['date'], 'equity': equity})
    cdf = pd.DataFrame(curve)
    final_ret = (cdf['equity'].iloc[-1] / STARTING_CAPITAL - 1) * 100
    st = stats_old if label == 'OLD' else stats_new
    ax1.plot(pd.to_datetime(cdf['date']), cdf['equity'] / 1000,
             label=f'{label} S3 ({st["n"]}t, avg {st["avg"]:+.2%}, ret {final_ret:+.0f}%)',
             color=color, linewidth=2)

ax1.axhline(y=STARTING_CAPITAL/1000, color='gray', linestyle='--', alpha=0.3)
ax1.axvline(x=pd.Timestamp(HALF2_START), color='red', linestyle=':', alpha=0.4)
ax1.set_ylabel('Equity ($K)', fontsize=11)
ax1.set_title('Equity Curves ($200K start)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


# ── Chart 2: Cumulative Portfolio Return ──
ax2 = axes[0, 1]
for label, trades, n_tk, color in [
    ('OLD', old_trades, len(old_configs), OLD_COLOR),
    ('NEW', new_trades, len(new_configs), NEW_COLOR),
]:
    if not trades:
        continue
    df_t = pd.DataFrame(trades).sort_values('date')
    cum = (df_t['pnl'] / max(1, n_tk)).cumsum() * 100
    dates = pd.to_datetime(df_t['date'].values)
    ax2.plot(dates, cum.values, label=f'{label} S3 ({cum.iloc[-1]:+.1f}%)',
             color=color, linewidth=2)

ax2.axhline(y=0, color='gray', linewidth=0.5)
ax2.axvline(x=pd.Timestamp(HALF2_START), color='red', linestyle=':', alpha=0.4)
ax2.set_ylabel('Portfolio Return (%)', fontsize=11)
ax2.set_title('Cumulative Portfolio Return', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


# ── Chart 3: Per-ticker avg PnL comparison ──
ax3 = axes[1, 0]
ticker_data = []
for tk in all_tickers:
    ot = [t['pnl'] for t in old_trades if t['ticker'] == tk]
    nt = [t['pnl'] for t in new_trades if t['ticker'] == tk]
    if ot or nt:
        ticker_data.append({
            'ticker': tk,
            'old_avg': np.mean(ot) * 100 if ot else 0,
            'new_avg': np.mean(nt) * 100 if nt else 0,
            'in_old': len(ot) > 0,
            'in_new': len(nt) > 0,
        })

if ticker_data:
    tdf = pd.DataFrame(ticker_data).sort_values('old_avg', ascending=False)
    x = np.arange(len(tdf))
    bar_w = 0.35
    ax3.bar(x - bar_w/2, tdf['old_avg'], bar_w, label='OLD S3', color=OLD_COLOR, alpha=0.85)
    ax3.bar(x + bar_w/2, tdf['new_avg'], bar_w, label='NEW S3', color=NEW_COLOR, alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(tdf['ticker'], fontsize=9, rotation=45, ha='right')
    ax3.axhline(y=0, color='gray', linewidth=0.5)
    ax3.set_ylabel('Avg PnL per Trade (%)', fontsize=11)
    ax3.set_title('Per-Ticker Avg PnL: OLD vs NEW', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis='y')


# ── Chart 4: Monthly avg PnL comparison ──
ax4 = axes[1, 1]
if months:
    x = np.arange(len(months))
    bar_w = 0.35
    o_vals = [stats_old.get('monthly', {}).get(m, {}).get('avg', 0) * 100 for m in months]
    n_vals = [stats_new.get('monthly', {}).get(m, {}).get('avg', 0) * 100 for m in months]
    bars_o = ax4.bar(x - bar_w/2, o_vals, bar_w, label='OLD S3', color=OLD_COLOR, alpha=0.85)
    bars_n = ax4.bar(x + bar_w/2, n_vals, bar_w, label='NEW S3', color=NEW_COLOR, alpha=0.85)
    for bars in [bars_o, bars_n]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 0.01:
                ax4.text(bar.get_x() + bar.get_width()/2, h + (0.05 if h >= 0 else -0.15),
                         f'{h:.1f}', ha='center', va='bottom', fontsize=8)
    ax4.set_xticks(x)
    ax4.set_xticklabels([m[-2:] for m in months], fontsize=10)
    ax4.axhline(y=0, color='gray', linewidth=0.5)
    ax4.set_ylabel('Avg PnL per Trade (%)', fontsize=11)
    ax4.set_title('Monthly Avg PnL: OLD vs NEW', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, framealpha=0.9)
    ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/tmp/backtest_s3_old_vs_new.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n  Charts saved to /tmp/backtest_s3_old_vs_new.png")


# ── Final summary ──
print()
print("=" * W)
print(f"  OLD: {len(old_configs)} tickers, {len(old_trades)} trades, avg {stats_old['avg']:+.2%}")
print(f"  NEW: {len(new_configs)} tickers, {len(new_trades)} trades, avg {stats_new['avg']:+.2%}")
print(f"  Config changes: {n_changed}/{len(common_tickers)} common tickers, "
      f"+{len(new_only)} gained, -{len(old_only)} dropped")
print("=" * W)
