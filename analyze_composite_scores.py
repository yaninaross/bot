"""
analyze_composite_scores.py — Selective Composite Analysis vs S3 Benchmark
==========================================================================
Loads per-ticker configs from BQ (WF v12), loads seed scores from GCS,
simulates trades for those specific configs only, then tests composite
agreement methods as selective portfolio filters.

Tables produced:
  A: S3 Benchmark (DEPLOY_TICKERS only)
  B: STATIC + composite gates (selective portfolios)
  C: ADAPTIVE + composite gates (selective portfolios)
  D: Methods that beat benchmark summary

Colab-ready (no if __name__). Uses limit_entry mode only (S3 strategy).
Configs loaded from BQ wf_optimization_results (robust_wf_v12).
"""

import os, io, gc, sys, subprocess, warnings, ctypes, pickle
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

try:
    from google.cloud import storage as gcs_storage
    from google.cloud import bigquery
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "google-cloud-storage", "google-cloud-bigquery", "pyarrow", "gcsfs", "--quiet"])
    from google.cloud import storage as gcs_storage
    from google.cloud import bigquery

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"
RESULTS_DIR = "Algos/Results_scored_20260703"
RESULTS_FALLBACK = "Algos/Results_all_v2"
OHLCV_DIR = "Data/Inference"
INFERENCE_DIR = "Data/Inference_v2"

PST = pytz.timezone('America/Los_Angeles')
SLIPPAGE_PCT = 0.0015
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

SCORE_EXPERIMENTS = [152, 160, 175, 185, 200, 201, 202, 203, 204, 205]
FULL_SCORED_UNIVERSE = [
    'ACHR','ALAB','AMD','APLD','ASTS','AVAV','BBAI','BE','BGDE','BHP',
    'BKSY','BTBT','CAT','CCJ','CEG','CIFR','CLSK','COIN','COP','CORZ',
    'CVX','DUK','EQIX','ETN','FCEL','FCX','FLNC','GD','GE','GEV',
    'HOOD','HUT','IREN','JOBY','KEEL','KTOS','LHX','LIN','LMT','LUNR',
    'MARA','MNTS','MOD','MP','MSTR','MU','MVIS','NBIS','NEE','NNDM',
    'NOC','NVDA','OKLO','PL','PLTR','PLUG','POWL','RCAT','RDW','RIOT',
    'RKLB','RTX','SMCI','SMR','SOUN','SPIR','TSLA','VOYG','VRT','VST',
    'WULF','XOM',
]
EXP_TEACHERS = {}
DEPLOY_TICKERS = ['APLD','ASTS','CIFR','CLSK','FLNC','HUT','KEEL','PL','RCAT','RIOT']
EXCLUDE_TICKERS = {'SMLR'}

TRAIN_TEACHERS_13 = ['BBAI','RCAT','LUNR','ASTS','ACHR','MVIS',
                     'WULF','CIFR','IREN','KEEL','HUT','CLSK','APLD']
TEACHERS_31 = ['CIFR','IREN','KEEL','HUT','APLD','WULF','RIOT','CLSK',
               'BTBT','ALAB','MARA','RCAT','ASTS','LUNR','BBAI','RKLB',
               'SPIR','BKSY','MNTS','PLUG','BE','FCEL','FLNC','OKLO',
               'SMR','MVIS','BGDE','RDW','PL','VOYG','SMLR']
EXP_TEACHER_SETS = {
    152: set(TRAIN_TEACHERS_13), 160: set(TRAIN_TEACHERS_13),
    175: set(TRAIN_TEACHERS_13), 185: set(TEACHERS_31),
    200: set(TEACHERS_31), 201: set(TEACHERS_31), 202: set(TEACHERS_31),
    203: set(TEACHERS_31), 204: set(TEACHERS_31), 205: set(TEACHERS_31),
}

DATA_START = '2025-12-01'
TEST_START = '2026-04-01'
TEST_END   = '2026-07-03'

ENTRY_MIN = 375
EXIT_TIME_MIN = 780
MIN_FAMILY_AGREE = 3
MIN_SEEDS = 3
MAX_VA_OPT_RATIO = 3.0
DEPLOY_THRESHOLD = 0.018
GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']

CACHE_FILE = '/tmp/composite_cache_v3.pkl'
CACHE_VERSION = f"{RESULTS_DIR}|{TEST_END}|{DATA_START}|{TEST_START}|bq_v12_limit_entry|exp{'_'.join(str(e) for e in SCORE_EXPERIMENTS)}"

gcs_client = gcs_storage.Client()
bq_client = bigquery.Client()


# ==============================================================================
# HELPERS
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df


def load_ohlcv(ticker):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for dir_path in [INFERENCE_DIR, OHLCV_DIR]:
        for suffix in ['_master_inference.parquet', '_1m_full_historical_data.parquet',
                        '_ohlcv_1min_pst.parquet']:
            uri = f"gs://{BUCKET_NAME}/{dir_path}/{clean}{suffix}"
            try:
                df = pd.read_parquet(uri)
                df = df.reset_index()
                dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                if dt_cols:
                    df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
                else:
                    df['Datetime'] = pd.to_datetime(df[df.columns[0]], unit='ms', utc=True)
                df = df.set_index('Datetime').sort_index()
                df = df[~df.index.duplicated(keep='first')]
                std_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
                lower_map = {k.lower():v for k,v in std_map.items()}
                df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
                start = pd.Timestamp(DATA_START, tz='UTC') - pd.Timedelta(days=5)
                end = pd.Timestamp(TEST_END, tz='UTC') + pd.Timedelta(days=2)
                df = df[(df.index >= start) & (df.index <= end)]
                df = heal_data(df)
                df.index = df.index.tz_convert(PST)
                return df
            except Exception:
                continue
    return None


def load_seed_scores(ticker, experiment, verbose=False):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    all_results = {}
    sources = []
    for results_dir in [RESULTS_FALLBACK, RESULTS_DIR]:
        prefix = f"{results_dir}/Universal_Exp{experiment}_XSECTOR/{clean}/Seed_"
        blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=prefix))
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
        sources.append("FALLBACK" if results_dir == RESULTS_FALLBACK else "NEW")
    if all_results and verbose:
        src = "+".join(sources)
        first = list(all_results.values())[0]
        print(f"      {ticker}/Exp{experiment}: {len(all_results)} seeds [{src}]")
    return all_results


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
    return 'other'


def exp_to_prob_key(exp_str):
    """Convert BQ experiment name to daily_probs key int.
    WF family 200 (folds 200-205) maps to aggregated key 200."""
    base = int(exp_str.split('_')[0])
    if 200 <= base <= 205:
        return 200
    return base


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


# ==============================================================================
# GATE FUNCTIONS
# ==============================================================================
def apply_config_for_dates(config, ticker, test_dates, daily_probs, daily_trades,
                           active_experiments, min_family_agree=MIN_FAMILY_AGREE,
                           min_families_present=2):
    if config is None:
        return []
    trades = []
    entry_time_str = config['entry_time']
    tp, sl = config['tp'], config['sl']
    prob_thresh = config['prob_thresh']
    for d in test_dates:
        signals = 0
        total_families = 0
        pnl_sum = 0
        pnl_count = 0
        for exp in active_experiments:
            prob_key = (ticker, exp)
            trade_key = (ticker, exp, entry_time_str, tp, sl)
            if prob_key not in daily_probs or trade_key not in daily_trades:
                continue
            total_families += 1
            if d in daily_probs[prob_key] and daily_probs[prob_key][d] >= prob_thresh:
                signals += 1
                if d in daily_trades[trade_key]:
                    pnl_sum += daily_trades[trade_key][d]
                    pnl_count += 1
        if total_families >= min_families_present and \
           signals >= min(min_family_agree, (total_families // 2 + 1)):
            if pnl_count > 0:
                trades.append({'date': d, 'ticker': ticker,
                               'pnl': pnl_sum / pnl_count, 'n_exps': signals})
    return trades


def apply_config_pct_gate(config, ticker, test_dates, daily_probs, daily_trades,
                          active_experiments, pct_threshold=0.60,
                          min_families_present=2):
    if config is None:
        return []
    trades = []
    entry_time_str = config['entry_time']
    tp, sl = config['tp'], config['sl']
    prob_thresh = config['prob_thresh']
    for d in test_dates:
        signals = 0
        total_families = 0
        pnl_sum = 0
        pnl_count = 0
        for exp in active_experiments:
            prob_key = (ticker, exp)
            trade_key = (ticker, exp, entry_time_str, tp, sl)
            if prob_key not in daily_probs or trade_key not in daily_trades:
                continue
            total_families += 1
            if d in daily_probs[prob_key] and daily_probs[prob_key][d] >= prob_thresh:
                signals += 1
                if d in daily_trades[trade_key]:
                    pnl_sum += daily_trades[trade_key][d]
                    pnl_count += 1
        if total_families >= min_families_present and \
           total_families > 0 and signals / total_families >= pct_threshold:
            if pnl_count > 0:
                trades.append({'date': d, 'ticker': ticker,
                               'pnl': pnl_sum / pnl_count, 'n_exps': signals})
    return trades


def apply_config_seed_gate(config, ticker, test_dates, daily_seed_probs, daily_trades,
                           active_experiments, seed_pct_threshold=0.60,
                           min_seeds_present=15):
    if config is None:
        return []
    trades = []
    entry_time_str = config['entry_time']
    tp, sl = config['tp'], config['sl']
    prob_thresh = config['prob_thresh']
    seed_keys = [(ticker, exp, sid) for (t, exp, sid) in daily_seed_probs
                 if t == ticker and exp in active_experiments]
    if not seed_keys:
        return []
    exp_set = set(k[1] for k in seed_keys)
    for d in test_dates:
        total_seeds = 0
        agree_seeds = 0
        for key in seed_keys:
            sprobs = daily_seed_probs[key]
            if d in sprobs:
                total_seeds += 1
                if sprobs[d] >= prob_thresh:
                    agree_seeds += 1
        if total_seeds >= min_seeds_present and \
           agree_seeds / total_seeds >= seed_pct_threshold:
            pnl_sum = 0
            pnl_count = 0
            for exp in exp_set:
                trade_key = (ticker, exp, entry_time_str, tp, sl)
                if trade_key in daily_trades and d in daily_trades[trade_key]:
                    pnl_sum += daily_trades[trade_key][d]
                    pnl_count += 1
            if pnl_count > 0:
                trades.append({'date': d, 'ticker': ticker,
                               'pnl': pnl_sum / pnl_count, 'n_exps': agree_seeds})
    return trades


def apply_config_teacher_gate(config, ticker, test_dates, daily_probs, daily_trades,
                              active_experiments, exp_teacher_sets,
                              pct_threshold=0.60, min_families_present=1):
    if config is None:
        return []
    relevant_exps = [e for e in active_experiments if ticker in exp_teacher_sets.get(e, set())]
    if not relevant_exps:
        relevant_exps = active_experiments
    trades = []
    entry_time_str = config['entry_time']
    tp, sl = config['tp'], config['sl']
    prob_thresh = config['prob_thresh']
    for d in test_dates:
        signals = 0
        total_families = 0
        pnl_sum = 0
        pnl_count = 0
        for exp in relevant_exps:
            prob_key = (ticker, exp)
            trade_key = (ticker, exp, entry_time_str, tp, sl)
            if prob_key not in daily_probs or trade_key not in daily_trades:
                continue
            total_families += 1
            if d in daily_probs[prob_key] and daily_probs[prob_key][d] >= prob_thresh:
                signals += 1
                if d in daily_trades[trade_key]:
                    pnl_sum += daily_trades[trade_key][d]
                    pnl_count += 1
        if total_families >= min_families_present and \
           total_families > 0 and signals / total_families >= pct_threshold:
            if pnl_count > 0:
                trades.append({'date': d, 'ticker': ticker,
                               'pnl': pnl_sum / pnl_count, 'n_exps': signals})
    return trades


def summarize(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    daily_rets = df.groupby('date')['pnl'].mean().sort_index()
    equity = (1 + daily_rets).cumprod()
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    june = df[df['month'] == '2026-06']
    june_daily = june.groupby('date')['pnl'].mean().sort_index() if len(june) else pd.Series(dtype=float)
    june_total = (1 + june_daily).prod() - 1 if len(june_daily) else 0
    return {
        'label': label,
        'n': len(df), 'total': equity.iloc[-1] - 1, 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': max_dd,
        'n_days': len(daily_rets),
        'j_n': len(june), 'j_total': june_total,
        'j_avg': june['pnl'].mean() if len(june) else 0,
        'j_wr': (june['pnl'] > 0).mean() if len(june) else 0,
    }


# ==============================================================================
# CACHE CHECK — skip Steps 0-3 if valid cache exists
# ==============================================================================
W = 120
print("=" * W)
print(f"  COMPOSITE SCORE ANALYSIS (isolated from deep optimization)")
print("=" * W)

_cache_hit = False
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        if cache.get('version') == CACHE_VERSION:
            daily_probs = cache['daily_probs']
            daily_seed_probs = cache['daily_seed_probs']
            daily_trades = cache['daily_trades']
            static_configs = cache['static_configs']
            all_tickers = cache['all_tickers']
            ACTIVE_EXPERIMENTS = cache['ACTIVE_EXPERIMENTS']
            all_trading_dates = cache['all_trading_dates']
            all_available_dates = cache['all_available_dates']
            static_opt_dates = cache['static_opt_dates']
            s3_tickers = cache.get('s3_tickers', [])
            print(f"\n  ✓ CACHE HIT — loaded from {CACHE_FILE}")
            print(f"    {len(all_tickers)} tickers, {len(daily_probs)} prob pairs, "
                  f"{len(daily_trades)} trade series, {len(all_trading_dates)} test days")
            _cache_hit = True
            del cache
        else:
            print(f"\n  Cache stale (config changed), recomputing...")
    except Exception as e:
        print(f"\n  Cache load failed ({e}), recomputing...")

if not _cache_hit:

    # ==============================================================================
    # STEP 0: DISCOVER TICKERS + LOAD CONFIGS FROM BQ
    # ==============================================================================
    print(f"\n  STEP 0: Discovering scored tickers + loading BQ configs...")
    coverage = {}
    for exp in SCORE_EXPERIMENTS:
        tickers_found = set()
        for results_dir in [RESULTS_DIR, RESULTS_FALLBACK]:
            prefix = f"{results_dir}/Universal_Exp{exp}_XSECTOR/"
            blobs = gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=prefix)
            for b in blobs:
                parts = b.name.replace(prefix, '').split('/')
                if len(parts) >= 2 and parts[0]:
                    raw = parts[0]
                    clean = {'BITF': 'KEEL', 'MIGI': 'BGDE'}.get(raw, raw)
                    tickers_found.add(clean)
        EXP_TEACHERS[exp] = sorted(tickers_found)

        test_ticker = DEPLOY_TICKERS[0]
        seeds = load_seed_scores(test_ticker, exp, verbose=True)
        if seeds:
            first_seed = list(seeds.values())[0]
            min_d = first_seed.index.min().date()
            max_d = first_seed.index.max().date()
            covers_june = max_d >= pd.Timestamp('2026-06-01').date()
            coverage[exp] = {'min': min_d, 'max': max_d, 'n_seeds': len(seeds),
                             'covers_june': covers_june}
            status = "✓" if covers_june else "✗"
            print(f"    Exp {exp}: {min_d} → {max_d}  ({len(seeds)} seeds)  "
                  f"{len(tickers_found)} tickers  {status}")
        else:
            coverage[exp] = None
            print(f"    Exp {exp}: NO SEED DATA  ({len(tickers_found)} tickers)")

    ACTIVE_EXPERIMENTS = [e for e, c in coverage.items() if c and c['covers_june']]
    if not ACTIVE_EXPERIMENTS:
        raise SystemExit("No June score data available.")
    print(f"    Active experiments: {ACTIVE_EXPERIMENTS}")

    all_scored = set()
    for exp in ACTIVE_EXPERIMENTS:
        all_scored.update(EXP_TEACHERS[exp])
    all_scored -= EXCLUDE_TICKERS
    all_tickers = sorted(all_scored)
    print(f"    Total scored universe: {len(all_tickers)} tickers")

    # Load BQ configs: individual seeds → seed groups → family agreement
    # (matches backtest_strategy_csv.py logic exactly)
    q = """
    SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           trail_start_mult, trail_end_mult, grace_minutes,
           dip_pct, max_wait_minutes, vol_filter_pct,
           val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
           val_b_avg_pnl, opt_avg_pnl, is_ensemble
    FROM `ys-time-series.scoring_logs.wf_optimization_results`
    WHERE model_version = 'robust_wf_v12'
      AND survived_a = TRUE
      AND val_a_count >= 3
      AND val_a_avg_pnl > 0
    """
    raw_bq = bq_client.query(q).to_dataframe()
    print(f"    {len(raw_bq)} raw BQ rows loaded")

    raw_bq = raw_bq[raw_bq['opt_avg_pnl'] > 0].copy()
    raw_bq = raw_bq[raw_bq['val_a_avg_pnl'] / raw_bq['opt_avg_pnl'] <= MAX_VA_OPT_RATIO]
    print(f"    {len(raw_bq)} after overfit filter")

    indiv = raw_bq[raw_bq['is_ensemble'] == False].copy()
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
    print(f"    {len(sa_limit)} limit_entry seed groups (n_seeds >= {MIN_SEEDS})")

    sa_limit['score'] = (sa_limit['mean_va_pnl'] ** 2) * sa_limit['agree_18']
    sa_limit['exp_fam'] = sa_limit['experiment'].apply(assign_family)
    sa_fam = sa_limit[sa_limit['exp_fam'] != 'other'].copy()

    best_per_fam = sa_fam.loc[sa_fam.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    best_per_fam = best_per_fam[best_per_fam['mean_va_pnl'] >= DEPLOY_THRESHOLD]

    bq_agreement = {}
    for tk in best_per_fam['ticker'].unique():
        tk_rows = best_per_fam[best_per_fam['ticker'] == tk]
        n_fams = len(tk_rows)
        n_pos = int((tk_rows['vb_pnl'] > 0).sum())
        bq_agreement[tk] = {
            'n_fams': n_fams, 'n_pos': n_pos,
            'majority_pos': n_pos > n_fams / 2 and n_fams >= 2,
        }

    s3_tickers = sorted([tk for tk, ag in bq_agreement.items() if ag['majority_pos']])
    print(f"    Family agreement: {len(s3_tickers)} tickers pass S3 gate")
    print(f"    S3 tickers: {', '.join(s3_tickers)}")

    sa_all = sa_limit[sa_limit['ticker'].isin(all_tickers)].copy()
    sa_all['score'] = (sa_all['mean_va_pnl'] ** 2) * sa_all['agree_18']
    best_per_tk = sa_all.loc[sa_all.groupby('ticker')['score'].idxmax()]
    best_per_tk = best_per_tk[best_per_tk['mean_va_pnl'] >= DEPLOY_THRESHOLD]

    static_configs = {}
    n_configured = 0
    for ticker in all_tickers:
        tk_rows = best_per_tk[best_per_tk['ticker'] == ticker]
        if tk_rows.empty:
            static_configs[ticker] = None
            continue
        best = tk_rows.iloc[0]
        entry_min = int(best['entry_minute']) if pd.notna(best['entry_minute']) else ENTRY_MIN
        static_configs[ticker] = {
            'entry_time': str(entry_min),
            'prob_thresh': float(best['prob_thresh']),
            'tp': float(best['tp_pct']),
            'sl': float(best['sl_pct']),
            'avg_pnl': float(best['mean_va_pnl']),
            'dip_pct': float(best['dip_pct']),
            'max_wait_minutes': float(best['max_wait_minutes']),
            'entry_minute': entry_min,
            'experiment': best['experiment'],
            'n_trades': 0,
        }
        n_configured += 1
        cfg = static_configs[ticker]
        s3_flag = "S3" if ticker in s3_tickers else "  "
        print(f"    {s3_flag} {ticker}: prob>{cfg['prob_thresh']:.2f} TP={cfg['tp']:.0%} "
              f"SL={cfg['sl']:.0%} dip={cfg['dip_pct']:.1%} wait={cfg['max_wait_minutes']:.0f}m "
              f"avgPnL={cfg['avg_pnl']:.2%} ({best['experiment']})")
    print(f"    {n_configured}/{len(all_tickers)} tickers configured from BQ")

    del raw_bq, indiv, sa, sa_limit, sa_fam, best_per_fam, sa_all, best_per_tk
    _release_memory()


    # ==============================================================================
    # STEP 1: LOAD OHLCV + SEED SCORES
    # ==============================================================================
    print(f"\n  STEP 1: Loading OHLCV + score CSVs for {len(all_tickers)} tickers...")

    ohlcv_cache = {}
    for ticker in all_tickers:
        df = load_ohlcv(ticker)
        if df is not None:
            ohlcv_cache[ticker] = df
            print(f"    {ticker}: {len(df):>8,} OHLCV bars")

    scores_cache = {}
    for exp in ACTIVE_EXPERIMENTS:
        exp_count = 0
        for ticker in all_tickers:
            if ticker not in EXP_TEACHERS.get(exp, []):
                continue
            seeds = load_seed_scores(ticker, exp)
            if seeds:
                scores_cache[(ticker, exp)] = seeds
                exp_count += 1
        print(f"    Exp {exp}: loaded {exp_count} tickers")
    _release_memory()


    # ==============================================================================
    # STEP 2: PRE-COMPUTE DAILY SIGNALS + TRADE OUTCOMES (BQ configs only)
    # ==============================================================================
    print(f"\n  STEP 2: Pre-computing daily signals + trade outcomes...")

    daily_probs = {}
    daily_seed_probs = {}
    all_score_dates_by_ticker = {}

    for (ticker, exp), seeds_dict in scores_cache.items():
        cfg = static_configs.get(ticker)
        entry_min = cfg['entry_minute'] if cfg else ENTRY_MIN
        avg_probs = None
        n_seeds = 0
        for seed_id, sdf in seeds_dict.items():
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
            daily_seed_probs[(ticker, exp, seed_id)] = daily_prob
            if avg_probs is None:
                avg_probs = daily_prob.copy()
            else:
                avg_probs = avg_probs.add(daily_prob, fill_value=0)
            n_seeds += 1
        if n_seeds == 0 or avg_probs is None:
            continue
        avg_probs = avg_probs / n_seeds
        daily_probs[(ticker, exp)] = avg_probs
        if ticker not in all_score_dates_by_ticker:
            all_score_dates_by_ticker[ticker] = set()
        all_score_dates_by_ticker[ticker].update(avg_probs.index)

    print(f"    Computed probs for {len(daily_probs)} (ticker, exp) pairs "
          f"(using entry_minute bar, not daily max)")
    del scores_cache

    # Pre-simulate trades using limit_entry mode (matching production)
    ohlcv_arrays = {}
    for ticker in ohlcv_cache:
        arr = precompute_ohlcv_arrays(ohlcv_cache[ticker], DATA_START, TEST_END)
        if arr is not None:
            ohlcv_arrays[ticker] = arr
    print(f"    Precomputed OHLCV arrays for {len(ohlcv_arrays)} tickers")

    ticker_trades = {}
    n_filled = 0
    n_unfilled = 0
    for ticker in all_tickers:
        cfg = static_configs.get(ticker)
        if cfg is None or ticker not in ohlcv_arrays:
            continue
        ohlcv = ohlcv_arrays[ticker]
        tp, sl = cfg['tp'], cfg['sl']
        dip_pct = cfg['dip_pct']
        max_wait = cfg['max_wait_minutes']
        entry_min = cfg['entry_minute']
        outcomes = {}
        unique_dates = np.unique(ohlcv['dates'])
        for d_int in unique_dates:
            d_str = str(int(d_int))
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
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
                n_filled += 1
            else:
                n_unfilled += 1
        if outcomes:
            ticker_trades[(ticker, cfg['entry_time'], tp, sl)] = outcomes
        print(f"    {ticker}: {len(outcomes)} filled (limit_entry, "
              f"dip={dip_pct:.1%} wait={max_wait:.0f}m)")

    daily_trades = {}
    for (ticker, exp) in daily_probs:
        cfg = static_configs.get(ticker)
        if cfg is None:
            continue
        key_t = (ticker, cfg['entry_time'], cfg['tp'], cfg['sl'])
        if key_t in ticker_trades:
            daily_trades[(ticker, exp, cfg['entry_time'], cfg['tp'], cfg['sl'])] = ticker_trades[key_t]

    print(f"    Pre-computed {len(daily_trades)} outcome series "
          f"({n_filled:,} filled, {n_unfilled:,} unfilled)")
    del ticker_trades
    del ohlcv_arrays
    del ohlcv_cache
    _release_memory()

    # ── STEP 2.5: Aggregate walkforward families ──
    # 200-205 are WF folds of one family — collapse to single '200' vote
    # using best-fold (max ensemble prob) per ticker per date.
    WF_FAMILIES = {200: [201, 202, 203, 204, 205]}
    for fam_base, member_folds in WF_FAMILIES.items():
        print(f"\n  Aggregating WF family {fam_base} (folds {member_folds})...")
        n_agg = 0
        for ticker in all_tickers:
            base_key = (ticker, fam_base)
            base_probs = daily_probs.get(base_key)
            fold_probs = {fam_base: base_probs} if base_probs is not None else {}
            for fold in member_folds:
                fp = daily_probs.get((ticker, fold))
                if fp is not None:
                    fold_probs[fold] = fp
            if len(fold_probs) <= 1:
                continue
            all_dates = set()
            for ps in fold_probs.values():
                all_dates.update(ps.index)
            best_prob = {}
            best_fold = {}
            for d in all_dates:
                bp, bf = -1, fam_base
                for fold, ps in fold_probs.items():
                    if d in ps and ps[d] > bp:
                        bp = ps[d]
                        bf = fold
                best_prob[d] = bp
                best_fold[d] = bf
            daily_probs[base_key] = pd.Series(best_prob)
            dominant_fold = Counter(best_fold.values()).most_common(1)[0][0]
            for sid in list({k[2] for k in daily_seed_probs if k[0] == ticker and k[1] == dominant_fold}):
                daily_seed_probs[(ticker, fam_base, sid)] = daily_seed_probs[(ticker, dominant_fold, sid)]
            for fold in member_folds:
                daily_probs.pop((ticker, fold), None)
                for sid in list({k[2] for k in daily_seed_probs if k[0] == ticker and k[1] == fold}):
                    daily_seed_probs.pop((ticker, fold, sid), None)
                cfg = static_configs.get(ticker)
                if cfg:
                    daily_trades.pop((ticker, fold, cfg['entry_time'], cfg['tp'], cfg['sl']), None)
            n_agg += 1
        ACTIVE_EXPERIMENTS = [e for e in ACTIVE_EXPERIMENTS if e not in member_folds]
        print(f"    Aggregated {n_agg} tickers → family {fam_base}. "
              f"Active experiments: {ACTIVE_EXPERIMENTS}")

    # Compute date lists
    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt = pd.Timestamp(TEST_END).date()

    all_trading_dates = set()
    for outcomes in daily_trades.values():
        all_trading_dates.update(outcomes.keys())
    all_trading_dates = sorted([d for d in all_trading_dates
                                if d >= test_start_dt and d <= test_end_dt])
    all_available_dates = sorted(set().union(*(set(o.keys()) for o in daily_trades.values())))

    static_opt_start = pd.Timestamp(DATA_START).date()
    static_opt_end = (pd.Timestamp(TEST_START) - pd.Timedelta(days=1)).date()
    static_opt_dates = [d for d in all_available_dates
                        if d >= static_opt_start and d <= static_opt_end]

    print(f"    {len(all_trading_dates)} test days, {len(all_available_dates)} total days, "
          f"{len(static_opt_dates)} pre-test days")

    # ── Save cache ──
    print(f"\n  Saving cache to {CACHE_FILE}...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({
            'version': CACHE_VERSION,
            'daily_probs': daily_probs,
            'daily_seed_probs': daily_seed_probs,
            'daily_trades': daily_trades,
            'static_configs': static_configs,
            'all_tickers': all_tickers,
            'ACTIVE_EXPERIMENTS': ACTIVE_EXPERIMENTS,
            'all_trading_dates': all_trading_dates,
            'all_available_dates': all_available_dates,
            'static_opt_dates': static_opt_dates,
            's3_tickers': s3_tickers,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    cache_mb = os.path.getsize(CACHE_FILE) / 1e6
    print(f"    ✓ Saved ({cache_mb:.0f} MB). Next run will skip Steps 0-2.")

REOPT_FREQ = 5
reopt_indices = list(range(0, len(all_trading_dates), REOPT_FREQ))


# ==============================================================================
# STEP 4: SELECTIVE COMPOSITE ANALYSIS vs S3 BENCHMARK
# ==============================================================================
print()
print("=" * W)
print(f"  STEP 4: SELECTIVE COMPOSITE ANALYSIS")
print("=" * W)
print(f"  Optimize on ALL {len(all_tickers)} tickers, then composite gate selects portfolio.")
print(f"  Only tickers with avg PnL >= {DEPLOY_THRESHOLD:.1%} survive.")
print(f"  Goal: beat S3 benchmark avg PnL AND be positive in June.")

def selective_portfolio(trades_list, label, min_avg=DEPLOY_THRESHOLD):
    if not trades_list:
        return None
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    per_ticker = []
    for ticker in sorted(df['ticker'].unique()):
        tt = df[df['ticker'] == ticker]
        june = tt[tt['month'] == '2026-06']
        avg = tt['pnl'].mean()
        if avg < min_avg:
            continue
        per_ticker.append({
            'ticker': ticker, 'trades': len(tt), 'avg': avg,
            'total': tt['pnl'].sum(), 'wr': (tt['pnl'] > 0).mean(),
            'j_n': len(june),
            'j_avg': june['pnl'].mean() if len(june) else 0,
            'j_wr': (june['pnl'] > 0).mean() if len(june) else 0,
        })
    if not per_ticker:
        return None
    keep = set(r['ticker'] for r in per_ticker)
    filt = df[df['ticker'].isin(keep)]
    daily_rets = filt.groupby('date')['pnl'].mean().sort_index()
    equity = (1 + daily_rets).cumprod()
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    june_all = filt[filt['month'] == '2026-06']
    june_daily = june_all.groupby('date')['pnl'].mean().sort_index() if len(june_all) else pd.Series(dtype=float)
    june_total = (1 + june_daily).prod() - 1 if len(june_daily) else 0
    return {
        'label': label, 'per_ticker': per_ticker,
        'n': len(filt), 'total': equity.iloc[-1] - 1,
        'avg': filt['pnl'].mean(), 'dd': max_dd,
        'wr': (filt['pnl'] > 0).mean(),
        'n_days': len(daily_rets),
        'j_n': len(june_all),
        'j_avg': june_all['pnl'].mean() if len(june_all) else 0,
        'j_total': june_total,
        'n_tickers': len(per_ticker),
    }


def display_portfolio(ps, benchmark_avg=None):
    if ps is None:
        print(f"    (no tickers survive {DEPLOY_THRESHOLD:.1%} threshold)")
        return
    tag = ""
    if benchmark_avg is not None:
        if ps['avg'] >= benchmark_avg and ps['j_avg'] > 0:
            tag = " ★ BEATS BENCHMARK + JUNE POSITIVE"
        elif ps['avg'] >= benchmark_avg:
            tag = " ✓ beats benchmark avg"
    print(f"    Portfolio: {ps['n_tickers']} tickers, {ps['n']} trades, "
          f"{ps.get('n_days', '?')} days, "
          f"avg={ps['avg']:+.2%}, total={ps['total']:+.1%}, "
          f"WR={ps['wr']:.0%}, MaxDD={ps['dd']:+.1%}")
    print(f"    June: {ps['j_n']} trades, avg={ps['j_avg']:+.2%}, "
          f"total={ps['j_total']:+.1%}{tag}")
    print(f"    {'Ticker':<7} {'Trades':>6} {'AvgPnL':>8} {'TotPnL':>9} {'WR':>6}"
          f" │ {'JunT':>5} {'JunAvg':>8} {'JunWR':>6}")
    print(f"    {'─'*7} {'─'*6} {'─'*8} {'─'*9} {'─'*6}"
          f" │ {'─'*5} {'─'*8} {'─'*6}")
    for t in sorted(ps['per_ticker'], key=lambda x: x['avg'], reverse=True):
        jm = " ✓" if t['j_avg'] > 0 and t['j_n'] > 0 else ""
        print(f"    {t['ticker']:<7} {t['trades']:>6} {t['avg']:>+7.2%} "
              f"{t['total']:>+8.1%} {t['wr']:>5.0%}"
              f" │ {t['j_n']:>5} {t['j_avg']:>+7.2%} {t['j_wr']:>5.0%}{jm}")


def run_adaptive(gate_func, gate_args):
    all_trades = []
    for ticker in all_tickers:
        if static_configs[ticker] is None:
            continue
        base_thresh = static_configs[ticker]['prob_thresh']
        current_thresh = base_thresh
        recent_results = []
        for ri in reopt_indices:
            test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
            test_dates = all_trading_dates[ri:test_end_idx]
            if len(recent_results) >= 3:
                last10 = recent_results[-10:]
                precision = sum(1 for r in last10 if r > 0) / len(last10)
                if precision < 0.40:
                    current_thresh = min(0.90, current_thresh + 0.10)
                elif precision > 0.60:
                    current_thresh = max(0.30, current_thresh - 0.10)
            adj_config = dict(static_configs[ticker])
            adj_config['prob_thresh'] = current_thresh
            wt = gate_func(adj_config, ticker, test_dates, *gate_args)
            all_trades.extend(wt)
            recent_results.extend([t['pnl'] for t in wt])
    return all_trades


# ── TABLE A: S3 BENCHMARK (single-experiment signal, matches production) ──
print(f"\n  TABLE A: S3 BENCHMARK (production-matching single-experiment signal)")
print(f"    S3 tickers from family agreement: {', '.join(s3_tickers)}")
bench_trades = []
for ticker in s3_tickers:
    cfg = static_configs.get(ticker)
    if cfg is None:
        print(f"    {ticker}: no config")
        continue
    exp_key = exp_to_prob_key(cfg['experiment'])
    prob_dict = daily_probs.get((ticker, exp_key))
    if prob_dict is None:
        print(f"    {ticker}: no probs for {cfg['experiment']} (key={exp_key})")
        continue
    trade_key = (ticker, exp_key, cfg['entry_time'], cfg['tp'], cfg['sl'])
    trade_outcomes = daily_trades.get(trade_key)
    if trade_outcomes is None:
        print(f"    {ticker}: no trade outcomes")
        continue
    n_sig = 0
    for d in all_trading_dates:
        p = prob_dict.get(d) if isinstance(prob_dict, dict) else prob_dict[d] if d in prob_dict.index else None
        if p is None:
            continue
        if p >= cfg['prob_thresh']:
            if d in trade_outcomes:
                bench_trades.append({
                    'date': d, 'ticker': ticker,
                    'pnl': trade_outcomes[d], 'n_exps': 1
                })
                n_sig += 1
    deploy_flag = "DEPLOY" if ticker in DEPLOY_TICKERS else "      "
    print(f"    {deploy_flag} {ticker}: {n_sig} signals, "
          f"prob>{cfg['prob_thresh']:.2f} ({cfg['experiment']})")

bench = selective_portfolio(bench_trades, 'S3 Deployed', min_avg=-999)
display_portfolio(bench)

BENCHMARK_AVG = bench['avg'] if bench and bench['n'] > 0 else 0.025
print(f"\n    >>> S3 BENCHMARK: {BENCHMARK_AVG:+.2%} avg PnL to beat")


# ── TABLE B: STATIC + composite gates (selective) ──
print()
print("=" * W)
print(f"  TABLE B: STATIC STRATEGY + COMPOSITE GATES (selective portfolios)")
print("=" * W)
print(f"  Tickers with avg PnL >= {DEPLOY_THRESHOLD:.1%} after gate applied")

all_sweep = []

for min_agree in [1, 2, 3]:
    trades = []
    for ticker in all_tickers:
        trades.extend(apply_config_for_dates(
            static_configs[ticker], ticker, all_trading_dates,
            daily_probs, daily_trades, ACTIVE_EXPERIMENTS,
            min_family_agree=min_agree, min_families_present=2))
    label = f"STATIC agree={min_agree}"
    ps = selective_portfolio(trades, label)
    all_sweep.append(ps)
    print(f"\n  ── {label} ──")
    display_portfolio(ps, BENCHMARK_AVG)

for seed_pct in [0.40, 0.50, 0.60, 0.75]:
    trades = []
    for ticker in all_tickers:
        trades.extend(apply_config_seed_gate(
            static_configs[ticker], ticker, all_trading_dates, daily_seed_probs,
            daily_trades, ACTIVE_EXPERIMENTS, seed_pct_threshold=seed_pct,
            min_seeds_present=15))
    label = f"STATIC seed={seed_pct:.0%}/15"
    ps = selective_portfolio(trades, label)
    all_sweep.append(ps)
    print(f"\n  ── {label} ──")
    display_portfolio(ps, BENCHMARK_AVG)

# Weighted vote
exp_precision = {}
for ticker in all_tickers:
    for exp in ACTIVE_EXPERIMENTS:
        prob_key = (ticker, exp)
        if prob_key not in daily_probs:
            continue
        cfg = static_configs[ticker]
        if cfg is None:
            continue
        trade_key = (ticker, exp, cfg['entry_time'], cfg['tp'], cfg['sl'])
        if trade_key not in daily_trades:
            continue
        outcomes = daily_trades[trade_key]
        probs = daily_probs[prob_key]
        n_sig, n_prof = 0, 0
        for d in static_opt_dates:
            if d in probs and probs[d] >= cfg['prob_thresh'] and d in outcomes:
                n_sig += 1
                if outcomes[d] > 0:
                    n_prof += 1
        exp_precision[(ticker, exp)] = n_prof / n_sig if n_sig >= 3 else 0.5

weighted_trades = []
for ticker in all_tickers:
    cfg = static_configs[ticker]
    if cfg is None:
        continue
    for d in all_trading_dates:
        tw, aw, ps, pc = 0, 0, 0, 0
        for exp in ACTIVE_EXPERIMENTS:
            prob_key = (ticker, exp)
            trade_key = (ticker, exp, cfg['entry_time'], cfg['tp'], cfg['sl'])
            if prob_key not in daily_probs or trade_key not in daily_trades:
                continue
            w = exp_precision.get((ticker, exp), 0.5)
            tw += w
            if d in daily_probs[prob_key] and daily_probs[prob_key][d] >= cfg['prob_thresh']:
                aw += w
                if d in daily_trades[trade_key]:
                    ps += daily_trades[trade_key][d]
                    pc += 1
        if tw > 0 and aw / tw > 0.5 and pc > 0:
            weighted_trades.append({'date': d, 'ticker': ticker,
                                    'pnl': ps / pc, 'n_exps': 0})

label = "STATIC weighted>50%"
wps = selective_portfolio(weighted_trades, label)
all_sweep.append(wps)
print(f"\n  ── {label} ──")
display_portfolio(wps, BENCHMARK_AVG)


# ── TABLE C: ADAPTIVE + composite gates (selective) ──
print()
print("=" * W)
print(f"  TABLE C: ADAPTIVE THRESHOLD + COMPOSITE GATES (selective portfolios)")
print("=" * W)
print(f"  Dynamic prob_thresh: +0.10 when trailing precision < 40%, -0.10 when > 60%")

for min_agree in [1, 2, 3]:
    trades = run_adaptive(apply_config_for_dates,
        (daily_probs, daily_trades, ACTIVE_EXPERIMENTS, min_agree, 2))
    label = f"ADAPTIVE agree={min_agree}"
    ps = selective_portfolio(trades, label)
    all_sweep.append(ps)
    print(f"\n  ── {label} ──")
    display_portfolio(ps, BENCHMARK_AVG)

for seed_pct in [0.40, 0.50, 0.60, 0.75]:
    trades = run_adaptive(apply_config_seed_gate,
        (daily_seed_probs, daily_trades, ACTIVE_EXPERIMENTS, seed_pct, 15))
    label = f"ADAPTIVE seed={seed_pct:.0%}/15"
    ps = selective_portfolio(trades, label)
    all_sweep.append(ps)
    print(f"\n  ── {label} ──")
    display_portfolio(ps, BENCHMARK_AVG)


# ── TABLE D: METHODS THAT BEAT BENCHMARK ──
print()
print("=" * W)
print(f"  TABLE D: METHODS THAT BEAT S3 BENCHMARK ({BENCHMARK_AVG:+.2%} avg)")
print("=" * W)

valid = [ps for ps in all_sweep if ps is not None]
winners = [ps for ps in valid if ps['avg'] >= BENCHMARK_AVG]
winners.sort(key=lambda x: x['avg'], reverse=True)

if not winners:
    print(f"\n  No methods beat benchmark. Showing top 5 by avg PnL:")
    winners = sorted(valid, key=lambda x: x['avg'], reverse=True)[:5]

print(f"\n  {'Method':<26} │ {'#Tk':>3} {'Trd':>4} {'AvgPnL':>7} {'TotPnL':>8} "
      f"{'WR':>4} {'MaxDD':>7} │ {'JAvg':>6} {'JTot':>7} {'JunOK':>5} │ Tickers")
print(f"  {'─'*26} │ {'─'*3} {'─'*4} {'─'*7} {'─'*8} "
      f"{'─'*4} {'─'*7} │ {'─'*6} {'─'*7} {'─'*5} │ {'─'*30}")

for ps in winners:
    tickers = [t['ticker'] for t in sorted(ps['per_ticker'],
               key=lambda x: x['avg'], reverse=True)]
    tstr = ','.join(tickers[:8])
    if len(tickers) > 8:
        tstr += f"+{len(tickers)-8}"
    jok = "✓" if ps['j_avg'] > 0 else "✗"
    beat = "★" if ps['avg'] >= BENCHMARK_AVG and ps['j_avg'] > 0 else \
           "✓" if ps['avg'] >= BENCHMARK_AVG else " "
    print(f" {beat}{ps['label']:<26} │ {ps['n_tickers']:>3} {ps['n']:>4} "
          f"{ps['avg']:>+6.2%} {ps['total']:>+7.1%} "
          f"{ps['wr']:>3.0%} {ps['dd']:>+6.1%} │ "
          f"{ps['j_avg']:>+5.2%} {ps['j_total']:>+6.1%} {jok:>5} │ {tstr}")


# ── BEST RESULT DETAIL ──
best = None
for ps in valid:
    if ps['avg'] >= BENCHMARK_AVG and ps['j_avg'] > 0:
        if best is None or ps['avg'] > best['avg']:
            best = ps

if best is None:
    best_no_june = max(valid, key=lambda x: x['avg']) if valid else None
    if best_no_june:
        print(f"\n  No method beats benchmark AND is June-positive.")
        print(f"  Best avg overall: {best_no_june['label']} at {best_no_june['avg']:+.2%}")
else:
    print(f"\n  >>> BEST: {best['label']}")
    print(f"      {best['avg']:+.2%} avg PnL ({best['avg']/BENCHMARK_AVG - 1:+.0%} vs benchmark)")
    print(f"      June: {best['j_avg']:+.2%} avg, {best['j_n']} trades")
    print(f"      Tickers: {', '.join(t['ticker'] for t in sorted(best['per_ticker'], key=lambda x: x['avg'], reverse=True))}")

# ── TABLE E: IMPACT OF FAMILY 200 — key methods with vs without ──
EXPS_WITHOUT_200 = [e for e in ACTIVE_EXPERIMENTS if e != 200]
print()
print("=" * W)
print(f"  TABLE E: IMPACT OF ADDING FAMILY 200 (with vs without)")
print("=" * W)
print(f"  Same methods run on {EXPS_WITHOUT_200} vs {ACTIVE_EXPERIMENTS}")

_e_methods = [
    ('STATIC agree=3', lambda exps: [
        t for tk in all_tickers
        for t in apply_config_for_dates(
            static_configs[tk], tk, all_trading_dates,
            daily_probs, daily_trades, exps,
            min_family_agree=3, min_families_present=2)]),
    ('STATIC agree=2', lambda exps: [
        t for tk in all_tickers
        for t in apply_config_for_dates(
            static_configs[tk], tk, all_trading_dates,
            daily_probs, daily_trades, exps,
            min_family_agree=2, min_families_present=2)]),
    ('STATIC seed=50%', lambda exps: [
        t for tk in all_tickers
        for t in apply_config_seed_gate(
            static_configs[tk], tk, all_trading_dates, daily_seed_probs,
            daily_trades, exps, seed_pct_threshold=0.50,
            min_seeds_present=15)]),
    ('STATIC seed=60%', lambda exps: [
        t for tk in all_tickers
        for t in apply_config_seed_gate(
            static_configs[tk], tk, all_trading_dates, daily_seed_probs,
            daily_trades, exps, seed_pct_threshold=0.60,
            min_seeds_present=15)]),
]

print(f"\n  {'Method':<22} │ {'── WITHOUT 200 ──':^30} │ {'──── WITH 200 ────':^30} │ {'── DELTA ──':^20}")
print(f"  {'':<22} │ {'Trd':>4} {'Avg':>7} {'Tot':>7} {'DD':>7} {'JAvg':>7} │ "
      f"{'Trd':>4} {'Avg':>7} {'Tot':>7} {'DD':>7} {'JAvg':>7} │ {'dAvg':>7} {'dTot':>7} {'dJ':>7}")
print(f"  {'─'*22} │ {'─'*4} {'─'*7} {'─'*7} {'─'*7} {'─'*7} │ "
      f"{'─'*4} {'─'*7} {'─'*7} {'─'*7} {'─'*7} │ {'─'*7} {'─'*7} {'─'*7}")

for name, fn in _e_methods:
    ps_no = selective_portfolio(fn(EXPS_WITHOUT_200), f'{name} no200', min_avg=-999)
    ps_yes = selective_portfolio(fn(ACTIVE_EXPERIMENTS), f'{name} +200', min_avg=-999)
    def _v(ps, k, default=0):
        return ps[k] if ps else default
    n_no, avg_no = _v(ps_no, 'n'), _v(ps_no, 'avg')
    tot_no, dd_no = _v(ps_no, 'total'), _v(ps_no, 'dd')
    javg_no = _v(ps_no, 'j_avg')
    n_yes, avg_yes = _v(ps_yes, 'n'), _v(ps_yes, 'avg')
    tot_yes, dd_yes = _v(ps_yes, 'total'), _v(ps_yes, 'dd')
    javg_yes = _v(ps_yes, 'j_avg')
    d_avg = avg_yes - avg_no
    d_tot = tot_yes - tot_no
    d_j = javg_yes - javg_no
    tag = " ★" if d_avg > 0 and d_j > 0 else " ✓" if d_avg > 0 else ""
    print(f"  {name:<22} │ {n_no:>4} {avg_no:>+6.2%} {tot_no:>+6.1%} {dd_no:>+6.1%} {javg_no:>+6.2%} │ "
          f"{n_yes:>4} {avg_yes:>+6.2%} {tot_yes:>+6.1%} {dd_yes:>+6.1%} {javg_yes:>+6.2%} │ "
          f"{d_avg:>+6.2%} {d_tot:>+6.1%} {d_j:>+6.2%}{tag}")

print()
print("=" * W)
print(f"  Analysis complete.")
print("=" * W)
