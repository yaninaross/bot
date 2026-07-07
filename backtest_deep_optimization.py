"""
backtest_deep_optimization.py — Strategy × Composite Optimization Matrix
=========================================================================
Compares 11 optimization strategies crossed with 2 composite methods.

Strategies (11):
  BQ-loaded:  STATIC, GATE lb=7, ADAPTIVE THRESH, MOMENTUM 10d, VOL GATE
  Grid-search: DECAY hl=30d, DECAY hl=60d, HYBRID 30d+gate10, HYBRID 60d+gate10,
               HYBRID 30d+gate21, SELECTIVE DECAY

Composite methods (2):
  agree=3: Family agreement (3/5 experiments, min 2 present)
  seed=50%: Seed-level agreement (50% of seeds across all families)

Output Tables:
  A: Strategy × Composite matrix (full period + June)
  B: June-only matrix
  C: Per-ticker breakdown for top 3 combos
  D: Monthly breakdown for top 3 combos

Entry: 06:15 only. Configs from BQ for BQ strategies, grid for DECAY/HYBRID.
Colab-ready (no if __name__). Limit_entry mode only (S3 strategy).
"""

import os, io, gc, sys, subprocess, warnings, ctypes, math, pickle
from datetime import datetime, timedelta
from collections import defaultdict
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

SCORE_EXPERIMENTS = [152, 160, 175, 185, 200]
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
DEPLOY_TICKERS = ['HUT','FLNC','CIFR','CLSK','KEEL','RCAT','RIOT','APLD','WULF']
EXCLUDE_TICKERS = {'SMLR'}

TRAIN_TEACHERS_13 = ['BBAI','RCAT','LUNR','ASTS','ACHR','MVIS',
                     'WULF','CIFR','IREN','KEEL','HUT','CLSK','APLD']
TEACHERS_31 = ['CIFR','IREN','KEEL','HUT','APLD','WULF','RIOT','CLSK',
               'BTBT','ALAB','MARA','RCAT','ASTS','LUNR','BBAI','RKLB',
               'SPIR','BKSY','MNTS','PLUG','BE','FCEL','FLNC','OKLO',
               'SMR','MVIS','BGDE','RDW','PL','VOYG','SMLR']
EXP_TEACHER_SETS = {
    152: set(TRAIN_TEACHERS_13), 160: set(TRAIN_TEACHERS_13),
    175: set(TRAIN_TEACHERS_13), 185: set(TEACHERS_31), 200: set(TEACHERS_31),
}

DATA_START = '2025-12-01'
TEST_START = '2026-04-01'
TEST_END   = '2026-07-03'
REOPT_FREQ = 5

PROB_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
TP_GRID   = [0.06, 0.10, 0.16, 0.22]
SL_GRID   = [0.03, 0.06, 0.10, 0.14]

ENTRY_TIME = '06:15'
ENTRY_MIN = 375
EXIT_TIME_MIN = 780
MIN_OPT_TRADES = 3
MIN_OPT_PNL = 0.005
MIN_FAMILY_AGREE = 3

CACHE_FILE = '/tmp/deep_optimization_cache_v2.pkl'
CACHE_VERSION = f"{RESULTS_DIR}|{TEST_END}|{DATA_START}|{TEST_START}|bq_v12|entry0615"

gcs_client = gcs_storage.Client()
bq_client = bigquery.Client()
LN2 = math.log(2)


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
        print(f"      {ticker}/Exp{experiment}: {len(all_results)} seeds [{src}]")
    return all_results


def simulate_trade(ohlcv_day, entry_min, tp_pct, sl_pct, exit_min=EXIT_TIME_MIN):
    if ohlcv_day is None or ohlcv_day.empty:
        return None
    pst_mins = ohlcv_day.index.hour * 60 + ohlcv_day.index.minute
    entry_mask = pst_mins >= entry_min
    after = ohlcv_day[entry_mask]
    if len(after) < 5 or 'Close' not in after.columns:
        return None
    entry_price = after.iloc[0]['Close']
    if entry_price <= 0:
        return None
    target_price = entry_price * (1 + tp_pct)
    stop_price = entry_price * (1 - sl_pct)
    for i in range(1, len(after)):
        bar = after.iloc[i]
        bar_min = after.index[i].hour * 60 + after.index[i].minute
        high = bar.get('High', bar['Close'])
        low = bar.get('Low', bar['Close'])
        sl_hit = low <= stop_price
        tp_hit = high >= target_price
        if sl_hit and tp_hit:
            return tp_pct if np.random.random() > 0.5 else -sl_pct
        if sl_hit:
            return -sl_pct
        if tp_hit:
            return tp_pct
        if bar_min >= exit_min:
            return (bar['Close'] - entry_price) / entry_price
    last_close = after.iloc[-1]['Close']
    return (last_close - entry_price) / entry_price


# ==============================================================================
# COMPOSITE GATE FUNCTIONS
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


def apply_composite(cfg, ticker, dates, method):
    if method == 'agree=3':
        return apply_config_for_dates(cfg, ticker, dates,
            daily_probs, daily_trades, ACTIVE_EXPERIMENTS,
            min_family_agree=3, min_families_present=2)
    elif method == 'seed=50%':
        return apply_config_seed_gate(cfg, ticker, dates,
            daily_seed_probs, daily_trades, ACTIVE_EXPERIMENTS,
            seed_pct_threshold=0.50, min_seeds_present=15)
    return []


# ==============================================================================
# CACHE CHECK
# ==============================================================================
W = 120
print("=" * W)
print(f"  DEEP OPTIMIZATION — 11 STRATEGIES × 2 COMPOSITES")
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
            bq_configs = cache['bq_configs']
            daily_closes = cache['daily_closes']
            all_tickers = cache['all_tickers']
            ACTIVE_EXPERIMENTS = cache['ACTIVE_EXPERIMENTS']
            all_trading_dates = cache['all_trading_dates']
            all_available_dates = cache['all_available_dates']
            static_opt_dates = cache['static_opt_dates']
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
    # STEP 0: DISCOVER TICKERS + LOAD BQ CONFIGS
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

    # Load per-ticker configs from BQ
    exp_names = ','.join(f"'{e}_XSECTOR'" for e in ACTIVE_EXPERIMENTS)
    q = f"""
    WITH latest AS (
      SELECT *, ROW_NUMBER() OVER (
        PARTITION BY experiment, ticker
        ORDER BY run_timestamp DESC
      ) as rn
      FROM `ys-time-series.scoring_logs.wf_optimization_results`
      WHERE model_version = 'robust_wf_v12'
        AND is_ensemble = TRUE
        AND tpsl_mode = 'static'
        AND experiment IN ({exp_names})
    )
    SELECT experiment, ticker, prob_thresh, tp_pct, sl_pct,
           val_a_avg_pnl, survived_a
    FROM latest WHERE rn = 1
    """
    bq_df = bq_client.query(q).to_dataframe()
    print(f"    Loaded {len(bq_df)} BQ configs across {bq_df['experiment'].nunique()} experiments")

    bq_configs = {}
    n_configured = 0
    for ticker in all_tickers:
        tc = bq_df[bq_df['ticker'] == ticker]
        survived = tc[tc['survived_a'] == True]
        if survived.empty:
            bq_configs[ticker] = None
            continue
        best = survived.loc[survived['val_a_avg_pnl'].idxmax()]
        bq_configs[ticker] = {
            'entry_time': ENTRY_TIME,
            'prob_thresh': float(best['prob_thresh']),
            'tp': float(best['tp_pct']),
            'sl': float(best['sl_pct']),
            'avg_pnl': float(best['val_a_avg_pnl']),
            'n_trades': 0,
        }
        n_configured += 1
        cfg = bq_configs[ticker]
        print(f"    {ticker}: prob>{cfg['prob_thresh']:.2f} TP={cfg['tp']:.0%} "
              f"SL={cfg['sl']:.0%} avgPnL={cfg['avg_pnl']:.2%} ({best['experiment']})")
    print(f"    {n_configured}/{len(all_tickers)} tickers configured from BQ")
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
    # STEP 2: PRE-COMPUTE DAILY SIGNALS + TRADE OUTCOMES
    # ==============================================================================
    print(f"\n  STEP 2: Pre-computing daily signals + trade outcomes...")

    daily_probs = {}
    daily_seed_probs = {}
    all_score_dates_by_ticker = {}

    for (ticker, exp), seeds_dict in scores_cache.items():
        avg_probs = None
        n_seeds = 0
        for seed_id, sdf in seeds_dict.items():
            if 'Prob_Spike' not in sdf.columns:
                col = [c for c in sdf.columns if 'prob' in c.lower() or 'spike' in c.lower()]
                if not col:
                    continue
                sdf = sdf.rename(columns={col[0]: 'Prob_Spike'})
            daily_prob = sdf.groupby(sdf.index.date)['Prob_Spike'].max()
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

    print(f"    Computed probs for {len(daily_probs)} (ticker, exp) pairs")
    del scores_cache

    ohlcv_by_date = {}
    for ticker in ohlcv_cache:
        ohlcv = ohlcv_cache[ticker]
        grouped = {d: g for d, g in ohlcv.groupby(ohlcv.index.date)}
        ohlcv_by_date[ticker] = grouped
    print(f"    Grouped OHLCV for {len(ohlcv_by_date)} tickers")

    # Simulate ALL TP/SL combos at ENTRY_MIN=375 (grid needed for DECAY/HYBRID)
    ticker_trades = {}
    n_sims = 0
    for ticker in ohlcv_by_date:
        if ticker not in all_score_dates_by_ticker:
            continue
        dates = all_score_dates_by_ticker[ticker]
        grouped = ohlcv_by_date[ticker]
        for tp in TP_GRID:
            for sl in SL_GRID:
                if sl > tp:
                    continue
                outcomes = {}
                for trade_date in dates:
                    day_bars = grouped.get(trade_date)
                    if day_bars is None or day_bars.empty:
                        continue
                    pnl = simulate_trade(day_bars, ENTRY_MIN, tp, sl)
                    if pnl is not None:
                        outcomes[trade_date] = pnl - SLIPPAGE_PCT
                        n_sims += 1
                if outcomes:
                    ticker_trades[(ticker, ENTRY_TIME, tp, sl)] = outcomes
        print(f"    {ticker}: {n_sims:,} cumulative sims")

    daily_trades = {}
    for (ticker, exp) in daily_probs:
        for tp in TP_GRID:
            for sl in SL_GRID:
                if sl > tp:
                    continue
                key_t = (ticker, ENTRY_TIME, tp, sl)
                if key_t in ticker_trades:
                    daily_trades[(ticker, exp, ENTRY_TIME, tp, sl)] = ticker_trades[key_t]

    print(f"    Pre-computed {len(daily_trades)} outcome series ({n_sims:,} total sims)")
    del ticker_trades

    # Pre-compute daily closes for momentum/volatility gates
    daily_closes = {}
    for ticker in ohlcv_cache:
        ohlcv = ohlcv_cache[ticker]
        pst_mins = ohlcv.index.hour * 60 + ohlcv.index.minute
        eod = ohlcv[(pst_mins >= 390) & (pst_mins <= 780) & (ohlcv['Volume'] > 0)]
        if eod.empty:
            continue
        dc = eod.groupby(eod.index.date)['Close'].last()
        daily_closes[ticker] = dc.to_dict()
    print(f"    Daily closes for {len(daily_closes)} tickers")

    del ohlcv_by_date
    del ohlcv_cache
    _release_memory()

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
            'bq_configs': bq_configs,
            'daily_closes': daily_closes,
            'all_tickers': all_tickers,
            'ACTIVE_EXPERIMENTS': ACTIVE_EXPERIMENTS,
            'all_trading_dates': all_trading_dates,
            'all_available_dates': all_available_dates,
            'static_opt_dates': static_opt_dates,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    cache_mb = os.path.getsize(CACHE_FILE) / 1e6
    print(f"    ✓ Saved ({cache_mb:.0f} MB). Next run will skip Steps 0-2.")


# ==============================================================================
# STEP 3: GRID INFRASTRUCTURE + DECAY PRE-COMPUTATION
# ==============================================================================
reopt_indices = list(range(0, len(all_trading_dates), REOPT_FREQ))
static_opt_start = pd.Timestamp(DATA_START).date()

CONFIG_LIST = []
for tp in TP_GRID:
    for sl in SL_GRID:
        if sl > tp:
            continue
        for prob_thresh in PROB_GRID:
            CONFIG_LIST.append((ENTRY_TIME, tp, sl, prob_thresh))

print(f"\n  Grid: {len(CONFIG_LIST)} configs (1 entry × {len(TP_GRID)} TP × SL × {len(PROB_GRID)} prob)")


def _eval_config(ticker, entry_time_str, tp, sl, prob_thresh, opt_dates, weights=None):
    w_pnl = 0.0
    w_n = 0.0
    for exp in ACTIVE_EXPERIMENTS:
        key = (ticker, exp, entry_time_str, tp, sl)
        if key not in daily_trades:
            continue
        prob_key = (ticker, exp)
        if prob_key not in daily_probs:
            continue
        outcomes = daily_trades[key]
        probs = daily_probs[prob_key]
        for d in opt_dates:
            if d in probs and d in outcomes:
                if probs[d] >= prob_thresh:
                    w = weights[d] if weights else 1.0
                    w_pnl += w * outcomes[d]
                    w_n += w
    return w_pnl, w_n


def optimize_configs(opt_dates, ticker, weights=None):
    best_pnl = -999
    best_config = None
    for entry_time_str, tp, sl, prob_thresh in CONFIG_LIST:
        w_pnl, w_n = _eval_config(ticker, entry_time_str, tp, sl, prob_thresh, opt_dates, weights)
        if w_n >= MIN_OPT_TRADES:
            avg = w_pnl / w_n
            if avg > best_pnl and avg > MIN_OPT_PNL:
                best_pnl = avg
                best_config = {
                    'entry_time': entry_time_str, 'prob_thresh': prob_thresh,
                    'tp': tp, 'sl': sl, 'avg_pnl': avg, 'n_trades': w_n,
                }
    return best_config


# Pre-compute decay configs (don't depend on composite method)
print(f"\n  STEP 3: Pre-computing DECAY configs (grid search at each window)...")

decay_configs_by_hl = {}
for half_life in [30, 60]:
    decay_rate = LN2 / half_life
    decay_configs = {}
    for ri in reopt_indices:
        reopt_date = all_trading_dates[ri]
        opt_end = reopt_date - timedelta(days=1)
        opt_dates = [d for d in all_available_dates if d >= static_opt_start and d <= opt_end]
        if len(opt_dates) < 10:
            continue
        weights = {d: math.exp(-decay_rate * (reopt_date - d).days) for d in opt_dates}
        week_configs = {}
        for ticker in all_tickers:
            week_configs[ticker] = optimize_configs(opt_dates, ticker, weights)
        decay_configs[reopt_date] = week_configs
    decay_configs_by_hl[half_life] = decay_configs
    print(f"    DECAY hl={half_life}d: {len(decay_configs)} windows pre-computed")

_release_memory()


# ==============================================================================
# STEP 4: RUN 11 STRATEGIES × 2 COMPOSITES
# ==============================================================================
COMPOSITE_METHODS = ['agree=3', 'seed=50%']

print(f"\n  STEP 4: Running 11 strategies × {len(COMPOSITE_METHODS)} composites...")

def summarize(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    cum = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    june = df[df['month'] == '2026-06']
    return {
        'label': label,
        'n': len(df), 'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': max_dd,
        'j_n': len(june), 'j_total': june['pnl'].sum() if len(june) else 0,
        'j_avg': june['pnl'].mean() if len(june) else 0,
        'j_wr': (june['pnl'] > 0).mean() if len(june) else 0,
        'monthly': df.groupby('month').agg(
            n=('pnl','count'), total=('pnl','sum'),
            avg=('pnl','mean'), wr=('pnl', lambda x: (x>0).mean())
        ).to_dict('index'),
        'trades_df': df,
    }


all_results = {}

for comp_method in COMPOSITE_METHODS:
    print(f"\n  ── Composite: {comp_method} ──")

    # 1. STATIC (BQ)
    static_trades = []
    for ticker in all_tickers:
        static_trades.extend(apply_composite(
            bq_configs[ticker], ticker, all_trading_dates, comp_method))
    all_results[('STATIC (BQ)', comp_method)] = static_trades
    print(f"    [1/11] STATIC (BQ): {len(static_trades)} trades")

    # Build static_by_ticker for GATE reference
    static_by_ticker = defaultdict(list)
    for t in static_trades:
        static_by_ticker[t['ticker']].append(t)

    # 2. GATE lb=7 (BQ)
    gate_trades = []
    for ri in reopt_indices:
        reopt_date = all_trading_dates[ri]
        test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
        test_dates = all_trading_dates[ri:test_end_idx]
        for ticker in all_tickers:
            if bq_configs[ticker] is None:
                continue
            prior = [t for t in static_by_ticker[ticker] if t['date'] < reopt_date]
            recent = prior[-7:]
            if len(recent) >= 2:
                recent_avg = np.mean([t['pnl'] for t in recent])
                if recent_avg < 0:
                    continue
            gate_trades.extend(apply_composite(
                bq_configs[ticker], ticker, test_dates, comp_method))
    all_results[('GATE lb=7 (BQ)', comp_method)] = gate_trades
    print(f"    [2/11] GATE lb=7 (BQ): {len(gate_trades)} trades")

    # 3. DECAY hl=30d
    decay30_trades = []
    for ri in reopt_indices:
        reopt_date = all_trading_dates[ri]
        test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
        test_dates = all_trading_dates[ri:test_end_idx]
        week_configs = decay_configs_by_hl[30].get(reopt_date, {})
        for ticker in all_tickers:
            cfg = week_configs.get(ticker)
            decay30_trades.extend(apply_composite(cfg, ticker, test_dates, comp_method))
    all_results[('DECAY hl=30d', comp_method)] = decay30_trades
    print(f"    [3/11] DECAY hl=30d: {len(decay30_trades)} trades")

    # 4. DECAY hl=60d
    decay60_trades = []
    for ri in reopt_indices:
        reopt_date = all_trading_dates[ri]
        test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
        test_dates = all_trading_dates[ri:test_end_idx]
        week_configs = decay_configs_by_hl[60].get(reopt_date, {})
        for ticker in all_tickers:
            cfg = week_configs.get(ticker)
            decay60_trades.extend(apply_composite(cfg, ticker, test_dates, comp_method))
    all_results[('DECAY hl=60d', comp_method)] = decay60_trades
    print(f"    [4/11] DECAY hl=60d: {len(decay60_trades)} trades")

    # 5-7. HYBRID Gate+Decay
    for hl, glb, label_suffix in [(30, 10, '30d+gate10'), (60, 10, '60d+gate10'), (30, 21, '30d+gate21')]:
        hybrid_trades = []
        accumulated_trades = defaultdict(list)
        for ri in reopt_indices:
            reopt_date = all_trading_dates[ri]
            test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
            test_dates = all_trading_dates[ri:test_end_idx]
            week_configs = decay_configs_by_hl[hl].get(reopt_date, {})
            for ticker in all_tickers:
                cfg = week_configs.get(ticker)
                if cfg is None:
                    continue
                prior = accumulated_trades[ticker]
                recent = [t for t in prior
                          if t['date'] >= (reopt_date - timedelta(days=int(glb * 1.5)))]
                recent = recent[-glb:]
                if len(recent) >= 2:
                    if np.mean([t['pnl'] for t in recent]) < 0:
                        continue
                wt = apply_composite(cfg, ticker, test_dates, comp_method)
                hybrid_trades.extend(wt)
                accumulated_trades[ticker].extend(wt)
        idx = 5 + [(30,10), (60,10), (30,21)].index((hl, glb))
        all_results[(f'HYBRID {label_suffix}', comp_method)] = hybrid_trades
        print(f"    [{idx}/11] HYBRID {label_suffix}: {len(hybrid_trades)} trades")

    # 8. ADAPTIVE THRESH (BQ)
    adaptive_trades = []
    for ticker in all_tickers:
        if bq_configs[ticker] is None:
            continue
        base_thresh = bq_configs[ticker]['prob_thresh']
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
            adj_config = dict(bq_configs[ticker])
            adj_config['prob_thresh'] = current_thresh
            wt = apply_composite(adj_config, ticker, test_dates, comp_method)
            adaptive_trades.extend(wt)
            recent_results.extend([t['pnl'] for t in wt])
    all_results[('ADAPTIVE THRESH (BQ)', comp_method)] = adaptive_trades
    print(f"    [8/11] ADAPTIVE THRESH (BQ): {len(adaptive_trades)} trades")

    # 9. MOMENTUM 10d (BQ)
    momentum_trades = []
    for ticker in all_tickers:
        if bq_configs[ticker] is None:
            continue
        closes = daily_closes.get(ticker, {})
        for ri in reopt_indices:
            reopt_date = all_trading_dates[ri]
            test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
            test_dates = all_trading_dates[ri:test_end_idx]
            prior_dates = sorted([d for d in closes if d < reopt_date])
            if len(prior_dates) >= 10:
                close_now = closes[prior_dates[-1]]
                close_10d = closes[prior_dates[-10]]
                if close_10d > 0 and (close_now - close_10d) / close_10d <= 0:
                    continue
            momentum_trades.extend(apply_composite(
                bq_configs[ticker], ticker, test_dates, comp_method))
    all_results[('MOMENTUM 10d (BQ)', comp_method)] = momentum_trades
    print(f"    [9/11] MOMENTUM 10d (BQ): {len(momentum_trades)} trades")

    # 10. SELECTIVE DECAY (BQ base, decay fallback)
    selective_trades = []
    for ticker in all_tickers:
        if bq_configs[ticker] is None:
            continue
        accumulated = []
        for ri in reopt_indices:
            reopt_date = all_trading_dates[ri]
            test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
            test_dates = all_trading_dates[ri:test_end_idx]
            recent = [t for t in accumulated
                      if t['date'] >= (reopt_date - timedelta(days=15))][-10:]
            use_decay = len(recent) >= 2 and np.mean([t['pnl'] for t in recent]) < 0
            if use_decay and reopt_date in decay_configs_by_hl[30]:
                cfg = decay_configs_by_hl[30][reopt_date].get(ticker, bq_configs[ticker])
            else:
                cfg = bq_configs[ticker]
            wt = apply_composite(cfg, ticker, test_dates, comp_method)
            selective_trades.extend(wt)
            accumulated.extend(wt)
    all_results[('SELECTIVE DECAY', comp_method)] = selective_trades
    print(f"    [10/11] SELECTIVE DECAY: {len(selective_trades)} trades")

    # 11. VOL GATE (BQ)
    vol_gate_trades = []
    for ticker in all_tickers:
        if bq_configs[ticker] is None:
            continue
        closes = daily_closes.get(ticker, {})
        for ri in reopt_indices:
            reopt_date = all_trading_dates[ri]
            test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
            test_dates = all_trading_dates[ri:test_end_idx]
            prior_dates = sorted([d for d in closes if d < reopt_date])
            if len(prior_dates) >= 20:
                prices = [closes[d] for d in prior_dates[-20:]]
                rets = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
                vol_20d = np.std(rets) if len(rets) >= 10 else 0
                vol_5d = np.std(rets[-5:]) if len(rets) >= 5 else 0
                if vol_20d > 0 and vol_5d / vol_20d > 2.0:
                    continue
            vol_gate_trades.extend(apply_composite(
                bq_configs[ticker], ticker, test_dates, comp_method))
    all_results[('VOL GATE (BQ)', comp_method)] = vol_gate_trades
    print(f"    [11/11] VOL GATE (BQ): {len(vol_gate_trades)} trades")

_release_memory()


# ==============================================================================
# STEP 5: RESULTS TABLES
# ==============================================================================
STRATEGY_ORDER = [
    'STATIC (BQ)', 'GATE lb=7 (BQ)', 'DECAY hl=30d', 'DECAY hl=60d',
    'HYBRID 30d+gate10', 'HYBRID 60d+gate10', 'HYBRID 30d+gate21',
    'ADAPTIVE THRESH (BQ)', 'MOMENTUM 10d (BQ)', 'SELECTIVE DECAY', 'VOL GATE (BQ)',
]

summaries = {}
for (strat, comp), trades in all_results.items():
    summaries[(strat, comp)] = summarize(trades, f"{strat} | {comp}")

print()
print("=" * W)
print(f"  STEP 5: RESULTS — {len(STRATEGY_ORDER)} STRATEGIES × {len(COMPOSITE_METHODS)} COMPOSITES")
print("=" * W)


# ── TABLE A: Strategy × Composite Matrix (Full Period + June) ──
print(f"\n  TABLE A: FULL PERIOD + JUNE — STRATEGY × COMPOSITE MATRIX")

col_w = 42
print(f"\n  {'Strategy':<25}", end="")
for cm in COMPOSITE_METHODS:
    print(f" │ {cm:^{col_w}}", end="")
print()
print(f"  {'':25}", end="")
for _ in COMPOSITE_METHODS:
    print(f" │ {'Trd':>4} {'AvgPnL':>7} {'TotPnL':>8} {'WR':>4} {'MaxDD':>7} {'JAvg':>6} {'JWR':>4}", end="")
print()
print(f"  {'─'*25}", end="")
for _ in COMPOSITE_METHODS:
    print(f" │ {'─'*4} {'─'*7} {'─'*8} {'─'*4} {'─'*7} {'─'*6} {'─'*4}", end="")
print()

for strat in STRATEGY_ORDER:
    print(f"  {strat:<25}", end="")
    for cm in COMPOSITE_METHODS:
        s = summaries.get((strat, cm))
        if s and s['n'] > 0:
            print(f" │ {s['n']:>4} {s['avg']:>+6.2%} {s['total']:>+7.1%} "
                  f"{s['wr']:>3.0%} {s['dd']:>+6.1%} {s['j_avg']:>+5.2%} "
                  f"{s['j_wr']:>3.0%}", end="")
        else:
            print(f" │ {'—':^{col_w}}", end="")
    print()


# ── TABLE B: June-Only Matrix ──
print(f"\n  TABLE B: JUNE-ONLY BREAKDOWN")

print(f"\n  {'Strategy':<25}", end="")
for cm in COMPOSITE_METHODS:
    print(f" │ {'Trd':>4} {'AvgPnL':>7} {'TotPnL':>8} {'WR':>4}", end="")
print()
print(f"  {'─'*25}", end="")
for _ in COMPOSITE_METHODS:
    print(f" │ {'─'*4} {'─'*7} {'─'*8} {'─'*4}", end="")
print()

for strat in STRATEGY_ORDER:
    print(f"  {strat:<25}", end="")
    for cm in COMPOSITE_METHODS:
        s = summaries.get((strat, cm))
        if s and s['j_n'] > 0:
            print(f" │ {s['j_n']:>4} {s['j_avg']:>+6.2%} {s['j_total']:>+7.1%} "
                  f"{s['j_wr']:>3.0%}", end="")
        else:
            print(f" │ {'—':>25}", end="")
    print()


# ── Identify top 3 combos ──
ranked_combos = sorted(
    [(strat, cm, summaries[(strat, cm)])
     for strat in STRATEGY_ORDER for cm in COMPOSITE_METHODS
     if summaries.get((strat, cm)) and summaries[(strat, cm)]['n'] > 0],
    key=lambda x: x[2]['total'], reverse=True
)
top3 = ranked_combos[:3]


# ── TABLE C: Per-Ticker for Top 3 ──
print(f"\n  TABLE C: PER-TICKER BREAKDOWN — TOP 3 COMBOS")

top3_labels = [f"{s}|{c}" for s, c, _ in top3]
print(f"\n  {'Ticker':<7}", end="")
for lbl in top3_labels:
    print(f"  {lbl:<24}", end="")
print()
print(f"  {'─'*7}", end="")
for _ in top3_labels:
    print(f"  {'─'*24}", end="")
print()

active_tickers = set()
for s, c, sm in top3:
    if 'trades_df' in sm:
        active_tickers.update(sm['trades_df']['ticker'].unique())
active_tickers = sorted(active_tickers)

for ticker in active_tickers:
    print(f"  {ticker:<7}", end="")
    for s, c, sm in top3:
        if 'trades_df' in sm:
            tt = sm['trades_df'][sm['trades_df']['ticker'] == ticker]
            if len(tt) > 0:
                print(f"  {tt['pnl'].mean():>+5.2%} {len(tt):>2}t {(tt['pnl']>0).mean():>3.0%}wr"
                      f"{'':>8}", end="")
            else:
                print(f"  {'no trades':<24}", end="")
        else:
            print(f"  {'—':<24}", end="")
    print()


# ── TABLE D: Monthly Breakdown for Top 3 ──
print(f"\n  TABLE D: MONTHLY BREAKDOWN — TOP 3 COMBOS")

months = ['2026-04', '2026-05', '2026-06']
mlabels = ['Apr', 'May', 'Jun']

print(f"\n  {'Combo':<30}", end="")
for ml in mlabels:
    print(f"  {ml:>24}", end="")
print()
print(f"  {'─'*30}", end="")
for _ in mlabels:
    print(f"  {'─'*24}", end="")
print()

for s, c, sm in top3:
    lbl = f"{s}|{c}"
    print(f"  {lbl:<30}", end="")
    for m in months:
        ms = sm.get('monthly', {}).get(m)
        if ms:
            print(f"  {ms['avg']:>+5.2%}({ms['n']:>3}t,{ms['wr']:>4.0%}wr){'':>5}", end="")
        else:
            print(f"  {'—':>24}", end="")
    print()


# ── DIAGNOSIS ──
print()
print("=" * W)
print(f"  DIAGNOSIS")
print("=" * W)

# Best overall
if ranked_combos:
    best = ranked_combos[0]
    print(f"\n  BEST TOTAL PnL: {best[0]} | {best[1]}")
    print(f"    {best[2]['total']:+.1%} total, {best[2]['avg']:+.2%} avg, "
          f"{best[2]['wr']:.0%} WR, {best[2]['dd']:+.1%} MaxDD")
    print(f"    June: {best[2]['j_avg']:+.2%} avg, {best[2]['j_n']} trades")

# Best risk-adjusted
if ranked_combos:
    static_total = summaries.get(('STATIC (BQ)', 'agree=3'), {}).get('total', 0)
    viable = [(s, c, sm) for s, c, sm in ranked_combos
              if sm['total'] >= static_total * 0.7]
    if viable:
        best_risk = min(viable, key=lambda x: abs(x[2]['dd']))
        print(f"\n  BEST RISK-ADJUSTED (within 70% of STATIC PnL): {best_risk[0]} | {best_risk[1]}")
        print(f"    {best_risk[2]['total']:+.1%} total, {best_risk[2]['dd']:+.1%} MaxDD")

# Best June
best_june = max([(s, c, sm) for s, c, sm in ranked_combos if sm['j_n'] > 0],
                key=lambda x: x[2]['j_avg'], default=None)
if best_june:
    print(f"\n  BEST JUNE: {best_june[0]} | {best_june[1]}")
    print(f"    {best_june[2]['j_avg']:+.2%} avg, {best_june[2]['j_n']} trades, "
          f"{best_june[2]['j_wr']:.0%} WR")

# Composite comparison
print(f"\n  ── COMPOSITE METHOD COMPARISON ──")
for cm in COMPOSITE_METHODS:
    cm_results = [(s, summaries[(s, cm)]) for s in STRATEGY_ORDER
                  if summaries.get((s, cm)) and summaries[(s, cm)]['n'] > 0]
    if cm_results:
        best_s = max(cm_results, key=lambda x: x[1]['total'])
        print(f"  {cm}: best = {best_s[0]} ({best_s[1]['total']:+.1%} total, "
              f"{best_s[1]['avg']:+.2%} avg)")

print()
print("=" * W)
print(f"  Analysis complete. {len(STRATEGY_ORDER)} strategies × {len(COMPOSITE_METHODS)} composites "
      f"= {len(STRATEGY_ORDER) * len(COMPOSITE_METHODS)} combos evaluated.")
print("=" * W)
