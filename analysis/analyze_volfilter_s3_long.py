# %% ── Volume Filter Analysis for S3 LONG ─────────────────────────────────────
#
# Evaluates whether per-ticker volume filter (entry-bar volume percentile)
# improves S3 LONG PnL.
#
# TP/SL/prob_thresh/mode are FROZEN from existing BQ configs.
# Only the volume filter threshold is swept on OPT, validated on TEST.
# Results saved to BQ as model_version = 'robust_wf_v12_stitched_v14'.
#
# NO GPU needed — scores loaded from pre-computed CSVs on GCS.
# Colab-ready.

import os, io, gc, uuid, warnings
import numpy as np
import pandas as pd
from google.cloud import storage, bigquery as bq
from numba import jit
import pytz
from datetime import timedelta

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    _IN_COLAB = True
    auth.authenticate_user()
except (ImportError, Exception):
    _IN_COLAB = False

import matplotlib
if not _IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIG
# ==============================================================================
OPT_START      = '2026-04-01'
OPT_END        = '2026-05-31'
TEST_START     = '2026-06-01'
TEST_END       = '2026-08-08'

ENTRY_TIME     = '06:15'
ENTRY_MINUTE   = 375
EXIT_TIME_MIN  = 780
SLIPPAGE_PCT   = 0.0015

MODEL_VERSIONS_IN  = ['robust_wf_v12', 'robust_wf_v12_stitched']
MODEL_VERSION_OUT  = 'robust_wf_v12_stitched_v14'
EXPERIMENT_OUT     = 's3_vol_filter_v14'
DEPLOY_PNL_THRESHOLD = 0.018
MIN_TRADES_ABSOLUTE  = 3

GCS_BUCKET     = os.environ.get("GCS_BUCKET", "")
GCS_INFER_DIR  = 'Data/Inference_v2'

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

PST = pytz.timezone('America/Los_Angeles')

NO_FILTER_DEFAULTS = {'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0}

STITCH_FOLD_MAP = {
    '170': [f'{i}_XSECTOR' for i in range(170, 176)],
    '180': [f'{i}_XSECTOR' for i in range(180, 186)],
    '201': [f'{i}_XSECTOR' for i in range(201, 206)],
    '220': [f'{i}_XSECTOR' for i in range(220, 226)],
    '240': [f'{i}_XSECTOR' for i in range(240, 246)],
}

MODE_SUFFIXES = ['_LIMIT', '_GRACE', '_ATRSL', '_ATRADP', '_ATR',
                 '_REGIME', '_TRAIL', '_REGADP']
SUPPORTED_MODES = {'static', 'grace', 'limit_entry'}

VOL_PCT_GRID = [0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75]
VOL_LOOKBACK_DAYS = 30

WRITE_TO_BQ = False
WIPE_EXISTING = True

SEED = 42
np.random.seed(SEED)
W = 100


# ==============================================================================
# GCS + BQ
# ==============================================================================
def _gcs_bucket():
    return storage.Client().bucket(GCS_BUCKET)

BQ_TABLE_ID = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"

def _ensure_wf_table(bq_client, table_id):
    from google.cloud.bigquery import SchemaField, Table
    schema = [
        SchemaField('run_id', 'STRING'),
        SchemaField('run_timestamp', 'TIMESTAMP'),
        SchemaField('experiment', 'STRING'),
        SchemaField('ticker', 'STRING'),
        SchemaField('seed', 'INT64'),
        SchemaField('entry_time', 'STRING'),
        SchemaField('entry_minute', 'INT64'),
        SchemaField('prob_thresh', 'FLOAT64'),
        SchemaField('tp_pct', 'FLOAT64'),
        SchemaField('sl_pct', 'FLOAT64'),
        SchemaField('vix_threshold', 'FLOAT64'),
        SchemaField('tlt_crash', 'FLOAT64'),
        SchemaField('spy_ignore', 'FLOAT64'),
        SchemaField('min_vol_bars', 'INT64'),
        SchemaField('vol_filter_pct', 'FLOAT64'),
        SchemaField('opt_avg_pnl', 'FLOAT64'),
        SchemaField('opt_win_rate', 'FLOAT64'),
        SchemaField('opt_freq', 'FLOAT64'),
        SchemaField('opt_count', 'INT64'),
        SchemaField('val_a_avg_pnl', 'FLOAT64'),
        SchemaField('val_a_win_rate', 'FLOAT64'),
        SchemaField('val_a_freq', 'FLOAT64'),
        SchemaField('val_a_count', 'INT64'),
        SchemaField('survived_a', 'BOOL'),
        SchemaField('val_b_avg_pnl', 'FLOAT64'),
        SchemaField('val_b_win_rate', 'FLOAT64'),
        SchemaField('val_b_freq', 'FLOAT64'),
        SchemaField('val_b_count', 'INT64'),
        SchemaField('survived_b', 'BOOL'),
        SchemaField('opt_start', 'STRING'),
        SchemaField('opt_end', 'STRING'),
        SchemaField('val_a_start', 'STRING'),
        SchemaField('val_a_end', 'STRING'),
        SchemaField('val_b_start', 'STRING'),
        SchemaField('val_b_end', 'STRING'),
        SchemaField('tpsl_mode', 'STRING'),
        SchemaField('tp_mult', 'FLOAT64'),
        SchemaField('sl_mult', 'FLOAT64'),
        SchemaField('trail_start_mult', 'FLOAT64'),
        SchemaField('trail_end_mult', 'FLOAT64'),
        SchemaField('model_version', 'STRING'),
        SchemaField('v0_fixed', 'BOOL'),
        SchemaField('is_ensemble', 'BOOL'),
        SchemaField('n_seeds_avg', 'INT64'),
        SchemaField('regime_tp_pct', 'FLOAT64'),
        SchemaField('regime_sl_pct', 'FLOAT64'),
        SchemaField('regime_lookback', 'INT64'),
        SchemaField('regime_spike_thresh', 'FLOAT64'),
        SchemaField('grace_minutes', 'INT64'),
        SchemaField('dip_pct', 'FLOAT64'),
        SchemaField('max_wait_minutes', 'INT64'),
        SchemaField('pre_range_filter_pct', 'FLOAT64'),
    ]
    bq_client.create_table(Table(table_id, schema=schema), exists_ok=True)

_bq_client = bq.Client()
_ensure_wf_table(_bq_client, BQ_TABLE_ID)


# ==============================================================================
# SCORE LOADING FROM GCS CSVs (from optimize_wf_v12.py)
# ==============================================================================

def load_seed_scores(ticker, experiment):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    results_dir = f"Algos/Results_all_v2/Universal_Exp{experiment}"
    client = storage.Client()
    prefix = f"{results_dir}/{clean}/Seed_"
    blobs = list(client.bucket(GCS_BUCKET).list_blobs(prefix=prefix))

    seed_series = {}
    for blob in blobs:
        name = blob.name.split('/')[-1]
        if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
            continue
        seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
        df = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Los_Angeles')
        if 'Prob_Spike' in df.columns:
            seed_series[seed_id] = df['Prob_Spike'].sort_index()
    return seed_series


def load_stitched_seed_scores(ticker, fold_experiments):
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
    return stitched


def _parse_base_experiment(experiment_str):
    base = experiment_str
    for sfx in sorted(MODE_SUFFIXES, key=len, reverse=True):
        if base.endswith(sfx):
            return base[:-len(sfx)]
    return base


def _get_fold_experiments(experiment_str, model_version):
    base = _parse_base_experiment(experiment_str)
    is_stitched = 'stitched' in model_version
    if not is_stitched:
        return [base], False
    num = base.split('_')[0]
    folds = STITCH_FOLD_MAP.get(num, [base])
    return folds, True


def load_s3_daily_scores(ticker, experiment_str, model_version):
    folds, is_stitched = _get_fold_experiments(experiment_str, model_version)

    if is_stitched:
        seed_series = load_stitched_seed_scores(ticker, folds)
    else:
        base = _parse_base_experiment(experiment_str)
        seed_series = load_seed_scores(ticker, base)

    if not seed_series:
        return {}

    daily_accum = {}
    for pos, series in seed_series.items():
        minutes = series.index.hour * 60 + series.index.minute
        at_entry = series[minutes == ENTRY_MINUTE]
        d_ints = (at_entry.index.year * 10000 + at_entry.index.month * 100
                  + at_entry.index.day).values.astype(int)
        for d_int, prob in zip(d_ints, at_entry.values):
            if d_int not in daily_accum:
                daily_accum[d_int] = []
            daily_accum[d_int].append(float(prob))

    return {d: float(np.mean(probs)) for d, probs in daily_accum.items()}


# ==============================================================================
# LONG SIMULATION ENGINES (Numba)
# ==============================================================================

@jit(nopython=True)
def run_trade_sim(entry_indices, prices, highs, lows, date_ints, time_mins,
                  tp, sl, exit_time):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    trades      = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)

    trade_count       = 0
    current_entry_ptr = 0
    last_traded_date  = -1

    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break

        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1
            continue

        entry_price  = prices[idx]
        target_price = entry_price * (1 + tp)
        stop_price   = entry_price * (1 - sl)

        outcome    = 0.0
        trade_done = False
        exit_idx   = idx

        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
            sl_hit = lows[k] <= stop_price
            tp_hit = highs[k] >= target_price
            if sl_hit and tp_hit:
                outcome = -sl if np.random.random() < 0.5 else tp
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (prices[k] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break

        if not trade_done:
            outcome  = (prices[n_prices-1] - entry_price) / entry_price
            exit_idx = n_prices

        trades[trade_count]      = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date

        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return trades[:trade_count], entry_dates[:trade_count]


@jit(nopython=True)
def run_trade_sim_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
                        tp, sl, exit_time, grace_minutes):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    trades      = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)

    trade_count       = 0
    current_entry_ptr = 0
    last_traded_date  = -1

    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break

        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1
            continue

        entry_price  = prices[idx]
        target_price = entry_price * (1 + tp)
        stop_price   = entry_price * (1 - sl)
        entry_time   = time_mins[idx]

        outcome    = 0.0
        trade_done = False
        exit_idx   = idx

        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
            minutes_elapsed = time_mins[k] - entry_time
            sl_active = minutes_elapsed >= grace_minutes
            tp_hit = highs[k] >= target_price
            sl_hit = sl_active and (lows[k] <= stop_price)
            if sl_hit and tp_hit:
                outcome = -sl if np.random.random() < 0.5 else tp
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (prices[k] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break

        if not trade_done:
            outcome  = (prices[n_prices-1] - entry_price) / entry_price
            exit_idx = n_prices

        trades[trade_count]      = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date

        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return trades[:trade_count], entry_dates[:trade_count]


@jit(nopython=True)
def run_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints, time_mins,
                              tp, sl, exit_time, dip_pct, max_wait_minutes):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    trades      = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)

    trade_count       = 0
    current_entry_ptr = 0
    last_traded_date  = -1

    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break

        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1
            continue

        signal_price = prices[idx]
        limit_price  = signal_price * (1.0 - dip_pct)
        signal_time  = time_mins[idx]

        fill_idx   = -1
        fill_price = 0.0
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                break
            if time_mins[k] - signal_time > max_wait_minutes:
                break
            if lows[k] <= limit_price:
                fill_price = limit_price
                fill_idx = k
                break

        if fill_idx < 0:
            current_entry_ptr += 1
            last_traded_date = entry_date
            continue

        target_price = fill_price * (1 + tp)
        stop_price   = fill_price * (1 - sl)

        outcome    = 0.0
        trade_done = False
        exit_idx   = fill_idx

        for k in range(fill_idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - fill_price) / fill_price
                exit_idx = k; trade_done = True; break
            sl_hit = lows[k] <= stop_price
            tp_hit = highs[k] >= target_price
            if sl_hit and tp_hit:
                outcome = -sl if np.random.random() < 0.5 else tp
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (prices[k] - fill_price) / fill_price
                exit_idx = k; trade_done = True; break

        if not trade_done:
            outcome  = (prices[n_prices-1] - fill_price) / fill_price
            exit_idx = n_prices

        trades[trade_count]      = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date

        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return trades[:trade_count], entry_dates[:trade_count]


def _run_sim(indices, ohlcv_arrays, s_cfg, mode):
    prices = ohlcv_arrays['prices']
    highs  = ohlcv_arrays['highs']
    lows   = ohlcv_arrays['lows']
    dates  = ohlcv_arrays['dates']
    times  = ohlcv_arrays['times'].astype(np.int64)

    if mode == 'grace':
        pnl, d = run_trade_sim_grace(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN, s_cfg['grace_minutes'])
    elif mode == 'limit_entry':
        pnl, d = run_trade_sim_limit_entry(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN,
            s_cfg['dip_pct'], s_cfg['max_wait'])
    else:
        pnl, d = run_trade_sim(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN)
    if len(pnl) > 0 and SLIPPAGE_PCT > 0:
        pnl = pnl - 2 * SLIPPAGE_PCT
    return pnl, d


# ==============================================================================
# DATA LOADING
# ==============================================================================

def _load_parquet_with_dt_index(gcs_uri, tz_out='America/Los_Angeles'):
    df = pd.read_parquet(gcs_uri)
    df = df.reset_index()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    else:
        num_col = next((c for c in df.columns if c in ('Timestamp', 'Date', 'Time')), df.columns[0])
        df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
    df = df.set_index('Datetime')
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.index = df.index.tz_convert(tz_out)
    return df


def load_ticker_ohlcv(ticker):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{GCS_BUCKET}/{GCS_INFER_DIR}/{clean}{suffix}.parquet"
        try:
            return _load_parquet_with_dt_index(uri)
        except Exception:
            continue
    raise FileNotFoundError(f"No OHLCV data found for {ticker}")


def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date, vol_baselines=None):
    eval_start = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    eval_end   = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= eval_start) & (df_ohlcv.index < eval_end)]
    if len(df) == 0:
        return None
    dates = (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64)
    times = (df.index.hour * 60 + df.index.minute).values
    result = {
        'index':  df.index,
        'prices': df['Close'].values.copy(),
        'highs':  df['High'].values.copy(),
        'lows':   df['Low'].values.copy(),
        'volumes': df['Volume'].values.copy(),
        'dates':  dates,
        'times':  times,
        'months': max(1.0, (df.index[-1] - df.index[0]).days / 30.44),
    }
    if vol_baselines is not None:
        result['vol_baselines'] = vol_baselines
    return result


# ==============================================================================
# VOLUME FILTER (from optimize_wf_v12.py)
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


def _compute_vol_pct_ok(volumes, times, dates, vol_baselines, entry_minute, threshold_pct):
    n = len(volumes)
    ok = np.ones(n, dtype=np.bool_)
    if threshold_pct <= 0:
        return ok
    at_entry = np.where(times == entry_minute)[0]
    for idx in at_entry:
        baseline = vol_baselines.get((int(dates[idx]), entry_minute), 0.0)
        if baseline > 0:
            pct = (volumes[idx] / baseline) * 100
            if pct < threshold_pct:
                ok[idx] = False
    return ok


# ==============================================================================
# PHASE 1: LOAD S3 SCORES FROM GCS
# ==============================================================================

def run_scoring_phase(frozen_configs):
    print(f"\n{'='*W}")
    print(f"  PHASE 1: LOAD S3 SCORES FROM GCS")
    print(f"{'='*W}")

    all_daily_scores = {}
    tickers = sorted(frozen_configs.keys())

    for t_idx, ticker in enumerate(tickers):
        cfg = frozen_configs[ticker]
        print(f"  [{t_idx+1}/{len(tickers)}] {ticker}  exp={cfg['experiment']}  "
              f"mv={cfg['model_version']}")
        try:
            daily_scores = load_s3_daily_scores(
                ticker, cfg['experiment'], cfg['model_version'])
            all_daily_scores[ticker] = daily_scores
            print(f"    {len(daily_scores)} daily scores loaded")
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\n  Loaded scores for {len(all_daily_scores)} tickers")
    return all_daily_scores


# ==============================================================================
# PHASE 2: LOAD FROZEN CONFIGS FROM BQ
# ==============================================================================

def load_frozen_configs():
    print(f"\n{'='*W}")
    print(f"  PHASE 2: LOAD FROZEN CONFIGS from BQ")
    print(f"  model_versions: {MODEL_VERSIONS_IN}")
    print(f"{'='*W}")

    mv_list = ', '.join(f"'{v}'" for v in MODEL_VERSIONS_IN)
    query = f"""
    SELECT ticker, experiment, model_version, prob_thresh, tp_pct, sl_pct, tpsl_mode,
           grace_minutes, dip_pct, max_wait_minutes,
           vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
           val_a_avg_pnl, val_b_avg_pnl, val_b_win_rate, val_b_count,
           opt_avg_pnl, opt_count, entry_minute
    FROM `{BQ_TABLE_ID}`
    WHERE (model_version IN ({mv_list})
           OR model_version LIKE 'robust_wf_v12_17%'
           OR model_version LIKE 'robust_wf_v12_18%'
           OR model_version LIKE 'robust_wf_v12_20%'
           OR model_version LIKE 'robust_wf_v12_22%'
           OR model_version LIKE 'robust_wf_v12_24%')
      AND survived_a = TRUE
      AND val_b_avg_pnl >= {DEPLOY_PNL_THRESHOLD}
    ORDER BY val_b_avg_pnl DESC
    """
    df = _bq_client.query(query).to_dataframe()

    if len(df) == 0:
        print(f"  WARNING: No configs found with val_b >= {DEPLOY_PNL_THRESHOLD}. Trying survived_a only...")
        query_all = f"""
        SELECT ticker, experiment, model_version, prob_thresh, tp_pct, sl_pct, tpsl_mode,
               grace_minutes, dip_pct, max_wait_minutes,
               vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
               val_a_avg_pnl, val_b_avg_pnl, val_b_win_rate, val_b_count,
               opt_avg_pnl, opt_count, entry_minute
        FROM `{BQ_TABLE_ID}`
        WHERE (model_version IN ({mv_list})
               OR model_version LIKE 'robust_wf_v12_17%'
               OR model_version LIKE 'robust_wf_v12_18%'
               OR model_version LIKE 'robust_wf_v12_20%'
               OR model_version LIKE 'robust_wf_v12_22%'
               OR model_version LIKE 'robust_wf_v12_24%')
          AND survived_a = TRUE
        ORDER BY val_b_avg_pnl DESC
        """
        df = _bq_client.query(query_all).to_dataframe()

    if len(df) == 0:
        print("  ERROR: No configs found at all!")
        return {}

    configs = {}
    for ticker in df['ticker'].unique():
        if ticker in EXCLUDE_TICKERS:
            continue
        tdf = df[df['ticker'] == ticker]
        best = tdf.iloc[0]
        mode = best['tpsl_mode'] if pd.notna(best['tpsl_mode']) else 'static'

        if mode not in SUPPORTED_MODES:
            continue

        cfg = {
            'ticker': ticker,
            'experiment': best['experiment'],
            'model_version': best['model_version'],
            'prob_thresh': float(best['prob_thresh']),
            'tp_pct': float(best['tp_pct']),
            'sl_pct': float(best['sl_pct']),
            'tpsl_mode': mode,
            'grace_minutes': int(best['grace_minutes']) if pd.notna(best.get('grace_minutes')) else 0,
            'dip_pct': float(best['dip_pct']) if pd.notna(best.get('dip_pct')) else 0.0,
            'max_wait': int(best['max_wait_minutes']) if pd.notna(best.get('max_wait_minutes')) else 0,
            'entry_minute': int(best['entry_minute']) if pd.notna(best.get('entry_minute')) else ENTRY_MINUTE,
            'existing_vix': float(best['vix_threshold']) if pd.notna(best.get('vix_threshold')) else 999.0,
            'existing_tlt': float(best['tlt_crash']) if pd.notna(best.get('tlt_crash')) else -999.0,
            'existing_spy': float(best['spy_ignore']) if pd.notna(best.get('spy_ignore')) else 999.0,
            'existing_vol': float(best['vol_filter_pct']) if pd.notna(best.get('vol_filter_pct')) else 0.0,
            'val_b_avg_pnl': float(best['val_b_avg_pnl']) if pd.notna(best.get('val_b_avg_pnl')) else 0.0,
            'val_b_win_rate': float(best['val_b_win_rate']) if pd.notna(best.get('val_b_win_rate')) else 0.0,
            'val_b_count': int(best['val_b_count']) if pd.notna(best.get('val_b_count')) else 0,
            'opt_avg_pnl': float(best['opt_avg_pnl']) if pd.notna(best.get('opt_avg_pnl')) else 0.0,
        }
        configs[ticker] = cfg

    print(f"  Loaded {len(configs)} ticker configs from BQ")
    n_skip = sum(1 for t in df['ticker'].unique()
                 if t not in EXCLUDE_TICKERS and t not in configs)
    if n_skip > 0:
        print(f"  Skipped {n_skip} tickers with unsupported modes (only static/grace/limit_entry)")
    for t, c in sorted(configs.items()):
        vol_tag = ""
        if c['existing_vol'] > 0:
            vol_tag = f"  Vol>{c['existing_vol']:.0f}%"
        print(f"    {t:<7} {c['tpsl_mode']:<14} "
              f"Prob>{c['prob_thresh']:.2f}  TP={c['tp_pct']:.0%}  SL={c['sl_pct']:.0%}  "
              f"TEST={c['val_b_avg_pnl']:+.2%}{vol_tag}")
    return configs


# ==============================================================================
# PHASE 3: VOLUME FILTER SWEEP
# ==============================================================================

def run_volume_filter_analysis(all_daily_scores, frozen_configs):
    print(f"\n{'='*W}")
    print(f"  PHASE 3: VOLUME FILTER SWEEP")
    print(f"  Grid: vol_filter_pct = {VOL_PCT_GRID}")
    print(f"  Lookback: {VOL_LOOKBACK_DAYS} days trailing median")
    print(f"  {len(VOL_PCT_GRID)} thresholds per ticker")
    print(f"{'='*W}\n")

    results = []
    tickers = sorted(frozen_configs.keys())

    for t_idx, ticker in enumerate(tickers):
        cfg = frozen_configs[ticker]
        mode = cfg['tpsl_mode']
        em = cfg['entry_minute']
        daily_scores = all_daily_scores.get(ticker, {})

        print(f"  [{t_idx+1}/{len(tickers)}] {ticker}  mode={mode}  "
              f"Prob>{cfg['prob_thresh']:.2f}  TP={cfg['tp_pct']:.0%}  SL={cfg['sl_pct']:.0%}")

        if not daily_scores:
            print(f"    SKIP: no scores")
            continue

        try:
            df_ohlcv = load_ticker_ohlcv(ticker)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        vol_baselines = precompute_vol_baselines(df_ohlcv)
        ohlcv_opt  = precompute_ohlcv_arrays(df_ohlcv, OPT_START, OPT_END, vol_baselines)
        ohlcv_test = precompute_ohlcv_arrays(df_ohlcv, TEST_START, TEST_END, vol_baselines)
        del df_ohlcv

        if ohlcv_opt is None:
            print(f"    SKIP: no OHLCV in OPT window")
            continue

        s_cfg = {
            'tp_pct': cfg['tp_pct'], 'sl_pct': cfg['sl_pct'],
            'grace_minutes': cfg['grace_minutes'],
            'dip_pct': cfg['dip_pct'], 'max_wait': cfg['max_wait'],
        }

        # ── Baseline on OPT (vol_filter_pct=0, no filter) ──
        opt_dates = ohlcv_opt['dates']
        opt_times = ohlcv_opt['times']

        probs_opt = np.zeros(len(opt_dates), dtype=np.float64)
        for i in range(len(opt_dates)):
            if opt_times[i] == em:
                probs_opt[i] = daily_scores.get(int(opt_dates[i]), 0.0)

        base_mask_opt = (probs_opt >= cfg['prob_thresh']) & (opt_times == em)
        base_indices_opt = np.where(base_mask_opt)[0]

        if len(base_indices_opt) == 0:
            print(f"    SKIP: no signals in OPT")
            continue

        raw_pnl_opt_base, raw_dates_opt_base = _run_sim(base_indices_opt, ohlcv_opt, s_cfg, mode)
        if len(raw_pnl_opt_base) < MIN_TRADES_ABSOLUTE:
            print(f"    SKIP: only {len(raw_pnl_opt_base)} OPT trades (need {MIN_TRADES_ABSOLUTE})")
            continue

        baseline_opt_pnl = float(np.mean(raw_pnl_opt_base))
        baseline_opt_wr  = float(np.mean(raw_pnl_opt_base > 0))
        baseline_opt_n   = len(raw_pnl_opt_base)
        n_signals_opt    = int(base_mask_opt.sum())

        # ── Baseline on TEST (vol_filter_pct=0) ──
        baseline_test_pnl = 0.0
        baseline_test_wr  = 0.0
        baseline_test_n   = 0
        base_mask_test    = None
        base_indices_test = np.array([], dtype=np.int64)

        if ohlcv_test is not None:
            test_dates = ohlcv_test['dates']
            test_times = ohlcv_test['times']

            probs_test = np.zeros(len(test_dates), dtype=np.float64)
            for i in range(len(test_dates)):
                if test_times[i] == em:
                    probs_test[i] = daily_scores.get(int(test_dates[i]), 0.0)

            base_mask_test = (probs_test >= cfg['prob_thresh']) & (test_times == em)
            base_indices_test = np.where(base_mask_test)[0]

            if len(base_indices_test) > 0:
                raw_pnl_test_base, _ = _run_sim(base_indices_test, ohlcv_test, s_cfg, mode)
                if len(raw_pnl_test_base) > 0:
                    baseline_test_pnl = float(np.mean(raw_pnl_test_base))
                    baseline_test_wr  = float(np.mean(raw_pnl_test_base > 0))
                    baseline_test_n   = len(raw_pnl_test_base)

        # ── Sweep vol_filter_pct on OPT ──
        best_vpct = 0
        best_opt_avg = baseline_opt_pnl

        for vpct in VOL_PCT_GRID:
            if vpct == 0:
                continue
            vol_ok = _compute_vol_pct_ok(
                ohlcv_opt['volumes'], ohlcv_opt['times'], ohlcv_opt['dates'],
                vol_baselines, em, vpct)
            filt_mask = base_mask_opt & vol_ok
            filt_indices = np.where(filt_mask)[0]
            if len(filt_indices) == 0:
                continue

            pnl, _ = _run_sim(filt_indices, ohlcv_opt, s_cfg, mode)
            if len(pnl) < MIN_TRADES_ABSOLUTE:
                continue
            avg = float(np.mean(pnl))
            if avg > best_opt_avg:
                best_opt_avg = avg
                best_vpct = vpct

        # ── Validate best vpct on TEST ──
        filtered_test_pnl = baseline_test_pnl
        filtered_test_wr  = baseline_test_wr
        filtered_test_n   = baseline_test_n
        test_blocked_pct  = 0.0
        opt_blocked_pct   = 0.0

        if best_vpct > 0:
            vol_ok_opt = _compute_vol_pct_ok(
                ohlcv_opt['volumes'], ohlcv_opt['times'], ohlcv_opt['dates'],
                vol_baselines, em, best_vpct)
            n_signals_blocked_opt = int((base_mask_opt & ~vol_ok_opt).sum())
            opt_blocked_pct = n_signals_blocked_opt / max(1, n_signals_opt) * 100

            if ohlcv_test is not None and base_mask_test is not None and len(base_indices_test) > 0:
                vol_ok_test = _compute_vol_pct_ok(
                    ohlcv_test['volumes'], ohlcv_test['times'], ohlcv_test['dates'],
                    vol_baselines, em, best_vpct)
                filt_mask_test = base_mask_test & vol_ok_test
                filt_indices_test = np.where(filt_mask_test)[0]
                n_signals_test = int(base_mask_test.sum())
                n_blocked_test = int((base_mask_test & ~vol_ok_test).sum())
                test_blocked_pct = n_blocked_test / max(1, n_signals_test) * 100

                if len(filt_indices_test) > 0:
                    pnl_test, _ = _run_sim(filt_indices_test, ohlcv_test, s_cfg, mode)
                    if len(pnl_test) > 0:
                        filtered_test_pnl = float(np.mean(pnl_test))
                        filtered_test_wr  = float(np.mean(pnl_test > 0))
                        filtered_test_n   = len(pnl_test)
                    else:
                        filtered_test_pnl = 0.0
                        filtered_test_wr  = 0.0
                        filtered_test_n   = 0
                else:
                    filtered_test_pnl = 0.0
                    filtered_test_wr  = 0.0
                    filtered_test_n   = 0

        delta = filtered_test_pnl - baseline_test_pnl if best_vpct > 0 else 0.0
        filter_found = best_vpct > 0
        filter_helps = filter_found and delta > 0

        result = {
            'ticker': ticker,
            'mode': mode,
            'prob_thresh': cfg['prob_thresh'],
            'tp_pct': cfg['tp_pct'],
            'sl_pct': cfg['sl_pct'],
            'grace_minutes': cfg['grace_minutes'],
            'dip_pct': cfg['dip_pct'],
            'max_wait': cfg['max_wait'],
            'entry_minute': em,
            'experiment': cfg['experiment'],
            'model_version': cfg['model_version'],
            'baseline_opt_pnl': baseline_opt_pnl,
            'baseline_opt_wr': baseline_opt_wr,
            'baseline_opt_n': baseline_opt_n,
            'baseline_test_pnl': baseline_test_pnl,
            'baseline_test_wr': baseline_test_wr,
            'baseline_test_n': baseline_test_n,
            'best_vpct': best_vpct,
            'filtered_opt_pnl': best_opt_avg,
            'filtered_test_pnl': filtered_test_pnl,
            'filtered_test_wr': filtered_test_wr,
            'filtered_test_n': filtered_test_n,
            'opt_blocked_pct': opt_blocked_pct,
            'test_blocked_pct': test_blocked_pct,
            'delta': delta,
            'filter_helps': filter_helps,
            'filter_found': filter_found,
        }
        results.append(result)

        if filter_found:
            verdict = "BETTER" if filter_helps else "WORSE"
            print(f"    Baseline OPT: {baseline_opt_pnl:+.2%}  ({baseline_opt_n} trades)")
            print(f"    Baseline TEST: {baseline_test_pnl:+.2%}  ({baseline_test_n} trades)")
            print(f"    Filter: vol_filter_pct >= {best_vpct}%")
            print(f"    Filtered OPT:  {best_opt_avg:+.2%}  (blocked {opt_blocked_pct:.0f}% signals)")
            print(f"    Filtered TEST: {filtered_test_pnl:+.2%}  ({filtered_test_n} trades, blocked {test_blocked_pct:.0f}%)  [{verdict}]")
            print(f"    Delta: {delta:+.2%}")
        else:
            print(f"    Baseline OPT: {baseline_opt_pnl:+.2%}  ({baseline_opt_n} trades)")
            print(f"    Baseline TEST: {baseline_test_pnl:+.2%}  ({baseline_test_n} trades)")
            print(f"    No vol_filter_pct improved OPT avg PnL")

        del ohlcv_opt, ohlcv_test
        gc.collect()

    return results


# ==============================================================================
# PHASE 4: CHARTS + SUMMARY
# ==============================================================================

def print_summary_and_charts(results):
    print(f"\n\n{'='*W}")
    print(f"  VOLUME FILTER ANALYSIS — S3 LONG SUMMARY")
    print(f"{'='*W}")

    if not results:
        print("  No results to display.")
        return

    df = pd.DataFrame(results)

    # ── Summary table ──
    print(f"\n  {'Ticker':<7} {'Mode':<14} {'BL_OPT':>7} {'BL_TEST':>8} {'FILT_TEST':>9} "
          f"{'Delta':>7} {'VolPct':>6} {'Blk%':>5} {'BL_N':>5} {'FLT_N':>5} {'Verdict':>8}")
    print(f"  {'─'*90}")

    for _, r in df.sort_values('delta', ascending=False).iterrows():
        if r['filter_found']:
            verdict = "BETTER" if r['filter_helps'] else "WORSE"
            print(f"  {r['ticker']:<7} {r['mode']:<14} "
                  f"{r['baseline_opt_pnl']:>+6.2%} {r['baseline_test_pnl']:>+7.2%} "
                  f"{r['filtered_test_pnl']:>+8.2%} {r['delta']:>+6.2%} "
                  f"{r['best_vpct']:>5.0f}% {r['test_blocked_pct']:>4.0f}% "
                  f"{r['baseline_test_n']:>5.0f} {r['filtered_test_n']:>5.0f} {verdict:>8}")
        else:
            print(f"  {r['ticker']:<7} {r['mode']:<14} "
                  f"{r['baseline_opt_pnl']:>+6.2%} {r['baseline_test_pnl']:>+7.2%} "
                  f"{'N/A':>9} {'0.00%':>7} "
                  f"{'---':>6} {'0%':>5} "
                  f"{r['baseline_test_n']:>5.0f} {r['baseline_test_n']:>5.0f} {'NO_FILT':>8}")

    # ── Aggregate stats ──
    n_total = len(df)
    n_filter_found = int(df['filter_found'].sum())
    n_helps = int(df['filter_helps'].sum())
    n_hurts = n_filter_found - n_helps
    n_no_filter = n_total - n_filter_found

    avg_baseline = df['baseline_test_pnl'].mean()
    filt_mask = df['filter_found']
    if filt_mask.any():
        avg_filtered = df.loc[filt_mask, 'filtered_test_pnl'].mean()
        avg_delta = df.loc[filt_mask, 'delta'].mean()
    else:
        avg_filtered = avg_baseline
        avg_delta = 0.0

    # Total PnL
    total_baseline = (df['baseline_test_pnl'] * df['baseline_test_n']).sum()
    total_baseline_n = int(df['baseline_test_n'].sum())
    selective_pnl = 0.0
    selective_n = 0
    for _, r in df.iterrows():
        if r['filter_helps']:
            selective_pnl += r['filtered_test_pnl'] * r['filtered_test_n']
            selective_n += int(r['filtered_test_n'])
        else:
            selective_pnl += r['baseline_test_pnl'] * r['baseline_test_n']
            selective_n += int(r['baseline_test_n'])

    print(f"\n  AGGREGATE:")
    print(f"    {n_total} tickers analyzed")
    print(f"    {n_filter_found} found a vol_filter_pct that beat OPT baseline")
    print(f"    {n_helps} filter HELPS on TEST (delta > 0)")
    print(f"    {n_hurts} filter HURTS on TEST (delta <= 0)")
    print(f"    {n_no_filter} no filter improved OPT")
    print(f"    Avg baseline TEST PnL: {avg_baseline:+.2%}")
    print(f"    Avg filtered TEST PnL: {avg_filtered:+.2%}")
    print(f"    Avg delta: {avg_delta:+.2%}")
    print(f"\n  TOTAL PnL:")
    print(f"    Baseline:  {total_baseline:+.2%} across {total_baseline_n} trades")
    print(f"    Selective: {selective_pnl:+.2%} across {selective_n} trades")
    print(f"    Delta:     {selective_pnl - total_baseline:+.2%}")

    # ── Charts ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Per-ticker PnL bars (baseline vs filtered)
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('delta', ascending=True)
    y_pos = range(len(df_sorted))
    baseline_vals = df_sorted['baseline_test_pnl'].values * 100
    filtered_vals = df_sorted.apply(
        lambda r: r['filtered_test_pnl'] * 100 if r['filter_found'] else r['baseline_test_pnl'] * 100,
        axis=1).values
    ax1.barh(y_pos, baseline_vals, height=0.4, align='edge', label='Baseline', color='steelblue', alpha=0.7)
    ax1.barh([y + 0.4 for y in y_pos], filtered_vals, height=0.4, align='edge', label='Filtered', color='green', alpha=0.7)
    ax1.set_yticks([y + 0.4 for y in y_pos])
    ax1.set_yticklabels(df_sorted['ticker'].values, fontsize=8)
    ax1.set_xlabel('TEST Avg PnL (%)')
    ax1.set_title('Baseline vs Filtered TEST PnL')
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.legend(fontsize=8)
    ax1.grid(axis='x', alpha=0.3)

    # Panel 2: Selected vol_filter_pct per ticker
    ax2 = axes[0, 1]
    df_with_filter = df[df['filter_found']].sort_values('ticker')
    if len(df_with_filter) > 0:
        x_pos = range(len(df_with_filter))
        colors = ['green' if d > 0 else 'red' for d in df_with_filter['delta']]
        ax2.bar(list(x_pos), df_with_filter['best_vpct'].values,
                color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xticks(list(x_pos))
        ax2.set_xticklabels(df_with_filter['ticker'].values, rotation=45, fontsize=8)
        ax2.set_ylabel('vol_filter_pct (%)')
        ax2.set_title('Selected Volume Threshold per Ticker')
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No filters found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Selected Volume Threshold per Ticker')

    # Panel 3: Scatter — % trades blocked vs PnL delta
    ax3 = axes[1, 0]
    if len(df_with_filter) > 0:
        colors_scatter = ['green' if d > 0 else 'red' for d in df_with_filter['delta']]
        ax3.scatter(df_with_filter['test_blocked_pct'], df_with_filter['delta'] * 100,
                    c=colors_scatter, s=60, edgecolors='black', linewidth=0.5, alpha=0.8)
        for _, r in df_with_filter.iterrows():
            ax3.annotate(r['ticker'], (r['test_blocked_pct'], r['delta'] * 100),
                        fontsize=7, ha='left', va='bottom')
        ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax3.set_xlabel('Trades Blocked (%)')
        ax3.set_ylabel('PnL Delta (%)')
        ax3.set_title('Filter Impact: Blocked % vs PnL Change')
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No filters found', ha='center', va='center', transform=ax3.transAxes)

    # Panel 4: Total PnL comparison
    ax4 = axes[1, 1]
    categories = ['Baseline\n(no filter)', 'Selective\n(filter where helps)']
    values = [total_baseline, selective_pnl]
    bar_colors = ['steelblue', 'green']
    ax4.bar(categories, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_ylabel('Total TEST PnL (%)')
    ax4.set_title(f'Total PnL Comparison\nBaseline: {total_baseline_n} trades  |  Selective: {selective_n} trades')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        ax4.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.suptitle(f'Volume Filter Analysis — S3 LONG\n'
                 f'OPT: {OPT_START} to {OPT_END}  |  TEST: {TEST_START} to {TEST_END}  |  '
                 f'{n_total} tickers, {len(VOL_PCT_GRID)} vol thresholds',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_path = '/tmp/ra_moe_models/vol_filter_analysis_s3_long.png'
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved: {chart_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()

    return df


# ==============================================================================
# PHASE 5: OPTIONAL BQ WRITE
# ==============================================================================

def write_results_to_bq(results):
    if not WRITE_TO_BQ:
        print(f"\n  WRITE_TO_BQ = False — skipping BQ write.")
        print(f"  Set WRITE_TO_BQ = True and re-run to save as '{MODEL_VERSION_OUT}'")
        return

    print(f"\n{'='*W}")
    print(f"  PHASE 5: WRITING TO BQ as model_version='{MODEL_VERSION_OUT}'")
    print(f"{'='*W}")

    if WIPE_EXISTING:
        try:
            del_q = (f"DELETE FROM `{BQ_TABLE_ID}` "
                     f"WHERE model_version = '{MODEL_VERSION_OUT}'")
            _bq_client.query(del_q).result()
            print(f"  WIPE: deleted existing model_version='{MODEL_VERSION_OUT}'")
        except Exception as e:
            if 'streaming buffer' in str(e).lower():
                copy_q = (f"CREATE OR REPLACE TABLE `{BQ_TABLE_ID}` AS "
                          f"SELECT * FROM `{BQ_TABLE_ID}` "
                          f"WHERE model_version != '{MODEL_VERSION_OUT}'")
                _bq_client.query(copy_q).result()
                print(f"  WIPE: rebuilt without '{MODEL_VERSION_OUT}'")
            else:
                raise

    run_id = str(uuid.uuid4())
    n_written = 0

    for r in results:
        vpct = r['best_vpct'] if r['filter_helps'] else 0.0

        test_pnl = r['filtered_test_pnl'] if r['filter_helps'] else r['baseline_test_pnl']
        test_wr  = r['filtered_test_wr'] if r['filter_helps'] else r['baseline_test_wr']
        test_n   = r['filtered_test_n'] if r['filter_helps'] else r['baseline_test_n']

        mode = r['mode']

        row = {
            'run_id': run_id,
            'run_timestamp': pd.Timestamp.utcnow().isoformat(),
            'experiment': f"{EXPERIMENT_OUT}_{mode}",
            'ticker': r['ticker'],
            'seed': 0,
            'entry_time': ENTRY_TIME,
            'entry_minute': r['entry_minute'],
            'prob_thresh': r['prob_thresh'],
            'tp_pct': r['tp_pct'],
            'sl_pct': r['sl_pct'],
            'vix_threshold': NO_FILTER_DEFAULTS['vix_threshold'],
            'tlt_crash': NO_FILTER_DEFAULTS['tlt_crash'],
            'spy_ignore': NO_FILTER_DEFAULTS['spy_ignore'],
            'min_vol_bars': 0,
            'vol_filter_pct': float(vpct),
            'opt_avg_pnl': r['baseline_opt_pnl'],
            'opt_win_rate': r['baseline_opt_wr'],
            'opt_freq': 0.0,
            'opt_count': r['baseline_opt_n'],
            'val_a_avg_pnl': r['filtered_opt_pnl'] if r['filter_helps'] else r['baseline_opt_pnl'],
            'val_a_win_rate': r['baseline_opt_wr'],
            'val_a_freq': 0.0,
            'val_a_count': r['baseline_opt_n'],
            'survived_a': True,
            'val_b_avg_pnl': test_pnl,
            'val_b_win_rate': test_wr,
            'val_b_freq': 0.0,
            'val_b_count': test_n,
            'survived_b': test_pnl > 0,
            'opt_start': OPT_START,
            'opt_end': OPT_END,
            'val_a_start': OPT_START,
            'val_a_end': OPT_END,
            'val_b_start': TEST_START,
            'val_b_end': TEST_END,
            'tpsl_mode': mode,
            'tp_mult': None, 'sl_mult': None,
            'trail_start_mult': None, 'trail_end_mult': None,
            'model_version': MODEL_VERSION_OUT,
            'v0_fixed': True,
            'is_ensemble': True, 'n_seeds_avg': 15,
            'regime_tp_pct': None, 'regime_sl_pct': None,
            'regime_lookback': None, 'regime_spike_thresh': None,
            'grace_minutes': r['grace_minutes'] if mode == 'grace' else None,
            'dip_pct': r['dip_pct'] if mode == 'limit_entry' else None,
            'max_wait_minutes': r['max_wait'] if mode == 'limit_entry' else None,
            'pre_range_filter_pct': 0.0,
        }

        try:
            errors = _bq_client.insert_rows_json(BQ_TABLE_ID, [row])
            if errors:
                print(f"    BQ error {r['ticker']}: {errors[0]}")
            else:
                filter_tag = f"vol>={vpct:.0f}%" if r['filter_helps'] else "NO_FILTER"
                print(f"    BQ: {r['ticker']}/{mode}  {filter_tag}  TEST={test_pnl:+.2%}")
                n_written += 1
        except Exception as e:
            print(f"    BQ save failed {r['ticker']}: {e}")

    print(f"\n  Written {n_written}/{len(results)} rows to BQ as '{MODEL_VERSION_OUT}'")


# ==============================================================================
# RUN ALL PHASES
# ==============================================================================

# Phase 2 first: load configs to know which tickers/experiments to score
frozen_configs = load_frozen_configs()

if not frozen_configs:
    print("\nERROR: No frozen configs loaded. Cannot proceed.")
else:
    # Phase 1: Load scores (needs configs to know experiments)
    all_daily_scores = run_scoring_phase(frozen_configs)

    # Phase 3: Volume filter sweep
    results = run_volume_filter_analysis(all_daily_scores, frozen_configs)

    # Phase 4: Summary + charts
    summary_df = print_summary_and_charts(results)

    # Phase 5: Optional BQ write
    write_results_to_bq(results)

    # Grand total
    n_helps = sum(1 for r in results if r['filter_helps'])
    n_total = len(results)
    print(f"\n{'='*W}")
    print(f"  COMPLETE — {n_total} tickers, {n_helps} improved by volume filter")
    print(f"  model_version_out: '{MODEL_VERSION_OUT}' (WRITE_TO_BQ={WRITE_TO_BQ})")
    print(f"{'='*W}")
