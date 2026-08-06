"""
optimize_wf_v12_170_to_175.py — WF Optimizer v12 for LONG/Spike Model (All Folds 170-175)
============================================================================================
Optimizes all 6 folds (170-175, standard transformer) independently in a single run.
Each fold gets its own model_version tag: robust_wf_v12_170 through robust_wf_v12_175.
No stitching — each fold's 15 seeds optimized separately.

Ticker-first loop: OHLCV loaded once per ticker, reused across all folds.
All folds share the same OPT/VAL/TEST windows.

LONG trade simulation: entry = BUY at signal bar, TP on price rise, SL on price drop.

REDUCED GRID (anti-overfit):
  - 3 modes only: static, grace, limit_entry
  - 7 prob thresholds (0.35-0.65)
  - 5 static + 3 grace + 10 limit_entry = 18 configs
  - Total: 126 grid points per seed per ticker (vs 738 in v12 original)

Ensemble val_a fix included from day one.
"""

import gc
import ctypes
import pandas as pd
import numpy as np
import io
import itertools
from google.cloud import storage, bigquery as bq
from numba import jit
try:
    from google.colab import auth
except ImportError:
    auth = None
import pytz
from datetime import timedelta
import uuid
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# MODES — reduced to 3 (from 9) to combat overfitting
# ==============================================================================
ALL_MODES = ['static', 'grace', 'limit_entry']
RUN_MODES = ['static', 'grace', 'limit_entry']
TPSL_MODE = ALL_MODES[0]

# ==============================================================================
# BIGQUERY
# ==============================================================================
BQ_TABLE_ID = "ys-time-series.scoring_logs.wf_optimization_results"

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
    table = bq_client.get_table(table_id)
    existing = {f.name for f in table.schema}
    new_fields = [f for f in schema if f.name not in existing]
    if new_fields:
        for f in new_fields:
            bq_client.query(f"ALTER TABLE `{table_id}` ADD COLUMN IF NOT EXISTS {f.name} {f.field_type}").result()

_bq_client = bq.Client()
_ensure_wf_table(_bq_client, BQ_TABLE_ID)


def save_row_to_bq(row_dict, run_id, experiment,
                    opt_s, opt_e, va_s, va_e, vb_s='', vb_e=''):
    try:
        h, m_ = map(int, row_dict['entry_time'].split(':'))
        row = {
            'run_id': run_id, 'run_timestamp': pd.Timestamp.utcnow().isoformat(),
            'experiment': experiment,
            'ticker': row_dict['Ticker'], 'seed': row_dict['Seed'],
            'entry_time': row_dict['entry_time'], 'entry_minute': h * 60 + m_,
            'prob_thresh': float(row_dict['prob_thresh']),
            'tp_pct': float(row_dict['tp_pct']),
            'sl_pct': float(row_dict['sl_pct']),
            'vix_threshold': float(row_dict['vix_threshold']),
            'tlt_crash': float(row_dict['tlt_crash']),
            'spy_ignore': float(row_dict['spy_ignore']),
            'min_vol_bars': 0,
            'vol_filter_pct': float(row_dict.get('vol_filter_pct', 0)),
            'opt_avg_pnl': float(row_dict['Avg_PnL']),
            'opt_win_rate': float(row_dict['Win_Rate']),
            'opt_freq': float(row_dict['Freq']),
            'opt_count': int(row_dict['Count']),
            'val_a_avg_pnl': float(row_dict.get('val_a_avg_pnl', 0)),
            'val_a_win_rate': float(row_dict.get('val_a_win_rate', 0)),
            'val_a_freq': float(row_dict.get('val_a_freq', 0)),
            'val_a_count': int(row_dict.get('val_a_count', 0)),
            'survived_a': bool(row_dict.get('survived_a', False)),
            'val_b_avg_pnl': float(row_dict.get('val_b_avg_pnl', 0)),
            'val_b_win_rate': float(row_dict.get('val_b_win_rate', 0)),
            'val_b_freq': float(row_dict.get('val_b_freq', 0)),
            'val_b_count': int(row_dict.get('val_b_count', 0)),
            'survived_b': bool(row_dict.get('survived_b', False)),
            'opt_start': opt_s, 'opt_end': opt_e,
            'val_a_start': va_s, 'val_a_end': va_e,
            'val_b_start': vb_s or '', 'val_b_end': vb_e or '',
            'tpsl_mode': TPSL_MODE,
            'tp_mult': None,
            'sl_mult': None,
            'trail_start_mult': None,
            'trail_end_mult': None,
            'model_version': MODEL_VERSION_TAG,
            'v0_fixed': True,
            'is_ensemble': bool(row_dict.get('is_ensemble', False)),
            'n_seeds_avg': int(row_dict.get('n_seeds_avg', 0)),
            'regime_tp_pct': None,
            'regime_sl_pct': None,
            'regime_lookback': None,
            'regime_spike_thresh': None,
            'grace_minutes': int(row_dict.get('grace_minutes', 0)) if TPSL_MODE == 'grace' else None,
            'dip_pct': float(row_dict.get('dip_pct', 0)) if TPSL_MODE == 'limit_entry' else None,
            'max_wait_minutes': int(row_dict.get('max_wait', 0)) if TPSL_MODE == 'limit_entry' else None,
            'pre_range_filter_pct': float(row_dict.get('pre_range_filter_pct', 0)),
        }
        errors = _bq_client.insert_rows_json(BQ_TABLE_ID, [row])
        if errors:
            print(f"      BQ error: {errors[0]}")
        else:
            label = "ENS" if row_dict.get('is_ensemble') else f"S{row_dict['Seed']}"
            print(f"      BQ: saved {row_dict['Ticker']}/{label}")
    except Exception as e:
        print(f"      BQ save failed: {e}")


# ==============================================================================
# CONFIG — Exp 170-175 LONG/Spike Model (Standard Transformer)
# ==============================================================================
BUCKET_NAME     = "time-series-bucket-ys"
TICKER_FILE_MAP = {'BITF': 'KEEL'}
TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}
DEPLOY_PNL_THRESHOLD = 0.018
MIN_BQ_TRADES   = 3

_FULL_TICKERS = [
    'LUNR', 'RCAT', 'ASTS', 'BBAI', 'ACHR', 'MVIS',
    'CIFR', 'IREN', 'HUT', 'KEEL', 'CLSK', 'APLD',
    'WULF', 'MARA', 'RIOT', 'MSTR', 'RKLB', 'JOBY',
    'ALAB', 'BGDE', 'PLUG', 'FCEL', 'BTBT', 'BE',
    'OKLO', 'SMR', 'FLNC', 'COIN', 'HOOD', 'SMCI',
    'KTOS', 'POWL', 'MOD', 'NNDM', 'CORZ', 'NBIS',
    'AVAV', 'MP', 'PL', 'RDW',
    'AMD', 'NVDA', 'TSLA', 'PLTR', 'SOUN', 'VRT', 'VST',
    'GD', 'GE', 'GEV', 'LHX', 'LMT', 'NOC', 'RTX',
    'BHP', 'CAT', 'CCJ', 'CEG', 'COP', 'CVX', 'XOM',
    'DUK', 'EQIX', 'ETN', 'LIN', 'NEE',
    'BKSY', 'MNTS', 'VOYG',
]

FOLD_CONFIGS = [
    {'fold': 170, 'experiment': '170_XSECTOR', 'model_version': 'robust_wf_v12_170', 'label': 'Exp 170 — cutoff Aug 2025'},
    {'fold': 171, 'experiment': '171_XSECTOR', 'model_version': 'robust_wf_v12_171', 'label': 'Exp 171 — cutoff Sep 2025'},
    {'fold': 172, 'experiment': '172_XSECTOR', 'model_version': 'robust_wf_v12_172', 'label': 'Exp 172 — cutoff Oct 2025'},
    {'fold': 173, 'experiment': '173_XSECTOR', 'model_version': 'robust_wf_v12_173', 'label': 'Exp 173 — cutoff Nov 2025'},
    {'fold': 174, 'experiment': '174_XSECTOR', 'model_version': 'robust_wf_v12_174', 'label': 'Exp 174 — cutoff Dec 2025'},
    {'fold': 175, 'experiment': '175_XSECTOR', 'model_version': 'robust_wf_v12_175', 'label': 'Exp 175 — cutoff Jan 2026'},
]

ACTIVE_FOLDS = [170, 171, 172, 173, 174, 175]

WF_ROUNDS = [
    {'name': 'R1', 'opt': ('2025-08-01', '2026-01-31'),
                   'val': ('2026-02-01', '2026-03-31'), 'min_trades': 3},
]
TEST_START = '2026-04-01'
TEST_END   = '2026-07-28'

NO_FILTER_DEFAULTS = {'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0}

EXIT_TIME_MIN        = 780
MIN_TRADES_PER_MONTH = 2.0
MIN_TRADES_ABSOLUTE  = 5
VOL_LOOKBACK_DAYS    = 30
VOL_PCT_GRID         = [0, 25]
SLIPPAGE_PCT         = 0.0015

FIXED_ENTRY_TIME   = '06:15'
FIXED_ENTRY_MINUTE = 375

MODEL_VERSION_TAG = None
SKIP_COMPLETED = True
WIPE_EXISTING  = True
TICKER_SUBSET  = []

# ==============================================================================
# REDUCED GRID — anti-overfit (126 grid points vs 738 original)
# ==============================================================================
TPSL_PAIRS = {
    'static': [
        {'tp_pct': 0.06, 'sl_pct': 0.03},
        {'tp_pct': 0.08, 'sl_pct': 0.04},
        {'tp_pct': 0.12, 'sl_pct': 0.06},
        {'tp_pct': 0.16, 'sl_pct': 0.08},
        {'tp_pct': 0.16, 'sl_pct': 0.12},
    ],
}

GRACE_CONFIGS = [
    {'tp_pct': 0.08, 'sl_pct': 0.04, 'grace_minutes': 15},
    {'tp_pct': 0.12, 'sl_pct': 0.06, 'grace_minutes': 15},
    {'tp_pct': 0.16, 'sl_pct': 0.08, 'grace_minutes': 15},
]

LIMIT_ENTRY_CONFIGS = [
    {'tp_pct': 0.08, 'sl_pct': 0.04, 'dip_pct': 0.005, 'max_wait': 30},
    {'tp_pct': 0.08, 'sl_pct': 0.04, 'dip_pct': 0.01, 'max_wait': 30},
    {'tp_pct': 0.08, 'sl_pct': 0.04, 'dip_pct': 0.0125, 'max_wait': 30},
    {'tp_pct': 0.16, 'sl_pct': 0.08, 'dip_pct': 0.01, 'max_wait': 30},
    {'tp_pct': 0.16, 'sl_pct': 0.08, 'dip_pct': 0.0125, 'max_wait': 30},
    {'tp_pct': 0.16, 'sl_pct': 0.08, 'dip_pct': 0.015, 'max_wait': 30},
    {'tp_pct': 0.12, 'sl_pct': 0.06, 'dip_pct': 0.0175, 'max_wait': 30},
    {'tp_pct': 0.12, 'sl_pct': 0.06, 'dip_pct': 0.02, 'max_wait': 30},
    {'tp_pct': 0.16, 'sl_pct': 0.08, 'dip_pct': 0.025, 'max_wait': 60},
    {'tp_pct': 0.20, 'sl_pct': 0.10, 'dip_pct': 0.04, 'max_wait': 90},
]

PROB_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

PRE_RANGE_FILTER_GRID = [0, 75]

_existing_limit_keys = set()


# ==============================================================================
# LONG SIMULATION ENGINES (Numba) — TP above entry, SL below entry
# ==============================================================================

@jit(nopython=True)
def run_trade_sim(entry_indices, prices, highs, lows, date_ints, time_mins,
                  tp, sl, exit_time):
    """Static TP/SL — LONG: TP above entry, SL below entry."""
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
    """Grace period on SL — LONG: ignore SL for first N minutes after entry."""
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
    """Limit entry — LONG: wait for price to DIP down, then buy at lower level."""
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


def _run_sim(indices, ohlcv_arrays, s_cfg):
    prices = ohlcv_arrays['prices']
    highs  = ohlcv_arrays['highs']
    lows   = ohlcv_arrays['lows']
    dates  = ohlcv_arrays['dates']
    times  = ohlcv_arrays['times'].astype(np.int64)

    if TPSL_MODE == 'grace':
        pnl, d = run_trade_sim_grace(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN, s_cfg['grace_minutes'])
    elif TPSL_MODE == 'limit_entry':
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
# VOLUME BASELINES
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


def precompute_pre_entry_range_baselines(df_ohlcv, entry_minute=FIXED_ENTRY_MINUTE, lookback_days=30):
    times = (df_ohlcv.index.hour * 60 + df_ohlcv.index.minute).values
    dates = (df_ohlcv.index.year * 10000 + df_ohlcv.index.month * 100
             + df_ohlcv.index.day).values.astype(np.int64)
    closes = df_ohlcv['Close'].values
    highs  = df_ohlcv['High'].values
    lows   = df_ohlcv['Low'].values

    open_minute = 390
    unique_dates = np.unique(dates)
    day_ranges = {}
    for d in unique_dates:
        mask = dates == d
        d_times  = times[mask]
        d_highs  = highs[mask]
        d_lows   = lows[mask]
        d_closes = closes[mask]
        pre_mask = (d_times >= open_minute) & (d_times <= entry_minute)
        if pre_mask.sum() < 2:
            continue
        pre_high = d_highs[pre_mask].max()
        pre_low  = d_lows[pre_mask].min()
        open_price = d_closes[d_times == open_minute]
        if len(open_price) == 0 or open_price[0] <= 0:
            continue
        day_ranges[int(d)] = (pre_high - pre_low) / open_price[0]

    baselines = {}
    sorted_dates = sorted(day_ranges.keys())
    for i, d in enumerate(sorted_dates):
        start_i = max(0, i - lookback_days)
        if i - start_i < 5:
            baselines[d] = 0.0
            continue
        prev_ranges = [day_ranges[sorted_dates[j]] for j in range(start_i, i)]
        baselines[d] = float(np.median(prev_ranges))
    return baselines


def _compute_pre_range_ok(dates, pre_range_baselines, day_range_vals, threshold_pct):
    n = len(dates)
    ok = np.ones(n, dtype=np.bool_)
    if threshold_pct <= 0:
        return ok
    unique_dates = np.unique(dates)
    for d in unique_dates:
        d_int = int(d)
        baseline = pre_range_baselines.get(d_int, 0.0)
        current  = day_range_vals.get(d_int, 0.0)
        if baseline > 0 and current > 0:
            pct = (current / baseline) * 100
            if pct > threshold_pct:
                mask = dates == d
                ok[mask] = False
    return ok


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


def preload_regime_data(eval_start, eval_end):
    print("Loading regime data (VIX, SPY, TLT)...")
    pst = pytz.timezone('US/Pacific')

    def _load_regime_asset(ticker):
        clean = TICKER_FILE_MAP.get(ticker, ticker)
        for suffix in ['_master_inference', '_1m_full_historical_data']:
            uri = f"gs://{BUCKET_NAME}/Data/Inference_v2/{clean}{suffix}.parquet"
            try:
                return _load_parquet_with_dt_index(uri)
            except Exception:
                continue
        return None

    vix_df = _load_regime_asset('VIX')
    spy_df = _load_regime_asset('SPY')
    tlt_df = _load_regime_asset('TLT')

    spy_daily = spy_df['Close'].resample('D').last().dropna() if spy_df is not None else pd.Series(dtype=float)
    tlt_daily = tlt_df['Close'].resample('D').last().dropna() if tlt_df is not None else pd.Series(dtype=float)

    entry_minutes = list(range(360, 391))

    start = pd.Timestamp(eval_start, tz=pst) - timedelta(days=10)
    end   = pd.Timestamp(eval_end, tz=pst) + timedelta(days=1)

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

    all_dates = pd.date_range(start, end, freq='D', tz=pst)
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
            tp = tlt_prev.get(d_int)
            if tc is not None and tp is not None and tp != 0:
                t_gap = ((tc - tp) / tp) * 100
            regime_dict[(d_int, em)] = np.array([v_val, t_gap, s_gap])

    print(f"   Regime data: {len(regime_dict)} (date, minute) combos")
    return regime_dict, vix_lk


def load_ticker_ohlcv(ticker):
    clean = TICKER_FILE_MAP.get(ticker.replace("I:", "").replace("X:", ""),
                                 ticker.replace("I:", "").replace("X:", ""))
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{BUCKET_NAME}/Data/Inference_v2/{clean}{suffix}.parquet"
        try:
            return _load_parquet_with_dt_index(uri)
        except Exception:
            continue
    raise FileNotFoundError(f"No OHLCV data found for {ticker}")


def load_seed_scores(ticker, experiment):
    clean = TICKER_FILE_MAP.get(ticker.replace("I:", "").replace("X:", ""),
                                 ticker.replace("I:", "").replace("X:", ""))
    results_dir = f"Algos/Results_all_v2/Universal_Exp{experiment}"
    client = storage.Client()
    prefix = f"{results_dir}/{clean}/Seed_"
    blobs = list(client.bucket(BUCKET_NAME).list_blobs(prefix=prefix))

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
        seed_series[seed_id] = df['Prob_Spike'].sort_index()
    return seed_series


# ==============================================================================
# HELPERS
# ==============================================================================
def _attach_regime(raw_dates, regime_dict, n, entry_minute):
    trade_vix = np.zeros(n)
    trade_tlt = np.zeros(n)
    trade_spy = np.zeros(n)
    for i in range(n):
        key = (raw_dates[i], entry_minute)
        if key in regime_dict:
            v = regime_dict[key]
            trade_vix[i] = v[0]
            trade_tlt[i] = v[1]
            trade_spy[i] = v[2]
    return trade_vix, trade_tlt, trade_spy


def _apply_filter(raw_pnl, trade_vix, trade_tlt, trade_spy, vix_t, tlt_c, spy_i):
    trap = (trade_vix > vix_t) & (trade_tlt < tlt_c) & (trade_spy > spy_i)
    return raw_pnl[~trap]


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
        'vol_baselines': vol_baselines or {},
    }
    return result


# ==============================================================================
# TICKER LIST
# ==============================================================================
def get_ticker_list():
    tickers = sorted([t for t in _FULL_TICKERS if t not in EXCLUDE_TICKERS])
    return tickers


# ==============================================================================
# STABILITY-WEIGHTED PARAMETER SELECTION
# ==============================================================================
def _select_robust_params(all_results):
    if not all_results:
        return None

    strategy_groups = {}
    for r in all_results:
        if TPSL_MODE == 'grace':
            key = (r['entry_time'], r['prob_thresh'],
                   r.get('tp_pct', 0), r.get('sl_pct', 0),
                   r.get('grace_minutes', 0))
        elif TPSL_MODE == 'limit_entry':
            key = (r['entry_time'], r['prob_thresh'],
                   r.get('tp_pct', 0), r.get('sl_pct', 0),
                   r.get('dip_pct', 0), r.get('max_wait', 0))
        else:
            key = (r['entry_time'], r['prob_thresh'],
                   r.get('tp_pct', 0), r.get('sl_pct', 0))
        strategy_groups.setdefault(key, []).append(r)

    best = None
    best_robust_score = -999.0
    for key, results in strategy_groups.items():
        pnls = np.array([r['Avg_PnL'] for r in results])
        mean_pnl = float(np.mean(pnls))
        if mean_pnl <= 0:
            continue
        frac_positive = float(np.mean(pnls > 0))

        consistency = 1.0
        best_in_group = max(results, key=lambda r: r['Avg_PnL'])
        raw = best_in_group.get('_raw_pnl')
        if raw is not None and len(raw) >= 4:
            std_pnl = float(np.std(raw))
            consistency = max(0.0, 1.0 - std_pnl / abs(mean_pnl))

        robust_score = mean_pnl * (0.5 + 0.5 * frac_positive) * (0.5 + 0.5 * consistency)
        if robust_score > best_robust_score:
            best_robust_score = robust_score
            best = best_in_group
    return best


def _grid_search(ohlcv_arrays, seed_prob, regime_dict, ticker, seed_id,
                 min_trades):
    idx = ohlcv_arrays['index']
    common = idx.intersection(seed_prob.index)
    if len(common) == 0:
        return []

    probs = np.zeros(len(idx), dtype=np.float64)
    common_locs = idx.get_indexer(common)
    probs[common_locs] = seed_prob.loc[common].values

    times  = ohlcv_arrays['times']
    months = ohlcv_arrays['months']
    t_min  = FIXED_ENTRY_MINUTE

    if TPSL_MODE == 'grace':
        configs = GRACE_CONFIGS
    elif TPSL_MODE == 'limit_entry':
        configs = LIMIT_ENTRY_CONFIGS
    else:
        configs = TPSL_PAIRS[TPSL_MODE]

    results = []
    for prob_thresh in PROB_THRESHOLDS:
        mask    = (probs >= prob_thresh) & (times == t_min)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        for pair in configs:
            if TPSL_MODE == 'limit_entry' and _existing_limit_keys:
                skip_key = (ticker, seed_id,
                            round(prob_thresh, 4),
                            round(pair.get('tp_pct', 0), 4),
                            round(pair.get('sl_pct', 0), 4),
                            round(pair.get('dip_pct', 0), 4),
                            int(pair.get('max_wait', 0)))
                if skip_key in _existing_limit_keys:
                    continue

            s_cfg = {'entry_time': FIXED_ENTRY_TIME, 'prob_thresh': prob_thresh, **pair}

            raw_pnl, raw_dates = _run_sim(indices, ohlcv_arrays, s_cfg)
            if len(raw_pnl) == 0:
                continue

            n = len(raw_pnl)
            if n < max(min_trades, MIN_TRADES_ABSOLUTE):
                continue
            freq = n / months
            if freq < MIN_TRADES_PER_MONTH:
                continue

            trade_vix, trade_tlt, trade_spy = _attach_regime(raw_dates, regime_dict, n, t_min)
            pnl = _apply_filter(raw_pnl, trade_vix, trade_tlt, trade_spy,
                                NO_FILTER_DEFAULTS['vix_threshold'],
                                NO_FILTER_DEFAULTS['tlt_crash'],
                                NO_FILTER_DEFAULTS['spy_ignore'])

            avg = float(np.mean(pnl))
            result_row = {
                'Ticker': ticker, 'Seed': seed_id,
                **s_cfg,
                'vix_threshold': NO_FILTER_DEFAULTS['vix_threshold'],
                'tlt_crash': NO_FILTER_DEFAULTS['tlt_crash'],
                'spy_ignore': NO_FILTER_DEFAULTS['spy_ignore'],
                'Avg_PnL': avg, 'Total_PnL': float(np.sum(pnl)),
                'Win_Rate': float(np.mean(pnl > 0)),
                'Freq': freq, 'Count': n,
                '_raw_pnl': pnl.copy(),
            }
            results.append(result_row)
    return results


def _eval_combo(combo, ohlcv_arrays, seed_prob, regime_dict):
    idx = ohlcv_arrays['index']
    common = idx.intersection(seed_prob.index)
    if len(common) == 0:
        return {'Avg_PnL': 0, 'Count': 0, 'Win_Rate': 0}

    probs = np.zeros(len(idx), dtype=np.float64)
    common_locs = idx.get_indexer(common)
    probs[common_locs] = seed_prob.loc[common].values

    h, m_ = map(int, combo['entry_time'].split(':'))
    t_min = h * 60 + m_
    mask = (probs >= combo['prob_thresh']) & (ohlcv_arrays['times'] == t_min)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return {'Avg_PnL': 0, 'Count': 0, 'Win_Rate': 0}

    raw_pnl, raw_dates = _run_sim(indices, ohlcv_arrays, combo)
    if len(raw_pnl) == 0:
        return {'Avg_PnL': 0, 'Count': 0, 'Win_Rate': 0}

    n = len(raw_pnl)
    trade_vix, trade_tlt, trade_spy = _attach_regime(raw_dates, regime_dict, n, t_min)
    pnl = _apply_filter(raw_pnl, trade_vix, trade_tlt, trade_spy,
                         combo['vix_threshold'], combo['tlt_crash'], combo['spy_ignore'])
    if len(pnl) == 0:
        return {'Avg_PnL': 0, 'Count': 0, 'Win_Rate': 0}
    return {
        'Avg_PnL': float(np.mean(pnl)),
        'Win_Rate': float(np.mean(pnl > 0)),
        'Count': len(pnl),
        'Total_PnL': float(np.sum(pnl)),
    }


# ==============================================================================
# ROLLING WALK-FORWARD OPTIMIZATION (per seed)
# ==============================================================================
def optimize_rolling_wf(ticker, df_ohlcv, vol_baselines,
                        regime_dict, seed_prob, seed_id,
                        pre_range_baselines=None, day_range_vals=None):

    def _precompute(start, end):
        return precompute_ohlcv_arrays(df_ohlcv, start, end, vol_baselines)

    r1 = WF_ROUNDS[0]
    ohlcv_opt = _precompute(*r1['opt'])
    ohlcv_val = _precompute(*r1['val'])
    if ohlcv_opt is None or ohlcv_val is None:
        return None

    r1_opt = _grid_search(ohlcv_opt, seed_prob, regime_dict, ticker, seed_id,
                           min_trades=r1['min_trades'])

    survivors = []
    for combo in r1_opt:
        if combo['Avg_PnL'] <= 0:
            continue
        val = _eval_combo(combo, ohlcv_val, seed_prob, regime_dict)
        if val['Avg_PnL'] > 0.005 and val['Count'] >= r1['min_trades']:
            survivors.append(combo)

    if TPSL_MODE == 'grace':
        n_configs = len(GRACE_CONFIGS)
    elif TPSL_MODE == 'limit_entry':
        n_configs = len(LIMIT_ENTRY_CONFIGS)
    else:
        n_configs = len(TPSL_PAIRS[TPSL_MODE])
    k_tested = len(PROB_THRESHOLDS) * n_configs
    print(f"        R1: {len(r1_opt)} tested (K={k_tested}) → {len(survivors)} survived  "
          f"[N_opt={r1_opt[0]['Count'] if r1_opt else 0}, K/N={k_tested/max(1, r1_opt[0]['Count'] if r1_opt else 1):.1f}]")

    if not survivors:
        return None

    best = _select_robust_params(survivors)
    if best is None:
        best = max(survivors, key=lambda r: r['Avg_PnL'])

    best_score = best['Avg_PnL']
    t_min_b = FIXED_ENTRY_MINUTE

    idx = ohlcv_opt['index']
    common = idx.intersection(seed_prob.index)
    probs = np.zeros(len(idx), dtype=np.float64)
    if len(common) > 0:
        common_locs = idx.get_indexer(common)
        probs[common_locs] = seed_prob.loc[common].values

    for vpct in VOL_PCT_GRID:
        if vpct == 0:
            continue
        vol_ok = _compute_vol_pct_ok(ohlcv_opt['volumes'], ohlcv_opt['times'],
                                      ohlcv_opt['dates'],
                                      ohlcv_opt.get('vol_baselines', {}), t_min_b, vpct)
        mask = (probs >= best['prob_thresh']) & (ohlcv_opt['times'] == t_min_b) & vol_ok
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        raw_pnl, raw_dates = _run_sim(indices, ohlcv_opt, best)
        if len(raw_pnl) == 0:
            continue
        n = len(raw_pnl)
        tv, tt, ts = _attach_regime(raw_dates, regime_dict, n, t_min_b)
        pnl = _apply_filter(raw_pnl, tv, tt, ts,
                             best['vix_threshold'], best['tlt_crash'], best['spy_ignore'])
        if len(pnl) < r1['min_trades']:
            continue
        avg = float(np.mean(pnl))
        if avg > best_score:
            best_score = avg
            best = {**best, 'vol_filter_pct': vpct,
                    'Avg_PnL': avg, 'Total_PnL': float(np.sum(pnl)),
                    'Win_Rate': float(np.mean(pnl > 0)),
                    'Freq': len(pnl) / ohlcv_opt['months'], 'Count': len(pnl)}

    best.setdefault('vol_filter_pct', 0)

    if pre_range_baselines and day_range_vals:
        for prpct in PRE_RANGE_FILTER_GRID:
            if prpct == 0:
                continue
            pre_ok = _compute_pre_range_ok(ohlcv_opt['dates'], pre_range_baselines,
                                            day_range_vals, prpct)
            vpct = best.get('vol_filter_pct', 0)
            vol_ok = np.ones(len(probs), dtype=np.bool_)
            if vpct > 0:
                vol_ok = _compute_vol_pct_ok(ohlcv_opt['volumes'], ohlcv_opt['times'],
                                              ohlcv_opt['dates'],
                                              ohlcv_opt.get('vol_baselines', {}), t_min_b, vpct)
            mask = (probs >= best['prob_thresh']) & (ohlcv_opt['times'] == t_min_b) & vol_ok & pre_ok
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            raw_pnl, raw_dates = _run_sim(indices, ohlcv_opt, best)
            if len(raw_pnl) == 0:
                continue
            n = len(raw_pnl)
            tv, tt, ts = _attach_regime(raw_dates, regime_dict, n, t_min_b)
            pnl = _apply_filter(raw_pnl, tv, tt, ts,
                                 best['vix_threshold'], best['tlt_crash'], best['spy_ignore'])
            if len(pnl) < r1['min_trades']:
                continue
            avg = float(np.mean(pnl))
            if avg > best_score:
                best_score = avg
                best = {**best, 'pre_range_filter_pct': prpct,
                        'Avg_PnL': avg, 'Total_PnL': float(np.sum(pnl)),
                        'Win_Rate': float(np.mean(pnl > 0)),
                        'Freq': len(pnl) / ohlcv_opt['months'], 'Count': len(pnl)}

    best.setdefault('pre_range_filter_pct', 0)
    _fill_defaults(best)

    val_result = validate_params(ticker, ohlcv_val, regime_dict, seed_prob, best,
                                  pre_range_baselines, day_range_vals)
    del ohlcv_opt, ohlcv_val
    if val_result is None:
        val_result = {'Avg_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}

    return best, val_result


def _fill_defaults(best):
    best.setdefault('tp_mult', 0.0)
    best.setdefault('sl_mult', 0.0)
    best.setdefault('trail_start_mult', 0.0)
    best.setdefault('trail_end_mult', 0.0)
    best.setdefault('grace_minutes', 0)
    best.setdefault('dip_pct', 0.0)
    best.setdefault('max_wait', 0)
    best.setdefault('pre_range_filter_pct', 0)
    best.pop('_raw_pnl', None)


# ==============================================================================
# VALIDATION
# ==============================================================================
def validate_params(ticker, ohlcv_val, regime_dict, seed_prob, params,
                    pre_range_baselines=None, day_range_vals=None):
    if ohlcv_val is None:
        return None

    idx = ohlcv_val['index']
    common = idx.intersection(seed_prob.index)
    if len(common) == 0:
        return None

    probs = np.zeros(len(idx), dtype=np.float64)
    common_locs = idx.get_indexer(common)
    probs[common_locs] = seed_prob.loc[common].values

    h, m_ = map(int, params['entry_time'].split(':'))
    t_min = h * 60 + m_

    mask = (probs >= params['prob_thresh']) & (ohlcv_val['times'] == t_min)
    vpct = params.get('vol_filter_pct', 0)
    if vpct > 0:
        vol_ok = _compute_vol_pct_ok(ohlcv_val['volumes'], ohlcv_val['times'],
                                      ohlcv_val['dates'], ohlcv_val.get('vol_baselines', {}),
                                      t_min, vpct)
        mask = mask & vol_ok
    prpct = params.get('pre_range_filter_pct', 0)
    if prpct > 0 and pre_range_baselines and day_range_vals:
        pre_ok = _compute_pre_range_ok(ohlcv_val['dates'], pre_range_baselines,
                                        day_range_vals, prpct)
        mask = mask & pre_ok
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return {'Avg_PnL': 0, 'Total_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}

    raw_pnl, raw_dates = _run_sim(indices, ohlcv_val, params)
    if len(raw_pnl) == 0:
        return {'Avg_PnL': 0, 'Total_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}

    n = len(raw_pnl)
    trade_vix, trade_tlt, trade_spy = _attach_regime(raw_dates, regime_dict, n, t_min)
    pnl = _apply_filter(raw_pnl, trade_vix, trade_tlt, trade_spy,
                         params['vix_threshold'], params['tlt_crash'], params['spy_ignore'])

    months = ohlcv_val['months']
    return {
        'Avg_PnL':   float(np.mean(pnl)) if len(pnl) > 0 else 0,
        'Total_PnL': float(np.sum(pnl)) if len(pnl) > 0 else 0,
        'Win_Rate':  float(np.mean(pnl > 0)) if len(pnl) > 0 else 0,
        'Freq':      len(pnl) / months if len(pnl) > 0 else 0,
        'Count':     len(pnl),
    }


# ==============================================================================
# MAIN — loops through RUN_MODES in a single execution
# ==============================================================================
if auth is not None:
    try: auth.authenticate_user()
    except Exception: pass

MODE_SUFFIX = {'static': '', 'grace': '_GRACE', 'limit_entry': '_LIMIT'}


# ==============================================================================
# MAIN — all folds 180-185, ticker-first for OHLCV reuse
# ==============================================================================
active_fold_configs = [fc for fc in FOLD_CONFIGS if fc['fold'] in ACTIVE_FOLDS]

print(f"\n{'='*110}")
print(f"  WALK-FORWARD OPTIMIZER v12 — LONG/SPIKE MODEL — ALL FOLDS 170-175 (Standard Transformer)")
print(f"  FOLDS: {', '.join(str(fc['fold']) for fc in active_fold_configs)}")
print(f"  MODES: {', '.join(m.upper() for m in RUN_MODES)}")
print(f"  TRADE DIRECTION: LONG (buy at signal, profit on price RISE)")
print(f"  ENTRY: FIXED at {FIXED_ENTRY_TIME} ({FIXED_ENTRY_MINUTE} min)")
print(f"  SLIPPAGE: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} round-trip)")
print(f"  PROB THRESHOLDS: {len(PROB_THRESHOLDS)} values ({PROB_THRESHOLDS[0]:.2f} — {PROB_THRESHOLDS[-1]:.2f})")
print(f"  GRID: REDUCED (anti-overfit) — 126 grid points per seed per ticker")
for rnd in WF_ROUNDS:
    print(f"  {rnd['name']}: OPT {rnd['opt'][0]}->{rnd['opt'][1]}  VAL {rnd['val'][0]}->{rnd['val'][1]}  min_trades={rnd['min_trades']}")
print(f"  TEST: {TEST_START} -> {TEST_END} (pure holdout)")
for _m in RUN_MODES:
    if _m == 'grace':
        _nc = len(GRACE_CONFIGS)
    elif _m == 'limit_entry':
        _nc = len(LIMIT_ENTRY_CONFIGS)
    else:
        _nc = len(TPSL_PAIRS[_m])
    _k = len(PROB_THRESHOLDS) * _nc
    _label = 'configs' if _m in ('grace', 'limit_entry') else 'pairs'
    print(f"    {_m.upper():>18}: K={_k} ({len(PROB_THRESHOLDS)} prob × {_nc} {_label}) → VAL → TEST")
print(f"  MIN_TRADES_ABSOLUTE = {MIN_TRADES_ABSOLUTE}")
print(f"{'='*110}\n")

tickers = get_ticker_list()
if TICKER_SUBSET:
    tickers = [t for t in tickers if t in TICKER_SUBSET]
    print(f"  Filtered to TICKER_SUBSET: {len(tickers)} tickers")
print(f"  {len(tickers)} tickers × {len(active_fold_configs)} folds = {len(tickers) * len(active_fold_configs)} ticker-fold combos")

# ── BQ WIPE & SKIP per fold ──────────────────────────────────────────────────
completed_seeds_by_fold = {}
existing_limit_keys_by_fold = {}

for fc in active_fold_configs:
    mv = fc['model_version']
    exp = fc['experiment']
    exp_like = f"experiment LIKE '{exp}%'"

    if WIPE_EXISTING:
        try:
            del_q = (f"DELETE FROM `{BQ_TABLE_ID}` "
                     f"WHERE model_version = '{mv}' "
                     f"AND ({exp_like})")
            _bq_client.query(del_q).result()
            print(f"  WIPE fold {fc['fold']}: deleted model_version='{mv}'")
        except Exception as e:
            if 'streaming buffer' in str(e).lower():
                copy_q = (f"CREATE OR REPLACE TABLE `{BQ_TABLE_ID}` AS "
                          f"SELECT * FROM `{BQ_TABLE_ID}` "
                          f"WHERE NOT (model_version = '{mv}' "
                          f"AND ({exp_like}))")
                _bq_client.query(copy_q).result()
                print(f"  WIPE fold {fc['fold']}: rebuilt without '{mv}'")
            else:
                raise

    completed = {}
    if SKIP_COMPLETED:
        skip_q = (f"SELECT experiment, ticker, tpsl_mode, MAX(n_seeds_avg) as max_seeds "
                  f"FROM `{BQ_TABLE_ID}` "
                  f"WHERE model_version = '{mv}' "
                  f"AND ({exp_like}) "
                  f"GROUP BY experiment, ticker, tpsl_mode")
        for row in _bq_client.query(skip_q):
            completed[(row.experiment, row.ticker, row.tpsl_mode or 'static')] = row.max_seeds
    completed_seeds_by_fold[fc['fold']] = completed

    limit_keys = set()
    if 'limit_entry' in RUN_MODES:
        completed_seeds_by_fold[fc['fold']] = {k: v for k, v in completed.items() if k[2] != 'limit_entry'}
        limit_q = (f"SELECT ticker, seed, prob_thresh, tp_pct, sl_pct, dip_pct, max_wait_minutes "
                   f"FROM `{BQ_TABLE_ID}` "
                   f"WHERE model_version = '{mv}' "
                   f"AND tpsl_mode = 'limit_entry' "
                   f"AND ({exp_like})")
        for row in _bq_client.query(limit_q):
            limit_keys.add((
                row.ticker, int(row.seed),
                round(float(row.prob_thresh), 4),
                round(float(row.tp_pct), 4),
                round(float(row.sl_pct), 4),
                round(float(row.dip_pct or 0), 4),
                int(row.max_wait_minutes or 0),
            ))
    existing_limit_keys_by_fold[fc['fold']] = limit_keys

if SKIP_COMPLETED:
    total_skip = sum(len(v) for v in completed_seeds_by_fold.values())
    if total_skip > 0:
        print(f"  SKIP_COMPLETED: {total_skip} total triples across all folds")

regime_dict, vix_lk = preload_regime_data(WF_ROUNDS[0]['opt'][0], TEST_END)

RUN_ID = str(uuid.uuid4())
print(f"  BQ run_id: {RUN_ID[:8]}...")

# ── Results tracking per fold ────────────────────────────────────────────────
all_results = {}
ensemble_results = {}
for fc in active_fold_configs:
    for m in RUN_MODES:
        all_results[(fc['fold'], m)] = []
        ensemble_results[(fc['fold'], m)] = []

n_tickers = len(tickers)

# ── MAIN LOOP: ticker-first, fold-second ─────────────────────────────────────
for t_idx, ticker in enumerate(tickers):

    print(f"\n{'='*70}")
    print(f"  [{t_idx+1}/{n_tickers}] {ticker} (LONG)")
    print(f"{'='*70}")

    try:
        df_ohlcv = load_ticker_ohlcv(ticker)
    except Exception as e:
        print(f"  Could not load OHLCV: {e}")
        continue

    vol_baselines = precompute_vol_baselines(df_ohlcv)
    pre_range_baselines = precompute_pre_entry_range_baselines(df_ohlcv)
    day_range_vals = {}
    _pr_times = (df_ohlcv.index.hour * 60 + df_ohlcv.index.minute).values
    _pr_dates = (df_ohlcv.index.year * 10000 + df_ohlcv.index.month * 100
                 + df_ohlcv.index.day).values.astype(np.int64)
    _pr_highs = df_ohlcv['High'].values
    _pr_lows  = df_ohlcv['Low'].values
    _pr_closes = df_ohlcv['Close'].values
    for _ud in np.unique(_pr_dates):
        _dm = _pr_dates == _ud
        _dt = _pr_times[_dm]
        _pre = (_dt >= 390) & (_dt <= FIXED_ENTRY_MINUTE)
        if _pre.sum() < 2:
            continue
        _ph = _pr_highs[_dm][_pre].max()
        _pl = _pr_lows[_dm][_pre].min()
        _op = _pr_closes[_dm][_dt == 390]
        if len(_op) == 0 or _op[0] <= 0:
            continue
        day_range_vals[int(_ud)] = (_ph - _pl) / _op[0]

    # ── Fold loop ────────────────────────────────────────────────────────
    for fc in active_fold_configs:
        fold_num = fc['fold']
        base_experiment = fc['experiment']
        MODEL_VERSION_TAG = fc['model_version']
        completed_seeds = completed_seeds_by_fold[fold_num]
        _existing_limit_keys = existing_limit_keys_by_fold[fold_num]

        print(f"\n  ┌── Fold {fold_num} ({fc['label']}) ──────────────────")

        seed_scores = load_seed_scores(ticker, base_experiment)
        if not seed_scores:
            print(f"  │  No seed scores for {base_experiment}")
            continue
        print(f"  │  {len(seed_scores)} seeds loaded")

        for mode in RUN_MODES:
            TPSL_MODE = mode
            mode_sfx = MODE_SUFFIX.get(mode, '')
            experiment = f"{base_experiment}{mode_sfx}"

            prev_seeds = completed_seeds.get((experiment, ticker, mode))
            if prev_seeds is not None and prev_seeds >= len(seed_scores):
                print(f"  │  [{mode.upper()}] Skipping — already in BQ")
                continue

            print(f"  │  [{mode.upper()}] {ticker} — {experiment}")

            ohlcv_test = precompute_ohlcv_arrays(df_ohlcv, TEST_START, TEST_END,
                                                  vol_baselines)

            seed_opt_results = {}

            for seed_id, seed_prob in sorted(seed_scores.items()):
                print(f"  │    Seed {seed_id}:")

                wf_result = optimize_rolling_wf(
                    ticker, df_ohlcv, vol_baselines,
                    regime_dict, seed_prob, seed_id,
                    pre_range_baselines=pre_range_baselines,
                    day_range_vals=day_range_vals)

                if wf_result is None:
                    print(f"  │      No valid optimization result")
                    continue

                opt_result, val_a_result = wf_result
                seed_opt_results[seed_id] = opt_result

                if mode == 'grace':
                    tp_sl_str = (f"TP={opt_result['tp_pct']:.0%}  SL={opt_result['sl_pct']:.0%}  "
                                 f"grace={opt_result.get('grace_minutes',0)}m")
                elif mode == 'limit_entry':
                    tp_sl_str = (f"TP={opt_result['tp_pct']:.0%}  SL={opt_result['sl_pct']:.0%}  "
                                 f"dip={opt_result.get('dip_pct',0):.1%}  "
                                 f"wait={opt_result.get('max_wait',0)}m")
                else:
                    tp_sl_str = f"TP={opt_result['tp_pct']:.0%}  SL={opt_result['sl_pct']:.0%}"

                print(f"  │      OPT:  AvgPnL={opt_result['Avg_PnL']:.2%}  WR={opt_result['Win_Rate']:.1%}  "
                      f"N={opt_result['Count']}  Prob>{opt_result['prob_thresh']:.3f}  {tp_sl_str}")
                tag_va = "PASS" if val_a_result['Avg_PnL'] > 0 else "FAIL"
                print(f"  │      VAL:  AvgPnL={val_a_result['Avg_PnL']:.2%}  WR={val_a_result['Win_Rate']:.1%}  "
                      f"N={val_a_result['Count']}  [{tag_va}]")

                test = validate_params(ticker, ohlcv_test, regime_dict, seed_prob, opt_result,
                                       pre_range_baselines, day_range_vals)
                if test is None:
                    test = {'Avg_PnL': 0, 'Total_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}
                tag_t = "PASS" if test['Avg_PnL'] > 0 else "FAIL"
                print(f"  │      TEST: AvgPnL={test['Avg_PnL']:.2%}  WR={test['Win_Rate']:.1%}  "
                      f"N={test['Count']}  [{tag_t}]")

                result_row = {
                    'Ticker': ticker, 'Seed': seed_id, 'Fold': fold_num,
                    'entry_time': opt_result['entry_time'],
                    'prob_thresh': opt_result['prob_thresh'],
                    'tp_pct': opt_result.get('tp_pct', 0), 'sl_pct': opt_result.get('sl_pct', 0),
                    'tp_mult': opt_result.get('tp_mult', 0), 'sl_mult': opt_result.get('sl_mult', 0),
                    'trail_start_mult': opt_result.get('trail_start_mult', 0),
                    'trail_end_mult': opt_result.get('trail_end_mult', 0),
                    'regime_tp_pct': opt_result.get('regime_tp_pct', 0),
                    'regime_sl_pct': opt_result.get('regime_sl_pct', 0),
                    'regime_lookback': opt_result.get('regime_lookback', 0),
                    'regime_spike_thresh': opt_result.get('regime_spike_thresh', 0),
                    'vix_threshold': opt_result['vix_threshold'],
                    'tlt_crash': opt_result['tlt_crash'],
                    'spy_ignore': opt_result['spy_ignore'],
                    'vol_filter_pct': opt_result.get('vol_filter_pct', 0),
                    'pre_range_filter_pct': opt_result.get('pre_range_filter_pct', 0),
                    'grace_minutes': opt_result.get('grace_minutes', 0),
                    'dip_pct': opt_result.get('dip_pct', 0),
                    'max_wait': opt_result.get('max_wait', 0),
                    'Avg_PnL': opt_result['Avg_PnL'],
                    'Win_Rate': opt_result['Win_Rate'],
                    'Freq': opt_result.get('Freq', 0),
                    'Count': opt_result['Count'],
                    'val_a_avg_pnl': val_a_result['Avg_PnL'],
                    'val_a_win_rate': val_a_result['Win_Rate'],
                    'val_a_freq': val_a_result.get('Freq', 0),
                    'val_a_count': val_a_result['Count'],
                    'survived_a': val_a_result['Avg_PnL'] > 0,
                    'val_b_avg_pnl': test['Avg_PnL'],
                    'val_b_win_rate': test['Win_Rate'],
                    'val_b_freq': test['Freq'],
                    'val_b_count': test['Count'],
                    'survived_b': test['Avg_PnL'] > 0,
                }
                all_results[(fold_num, mode)].append(result_row)
                save_row_to_bq(result_row, RUN_ID, experiment,
                                WF_ROUNDS[0]['opt'][0], WF_ROUNDS[-1]['val'][1],
                                WF_ROUNDS[-1]['val'][0], WF_ROUNDS[-1]['val'][1],
                                TEST_START, TEST_END)

            # ── ENSEMBLE (with val_a fix) ───────────────────────────────
            if seed_opt_results:
                all_prob_series = [seed_scores[s] for s in sorted(seed_scores.keys())]
                ensemble_prob = pd.concat(all_prob_series, axis=1).mean(axis=1)

                best_seed_id = max(seed_opt_results, key=lambda s: seed_opt_results[s]['Avg_PnL'])
                ens_params = dict(seed_opt_results[best_seed_id])

                ens_r = WF_ROUNDS[0]
                ohlcv_ens_opt = precompute_ohlcv_arrays(df_ohlcv, ens_r['opt'][0], ens_r['opt'][1],
                                                         vol_baselines)
                best_pt, best_pt_pnl = ens_params['prob_thresh'], -999.0
                for pt in PROB_THRESHOLDS:
                    trial = {**ens_params, 'prob_thresh': pt}
                    ens_opt_result = validate_params(ticker, ohlcv_ens_opt, regime_dict, ensemble_prob, trial,
                                                    pre_range_baselines, day_range_vals)
                    if ens_opt_result is None or ens_opt_result['Count'] < ens_r['min_trades']:
                        continue
                    if ens_opt_result['Avg_PnL'] > best_pt_pnl:
                        best_pt_pnl = ens_opt_result['Avg_PnL']
                        best_pt = pt
                del ohlcv_ens_opt
                ens_params['prob_thresh'] = best_pt

                ohlcv_ens_val = precompute_ohlcv_arrays(df_ohlcv, ens_r['val'][0], ens_r['val'][1],
                                                         vol_baselines)
                ens_val = validate_params(ticker, ohlcv_ens_val, regime_dict, ensemble_prob, ens_params,
                                          pre_range_baselines, day_range_vals)
                del ohlcv_ens_val
                if ens_val is None:
                    ens_val = {'Avg_PnL': 0, 'Total_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}

                ens_test = validate_params(ticker, ohlcv_test, regime_dict, ensemble_prob, ens_params,
                                           pre_range_baselines, day_range_vals)
                if ens_test is None:
                    ens_test = {'Avg_PnL': 0, 'Total_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}

                ens_tag = "PASS" if ens_test['Avg_PnL'] > 0 else "FAIL"
                print(f"  │    ENS({len(seed_scores)}seeds): VAL={ens_val['Avg_PnL']:.2%}  "
                      f"TEST={ens_test['Avg_PnL']:.2%} [{ens_tag}]")

                ens_row = {
                    'Ticker': ticker, 'Fold': fold_num,
                    'param_seed': best_seed_id,
                    'n_seeds_avg': len(seed_scores),
                    'n_seeds_survived': len(seed_opt_results),
                    'test_avg_pnl': ens_test['Avg_PnL'],
                    'test_win_rate': ens_test['Win_Rate'],
                    'test_count': ens_test['Count'],
                    'test_pass': ens_test['Avg_PnL'] > 0,
                }
                ensemble_results[(fold_num, mode)].append(ens_row)

                ens_bq_row = {
                    'Ticker': ticker, 'Seed': -1,
                    'entry_time': ens_params['entry_time'],
                    'prob_thresh': ens_params['prob_thresh'],
                    'tp_pct': ens_params.get('tp_pct', 0),
                    'sl_pct': ens_params.get('sl_pct', 0),
                    'tp_mult': ens_params.get('tp_mult', 0),
                    'sl_mult': ens_params.get('sl_mult', 0),
                    'trail_start_mult': ens_params.get('trail_start_mult', 0),
                    'trail_end_mult': ens_params.get('trail_end_mult', 0),
                    'regime_tp_pct': ens_params.get('regime_tp_pct', 0),
                    'regime_sl_pct': ens_params.get('regime_sl_pct', 0),
                    'regime_lookback': ens_params.get('regime_lookback', 0),
                    'regime_spike_thresh': ens_params.get('regime_spike_thresh', 0),
                    'vix_threshold': ens_params.get('vix_threshold', 999.0),
                    'tlt_crash': ens_params.get('tlt_crash', -999.0),
                    'spy_ignore': ens_params.get('spy_ignore', 999.0),
                    'vol_filter_pct': ens_params.get('vol_filter_pct', 0),
                    'pre_range_filter_pct': ens_params.get('pre_range_filter_pct', 0),
                    'grace_minutes': ens_params.get('grace_minutes', 0),
                    'dip_pct': ens_params.get('dip_pct', 0),
                    'max_wait': ens_params.get('max_wait', 0),
                    'Avg_PnL': ens_params['Avg_PnL'],
                    'Win_Rate': ens_params['Win_Rate'],
                    'Freq': ens_params.get('Freq', 0),
                    'Count': ens_params['Count'],
                    'val_a_avg_pnl': ens_val['Avg_PnL'],
                    'val_a_win_rate': ens_val['Win_Rate'],
                    'val_a_freq': ens_val.get('Freq', 0),
                    'val_a_count': ens_val['Count'],
                    'survived_a': ens_val['Avg_PnL'] > 0,
                    'val_b_avg_pnl': ens_test['Avg_PnL'],
                    'val_b_win_rate': ens_test['Win_Rate'],
                    'val_b_freq': ens_test.get('Freq', 0),
                    'val_b_count': ens_test['Count'],
                    'survived_b': ens_test['Avg_PnL'] > 0,
                    'is_ensemble': True,
                    'n_seeds_avg': len(seed_scores),
                }
                save_row_to_bq(ens_bq_row, RUN_ID, experiment,
                                WF_ROUNDS[0]['opt'][0], WF_ROUNDS[-1]['val'][1],
                                WF_ROUNDS[-1]['val'][0], WF_ROUNDS[-1]['val'][1],
                                TEST_START, TEST_END)

            del ohlcv_test; gc.collect()

        print(f"  └── Fold loop complete for {ticker}")

    del df_ohlcv, seed_scores, vol_baselines
    del pre_range_baselines, day_range_vals; gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ==============================================================================
# SUMMARY — per fold + cross-fold comparison
# ==============================================================================
W = 145

for fc in active_fold_configs:
    fold_num = fc['fold']
    fold_total = sum(len(all_results[(fold_num, m)]) for m in RUN_MODES)
    if fold_total == 0:
        print(f"\n  Fold {fold_num}: No results")
        continue

    print(f"\n\n{'='*W}")
    print(f"  FOLD {fold_num} — {fc['label']} — model_version: {fc['model_version']}")
    print(f"{'='*W}")

    for mode in RUN_MODES:
        mode_results = all_results[(fold_num, mode)]
        if not mode_results:
            continue

        df = pd.DataFrame(mode_results)
        best_per_ticker = df.loc[df.groupby('Ticker')['Avg_PnL'].idxmax()]
        passed = best_per_ticker[best_per_ticker['survived_b']]

        print(f"\n  {mode.upper()}: {len(best_per_ticker)} tickers, "
              f"{len(passed)} PASSED TEST")
        if len(passed) > 0:
            print(f"    PASS avg: OPT={passed['Avg_PnL'].mean():.2%} → "
                  f"TEST={passed['val_b_avg_pnl'].mean():.2%}")

# ── Cross-fold comparison ────────────────────────────────────────────────────
print(f"\n\n{'='*W}")
print(f"  CROSS-FOLD COMPARISON — FOLD AGREEMENT PER TICKER")
print(f"  Tickers with VAL > 0 in >= 4/6 folds are most robust")
print(f"{'='*W}")

for mode in RUN_MODES:
    mode_data = []
    for fc in active_fold_configs:
        for r in all_results[(fc['fold'], mode)]:
            mode_data.append(r)
    if not mode_data:
        continue

    df = pd.DataFrame(mode_data)
    if df.empty:
        continue

    ticker_fold = df.groupby(['Ticker', 'Fold']).agg(
        best_val_pnl=('val_a_avg_pnl', 'max'),
        best_test_pnl=('val_b_avg_pnl', 'max'),
        best_opt_pnl=('Avg_PnL', 'max'),
    ).reset_index()

    agreement = ticker_fold.groupby('Ticker').agg(
        n_folds=('Fold', 'count'),
        folds_val_pos=('best_val_pnl', lambda x: (x > 0).sum()),
        folds_test_pos=('best_test_pnl', lambda x: (x > 0).sum()),
        avg_val_pnl=('best_val_pnl', 'mean'),
        avg_test_pnl=('best_test_pnl', 'mean'),
    ).sort_values('folds_val_pos', ascending=False)

    print(f"\n  {mode.upper()}:")
    print(f"  {'Ticker':<7} {'Folds':>6} {'VAL+':>6} {'TEST+':>6} {'Avg VAL':>9} {'Avg TEST':>9} {'Robust':>7}")
    print(f"  {'─'*55}")
    for ticker, r in agreement.iterrows():
        robust = "YES" if r['folds_val_pos'] >= 4 else "no"
        print(f"  {ticker:<7} {int(r['n_folds']):>6} "
              f"{int(r['folds_val_pos']):>5}/6 "
              f"{int(r['folds_test_pos']):>5}/6 "
              f"{r['avg_val_pnl']:>+8.2%} "
              f"{r['avg_test_pnl']:>+8.2%} "
              f"{'  ' + robust:>7}")

    robust_count = (agreement['folds_val_pos'] >= 4).sum()
    print(f"  {'─'*55}")
    print(f"  {robust_count}/{len(agreement)} tickers pass fold-agreement gate (>=4/6)")

# ── Grand total ──────────────────────────────────────────────────────────────
grand_total = sum(len(v) for v in all_results.values())
ens_total = sum(len(v) for v in ensemble_results.values())

print(f"\n\n{'='*W}")
print(f"  ALL FOLDS COMPLETE — {grand_total} per-seed results, {ens_total} ensemble results")
for fc in active_fold_configs:
    fold_n = sum(len(all_results[(fc['fold'], m)]) for m in RUN_MODES)
    fold_e = sum(len(ensemble_results[(fc['fold'], m)]) for m in RUN_MODES)
    print(f"    Fold {fc['fold']}: {fold_n} per-seed, {fold_e} ensemble — {fc['model_version']}")
print(f"{'='*W}")
