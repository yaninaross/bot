"""
optimize_wf_dynamic.py — DEPRECATED (2026-05-24)
======================================================================
SUPERSEDED BY: optimize_wf_robust.py

Do NOT use this optimizer for new work. Known issues:
  - No slippage applied (0.30% round-trip missing vs robust optimizer)
  - Massive combinatorial grid (~2.7B combos) without multiple testing correction
  - val_a metrics were OPT, not actual validation (fixed in robust)

Use optimize_wf_robust.py instead, which handles all 5 TPSL modes with
proper slippage, R1→R2 expanding-window validation, and correct BQ metrics.

Original description:
  Walk-Forward Optimization with Dynamic TP/SL — 5 modes (static, atr,
  atr_adaptive, regime_scaled, trailing_decay). Results saved to BQ.
"""

import gc
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
# ALL MODES — script loops through every mode in a single run
# ==============================================================================
ALL_MODES = ['static', 'atr', 'atr_adaptive', 'regime_scaled', 'trailing_decay']
TPSL_MODE = ALL_MODES[0]  # updated by main loop; functions read this global

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
            'tp_mult': float(row_dict.get('tp_mult', 0)) if TPSL_MODE in ('atr', 'atr_adaptive') else None,
            'sl_mult': float(row_dict.get('sl_mult', 0)) if TPSL_MODE in ('atr', 'atr_adaptive', 'trailing_decay') else None,
            'trail_start_mult': float(row_dict.get('trail_start_mult', 0)) if TPSL_MODE == 'trailing_decay' else None,
            'trail_end_mult': float(row_dict.get('trail_end_mult', 0)) if TPSL_MODE == 'trailing_decay' else None,
            'model_version': 'dynamic_wf_v2',
            'v0_fixed': True,
        }
        errors = _bq_client.insert_rows_json(BQ_TABLE_ID, [row])
        if errors:
            print(f"      BQ error: {errors[0]}")
        else:
            print(f"      BQ: saved {row_dict['Ticker']}/S{row_dict['Seed']}")
    except Exception as e:
        print(f"      BQ save failed: {e}")


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME     = "time-series-bucket-ys"
TICKER_FILE_MAP = {'BITF': 'KEEL'}
TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}
DEPLOY_PNL_THRESHOLD = 0.018
MIN_BQ_TRADES   = 3

CLEAN_EXPERIMENTS = {
    '105_DEFENSE', '106_DEFENSE', '152_XSECTOR',
    '160_XSECTOR',
    '170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
    '173_XSECTOR', '174_XSECTOR', '175_XSECTOR',
}

OPT_START   = '2025-10-02'
OPT_END     = '2025-12-31'
VAL_A_START = '2026-01-02'
VAL_A_END   = '2026-04-25'
VAL_B_START = '2025-11-01'
VAL_B_END   = '2026-04-25'

EXIT_TIME_MIN        = 780
MIN_TRADES_PER_MONTH = 2.0
VOL_LOOKBACK_DAYS    = 30
VOL_PCT_GRID         = [0, 10, 25, 35]

SKIP_COMPLETED = True
TICKER_SUBSET  = []  # empty = run all tickers from deploy_configs

# ==============================================================================
# GRIDS
# ==============================================================================
GRID_STRATEGY_STATIC = {
    'entry_time':  ["06:00", "06:05", "06:10", "06:15", "06:20", "06:25", "06:30"],
    'prob_thresh': np.arange(0.01, 0.99, 0.10).round(2).tolist(),
    'tp_pct':      np.arange(0.10, 0.41, 0.05).round(2).tolist(),
    'sl_pct':      np.arange(0.02, 0.16, 0.02).round(2).tolist(),
}

GRID_STRATEGY_ATR = {
    'entry_time':  ["06:00", "06:05", "06:10", "06:15", "06:20", "06:25", "06:30"],
    'prob_thresh': np.arange(0.01, 0.99, 0.10).round(2).tolist(),
    'tp_mult':     [1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0],
    'sl_mult':     [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
}

GRID_STRATEGY_REGIME = GRID_STRATEGY_STATIC.copy()

GRID_STRATEGY_TRAILING = {
    'entry_time':       ["06:00", "06:05", "06:10", "06:15", "06:20", "06:25", "06:30"],
    'prob_thresh':      np.arange(0.01, 0.99, 0.10).round(2).tolist(),
    'trail_start_mult': [2.0, 3.0, 5.0, 8.0],
    'trail_end_mult':   [0.5, 1.0, 1.5],
    'sl_mult':          [1.0, 1.5, 2.0, 3.0, 5.0],
}

GRID_FILTER = {
    'vix_threshold': [15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0],
    'tlt_crash':     [0.0, -0.10, -0.20, -0.30, -0.50],
    'spy_ignore':    [0.0, 0.10, 0.20, 0.30, 0.50],
}

REGIME_TP_SCALE = np.array([0.70, 0.85, 1.00, 1.30, 1.60], dtype=np.float64)
REGIME_SL_SCALE = np.array([0.60, 0.80, 1.00, 1.30, 1.80], dtype=np.float64)

VIX_REGIME_BOUNDS = np.array([15.0, 20.0, 25.0, 35.0], dtype=np.float64)


def get_grid_strategy():
    if TPSL_MODE in ('atr', 'atr_adaptive'):
        return GRID_STRATEGY_ATR
    elif TPSL_MODE == 'regime_scaled':
        return GRID_STRATEGY_REGIME
    elif TPSL_MODE == 'trailing_decay':
        return GRID_STRATEGY_TRAILING
    return GRID_STRATEGY_STATIC


# ==============================================================================
# SIMULATION ENGINES (Numba)
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
            if lows[k] <= stop_price:
                outcome = -sl; exit_idx = k; trade_done = True; break
            if highs[k] >= target_price:
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
def run_trade_sim_atr(entry_indices, prices, highs, lows, date_ints, time_mins,
                      atr_pct_arr, tp_mult, sl_mult, exit_time):
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

        entry_price = prices[idx]
        atr = atr_pct_arr[idx]
        if atr != atr or atr <= 0:
            atr = 0.05
        tp = min(max(atr * tp_mult, 0.02), 0.60)
        sl = min(max(atr * sl_mult, 0.01), 0.25)
        target_price = entry_price * (1 + tp)
        stop_price   = entry_price * (1 - sl)

        outcome    = 0.0
        trade_done = False
        exit_idx   = idx

        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
            if lows[k] <= stop_price:
                outcome = -sl; exit_idx = k; trade_done = True; break
            if highs[k] >= target_price:
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
def run_trade_sim_regime(entry_indices, prices, highs, lows, date_ints, time_mins,
                         vix_regime_arr, tp_base, sl_base,
                         tp_scales, sl_scales, exit_time):
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

        entry_price = prices[idx]
        r = vix_regime_arr[idx]
        if r < 0: r = 2
        if r > 4: r = 4
        tp = tp_base * tp_scales[r]
        sl = sl_base * sl_scales[r]
        target_price = entry_price * (1 + tp)
        stop_price   = entry_price * (1 - sl)

        outcome    = 0.0
        trade_done = False
        exit_idx   = idx

        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
            if lows[k] <= stop_price:
                outcome = -sl; exit_idx = k; trade_done = True; break
            if highs[k] >= target_price:
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
def run_trade_sim_trailing_decay(entry_indices, prices, highs, lows, date_ints, time_mins,
                                  atr_pct_arr, trail_start_mult, trail_end_mult,
                                  sl_mult, exit_time):
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

        entry_price = prices[idx]
        atr = atr_pct_arr[idx]
        if atr != atr or atr <= 0:
            atr = 0.05

        sl = min(max(atr * sl_mult, 0.01), 0.25)
        hard_stop = entry_price * (1 - sl)

        trail_start = atr * trail_start_mult
        trail_end   = atr * trail_end_mult
        decay_target = trail_start

        high_watermark = entry_price

        outcome    = 0.0
        trade_done = False
        exit_idx   = idx

        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break

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
                exit_idx = k; trade_done = True; break

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


def _run_sim(indices, ohlcv_arrays, s_cfg):
    prices = ohlcv_arrays['prices']
    highs  = ohlcv_arrays['highs']
    lows   = ohlcv_arrays['lows']
    dates  = ohlcv_arrays['dates']
    times  = ohlcv_arrays['times'].astype(np.int64)

    if TPSL_MODE in ('atr', 'atr_adaptive'):
        return run_trade_sim_atr(
            indices, prices, highs, lows, dates, times,
            ohlcv_arrays['atr_pct'], s_cfg['tp_mult'], s_cfg['sl_mult'], EXIT_TIME_MIN)
    elif TPSL_MODE == 'trailing_decay':
        return run_trade_sim_trailing_decay(
            indices, prices, highs, lows, dates, times,
            ohlcv_arrays['atr_pct'], s_cfg['trail_start_mult'], s_cfg['trail_end_mult'],
            s_cfg['sl_mult'], EXIT_TIME_MIN)
    elif TPSL_MODE == 'regime_scaled':
        return run_trade_sim_regime(
            indices, prices, highs, lows, dates, times,
            ohlcv_arrays['vix_regime'], s_cfg['tp_pct'], s_cfg['sl_pct'],
            REGIME_TP_SCALE, REGIME_SL_SCALE, EXIT_TIME_MIN)
    else:
        return run_trade_sim(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN)


# ==============================================================================
# ATR COMPUTATION
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


def classify_vix_regime(vix_val):
    if vix_val <= 15: return 0
    if vix_val <= 20: return 1
    if vix_val <= 25: return 2
    if vix_val <= 35: return 3
    return 4


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


def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date, vol_baselines=None,
                             atr_pct_series=None, vix_lk=None):
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

    if atr_pct_series is not None and TPSL_MODE in ('atr', 'atr_adaptive', 'trailing_decay'):
        full_indices = np.searchsorted(atr_pct_series.index.asi8, df.index.asi8)
        atr_vals = atr_pct_series.values[np.clip(full_indices, 0, len(atr_pct_series) - 1)]
        result['atr_pct'] = np.nan_to_num(atr_vals, nan=0.05).astype(np.float64)

    if vix_lk is not None and TPSL_MODE == 'regime_scaled':
        vix_regime = np.full(len(df), 2, dtype=np.int64)
        for i in range(len(df)):
            key = (int(dates[i]), int(times[i]))
            vix_val = vix_lk.get(key)
            if vix_val is not None:
                vix_regime[i] = classify_vix_regime(vix_val)
        result['vix_regime'] = vix_regime

    return result


# ==============================================================================
# DEPLOYED CONFIGS LOADER (from BQ)
# ==============================================================================
def load_deploy_configs():
    print("   Loading deployed configs from BQ...")
    query = f"SELECT * FROM `{BQ_TABLE_ID}` ORDER BY experiment, ticker, seed"
    df = _bq_client.query(query).to_dataframe()

    df['ticker'] = df['ticker'].replace(TICKER_RENAME)
    df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
    df = df[df['experiment'].isin(CLEAN_EXPERIMENTS)]
    df['survived_a'] = df['survived_a'].fillna(False).astype(bool)
    if 'vol_filter_pct' not in df.columns:
        df['vol_filter_pct'] = 0
    df['vol_filter_pct'] = df['vol_filter_pct'].fillna(0).astype(int)
    if 'tpsl_mode' in df.columns:
        df = df[(df['tpsl_mode'].isna()) | (df['tpsl_mode'] == '') | (df['tpsl_mode'] == 'static')]
    df['val_a_total_pnl'] = df['val_a_avg_pnl'] * df['val_a_count']
    df['composite'] = df['val_a_avg_pnl'] * df['val_a_win_rate'] * np.log2(1 + df['val_a_count'])

    df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])
    _latest = (df.groupby(['experiment', 'ticker', 'run_id'])['run_timestamp']
               .max().reset_index()
               .sort_values('run_timestamp')
               .groupby(['experiment', 'ticker']).last()
               .reset_index()[['experiment', 'ticker', 'run_id']])
    df = df.merge(_latest, on=['experiment', 'ticker', 'run_id'], how='inner')

    configs = []
    for ticker in sorted(df['ticker'].unique()):
        td = df[df['ticker'] == ticker]
        surv = td[td['survived_a']]
        eligible = surv[surv['val_a_count'] >= MIN_BQ_TRADES]

        if not eligible.empty:
            best_row = eligible.loc[eligible['composite'].idxmax()]
        elif not surv.empty:
            best_row = surv.loc[surv['composite'].idxmax()]
        else:
            continue

        if best_row['val_a_avg_pnl'] < DEPLOY_PNL_THRESHOLD:
            continue

        configs.append({
            'ticker': ticker,
            'experiment': best_row['experiment'],
        })

    configs.sort(key=lambda c: c['ticker'])
    print(f"      {len(configs)} tickers loaded (PnL >= {DEPLOY_PNL_THRESHOLD:.1%})")
    return configs


# ==============================================================================
# OPTIMIZATION (4-stage, per seed) — mode-aware
# ==============================================================================
def optimize_single_seed(ticker, ohlcv_arrays, regime_dict, seed_prob, seed_id):
    if ohlcv_arrays is None:
        return None

    idx = ohlcv_arrays['index']
    common = idx.intersection(seed_prob.index)
    if len(common) == 0:
        return None

    probs = np.zeros(len(idx), dtype=np.float64)
    common_locs = idx.get_indexer(common)
    probs[common_locs] = seed_prob.loc[common].values

    prices = ohlcv_arrays['prices']
    highs  = ohlcv_arrays['highs']
    lows   = ohlcv_arrays['lows']
    dates  = ohlcv_arrays['dates']
    times  = ohlcv_arrays['times']
    months = ohlcv_arrays['months']

    grid_strategy = get_grid_strategy()
    strat_keys   = list(grid_strategy.keys())
    strat_combos = list(itertools.product(*grid_strategy.values()))
    filt_keys    = list(GRID_FILTER.keys())
    filt_combos  = list(itertools.product(*GRID_FILTER.values()))

    best       = None
    best_score = -999.0

    for s_vals in strat_combos:
        s_cfg  = dict(zip(strat_keys, s_vals))
        if TPSL_MODE in ('atr', 'atr_adaptive') and s_cfg.get('sl_mult', 0) > s_cfg.get('tp_mult', 999):
            continue
        if TPSL_MODE in ('static', 'regime_scaled') and s_cfg.get('sl_pct', 0) > s_cfg.get('tp_pct', 999):
            continue
        h, m_  = map(int, s_cfg['entry_time'].split(':'))
        t_min  = h * 60 + m_

        mask    = (probs >= s_cfg['prob_thresh']) & (times == t_min)
        indices = np.where(mask)[0]
        if len(indices) == 0: continue

        raw_pnl, raw_dates = _run_sim(indices, ohlcv_arrays, s_cfg)
        if len(raw_pnl) == 0: continue

        n = len(raw_pnl)
        trade_vix, trade_tlt, trade_spy = _attach_regime(raw_dates, regime_dict, n, t_min)

        for f_vals in filt_combos:
            vix_t, tlt_c, spy_i = f_vals
            pnl = _apply_filter(raw_pnl, trade_vix, trade_tlt, trade_spy, vix_t, tlt_c, spy_i)
            if len(pnl) == 0: continue
            freq = len(pnl) / months
            if freq < MIN_TRADES_PER_MONTH: continue
            avg = float(np.mean(pnl))
            if avg > best_score:
                best_score = avg
                best = {
                    'Ticker': ticker, 'Seed': seed_id,
                    **s_cfg,
                    'vix_threshold': vix_t, 'tlt_crash': tlt_c, 'spy_ignore': spy_i,
                    'Avg_PnL': avg, 'Total_PnL': float(np.sum(pnl)),
                    'Win_Rate': float(np.mean(pnl > 0)),
                    'Freq': freq, 'Count': len(pnl), 'Stage': 1,
                }

    if best is None:
        return None

    # Stage 2: fine-tune prob_thresh
    coarse_prob = best['prob_thresh']
    fine_probs  = np.arange(
        max(0.01, round(coarse_prob - 0.15, 3)),
        min(0.99, round(coarse_prob + 0.151, 3)),
        0.005,
    ).round(3).tolist()

    h2, m2 = map(int, best['entry_time'].split(':'))
    t_min2  = h2 * 60 + m2

    for fp in fine_probs:
        mask2    = (probs >= fp) & (times == t_min2)
        indices2 = np.where(mask2)[0]
        if len(indices2) == 0: continue
        raw_pnl2, raw_dates2 = _run_sim(indices2, ohlcv_arrays, best)
        if len(raw_pnl2) == 0: continue
        n2 = len(raw_pnl2)
        tv2, tt2, ts2 = _attach_regime(raw_dates2, regime_dict, n2, t_min2)
        pnl2 = _apply_filter(raw_pnl2, tv2, tt2, ts2,
                              best['vix_threshold'], best['tlt_crash'], best['spy_ignore'])
        if len(pnl2) == 0: continue
        freq2 = len(pnl2) / months
        if freq2 < MIN_TRADES_PER_MONTH: continue
        avg2 = float(np.mean(pnl2))
        if avg2 > best_score:
            best_score = avg2
            best = {**best, 'prob_thresh': fp, 'Avg_PnL': avg2,
                    'Total_PnL': float(np.sum(pnl2)),
                    'Win_Rate': float(np.mean(pnl2 > 0)),
                    'Freq': freq2, 'Count': len(pnl2), 'Stage': 2}

    # Stage 3: fine-tune TP/SL (or multipliers) + filters + entry_time
    b_vix = best['vix_threshold']; b_tlt = best['tlt_crash']; b_spy = best['spy_ignore']
    fine_vix = np.arange(max(10, b_vix - 15), min(100, b_vix + 16), 2.0).round(1)
    fine_tlt = np.arange(max(-0.5, b_tlt - 0.10), min(0.0, b_tlt + 0.101), 0.02).round(3)
    fine_spy = np.arange(max(0.0, b_spy - 0.10), min(0.6, b_spy + 0.101), 0.02).round(3)

    b_entry_h, b_entry_m = map(int, best['entry_time'].split(':'))
    b_entry_min = b_entry_h * 60 + b_entry_m
    fine_entry = list(range(max(360, b_entry_min - 2), min(391, b_entry_min + 3)))

    if TPSL_MODE == 'trailing_decay':
        b_ts = best['trail_start_mult']; b_te = best['trail_end_mult']; b_sl_m = best['sl_mult']
        fine_ts = np.array([b_ts * x for x in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]])
        fine_te = np.array([b_te * x for x in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]])
        fine_slm = np.array([b_sl_m * x for x in [0.7, 0.85, 1.0, 1.15, 1.3]])
        s3_combos = list(itertools.product(fine_ts, fine_te, fine_slm))
        s3_is_trailing = True
    elif TPSL_MODE in ('atr', 'atr_adaptive'):
        b_tp_m = best['tp_mult']; b_sl_m = best['sl_mult']
        fine_tp_vals = np.array([b_tp_m * x for x in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]])
        fine_sl_vals = np.array([b_sl_m * x for x in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]])
        s3_combos = list(itertools.product(fine_tp_vals, fine_sl_vals))
        tp_key, sl_key = 'tp_mult', 'sl_mult'
        s3_is_trailing = False
    else:
        b_tp = best['tp_pct']; b_sl = best['sl_pct']
        fine_tp_vals = np.arange(max(0.05, b_tp - 0.05), min(0.50, b_tp + 0.051), 0.01).round(3)
        fine_sl_vals = np.arange(max(0.01, b_sl - 0.02), min(0.20, b_sl + 0.021), 0.005).round(3)
        s3_combos = list(itertools.product(fine_tp_vals, fine_sl_vals))
        tp_key, sl_key = 'tp_pct', 'sl_pct'
        s3_is_trailing = False

    f3_combos = list(itertools.product(fine_vix, fine_tlt, fine_spy))
    best_prob = best['prob_thresh']

    for t_min3 in fine_entry:
        mask3 = (probs >= best_prob) & (times == t_min3)
        for s3_vals in s3_combos:
            indices3 = np.where(mask3)[0]
            if len(indices3) == 0: continue
            if s3_is_trailing:
                s3_cfg = {**best, 'trail_start_mult': float(s3_vals[0]),
                          'trail_end_mult': float(s3_vals[1]), 'sl_mult': float(s3_vals[2])}
            else:
                s3_cfg = {**best, tp_key: float(s3_vals[0]), sl_key: float(s3_vals[1])}
            raw_pnl3, raw_dates3 = _run_sim(indices3, ohlcv_arrays, s3_cfg)
            if len(raw_pnl3) == 0: continue
            n3 = len(raw_pnl3)
            tv3, tt3, ts3 = _attach_regime(raw_dates3, regime_dict, n3, t_min3)
            for vix3, tlt3, spy3 in f3_combos:
                pnl3 = _apply_filter(raw_pnl3, tv3, tt3, ts3, float(vix3), float(tlt3), float(spy3))
                if len(pnl3) == 0: continue
                freq3 = len(pnl3) / months
                if freq3 < MIN_TRADES_PER_MONTH: continue
                avg3 = float(np.mean(pnl3))
                if avg3 > best_score:
                    best_score = avg3
                    if s3_is_trailing:
                        best = {**best,
                                'entry_time': f"{t_min3//60}:{t_min3%60:02d}",
                                'trail_start_mult': float(s3_vals[0]),
                                'trail_end_mult': float(s3_vals[1]),
                                'sl_mult': float(s3_vals[2]),
                                'vix_threshold': float(vix3), 'tlt_crash': float(tlt3),
                                'spy_ignore': float(spy3),
                                'Avg_PnL': avg3, 'Total_PnL': float(np.sum(pnl3)),
                                'Win_Rate': float(np.mean(pnl3 > 0)),
                                'Freq': freq3, 'Count': len(pnl3), 'Stage': 3}
                    else:
                        best = {**best,
                                'entry_time': f"{t_min3//60}:{t_min3%60:02d}",
                                tp_key: float(s3_vals[0]), sl_key: float(s3_vals[1]),
                                'vix_threshold': float(vix3), 'tlt_crash': float(tlt3),
                                'spy_ignore': float(spy3),
                                'Avg_PnL': avg3, 'Total_PnL': float(np.sum(pnl3)),
                                'Win_Rate': float(np.mean(pnl3 > 0)),
                                'Freq': freq3, 'Count': len(pnl3), 'Stage': 3}

    # Stage 4: Volume percentile filter
    best_entry_h4, best_entry_m4 = map(int, best['entry_time'].split(':'))
    best_t_min4 = best_entry_h4 * 60 + best_entry_m4
    vol_baselines = ohlcv_arrays.get('vol_baselines', {})

    for vpct in VOL_PCT_GRID:
        vol_ok = _compute_vol_pct_ok(ohlcv_arrays['volumes'], times, dates,
                                      vol_baselines, best_t_min4, vpct)
        mask4 = (probs >= best['prob_thresh']) & (times == best_t_min4) & vol_ok
        indices4 = np.where(mask4)[0]
        if len(indices4) == 0: continue
        raw_pnl4, raw_dates4 = _run_sim(indices4, ohlcv_arrays, best)
        if len(raw_pnl4) == 0: continue
        n4 = len(raw_pnl4)
        tv4, tt4, ts4 = _attach_regime(raw_dates4, regime_dict, n4, best_t_min4)
        pnl4 = _apply_filter(raw_pnl4, tv4, tt4, ts4,
                              best['vix_threshold'], best['tlt_crash'], best['spy_ignore'])
        if len(pnl4) == 0: continue
        freq4 = len(pnl4) / months
        if freq4 < MIN_TRADES_PER_MONTH: continue
        avg4 = float(np.mean(pnl4))
        if avg4 > best_score:
            best_score = avg4
            best = {**best, 'vol_filter_pct': vpct,
                    'Avg_PnL': avg4, 'Total_PnL': float(np.sum(pnl4)),
                    'Win_Rate': float(np.mean(pnl4 > 0)),
                    'Freq': freq4, 'Count': len(pnl4), 'Stage': 4}

    if 'vol_filter_pct' not in best:
        best['vol_filter_pct'] = 0

    if TPSL_MODE in ('atr', 'atr_adaptive'):
        best.setdefault('tp_pct', 0.0)
        best.setdefault('sl_pct', 0.0)
        best.setdefault('trail_start_mult', 0.0)
        best.setdefault('trail_end_mult', 0.0)
    elif TPSL_MODE == 'trailing_decay':
        best.setdefault('tp_pct', 0.0)
        best.setdefault('sl_pct', 0.0)
        best.setdefault('tp_mult', 0.0)
    elif TPSL_MODE == 'regime_scaled':
        best.setdefault('tp_mult', 0.0)
        best.setdefault('sl_mult', 0.0)
        best.setdefault('trail_start_mult', 0.0)
        best.setdefault('trail_end_mult', 0.0)
    else:
        best.setdefault('tp_mult', 0.0)
        best.setdefault('sl_mult', 0.0)
        best.setdefault('trail_start_mult', 0.0)
        best.setdefault('trail_end_mult', 0.0)

    return best


# ==============================================================================
# VALIDATION
# ==============================================================================
def validate_params(ticker, ohlcv_val, regime_dict, seed_prob, params):
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
# MAIN — loops through ALL 5 TPSL modes in a single execution
# ==============================================================================
if auth is not None:
    try: auth.authenticate_user()
    except Exception: pass

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL'}

print(f"\n{'='*110}")
print(f"  WALK-FORWARD OPTIMIZATION — ALL Dynamic TP/SL Modes")
print(f"  MODES: {', '.join(m.upper() for m in ALL_MODES)}")
print(f"  OPT: {OPT_START} -> {OPT_END}")
print(f"  VAL-A: {VAL_A_START} -> {VAL_A_END}")
print(f"  VAL-B: {VAL_B_START} -> {VAL_B_END}")
for _m in ALL_MODES:
    TPSL_MODE = _m
    _gs = get_grid_strategy()
    _ns = len(list(itertools.product(*_gs.values())))
    _nf = len(list(itertools.product(*GRID_FILTER.values())))
    print(f"    {_m.upper():>15}: {_ns:,} strategy x {_nf:,} filter = {_ns*_nf:,} combos")
TPSL_MODE = ALL_MODES[0]
print(f"{'='*110}\n")

deploy_configs = load_deploy_configs()

if TICKER_SUBSET:
    deploy_configs = [c for c in deploy_configs if c['ticker'] in TICKER_SUBSET]
    print(f"  Filtered to TICKER_SUBSET: {len(deploy_configs)} tickers")

completed_triples = set()
if SKIP_COMPLETED:
    skip_q = f"SELECT DISTINCT experiment, ticker, tpsl_mode FROM `{BQ_TABLE_ID}` WHERE tpsl_mode IS NOT NULL"
    for row in _bq_client.query(skip_q):
        completed_triples.add((row.experiment, row.ticker, row.tpsl_mode or 'static'))

regime_dict, vix_lk = preload_regime_data(OPT_START, VAL_B_END)

RUN_ID = str(uuid.uuid4())
print(f"  BQ run_id: {RUN_ID[:8]}...")

all_results_by_mode = {m: [] for m in ALL_MODES}
n_total = len(deploy_configs)

for cfg_idx, cfg in enumerate(deploy_configs):
    ticker = cfg['ticker']
    base_experiment = cfg['experiment']

    print(f"\n{'='*70}")
    print(f"  [{cfg_idx+1}/{n_total}] {ticker} — {base_experiment}")
    print(f"{'='*70}")

    try:
        df_ohlcv = load_ticker_ohlcv(ticker)
    except Exception as e:
        print(f"  Could not load OHLCV: {e}")
        continue

    atr_standard = compute_daily_atr_series(df_ohlcv)
    atr_adaptive_s = compute_adaptive_atr_series(df_ohlcv)
    print(f"  ATR_pct mean={atr_standard.dropna().mean():.4f}  |  "
          f"Adaptive ATR_pct mean={atr_adaptive_s.dropna().mean():.4f}")

    vol_baselines = precompute_vol_baselines(df_ohlcv)

    seed_scores = load_seed_scores(ticker, base_experiment)
    if not seed_scores:
        print(f"  No seed scores found for {base_experiment}")
        del df_ohlcv, atr_standard, atr_adaptive_s, vol_baselines; gc.collect()
        continue
    print(f"  {len(seed_scores)} seeds loaded")

    for mode in ALL_MODES:
        TPSL_MODE = mode
        mode_sfx = MODE_SUFFIX.get(mode, '')
        experiment = f"{base_experiment}{mode_sfx}"

        if (experiment, ticker, mode) in completed_triples:
            print(f"\n  [{mode.upper()}] Skipping — already completed in BQ")
            continue

        print(f"\n{'─'*70}")
        print(f"  [{mode.upper()}] {ticker} — {experiment}")
        print(f"{'─'*70}")

        if mode in ('atr', 'trailing_decay'):
            atr_pct_series = atr_standard
        elif mode == 'atr_adaptive':
            atr_pct_series = atr_adaptive_s
        else:
            atr_pct_series = None

        ohlcv_opt   = precompute_ohlcv_arrays(df_ohlcv, OPT_START, OPT_END,
                                               vol_baselines, atr_pct_series, vix_lk)
        ohlcv_val_a = precompute_ohlcv_arrays(df_ohlcv, VAL_A_START, VAL_A_END,
                                               vol_baselines, atr_pct_series, vix_lk)
        ohlcv_val_b = precompute_ohlcv_arrays(df_ohlcv, VAL_B_START, VAL_B_END,
                                               vol_baselines, atr_pct_series, vix_lk)

        if ohlcv_opt is None:
            print(f"    No data in optimization window")
            continue

        for seed_id, seed_prob in sorted(seed_scores.items()):
            print(f"\n    Seed {seed_id}:")

            opt_result = optimize_single_seed(ticker, ohlcv_opt, regime_dict, seed_prob, seed_id)
            if opt_result is None:
                print(f"      No valid optimization result")
                continue

            if mode in ('atr', 'atr_adaptive'):
                tp_sl_str = f"tp_m={opt_result.get('tp_mult',0):.1f}  sl_m={opt_result.get('sl_mult',0):.1f}"
            elif mode == 'trailing_decay':
                tp_sl_str = (f"ts={opt_result.get('trail_start_mult',0):.1f}  "
                             f"te={opt_result.get('trail_end_mult',0):.1f}  "
                             f"sl_m={opt_result.get('sl_mult',0):.1f}")
            else:
                tp_sl_str = f"TP={opt_result['tp_pct']:.0%}  SL={opt_result['sl_pct']:.0%}"

            print(f"      OPT: AvgPnL={opt_result['Avg_PnL']:.2%}  WR={opt_result['Win_Rate']:.1%}  "
                  f"N={opt_result['Count']}  Entry={opt_result['entry_time']}  "
                  f"Prob>{opt_result['prob_thresh']:.3f}  {tp_sl_str}")

            val_a = validate_params(ticker, ohlcv_val_a, regime_dict, seed_prob, opt_result)
            if val_a is None:
                val_a = {'Avg_PnL': 0, 'Total_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}
            survived_a = val_a['Avg_PnL'] > 0
            tag_a = "PASS" if survived_a else "FAIL"
            print(f"      VAL-A: AvgPnL={val_a['Avg_PnL']:.2%}  WR={val_a['Win_Rate']:.1%}  "
                  f"N={val_a['Count']}  [{tag_a}]")

            val_b = validate_params(ticker, ohlcv_val_b, regime_dict, seed_prob, opt_result)
            if val_b is None:
                val_b = {'Avg_PnL': 0, 'Total_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}
            survived_b = val_b['Avg_PnL'] > 0
            tag_b = "PASS" if survived_b else "FAIL"
            print(f"      VAL-B: AvgPnL={val_b['Avg_PnL']:.2%}  WR={val_b['Win_Rate']:.1%}  "
                  f"N={val_b['Count']}  [{tag_b}]")

            result_row = {
                'Ticker': ticker, 'Seed': seed_id,
                'entry_time': opt_result['entry_time'],
                'prob_thresh': opt_result['prob_thresh'],
                'tp_pct': opt_result.get('tp_pct', 0), 'sl_pct': opt_result.get('sl_pct', 0),
                'tp_mult': opt_result.get('tp_mult', 0), 'sl_mult': opt_result.get('sl_mult', 0),
                'trail_start_mult': opt_result.get('trail_start_mult', 0),
                'trail_end_mult': opt_result.get('trail_end_mult', 0),
                'vix_threshold': opt_result['vix_threshold'],
                'tlt_crash': opt_result['tlt_crash'],
                'spy_ignore': opt_result['spy_ignore'],
                'vol_filter_pct': opt_result.get('vol_filter_pct', 0),
                'Avg_PnL': opt_result['Avg_PnL'],
                'Win_Rate': opt_result['Win_Rate'],
                'Freq': opt_result['Freq'],
                'Count': opt_result['Count'],
                'val_a_avg_pnl': val_a['Avg_PnL'],
                'val_a_win_rate': val_a['Win_Rate'],
                'val_a_freq': val_a['Freq'],
                'val_a_count': val_a['Count'],
                'survived_a': survived_a,
                'val_b_avg_pnl': val_b['Avg_PnL'],
                'val_b_win_rate': val_b['Win_Rate'],
                'val_b_freq': val_b['Freq'],
                'val_b_count': val_b['Count'],
                'survived_b': survived_b,
            }
            all_results_by_mode[mode].append(result_row)
            save_row_to_bq(result_row, RUN_ID, experiment,
                            OPT_START, OPT_END, VAL_A_START, VAL_A_END,
                            VAL_B_START, VAL_B_END)

        del ohlcv_opt, ohlcv_val_a, ohlcv_val_b; gc.collect()

    del df_ohlcv, atr_standard, atr_adaptive_s, seed_scores, vol_baselines; gc.collect()


# ==============================================================================
# SUMMARY — per-mode breakdown
# ==============================================================================
W = 145
grand_total = sum(len(v) for v in all_results_by_mode.values())

if grand_total == 0:
    print("\nNo results found across any mode.")
else:
    for mode in ALL_MODES:
        mode_results = all_results_by_mode[mode]
        if not mode_results:
            print(f"\n  {mode.upper()}: No results")
            continue

        df = pd.DataFrame(mode_results)
        best_per_ticker = df.loc[df.groupby('Ticker')['Avg_PnL'].idxmax()]

        print(f"\n\n{'='*W}")
        print(f"  WF OPTIMIZATION — {mode.upper()} MODE — VAL-A Results")
        print(f"  OPT: {OPT_START} -> {OPT_END}  |  VAL: {VAL_A_START} -> {VAL_A_END}")
        print(f"{'='*W}")

        if mode in ('atr', 'atr_adaptive'):
            print(f"  {'Ticker':<7} {'Seed':>5} {'Prob':>6} {'Entry':>6} {'tp_m':>5} {'sl_m':>5} "
                  f"{'VIX<':>5} {'TLT>':>6} {'SPY<':>5} "
                  f"{'|':>2} {'OPT PnL':>8} {'OPT WR':>7} {'OPT N':>6} "
                  f"{'|':>2} {'VAL PnL':>8} {'VAL WR':>7} {'VAL N':>6} {'Result':>8}")
        elif mode == 'trailing_decay':
            print(f"  {'Ticker':<7} {'Seed':>5} {'Prob':>6} {'Entry':>6} {'t_s':>5} {'t_e':>5} {'sl_m':>5} "
                  f"{'VIX<':>5} {'TLT>':>6} {'SPY<':>5} "
                  f"{'|':>2} {'OPT PnL':>8} {'OPT WR':>7} {'OPT N':>6} "
                  f"{'|':>2} {'VAL PnL':>8} {'VAL WR':>7} {'VAL N':>6} {'Result':>8}")
        else:
            print(f"  {'Ticker':<7} {'Seed':>5} {'Prob':>6} {'Entry':>6} {'TP':>4} {'SL':>4} "
                  f"{'VIX<':>5} {'TLT>':>6} {'SPY<':>5} "
                  f"{'|':>2} {'OPT PnL':>8} {'OPT WR':>7} {'OPT N':>6} "
                  f"{'|':>2} {'VAL PnL':>8} {'VAL WR':>7} {'VAL N':>6} {'Result':>8}")
        print(f"  {'─'*W}")

        for _, r in best_per_ticker.sort_values('val_a_avg_pnl', ascending=False).iterrows():
            tag = "PASS" if r['survived_a'] else "FAIL"
            if mode in ('atr', 'atr_adaptive'):
                print(f"  {r['Ticker']:<7} {r['Seed']:>5} {r['prob_thresh']:>6.3f} {r['entry_time']:>6} "
                      f"{r['tp_mult']:>5.1f} {r['sl_mult']:>5.1f} "
                      f"{r['vix_threshold']:>5.0f} {r['tlt_crash']:>6.2f} {r['spy_ignore']:>5.2f} "
                      f"{'|':>2} {r['Avg_PnL']:>7.2%} {r['Win_Rate']:>6.1%} {int(r['Count']):>6} "
                      f"{'|':>2} {r['val_a_avg_pnl']:>7.2%} {r['val_a_win_rate']:>6.1%} {int(r['val_a_count']):>6} "
                      f"{'  ' + tag:>8}")
            elif mode == 'trailing_decay':
                print(f"  {r['Ticker']:<7} {r['Seed']:>5} {r['prob_thresh']:>6.3f} {r['entry_time']:>6} "
                      f"{r['trail_start_mult']:>5.1f} {r['trail_end_mult']:>5.1f} {r['sl_mult']:>5.1f} "
                      f"{r['vix_threshold']:>5.0f} {r['tlt_crash']:>6.2f} {r['spy_ignore']:>5.2f} "
                      f"{'|':>2} {r['Avg_PnL']:>7.2%} {r['Win_Rate']:>6.1%} {int(r['Count']):>6} "
                      f"{'|':>2} {r['val_a_avg_pnl']:>7.2%} {r['val_a_win_rate']:>6.1%} {int(r['val_a_count']):>6} "
                      f"{'  ' + tag:>8}")
            else:
                print(f"  {r['Ticker']:<7} {r['Seed']:>5} {r['prob_thresh']:>6.3f} {r['entry_time']:>6} "
                      f"{r['tp_pct']:>3.0%} {r['sl_pct']:>3.0%} "
                      f"{r['vix_threshold']:>5.0f} {r['tlt_crash']:>6.2f} {r['spy_ignore']:>5.2f} "
                      f"{'|':>2} {r['Avg_PnL']:>7.2%} {r['Win_Rate']:>6.1%} {int(r['Count']):>6} "
                      f"{'|':>2} {r['val_a_avg_pnl']:>7.2%} {r['val_a_win_rate']:>6.1%} {int(r['val_a_count']):>6} "
                      f"{'  ' + tag:>8}")

        passed_a = best_per_ticker[best_per_ticker['survived_a']]
        print(f"  {'─'*W}")
        print(f"  {len(passed_a)}/{len(best_per_ticker)} tickers PASSED VAL-A")
        if len(passed_a) > 0:
            print(f"  PASS avg OPT PnL: {passed_a['Avg_PnL'].mean():.2%}  ->  "
                  f"VAL-A PnL: {passed_a['val_a_avg_pnl'].mean():.2%}")

        ticker_pass = df.groupby('Ticker').agg(
            seeds=('Seed', 'count'),
            pass_a=('survived_a', 'sum'),
            pass_b=('survived_b', 'sum'),
        ).sort_values('pass_a', ascending=False)
        print(f"\n  {'Ticker':<7} {'Seeds':>6} {'PASS-A':>7} {'PASS-B':>7}")
        print(f"  {'─'*30}")
        for ticker, r in ticker_pass.iterrows():
            print(f"  {ticker:<7} {int(r['seeds']):>6} "
                  f"{int(r['pass_a']):>4}/{int(r['seeds'])} "
                  f"{int(r['pass_b']):>4}/{int(r['seeds'])}")

    print(f"\n\n{'='*W}")
    print(f"  ALL MODES COMPLETE — {grand_total} total seed results")
    for mode in ALL_MODES:
        n_mode = len(all_results_by_mode[mode])
        if n_mode > 0:
            print(f"    {mode.upper():>15}: {n_mode} results")
    print(f"  Run analyze_dynamic_tpsl.py to compare all modes head-to-head")
    print(f"{'='*W}")
