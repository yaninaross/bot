"""
optimize_wf_robust.py — Rolling Walk-Forward Optimizer v3
=========================================================
Anchored expanding-window walk-forward optimization in 2 rounds:

  R1 (broad):   OPT Oct-Nov   → VAL Dec-Jan    sparse grid, no filters
  R2 (refine):  OPT Oct-Jan   → VAL Feb-Mar    expand around R1 survivors + filters
  TEST:         Apr 1 → May 21                  2-month pure holdout

Per-seed optimization + ensemble evaluation (average probs across all seeds,
test with best seed's params). Results tagged 'robust_wf_v3' in BQ.
Supports all 5 TPSL modes. Same Numba sim engines, data loading, BQ.
Run in Colab with GPU (for numba JIT).
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
            'model_version': MODEL_VERSION_TAG,
            'v0_fixed': True,
            'is_ensemble': bool(row_dict.get('is_ensemble', False)),
            'n_seeds_avg': int(row_dict.get('n_seeds_avg', 0)),
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
# CONFIG
# ==============================================================================
BUCKET_NAME     = os.environ.get("GCS_BUCKET", "")
TICKER_FILE_MAP = {'BITF': 'KEEL'}
TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}
DEPLOY_PNL_THRESHOLD = 0.018
MIN_BQ_TRADES   = 3

EXP_FAMILIES = {
    'late_cutoff': {
        'label': 'Late Cutoff (Oct 2025)',
        'experiments': {'105_DEFENSE', '106_DEFENSE', '152_XSECTOR'},
        'wf_rounds': [
            {'name': 'R1', 'opt': ('2025-10-02', '2025-11-30'),
                           'val': ('2025-12-01', '2026-01-31'), 'min_trades': 6},
            {'name': 'R2', 'opt': ('2025-12-01', '2026-01-31'),
                           'val': ('2026-02-01', '2026-03-31'), 'min_trades': 10},
        ],
        'test_start': '2026-04-01',
        'test_end':   '2026-05-21',
        'fallback_tickers': [
            'LUNR', 'RCAT', 'ASTS', 'BBAI', 'ACHR', 'MVIS',
            'CIFR', 'IREN', 'HUT', 'KEEL', 'CLSK', 'APLD',
            'WULF', 'MARA', 'RIOT', 'MSTR', 'RKLB', 'JOBY',
            'ALAB', 'BGDE', 'PLUG', 'FCEL', 'BTBT', 'BE',
        ],
    },
    'early_cutoff': {
        'label': 'Early Cutoff (Mar 2025)',
        'experiments': {'160_XSECTOR'},
        'wf_rounds': [
            {'name': 'R1', 'opt': ('2025-04-01', '2025-07-31'),
                           'val': ('2025-08-01', '2025-09-30'), 'min_trades': 6},
            {'name': 'R2', 'opt': ('2025-08-01', '2025-11-30'),
                           'val': ('2025-12-01', '2026-01-31'), 'min_trades': 8},
            {'name': 'R3', 'opt': ('2025-12-01', '2026-01-31'),
                           'val': ('2026-02-01', '2026-03-31'), 'min_trades': 10},
        ],
        'test_start': '2026-04-01',
        'test_end':   '2026-05-21',
        'fallback_tickers': [
            'LUNR', 'RCAT', 'ASTS', 'BBAI', 'ACHR', 'MVIS',
            'CIFR', 'IREN', 'HUT', 'KEEL', 'CLSK', 'APLD',
            'WULF', 'MARA', 'RIOT', 'MSTR', 'RKLB', 'JOBY',
            'ALAB', 'BGDE', 'PLUG', 'FCEL', 'BTBT', 'BE',
        ],
    },
    'walkforward': {
        'label': 'Walk-Forward (6 folds)',
        'experiments': {'170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                        '173_XSECTOR', '174_XSECTOR', '175_XSECTOR'},
        'wf_rounds': [
            {'name': 'R1', 'opt': ('2025-04-01', '2025-07-31'),
                           'val': ('2025-08-01', '2025-09-30'), 'min_trades': 6},
            {'name': 'R2', 'opt': ('2025-08-01', '2025-11-30'),
                           'val': ('2025-12-01', '2026-01-31'), 'min_trades': 8},
            {'name': 'R3', 'opt': ('2025-12-01', '2026-01-31'),
                           'val': ('2026-02-01', '2026-03-31'), 'min_trades': 10},
        ],
        'test_start': '2026-04-01',
        'test_end':   '2026-05-21',
        'stitch': True,
        'fallback_tickers': [
            'LUNR', 'RCAT', 'ASTS', 'BBAI', 'ACHR', 'MVIS',
            'CIFR', 'IREN', 'HUT', 'KEEL', 'CLSK', 'APLD',
            'WULF', 'MARA', 'RIOT', 'MSTR', 'RKLB', 'JOBY',
            'ALAB', 'BGDE', 'PLUG', 'FCEL', 'BTBT', 'BE',
        ],
    },
}

ACTIVE_FAMILY = 'walkforward'  # change per run: 'late_cutoff', 'early_cutoff', 'walkforward'

_family = EXP_FAMILIES[ACTIVE_FAMILY]
CLEAN_EXPERIMENTS = _family['experiments']
WF_ROUNDS  = _family['wf_rounds']
TEST_START = _family['test_start']
TEST_END   = _family['test_end']

NO_FILTER_DEFAULTS = {'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0}

EXIT_TIME_MIN        = 780
MIN_TRADES_PER_MONTH = 2.0
VOL_LOOKBACK_DAYS    = 30
VOL_PCT_GRID         = [0, 25]
SLIPPAGE_PCT         = 0.0015  # 0.15% each way = 0.3% round-trip

MODEL_VERSION_TAG = 'robust_wf_v5'
SKIP_COMPLETED = True
TICKER_SUBSET  = ['RKLB', 'JOBY', 'ALAB', 'BGDE']  # partition 3/6

# ==============================================================================
# GRIDS — Tiered for rolling walk-forward
# ==============================================================================
# Round 1: Broad, sparse (81 combos for static)
GRID_R1 = {
    'static': {
        'entry_time':  ["06:15", "06:20", "06:25"],
        'prob_thresh': [0.30, 0.50, 0.70],
        'tp_pct':      [0.08, 0.16, 0.25],
        'sl_pct':      [0.03, 0.08, 0.14],
    },
    'atr': {
        'entry_time':  ["06:15", "06:20", "06:25"],
        'prob_thresh': [0.30, 0.50, 0.70],
        'tp_mult':     [1.5, 3.0, 8.0],
        'sl_mult':     [0.5, 1.5, 4.0],
    },
    'atr_adaptive': {
        'entry_time':  ["06:15", "06:20", "06:25"],
        'prob_thresh': [0.30, 0.50, 0.70],
        'tp_mult':     [1.5, 3.0, 8.0],
        'sl_mult':     [0.5, 1.5, 4.0],
    },
    'regime_scaled': {
        'entry_time':  ["06:15", "06:20", "06:25"],
        'prob_thresh': [0.30, 0.50, 0.70],
        'tp_pct':      [0.08, 0.16, 0.25],
        'sl_pct':      [0.03, 0.08, 0.14],
    },
    'trailing_decay': {
        'entry_time':       ["06:15", "06:20", "06:25"],
        'prob_thresh':      [0.30, 0.50, 0.70],
        'trail_start_mult': [2.0, 4.0, 7.0],
        'trail_end_mult':   [0.5, 1.0, 1.5],
        'sl_mult':          [1.0, 2.0, 4.0],
    },
}

# Round 2 reference: Full resolution (expand around R1 survivors)
GRID_R2_REF = {
    'static': {
        'entry_time':  ["06:15", "06:20", "06:25"],
        'prob_thresh': [0.30, 0.40, 0.50, 0.60, 0.70],
        'tp_pct':      [0.08, 0.12, 0.16, 0.20, 0.25],
        'sl_pct':      [0.03, 0.05, 0.08, 0.10, 0.14],
    },
    'atr': {
        'entry_time':  ["06:15", "06:20", "06:25"],
        'prob_thresh': [0.30, 0.40, 0.50, 0.60, 0.70],
        'tp_mult':     [1.5, 2.0, 3.0, 5.0, 8.0],
        'sl_mult':     [0.5, 1.0, 1.5, 2.5, 4.0],
    },
    'atr_adaptive': {
        'entry_time':  ["06:15", "06:20", "06:25"],
        'prob_thresh': [0.30, 0.40, 0.50, 0.60, 0.70],
        'tp_mult':     [1.5, 2.0, 3.0, 5.0, 8.0],
        'sl_mult':     [0.5, 1.0, 1.5, 2.5, 4.0],
    },
    'regime_scaled': {
        'entry_time':  ["06:15", "06:20", "06:25"],
        'prob_thresh': [0.30, 0.40, 0.50, 0.60, 0.70],
        'tp_pct':      [0.08, 0.12, 0.16, 0.20, 0.25],
        'sl_pct':      [0.03, 0.05, 0.08, 0.10, 0.14],
    },
    'trailing_decay': {
        'entry_time':       ["06:15", "06:20", "06:25"],
        'prob_thresh':      [0.30, 0.40, 0.50, 0.60, 0.70],
        'trail_start_mult': [2.0, 4.0, 7.0],
        'trail_end_mult':   [0.5, 1.0, 1.5],
        'sl_mult':          [1.0, 2.0, 4.0],
    },
}

GRID_FILTER = {
    'vix_threshold': [20.0, 30.0, 50.0],
    'tlt_crash':     [0.0, -0.20, -0.50],
    'spy_ignore':    [0.0, 0.20, 0.50],
}

REGIME_TP_SCALE = np.array([0.70, 0.85, 1.00, 1.30, 1.60], dtype=np.float64)
REGIME_SL_SCALE = np.array([0.60, 0.80, 1.00, 1.30, 1.80], dtype=np.float64)

VIX_REGIME_BOUNDS = np.array([15.0, 20.0, 25.0, 35.0], dtype=np.float64)


def get_grid_strategy(round_num=1):
    grid_set = GRID_R1 if round_num == 1 else GRID_R2_REF
    return grid_set.get(TPSL_MODE, grid_set.get('static'))


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
        pnl, d = run_trade_sim_atr(
            indices, prices, highs, lows, dates, times,
            ohlcv_arrays['atr_pct'], s_cfg['tp_mult'], s_cfg['sl_mult'], EXIT_TIME_MIN)
    elif TPSL_MODE == 'trailing_decay':
        pnl, d = run_trade_sim_trailing_decay(
            indices, prices, highs, lows, dates, times,
            ohlcv_arrays['atr_pct'], s_cfg['trail_start_mult'], s_cfg['trail_end_mult'],
            s_cfg['sl_mult'], EXIT_TIME_MIN)
    elif TPSL_MODE == 'regime_scaled':
        pnl, d = run_trade_sim_regime(
            indices, prices, highs, lows, dates, times,
            ohlcv_arrays['vix_regime'], s_cfg['tp_pct'], s_cfg['sl_pct'],
            REGIME_TP_SCALE, REGIME_SL_SCALE, EXIT_TIME_MIN)
    else:
        pnl, d = run_trade_sim(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN)
    if len(pnl) > 0 and SLIPPAGE_PCT > 0:
        pnl = pnl - 2 * SLIPPAGE_PCT
    return pnl, d


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
    family = EXP_FAMILIES[ACTIVE_FAMILY]
    fallback = [t for t in family['fallback_tickers'] if t not in EXCLUDE_TICKERS]

    print(f"   Loading configs for family '{ACTIVE_FAMILY}' ({family['label']})...")

    configs = []
    seen = set()
    for exp in sorted(family['experiments']):
        for t in sorted(fallback):
            key = (t, exp)
            if key not in seen:
                configs.append({'ticker': t, 'experiment': exp})
                seen.add(key)

    configs.sort(key=lambda c: (c['ticker'], c['experiment']))
    print(f"      {len(configs)} ticker-experiment combos across {len(family['experiments'])} experiments")
    return configs


# ==============================================================================
# STABILITY-WEIGHTED PARAMETER SELECTION
# ==============================================================================
def _select_robust_params(all_results):
    if not all_results:
        return None

    strategy_groups = {}
    for r in all_results:
        if TPSL_MODE == 'trailing_decay':
            key = (r['entry_time'], r['prob_thresh'],
                   r.get('trail_start_mult', 0), r.get('trail_end_mult', 0),
                   r.get('sl_mult', 0))
        elif TPSL_MODE in ('atr', 'atr_adaptive'):
            key = (r['entry_time'], r['prob_thresh'],
                   r.get('tp_mult', 0), r.get('sl_mult', 0))
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
        robust_score = mean_pnl * (0.5 + 0.5 * frac_positive)
        if robust_score > best_robust_score:
            best_robust_score = robust_score
            best = max(results, key=lambda r: r['Avg_PnL'])
    return best


def _expand_around_survivors(survivors, param_name, reference_values):
    ref_sorted = sorted(reference_values)
    survivor_vals = set(s[param_name] for s in survivors if param_name in s)
    if not survivor_vals:
        return ref_sorted
    expanded = set()
    for sv in survivor_vals:
        if sv in ref_sorted:
            idx = ref_sorted.index(sv)
            for offset in [-1, 0, 1]:
                ni = idx + offset
                if 0 <= ni < len(ref_sorted):
                    expanded.add(ref_sorted[ni])
        else:
            expanded.add(sv)
    return sorted(expanded)


def _grid_search(ohlcv_arrays, seed_prob, regime_dict, ticker, seed_id,
                 strat_grid, use_filters, min_trades):
    idx = ohlcv_arrays['index']
    common = idx.intersection(seed_prob.index)
    if len(common) == 0:
        return []

    probs = np.zeros(len(idx), dtype=np.float64)
    common_locs = idx.get_indexer(common)
    probs[common_locs] = seed_prob.loc[common].values

    times  = ohlcv_arrays['times']
    months = ohlcv_arrays['months']

    strat_keys   = list(strat_grid.keys())
    strat_combos = list(itertools.product(*strat_grid.values()))

    if use_filters:
        filt_combos = list(itertools.product(*GRID_FILTER.values()))
    else:
        filt_combos = [(NO_FILTER_DEFAULTS['vix_threshold'],
                        NO_FILTER_DEFAULTS['tlt_crash'],
                        NO_FILTER_DEFAULTS['spy_ignore'])]

    results = []
    for s_vals in strat_combos:
        s_cfg = dict(zip(strat_keys, s_vals))
        if TPSL_MODE in ('atr', 'atr_adaptive') and s_cfg.get('sl_mult', 0) > s_cfg.get('tp_mult', 999):
            continue
        if TPSL_MODE in ('static', 'regime_scaled') and s_cfg.get('sl_pct', 0) > s_cfg.get('tp_pct', 999):
            continue
        h, m_ = map(int, s_cfg['entry_time'].split(':'))
        t_min = h * 60 + m_

        mask    = (probs >= s_cfg['prob_thresh']) & (times == t_min)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        raw_pnl, raw_dates = _run_sim(indices, ohlcv_arrays, s_cfg)
        if len(raw_pnl) == 0:
            continue

        n = len(raw_pnl)
        trade_vix, trade_tlt, trade_spy = _attach_regime(raw_dates, regime_dict, n, t_min)

        for vix_t, tlt_c, spy_i in filt_combos:
            pnl = _apply_filter(raw_pnl, trade_vix, trade_tlt, trade_spy, vix_t, tlt_c, spy_i)
            if len(pnl) < min_trades:
                continue
            freq = len(pnl) / months
            if freq < MIN_TRADES_PER_MONTH:
                continue
            avg = float(np.mean(pnl))
            results.append({
                'Ticker': ticker, 'Seed': seed_id,
                **s_cfg,
                'vix_threshold': vix_t, 'tlt_crash': tlt_c, 'spy_ignore': spy_i,
                'Avg_PnL': avg, 'Total_PnL': float(np.sum(pnl)),
                'Win_Rate': float(np.mean(pnl > 0)),
                'Freq': freq, 'Count': len(pnl),
            })
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
def optimize_rolling_wf(ticker, df_ohlcv, vol_baselines, atr_pct_series,
                        vix_lk, regime_dict, seed_prob, seed_id):

    def _precompute(start, end):
        return precompute_ohlcv_arrays(df_ohlcv, start, end,
                                        vol_baselines, atr_pct_series, vix_lk)

    # ── Round 1: Broad grid, no filters ──────────────────────────────────────
    r1 = WF_ROUNDS[0]
    ohlcv_r1_opt = _precompute(*r1['opt'])
    ohlcv_r1_val = _precompute(*r1['val'])
    if ohlcv_r1_opt is None:
        return None

    r1_grid = get_grid_strategy(round_num=1)
    r1_opt = _grid_search(ohlcv_r1_opt, seed_prob, regime_dict, ticker, seed_id,
                           r1_grid, use_filters=False, min_trades=r1['min_trades'])

    r1_survivors = []
    for combo in r1_opt:
        if combo['Avg_PnL'] <= 0:
            continue
        val = _eval_combo(combo, ohlcv_r1_val, seed_prob, regime_dict)
        if val['Avg_PnL'] > 0.005 and val['Count'] >= r1['min_trades']:
            r1_survivors.append(combo)

    del ohlcv_r1_opt, ohlcv_r1_val
    print(f"        R1: {len(r1_opt)} tested → {len(r1_survivors)} survived")

    if not r1_survivors:
        return None

    # ── Round 2: Expand around survivors + add filters ────────────────────────
    r2 = WF_ROUNDS[1]
    ohlcv_r2_opt = _precompute(*r2['opt'])
    ohlcv_r2_val = _precompute(*r2['val'])

    r2_ref = get_grid_strategy(round_num=2)
    r2_grid = {}
    for key, ref_vals in r2_ref.items():
        r2_grid[key] = _expand_around_survivors(r1_survivors, key, ref_vals)

    r2_opt = _grid_search(ohlcv_r2_opt, seed_prob, regime_dict, ticker, seed_id,
                           r2_grid, use_filters=True, min_trades=r2['min_trades'])

    r2_survivors = []
    for combo in r2_opt:
        if combo['Avg_PnL'] <= 0:
            continue
        val = _eval_combo(combo, ohlcv_r2_val, seed_prob, regime_dict)
        if val['Avg_PnL'] > 0:
            r2_survivors.append(combo)

    print(f"        R2: {len(r2_opt)} tested → {len(r2_survivors)} survived")

    if not r2_survivors:
        best = max(r1_survivors, key=lambda r: r['Avg_PnL'])
        best.update(NO_FILTER_DEFAULTS)
        best['vol_filter_pct'] = 0
        _fill_defaults(best)
        val_result = validate_params(ticker, ohlcv_r2_val, regime_dict, seed_prob, best)
        del ohlcv_r2_val
        if val_result is None:
            val_result = {'Avg_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}
        return best, val_result

    # ── Select best from R2 survivors + volume filter ──────────────────────────
    best = _select_robust_params(r2_survivors)
    if best is None:
        best = max(r2_survivors, key=lambda r: r['Avg_PnL'])

    best_score = best['Avg_PnL']
    h_b, m_b = map(int, best['entry_time'].split(':'))
    t_min_b = h_b * 60 + m_b

    idx = ohlcv_r2_opt['index']
    common = idx.intersection(seed_prob.index)
    probs = np.zeros(len(idx), dtype=np.float64)
    if len(common) > 0:
        common_locs = idx.get_indexer(common)
        probs[common_locs] = seed_prob.loc[common].values

    for vpct in VOL_PCT_GRID:
        if vpct == 0:
            continue
        vol_ok = _compute_vol_pct_ok(ohlcv_r2_opt['volumes'], ohlcv_r2_opt['times'],
                                      ohlcv_r2_opt['dates'],
                                      ohlcv_r2_opt.get('vol_baselines', {}), t_min_b, vpct)
        mask = (probs >= best['prob_thresh']) & (ohlcv_r2_opt['times'] == t_min_b) & vol_ok
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        raw_pnl, raw_dates = _run_sim(indices, ohlcv_r2_opt, best)
        if len(raw_pnl) == 0:
            continue
        n = len(raw_pnl)
        tv, tt, ts = _attach_regime(raw_dates, regime_dict, n, t_min_b)
        pnl = _apply_filter(raw_pnl, tv, tt, ts,
                             best['vix_threshold'], best['tlt_crash'], best['spy_ignore'])
        if len(pnl) < r2['min_trades']:
            continue
        avg = float(np.mean(pnl))
        if avg > best_score:
            best_score = avg
            best = {**best, 'vol_filter_pct': vpct,
                    'Avg_PnL': avg, 'Total_PnL': float(np.sum(pnl)),
                    'Win_Rate': float(np.mean(pnl > 0)),
                    'Freq': len(pnl) / ohlcv_r2_opt['months'], 'Count': len(pnl)}

    del ohlcv_r2_opt

    best.setdefault('vol_filter_pct', 0)
    _fill_defaults(best)

    val_result = validate_params(ticker, ohlcv_r2_val, regime_dict, seed_prob, best)
    del ohlcv_r2_val
    if val_result is None:
        val_result = {'Avg_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}

    return best, val_result


def _fill_defaults(best):
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
print(f"  ROLLING WALK-FORWARD OPTIMIZATION — {_family['label']}")
print(f"  FAMILY: {ACTIVE_FAMILY}  |  MODES: {', '.join(m.upper() for m in ALL_MODES)}")
for rnd in WF_ROUNDS:
    print(f"  {rnd['name']}: OPT {rnd['opt'][0]}->{rnd['opt'][1]}  VAL {rnd['val'][0]}->{rnd['val'][1]}  min_trades={rnd['min_trades']}")
print(f"  TEST: {TEST_START} -> {TEST_END} (pure holdout)")
for _m in ALL_MODES:
    TPSL_MODE = _m
    _gs = get_grid_strategy(round_num=1)
    _ns = len(list(itertools.product(*_gs.values())))
    _nf = len(list(itertools.product(*GRID_FILTER.values())))
    print(f"    {_m.upper():>15}: R1={_ns} broad → R2=survivors×{_nf} filters → TEST")
TPSL_MODE = ALL_MODES[0]
print(f"{'='*110}\n")

deploy_configs = load_deploy_configs()

if TICKER_SUBSET:
    deploy_configs = [c for c in deploy_configs if c['ticker'] in TICKER_SUBSET]
    print(f"  Filtered to TICKER_SUBSET: {len(deploy_configs)} tickers")

completed_triples = set()
if SKIP_COMPLETED:
    exp_filter = ','.join(repr(e) for e in sorted(CLEAN_EXPERIMENTS))
    skip_q = (f"SELECT DISTINCT experiment, ticker, tpsl_mode FROM `{BQ_TABLE_ID}` "
              f"WHERE model_version = '{MODEL_VERSION_TAG}' "
              f"AND experiment IN ({exp_filter})")
    for row in _bq_client.query(skip_q):
        completed_triples.add((row.experiment, row.ticker, row.tpsl_mode or 'static'))

regime_dict, vix_lk = preload_regime_data(WF_ROUNDS[0]['opt'][0], TEST_END)

RUN_ID = str(uuid.uuid4())
print(f"  BQ run_id: {RUN_ID[:8]}...")

all_results_by_mode = {m: [] for m in ALL_MODES}
ensemble_results_by_mode = {m: [] for m in ALL_MODES}
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

    if _family.get('stitch'):
        seed_scores = load_stitched_seed_scores(ticker, sorted(_family['experiments']))
        if not seed_scores:
            print(f"  No stitched seed scores found across folds")
            del df_ohlcv, atr_standard, atr_adaptive_s, vol_baselines; gc.collect()
            continue
        print(f"  {len(seed_scores)} stitched seed positions loaded")
        base_experiment = sorted(_family['experiments'])[0]
    else:
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

        ohlcv_test = precompute_ohlcv_arrays(df_ohlcv, TEST_START, TEST_END,
                                              vol_baselines, atr_pct_series, vix_lk)

        seed_opt_results = {}

        for seed_id, seed_prob in sorted(seed_scores.items()):
            print(f"\n    Seed {seed_id}:")

            wf_result = optimize_rolling_wf(
                ticker, df_ohlcv, vol_baselines, atr_pct_series,
                vix_lk, regime_dict, seed_prob, seed_id)

            if wf_result is None:
                print(f"      No valid optimization result")
                continue

            opt_result, val_a_result = wf_result
            seed_opt_results[seed_id] = opt_result

            if mode in ('atr', 'atr_adaptive'):
                tp_sl_str = f"tp_m={opt_result.get('tp_mult',0):.1f}  sl_m={opt_result.get('sl_mult',0):.1f}"
            elif mode == 'trailing_decay':
                tp_sl_str = (f"ts={opt_result.get('trail_start_mult',0):.1f}  "
                             f"te={opt_result.get('trail_end_mult',0):.1f}  "
                             f"sl_m={opt_result.get('sl_mult',0):.1f}")
            else:
                tp_sl_str = f"TP={opt_result['tp_pct']:.0%}  SL={opt_result['sl_pct']:.0%}"

            print(f"      OPT:  AvgPnL={opt_result['Avg_PnL']:.2%}  WR={opt_result['Win_Rate']:.1%}  "
                  f"N={opt_result['Count']}  Entry={opt_result['entry_time']}  "
                  f"Prob>{opt_result['prob_thresh']:.3f}  {tp_sl_str}")
            tag_va = "PASS" if val_a_result['Avg_PnL'] > 0 else "FAIL"
            print(f"      VAL:  AvgPnL={val_a_result['Avg_PnL']:.2%}  WR={val_a_result['Win_Rate']:.1%}  "
                  f"N={val_a_result['Count']}  [{tag_va}]")

            test = validate_params(ticker, ohlcv_test, regime_dict, seed_prob, opt_result)
            if test is None:
                test = {'Avg_PnL': 0, 'Total_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}
            tag_t = "PASS" if test['Avg_PnL'] > 0 else "FAIL"
            print(f"      TEST: AvgPnL={test['Avg_PnL']:.2%}  WR={test['Win_Rate']:.1%}  "
                  f"N={test['Count']}  [{tag_t}]  (pure holdout)")

            baseline_params = {**opt_result, 'prob_thresh': 0.30, **NO_FILTER_DEFAULTS, 'vol_filter_pct': 0}
            baseline = validate_params(ticker, ohlcv_test, regime_dict, seed_prob, baseline_params)
            if baseline is None:
                baseline = {'Avg_PnL': 0, 'Count': 0, 'Win_Rate': 0}
            missed_n = max(0, baseline['Count'] - test['Count'])
            if missed_n > 0 and baseline['Count'] > test['Count']:
                missed_pnl = ((baseline['Avg_PnL'] * baseline['Count']
                               - test['Avg_PnL'] * test['Count']) / missed_n)
                print(f"      MISSED: {missed_n} trades not taken "
                      f"(avg {missed_pnl:+.2%}) — baseline prob≥0.30, no filters")
            elif baseline['Count'] == test['Count']:
                print(f"      MISSED: 0 — all baseline trades taken")
            else:
                print(f"      MISSED: baseline={baseline['Count']} trades at prob≥0.30")

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
                'baseline_count': baseline['Count'],
                'baseline_avg_pnl': baseline['Avg_PnL'],
                'missed_count': missed_n,
            }
            all_results_by_mode[mode].append(result_row)
            save_row_to_bq(result_row, RUN_ID, experiment,
                            WF_ROUNDS[0]['opt'][0], WF_ROUNDS[-1]['val'][1],
                            WF_ROUNDS[-1]['val'][0], WF_ROUNDS[-1]['val'][1],
                            TEST_START, TEST_END)

        # ── ENSEMBLE TEST ────────────────────────────────────────────────
        if seed_opt_results:
            all_prob_series = [seed_scores[s] for s in sorted(seed_scores.keys())]
            ensemble_prob = pd.concat(all_prob_series, axis=1).mean(axis=1)

            best_seed_id = max(seed_opt_results, key=lambda s: seed_opt_results[s]['Avg_PnL'])
            ens_params = dict(seed_opt_results[best_seed_id])

            r2 = WF_ROUNDS[-1]
            ohlcv_r2_ens = precompute_ohlcv_arrays(df_ohlcv, r2['opt'][0], r2['opt'][1],
                                                    vol_baselines, atr_pct_series, vix_lk)
            ens_prob_grid = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
            best_pt, best_pt_pnl = ens_params['prob_thresh'], -999.0
            for pt in ens_prob_grid:
                trial = {**ens_params, 'prob_thresh': pt}
                r2_result = validate_params(ticker, ohlcv_r2_ens, regime_dict, ensemble_prob, trial)
                if r2_result is None or r2_result['Count'] < r2['min_trades']:
                    continue
                if r2_result['Avg_PnL'] > best_pt_pnl:
                    best_pt_pnl = r2_result['Avg_PnL']
                    best_pt = pt
            del ohlcv_r2_ens
            if best_pt != ens_params['prob_thresh']:
                print(f"\n      Ensemble prob_thresh re-optimized: "
                      f"{ens_params['prob_thresh']:.2f} → {best_pt:.2f}")
            ens_params['prob_thresh'] = best_pt

            ens_test = validate_params(ticker, ohlcv_test, regime_dict, ensemble_prob, ens_params)
            if ens_test is None:
                ens_test = {'Avg_PnL': 0, 'Total_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}

            ens_tag = "PASS" if ens_test['Avg_PnL'] > 0 else "FAIL"

            if mode in ('atr', 'atr_adaptive'):
                ens_tp_sl = f"tp_m={ens_params.get('tp_mult',0):.1f}  sl_m={ens_params.get('sl_mult',0):.1f}"
            elif mode == 'trailing_decay':
                ens_tp_sl = (f"ts={ens_params.get('trail_start_mult',0):.1f}  "
                             f"te={ens_params.get('trail_end_mult',0):.1f}  "
                             f"sl_m={ens_params.get('sl_mult',0):.1f}")
            else:
                ens_tp_sl = f"TP={ens_params['tp_pct']:.0%}  SL={ens_params['sl_pct']:.0%}"

            print(f"\n    ENSEMBLE ({len(seed_scores)} seeds averaged, "
                  f"params from S{best_seed_id}):")
            print(f"      Params: Entry={ens_params['entry_time']}  "
                  f"Prob>{ens_params['prob_thresh']:.3f}  {ens_tp_sl}")
            print(f"      TEST: AvgPnL={ens_test['Avg_PnL']:.2%}  "
                  f"WR={ens_test['Win_Rate']:.1%}  N={ens_test['Count']}  [{ens_tag}]")

            best_indiv_test = max(
                [r for r in all_results_by_mode[mode] if r['Ticker'] == ticker],
                key=lambda r: r['val_b_avg_pnl'], default=None)
            if best_indiv_test:
                print(f"      vs Best Seed: AvgPnL={best_indiv_test['val_b_avg_pnl']:.2%}  "
                      f"N={best_indiv_test['val_b_count']}")

            ens_row = {
                'Ticker': ticker,
                'param_seed': best_seed_id,
                'n_seeds_avg': len(seed_scores),
                'n_seeds_survived': len(seed_opt_results),
                'entry_time': ens_params['entry_time'],
                'prob_thresh': ens_params['prob_thresh'],
                'tp_pct': ens_params.get('tp_pct', 0),
                'sl_pct': ens_params.get('sl_pct', 0),
                'tp_mult': ens_params.get('tp_mult', 0),
                'sl_mult': ens_params.get('sl_mult', 0),
                'trail_start_mult': ens_params.get('trail_start_mult', 0),
                'trail_end_mult': ens_params.get('trail_end_mult', 0),
                'opt_avg_pnl': ens_params['Avg_PnL'],
                'test_avg_pnl': ens_test['Avg_PnL'],
                'test_win_rate': ens_test['Win_Rate'],
                'test_count': ens_test['Count'],
                'test_pass': ens_test['Avg_PnL'] > 0,
            }
            ensemble_results_by_mode[mode].append(ens_row)

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
                'vix_threshold': ens_params.get('vix_threshold', 999.0),
                'tlt_crash': ens_params.get('tlt_crash', -999.0),
                'spy_ignore': ens_params.get('spy_ignore', 999.0),
                'vol_filter_pct': ens_params.get('vol_filter_pct', 0),
                'Avg_PnL': ens_params['Avg_PnL'],
                'Win_Rate': ens_params['Win_Rate'],
                'Freq': ens_params.get('Freq', 0),
                'Count': ens_params['Count'],
                'val_a_avg_pnl': ens_params['Avg_PnL'],
                'val_a_win_rate': ens_params['Win_Rate'],
                'val_a_freq': ens_params.get('Freq', 0),
                'val_a_count': ens_params['Count'],
                'survived_a': True,
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
        print(f"  ROLLING WF — {mode.upper()} MODE — Results")
        print(f"  {WF_ROUNDS[0]['name']}-{WF_ROUNDS[-1]['name']}: {WF_ROUNDS[0]['opt'][0]} -> {WF_ROUNDS[-1]['val'][1]}  |  TEST: {TEST_START} -> {TEST_END}")
        print(f"{'='*W}")

        if mode in ('atr', 'atr_adaptive'):
            print(f"  {'Ticker':<7} {'Seed':>5} {'Prob':>6} {'Entry':>6} {'tp_m':>5} {'sl_m':>5} "
                  f"{'VIX<':>5} {'TLT>':>6} {'SPY<':>5} "
                  f"{'|':>2} {'OPT PnL':>8} {'OPT WR':>7} {'OPT N':>6} "
                  f"{'|':>2} {'TEST PnL':>9} {'TEST WR':>8} {'TEST N':>7} {'Result':>8}")
        elif mode == 'trailing_decay':
            print(f"  {'Ticker':<7} {'Seed':>5} {'Prob':>6} {'Entry':>6} {'t_s':>5} {'t_e':>5} {'sl_m':>5} "
                  f"{'VIX<':>5} {'TLT>':>6} {'SPY<':>5} "
                  f"{'|':>2} {'OPT PnL':>8} {'OPT WR':>7} {'OPT N':>6} "
                  f"{'|':>2} {'TEST PnL':>9} {'TEST WR':>8} {'TEST N':>7} {'Result':>8}")
        else:
            print(f"  {'Ticker':<7} {'Seed':>5} {'Prob':>6} {'Entry':>6} {'TP':>4} {'SL':>4} "
                  f"{'VIX<':>5} {'TLT>':>6} {'SPY<':>5} "
                  f"{'|':>2} {'OPT PnL':>8} {'OPT WR':>7} {'OPT N':>6} "
                  f"{'|':>2} {'TEST PnL':>9} {'TEST WR':>8} {'TEST N':>7} {'Result':>8}")
        print(f"  {'─'*W}")

        for _, r in best_per_ticker.sort_values('val_b_avg_pnl', ascending=False).iterrows():
            tag = "PASS" if r['survived_b'] else "FAIL"
            if mode in ('atr', 'atr_adaptive'):
                print(f"  {r['Ticker']:<7} {r['Seed']:>5} {r['prob_thresh']:>6.3f} {r['entry_time']:>6} "
                      f"{r['tp_mult']:>5.1f} {r['sl_mult']:>5.1f} "
                      f"{r['vix_threshold']:>5.0f} {r['tlt_crash']:>6.2f} {r['spy_ignore']:>5.2f} "
                      f"{'|':>2} {r['Avg_PnL']:>7.2%} {r['Win_Rate']:>6.1%} {int(r['Count']):>6} "
                      f"{'|':>2} {r['val_b_avg_pnl']:>7.2%} {r['val_b_win_rate']:>6.1%} {int(r['val_b_count']):>6} "
                      f"{'  ' + tag:>8}")
            elif mode == 'trailing_decay':
                print(f"  {r['Ticker']:<7} {r['Seed']:>5} {r['prob_thresh']:>6.3f} {r['entry_time']:>6} "
                      f"{r['trail_start_mult']:>5.1f} {r['trail_end_mult']:>5.1f} {r['sl_mult']:>5.1f} "
                      f"{r['vix_threshold']:>5.0f} {r['tlt_crash']:>6.2f} {r['spy_ignore']:>5.2f} "
                      f"{'|':>2} {r['Avg_PnL']:>7.2%} {r['Win_Rate']:>6.1%} {int(r['Count']):>6} "
                      f"{'|':>2} {r['val_b_avg_pnl']:>7.2%} {r['val_b_win_rate']:>6.1%} {int(r['val_b_count']):>6} "
                      f"{'  ' + tag:>8}")
            else:
                print(f"  {r['Ticker']:<7} {r['Seed']:>5} {r['prob_thresh']:>6.3f} {r['entry_time']:>6} "
                      f"{r['tp_pct']:>3.0%} {r['sl_pct']:>3.0%} "
                      f"{r['vix_threshold']:>5.0f} {r['tlt_crash']:>6.2f} {r['spy_ignore']:>5.2f} "
                      f"{'|':>2} {r['Avg_PnL']:>7.2%} {r['Win_Rate']:>6.1%} {int(r['Count']):>6} "
                      f"{'|':>2} {r['val_b_avg_pnl']:>7.2%} {r['val_b_win_rate']:>6.1%} {int(r['val_b_count']):>6} "
                      f"{'  ' + tag:>8}")

        passed_test = best_per_ticker[best_per_ticker['survived_b']]
        print(f"  {'─'*W}")
        print(f"  {len(passed_test)}/{len(best_per_ticker)} tickers PASSED TEST")
        if len(passed_test) > 0:
            print(f"  PASS avg: OPT={passed_test['Avg_PnL'].mean():.2%} -> "
                  f"TEST={passed_test['val_b_avg_pnl'].mean():.2%}  "
                  f"(degradation={1 - passed_test['val_b_avg_pnl'].mean() / passed_test['Avg_PnL'].mean():.0%})")

        ticker_pass = df.groupby('Ticker').agg(
            seeds=('Seed', 'count'),
            pass_test=('survived_b', 'sum'),
            opt_pnl=('Avg_PnL', 'mean'),
            test_pnl=('val_b_avg_pnl', 'mean'),
        ).sort_values('test_pnl', ascending=False)
        print(f"\n  {'Ticker':<7} {'Seeds':>6} {'PASS-TEST':>10} {'OPT PnL':>9} {'Test PnL':>9} {'Degrad':>8}")
        print(f"  {'─'*55}")
        for ticker, r in ticker_pass.iterrows():
            degrad = 1 - r['test_pnl'] / r['opt_pnl'] if r['opt_pnl'] != 0 else 0
            print(f"  {ticker:<7} {int(r['seeds']):>6} "
                  f"{int(r['pass_test']):>6}/{int(r['seeds'])} "
                  f"{r['opt_pnl']:>+8.2%} "
                  f"{r['test_pnl']:>+8.2%} "
                  f"{degrad:>+7.0%}")

    # ── False Negative Analysis (missed opportunities) ─────────────────────
    print(f"\n\n{'='*W}")
    print(f"  FALSE NEGATIVE ANALYSIS — Missed Opportunities on TEST ({TEST_START} → {TEST_END})")
    print(f"  Baseline: prob≥0.30, no filters, same TP/SL as selected params")
    print(f"{'='*W}")
    for mode in ALL_MODES:
        mode_results = all_results_by_mode[mode]
        if not mode_results:
            continue
        df_fn = pd.DataFrame(mode_results)
        fn_by_ticker = df_fn.groupby('Ticker').agg(
            seeds=('Seed', 'count'),
            test_trades=('val_b_count', 'sum'),
            baseline_trades=('baseline_count', 'sum'),
            missed_trades=('missed_count', 'sum'),
            test_pnl=('val_b_avg_pnl', 'mean'),
            baseline_pnl=('baseline_avg_pnl', 'mean'),
        ).sort_values('missed_trades', ascending=False)
        print(f"\n  {mode.upper()}:")
        print(f"  {'Ticker':<7} {'Seeds':>6} {'Taken':>7} {'Baseline':>9} {'Missed':>7} "
              f"{'Taken PnL':>10} {'Base PnL':>9}")
        print(f"  {'─'*60}")
        for ticker, r in fn_by_ticker.iterrows():
            print(f"  {ticker:<7} {int(r['seeds']):>6} {int(r['test_trades']):>7} "
                  f"{int(r['baseline_trades']):>9} {int(r['missed_trades']):>7} "
                  f"{r['test_pnl']:>+9.2%} {r['baseline_pnl']:>+8.2%}")
        total_taken = int(fn_by_ticker['test_trades'].sum())
        total_missed = int(fn_by_ticker['missed_trades'].sum())
        print(f"  {'─'*60}")
        print(f"  Total: {total_taken} trades taken, {total_missed} missed "
              f"({total_missed / max(1, total_taken + total_missed):.0%} of baseline filtered out)")

    # ── Ensemble Summary ─────────────────────────────────────────────
    ens_total = sum(len(v) for v in ensemble_results_by_mode.values())
    if ens_total > 0:
        print(f"\n\n{'='*W}")
        print(f"  ENSEMBLE RESULTS — Averaged Probabilities Across All Seeds")
        print(f"  TEST: {TEST_START} -> {TEST_END}")
        print(f"{'='*W}")

        for mode in ALL_MODES:
            ens_results = ensemble_results_by_mode[mode]
            if not ens_results:
                continue

            print(f"\n  {mode.upper()}:")
            if mode in ('atr', 'atr_adaptive'):
                print(f"  {'Ticker':<7} {'PSeed':>6} {'#Avg':>5} {'#Surv':>6} "
                      f"{'Entry':>6} {'Prob':>6} {'tp_m':>5} {'sl_m':>5} "
                      f"{'|':>2} {'OPT PnL':>8} "
                      f"{'|':>2} {'TEST PnL':>9} {'TEST WR':>8} {'TEST N':>7} {'Result':>8}")
            elif mode == 'trailing_decay':
                print(f"  {'Ticker':<7} {'PSeed':>6} {'#Avg':>5} {'#Surv':>6} "
                      f"{'Entry':>6} {'Prob':>6} {'t_s':>5} {'t_e':>5} {'sl_m':>5} "
                      f"{'|':>2} {'OPT PnL':>8} "
                      f"{'|':>2} {'TEST PnL':>9} {'TEST WR':>8} {'TEST N':>7} {'Result':>8}")
            else:
                print(f"  {'Ticker':<7} {'PSeed':>6} {'#Avg':>5} {'#Surv':>6} "
                      f"{'Entry':>6} {'Prob':>6} {'TP':>4} {'SL':>4} "
                      f"{'|':>2} {'OPT PnL':>8} "
                      f"{'|':>2} {'TEST PnL':>9} {'TEST WR':>8} {'TEST N':>7} {'Result':>8}")
            print(f"  {'─'*100}")

            for r in sorted(ens_results, key=lambda x: x['test_avg_pnl'], reverse=True):
                tag = "PASS" if r['test_pass'] else "FAIL"
                if mode in ('atr', 'atr_adaptive'):
                    print(f"  {r['Ticker']:<7} {r['param_seed']:>6} {r['n_seeds_avg']:>5} "
                          f"{r['n_seeds_survived']:>6} "
                          f"{r['entry_time']:>6} {r['prob_thresh']:>6.3f} "
                          f"{r['tp_mult']:>5.1f} {r['sl_mult']:>5.1f} "
                          f"{'|':>2} {r['opt_avg_pnl']:>7.2%} "
                          f"{'|':>2} {r['test_avg_pnl']:>7.2%} {r['test_win_rate']:>6.1%} "
                          f"{r['test_count']:>6} {'  ' + tag:>8}")
                elif mode == 'trailing_decay':
                    print(f"  {r['Ticker']:<7} {r['param_seed']:>6} {r['n_seeds_avg']:>5} "
                          f"{r['n_seeds_survived']:>6} "
                          f"{r['entry_time']:>6} {r['prob_thresh']:>6.3f} "
                          f"{r['trail_start_mult']:>5.1f} {r['trail_end_mult']:>5.1f} "
                          f"{r['sl_mult']:>5.1f} "
                          f"{'|':>2} {r['opt_avg_pnl']:>7.2%} "
                          f"{'|':>2} {r['test_avg_pnl']:>7.2%} {r['test_win_rate']:>6.1%} "
                          f"{r['test_count']:>6} {'  ' + tag:>8}")
                else:
                    print(f"  {r['Ticker']:<7} {r['param_seed']:>6} {r['n_seeds_avg']:>5} "
                          f"{r['n_seeds_survived']:>6} "
                          f"{r['entry_time']:>6} {r['prob_thresh']:>6.3f} "
                          f"{r['tp_pct']:>3.0%} {r['sl_pct']:>3.0%} "
                          f"{'|':>2} {r['opt_avg_pnl']:>7.2%} "
                          f"{'|':>2} {r['test_avg_pnl']:>7.2%} {r['test_win_rate']:>6.1%} "
                          f"{r['test_count']:>6} {'  ' + tag:>8}")

            passed = sum(1 for r in ens_results if r['test_pass'])
            print(f"  {'─'*100}")
            print(f"  {passed}/{len(ens_results)} tickers PASSED ensemble TEST")

    print(f"\n\n{'='*W}")
    print(f"  ALL MODES COMPLETE — {grand_total} per-seed results, {ens_total} ensemble results")
    for mode in ALL_MODES:
        n_mode = len(all_results_by_mode[mode])
        n_ens = len(ensemble_results_by_mode[mode])
        if n_mode > 0 or n_ens > 0:
            print(f"    {mode.upper():>15}: {n_mode} per-seed, {n_ens} ensemble")
    print(f"{'='*W}")
