"""
optimize_walkforward_106.py — Walk-Forward Optimization for Exp-106
====================================================================
Exp-106 = Defense cluster retrain: 13 teachers (5 defense + 5 crypto crossover
+ 3 speculative), 8% spike threshold (lowered from 10% in Exp-105).

Optimizes on Oct 2025 - Dec 2025, validates on TWO windows:
  VAL-A: Jan 2026 - Apr 2026   (standard OOS window)
  VAL-B: Nov 2025 - Apr 2026   (6-month stress test, overlaps OPT)

Same grid, same simulation engine as optimize_walkforward_97/112.
"""

import gc
import pandas as pd
import numpy as np
import io
import itertools
from google.cloud import storage
from numba import jit
try:
    from google.colab import auth
except ImportError:
    auth = None
import pytz
from datetime import timedelta
import uuid
from google.cloud import bigquery as bq

# ==============================================================================
# BIGQUERY — incremental save (one row per seed completion)
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
        SchemaField('model_version', 'STRING'),
        SchemaField('v0_fixed', 'BOOL'),
    ]
    bq_client.create_table(Table(table_id, schema=schema), exists_ok=True)
    # Schema evolution: add any new columns to existing table
    table = bq_client.get_table(table_id)
    existing_fields = {f.name for f in table.schema}
    new_fields = [f for f in schema if f.name not in existing_fields]
    if new_fields:
        table.schema = list(table.schema) + new_fields
        bq_client.update_table(table, ['schema'])
        import time; time.sleep(5)

_bq_client = None
_bq_table_ready = False

def save_row_to_bq(row_dict, run_id, model_id, opt_s, opt_e,
                    va_s, va_e, vb_s=None, vb_e=None):
    global _bq_client, _bq_table_ready
    try:
        if _bq_client is None:
            _bq_client = bq.Client()
        if not _bq_table_ready:
            _ensure_wf_table(_bq_client, BQ_TABLE_ID)
            _bq_table_ready = True
        h_, m_ = map(int, row_dict['entry_time'].split(':'))
        row = {
            'run_id': run_id,
            'run_timestamp': pd.Timestamp.now(tz='UTC').isoformat(),
            'experiment': model_id,
            'ticker': row_dict['Ticker'], 'seed': int(row_dict['Seed']),
            'entry_time': row_dict['entry_time'], 'entry_minute': h_ * 60 + m_,
            'prob_thresh': float(row_dict['prob_thresh']),
            'tp_pct': float(row_dict['tp_pct']),
            'sl_pct': float(row_dict['sl_pct']),
            'vix_threshold': float(row_dict['vix_threshold']),
            'tlt_crash': float(row_dict['tlt_crash']),
            'spy_ignore': float(row_dict['spy_ignore']),
            'min_vol_bars': 0,
            'vol_filter_pct': float(row_dict.get('vol_filter_pct', 0)),
            'opt_avg_pnl': float(row_dict['opt_avg_pnl']),
            'opt_win_rate': float(row_dict['opt_win_rate']),
            'opt_freq': float(row_dict['opt_freq']),
            'opt_count': int(row_dict['opt_count']),
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
            'model_version': 'v2_bar_is_real',
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
MODEL_ID       = '106_DEFENSE'
SEEDS          = list(range(10601, 10616))
SCORING_POOL   = [
    # 13 teachers
    'LUNR', 'RCAT', 'SPIR', 'ASTS', 'BBAI', 'CIFR', 'IREN',
    'HUT', 'MARA', 'KEEL', 'MVIS', 'APLD', 'ACHR',
    # 7 scoring-only
    'CLSK', 'WULF', 'BTBT', 'RIOT', 'MSTR', 'RKLB', 'JOBY',
]
RESULTS_DIR    = f"Algos/Results_all_v2/Universal_Exp{MODEL_ID}"
BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")

TICKER_FILE_MAP = {'BITF': 'KEEL'}

# Walk-forward windows
OPT_START  = '2025-10-02'
OPT_END    = '2025-12-31'

VAL_A_START = '2026-01-02'
VAL_A_END   = '2026-04-25'
VAL_A_LABEL = 'VAL-A (Jan-Apr 2026)'

VAL_B_START = '2025-11-01'
VAL_B_END   = '2026-04-25'
VAL_B_LABEL = 'VAL-B (Nov 2025-Apr 2026)'

# ==============================================================================
# GRIDS — identical to optimize_walkforward_97/112
# ==============================================================================
GRID_STRATEGY = {
    'entry_time':  ["06:00", "06:05", "06:10", "06:15", "06:20", "06:25", "06:30"],
    'prob_thresh': np.arange(0.01, 0.99, 0.10).round(2).tolist(),
    'tp_pct':      np.arange(0.10, 0.41, 0.05).round(2).tolist(),
    'sl_pct':      np.arange(0.02, 0.16, 0.02).round(2).tolist(),
}
GRID_FILTER = {
    'vix_threshold': [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0],
    'tlt_crash':     np.arange(0.0, -0.51, -0.10).round(2).tolist(),
    'spy_ignore':    np.arange(0.0, 0.51, 0.10).round(2).tolist(),
}

MIN_TRADES_PER_MONTH = 2.0
EXIT_TIME_MIN        = 780

VOL_LOOKBACK_DAYS    = 30
VOL_PCT_GRID         = [0, 10, 25, 35]

# ==============================================================================
# SIMULATION ENGINE (Numba)
# ==============================================================================
@jit(nopython=True)
def run_trade_sim(entry_indices, prices, highs, lows, date_ints, time_mins, tp, sl, exit_time):
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
                outcome = tp;  exit_idx = k; trade_done = True; break
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
    if threshold_pct <= 0:
        return np.ones(n, dtype=np.bool_)
    vol_ok = np.ones(n, dtype=np.bool_)
    at_entry = np.where(times == entry_minute)[0]
    thresh_frac = threshold_pct / 100.0
    for idx in at_entry:
        baseline = vol_baselines.get((int(dates[idx]), entry_minute), 0.0)
        if baseline > 0 and volumes[idx] < thresh_frac * baseline:
            vol_ok[idx] = False
    return vol_ok


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
    return regime_dict

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

def load_seed_scores(ticker):
    clean  = TICKER_FILE_MAP.get(ticker.replace("I:", "").replace("X:", ""),
                                  ticker.replace("I:", "").replace("X:", ""))
    client = storage.Client()
    seed_series = {}
    for seed in SEEDS:
        path = f"{RESULTS_DIR}/{clean}/Seed_{seed}_Results.csv"
        blob = client.bucket(BUCKET_NAME).blob(path)
        if not blob.exists():
            continue
        df = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Los_Angeles')
        seed_series[seed] = df['Prob_Spike'].sort_index()
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
    return {
        'index':  df.index,
        'prices': df['Close'].values.copy(),
        'highs':  df['High'].values.copy(),
        'lows':   df['Low'].values.copy(),
        'volumes': df['Volume'].values.copy(),
        'dates':  (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64),
        'times':  (df.index.hour * 60 + df.index.minute).values,
        'months': max(1.0, (df.index[-1] - df.index[0]).days / 30.44),
        'vol_baselines': vol_baselines or {},
    }

# ==============================================================================
# OPTIMIZATION (3-stage, per seed)
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

    prices  = ohlcv_arrays['prices']
    highs   = ohlcv_arrays['highs']
    lows    = ohlcv_arrays['lows']
    volumes = ohlcv_arrays['volumes']
    dates   = ohlcv_arrays['dates']
    times   = ohlcv_arrays['times']
    months  = ohlcv_arrays['months']

    strat_keys   = list(GRID_STRATEGY.keys())
    strat_combos = list(itertools.product(*GRID_STRATEGY.values()))
    filt_keys    = list(GRID_FILTER.keys())
    filt_combos  = list(itertools.product(*GRID_FILTER.values()))

    # Stage 1: coarse sweep
    best       = None
    best_score = -999.0

    for s_vals in strat_combos:
        s_cfg  = dict(zip(strat_keys, s_vals))
        h, m_  = map(int, s_cfg['entry_time'].split(':'))
        t_min  = h * 60 + m_

        mask    = (probs >= s_cfg['prob_thresh']) & (times == t_min) & (volumes > 0)
        indices = np.where(mask)[0]
        if len(indices) == 0: continue

        raw_pnl, raw_dates = run_trade_sim(
            indices, prices, highs, lows, dates,
            times.astype(np.int64), s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN
        )
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
        mask2    = (probs >= fp) & (times == t_min2) & (volumes > 0)
        indices2 = np.where(mask2)[0]
        if len(indices2) == 0: continue
        raw_pnl2, raw_dates2 = run_trade_sim(
            indices2, prices, highs, lows, dates,
            times.astype(np.int64), best['tp_pct'], best['sl_pct'], EXIT_TIME_MIN
        )
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

    # Stage 3: fine-tune TP/SL + filters + entry_time
    b_tp = best['tp_pct']; b_sl = best['sl_pct']
    b_vix = best['vix_threshold']; b_tlt = best['tlt_crash']; b_spy = best['spy_ignore']

    fine_tp  = np.arange(max(0.05, b_tp - 0.05), min(0.50, b_tp + 0.051), 0.01).round(3)
    fine_sl  = np.arange(max(0.01, b_sl - 0.02), min(0.20, b_sl + 0.021), 0.005).round(3)
    fine_vix = np.arange(max(10, b_vix - 15), min(100, b_vix + 16), 2.0).round(1)
    fine_tlt = np.arange(max(-0.5, b_tlt - 0.10), min(0.0, b_tlt + 0.101), 0.02).round(3)
    fine_spy = np.arange(max(0.0, b_spy - 0.10), min(0.6, b_spy + 0.101), 0.02).round(3)

    b_entry_h, b_entry_m = map(int, best['entry_time'].split(':'))
    b_entry_min = b_entry_h * 60 + b_entry_m
    fine_entry = list(range(max(360, b_entry_min - 2), min(391, b_entry_min + 3)))

    best_prob = best['prob_thresh']
    s3_combos = list(itertools.product(fine_tp, fine_sl))
    f3_combos = list(itertools.product(fine_vix, fine_tlt, fine_spy))

    for t_min3 in fine_entry:
        mask3 = (probs >= best_prob) & (times == t_min3) & (volumes > 0)
        for tp3, sl3 in s3_combos:
            indices3 = np.where(mask3)[0]
            if len(indices3) == 0: continue
            raw_pnl3, raw_dates3 = run_trade_sim(
                indices3, prices, highs, lows, dates,
                times.astype(np.int64), float(tp3), float(sl3), EXIT_TIME_MIN
            )
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
                    best = {**best,
                            'entry_time': f"{t_min3//60}:{t_min3%60:02d}",
                            'tp_pct': float(tp3), 'sl_pct': float(sl3),
                            'vix_threshold': float(vix3), 'tlt_crash': float(tlt3),
                            'spy_ignore': float(spy3),
                            'Avg_PnL': avg3, 'Total_PnL': float(np.sum(pnl3)),
                            'Win_Rate': float(np.mean(pnl3 > 0)),
                            'Freq': freq3, 'Count': len(pnl3), 'Stage': 3}

    # ── Stage 4: Volume percentile filter sweep ───────────────────────
    best_entry_h4, best_entry_m4 = map(int, best['entry_time'].split(':'))
    best_t_min4 = best_entry_h4 * 60 + best_entry_m4
    vol_baselines = ohlcv_arrays.get('vol_baselines', {})

    for vpct in VOL_PCT_GRID:
        vol_ok = _compute_vol_pct_ok(ohlcv_arrays['volumes'], times, dates,
                                      vol_baselines, best_t_min4, vpct)
        mask4 = (probs >= best['prob_thresh']) & (times == best_t_min4) & vol_ok & (volumes > 0)
        indices4 = np.where(mask4)[0]
        if len(indices4) == 0:
            continue
        raw_pnl4, raw_dates4 = run_trade_sim(
            indices4, prices, highs, lows, dates,
            times.astype(np.int64), best['tp_pct'], best['sl_pct'], EXIT_TIME_MIN
        )
        if len(raw_pnl4) == 0:
            continue
        n4 = len(raw_pnl4)
        tv4, tt4, ts4 = _attach_regime(raw_dates4, regime_dict, n4, best_t_min4)
        pnl4 = _apply_filter(raw_pnl4, tv4, tt4, ts4,
                              best['vix_threshold'], best['tlt_crash'], best['spy_ignore'])
        if len(pnl4) == 0:
            continue
        freq4 = len(pnl4) / months
        if freq4 < MIN_TRADES_PER_MONTH:
            continue
        avg4 = float(np.mean(pnl4))
        if avg4 > best_score:
            best_score = avg4
            best = {**best, 'vol_filter_pct': vpct,
                    'Avg_PnL': avg4, 'Total_PnL': float(np.sum(pnl4)),
                    'Win_Rate': float(np.mean(pnl4 > 0)),
                    'Freq': freq4, 'Count': len(pnl4), 'Stage': 4}

    if 'vol_filter_pct' not in best:
        best['vol_filter_pct'] = 0

    return best

# ==============================================================================
# VALIDATION — apply optimized params on OOS window
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

    mask = (probs >= params['prob_thresh']) & (ohlcv_val['times'] == t_min) & (ohlcv_val['volumes'] > 0)
    vpct = params.get('vol_filter_pct', 0)
    if vpct > 0:
        vol_ok = _compute_vol_pct_ok(ohlcv_val['volumes'], ohlcv_val['times'],
                                      ohlcv_val['dates'], ohlcv_val.get('vol_baselines', {}),
                                      t_min, vpct)
        mask = mask & vol_ok
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return {'Avg_PnL': 0, 'Total_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}

    raw_pnl, raw_dates = run_trade_sim(
        indices, ohlcv_val['prices'], ohlcv_val['highs'], ohlcv_val['lows'],
        ohlcv_val['dates'], ohlcv_val['times'].astype(np.int64),
        params['tp_pct'], params['sl_pct'], EXIT_TIME_MIN
    )
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
# MAIN
# ==============================================================================
if auth is not None:
    try: auth.authenticate_user()
    except Exception: pass

n_s1 = len(list(itertools.product(*GRID_STRATEGY.values())))
n_f1 = len(list(itertools.product(*GRID_FILTER.values())))
print(f"\n{'='*100}")
print(f"  EXP-106 WALK-FORWARD OPTIMIZATION (dual validation)")
print(f"  Model: {MODEL_ID}")
print(f"  OPTIMIZE:  {OPT_START} -> {OPT_END}")
print(f"  {VAL_A_LABEL}:  {VAL_A_START} -> {VAL_A_END}")
print(f"  {VAL_B_LABEL}:  {VAL_B_START} -> {VAL_B_END}")
print(f"  Seeds: {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} seeds)")
print(f"  Tickers: {len(SCORING_POOL)}")
print(f"  Stage 1 combos: {n_s1:,} strategy x {n_f1:,} filter = {n_s1*n_f1:,}")
print(f"{'='*100}\n")

regime_dict = preload_regime_data(OPT_START, VAL_B_END)

RUN_ID = str(uuid.uuid4())
print(f"  BQ run_id: {RUN_ID[:8]}...")

all_results = []

for ticker in SCORING_POOL:
    print(f"\n{'─'*70}")
    print(f"  {ticker}")
    print(f"{'─'*70}")

    try:
        df_ohlcv = load_ticker_ohlcv(ticker)
    except Exception as e:
        print(f"  Could not load OHLCV: {e}")
        continue

    vol_baselines = precompute_vol_baselines(df_ohlcv)
    ohlcv_opt   = precompute_ohlcv_arrays(df_ohlcv, OPT_START, OPT_END, vol_baselines)
    ohlcv_val_a = precompute_ohlcv_arrays(df_ohlcv, VAL_A_START, VAL_A_END, vol_baselines)
    ohlcv_val_b = precompute_ohlcv_arrays(df_ohlcv, VAL_B_START, VAL_B_END, vol_baselines)
    del df_ohlcv; gc.collect()

    if ohlcv_opt is None:
        print(f"  No data in optimization window")
        continue

    seed_scores = load_seed_scores(ticker)
    if not seed_scores:
        print(f"  No seed scores found")
        continue
    print(f"  {len(seed_scores)} seeds loaded")

    for seed_id, seed_prob in sorted(seed_scores.items()):
        print(f"\n  Seed {seed_id}:")

        opt_result = optimize_single_seed(ticker, ohlcv_opt, regime_dict, seed_prob, seed_id)
        if opt_result is None:
            print(f"    No valid optimization result")
            continue

        print(f"    OPT: AvgPnL={opt_result['Avg_PnL']:.2%}  WR={opt_result['Win_Rate']:.1%}  "
              f"Freq={opt_result['Freq']:.1f}/mo  N={opt_result['Count']}  "
              f"Entry={opt_result['entry_time']}  "
              f"Prob>{opt_result['prob_thresh']:.3f}  TP={opt_result['tp_pct']:.0%}  SL={opt_result['sl_pct']:.0%}  "
              f"VIX<{opt_result['vix_threshold']:.0f}  TLT>{opt_result['tlt_crash']:.2f}  SPY<{opt_result['spy_ignore']:.2f}  "
              f"VF={opt_result.get('vol_filter_pct', 0)}%")

        val_a = validate_params(ticker, ohlcv_val_a, regime_dict, seed_prob, opt_result)
        if val_a is None:
            val_a = {'Avg_PnL': 0, 'Total_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}
        survived_a = val_a['Avg_PnL'] > 0
        tag_a = "PASS" if survived_a else "FAIL"
        print(f"    VAL-A: AvgPnL={val_a['Avg_PnL']:.2%}  WR={val_a['Win_Rate']:.1%}  "
              f"Freq={val_a['Freq']:.1f}/mo  N={val_a['Count']}  [{tag_a}]")

        val_b = validate_params(ticker, ohlcv_val_b, regime_dict, seed_prob, opt_result)
        if val_b is None:
            val_b = {'Avg_PnL': 0, 'Total_PnL': 0, 'Win_Rate': 0, 'Freq': 0, 'Count': 0}
        survived_b = val_b['Avg_PnL'] > 0
        tag_b = "PASS" if survived_b else "FAIL"
        print(f"    VAL-B: AvgPnL={val_b['Avg_PnL']:.2%}  WR={val_b['Win_Rate']:.1%}  "
              f"Freq={val_b['Freq']:.1f}/mo  N={val_b['Count']}  [{tag_b}]")

        result_row = {
            'Ticker': ticker, 'Seed': seed_id,
            'entry_time': opt_result['entry_time'],
            'prob_thresh': opt_result['prob_thresh'],
            'tp_pct': opt_result['tp_pct'], 'sl_pct': opt_result['sl_pct'],
            'vix_threshold': opt_result['vix_threshold'],
            'tlt_crash': opt_result['tlt_crash'],
            'spy_ignore': opt_result['spy_ignore'],
            'vol_filter_pct': opt_result.get('vol_filter_pct', 0),
            'opt_avg_pnl': opt_result['Avg_PnL'],
            'opt_win_rate': opt_result['Win_Rate'],
            'opt_freq': opt_result['Freq'],
            'opt_count': opt_result['Count'],
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
        all_results.append(result_row)
        save_row_to_bq(result_row, RUN_ID, MODEL_ID,
                        OPT_START, OPT_END, VAL_A_START, VAL_A_END,
                        VAL_B_START, VAL_B_END)

        del seed_prob; gc.collect()

    del ohlcv_opt, ohlcv_val_a, ohlcv_val_b, seed_scores, vol_baselines; gc.collect()

# ==============================================================================
# SUMMARY
# ==============================================================================
if not all_results:
    print("\nNo results found.")
else:
    df = pd.DataFrame(all_results)

    best_per_ticker = df.loc[df.groupby('Ticker')['opt_avg_pnl'].idxmax()]

    # -- TABLE 1: VAL-A (Jan-Apr 2026, standard OOS) --
    W = 145
    print(f"\n\n{'='*W}")
    print(f"  EXP-106 WALK-FORWARD — {VAL_A_LABEL}")
    print(f"  Optimize: {OPT_START} -> {OPT_END}  |  Validate: {VAL_A_START} -> {VAL_A_END}")
    print(f"{'='*W}")
    print(f"  {'Ticker':<7} {'Seed':>5} {'Prob':>6} {'Entry':>6} {'TP':>4} {'SL':>4} "
          f"{'VIX<':>5} {'TLT>':>6} {'SPY<':>5} {'VF%':>4} "
          f"{'|':>2} {'OPT PnL':>8} {'OPT WR':>7} {'OPT N':>6} "
          f"{'|':>2} {'VAL PnL':>8} {'VAL WR':>7} {'VAL N':>6} {'Result':>8}")
    print(f"  {'─'*W}")

    for _, r in best_per_ticker.sort_values('val_a_avg_pnl', ascending=False).iterrows():
        tag = "PASS" if r['survived_a'] else "FAIL"
        print(f"  {r['Ticker']:<7} {r['Seed']:>5} {r['prob_thresh']:>6.3f} {r['entry_time']:>6} "
              f"{r['tp_pct']:>3.0%} {r['sl_pct']:>3.0%} "
              f"{r['vix_threshold']:>5.0f} {r['tlt_crash']:>6.2f} {r['spy_ignore']:>5.2f} {int(r.get('vol_filter_pct', 0)):>4} "
              f"{'|':>2} {r['opt_avg_pnl']:>7.2%} {r['opt_win_rate']:>6.1%} {int(r['opt_count']):>6} "
              f"{'|':>2} {r['val_a_avg_pnl']:>7.2%} {r['val_a_win_rate']:>6.1%} {int(r['val_a_count']):>6} "
              f"{'  ' + tag:>8}")

    passed_a = best_per_ticker[best_per_ticker['survived_a']]
    failed_a = best_per_ticker[~best_per_ticker['survived_a']]
    print(f"  {'─'*W}")
    print(f"  {len(passed_a)}/{len(best_per_ticker)} tickers PASSED VAL-A (OOS avg_pnl > 0)")
    if len(passed_a) > 0:
        print(f"  PASS avg OPT PnL: {passed_a['opt_avg_pnl'].mean():.2%}  ->  VAL-A PnL: {passed_a['val_a_avg_pnl'].mean():.2%}  "
              f"(decay: {(1 - passed_a['val_a_avg_pnl'].mean() / passed_a['opt_avg_pnl'].mean()) * 100:.0f}%)")
    if len(failed_a) > 0:
        print(f"  FAIL avg OPT PnL: {failed_a['opt_avg_pnl'].mean():.2%}  ->  VAL-A PnL: {failed_a['val_a_avg_pnl'].mean():.2%}")

    # -- TABLE 2: VAL-B (Nov 2025 - Apr 2026, 6-month stress test) --
    print(f"\n\n{'='*W}")
    print(f"  EXP-106 WALK-FORWARD — {VAL_B_LABEL} (6-month stress test)")
    print(f"  Optimize: {OPT_START} -> {OPT_END}  |  Validate: {VAL_B_START} -> {VAL_B_END}")
    print(f"{'='*W}")
    print(f"  {'Ticker':<7} {'Seed':>5} {'Prob':>6} {'Entry':>6} {'TP':>4} {'SL':>4} "
          f"{'VIX<':>5} {'TLT>':>6} {'SPY<':>5} {'VF%':>4} "
          f"{'|':>2} {'OPT PnL':>8} {'OPT WR':>7} {'OPT N':>6} "
          f"{'|':>2} {'VAL PnL':>8} {'VAL WR':>7} {'VAL N':>6} {'Result':>8}")
    print(f"  {'─'*W}")

    for _, r in best_per_ticker.sort_values('val_b_avg_pnl', ascending=False).iterrows():
        tag = "PASS" if r['survived_b'] else "FAIL"
        print(f"  {r['Ticker']:<7} {r['Seed']:>5} {r['prob_thresh']:>6.3f} {r['entry_time']:>6} "
              f"{r['tp_pct']:>3.0%} {r['sl_pct']:>3.0%} "
              f"{r['vix_threshold']:>5.0f} {r['tlt_crash']:>6.2f} {r['spy_ignore']:>5.2f} {int(r.get('vol_filter_pct', 0)):>4} "
              f"{'|':>2} {r['opt_avg_pnl']:>7.2%} {r['opt_win_rate']:>6.1%} {int(r['opt_count']):>6} "
              f"{'|':>2} {r['val_b_avg_pnl']:>7.2%} {r['val_b_win_rate']:>6.1%} {int(r['val_b_count']):>6} "
              f"{'  ' + tag:>8}")

    passed_b = best_per_ticker[best_per_ticker['survived_b']]
    failed_b = best_per_ticker[~best_per_ticker['survived_b']]
    print(f"  {'─'*W}")
    print(f"  {len(passed_b)}/{len(best_per_ticker)} tickers PASSED VAL-B (OOS avg_pnl > 0)")
    if len(passed_b) > 0:
        print(f"  PASS avg OPT PnL: {passed_b['opt_avg_pnl'].mean():.2%}  ->  VAL-B PnL: {passed_b['val_b_avg_pnl'].mean():.2%}  "
              f"(decay: {(1 - passed_b['val_b_avg_pnl'].mean() / passed_b['opt_avg_pnl'].mean()) * 100:.0f}%)")
    if len(failed_b) > 0:
        print(f"  FAIL avg OPT PnL: {failed_b['opt_avg_pnl'].mean():.2%}  ->  VAL-B PnL: {failed_b['val_b_avg_pnl'].mean():.2%}")

    # -- CROSS-COMPARISON --
    print(f"\n\n{'='*W}")
    print(f"  CROSS-COMPARISON: VAL-A vs VAL-B")
    print(f"{'='*W}")
    both_pass  = best_per_ticker[best_per_ticker['survived_a'] & best_per_ticker['survived_b']]
    a_only     = best_per_ticker[best_per_ticker['survived_a'] & ~best_per_ticker['survived_b']]
    b_only     = best_per_ticker[~best_per_ticker['survived_a'] & best_per_ticker['survived_b']]
    both_fail  = best_per_ticker[~best_per_ticker['survived_a'] & ~best_per_ticker['survived_b']]

    print(f"  Both PASS:    {len(both_pass):>3}  — {', '.join(both_pass['Ticker'].tolist()) or 'none'}")
    print(f"  VAL-A only:   {len(a_only):>3}  — {', '.join(a_only['Ticker'].tolist()) or 'none'}")
    print(f"  VAL-B only:   {len(b_only):>3}  — {', '.join(b_only['Ticker'].tolist()) or 'none'}")
    print(f"  Both FAIL:    {len(both_fail):>3}  — {', '.join(both_fail['Ticker'].tolist()) or 'none'}")

    if len(both_pass) > 0:
        print(f"\n  DEPLOY CANDIDATES (passed both windows):")
        for _, r in both_pass.sort_values('val_b_avg_pnl', ascending=False).iterrows():
            print(f"    {r['Ticker']:<7}  VAL-A={r['val_a_avg_pnl']:.2%}  VAL-B={r['val_b_avg_pnl']:.2%}  "
                  f"Seed={int(r['Seed'])}  Entry={r['entry_time']}  Prob>{r['prob_thresh']:.3f}  "
                  f"TP={r['tp_pct']:.0%}  SL={r['sl_pct']:.0%}")

    # -- ALL SEEDS PER TICKER (PASS count) --
    print(f"\n\n{'='*W}")
    print(f"  PASS RATE PER TICKER (all 15 seeds)")
    print(f"{'='*W}")
    ticker_pass_counts = df.groupby('Ticker').agg(
        seeds_total=('Seed', 'count'),
        pass_a=('survived_a', 'sum'),
        pass_b=('survived_b', 'sum'),
    ).sort_values('pass_a', ascending=False)
    ticker_pass_counts['pass_both'] = df.groupby('Ticker').apply(
        lambda g: (g['survived_a'] & g['survived_b']).sum()
    )
    print(f"  {'Ticker':<7} {'Seeds':>6} {'PASS-A':>7} {'PASS-B':>7} {'PASS-Both':>10}")
    print(f"  {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*10}")
    for ticker, r in ticker_pass_counts.iterrows():
        print(f"  {ticker:<7} {int(r['seeds_total']):>6} "
              f"{int(r['pass_a']):>4}/{int(r['seeds_total'])} "
              f"{int(r['pass_b']):>4}/{int(r['seeds_total'])} "
              f"{int(r['pass_both']):>7}/{int(r['seeds_total'])}")

    print(f"\n{'='*W}")
