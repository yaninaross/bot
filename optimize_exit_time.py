"""
optimize_exit_time.py — Exit Time Optimization (Analysis Only)
==============================================================
Tests 7 exit times (7am-1pm PST) per asset on the TEST holdout period,
using winning configs from the walkforward WF optimization.
No grid search — just varies exit hour with the best config per ticker.

Prints comparison dashboard showing PnL/WR/N for each exit time vs baseline (1pm/780).
Run in Colab after WF optimization completes.
"""

import gc
import math
import pandas as pd
import numpy as np
import io
from google.cloud import storage, bigquery as bq
from numba import jit
try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass
import pytz
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME     = "time-series-bucket-ys"
TICKER_FILE_MAP = {'BITF': 'KEEL'}
TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SPY', 'IWM', 'SMLR', 'SPIR'}

BQ_TABLE_ID     = "ys-time-series.scoring_logs.wf_optimization_results"
MODEL_VERSION   = 'robust_wf_v5'
DEPLOY_THRESHOLD = 0.018
MIN_TRADES      = 3

TEST_START = '2026-04-01'
TEST_END   = '2026-05-21'

SLIPPAGE_PCT = 0.0015

EXIT_TIMES = [420, 480, 540, 600, 660, 720, 780]
EXIT_LABELS = ['7am', '8am', '9am', '10am', '11am', '12pm', '1pm*']

FOLD_EXPERIMENTS = ['170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                    '173_XSECTOR', '174_XSECTOR', '175_XSECTOR']

REGIME_TP_SCALE = np.array([0.70, 0.85, 1.00, 1.30, 1.60], dtype=np.float64)
REGIME_SL_SCALE = np.array([0.60, 0.80, 1.00, 1.30, 1.80], dtype=np.float64)

ALL_TICKERS = [
    'LUNR', 'RCAT', 'ASTS', 'BBAI', 'ACHR', 'MVIS',
    'CIFR', 'IREN', 'HUT', 'KEEL', 'CLSK', 'APLD',
    'WULF', 'MARA', 'RIOT', 'MSTR', 'RKLB', 'JOBY',
    'ALAB', 'BGDE', 'PLUG', 'FCEL', 'BTBT', 'BE',
]

# ==============================================================================
# SIMULATION ENGINES (Numba) — copied from optimize_wf_robust.py
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


def _run_sim(indices, ohlcv_arrays, s_cfg, tpsl_mode, exit_time):
    """Dispatch to mode-specific sim with explicit exit_time and tpsl_mode."""
    prices = ohlcv_arrays['prices']
    highs  = ohlcv_arrays['highs']
    lows   = ohlcv_arrays['lows']
    dates  = ohlcv_arrays['dates']
    times  = ohlcv_arrays['times'].astype(np.int64)

    if tpsl_mode in ('atr', 'atr_adaptive'):
        pnl, d = run_trade_sim_atr(
            indices, prices, highs, lows, dates, times,
            ohlcv_arrays['atr_pct'], s_cfg['tp_mult'], s_cfg['sl_mult'], exit_time)
    elif tpsl_mode == 'trailing_decay':
        pnl, d = run_trade_sim_trailing_decay(
            indices, prices, highs, lows, dates, times,
            ohlcv_arrays['atr_pct'], s_cfg['trail_start_mult'], s_cfg['trail_end_mult'],
            s_cfg['sl_mult'], exit_time)
    elif tpsl_mode == 'regime_scaled':
        pnl, d = run_trade_sim_regime(
            indices, prices, highs, lows, dates, times,
            ohlcv_arrays['vix_regime'], s_cfg['tp_pct'], s_cfg['sl_pct'],
            REGIME_TP_SCALE, REGIME_SL_SCALE, exit_time)
    else:
        pnl, d = run_trade_sim(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], exit_time)
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
    clean = TICKER_FILE_MAP.get(ticker.replace("I:", "").replace("X:", ""),
                                 ticker.replace("I:", "").replace("X:", ""))
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{BUCKET_NAME}/Data/Inference_v2/{clean}{suffix}.parquet"
        try:
            return _load_parquet_with_dt_index(uri)
        except Exception:
            continue
    raise FileNotFoundError(f"No OHLCV data found for {ticker}")


_gcs_client = storage.Client()
_gcs_bucket = _gcs_client.bucket(BUCKET_NAME)

def load_seed_scores(ticker, experiment):
    clean = TICKER_FILE_MAP.get(ticker.replace("I:", "").replace("X:", ""),
                                 ticker.replace("I:", "").replace("X:", ""))
    results_dir = f"Algos/Results_all_v2/Universal_Exp{experiment}"
    prefix = f"{results_dir}/{clean}/Seed_"

    seed_series = {}

    if USE_GCSFS:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        files = fs.glob(f"{BUCKET_NAME}/{prefix}*_Results.csv")
        for fpath in files:
            name = fpath.split('/')[-1]
            if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
                continue
            seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
            with fs.open(fpath, 'rb') as f:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('America/Los_Angeles')
            seed_series[seed_id] = df['Prob_Spike'].sort_index()
        return seed_series

    try:
        blobs = list(_gcs_bucket.list_blobs(prefix=prefix))
    except Exception as e:
        print(f"      GCS list_blobs error for {prefix}: {e}")
        return seed_series

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
# REGIME FILTER
# ==============================================================================
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


# ==============================================================================
# PRECOMPUTE OHLCV ARRAYS
# ==============================================================================
def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date, tpsl_mode,
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
    }

    if atr_pct_series is not None and tpsl_mode in ('atr', 'atr_adaptive', 'trailing_decay'):
        full_indices = np.searchsorted(atr_pct_series.index.asi8, df.index.asi8)
        atr_vals = atr_pct_series.values[np.clip(full_indices, 0, len(atr_pct_series) - 1)]
        result['atr_pct'] = np.nan_to_num(atr_vals, nan=0.05).astype(np.float64)

    if vix_lk is not None and tpsl_mode == 'regime_scaled':
        vix_regime = np.full(len(df), 2, dtype=np.int64)
        for i in range(len(df)):
            key = (int(dates[i]), int(times[i]))
            vix_val = vix_lk.get(key)
            if vix_val is not None:
                vix_regime[i] = classify_vix_regime(vix_val)
        result['vix_regime'] = vix_regime

    return result


# ==============================================================================
# LOAD WINNING CONFIGS FROM BQ
# ==============================================================================
def load_winning_configs():
    """Load best config per ticker from WF optimization results."""
    client = bq.Client()
    query = f"""
        SELECT ticker, tpsl_mode, entry_time, entry_minute, prob_thresh,
               tp_pct, sl_pct, tp_mult, sl_mult,
               trail_start_mult, trail_end_mult,
               vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
               seed, is_ensemble,
               val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq
        FROM `{BQ_TABLE_ID}`
        WHERE model_version = '{MODEL_VERSION}'
          AND experiment LIKE '170%%XSECTOR%%'
    """
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df)} rows from BQ (model_version={MODEL_VERSION})")

    if df.empty:
        print("No data found. Run WF optimizer first.")
        return {}

    df['ticker'] = df['ticker'].replace(TICKER_RENAME)
    df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]

    for col in ['val_b_avg_pnl', 'val_b_win_rate']:
        df[col] = df[col].fillna(0.0)
    df['val_b_count'] = df['val_b_count'].fillna(0).astype(int)
    df['is_ensemble'] = df['is_ensemble'].fillna(False).astype(bool)
    df['tpsl_mode'] = df['tpsl_mode'].fillna('static')
    for col in ['tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
                'tp_pct', 'sl_pct', 'vix_threshold', 'tlt_crash', 'spy_ignore', 'vol_filter_pct']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    df['test_composite'] = df['val_b_avg_pnl'] * df['val_b_win_rate'] * np.log2(1 + df['val_b_count'])

    configs = {}
    for ticker in sorted(df['ticker'].unique()):
        t_df = df[df['ticker'] == ticker]
        eligible = t_df[(t_df['val_b_count'] >= MIN_TRADES) & (t_df['val_b_avg_pnl'] > 0)]
        if eligible.empty:
            continue
        best = eligible.loc[eligible['test_composite'].idxmax()]
        configs[ticker] = {
            'ticker': ticker,
            'tpsl_mode': best['tpsl_mode'],
            'entry_time': best['entry_time'],
            'entry_minute': int(best['entry_minute']),
            'prob_thresh': float(best['prob_thresh']),
            'tp_pct': float(best['tp_pct']),
            'sl_pct': float(best['sl_pct']),
            'tp_mult': float(best['tp_mult']),
            'sl_mult': float(best['sl_mult']),
            'trail_start_mult': float(best['trail_start_mult']),
            'trail_end_mult': float(best['trail_end_mult']),
            'vix_threshold': float(best['vix_threshold']),
            'tlt_crash': float(best['tlt_crash']),
            'spy_ignore': float(best['spy_ignore']),
            'vol_filter_pct': float(best['vol_filter_pct']),
            'seed': int(best['seed']),
            'is_ensemble': bool(best['is_ensemble']),
            'test_pnl': float(best['val_b_avg_pnl']),
            'test_wr': float(best['val_b_win_rate']),
            'test_n': int(best['val_b_count']),
            'test_composite': float(best['test_composite']),
        }
    print(f"Best configs loaded for {len(configs)} tickers")
    return configs


# ==============================================================================
# RUN EXIT TIME SWEEP FOR ONE TICKER
# ==============================================================================
def run_exit_sweep(ticker, cfg, df_ohlcv, seed_scores, regime_dict, vix_lk):
    """Run TEST period backtest at each exit time. Returns dict of exit_time -> stats."""
    mode = cfg['tpsl_mode']
    entry_minute = cfg['entry_minute']
    prob_thresh = cfg['prob_thresh']

    atr_pct_series = None
    if mode in ('atr', 'atr_adaptive', 'trailing_decay'):
        if mode == 'atr_adaptive':
            atr_pct_series = compute_adaptive_atr_series(df_ohlcv)
        else:
            atr_pct_series = compute_daily_atr_series(df_ohlcv)

    ohlcv_arrays = precompute_ohlcv_arrays(
        df_ohlcv, TEST_START, TEST_END, mode,
        atr_pct_series=atr_pct_series, vix_lk=vix_lk)

    if ohlcv_arrays is None:
        print(f"   {ticker}: No OHLCV data in TEST period")
        return {}

    times_arr = ohlcv_arrays['times']
    dates_arr = ohlcv_arrays['dates']

    results = {}
    for exit_time in EXIT_TIMES:
        all_pnl = []
        all_wr = []
        all_n = []

        for pos, score_series in seed_scores.items():
            eval_start_ts = pd.to_datetime(TEST_START).tz_localize('America/Los_Angeles')
            eval_end_ts   = pd.to_datetime(TEST_END).tz_localize('America/Los_Angeles') + timedelta(days=1)
            score_sub = score_series[(score_series.index >= eval_start_ts) &
                                     (score_series.index < eval_end_ts)]

            signal_mask = np.zeros(len(ohlcv_arrays['prices']), dtype=np.bool_)
            idx_arr = ohlcv_arrays['index']
            for ts, prob in score_sub.items():
                bar_min = ts.hour * 60 + ts.minute
                if bar_min != entry_minute:
                    continue
                if prob < prob_thresh:
                    continue
                loc = idx_arr.searchsorted(ts)
                if loc < len(idx_arr) and abs((idx_arr[loc] - ts).total_seconds()) < 120:
                    signal_mask[loc] = True

            entry_indices = np.where(signal_mask)[0].astype(np.int64)
            if len(entry_indices) == 0:
                continue

            pnl, raw_dates = _run_sim(entry_indices, ohlcv_arrays, cfg, mode, exit_time)

            if len(pnl) > 0 and cfg['vix_threshold'] < 900:
                trade_vix, trade_tlt, trade_spy = _attach_regime(
                    raw_dates, regime_dict, len(pnl), entry_minute)
                pnl = _apply_filter(pnl, trade_vix, trade_tlt, trade_spy,
                                    cfg['vix_threshold'], cfg['tlt_crash'], cfg['spy_ignore'])

            if len(pnl) > 0:
                all_pnl.append(float(np.mean(pnl)))
                all_wr.append(float(np.mean(pnl > 0)))
                all_n.append(len(pnl))

        # ensemble: average probs across all seeds
        if len(seed_scores) > 1:
            eval_start_ts = pd.to_datetime(TEST_START).tz_localize('America/Los_Angeles')
            eval_end_ts   = pd.to_datetime(TEST_END).tz_localize('America/Los_Angeles') + timedelta(days=1)
            all_series = []
            for pos, s in seed_scores.items():
                sub = s[(s.index >= eval_start_ts) & (s.index < eval_end_ts)]
                all_series.append(sub)
            if all_series:
                ens_df = pd.concat(all_series, axis=1)
                ens_avg = ens_df.mean(axis=1)

                signal_mask = np.zeros(len(ohlcv_arrays['prices']), dtype=np.bool_)
                idx_arr = ohlcv_arrays['index']
                for ts, prob in ens_avg.items():
                    bar_min = ts.hour * 60 + ts.minute
                    if bar_min != entry_minute:
                        continue
                    if prob < prob_thresh:
                        continue
                    loc = idx_arr.searchsorted(ts)
                    if loc < len(idx_arr) and abs((idx_arr[loc] - ts).total_seconds()) < 120:
                        signal_mask[loc] = True

                entry_indices = np.where(signal_mask)[0].astype(np.int64)
                if len(entry_indices) > 0:
                    pnl, raw_dates = _run_sim(entry_indices, ohlcv_arrays, cfg, mode, exit_time)
                    if len(pnl) > 0 and cfg['vix_threshold'] < 900:
                        trade_vix, trade_tlt, trade_spy = _attach_regime(
                            raw_dates, regime_dict, len(pnl), entry_minute)
                        pnl = _apply_filter(pnl, trade_vix, trade_tlt, trade_spy,
                                            cfg['vix_threshold'], cfg['tlt_crash'], cfg['spy_ignore'])
                    if len(pnl) > 0:
                        all_pnl.append(float(np.mean(pnl)))
                        all_wr.append(float(np.mean(pnl > 0)))
                        all_n.append(len(pnl))

        if all_pnl:
            results[exit_time] = {
                'avg_pnl': float(np.median(all_pnl)),
                'avg_wr': float(np.median(all_wr)),
                'avg_n': int(np.median(all_n)),
            }
        else:
            results[exit_time] = {'avg_pnl': 0.0, 'avg_wr': 0.0, 'avg_n': 0}

    return results


# ==============================================================================
# MAIN — DASHBOARD
# ==============================================================================
print("=" * 120)
print("  EXIT TIME OPTIMIZATION — TEST period ({} to {})".format(TEST_START, TEST_END))
print("=" * 120)

configs = load_winning_configs()
if not configs:
    print("No winning configs found in BQ. Run WF optimizer first.")
    raise SystemExit

test_ticker = list(configs.keys())[0]
test_clean = TICKER_FILE_MAP.get(test_ticker, test_ticker)
test_prefix = f"Algos/Results_all_v2/Universal_Exp170_XSECTOR/{test_clean}/Seed_"
print(f"GCS connectivity test: {test_ticker} at {test_prefix}")
try:
    test_blobs = list(_gcs_bucket.list_blobs(prefix=test_prefix, max_results=3))
    print(f"  list_blobs returned {len(test_blobs)} blobs")
    if test_blobs:
        print(f"  ✅ GCS OK: {[b.name.split('/')[-1] for b in test_blobs]}")
    else:
        print(f"  ⚠️  0 blobs! Trying alternative GCS access...")
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        alt_path = f"{BUCKET_NAME}/{test_prefix}"
        alt_files = fs.glob(f"{alt_path}*_Results.csv")
        print(f"  gcsfs.glob found {len(alt_files)} files")
        if alt_files:
            print(f"  ✅ gcsfs works: {[f.split('/')[-1] for f in alt_files[:3]]}")
            USE_GCSFS = True
        else:
            print(f"  ❌ Both methods failed. Check permissions.")
except Exception as e:
    print(f"  ❌ GCS error: {e}")

USE_GCSFS = False
try:
    test_blobs_check = list(_gcs_bucket.list_blobs(prefix=test_prefix, max_results=1))
    if not test_blobs_check:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        alt_files = fs.glob(f"{BUCKET_NAME}/{test_prefix}*_Results.csv")
        if alt_files:
            USE_GCSFS = True
            print(f"  Switching to gcsfs for seed score loading")
except Exception:
    pass

regime_dict, vix_lk = preload_regime_data(TEST_START, TEST_END)

W = 130
print(f"\n{'=' * W}")
print(f"  {'Ticker':<7} {'Mode':<16} {'Entry':>6} ", end='')
for lbl in EXIT_LABELS:
    print(f"{'':>2}{lbl:>6}", end='')
print(f"  {'Best':>6} {'Delta':>7}")
print(f"  {'─'*7} {'─'*16} {'─'*6} ", end='')
for _ in EXIT_LABELS:
    print(f"{'':>2}{'─'*6}", end='')
print(f"  {'─'*6} {'─'*7}")

summary_rows = []

for ticker in sorted(ALL_TICKERS):
    if ticker not in configs:
        print(f"  {ticker:<7} {'—':^16} (no winning config)")
        continue

    cfg = configs[ticker]
    mode = cfg['tpsl_mode']
    entry_str = cfg['entry_time']

    print(f"  Processing {ticker} ({mode})...", flush=True)

    try:
        df_ohlcv = load_ticker_ohlcv(ticker)
        print(f"     OHLCV: {len(df_ohlcv)} rows", flush=True)
        seed_scores = load_stitched_seed_scores(ticker, FOLD_EXPERIMENTS)
        print(f"     Seeds: {len(seed_scores)} stitched positions", flush=True)
    except Exception as e:
        print(f"SKIP ({e})")
        continue

    if not seed_scores:
        print("SKIP (no seed scores)")
        continue

    results = run_exit_sweep(ticker, cfg, df_ohlcv, seed_scores, regime_dict, vix_lk)

    if not results:
        print("SKIP (no results)")
        continue

    baseline_pnl = results.get(780, {}).get('avg_pnl', 0)
    best_exit = max(results.keys(), key=lambda et: results[et]['avg_pnl'])
    best_pnl = results[best_exit]['avg_pnl']
    delta = best_pnl - baseline_pnl

    best_label = EXIT_LABELS[EXIT_TIMES.index(best_exit)]

    row_str = f"\r  {ticker:<7} {mode:<16} {entry_str:>6} "
    for et in EXIT_TIMES:
        pnl = results.get(et, {}).get('avg_pnl', 0)
        row_str += f"  {pnl:>5.1%}"
    row_str += f"  {best_label:>6} {delta:>+6.1%}"
    print(row_str)

    summary_rows.append({
        'ticker': ticker,
        'mode': mode,
        'entry': entry_str,
        'baseline_pnl': baseline_pnl,
        'best_exit': best_exit,
        'best_label': best_label,
        'best_pnl': best_pnl,
        'delta': delta,
        'results': results,
    })

    del df_ohlcv, seed_scores
    gc.collect()

print(f"\n{'=' * W}")
print(f"  * = current baseline (EXIT_TIME_MIN=780 / 1pm PST)")

# ==============================================================================
# SUMMARY: Candidates for exit time change
# ==============================================================================
candidates = [r for r in summary_rows if r['delta'] > 0.005]
if candidates:
    print(f"\n{'=' * W}")
    print(f"  CANDIDATES FOR EXIT TIME UPDATE (delta > +0.5%)")
    print(f"{'=' * W}")
    print(f"  {'Ticker':<7} {'Mode':<16} {'Current':>9} {'Best Exit':>10} {'Delta':>8}")
    print(f"  {'─'*7} {'─'*16} {'─'*9} {'─'*10} {'─'*8}")
    for r in sorted(candidates, key=lambda x: -x['delta']):
        print(f"  {r['ticker']:<7} {r['mode']:<16} {r['baseline_pnl']:>8.1%} "
              f"{r['best_label']:>10} {r['delta']:>+7.1%}")
else:
    print(f"\n  No tickers benefit from earlier exit (all deltas <= +0.5%)")

# ==============================================================================
# DETAIL TABLE: WR and N per exit time
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  DETAIL: Win Rate by Exit Time")
print(f"{'=' * W}")
print(f"  {'Ticker':<7} ", end='')
for lbl in EXIT_LABELS:
    print(f"{'':>2}{lbl:>6}", end='')
print()
print(f"  {'─'*7} ", end='')
for _ in EXIT_LABELS:
    print(f"{'':>2}{'─'*6}", end='')
print()
for r in summary_rows:
    print(f"  {r['ticker']:<7} ", end='')
    for et in EXIT_TIMES:
        wr = r['results'].get(et, {}).get('avg_wr', 0)
        print(f"  {wr:>5.0%}", end='')
    print()

print(f"\n{'=' * W}")
print(f"  DETAIL: Trade Count by Exit Time")
print(f"{'=' * W}")
print(f"  {'Ticker':<7} ", end='')
for lbl in EXIT_LABELS:
    print(f"{'':>2}{lbl:>6}", end='')
print()
print(f"  {'─'*7} ", end='')
for _ in EXIT_LABELS:
    print(f"{'':>2}{'─'*6}", end='')
print()
for r in summary_rows:
    print(f"  {r['ticker']:<7} ", end='')
    for et in EXIT_TIMES:
        n = r['results'].get(et, {}).get('avg_n', 0)
        print(f"  {n:>6}", end='')
    print()

print(f"\nDone. {len(summary_rows)} tickers analyzed, {len(candidates)} candidates for exit time update.")
