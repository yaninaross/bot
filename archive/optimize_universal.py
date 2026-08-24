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
from polygon import RESTClient
from datetime import datetime, time, timedelta
import pytz

# ==============================================================================
# ⚙️ SINGLE CONFIG — only change this number
# ==============================================================================
EXP = 103

EXP_CONFIGS = {
    87: {
        'model_id':    '87_MICRO',
        'seeds':       list(range(8701, 8709)),
        'scoring_pool': ['NVDA', 'SMCI', 'VRT', 'MARA', 'CORZ', 'IREN', 'WULF', 'APLD',
                         'SMH', 'MOD', 'IWM'],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-03-23',
    },
    88: {
        'model_id':    '88_MICRO',
        'seeds':       list(range(8801, 8809)),
        'scoring_pool': ['NVDA', 'SMCI', 'VRT', 'MARA', 'CORZ', 'IREN', 'WULF', 'APLD',
                         'SMH', 'MOD', 'IWM'],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-03-23',
    },
    89: {
        'model_id':    '89_MICRO',
        'seeds':       list(range(8901, 8909)),
        'scoring_pool': ['MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ',
                         'SMCI', 'NVDA', 'SMH', 'MOD', 'IWM'],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-03-23',
    },
    90: {
        'model_id':    '90_MICRO',
        'seeds':       list(range(9001, 9009)),
        'scoring_pool': ['MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ',
                         'SMCI', 'NVDA', 'SMH', 'MOD', 'IWM'],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-03-23',
    },
    91: {
        'model_id':    '91_MICRO',
        'seeds':       list(range(9101, 9109)),
        'scoring_pool': ['MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ',
                         'SMCI', 'NVDA', 'SMH', 'MOD', 'IWM'],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-03-23',
    },
    92: {
        'model_id':    '92_MICRO',
        'seeds':       list(range(9201, 9209)),
        'scoring_pool': ['MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ',
                         'SMCI', 'NVDA', 'SMH', 'MOD', 'IWM'],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-03-23',
    },
    93: {
        'model_id':    '93_MICRO',
        'seeds':       list(range(9301, 9309)),
        'scoring_pool': ['MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ',
                         'SMCI', 'NVDA', 'SMH', 'MOD', 'IWM'],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-03-23',
    },
    94: {
        'model_id':    '94_MICRO',
        'seeds':       list(range(9401, 9409)),
        'scoring_pool': ['MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ',
                         'SMCI', 'NVDA', 'SMH', 'MOD', 'IWM'],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-03-23',
    },
    95: {
        'model_id':    '95_MICRO',
        'seeds':       list(range(9501, 9509)),
        'scoring_pool': [
            'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ',
            'SMCI', 'NVDA', 'SMH', 'MOD', 'IWM',
            'MSTR', 'COIN',
        ],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-03-23',
    },
    96: {
        'model_id':    '96_MICRO',
        'seeds':       list(range(9601, 9609)),
        'scoring_pool': [
            'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ',
            'SMCI', 'NVDA', 'SMH', 'MOD', 'IWM',
            'MSTR', 'COIN',
        ],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-03-23',
    },
    97: {
        'model_id':    '97_MICRO',
        'seeds':       list(range(9701, 9716)),   # 12 seeds
        'scoring_pool': [
            # exp_92 pool + MSTR and COIN (no other changes)
            'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR', 'COIN',
            'BITF', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'SMLR', 'SMCI', 'NVDA', 'VRT', 'AMD', 'PLTR', 'TSLA',
            'SMH', 'TLT', 'IWM'
        ],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-03-31',
    },
    98: {
        'model_id':    '98_MICRO',
        'seeds':       list(range(9801, 9813)),
        'scoring_pool': [
            # exp_92 pool + MSTR and COIN (no other changes)
            'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR', 'COIN',
            'BITF', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'SMLR', 'SMCI', 'NVDA', 'VRT', 'AMD', 'PLTR', 'TSLA',
            'SMH', 'TLT', 'IWM'
        ],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-03-30',
    },
    100: {
        'model_id':    '100_DEFENSE',
        'seeds':       list(range(10001, 10013)),
        'scoring_pool': [
            'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS', 'LUNR', 'SPIR', 'JOBY', 'NNDM', 'MVIS',
            'GD', 'LMT', 'NOC', 'RTX', 'LHX', 'GE', 'RCAT', 'ACHR'
        ],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-04-01',
    },
    101: {
        'model_id':    '101_DEFENSE',
        'seeds':       list(range(10101, 10113)),
        'scoring_pool': [
            'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS', 'LUNR', 'SPIR', 'JOBY', 'NNDM', 'MVIS',
            'GD', 'LMT', 'NOC', 'RTX', 'LHX', 'GE', 'RCAT', 'ACHR'
        ],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-04-01',
    },
    103: {
        'model_id':    '103_DEFENSE',
        'seeds':       list(range(10301, 10316)),   # 15 seeds
        'scoring_pool': [
            'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS', 'LUNR', 'SPIR', 'JOBY', 'NNDM', 'MVIS',
            'GD', 'LMT', 'NOC', 'RTX', 'LHX', 'GE', 'RCAT', 'ACHR'
        ],
        'eval_start':  '2025-10-02',
        'eval_end':    '2026-04-15',
    },
}

cfg             = EXP_CONFIGS[EXP]
MODEL_ID        = cfg['model_id']
SEEDS           = cfg['seeds']
SCORING_POOL    = cfg['scoring_pool']
EVAL_START_DATE = cfg['eval_start']
EVAL_END_DATE   = cfg['eval_end']
RESULTS_DIR     = f"Algos/Results_all_v2/Universal_Exp{MODEL_ID}"

BUCKET_NAME = "time-series-bucket-ys"
POLYGON_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"

# ==============================================================================
# ⚙️ PROD OPTIMIZATION GRIDS
# ==============================================================================
# Stage 1 — coarse sweep across all parameters
# Entry time: every 5 min across full pre-market window (6:00–6:30 PST)
# TP/SL: wider ranges to avoid missing edge-optimal values
# VIX: regime-meaningful breakpoints rather than uniform spacing
GRID_STRATEGY = {
    'entry_time':  ["06:00", "06:05", "06:10", "06:15", "06:20", "06:25", "06:30"],
    'prob_thresh': np.arange(0.01, 0.99, 0.10).round(2).tolist(),
    'tp_pct':      np.arange(0.10, 0.41, 0.05).round(2).tolist(),   # 7 values, 10%–40%
    'sl_pct':      np.arange(0.02, 0.16, 0.02).round(2).tolist(),   # 7 values, 2%–14%
}
GRID_FILTER = {
    # Market-regime-meaningful VIX breakpoints
    # <15 calm, 15–20 normal, 20–25 elevated, 25–30 stressed, 30–40 fearful, 40+ extreme
    'vix_threshold': [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0],
    # TLT pre-market gap: skip trade when TLT drops more than this % (bonds crashing = risk-off)
    'tlt_crash':     np.arange(0.0, -0.51, -0.10).round(2).tolist(),   # 0 → -0.5
    # SPY pre-market gap: skip trade when SPY gaps UP more than this % (overbought open)
    'spy_ignore':    np.arange(0.0, 0.51, 0.10).round(2).tolist(),    # 0 → 0.5
}

MIN_TRADES_PER_MONTH = 1.5
EXIT_TIME_MIN        = 780   # 1:00 PM PST

# ==============================================================================
# ⚡ SIMULATION ENGINE (Numba) — unchanged, strict 1 trade per day
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

# ==============================================================================
# 📥 DATA LOADING
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

def preload_regime_data():
    """Load VIX, SPY, TLT from master_inference parquets and pre-compute
    regime values using trailing 30-min median ending at each entry minute.
    Matches scoring: if entry is at 06:15, regime = median of 05:45-06:15."""
    print("📥 Preloading Regime Data from master_inference parquets...")
    pst = pytz.timezone('US/Pacific')

    def _load_regime_asset(ticker):
        for suffix in ['_master_inference', '_1m_full_historical_data']:
            uri = f"gs://{BUCKET_NAME}/Data/Inference_v2/{ticker}{suffix}.parquet"
            try:
                return _load_parquet_with_dt_index(uri)
            except Exception:
                continue
        return None

    vix_df = _load_regime_asset('VIX')
    spy_df = _load_regime_asset('SPY')
    tlt_df = _load_regime_asset('TLT')

    if vix_df is None: print("   ⚠️ VIX data missing — using default VIX=15")
    if spy_df is None: print("   ⚠️ SPY data missing — SPY gap will be 0")
    if tlt_df is None: print("   ⚠️ TLT data missing — TLT gap will be 0")

    # Daily closes for gap computation
    spy_daily = spy_df['Close'].resample('D').last().dropna() if spy_df is not None else pd.Series(dtype=float)
    tlt_daily = tlt_df['Close'].resample('D').last().dropna() if tlt_df is not None else pd.Series(dtype=float)

    # Entry minutes from the strategy grid
    entry_minutes = sorted(set(
        int(t.split(':')[0]) * 60 + int(t.split(':')[1])
        for t in GRID_STRATEGY['entry_time']
    ))

    # Build fast (date_int, minute) -> Close lookups for trailing-window median
    eval_start = pd.Timestamp(EVAL_START_DATE, tz=pst) - timedelta(days=10)
    eval_end   = pd.Timestamp(EVAL_END_DATE, tz=pst) + timedelta(days=1)

    def _build_lookup(df):
        """Per-minute Close lookup: (date_int, minute) -> close."""
        if df is None: return {}
        sub = df[(df.index >= eval_start) & (df.index <= eval_end)]
        d_ints = (sub.index.year * 10000 + sub.index.month * 100 + sub.index.day).astype(int)
        m_ints = (sub.index.hour * 60 + sub.index.minute).astype(int)
        closes = sub['Close'].values
        return {(int(d_ints[i]), int(m_ints[i])): float(closes[i])
                for i in range(len(sub))}

    vix_lk = _build_lookup(vix_df)
    spy_lk = _build_lookup(spy_df)
    tlt_lk = _build_lookup(tlt_df)

    def _trailing_median(lookup, d_int, entry_min, window=30):
        """Median Close over the trailing `window` minutes ending at entry_min."""
        vals = []
        for m in range(entry_min - window + 1, entry_min + 1):
            v = lookup.get((d_int, m))
            if v is not None:
                vals.append(v)
        if vals:
            return float(np.median(vals))
        return None

    # Build prev-day-close lookups
    def _prev_close_map(daily_s):
        m = {}
        for i in range(1, len(daily_s)):
            d_int = int(daily_s.index[i].strftime('%Y%m%d'))
            m[d_int] = float(daily_s.iloc[i - 1])
        return m

    spy_prev = _prev_close_map(spy_daily)
    tlt_prev = _prev_close_map(tlt_daily)

    # Build regime_dict keyed by (date_int, entry_minute)
    # Each entry uses trailing 30-min median ending at that entry minute
    all_dates = pd.date_range(eval_start, eval_end, freq='D', tz=pst)
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

    print(f"   ✅ Regime data built for {len(regime_dict)} (date, minute) combos")
    return regime_dict

def load_ticker_ohlcv(ticker):
    clean = ticker.replace("I:", "").replace("X:", "")
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{BUCKET_NAME}/Data/Inference_v2/{clean}{suffix}.parquet"
        try:
            return _load_parquet_with_dt_index(uri)
        except Exception:
            continue
    raise FileNotFoundError(f"No OHLCV data found for {ticker}")

def load_seed_scores(ticker):
    clean  = ticker.replace("I:", "").replace("X:", "")
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
# 🔧 HELPERS — shared across stages
# ==============================================================================
def _attach_regime(raw_dates, regime_dict, n, entry_minute):
    """Build per-trade VIX/TLT/SPY arrays from regime_dict.
    Lookup key is (date_int, entry_minute) to measure regime at entry time."""
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
    """Apply (VIX, TLT, SPY) filter and return filtered PnL array."""
    trap = (trade_vix > vix_t) & (trade_tlt < tlt_c) & (trade_spy > spy_i)
    return raw_pnl[~trap]

# ==============================================================================
# 🔧 OPTIMIZATION CORE — 3-stage, per seed
# ==============================================================================
def precompute_ohlcv_arrays(df_ohlcv):
    """Extract numpy arrays from OHLCV once per ticker to avoid repeated df.copy()."""
    eval_start = pd.to_datetime(EVAL_START_DATE).tz_localize('America/Los_Angeles')
    eval_end   = pd.to_datetime(EVAL_END_DATE).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= eval_start) & (df_ohlcv.index < eval_end)]
    if len(df) == 0:
        return None
    return {
        'index':  df.index,
        'prices': df['Close'].values.copy(),
        'highs':  df['High'].values.copy(),
        'lows':   df['Low'].values.copy(),
        'dates':  (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64),
        'times':  (df.index.hour * 60 + df.index.minute).values,
        'months': max(1.0, (df.index[-1] - df.index[0]).days / 30.44),
    }

def optimize_single_seed(ticker, ohlcv_arrays, regime_dict, seed_prob, seed_id):
    """
    3-stage optimization for a single seed's Prob_Spike signal.

    Stage 1: Coarse sweep — entry_time × prob × tp × sl (strategy)
                           × vix × tlt × spy (regime filters).
    Stage 2: Fine-tune prob_thresh around Stage 1 winner (±0.10, step 0.005).
    Stage 3: Fine-tune tp × sl (strategy) and vix × tlt × spy (filters) jointly
             around Stage 2 winner — the key new step not in the base optimizer.
    """
    if ohlcv_arrays is None:
        return None

    idx = ohlcv_arrays['index']
    common = idx.intersection(seed_prob.index)
    if len(common) == 0:
        return None

    # Build probs array without copying the entire DataFrame
    probs = np.zeros(len(idx), dtype=np.float64)
    common_locs = idx.get_indexer(common)
    probs[common_locs] = seed_prob.loc[common].values

    prices = ohlcv_arrays['prices']
    highs  = ohlcv_arrays['highs']
    lows   = ohlcv_arrays['lows']
    dates  = ohlcv_arrays['dates']
    times  = ohlcv_arrays['times']
    months = ohlcv_arrays['months']

    strat_keys   = list(GRID_STRATEGY.keys())
    strat_combos = list(itertools.product(*GRID_STRATEGY.values()))
    filt_keys    = list(GRID_FILTER.keys())
    filt_combos  = list(itertools.product(*GRID_FILTER.values()))

    # ── STAGE 1: coarse sweep ────────────────────────────────────────────────
    best       = None
    best_score = -999.0

    for s_vals in strat_combos:
        s_cfg  = dict(zip(strat_keys, s_vals))
        h, m_  = map(int, s_cfg['entry_time'].split(':'))
        t_min  = h * 60 + m_

        mask    = (probs >= s_cfg['prob_thresh']) & (times == t_min)
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
                    'Avg_PnL':   avg,
                    'Total_PnL': float(np.sum(pnl)),
                    'Win_Rate':  float(np.mean(pnl > 0)),
                    'Freq':      freq,
                    'Count':     len(pnl),
                    'Stage':     1,
                }

    if best is None:
        return None

    # ── STAGE 2: fine-tune prob_thresh ──────────────────────────────────────
    coarse_prob = best['prob_thresh']
    fine_probs  = np.arange(
        max(0.01, round(coarse_prob - 0.10, 3)),
        min(0.99, round(coarse_prob + 0.101, 3)),
        0.005,
    ).round(3).tolist()

    h2, m2 = map(int, best['entry_time'].split(':'))
    t_min2  = h2 * 60 + m2

    for fp in fine_probs:
        mask2    = (probs >= fp) & (times == t_min2)
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
            best = {**best,
                    'prob_thresh': fp, 'Avg_PnL': avg2,
                    'Total_PnL': float(np.sum(pnl2)),
                    'Win_Rate': float(np.mean(pnl2 > 0)),
                    'Freq': freq2, 'Count': len(pnl2), 'Stage': 2}

    s2_prob = best['prob_thresh']
    if s2_prob != coarse_prob:
        print(f"    ✅ Stage 2: prob {coarse_prob:.2f} → {s2_prob:.3f}  AvgPnL {best['Avg_PnL']:.2%}")
    else:
        print(f"    ── Stage 2: prob {coarse_prob:.2f} already optimal")

    # ── STAGE 3: jointly fine-tune TP, SL + all filters ─────────────────────
    # Build fine grids centred on the Stage 2 winner.
    # tp and sl drive trade sim; vix/tlt/spy are cheap post-hoc masking.
    b_tp  = best['tp_pct'];      b_sl  = best['sl_pct']
    b_vix = best['vix_threshold']; b_tlt = best['tlt_crash']; b_spy = best['spy_ignore']

    fine_tp  = np.arange(max(0.05, b_tp  - 0.05), min(0.50, b_tp  + 0.051), 0.01).round(3)
    fine_sl  = np.arange(max(0.01, b_sl  - 0.02), min(0.20, b_sl  + 0.021), 0.005).round(3)
    fine_vix = np.arange(max(10,   b_vix - 15),   min(100,  b_vix + 16),    2.0).round(1)
    fine_tlt = np.arange(max(-0.5, b_tlt - 0.10), min(0.0,  b_tlt + 0.101), 0.02).round(3)
    fine_spy = np.arange(max(0.0,  b_spy - 0.10), min(0.6,  b_spy + 0.101), 0.02).round(3)

    # Use the best prob and entry_time from Stage 2 (already optimal)
    best_prob = best['prob_thresh']
    mask3     = (probs >= best_prob) & (times == t_min2)

    s3_combos = list(itertools.product(fine_tp, fine_sl))
    f3_combos = list(itertools.product(fine_vix, fine_tlt, fine_spy))

    for tp3, sl3 in s3_combos:
        indices3 = np.where(mask3)[0]
        if len(indices3) == 0: continue

        raw_pnl3, raw_dates3 = run_trade_sim(
            indices3, prices, highs, lows, dates,
            times.astype(np.int64), float(tp3), float(sl3), EXIT_TIME_MIN
        )
        if len(raw_pnl3) == 0: continue

        n3 = len(raw_pnl3)
        tv3, tt3, ts3 = _attach_regime(raw_dates3, regime_dict, n3, t_min2)

        for vix3, tlt3, spy3 in f3_combos:
            pnl3 = _apply_filter(raw_pnl3, tv3, tt3, ts3,
                                  float(vix3), float(tlt3), float(spy3))
            if len(pnl3) == 0: continue

            freq3 = len(pnl3) / months
            if freq3 < MIN_TRADES_PER_MONTH: continue

            avg3 = float(np.mean(pnl3))
            if avg3 > best_score:
                best_score = avg3
                best = {**best,
                        'tp_pct': float(tp3), 'sl_pct': float(sl3),
                        'vix_threshold': float(vix3),
                        'tlt_crash': float(tlt3),
                        'spy_ignore': float(spy3),
                        'Avg_PnL':   avg3,
                        'Total_PnL': float(np.sum(pnl3)),
                        'Win_Rate':  float(np.mean(pnl3 > 0)),
                        'Freq':      freq3,
                        'Count':     len(pnl3),
                        'Stage':     3}

    if best.get('Stage', 1) == 3:
        print(f"    ✅ Stage 3: TP {b_tp:.0%}→{best['tp_pct']:.0%}  "
              f"SL {b_sl:.0%}→{best['sl_pct']:.0%}  "
              f"VIX {b_vix:.0f}→{best['vix_threshold']:.0f}  "
              f"TLT {b_tlt:.2f}→{best['tlt_crash']:.2f}  "
              f"SPY {b_spy:.2f}→{best['spy_ignore']:.2f}  "
              f"AvgPnL {best['Avg_PnL']:.2%}")
    else:
        print(f"    ── Stage 3: Stage 2 params already optimal")

    return best

# ==============================================================================
# 🚀 MAIN LOOP — per-seed optimization, select best seed per ticker
# ==============================================================================
def run_optimization():
    if auth is not None:
        try: auth.authenticate_user()
        except Exception: pass

    n_s1 = len(list(itertools.product(*GRID_STRATEGY.values())))
    n_f1 = len(list(itertools.product(*GRID_FILTER.values())))
    print(f"\n{'='*72}")
    print(f"  EXP {EXP} PROD OPTIMIZATION  |  Model: {MODEL_ID}")
    print(f"  Eval: {EVAL_START_DATE} → {EVAL_END_DATE}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Tickers: {SCORING_POOL}")
    print(f"  Stage 1 combos: {n_s1:,} strategy × {n_f1:,} filter = {n_s1*n_f1:,}")
    print(f"  Stage 2: fine prob (±0.10, step 0.005) with fixed params")
    print(f"  Stage 3: fine TP/SL × fine VIX/TLT/SPY around Stage 2 winner")
    print(f"{'='*72}\n")

    regime_dict = preload_regime_data()

    all_best   = []
    all_detail = []

    for ticker in SCORING_POOL:
        print(f"\n{'─'*62}")
        print(f"  Ticker: {ticker}")
        print(f"{'─'*62}")

        try:
            df_ohlcv = load_ticker_ohlcv(ticker)
        except Exception as e:
            print(f"  ❌ Could not load OHLCV: {e}")
            continue

        ohlcv_arrays = precompute_ohlcv_arrays(df_ohlcv)
        del df_ohlcv; gc.collect()

        if ohlcv_arrays is None:
            print(f"  ⚠️  No data in eval window for {ticker}.")
            continue

        seed_scores = load_seed_scores(ticker)
        if not seed_scores:
            print(f"  ⚠️  No seed scores found for {ticker}.")
            continue
        print(f"  Found {len(seed_scores)} seeds: {sorted(seed_scores.keys())}")

        ticker_seed_results = []

        for seed_id, seed_prob in sorted(seed_scores.items()):
            print(f"\n  → Seed {seed_id}:")
            result = optimize_single_seed(ticker, ohlcv_arrays, regime_dict, seed_prob, seed_id)
            if result is None:
                print(f"    ⚠️  No valid result for seed {seed_id}.")
                continue

            print(f"    ★ Seed {seed_id}  "
                  f"AvgPnL={result['Avg_PnL']:.2%}  "
                  f"WinRate={result['Win_Rate']:.1%}  "
                  f"Freq={result['Freq']:.1f}/mo  "
                  f"Prob>{result['prob_thresh']:.3f}  "
                  f"TP={result['tp_pct']:.0%}  SL={result['sl_pct']:.0%}  "
                  f"VIX<{result['vix_threshold']:.0f}  "
                  f"TLT>{result['tlt_crash']:.2f}  "
                  f"SPY<{result['spy_ignore']:.2f}  "
                  f"[Stage {result.get('Stage',1)}]")

            ticker_seed_results.append(result)
            all_detail.append(result)
            del seed_prob; gc.collect()

        if not ticker_seed_results:
            print(f"  ⚠️  No valid seeds for {ticker}.")
            continue

        winner = max(ticker_seed_results, key=lambda r: r['Avg_PnL'])
        all_best.append(winner)

        # Per-seed comparison table
        print(f"\n  {'─'*50}")
        print(f"  Per-seed comparison for {ticker}:")
        print(f"  {'Seed':>6} {'AvgPnL':>8} {'TotalPnL':>10} {'WinRate':>8} {'Freq':>6} "
              f"{'Prob':>7} {'TP':>5} {'SL':>5} {'VIX':>5} {'TLT':>6} {'SPY':>5} {'Trades':>7}")
        for r in sorted(ticker_seed_results, key=lambda x: x['Avg_PnL'], reverse=True):
            marker = " ◀ WINNER" if r['Seed'] == winner['Seed'] else ""
            print(f"  {r['Seed']:>6} {r['Avg_PnL']:>7.2%} {r['Total_PnL']:>9.2%} "
                  f"{r['Win_Rate']:>7.1%} {r['Freq']:>5.1f}/mo "
                  f"{r['prob_thresh']:>7.3f} {r['tp_pct']:>4.0%} {r['sl_pct']:>4.0%} "
                  f"{r['vix_threshold']:>5.0f} {r['tlt_crash']:>6.2f} {r['spy_ignore']:>5.2f} "
                  f"{int(r['Count']):>7}{marker}")

        del ohlcv_arrays, seed_scores, ticker_seed_results
        gc.collect()

    # ── FINAL SUMMARY TABLE ──────────────────────────────────────────────────
    if not all_best:
        print("\n❌ No results found across all tickers.")
        return

    df_out = pd.DataFrame(all_best).sort_values('Avg_PnL', ascending=False)

    print(f"\n\n{'='*120}")
    print(f"  EXP {EXP} PROD — BEST-SEED SUMMARY")
    print(f"{'='*120}")
    print(f"{'Ticker':<7} {'Seed':>6} {'AvgPnL':>8} {'TotalPnL':>10} {'WinRate':>8} {'Freq':>6} "
          f"{'TP':>5} {'SL':>5} {'Prob':>8} {'Entry':>7} {'VIX<':>6} {'TLT>':>6} {'SPY<':>6} {'N':>5} {'St':>3}")
    print("─" * 120)
    for _, r in df_out.iterrows():
        print(f"{r['Ticker']:<7} {str(r['Seed']):>6} {r['Avg_PnL']:>7.2%} "
              f"{r['Total_PnL']:>9.2%} {r['Win_Rate']:>7.1%} {r['Freq']:>5.1f}/mo "
              f"{r['tp_pct']:>4.0%} {r['sl_pct']:>4.0%} {r['prob_thresh']:>8.3f} "
              f"{r['entry_time']:>7} {r['vix_threshold']:>6.0f} "
              f"{r['tlt_crash']:>6.2f} {r['spy_ignore']:>6.2f} "
              f"{int(r['Count']):>5} {int(r.get('Stage',1)):>3}")
    print("─" * 120)
    print(f"{'MEAN':<7} {'':>6} {df_out['Avg_PnL'].mean():>7.2%} "
          f"{df_out['Total_PnL'].mean():>9.2%} {df_out['Win_Rate'].mean():>7.1%}")
    print(f"{'='*120}\n")

    seed_wins = pd.DataFrame(all_best)['Seed'].value_counts()
    print(f"  Seed win counts across {len(all_best)} tickers:")
    for seed, count in seed_wins.items():
        print(f"    Seed {seed}: {'█' * count} ({count})")
    print()

if __name__ == "__main__":
    run_optimization()
