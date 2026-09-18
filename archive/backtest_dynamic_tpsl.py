"""
backtest_dynamic_tpsl.py — Compare Static vs ATR-Scaled vs Regime-Conditional TP/SL
====================================================================================
Tests 3 TP/SL approaches on ALL deployed tickers using identical entry signals.
Configs loaded from BQ (same source as WF analyzer).

Approaches:
  A. Static     — fixed TP/SL from WF optimization (baseline)
  B. ATR-scaled — TP/SL proportional to 14-day daily ATR at entry
  C. Regime     — 5 VIX regimes, each with independently optimized TP/SL

Optimization objective: avg_pnl × win_rate (N is constant across TP/SL grid).
Evaluation metric: composite = avg_pnl × win_rate × log2(1+N).

Periods (match WF optimizer):
  OPT = Oct 2 – Dec 31, 2025  (fit dynamic params)
  VAL = Jan 2 – Apr 25, 2026  (out-of-sample evaluation)

Run in Colab or any environment with GCS + BQ access.
"""

import numpy as np
import pandas as pd
import pytz
import io
from datetime import timedelta
from itertools import product
from google.cloud import storage, bigquery
import warnings

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

gcs_client = storage.Client()
bq_client  = bigquery.Client()
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
PST = pytz.timezone('America/Los_Angeles')

TICKER_FILE_MAP = {'BITF': 'KEEL'}
TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}

CLEAN_EXP_BASES = ('105_DEFENSE', '106_DEFENSE', '152_XSECTOR', '160_XSECTOR',
                   '170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                   '173_XSECTOR', '174_XSECTOR', '175_XSECTOR')

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
DEPLOY_PNL_THRESHOLD = 0.018
MIN_TRADES = 3

OPT_START = '2025-10-02'
OPT_END   = '2025-12-31'
VAL_START = '2026-01-02'
VAL_END   = '2026-04-25'

EXIT_TIME_MIN = 780

VIX_REGIMES = {
    1: (0, 15),
    2: (15, 20),
    3: (20, 25),
    4: (25, 35),
    5: (35, 999),
}
VIX_LABELS = {1: 'Calm', 2: 'Low', 3: 'Normal', 4: 'Elevated', 5: 'Crisis'}

ATR_TP_MULTS = [1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
ATR_SL_MULTS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

REGIME_TP_GRID = [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50]
REGIME_SL_GRID = [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20]

MIN_REGIME_TRADES = 3


# ==============================================================================
# CONFIG LOADER — from BQ (same logic as analyze_wf_results.py)
# ==============================================================================
def load_deploy_configs():
    print("   Loading WF configs from BQ...")
    query = f"SELECT * FROM `{BQ_TABLE}` ORDER BY experiment, ticker, seed"
    df = bq_client.query(query).to_dataframe()

    df['ticker'] = df['ticker'].replace(TICKER_RENAME)
    df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
    df = df[df['experiment'].apply(lambda e: any(e == b or e.startswith(b + '_') for b in CLEAN_EXP_BASES))]
    df['survived_a'] = df['survived_a'].fillna(False).astype(bool)
    if 'vol_filter_pct' not in df.columns:
        df['vol_filter_pct'] = 0
    df['vol_filter_pct'] = df['vol_filter_pct'].fillna(0).astype(int)

    df['val_a_total_pnl'] = df['val_a_avg_pnl'] * df['val_a_count']
    df['composite'] = df['val_a_avg_pnl'] * df['val_a_win_rate'] * np.sqrt(df['val_a_count']) * np.log2(1 + df['val_a_count'])

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
        eligible = surv[surv['val_a_count'] >= MIN_TRADES]

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
            'seed': int(best_row['seed']),
            'entry_minute': int(best_row['entry_minute']),
            'prob_thresh': float(best_row['prob_thresh']),
            'tp_pct': float(best_row['tp_pct']),
            'sl_pct': float(best_row['sl_pct']),
            'vix_threshold': float(best_row['vix_threshold']),
            'tlt_crash': float(best_row['tlt_crash']),
            'spy_ignore': float(best_row['spy_ignore']),
            'vol_filter_pct': int(best_row['vol_filter_pct']),
            'wf_val_a_n': int(best_row['val_a_count']),
            'wf_val_a_pnl': float(best_row['val_a_avg_pnl']),
        })

    configs.sort(key=lambda c: c['wf_val_a_n'], reverse=True)
    print(f"      {len(configs)} tickers loaded (PnL >= {DEPLOY_PNL_THRESHOLD:.1%})")
    return configs


# ==============================================================================
# DATA LOADING (adapted from optimize_walkforward_153.py)
# ==============================================================================
def _load_parquet_with_dt_index(gcs_uri, tz_out='America/Los_Angeles'):
    df = pd.read_parquet(gcs_uri)
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
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep='first')]
    std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
               'close': 'Close', 'volume': 'Volume'}
    df.columns = [std_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
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


def load_scoring_results(ticker, experiment, seed):
    clean = TICKER_FILE_MAP.get(ticker.replace("I:", "").replace("X:", ""),
                                 ticker.replace("I:", "").replace("X:", ""))
    path = f"Algos/Results_all_v2/Universal_Exp{experiment}/{clean}/Seed_{seed}_Results.csv"
    blob = gcs_client.bucket(BUCKET_NAME).blob(path)
    if not blob.exists():
        raise FileNotFoundError(f"Scoring results not found: {path}")
    df = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('America/Los_Angeles')
    return df


# ==============================================================================
# REGIME DATA (VIX/SPY/TLT — same as WF optimizer)
# ==============================================================================
def preload_regime_data(eval_start, eval_end):
    print("   Loading regime data (VIX, SPY, TLT)...")

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

    pst = pytz.timezone('US/Pacific')
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
        for em in range(358, 393):
            v_val = _trailing_median(vix_lk, d_int, em) or 15.0
            s_gap = 0.0
            sc = _trailing_median(spy_lk, d_int, em)
            sp = spy_prev.get(d_int)
            if sc is not None and sp is not None and sp != 0:
                s_gap = ((sc - sp) / sp) * 100
            t_gap = 0.0
            tc = _trailing_median(tlt_lk, d_int, em)
            tp_ = tlt_prev.get(d_int)
            if tc is not None and tp_ is not None and tp_ != 0:
                t_gap = ((tc - tp_) / tp_) * 100
            regime_dict[(d_int, em)] = np.array([v_val, t_gap, s_gap])

    print(f"      {len(regime_dict)} (date, minute) combos loaded")
    return regime_dict


# ==============================================================================
# ATR COMPUTATION (14-day daily ATR → mapped to minute bars)
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


# ==============================================================================
# TRADE SIMULATION (per-trade TP/SL, pure Python)
# ==============================================================================
def simulate_trades_dynamic(entry_indices, prices, highs, lows, date_ints,
                            time_mins, tp_pcts, sl_pcts, exit_time=EXIT_TIME_MIN):
    n_prices = len(prices)
    results_pnl = []
    results_date = []
    results_exit = []
    last_traded_date = -1

    for ptr in range(len(entry_indices)):
        idx = entry_indices[ptr]
        if idx >= n_prices - 1:
            break

        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            continue

        entry_price = prices[idx]
        tp = tp_pcts[ptr]
        sl = sl_pcts[ptr]
        target_price = entry_price * (1 + tp)
        stop_price   = entry_price * (1 - sl)

        outcome = 0.0
        exit_type = 'EOD'

        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - entry_price) / entry_price
                exit_type = 'EOD'
                break
            if lows[k] <= stop_price:
                outcome = -sl
                exit_type = 'SL'
                break
            if highs[k] >= target_price:
                outcome = tp
                exit_type = 'TP'
                break
            if time_mins[k] >= exit_time:
                outcome = (prices[k] - entry_price) / entry_price
                exit_type = 'EOD'
                break
        else:
            outcome = (prices[-1] - entry_price) / entry_price
            exit_type = 'EOD'

        results_pnl.append(outcome)
        results_date.append(entry_date)
        results_exit.append(exit_type)
        last_traded_date = entry_date

    return (np.array(results_pnl), np.array(results_date),
            np.array(results_exit) if results_exit else np.array([], dtype='U3'))


# ==============================================================================
# ENTRY IDENTIFICATION
# ==============================================================================
def identify_entries(scoring_df, ohlcv_df, regime_dict, config, start_date, end_date):
    eval_start = pd.to_datetime(start_date).tz_localize(PST)
    eval_end   = pd.to_datetime(end_date).tz_localize(PST) + timedelta(days=1)
    df = ohlcv_df[(ohlcv_df.index >= eval_start) & (ohlcv_df.index < eval_end)].copy()
    if len(df) == 0:
        return None

    times = (df.index.hour * 60 + df.index.minute).values
    dates = (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64)

    common = df.index.intersection(scoring_df.index)
    probs = np.zeros(len(df), dtype=np.float64)
    if len(common) > 0:
        common_locs = df.index.get_indexer(common)
        probs[common_locs] = scoring_df.loc[common, 'Prob_Spike'].values

    entry_min = config['entry_minute']
    mask = (probs >= config['prob_thresh']) & (times == entry_min)

    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None

    entry_vix = np.zeros(len(indices))
    entry_tlt = np.zeros(len(indices))
    entry_spy = np.zeros(len(indices))
    for i, idx in enumerate(indices):
        key = (int(dates[idx]), entry_min)
        if key in regime_dict:
            v = regime_dict[key]
            entry_vix[i] = v[0]
            entry_tlt[i] = v[1]
            entry_spy[i] = v[2]

    vix_t = config['vix_threshold']
    tlt_c = config['tlt_crash']
    spy_i = config['spy_ignore']
    trap = (entry_vix > vix_t) & (entry_tlt < tlt_c) & (entry_spy > spy_i)
    keep = ~trap

    indices = indices[keep]
    entry_vix = entry_vix[keep]

    if len(indices) == 0:
        return None

    full_indices = np.searchsorted(ohlcv_df.index.asi8, df.index[indices].asi8)

    return {
        'indices': indices,
        'full_indices': full_indices,
        'prices': df['Close'].values,
        'highs': df['High'].values,
        'lows': df['Low'].values,
        'date_ints': dates,
        'time_mins': times,
        'vix_at_entry': entry_vix,
        'ohlcv_index': df.index,
    }


def classify_vix_regime(vix_val):
    for regime, (lo, hi) in VIX_REGIMES.items():
        if lo <= vix_val < hi:
            return regime
    return 5


# ==============================================================================
# APPROACH A: STATIC
# ==============================================================================
def run_static(entries, config):
    tp_arr = np.full(len(entries['indices']), config['tp_pct'])
    sl_arr = np.full(len(entries['indices']), config['sl_pct'])
    return simulate_trades_dynamic(
        entries['indices'], entries['prices'], entries['highs'], entries['lows'],
        entries['date_ints'], entries['time_mins'], tp_arr, sl_arr)


# ==============================================================================
# APPROACH B: ATR-SCALED
# ==============================================================================
def get_atr_tpsl(entries, atr_pct_series, tp_mult, sl_mult):
    atr_at_entry = atr_pct_series.values[entries['full_indices']]
    atr_at_entry = np.nan_to_num(atr_at_entry, nan=0.05)
    tp_arr = np.clip(atr_at_entry * tp_mult, 0.02, 0.60)
    sl_arr = np.clip(atr_at_entry * sl_mult, 0.01, 0.25)
    return tp_arr, sl_arr


def optimize_atr(entries, atr_pct_series):
    best_score = -999.0
    best_params = (5.0, 2.0)

    for tp_m, sl_m in product(ATR_TP_MULTS, ATR_SL_MULTS):
        tp_arr, sl_arr = get_atr_tpsl(entries, atr_pct_series, tp_m, sl_m)
        pnl, dates, exits = simulate_trades_dynamic(
            entries['indices'], entries['prices'], entries['highs'], entries['lows'],
            entries['date_ints'], entries['time_mins'], tp_arr, sl_arr)
        if len(pnl) < 3:
            continue
        avg = float(np.mean(pnl))
        wr  = float(np.mean(pnl > 0))
        score = avg * wr
        if score > best_score:
            best_score = score
            best_params = (tp_m, sl_m)

    return best_params, best_score


# ==============================================================================
# APPROACH C: REGIME-CONDITIONAL
# ==============================================================================
def get_regime_tpsl(entries, regime_params, default_tp, default_sl):
    regimes = np.array([classify_vix_regime(v) for v in entries['vix_at_entry']])
    tp_arr = np.array([regime_params.get(r, (default_tp, default_sl))[0] for r in regimes])
    sl_arr = np.array([regime_params.get(r, (default_tp, default_sl))[1] for r in regimes])
    return tp_arr, sl_arr, regimes


def optimize_regime(entries, static_tp, static_sl):
    regimes = np.array([classify_vix_regime(v) for v in entries['vix_at_entry']])
    regime_params = {}

    for regime in sorted(VIX_REGIMES.keys()):
        r_mask = regimes == regime
        n_r = r_mask.sum()

        if n_r < MIN_REGIME_TRADES:
            regime_params[regime] = (static_tp, static_sl)
            continue

        r_indices = entries['indices'][r_mask]
        best_score = -999.0
        best_tp_sl = (static_tp, static_sl)

        for tp_val, sl_val in product(REGIME_TP_GRID, REGIME_SL_GRID):
            tp_arr = np.full(len(r_indices), tp_val)
            sl_arr = np.full(len(r_indices), sl_val)
            pnl, _, _ = simulate_trades_dynamic(
                r_indices, entries['prices'], entries['highs'], entries['lows'],
                entries['date_ints'], entries['time_mins'], tp_arr, sl_arr)
            if len(pnl) == 0:
                continue
            avg = float(np.mean(pnl))
            wr  = float(np.mean(pnl > 0))
            score = avg * wr
            if score > best_score:
                best_score = score
                best_tp_sl = (tp_val, sl_val)

        regime_params[regime] = best_tp_sl

    return regime_params


# ==============================================================================
# STATS HELPER
# ==============================================================================
def compute_stats(pnl, exits):
    if len(pnl) == 0:
        return {'n': 0, 'avg_pnl': 0, 'win_rate': 0, 'tp_rate': 0,
                'sl_rate': 0, 'eod_rate': 0, 'total_pnl': 0,
                'median_pnl': 0, 'max_loss': 0, 'max_win': 0,
                'composite': 0}
    n = len(pnl)
    avg = float(np.mean(pnl))
    wr  = float(np.mean(pnl > 0))
    return {
        'n': n,
        'avg_pnl': avg * 100,
        'win_rate': wr * 100,
        'tp_rate': float(np.mean(exits == 'TP')) * 100,
        'sl_rate': float(np.mean(exits == 'SL')) * 100,
        'eod_rate': float(np.mean(exits == 'EOD')) * 100,
        'total_pnl': float(np.sum(pnl)) * 100,
        'median_pnl': float(np.median(pnl)) * 100,
        'max_loss': float(np.min(pnl)) * 100,
        'max_win': float(np.max(pnl)) * 100,
        'composite': avg * wr * np.sqrt(n) * np.log2(1 + n),
    }


# ==============================================================================
# MAIN
# ==============================================================================
print("=" * 110)
print("  DYNAMIC TP/SL COMPARISON v2 — Static vs ATR-Scaled vs Regime-Conditional")
print(f"  OPT: {OPT_START} -> {OPT_END}  |  VAL: {VAL_START} -> {VAL_END}")
print(f"  Optimization objective: avg_pnl × win_rate")
print("=" * 110)

deploy_configs = load_deploy_configs()
regime_dict = preload_regime_data(OPT_START, VAL_END)

print(f"\n  {'#':<4} {'Ticker':<7} {'Experiment':<16} {'Seed':>5} {'Entry':>6} "
      f"{'Prob':>6} {'TP':>5} {'SL':>5} {'WF N':>5} {'WF PnL':>7}")
print(f"  {'─'*80}")
for i, c in enumerate(deploy_configs):
    print(f"  {i+1:<4} {c['ticker']:<7} {c['experiment']:<16} {c['seed']:>5} "
          f"{c['entry_minute']:>6} {c['prob_thresh']:>5.3f} "
          f"{c['tp_pct']:>4.0%} {c['sl_pct']:>4.0%} "
          f"{c['wf_val_a_n']:>5} {c['wf_val_a_pnl']:>6.2%}")

all_results = []
n_total = len(deploy_configs)

for idx, config in enumerate(deploy_configs):
    ticker = config['ticker']
    print(f"\n{'═' * 90}")
    print(f"  [{idx+1}/{n_total}] {ticker} — {config['experiment']} seed {config['seed']}")
    print(f"  Static: TP={config['tp_pct']:.1%}  SL={config['sl_pct']:.1%}  "
          f"entry={config['entry_minute']}  prob>{config['prob_thresh']:.3f}")
    print(f"{'═' * 90}")

    try:
        print(f"   Loading OHLCV...")
        ohlcv_df = load_ticker_ohlcv(ticker)
        print(f"      {len(ohlcv_df):,} rows")

        print(f"   Loading scoring results...")
        scoring_df = load_scoring_results(ticker, config['experiment'], config['seed'])
        print(f"      {len(scoring_df):,} scored bars")

        print(f"   Computing daily ATR...")
        atr_pct = compute_daily_atr_series(ohlcv_df)
        print(f"      ATR_pct mean={atr_pct.dropna().mean():.4f}  "
              f"std={atr_pct.dropna().std():.4f}")

        print(f"   Identifying entries...")
        entries_opt = identify_entries(scoring_df, ohlcv_df, regime_dict, config, OPT_START, OPT_END)
        entries_val = identify_entries(scoring_df, ohlcv_df, regime_dict, config, VAL_START, VAL_END)

        n_opt = len(entries_opt['indices']) if entries_opt else 0
        n_val = len(entries_val['indices']) if entries_val else 0
        print(f"      OPT: {n_opt} entries  |  VAL: {n_val} entries")

        if entries_opt is None or n_opt == 0:
            print(f"   No OPT entries — skipping {ticker}")
            continue

        # ATR at entry stats
        opt_atr = atr_pct.values[entries_opt['full_indices']]
        opt_atr = np.nan_to_num(opt_atr, nan=0.05)
        print(f"      ATR_pct at OPT entries: mean={opt_atr.mean():.4f}  "
              f"range=[{opt_atr.min():.4f}, {opt_atr.max():.4f}]")

        # ── A: STATIC ─────────────────────────────────────────────────
        pnl_s_opt, _, exit_s_opt = run_static(entries_opt, config)
        st_opt = compute_stats(pnl_s_opt, exit_s_opt)
        print(f"\n   [A] STATIC  OPT: {st_opt['n']}t  avg={st_opt['avg_pnl']:+.2f}%  "
              f"WR={st_opt['win_rate']:.0f}%")

        if entries_val:
            pnl_s_val, _, exit_s_val = run_static(entries_val, config)
            st_val = compute_stats(pnl_s_val, exit_s_val)
        else:
            st_val = compute_stats(np.array([]), np.array([]))
        print(f"               VAL: {st_val['n']}t  avg={st_val['avg_pnl']:+.2f}%  "
              f"WR={st_val['win_rate']:.0f}%  comp={st_val['composite']:.6f}")

        # ── B: ATR-SCALED ──────────────────────────────────────────────
        (best_tp_m, best_sl_m), best_atr_score = optimize_atr(entries_opt, atr_pct)

        tp_arr_opt, sl_arr_opt = get_atr_tpsl(entries_opt, atr_pct, best_tp_m, best_sl_m)
        pnl_a_opt, _, exit_a_opt = simulate_trades_dynamic(
            entries_opt['indices'], entries_opt['prices'], entries_opt['highs'],
            entries_opt['lows'], entries_opt['date_ints'], entries_opt['time_mins'],
            tp_arr_opt, sl_arr_opt)
        at_opt = compute_stats(pnl_a_opt, exit_a_opt)

        if entries_val:
            tp_arr_val, sl_arr_val = get_atr_tpsl(entries_val, atr_pct, best_tp_m, best_sl_m)
            pnl_a_val, _, exit_a_val = simulate_trades_dynamic(
                entries_val['indices'], entries_val['prices'], entries_val['highs'],
                entries_val['lows'], entries_val['date_ints'], entries_val['time_mins'],
                tp_arr_val, sl_arr_val)
            at_val = compute_stats(pnl_a_val, exit_a_val)
        else:
            at_val = compute_stats(np.array([]), np.array([]))
        print(f"   [B] ATR     OPT: {at_opt['n']}t  avg={at_opt['avg_pnl']:+.2f}%  "
              f"WR={at_opt['win_rate']:.0f}%  (tp_m={best_tp_m:.1f} sl_m={best_sl_m:.1f})")
        print(f"               VAL: {at_val['n']}t  avg={at_val['avg_pnl']:+.2f}%  "
              f"WR={at_val['win_rate']:.0f}%  comp={at_val['composite']:.6f}")

        # ── C: REGIME-CONDITIONAL ──────────────────────────────────────
        regime_params = optimize_regime(entries_opt, config['tp_pct'], config['sl_pct'])

        tp_r_opt, sl_r_opt, _ = get_regime_tpsl(entries_opt, regime_params, config['tp_pct'], config['sl_pct'])
        pnl_r_opt, _, exit_r_opt = simulate_trades_dynamic(
            entries_opt['indices'], entries_opt['prices'], entries_opt['highs'],
            entries_opt['lows'], entries_opt['date_ints'], entries_opt['time_mins'],
            tp_r_opt, sl_r_opt)
        rg_opt = compute_stats(pnl_r_opt, exit_r_opt)

        if entries_val:
            tp_r_val, sl_r_val, _ = get_regime_tpsl(entries_val, regime_params, config['tp_pct'], config['sl_pct'])
            pnl_r_val, _, exit_r_val = simulate_trades_dynamic(
                entries_val['indices'], entries_val['prices'], entries_val['highs'],
                entries_val['lows'], entries_val['date_ints'], entries_val['time_mins'],
                tp_r_val, sl_r_val)
            rg_val = compute_stats(pnl_r_val, exit_r_val)
        else:
            rg_val = compute_stats(np.array([]), np.array([]))
        print(f"   [C] REGIME  OPT: {rg_opt['n']}t  avg={rg_opt['avg_pnl']:+.2f}%  "
              f"WR={rg_opt['win_rate']:.0f}%")
        print(f"               VAL: {rg_val['n']}t  avg={rg_val['avg_pnl']:+.2f}%  "
              f"WR={rg_val['win_rate']:.0f}%  comp={rg_val['composite']:.6f}")

        all_results.append({
            'ticker': ticker,
            'experiment': config['experiment'],
            'static_opt': st_opt, 'static_val': st_val,
            'atr_opt': at_opt, 'atr_val': at_val,
            'regime_opt': rg_opt, 'regime_val': rg_val,
            'atr_params': (best_tp_m, best_sl_m),
            'regime_params': regime_params,
        })

        del ohlcv_df, scoring_df, entries_opt, entries_val

    except Exception as e:
        print(f"   ERROR: {e}")
        continue


# ==============================================================================
# TABLE 1: VALIDATION COMPARISON (sorted by N descending)
# ==============================================================================
all_results.sort(key=lambda r: r['static_val']['n'], reverse=True)

print(f"\n\n{'═' * 130}")
print(f"  TABLE 1: VALIDATION COMPARISON ({VAL_START} to {VAL_END}) — sorted by N")
print(f"{'═' * 130}")
print(f"  {'Ticker':<7} {'Approach':<12} {'N':>4} {'AvgPnL':>8} {'WinRate':>7} "
      f"{'TotalPnL':>9} {'TP%':>5} {'SL%':>5} {'EOD%':>5} {'Composite':>10}")
print(f"  {'-' * 115}")

for r in all_results:
    for approach, key in [('Static', 'static_val'), ('ATR-scaled', 'atr_val'),
                          ('Regime', 'regime_val')]:
        s = r[key]
        tag = " *" if approach == 'Static' else ""
        print(f"  {r['ticker']:<7} {approach:<12} {s['n']:>4} {s['avg_pnl']:>+7.2f}% "
              f"{s['win_rate']:>6.0f}% {s['total_pnl']:>+8.2f}% "
              f"{s['tp_rate']:>4.0f}% {s['sl_rate']:>4.0f}% {s['eod_rate']:>4.0f}% "
              f"{s['composite']:>10.6f}{tag}")
    print()


# ==============================================================================
# TABLE 2: DELTA vs STATIC (all tickers)
# ==============================================================================
print(f"{'═' * 110}")
print(f"  TABLE 2: DELTA vs STATIC BASELINE (Validation) — sorted by N")
print(f"{'═' * 110}")
print(f"  {'Ticker':<7} {'N':>4} {'Stat PnL':>9} {'ATR PnL':>9} {'ATR Δ':>8} "
      f"{'Rgm PnL':>9} {'Rgm Δ':>8} {'Stat Comp':>10} {'ATR Comp':>10} {'Rgm Comp':>10}")
print(f"  {'-' * 100}")

for r in all_results:
    sv, av, rv = r['static_val'], r['atr_val'], r['regime_val']
    print(f"  {r['ticker']:<7} {sv['n']:>4} "
          f"{sv['avg_pnl']:>+8.2f}% {av['avg_pnl']:>+8.2f}% {av['avg_pnl'] - sv['avg_pnl']:>+7.2f}% "
          f"{rv['avg_pnl']:>+8.2f}% {rv['avg_pnl'] - sv['avg_pnl']:>+7.2f}% "
          f"{sv['composite']:>10.6f} {av['composite']:>10.6f} {rv['composite']:>10.6f}")


# ==============================================================================
# TABLE 3: PARAMETER SUMMARY
# ==============================================================================
print(f"\n\n{'═' * 90}")
print(f"  TABLE 3: FITTED PARAMETERS")
print(f"{'═' * 90}")

for r in all_results:
    tp_m, sl_m = r['atr_params']
    print(f"\n  {r['ticker']} ({r['experiment']}):")
    print(f"    ATR: tp_mult={tp_m:.1f}  sl_mult={sl_m:.1f}")
    r_params = r['regime_params']
    changed = [reg for reg in sorted(r_params) if r_params[reg] != (r['static_val'].get('tp_pct', 0), r['static_val'].get('sl_pct', 0))]
    for reg in sorted(r_params):
        tp_r, sl_r = r_params[reg]
        print(f"    R{reg} ({VIX_LABELS[reg]:<8}): TP={tp_r:.1%}  SL={sl_r:.1%}")


# ==============================================================================
# TABLE 4: OPT → VAL DECAY
# ==============================================================================
print(f"\n\n{'═' * 100}")
print(f"  TABLE 4: OPT → VAL DECAY BY APPROACH")
print(f"{'═' * 100}")
print(f"  {'Ticker':<7} {'Approach':<12} {'OPT PnL':>8} {'VAL PnL':>8} {'Decay':>8}")
print(f"  {'-' * 50}")

for r in all_results:
    for label, opt_k, val_k in [('Static', 'static_opt', 'static_val'),
                                ('ATR-scaled', 'atr_opt', 'atr_val'),
                                ('Regime', 'regime_opt', 'regime_val')]:
        o, v = r[opt_k], r[val_k]
        decay = (o['avg_pnl'] - v['avg_pnl']) if o['avg_pnl'] != 0 else 0
        print(f"  {r['ticker']:<7} {label:<12} {o['avg_pnl']:>+7.2f}% {v['avg_pnl']:>+7.2f}% "
              f"{decay:>+7.2f}%")
    print()


# ==============================================================================
# TABLE 5: HIGH-N SUMMARY (>= 10 VAL trades)
# ==============================================================================
HIGH_N_THRESHOLD = 10
high_n = [r for r in all_results if r['static_val']['n'] >= HIGH_N_THRESHOLD]

print(f"{'═' * 110}")
print(f"  TABLE 5: HIGH-N TICKERS ONLY (N >= {HIGH_N_THRESHOLD} VAL trades) — {len(high_n)} tickers")
print(f"{'═' * 110}")

if high_n:
    print(f"  {'Ticker':<7} {'N':>4} {'Stat PnL':>9} {'ATR PnL':>9} {'ATR Δ':>8} "
          f"{'Rgm PnL':>9} {'Rgm Δ':>8} {'ATR WR':>7} {'Rgm WR':>7}")
    print(f"  {'-' * 85}")

    atr_deltas = []
    rgm_deltas = []
    for r in high_n:
        sv, av, rv = r['static_val'], r['atr_val'], r['regime_val']
        a_delta = av['avg_pnl'] - sv['avg_pnl']
        r_delta = rv['avg_pnl'] - sv['avg_pnl']
        atr_deltas.append(a_delta)
        rgm_deltas.append(r_delta)
        print(f"  {r['ticker']:<7} {sv['n']:>4} "
              f"{sv['avg_pnl']:>+8.2f}% {av['avg_pnl']:>+8.2f}% {a_delta:>+7.2f}% "
              f"{rv['avg_pnl']:>+8.2f}% {r_delta:>+7.2f}% "
              f"{av['win_rate']:>6.0f}% {rv['win_rate']:>6.0f}%")

    print(f"\n  AVERAGE DELTA (N>={HIGH_N_THRESHOLD}):")
    print(f"    ATR vs Static: {np.mean(atr_deltas):+.2f}% avg PnL delta")
    print(f"    Regime vs Static: {np.mean(rgm_deltas):+.2f}% avg PnL delta")
    n_atr_better = sum(1 for d in atr_deltas if d > 0)
    n_rgm_better = sum(1 for d in rgm_deltas if d > 0)
    print(f"    ATR wins: {n_atr_better}/{len(high_n)} tickers")
    print(f"    Regime wins: {n_rgm_better}/{len(high_n)} tickers")
else:
    print(f"  No tickers with N >= {HIGH_N_THRESHOLD} VAL trades.")


# ==============================================================================
# AGGREGATE SUMMARY
# ==============================================================================
print(f"\n\n{'═' * 90}")
print(f"  AGGREGATE SUMMARY ({len(all_results)} tickers)")
print(f"{'═' * 90}")

if all_results:
    all_atr_d = [r['atr_val']['avg_pnl'] - r['static_val']['avg_pnl'] for r in all_results]
    all_rgm_d = [r['regime_val']['avg_pnl'] - r['static_val']['avg_pnl'] for r in all_results]
    all_n = [r['static_val']['n'] for r in all_results]

    print(f"  Total VAL trades (static): {sum(all_n)}")
    print(f"  Mean N per ticker: {np.mean(all_n):.1f}")
    print(f"  Tickers with N >= 10: {sum(1 for n in all_n if n >= 10)}")
    print(f"  Tickers with N >= 5:  {sum(1 for n in all_n if n >= 5)}")
    print(f"\n  ATR vs Static  — mean Δ: {np.mean(all_atr_d):+.2f}%  "
          f"wins: {sum(1 for d in all_atr_d if d > 0)}/{len(all_results)}")
    print(f"  Regime vs Static — mean Δ: {np.mean(all_rgm_d):+.2f}%  "
          f"wins: {sum(1 for d in all_rgm_d if d > 0)}/{len(all_results)}")

    # Weighted by N
    total_n = sum(all_n)
    if total_n > 0:
        w_atr = sum(d * n for d, n in zip(all_atr_d, all_n)) / total_n
        w_rgm = sum(d * n for d, n in zip(all_rgm_d, all_n)) / total_n
        print(f"\n  N-weighted ATR Δ:    {w_atr:+.2f}%")
        print(f"  N-weighted Regime Δ: {w_rgm:+.2f}%")

print(f"\nDONE — backtest_dynamic_tpsl.py v2 complete")
