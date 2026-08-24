"""
backtest_ensemble.py — Multi-Seed Ensemble Backtest
=====================================================
Compares 1-seed vs 3-seed vs 5-seed majority-vote strategies.

For each ticker:
  - Picks the best experiment (highest pass rate + median PnL)
  - Selects top N seeds by val_a_avg_pnl
  - Uses best seed's config (entry_minute, tp/sl, prob_thresh, regime filters)
  - Generates signals: 1-seed baseline, 2/3 majority, 3/5 majority
  - Simulates trades with same Numba engine as WF optimizer

Run in Colab:  python backtest_ensemble.py
"""

import io
import gc
import sys
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from numba import jit
from google.cloud import storage, bigquery

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
DEPLOY_PNL_THRESHOLD = 0.018
BT_START = '2026-02-03'
BT_END   = '2026-05-02'
BUCKET_NAME = "time-series-bucket-ys"
EXIT_TIME_MIN = 780
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDED_TICKERS = ['SMLR', 'SPIR']

CLEAN_EXP_BASES = ('105_DEFENSE', '106_DEFENSE', '152_XSECTOR', '160_XSECTOR',
                   '170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                   '173_XSECTOR', '174_XSECTOR', '175_XSECTOR')

MAX_SEEDS = 5
STRATEGIES = {
    '1-seed': (1, 1),
    '3-seed (2/3)': (3, 2),
    '5-seed (3/5)': (5, 3),
}

W = 130

# ==============================================================================
# SIMULATION ENGINE (Numba) — verbatim from WF optimizer
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
# DATA LOADING — from backtest_portfolio.py
# ==============================================================================
gcs_client = storage.Client()


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
        uri = f"gs://{BUCKET_NAME}/Data/Inference_v2/{clean}{suffix}.parquet"
        try:
            return _load_parquet_with_dt_index(uri)
        except Exception:
            continue
    raise FileNotFoundError(f"No OHLCV data found for {ticker}")


def load_single_seed_scores(ticker, experiment, seed):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    results_dir = f"Algos/Results_all_v2/Universal_Exp{experiment}"
    path = f"{results_dir}/{clean}/Seed_{seed}_Results.csv"
    blob = gcs_client.bucket(BUCKET_NAME).blob(path)
    if not blob.exists():
        return None
    df = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('America/Los_Angeles')
    return df['Prob_Spike'].sort_index()


def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date):
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
        'dates':  (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64),
        'times':  (df.index.hour * 60 + df.index.minute).values,
        'months': max(1.0, (df.index[-1] - df.index[0]).days / 30.44),
    }


# ==============================================================================
# REGIME FILTER — from backtest_portfolio.py
# ==============================================================================
def preload_regime_data(eval_start, eval_end):
    print("  Loading regime data (VIX, SPY, TLT)...")
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

    print(f"  Regime data: {len(regime_dict)} (date, minute) combos")
    return regime_dict


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


# ==============================================================================
# STEP 1: LOAD CONFIGS FROM BQ
# ==============================================================================
print(f"{'='*W}")
print(f"  ENSEMBLE BACKTEST — 1 vs 3 vs 5 Seeds ({BT_START} → {BT_END})")
print(f"{'='*W}")

bq_client = bigquery.Client()
query = f"SELECT * FROM `{BQ_TABLE}` ORDER BY experiment, ticker, seed"
try:
    df_bq = bq_client.query(query).to_dataframe()
except Exception as e:
    if 'Not found' in str(e) or 'notFound' in str(e):
        print(f"\n  Table {BQ_TABLE} does not exist yet.")
        raise SystemExit(0)
    raise

if df_bq.empty:
    print("\n  No data in BQ table yet.")
    raise SystemExit(0)

df_bq['survived_a'] = df_bq['survived_a'].fillna(False).astype(bool)
df_bq['run_timestamp'] = pd.to_datetime(df_bq['run_timestamp'], utc=True)
df_bq = df_bq[df_bq['experiment'].apply(lambda e: any(e == b or e.startswith(b + '_') for b in CLEAN_EXP_BASES))]

# Deduplicate: keep latest run_id per experiment
for exp in df_bq['experiment'].unique():
    mask = df_bq['experiment'] == exp
    latest_rid = df_bq.loc[mask].sort_values('run_timestamp').iloc[-1]['run_id']
    df_bq = df_bq[~(mask & (df_bq['run_id'] != latest_rid))]

# Select top seeds per ticker
configs = []

for ticker in sorted(df_bq['ticker'].unique()):
    if ticker in EXCLUDED_TICKERS:
        continue

    td = df_bq[df_bq['ticker'] == ticker]

    exp_pass_rates = td.groupby('experiment').agg(
        pass_a=('survived_a', 'sum'),
        total=('survived_a', 'count'),
    )
    exp_pass_rates['rate'] = exp_pass_rates['pass_a'] / exp_pass_rates['total']

    survived_med = td[td['survived_a']].groupby('experiment')['val_a_avg_pnl'].median()
    exp_pass_rates['med_pnl'] = exp_pass_rates.index.map(
        lambda e: survived_med.get(e, 0.0))

    best_exp = exp_pass_rates.sort_values(['rate', 'med_pnl'], ascending=False).index[0]

    candidates = td[(td['experiment'] == best_exp) & td['survived_a']]
    if len(candidates) == 0:
        continue

    top_seeds = candidates.sort_values('val_a_avg_pnl', ascending=False).head(MAX_SEEDS)
    best_row = top_seeds.iloc[0]

    if best_row['val_a_avg_pnl'] < DEPLOY_PNL_THRESHOLD:
        continue

    seed_list = top_seeds['seed'].astype(int).tolist()

    configs.append({
        'ticker': ticker,
        'experiment': best_row['experiment'],
        'seeds': seed_list,
        'n_seeds': len(seed_list),
        'entry_minute': int(best_row['entry_minute']),
        'prob_thresh': float(best_row['prob_thresh']),
        'tp_pct': float(best_row['tp_pct']),
        'sl_pct': float(best_row['sl_pct']),
        'vix_threshold': float(best_row['vix_threshold']),
        'tlt_crash': float(best_row['tlt_crash']),
        'spy_ignore': float(best_row['spy_ignore']),
        'wf_pnl': float(best_row['val_a_avg_pnl']),
    })

if not configs:
    print("\n  No configs meet deploy threshold.")
    raise SystemExit(0)

print(f"\n  TABLE 1: CONFIGS ({len(configs)} tickers, top {MAX_SEEDS} seeds each)")
print(f"  {'─'*(W-2)}")
print(f"  {'Ticker':<7} {'Experiment':<14} {'Seeds':>5} {'Entry':>6} {'Prob':>6} "
      f"{'TP':>5} {'SL':>5} {'VIX<':>5} {'TLT>':>6} {'SPY<':>5}  {'WF%':>6}  Seed IDs")
print(f"  {'─'*(W-2)}")
for c in configs:
    em = c['entry_minute']
    seed_str = ','.join(str(s) for s in c['seeds'])
    print(f"  {c['ticker']:<7} {c['experiment']:<14} {c['n_seeds']:>5} "
          f"{em//60}:{em%60:02d}  {c['prob_thresh']:>5.3f} "
          f"{c['tp_pct']:>4.0%} {c['sl_pct']:>4.0%} "
          f"{c['vix_threshold']:>5.0f} {c['tlt_crash']:>5.2f} {c['spy_ignore']:>5.2f}  "
          f"{c['wf_pnl']:>5.2%}  {seed_str}")

# ==============================================================================
# STEP 2: LOAD DATA & RUN ENSEMBLE BACKTEST
# ==============================================================================
regime_dict = preload_regime_data(BT_START, BT_END)

# {strategy_name: [trade_dicts]}
strategy_trades = {name: [] for name in STRATEGIES}
ticker_results = []

print(f"\n  Simulating trades for {len(configs)} tickers...")
print(f"  {'─'*(W-2)}")

for cfg in configs:
    ticker = cfg['ticker']
    experiment = cfg['experiment']
    entry_minute = cfg['entry_minute']
    prob_thresh = cfg['prob_thresh']
    tp_pct = cfg['tp_pct']
    sl_pct = cfg['sl_pct']
    vix_threshold = cfg['vix_threshold']
    tlt_crash = cfg['tlt_crash']
    spy_ignore = cfg['spy_ignore']
    all_seeds = cfg['seeds']

    # Load OHLCV
    try:
        df_ohlcv = load_ticker_ohlcv(ticker)
    except Exception as e:
        print(f"  {ticker}: OHLCV failed — {e}")
        continue

    ohlcv = precompute_ohlcv_arrays(df_ohlcv, BT_START, BT_END)
    del df_ohlcv; gc.collect()
    if ohlcv is None:
        print(f"  {ticker}: No OHLCV data in backtest window")
        continue

    # Load score files for all seeds
    seed_probs = {}
    for seed in all_seeds:
        sp = load_single_seed_scores(ticker, experiment, seed)
        if sp is not None:
            seed_probs[seed] = sp

    if not seed_probs:
        print(f"  {ticker}: No score files found")
        continue

    # Align probabilities to OHLCV index
    idx = ohlcv['index']
    n_bars = len(idx)
    ordered_seeds = [s for s in all_seeds if s in seed_probs]
    n_available = len(ordered_seeds)

    prob_matrix = np.zeros((n_bars, n_available), dtype=np.float64)
    for j, seed in enumerate(ordered_seeds):
        sp = seed_probs[seed]
        common = idx.intersection(sp.index)
        if len(common) > 0:
            locs = idx.get_indexer(common)
            prob_matrix[locs, j] = sp.loc[common].values

    # Generate signals for each strategy
    times = ohlcv['times']
    at_entry = times == entry_minute

    ticker_row = {'ticker': ticker, 'experiment': experiment, 'n_seeds': n_available}

    for strat_name, (n_needed, min_agree) in STRATEGIES.items():
        if n_available < n_needed:
            ticker_row[f'{strat_name}_trades'] = -1
            ticker_row[f'{strat_name}_wr'] = 0
            ticker_row[f'{strat_name}_pnl'] = 0
            continue

        # Count how many of the top N seeds agree
        top_n_probs = prob_matrix[:, :n_needed]
        n_agree = np.sum(top_n_probs >= prob_thresh, axis=1)
        signal = (n_agree >= min_agree) & at_entry
        indices = np.where(signal)[0]

        if len(indices) == 0:
            ticker_row[f'{strat_name}_trades'] = 0
            ticker_row[f'{strat_name}_wr'] = 0
            ticker_row[f'{strat_name}_pnl'] = 0
            continue

        raw_pnl, raw_dates = run_trade_sim(
            indices, ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
            ohlcv['dates'], times.astype(np.int64),
            tp_pct, sl_pct, EXIT_TIME_MIN
        )

        if len(raw_pnl) == 0:
            ticker_row[f'{strat_name}_trades'] = 0
            ticker_row[f'{strat_name}_wr'] = 0
            ticker_row[f'{strat_name}_pnl'] = 0
            continue

        # Apply regime filter
        n = len(raw_pnl)
        trade_vix, trade_tlt, trade_spy = _attach_regime(raw_dates, regime_dict, n, entry_minute)
        trap = (trade_vix > vix_threshold) & (trade_tlt < tlt_crash) & (trade_spy > spy_ignore)
        kept = ~trap

        final_pnl = raw_pnl[kept]
        final_dates = raw_dates[kept]

        for i in range(len(final_pnl)):
            strategy_trades[strat_name].append({
                'ticker': ticker,
                'experiment': experiment,
                'date_int': int(final_dates[i]),
                'pnl': float(final_pnl[i]),
            })

        n_trades = len(final_pnl)
        wins = int(np.sum(final_pnl > 0))
        wr = wins / n_trades if n_trades > 0 else 0
        avg = float(np.mean(final_pnl)) if n_trades > 0 else 0

        ticker_row[f'{strat_name}_trades'] = n_trades
        ticker_row[f'{strat_name}_wr'] = wr
        ticker_row[f'{strat_name}_pnl'] = avg

    ticker_results.append(ticker_row)

    # Print compact progress
    parts = []
    for sn in STRATEGIES:
        t = ticker_row.get(f'{sn}_trades', -1)
        if t < 0:
            parts.append(f"{sn}:N/A")
        else:
            p = ticker_row.get(f'{sn}_pnl', 0)
            parts.append(f"{sn}:{t}t/{p:+.1%}")
    print(f"  {ticker:<7} {n_available} seeds  {'  '.join(parts)}")

    del ohlcv, seed_probs, prob_matrix; gc.collect()


# ==============================================================================
# STEP 3: RESULTS
# ==============================================================================
if not ticker_results:
    print("\n  No trades simulated.")
    raise SystemExit(0)

# --- TABLE 2: PER-TICKER COMPARISON ---
print(f"\n\n{'='*W}")
print(f"  TABLE 2: PER-TICKER COMPARISON")
print(f"{'='*W}")

hdr = f"  {'Ticker':<7} {'Exp':<14} {'#S':>2}"
for sn in STRATEGIES:
    hdr += f"  {'Tr':>3} {'WR':>5} {'PnL':>7}"
print(hdr)

sub_hdr = f"  {'─'*7} {'─'*14} {'─'*2}"
for _ in STRATEGIES:
    sub_hdr += f"  {'─'*3} {'─'*5} {'─'*7}"
print(sub_hdr)

# Strategy column headers
col_hdr = f"  {'':7} {'':14} {'':2}"
for sn in STRATEGIES:
    col_hdr += f"  {'--- ' + sn + ' ---':>15}"
print(col_hdr)

for row in ticker_results:
    line = f"  {row['ticker']:<7} {row['experiment']:<14} {row['n_seeds']:>2}"
    for sn in STRATEGIES:
        t = row.get(f'{sn}_trades', -1)
        if t < 0:
            line += f"  {'--':>3} {'--':>5} {'--':>7}"
        elif t == 0:
            line += f"  {'0':>3} {'--':>5} {'--':>7}"
        else:
            wr = row.get(f'{sn}_wr', 0)
            pnl = row.get(f'{sn}_pnl', 0)
            line += f"  {t:>3} {wr:>4.0%} {pnl:>+6.2%}"
    print(line)

# --- TABLE 3: PORTFOLIO SUMMARY ---
print(f"\n\n{'='*W}")
print(f"  TABLE 3: PORTFOLIO SUMMARY  ({BT_START} → {BT_END})")
print(f"{'='*W}")
print(f"  {'Metric':<22}", end="")
for sn in STRATEGIES:
    print(f"  {sn:>15}", end="")
print()
print(f"  {'─'*22}", end="")
for _ in STRATEGIES:
    print(f"  {'─'*15}", end="")
print()

summaries = {}
for sn in STRATEGIES:
    trades = strategy_trades[sn]
    if not trades:
        summaries[sn] = None
        continue

    tdf = pd.DataFrame(trades)
    tdf['date'] = pd.to_datetime(tdf['date_int'].astype(str), format='%Y%m%d')

    n_trades = len(tdf)
    wins = int((tdf['pnl'] > 0).sum())
    losses = n_trades - wins
    wr = wins / n_trades if n_trades > 0 else 0
    avg = tdf['pnl'].mean()
    med = tdf['pnl'].median()
    total = tdf['pnl'].sum()
    months = max(1.0, (tdf['date'].max() - tdf['date'].min()).days / 30.44)
    freq = n_trades / months
    n_tickers = tdf['ticker'].nunique()

    daily_pnl = tdf.groupby('date')['pnl'].sum().sort_index()
    cum = daily_pnl.cumsum()
    max_dd = (cum - cum.cummax()).min()

    summaries[sn] = {
        'trades': n_trades, 'wins': wins, 'losses': losses,
        'wr': wr, 'avg': avg, 'median': med, 'total': total,
        'freq': freq, 'n_tickers': n_tickers, 'max_dd': max_dd,
        'daily_pnl': daily_pnl, 'cum_pnl': cum,
    }

metrics = [
    ('Tickers', 'n_tickers', 'd'),
    ('Total trades', 'trades', 'd'),
    ('Wins / Losses', None, 'wl'),
    ('Win rate', 'wr', '%'),
    ('Avg PnL/trade', 'avg', 'pct'),
    ('Median PnL/trade', 'median', 'pct'),
    ('Total PnL (sum)', 'total', 'pct'),
    ('Max drawdown', 'max_dd', 'pct'),
    ('Trades/month', 'freq', 'f'),
]

for label, key, fmt in metrics:
    line = f"  {label:<22}"
    for sn in STRATEGIES:
        s = summaries.get(sn)
        if s is None:
            line += f"  {'N/A':>15}"
        elif fmt == 'wl':
            line += f"  {s['wins']:>6} / {s['losses']:<6}"
        elif fmt == 'd':
            line += f"  {s[key]:>15d}"
        elif fmt == '%':
            line += f"  {s[key]:>14.1%}"
        elif fmt == 'pct':
            line += f"  {s[key]:>+14.2%}"
        elif fmt == 'f':
            line += f"  {s[key]:>14.1f}"
    print(line)

# --- Improvement summary ---
print(f"\n  {'─'*(W-2)}")
s1 = summaries.get('1-seed')
for sn in list(STRATEGIES.keys())[1:]:
    s = summaries.get(sn)
    if s1 and s:
        delta_wr = s['wr'] - s1['wr']
        delta_pnl = s['avg'] - s1['avg']
        delta_trades = s['trades'] - s1['trades']
        print(f"  {sn} vs 1-seed:  "
              f"WR {delta_wr:+.1%}  "
              f"AvgPnL {delta_pnl:+.2%}  "
              f"Trades {delta_trades:+d}")


# ==============================================================================
# STEP 4: CHARTS
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f'Ensemble Backtest — 1 vs 3 vs 5 Seeds ({BT_START} → {BT_END})',
             fontsize=14, fontweight='bold')

colors = {'1-seed': '#2196F3', '3-seed (2/3)': '#FF9800', '5-seed (3/5)': '#4CAF50'}

# Chart 1: Cumulative PnL
ax1 = axes[0]
for sn in STRATEGIES:
    s = summaries.get(sn)
    if s is None:
        continue
    cum = s['cum_pnl']
    ax1.plot(cum.index, cum.values * 100, color=colors[sn], linewidth=2, label=sn)

ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax1.set_title('Cumulative Portfolio PnL', fontsize=12)
ax1.set_ylabel('Cumulative PnL (%)')
ax1.legend(fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Chart 2: Per-ticker PnL comparison (grouped bars)
ax2 = axes[1]
tickers_with_data = [r['ticker'] for r in ticker_results
                     if r.get('1-seed_trades', 0) > 0]
x = np.arange(len(tickers_with_data))
bar_width = 0.25

for i, sn in enumerate(STRATEGIES):
    vals = []
    for t in tickers_with_data:
        row = next(r for r in ticker_results if r['ticker'] == t)
        n_t = row.get(f'{sn}_trades', 0)
        vals.append(row.get(f'{sn}_pnl', 0) * 100 if n_t > 0 else 0)
    offset = (i - 1) * bar_width
    ax2.bar(x + offset, vals, bar_width, label=sn, color=colors[sn], alpha=0.8)

ax2.set_title('Avg PnL per Trade by Ticker', fontsize=12)
ax2.set_ylabel('Avg PnL (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(tickers_with_data, rotation=45, ha='right')
ax2.legend(fontsize=11)
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
chart_path = '/tmp/backtest_ensemble_charts.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Charts saved: {chart_path}")
try:
    from IPython.display import display, Image
    display(Image(filename=chart_path))
except ImportError:
    plt.show()

print(f"\n{'='*W}")
print(f"  DONE")
print(f"{'='*W}")
