"""
backtest_portfolio_trailing.py — Trailing Stop Portfolio Backtest
=================================================================
Same as backtest_portfolio.py but models a trailing-stop exit strategy:

  1. Entry with fixed SL (from WF config)
  2. Each bar, ratchet stop up to max(stop, close - ATR_24h)
  3. No TP cap — profits run until the trailing stop is hit or EOD

ATR_24h = rolling mean of (High - Low) over last 1440 1-min bars.

Compare output to backtest_portfolio.py to see the impact of trailing
stop management vs. fixed TP/SL exits.
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
BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
DEPLOY_PNL_THRESHOLD = 0.018
VAL_START = '2026-01-02'
VAL_END   = '2026-04-29'
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
EXIT_TIME_MIN = 780
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDED_TICKERS = ['SMLR']

CLEAN_EXP_BASES = ('105_DEFENSE', '106_DEFENSE', '152_XSECTOR', '160_XSECTOR',
                   '170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                   '173_XSECTOR', '174_XSECTOR', '175_XSECTOR')

ATR_WINDOW = 1440  # 24h of 1-min bars

W = 120

# ==============================================================================
# STEP 1: LOAD CONFIGS FROM BQ
# ==============================================================================
print(f"{'='*W}")
print(f"  TRAILING STOP BACKTEST — VAL-A Period ({VAL_START} → {VAL_END})")
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

df_bq['ticker'] = df_bq['ticker'].replace(TICKER_RENAME)
df_bq['survived_a'] = df_bq['survived_a'].fillna(False).astype(bool)
df_bq = df_bq[df_bq['experiment'].apply(lambda e: any(e == b or e.startswith(b + '_') for b in CLEAN_EXP_BASES))]

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

    best_row = candidates.loc[candidates['val_a_avg_pnl'].idxmax()]

    if best_row['val_a_avg_pnl'] < DEPLOY_PNL_THRESHOLD:
        continue

    configs.append({
        'ticker': ticker,
        'experiment': best_row['experiment'],
        'seed': int(best_row['seed']),
        'entry_minute': int(best_row['entry_minute']),
        'prob_thresh': float(best_row['prob_thresh']),
        'sl_pct': float(best_row['sl_pct']),
        'vix_threshold': float(best_row['vix_threshold']),
        'tlt_crash': float(best_row['tlt_crash']),
        'spy_ignore': float(best_row['spy_ignore']),
        'wf_val_a_pnl': float(best_row['val_a_avg_pnl']),
        'wf_val_a_wr': float(best_row['val_a_win_rate']),
        'wf_val_a_n': int(best_row['val_a_count']),
    })

configs_df = pd.DataFrame(configs)
if configs_df.empty:
    print("\n  No configs meet deploy threshold.")
    raise SystemExit(0)

print(f"\n  TABLE 1: CONFIGS LOADED ({len(configs_df)} tickers, PnL >= {DEPLOY_PNL_THRESHOLD:.1%})")
print(f"  {'─'*(W-2)}")
print(f"  {'Ticker':<7} {'Experiment':<14} {'Seed':>6} {'Entry':>6} {'Prob':>6} "
      f"{'SL':>5} {'VIX<':>5} {'TLT>':>6} {'SPY<':>5}  {'WF Val%':>8}")
print(f"  {'─'*(W-2)}")
for _, c in configs_df.iterrows():
    em = int(c['entry_minute'])
    print(f"  {c['ticker']:<7} {c['experiment']:<14} {c['seed']:>6} "
          f"{em//60}:{em%60:02d}  {c['prob_thresh']:>5.3f} "
          f"{c['sl_pct']:>4.0%} "
          f"{c['vix_threshold']:>5.0f} {c['tlt_crash']:>5.2f} {c['spy_ignore']:>5.2f}  "
          f"{c['wf_val_a_pnl']:>7.2%}")

# ==============================================================================
# SIMULATION ENGINE — Trailing Stop (Numba)
# ==============================================================================
@jit(nopython=True)
def run_trade_sim_trailing(entry_indices, prices, highs, lows, date_ints,
                           time_mins, sl, exit_time, atr_24h):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    trades      = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)

    trade_count       = 0
    current_entry_ptr = 0
    last_traded_date  = -1

    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1:
            break

        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1
            continue

        entry_price = prices[idx]
        stop_price  = entry_price * (1.0 - sl)

        outcome    = 0.0
        trade_done = False
        exit_idx   = idx

        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k - 1] - entry_price) / entry_price
                exit_idx = k
                trade_done = True
                break

            if lows[k] <= stop_price:
                outcome = (stop_price - entry_price) / entry_price
                exit_idx = k
                trade_done = True
                break

            if time_mins[k] >= exit_time:
                outcome = (prices[k] - entry_price) / entry_price
                exit_idx = k
                trade_done = True
                break

            trailing = prices[k] - atr_24h[k]
            if trailing > stop_price:
                stop_price = trailing

        if not trade_done:
            outcome  = (prices[n_prices - 1] - entry_price) / entry_price
            exit_idx = n_prices

        trades[trade_count]      = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date

        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx:
                break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return trades[:trade_count], entry_dates[:trade_count]


# ==============================================================================
# DATA LOADING
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

    tr = df['High'].values - df['Low'].values
    n = len(tr)
    atr_24h = np.empty(n, dtype=np.float64)
    cumtr = np.cumsum(tr)
    for i in range(min(ATR_WINDOW - 1, n)):
        atr_24h[i] = cumtr[i] / (i + 1)
    if n >= ATR_WINDOW:
        atr_24h[ATR_WINDOW - 1:] = (cumtr[ATR_WINDOW - 1:] - np.concatenate([[0.0], cumtr[:-ATR_WINDOW]])) / ATR_WINDOW

    return {
        'index':    df.index,
        'prices':   df['Close'].values.copy(),
        'highs':    df['High'].values.copy(),
        'lows':     df['Low'].values.copy(),
        'dates':    (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64),
        'times':    (df.index.hour * 60 + df.index.minute).values,
        'months':   max(1.0, (df.index[-1] - df.index[0]).days / 30.44),
        'atr_24h':  atr_24h,
    }


# ==============================================================================
# REGIME FILTER
# ==============================================================================
def preload_regime_data(eval_start, eval_end):
    print("\n  Loading regime data (VIX, SPY, TLT)...")
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
# STEP 2: LOAD DATA & SIMULATE TRADES
# ==============================================================================
regime_dict = preload_regime_data(VAL_START, VAL_END)

all_trades = []

print(f"\n  Simulating trades (trailing stop) for {len(configs_df)} tickers...")
print(f"  {'─'*(W-2)}")

for _, cfg in configs_df.iterrows():
    ticker = cfg['ticker']
    experiment = cfg['experiment']
    seed = int(cfg['seed'])
    entry_minute = int(cfg['entry_minute'])
    prob_thresh = cfg['prob_thresh']
    sl_pct = cfg['sl_pct']
    vix_threshold = cfg['vix_threshold']
    tlt_crash = cfg['tlt_crash']
    spy_ignore = cfg['spy_ignore']

    try:
        df_ohlcv = load_ticker_ohlcv(ticker)
    except Exception as e:
        print(f"  {ticker}: OHLCV load failed — {e}")
        continue

    ohlcv = precompute_ohlcv_arrays(df_ohlcv, VAL_START, VAL_END)
    del df_ohlcv; gc.collect()
    if ohlcv is None:
        print(f"  {ticker}: No OHLCV data in VAL-A window")
        continue

    seed_prob = load_single_seed_scores(ticker, experiment, seed)
    if seed_prob is None:
        print(f"  {ticker}: Seed {seed} scores not found ({experiment})")
        continue

    idx = ohlcv['index']
    common = idx.intersection(seed_prob.index)
    if len(common) == 0:
        print(f"  {ticker}: No overlapping timestamps")
        continue

    probs = np.zeros(len(idx), dtype=np.float64)
    common_locs = idx.get_indexer(common)
    probs[common_locs] = seed_prob.loc[common].values

    mask = (probs >= prob_thresh) & (ohlcv['times'] == entry_minute)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        print(f"  {ticker}: No entry signals (prob>={prob_thresh:.3f} at min {entry_minute})")
        continue

    raw_pnl, raw_dates = run_trade_sim_trailing(
        indices, ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
        ohlcv['dates'], ohlcv['times'].astype(np.int64),
        sl_pct, EXIT_TIME_MIN, ohlcv['atr_24h']
    )

    if len(raw_pnl) == 0:
        print(f"  {ticker}: No trades simulated")
        continue

    n = len(raw_pnl)
    trade_vix, trade_tlt, trade_spy = _attach_regime(raw_dates, regime_dict, n, entry_minute)
    trap = (trade_vix > vix_threshold) & (trade_tlt < tlt_crash) & (trade_spy > spy_ignore)
    kept = ~trap

    final_pnl = raw_pnl[kept]
    final_dates = raw_dates[kept]
    blocked_count = int(trap.sum())

    for i in range(len(final_pnl)):
        all_trades.append({
            'ticker': ticker,
            'experiment': experiment,
            'date_int': int(final_dates[i]),
            'pnl': float(final_pnl[i]),
        })

    wins = int(np.sum(final_pnl > 0))
    losses = int(np.sum(final_pnl <= 0))
    avg_pnl = float(np.mean(final_pnl)) if len(final_pnl) > 0 else 0
    print(f"  {ticker:<7} {experiment:<14} seed {seed:>5}  "
          f"{len(final_pnl):>3} trades ({wins}W/{losses}L)  "
          f"avg {avg_pnl:>+6.2%}  blocked {blocked_count}")

    del ohlcv, seed_prob; gc.collect()


# ==============================================================================
# STEP 3: ANALYTICS
# ==============================================================================
if not all_trades:
    print("\n  No trades were simulated.")
    raise SystemExit(0)

trades_df = pd.DataFrame(all_trades)
trades_df['date'] = pd.to_datetime(trades_df['date_int'].astype(str), format='%Y%m%d')

# --- TABLE 2: PER-ASSET RESULTS ---
print(f"\n\n{'='*W}")
print(f"  TABLE 2: PER-ASSET RESULTS (Trailing Stop)")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'Exp':<14} {'Trades':>6} {'W/L':>7} {'WinRate':>8} "
      f"{'AvgPnL':>8} {'TotalPnL':>9} {'Freq/mo':>8}")
print(f"  {'─'*(W-2)}")

asset_stats = []
months = max(1.0, (trades_df['date'].max() - trades_df['date'].min()).days / 30.44)

for ticker in sorted(trades_df['ticker'].unique()):
    td = trades_df[trades_df['ticker'] == ticker]
    n_trades = len(td)
    wins = int((td['pnl'] > 0).sum())
    losses = n_trades - wins
    wr = wins / n_trades if n_trades > 0 else 0
    avg = td['pnl'].mean()
    total = td['pnl'].sum()
    freq = n_trades / months
    exp = td['experiment'].iloc[0]

    asset_stats.append({
        'ticker': ticker, 'experiment': exp, 'trades': n_trades,
        'wins': wins, 'losses': losses, 'win_rate': wr,
        'avg_pnl': avg, 'total_pnl': total, 'freq': freq,
    })

    print(f"  {ticker:<7} {exp:<14} {n_trades:>6} {wins:>3}/{losses:<3} "
          f"{wr:>7.0%} {avg:>+7.2%} {total:>+8.2%} {freq:>7.1f}")

asset_df = pd.DataFrame(asset_stats).sort_values('total_pnl', ascending=False)

# --- TABLE 3: PORTFOLIO SUMMARY ---
total_trades = len(trades_df)
total_wins = int((trades_df['pnl'] > 0).sum())
total_losses = total_trades - total_wins
total_pnl = trades_df['pnl'].sum()
avg_pnl = trades_df['pnl'].mean()
win_rate = total_wins / total_trades if total_trades > 0 else 0
median_pnl = trades_df['pnl'].median()

daily_pnl = trades_df.groupby('date')['pnl'].sum().sort_index()
cum_pnl = daily_pnl.cumsum()
running_max = cum_pnl.cummax()
drawdown = cum_pnl - running_max
max_dd = drawdown.min()

print(f"\n\n{'='*W}")
print(f"  TABLE 3: PORTFOLIO SUMMARY (Trailing Stop)")
print(f"{'='*W}")
print(f"  Period:           {VAL_START} → {VAL_END} ({months:.1f} months)")
print(f"  Tickers:          {trades_df['ticker'].nunique()}")
print(f"  Total trades:     {total_trades}")
print(f"  Wins / Losses:    {total_wins} / {total_losses}")
print(f"  Win rate:         {win_rate:.1%}")
print(f"  Avg PnL/trade:    {avg_pnl:+.2%}")
print(f"  Median PnL/trade: {median_pnl:+.2%}")
print(f"  Total PnL (sum):  {total_pnl:+.2%}")
print(f"  Max drawdown:     {max_dd:+.2%}")
print(f"  Trades/month:     {total_trades / months:.1f}")

# --- TABLE 4: DAY-BY-DAY TRADE LOG ---
print(f"\n\n{'='*W}")
print(f"  TABLE 4: DAY-BY-DAY TRADE LOG")
print(f"{'='*W}")
print(f"  {'Date':<12} {'#Signals':>8} {'Tickers':<50} {'DailyPnL':>9} {'CumPnL':>9}")
print(f"  {'─'*(W-2)}")

daily_detail = trades_df.groupby('date').agg(
    n_signals=('ticker', 'count'),
    tickers=('ticker', lambda x: ', '.join(sorted(x))),
    daily_pnl=('pnl', 'sum'),
).sort_index()
daily_detail['cum_pnl'] = daily_detail['daily_pnl'].cumsum()

for date, row in daily_detail.iterrows():
    tickers_str = row['tickers']
    if len(tickers_str) > 48:
        tickers_str = tickers_str[:45] + '...'
    print(f"  {date.strftime('%Y-%m-%d'):<12} {row['n_signals']:>8} "
          f"{tickers_str:<50} {row['daily_pnl']:>+8.2%} {row['cum_pnl']:>+8.2%}")

# --- TABLE 5: DAILY SIGNAL DENSITY ---
print(f"\n\n{'='*W}")
print(f"  TABLE 5: DAILY SIGNAL DENSITY")
print(f"{'='*W}")

all_trading_days = pd.bdate_range(VAL_START, VAL_END)
n_total_days = len(all_trading_days)
signal_counts = daily_detail['n_signals']
n_days_with_signals = len(signal_counts)
n_days_no_signal = n_total_days - n_days_with_signals

print(f"  Trading days in period:  {n_total_days}")
print(f"  Days with 0 signals:     {n_days_no_signal} ({n_days_no_signal/n_total_days:.0%})")

for n in range(1, 5):
    count = int((signal_counts == n).sum())
    print(f"  Days with {n} signal(s):    {count} ({count/n_total_days:.0%})")

count_5plus = int((signal_counts >= 5).sum())
print(f"  Days with 5+ signals:    {count_5plus} ({count_5plus/n_total_days:.0%})")

max_signals = int(signal_counts.max())
max_date = signal_counts.idxmax()
max_tickers = daily_detail.loc[max_date, 'tickers']
print(f"\n  Busiest day: {max_date.strftime('%Y-%m-%d')} with {max_signals} signals")
print(f"    Tickers: {max_tickers}")


# ==============================================================================
# STEP 4: MATPLOTLIB CHARTS
# ==============================================================================
print(f"\n\n  Generating charts...")

fig, axes = plt.subplots(3, 1, figsize=(14, 16))
fig.suptitle(f'Trailing Stop Backtest — {len(configs_df)} Tickers, VAL-A ({VAL_START} → {VAL_END})',
             fontsize=14, fontweight='bold')

ax1 = axes[0]
ax1.plot(cum_pnl.index, cum_pnl.values * 100, color='#2196F3', linewidth=2)
ax1.fill_between(cum_pnl.index, 0, cum_pnl.values * 100, alpha=0.15, color='#2196F3')
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax1.set_title(f'Cumulative Portfolio PnL — Trailing Stop (Total: {total_pnl*100:+.1f}%)', fontsize=12)
ax1.set_ylabel('Cumulative PnL (%)')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
all_bdays = pd.bdate_range(VAL_START, VAL_END)
daily_counts = daily_detail['n_signals'].reindex(all_bdays, fill_value=0)
colors = ['#4CAF50' if c > 0 else '#E0E0E0' for c in daily_counts.values]
ax2.bar(daily_counts.index, daily_counts.values, color=colors, width=0.8)
ax2.set_title(f'Daily Signal Count (avg: {daily_counts.mean():.1f}/day)', fontsize=12)
ax2.set_ylabel('# Signals')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

ax3 = axes[2]
asset_sorted = asset_df.sort_values('total_pnl', ascending=True)
bar_colors = ['#4CAF50' if p > 0 else '#F44336' for p in asset_sorted['total_pnl']]
ax3.barh(asset_sorted['ticker'], asset_sorted['total_pnl'] * 100, color=bar_colors)
ax3.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax3.set_title('Total PnL by Asset — Trailing Stop (%)', fontsize=12)
ax3.set_xlabel('Total PnL (%)')
ax3.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
chart_path = '/tmp/backtest_trailing_stop_charts.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Charts saved: {chart_path}")
try:
    from IPython.display import display, Image
    display(Image(filename=chart_path))
except ImportError:
    plt.show()

print(f"\n{'='*W}")
print(f"  DONE — {total_trades} trades across {trades_df['ticker'].nunique()} tickers")
print(f"  Trailing Stop Portfolio: {total_pnl:+.2%} total, {avg_pnl:+.2%} avg, {win_rate:.0%} WR")
print(f"{'='*W}")
