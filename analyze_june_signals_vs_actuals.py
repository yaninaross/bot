"""
analyze_june_signals_vs_actuals.py — June Live Signal Analysis
==============================================================
Pulls all logged GO signals from BQ (live_predictions) for June 2026,
then loads actual OHLCV price data to compute what happened after each
signal: TP hit, SL hit, or EOD exit. Compares signal expectations vs
actual outcomes day by day.

IMPORTANT: BQ logs multiple rows per (date, ticker) — different seeds,
different scoring runs at different entry_minutes. We deduplicate to
ONE TRADE per (date, ticker) since you only trade each ticker once/day.

Colab-ready (no if __name__). Requires: GCS, BQ, pandas.
"""

import os, gc, sys, subprocess, warnings, ctypes
from datetime import datetime, timedelta
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

try:
    from google.cloud import storage as gcs_storage, bigquery
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-cloud-storage", "google-cloud-bigquery",
                           "pyarrow", "--quiet"])
    from google.cloud import storage as gcs_storage, bigquery

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BQ_LOG_TABLE   = "ys-time-series.scoring_logs.live_predictions"
BUCKET_NAME    = "time-series-bucket-ys"
OHLCV_DIR      = "Data/Inference"
INFERENCE_DIR  = "Data/Inference_v2"

JUNE_START = '2026-06-01'
JUNE_END   = '2026-06-09'
STARTING_CAPITAL = 200_000

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

W = 120

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()


# ==============================================================================
# STEP 1: PULL ALL JUNE SIGNALS FROM BQ
# ==============================================================================
print(f"{'='*W}")
print(f"  JUNE LIVE SIGNAL ANALYSIS — Logged Signals vs Actual Price Action")
print(f"  Period: {JUNE_START} → {JUNE_END}")
print(f"{'='*W}")

q = f"""
SELECT
    score_date,
    ticker,
    experiment,
    seed,
    prob,
    threshold,
    signal,
    tp_pct,
    sl_pct,
    entry_minute,
    close AS entry_close,
    filter_status,
    vix_value,
    tlt_gap_pct,
    spy_gap_pct,
    vol_filter_pct,
    vol_filter_ok,
    entry_volume,
    dollar_volume,
    vol_median,
    tpsl_mode,
    atr_pct_entry,
    bar_timestamp,
    run_timestamp
FROM `{BQ_LOG_TABLE}`
WHERE score_date >= '{JUNE_START}'
  AND score_date <= '{JUNE_END}'
ORDER BY score_date, ticker, run_timestamp
"""
print("  Loading BQ scoring logs...")
df_log = bq_client.query(q).to_dataframe()
print(f"  Total logged rows: {len(df_log)}")

if len(df_log) == 0:
    print("  ERROR: No scoring logs found for June. Check BQ table.")
    import sys; sys.exit(1)

# Parse dates
df_log['score_date'] = pd.to_datetime(df_log['score_date']).dt.strftime('%Y-%m-%d')
df_log['entry_minute'] = df_log['entry_minute'].fillna(0).astype(int)
df_log['tp_pct'] = df_log['tp_pct'].fillna(0).astype(float)
df_log['sl_pct'] = df_log['sl_pct'].fillna(0).astype(float)
df_log['prob'] = df_log['prob'].fillna(0).astype(float)
df_log['threshold'] = df_log['threshold'].fillna(0).astype(float)
df_log['entry_close'] = df_log['entry_close'].fillna(0).astype(float)


# ==============================================================================
# DEDUPLICATION: One trade decision per (date, ticker)
# ==============================================================================
# Multiple BQ rows per (date, ticker) come from:
#   - Multiple seeds scoring the same ticker
#   - Scorer running every few minutes (6:00-6:30), logging each run
# In reality: you trade each ticker AT MOST ONCE per day.
#
# Logic: For each (date, ticker), determine the FINAL trade decision:
#   - If ANY row has signal=GO + filter=OK → TRADED (take last GO-OK row)
#   - If ANY row has signal=GO + filter=BLOCKED (and no GO-OK) → BLOCKED
#   - Otherwise → NO (model prob below threshold)

def dedup_signals(df):
    """Deduplicate to one row per (score_date, ticker).
    Returns 3 DataFrames: traded, blocked, no_signal."""
    traded = []
    blocked = []
    no_signal = []

    for (date, ticker), grp in df.groupby(['score_date', 'ticker']):
        go_ok = grp[(grp['signal'] == 'GO') & (grp['filter_status'] == 'OK')]
        go_blocked = grp[(grp['signal'] == 'GO') & (grp['filter_status'] == 'BLOCKED')]

        if len(go_ok) > 0:
            rep = go_ok.iloc[-1]
            rep_dict = rep.to_dict()
            rep_dict['n_go_ok_rows'] = len(go_ok)
            rep_dict['avg_prob'] = go_ok['prob'].mean()
            rep_dict['max_prob'] = go_ok['prob'].max()
            traded.append(rep_dict)
        elif len(go_blocked) > 0:
            rep = go_blocked.iloc[-1]
            rep_dict = rep.to_dict()
            rep_dict['n_go_blocked_rows'] = len(go_blocked)
            rep_dict['avg_prob'] = go_blocked['prob'].mean()
            rep_dict['max_prob'] = go_blocked['prob'].max()
            blocked.append(rep_dict)
        else:
            rep = grp.iloc[-1]
            rep_dict = rep.to_dict()
            rep_dict['n_no_rows'] = len(grp)
            rep_dict['avg_prob'] = grp['prob'].mean()
            rep_dict['max_prob'] = grp['prob'].max()
            no_signal.append(rep_dict)

    return (pd.DataFrame(traded) if traded else pd.DataFrame(),
            pd.DataFrame(blocked) if blocked else pd.DataFrame(),
            pd.DataFrame(no_signal) if no_signal else pd.DataFrame())

df_traded, df_blocked_go, df_no = dedup_signals(df_log)

print(f"\n  After dedup (one per date×ticker):")
print(f"    TRADED (GO-OK):   {len(df_traded)} unique ticker-days")
print(f"    BLOCKED (GO):     {len(df_blocked_go)} unique ticker-days")
print(f"    NO (below thresh): {len(df_no)} unique ticker-days")


# ==============================================================================
# TABLE 1: DAILY SIGNAL SUMMARY (deduplicated)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 1: DAILY SIGNAL SUMMARY — One row per (date, ticker)")
print(f"{'='*W}")

all_dedup = pd.concat([
    df_traded.assign(decision='TRADED') if len(df_traded) > 0 else pd.DataFrame(),
    df_blocked_go.assign(decision='BLOCKED') if len(df_blocked_go) > 0 else pd.DataFrame(),
    df_no.assign(decision='NO') if len(df_no) > 0 else pd.DataFrame(),
], ignore_index=True)

if len(all_dedup) > 0:
    for date in sorted(all_dedup['score_date'].unique()):
        day = all_dedup[all_dedup['score_date'] == date]
        t = day[day['decision'] == 'TRADED']
        b = day[day['decision'] == 'BLOCKED']
        n = day[day['decision'] == 'NO']

        print(f"\n  {date}  |  {len(day)} tickers scored  |  {len(t)} TRADED  |  {len(b)} BLOCKED  |  {len(n)} NO")
        print(f"  {'':>2}{'Ticker':<7} {'Decision':<9} {'AvgProb':>8} {'Thresh':>7} {'TP%':>5} {'SL%':>5} "
              f"{'Entry$':>8} {'Mode':>8} {'VIX':>5} {'Vol$':>10}")
        print(f"  {'':>2}{'─'*7} {'─'*9} {'─'*8} {'─'*7} {'─'*5} {'─'*5} "
              f"{'─'*8} {'─'*8} {'─'*5} {'─'*10}")

        for _, r in day.sort_values(['decision', 'ticker']).iterrows():
            marker = '>>' if r['decision'] == 'TRADED' else '  '
            vix_str = f"{r['vix_value']:.0f}" if pd.notna(r.get('vix_value')) else '—'
            dvol_str = f"${r['dollar_volume']:,.0f}" if pd.notna(r.get('dollar_volume')) and r.get('dollar_volume', 0) > 0 else '—'
            mode = str(r['tpsl_mode'])[:8] if pd.notna(r.get('tpsl_mode')) else '—'
            print(f"{marker}{r['ticker']:<7} {r['decision']:<9} {r['avg_prob']:>8.3f} {r['threshold']:>7.3f} "
                  f"{r['tp_pct']*100:>4.1f}% {r['sl_pct']*100:>4.1f}% "
                  f"${r['entry_close']:>7.2f} {mode:>8} {vix_str:>5} {dvol_str:>10}")


# ==============================================================================
# STEP 2: LOAD ACTUAL OHLCV FOR TRADED TICKERS, COMPUTE OUTCOMES
# ==============================================================================
if len(df_traded) == 0:
    print("\n  No traded signals found. Exiting.")
    import sys; sys.exit(1)

print(f"\n\n{'='*W}")
print(f"  STEP 2: Loading OHLCV for {df_traded['ticker'].nunique()} traded tickers...")
print(f"{'='*W}")

def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open','High','Low','Close']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df

all_tickers_needed = set()
if len(df_traded) > 0:
    all_tickers_needed |= set(df_traded['ticker'].unique())
if len(df_blocked_go) > 0:
    all_tickers_needed |= set(df_blocked_go['ticker'].unique())

ohlcv_cache = {}
for ticker in sorted(all_tickers_needed):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for dir_path in [INFERENCE_DIR, OHLCV_DIR]:
        uri = f"gs://{BUCKET_NAME}/{dir_path}/{clean}_master_inference.parquet"
        try:
            df_raw = pd.read_parquet(uri)
            break
        except Exception:
            try:
                uri = f"gs://{BUCKET_NAME}/{dir_path}/{clean}_ohlcv_1min_pst.parquet"
                df_raw = pd.read_parquet(uri)
                break
            except Exception:
                continue
    else:
        print(f"  {ticker}: NO DATA FOUND")
        continue

    df_raw = df_raw.reset_index()
    dt_cols = [c for c in df_raw.columns if pd.api.types.is_datetime64_any_dtype(df_raw[c])]
    if dt_cols:
        df_raw['Datetime'] = pd.to_datetime(df_raw[dt_cols[0]], utc=True)
    else:
        df_raw['Datetime'] = pd.to_datetime(df_raw[df_raw.columns[0]], unit='ms', utc=True)
    df_raw = df_raw.set_index('Datetime').sort_index()
    df_raw = df_raw[~df_raw.index.duplicated(keep='first')]
    std_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
    lower_map = {k.lower():v for k,v in std_map.items()}
    df_raw.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df_raw.columns]

    june_start_ts = pd.Timestamp(JUNE_START, tz='UTC') - pd.Timedelta(days=1)
    june_end_ts = pd.Timestamp(JUNE_END, tz='UTC') + pd.Timedelta(days=2)
    df_raw = df_raw[(df_raw.index >= june_start_ts) & (df_raw.index <= june_end_ts)]
    df_raw = heal_data(df_raw)

    df_pst = df_raw.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    ohlcv_cache[ticker] = df_pst
    print(f"  {ticker}: {len(df_pst)} bars loaded")


MIN_BARS_FULL_DAY = 200

def compute_trade_outcome(ticker, date_str, entry_min, entry_price, tp_pct, sl_pct):
    """Compute actual TP/SL/EOD outcome for a single trade."""
    _empty = {'actual_pnl': None, 'actual_exit': 'NO_DATA',
              'max_gain': None, 'max_drop': None, 'eod_price': None, 'eod_ret': None,
              'time_to_exit': None, 'n_bars': 0, 'last_bar': None, 'data_complete': False}

    if ticker not in ohlcv_cache or entry_price <= 0:
        return _empty

    df_pst = ohlcv_cache[ticker]
    trade_date = pd.Timestamp(date_str).date()
    entry_hour = entry_min // 60
    entry_minute = entry_min % 60
    day_bars = df_pst[df_pst.index.date == trade_date]

    if len(day_bars) == 0:
        return _empty

    entry_ts = pd.Timestamp(f"{date_str} {entry_hour:02d}:{entry_minute:02d}:00", tz=PST)
    after_entry = day_bars[day_bars.index >= entry_ts]

    if len(after_entry) == 0:
        _empty['actual_exit'] = 'NO_BARS'
        return _empty

    n_bars = len(after_entry)
    last_bar = after_entry.index[-1]
    data_complete = n_bars >= MIN_BARS_FULL_DAY

    highs_ret = (after_entry['High'].values / entry_price - 1)
    lows_ret = (after_entry['Low'].values / entry_price - 1)
    max_gain = float(np.max(highs_ret))
    max_drop = float(np.min(lows_ret))
    eod_price = float(after_entry['Close'].iloc[-1])
    eod_ret = (eod_price / entry_price - 1)

    trade_pnl = eod_ret
    exit_type = 'EOD'
    if sl_pct > 0 and max_drop <= -sl_pct:
        trade_pnl = -sl_pct
        exit_type = 'SL'
    elif tp_pct > 0 and max_gain >= tp_pct:
        trade_pnl = tp_pct
        exit_type = 'TP'

    if not data_complete and exit_type == 'EOD':
        exit_type = 'PARTIAL'

    time_to_exit = None
    if exit_type == 'SL':
        sl_bars = after_entry[(after_entry['Low'] / entry_price - 1) <= -sl_pct]
        if len(sl_bars) > 0:
            time_to_exit = after_entry.index.get_loc(sl_bars.index[0]) - after_entry.index.get_loc(after_entry.index[0])
    elif exit_type == 'TP':
        tp_bars = after_entry[(after_entry['High'] / entry_price - 1) >= tp_pct]
        if len(tp_bars) > 0:
            time_to_exit = after_entry.index.get_loc(tp_bars.index[0]) - after_entry.index.get_loc(after_entry.index[0])

    return {'actual_pnl': trade_pnl, 'actual_exit': exit_type,
            'max_gain': max_gain, 'max_drop': max_drop,
            'eod_price': eod_price, 'eod_ret': eod_ret,
            'time_to_exit': time_to_exit,
            'n_bars': n_bars, 'last_bar': last_bar, 'data_complete': data_complete}


# Compute outcomes for TRADED signals
outcomes = []
for _, sig in df_traded.iterrows():
    result = compute_trade_outcome(
        sig['ticker'], sig['score_date'], sig['entry_minute'],
        sig['entry_close'], sig['tp_pct'], sig['sl_pct'])
    outcomes.append({**sig.to_dict(), **result})

df_out = pd.DataFrame(outcomes)
print(f"\n  Computed outcomes for {len(df_out)} actual trades")

# Flag incomplete data
incomplete = df_out[~df_out['data_complete'] & df_out['actual_pnl'].notna()]
if len(incomplete) > 0:
    print(f"\n  *** WARNING: {len(incomplete)} trade(s) have INCOMPLETE OHLCV data (< {MIN_BARS_FULL_DAY} bars after entry) ***")
    print(f"  *** These trades show 'PARTIAL' exit — PnL is unreliable, SL/TP may not have been detected ***")
    for _, r in incomplete.iterrows():
        last_str = r['last_bar'].strftime('%H:%M') if pd.notna(r.get('last_bar')) else '?'
        print(f"      {r['score_date']} {r['ticker']}: only {r['n_bars']} bars, last bar at {last_str} PST")


# ==============================================================================
# TABLE 2: TRADED SIGNALS WITH ACTUAL OUTCOMES
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 2: TRADED SIGNALS — Actual Outcome (one per ticker-day)")
print(f"{'='*W}")

print(f"\n  {'Date':<12} {'Ticker':<7} {'AvgP':>5} {'Entry$':>8} {'TP%':>5} {'SL%':>5} "
      f"{'Exit':>7} {'PnL':>7} {'MaxGain':>8} {'MaxDrop':>8} {'EOD$':>8} {'EOD%':>7} {'Mins':>5} {'Bars':>5}")
print(f"  {'─'*12} {'─'*7} {'─'*5} {'─'*8} {'─'*5} {'─'*5} "
      f"{'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*5} {'─'*5}")

for _, r in df_out.sort_values(['score_date', 'ticker']).iterrows():
    if r['actual_pnl'] is None:
        pnl_str = '  N/A '
        mg_str = '    N/A '
        md_str = '    N/A '
        eod_str = '    N/A '
        eodr_str = '   N/A '
        mins_str = '  — '
    else:
        pnl_str = f"{r['actual_pnl']*100:>+6.2f}%"
        mg_str = f"{r['max_gain']*100:>+7.2f}%"
        md_str = f"{r['max_drop']*100:>+7.2f}%"
        eod_str = f"${r['eod_price']:>7.2f}"
        eodr_str = f"{r['eod_ret']*100:>+6.2f}%"
        mins_str = f"{int(r['time_to_exit']):>4}m" if pd.notna(r.get('time_to_exit')) else '  — '

    bars_str = f"{r['n_bars']:>5}" if r.get('n_bars', 0) > 0 else '    —'
    warn = '!!' if not r.get('data_complete', True) and r['actual_pnl'] is not None else '  '
    win = '+' if r['actual_pnl'] is not None and r['actual_pnl'] > 0 else '-' if r['actual_pnl'] is not None and r['actual_pnl'] < 0 else ' '
    avg_p = r.get('avg_prob', r['prob'])
    print(f"{warn}{win}{r['score_date']:<12} {r['ticker']:<7} {avg_p:.3f} "
          f"${r['entry_close']:>7.2f} {r['tp_pct']*100:>4.1f}% {r['sl_pct']*100:>4.1f}% "
          f"{r['actual_exit']:>7} {pnl_str} {mg_str} {md_str} {eod_str} {eodr_str} {mins_str} {bars_str}")


# ==============================================================================
# TABLE 3: DAY-BY-DAY SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 3: DAY-BY-DAY PERFORMANCE SUMMARY (actual trades)")
print(f"{'='*W}")

valid = df_out[df_out['actual_pnl'].notna()].copy()
n_partial = (~valid['data_complete']).sum()

print(f"\n  {'Date':<12} {'Trades':>7} {'Wins':>5} {'Loss':>5} {'WR':>5} "
      f"{'AvgPnL':>8} {'AvgMaxG':>8} {'AvgMaxD':>8} "
      f"{'TP':>3} {'SL':>3} {'EOD':>4} {'$PnL(200K)':>12}")
print(f"  {'─'*12} {'─'*7} {'─'*5} {'─'*5} {'─'*5} "
      f"{'─'*8} {'─'*8} {'─'*8} "
      f"{'─'*3} {'─'*3} {'─'*4} {'─'*12}")

total_dollar_pnl = 0
for date in sorted(valid['score_date'].unique()):
    day = valid[valid['score_date'] == date]
    n = len(day)
    wins = (day['actual_pnl'] > 0).sum()
    losses = (day['actual_pnl'] <= 0).sum()
    wr = wins / n * 100 if n > 0 else 0
    avg_pnl = day['actual_pnl'].mean() * 100
    avg_mg = day['max_gain'].mean() * 100
    avg_md = day['max_drop'].mean() * 100
    tp_count = (day['actual_exit'] == 'TP').sum()
    sl_count = (day['actual_exit'] == 'SL').sum()
    eod_count = (day['actual_exit'] == 'EOD').sum()
    partial_count = (day['actual_exit'] == 'PARTIAL').sum()

    alloc_per = STARTING_CAPITAL / n if n > 0 else 0
    day_dollar = sum(alloc_per * pnl for pnl in day['actual_pnl'])
    total_dollar_pnl += day_dollar

    partial_tag = f" !! {partial_count} PARTIAL" if partial_count > 0 else ""
    marker = '  ' if day_dollar >= 0 else '>>'
    print(f"{marker}{date:<12} {n:>7} {wins:>5} {losses:>5} {wr:>4.0f}% "
          f"{avg_pnl:>+7.2f}% {avg_mg:>+7.2f}% {avg_md:>+7.2f}% "
          f"{tp_count:>3} {sl_count:>3} {eod_count:>4} ${day_dollar:>+11,.0f}{partial_tag}")

partial_note = f" ({n_partial} trades have incomplete OHLCV — re-run after market close)" if n_partial > 0 else ""
print(f"\n  {'TOTAL':<12} {len(valid):>7} {(valid['actual_pnl'] > 0).sum():>5} "
      f"{(valid['actual_pnl'] <= 0).sum():>5} "
      f"{(valid['actual_pnl'] > 0).mean()*100:>4.0f}% "
      f"{valid['actual_pnl'].mean()*100:>+7.2f}% "
      f"{valid['max_gain'].mean()*100:>+7.2f}% "
      f"{valid['max_drop'].mean()*100:>+7.2f}% "
      f"{(valid['actual_exit']=='TP').sum():>3} "
      f"{(valid['actual_exit']=='SL').sum():>3} "
      f"{(valid['actual_exit']=='EOD').sum():>4} "
      f"${total_dollar_pnl:>+11,.0f}{partial_note}")


# ==============================================================================
# TABLE 4: BLOCKED GO SIGNALS — What would have happened?
# ==============================================================================
if len(df_blocked_go) > 0:
    print(f"\n\n{'='*W}")
    print(f"  TABLE 4: BLOCKED GO SIGNALS — What the filters prevented ({len(df_blocked_go)} ticker-days)")
    print(f"{'='*W}")

    blocked_outcomes = []
    for _, sig in df_blocked_go.iterrows():
        result = compute_trade_outcome(
            sig['ticker'], sig['score_date'], sig['entry_minute'],
            sig['entry_close'], sig['tp_pct'], sig['sl_pct'])
        blocked_outcomes.append({**sig.to_dict(), **result})

    df_blocked_out = pd.DataFrame(blocked_outcomes)
    valid_blocked = df_blocked_out[df_blocked_out['actual_pnl'].notna()]

    print(f"\n  {'Date':<12} {'Ticker':<7} {'AvgP':>5} {'Entry$':>8} {'WouldPnL':>9} {'WouldExit':>10} "
          f"{'MaxGain':>8} {'MaxDrop':>8} {'Saved?':>7}")
    print(f"  {'─'*12} {'─'*7} {'─'*5} {'─'*8} {'─'*9} {'─'*10} {'─'*8} {'─'*8} {'─'*7}")

    for _, r in valid_blocked.sort_values(['score_date', 'ticker']).iterrows():
        saved = 'YES' if r['actual_pnl'] < 0 else 'no'
        avg_p = r.get('avg_prob', r['prob'])
        print(f"  {r['score_date']:<12} {r['ticker']:<7} {avg_p:.3f} "
              f"${r['entry_close']:>7.2f} "
              f"{r['actual_pnl']*100:>+8.2f}% {r['actual_exit']:>10} "
              f"{r['max_gain']*100:>+7.2f}% {r['max_drop']*100:>+7.2f}% {saved:>7}")

    if len(valid_blocked) > 0:
        blocked_losses = valid_blocked[valid_blocked['actual_pnl'] < 0]
        blocked_wins = valid_blocked[valid_blocked['actual_pnl'] >= 0]
        print(f"\n  Summary of blocked signals:")
        if len(blocked_losses) > 0:
            print(f"    Would have been LOSSES: {len(blocked_losses)} trades, avg={blocked_losses['actual_pnl'].mean()*100:+.2f}%")
        else:
            print(f"    Would have been LOSSES: 0")
        if len(blocked_wins) > 0:
            print(f"    Would have been WINS:   {len(blocked_wins)} trades, avg={blocked_wins['actual_pnl'].mean()*100:+.2f}%")
        else:
            print(f"    Would have been WINS:   0")
else:
    df_blocked_out = pd.DataFrame()
    valid_blocked = pd.DataFrame()
    print(f"\n  No blocked GO signals found in June logs.")


# ==============================================================================
# TABLE 5: NO SIGNALS — Missed opportunities?
# ==============================================================================
if len(df_no) > 0:
    print(f"\n\n{'='*W}")
    print(f"  TABLE 5: NO SIGNALS — Model prob below threshold ({len(df_no)} ticker-days)")
    print(f"{'='*W}")

    no_outcomes = []
    for _, sig in df_no.iterrows():
        ticker = sig['ticker']
        if ticker not in ohlcv_cache or sig['entry_close'] <= 0:
            continue

        result = compute_trade_outcome(
            ticker, sig['score_date'], sig['entry_minute'],
            sig['entry_close'], 0.08, 0.06)

        no_outcomes.append({
            'score_date': sig['score_date'], 'ticker': ticker,
            'avg_prob': sig['avg_prob'], 'threshold': sig['threshold'],
            'max_gain': result['max_gain'], 'max_drop': result['max_drop'],
            'eod_ret': result['eod_ret'],
        })

    df_no_out = pd.DataFrame(no_outcomes)

    if len(df_no_out) > 0:
        df_no_out['would_spike_8pct'] = df_no_out['max_gain'].apply(lambda x: x >= 0.08 if pd.notna(x) else False)
        df_no_out['would_spike_6pct'] = df_no_out['max_gain'].apply(lambda x: x >= 0.06 if pd.notna(x) else False)

        print(f"\n  {'Date':<12} {'Tickers':>8} {'AvgProb':>8} {'≥8%Spike':>9} {'≥6%Spike':>9} {'AvgMaxG':>8} {'AvgMaxD':>8}")
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*9} {'─'*9} {'─'*8} {'─'*8}")

        for date in sorted(df_no_out['score_date'].unique()):
            day = df_no_out[df_no_out['score_date'] == date]
            n = len(day)
            avg_prob = day['avg_prob'].mean()
            spike8 = day['would_spike_8pct'].sum()
            spike6 = day['would_spike_6pct'].sum()
            valid_mg = day['max_gain'].dropna()
            valid_md = day['max_drop'].dropna()
            avg_mg = valid_mg.mean() * 100 if len(valid_mg) > 0 else 0
            avg_md = valid_md.mean() * 100 if len(valid_md) > 0 else 0
            print(f"  {date:<12} {n:>8} {avg_prob:>8.3f} {spike8:>9} {spike6:>9} {avg_mg:>+7.2f}% {avg_md:>+7.2f}%")

        big_spikes = df_no_out[df_no_out['would_spike_8pct']]
        if len(big_spikes) > 0:
            print(f"\n  MISSED ≥8% SPIKES (model said NO but price spiked):")
            for _, r in big_spikes.iterrows():
                print(f"    {r['score_date']} {r['ticker']:<7} avg_prob={r['avg_prob']:.3f} (thresh={r['threshold']:.3f}) "
                      f"max_gain={r['max_gain']*100:+.1f}%")
        else:
            print(f"\n  No missed ≥8% spikes — model correctly passed on all NO tickers.")


# ==============================================================================
# TABLE 6: PER-TICKER JUNE STATS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 6: PER-TICKER JUNE PERFORMANCE — Actual trades only")
print(f"{'='*W}")

if len(valid) > 0:
    print(f"\n  {'Ticker':<7} {'Trades':>7} {'Wins':>5} {'WR':>5} {'AvgPnL':>8} {'TotPnL':>8} "
          f"{'TP':>3} {'SL':>3} {'EOD':>4} {'AvgMaxG':>8} {'AvgMaxD':>8} {'$PnL(eq)':>12}")
    print(f"  {'─'*7} {'─'*7} {'─'*5} {'─'*5} {'─'*8} {'─'*8} "
          f"{'─'*3} {'─'*3} {'─'*4} {'─'*8} {'─'*8} {'─'*12}")

    for ticker in sorted(valid['ticker'].unique()):
        tsub = valid[valid['ticker'] == ticker]
        n = len(tsub)
        wins = (tsub['actual_pnl'] > 0).sum()
        wr = wins / n * 100
        avg_pnl = tsub['actual_pnl'].mean() * 100
        tot_pnl = tsub['actual_pnl'].sum() * 100
        tp = (tsub['actual_exit'] == 'TP').sum()
        sl = (tsub['actual_exit'] == 'SL').sum()
        eod = (tsub['actual_exit'] == 'EOD').sum()
        avg_mg = tsub['max_gain'].mean() * 100
        avg_md = tsub['max_drop'].mean() * 100
        n_partial = (tsub['actual_exit'] == 'PARTIAL').sum()

        total_dollar = 0
        for _, trade in tsub.iterrows():
            day_trades = len(valid[valid['score_date'] == trade['score_date']])
            alloc = STARTING_CAPITAL / day_trades
            total_dollar += alloc * trade['actual_pnl']

        flag = '!' if wr < 40 and n >= 2 else '*' if n_partial > 0 else ' '
        print(f"{flag} {ticker:<7} {n:>7} {wins:>5} {wr:>4.0f}% {avg_pnl:>+7.2f}% {tot_pnl:>+7.1f}% "
              f"{tp:>3} {sl:>3} {eod:>4} {avg_mg:>+7.2f}% {avg_md:>+7.2f}% ${total_dollar:>+11,.0f}")


# ==============================================================================
# TABLE 7: FILTER EFFECTIVENESS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 7: FILTER STATUS PER DAY")
print(f"{'='*W}")

if len(all_dedup) > 0:
    print(f"\n  {'Date':<12} {'Traded':>7} {'Blocked':>8} {'NO':>4} {'Total':>6}")
    print(f"  {'─'*12} {'─'*7} {'─'*8} {'─'*4} {'─'*6}")
    for date in sorted(all_dedup['score_date'].unique()):
        day = all_dedup[all_dedup['score_date'] == date]
        t = (day['decision'] == 'TRADED').sum()
        b = (day['decision'] == 'BLOCKED').sum()
        n = (day['decision'] == 'NO').sum()
        print(f"  {date:<12} {t:>7} {b:>8} {n:>4} {t+b+n:>6}")


# ==============================================================================
# TODAY'S DEEP DIVE (latest date)
# ==============================================================================
latest_date = sorted(valid['score_date'].unique())[-1] if len(valid) > 0 else None

if latest_date:
    print(f"\n\n{'='*W}")
    print(f"  TODAY'S DEEP DIVE — {latest_date}")
    print(f"{'='*W}")

    today_trades = valid[valid['score_date'] == latest_date]
    today_blocked = df_blocked_out[df_blocked_out['score_date'] == latest_date] if len(df_blocked_out) > 0 else pd.DataFrame()
    today_no = df_no[df_no['score_date'] == latest_date] if len(df_no) > 0 else pd.DataFrame()

    today_partial = today_trades[~today_trades['data_complete']]

    if len(today_partial) > 0:
        print(f"\n  *** WARNING: {len(today_partial)} trade(s) have INCOMPLETE OHLCV data (< {MIN_BARS_FULL_DAY} bars) ***")
        print(f"  *** PnL below may be wrong for PARTIAL trades — re-run after market close for accurate results ***")
        for _, r in today_partial.iterrows():
            last_str = r['last_bar'].strftime('%Y-%m-%d %H:%M PST') if pd.notna(r.get('last_bar')) else '?'
            print(f"      {r['ticker']}: {r['n_bars']} bars after entry, last bar = {last_str}")
        print()

    n = len(today_trades)
    alloc_per = STARTING_CAPITAL / n if n > 0 else 0
    day_dollar = sum(alloc_per * pnl for pnl in today_trades['actual_pnl'])

    print(f"\n  Traded {n} tickers with ${STARTING_CAPITAL:,} equal-weight (${alloc_per:,.0f} per ticker)")
    print(f"  Day P&L: ${day_dollar:+,.0f}")

    print(f"\n  {'Ticker':<7} {'Alloc$':>10} {'Entry$':>8} {'Exit':>7} {'PnL%':>7} {'$PnL':>10} "
          f"{'MaxGain':>8} {'MaxDrop':>8} {'LastBar$':>9} {'Bars':>5} {'Status':>10}")
    print(f"  {'─'*7} {'─'*10} {'─'*8} {'─'*7} {'─'*7} {'─'*10} "
          f"{'─'*8} {'─'*8} {'─'*9} {'─'*5} {'─'*10}")

    for _, r in today_trades.sort_values('actual_pnl').iterrows():
        dollar_pnl = alloc_per * r['actual_pnl']
        status = 'COMPLETE' if r['data_complete'] else 'PARTIAL!!'
        last_str = r['last_bar'].strftime('%H:%M') if pd.notna(r.get('last_bar')) else '?'
        print(f"  {r['ticker']:<7} ${alloc_per:>9,.0f} ${r['entry_close']:>7.2f} "
              f"{r['actual_exit']:>7} {r['actual_pnl']*100:>+6.2f}% ${dollar_pnl:>+9,.0f} "
              f"{r['max_gain']*100:>+7.2f}% {r['max_drop']*100:>+7.2f}% "
              f"${r['eod_price']:>8.2f} {r['n_bars']:>5} {status:>10} (last={last_str})")

    if len(today_blocked) > 0:
        print(f"\n  Blocked tickers today:")
        for _, r in today_blocked.iterrows():
            would_pnl = r.get('actual_pnl')
            pnl_str = f"{would_pnl*100:+.2f}%" if pd.notna(would_pnl) else 'N/A'
            saved = '(saved loss)' if pd.notna(would_pnl) and would_pnl < 0 else '(missed win)' if pd.notna(would_pnl) and would_pnl > 0 else ''
            print(f"    {r['ticker']:<7} prob={r.get('avg_prob', r['prob']):.3f} — would have been {pnl_str} {saved}")


# ==============================================================================
# SYNTHESIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SYNTHESIS — JUNE LIVE SIGNAL PERFORMANCE")
print(f"{'='*W}")

if len(valid) > 0:
    total_trades = len(valid)
    total_wins = (valid['actual_pnl'] > 0).sum()
    total_wr = total_wins / total_trades * 100
    total_avg_pnl = valid['actual_pnl'].mean() * 100
    total_sl = (valid['actual_exit'] == 'SL').sum()
    total_tp = (valid['actual_exit'] == 'TP').sum()
    total_partial = (valid['actual_exit'] == 'PARTIAL').sum()

    worst_day = valid.groupby('score_date')['actual_pnl'].mean().idxmin()
    worst_pnl = valid.groupby('score_date')['actual_pnl'].mean().min() * 100
    best_day = valid.groupby('score_date')['actual_pnl'].mean().idxmax()
    best_pnl = valid.groupby('score_date')['actual_pnl'].mean().max() * 100

    partial_msg = f"\n    !! {total_partial} trade(s) have incomplete OHLCV — re-run after market close for accurate results" if total_partial > 0 else ""
    print(f"""
  JUNE ACTUAL TRADES (deduplicated):
    Total: {total_trades} trades across {len(valid['score_date'].unique())} days
    Wins: {total_wins} ({total_wr:.0f}% WR)
    Avg PnL per trade: {total_avg_pnl:+.2f}%
    Exits: {total_tp} TP, {total_sl} SL, {total_trades - total_tp - total_sl - total_partial} EOD, {total_partial} PARTIAL
    Est. P&L on ${STARTING_CAPITAL:,} (equal-weight): ${total_dollar_pnl:+,.0f}

    Worst day: {worst_day} ({worst_pnl:+.2f}% avg)
    Best day:  {best_day} ({best_pnl:+.2f}% avg){partial_msg}
""")

    if len(valid_blocked) > 0:
        saved_losses = valid_blocked[valid_blocked['actual_pnl'] < 0]['actual_pnl'].sum() * 100
        missed_wins = valid_blocked[valid_blocked['actual_pnl'] > 0]['actual_pnl'].sum() * 100
        print(f"  FILTER IMPACT:")
        print(f"    Blocked GO ticker-days: {len(valid_blocked)}")
        print(f"    Losses avoided: {saved_losses:+.2f}% total")
        print(f"    Wins missed: {missed_wins:+.2f}% total")
        net = abs(saved_losses) - missed_wins
        print(f"    Net filter value: {'POSITIVE' if net > 0 else 'NEGATIVE'} ({net:+.2f}%)")

print(f"\n{'='*W}")
print(f"  JUNE LIVE SIGNAL ANALYSIS COMPLETE")
print(f"{'='*W}")
