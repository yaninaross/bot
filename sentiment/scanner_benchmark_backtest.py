# %% ── Scanner Benchmark Backtest — Sep 8-11 2026 ─────────────────────────────
#
# Evaluates the full Sep 8-11 scanner portfolio as the baseline benchmark.
# All future scanner improvements are measured against this.
#
# Methodology:
#   - Pulls ALL selected predictions from BQ for the benchmark period
#   - For days with multiple scanner versions, uses the LATEST version per day
#   - One position per ticker per day (earliest selected signal wins)
#   - Entry at signal time (pre-open → 9:30 ET open)
#   - 2.5% stop loss checked bar-by-bar
#   - Equal-weight portfolio allocation per day
#   - Alpha measured vs SPY same-window return
#
# Output:
#   - Detailed per-trade and per-day metrics
#   - Cumulative portfolio equity curve
#   - Benchmark snapshot saved as JSON for future comparison
#
# Colab-ready, config vars at top.

import os, sys, json, warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz
import matplotlib
try:
    _ipy = get_ipython()
    if _ipy is not None:
        _ipy.run_line_magic('matplotlib', 'inline')
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display

warnings.filterwarnings('ignore')

try:
    from google.cloud import bigquery
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-cloud-bigquery", "--quiet"])
    from google.cloud import bigquery

try:
    from polygon import RESTClient as PolygonClient
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "polygon-api-client", "--quiet"])
    from polygon import RESTClient as PolygonClient

try:
    from google.colab import auth
    try:
        auth.authenticate_user()
    except Exception:
        pass
except (ImportError, Exception):
    pass


# ==============================================================================
# CONFIG
# ==============================================================================
BENCHMARK_DATES = ['2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11']
BENCHMARK_NAME = "baseline_v3v4_sep8_11"

POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
BQ_PREMARKET_TABLE = "ys-time-series.scoring_logs.premarket_scanner"
BQ_DISCOVERY_TABLE = "ys-time-series.scoring_logs.discovery_scanner"

USE_LATEST_VERSION_ONLY = True
SELECTED_ONLY = False

HORIZONS_MIN = [30, 60]
INTRADAY_MINUTES = 390
MARKET_OPEN_ET = "09:30"
MARKET_CLOSE_ET = "16:00"
STOP_LOSS_PCT = 2.5

ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')

try:
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    OUTPUT_DIR = os.getcwd()

# Version preference per date (latest version for each day)
PREFERRED_VERSIONS = {
    '2026-09-08': 'V3',
    '2026-09-09': 'V3',
    '2026-09-10': 'V4.0',
    '2026-09-11': 'V4.0',
}

# Strategy registry — each fix is cumulative, add new strategies here
STRATEGIES = [
    {'key': 'baseline', 'label': 'V4 Baseline', 'short': 'Base',
     'color': '#2ecc71', 'marker': 'o', 'sel_col': 'strat_baseline'},
    {'key': 'fix3', 'label': '#3 informed', 'short': '#3',
     'color': '#f39c12', 'marker': 'D', 'sel_col': 'strat_fix3'},
    {'key': 'fix3_4', 'label': '#3+#4 NEUTRAL', 'short': '#3+4',
     'color': '#9b59b6', 'marker': 's', 'sel_col': 'strat_fix3_4'},
]
NEUTRAL_PROXY_CATALYSTS = {'unknown', '', 'nan', 'none'}


# ==============================================================================
# FETCH PREDICTIONS FROM BQ
# ==============================================================================
print(f"\n{'='*100}")
print(f"  SCANNER BENCHMARK BACKTEST — {BENCHMARK_NAME}")
print(f"  Dates: {', '.join(BENCHMARK_DATES)}")
print(f"  SL={STOP_LOSS_PCT}% | Latest version per day: {USE_LATEST_VERSION_ONLY}")
print(f"{'='*100}")

bq_client = bigquery.Client()
_dates_sql = ", ".join(f"'{d}'" for d in BENCHMARK_DATES)


def fetch_predictions(table, scanner_name):
    if scanner_name == 'premarket':
        q = f"""
        SELECT
            CAST(scan_date AS STRING) AS eval_date,
            ticker, direction, confidence, remaining_move_pct,
            move_already_absorbed_pct AS move_already_priced_pct,
            expected_move_pct, composite_score AS tradability_score,
            catalyst_type, catalyst, selected, scanner_version,
            scan_time_pt, gate_reason, move_is_exhausted,
            scan_score, 0 AS catalyst_strength, 0.0 AS scan_move_score,
            rank_in_direction,
            COALESCE(stage2_model, '') AS stage2_model
        FROM `{table}`
        WHERE scan_date IN ({_dates_sql})
          AND direction IN ('LONG', 'SHORT')
        ORDER BY scan_date, direction, rank_in_direction
        """
    else:
        q = f"""
        SELECT
            CAST(scan_date AS STRING) AS eval_date,
            ticker, direction, confidence, remaining_move_pct,
            move_already_priced_pct,
            expected_move_pct, tradability_score,
            catalyst_type, catalyst, selected, scanner_version,
            scan_time_pt, gate_reason, move_is_exhausted,
            0 AS scan_score, catalyst_strength, scan_move_score,
            rank_in_direction,
            COALESCE(stage2_model, '') AS stage2_model
        FROM `{table}`
        WHERE scan_date IN ({_dates_sql})
          AND direction IN ('LONG', 'SHORT')
        ORDER BY scan_date, direction, rank_in_direction
        """
    df = bq_client.query(q).to_dataframe()
    df['scanner'] = scanner_name
    return df


print(f"\n  Fetching predictions from BQ...")
df_pre = fetch_predictions(BQ_PREMARKET_TABLE, 'premarket')
df_disc = fetch_predictions(BQ_DISCOVERY_TABLE, 'discovery')

print(f"    Premarket: {len(df_pre)} signals ({df_pre['selected'].sum()} selected)")
print(f"    Discovery: {len(df_disc)} signals ({df_disc['selected'].sum()} selected)")

df_all = pd.concat([df_pre, df_disc], ignore_index=True)

# Clean up gate_reason: normalize None/nan strings to empty
df_all['gate_reason'] = df_all['gate_reason'].fillna('')
df_all.loc[df_all['gate_reason'].isin(['None', 'none', 'nan']), 'gate_reason'] = ''

# Filter to latest version per day if configured — with fallback
if USE_LATEST_VERSION_ONLY:
    mask = pd.Series(True, index=df_all.index)
    _effective_versions = {}
    for date, ver in PREFERRED_VERSIONS.items():
        date_rows = df_all[df_all['eval_date'] == date]
        available_versions = sorted(date_rows['scanner_version'].dropna().unique())
        if ver in available_versions:
            _effective_versions[date] = ver
        elif available_versions:
            fallback = available_versions[-1]
            _effective_versions[date] = fallback
            print(f"    ⚠ {date}: preferred {ver} not found, falling back to {fallback} "
                  f"(available: {', '.join(available_versions)})")
        else:
            _effective_versions[date] = ver
    for date, ver in _effective_versions.items():
        date_mask = df_all['eval_date'] == date
        ver_mask = df_all['scanner_version'] == ver
        mask = mask & (~date_mask | ver_mask)
    n_before = len(df_all)
    df_all = df_all[mask].copy()
    print(f"    Version filter: {n_before} → {len(df_all)} signals")
    for date in sorted(df_all['eval_date'].unique()):
        versions = df_all[df_all['eval_date'] == date]['scanner_version'].unique()
        n_date = len(df_all[df_all['eval_date'] == date])
        n_sel = df_all[(df_all['eval_date'] == date) & (df_all['selected'] == True)].shape[0]
        print(f"      {date}: {', '.join(versions)} — {n_date} signals, {n_sel} selected")

if SELECTED_ONLY:
    n_before = len(df_all)
    df_all = df_all[df_all['selected'] == True].copy()
    print(f"    Selected only: {n_before} → {len(df_all)} signals")


# ==============================================================================
# PARSE SCAN TIME → ENTRY TIME
# ==============================================================================
def _parse_scan_time(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if len(s) == 5 and ':' in s:
        return s
    if len(s) >= 5 and s[2] == ':':
        return s[:5]
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(s[:19], fmt).strftime('%H:%M')
        except (ValueError, IndexError):
            continue
    return s[:5] if len(s) >= 5 else None


df_all['scan_time_hhmm'] = df_all['scan_time_pt'].apply(_parse_scan_time)

# Diagnostics: show how many signals per date have missing scan_time
_null_scan = df_all[df_all['scan_time_hhmm'].isna()]
if len(_null_scan) > 0:
    for dt in sorted(_null_scan['eval_date'].astype(str).unique()):
        n_null = len(_null_scan[_null_scan['eval_date'].astype(str) == dt])
        print(f"    ⚠ {dt}: {n_null} signals have NULL scan_time_pt → defaulting to pre-market (entry=09:30 ET)")

# Fallback: missing scan_time → assume pre-market scan (06:30 PT = 09:30 ET open)
df_all['scan_time_hhmm'] = df_all['scan_time_hhmm'].fillna('06:30')


def _scan_pt_to_entry_et(scan_pt_hhmm, eval_date):
    pt_dt = PT.localize(datetime.strptime(f"{eval_date} {scan_pt_hhmm}", '%Y-%m-%d %H:%M'))
    et_dt = pt_dt.astimezone(ET)
    et_hhmm = et_dt.strftime('%H:%M')
    if et_hhmm < MARKET_OPEN_ET:
        return MARKET_OPEN_ET
    if et_hhmm >= MARKET_CLOSE_ET:
        return MARKET_OPEN_ET
    return et_hhmm


df_all['eval_date'] = df_all['eval_date'].astype(str)
df_all['entry_time_et'] = df_all.apply(
    lambda r: _scan_pt_to_entry_et(r['scan_time_hhmm'], r['eval_date']), axis=1)

# Diagnostic: count after-hours scans that got mapped to open
_mapped_to_open = df_all[df_all['entry_time_et'] == MARKET_OPEN_ET]
for dt in sorted(df_all['eval_date'].unique()):
    dt_rows = df_all[df_all['eval_date'] == dt]
    dt_open = dt_rows[dt_rows['entry_time_et'] == MARKET_OPEN_ET]
    if len(dt_open) > 0:
        _pre_or_after = dt_rows[dt_rows['scan_time_hhmm'].apply(
            lambda x: x < '06:30' or x >= '13:00')]
        if len(_pre_or_after) > 0:
            print(f"    {dt}: {len(_pre_or_after)} pre/after-market scans → entry at 09:30 ET")


# Cross-scanner direction conflict: gate tickers that are LONG in one scanner, SHORT in another
_cross_conflict_tickers = set()
_ticker_directions = defaultdict(lambda: defaultdict(set))
for _, r in df_all.iterrows():
    if bool(r['selected']) if pd.notna(r.get('selected')) else False:
        _ticker_directions[r['eval_date']][r['ticker']].add(r['direction'])
for dt in _ticker_directions:
    for tk, dirs in _ticker_directions[dt].items():
        if 'LONG' in dirs and 'SHORT' in dirs:
            _cross_conflict_tickers.add((tk, dt))

if _cross_conflict_tickers:
    for tk, dt in _cross_conflict_tickers:
        print(f"    CONFLICT: {tk} on {dt} — both LONG+SHORT, gating both")
    conflict_mask = df_all.apply(
        lambda r: (r['ticker'], r['eval_date']) in _cross_conflict_tickers, axis=1)
    df_all.loc[conflict_mask, 'selected'] = False
    df_all.loc[conflict_mask, 'gate_reason'] = 'CROSS_SCANNER_DIRECTION_CONFLICT'

# Per-ticker per-day dedup: prefer selected signal (earliest), fallback to first appearance
df_sel_first = (df_all[df_all['selected'] == True]
                .sort_values('scan_time_hhmm')
                .drop_duplicates(['ticker', 'eval_date'], keep='first'))
_sel_keys = set(zip(df_sel_first['ticker'], df_sel_first['eval_date']))

df_unsel_first = (df_all[df_all.apply(
                      lambda r: (r['ticker'], r['eval_date']) not in _sel_keys, axis=1)]
                  .sort_values('scan_time_hhmm')
                  .drop_duplicates(['ticker', 'eval_date'], keep='first'))

df_all = pd.concat([df_sel_first, df_unsel_first]).reset_index(drop=True)

n_selected = int(df_all['selected'].sum())
n_not_selected = len(df_all) - n_selected
print(f"\n  After dedup: {len(df_all)} unique (ticker, date) positions "
      f"({n_selected} selected, {n_not_selected} not selected)")
for dt in sorted(df_all['eval_date'].unique()):
    df_dt = df_all[df_all['eval_date'] == dt]
    n_sel_dt = int(df_dt['selected'].sum())
    n_long = (df_dt['direction'] == 'LONG').sum()
    n_short = (df_dt['direction'] == 'SHORT').sum()
    print(f"    {dt}: {len(df_dt)} total ({n_sel_dt} sel, {n_long}L {n_short}S)")


# ==============================================================================
# FETCH POLYGON PRICE DATA
# ==============================================================================
poly = PolygonClient(POLYGON_API_KEY)


def fetch_minute_bars(ticker, eval_date):
    try:
        bars = list(poly.get_aggs(
            ticker=ticker, multiplier=1, timespan="minute",
            from_=eval_date, to=eval_date, limit=500,
        ))
        if not bars:
            return None, None

        bar_map = {}
        for b in bars:
            ts = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').tz_convert(ET)
            key = ts.strftime('%H:%M')
            bar_map[key] = {
                'open': float(b.open), 'high': float(b.high),
                'low': float(b.low), 'close': float(b.close),
                'volume': int(b.volume),
            }

        prev_close = None
        try:
            from_date = (datetime.strptime(eval_date, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
            daily = list(poly.get_aggs(
                ticker=ticker, multiplier=1, timespan="day",
                from_=from_date, to=eval_date, limit=10,
            ))
            ref_dt = datetime.strptime(eval_date, '%Y-%m-%d')
            prev = [b for b in daily
                    if pd.Timestamp(b.timestamp, unit='ms', tz='UTC').date() < ref_dt.date()]
            if prev:
                prev_close = float(prev[-1].close)
        except Exception:
            pass

        return bar_map, prev_close

    except Exception as e:
        print(f"      [ERROR] {ticker}: {e}")
        return None, None


ticker_date_pairs = df_all[['ticker', 'eval_date']].drop_duplicates().values.tolist()
# Add SPY for each date
spy_dates = list(set(r[1] for r in ticker_date_pairs))

print(f"\n  Fetching Polygon 1-min bars for {len(ticker_date_pairs)} trades + SPY...")

bar_data = {}
for i, (ticker, eval_date) in enumerate(ticker_date_pairs):
    bar_map, prev_close = fetch_minute_bars(ticker, eval_date)
    if bar_map:
        bar_data[(ticker, eval_date)] = {'bars': bar_map, 'prev_close': prev_close}
        n_bars = len(bar_map)
        print(f"    {i+1:>3}/{len(ticker_date_pairs)} {ticker:<6} {eval_date} {n_bars} bars")
    else:
        print(f"    {i+1:>3}/{len(ticker_date_pairs)} {ticker:<6} {eval_date} NO DATA")

# Fetch SPY for alpha calculation
spy_bars = {}
for dt in spy_dates:
    bm, _ = fetch_minute_bars('SPY', dt)
    if bm:
        spy_bars[dt] = bm
        print(f"    SPY {dt}: {len(bm)} bars")

print(f"\n  Got price data for {len(bar_data)}/{len(ticker_date_pairs)} trades")


def _spy_return(entry_hhmm, horizon_min, eval_date):
    bm = spy_bars.get(eval_date)
    if not bm:
        return np.nan
    entry_bar = bm.get(entry_hhmm)
    if not entry_bar:
        _cands = sorted(k for k in bm if k >= entry_hhmm)
        if not _cands:
            return np.nan
        entry_bar = bm[_cands[0]]
        entry_hhmm = _cands[0]
    entry_p = entry_bar['open']
    eh, em = map(int, entry_hhmm.split(':'))
    total_m = eh * 60 + em + horizon_min
    th, tm = total_m // 60, total_m % 60
    target = f"{th:02d}:{tm:02d}"
    if target >= MARKET_CLOSE_ET:
        target = sorted(bm.keys())[-1]
    t_bar = bm.get(target)
    if not t_bar:
        return np.nan
    return round(((t_bar['close'] - entry_p) / entry_p) * 100, 4)


# ==============================================================================
# COMPUTE PnL — per-trade stop-loss simulation
# ==============================================================================
records = []

for _, row in df_all.iterrows():
    ticker = row['ticker']
    direction = row['direction']
    entry_hhmm = row['entry_time_et']
    eval_date = row['eval_date']

    td = bar_data.get((ticker, eval_date))
    if td is None:
        continue

    bar_map = td['bars']
    prev_close = td['prev_close']

    actual_entry_hhmm = entry_hhmm
    entry_bar = bar_map.get(entry_hhmm)
    if not entry_bar:
        candidates = sorted(k for k in bar_map if k >= entry_hhmm)
        if not candidates:
            continue
        actual_entry_hhmm = candidates[0]
        entry_bar = bar_map[actual_entry_hhmm]

    entry_price = entry_bar['open']
    entry_h, entry_m_val = map(int, actual_entry_hhmm.split(':'))
    sign = 1.0 if direction == 'LONG' else -1.0

    sl_curve = {}
    stopped_out = False
    stop_minute = None

    for m in range(INTRADAY_MINUTES + 1):
        total_min = entry_h * 60 + entry_m_val + m
        target_h = total_min // 60
        target_m = total_min % 60
        target_hhmm = f"{target_h:02d}:{target_m:02d}"

        if target_hhmm >= MARKET_CLOSE_ET:
            break

        bar = bar_map.get(target_hhmm)
        if bar is None:
            if m > 0 and (m - 1) in sl_curve:
                sl_curve[m] = -STOP_LOSS_PCT if stopped_out else sl_curve[m - 1]
            continue

        ret = ((bar['close'] - entry_price) / entry_price) * 100
        raw_pnl = ret * sign

        if stopped_out:
            sl_curve[m] = -STOP_LOSS_PCT
            continue

        if direction == 'LONG':
            worst = ((bar['low'] - entry_price) / entry_price) * 100
        else:
            worst = -((bar['high'] - entry_price) / entry_price) * 100

        if worst <= -STOP_LOSS_PCT:
            stopped_out = True
            stop_minute = m
            sl_curve[m] = -STOP_LOSS_PCT
        else:
            sl_curve[m] = round(raw_pnl, 4)

    if not sl_curve:
        continue

    all_bars_sorted = sorted(bar_map.items(), key=lambda x: x[0])
    eod_price = all_bars_sorted[-1][1]['close'] if all_bars_sorted else entry_price
    eod_ret = round(((eod_price - entry_price) / entry_price) * 100 * sign, 2)

    gap_pct = round(((entry_price - prev_close) / prev_close) * 100, 2) if prev_close else 0.0

    max_m = max(sl_curve.keys()) if sl_curve else 0

    _selected = bool(row['selected']) if pd.notna(row.get('selected')) else False
    _gate = str(row.get('gate_reason', '')) if pd.notna(row.get('gate_reason')) else ''
    if _gate in ('nan', 'None', 'none'):
        _gate = ''

    rec = {
        'ticker': ticker,
        'eval_date': eval_date,
        'scanner': row['scanner'],
        'scanner_version': row['scanner_version'],
        'direction': direction,
        'confidence': int(row['confidence']),
        'catalyst_type': str(row.get('catalyst_type', '')),
        'selected': _selected,
        'gate_reason': _gate,
        'entry_time_et': actual_entry_hhmm,
        'entry_price': entry_price,
        'gap_pct': gap_pct,
        'sl_curve': sl_curve,
        'stopped_out': stopped_out,
        'stop_minute': stop_minute,
        'max_minutes': max_m,
        'sl_pnl_eod': -STOP_LOSS_PCT if stopped_out else eod_ret,
    }

    for h in HORIZONS_MIN:
        rec[f'sl_pnl_{h}m'] = sl_curve.get(h, np.nan) if h <= max_m else np.nan
        spy_h = _spy_return(actual_entry_hhmm, h, eval_date)
        rec[f'SPY_{h}m'] = spy_h
        if pd.notna(rec[f'sl_pnl_{h}m']) and pd.notna(spy_h):
            rec[f'alpha_{h}m'] = round(rec[f'sl_pnl_{h}m'] - spy_h * sign, 4)
        else:
            rec[f'alpha_{h}m'] = np.nan

    records.append(rec)

df_pnl = pd.DataFrame(records)
df_pnl = df_pnl[df_pnl['max_minutes'] >= 30].copy()

print(f"\n  {len(df_pnl)} trades with 30m+ data")

if df_pnl.empty:
    print("  No trades to evaluate. Check API key / data availability.")
    sys.exit(0)


# ==============================================================================
# FABLE #3: Compute new informed_score + re-selection simulation
# (run before charts so ALL visualizations compare baseline vs #3 fix)
# ==============================================================================
print(f"\n  Computing Fable #3 re-selection simulation...")

_ab_bq_q = f"""
SELECT CAST(scan_date AS STRING) AS eval_date, ticker, direction, selected,
       ev, informed_score AS bq_informed, tape_discount AS bq_tape,
       gap_at_entry, residual_mid, residual_clarity, ev_penalty,
       rank_in_direction, gate_reason
FROM `{BQ_DISCOVERY_TABLE}`
WHERE scan_date IN ({_dates_sql})
  AND direction IN ('LONG', 'SHORT')
UNION ALL
SELECT CAST(scan_date AS STRING) AS eval_date, ticker, direction, selected,
       ev, informed_score AS bq_informed, tape_discount AS bq_tape,
       gap_at_entry, residual_mid, residual_clarity, ev_penalty,
       rank_in_direction, gate_reason
FROM `{BQ_PREMARKET_TABLE}`
WHERE scan_date IN ({_dates_sql})
  AND direction IN ('LONG', 'SHORT')
"""
df_ab_bq = bq_client.query(_ab_bq_q).to_dataframe()

if USE_LATEST_VERSION_ONLY:
    _pnl_keys = set(zip(df_pnl['ticker'], df_pnl['eval_date']))
    df_ab_bq = df_ab_bq[df_ab_bq.apply(
        lambda r: (r['ticker'], r['eval_date']) in _pnl_keys, axis=1)].copy()

_ab_records = []
for _, row in df_pnl.iterrows():
    ticker = row['ticker']
    eval_date = row['eval_date']
    td = bar_data.get((ticker, eval_date))
    if td is None:
        continue

    gap_pct = row['gap_pct']
    bar_map = td.get('bars', {})
    pm_vol_sum = sum(b.get('volume', 0) for k, b in bar_map.items() if k < '09:30')
    total_vol = sum(b.get('volume', 0) for k, b in bar_map.items())

    try:
        from_20d = (datetime.strptime(eval_date, '%Y-%m-%d') - timedelta(days=25)).strftime('%Y-%m-%d')
        yesterday = (datetime.strptime(eval_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        daily_bars = list(poly.get_aggs(
            ticker=ticker, multiplier=1, timespan="day",
            from_=from_20d, to=yesterday, limit=20,
        ))
        if daily_bars:
            avg_daily_vol = int(sum(int(getattr(b, 'volume', 0) or 0) for b in daily_bars) / len(daily_bars))
            closes = [float(b.close) for b in daily_bars if float(b.close) > 0]
            if len(closes) >= 3:
                rets = [(closes[i] - closes[i-1]) / closes[i-1] * 100 for i in range(1, len(closes))]
                daily_ret_std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 2.0
            else:
                daily_ret_std = 2.0
        else:
            avg_daily_vol = total_vol
            daily_ret_std = 2.0
    except Exception:
        avg_daily_vol = total_vol
        daily_ret_std = 2.0

    old_vol_ratio = pm_vol_sum / avg_daily_vol if avg_daily_vol > 0 else 0
    old_vol_signal = min(1.0, old_vol_ratio / 3.0)
    old_idio_signal = min(1.0, abs(gap_pct) / 5.0)
    old_informed = old_vol_signal * 0.6 + old_idio_signal * 0.4

    pm_baseline = avg_daily_vol * 0.02 if avg_daily_vol > 0 else 1
    new_vol_ratio = pm_vol_sum / pm_baseline if pm_baseline > 0 else 0
    gap_z = abs(gap_pct) / daily_ret_std if daily_ret_std > 0.5 else abs(gap_pct) / 2.0
    new_vol_signal = min(1.0, np.log1p(new_vol_ratio) / np.log1p(10))
    new_gap_signal = min(1.0, gap_z / 3.0)
    new_informed = new_vol_signal * 0.6 + new_gap_signal * 0.4

    bq_match = df_ab_bq[(df_ab_bq['ticker'] == ticker) & (df_ab_bq['eval_date'] == eval_date)]
    if len(bq_match) > 0:
        bq_row = bq_match.iloc[0]
        residual_mid = float(bq_row.get('residual_mid', 0))
        residual_clarity = str(bq_row.get('residual_clarity', 'ambiguous'))
        ev_penalty = str(bq_row.get('ev_penalty', ''))
    else:
        residual_mid = 0
        residual_clarity = 'ambiguous'
        ev_penalty = ''

    clarity_mult = {'clear': 1.0, 'lean': 0.6, 'ambiguous': 0.2}.get(residual_clarity, 0.2)

    catalyst_dir = row['direction']
    tape_opposes = (
        (catalyst_dir == 'LONG' and gap_pct < -1.0) or
        (catalyst_dir == 'SHORT' and gap_pct > 1.0)
    )
    old_tape = 1.0
    if tape_opposes and old_informed > 0.5:
        old_tape = 1.0 - (old_informed - 0.5) * 0.6
    new_tape = 1.0
    if tape_opposes and new_informed > 0.3:
        new_tape = max(0.0, 1.0 - (new_informed - 0.3) * 1.43)

    base_ev = abs(residual_mid) * clarity_mult
    ev_mult = 1.0
    if ev_penalty:
        if 'self_contradiction' in ev_penalty:
            ev_mult *= 0.3
        if 'dir_residual_conflict' in ev_penalty:
            ev_mult *= 0.5
        if 'immaterial_deal' in ev_penalty:
            ev_mult *= 0.2
        if 'vague_analyst' in ev_penalty:
            ev_mult *= 0.3
        if 'pre_earnings' in ev_penalty:
            ev_mult *= 0.1
    old_ev = base_ev * old_tape * ev_mult
    new_ev = base_ev * new_tape * ev_mult

    _ab_records.append({
        'ticker': ticker, 'eval_date': eval_date,
        'direction': row['direction'], 'selected': bool(row['selected']),
        'catalyst_type': str(row.get('catalyst_type', '')),
        'gap_pct': gap_pct, 'pm_vol': pm_vol_sum, 'avg_daily_vol': avg_daily_vol,
        'daily_ret_std': daily_ret_std, 'gap_z': gap_z,
        'old_informed': round(old_informed, 3), 'new_informed': round(new_informed, 3),
        'old_tape': round(old_tape, 3), 'new_tape': round(new_tape, 3),
        'old_ev': round(old_ev, 3), 'new_ev': round(new_ev, 3),
        'residual_mid': residual_mid, 'residual_clarity': residual_clarity,
        'sl_pnl_30m': row.get('sl_pnl_30m', np.nan),
        'sl_pnl_60m': row.get('sl_pnl_60m', np.nan),
        'sl_pnl_eod': row.get('sl_pnl_eod', np.nan),
        'stopped_out': row.get('stopped_out', False),
        'sl_curve': row.get('sl_curve', {}),
    })

df_ab = pd.DataFrame(_ab_records)

if len(df_ab) > 0:
    _sel_counts = df_ab[df_ab['selected'] == True].groupby(['eval_date', 'direction']).size().to_dict()
    df_ab['new_selected'] = False
    for (dt, dirn), n_sel in _sel_counts.items():
        mask = (df_ab['eval_date'] == dt) & (df_ab['direction'] == dirn)
        sub = df_ab[mask].sort_values('new_ev', ascending=False)
        top_idx = sub.head(n_sel).index
        df_ab.loc[top_idx, 'new_selected'] = True

    _new_sel_keys = set(zip(df_ab[df_ab['new_selected'] == True]['ticker'],
                            df_ab[df_ab['new_selected'] == True]['eval_date']))
    df_pnl['new_selected'] = df_pnl.apply(
        lambda r: (r['ticker'], r['eval_date']) in _new_sel_keys, axis=1)

    # --- #4 simulation: from #3 pool, exclude unknown-catalyst trades, re-select ---
    _cat_lower = df_ab['catalyst_type'].fillna('').str.lower().str.strip()
    df_ab['eligible_fix4'] = ~_cat_lower.isin(NEUTRAL_PROXY_CATALYSTS)
    df_ab['new_selected_4'] = False
    for (dt, dirn), n_sel in _sel_counts.items():
        mask = (df_ab['eval_date'] == dt) & (df_ab['direction'] == dirn) & (df_ab['eligible_fix4'])
        sub = df_ab[mask].sort_values('new_ev', ascending=False)
        top_idx = sub.head(n_sel).index
        df_ab.loc[top_idx, 'new_selected_4'] = True
    _fix4_keys = set(zip(df_ab[df_ab['new_selected_4']]['ticker'],
                         df_ab[df_ab['new_selected_4']]['eval_date']))
    df_pnl['new_selected_4'] = df_pnl.apply(
        lambda r: (r['ticker'], r['eval_date']) in _fix4_keys, axis=1)

    # Map strategy columns
    df_pnl['strat_baseline'] = df_pnl['selected']
    df_pnl['strat_fix3'] = df_pnl['new_selected']
    df_pnl['strat_fix3_4'] = df_pnl['new_selected_4']

    _n_neutral_proxy = int((~df_ab['eligible_fix4']).sum())
    print(f"    Baseline: {int(df_pnl['strat_baseline'].sum())} | "
          f"#3: {int(df_pnl['strat_fix3'].sum())} | "
          f"#3+#4: {int(df_pnl['strat_fix3_4'].sum())} "
          f"({_n_neutral_proxy} unknown-catalyst removed)")
else:
    df_pnl['new_selected'] = df_pnl['selected']
    df_pnl['new_selected_4'] = df_pnl['selected']
    df_pnl['strat_baseline'] = df_pnl['selected']
    df_pnl['strat_fix3'] = df_pnl['selected']
    df_pnl['strat_fix3_4'] = df_pnl['selected']
    print(f"    No A/B data — using baseline selection for all strategies")


# ==============================================================================
# PORTFOLIO METRICS — helpers
# ==============================================================================
def build_portfolio_curve(df_subset, curve_col='sl_curve'):
    port = {}
    for m in range(INTRADAY_MINUTES + 1):
        vals = [r[curve_col].get(m) for _, r in df_subset.iterrows()
                if r[curve_col].get(m) is not None]
        if vals:
            port[m] = np.mean(vals)
    return port


def _h_stats(df_sub, h):
    v = df_sub[f'sl_pnl_{h}m'].dropna()
    if len(v) == 0:
        return {'avg': np.nan, 'wr': np.nan, 'se': np.nan, 'n': 0, 'med': np.nan}
    return {
        'avg': float(v.mean()), 'med': float(v.median()),
        'wr': float((v > 0).sum() / len(v) * 100),
        'se': float(v.std() / np.sqrt(len(v))) if len(v) > 1 else 0.0,
        'n': len(v),
    }


# ==============================================================================
# ANALYSIS
# ==============================================================================
# Per-strategy subsets
strat_dfs = {}
for s in STRATEGIES:
    strat_dfs[s['key']] = df_pnl[df_pnl[s['sel_col']] == True].copy()

df_sel = strat_dfs['baseline']
df_not_sel = df_pnl[df_pnl['selected'] == False].copy()
df_new_sel = strat_dfs['fix3']
df_gated = df_pnl[(df_pnl['gate_reason'] != '') & (df_pnl['selected'] == False)].copy()

n_total = len(df_pnl)
n_sel_total = len(df_sel)
n_not_total = len(df_not_sel)
n_days = len(df_pnl['eval_date'].unique())
n_stopped = int(df_pnl['stopped_out'].sum())
agg = {}

# --- Build daily data ---
daily_results = []
daily_pnl = []
for dt in sorted(df_pnl['eval_date'].unique()):
    df_day = df_pnl[df_pnl['eval_date'] == dt]
    df_day_sel = df_day[df_day['selected'] == True]
    df_day_not = df_day[df_day['selected'] == False]
    n_long = int((df_day['direction'] == 'LONG').sum())
    n_short = int((df_day['direction'] == 'SHORT').sum())
    sel_ret = df_day_sel['sl_pnl_eod'].dropna().mean() if len(df_day_sel) > 0 else np.nan
    not_ret = df_day_not['sl_pnl_eod'].dropna().mean() if len(df_day_not) > 0 else np.nan
    all_ret = df_day['sl_pnl_eod'].dropna().mean()
    spy_ret = _spy_return('09:30', 390, dt)

    day_rec = {
        'date': dt, 'n_trades': len(df_day),
        'n_selected': len(df_day_sel), 'n_not_selected': len(df_day_not),
        'n_long': n_long, 'n_short': n_short, 'n_stopped': int(df_day['stopped_out'].sum()),
    }
    dp = {'date': dt, 'notsel_eod': not_ret, 'all_eod': all_ret,
          'spy_eod': spy_ret if pd.notna(spy_ret) else 0, 'n_all': len(df_day)}
    for s in STRATEGIES:
        df_day_s = df_day[df_day[s['sel_col']] == True]
        s_ret = df_day_s['sl_pnl_eod'].dropna().mean() if len(df_day_s) > 0 else np.nan
        dp[f"{s['key']}_eod"] = s_ret
        dp[f"n_{s['key']}"] = len(df_day_s)
    for h in HORIZONS_MIN:
        s_all = _h_stats(df_day, h)
        s_sel = _h_stats(df_day_sel, h)
        if s_all['n'] > 0:
            day_rec[f'pnl_{h}m'] = round(s_all['avg'], 4)
            day_rec[f'wr_{h}m'] = round(s_all['wr'], 1)
            day_rec[f'n_valid_{h}m'] = s_all['n']
            alpha_v = df_day[f'alpha_{h}m'].dropna()
            day_rec[f'alpha_{h}m'] = round(float(alpha_v.mean()), 4) if len(alpha_v) > 0 else None
        if s_sel['n'] > 0:
            day_rec[f'sel_pnl_{h}m'] = round(s_sel['avg'], 4)
            day_rec[f'sel_wr_{h}m'] = round(s_sel['wr'], 1)
    eod = df_day['sl_pnl_eod'].dropna()
    if len(eod) > 0:
        day_rec['pnl_eod'] = round(float(eod.mean()), 4)
        day_rec['wr_eod'] = round(float((eod > 0).sum() / len(eod) * 100), 1)
    daily_results.append(day_rec)
    daily_pnl.append(dp)

df_daily = pd.DataFrame(daily_pnl)

# Cumulative equity per strategy
cum_strat = {}
strat_total = {}
for s in STRATEGIES:
    col = f"{s['key']}_eod"
    cum_strat[s['key']] = (1 + df_daily[col].fillna(0) / 100).cumprod() * 100
    strat_total[s['key']] = float(cum_strat[s['key']].iloc[-1] - 100) if len(cum_strat[s['key']]) > 0 else 0
cum_all = (1 + df_daily['all_eod'].fillna(0) / 100).cumprod() * 100
cum_not = (1 + df_daily['notsel_eod'].fillna(0) / 100).cumprod() * 100
cum_spy = (1 + df_daily['spy_eod'].fillna(0) / 100).cumprod() * 100
# Backward-compat aliases
sel_total = strat_total['baseline']
fix_total = strat_total['fix3']
all_total = float(cum_all.iloc[-1] - 100) if len(cum_all) > 0 else 0
not_total = float(cum_not.iloc[-1] - 100) if len(cum_not) > 0 else 0
spy_total = float(cum_spy.iloc[-1] - 100) if len(cum_spy) > 0 else 0

# --- Save agg stats ---
for _prefix, _df in [('all_', df_pnl), ('sel_', df_sel), ('notsel_', df_not_sel)]:
    for h in HORIZONS_MIN:
        s = _h_stats(_df, h)
        if s['n'] > 0:
            agg[f'{_prefix}pnl_{h}m'] = round(s['avg'], 4)
            agg[f'{_prefix}median_{h}m'] = round(s['med'], 4)
            agg[f'{_prefix}wr_{h}m'] = round(s['wr'], 1)
            agg[f'{_prefix}se_{h}m'] = round(s['se'], 4)
            agg[f'{_prefix}n_valid_{h}m'] = s['n']
            alpha_v = _df[f'alpha_{h}m'].dropna()
            agg[f'{_prefix}alpha_{h}m'] = round(float(alpha_v.mean()), 4) if len(alpha_v) > 0 else None
    eod = _df['sl_pnl_eod'].dropna()
    if len(eod) > 0:
        agg[f'{_prefix}pnl_eod'] = round(float(eod.mean()), 4)
        agg[f'{_prefix}wr_eod'] = round(float((eod > 0).sum() / len(eod) * 100), 1)

for h in HORIZONS_MIN:
    s_s = _h_stats(df_sel, h)
    s_n = _h_stats(df_not_sel, h)
    if s_s['n'] > 0 and s_n['n'] > 0:
        agg[f'selection_edge_{h}m'] = round(s_s['avg'] - s_n['avg'], 4)

# Confusion matrix @30m and @60m
_cm = {}
for h in HORIZONS_MIN:
    col = f'sl_pnl_{h}m'
    _vh = df_pnl.dropna(subset=[col])
    tp = _vh[(_vh['selected'] == True) & (_vh[col] > 0)]
    fp = _vh[(_vh['selected'] == True) & (_vh[col] <= 0)]
    tn = _vh[(_vh['selected'] == False) & (_vh[col] <= 0)]
    fn = _vh[(_vh['selected'] == False) & (_vh[col] > 0)]
    prec = len(tp) / max(len(tp) + len(fp), 1) * 100
    rec = len(tp) / max(len(tp) + len(fn), 1) * 100
    _cm[h] = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'precision': prec, 'recall': rec,
              'vh': _vh}
    agg.update({f'precision_{h}m': round(prec, 1), f'recall_{h}m': round(rec, 1),
                f'n_tp_{h}m': len(tp), f'n_fp_{h}m': len(fp),
                f'n_tn_{h}m': len(tn), f'n_fn_{h}m': len(fn)})

# Backward compat aliases for 60m
_tp, _fp, _tn, _fn = _cm[60]['tp'], _cm[60]['fp'], _cm[60]['tn'], _cm[60]['fn']
precision, recall = _cm[60]['precision'], _cm[60]['recall']
_v60 = _cm[60]['vh']

# Gate value
for h in HORIZONS_MIN:
    gated_v = df_gated[f'sl_pnl_{h}m'].dropna()
    ungated_v = df_pnl[df_pnl['gate_reason'] == ''][f'sl_pnl_{h}m'].dropna()
    if len(gated_v) > 0 and len(ungated_v) > 0:
        agg[f'gate_value_{h}m'] = round(float(ungated_v.mean() - gated_v.mean()), 4)


# ==============================================================================
# CLEAN OUTPUT — summary tables + charts
# ==============================================================================
_W = 90
_sep = '=' * _W

print(f"\n\n{_sep}")
print(f"  SCANNER BENCHMARK: {BENCHMARK_NAME}")
print(f"  {BENCHMARK_DATES[0]} to {BENCHMARK_DATES[-1]}  |  {n_days} trading days  |  SL={STOP_LOSS_PCT}%")
print(f"{_sep}")

# ── TABLE 1: Overview ──
print(f"\n  1. OVERVIEW")
print(f"  {'':─<{_W-4}}")
print(f"  {'Metric':<30} {'Value':>12}")
print(f"  {'':─<{_W-4}}")
print(f"  {'Total signals':<30} {n_total:>12}")
print(f"  {'  Selected':<30} {n_sel_total:>12}")
print(f"  {'  Not-selected':<30} {n_not_total:>12}")
print(f"  {'Trades/day (selected)':<30} {n_sel_total/n_days:>11.1f}")
print(f"  {'Trades/day (total)':<30} {n_total/n_days:>11.1f}")
print(f"  {'Stopped out':<30} {f'{n_stopped}/{n_total} ({n_stopped/n_total*100:.0f}%)':>12}")
print(f"  {'Cross-scanner conflicts':<30} {len(_cross_conflict_tickers):>12}")

# ── TABLE 2: Performance comparison ──
print(f"\n\n  2. STRATEGY COMPARISON")
print(f"  {'':─<{_W-4}}")
_hdr = f"  {'':20} {'n':>5} {'Avg@30m':>8} {'Avg@60m':>8} {'WR@60m':>7} {'AvgEOD':>8} {'Cum%':>7} {'Alpha':>7}"
print(_hdr)
print(f"  {'':─<{_W-4}}")
for s in STRATEGIES:
    _df = strat_dfs[s['key']]
    _cum = strat_total[s['key']]
    s30 = _h_stats(_df, 30)
    s60 = _h_stats(_df, 60)
    eod_v = _df['sl_pnl_eod'].dropna()
    eod_avg = float(eod_v.mean()) if len(eod_v) > 0 else np.nan
    alpha_cum = float(_cum) - float(spy_total)
    print(f"  {s['label']:<20} {s60['n']:>5} {s30['avg']:>+7.2f}% {s60['avg']:>+7.2f}% "
          f"{s60['wr']:>5.0f}%  {eod_avg:>+7.2f}% {float(_cum):>+6.2f}% {alpha_cum:>+6.2f}%")
for _lbl, _df, _cum in [('NOT-SELECTED', df_not_sel, not_total), ('ALL SIGNALS', df_pnl, all_total)]:
    s30 = _h_stats(_df, 30)
    s60 = _h_stats(_df, 60)
    eod_v = _df['sl_pnl_eod'].dropna()
    eod_avg = float(eod_v.mean()) if len(eod_v) > 0 else np.nan
    alpha_cum = float(_cum) - float(spy_total)
    print(f"  {_lbl:<20} {s60['n']:>5} {s30['avg']:>+7.2f}% {s60['avg']:>+7.2f}% "
          f"{s60['wr']:>5.0f}%  {eod_avg:>+7.2f}% {float(_cum):>+6.2f}% {alpha_cum:>+6.2f}%")
print(f"  {'SPY':<20} {'':>5} {'':>8} {'':>8} {'':>7} {float(spy_total):>+7.2f}% {float(spy_total):>+6.2f}% {'0.00%':>7}")
print(f"  {'':─<{_W-4}}")
_edge = agg.get('selection_edge_60m')
if _edge is not None:
    _verdict = "POSITIVE" if _edge > 0 else "NEGATIVE"
    print(f"  Selection edge @60m: {_edge:+.3f}%  ({_verdict} value)")

# ── TABLE 3: Daily breakdown — all strategies ──
_strat_hdr = '  '.join(f"{'n'+s['short']:>5}" for s in STRATEGIES)
_strat_eod_hdr = '  '.join(f"{s['short']+'EOD':>8}" for s in STRATEGIES)
print(f"\n\n  3. DAILY BREAKDOWN: Strategy Comparison")
print(f"  {'':─<{_W-4}}")
print(f"  {'Date':<12} {_strat_hdr}  {_strat_eod_hdr}  {'SPY':>7}")
print(f"  {'':─<{_W-4}}")
for dp in daily_pnl:
    dt = dp['date']
    ns = '  '.join(f"{dp.get('n_' + s['key'], 0):>5}" for s in STRATEGIES)
    eods = '  '.join(
        f"{dp[s['key'] + '_eod']:+7.2f}%" if pd.notna(dp.get(s['key'] + '_eod')) else '      n/a'
        for s in STRATEGIES)
    spy_e = f"{dp['spy_eod']:+6.2f}%" if dp['spy_eod'] != 0 else '    n/a'
    print(f"  {dt:<12} {ns}  {eods}  {spy_e:>7}")

# ── TABLE 4: Confusion matrix @30m and @60m ──
for h in HORIZONS_MIN:
    cm_h = _cm[h]
    tp_h, fp_h, tn_h, fn_h = cm_h['tp'], cm_h['fp'], cm_h['tn'], cm_h['fn']
    prec_h, rec_h = cm_h['precision'], cm_h['recall']
    col = f'sl_pnl_{h}m'
    print(f"\n\n  4{'a' if h == 30 else 'b'}. SELECTION CONFUSION MATRIX @{h}m")
    print(f"  {'':─<{_W-4}}")
    print(f"  {'':22} {'Profitable':>14} {'Unprofitable':>14} {'Total':>8}")
    print(f"  {'':─<{_W-4}}")
    print(f"  {'Selected':<22} {'TP=' + str(len(tp_h)):>14} {'FP=' + str(len(fp_h)):>14} {len(tp_h)+len(fp_h):>8}")
    print(f"  {'Not-selected':<22} {'FN=' + str(len(fn_h)):>14} {'TN=' + str(len(tn_h)):>14} {len(fn_h)+len(tn_h):>8}")
    print(f"  {'Total':<22} {len(tp_h)+len(fn_h):>14} {len(fp_h)+len(tn_h):>14} {len(cm_h['vh']):>8}")
    print(f"  {'':─<{_W-4}}")
    print(f"  Precision: {prec_h:5.1f}%  (of selected trades, what % was profitable)")
    print(f"  Recall:    {rec_h:5.1f}%  (of profitable trades, what % did we select)")
    if len(tp_h) > 0:
        print(f"  TP avg:   {tp_h[col].mean():+.2f}%   FP avg: {fp_h[col].mean():+.2f}%")
    if len(fn_h) > 0:
        print(f"  FN avg:   {fn_h[col].mean():+.2f}%   TN avg: {tn_h[col].mean():+.2f}%")
        print(f"  FN total left on table: {fn_h[col].sum():+.1f}%  ({len(fn_h)} trades)")

# ── TABLE 5: By direction ──
print(f"\n\n  5. BY DIRECTION")
print(f"  {'':─<{_W-4}}")
print(f"  {'Direction':<10} {'Pool':<12} {'n':>5} {'Avg@60m':>8} {'WR':>5} {'AvgEOD':>8}")
print(f"  {'':─<{_W-4}}")
for d in ['LONG', 'SHORT']:
    for _lbl, _sel_v in [('all', None), ('selected', True), ('not-sel', False)]:
        if _sel_v is None:
            _df = df_pnl[df_pnl['direction'] == d]
        else:
            _df = df_pnl[(df_pnl['direction'] == d) & (df_pnl['selected'] == _sel_v)]
        s = _h_stats(_df, 60)
        if s['n'] == 0:
            continue
        eod_v = _df['sl_pnl_eod'].dropna()
        eod_s = f"{eod_v.mean():+7.2f}%" if len(eod_v) > 0 else '     n/a'
        print(f"  {d if _lbl == 'all' else '':<10} {_lbl:<12} {s['n']:>5} {s['avg']:>+7.2f}% {s['wr']:>4.0f}% {eod_s:>8}")

# ── TABLE 6: By scanner ──
print(f"\n\n  6. BY SCANNER")
print(f"  {'':─<{_W-4}}")
print(f"  {'Scanner':<12} {'Pool':<12} {'n':>5} {'Avg@60m':>8} {'WR':>5}")
print(f"  {'':─<{_W-4}}")
for sc in sorted(df_pnl['scanner'].unique()):
    for _lbl, _sel_v in [('all', None), ('selected', True), ('not-sel', False)]:
        if _sel_v is None:
            _df = df_pnl[df_pnl['scanner'] == sc]
        else:
            _df = df_pnl[(df_pnl['scanner'] == sc) & (df_pnl['selected'] == _sel_v)]
        s = _h_stats(_df, 60)
        if s['n'] == 0:
            continue
        print(f"  {sc if _lbl == 'all' else '':<12} {_lbl:<12} {s['n']:>5} {s['avg']:>+7.2f}% {s['wr']:>4.0f}%")

# ── TABLE 7: By catalyst ──
print(f"\n\n  7. BY CATALYST TYPE")
print(f"  {'':─<{_W-4}}")
print(f"  {'Catalyst':<16} {'n':>4} {'nSel':>5} {'Avg@60':>7} {'SelAvg':>7} {'WR':>5} {'Label':>8}")
print(f"  {'':─<{_W-4}}")
for cat in sorted(df_pnl['catalyst_type'].unique()):
    if not cat or cat == 'nan':
        continue
    df_c = df_pnl[df_pnl['catalyst_type'] == cat]
    v60 = df_c['sl_pnl_60m'].dropna()
    if len(v60) < 2:
        continue
    v60_sel = df_c[df_c['selected'] == True]['sl_pnl_60m'].dropna()
    sel_avg = f"{v60_sel.mean():+6.2f}%" if len(v60_sel) > 0 else '    n/a'
    wr = (v60 > 0).sum() / len(v60) * 100
    label = "TOXIC" if v60.mean() < -1.0 else ("GOLDEN" if v60.mean() > 1.0 else "")
    print(f"  {cat:<16} {len(v60):>4} {len(v60_sel):>5} {v60.mean():>+6.2f}% {sel_avg:>7} {wr:>4.0f}% {label:>8}")

# ── TABLE 8: Top 10 winners + losers (selected) ──
print(f"\n\n  8. TOP SELECTED TRADES @60m")
print(f"  {'':─<{_W-4}}")
_sel60 = df_sel.dropna(subset=['sl_pnl_60m'])
if len(_sel60) > 0:
    print(f"  TOP 5 WINNERS:")
    print(f"  {'Date':<12} {'Ticker':<7} {'Dir':<6} {'@60m':>7} {'EOD':>7} {'Catalyst':<16} {'Scanner'}")
    for _, r in _sel60.nlargest(5, 'sl_pnl_60m').iterrows():
        print(f"  {r['eval_date']:<12} {r['ticker']:<7} {r['direction']:<6} "
              f"{r['sl_pnl_60m']:>+6.2f}% {r['sl_pnl_eod']:>+6.2f}% {str(r['catalyst_type'])[:14]:<16} {r['scanner']}")
    print(f"\n  TOP 5 LOSERS:")
    print(f"  {'Date':<12} {'Ticker':<7} {'Dir':<6} {'@60m':>7} {'EOD':>7} {'Catalyst':<16} {'Scanner'}")
    for _, r in _sel60.nsmallest(5, 'sl_pnl_60m').iterrows():
        print(f"  {r['eval_date']:<12} {r['ticker']:<7} {r['direction']:<6} "
              f"{r['sl_pnl_60m']:>+6.2f}% {r['sl_pnl_eod']:>+6.2f}% {str(r['catalyst_type'])[:14]:<16} {r['scanner']}")

# ── TABLE 9: Top missed opportunities (FN) ──
if len(_fn) > 0:
    print(f"\n\n  9. TOP MISSED OPPORTUNITIES (False Negatives)")
    print(f"  {'':─<{_W-4}}")
    print(f"  {'Date':<12} {'Ticker':<7} {'Dir':<6} {'@60m':>7} {'EOD':>7} {'Catalyst':<16} {'Gate reason'}")
    for _, r in _fn.nlargest(10, 'sl_pnl_60m').iterrows():
        gate = str(r.get('gate_reason', ''))[:25] if r.get('gate_reason') else ''
        print(f"  {r['eval_date']:<12} {r['ticker']:<7} {r['direction']:<6} "
              f"{r['sl_pnl_60m']:>+6.2f}% {r['sl_pnl_eod']:>+6.2f}% {str(r['catalyst_type'])[:14]:<16} {gate}")

# ── TABLE 10: Gate effectiveness ──
if len(df_gated) > 0:
    print(f"\n\n  10. GATE EFFECTIVENESS")
    print(f"  {'':─<{_W-4}}")
    s_gated = _h_stats(df_gated, 60)
    s_ungated = _h_stats(df_pnl[df_pnl['gate_reason'] == ''], 60)
    print(f"  {'Gated signals @60m:':<30} avg={s_gated['avg']:+.2f}%  WR={s_gated['wr']:.0f}%  n={s_gated['n']}")
    print(f"  {'Ungated signals @60m:':<30} avg={s_ungated['avg']:+.2f}%  WR={s_ungated['wr']:.0f}%  n={s_ungated['n']}")
    delta = s_ungated['avg'] - s_gated['avg']
    print(f"  Gate value: {delta:+.3f}%  ({'gates SAVED money' if delta > 0 else 'gates COST money'})")
    print(f"\n  Gate reasons:")
    for gr, cnt in df_gated['gate_reason'].value_counts().head(10).items():
        gr_v = df_gated[df_gated['gate_reason'] == gr]['sl_pnl_60m'].dropna()
        avg_s = f"{gr_v.mean():+.2f}%" if len(gr_v) > 0 else "n/a"
        print(f"    {gr[:40]:<42} n={cnt:>3}  avg@60m={avg_s}")

# ── TABLE 11: Performance by entry hour ──
# Use 1-hour bins (matches Chart 5)
df_pnl['entry_hour'] = df_pnl['entry_time_et'].apply(
    lambda x: f"{int(x.split(':')[0]):02d}:00" if pd.notna(x) else None)
_hour_bins_tbl = sorted(df_pnl['entry_hour'].dropna().unique())

print(f"\n\n  11. PERFORMANCE BY ENTRY HOUR")
print(f"  {'':─<{_W-4}}")
print(f"  {'Hour(ET)':<10} {'n':>4} {'nSel':>5} {'nL':>4} {'nS':>4} "
      f"{'Sel@60':>7} {'Not@60':>7} {'L@60':>7} {'S@60':>7} {'WR':>5} {'EOD':>7}")
print(f"  {'':─<{_W-4}}")
for hb in _hour_bins_tbl:
    df_hb = df_pnl[df_pnl['entry_hour'] == hb]
    if len(df_hb) < 2:
        continue
    df_hb_sel = df_hb[df_hb['selected'] == True]
    df_hb_not = df_hb[df_hb['selected'] == False]
    df_hb_l = df_hb[df_hb['direction'] == 'LONG']
    df_hb_s = df_hb[df_hb['direction'] == 'SHORT']
    s_sel = _h_stats(df_hb_sel, 60)
    s_not = _h_stats(df_hb_not, 60)
    s_l = _h_stats(df_hb_l, 60)
    s_s = _h_stats(df_hb_s, 60)
    s_all = _h_stats(df_hb, 60)
    eod_v = df_hb['sl_pnl_eod'].dropna()
    sel60 = f"{s_sel['avg']:+6.2f}%" if s_sel['n'] > 0 else '    n/a'
    not60 = f"{s_not['avg']:+6.2f}%" if s_not['n'] > 0 else '    n/a'
    l60 = f"{s_l['avg']:+6.2f}%" if s_l['n'] > 0 else '    n/a'
    s60 = f"{s_s['avg']:+6.2f}%" if s_s['n'] > 0 else '    n/a'
    wr60 = f"{s_all['wr']:>4.0f}%" if s_all['n'] > 0 else '  n/a'
    eod_s = f"{eod_v.mean():+6.2f}%" if len(eod_v) > 0 else '    n/a'
    print(f"  {hb:<10} {len(df_hb):>4} {len(df_hb_sel):>5} {len(df_hb_l):>4} {len(df_hb_s):>4} "
          f"{sel60:>7} {not60:>7} {l60:>7} {s60:>7} {wr60:>5} {eod_s:>7}")


# ==============================================================================
# CHARTS
# ==============================================================================
plt.rcParams.update({'figure.max_open_warning': 0})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 1: Equity & Daily Returns (2x2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig1, axes1 = plt.subplots(2, 2, figsize=(18, 12))
_strat_summary = ' | '.join(f"{s['short']}={len(strat_dfs[s['key']])}" for s in STRATEGIES)
fig1.suptitle(f'Scanner Benchmark: {BENCHMARK_NAME}  |  '
              f'{BENCHMARK_DATES[0]} to {BENCHMARK_DATES[-1]}  |  '
              f'{_strat_summary}  |  SL={STOP_LOSS_PCT}%',
              fontsize=13, fontweight='bold')

# 1a. Cumulative equity — all strategies
ax = axes1[0, 0]
_ls_cycle = ['-', '--', '-.']
for i, s in enumerate(STRATEGIES):
    c = cum_strat[s['key']]
    ax.plot(range(len(c)), c, s['color'], lw=2.5, ls=_ls_cycle[i % len(_ls_cycle)],
            label=f"{s['label']} ({strat_total[s['key']]:+.1f}%)")
ax.plot(range(len(cum_not)), cum_not, '#e74c3c', lw=1, alpha=0.5, label=f'Not-sel ({not_total:+.1f}%)')
ax.plot(range(len(cum_spy)), cum_spy, 'gray', lw=1.5, ls=':', label=f'SPY ({spy_total:+.1f}%)')
ax.set_xticks(range(len(df_daily)))
ax.set_xticklabels(df_daily['date'].apply(lambda x: x[-5:]), rotation=45)
ax.set_ylabel('Equity ($100 start)')
ax.set_title('1. Cumulative Equity')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(y=100, color='black', lw=0.5, ls=':')

# 1b. Daily returns grouped bar — all strategies
ax = axes1[0, 1]
x = np.arange(len(df_daily))
n_bars = len(STRATEGIES) + 2
w = 0.8 / n_bars
for i, s in enumerate(STRATEGIES):
    offset = (i - n_bars/2 + 0.5) * w
    col = f"{s['key']}_eod"
    ax.bar(x + offset, df_daily[col].fillna(0), w, label=s['label'], color=s['color'], alpha=0.85)
ax.bar(x + (len(STRATEGIES) - n_bars/2 + 0.5) * w, df_daily['notsel_eod'].fillna(0), w,
       label='Not-sel', color='#e74c3c', alpha=0.4)
ax.bar(x + (len(STRATEGIES) + 1 - n_bars/2 + 0.5) * w, df_daily['spy_eod'].fillna(0), w,
       label='SPY', color='gray', alpha=0.4)
ax.set_xticks(range(len(df_daily)))
ax.set_xticklabels(df_daily['date'].apply(lambda x: x[-5:]), rotation=45)
ax.set_ylabel('Daily Return %')
ax.set_title('2. Daily Returns')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', lw=0.8)

# 1c. Intraday curves — all strategies
ax = axes1[1, 0]
for i, s in enumerate(STRATEGIES):
    _sdf = strat_dfs[s['key']]
    if len(_sdf) > 0:
        c = build_portfolio_curve(_sdf)
        xs = sorted(c.keys())
        ax.plot(xs, [c[x] for x in xs], s['color'], lw=2.5, ls=_ls_cycle[i % len(_ls_cycle)],
                label=f"{s['label']} (n={len(_sdf)})")
if len(df_not_sel) > 0:
    c_not = build_portfolio_curve(df_not_sel)
    xs_n = sorted(c_not.keys())
    ax.plot(xs_n, [c_not[x] for x in xs_n], '#e74c3c', lw=1, alpha=0.4, label=f'Not-sel (n={len(df_not_sel)})')
ax.axhline(y=0, color='black', lw=0.8)
ax.axvline(x=30, color='orange', lw=0.8, ls=':', alpha=0.6, label='30m mark')
ax.axvline(x=60, color='purple', lw=0.8, ls=':', alpha=0.6, label='60m mark')
ax.set_xlabel('Minutes from entry')
ax.set_ylabel('Avg PnL %')
ax.set_title('3. Intraday Curves')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 1d. PnL distribution overlay — all strategies
ax = axes1[1, 1]
_bins = np.linspace(-3, 10, 30)
for s in STRATEGIES:
    v60 = strat_dfs[s['key']]['sl_pnl_60m'].dropna()
    if len(v60) > 0:
        ax.hist(v60, bins=_bins, alpha=0.4, color=s['color'],
                label=f"{s['label']} (avg={v60.mean():+.2f}%)")
ax.axvline(x=0, color='black', lw=1, ls='--')
ax.axvline(x=-STOP_LOSS_PCT, color='red', lw=1, ls=':', alpha=0.6, label=f'SL={STOP_LOSS_PCT}%')
ax.set_xlabel('PnL % @60m')
ax.set_ylabel('Count')
ax.set_title('4. PnL Distribution')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
display(fig1)
plt.close(fig1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 2: Confusion Matrix Heatmaps — per strategy @60m
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_cm_strat = {}
for s in STRATEGIES:
    _cm_strat[s['key']] = {}
    for h in HORIZONS_MIN:
        col = f'sl_pnl_{h}m'
        _vh_s = df_pnl.dropna(subset=[col])
        tp_s = _vh_s[(_vh_s[s['sel_col']] == True) & (_vh_s[col] > 0)]
        fp_s = _vh_s[(_vh_s[s['sel_col']] == True) & (_vh_s[col] <= 0)]
        tn_s = _vh_s[(_vh_s[s['sel_col']] == False) & (_vh_s[col] <= 0)]
        fn_s = _vh_s[(_vh_s[s['sel_col']] == False) & (_vh_s[col] > 0)]
        prec_s = len(tp_s) / max(len(tp_s) + len(fp_s), 1) * 100
        rec_s = len(tp_s) / max(len(tp_s) + len(fn_s), 1) * 100
        _cm_strat[s['key']][h] = {'tp': tp_s, 'fp': fp_s, 'tn': tn_s, 'fn': fn_s,
                                   'precision': prec_s, 'recall': rec_s}

fig2, axes2 = plt.subplots(1, len(STRATEGIES), figsize=(7 * len(STRATEGIES), 6))
fig2.suptitle('Confusion Matrix @60m — Strategy Comparison', fontsize=13, fontweight='bold')
if len(STRATEGIES) == 1:
    axes2 = [axes2]

for idx, s in enumerate(STRATEGIES):
    ax = axes2[idx]
    cm_src = _cm_strat[s['key']][60]
    tp_h, fp_h, tn_h, fn_h = cm_src['tp'], cm_src['fp'], cm_src['tn'], cm_src['fn']
    col = 'sl_pnl_60m'
    cm_arr = np.array([[len(tp_h), len(fp_h)], [len(fn_h), len(tn_h)]])
    ax.imshow(cm_arr, cmap='RdYlGn', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Profitable', 'Unprofitable'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Selected', 'Not-selected'])
    labels_cm = [['TP', 'FP'], ['FN', 'TN']]
    dfs_cm = [[tp_h, fp_h], [fn_h, tn_h]]
    for i in range(2):
        for j in range(2):
            avg_v = dfs_cm[i][j][col].mean() if len(dfs_cm[i][j]) > 0 else 0
            ax.text(j, i, f'{labels_cm[i][j]}\nn={cm_arr[i, j]}\navg={avg_v:+.1f}%',
                    ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_title(f'{s["label"]} @60m  |  P={cm_src["precision"]:.0f}%  R={cm_src["recall"]:.0f}%')

plt.tight_layout(rect=[0, 0, 1, 0.92])
display(fig2)
plt.close(fig2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 3: Direction — all strategies (2x2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
fig3.suptitle('By Direction: Strategy Comparison', fontsize=13, fontweight='bold')

# 3a. Direction x Selection @60m — all strategies
ax = axes3[0, 0]
dir_data = []
for d in ['LONG', 'SHORT']:
    for s in STRATEGIES:
        v = df_pnl[(df_pnl['direction'] == d) & (df_pnl[s['sel_col']] == True)]['sl_pnl_60m'].dropna()
        if len(v) > 0:
            dir_data.append((f'{d}\n{s["short"]}', v.mean(), s['color'], len(v)))
if dir_data:
    x_d = range(len(dir_data))
    ax.bar(x_d, [d[1] for d in dir_data], color=[d[2] for d in dir_data], alpha=0.8)
    ax.set_xticks(x_d)
    ax.set_xticklabels([f"{d[0]}\nn={d[3]}" for d in dir_data], fontsize=8)
    for i, d in enumerate(dir_data):
        ax.annotate(f'{d[1]:+.2f}%', xy=(i, d[1]), ha='center',
                    va='bottom' if d[1] >= 0 else 'top', fontsize=9, fontweight='bold')
ax.set_ylabel('Avg PnL @60m %')
ax.set_title('5. Direction @60m')
ax.axhline(y=0, color='black', lw=0.8)
ax.grid(True, alpha=0.3, axis='y')

# 3b. Direction x Selection @30m — all strategies
ax = axes3[0, 1]
dir_data_30 = []
for d in ['LONG', 'SHORT']:
    for s in STRATEGIES:
        v = df_pnl[(df_pnl['direction'] == d) & (df_pnl[s['sel_col']] == True)]['sl_pnl_30m'].dropna()
        if len(v) > 0:
            dir_data_30.append((f'{d}\n{s["short"]}', v.mean(), s['color'], len(v)))
if dir_data_30:
    x_d = range(len(dir_data_30))
    ax.bar(x_d, [d[1] for d in dir_data_30], color=[d[2] for d in dir_data_30], alpha=0.8)
    ax.set_xticks(x_d)
    ax.set_xticklabels([f"{d[0]}\nn={d[3]}" for d in dir_data_30], fontsize=8)
    for i, d in enumerate(dir_data_30):
        ax.annotate(f'{d[1]:+.2f}%', xy=(i, d[1]), ha='center',
                    va='bottom' if d[1] >= 0 else 'top', fontsize=9, fontweight='bold')
ax.set_ylabel('Avg PnL @30m %')
ax.set_title('5b. Direction @30m')
ax.axhline(y=0, color='black', lw=0.8)
ax.grid(True, alpha=0.3, axis='y')

# 3c. Scanner performance @60m — all strategies
ax = axes3[1, 0]
sc_data = []
for sc in sorted(df_pnl['scanner'].unique()):
    for s in STRATEGIES:
        v = df_pnl[(df_pnl['scanner'] == sc) & (df_pnl[s['sel_col']] == True)]['sl_pnl_60m'].dropna()
        if len(v) > 0:
            sc_data.append((f'{sc}\n{s["short"]}', v.mean(), s['color'], len(v)))
if sc_data:
    x_s = range(len(sc_data))
    ax.bar(x_s, [d[1] for d in sc_data], color=[d[2] for d in sc_data], alpha=0.8)
    ax.set_xticks(x_s)
    ax.set_xticklabels([f"{d[0]}\nn={d[3]}" for d in sc_data], fontsize=7)
    for i, d in enumerate(sc_data):
        ax.annotate(f'{d[1]:+.2f}%', xy=(i, d[1]), ha='center',
                    va='bottom' if d[1] >= 0 else 'top', fontsize=8, fontweight='bold')
ax.set_ylabel('Avg PnL @60m %')
ax.set_title('6. Scanner @60m')
ax.axhline(y=0, color='black', lw=0.8)
ax.grid(True, alpha=0.3, axis='y')

# 3d. Intraday by direction — one line per strategy per direction
ax = axes3[1, 1]
for d, c_dir in [('LONG', '#27ae60'), ('SHORT', '#c0392b')]:
    for i, s in enumerate(STRATEGIES):
        df_ds = df_pnl[(df_pnl['direction'] == d) & (df_pnl[s['sel_col']] == True)]
        if len(df_ds) > 0:
            curve = build_portfolio_curve(df_ds)
            xs = sorted(curve.keys())
            ax.plot(xs, [curve[x] for x in xs], color=c_dir, lw=2, ls=_ls_cycle[i % len(_ls_cycle)],
                    label=f'{d} {s["short"]} (n={len(df_ds)})')
ax.axhline(y=0, color='black', lw=0.8)
ax.axvline(x=30, color='orange', lw=0.8, ls=':', alpha=0.6)
ax.axvline(x=60, color='purple', lw=0.8, ls=':', alpha=0.6)
ax.set_xlabel('Minutes from entry')
ax.set_ylabel('Avg PnL %')
ax.set_title('Intraday by Direction')
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
display(fig3)
plt.close(fig3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 4: Catalyst Type — Baseline vs #3 Fix (2x2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12))
fig4.suptitle('Catalyst Type & Confidence: Strategy Comparison', fontsize=13, fontweight='bold')

# 4a. Catalyst type — per-strategy avg @60m
ax = axes4[0, 0]
_cat_dfs = {s['key']: strat_dfs[s['key']].dropna(subset=['sl_pnl_60m']) for s in STRATEGIES}
all_cats_set = set()
for _cdf in _cat_dfs.values():
    all_cats_set.update(_cdf['catalyst_type'].unique())
all_cats = sorted(c for c in all_cats_set if c and c != 'nan')
if all_cats:
    cat_labels = []
    strat_avgs = {s['key']: [] for s in STRATEGIES}
    cat_ns = []
    for cat in all_cats:
        total_n = sum(len(_cat_dfs[s['key']][_cat_dfs[s['key']]['catalyst_type'] == cat]) for s in STRATEGIES)
        if total_n < 2:
            continue
        cat_labels.append(cat)
        ns_parts = []
        for s in STRATEGIES:
            cv = _cat_dfs[s['key']][_cat_dfs[s['key']]['catalyst_type'] == cat]['sl_pnl_60m']
            strat_avgs[s['key']].append(cv.mean() if len(cv) > 0 else 0)
            ns_parts.append(f"{s['short'][0]}{len(cv)}")
        cat_ns.append('/'.join(ns_parts))
    if cat_labels:
        y_p = np.arange(len(cat_labels))
        h_bar = 0.8 / len(STRATEGIES)
        for i, s in enumerate(STRATEGIES):
            offset = (i - len(STRATEGIES)/2 + 0.5) * h_bar
            ax.barh(y_p + offset, strat_avgs[s['key']], h_bar, color=s['color'],
                    alpha=0.8, label=s['label'])
        ax.set_yticks(y_p)
        ax.set_yticklabels([f"{l} ({n})" for l, n in zip(cat_labels, cat_ns)], fontsize=7)
        ax.axvline(x=0, color='black', lw=0.8)
        ax.set_xlabel('Avg PnL @60m %')
        ax.set_title('7. Catalyst Type')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='x')

# 4b. Confidence vs PnL — per strategy marker
ax = axes4[0, 1]
_markers = ['o', 'D', 's', '^']
for si, s in enumerate(STRATEGIES):
    for d, c in [('LONG', '#27ae60'), ('SHORT', '#c0392b')]:
        mask_d = (df_pnl['direction'] == d) & (df_pnl[s['sel_col']] == True)
        v = df_pnl[mask_d].dropna(subset=['sl_pnl_60m'])
        if len(v) > 0:
            ax.scatter(v['confidence'], v['sl_pnl_60m'], c=c,
                       marker=_markers[si % len(_markers)],
                       s=50, alpha=0.6, label=f'{d} {s["short"]}',
                       edgecolors='black', linewidths=0.5)
ax.axhline(y=0, color='black', lw=0.8)
ax.set_xlabel('Confidence')
ax.set_ylabel('PnL @60m %')
_mkr_legend = ' '.join(f'{_markers[i]}={STRATEGIES[i]["short"]}' for i in range(len(STRATEGIES)))
ax.set_title(f'Confidence vs PnL ({_mkr_legend})')
ax.legend(fontsize=6, loc='upper right')
ax.grid(True, alpha=0.3)

# 4c. Gate effectiveness — gated vs ungated
ax = axes4[1, 0]
if len(df_gated) > 0:
    gate_cats = df_gated['gate_reason'].value_counts().head(8)
    gate_labels = []
    gate_avgs = []
    gate_counts = []
    for gr, cnt in gate_cats.items():
        gr_v = df_gated[df_gated['gate_reason'] == gr]['sl_pnl_60m'].dropna()
        if len(gr_v) > 0:
            gate_labels.append(gr[:30])
            gate_avgs.append(gr_v.mean())
            gate_counts.append(cnt)
    if gate_labels:
        y_g = range(len(gate_labels))
        colors_g = ['#27ae60' if x > 0 else '#c0392b' for x in gate_avgs]
        ax.barh(y_g, gate_avgs, color=colors_g, height=0.6)
        ax.set_yticks(y_g)
        ax.set_yticklabels([f"{l} (n={c})" for l, c in zip(gate_labels, gate_counts)], fontsize=7)
        for i, v in enumerate(gate_avgs):
            ax.text(v + 0.05 * (1 if v >= 0 else -1), i, f"{v:+.1f}%", va='center', fontsize=8)
        ax.axvline(x=0, color='black', lw=0.8)
        ax.set_xlabel('Avg PnL @60m %')
        ax.set_title('10. Gate Reason Breakdown')
        ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No gated signals', ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('10. Gate Reason Breakdown')

# 4d. Gated vs Ungated summary
ax = axes4[1, 1]
if len(df_gated) > 0:
    s_gated = _h_stats(df_gated, 60)
    s_ungated = _h_stats(df_pnl[df_pnl['gate_reason'] == ''], 60)
    s_gated_30 = _h_stats(df_gated, 30)
    s_ungated_30 = _h_stats(df_pnl[df_pnl['gate_reason'] == ''], 30)
    cats = ['Gated\n@30m', 'Ungated\n@30m', 'Gated\n@60m', 'Ungated\n@60m']
    vals = [s_gated_30['avg'], s_ungated_30['avg'], s_gated['avg'], s_ungated['avg']]
    ns = [s_gated_30['n'], s_ungated_30['n'], s_gated['n'], s_ungated['n']]
    clrs = ['#c0392b', '#27ae60', '#c0392b', '#27ae60']
    ax.bar(range(4), vals, color=clrs, alpha=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{c}\nn={n}" for c, n in zip(cats, ns)])
    for i, v in enumerate(vals):
        ax.annotate(f'{v:+.2f}%', xy=(i, v), ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='black', lw=0.8)
    ax.set_ylabel('Avg PnL %')
    ax.set_title('Gated vs Ungated Performance')
    ax.grid(True, alpha=0.3, axis='y')
else:
    ax.text(0.5, 0.5, 'No gated signals', ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('Gated vs Ungated Performance')

plt.tight_layout(rect=[0, 0, 1, 0.94])
display(fig4)
plt.close(fig4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 5: Strategy PnL by Entry Hour (2x2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_hour_bins = _hour_bins_tbl

fig5, axes5 = plt.subplots(2, 2, figsize=(16, 12))
fig5.suptitle('Entry Hour: Strategy Comparison (ET)', fontsize=13, fontweight='bold')

# 5a. Avg PnL @60m by hour — all strategies
ax = axes5[0, 0]
if _hour_bins:
    hour_labels = []
    strat_avgs_h = {s['key']: [] for s in STRATEGIES}
    strat_ns_h = {s['key']: [] for s in STRATEGIES}
    for hb in _hour_bins:
        total_n = sum(_h_stats(df_pnl[(df_pnl['entry_hour'] == hb) & (df_pnl[s['sel_col']] == True)], 60)['n']
                      for s in STRATEGIES)
        if total_n < 2:
            continue
        hour_labels.append(hb)
        for s in STRATEGIES:
            hs = _h_stats(df_pnl[(df_pnl['entry_hour'] == hb) & (df_pnl[s['sel_col']] == True)], 60)
            strat_avgs_h[s['key']].append(hs['avg'] if hs['n'] > 0 else 0)
            strat_ns_h[s['key']].append(hs['n'])
    if hour_labels:
        x_h = np.arange(len(hour_labels))
        w_h = 0.8 / len(STRATEGIES)
        for i, s in enumerate(STRATEGIES):
            offset = (i - len(STRATEGIES)/2 + 0.5) * w_h
            ax.bar(x_h + offset, strat_avgs_h[s['key']], w_h, label=s['label'],
                   color=s['color'], alpha=0.85)
        ax.set_xticks(x_h)
        ax.set_xticklabels(hour_labels, rotation=45, fontsize=8)
        ax.axhline(y=0, color='black', lw=0.8)
        ax.set_ylabel('Avg PnL @60m %')
        ax.set_title('Strategies by Entry Hour')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

# 5b. Avg PnL @60m by hour — LONG vs SHORT (unchanged)
ax = axes5[0, 1]
if _hour_bins:
    long_avgs = []
    short_avgs = []
    hour_labels2 = []
    long_ns = []
    short_ns = []
    w_h2 = 0.35
    for hb in _hour_bins:
        df_hb_l = df_pnl[(df_pnl['entry_hour'] == hb) & (df_pnl['direction'] == 'LONG')]
        df_hb_s = df_pnl[(df_pnl['entry_hour'] == hb) & (df_pnl['direction'] == 'SHORT')]
        s_l = _h_stats(df_hb_l, 60)
        s_s = _h_stats(df_hb_s, 60)
        if s_l['n'] + s_s['n'] < 2:
            continue
        hour_labels2.append(hb)
        long_avgs.append(s_l['avg'] if s_l['n'] > 0 else 0)
        short_avgs.append(s_s['avg'] if s_s['n'] > 0 else 0)
        long_ns.append(s_l['n'])
        short_ns.append(s_s['n'])
    if hour_labels2:
        x_h2 = np.arange(len(hour_labels2))
        ax.bar(x_h2 - w_h2/2, long_avgs, w_h2, label='LONG', color='#27ae60', alpha=0.8)
        ax.bar(x_h2 + w_h2/2, short_avgs, w_h2, label='SHORT', color='#c0392b', alpha=0.7)
        ax.set_xticks(x_h2)
        ax.set_xticklabels([f"{l}\n(L{ln}/S{sn})" for l, ln, sn in zip(hour_labels2, long_ns, short_ns)],
                           rotation=45, fontsize=8)
        ax.axhline(y=0, color='black', lw=0.8)
        ax.set_ylabel('Avg PnL @60m %')
        ax.set_title('LONG vs SHORT by Entry Hour')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

# 5c. Win rate by entry hour — all strategies
ax = axes5[1, 0]
if _hour_bins:
    wr_labels_h = []
    strat_wr_h = {s['key']: [] for s in STRATEGIES}
    for hb in _hour_bins:
        total_n = sum(_h_stats(df_pnl[(df_pnl['entry_hour'] == hb) & (df_pnl[s['sel_col']] == True)], 60)['n']
                      for s in STRATEGIES)
        if total_n < 2:
            continue
        wr_labels_h.append(hb)
        for s in STRATEGIES:
            hs = _h_stats(df_pnl[(df_pnl['entry_hour'] == hb) & (df_pnl[s['sel_col']] == True)], 60)
            strat_wr_h[s['key']].append(hs['wr'] if hs['n'] > 0 else 0)
    if wr_labels_h:
        x_wr = np.arange(len(wr_labels_h))
        w_wr = 0.8 / len(STRATEGIES)
        for i, s in enumerate(STRATEGIES):
            offset = (i - len(STRATEGIES)/2 + 0.5) * w_wr
            ax.bar(x_wr + offset, strat_wr_h[s['key']], w_wr, label=s['label'],
                   color=s['color'], alpha=0.8)
        ax.set_xticks(x_wr)
        ax.set_xticklabels(wr_labels_h, rotation=45, fontsize=8)
        ax.axhline(y=50, color='black', lw=0.8, ls='--')
        ax.set_ylabel('Win Rate @60m %')
        ax.set_title('Win Rate by Entry Hour')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

# 5d. Entry time scatter — per strategy marker
ax = axes5[1, 1]
for si, s in enumerate(STRATEGIES):
    for d, c in [('LONG', '#27ae60'), ('SHORT', '#c0392b')]:
        mask_d = (df_pnl['direction'] == d) & (df_pnl[s['sel_col']] == True)
        v = df_pnl[mask_d].dropna(subset=['sl_pnl_60m'])
        if len(v) > 0:
            entry_mins = v['entry_time_et'].apply(
                lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) - 570 if pd.notna(x) else np.nan)
            ax.scatter(entry_mins, v['sl_pnl_60m'], c=c,
                       marker=_markers[si % len(_markers)],
                       s=50, alpha=0.6, label=f'{d} {s["short"]}',
                       edgecolors='black', linewidths=0.5)
ax.axhline(y=0, color='black', lw=0.8)
ax.set_xlabel('Minutes after market open (9:30 ET)')
ax.set_ylabel('PnL @60m %')
ax.set_title(f'Entry Time vs PnL ({_mkr_legend})')
ax.legend(fontsize=6, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
display(fig5)
plt.close(fig5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART 6: Per-trade waterfall — all strategies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_n_strat6 = len(STRATEGIES)
fig6, axes6 = plt.subplots(1, _n_strat6, figsize=(9 * _n_strat6, 10))
if _n_strat6 == 1:
    axes6 = [axes6]
_strat_title6 = ' vs '.join(s['short'] for s in STRATEGIES)
fig6.suptitle(f'Per-Trade PnL @60m — {_strat_title6}', fontsize=14, fontweight='bold')

_n_show = 30
for idx, s in enumerate(STRATEGIES):
    ax = axes6[idx]
    _df = strat_dfs[s['key']]
    df_show = _df.dropna(subset=['sl_pnl_60m']).sort_values('sl_pnl_60m', ascending=True)
    if len(df_show) > _n_show:
        df_top = pd.concat([df_show.head(_n_show // 2), df_show.tail(_n_show // 2)])
    else:
        df_top = df_show
    if len(df_top) == 0:
        continue
    colors = ['#27ae60' if v > 0 else '#c0392b' for v in df_top['sl_pnl_60m']]
    y_pos = range(len(df_top))
    ax.barh(y_pos, df_top['sl_pnl_60m'], color=colors, height=0.7)
    ax.set_yticks(y_pos)
    tick_fs = max(5, min(8, 300 // max(len(df_top), 1)))
    ax.set_yticklabels(
        [f"{r['ticker']} {r['direction'][0]} {r['eval_date'][-5:]}" for _, r in df_top.iterrows()],
        fontsize=tick_fs)
    ax.set_xlabel('PnL % @60m')
    avg_v = _df['sl_pnl_60m'].dropna()
    ax.set_title(f'{s["label"]} (n={len(avg_v)}, avg={avg_v.mean():+.2f}%)')
    ax.axvline(x=0, color='black', lw=0.8)
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
display(fig6)
plt.close(fig6)


# ==============================================================================
# SAVE BENCHMARK JSON
# ==============================================================================
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

benchmark = {
    'name': BENCHMARK_NAME,
    'created': datetime.utcnow().isoformat(),
    'config': {
        'dates': BENCHMARK_DATES, 'stop_loss_pct': STOP_LOSS_PCT,
        'horizons_min': HORIZONS_MIN, 'selected_only': SELECTED_ONLY,
        'latest_version_only': USE_LATEST_VERSION_ONLY,
        'preferred_versions': PREFERRED_VERSIONS,
    },
    'summary': {
        'n_days': n_days, 'n_total': n_total,
        'n_selected': n_sel_total, 'n_not_selected': n_not_total,
        'n_stopped': n_stopped,
        'trades_per_day_total': round(n_total / n_days, 1),
        'trades_per_day_selected': round(n_sel_total / n_days, 1),
        'pct_stopped': round(n_stopped / n_total * 100, 1),
    },
    'aggregate': agg,
    'daily': daily_results,
    'equity': {
        'selected_return_pct': round(float(sel_total), 4),
        'all_return_pct': round(float(all_total), 4),
        'not_selected_return_pct': round(float(not_total), 4),
        'spy_return_pct': round(float(spy_total), 4),
        'selected_alpha_pct': round(float(sel_total - spy_total), 4),
        'selection_edge_pct': round(float(sel_total - all_total), 4),
    },
    'confusion_matrix': {
        h: {'tp': len(_cm[h]['tp']), 'fp': len(_cm[h]['fp']),
            'tn': len(_cm[h]['tn']), 'fn': len(_cm[h]['fn']),
            'precision': round(_cm[h]['precision'], 1),
            'recall': round(_cm[h]['recall'], 1)}
        for h in HORIZONS_MIN
    },
    'per_trade': [
        {
            'ticker': r['ticker'], 'date': r['eval_date'], 'direction': r['direction'],
            'selected': bool(r['selected']), 'gate_reason': r.get('gate_reason', ''),
            'scanner': r['scanner'], 'version': r['scanner_version'],
            'entry_time': r['entry_time_et'], 'entry_price': round(float(r['entry_price']), 2),
            'gap_pct': round(float(r['gap_pct']), 2), 'catalyst': r['catalyst_type'],
            'sl_pnl_30m': round(float(r['sl_pnl_30m']), 4) if pd.notna(r.get('sl_pnl_30m')) else None,
            'sl_pnl_60m': round(float(r['sl_pnl_60m']), 4) if pd.notna(r.get('sl_pnl_60m')) else None,
            'alpha_60m': round(float(r['alpha_60m']), 4) if pd.notna(r.get('alpha_60m')) else None,
            'sl_pnl_eod': round(float(r['sl_pnl_eod']), 4) if pd.notna(r.get('sl_pnl_eod')) else None,
            'stopped_out': bool(r['stopped_out']),
            'stop_minute': int(r['stop_minute']) if r['stopped_out'] else None,
        }
        for _, r in df_pnl.iterrows()
    ],
}

json_path = f"{OUTPUT_DIR}/scanner_benchmark_{BENCHMARK_NAME}.json"
with open(json_path, 'w') as f:
    json.dump(benchmark, f, indent=2, cls=_NumpyEncoder)
print(f"  Benchmark JSON saved: {json_path}")


# ==============================================================================
# FABLE #3 DIAGNOSTIC: informed_score distribution + trade swaps
# (computation already done above; this section prints details + diagnostic chart)
# ==============================================================================

if len(df_ab) > 0:
    print(f"\n\n{'='*_W}")
    print(f"  FABLE #3 DIAGNOSTIC: informed_score OLD vs NEW")
    print(f"{'='*_W}")

    print(f"\n  Distribution comparison (n={len(df_ab)} trades):")
    print(f"  {'':─<{_W-4}}")
    print(f"  {'Metric':<25} {'OLD':>10} {'NEW':>10} {'Change':>10}")
    print(f"  {'':─<{_W-4}}")
    for label, col_old, col_new in [
        ('informed_score mean', 'old_informed', 'new_informed'),
        ('informed_score std', 'old_informed', 'new_informed'),
        ('informed_score min', 'old_informed', 'new_informed'),
        ('informed_score max', 'old_informed', 'new_informed'),
    ]:
        if 'mean' in label:
            ov, nv = df_ab[col_old].mean(), df_ab[col_new].mean()
        elif 'std' in label:
            ov, nv = df_ab[col_old].std(), df_ab[col_new].std()
        elif 'min' in label:
            ov, nv = df_ab[col_old].min(), df_ab[col_new].min()
        elif 'max' in label:
            ov, nv = df_ab[col_old].max(), df_ab[col_new].max()
        print(f"  {label:<25} {ov:>9.3f} {nv:>9.3f} {nv-ov:>+9.3f}")

    old_fires = (df_ab['old_tape'] < 1.0).sum()
    new_fires = (df_ab['new_tape'] < 1.0).sum()
    print(f"  {'tape_discount fires':<25} {old_fires:>9} {new_fires:>9} {new_fires-old_fires:>+9}")

    _v60_ab = df_ab.dropna(subset=['sl_pnl_60m'])
    if len(_v60_ab) > 5:
        old_corr = _v60_ab['old_informed'].corr(_v60_ab['sl_pnl_60m'])
        new_corr = _v60_ab['new_informed'].corr(_v60_ab['sl_pnl_60m'])
        print(f"\n  Correlation with PnL @60m:")
        print(f"  {'OLD informed vs PnL':<25} r={old_corr:+.3f}")
        print(f"  {'NEW informed vs PnL':<25} r={new_corr:+.3f}")

    _old_sel_ab = df_ab[df_ab['selected'] == True]
    _new_sel_ab = df_ab[df_ab['new_selected'] == True]
    _old_only = set(zip(_old_sel_ab['ticker'], _old_sel_ab['eval_date'])) - set(zip(_new_sel_ab['ticker'], _new_sel_ab['eval_date']))
    _new_only = set(zip(_new_sel_ab['ticker'], _new_sel_ab['eval_date'])) - set(zip(_old_sel_ab['ticker'], _old_sel_ab['eval_date']))
    _both = set(zip(_old_sel_ab['ticker'], _old_sel_ab['eval_date'])) & set(zip(_new_sel_ab['ticker'], _new_sel_ab['eval_date']))

    print(f"\n\n  RE-SELECTION SIMULATION:")
    print(f"  {'':─<{_W-4}}")
    print(f"  Trades in both:     {len(_both)}")
    print(f"  Dropped by #3 fix: {len(_old_only)}")
    print(f"  Added by #3 fix:   {len(_new_only)}")

    if _old_only:
        print(f"\n  DROPPED (selected by baseline, not by #3 fix):")
        print(f"  {'Ticker':<8} {'Date':<12} {'Dir':<6} {'OldEV':>7} {'NewEV':>7} {'TD':>5} {'PnL60':>7}")
        for tk, dt in sorted(_old_only):
            r = df_ab[(df_ab['ticker']==tk) & (df_ab['eval_date']==dt)].iloc[0]
            pnl = f"{r['sl_pnl_60m']:+6.2f}%" if pd.notna(r['sl_pnl_60m']) else '    n/a'
            print(f"  {tk:<8} {dt:<12} {r['direction']:<6} {r['old_ev']:>6.2f} {r['new_ev']:>6.2f} "
                  f"{r['new_tape']:>4.2f} {pnl}")

    if _new_only:
        print(f"\n  ADDED (not in baseline, selected by #3 fix):")
        print(f"  {'Ticker':<8} {'Date':<12} {'Dir':<6} {'OldEV':>7} {'NewEV':>7} {'TD':>5} {'PnL60':>7}")
        for tk, dt in sorted(_new_only):
            r = df_ab[(df_ab['ticker']==tk) & (df_ab['eval_date']==dt)].iloc[0]
            pnl = f"{r['sl_pnl_60m']:+6.2f}%" if pd.notna(r['sl_pnl_60m']) else '    n/a'
            print(f"  {tk:<8} {dt:<12} {r['direction']:<6} {r['old_ev']:>6.2f} {r['new_ev']:>6.2f} "
                  f"{r['new_tape']:>4.2f} {pnl}")

    print(f"\n\n  PORTFOLIO COMPARISON: All Strategies")
    print(f"  {'':─<{_W-4}}")
    _hdr_strats = ''.join(f'{s["short"]:>12}' for s in STRATEGIES)
    print(f"  {'Metric':<25}{_hdr_strats}")
    print(f"  {'':─<{_W-4}}")
    for h in HORIZONS_MIN:
        col = f'sl_pnl_{h}m'
        _vals = []
        for s in STRATEGIES:
            sv = strat_dfs[s['key']][col].dropna()
            _vals.append((float(sv.mean()) if len(sv) > 0 else 0,
                          float((sv > 0).sum() / len(sv) * 100) if len(sv) > 0 else 0))
        print(f"  {'Sel avg @'+str(h)+'m':<25}" + ''.join(f'{v[0]:>+11.3f}%' for v in _vals))
        print(f"  {'Sel WR @'+str(h)+'m':<25}" + ''.join(f'{v[1]:>10.0f}% ' for v in _vals))
    _eod_vals = []
    for s in STRATEGIES:
        sv_eod = strat_dfs[s['key']]['sl_pnl_eod'].dropna()
        _eod_vals.append(float(sv_eod.mean()) if len(sv_eod) > 0 else 0)
    print(f"  {'Sel avg EOD':<25}" + ''.join(f'{v:>+11.3f}%' for v in _eod_vals))
    print(f"  {'n selected':<25}" + ''.join(f'{len(strat_dfs[s["key"]]):>12}' for s in STRATEGIES))

    print(f"\n  Daily breakdown:")
    _day_hdr = '  '.join(f'{s["short"]+"@60":>8}' for s in STRATEGIES)
    _day_eod_hdr = '  '.join(f'{s["short"]+"EOD":>8}' for s in STRATEGIES)
    print(f"  {'Date':<12} {_day_hdr}  {_day_eod_hdr}")
    for dt in sorted(df_ab['eval_date'].unique()):
        parts_60 = []
        parts_eod = []
        for s in STRATEGIES:
            _sdf_day = strat_dfs[s['key']][strat_dfs[s['key']]['eval_date'] == dt]
            sday60 = _sdf_day['sl_pnl_60m'].dropna()
            sday_eod = _sdf_day['sl_pnl_eod'].dropna()
            parts_60.append(f"{sday60.mean():+7.2f}%" if len(sday60) > 0 else '     n/a')
            parts_eod.append(f"{sday_eod.mean():+7.2f}%" if len(sday_eod) > 0 else '     n/a')
        print(f"  {dt:<12} {'  '.join(parts_60)}  {'  '.join(parts_eod)}")

    _best_s = max(STRATEGIES, key=lambda s: _eod_vals[STRATEGIES.index(s)])
    _best_eod = _eod_vals[STRATEGIES.index(_best_s)]
    _base_eod = _eod_vals[0]
    _verdict_strat = "BETTER" if _best_eod > _base_eod else ("SAME" if abs(_best_eod - _base_eod) < 0.01 else "WORSE")
    print(f"\n  Best strategy: {_best_s['label']} ({_verdict_strat} than baseline, "
          f"EOD: {_best_eod-_base_eod:+.3f}%, n={len(strat_dfs[_best_s['key']])})")

    # ── Diagnostic chart: informed_score distribution + EV scatter ──
    fig_ab, axes_ab = plt.subplots(2, 2, figsize=(16, 12))
    fig_ab.suptitle('Fable #3 Diagnostic: informed_score OLD vs NEW', fontsize=13, fontweight='bold')

    ax = axes_ab[0, 0]
    _bins_is = np.linspace(0, 1, 25)
    ax.hist(df_ab['old_informed'], bins=_bins_is, alpha=0.6, color='#e74c3c',
            label=f'OLD (mean={df_ab["old_informed"].mean():.3f})')
    ax.hist(df_ab['new_informed'], bins=_bins_is, alpha=0.6, color='#2ecc71',
            label=f'NEW (mean={df_ab["new_informed"].mean():.3f})')
    ax.set_xlabel('informed_score')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution: OLD vs NEW')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes_ab[0, 1]
    ax.scatter(df_ab['old_informed'], df_ab['new_informed'], c='#3498db', alpha=0.6, s=40)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('OLD informed_score')
    ax.set_ylabel('NEW informed_score')
    ax.set_title('OLD vs NEW (above diagonal = higher)')
    ax.grid(True, alpha=0.3)

    ax = axes_ab[1, 0]
    _v_ev = df_ab[(df_ab['old_ev'] > 0) | (df_ab['new_ev'] > 0)]
    if len(_v_ev) > 0:
        colors_ev = ['#2ecc71' if r['selected'] else '#e74c3c' for _, r in _v_ev.iterrows()]
        ax.scatter(_v_ev['old_ev'], _v_ev['new_ev'], c=colors_ev, alpha=0.6, s=40)
        _max_ev = max(_v_ev['old_ev'].max(), _v_ev['new_ev'].max()) * 1.1
        ax.plot([0, _max_ev], [0, _max_ev], 'k--', lw=0.8)
        ax.set_xlabel('OLD EV')
        ax.set_ylabel('NEW EV (with tape discount)')
        ax.set_title('EV: OLD vs NEW (green=selected)')
        ax.grid(True, alpha=0.3)

    ax = axes_ab[1, 1]
    _horizons_plus_eod = list(HORIZONS_MIN) + ['eod']
    _horizon_labels = [f'@{h}m' for h in HORIZONS_MIN] + ['EOD']
    _n_s_ab = len(STRATEGIES)
    w_m = 0.8 / _n_s_ab
    x_m = np.arange(len(_horizons_plus_eod))
    for si, s in enumerate(STRATEGIES):
        _sdf_ab = strat_dfs[s['key']]
        vals = []
        for h in HORIZONS_MIN:
            col = f'sl_pnl_{h}m'
            sv = _sdf_ab[col].dropna()
            vals.append(float(sv.mean()) if len(sv) > 0 else 0)
        sv_eod = _sdf_ab['sl_pnl_eod'].dropna()
        vals.append(float(sv_eod.mean()) if len(sv_eod) > 0 else 0)
        offset = (si - (_n_s_ab - 1) / 2) * w_m
        ax.bar(x_m + offset, vals, w_m, label=s['label'], color=s['color'], alpha=0.8)
        for i, v in enumerate(vals):
            ax.annotate(f'{v:+.2f}%', xy=(i + offset, v), ha='center',
                        va='bottom' if v >= 0 else 'top', fontsize=7)
    ax.set_xticks(x_m)
    ax.set_xticklabels(_horizon_labels)
    ax.axhline(y=0, color='black', lw=0.8)
    ax.set_ylabel('Avg PnL %')
    _n_lbl = '/'.join(str(len(strat_dfs[s['key']])) for s in STRATEGIES)
    ax.set_title(f'Selected Portfolio: All Strategies (n={_n_lbl})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    display(fig_ab)
    plt.close(fig_ab)


# ── Final verdict ──
print(f"\n\n{_sep}")
print(f"  BENCHMARK: {BENCHMARK_NAME}")
_strat_line = '  |  '.join(f"{s['label']}: {strat_total[s['key']]:+.2f}%" for s in STRATEGIES)
print(f"  {_strat_line}  |  SPY: {spy_total:+.2f}%")
for h in HORIZONS_MIN:
    _edge_h = agg.get(f'selection_edge_{h}m', 0)
    _verdict_h = "POSITIVE" if _edge_h > 0 else "NEGATIVE"
    parts = [f"@{h}m — base edge: {_edge_h:+.3f}% ({_verdict_h})"]
    for s in STRATEGIES:
        cm_s = _cm_strat[s['key']][h]
        parts.append(f"{s['short']} P:{cm_s['precision']:.0f}% R:{cm_s['recall']:.0f}%")
    print(f"  {'  |  '.join(parts)}")
print(f"{_sep}\n")
