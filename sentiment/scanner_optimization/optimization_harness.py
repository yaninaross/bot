# %% ── Scanner Optimization Harness ────────────────────────────────────────────
#
# Systematic strategy search over Sep 8-11 2026 scanner data.
# Pulls ALL signals from BQ, computes PnL with SL simulation,
# then tests 50+ filter/selection strategies and ranks them.
#
# Cross-validates: rules derived on Sep 8-9 tested on Sep 10-11 and vice versa.
#
# Output:
#   - Per-strategy stats (avg PnL, WR, n, SE, Sharpe proxy)
#   - Top strategies that survive cross-validation
#   - Full results JSON for downstream analysis
#
# Colab-ready, config vars at top.

import os, sys, json, warnings, itertools
from datetime import datetime, timedelta
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import pytz
import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
ALL_DATES = ['2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11']
FOLD_A = ['2026-09-08', '2026-09-09']
FOLD_B = ['2026-09-10', '2026-09-11']

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
BQ_PREMARKET_TABLE = "YOUR_PROJECT.scoring_logs.premarket_scanner"
BQ_DISCOVERY_TABLE = "YOUR_PROJECT.scoring_logs.discovery_scanner"

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

RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# DATA FETCH — BQ predictions
# ==============================================================================
print(f"\n{'='*100}")
print(f"  SCANNER OPTIMIZATION HARNESS")
print(f"  Dates: {', '.join(ALL_DATES)}")
print(f"  SL={STOP_LOSS_PCT}% | Cross-validation: FoldA={FOLD_A} vs FoldB={FOLD_B}")
print(f"{'='*100}")

bq_client = bigquery.Client()
_dates_sql = ", ".join(f"'{d}'" for d in ALL_DATES)


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
print(f"    Premarket: {len(df_pre)} signals")
print(f"    Discovery: {len(df_disc)} signals")

df_all = pd.concat([df_pre, df_disc], ignore_index=True)
df_all['gate_reason'] = df_all['gate_reason'].fillna('')
df_all.loc[df_all['gate_reason'].isin(['None', 'none', 'nan']), 'gate_reason'] = ''
df_all['eval_date'] = df_all['eval_date'].astype(str)


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
df_all = df_all[df_all['scan_time_hhmm'].notna()].copy()


def _scan_pt_to_entry_et(scan_pt_hhmm, eval_date):
    pt_dt = PT.localize(datetime.strptime(f"{eval_date} {scan_pt_hhmm}", '%Y-%m-%d %H:%M'))
    et_dt = pt_dt.astimezone(ET)
    et_hhmm = et_dt.strftime('%H:%M')
    if et_hhmm < MARKET_OPEN_ET:
        return MARKET_OPEN_ET
    if et_hhmm >= MARKET_CLOSE_ET:
        return None
    return et_hhmm


df_all['entry_time_et'] = df_all.apply(
    lambda r: _scan_pt_to_entry_et(r['scan_time_hhmm'], r['eval_date']), axis=1)
df_all = df_all[df_all['entry_time_et'].notna()].copy()


# Per-ticker per-day dedup: earliest signal per (ticker, date, direction)
df_all = (df_all.sort_values('scan_time_hhmm')
          .drop_duplicates(['ticker', 'eval_date'], keep='first')
          .reset_index(drop=True))

print(f"\n  After dedup: {len(df_all)} unique (ticker, date) signals")
for dt in sorted(df_all['eval_date'].unique()):
    n = len(df_all[df_all['eval_date'] == dt])
    print(f"    {dt}: {n} signals")


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
spy_dates = list(set(r[1] for r in ticker_date_pairs))

print(f"\n  Fetching Polygon 1-min bars for {len(ticker_date_pairs)} signals + SPY...")

bar_data = {}
n_ok = 0
for i, (ticker, eval_date) in enumerate(ticker_date_pairs):
    bar_map, prev_close = fetch_minute_bars(ticker, eval_date)
    if bar_map:
        bar_data[(ticker, eval_date)] = {'bars': bar_map, 'prev_close': prev_close}
        n_ok += 1
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(ticker_date_pairs)} fetched ({n_ok} ok)")

spy_bars = {}
for dt in spy_dates:
    bm, _ = fetch_minute_bars('SPY', dt)
    if bm:
        spy_bars[dt] = bm

print(f"\n  Got price data for {n_ok}/{len(ticker_date_pairs)} signals")


# ==============================================================================
# COMPUTE PnL FOR ALL SIGNALS (multiple SL levels + horizons)
# ==============================================================================
SL_LEVELS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 999.0]  # 999 = no SL


def compute_pnl_curve(ticker, direction, entry_hhmm, eval_date, sl_pct):
    td = bar_data.get((ticker, eval_date))
    if td is None:
        return None

    bar_map = td['bars']
    prev_close = td['prev_close']

    actual_entry_hhmm = entry_hhmm
    entry_bar = bar_map.get(entry_hhmm)
    if not entry_bar:
        candidates = sorted(k for k in bar_map if k >= entry_hhmm)
        if not candidates:
            return None
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
                sl_curve[m] = -sl_pct if stopped_out else sl_curve[m - 1]
            continue

        ret = ((bar['close'] - entry_price) / entry_price) * 100
        raw_pnl = ret * sign

        if stopped_out:
            sl_curve[m] = -sl_pct
            continue

        if direction == 'LONG':
            worst = ((bar['low'] - entry_price) / entry_price) * 100
        else:
            worst = -((bar['high'] - entry_price) / entry_price) * 100

        if worst <= -sl_pct:
            stopped_out = True
            stop_minute = m
            sl_curve[m] = -sl_pct
        else:
            sl_curve[m] = round(raw_pnl, 4)

    if not sl_curve:
        return None

    gap_pct = round(((entry_price - prev_close) / prev_close) * 100, 2) if prev_close else 0.0

    return {
        'sl_curve': sl_curve,
        'stopped_out': stopped_out,
        'stop_minute': stop_minute,
        'entry_price': entry_price,
        'entry_hhmm': actual_entry_hhmm,
        'gap_pct': gap_pct,
    }


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


# Build master trade table with PnL at all SL levels
print(f"\n  Computing PnL curves for all SL levels: {SL_LEVELS}...")
records = []

for _, row in df_all.iterrows():
    ticker = row['ticker']
    direction = row['direction']
    entry_hhmm = row['entry_time_et']
    eval_date = row['eval_date']

    base_result = compute_pnl_curve(ticker, direction, entry_hhmm, eval_date, STOP_LOSS_PCT)
    if base_result is None:
        continue
    if max(base_result['sl_curve'].keys()) < 30:
        continue

    rec = {
        'ticker': ticker,
        'eval_date': eval_date,
        'scanner': row['scanner'],
        'scanner_version': str(row.get('scanner_version', '')),
        'direction': direction,
        'confidence': int(row['confidence']) if pd.notna(row['confidence']) else 0,
        'catalyst_type': str(row.get('catalyst_type', '')),
        'catalyst': str(row.get('catalyst', ''))[:200],
        'selected': bool(row['selected']) if pd.notna(row.get('selected')) else False,
        'gate_reason': str(row.get('gate_reason', '')) if pd.notna(row.get('gate_reason')) else '',
        'move_is_exhausted': bool(row.get('move_is_exhausted')) if pd.notna(row.get('move_is_exhausted')) else False,
        'remaining_move_pct': float(row['remaining_move_pct']) if pd.notna(row.get('remaining_move_pct')) else 0.0,
        'expected_move_pct': float(row['expected_move_pct']) if pd.notna(row.get('expected_move_pct')) else 0.0,
        'entry_time_et': base_result['entry_hhmm'],
        'entry_price': base_result['entry_price'],
        'gap_pct': base_result['gap_pct'],
    }

    for sl in SL_LEVELS:
        r = compute_pnl_curve(ticker, direction, entry_hhmm, eval_date, sl)
        if r is None:
            continue
        sl_key = f"sl{sl:.1f}".replace('.', '_')
        for h in HORIZONS_MIN:
            max_m = max(r['sl_curve'].keys()) if r['sl_curve'] else 0
            rec[f'pnl_{h}m_{sl_key}'] = r['sl_curve'].get(h, np.nan) if h <= max_m else np.nan
        eod_m = max(r['sl_curve'].keys()) if r['sl_curve'] else 0
        rec[f'pnl_eod_{sl_key}'] = r['sl_curve'].get(eod_m, np.nan)
        rec[f'stopped_{sl_key}'] = r['stopped_out']

    for h in HORIZONS_MIN:
        spy_h = _spy_return(base_result['entry_hhmm'], h, eval_date)
        rec[f'spy_{h}m'] = spy_h

    # Direction-remaining contradiction flag
    rem = rec['remaining_move_pct']
    if direction == 'LONG' and rem < -0.5:
        rec['dir_remaining_contradiction'] = True
    elif direction == 'SHORT' and rem > 0.5:
        rec['dir_remaining_contradiction'] = True
    else:
        rec['dir_remaining_contradiction'] = False

    # Entry time bucket
    eh = int(rec['entry_time_et'].split(':')[0])
    em = int(rec['entry_time_et'].split(':')[1])
    if eh * 60 + em <= 9 * 60 + 35:
        rec['entry_bucket'] = 'open_5min'
    elif eh * 60 + em <= 10 * 60:
        rec['entry_bucket'] = 'first_30min'
    elif eh * 60 + em <= 11 * 60:
        rec['entry_bucket'] = 'mid_morning'
    else:
        rec['entry_bucket'] = 'afternoon'

    records.append(rec)

df_trades = pd.DataFrame(records)
print(f"\n  Master trade table: {len(df_trades)} trades with 30m+ data")
print(f"    FoldA (Sep 8-9): {len(df_trades[df_trades['eval_date'].isin(FOLD_A)])}")
print(f"    FoldB (Sep 10-11): {len(df_trades[df_trades['eval_date'].isin(FOLD_B)])}")

# Save raw trade data
df_trades.to_csv(f"{RESULTS_DIR}/all_trades_raw.csv", index=False)
print(f"  Raw trades saved: {RESULTS_DIR}/all_trades_raw.csv")


# ==============================================================================
# STRATEGY EVALUATION ENGINE
# ==============================================================================
def eval_strategy(df, name, sl_pct=2.5, horizon=60):
    sl_key = f"sl{sl_pct:.1f}".replace('.', '_')
    col = f'pnl_{horizon}m_{sl_key}'
    eod_col = f'pnl_eod_{sl_key}'
    stop_col = f'stopped_{sl_key}'

    v = df[col].dropna()
    if len(v) < 3:
        return None

    eod_v = df[eod_col].dropna()
    spy_col = f'spy_{horizon}m'
    spy_v = df[spy_col].dropna()

    avg_pnl = float(v.mean())
    med_pnl = float(v.median())
    wr = float((v > 0).sum() / len(v) * 100)
    se = float(v.std() / np.sqrt(len(v))) if len(v) > 1 else 0.0
    n_stopped = int(df[stop_col].sum()) if stop_col in df else 0

    alpha_vals = []
    for _, r in df.iterrows():
        pnl = r.get(col)
        spy = r.get(spy_col)
        if pd.notna(pnl) and pd.notna(spy):
            sign = 1.0 if r['direction'] == 'LONG' else -1.0
            alpha_vals.append(pnl - spy * sign)
    avg_alpha = float(np.mean(alpha_vals)) if alpha_vals else np.nan

    sharpe = float(v.mean() / v.std()) if v.std() > 0 else 0.0

    return {
        'strategy': name,
        'n': len(v),
        'avg_pnl': round(avg_pnl, 4),
        'median_pnl': round(med_pnl, 4),
        'win_rate': round(wr, 1),
        'se': round(se, 4),
        'avg_alpha': round(avg_alpha, 4) if pd.notna(avg_alpha) else None,
        'sharpe': round(sharpe, 3),
        'stop_rate': round(n_stopped / len(v) * 100, 1) if len(v) > 0 else 0,
        'avg_eod': round(float(eod_v.mean()), 4) if len(eod_v) > 0 else None,
        'total_pnl': round(float(v.sum()), 2),
        'sl_pct': sl_pct,
        'horizon': horizon,
    }


def eval_cross_validated(df, name, filter_fn, sl_pct=2.5, horizon=60):
    df_a = df[df['eval_date'].isin(FOLD_A)]
    df_b = df[df['eval_date'].isin(FOLD_B)]

    df_a_filt = filter_fn(df_a)
    df_b_filt = filter_fn(df_b)
    df_all_filt = filter_fn(df)

    s_all = eval_strategy(df_all_filt, name, sl_pct, horizon)
    s_a = eval_strategy(df_a_filt, f"{name}_foldA", sl_pct, horizon)
    s_b = eval_strategy(df_b_filt, f"{name}_foldB", sl_pct, horizon)

    if s_all is None:
        return None

    result = s_all.copy()
    result['fold_a'] = s_a
    result['fold_b'] = s_b

    # Cross-validation check: positive in both folds
    a_pos = s_a is not None and s_a['avg_pnl'] > 0
    b_pos = s_b is not None and s_b['avg_pnl'] > 0
    result['cv_pass'] = a_pos and b_pos
    result['cv_both_positive'] = a_pos and b_pos

    return result


# ==============================================================================
# DEFINE STRATEGIES
# ==============================================================================
strategies = []


# ── BASELINE ──
strategies.append(('BASELINE_all', lambda df: df))
strategies.append(('BASELINE_selected', lambda df: df[df['selected'] == True]))

# ── DIRECTION FILTERS ──
strategies.append(('DIR_short_only', lambda df: df[df['direction'] == 'SHORT']))
strategies.append(('DIR_long_only', lambda df: df[df['direction'] == 'LONG']))

# ── SCANNER FILTERS ──
strategies.append(('SCAN_discovery_only', lambda df: df[df['scanner'] == 'discovery']))
strategies.append(('SCAN_premarket_only', lambda df: df[df['scanner'] == 'premarket']))

# ── CATALYST TYPE FILTERS ──
GOOD_CATALYSTS = ['regulatory', 'offering', 'squeeze']
BAD_CATALYSTS = ['analyst', 'earnings', 'deal', 'unknown', 'bankruptcy', 'breaking']
BEST_CATALYSTS = ['offering', 'regulatory']

strategies.append(('CAT_good_only', lambda df: df[df['catalyst_type'].isin(GOOD_CATALYSTS)]))
strategies.append(('CAT_best_only', lambda df: df[df['catalyst_type'].isin(BEST_CATALYSTS)]))
strategies.append(('CAT_exclude_bad', lambda df: df[~df['catalyst_type'].isin(BAD_CATALYSTS)]))
strategies.append(('CAT_exclude_analyst', lambda df: df[df['catalyst_type'] != 'analyst']))
strategies.append(('CAT_exclude_earnings', lambda df: df[df['catalyst_type'] != 'earnings']))
strategies.append(('CAT_exclude_unknown', lambda df: df[df['catalyst_type'] != 'unknown']))

# ── COMBINED: DIRECTION + CATALYST ──
strategies.append(('COMBO_short_good_cat',
    lambda df: df[(df['direction'] == 'SHORT') & (df['catalyst_type'].isin(GOOD_CATALYSTS))]))
strategies.append(('COMBO_short_best_cat',
    lambda df: df[(df['direction'] == 'SHORT') & (df['catalyst_type'].isin(BEST_CATALYSTS))]))
strategies.append(('COMBO_short_offering',
    lambda df: df[(df['direction'] == 'SHORT') & (df['catalyst_type'] == 'offering')]))
strategies.append(('COMBO_short_regulatory',
    lambda df: df[(df['direction'] == 'SHORT') & (df['catalyst_type'] == 'regulatory')]))
strategies.append(('COMBO_long_analyst',
    lambda df: df[(df['direction'] == 'LONG') & (df['catalyst_type'] == 'analyst')]))
strategies.append(('COMBO_short_excl_analyst',
    lambda df: df[(df['direction'] == 'SHORT') & (df['catalyst_type'] != 'analyst')]))

# Catalyst-direction alignment (from root cause analysis Table)
def catalyst_direction_filter(df):
    masks = []
    masks.append((df['catalyst_type'] == 'offering') & (df['direction'] == 'SHORT'))
    masks.append((df['catalyst_type'] == 'deal') & (df['direction'] == 'SHORT'))
    masks.append((df['catalyst_type'] == 'earnings') & (df['direction'] == 'SHORT'))
    masks.append((df['catalyst_type'] == 'analyst') & (df['direction'] == 'LONG'))
    masks.append((df['catalyst_type'] == 'regulatory') & (df['direction'] == 'SHORT'))
    masks.append((df['catalyst_type'] == 'squeeze') & (df['direction'] == 'LONG'))
    masks.append((df['catalyst_type'] == 'macro'))
    masks.append((df['catalyst_type'] == 'insider'))
    combined = pd.Series(False, index=df.index)
    for m in masks:
        combined = combined | m
    return df[combined]

strategies.append(('COMBO_catalyst_direction_aligned', catalyst_direction_filter))

# ── SCANNER + DIRECTION ──
strategies.append(('COMBO_discovery_short',
    lambda df: df[(df['scanner'] == 'discovery') & (df['direction'] == 'SHORT')]))
strategies.append(('COMBO_discovery_long',
    lambda df: df[(df['scanner'] == 'discovery') & (df['direction'] == 'LONG')]))

# ── ENTRY TIME FILTERS ──
strategies.append(('TIME_open_5min',
    lambda df: df[df['entry_bucket'] == 'open_5min']))
strategies.append(('TIME_first_30min',
    lambda df: df[df['entry_bucket'].isin(['open_5min', 'first_30min'])]))
strategies.append(('TIME_morning_only',
    lambda df: df[df['entry_bucket'].isin(['open_5min', 'first_30min', 'mid_morning'])]))
strategies.append(('TIME_no_afternoon',
    lambda df: df[df['entry_bucket'] != 'afternoon']))

# ── GAP FILTERS ──
strategies.append(('GAP_small_abs_lt5',
    lambda df: df[df['gap_pct'].abs() < 5]))
strategies.append(('GAP_small_abs_lt10',
    lambda df: df[df['gap_pct'].abs() < 10]))
strategies.append(('GAP_medium_2to10',
    lambda df: df[(df['gap_pct'].abs() >= 2) & (df['gap_pct'].abs() < 10)]))
strategies.append(('GAP_large_gt10',
    lambda df: df[df['gap_pct'].abs() >= 10]))

# ── EXHAUSTION FILTERS ──
strategies.append(('EXHAUST_not_exhausted',
    lambda df: df[df['move_is_exhausted'] == False]))
strategies.append(('EXHAUST_exhausted_only',
    lambda df: df[df['move_is_exhausted'] == True]))

# ── DIRECTION-REMAINING CONTRADICTION ──
strategies.append(('CONTRA_no_contradiction',
    lambda df: df[df['dir_remaining_contradiction'] == False]))

# ── GATE REMOVAL (include gated signals) ──
strategies.append(('GATE_include_all',
    lambda df: df))  # same as baseline
strategies.append(('GATE_no_gate_only',
    lambda df: df[df['gate_reason'] == '']))

# ── CONFIDENCE FILTERS ──
strategies.append(('CONF_ge4', lambda df: df[df['confidence'] >= 4]))
strategies.append(('CONF_ge5', lambda df: df[df['confidence'] >= 5]))
strategies.append(('CONF_le5', lambda df: df[df['confidence'] <= 5]))
strategies.append(('CONF_3to5', lambda df: df[(df['confidence'] >= 3) & (df['confidence'] <= 5)]))

# ── MULTI-FACTOR COMPOSITES ──

# Strategy S1: SHORT + good catalyst + no contradiction
def s1_filter(df):
    return df[
        (df['direction'] == 'SHORT') &
        (df['catalyst_type'].isin(GOOD_CATALYSTS)) &
        (df['dir_remaining_contradiction'] == False)
    ]
strategies.append(('S1_short_goodcat_nocontra', s1_filter))

# Strategy S2: Discovery + SHORT + no bad catalyst
def s2_filter(df):
    return df[
        (df['scanner'] == 'discovery') &
        (df['direction'] == 'SHORT') &
        (~df['catalyst_type'].isin(BAD_CATALYSTS))
    ]
strategies.append(('S2_disc_short_nobadcat', s2_filter))

# Strategy S3: Catalyst-direction aligned + no contradiction + morning
def s3_filter(df):
    aligned = catalyst_direction_filter(df)
    return aligned[
        (aligned['dir_remaining_contradiction'] == False) &
        (aligned['entry_bucket'].isin(['open_5min', 'first_30min', 'mid_morning']))
    ]
strategies.append(('S3_aligned_nocontra_morning', s3_filter))

# Strategy S4: SHORT + offering/regulatory + discovery + small gap
def s4_filter(df):
    return df[
        (df['direction'] == 'SHORT') &
        (df['catalyst_type'].isin(['offering', 'regulatory'])) &
        (df['scanner'] == 'discovery') &
        (df['gap_pct'].abs() < 15)
    ]
strategies.append(('S4_short_offreg_disc_smallgap', s4_filter))

# Strategy S5: No bad catalyst + no contradiction + not exhausted
def s5_filter(df):
    return df[
        (~df['catalyst_type'].isin(BAD_CATALYSTS)) &
        (df['dir_remaining_contradiction'] == False) &
        (df['move_is_exhausted'] == False)
    ]
strategies.append(('S5_nobadcat_nocontra_notexhaust', s5_filter))

# Strategy S6: SHORT + no bad catalyst + morning + gap < 20%
def s6_filter(df):
    return df[
        (df['direction'] == 'SHORT') &
        (~df['catalyst_type'].isin(BAD_CATALYSTS)) &
        (df['entry_bucket'].isin(['open_5min', 'first_30min', 'mid_morning'])) &
        (df['gap_pct'].abs() < 20)
    ]
strategies.append(('S6_short_nobad_morning_smallgap', s6_filter))

# Strategy S7: Only trades where gap aligns with direction
def s7_filter(df):
    long_gap_down = (df['direction'] == 'LONG') & (df['gap_pct'] < -1)
    short_gap_up = (df['direction'] == 'SHORT') & (df['gap_pct'] > 1)
    return df[long_gap_down | short_gap_up]
strategies.append(('S7_gap_opposes_direction', s7_filter))

# Strategy S8: Fable's residual fix proxy — trade opposite gap direction for small gaps
def s8_filter(df):
    small_gap = df['gap_pct'].abs() < 5
    return df[small_gap & (~df['catalyst_type'].isin(BAD_CATALYSTS))]
strategies.append(('S8_small_gap_no_bad_cat', s8_filter))

# Strategy S9: HIGH conviction subset — selected + no contradiction + good catalyst
def s9_filter(df):
    return df[
        (df['selected'] == True) &
        (df['dir_remaining_contradiction'] == False) &
        (~df['catalyst_type'].isin(BAD_CATALYSTS))
    ]
strategies.append(('S9_selected_clean', s9_filter))

# Strategy S10: Discovery + good catalyst only (any direction)
def s10_filter(df):
    return df[
        (df['scanner'] == 'discovery') &
        (df['catalyst_type'].isin(GOOD_CATALYSTS))
    ]
strategies.append(('S10_discovery_goodcat', s10_filter))

# Strategy S11: SHORT + not analyst + not unknown + gap > 2%
def s11_filter(df):
    return df[
        (df['direction'] == 'SHORT') &
        (df['catalyst_type'] != 'analyst') &
        (df['catalyst_type'] != 'unknown') &
        (df['gap_pct'].abs() >= 2)
    ]
strategies.append(('S11_short_known_cat_gap2', s11_filter))

# Strategy S12: Catalyst-aligned + discovery only
def s12_filter(df):
    aligned = catalyst_direction_filter(df)
    return aligned[aligned['scanner'] == 'discovery']
strategies.append(('S12_aligned_discovery', s12_filter))


# ==============================================================================
# RUN ALL STRATEGIES AT MULTIPLE SL LEVELS AND HORIZONS
# ==============================================================================
print(f"\n  Running {len(strategies)} strategies × {len(SL_LEVELS)} SL levels × {len(HORIZONS_MIN)} horizons...")

all_results = []

for sl in SL_LEVELS:
    for h in HORIZONS_MIN:
        for name, filter_fn in strategies:
            result = eval_cross_validated(df_trades, name, filter_fn, sl_pct=sl, horizon=h)
            if result:
                all_results.append(result)

print(f"  {len(all_results)} strategy evaluations completed")


# ==============================================================================
# FILTER TO BEST STRATEGIES
# ==============================================================================
# Primary criteria at SL=2.5, horizon=60m
primary = [r for r in all_results if r['sl_pct'] == 2.5 and r['horizon'] == 60]
primary_sorted = sorted(primary, key=lambda x: x['avg_pnl'], reverse=True)

# Top strategies by avg PnL
print(f"\n\n{'='*120}")
print(f"  TOP 20 STRATEGIES @ SL=2.5%, 60m horizon (sorted by avg PnL)")
print(f"{'='*120}")
print(f"  {'#':>3} {'Strategy':<42} {'n':>4} {'AvgPnL':>8} {'MedPnL':>8} {'WR':>6} {'SE':>6} {'Alpha':>8} "
      f"{'Sharpe':>7} {'StopR':>6} {'CV':>4} {'FoldA':>8} {'FoldB':>8}")
print(f"  {'':─<118}")

for i, r in enumerate(primary_sorted[:20]):
    fa = r['fold_a']['avg_pnl'] if r['fold_a'] else float('nan')
    fb = r['fold_b']['avg_pnl'] if r['fold_b'] else float('nan')
    cv = "YES" if r['cv_pass'] else "no"
    alpha_s = f"{r['avg_alpha']:+7.3f}%" if r['avg_alpha'] is not None else '     n/a'
    print(f"  {i+1:>3} {r['strategy']:<42} {r['n']:>4} {r['avg_pnl']:>+7.3f}% {r['median_pnl']:>+7.3f}% "
          f"{r['win_rate']:>5.1f}% {r['se']:>5.3f} {alpha_s} {r['sharpe']:>+6.3f} "
          f"{r['stop_rate']:>5.1f}% {cv:>4} {fa:>+7.3f}% {fb:>+7.3f}%")


# ==============================================================================
# CROSS-VALIDATED WINNERS (positive in both folds)
# ==============================================================================
cv_winners = [r for r in primary if r['cv_pass'] and r['n'] >= 5]
cv_winners_sorted = sorted(cv_winners, key=lambda x: x['avg_pnl'], reverse=True)

print(f"\n\n{'='*120}")
print(f"  CROSS-VALIDATED WINNERS (positive PnL in BOTH folds, n>=5)")
print(f"{'='*120}")
print(f"  {'#':>3} {'Strategy':<42} {'n':>4} {'AvgPnL':>8} {'WR':>6} {'Alpha':>8} "
      f"{'FoldA':>8} {'nA':>4} {'FoldB':>8} {'nB':>4}")
print(f"  {'':─<100}")

for i, r in enumerate(cv_winners_sorted):
    fa = r['fold_a']['avg_pnl'] if r['fold_a'] else float('nan')
    fb = r['fold_b']['avg_pnl'] if r['fold_b'] else float('nan')
    na = r['fold_a']['n'] if r['fold_a'] else 0
    nb = r['fold_b']['n'] if r['fold_b'] else 0
    alpha_s = f"{r['avg_alpha']:+7.3f}%" if r['avg_alpha'] is not None else '     n/a'
    print(f"  {i+1:>3} {r['strategy']:<42} {r['n']:>4} {r['avg_pnl']:>+7.3f}% {r['win_rate']:>5.1f}% "
          f"{alpha_s} {fa:>+7.3f}% {na:>4} {fb:>+7.3f}% {nb:>4}")


# ==============================================================================
# OPTIMAL SL ANALYSIS — for top strategies
# ==============================================================================
top_strategy_names = [r['strategy'] for r in cv_winners_sorted[:10]]
if not top_strategy_names:
    top_strategy_names = [r['strategy'] for r in primary_sorted[:5]]

print(f"\n\n{'='*120}")
print(f"  OPTIMAL STOP LOSS — Top strategies across SL levels (60m)")
print(f"{'='*120}")

for sname in top_strategy_names[:5]:
    print(f"\n  {sname}:")
    print(f"  {'SL':>5} {'n':>4} {'AvgPnL':>8} {'WR':>6} {'StopR':>6} {'Sharpe':>7}")
    for sl in SL_LEVELS:
        r = next((x for x in all_results if x['strategy'] == sname and x['sl_pct'] == sl and x['horizon'] == 60), None)
        if r:
            print(f"  {sl:>4.1f}% {r['n']:>4} {r['avg_pnl']:>+7.3f}% {r['win_rate']:>5.1f}% "
                  f"{r['stop_rate']:>5.1f}% {r['sharpe']:>+6.3f}")


# ==============================================================================
# STRATEGIES ABOVE 2% TARGET
# ==============================================================================
target_strategies = [r for r in primary if r['avg_pnl'] >= 2.0 and r['n'] >= 3]
target_sorted = sorted(target_strategies, key=lambda x: (x['cv_pass'], x['n'], x['avg_pnl']), reverse=True)

print(f"\n\n{'='*120}")
print(f"  STRATEGIES WITH AVG PnL >= +2.0% (SL=2.5%, 60m)")
print(f"{'='*120}")

if target_sorted:
    print(f"  {'#':>3} {'Strategy':<42} {'n':>4} {'AvgPnL':>8} {'WR':>6} {'CV':>4} {'FoldA':>8} {'FoldB':>8}")
    print(f"  {'':─<90}")
    for i, r in enumerate(target_sorted):
        fa = r['fold_a']['avg_pnl'] if r['fold_a'] else float('nan')
        fb = r['fold_b']['avg_pnl'] if r['fold_b'] else float('nan')
        cv = "YES" if r['cv_pass'] else "no"
        print(f"  {i+1:>3} {r['strategy']:<42} {r['n']:>4} {r['avg_pnl']:>+7.3f}% {r['win_rate']:>5.1f}% "
              f"{cv:>4} {fa:>+7.3f}% {fb:>+7.3f}%")
else:
    print("  No strategies reached +2.0% avg PnL at SL=2.5%/60m. Checking other SL levels...")
    for sl in SL_LEVELS:
        t2 = [r for r in all_results if r['avg_pnl'] >= 2.0 and r['n'] >= 3
              and r['sl_pct'] == sl and r['horizon'] == 60]
        if t2:
            t2s = sorted(t2, key=lambda x: x['avg_pnl'], reverse=True)
            print(f"\n  At SL={sl}%:")
            for r in t2s[:5]:
                cv = "YES" if r['cv_pass'] else "no"
                print(f"    {r['strategy']:<42} n={r['n']:>3} avg={r['avg_pnl']:>+.3f}% WR={r['win_rate']:.0f}% CV={cv}")


# ==============================================================================
# CHARTS
# ==============================================================================

# Chart 1: Strategy comparison bar chart
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig1.suptitle('Scanner Optimization: Strategy Comparison @ SL=2.5%, 60m', fontsize=14, fontweight='bold')

# Top 15 by avg PnL
top15 = primary_sorted[:15]
names = [r['strategy'] for r in top15]
vals = [r['avg_pnl'] for r in top15]
colors = ['#27ae60' if v > 0 else '#c0392b' for v in vals]
ns = [r['n'] for r in top15]

ax1.barh(range(len(names)), vals, color=colors, height=0.7)
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels([f"{n} (n={nn})" for n, nn in zip(names, ns)], fontsize=8)
for i, v in enumerate(vals):
    ax1.text(v + 0.05 * (1 if v >= 0 else -1), i, f"{v:+.2f}%", va='center', fontsize=8)
ax1.axvline(x=0, color='black', lw=0.8)
ax1.axvline(x=2.0, color='green', lw=1, ls='--', alpha=0.5, label='+2% target')
ax1.set_xlabel('Avg PnL @60m %')
ax1.set_title('Top 15 Strategies by Avg PnL')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

# CV winners
if cv_winners_sorted:
    cv_names = [r['strategy'] for r in cv_winners_sorted[:10]]
    cv_fa = [r['fold_a']['avg_pnl'] if r['fold_a'] else 0 for r in cv_winners_sorted[:10]]
    cv_fb = [r['fold_b']['avg_pnl'] if r['fold_b'] else 0 for r in cv_winners_sorted[:10]]

    x = np.arange(len(cv_names))
    w = 0.35
    ax2.barh(x - w/2, cv_fa, w, label='Fold A (Sep 8-9)', color='#3498db', alpha=0.8)
    ax2.barh(x + w/2, cv_fb, w, label='Fold B (Sep 10-11)', color='#e67e22', alpha=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(cv_names, fontsize=8)
    ax2.axvline(x=0, color='black', lw=0.8)
    ax2.set_xlabel('Avg PnL %')
    ax2.set_title('Cross-Validated Winners: Fold A vs Fold B')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
else:
    ax2.text(0.5, 0.5, 'No CV winners', ha='center', va='center', transform=ax2.transAxes)

plt.tight_layout()
fig1_path = f"{OUTPUT_DIR}/optimization_strategies.png"
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart 1 saved: {fig1_path}")
plt.close()


# Chart 2: SL sensitivity for top strategies
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))
fig2.suptitle('Stop Loss Sensitivity — Top Strategies', fontsize=13, fontweight='bold')

for ax, metric, ylabel in [(axes2[0], 'avg_pnl', 'Avg PnL @60m %'),
                             (axes2[1], 'win_rate', 'Win Rate @60m %')]:
    for sname in top_strategy_names[:5]:
        xs = []
        ys = []
        for sl in SL_LEVELS:
            if sl > 10:
                continue
            r = next((x for x in all_results if x['strategy'] == sname and x['sl_pct'] == sl and x['horizon'] == 60), None)
            if r:
                xs.append(sl)
                ys.append(r[metric])
        if xs:
            ax.plot(xs, ys, 'o-', label=sname, lw=1.5, markersize=5)
    ax.set_xlabel('Stop Loss %')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    if metric == 'avg_pnl':
        ax.axhline(y=0, color='black', lw=0.8)
        ax.axhline(y=2.0, color='green', lw=0.8, ls='--', alpha=0.5)

plt.tight_layout()
fig2_path = f"{OUTPUT_DIR}/optimization_sl_sensitivity.png"
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"  Chart 2 saved: {fig2_path}")
plt.close()


# Chart 3: Trade-level detail for top strategy
if cv_winners_sorted:
    best = cv_winners_sorted[0]
    best_name = best['strategy']
    best_filter = dict(strategies)[best_name]
    df_best = best_filter(df_trades)

    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
    fig3.suptitle(f'Best CV Strategy: {best_name} (n={best["n"]}, avg={best["avg_pnl"]:+.2f}%)',
                  fontsize=13, fontweight='bold')

    # 3a: Per-trade PnL waterfall
    ax = axes3[0, 0]
    sl_key = 'sl2_5'
    df_show = df_best.dropna(subset=[f'pnl_60m_{sl_key}']).sort_values(f'pnl_60m_{sl_key}')
    if len(df_show) > 0:
        colors = ['#27ae60' if v > 0 else '#c0392b' for v in df_show[f'pnl_60m_{sl_key}']]
        ax.barh(range(len(df_show)), df_show[f'pnl_60m_{sl_key}'], color=colors, height=0.7)
        ax.set_yticks(range(len(df_show)))
        ax.set_yticklabels(
            [f"{r['ticker']} {r['direction'][0]} {r['eval_date'][-5:]}" for _, r in df_show.iterrows()],
            fontsize=7)
        ax.axvline(x=0, color='black', lw=0.8)
        ax.set_xlabel('PnL @60m %')
        ax.set_title(f'Per-Trade PnL')
        ax.grid(True, alpha=0.3, axis='x')

    # 3b: By catalyst type
    ax = axes3[0, 1]
    cat_stats = df_best.groupby('catalyst_type')[f'pnl_60m_{sl_key}'].agg(['mean', 'count']).reset_index()
    cat_stats = cat_stats[cat_stats['count'] >= 1].sort_values('mean')
    if len(cat_stats) > 0:
        colors = ['#27ae60' if v > 0 else '#c0392b' for v in cat_stats['mean']]
        ax.barh(range(len(cat_stats)), cat_stats['mean'], color=colors, height=0.6)
        ax.set_yticks(range(len(cat_stats)))
        ax.set_yticklabels([f"{r['catalyst_type']} (n={r['count']})" for _, r in cat_stats.iterrows()], fontsize=9)
        ax.axvline(x=0, color='black', lw=0.8)
        ax.set_xlabel('Avg PnL @60m %')
        ax.set_title('By Catalyst Type')
        ax.grid(True, alpha=0.3, axis='x')

    # 3c: By date
    ax = axes3[1, 0]
    date_stats = df_best.groupby('eval_date')[f'pnl_60m_{sl_key}'].agg(['mean', 'count']).reset_index()
    if len(date_stats) > 0:
        colors = ['#27ae60' if v > 0 else '#c0392b' for v in date_stats['mean']]
        ax.bar(range(len(date_stats)), date_stats['mean'], color=colors, alpha=0.8)
        ax.set_xticks(range(len(date_stats)))
        ax.set_xticklabels([f"{r['eval_date'][-5:]}\nn={r['count']}" for _, r in date_stats.iterrows()])
        for i, (_, r) in enumerate(date_stats.iterrows()):
            ax.annotate(f"{r['mean']:+.2f}%", xy=(i, r['mean']), ha='center',
                        va='bottom' if r['mean'] >= 0 else 'top', fontsize=10, fontweight='bold')
        ax.axhline(y=0, color='black', lw=0.8)
        ax.set_ylabel('Avg PnL @60m %')
        ax.set_title('By Date')
        ax.grid(True, alpha=0.3, axis='y')

    # 3d: Direction split
    ax = axes3[1, 1]
    for d, c in [('LONG', '#27ae60'), ('SHORT', '#c0392b')]:
        df_d = df_best[df_best['direction'] == d]
        v = df_d[f'pnl_60m_{sl_key}'].dropna()
        if len(v) > 0:
            ax.hist(v, bins=np.linspace(-3, 10, 25), alpha=0.6, color=c,
                    label=f'{d} (n={len(v)}, avg={v.mean():+.2f}%)')
    ax.axvline(x=0, color='black', lw=1, ls='--')
    ax.set_xlabel('PnL @60m %')
    ax.set_ylabel('Count')
    ax.set_title('PnL Distribution by Direction')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig3_path = f"{OUTPUT_DIR}/optimization_best_strategy.png"
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"  Chart 3 saved: {fig3_path}")
    plt.close()


# ==============================================================================
# SAVE FULL RESULTS
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


results_json = {
    'generated': datetime.utcnow().isoformat(),
    'config': {
        'dates': ALL_DATES, 'fold_a': FOLD_A, 'fold_b': FOLD_B,
        'sl_levels': SL_LEVELS, 'horizons': HORIZONS_MIN,
    },
    'n_strategies': len(strategies),
    'n_evaluations': len(all_results),
    'n_trades': len(df_trades),
    'target_pnl': 2.0,
    'top_20_60m_sl25': [
        {k: v for k, v in r.items() if k not in ('fold_a', 'fold_b')}
        for r in primary_sorted[:20]
    ],
    'cv_winners': [
        r for r in cv_winners_sorted
    ],
    'above_2pct': [
        {k: v for k, v in r.items() if k not in ('fold_a', 'fold_b')}
        for r in target_sorted
    ],
}

json_path = f"{RESULTS_DIR}/optimization_results.json"
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2, cls=_NumpyEncoder)
print(f"\n  Full results JSON saved: {json_path}")


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"\n\n{'='*120}")
print(f"  OPTIMIZATION SUMMARY")
print(f"{'='*120}")
print(f"\n  Total trades: {len(df_trades)}")
print(f"  Total strategies tested: {len(strategies)}")
print(f"  Total evaluations: {len(all_results)}")
print(f"\n  Cross-validated winners (positive both folds): {len(cv_winners_sorted)}")
print(f"  Strategies above +2% avg PnL: {len(target_sorted)}")

if cv_winners_sorted:
    print(f"\n  BEST CROSS-VALIDATED STRATEGY:")
    best = cv_winners_sorted[0]
    print(f"    {best['strategy']}")
    print(f"    avg={best['avg_pnl']:+.3f}%  WR={best['win_rate']:.0f}%  n={best['n']}  "
          f"alpha={best['avg_alpha']:+.3f}%  sharpe={best['sharpe']:+.3f}")
    if best['fold_a']:
        print(f"    FoldA: avg={best['fold_a']['avg_pnl']:+.3f}%  n={best['fold_a']['n']}")
    if best['fold_b']:
        print(f"    FoldB: avg={best['fold_b']['avg_pnl']:+.3f}%  n={best['fold_b']['n']}")

print(f"\n{'='*120}\n")
