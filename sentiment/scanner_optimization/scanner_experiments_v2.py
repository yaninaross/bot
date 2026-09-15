# %% ── Scanner Experiments V2 — Extended Analysis ──────────────────────────────
#
# Goes beyond the original creative_experiments.py with 10 new experiments
# testing fundamentally different questions about the scanner pipeline.
#
# EXP S1: Stop Design — MAE/MFE analysis, ATR-scaled stops, percentile stops
# EXP S2: Horizon Extension — 2h, EOD, T+1, T+3 (beta-adjusted)
# EXP S3: Counterfactual Universe — all gappers vs scanner-flagged
# EXP S4: Cross-Sectional Ranking — daily tercile long/short, dollar-neutral
# EXP S5: Magnitude Without Direction — |expected_move| → |realized|
# EXP S6: Stability Across Runs — multi-run consistency as feature
# EXP S7: Premarket Path Shape — retrace, time-since-high, late volume
# EXP S8: Opening Range Breakout — ORB with catalyst qualifier
# EXP S9: Ticker Reaction History — per-ticker gap continuation rates
# EXP S10: Earnings Module — dedicated earnings analysis
# BONUS: Version Breakdown, Robustness re-evaluation of V1 findings
#
# All statistics: median + 10%-trimmed mean + leave-one-out for n<50.
# Cross-validation: Fold A (Sep 8-9) vs Fold B (Sep 10-11).
# CAVEAT: Fold-CV = version-CV (A=V2-V3.3, B=V3.8-V4).
#
# Colab-ready, config vars at top.

import os, sys, json, warnings, re, time
from datetime import datetime, timedelta
from collections import defaultdict, Counter

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
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-bigquery", "--quiet"])
    from google.cloud import bigquery

try:
    from polygon import RESTClient as PolygonClient
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "polygon-api-client", "--quiet"])
    from polygon import RESTClient as PolygonClient

try:
    import requests as _requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "--quiet"])
    import requests as _requests

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

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY")
BQ_PREMARKET_TABLE = "YOUR_GCP_PROJECT.scoring_logs.premarket_scanner"
BQ_DISCOVERY_TABLE = "YOUR_GCP_PROJECT.scoring_logs.discovery_scanner"

INTRADAY_MINUTES = 390
MARKET_OPEN_ET = "09:30"
MARKET_CLOSE_ET = "16:00"

ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')

try:
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    OUTPUT_DIR = os.getcwd()
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Experiment toggles — set False to skip expensive experiments
RUN_STOP_DESIGN    = True   # S1: MAE/MFE + stop grid
RUN_HORIZON        = True   # S2: multi-day horizon
RUN_COUNTERFACTUAL = True   # S3: all gappers vs flagged
RUN_CROSS_SECTION  = True   # S4: daily tercile L/S
RUN_MAGNITUDE      = True   # S5: |move| prediction
RUN_STABILITY      = True   # S6: multi-run consistency
RUN_PM_SHAPE       = True   # S7: premarket path
RUN_ORB            = True   # S8: opening range breakout
RUN_TICKER_HISTORY = True   # S9: gap continuation history
RUN_EARNINGS       = True   # S10: earnings module
RUN_VERSION_SPLIT  = True   # version breakdown

# Counterfactual: max non-flagged gappers to fetch per day (API budget)
COUNTERFACTUAL_MAX_PER_DAY = 40
COUNTERFACTUAL_MIN_GAP_PCT = 3.0

# Horizon extension: trading days forward
HORIZON_FORWARD_DAYS = 10


# ==============================================================================
# ROBUSTNESS UTILITIES
# ==============================================================================
_W = 100

def robust_stats(values, label='', direction_filter=None):
    """Mean, median, 10%-trimmed mean, SE, WR, and leave-one-out for n<50."""
    v = pd.Series(values).dropna()
    n = len(v)
    if n == 0:
        return {'n': 0, 'avg': np.nan, 'med': np.nan, 'trim': np.nan,
                'se': np.nan, 'wr': np.nan, 'loo_min': np.nan, 'loo_max': np.nan}
    avg = float(v.mean())
    med = float(v.median())
    se = float(v.std() / np.sqrt(n)) if n > 1 else 0.0
    wr = float((v > 0).sum() / n * 100)
    # 10%-trimmed mean
    if n >= 10:
        k = max(1, int(n * 0.1))
        trim = float(v.sort_values().iloc[k:-k].mean())
    else:
        trim = avg
    # leave-one-out range (for small samples)
    loo_min, loo_max = avg, avg
    if n < 50 and n > 1:
        loo_means = [(v.sum() - val) / (n - 1) for val in v]
        loo_min = min(loo_means)
        loo_max = max(loo_means)
    return {'n': n, 'avg': avg, 'med': med, 'trim': trim,
            'se': se, 'wr': wr, 'loo_min': loo_min, 'loo_max': loo_max}


def fold_robust(values, dates, fold_a=FOLD_A, fold_b=FOLD_B):
    """Per-fold robust stats + cross-validation check."""
    v = pd.Series(values).dropna()
    d = pd.Series(dates).loc[v.index]
    a = v[d.isin(fold_a)]
    b = v[d.isin(fold_b)]
    sa = robust_stats(a)
    sb = robust_stats(b)
    cv = (sa['n'] > 0 and sb['n'] > 0 and sa['avg'] > 0 and sb['avg'] > 0)
    return {'all': robust_stats(v), 'A': sa, 'B': sb, 'cv': cv}


def print_robust_row(label, rs, fold_a_rs=None, fold_b_rs=None, cv=None):
    """Print one row with full robustness stats."""
    s = rs
    cv_s = 'YES' if cv else 'no ' if cv is not None else '   '
    loo = ''
    if s['n'] < 50 and s['n'] > 1:
        loo = f" LOO=[{s['loo_min']:+.2f},{s['loo_max']:+.2f}]"
    fa = f"A={fold_a_rs['avg']:+.3f}%({fold_a_rs['n']})" if fold_a_rs and fold_a_rs['n'] > 0 else ''
    fb = f"B={fold_b_rs['avg']:+.3f}%({fold_b_rs['n']})" if fold_b_rs and fold_b_rs['n'] > 0 else ''
    print(f"    {label:<40} n={s['n']:>4}  avg={s['avg']:+.3f}%  med={s['med']:+.3f}%  "
          f"trim={s['trim']:+.3f}%  WR={s['wr']:.0f}%  SE={s['se']:.3f}  CV:{cv_s}  "
          f"{fa}  {fb}{loo}")


# ==============================================================================
# DATA FETCH — BQ SIGNALS (ALL RUNS for stability analysis)
# ==============================================================================
print(f"\n{'='*_W}")
print(f"  SCANNER EXPERIMENTS V2 — EXTENDED ANALYSIS")
print(f"  Dates: {', '.join(ALL_DATES)}")
print(f"  Folds: A={FOLD_A}, B={FOLD_B}")
print(f"{'='*_W}")

bq_client = bigquery.Client()
_dates_sql = ", ".join(f"'{d}'" for d in ALL_DATES)


def _bq_fetch(table, scanner_name, directional_only=True):
    dir_filter = "AND direction IN ('LONG', 'SHORT')" if directional_only else ""
    if scanner_name == 'premarket':
        q = f"""
        SELECT CAST(scan_date AS STRING) AS eval_date,
            ticker, direction, confidence, remaining_move_pct,
            move_already_absorbed_pct AS move_already_priced_pct,
            expected_move_pct, composite_score AS tradability_score,
            catalyst_type, catalyst, selected, scanner_version,
            scan_time_pt, gate_reason, move_is_exhausted,
            rank_in_direction, COALESCE(stage2_model, '') AS stage2_model
        FROM `{table}`
        WHERE scan_date IN ({_dates_sql}) {dir_filter}
        ORDER BY scan_date, scan_time_pt, ticker
        """
    else:
        q = f"""
        SELECT CAST(scan_date AS STRING) AS eval_date,
            ticker, direction, confidence, remaining_move_pct,
            move_already_priced_pct, expected_move_pct, tradability_score,
            catalyst_type, catalyst, selected, scanner_version,
            scan_time_pt, gate_reason, move_is_exhausted,
            rank_in_direction, COALESCE(stage2_model, '') AS stage2_model
        FROM `{table}`
        WHERE scan_date IN ({_dates_sql}) {dir_filter}
        ORDER BY scan_date, scan_time_pt, ticker
        """
    df = bq_client.query(q).to_dataframe()
    df['scanner'] = scanner_name
    return df


print(f"\n  Fetching ALL signals from BQ...")
df_pre_all = _bq_fetch(BQ_PREMARKET_TABLE, 'premarket', directional_only=False)
df_disc_all = _bq_fetch(BQ_DISCOVERY_TABLE, 'discovery', directional_only=False)
print(f"    Premarket: {len(df_pre_all)} total, Discovery: {len(df_disc_all)} total")

df_raw = pd.concat([df_pre_all, df_disc_all], ignore_index=True)
df_raw['gate_reason'] = df_raw['gate_reason'].fillna('')
df_raw.loc[df_raw['gate_reason'].isin(['None', 'none', 'nan']), 'gate_reason'] = ''
df_raw['eval_date'] = df_raw['eval_date'].astype(str)
df_raw['catalyst_type'] = df_raw['catalyst_type'].fillna('unknown')
df_raw['catalyst'] = df_raw['catalyst'].fillna('')

# Stability data: count per ticker-date-direction BEFORE dedup
_stability_raw = (df_raw[df_raw['direction'].isin(['LONG', 'SHORT'])]
                  .groupby(['ticker', 'eval_date', 'direction', 'scanner_version'])
                  .size().reset_index(name='count'))

df_dir = df_raw[df_raw['direction'].isin(['LONG', 'SHORT'])].copy()


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


def _scan_pt_to_entry_et(scan_pt_hhmm, eval_date):
    pt_dt = PT.localize(datetime.strptime(f"{eval_date} {scan_pt_hhmm}", '%Y-%m-%d %H:%M'))
    et_dt = pt_dt.astimezone(ET)
    et_hhmm = et_dt.strftime('%H:%M')
    if et_hhmm < MARKET_OPEN_ET:
        return MARKET_OPEN_ET
    if et_hhmm >= MARKET_CLOSE_ET:
        return None
    return et_hhmm


df_dir['scan_time_hhmm'] = df_dir['scan_time_pt'].apply(_parse_scan_time)
df_dir = df_dir[df_dir['scan_time_hhmm'].notna()].copy()
df_dir['entry_time_et'] = df_dir.apply(
    lambda r: _scan_pt_to_entry_et(r['scan_time_hhmm'], r['eval_date']), axis=1)
df_dir = df_dir[df_dir['entry_time_et'].notna()].copy()

# Dedup: one signal per (ticker, date) — keep earliest
df_dir = (df_dir.sort_values('scan_time_hhmm')
          .drop_duplicates(['ticker', 'eval_date'], keep='first')
          .reset_index(drop=True))

print(f"  Directional signals after dedup: {len(df_dir)}")


# ==============================================================================
# FETCH POLYGON — MINUTE BARS + DAILY BARS (extended horizon)
# ==============================================================================
poly = PolygonClient(POLYGON_API_KEY)


def _fetch_minute_bars(ticker, eval_date):
    try:
        bars = list(poly.get_aggs(ticker=ticker, multiplier=1, timespan="minute",
                                   from_=eval_date, to=eval_date, limit=500))
        if not bars:
            return None
        bar_map = {}
        for b in bars:
            ts = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').tz_convert(ET)
            bar_map[ts.strftime('%H:%M')] = {
                'open': float(b.open), 'high': float(b.high),
                'low': float(b.low), 'close': float(b.close),
                'volume': int(b.volume)}
        return bar_map
    except Exception as e:
        return None


def _fetch_daily_bars(ticker, from_date, to_date):
    try:
        bars = list(poly.get_aggs(ticker=ticker, multiplier=1, timespan="day",
                                   from_=from_date, to=to_date, limit=60))
        result = {}
        for b in bars:
            dt = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').strftime('%Y-%m-%d')
            result[dt] = {'open': float(b.open), 'high': float(b.high),
                          'low': float(b.low), 'close': float(b.close),
                          'volume': int(b.volume)}
        return result
    except Exception:
        return {}


# Phase 1: minute bars for all scanner signals
ticker_date_pairs = df_dir[['ticker', 'eval_date']].drop_duplicates().values.tolist()
all_tickers = sorted(set(t for t, _ in ticker_date_pairs))
spy_dates = sorted(set(d for _, d in ticker_date_pairs))

print(f"\n  Phase 1: Fetching minute bars for {len(ticker_date_pairs)} signals...")
bar_data = {}
n_ok = 0
for i, (tk, dt) in enumerate(ticker_date_pairs):
    bm = _fetch_minute_bars(tk, dt)
    if bm:
        bar_data[(tk, dt)] = bm
        n_ok += 1
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(ticker_date_pairs)} ({n_ok} ok)")

spy_minute = {}
for dt in spy_dates:
    bm = _fetch_minute_bars('SPY', dt)
    if bm:
        spy_minute[dt] = bm

print(f"  Got minute bars for {n_ok}/{len(ticker_date_pairs)} signals")

# Phase 2: daily bars for all tickers (history + forward for horizon)
# Fetch from 30 days before to 10 days after
_from_daily = (datetime.strptime(ALL_DATES[0], '%Y-%m-%d') - timedelta(days=35)).strftime('%Y-%m-%d')
_to_daily = (datetime.strptime(ALL_DATES[-1], '%Y-%m-%d') + timedelta(days=HORIZON_FORWARD_DAYS + 5)).strftime('%Y-%m-%d')

print(f"\n  Phase 2: Fetching daily bars {_from_daily} to {_to_daily} for {len(all_tickers)} tickers + SPY...")
daily_data = {}
for i, tk in enumerate(all_tickers):
    db = _fetch_daily_bars(tk, _from_daily, _to_daily)
    if db:
        daily_data[tk] = db
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(all_tickers)}")

spy_daily = _fetch_daily_bars('SPY', _from_daily, _to_daily)
print(f"  Got daily bars for {len(daily_data)}/{len(all_tickers)} tickers")


# ==============================================================================
# COMPUTE FEATURES — comprehensive per-trade
# ==============================================================================
print(f"\n  Computing features...")
records = []

for idx, row in df_dir.iterrows():
    tk = row['ticker']
    dirn = row['direction']
    entry_hhmm = row['entry_time_et']
    dt = row['eval_date']
    bm = bar_data.get((tk, dt))
    if bm is None:
        continue

    # Find entry bar
    actual_entry = entry_hhmm
    entry_bar = bm.get(entry_hhmm)
    if not entry_bar:
        cands = sorted(k for k in bm if k >= entry_hhmm and k < MARKET_CLOSE_ET)
        if not cands:
            continue
        actual_entry = cands[0]
        entry_bar = bm[actual_entry]

    ep = entry_bar['open']
    if ep <= 0:
        continue
    eh, em = map(int, actual_entry.split(':'))
    sign = 1.0 if dirn == 'LONG' else -1.0

    # Daily context
    tk_daily = daily_data.get(tk, {})
    trading_days = sorted(tk_daily.keys())
    prev_days = [d for d in trading_days if d < dt]
    fwd_days = [d for d in trading_days if d > dt]

    prev_close = tk_daily[prev_days[-1]]['close'] if prev_days else None
    gap_pct = ((ep - prev_close) / prev_close * 100) if prev_close else 0.0

    # ATR14
    atr14 = None
    if len(prev_days) >= 15:
        trs = []
        for i in range(1, min(15, len(prev_days))):
            h = tk_daily[prev_days[-i]]['high']
            l = tk_daily[prev_days[-i]]['low']
            pc = tk_daily[prev_days[-i-1]]['close'] if i+1 <= len(prev_days) else l
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr14 = np.mean(trs) if trs else None

    atr_pct = (atr14 / ep * 100) if atr14 and ep > 0 else 2.5
    atr_gap = abs(gap_pct * ep / 100) / atr14 if atr14 and atr14 > 0 else 0.0

    # Volume
    avg_daily_vol = np.mean([tk_daily[d]['volume'] for d in prev_days[-20:]]) if prev_days else 0
    pm_bars = {k: v for k, v in bm.items() if k < MARKET_OPEN_ET}
    pm_vol = sum(v['volume'] for v in pm_bars.values())
    log_vol_ratio = np.log1p(pm_vol / avg_daily_vol) if avg_daily_vol > 0 else 0.0

    # Premarket shape features (EXP S7)
    pm_high, pm_low, pm_close_price, pm_high_time = 0, 1e9, ep, '09:30'
    pm_vol_last30 = 0
    for k in sorted(pm_bars.keys()):
        b = pm_bars[k]
        if b['high'] > pm_high:
            pm_high = b['high']
            pm_high_time = k
        pm_low = min(pm_low, b['low'])
        pm_close_price = b['close']
        if k >= '09:00':
            pm_vol_last30 += b['volume']
    if pm_low > 1e8:
        pm_low = ep
    pm_range = pm_high - pm_low
    pm_retrace = (pm_high - pm_close_price) / pm_range if pm_range > 0.01 else 0.0
    pm_high_h, pm_high_m = map(int, pm_high_time.split(':'))
    pm_time_since_high = (9 * 60 + 30) - (pm_high_h * 60 + pm_high_m)
    pm_vol_late_ratio = pm_vol_last30 / pm_vol if pm_vol > 0 else 0.0

    # First 5 minutes
    first5_ret = 0.0
    first5_range = 0.0
    for m in range(1, 6):
        tot = eh * 60 + em + m
        target = f"{tot // 60:02d}:{tot % 60:02d}"
        b = bm.get(target)
        if b:
            first5_ret = ((b['close'] - ep) / ep) * 100
            first5_range = max(first5_range, ((b['high'] - b['low']) / ep) * 100)
    first5_confirms = (first5_ret > 0 and dirn == 'LONG') or (first5_ret < 0 and dirn == 'SHORT')

    # T+5 rebased entry
    t5_tot = eh * 60 + em + 5
    t5_target = f"{t5_tot // 60:02d}:{t5_tot % 60:02d}"
    t5_bar = bm.get(t5_target)
    entry_price_5m = t5_bar['close'] if t5_bar else None

    # PnL curve (no SL) — for MFE/MAE
    pnl_curve = {}
    mfe, mae = 0.0, 0.0
    mfe_time, mae_time = 0, 0
    for m in range(INTRADAY_MINUTES + 1):
        tot = eh * 60 + em + m
        target = f"{tot // 60:02d}:{tot % 60:02d}"
        if target >= MARKET_CLOSE_ET:
            break
        b = bm.get(target)
        if b is None:
            continue
        ret_close = ((b['close'] - ep) / ep) * 100 * sign
        pnl_curve[m] = ret_close
        # MFE from high/low
        if dirn == 'LONG':
            fav = ((b['high'] - ep) / ep) * 100
            adv = ((b['low'] - ep) / ep) * 100
        else:
            fav = -((b['low'] - ep) / ep) * 100
            adv = -((b['high'] - ep) / ep) * 100
        if fav > mfe:
            mfe = fav
            mfe_time = m
        if adv < mae:
            mae = adv
            mae_time = m

    if not pnl_curve or max(pnl_curve.keys()) < 30:
        continue

    max_m = max(pnl_curve.keys())
    pnl_60m_nosl = pnl_curve.get(60, np.nan)
    pnl_30m_nosl = pnl_curve.get(30, np.nan)
    pnl_120m_nosl = pnl_curve.get(120, np.nan) if 120 <= max_m else np.nan
    pnl_eod_nosl = pnl_curve.get(max_m, np.nan)

    # PnL with various SL levels
    def _sim_sl(entry_p, start_h, start_m, sl_pct, bar_map_ref):
        stopped = False
        pnl_at = {}
        for m in range(INTRADAY_MINUTES + 1):
            tot = start_h * 60 + start_m + m
            target = f"{tot // 60:02d}:{tot % 60:02d}"
            if target >= MARKET_CLOSE_ET:
                break
            b = bar_map_ref.get(target)
            if b is None:
                continue
            if stopped:
                pnl_at[m] = -sl_pct
                continue
            ret = ((b['close'] - entry_p) / entry_p) * 100 * sign
            if dirn == 'LONG':
                worst = ((b['low'] - entry_p) / entry_p) * 100
            else:
                worst = -((b['high'] - entry_p) / entry_p) * 100
            if worst <= -sl_pct:
                stopped = True
                pnl_at[m] = -sl_pct
            else:
                pnl_at[m] = round(ret, 4)
        return pnl_at, stopped

    sl_results = {}
    for sl in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        pa, st = _sim_sl(ep, eh, em, sl, bm)
        sl_results[f'pnl_60m_sl{sl}'] = pa.get(60, np.nan) if 60 <= (max(pa.keys()) if pa else 0) else np.nan
        sl_results[f'pnl_30m_sl{sl}'] = pa.get(30, np.nan) if 30 <= (max(pa.keys()) if pa else 0) else np.nan
        sl_results[f'stopped_sl{sl}'] = st

    # ATR-scaled stops
    for k in [0.25, 0.5, 0.75, 1.0]:
        atr_sl = atr_pct * k
        pa, st = _sim_sl(ep, eh, em, atr_sl, bm)
        sl_results[f'pnl_60m_atr{k}'] = pa.get(60, np.nan) if 60 <= (max(pa.keys()) if pa else 0) else np.nan
        sl_results[f'pnl_30m_atr{k}'] = pa.get(30, np.nan) if 30 <= (max(pa.keys()) if pa else 0) else np.nan
        sl_results[f'stopped_atr{k}'] = st
        sl_results[f'atr_sl_pct_{k}'] = atr_sl

    # T+5 entry: repeat SL grid
    if entry_price_5m and entry_price_5m > 0:
        t5h, t5m_val = t5_tot // 60, t5_tot % 60
        for sl in [2.5, 4.0, 5.0]:
            pa5, st5 = _sim_sl(entry_price_5m, t5h, t5m_val, sl, bm)
            sl_results[f'pnl_60m_t5_sl{sl}'] = pa5.get(60, np.nan) if 60 <= (max(pa5.keys()) if pa5 else 0) else np.nan
        for k in [0.5, 0.75]:
            atr_sl = atr_pct * k
            pa5, st5 = _sim_sl(entry_price_5m, t5h, t5m_val, atr_sl, bm)
            sl_results[f'pnl_60m_t5_atr{k}'] = pa5.get(60, np.nan) if 60 <= (max(pa5.keys()) if pa5 else 0) else np.nan

    # Opening range (first 5 min) for ORB
    or_high, or_low = -1e9, 1e9
    for m in range(5):
        tot = 9 * 60 + 30 + m
        target = f"{tot // 60:02d}:{tot % 60:02d}"
        b = bm.get(target)
        if b:
            or_high = max(or_high, b['high'])
            or_low = min(or_low, b['low'])
    if or_high < -1e8:
        or_high = ep
    if or_low > 1e8:
        or_low = ep

    # ORB detection: first break of OR after 09:35
    orb_dir = None
    orb_entry_price = None
    orb_entry_time = None
    orb_sl = None
    for m in range(5, INTRADAY_MINUTES):
        tot = 9 * 60 + 30 + m
        target = f"{tot // 60:02d}:{tot % 60:02d}"
        if target >= MARKET_CLOSE_ET:
            break
        b = bm.get(target)
        if b is None:
            continue
        if b['high'] > or_high and orb_dir is None:
            orb_dir = 'LONG'
            orb_entry_price = or_high
            orb_entry_time = m
            orb_sl = or_high - or_low
            break
        if b['low'] < or_low and orb_dir is None:
            orb_dir = 'SHORT'
            orb_entry_price = or_low
            orb_entry_time = m
            orb_sl = or_high - or_low
            break

    # ORB PnL (if triggered)
    orb_pnl_60m = np.nan
    orb_stopped = False
    if orb_dir and orb_entry_price and orb_entry_price > 0:
        orb_sign = 1.0 if orb_dir == 'LONG' else -1.0
        orb_sl_pct = (orb_sl / orb_entry_price * 100) if orb_sl else 5.0
        orb_stopped_flag = False
        for m2 in range(orb_entry_time, min(orb_entry_time + 60, INTRADAY_MINUTES)):
            tot2 = 9 * 60 + 30 + m2
            target2 = f"{tot2 // 60:02d}:{tot2 % 60:02d}"
            if target2 >= MARKET_CLOSE_ET:
                break
            b2 = bm.get(target2)
            if b2 is None:
                continue
            if orb_dir == 'LONG':
                worst2 = ((b2['low'] - orb_entry_price) / orb_entry_price) * 100
            else:
                worst2 = -((b2['high'] - orb_entry_price) / orb_entry_price) * 100
            if worst2 <= -orb_sl_pct:
                orb_stopped = True
                break
            if m2 == min(orb_entry_time + 60 - 1, INTRADAY_MINUTES - 1):
                ret2 = ((b2['close'] - orb_entry_price) / orb_entry_price) * 100
                orb_pnl_60m = ret2 * orb_sign
        if orb_stopped:
            orb_pnl_60m = -orb_sl_pct

    # Multi-day horizon (EXP S2)
    horizon_pnl = {}
    spy_daily_data = spy_daily or {}
    for i_fwd, fd in enumerate(fwd_days[:5]):
        fd_bar = tk_daily.get(fd)
        if fd_bar:
            horizon_pnl[f'pnl_t{i_fwd+1}_open'] = ((fd_bar['open'] - ep) / ep * 100) * sign
            horizon_pnl[f'pnl_t{i_fwd+1}_close'] = ((fd_bar['close'] - ep) / ep * 100) * sign
            spy_fd = spy_daily_data.get(fd)
            if spy_fd:
                spy_ret = (spy_fd['close'] - spy_fd['open']) / spy_fd['open'] * 100
                horizon_pnl[f'alpha_t{i_fwd+1}_close'] = horizon_pnl[f'pnl_t{i_fwd+1}_close'] - spy_ret * sign

    # Stability features (EXP S6)
    stab = _stability_raw[((_stability_raw['ticker'] == tk) & (_stability_raw['eval_date'] == dt))]
    n_runs = int(stab['count'].sum()) if len(stab) > 0 else 1
    n_long_runs = int(stab[stab['direction'] == 'LONG']['count'].sum()) if len(stab) > 0 else 0
    n_short_runs = int(stab[stab['direction'] == 'SHORT']['count'].sum()) if len(stab) > 0 else 0
    direction_flipped = (n_long_runs > 0 and n_short_runs > 0)
    stability_score = max(n_long_runs, n_short_runs) / max(n_runs, 1)

    # Catalyst text parsing for earnings (EXP S10)
    cat_text = str(row.get('catalyst', '')).lower()
    is_earnings = ('earnings' in cat_text or 'eps' in cat_text or 'revenue' in cat_text
                   or 'quarter' in cat_text or 'q1' in cat_text or 'q2' in cat_text
                   or 'q3' in cat_text or 'q4' in cat_text or 'fiscal' in cat_text)
    cat_beat = ('beat' in cat_text or 'exceeded' in cat_text or 'surpass' in cat_text
                or 'topped' in cat_text or 'above estimates' in cat_text)
    cat_miss = ('miss' in cat_text or 'below' in cat_text or 'disappoint' in cat_text
                or 'short of' in cat_text or 'fell short' in cat_text)
    cat_guidance_up = ('raised' in cat_text or 'upward' in cat_text or 'increased outlook' in cat_text
                       or 'above consensus' in cat_text)
    cat_guidance_down = ('lowered' in cat_text or 'cut guidance' in cat_text or 'warned' in cat_text
                         or 'reduced outlook' in cat_text)

    # Earnings rule direction
    earnings_rule_dir = None
    if is_earnings:
        if cat_beat and not cat_miss:
            earnings_rule_dir = 'LONG'
        elif cat_miss and not cat_beat:
            earnings_rule_dir = 'SHORT'
        if cat_guidance_up:
            earnings_rule_dir = 'LONG'
        elif cat_guidance_down:
            earnings_rule_dir = 'SHORT'

    # Build record
    conf = float(row['confidence']) if pd.notna(row.get('confidence')) else 0
    rem = float(row['remaining_move_pct']) if pd.notna(row.get('remaining_move_pct')) else 0
    exp_move = float(row['expected_move_pct']) if pd.notna(row.get('expected_move_pct')) else 0

    rec = {
        'ticker': tk, 'eval_date': dt, 'direction': dirn, 'sign': sign,
        'selected': bool(row.get('selected', False)),
        'scanner': row.get('scanner', ''),
        'scanner_version': str(row.get('scanner_version', '')),
        'catalyst_type': str(row.get('catalyst_type', '')),
        'catalyst': cat_text[:200],
        'entry_time_et': actual_entry, 'entry_price': ep,
        'confidence': conf, 'remaining_move_pct': rem, 'expected_move_pct': exp_move,
        'gap_pct': gap_pct, 'abs_gap_pct': abs(gap_pct), 'atr_gap': atr_gap,
        'atr_pct': atr_pct, 'log_vol_ratio': log_vol_ratio,
        'pm_retrace': pm_retrace, 'pm_time_since_high': pm_time_since_high,
        'pm_vol_late_ratio': pm_vol_late_ratio, 'pm_vol': pm_vol,
        'first5_ret': first5_ret, 'first5_confirms': first5_confirms,
        'mfe': mfe, 'mae': mae, 'mfe_time': mfe_time, 'mae_time': mae_time,
        'pnl_30m_nosl': pnl_30m_nosl, 'pnl_60m_nosl': pnl_60m_nosl,
        'pnl_120m_nosl': pnl_120m_nosl, 'pnl_eod_nosl': pnl_eod_nosl,
        'or_high': or_high, 'or_low': or_low,
        'orb_dir': orb_dir, 'orb_pnl_60m': orb_pnl_60m, 'orb_stopped': orb_stopped,
        'orb_entry_time': orb_entry_time,
        'n_runs': n_runs, 'direction_flipped': direction_flipped,
        'stability_score': stability_score,
        'is_earnings': is_earnings, 'cat_beat': cat_beat, 'cat_miss': cat_miss,
        'cat_guidance_up': cat_guidance_up, 'cat_guidance_down': cat_guidance_down,
        'earnings_rule_dir': earnings_rule_dir,
        'catalyst_len': len(cat_text),
    }
    rec.update(sl_results)
    rec.update(horizon_pnl)
    records.append(rec)
    if (len(records)) % 50 == 0:
        print(f"    {len(records)} trades processed")

df = pd.DataFrame(records)
print(f"\n  Master table: {len(df)} trades (Fold A: {(df['eval_date'].isin(FOLD_A)).sum()}, "
      f"Fold B: {(df['eval_date'].isin(FOLD_B)).sum()})")

# Save features
df.to_csv(f'{RESULTS_DIR}/v2_features_full.csv', index=False)


# ==============================================================================
# EXP S1: STOP DESIGN — MAE/MFE Analysis
# ==============================================================================
if RUN_STOP_DESIGN:
    print(f"\n\n{'='*_W}")
    print(f"  EXP S1: STOP DESIGN — MAE/MFE ANALYSIS")
    print(f"{'='*_W}")

    # Winners vs losers (using no-SL 60m PnL as ground truth)
    winners = df[df['pnl_60m_nosl'] > 0]
    losers = df[df['pnl_60m_nosl'] <= 0]

    print(f"\n  MAE/MFE Summary (no SL, 60m horizon):")
    print(f"  {'':─<{_W-4}}")
    print(f"  {'':30} {'Winners':>12} {'Losers':>12} {'All':>12}")
    print(f"  {'':─<{_W-4}}")
    for stat, col in [('n', None), ('Avg MAE %', 'mae'), ('Med MAE %', 'mae'),
                       ('P80 MAE %', 'mae'), ('Avg MFE %', 'mfe'), ('Med MFE %', 'mfe')]:
        if col is None:
            print(f"  {stat:<30} {len(winners):>12} {len(losers):>12} {len(df):>12}")
        elif 'P80' in stat:
            w_v = winners[col].quantile(0.2)  # 20th percentile of MAE = deepest 80% survived
            l_v = losers[col].quantile(0.2)
            a_v = df[col].quantile(0.2)
            print(f"  {stat:<30} {w_v:>+11.2f}% {l_v:>+11.2f}% {a_v:>+11.2f}%")
        elif 'Med' in stat:
            print(f"  {stat:<30} {winners[col].median():>+11.2f}% {losers[col].median():>+11.2f}% {df[col].median():>+11.2f}%")
        else:
            print(f"  {stat:<30} {winners[col].mean():>+11.2f}% {losers[col].mean():>+11.2f}% {df[col].mean():>+11.2f}%")

    # Optimal stop = P80 of winners' MAE (in absolute terms)
    w_mae_p80 = abs(winners['mae'].quantile(0.2))
    print(f"\n  Optimal stop (80th pct of winners' MAE): {w_mae_p80:.2f}%")
    print(f"  In ATR units: {w_mae_p80 / df['atr_pct'].median():.2f} × ATR")

    # Direction breakdown
    for d in ['LONG', 'SHORT']:
        w_d = winners[winners['direction'] == d]
        l_d = losers[losers['direction'] == d]
        if len(w_d) > 0:
            p80 = abs(w_d['mae'].quantile(0.2))
            print(f"    {d}: winners MAE P80 = {p80:.2f}%, n_win={len(w_d)}, n_lose={len(l_d)}")

    # Stop grid: compare all SL levels
    print(f"\n  Stop Grid Results @60m:")
    print(f"  {'':─<{_W-4}}")
    print(f"  {'Stop Level':<25} {'n':>5} {'Avg':>8} {'Med':>8} {'Trim':>8} {'WR':>6} {'SE':>6} {'CV':>4} "
          f"{'A':>8} {'B':>8}")
    print(f"  {'':─<{_W-4}}")

    sl_cols = {}
    # Fixed SL
    for sl in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        col = f'pnl_60m_sl{sl}'
        if col in df.columns:
            sl_cols[f'Fixed {sl}%'] = col
    # ATR-scaled
    for k in [0.25, 0.5, 0.75, 1.0]:
        col = f'pnl_60m_atr{k}'
        if col in df.columns:
            sl_cols[f'ATR×{k}'] = col
    # No SL
    sl_cols['No SL'] = 'pnl_60m_nosl'
    # T+5 entry
    for sl in [2.5, 4.0, 5.0]:
        col = f'pnl_60m_t5_sl{sl}'
        if col in df.columns:
            sl_cols[f'T+5 Fixed {sl}%'] = col
    for k in [0.5, 0.75]:
        col = f'pnl_60m_t5_atr{k}'
        if col in df.columns:
            sl_cols[f'T+5 ATR×{k}'] = col

    sl_summary = []
    for label, col in sl_cols.items():
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        fr = fold_robust(df[col], df['eval_date'])
        s = fr['all']
        print(f"  {label:<25} {s['n']:>5} {s['avg']:>+7.3f}% {s['med']:>+7.3f}% {s['trim']:>+7.3f}% "
              f"{s['wr']:>5.0f}% {s['se']:>.3f} {'YES' if fr['cv'] else 'no ':>3} "
              f"{fr['A']['avg']:>+7.3f}% {fr['B']['avg']:>+7.3f}%")
        sl_summary.append({'label': label, 'col': col, **fr})

    # SHORT-only stop grid
    df_short = df[df['direction'] == 'SHORT']
    print(f"\n  SHORT-only Stop Grid @60m (n={len(df_short)}):")
    print(f"  {'':─<{_W-4}}")
    for label, col in sl_cols.items():
        if 'T+5' in label:
            continue
        vals = df_short[col].dropna()
        if len(vals) == 0:
            continue
        fr = fold_robust(df_short[col], df_short['eval_date'])
        s = fr['all']
        print(f"  {label:<25} {s['n']:>5} {s['avg']:>+7.3f}% {s['med']:>+7.3f}% {s['trim']:>+7.3f}% "
              f"{s['wr']:>5.0f}% {'YES' if fr['cv'] else 'no ':>3}")

    # MAE/MFE chart
    fig_s1, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig_s1.suptitle('EXP S1: Stop Design — MAE/MFE Analysis', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    bins_mae = np.linspace(-15, 0, 40)
    ax.hist(winners['mae'], bins=bins_mae, alpha=0.7, color='#2ecc71', label=f'Winners (n={len(winners)})')
    ax.hist(losers['mae'], bins=bins_mae, alpha=0.5, color='#e74c3c', label=f'Losers (n={len(losers)})')
    ax.axvline(x=-w_mae_p80, color='blue', lw=2, ls='--', label=f'P80 stop = {w_mae_p80:.1f}%')
    ax.axvline(x=-2.5, color='orange', lw=1.5, ls=':', label='Current SL=2.5%')
    ax.set_xlabel('Max Adverse Excursion %')
    ax.set_title('MAE: Winners vs Losers')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(df['mae'], df['mfe'], c=['#2ecc71' if p > 0 else '#e74c3c' for p in df['pnl_60m_nosl']],
               alpha=0.5, s=30)
    ax.set_xlabel('MAE %')
    ax.set_ylabel('MFE %')
    ax.set_title('MAE vs MFE (green=winner)')
    ax.axvline(x=-2.5, color='orange', lw=1, ls=':')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    sl_labels = [l for l in sl_cols.keys() if 'T+5' not in l]
    sl_avgs = [df[sl_cols[l]].dropna().mean() for l in sl_labels]
    sl_meds = [df[sl_cols[l]].dropna().median() for l in sl_labels]
    x_sl = range(len(sl_labels))
    ax.bar([x - 0.15 for x in x_sl], sl_avgs, 0.3, label='Mean', color='#3498db', alpha=0.8)
    ax.bar([x + 0.15 for x in x_sl], sl_meds, 0.3, label='Median', color='#9b59b6', alpha=0.7)
    ax.set_xticks(x_sl)
    ax.set_xticklabels(sl_labels, rotation=45, fontsize=7)
    ax.axhline(y=0, color='black', lw=0.8)
    ax.set_ylabel('PnL @60m %')
    ax.set_title('Stop Level Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    for d, c in [('LONG', '#27ae60'), ('SHORT', '#c0392b')]:
        d_df = df[df['direction'] == d]
        d_w = d_df[d_df['pnl_60m_nosl'] > 0]
        if len(d_w) > 0:
            mae_sorted = np.sort(d_w['mae'].values)
            pcts = np.arange(1, len(mae_sorted) + 1) / len(mae_sorted) * 100
            ax.plot(mae_sorted, pcts, color=c, lw=2, label=f'{d} winners (n={len(d_w)})')
    ax.axhline(y=80, color='gray', ls='--', lw=0.8, label='80th percentile')
    ax.set_xlabel('MAE %')
    ax.set_ylabel('Cumulative % of winners')
    ax.set_title('Winner MAE CDF — where to set stop')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_s1.savefig(f'{OUTPUT_DIR}/v2_s1_stop_design.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==============================================================================
# EXP S2: HORIZON EXTENSION
# ==============================================================================
if RUN_HORIZON:
    print(f"\n\n{'='*_W}")
    print(f"  EXP S2: HORIZON EXTENSION — Is this a scalp or a swing signal?")
    print(f"{'='*_W}")

    horizon_cols = ['pnl_30m_nosl', 'pnl_60m_nosl', 'pnl_120m_nosl', 'pnl_eod_nosl']
    horizon_labels = ['30m', '60m', '2h', 'EOD']

    # Add T+1 through T+5 if available
    for i in range(1, 6):
        col = f'pnl_t{i}_close'
        if col in df.columns and df[col].notna().sum() > 10:
            horizon_cols.append(col)
            horizon_labels.append(f'T+{i}')

    print(f"\n  PnL by Horizon (no SL, all signals):")
    print(f"  {'':─<{_W-4}}")
    print(f"  {'Horizon':<10} {'n':>5} {'Avg':>8} {'Med':>8} {'Trim':>8} {'WR':>6} {'SE':>6} {'CV':>4}")
    print(f"  {'':─<{_W-4}}")
    for lbl, col in zip(horizon_labels, horizon_cols):
        fr = fold_robust(df[col], df['eval_date'])
        s = fr['all']
        print(f"  {lbl:<10} {s['n']:>5} {s['avg']:>+7.3f}% {s['med']:>+7.3f}% {s['trim']:>+7.3f}% "
              f"{s['wr']:>5.0f}% {s['se']:>.3f} {'YES' if fr['cv'] else 'no ':>3}")

    # Feature correlation vs horizon
    feat_cols = ['remaining_move_pct', 'expected_move_pct', 'log_vol_ratio',
                 'gap_pct', 'confidence', 'atr_gap']
    print(f"\n  Feature Correlation vs Horizon (Pearson r):")
    print(f"  {'':─<{_W-4}}")
    hdr = f"  {'Feature':<22}" + "".join(f" {l:>8}" for l in horizon_labels)
    print(hdr)
    print(f"  {'':─<{_W-4}}")
    corr_data = {}
    for feat in feat_cols:
        if feat not in df.columns:
            continue
        corrs = []
        for col in horizon_cols:
            v = df[[feat, col]].dropna()
            if len(v) > 5:
                corrs.append(v[feat].corr(v[col]))
            else:
                corrs.append(np.nan)
        corr_data[feat] = corrs
        row_str = f"  {feat:<22}" + "".join(f" {c:>+7.3f}%" if pd.notna(c) else " " * 9 for c in corrs)
        print(row_str)

    # SHORT only
    df_s = df[df['direction'] == 'SHORT']
    print(f"\n  SHORT-only by Horizon (no SL, n={len(df_s)}):")
    for lbl, col in zip(horizon_labels, horizon_cols):
        vals = df_s[col].dropna()
        if len(vals) > 0:
            fr = fold_robust(df_s[col], df_s['eval_date'])
            s = fr['all']
            print(f"    {lbl:<10} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  "
                  f"WR={s['wr']:.0f}%  CV:{'YES' if fr['cv'] else 'no'}")

    # Chart
    fig_s2, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig_s2.suptitle('EXP S2: Horizon Extension', fontsize=13, fontweight='bold')

    ax = axes[0]
    avgs = [df[c].dropna().mean() for c in horizon_cols]
    meds = [df[c].dropna().median() for c in horizon_cols]
    x_h = range(len(horizon_labels))
    ax.plot(x_h, avgs, 'o-', color='#3498db', lw=2, label='Mean')
    ax.plot(x_h, meds, 's--', color='#9b59b6', lw=1.5, label='Median')
    ax.set_xticks(x_h)
    ax.set_xticklabels(horizon_labels)
    ax.axhline(y=0, color='black', lw=0.8)
    ax.set_ylabel('PnL %')
    ax.set_title('All signals by horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for d, c in [('LONG', '#27ae60'), ('SHORT', '#c0392b')]:
        d_df = df[df['direction'] == d]
        avgs_d = [d_df[col].dropna().mean() for col in horizon_cols]
        ax.plot(x_h, avgs_d, 'o-', color=c, lw=2, label=d)
    ax.set_xticks(x_h)
    ax.set_xticklabels(horizon_labels)
    ax.axhline(y=0, color='black', lw=0.8)
    ax.set_ylabel('Avg PnL %')
    ax.set_title('By direction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for feat in ['remaining_move_pct', 'expected_move_pct', 'log_vol_ratio']:
        if feat in corr_data:
            ax.plot(x_h[:len(corr_data[feat])], corr_data[feat], 'o-', lw=1.5, label=feat)
    ax.set_xticks(x_h)
    ax.set_xticklabels(horizon_labels)
    ax.axhline(y=0, color='black', lw=0.8)
    ax.set_ylabel('Correlation (r)')
    ax.set_title('Feature correlation vs horizon')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig_s2.savefig(f'{OUTPUT_DIR}/v2_s2_horizon.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==============================================================================
# EXP S3: COUNTERFACTUAL UNIVERSE — all gappers vs scanner-flagged
# ==============================================================================
if RUN_COUNTERFACTUAL:
    print(f"\n\n{'='*_W}")
    print(f"  EXP S3: COUNTERFACTUAL UNIVERSE")
    print(f"  Do scanner-flagged gappers outperform ALL gappers?")
    print(f"{'='*_W}")

    # Fetch grouped daily from Polygon REST API
    prev_dates = {}
    for dt in ALL_DATES:
        dt_obj = datetime.strptime(dt, '%Y-%m-%d')
        for offset in range(1, 7):
            prev = (dt_obj - timedelta(days=offset)).strftime('%Y-%m-%d')
            if prev in (spy_daily or {}):
                prev_dates[dt] = prev
                break
        if dt not in prev_dates:
            prev_dates[dt] = (dt_obj - timedelta(days=1)).strftime('%Y-%m-%d')

    all_gappers = []
    scanner_tickers = set(zip(df['ticker'], df['eval_date']))

    for dt in ALL_DATES:
        prev_dt = prev_dates.get(dt)
        print(f"\n  Fetching grouped daily for {dt} (prev={prev_dt})...")
        try:
            url_today = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{dt}"
            url_prev = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{prev_dt}"
            resp_today = _requests.get(url_today, params={"apiKey": POLYGON_API_KEY, "adjusted": "true"}, timeout=30)
            resp_prev = _requests.get(url_prev, params={"apiKey": POLYGON_API_KEY, "adjusted": "true"}, timeout=30)

            if resp_today.status_code != 200 or resp_prev.status_code != 200:
                print(f"    API error: today={resp_today.status_code}, prev={resp_prev.status_code}")
                continue

            today_data = {r['T']: r for r in resp_today.json().get('results', [])}
            prev_data = {r['T']: r for r in resp_prev.json().get('results', [])}

            n_gappers = 0
            for tk, td_r in today_data.items():
                if not re.match(r'^[A-Z]{1,5}$', tk):
                    continue
                if tk not in prev_data:
                    continue
                prev_c = prev_data[tk].get('c', 0)
                today_o = td_r.get('o', 0)
                if prev_c <= 0 or today_o <= 0:
                    continue
                gap = (today_o - prev_c) / prev_c * 100
                if abs(gap) < COUNTERFACTUAL_MIN_GAP_PCT:
                    continue
                today_v = td_r.get('v', 0)
                prev_v = prev_data[tk].get('v', 0)
                vol_ratio = today_v / prev_v if prev_v > 0 else 0

                flagged = (tk, dt) in scanner_tickers
                all_gappers.append({
                    'ticker': tk, 'eval_date': dt, 'gap_pct': gap,
                    'volume_ratio': vol_ratio, 'open': today_o, 'close': td_r.get('c', 0),
                    'prev_close': prev_c, 'flagged': flagged,
                    'day_return': (td_r.get('c', today_o) - today_o) / today_o * 100,
                })
                n_gappers += 1
            print(f"    {n_gappers} gappers (>{COUNTERFACTUAL_MIN_GAP_PCT}% gap)")
        except Exception as e:
            print(f"    Error: {e}")

    df_gap = pd.DataFrame(all_gappers) if all_gappers else pd.DataFrame()

    if len(df_gap) > 0:
        # Fetch minute bars for non-flagged gappers (sample)
        non_flagged = df_gap[~df_gap['flagged']].copy()
        non_flagged_sorted = non_flagged.reindex(non_flagged['gap_pct'].abs().sort_values(ascending=False).index)

        print(f"\n  Total gappers: {len(df_gap)} ({df_gap['flagged'].sum()} flagged, "
              f"{(~df_gap['flagged']).sum()} not flagged)")

        nf_bars = {}
        nf_to_fetch = non_flagged_sorted.head(COUNTERFACTUAL_MAX_PER_DAY * len(ALL_DATES))
        print(f"  Fetching minute bars for {len(nf_to_fetch)} non-flagged gappers...")
        for i, (_, row) in enumerate(nf_to_fetch.iterrows()):
            bm = _fetch_minute_bars(row['ticker'], row['eval_date'])
            if bm:
                nf_bars[(row['ticker'], row['eval_date'])] = bm
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(nf_to_fetch)}")

        # Compute 60m PnL for non-flagged (SHORT-side: sell at open, buy at +60m)
        nf_pnl = []
        for _, row in nf_to_fetch.iterrows():
            bm = nf_bars.get((row['ticker'], row['eval_date']))
            if bm is None:
                continue
            entry_bar = bm.get('09:30')
            if not entry_bar:
                cands = sorted(k for k in bm if k >= '09:30')
                if not cands:
                    continue
                entry_bar = bm[cands[0]]
            ep_nf = entry_bar['open']
            if ep_nf <= 0:
                continue

            # Compute 60m return (direction-agnostic and SHORT)
            bar_60 = None
            for m in range(55, 66):
                target = f"{(9*60+30+m)//60:02d}:{(9*60+30+m)%60:02d}"
                if target in bm:
                    bar_60 = bm[target]
                    break
            if bar_60:
                raw_ret = (bar_60['close'] - ep_nf) / ep_nf * 100
                short_ret = -raw_ret
                nf_pnl.append({
                    'ticker': row['ticker'], 'eval_date': row['eval_date'],
                    'gap_pct': row['gap_pct'], 'raw_60m': raw_ret,
                    'short_60m': short_ret,
                    'gap_fade_60m': -raw_ret if row['gap_pct'] > 0 else raw_ret,
                })

        df_nf_pnl = pd.DataFrame(nf_pnl) if nf_pnl else pd.DataFrame()

        # Compare flagged vs non-flagged
        print(f"\n  Counterfactual Comparison:")
        print(f"  {'':─<{_W-4}}")

        # Flagged: use df (scanner signals) — SHORT-side PnL
        flagged_gap_ups = df[(df['gap_pct'] > COUNTERFACTUAL_MIN_GAP_PCT)]
        flagged_shorts = df[(df['direction'] == 'SHORT') & (df['gap_pct'] > COUNTERFACTUAL_MIN_GAP_PCT)]

        print(f"  {'Group':<30} {'n':>5} {'Avg PnL':>9} {'Med':>9} {'WR':>6}")
        print(f"  {'':─<{_W-4}}")

        # All flagged signals
        f_s = robust_stats(df['pnl_60m_nosl'])
        print(f"  {'Flagged: all signals':<30} {f_s['n']:>5} {f_s['avg']:>+8.3f}% {f_s['med']:>+8.3f}% {f_s['wr']:>5.0f}%")

        # Flagged gap-up shorts
        if len(flagged_shorts) > 0:
            fs_s = robust_stats(flagged_shorts['pnl_60m_nosl'])
            print(f"  {'Flagged: SHORT gap>3%':<30} {fs_s['n']:>5} {fs_s['avg']:>+8.3f}% {fs_s['med']:>+8.3f}% {fs_s['wr']:>5.0f}%")

        # Non-flagged gappers
        if len(df_nf_pnl) > 0:
            nf_s = robust_stats(df_nf_pnl['gap_fade_60m'])
            print(f"  {'Non-flagged: gap-fade':<30} {nf_s['n']:>5} {nf_s['avg']:>+8.3f}% {nf_s['med']:>+8.3f}% {nf_s['wr']:>5.0f}%")
            nf_sh = robust_stats(df_nf_pnl[df_nf_pnl['gap_pct'] > 0]['short_60m'])
            print(f"  {'Non-flagged: SHORT gap-ups':<30} {nf_sh['n']:>5} {nf_sh['avg']:>+8.3f}% {nf_sh['med']:>+8.3f}% {nf_sh['wr']:>5.0f}%")

        # All gappers: EOD day return
        gap_up = df_gap[df_gap['gap_pct'] > 0]
        gap_dn = df_gap[df_gap['gap_pct'] < 0]
        print(f"\n  All gappers day return (gap-fade = SHORT gap-ups, LONG gap-downs):")
        for lbl, sub in [('Gap-up', gap_up), ('Gap-down', gap_dn), ('All', df_gap)]:
            if len(sub) > 0:
                fade_ret = sub['day_return'].apply(lambda x: -x) if lbl == 'Gap-up' else sub['day_return']
                s = robust_stats(fade_ret if lbl != 'All' else sub['day_return'])
                fl = sub['flagged'].sum()
                print(f"    {lbl:<12} n={len(sub):>4} (flagged={fl:>3})  "
                      f"avg={s['avg']:+.3f}%  med={s['med']:+.3f}%")

        # Chart
        fig_s3, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig_s3.suptitle('EXP S3: Counterfactual Universe', fontsize=13, fontweight='bold')

        ax = axes[0]
        if len(df_gap) > 0:
            fl = df_gap[df_gap['flagged']]
            nfl = df_gap[~df_gap['flagged']]
            ax.scatter(fl['gap_pct'], -fl['day_return'], c='#2ecc71', s=60, alpha=0.8,
                       label=f'Flagged (n={len(fl)})', edgecolors='black', lw=0.5)
            ax.scatter(nfl['gap_pct'], -nfl['day_return'], c='#e74c3c', s=20, alpha=0.3,
                       label=f'Not flagged (n={len(nfl)})', marker='x')
            ax.set_xlabel('Gap %')
            ax.set_ylabel('Gap-Fade Return (intraday)')
            ax.set_title('Flagged vs Non-flagged Gappers')
            ax.axhline(y=0, color='black', lw=0.8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        ax = axes[1]
        if len(df_nf_pnl) > 0 and len(flagged_shorts) > 0:
            bins_cf = np.linspace(-10, 10, 30)
            ax.hist(flagged_shorts['pnl_60m_nosl'].dropna(), bins=bins_cf, alpha=0.7,
                    color='#2ecc71', label=f'Flagged SHORT (n={len(flagged_shorts)})')
            ax.hist(df_nf_pnl['short_60m'], bins=bins_cf, alpha=0.4,
                    color='#e74c3c', label=f'Non-flagged SHORT (n={len(df_nf_pnl)})')
            ax.axvline(x=0, color='black', lw=0.8)
            ax.set_xlabel('PnL @60m %')
            ax.set_title('60m PnL Distribution: Flagged vs Not')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig_s3.savefig(f'{OUTPUT_DIR}/v2_s3_counterfactual.png', dpi=150, bbox_inches='tight')
        plt.show()


# ==============================================================================
# EXP S4: CROSS-SECTIONAL RANKING
# ==============================================================================
if RUN_CROSS_SECTION:
    print(f"\n\n{'='*_W}")
    print(f"  EXP S4: CROSS-SECTIONAL RANKING — Daily tercile L/S")
    print(f"{'='*_W}")

    rank_features = ['remaining_move_pct', 'expected_move_pct', 'log_vol_ratio', 'gap_pct', 'abs_gap_pct']
    print(f"\n  For each day, rank signals by feature, go LONG top tercile / SHORT bottom tercile.")
    print(f"  Dollar-neutral — removes beta and day effect.")

    for feat in rank_features:
        if feat not in df.columns:
            continue
        daily_rets = []
        for dt in ALL_DATES:
            day_df = df[(df['eval_date'] == dt) & df[feat].notna() & df['pnl_60m_nosl'].notna()].copy()
            if len(day_df) < 6:
                continue
            day_df['rank'] = day_df[feat].rank(pct=True)
            top = day_df[day_df['rank'] >= 0.67]['pnl_60m_nosl']
            bot = day_df[day_df['rank'] <= 0.33]['pnl_60m_nosl']
            if len(top) > 0 and len(bot) > 0:
                ls_ret = top.mean() - bot.mean()
                daily_rets.append({'date': dt, 'ls_return': ls_ret,
                                   'top_avg': top.mean(), 'bot_avg': bot.mean(),
                                   'n_top': len(top), 'n_bot': len(bot)})
        if daily_rets:
            df_ls = pd.DataFrame(daily_rets)
            avg_ls = df_ls['ls_return'].mean()
            all_pos = (df_ls['ls_return'] > 0).all()
            print(f"\n  {feat}:")
            print(f"    Avg L/S return: {avg_ls:+.3f}%  ({'all days positive' if all_pos else 'mixed'})")
            for _, r in df_ls.iterrows():
                print(f"      {r['date']}: L/S={r['ls_return']:+.3f}%  "
                      f"(top={r['top_avg']:+.3f}% n={r['n_top']}, bot={r['bot_avg']:+.3f}% n={r['n_bot']})")


# ==============================================================================
# EXP S5: MAGNITUDE WITHOUT DIRECTION
# ==============================================================================
if RUN_MAGNITUDE:
    print(f"\n\n{'='*_W}")
    print(f"  EXP S5: MAGNITUDE WITHOUT DIRECTION")
    print(f"  Does |expected_move| predict |realized move|?")
    print(f"{'='*_W}")

    df['abs_pnl_60m'] = df['pnl_60m_nosl'].abs()
    df['abs_expected'] = df['expected_move_pct'].abs()
    df['abs_remaining'] = df['remaining_move_pct'].abs()
    df['realized_range_60m'] = df['mfe'] + df['mae'].abs()
    df['total_excursion'] = df['mfe'] + df['mae'].abs()

    mag_pairs = [
        ('|expected_move|', 'abs_expected', '|realized_60m|', 'abs_pnl_60m'),
        ('|expected_move|', 'abs_expected', 'MFE+|MAE|', 'total_excursion'),
        ('|remaining_move|', 'abs_remaining', '|realized_60m|', 'abs_pnl_60m'),
        ('log_vol_ratio', 'log_vol_ratio', '|realized_60m|', 'abs_pnl_60m'),
        ('|gap_pct|', 'abs_gap_pct', '|realized_60m|', 'abs_pnl_60m'),
    ]

    print(f"\n  Magnitude correlations (n={len(df)}):")
    print(f"  {'':─<{_W-4}}")
    print(f"  {'Predictor':<22} {'Target':<18} {'r':>7} {'p-value':>10}")
    print(f"  {'':─<{_W-4}}")
    from scipy import stats as _scipy_stats
    for pred_lbl, pred_col, tgt_lbl, tgt_col in mag_pairs:
        v = df[[pred_col, tgt_col]].dropna()
        if len(v) > 5:
            r, p = _scipy_stats.pearsonr(v[pred_col], v[tgt_col])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {pred_lbl:<22} {tgt_lbl:<18} {r:>+6.3f} {p:>9.4f} {sig}")

    # Chart
    fig_s5, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig_s5.suptitle('EXP S5: Magnitude Without Direction', fontsize=13, fontweight='bold')

    ax = axes[0]
    v = df[['abs_expected', 'abs_pnl_60m']].dropna()
    ax.scatter(v['abs_expected'], v['abs_pnl_60m'], alpha=0.4, s=30, c='#3498db')
    if len(v) > 5:
        z = np.polyfit(v['abs_expected'], v['abs_pnl_60m'], 1)
        ax.plot([0, v['abs_expected'].max()], [z[1], z[1] + z[0] * v['abs_expected'].max()],
                'r--', lw=1.5, label=f'r={v["abs_expected"].corr(v["abs_pnl_60m"]):+.3f}')
    ax.set_xlabel('|Expected Move| %')
    ax.set_ylabel('|Realized Move @60m| %')
    ax.set_title('Magnitude prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    v2 = df[['abs_expected', 'total_excursion']].dropna()
    ax.scatter(v2['abs_expected'], v2['total_excursion'], alpha=0.4, s=30, c='#9b59b6')
    if len(v2) > 5:
        z2 = np.polyfit(v2['abs_expected'], v2['total_excursion'], 1)
        ax.plot([0, v2['abs_expected'].max()], [z2[1], z2[1] + z2[0] * v2['abs_expected'].max()],
                'r--', lw=1.5, label=f'r={v2["abs_expected"].corr(v2["total_excursion"]):+.3f}')
    ax.set_xlabel('|Expected Move| %')
    ax.set_ylabel('MFE + |MAE| %')
    ax.set_title('Total excursion prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig_s5.savefig(f'{OUTPUT_DIR}/v2_s5_magnitude.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==============================================================================
# EXP S6: STABILITY ACROSS RUNS
# ==============================================================================
if RUN_STABILITY:
    print(f"\n\n{'='*_W}")
    print(f"  EXP S6: STABILITY ACROSS RUNS")
    print(f"{'='*_W}")

    # Split by stability
    stable = df[df['stability_score'] >= 0.8]
    unstable = df[df['stability_score'] < 0.8]
    flipped = df[df['direction_flipped'] == True]
    not_flipped = df[df['direction_flipped'] == False]

    print(f"\n  Stability breakdown:")
    for lbl, sub in [('Stable (score>=0.8)', stable), ('Unstable (score<0.8)', unstable),
                      ('Direction flipped', flipped), ('Not flipped', not_flipped)]:
        fr = fold_robust(sub['pnl_60m_nosl'], sub['eval_date'])
        s = fr['all']
        if s['n'] > 0:
            print(f"    {lbl:<25} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  "
                  f"WR={s['wr']:.0f}%  CV:{'YES' if fr['cv'] else 'no'}")

    # Multi-run count as feature
    for n_min in [2, 3, 4]:
        sub = df[df['n_runs'] >= n_min]
        if len(sub) > 0:
            fr = fold_robust(sub['pnl_60m_nosl'], sub['eval_date'])
            s = fr['all']
            print(f"    n_runs >= {n_min:<20} n={s['n']:>4}  avg={s['avg']:>+.3f}%  "
                  f"med={s['med']:>+.3f}%  CV:{'YES' if fr['cv'] else 'no'}")


# ==============================================================================
# EXP S7: PREMARKET PATH SHAPE
# ==============================================================================
if RUN_PM_SHAPE:
    print(f"\n\n{'='*_W}")
    print(f"  EXP S7: PREMARKET PATH SHAPE")
    print(f"{'='*_W}")

    # Retrace bins: did price hold at highs or fade?
    df['retrace_bin'] = pd.cut(df['pm_retrace'], bins=[0, 0.2, 0.5, 0.8, 1.0],
                                labels=['held', 'mild', 'moderate', 'full_retrace'])

    print(f"\n  PM Retrace vs PnL @60m (no SL):")
    print(f"  {'':─<{_W-4}}")
    for cat in ['held', 'mild', 'moderate', 'full_retrace']:
        sub = df[df['retrace_bin'] == cat]
        if len(sub) < 3:
            continue
        s = robust_stats(sub['pnl_60m_nosl'])
        print(f"    {cat:<20} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  WR={s['wr']:.0f}%")

    # Late PM volume ratio
    df['pm_vol_late_bin'] = pd.cut(df['pm_vol_late_ratio'],
                                    bins=[0, 0.3, 0.5, 0.7, 1.0],
                                    labels=['early_vol', 'spread', 'late_heavy', 'very_late'])
    print(f"\n  Late PM Volume Ratio vs PnL:")
    for cat in ['early_vol', 'spread', 'late_heavy', 'very_late']:
        sub = df[df['pm_vol_late_bin'] == cat]
        if len(sub) < 3:
            continue
        s = robust_stats(sub['pnl_60m_nosl'])
        print(f"    {cat:<20} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  WR={s['wr']:.0f}%")

    # Correlation table
    pm_feats = ['pm_retrace', 'pm_time_since_high', 'pm_vol_late_ratio', 'pm_vol']
    print(f"\n  PM feature correlations with PnL:")
    for feat in pm_feats:
        v = df[[feat, 'pnl_60m_nosl']].dropna()
        if len(v) > 5:
            r = v[feat].corr(v['pnl_60m_nosl'])
            print(f"    {feat:<25} r={r:+.3f}")


# ==============================================================================
# EXP S8: OPENING RANGE BREAKOUT
# ==============================================================================
if RUN_ORB:
    print(f"\n\n{'='*_W}")
    print(f"  EXP S8: OPENING RANGE BREAKOUT")
    print(f"  Direction from price, catalyst qualifies which names.")
    print(f"{'='*_W}")

    df_orb = df[df['orb_dir'].notna()].copy()
    df_no_orb = df[df['orb_dir'].isna()].copy()

    print(f"\n  ORB triggered: {len(df_orb)}/{len(df)} trades")
    print(f"  ORB direction matches LLM: "
          f"{(df_orb['orb_dir'] == df_orb['direction']).sum()}/{len(df_orb)}")

    # ORB PnL
    if len(df_orb) > 0:
        orb_s = robust_stats(df_orb['orb_pnl_60m'])
        print(f"\n  ORB PnL @60m (OR as natural stop):")
        print(f"    All ORB:    n={orb_s['n']}  avg={orb_s['avg']:+.3f}%  "
              f"med={orb_s['med']:+.3f}%  WR={orb_s['wr']:.0f}%")

        # Compare: ORB direction vs LLM direction on same trades
        llm_on_orb = robust_stats(df_orb['pnl_60m_nosl'])
        print(f"    LLM dir:    n={llm_on_orb['n']}  avg={llm_on_orb['avg']:+.3f}%  "
              f"med={llm_on_orb['med']:+.3f}%  WR={llm_on_orb['wr']:.0f}%")

        # ORB agreement filter
        agree = df_orb[df_orb['orb_dir'] == df_orb['direction']]
        disagree = df_orb[df_orb['orb_dir'] != df_orb['direction']]
        if len(agree) > 0:
            s_ag = robust_stats(agree['pnl_60m_nosl'])
            print(f"    ORB agrees: n={s_ag['n']}  avg={s_ag['avg']:+.3f}%  WR={s_ag['wr']:.0f}%")
        if len(disagree) > 0:
            s_dis = robust_stats(disagree['pnl_60m_nosl'])
            print(f"    ORB disagrees: n={s_dis['n']}  avg={s_dis['avg']:+.3f}%  WR={s_dis['wr']:.0f}%")


# ==============================================================================
# EXP S9: TICKER REACTION HISTORY
# ==============================================================================
if RUN_TICKER_HISTORY:
    print(f"\n\n{'='*_W}")
    print(f"  EXP S9: TICKER REACTION HISTORY")
    print(f"  Per-ticker gap continuation vs fade tendency")
    print(f"{'='*_W}")

    # Compute historical gap stats for each ticker
    ticker_history = {}
    for tk in all_tickers:
        tk_d = daily_data.get(tk, {})
        dates_sorted = sorted(tk_d.keys())
        if len(dates_sorted) < 10:
            continue
        gaps = []
        for i in range(1, len(dates_sorted)):
            d = dates_sorted[i]
            prev_d = dates_sorted[i - 1]
            if d in ALL_DATES:
                continue
            prev_c = tk_d[prev_d]['close']
            today_o = tk_d[d]['open']
            today_c = tk_d[d]['close']
            if prev_c <= 0 or today_o <= 0:
                continue
            gap = (today_o - prev_c) / prev_c * 100
            if abs(gap) < 2.0:
                continue
            # Continuation = close further than open in gap direction
            if gap > 0:
                continuation = today_c > today_o
            else:
                continuation = today_c < today_o
            gaps.append({'gap': gap, 'continuation': continuation,
                         'day_ret': (today_c - today_o) / today_o * 100})

        if len(gaps) >= 3:
            cont_rate = sum(1 for g in gaps if g['continuation']) / len(gaps)
            avg_fade = np.mean([-g['day_ret'] * np.sign(g['gap']) for g in gaps])
            ticker_history[tk] = {
                'n_gaps': len(gaps), 'continuation_rate': cont_rate,
                'avg_fade_return': avg_fade}

    print(f"\n  Computed gap history for {len(ticker_history)}/{len(all_tickers)} tickers")

    # Add to df and test
    df['hist_cont_rate'] = df['ticker'].map(lambda t: ticker_history.get(t, {}).get('continuation_rate', np.nan))
    df['hist_n_gaps'] = df['ticker'].map(lambda t: ticker_history.get(t, {}).get('n_gaps', 0))

    has_hist = df[df['hist_cont_rate'].notna()]
    if len(has_hist) > 10:
        faders = has_hist[has_hist['hist_cont_rate'] < 0.4]
        runners = has_hist[has_hist['hist_cont_rate'] > 0.6]
        print(f"\n  Tickers that historically FADE gaps (cont_rate < 40%):")
        if len(faders) > 0:
            s = robust_stats(faders['pnl_60m_nosl'])
            print(f"    n={s['n']}  avg={s['avg']:+.3f}%  med={s['med']:+.3f}%  WR={s['wr']:.0f}%")
        print(f"  Tickers that historically RUN (cont_rate > 60%):")
        if len(runners) > 0:
            s = robust_stats(runners['pnl_60m_nosl'])
            print(f"    n={s['n']}  avg={s['avg']:+.3f}%  med={s['med']:+.3f}%  WR={s['wr']:.0f}%")

        r_hist = has_hist['hist_cont_rate'].corr(has_hist['pnl_60m_nosl'])
        print(f"\n  Correlation: hist_cont_rate vs PnL @60m: r={r_hist:+.3f}")


# ==============================================================================
# EXP S10: EARNINGS MODULE
# ==============================================================================
if RUN_EARNINGS:
    print(f"\n\n{'='*_W}")
    print(f"  EXP S10: EARNINGS MODULE")
    print(f"{'='*_W}")

    df_earn = df[df['is_earnings']].copy()
    df_non_earn = df[~df['is_earnings']].copy()

    print(f"\n  Earnings trades: {len(df_earn)}, Non-earnings: {len(df_non_earn)}")

    if len(df_earn) > 0:
        # Earnings vs non-earnings
        for lbl, sub in [('Earnings', df_earn), ('Non-earnings', df_non_earn)]:
            s = robust_stats(sub['pnl_60m_nosl'])
            print(f"    {lbl:<20} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  WR={s['wr']:.0f}%")

        # Beat vs miss
        beats = df_earn[df_earn['cat_beat'] & ~df_earn['cat_miss']]
        misses = df_earn[df_earn['cat_miss'] & ~df_earn['cat_beat']]
        ambig = df_earn[~df_earn['cat_beat'] & ~df_earn['cat_miss']]

        for lbl, sub in [('Beat', beats), ('Miss', misses), ('Ambiguous', ambig)]:
            if len(sub) > 0:
                s = robust_stats(sub['pnl_60m_nosl'])
                print(f"    {lbl:<20} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  WR={s['wr']:.0f}%")

        # Rule direction vs LLM direction
        has_rule = df_earn[df_earn['earnings_rule_dir'].notna()]
        if len(has_rule) > 0:
            agrees = has_rule[has_rule['earnings_rule_dir'] == has_rule['direction']]
            disagrees = has_rule[has_rule['earnings_rule_dir'] != has_rule['direction']]
            print(f"\n  Earnings rule direction vs LLM:")
            print(f"    Rule agrees with LLM:    {len(agrees)}/{len(has_rule)}")
            for lbl, sub in [('Agrees', agrees), ('Disagrees', disagrees)]:
                if len(sub) > 0:
                    s = robust_stats(sub['pnl_60m_nosl'])
                    print(f"      {lbl}: n={s['n']}  avg={s['avg']:+.3f}%  WR={s['wr']:.0f}%")

        # Horizon: do earnings trades work better at longer horizons?
        h_cols = ['pnl_60m_nosl', 'pnl_eod_nosl'] + [f'pnl_t{i}_close' for i in range(1, 4)]
        h_labels = ['60m', 'EOD', 'T+1', 'T+2', 'T+3']
        print(f"\n  Earnings by horizon:")
        for lbl, col in zip(h_labels, h_cols):
            if col in df_earn.columns:
                vals = df_earn[col].dropna()
                if len(vals) > 0:
                    s = robust_stats(vals)
                    print(f"    {lbl:<8} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%")


# ==============================================================================
# VERSION BREAKDOWN
# ==============================================================================
if RUN_VERSION_SPLIT:
    print(f"\n\n{'='*_W}")
    print(f"  VERSION BREAKDOWN — Is fold-CV actually version-CV?")
    print(f"{'='*_W}")

    versions = df['scanner_version'].unique()
    print(f"\n  Versions in data: {sorted(versions)}")
    print(f"\n  PnL by scanner version (no SL, 60m):")
    print(f"  {'':─<{_W-4}}")
    for ver in sorted(versions):
        sub = df[df['scanner_version'] == ver]
        if len(sub) < 3:
            continue
        s = robust_stats(sub['pnl_60m_nosl'])
        dates = sorted(sub['eval_date'].unique())
        date_str = ','.join(d[-5:] for d in dates)
        print(f"    {ver:<12} n={s['n']:>4}  avg={s['avg']:>+.3f}%  med={s['med']:>+.3f}%  "
              f"WR={s['wr']:.0f}%  dates={date_str}")

    # Direction x version
    print(f"\n  Direction x Version:")
    for ver in sorted(versions):
        for d in ['LONG', 'SHORT']:
            sub = df[(df['scanner_version'] == ver) & (df['direction'] == d)]
            if len(sub) < 3:
                continue
            s = robust_stats(sub['pnl_60m_nosl'])
            print(f"    {ver:<12} {d:<6} n={s['n']:>4}  avg={s['avg']:>+.3f}%  "
                  f"med={s['med']:>+.3f}%  WR={s['wr']:.0f}%")


# ==============================================================================
# ROBUSTNESS RE-EVALUATION OF V1 FINDINGS
# ==============================================================================
print(f"\n\n{'='*_W}")
print(f"  ROBUSTNESS RE-EVALUATION — V1 Findings with median/trimmed/LOO")
print(f"{'='*_W}")

# SHORT + good catalyst, no SL
good_cats = ['offering', 'regulatory', 'squeeze', 'secondary']
df['good_cat'] = df['catalyst_type'].str.lower().apply(
    lambda x: any(c in str(x) for c in good_cats))
short_gc = df[(df['direction'] == 'SHORT') & df['good_cat']]

print(f"\n  SHORT + good catalyst, no SL @60m:")
print_robust_row('SHORT+goodcat noSL', robust_stats(short_gc['pnl_60m_nosl']),
                 robust_stats(short_gc[short_gc['eval_date'].isin(FOLD_A)]['pnl_60m_nosl']),
                 robust_stats(short_gc[short_gc['eval_date'].isin(FOLD_B)]['pnl_60m_nosl']),
                 fold_robust(short_gc['pnl_60m_nosl'], short_gc['eval_date'])['cv'])

# With top trade removed
if len(short_gc) > 1:
    vals = short_gc['pnl_60m_nosl'].dropna().sort_values()
    top_trade = vals.iloc[-1]
    without_top = vals.iloc[:-1]
    s_wt = robust_stats(without_top)
    print(f"    WITHOUT top trade ({top_trade:+.1f}%): avg={s_wt['avg']:+.3f}%  "
          f"med={s_wt['med']:+.3f}%  n={s_wt['n']}")
    n_a = short_gc[short_gc['eval_date'].isin(FOLD_A)].shape[0]
    n_b = short_gc[short_gc['eval_date'].isin(FOLD_B)].shape[0]
    print(f"    n per fold: A={n_a}, B={n_b}")

# SHORT only, no SL
df_short_all = df[df['direction'] == 'SHORT']
print(f"\n  SHORT only, no SL @60m:")
print_robust_row('SHORT noSL', robust_stats(df_short_all['pnl_60m_nosl']),
                 robust_stats(df_short_all[df_short_all['eval_date'].isin(FOLD_A)]['pnl_60m_nosl']),
                 robust_stats(df_short_all[df_short_all['eval_date'].isin(FOLD_B)]['pnl_60m_nosl']),
                 fold_robust(df_short_all['pnl_60m_nosl'], df_short_all['eval_date'])['cv'])

# All signals baseline
print(f"\n  All signals baseline @60m (SL=2.5%):")
print_robust_row('All SL=2.5%', robust_stats(df['pnl_60m_sl2.5']),
                 robust_stats(df[df['eval_date'].isin(FOLD_A)]['pnl_60m_sl2.5']),
                 robust_stats(df[df['eval_date'].isin(FOLD_B)]['pnl_60m_sl2.5']),
                 fold_robust(df['pnl_60m_sl2.5'], df['eval_date'])['cv'])

print(f"\n  All signals baseline @60m (no SL):")
print_robust_row('All noSL', robust_stats(df['pnl_60m_nosl']),
                 robust_stats(df[df['eval_date'].isin(FOLD_A)]['pnl_60m_nosl']),
                 robust_stats(df[df['eval_date'].isin(FOLD_B)]['pnl_60m_nosl']),
                 fold_robust(df['pnl_60m_nosl'], df['eval_date'])['cv'])


# ==============================================================================
# SUMMARY DASHBOARD
# ==============================================================================
print(f"\n\n{'='*_W}")
print(f"  SUMMARY DASHBOARD")
print(f"{'='*_W}")

# Collect all experiment findings
findings = []

# S1: best stop level
if RUN_STOP_DESIGN:
    best_sl = max(sl_summary, key=lambda x: x['all']['avg']) if sl_summary else None
    if best_sl:
        findings.append(f"S1: Best stop = {best_sl['label']} (avg={best_sl['all']['avg']:+.3f}%, "
                        f"CV:{'YES' if best_sl['cv'] else 'no'})")

# S2: horizon where signal peaks
if RUN_HORIZON:
    h_avgs = [(lbl, df[col].dropna().mean()) for lbl, col in zip(horizon_labels, horizon_cols)
              if col in df.columns and df[col].notna().sum() > 10]
    if h_avgs:
        best_h = max(h_avgs, key=lambda x: x[1])
        findings.append(f"S2: Best horizon = {best_h[0]} (avg={best_h[1]:+.3f}%)")

if RUN_MAGNITUDE:
    v = df[['abs_expected', 'abs_pnl_60m']].dropna()
    if len(v) > 5:
        r_mag = v['abs_expected'].corr(v['abs_pnl_60m'])
        findings.append(f"S5: |expected| → |realized| r={r_mag:+.3f}")

if RUN_ORB and len(df_orb) > 0:
    orb_avg = df_orb['orb_pnl_60m'].dropna().mean()
    findings.append(f"S8: ORB avg PnL = {orb_avg:+.3f}% (n={len(df_orb)})")

print(f"\n  Key findings:")
for f in findings:
    print(f"    {f}")

# Save results
results_out = {
    'n_trades': len(df),
    'n_fold_a': int((df['eval_date'].isin(FOLD_A)).sum()),
    'n_fold_b': int((df['eval_date'].isin(FOLD_B)).sum()),
    'baseline_sl25_avg': float(df['pnl_60m_sl2.5'].dropna().mean()),
    'baseline_nosl_avg': float(df['pnl_60m_nosl'].dropna().mean()),
    'short_nosl_avg': float(df_short_all['pnl_60m_nosl'].dropna().mean()),
}

class _NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)

with open(f'{RESULTS_DIR}/v2_results.json', 'w') as f:
    json.dump(results_out, f, indent=2, cls=_NpEnc)

print(f"\n  Results saved to {RESULTS_DIR}/v2_results.json")
print(f"  Features saved to {RESULTS_DIR}/v2_features_full.csv")
print(f"\n{'='*_W}")
print(f"  DONE — Scanner Experiments V2")
print(f"{'='*_W}")
