# %% ── Creative Scanner Experiments ────────────────────────────────────────────
#
# Goes BEYOND filter combinations. Tests fundamentally different approaches:
#
# EXP 1: ML classifier (XGBoost) — predict profitable trades from all features
# EXP 2: Price-only model — ignore LLM, use gap + volume + ATR
# EXP 3: Anti-selection — systematically invert scanner selections
# EXP 4: Gap-reversion system — fade large gaps, ride small gaps
# EXP 5: Catalyst keyword NLP — mine raw text for predictive tokens
# EXP 6: First-5-min confirmation — use early price action to filter
# EXP 7: Ensemble of micro-signals — combine weak signals
# EXP 8: Adaptive SL/TP — scale risk to gap size and volatility
#
# All experiments use Sep 8-11 data, cross-validated FoldA vs FoldB.
# Target: >2% avg PnL per trade.
#
# Colab-ready, config vars at top.

import os, sys, json, warnings, re
from datetime import datetime, timedelta
from collections import defaultdict, Counter
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
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "scikit-learn", "--quiet", "--break-system-packages"])
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report

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

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
BQ_PREMARKET_TABLE = "ys-time-series.scoring_logs.premarket_scanner"
BQ_DISCOVERY_TABLE = "ys-time-series.scoring_logs.discovery_scanner"

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


# ==============================================================================
# DATA FETCH — BQ predictions (ALL signals, including NEUTRAL)
# ==============================================================================
print(f"\n{'='*100}")
print(f"  CREATIVE SCANNER EXPERIMENTS")
print(f"  Dates: {', '.join(ALL_DATES)}")
print(f"  Cross-validation: FoldA={FOLD_A} vs FoldB={FOLD_B}")
print(f"{'='*100}")

bq_client = bigquery.Client()
_dates_sql = ", ".join(f"'{d}'" for d in ALL_DATES)


def fetch_all_signals(table, scanner_name):
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
        ORDER BY scan_date, scan_time_pt, ticker
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
        ORDER BY scan_date, scan_time_pt, ticker
        """
    df = bq_client.query(q).to_dataframe()
    df['scanner'] = scanner_name
    return df


print(f"\n  Fetching ALL signals from BQ (including NEUTRAL)...")
df_pre = fetch_all_signals(BQ_PREMARKET_TABLE, 'premarket')
df_disc = fetch_all_signals(BQ_DISCOVERY_TABLE, 'discovery')
print(f"    Premarket: {len(df_pre)} total signals")
print(f"    Discovery: {len(df_disc)} total signals")

df_raw = pd.concat([df_pre, df_disc], ignore_index=True)
df_raw['gate_reason'] = df_raw['gate_reason'].fillna('')
df_raw.loc[df_raw['gate_reason'].isin(['None', 'none', 'nan']), 'gate_reason'] = ''
df_raw['eval_date'] = df_raw['eval_date'].astype(str)
df_raw['catalyst_type'] = df_raw['catalyst_type'].fillna('unknown')
df_raw['catalyst'] = df_raw['catalyst'].fillna('')

# Only directional signals for trading
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


df_dir['scan_time_hhmm'] = df_dir['scan_time_pt'].apply(_parse_scan_time)
df_dir = df_dir[df_dir['scan_time_hhmm'].notna()].copy()


def _scan_pt_to_entry_et(scan_pt_hhmm, eval_date):
    pt_dt = PT.localize(datetime.strptime(f"{eval_date} {scan_pt_hhmm}", '%Y-%m-%d %H:%M'))
    et_dt = pt_dt.astimezone(ET)
    et_hhmm = et_dt.strftime('%H:%M')
    if et_hhmm < MARKET_OPEN_ET:
        return MARKET_OPEN_ET
    if et_hhmm >= MARKET_CLOSE_ET:
        return None
    return et_hhmm


df_dir['entry_time_et'] = df_dir.apply(
    lambda r: _scan_pt_to_entry_et(r['scan_time_hhmm'], r['eval_date']), axis=1)
df_dir = df_dir[df_dir['entry_time_et'].notna()].copy()

# Dedup: one signal per (ticker, date) — keep earliest
df_dir = (df_dir.sort_values('scan_time_hhmm')
          .drop_duplicates(['ticker', 'eval_date'], keep='first')
          .reset_index(drop=True))

print(f"\n  Directional signals after dedup: {len(df_dir)}")


# ==============================================================================
# FETCH POLYGON PRICE DATA — with extended context (5-min, premarket volume)
# ==============================================================================
poly = PolygonClient(POLYGON_API_KEY)


def fetch_minute_bars_extended(ticker, eval_date):
    try:
        bars = list(poly.get_aggs(
            ticker=ticker, multiplier=1, timespan="minute",
            from_=eval_date, to=eval_date, limit=500,
        ))
        if not bars:
            return None

        bar_map = {}
        premarket_bars = []
        for b in bars:
            ts = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').tz_convert(ET)
            key = ts.strftime('%H:%M')
            bar_data = {
                'open': float(b.open), 'high': float(b.high),
                'low': float(b.low), 'close': float(b.close),
                'volume': int(b.volume),
            }
            bar_map[key] = bar_data
            if key < MARKET_OPEN_ET:
                premarket_bars.append(bar_data)

        prev_close = None
        atr14 = None
        avg_daily_vol = None
        try:
            from_date = (datetime.strptime(eval_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
            daily = list(poly.get_aggs(
                ticker=ticker, multiplier=1, timespan="day",
                from_=from_date, to=eval_date, limit=30,
            ))
            ref_dt = datetime.strptime(eval_date, '%Y-%m-%d')
            prev_days = [b for b in daily
                         if pd.Timestamp(b.timestamp, unit='ms', tz='UTC').date() < ref_dt.date()]
            if prev_days:
                prev_close = float(prev_days[-1].close)
                if len(prev_days) >= 14:
                    trs = []
                    for i in range(1, min(15, len(prev_days))):
                        h = float(prev_days[i].high)
                        l = float(prev_days[i].low)
                        pc = float(prev_days[i-1].close)
                        tr = max(h - l, abs(h - pc), abs(l - pc))
                        trs.append(tr)
                    atr14 = np.mean(trs) if trs else None
                vols = [float(b.volume) for b in prev_days[-20:]]
                avg_daily_vol = np.mean(vols) if vols else None
        except Exception:
            pass

        premarket_volume = sum(b['volume'] for b in premarket_bars)

        return {
            'bars': bar_map,
            'prev_close': prev_close,
            'atr14': atr14,
            'avg_daily_vol': avg_daily_vol,
            'premarket_volume': premarket_volume,
            'premarket_bars': premarket_bars,
        }
    except Exception as e:
        print(f"      [ERROR] {ticker}: {e}")
        return None


ticker_date_pairs = df_dir[['ticker', 'eval_date']].drop_duplicates().values.tolist()
spy_dates = list(set(r[1] for r in ticker_date_pairs))

print(f"\n  Fetching Polygon extended data for {len(ticker_date_pairs)} signals + SPY...")

bar_data = {}
n_ok = 0
for i, (ticker, eval_date) in enumerate(ticker_date_pairs):
    result = fetch_minute_bars_extended(ticker, eval_date)
    if result and result['bars']:
        bar_data[(ticker, eval_date)] = result
        n_ok += 1
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(ticker_date_pairs)} fetched ({n_ok} ok)")

spy_data = {}
for dt in spy_dates:
    result = fetch_minute_bars_extended('SPY', dt)
    if result:
        spy_data[dt] = result

print(f"\n  Got extended data for {n_ok}/{len(ticker_date_pairs)} signals")


# ==============================================================================
# COMPUTE COMPREHENSIVE FEATURES FOR EACH SIGNAL
# ==============================================================================
def compute_trade_features(row, sl_pct=2.5):
    ticker = row['ticker']
    direction = row['direction']
    entry_hhmm = row['entry_time_et']
    eval_date = row['eval_date']

    td = bar_data.get((ticker, eval_date))
    if td is None:
        return None

    bar_map = td['bars']
    prev_close = td['prev_close']
    atr14 = td['atr14']
    avg_daily_vol = td['avg_daily_vol']
    premarket_vol = td['premarket_volume']

    # Find actual entry bar
    actual_entry_hhmm = entry_hhmm
    entry_bar = bar_map.get(entry_hhmm)
    if not entry_bar:
        candidates = sorted(k for k in bar_map if k >= entry_hhmm)
        if not candidates:
            return None
        actual_entry_hhmm = candidates[0]
        entry_bar = bar_map[actual_entry_hhmm]

    entry_price = entry_bar['open']
    if entry_price <= 0:
        return None

    entry_h, entry_m = map(int, actual_entry_hhmm.split(':'))
    sign = 1.0 if direction == 'LONG' else -1.0

    # Gap
    gap_pct = ((entry_price - prev_close) / prev_close * 100) if prev_close else 0.0

    # ATR-normalized gap
    atr_gap = abs(gap_pct * entry_price / 100) / atr14 if atr14 and atr14 > 0 else 0.0

    # Volume ratio
    vol_ratio = premarket_vol / avg_daily_vol if avg_daily_vol and avg_daily_vol > 0 else 0.0

    # First 5 minutes after entry — momentum/reversal signal
    first_5min_ret = 0.0
    first_5min_range = 0.0
    first_5min_vol = 0
    for m in range(1, 6):
        total_min = entry_h * 60 + entry_m + m
        th, tm = total_min // 60, total_min % 60
        target = f"{th:02d}:{tm:02d}"
        bar = bar_map.get(target)
        if bar:
            first_5min_ret = ((bar['close'] - entry_price) / entry_price) * 100
            first_5min_range = max(first_5min_range,
                                   ((bar['high'] - bar['low']) / entry_price) * 100)
            first_5min_vol += bar['volume']

    # First 5min directional confirmation
    first_5min_confirms = (first_5min_ret > 0 and direction == 'LONG') or \
                          (first_5min_ret < 0 and direction == 'SHORT')

    # Rebased entry at t+5: for EXP 6 leakage fix.
    # Entry price = close of bar at entry+5 minutes.
    entry_price_5m = None
    entry_hhmm_5m = None
    t5_total = entry_h * 60 + entry_m + 5
    t5_h, t5_m = t5_total // 60, t5_total % 60
    t5_target = f"{t5_h:02d}:{t5_m:02d}"
    if t5_target < MARKET_CLOSE_ET:
        t5_bar = bar_map.get(t5_target)
        if t5_bar:
            entry_price_5m = t5_bar['close']
            entry_hhmm_5m = t5_target
        else:
            # Try next available bar after t+5
            t5_cands = sorted(k for k in bar_map if k > t5_target and k < MARKET_CLOSE_ET)
            if t5_cands:
                entry_price_5m = bar_map[t5_cands[0]]['open']
                entry_hhmm_5m = t5_cands[0]

    # Compute PnL at multiple horizons and SL levels
    pnl_results = {}
    for sl in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 999.0]:
        stopped = False
        stop_min = None
        pnl_at = {}
        for m_offset in range(INTRADAY_MINUTES + 1):
            total_min = entry_h * 60 + entry_m + m_offset
            th, tm = total_min // 60, total_min % 60
            target = f"{th:02d}:{tm:02d}"
            if target >= MARKET_CLOSE_ET:
                break
            bar = bar_map.get(target)
            if bar is None:
                continue
            if stopped:
                pnl_at[m_offset] = -sl
                continue
            ret = ((bar['close'] - entry_price) / entry_price) * 100
            raw_pnl = ret * sign
            if direction == 'LONG':
                worst = ((bar['low'] - entry_price) / entry_price) * 100
            else:
                worst = -((bar['high'] - entry_price) / entry_price) * 100
            if worst <= -sl:
                stopped = True
                stop_min = m_offset
                pnl_at[m_offset] = -sl
            else:
                pnl_at[m_offset] = round(raw_pnl, 4)

        sl_key = f"sl{sl:.1f}".replace('.', '_')
        for h in [30, 60]:
            max_m = max(pnl_at.keys()) if pnl_at else 0
            pnl_results[f'pnl_{h}m_{sl_key}'] = pnl_at.get(h, np.nan) if h <= max_m else np.nan
        eod_m = max(pnl_at.keys()) if pnl_at else 0
        pnl_results[f'pnl_eod_{sl_key}'] = pnl_at.get(eod_m, np.nan)
        pnl_results[f'stopped_{sl_key}'] = stopped
        pnl_results[f'stop_min_{sl_key}'] = stop_min

        # MFE/MAE (max favorable/adverse excursion)
        if pnl_at and sl > 100:  # no-SL version for true MFE/MAE
            vals = list(pnl_at.values())
            pnl_results['mfe'] = max(vals) if vals else 0
            pnl_results['mae'] = min(vals) if vals else 0

    if not pnl_results:
        return None

    # Check we have at least 30min of data
    if pd.isna(pnl_results.get('pnl_30m_sl2_5', np.nan)):
        return None

    # Rebased PnL from t+5 entry (for leakage-free EXP 6)
    # Measures PnL ONLY from the t+5 close price forward.
    pnl_rebased = {}
    if entry_price_5m and entry_price_5m > 0 and entry_hhmm_5m:
        t5_h_r, t5_m_r = map(int, entry_hhmm_5m.split(':'))
        for sl in [2.5, 5.0, 999.0]:
            stopped = False
            pnl_at = {}
            for m_offset in range(INTRADAY_MINUTES + 1):
                total_min = t5_h_r * 60 + t5_m_r + m_offset
                th, tm = total_min // 60, total_min % 60
                target = f"{th:02d}:{tm:02d}"
                if target >= MARKET_CLOSE_ET:
                    break
                bar = bar_map.get(target)
                if bar is None:
                    continue
                if stopped:
                    pnl_at[m_offset] = -sl
                    continue
                ret = ((bar['close'] - entry_price_5m) / entry_price_5m) * 100
                raw_pnl = ret * sign
                if direction == 'LONG':
                    worst = ((bar['low'] - entry_price_5m) / entry_price_5m) * 100
                else:
                    worst = -((bar['high'] - entry_price_5m) / entry_price_5m) * 100
                if worst <= -sl:
                    stopped = True
                    pnl_at[m_offset] = -sl
                else:
                    pnl_at[m_offset] = round(raw_pnl, 4)
            sl_key = f"sl{sl:.1f}".replace('.', '_')
            for h in [30, 60]:
                max_m = max(pnl_at.keys()) if pnl_at else 0
                pnl_rebased[f'pnl_5m_{h}m_{sl_key}'] = pnl_at.get(h, np.nan) if h <= max_m else np.nan
            eod_m = max(pnl_at.keys()) if pnl_at else 0
            pnl_rebased[f'pnl_5m_eod_{sl_key}'] = pnl_at.get(eod_m, np.nan)
    else:
        for sl in [2.5, 5.0, 999.0]:
            sl_key = f"sl{sl:.1f}".replace('.', '_')
            for h in [30, 60]:
                pnl_rebased[f'pnl_5m_{h}m_{sl_key}'] = np.nan
            pnl_rebased[f'pnl_5m_eod_{sl_key}'] = np.nan

    pnl_results.update(pnl_rebased)
    pnl_results['entry_price_5m'] = entry_price_5m

    # SPY return for alpha
    spy_td = spy_data.get(eval_date)
    spy_30m = np.nan
    spy_60m = np.nan
    if spy_td:
        spy_bars_map = spy_td['bars']
        spy_entry = spy_bars_map.get(actual_entry_hhmm)
        if not spy_entry:
            cands = sorted(k for k in spy_bars_map if k >= actual_entry_hhmm)
            if cands:
                spy_entry = spy_bars_map[cands[0]]
        if spy_entry:
            spy_ep = spy_entry['open']
            for h, h_key in [(30, 'spy_30m'), (60, 'spy_60m')]:
                total_m = entry_h * 60 + entry_m + h
                th, tm = total_m // 60, total_m % 60
                target = f"{th:02d}:{tm:02d}"
                if target >= MARKET_CLOSE_ET:
                    target = sorted(spy_bars_map.keys())[-1]
                sb = spy_bars_map.get(target)
                if sb:
                    if h == 30:
                        spy_30m = round(((sb['close'] - spy_ep) / spy_ep) * 100, 4)
                    else:
                        spy_60m = round(((sb['close'] - spy_ep) / spy_ep) * 100, 4)

    # Catalyst text features
    catalyst_text = str(row.get('catalyst', '')).lower()
    catalyst_len = len(catalyst_text)

    # Keyword flags
    kw_offering = int('offering' in catalyst_text or 'secondary' in catalyst_text or 'dilut' in catalyst_text)
    kw_fda = int('fda' in catalyst_text or 'approval' in catalyst_text or 'clinical' in catalyst_text)
    kw_earnings = int('earnings' in catalyst_text or 'revenue' in catalyst_text or 'eps' in catalyst_text
                       or 'quarter' in catalyst_text)
    kw_analyst = int('upgrade' in catalyst_text or 'downgrade' in catalyst_text or 'initiat' in catalyst_text
                      or 'price target' in catalyst_text or 'analyst' in catalyst_text)
    kw_deal = int('acqui' in catalyst_text or 'merger' in catalyst_text or 'deal' in catalyst_text
                   or 'buyout' in catalyst_text)
    kw_squeeze = int('squeeze' in catalyst_text or 'short interest' in catalyst_text)
    kw_beat = int('beat' in catalyst_text or 'surpass' in catalyst_text or 'exceeded' in catalyst_text)
    kw_miss = int('miss' in catalyst_text or 'below' in catalyst_text or 'disappoint' in catalyst_text)
    kw_guidance_up = int('raised' in catalyst_text or 'raised guidance' in catalyst_text
                         or 'upward' in catalyst_text or 'increased outlook' in catalyst_text)
    kw_guidance_down = int('lowered' in catalyst_text or 'cut guidance' in catalyst_text
                           or 'reduced outlook' in catalyst_text or 'warned' in catalyst_text)
    kw_contract = int('contract' in catalyst_text or 'awarded' in catalyst_text or 'partnership' in catalyst_text)
    kw_bankruptcy = int('bankrupt' in catalyst_text or 'chapter 11' in catalyst_text or 'default' in catalyst_text)

    # Direction-remaining contradiction
    rem = float(row['remaining_move_pct']) if pd.notna(row.get('remaining_move_pct')) else 0.0
    dir_contra = (direction == 'LONG' and rem < -0.5) or (direction == 'SHORT' and rem > 0.5)

    # Gap aligns with direction?
    gap_aligns = (direction == 'LONG' and gap_pct > 0) or (direction == 'SHORT' and gap_pct < 0)
    gap_opposes = (direction == 'LONG' and gap_pct < -1) or (direction == 'SHORT' and gap_pct > 1)

    # Entry time bucket
    entry_minutes_from_open = entry_h * 60 + entry_m - 570  # 570 = 9:30
    if entry_minutes_from_open <= 5:
        entry_bucket = 'open_5min'
    elif entry_minutes_from_open <= 30:
        entry_bucket = 'first_30min'
    elif entry_minutes_from_open <= 90:
        entry_bucket = 'mid_morning'
    else:
        entry_bucket = 'afternoon'

    rec = {
        'ticker': ticker,
        'eval_date': eval_date,
        'scanner': row['scanner'],
        'direction': direction,
        'direction_int': 1 if direction == 'LONG' else -1,
        'confidence': int(row['confidence']) if pd.notna(row['confidence']) else 0,
        'catalyst_type': str(row.get('catalyst_type', 'unknown')),
        'catalyst_text': catalyst_text,
        'selected': bool(row['selected']) if pd.notna(row.get('selected')) else False,
        'gate_reason': str(row.get('gate_reason', '')),
        'move_is_exhausted': bool(row.get('move_is_exhausted')) if pd.notna(row.get('move_is_exhausted')) else False,
        'remaining_move_pct': rem,
        'expected_move_pct': float(row['expected_move_pct']) if pd.notna(row.get('expected_move_pct')) else 0.0,
        'entry_time_et': actual_entry_hhmm,
        'entry_price': entry_price,

        # Price/volume features
        'gap_pct': round(gap_pct, 4),
        'abs_gap_pct': round(abs(gap_pct), 4),
        'atr_gap': round(atr_gap, 4),
        'vol_ratio': round(vol_ratio, 6),
        'log_vol_ratio': round(np.log1p(vol_ratio), 4) if vol_ratio > 0 else 0.0,
        'premarket_volume': premarket_vol,

        # First 5-min features
        'first_5min_ret': round(first_5min_ret, 4),
        'first_5min_range': round(first_5min_range, 4),
        'first_5min_vol': first_5min_vol,
        'first_5min_confirms': first_5min_confirms,

        # Derived features
        'dir_remaining_contradiction': dir_contra,
        'gap_aligns_direction': gap_aligns,
        'gap_opposes_direction': gap_opposes,
        'entry_bucket': entry_bucket,
        'entry_minutes_from_open': entry_minutes_from_open,

        # Keyword features
        'kw_offering': kw_offering,
        'kw_fda': kw_fda,
        'kw_earnings': kw_earnings,
        'kw_analyst': kw_analyst,
        'kw_deal': kw_deal,
        'kw_squeeze': kw_squeeze,
        'kw_beat': kw_beat,
        'kw_miss': kw_miss,
        'kw_guidance_up': kw_guidance_up,
        'kw_guidance_down': kw_guidance_down,
        'kw_contract': kw_contract,
        'kw_bankruptcy': kw_bankruptcy,
        'catalyst_len': catalyst_len,

        # SPY
        'spy_30m': spy_30m,
        'spy_60m': spy_60m,
    }

    rec.update(pnl_results)
    return rec


print(f"\n  Computing comprehensive features for {len(df_dir)} signals...")
records = []
for i, (_, row) in enumerate(df_dir.iterrows()):
    rec = compute_trade_features(row)
    if rec:
        records.append(rec)
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(df_dir)} processed ({len(records)} valid)")

df = pd.DataFrame(records)
print(f"\n  Master feature table: {len(df)} trades")
print(f"    FoldA: {len(df[df['eval_date'].isin(FOLD_A)])}")
print(f"    FoldB: {len(df[df['eval_date'].isin(FOLD_B)])}")

# ASSERTION: no post-entry feature may be used unless entry is rebased.
# first_5min_* features observe bars AFTER entry. They may only be used
# as a filter when PnL is measured from the rebased t+5 entry price
# (pnl_5m_* columns), never with the original pnl_*m_sl* columns.
_POST_ENTRY_FEATURES = {'first_5min_ret', 'first_5min_range', 'first_5min_vol',
                         'first_5min_confirms', 'first_5min_confirms_int'}
print(f"  ASSERTION: post-entry features {_POST_ENTRY_FEATURES} "
      f"must only gate into rebased (pnl_5m_*) columns")

# Save features
df.to_csv(f"{RESULTS_DIR}/creative_features.csv", index=False)


# ==============================================================================
# EVALUATION HELPERS
# ==============================================================================
def eval_pnl(v, label=""):
    if len(v) < 3:
        return {'label': label, 'n': len(v), 'avg': np.nan, 'wr': np.nan}
    return {
        'label': label,
        'n': len(v),
        'avg': round(float(v.mean()), 4),
        'median': round(float(v.median()), 4),
        'wr': round(float((v > 0).sum() / len(v) * 100), 1),
        'se': round(float(v.std() / np.sqrt(len(v))), 4),
        'total': round(float(v.sum()), 2),
        'sharpe': round(float(v.mean() / v.std()), 3) if v.std() > 0 else 0,
    }


def cross_validate(df, filter_fn, label, pnl_col='pnl_60m_sl2_5'):
    df_a = df[df['eval_date'].isin(FOLD_A)]
    df_b = df[df['eval_date'].isin(FOLD_B)]
    r_all = eval_pnl(filter_fn(df)[pnl_col].dropna(), f"{label}_ALL")
    r_a = eval_pnl(filter_fn(df_a)[pnl_col].dropna(), f"{label}_foldA")
    r_b = eval_pnl(filter_fn(df_b)[pnl_col].dropna(), f"{label}_foldB")
    cv_pass = (not np.isnan(r_a.get('avg', np.nan)) and r_a['avg'] > 0 and
               not np.isnan(r_b.get('avg', np.nan)) and r_b['avg'] > 0)
    return {'label': label, 'all': r_all, 'fold_a': r_a, 'fold_b': r_b, 'cv_pass': cv_pass}


all_experiment_results = []


def log_experiment(name, results):
    all_experiment_results.append({'experiment': name, 'results': results})
    print(f"\n  {'─'*80}")
    print(f"  {name}")
    print(f"  {'─'*80}")
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict) and 'all' in r:
                a = r['all']
                cv = "CV:YES" if r.get('cv_pass') else "CV:no"
                fa = r.get('fold_a', {}).get('avg', np.nan)
                fb = r.get('fold_b', {}).get('avg', np.nan)
                print(f"    {r['label']:<45} n={a.get('n',0):>4}  "
                      f"avg={a.get('avg',0):>+7.3f}%  WR={a.get('wr',0):>5.1f}%  "
                      f"SE={a.get('se',0):>5.3f}  {cv}  A={fa:>+.3f}% B={fb:>+.3f}%")
            elif isinstance(r, dict):
                for k, v in r.items():
                    print(f"    {k}: {v}")
    elif isinstance(results, dict):
        for k, v in results.items():
            print(f"    {k}: {v}")


# ==============================================================================
# BASELINE
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"  BASELINE MEASUREMENTS")
print(f"{'='*100}")

baselines = [
    cross_validate(df, lambda d: d, 'BASELINE_all_signals'),
    cross_validate(df, lambda d: d[d['selected'] == True], 'BASELINE_selected'),
    cross_validate(df, lambda d: d[d['direction'] == 'SHORT'], 'BASELINE_short_only'),
    cross_validate(df, lambda d: d[d['direction'] == 'LONG'], 'BASELINE_long_only'),
]
log_experiment('BASELINES', baselines)


# ==============================================================================
# EXP 1: ML CLASSIFIER — predict profitable trades
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"  EXPERIMENT 1: ML CLASSIFIER")
print(f"{'='*100}")

# first_5min_ret, first_5min_range removed: they leak future price into
# features because PnL is measured from the same entry_price those bars
# are relative to. The 27% importance was partly mechanical correlation.
FEATURE_COLS = [
    'confidence', 'remaining_move_pct', 'expected_move_pct',
    'gap_pct', 'abs_gap_pct', 'atr_gap', 'log_vol_ratio',
    'entry_minutes_from_open', 'direction_int',
    'kw_offering', 'kw_fda', 'kw_earnings', 'kw_analyst',
    'kw_deal', 'kw_squeeze', 'kw_beat', 'kw_miss',
    'kw_guidance_up', 'kw_guidance_down', 'kw_contract', 'kw_bankruptcy',
    'catalyst_len',
]

# Binary target: profitable at 60m with SL
df['target_60m'] = (df['pnl_60m_sl2_5'] > 0).astype(int)
df['target_30m'] = (df['pnl_30m_sl2_5'] > 0).astype(int)

# Also regression target
df['pnl_60m'] = df['pnl_60m_sl2_5']

# Add one-hot for catalyst_type and scanner
for cat in df['catalyst_type'].unique():
    df[f'cat_{cat}'] = (df['catalyst_type'] == cat).astype(int)
    FEATURE_COLS.append(f'cat_{cat}')

df['scanner_discovery'] = (df['scanner'] == 'discovery').astype(int)
FEATURE_COLS.append('scanner_discovery')

# Add exhaustion and contradiction flags
df['exhausted_int'] = df['move_is_exhausted'].astype(int)
df['contra_int'] = df['dir_remaining_contradiction'].astype(int)
df['gap_aligns_int'] = df['gap_aligns_direction'].astype(int)
df['gap_opposes_int'] = df['gap_opposes_direction'].astype(int)
df['first_5min_confirms_int'] = df['first_5min_confirms'].astype(int)
# first_5min_confirms_int removed: leaks post-entry price data
FEATURE_COLS.extend(['exhausted_int', 'contra_int', 'gap_aligns_int',
                      'gap_opposes_int'])

# Clean features
df_ml = df.dropna(subset=['pnl_60m_sl2_5'] + FEATURE_COLS).copy()
for c in FEATURE_COLS:
    df_ml[c] = pd.to_numeric(df_ml[c], errors='coerce').fillna(0)

print(f"\n  ML dataset: {len(df_ml)} samples, {len(FEATURE_COLS)} features")

# Cross-validated ML: train on FoldA, test on FoldB; then reverse
ml_results = []

for train_fold, test_fold, fold_name in [
    (FOLD_A, FOLD_B, 'train_A_test_B'),
    (FOLD_B, FOLD_A, 'train_B_test_A'),
]:
    df_train = df_ml[df_ml['eval_date'].isin(train_fold)]
    df_test = df_ml[df_ml['eval_date'].isin(test_fold)]

    if len(df_train) < 10 or len(df_test) < 10:
        print(f"  Skipping {fold_name}: not enough data")
        continue

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train['target_60m'].values
    X_test = df_test[FEATURE_COLS].values
    y_test = df_test['target_60m'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    for model_name, model in [
        ('GBM', GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42,
                                            min_samples_leaf=5, subsample=0.8)),
        ('RF', RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42,
                                       min_samples_leaf=5)),
        ('Logistic', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
    ]:
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:, 1]

        # Strategy: trade only when model says >60% probability of profit
        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
            mask = proba >= threshold
            if mask.sum() < 3:
                continue
            selected_pnl = df_test.loc[mask, 'pnl_60m_sl2_5'].dropna()
            r = eval_pnl(selected_pnl, f"{model_name}_p{threshold:.0%}_{fold_name}")
            r['threshold'] = threshold
            r['model'] = model_name
            r['fold'] = fold_name
            ml_results.append(r)

        # Feature importance (for GBM)
        if model_name == 'GBM' and fold_name == 'train_A_test_B':
            importances = model.feature_importances_
            feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
            print(f"\n  GBM Feature Importance ({fold_name}):")
            for fname, imp in feat_imp[:15]:
                print(f"    {fname:<35} {imp:.4f}")

log_experiment('EXP 1: ML CLASSIFIER', ml_results)

# Find best ML strategy (positive in both folds)
ml_by_model_thresh = defaultdict(list)
for r in ml_results:
    key = f"{r['model']}_p{r['threshold']:.0%}"
    ml_by_model_thresh[key].append(r)

print(f"\n  ML Cross-Fold Summary:")
for key, folds in ml_by_model_thresh.items():
    if len(folds) == 2:
        avg_a = folds[0]['avg']
        avg_b = folds[1]['avg']
        n_a = folds[0]['n']
        n_b = folds[1]['n']
        both_pos = avg_a > 0 and avg_b > 0
        cv = "CV:YES" if both_pos else "CV:no"
        combined_avg = (avg_a * n_a + avg_b * n_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0
        print(f"    {key:<30} combined_avg={combined_avg:+.3f}%  "
              f"A={avg_a:+.3f}% (n={n_a})  B={avg_b:+.3f}% (n={n_b})  {cv}")


# ==============================================================================
# EXP 2: PRICE-ONLY MODEL — ignore LLM entirely
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"  EXPERIMENT 2: PRICE-ONLY MODEL (no LLM)")
print(f"{'='*100}")

# Strategy: use ONLY gap + volume to decide direction and trade
# Hypothesis: fade large gaps (mean reversion), ride confirmed momentum
price_results = []

# 2a: Gap-fade — SHORT stocks gapping up > X%, LONG stocks gapping down > X%
for gap_thresh in [3, 5, 8, 10, 15]:
    # Fade: trade opposite of gap
    def _fade(d, gt=gap_thresh):
        mask = (
            ((d['gap_pct'] > gt) & (d['direction'] == 'SHORT')) |
            ((d['gap_pct'] < -gt) & (d['direction'] == 'LONG'))
        )
        return d[mask]

    r = cross_validate(df, _fade, f'FADE_gap>{gap_thresh}%')
    price_results.append(r)

# 2b: Momentum — trade same as gap direction
for gap_thresh in [2, 3, 5]:
    def _momentum(d, gt=gap_thresh):
        mask = (
            ((d['gap_pct'] > gt) & (d['direction'] == 'LONG')) |
            ((d['gap_pct'] < -gt) & (d['direction'] == 'SHORT'))
        )
        return d[mask]

    r = cross_validate(df, _momentum, f'MOMENTUM_gap>{gap_thresh}%')
    price_results.append(r)

# 2c: Volume-confirmed momentum — high premarket vol + gap direction
for vol_thresh in [0.01, 0.03, 0.05]:
    def _vol_momentum(d, vt=vol_thresh):
        mask = (
            (d['vol_ratio'] >= vt) &
            (
                ((d['gap_pct'] > 2) & (d['direction'] == 'LONG')) |
                ((d['gap_pct'] < -2) & (d['direction'] == 'SHORT'))
            )
        )
        return d[mask]

    r = cross_validate(df, _vol_momentum, f'VOL_MOMENTUM_vr>{vol_thresh}')
    price_results.append(r)

# 2d: Small gap + high volume (stealth moves)
def _stealth(d):
    return d[(d['abs_gap_pct'] < 3) & (d['vol_ratio'] > 0.02)]
price_results.append(cross_validate(df, _stealth, 'STEALTH_smallgap_highvol'))

# 2e: ATR-normalized gap filter
for atr_thresh in [1.0, 1.5, 2.0, 3.0]:
    def _atr_small(d, at=atr_thresh):
        return d[d['atr_gap'] < at]
    price_results.append(cross_validate(df, _atr_small, f'ATR_gap<{atr_thresh}'))

    def _atr_large(d, at=atr_thresh):
        return d[d['atr_gap'] >= at]
    price_results.append(cross_validate(df, _atr_large, f'ATR_gap>={atr_thresh}'))

log_experiment('EXP 2: PRICE-ONLY MODELS', price_results)


# ==============================================================================
# EXP 3: ANTI-SELECTION — invert the scanner
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"  EXPERIMENT 3: ANTI-SELECTION (invert scanner)")
print(f"{'='*100}")

anti_results = []

# 3a: Trade only NOT-selected signals
anti_results.append(cross_validate(df, lambda d: d[d['selected'] == False],
                                    'ANTI_not_selected'))

# 3b: Trade only gated signals (scanner's rejects)
anti_results.append(cross_validate(df, lambda d: d[d['gate_reason'] != ''],
                                    'ANTI_gated_only'))

# 3c: Trade low-confidence signals (conf <= 3)
anti_results.append(cross_validate(df, lambda d: d[d['confidence'] <= 3],
                                    'ANTI_low_conf_le3'))

# 3d: Trade exhausted signals only
anti_results.append(cross_validate(df, lambda d: d[d['move_is_exhausted'] == True],
                                    'ANTI_exhausted_only'))

# 3e: Flip direction — take the opposite of what scanner says
# DIAGNOSTIC ONLY, NOT A STRATEGY:
#   1. pnl_flipped = -pnl inverts a stop-adjusted series (approximate)
#   2. Selection function changed 4x across V2→V4, so "flip selected" is
#      not a stable tradeable object
# Kept as evidence about selection quality, not as an implementable filter.
df['pnl_60m_flipped_sl2_5'] = -df['pnl_60m_sl2_5']
anti_results.append(cross_validate(df, lambda d: d[d['selected'] == True],
                                    'DIAG_flip_selected_direction',
                                    pnl_col='pnl_60m_flipped_sl2_5'))

# 3f: Only signals where scanner was confident but wrong before (anti-pattern)
anti_results.append(cross_validate(
    df,
    lambda d: d[(d['confidence'] >= 5) & (d['dir_remaining_contradiction'] == True)],
    'ANTI_highconf_contradiction'))

log_experiment('EXP 3: ANTI-SELECTION', anti_results)


# ==============================================================================
# EXP 4: GAP-REVERSION SYSTEM
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"  EXPERIMENT 4: GAP-REVERSION SYSTEM")
print(f"{'='*100}")

gap_results = []

# 4a: Pure gap-fade on large gappers (any direction, always fade)
# We need to look at this differently — ignore LLM direction
# Check if gapping up stocks tend to reverse (SHORT them) and vice versa
for gap_thresh in [5, 8, 10, 15, 20]:
    # Gap-up stocks that scanner calls SHORT
    gap_results.append(cross_validate(
        df,
        lambda d, gt=gap_thresh: d[(d['gap_pct'] > gt) & (d['direction'] == 'SHORT')],
        f'GAP_FADE_up>{gap_thresh}%_short'))

    # Gap-down stocks that scanner calls LONG
    gap_results.append(cross_validate(
        df,
        lambda d, gt=gap_thresh: d[(d['gap_pct'] < -gt) & (d['direction'] == 'LONG')],
        f'GAP_FADE_down>{gap_thresh}%_long'))

# 4b: Small gap continuation (momentum)
gap_results.append(cross_validate(
    df,
    lambda d: d[(d['abs_gap_pct'] >= 2) & (d['abs_gap_pct'] <= 8) & (d['gap_aligns_direction'] == True)],
    'GAP_CONT_2to8_aligned'))

# 4c: Overshoot detection (Fable's 6b) — |gap| > 1.5x expected_move
def _overshoot(d):
    has_exp = d['expected_move_pct'].abs() > 0.1
    overshoot = d['abs_gap_pct'] > 1.5 * d['expected_move_pct'].abs()
    return d[has_exp & overshoot]

def _not_overshoot(d):
    has_exp = d['expected_move_pct'].abs() > 0.1
    overshoot = d['abs_gap_pct'] > 1.5 * d['expected_move_pct'].abs()
    return d[has_exp & ~overshoot]

gap_results.append(cross_validate(df, _overshoot, 'GAP_overshoot'))
gap_results.append(cross_validate(df, _not_overshoot, 'GAP_not_overshoot'))

log_experiment('EXP 4: GAP-REVERSION', gap_results)


# ==============================================================================
# EXP 5: CATALYST KEYWORD NLP
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"  EXPERIMENT 5: CATALYST KEYWORD NLP")
print(f"{'='*100}")

nlp_results = []

# Mine the catalyst text for predictive tokens
# FIX: token selection must be fold-local. Previous version mined tokens
# across all 4 days then "cross-validated" — the test fold picked the tokens.
from collections import Counter

def tokenize(text):
    return re.findall(r'[a-z]+', text.lower())


def _mine_tokens(df_train, lift_thresh, min_n=5):
    """Mine bullish/bearish tokens from TRAINING fold only."""
    all_tok = Counter()
    pos_tok = Counter()
    valid = df_train.dropna(subset=['pnl_60m_sl2_5'])
    base_rate = (valid['pnl_60m_sl2_5'] > 0).mean() * 100
    for _, row in valid.iterrows():
        tokens = set(tokenize(str(row['catalyst_text'])))
        for t in tokens:
            all_tok[t] += 1
            if row['pnl_60m_sl2_5'] > 0:
                pos_tok[t] += 1
    lifts = []
    for t, n in all_tok.items():
        if n < min_n:
            continue
        pos_rate = pos_tok.get(t, 0) / n * 100
        lift = pos_rate - base_rate
        lifts.append((t, n, pos_rate, lift))
    bullish = set(t for t, n, pr, l in lifts if l > lift_thresh)
    bearish = set(t for t, n, pr, l in lifts if l < -lift_thresh)
    return bullish, bearish, lifts


# Print full-sample token analysis for reference (NOT used for CV)
print(f"\n  Full-sample token analysis (diagnostic only):")
_, _, token_lifts_all = _mine_tokens(df, lift_thresh=0, min_n=5)
token_lifts_all.sort(key=lambda x: -x[3])
# Keep token_lifts for the JSON export
token_lifts = token_lifts_all
print(f"\n  TOP BULLISH TOKENS (high lift):")
for t, n, pr, lift in token_lifts_all[:15]:
    print(f"    {t:<20} n={n:>4}  WR={pr:>5.1f}%  lift={lift:>+5.1f}pp")
print(f"\n  TOP BEARISH TOKENS (negative lift):")
for t, n, pr, lift in sorted(token_lifts_all, key=lambda x: x[3])[:15]:
    print(f"    {t:<20} n={n:>4}  WR={pr:>5.1f}%  lift={lift:>+5.1f}pp")


# Fold-local NLP CV: mine tokens in train fold, test on held-out fold
def _nlp_fold_cv(df, label, mode='no_bearish', lift_thresh=10, min_n=5,
                 pnl_col='pnl_60m_sl2_5'):
    """Proper CV: tokens selected in train fold, applied to test fold."""
    df_a = df[df['eval_date'].isin(FOLD_A)]
    df_b = df[df['eval_date'].isin(FOLD_B)]

    # Train on A, test on B
    bull_a, bear_a, _ = _mine_tokens(df_a, lift_thresh, min_n)
    # Train on B, test on A
    bull_b, bear_b, _ = _mine_tokens(df_b, lift_thresh, min_n)

    if mode == 'no_bearish':
        # Test on B using A's bearish tokens
        if bear_a:
            mask_b = df_b['catalyst_text'].apply(
                lambda x: not bool(bear_a & set(tokenize(str(x)))))
            test_b = df_b[mask_b]
        else:
            test_b = df_b
        # Test on A using B's bearish tokens
        if bear_b:
            mask_a = df_a['catalyst_text'].apply(
                lambda x: not bool(bear_b & set(tokenize(str(x)))))
            test_a = df_a[mask_a]
        else:
            test_a = df_a
    elif mode == 'has_bullish':
        if bull_a:
            mask_b = df_b['catalyst_text'].apply(
                lambda x: bool(bull_a & set(tokenize(str(x)))))
            test_b = df_b[mask_b]
        else:
            test_b = df_b.head(0)
        if bull_b:
            mask_a = df_a['catalyst_text'].apply(
                lambda x: bool(bull_b & set(tokenize(str(x)))))
            test_a = df_a[mask_a]
        else:
            test_a = df_a.head(0)
    else:
        return None

    r_a = eval_pnl(test_a[pnl_col].dropna(), f"{label}_foldA")
    r_b = eval_pnl(test_b[pnl_col].dropna(), f"{label}_foldB")
    # Combined stats
    combined = pd.concat([test_a, test_b])
    r_all = eval_pnl(combined[pnl_col].dropna(), f"{label}_ALL")
    cv_pass = (not np.isnan(r_a.get('avg', np.nan)) and r_a['avg'] > 0 and
               not np.isnan(r_b.get('avg', np.nan)) and r_b['avg'] > 0)

    # Log which tokens were found
    if mode == 'no_bearish':
        print(f"    {label}: bearish_A={bear_a if bear_a else '{}'}, "
              f"bearish_B={bear_b if bear_b else '{}'}")
    else:
        print(f"    {label}: bullish_A={bull_a if bull_a else '{}'}, "
              f"bullish_B={bull_b if bull_b else '{}'}")

    return {'label': label, 'all': r_all, 'fold_a': r_a, 'fold_b': r_b, 'cv_pass': cv_pass}


for lt in [5, 10, 15]:
    r = _nlp_fold_cv(df, f'NLP_no_bearish_lift>{lt}', mode='no_bearish', lift_thresh=lt)
    if r:
        nlp_results.append(r)
    r = _nlp_fold_cv(df, f'NLP_has_bullish_lift>{lt}', mode='has_bullish', lift_thresh=lt)
    if r:
        nlp_results.append(r)

# Specific keyword strategies based on root cause analysis
kw_strats = [
    ('NLP_offering_short', lambda d: d[(d['kw_offering'] == 1) & (d['direction'] == 'SHORT')]),
    ('NLP_fda_any', lambda d: d[d['kw_fda'] == 1]),
    ('NLP_no_analyst_kw', lambda d: d[d['kw_analyst'] == 0]),
    ('NLP_no_earnings_kw', lambda d: d[d['kw_earnings'] == 0]),
    ('NLP_has_miss_short', lambda d: d[(d['kw_miss'] == 1) & (d['direction'] == 'SHORT')]),
    ('NLP_has_beat_long', lambda d: d[(d['kw_beat'] == 1) & (d['direction'] == 'LONG')]),
    ('NLP_guidance_down_short', lambda d: d[(d['kw_guidance_down'] == 1) & (d['direction'] == 'SHORT')]),
    ('NLP_contract_long', lambda d: d[(d['kw_contract'] == 1) & (d['direction'] == 'LONG')]),
    ('NLP_long_catalyst_text', lambda d: d[d['catalyst_len'] > 200]),
    ('NLP_short_catalyst_text', lambda d: d[d['catalyst_len'] < 100]),
]

for name, fn in kw_strats:
    nlp_results.append(cross_validate(df, fn, name))

log_experiment('EXP 5: CATALYST KEYWORD NLP', nlp_results)


# ==============================================================================
# EXP 6: FIRST-5-MIN CONFIRMATION
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"  EXPERIMENT 6: FIRST-5-MIN CONFIRMATION")
print(f"{'='*100}")

confirm_results = []

# LEAKAGE FIX: Previous version measured PnL from entry_price (open at t=0)
# but filtered on first_5min_ret (also measured from entry_price). You can't
# buy at 9:30 knowing what 9:35 looks like. Fix: entry_price rebased to
# close of bar at t+5, PnL measured from that price forward.
# The rebased columns (pnl_5m_*) were computed in compute_trade_features.

# Only include trades that have a valid t+5 entry price
df_5m = df[df['entry_price_5m'].notna() & (df['entry_price_5m'] > 0)].copy()
print(f"\n  Trades with valid t+5 entry: {len(df_5m)} / {len(df)}")

REBASED_COL = 'pnl_5m_60m_sl2_5'
REBASED_COL_NOSL = 'pnl_5m_60m_sl999_0'

# 6a: Only trade if first 5 min confirms direction — REBASED ENTRY
confirm_results.append(cross_validate(
    df_5m, lambda d: d[d['first_5min_confirms'] == True],
    'CONFIRM_5min_confirms_REBASED', pnl_col=REBASED_COL))

# 6b: Contrarian — REBASED
confirm_results.append(cross_validate(
    df_5m, lambda d: d[d['first_5min_confirms'] == False],
    'CONFIRM_5min_contrarian_REBASED', pnl_col=REBASED_COL))

# 6c: Strong confirmation >0.3% — REBASED
def _strong_confirm(d):
    long_confirm = (d['direction'] == 'LONG') & (d['first_5min_ret'] > 0.3)
    short_confirm = (d['direction'] == 'SHORT') & (d['first_5min_ret'] < -0.3)
    return d[long_confirm | short_confirm]
confirm_results.append(cross_validate(df_5m, _strong_confirm,
    'CONFIRM_5min_strong>0.3%_REBASED', pnl_col=REBASED_COL))

# 6d: Strong >0.5% — REBASED
def _strong_confirm_05(d):
    long_confirm = (d['direction'] == 'LONG') & (d['first_5min_ret'] > 0.5)
    short_confirm = (d['direction'] == 'SHORT') & (d['first_5min_ret'] < -0.5)
    return d[long_confirm | short_confirm]
confirm_results.append(cross_validate(df_5m, _strong_confirm_05,
    'CONFIRM_5min_strong>0.5%_REBASED', pnl_col=REBASED_COL))

# 6e: Combined confirm + good catalyst — REBASED
confirm_results.append(cross_validate(
    df_5m,
    lambda d: d[(d['first_5min_confirms'] == True) &
                (d['catalyst_type'].isin(['regulatory', 'offering', 'squeeze']))],
    'CONFIRM_5min+goodcat_REBASED', pnl_col=REBASED_COL))

# 6f: Confirmation + SHORT bias — REBASED
confirm_results.append(cross_validate(
    df_5m,
    lambda d: d[(d['first_5min_confirms'] == True) & (d['direction'] == 'SHORT')],
    'CONFIRM_5min+SHORT_REBASED', pnl_col=REBASED_COL))

# 6g: Confirmation with wider SL — REBASED
confirm_results.append(cross_validate(
    df_5m, lambda d: d[d['first_5min_confirms'] == True],
    'CONFIRM_5min_confirms_REBASED_noSL', pnl_col=REBASED_COL_NOSL))

# Also run the ORIGINAL (leaky) versions for comparison, clearly labeled
confirm_results.append(cross_validate(
    df, lambda d: d[d['first_5min_confirms'] == True],
    'LEAKY_5min_confirms_original'))
confirm_results.append(cross_validate(
    df, _strong_confirm, 'LEAKY_5min_strong>0.3%_original'))

log_experiment('EXP 6: FIRST-5-MIN CONFIRMATION (REBASED t+5 ENTRY)', confirm_results)


# ==============================================================================
# EXP 7: ENSEMBLE OF MICRO-SIGNALS
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"  EXPERIMENT 7: ENSEMBLE OF MICRO-SIGNALS")
print(f"{'='*100}")

# Build a simple scoring system from individual weak signals
# Each signal contributes +1 or -1, sum decides trade/no-trade

def compute_ensemble_score(row):
    score = 0

    # Direction: SHORT bias (+1)
    if row['direction'] == 'SHORT':
        score += 1

    # Scanner: discovery (+1)
    if row['scanner'] == 'discovery':
        score += 1

    # Good catalyst (+2) vs bad catalyst (-2)
    if row['catalyst_type'] in ('offering', 'regulatory', 'squeeze'):
        score += 2
    elif row['catalyst_type'] in ('analyst', 'unknown', 'bankruptcy'):
        score -= 2
    elif row['catalyst_type'] in ('earnings', 'deal', 'breaking'):
        score -= 1

    # No contradiction (+1)
    if not row['dir_remaining_contradiction']:
        score += 1
    else:
        score -= 2

    # Not exhausted (+1)
    if not row['move_is_exhausted']:
        score += 1

    # Gap opposes direction (+1 for contrarian)
    if row['gap_opposes_direction']:
        score += 1

    # first_5min_confirms removed: leaks post-entry price data

    # Small gap (+1)
    if row['abs_gap_pct'] < 10:
        score += 1
    elif row['abs_gap_pct'] > 20:
        score -= 1

    # Morning entry (+1)
    if row['entry_minutes_from_open'] <= 30:
        score += 1

    # Volume ratio above baseline (+1)
    if row['vol_ratio'] > 0.02:
        score += 1

    return score

df['ensemble_score'] = df.apply(compute_ensemble_score, axis=1)

ensemble_results = []
for thresh in range(0, 8):
    ensemble_results.append(cross_validate(
        df, lambda d, t=thresh: d[d['ensemble_score'] >= t],
        f'ENSEMBLE_score>={thresh}'))

# Also test negative ensemble (anti-selection)
for thresh in [-2, -1, 0, 1]:
    ensemble_results.append(cross_validate(
        df, lambda d, t=thresh: d[d['ensemble_score'] <= t],
        f'ENSEMBLE_score<={thresh}'))

log_experiment('EXP 7: ENSEMBLE OF MICRO-SIGNALS', ensemble_results)


# ==============================================================================
# EXP 8: ADAPTIVE SL/TP — scale risk to gap and ATR
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"  EXPERIMENT 8: ADAPTIVE SL/TP")
print(f"{'='*100}")

adaptive_results = []

# Test different SL levels for different gap sizes
# Fable's recommendation: SL = max(2.5%, 0.25 × ATR14/price)

# 8a: Different fixed SL across strategies
best_filters = [
    ('all', lambda d: d),
    ('short_only', lambda d: d[d['direction'] == 'SHORT']),
    ('short_goodcat', lambda d: d[(d['direction'] == 'SHORT') &
                                   (d['catalyst_type'].isin(['regulatory', 'offering', 'squeeze']))]),
]

for fname, ffn in best_filters:
    for sl in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 999.0]:
        sl_key = f"sl{sl:.1f}".replace('.', '_')
        col_60 = f'pnl_60m_{sl_key}'
        col_30 = f'pnl_30m_{sl_key}'
        if col_60 not in df.columns:
            continue
        r60 = cross_validate(df, ffn, f'SL_{fname}_sl{sl:.1f}_60m', pnl_col=col_60)
        r30 = cross_validate(df, ffn, f'SL_{fname}_sl{sl:.1f}_30m', pnl_col=col_30)
        adaptive_results.append(r60)
        adaptive_results.append(r30)

# 8b: ATR-scaled SL — use wider SL for volatile stocks
# Proxy: if gap > 10%, use SL=5%; if gap < 5%, use SL=2.5%
def _adaptive_sl_pnl(d):
    results = []
    for _, r in d.iterrows():
        if r['abs_gap_pct'] > 10:
            pnl = r.get('pnl_60m_sl5_0', np.nan)
        elif r['abs_gap_pct'] > 5:
            pnl = r.get('pnl_60m_sl3_0', np.nan)
        else:
            pnl = r.get('pnl_60m_sl2_5', np.nan)
        results.append(pnl)
    return pd.Series(results, index=d.index)

df['pnl_60m_adaptive_sl'] = _adaptive_sl_pnl(df)

adaptive_results.append(cross_validate(df, lambda d: d, 'ADAPTIVE_SL_gap_scaled',
                                        pnl_col='pnl_60m_adaptive_sl'))
adaptive_results.append(cross_validate(
    df,
    lambda d: d[d['direction'] == 'SHORT'],
    'ADAPTIVE_SL_short_gap_scaled',
    pnl_col='pnl_60m_adaptive_sl'))

# 8c: 30m vs 60m horizon comparison for top strategies
for ffn_name, ffn in [
    ('short_goodcat', lambda d: d[(d['direction'] == 'SHORT') &
                                   (d['catalyst_type'].isin(['regulatory', 'offering', 'squeeze']))]),
    ('ensemble_ge5', lambda d: d[d['ensemble_score'] >= 5]),
]:
    for h_col in ['pnl_30m_sl2_5', 'pnl_60m_sl2_5']:
        adaptive_results.append(cross_validate(df, ffn, f'HORIZON_{ffn_name}_{h_col}',
                                                pnl_col=h_col))

log_experiment('EXP 8: ADAPTIVE SL/TP', adaptive_results)


# ==============================================================================
# GRAND SUMMARY — ALL EXPERIMENTS
# ==============================================================================
print(f"\n\n{'='*120}")
print(f"  GRAND SUMMARY — ALL EXPERIMENTS")
print(f"{'='*120}")

# Collect all cross-validated results
all_cv_results = []
for exp in all_experiment_results:
    for r in exp['results']:
        if isinstance(r, dict) and 'all' in r:
            all_cv_results.append({
                'experiment': exp['experiment'],
                'label': r['label'],
                'n': r['all'].get('n', 0),
                'avg_pnl': r['all'].get('avg', np.nan),
                'wr': r['all'].get('wr', np.nan),
                'se': r['all'].get('se', np.nan),
                'sharpe': r['all'].get('sharpe', np.nan),
                'cv_pass': r.get('cv_pass', False),
                'fold_a_avg': r['fold_a'].get('avg', np.nan) if r.get('fold_a') else np.nan,
                'fold_b_avg': r['fold_b'].get('avg', np.nan) if r.get('fold_b') else np.nan,
                'fold_a_n': r['fold_a'].get('n', 0) if r.get('fold_a') else 0,
                'fold_b_n': r['fold_b'].get('n', 0) if r.get('fold_b') else 0,
            })

df_results = pd.DataFrame(all_cv_results)

# Sort by avg PnL
df_results_sorted = df_results.sort_values('avg_pnl', ascending=False)

print(f"\n  Total strategies evaluated: {len(df_results_sorted)}")
print(f"  Cross-validated winners: {df_results_sorted['cv_pass'].sum()}")

# TOP 30 overall
print(f"\n  TOP 30 STRATEGIES (by avg PnL):")
print(f"  {'#':>3} {'Experiment':<25} {'Strategy':<45} {'n':>4} {'Avg':>8} {'WR':>6} "
      f"{'SE':>6} {'CV':>4} {'A':>8} {'B':>8}")
print(f"  {'':─<118}")

for i, (_, r) in enumerate(df_results_sorted.head(30).iterrows()):
    cv = "YES" if r['cv_pass'] else "no"
    print(f"  {i+1:>3} {str(r['experiment'])[:24]:<25} {r['label']:<45} {r['n']:>4} "
          f"{r['avg_pnl']:>+7.3f}% {r['wr']:>5.1f}% {r['se']:>5.3f} {cv:>4} "
          f"{r['fold_a_avg']:>+7.3f}% {r['fold_b_avg']:>+7.3f}%")

# Cross-validated winners above +1%
cv_winners = df_results_sorted[(df_results_sorted['cv_pass'] == True) &
                                (df_results_sorted['n'] >= 5)]
print(f"\n\n  CROSS-VALIDATED WINNERS (n>=5, positive in BOTH folds):")
print(f"  {'#':>3} {'Strategy':<50} {'n':>4} {'Avg':>8} {'WR':>6} {'A_avg':>8} {'A_n':>4} {'B_avg':>8} {'B_n':>4}")
print(f"  {'':─<100}")

for i, (_, r) in enumerate(cv_winners.iterrows()):
    print(f"  {i+1:>3} {r['label']:<50} {r['n']:>4} {r['avg_pnl']:>+7.3f}% {r['wr']:>5.1f}% "
          f"{r['fold_a_avg']:>+7.3f}% {r['fold_a_n']:>4} {r['fold_b_avg']:>+7.3f}% {r['fold_b_n']:>4}")

# Strategies above 2%
above_2 = df_results_sorted[df_results_sorted['avg_pnl'] >= 2.0]
print(f"\n\n  STRATEGIES WITH AVG PnL >= +2.0%:")
if len(above_2) > 0:
    for _, r in above_2.iterrows():
        cv = "CV:YES" if r['cv_pass'] else "CV:no"
        print(f"    {r['label']:<50} n={r['n']:>4} avg={r['avg_pnl']:>+.3f}% WR={r['wr']:.0f}% {cv}")
else:
    print("    None at 60m/SL2.5%. Checking other configurations...")
    above_1 = df_results_sorted[df_results_sorted['avg_pnl'] >= 1.0]
    if len(above_1) > 0:
        print(f"\n    Strategies above +1.0%:")
        for _, r in above_1.iterrows():
            cv = "CV:YES" if r['cv_pass'] else "CV:no"
            print(f"    {r['label']:<50} n={r['n']:>4} avg={r['avg_pnl']:>+.3f}% WR={r['wr']:.0f}% {cv}")


# ==============================================================================
# CHARTS
# ==============================================================================

# Chart 1: Experiment comparison — top strategies per experiment
fig1, axes = plt.subplots(2, 2, figsize=(20, 16))
fig1.suptitle('Creative Experiments: Top Strategies per Approach', fontsize=14, fontweight='bold')

exp_groups = {
    'ML Classifier': [r for r in all_cv_results if 'EXP 1' in str(r.get('experiment', ''))],
    'Price-Only': [r for r in all_cv_results if 'EXP 2' in str(r.get('experiment', ''))],
    'Gap-Reversion': [r for r in all_cv_results if 'EXP 4' in str(r.get('experiment', ''))],
    'Ensemble': [r for r in all_cv_results if 'EXP 7' in str(r.get('experiment', ''))],
}

for ax, (title, group) in zip(axes.flat, exp_groups.items()):
    if not group:
        ax.text(0.5, 0.5, f'No data for {title}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        continue
    top = sorted(group, key=lambda x: x.get('avg_pnl', -999), reverse=True)[:10]
    names = [r['label'][:35] for r in top]
    vals = [r.get('avg_pnl', 0) for r in top]
    ns = [r.get('n', 0) for r in top]
    colors = ['#27ae60' if v > 0 else '#c0392b' for v in vals]
    ax.barh(range(len(names)), vals, color=colors, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} (n={nn})" for n, nn in zip(names, ns)], fontsize=7)
    for i, v in enumerate(vals):
        ax.text(v + 0.05 * (1 if v >= 0 else -1), i, f"{v:+.2f}%", va='center', fontsize=7)
    ax.axvline(x=0, color='black', lw=0.8)
    ax.set_xlabel('Avg PnL @60m %')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig1_path = f"{OUTPUT_DIR}/creative_experiments_comparison.png"
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart 1 saved: {fig1_path}")
plt.close()


# Chart 2: Ensemble score distribution vs PnL
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig2.suptitle('Ensemble Score Analysis', fontsize=13, fontweight='bold')

scores = sorted(df['ensemble_score'].unique())
avg_by_score = []
n_by_score = []
for s in scores:
    v = df[df['ensemble_score'] == s]['pnl_60m_sl2_5'].dropna()
    avg_by_score.append(v.mean() if len(v) > 0 else 0)
    n_by_score.append(len(v))

colors_s = ['#27ae60' if v > 0 else '#c0392b' for v in avg_by_score]
ax1.bar(scores, avg_by_score, color=colors_s, alpha=0.8)
ax1.set_xlabel('Ensemble Score')
ax1.set_ylabel('Avg PnL @60m %')
ax1.set_title('PnL by Ensemble Score')
ax1.axhline(y=0, color='black', lw=0.8)
for i, (s, v, n) in enumerate(zip(scores, avg_by_score, n_by_score)):
    ax1.annotate(f"n={n}\n{v:+.1f}%", xy=(s, v), ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

# Cumulative: how PnL changes as we raise the threshold
thresholds = list(range(min(scores), max(scores) + 1))
cum_avg = []
cum_n = []
for t in thresholds:
    v = df[df['ensemble_score'] >= t]['pnl_60m_sl2_5'].dropna()
    cum_avg.append(v.mean() if len(v) > 0 else 0)
    cum_n.append(len(v))

ax2.plot(thresholds, cum_avg, 'b-o', lw=2, label='Avg PnL')
ax2_twin = ax2.twinx()
ax2_twin.bar(thresholds, cum_n, alpha=0.2, color='gray', label='Trade count')
ax2.set_xlabel('Minimum Ensemble Score')
ax2.set_ylabel('Avg PnL @60m %')
ax2_twin.set_ylabel('Trade Count')
ax2.set_title('PnL vs Minimum Score Threshold')
ax2.axhline(y=0, color='black', lw=0.8)
ax2.axhline(y=2.0, color='green', lw=0.8, ls='--', alpha=0.5, label='+2% target')
ax2.legend(loc='upper left', fontsize=9)
ax2_twin.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig2_path = f"{OUTPUT_DIR}/creative_ensemble_analysis.png"
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"  Chart 2 saved: {fig2_path}")
plt.close()


# Chart 3: ML feature importance + cross-validation
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig3.suptitle('ML Model Analysis', fontsize=13, fontweight='bold')

# Feature importance (if GBM was trained)
if 'feat_imp' in dir():
    top_feats = feat_imp[:15]
    fnames = [f[0] for f in top_feats]
    fimps = [f[1] for f in top_feats]
    ax1.barh(range(len(fnames)), fimps, color='#3498db', height=0.6)
    ax1.set_yticks(range(len(fnames)))
    ax1.set_yticklabels(fnames, fontsize=8)
    ax1.set_xlabel('Feature Importance')
    ax1.set_title('GBM Feature Importance')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

# ML strategies performance
ml_strats = [r for r in all_cv_results if 'EXP 1' in str(r.get('experiment', ''))
             and r.get('n', 0) >= 3]
if ml_strats:
    ml_sorted = sorted(ml_strats, key=lambda x: x.get('avg_pnl', -999), reverse=True)[:10]
    names_ml = [r['label'][:30] for r in ml_sorted]
    vals_ml = [r.get('avg_pnl', 0) for r in ml_sorted]
    colors_ml = ['#27ae60' if v > 0 else '#c0392b' for v in vals_ml]
    ax2.barh(range(len(names_ml)), vals_ml, color=colors_ml, height=0.7)
    ax2.set_yticks(range(len(names_ml)))
    ax2.set_yticklabels(names_ml, fontsize=8)
    ax2.axvline(x=0, color='black', lw=0.8)
    ax2.set_xlabel('Avg PnL @60m %')
    ax2.set_title('ML Strategy Performance')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

plt.tight_layout()
fig3_path = f"{OUTPUT_DIR}/creative_ml_analysis.png"
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


results_export = {
    'generated': datetime.utcnow().isoformat(),
    'config': {
        'dates': ALL_DATES, 'fold_a': FOLD_A, 'fold_b': FOLD_B,
    },
    'n_trades': len(df),
    'n_strategies_total': len(all_cv_results),
    'all_results': df_results_sorted.to_dict('records'),
    'cv_winners': cv_winners.to_dict('records') if len(cv_winners) > 0 else [],
    'above_2pct': above_2.to_dict('records') if len(above_2) > 0 else [],
    'ensemble_score_curve': [
        {'threshold': t, 'avg_pnl': a, 'n': n}
        for t, a, n in zip(thresholds, cum_avg, cum_n)
    ],
    'token_analysis': {
        'bullish': [(t, n, round(pr, 1), round(lift, 1))
                    for t, n, pr, lift in token_lifts[:20] if lift > 5],
        'bearish': [(t, n, round(pr, 1), round(lift, 1))
                    for t, n, pr, lift in sorted(token_lifts, key=lambda x: x[3])[:20]],
    },
}

json_path = f"{RESULTS_DIR}/creative_experiments_results.json"
with open(json_path, 'w') as f:
    json.dump(results_export, f, indent=2, cls=_NumpyEncoder)
print(f"\n  Results JSON saved: {json_path}")

# Save features for further analysis
df.to_csv(f"{RESULTS_DIR}/creative_features_full.csv", index=False)
print(f"  Full features CSV saved: {RESULTS_DIR}/creative_features_full.csv")

print(f"\n\n{'='*100}")
print(f"  CREATIVE EXPERIMENTS COMPLETE")
print(f"  {len(all_cv_results)} strategies across 8 experiments")
print(f"  CV winners: {len(cv_winners) if len(cv_winners) > 0 else 0}")
print(f"  Above +2%: {len(above_2) if len(above_2) > 0 else 0}")
print(f"{'='*100}\n")
