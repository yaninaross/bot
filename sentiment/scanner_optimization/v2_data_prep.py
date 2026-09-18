# %% V2 Data Prep — fetch all data, compute features, save to CSV/JSON
# Run this FIRST, then run experiment scripts in parallel.

import os, sys, json, warnings, re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings('ignore')

try:
    from google.cloud import bigquery
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-bigquery", "--quiet"])
    from google.cloud import bigquery
try:
    from polygon import RESTClient as PolygonClient
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "polygon-api-client", "--quiet"])
    from polygon import RESTClient as PolygonClient
try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass

ALL_DATES = ['2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11']
FOLD_A = ['2026-09-08', '2026-09-09']
FOLD_B = ['2026-09-10', '2026-09-11']
POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
BQ_PREMARKET_TABLE = f"{GCP_PROJECT}.scoring_logs.premarket_scanner"
BQ_DISCOVERY_TABLE = f"{GCP_PROJECT}.scoring_logs.discovery_scanner"
INTRADAY_MINUTES = 390
MARKET_OPEN_ET = "09:30"
MARKET_CLOSE_ET = "16:00"
ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')
HORIZON_FORWARD_DAYS = 10

try:
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    OUTPUT_DIR = os.getcwd()
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

_W = 100
print(f"\n{'='*_W}")
print(f"  V2 DATA PREP — Fetch + Feature Computation")
print(f"  Dates: {', '.join(ALL_DATES)}")
print(f"{'='*_W}")

# ── BQ FETCH ──────────────────────────────────────────────────────────────────
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
print(f"    Premarket: {len(df_pre_all)}, Discovery: {len(df_disc_all)}")

df_raw = pd.concat([df_pre_all, df_disc_all], ignore_index=True)
df_raw['gate_reason'] = df_raw['gate_reason'].fillna('')
df_raw.loc[df_raw['gate_reason'].isin(['None', 'none', 'nan']), 'gate_reason'] = ''
df_raw['eval_date'] = df_raw['eval_date'].astype(str)
df_raw['catalyst_type'] = df_raw['catalyst_type'].fillna('unknown')
df_raw['catalyst'] = df_raw['catalyst'].fillna('')

_stability_raw = (df_raw[df_raw['direction'].isin(['LONG', 'SHORT'])]
                  .groupby(['ticker', 'eval_date', 'direction', 'scanner_version'])
                  .size().reset_index(name='count'))

df_dir = df_raw[df_raw['direction'].isin(['LONG', 'SHORT'])].copy()

def _parse_scan_time(val):
    if pd.isna(val): return None
    s = str(val).strip()
    if len(s) == 5 and ':' in s: return s
    if len(s) >= 5 and s[2] == ':': return s[:5]
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try: return datetime.strptime(s[:19], fmt).strftime('%H:%M')
        except: continue
    return s[:5] if len(s) >= 5 else None

def _scan_pt_to_entry_et(scan_pt_hhmm, eval_date):
    pt_dt = PT.localize(datetime.strptime(f"{eval_date} {scan_pt_hhmm}", '%Y-%m-%d %H:%M'))
    et_dt = pt_dt.astimezone(ET)
    et_hhmm = et_dt.strftime('%H:%M')
    if et_hhmm < MARKET_OPEN_ET: return MARKET_OPEN_ET
    if et_hhmm >= MARKET_CLOSE_ET: return None
    return et_hhmm

df_dir['scan_time_hhmm'] = df_dir['scan_time_pt'].apply(_parse_scan_time)
df_dir = df_dir[df_dir['scan_time_hhmm'].notna()].copy()
df_dir['entry_time_et'] = df_dir.apply(
    lambda r: _scan_pt_to_entry_et(r['scan_time_hhmm'], r['eval_date']), axis=1)
df_dir = df_dir[df_dir['entry_time_et'].notna()].copy()
df_dir = (df_dir.sort_values('scan_time_hhmm')
          .drop_duplicates(['ticker', 'eval_date'], keep='first')
          .reset_index(drop=True))
print(f"  Directional signals after dedup: {len(df_dir)}")

# ── POLYGON MINUTE + DAILY BARS ──────────────────────────────────────────────
poly = PolygonClient(POLYGON_API_KEY)

def _fetch_minute_bars(ticker, eval_date):
    try:
        bars = list(poly.get_aggs(ticker=ticker, multiplier=1, timespan="minute",
                                   from_=eval_date, to=eval_date, limit=500))
        if not bars: return None
        bar_map = {}
        for b in bars:
            ts = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').tz_convert(ET)
            bar_map[ts.strftime('%H:%M')] = {
                'open': float(b.open), 'high': float(b.high),
                'low': float(b.low), 'close': float(b.close),
                'volume': int(b.volume)}
        return bar_map
    except: return None

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
    except: return {}

ticker_date_pairs = df_dir[['ticker', 'eval_date']].drop_duplicates().values.tolist()
all_tickers = sorted(set(t for t, _ in ticker_date_pairs))
spy_dates = sorted(set(d for _, d in ticker_date_pairs))

print(f"\n  Fetching minute bars for {len(ticker_date_pairs)} signals...")
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
    if bm: spy_minute[dt] = bm
print(f"  Got minute bars for {n_ok}/{len(ticker_date_pairs)} signals")

_from_daily = (datetime.strptime(ALL_DATES[0], '%Y-%m-%d') - timedelta(days=35)).strftime('%Y-%m-%d')
_to_daily = (datetime.strptime(ALL_DATES[-1], '%Y-%m-%d') + timedelta(days=HORIZON_FORWARD_DAYS + 5)).strftime('%Y-%m-%d')

print(f"\n  Fetching daily bars {_from_daily} to {_to_daily} for {len(all_tickers)} tickers + SPY...")
daily_data = {}
for i, tk in enumerate(all_tickers):
    db = _fetch_daily_bars(tk, _from_daily, _to_daily)
    if db: daily_data[tk] = db
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(all_tickers)}")

spy_daily = _fetch_daily_bars('SPY', _from_daily, _to_daily)
print(f"  Got daily bars for {len(daily_data)}/{len(all_tickers)} tickers")

# ── FEATURE COMPUTATION ──────────────────────────────────────────────────────
print(f"\n  Computing features...")
records = []

for idx, row in df_dir.iterrows():
    tk = row['ticker']
    dirn = row['direction']
    entry_hhmm = row['entry_time_et']
    dt = row['eval_date']
    bm = bar_data.get((tk, dt))
    if bm is None: continue

    actual_entry = entry_hhmm
    entry_bar = bm.get(entry_hhmm)
    if not entry_bar:
        cands = sorted(k for k in bm if k >= entry_hhmm and k < MARKET_CLOSE_ET)
        if not cands: continue
        actual_entry = cands[0]
        entry_bar = bm[actual_entry]

    ep = entry_bar['open']
    if ep <= 0: continue
    eh, em = map(int, actual_entry.split(':'))
    sign = 1.0 if dirn == 'LONG' else -1.0

    tk_daily = daily_data.get(tk, {})
    trading_days = sorted(tk_daily.keys())
    prev_days = [d for d in trading_days if d < dt]
    fwd_days = [d for d in trading_days if d > dt]
    prev_close = tk_daily[prev_days[-1]]['close'] if prev_days else None
    gap_pct = ((ep - prev_close) / prev_close * 100) if prev_close else 0.0

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

    avg_daily_vol = np.mean([tk_daily[d]['volume'] for d in prev_days[-20:]]) if prev_days else 0
    pm_bars = {k: v for k, v in bm.items() if k < MARKET_OPEN_ET}
    pm_vol = sum(v['volume'] for v in pm_bars.values())
    log_vol_ratio = np.log1p(pm_vol / avg_daily_vol) if avg_daily_vol > 0 else 0.0

    # PM shape
    pm_high, pm_low, pm_close_price, pm_high_time = 0, 1e9, ep, '09:30'
    pm_vol_last30 = 0
    for k in sorted(pm_bars.keys()):
        b = pm_bars[k]
        if b['high'] > pm_high: pm_high = b['high']; pm_high_time = k
        pm_low = min(pm_low, b['low'])
        pm_close_price = b['close']
        if k >= '09:00': pm_vol_last30 += b['volume']
    if pm_low > 1e8: pm_low = ep
    pm_range = pm_high - pm_low
    pm_retrace = (pm_high - pm_close_price) / pm_range if pm_range > 0.01 else 0.0
    pm_high_h, pm_high_m = map(int, pm_high_time.split(':'))
    pm_time_since_high = (9 * 60 + 30) - (pm_high_h * 60 + pm_high_m)
    pm_vol_late_ratio = pm_vol_last30 / pm_vol if pm_vol > 0 else 0.0

    first5_ret, first5_range = 0.0, 0.0
    for m in range(1, 6):
        tot = eh * 60 + em + m
        target = f"{tot // 60:02d}:{tot % 60:02d}"
        b = bm.get(target)
        if b:
            first5_ret = ((b['close'] - ep) / ep) * 100
            first5_range = max(first5_range, ((b['high'] - b['low']) / ep) * 100)
    first5_confirms = (first5_ret > 0 and dirn == 'LONG') or (first5_ret < 0 and dirn == 'SHORT')

    t5_tot = eh * 60 + em + 5
    t5_target = f"{t5_tot // 60:02d}:{t5_tot % 60:02d}"
    t5_bar = bm.get(t5_target)
    entry_price_5m = t5_bar['close'] if t5_bar else None

    # PnL curve no SL
    pnl_curve = {}
    mfe, mae = 0.0, 0.0
    mfe_time, mae_time = 0, 0
    for m in range(INTRADAY_MINUTES + 1):
        tot = eh * 60 + em + m
        target = f"{tot // 60:02d}:{tot % 60:02d}"
        if target >= MARKET_CLOSE_ET: break
        b = bm.get(target)
        if b is None: continue
        ret_close = ((b['close'] - ep) / ep) * 100 * sign
        pnl_curve[m] = ret_close
        if dirn == 'LONG':
            fav = ((b['high'] - ep) / ep) * 100
            adv = ((b['low'] - ep) / ep) * 100
        else:
            fav = -((b['low'] - ep) / ep) * 100
            adv = -((b['high'] - ep) / ep) * 100
        if fav > mfe: mfe = fav; mfe_time = m
        if adv < mae: mae = adv; mae_time = m

    if not pnl_curve or max(pnl_curve.keys()) < 30: continue
    max_m = max(pnl_curve.keys())
    pnl_60m_nosl = pnl_curve.get(60, np.nan)
    pnl_30m_nosl = pnl_curve.get(30, np.nan)
    pnl_120m_nosl = pnl_curve.get(120, np.nan) if 120 <= max_m else np.nan
    pnl_eod_nosl = pnl_curve.get(max_m, np.nan)

    def _sim_sl(entry_p, start_h, start_m, sl_pct, bar_map_ref):
        stopped = False
        pnl_at = {}
        for m in range(INTRADAY_MINUTES + 1):
            tot = start_h * 60 + start_m + m
            target = f"{tot // 60:02d}:{tot % 60:02d}"
            if target >= MARKET_CLOSE_ET: break
            b = bar_map_ref.get(target)
            if b is None: continue
            if stopped: pnl_at[m] = -sl_pct; continue
            ret = ((b['close'] - entry_p) / entry_p) * 100 * sign
            if dirn == 'LONG':
                worst = ((b['low'] - entry_p) / entry_p) * 100
            else:
                worst = -((b['high'] - entry_p) / entry_p) * 100
            if worst <= -sl_pct: stopped = True; pnl_at[m] = -sl_pct
            else: pnl_at[m] = round(ret, 4)
        return pnl_at, stopped

    sl_results = {}
    for sl in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        pa, st = _sim_sl(ep, eh, em, sl, bm)
        sl_results[f'pnl_60m_sl{sl}'] = pa.get(60, np.nan) if 60 <= (max(pa.keys()) if pa else 0) else np.nan
        sl_results[f'pnl_30m_sl{sl}'] = pa.get(30, np.nan) if 30 <= (max(pa.keys()) if pa else 0) else np.nan
        sl_results[f'stopped_sl{sl}'] = st

    for k in [0.25, 0.5, 0.75, 1.0]:
        atr_sl = atr_pct * k
        pa, st = _sim_sl(ep, eh, em, atr_sl, bm)
        sl_results[f'pnl_60m_atr{k}'] = pa.get(60, np.nan) if 60 <= (max(pa.keys()) if pa else 0) else np.nan
        sl_results[f'pnl_30m_atr{k}'] = pa.get(30, np.nan) if 30 <= (max(pa.keys()) if pa else 0) else np.nan
        sl_results[f'stopped_atr{k}'] = st
        sl_results[f'atr_sl_pct_{k}'] = atr_sl

    if entry_price_5m and entry_price_5m > 0:
        t5h, t5m_val = t5_tot // 60, t5_tot % 60
        for sl in [2.5, 4.0, 5.0]:
            pa5, st5 = _sim_sl(entry_price_5m, t5h, t5m_val, sl, bm)
            sl_results[f'pnl_60m_t5_sl{sl}'] = pa5.get(60, np.nan) if 60 <= (max(pa5.keys()) if pa5 else 0) else np.nan
        for k in [0.5, 0.75]:
            atr_sl = atr_pct * k
            pa5, st5 = _sim_sl(entry_price_5m, t5h, t5m_val, atr_sl, bm)
            sl_results[f'pnl_60m_t5_atr{k}'] = pa5.get(60, np.nan) if 60 <= (max(pa5.keys()) if pa5 else 0) else np.nan

    # ORB
    or_high, or_low = -1e9, 1e9
    for m in range(5):
        tot = 9 * 60 + 30 + m
        target = f"{tot // 60:02d}:{tot % 60:02d}"
        b = bm.get(target)
        if b: or_high = max(or_high, b['high']); or_low = min(or_low, b['low'])
    if or_high < -1e8: or_high = ep
    if or_low > 1e8: or_low = ep

    orb_dir, orb_entry_price, orb_entry_time, orb_sl = None, None, None, None
    for m in range(5, INTRADAY_MINUTES):
        tot = 9 * 60 + 30 + m
        target = f"{tot // 60:02d}:{tot % 60:02d}"
        if target >= MARKET_CLOSE_ET: break
        b = bm.get(target)
        if b is None: continue
        if b['high'] > or_high:
            orb_dir = 'LONG'; orb_entry_price = or_high; orb_entry_time = m; orb_sl = or_high - or_low; break
        if b['low'] < or_low:
            orb_dir = 'SHORT'; orb_entry_price = or_low; orb_entry_time = m; orb_sl = or_high - or_low; break

    orb_pnl_60m, orb_stopped = np.nan, False
    if orb_dir and orb_entry_price and orb_entry_price > 0:
        orb_sign = 1.0 if orb_dir == 'LONG' else -1.0
        orb_sl_pct = (orb_sl / orb_entry_price * 100) if orb_sl else 5.0
        for m2 in range(orb_entry_time, min(orb_entry_time + 60, INTRADAY_MINUTES)):
            tot2 = 9 * 60 + 30 + m2
            target2 = f"{tot2 // 60:02d}:{tot2 % 60:02d}"
            if target2 >= MARKET_CLOSE_ET: break
            b2 = bm.get(target2)
            if b2 is None: continue
            worst2 = ((b2['low'] - orb_entry_price) / orb_entry_price) * 100 if orb_dir == 'LONG' else -((b2['high'] - orb_entry_price) / orb_entry_price) * 100
            if worst2 <= -orb_sl_pct: orb_stopped = True; break
            if m2 == min(orb_entry_time + 60 - 1, INTRADAY_MINUTES - 1):
                orb_pnl_60m = ((b2['close'] - orb_entry_price) / orb_entry_price) * 100 * orb_sign
        if orb_stopped: orb_pnl_60m = -orb_sl_pct

    # Multi-day horizon
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

    # Stability
    stab = _stability_raw[((_stability_raw['ticker'] == tk) & (_stability_raw['eval_date'] == dt))]
    n_runs = int(stab['count'].sum()) if len(stab) > 0 else 1
    n_long_runs = int(stab[stab['direction'] == 'LONG']['count'].sum()) if len(stab) > 0 else 0
    n_short_runs = int(stab[stab['direction'] == 'SHORT']['count'].sum()) if len(stab) > 0 else 0
    direction_flipped = (n_long_runs > 0 and n_short_runs > 0)
    stability_score = max(n_long_runs, n_short_runs) / max(n_runs, 1)

    # Catalyst / earnings parsing
    cat_text = str(row.get('catalyst', '')).lower()
    is_earnings = any(w in cat_text for w in ['earnings', 'eps', 'revenue', 'quarter', 'q1', 'q2', 'q3', 'q4', 'fiscal'])
    cat_beat = any(w in cat_text for w in ['beat', 'exceeded', 'surpass', 'topped', 'above estimates'])
    cat_miss = any(w in cat_text for w in ['miss', 'below', 'disappoint', 'short of', 'fell short'])
    cat_guidance_up = any(w in cat_text for w in ['raised', 'upward', 'increased outlook', 'above consensus'])
    cat_guidance_down = any(w in cat_text for w in ['lowered', 'cut guidance', 'warned', 'reduced outlook'])
    earnings_rule_dir = None
    if is_earnings:
        if cat_beat and not cat_miss: earnings_rule_dir = 'LONG'
        elif cat_miss and not cat_beat: earnings_rule_dir = 'SHORT'
        if cat_guidance_up: earnings_rule_dir = 'LONG'
        elif cat_guidance_down: earnings_rule_dir = 'SHORT'

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

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv(f'{RESULTS_DIR}/v2_features_full.csv', index=False)

with open(f'{RESULTS_DIR}/v2_daily_data.json', 'w') as f:
    json.dump(daily_data, f)
with open(f'{RESULTS_DIR}/v2_spy_daily.json', 'w') as f:
    json.dump(spy_daily, f)

print(f"\n  Saved: v2_features_full.csv ({len(df)} rows), v2_daily_data.json ({len(daily_data)} tickers)")
print(f"  Data prep COMPLETE — run experiment scripts now.")
