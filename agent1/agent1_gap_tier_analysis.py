# %% Agent 1 — Gap Tier Analysis: Distribution & Performance by Period
#
# Run AFTER backtest_e2_reproduction.py in same Colab session to reuse cached data.
# If run standalone, pipeline executes automatically (~10 min).
#
# Question: Did gap size distribution shift in Jul-Sep 2026, and which tier
# drove the performance change?

import os, sys, time, json, re, asyncio, copy
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz
import requests

try:
    import aiohttp
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "--quiet", "--break-system-packages"])
    import aiohttp

try:
    import nest_asyncio
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "nest_asyncio", "--quiet", "--break-system-packages"])
    import nest_asyncio
nest_asyncio.apply()

try:
    from tabulate import tabulate
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate", "--quiet", "--break-system-packages"])
    from tabulate import tabulate

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — E2 BENCHMARK (DO NOT CHANGE)
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_START = '2026-01-02'
BACKTEST_END   = '2026-09-11'

GAP_MIN_PCT    = 5.0
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
TOP_N          = 1
PRE_FILTER_N   = 30

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

EXIT_TIME      = '17:00'
SCORING_TIME_ET = '09:20'
PM_MIN_BARS    = 5

CAPITAL        = 100_000

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')
ET = pytz.timezone('America/New_York')

_EXCLUDE_TK = {'SPY','QQQ','IWM','DIA','VXX','UVXY','SQQQ','TQQQ','SPXU','SPXL',
               'SDOW','UDOW','SDS','SSO','SH','LABU','LABD','SOXL','SOXS',
               'FNGU','FNGD','NUGT','DUST','JNUG','JDST','TNA','TZA',
               'UPRO','VIXY','SVXY','TVIX'}
_EXCLUDE_SUFFIX = ('W', 'WS', 'U', 'R')

_HOLIDAYS = {
    '2025-01-01','2025-01-20','2025-02-17','2025-04-18','2025-05-26',
    '2025-06-19','2025-07-04','2025-09-01','2025-11-27','2025-12-25',
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25',
    '2026-06-19','2026-07-03','2026-09-07','2026-11-26','2026-12-25',
    '2027-01-01','2027-01-18','2027-02-15','2027-03-26','2027-05-31',
    '2027-06-18','2027-07-05','2027-09-06','2027-11-25','2027-12-24',
}

_FLOAT = {
    'AAPL':14594180000,'AARD':21884158,'ABM':58580923,'ACHR':770023800,'ACMR':64657388,
    'ACV':10396028,'ACVA':169807980,'ADBE':397500000,'AEHR':32620450,'AENT':50979630,
    'AEON':49882790,'AGCO':70031729,'AGRZ':22573405,'AHMA':10965000,'ALAB':173485104,
    'AMAT':793597443,'AMD':1632475042,'AME':229203002,'APA':350351510,'APLD':291469112,
    'ARBE':122649743,'ARM':1068078760,'ASAN':159441754,'ASO':62018371,'ASTS':299789305,
}

# NOTE: Full _FLOAT dict is loaded from prior cell or from the reference script.
# The subset above is for standalone fallback — the pipeline populates the rest.


# ═══════════════════════════════════════════════════════════════════════════════
# GAP TIER & PERIOD DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

GAP_TIERS = {
    'T4':  (0.05, 0.08),    # 5-8% gap
    'T5':  (0.08, 0.15),    # 8-15% gap
    'T6':  (0.15, 0.30),    # 15-30% gap
    'T7':  (0.30, 9.99),    # 30%+ gap
}

PERIODS = {
    'Q1': ('2026-01-01', '2026-03-31'),
    'Q2': ('2026-04-01', '2026-06-30'),
    'Q3': ('2026-07-01', '2026-09-30'),
}


def get_gap_tier(gap_pct):
    """Classify gap_pct (decimal) into tier label."""
    for tier, (lo, hi) in GAP_TIERS.items():
        if lo <= gap_pct < hi:
            return tier
    if gap_pct >= 0.30:
        return 'T7'
    return None  # below 5%


def get_period(date_str):
    """Classify date string into period label."""
    for period, (start, end) in PERIODS.items():
        if start <= date_str <= end:
            return period
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _is_td(d):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    return d.weekday() < 5 and d.strftime('%Y-%m-%d') not in _HOLIDAYS

def _prior_td(d):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    d -= timedelta(days=1)
    for _ in range(10):
        if _is_td(d): return d
        d -= timedelta(days=1)
    raise ValueError(f"No trading day before {d}")

def _next_n_td(d, n=2):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    out, d = [], d + timedelta(days=1)
    for _ in range(30):
        if _is_td(d):
            out.append(d)
            if len(out) >= n: return out
        d += timedelta(days=1)
    return out

def _pg(url, params=None):
    p = dict(params or {}); p['apiKey'] = POLYGON_API_KEY
    for attempt in range(5):
        try:
            r = requests.get(url, params=p, timeout=30)
            if r.status_code == 429:
                time.sleep(12 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            wait = 5 * (2 ** attempt)
            print(f"    [retry {attempt+1}/5] waiting {wait}s...")
            time.sleep(wait)
    r = requests.get(url, params=p, timeout=60)
    r.raise_for_status()
    return r.json()

try:
    _gd_cache
except NameError:
    _gd_cache = {}

def fetch_grouped_daily(dt_str):
    if dt_str in _gd_cache: return _gd_cache[dt_str]
    data = _pg(f"{_BASE}/v2/aggs/grouped/locale/us/market/stocks/{dt_str}",
               {"adjusted":"true","include_otc":"false"})
    out = {}
    for r in data.get('results', []):
        tk = r.get('T','')
        if not _TK_RE.match(tk): continue
        out[tk] = {'o':float(r.get('o',0)),'h':float(r.get('h',0)),
                   'l':float(r.get('l',0)),'c':float(r.get('c',0)),'v':int(r.get('v',0))}
    _gd_cache[dt_str] = out
    return out

async def _fetch_bars_one(session, tk, dt, sem):
    url = f"{_BASE}/v2/aggs/ticker/{tk}/range/1/minute/{dt}/{dt}"
    params = {"adjusted":"true","sort":"asc","limit":"1000","apiKey":POLYGON_API_KEY}
    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    if resp.status != 200:
                        return (tk, dt), None
                    data = await resp.json()
            except Exception:
                await asyncio.sleep(1)
                continue
            bars = data.get('results', [])
            if not bars:
                return (tk, dt), None
            pm, rth, ah = [], [], []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                bar_d = {
                    'o': float(b['o']), 'h': float(b['h']),
                    'l': float(b['l']), 'c': float(b['c']),
                    'v': int(b['v'])
                }
                if '04:00' <= t_str < '09:30':
                    pm.append((t_str, bar_d))
                elif '09:30' <= t_str < '16:00':
                    rth.append((t_str, bar_d))
                elif '16:00' <= t_str <= '20:00':
                    ah.append((t_str, bar_d))
            return (tk, dt), {'pm': pm, 'rth': rth, 'ah': ah}
        return (tk, dt), None

async def _fetch_bars_batch(pairs):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_bars_one(session, tk, dt, sem) for tk, dt in pairs]
        results = await asyncio.gather(*tasks)
    return {key: val for key, val in results if val is not None}

def fetch_minute_bars_batch(pairs):
    if not pairs: return {}
    return asyncio.run(_fetch_bars_batch(pairs))

try:
    _ticker_ref
except NameError:
    _ticker_ref = {}

async def _fetch_ref_one(session, tk, sem):
    url = f"{_BASE}/v3/reference/tickers/{tk}"
    params = {"apiKey": POLYGON_API_KEY}
    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    if resp.status != 200:
                        return tk, {}
                    data = await resp.json()
            except Exception:
                await asyncio.sleep(1)
                continue
            res = data.get('results', {})
            return tk, {
                'type': res.get('type'),
                'name': res.get('name', ''),
                'shares': res.get('share_class_shares_outstanding'),
            }
        return tk, {}

async def _fetch_refs_batch(tickers):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_ref_one(session, tk, sem) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: val for tk, val in results}

def fetch_ticker_refs(tickers):
    uncached = [tk for tk in tickers if tk not in _ticker_ref]
    if uncached:
        batch = asyncio.run(_fetch_refs_batch(uncached))
        _ticker_ref.update(batch)

def classify_open_pattern(rth_bars):
    if len(rth_bars) < 60:
        return 'no_data'
    open_price = rth_bars[0][1]['o']
    low_30m = min(b[1]['l'] for b in rth_bars[:30])
    close_60m = rth_bars[59][1]['c']
    drop_30m = (low_30m - open_price) / open_price if open_price > 0 else 0
    recovery_denom = abs(open_price - low_30m)
    recovery = (close_60m - low_30m) / recovery_denom if recovery_denom > 0 else 0
    if drop_30m < -0.03 and recovery > 0.5:
        return 'drop_then_recover'
    elif drop_30m < -0.03 and recovery <= 0.5:
        return 'sustained_drop'
    elif drop_30m >= -0.01:
        return 'flat_or_bounce'
    else:
        return 'mild_drop'

def find_local_high(bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    fallback_bar=FALLBACK_BAR):
    if len(bars) < window * 2 + 1:
        return None
    for i in range(window, min(max_peak_bar, len(bars) - window)):
        ch = bars[i][1]['h']
        if all(bars[i - j][1]['h'] < ch and bars[i + j][1]['h'] < ch
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(bars):
                return {
                    'entry_price': bars[ci][1]['c'],
                    'entry_time': bars[ci][0],
                    'entry_type': 'local_high',
                    'peak_bar': i,
                }
    if fallback_bar is not None and fallback_bar < len(bars):
        return {
            'entry_price': bars[fallback_bar][1]['c'],
            'entry_time': bars[fallback_bar][0],
            'entry_type': 'fallback',
            'peak_bar': None,
        }
    return None

def scan_day(dt_str):
    if not _is_td(dt_str): return []
    prior_str = _prior_td(dt_str).strftime('%Y-%m-%d')
    prior = fetch_grouped_daily(prior_str)
    today = fetch_grouped_daily(dt_str)
    gap_min = 0.03
    candidates = []
    for tk in today:
        if tk in _EXCLUDE_TK: continue
        if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
        if tk not in prior: continue
        prev_c = prior[tk]['c']
        if prev_c <= 0: continue
        cur_p = today[tk]['o']
        if cur_p <= 0 or cur_p < PRICE_MIN: continue
        gap = (cur_p - prev_c) / prev_c
        if gap < gap_min: continue
        fs = _FLOAT.get(tk)
        if fs is not None and fs < FLOAT_MIN: continue
        candidates.append({
            'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
            'open_price': cur_p, 'scan_date': dt_str,
            'volume_today': today[tk].get('v', 0),
        })
    candidates.sort(key=lambda x: -x['gap_pct'])
    nxt = _next_n_td(dt_str, 2)
    t1 = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
    t2 = nxt[1].strftime('%Y-%m-%d') if len(nxt) >= 2 else None
    for s in candidates:
        s['t1_date'] = t1
        s['t2_date'] = t2
    return candidates

def compute_metrics(pnl_list, label=''):
    if not pnl_list:
        return {'label':label,'n':0,'avg':0,'med':0,'std':0,'wr':0,
                'sharpe':0,'pf':0,'ci_lo':0,'ci_hi':0,'trimmed_mean':0,
                'max_loss':0,'max_gain':0}
    a = np.array(pnl_list)
    n = len(a); avg = float(np.mean(a)); med = float(np.median(a))
    std = float(np.std(a, ddof=1)) if n > 1 else 0
    wr = float((a > 0).mean())
    sharpe = avg / std if std > 0 else 0
    se = std / np.sqrt(n) if n > 0 else 0
    w = a[a > 0]; l = a[a < 0]
    pf = float(w.sum() / abs(l.sum())) if len(l) > 0 and l.sum() != 0 else float('inf')
    p5, p95 = np.percentile(a, 5), np.percentile(a, 95)
    trimmed = a[(a >= p5) & (a <= p95)]
    tm = float(np.mean(trimmed)) if len(trimmed) > 0 else avg
    return {'label':label,'n':n,'avg':avg,'med':med,'std':std,'wr':wr,
            'sharpe':sharpe,'pf':pf,'ci_lo':avg-1.96*se,'ci_hi':avg+1.96*se,
            'trimmed_mean':tm,'max_loss':float(a.min()),'max_gain':float(a.max())}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE (skips if already loaded from prior cell)
# ═══════════════════════════════════════════════════════════════════════════════

PIPELINE_CACHED = False
try:
    _ = all_bars; _ = pattern_cache; _ = entry_cache; _ = all_candidates; _ = dates
    PIPELINE_CACHED = True
    print("  Using cached pipeline data from prior cell")
except NameError:
    pass

if not PIPELINE_CACHED:
    t_total = time.time()
    print(f"\n{'='*80}")
    print(f"  GAP TIER ANALYSIS — Data Pipeline")
    print(f"  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"{'='*80}")

    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    print(f"\n  {len(dates)} trading days")

    print(f"\n  Phase 1: Scanning for gap-ups...")
    all_candidates = {}
    total_cands = 0
    for i, dt in enumerate(dates):
        cands = scan_day(dt)
        all_candidates[dt] = cands
        total_cands += len(cands)
        if (i+1) % 30 == 0 or i == len(dates) - 1:
            print(f"    [{i+1:>3}/{len(dates)}] {dt}  |  {total_cands} total  |  "
                  f"{time.time()-t_total:.0f}s")

    all_pairs = set()
    for dt, cands in all_candidates.items():
        for c in cands[:PRE_FILTER_N]:
            all_pairs.add((c['ticker'], dt))
    print(f"\n  Phase 2: Fetching 1-min bars for {len(all_pairs)} pairs...")
    pairs_list = list(all_pairs)
    all_bars = {}
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start + 100]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        done = min(c_start + 100, len(pairs_list))
        if done % 500 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    print(f"  -> Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

    all_tickers = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            all_tickers.add(c['ticker'])
    print(f"\n  Phase 3: Fetching reference for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    for c_start in range(0, len(tk_list), 200):
        chunk = tk_list[c_start:c_start + 200]
        fetch_ticker_refs(chunk)
        done = min(c_start + 200, len(tk_list))
        if done % 400 == 0 or done == len(tk_list):
            print(f"    {done}/{len(tk_list)}  |  {time.time()-t_total:.0f}s")

    pattern_cache = {}
    entry_cache = {}
    for key, bar_data in all_bars.items():
        rth_bars = bar_data.get('rth', []) if isinstance(bar_data, dict) else bar_data
        pattern_cache[key] = classify_open_pattern(rth_bars)
        if rth_bars and len(rth_bars) >= PEAK_WINDOW * 2 + 1:
            entry_cache[key] = find_local_high(rth_bars)
        else:
            entry_cache[key] = None

    outcome_dates = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            outcome_dates.add(c['scan_date'])
            if c.get('t1_date'): outcome_dates.add(c['t1_date'])
            if c.get('t2_date'): outcome_dates.add(c['t2_date'])
    for od in sorted(od for od in outcome_dates if od not in _gd_cache):
        try: fetch_grouped_daily(od)
        except: pass

    print(f"\n  Pipeline: {time.time()-t_total:.0f}s")


# Ensure ticker refs are populated
_ref_tickers_needed = set()
for _cands in all_candidates.values():
    for _c in _cands[:PRE_FILTER_N]:
        _ref_tickers_needed.add(_c['ticker'])
_ref_missing = [tk for tk in _ref_tickers_needed if tk not in _ticker_ref]
if _ref_missing:
    print(f"  Fetching ticker refs for {len(_ref_missing)} tickers...")
    for _cs in range(0, len(_ref_missing), 200):
        _chunk = _ref_missing[_cs:_cs + 200]
        fetch_ticker_refs(_chunk)
    print(f"  -> _ticker_ref has {len(_ticker_ref)} entries")
del _ref_tickers_needed, _ref_missing


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD E2 BASELINE TRADES (enriched with intraday data)
# ═══════════════════════════════════════════════════════════════════════════════

def build_e2_trades_enriched():
    trades = []
    for dt in dates:
        cands = all_candidates.get(dt, [])
        cs_qualified = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            ref = _ticker_ref.get(tk, {})
            if ref.get('type', 'UNK') != 'CS':
                continue
            if pattern_cache.get((tk, dt), 'no_data') != 'sustained_drop':
                continue
            cc = copy.deepcopy(c)
            cc['asset_type'] = 'CS'
            cc['open_pattern'] = 'sustained_drop'
            cs_qualified.append(cc)

        for c in cs_qualified[:TOP_N]:
            tk, dt_s = c['ticker'], c['scan_date']
            entry = entry_cache.get((tk, dt_s))
            if entry is None:
                continue

            bar_data = all_bars.get((tk, dt_s))
            if bar_data is None:
                continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = entry['entry_price']
            tc['entry_time'] = entry['entry_time']
            tc['entry_type'] = entry['entry_type']
            ep = tc['entry_price']

            rth_bars = bar_data.get('rth', [])
            ah_bars = bar_data.get('ah', [])
            session_bars = rth_bars + ah_bars

            # Final PnL at 17:00
            exit_price = None
            for t_str, bar in session_bars:
                if t_str > EXIT_TIME:
                    break
                if t_str >= tc['entry_time']:
                    exit_price = bar['c']
            if exit_price is None:
                daily = _gd_cache.get(dt_s, {}).get(tk, {})
                if daily.get('c'):
                    exit_price = daily['c']
            if exit_price is None:
                continue

            tc['pnl_eod'] = round((ep - exit_price) / ep * 100, 3)

            # MAX ADVERSE EXCURSION
            max_adverse_pnl = tc['pnl_eod']
            for t_str, bar in session_bars:
                if t_str > EXIT_TIME:
                    break
                if t_str >= tc['entry_time']:
                    bar_pnl = (ep - bar['h']) / ep * 100
                    if bar_pnl < max_adverse_pnl:
                        max_adverse_pnl = bar_pnl
            tc['max_adverse_pnl'] = round(max_adverse_pnl, 3)

            # OPENING DROP
            if len(rth_bars) > 15:
                open_p = rth_bars[0][1]['o']
                bar15_c = rth_bars[15][1]['c']
                tc['opening_drop'] = round((open_p - bar15_c) / open_p, 4) if open_p > 0 else 0
            else:
                tc['opening_drop'] = None

            # GAP CONSUMED
            gap_size = tc['open_price'] - tc['prior_close']
            if gap_size > 0 and tc['opening_drop'] is not None and len(rth_bars) > 15:
                bar15_c = rth_bars[15][1]['c']
                tc['gap_consumed'] = round((tc['open_price'] - bar15_c) / gap_size, 4)
            else:
                tc['gap_consumed'] = None

            trades.append(tc)

    return trades

print(f"\n{'='*80}")
print(f"  Building E2 baseline trades with intraday enrichment...")
print(f"{'='*80}")

try:
    _ = e2_trades
    if len(e2_trades) > 0 and 'pnl_eod' in e2_trades[0]:
        print(f"  Using cached e2_trades ({len(e2_trades)} trades)")
    else:
        raise NameError
except NameError:
    e2_trades = build_e2_trades_enriched()

raw_pnls = [t['pnl_eod'] for t in e2_trades]
base_m = compute_metrics(raw_pnls, 'E2_BASE')

print(f"  {len(e2_trades)} trades, avg {base_m['avg']:+.2f}%, "
      f"Sharpe {base_m['sharpe']:.3f}, WR {base_m['wr']*100:.0f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: PERIOD x TIER PERFORMANCE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  ANALYSIS 1: GAP TIER PERFORMANCE BY PERIOD")
print(f"  Tiers: T4 (5-8%), T5 (8-15%), T6 (15-30%), T7 (30%+)")
print(f"  Periods: Q1 (Jan-Mar), Q2 (Apr-Jun), Q3 (Jul-Sep)")
print(f"{'='*80}")

# Tag each trade with tier and period
for t in e2_trades:
    t['_tier'] = get_gap_tier(t['gap_pct'])
    t['_period'] = get_period(t['scan_date'])

# Build period x tier grid
period_tier_data = {}
for period in PERIODS:
    for tier in GAP_TIERS:
        subset = [t for t in e2_trades if t['_tier'] == tier and t['_period'] == period]
        pnls = [t['pnl_eod'] for t in subset]
        m = compute_metrics(pnls, f'{period}_{tier}')
        total_dol = sum(CAPITAL * p / 100 for p in pnls)
        period_tier_data[(period, tier)] = {
            'n': m['n'],
            'avg_pnl': m['avg'],
            'med_pnl': m['med'],
            'wr': m['wr'],
            'sharpe': m['sharpe'],
            'total_dol': total_dol,
            'max_gain': m['max_gain'],
            'max_loss': m['max_loss'],
            'pf': m['pf'],
        }


# ─── PRINT 3D TABLE ────────────────────────────────────────────────────────

for period in PERIODS:
    period_label = {'Q1': 'Q1 (Jan-Mar)', 'Q2': 'Q2 (Apr-Jun)', 'Q3': 'Q3 (Jul-Sep)'}
    print(f"\n  --- {period_label[period]} ---")
    rows = []
    for tier in GAP_TIERS:
        d = period_tier_data[(period, tier)]
        tier_range = {
            'T4': '5-8%', 'T5': '8-15%', 'T6': '15-30%', 'T7': '30%+'
        }
        rows.append([
            f"{tier} ({tier_range[tier]})",
            d['n'],
            f"{d['avg_pnl']:+.2f}%",
            f"{d['med_pnl']:+.2f}%",
            f"{d['wr']*100:.0f}%",
            f"{d['sharpe']:.3f}",
            f"${d['total_dol']:+,.0f}",
            f"{d['max_gain']:+.1f}%" if d['n'] > 0 else "---",
            f"{d['max_loss']:+.1f}%" if d['n'] > 0 else "---",
        ])
    # Period subtotal
    period_trades = [t for t in e2_trades if t['_period'] == period]
    period_pnls = [t['pnl_eod'] for t in period_trades]
    pm = compute_metrics(period_pnls, period)
    pt_dol = sum(CAPITAL * p / 100 for p in period_pnls)
    rows.append([
        'TOTAL',
        pm['n'],
        f"{pm['avg']:+.2f}%",
        f"{pm['med']:+.2f}%",
        f"{pm['wr']*100:.0f}%",
        f"{pm['sharpe']:.3f}",
        f"${pt_dol:+,.0f}",
        f"{pm['max_gain']:+.1f}%" if pm['n'] > 0 else "---",
        f"{pm['max_loss']:+.1f}%" if pm['n'] > 0 else "---",
    ])
    print(tabulate(rows,
                   headers=['Tier', 'N', 'Avg PnL', 'Med PnL', 'WR', 'Sharpe',
                            'Total $', 'Best', 'Worst'],
                   tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: GAP SIZE DISTRIBUTION BY PERIOD (OPPORTUNITY SET)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  ANALYSIS 2: GAP SIZE DISTRIBUTION — OPPORTUNITY SET")
print(f"  All stocks gapping >= 5% (pre-filter, not just CS+SD trades)")
print(f"{'='*80}")

# Count qualifying gap-ups (>= 5%) per day and per period
period_gap_data = defaultdict(list)       # period -> list of gap_pct values
period_daily_counts = defaultdict(list)   # period -> list of daily gapup counts
period_trading_days = defaultdict(int)    # period -> number of trading days

for dt in dates:
    period = get_period(dt)
    if period is None:
        continue
    period_trading_days[period] += 1
    cands = all_candidates.get(dt, [])
    day_gaps = [c['gap_pct'] for c in cands if c['gap_pct'] >= 0.05]
    period_daily_counts[period].append(len(day_gaps))
    period_gap_data[period].extend(day_gaps)

print(f"\n  {'Period':<18s}  {'Avg Gap':>8s}  {'Med Gap':>8s}  "
      f"{'GapUps/Day':>10s}  {'Total GapUps':>12s}  {'Trd Days':>9s}  "
      f"{'Trades':>6s}  {'Trd/Wk':>6s}")
print(f"  {'─'*18}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*9}  {'─'*6}  {'─'*6}")

gap_dist_table = {}
for period in ['Q1', 'Q2', 'Q3']:
    gaps = period_gap_data[period]
    daily_counts = period_daily_counts[period]
    n_days = period_trading_days[period]
    n_weeks = n_days / 5.0
    period_trades = [t for t in e2_trades if t['_period'] == period]
    n_trades = len(period_trades)
    trades_per_week = n_trades / n_weeks if n_weeks > 0 else 0

    avg_gap = np.mean(gaps) * 100 if gaps else 0
    med_gap = np.median(gaps) * 100 if gaps else 0
    avg_gapups = np.mean(daily_counts) if daily_counts else 0

    gap_dist_table[period] = {
        'avg_gap_pct': round(avg_gap, 2),
        'med_gap_pct': round(med_gap, 2),
        'avg_gapups_per_day': round(avg_gapups, 1),
        'total_gapups': len(gaps),
        'trading_days': n_days,
        'n_trades': n_trades,
        'trades_per_week': round(trades_per_week, 2),
    }

    label = {'Q1': 'Q1 (Jan-Mar)', 'Q2': 'Q2 (Apr-Jun)', 'Q3': 'Q3 (Jul-Sep)'}
    print(f"  {label[period]:<18s}  {avg_gap:>7.1f}%  {med_gap:>7.1f}%  "
          f"{avg_gapups:>10.1f}  {len(gaps):>12d}  {n_days:>9d}  "
          f"{n_trades:>6d}  {trades_per_week:>6.2f}")


# ─── TIER DISTRIBUTION IN OPPORTUNITY SET ──────────────────────────────────

print(f"\n\n  --- Gap Tier Distribution in Opportunity Set (>= 5% gap) ---\n")
print(f"  {'Period':<18s}  {'T4 (5-8%)':>12s}  {'T5 (8-15%)':>12s}  "
      f"{'T6 (15-30%)':>12s}  {'T7 (30%+)':>12s}")
print(f"  {'─'*18}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")

opp_set_tiers = {}
for period in ['Q1', 'Q2', 'Q3']:
    gaps = period_gap_data[period]
    total = len(gaps)
    tier_counts = defaultdict(int)
    for g in gaps:
        t = get_gap_tier(g)
        if t:
            tier_counts[t] += 1
    opp_set_tiers[period] = dict(tier_counts)
    label = {'Q1': 'Q1 (Jan-Mar)', 'Q2': 'Q2 (Apr-Jun)', 'Q3': 'Q3 (Jul-Sep)'}
    parts = []
    for tier in ['T4', 'T5', 'T6', 'T7']:
        c = tier_counts[tier]
        pct = c / total * 100 if total > 0 else 0
        parts.append(f"{c:>4d} ({pct:4.1f}%)")
    print(f"  {label[period]:<18s}  {parts[0]:>12s}  {parts[1]:>12s}  "
          f"{parts[2]:>12s}  {parts[3]:>12s}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: WHICH TIER DROVE Q3 DETERIORATION?
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  ANALYSIS 3: Q3 vs Q1+Q2 — TIER CONTRIBUTION TO DETERIORATION")
print(f"{'='*80}")

# Combine Q1+Q2 as baseline
q12_trades = [t for t in e2_trades if t['_period'] in ('Q1', 'Q2')]
q3_trades = [t for t in e2_trades if t['_period'] == 'Q3']

q12_m = compute_metrics([t['pnl_eod'] for t in q12_trades], 'Q1+Q2')
q3_m = compute_metrics([t['pnl_eod'] for t in q3_trades], 'Q3')

print(f"\n  Q1+Q2 combined: {q12_m['n']} trades, {q12_m['avg']:+.2f}% avg, "
      f"Sharpe {q12_m['sharpe']:.3f}")
print(f"  Q3 alone:       {q3_m['n']} trades, {q3_m['avg']:+.2f}% avg, "
      f"Sharpe {q3_m['sharpe']:.3f}")

if q12_m['n'] > 0 and q3_m['n'] > 0:
    delta_avg = q3_m['avg'] - q12_m['avg']
    print(f"  Delta:          {delta_avg:+.2f}% avg PnL change")

print(f"\n  {'Tier':<15s}  {'Q1+Q2 Avg':>10s}  {'Q1+Q2 N':>8s}  "
      f"{'Q3 Avg':>10s}  {'Q3 N':>5s}  {'Delta':>10s}  {'Q3 $ Impact':>12s}  {'Verdict':>12s}")
print(f"  {'─'*15}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*12}")

tier_contribution = {}
for tier in GAP_TIERS:
    q12_sub = [t['pnl_eod'] for t in q12_trades if t['_tier'] == tier]
    q3_sub = [t['pnl_eod'] for t in q3_trades if t['_tier'] == tier]
    m12 = compute_metrics(q12_sub, f'Q12_{tier}')
    m3 = compute_metrics(q3_sub, f'Q3_{tier}')
    delta = m3['avg'] - m12['avg'] if m12['n'] > 0 and m3['n'] > 0 else 0
    q3_dol = sum(CAPITAL * p / 100 for p in q3_sub)

    # Verdict
    if m3['n'] == 0:
        verdict = 'NO DATA'
    elif delta < -5:
        verdict = 'COLLAPSED'
    elif delta < -2:
        verdict = 'WEAKENED'
    elif abs(delta) <= 2:
        verdict = 'STABLE'
    elif delta > 2:
        verdict = 'IMPROVED'
    else:
        verdict = '---'

    tier_range = {'T4': '5-8%', 'T5': '8-15%', 'T6': '15-30%', 'T7': '30%+'}
    tier_contribution[tier] = {
        'q12_avg': m12['avg'], 'q12_n': m12['n'],
        'q3_avg': m3['avg'], 'q3_n': m3['n'],
        'delta': delta, 'q3_dol': q3_dol,
        'verdict': verdict,
    }

    print(f"  {tier+' ('+tier_range[tier]+')':15s}  "
          f"{m12['avg']:>+9.2f}%  {m12['n']:>8d}  "
          f"{m3['avg']:>+9.2f}%  {m3['n']:>5d}  "
          f"{delta:>+9.2f}%  ${q3_dol:>+11,.0f}  {verdict:>12s}")

# Identify the worst tier
worst_tier = min(tier_contribution, key=lambda t: tier_contribution[t]['delta']
                 if tier_contribution[t]['q3_n'] > 0 and tier_contribution[t]['q12_n'] > 0
                 else 999)
wt = tier_contribution[worst_tier]
print(f"\n  >>> Worst deterioration: {worst_tier} "
      f"(Q1+Q2: {wt['q12_avg']:+.2f}% -> Q3: {wt['q3_avg']:+.2f}%, "
      f"delta {wt['delta']:+.2f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: MONTHLY GAP-UP FREQUENCY
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  ANALYSIS 4: MONTHLY GAP-UP FREQUENCY & TRADE COUNT")
print(f"{'='*80}\n")

monthly_gaps = defaultdict(list)    # YYYY-MM -> list of gap_pct
monthly_trades = defaultdict(int)   # YYYY-MM -> trade count
monthly_days = defaultdict(int)     # YYYY-MM -> trading day count

for dt in dates:
    ym = dt[:7]
    monthly_days[ym] += 1
    cands = all_candidates.get(dt, [])
    for c in cands:
        if c['gap_pct'] >= 0.05:
            monthly_gaps[ym].append(c['gap_pct'])

for t in e2_trades:
    ym = t['scan_date'][:7]
    monthly_trades[ym] += 1

months_sorted = sorted(monthly_gaps.keys())

print(f"  {'Month':>8s}  {'GapUps':>7s}  {'Per Day':>8s}  {'Trades':>7s}  "
      f"{'Avg Gap':>8s}  {'Med Gap':>8s}  {'T4':>4s}  {'T5':>4s}  {'T6':>4s}  {'T7':>4s}")
print(f"  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*8}  "
      f"{'─'*4}  {'─'*4}  {'─'*4}  {'─'*4}")

monthly_data = {}
for ym in months_sorted:
    gaps = monthly_gaps[ym]
    n_days = monthly_days[ym]
    n_trd = monthly_trades.get(ym, 0)
    avg_g = np.mean(gaps) * 100 if gaps else 0
    med_g = np.median(gaps) * 100 if gaps else 0
    per_day = len(gaps) / n_days if n_days > 0 else 0

    tier_c = defaultdict(int)
    for g in gaps:
        t = get_gap_tier(g)
        if t: tier_c[t] += 1

    monthly_data[ym] = {
        'n_gapups': len(gaps), 'per_day': per_day, 'n_trades': n_trd,
        'avg_gap': avg_g, 'med_gap': med_g,
        'T4': tier_c['T4'], 'T5': tier_c['T5'],
        'T6': tier_c['T6'], 'T7': tier_c['T7'],
    }

    print(f"  {ym:>8s}  {len(gaps):>7d}  {per_day:>8.1f}  {n_trd:>7d}  "
          f"{avg_g:>7.1f}%  {med_g:>7.1f}%  "
          f"{tier_c['T4']:>4d}  {tier_c['T5']:>4d}  {tier_c['T6']:>4d}  {tier_c['T7']:>4d}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: TOP / BOTTOM Q3 TRADES BY TIER
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  ANALYSIS 5: Q3 TRADE DETAIL (all trades, sorted by PnL)")
print(f"{'='*80}\n")

q3_sorted = sorted(q3_trades, key=lambda t: t['pnl_eod'])

print(f"  {'Date':>12s}  {'Ticker':>7s}  {'Gap%':>7s}  {'Tier':>4s}  "
      f"{'Entry$':>8s}  {'Type':>10s}  {'PnL%':>8s}  {'$PnL':>10s}")
print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}")
for t in q3_sorted:
    dol = CAPITAL * t['pnl_eod'] / 100
    print(f"  {t['scan_date']:>12s}  {t['ticker']:>7s}  "
          f"{t['gap_pct']*100:>6.1f}%  {t.get('_tier','?'):>4s}  "
          f"${t['entry_price']:>7.2f}  {t['entry_type']:>10s}  "
          f"{t['pnl_eod']:>+7.2f}%  ${dol:>+9,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  CHARTS")
print(f"{'='*80}")

os.makedirs('results', exist_ok=True)

# ─── CHART 1: Avg PnL by Tier, Grouped by Period ──────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

tiers_list = list(GAP_TIERS.keys())
periods_list = list(PERIODS.keys())
n_tiers = len(tiers_list)
n_periods = len(periods_list)
bar_width = 0.22
x = np.arange(n_tiers)

colors = ['#2E86AB', '#A23B72', '#F18F01']
period_labels = ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)']

for i, period in enumerate(periods_list):
    vals = [period_tier_data[(period, tier)]['avg_pnl'] for tier in tiers_list]
    counts = [period_tier_data[(period, tier)]['n'] for tier in tiers_list]
    bars = ax.bar(x + i * bar_width, vals, bar_width, label=period_labels[i],
                  color=colors[i], edgecolor='white', linewidth=0.5)
    # Annotate n on each bar
    for j, (bar, n_val) in enumerate(zip(bars, counts)):
        if n_val > 0:
            y_pos = bar.get_height()
            va = 'bottom' if y_pos >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'n={n_val}', ha='center', va=va, fontsize=8, fontweight='bold')

ax.set_xlabel('Gap Tier', fontsize=12)
ax.set_ylabel('Avg PnL (%)', fontsize=12)
ax.set_title('Agent 1 SHORT: Avg PnL by Gap Tier & Period', fontsize=14, fontweight='bold')
ax.set_xticks(x + bar_width)
ax.set_xticklabels(['T4 (5-8%)', 'T5 (8-15%)', 'T6 (15-30%)', 'T7 (30%+)'])
ax.legend(loc='upper right')
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/chart1_pnl_by_tier_period.png', dpi=150, bbox_inches='tight')
print("  Saved: results/chart1_pnl_by_tier_period.png")
plt.show()


# ─── CHART 2: Gap Size Distribution Histograms ────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

tier_boundaries = [5, 8, 15, 30, 80]
hist_colors = ['#2E86AB', '#A23B72', '#F18F01']

for i, period in enumerate(periods_list):
    ax = axes[i]
    gaps_pct = [g * 100 for g in period_gap_data[period]]

    if gaps_pct:
        bins = np.arange(5, max(max(gaps_pct) + 2, 35), 2)
        ax.hist(gaps_pct, bins=bins, color=hist_colors[i], edgecolor='white',
                alpha=0.85, linewidth=0.5)

    # Mark tier boundaries
    for boundary in [8, 15, 30]:
        ax.axvline(x=boundary, color='red', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xlabel('Gap Size (%)', fontsize=11)
    if i == 0:
        ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{period_labels[i]}\n(n={len(gaps_pct)} gap-ups)', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add tier labels at top
    ax.text(6.5, ax.get_ylim()[1] * 0.92, 'T4', ha='center', fontsize=8,
            color='gray', fontstyle='italic')
    ax.text(11.5, ax.get_ylim()[1] * 0.92, 'T5', ha='center', fontsize=8,
            color='gray', fontstyle='italic')
    ax.text(22.5, ax.get_ylim()[1] * 0.92, 'T6', ha='center', fontsize=8,
            color='gray', fontstyle='italic')

fig.suptitle('Gap Size Distribution by Period (Opportunity Set, >= 5%)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/chart2_gap_distribution.png', dpi=150, bbox_inches='tight')
print("  Saved: results/chart2_gap_distribution.png")
plt.show()


# ─── CHART 3: Monthly Gap-Up Count & Trade Count ──────────────────────────

fig, ax1 = plt.subplots(figsize=(12, 5))

x_months = np.arange(len(months_sorted))
gapup_counts = [monthly_data[ym]['n_gapups'] for ym in months_sorted]
trade_counts = [monthly_data[ym]['n_trades'] for ym in months_sorted]

bar_w = 0.35
bars1 = ax1.bar(x_months - bar_w/2, gapup_counts, bar_w, label='Gap-ups >= 5%',
                color='#2E86AB', edgecolor='white', alpha=0.85)
bars2 = ax1.bar(x_months + bar_w/2, trade_counts, bar_w, label='E2 Trades (CS+SD)',
                color='#F18F01', edgecolor='white', alpha=0.85)

# Annotate counts
for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, h, str(int(h)),
                 ha='center', va='bottom', fontsize=8)
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, h, str(int(h)),
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Monthly Gap-Up Opportunities vs E2 Trades', fontsize=14, fontweight='bold')
ax1.set_xticks(x_months)
ax1.set_xticklabels(months_sorted, rotation=45, ha='right')
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Add Q boundary markers
q_starts = {'Q2': '2026-04', 'Q3': '2026-07'}
for qlabel, ym_start in q_starts.items():
    if ym_start in months_sorted:
        idx = months_sorted.index(ym_start)
        ax1.axvline(x=idx - 0.5, color='red', linewidth=1.2, linestyle='--', alpha=0.6)
        ax1.text(idx - 0.5, ax1.get_ylim()[1] * 0.95, f' {qlabel} start',
                 fontsize=9, color='red', alpha=0.7)

plt.tight_layout()
plt.savefig('results/chart3_monthly_gapup_frequency.png', dpi=150, bbox_inches='tight')
print("  Saved: results/chart3_monthly_gapup_frequency.png")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS JSON
# ═══════════════════════════════════════════════════════════════════════════════

results = {
    'analysis': 'gap_tier_period',
    'backtest_start': BACKTEST_START,
    'backtest_end': BACKTEST_END,
    'total_trades': len(e2_trades),
    'baseline': {
        'n': base_m['n'], 'avg_pnl': base_m['avg'], 'sharpe': base_m['sharpe'],
        'wr': base_m['wr'], 'med_pnl': base_m['med'],
    },
    'period_tier_matrix': {
        f'{p}_{t}': period_tier_data[(p, t)]
        for p in PERIODS for t in GAP_TIERS
    },
    'gap_distribution': gap_dist_table,
    'opportunity_set_tiers': opp_set_tiers,
    'tier_deterioration': tier_contribution,
    'worst_tier': worst_tier,
    'monthly_data': monthly_data,
    'q3_trades': [
        {
            'date': t['scan_date'], 'ticker': t['ticker'],
            'gap_pct': round(t['gap_pct'] * 100, 2), 'tier': t.get('_tier'),
            'pnl_eod': t['pnl_eod'], 'entry_type': t['entry_type'],
        }
        for t in q3_sorted
    ],
}

# Handle inf values for JSON serialization
def _json_safe(obj):
    if isinstance(obj, float):
        if obj == float('inf'): return "Inf"
        if obj == float('-inf'): return "-Inf"
        if obj != obj: return "NaN"  # NaN
    raise TypeError(f"Not serializable: {obj}")

with open('results/gap_tier_period_analysis.json', 'w') as f:
    json.dump(results, f, indent=2, default=_json_safe)
print(f"\n  Saved: results/gap_tier_period_analysis.json")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")
print(f"\n  E2 Baseline: {base_m['n']} trades, {base_m['avg']:+.2f}% avg, "
      f"Sharpe {base_m['sharpe']:.3f}")
print(f"  Q1+Q2: {q12_m['n']} trades, {q12_m['avg']:+.2f}% avg")
print(f"  Q3:    {q3_m['n']} trades, {q3_m['avg']:+.2f}% avg")

if q12_m['n'] > 0 and q3_m['n'] > 0:
    print(f"  Delta: {q3_m['avg'] - q12_m['avg']:+.2f}% avg PnL")

print(f"\n  Worst tier in Q3: {worst_tier} "
      f"(delta {tier_contribution[worst_tier]['delta']:+.2f}%)")

# Opportunity set trend
q1_opp = gap_dist_table.get('Q1', {}).get('avg_gapups_per_day', 0)
q3_opp = gap_dist_table.get('Q3', {}).get('avg_gapups_per_day', 0)
if q1_opp > 0:
    opp_chg = (q3_opp - q1_opp) / q1_opp * 100
    print(f"  Opportunity set: {q1_opp:.1f} -> {q3_opp:.1f} gap-ups/day "
          f"({opp_chg:+.0f}% change)")

print(f"\n  Charts saved to results/")
print(f"  JSON saved to results/gap_tier_period_analysis.json")
print(f"\n{'='*80}")
