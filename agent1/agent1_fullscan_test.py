# %% Agent 1 — Full Universe Single-Day Timing Test
# Scans ALL US equities via Polygon grouped daily, fetches real float data,
# applies gap-fade SHORT logic. Measures timing for each step.

import os, sys, time, re, asyncio
from datetime import datetime, timedelta

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

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

TEST_DATE = '2026-09-11'      # single day to test
TOP_N = 5
GAP_MIN_PCT = 5.0
PRICE_MIN = 10.0
FLOAT_MIN = 10_000_000

PEAK_WINDOW = 3
MAX_PEAK_BAR = 12
FALLBACK_BAR = 15

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY")
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')
ET = pytz.timezone('America/New_York')

_EXCLUDE_TK = {'SPY','QQQ','IWM','DIA','VXX','UVXY','SQQQ','TQQQ','SPXU','SPXL',
               'SDOW','UDOW','SDS','SSO','SH','LABU','LABD','SOXL','SOXS',
               'FNGU','FNGD','NUGT','DUST','JNUG','JDST','TNA','TZA',
               'UPRO','VIXY','SVXY','TVIX'}
_EXCLUDE_SUFFIX = ('W', 'WS', 'U', 'R')

_HOLIDAYS = {
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25',
    '2026-06-19','2026-07-03','2026-09-07','2026-11-26','2026-12-25',
}


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
            print(f"    [retry {attempt+1}/5] connection error, waiting {wait}s...")
            time.sleep(wait)
    r = requests.get(url, params=p, timeout=60)
    r.raise_for_status()
    return r.json()

def fetch_grouped_daily(dt_str):
    data = _pg(f"{_BASE}/v2/aggs/grouped/locale/us/market/stocks/{dt_str}",
               {"adjusted": "true", "include_otc": "false"})
    out = {}
    for r in data.get('results', []):
        tk = r.get('T', '')
        if not _TK_RE.match(tk): continue
        out[tk] = {'o': float(r.get('o', 0)), 'h': float(r.get('h', 0)),
                   'l': float(r.get('l', 0)), 'c': float(r.get('c', 0)),
                   'v': int(r.get('v', 0))}
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC FLOAT FETCH — Polygon ticker details
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_float_one(session, tk, sem):
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
                        return tk, None
                    data = await resp.json()
            except Exception:
                await asyncio.sleep(1)
                continue
            results = data.get('results', {})
            wso = results.get('weighted_shares_outstanding')
            sso = results.get('share_class_shares_outstanding')
            return tk, wso or sso
        return tk, None

async def _fetch_floats_batch(tickers):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_float_one(session, tk, sem) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: val for tk, val in results}

def fetch_floats(tickers):
    if not tickers: return {}
    return asyncio.run(_fetch_floats_batch(tickers))


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC MINUTE BAR FETCH
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_bars_one(session, tk, dt, sem):
    url = f"{_BASE}/v2/aggs/ticker/{tk}/range/1/minute/{dt}/{dt}"
    params = {"adjusted": "true", "sort": "asc", "limit": "500", "apiKey": POLYGON_API_KEY}
    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    if resp.status != 200:
                        return tk, None
                    data = await resp.json()
            except Exception:
                await asyncio.sleep(1)
                continue
            bars = data.get('results', [])
            if not bars: return tk, None
            rth = []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                if '09:30' <= t_str < '16:00':
                    rth.append((t_str, {
                        'o': float(b['o']), 'h': float(b['h']),
                        'l': float(b['l']), 'c': float(b['c']),
                        'v': int(b['v'])
                    }))
            return tk, rth
        return tk, None

async def _fetch_bars_batch(tickers, dt):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_bars_one(session, tk, dt, sem) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: val for tk, val in results if val is not None}

def fetch_minute_bars(tickers, dt):
    if not tickers: return {}
    return asyncio.run(_fetch_bars_batch(tickers, dt))


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL HIGH DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

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
                    'peak_price': bars[i][1]['h'],
                }
    if fallback_bar is not None and fallback_bar < len(bars):
        return {
            'entry_price': bars[fallback_bar][1]['c'],
            'entry_time': bars[fallback_bar][0],
            'entry_type': 'fallback',
        }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE DAY FULL SCAN
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_day(dt_str=TEST_DATE):
    timings = {}
    t0 = time.time()

    print(f"\n{'='*80}")
    print(f"  AGENT 1 — FULL UNIVERSE SINGLE-DAY SCAN")
    print(f"  Date: {dt_str}  |  TOP_N: {TOP_N}")
    print(f"  Gap >= {GAP_MIN_PCT}%  |  Price >= ${PRICE_MIN}  |  Float >= {FLOAT_MIN/1e6:.0f}M")
    print(f"{'='*80}")

    # ── Step 1: Fetch grouped daily (today + prior) ──
    t1 = time.time()
    prior_str = _prior_td(dt_str).strftime('%Y-%m-%d')
    prior = fetch_grouped_daily(prior_str)
    today = fetch_grouped_daily(dt_str)
    timings['1_grouped_daily'] = time.time() - t1
    print(f"\n  Step 1: Grouped daily ({prior_str} + {dt_str})")
    print(f"    Prior: {len(prior)} tickers  |  Today: {len(today)} tickers")
    print(f"    Time: {timings['1_grouped_daily']:.1f}s")

    # ── Step 2: Raw gap filter (no float check yet) ──
    t2 = time.time()
    gap_min = GAP_MIN_PCT / 100.0
    raw_gaps = []
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
        raw_gaps.append({
            'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
            'open_price': cur_p, 'volume': today[tk]['v'],
        })
    raw_gaps.sort(key=lambda x: -x['gap_pct'])
    timings['2_gap_filter'] = time.time() - t2
    print(f"\n  Step 2: Gap filter (>= {GAP_MIN_PCT}%, price >= ${PRICE_MIN})")
    print(f"    Raw gap-ups found: {len(raw_gaps)}")
    print(f"    Time: {timings['2_gap_filter']:.3f}s")

    if raw_gaps:
        print(f"\n    Top 20 raw (before float filter):")
        rows = []
        for g in raw_gaps[:20]:
            rows.append([g['ticker'], f"+{g['gap_pct']*100:.1f}%",
                         f"${g['open_price']:.2f}", f"{g['volume']:,}"])
        print(tabulate(rows, headers=['Ticker', 'Gap', 'Open', 'Volume'],
                       tablefmt='simple', stralign='right'))

    # ── Step 3: Fetch real float data for all candidates ──
    t3 = time.time()
    tickers_to_check = [g['ticker'] for g in raw_gaps]
    float_data = fetch_floats(tickers_to_check)
    timings['3_float_fetch'] = time.time() - t3

    n_has_float = sum(1 for v in float_data.values() if v is not None)
    print(f"\n  Step 3: Float data fetch ({len(tickers_to_check)} tickers)")
    print(f"    Got float for: {n_has_float}/{len(tickers_to_check)}")
    print(f"    Time: {timings['3_float_fetch']:.1f}s")

    # ── Step 4: Apply float filter ──
    t4 = time.time()
    candidates = []
    n_no_float = 0
    n_low_float = 0
    for g in raw_gaps:
        tk = g['ticker']
        flt = float_data.get(tk)
        if flt is None:
            n_no_float += 1
            continue
        if flt < FLOAT_MIN:
            n_low_float += 1
            continue
        g['float'] = flt
        g['tier'] = 'T4' if g['gap_pct'] <= 0.08 else 'T5'
        candidates.append(g)
    timings['4_float_filter'] = time.time() - t4
    print(f"\n  Step 4: Float filter (>= {FLOAT_MIN/1e6:.0f}M shares)")
    print(f"    Passed: {len(candidates)}  |  No float data: {n_no_float}  |  Low float: {n_low_float}")
    print(f"    Time: {timings['4_float_filter']:.3f}s")

    # ── Step 5: Select top N ──
    selected = candidates[:TOP_N]
    for i, s in enumerate(selected):
        s['rank'] = i + 1
    print(f"\n  Step 5: Top {TOP_N} selected:")
    if selected:
        rows = []
        for s in selected:
            rows.append([f"#{s['rank']}", s['ticker'], f"+{s['gap_pct']*100:.1f}%",
                         s['tier'], f"${s['open_price']:.2f}",
                         f"{s['float']/1e6:.1f}M", f"{s['volume']:,}"])
        print(tabulate(rows, headers=['Rank', 'Ticker', 'Gap', 'Tier', 'Open', 'Float', 'Volume'],
                       tablefmt='simple'))

    # ── Step 6: Fetch minute bars ──
    t6 = time.time()
    sel_tickers = [s['ticker'] for s in selected]
    bars = fetch_minute_bars(sel_tickers, dt_str)
    timings['6_minute_bars'] = time.time() - t6
    print(f"\n  Step 6: Minute bars for {len(sel_tickers)} tickers")
    print(f"    Got bars: {len(bars)}/{len(sel_tickers)}")
    print(f"    Time: {timings['6_minute_bars']:.1f}s")

    # ── Step 7: Find local high entries ──
    t7 = time.time()
    entries = {}
    for tk in sel_tickers:
        if tk in bars and bars[tk]:
            entries[tk] = find_local_high(bars[tk])
    timings['7_local_high'] = time.time() - t7

    print(f"\n  Step 7: Local high detection")
    for s in selected:
        tk = s['ticker']
        e = entries.get(tk)
        if e:
            print(f"    {tk}: {e['entry_type']} at {e['entry_time']} → ${e['entry_price']:.2f}")
        else:
            print(f"    {tk}: NO ENTRY")
    print(f"    Time: {timings['7_local_high']:.3f}s")

    # ── Step 8: Fetch T+2 outcomes ──
    t8 = time.time()
    nxt = _next_n_td(dt_str, 2)
    t1_str = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
    t2_str = nxt[1].strftime('%Y-%m-%d') if len(nxt) >= 2 else None

    t0_daily = fetch_grouped_daily(dt_str)
    t1_daily = fetch_grouped_daily(t1_str) if t1_str else {}
    t2_daily = fetch_grouped_daily(t2_str) if t2_str else {}
    timings['8_outcomes'] = time.time() - t8

    print(f"\n  Step 8: T+2 outcomes (exit dates: {t1_str}, {t2_str})")
    results = []
    for s in selected:
        tk = s['ticker']
        e = entries.get(tk)
        if not e:
            results.append({**s, 'entry': None, 'pnl_eod': None, 'pnl_t1': None, 'pnl_t2': None})
            continue
        ep = e['entry_price']
        t0c = t0_daily.get(tk, {}).get('c')
        t1c = t1_daily.get(tk, {}).get('c')
        t2c = t2_daily.get(tk, {}).get('c')
        pnl_eod = round((ep - t0c) / ep * 100, 2) if t0c else None
        pnl_t1 = round((ep - t1c) / ep * 100, 2) if t1c else None
        pnl_t2 = round((ep - t2c) / ep * 100, 2) if t2c else None
        results.append({**s, 'entry': e, 'pnl_eod': pnl_eod, 'pnl_t1': pnl_t1, 'pnl_t2': pnl_t2})

    rows = []
    for r in results:
        e = r.get('entry')
        rows.append([
            f"#{r['rank']}", r['ticker'], f"+{r['gap_pct']*100:.1f}%",
            e['entry_type'][:3] if e else 'SKIP',
            f"${e['entry_price']:.2f}" if e else '-',
            e['entry_time'] if e else '-',
            f"{r['pnl_eod']:+.2f}%" if r['pnl_eod'] is not None else '-',
            f"{r['pnl_t1']:+.2f}%" if r['pnl_t1'] is not None else '-',
            f"{r['pnl_t2']:+.2f}%" if r['pnl_t2'] is not None else '-',
        ])
    print(tabulate(rows, headers=['Rank','Ticker','Gap','Ent','Entry$','Time',
                                  'EOD','T+1','T+2'], tablefmt='simple'))
    print(f"    Time: {timings['8_outcomes']:.1f}s")

    # ── TIMING SUMMARY ──
    total = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  TIMING SUMMARY — {dt_str}")
    print(f"{'='*80}")
    timing_rows = []
    for step, dur in sorted(timings.items()):
        pct = dur / total * 100
        bar = '#' * int(pct / 2)
        timing_rows.append([step, f"{dur:.1f}s", f"{pct:.0f}%", bar])
    print(tabulate(timing_rows, headers=['Step', 'Time', '%', ''], tablefmt='simple'))
    print(f"\n  TOTAL: {total:.1f}s")
    print(f"\n  Breakdown:")
    print(f"    API calls (grouped daily):  2 + {2 if t2_str else 1} outcome = ~{2 + (2 if t2_str else 1)} calls")
    print(f"    API calls (float):          {len(tickers_to_check)} async calls")
    print(f"    API calls (minute bars):    {len(sel_tickers)} async calls")
    print(f"    Total API calls:            ~{2 + (2 if t2_str else 1) + len(tickers_to_check) + len(sel_tickers)}")

    # Estimate for full backtest
    est_bt = total * 174
    print(f"\n  Estimated full backtest (174 days): ~{est_bt/60:.0f} min")
    print(f"  (Float fetch is the bottleneck — {timings.get('3_float_fetch',0):.1f}s × 174 = "
          f"~{timings.get('3_float_fetch',0)*174/60:.0f} min)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

run_single_day()
