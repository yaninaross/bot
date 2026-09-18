# %% Agent 1 — Raw Gap & Mover Scanner (All US Equities)
# Shows ALL tickers moving >= GAP_MIN% today, with type/sector from Polygon.
# Works pre-market AND during regular hours.
# No CS filter, no float filter, no retrace filter — raw universe view.
#
# Usage:
#   SCAN_DATE = None         → today (live snapshot)
#   SCAN_DATE = '2026-09-16' → historical (uses open price from grouped daily)
#   GAP_MIN = 0.05           → 5% minimum gap/change
#   PRICE_MIN = 10.0         → $10 minimum price
#   CS_FLOOR = 0.02          → 2% floor for CS-only movers section (context)

import os, sys, time, re, json

import numpy as np
import pandas as pd
import requests

try:
    import aiohttp, asyncio
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "--quiet", "--break-system-packages"])
    import aiohttp, asyncio

try:
    import nest_asyncio
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "nest_asyncio", "--quiet", "--break-system-packages"])
    import nest_asyncio

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

SCAN_DATE = None             # None = today (live snapshot), 'YYYY-MM-DD' = historical
GAP_MIN = 0.05               # 5% minimum gap/change
PRICE_MIN = 10.0             # $10 minimum price
SHOW_TOP = 50                # max rows in main table
CS_FLOOR = 0.02              # 2% floor for CS-only movers (shows context on rally days)

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
POLYGON_BASE = "https://api.polygon.io"
TICKER_REGEX = re.compile(r'^[A-Z]{1,5}$')

_EXCLUDED_SUFFIXES = ('W', 'WS', 'U', 'R')
_EXCLUDED_TICKERS = {'SPY', 'QQQ', 'IWM', 'DIA', 'VXX', 'UVXY', 'SQQQ', 'TQQQ',
                     'SPXU', 'SPXL', 'SDOW', 'UDOW', 'SDS', 'SSO', 'SH',
                     'LABU', 'LABD', 'SOXL', 'SOXS', 'FNGU', 'FNGD',
                     'NUGT', 'DUST', 'JNUG', 'JDST', 'TNA', 'TZA',
                     'UPRO', 'VIXY', 'SVXY', 'TVIX'}

_NYSE_HOLIDAYS = {
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
    '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
}

from datetime import date, datetime, timedelta
import pytz
ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')


# ═══════════════════════════════════════════════════════════════════════════════
# POLYGON HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _polygon_get(url, params=None):
    if params is None: params = {}
    params['apiKey'] = POLYGON_API_KEY
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 429:
        time.sleep(12)
        resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_grouped_daily(date_str):
    url = f"{POLYGON_BASE}/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
    data = _polygon_get(url, {"adjusted": "true", "include_otc": "false"})
    out = {}
    for r in data.get('results', []):
        tk = r.get('T', '')
        if not TICKER_REGEX.match(tk):
            continue
        out[tk] = {
            'open': float(r.get('o', 0)),
            'high': float(r.get('h', 0)),
            'low': float(r.get('l', 0)),
            'close': float(r.get('c', 0)),
            'volume': int(r.get('v', 0)),
        }
    return out


def fetch_snapshot_full():
    """Fetch full snapshot with all fields — not just price."""
    url = f"{POLYGON_BASE}/v2/snapshot/locale/us/markets/stocks/tickers"
    data = _polygon_get(url, {"include_otc": "false"})
    out = {}
    for t in data.get('tickers', []):
        tk = t.get('ticker', '')
        if not TICKER_REGEX.match(tk):
            continue
        prev_c = t.get('prevDay', {}).get('c')
        if not prev_c or prev_c <= 0:
            continue

        day = t.get('day', {})
        day_open = day.get('o', 0) or 0
        day_close = day.get('c', 0) or 0
        day_vol = day.get('v', 0) or 0

        min_data = t.get('min', {})
        min_close = min_data.get('c', 0) or 0

        last_trade = t.get('lastTrade', {})
        last_price = last_trade.get('p', 0) or 0

        chg_pct = t.get('todaysChangePerc', 0) or 0

        # best current price: day close > min close > last trade > prev close
        if day_close > 0:
            cur_price = day_close
        elif min_close > 0:
            cur_price = min_close
        elif last_price > 0:
            cur_price = last_price
        else:
            cur_price = float(prev_c)

        market_open = day_open > 0

        out[tk] = {
            'prev_close': float(prev_c),
            'cur_price': cur_price,
            'day_open': day_open,
            'day_vol': day_vol,
            'chg_pct': chg_pct,
            'last_price': last_price,
            'market_open': market_open,
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TICKER TYPE LOOKUP (async, batched)
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_ticker_info(session, ticker, semaphore):
    url = f"{POLYGON_BASE}/v3/reference/tickers/{ticker}"
    params = {"apiKey": POLYGON_API_KEY}
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    if resp.status != 200:
                        return ticker, {'type': '?', 'name': '', 'sic': '', 'shares': 0}
                    data = await resp.json()
                    r = data.get('results', {})
                    return ticker, {
                        'type': r.get('type', '?'),
                        'name': (r.get('name') or '')[:40],
                        'sic': r.get('sic_description', '') or '',
                        'shares': r.get('weighted_shares_outstanding') or r.get('share_class_shares_outstanding') or 0,
                    }
            except Exception:
                await asyncio.sleep(1)
        return ticker, {'type': '?', 'name': '', 'sic': '', 'shares': 0}


async def _fetch_all_info(tickers):
    sem = asyncio.Semaphore(10)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_ticker_info(session, tk, sem) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return dict(results)


def fetch_ticker_info_batch(tickers):
    if not tickers:
        return {}
    try:
        asyncio.get_running_loop()
        nest_asyncio.apply()
    except RuntimeError:
        pass
    return asyncio.run(_fetch_all_info(tickers))


# ═══════════════════════════════════════════════════════════════════════════════
# CALENDAR
# ═══════════════════════════════════════════════════════════════════════════════

def is_trading_day(ds):
    dt = datetime.strptime(ds, '%Y-%m-%d').date() if isinstance(ds, str) else ds
    return dt.weekday() < 5 and ds not in _NYSE_HOLIDAYS

def get_prior_trading_date(target_date):
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    d = target_date - timedelta(days=1)
    for _ in range(10):
        if is_trading_day(d.strftime('%Y-%m-%d')):
            return d
        d -= timedelta(days=1)
    raise ValueError(f"No trading day found before {target_date}")


def _fmt_shares(shares):
    if shares >= 1e9: return f"{shares/1e9:.1f}B"
    if shares >= 1e6: return f"{shares/1e6:.0f}M"
    if shares > 0: return f"{shares/1e3:.0f}K"
    return 'n/a'


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCAN
# ═══════════════════════════════════════════════════════════════════════════════

t_start = time.time()

if SCAN_DATE is None:
    scan_date_str = datetime.now(ET).strftime('%Y-%m-%d')
    mode = 'live'
else:
    scan_date_str = SCAN_DATE
    mode = 'historical'

now_et = datetime.now(ET)
now_str = now_et.strftime('%H:%M ET')

print(f"\n{'='*80}")
print(f"  RAW GAP & MOVER SCANNER — All US >= {GAP_MIN*100:.0f}% move, >= ${PRICE_MIN}")
print(f"  Date: {scan_date_str}  |  Mode: {mode}  |  Time: {now_str}")
print(f"{'='*80}\n")

if not is_trading_day(scan_date_str):
    print(f"  {scan_date_str} is not a trading day.")
else:
    # ─── HISTORICAL MODE ─────────────────────────────────────────────────────
    if mode == 'historical':
        prior_date = get_prior_trading_date(scan_date_str)
        prior_str = prior_date.strftime('%Y-%m-%d')
        print(f"  Prior trading day: {prior_str}")

        t1 = time.time()
        prior_data = fetch_grouped_daily(prior_str)
        print(f"  Prior closes: {len(prior_data)} tickers ({time.time()-t1:.1f}s)")

        t2 = time.time()
        current_data = fetch_grouped_daily(scan_date_str)
        print(f"  Today's data: {len(current_data)} tickers ({time.time()-t2:.1f}s)")

        candidates = []
        for tk in current_data:
            if tk not in prior_data: continue
            if tk in _EXCLUDED_TICKERS: continue
            if any(tk.endswith(s) for s in _EXCLUDED_SUFFIXES) and len(tk) > 2: continue
            prev_c = prior_data[tk].get('close', 0)
            if prev_c <= 0: continue
            cur_p = current_data[tk].get('open', 0)
            if cur_p <= 0: continue
            gap_pct = (cur_p - prev_c) / prev_c
            if gap_pct < GAP_MIN: continue
            if cur_p < PRICE_MIN: continue
            candidates.append({
                'ticker': tk, 'gap_pct': gap_pct, 'chg_pct': gap_pct * 100,
                'prior_close': prev_c, 'cur_price': cur_p, 'source': 'open_gap',
            })

        spy_gap = None
        if 'SPY' in prior_data and 'SPY' in current_data:
            spy_prev = prior_data['SPY']['close']
            spy_cur = current_data['SPY'].get('open', 0)
            if spy_prev > 0 and spy_cur > 0:
                spy_gap = (spy_cur - spy_prev) / spy_prev * 100
                print(f"  SPY gap: {spy_gap:+.2f}%")

        is_premarket = False

    # ─── LIVE MODE ────────────────────────────────────────────────────────────
    else:
        t1 = time.time()
        snap = fetch_snapshot_full()
        print(f"  Snapshot: {len(snap)} tickers ({time.time()-t1:.1f}s)")

        # detect pre-market vs market open
        n_market_open = sum(1 for v in snap.values() if v['market_open'])
        is_premarket = n_market_open < 100
        if is_premarket:
            print(f"\n  ⚠ PRE-MARKET — regular session not open yet ({n_market_open} tickers with day data)")
            print(f"  Using todaysChangePerc from pre-market trades.")
            print(f"  Most CS stocks have NO pre-market trades → invisible to scanner.")
            print(f"  Re-run after 9:30 AM ET for real opening gaps.\n")
        else:
            print(f"  Market OPEN — {n_market_open} tickers with day data")

        # SPY
        spy_snap = snap.get('SPY', {})
        spy_gap = spy_snap.get('chg_pct')
        if spy_gap:
            print(f"  SPY change: {spy_gap:+.2f}%")

        candidates = []
        # also collect CS movers at lower threshold
        cs_movers_low = []

        for tk, sd in snap.items():
            if tk in _EXCLUDED_TICKERS: continue
            if any(tk.endswith(s) for s in _EXCLUDED_SUFFIXES) and len(tk) > 2: continue

            prev_c = sd['prev_close']
            cur_p = sd['cur_price']
            chg = sd['chg_pct']

            if cur_p < PRICE_MIN: continue

            if chg >= GAP_MIN * 100:
                candidates.append({
                    'ticker': tk,
                    'gap_pct': chg / 100,
                    'chg_pct': chg,
                    'prior_close': prev_c,
                    'cur_price': cur_p,
                    'day_open': sd['day_open'],
                    'source': 'snapshot_chg',
                })

            if chg >= CS_FLOOR * 100:
                cs_movers_low.append({
                    'ticker': tk,
                    'chg_pct': chg,
                    'prior_close': prev_c,
                    'cur_price': cur_p,
                })

    candidates.sort(key=lambda x: -x['chg_pct'])
    n_total = len(candidates)
    print(f"\n  Found {n_total} tickers with change >= {GAP_MIN*100:.0f}% and price >= ${PRICE_MIN}")

    if n_total == 0:
        print(f"\n  No tickers moving >= {GAP_MIN*100:.0f}% today.")
        if is_premarket:
            print(f"  This is expected pre-market. Re-run after 9:30 AM ET.")
    else:
        tickers_to_check = [c['ticker'] for c in candidates[:SHOW_TOP]]

        # also add CS movers at lower threshold for context
        if mode == 'live' and cs_movers_low:
            extra_tks = [m['ticker'] for m in cs_movers_low if m['ticker'] not in set(tickers_to_check)]
            tickers_to_check.extend(extra_tks[:50])

        print(f"  Fetching ticker info for {len(tickers_to_check)} tickers...")
        t3 = time.time()
        info = fetch_ticker_info_batch(tickers_to_check)
        print(f"  Ticker info fetched in {time.time()-t3:.1f}s")

        # ─── MAIN TABLE: all movers >= GAP_MIN ───────────────────────────────
        type_counts = {}
        cs_list = []
        non_cs_list = []

        rows = []
        for i, c in enumerate(candidates[:SHOW_TOP]):
            tk = c['ticker']
            ti = info.get(tk, {})
            tk_type = ti.get('type', '?')
            name = ti.get('name', '')
            shares = ti.get('shares', 0)
            type_counts[tk_type] = type_counts.get(tk_type, 0) + 1

            if tk_type == 'CS':
                cs_list.append(c)
            else:
                non_cs_list.append(c)

            rows.append([
                i + 1, tk, f"+{c['chg_pct']:.1f}%", f"${c['cur_price']:.2f}",
                tk_type, _fmt_shares(shares), name[:35],
            ])

        print(f"\n  {'─'*90}")
        print(f"  ALL MOVERS >= {GAP_MIN*100:.0f}% (top {min(SHOW_TOP, n_total)} of {n_total})")
        print(f"  {'─'*90}\n")
        print(tabulate(rows,
                       headers=['#', 'Ticker', 'Chg%', 'Price', 'Type', 'Shares', 'Name'],
                       tablefmt='simple',
                       colalign=('right', 'left', 'right', 'right', 'center', 'right', 'left')))

        # Type breakdown
        print(f"\n  {'─'*50}")
        print(f"  TYPE BREAKDOWN")
        print(f"  {'─'*50}")
        type_labels = {
            'CS': 'Common Stock', 'ETF': 'Exchange Traded Fund',
            'ETN': 'Exchange Traded Note', 'ETV': 'Exchange Traded Vehicle',
            'ETS': 'Exchange Traded Synthetic', 'ADRC': 'ADR Common',
            'FUND': 'Fund', 'SP': 'Structured Product', 'PFD': 'Preferred Stock',
        }
        for tk_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            label = type_labels.get(tk_type, tk_type)
            pct = count / min(SHOW_TOP, n_total) * 100
            print(f"    {tk_type:6s}  {count:3d}  ({pct:5.1f}%)  {label}")

        # ─── CS-ONLY: >= GAP_MIN ─────────────────────────────────────────────
        print(f"\n  {'─'*50}")
        print(f"  COMMON STOCKS >= {GAP_MIN*100:.0f}% ({len(cs_list)} found)")
        print(f"  {'─'*50}")
        if cs_list:
            cs_rows = []
            for i, c in enumerate(cs_list):
                tk = c['ticker']
                ti = info.get(tk, {})
                cs_rows.append([
                    i + 1, tk, f"+{c['chg_pct']:.1f}%", f"${c['cur_price']:.2f}",
                    _fmt_shares(ti.get('shares', 0)), ti.get('name', '')[:35],
                ])
            print()
            print(tabulate(cs_rows,
                           headers=['#', 'Ticker', 'Chg%', 'Price', 'Shares', 'Name'],
                           tablefmt='simple',
                           colalign=('right', 'left', 'right', 'right', 'right', 'left')))
            print(f"\n  Agent 1 would pick: #{1} {cs_list[0]['ticker']} (+{cs_list[0]['chg_pct']:.1f}%)")
        else:
            print(f"\n  NONE — no common stocks moving >= {GAP_MIN*100:.0f}% today.")
            if is_premarket:
                print(f"  Pre-market: most CS stocks don't trade. Wait for 9:30 AM ET open.")

        # ─── CS MOVERS at lower threshold (context) ──────────────────────────
        if mode == 'live' and cs_movers_low:
            cs_low_with_type = []
            for m in cs_movers_low:
                ti = info.get(m['ticker'], {})
                if ti.get('type') == 'CS' and m['chg_pct'] < GAP_MIN * 100:
                    cs_low_with_type.append(m)
            cs_low_with_type.sort(key=lambda x: -x['chg_pct'])

            if cs_low_with_type:
                print(f"\n  {'─'*50}")
                print(f"  CS MOVERS {CS_FLOOR*100:.0f}%-{GAP_MIN*100:.0f}% (below Agent 1 threshold)")
                print(f"  {'─'*50}")
                lo_rows = []
                for i, m in enumerate(cs_low_with_type[:20]):
                    tk = m['ticker']
                    ti = info.get(tk, {})
                    lo_rows.append([
                        i + 1, tk, f"+{m['chg_pct']:.1f}%", f"${m['cur_price']:.2f}",
                        _fmt_shares(ti.get('shares', 0)), ti.get('name', '')[:35],
                    ])
                print()
                print(tabulate(lo_rows,
                               headers=['#', 'Ticker', 'Chg%', 'Price', 'Shares', 'Name'],
                               tablefmt='simple',
                               colalign=('right', 'left', 'right', 'right', 'right', 'left')))
                print(f"\n  ({len(cs_low_with_type)} CS stocks between {CS_FLOOR*100:.0f}%-{GAP_MIN*100:.0f}%)")

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print()
