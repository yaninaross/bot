# %% Agent 1 — Scoring Time Sensitivity Analysis (Sep 18 only)
#
# Simulates the live scanner at every minute from 6:10 AM to 6:30 AM ET
# using premarket 1-minute bars from Polygon.
#
# For each scoring time:
#   1. Uses the last premarket trade price as of that minute
#   2. Computes gap vs prior close
#   3. Applies float + CS filters, ranks by gap
#   4. Shows top 3 candidates
#   5. Computes PnL for #1 pick (SHORT at local_high, exit EOD)
#
# Live scanner output on Sep 18 (at ~6:28 AM PT / 9:28 AM ET):
#   #1 DSP +15.9%  |  #2 HELP +6.4%  |  #3 BWMX +5.4%
#
# Colab-cell friendly.

import os, sys, time, re, asyncio
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

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SCAN_DATE      = '2026-09-18'
SCAN_TIMES_ET  = [(6, m) for m in range(10, 31)]   # 6:10 to 6:30 AM ET (21 minutes)

GAP_MIN_PCT    = 5.0
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
TOP_N          = 3
PRE_FILTER_N   = 15

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

CAPITAL        = 100_000

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')
ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')

_EXCLUDE_TK = {'SPY','QQQ','IWM','DIA','VXX','UVXY','SQQQ','TQQQ','SPXU','SPXL',
               'SDOW','UDOW','SDS','SSO','SH','LABU','LABD','SOXL','SOXS',
               'FNGU','FNGD','NUGT','DUST','JNUG','JDST','TNA','TZA',
               'UPRO','VIXY','SVXY','TVIX'}
_EXCLUDE_SUFFIX = ('W', 'WS', 'U', 'R')
_ETF_TYPES = {'ETF', 'ETV', 'ETN', 'ETP'}

_HOLIDAYS = {
    '2025-01-01','2025-01-20','2025-02-17','2025-04-18','2025-05-26',
    '2025-06-19','2025-07-04','2025-09-01','2025-11-27','2025-12-25',
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25',
    '2026-06-19','2026-07-03','2026-09-07','2026-11-26','2026-12-25',
}

_FLOAT = {
    'AAPL':14594180000,'AARD':21884158,'ABM':58580923,'ACHR':770023800,'ACMR':64657388,
    'ACV':10396028,'ACVA':169807980,'ADBE':397500000,'AEHR':32620450,'AENT':50979630,
    'AEON':49882790,'AGCO':70031729,'AGRZ':22573405,'AHMA':10965000,'ALAB':173485104,
    'AMAT':793597443,'AMD':1632475042,'AME':229203002,'APA':350351510,'APLD':291469112,
    'ARBE':122649743,'ARM':1068078760,'ASAN':159441754,'ASO':62018371,'ASTS':299789305,
    'ATEC':154451076,'ATHE':18208368,'ATI':136170383,'AVAV':50822615,'AXGN':53761513,
    'AXTA':214019997,'AXTI':65573212,'BABA':2485623615,'BBIO':195487474,'BBNX':44989688,
    'BE':294527346,'BEAM':103287665,'BHVN':151043781,'BJDX':4468051,'BLZE':61900000,
    'BNC':41173850,'BNKK':8000940,'BOW':32943005,'BRNX':720888,'BRZE':112631669,
    'BTBT':360633503,'CAN':749194858,'CAT':459674889,'CC':150498995,'CCB':15286327,
    'CIFR':415030722,'CLSK':256817073,'COIN':222803032,'COLA':4494439,'COO':195030630,
    'CORZ':321340441,'CPRT':925811482,'CRDO':187951918,'CRMT':8663493,'CSR':16794151,
    'CVX':1975771274,'DBGI':933509,'DELL':324873640,'DFNS':1663806,'DKS':65355923,
    'DNOW':180785891,'DOV':134680225,'DPRO':34375971,'DSS':10042518,'EAT':41765010,
    'EDHL':5959275,'ELOG':12832000,'ET':3443299601,'ETN':388400000,'EVRG':230567172,
    'EXEL':247781692,'FCEL':79954196,'FCX':1436017523,'FEIM':11177530,'FIZZ':93615302,
    'FLNC':143163588,'FLWS':37030262,'GDDY':126647704,'GENC':12338845,'GLXG':5601515,
    'GME':448691257,'GMEX':903642,'GOOS':46656876,'GRI':2186115,'HBM':444144760,
    'HCAI':7629942,'HGTY':102070523,'HOFT':10741650,'HOOD':790630234,'HPE':1327464779,
    'HPQ':901790969,'HTLD':77332611,'HUT':123259468,'HXHX':8950000,'IBTA':20197545,
    'IESC':19924356,'IMO':481577721,'INBX':14717660,'INDP':133242324,'INFQ':225357052,
    'INTC':5044000000,'INV':84612657,'IONQ':381002314,'IP':529569895,'IPDN':14450323,
    'IREN':394058648,'KEEL':617573212,'KPTI':22681460,'KR':612647282,'KTOS':187721327,
    'LASE':50177716,'LEU':19233658,'LOVE':14638418,'LULU':105594064,'LUNR':173231343,
    'M':263037954,'MARA':386299297,'MAS':197187430,'MCFT':16279890,'MDCX':63180610,
    'META':2205128509,'MGN':16250000,'MOBX':16847921,'MRNA':399235889,'MSTR':364585501,
    'MU':1129393151,'MVIS':23788853,'MXL':90693191,'NBIS':219465088,'NEE':2085803189,
    'NEGG':20973423,'NNE':53698146,'NTHI':25936365,'NTLA':140126693,'NVDA':24100000000,
    'NVS':2034819509,'OCGN':339110401,'OKLO':186017650,'ONDS':571454656,'OPTT':270138823,
    'OPTX':40279878,'ORCL':2880471000,'OTLK':243447554,'PAVS':856897,'PCLA':9613805,
    'PDEX':3203334,'PDSB':55971338,'PGY':72137266,'PL':340354193,'PLNT':75216965,
    'PLTR':2300713329,'PLUG':1397195278,'PMTS':11530669,'PRLD':79803485,'QBTS':368835324,
    'QNTM':3816937,'RDDT':146103200,'RDW':249992609,'REGN':101137842,'RGTI':333768747,
    'RH':18900969,'RIOT':375258935,'RKLB':598350482,'RLMD':106669846,'ROIV':722323015,
    'RTX':1347758144,'SCCO':834331706,'SDRL':62541443,'SHMD':60958903,'SHOE':27151308,
    'SIG':39329783,'SKYX':135430994,'SLS':201945709,'SLXN':5755872,'SMCI':656965384,
    'SMR':410389522,'SNDK':146419001,'SNX':79958752,'SNYR':15079956,'SOUN':411576753,
    'SRPT':105627021,'SSL':647761670,'SSNC':234654996,'STRL':30588518,'STX':226644518,
    'SURG':52908423,'SWKS':150472782,'SWRD':211149963,'SWVL':9964344,'TEN':30127603,
    'TNON':1032578,'TPST':16539717,'TRUG':1903708,'TSLA':3949547394,'TTAN':82746425,
    'TTD':426771565,'TYRA':59676939,'UBER':2042560121,'UNFI':60519140,'UPST':97313046,
    'URI':62242489,'VAC':34395320,'VICR':34389014,'VOYG':58217592,'VRT':384988173,
    'VSME':2750784,'VST':335635195,'WDH':279111455,'WULF':498968677,'YARW':2669746,
    'ZEO':41703489,'ZUMZ':16872215,
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

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

def _prior_td(d):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    d -= timedelta(days=1)
    for _ in range(10):
        if d.weekday() < 5 and d.strftime('%Y-%m-%d') not in _HOLIDAYS:
            return d
        d -= timedelta(days=1)
    raise ValueError(f"No trading day before {d}")


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC PREMARKET BAR FETCH (extended hours: 4:00 AM – 9:30 AM ET)
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_pm_bars_one(session, tk, dt, sem):
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
                        return tk, []
                    data = await resp.json()
            except Exception:
                await asyncio.sleep(1)
                continue
            bars = data.get('results', [])
            pm_bars = []
            rth_bars = []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                bar = {'o':float(b['o']),'h':float(b['h']),
                       'l':float(b['l']),'c':float(b['c']),'v':int(b['v']),
                       'time':t_str}
                if '04:00' <= t_str < '09:30':
                    pm_bars.append(bar)
                elif '09:30' <= t_str < '16:00':
                    rth_bars.append(bar)
            return tk, {'pm': pm_bars, 'rth': rth_bars}
        return tk, {'pm': [], 'rth': []}

async def _fetch_pm_bars_batch(tickers, dt):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_pm_bars_one(session, tk, dt, sem) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: val for tk, val in results}

def fetch_all_bars(tickers, dt):
    if not tickers: return {}
    return asyncio.run(_fetch_pm_bars_batch(tickers, dt))


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC TICKER REFERENCE FETCH (CS filter)
# ═══════════════════════════════════════════════════════════════════════════════

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
            return tk, {'type': res.get('type'), 'name': res.get('name', '')}
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


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL HIGH DETECTION (matches live scanner)
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_high(rth_bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    fallback_bar=FALLBACK_BAR):
    if len(rth_bars) < window * 2 + 1:
        return None
    for i in range(window, min(max_peak_bar, len(rth_bars) - window)):
        ch = rth_bars[i]['h']
        if all(rth_bars[i - j]['h'] < ch and rth_bars[i + j]['h'] < ch
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(rth_bars):
                return {
                    'entry_price': rth_bars[ci]['c'],
                    'entry_time': rth_bars[ci]['time'],
                    'entry_type': 'local_high',
                    'peak_bar': i,
                }
    if fallback_bar is not None and fallback_bar < len(rth_bars):
        return {
            'entry_price': rth_bars[fallback_bar]['c'],
            'entry_time': rth_bars[fallback_bar]['time'],
            'entry_type': 'fallback',
            'peak_bar': None,
        }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SCAN AT A SPECIFIC TIME
# ═══════════════════════════════════════════════════════════════════════════════

def get_premarket_price(pm_bars, as_of_time):
    """Get the last traded price as of as_of_time (HH:MM ET)."""
    last_price = None
    for bar in pm_bars:
        if bar['time'] <= as_of_time:
            last_price = bar['c']
        else:
            break
    return last_price


def scan_at_time(prior_closes, all_bars_data, cs_tickers, scan_time_str):
    """Run the full scanner logic using premarket prices as of scan_time_str."""
    candidates = []
    gap_min = GAP_MIN_PCT / 100.0

    for tk, bar_data in all_bars_data.items():
        if tk in _EXCLUDE_TK: continue
        if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
        if tk not in prior_closes: continue

        prev_c = prior_closes[tk]
        if prev_c <= 0: continue

        pm_bars = bar_data.get('pm', [])
        cur_p = get_premarket_price(pm_bars, scan_time_str)
        if cur_p is None or cur_p <= 0 or cur_p < PRICE_MIN:
            continue

        gap = (cur_p - prev_c) / prev_c
        if gap < gap_min:
            continue

        fs = _FLOAT.get(tk)
        if fs is not None and fs < FLOAT_MIN:
            continue

        if _ticker_ref.get(tk, {}).get('type') != 'CS':
            if tk in cs_tickers:
                pass
            else:
                continue

        candidates.append({
            'ticker': tk,
            'gap_pct': gap,
            'price': round(cur_p, 4),
            'prior_close': round(prev_c, 4),
        })

    candidates.sort(key=lambda x: -x['gap_pct'])
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

t_total = time.time()

print(f"\n{'='*80}")
print(f"  AGENT 1 — Scoring Time Sensitivity Analysis")
print(f"  Date: {SCAN_DATE}")
print(f"  Scoring window: 6:10 AM → 6:30 AM ET  (21 time points)")
print(f"  Live output: #1 DSP +15.9%  |  #2 HELP +6.4%  |  #3 BWMX +5.4%")
print(f"{'='*80}")

# ── Step 1: Prior closes ──
prior_str = _prior_td(SCAN_DATE).strftime('%Y-%m-%d')
print(f"\n  Step 1: Fetching prior day closes ({prior_str})...")
prior_gd = fetch_grouped_daily(prior_str)
prior_closes = {tk: d['c'] for tk, d in prior_gd.items() if d['c'] > 0}
print(f"  → {len(prior_closes)} tickers with prior close")

# ── Step 2: Find candidate universe ──
# Use grouped daily for scan_date to find tickers with big moves,
# then also include any ticker from embedded float dict as potential PM movers
print(f"\n  Step 2: Finding candidate universe...")
scan_gd = fetch_grouped_daily(SCAN_DATE)

# Tickers with actual open gap >= 3% (wider net to catch PM movers)
candidate_tickers = set()
gap_min_wide = 0.03
for tk in scan_gd:
    if tk in _EXCLUDE_TK: continue
    if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
    if tk not in prior_closes: continue
    if not _TK_RE.match(tk): continue
    prev_c = prior_closes[tk]
    if prev_c <= 0: continue
    cur_p = scan_gd[tk]['o']
    if cur_p <= 0 or cur_p < PRICE_MIN: continue
    gap = (cur_p - prev_c) / prev_c
    if gap >= gap_min_wide:
        candidate_tickers.add(tk)

# Also add the live scanner's picks to make sure we have them
for tk in ['DSP', 'HELP', 'BWMX', 'SVRN']:
    candidate_tickers.add(tk)

print(f"  → {len(candidate_tickers)} candidate tickers (open gap >= {gap_min_wide*100:.0f}%)")

# ── Step 3: Fetch premarket + RTH bars ──
print(f"\n  Step 3: Fetching premarket + RTH 1-min bars for {len(candidate_tickers)} tickers...")
tk_list = sorted(candidate_tickers)
all_bars_data = {}

# Batch in chunks to avoid overwhelming API
for c_start in range(0, len(tk_list), 100):
    chunk = tk_list[c_start:c_start + 100]
    result = fetch_all_bars(chunk, SCAN_DATE)
    all_bars_data.update(result)
    done = min(c_start + 100, len(tk_list))
    print(f"    {done}/{len(tk_list)} tickers  |  {time.time()-t_total:.0f}s")

n_with_pm = sum(1 for d in all_bars_data.values() if d.get('pm'))
n_with_rth = sum(1 for d in all_bars_data.values() if d.get('rth'))
print(f"  → PM bars: {n_with_pm}  |  RTH bars: {n_with_rth}")

# ── Step 4: Fetch reference data for CS filter ──
print(f"\n  Step 4: Fetching ticker reference (CS filter)...")
fetch_ticker_refs(tk_list)
cs_tickers = {tk for tk in tk_list if _ticker_ref.get(tk, {}).get('type') == 'CS'}
print(f"  → {len(cs_tickers)} common stocks out of {len(tk_list)}")

# ── Step 5: Scan at each time point ──
print(f"\n  Step 5: Scanning at each minute from 6:10 to 6:30 AM ET...")
print(f"\n  {'─'*90}")

results_by_time = {}
for h, m in SCAN_TIMES_ET:
    t_str = f"{h:02d}:{m:02d}"
    cands = scan_at_time(prior_closes, all_bars_data, cs_tickers, t_str)
    top = cands[:TOP_N]
    results_by_time[t_str] = {'candidates': cands, 'top': top}

    top_str = '  |  '.join(
        f"#{i+1} {c['ticker']} +{c['gap_pct']*100:.1f}% ${c['price']:.2f}"
        for i, c in enumerate(top)
    )
    n_qual = len(cands)
    match = '  ← MATCHES LIVE' if (top and top[0]['ticker'] == 'DSP') else ''
    print(f"  {t_str} ET  [{n_qual:>3} qualified]  {top_str}{match}")

print(f"  {'─'*90}")

# ── Step 6: PnL for #1 pick at each scoring time ──
print(f"\n  Step 6: Computing PnL for #1 pick at each scoring time...")

# Get EOD close from grouped daily
eod_data = scan_gd

pnl_table = []
for h, m in SCAN_TIMES_ET:
    t_str = f"{h:02d}:{m:02d}"
    top = results_by_time[t_str]['top']
    if not top:
        pnl_table.append({'time': t_str, 'ticker': '—', 'gap': 0, 'entry': None,
                          'entry_time': None, 'entry_type': None, 'close': None,
                          'pnl_eod': None})
        continue

    tk = top[0]['ticker']
    bar_data = all_bars_data.get(tk, {})
    rth = bar_data.get('rth', [])

    entry = find_local_high(rth) if rth else None
    eod_close = eod_data.get(tk, {}).get('c')

    pnl = None
    if entry and eod_close:
        ep = entry['entry_price']
        pnl = round((ep - eod_close) / ep * 100, 3)

    pnl_table.append({
        'time': t_str,
        'ticker': tk,
        'gap': top[0]['gap_pct'] * 100,
        'entry': entry['entry_price'] if entry else None,
        'entry_time': entry['entry_time'] if entry else None,
        'entry_type': entry['entry_type'] if entry else None,
        'close': eod_close,
        'pnl_eod': pnl,
    })

# Print PnL table
print(f"\n  {'─'*100}")
print(f"  {'Time ET':<10} {'Ticker':<8} {'Gap%':>8} {'Entry':>8} {'Entry@':>8} "
      f"{'Type':>12} {'Close':>8} {'PnL%':>8}  Notes")
print(f"  {'─'*100}")

for row in pnl_table:
    note = ''
    if row['ticker'] == 'DSP':
        note = '← LIVE PICK'
    elif row['ticker'] == '—':
        note = 'no candidates'

    entry_s = f"${row['entry']:.2f}" if row['entry'] else '—'
    etime_s = row['entry_time'] or '—'
    etype_s = row['entry_type'] or '—'
    close_s = f"${row['close']:.2f}" if row['close'] else '—'
    pnl_s = f"{row['pnl_eod']:+.2f}%" if row['pnl_eod'] is not None else '—'
    gap_s = f"+{row['gap']:.1f}%"

    print(f"  {row['time']:<10} {row['ticker']:<8} {gap_s:>8} {entry_s:>8} "
          f"{etime_s:>8} {etype_s:>12} {close_s:>8} {pnl_s:>8}  {note}")
print(f"  {'─'*100}")

# ── Step 7: Charts ──
print(f"\n  Step 7: Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Agent 1 Scoring Time Sensitivity — {SCAN_DATE}', fontsize=14, fontweight='bold')

# Chart 1: Top candidate ticker over time
ax1 = axes[0, 0]
times = [r['time'] for r in pnl_table]
tickers_at_time = [r['ticker'] for r in pnl_table]
unique_tickers = sorted(set(t for t in tickers_at_time if t != '—'))
ticker_to_idx = {tk: i for i, tk in enumerate(unique_tickers)}
y_vals = [ticker_to_idx.get(tk, -1) for tk in tickers_at_time]

colors = []
for tk in tickers_at_time:
    if tk == 'DSP':
        colors.append('green')
    elif tk == '—':
        colors.append('gray')
    else:
        colors.append('red')

ax1.scatter(range(len(times)), y_vals, c=colors, s=100, zorder=5)
ax1.set_xticks(range(len(times)))
ax1.set_xticklabels(times, rotation=45, ha='right', fontsize=8)
ax1.set_yticks(range(len(unique_tickers)))
ax1.set_yticklabels(unique_tickers, fontsize=9)
ax1.set_title('#1 Pick by Scoring Time (green = DSP = live match)')
ax1.set_xlabel('Scoring Time (ET)')
ax1.set_ylabel('Ticker')
ax1.grid(True, alpha=0.3)

# Chart 2: Gap % of #1 pick over time
ax2 = axes[0, 1]
gaps = [r['gap'] for r in pnl_table]
bar_colors = ['green' if r['ticker'] == 'DSP' else 'steelblue' for r in pnl_table]
ax2.bar(range(len(times)), gaps, color=bar_colors, alpha=0.8)
ax2.set_xticks(range(len(times)))
ax2.set_xticklabels(times, rotation=45, ha='right', fontsize=8)
ax2.set_title('Gap % of #1 Pick by Scoring Time')
ax2.set_xlabel('Scoring Time (ET)')
ax2.set_ylabel('Gap %')
ax2.grid(True, alpha=0.3, axis='y')

# Chart 3: PnL of #1 pick by scoring time
ax3 = axes[1, 0]
pnls = [r['pnl_eod'] if r['pnl_eod'] is not None else 0 for r in pnl_table]
pnl_colors = ['green' if p > 0 else 'red' for p in pnls]
ax3.bar(range(len(times)), pnls, color=pnl_colors, alpha=0.8)
ax3.axhline(y=0, color='black', linewidth=0.5)
ax3.set_xticks(range(len(times)))
ax3.set_xticklabels(times, rotation=45, ha='right', fontsize=8)
ax3.set_title('SHORT PnL (EOD) of #1 Pick by Scoring Time')
ax3.set_xlabel('Scoring Time (ET)')
ax3.set_ylabel('PnL %')
ax3.grid(True, alpha=0.3, axis='y')

# Chart 4: Number of qualifying candidates over time
ax4 = axes[1, 1]
n_cands = [len(results_by_time[f"{h:02d}:{m:02d}"]['candidates'])
           for h, m in SCAN_TIMES_ET]
ax4.plot(range(len(times)), n_cands, 'o-', color='steelblue', markersize=5)
ax4.set_xticks(range(len(times)))
ax4.set_xticklabels(times, rotation=45, ha='right', fontsize=8)
ax4.set_title('Number of Qualifying Candidates (gap≥5%, CS, float≥10M)')
ax4.set_xlabel('Scoring Time (ET)')
ax4.set_ylabel('Count')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('agent1_time_sensitivity.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: agent1_time_sensitivity.png")

# ── Step 8: Premarket price evolution for DSP, HELP, BWMX ──
print(f"\n  Step 8: Premarket price evolution for live picks...")

live_picks = ['DSP', 'HELP', 'BWMX']
fig2, ax = plt.subplots(figsize=(14, 6))
for tk in live_picks:
    bar_data = all_bars_data.get(tk, {})
    pm_bars = bar_data.get('pm', [])
    if not pm_bars:
        print(f"  ⚠ {tk}: no premarket bars")
        continue
    prev_c = prior_closes.get(tk, 0)
    if prev_c <= 0:
        continue
    pm_times = [b['time'] for b in pm_bars if '06:00' <= b['time'] <= '09:30']
    pm_gaps = [(b['c'] - prev_c) / prev_c * 100 for b in pm_bars if '06:00' <= b['time'] <= '09:30']
    ax.plot(range(len(pm_times)), pm_gaps, 'o-', markersize=3, label=f"{tk} (prior close ${prev_c:.2f})")
    if len(pm_times) > 0:
        ax.set_xticks(range(0, len(pm_times), max(1, len(pm_times)//15)))
        tick_labels = [pm_times[i] for i in range(0, len(pm_times), max(1, len(pm_times)//15))]

ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% gap threshold')
ax.set_xlabel('Time (ET)')
ax.set_ylabel('Gap % vs Prior Close')
ax.set_title(f'Premarket Gap Evolution — Live Picks ({SCAN_DATE})')
ax.legend()
ax.grid(True, alpha=0.3)
if pm_times:
    step = max(1, len(pm_times) // 20)
    ax.set_xticks(range(0, len(pm_times), step))
    ax.set_xticklabels([pm_times[i] for i in range(0, len(pm_times), step)], rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('agent1_pm_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: agent1_pm_evolution.png")

elapsed = time.time() - t_total
print(f"\n  DONE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
