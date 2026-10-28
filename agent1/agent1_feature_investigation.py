# %% Agent 1 — Feature Investigation (Bias-Free)
#
# PURPOSE: Identify ALL features that predict SHORT PnL, tag each with
# WHEN it becomes knowable, and surface only tradeable filter ideas.
#
# Two entry windows exist:
#   EARLY  bars 3-11  (9:33-9:41 ET) — must use pre-9:33 features only
#   LATE   bars 60-72 (10:30-10:42 ET) — can use first-hour features
#
# Every feature gets a timing tag:
#   PRE_SCAN   — known before 9:31 ET (gap, float, PM price action)
#   PRE_EARLY  — known before 9:33 ET (first RTH bar)
#   PRE_LATE   — known before 10:30 ET (first 60 bars of RTH)
#   POST_ENTRY — only known after trade is entered (NOT tradeable)
#
# Reuses the backtest pipeline from agent1_baseline_bt.py for data fetch.
# Runs E2-A baseline at TOP_N=5 to get a broad trade set for analysis.
# Colab-cell friendly.

import os, sys, time, re, asyncio, copy
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
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from scipy import stats as sp_stats

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_START = '2026-07-01'
BACKTEST_END   = '2026-10-09'

GAP_MIN_PCT    = 3.0
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
TOP_N          = 5          # wide net for investigation
PRE_FILTER_N   = 30

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

SD_MAX_PEAK_BAR = 72
SD_FALLBACK_BAR = 75
SD_CONFIRM_BAR  = 59

EXIT_TIME      = '17:00'
SCAN_TIME_ET   = '09:31'
GAP_CONSUMED_MAX = 0.50

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
_ETF_TYPES = {'ETF', 'ETV', 'ETN', 'ETP'}

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
# HELPERS (copied from backtest — self-contained script)
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
                time.sleep(12 * (attempt + 1)); continue
            r.raise_for_status(); return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(5 * (2 ** attempt))
    r = requests.get(url, params=p, timeout=60); r.raise_for_status(); return r.json()

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
                    if resp.status == 429: await asyncio.sleep(2 ** (attempt + 1)); continue
                    if resp.status != 200: return (tk, dt), None
                    data = await resp.json()
            except Exception: await asyncio.sleep(1); continue
            bars = data.get('results', [])
            if not bars: return (tk, dt), None
            pm, rth, ah = [], [], []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                bar_d = {'o':float(b['o']),'h':float(b['h']),'l':float(b['l']),'c':float(b['c']),'v':int(b['v'])}
                if '04:00' <= t_str < '09:30': pm.append((t_str, bar_d))
                elif '09:30' <= t_str < '16:00': rth.append((t_str, bar_d))
                elif '16:00' <= t_str <= '20:00': ah.append((t_str, bar_d))
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

_ticker_ref = {}

async def _fetch_ref_one(session, tk, sem):
    url = f"{_BASE}/v3/reference/tickers/{tk}"
    params = {"apiKey": POLYGON_API_KEY}
    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 429: await asyncio.sleep(2 ** (attempt + 1)); continue
                    if resp.status != 200: return tk, {}
                    data = await resp.json()
            except Exception: await asyncio.sleep(1); continue
            res = data.get('results', {})
            return tk, {'type': res.get('type'), 'name': res.get('name', ''),
                        'shares': res.get('share_class_shares_outstanding')}
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
    if len(rth_bars) < 60: return 'no_data'
    open_price = rth_bars[0][1]['o']
    low_30m = min(b[1]['l'] for b in rth_bars[:30])
    close_60m = rth_bars[59][1]['c']
    drop_30m = (low_30m - open_price) / open_price if open_price > 0 else 0
    recovery_denom = abs(open_price - low_30m)
    recovery = (close_60m - low_30m) / recovery_denom if recovery_denom > 0 else 0
    if drop_30m < -0.03 and recovery > 0.5: return 'drop_then_recover'
    elif drop_30m < -0.03 and recovery <= 0.5: return 'sustained_drop'
    elif drop_30m >= -0.01: return 'flat_or_bounce'
    else: return 'mild_drop'

def find_local_high(bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    fallback_bar=FALLBACK_BAR, min_entry_bar=0):
    if len(bars) < window * 2 + 1: return None
    start = max(window, min_entry_bar - window)
    for i in range(start, min(max_peak_bar, len(bars) - window)):
        ch = bars[i][1]['h']
        if all(bars[i - j][1]['h'] < ch and bars[i + j][1]['h'] < ch
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(bars):
                return {'entry_price': bars[ci][1]['c'], 'entry_time': bars[ci][0],
                        'entry_type': 'local_high', 'peak_bar': i}
    if fallback_bar is not None:
        fb = max(fallback_bar, min_entry_bar)
        if fb < len(bars):
            return {'entry_price': bars[fb][1]['c'], 'entry_time': bars[fb][0],
                    'entry_type': 'fallback', 'peak_bar': None}
    return None

def _T_to_min_entry_bar(T):
    h, m = int(T[:2]), int(T[3:5])
    T_min = h * 60 + m
    rth_start = 9 * 60 + 30
    return max(0, T_min - rth_start)

def get_price_at_T(bar_data, T):
    if bar_data is None: return None
    all_bars = (bar_data.get('pm') or []) + (bar_data.get('rth') or [])
    best = None
    for t_str, bd in all_bars:
        if t_str <= T: best = bd['c']
        else: break
    return best

def scan_day(dt_str):
    if not _is_td(dt_str): return []
    prior_str = _prior_td(dt_str).strftime('%Y-%m-%d')
    prior = fetch_grouped_daily(prior_str)
    today = fetch_grouped_daily(dt_str)
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
        if gap < 0.03: continue
        fs = _FLOAT.get(tk)
        if fs is not None and fs < FLOAT_MIN: continue
        candidates.append({'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                           'open_price': cur_p, 'scan_date': dt_str,
                           'volume_today': today[tk].get('v', 0)})
    candidates.sort(key=lambda x: -x['gap_pct'])
    nxt = _next_n_td(dt_str, 2)
    t1 = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
    t2 = nxt[1].strftime('%Y-%m-%d') if len(nxt) >= 2 else None
    for s in candidates:
        s['t1_date'] = t1; s['t2_date'] = t2
        s['prior_str'] = prior_str
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATA PIPELINE (same as backtest)
# ═══════════════════════════════════════════════════════════════════════════════

t_total = time.time()
print(f"\n{'='*80}")
print(f"  AGENT 1 — FEATURE INVESTIGATION (Bias-Free)")
print(f"  {BACKTEST_START} → {BACKTEST_END}")
print(f"  TOP_N = {TOP_N} (wide net for feature analysis)")
print(f"{'='*80}")

d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
dates = []
while d <= end:
    if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
    d += timedelta(days=1)
print(f"\n  {len(dates)} trading days")

print(f"\n  Phase 1: Scanning...")
all_candidates = {}
for i, dt in enumerate(dates):
    all_candidates[dt] = scan_day(dt)
    if (i+1) % 30 == 0 or i == len(dates) - 1:
        print(f"    [{i+1:>3}/{len(dates)}] {dt}  |  {time.time()-t_total:.0f}s")

all_pairs = set()
for dt, cands in all_candidates.items():
    for c in cands[:PRE_FILTER_N]:
        all_pairs.add((c['ticker'], dt))
        if c.get('prior_str'): all_pairs.add((c['ticker'], c['prior_str']))

print(f"\n  Phase 2: Fetching 1-min bars for {len(all_pairs)} pairs...")
pairs_list = list(all_pairs)
all_bars = {}
for c_start in range(0, len(pairs_list), 100):
    chunk = pairs_list[c_start:c_start + 100]
    all_bars.update(fetch_minute_bars_batch(chunk))
    done = min(c_start + 100, len(pairs_list))
    if done % 200 == 0 or done == len(pairs_list):
        print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

gap_min_final = GAP_MIN_PCT / 100.0
n_stale = 0
for dt in dates:
    for c in all_candidates.get(dt, [])[:PRE_FILTER_N]:
        tk = c['ticker']
        prior_bars = all_bars.get((tk, c.get('prior_str', '')))
        if prior_bars:
            ah_bars = prior_bars.get('ah', [])
            if ah_bars:
                ah_ref = ah_bars[-1][1]['c']
                c['ah_ref_price'] = ah_ref
                fresh_gap = (c['open_price'] - ah_ref) / ah_ref
                c['gap_pct_daily'] = c['gap_pct']
                c['gap_pct'] = fresh_gap
                if fresh_gap < gap_min_final: c['ah_stale'] = True; n_stale += 1

n_eroded = 0
for dt in dates:
    cands = all_candidates.get(dt, [])
    reranked = []
    for c in cands:
        if c.get('ah_stale'): continue
        tk = c['ticker']
        bar_data = all_bars.get((tk, dt))
        price_T = get_price_at_T(bar_data, SCAN_TIME_ET)
        if price_T is None: continue
        ref_price = c.get('ah_ref_price', c['prior_close'])
        gap_T = (price_T - ref_price) / ref_price
        if gap_T < gap_min_final or price_T < PRICE_MIN: continue
        c['gap_pct'] = gap_T; c['scan_price'] = price_T
        rth = bar_data.get('rth', []) if bar_data else []
        if rth:
            daily_open = rth[0][1]['o']
            scan_bars = [b for b in rth if b[0] <= SCAN_TIME_ET]
            gap_range = daily_open - ref_price
            if scan_bars and gap_range > 0:
                low_to_scan = min(b[1]['l'] for b in scan_bars)
                gap_consumed = (daily_open - low_to_scan) / gap_range
                c['gap_consumed'] = gap_consumed
                if gap_consumed > GAP_CONSUMED_MAX: n_eroded += 1; continue
        reranked.append(c)
    reranked.sort(key=lambda x: -x['gap_pct'])
    all_candidates[dt] = reranked

print(f"  → {n_stale} stale gaps excluded, {n_eroded} eroded")

all_tickers = set()
for cands in all_candidates.values():
    for c in cands[:PRE_FILTER_N]: all_tickers.add(c['ticker'])
print(f"\n  Phase 3: Fetching reference for {len(all_tickers)} tickers...")
tk_list = sorted(all_tickers)
for c_start in range(0, len(tk_list), 200):
    fetch_ticker_refs(tk_list[c_start:c_start + 200])

min_eb = _T_to_min_entry_bar(SCAN_TIME_ET)
pattern_cache = {}
entry_cache_bl = {}
entry_cache_sd = {}
for key, bar_data in all_bars.items():
    rth_bars = bar_data.get('rth', []) if isinstance(bar_data, dict) else bar_data
    pattern_cache[key] = classify_open_pattern(rth_bars)
    entry_cache_bl[key] = find_local_high(rth_bars, min_entry_bar=min_eb) if rth_bars and len(rth_bars) >= PEAK_WINDOW * 2 + 1 else None
    entry_cache_sd[key] = find_local_high(rth_bars, window=PEAK_WINDOW, max_peak_bar=SD_MAX_PEAK_BAR,
                                           fallback_bar=SD_FALLBACK_BAR, min_entry_bar=60) if rth_bars and len(rth_bars) > SD_MAX_PEAK_BAR + PEAK_WINDOW else None

for cands in all_candidates.values():
    for c in cands[:PRE_FILTER_N]:
        outcome_dates = [c['scan_date'], c.get('t1_date',''), c.get('t2_date','')]
        for od in outcome_dates:
            if od and od not in _gd_cache:
                try: fetch_grouped_daily(od)
                except: pass

print(f"\n  Pipeline done in {time.time()-t_total:.0f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: BUILD TRADES + COMPUTE ALL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print(f"  PHASE 2: Computing features for every trade")
print(f"{'='*80}")

trades = []
for dt in dates:
    cands = all_candidates.get(dt, [])
    cs_qualified = []
    for c in cands[:PRE_FILTER_N]:
        tk = c['ticker']
        ref = _ticker_ref.get(tk, {})
        if ref.get('type') != 'CS': continue
        cc = copy.deepcopy(c)
        cc['asset_type'] = 'CS'
        cs_qualified.append(cc)

    for c in cs_qualified[:TOP_N]:
        tk, dt_s = c['ticker'], c['scan_date']
        entry = entry_cache_bl.get((tk, dt_s))
        if entry is None: continue

        tc = copy.deepcopy(c)
        tc['entry_price'] = entry['entry_price']
        tc['entry_time'] = entry['entry_time']
        tc['entry_type'] = entry['entry_type']
        tc['open_pattern'] = pattern_cache.get((tk, dt_s), 'no_data')
        ep = tc['entry_price']

        bar_data = all_bars.get((tk, dt_s))
        rth = bar_data.get('rth', []) if bar_data else []
        pm_bars = bar_data.get('pm', []) if bar_data else []

        # ── EXIT PRICE (17:00 ET) ──
        exit_price = None
        if bar_data:
            session_bars = rth + bar_data.get('ah', [])
            for t_str, bar in session_bars:
                if t_str > EXIT_TIME: break
                if t_str >= tc['entry_time']: exit_price = bar['c']
        if exit_price is not None:
            tc['pnl_eod'] = round((ep - exit_price) / ep * 100, 3)
        else:
            t0d = _gd_cache.get(dt_s, {}).get(tk, {})
            tc['pnl_eod'] = round((ep - t0d['c']) / ep * 100, 3) if t0d.get('c') else None
        tc['exit_price'] = exit_price

        t1d = _gd_cache.get(tc.get('t1_date', ''), {}).get(tk, {})
        tc['pnl_t1'] = round((ep - t1d['c']) / ep * 100, 3) if t1d.get('c') else None
        t2d = _gd_cache.get(tc.get('t2_date', ''), {}).get(tk, {})
        tc['pnl_t2'] = round((ep - t2d['c']) / ep * 100, 3) if t2d.get('c') else None

        if tc.get('pnl_eod') is None: continue

        # ═══════════════════════════════════════════════════════════════
        # FEATURE COMPUTATION — tagged by timing
        # ═══════════════════════════════════════════════════════════════

        # ── PRE_SCAN features (known before 9:31 ET) ──

        # Gap tier
        gp = tc['gap_pct']
        if gp < 0.05: tc['gap_tier'] = 'T5_3-5%'
        elif gp < 0.10: tc['gap_tier'] = 'T5b_5-10%'
        elif gp < 0.15: tc['gap_tier'] = 'T6a_10-15%'
        elif gp < 0.30: tc['gap_tier'] = 'T6_15-30%'
        else: tc['gap_tier'] = 'T7_30%+'

        tc['gap_pct_f'] = gp
        tc['log_gap'] = np.log1p(gp)

        # Float
        fs = _FLOAT.get(tk)
        tc['has_float'] = 1 if fs is not None else 0
        tc['float_shares'] = fs if fs else None
        tc['float_bucket'] = ('small' if fs and fs < 100_000_000
                              else 'mid' if fs and fs < 500_000_000
                              else 'large' if fs else 'unknown')

        # Day of week
        dt_obj = datetime.strptime(dt_s, '%Y-%m-%d')
        tc['dow'] = dt_obj.strftime('%A')
        tc['dow_num'] = dt_obj.weekday()

        # PM price features
        pm_after_830 = [(t_s, b) for t_s, b in pm_bars if t_s >= '08:30']
        prior_close = tc.get('prior_close', 0)
        if pm_after_830 and prior_close > 0:
            pm_high = max(b['h'] for _, b in pm_after_830)
            pm_low = min(b['l'] for _, b in pm_after_830)
            last_pm_close = pm_after_830[-1][1]['c']
            first_pm_open = pm_after_830[0][1]['o']
            pm_range = pm_high - pm_low

            denom = pm_high - prior_close
            tc['pm_retrace'] = (pm_high - last_pm_close) / denom if denom > 0 else 0
            tc['pm_trend'] = (last_pm_close - first_pm_open) / first_pm_open if first_pm_open > 0 else 0
            tc['pm_range_pct'] = pm_range / prior_close if prior_close > 0 else 0
            tc['pm_high_pct'] = (pm_high - prior_close) / prior_close
            tc['pm_low_pct'] = (pm_low - prior_close) / prior_close
            tc['pm_bar_count'] = len(pm_after_830)
            tc['pm_volume'] = sum(b['v'] for _, b in pm_after_830)

            # PM retrace bucket
            r = tc['pm_retrace']
            tc['pm_retrace_bucket'] = ('low' if r < 0.05 else 'mid' if r < 0.15 else 'high')
        else:
            tc['pm_retrace'] = None; tc['pm_trend'] = None; tc['pm_range_pct'] = None
            tc['pm_high_pct'] = None; tc['pm_low_pct'] = None; tc['pm_bar_count'] = 0
            tc['pm_volume'] = 0; tc['pm_retrace_bucket'] = 'no_pm'

        # Gap freshness (daily vs AH gap)
        tc['gap_freshness'] = tc.get('gap_pct', 0) / tc.get('gap_pct_daily', tc.get('gap_pct', 0.01)) if tc.get('gap_pct_daily', 0) > 0 else 1.0

        # Gap consumed by scan time
        tc['gap_consumed_f'] = tc.get('gap_consumed', 0)

        # ── PRE_EARLY features (known before 9:33 ET = bar 3) ──

        if rth and len(rth) >= 2:
            bar0 = rth[0][1]
            tc['open_bar_direction'] = 1 if bar0['c'] > bar0['o'] else -1
            tc['open_bar_range_pct'] = (bar0['h'] - bar0['l']) / bar0['o'] if bar0['o'] > 0 else 0
            tc['open_bar_body_pct'] = abs(bar0['c'] - bar0['o']) / bar0['o'] if bar0['o'] > 0 else 0
            tc['bar1_direction'] = 1 if rth[1][1]['c'] > rth[1][1]['o'] else -1
            # first 2 bars combined drop
            tc['first_2bar_drop'] = (rth[1][1]['c'] - rth[0][1]['o']) / rth[0][1]['o'] if rth[0][1]['o'] > 0 else 0
        else:
            tc['open_bar_direction'] = 0; tc['open_bar_range_pct'] = 0
            tc['open_bar_body_pct'] = 0; tc['bar1_direction'] = 0; tc['first_2bar_drop'] = 0

        # ── PRE_LATE features (known before 10:30 ET = bar 60) ──
        # These are ONLY usable for E2-B (late entry) strategies

        if rth and len(rth) >= 30:
            open_price = rth[0][1]['o']
            low_30m = min(b[1]['l'] for b in rth[:30])
            close_30m = rth[29][1]['c']
            tc['drop_30m'] = (low_30m - open_price) / open_price if open_price > 0 else 0
            tc['close_30m_vs_open'] = (close_30m - open_price) / open_price if open_price > 0 else 0
            tc['range_30m'] = (max(b[1]['h'] for b in rth[:30]) - low_30m) / open_price if open_price > 0 else 0
            tc['vol_30m'] = sum(b[1]['v'] for b in rth[:30])
            # momentum: bar 15 vs bar 30
            mom_0_15 = (rth[14][1]['c'] - open_price) / open_price if open_price > 0 else 0
            mom_15_30 = (rth[29][1]['c'] - rth[14][1]['c']) / open_price if open_price > 0 else 0
            tc['momentum_0_15'] = mom_0_15
            tc['momentum_15_30'] = mom_15_30
            tc['price_accel'] = mom_15_30 - mom_0_15
            # drift at bar 30
            tc['bar30_above_open'] = 1 if close_30m > open_price else 0
        else:
            tc['drop_30m'] = None; tc['close_30m_vs_open'] = None
            tc['range_30m'] = None; tc['vol_30m'] = None
            tc['momentum_0_15'] = None; tc['momentum_15_30'] = None
            tc['price_accel'] = None; tc['bar30_above_open'] = None

        if rth and len(rth) >= 60:
            open_price = rth[0][1]['o']
            low_60m = min(b[1]['l'] for b in rth[:60])
            close_60m = rth[59][1]['c']
            tc['drop_60m'] = (low_60m - open_price) / open_price if open_price > 0 else 0
            tc['close_60m_vs_open'] = (close_60m - open_price) / open_price if open_price > 0 else 0
            recovery_denom = abs(open_price - low_60m)
            tc['recovery_60m'] = (close_60m - low_60m) / recovery_denom if recovery_denom > 0 else 0
            tc['vol_60m'] = sum(b[1]['v'] for b in rth[:60])
        else:
            tc['drop_60m'] = None; tc['close_60m_vs_open'] = None
            tc['recovery_60m'] = None; tc['vol_60m'] = None

        # ── POST_ENTRY features (NOT tradeable — for analysis only) ──
        tc['is_sd'] = 1 if tc['open_pattern'] == 'sustained_drop' else 0
        has_late_entry = entry_cache_sd.get((tk, dt_s)) is not None
        tc['has_late_entry'] = 1 if has_late_entry else 0

        # period tag
        tc['period'] = 'post_aug' if dt_s >= '2026-08-16' else 'pre_aug'

        trades.append(tc)

print(f"\n  Built {len(trades)} trades with features")
n_pre = sum(1 for t in trades if t['period'] == 'pre_aug')
n_post = sum(1 for t in trades if t['period'] == 'post_aug')
print(f"  Pre-Aug: {n_pre}  |  Post-Aug: {n_post}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: FEATURE ANALYSIS — correlation with PnL
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print(f"  PHASE 3: FEATURE → PnL CORRELATION (all trades)")
print(f"  Each feature tagged: PRE_SCAN / PRE_EARLY / PRE_LATE / POST_ENTRY")
print(f"{'='*120}")

CONTINUOUS_FEATURES = [
    # (feature_key, label, timing)
    ('gap_pct_f',          'Gap %',               'PRE_SCAN'),
    ('log_gap',            'Log(1+gap)',           'PRE_SCAN'),
    ('gap_consumed_f',     'Gap consumed @scan',   'PRE_SCAN'),
    ('gap_freshness',      'Gap freshness',        'PRE_SCAN'),
    ('pm_retrace',         'PM retrace',           'PRE_SCAN'),
    ('pm_trend',           'PM trend',             'PRE_SCAN'),
    ('pm_range_pct',       'PM range %',           'PRE_SCAN'),
    ('pm_high_pct',        'PM high vs close',     'PRE_SCAN'),
    ('pm_bar_count',       'PM bar count',         'PRE_SCAN'),
    ('pm_volume',          'PM volume',            'PRE_SCAN'),
    ('open_bar_direction', 'Bar0 direction',       'PRE_EARLY'),
    ('open_bar_range_pct', 'Bar0 range %',         'PRE_EARLY'),
    ('open_bar_body_pct',  'Bar0 body %',          'PRE_EARLY'),
    ('bar1_direction',     'Bar1 direction',       'PRE_EARLY'),
    ('first_2bar_drop',    'First 2-bar drop',     'PRE_EARLY'),
    ('drop_30m',           '30m drop',             'PRE_LATE'),
    ('close_30m_vs_open',  '30m close vs open',    'PRE_LATE'),
    ('range_30m',          '30m range',            'PRE_LATE'),
    ('momentum_0_15',      'Mom 0-15m',            'PRE_LATE'),
    ('momentum_15_30',     'Mom 15-30m',           'PRE_LATE'),
    ('price_accel',        'Price accel',          'PRE_LATE'),
    ('bar30_above_open',   'Bar30 > open',         'PRE_LATE'),
    ('drop_60m',           '60m drop',             'PRE_LATE'),
    ('close_60m_vs_open',  '60m close vs open',    'PRE_LATE'),
    ('recovery_60m',       '60m recovery',         'PRE_LATE'),
    ('is_sd',              'Is sust_drop',         'POST_ENTRY'),
]

CATEGORICAL_FEATURES = [
    ('gap_tier',           'Gap tier',             'PRE_SCAN'),
    ('float_bucket',       'Float bucket',         'PRE_SCAN'),
    ('pm_retrace_bucket',  'PM retrace bucket',    'PRE_SCAN'),
    ('dow',                'Day of week',          'PRE_SCAN'),
    ('entry_type',         'Entry type',           'PRE_EARLY'),
    ('open_pattern',       'Open pattern (60m)',   'POST_ENTRY'),
]

# ── Continuous features: Pearson correlation ──
print(f"\n  ── Continuous features: correlation with EOD PnL ──")
print(f"  {'Feature':<25s} {'Timing':<12s} {'r':>8s} {'p-value':>10s} {'n':>5s}  {'Direction':<12s}")
print(f"  {'-'*80}")

corr_results = []
for feat_key, label, timing in CONTINUOUS_FEATURES:
    vals = [(t[feat_key], t['pnl_eod']) for t in trades
            if t.get(feat_key) is not None and not np.isnan(t[feat_key]) and not np.isinf(t[feat_key])]
    if len(vals) < 10: continue
    x, y = zip(*vals)
    x, y = np.array(x), np.array(y)
    r, p = sp_stats.pearsonr(x, y)
    direction = 'POSITIVE' if r > 0 else 'NEGATIVE'
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
    corr_results.append((abs(r), feat_key, label, timing, r, p, len(vals), direction, sig))
    print(f"  {label:<25s} {timing:<12s} {r:+.4f}   {p:.4f}     {len(vals):>5d}  {direction} {sig}")

# sort by absolute correlation
corr_results.sort(key=lambda x: -x[0])
print(f"\n  ── TOP 10 by |r| ──")
for i, (_, key, label, timing, r, p, n, direction, sig) in enumerate(corr_results[:10], 1):
    tradeable = 'EARLY+LATE' if timing in ('PRE_SCAN', 'PRE_EARLY') else 'LATE only' if timing == 'PRE_LATE' else 'NOT TRADEABLE'
    print(f"  {i:>2d}. {label:<25s} {timing:<12s} r={r:+.4f}  p={p:.4f}  n={n}  → {tradeable}")

# ── Categorical features: group means ──
print(f"\n\n  ── Categorical features: group PnL ──")
for feat_key, label, timing in CATEGORICAL_FEATURES:
    print(f"\n  {label} ({timing}):")
    groups = defaultdict(list)
    for t in trades:
        v = t.get(feat_key, 'unknown')
        groups[v].append(t['pnl_eod'])
    rows = []
    for grp in sorted(groups.keys()):
        a = np.array(groups[grp])
        sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
        rows.append([grp, len(a), f"{np.mean(a):+.2f}%", f"{np.median(a):+.2f}%",
                      f"{float((a>0).mean())*100:.0f}%", f"{sh:.3f}",
                      f"${100000 * np.sum(a) / 100:+,.0f}"])
    print(tabulate(rows, headers=['Group', 'N', 'Avg', 'Median', 'WR', 'Sharpe', '$NC'],
                   tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: PRE-AUG vs POST-AUG feature shift
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print(f"  PHASE 4: PRE-AUG vs POST-AUG — what changed?")
print(f"{'='*120}")

pre_trades = [t for t in trades if t['period'] == 'pre_aug']
post_trades = [t for t in trades if t['period'] == 'post_aug']

print(f"\n  Pre-Aug:  {len(pre_trades)} trades, avg PnL {np.mean([t['pnl_eod'] for t in pre_trades]):+.2f}%")
print(f"  Post-Aug: {len(post_trades)} trades, avg PnL {np.mean([t['pnl_eod'] for t in post_trades]):+.2f}%")

print(f"\n  ── Feature means: Pre vs Post ──")
print(f"  {'Feature':<25s} {'Timing':<12s} {'Pre':>8s} {'Post':>8s} {'Δ':>8s} {'p':>8s}")
print(f"  {'-'*75}")

for feat_key, label, timing in CONTINUOUS_FEATURES:
    pre_vals = [t[feat_key] for t in pre_trades if t.get(feat_key) is not None and not np.isnan(t[feat_key])]
    post_vals = [t[feat_key] for t in post_trades if t.get(feat_key) is not None and not np.isnan(t[feat_key])]
    if len(pre_vals) < 5 or len(post_vals) < 5: continue
    pre_m = np.mean(pre_vals); post_m = np.mean(post_vals)
    delta = post_m - pre_m
    try:
        _, p = sp_stats.mannwhitneyu(pre_vals, post_vals, alternative='two-sided')
    except: p = 1.0
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
    print(f"  {label:<25s} {timing:<12s} {pre_m:+.4f}  {post_m:+.4f}  {delta:+.4f}  {p:.4f} {sig}")

print(f"\n  ── Categorical shifts ──")
for feat_key, label, timing in CATEGORICAL_FEATURES:
    print(f"\n  {label} ({timing}):")
    for period_lbl, period_trades in [('Pre-Aug', pre_trades), ('Post-Aug', post_trades)]:
        groups = defaultdict(list)
        for t in period_trades:
            groups[t.get(feat_key, 'unknown')].append(t['pnl_eod'])
        parts = []
        for grp in sorted(groups.keys()):
            a = np.array(groups[grp])
            parts.append(f"{grp}: n={len(a)} avg={np.mean(a):+.1f}%")
        print(f"    {period_lbl}: {' | '.join(parts)}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: TERCILE ANALYSIS — top features
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print(f"  PHASE 5: TERCILE ANALYSIS — top predictive features")
print(f"{'='*120}")

for feat_key, label, timing in CONTINUOUS_FEATURES:
    vals = [(t[feat_key], t['pnl_eod'], t['period']) for t in trades
            if t.get(feat_key) is not None and not np.isnan(t[feat_key]) and not np.isinf(t[feat_key])]
    if len(vals) < 15: continue
    x_all = [v[0] for v in vals]
    p33, p67 = np.percentile(x_all, 33), np.percentile(x_all, 67)
    if p33 == p67: continue

    terciles = {'T1 (low)': [], 'T2 (mid)': [], 'T3 (high)': []}
    for x, pnl, period in vals:
        if x <= p33: terciles['T1 (low)'].append((pnl, period))
        elif x <= p67: terciles['T2 (mid)'].append((pnl, period))
        else: terciles['T3 (high)'].append((pnl, period))

    avgs = {k: np.mean([v[0] for v in vs]) for k, vs in terciles.items() if vs}
    spread = avgs.get('T3 (high)', 0) - avgs.get('T1 (low)', 0)
    if abs(spread) < 1.0: continue  # skip weak features

    tradeable = 'EARLY+LATE' if timing in ('PRE_SCAN', 'PRE_EARLY') else 'LATE only' if timing == 'PRE_LATE' else 'NOT TRADEABLE'
    print(f"\n  {label} ({timing}) — T3-T1 spread: {spread:+.2f}pp — Usable for: {tradeable}")
    print(f"    Cutpoints: T1 ≤ {p33:.4f} | T2 ≤ {p67:.4f} | T3 > {p67:.4f}")
    rows = []
    for t_lbl in ['T1 (low)', 'T2 (mid)', 'T3 (high)']:
        vs = terciles[t_lbl]
        if not vs: continue
        pnls = np.array([v[0] for v in vs])
        n_post = sum(1 for v in vs if v[1] == 'post_aug')
        sh = np.mean(pnls) / np.std(pnls, ddof=1) if len(pnls) > 1 and np.std(pnls, ddof=1) > 0 else 0
        rows.append([t_lbl, len(pnls), n_post, f"{np.mean(pnls):+.2f}%",
                      f"{np.median(pnls):+.2f}%",
                      f"{float((pnls>0).mean())*100:.0f}%", f"{sh:.3f}",
                      f"${100000 * np.sum(pnls) / 100:+,.0f}"])
    print(tabulate(rows, headers=['Tercile', 'N', 'PostAug', 'Avg', 'Med', 'WR', 'Sharpe', '$NC'],
                   tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: TRADEABLE FILTER IDEAS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print(f"  PHASE 6: TRADEABLE FILTER IDEAS")
print(f"  Only filters using PRE_SCAN or PRE_EARLY features (no look-ahead)")
print(f"{'='*120}")

# Test each filter: compute PnL for PASS vs FAIL groups
FILTER_IDEAS = [
    # (name, timing, filter_fn, description)
    ('Skip green bar0',      'PRE_EARLY', lambda t: t.get('open_bar_direction', 0) != 1,
     'Skip if first RTH bar closes green (close > open)'),
    ('Require red bar0',     'PRE_EARLY', lambda t: t.get('open_bar_direction', 0) == -1,
     'Only trade if first RTH bar closes red'),
    ('PM retrace > 0.05',    'PRE_SCAN',  lambda t: t.get('pm_retrace') is not None and t['pm_retrace'] > 0.05,
     'Require some premarket retrace of the gap'),
    ('PM retrace > 0.08',    'PRE_SCAN',  lambda t: t.get('pm_retrace') is not None and t['pm_retrace'] > 0.08,
     'Stronger PM retrace requirement'),
    ('PM retrace > 0.12',    'PRE_SCAN',  lambda t: t.get('pm_retrace') is not None and t['pm_retrace'] > 0.12,
     'Aggressive PM retrace requirement'),
    ('PM trend < 0',         'PRE_SCAN',  lambda t: t.get('pm_trend') is not None and t['pm_trend'] < 0,
     'PM trending down (last PM close < first PM open)'),
    ('Skip T6 (15-30%)',     'PRE_SCAN',  lambda t: not (0.15 <= t.get('gap_pct_f', 0) < 0.30),
     'Skip gap tier 15-30%'),
    ('Skip T6a (10-15%)',    'PRE_SCAN',  lambda t: not (0.10 <= t.get('gap_pct_f', 0) < 0.15),
     'Skip gap tier 10-15%'),
    ('Gap < 15%',            'PRE_SCAN',  lambda t: t.get('gap_pct_f', 0) < 0.15,
     'Only trade gaps under 15%'),
    ('Gap < 10%',            'PRE_SCAN',  lambda t: t.get('gap_pct_f', 0) < 0.10,
     'Only trade gaps under 10%'),
    ('Gap consumed < 0.30',  'PRE_SCAN',  lambda t: t.get('gap_consumed_f', 0) < 0.30,
     'Tighter gap erosion threshold'),
    ('First 2bar drop',      'PRE_EARLY', lambda t: t.get('first_2bar_drop', 0) < 0,
     'First 2 RTH bars combined are negative (fade started)'),
    ('Bar0+Bar1 both red',   'PRE_EARLY', lambda t: t.get('open_bar_direction', 0) == -1 and t.get('bar1_direction', 0) == -1,
     'Both first two bars close red'),
    ('No fallback entry',    'PRE_EARLY', lambda t: t.get('entry_type') == 'local_high',
     'Only trade local_high entries, skip fallback bar 15'),
    ('PM retrace+red bar',   'PRE_EARLY', lambda t: (t.get('pm_retrace') is not None and t['pm_retrace'] > 0.08
                                                      and t.get('open_bar_direction', 0) == -1),
     'Combined: PM retrace > 0.08 AND first bar red'),
    ('Small gap + red bar',  'PRE_EARLY', lambda t: t.get('gap_pct_f', 0) < 0.15 and t.get('open_bar_direction', 0) == -1,
     'Gap < 15% AND first bar red'),
    ('Mon-Wed only',         'PRE_SCAN',  lambda t: t.get('dow_num', 0) <= 2,
     'Only trade Monday through Wednesday'),
    ('No Friday',            'PRE_SCAN',  lambda t: t.get('dow_num', 0) != 4,
     'Skip Friday trades'),
]

# For late entry (E2-B), also test PRE_LATE features
LATE_FILTER_IDEAS = [
    ('30m drop < -3%',       'PRE_LATE',  lambda t: t.get('drop_30m') is not None and t['drop_30m'] < -0.03,
     'First 30 minutes dropped > 3% from open'),
    ('30m drop < -5%',       'PRE_LATE',  lambda t: t.get('drop_30m') is not None and t['drop_30m'] < -0.05,
     'First 30 minutes dropped > 5% from open'),
    ('Bar30 below open',     'PRE_LATE',  lambda t: t.get('bar30_above_open') == 0,
     'Price at bar 30 is below open'),
    ('Negative accel',       'PRE_LATE',  lambda t: t.get('price_accel') is not None and t['price_accel'] < 0,
     'Price accelerating downward (mom_15_30 < mom_0_15)'),
    ('60m recovery < 0.5',   'PRE_LATE',  lambda t: t.get('recovery_60m') is not None and t['recovery_60m'] < 0.5,
     'Recovery at 60 min < 50% (same as SD classification)'),
    ('SD pattern',           'PRE_LATE',  lambda t: t.get('open_pattern') == 'sustained_drop',
     'Sustained drop confirmed at bar 60 — tradeable for LATE entry only'),
]

def _eval_filter(trades_list, filter_fn, label):
    passing = [t for t in trades_list if filter_fn(t)]
    failing = [t for t in trades_list if not filter_fn(t)]
    if len(passing) < 3: return None
    pa = np.array([t['pnl_eod'] for t in passing])
    fa = np.array([t['pnl_eod'] for t in failing]) if failing else np.array([0])
    sh_p = np.mean(pa) / np.std(pa, ddof=1) if len(pa) > 1 and np.std(pa, ddof=1) > 0 else 0
    sh_f = np.mean(fa) / np.std(fa, ddof=1) if len(fa) > 1 and np.std(fa, ddof=1) > 0 else 0
    pa_post = np.array([t['pnl_eod'] for t in passing if t['period'] == 'post_aug'])
    sh_post = np.mean(pa_post) / np.std(pa_post, ddof=1) if len(pa_post) > 1 and np.std(pa_post, ddof=1) > 0 else 0
    return {
        'label': label, 'n_pass': len(passing), 'n_fail': len(failing),
        'avg_pass': float(np.mean(pa)), 'avg_fail': float(np.mean(fa)) if len(fa) > 0 else 0,
        'sh_pass': sh_p, 'sh_fail': sh_f,
        'wr_pass': float((pa > 0).mean()), 'dollar_pass': float(100000 * np.sum(pa) / 100),
        'n_post': len(pa_post), 'avg_post': float(np.mean(pa_post)) if len(pa_post) > 0 else 0,
        'sh_post': sh_post,
        'dollar_post': float(100000 * np.sum(pa_post) / 100) if len(pa_post) > 0 else 0,
    }

print(f"\n  ── EARLY ENTRY filters (bars 3-11, usable for E2-A/Strategy B) ──")
print(f"  Using PRE_SCAN and PRE_EARLY features only\n")
early_results = []
for name, timing, fn, desc in FILTER_IDEAS:
    r = _eval_filter(trades, fn, name)
    if r: early_results.append(r)

early_results.sort(key=lambda x: x['sh_post'], reverse=True)
rows = []
for r in early_results:
    rows.append([
        r['label'], r['n_pass'], r['n_fail'],
        f"{r['avg_pass']:+.2f}%", f"{r['avg_fail']:+.2f}%",
        f"{r['wr_pass']*100:.0f}%", f"{r['sh_pass']:.3f}",
        f"${r['dollar_pass']:+,.0f}",
        f"| {r['n_post']}", f"{r['avg_post']:+.2f}%", f"{r['sh_post']:.3f}",
        f"${r['dollar_post']:+,.0f}",
    ])
print(tabulate(rows,
               headers=['Filter', 'Pass', 'Fail', 'Avg(P)', 'Avg(F)', 'WR(P)', 'Sh(P)',
                        '$(P)', '| PostN', 'PostAvg', 'PostSh', 'Post$'],
               tablefmt='simple', stralign='right'))

print(f"\n  ── LATE ENTRY filters (bars 60-72, usable for E2-B) ──")
print(f"  Using PRE_SCAN + PRE_EARLY + PRE_LATE features\n")
all_late_filters = FILTER_IDEAS + LATE_FILTER_IDEAS
late_results = []
for name, timing, fn, desc in all_late_filters:
    r = _eval_filter(trades, fn, name)
    if r: late_results.append(r)

late_results.sort(key=lambda x: x['sh_post'], reverse=True)
rows = []
for r in late_results:
    rows.append([
        r['label'], r['n_pass'], r['n_fail'],
        f"{r['avg_pass']:+.2f}%", f"{r['avg_fail']:+.2f}%",
        f"{r['wr_pass']*100:.0f}%", f"{r['sh_pass']:.3f}",
        f"${r['dollar_pass']:+,.0f}",
        f"| {r['n_post']}", f"{r['avg_post']:+.2f}%", f"{r['sh_post']:.3f}",
        f"${r['dollar_post']:+,.0f}",
    ])
print(tabulate(rows,
               headers=['Filter', 'Pass', 'Fail', 'Avg(P)', 'Avg(F)', 'WR(P)', 'Sh(P)',
                        '$(P)', '| PostN', 'PostAvg', 'PostSh', 'Post$'],
               tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

fig, axes = plt.subplots(3, 3, figsize=(24, 18))
fig.suptitle(f'Agent 1 — Feature Investigation (Bias-Free)\n'
             f'{BACKTEST_START} → {BACKTEST_END}  |  {len(trades)} trades  |  TOP_N={TOP_N}',
             fontsize=12, fontweight='bold')

pnls = np.array([t['pnl_eod'] for t in trades])

# 1. PnL by gap tier
ax = axes[0, 0]
tiers = defaultdict(list)
for t in trades: tiers[t['gap_tier']].append(t['pnl_eod'])
tier_order = sorted(tiers.keys())
ax.bar(range(len(tier_order)), [np.mean(tiers[k]) for k in tier_order], color='#2c3e50')
ax.set_xticks(range(len(tier_order))); ax.set_xticklabels(tier_order, rotation=45, fontsize=8)
ax.set_title('Avg PnL by Gap Tier (PRE_SCAN)'); ax.set_ylabel('Avg PnL %')
ax.axhline(0, color='#333', ls=':')

# 2. PnL by open bar direction
ax = axes[0, 1]
for lbl, val in [('Red (-1)', -1), ('Green (+1)', 1)]:
    sub = [t['pnl_eod'] for t in trades if t.get('open_bar_direction') == val]
    if sub: ax.hist(sub, bins=20, alpha=0.5, label=f'{lbl} n={len(sub)} avg={np.mean(sub):+.1f}%')
ax.set_title('PnL by Bar0 Direction (PRE_EARLY)'); ax.legend(fontsize=8)
ax.axvline(0, color='#333', lw=1)

# 3. PM retrace vs PnL scatter
ax = axes[0, 2]
pm_r = [(t['pm_retrace'], t['pnl_eod']) for t in trades if t.get('pm_retrace') is not None]
if pm_r:
    x, y = zip(*pm_r)
    colors = ['#e74c3c' if t['period'] == 'post_aug' else '#3498db'
              for t in trades if t.get('pm_retrace') is not None]
    ax.scatter(x, y, c=colors, alpha=0.5, s=15)
    ax.set_xlabel('PM Retrace'); ax.set_ylabel('PnL %')
    ax.set_title('PM Retrace vs PnL (PRE_SCAN)')
    ax.axhline(0, color='#333', ls=':'); ax.axvline(0.08, color='red', ls='--', alpha=0.5)

# 4. Pre-Aug vs Post-Aug by pattern
ax = axes[1, 0]
patterns = ['sustained_drop', 'drop_then_recover', 'mild_drop', 'flat_or_bounce']
x_idx = np.arange(len(patterns))
for i, (period, color) in enumerate([('pre_aug', '#3498db'), ('post_aug', '#e74c3c')]):
    avgs = []
    for pat in patterns:
        sub = [t['pnl_eod'] for t in trades if t['open_pattern'] == pat and t['period'] == period]
        avgs.append(np.mean(sub) if sub else 0)
    ax.bar(x_idx + i * 0.35, avgs, 0.35, label=period, color=color)
ax.set_xticks(x_idx + 0.175); ax.set_xticklabels([p[:15] for p in patterns], fontsize=8)
ax.set_title('Pattern PnL: Pre vs Post Aug (POST_ENTRY)'); ax.legend(fontsize=8)
ax.axhline(0, color='#333', ls=':')

# 5. First 2-bar drop vs PnL
ax = axes[1, 1]
drop2 = [(t['first_2bar_drop'], t['pnl_eod']) for t in trades if t.get('first_2bar_drop') is not None]
if drop2:
    x, y = zip(*drop2)
    ax.scatter(x, y, alpha=0.4, s=15, c='#2c3e50')
    ax.set_xlabel('First 2-bar drop'); ax.set_ylabel('PnL %')
    ax.set_title('First 2-Bar Drop vs PnL (PRE_EARLY)')
    ax.axhline(0, color='#333', ls=':'); ax.axvline(0, color='red', ls='--', alpha=0.5)

# 6. Gap consumed vs PnL
ax = axes[1, 2]
gc = [(t['gap_consumed_f'], t['pnl_eod']) for t in trades if t.get('gap_consumed_f', 0) > 0]
if gc:
    x, y = zip(*gc)
    ax.scatter(x, y, alpha=0.4, s=15, c='#8e44ad')
    ax.set_xlabel('Gap Consumed @Scan'); ax.set_ylabel('PnL %')
    ax.set_title('Gap Consumed vs PnL (PRE_SCAN)')
    ax.axhline(0, color='#333', ls=':')

# 7. Day of week
ax = axes[2, 0]
dow_pnl = defaultdict(list)
for t in trades: dow_pnl[t['dow']].append(t['pnl_eod'])
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
dow_avgs = [np.mean(dow_pnl.get(d, [0])) for d in dow_order]
dow_ns = [len(dow_pnl.get(d, [])) for d in dow_order]
ax.bar(range(5), dow_avgs, color='#27ae60')
ax.set_xticks(range(5)); ax.set_xticklabels([d[:3] for d in dow_order])
for i, n in enumerate(dow_ns): ax.annotate(f'n={n}', (i, dow_avgs[i]), ha='center', va='bottom', fontsize=7)
ax.set_title('Avg PnL by Day of Week (PRE_SCAN)'); ax.axhline(0, color='#333', ls=':')

# 8. Entry type
ax = axes[2, 1]
et_pnl = defaultdict(list)
for t in trades: et_pnl[t['entry_type']].append(t['pnl_eod'])
for i, et in enumerate(['local_high', 'fallback']):
    sub = et_pnl.get(et, [])
    if sub: ax.hist(sub, bins=20, alpha=0.5, label=f'{et} n={len(sub)} avg={np.mean(sub):+.1f}%')
ax.set_title('PnL by Entry Type (PRE_EARLY)'); ax.legend(fontsize=8)
ax.axvline(0, color='#333', lw=1)

# 9. Monthly PnL timeline
ax = axes[2, 2]
monthly = defaultdict(list)
for t in trades: monthly[t['scan_date'][:7]].append(t['pnl_eod'])
months = sorted(monthly.keys())
ax.bar(range(len(months)), [np.mean(monthly[m]) for m in months], color='#e67e22')
ax.set_xticks(range(len(months))); ax.set_xticklabels(months, rotation=45, fontsize=8)
ax.set_title('Avg PnL by Month'); ax.axhline(0, color='#333', ls=':')

plt.tight_layout(rect=[0, 0, 1, 0.94])
p1 = os.path.join(BASE_DIR, 'agent1_feature_investigation.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {p1}")

print(f"\n\n  Total runtime: {time.time() - t_total:.0f}s ({(time.time() - t_total)/60:.1f} min)")
print(f"\n  NEXT: Use the filter rankings from Phase 6 to build tradeable strategy")
print(f"        variants in agent1_baseline_bt.py. Only PRE_SCAN and PRE_EARLY")
print(f"        filters are usable for early entry (bars 3-11). PRE_LATE filters")
print(f"        are only usable for E2-B late entry (bars 60-72).")
