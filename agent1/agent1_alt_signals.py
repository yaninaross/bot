# %% Agent 1 — Alternative N=1 Selection Signals
#
# Run AFTER backtest_e2_reproduction.py in same Colab session to reuse cached data.
# If run standalone, pipeline executes automatically (~10 min).
#
# Tests 5 alternative ranking methods vs the baseline (gap_pct):
#   R1: Volume Spike (today vol / 20d avg vol)
#   R2: Idiosyncratic Gap (stock gap minus sector ETF gap)
#   R3: PM Momentum Reversal (pre-market retrace from PM high)
#   R4: Price Extension (distance from 20d high)
#   R5: Combined Score (equal-weight R1-R4)
#
# Output: comparison tables (full period + Jul-Sep), overlap analysis, charts.

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
# CONFIG
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

# Jul-Sep cutoff for sub-period analysis
JULSEP_START   = '2026-07-01'

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
# SECTOR MAPPING — ticker to sector ETF
# ═══════════════════════════════════════════════════════════════════════════════

_TICKER_SECTOR = {
    # Technology (XLK)
    'AAPL':'XLK','ADBE':'XLK','ALAB':'XLK','AMAT':'XLK','AMD':'XLK','ARM':'XLK',
    'CRDO':'XLK','DELL':'XLK','HPE':'XLK','HPQ':'XLK','INTC':'XLK','META':'XLK',
    'MU':'XLK','NVDA':'XLK','ORCL':'XLK','PLTR':'XLK','SMCI':'XLK','STX':'XLK',
    'SWKS':'XLK','TTD':'XLK','ACMR':'XLK','AXTI':'XLK','MXL':'XLK','SSNC':'XLK',
    'GDDY':'XLK','CPRT':'XLK',
    # Software / Internet (XLK)
    'ASAN':'XLK','BRZE':'XLK','COIN':'XLK','HOOD':'XLK','RDDT':'XLK','SOUN':'XLK',
    'IONQ':'XLK','QBTS':'XLK','RGTI':'XLK','UPST':'XLK',
    # Financials (XLF)
    'BNC':'XLF','SIG':'XLF',
    # Energy (XLE)
    'APA':'XLE','CVX':'XLE','ET':'XLE','FCX':'XLE','HBM':'XLE','SCCO':'XLE',
    'SDRL':'XLE','SSL':'XLE',
    # Healthcare (XLV)
    'ATEC':'XLV','AXGN':'XLV','BBIO':'XLV','BEAM':'XLV','BHVN':'XLV','COO':'XLV',
    'EXEL':'XLV','INBX':'XLV','MRNA':'XLV','NTLA':'XLV','REGN':'XLV','SRPT':'XLV',
    'NVS':'XLV','PRLD':'XLV','ROIV':'XLV','TYRA':'XLV',
    # Industrials (XLI)
    'ABM':'XLI','AGCO':'XLI','AME':'XLI','ATI':'XLI','AVAV':'XLI','CAT':'XLI',
    'DOV':'XLI','ETN':'XLI','IESC':'XLI','MAS':'XLI','RTX':'XLI','STRL':'XLI',
    'URI':'XLI','DNOW':'XLI','IP':'XLI','SNX':'XLI',
    # Consumer Discretionary (XLY)
    'ACHR':'XLY','DKS':'XLY','EAT':'XLY','GME':'XLY','GOOS':'XLY','LULU':'XLY',
    'M':'XLY','PLNT':'XLY','RH':'XLY','TSLA':'XLY','UBER':'XLY','ZUMZ':'XLY',
    'SHOE':'XLY','FLWS':'XLY','LOVE':'XLY','MCFT':'XLY',
    # Consumer Staples (XLP)
    'KR':'XLP','UNFI':'XLP','FIZZ':'XLP',
    # Utilities (XLU)
    'EVRG':'XLU','NEE':'XLU','VST':'XLU',
    # Real Estate (XLRE)
    'VAC':'XLRE','HGTY':'XLRE',
    # Communications (XLC)
    'BABA':'XLC',
    # Materials (XLB)
    'AXTA':'XLB','CC':'XLB',
    # Crypto / Mining (use XLK proxy)
    'MSTR':'XLK','MARA':'XLK','RIOT':'XLK','CLSK':'XLK','CIFR':'XLK','CORZ':'XLK',
    'HUT':'XLK','IREN':'XLK','WULF':'XLK','BTBT':'XLK','CAN':'XLK','KEEL':'XLK',
    # Clean Energy / Nuclear (use XLE proxy)
    'BE':'XLE','FCEL':'XLE','FLNC':'XLE','NNE':'XLE','OKLO':'XLE','PLUG':'XLE',
    'SMR':'XLE','VRT':'XLE',
    # Space / Defense (use XLI proxy)
    'ASTS':'XLI','KTOS':'XLI','LUNR':'XLI','PL':'XLI','RKLB':'XLI','RDW':'XLI',
    'NBIS':'XLI',
    # EV / Mobility
    'APLD':'XLK',
}

_DEFAULT_SECTOR_ETF = 'SPY'

# Sector ETFs to fetch daily data for
_SECTOR_ETFS = {'XLK','XLF','XLE','XLV','XLI','XLC','XLY','XLP','XLU','XLRE','XLB','SPY'}


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

def _prior_n_td(d, n=20):
    """Return list of n prior trading days (most recent first)."""
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    out = []
    cur = d - timedelta(days=1)
    for _ in range(60):
        if _is_td(cur):
            out.append(cur.strftime('%Y-%m-%d'))
            if len(out) >= n: return out
        cur -= timedelta(days=1)
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
    print(f"  ALT SIGNALS — Data Pipeline")
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
# FETCH PRIOR 20-DAY DATA FOR SIGNALS (volume avg, 20d high)
# Ensures _gd_cache has data for the 20 trading days before each date.
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  Fetching 20d lookback daily data for signal computation...")
_lookback_dates_needed = set()
for dt in dates:
    prior_days = _prior_n_td(dt, 20)
    for pd_str in prior_days:
        if pd_str not in _gd_cache:
            _lookback_dates_needed.add(pd_str)

if _lookback_dates_needed:
    print(f"  -> {len(_lookback_dates_needed)} dates to fetch")
    for i, od in enumerate(sorted(_lookback_dates_needed)):
        try: fetch_grouped_daily(od)
        except: pass
        if (i+1) % 20 == 0:
            print(f"    {i+1}/{len(_lookback_dates_needed)}...")
    print(f"  -> Done, _gd_cache has {len(_gd_cache)} dates")
else:
    print(f"  -> All lookback data already cached ({len(_gd_cache)} dates)")


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD QUALIFIED CANDIDATE POOL PER DAY
# Same filter as build_e2_trades_enriched: CS + sustained_drop + entry + bars
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print(f"  Building qualified candidate pools per day...")
print(f"{'='*80}")

# For each day, build the full pool of shortable candidates
# (not just the top-1 by gap_pct, but ALL qualified)
daily_pools = {}
for dt in dates:
    cands = all_candidates.get(dt, [])
    pool = []
    for c in cands[:PRE_FILTER_N]:
        tk = c['ticker']
        ref = _ticker_ref.get(tk, {})
        if ref.get('type', 'UNK') != 'CS':
            continue
        if pattern_cache.get((tk, dt), 'no_data') != 'sustained_drop':
            continue
        entry = entry_cache.get((tk, dt))
        if entry is None:
            continue
        bar_data = all_bars.get((tk, dt))
        if bar_data is None:
            continue
        cc = copy.deepcopy(c)
        cc['entry_price'] = entry['entry_price']
        cc['entry_time'] = entry['entry_time']
        cc['entry_type'] = entry['entry_type']
        pool.append(cc)
    daily_pools[dt] = pool

n_with_pool = sum(1 for p in daily_pools.values() if len(p) > 0)
total_pool = sum(len(p) for p in daily_pools.values())
multi_pool = sum(1 for p in daily_pools.values() if len(p) > 1)
print(f"  {n_with_pool} days have at least 1 candidate (total {total_pool})")
print(f"  {multi_pool} days have 2+ candidates (ranking matters)")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE EXIT PNL FOR A CANDIDATE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_exit_pnl(tk, dt, entry_price, entry_time):
    """Compute PnL at 17:00 for a SHORT trade. Returns pnl_pct or None."""
    bar_data = all_bars.get((tk, dt))
    if bar_data is None:
        return None
    rth_bars = bar_data.get('rth', [])
    ah_bars = bar_data.get('ah', [])
    session_bars = rth_bars + ah_bars
    exit_price = None
    for t_str, bar in session_bars:
        if t_str > EXIT_TIME:
            break
        if t_str >= entry_time:
            exit_price = bar['c']
    if exit_price is None:
        daily = _gd_cache.get(dt, {}).get(tk, {})
        if daily.get('c'):
            exit_price = daily['c']
    if exit_price is None:
        return None
    return round((entry_price - exit_price) / entry_price * 100, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE ALTERNATIVE SIGNALS PER CANDIDATE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print(f"  Computing alternative signals for all candidates...")
print(f"{'='*80}")


def compute_vol_ratio(tk, dt):
    """R1: Volume Spike — today_volume / prior_20d_avg_volume."""
    cand_data = None
    for c in all_candidates.get(dt, []):
        if c['ticker'] == tk:
            cand_data = c
            break
    if cand_data is None:
        return None
    vol_today = cand_data.get('volume_today', 0)
    if vol_today <= 0:
        return None
    prior_days = _prior_n_td(dt, 20)
    vols = []
    for pd_str in prior_days:
        gd = _gd_cache.get(pd_str, {})
        if tk in gd and gd[tk].get('v', 0) > 0:
            vols.append(gd[tk]['v'])
    if len(vols) < 5:
        return None
    avg_vol = np.mean(vols)
    if avg_vol <= 0:
        return None
    return vol_today / avg_vol


def compute_idio_gap(tk, dt, gap_pct):
    """R2: Idiosyncratic Gap — stock gap minus sector ETF gap."""
    sector_etf = _TICKER_SECTOR.get(tk, _DEFAULT_SECTOR_ETF)
    prior_str = _prior_td(dt).strftime('%Y-%m-%d')
    prior_gd = _gd_cache.get(prior_str, {})
    today_gd = _gd_cache.get(dt, {})
    if sector_etf not in today_gd or sector_etf not in prior_gd:
        return gap_pct  # fallback: assume sector gap is 0
    etf_open = today_gd[sector_etf].get('o', 0)
    etf_prior_c = prior_gd[sector_etf].get('c', 0)
    if etf_prior_c <= 0 or etf_open <= 0:
        return gap_pct
    sector_gap = (etf_open - etf_prior_c) / etf_prior_c
    return gap_pct - sector_gap


def compute_pm_retrace(tk, dt, prior_close):
    """R3: PM Momentum Reversal — how much has stock already faded in pre-market."""
    bar_data = all_bars.get((tk, dt))
    if bar_data is None:
        return None
    pm_bars = bar_data.get('pm', [])
    if len(pm_bars) < 3:
        return None
    pm_high = max(b['h'] for _, b in pm_bars)
    last_pm_close = pm_bars[-1][1]['c']
    if pm_high <= prior_close:
        return None  # no gap-up in PM
    retrace_denom = pm_high - prior_close
    if retrace_denom <= 0:
        return None
    pm_retrace = (pm_high - last_pm_close) / retrace_denom
    return pm_retrace


def compute_price_extension(tk, dt):
    """R4: Price Extension — distance from 20d high."""
    today_gd = _gd_cache.get(dt, {})
    if tk not in today_gd:
        return None
    open_price = today_gd[tk].get('o', 0)
    if open_price <= 0:
        return None
    prior_days = _prior_n_td(dt, 20)
    highs = []
    for pd_str in prior_days:
        gd = _gd_cache.get(pd_str, {})
        if tk in gd and gd[tk].get('h', 0) > 0:
            highs.append(gd[tk]['h'])
    if len(highs) < 5:
        return None
    high_20d = max(highs)
    if high_20d <= 0:
        return None
    return (open_price - high_20d) / high_20d


# Attach signals to all pool candidates
for dt in dates:
    pool = daily_pools.get(dt, [])
    for c in pool:
        tk = c['ticker']
        c['r1_vol_ratio'] = compute_vol_ratio(tk, dt)
        c['r2_idio_gap'] = compute_idio_gap(tk, dt, c['gap_pct'])
        c['r3_pm_retrace'] = compute_pm_retrace(tk, dt, c['prior_close'])
        c['r4_price_ext'] = compute_price_extension(tk, dt)
        c['pnl_eod'] = compute_exit_pnl(tk, dt, c['entry_price'], c['entry_time'])

# Count signal availability
sig_counts = {'r1': 0, 'r2': 0, 'r3': 0, 'r4': 0, 'pnl': 0, 'total': 0}
for dt in dates:
    for c in daily_pools.get(dt, []):
        sig_counts['total'] += 1
        if c['r1_vol_ratio'] is not None: sig_counts['r1'] += 1
        if c['r2_idio_gap'] is not None: sig_counts['r2'] += 1
        if c['r3_pm_retrace'] is not None: sig_counts['r3'] += 1
        if c['r4_price_ext'] is not None: sig_counts['r4'] += 1
        if c['pnl_eod'] is not None: sig_counts['pnl'] += 1

print(f"\n  Signal availability (out of {sig_counts['total']} candidates):")
print(f"    R1 vol_ratio:   {sig_counts['r1']:>4d} ({sig_counts['r1']/max(sig_counts['total'],1)*100:.0f}%)")
print(f"    R2 idio_gap:    {sig_counts['r2']:>4d} ({sig_counts['r2']/max(sig_counts['total'],1)*100:.0f}%)")
print(f"    R3 pm_retrace:  {sig_counts['r3']:>4d} ({sig_counts['r3']/max(sig_counts['total'],1)*100:.0f}%)")
print(f"    R4 price_ext:   {sig_counts['r4']:>4d} ({sig_counts['r4']/max(sig_counts['total'],1)*100:.0f}%)")
print(f"    PnL available:  {sig_counts['pnl']:>4d} ({sig_counts['pnl']/max(sig_counts['total'],1)*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# RANKING METHODS — select N=1 per day
# ═══════════════════════════════════════════════════════════════════════════════

def select_by_ranking(daily_pools, dates, ranking_key, descending=True,
                      combined_mode=False):
    """Select top-1 candidate per day by ranking_key.
    Returns list of trade dicts with pnl_eod.
    """
    trades = []
    for dt in dates:
        pool = daily_pools.get(dt, [])
        # Filter to candidates with valid PnL and valid signal
        if combined_mode:
            valid = [c for c in pool if c.get('pnl_eod') is not None]
        else:
            valid = [c for c in pool
                     if c.get('pnl_eod') is not None
                     and c.get(ranking_key) is not None]
        if not valid:
            continue

        if combined_mode:
            # Compute min-max normalized combined score across today's candidates
            r_keys = ['r1_vol_ratio', 'r2_idio_gap', 'r3_pm_retrace', 'r4_price_ext']
            for c in valid:
                c['r5_combined'] = 0.0
            for rk in r_keys:
                vals = [c.get(rk) for c in valid if c.get(rk) is not None]
                if not vals or len(vals) < 1:
                    continue
                vmin, vmax = min(vals), max(vals)
                rng = vmax - vmin if vmax > vmin else 1.0
                for c in valid:
                    v = c.get(rk)
                    if v is not None:
                        c['r5_combined'] += 0.25 * (v - vmin) / rng
                    # If signal missing, contributes 0
            best = max(valid, key=lambda c: c.get('r5_combined', 0))
        elif descending:
            best = max(valid, key=lambda c: c[ranking_key])
        else:
            best = min(valid, key=lambda c: c[ranking_key])

        tc = copy.deepcopy(best)
        tc['selected_date'] = dt
        trades.append(tc)
    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL RANKINGS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print(f"  Running all ranking methods...")
print(f"{'='*80}")

RANKINGS = [
    ('Baseline (gap_pct)', 'gap_pct', True, False),
    ('R1: Vol Spike',      'r1_vol_ratio', True, False),
    ('R2: Idio Gap',       'r2_idio_gap', True, False),
    ('R3: PM Retrace',     'r3_pm_retrace', True, False),
    ('R4: Price Ext',      'r4_price_ext', True, False),
    ('R5: Combined',       None, True, True),
]

all_results = {}
for label, key, desc, combined in RANKINGS:
    trades = select_by_ranking(daily_pools, dates, key, desc, combined)
    pnls = [t['pnl_eod'] for t in trades if t['pnl_eod'] is not None]
    m = compute_metrics(pnls, label)
    total_dol = sum(CAPITAL * p / 100 for p in pnls)

    # Jul-Sep subset
    js_pnls = [t['pnl_eod'] for t in trades
                if t['pnl_eod'] is not None and t['scan_date'] >= JULSEP_START]
    js_m = compute_metrics(js_pnls, f'{label} Jul-Sep')
    js_total_dol = sum(CAPITAL * p / 100 for p in js_pnls)

    all_results[label] = {
        'trades': trades,
        'metrics': m,
        'total_dol': round(total_dol),
        'js_metrics': js_m,
        'js_total_dol': round(js_total_dol),
    }
    print(f"  {label:25s}  n={m['n']:>3d}  avg={m['avg']:>+7.2f}%  "
          f"Sharpe={m['sharpe']:>6.3f}  ${total_dol:>+10,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE 1: FULL PERIOD (Jan-Sep)
# ═══════════════════════════════════════════════════════════════════════════════

base_res = all_results['Baseline (gap_pct)']
base_dol = base_res['total_dol']

print(f"\n\n{'='*80}")
print(f"  FULL PERIOD ({BACKTEST_START} -> {BACKTEST_END})")
print(f"{'='*80}\n")

print(f"  {'Ranking':25s}  {'n':>4s}  {'Avg PnL':>8s}  {'WR':>5s}  "
      f"{'Sharpe':>7s}  {'Total $':>12s}  {'vs Baseline':>12s}")
print(f"  {'~'*25}  {'~'*4}  {'~'*8}  {'~'*5}  "
      f"{'~'*7}  {'~'*12}  {'~'*12}")

for label, _, _, _ in RANKINGS:
    r = all_results[label]
    m = r['metrics']
    delta = r['total_dol'] - base_dol
    delta_str = f"${delta:>+10,.0f}" if label != 'Baseline (gap_pct)' else "---"
    print(f"  {label:25s}  {m['n']:>4d}  {m['avg']:>+7.2f}%  {m['wr']*100:>4.0f}%  "
          f"{m['sharpe']:>7.3f}  ${r['total_dol']:>+11,.0f}  {delta_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE 2: JUL-SEP ONLY
# ═══════════════════════════════════════════════════════════════════════════════

base_js_dol = base_res['js_total_dol']

print(f"\n\n{'='*80}")
print(f"  JUL-SEP ONLY ({JULSEP_START} -> {BACKTEST_END})")
print(f"{'='*80}\n")

print(f"  {'Ranking':25s}  {'n':>4s}  {'Avg PnL':>8s}  {'WR':>5s}  "
      f"{'Sharpe':>7s}  {'Total $':>12s}  {'vs Baseline':>12s}")
print(f"  {'~'*25}  {'~'*4}  {'~'*8}  {'~'*5}  "
      f"{'~'*7}  {'~'*12}  {'~'*12}")

for label, _, _, _ in RANKINGS:
    r = all_results[label]
    m = r['js_metrics']
    delta = r['js_total_dol'] - base_js_dol
    delta_str = f"${delta:>+10,.0f}" if label != 'Baseline (gap_pct)' else "---"
    print(f"  {label:25s}  {m['n']:>4d}  {m['avg']:>+7.2f}%  {m['wr']*100:>4.0f}%  "
          f"{m['sharpe']:>7.3f}  ${r['js_total_dol']:>+11,.0f}  {delta_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# OVERLAP ANALYSIS — how often does each alt pick the same ticker as baseline?
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  OVERLAP ANALYSIS — When does alternative ranking diverge from baseline?")
print(f"{'='*80}\n")

# Build baseline date->ticker map
base_trades = all_results['Baseline (gap_pct)']['trades']
base_map = {t['scan_date']: t for t in base_trades}

print(f"  {'Ranking':25s}  {'Same':>5s}  {'Diff':>5s}  {'Overlap':>8s}  "
      f"{'Diff better':>11s}  {'Diff worse':>10s}  {'Avg gain':>9s}")
print(f"  {'~'*25}  {'~'*5}  {'~'*5}  {'~'*8}  "
      f"{'~'*11}  {'~'*10}  {'~'*9}")

overlap_details = {}
for label, _, _, _ in RANKINGS:
    if label == 'Baseline (gap_pct)':
        continue
    alt_trades = all_results[label]['trades']
    alt_map = {t['scan_date']: t for t in alt_trades}

    same, diff = 0, 0
    diff_better, diff_worse = 0, 0
    diff_pnl_delta = []

    common_dates = set(base_map.keys()) & set(alt_map.keys())
    for dt in common_dates:
        bt = base_map[dt]
        at = alt_map[dt]
        if bt['ticker'] == at['ticker']:
            same += 1
        else:
            diff += 1
            b_pnl = bt.get('pnl_eod', 0) or 0
            a_pnl = at.get('pnl_eod', 0) or 0
            delta_pnl = a_pnl - b_pnl
            diff_pnl_delta.append(delta_pnl)
            if a_pnl > b_pnl:
                diff_better += 1
            elif a_pnl < b_pnl:
                diff_worse += 1

    total = same + diff
    overlap_pct = same / total * 100 if total > 0 else 0
    avg_gain = np.mean(diff_pnl_delta) if diff_pnl_delta else 0

    overlap_details[label] = {
        'same': same, 'diff': diff, 'overlap_pct': round(overlap_pct, 1),
        'diff_better': diff_better, 'diff_worse': diff_worse,
        'avg_gain_on_diff': round(avg_gain, 2),
    }

    print(f"  {label:25s}  {same:>5d}  {diff:>5d}  {overlap_pct:>6.1f}%  "
          f"{diff_better:>11d}  {diff_worse:>10d}  {avg_gain:>+8.2f}%")

print(f"\n  WHAT THIS SAYS:")
print(f"  'Same'  = days where this ranking picked the same ticker as gap_pct baseline")
print(f"  'Diff'  = days where a different ticker was selected")
print(f"  'Diff better' = of divergent picks, alt outperformed baseline")
print(f"  'Avg gain' = average PnL improvement on days where picks diverged")


# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM LINE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  BOTTOM LINE")
print(f"{'='*80}")

# Find best full-period alternative
best_full_label = None
best_full_dol = base_dol
for label, _, _, _ in RANKINGS:
    if label == 'Baseline (gap_pct)':
        continue
    if all_results[label]['total_dol'] > best_full_dol:
        best_full_dol = all_results[label]['total_dol']
        best_full_label = label

# Find best Jul-Sep alternative
best_js_label = None
best_js_dol = base_js_dol
for label, _, _, _ in RANKINGS:
    if label == 'Baseline (gap_pct)':
        continue
    if all_results[label]['js_total_dol'] > best_js_dol:
        best_js_dol = all_results[label]['js_total_dol']
        best_js_label = label

print(f"\n  FULL PERIOD:")
if best_full_label:
    delta = best_full_dol - base_dol
    print(f"    Best alternative: {best_full_label}")
    print(f"    ${best_full_dol:+,.0f} vs baseline ${base_dol:+,.0f} (delta: ${delta:+,.0f})")
else:
    print(f"    NO alternative beats baseline over full period.")

print(f"\n  JUL-SEP:")
if best_js_label:
    delta = best_js_dol - base_js_dol
    print(f"    Best alternative: {best_js_label}")
    print(f"    ${best_js_dol:+,.0f} vs baseline ${base_js_dol:+,.0f} (delta: ${delta:+,.0f})")
else:
    print(f"    NO alternative beats baseline in Jul-Sep.")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


# ─── Chart 1: Avg PnL by ranking method — grouped bars (full vs Jul-Sep) ────

fig, ax = plt.subplots(figsize=(14, 7))

labels_short = ['Baseline', 'R1:Vol', 'R2:Idio', 'R3:PM', 'R4:Ext', 'R5:Comb']
x = np.arange(len(labels_short))
width = 0.35

full_avgs = [all_results[lbl]['metrics']['avg'] for lbl, _, _, _ in RANKINGS]
js_avgs = [all_results[lbl]['js_metrics']['avg'] for lbl, _, _, _ in RANKINGS]

bars1 = ax.bar(x - width/2, full_avgs, width, label='Full Period (Jan-Sep)',
               color='#2c3e50', alpha=0.85, edgecolor='black', linewidth=0.3)
bars2 = ax.bar(x + width/2, js_avgs, width, label='Jul-Sep Only',
               color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.3)

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels_short, fontsize=10)
ax.set_ylabel('Average PnL %', fontsize=11)
ax.set_title('Average PnL by Ranking Method\nFull Period vs Jul-Sep',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

for b, v in zip(bars1, full_avgs):
    ax.text(b.get_x() + b.get_width()/2, v + (0.3 if v >= 0 else -0.8),
            f'{v:+.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
for b, v in zip(bars2, js_avgs):
    ax.text(b.get_x() + b.get_width()/2, v + (0.3 if v >= 0 else -0.8),
            f'{v:+.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
p1 = os.path.join(BASE_DIR, 'alt_signals_avg_pnl.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {p1}")


# ─── Chart 2: Sharpe by ranking method ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 7))

full_sharpes = [all_results[lbl]['metrics']['sharpe'] for lbl, _, _, _ in RANKINGS]
js_sharpes = [all_results[lbl]['js_metrics']['sharpe'] for lbl, _, _, _ in RANKINGS]

bars1 = ax.bar(x - width/2, full_sharpes, width, label='Full Period (Jan-Sep)',
               color='#2c3e50', alpha=0.85, edgecolor='black', linewidth=0.3)
bars2 = ax.bar(x + width/2, js_sharpes, width, label='Jul-Sep Only',
               color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.3)

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels_short, fontsize=10)
ax.set_ylabel('Sharpe Ratio', fontsize=11)
ax.set_title('Sharpe Ratio by Ranking Method\nFull Period vs Jul-Sep',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

for b, v in zip(bars1, full_sharpes):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for b, v in zip(bars2, js_sharpes):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
p2 = os.path.join(BASE_DIR, 'alt_signals_sharpe.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {p2}")


# ─── Chart 3: Best alternative per-trade scatter vs baseline ────────────────

# Find the best alternative (full period total $)
best_alt_label = best_full_label or best_js_label
if best_alt_label is None:
    # Just pick R5 as a comparison
    best_alt_label = 'R5: Combined'

alt_trades = all_results[best_alt_label]['trades']
alt_map_for_scatter = {t['scan_date']: t for t in alt_trades}

scatter_dates = sorted(set(base_map.keys()) & set(alt_map_for_scatter.keys()))
base_pnls_sc = []
alt_pnls_sc = []
scatter_labels = []
for dt in scatter_dates:
    bp = base_map[dt].get('pnl_eod')
    ap = alt_map_for_scatter[dt].get('pnl_eod')
    if bp is not None and ap is not None:
        base_pnls_sc.append(bp)
        alt_pnls_sc.append(ap)
        scatter_labels.append(dt)

fig, ax = plt.subplots(figsize=(10, 10))

base_arr = np.array(base_pnls_sc)
alt_arr = np.array(alt_pnls_sc)

# Color by which is better
colors = []
for b, a in zip(base_pnls_sc, alt_pnls_sc):
    if abs(a - b) < 0.01:  # same ticker
        colors.append('#7f8c8d')
    elif a > b:
        colors.append('#27ae60')  # alt better
    else:
        colors.append('#e74c3c')  # baseline better

ax.scatter(base_pnls_sc, alt_pnls_sc, c=colors, alpha=0.6, s=40, edgecolors='black',
           linewidth=0.3)

# Diagonal line
lim_min = min(min(base_pnls_sc), min(alt_pnls_sc)) - 5
lim_max = max(max(base_pnls_sc), max(alt_pnls_sc)) + 5
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.4, label='Equal PnL')
ax.axhline(0, color='black', lw=0.5, alpha=0.3)
ax.axvline(0, color='black', lw=0.5, alpha=0.3)

ax.set_xlabel(f'Baseline (gap_pct) PnL %', fontsize=11)
ax.set_ylabel(f'{best_alt_label} PnL %', fontsize=11)
ax.set_title(f'Per-Trade Comparison: Baseline vs {best_alt_label}\n'
             f'Green = alt wins, Red = baseline wins, Gray = same pick',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_aspect('equal')

plt.tight_layout()
p3 = os.path.join(BASE_DIR, 'alt_signals_scatter.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {p3}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

results_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)

output = {
    'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'period': f'{BACKTEST_START} to {BACKTEST_END}',
    'julsep_start': JULSEP_START,
    'rankings': {},
    'overlap': overlap_details,
}

for label, _, _, _ in RANKINGS:
    r = all_results[label]
    m = r['metrics']
    js_m = r['js_metrics']
    output['rankings'][label] = {
        'full_period': {
            'n': m['n'], 'avg': round(m['avg'], 2), 'med': round(m['med'], 2),
            'wr': round(m['wr'], 3), 'sharpe': round(m['sharpe'], 3),
            'pf': round(m['pf'], 2) if m['pf'] != float('inf') else 'inf',
            'total_dollars': r['total_dol'],
            'ci_lo': round(m['ci_lo'], 2), 'ci_hi': round(m['ci_hi'], 2),
            'max_loss': round(m['max_loss'], 2), 'max_gain': round(m['max_gain'], 2),
        },
        'jul_sep': {
            'n': js_m['n'], 'avg': round(js_m['avg'], 2), 'med': round(js_m['med'], 2),
            'wr': round(js_m['wr'], 3), 'sharpe': round(js_m['sharpe'], 3),
            'pf': round(js_m['pf'], 2) if js_m['pf'] != float('inf') else 'inf',
            'total_dollars': r['js_total_dol'],
        },
        'trades': [{
            'date': t['scan_date'], 'ticker': t['ticker'],
            'gap_pct': round(t['gap_pct'] * 100, 2),
            'entry_price': round(t['entry_price'], 2),
            'pnl_eod': t['pnl_eod'],
            'r1_vol_ratio': round(t['r1_vol_ratio'], 2) if t.get('r1_vol_ratio') is not None else None,
            'r2_idio_gap': round(t['r2_idio_gap'] * 100, 2) if t.get('r2_idio_gap') is not None else None,
            'r3_pm_retrace': round(t['r3_pm_retrace'], 3) if t.get('r3_pm_retrace') is not None else None,
            'r4_price_ext': round(t['r4_price_ext'] * 100, 2) if t.get('r4_price_ext') is not None else None,
        } for t in r['trades']],
    }

save_path = os.path.join(results_dir, 'alternative_signals_jul_sep.json')
with open(save_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  Saved: {save_path}")


print(f"\n{'='*80}")
print(f"  DONE — Alternative signal analysis complete")
print(f"{'='*80}")
