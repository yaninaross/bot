# %% Agent 1 — Regime Analysis: Market Conditions vs Trade Performance
#
# Run AFTER backtest_e2_reproduction.py or agent1_eval_sl_skip.py in same
# Colab session to reuse cached data. If run standalone, pipeline executes
# automatically (~10 min).
#
# Questions answered:
#   1. Does SPY regime (bull/flat/bear) explain Agent 1 edge?
#   2. Does VIX level predict trade quality?
#   3. Did regime shift in Jul-Sep cause performance decay?
#   4. Can a simple regime gate improve risk-adjusted returns?

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
# SECTOR MAPPING — ticker → sector ETF
# ═══════════════════════════════════════════════════════════════════════════════

_SECTOR_MAP = {
    # Technology (XLK)
    'AAPL':'XLK','ADBE':'XLK','AMD':'XLK','AMAT':'XLK','ARM':'XLK',
    'CRDO':'XLK','DELL':'XLK','GDDY':'XLK','HPE':'XLK','HPQ':'XLK',
    'INTC':'XLK','MU':'XLK','NVDA':'XLK','ORCL':'XLK','PLTR':'XLK',
    'SMCI':'XLK','STX':'XLK','SWKS':'XLK','TTD':'XLK','MXL':'XLK',
    'MVIS':'XLK','SOUN':'XLK','SSNC':'XLK','ACMR':'XLK','AXTI':'XLK',
    'META':'XLK','IONQ':'XLK','QBTS':'XLK','RGTI':'XLK','NBIS':'XLK',
    'VRT':'XLK','VST':'XLK','ALAB':'XLK','ASAN':'XLK','BRZE':'XLK',
    'UPST':'XLK','APLD':'XLK','VICR':'XLK','ACVA':'XLK',
    # Crypto-adjacent (XLK proxy — no dedicated sector ETF)
    'COIN':'XLK','HOOD':'XLK','MSTR':'XLK','MARA':'XLK','RIOT':'XLK',
    'CLSK':'XLK','HUT':'XLK','BTBT':'XLK','CIFR':'XLK','CORZ':'XLK',
    'IREN':'XLK','WULF':'XLK','CAN':'XLK','KEEL':'XLK',
    # Energy / Clean Energy (XLE)
    'APA':'XLE','CVX':'XLE','ET':'XLE','SDRL':'XLE','PLUG':'XLE',
    'FCEL':'XLE','BE':'XLE','FLNC':'XLE','SMR':'XLE','NNE':'XLE',
    'LEU':'XLE','OKLO':'XLE','IMO':'XLE',
    # Materials / Mining (XLB)
    'FCX':'XLB','SCCO':'XLB','HBM':'XLB','SSL':'XLB','AXTA':'XLB',
    'CC':'XLB','ATI':'XLB','IP':'XLB',
    # Healthcare / Biotech (XLV)
    'MRNA':'XLV','REGN':'XLV','SRPT':'XLV','EXEL':'XLV','BEAM':'XLV',
    'BHVN':'XLV','NTLA':'XLV','BBIO':'XLV','OCGN':'XLV','ROIV':'XLV',
    'ATEC':'XLV','AXGN':'XLV','INBX':'XLV','PRLD':'XLV','TYRA':'XLV',
    'NVS':'XLV','RLMD':'XLV','COO':'XLV','KPTI':'XLV',
    # Consumer Discretionary (XLY)
    'TSLA':'XLY','LULU':'XLY','GME':'XLY','UBER':'XLY','M':'XLY',
    'RH':'XLY','DKS':'XLY','EAT':'XLY','GOOS':'XLY','PLNT':'XLY',
    'LOVE':'XLY','FLWS':'XLY','ZUMZ':'XLY','ASO':'XLY','MCFT':'XLY',
    'FIZZ':'XLY','SHOE':'XLY','HOFT':'XLY','RDDT':'XLY','BABA':'XLY',
    # Industrials / Aerospace / Defense (XLI)
    'CAT':'XLI','DOV':'XLI','ETN':'XLI','URI':'XLI','RTX':'XLI',
    'KTOS':'XLI','AVAV':'XLI','STRL':'XLI','IESC':'XLI','AME':'XLI',
    'MAS':'XLI','ABM':'XLI','HTLD':'XLI','AGCO':'XLI','SNX':'XLI',
    'DNOW':'XLI','CPRT':'XLI','SIG':'XLI','LUNR':'XLI','RKLB':'XLI',
    'ASTS':'XLI','PL':'XLI','RDW':'XLI','ACHR':'XLI',
    # Consumer Staples (XLP)
    'KR':'XLP','UNFI':'XLP',
    # Utilities (XLU)
    'NEE':'XLU','EVRG':'XLU',
}

_SECTOR_ETFS = sorted(set(_SECTOR_MAP.values()))  # unique sector ETFs needed


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS (identical to eval script)
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
    print(f"  REGIME ANALYSIS — Data Pipeline")
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
# BUILD E2 BASELINE TRADES (reuse from prior cell or build fresh)
# ═══════════════════════════════════════════════════════════════════════════════

E2_CACHED = False
try:
    _ = e2_trades
    if len(e2_trades) > 0:
        E2_CACHED = True
        print(f"  Using cached e2_trades ({len(e2_trades)} trades) from prior cell")
except NameError:
    pass

if not E2_CACHED:
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

                max_adverse_pnl = tc['pnl_eod']
                for t_str, bar in session_bars:
                    if t_str > EXIT_TIME:
                        break
                    if t_str >= tc['entry_time']:
                        bar_pnl = (ep - bar['h']) / ep * 100
                        if bar_pnl < max_adverse_pnl:
                            max_adverse_pnl = bar_pnl
                tc['max_adverse_pnl'] = round(max_adverse_pnl, 3)

                if len(rth_bars) > 15:
                    open_p = rth_bars[0][1]['o']
                    bar15_c = rth_bars[15][1]['c']
                    tc['opening_drop'] = round((open_p - bar15_c) / open_p, 4) if open_p > 0 else 0
                else:
                    tc['opening_drop'] = None

                gap_size = tc['open_price'] - tc['prior_close']
                if gap_size > 0 and tc['opening_drop'] is not None and len(rth_bars) > 15:
                    bar15_c = rth_bars[15][1]['c']
                    tc['gap_consumed'] = round((tc['open_price'] - bar15_c) / gap_size, 4)
                else:
                    tc['gap_consumed'] = None

                trades.append(tc)

        return trades

    print(f"\n  Building E2 baseline trades with intraday enrichment...")
    e2_trades = build_e2_trades_enriched()

raw_pnls = [t['pnl_eod'] for t in e2_trades]
base_m = compute_metrics(raw_pnls, 'E2_BASE')

print(f"  E2 baseline: {len(e2_trades)} trades, avg {base_m['avg']:+.2f}%, "
      f"Sharpe {base_m['sharpe']:.3f}, WR {base_m['wr']*100:.0f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH REGIME DATA — SPY, VIX, SECTOR ETFs
# Need history back to ~80 trading days before backtest start for 50-day SMA
# ═══════════════════════════════════════════════════════════════════════════════

REGIME_DATA_START = '2025-09-01'  # 80+ trading days before 2026-01-02

print(f"\n{'='*80}")
print(f"  REGIME ANALYSIS — Fetching Market Data")
print(f"  SPY / VIX / sector ETFs from {REGIME_DATA_START} to {BACKTEST_END}")
print(f"{'='*80}")


def fetch_daily_bars(ticker, start, end):
    """Fetch daily OHLCV bars for a single ticker via Polygon v2/aggs."""
    url = f"{_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    data = _pg(url, {"adjusted": "true", "sort": "asc", "limit": "5000"})
    results = data.get('results', [])
    rows = []
    for r in results:
        dt = pd.Timestamp(r['t'], unit='ms', tz='UTC').strftime('%Y-%m-%d')
        rows.append({
            'date': dt,
            'open': float(r.get('o', 0)),
            'high': float(r.get('h', 0)),
            'low': float(r.get('l', 0)),
            'close': float(r.get('c', 0)),
            'volume': int(r.get('v', 0)),
        })
    return pd.DataFrame(rows)


# --- SPY daily ---
print(f"  Fetching SPY daily bars...")
spy_df = fetch_daily_bars('SPY', REGIME_DATA_START, BACKTEST_END)
spy_df = spy_df.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
print(f"    SPY: {len(spy_df)} bars, {spy_df['date'].iloc[0]} to {spy_df['date'].iloc[-1]}")

spy_df['ret_5d'] = spy_df['close'] / spy_df['close'].shift(5) - 1
spy_df['ret_20d'] = spy_df['close'] / spy_df['close'].shift(20) - 1
spy_df['sma_50'] = spy_df['close'].rolling(50).mean()
spy_df['above_50sma'] = spy_df['close'] > spy_df['sma_50']

spy_lookup = spy_df.set_index('date').to_dict('index')


# --- VIX daily ---
# Try I:VIX (Polygon index format), fall back to VIXY ETF as proxy
print(f"  Fetching VIX daily bars...")
vix_df = None
for vix_ticker in ['I:VIX', 'VIX', 'VIXY']:
    try:
        _vdf = fetch_daily_bars(vix_ticker, REGIME_DATA_START, BACKTEST_END)
        if len(_vdf) > 20:
            vix_df = _vdf.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
            print(f"    VIX ({vix_ticker}): {len(vix_df)} bars")
            break
    except Exception as e:
        print(f"    {vix_ticker} failed: {e}")
        continue

if vix_df is None or len(vix_df) == 0:
    print(f"    WARNING: Could not fetch VIX data. VIX features will be NaN.")
    vix_lookup = {}
else:
    vix_lookup = {row['date']: row['close'] for _, row in vix_df.iterrows()}


# --- Sector ETF daily bars ---
print(f"  Fetching sector ETF daily bars ({len(_SECTOR_ETFS)} ETFs)...")
sector_dfs = {}
for etf in _SECTOR_ETFS:
    try:
        sdf = fetch_daily_bars(etf, REGIME_DATA_START, BACKTEST_END)
        sdf = sdf.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
        sdf['ret_5d'] = sdf['close'] / sdf['close'].shift(5) - 1
        sector_dfs[etf] = sdf.set_index('date')['ret_5d'].to_dict()
        print(f"    {etf}: {len(sdf)} bars")
    except Exception as e:
        print(f"    {etf} FAILED: {e}")
        sector_dfs[etf] = {}
    time.sleep(0.2)  # rate limit courtesy

print(f"  Regime data fetch complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE REGIME FEATURES PER TRADE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print(f"  COMPUTING REGIME FEATURES FOR {len(e2_trades)} TRADES")
print(f"{'='*80}")

# Build list of all trading dates (for prior-day lookups)
all_spy_dates = sorted(spy_lookup.keys())

def _get_prior_spy_date(dt_str):
    """Find the most recent SPY date on or before the prior trading day."""
    try:
        ptd = _prior_td(dt_str).strftime('%Y-%m-%d')
    except:
        ptd = dt_str
    # Find closest available SPY date <= ptd
    for d in reversed(all_spy_dates):
        if d <= ptd:
            return d
    return None

n_matched = 0
n_vix_matched = 0
for t in e2_trades:
    dt_str = t['scan_date']

    # SPY features: use prior trading day's close
    spy_dt = _get_prior_spy_date(dt_str)
    if spy_dt and spy_dt in spy_lookup:
        spy_row = spy_lookup[spy_dt]
        t['spy_5d_return'] = round(spy_row.get('ret_5d', float('nan')) * 100, 2) \
            if not pd.isna(spy_row.get('ret_5d')) else None
        t['spy_20d_return'] = round(spy_row.get('ret_20d', float('nan')) * 100, 2) \
            if not pd.isna(spy_row.get('ret_20d')) else None
        t['spy_above_50sma'] = bool(spy_row.get('above_50sma', False)) \
            if not pd.isna(spy_row.get('above_50sma', float('nan'))) else None
        n_matched += 1
    else:
        t['spy_5d_return'] = None
        t['spy_20d_return'] = None
        t['spy_above_50sma'] = None

    # VIX: prior trading day close
    vix_val = None
    if spy_dt:
        vix_val = vix_lookup.get(spy_dt)
    if vix_val is None:
        # Try exact trade date
        vix_val = vix_lookup.get(dt_str)
    t['vix_level'] = round(vix_val, 2) if vix_val is not None else None
    if vix_val is not None:
        n_vix_matched += 1

    # Sector momentum: 5d return of sector ETF
    sector_etf = _SECTOR_MAP.get(t['ticker'], 'SPY')
    if sector_etf == 'SPY':
        # Use SPY 5d return
        t['sector_etf'] = 'SPY'
        t['sector_momentum'] = t['spy_5d_return']
    else:
        t['sector_etf'] = sector_etf
        sec_5d = sector_dfs.get(sector_etf, {}).get(spy_dt)
        if sec_5d is not None and not pd.isna(sec_5d):
            t['sector_momentum'] = round(sec_5d * 100, 2)
        else:
            t['sector_momentum'] = t['spy_5d_return']  # fallback

    # Regime classification
    if t['spy_20d_return'] is not None:
        if t['spy_20d_return'] > 5.0:
            t['regime'] = 'bull'
        elif t['spy_20d_return'] < -5.0:
            t['regime'] = 'bear'
        else:
            t['regime'] = 'flat'
    else:
        t['regime'] = 'unknown'

print(f"  SPY match: {n_matched}/{len(e2_trades)} trades")
print(f"  VIX match: {n_vix_matched}/{len(e2_trades)} trades")

# Print sample
print(f"\n  Sample regime features (first 5 trades):")
print(f"  {'Date':>12s}  {'Ticker':>6s}  {'PnL%':>7s}  {'SPY5d':>6s}  "
      f"{'SPY20d':>7s}  {'>50SMA':>6s}  {'VIX':>6s}  {'Sector':>7s}  {'Regime':>6s}")
print(f"  {'---':>12s}  {'---':>6s}  {'---':>7s}  {'---':>6s}  "
      f"{'---':>7s}  {'---':>6s}  {'---':>6s}  {'---':>7s}  {'---':>6s}")
for t in e2_trades[:5]:
    spy5 = f"{t['spy_5d_return']:+.1f}%" if t['spy_5d_return'] is not None else 'N/A'
    spy20 = f"{t['spy_20d_return']:+.1f}%" if t['spy_20d_return'] is not None else 'N/A'
    sma = 'Y' if t['spy_above_50sma'] else 'N' if t['spy_above_50sma'] is not None else '?'
    vix = f"{t['vix_level']:.1f}" if t['vix_level'] is not None else 'N/A'
    sec = f"{t['sector_momentum']:+.1f}%" if t['sector_momentum'] is not None else 'N/A'
    print(f"  {t['scan_date']:>12s}  {t['ticker']:>6s}  {t['pnl_eod']:>+6.1f}%  "
          f"{spy5:>6s}  {spy20:>7s}  {sma:>6s}  {vix:>6s}  {sec:>7s}  {t['regime']:>6s}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: STRATIFY BY REGIME (bull / flat / bear)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  SECTION 1: PERFORMANCE BY REGIME")
print(f"  bull  = SPY 20d return > +5%")
print(f"  flat  = SPY 20d return between -5% and +5%")
print(f"  bear  = SPY 20d return < -5%")
print(f"{'='*80}")

regime_trades = defaultdict(list)
for t in e2_trades:
    regime_trades[t['regime']].append(t)

print(f"\n  {'Regime':>8s}  {'N':>4s}  {'Avg PnL':>8s}  {'Med PnL':>8s}  "
      f"{'WR%':>5s}  {'Sharpe':>7s}  {'Total $':>12s}  {'Best':>8s}  {'Worst':>8s}")
print(f"  {'---':>8s}  {'---':>4s}  {'---':>8s}  {'---':>8s}  "
      f"{'---':>5s}  {'---':>7s}  {'---':>12s}  {'---':>8s}  {'---':>8s}")

regime_summary = {}
for regime in ['bull', 'flat', 'bear', 'unknown']:
    trades = regime_trades.get(regime, [])
    if not trades:
        continue
    pnls = [t['pnl_eod'] for t in trades]
    m = compute_metrics(pnls, regime)
    total_dol = sum(CAPITAL * p / 100 for p in pnls)

    regime_summary[regime] = {
        'n': m['n'], 'avg': round(m['avg'], 2), 'med': round(m['med'], 2),
        'wr': round(m['wr'], 3), 'sharpe': round(m['sharpe'], 3),
        'total_dol': round(total_dol), 'max_gain': round(m['max_gain'], 1),
        'max_loss': round(m['max_loss'], 1),
    }

    print(f"  {regime:>8s}  {m['n']:>4d}  {m['avg']:>+7.2f}%  {m['med']:>+7.2f}%  "
          f"{m['wr']*100:>4.0f}%  {m['sharpe']:>7.3f}  ${total_dol:>+11,.0f}  "
          f"{m['max_gain']:>+7.1f}%  {m['max_loss']:>+7.1f}%")

# All trades baseline for comparison
print(f"  {'ALL':>8s}  {base_m['n']:>4d}  {base_m['avg']:>+7.2f}%  {base_m['med']:>+7.2f}%  "
      f"{base_m['wr']*100:>4.0f}%  {base_m['sharpe']:>7.3f}  "
      f"${sum(CAPITAL * p / 100 for p in raw_pnls):>+11,.0f}  "
      f"{base_m['max_gain']:>+7.1f}%  {base_m['max_loss']:>+7.1f}%")

print(f"\n  WHAT THIS SAYS:")
bull_n = regime_summary.get('bull', {}).get('n', 0)
flat_n = regime_summary.get('flat', {}).get('n', 0)
bear_n = regime_summary.get('bear', {}).get('n', 0)
bull_avg = regime_summary.get('bull', {}).get('avg', 0)
flat_avg = regime_summary.get('flat', {}).get('avg', 0)
bear_avg = regime_summary.get('bear', {}).get('avg', 0)
best_regime = max(['bull', 'flat', 'bear'],
                  key=lambda r: regime_summary.get(r, {}).get('avg', -999))
print(f"  Best regime for SHORT gap-fade: {best_regime.upper()} "
      f"(avg {regime_summary.get(best_regime, {}).get('avg', 0):+.2f}%)")
print(f"  bull={bull_n} trades ({bull_avg:+.2f}%), "
      f"flat={flat_n} trades ({flat_avg:+.2f}%), "
      f"bear={bear_n} trades ({bear_avg:+.2f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MONTHLY REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  SECTION 2: MONTHLY REGIME MAP (Jan-Sep 2026)")
print(f"{'='*80}")

MONTHS = [
    ('Jan', '2026-01'), ('Feb', '2026-02'), ('Mar', '2026-03'),
    ('Apr', '2026-04'), ('May', '2026-05'), ('Jun', '2026-06'),
    ('Jul', '2026-07'), ('Aug', '2026-08'), ('Sep', '2026-09'),
]

print(f"\n  {'Month':>5s}  {'SPY 20d%':>9s}  {'SPY 5d%':>8s}  {'SPY >50SMA':>10s}  "
      f"{'VIX':>6s}  {'Regime':>8s}  {'Trades':>6s}  {'Avg PnL':>8s}")
print(f"  {'---':>5s}  {'---':>9s}  {'---':>8s}  {'---':>10s}  "
      f"{'---':>6s}  {'---':>8s}  {'---':>6s}  {'---':>8s}")

monthly_regime = {}
for label, ym in MONTHS:
    # Find first trading day of month
    month_start = None
    for d_str in sorted(spy_lookup.keys()):
        if d_str.startswith(ym):
            month_start = d_str
            break

    if month_start and month_start in spy_lookup:
        row = spy_lookup[month_start]
        spy_20d = row.get('ret_20d')
        spy_5d = row.get('ret_5d')
        above_sma = row.get('above_50sma')
        vix_val = vix_lookup.get(month_start)

        spy_20d_pct = spy_20d * 100 if spy_20d is not None and not pd.isna(spy_20d) else None
        spy_5d_pct = spy_5d * 100 if spy_5d is not None and not pd.isna(spy_5d) else None

        if spy_20d_pct is not None:
            if spy_20d_pct > 5: regime = 'BULL'
            elif spy_20d_pct < -5: regime = 'BEAR'
            else: regime = 'FLAT'
        else:
            regime = '?'
    else:
        spy_20d_pct, spy_5d_pct, above_sma, vix_val, regime = None, None, None, None, '?'

    # Month's trades
    month_trades = [t for t in e2_trades if t['scan_date'].startswith(ym)]
    month_pnls = [t['pnl_eod'] for t in month_trades]
    month_avg = float(np.mean(month_pnls)) if month_pnls else 0

    monthly_regime[label] = {
        'spy_20d': round(spy_20d_pct, 1) if spy_20d_pct is not None else None,
        'spy_5d': round(spy_5d_pct, 1) if spy_5d_pct is not None else None,
        'above_50sma': bool(above_sma) if above_sma is not None and not pd.isna(above_sma) else None,
        'vix': round(vix_val, 1) if vix_val is not None else None,
        'regime': regime,
        'n_trades': len(month_trades),
        'avg_pnl': round(month_avg, 2),
    }

    s20 = f"{spy_20d_pct:>+8.1f}%" if spy_20d_pct is not None else '     N/A'
    s5 = f"{spy_5d_pct:>+7.1f}%" if spy_5d_pct is not None else '    N/A'
    sma_s = 'Y' if above_sma else 'N' if above_sma is not None else '?'
    v_s = f"{vix_val:>5.1f}" if vix_val is not None else '  N/A'

    print(f"  {label:>5s}  {s20}  {s5}  {sma_s:>10s}  {v_s}  {regime:>8s}  "
          f"{len(month_trades):>6d}  {month_avg:>+7.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DETAILED REGIME FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  SECTION 3: REGIME FEATURE BREAKDOWN")
print(f"{'='*80}")

# --- SPY 20d return buckets ---
print(f"\n  --- SPY 20d Return Buckets ---")

SPY20_BUCKETS = [
    ('<-10%',   None,   -10),
    ('-10 to -5%', -10,  -5),
    ('-5 to -2%',   -5,  -2),
    ('-2 to 0%',    -2,   0),
    ('0 to +2%',     0,   2),
    ('+2 to +5%',    2,   5),
    ('+5 to +10%',   5,  10),
    ('>+10%',       10,  None),
]

print(f"  {'Bucket':>14s}  {'N':>4s}  {'Avg PnL':>8s}  {'Med PnL':>8s}  "
      f"{'WR%':>5s}  {'Sharpe':>7s}  {'Total $':>10s}")
print(f"  {'---':>14s}  {'---':>4s}  {'---':>8s}  {'---':>8s}  "
      f"{'---':>5s}  {'---':>7s}  {'---':>10s}")

for label, lo, hi in SPY20_BUCKETS:
    bt = []
    for t in e2_trades:
        v = t.get('spy_20d_return')
        if v is None: continue
        if lo is None and v < hi: bt.append(t)
        elif hi is None and v >= lo: bt.append(t)
        elif lo is not None and hi is not None and lo <= v < hi: bt.append(t)
    if not bt: continue
    pnls = [t['pnl_eod'] for t in bt]
    m = compute_metrics(pnls)
    dol = sum(CAPITAL * p / 100 for p in pnls)
    print(f"  {label:>14s}  {m['n']:>4d}  {m['avg']:>+7.2f}%  {m['med']:>+7.2f}%  "
          f"{m['wr']*100:>4.0f}%  {m['sharpe']:>7.3f}  ${dol:>+9,.0f}")


# --- VIX level buckets ---
vix_trades = [t for t in e2_trades if t.get('vix_level') is not None]
if vix_trades:
    print(f"\n  --- VIX Level Buckets ({len(vix_trades)} trades with VIX data) ---")

    VIX_BUCKETS = [
        ('<12',    None,  12),
        ('12-15',  12,    15),
        ('15-18',  15,    18),
        ('18-22',  18,    22),
        ('22-28',  22,    28),
        ('28-35',  28,    35),
        ('>35',    35,    None),
    ]

    print(f"  {'Bucket':>10s}  {'N':>4s}  {'Avg PnL':>8s}  {'Med PnL':>8s}  "
          f"{'WR%':>5s}  {'Sharpe':>7s}  {'Total $':>10s}")
    print(f"  {'---':>10s}  {'---':>4s}  {'---':>8s}  {'---':>8s}  "
          f"{'---':>5s}  {'---':>7s}  {'---':>10s}")

    for label, lo, hi in VIX_BUCKETS:
        bt = []
        for t in vix_trades:
            v = t['vix_level']
            if lo is None and v < hi: bt.append(t)
            elif hi is None and v >= lo: bt.append(t)
            elif lo is not None and hi is not None and lo <= v < hi: bt.append(t)
        if not bt: continue
        pnls = [t['pnl_eod'] for t in bt]
        m = compute_metrics(pnls)
        dol = sum(CAPITAL * p / 100 for p in pnls)
        print(f"  {label:>10s}  {m['n']:>4d}  {m['avg']:>+7.2f}%  {m['med']:>+7.2f}%  "
              f"{m['wr']*100:>4.0f}%  {m['sharpe']:>7.3f}  ${dol:>+9,.0f}")
else:
    print(f"\n  --- VIX: No data available ---")


# --- SPY above/below 50 SMA ---
sma_trades = [t for t in e2_trades if t.get('spy_above_50sma') is not None]
if sma_trades:
    print(f"\n  --- SPY vs 50-day SMA ---")

    for sma_state, sma_label in [(True, 'SPY > 50 SMA'), (False, 'SPY < 50 SMA')]:
        bt = [t for t in sma_trades if t['spy_above_50sma'] == sma_state]
        if not bt: continue
        pnls = [t['pnl_eod'] for t in bt]
        m = compute_metrics(pnls)
        dol = sum(CAPITAL * p / 100 for p in pnls)
        print(f"  {sma_label:>16s}  n={m['n']:>3d}  avg={m['avg']:>+.2f}%  "
              f"WR={m['wr']*100:.0f}%  Sharpe={m['sharpe']:.3f}  ${dol:>+,.0f}")


# --- Sector momentum ---
sec_trades = [t for t in e2_trades if t.get('sector_momentum') is not None]
if sec_trades:
    print(f"\n  --- Sector 5d Momentum ---")

    SEC_BUCKETS = [
        ('<-3%',     None, -3),
        ('-3 to 0%',  -3,   0),
        ('0 to +3%',   0,   3),
        ('>+3%',       3,  None),
    ]

    print(f"  {'Bucket':>12s}  {'N':>4s}  {'Avg PnL':>8s}  {'WR%':>5s}  "
          f"{'Sharpe':>7s}  {'Total $':>10s}")
    print(f"  {'---':>12s}  {'---':>4s}  {'---':>8s}  {'---':>5s}  "
          f"{'---':>7s}  {'---':>10s}")

    for label, lo, hi in SEC_BUCKETS:
        bt = []
        for t in sec_trades:
            v = t['sector_momentum']
            if lo is None and v < hi: bt.append(t)
            elif hi is None and v >= lo: bt.append(t)
            elif lo is not None and hi is not None and lo <= v < hi: bt.append(t)
        if not bt: continue
        pnls = [t['pnl_eod'] for t in bt]
        m = compute_metrics(pnls)
        dol = sum(CAPITAL * p / 100 for p in pnls)
        print(f"  {label:>12s}  {m['n']:>4d}  {m['avg']:>+7.2f}%  "
              f"{m['wr']*100:>4.0f}%  {m['sharpe']:>7.3f}  ${dol:>+9,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: IN-SAMPLE / OUT-OF-SAMPLE REGIME FILTER OPTIMIZATION
# IS = Jan-Jun 2026, OOS = Jul-Sep 2026
# ═══════════════════════════════════════════════════════════════════════════════

IS_CUTOFF = '2026-07-01'

is_trades = [t for t in e2_trades if t['scan_date'] < IS_CUTOFF]
oos_trades = [t for t in e2_trades if t['scan_date'] >= IS_CUTOFF]

is_m = compute_metrics([t['pnl_eod'] for t in is_trades], 'IS')
oos_m = compute_metrics([t['pnl_eod'] for t in oos_trades], 'OOS')

print(f"\n\n{'='*80}")
print(f"  SECTION 4: REGIME FILTER OPTIMIZATION")
print(f"  In-sample:      Jan-Jun ({len(is_trades)} trades, avg {is_m['avg']:+.2f}%)")
print(f"  Out-of-sample:  Jul-Sep ({len(oos_trades)} trades, avg {oos_m['avg']:+.2f}%)")
print(f"{'='*80}")

# Count total trading days in OOS period for exclusion reporting
oos_td_dates = []
_d = datetime.strptime(IS_CUTOFF, '%Y-%m-%d').date()
_end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
while _d <= _end:
    if _is_td(_d):
        oos_td_dates.append(_d.strftime('%Y-%m-%d'))
    _d += timedelta(days=1)
n_oos_trading_days = len(oos_td_dates)


def apply_filter(trades, rule_fn):
    """Apply a regime filter. Returns (kept, excluded)."""
    kept = [t for t in trades if rule_fn(t)]
    excluded = [t for t in trades if not rule_fn(t)]
    return kept, excluded


def count_oos_days_excluded(rule_fn_on_date):
    """Count how many OOS trading days would be excluded by a date-level rule."""
    n_excluded = 0
    for dt_str in oos_td_dates:
        if not rule_fn_on_date(dt_str):
            n_excluded += 1
    return n_excluded


# Build date-level regime lookup for day-level exclusion counting
date_regime = {}
for d_str in sorted(spy_lookup.keys()):
    row = spy_lookup[d_str]
    date_regime[d_str] = {
        'spy_20d': row.get('ret_20d', float('nan')),
        'spy_5d': row.get('ret_5d', float('nan')),
        'above_50sma': row.get('above_50sma', None),
        'vix': vix_lookup.get(d_str),
    }

# ─── Define filter rules ───

filter_rules = []

# Rule 1: SPY_20d < threshold (only trade when SPY is not too bullish)
for thresh in [-10, -5, -3, -1, 0, 1, 2, 3, 5, 8, 10, 15]:
    rule_name = f"SPY_20d < {thresh:+d}%"
    thresh_frac = thresh / 100.0

    def make_trade_fn(th=thresh):
        def fn(t):
            v = t.get('spy_20d_return')
            if v is None: return True  # no data = allow
            return v < th
        return fn

    def make_date_fn(th_f=thresh_frac):
        def fn(dt):
            r = date_regime.get(dt)
            if r is None: return True
            v = r.get('spy_20d')
            if v is None or pd.isna(v): return True
            return v < th_f
        return fn

    filter_rules.append((rule_name, make_trade_fn(), make_date_fn()))

# Rule 2: VIX > threshold
for thresh in [12, 15, 18, 20, 22, 25, 28, 30, 35]:
    rule_name = f"VIX > {thresh}"

    def make_trade_fn(th=thresh):
        def fn(t):
            v = t.get('vix_level')
            if v is None: return True
            return v > th
        return fn

    def make_date_fn(th=thresh):
        def fn(dt):
            r = date_regime.get(dt)
            if r is None: return True
            v = r.get('vix')
            if v is None: return True
            return v > th
        return fn

    filter_rules.append((rule_name, make_trade_fn(), make_date_fn()))

# Rule 3: SPY below 50 SMA
rule_name = "SPY < 50 SMA"

def _trade_fn_sma(t):
    v = t.get('spy_above_50sma')
    if v is None: return True
    return not v  # trade only when BELOW SMA

def _date_fn_sma(dt):
    r = date_regime.get(dt)
    if r is None: return True
    v = r.get('above_50sma')
    if v is None or (isinstance(v, float) and pd.isna(v)): return True
    return not v

filter_rules.append((rule_name, _trade_fn_sma, _date_fn_sma))

# Rule 4: Sector 5d < threshold
for thresh in [-5, -3, -1, 0, 1, 3, 5]:
    rule_name = f"Sector_5d < {thresh:+d}%"

    def make_trade_fn(th=thresh):
        def fn(t):
            v = t.get('sector_momentum')
            if v is None: return True
            return v < th
        return fn

    # Date-level: use SPY 5d as proxy (no per-ticker sector at date level)
    def make_date_fn(th_f=thresh / 100.0):
        def fn(dt):
            r = date_regime.get(dt)
            if r is None: return True
            v = r.get('spy_5d')
            if v is None or pd.isna(v): return True
            return v < th_f
        return fn

    filter_rules.append((rule_name, make_trade_fn(), make_date_fn()))


# ─── Run filter sweep ───

print(f"\n  {'Rule':>20s}  |  {'--- IN-SAMPLE (Jan-Jun) ---':^40s}  |  "
      f"{'--- OUT-OF-SAMPLE (Jul-Sep) ---':^42s}  |  {'OOS':>5s}")
print(f"  {'':>20s}  |  {'N':>4s}  {'Avg':>7s}  {'WR%':>5s}  {'Sharpe':>7s}  {'$':>10s}  |  "
      f"{'N':>4s}  {'Avg':>7s}  {'WR%':>5s}  {'Sharpe':>7s}  {'$':>10s}  |  {'Skip%':>5s}")
print(f"  {'---':>20s}  |  {'---':>4s}  {'---':>7s}  {'---':>5s}  {'---':>7s}  {'---':>10s}  |  "
      f"{'---':>4s}  {'---':>7s}  {'---':>5s}  {'---':>7s}  {'---':>10s}  |  {'---':>5s}")

# Baseline rows
is_dol = sum(CAPITAL * t['pnl_eod'] / 100 for t in is_trades)
oos_dol = sum(CAPITAL * t['pnl_eod'] / 100 for t in oos_trades)
print(f"  {'(no filter)':>20s}  |  {len(is_trades):>4d}  {is_m['avg']:>+6.2f}%  "
      f"{is_m['wr']*100:>4.0f}%  {is_m['sharpe']:>7.3f}  ${is_dol:>+9,.0f}  |  "
      f"{len(oos_trades):>4d}  {oos_m['avg']:>+6.2f}%  {oos_m['wr']*100:>4.0f}%  "
      f"{oos_m['sharpe']:>7.3f}  ${oos_dol:>+9,.0f}  |  {'0%':>5s}")

filter_results = []
for rule_name, trade_fn, date_fn in filter_rules:
    is_kept, _ = apply_filter(is_trades, trade_fn)
    oos_kept, oos_excl = apply_filter(oos_trades, trade_fn)

    is_pnls = [t['pnl_eod'] for t in is_kept]
    oos_pnls = [t['pnl_eod'] for t in oos_kept]

    is_r = compute_metrics(is_pnls, f'IS_{rule_name}')
    oos_r = compute_metrics(oos_pnls, f'OOS_{rule_name}')

    is_d = sum(CAPITAL * p / 100 for p in is_pnls)
    oos_d = sum(CAPITAL * p / 100 for p in oos_pnls)

    # Count OOS trading days excluded
    n_days_excl = count_oos_days_excluded(date_fn)
    skip_pct = n_days_excl / n_oos_trading_days * 100 if n_oos_trading_days > 0 else 0

    fr = {
        'rule': rule_name,
        'is_n': is_r['n'], 'is_avg': round(is_r['avg'], 2), 'is_wr': round(is_r['wr'], 3),
        'is_sharpe': round(is_r['sharpe'], 3), 'is_dol': round(is_d),
        'oos_n': oos_r['n'], 'oos_avg': round(oos_r['avg'], 2), 'oos_wr': round(oos_r['wr'], 3),
        'oos_sharpe': round(oos_r['sharpe'], 3), 'oos_dol': round(oos_d),
        'oos_days_excluded': n_days_excl,
        'oos_skip_pct': round(skip_pct, 1),
    }
    filter_results.append(fr)

    # Mark rows that beat baseline IS Sharpe AND maintain positive OOS avg
    mark = ''
    if is_r['sharpe'] > is_m['sharpe'] and oos_r['avg'] > 0 and oos_r['n'] >= 3:
        mark = ' ***'
    elif is_r['sharpe'] > is_m['sharpe']:
        mark = ' *'

    print(f"  {rule_name:>20s}  |  {is_r['n']:>4d}  {is_r['avg']:>+6.2f}%  "
          f"{is_r['wr']*100:>4.0f}%  {is_r['sharpe']:>7.3f}  ${is_d:>+9,.0f}  |  "
          f"{oos_r['n']:>4d}  {oos_r['avg']:>+6.2f}%  {oos_r['wr']*100:>4.0f}%  "
          f"{oos_r['sharpe']:>7.3f}  ${oos_d:>+9,.0f}  |  "
          f"{skip_pct:>4.0f}%{mark}")

# ─── Find best filter ───

# Best = highest IS Sharpe among rules where OOS avg > 0 and OOS n >= 3
viable = [f for f in filter_results
          if f['oos_avg'] > 0 and f['oos_n'] >= 3 and f['is_sharpe'] > is_m['sharpe']]
if viable:
    best_filter = max(viable, key=lambda f: f['is_sharpe'])
    print(f"\n  BEST FILTER: {best_filter['rule']}")
    print(f"    IS:  n={best_filter['is_n']}, avg={best_filter['is_avg']:+.2f}%, "
          f"Sharpe={best_filter['is_sharpe']:.3f} (vs baseline {is_m['sharpe']:.3f})")
    print(f"    OOS: n={best_filter['oos_n']}, avg={best_filter['oos_avg']:+.2f}%, "
          f"Sharpe={best_filter['oos_sharpe']:.3f} (vs baseline {oos_m['sharpe']:.3f})")
    print(f"    OOS days excluded: {best_filter['oos_days_excluded']}/{n_oos_trading_days} "
          f"({best_filter['oos_skip_pct']:.0f}%)")
else:
    # Fallback: best IS Sharpe among rules with positive OOS avg
    fallback = [f for f in filter_results if f['oos_avg'] > 0 and f['oos_n'] >= 3]
    if fallback:
        best_filter = max(fallback, key=lambda f: f['is_sharpe'])
        print(f"\n  BEST AVAILABLE FILTER (does NOT beat IS baseline Sharpe): {best_filter['rule']}")
        print(f"    IS:  n={best_filter['is_n']}, avg={best_filter['is_avg']:+.2f}%, "
              f"Sharpe={best_filter['is_sharpe']:.3f} (vs baseline {is_m['sharpe']:.3f})")
        print(f"    OOS: n={best_filter['oos_n']}, avg={best_filter['oos_avg']:+.2f}%, "
              f"Sharpe={best_filter['oos_sharpe']:.3f}")
        print(f"    OOS days excluded: {best_filter['oos_days_excluded']}/{n_oos_trading_days}")
    else:
        best_filter = None
        print(f"\n  NO VIABLE FILTER: Every filter either kills OOS performance or leaves <3 trades.")

print(f"\n  Legend:")
print(f"    * = beats IS baseline Sharpe")
print(f"    *** = beats IS Sharpe AND OOS avg > 0 with n >= 3")
print(f"    OOS Skip% = fraction of Jul-Sep trading days that would be excluded")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: HALF-YEAR PERFORMANCE SHIFT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  SECTION 5: PERFORMANCE SHIFT — H1 vs H2")
print(f"{'='*80}")

h1 = [t for t in e2_trades if t['scan_date'] < IS_CUTOFF]
h2 = [t for t in e2_trades if t['scan_date'] >= IS_CUTOFF]

h1_m = compute_metrics([t['pnl_eod'] for t in h1], 'H1')
h2_m = compute_metrics([t['pnl_eod'] for t in h2], 'H2')

print(f"\n  {'Period':>12s}  {'N':>4s}  {'Avg PnL':>8s}  {'Med PnL':>8s}  "
      f"{'WR%':>5s}  {'Sharpe':>7s}  {'Total $':>12s}")
print(f"  {'---':>12s}  {'---':>4s}  {'---':>8s}  {'---':>8s}  "
      f"{'---':>5s}  {'---':>7s}  {'---':>12s}")

h1_dol = sum(CAPITAL * t['pnl_eod'] / 100 for t in h1)
h2_dol = sum(CAPITAL * t['pnl_eod'] / 100 for t in h2)

print(f"  {'Jan-Jun':>12s}  {h1_m['n']:>4d}  {h1_m['avg']:>+7.2f}%  {h1_m['med']:>+7.2f}%  "
      f"{h1_m['wr']*100:>4.0f}%  {h1_m['sharpe']:>7.3f}  ${h1_dol:>+11,.0f}")
print(f"  {'Jul-Sep':>12s}  {h2_m['n']:>4d}  {h2_m['avg']:>+7.2f}%  {h2_m['med']:>+7.2f}%  "
      f"{h2_m['wr']*100:>4.0f}%  {h2_m['sharpe']:>7.3f}  ${h2_dol:>+11,.0f}")

if h1_m['avg'] != 0:
    degradation = (h2_m['avg'] - h1_m['avg']) / abs(h1_m['avg']) * 100
    print(f"\n  PnL degradation H1 -> H2: {degradation:+.1f}%")
else:
    degradation = 0

# Average regime conditions in each half
h1_spy20 = [t['spy_20d_return'] for t in h1 if t.get('spy_20d_return') is not None]
h2_spy20 = [t['spy_20d_return'] for t in h2 if t.get('spy_20d_return') is not None]
h1_vix = [t['vix_level'] for t in h1 if t.get('vix_level') is not None]
h2_vix = [t['vix_level'] for t in h2 if t.get('vix_level') is not None]

print(f"\n  Regime conditions:")
if h1_spy20:
    print(f"    H1 avg SPY 20d: {np.mean(h1_spy20):+.1f}%")
if h2_spy20:
    print(f"    H2 avg SPY 20d: {np.mean(h2_spy20):+.1f}%")
if h1_vix:
    print(f"    H1 avg VIX:     {np.mean(h1_vix):.1f}")
if h2_vix:
    print(f"    H2 avg VIX:     {np.mean(h2_vix):.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


# ─── Chart 1: Avg PnL by regime (bull / flat / bear) ──────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

regime_labels = []
regime_avgs = []
regime_meds = []
regime_counts = []
regime_colors_bar = {'bull': '#27ae60', 'flat': '#3498db', 'bear': '#e74c3c'}

for regime in ['bull', 'flat', 'bear']:
    if regime in regime_summary:
        regime_labels.append(regime.upper())
        regime_avgs.append(regime_summary[regime]['avg'])
        regime_meds.append(regime_summary[regime]['med'])
        regime_counts.append(regime_summary[regime]['n'])

if regime_labels:
    x = np.arange(len(regime_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, regime_avgs, width, color=[regime_colors_bar.get(r.lower(), '#95a5a6') for r in regime_labels],
                   alpha=0.85, edgecolor='black', linewidth=0.5, label='Avg PnL %')
    bars2 = ax.bar(x + width/2, regime_meds, width, color=[regime_colors_bar.get(r.lower(), '#95a5a6') for r in regime_labels],
                   alpha=0.45, edgecolor='black', linewidth=0.5, label='Median PnL %', hatch='//')

    ax.axhline(0, color='black', lw=1)
    ax.axhline(base_m['avg'], color='gray', ls='--', alpha=0.5, label=f'Baseline avg ({base_m["avg"]:+.1f}%)')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(regime_labels, regime_counts)])
    ax.set_ylabel('PnL %')
    ax.set_title('Agent 1 SHORT Gap-Fade: Avg PnL by Market Regime\n'
                 '(bull = SPY 20d > +5%, flat = -5% to +5%, bear = SPY 20d < -5%)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    for b, v in zip(bars1, regime_avgs):
        va = 'bottom' if v >= 0 else 'top'
        ax.text(b.get_x() + b.get_width()/2, v + (0.3 if v >= 0 else -0.3),
                f"{v:+.1f}%", ha='center', va=va, fontsize=9, fontweight='bold')

plt.tight_layout()
p1 = os.path.join(BASE_DIR, 'regime_pnl_by_regime.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {p1}")


# ─── Chart 2: SPY 20d return timeline + trade PnL dots ────────────────────────

fig, ax1 = plt.subplots(figsize=(16, 7))

# Plot SPY 20d return as a line
spy_plot = spy_df[spy_df['date'] >= BACKTEST_START].copy()
spy_plot = spy_plot[spy_plot['ret_20d'].notna()].copy()

if len(spy_plot) > 0:
    spy_dates = pd.to_datetime(spy_plot['date'])
    spy_ret20 = spy_plot['ret_20d'].values * 100

    ax1.plot(spy_dates, spy_ret20, color='#2c3e50', linewidth=1.5, alpha=0.7,
             label='SPY 20d Return %')
    ax1.fill_between(spy_dates, spy_ret20, 0, where=spy_ret20 > 5,
                     alpha=0.1, color='green', label='Bull zone (>+5%)')
    ax1.fill_between(spy_dates, spy_ret20, 0, where=spy_ret20 < -5,
                     alpha=0.1, color='red', label='Bear zone (<-5%)')
    ax1.axhline(5, color='green', ls=':', alpha=0.4)
    ax1.axhline(-5, color='red', ls=':', alpha=0.4)
    ax1.axhline(0, color='black', lw=0.5)

    # Overlay trade PnL as scatter dots
    ax2 = ax1.twinx()
    trade_dates = pd.to_datetime([t['scan_date'] for t in e2_trades])
    trade_pnls = [t['pnl_eod'] for t in e2_trades]
    colors = ['#27ae60' if p > 0 else '#e74c3c' for p in trade_pnls]
    sizes = [max(15, min(80, abs(p) * 3)) for p in trade_pnls]

    ax2.scatter(trade_dates, trade_pnls, c=colors, s=sizes, alpha=0.6,
                edgecolors='black', linewidth=0.3, zorder=5)
    ax2.axhline(0, color='gray', ls='--', alpha=0.3)
    ax2.set_ylabel('Trade PnL % (dots)', fontsize=9)

    # Add H1/H2 divider
    ax1.axvline(pd.Timestamp(IS_CUTOFF), color='purple', ls='--', lw=2, alpha=0.5)
    ax1.text(pd.Timestamp(IS_CUTOFF), ax1.get_ylim()[1] * 0.9,
             '  H2 starts', color='purple', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('SPY 20d Return %', fontsize=9)
    ax1.set_title('SPY 20d Regime + Agent 1 Trade PnL\n'
                  'Green dots = winners, Red dots = losers, Size ~ |PnL|',
                  fontsize=10, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.2)

plt.tight_layout()
p2 = os.path.join(BASE_DIR, 'regime_spy_timeline.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {p2}")


# ─── Chart 3: VIX level vs trade PnL scatter ──────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7))

vix_vals = [t['vix_level'] for t in e2_trades if t.get('vix_level') is not None]
pnl_vals = [t['pnl_eod'] for t in e2_trades if t.get('vix_level') is not None]

if vix_vals:
    colors = ['#27ae60' if p > 0 else '#e74c3c' for p in pnl_vals]
    ax.scatter(vix_vals, pnl_vals, c=colors, alpha=0.5, s=40,
               edgecolors='black', linewidth=0.3)
    ax.axhline(0, color='black', lw=1)

    # Add trend line
    if len(vix_vals) > 5:
        z = np.polyfit(vix_vals, pnl_vals, 1)
        p_fn = np.poly1d(z)
        vix_range = np.linspace(min(vix_vals), max(vix_vals), 100)
        ax.plot(vix_range, p_fn(vix_range), '--', color='#8e44ad', linewidth=2,
                alpha=0.7, label=f'Trend: slope={z[0]:+.2f}%/pt')

        # Correlation
        r = np.corrcoef(vix_vals, pnl_vals)[0, 1]
        ax.text(0.02, 0.98, f'r = {r:+.3f}', transform=ax.transAxes,
                fontsize=10, va='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('VIX Level (prior day close)')
    ax.set_ylabel('Trade PnL %')
    ax.set_title('VIX Level vs Agent 1 Trade PnL\n'
                 'Green = winner, Red = loser',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
else:
    ax.text(0.5, 0.5, 'No VIX data available', transform=ax.transAxes,
            ha='center', va='center', fontsize=14)

plt.tight_layout()
p3 = os.path.join(BASE_DIR, 'regime_vix_scatter.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {p3}")


# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM LINE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  REGIME ANALYSIS — BOTTOM LINE")
print(f"{'='*80}")

print(f"\n  1. REGIME IMPACT ON PnL:")
for regime in ['bull', 'flat', 'bear']:
    if regime in regime_summary:
        rs = regime_summary[regime]
        print(f"     {regime.upper():>5s}: n={rs['n']:>3d}, avg={rs['avg']:>+.2f}%, "
              f"Sharpe={rs['sharpe']:.3f}, total=${rs['total_dol']:>+,.0f}")

print(f"\n  2. H1 vs H2 PERFORMANCE:")
print(f"     Jan-Jun: avg={h1_m['avg']:>+.2f}%, Sharpe={h1_m['sharpe']:.3f}")
print(f"     Jul-Sep: avg={h2_m['avg']:>+.2f}%, Sharpe={h2_m['sharpe']:.3f}")
if h1_m['avg'] != 0:
    print(f"     Degradation: {degradation:+.1f}%")

print(f"\n  3. REGIME CONDITIONS SHIFT:")
if h1_spy20 and h2_spy20:
    print(f"     SPY 20d: H1 avg {np.mean(h1_spy20):+.1f}% -> H2 avg {np.mean(h2_spy20):+.1f}%")
if h1_vix and h2_vix:
    print(f"     VIX:     H1 avg {np.mean(h1_vix):.1f} -> H2 avg {np.mean(h2_vix):.1f}")

print(f"\n  4. BEST REGIME FILTER:")
if best_filter:
    print(f"     Rule: {best_filter['rule']}")
    print(f"     IS Sharpe: {best_filter['is_sharpe']:.3f} (baseline {is_m['sharpe']:.3f})")
    print(f"     OOS avg: {best_filter['oos_avg']:+.2f}%, Sharpe: {best_filter['oos_sharpe']:.3f}")
    print(f"     OOS days skipped: {best_filter['oos_days_excluded']}/{n_oos_trading_days} "
          f"({best_filter['oos_skip_pct']:.0f}%)")
else:
    print(f"     No viable regime filter found.")
    print(f"     The strategy edge is not regime-dependent, OR the regime shift")
    print(f"     is too recent to separate from noise with {len(oos_trades)} OOS trades.")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

results_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)

output = {
    'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'period': f'{BACKTEST_START} to {BACKTEST_END}',
    'n_trades': len(e2_trades),
    'baseline': {
        'avg': round(base_m['avg'], 2),
        'sharpe': round(base_m['sharpe'], 3),
        'wr': round(base_m['wr'], 3),
        'total_dollars': round(sum(CAPITAL * p / 100 for p in raw_pnls)),
    },
    'regime_stratification': regime_summary,
    'monthly_regime': monthly_regime,
    'h1_vs_h2': {
        'h1': {
            'n': h1_m['n'], 'avg': round(h1_m['avg'], 2),
            'sharpe': round(h1_m['sharpe'], 3), 'wr': round(h1_m['wr'], 3),
            'total_dol': round(h1_dol),
        },
        'h2': {
            'n': h2_m['n'], 'avg': round(h2_m['avg'], 2),
            'sharpe': round(h2_m['sharpe'], 3), 'wr': round(h2_m['wr'], 3),
            'total_dol': round(h2_dol),
        },
        'degradation_pct': round(degradation, 1) if h1_m['avg'] != 0 else None,
        'h1_avg_spy20': round(float(np.mean(h1_spy20)), 1) if h1_spy20 else None,
        'h2_avg_spy20': round(float(np.mean(h2_spy20)), 1) if h2_spy20 else None,
        'h1_avg_vix': round(float(np.mean(h1_vix)), 1) if h1_vix else None,
        'h2_avg_vix': round(float(np.mean(h2_vix)), 1) if h2_vix else None,
    },
    'filter_optimization': {
        'is_period': f'Jan-Jun ({len(is_trades)} trades)',
        'oos_period': f'Jul-Sep ({len(oos_trades)} trades)',
        'is_baseline_sharpe': round(is_m['sharpe'], 3),
        'oos_baseline_sharpe': round(oos_m['sharpe'], 3),
        'n_oos_trading_days': n_oos_trading_days,
        'best_filter': best_filter,
        'all_filters': filter_results,
    },
    'trades_with_regime': [{
        'date': t['scan_date'], 'ticker': t['ticker'],
        'pnl': round(t['pnl_eod'], 2),
        'regime': t.get('regime'),
        'spy_5d': t.get('spy_5d_return'),
        'spy_20d': t.get('spy_20d_return'),
        'spy_above_50sma': t.get('spy_above_50sma'),
        'vix': t.get('vix_level'),
        'sector_etf': t.get('sector_etf'),
        'sector_5d': t.get('sector_momentum'),
    } for t in e2_trades],
}

out_path = os.path.join(results_dir, 'regime_analysis.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  Saved: {out_path}")

print(f"\n{'='*80}")
print(f"  DONE — Regime analysis complete")
print(f"{'='*80}")
