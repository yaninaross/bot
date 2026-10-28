# %% Agent 1 — Aug-Oct 2026 Investigation
#
# R3:  E2-A SHORT revalidation (Aug 1 – Oct 6 only)
# R4:  Regime characterization (SPY, VIX, sector dispersion)
# N1:  LONG gap-up momentum (bull regime test)
# N2:  Sector rotation analysis
# N5:  Sympathy / sector peer plays
# N9:  Float-adjusted gap quality
# N10: E2-A + A6 mirror portfolio (SHORT gap-up + LONG gap-down)
#
# Colab-cell friendly. Config vars at top. No if __name__.

import os, sys, time, re, asyncio, copy, json
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

from scipy import stats as sp_stats

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# =============================================================================
# CONFIG
# =============================================================================

BACKTEST_START = '2026-08-01'
BACKTEST_END   = '2026-10-06'

# For N1 full-period comparison
FULL_START     = '2026-01-02'
FULL_END       = '2026-10-06'

TOP_N          = 5

GAP_MIN_PCT    = 3.0          # gap-up minimum for SHORT / LONG-momentum
GAP_DOWN_MIN   = 5.0          # gap-down minimum for A6 mirror (N10)
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
PRE_FILTER_N   = 30

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

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


# =============================================================================
# HELPERS  (exact copies from agent1_entry_mechanics.py)
# =============================================================================

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
    """Return the n prior trading days (ascending order)."""
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    out = []
    cur = d - timedelta(days=1)
    for _ in range(60):
        if _is_td(cur):
            out.append(cur)
            if len(out) >= n: return list(reversed(out))
        cur -= timedelta(days=1)
    return list(reversed(out))

def _pg(url, params=None):
    p = dict(params or {}); p['apiKey'] = POLYGON_API_KEY
    for attempt in range(5):
        try:
            r = requests.get(url, params=p, timeout=30)
            if r.status_code == 429:
                wait = 12 * (attempt + 1)
                print(f"    [429 rate limit] waiting {wait}s...")
                time.sleep(wait)
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
                bar_d = {'o':float(b['o']),'h':float(b['h']),
                         'l':float(b['l']),'c':float(b['c']),'v':int(b['v'])}
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
            return tk, {'type': res.get('type'), 'name': res.get('name', ''),
                        'sic_code': res.get('sic_code', ''),
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
    """Classify gap-up open pattern (for SHORT)."""
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


def classify_gap_down_pattern(rth_bars):
    """Classify gap-down open pattern (for LONG A6 mirror)."""
    if len(rth_bars) < 60: return 'no_data'
    high_30m = max(b[1]['h'] for b in rth_bars[:30])
    close_60m = rth_bars[59][1]['c']
    open_p = rth_bars[0][1]['o']
    rise = (high_30m - open_p) / abs(open_p) if open_p != 0 else 0
    recovery = (close_60m - open_p) / abs(open_p) if open_p != 0 else 0
    if rise > 0.03 and recovery > 0.5 * rise:
        return 'sustained_rise'
    elif rise > 0.03 and recovery <= 0.5 * rise:
        return 'rise_then_drop'
    elif rise <= 0.01:
        return 'continued_drop'
    return 'mild_rise'


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
                        'entry_type': 'local_high', 'peak_bar': i, 'entry_bar': ci}
    fb = max(fallback_bar, min_entry_bar) if fallback_bar is not None else None
    if fb is not None and fb < len(bars):
        return {'entry_price': bars[fb][1]['c'], 'entry_time': bars[fb][0],
                'entry_type': 'fallback', 'peak_bar': None, 'entry_bar': fb}
    return None


def find_local_low(bars, window=PEAK_WINDOW, max_trough_bar=MAX_PEAK_BAR,
                   fallback_bar=FALLBACK_BAR):
    """Mirror of find_local_high for gap-down LONG entries."""
    if len(bars) < window * 2 + 1: return None
    for i in range(window, min(max_trough_bar, len(bars) - window)):
        cl = bars[i][1]['l']
        if all(bars[i - j][1]['l'] > cl and bars[i + j][1]['l'] > cl
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(bars):
                return {'entry_price': bars[ci][1]['c'], 'entry_time': bars[ci][0],
                        'entry_type': 'local_low', 'entry_bar': ci,
                        'trough_bar': i}
    if fallback_bar is not None and fallback_bar < len(bars):
        return {'entry_price': bars[fallback_bar][1]['c'], 'entry_time': bars[fallback_bar][0],
                'entry_type': 'fallback', 'entry_bar': fallback_bar,
                'trough_bar': None}
    return None


def _T_to_min_entry_bar(T):
    h, m = int(T[:2]), int(T[3:5])
    return max(0, h * 60 + m - 9 * 60 - 30)

def get_price_at_T(bar_data, T):
    if bar_data is None: return None
    all_bars = (bar_data.get('pm') or []) + (bar_data.get('rth') or [])
    best = None
    for t_str, bd in all_bars:
        if t_str <= T: best = bd['c']
        else: break
    return best

def find_exit_price(bar_data, entry_time, exit_time='17:00'):
    session_bars = bar_data.get('rth', []) + bar_data.get('ah', [])
    exit_p = None
    for t_str, bar in session_bars:
        if t_str > exit_time: break
        if t_str >= entry_time: exit_p = bar['c']
    return exit_p

def short_pnl(entry_price, exit_price):
    if exit_price is None or entry_price is None or entry_price <= 0: return None
    return round((entry_price - exit_price) / entry_price * 100, 3)

def long_pnl(entry_price, exit_price):
    if exit_price is None or entry_price is None or entry_price <= 0: return None
    return round((exit_price - entry_price) / entry_price * 100, 3)


def compute_metrics(pnl_list, label=''):
    if not pnl_list:
        return {'label':label,'n':0,'avg':0,'med':0,'std':0,'wr':0,
                'sharpe':0,'pf':0,'ci_lo':0,'ci_hi':0,'trimmed_mean':0,
                'max_loss':0,'max_gain':0}
    a = np.array(pnl_list, dtype=float)
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


# =============================================================================
# GAP-UP SCANNER  (for SHORT E2-A and LONG momentum)
# =============================================================================

def scan_day_gapup(dt_str, gap_min_pct=GAP_MIN_PCT):
    """Scan for gap-up >= gap_min_pct% CS candidates."""
    if not _is_td(dt_str): return []
    prior_str = _prior_td(dt_str).strftime('%Y-%m-%d')
    prior = fetch_grouped_daily(prior_str)
    today = fetch_grouped_daily(dt_str)
    candidates = []
    gap_min = gap_min_pct / 100.0
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
        candidates.append({'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                           'open_price': cur_p, 'scan_date': dt_str,
                           'volume_today': today[tk].get('v', 0), 'prior_str': prior_str})
    candidates.sort(key=lambda x: -x['gap_pct'])
    nxt = _next_n_td(dt_str, 2)
    t1 = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
    for s in candidates:
        s['t1_date'] = t1
    return candidates


def scan_day_gapdown(dt_str, gap_min_pct=GAP_DOWN_MIN):
    """Scan for gap-down >= gap_min_pct% CS candidates (for LONG A6)."""
    if not _is_td(dt_str): return []
    prior_str = _prior_td(dt_str).strftime('%Y-%m-%d')
    prior = fetch_grouped_daily(prior_str)
    today = fetch_grouped_daily(dt_str)
    candidates = []
    gap_min = gap_min_pct / 100.0
    for tk in today:
        if tk in _EXCLUDE_TK: continue
        if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
        if tk not in prior: continue
        prev_c = prior[tk]['c']
        if prev_c <= 0: continue
        cur_p = today[tk]['o']
        if cur_p <= 0 or cur_p < PRICE_MIN: continue
        gap = (prev_c - cur_p) / prev_c  # positive = stock dropped
        if gap < gap_min: continue
        fs = _FLOAT.get(tk)
        if fs is not None and fs < FLOAT_MIN: continue
        candidates.append({'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                           'open_price': cur_p, 'scan_date': dt_str,
                           'volume_today': today[tk].get('v', 0), 'prior_str': prior_str})
    candidates.sort(key=lambda x: -x['gap_pct'])
    return candidates


# =============================================================================
# SIC CODE SECTOR MAPPING
# =============================================================================

_SIC_SECTORS = {
    '01': 'Agriculture', '02': 'Agriculture', '07': 'Agriculture', '08': 'Agriculture', '09': 'Agriculture',
    '10': 'Mining', '12': 'Mining', '13': 'Oil&Gas', '14': 'Mining',
    '15': 'Construction', '16': 'Construction', '17': 'Construction',
    '20': 'Food', '21': 'Tobacco', '22': 'Textiles', '23': 'Apparel',
    '24': 'Lumber', '25': 'Furniture', '26': 'Paper', '27': 'Publishing',
    '28': 'Chemicals/Pharma', '29': 'Petroleum', '30': 'Rubber/Plastic',
    '31': 'Leather', '32': 'Stone/Glass', '33': 'Metals', '34': 'Fabricated Metals',
    '35': 'Machinery', '36': 'Electronics', '37': 'Transportation Equip',
    '38': 'Instruments', '39': 'Manufacturing',
    '40': 'Railroads', '41': 'Transit', '42': 'Trucking', '43': 'USPS',
    '44': 'Water Transport', '45': 'Air Transport', '46': 'Pipelines', '47': 'Transport Services',
    '48': 'Telecom', '49': 'Utilities',
    '50': 'Wholesale-Durable', '51': 'Wholesale-Nondurable',
    '52': 'Retail-Building', '53': 'Retail-General', '54': 'Retail-Food',
    '55': 'Retail-Auto', '56': 'Retail-Apparel', '57': 'Retail-Home',
    '58': 'Retail-Eating', '59': 'Retail-Misc',
    '60': 'Banking', '61': 'Credit', '62': 'Securities', '63': 'Insurance',
    '64': 'Insurance Agents', '65': 'Real Estate', '67': 'Holding Companies',
    '70': 'Hotels', '72': 'Personal Services', '73': 'Business Services/Tech',
    '75': 'Auto Repair', '76': 'Misc Repair', '78': 'Motion Pictures',
    '79': 'Recreation', '80': 'Health Services', '81': 'Legal',
    '82': 'Education', '83': 'Social Services', '84': 'Museums',
    '86': 'Membership Orgs', '87': 'Engineering/Accounting',
    '89': 'Services NEC', '91': 'Government', '92': 'Government',
    '93': 'Government', '94': 'Government', '95': 'Government',
    '96': 'Government', '97': 'Government', '99': 'Nonclassifiable',
}

def sic_to_sector(sic_code):
    if not sic_code or len(str(sic_code)) < 2:
        return 'Unknown'
    prefix = str(sic_code)[:2]
    return _SIC_SECTORS.get(prefix, 'Unknown')


# =============================================================================
# PIPELINE: FETCH ALL DATA
# =============================================================================

t_total = time.time()
print(f"\n{'='*80}")
print(f"  AGENT 1 — AUG-OCT 2026 INVESTIGATION")
print(f"  {BACKTEST_START} -> {BACKTEST_END}")
print(f"  Experiments: R3, R4, N1, N2, N5, N9, N10")
print(f"{'='*80}")

# Build trading day lists
d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
dates = []
while d <= end:
    if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
    d += timedelta(days=1)
print(f"\n  {len(dates)} trading days in Aug-Oct window")

# Also build full-period dates for N1 comparison
d_full = datetime.strptime(FULL_START, '%Y-%m-%d').date()
end_full = datetime.strptime(FULL_END, '%Y-%m-%d').date()
full_dates = []
while d_full <= end_full:
    if _is_td(d_full): full_dates.append(d_full.strftime('%Y-%m-%d'))
    d_full += timedelta(days=1)
print(f"  {len(full_dates)} trading days in full period ({FULL_START} -> {FULL_END})")

# We need lookback data for regime: fetch 30 days before BACKTEST_START
lookback_start = _prior_n_td(BACKTEST_START, 30)
lookback_dates = [d.strftime('%Y-%m-%d') for d in lookback_start]
all_daily_dates = sorted(set(lookback_dates + full_dates))


# ============================================================
# Phase 1: Scan gap-ups across full period (for N1 full comparison)
# ============================================================
print(f"\n  Phase 1: Scanning gap-ups for full period...")
all_gapup_candidates = {}
for i, dt in enumerate(full_dates):
    all_gapup_candidates[dt] = scan_day_gapup(dt)
    if (i+1) % 30 == 0 or i == len(full_dates) - 1:
        print(f"    [{i+1:>3}/{len(full_dates)}] {dt}  |  {time.time()-t_total:.0f}s")

# Also prefetch lookback daily data for regime
print(f"  Fetching lookback daily data for regime...")
for dt in lookback_dates:
    try: fetch_grouped_daily(dt)
    except: pass

# Scan gap-downs for Aug-Oct only (for N10)
print(f"\n  Phase 1b: Scanning gap-downs for Aug-Oct (N10)...")
all_gapdown_candidates = {}
for dt in dates:
    all_gapdown_candidates[dt] = scan_day_gapdown(dt)
print(f"  -> {sum(len(v) for v in all_gapdown_candidates.values())} raw gap-down candidates")


# ============================================================
# Phase 2: Fetch minute bars
# ============================================================
all_pairs = set()
# Gap-up candidates (full period for N1, Aug-Oct for everything else)
for dt, cands in all_gapup_candidates.items():
    for c in cands[:PRE_FILTER_N]:
        all_pairs.add((c['ticker'], dt))
        if c.get('prior_str'): all_pairs.add((c['ticker'], c['prior_str']))

# Gap-down candidates (Aug-Oct)
for dt, cands in all_gapdown_candidates.items():
    for c in cands[:PRE_FILTER_N]:
        all_pairs.add((c['ticker'], dt))
        if c.get('prior_str'): all_pairs.add((c['ticker'], c['prior_str']))

# SPY, IWM, XLK for regime (full period)
for dt in all_daily_dates:
    all_pairs.add(('SPY', dt))
for dt in dates:
    all_pairs.add(('IWM', dt))
    all_pairs.add(('XLK', dt))

print(f"\n  Phase 2: Fetching {len(all_pairs)} bar pairs...")
pairs_list = list(all_pairs)
all_bars = {}
for cs in range(0, len(pairs_list), 100):
    chunk = pairs_list[cs:cs + 100]
    all_bars.update(fetch_minute_bars_batch(chunk))
    done = min(cs + 100, len(pairs_list))
    if done % 500 == 0 or done == len(pairs_list):
        print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
print(f"  -> {len(all_bars)}/{len(all_pairs)} pairs fetched")


# ============================================================
# Phase 2a: AH freshness filter (for gap-ups)
# ============================================================
gap_min_final = GAP_MIN_PCT / 100.0
print(f"\n  Phase 2a: AH freshness filter...")
n_stale = 0
for dt in full_dates:
    for c in all_gapup_candidates.get(dt, [])[:PRE_FILTER_N]:
        tk = c['ticker']
        prior_bars = all_bars.get((tk, c.get('prior_str', '')))
        if prior_bars is None: continue
        ah = prior_bars.get('ah', [])
        if ah:
            ah_ref = ah[-1][1]['c']
            c['ah_ref_price'] = ah_ref
            fresh_gap = (c['open_price'] - ah_ref) / ah_ref
            c['gap_pct_daily'] = c['gap_pct']
            c['gap_pct'] = fresh_gap
            if fresh_gap < gap_min_final:
                c['ah_stale'] = True; n_stale += 1
print(f"  -> {n_stale} stale excluded")


# ============================================================
# Phase 2b: Re-rank + erosion (for gap-ups)
# ============================================================
print(f"  Phase 2b: Re-rank + gap erosion...")
n_eroded = 0
for dt in full_dates:
    reranked = []
    for c in all_gapup_candidates.get(dt, []):
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
                gc = (daily_open - low_to_scan) / gap_range
                c['gap_consumed'] = gc
                if gc > GAP_CONSUMED_MAX:
                    n_eroded += 1; continue
        reranked.append(c)
    reranked.sort(key=lambda x: -x['gap_pct'])
    all_gapup_candidates[dt] = reranked
print(f"  -> {n_eroded} eroded")


# ============================================================
# Phase 3: Fetch ticker reference data (CS filter + SIC codes)
# ============================================================
all_tickers = set()
for cands in all_gapup_candidates.values():
    for c in cands[:PRE_FILTER_N]: all_tickers.add(c['ticker'])
for cands in all_gapdown_candidates.values():
    for c in cands[:PRE_FILTER_N]: all_tickers.add(c['ticker'])
print(f"\n  Phase 3: Fetching refs for {len(all_tickers)} tickers...")
for cs_i in range(0, len(sorted(all_tickers)), 200):
    fetch_ticker_refs(sorted(all_tickers)[cs_i:cs_i + 200])
n_cs = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('type') == 'CS')
print(f"  -> {n_cs} CS / {len(all_tickers)} total")

# Precompute patterns and entries for Aug-Oct gap-up candidates
min_eb = _T_to_min_entry_bar(SCAN_TIME_ET)
pattern_cache = {}
entry_cache = {}
for key, bar_data in all_bars.items():
    if not isinstance(bar_data, dict): continue
    rth_bars = bar_data.get('rth', [])
    pattern_cache[key] = classify_open_pattern(rth_bars)
    entry_cache[key] = find_local_high(rth_bars, min_entry_bar=min_eb) \
        if rth_bars and len(rth_bars) >= PEAK_WINDOW * 2 + 1 else None

print(f"\n  Pipeline done: {time.time()-t_total:.0f}s")


# =============================================================================
# R3. E2-A SHORT REVALIDATION (Aug-Oct ONLY)
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  R3 -- E2-A SHORT REVALIDATION (Aug 1 - Oct 6)")
print(f"{'='*80}")

r3_trades = []
for dt in dates:
    cs_qualified = []
    for c in all_gapup_candidates.get(dt, [])[:PRE_FILTER_N]:
        if _ticker_ref.get(c['ticker'], {}).get('type') != 'CS': continue
        cs_qualified.append(copy.deepcopy(c))
    for c in cs_qualified[:TOP_N]:
        tk = c['ticker']
        entry = entry_cache.get((tk, dt))
        if entry is None: continue
        tc = copy.deepcopy(c)
        tc['entry_price'] = entry['entry_price']
        tc['entry_time'] = entry['entry_time']
        tc['entry_type'] = entry['entry_type']
        tc['peak_bar'] = entry.get('peak_bar')
        tc['entry_bar'] = entry.get('entry_bar', FALLBACK_BAR)
        ep = tc['entry_price']

        bar_data = all_bars.get((tk, dt))
        exit_price = find_exit_price(bar_data, tc['entry_time'], EXIT_TIME) if bar_data else None

        if exit_price is not None:
            tc['pnl_eod'] = round((ep - exit_price) / ep * 100, 3)
        else:
            t0d = _gd_cache.get(dt, {}).get(tk, {})
            tc['pnl_eod'] = round((ep - t0d['c']) / ep * 100, 3) if t0d.get('c') else None

        tc['exit_price'] = exit_price
        tc['open_pattern'] = pattern_cache.get((tk, dt), 'no_data')

        # T+1 close
        if c.get('t1_date'):
            t1d = _gd_cache.get(c['t1_date'], {}).get(tk, {})
            tc['pnl_t1'] = round((ep - t1d['c']) / ep * 100, 3) if t1d.get('c') else None
        else:
            tc['pnl_t1'] = None

        if tc.get('pnl_eod') is not None:
            r3_trades.append(tc)

r3_m = compute_metrics([t['pnl_eod'] for t in r3_trades], 'R3 E2-A Aug-Oct')

print(f"\n  R3 E2-A SHORT revalidation (Aug 1 - Oct 6):")
print(f"    n = {r3_m['n']}")
print(f"    avg PnL = {r3_m['avg']:+.2f}%")
print(f"    median  = {r3_m['med']:+.2f}%")
print(f"    Sharpe  = {r3_m['sharpe']:.3f}")
print(f"    WR      = {r3_m['wr']*100:.1f}%")
print(f"    CI 95%  = [{r3_m['ci_lo']:+.2f}%, {r3_m['ci_hi']:+.2f}%]")
print(f"    trimmed = {r3_m['trimmed_mean']:+.2f}%")
print(f"    PF      = {r3_m['pf']:.2f}")
if r3_m['n'] > 0:
    print(f"    max loss = {r3_m['max_loss']:+.2f}%")
    print(f"    max gain = {r3_m['max_gain']:+.2f}%")

# Entry type breakdown
lh_n = sum(1 for t in r3_trades if t['entry_type'] == 'local_high')
fb_n = sum(1 for t in r3_trades if t['entry_type'] == 'fallback')
print(f"    entry types: local_high={lh_n}, fallback={fb_n}")

# Pattern breakdown
pat_counts = defaultdict(list)
for t in r3_trades:
    pat_counts[t.get('open_pattern', 'no_data')].append(t['pnl_eod'])
print(f"\n    Pattern breakdown:")
pat_rows = []
for pat, pnls in sorted(pat_counts.items(), key=lambda x: -len(x[1])):
    m = compute_metrics(pnls)
    pat_rows.append([pat, m['n'], f"{m['avg']:+.2f}%", f"{m['wr']*100:.0f}%", f"{m['sharpe']:.3f}"])
print(tabulate(pat_rows, headers=['Pattern', 'N', 'Avg PnL', 'WR', 'Sharpe'], tablefmt='simple'))

# Monthly breakdown
monthly = defaultdict(list)
for t in r3_trades:
    mo = t['scan_date'][:7]
    monthly[mo].append(t['pnl_eod'])
print(f"\n    Monthly breakdown:")
mo_rows = []
for mo in sorted(monthly.keys()):
    m = compute_metrics(monthly[mo])
    mo_rows.append([mo, m['n'], f"{m['avg']:+.2f}%", f"{m['wr']*100:.0f}%", f"{m['sharpe']:.3f}"])
print(tabulate(mo_rows, headers=['Month', 'N', 'Avg PnL', 'WR', 'Sharpe'], tablefmt='simple'))

# Compare to Phase 1 full-period reference
print(f"\n    Phase 1 reference (full period):")
print(f"      E2-A: n=476, avg=+12.09%, WR=78%, Sharpe=0.254, CI=[+7.82, +16.36]")

exp_results = {}
exp_results['R3'] = {
    'metrics': r3_m,
    'n_local_high': lh_n,
    'n_fallback': fb_n,
    'monthly': {mo: compute_metrics(pnls) for mo, pnls in monthly.items()},
}


# =============================================================================
# R4. REGIME CHARACTERIZATION
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  R4 -- REGIME CHARACTERIZATION (Aug 1 - Oct 6)")
print(f"{'='*80}")

# Gather SPY daily data for full dates + lookback
spy_daily = {}
for dt in all_daily_dates:
    gd = _gd_cache.get(dt, {})
    if 'SPY' in gd:
        spy_daily[dt] = gd['SPY']

spy_dates_sorted = sorted(spy_daily.keys())
spy_closes = {dt: spy_daily[dt]['c'] for dt in spy_dates_sorted}

# Compute regime features for Aug-Oct days
regime_data = {}
for dt in dates:
    # Find position in spy_dates_sorted
    if dt not in spy_closes: continue
    idx = spy_dates_sorted.index(dt)

    # SPY 5d cumulative return
    if idx >= 5:
        spy5d_ret = (spy_closes[spy_dates_sorted[idx]] - spy_closes[spy_dates_sorted[idx-5]]) / spy_closes[spy_dates_sorted[idx-5]]
    else:
        spy5d_ret = 0.0

    # SPY 10d realized vol
    if idx >= 10:
        rets_10d = []
        for j in range(idx-9, idx+1):
            if j > 0:
                rets_10d.append(
                    (spy_closes[spy_dates_sorted[j]] - spy_closes[spy_dates_sorted[j-1]])
                    / spy_closes[spy_dates_sorted[j-1]]
                )
        vol_10d = np.std(rets_10d) * np.sqrt(252) if rets_10d else 0
    else:
        vol_10d = 0.0

    # SPY 20d trend slope (linear regression)
    if idx >= 20:
        window_closes = [spy_closes[spy_dates_sorted[idx-20+k]] for k in range(21)]
        x_arr = np.arange(21)
        slope, _, _, _, _ = sp_stats.linregress(x_arr, window_closes)
        slope_20d = slope / np.mean(window_closes) * 100  # normalized to %/day
    else:
        slope_20d = 0.0

    # Gap-universe size for this day
    n_gapup = len(all_gapup_candidates.get(dt, []))

    regime_data[dt] = {
        'spy5d_ret': spy5d_ret,
        'vol_10d': vol_10d,
        'slope_20d': slope_20d,
        'n_gapup': n_gapup,
    }

# Classify regimes
vol_values = [v['vol_10d'] for v in regime_data.values()]
vol_median = np.median(vol_values) if vol_values else 0.15

regime_days = defaultdict(list)
for dt, rd in regime_data.items():
    if rd['spy5d_ret'] > 0 and rd['vol_10d'] < vol_median:
        regime = 'calm_bull'
    elif rd['spy5d_ret'] <= 0 and rd['vol_10d'] < vol_median:
        regime = 'calm_bear'
    elif rd['spy5d_ret'] > 0 and rd['vol_10d'] >= vol_median:
        regime = 'vol_bull'
    else:
        regime = 'vol_bear'
    rd['regime'] = regime
    regime_days[regime].append(dt)

print(f"\n  Vol 10d median: {vol_median:.2%}")
print(f"\n  Regime distribution:")
reg_rows = []
for regime in ['calm_bull', 'calm_bear', 'vol_bull', 'vol_bear']:
    dts = regime_days[regime]
    avg_gap_n = np.mean([regime_data[d]['n_gapup'] for d in dts]) if dts else 0
    avg_spy5d = np.mean([regime_data[d]['spy5d_ret'] for d in dts]) if dts else 0
    avg_vol = np.mean([regime_data[d]['vol_10d'] for d in dts]) if dts else 0
    reg_rows.append([regime, len(dts), f"{avg_spy5d:+.2%}", f"{avg_vol:.2%}", f"{avg_gap_n:.1f}"])
print(tabulate(reg_rows, headers=['Regime', 'N Days', 'Avg SPY 5d', 'Avg Vol 10d', 'Avg Gap-Up Universe'],
               tablefmt='simple'))

# E2-A SHORT PnL by regime
print(f"\n  R3 SHORT PnL by regime:")
for regime in ['calm_bull', 'calm_bear', 'vol_bull', 'vol_bear']:
    dts_set = set(regime_days[regime])
    pnls = [t['pnl_eod'] for t in r3_trades if t['scan_date'] in dts_set]
    m = compute_metrics(pnls, regime)
    print(f"    {regime:12s}: n={m['n']:3d}  avg={m['avg']:+.2f}%  Sharpe={m['sharpe']:.3f}  WR={m['wr']*100:.0f}%")

# IWM/SPY and XLK/SPY ratios
print(f"\n  Sector dispersion (IWM/SPY, XLK/SPY):")
dispersion_rows = []
for dt in dates[:5] + dates[-5:]:  # show first/last 5 days
    spy_c = spy_closes.get(dt, 0)
    iwm_c = _gd_cache.get(dt, {}).get('IWM', {}).get('c', 0)
    xlk_c = _gd_cache.get(dt, {}).get('XLK', {}).get('c', 0)
    iwm_spy = iwm_c / spy_c if spy_c > 0 else 0
    xlk_spy = xlk_c / spy_c if spy_c > 0 else 0
    r_data = regime_data.get(dt, {})
    dispersion_rows.append([dt, f"{iwm_spy:.4f}", f"{xlk_spy:.4f}", r_data.get('regime', '?')])
print(tabulate(dispersion_rows, headers=['Date', 'IWM/SPY', 'XLK/SPY', 'Regime'], tablefmt='simple'))

# SPY 20d slope distribution
slopes = [v['slope_20d'] for v in regime_data.values()]
print(f"\n  SPY 20d trend slope: mean={np.mean(slopes):+.4f}%/day, med={np.median(slopes):+.4f}%/day")

# VIX proxy: try VIXY
print(f"\n  VIX proxy (VIXY daily):")
vixy_data = {}
for dt in dates:
    gd = _gd_cache.get(dt, {})
    if 'VIXY' in gd:
        vixy_data[dt] = gd['VIXY']['c']
    elif 'VIX' in gd:
        vixy_data[dt] = gd['VIX']['c']
if vixy_data:
    vix_vals = list(vixy_data.values())
    print(f"    n={len(vix_vals)}, mean={np.mean(vix_vals):.2f}, min={np.min(vix_vals):.2f}, max={np.max(vix_vals):.2f}")
else:
    print(f"    No VIX/VIXY data found in grouped daily")

exp_results['R4'] = {
    'regime_distribution': {r: len(d) for r, d in regime_days.items()},
    'vol_median': round(vol_median, 4),
    'spy_slope_mean': round(float(np.mean(slopes)), 4),
}


# =============================================================================
# N1. LONG GAP-UP MOMENTUM (Bull Regime Test)
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  N1 -- LONG GAP-UP MOMENTUM (Bull Regime Test)")
print(f"{'='*80}")

def build_long_gapup_trades(date_list, label=''):
    """Build LONG trades on gap-up candidates using same entry mechanics."""
    trades = []
    for dt in date_list:
        cs_qualified = []
        for c in all_gapup_candidates.get(dt, [])[:PRE_FILTER_N]:
            if _ticker_ref.get(c['ticker'], {}).get('type') != 'CS': continue
            cs_qualified.append(copy.deepcopy(c))
        for c in cs_qualified[:TOP_N]:
            tk = c['ticker']
            entry = entry_cache.get((tk, dt))
            if entry is None: continue
            tc = copy.deepcopy(c)
            tc['entry_price'] = entry['entry_price']
            tc['entry_time'] = entry['entry_time']
            tc['entry_type'] = entry['entry_type']
            tc['entry_bar'] = entry.get('entry_bar', FALLBACK_BAR)
            ep = tc['entry_price']

            bar_data = all_bars.get((tk, dt))
            exit_p_eod = find_exit_price(bar_data, tc['entry_time'], EXIT_TIME) if bar_data else None
            if exit_p_eod is not None:
                tc['pnl_eod'] = long_pnl(ep, exit_p_eod)
            else:
                t0d = _gd_cache.get(dt, {}).get(tk, {})
                tc['pnl_eod'] = long_pnl(ep, t0d.get('c')) if t0d.get('c') else None

            # T+1 close
            if c.get('t1_date'):
                t1d = _gd_cache.get(c['t1_date'], {}).get(tk, {})
                tc['pnl_t1'] = long_pnl(ep, t1d.get('c')) if t1d.get('c') else None
            else:
                tc['pnl_t1'] = None

            if tc.get('pnl_eod') is not None:
                trades.append(tc)
    return trades

# Aug-Oct LONG momentum trades
n1_trades_aug = build_long_gapup_trades(dates, 'N1 Aug-Oct')
n1_m_aug = compute_metrics([t['pnl_eod'] for t in n1_trades_aug], 'N1 Aug-Oct EOD')
n1_m_aug_t1 = compute_metrics([t['pnl_t1'] for t in n1_trades_aug if t.get('pnl_t1') is not None], 'N1 Aug-Oct T+1')

print(f"\n  N1 LONG gap-up momentum (Aug-Oct, EOD exit):")
print(f"    n = {n1_m_aug['n']},  avg = {n1_m_aug['avg']:+.2f}%,  Sharpe = {n1_m_aug['sharpe']:.3f},  WR = {n1_m_aug['wr']*100:.0f}%")
print(f"    CI 95% = [{n1_m_aug['ci_lo']:+.2f}%, {n1_m_aug['ci_hi']:+.2f}%]")

print(f"\n  N1 LONG gap-up momentum (Aug-Oct, T+1 exit):")
print(f"    n = {n1_m_aug_t1['n']},  avg = {n1_m_aug_t1['avg']:+.2f}%,  Sharpe = {n1_m_aug_t1['sharpe']:.3f}")

# Full-period LONG conditioned on SPY 5d > 0
# Need SPY daily for full period too
spy_daily_full = {}
for dt in full_dates:
    gd = _gd_cache.get(dt, {})
    if 'SPY' in gd:
        spy_daily_full[dt] = gd['SPY']['c']

full_dates_sorted = sorted(spy_daily_full.keys())

def is_spy_5d_bull(dt):
    """Check if SPY 5d return > 0 for a given date."""
    if dt not in spy_daily_full: return False
    try:
        idx = full_dates_sorted.index(dt)
    except ValueError:
        return False
    if idx < 5: return False
    ret = (spy_daily_full[full_dates_sorted[idx]] - spy_daily_full[full_dates_sorted[idx-5]]) / spy_daily_full[full_dates_sorted[idx-5]]
    return ret > 0

bull_dates_full = [dt for dt in full_dates if is_spy_5d_bull(dt)]
print(f"\n  SPY 5d>0 bull days in full period: {len(bull_dates_full)} / {len(full_dates)}")

n1_trades_full_bull = build_long_gapup_trades(bull_dates_full, 'N1 full-bull')
n1_m_full_bull = compute_metrics([t['pnl_eod'] for t in n1_trades_full_bull], 'N1 Full Bull EOD')

print(f"\n  N1 LONG gap-up momentum (full period, SPY 5d>0 only, EOD exit):")
print(f"    n = {n1_m_full_bull['n']},  avg = {n1_m_full_bull['avg']:+.2f}%,  Sharpe = {n1_m_full_bull['sharpe']:.3f},  WR = {n1_m_full_bull['wr']*100:.0f}%")
print(f"    CI 95% = [{n1_m_full_bull['ci_lo']:+.2f}%, {n1_m_full_bull['ci_hi']:+.2f}%]")

# Also the SHORT E2-A on bull vs bear days for comparison
bull_dates_aug = [dt for dt in dates if is_spy_5d_bull(dt)]
bear_dates_aug = [dt for dt in dates if not is_spy_5d_bull(dt)]
r3_bull = [t for t in r3_trades if t['scan_date'] in set(bull_dates_aug)]
r3_bear = [t for t in r3_trades if t['scan_date'] in set(bear_dates_aug)]
print(f"\n  R3 SHORT PnL: bull days avg={compute_metrics([t['pnl_eod'] for t in r3_bull])['avg']:+.2f}% n={len(r3_bull)}")
print(f"  R3 SHORT PnL: bear days avg={compute_metrics([t['pnl_eod'] for t in r3_bear])['avg']:+.2f}% n={len(r3_bear)}")

exp_results['N1'] = {
    'aug_oct_eod': n1_m_aug,
    'aug_oct_t1': n1_m_aug_t1,
    'full_bull_eod': n1_m_full_bull,
}


# =============================================================================
# N2. SECTOR ROTATION
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  N2 -- SECTOR ROTATION ANALYSIS")
print(f"{'='*80}")

# Classify all gap-up candidates by sector
# Split into Jan-Jul vs Aug-Oct
jan_jul_dates = [dt for dt in full_dates if dt < BACKTEST_START]

sector_counts_pre = defaultdict(int)
sector_counts_post = defaultdict(int)
sector_short_pnls = defaultdict(list)  # Aug-Oct SHORT PnL by sector
sector_long_pnls = defaultdict(list)   # Aug-Oct LONG PnL by sector

# Build sector map
for dt in full_dates:
    for c in all_gapup_candidates.get(dt, [])[:PRE_FILTER_N]:
        if _ticker_ref.get(c['ticker'], {}).get('type') != 'CS': continue
        tk = c['ticker']
        sic = _ticker_ref.get(tk, {}).get('sic_code', '')
        sector = sic_to_sector(sic)
        if dt < BACKTEST_START:
            sector_counts_pre[sector] += 1
        else:
            sector_counts_post[sector] += 1

# Map R3 trades to sectors
for t in r3_trades:
    sic = _ticker_ref.get(t['ticker'], {}).get('sic_code', '')
    sector = sic_to_sector(sic)
    sector_short_pnls[sector].append(t['pnl_eod'])

# Map N1 trades to sectors for LONG PnL
for t in n1_trades_aug:
    sic = _ticker_ref.get(t['ticker'], {}).get('sic_code', '')
    sector = sic_to_sector(sic)
    sector_long_pnls[sector].append(t['pnl_eod'])

# Report
all_sectors = sorted(set(list(sector_counts_pre.keys()) + list(sector_counts_post.keys())),
                     key=lambda s: -(sector_counts_pre.get(s, 0) + sector_counts_post.get(s, 0)))

print(f"\n  Gap-up candidate counts by sector (Jan-Jul vs Aug-Oct):")
sect_rows = []
for s in all_sectors[:15]:
    pre_n = sector_counts_pre.get(s, 0)
    post_n = sector_counts_post.get(s, 0)
    n_months_pre = 7  # Jan-Jul
    n_months_post = 2.2  # Aug-Oct partial
    rate_pre = pre_n / n_months_pre
    rate_post = post_n / n_months_post
    short_m = compute_metrics(sector_short_pnls.get(s, []))
    long_m = compute_metrics(sector_long_pnls.get(s, []))
    sect_rows.append([s, pre_n, f"{rate_pre:.1f}/mo", post_n, f"{rate_post:.1f}/mo",
                      f"{short_m['avg']:+.2f}%" if short_m['n'] > 0 else "---",
                      f"{long_m['avg']:+.2f}%" if long_m['n'] > 0 else "---"])
print(tabulate(sect_rows,
               headers=['Sector', 'Jan-Jul N', 'Rate', 'Aug-Oct N', 'Rate', 'SHORT Avg', 'LONG Avg'],
               tablefmt='simple'))

# Which sectors' gaps continue (LONG profitable) vs fade (SHORT profitable)
print(f"\n  Sector gap behavior (SHORT vs LONG profitability, Aug-Oct):")
for s in all_sectors[:10]:
    sp = sector_short_pnls.get(s, [])
    lp = sector_long_pnls.get(s, [])
    if sp and lp:
        short_avg = np.mean(sp)
        long_avg = np.mean(lp)
        verdict = "FADES (SHORT wins)" if short_avg > long_avg else "CONTINUES (LONG wins)"
        print(f"    {s:25s}: SHORT={short_avg:+.2f}% vs LONG={long_avg:+.2f}%  -> {verdict}")

exp_results['N2'] = {
    'sector_counts_pre': dict(sector_counts_pre),
    'sector_counts_post': dict(sector_counts_post),
    'sector_short_avg': {s: round(float(np.mean(v)), 3) for s, v in sector_short_pnls.items() if v},
    'sector_long_avg': {s: round(float(np.mean(v)), 3) for s, v in sector_long_pnls.items() if v},
}


# =============================================================================
# N5. SYMPATHY / SECTOR PEER PLAYS
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  N5 -- SYMPATHY / SECTOR PEER PLAYS")
print(f"{'='*80}")

# For gap-up candidates in Aug-Oct, check if same-day other stocks in same SIC
# group moved >= 1% pre-market but did NOT gap >= 3%
sympathy_trades = []
n5_gap_days = 0

for dt in dates:
    cs_gapups = []
    for c in all_gapup_candidates.get(dt, [])[:PRE_FILTER_N]:
        if _ticker_ref.get(c['ticker'], {}).get('type') != 'CS': continue
        cs_gapups.append(c)
    if not cs_gapups: continue
    n5_gap_days += 1

    # Build set of SIC groups that have gap-up candidates today
    gap_sic_groups = set()
    for c in cs_gapups[:TOP_N]:
        sic = _ticker_ref.get(c['ticker'], {}).get('sic_code', '')
        if sic and len(str(sic)) >= 2:
            gap_sic_groups.add(str(sic)[:2])

    if not gap_sic_groups: continue

    # Scan all tickers in today's data for sympathy candidates
    prior_str = _prior_td(dt).strftime('%Y-%m-%d') if _is_td(dt) else None
    if prior_str is None: continue
    today_data = _gd_cache.get(dt, {})
    prior_data = _gd_cache.get(prior_str, {})

    for tk in today_data:
        if tk in _EXCLUDE_TK: continue
        if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
        if _ticker_ref.get(tk, {}).get('type') != 'CS': continue
        if tk not in prior_data: continue

        prev_c = prior_data[tk]['c']
        if prev_c <= 0: continue
        cur_o = today_data[tk]['o']
        if cur_o <= 0 or cur_o < PRICE_MIN: continue

        gap = (cur_o - prev_c) / prev_c
        # Sympathy: moved >= 1% but < 3% (not a gap-up candidate itself)
        if gap < 0.01 or gap >= 0.03: continue

        # Check if this ticker's SIC matches a gap-up leader's SIC
        tk_sic = _ticker_ref.get(tk, {}).get('sic_code', '')
        if not tk_sic or len(str(tk_sic)) < 2: continue
        tk_sic_2 = str(tk_sic)[:2]
        if tk_sic_2 not in gap_sic_groups: continue

        # Enter LONG at open, exit EOD
        eod_data = today_data[tk]
        eod_close = eod_data['c']
        if eod_close <= 0: continue

        pnl = long_pnl(cur_o, eod_close)
        if pnl is not None:
            sympathy_trades.append({
                'ticker': tk, 'scan_date': dt, 'gap_pct': gap,
                'sic': tk_sic_2, 'pnl_eod': pnl,
                'sector': sic_to_sector(tk_sic),
            })

n5_m = compute_metrics([t['pnl_eod'] for t in sympathy_trades], 'N5 Sympathy')
print(f"\n  N5 Sympathy LONG trades (1-3% gap, same SIC as leader, enter at open, EOD exit):")
print(f"    n = {n5_m['n']}")
print(f"    avg PnL = {n5_m['avg']:+.2f}%")
print(f"    Sharpe  = {n5_m['sharpe']:.3f}")
print(f"    WR      = {n5_m['wr']*100:.0f}%")
print(f"    CI 95%  = [{n5_m['ci_lo']:+.2f}%, {n5_m['ci_hi']:+.2f}%]")
print(f"    Gap days with leaders: {n5_gap_days}")

if sympathy_trades:
    # Sector breakdown
    symp_sectors = defaultdict(list)
    for t in sympathy_trades:
        symp_sectors[t['sector']].append(t['pnl_eod'])
    print(f"\n    By sector:")
    for s, pnls in sorted(symp_sectors.items(), key=lambda x: -len(x[1])):
        print(f"      {s:25s}: n={len(pnls):3d}  avg={np.mean(pnls):+.2f}%")
else:
    print(f"\n    [THIN DATA] No sympathy trades found -- this is expected if SIC overlap is low")

exp_results['N5'] = n5_m


# =============================================================================
# N9. FLOAT-ADJUSTED GAP QUALITY
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  N9 -- FLOAT-ADJUSTED GAP QUALITY")
print(f"{'='*80}")

# Bin R3 trades by float
float_bins = {'low_10_50M': [], 'mid_50_200M': [], 'high_200M+': [], 'unknown': []}
for t in r3_trades:
    fs = _FLOAT.get(t['ticker'])
    if fs is None:
        float_bins['unknown'].append(t)
    elif fs < 50_000_000:
        float_bins['low_10_50M'].append(t)
    elif fs < 200_000_000:
        float_bins['mid_50_200M'].append(t)
    else:
        float_bins['high_200M+'].append(t)

print(f"\n  E2-A SHORT by float bin (Aug-Oct):")
float_rows = []
for fb_label in ['low_10_50M', 'mid_50_200M', 'high_200M+', 'unknown']:
    pnls = [t['pnl_eod'] for t in float_bins[fb_label]]
    m = compute_metrics(pnls, fb_label)
    float_rows.append([fb_label, m['n'], f"{m['avg']:+.2f}%", f"{m['med']:+.2f}%",
                        f"{m['sharpe']:.3f}", f"{m['wr']*100:.0f}%",
                        f"[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]"])
print(tabulate(float_rows,
               headers=['Float Bin', 'N', 'Avg PnL', 'Median', 'Sharpe', 'WR', 'CI 95%'],
               tablefmt='simple'))

# Hypothesis: high-float gaps revert more reliably
if float_bins['high_200M+'] and float_bins['low_10_50M']:
    high_pnls = [t['pnl_eod'] for t in float_bins['high_200M+']]
    low_pnls = [t['pnl_eod'] for t in float_bins['low_10_50M']]
    if len(high_pnls) > 2 and len(low_pnls) > 2:
        t_stat, p_val = sp_stats.ttest_ind(high_pnls, low_pnls)
        print(f"\n  High vs Low float t-test: t={t_stat:.3f}, p={p_val:.3f}")
        print(f"    High-float gaps {'DO' if np.mean(high_pnls) > np.mean(low_pnls) else 'DO NOT'} revert more reliably")

exp_results['N9'] = {
    fb_label: compute_metrics([t['pnl_eod'] for t in float_bins[fb_label]], fb_label)
    for fb_label in float_bins
}


# =============================================================================
# N10. E2-A + A6 MIRROR PORTFOLIO (SHORT gap-up + LONG gap-down)
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  N10 -- E2-A + A6 MIRROR PORTFOLIO")
print(f"{'='*80}")

# SHORT side: use R3 trades (already computed)
# LONG side: gap-down >= 5% CS candidates, sustained_rise pattern, local_low entry

a6_trades = []
gapdown_need_bars = set()
for dt in dates:
    for c in all_gapdown_candidates.get(dt, [])[:PRE_FILTER_N]:
        gapdown_need_bars.add((c['ticker'], dt))

# Fetch any missing minute bars for gap-down candidates
missing_gapdown = [p for p in gapdown_need_bars if p not in all_bars]
if missing_gapdown:
    print(f"  Fetching {len(missing_gapdown)} additional gap-down bar pairs...")
    for cs_i in range(0, len(missing_gapdown), 100):
        chunk = missing_gapdown[cs_i:cs_i + 100]
        all_bars.update(fetch_minute_bars_batch(chunk))
    print(f"  -> done")

# Fetch refs for gap-down tickers
gd_tickers = set(c['ticker'] for cands in all_gapdown_candidates.values() for c in cands[:PRE_FILTER_N])
uncached_gd = [tk for tk in gd_tickers if tk not in _ticker_ref]
if uncached_gd:
    fetch_ticker_refs(uncached_gd)

# Build A6 LONG trades
for dt in dates:
    cs_qualified = []
    for c in all_gapdown_candidates.get(dt, [])[:PRE_FILTER_N]:
        if _ticker_ref.get(c['ticker'], {}).get('type') != 'CS': continue
        cs_qualified.append(copy.deepcopy(c))

    for c in cs_qualified[:TOP_N]:
        tk = c['ticker']
        bar_data = all_bars.get((tk, dt))
        if bar_data is None: continue
        rth = bar_data.get('rth', [])
        if not rth or len(rth) < PEAK_WINDOW * 2 + 1: continue

        # Classify pattern
        pattern = classify_gap_down_pattern(rth)
        # Only take sustained_rise for the A6 mirror
        # (but also report all for analysis)

        # Entry via local_low
        entry = find_local_low(rth, window=PEAK_WINDOW, max_trough_bar=MAX_PEAK_BAR,
                               fallback_bar=FALLBACK_BAR)
        if entry is None: continue

        ep = entry['entry_price']
        exit_p = find_exit_price(bar_data, entry['entry_time'], EXIT_TIME)
        pnl = long_pnl(ep, exit_p)
        if pnl is None: continue

        a6_trades.append({
            'ticker': tk, 'scan_date': dt, 'gap_pct': c['gap_pct'],
            'entry_price': ep, 'entry_time': entry['entry_time'],
            'entry_type': entry['entry_type'],
            'pnl_eod': pnl, 'open_pattern': pattern,
            'exit_price': exit_p,
        })

# Report A6 standalone
a6_all_m = compute_metrics([t['pnl_eod'] for t in a6_trades], 'A6 All')
a6_sr = [t for t in a6_trades if t['open_pattern'] == 'sustained_rise']
a6_sr_m = compute_metrics([t['pnl_eod'] for t in a6_sr], 'A6 Sustained Rise')

print(f"\n  A6 LONG gap-down (all patterns):")
print(f"    n = {a6_all_m['n']},  avg = {a6_all_m['avg']:+.2f}%,  Sharpe = {a6_all_m['sharpe']:.3f},  WR = {a6_all_m['wr']*100:.0f}%")
print(f"\n  A6 LONG gap-down (sustained_rise only):")
print(f"    n = {a6_sr_m['n']},  avg = {a6_sr_m['avg']:+.2f}%,  Sharpe = {a6_sr_m['sharpe']:.3f},  WR = {a6_sr_m['wr']*100:.0f}%")

# Pattern breakdown
a6_pats = defaultdict(list)
for t in a6_trades:
    a6_pats[t['open_pattern']].append(t['pnl_eod'])
print(f"\n  A6 pattern breakdown:")
for pat, pnls in sorted(a6_pats.items(), key=lambda x: -len(x[1])):
    m = compute_metrics(pnls)
    print(f"    {pat:20s}: n={m['n']:3d}  avg={m['avg']:+.2f}%  Sharpe={m['sharpe']:.3f}")

# Combined portfolio: SHORT E2-A + LONG A6 (sustained_rise only)
print(f"\n  --- Combined Portfolio: SHORT E2-A + LONG A6(SR) ---")

# Build daily PnL series
daily_short = defaultdict(float)
daily_long = defaultdict(float)
for t in r3_trades:
    daily_short[t['scan_date']] += t['pnl_eod']
for t in a6_sr:
    daily_long[t['scan_date']] += t['pnl_eod']

all_port_dates = sorted(set(list(daily_short.keys()) + list(daily_long.keys())))
daily_combined = []
daily_short_only = []
daily_long_only = []

for dt in all_port_dates:
    s_pnl = daily_short.get(dt, 0.0)
    l_pnl = daily_long.get(dt, 0.0)
    daily_combined.append(s_pnl + l_pnl)
    daily_short_only.append(s_pnl)
    daily_long_only.append(l_pnl)

port_m = compute_metrics(daily_combined, 'Portfolio')
print(f"    n active days = {len(daily_combined)}")
print(f"    total PnL     = {sum(daily_combined):+.2f}%")
print(f"    avg daily PnL = {port_m['avg']:+.2f}%")
print(f"    Sharpe (daily) = {port_m['sharpe']:.3f}")
print(f"    WR (daily)     = {port_m['wr']*100:.0f}%")

# Max drawdown
cum = np.cumsum(daily_combined)
peak = np.maximum.accumulate(cum)
dd = cum - peak
max_dd = float(np.min(dd)) if len(dd) > 0 else 0
print(f"    Max drawdown  = {max_dd:+.2f}% (cumulative PnL terms)")

# Correlation to SPY daily returns
spy_rets_matched = []
port_rets_matched = []
for i, dt in enumerate(all_port_dates):
    if dt in spy_closes:
        prior_dates_list = [d for d in spy_dates_sorted if d < dt]
        if prior_dates_list:
            prev_dt = prior_dates_list[-1]
            spy_ret = (spy_closes[dt] - spy_closes[prev_dt]) / spy_closes[prev_dt]
            spy_rets_matched.append(spy_ret)
            port_rets_matched.append(daily_combined[i])

if len(spy_rets_matched) > 5:
    corr, p_val = sp_stats.pearsonr(spy_rets_matched, port_rets_matched)
    print(f"    Corr to SPY   = {corr:+.3f} (p={p_val:.3f})")

exp_results['N10'] = {
    'a6_all': a6_all_m,
    'a6_sustained_rise': a6_sr_m,
    'portfolio': port_m,
    'max_drawdown': round(max_dd, 3),
    'corr_to_spy': round(float(corr), 3) if len(spy_rets_matched) > 5 else None,
    'total_pnl': round(sum(daily_combined), 3),
}


# =============================================================================
# CHARTS
# =============================================================================

print(f"\n\n  Generating charts...")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle('Agent 1 Aug-Oct 2026 Investigation', fontsize=14, fontweight='bold')

# 1. R3: Monthly PnL bars
ax = axes[0, 0]
months = sorted(monthly.keys())
mo_avgs = [compute_metrics(monthly[mo])['avg'] for mo in months]
mo_ns = [compute_metrics(monthly[mo])['n'] for mo in months]
colors = ['#27ae60' if v > 0 else '#e74c3c' for v in mo_avgs]
bars = ax.bar(range(len(months)), mo_avgs, color=colors, alpha=0.8)
ax.set_xticks(range(len(months)))
ax.set_xticklabels(months, rotation=30)
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('R3: E2-A SHORT Monthly Avg PnL')
ax.set_ylabel('Avg PnL %')
ax.grid(True, alpha=0.3, axis='y')
for i, (v, n) in enumerate(zip(mo_avgs, mo_ns)):
    ax.text(i, v + 0.1 * np.sign(v), f"n={n}", ha='center', fontsize=8)

# 2. R4: Regime PnL
ax = axes[0, 1]
regime_labels = ['calm_bull', 'calm_bear', 'vol_bull', 'vol_bear']
regime_avgs = []
regime_ns = []
for r in regime_labels:
    dts_set = set(regime_days[r])
    pnls = [t['pnl_eod'] for t in r3_trades if t['scan_date'] in dts_set]
    m = compute_metrics(pnls)
    regime_avgs.append(m['avg'])
    regime_ns.append(m['n'])
r_colors = ['#27ae60' if v > 0 else '#e74c3c' for v in regime_avgs]
ax.bar(range(len(regime_labels)), regime_avgs, color=r_colors, alpha=0.8)
ax.set_xticks(range(len(regime_labels)))
ax.set_xticklabels(regime_labels, rotation=30)
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('R4: E2-A SHORT PnL by Regime')
ax.set_ylabel('Avg PnL %')
ax.grid(True, alpha=0.3, axis='y')
for i, (v, n) in enumerate(zip(regime_avgs, regime_ns)):
    ax.text(i, v + 0.1 * np.sign(v) if v != 0 else 0.1, f"n={n}", ha='center', fontsize=8)

# 3. N1: LONG vs SHORT comparison
ax = axes[0, 2]
comparison_labels = ['SHORT\n(E2-A)', 'LONG\n(N1 Aug-Oct)', 'LONG\n(N1 Full Bull)']
comparison_avgs = [r3_m['avg'], n1_m_aug['avg'], n1_m_full_bull['avg']]
comparison_ns = [r3_m['n'], n1_m_aug['n'], n1_m_full_bull['n']]
c_colors = ['#2c3e50' if v > 0 else '#e74c3c' for v in comparison_avgs]
ax.bar(range(len(comparison_labels)), comparison_avgs, color=c_colors, alpha=0.8)
ax.set_xticks(range(len(comparison_labels)))
ax.set_xticklabels(comparison_labels)
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('N1: SHORT vs LONG Gap-Up Momentum')
ax.set_ylabel('Avg PnL %')
ax.grid(True, alpha=0.3, axis='y')
for i, (v, n) in enumerate(zip(comparison_avgs, comparison_ns)):
    ax.text(i, v + 0.1 * np.sign(v) if v != 0 else 0.1, f"n={n}", ha='center', fontsize=8)

# 4. N9: Float bins
ax = axes[1, 0]
fb_labels = ['low_10_50M', 'mid_50_200M', 'high_200M+']
fb_avgs = [compute_metrics([t['pnl_eod'] for t in float_bins[fb]])['avg'] for fb in fb_labels]
fb_ns = [compute_metrics([t['pnl_eod'] for t in float_bins[fb]])['n'] for fb in fb_labels]
fb_colors = ['#27ae60' if v > 0 else '#e74c3c' for v in fb_avgs]
ax.bar(range(len(fb_labels)), fb_avgs, color=fb_colors, alpha=0.8)
ax.set_xticks(range(len(fb_labels)))
ax.set_xticklabels(['Low\n10-50M', 'Mid\n50-200M', 'High\n>200M'])
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('N9: SHORT PnL by Float Bin')
ax.set_ylabel('Avg PnL %')
ax.grid(True, alpha=0.3, axis='y')
for i, (v, n) in enumerate(zip(fb_avgs, fb_ns)):
    ax.text(i, v + 0.1 * np.sign(v) if v != 0 else 0.1, f"n={n}", ha='center', fontsize=8)

# 5. N10: Portfolio cumulative PnL
ax = axes[1, 1]
if daily_combined:
    cum_short = np.cumsum(daily_short_only)
    cum_long = np.cumsum(daily_long_only)
    cum_total = np.cumsum(daily_combined)
    ax.plot(range(len(cum_short)), cum_short, '-', color='#e74c3c', linewidth=1.5, label='SHORT E2-A', alpha=0.7)
    ax.plot(range(len(cum_long)), cum_long, '-', color='#27ae60', linewidth=1.5, label='LONG A6(SR)', alpha=0.7)
    ax.plot(range(len(cum_total)), cum_total, '-', color='#2c3e50', linewidth=2.5, label='Combined')
    ax.axhline(0, color='#333', lw=1, ls=':')
    ax.set_title('N10: Portfolio Cumulative PnL')
    ax.set_xlabel('Active Trading Day')
    ax.set_ylabel('Cumulative PnL %')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 6. N2: Top sectors SHORT vs LONG
ax = axes[1, 2]
top_sectors = [s for s in all_sectors[:8] if sector_short_pnls.get(s) or sector_long_pnls.get(s)]
if top_sectors:
    x_s = np.arange(len(top_sectors))
    w = 0.35
    s_avgs = [np.mean(sector_short_pnls.get(s, [0])) for s in top_sectors]
    l_avgs = [np.mean(sector_long_pnls.get(s, [0])) for s in top_sectors]
    ax.bar(x_s - w/2, s_avgs, w, color='#e74c3c', alpha=0.8, label='SHORT')
    ax.bar(x_s + w/2, l_avgs, w, color='#27ae60', alpha=0.8, label='LONG')
    ax.set_xticks(x_s)
    ax.set_xticklabels([s[:12] for s in top_sectors], rotation=45, ha='right', fontsize=7)
    ax.axhline(0, color='#333', lw=1, ls=':')
    ax.set_title('N2: Sector SHORT vs LONG Avg PnL')
    ax.set_ylabel('Avg PnL %')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.95])
chart_path = os.path.join(BASE_DIR, 'investigation_aug_oct.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {chart_path}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print(f"\n\n{'='*100}")
print(f"  COMPREHENSIVE SUMMARY")
print(f"{'='*100}")

summary_rows = [
    ['R3', 'E2-A SHORT Revalidation',
     f"n={r3_m['n']}", f"{r3_m['avg']:+.2f}%", f"{r3_m['sharpe']:.3f}", f"{r3_m['wr']*100:.0f}%",
     f"[{r3_m['ci_lo']:+.2f}, {r3_m['ci_hi']:+.2f}]"],
    ['R4', 'Regime Characterization',
     f"{len(regime_data)} days", f"vol_med={vol_median:.2%}", f"slope={np.mean(slopes):+.4f}",
     f"", f""],
    ['N1a', 'LONG Momentum Aug-Oct',
     f"n={n1_m_aug['n']}", f"{n1_m_aug['avg']:+.2f}%", f"{n1_m_aug['sharpe']:.3f}", f"{n1_m_aug['wr']*100:.0f}%",
     f"[{n1_m_aug['ci_lo']:+.2f}, {n1_m_aug['ci_hi']:+.2f}]"],
    ['N1b', 'LONG Full Bull (SPY5d>0)',
     f"n={n1_m_full_bull['n']}", f"{n1_m_full_bull['avg']:+.2f}%", f"{n1_m_full_bull['sharpe']:.3f}", f"{n1_m_full_bull['wr']*100:.0f}%",
     f"[{n1_m_full_bull['ci_lo']:+.2f}, {n1_m_full_bull['ci_hi']:+.2f}]"],
    ['N2', 'Sector Rotation',
     f"{len(all_sectors)} sectors", f"see detail", f"", f"", f""],
    ['N5', 'Sympathy Peers',
     f"n={n5_m['n']}", f"{n5_m['avg']:+.2f}%", f"{n5_m['sharpe']:.3f}", f"{n5_m['wr']*100:.0f}%",
     f"[{n5_m['ci_lo']:+.2f}, {n5_m['ci_hi']:+.2f}]"],
    ['N9a', 'Float LOW (10-50M)',
     f"n={compute_metrics([t['pnl_eod'] for t in float_bins['low_10_50M']])['n']}",
     f"{compute_metrics([t['pnl_eod'] for t in float_bins['low_10_50M']])['avg']:+.2f}%",
     f"{compute_metrics([t['pnl_eod'] for t in float_bins['low_10_50M']])['sharpe']:.3f}", f"", f""],
    ['N9b', 'Float MID (50-200M)',
     f"n={compute_metrics([t['pnl_eod'] for t in float_bins['mid_50_200M']])['n']}",
     f"{compute_metrics([t['pnl_eod'] for t in float_bins['mid_50_200M']])['avg']:+.2f}%",
     f"{compute_metrics([t['pnl_eod'] for t in float_bins['mid_50_200M']])['sharpe']:.3f}", f"", f""],
    ['N9c', 'Float HIGH (>200M)',
     f"n={compute_metrics([t['pnl_eod'] for t in float_bins['high_200M+']])['n']}",
     f"{compute_metrics([t['pnl_eod'] for t in float_bins['high_200M+']])['avg']:+.2f}%",
     f"{compute_metrics([t['pnl_eod'] for t in float_bins['high_200M+']])['sharpe']:.3f}", f"", f""],
    ['N10', 'Mirror Portfolio',
     f"n={port_m['n']} days", f"total={sum(daily_combined):+.2f}%", f"Sharpe={port_m['sharpe']:.3f}",
     f"DD={max_dd:+.2f}%",
     f"corr={corr:+.3f}" if len(spy_rets_matched) > 5 else ""],
]

print()
print(tabulate(summary_rows,
               headers=['Exp', 'Description', 'Size', 'PnL', 'Sharpe', 'WR/Other', 'CI/Detail'],
               tablefmt='simple'))


# =============================================================================
# SAVE RESULTS JSON
# =============================================================================

def _clean(obj):
    if obj is None: return None
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return round(float(obj), 4)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray): return [_clean(v) for v in obj.tolist()]
    return obj

json_out = {
    'config': {
        'backtest_start': BACKTEST_START,
        'backtest_end': BACKTEST_END,
        'full_start': FULL_START,
        'gap_min_pct': GAP_MIN_PCT,
        'gap_down_min': GAP_DOWN_MIN,
        'top_n': TOP_N,
        'exit_time': EXIT_TIME,
    },
    'R3': _clean(exp_results.get('R3')),
    'R4': _clean(exp_results.get('R4')),
    'N1': _clean(exp_results.get('N1')),
    'N2': _clean(exp_results.get('N2')),
    'N5': _clean(exp_results.get('N5')),
    'N9': _clean(exp_results.get('N9')),
    'N10': _clean(exp_results.get('N10')),
}

results_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)
jp = os.path.join(results_dir, 'investigation_aug_oct.json')
with open(jp, 'w') as f:
    json.dump(json_out, f, indent=2, default=lambda x: round(float(x), 4) if isinstance(x, (np.floating, np.integer)) else str(x))
print(f"\n  Saved: {jp}")

print(f"\n{'='*80}")
print(f"  TOTAL RUNTIME: {time.time() - t_total:.0f}s ({(time.time() - t_total)/60:.1f} min)")
print(f"{'='*80}")
