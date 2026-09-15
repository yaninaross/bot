# %% Agent 1 — Composite Backtest with Trade Execution Rules
# Exact execution logic: gap >= 5%, top 2 by gap, local high entry,
# fallback at bar 15 with 15% skip rule, T+2 hold, no SL.
# PM retrace filter: exclude pm_retrace > 0.80.
# Fixed capital model: $100K/day, equal split, non-compounding.
# Multiple universes compared side by side.
# Colab-cell friendly. ~25-35 min runtime.

import os, sys, time, re, asyncio, copy
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict

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
from matplotlib.patches import Patch

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — TRADE EXECUTION RULES
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_START = '2026-01-01'
BACKTEST_END   = '2026-09-12'

# Pre-market selection
GAP_MIN_PCT    = 5.0          # gap-up >= 5%
GAP_MAX_PCT    = 200.0        # cap at 200% — anything above is reverse split / data error
PRICE_MIN      = 20.0         # open price >= $20 (was $10; raised to avoid micro-cap squeeze risk)
FLOAT_MIN      = 10_000_000   # exclude float < 10M (cache miss = include)
PM_RETRACE_MAX = 0.80         # exclude pm_retrace_pct > 0.80 (None = include)
TOP_N          = 2            # take top 2 by gap
POOL_N         = 40           # wider pool for PM bar fetch before filtering

# Entry logic
PEAK_WINDOW    = 3            # bars before AND after for local high
MAX_PEAK_BAR   = 12           # scan bars 3..11 for peaks (confirmed by bar 14)
FALLBACK_BAR   = 15           # fallback entry at bar 15 close
SKIP_DROP_PCT  = 0.15         # skip fallback if price already down > 15%
FALLBACK_ONLY  = True         # skip local_high entries; only take 9:45 fallback (CS_ONLY data: fallback +2.08% vs local_high -0.25%)

# Hold and exit
HOLD_DAYS      = 2            # T+2 close, no SL, no intraday exit
PNL_CAP        = -50.0        # cap worst-case PnL per trade (% — clips catastrophic shorts like VELL -166%)

# Capital model
CAPITAL        = 100_000      # fixed capital each morning, equal split

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
_ADRC_TYPES = {'ADRC'}
_CS_TYPES = {'CS'}
_LEVERAGED_KEYWORDS = {'2X','3X','ULTRA','BULL','BEAR','INVERSE','SHORT','LONG',
                        'LEVERAGED','DIREXION','PROSHARES'}

_HOLIDAYS = {
    '2025-01-01','2025-01-20','2025-02-17','2025-04-18','2025-05-26',
    '2025-06-19','2025-07-04','2025-09-01','2025-11-27','2025-12-25',
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25',
    '2026-06-19','2026-07-03','2026-09-07','2026-11-26','2026-12-25',
    '2027-01-01','2027-01-18','2027-02-15','2027-03-26','2027-05-31',
    '2027-06-18','2027-07-05','2027-09-06','2027-11-25','2027-12-24',
}


# ═══════════════════════════════════════════════════════════════════════════════
# HARDCODED _FLOAT UNIVERSE (217 tickers — used for OG_FLOAT universe)
# ═══════════════════════════════════════════════════════════════════════════════

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
# UNIVERSE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
# Each universe applies additional filters ON TOP of the base rules.
# Base rules (always applied): gap>=5%, price>=$10, float>=10M (Polygon),
# pm_retrace<=0.80, top N by gap, local high entry + skip rule, T+2 hold.
#
# Per-universe overrides (defaults to global config if not set):
#   top_n          — override TOP_N (e.g. 5 for original strategy)
#   pm_retrace_max — override PM_RETRACE_MAX (None = disabled)
#   skip_drop_pct  — override SKIP_DROP_PCT (None = disabled)
#   no_float_filter— skip Polygon float check (for hardcoded universes)
#
# Filter keys:
#   ticker_list    — restrict to these tickers only (e.g. list(_FLOAT.keys()))
#   exclude_etf    — drop ETF/ETV/ETN/ETP types
#   exclude_adrc   — drop ADRCs (American Depositary Receipts)
#   require_cs     — require type == CS (Common Stock only)
#   exclude_lev    — drop leveraged products
#   pattern        — require this open_pattern (e.g. 'sustained_drop')
#   exclude_fresh  — drop very_fresh gap freshness
#   require_fresh  — require freshness in this list
#   price_min      — override minimum price
# ═══════════════════════════════════════════════════════════════════════════════

UNIVERSES = OrderedDict([
    ('OG_ORIGINAL', {
        'label': '217 _FLOAT tickers, N=5, no filters',
        'ticker_list': list(_FLOAT.keys()),
        'top_n': 5,
        'pm_retrace_max': None,
        'skip_drop_pct': None,
        'no_float_filter': True,
        'price_min': 10.0,
        'fallback_only': False,
        'pnl_cap': None,
    }),
    ('BASELINE', {
        'label': 'All US equities (base rules only)',
    }),
    ('CS_ONLY', {
        'label': 'Common Stock only (no ETFs, no ADRCs)',
        'exclude_etf': True,
        'exclude_adrc': True,
    }),
    ('OG_FLOAT', {
        'label': '217 tickers, N=2, new filters',
        'ticker_list': list(_FLOAT.keys()),
    }),
    # ── Add new universes below ──
    # ('NO_ETF', {
    #     'label': 'Baseline excluding ETFs',
    #     'exclude_etf': True,
    # }),
    # ('QUALITY', {
    #     'label': 'No ETF + sustained_drop + not very_fresh',
    #     'exclude_etf': True,
    #     'pattern': 'sustained_drop',
    #     'exclude_fresh': True,
    # }),
])


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


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC TICKER REFERENCE (float + type)
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
            wso = res.get('weighted_shares_outstanding')
            sso = res.get('share_class_shares_outstanding')
            return tk, {
                'float': wso or sso,
                'type': res.get('type'),
                'name': res.get('name', ''),
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
    return {tk: _ticker_ref.get(tk, {}) for tk in tickers}


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC MINUTE BAR FETCH — PM + RTH
# ═══════════════════════════════════════════════════════════════════════════════

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
            if not bars: return (tk, dt), None
            pm_bars, rth_bars = [], []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                rec = (t_str, {'o':float(b['o']),'h':float(b['h']),
                               'l':float(b['l']),'c':float(b['c']),'v':int(b['v'])})
                if '04:00' <= t_str < '09:30':
                    pm_bars.append(rec)
                elif '09:30' <= t_str < '16:00':
                    rth_bars.append(rec)
            return (tk, dt), {'pm': pm_bars, 'rth': rth_bars}
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


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY LOGIC — exact rules from spec
# ═══════════════════════════════════════════════════════════════════════════════

def find_entry(rth_bars, open_price):
    """
    Scan bars 3..11 for local high (window=3), fallback at bar 15.
    Always returns entry dict (skip filtering deferred to universe config).
    Returns None only if too few bars to enter at all.
    """
    n = len(rth_bars)
    if n < PEAK_WINDOW * 2 + 1:
        return None

    for i in range(PEAK_WINDOW, min(MAX_PEAK_BAR, n - PEAK_WINDOW)):
        ch = rth_bars[i][1]['h']
        if all(rth_bars[i-j][1]['h'] < ch and rth_bars[i+j][1]['h'] < ch
               for j in range(1, PEAK_WINDOW + 1)):
            ci = i + PEAK_WINDOW
            if ci < n:
                return {'entry_price': rth_bars[ci][1]['c'],
                        'entry_time': rth_bars[ci][0],
                        'entry_type': 'local_high',
                        'entry_bar': ci, 'peak_bar': i,
                        'fallback_drop_pct': 0}

    if FALLBACK_BAR is not None and FALLBACK_BAR < n:
        bar15_close = rth_bars[FALLBACK_BAR][1]['c']
        drop_pct = (open_price - bar15_close) / open_price if open_price > 0 else 0
        return {'entry_price': bar15_close,
                'entry_time': rth_bars[FALLBACK_BAR][0],
                'entry_type': 'fallback',
                'entry_bar': FALLBACK_BAR, 'peak_bar': None,
                'fallback_drop_pct': drop_pct}

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PM RETRACE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pm_retrace(pm_bars, open_price):
    """
    pm_retrace_pct = (pm_high - pm_last) / (pm_high - open_price)
    if (pm_high - open_price) > 0 else 0.0
    Returns None if no PM bars.
    """
    if not pm_bars:
        return None
    pm_high = max(b[1]['h'] for b in pm_bars)
    pm_last = pm_bars[-1][1]['c']
    denom = pm_high - open_price
    if denom <= 0:
        return 0.0
    return (pm_high - pm_last) / denom


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN + FRESHNESS (for universe filters)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_pattern(rth_bars):
    if len(rth_bars) < 60:
        return 'no_data'
    low_30m = min(b[1]['l'] for b in rth_bars[:30])
    close_60m = rth_bars[59][1]['c']
    open_p = rth_bars[0][1]['o']
    drop = (low_30m - open_p) / open_p if open_p > 0 else 0
    rec_d = abs(open_p - low_30m)
    recovery = (close_60m - low_30m) / rec_d if rec_d > 0 else 0
    if drop < -0.03 and recovery > 0.5:
        return 'drop_then_recover'
    elif drop < -0.03 and recovery <= 0.5:
        return 'sustained_drop'
    elif drop >= -0.01:
        return 'flat_or_bounce'
    return 'mild_drop'

def classify_freshness(pm_bars):
    if not pm_bars:
        return 'no_pm'
    pm_high = max(b[1]['h'] for b in pm_bars)
    for b in pm_bars:
        if b[1]['h'] >= pm_high * 0.98:
            h, m = int(b[0][:2]), int(b[0][3:5])
            mins = (9 * 60 + 30) - (h * 60 + m)
            if mins <= 30: return 'very_fresh'
            if mins <= 120: return 'fresh'
            if mins <= 300: return 'moderate'
            return 'stale'
    return 'unknown'


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pnl_list, label=''):
    if not pnl_list:
        return {'label':label,'n':0,'avg':0,'med':0,'std':0,'wr':0,
                'sharpe':0,'pf':0,'ci_lo':0,'ci_hi':0,'trimmed_mean':0}
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
            'trimmed_mean':tm}


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def passes_universe(c, spec):
    tl = spec.get('ticker_list')
    if tl and c['ticker'] not in tl:
        return False

    # float filter (per-universe: 'no_float_filter' disables)
    if not spec.get('no_float_filter'):
        flt = c.get('float_shares')
        if flt is not None and flt < FLOAT_MIN:
            return False

    # PM retrace (per-universe: override with 'pm_retrace_max', None = disabled)
    pm_max = spec.get('pm_retrace_max', PM_RETRACE_MAX)
    if pm_max is not None:
        pr = c.get('pm_retrace')
        if pr is not None and pr > pm_max:
            return False

    # fallback-only: skip local_high entries (per-universe: override with 'fallback_only')
    fo = spec.get('fallback_only', FALLBACK_ONLY)
    if fo and c.get('entry_type') == 'local_high':
        return False

    # skip rule for fallback entries (per-universe: override with 'skip_drop_pct', None = disabled)
    skip_pct = spec.get('skip_drop_pct', SKIP_DROP_PCT)
    if skip_pct is not None:
        if c.get('entry_type') == 'fallback' and c.get('fallback_drop_pct', 0) > skip_pct:
            return False

    if spec.get('exclude_etf') and c.get('is_etf'):
        return False
    if spec.get('exclude_adrc') and c.get('is_adrc'):
        return False
    if spec.get('require_cs') and not c.get('is_cs'):
        return False
    if spec.get('exclude_lev') and c.get('is_leveraged'):
        return False
    pat = spec.get('pattern')
    if pat and c.get('open_pattern') != pat:
        return False
    if spec.get('exclude_fresh') and c.get('gap_freshness') == 'very_fresh':
        return False
    req = spec.get('require_fresh')
    if req and c.get('gap_freshness') not in req:
        return False
    pm = spec.get('price_min', PRICE_MIN)
    if pm and c['open_price'] < pm:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  DATA COLLECTION (single pass)
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def collect_data():
    t_total = time.time()
    timings = {}

    print(f"\n{'='*80}")
    print(f"  AGENT 1 — COMPOSITE BACKTEST")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Rules: gap>={GAP_MIN_PCT}%, px>=${PRICE_MIN}, float>={FLOAT_MIN/1e6:.0f}M")
    print(f"  PM retrace<={PM_RETRACE_MAX}, top {TOP_N} by gap, T+{HOLD_DAYS} hold")
    print(f"  Fallback only: {FALLBACK_ONLY}  |  PnL cap: {PNL_CAP}%")
    print(f"  Capital: ${CAPITAL:,.0f}/day, equal split, non-compounding")
    print(f"  Universes: {list(UNIVERSES.keys())}")
    print(f"{'='*80}")

    # ── Trading days ──
    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"\n  {n_dates} trading days")

    # ── Phase 1: Scan all days ──
    t1 = time.time()
    print(f"\n  Phase 1: Scanning for gap-ups...")
    pool_by_date = {}
    all_tickers = set()
    total_pool = 0

    gap_min = GAP_MIN_PCT / 100.0
    gap_max = GAP_MAX_PCT / 100.0

    # use lowest price_min across all universes for scan (per-universe filters applied later)
    scan_price_min = min(spec.get('price_min', PRICE_MIN) for spec in UNIVERSES.values())

    # collect ticker_list tickers from all universes (for pool expansion)
    _uni_tickers = set()
    for spec in UNIVERSES.values():
        tl = spec.get('ticker_list')
        if tl:
            _uni_tickers.update(tl)

    for i, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = fetch_grouped_daily(prior_str)
        today = fetch_grouped_daily(dt)

        cands = []
        for tk in today:
            if tk in _EXCLUDE_TK: continue
            if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
            if tk not in prior: continue
            prev_c = prior[tk]['c']
            if prev_c <= 0: continue
            cur_p = today[tk]['o']
            if cur_p <= 0 or cur_p < scan_price_min: continue
            gap = (cur_p - prev_c) / prev_c
            if gap < gap_min: continue
            if gap > gap_max: continue
            cands.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_p, 'scan_date': dt,
                'volume_today': today[tk]['v'],
            })
            all_tickers.add(tk)

        cands.sort(key=lambda x: -x['gap_pct'])
        top_pool = cands[:POOL_N]

        # expand pool: include ALL candidates from ticker_list universes
        if _uni_tickers:
            top_tks = {c['ticker'] for c in top_pool}
            for c in cands[POOL_N:]:
                if c['ticker'] in _uni_tickers and c['ticker'] not in top_tks:
                    top_pool.append(c)
                    top_tks.add(c['ticker'])

        pool_by_date[dt] = top_pool
        total_pool += len(top_pool)

        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_pool} pool  |  {time.time()-t_total:.0f}s")

    timings['scan'] = time.time() - t1
    print(f"  → {total_pool} pool, {len(all_tickers)} unique tickers")

    # ── Phase 2: Fetch ticker reference (float + type) ──
    t2 = time.time()
    print(f"\n  Phase 2: Fetching float + type for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    for c_start in range(0, len(tk_list), 200):
        chunk = tk_list[c_start:c_start + 200]
        fetch_ticker_refs(chunk)
        done = min(c_start + 200, len(tk_list))
        if done % 1000 == 0 or done == len(tk_list):
            print(f"    {done}/{len(tk_list)}  |  {time.time()-t_total:.0f}s")
    timings['refs'] = time.time() - t2

    # ── Phase 3: Fetch PM+RTH minute bars ──
    t3 = time.time()
    all_pairs = set()
    for picks in pool_by_date.values():
        for p in picks:
            all_pairs.add((p['ticker'], p['scan_date']))

    print(f"\n  Phase 3: Fetching 1-min bars (PM+RTH) for {len(all_pairs)} pairs...")
    pairs_list = list(all_pairs)
    all_bars = {}
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start + 100]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        done = min(c_start + 100, len(pairs_list))
        if done % 500 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    timings['bars'] = time.time() - t3
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

    # ── Phase 4: Enrich, compute entry/retrace/pattern ──
    t4 = time.time()
    print(f"\n  Phase 4: Enriching candidates (entry, retrace, pattern, freshness)...")

    # pre-fetch outcome dates
    outcome_dates = set()
    for picks in pool_by_date.values():
        for p in picks:
            outcome_dates.add(p['scan_date'])
            nxt = _next_n_td(p['scan_date'], HOLD_DAYS)
            for nd in nxt:
                outcome_dates.add(nd.strftime('%Y-%m-%d'))
    to_fetch = sorted(od for od in outcome_dates if od not in _gd_cache)
    for od in to_fetch:
        try: fetch_grouped_daily(od)
        except: pass

    enriched_by_date = {}
    n_no_entry = 0
    n_no_bars = 0
    n_entered = 0

    for dt in dates:
        day_enriched = []
        for c in pool_by_date.get(dt, []):
            tk = c['ticker']
            key = (tk, dt)
            ref = _ticker_ref.get(tk, {})

            bar_data = all_bars.get(key)
            if bar_data is None:
                n_no_bars += 1
                continue

            rth_bars = bar_data.get('rth', [])
            pm_bars = bar_data.get('pm', [])

            entry = find_entry(rth_bars, c['open_price'])
            if entry is None:
                n_no_entry += 1
                continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = entry['entry_price']
            tc['entry_time'] = entry['entry_time']
            tc['entry_type'] = entry['entry_type']
            tc['entry_bar'] = entry['entry_bar']
            tc['fallback_drop_pct'] = entry.get('fallback_drop_pct', 0)

            # type + float (compute, don't filter — filtering is per-universe)
            tc['asset_type'] = ref.get('type', 'UNK')
            tc['asset_name'] = ref.get('name', '')
            tc['is_etf'] = tc['asset_type'] in _ETF_TYPES
            tc['is_adrc'] = tc['asset_type'] in _ADRC_TYPES
            tc['is_cs'] = tc['asset_type'] in _CS_TYPES
            tc['is_leveraged'] = any(kw in tc['asset_name'].upper() for kw in _LEVERAGED_KEYWORDS)
            tc['float_shares'] = ref.get('float')
            tc['pm_retrace'] = compute_pm_retrace(pm_bars, c['open_price'])
            tc['open_pattern'] = classify_pattern(rth_bars)
            tc['gap_freshness'] = classify_freshness(pm_bars)

            # T+2 outcome
            nxt = _next_n_td(dt, HOLD_DAYS)
            t2_date = nxt[-1].strftime('%Y-%m-%d') if len(nxt) >= HOLD_DAYS else None
            tc['t2_date'] = t2_date
            ep = tc['entry_price']
            if t2_date:
                dd = _gd_cache.get(t2_date, {}).get(tk, {})
                tc['pnl_t2'] = round((ep - dd['c']) / ep * 100, 3) if dd.get('c') else None
            else:
                tc['pnl_t2'] = None

            day_enriched.append(tc)
            n_entered += 1

        enriched_by_date[dt] = day_enriched

    timings['enrich'] = time.time() - t4
    elapsed = time.time() - t_total

    print(f"  → {n_entered} enriched  |  {n_no_entry} no entry (too few bars)  |  {n_no_bars} no bars")
    print(f"\n  {'─'*50}")
    for step, dur in [('Scan', timings['scan']), ('Refs', timings['refs']),
                      ('Bars', timings['bars']), ('Enrich', timings['enrich'])]:
        print(f"    {step:<20s}  {dur:>6.0f}s  ({dur/elapsed*100:>4.0f}%)")
    print(f"  TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return enriched_by_date, dates


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  BACKTEST ENGINE
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(enriched_by_date, dates):
    results = {}

    for uni_name, spec in UNIVERSES.items():
        trades = []
        daily_stats = []

        for dt in dates:
            pool = enriched_by_date.get(dt, [])
            passed = [c for c in pool if passes_universe(c, spec)]
            passed.sort(key=lambda x: -x['gap_pct'])
            uni_top_n = spec.get('top_n', TOP_N)
            selected = passed[:uni_top_n]

            valid_raw = [t for t in selected if t.get('pnl_t2') is not None]

            # deep copy to avoid mutating shared dicts across universes
            valid = [copy.deepcopy(t) for t in valid_raw]

            # apply per-trade PnL cap (clips catastrophic shorts)
            uni_pnl_cap = spec.get('pnl_cap', PNL_CAP)
            if uni_pnl_cap is not None:
                for t in valid:
                    if t['pnl_t2'] < uni_pnl_cap:
                        t['pnl_t2_raw'] = t['pnl_t2']
                        t['pnl_t2'] = uni_pnl_cap

            trades.extend(valid)

            if valid:
                n_trades = len(valid)
                pos_size = CAPITAL / n_trades
                day_pnl_dollar = sum(pos_size * (t['pnl_t2'] / 100) for t in valid)
                day_pnl_pct = day_pnl_dollar / CAPITAL * 100
            else:
                n_trades = 0
                day_pnl_dollar = 0.0
                day_pnl_pct = 0.0

            daily_stats.append({
                'date': dt, 'n_trades': n_trades,
                'day_pnl_dollar': day_pnl_dollar,
                'day_pnl_pct': day_pnl_pct,
            })

        # compute cumulative PnL
        cum_pnl = []
        running = 0.0
        for ds in daily_stats:
            running += ds['day_pnl_dollar']
            ds['cum_pnl'] = running
            cum_pnl.append(running)

        # trade-level metrics
        pnls = [t['pnl_t2'] for t in trades]
        m = compute_metrics(pnls, uni_name)

        # drawdown from peak cumulative PnL
        cum_arr = np.array(cum_pnl)
        peak = np.maximum.accumulate(cum_arr) if len(cum_arr) > 0 else np.array([0])
        dd = cum_arr - peak
        max_dd_dollar = float(dd.min()) if len(dd) > 0 else 0
        max_dd_pct = max_dd_dollar / CAPITAL * 100

        # monthly breakdown
        monthly = defaultdict(lambda: {'pnl': 0, 'n': 0, 'wins': 0})
        for t in trades:
            mo = t['scan_date'][:7]
            monthly[mo]['pnl'] += t['pnl_t2']
            monthly[mo]['n'] += 1
            if t['pnl_t2'] > 0:
                monthly[mo]['wins'] += 1

        # entry type breakdown
        n_local = sum(1 for t in trades if t['entry_type'] == 'local_high')
        n_fall = sum(1 for t in trades if t['entry_type'] == 'fallback')

        active_days = sum(1 for ds in daily_stats if ds['n_trades'] > 0)
        total_return = running
        total_return_pct = running / CAPITAL * 100

        results[uni_name] = {
            'label': spec.get('label', uni_name),
            'trades': trades, 'daily_stats': daily_stats,
            'metrics': m, 'monthly': dict(monthly),
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'max_dd_dollar': max_dd_dollar,
            'max_dd_pct': max_dd_pct,
            'active_days': active_days,
            'n_local_high': n_local, 'n_fallback': n_fall,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(results, dates):
    print(f"\n\n{'='*120}")
    print(f"  COMPOSITE BACKTEST RESULTS — {dates[0]} → {dates[-1]}")
    print(f"  Capital: ${CAPITAL:,.0f}/day  |  Top {TOP_N}  |  T+{HOLD_DAYS} hold  |  Non-compounding")
    print(f"{'='*120}")

    # ── Summary table ──
    print(f"\n  ── UNIVERSE COMPARISON ──")
    rows = []
    for name, r in results.items():
        m = r['metrics']
        ci_flag = '✓' if m['ci_lo'] > 0 else '✗'
        rows.append([
            name, r['label'][:35], m['n'],
            f"{r['active_days']}", f"{m['n']/r['active_days']:.1f}" if r['active_days'] > 0 else "0",
            f"{m['avg']:+.2f}%", f"{m['med']:+.2f}%", f"{m['trimmed_mean']:+.2f}%",
            f"{m['wr']*100:.0f}%", f"{m['sharpe']:.3f}", f"{m['pf']:.2f}",
            f"[{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}] {ci_flag}",
            f"${r['total_return']:+,.0f}", f"{r['total_return_pct']:+.1f}%",
            f"${r['max_dd_dollar']:,.0f}", f"{r['max_dd_pct']:.1f}%",
        ])
    print(tabulate(rows, headers=[
        'Universe','Description','Trades','Active','TPD',
        'Avg PnL','Median','TrimMean','WR','Sharpe','PF','CI 95%',
        'Total $','Return %','MaxDD $','MaxDD %'
    ], tablefmt='simple'))

    # ── Entry type breakdown ──
    print(f"\n  ── ENTRY TYPE ──")
    for name, r in results.items():
        n = r['metrics']['n']
        lh = r['n_local_high']
        fb = r['n_fallback']
        lh_pnls = [t['pnl_t2'] for t in r['trades'] if t['entry_type'] == 'local_high']
        fb_pnls = [t['pnl_t2'] for t in r['trades'] if t['entry_type'] == 'fallback']
        lh_avg = np.mean(lh_pnls) if lh_pnls else 0
        fb_avg = np.mean(fb_pnls) if fb_pnls else 0
        print(f"  {name:>15s}: local_high={lh} ({lh_avg:+.2f}%)  "
              f"fallback={fb} ({fb_avg:+.2f}%)  total={n}")

    # ── Monthly breakdown per universe ──
    for name, r in results.items():
        print(f"\n  ── MONTHLY P&L: {name} ──")
        rows = []
        for mo in sorted(r['monthly']):
            md = r['monthly'][mo]
            wr = md['wins'] / md['n'] * 100 if md['n'] > 0 else 0
            month_dollar = 0
            active_days_mo = 0
            for ds in r['daily_stats']:
                if ds['date'].startswith(mo):
                    month_dollar += ds['day_pnl_dollar']
                    if ds['n_trades'] > 0:
                        active_days_mo += 1
            avg_daily_pct = (month_dollar / CAPITAL / active_days_mo * 100
                             if active_days_mo > 0 else 0)
            rows.append([mo, md['n'], f"{active_days_mo}",
                         f"{avg_daily_pct:+.2f}%", f"{wr:.0f}%",
                         f"${month_dollar:+,.0f}"])
        print(tabulate(rows,
                        headers=['Month','Trades','Days','Avg Daily %','WR','Dollar P&L'],
                        tablefmt='simple'))

    # ── Worst/best trades ──
    for name, r in results.items():
        if not r['trades']: continue
        sorted_trades = sorted(r['trades'], key=lambda x: x['pnl_t2'])
        print(f"\n  ── WORST 5 TRADES: {name} ──")
        for t in sorted_trades[:5]:
            raw = t.get('pnl_t2_raw')
            cap_flag = f" (RAW={raw:+.2f}%)" if raw is not None and raw != t['pnl_t2'] else ""
            print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={t['gap_pct']*100:.1f}%  "
                  f"entry={t['entry_type']:>10s}  pnl={t['pnl_t2']:+.2f}%{cap_flag}")
        print(f"\n  ── BEST 5 TRADES: {name} ──")
        for t in sorted_trades[-5:]:
            print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={t['gap_pct']*100:.1f}%  "
                  f"entry={t['entry_type']:>10s}  pnl={t['pnl_t2']:+.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results, dates):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    uni_colors = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#e67e22',
                  '#9b59b6', '#1abc9c', '#f39c12']

    # ── Figure 1: Cumulative PnL ($) time series ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})

    ax = axes[0]
    for idx, (name, r) in enumerate(results.items()):
        ds = r['daily_stats']
        plot_dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in ds]
        cum_pnl = [d['cum_pnl'] for d in ds]
        color = uni_colors[idx % len(uni_colors)]
        ax.plot(plot_dates, cum_pnl, color=color, linewidth=2,
                label=f"{name} (${r['total_return']:+,.0f})", alpha=0.9)
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_ylabel('Cumulative P&L ($)')
    ax.set_title(f'Agent 1 SHORT — Cumulative P&L (${CAPITAL:,.0f}/day, top {TOP_N}, T+{HOLD_DAYS})\n'
                 f'{dates[0]} → {dates[-1]}', fontsize=13, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:+,.0f}'))
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Daily PnL bars ──
    ax = axes[1]
    for idx, (name, r) in enumerate(results.items()):
        ds = r['daily_stats']
        plot_dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in ds if d['n_trades'] > 0]
        day_pnl = [d['day_pnl_dollar'] for d in ds if d['n_trades'] > 0]
        color = uni_colors[idx % len(uni_colors)]
        ax.bar(plot_dates, day_pnl, width=1.5, alpha=0.4, color=color, label=name)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_ylabel('Daily P&L ($)')
    ax.set_xlabel('Date')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:+,.0f}'))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(BASE_DIR, 'agent1_composite_cumulative.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")

    # ── Figure 2: Rolling average PnL (%) + Win rate ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 1]})
    roll_window = 20

    ax = axes[0]
    for idx, (name, r) in enumerate(results.items()):
        if not r['trades']: continue
        trades_sorted = sorted(r['trades'], key=lambda t: t['scan_date'])
        pnls = [t['pnl_t2'] for t in trades_sorted]
        trade_dates = [datetime.strptime(t['scan_date'], '%Y-%m-%d') for t in trades_sorted]
        if len(pnls) >= roll_window:
            rolling = pd.Series(pnls).rolling(roll_window).mean()
            color = uni_colors[idx % len(uni_colors)]
            ax.plot(trade_dates, rolling, color=color, linewidth=2, label=f"{name} ({roll_window}-trade MA)")
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_ylabel('Avg PnL (%)')
    ax.set_title(f'Rolling {roll_window}-Trade Average PnL (%)', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # rolling win rate
    ax = axes[1]
    for idx, (name, r) in enumerate(results.items()):
        if not r['trades']: continue
        trades_sorted = sorted(r['trades'], key=lambda t: t['scan_date'])
        wins = [1 if t['pnl_t2'] > 0 else 0 for t in trades_sorted]
        trade_dates = [datetime.strptime(t['scan_date'], '%Y-%m-%d') for t in trades_sorted]
        if len(wins) >= roll_window:
            rolling = pd.Series(wins).rolling(roll_window).mean() * 100
            color = uni_colors[idx % len(uni_colors)]
            ax.plot(trade_dates, rolling, color=color, linewidth=2, label=f"{name}")
    ax.axhline(50, color='#999', lw=1, ls='--', label='50% WR')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(f'Rolling {roll_window}-Trade Win Rate', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    p2 = os.path.join(BASE_DIR, 'agent1_composite_rolling.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p2}")

    # ── Figure 3: Monthly P&L breakdown ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # monthly dollar PnL
    ax = axes[0]
    all_months = sorted(set(mo for r in results.values() for mo in r['monthly']))
    width = 0.8 / max(len(results), 1)
    for idx, (name, r) in enumerate(results.items()):
        month_pnls = []
        for mo in all_months:
            mpnl = 0
            for ds in r['daily_stats']:
                if ds['date'].startswith(mo):
                    mpnl += ds['day_pnl_dollar']
            month_pnls.append(mpnl)
        x = np.arange(len(all_months))
        color = uni_colors[idx % len(uni_colors)]
        ax.bar(x + idx * width, month_pnls, width, color=color, alpha=0.8, label=name)
    ax.set_xticks(np.arange(len(all_months)) + width * (len(results)-1) / 2)
    ax.set_xticklabels([m[2:] for m in all_months], fontsize=8, rotation=45)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_ylabel('Dollar P&L')
    ax.set_title('Monthly Dollar P&L', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:+,.0f}'))
    ax.legend(fontsize=8)

    # monthly avg daily pnl % (capital-weighted, consistent with dollar PnL)
    ax = axes[1]
    for idx, (name, r) in enumerate(results.items()):
        month_avgs = []
        for mo in all_months:
            mpnl = 0
            active_days_mo = 0
            for ds in r['daily_stats']:
                if ds['date'].startswith(mo):
                    mpnl += ds['day_pnl_dollar']
                    if ds['n_trades'] > 0:
                        active_days_mo += 1
            avg_daily = mpnl / CAPITAL / active_days_mo * 100 if active_days_mo > 0 else 0
            month_avgs.append(avg_daily)
        x = np.arange(len(all_months))
        color = uni_colors[idx % len(uni_colors)]
        ax.bar(x + idx * width, month_avgs, width, color=color, alpha=0.8, label=name)
    ax.set_xticks(np.arange(len(all_months)) + width * (len(results)-1) / 2)
    ax.set_xticklabels([m[2:] for m in all_months], fontsize=8, rotation=45)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_ylabel('Avg Daily PnL (%)')
    ax.set_title('Monthly Avg Daily PnL (%, capital-weighted)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)

    plt.tight_layout()
    p3 = os.path.join(BASE_DIR, 'agent1_composite_monthly.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p3}")

    # ── Figure 4: PnL distribution ──
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for idx, (name, r) in enumerate(results.items()):
        ax = axes[idx]
        pnls = [t['pnl_t2'] for t in r['trades']]
        if pnls:
            color = uni_colors[idx % len(uni_colors)]
            ax.hist(pnls, bins=50, color=color, alpha=0.7, edgecolor='white')
            ax.axvline(0, color='#333', lw=1)
            avg = np.mean(pnls)
            ax.axvline(avg, color='red', lw=2, ls='--', label=f'Mean: {avg:+.2f}%')
            ax.set_xlabel('PnL (%)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name}\n(n={len(pnls)}, WR={r["metrics"]["wr"]*100:.0f}%)', fontsize=11)
            ax.legend(fontsize=9)

    plt.tight_layout()
    p4 = os.path.join(BASE_DIR, 'agent1_composite_distribution.png')
    plt.savefig(p4, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p4}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

enriched_by_date, dates = collect_data()
results = run_backtest(enriched_by_date, dates)
print_results(results, dates)
plot_results(results, dates)
