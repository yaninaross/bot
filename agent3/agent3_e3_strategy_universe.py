# %% Agent 3 — E3: Strategy × Universe Grid
#
# Tests ALL Phase 1 strategies (A1-A7, B1-B3, C1-C2) across 7 universes:
#
#   _FLOAT   : 194 hardcoded tickers (original baseline)
#   M1_200   : Top 200 CS tickers by avg opening volatility (93-day lookback)
#   M1_500   : Top 500 CS tickers by avg opening volatility
#   M2_15    : CS tickers gapping ≥3% on ≥15% of days
#   M3_60    : CS tickers with mean-reversion rate ≥60% after gap-downs (min 3 events)
#   M5_200   : Top 200 by composite score (0.35×M1 + 0.25×M2 + 0.25×M3 + 0.15×ADV)
#   M5_500   : Top 500 by composite score
#
# Universe construction: 93 trading days before BACKTEST_START.
# Strategies match original Phase 1 investigation (long_investigation.py).
# IS/OOS split: 2026-01-01. Pension LONG only.
# Colab-cell friendly. ~45-75 min runtime (API-bound).

import os, sys, time, re, copy, asyncio
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

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_START = '2025-09-15'
BACKTEST_END   = '2026-09-12'
OOS_CUTOFF     = '2026-01-01'

GAP_DOWN_MIN    = 0.05      # |gap| >= 5% for hypothesis A
GAP_UP_MIN      = 0.05      # gap >= 5% for hypothesis B
PRICE_MIN       = 10.0      # A and B minimum price
PRICE_MIN_C     = 20.0      # C minimum price
FLOAT_MIN       = 10_000_000
TOP_N           = 5

PEAK_WINDOW     = 3
MAX_PEAK_BAR    = 12
FALLBACK_BAR    = 15
SKIP_BOUNCE_PCT = 0.15
ORB_BARS        = 5
ORB_MAX_BAR     = 25

ADV_LOOKBACK      = 20
PRIOR_DECLINE_5D  = 0.15
PRIOR_DECLINE_10D = 0.20
REVERSAL_GAP_MIN  = 0.03
VOLUME_SPIKE_MULT = 3.0

UNIVERSE_LOOKBACK = 93     # trading days for M1-M5 construction

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
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


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDED _FLOAT (current Agent 3 baseline universe)
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
_FLOAT_SET = {tk for tk, fs in _FLOAT.items() if fs >= FLOAT_MIN}

UNIVERSES = ['_FLOAT', 'M1_200', 'M1_500', 'M2_15', 'M3_60', 'M5_200', 'M5_500']
UNIVERSE_LABELS = {
    '_FLOAT': '_FLOAT (194 hardcoded)',
    'M1_200': 'M1: Top 200 opening vol',
    'M1_500': 'M1: Top 500 opening vol',
    'M2_15':  'M2: Gap freq ≥15%',
    'M3_60':  'M3: Reversion ≥60%',
    'M5_200': 'M5: Top 200 composite',
    'M5_500': 'M5: Top 500 composite',
}


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY DEFINITIONS
# Each: (id, hypothesis, filter_fn, pnl_key)
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGIES = [
    ('A1 T+2',          'A', None,                                                  'pnl_t2'),
    ('A1 T+3',          'A', None,                                                  'pnl_t3'),
    ('A1 T+5',          'A', None,                                                  'pnl_t5'),
    ('A3cs T+2',        'A', lambda t: t.get('is_cs'),                             'pnl_t2'),
    ('A5 D1 T+2',       'A', lambda t: -0.08 <= t.get('gap_pct', 0) < -0.05,      'pnl_t2'),
    ('A5 D2 T+2',       'A', lambda t: -0.12 <= t.get('gap_pct', 0) < -0.08,      'pnl_t2'),
    ('A5 D3 T+2',       'A', lambda t: -0.20 <= t.get('gap_pct', 0) < -0.12,      'pnl_t2'),
    ('A5 D4 T+2',       'A', lambda t: -0.35 <= t.get('gap_pct', 0) < -0.20,      'pnl_t2'),
    ('A6sr T+1',        'A', lambda t: t.get('pat') == 'sustained_rise',           'pnl_t1'),
    ('A6sr T+2',        'A', lambda t: t.get('pat') == 'sustained_rise',           'pnl_t2'),
    ('A6sr T+3',        'A', lambda t: t.get('pat') == 'sustained_rise',           'pnl_t3'),
    ('A6rd T+2',        'A', lambda t: t.get('pat') == 'rise_then_drop',           'pnl_t2'),
    ('A6mr T+2',        'A', lambda t: t.get('pat') == 'mild_rise',               'pnl_t2'),
    ('A6cd T+2',        'A', lambda t: t.get('pat') == 'continued_drop',           'pnl_t2'),
    ('A7 EOD',          'A', None,                                                  'pnl_eod'),
    ('A7 T+1',          'A', None,                                                  'pnl_t1'),
    # Hypothesis B
    ('B1 EOD',          'B', None,                                                  'pnl_eod'),
    ('B1 T+1',          'B', None,                                                  'pnl_t1'),
    ('B1 T+2',          'B', None,                                                  'pnl_t2'),
    ('B2vol EOD',       'B', lambda t: (t.get('bbv') or 0) > 1.5 * max(t.get('aov') or 1, 1), 'pnl_eod'),
    ('B3str EOD',       'B', lambda t: (t.get('break_pct') or 0) >= 0.0075,        'pnl_eod'),
    # Hypothesis C
    ('C1 T+3',          'C', lambda t: (t.get('hyp_c') or '').startswith('C1'),    'pnl_t3'),
    ('C1 T+5',          'C', lambda t: (t.get('hyp_c') or '').startswith('C1'),    'pnl_t5'),
    ('C2 T+3',          'C', lambda t: 'C2' in (t.get('hyp_c') or ''),            'pnl_t3'),
]


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

def _next_n_td(d, n=5):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    out, d = [], d + timedelta(days=1)
    for _ in range(30):
        if _is_td(d):
            out.append(d)
            if len(out) >= n: return out
        d += timedelta(days=1)
    return out

def _prior_n_td(d, n=20):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    out = []
    d = d - timedelta(days=1)
    for _ in range(60 + n):
        if _is_td(d):
            out.append(d)
            if len(out) >= n: return list(reversed(out))
        d -= timedelta(days=1)
    return list(reversed(out))

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
            time.sleep(5 * (2 ** attempt))
    r = requests.get(url, params=p, timeout=60)
    r.raise_for_status()
    return r.json()

def _tk_ok(tk):
    if not _TK_RE.match(tk): return False
    if tk in _EXCLUDE_TK: return False
    if len(tk) > 2 and any(tk.endswith(s) for s in _EXCLUDE_SUFFIX): return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# GROUPED DAILY CACHE
# ═══════════════════════════════════════════════════════════════════════════════

_gd_cache = {}

def fetch_grouped_daily(dt_str):
    if dt_str in _gd_cache: return _gd_cache[dt_str]
    data = _pg(f"{_BASE}/v2/aggs/grouped/locale/us/market/stocks/{dt_str}",
               {"adjusted":"true","include_otc":"false"})
    out = {}
    for r in data.get('results', []):
        tk = r.get('T', '')
        if not _TK_RE.match(tk): continue
        out[tk] = {'o':float(r.get('o',0)),'h':float(r.get('h',0)),
                   'l':float(r.get('l',0)),'c':float(r.get('c',0)),'v':int(r.get('v',0))}
    _gd_cache[dt_str] = out
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC REFERENCE FETCH (individual /v3/reference/tickers/{tk})
# ═══════════════════════════════════════════════════════════════════════════════

_ref_cache = {}

async def _fetch_ref_one(session, tk, sem):
    url = f"{_BASE}/v3/reference/tickers/{tk}"
    params = {"apiKey": POLYGON_API_KEY}
    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        return tk, None
                    data = await resp.json()
            except Exception:
                await asyncio.sleep(1)
                continue
            res = data.get('results', {})
            return tk, {
                'type':       res.get('type', ''),
                'market_cap': res.get('market_cap') or 0,
                'shares_out': res.get('share_class_shares_outstanding') or 0,
                'sic_code':   res.get('sic_code') or 0,
                'name':       res.get('name', ''),
            }
        return tk, None

async def _fetch_refs_batch(tickers):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_ref_one(session, tk, sem) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: val for tk, val in results if val is not None}

def fetch_ticker_references(tickers, t_total=0):
    global _ref_cache
    uncached = [tk for tk in tickers if tk not in _ref_cache]
    if not uncached: return
    for c_start in range(0, len(uncached), 200):
        chunk = uncached[c_start:c_start + 200]
        batch = asyncio.run(_fetch_refs_batch(chunk))
        _ref_cache.update(batch)
        done = min(c_start + 200, len(uncached))
        elapsed = f"  |  {time.time()-t_total:.0f}s" if t_total else ""
        if done % 1000 == 0 or done == len(uncached):
            print(f"      {done}/{len(uncached)} refs{elapsed}")


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC MINUTE BAR FETCH (PM + RTH)
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
# ENTRY DETECTION + PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_low(bars, window=PEAK_WINDOW, max_bar=MAX_PEAK_BAR, fallback_bar=FALLBACK_BAR):
    if len(bars) < window * 2 + 1: return None
    for i in range(window, min(max_bar, len(bars) - window)):
        cl = bars[i][1]['l']
        if all(bars[i-j][1]['l'] > cl and bars[i+j][1]['l'] > cl
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(bars):
                return {'entry_price': bars[ci][1]['c'], 'entry_time': bars[ci][0],
                        'entry_type': 'local_low', 'entry_bar': ci}
    if fallback_bar is not None and fallback_bar < len(bars):
        return {'entry_price': bars[fallback_bar][1]['c'], 'entry_time': bars[fallback_bar][0],
                'entry_type': 'fallback', 'entry_bar': fallback_bar}
    return None

def find_orb_break_up(rth_bars, orb_bars=ORB_BARS, max_bar=ORB_MAX_BAR):
    if len(rth_bars) < orb_bars + 1: return None
    orb_high = max(b[1]['h'] for b in rth_bars[:orb_bars])
    for i in range(orb_bars, min(max_bar, len(rth_bars))):
        if rth_bars[i][1]['c'] > orb_high:
            break_pct = (rth_bars[i][1]['c'] - orb_high) / orb_high if orb_high > 0 else 0
            avg_vol_orb = np.mean([b[1]['v'] for b in rth_bars[:orb_bars]])
            return {'entry_price': rth_bars[i][1]['c'], 'entry_time': rth_bars[i][0],
                    'entry_type': 'orb_break', 'entry_bar': i,
                    'break_pct': round(break_pct, 6),
                    'bbv': rth_bars[i][1]['v'], 'aov': avg_vol_orb}
    return None

def classify_gap_down_pattern(rth_bars):
    if len(rth_bars) < 60: return 'no_data'
    op = rth_bars[0][1]['o']
    if op <= 0: return 'no_data'
    high_30m = max(b[1]['h'] for b in rth_bars[:30])
    close_60m = rth_bars[59][1]['c']
    rise = (high_30m - op) / abs(op)
    recovery = (close_60m - op) / abs(op)
    if rise > 0.03 and recovery > 0.5 * rise: return 'sustained_rise'
    if rise > 0.03 and recovery <= 0.5 * rise: return 'rise_then_drop'
    if rise <= 0.01: return 'continued_drop'
    return 'mild_rise'


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pnl_list, label=''):
    if not pnl_list:
        return {'label':label,'n':0,'avg':0,'med':0,'std':0,'wr':0,
                'sharpe':0,'pf':0,'ci_lo':0,'ci_hi':0,'trimmed_mean':0}
    a = np.array(pnl_list)
    n = len(a)
    avg = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if n > 1 else 0
    wr = float((a > 0).mean())
    sharpe = avg / std if std > 0 else 0
    se = std / np.sqrt(n) if n > 0 else 0
    ci_lo, ci_hi = avg - 1.96 * se, avg + 1.96 * se
    w = a[a > 0]; l = a[a < 0]
    pf = float(w.sum() / abs(l.sum())) if len(l) > 0 and l.sum() != 0 else float('inf')
    p5, p95 = np.percentile(a, 5), np.percentile(a, 95)
    trimmed = a[(a >= p5) & (a <= p95)]
    tm = float(np.mean(trimmed)) if len(trimmed) > 0 else avg
    return {'label':label,'n':n,'avg':avg,'med':float(np.median(a)),'std':std,'wr':wr,
            'sharpe':sharpe,'pf':pf,'ci_lo':ci_lo,'ci_hi':ci_hi,'trimmed_mean':tm}

def classify_status(m):
    if m['n'] < 10: return 'FEW'
    if m['sharpe'] >= 0.40 and m['ci_lo'] > 0 and m['avg'] >= 1.0: return 'PENSION'
    if m['sharpe'] >= 0.30 and m['avg'] >= 0.5: return 'NEAR'
    if m['sharpe'] >= 0.10: return 'DEGR'
    return 'DEAD'


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0: UNIVERSE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_universes(t_total):
    print(f"\n{'='*80}")
    print(f"  PHASE 0: UNIVERSE CONSTRUCTION ({UNIVERSE_LOOKBACK}-day lookback)")
    print(f"{'='*80}")

    # Build lookback date list: UNIVERSE_LOOKBACK trading days ending before BACKTEST_START
    lb_dates = []
    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date() - timedelta(days=1)
    for _ in range(300):
        if _is_td(d):
            lb_dates.append(d.strftime('%Y-%m-%d'))
            if len(lb_dates) >= UNIVERSE_LOOKBACK: break
        d -= timedelta(days=1)
    lb_dates = list(reversed(lb_dates))
    print(f"  Lookback: {lb_dates[0]} → {lb_dates[-1]} ({len(lb_dates)} days)")

    # Fetch grouped daily for lookback
    t1 = time.time()
    print(f"  Fetching {len(lb_dates)} lookback days...")
    for i, dt in enumerate(lb_dates):
        fetch_grouped_daily(dt)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(lb_dates)}  |  {time.time()-t_total:.0f}s")
    print(f"  Lookback fetch: {time.time()-t1:.0f}s")

    # Collect all unique tickers from lookback
    all_tickers = set()
    for dt in lb_dates:
        all_tickers.update(tk for tk in _gd_cache[dt] if _tk_ok(tk))
    print(f"  {len(all_tickers)} unique tickers in lookback")

    # Fetch individual reference data for all tickers
    t2 = time.time()
    print(f"  Fetching reference data for {len(all_tickers)} tickers...")
    fetch_ticker_references(sorted(all_tickers), t_total=t_total)
    # Backfill _FLOAT tickers missing from ref
    for tk, fs in _FLOAT.items():
        if tk not in _ref_cache:
            _ref_cache[tk] = {'type': 'CS', 'market_cap': 0, 'shares_out': fs,
                              'sic_code': 0, 'name': ''}
        elif (_ref_cache[tk].get('shares_out') or 0) == 0 and fs > 0:
            _ref_cache[tk]['shares_out'] = fs
    print(f"  Reference fetch: {time.time()-t2:.0f}s")

    # Filter to CS tickers
    cs_tickers = {tk for tk in all_tickers if _ref_cache.get(tk, {}).get('type') == 'CS'}
    print(f"  {len(cs_tickers)} CS tickers")

    # ── Compute M1, M2, M3, ADV for each CS ticker ──
    t3 = time.time()
    print(f"  Computing M1/M2/M3/ADV scores...")

    m1_scores = {}  # avg |gap|
    m2_scores = {}  # fraction of days with |gap| >= 3%
    m3_scores = {}  # fraction of gap-downs where T+2 close > gap-day open
    m3_events = {}
    adv_scores = {}

    for tk in cs_tickers:
        gaps_abs = []
        gap_big = 0
        gap_down_bounces = []
        dollar_vols = []

        for i in range(1, len(lb_dates)):
            prev_bar = _gd_cache.get(lb_dates[i-1], {}).get(tk)
            cur_bar = _gd_cache.get(lb_dates[i], {}).get(tk)
            if not prev_bar or not cur_bar: continue
            if prev_bar['c'] <= 0 or cur_bar['o'] <= 0: continue

            gap = (cur_bar['o'] - prev_bar['c']) / prev_bar['c']
            gaps_abs.append(abs(gap))
            if abs(gap) >= 0.03:
                gap_big += 1

            if cur_bar['c'] > 0 and cur_bar['v'] > 0:
                dollar_vols.append(cur_bar['c'] * cur_bar['v'])

            # M3: gap-down <= -3%, check T+2 bounce
            if gap <= -0.03 and i + 2 < len(lb_dates):
                t2_bar = _gd_cache.get(lb_dates[i + 2], {}).get(tk)
                if t2_bar and t2_bar['c'] > 0:
                    gap_down_bounces.append(t2_bar['c'] > cur_bar['o'])

        if len(gaps_abs) >= 20:
            m1_scores[tk] = np.mean(gaps_abs)
            m2_scores[tk] = gap_big / len(gaps_abs)

        if len(gap_down_bounces) >= 3:
            m3_scores[tk] = sum(gap_down_bounces) / len(gap_down_bounces)
            m3_events[tk] = len(gap_down_bounces)

        if dollar_vols:
            adv_scores[tk] = np.mean(dollar_vols)

    print(f"    M1: {len(m1_scores)} tickers scored")
    print(f"    M2: {sum(1 for v in m2_scores.values() if v >= 0.15)} with gap_freq ≥15%")
    print(f"    M3: {len(m3_scores)} with ≥3 gap events, "
          f"{sum(1 for v in m3_scores.values() if v >= 0.60)} with rate ≥60%")

    # ── Build universe sets ──

    # M1: Top N by avg opening volatility
    m1_ranked = sorted(m1_scores, key=m1_scores.get, reverse=True)
    M1_200 = set(m1_ranked[:200])
    M1_500 = set(m1_ranked[:500])

    # M2: Gap frequency >= 15%
    M2_15 = {tk for tk, freq in m2_scores.items() if freq >= 0.15}

    # M3: Mean reversion rate >= 60%
    M3_60 = {tk for tk, rate in m3_scores.items() if rate >= 0.60}

    # M5: Composite score with filters
    def _normalize(scores):
        if not scores: return {}
        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-12: return {k: 0.5 for k in scores}
        return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

    n_m1 = _normalize(m1_scores)
    n_m2 = _normalize(m2_scores)
    n_m3 = _normalize(m3_scores)
    n_adv = _normalize(adv_scores)

    m5_composite = {}
    for tk in cs_tickers:
        ref = _ref_cache.get(tk, {})
        so = ref.get('shares_out') or 0
        if so < FLOAT_MIN: continue
        adv = adv_scores.get(tk, 0)
        if adv < 1_000_000: continue
        closes = [_gd_cache.get(dt, {}).get(tk, {}).get('c', 0) for dt in lb_dates[-20:]]
        closes = [c for c in closes if c > 0]
        if not closes or np.mean(closes) < 5.0: continue

        score = (0.35 * n_m1.get(tk, 0) +
                 0.25 * n_m2.get(tk, 0) +
                 0.25 * n_m3.get(tk, 0) +
                 0.15 * n_adv.get(tk, 0))
        m5_composite[tk] = score

    m5_ranked = sorted(m5_composite, key=m5_composite.get, reverse=True)
    M5_200 = set(m5_ranked[:200])
    M5_500 = set(m5_ranked[:500])

    univ_sets = OrderedDict([
        ('_FLOAT', _FLOAT_SET.copy()),
        ('M1_200', M1_200),
        ('M1_500', M1_500),
        ('M2_15',  M2_15),
        ('M3_60',  M3_60),
        ('M5_200', M5_200),
        ('M5_500', M5_500),
    ])

    # Print universe sizes and overlap
    print(f"\n  ── Universe Sizes ──")
    rows = []
    for uid in UNIVERSES:
        us = univ_sets[uid]
        ov = len(us & _FLOAT_SET)
        rows.append([uid, len(us), ov, f"{ov/len(_FLOAT_SET)*100:.0f}%"])
    print(tabulate(rows, headers=['Universe', 'Size', '∩ _FLOAT', '%_FLOAT'], tablefmt='simple'))

    union = set()
    for s in univ_sets.values(): union.update(s)
    print(f"\n  Union: {len(union)} tickers")

    # Top 10 for each dimension
    print(f"\n  ── Top 10 M1 (avg |gap|) ──")
    for tk in m1_ranked[:10]:
        in_float = '✓' if tk in _FLOAT_SET else ' '
        print(f"    {tk:>6} [{in_float}]: {m1_scores[tk]*100:.2f}%")

    print(f"\n  ── Top 10 M2 (gap freq) ──")
    m2_ranked = sorted(m2_scores, key=m2_scores.get, reverse=True)
    for tk in m2_ranked[:10]:
        in_float = '✓' if tk in _FLOAT_SET else ' '
        print(f"    {tk:>6} [{in_float}]: {m2_scores[tk]*100:.1f}%")

    print(f"\n  ── Top 10 M3 (reversion rate, ≥3 events) ──")
    m3_ranked = sorted(m3_scores, key=m3_scores.get, reverse=True)
    for tk in m3_ranked[:10]:
        in_float = '✓' if tk in _FLOAT_SET else ' '
        print(f"    {tk:>6} [{in_float}]: {m3_scores[tk]*100:.0f}% ({m3_events[tk]} events)")

    print(f"\n  ── Top 10 M5 (composite) ──")
    for tk in m5_ranked[:10]:
        in_float = '✓' if tk in _FLOAT_SET else ' '
        print(f"    {tk:>6} [{in_float}]: {m5_composite[tk]:.4f}")

    print(f"\n  Universe construction: {time.time()-t3:.0f}s")

    return univ_sets, union, lb_dates


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: SCANNING + ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def collect_data(univ_sets, union_tickers, t_total):
    print(f"\n{'='*80}")
    print(f"  PHASE 1: SCANNING — {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Union: {len(union_tickers)} tickers across {len(UNIVERSES)} universes")
    print(f"{'='*80}")

    # Trading dates
    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"  {n_dates} trading days")

    # Fetch backtest grouped daily
    t1 = time.time()
    print(f"\n  Fetching {n_dates} backtest days...")
    for i, dt in enumerate(dates):
        fetch_grouped_daily(dt)
        if (i + 1) % 50 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {time.time()-t_total:.0f}s")

    # Fetch outcome dates (T+1 through T+10)
    outcome_dates = set()
    for dt in dates:
        for nd in _next_n_td(dt, 10):
            outcome_dates.add(nd.strftime('%Y-%m-%d'))
    to_fetch = sorted(od for od in outcome_dates if od not in _gd_cache)
    for od in to_fetch:
        try: fetch_grouped_daily(od)
        except: pass
    print(f"  + {len(to_fetch)} outcome dates  |  {time.time()-t_total:.0f}s")

    # ── Scan all dates for A/B/C candidates ──
    t2 = time.time()
    print(f"\n  Scanning for candidates...")

    # Per-universe per-day selections (top N tickers for A/B)
    a_sel = {u: {} for u in UNIVERSES}  # a_sel[univ][dt] = [tk1, tk2, ...]
    b_sel = {u: {} for u in UNIVERSES}

    # All candidate dicts keyed by (tk, dt)
    a_cands = {}  # (tk, dt) -> candidate dict
    b_cands = {}
    c_cands = {}  # no TOP_N for C

    bar_requests = set()  # (tk, dt) pairs needing minute bars
    n_a = n_b = n_c = 0

    for idx, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = _gd_cache.get(prior_str, {})
        today = _gd_cache.get(dt, {})

        day_a_all = []  # (gap_pct, tk, cand_dict)
        day_b_all = []

        for tk in union_tickers:
            if tk not in prior or tk not in today: continue
            prev_c = prior[tk]['c']
            if prev_c <= 0: continue
            cur_o = today[tk]['o']
            if cur_o <= 0: continue

            gap = (cur_o - prev_c) / prev_c
            ref = _ref_cache.get(tk, {})
            is_cs = ref.get('type') == 'CS'

            # Hypothesis A: gap-DOWN >= 5%, price >= $10
            if gap <= -GAP_DOWN_MIN and cur_o >= PRICE_MIN:
                cand = {
                    'ticker': tk, 'gap_pct': gap, 'open_price': cur_o,
                    'prior_close': prev_c, 'scan_date': dt,
                    'volume_today': today[tk]['v'], 'is_cs': is_cs,
                }
                # Pre-compute future dates
                nxt = _next_n_td(dt, 10)
                for k in range(min(10, len(nxt))):
                    cand[f't{k+1}_date'] = nxt[k].strftime('%Y-%m-%d')
                day_a_all.append((gap, tk, cand))

            # Hypothesis B: gap-UP >= 5%, price >= $10
            if gap >= GAP_UP_MIN and cur_o >= PRICE_MIN:
                cand = {
                    'ticker': tk, 'gap_pct': gap, 'open_price': cur_o,
                    'prior_close': prev_c, 'scan_date': dt,
                    'volume_today': today[tk]['v'], 'is_cs': is_cs,
                }
                nxt = _next_n_td(dt, 5)
                for k in range(min(5, len(nxt))):
                    cand[f't{k+1}_date'] = nxt[k].strftime('%Y-%m-%d')
                day_b_all.append((-gap, tk, cand))  # sort descending by gap

            # Hypothesis C: gap-UP >= 3%, price >= $20, prior decline
            if gap >= REVERSAL_GAP_MIN and cur_o >= PRICE_MIN_C:
                prior_5d = _prior_n_td(dt, 5)
                c_added = False
                if len(prior_5d) >= 5:
                    c5_str = prior_5d[0].strftime('%Y-%m-%d')
                    c5_data = _gd_cache.get(c5_str, {}).get(tk, {})
                    if c5_data.get('c') and c5_data['c'] > 0:
                        ret_5d = (prev_c - c5_data['c']) / c5_data['c']
                        if ret_5d <= -PRIOR_DECLINE_5D:
                            cand = {
                                'ticker': tk, 'gap_pct': gap, 'open_price': cur_o,
                                'prior_close': prev_c, 'scan_date': dt,
                                'volume_today': today[tk]['v'], 'is_cs': is_cs,
                                'hyp_c': 'C1', 'prior_5d_return': ret_5d,
                            }
                            nxt = _next_n_td(dt, 10)
                            for k in range(min(10, len(nxt))):
                                cand[f't{k+1}_date'] = nxt[k].strftime('%Y-%m-%d')
                            c_cands[(tk, dt)] = cand
                            c_added = True
                            n_c += 1

                # C2: 10-day decline + volume spike
                prior_10d = _prior_n_td(dt, 10)
                if len(prior_10d) >= 10:
                    c10_str = prior_10d[0].strftime('%Y-%m-%d')
                    c10_data = _gd_cache.get(c10_str, {}).get(tk, {})
                    if c10_data.get('c') and c10_data['c'] > 0:
                        ret_10d = (prev_c - c10_data['c']) / c10_data['c']
                        adv_dates_c = _prior_n_td(dt, ADV_LOOKBACK)
                        vols = [_gd_cache.get(dd.strftime('%Y-%m-%d'), {}).get(tk, {}).get('v', 0)
                                for dd in adv_dates_c]
                        vols = [v for v in vols if v > 0]
                        avg_vol = np.mean(vols) if vols else 0
                        vol_ratio = today[tk]['v'] / avg_vol if avg_vol > 0 else 0
                        if ret_10d <= -PRIOR_DECLINE_10D and vol_ratio >= VOLUME_SPIKE_MULT:
                            if (tk, dt) in c_cands:
                                c_cands[(tk, dt)]['hyp_c'] = 'C1+C2'
                                c_cands[(tk, dt)]['vol_ratio'] = vol_ratio
                            else:
                                cand = {
                                    'ticker': tk, 'gap_pct': gap, 'open_price': cur_o,
                                    'prior_close': prev_c, 'scan_date': dt,
                                    'volume_today': today[tk]['v'], 'is_cs': is_cs,
                                    'hyp_c': 'C2', 'prior_10d_return': ret_10d,
                                    'vol_ratio': vol_ratio,
                                }
                                nxt = _next_n_td(dt, 10)
                                for k in range(min(10, len(nxt))):
                                    cand[f't{k+1}_date'] = nxt[k].strftime('%Y-%m-%d')
                                c_cands[(tk, dt)] = cand
                                n_c += 1

        # Per-universe TOP_N for A (sort: most negative gap first)
        day_a_all.sort()
        for uid, uset in univ_sets.items():
            u_cands = [(g, tk, c) for g, tk, c in day_a_all if tk in uset][:TOP_N]
            if u_cands:
                a_sel[uid][dt] = [tk for _, tk, _ in u_cands]
                for _, tk, c in u_cands:
                    key = (tk, dt)
                    if key not in a_cands:
                        a_cands[key] = c
                    bar_requests.add(key)
                    n_a += 1

        # Per-universe TOP_N for B (sort: most positive gap first)
        day_b_all.sort()
        for uid, uset in univ_sets.items():
            u_cands = [(g, tk, c) for g, tk, c in day_b_all if tk in uset][:TOP_N]
            if u_cands:
                b_sel[uid][dt] = [tk for _, tk, _ in u_cands]
                for _, tk, c in u_cands:
                    key = (tk, dt)
                    if key not in b_cands:
                        b_cands[key] = c
                    bar_requests.add(key)
                    n_b += 1

        if (idx + 1) % 20 == 0 or idx == n_dates - 1:
            print(f"    [{idx+1:>3}/{n_dates}] {dt}  |  A:{len(a_cands)} B:{len(b_cands)} C:{len(c_cands)}  |  {time.time()-t_total:.0f}s")

    print(f"\n  Candidates: A={len(a_cands)}, B={len(b_cands)}, C={len(c_cands)}")
    print(f"  Minute bar requests: {len(bar_requests)} (A+B only, C uses grouped daily)")

    # ── Fetch minute bars for A + B candidates ──
    t3 = time.time()
    print(f"\n  Fetching 1-min bars for {len(bar_requests)} pairs...")
    pairs_list = list(bar_requests)
    all_bars = {}
    for c_start in range(0, len(pairs_list), 150):
        chunk = pairs_list[c_start:c_start + 150]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        done = min(c_start + 150, len(pairs_list))
        if done % 300 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    print(f"  Got bars for {len(all_bars)}/{len(bar_requests)} pairs  |  {time.time()-t3:.0f}s")

    # ── Enrich A candidates ──
    t4 = time.time()
    print(f"\n  Enriching candidates...")
    enriched_a = {}  # (tk, dt) -> enriched dict or None

    for key, cand in a_cands.items():
        tk, dt = key
        bar_data = all_bars.get(key)
        if not bar_data: continue
        rth = bar_data.get('rth', [])
        if not rth: continue

        entry = find_local_low(rth)
        if entry is None: continue

        # Skip bounce check for fallback entry
        if entry['entry_type'] == 'fallback':
            op = cand['open_price']
            if op > 0 and entry['entry_price'] > op * (1 + SKIP_BOUNCE_PCT):
                continue

        tc = copy.deepcopy(cand)
        tc['entry_price'] = entry['entry_price']
        tc['entry_type'] = entry['entry_type']
        tc['pat'] = classify_gap_down_pattern(rth)
        ep = tc['entry_price']

        # Intraday PnL
        for bar_idx, col in [(30, 'pnl_30m'), (60, 'pnl_60m')]:
            tc[col] = round((rth[bar_idx][1]['c'] - ep) / ep * 100, 3) if bar_idx < len(rth) else None

        # EOD + T+1 through T+10
        for offset in range(11):
            col = 'pnl_eod' if offset == 0 else f'pnl_t{offset}'
            date_key = dt if offset == 0 else tc.get(f't{offset}_date', '')
            dd = _gd_cache.get(date_key, {}).get(tk, {})
            tc[col] = round((dd['c'] - ep) / ep * 100, 3) if dd.get('c') else None

        enriched_a[key] = tc

    # ── Enrich B candidates ──
    enriched_b = {}
    for key, cand in b_cands.items():
        tk, dt = key
        bar_data = all_bars.get(key)
        if not bar_data: continue
        rth = bar_data.get('rth', [])
        if not rth: continue

        entry = find_orb_break_up(rth)
        if entry is None: continue

        tc = copy.deepcopy(cand)
        tc['entry_price'] = entry['entry_price']
        tc['entry_type'] = entry['entry_type']
        tc['break_pct'] = entry.get('break_pct', 0)
        tc['bbv'] = entry.get('bbv', 0)
        tc['aov'] = entry.get('aov', 0)
        ep = tc['entry_price']

        for bar_idx, col in [(30, 'pnl_30m'), (60, 'pnl_60m')]:
            tc[col] = round((rth[bar_idx][1]['c'] - ep) / ep * 100, 3) if bar_idx < len(rth) else None
        for offset in range(6):
            col = 'pnl_eod' if offset == 0 else f'pnl_t{offset}'
            date_key = dt if offset == 0 else tc.get(f't{offset}_date', '')
            dd = _gd_cache.get(date_key, {}).get(tk, {})
            tc[col] = round((dd['c'] - ep) / ep * 100, 3) if dd.get('c') else None

        enriched_b[key] = tc

    # ── Enrich C candidates (no minute bars — use grouped daily open as entry) ──
    enriched_c = {}
    for key, cand in c_cands.items():
        tk, dt = key
        today_bar = _gd_cache.get(dt, {}).get(tk)
        if not today_bar or today_bar['o'] <= 0: continue

        tc = copy.deepcopy(cand)
        ep = today_bar['o']
        tc['entry_price'] = ep
        tc['entry_type'] = 'market_open'

        for offset in range(11):
            col = 'pnl_eod' if offset == 0 else f'pnl_t{offset}'
            date_key = dt if offset == 0 else tc.get(f't{offset}_date', '')
            dd = _gd_cache.get(date_key, {}).get(tk, {})
            tc[col] = round((dd['c'] - ep) / ep * 100, 3) if dd.get('c') else None

        enriched_c[key] = tc

    print(f"  Enriched: A={len(enriched_a)}, B={len(enriched_b)}, C={len(enriched_c)}")
    print(f"  Enrichment: {time.time()-t4:.0f}s")

    return enriched_a, enriched_b, enriched_c, a_sel, b_sel, dates


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: STRATEGY × UNIVERSE GRID
# ═══════════════════════════════════════════════════════════════════════════════

def compute_grid(enriched_a, enriched_b, enriched_c, a_sel, b_sel, univ_sets, dates):
    print(f"\n{'='*80}")
    print(f"  PHASE 2: COMPUTING {len(STRATEGIES)} STRATEGIES × {len(UNIVERSES)} UNIVERSES")
    print(f"{'='*80}")

    # Build per-universe trade lists
    # For A/B: use pre-selected top-N per day
    # For C: filter to tickers in universe (no TOP_N)

    grid = OrderedDict()  # grid[strat_id][univ_id] = metrics dict

    for strat_id, hyp, filt_fn, pnl_key in STRATEGIES:
        grid[strat_id] = OrderedDict()

        for uid in UNIVERSES:
            uset = univ_sets[uid]
            trades = []

            if hyp == 'A':
                for dt, tickers in a_sel[uid].items():
                    for tk in tickers:
                        t = enriched_a.get((tk, dt))
                        if t is None: continue
                        if filt_fn and not filt_fn(t): continue
                        if t.get(pnl_key) is not None:
                            trades.append(t)

            elif hyp == 'B':
                for dt, tickers in b_sel[uid].items():
                    for tk in tickers:
                        t = enriched_b.get((tk, dt))
                        if t is None: continue
                        if filt_fn and not filt_fn(t): continue
                        if t.get(pnl_key) is not None:
                            trades.append(t)

            elif hyp == 'C':
                for key, t in enriched_c.items():
                    tk, dt = key
                    if tk not in uset: continue
                    if filt_fn and not filt_fn(t): continue
                    if t.get(pnl_key) is not None:
                        trades.append(t)

            pnl = [t[pnl_key] for t in trades]
            m = compute_metrics(pnl, strat_id)
            m['n_dates'] = len(dates)
            grid[strat_id][uid] = m

    return grid


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_grid(grid, dates):
    n_dates = len(dates)

    # ── SHARPE GRID ──
    print(f"\n{'='*100}")
    print(f"  SHARPE GRID — Strategy × Universe")
    print(f"{'='*100}")

    header = ['Strategy'] + [uid for uid in UNIVERSES]
    rows = []
    for strat_id in grid:
        row = [strat_id]
        for uid in UNIVERSES:
            m = grid[strat_id][uid]
            if m['n'] == 0:
                row.append('—')
            else:
                ci_flag = '✓' if m['ci_lo'] > 0 else ' '
                status = classify_status(m)
                s = f"{m['sharpe']:+.3f}"
                if status == 'PENSION': s += ' ★'
                elif status == 'NEAR': s += ' ~'
                row.append(f"{s} {ci_flag}")
        rows.append(row)
    print('\n' + tabulate(rows, headers=header, tablefmt='simple'))

    print(f"\n  Legend: ★ = PENSION (Sharpe≥0.40, CI>0, avg≥1.0), ~ = NEAR MISS, ✓ = CI>0")

    # ── TRADE COUNT GRID ──
    print(f"\n{'='*100}")
    print(f"  TRADE COUNT (n) — Strategy × Universe")
    print(f"{'='*100}")
    rows = []
    for strat_id in grid:
        row = [strat_id]
        for uid in UNIVERSES:
            m = grid[strat_id][uid]
            row.append(str(m['n']) if m['n'] > 0 else '—')
        rows.append(row)
    print('\n' + tabulate(rows, headers=header, tablefmt='simple'))

    # ── AVG PNL GRID ──
    print(f"\n{'='*100}")
    print(f"  AVG PNL (%) — Strategy × Universe")
    print(f"{'='*100}")
    rows = []
    for strat_id in grid:
        row = [strat_id]
        for uid in UNIVERSES:
            m = grid[strat_id][uid]
            if m['n'] == 0:
                row.append('—')
            else:
                row.append(f"{m['avg']:+.2f}")
        rows.append(row)
    print('\n' + tabulate(rows, headers=header, tablefmt='simple'))

    # ── DETAILED TABLE FOR TOP PERFORMERS ──
    print(f"\n{'='*100}")
    print(f"  TOP PERFORMERS — CI > 0 AND Sharpe ≥ 0.30")
    print(f"{'='*100}")
    top_rows = []
    for strat_id in grid:
        for uid in UNIVERSES:
            m = grid[strat_id][uid]
            if m['n'] >= 10 and m['ci_lo'] > 0 and m['sharpe'] >= 0.30:
                tpd = m['n'] / n_dates if n_dates > 0 else 0
                top_rows.append([
                    strat_id, uid, m['n'], f"{tpd:.2f}",
                    f"{m['avg']:+.2f}%", f"{m['wr']*100:.0f}%",
                    f"{m['sharpe']:.3f}", f"{m['pf']:.2f}",
                    f"[{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}]",
                    classify_status(m),
                ])
    top_rows.sort(key=lambda r: float(r[6]), reverse=True)
    if top_rows:
        print('\n' + tabulate(top_rows,
                              headers=['Strategy','Universe','n','TPD','Avg','WR','Sharpe','PF','CI 95%','Status'],
                              tablefmt='simple'))
    else:
        print("  No strategy×universe pairs meet criteria.")

    # ── BEST UNIVERSE PER STRATEGY ──
    print(f"\n{'='*100}")
    print(f"  BEST UNIVERSE PER STRATEGY (by Sharpe)")
    print(f"{'='*100}")
    best_rows = []
    for strat_id in grid:
        best_uid = max(UNIVERSES, key=lambda u: grid[strat_id][u]['sharpe'] if grid[strat_id][u]['n'] >= 5 else -999)
        m = grid[strat_id][best_uid]
        m_float = grid[strat_id]['_FLOAT']
        if m['n'] < 5: continue
        delta = m['sharpe'] - m_float['sharpe'] if m_float['n'] >= 5 else 0
        best_rows.append([
            strat_id, best_uid,
            f"{m['sharpe']:.3f}", f"{m['avg']:+.2f}%", m['n'],
            f"{m_float['sharpe']:.3f}" if m_float['n'] >= 5 else '—',
            f"{delta:+.3f}" if m_float['n'] >= 5 else '—',
        ])
    print('\n' + tabulate(best_rows,
                          headers=['Strategy', 'Best Univ', 'Sharpe', 'Avg', 'n', '_FLOAT Sharpe', 'Δ Sharpe'],
                          tablefmt='simple'))


def run_oos_validation(grid, enriched_a, enriched_b, enriched_c, a_sel, b_sel, univ_sets, dates):
    print(f"\n{'='*100}")
    print(f"  OOS VALIDATION — Cutoff: {OOS_CUTOFF}")
    print(f"{'='*100}")

    # Find top performers
    candidates = []
    for strat_id in grid:
        for uid in UNIVERSES:
            m = grid[strat_id][uid]
            if m['n'] >= 10 and m['ci_lo'] > 0 and m['sharpe'] >= 0.30:
                candidates.append((strat_id, uid, m['sharpe']))
    candidates.sort(key=lambda x: x[2], reverse=True)

    if not candidates:
        print("  No strategy×universe pairs qualify for OOS validation.")
        return

    # Validate top 15
    oos_rows = []
    for strat_id, uid, _ in candidates[:15]:
        hyp = None
        filt_fn = None
        pnl_key = None
        for sid, h, f, pk in STRATEGIES:
            if sid == strat_id:
                hyp, filt_fn, pnl_key = h, f, pk
                break
        if hyp is None: continue

        uset = univ_sets[uid]
        is_trades, oos_trades = [], []

        if hyp == 'A':
            for dt, tickers in a_sel[uid].items():
                for tk in tickers:
                    t = enriched_a.get((tk, dt))
                    if t is None: continue
                    if filt_fn and not filt_fn(t): continue
                    if t.get(pnl_key) is None: continue
                    if dt < OOS_CUTOFF:
                        is_trades.append(t[pnl_key])
                    else:
                        oos_trades.append(t[pnl_key])
        elif hyp == 'B':
            for dt, tickers in b_sel[uid].items():
                for tk in tickers:
                    t = enriched_b.get((tk, dt))
                    if t is None: continue
                    if filt_fn and not filt_fn(t): continue
                    if t.get(pnl_key) is None: continue
                    if dt < OOS_CUTOFF:
                        is_trades.append(t[pnl_key])
                    else:
                        oos_trades.append(t[pnl_key])
        elif hyp == 'C':
            for key, t in enriched_c.items():
                tk, dt = key
                if tk not in uset: continue
                if filt_fn and not filt_fn(t): continue
                if t.get(pnl_key) is None: continue
                if dt < OOS_CUTOFF:
                    is_trades.append(t[pnl_key])
                else:
                    oos_trades.append(t[pnl_key])

        m_is = compute_metrics(is_trades, f'{strat_id} IS')
        m_oos = compute_metrics(oos_trades, f'{strat_id} OOS')

        pension = (m_is['ci_lo'] > 0 and m_oos['ci_lo'] > 0
                   and m_is['sharpe'] >= 0.15 and m_oos['sharpe'] >= 0.15)

        oos_rows.append([
            strat_id, uid,
            f"{m_is['n']}", f"{m_is['sharpe']:.3f}", f"{m_is['avg']:+.2f}%",
            f"{m_oos['n']}", f"{m_oos['sharpe']:.3f}", f"{m_oos['avg']:+.2f}%",
            '★' if pension else '✗',
        ])

    print('\n' + tabulate(oos_rows,
                          headers=['Strategy','Universe',
                                   'IS n','IS Sharpe','IS Avg',
                                   'OOS n','OOS Sharpe','OOS Avg', 'Pension'],
                          tablefmt='simple'))


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(grid):
    # Chart 1: Sharpe heatmap (strategy × universe)
    strat_ids = [s for s in grid if any(grid[s][u]['n'] >= 5 for u in UNIVERSES)]
    if not strat_ids:
        print("  No data for charts.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(strat_ids) * 0.35)))
    fig.suptitle('Agent 3 E3 — Strategy × Universe Grid', fontsize=13, fontweight='bold')

    # Sharpe bar chart: for each strategy, show all universes
    ax = axes[0]
    key_strats = ['A6sr T+1', 'A6sr T+2', 'A1 T+2', 'A7 T+1', 'C1 T+3', 'C1 T+5', 'B1 EOD']
    key_strats = [s for s in key_strats if s in grid]
    if not key_strats:
        key_strats = strat_ids[:7]

    x = np.arange(len(key_strats))
    width = 0.11
    colors = ['#2196F3', '#4CAF50', '#8BC34A', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
    for i, uid in enumerate(UNIVERSES):
        sharpes = [grid[s][uid]['sharpe'] if grid[s][uid]['n'] >= 5 else 0 for s in key_strats]
        ax.bar(x + i * width - 3 * width, sharpes, width, label=uid, color=colors[i], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(key_strats, rotation=45, ha='right', fontsize=8)
    ax.axhline(0, color='#333', lw=0.8)
    ax.axhline(0.40, color='green', ls='--', lw=0.8, alpha=0.5, label='Pension (0.40)')
    ax.set_ylabel('Sharpe')
    ax.set_title('Sharpe by Strategy × Universe')
    ax.legend(fontsize=6, loc='upper right')

    # Chart 2: Best universe for each strategy
    ax = axes[1]
    all_strats_with_data = [s for s in strat_ids if max(grid[s][u]['n'] for u in UNIVERSES) >= 5]
    best_data = []
    for s in all_strats_with_data:
        best_u = max(UNIVERSES, key=lambda u: grid[s][u]['sharpe'] if grid[s][u]['n'] >= 5 else -999)
        float_s = grid[s]['_FLOAT']['sharpe'] if grid[s]['_FLOAT']['n'] >= 5 else 0
        best_s = grid[s][best_u]['sharpe'] if grid[s][best_u]['n'] >= 5 else 0
        delta = best_s - float_s
        best_data.append((s, best_u, delta, best_s))

    best_data.sort(key=lambda x: x[2], reverse=True)
    labels = [f"{d[0]} ({d[1]})" for d in best_data[:15]]
    deltas = [d[2] for d in best_data[:15]]
    bar_colors = ['#27ae60' if d > 0 else '#e74c3c' for d in deltas]
    y = range(len(labels))
    ax.barh(y, deltas, color=bar_colors, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color='#333', lw=0.8)
    ax.set_xlabel('Δ Sharpe vs _FLOAT')
    ax.set_title('Best Universe Improvement Over _FLOAT')
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('agent3_e3_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Chart saved: agent3_e3_grid.png")

    # Chart 3: Universe comparison for key strategies
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('E3 — Key Strategies Across Universes', fontsize=13, fontweight='bold')

    panels = [
        ('A6sr T+1', axes2[0, 0], 'A6 sustained_rise T+1'),
        ('A6sr T+2', axes2[0, 1], 'A6 sustained_rise T+2'),
        ('C1 T+3',   axes2[1, 0], 'C1 Reversal T+3'),
        ('A1 T+2',   axes2[1, 1], 'A1 Raw Baseline T+2'),
    ]
    for strat_id, ax, title in panels:
        if strat_id not in grid: continue
        avgs = [grid[strat_id][u]['avg'] for u in UNIVERSES]
        ci_lo = [grid[strat_id][u]['ci_lo'] for u in UNIVERSES]
        ci_hi = [grid[strat_id][u]['ci_hi'] for u in UNIVERSES]
        ns = [grid[strat_id][u]['n'] for u in UNIVERSES]
        x = range(len(UNIVERSES))
        bar_colors = ['#2196F3' if u == '_FLOAT' else '#4CAF50' for u in UNIVERSES]
        ax.bar(x, avgs, color=bar_colors, alpha=0.7)
        ax.errorbar(x, avgs,
                    yerr=[np.array(avgs) - np.array(ci_lo), np.array(ci_hi) - np.array(avgs)],
                    fmt='none', color='black', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(UNIVERSES, rotation=45, ha='right', fontsize=7)
        ax.axhline(0, color='#333', lw=0.8)
        ax.set_ylabel('Avg PnL (%)')
        ax.set_title(title, fontsize=10)
        for xi, ni in zip(x, ns):
            ax.text(xi, avgs[xi] + 0.3, f'n={ni}', ha='center', fontsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('agent3_e3_panels.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Chart saved: agent3_e3_panels.png")


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════

t_total = time.time()

# Phase 0: Build universes
univ_sets, union_tickers, lb_dates = build_universes(t_total)

# Phase 1: Scan + enrich
enriched_a, enriched_b, enriched_c, a_sel, b_sel, dates = collect_data(univ_sets, union_tickers, t_total)

# Phase 2: Strategy × Universe grid
grid = compute_grid(enriched_a, enriched_b, enriched_c, a_sel, b_sel, univ_sets, dates)

# Phase 3: Reporting
print_grid(grid, dates)
run_oos_validation(grid, enriched_a, enriched_b, enriched_c, a_sel, b_sel, univ_sets, dates)
plot_results(grid)

elapsed = time.time() - t_total
print(f"\n{'='*80}")
print(f"  E3 COMPLETE — {len(STRATEGIES)} strategies × {len(UNIVERSES)} universes = {len(STRATEGIES) * len(UNIVERSES)} cells")
print(f"  Trades: A={len(enriched_a)}, B={len(enriched_b)}, C={len(enriched_c)}")
print(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'='*80}")
