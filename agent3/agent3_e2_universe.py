# %% Agent 3 — E2: Universe Comparison (7 Universes)
#
# Evaluate A6 (gap-down bounce) and C1 (multi-day reversal) across 8 universes:
#
#   _FLOAT  : hardcoded 194-ticker set (current baseline)
#   U1      : full US CS, price >= $10, float >= 10M (hard exclude on miss)
#   U2      : large cap >= $1B, CS, price >= $10
#   U3      : mid cap $300M–$10B, CS, price >= $10
#   U4      : Russell 1000 proxy (market cap >= $3B), CS, price >= $10
#   U5      : high liquidity (20-day ADV dollar vol >= $10M), CS, price >= $5
#   U6      : U1 minus biotech/healthcare SIC codes
#   U7      : _FLOAT + U1 tickers with extra guards (mcap >= $100M, adv >= $5M)
#
# Hold: A6 T+1, C1 T+3  (per E1/E1b findings)
# Guards: A6 gap cap -50%, C1 gap cap 50%
# IS/OOS split: 2026-01-01
# Colab-cell friendly. ~15-25 min runtime (API-bound).

import os, sys, time, re, copy, asyncio, math
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

A6_GAP_MIN_PCT   = 5.0
A6_GAP_MAX_PCT   = 50.0      # cap: drops > 50% are delistings
A6_PRICE_MIN     = 10.0
A6_TOP_N         = 5
A6_PRE_FILTER_N  = 15
A6_HOLD_DAYS     = 1          # T+1

C1_GAP_MIN_PCT     = 3.0
C1_GAP_MAX_PCT     = 50.0
C1_PRICE_MIN       = 20.0
C1_DECLINE_5D_MIN  = 0.15
C1_HOLD_DAYS       = 3        # T+3

LOW_WINDOW      = 3
MAX_LOW_BAR     = 12
FALLBACK_BAR    = 15
SKIP_BOUNCE_PCT = 0.15

FLOAT_MIN         = 10_000_000
ADV_MIN_U5        = 10_000_000
ADV_MIN_U7        = 5_000_000
MCAP_U2           = 1_000_000_000
MCAP_U3_LO        = 300_000_000
MCAP_U3_HI        = 10_000_000_000
MCAP_U4           = 3_000_000_000     # Russell 1000 proxy
MCAP_U7           = 100_000_000
ADV_LOOKBACK      = 20
CAPITAL           = 100_000

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')
ET = pytz.timezone('America/New_York')

_EXCLUDE_TK = {'SPY','QQQ','IWM','DIA','VXX','UVXY','SQQQ','TQQQ','SPXU','SPXL',
               'SDOW','UDOW','SDS','SSO','SH','LABU','LABD','SOXL','SOXS',
               'FNGU','FNGD','NUGT','DUST','JNUG','JDST','TNA','TZA',
               'UPRO','VIXY','SVXY','TVIX'}
_EXCLUDE_SUFFIX = ('W', 'WS', 'U', 'R')

_BIOTECH_HEALTHCARE_SIC = (
    set(range(2833, 2837)) |    # pharma / biologics
    set(range(3841, 3852)) |    # medical instruments
    set(range(8000, 8100)) |    # health services
    {5047, 5122}                # medical/drug wholesale
)

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

UNIVERSES = ['_FLOAT', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7']
UNIVERSE_LABELS = {
    '_FLOAT': '_FLOAT (194 hardcoded)',
    'U1':     'U1: Full CS (hard float)',
    'U2':     'U2: Large cap ≥$1B',
    'U3':     'U3: Mid cap $300M–$10B',
    'U4':     'U4: Russell 1000 proxy ≥$3B',
    'U5':     'U5: High ADV ≥$10M/day',
    'U6':     'U6: Full CS ex biotech',
    'U7':     'U7: _FLOAT + guarded open',
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

def _nth_prior_td(d, n):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    count = 0
    d -= timedelta(days=1)
    for _ in range(60):
        if _is_td(d):
            count += 1
            if count >= n: return d
        d -= timedelta(days=1)
    raise ValueError(f"Couldn't find {n}th prior TD from {d}")

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
            time.sleep(5 * (2 ** attempt))
    r = requests.get(url, params=p, timeout=60)
    r.raise_for_status()
    return r.json()


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
        tk = r.get('T','')
        if not _TK_RE.match(tk): continue
        out[tk] = {'o':float(r.get('o',0)),'h':float(r.get('h',0)),
                   'l':float(r.get('l',0)),'c':float(r.get('c',0)),'v':int(r.get('v',0))}
    _gd_cache[dt_str] = out
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# BULK TICKER REFERENCE FETCH (paginated, gets ALL US CS tickers)
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
    """Async-fetch individual ticker reference data (market_cap, shares_out, sic_code).
    Uses /v3/reference/tickers/{ticker} which returns the full detail."""
    global _ref_cache
    uncached = [tk for tk in tickers if tk not in _ref_cache]
    if not uncached:
        return _ref_cache

    for c_start in range(0, len(uncached), 200):
        chunk = uncached[c_start:c_start + 200]
        batch = asyncio.run(_fetch_refs_batch(chunk))
        _ref_cache.update(batch)
        done = min(c_start + 200, len(uncached))
        if done % 500 == 0 or done == len(uncached):
            print(f"    {done}/{len(uncached)}  |  {time.time()-t_total:.0f}s" if t_total else f"    {done}/{len(uncached)}")

    # Merge embedded _FLOAT for any still missing
    for tk, fs in _FLOAT.items():
        if tk not in _ref_cache:
            _ref_cache[tk] = {
                'type': 'CS', 'market_cap': 0, 'shares_out': fs,
                'sic_code': 0, 'name': '',
            }
        elif _ref_cache[tk].get('shares_out', 0) == 0 and fs > 0:
            _ref_cache[tk]['shares_out'] = fs

    return _ref_cache


# ═══════════════════════════════════════════════════════════════════════════════
# ADV (Average Daily Dollar Volume) — computed from grouped daily cache
# ═══════════════════════════════════════════════════════════════════════════════

_adv_cache = {}

def compute_adv(tk, date_str, trading_dates):
    """Compute 20-day trailing average dollar volume for a ticker."""
    key = (tk, date_str)
    if key in _adv_cache:
        return _adv_cache[key]

    idx = None
    for i, d in enumerate(trading_dates):
        if d == date_str:
            idx = i
            break
    if idx is None:
        _adv_cache[key] = 0
        return 0

    dollar_vols = []
    for j in range(max(0, idx - ADV_LOOKBACK), idx):
        gd = _gd_cache.get(trading_dates[j], {})
        bar = gd.get(tk)
        if bar and bar['c'] > 0 and bar['v'] > 0:
            dollar_vols.append(bar['c'] * bar['v'])

    adv = sum(dollar_vols) / len(dollar_vols) if dollar_vols else 0
    _adv_cache[key] = adv
    return adv


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC MINUTE BAR FETCH (RTH only — for A6 entry detection)
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_bars_one(session, tk, dt, sem):
    url = f"{_BASE}/v2/aggs/ticker/{tk}/range/1/minute/{dt}/{dt}"
    params = {"adjusted":"true","sort":"asc","limit":"500","apiKey":POLYGON_API_KEY}
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
            rth = []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                if '09:30' <= t_str < '16:00':
                    rth.append((t_str, {
                        'o':float(b['o']),'h':float(b['h']),
                        'l':float(b['l']),'c':float(b['c']),'v':int(b['v'])}))
            return (tk, dt), rth
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
# A6 PATTERN + ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

def classify_gap_down_pattern(rth_bars):
    if len(rth_bars) < 60: return 'no_data'
    op = rth_bars[0][1]['o']
    if op <= 0: return 'no_data'
    max_h30 = max(b[1]['h'] for b in rth_bars[:30])
    c60 = rth_bars[59][1]['c']
    rise = (max_h30 - op) / abs(op)
    rec = (c60 - op) / (max_h30 - op) if max_h30 - op > 0 else 0
    if rise > 0.03 and rec > 0.50: return 'sustained_rise'
    if rise > 0.03 and rec <= 0.50: return 'rise_then_drop'
    if rise <= 0.01: return 'continued_drop'
    return 'mild_rise'

def find_local_low(bars, window=LOW_WINDOW, max_bar=MAX_LOW_BAR, fb=FALLBACK_BAR):
    if len(bars) < window * 2 + 1: return None
    op = bars[0][1]['o']
    for i in range(window, min(max_bar, len(bars) - window)):
        cl = bars[i][1]['l']
        if all(bars[i-j][1]['l'] > cl and bars[i+j][1]['l'] > cl
               for j in range(1, window+1)):
            ci = i + window
            if ci < len(bars):
                return {'entry_price': bars[ci][1]['c'], 'entry_time': bars[ci][0],
                        'entry_type': 'local_low'}
    return None  # no fallback — local_low only for A6


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE MEMBERSHIP
# ═══════════════════════════════════════════════════════════════════════════════

def get_universes(tk, open_price, ref, adv_dollar):
    """Return set of universe IDs this ticker qualifies for on a given day.

    Args:
        tk:          ticker symbol
        open_price:  today's open price
        ref:         dict from _ref_cache (type, market_cap, shares_out, sic_code)
        adv_dollar:  20-day average daily dollar volume
    """
    if tk in _EXCLUDE_TK: return set()
    if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: return set()

    uvs = set()
    rtype = ref.get('type', '')
    mcap = ref.get('market_cap', 0) or 0
    so = ref.get('shares_out', 0) or 0
    sic = ref.get('sic_code', 0) or 0

    # _FLOAT baseline: must be in embedded set
    if tk in _FLOAT_SET:
        uvs.add('_FLOAT')

    # All remaining universes require CS type
    if rtype != 'CS':
        return uvs

    has_float = so > 0
    float_ok = has_float and so >= FLOAT_MIN

    # U1: Full CS, price >= $10, float >= 10M, hard exclude on miss
    if open_price >= 10.0 and float_ok:
        uvs.add('U1')

    # U2: Large cap >= $1B
    if mcap >= MCAP_U2 and open_price >= 10.0:
        uvs.add('U2')

    # U3: Mid cap $300M–$10B
    if MCAP_U3_LO <= mcap < MCAP_U3_HI and open_price >= 10.0:
        uvs.add('U3')

    # U4: Russell 1000 proxy (>= $3B)
    if mcap >= MCAP_U4 and open_price >= 10.0:
        uvs.add('U4')

    # U5: High ADV >= $10M/day, price >= $5
    if adv_dollar >= ADV_MIN_U5 and open_price >= 5.0:
        uvs.add('U5')

    # U6: U1 minus biotech/healthcare
    if 'U1' in uvs and sic not in _BIOTECH_HEALTHCARE_SIC:
        uvs.add('U6')

    # U7: _FLOAT + U1 tickers with extra guards
    if tk in _FLOAT_SET:
        uvs.add('U7')
    elif float_ok and open_price >= 10.0 and mcap >= MCAP_U7 and adv_dollar >= ADV_MIN_U7:
        uvs.add('U7')

    return uvs


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pnl_list, label=''):
    if not pnl_list:
        return {'label':label,'n':0,'avg':0,'med':0,'std':0,'wr':0,
                'sharpe':0,'pf':0,'ci_lo':0,'ci_hi':0,'trimmed_mean':0,
                'max_loss':0,'max_gain':0,'tpd':0}
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


def classify_status(m):
    if m['n'] < 10: return 'TOO FEW'
    if m['sharpe'] >= 0.40 and m['ci_lo'] > 0 and m['avg'] >= 1.0: return 'PENSION'
    if m['sharpe'] >= 0.30 and m['avg'] >= 0.5: return 'NEAR MISS'
    if m['sharpe'] >= 0.10: return 'DEGRADED'
    return 'DEAD'


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  MAIN BACKTEST
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest():
    t_total = time.time()

    print(f"\n{'='*90}")
    print(f"  AGENT 3 — E2: UNIVERSE COMPARISON (8 UNIVERSES)")
    print(f"  {BACKTEST_START} → {BACKTEST_END}  |  OOS: {OOS_CUTOFF}")
    print(f"  A6: gap-DOWN {A6_GAP_MIN_PCT}–{A6_GAP_MAX_PCT}%, sustained_rise, local_low, T+{A6_HOLD_DAYS}")
    print(f"  C1: 5d decline ≥{C1_DECLINE_5D_MIN*100:.0f}%, gap-UP {C1_GAP_MIN_PCT}–{C1_GAP_MAX_PCT}%, T+{C1_HOLD_DAYS}")
    print(f"{'='*90}")

    # ── Trading dates ──
    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"\n  {n_dates} trading days")

    # ── Phase 1: Fetch grouped daily (backtest + lookback) ──
    t1 = time.time()
    lookback_start = _nth_prior_td(BACKTEST_START, ADV_LOOKBACK + 3)
    lb = lookback_start
    lookback_dates = []
    while lb < datetime.strptime(BACKTEST_START, '%Y-%m-%d').date():
        if _is_td(lb): lookback_dates.append(lb.strftime('%Y-%m-%d'))
        lb += timedelta(days=1)

    all_dates = sorted(set(lookback_dates + dates))
    print(f"\n  Phase 1: Fetching grouped daily for {len(all_dates)} dates...")
    for i, dt in enumerate(all_dates):
        fetch_grouped_daily(dt)
        if (i+1) % 50 == 0 or i == len(all_dates)-1:
            print(f"    [{i+1:>3}/{len(all_dates)}] {dt}  |  {time.time()-t_total:.0f}s")

    # Outcome dates
    max_hold = max(A6_HOLD_DAYS, C1_HOLD_DAYS)
    outcome_dates = set()
    for dt in dates:
        for nd in _next_n_td(dt, max_hold):
            outcome_dates.add(nd.strftime('%Y-%m-%d'))
    for od in sorted(outcome_dates):
        if od not in _gd_cache:
            try: fetch_grouped_daily(od)
            except: pass
    print(f"    + {len(outcome_dates)} outcome dates  |  {time.time()-t_total:.0f}s")

    # ── Phase 2: Fetch individual ticker reference data ──
    # Bulk list endpoint doesn't return market_cap / shares_out.
    # We need /v3/reference/tickers/{ticker} for each ticker.
    # Collect all unique tickers seen in grouped daily, then async-fetch.
    t2 = time.time()
    all_tickers_seen = set()
    for gd in _gd_cache.values():
        for tk in gd:
            if _TK_RE.match(tk) and tk not in _EXCLUDE_TK:
                all_tickers_seen.add(tk)
    all_tickers_seen |= _FLOAT_SET
    print(f"\n  Phase 2: Fetching individual ticker reference for {len(all_tickers_seen)} tickers...")
    ref = fetch_ticker_references(sorted(all_tickers_seen), t_total=t_total)
    n_cs = sum(1 for v in ref.values() if v.get('type') == 'CS')
    n_mcap = sum(1 for v in ref.values() if v.get('market_cap', 0) > 0)
    n_so = sum(1 for v in ref.values() if v.get('shares_out', 0) > 0)
    print(f"    {len(ref)} tickers, {n_cs} CS, {n_mcap} with market_cap, {n_so} with shares_out")
    print(f"    ({time.time()-t2:.0f}s)")

    # ── Phase 3: Scan ALL universes ──
    t3 = time.time()
    print(f"\n  Phase 3: Scanning candidates across all universes...")

    gap_min_a6 = A6_GAP_MIN_PCT / 100.0
    gap_max_a6 = A6_GAP_MAX_PCT / 100.0
    gap_min_c1 = C1_GAP_MIN_PCT / 100.0
    gap_max_c1 = C1_GAP_MAX_PCT / 100.0

    # Per-universe candidate lists: {univ: {date: [candidates]}}
    a6_cands = {u: {} for u in UNIVERSES}
    c1_cands = {u: {} for u in UNIVERSES}

    # Universe daily sizes for reporting
    univ_daily_sizes = {u: [] for u in UNIVERSES}
    minute_bar_pairs = set()

    for i_dt, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = _gd_cache.get(prior_str, {})
        today = _gd_cache.get(dt, {})
        nxt = _next_n_td(dt, max_hold)

        # C1 lookback: 5 TDs ago
        try:
            c5d_str = _nth_prior_td(prior_str, 4).strftime('%Y-%m-%d')
            c5d_data = _gd_cache.get(c5d_str, {})
        except ValueError:
            c5d_data = {}

        # Count qualifying tickers per universe for this day
        day_univ_counts = {u: 0 for u in UNIVERSES}

        # Scan ALL tickers from today's grouped daily
        all_tickers_today = set(today.keys()) | _FLOAT_SET

        for tk in all_tickers_today:
            if tk not in today or tk not in prior: continue
            prev_c = prior[tk]['c']
            if prev_c <= 0: continue
            cur_o = today[tk]['o']
            if cur_o <= 0: continue

            r = ref.get(tk, {'type':'','market_cap':0,'shares_out':0,'sic_code':0})
            adv = compute_adv(tk, dt, all_dates)
            ticker_uvs = get_universes(tk, cur_o, r, adv)
            if not ticker_uvs: continue

            for u in ticker_uvs:
                day_univ_counts[u] += 1

            gap = (cur_o - prev_c) / prev_c
            t_dates = [nd.strftime('%Y-%m-%d') for nd in nxt]

            # ── A6: gap-DOWN ──
            if gap <= -gap_min_a6 and gap >= -gap_max_a6 and cur_o >= A6_PRICE_MIN:
                base = {
                    'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                    'open_price': cur_o, 'scan_date': dt,
                    'volume_today': today[tk]['v'], 't_dates': t_dates,
                }
                for u in ticker_uvs:
                    if dt not in a6_cands[u]: a6_cands[u][dt] = []
                    a6_cands[u][dt].append(copy.copy(base))

            # ── C1: gap-UP + prior 5d decline ──
            if gap >= gap_min_c1 and gap <= gap_max_c1 and cur_o >= C1_PRICE_MIN:
                c5d_tk = c5d_data.get(tk)
                if c5d_tk and c5d_tk['c'] > 0:
                    ret_5d = (prev_c - c5d_tk['c']) / c5d_tk['c']
                    if ret_5d <= -C1_DECLINE_5D_MIN:
                        base = {
                            'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                            'open_price': cur_o, 'scan_date': dt,
                            'volume_today': today[tk]['v'],
                            'prior_5d_return': ret_5d, 't_dates': t_dates,
                        }
                        for u in ticker_uvs:
                            if dt not in c1_cands[u]: c1_cands[u][dt] = []
                            c1_cands[u][dt].append(copy.copy(base))

        for u in UNIVERSES:
            univ_daily_sizes[u].append(day_univ_counts[u])

        if (i_dt+1) % 50 == 0:
            print(f"    [{i_dt+1:>3}/{n_dates}] {dt}  |  {time.time()-t_total:.0f}s")

    # Sort A6 candidates per day by gap magnitude, keep top PRE_FILTER_N for minute bars
    for u in UNIVERSES:
        for dt, cands in a6_cands[u].items():
            cands.sort(key=lambda x: x['gap_pct'])
            for c in cands[:A6_PRE_FILTER_N]:
                minute_bar_pairs.add((c['ticker'], dt))

    # Scan summary
    print(f"\n  Scan complete ({time.time()-t3:.0f}s)")
    print(f"\n  {'Universe':<30} {'Avg tickers/day':>16} {'A6 cands':>10} {'C1 cands':>10}")
    print(f"  {'─'*70}")
    for u in UNIVERSES:
        avg_size = np.mean(univ_daily_sizes[u]) if univ_daily_sizes[u] else 0
        a6_n = sum(len(v) for v in a6_cands[u].values())
        c1_n = sum(len(v) for v in c1_cands[u].values())
        print(f"  {UNIVERSE_LABELS[u]:<30} {avg_size:>12.0f}     {a6_n:>8}   {c1_n:>8}")

    # ── Phase 4: Fetch minute bars (union across all universes) ──
    t4 = time.time()
    print(f"\n  Phase 4: Fetching 1-min bars for {len(minute_bar_pairs)} A6 pairs...")
    all_bars = {}
    pairs_list = list(minute_bar_pairs)
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start+100]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        done = min(c_start+100, len(pairs_list))
        if done % 300 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    print(f"    → {len(all_bars)}/{len(minute_bar_pairs)} pairs with bars")

    # ── Phase 5: Classify patterns + find entries ──
    t5 = time.time()
    pattern_cache = {}
    entry_cache = {}
    for key, rth_bars in all_bars.items():
        pattern_cache[key] = classify_gap_down_pattern(rth_bars)
        entry_cache[key] = find_local_low(rth_bars) if rth_bars else None

    # ── Phase 6: Build trades per universe ──
    t6 = time.time()
    print(f"\n  Phase 6: Building trades for all universes...")

    results = {}
    for u in UNIVERSES:
        a6_trades = []
        c1_trades = []

        for dt in dates:
            # A6
            cands = a6_cands[u].get(dt, [])
            cands.sort(key=lambda x: x['gap_pct'])
            qualified = []
            for c in cands[:A6_PRE_FILTER_N]:
                tk = c['ticker']
                key = (tk, dt)
                r = ref.get(tk, {})
                if r.get('type', '') != 'CS' and tk not in _FLOAT_SET: continue
                pat = pattern_cache.get(key, 'no_data')
                if pat != 'sustained_rise': continue
                entry = entry_cache.get(key)
                if entry is None: continue

                tc = copy.deepcopy(c)
                tc['entry_price'] = entry['entry_price']
                tc['entry_time'] = entry['entry_time']
                tc['entry_type'] = entry['entry_type']
                tc['strategy'] = 'A6'
                tc['universe'] = u

                ep = tc['entry_price']
                for hi, td_str in enumerate(tc['t_dates'], 1):
                    td_d = _gd_cache.get(td_str, {}).get(tk, {})
                    tc[f'pnl_t{hi}'] = round((td_d['c'] - ep) / ep * 100, 3) if td_d.get('c') else None

                qualified.append(tc)

            for tc in qualified[:A6_TOP_N]:
                if tc.get(f'pnl_t{A6_HOLD_DAYS}') is not None:
                    a6_trades.append(tc)

            # C1
            cands = c1_cands[u].get(dt, [])
            for c in cands:
                tk = c['ticker']
                r = ref.get(tk, {})
                if r.get('type', '') != 'CS' and tk not in _FLOAT_SET: continue

                tc = copy.deepcopy(c)
                tc['entry_price'] = tc['open_price']
                tc['entry_time'] = '09:30'
                tc['entry_type'] = 'open'
                tc['strategy'] = 'C1'
                tc['universe'] = u

                ep = tc['entry_price']
                for hi, td_str in enumerate(tc['t_dates'], 1):
                    td_d = _gd_cache.get(td_str, {}).get(tk, {})
                    tc[f'pnl_t{hi}'] = round((td_d['c'] - ep) / ep * 100, 3) if td_d.get('c') else None

                if tc.get(f'pnl_t{C1_HOLD_DAYS}') is not None:
                    c1_trades.append(tc)

        results[u] = {'a6': a6_trades, 'c1': c1_trades}
        print(f"    {UNIVERSE_LABELS[u]:<35} A6={len(a6_trades):>4}  C1={len(c1_trades):>4}")

    elapsed = time.time() - t_total
    print(f"\n  TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return results, dates, all_dates


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON + VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

def compare_results(results, dates, all_dates):
    n_dates = len(dates)

    print(f"\n\n{'='*120}")
    print(f"  E2: UNIVERSE COMPARISON RESULTS")
    print(f"  {dates[0]} → {dates[-1]}  |  {n_dates} trading days  |  OOS: {OOS_CUTOFF}")
    print(f"{'='*120}")

    for strat, hold in [('a6', A6_HOLD_DAYS), ('c1', C1_HOLD_DAYS)]:
        pk = f'pnl_t{hold}'
        strat_upper = strat.upper()

        print(f"\n  {'━'*115}")
        print(f"  {strat_upper} — T+{hold} LONG")
        print(f"  {'━'*115}")

        # Build comparison table
        headers = ['Universe', 'n', 'TPD', 'Avg PnL', 'Median', 'Trim Mean',
                   'WR%', 'Sharpe', 'PF', 'CI Low', 'Worst', 'Status']
        rows = []

        for u in UNIVERSES:
            trades = results[u][strat]
            pnls = [t[pk] for t in trades if t.get(pk) is not None]
            m = compute_metrics(pnls, u)
            results[u][f'{strat}_m'] = m
            results[u][f'{strat}_pnls'] = pnls
            tpd = m['n'] / n_dates if n_dates > 0 else 0
            m['tpd'] = tpd
            status = classify_status(m)

            rows.append([
                UNIVERSE_LABELS[u],
                m['n'],
                f"{tpd:.2f}",
                f"{m['avg']:+.2f}%" if m['n'] else '—',
                f"{m['med']:+.2f}%" if m['n'] else '—',
                f"{m['trimmed_mean']:+.2f}%" if m['n'] else '—',
                f"{m['wr']*100:.0f}%" if m['n'] else '—',
                f"{m['sharpe']:.3f}" if m['n'] else '—',
                f"{m['pf']:.2f}" if m['n'] and m['pf'] < 999 else ('∞' if m['n'] else '—'),
                f"{m['ci_lo']:+.2f}" if m['n'] else '—',
                f"{m['max_loss']:+.1f}%" if m['n'] else '—',
                status,
            ])

        print()
        print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right'))

        # IS/OOS breakdown
        print(f"\n  IS/OOS Split (cutoff {OOS_CUTOFF}):")
        headers2 = ['Universe', 'IS n', 'IS Avg', 'IS Sharpe', 'OOS n', 'OOS Avg', 'OOS Sharpe', 'IS→OOS Δ']
        rows2 = []
        for u in UNIVERSES:
            trades = results[u][strat]
            is_pnls = [t[pk] for t in trades if t.get(pk) is not None and t['scan_date'] < OOS_CUTOFF]
            oos_pnls = [t[pk] for t in trades if t.get(pk) is not None and t['scan_date'] >= OOS_CUTOFF]
            is_m = compute_metrics(is_pnls)
            oos_m = compute_metrics(oos_pnls)
            results[u][f'{strat}_is'] = is_m
            results[u][f'{strat}_oos'] = oos_m
            delta = oos_m['sharpe'] - is_m['sharpe'] if is_m['n'] and oos_m['n'] else 0
            rows2.append([
                UNIVERSE_LABELS[u],
                is_m['n'],
                f"{is_m['avg']:+.2f}%" if is_m['n'] else '—',
                f"{is_m['sharpe']:.3f}" if is_m['n'] else '—',
                oos_m['n'],
                f"{oos_m['avg']:+.2f}%" if oos_m['n'] else '—',
                f"{oos_m['sharpe']:.3f}" if oos_m['n'] else '—',
                f"{delta:+.3f}" if is_m['n'] and oos_m['n'] else '—',
            ])
        print()
        print(tabulate(rows2, headers=headers2, tablefmt='simple', stralign='right'))

        # New tickers per universe (vs _FLOAT baseline)
        base_tks = set(t['ticker'] for t in results['_FLOAT'][strat])
        print(f"\n  New tickers vs _FLOAT baseline:")
        for u in UNIVERSES:
            if u == '_FLOAT': continue
            u_tks = set(t['ticker'] for t in results[u][strat])
            new_tks = sorted(u_tks - base_tks)
            if new_tks:
                new_pnls = [t[pk] for t in results[u][strat]
                            if t.get(pk) is not None and t['ticker'] in (u_tks - base_tks)]
                if new_pnls:
                    nm = compute_metrics(new_pnls)
                    print(f"    {UNIVERSE_LABELS[u]:<35} +{len(new_tks):>3} tickers  "
                          f"avg={nm['avg']:+.2f}%  Sh={nm['sharpe']:.3f}  WR={nm['wr']*100:.0f}%  "
                          f"({', '.join(new_tks[:10])}{'...' if len(new_tks)>10 else ''})")
                else:
                    print(f"    {UNIVERSE_LABELS[u]:<35} +{len(new_tks):>3} tickers (no trades with PnL)")
            else:
                print(f"    {UNIVERSE_LABELS[u]:<35}  same as _FLOAT")

        # Daily PnL for equity curves
        for u in UNIVERSES:
            daily_pnl = defaultdict(list)
            for t in results[u][strat]:
                p = t.get(pk)
                if p is not None:
                    daily_pnl[t['scan_date']].append(p)
            results[u][f'{strat}_daily'] = dict(daily_pnl)

    # ── VERDICT ──
    print(f"\n\n  {'═'*115}")
    print(f"  VERDICT")
    print(f"  {'═'*115}")

    for strat, hold in [('a6', A6_HOLD_DAYS), ('c1', C1_HOLD_DAYS)]:
        strat_upper = strat.upper()
        print(f"\n  {strat_upper} T+{hold}:")

        base = results['_FLOAT'][f'{strat}_m']
        ranked = []
        for u in UNIVERSES:
            m = results[u][f'{strat}_m']
            ranked.append((u, m))

        ranked.sort(key=lambda x: -x[1]['sharpe'])

        for rank, (u, m) in enumerate(ranked, 1):
            status = classify_status(m)
            delta_sh = m['sharpe'] - base['sharpe']
            delta_n = m['n'] - base['n']
            flag = '★' if rank == 1 else ' '
            print(f"    {flag} #{rank} {UNIVERSE_LABELS[u]:<35} "
                  f"Sharpe {m['sharpe']:.3f} ({delta_sh:+.3f})  "
                  f"n={m['n']} ({delta_n:+d})  "
                  f"Avg {m['avg']:+.2f}%  [{status}]")

        best_u, best_m = ranked[0]
        if best_u != '_FLOAT' and best_m['sharpe'] > base['sharpe']:
            print(f"\n    → RECOMMENDATION: Switch to {UNIVERSE_LABELS[best_u]} "
                  f"(+{best_m['sharpe']-base['sharpe']:.3f} Sharpe, "
                  f"+{best_m['n']-base['n']} trades)")
        elif best_u == '_FLOAT':
            print(f"\n    → Current _FLOAT baseline is already optimal")
        else:
            print(f"\n    → _FLOAT holds: best universe {UNIVERSE_LABELS[best_u]} "
                  f"gains {best_m['sharpe']-base['sharpe']:+.3f} Sharpe — not enough to switch")


def plot_comparison(results, dates):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    colors = ['#2980b9', '#e74c3c', '#27ae60', '#f39c12', '#8e44ad',
              '#1abc9c', '#d35400', '#34495e']

    # ── Chart 1: Equity curves (2 panels: A6, C1) ──
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle('E2: Universe Comparison — Equity Curves\n'
                 f'{dates[0]} → {dates[-1]}  |  $100K capital',
                 fontsize=13, fontweight='bold')

    for ax_idx, (strat, hold) in enumerate([('a6', A6_HOLD_DAYS), ('c1', C1_HOLD_DAYS)]):
        ax = axes[ax_idx]
        for u_idx, u in enumerate(UNIVERSES):
            daily = results[u].get(f'{strat}_daily', {})
            m = results[u][f'{strat}_m']
            if m['n'] == 0: continue
            cum = 0.0
            cum_series = [0.0]
            d_dates = [datetime.strptime(dates[0], '%Y-%m-%d')]
            for dt in dates:
                dp = daily.get(dt, [])
                if dp:
                    pos_size = CAPITAL / len(dp)
                    cum += sum(pos_size * (r/100) for r in dp)
                cum_series.append(cum)
                d_dates.append(datetime.strptime(dt, '%Y-%m-%d'))
            label = f'{u}: ${cum:+,.0f} Sh={m["sharpe"]:.3f} n={m["n"]}'
            ax.plot(d_dates, cum_series, color=colors[u_idx], linewidth=1.8 if u_idx==0 else 1.2,
                    ls='-' if u_idx == 0 else '--', label=label, alpha=0.9)

        oos_dt = datetime.strptime(OOS_CUTOFF, '%Y-%m-%d')
        ax.axvline(oos_dt, color='red', ls=':', lw=1, alpha=0.5)
        ax.axhline(0, color='#333', lw=0.8, ls=':')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.set_title(f'{strat.upper()} — T+{hold}', fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    p = os.path.join(BASE_DIR, 'agent3_e2_equity.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p}")

    # ── Chart 2: Bar chart comparison (Sharpe, Avg PnL, Trade count) ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('E2: Universe Comparison — Key Metrics', fontsize=13, fontweight='bold')

    x = np.arange(len(UNIVERSES))
    short_labels = ['_FL', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7']

    for row, (strat, hold) in enumerate([('a6', A6_HOLD_DAYS), ('c1', C1_HOLD_DAYS)]):
        for col, (title, fn) in enumerate([
            ('Sharpe', lambda r, s: r[f'{s}_m']['sharpe']),
            ('Avg PnL (%)', lambda r, s: r[f'{s}_m']['avg']),
            ('Trades', lambda r, s: r[f'{s}_m']['n']),
        ]):
            ax = axes[row, col]
            vals = [fn(results[u], strat) for u in UNIVERSES]
            bar_colors = [colors[i] for i in range(len(UNIVERSES))]
            bars = ax.bar(x, vals, color=bar_colors, alpha=0.8, edgecolor='white')
            ax.set_xticks(x)
            ax.set_xticklabels(short_labels, fontsize=8)
            ax.set_title(f'{strat.upper()} T+{hold}: {title}', fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.2, axis='y')
            if 'Sharpe' in title:
                ax.axhline(0.4, color='green', ls='--', lw=0.8, alpha=0.5, label='Pension threshold')
                ax.legend(fontsize=7)
            for bar, v in zip(bars, vals):
                fmt = f'{v:.2f}' if abs(v) < 100 else f'{v:.0f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       fmt, ha='center', va='bottom', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(BASE_DIR, 'agent3_e2_bars.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p}")

    # ── Chart 3: IS vs OOS Sharpe comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('E2: IS vs OOS Sharpe by Universe', fontsize=13, fontweight='bold')

    w = 0.35
    for ax_idx, (strat, hold) in enumerate([('a6', A6_HOLD_DAYS), ('c1', C1_HOLD_DAYS)]):
        ax = axes[ax_idx]
        is_vals = [results[u][f'{strat}_is']['sharpe'] for u in UNIVERSES]
        oos_vals = [results[u][f'{strat}_oos']['sharpe'] for u in UNIVERSES]
        ax.bar(x - w/2, is_vals, w, color='#3498db', alpha=0.8, label='IS')
        ax.bar(x + w/2, oos_vals, w, color='#e67e22', alpha=0.8, label='OOS')
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_title(f'{strat.upper()} T+{hold}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')
        ax.axhline(0, color='#333', lw=0.8, ls=':')

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(BASE_DIR, 'agent3_e2_isoos.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

results, dates, all_dates = run_backtest()
compare_results(results, dates, all_dates)
plot_comparison(results, dates)
