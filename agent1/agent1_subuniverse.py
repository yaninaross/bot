# %% Agent 1 — Sub-Universe Experiments
# Scan full US equity market once (gap>=1.5%, price>=$5), then test 45 sub-universes.
# Fetches data ONCE, filters many times. Each experiment uses TOP_N=5 for fair comparison.

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

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_START = '2026-01-02'
BACKTEST_END   = '2026-09-12'

SCAN_GAP_MIN = 0.015   # loosest gap (1.5%) to capture SU1 experiments
SCAN_PRICE_MIN = 5      # loosest price ($5) to capture SU2 experiments
COMPARE_N = 5           # fixed N for fair comparison

PEAK_WINDOW = 3
MAX_PEAK_BAR = 12
FALLBACK_BAR = 15
INITIAL_CAPITAL = 100_000

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
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25',
    '2026-06-19','2026-07-03','2026-09-07','2026-11-26','2026-12-25',
    '2027-01-01','2027-01-18','2027-02-15','2027-03-26','2027-05-31',
    '2027-06-18','2027-07-05','2027-09-06','2027-11-25','2027-12-24',
}

# Hardcoded float dict (from original Agent 1 backtest — 217 tickers)
_HARDCODED_FLOAT = {
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
# EXPERIMENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
# Each spec is a dict of filter criteria. Universal filter function checks all.
# Omitted keys = no filter for that dimension.

EXPERIMENTS = OrderedDict([
    # ── SU1: Gap Magnitude Tiers ──
    ('gap_1.5-2%',  {'grp':'SU1_Gap','gap_min':0.015,'gap_max':0.020,'price_min':5}),
    ('gap_2-3%',    {'grp':'SU1_Gap','gap_min':0.020,'gap_max':0.030,'price_min':5}),
    ('gap_3-5%',    {'grp':'SU1_Gap','gap_min':0.030,'gap_max':0.050,'price_min':5}),
    ('gap_5-8%',    {'grp':'SU1_Gap','gap_min':0.050,'gap_max':0.080,'price_min':5}),
    ('gap_8%+',     {'grp':'SU1_Gap','gap_min':0.080,'price_min':5}),
    ('gap_5%+',     {'grp':'SU1_Gap','gap_min':0.050,'price_min':5}),

    # ── SU2: Price Tiers (gap >= 5%) ──
    ('px_$5-10',    {'grp':'SU2_Price','gap_min':0.05,'price_min':5,'price_max':10}),
    ('px_$10-20',   {'grp':'SU2_Price','gap_min':0.05,'price_min':10,'price_max':20}),
    ('px_$20-50',   {'grp':'SU2_Price','gap_min':0.05,'price_min':20,'price_max':50}),
    ('px_$50-100',  {'grp':'SU2_Price','gap_min':0.05,'price_min':50,'price_max':100}),
    ('px_$100+',    {'grp':'SU2_Price','gap_min':0.05,'price_min':100}),

    # ── SU3: Market Cap Proxy (gap >= 5%, price >= $5) ──
    ('cap_<300M',   {'grp':'SU3_Cap','gap_min':0.05,'price_min':5,'cap_max':300e6}),
    ('cap_300M-2B', {'grp':'SU3_Cap','gap_min':0.05,'price_min':5,'cap_min':300e6,'cap_max':2e9}),
    ('cap_2B-10B',  {'grp':'SU3_Cap','gap_min':0.05,'price_min':5,'cap_min':2e9,'cap_max':10e9}),
    ('cap_10B+',    {'grp':'SU3_Cap','gap_min':0.05,'price_min':5,'cap_min':10e9}),

    # ── SU4: Gap + Price Joint ──
    ('g5_p10',      {'grp':'SU4_Joint','gap_min':0.05,'price_min':10,'float_min':10e6}),
    ('g5_p20',      {'grp':'SU4_Joint','gap_min':0.05,'price_min':20,'float_min':10e6}),
    ('g5_p50',      {'grp':'SU4_Joint','gap_min':0.05,'price_min':50,'float_min':10e6}),
    ('g7_p10',      {'grp':'SU4_Joint','gap_min':0.07,'price_min':10,'float_min':10e6}),
    ('g7_p20',      {'grp':'SU4_Joint','gap_min':0.07,'price_min':20,'float_min':10e6}),
    ('g10_p10',     {'grp':'SU4_Joint','gap_min':0.10,'price_min':10,'float_min':10e6}),

    # ── SU5: Absolute Gap $ (gap >= 5%) ──
    ('abs_<$1',     {'grp':'SU5_AbsGap','gap_min':0.05,'price_min':5,'abs_gap_max':1}),
    ('abs_$1-3',    {'grp':'SU5_AbsGap','gap_min':0.05,'price_min':5,'abs_gap_min':1,'abs_gap_max':3}),
    ('abs_$3-5',    {'grp':'SU5_AbsGap','gap_min':0.05,'price_min':5,'abs_gap_min':3,'abs_gap_max':5}),
    ('abs_$5-10',   {'grp':'SU5_AbsGap','gap_min':0.05,'price_min':5,'abs_gap_min':5,'abs_gap_max':10}),
    ('abs_$10+',    {'grp':'SU5_AbsGap','gap_min':0.05,'price_min':5,'abs_gap_min':10}),

    # ── SU6: Float Size Tiers (gap >= 5%) ──
    ('flt_<10M',    {'grp':'SU6_Float','gap_min':0.05,'price_min':5,'float_max':10e6}),
    ('flt_10-50M',  {'grp':'SU6_Float','gap_min':0.05,'price_min':5,'float_min':10e6,'float_max':50e6}),
    ('flt_50-200M', {'grp':'SU6_Float','gap_min':0.05,'price_min':5,'float_min':50e6,'float_max':200e6}),
    ('flt_200-500M',{'grp':'SU6_Float','gap_min':0.05,'price_min':5,'float_min':200e6,'float_max':500e6}),
    ('flt_500M+',   {'grp':'SU6_Float','gap_min':0.05,'price_min':5,'float_min':500e6}),

    # ── SU7: Exchange (gap >= 5%, price >= $10) ──
    ('exch_nasdaq', {'grp':'SU7_Exch','gap_min':0.05,'price_min':10,'float_min':10e6,
                     'exchanges':['XNAS','XNMS','XNCM']}),
    ('exch_nyse',   {'grp':'SU7_Exch','gap_min':0.05,'price_min':10,'float_min':10e6,
                     'exchanges':['XNYS','XASE','ARCX','BATS']}),
    ('exch_all',    {'grp':'SU7_Exch','gap_min':0.05,'price_min':10,'float_min':10e6}),

    # ── SU8: Prior-Day Volatility (gap >= 5%, price >= $10) ──
    ('vol_<2%',     {'grp':'SU8_Vol','gap_min':0.05,'price_min':10,'float_min':10e6,
                     'adr_max':0.02}),
    ('vol_2-5%',    {'grp':'SU8_Vol','gap_min':0.05,'price_min':10,'float_min':10e6,
                     'adr_min':0.02,'adr_max':0.05}),
    ('vol_5%+',     {'grp':'SU8_Vol','gap_min':0.05,'price_min':10,'float_min':10e6,
                     'adr_min':0.05}),

    # ── SU9: Day of Week (gap >= 5%, price >= $10) ──
    ('dow_mon',     {'grp':'SU9_DoW','gap_min':0.05,'price_min':10,'float_min':10e6,'weekdays':[0]}),
    ('dow_tue',     {'grp':'SU9_DoW','gap_min':0.05,'price_min':10,'float_min':10e6,'weekdays':[1]}),
    ('dow_wed',     {'grp':'SU9_DoW','gap_min':0.05,'price_min':10,'float_min':10e6,'weekdays':[2]}),
    ('dow_thu',     {'grp':'SU9_DoW','gap_min':0.05,'price_min':10,'float_min':10e6,'weekdays':[3]}),
    ('dow_fri',     {'grp':'SU9_DoW','gap_min':0.05,'price_min':10,'float_min':10e6,'weekdays':[4]}),
    ('dow_tue-thu', {'grp':'SU9_DoW','gap_min':0.05,'price_min':10,'float_min':10e6,'weekdays':[1,2,3]}),

    # ── Baselines ──
    ('BASE_full',   {'grp':'BASELINE','gap_min':0.05,'price_min':10,'float_min':10e6}),
    ('BASE_hardcoded',{'grp':'BASELINE','gap_min':0.05,'price_min':10,'hardcoded_only':True}),
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
# ASYNC FLOAT + EXCHANGE FETCH
# ═══════════════════════════════════════════════════════════════════════════════

_ticker_info = {}  # {ticker: {'float': int|None, 'exchange': str|None}}

async def _fetch_info_one(session, tk, sem):
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
                        return tk, {'float': None, 'exchange': None}
                    data = await resp.json()
            except Exception:
                await asyncio.sleep(1)
                continue
            res = data.get('results', {})
            wso = res.get('weighted_shares_outstanding')
            sso = res.get('share_class_shares_outstanding')
            exch = res.get('primary_exchange')
            return tk, {'float': wso or sso, 'exchange': exch}
        return tk, {'float': None, 'exchange': None}

async def _fetch_info_batch(tickers):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_info_one(session, tk, sem) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: val for tk, val in results}

def fetch_ticker_info(tickers):
    uncached = [tk for tk in tickers if tk not in _ticker_info]
    if uncached:
        batch = asyncio.run(_fetch_info_batch(uncached))
        _ticker_info.update(batch)
    return {tk: _ticker_info.get(tk, {'float':None,'exchange':None}) for tk in tickers}


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC MINUTE BAR FETCH
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
# LOCAL HIGH + EQUITY
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_high(bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    fallback_bar=FALLBACK_BAR):
    if len(bars) < window * 2 + 1:
        return None
    for i in range(window, min(max_peak_bar, len(bars) - window)):
        ch = bars[i][1]['h']
        if all(bars[i-j][1]['h'] < ch and bars[i+j][1]['h'] < ch
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(bars):
                return {'entry_price':bars[ci][1]['c'],'entry_time':bars[ci][0],
                        'entry_type':'local_high','peak_bar':i,
                        'peak_time':bars[i][0],'peak_price':bars[i][1]['h']}
    if fallback_bar is not None and fallback_bar < len(bars):
        return {'entry_price':bars[fallback_bar][1]['c'],'entry_time':bars[fallback_bar][0],
                'entry_type':'fallback','peak_bar':None,'peak_time':None,'peak_price':None}
    return None

def compute_equity(t2_trades):
    capital = INITIAL_CAPITAL
    equity = [capital]
    eq_dates = []
    daily_pnl = defaultdict(list)
    for t in t2_trades:
        daily_pnl[t['scan_date']].append(t['pnl_t2'])
    for dt in sorted(daily_pnl):
        n = len(daily_pnl[dt])
        per_trade = capital / n
        day_dollar = sum(per_trade * (r / 100) for r in daily_pnl[dt])
        capital += day_dollar
        equity.append(capital)
        eq_dates.append(dt)
    return equity, eq_dates, capital


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def passes_filter(c, spec):
    g = c['gap_pct']
    p = c['open_price']

    if 'gap_min' in spec and g < spec['gap_min']: return False
    if 'gap_max' in spec and g >= spec['gap_max']: return False
    if 'price_min' in spec and p < spec['price_min']: return False
    if 'price_max' in spec and p >= spec['price_max']: return False

    flt = c.get('float_shares')
    if 'float_min' in spec:
        if flt is None: return False
        if flt < spec['float_min']: return False
    if 'float_max' in spec:
        if flt is None: return False
        if flt >= spec['float_max']: return False

    cap = c.get('market_cap')
    if 'cap_min' in spec:
        if cap is None: return False
        if cap < spec['cap_min']: return False
    if 'cap_max' in spec:
        if cap is None: return False
        if cap >= spec['cap_max']: return False

    ag = c.get('abs_gap')
    if 'abs_gap_min' in spec:
        if ag is None: return False
        if ag < spec['abs_gap_min']: return False
    if 'abs_gap_max' in spec:
        if ag is None: return False
        if ag >= spec['abs_gap_max']: return False

    if 'exchanges' in spec:
        exch = c.get('exchange')
        if exch is None: return False
        if exch not in spec['exchanges']: return False

    adr = c.get('prior_adr')
    if 'adr_min' in spec:
        if adr is None: return False
        if adr < spec['adr_min']: return False
    if 'adr_max' in spec:
        if adr is None: return False
        if adr >= spec['adr_max']: return False

    if 'weekdays' in spec:
        wd = c.get('weekday')
        if wd not in spec['weekdays']: return False

    if spec.get('hardcoded_only'):
        tk = c['ticker']
        if tk not in _HARDCODED_FLOAT: return False
        if _HARDCODED_FLOAT[tk] < 10_000_000: return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiments():
    t_total = time.time()
    timings = {}

    print(f"\n{'='*80}")
    print(f"  AGENT 1 — SUB-UNIVERSE EXPERIMENTS")
    print(f"  {BACKTEST_START} → {BACKTEST_END}  |  {len(EXPERIMENTS)} experiments  |  N={COMPARE_N}")
    print(f"  Scan: gap >= {SCAN_GAP_MIN*100:.1f}%, price >= ${SCAN_PRICE_MIN}")
    print(f"{'='*80}")

    # ── Enumerate dates ──
    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)

    # ── Phase 1: Scan all days (loosest filter) ──
    t1 = time.time()
    print(f"\n  Phase 1: Scanning {n_dates} days (gap >= {SCAN_GAP_MIN*100:.1f}%, price >= ${SCAN_PRICE_MIN})...")
    raw_by_date = {}
    all_tickers = set()
    total_raw = 0

    for i, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = fetch_grouped_daily(prior_str)
        today = fetch_grouped_daily(dt)

        raw = []
        weekday = datetime.strptime(dt, '%Y-%m-%d').weekday()
        for tk in today:
            if tk in _EXCLUDE_TK: continue
            if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
            if tk not in prior: continue
            prev_c = prior[tk]['c']
            if prev_c <= 0: continue
            cur_p = today[tk]['o']
            if cur_p <= 0 or cur_p < SCAN_PRICE_MIN: continue
            gap = (cur_p - prev_c) / prev_c
            if gap < SCAN_GAP_MIN: continue

            prior_bar = prior.get(tk, {})
            prior_adr = None
            if prior_bar.get('c', 0) > 0:
                prior_adr = (prior_bar['h'] - prior_bar['l']) / prior_bar['c']

            raw.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_p, 'abs_gap': cur_p - prev_c,
                'volume': today[tk]['v'], 'scan_date': dt,
                'weekday': weekday, 'prior_adr': prior_adr,
                'float_shares': None, 'exchange': None, 'market_cap': None,
            })
            all_tickers.add(tk)

        raw.sort(key=lambda x: -x['gap_pct'])
        raw_by_date[dt] = raw
        total_raw += len(raw)

        if (i+1) % 20 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_raw} raw  |  "
                  f"{len(all_tickers)} unique  |  {time.time()-t_total:.0f}s")

    timings['scan'] = time.time() - t1

    # ── Phase 2: Fetch float + exchange for all tickers ──
    t2 = time.time()
    print(f"\n  Phase 2: Fetching float + exchange for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    chunk = 200
    for c_start in range(0, len(tk_list), chunk):
        batch = tk_list[c_start:c_start + chunk]
        fetch_ticker_info(batch)
        done = min(c_start + chunk, len(tk_list))
        if done % 1000 == 0 or done == len(tk_list):
            print(f"    fetched {done}/{len(tk_list)}  |  {time.time()-t_total:.0f}s")

    n_has = sum(1 for tk in all_tickers if (_ticker_info.get(tk,{}).get('float') is not None))
    timings['info'] = time.time() - t2
    print(f"  → Got float for {n_has}/{len(all_tickers)}  |  {timings['info']:.0f}s")

    # ── Enrich candidates ──
    t3 = time.time()
    print(f"\n  Phase 3: Enriching candidates with float/exchange/cap...")
    for dt in dates:
        for c in raw_by_date[dt]:
            info = _ticker_info.get(c['ticker'], {})
            c['float_shares'] = info.get('float')
            c['exchange'] = info.get('exchange')
            if c['float_shares'] is not None and c['open_price'] > 0:
                c['market_cap'] = c['open_price'] * c['float_shares']
            # T+1 / T+2 dates
            nxt = _next_n_td(dt, 2)
            c['t1_date'] = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
            c['t2_date'] = nxt[1].strftime('%Y-%m-%d') if len(nxt) >= 2 else None
    timings['enrich'] = time.time() - t3

    # ── Phase 4: Select top N per day for each experiment → union of (tk, dt) pairs ──
    t4 = time.time()
    print(f"\n  Phase 4: Selecting top {COMPARE_N} per day for {len(EXPERIMENTS)} experiments...")
    exp_selections = {}  # {exp_name: {date: [candidates]}}
    all_pairs = set()

    for exp_name, spec in EXPERIMENTS.items():
        daily_picks = {}
        for dt in dates:
            passed = [c for c in raw_by_date[dt] if passes_filter(c, spec)]
            passed.sort(key=lambda x: -x['gap_pct'])
            top = passed[:COMPARE_N]
            daily_picks[dt] = top
            for c in top:
                all_pairs.add((c['ticker'], c['scan_date']))
        exp_selections[exp_name] = daily_picks

    timings['select'] = time.time() - t4
    print(f"  → {len(all_pairs)} unique (ticker, date) pairs needed")

    # ── Phase 5: Fetch minute bars ──
    t5 = time.time()
    print(f"\n  Phase 5: Fetching 1-min bars for {len(all_pairs)} pairs...")
    pairs_list = list(all_pairs)
    chunk = 100
    all_bars = {}
    for c_start in range(0, len(pairs_list), chunk):
        batch = pairs_list[c_start:c_start + chunk]
        result = fetch_minute_bars_batch(batch)
        all_bars.update(result)
        done = min(c_start + chunk, len(pairs_list))
        if done % 500 == 0 or done == len(pairs_list):
            print(f"    fetched {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")

    timings['bars'] = time.time() - t5
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)}  |  {timings['bars']:.0f}s")

    # ── Phase 6: Local high entries ──
    t6 = time.time()
    print(f"\n  Phase 6: Finding local high entries...")
    entry_cache = {}
    for key, bars in all_bars.items():
        if bars and len(bars) >= PEAK_WINDOW * 2 + 1:
            entry_cache[key] = find_local_high(bars)
        else:
            entry_cache[key] = None
    timings['entry'] = time.time() - t6

    # ── Phase 7: Outcome dates ──
    t7 = time.time()
    print(f"  Phase 7: Fetching outcome dates...")
    outcome_dates = set()
    for dt in dates:
        for exp_picks in exp_selections.values():
            for c in exp_picks.get(dt, []):
                outcome_dates.add(c['scan_date'])
                if c.get('t1_date'): outcome_dates.add(c['t1_date'])
                if c.get('t2_date'): outcome_dates.add(c['t2_date'])

    to_fetch = sorted(od for od in outcome_dates if od not in _gd_cache)
    for j, od in enumerate(to_fetch):
        try: fetch_grouped_daily(od)
        except: pass
        if (j+1) % 20 == 0:
            print(f"    {j+1}/{len(to_fetch)}")
    timings['outcomes'] = time.time() - t7

    # ── Phase 8: Compute stats for each experiment ──
    t8 = time.time()
    print(f"\n  Phase 8: Computing stats for {len(EXPERIMENTS)} experiments...")

    results = {}
    for exp_name, spec in EXPERIMENTS.items():
        trades = []
        for dt in dates:
            for c in exp_selections[exp_name].get(dt, []):
                tc = copy.deepcopy(c)
                key = (tc['ticker'], tc['scan_date'])
                entry = entry_cache.get(key)
                if entry is None:
                    continue

                tc['entry_price'] = entry['entry_price']
                tc['entry_time'] = entry['entry_time']
                tc['entry_type'] = entry['entry_type']

                tk, ep = tc['ticker'], tc['entry_price']
                t0d = _gd_cache.get(tc['scan_date'],{}).get(tk,{})
                tc['pnl_eod'] = round((ep-t0d['c'])/ep*100,3) if t0d.get('c') else None
                t1d = _gd_cache.get(tc.get('t1_date',''),{}).get(tk,{})
                tc['pnl_t1'] = round((ep-t1d['c'])/ep*100,3) if t1d.get('c') else None
                t2d = _gd_cache.get(tc.get('t2_date',''),{}).get(tk,{})
                tc['t2_close'] = t2d.get('c')
                tc['pnl_t2'] = round((ep-t2d['c'])/ep*100,3) if t2d.get('c') else None

                trades.append(tc)

        t2_trades = [t for t in trades if t.get('pnl_t2') is not None]
        pnl_arr = np.array([t['pnl_t2'] for t in t2_trades]) if t2_trades else np.array([])
        n = len(pnl_arr)

        if n == 0:
            results[exp_name] = {
                'grp':spec.get('grp','?'),'n':0,'avg':0,'med':0,'std':0,
                'wr':0,'sharpe':0,'pf':0,'ci_lo':0,'ret_pct':0,'max_dd':0,
                'equity':[],'eq_dates':[],'final_cap':INITIAL_CAPITAL,
                't2_trades':[],'trades_per_day':0,
            }
            continue

        equity, eq_dates, final_cap = compute_equity(t2_trades)
        avg = float(np.mean(pnl_arr))
        std = float(np.std(pnl_arr, ddof=1)) if n > 1 else 0
        wr = float((pnl_arr > 0).mean())
        sharpe = avg / std if std > 0 else 0
        se = std / np.sqrt(n) if n > 0 else 0
        ci_lo = avg - 1.96 * se
        w = pnl_arr[pnl_arr > 0]; l = pnl_arr[pnl_arr < 0]
        pf = float(w.sum() / abs(l.sum())) if len(l) > 0 and l.sum() != 0 else float('inf')
        ret_pct = (final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        eq_arr = np.array(equity)
        peak = np.maximum.accumulate(eq_arr)
        max_dd = float(((eq_arr - peak) / peak * 100).min())

        trading_days_with_trades = len(set(t['scan_date'] for t in t2_trades))
        tpd = n / trading_days_with_trades if trading_days_with_trades > 0 else 0

        results[exp_name] = {
            'grp': spec.get('grp','?'), 'n': n, 'avg': avg, 'med': float(np.median(pnl_arr)),
            'std': std, 'wr': wr, 'sharpe': sharpe, 'pf': pf, 'ci_lo': ci_lo,
            'ret_pct': ret_pct, 'max_dd': max_dd, 'equity': equity, 'eq_dates': eq_dates,
            'final_cap': final_cap, 't2_trades': t2_trades, 'trades_per_day': tpd,
        }

    timings['compute'] = time.time() - t8
    elapsed = time.time() - t_total

    # ── Timing ──
    print(f"\n  {'─'*60}")
    print(f"  TIMING BREAKDOWN")
    print(f"  {'─'*60}")
    timing_rows = []
    for step, dur in [('Scan (grouped daily)', timings['scan']),
                      ('Float+exchange fetch', timings['info']),
                      ('Enrich candidates', timings['enrich']),
                      ('Select top-N per exp', timings['select']),
                      ('Minute bars fetch', timings['bars']),
                      ('Local high entries', timings['entry']),
                      ('Outcome dates', timings['outcomes']),
                      ('Compute stats', timings['compute'])]:
        pct = dur / elapsed * 100
        timing_rows.append([step, f"{dur:.0f}s", f"{pct:.0f}%"])
    print(tabulate(timing_rows, headers=['Phase','Time','%'], tablefmt='simple'))
    print(f"\n  TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Print results ──
    print_results(results, dates)
    plot_results(results, dates)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(results, dates):
    baseline_sharpe = results.get('BASE_hardcoded', {}).get('sharpe', 0)

    # Group by category
    groups = OrderedDict()
    for name, r in results.items():
        grp = r['grp']
        if grp not in groups:
            groups[grp] = []
        groups[grp].append((name, r))

    print(f"\n{'='*100}")
    print(f"  SUB-UNIVERSE EXPERIMENT RESULTS — {dates[0]} → {dates[-1]}")
    print(f"  Baseline (hardcoded): Sharpe={baseline_sharpe:.3f}")
    print(f"{'='*100}")

    for grp, items in groups.items():
        print(f"\n  ── {grp} ──")
        rows = []
        for name, r in sorted(items, key=lambda x: -x[1]['sharpe']):
            ci_flag = '✓' if r['ci_lo'] > 0 else '✗'
            beat = '★' if r['sharpe'] > baseline_sharpe and r['n'] >= 20 else ''
            rows.append([
                f"{name}{beat}", r['n'], f"{r['trades_per_day']:.1f}",
                f"{r['avg']:+.2f}%", f"{r['med']:+.2f}%",
                f"{r['wr']*100:.0f}%", f"{r['sharpe']:.3f}", f"{r['pf']:.2f}",
                f"{r['ci_lo']:+.2f}% {ci_flag}", f"{r['max_dd']:.1f}%",
                f"{r['ret_pct']:+.1f}%",
            ])
        print(tabulate(rows, headers=[
            'Experiment','n','TPD','Avg','Median','WR','Sharpe','PF','CI_lo','MaxDD','Return'
        ], tablefmt='simple'))

    # ── Overall ranking (top 15 by Sharpe, min 20 trades) ──
    valid = [(name, r) for name, r in results.items() if r['n'] >= 20]
    top = sorted(valid, key=lambda x: -x[1]['sharpe'])[:15]

    print(f"\n\n{'='*100}")
    print(f"  TOP 15 SUB-UNIVERSES BY SHARPE (min n=20)")
    print(f"  ★ = beats hardcoded baseline (Sharpe {baseline_sharpe:.3f})")
    print(f"{'='*100}\n")

    rows = []
    for rank, (name, r) in enumerate(top, 1):
        beat = '★' if r['sharpe'] > baseline_sharpe else ''
        ci_flag = '✓' if r['ci_lo'] > 0 else '✗'
        rows.append([
            f"#{rank}", f"{name}{beat}", r['grp'], r['n'],
            f"{r['avg']:+.2f}%", f"{r['wr']*100:.0f}%",
            f"{r['sharpe']:.3f}", f"{r['pf']:.2f}",
            f"{r['ci_lo']:+.2f}% {ci_flag}",
            f"{r['max_dd']:.1f}%", f"{r['ret_pct']:+.1f}%",
        ])
    print(tabulate(rows, headers=[
        'Rank','Experiment','Group','n','Avg','WR','Sharpe','PF','CI_lo','MaxDD','Return'
    ], tablefmt='simple'))

    # ── Best per group ──
    print(f"\n\n  ── BEST PER GROUP ──")
    rows = []
    for grp, items in groups.items():
        best = max(items, key=lambda x: x[1]['sharpe'])
        name, r = best
        beat = '★' if r['sharpe'] > baseline_sharpe and r['n'] >= 20 else ''
        rows.append([grp, f"{name}{beat}", r['n'], f"{r['avg']:+.2f}%",
                     f"{r['wr']*100:.0f}%", f"{r['sharpe']:.3f}",
                     f"{r['ret_pct']:+.1f}%"])
    print(tabulate(rows, headers=['Group','Best','n','Avg','WR','Sharpe','Return'],
                   tablefmt='simple'))


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results, dates):
    baseline_sharpe = results.get('BASE_hardcoded', {}).get('sharpe', 0)

    # Color per group
    grp_colors = {
        'SU1_Gap':'#e74c3c','SU2_Price':'#e67e22','SU3_Cap':'#f1c40f',
        'SU4_Joint':'#2ecc71','SU5_AbsGap':'#1abc9c','SU6_Float':'#3498db',
        'SU7_Exch':'#9b59b6','SU8_Vol':'#e91e63','SU9_DoW':'#795548',
        'BASELINE':'#2c3e50',
    }

    # ── Figure 1: Sharpe bar chart (all experiments) ──
    valid = [(n, r) for n, r in results.items() if r['n'] >= 10]
    valid.sort(key=lambda x: -x[1]['sharpe'])

    fig, ax = plt.subplots(figsize=(14, max(8, len(valid) * 0.35)))
    names = [v[0] for v in valid]
    sharpes = [v[1]['sharpe'] for v in valid]
    colors = [grp_colors.get(v[1]['grp'], '#999') for v in valid]

    y = range(len(valid))
    bars = ax.barh(y, sharpes, color=colors, alpha=0.8, edgecolor='white')
    ax.axvline(baseline_sharpe, color='#e74c3c', ls='--', lw=2,
               label=f'Hardcoded baseline ({baseline_sharpe:.3f})')
    ax.axvline(0, color='#333', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title(f'Sub-Universe Experiments — Sharpe Comparison\n'
                 f'{dates[0]} → {dates[-1]}  |  N={COMPARE_N}', fontsize=12)
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    for i, (b, s, v) in enumerate(zip(bars, sharpes, valid)):
        n_trades = v[1]['n']
        ax.text(max(s, 0) + 0.005, i, f'{s:.3f} (n={n_trades})',
                va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig('agent1_subuniverse_sharpe.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Chart saved: agent1_subuniverse_sharpe.png")

    # ── Figure 2: Top 8 equity curves ──
    top8 = [v for v in valid if v[1].get('eq_dates')][:8]
    if len(top8) < 2:
        return

    fig2, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    for name, r in top8:
        if not r['eq_dates']: continue
        eq_d = [datetime.strptime(dates[0], '%Y-%m-%d')] + \
               [datetime.strptime(d, '%Y-%m-%d') for d in r['eq_dates']]
        color = grp_colors.get(r['grp'], '#999')
        ax.plot(eq_d, r['equity'], label=f"{name} ({r['ret_pct']:+.0f}%)",
                color=color, linewidth=1.5, alpha=0.8)

    # Add baseline
    br = results.get('BASE_hardcoded', {})
    if br.get('eq_dates'):
        eq_d = [datetime.strptime(dates[0], '%Y-%m-%d')] + \
               [datetime.strptime(d, '%Y-%m-%d') for d in br['eq_dates']]
        ax.plot(eq_d, br['equity'], label=f"BASELINE ({br['ret_pct']:+.0f}%)",
                color='#2c3e50', linewidth=2.5, ls='--')

    ax.axhline(INITIAL_CAPITAL, color='#999', ls=':', lw=0.8)
    ax.set_ylabel('Equity ($)')
    ax.set_title('Top 8 Sub-Universes — Equity Curves')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=7, loc='upper left')

    # ── Avg PnL vs WR scatter for all experiments ──
    ax = axes[1]
    for name, r in valid:
        if r['n'] < 20: continue
        color = grp_colors.get(r['grp'], '#999')
        marker = '★' if r['sharpe'] > baseline_sharpe else 'o'
        size = max(20, min(200, r['n'] / 2))
        ax.scatter(r['wr'] * 100, r['avg'], s=size, color=color, alpha=0.7,
                   edgecolors='white', linewidth=0.5)
        if r['sharpe'] > baseline_sharpe or r['n'] > 300:
            ax.annotate(name, (r['wr']*100, r['avg']), fontsize=6,
                        xytext=(3, 3), textcoords='offset points')

    ax.axhline(0, color='#333', lw=0.8)
    ax.set_xlabel('Win Rate (%)')
    ax.set_ylabel('Avg T+2 PnL (%)')
    ax.set_title('Avg PnL vs Win Rate (size = trade count)')

    # legend for groups
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
                              markersize=8, label=g)
                       for g, c in grp_colors.items() if g != 'BASELINE']
    ax.legend(handles=legend_elements, fontsize=7, loc='best')

    plt.tight_layout()
    plt.savefig('agent1_subuniverse_equity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Chart saved: agent1_subuniverse_equity.png")

    # ── Figure 3: Best per group bar chart ──
    groups = OrderedDict()
    for name, r in results.items():
        grp = r['grp']
        if grp not in groups: groups[grp] = []
        groups[grp].append((name, r))

    best_per_grp = []
    for grp, items in groups.items():
        best = max(items, key=lambda x: x[1]['sharpe'])
        best_per_grp.append(best)

    fig3, ax = plt.subplots(figsize=(14, 6))
    names_b = [b[0] for b in best_per_grp]
    sharpes_b = [b[1]['sharpe'] for b in best_per_grp]
    colors_b = [grp_colors.get(b[1]['grp'], '#999') for b in best_per_grp]

    bars = ax.bar(range(len(best_per_grp)), sharpes_b, color=colors_b, alpha=0.8,
                  edgecolor='white')
    ax.axhline(baseline_sharpe, color='#e74c3c', ls='--', lw=2,
               label=f'Hardcoded baseline ({baseline_sharpe:.3f})')
    ax.set_xticks(range(len(best_per_grp)))
    ax.set_xticklabels(names_b, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Best Sub-Universe per Group vs Baseline')
    ax.legend()
    for b, s, item in zip(bars, sharpes_b, best_per_grp):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f'{s:.3f}\nn={item[1]["n"]}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig('agent1_subuniverse_groups.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Chart saved: agent1_subuniverse_groups.png")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

all_results = run_experiments()
