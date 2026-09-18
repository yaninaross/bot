# %% Agent 3 — E1: Hold Period Grid + Capital Efficiency
#
# Part 1: A6 — N×Hold grid: N = {1,2,3,4,5} × Hold = {EOD,T+1,...,T+5}
# Part 2: C1 — Hold grid:   Hold = {EOD,T+1,...,T+5} (all qualifying, no top-N)
# Part 3: Capital efficiency — pool-adjusted $ for both strategies
#         Hold EOD → 1 pool → $100K/day
#         Hold T+2 → 3 pools → $33K/day
#         Hold T+5 → 6 pools → $17K/day
#
# LONG strategies (pension account):
#   A6: gap-down sustained rise, local_low entry, LONG PnL
#   C1: multi-day reversal, open entry, LONG PnL
#
# Colab-cell friendly. ~10-15 min runtime (API-bound).

import os, sys, time, re, copy, asyncio
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

# A6 config
A6_GAP_MIN_PCT   = 5.0
A6_PRICE_MIN     = 10.0
A6_TOP_N         = 5
A6_PRE_FILTER_N  = 15
A6_HOLD_BASELINE = 2       # baseline is T+2

# C1 config
C1_GAP_MIN_PCT     = 3.0
C1_GAP_MAX_PCT     = 50.0
C1_PRICE_MIN       = 20.0
C1_DECLINE_5D_MIN  = 0.15
C1_HOLD_BASELINE   = 3     # baseline is T+3

# Entry config (A6 local low)
LOW_WINDOW      = 3
MAX_LOW_BAR     = 12
FALLBACK_BAR    = 15
SKIP_BOUNCE_PCT = 0.15

FLOAT_MIN       = 10_000_000
CAPITAL         = 100_000
TOTAL_POOL      = 100_000

# Grid dimensions
N_VALUES  = [1, 2, 3, 4, 5]
HOLD_DAYS = [0, 1, 2, 3, 4, 5]    # 0 = EOD (T+0 close)
MAX_HOLD  = max(HOLD_DAYS)

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')
ET = pytz.timezone('America/New_York')

_EXCLUDE_TK = {'SPY','QQQ','IWM','DIA','VXX','UVXY','SQQQ','TQQQ','SPXU','SPXL',
               'SDOW','UDOW','SDS','SSO','SH','LABU','LABD','SOXL','SOXS',
               'FNGU','FNGD','NUGT','DUST','JNUG','JDST','TNA','TZA',
               'UPRO','VIXY','SVXY','TVIX'}
_EXCLUDE_SUFFIX = ('W', 'WS', 'U', 'R')

_NON_CS_TYPES = {'ETF', 'ETV', 'ETN', 'ETP', 'ETS', 'ADRC', 'FUND', 'PFD'}

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
_FLOAT_SET = {tk for tk, fs in _FLOAT.items() if fs >= FLOAT_MIN}


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
# ASYNC MINUTE BAR FETCH (RTH only)
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
            if not bars:
                return (tk, dt), None
            rth = []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                if '09:30' <= t_str < '16:00':
                    rth.append((t_str, {
                        'o': float(b['o']), 'h': float(b['h']),
                        'l': float(b['l']), 'c': float(b['c']),
                        'v': int(b['v'])
                    }))
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
# ASYNC TICKER REFERENCE FETCH
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
# GAP-DOWN PATTERN CLASSIFICATION (A6)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_gap_down_pattern(rth_bars):
    if len(rth_bars) < 60:
        return 'no_data'
    open_price = rth_bars[0][1]['o']
    if open_price <= 0:
        return 'no_data'
    max_high_30m = max(b[1]['h'] for b in rth_bars[:30])
    close_bar60 = rth_bars[59][1]['c']
    rise_30m = (max_high_30m - open_price) / abs(open_price)
    recovery_denom = max_high_30m - open_price
    recovery = (close_bar60 - open_price) / recovery_denom if recovery_denom > 0 else 0
    if rise_30m > 0.03 and recovery > 0.50:
        return 'sustained_rise'
    elif rise_30m > 0.03 and recovery <= 0.50:
        return 'rise_then_drop'
    elif rise_30m <= 0.01:
        return 'continued_drop'
    else:
        return 'mild_rise'


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL LOW DETECTION (A6 entry)
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_low(bars, window=LOW_WINDOW, max_low_bar=MAX_LOW_BAR,
                   fallback_bar=FALLBACK_BAR):
    if len(bars) < window * 2 + 1:
        return None
    open_price = bars[0][1]['o']
    for i in range(window, min(max_low_bar, len(bars) - window)):
        cl = bars[i][1]['l']
        if all(bars[i - j][1]['l'] > cl and bars[i + j][1]['l'] > cl
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(bars):
                return {
                    'entry_price': bars[ci][1]['c'],
                    'entry_time': bars[ci][0],
                    'entry_type': 'local_low',
                    'low_bar': i,
                }
    if fallback_bar is not None and fallback_bar < len(bars):
        fb_price = bars[fallback_bar][1]['c']
        if open_price > 0 and (fb_price - open_price) / open_price > SKIP_BOUNCE_PCT:
            return None
        return {
            'entry_price': fb_price,
            'entry_time': bars[fallback_bar][0],
            'entry_type': 'fallback',
            'low_bar': None,
        }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

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


def _pnl_key(h):
    return 'pnl_eod' if h == 0 else f'pnl_t{h}'

def _hold_label(h):
    return 'EOD' if h == 0 else f'T+{h}'


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  BACKTEST ENGINE
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest():
    t_total = time.time()
    timings = {}

    print(f"\n{'='*80}")
    print(f"  AGENT 3 — E1: HOLD PERIOD GRID (LONG)")
    print(f"  {BACKTEST_START} -> {BACKTEST_END}  |  OOS cutoff: {OOS_CUTOFF}")
    print(f"  Universe: _FLOAT ({len(_FLOAT_SET)} tickers, float >= 10M)")
    print(f"  A6: gap-down >= {A6_GAP_MIN_PCT}%, sustained_rise, local_low ONLY")
    print(f"  C1: 5d decline >= {C1_DECLINE_5D_MIN*100:.0f}%, gap-up {C1_GAP_MIN_PCT}%-{C1_GAP_MAX_PCT}%")
    print(f"  Grid: N = {N_VALUES} × Hold = {{{', '.join(_hold_label(h) for h in HOLD_DAYS)}}}")
    print(f"  Capital: ${CAPITAL:,.0f}/day, pool-adjusted by hold period")
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

    # ── Lookback dates ──
    lookback_start = _nth_prior_td(BACKTEST_START, 8)
    lb = lookback_start
    lookback_dates = []
    while lb < datetime.strptime(BACKTEST_START, '%Y-%m-%d').date():
        if _is_td(lb): lookback_dates.append(lb.strftime('%Y-%m-%d'))
        lb += timedelta(days=1)

    # ── Phase 1: Fetch grouped daily ──
    t1 = time.time()
    all_fetch_dates = sorted(set(lookback_dates + dates))
    print(f"\n  Phase 1: Fetching grouped daily for {len(all_fetch_dates)} dates...")
    for i, dt in enumerate(all_fetch_dates):
        fetch_grouped_daily(dt)
        if (i+1) % 30 == 0 or i == len(all_fetch_dates) - 1:
            print(f"    [{i+1:>3}/{len(all_fetch_dates)}] {dt}  |  {time.time()-t_total:.0f}s")
    timings['daily'] = time.time() - t1

    # Outcome dates — need up to T+5
    outcome_dates = set()
    for dt in dates:
        for nd in _next_n_td(dt, MAX_HOLD):
            outcome_dates.add(nd.strftime('%Y-%m-%d'))
    for od in sorted(outcome_dates):
        if od not in _gd_cache:
            try: fetch_grouped_daily(od)
            except: pass
    print(f"    + {len(outcome_dates)} outcome dates fetched")
    timings['outcomes'] = time.time() - t1 - timings['daily']

    # ── Phase 2: Scan A6 and C1 candidates ──
    t2 = time.time()
    print(f"\n  Phase 2: Scanning candidates...")
    a6_candidates = {}
    c1_candidates = {}
    total_a6 = 0
    total_c1 = 0

    gap_min_a6 = A6_GAP_MIN_PCT / 100.0
    gap_min_c1 = C1_GAP_MIN_PCT / 100.0

    for i, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = _gd_cache.get(prior_str, {})
        today = _gd_cache.get(dt, {})
        nxt = _next_n_td(dt, MAX_HOLD)

        # ── A6: Gap-DOWN ──
        a6_day = []
        for tk in _FLOAT_SET:
            if tk in _EXCLUDE_TK: continue
            if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
            if tk not in prior or tk not in today: continue
            prev_c = prior[tk]['c']
            if prev_c <= 0: continue
            cur_o = today[tk]['o']
            if cur_o <= 0 or cur_o < A6_PRICE_MIN: continue
            gap = (cur_o - prev_c) / prev_c
            if gap > -gap_min_a6: continue
            a6_day.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_o, 'scan_date': dt,
                'volume_today': today[tk]['v'],
                't_dates': [nd.strftime('%Y-%m-%d') for nd in nxt],
            })
        a6_day.sort(key=lambda x: x['gap_pct'])
        a6_candidates[dt] = a6_day
        total_a6 += len(a6_day)

        # ── C1: Gap-UP + prior 5d decline ──
        c1_day = []
        for tk in _FLOAT_SET:
            if tk in _EXCLUDE_TK: continue
            if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
            if tk not in prior or tk not in today: continue
            prev_c = prior[tk]['c']
            if prev_c <= 0: continue
            cur_o = today[tk]['o']
            if cur_o <= 0 or cur_o < C1_PRICE_MIN: continue
            gap = (cur_o - prev_c) / prev_c
            if gap < gap_min_c1: continue
            if gap > C1_GAP_MAX_PCT / 100.0: continue
            try:
                close_5d_ago_date = _nth_prior_td(prior_str, 4).strftime('%Y-%m-%d')
            except ValueError:
                continue
            close_5d_ago_data = _gd_cache.get(close_5d_ago_date, {}).get(tk)
            if not close_5d_ago_data or close_5d_ago_data['c'] <= 0:
                continue
            prior_5d_return = (prev_c - close_5d_ago_data['c']) / close_5d_ago_data['c']
            if prior_5d_return > -C1_DECLINE_5D_MIN:
                continue
            c1_day.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_o, 'scan_date': dt,
                'volume_today': today[tk]['v'],
                'prior_5d_return': prior_5d_return,
                't_dates': [nd.strftime('%Y-%m-%d') for nd in nxt],
            })
        c1_day.sort(key=lambda x: -x['gap_pct'])
        c1_candidates[dt] = c1_day
        total_c1 += len(c1_day)

        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  A6={total_a6}, C1={total_c1}  |  {time.time()-t_total:.0f}s")
    timings['scan'] = time.time() - t2

    # ── Phase 3: Fetch minute bars for A6 ──
    t3 = time.time()
    a6_pairs = set()
    for dt, cands in a6_candidates.items():
        for c in cands[:A6_PRE_FILTER_N]:
            a6_pairs.add((c['ticker'], dt))

    print(f"\n  Phase 3: Fetching 1-min bars for {len(a6_pairs)} A6 pairs...")
    pairs_list = list(a6_pairs)
    all_bars = {}
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start + 100]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        done = min(c_start + 100, len(pairs_list))
        if done % 200 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    timings['bars'] = time.time() - t3
    print(f"  -> Got bars for {len(all_bars)}/{len(a6_pairs)} pairs")

    # ── Phase 4: Fetch ticker references ──
    t4 = time.time()
    all_tickers = set()
    for cands in a6_candidates.values():
        for c in cands[:A6_PRE_FILTER_N]:
            all_tickers.add(c['ticker'])
    for cands in c1_candidates.values():
        for c in cands:
            all_tickers.add(c['ticker'])
    print(f"\n  Phase 4: Fetching reference data for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    for c_start in range(0, len(tk_list), 200):
        chunk = tk_list[c_start:c_start + 200]
        fetch_ticker_refs(chunk)
        done = min(c_start + 200, len(tk_list))
        if done % 400 == 0 or done == len(tk_list):
            print(f"    {done}/{len(tk_list)}  |  {time.time()-t_total:.0f}s")
    timings['refs'] = time.time() - t4

    # ── Phase 5: Classify A6 patterns + find entries ──
    t5 = time.time()
    pattern_cache = {}
    entry_cache = {}
    for key, rth_bars in all_bars.items():
        pattern_cache[key] = classify_gap_down_pattern(rth_bars)
        if rth_bars and len(rth_bars) >= LOW_WINDOW * 2 + 1:
            entry_cache[key] = find_local_low(rth_bars)
        else:
            entry_cache[key] = None
    timings['classify'] = time.time() - t5

    # ── Phase 6: Build A6 qualified trades (full list per day for N-grid) ──
    t6 = time.time()
    print(f"\n  Phase 6: Building trade lists...")

    a6_daily = {}
    a6_total = 0

    for dt in dates:
        cands = a6_candidates.get(dt, [])
        qualified = []
        for c in cands[:A6_PRE_FILTER_N]:
            tk = c['ticker']
            key = (tk, dt)
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS': continue
            pat = pattern_cache.get(key, 'no_data')
            if pat != 'sustained_rise': continue
            entry = entry_cache.get(key)
            if entry is None: continue
            if entry['entry_type'] != 'local_low': continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = entry['entry_price']
            tc['entry_time'] = entry['entry_time']
            tc['entry_type'] = entry['entry_type']
            tc['open_pattern'] = pat
            tc['asset_type'] = at
            tc['strategy'] = 'A6'

            ep = tc['entry_price']
            # LONG PnL: (exit - entry) / entry
            # EOD = T+0 close
            t0_data = _gd_cache.get(dt, {}).get(tk, {})
            tc['pnl_eod'] = round((t0_data['c'] - ep) / ep * 100, 3) if t0_data.get('c') else None

            for hi, td_str in enumerate(tc['t_dates'], 1):
                td_data = _gd_cache.get(td_str, {}).get(tk, {})
                if td_data.get('c'):
                    tc[f'pnl_t{hi}'] = round((td_data['c'] - ep) / ep * 100, 3)
                else:
                    tc[f'pnl_t{hi}'] = None

            qualified.append(tc)

        a6_daily[dt] = qualified
        a6_total += len(qualified)

    # ── Phase 7: Build C1 trades (all qualifying — no top-N) ──
    c1_daily = {}
    c1_total = 0

    for dt in dates:
        cands = c1_candidates.get(dt, [])
        qualified = []
        for c in cands:
            tk = c['ticker']
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS': continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = tc['open_price']
            tc['entry_time'] = '09:30'
            tc['entry_type'] = 'open'
            tc['open_pattern'] = 'c1_reversal'
            tc['asset_type'] = at
            tc['strategy'] = 'C1'

            ep = tc['entry_price']
            # EOD = T+0 close
            t0_data = _gd_cache.get(dt, {}).get(tk, {})
            tc['pnl_eod'] = round((t0_data['c'] - ep) / ep * 100, 3) if t0_data.get('c') else None

            for hi, td_str in enumerate(tc['t_dates'], 1):
                td_data = _gd_cache.get(td_str, {}).get(tk, {})
                if td_data.get('c'):
                    tc[f'pnl_t{hi}'] = round((td_data['c'] - ep) / ep * 100, 3)
                else:
                    tc[f'pnl_t{hi}'] = None

            qualified.append(tc)

        c1_daily[dt] = qualified
        c1_total += len(qualified)

    timings['build'] = time.time() - t6
    elapsed = time.time() - t_total

    print(f"\n  -> A6: {a6_total} qualified trades (for N×Hold grid)")
    print(f"  -> C1: {c1_total} qualifying trades (for Hold grid)")

    print(f"\n  {'─'*50}")
    for step, dur in [('Daily+Outcomes', timings['daily']+timings.get('outcomes',0)),
                      ('Scan', timings['scan']), ('Bars', timings['bars']),
                      ('Refs', timings['refs']), ('Classify', timings['classify']),
                      ('Build', timings['build'])]:
        print(f"    {step:<16s}  {dur:>6.0f}s  ({dur/elapsed*100:>4.0f}%)")
    print(f"  TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return a6_daily, c1_daily, dates


# ═══════════════════════════════════════════════════════════════════════════════
# A6 GRID ANALYSIS — N × Hold
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_a6_grid(a6_daily, dates):
    grid = {}
    for n_val in N_VALUES:
        for hold in HOLD_DAYS:
            pk = _pnl_key(hold)
            all_trades = []
            daily_pnl = defaultdict(list)

            for dt in dates:
                day_trades = a6_daily.get(dt, [])[:n_val]
                for t in day_trades:
                    p = t.get(pk)
                    if p is not None:
                        all_trades.append(t)
                        daily_pnl[dt].append(p)

            pnls = [t[pk] for t in all_trades]
            m = compute_metrics(pnls, f'N={n_val} {_hold_label(hold)}')

            active_days = len(daily_pnl)
            tpd = len(all_trades) / active_days if active_days > 0 else 0

            # Raw $ (unlimited capital)
            cum_raw = 0.0
            for dt in dates:
                dp = daily_pnl.get(dt, [])
                if dp:
                    pos_size = CAPITAL / len(dp)
                    cum_raw += sum(pos_size * (r / 100) for r in dp)

            # Pool-adjusted $
            slots = hold + 1 if hold > 0 else 1
            pool_daily = TOTAL_POOL / slots
            cum_adj = 0.0
            for dt in dates:
                dp = daily_pnl.get(dt, [])
                if dp:
                    pos_size = pool_daily / len(dp)
                    cum_adj += sum(pos_size * (r / 100) for r in dp)

            # OOS split
            is_pnls = [t[pk] for t in all_trades
                       if t.get(pk) is not None and t['scan_date'] < OOS_CUTOFF]
            oos_pnls = [t[pk] for t in all_trades
                        if t.get(pk) is not None and t['scan_date'] >= OOS_CUTOFF]
            is_m = compute_metrics(is_pnls, f'IS')
            oos_m = compute_metrics(oos_pnls, f'OOS')

            grid[(n_val, hold)] = {
                'metrics': m, 'tpd': tpd,
                'cum_raw': cum_raw, 'cum_adj': cum_adj,
                'pool_daily': pool_daily, 'slots': slots,
                'ci_confirmed': m['ci_lo'] > 0,
                'active_days': active_days,
                'daily_pnl': dict(daily_pnl),
                'trades': all_trades,
                'is_metrics': is_m, 'oos_metrics': oos_m,
            }
    return grid


# ═══════════════════════════════════════════════════════════════════════════════
# C1 GRID ANALYSIS — Hold only (all qualifying trades, no N)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_c1_grid(c1_daily, dates):
    grid = {}
    for hold in HOLD_DAYS:
        pk = _pnl_key(hold)
        all_trades = []
        daily_pnl = defaultdict(list)

        for dt in dates:
            for t in c1_daily.get(dt, []):
                p = t.get(pk)
                if p is not None:
                    all_trades.append(t)
                    daily_pnl[dt].append(p)

        pnls = [t[pk] for t in all_trades]
        m = compute_metrics(pnls, f'C1 {_hold_label(hold)}')

        active_days = len(daily_pnl)
        tpd = len(all_trades) / active_days if active_days > 0 else 0

        cum_raw = 0.0
        for dt in dates:
            dp = daily_pnl.get(dt, [])
            if dp:
                pos_size = CAPITAL / len(dp)
                cum_raw += sum(pos_size * (r / 100) for r in dp)

        slots = hold + 1 if hold > 0 else 1
        pool_daily = TOTAL_POOL / slots
        cum_adj = 0.0
        for dt in dates:
            dp = daily_pnl.get(dt, [])
            if dp:
                pos_size = pool_daily / len(dp)
                cum_adj += sum(pos_size * (r / 100) for r in dp)

        is_pnls = [t[pk] for t in all_trades
                   if t.get(pk) is not None and t['scan_date'] < OOS_CUTOFF]
        oos_pnls = [t[pk] for t in all_trades
                    if t.get(pk) is not None and t['scan_date'] >= OOS_CUTOFF]
        is_m = compute_metrics(is_pnls, f'IS')
        oos_m = compute_metrics(oos_pnls, f'OOS')

        grid[hold] = {
            'metrics': m, 'tpd': tpd,
            'cum_raw': cum_raw, 'cum_adj': cum_adj,
            'pool_daily': pool_daily, 'slots': slots,
            'ci_confirmed': m['ci_lo'] > 0,
            'active_days': active_days,
            'daily_pnl': dict(daily_pnl),
            'trades': all_trades,
            'is_metrics': is_m, 'oos_metrics': oos_m,
        }
    return grid


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def print_a6_grid(a6_grid, dates):
    print(f"\n\n{'='*100}")
    print(f"  A6 — N×HOLD GRID (LONG: gap-down sustained rise)")
    print(f"  {dates[0]} -> {dates[-1]}  |  ${CAPITAL:,.0f}/day  |  Pool: ${TOTAL_POOL:,.0f}")
    print(f"  Baseline: N=5 T+2  |  LONG PnL: (exit - entry) / entry")
    print(f"{'='*100}")

    hl = [_hold_label(h) for h in HOLD_DAYS]

    def _tbl(title, fn):
        print(f"\n  ── {title} ──")
        headers = [''] + hl
        rows = [[f'N={n}'] + [fn(a6_grid[(n, h)]) for h in HOLD_DAYS]
                for n in N_VALUES]
        print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right'))

    _tbl('SHARPE RATIO',       lambda c: f"{c['metrics']['sharpe']:.3f}")
    _tbl('AVG PnL (%)',        lambda c: f"{c['metrics']['avg']:+.2f}%")
    _tbl('MEDIAN PnL (%)',     lambda c: f"{c['metrics']['med']:+.2f}%")
    _tbl('TRIMMED MEAN (%)',   lambda c: f"{c['metrics']['trimmed_mean']:+.2f}%")
    _tbl('WIN RATE',           lambda c: f"{c['metrics']['wr']*100:.0f}%")
    _tbl('PROFIT FACTOR',      lambda c: f"{c['metrics']['pf']:.2f}")
    _tbl('TRADE COUNT',        lambda c: f"{c['metrics']['n']}")
    _tbl('TRADES/DAY',         lambda c: f"{c['tpd']:.2f}")
    _tbl('95% CI',             lambda c: f"[{c['metrics']['ci_lo']:+.1f},{c['metrics']['ci_hi']:+.1f}]")
    _tbl('CI CONFIRMED',       lambda c: '  ✓' if c['ci_confirmed'] else '  ✗')
    _tbl(f'RAW $ (${CAPITAL/1000:.0f}K/day)',   lambda c: f"${c['cum_raw']:+,.0f}")
    _tbl(f'POOL $ (${TOTAL_POOL/1000:.0f}K pool)', lambda c: f"${c['cum_adj']:+,.0f}")
    _tbl('POOL ALLOC/DAY',     lambda c: f"${c['pool_daily']:,.0f}")

    # OOS validation
    _tbl('IS AVG (%)',         lambda c: f"{c['is_metrics']['avg']:+.2f}%")
    _tbl('OOS AVG (%)',        lambda c: f"{c['oos_metrics']['avg']:+.2f}%")
    _tbl('IS SHARPE',          lambda c: f"{c['is_metrics']['sharpe']:.3f}")
    _tbl('OOS SHARPE',         lambda c: f"{c['oos_metrics']['sharpe']:.3f}")

    # Best cells
    best_sharpe = max(a6_grid, key=lambda k: a6_grid[k]['metrics']['sharpe'])
    best_raw = max(a6_grid, key=lambda k: a6_grid[k]['cum_raw'])
    best_pool = max(a6_grid, key=lambda k: a6_grid[k]['cum_adj'])

    print(f"\n  {'═'*80}")
    print(f"  A6 OPTIMAL HOLD PERIOD")
    print(f"  {'═'*80}")

    for label, key in [('Best Sharpe', best_sharpe),
                       ('Best Raw $', best_raw),
                       ('Best Pool $ (FIXED CAPITAL)', best_pool)]:
        c = a6_grid[key]
        m = c['metrics']
        print(f"\n  {label}: N={key[0]}, {_hold_label(key[1])}")
        print(f"    Sharpe={m['sharpe']:.3f}  avg={m['avg']:+.2f}%  WR={m['wr']*100:.0f}%  n={m['n']}")
        print(f"    Pool $: ${c['cum_adj']:+,.0f}  |  Raw $: ${c['cum_raw']:+,.0f}")
        print(f"    IS avg={c['is_metrics']['avg']:+.2f}%  OOS avg={c['oos_metrics']['avg']:+.2f}%")

    # Baseline reference
    ref = a6_grid.get((5, A6_HOLD_BASELINE))
    if ref:
        m = ref['metrics']
        print(f"\n  Baseline (N=5 T+{A6_HOLD_BASELINE}): Sharpe={m['sharpe']:.3f}  "
              f"avg={m['avg']:+.2f}%  n={m['n']}  ${ref['cum_raw']:+,.0f}")

    # Hold period trends
    print(f"\n  ── HOLD PERIOD TREND (Sharpe, per N) ──")
    for n in N_VALUES:
        sharpes = [a6_grid[(n, h)]['metrics']['sharpe'] for h in HOLD_DAYS]
        trend = '→'.join(f'{s:.3f}' for s in sharpes)
        best_h = HOLD_DAYS[np.argmax(sharpes)]
        print(f"    N={n}: {trend}  (best: {_hold_label(best_h)})")

    print(f"\n  ── N TREND (Sharpe, per hold) ──")
    for h in HOLD_DAYS:
        sharpes = [a6_grid[(n, h)]['metrics']['sharpe'] for n in N_VALUES]
        trend = '→'.join(f'{s:.3f}' for s in sharpes)
        best_n = N_VALUES[np.argmax(sharpes)]
        print(f"    {_hold_label(h):>4s}: {trend}  (best: N={best_n})")

    # Top 5 cells
    print(f"\n  ── TOP 5 CELLS BY SHARPE ──")
    ranked = sorted(a6_grid.items(), key=lambda x: -x[1]['metrics']['sharpe'])
    for i, ((n, h), c) in enumerate(ranked[:5]):
        m = c['metrics']
        ci = '✓' if c['ci_confirmed'] else '✗'
        print(f"    #{i+1}  N={n} {_hold_label(h):>4s}  Sh={m['sharpe']:.3f}  "
              f"avg={m['avg']:+.2f}%  WR={m['wr']*100:.0f}%  n={m['n']}  "
              f"pool$={c['cum_adj']:+,.0f}  CI {ci}")

    # Worst 5 trades at baseline
    ref_trades = sorted(a6_grid.get((5, A6_HOLD_BASELINE), {}).get('trades', []),
                        key=lambda t: t.get(f'pnl_t{A6_HOLD_BASELINE}', 0))
    pk = f'pnl_t{A6_HOLD_BASELINE}'
    if ref_trades:
        print(f"\n  ── WORST 5 TRADES — N=5 T+{A6_HOLD_BASELINE} ──")
        for t in ref_trades[:5]:
            print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={t['gap_pct']*100:>+7.1f}%  pnl={t.get(pk,0):>+8.2f}%")

    print()
    return best_sharpe, best_pool


def print_c1_grid(c1_grid, dates):
    print(f"\n\n{'='*100}")
    print(f"  C1 — HOLD PERIOD GRID (LONG: multi-day reversal)")
    print(f"  {dates[0]} -> {dates[-1]}  |  ${CAPITAL:,.0f}/day  |  Pool: ${TOTAL_POOL:,.0f}")
    print(f"  Baseline: T+3  |  All qualifying trades  |  LONG PnL")
    print(f"{'='*100}")

    hl = [_hold_label(h) for h in HOLD_DAYS]
    headers = ['Metric'] + hl

    def _row(name, fn):
        return [name] + [fn(c1_grid[h]) for h in HOLD_DAYS]

    tbl = [
        _row('Trades',        lambda c: f"{c['metrics']['n']}"),
        _row('Trades/day',    lambda c: f"{c['tpd']:.2f}"),
        _row('Avg PnL',       lambda c: f"{c['metrics']['avg']:+.2f}%"),
        _row('Median PnL',    lambda c: f"{c['metrics']['med']:+.2f}%"),
        _row('Trimmed Mean',  lambda c: f"{c['metrics']['trimmed_mean']:+.2f}%"),
        _row('Win Rate',      lambda c: f"{c['metrics']['wr']*100:.0f}%"),
        _row('Sharpe',        lambda c: f"{c['metrics']['sharpe']:.3f}"),
        _row('Profit Factor', lambda c: f"{c['metrics']['pf']:.2f}"),
        _row('Max Loss',      lambda c: f"{c['metrics']['max_loss']:+.2f}%"),
        _row('Max Gain',      lambda c: f"{c['metrics']['max_gain']:+.2f}%"),
        _row('95% CI',        lambda c: f"[{c['metrics']['ci_lo']:+.1f},{c['metrics']['ci_hi']:+.1f}]"),
        _row('CI > 0',        lambda c: '✓' if c['ci_confirmed'] else '✗'),
        _row(f'Raw $ ({CAPITAL/1000:.0f}K)',  lambda c: f"${c['cum_raw']:+,.0f}"),
        _row(f'Pool $ ({TOTAL_POOL/1000:.0f}K)', lambda c: f"${c['cum_adj']:+,.0f}"),
        _row('Pool alloc/day', lambda c: f"${c['pool_daily']:,.0f}"),
        _row('IS avg',        lambda c: f"{c['is_metrics']['avg']:+.2f}%"),
        _row('OOS avg',       lambda c: f"{c['oos_metrics']['avg']:+.2f}%"),
        _row('IS Sharpe',     lambda c: f"{c['is_metrics']['sharpe']:.3f}"),
        _row('OOS Sharpe',    lambda c: f"{c['oos_metrics']['sharpe']:.3f}"),
        _row('IS n',          lambda c: f"{c['is_metrics']['n']}"),
        _row('OOS n',         lambda c: f"{c['oos_metrics']['n']}"),
    ]
    print()
    print(tabulate(tbl, headers=headers, tablefmt='simple', stralign='right'))

    best_sharpe_h = max(HOLD_DAYS, key=lambda h: c1_grid[h]['metrics']['sharpe'])
    best_raw_h = max(HOLD_DAYS, key=lambda h: c1_grid[h]['cum_raw'])
    best_pool_h = max(HOLD_DAYS, key=lambda h: c1_grid[h]['cum_adj'])

    print(f"\n  {'═'*80}")
    print(f"  C1 OPTIMAL HOLD PERIOD")
    print(f"  {'═'*80}")

    for label, h in [('Best Sharpe', best_sharpe_h),
                     ('Best Raw $', best_raw_h),
                     ('Best Pool $ (FIXED CAPITAL)', best_pool_h)]:
        c = c1_grid[h]
        m = c['metrics']
        print(f"\n  {label}: {_hold_label(h)}")
        print(f"    Sharpe={m['sharpe']:.3f}  avg={m['avg']:+.2f}%  WR={m['wr']*100:.0f}%  n={m['n']}")
        print(f"    Pool $: ${c['cum_adj']:+,.0f}  |  Raw $: ${c['cum_raw']:+,.0f}")
        print(f"    IS avg={c['is_metrics']['avg']:+.2f}%  OOS avg={c['oos_metrics']['avg']:+.2f}%")

    # Baseline reference
    ref = c1_grid.get(C1_HOLD_BASELINE)
    if ref:
        m = ref['metrics']
        print(f"\n  Baseline (T+{C1_HOLD_BASELINE}): Sharpe={m['sharpe']:.3f}  "
              f"avg={m['avg']:+.2f}%  n={m['n']}  ${ref['cum_raw']:+,.0f}")

    # Hold period trend
    print(f"\n  ── HOLD PERIOD TREND ──")
    for metric_name, fn in [('Sharpe', lambda c: c['metrics']['sharpe']),
                             ('Avg PnL', lambda c: c['metrics']['avg']),
                             ('WR', lambda c: c['metrics']['wr']*100),
                             ('Pool $', lambda c: c['cum_adj'])]:
        vals = [fn(c1_grid[h]) for h in HOLD_DAYS]
        if 'PnL' in metric_name or 'Pool' in metric_name:
            trend = '→'.join(f'{v:+.1f}' for v in vals)
        else:
            trend = '→'.join(f'{v:.3f}' if isinstance(v, float) and v < 10 else f'{v:.0f}' for v in vals)
        best_idx = np.argmax(vals)
        print(f"    {metric_name:>8s}: {trend}  (best: {_hold_label(HOLD_DAYS[best_idx])})")

    # Worst 5 trades at baseline
    pk = _pnl_key(C1_HOLD_BASELINE)
    ref_trades = sorted(c1_grid.get(C1_HOLD_BASELINE, {}).get('trades', []),
                        key=lambda t: t.get(pk, 0))
    if ref_trades:
        print(f"\n  ── WORST 5 TRADES — T+{C1_HOLD_BASELINE} ──")
        for t in ref_trades[:5]:
            pnl = t.get(pk, 0)
            p5d = t.get('prior_5d_return', 0)
            print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={t['gap_pct']*100:>+7.1f}%  "
                  f"5d_dec={p5d*100:+.1f}%  pnl={pnl:>+8.2f}%")

    print()
    return best_sharpe_h, best_pool_h


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_a6_grid(a6_grid, dates, best_sharpe, best_pool):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    # ── Figure 1: Heatmaps ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'A6 — N×Hold Grid (LONG: gap-down sustained rise)\n'
                 f'{dates[0]} → {dates[-1]}  |  Pool: ${TOTAL_POOL/1000:.0f}K',
                 fontsize=12, fontweight='bold')

    panels = [
        ('Sharpe Ratio', lambda c: c['metrics']['sharpe'], '.3f'),
        ('Avg PnL (%)', lambda c: c['metrics']['avg'], '+.1f'),
        (f'Pool $ ($K)', lambda c: c['cum_adj'] / 1000, '+.0f'),
        ('Win Rate (%)', lambda c: c['metrics']['wr'] * 100, '.0f'),
    ]

    for idx, (title, fn, fmt) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        data = np.array([[fn(a6_grid[(n, h)]) for h in HOLD_DAYS] for n in N_VALUES])
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(HOLD_DAYS)))
        ax.set_xticklabels([_hold_label(h) for h in HOLD_DAYS], fontsize=10)
        ax.set_yticks(range(len(N_VALUES)))
        ax.set_yticklabels([f'N={n}' for n in N_VALUES], fontsize=10)
        for i in range(len(N_VALUES)):
            for j in range(len(HOLD_DAYS)):
                val = data[i, j]
                is_best = (N_VALUES[i], HOLD_DAYS[j]) == best_pool
                weight = 'bold' if is_best else 'normal'
                color = 'white' if abs(val - data.mean()) > data.std() * 0.8 else 'black'
                ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                        fontsize=9, fontweight=weight, color=color)
                if is_best:
                    rect = plt.Rectangle((j-0.48, i-0.48), 0.96, 0.96,
                                         linewidth=3, edgecolor='gold', facecolor='none')
                    ax.add_patch(rect)
        ax.set_title(title, fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p1 = os.path.join(BASE_DIR, 'agent3_e1_a6_heatmaps.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")

    # ── Figure 2: Hold period curves ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
    fig.suptitle('A6 — Hold Period Curves', fontsize=12, fontweight='bold')

    for panel_idx, (title, fn, ylabel) in enumerate([
        ('Sharpe', lambda c: c['metrics']['sharpe'], 'Sharpe'),
        ('Avg PnL', lambda c: c['metrics']['avg'], 'Avg PnL (%)'),
        ('Pool $K', lambda c: c['cum_adj']/1000, 'Pool-Adjusted ($K)')
    ]):
        ax = axes[panel_idx]
        for i, n in enumerate(N_VALUES):
            vals = [fn(a6_grid[(n, h)]) for h in HOLD_DAYS]
            ax.plot(range(len(HOLD_DAYS)), vals, 'o-', color=colors[i],
                    linewidth=2, markersize=8, label=f'N={n}')
            best_idx = int(np.argmax(vals))
            ax.plot(best_idx, vals[best_idx], '*', color=colors[i], markersize=16, zorder=5)
        ax.axhline(0, color='#333', lw=0.8, ls=':')
        ax.set_xticks(range(len(HOLD_DAYS)))
        ax.set_xticklabels([_hold_label(h) for h in HOLD_DAYS])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p2 = os.path.join(BASE_DIR, 'agent3_e1_a6_curves.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p2}")


def plot_c1_grid(c1_grid, dates, best_sharpe_h, best_pool_h):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    # ── Figure: C1 hold period curves ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'C1 — Hold Period Grid (LONG: multi-day reversal)\n'
                 f'{dates[0]} → {dates[-1]}  |  Pool: ${TOTAL_POOL/1000:.0f}K',
                 fontsize=12, fontweight='bold')

    for panel_idx, (title, fn, ylabel) in enumerate([
        ('Sharpe', lambda c: c['metrics']['sharpe'], 'Sharpe'),
        ('Avg PnL', lambda c: c['metrics']['avg'], 'Avg PnL (%)'),
        ('Pool $K', lambda c: c['cum_adj']/1000, 'Pool-Adjusted ($K)')
    ]):
        ax = axes[panel_idx]
        vals = [fn(c1_grid[h]) for h in HOLD_DAYS]
        ax.plot(range(len(HOLD_DAYS)), vals, 'o-', color='#2980b9',
                linewidth=2.5, markersize=10, label='C1 all-qualifying')

        # Mark best and baseline
        best_idx = int(np.argmax(vals))
        baseline_idx = HOLD_DAYS.index(C1_HOLD_BASELINE)
        ax.plot(best_idx, vals[best_idx], '*', color='#e74c3c',
                markersize=20, zorder=5, label=f'Best: {_hold_label(HOLD_DAYS[best_idx])}')
        ax.plot(baseline_idx, vals[baseline_idx], 'D', color='#27ae60',
                markersize=12, zorder=5, label=f'Baseline: T+{C1_HOLD_BASELINE}')

        ax.axhline(0, color='#333', lw=0.8, ls=':')
        ax.set_xticks(range(len(HOLD_DAYS)))
        ax.set_xticklabels([_hold_label(h) for h in HOLD_DAYS])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    p1 = os.path.join(BASE_DIR, 'agent3_e1_c1_curves.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")

    # ── IS vs OOS bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('C1 — IS vs OOS by Hold Period', fontsize=12, fontweight='bold')

    x = np.arange(len(HOLD_DAYS))
    w = 0.35

    for panel_idx, (title, fn) in enumerate([
        ('Avg PnL (%)', lambda c: c['avg']),
        ('Sharpe', lambda c: c['sharpe'])
    ]):
        ax = axes[panel_idx]
        is_vals = [fn(c1_grid[h]['is_metrics']) for h in HOLD_DAYS]
        oos_vals = [fn(c1_grid[h]['oos_metrics']) for h in HOLD_DAYS]
        ax.bar(x - w/2, is_vals, w, color='#3498db', alpha=0.8, label='IS')
        ax.bar(x + w/2, oos_vals, w, color='#e67e22', alpha=0.8, label='OOS')
        ax.set_xticks(x)
        ax.set_xticklabels([_hold_label(h) for h in HOLD_DAYS])
        ax.axhline(0, color='#333', lw=0.8)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p2 = os.path.join(BASE_DIR, 'agent3_e1_c1_isoos.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p2}")


def plot_combined_equity(a6_grid, c1_grid, dates):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle('A6 + C1 — Pool-Adjusted Equity Comparison', fontsize=12, fontweight='bold')

    best_a6 = max(a6_grid, key=lambda k: a6_grid[k]['cum_adj'])
    best_c1 = max(HOLD_DAYS, key=lambda h: c1_grid[h]['cum_adj'])

    comparisons = [
        (a6_grid[(5, A6_HOLD_BASELINE)], '#27ae60', f'A6 baseline N=5 T+{A6_HOLD_BASELINE}'),
        (a6_grid[best_a6], '#2ecc71', f'A6 best N={best_a6[0]} {_hold_label(best_a6[1])}'),
        (c1_grid[C1_HOLD_BASELINE], '#2980b9', f'C1 baseline T+{C1_HOLD_BASELINE}'),
        (c1_grid[best_c1], '#e74c3c', f'C1 best {_hold_label(best_c1)}'),
    ]

    for c, color, lbl in comparisons:
        cum = 0.0
        cum_series = [0.0]
        d_dates = [datetime.strptime(dates[0], '%Y-%m-%d')]
        pd_alloc = c['pool_daily']
        for dt in dates:
            dp = c['daily_pnl'].get(dt, [])
            if dp:
                pos_size = pd_alloc / len(dp)
                cum += sum(pos_size * (r / 100) for r in dp)
            cum_series.append(cum)
            d_dates.append(datetime.strptime(dt, '%Y-%m-%d'))
        m = c['metrics']
        ax.plot(d_dates, cum_series, color=color, linewidth=2,
                label=f'{lbl}: ${cum:+,.0f} Sh={m["sharpe"]:.3f}')

    ax.axhline(0, color='#333', lw=0.8, ls=':')
    oos_dt = datetime.strptime(OOS_CUTOFF, '%Y-%m-%d')
    ax.axvline(oos_dt, color='red', ls=':', lw=1, alpha=0.5, label='OOS cutoff')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(BASE_DIR, 'agent3_e1_combined_equity.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

a6_daily, c1_daily, dates = run_backtest()

a6_grid = analyze_a6_grid(a6_daily, dates)
a6_best_sharpe, a6_best_pool = print_a6_grid(a6_grid, dates)
plot_a6_grid(a6_grid, dates, a6_best_sharpe, a6_best_pool)

c1_grid = analyze_c1_grid(c1_daily, dates)
c1_best_sharpe_h, c1_best_pool_h = print_c1_grid(c1_grid, dates)
plot_c1_grid(c1_grid, dates, c1_best_sharpe_h, c1_best_pool_h)

plot_combined_equity(a6_grid, c1_grid, dates)
