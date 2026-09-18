# %% Agent 3 — Baseline Backtest
#
# Two LONG strategies (pension account, no shorting, no leverage):
#   BASELINE-A6: Gap-down sustained rise — buy local low, hold T+1
#   BASELINE-C1: Multi-day reversal — prior 5d decline + gap-up, hold T+3
#   COMBINED:    A6 + C1 portfolio (uncorrelated)
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

BACKTEST_START = '2026-09-12'
BACKTEST_END   = '2026-09-18'

# A6 config
A6_GAP_MIN_PCT   = 5.0      # gap-DOWN >= 5%
A6_PRICE_MIN     = 10.0
A6_TOP_N         = 5         # top 5 by abs(gap_pct)
A6_PRE_FILTER_N  = 15        # scan top 15 before filtering
A6_HOLD_DAYS     = 1         # T+1 (E1/E1b: Sharpe 0.600 vs T+2 0.458)

# C1 config
C1_GAP_MIN_PCT     = 3.0    # gap-UP >= 3%
C1_GAP_MAX_PCT     = 50.0   # gap-UP cap (exclude splits/corp actions)
C1_PRICE_MIN       = 20.0
C1_DECLINE_5D_MIN  = 0.15   # prior 5d decline >= 15%
C1_HOLD_DAYS       = 3      # T+3

# Entry config (A6 local low detection)
LOW_WINDOW     = 3
MAX_LOW_BAR    = 12
FALLBACK_BAR   = 15
SKIP_BOUNCE_PCT = 0.15      # skip if price up >15% from open at fallback

FLOAT_MIN      = 10_000_000
CAPITAL        = 100_000

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
# LOCAL LOW DETECTION (A6 entry — mirror of Agent 1's find_local_high)
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
    print(f"  AGENT 3 — BASELINE BACKTEST (LONG)")
    print(f"  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"  Universe: _FLOAT ({len(_FLOAT_SET)} tickers with float >= 10M, from {len(_FLOAT)} total)")
    print(f"  BASELINE-A6: gap-down >= {A6_GAP_MIN_PCT}%, sustained_rise, local_low ONLY (no fallback), T+{A6_HOLD_DAYS}")
    print(f"  BASELINE-C1: 5d decline >= {C1_DECLINE_5D_MIN*100:.0f}%, gap-up {C1_GAP_MIN_PCT}%-{C1_GAP_MAX_PCT}%, open entry, T+{C1_HOLD_DAYS}")
    print(f"  Capital: ${CAPITAL:,.0f}/day, equal split, non-compounding")
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

    # ── Lookback dates for C1 (need 5d prior closes) ──
    lookback_start = _nth_prior_td(BACKTEST_START, 8)
    lb = lookback_start
    lookback_dates = []
    while lb < datetime.strptime(BACKTEST_START, '%Y-%m-%d').date():
        if _is_td(lb): lookback_dates.append(lb.strftime('%Y-%m-%d'))
        lb += timedelta(days=1)

    # ── Phase 1: Fetch grouped daily for all dates + lookback ──
    t1 = time.time()
    all_fetch_dates = sorted(set(lookback_dates + dates))
    print(f"\n  Phase 1: Fetching grouped daily for {len(all_fetch_dates)} dates...")
    for i, dt in enumerate(all_fetch_dates):
        fetch_grouped_daily(dt)
        if (i+1) % 30 == 0 or i == len(all_fetch_dates) - 1:
            print(f"    [{i+1:>3}/{len(all_fetch_dates)}] {dt}  |  {time.time()-t_total:.0f}s")
    timings['daily'] = time.time() - t1

    # Also fetch outcome dates (T+2 for A6, T+3 for C1)
    outcome_dates = set()
    for dt in dates:
        for nd in _next_n_td(dt, max(A6_HOLD_DAYS, C1_HOLD_DAYS)):
            outcome_dates.add(nd.strftime('%Y-%m-%d'))
    for od in sorted(outcome_dates):
        if od not in _gd_cache:
            try: fetch_grouped_daily(od)
            except: pass
    print(f"    + {len(outcome_dates)} outcome dates fetched")

    # Build trading date index for lookback
    all_td_sorted = sorted(all_fetch_dates + [od for od in outcome_dates if od not in set(all_fetch_dates)])
    td_index = {dt: i for i, dt in enumerate(sorted(set(d for d in all_td_sorted if _is_td(d))))}
    td_list = sorted(td_index.keys())

    # ── Phase 2: Scan A6 and C1 candidates ──
    t2 = time.time()
    print(f"\n  Phase 2: Scanning for A6 (gap-down) and C1 (reversal) candidates...")
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
        nxt_a6 = _next_n_td(dt, A6_HOLD_DAYS)
        nxt_c1 = _next_n_td(dt, C1_HOLD_DAYS)

        # ── A6: Gap-DOWN candidates ──
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
            if gap > -gap_min_a6: continue  # want gap-DOWN
            a6_day.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_o, 'scan_date': dt,
                'volume_today': today[tk]['v'],
                't_dates': [nd.strftime('%Y-%m-%d') for nd in nxt_a6],
            })
        a6_day.sort(key=lambda x: x['gap_pct'])  # most negative first
        a6_candidates[dt] = a6_day
        total_a6 += len(a6_day)

        # ── C1: Gap-UP + prior 5d decline candidates ──
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
            if gap < gap_min_c1: continue  # want gap-UP
            if gap > C1_GAP_MAX_PCT / 100.0: continue  # cap: exclude splits/corp actions

            # Check prior 5d decline
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
                't_dates': [nd.strftime('%Y-%m-%d') for nd in nxt_c1],
            })
        c1_day.sort(key=lambda x: -x['gap_pct'])  # largest gap-up first
        c1_candidates[dt] = c1_day
        total_c1 += len(c1_day)

        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  A6={total_a6}, C1={total_c1}  |  {time.time()-t_total:.0f}s")
    timings['scan'] = time.time() - t2

    # ── Phase 3: Fetch minute bars for A6 candidates ──
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

    # ── Phase 4: Fetch ticker reference data ──
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

    # ── Phase 5: Classify A6 patterns + find local low entries ──
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

    pat_counts = defaultdict(int)
    for p in pattern_cache.values():
        pat_counts[p] += 1
    print(f"\n  Phase 5: Gap-down pattern classification")
    for pat in ['sustained_rise', 'rise_then_drop', 'mild_rise', 'continued_drop', 'no_data']:
        cnt = pat_counts.get(pat, 0)
        pct = cnt / len(pattern_cache) * 100 if pattern_cache else 0
        print(f"    {pat:25s}  {cnt:>5d}  ({pct:.1f}%)")

    # ── Phase 6: Build A6 trades ──
    t6 = time.time()
    print(f"\n  Phase 6: Building trade lists...")
    a6_trades = []
    a6_cnt = {'ll': 0, 'fb': 0, 'ne': 0, 'nb': 0, 'nsr': 0, 'ncs': 0}

    for dt in dates:
        cands = a6_candidates.get(dt, [])
        qualified = []
        for c in cands[:A6_PRE_FILTER_N]:
            tk = c['ticker']
            key = (tk, dt)
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS':
                a6_cnt['ncs'] += 1
                continue
            pat = pattern_cache.get(key, 'no_data')
            if pat != 'sustained_rise':
                a6_cnt['nsr'] += 1
                continue
            entry = entry_cache.get(key)
            if entry is None:
                if key not in all_bars:
                    a6_cnt['nb'] += 1
                else:
                    a6_cnt['ne'] += 1
                continue

            if entry['entry_type'] != 'local_low':
                a6_cnt['fb'] += 1
                continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = entry['entry_price']
            tc['entry_time'] = entry['entry_time']
            tc['entry_type'] = entry['entry_type']
            tc['open_pattern'] = pat
            tc['asset_type'] = at
            tc['strategy'] = 'A6'

            ep = tc['entry_price']
            # LONG PnL: (exit - entry) / entry
            for hi, td_str in enumerate(tc['t_dates'], 1):
                td_data = _gd_cache.get(td_str, {}).get(tk, {})
                if td_data.get('c'):
                    tc[f'pnl_t{hi}'] = round((td_data['c'] - ep) / ep * 100, 3)
                else:
                    tc[f'pnl_t{hi}'] = None

            qualified.append(tc)
            a6_cnt['ll'] += 1

        for tc in qualified[:A6_TOP_N]:
            if tc.get(f'pnl_t{A6_HOLD_DAYS}') is not None:
                a6_trades.append(tc)

    # ── Phase 7: Build C1 trades ──
    c1_trades = []
    c1_cnt = {'ok': 0, 'ncs': 0, 'no_exit': 0}

    for dt in dates:
        cands = c1_candidates.get(dt, [])
        for c in cands:
            tk = c['ticker']
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS':
                c1_cnt['ncs'] += 1
                continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = tc['open_price']
            tc['entry_time'] = '09:30'
            tc['entry_type'] = 'open'
            tc['open_pattern'] = 'c1_reversal'
            tc['asset_type'] = at
            tc['strategy'] = 'C1'

            ep = tc['entry_price']
            has_exit = False
            for hi, td_str in enumerate(tc['t_dates'], 1):
                td_data = _gd_cache.get(td_str, {}).get(tk, {})
                if td_data.get('c'):
                    tc[f'pnl_t{hi}'] = round((td_data['c'] - ep) / ep * 100, 3)
                    if hi == C1_HOLD_DAYS:
                        has_exit = True
                else:
                    tc[f'pnl_t{hi}'] = None

            if has_exit:
                c1_trades.append(tc)
                c1_cnt['ok'] += 1
            else:
                c1_cnt['no_exit'] += 1

    timings['build'] = time.time() - t6
    elapsed = time.time() - t_total

    print(f"\n  -> A6: {len(a6_trades)} trades with T+{A6_HOLD_DAYS} outcome")
    print(f"    local_low: {a6_cnt['ll']}  |  fallback: {a6_cnt['fb']}  |  "
          f"no_entry: {a6_cnt['ne']}  |  no_bars: {a6_cnt['nb']}")
    print(f"    filtered: not_SR={a6_cnt['nsr']}, not_CS={a6_cnt['ncs']}")
    print(f"\n  -> C1: {len(c1_trades)} trades with T+{C1_HOLD_DAYS} outcome")
    print(f"    ok: {c1_cnt['ok']}  |  not_CS: {c1_cnt['ncs']}  |  no_exit: {c1_cnt['no_exit']}")

    print(f"\n  {'─'*50}")
    for step, dur in [('Daily', timings['daily']), ('Scan', timings['scan']),
                      ('Bars', timings['bars']), ('Refs', timings['refs']),
                      ('Classify', timings['classify']), ('Build', timings['build'])]:
        print(f"    {step:<12s}  {dur:>6.0f}s  ({dur/elapsed*100:>4.0f}%)")
    print(f"  TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return a6_trades, c1_trades, dates


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD RESULT
# ═══════════════════════════════════════════════════════════════════════════════

def build_result(trades, dates, label, hold_days):
    pnl_key = f'pnl_t{hold_days}'
    pnls = [t[pnl_key] for t in trades if t.get(pnl_key) is not None]
    m = compute_metrics(pnls, label)

    daily_pnl = defaultdict(list)
    for t in trades:
        if t.get(pnl_key) is not None:
            daily_pnl[t['scan_date']].append(t[pnl_key])

    equity = [CAPITAL]
    eq_dates = []
    running_capital = CAPITAL
    for dt in sorted(daily_pnl):
        n = len(daily_pnl[dt])
        pos_size = running_capital / n
        day_dollar = sum(pos_size * (r / 100) for r in daily_pnl[dt])
        running_capital += day_dollar
        equity.append(running_capital)
        eq_dates.append(dt)

    cum_nc = 0.0
    daily_stats = []
    for dt in dates:
        dp = daily_pnl.get(dt, [])
        if dp:
            n = len(dp)
            pos_size = CAPITAL / n
            day_dollar = sum(pos_size * (r / 100) for r in dp)
        else:
            day_dollar = 0.0
            n = 0
        cum_nc += day_dollar
        daily_stats.append({'date': dt, 'n_trades': n,
                            'day_pnl_dollar': day_dollar, 'cum_pnl': cum_nc})

    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = float(dd.min())
    active_days = sum(1 for ds in daily_stats if ds['n_trades'] > 0)
    tpd = len(pnls) / active_days if active_days > 0 else 0

    return {
        'label': label, 'trades': trades, 'metrics': m,
        'equity': equity, 'eq_dates': eq_dates,
        'daily_stats': daily_stats, 'max_dd': max_dd,
        'final_capital': running_capital, 'cum_nc': cum_nc,
        'active_days': active_days, 'tpd': tpd, 'hold_days': hold_days,
    }


def build_combined_result(a6_res, c1_res, dates):
    pnl_key_a6 = f'pnl_t{a6_res["hold_days"]}'
    pnl_key_c1 = f'pnl_t{c1_res["hold_days"]}'
    all_trades = []
    for t in a6_res['trades']:
        if t.get(pnl_key_a6) is not None:
            tc = copy.deepcopy(t)
            tc['combo_pnl'] = tc[pnl_key_a6]
            all_trades.append(tc)
    for t in c1_res['trades']:
        if t.get(pnl_key_c1) is not None:
            tc = copy.deepcopy(t)
            tc['combo_pnl'] = tc[pnl_key_c1]
            all_trades.append(tc)

    pnls = [t['combo_pnl'] for t in all_trades]
    m = compute_metrics(pnls, 'COMBINED')

    # Overlap analysis
    a6_td_set = set((t['ticker'], t['scan_date']) for t in a6_res['trades']
                     if t.get(pnl_key_a6) is not None)
    c1_td_set = set((t['ticker'], t['scan_date']) for t in c1_res['trades']
                     if t.get(pnl_key_c1) is not None)
    ticker_date_overlap = len(a6_td_set & c1_td_set)

    a6_dates = set(t['scan_date'] for t in a6_res['trades'] if t.get(pnl_key_a6) is not None)
    c1_dates = set(t['scan_date'] for t in c1_res['trades'] if t.get(pnl_key_c1) is not None)
    date_overlap = len(a6_dates & c1_dates)

    daily_pnl = defaultdict(list)
    for t in all_trades:
        daily_pnl[t['scan_date']].append(t['combo_pnl'])

    cum_nc = 0.0
    daily_stats = []
    for dt in dates:
        dp = daily_pnl.get(dt, [])
        if dp:
            n = len(dp)
            pos_size = CAPITAL / n
            day_dollar = sum(pos_size * (r / 100) for r in dp)
        else:
            day_dollar = 0.0
            n = 0
        cum_nc += day_dollar
        daily_stats.append({'date': dt, 'n_trades': n,
                            'day_pnl_dollar': day_dollar, 'cum_pnl': cum_nc})

    equity = [CAPITAL]
    eq_dates = []
    running_capital = CAPITAL
    for dt in sorted(daily_pnl):
        n = len(daily_pnl[dt])
        pos_size = running_capital / n
        day_dollar = sum(pos_size * (r / 100) for r in daily_pnl[dt])
        running_capital += day_dollar
        equity.append(running_capital)
        eq_dates.append(dt)

    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = float(dd.min())
    active_days = sum(1 for ds in daily_stats if ds['n_trades'] > 0)
    tpd = len(pnls) / active_days if active_days > 0 else 0

    return {
        'label': 'COMBINED', 'trades': all_trades, 'metrics': m,
        'equity': equity, 'eq_dates': eq_dates,
        'daily_stats': daily_stats, 'max_dd': max_dd,
        'final_capital': running_capital, 'cum_nc': cum_nc,
        'active_days': active_days, 'tpd': tpd,
        'ticker_date_overlap': ticker_date_overlap,
        'date_overlap': date_overlap,
        'a6_active_dates': len(a6_dates),
        'c1_active_dates': len(c1_dates),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(a6_res, c1_res, combo_res, dates):
    print(f"\n\n{'='*110}")
    print(f"  AGENT 3 — BASELINE RESULTS  |  {dates[0]} -> {dates[-1]}")
    print(f"  Capital: ${CAPITAL:,.0f}/day  |  Non-compounding")
    print(f"{'='*110}")

    all_res = [a6_res, c1_res, combo_res]

    # ── Main comparison table ──
    headers = ['Metric'] + [r['label'] for r in all_res]
    def _row(name, fn):
        return [name] + [fn(r) for r in all_res]

    tbl = [
        _row('Trades',       lambda r: f"{r['metrics']['n']}"),
        _row('Active days',  lambda r: f"{r['active_days']}"),
        _row('Trades/day',   lambda r: f"{r['tpd']:.2f}"),
        _row('Avg PnL',      lambda r: f"{r['metrics']['avg']:+.2f}%"),
        _row('Median PnL',   lambda r: f"{r['metrics']['med']:+.2f}%"),
        _row('Trimmed Mean',  lambda r: f"{r['metrics']['trimmed_mean']:+.2f}%"),
        _row('Win Rate',     lambda r: f"{r['metrics']['wr']*100:.0f}%"),
        _row('Sharpe',       lambda r: f"{r['metrics']['sharpe']:.3f}"),
        _row('Profit Factor', lambda r: f"{r['metrics']['pf']:.2f}"),
        _row('95% CI',       lambda r: f"[{r['metrics']['ci_lo']:+.2f}, {r['metrics']['ci_hi']:+.2f}]"),
        _row('CI > 0',       lambda r: f"{'YES' if r['metrics']['ci_lo'] > 0 else 'NO'}"),
        _row('Max Loss',     lambda r: f"{r['metrics']['max_loss']:+.2f}%"),
        _row('Max Gain',     lambda r: f"{r['metrics']['max_gain']:+.2f}%"),
        _row('Total $ (NC)', lambda r: f"${r['cum_nc']:+,.0f}"),
        _row('Total $ (CMP)', lambda r: f"${r['final_capital']-CAPITAL:+,.0f}"),
        _row('Max DD',       lambda r: f"{r['max_dd']:.1f}%"),
    ]
    print()
    print(tabulate(tbl, headers=headers, tablefmt='simple', stralign='right'))

    # ── Overlap analysis ──
    print(f"\n\n{'─'*80}")
    print(f"  PORTFOLIO DIVERSIFICATION")
    print(f"{'─'*80}")
    print(f"  Ticker+date overlap: {combo_res['ticker_date_overlap']}")
    print(f"  Date overlap: {combo_res['date_overlap']} / "
          f"{combo_res['a6_active_dates'] + combo_res['c1_active_dates']} active dates")
    print(f"  -> {'LOW' if combo_res['ticker_date_overlap'] == 0 else 'HIGH'} correlation")

    # ── Monthly comparison ──
    print(f"\n\n{'─'*80}")
    print(f"  MONTHLY BREAKDOWN")
    print(f"{'─'*80}")

    pnl_key_a6 = f'pnl_t{a6_res["hold_days"]}'
    pnl_key_c1 = f'pnl_t{c1_res["hold_days"]}'
    monthly_a6 = defaultdict(list)
    monthly_c1 = defaultdict(list)
    monthly_combo = defaultdict(list)
    for t in a6_res['trades']:
        if t.get(pnl_key_a6) is not None:
            monthly_a6[t['scan_date'][:7]].append(t[pnl_key_a6])
            monthly_combo[t['scan_date'][:7]].append(t[pnl_key_a6])
    for t in c1_res['trades']:
        if t.get(pnl_key_c1) is not None:
            monthly_c1[t['scan_date'][:7]].append(t[pnl_key_c1])
            monthly_combo[t['scan_date'][:7]].append(t[pnl_key_c1])

    all_months = sorted(set(list(monthly_a6.keys()) + list(monthly_c1.keys())))
    mo_headers = ['Month', 'A6 n', 'A6 avg', 'A6 $', 'C1 n', 'C1 avg', 'C1 $', 'Combo n', 'Combo $']
    rows = []
    for mo in all_months:
        a6p = monthly_a6.get(mo, [])
        c1p = monthly_c1.get(mo, [])
        cp = monthly_combo.get(mo, [])
        a6_dollar = sum(CAPITAL / max(1, len(a6p)) * (r / 100) for r in a6p) if a6p else 0
        c1_dollar = sum(CAPITAL / max(1, len(c1p)) * (r / 100) for r in c1p) if c1p else 0
        combo_dollar = a6_dollar + c1_dollar
        rows.append([
            mo,
            f"{len(a6p)}", f"{np.mean(a6p):+.2f}%" if a6p else '—', f"${a6_dollar:+,.0f}",
            f"{len(c1p)}", f"{np.mean(c1p):+.2f}%" if c1p else '—', f"${c1_dollar:+,.0f}",
            f"{len(cp)}", f"${combo_dollar:+,.0f}",
        ])
    print(tabulate(rows, headers=mo_headers, tablefmt='simple'))

    # ── Entry type for A6 ──
    print(f"\n\n{'─'*80}")
    print(f"  A6 ENTRY TYPE BREAKDOWN")
    print(f"{'─'*80}")
    for lbl in ['local_low', 'fallback']:
        arr = [t[pnl_key_a6] for t in a6_res['trades']
               if t['entry_type'] == lbl and t.get(pnl_key_a6) is not None]
        if arr:
            a = np.array(arr)
            sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
            print(f"    {lbl:12s}  n={len(a):<4d}  avg={np.mean(a):+.2f}%  "
                  f"WR={float((a>0).mean())*100:.0f}%  Sharpe={sh:.3f}")

    # ── Worst / Best ──
    for res in [a6_res, c1_res]:
        pnl_key = f'pnl_t{res["hold_days"]}'
        valid_trades = [t for t in res['trades'] if t.get(pnl_key) is not None]
        trades_s = sorted(valid_trades, key=lambda x: x[pnl_key])
        for lbl, sl in [('WORST 5', trades_s[:5]), ('BEST 5', trades_s[-5:])]:
            print(f"\n  -- {lbl} -- {res['label']} --")
            for t in sl:
                pnl = t[pnl_key]
                gap_pct = t['gap_pct'] * 100
                p5d = t.get('prior_5d_return')
                p5d_str = f"  5d_dec={p5d*100:+.1f}%" if p5d is not None else ''
                print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={gap_pct:>+7.1f}%  "
                      f"entry={t['entry_type']:>10s}  pnl={pnl:>+8.2f}%{p5d_str}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {'A6': '#27ae60', 'C1': '#2980b9', 'COMBINED': '#8e44ad'}

def plot_results(a6_res, c1_res, combo_res, dates):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    all_res = [a6_res, c1_res, combo_res]

    # ── Figure 1: Equity, Drawdown, Monthly, Distribution ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f'Agent 3 — Baseline Backtest (LONG): {dates[0]} -> {dates[-1]}\n'
        f'A6: {a6_res["metrics"]["n"]}t Sh={a6_res["metrics"]["sharpe"]:.3f}  |  '
        f'C1: {c1_res["metrics"]["n"]}t Sh={c1_res["metrics"]["sharpe"]:.3f}  |  '
        f'Combined: {combo_res["metrics"]["n"]}t Sh={combo_res["metrics"]["sharpe"]:.3f}',
        fontsize=10, fontweight='bold')

    # 1. Equity curves
    ax = axes[0, 0]
    for res in all_res:
        color = COLORS.get(res['label'].split()[0] if ' ' in res['label'] else res['label'], '#333')
        eq_d = [datetime.strptime(dates[0], '%Y-%m-%d')] + \
               [datetime.strptime(d, '%Y-%m-%d') for d in res['eq_dates']]
        ax.plot(eq_d, res['equity'], color=color, linewidth=2,
                label=f'{res["label"]} (${res["final_capital"]:,.0f})')
    ax.axhline(CAPITAL, color='#999', ls='--', lw=0.8)
    ax.set_title('Equity Curves (compounding)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # 2. Drawdown
    ax = axes[0, 1]
    for res in all_res:
        color = COLORS.get(res['label'].split()[0] if ' ' in res['label'] else res['label'], '#333')
        eq_d = [datetime.strptime(dates[0], '%Y-%m-%d')] + \
               [datetime.strptime(d, '%Y-%m-%d') for d in res['eq_dates']]
        eq_arr = np.array(res['equity'])
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak * 100
        ax.fill_between(eq_d, dd, 0, alpha=0.12, color=color)
        ax.plot(eq_d, dd, color=color, linewidth=1.5,
                label=f'{res["label"]} ({dd.min():.1f}%)')
    ax.set_title('Drawdown')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # 3. Monthly avg PnL
    ax = axes[1, 0]
    pnl_key_a6 = f'pnl_t{a6_res["hold_days"]}'
    pnl_key_c1 = f'pnl_t{c1_res["hold_days"]}'
    monthly = {'A6': defaultdict(list), 'C1': defaultdict(list)}
    for t in a6_res['trades']:
        if t.get(pnl_key_a6) is not None:
            monthly['A6'][t['scan_date'][:7]].append(t[pnl_key_a6])
    for t in c1_res['trades']:
        if t.get(pnl_key_c1) is not None:
            monthly['C1'][t['scan_date'][:7]].append(t[pnl_key_c1])
    months = sorted(set(list(monthly['A6'].keys()) + list(monthly['C1'].keys())))
    x = np.arange(len(months))
    w = 0.35
    a6_avgs = [np.mean(monthly['A6'].get(mo, [0])) for mo in months]
    c1_avgs = [np.mean(monthly['C1'].get(mo, [0])) for mo in months]
    ax.bar(x - w/2, a6_avgs, w, color=COLORS['A6'], alpha=0.8, label='A6')
    ax.bar(x + w/2, c1_avgs, w, color=COLORS['C1'], alpha=0.8, label='C1')
    ax.set_xticks(x)
    ax.set_xticklabels([mo[5:] for mo in months], fontsize=9)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_title('Monthly Avg PnL (%)')
    ax.set_ylabel('Avg PnL (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # 4. PnL distribution
    ax = axes[1, 1]
    a6_pnls = [t[pnl_key_a6] for t in a6_res['trades'] if t.get(pnl_key_a6) is not None]
    c1_pnls = [t[pnl_key_c1] for t in c1_res['trades'] if t.get(pnl_key_c1) is not None]
    all_pnls = a6_pnls + c1_pnls
    if all_pnls:
        clip_lo, clip_hi = np.percentile(all_pnls, 1), np.percentile(all_pnls, 99)
        bins = np.linspace(clip_lo, clip_hi, 50)
        ax.hist(a6_pnls, bins=bins, color=COLORS['A6'], alpha=0.45,
                label=f'A6 ({np.mean(a6_pnls):+.2f}%)')
        ax.hist(c1_pnls, bins=bins, color=COLORS['C1'], alpha=0.45,
                label=f'C1 ({np.mean(c1_pnls):+.2f}%)')
        ax.axvline(np.mean(a6_pnls), color=COLORS['A6'], ls='--', lw=1.5)
        ax.axvline(np.mean(c1_pnls), color=COLORS['C1'], ls='--', lw=1.5)
    ax.axvline(0, color='#333', lw=1.2)
    ax.set_xlabel('PnL (%)')
    ax.set_title('PnL Distribution')
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    p1 = os.path.join(BASE_DIR, 'agent3_baseline_bt.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")

    # ── Figure 2: Cumulative non-compounding P&L ──
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    for res in all_res:
        color = COLORS.get(res['label'].split()[0] if ' ' in res['label'] else res['label'], '#333')
        ds = res['daily_stats']
        d_dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in ds]
        cum = [s['cum_pnl'] for s in ds]
        ax.plot(d_dates, cum, color=color, linewidth=2,
                label=f'{res["label"]} (${res["cum_nc"]:+,.0f})')
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_title('Cumulative P&L (non-compounding, $100K/day)', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(BASE_DIR, 'agent3_baseline_cumulative.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p2}")



# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

a6_trades, c1_trades, dates = run_backtest()

# ── DIAGNOSTICS ──
print(f"\n{'='*80}")
print(f"  DIAGNOSTICS")
print(f"{'='*80}")

low_price_a6 = [t for t in a6_trades if t['open_price'] < 5]
low_price_c1 = [t for t in c1_trades if t['open_price'] < 5]
print(f"  A6 trades below $5: {len(low_price_a6)}")
for t in low_price_a6:
    pnl = t.get(f'pnl_t{A6_HOLD_DAYS}', 0) or 0
    print(f"    {t['scan_date']} {t['ticker']:>6s}  open=${t['open_price']:.2f}  pnl={pnl:+.2f}%")
print(f"  C1 trades below $5: {len(low_price_c1)}")
for t in low_price_c1:
    pnl = t.get(f'pnl_t{C1_HOLD_DAYS}', 0) or 0
    print(f"    {t['scan_date']} {t['ticker']:>6s}  open=${t['open_price']:.2f}  pnl={pnl:+.2f}%")

problem_tickers = ['PAVS', 'DBGI', 'GRI', 'BJDX', 'BRNX', 'GMEX']
for ptk in problem_tickers:
    a6_ptk = [t for t in a6_trades if t['ticker'] == ptk]
    c1_ptk = [t for t in c1_trades if t['ticker'] == ptk]
    if a6_ptk or c1_ptk:
        a6_sum = sum(t.get(f'pnl_t{A6_HOLD_DAYS}', 0) or 0 for t in a6_ptk)
        c1_sum = sum(t.get(f'pnl_t{C1_HOLD_DAYS}', 0) or 0 for t in c1_ptk)
        print(f"  {ptk}: A6={len(a6_ptk)} trades (impact {a6_sum:+.2f}%), "
              f"C1={len(c1_ptk)} trades (impact {c1_sum:+.2f}%)")

a6_ll = [t for t in a6_trades if t['entry_type'] == 'local_low']
a6_fb = [t for t in a6_trades if t['entry_type'] == 'fallback']
print(f"\n  A6 entry breakdown: local_low={len(a6_ll)}, fallback={len(a6_fb)}")
print(f"  Float universe: {len(_FLOAT_SET)} tickers (filtered from {len(_FLOAT)} total)")

a6_result = build_result(a6_trades, dates, 'A6', A6_HOLD_DAYS)
c1_result = build_result(c1_trades, dates, 'C1', C1_HOLD_DAYS)
combo_result = build_combined_result(a6_result, c1_result, dates)
print_results(a6_result, c1_result, combo_result, dates)
plot_results(a6_result, c1_result, combo_result, dates)
