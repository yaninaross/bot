# %% Agent 3 — Aug-Oct 2026 Investigation
#
# Comprehensive revalidation + new experiments for LONG strategies
# A6 (gap-down sustained rise) and C1 (multi-day reversal) on Aug 1 - Oct 6.
#
# Experiments:
#   R1  A6 Revalidation (Aug-Oct)
#   R2  C1 Revalidation (Aug-Oct) + Wed/Thu subset
#   E1  SPY Regime Gate Calibration
#   E2  Vol Regime Gate
#   E3  C1 Decline Threshold Sensitivity
#   E4  C1 Gap Magnitude Buckets
#   E5  Wed/Thu Effect Stability (C1 by day-of-week)
#   E6  Open Entry vs Local Low (A6)
#   N3  Reduced Gap Threshold (A6)
#   N4  Shorter Lookback for C1
#   N7  T+1 Gap-Up Prediction (overnight vs intraday)
#   N8  Intraday Window (LONG side, A6)
#
# Colab-cell friendly. ~15-20 min runtime (API-bound).

import os, sys, time, re, copy, json, asyncio
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

# A6 config
A6_GAP_MIN_PCT   = 5.0      # gap-DOWN >= 5%
A6_PRICE_MIN     = 10.0
A6_TOP_N         = 5         # top 5 by abs(gap_pct)
A6_PRE_FILTER_N  = 15        # scan top 15 before filtering
A6_HOLD_DAYS     = 1         # T+1

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

# Phase 1 OOS reference values
PHASE1_A6_N     = 71
PHASE1_A6_AVG   = 4.27
PHASE1_A6_SHARPE = 0.365

PHASE1_C1_N     = 176
PHASE1_C1_AVG   = 6.41
PHASE1_C1_SHARPE = 0.454

PHASE1_C1_WEDTHU_AVG    = 13.13
PHASE1_C1_WEDTHU_SHARPE = 0.848

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
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


# =============================================================================
# HELPERS
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
                wait = 12 * (attempt + 1)
                print(f"    [429 rate limit] waiting {wait}s (attempt {attempt+1}/5)...")
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


# =============================================================================
# ASYNC MINUTE BAR FETCH (RTH only)
# =============================================================================

async def _fetch_bars_one(session, tk, dt, sem):
    url = f"{_BASE}/v2/aggs/ticker/{tk}/range/1/minute/{dt}/{dt}"
    params = {"adjusted":"true","sort":"asc","limit":"500","apiKey":POLYGON_API_KEY}
    async with sem:
        for attempt in range(5):
            try:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        return (tk, dt), None
                    data = await resp.json()
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
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


# =============================================================================
# ASYNC TICKER REFERENCE FETCH
# =============================================================================

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


# =============================================================================
# GAP-DOWN PATTERN CLASSIFICATION (A6)
# =============================================================================

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


# =============================================================================
# LOCAL LOW DETECTION (A6 entry)
# =============================================================================

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


# =============================================================================
# METRICS
# =============================================================================

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

def _fmt_metrics(m):
    """Format a metrics dict for printing."""
    if m['n'] == 0:
        return "  n=0 (no trades)"
    return (f"  n={m['n']}  avg={m['avg']:+.2f}%  med={m['med']:+.2f}%  "
            f"Sharpe={m['sharpe']:.3f}  WR={m['wr']*100:.0f}%  "
            f"CI=[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]  "
            f"trimmed={m['trimmed_mean']:+.2f}%")


# =============================================================================
# =============================================================================
#
#  DATA PIPELINE
#
# =============================================================================
# =============================================================================

def run_data_pipeline():
    """Fetch all daily + minute bar data, build candidate lists, classify patterns."""
    t_total = time.time()

    print(f"\n{'='*80}")
    print(f"  AGENT 3 — AUG-OCT INVESTIGATION")
    print(f"  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"  Universe: _FLOAT ({len(_FLOAT_SET)} tickers with float >= 10M)")
    print(f"{'='*80}")

    # -- Trading days --
    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"\n  {n_dates} trading days")

    # -- Lookback dates for C1 (need 5d prior closes) + regime (need 10d prior) --
    lookback_start = _nth_prior_td(BACKTEST_START, 12)
    lb = lookback_start
    lookback_dates = []
    while lb < datetime.strptime(BACKTEST_START, '%Y-%m-%d').date():
        if _is_td(lb): lookback_dates.append(lb.strftime('%Y-%m-%d'))
        lb += timedelta(days=1)

    # -- Phase 1: Fetch grouped daily --
    t1 = time.time()
    all_fetch_dates = sorted(set(lookback_dates + dates))
    print(f"\n  Phase 1: Fetching grouped daily for {len(all_fetch_dates)} dates...")
    for i, dt in enumerate(all_fetch_dates):
        fetch_grouped_daily(dt)
        if (i+1) % 30 == 0 or i == len(all_fetch_dates) - 1:
            print(f"    [{i+1:>3}/{len(all_fetch_dates)}] {dt}  |  {time.time()-t_total:.0f}s")

    # Outcome dates (T+3 for C1 is the max)
    outcome_dates = set()
    for dt in dates:
        for nd in _next_n_td(dt, 3):
            outcome_dates.add(nd.strftime('%Y-%m-%d'))
    for od in sorted(outcome_dates):
        if od not in _gd_cache:
            try: fetch_grouped_daily(od)
            except: pass
    print(f"    + {len(outcome_dates)} outcome dates fetched")
    print(f"  Phase 1: {time.time()-t1:.0f}s")

    # -- Phase 2: Scan A6 + C1 candidates (broad scan for experiments) --
    t2 = time.time()
    print(f"\n  Phase 2: Scanning candidates (broad for experiments)...")

    # We scan with LOWER gap thresholds for N3 experiment (A6 >= 2%)
    # and LOWER decline thresholds / lookbacks for N4 experiment
    a6_candidates = {}    # dt -> list of candidate dicts
    c1_candidates = {}    # dt -> list of candidate dicts
    total_a6, total_c1 = 0, 0

    for i, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = _gd_cache.get(prior_str, {})
        today = _gd_cache.get(dt, {})
        nxt_t1 = _next_n_td(dt, 1)
        nxt_t2 = _next_n_td(dt, 2)
        nxt_t3 = _next_n_td(dt, 3)

        # -- A6: Gap-DOWN candidates (use 2% floor for N3 sweep) --
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
            if gap > -0.02: continue  # broad: gap-DOWN >= 2% for N3
            a6_day.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_o, 'scan_date': dt,
                'volume_today': today[tk]['v'],
                't1_dates': [nd.strftime('%Y-%m-%d') for nd in nxt_t1],
                't2_dates': [nd.strftime('%Y-%m-%d') for nd in nxt_t2],
                't3_dates': [nd.strftime('%Y-%m-%d') for nd in nxt_t3],
                'day_close': today[tk]['c'],
                'day_high': today[tk]['h'],
                'day_low': today[tk]['l'],
            })
        a6_day.sort(key=lambda x: x['gap_pct'])  # most negative first
        a6_candidates[dt] = a6_day
        total_a6 += len(a6_day)

        # -- C1: Gap-UP + prior decline (broad: 2d/3d/5d for N4) --
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
            if gap < C1_GAP_MIN_PCT / 100.0: continue
            if gap > C1_GAP_MAX_PCT / 100.0: continue

            # Compute prior 2d, 3d, 5d declines
            declines = {}
            for nd_lbl, nd_n in [('2d', 1), ('3d', 2), ('5d', 4)]:
                try:
                    close_nd_ago_date = _nth_prior_td(prior_str, nd_n).strftime('%Y-%m-%d')
                    close_nd_ago_data = _gd_cache.get(close_nd_ago_date, {}).get(tk)
                    if close_nd_ago_data and close_nd_ago_data['c'] > 0:
                        declines[nd_lbl] = (prev_c - close_nd_ago_data['c']) / close_nd_ago_data['c']
                except ValueError:
                    pass

            # Must have at least 5d decline computed
            if '5d' not in declines:
                continue

            c1_day.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_o, 'scan_date': dt,
                'volume_today': today[tk]['v'],
                'prior_2d_return': declines.get('2d'),
                'prior_3d_return': declines.get('3d'),
                'prior_5d_return': declines['5d'],
                't1_dates': [nd.strftime('%Y-%m-%d') for nd in nxt_t1],
                't2_dates': [nd.strftime('%Y-%m-%d') for nd in nxt_t2],
                't3_dates': [nd.strftime('%Y-%m-%d') for nd in nxt_t3],
                'day_of_week': datetime.strptime(dt, '%Y-%m-%d').strftime('%A'),
            })
        c1_day.sort(key=lambda x: -x['gap_pct'])
        c1_candidates[dt] = c1_day
        total_c1 += len(c1_day)

        if (i+1) % 20 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  A6={total_a6}, C1={total_c1}  |  {time.time()-t_total:.0f}s")
    print(f"  Phase 2: {time.time()-t2:.0f}s")

    # -- Phase 3: Fetch minute bars for A6 candidates --
    # Broader than baseline: fetch for all A6 candidates (not just top 15 per day)
    # because we need bars for N3 (lower gap thresholds) and N8 (intraday windows)
    t3 = time.time()
    a6_pairs = set()
    for dt, cands in a6_candidates.items():
        for c in cands[:20]:  # top 20 per day (generous for experiments)
            a6_pairs.add((c['ticker'], dt))

    # Also fetch SPY minute bars for E1/E2 intraday metrics
    for dt in dates:
        a6_pairs.add(('SPY', dt))

    print(f"\n  Phase 3: Fetching 1-min bars for {len(a6_pairs)} pairs...")
    pairs_list = list(a6_pairs)
    all_bars = {}
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start + 100]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        done = min(c_start + 100, len(pairs_list))
        if done % 200 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    print(f"  -> Got bars for {len(all_bars)}/{len(a6_pairs)} pairs")
    print(f"  Phase 3: {time.time()-t3:.0f}s")

    # -- Phase 4: Fetch ticker reference data --
    t4 = time.time()
    all_tickers = set()
    for cands in a6_candidates.values():
        for c in cands[:20]:
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
    print(f"  Phase 4: {time.time()-t4:.0f}s")

    # -- Phase 5: Classify patterns + find local low entries --
    t5 = time.time()
    pattern_cache = {}
    entry_cache = {}
    for key, rth_bars in all_bars.items():
        pattern_cache[key] = classify_gap_down_pattern(rth_bars)
        if rth_bars and len(rth_bars) >= LOW_WINDOW * 2 + 1:
            entry_cache[key] = find_local_low(rth_bars)
        else:
            entry_cache[key] = None

    pat_counts = defaultdict(int)
    for p in pattern_cache.values():
        pat_counts[p] += 1
    print(f"\n  Phase 5: Pattern classification ({len(pattern_cache)} candidates)")
    for pat in ['sustained_rise', 'rise_then_drop', 'mild_rise', 'continued_drop', 'no_data']:
        cnt = pat_counts.get(pat, 0)
        pct = cnt / len(pattern_cache) * 100 if pattern_cache else 0
        print(f"    {pat:25s}  {cnt:>5d}  ({pct:.1f}%)")
    print(f"  Phase 5: {time.time()-t5:.0f}s")

    elapsed = time.time() - t_total
    print(f"\n  DATA PIPELINE COMPLETE: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return {
        'dates': dates,
        'a6_candidates': a6_candidates,
        'c1_candidates': c1_candidates,
        'all_bars': all_bars,
        'pattern_cache': pattern_cache,
        'entry_cache': entry_cache,
    }


# =============================================================================
# =============================================================================
#
#  TRADE BUILDERS
#
# =============================================================================
# =============================================================================

def build_a6_trades(data, gap_min_pct=A6_GAP_MIN_PCT, hold_days=A6_HOLD_DAYS,
                    entry_mode='local_low', require_sustained_rise=True,
                    top_n=A6_TOP_N, pre_filter_n=A6_PRE_FILTER_N):
    """Build A6 trades with configurable parameters.
    entry_mode: 'local_low' (default), 'open' (buy at 9:30 open), 'both' (ll + fallback)
    """
    trades = []
    gap_min = gap_min_pct / 100.0

    for dt in data['dates']:
        cands = data['a6_candidates'].get(dt, [])
        # Filter by gap threshold
        cands_filtered = [c for c in cands if c['gap_pct'] <= -gap_min]
        qualified = []

        for c in cands_filtered[:pre_filter_n]:
            tk = c['ticker']
            key = (tk, dt)
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS': continue

            pat = data['pattern_cache'].get(key, 'no_data')
            if require_sustained_rise and pat != 'sustained_rise':
                continue

            if entry_mode == 'open':
                ep = c['open_price']
                entry_type = 'open'
                entry_time = '09:30'
            elif entry_mode in ('local_low', 'both'):
                entry = data['entry_cache'].get(key)
                if entry is None: continue
                if entry_mode == 'local_low' and entry['entry_type'] != 'local_low':
                    continue
                ep = entry['entry_price']
                entry_type = entry['entry_type']
                entry_time = entry['entry_time']
            else:
                continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = ep
            tc['entry_time'] = entry_time
            tc['entry_type'] = entry_type
            tc['open_pattern'] = pat
            tc['asset_type'] = at
            tc['strategy'] = 'A6'

            # Compute PnL for T+1, T+2, T+3
            for h, td_list_key in [(1, 't1_dates'), (2, 't2_dates'), (3, 't3_dates')]:
                for hi, td_str in enumerate(tc.get(td_list_key, []), 1):
                    td_data = _gd_cache.get(td_str, {}).get(tk, {})
                    if td_data.get('c'):
                        tc[f'pnl_t{h if h == hi else hi}'] = round((td_data['c'] - ep) / ep * 100, 3)
                    elif hi == h:
                        tc[f'pnl_t{h}'] = None
            # Correct approach: compute each hold period
            for h in [1, 2, 3]:
                td_list_key = f't{h}_dates'
                tds = tc.get(td_list_key, [])
                if tds:
                    last_td = tds[-1]
                    td_data = _gd_cache.get(last_td, {}).get(tk, {})
                    if td_data.get('c'):
                        tc[f'pnl_t{h}'] = round((td_data['c'] - ep) / ep * 100, 3)
                    else:
                        tc[f'pnl_t{h}'] = None
                else:
                    tc[f'pnl_t{h}'] = None

            # T+1 open (for N7)
            t1_dates = tc.get('t1_dates', [])
            if t1_dates:
                t1_data = _gd_cache.get(t1_dates[0], {}).get(tk, {})
                if t1_data.get('o') and ep > 0:
                    tc['t1_open_return'] = round((t1_data['o'] - ep) / ep * 100, 3)
                    if t1_data.get('c'):
                        tc['t1_intraday_return'] = round((t1_data['c'] - t1_data['o']) / t1_data['o'] * 100, 3)
                else:
                    tc['t1_open_return'] = None
                    tc['t1_intraday_return'] = None

            qualified.append(tc)

        for tc in qualified[:top_n]:
            if tc.get(f'pnl_t{hold_days}') is not None:
                trades.append(tc)

    return trades


def build_c1_trades(data, decline_min=C1_DECLINE_5D_MIN, decline_lookback='5d',
                    gap_min_pct=C1_GAP_MIN_PCT, hold_days=C1_HOLD_DAYS,
                    day_filter=None):
    """Build C1 trades with configurable parameters.
    decline_lookback: '2d', '3d', '5d'
    day_filter: None (all days), list of day names to include
    """
    trades = []
    gap_min = gap_min_pct / 100.0

    decline_key_map = {'2d': 'prior_2d_return', '3d': 'prior_3d_return', '5d': 'prior_5d_return'}
    decline_key = decline_key_map.get(decline_lookback, 'prior_5d_return')

    for dt in data['dates']:
        cands = data['c1_candidates'].get(dt, [])

        for c in cands:
            tk = c['ticker']
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS': continue

            # Gap filter
            if c['gap_pct'] < gap_min: continue

            # Decline filter
            decline_val = c.get(decline_key)
            if decline_val is None: continue
            if decline_val > -decline_min: continue

            # Day filter
            if day_filter is not None:
                dow = datetime.strptime(dt, '%Y-%m-%d').strftime('%A')
                if dow not in day_filter: continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = tc['open_price']
            tc['entry_time'] = '09:30'
            tc['entry_type'] = 'open'
            tc['open_pattern'] = 'c1_reversal'
            tc['asset_type'] = at
            tc['strategy'] = 'C1'

            ep = tc['entry_price']
            has_exit = False
            for h in [1, 2, 3]:
                td_list_key = f't{h}_dates'
                tds = tc.get(td_list_key, [])
                if tds:
                    last_td = tds[-1]
                    td_data = _gd_cache.get(last_td, {}).get(tk, {})
                    if td_data.get('c'):
                        tc[f'pnl_t{h}'] = round((td_data['c'] - ep) / ep * 100, 3)
                        if h == hold_days:
                            has_exit = True
                    else:
                        tc[f'pnl_t{h}'] = None
                else:
                    tc[f'pnl_t{h}'] = None

            if has_exit:
                trades.append(tc)

    return trades


# =============================================================================
# =============================================================================
#
#  EXPERIMENTS
#
# =============================================================================
# =============================================================================

def run_experiments(data):
    """Run all experiments and return results dict."""
    results = {}
    dates = data['dates']

    # =========================================================================
    # R1. A6 REVALIDATION (Aug-Oct)
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  R1. A6 REVALIDATION (Aug-Oct)")
    print(f"{'='*80}")

    a6_trades = build_a6_trades(data, gap_min_pct=A6_GAP_MIN_PCT, hold_days=A6_HOLD_DAYS,
                                entry_mode='local_low', require_sustained_rise=True)
    a6_pnls = [t[f'pnl_t{A6_HOLD_DAYS}'] for t in a6_trades if t.get(f'pnl_t{A6_HOLD_DAYS}') is not None]
    a6_m = compute_metrics(a6_pnls, 'A6_AugOct')
    print(_fmt_metrics(a6_m))

    # Compare to Phase 1 OOS
    print(f"\n  Phase 1 OOS reference: n={PHASE1_A6_N}, avg={PHASE1_A6_AVG:+.2f}%, Sharpe={PHASE1_A6_SHARPE:.3f}")
    print(f"  Delta n:      {a6_m['n'] - PHASE1_A6_N:+d}")
    print(f"  Delta avg:    {a6_m['avg'] - PHASE1_A6_AVG:+.2f}%")
    print(f"  Delta med:    n/a (no Phase 1 median)")
    print(f"  Delta Sharpe: {a6_m['sharpe'] - PHASE1_A6_SHARPE:+.3f}")
    print(f"  Delta WR:     n/a (no Phase 1 WR)")

    results['R1'] = {
        'label': 'A6 Revalidation Aug-Oct',
        'metrics': a6_m,
        'phase1_ref': {'n': PHASE1_A6_N, 'avg': PHASE1_A6_AVG, 'sharpe': PHASE1_A6_SHARPE},
        'delta_n': a6_m['n'] - PHASE1_A6_N,
        'delta_avg': round(a6_m['avg'] - PHASE1_A6_AVG, 3),
        'delta_sharpe': round(a6_m['sharpe'] - PHASE1_A6_SHARPE, 3),
    }

    # Print worst/best 5
    sorted_trades = sorted(a6_trades, key=lambda t: t.get(f'pnl_t{A6_HOLD_DAYS}', 0) or 0)
    for lbl, sl in [('WORST 5', sorted_trades[:5]), ('BEST 5', sorted_trades[-5:])]:
        print(f"\n  -- {lbl} --")
        for t in sl:
            pnl = t.get(f'pnl_t{A6_HOLD_DAYS}', 0) or 0
            print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={t['gap_pct']*100:>+7.1f}%  pnl={pnl:>+8.2f}%")


    # =========================================================================
    # R2. C1 REVALIDATION (Aug-Oct) + Wed/Thu subset
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  R2. C1 REVALIDATION (Aug-Oct)")
    print(f"{'='*80}")

    c1_trades = build_c1_trades(data, decline_min=C1_DECLINE_5D_MIN, decline_lookback='5d',
                                gap_min_pct=C1_GAP_MIN_PCT, hold_days=C1_HOLD_DAYS)
    c1_pnls = [t[f'pnl_t{C1_HOLD_DAYS}'] for t in c1_trades if t.get(f'pnl_t{C1_HOLD_DAYS}') is not None]
    c1_m = compute_metrics(c1_pnls, 'C1_AugOct')
    print(_fmt_metrics(c1_m))

    print(f"\n  Phase 1 OOS reference: n={PHASE1_C1_N}, avg={PHASE1_C1_AVG:+.2f}%, Sharpe={PHASE1_C1_SHARPE:.3f}")
    print(f"  Delta n:      {c1_m['n'] - PHASE1_C1_N:+d}")
    print(f"  Delta avg:    {c1_m['avg'] - PHASE1_C1_AVG:+.2f}%")
    print(f"  Delta Sharpe: {c1_m['sharpe'] - PHASE1_C1_SHARPE:+.3f}")

    # Wed/Thu subset
    c1_wedthu = build_c1_trades(data, decline_min=C1_DECLINE_5D_MIN, decline_lookback='5d',
                                gap_min_pct=C1_GAP_MIN_PCT, hold_days=C1_HOLD_DAYS,
                                day_filter=['Wednesday', 'Thursday'])
    c1_wt_pnls = [t[f'pnl_t{C1_HOLD_DAYS}'] for t in c1_wedthu if t.get(f'pnl_t{C1_HOLD_DAYS}') is not None]
    c1_wt_m = compute_metrics(c1_wt_pnls, 'C1_WedThu_AugOct')
    print(f"\n  Wed/Thu subset:")
    print(_fmt_metrics(c1_wt_m))
    print(f"\n  N5C1_WedThu OOS reference: avg={PHASE1_C1_WEDTHU_AVG:+.2f}%, Sharpe={PHASE1_C1_WEDTHU_SHARPE:.3f}")
    print(f"  Delta avg:    {c1_wt_m['avg'] - PHASE1_C1_WEDTHU_AVG:+.2f}%")
    print(f"  Delta Sharpe: {c1_wt_m['sharpe'] - PHASE1_C1_WEDTHU_SHARPE:+.3f}")

    results['R2'] = {
        'label': 'C1 Revalidation Aug-Oct',
        'metrics': c1_m,
        'phase1_ref': {'n': PHASE1_C1_N, 'avg': PHASE1_C1_AVG, 'sharpe': PHASE1_C1_SHARPE},
        'delta_n': c1_m['n'] - PHASE1_C1_N,
        'delta_avg': round(c1_m['avg'] - PHASE1_C1_AVG, 3),
        'delta_sharpe': round(c1_m['sharpe'] - PHASE1_C1_SHARPE, 3),
        'wedthu_metrics': c1_wt_m,
        'wedthu_ref': {'avg': PHASE1_C1_WEDTHU_AVG, 'sharpe': PHASE1_C1_WEDTHU_SHARPE},
        'wedthu_delta_avg': round(c1_wt_m['avg'] - PHASE1_C1_WEDTHU_AVG, 3),
        'wedthu_delta_sharpe': round(c1_wt_m['sharpe'] - PHASE1_C1_WEDTHU_SHARPE, 3),
    }

    # Print worst/best 5
    sorted_c1 = sorted(c1_trades, key=lambda t: t.get(f'pnl_t{C1_HOLD_DAYS}', 0) or 0)
    for lbl, sl in [('WORST 5', sorted_c1[:5]), ('BEST 5', sorted_c1[-5:])]:
        print(f"\n  -- {lbl} --")
        for t in sl:
            pnl = t.get(f'pnl_t{C1_HOLD_DAYS}', 0) or 0
            p5d = t.get('prior_5d_return')
            p5d_str = f"  5d_dec={p5d*100:+.1f}%" if p5d is not None else ''
            print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={t['gap_pct']*100:>+7.1f}%  "
                  f"pnl={pnl:>+8.2f}%{p5d_str}")


    # =========================================================================
    # E1. SPY REGIME GATE CALIBRATION
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  E1. SPY REGIME GATE CALIBRATION")
    print(f"{'='*80}")

    # Build SPY daily close series
    spy_closes = {}
    for dt_str in sorted(_gd_cache.keys()):
        spy_data = _gd_cache[dt_str].get('SPY')
        if spy_data and spy_data['c'] > 0:
            spy_closes[dt_str] = spy_data['c']

    spy_dates_sorted = sorted(spy_closes.keys())
    spy_5d_return = {}
    for i, dt in enumerate(spy_dates_sorted):
        if i >= 5:
            close_5d_ago = spy_closes[spy_dates_sorted[i - 5]]
            spy_5d_return[dt] = (spy_closes[dt] - close_5d_ago) / close_5d_ago
        else:
            spy_5d_return[dt] = None

    thresholds_e1 = [0.01, 0.015, 0.02, 0.03]  # +1%, +1.5%, +2%, +3%
    e1_rows = []
    print(f"\n  {'Threshold':>12s}  {'Excl %':>8s}  {'A6 n':>6s}  {'A6 avg':>8s}  {'C1 n':>6s}  {'C1 avg':>8s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*8}")

    for thr in thresholds_e1:
        # Days excluded: SPY 5d return > threshold (bull regime excluded)
        excluded_days = set()
        for dt in dates:
            r5d = spy_5d_return.get(dt)
            if r5d is not None and r5d > thr:
                excluded_days.add(dt)
        excl_pct = len(excluded_days) / len(dates) * 100 if dates else 0
        gated_dates = [dt for dt in dates if dt not in excluded_days]

        # A6 gated trades
        a6_gated_pnls = [t[f'pnl_t{A6_HOLD_DAYS}'] for t in a6_trades
                         if t['scan_date'] in set(gated_dates) and t.get(f'pnl_t{A6_HOLD_DAYS}') is not None]
        a6_g_m = compute_metrics(a6_gated_pnls)

        # C1 gated trades
        c1_gated_pnls = [t[f'pnl_t{C1_HOLD_DAYS}'] for t in c1_trades
                         if t['scan_date'] in set(gated_dates) and t.get(f'pnl_t{C1_HOLD_DAYS}') is not None]
        c1_g_m = compute_metrics(c1_gated_pnls)

        row = {
            'threshold': f'+{thr*100:.1f}%',
            'excluded_pct': round(excl_pct, 1),
            'a6_n': a6_g_m['n'],
            'a6_avg': round(a6_g_m['avg'], 2),
            'c1_n': c1_g_m['n'],
            'c1_avg': round(c1_g_m['avg'], 2),
        }
        e1_rows.append(row)
        print(f"  {row['threshold']:>12s}  {row['excluded_pct']:>7.1f}%  "
              f"{row['a6_n']:>6d}  {row['a6_avg']:>+7.2f}%  "
              f"{row['c1_n']:>6d}  {row['c1_avg']:>+7.2f}%")

    # Add no-gate row for reference
    print(f"  {'NO GATE':>12s}  {'0.0':>7s}%  "
          f"{a6_m['n']:>6d}  {a6_m['avg']:>+7.2f}%  "
          f"{c1_m['n']:>6d}  {c1_m['avg']:>+7.2f}%")

    results['E1'] = {'label': 'SPY Regime Gate Calibration', 'rows': e1_rows,
                     'baseline_a6': {'n': a6_m['n'], 'avg': round(a6_m['avg'], 2)},
                     'baseline_c1': {'n': c1_m['n'], 'avg': round(c1_m['avg'], 2)}}


    # =========================================================================
    # E2. VOL REGIME GATE
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  E2. VOL REGIME GATE (SPY 10d realized vol)")
    print(f"{'='*80}")

    # Compute SPY 10d realized vol
    spy_log_returns = {}
    for i in range(1, len(spy_dates_sorted)):
        dt = spy_dates_sorted[i]
        prev_dt = spy_dates_sorted[i - 1]
        if spy_closes[prev_dt] > 0:
            spy_log_returns[dt] = np.log(spy_closes[dt] / spy_closes[prev_dt])

    spy_10d_vol = {}
    for i in range(10, len(spy_dates_sorted)):
        dt = spy_dates_sorted[i]
        rets_10d = []
        for j in range(i - 9, i + 1):
            r = spy_log_returns.get(spy_dates_sorted[j])
            if r is not None:
                rets_10d.append(r)
        if len(rets_10d) >= 5:
            spy_10d_vol[dt] = float(np.std(rets_10d, ddof=1) * np.sqrt(252) * 100)  # annualized %

    # Median vol in period
    period_vols = [spy_10d_vol[dt] for dt in dates if dt in spy_10d_vol]
    if period_vols:
        med_vol = np.median(period_vols)
        print(f"  SPY 10d realized vol range: {min(period_vols):.1f}% - {max(period_vols):.1f}%")
        print(f"  Median: {med_vol:.1f}%")
    else:
        med_vol = 11.9

    vol_thresholds = [10.0, 11.9, 14.0, 16.0]
    e2_rows = []
    print(f"\n  {'Vol Gate':>12s}  {'Excl %':>8s}  {'A6 n':>6s}  {'A6 avg':>8s}  {'C1 n':>6s}  {'C1 avg':>8s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*8}")

    for vthr in vol_thresholds:
        # Exclude days where vol > threshold (high vol regime excluded)
        excluded_days = set()
        for dt in dates:
            vol = spy_10d_vol.get(dt)
            if vol is not None and vol > vthr:
                excluded_days.add(dt)
        excl_pct = len(excluded_days) / len(dates) * 100 if dates else 0
        gated_dates_set = set(dt for dt in dates if dt not in excluded_days)

        a6_gated_pnls = [t[f'pnl_t{A6_HOLD_DAYS}'] for t in a6_trades
                         if t['scan_date'] in gated_dates_set and t.get(f'pnl_t{A6_HOLD_DAYS}') is not None]
        c1_gated_pnls = [t[f'pnl_t{C1_HOLD_DAYS}'] for t in c1_trades
                         if t['scan_date'] in gated_dates_set and t.get(f'pnl_t{C1_HOLD_DAYS}') is not None]

        a6_g_m = compute_metrics(a6_gated_pnls)
        c1_g_m = compute_metrics(c1_gated_pnls)

        row = {
            'vol_gate': f'<{vthr:.1f}%',
            'excluded_pct': round(excl_pct, 1),
            'a6_n': a6_g_m['n'],
            'a6_avg': round(a6_g_m['avg'], 2),
            'c1_n': c1_g_m['n'],
            'c1_avg': round(c1_g_m['avg'], 2),
        }
        e2_rows.append(row)
        print(f"  {row['vol_gate']:>12s}  {row['excluded_pct']:>7.1f}%  "
              f"{row['a6_n']:>6d}  {row['a6_avg']:>+7.2f}%  "
              f"{row['c1_n']:>6d}  {row['c1_avg']:>+7.2f}%")

    print(f"  {'NO GATE':>12s}  {'0.0':>7s}%  "
          f"{a6_m['n']:>6d}  {a6_m['avg']:>+7.2f}%  "
          f"{c1_m['n']:>6d}  {c1_m['avg']:>+7.2f}%")

    results['E2'] = {'label': 'Vol Regime Gate', 'rows': e2_rows,
                     'median_vol': round(med_vol, 1),
                     'baseline_a6': {'n': a6_m['n'], 'avg': round(a6_m['avg'], 2)},
                     'baseline_c1': {'n': c1_m['n'], 'avg': round(c1_m['avg'], 2)}}


    # =========================================================================
    # E3. C1 DECLINE THRESHOLD SENSITIVITY
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  E3. C1 DECLINE THRESHOLD SENSITIVITY")
    print(f"{'='*80}")

    e3_rows = []
    print(f"\n  {'Decline':>12s}  {'n':>6s}  {'avg':>8s}  {'Sharpe':>8s}  {'WR':>6s}  {'CI':>20s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*20}")

    for dec_thr in [0.15, 0.20, 0.25]:
        c1_dec = build_c1_trades(data, decline_min=dec_thr, decline_lookback='5d',
                                 gap_min_pct=C1_GAP_MIN_PCT, hold_days=C1_HOLD_DAYS)
        pnls = [t[f'pnl_t{C1_HOLD_DAYS}'] for t in c1_dec if t.get(f'pnl_t{C1_HOLD_DAYS}') is not None]
        m = compute_metrics(pnls, f'C1_dec{int(dec_thr*100)}')
        row = {
            'decline': f'>={dec_thr*100:.0f}%',
            'n': m['n'], 'avg': round(m['avg'], 2),
            'sharpe': round(m['sharpe'], 3), 'wr': round(m['wr']*100, 0),
            'ci': f"[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]",
        }
        e3_rows.append(row)
        print(f"  {row['decline']:>12s}  {row['n']:>6d}  {row['avg']:>+7.2f}%  "
              f"{row['sharpe']:>8.3f}  {row['wr']:>5.0f}%  {row['ci']:>20s}")

    results['E3'] = {'label': 'C1 Decline Threshold Sensitivity', 'rows': e3_rows}


    # =========================================================================
    # E4. C1 GAP MAGNITUDE BUCKETS
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  E4. C1 GAP MAGNITUDE BUCKETS")
    print(f"{'='*80}")

    e4_rows = []
    gap_buckets = [(3, 5), (5, 8), (8, 12), (12, 50)]
    print(f"\n  {'Gap Bucket':>12s}  {'n':>6s}  {'avg':>8s}  {'Sharpe':>8s}  {'WR':>6s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}")

    for lo, hi in gap_buckets:
        bucket_pnls = []
        for t in c1_trades:
            gap = t['gap_pct'] * 100
            if lo <= gap < hi and t.get(f'pnl_t{C1_HOLD_DAYS}') is not None:
                bucket_pnls.append(t[f'pnl_t{C1_HOLD_DAYS}'])
        m = compute_metrics(bucket_pnls)
        lbl = f'{lo}%-{hi}%' if hi < 50 else f'{lo}%+'
        row = {
            'bucket': lbl, 'n': m['n'], 'avg': round(m['avg'], 2),
            'sharpe': round(m['sharpe'], 3), 'wr': round(m['wr']*100, 0),
        }
        e4_rows.append(row)
        print(f"  {row['bucket']:>12s}  {row['n']:>6d}  {row['avg']:>+7.2f}%  "
              f"{row['sharpe']:>8.3f}  {row['wr']:>5.0f}%")

    results['E4'] = {'label': 'C1 Gap Magnitude Buckets', 'rows': e4_rows}


    # =========================================================================
    # E5. WED/THU EFFECT STABILITY (C1 by day-of-week)
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  E5. WED/THU EFFECT STABILITY (C1 by day-of-week)")
    print(f"{'='*80}")

    e5_rows = []
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    print(f"\n  {'Day':>12s}  {'n':>6s}  {'avg':>8s}  {'Sharpe':>8s}  {'WR':>6s}  {'CI':>20s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*20}")

    for day in day_names:
        c1_day = build_c1_trades(data, decline_min=C1_DECLINE_5D_MIN, decline_lookback='5d',
                                 gap_min_pct=C1_GAP_MIN_PCT, hold_days=C1_HOLD_DAYS,
                                 day_filter=[day])
        pnls = [t[f'pnl_t{C1_HOLD_DAYS}'] for t in c1_day if t.get(f'pnl_t{C1_HOLD_DAYS}') is not None]
        m = compute_metrics(pnls, f'C1_{day[:3]}')
        row = {
            'day': day, 'n': m['n'], 'avg': round(m['avg'], 2),
            'sharpe': round(m['sharpe'], 3), 'wr': round(m['wr']*100, 0),
            'ci': f"[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]",
        }
        e5_rows.append(row)
        print(f"  {row['day']:>12s}  {row['n']:>6d}  {row['avg']:>+7.2f}%  "
              f"{row['sharpe']:>8.3f}  {row['wr']:>5.0f}%  {row['ci']:>20s}")

    results['E5'] = {'label': 'Wed/Thu Effect Stability', 'rows': e5_rows}


    # =========================================================================
    # E6. OPEN ENTRY vs LOCAL LOW (A6)
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  E6. OPEN ENTRY vs LOCAL LOW (A6)")
    print(f"{'='*80}")

    a6_open = build_a6_trades(data, gap_min_pct=A6_GAP_MIN_PCT, hold_days=A6_HOLD_DAYS,
                              entry_mode='open', require_sustained_rise=True)
    a6_open_pnls = [t[f'pnl_t{A6_HOLD_DAYS}'] for t in a6_open if t.get(f'pnl_t{A6_HOLD_DAYS}') is not None]
    a6_open_m = compute_metrics(a6_open_pnls, 'A6_open')

    a6_ll = build_a6_trades(data, gap_min_pct=A6_GAP_MIN_PCT, hold_days=A6_HOLD_DAYS,
                            entry_mode='local_low', require_sustained_rise=True)
    a6_ll_pnls = [t[f'pnl_t{A6_HOLD_DAYS}'] for t in a6_ll if t.get(f'pnl_t{A6_HOLD_DAYS}') is not None]
    a6_ll_m = compute_metrics(a6_ll_pnls, 'A6_local_low')

    print(f"\n  {'Entry':>12s}  {'n':>6s}  {'avg':>8s}  {'Sharpe':>8s}  {'WR':>6s}  {'CI':>20s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*20}")
    for lbl, m in [('open', a6_open_m), ('local_low', a6_ll_m)]:
        print(f"  {lbl:>12s}  {m['n']:>6d}  {m['avg']:>+7.2f}%  "
              f"{m['sharpe']:>8.3f}  {m['wr']*100:>5.0f}%  "
              f"[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]")

    results['E6'] = {
        'label': 'Open Entry vs Local Low',
        'open': {'n': a6_open_m['n'], 'avg': round(a6_open_m['avg'], 2),
                 'sharpe': round(a6_open_m['sharpe'], 3)},
        'local_low': {'n': a6_ll_m['n'], 'avg': round(a6_ll_m['avg'], 2),
                      'sharpe': round(a6_ll_m['sharpe'], 3)},
    }


    # =========================================================================
    # N3. REDUCED GAP THRESHOLD (A6)
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  N3. REDUCED GAP THRESHOLD (A6)")
    print(f"{'='*80}")

    n3_rows = []
    print(f"\n  {'Gap Thr':>12s}  {'n':>6s}  {'avg':>8s}  {'Sharpe':>8s}  {'WR':>6s}  {'CI':>20s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*20}")

    for gap_thr in [2.0, 3.0, 4.0, 5.0]:
        a6_var = build_a6_trades(data, gap_min_pct=gap_thr, hold_days=A6_HOLD_DAYS,
                                 entry_mode='local_low', require_sustained_rise=True)
        pnls = [t[f'pnl_t{A6_HOLD_DAYS}'] for t in a6_var if t.get(f'pnl_t{A6_HOLD_DAYS}') is not None]
        m = compute_metrics(pnls, f'A6_gap{int(gap_thr)}')
        row = {
            'gap_thr': f'>={gap_thr:.0f}%',
            'n': m['n'], 'avg': round(m['avg'], 2),
            'sharpe': round(m['sharpe'], 3), 'wr': round(m['wr']*100, 0),
            'ci': f"[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]",
        }
        n3_rows.append(row)
        print(f"  {row['gap_thr']:>12s}  {row['n']:>6d}  {row['avg']:>+7.2f}%  "
              f"{row['sharpe']:>8.3f}  {row['wr']:>5.0f}%  {row['ci']:>20s}")

    results['N3'] = {'label': 'Reduced Gap Threshold (A6)', 'rows': n3_rows}


    # =========================================================================
    # N4. SHORTER LOOKBACK FOR C1
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  N4. SHORTER LOOKBACK FOR C1")
    print(f"{'='*80}")

    n4_rows = []
    lookback_configs = [
        ('2d >= 7%',  0.07, '2d'),
        ('3d >= 10%', 0.10, '3d'),
        ('5d >= 15%', 0.15, '5d'),
    ]
    print(f"\n  {'Lookback':>12s}  {'n':>6s}  {'avg':>8s}  {'Sharpe':>8s}  {'WR':>6s}  {'CI':>20s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*20}")

    for lbl, dec_min, lb_key in lookback_configs:
        c1_var = build_c1_trades(data, decline_min=dec_min, decline_lookback=lb_key,
                                 gap_min_pct=C1_GAP_MIN_PCT, hold_days=C1_HOLD_DAYS)
        pnls = [t[f'pnl_t{C1_HOLD_DAYS}'] for t in c1_var if t.get(f'pnl_t{C1_HOLD_DAYS}') is not None]
        m = compute_metrics(pnls, f'C1_{lbl}')
        row = {
            'lookback': lbl, 'n': m['n'], 'avg': round(m['avg'], 2),
            'sharpe': round(m['sharpe'], 3), 'wr': round(m['wr']*100, 0),
            'ci': f"[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]",
        }
        n4_rows.append(row)
        print(f"  {row['lookback']:>12s}  {row['n']:>6d}  {row['avg']:>+7.2f}%  "
              f"{row['sharpe']:>8.3f}  {row['wr']:>5.0f}%  {row['ci']:>20s}")

    results['N4'] = {'label': 'Shorter Lookback for C1', 'rows': n4_rows}


    # =========================================================================
    # N7. T+1 GAP-UP PREDICTION (overnight vs intraday)
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  N7. T+1 GAP-UP PREDICTION (overnight vs intraday)")
    print(f"{'='*80}")

    overnight_returns = []
    intraday_returns = []
    total_t1_returns = []
    for t in a6_trades:
        t1_open = t.get('t1_open_return')
        t1_intra = t.get('t1_intraday_return')
        t1_total = t.get('pnl_t1')
        if t1_open is not None:
            overnight_returns.append(t1_open)
        if t1_intra is not None:
            intraday_returns.append(t1_intra)
        if t1_total is not None:
            total_t1_returns.append(t1_total)

    if overnight_returns:
        ovn_m = compute_metrics(overnight_returns, 'overnight_gap')
        intra_m = compute_metrics(intraday_returns, 'intraday')
        total_m = compute_metrics(total_t1_returns, 'total_T+1')

        print(f"\n  For A6 sustained_rise + local_low candidates (T+1):")
        print(f"  {'Component':>15s}  {'n':>6s}  {'avg':>8s}  {'med':>8s}  {'std':>8s}")
        print(f"  {'-'*15}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
        for lbl, m in [('Overnight gap', ovn_m), ('Intraday', intra_m), ('Total T+1', total_m)]:
            print(f"  {lbl:>15s}  {m['n']:>6d}  {m['avg']:>+7.2f}%  {m['med']:>+7.2f}%  {m['std']:>7.2f}%")

        ovn_contrib = ovn_m['avg'] / total_m['avg'] * 100 if total_m['avg'] != 0 else 0
        intra_contrib = intra_m['avg'] / total_m['avg'] * 100 if total_m['avg'] != 0 else 0
        print(f"\n  Overnight gap contribution: {ovn_contrib:.0f}%")
        print(f"  Intraday contribution:     {intra_contrib:.0f}%")
        print(f"  -> {'OVERNIGHT GAP' if abs(ovn_m['avg']) > abs(intra_m['avg']) else 'INTRADAY'} is the primary driver")
    else:
        print("  No T+1 return data available.")

    results['N7'] = {
        'label': 'T+1 Gap-Up Prediction',
        'overnight': {'n': len(overnight_returns),
                      'avg': round(np.mean(overnight_returns), 2) if overnight_returns else 0},
        'intraday': {'n': len(intraday_returns),
                     'avg': round(np.mean(intraday_returns), 2) if intraday_returns else 0},
        'total': {'n': len(total_t1_returns),
                  'avg': round(np.mean(total_t1_returns), 2) if total_t1_returns else 0},
    }


    # =========================================================================
    # N8. INTRADAY WINDOW (LONG side, A6)
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  N8. INTRADAY WINDOW (LONG side, A6)")
    print(f"{'='*80}")

    # For A6 sustained_rise candidates with minute bars:
    # entry at 9:45, exit at various times
    exit_times = ['10:30', '11:00', '11:30', '12:00']
    entry_time = '09:45'

    n8_rows = []
    print(f"\n  Entry={entry_time}")
    print(f"  {'Exit Time':>12s}  {'n':>6s}  {'avg':>8s}  {'Sharpe':>8s}  {'WR':>6s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}")

    for exit_t in exit_times:
        window_pnls = []
        for dt in data['dates']:
            cands = data['a6_candidates'].get(dt, [])
            gap_min = A6_GAP_MIN_PCT / 100.0
            cands_filtered = [c for c in cands if c['gap_pct'] <= -gap_min]

            for c in cands_filtered[:A6_PRE_FILTER_N]:
                tk = c['ticker']
                key = (tk, dt)
                at = _ticker_ref.get(tk, {}).get('type', 'UNK')
                if at != 'CS': continue

                pat = data['pattern_cache'].get(key, 'no_data')
                if pat != 'sustained_rise': continue

                bars = data['all_bars'].get(key)
                if bars is None or len(bars) < 30: continue

                # Find entry bar (9:45)
                entry_price = None
                exit_price = None
                for t_str, bar in bars:
                    if t_str == entry_time:
                        entry_price = bar['c']  # buy at close of 9:45 bar
                    if t_str == exit_t:
                        exit_price = bar['c']

                if entry_price and exit_price and entry_price > 0:
                    pnl = (exit_price - entry_price) / entry_price * 100
                    window_pnls.append(pnl)

        m = compute_metrics(window_pnls, f'intra_{exit_t}')
        row = {
            'exit_time': exit_t, 'n': m['n'], 'avg': round(m['avg'], 2),
            'sharpe': round(m['sharpe'], 3), 'wr': round(m['wr']*100, 0),
        }
        n8_rows.append(row)
        print(f"  {row['exit_time']:>12s}  {row['n']:>6d}  {row['avg']:>+7.2f}%  "
              f"{row['sharpe']:>8.3f}  {row['wr']:>5.0f}%")

    results['N8'] = {'label': 'Intraday Window (LONG)', 'entry': entry_time, 'rows': n8_rows}


    # =========================================================================
    # COMPREHENSIVE SUMMARY
    # =========================================================================
    print(f"\n\n{'='*80}")
    print(f"  COMPREHENSIVE SUMMARY")
    print(f"  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"{'='*80}")

    summary_rows = []

    # R1
    summary_rows.append(['R1 A6 baseline', f"{a6_m['n']}", f"{a6_m['avg']:+.2f}%",
                          f"{a6_m['sharpe']:.3f}",
                          f"{'PASS' if a6_m['ci_lo'] > 0 else 'FAIL'} CI>0"])
    # R2
    summary_rows.append(['R2 C1 baseline', f"{c1_m['n']}", f"{c1_m['avg']:+.2f}%",
                          f"{c1_m['sharpe']:.3f}",
                          f"{'PASS' if c1_m['ci_lo'] > 0 else 'FAIL'} CI>0"])
    summary_rows.append(['R2 C1 Wed/Thu', f"{c1_wt_m['n']}", f"{c1_wt_m['avg']:+.2f}%",
                          f"{c1_wt_m['sharpe']:.3f}",
                          f"{'PASS' if c1_wt_m['ci_lo'] > 0 else 'FAIL'} CI>0"])

    # E1 best
    if e1_rows:
        best_e1 = max(e1_rows, key=lambda r: r['a6_avg'] + r['c1_avg'])
        summary_rows.append([f"E1 best gate ({best_e1['threshold']})",
                              f"A6:{best_e1['a6_n']} C1:{best_e1['c1_n']}",
                              f"A6:{best_e1['a6_avg']:+.2f}% C1:{best_e1['c1_avg']:+.2f}%",
                              '', f"excl={best_e1['excluded_pct']:.0f}%"])

    # E2 best
    if e2_rows:
        best_e2 = max(e2_rows, key=lambda r: r['a6_avg'] + r['c1_avg'])
        summary_rows.append([f"E2 best vol ({best_e2['vol_gate']})",
                              f"A6:{best_e2['a6_n']} C1:{best_e2['c1_n']}",
                              f"A6:{best_e2['a6_avg']:+.2f}% C1:{best_e2['c1_avg']:+.2f}%",
                              '', f"excl={best_e2['excluded_pct']:.0f}%"])

    # E3 best
    if e3_rows:
        best_e3 = max(e3_rows, key=lambda r: r['avg'])
        summary_rows.append([f"E3 best decline ({best_e3['decline']})",
                              f"{best_e3['n']}", f"{best_e3['avg']:+.2f}%",
                              f"{best_e3['sharpe']:.3f}", ''])

    # E4 best
    if e4_rows:
        non_zero = [r for r in e4_rows if r['n'] > 0]
        if non_zero:
            best_e4 = max(non_zero, key=lambda r: r['avg'])
            summary_rows.append([f"E4 best gap bucket ({best_e4['bucket']})",
                                  f"{best_e4['n']}", f"{best_e4['avg']:+.2f}%",
                                  f"{best_e4['sharpe']:.3f}", ''])

    # E5 best day
    if e5_rows:
        non_zero_e5 = [r for r in e5_rows if r['n'] > 0]
        if non_zero_e5:
            best_e5 = max(non_zero_e5, key=lambda r: r['avg'])
            summary_rows.append([f"E5 best day ({best_e5['day']})",
                                  f"{best_e5['n']}", f"{best_e5['avg']:+.2f}%",
                                  f"{best_e5['sharpe']:.3f}", ''])

    # E6
    summary_rows.append(['E6 open entry',
                          f"{a6_open_m['n']}", f"{a6_open_m['avg']:+.2f}%",
                          f"{a6_open_m['sharpe']:.3f}", ''])
    summary_rows.append(['E6 local_low entry',
                          f"{a6_ll_m['n']}", f"{a6_ll_m['avg']:+.2f}%",
                          f"{a6_ll_m['sharpe']:.3f}", ''])

    # N3 best
    if n3_rows:
        best_n3 = max(n3_rows, key=lambda r: r['avg'] if r['n'] > 0 else -999)
        summary_rows.append([f"N3 best gap thr ({best_n3['gap_thr']})",
                              f"{best_n3['n']}", f"{best_n3['avg']:+.2f}%",
                              f"{best_n3['sharpe']:.3f}", ''])

    # N4 best
    if n4_rows:
        best_n4 = max(n4_rows, key=lambda r: r['avg'] if r['n'] > 0 else -999)
        summary_rows.append([f"N4 best lookback ({best_n4['lookback']})",
                              f"{best_n4['n']}", f"{best_n4['avg']:+.2f}%",
                              f"{best_n4['sharpe']:.3f}", ''])

    # N7
    if overnight_returns and total_t1_returns:
        driver = 'overnight' if abs(np.mean(overnight_returns)) > abs(np.mean(intraday_returns)) else 'intraday'
        summary_rows.append([f"N7 driver: {driver}",
                              f"{len(total_t1_returns)}",
                              f"ovn={np.mean(overnight_returns):+.2f}% intra={np.mean(intraday_returns):+.2f}%",
                              '', ''])

    # N8 best
    if n8_rows:
        non_zero_n8 = [r for r in n8_rows if r['n'] > 0]
        if non_zero_n8:
            best_n8 = max(non_zero_n8, key=lambda r: r['avg'])
            summary_rows.append([f"N8 best window (9:45->{best_n8['exit_time']})",
                                  f"{best_n8['n']}", f"{best_n8['avg']:+.2f}%",
                                  f"{best_n8['sharpe']:.3f}", ''])

    print()
    print(tabulate(summary_rows,
                   headers=['Experiment', 'n', 'Avg PnL', 'Sharpe', 'Notes'],
                   tablefmt='simple'))

    return results


# =============================================================================
# =============================================================================
#
#  MAIN EXECUTION
#
# =============================================================================
# =============================================================================

t_start = time.time()

# Phase A: Data pipeline
data = run_data_pipeline()

# Phase B: Run all experiments
results = run_experiments(data)

# Phase C: Save to JSON
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

results_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)

# Convert numpy types for JSON serialization
def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj == float('inf'):
        return "inf"
    if obj == float('-inf'):
        return "-inf"
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable: {obj}")

json_path = os.path.join(results_dir, 'investigation_aug_oct.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2, default=_json_safe)

elapsed_total = time.time() - t_start
print(f"\n\n{'='*80}")
print(f"  COMPLETE")
print(f"  Total runtime: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
print(f"  Results saved: {json_path}")
print(f"{'='*80}")
