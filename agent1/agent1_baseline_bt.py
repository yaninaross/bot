# %% Agent 1 — Gap-Fade SHORT Backtest
#
# Two variants, shared pipeline:
#   E2-A  Agent 1:     CS only, local_high bars 3-11, fb 15 (6:33-6:45 PST)
#   E2-B  Agent 1-SD:  CS + sustained_drop (confirmed 7:30 AM PST),
#                       local_high bars 60-72, fb 75 (7:30-7:45 PST)
#
# Scan at 09:31 ET (6:31 AM PST). Re-rank by gap at scan time.
# AH freshness: gap vs last AH price (16:00-20:00) on prior day, not prior close.
# Gap erosion: skip if >50% of gap consumed by scan time.
# SOFT float (frozen dict, miss=INCLUDE, hit+<10M=EXCLUDE), top 1 by gap.
# Exit 17:00 ET (2:00 PM PST). No TP, no SL, no PNL cap.
#
# Colab-cell friendly.

import os, sys, time, re, asyncio, copy
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

BACKTEST_START = '2026-07-01'
BACKTEST_END   = '2026-10-09'

GAP_MIN_PCT    = 3.0           # 3% threshold applied after re-ranking at SCAN_TIME
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
TOP_N          = 1
PRE_FILTER_N   = 30

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

SD_MAX_PEAK_BAR = 72       # local_high search bars 60-71 (10:30-10:41 ET)
SD_FALLBACK_BAR = 75       # fallback at 10:45 ET
SD_CONFIRM_BAR  = 59       # bar 59 close = SD confirmed at 10:30 ET

EXIT_TIME      = '17:00'      # exit at 17:00 ET (1h after close, after-hours)
SCAN_TIME_ET   = '09:31'      # 6:31 AM PST — rank candidates by gap at this time
GAP_CONSUMED_MAX = 0.50       # skip if >50% of gap eroded by scan time

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

_ETF_TYPES = {'ETF', 'ETV', 'ETN', 'ETP'}

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
# ASYNC MINUTE BAR FETCH (premarket + RTH + after-hours)
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
            if not bars:
                return (tk, dt), None
            pm = []
            rth = []
            ah = []
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


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC TICKER REFERENCE FETCH (asset type for CS filter)
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


# ═══════════════════════════════════════════════════════════════════════════════
# OPEN PATTERN CLASSIFICATION (E7 logic)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL HIGH DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_high(bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    fallback_bar=FALLBACK_BAR, min_entry_bar=0):
    if len(bars) < window * 2 + 1:
        return None
    start = max(window, min_entry_bar - window)
    for i in range(start, min(max_peak_bar, len(bars) - window)):
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
    fb = max(fallback_bar, min_entry_bar) if fallback_bar is not None else None
    if fb is not None and fb < len(bars):
        return {
            'entry_price': bars[fb][1]['c'],
            'entry_time': bars[fb][0],
            'entry_type': 'fallback',
            'peak_bar': None,
        }
    return None


def _T_to_min_entry_bar(T):
    h, m = int(T[:2]), int(T[3:5])
    T_min = h * 60 + m
    rth_start = 9 * 60 + 30
    if T_min <= rth_start:
        return 0
    return T_min - rth_start


def get_price_at_T(bar_data, T):
    if bar_data is None:
        return None
    all_bars = (bar_data.get('pm') or []) + (bar_data.get('rth') or [])
    best = None
    for t_str, bd in all_bars:
        if t_str <= T:
            best = bd['c']
        else:
            break
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER — returns ALL qualifying candidates sorted by gap
# ═══════════════════════════════════════════════════════════════════════════════

def scan_day(dt_str):
    """Initial scan using grouped daily open — wider 3% net to catch PM movers."""
    if not _is_td(dt_str): return []
    prior_str = _prior_td(dt_str).strftime('%Y-%m-%d')
    prior = fetch_grouped_daily(prior_str)
    today = fetch_grouped_daily(dt_str)

    gap_min = 0.03  # 3% wide net — reranked by PM price later
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
        s['prior_str'] = prior_str
    return candidates


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
    print(f"  AGENT 1 — Gap-Fade SHORT (E2 benchmark config)")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Universe: ALL US equities, gap >= {GAP_MIN_PCT}%, price >= ${PRICE_MIN}")
    print(f"  Gap source: re-ranked by gap at {SCAN_TIME_ET} ET (6:31 AM PST)")
    print(f"  Gap freshness: vs AH ref (last 16:00-20:00 bar on prior day)")
    print(f"  Gap erosion: skip if >{GAP_CONSUMED_MAX*100:.0f}% consumed by scan time")
    print(f"  Filter: CS + float SOFT (dict only, miss=INCLUDE)")
    print(f"  E2-A: LH bars 3-11, fb 15  |  E2-B: SD confirmed @10:30, LH bars 60-72, fb 75")
    print(f"  Exit: {EXIT_TIME} ET (2:00 PM PST)  |  Direction: SHORT")
    print(f"  Capital: ${CAPITAL:,.0f}/day, non-compounding")
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

    # ── Phase 1: Scan ──
    t1 = time.time()
    print(f"\n  Phase 1: Scanning for gap-ups (all US equities)...")
    all_candidates = {}
    total_cands = 0
    for i, dt in enumerate(dates):
        cands = scan_day(dt)
        all_candidates[dt] = cands
        total_cands += len(cands)
        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_cands} total  |  {time.time()-t_total:.0f}s")
    timings['scan'] = time.time() - t1

    # ── Phase 2: Fetch minute bars for top PRE_FILTER_N per day ──
    t2 = time.time()
    all_pairs = set()
    for dt, cands in all_candidates.items():
        for c in cands[:PRE_FILTER_N]:
            all_pairs.add((c['ticker'], dt))
            if c.get('prior_str'):
                all_pairs.add((c['ticker'], c['prior_str']))

    print(f"\n  Phase 2: Fetching 1-min bars for {len(all_pairs)} pairs...")
    pairs_list = list(all_pairs)
    all_bars = {}
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start + 100]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        done = min(c_start + 100, len(pairs_list))
        if done % 200 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    timings['bars'] = time.time() - t2
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

    # ── Phase 2a: AH freshness — recompute gap vs last after-hours price ──
    gap_min_final = GAP_MIN_PCT / 100.0
    print(f"\n  Phase 2a: Computing AH reference prices (gap freshness)...")
    n_ah_found = 0
    n_stale = 0
    for dt in dates:
        cands = all_candidates.get(dt, [])
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            prior_str = c.get('prior_str')
            if not prior_str:
                continue
            prior_bars = all_bars.get((tk, prior_str))
            if prior_bars is None:
                continue
            ah_bars = prior_bars.get('ah', [])
            if ah_bars:
                ah_ref = ah_bars[-1][1]['c']
                c['ah_ref_price'] = ah_ref
                n_ah_found += 1
                old_gap = c['gap_pct']
                fresh_gap = (c['open_price'] - ah_ref) / ah_ref
                c['gap_pct_daily'] = old_gap
                c['gap_pct'] = fresh_gap
                if fresh_gap < gap_min_final:
                    c['ah_stale'] = True
                    n_stale += 1
    print(f"  → AH reference: {n_ah_found} tickers with AH data, {n_stale} stale gaps excluded")

    # ── Phase 2b: Re-rank by gap at SCAN_TIME_ET + gap erosion ──
    print(f"\n  Phase 2b: Re-ranking by gap at {SCAN_TIME_ET} ET + gap erosion...")
    n_reranked = 0
    n_dropped = 0
    n_eroded = 0
    for dt in dates:
        cands = all_candidates.get(dt, [])
        reranked = []
        for c in cands:
            if c.get('ah_stale'):
                continue
            tk = c['ticker']
            bar_data = all_bars.get((tk, dt))
            price_T = get_price_at_T(bar_data, SCAN_TIME_ET)
            if price_T is None:
                n_dropped += 1
                continue
            ref_price = c.get('ah_ref_price', c['prior_close'])
            gap_T = (price_T - ref_price) / ref_price
            if gap_T < gap_min_final:
                continue
            if price_T < PRICE_MIN:
                continue
            c['gap_pct'] = gap_T
            c['scan_price'] = price_T

            rth = bar_data.get('rth', []) if bar_data else []
            if rth:
                daily_open = rth[0][1]['o']
                scan_bars = [b for b in rth if b[0] <= SCAN_TIME_ET]
                gap_range = daily_open - ref_price
                if scan_bars and gap_range > 0:
                    low_to_scan = min(b[1]['l'] for b in scan_bars)
                    gap_consumed = (daily_open - low_to_scan) / gap_range
                    c['gap_consumed'] = gap_consumed
                    if gap_consumed > GAP_CONSUMED_MAX:
                        n_eroded += 1
                        continue

            reranked.append(c)
            n_reranked += 1
        reranked.sort(key=lambda x: -x['gap_pct'])
        all_candidates[dt] = reranked

    print(f"  → Re-ranked {n_reranked} by gap at {SCAN_TIME_ET} "
          f"({n_dropped} no bar, {n_stale} stale, {n_eroded} eroded)")

    # ── Phase 3: Fetch ticker reference data (asset type for CS filter) ──
    t3 = time.time()
    all_tickers = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            all_tickers.add(c['ticker'])
    print(f"\n  Phase 3: Fetching reference data for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    for c_start in range(0, len(tk_list), 200):
        chunk = tk_list[c_start:c_start + 200]
        fetch_ticker_refs(chunk)
        done = min(c_start + 200, len(tk_list))
        if done % 400 == 0 or done == len(tk_list):
            print(f"    {done}/{len(tk_list)}  |  {time.time()-t_total:.0f}s")
    timings['refs'] = time.time() - t3
    n_cs = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('type') == 'CS')
    n_etf = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('type') in _ETF_TYPES)
    n_has_float = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('shares') is not None)
    n_float_ok = sum(1 for tk in all_tickers
                     if (_ticker_ref.get(tk, {}).get('shares') or 0) >= FLOAT_MIN)
    print(f"  → {n_cs} CS, {n_etf} ETF, {len(all_tickers) - n_cs - n_etf} other")
    print(f"  → Float data: {n_has_float}/{len(all_tickers)} have shares, "
          f"{n_float_ok} >= {FLOAT_MIN/1e6:.0f}M")

    # ── Phase 4: Classify patterns + find entries ──
    t4 = time.time()
    min_eb = _T_to_min_entry_bar(SCAN_TIME_ET)
    pattern_cache = {}
    entry_cache_bl = {}    # baseline: LH anchored to 09:31 (bars 1-12, fb 15)
    entry_cache_sd = {}    # SD LH: post-10:30 (bars 60-72, fb 75)
    for key, bar_data in all_bars.items():
        rth_bars = bar_data.get('rth', []) if isinstance(bar_data, dict) else bar_data
        pattern_cache[key] = classify_open_pattern(rth_bars)
        if rth_bars and len(rth_bars) >= PEAK_WINDOW * 2 + 1:
            entry_cache_bl[key] = find_local_high(rth_bars, min_entry_bar=min_eb)
        else:
            entry_cache_bl[key] = None
        if rth_bars and len(rth_bars) > SD_MAX_PEAK_BAR + PEAK_WINDOW:
            entry_cache_sd[key] = find_local_high(
                rth_bars, window=PEAK_WINDOW,
                max_peak_bar=SD_MAX_PEAK_BAR, fallback_bar=SD_FALLBACK_BAR,
                min_entry_bar=60)
        else:
            entry_cache_sd[key] = None
    timings['classify'] = time.time() - t4

    pat_counts = defaultdict(int)
    for p in pattern_cache.values():
        pat_counts[p] += 1
    print(f"\n  Phase 4: Pattern classification")
    for pat in ['sustained_drop', 'drop_then_recover', 'mild_drop', 'flat_or_bounce', 'no_data']:
        print(f"    {pat:25s}  {pat_counts.get(pat, 0):>5d}  "
              f"({pat_counts.get(pat, 0)/len(pattern_cache)*100:.1f}%)" if pattern_cache else '')

    # ── Phase 5: Fetch outcome dates ──
    print(f"\n  Phase 5: Fetching outcome dates...")
    outcome_dates = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            outcome_dates.add(c['scan_date'])
            if c.get('t1_date'): outcome_dates.add(c['t1_date'])
            if c.get('t2_date'): outcome_dates.add(c['t2_date'])
    to_fetch = sorted(od for od in outcome_dates if od not in _gd_cache)
    for od in to_fetch:
        try: fetch_grouped_daily(od)
        except: pass
    timings['outcomes'] = time.time() - t4

    elapsed = time.time() - t_total
    print(f"\n  Shared pipeline: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return all_candidates, all_bars, pattern_cache, entry_cache_bl, entry_cache_sd, dates, timings, t_total


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Build trades
#   mode='e2a':  CS only, entry via LH bars 3-11, fb 15 (entry_cache_bl)
#   mode='e2b':  CS + sustained_drop, LH bars 60-72, fb 75 (entry_cache_sd)
# ═══════════════════════════════════════════════════════════════════════════════

_MODE_LABELS = {
    'e2a': 'E2-A: Agent 1',
    'e2b': 'E2-B: Agent 1-SD',
    'e2a_nofb': 'Strategy B: No Fallback',
    'e2a_sd_nofb': 'Strategy B+SD: SD + No Fallback',
    'e2a_nofb_drift': 'C1: NoFB + Drift Exit',
    'e2a_nofb_accel': 'C2: NoFB + Accel Gate',
    'e2a_nofb_skip_t6': 'C3: NoFB + Skip T6',
    'e2a_nofb_combo': 'C4: NoFB + Drift+Accel+SkipT6',
    'e2a_nofb_pm': 'C5: NoFB + PM Retrace+RedBar',
    'e2a_nofb_all': 'C6: NoFB + ALL filters',
    'e2a_nofb_t6_pm': 'C3+C5: SkipT6+PM',
    'e2b_skip_t6': 'E2-B+C3: SD+SkipT6',
    'e2b_pm': 'E2-B+C5: SD+PM+Red',
    'e2b_t6_pm': 'E2-B+C3+C5: SD+SkipT6+PM',
}

def build_trades(all_candidates, all_bars, pattern_cache, entry_cache, dates,
                 mode='e2a', verbose=True, top_n=None):

    label = _MODE_LABELS[mode]
    require_sd = (mode in ('e2b', 'e2b_skip_t6', 'e2b_pm', 'e2b_t6_pm'))
    skip_fallback = (mode not in ('e2a', 'e2b', 'e2b_skip_t6', 'e2b_pm', 'e2b_t6_pm'))
    drift_exit = (mode in ('e2a_nofb_drift', 'e2a_nofb_combo', 'e2a_nofb_all'))
    accel_gate = (mode in ('e2a_nofb_accel', 'e2a_nofb_combo', 'e2a_nofb_all'))
    skip_t6 = (mode in ('e2a_nofb_skip_t6', 'e2a_nofb_combo', 'e2a_nofb_all',
                         'e2a_nofb_t6_pm', 'e2b_skip_t6', 'e2b_t6_pm'))
    pm_filters = (mode in ('e2a_nofb_pm', 'e2a_nofb_all',
                           'e2a_nofb_t6_pm', 'e2b_pm', 'e2b_t6_pm'))

    def _build_trade(c):
        tk, dt = c['ticker'], c['scan_date']
        entry = entry_cache.get((tk, dt))
        if entry is None:
            return None
        tc = copy.deepcopy(c)
        tc['entry_price'] = entry['entry_price']
        tc['entry_time'] = entry['entry_time']
        tc['entry_type'] = entry['entry_type']
        tc['open_pattern'] = pattern_cache.get((tk, dt), 'no_data')
        tc['asset_type'] = _ticker_ref.get(tk, {}).get('type', 'UNK')
        ep = tc['entry_price']

        bar_data = all_bars.get((tk, dt))
        rth = bar_data.get('rth', []) if bar_data else []

        # bar-30 early exit checks
        early_exit_bar30 = False
        if (drift_exit or accel_gate) and len(rth) > 30:
            open_price = rth[0][1]['o']
            bar30_close = rth[29][1]['c']
            if drift_exit and bar30_close > open_price:
                early_exit_bar30 = True
                tc['exit_reason'] = 'drift_exit'
            if accel_gate and not early_exit_bar30 and len(rth) > 29:
                mom_0_15 = (rth[14][1]['c'] - open_price) / open_price
                mom_15_30 = (rth[29][1]['c'] - rth[14][1]['c']) / open_price
                price_accel = mom_15_30 - mom_0_15
                if price_accel > 0:
                    early_exit_bar30 = True
                    tc['exit_reason'] = 'accel_exit'

        exit_price = None
        if bar_data:
            session_bars = rth + bar_data.get('ah', [])
            entry_t = tc['entry_time']
            if early_exit_bar30 and len(rth) > 30:
                exit_price = rth[30][1]['c'] if len(rth) > 30 else rth[-1][1]['c']
            else:
                tc['exit_reason'] = 'eod'
                for t_str, bar in session_bars:
                    if t_str > EXIT_TIME:
                        break
                    if t_str >= entry_t:
                        exit_price = bar['c']

        if exit_price is not None:
            tc['pnl_eod'] = round((ep - exit_price) / ep * 100, 3)
        else:
            t0d = _gd_cache.get(dt, {}).get(tk, {})
            if t0d.get('c'):
                tc['pnl_eod'] = round((ep - t0d['c']) / ep * 100, 3)
            else:
                tc['pnl_eod'] = None

        tc['exit_price'] = exit_price
        t1d = _gd_cache.get(tc.get('t1_date', ''), {}).get(tk, {})
        tc['pnl_t1'] = round((ep - t1d['c']) / ep * 100, 3) if t1d.get('c') else None
        t2d = _gd_cache.get(tc.get('t2_date', ''), {}).get(tk, {})
        tc['pnl_t2'] = round((ep - t2d['c']) / ep * 100, 3) if t2d.get('c') else None
        return tc

    trades = []
    cnt = {'lh': 0, 'fb': 0, 'ne': 0, 'nb': 0}
    daily_picks = {}

    for dt in dates:
        cands = all_candidates.get(dt, [])

        cs_qualified = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            ref = _ticker_ref.get(tk, {})
            at = ref.get('type', 'UNK')
            if at != 'CS':
                continue
            pattern = pattern_cache.get((tk, dt), 'no_data')
            if require_sd and pattern != 'sustained_drop':
                continue
            cc = copy.deepcopy(c)
            cc['asset_type'] = at
            cc['open_pattern'] = pattern
            cs_qualified.append(cc)

        _n = top_n if top_n is not None else TOP_N
        daily_picks[dt] = [c['ticker'] for c in cs_qualified[:_n]]

        for c in cs_qualified[:_n]:
            if skip_t6:
                gp = c.get('gap_pct', 0)
                if 0.15 <= gp < 0.30:
                    cnt['t6_skip'] = cnt.get('t6_skip', 0) + 1
                    continue
            if pm_filters:
                tk, ddt = c['ticker'], c['scan_date']
                bar_data = all_bars.get((tk, ddt))
                if bar_data:
                    pm_bars = bar_data.get('pm', [])
                    rth_bars_local = bar_data.get('rth', [])
                    # PM retrace check: require some gap retrace in PM
                    pm_after_830 = [(t_s, b) for t_s, b in pm_bars if t_s >= '08:30']
                    prior_close = c.get('prior_close', 0)
                    if pm_after_830 and prior_close > 0:
                        pm_high = max(b['h'] for _, b in pm_after_830)
                        last_pm_close = pm_after_830[-1][1]['c']
                        denom = pm_high - prior_close
                        pm_retrace = (pm_high - last_pm_close) / denom if denom > 0 else 0
                        if pm_retrace < 0.08:
                            cnt['pm_skip'] = cnt.get('pm_skip', 0) + 1
                            continue
                    # Green opening bar check: skip if first RTH bar is green
                    if rth_bars_local:
                        bar0 = rth_bars_local[0][1]
                        if bar0['c'] > bar0['o']:
                            cnt['green_skip'] = cnt.get('green_skip', 0) + 1
                            continue
            trade = _build_trade(c)
            if trade:
                if skip_fallback and trade['entry_type'] == 'fallback':
                    cnt['fb_skip'] = cnt.get('fb_skip', 0) + 1
                    continue
                cnt['lh' if trade['entry_type'] == 'local_high' else 'fb'] += 1
                trades.append(trade)
            elif (c['ticker'], c['scan_date']) not in all_bars:
                cnt['nb'] += 1
            else:
                cnt['ne'] += 1

    valid_trades = [t for t in trades if t.get('pnl_eod') is not None]

    if verbose:
        parts = []
        for k in ['lh', 'fb', 'fb_skip', 't6_skip', 'pm_skip', 'green_skip', 'ne', 'nb']:
            if cnt.get(k, 0) > 0:
                parts.append(f"{k}={cnt[k]}")
        # count early exits
        n_drift = sum(1 for t in valid_trades if t.get('exit_reason') == 'drift_exit')
        n_accel = sum(1 for t in valid_trades if t.get('exit_reason') == 'accel_exit')
        if n_drift: parts.append(f"drift_exit={n_drift}")
        if n_accel: parts.append(f"accel_exit={n_accel}")
        print(f"\n  {label}: {len(valid_trades)} trades  ({', '.join(parts)})")

    return valid_trades, daily_picks


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD RESULT
# ═══════════════════════════════════════════════════════════════════════════════

def build_result(trades, dates, label):
    pnls = [t['pnl_eod'] for t in trades]
    m = compute_metrics(pnls, label)

    daily_pnl = defaultdict(list)
    for t in trades:
        daily_pnl[t['scan_date']].append(t['pnl_eod'])

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
    tpd = len(trades) / active_days if active_days > 0 else 0

    return {
        'label': label, 'trades': trades, 'metrics': m,
        'equity': equity, 'eq_dates': eq_dates,
        'daily_stats': daily_stats, 'max_dd': max_dd,
        'final_capital': running_capital, 'cum_nc': cum_nc,
        'active_days': active_days, 'tpd': tpd,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(res, dates):
    m = res['metrics']
    print(f"\n\n{'='*90}")
    print(f"  AGENT 1 — Gap-Fade SHORT — {res['label']} — {dates[0]} → {dates[-1]}")
    print(f"  Exit: {EXIT_TIME} ET  |  Float: SOFT (dict, miss=INCLUDE)  |  Capital: ${CAPITAL:,.0f}/day NC")
    print(f"{'='*90}")

    tbl = [
        ['Trades',        f"{m['n']}"],
        ['Active days',   f"{res['active_days']}"],
        ['Trades/day',    f"{res['tpd']:.1f}"],
        ['Avg PnL',       f"{m['avg']:+.2f}%"],
        ['Median PnL',    f"{m['med']:+.2f}%"],
        ['Trimmed Mean',  f"{m['trimmed_mean']:+.2f}%"],
        ['Win Rate',      f"{m['wr']*100:.0f}%"],
        ['Sharpe',        f"{m['sharpe']:.3f}"],
        ['Profit Factor', f"{m['pf']:.2f}"],
        ['95% CI',        f"[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]"],
        ['Total $ (NC)',  f"${res['cum_nc']:+,.0f}"],
        ['Total $ (CMP)', f"${res['final_capital']-CAPITAL:+,.0f}"],
        ['Max DD',        f"{res['max_dd']:.1f}%"],
    ]
    print()
    print(tabulate(tbl, headers=['Metric', 'Value'], tablefmt='simple', stralign='right'))

    print(f"\n  ── ENTRY TYPE ──")
    for lbl in ['local_high', 'fallback']:
        arr = [t['pnl_eod'] for t in res['trades'] if t['entry_type'] == lbl]
        if arr:
            a = np.array(arr)
            sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
            print(f"    {lbl:12s}  n={len(a):<4d}  avg={np.mean(a):+.2f}%  "
                  f"WR={float((a>0).mean())*100:.0f}%  Sharpe={sh:.3f}")

    print(f"\n  ── PNL BY HORIZON ──")
    for key, lbl in [('pnl_eod', 'EOD'), ('pnl_t1', 'T+1'), ('pnl_t2', 'T+2')]:
        vals = [t[key] for t in res['trades'] if t.get(key) is not None]
        if vals:
            a = np.array(vals)
            sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
            print(f"    {lbl:6s}  n={len(a):<4d}  avg={np.mean(a):+.2f}%  "
                  f"med={np.median(a):+.2f}%  WR={float((a>0).mean())*100:.0f}%  Sharpe={sh:.3f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(res, dates, filename='agent1_baseline_bt.png'):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    color = '#2c3e50'
    m = res['metrics']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Agent 1 — Gap-Fade SHORT ({res["label"]}): {dates[0]} → {dates[-1]}\n'
        f'{m["n"]} trades  |  Avg={m["avg"]:+.2f}%  |  WR={m["wr"]*100:.0f}%  |  '
        f'Sharpe={m["sharpe"]:.3f}  |  ${res["cum_nc"]:+,.0f} NC',
        fontsize=10, fontweight='bold')

    ax = axes[0]
    ds = res['daily_stats']
    d_dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in ds]
    cum = [s['cum_pnl'] for s in ds]
    ax.plot(d_dates, cum, color=color, linewidth=2, label=f'${res["cum_nc"]:+,.0f}')
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_title('Cumulative P&L (non-compounding)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    pnls = [t['pnl_eod'] for t in res['trades']]
    if pnls:
        ax.hist(pnls, bins=max(10, len(pnls)//2), color=color, alpha=0.6,
                label=f'Avg={np.mean(pnls):+.2f}%')
        ax.axvline(np.mean(pnls), color=color, ls='--', lw=2)
    ax.axvline(0, color='#333', lw=1.2)
    ax.set_xlabel('EOD PnL (%)')
    ax.set_title('PnL Distribution')
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    p1 = os.path.join(BASE_DIR, filename)
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN — TOP_N SWEEP (9 tradeable strategies × N=1..5)
#
# LOOK-AHEAD AUDIT — every strategy verified tradeable:
#   Early entry (bars 3-11, entry_cache_bl):
#     E2-A          — baseline, all signals known at scan time
#     Strategy B    — E2-A minus fallback, no future info needed
#     C3            — skip T6 gap tier, known at scan time
#     C5            — PM retrace + red bar, both known before entry
#     C3+C5         — both filters pre-entry
#   Late entry (bars 60-72, entry_cache_sd):
#     E2-B          — SD confirmed at bar 60, entry at 60-72, time-consistent
#     E2-B+C3       — SD + skip T6, both known before entry
#     E2-B+C5       — SD + PM+red bar, both known before entry
#     E2-B+C3+C5    — all three filters, all known before entry
#
# EXCLUDED (look-ahead or proven harmful):
#   B+SD and all combos — SD at bar 60 but entry at bars 3-11 = LOOK-AHEAD
#   C1 (drift exit)     — tradeable but PROVEN HARMFUL (kills winners)
#   C2 (accel gate)     — tradeable but PROVEN HARMFUL
#   C4, C6              — contain C1/C2
# ═══════════════════════════════════════════════════════════════════════════════

all_candidates, all_bars, pattern_cache, entry_cache_bl, entry_cache_sd, dates, timings, t_total = run_backtest()

# (mode, short_label, entry_cache_key)
SWEEP_MODES = [
    ('e2a',              'E2-A',          'bl'),
    ('e2a_nofb',         'Strat B',       'bl'),
    ('e2a_nofb_skip_t6', 'C3: SkipT6',    'bl'),
    ('e2a_nofb_pm',      'C5: PM+Red',    'bl'),
    ('e2a_nofb_t6_pm',   'C3+C5',         'bl'),
    ('e2b',              'E2-B: SD',       'sd'),
    ('e2b_skip_t6',      'E2-B+C3',       'sd'),
    ('e2b_pm',           'E2-B+C5',       'sd'),
    ('e2b_t6_pm',        'E2-B+C3+C5',    'sd'),
]
N_VALUES = [1, 2, 3, 4, 5]
_CACHES = {'bl': entry_cache_bl, 'sd': entry_cache_sd}

n_strats = len(SWEEP_MODES)
n_configs = n_strats * len(N_VALUES)
print(f"\n  Running {n_strats} strategies × {len(N_VALUES)} TOP_N = {n_configs} configs...")

grid = {}
for mode, short_lbl, cache_key in SWEEP_MODES:
    for n in N_VALUES:
        trades, _ = build_trades(all_candidates, all_bars, pattern_cache,
                                 _CACHES[cache_key], dates, mode=mode,
                                 verbose=False, top_n=n)
        lbl = f"{short_lbl} N={n}"
        res = build_result(trades, dates, lbl)
        grid[(mode, n)] = res

# ── Full-period comparison table ──
print(f"\n\n{'='*140}")
print(f"  TOP_N SWEEP — {n_strats} Tradeable Strategies × N=1..5 (NO LOOK-AHEAD)")
print(f"  scan {SCAN_TIME_ET} ET, exit {EXIT_TIME} ET, {dates[0]} → {dates[-1]}")
print(f"{'='*140}")

for metric_label, fmt_fn in [
    ('TRADES',   lambda r: f"{r['metrics']['n']}"),
    ('AVG PNL',  lambda r: f"{r['metrics']['avg']:+.2f}%"),
    ('SHARPE',   lambda r: f"{r['metrics']['sharpe']:.3f}"),
    ('WIN RATE', lambda r: f"{r['metrics']['wr']*100:.0f}%"),
    ('$ NC',     lambda r: f"${r['cum_nc']:+,.0f}"),
    ('MAX DD',   lambda r: f"{r['max_dd']:.1f}%"),
    ('PF',       lambda r: f"{r['metrics']['pf']:.2f}"),
]:
    print(f"\n  ── {metric_label} ──")
    header = ['Strategy', 'Entry'] + [f"N={n}" for n in N_VALUES]
    rows = []
    for mode, short_lbl, cache_key in SWEEP_MODES:
        entry_lbl = '3-11' if cache_key == 'bl' else '60-72'
        row = [short_lbl, entry_lbl]
        for n in N_VALUES:
            row.append(fmt_fn(grid[(mode, n)]))
        rows.append(row)
    print(tabulate(rows, headers=header, tablefmt='simple', stralign='right'))

# ── Post-Aug 15 breakdown ──
print(f"\n\n{'='*140}")
print(f"  POST-AUG 15 — TOP_N SWEEP (Aug 16+ only)")
print(f"{'='*140}")

def _post_aug_metrics(res):
    pt = [t for t in res['trades'] if t['scan_date'] >= '2026-08-16']
    if not pt:
        return {'n': 0, 'avg': 0, 'wr': 0, 'sharpe': 0, 'pf': 0, 'dollar': 0}
    pa = np.array([t['pnl_eod'] for t in pt])
    sh = np.mean(pa) / np.std(pa, ddof=1) if len(pa) > 1 and np.std(pa, ddof=1) > 0 else 0
    w = pa[pa > 0]; l = pa[pa < 0]
    pf = float(w.sum() / abs(l.sum())) if len(l) > 0 and l.sum() != 0 else float('inf')
    return {'n': len(pt), 'avg': float(np.mean(pa)), 'wr': float((pa > 0).mean()),
            'sharpe': sh, 'pf': pf, 'dollar': float(100000 * np.sum(pa) / 100)}

post_grid = {}
for mode, short_lbl, _ in SWEEP_MODES:
    for n in N_VALUES:
        post_grid[(mode, n)] = _post_aug_metrics(grid[(mode, n)])

for metric_label, fmt_fn in [
    ('TRADES',   lambda m: f"{m['n']}"),
    ('AVG PNL',  lambda m: f"{m['avg']:+.2f}%"),
    ('SHARPE',   lambda m: f"{m['sharpe']:.3f}"),
    ('WIN RATE', lambda m: f"{m['wr']*100:.0f}%"),
    ('$ NC',     lambda m: f"${m['dollar']:+,.0f}"),
    ('PF',       lambda m: f"{m['pf']:.2f}"),
]:
    print(f"\n  ── {metric_label} (post-Aug 15) ──")
    header = ['Strategy', 'Entry'] + [f"N={n}" for n in N_VALUES]
    rows = []
    for mode, short_lbl, cache_key in SWEEP_MODES:
        entry_lbl = '3-11' if cache_key == 'bl' else '60-72'
        row = [short_lbl, entry_lbl]
        for n in N_VALUES:
            row.append(fmt_fn(post_grid[(mode, n)]))
        rows.append(row)
    print(tabulate(rows, headers=header, tablefmt='simple', stralign='right'))

# ── Best configs ranked by post-Aug Sharpe ──
print(f"\n\n{'='*100}")
print(f"  RANKED BY POST-AUG SHARPE (all {n_configs} configs)")
print(f"{'='*100}")
ranked = []
for mode, short_lbl, cache_key in SWEEP_MODES:
    entry_lbl = '3-11' if cache_key == 'bl' else '60-72'
    for n in N_VALUES:
        pm = post_grid[(mode, n)]
        fm = grid[(mode, n)]
        if pm['n'] == 0:
            continue
        ranked.append({
            'label': f"{short_lbl} N={n}",
            'entry': entry_lbl,
            'n_post': pm['n'], 'avg_post': pm['avg'],
            'sharpe_post': pm['sharpe'], 'wr_post': pm['wr'],
            'dollar_post': pm['dollar'], 'pf_post': pm['pf'],
            'n_full': fm['metrics']['n'], 'avg_full': fm['metrics']['avg'],
            'sharpe_full': fm['metrics']['sharpe'],
            'dollar_full': fm['cum_nc'],
        })
ranked.sort(key=lambda x: x['sharpe_post'], reverse=True)
rank_rows = []
for i, r in enumerate(ranked[:15], 1):
    rank_rows.append([
        i, r['label'], r['entry'],
        r['n_post'], f"{r['avg_post']:+.2f}%", f"{r['wr_post']*100:.0f}%",
        f"{r['sharpe_post']:.3f}", f"{r['pf_post']:.2f}",
        f"${r['dollar_post']:+,.0f}",
        f"| {r['n_full']}t", f"{r['sharpe_full']:.3f}", f"${r['dollar_full']:+,.0f}",
    ])
print()
print(tabulate(rank_rows,
               headers=['#', 'Config', 'Entry', 'N(post)', 'Avg', 'WR', 'Sharpe', 'PF',
                        '$(post)', '| N(full)', 'Sh(full)', '$(full)'],
               tablefmt='simple', stralign='right'))

# ── Charts ──
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

colors_9 = ['#2c3e50', '#7f8c8d', '#e74c3c', '#8e44ad', '#9b59b6',
            '#27ae60', '#3498db', '#e67e22', '#1abc9c']

fig, axes = plt.subplots(2, 2, figsize=(22, 14))
fig.suptitle(f'Agent 1 — TOP_N Sweep ({n_strats} tradeable strategies × N=1..5)\n'
             f'{dates[0]} → {dates[-1]}  |  Exit {EXIT_TIME} ET  |  NO LOOK-AHEAD',
             fontsize=11, fontweight='bold')

# Chart 1: Post-Aug $ NC by strategy and N
ax = axes[0, 0]
x_pos = np.arange(len(N_VALUES))
bar_w = 0.09
for i, (mode, short_lbl, _) in enumerate(SWEEP_MODES):
    vals = [post_grid[(mode, n)]['dollar'] for n in N_VALUES]
    ax.bar(x_pos + i * bar_w, vals, bar_w, label=short_lbl, color=colors_9[i])
ax.set_xticks(x_pos + bar_w * (n_strats // 2))
ax.set_xticklabels([f'N={n}' for n in N_VALUES])
ax.set_title('Post-Aug 15: $ P&L by TOP_N')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend(fontsize=6, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='#333', lw=0.8, ls=':')

# Chart 2: Post-Aug Sharpe by strategy and N
ax = axes[0, 1]
for i, (mode, short_lbl, _) in enumerate(SWEEP_MODES):
    sharpes = [post_grid[(mode, n)]['sharpe'] for n in N_VALUES]
    ax.plot(N_VALUES, sharpes, 'o-', color=colors_9[i], label=short_lbl, linewidth=2, markersize=6)
ax.set_xlabel('TOP_N')
ax.set_ylabel('Sharpe')
ax.set_title('Post-Aug 15: Sharpe by TOP_N')
ax.legend(fontsize=6, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xticks(N_VALUES)

# Chart 3: Equity curves for best N per strategy (post-Aug Sharpe argmax)
ax = axes[1, 0]
origin = datetime.strptime(BACKTEST_START, '%Y-%m-%d')
for i, (mode, short_lbl, _) in enumerate(SWEEP_MODES):
    best_n = max(N_VALUES, key=lambda n: post_grid[(mode, n)]['sharpe'])
    res = grid[(mode, best_n)]
    d_dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in res['daily_stats']]
    ax.plot([origin] + d_dates, [0] + [s['cum_pnl'] for s in res['daily_stats']],
            color=colors_9[i], linewidth=2,
            label=f'{short_lbl} N={best_n}: ${res["cum_nc"]:+,.0f}')
ax.axhline(0, color='#333', lw=0.8, ls=':')
ax.set_title('Equity Curves (best N per strategy)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

# Chart 4: Avg PnL vs Trades scatter (post-Aug)
ax = axes[1, 1]
for i, (mode, short_lbl, _) in enumerate(SWEEP_MODES):
    ns = [post_grid[(mode, n)]['n'] for n in N_VALUES]
    avgs = [post_grid[(mode, n)]['avg'] for n in N_VALUES]
    ax.plot(ns, avgs, 'o-', color=colors_9[i], label=short_lbl, linewidth=1.5, markersize=6)
    for j, n in enumerate(N_VALUES):
        ax.annotate(f'N={n}', (ns[j], avgs[j]), textcoords='offset points',
                    xytext=(4, 4), fontsize=6, color=colors_9[i])
ax.set_xlabel('# Trades (post-Aug)')
ax.set_ylabel('Avg PnL %')
ax.set_title('Post-Aug: Avg PnL vs Trade Count')
ax.legend(fontsize=6, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.92])
p1 = os.path.join(BASE_DIR, 'agent1_topn_sweep.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {p1}")

print(f"\n  Total runtime: {time.time() - t_total:.0f}s ({(time.time() - t_total)/60:.1f} min)")
