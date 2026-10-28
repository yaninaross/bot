# %% Agent 1 — 3-Arm Float Filter Comparison
#
# Tests E2-A (no SD) and E2-B (SD @10:30) under three float dictionaries:
#   Arm 1: Frozen  — _FLOAT (217 tickers), miss=INCLUDE
#   Arm 2: Live    — _FLOAT ∪ float_cache.json (219 tickers), miss=INCLUDE
#   Arm 3: Rebuilt — Polygon share_class_shares_outstanding, miss=EXCLUDE
#
# Shared pipeline: scan without float filter → apply per-arm in build_trades.
# Colab-cell friendly.

import os, sys, time, re, json, asyncio, copy
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

BACKTEST_START = '2026-08-01'
BACKTEST_END   = '2026-09-20'

GAP_MIN_PCT    = 3.0
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
TOP_N          = 1
PRE_FILTER_N   = 30

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

SD_MAX_PEAK_BAR = 72
SD_FALLBACK_BAR = 75
SD_CONFIRM_BAR  = 59

EXIT_TIME      = '17:00'
SCAN_TIME_ET   = '09:31'

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


# ═══════════════════════════════════════════════════════════════════════════════
# FLOAT DICTIONARIES
# ═══════════════════════════════════════════════════════════════════════════════

# Arm 1: Frozen (217 tickers — snapshot from baseline)
_FLOAT_FROZEN = {
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

# Arm 2: Live — frozen + float_cache.json extras (219 tickers)
# float_cache.json has identical values for all 217 frozen tickers
# plus 2 extras: BIAF (600K, <10M → excluded) and ODD (34.6M, >10M → included)
_FLOAT_LIVE = dict(_FLOAT_FROZEN)
_FLOAT_LIVE['BIAF'] = 600736
_FLOAT_LIVE['ODD'] = 34577554

# Arm 3: Rebuilt — constructed after Phase 3 from _ticker_ref
_FLOAT_REBUILT = {}   # populated after pipeline runs


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
# ASYNC MINUTE BAR FETCH
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
# PATTERN CLASSIFICATION + LOCAL HIGH
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
# SCANNER — NO FLOAT FILTER (applied per-arm in build_trades)
# ═══════════════════════════════════════════════════════════════════════════════

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
        # NO float filter here — applied per-arm in build_trades
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
    max_loss = float(a.min()) if n > 0 else 0
    return {'label':label,'n':n,'avg':avg,'med':med,'std':std,'wr':wr,
            'sharpe':sharpe,'pf':pf,'ci_lo':avg-1.96*se,'ci_hi':avg+1.96*se,
            'trimmed_mean':tm,'max_loss':max_loss}


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    t_total = time.time()

    print(f"\n{'='*80}")
    print(f"  AGENT 1 — 3-ARM FLOAT COMPARISON")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Pipeline: scan WITHOUT float filter → apply per-arm in build_trades")
    print(f"  Arm 1: Frozen ({len(_FLOAT_FROZEN)} tickers, miss=INCLUDE)")
    print(f"  Arm 2: Live   ({len(_FLOAT_LIVE)} tickers, miss=INCLUDE)")
    print(f"  Arm 3: Rebuilt (Polygon shares_outstanding, miss=EXCLUDE)")
    print(f"{'='*80}")

    # Trading days
    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"\n  {n_dates} trading days")

    # Phase 1: Scan (no float filter)
    t1 = time.time()
    print(f"\n  Phase 1: Scanning for gap-ups (no float filter)...")
    all_candidates = {}
    total_cands = 0
    for i, dt in enumerate(dates):
        cands = scan_day(dt)
        all_candidates[dt] = cands
        total_cands += len(cands)
        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_cands} total  |  {time.time()-t_total:.0f}s")

    # Phase 2: Fetch minute bars
    t2 = time.time()
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
        if done % 200 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

    # Phase 2b: Re-rank by gap at SCAN_TIME_ET
    gap_min_final = GAP_MIN_PCT / 100.0
    print(f"\n  Phase 2b: Re-ranking by gap at {SCAN_TIME_ET} ET...")
    n_reranked = 0
    n_dropped = 0
    for dt in dates:
        cands = all_candidates.get(dt, [])
        reranked = []
        for c in cands:
            tk = c['ticker']
            bar_data = all_bars.get((tk, dt))
            price_T = get_price_at_T(bar_data, SCAN_TIME_ET)
            if price_T is None:
                n_dropped += 1
                continue
            prev_c = c['prior_close']
            gap_T = (price_T - prev_c) / prev_c
            if gap_T < gap_min_final:
                continue
            c['gap_pct'] = gap_T
            c['scan_price'] = price_T
            reranked.append(c)
            n_reranked += 1
        reranked.sort(key=lambda x: -x['gap_pct'])
        all_candidates[dt] = reranked
    print(f"  → Re-ranked {n_reranked} candidates ({n_dropped} dropped: no bar at T)")

    # Phase 3: Ticker reference
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
    n_cs = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('type') == 'CS')
    n_has_shares = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('shares') is not None)
    print(f"  → {n_cs} CS  |  {n_has_shares}/{len(all_tickers)} have shares_outstanding")

    # Build Arm 3: Rebuilt float dict from _ticker_ref
    global _FLOAT_REBUILT
    _FLOAT_REBUILT = {}
    for tk, ref in _ticker_ref.items():
        shares = ref.get('shares')
        if shares is not None:
            _FLOAT_REBUILT[tk] = int(shares)
    n_rebuilt_lt10m = sum(1 for v in _FLOAT_REBUILT.values() if v < FLOAT_MIN)
    print(f"\n  Arm 3 dict built: {len(_FLOAT_REBUILT)} tickers from Polygon "
          f"({n_rebuilt_lt10m} below {FLOAT_MIN/1e6:.0f}M)")

    # Phase 4: Classify patterns + find entries
    t4 = time.time()
    min_eb = _T_to_min_entry_bar(SCAN_TIME_ET)
    pattern_cache = {}
    entry_cache_bl = {}
    entry_cache_sd = {}
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

    pat_counts = defaultdict(int)
    for p in pattern_cache.values():
        pat_counts[p] += 1
    print(f"\n  Phase 4: Pattern classification")
    for pat in ['sustained_drop', 'drop_then_recover', 'mild_drop', 'flat_or_bounce', 'no_data']:
        if pattern_cache:
            print(f"    {pat:25s}  {pat_counts.get(pat, 0):>5d}  "
                  f"({pat_counts.get(pat, 0)/len(pattern_cache)*100:.1f}%)")

    # Phase 5: Outcome dates
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

    elapsed = time.time() - t_total
    print(f"\n  Shared pipeline: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return (all_candidates, all_bars, pattern_cache,
            entry_cache_bl, entry_cache_sd, dates, t_total)


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD TRADES — per-arm float filter
# ═══════════════════════════════════════════════════════════════════════════════

_MODE_LABELS = {
    'e2a': 'E2-A: Agent 1',
    'e2b': 'E2-B: Agent 1-SD',
}

def build_trades(all_candidates, all_bars, pattern_cache, entry_cache, dates,
                 mode='e2a', float_dict=None, float_miss='include',
                 arm_label='', verbose=True):

    label = f"{_MODE_LABELS[mode]} — {arm_label}"
    require_sd = (mode == 'e2b')

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
        exit_price = None
        if bar_data:
            session_bars = bar_data.get('rth', []) + bar_data.get('ah', [])
            entry_t = tc['entry_time']
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
    excl = {'cs': 0, 'float_miss': 0, 'float_low': 0, 'sd': 0}

    for dt in dates:
        cands = all_candidates.get(dt, [])
        qualified = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            ref = _ticker_ref.get(tk, {})
            at = ref.get('type', 'UNK')
            if at != 'CS':
                excl['cs'] += 1
                continue
            if float_dict is not None:
                fs = float_dict.get(tk)
                if fs is None:
                    if float_miss == 'exclude':
                        excl['float_miss'] += 1
                        continue
                elif fs < FLOAT_MIN:
                    excl['float_low'] += 1
                    continue
            pattern = pattern_cache.get((tk, dt), 'no_data')
            if require_sd and pattern != 'sustained_drop':
                excl['sd'] += 1
                continue
            cc = copy.deepcopy(c)
            cc['asset_type'] = at
            cc['open_pattern'] = pattern
            qualified.append(cc)

        daily_picks[dt] = [c['ticker'] for c in qualified[:TOP_N]]

        for c in qualified[:TOP_N]:
            trade = _build_trade(c)
            if trade:
                cnt['lh' if trade['entry_type'] == 'local_high' else 'fb'] += 1
                trades.append(trade)
            elif (c['ticker'], c['scan_date']) not in all_bars:
                cnt['nb'] += 1
            else:
                cnt['ne'] += 1

    valid_trades = [t for t in trades if t.get('pnl_eod') is not None]

    if verbose:
        parts = [f"{k}={v}" for k, v in cnt.items() if v > 0]
        ef = [f"{k}={v}" for k, v in excl.items() if v > 0]
        print(f"\n  {label}: {len(valid_trades)} trades  ({', '.join(parts)})")
        if ef:
            print(f"    exclusions: {', '.join(ef)}")

    return valid_trades, daily_picks, excl


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD RESULT
# ═══════════════════════════════════════════════════════════════════════════════

def build_result(trades, dates, label):
    pnls = [t['pnl_eod'] for t in trades]
    m = compute_metrics(pnls, label)

    daily_pnl = defaultdict(list)
    for t in trades:
        daily_pnl[t['scan_date']].append(t['pnl_eod'])

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
    running = CAPITAL
    eq_dates = []
    for dt in sorted(daily_pnl):
        n = len(daily_pnl[dt])
        pos_size = running / n
        day_dollar = sum(pos_size * (r / 100) for r in daily_pnl[dt])
        running += day_dollar
        equity.append(running)
        eq_dates.append(dt)

    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = float(dd.min())
    active_days = sum(1 for ds in daily_stats if ds['n_trades'] > 0)

    return {
        'label': label, 'trades': trades, 'metrics': m,
        'daily_stats': daily_stats, 'max_dd': max_dd,
        'cum_nc': cum_nc, 'active_days': active_days,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#   RUN
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

(all_candidates, all_bars, pattern_cache,
 entry_cache_bl, entry_cache_sd, dates, t_total) = run_pipeline()

# ── Dict diagnostics ──
print(f"\n\n{'='*90}")
print(f"  FLOAT DICTIONARY SUMMARY")
print(f"{'='*90}")

_arm_dicts = {
    'Arm 1: Frozen':  (_FLOAT_FROZEN, 'include'),
    'Arm 2: Live':    (_FLOAT_LIVE,   'include'),
    'Arm 3: Rebuilt': (_FLOAT_REBUILT, 'exclude'),
}

dict_rows = []
for name, (fd, miss) in _arm_dicts.items():
    n_total = len(fd)
    n_lt = sum(1 for v in fd.values() if v < FLOAT_MIN)
    n_ge = n_total - n_lt
    dict_rows.append([name, n_total, n_ge, n_lt, miss])
print()
print(tabulate(dict_rows,
               headers=['Arm', 'Dict Size', '>= 10M', '< 10M', 'Miss Policy'],
               tablefmt='simple', stralign='right'))

# Overlap analysis
s1 = set(_FLOAT_FROZEN); s2 = set(_FLOAT_LIVE); s3 = set(_FLOAT_REBUILT)
print(f"\n  Overlap:")
print(f"    Arm 1 ∩ Arm 2: {len(s1 & s2)} tickers")
print(f"    Arm 1 ∩ Arm 3: {len(s1 & s3)} tickers")
print(f"    Arm 2 ∩ Arm 3: {len(s2 & s3)} tickers")
print(f"    All three:      {len(s1 & s2 & s3)} tickers")
only_2 = s2 - s1
if only_2:
    print(f"    Only in Arm 2 (not Frozen): {sorted(only_2)}")
only_3 = s3 - s2
if only_3 and len(only_3) <= 20:
    print(f"    Only in Arm 3 (not Live): {sorted(only_3)[:20]}{'...' if len(only_3) > 20 else ''}")
elif only_3:
    print(f"    Only in Arm 3 (not Live): {len(only_3)} tickers")
missing_3 = (s1 | s2) - s3
if missing_3:
    print(f"    In Arm 1/2 but NOT Arm 3: {sorted(missing_3)[:20]}{'...' if len(missing_3) > 20 else ''}")

# ── Candidate universe coverage (Arm 3) ──
all_candidate_tickers = set()
for cands in all_candidates.values():
    for c in cands[:PRE_FILTER_N]:
        all_candidate_tickers.add(c['ticker'])
arm3_coverage = sum(1 for tk in all_candidate_tickers if tk in _FLOAT_REBUILT)
print(f"\n  Arm 3 coverage of candidate universe: {arm3_coverage}/{len(all_candidate_tickers)} "
      f"({arm3_coverage/len(all_candidate_tickers)*100:.1f}%)")
uncovered = sorted(tk for tk in all_candidate_tickers if tk not in _FLOAT_REBUILT)
if uncovered and len(uncovered) <= 30:
    print(f"    Uncovered: {uncovered}")
elif uncovered:
    print(f"    Uncovered: {len(uncovered)} tickers (first 30: {uncovered[:30]})")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN 6 VARIANTS (2 modes × 3 arms)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*90}")
print(f"  BUILDING TRADES: 6 VARIANTS")
print(f"{'='*90}")

arms = [
    ('Arm 1: Frozen',  _FLOAT_FROZEN,  'include'),
    ('Arm 2: Live',    _FLOAT_LIVE,    'include'),
    ('Arm 3: Rebuilt', _FLOAT_REBUILT,  'exclude'),
]

results_a = {}   # mode='e2a', keyed by arm name
results_b = {}   # mode='e2b'
picks_a = {}
picks_b = {}
excl_a = {}
excl_b = {}

for arm_name, fd, miss in arms:
    trades, picks, excl = build_trades(
        all_candidates, all_bars, pattern_cache,
        entry_cache_bl, dates,
        mode='e2a', float_dict=fd, float_miss=miss,
        arm_label=arm_name)
    results_a[arm_name] = build_result(trades, dates, f"E2-A — {arm_name}")
    picks_a[arm_name] = picks
    excl_a[arm_name] = excl

    trades, picks, excl = build_trades(
        all_candidates, all_bars, pattern_cache,
        entry_cache_sd, dates,
        mode='e2b', float_dict=fd, float_miss=miss,
        arm_label=arm_name)
    results_b[arm_name] = build_result(trades, dates, f"E2-B — {arm_name}")
    picks_b[arm_name] = picks
    excl_b[arm_name] = excl


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(results, mode_label):
    arm_names = list(results.keys())
    ms = {a: results[a]['metrics'] for a in arm_names}
    base = arm_names[0]  # Arm 1 is baseline

    print(f"\n\n{'='*100}")
    print(f"  {mode_label} — 3-ARM FLOAT COMPARISON")
    print(f"  {BACKTEST_START} → {BACKTEST_END}  |  Exit {EXIT_TIME} ET  |  TOP_N={TOP_N}")
    print(f"{'='*100}")

    rows = []
    metrics = [
        ('Trades',       lambda a: f"{ms[a]['n']}",
                         lambda a: f"{ms[a]['n'] - ms[base]['n']:+d}"),
        ('Active days',  lambda a: f"{results[a]['active_days']}",
                         lambda a: f"{results[a]['active_days'] - results[base]['active_days']:+d}"),
        ('Avg PnL',      lambda a: f"{ms[a]['avg']:+.2f}%",
                         lambda a: f"{ms[a]['avg'] - ms[base]['avg']:+.2f}pp"),
        ('Median PnL',   lambda a: f"{ms[a]['med']:+.2f}%",
                         lambda a: f"{ms[a]['med'] - ms[base]['med']:+.2f}pp"),
        ('Trimmed Mean', lambda a: f"{ms[a]['trimmed_mean']:+.2f}%",
                         lambda a: f"{ms[a]['trimmed_mean'] - ms[base]['trimmed_mean']:+.2f}pp"),
        ('Win Rate',     lambda a: f"{ms[a]['wr']*100:.0f}%",
                         lambda a: f"{(ms[a]['wr']-ms[base]['wr'])*100:+.0f}pp"),
        ('Sharpe',       lambda a: f"{ms[a]['sharpe']:.3f}",
                         lambda a: f"{ms[a]['sharpe'] - ms[base]['sharpe']:+.3f}"),
        ('Profit Factor',lambda a: f"{ms[a]['pf']:.2f}",
                         lambda a: f"{ms[a]['pf'] - ms[base]['pf']:+.2f}"),
        ('$ NC',         lambda a: f"${results[a]['cum_nc']:+,.0f}",
                         lambda a: f"${results[a]['cum_nc'] - results[base]['cum_nc']:+,.0f}"),
        ('Max Loss',     lambda a: f"{ms[a]['max_loss']:+.2f}%",
                         lambda a: ''),
        ('Max DD',       lambda a: f"{results[a]['max_dd']:.1f}%",
                         lambda a: f"{results[a]['max_dd'] - results[base]['max_dd']:+.1f}pp"),
        ('95% CI',       lambda a: f"[{ms[a]['ci_lo']:+.1f},{ms[a]['ci_hi']:+.1f}]",
                         lambda a: ''),
    ]

    for metric_name, val_fn, delta_fn in metrics:
        row = [metric_name]
        for a in arm_names:
            row.append(val_fn(a))
        row.append(delta_fn(arm_names[1]))
        row.append(delta_fn(arm_names[2]))
        rows.append(row)

    headers = ['Metric'] + arm_names + ['Δ Arm2-1', 'Δ Arm3-1']
    print()
    print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right'))

    # Entry type breakdown
    print(f"\n  ── Entry type breakdown ──")
    for a in arm_names:
        by_et = defaultdict(list)
        for t in results[a]['trades']:
            by_et[t['entry_type']].append(t['pnl_eod'])
        parts = []
        for et in ['local_high', 'fallback']:
            arr = by_et.get(et, [])
            if arr:
                ar = np.array(arr)
                sh = np.mean(ar)/np.std(ar, ddof=1) if len(ar)>1 and np.std(ar, ddof=1)>0 else 0
                parts.append(f"{et}={len(ar)} avg={np.mean(ar):+.1f}% Sh={sh:.2f}")
        print(f"    {a:20s}  {' | '.join(parts)}")


print_comparison(results_a, 'E2-A: Agent 1 (no SD)')
print_comparison(results_b, 'E2-B: Agent 1-SD (SD @10:30)')


# ═══════════════════════════════════════════════════════════════════════════════
# EXCLUSION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*90}")
print(f"  EXCLUSION DIAGNOSTICS")
print(f"{'='*90}")

excl_rows = []
for arm_name in [a[0] for a in arms]:
    ea = excl_a[arm_name]
    eb = excl_b[arm_name]
    excl_rows.append([
        arm_name, 'E2-A',
        ea.get('cs', 0), ea.get('float_miss', 0), ea.get('float_low', 0),
        ea.get('sd', 0),
        ea.get('float_miss', 0) + ea.get('float_low', 0),
    ])
    excl_rows.append([
        arm_name, 'E2-B',
        eb.get('cs', 0), eb.get('float_miss', 0), eb.get('float_low', 0),
        eb.get('sd', 0),
        eb.get('float_miss', 0) + eb.get('float_low', 0),
    ])

print()
print(tabulate(excl_rows,
    headers=['Arm', 'Mode', 'CS excl', 'Float miss', 'Float <10M', 'SD excl', 'Total float'],
    tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# PICK DIVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*90}")
print(f"  PICK DIVERGENCE (E2-A)")
print(f"{'='*90}")

arm_names = [a[0] for a in arms]
diverge_days_a = []
for dt in dates:
    p1 = picks_a[arm_names[0]].get(dt, [])
    p2 = picks_a[arm_names[1]].get(dt, [])
    p3 = picks_a[arm_names[2]].get(dt, [])
    if p1 != p2 or p1 != p3:
        diverge_days_a.append([dt,
            p1[0] if p1 else '—',
            p2[0] if p2 else '—',
            p3[0] if p3 else '—'])

if diverge_days_a:
    print(f"\n  {len(diverge_days_a)}/{len(dates)} days with different picks:\n")
    print(tabulate(diverge_days_a,
        headers=['Date', 'Arm 1', 'Arm 2', 'Arm 3'],
        tablefmt='simple', stralign='right'))
else:
    print(f"\n  All {len(dates)} days: same pick across all arms")

print(f"\n\n{'='*90}")
print(f"  PICK DIVERGENCE (E2-B)")
print(f"{'='*90}")

diverge_days_b = []
for dt in dates:
    p1 = picks_b[arm_names[0]].get(dt, [])
    p2 = picks_b[arm_names[1]].get(dt, [])
    p3 = picks_b[arm_names[2]].get(dt, [])
    if p1 != p2 or p1 != p3:
        diverge_days_b.append([dt,
            p1[0] if p1 else '—',
            p2[0] if p2 else '—',
            p3[0] if p3 else '—'])

if diverge_days_b:
    print(f"\n  {len(diverge_days_b)}/{len(dates)} days with different picks:\n")
    print(tabulate(diverge_days_b,
        headers=['Date', 'Arm 1', 'Arm 2', 'Arm 3'],
        tablefmt='simple', stralign='right'))
else:
    print(f"\n  All {len(dates)} days: same pick across all arms")


# ═══════════════════════════════════════════════════════════════════════════════
# ARM 2 vs ARM 1 TRADE-LEVEL IMPACT (E2-A)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*90}")
print(f"  ARM 2 vs ARM 1 TRADE-LEVEL IMPACT (E2-A)")
print(f"{'='*90}")

trades_a1 = {t['scan_date']: t for t in results_a[arm_names[0]]['trades']}
trades_a2 = {t['scan_date']: t for t in results_a[arm_names[1]]['trades']}

diff_rows = []
for dt in dates:
    t1 = trades_a1.get(dt)
    t2 = trades_a2.get(dt)
    tk1 = t1['ticker'] if t1 else '—'
    tk2 = t2['ticker'] if t2 else '—'
    if tk1 != tk2:
        pnl1 = f"{t1['pnl_eod']:+.2f}%" if t1 else '—'
        pnl2 = f"{t2['pnl_eod']:+.2f}%" if t2 else '—'
        diff_rows.append([dt, tk1, pnl1, tk2, pnl2])

if diff_rows:
    print(f"\n  {len(diff_rows)} days with different trades:\n")
    print(tabulate(diff_rows,
        headers=['Date', 'Arm1 Ticker', 'Arm1 PnL', 'Arm2 Ticker', 'Arm2 PnL'],
        tablefmt='simple', stralign='right'))
    if len(diff_rows) > 0:
        a1_pnls = [trades_a1[r[0]]['pnl_eod'] for r in diff_rows if r[0] in trades_a1]
        a2_pnls = [trades_a2[r[0]]['pnl_eod'] for r in diff_rows if r[0] in trades_a2]
        if a1_pnls:
            print(f"\n    Arm 1 divergent trades: n={len(a1_pnls)}, avg={np.mean(a1_pnls):+.2f}%")
        if a2_pnls:
            print(f"    Arm 2 divergent trades: n={len(a2_pnls)}, avg={np.mean(a2_pnls):+.2f}%")
else:
    print(f"\n  All trades identical between Arm 1 and Arm 2")


# ═══════════════════════════════════════════════════════════════════════════════
# VALUE DIVERGENCE — tickers in Arm 1/2 with different float values
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*90}")
print(f"  FLOAT VALUE DIVERGENCE (Frozen vs Live vs Rebuilt)")
print(f"{'='*90}")

val_diff_rows = []
all_tks = sorted(set(_FLOAT_FROZEN) | set(_FLOAT_LIVE) | set(_FLOAT_REBUILT))
for tk in all_tks:
    v1 = _FLOAT_FROZEN.get(tk)
    v2 = _FLOAT_LIVE.get(tk)
    v3 = _FLOAT_REBUILT.get(tk)
    if v1 == v2 and v1 == v3:
        continue
    cross_10m = False
    vals = [v for v in [v1, v2, v3] if v is not None]
    if vals and (min(vals) < FLOAT_MIN <= max(vals)):
        cross_10m = True
    if cross_10m or (v1 is not None and v3 is not None and abs(v1 - v3) / max(v1, 1) > 0.10):
        val_diff_rows.append([
            tk,
            f"{v1/1e6:.1f}M" if v1 is not None else '—',
            f"{v2/1e6:.1f}M" if v2 is not None else '—',
            f"{v3/1e6:.1f}M" if v3 is not None else '—',
            '⚠ CROSS' if cross_10m else 'diverge',
        ])

if val_diff_rows:
    val_diff_rows.sort(key=lambda r: r[4], reverse=True)
    print(f"\n  {len(val_diff_rows)} tickers with material float divergence:\n")
    print(tabulate(val_diff_rows[:30],
        headers=['Ticker', 'Arm 1', 'Arm 2', 'Arm 3', 'Note'],
        tablefmt='simple', stralign='right'))
    if len(val_diff_rows) > 30:
        print(f"  ... and {len(val_diff_rows) - 30} more")
else:
    print(f"\n  No material float divergence found")


# ═══════════════════════════════════════════════════════════════════════════════
# LIMITATIONS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*90}")
print(f"  LIMITATIONS")
print(f"{'='*90}")
print(f"""
  1. Arm 3 uses Polygon share_class_shares_outstanding, NOT true float.
     True float = shares_outstanding - restricted/insider shares.
     Polygon does not provide float directly.

  2. Point-in-time: Polygon reference returns CURRENT shares outstanding.
     For historical backtests, the share count may have changed (dilution,
     buybacks, splits). This introduces look-ahead bias for Arm 3.

  3. Arm 1/2 dicts are frozen snapshots — they also have point-in-time issues
     but are consistent with the deployed trading system.

  4. PRE_FILTER_N={PRE_FILTER_N}: if removing float from scan_day causes >30
     candidates on a day, some names that would pass one arm but not another
     may fall outside the bar-fetch window.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

arm_colors = {'Arm 1: Frozen': '#2c3e50', 'Arm 2: Live': '#2980b9', 'Arm 3: Rebuilt': '#e67e22'}

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'Agent 1 — 3-Arm Float Comparison\n'
             f'{BACKTEST_START} → {BACKTEST_END}  |  Exit {EXIT_TIME} ET  |  TOP_N={TOP_N}',
             fontsize=11, fontweight='bold')

origin = datetime.strptime(BACKTEST_START, '%Y-%m-%d')

for col, (mode_results, mode_label) in enumerate([
    (results_a, 'E2-A: Agent 1 (no SD)'),
    (results_b, 'E2-B: Agent 1-SD'),
]):
    ax = axes[0][col]
    d_dates = [datetime.strptime(s['date'], '%Y-%m-%d')
               for s in list(mode_results.values())[0]['daily_stats']]
    for arm_name in arm_names:
        res = mode_results[arm_name]
        m = res['metrics']
        cum = [s['cum_pnl'] for s in res['daily_stats']]
        ax.plot([origin] + d_dates, [0] + cum,
                color=arm_colors[arm_name], linewidth=2,
                label=f'{arm_name}: ${res["cum_nc"]:+,.0f}  Sh={m["sharpe"]:.3f}')
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_title(f'{mode_label}\nCumulative P&L (NC)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1][col]
    for arm_name in arm_names:
        pnls = [t['pnl_eod'] for t in mode_results[arm_name]['trades']]
        if pnls:
            ax.hist(pnls, bins=20, color=arm_colors[arm_name], alpha=0.4,
                    label=f'{arm_name} (n={len(pnls)})')
    ax.axvline(0, color='#333', lw=1.2)
    ax.set_xlabel('EOD PnL (%)')
    ax.set_title(f'{mode_label}\nPnL Distribution')
    ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.90])
p = os.path.join(BASE_DIR, 'agent1_float_comparison.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {p}")

print(f"\n  Total runtime: {time.time() - t_total:.0f}s ({(time.time() - t_total)/60:.1f} min)")
