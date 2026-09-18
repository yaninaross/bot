# %% Agent 3 — E1b: T+1 Exit Hour Analysis
#
# A6 gap-down sustained rise — LONG
# Entry: local_low on T+0 (same as baseline)
# Exit: T+1 at each hour from 04:00 to 20:00
# Grid: N = {1,2,3,4,5} × Exit Hour = {04:00..20:00}
#
# Includes premarket (04:00-09:29) and after-hours (16:01-20:00)
# Heatmap: Sharpe, Avg PnL, Pool $, WR
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
A6_PRICE_MIN     = 10.0
A6_PRE_FILTER_N  = 15

LOW_WINDOW      = 3
MAX_LOW_BAR     = 12
FALLBACK_BAR    = 15
SKIP_BOUNCE_PCT = 0.15

FLOAT_MIN       = 10_000_000
CAPITAL         = 100_000
TOTAL_POOL      = 100_000

N_VALUES  = [1, 2, 3, 4, 5]

EXIT_HOURS = ['04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
              '09:30', '10:00', '11:00', '12:00', '13:00', '14:00',
              '15:00', '16:00', '17:00', '18:00', '19:00', '20:00']

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

def _next_td(d):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    d += timedelta(days=1)
    for _ in range(10):
        if _is_td(d): return d
        d += timedelta(days=1)
    raise ValueError(f"No trading day after {d}")

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
# ASYNC MINUTE BAR FETCH — RTH only (for T+0 entry detection)
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_bars_rth_one(session, tk, dt, sem):
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


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC MINUTE BAR FETCH — EXTENDED HOURS (04:00-20:00, for T+1 exit)
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_bars_ext_one(session, tk, dt, sem):
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
            extended = []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                if '04:00' <= t_str < '20:00':
                    extended.append((t_str, {
                        'o': float(b['o']), 'h': float(b['h']),
                        'l': float(b['l']), 'c': float(b['c']),
                        'v': int(b['v'])
                    }))
            return (tk, dt), extended
        return (tk, dt), None


async def _fetch_bars_batch(fetch_fn, pairs):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_fn(session, tk, dt, sem) for tk, dt in pairs]
        results = await asyncio.gather(*tasks)
    return {key: val for key, val in results if val is not None}

def fetch_rth_bars_batch(pairs):
    if not pairs: return {}
    return asyncio.run(_fetch_bars_batch(_fetch_bars_rth_one, pairs))

def fetch_ext_bars_batch(pairs):
    if not pairs: return {}
    return asyncio.run(_fetch_bars_batch(_fetch_bars_ext_one, pairs))


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
# GAP-DOWN PATTERN + LOCAL LOW (same as E1)
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
    print(f"  AGENT 3 — E1b: T+1 EXIT HOUR ANALYSIS (LONG)")
    print(f"  {BACKTEST_START} -> {BACKTEST_END}  |  OOS cutoff: {OOS_CUTOFF}")
    print(f"  A6: gap-down >= {A6_GAP_MIN_PCT}%, sustained_rise, local_low ONLY")
    print(f"  Exit hours: {EXIT_HOURS[0]} → {EXIT_HOURS[-1]} (pre-market + RTH + after-hours)")
    print(f"  Grid: N = {N_VALUES} × Exit Hour = {len(EXIT_HOURS)} hours")
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

    # ── Phase 1: Fetch grouped daily ──
    t1 = time.time()
    print(f"\n  Phase 1: Fetching grouped daily for {len(dates)+8} dates...")
    for i, dt in enumerate(dates):
        fetch_grouped_daily(dt)
        if (i+1) % 50 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {time.time()-t_total:.0f}s")

    # Also fetch prior day for each date (for gap computation)
    prior_dates = set()
    for dt in dates:
        prior_dates.add(_prior_td(dt).strftime('%Y-%m-%d'))
    for pd_str in sorted(prior_dates):
        if pd_str not in _gd_cache:
            fetch_grouped_daily(pd_str)
    timings['daily'] = time.time() - t1

    # ── Phase 2: Scan A6 candidates ──
    t2 = time.time()
    print(f"\n  Phase 2: Scanning A6 candidates...")
    gap_min = A6_GAP_MIN_PCT / 100.0
    a6_candidates = {}
    total_cands = 0

    for i, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = _gd_cache.get(prior_str, {})
        today = _gd_cache.get(dt, {})

        day_cands = []
        for tk in _FLOAT_SET:
            if tk in _EXCLUDE_TK: continue
            if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
            if tk not in prior or tk not in today: continue
            prev_c = prior[tk]['c']
            if prev_c <= 0: continue
            cur_o = today[tk]['o']
            if cur_o <= 0 or cur_o < A6_PRICE_MIN: continue
            gap = (cur_o - prev_c) / prev_c
            if gap > -gap_min: continue
            t1_date = _next_td(dt)
            day_cands.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_o, 'scan_date': dt,
                'volume_today': today[tk]['v'],
                't1_date': t1_date.strftime('%Y-%m-%d'),
            })
        day_cands.sort(key=lambda x: x['gap_pct'])
        a6_candidates[dt] = day_cands
        total_cands += len(day_cands)

        if (i+1) % 50 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  cands={total_cands}  |  {time.time()-t_total:.0f}s")
    timings['scan'] = time.time() - t2

    # ── Phase 3a: Fetch RTH minute bars for T+0 (entry detection) ──
    t3 = time.time()
    t0_pairs = set()
    for dt, cands in a6_candidates.items():
        for c in cands[:A6_PRE_FILTER_N]:
            t0_pairs.add((c['ticker'], dt))

    print(f"\n  Phase 3a: Fetching T+0 RTH bars for {len(t0_pairs)} pairs (entry detection)...")
    t0_bars = {}
    pairs_list = list(t0_pairs)
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start + 100]
        result = fetch_rth_bars_batch(chunk)
        t0_bars.update(result)
        done = min(c_start + 100, len(pairs_list))
        if done % 200 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    timings['t0_bars'] = time.time() - t3
    print(f"  -> Got T+0 bars for {len(t0_bars)}/{len(t0_pairs)} pairs")

    # ── Phase 4: Fetch ticker references ──
    t4 = time.time()
    all_tickers = set()
    for cands in a6_candidates.values():
        for c in cands[:A6_PRE_FILTER_N]:
            all_tickers.add(c['ticker'])
    print(f"\n  Phase 4: Fetching reference data for {len(all_tickers)} tickers...")
    fetch_ticker_refs(sorted(all_tickers))
    timings['refs'] = time.time() - t4

    # ── Phase 5: Classify patterns + find entries ──
    t5 = time.time()
    pattern_cache = {}
    entry_cache = {}
    for key, rth_bars in t0_bars.items():
        pattern_cache[key] = classify_gap_down_pattern(rth_bars)
        if rth_bars and len(rth_bars) >= LOW_WINDOW * 2 + 1:
            entry_cache[key] = find_local_low(rth_bars)
        else:
            entry_cache[key] = None
    timings['classify'] = time.time() - t5

    # ── Phase 6: Build qualified trades (before exit PnL) ──
    t6 = time.time()
    print(f"\n  Phase 6: Building qualified trade list...")
    qualified_trades = []

    for dt in dates:
        cands = a6_candidates.get(dt, [])
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
            tc['asset_type'] = at
            qualified_trades.append(tc)

    print(f"  -> {len(qualified_trades)} qualified A6 trades")
    timings['build'] = time.time() - t6

    # ── Phase 7: Fetch T+1 EXTENDED HOURS bars ──
    t7 = time.time()
    t1_pairs = set()
    for tc in qualified_trades:
        t1_pairs.add((tc['ticker'], tc['t1_date']))

    print(f"\n  Phase 7: Fetching T+1 extended-hours bars for {len(t1_pairs)} pairs...")
    t1_bars = {}
    pairs_list = list(t1_pairs)
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start + 100]
        result = fetch_ext_bars_batch(chunk)
        t1_bars.update(result)
        done = min(c_start + 100, len(pairs_list))
        if done % 200 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    timings['t1_bars'] = time.time() - t7
    print(f"  -> Got T+1 extended bars for {len(t1_bars)}/{len(t1_pairs)} pairs")

    # ── Phase 8: Compute exit PnL at each hour ──
    t8 = time.time()
    print(f"\n  Phase 8: Computing PnL at each exit hour...")

    for tc in qualified_trades:
        ep = tc['entry_price']
        key = (tc['ticker'], tc['t1_date'])
        bars = t1_bars.get(key, [])

        for exit_h in EXIT_HOURS:
            exit_price = None
            for t_str, bar in bars:
                if t_str > exit_h:
                    break
                exit_price = bar['c']
            pk = f'pnl_{exit_h.replace(":", "")}'
            tc[pk] = round((exit_price - ep) / ep * 100, 3) if exit_price is not None else None

    # Also compute T+1 close (16:00) as reference
    for tc in qualified_trades:
        ep = tc['entry_price']
        t1d = _gd_cache.get(tc['t1_date'], {}).get(tc['ticker'], {})
        tc['pnl_t1_close'] = round((t1d['c'] - ep) / ep * 100, 3) if t1d.get('c') else None

    timings['compute'] = time.time() - t8

    elapsed = time.time() - t_total
    print(f"\n  {'─'*50}")
    for step, dur in [('Daily', timings['daily']), ('Scan', timings['scan']),
                      ('T+0 bars', timings['t0_bars']), ('Refs', timings['refs']),
                      ('Classify', timings['classify']), ('Build', timings['build']),
                      ('T+1 ext bars', timings['t1_bars']), ('Compute', timings['compute'])]:
        print(f"    {step:<16s}  {dur:>6.0f}s  ({dur/elapsed*100:>4.0f}%)")
    print(f"  TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return qualified_trades, dates


# ═══════════════════════════════════════════════════════════════════════════════
# GRID ANALYSIS — N × Exit Hour
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_grid(trades, dates):
    # Sort trades per day by gap (most negative first) — same as E1
    daily_trades = defaultdict(list)
    for t in trades:
        daily_trades[t['scan_date']].append(t)
    for dt in daily_trades:
        daily_trades[dt].sort(key=lambda x: x['gap_pct'])

    grid = {}
    for n_val in N_VALUES:
        for exit_h in EXIT_HOURS:
            pk = f'pnl_{exit_h.replace(":", "")}'
            selected = []
            daily_pnl = defaultdict(list)

            for dt in dates:
                day_trades = daily_trades.get(dt, [])[:n_val]
                for t in day_trades:
                    p = t.get(pk)
                    if p is not None:
                        selected.append(t)
                        daily_pnl[dt].append(p)

            pnls = [t[pk] for t in selected]
            m = compute_metrics(pnls, f'N={n_val} @{exit_h}')

            active_days = len(daily_pnl)
            tpd = len(selected) / active_days if active_days > 0 else 0

            # T+1 hold = 2 pools
            slots = 2
            pool_daily = TOTAL_POOL / slots
            cum_adj = 0.0
            for dt in dates:
                dp = daily_pnl.get(dt, [])
                if dp:
                    pos_size = pool_daily / len(dp)
                    cum_adj += sum(pos_size * (r / 100) for r in dp)

            # Raw $ (unlimited)
            cum_raw = 0.0
            for dt in dates:
                dp = daily_pnl.get(dt, [])
                if dp:
                    pos_size = CAPITAL / len(dp)
                    cum_raw += sum(pos_size * (r / 100) for r in dp)

            # IS/OOS
            is_pnls = [t[pk] for t in selected if t['scan_date'] < OOS_CUTOFF]
            oos_pnls = [t[pk] for t in selected if t['scan_date'] >= OOS_CUTOFF]
            is_m = compute_metrics(is_pnls, 'IS')
            oos_m = compute_metrics(oos_pnls, 'OOS')

            grid[(n_val, exit_h)] = {
                'metrics': m, 'tpd': tpd,
                'cum_raw': cum_raw, 'cum_adj': cum_adj,
                'pool_daily': pool_daily, 'slots': slots,
                'ci_confirmed': m['ci_lo'] > 0,
                'active_days': active_days,
                'daily_pnl': dict(daily_pnl),
                'trades': selected,
                'is_metrics': is_m, 'oos_metrics': oos_m,
            }
    return grid


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def _hour_label(h):
    hh = int(h.split(':')[0])
    if hh < 9 or (hh == 9 and h < '09:30'):
        return f'{h} PM'
    elif hh >= 16:
        return f'{h} AH'
    else:
        return f'{h}'

def print_grid(grid):
    print(f"\n\n{'='*120}")
    print(f"  A6 — N×EXIT HOUR GRID (T+1 day, LONG: gap-down sustained rise)")
    print(f"  Entry: local_low on T+0  |  Exit: T+1 at each hour")
    print(f"  Pre-market: 04:00-09:29  |  RTH: 09:30-15:59  |  After-hours: 16:00-20:00")
    print(f"  Pool: 2 slots (T+1 hold) → ${TOTAL_POOL/2:,.0f}/day")
    print(f"{'='*120}")

    hl = [_hour_label(h) for h in EXIT_HOURS]

    def _tbl(title, fn):
        print(f"\n  ── {title} ──")
        headers = [''] + hl
        rows = [[f'N={n}'] + [fn(grid[(n, h)]) for h in EXIT_HOURS]
                for n in N_VALUES]
        print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right'))

    _tbl('SHARPE RATIO',    lambda c: f"{c['metrics']['sharpe']:.3f}")
    _tbl('AVG PnL (%)',     lambda c: f"{c['metrics']['avg']:+.2f}%")
    _tbl('MEDIAN PnL (%)',  lambda c: f"{c['metrics']['med']:+.2f}%")
    _tbl('WIN RATE',        lambda c: f"{c['metrics']['wr']*100:.0f}%")
    _tbl('TRADE COUNT',     lambda c: f"{c['metrics']['n']}")
    _tbl('PROFIT FACTOR',   lambda c: f"{c['metrics']['pf']:.2f}")
    _tbl('95% CI',          lambda c: f"[{c['metrics']['ci_lo']:+.1f},{c['metrics']['ci_hi']:+.1f}]")
    _tbl('CI > 0',          lambda c: '  ✓' if c['ci_confirmed'] else '  ✗')
    _tbl(f'POOL $ ($K)',    lambda c: f"${c['cum_adj']/1000:+.1f}K")
    _tbl('IS SHARPE',       lambda c: f"{c['is_metrics']['sharpe']:.3f}")
    _tbl('OOS SHARPE',      lambda c: f"{c['oos_metrics']['sharpe']:.3f}")

    # Best cells
    best_sharpe = max(grid, key=lambda k: grid[k]['metrics']['sharpe'])
    best_pool = max(grid, key=lambda k: grid[k]['cum_adj'])

    print(f"\n  {'═'*100}")
    print(f"  OPTIMAL EXIT HOUR (T+1)")
    print(f"  {'═'*100}")

    for label, key in [('Best Sharpe', best_sharpe), ('Best Pool $', best_pool)]:
        c = grid[key]
        m = c['metrics']
        print(f"\n  {label}: N={key[0]}, exit@{key[1]} ({_hour_label(key[1])})")
        print(f"    Sharpe={m['sharpe']:.3f}  avg={m['avg']:+.2f}%  WR={m['wr']*100:.0f}%  n={m['n']}")
        print(f"    Pool $: ${c['cum_adj']:+,.0f}  |  Raw $: ${c['cum_raw']:+,.0f}")
        print(f"    IS Sharpe={c['is_metrics']['sharpe']:.3f}  OOS Sharpe={c['oos_metrics']['sharpe']:.3f}")

    # E1 reference: N=2 T+1 close = Sharpe 0.600
    ref_key = (2, '16:00')
    if ref_key in grid:
        c = grid[ref_key]
        m = c['metrics']
        print(f"\n  E1 reference (N=2 T+1 close @16:00): Sharpe={m['sharpe']:.3f}  avg={m['avg']:+.2f}%")

    # Per-N best hour
    print(f"\n  ── BEST EXIT HOUR PER N ──")
    for n in N_VALUES:
        best_h = max(EXIT_HOURS, key=lambda h: grid[(n, h)]['metrics']['sharpe'])
        c = grid[(n, best_h)]
        m = c['metrics']
        print(f"    N={n}: {best_h} ({_hour_label(best_h)})  Sharpe={m['sharpe']:.3f}  "
              f"avg={m['avg']:+.2f}%  WR={m['wr']*100:.0f}%  n={m['n']}")

    # Sharpe trend per N
    print(f"\n  ── SHARPE BY EXIT HOUR (per N) ──")
    for n in N_VALUES:
        sharpes = [grid[(n, h)]['metrics']['sharpe'] for h in EXIT_HOURS]
        parts = [f"{s:.2f}" for s in sharpes]
        print(f"    N={n}: {'→'.join(parts)}")

    # Coverage: how many trades have data at each hour
    print(f"\n  ── DATA COVERAGE (trades with price at exit hour) ──")
    for n in N_VALUES:
        counts = [grid[(n, h)]['metrics']['n'] for h in EXIT_HOURS]
        max_n = max(counts) if counts else 1
        parts = [f"{c}" for c in counts]
        print(f"    N={n}: {' | '.join(parts)}")

    print()
    return best_sharpe, best_pool


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmaps(grid):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    fig, axes = plt.subplots(2, 2, figsize=(24, 10))
    fig.suptitle(f'A6 — T+1 Exit Hour Grid (LONG: gap-down sustained rise)\n'
                 f'Entry: local_low on T+0  |  Exit: T+1 at each hour  |  '
                 f'Pre-market | RTH | After-hours',
                 fontsize=12, fontweight='bold')

    panels = [
        ('Sharpe Ratio', lambda c: c['metrics']['sharpe'], '.3f', 'RdYlGn'),
        ('Avg PnL (%)', lambda c: c['metrics']['avg'], '+.1f', 'RdYlGn'),
        ('Pool $ ($K)', lambda c: c['cum_adj'] / 1000, '+.0f', 'RdYlGn'),
        ('Win Rate (%)', lambda c: c['metrics']['wr'] * 100, '.0f', 'RdYlGn'),
    ]

    best_pool = max(grid, key=lambda k: grid[k]['cum_adj'])

    for idx, (title, fn, fmt, cmap) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        data = np.array([[fn(grid[(n, h)]) for h in EXIT_HOURS] for n in N_VALUES])

        im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(EXIT_HOURS)))
        xlabels = []
        for h in EXIT_HOURS:
            hh = int(h.split(':')[0])
            mm = h.split(':')[1]
            xlabels.append(f'{hh}:{mm}')
        ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(len(N_VALUES)))
        ax.set_yticklabels([f'N={n}' for n in N_VALUES], fontsize=10)

        for i in range(len(N_VALUES)):
            for j in range(len(EXIT_HOURS)):
                val = data[i, j]
                is_best = (N_VALUES[i], EXIT_HOURS[j]) == best_pool
                weight = 'bold' if is_best else 'normal'
                color = 'white' if abs(val - data.mean()) > data.std() * 0.8 else 'black'
                ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                        fontsize=6, fontweight=weight, color=color)
                if is_best:
                    rect = plt.Rectangle((j-0.48, i-0.48), 0.96, 0.96,
                                         linewidth=3, edgecolor='gold', facecolor='none')
                    ax.add_patch(rect)

        # Mark session boundaries
        pm_end = EXIT_HOURS.index('09:30') - 0.5
        ah_start = EXIT_HOURS.index('16:00') - 0.5
        ax.axvline(pm_end, color='white', lw=2, ls='--', alpha=0.7)
        ax.axvline(ah_start, color='white', lw=2, ls='--', alpha=0.7)

        ax.set_title(title, fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(BASE_DIR, 'agent3_e1b_heatmaps.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p}")


def plot_curves(grid):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
    fig.suptitle('A6 — T+1 Exit Hour Curves', fontsize=12, fontweight='bold')

    x_pos = list(range(len(EXIT_HOURS)))

    for panel_idx, (title, fn, ylabel) in enumerate([
        ('Sharpe', lambda c: c['metrics']['sharpe'], 'Sharpe'),
        ('Avg PnL', lambda c: c['metrics']['avg'], 'Avg PnL (%)'),
        ('Pool $K', lambda c: c['cum_adj']/1000, 'Pool-Adjusted ($K)')
    ]):
        ax = axes[panel_idx]
        for i, n in enumerate(N_VALUES):
            vals = [fn(grid[(n, h)]) for h in EXIT_HOURS]
            ax.plot(x_pos, vals, 'o-', color=colors[i],
                    linewidth=2, markersize=5, label=f'N={n}')
            best_idx = int(np.argmax(vals))
            ax.plot(best_idx, vals[best_idx], '*', color=colors[i], markersize=14, zorder=5)

        ax.axhline(0, color='#333', lw=0.8, ls=':')

        # Session boundaries
        pm_end = EXIT_HOURS.index('09:30') - 0.5
        ah_start = EXIT_HOURS.index('16:00') - 0.5
        ax.axvline(pm_end, color='gray', lw=1, ls='--', alpha=0.5)
        ax.axvline(ah_start, color='gray', lw=1, ls='--', alpha=0.5)
        ax.text(pm_end/2, ax.get_ylim()[1]*0.95, 'PRE', ha='center', fontsize=8, color='gray')
        rth_mid = (pm_end + ah_start) / 2
        ax.text(rth_mid, ax.get_ylim()[1]*0.95, 'RTH', ha='center', fontsize=8, color='gray')
        ah_mid = (ah_start + len(EXIT_HOURS) - 1) / 2
        ax.text(ah_mid, ax.get_ylim()[1]*0.95, 'AH', ha='center', fontsize=8, color='gray')

        ax.set_xticks(x_pos)
        xlabels = [f'{int(h.split(":")[0])}:{h.split(":")[1]}' for h in EXIT_HOURS]
        ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(BASE_DIR, 'agent3_e1b_curves.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p}")


def plot_is_oos(grid):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    # Pick N=2 (best from E1) for IS vs OOS comparison
    n_show = 2
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle(f'A6 N={n_show} — IS vs OOS by T+1 Exit Hour', fontsize=12, fontweight='bold')

    x = np.arange(len(EXIT_HOURS))
    w = 0.35

    for panel_idx, (title, fn) in enumerate([
        ('Avg PnL (%)', lambda c: c['avg']),
        ('Sharpe', lambda c: c['sharpe'])
    ]):
        ax = axes[panel_idx]
        is_vals = [fn(grid[(n_show, h)]['is_metrics']) for h in EXIT_HOURS]
        oos_vals = [fn(grid[(n_show, h)]['oos_metrics']) for h in EXIT_HOURS]
        ax.bar(x - w/2, is_vals, w, color='#3498db', alpha=0.8, label='IS')
        ax.bar(x + w/2, oos_vals, w, color='#e67e22', alpha=0.8, label='OOS')
        ax.set_xticks(x)
        xlabels = [f'{int(h.split(":")[0])}:{h.split(":")[1]}' for h in EXIT_HOURS]
        ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha='right')
        ax.axhline(0, color='#333', lw=0.8)

        pm_end = EXIT_HOURS.index('09:30') - 0.5
        ah_start = EXIT_HOURS.index('16:00') - 0.5
        ax.axvline(pm_end, color='gray', lw=1, ls='--', alpha=0.5)
        ax.axvline(ah_start, color='gray', lw=1, ls='--', alpha=0.5)

        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(BASE_DIR, 'agent3_e1b_isoos.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

trades, dates = run_backtest()
grid = analyze_grid(trades, dates)
best_sharpe, best_pool = print_grid(grid)
plot_heatmaps(grid)
plot_curves(grid)
plot_is_oos(grid)
