# %% Agent 1 — E2-A TOP_N Sweep (N=1..5)
#
# Re-evaluates N with current filters:
#   AH freshness (gap vs last AH price on prior day)
#   Gap erosion (skip if >50% consumed by scan time)
#   CS filter, SOFT float, re-rank at 09:31 ET
#
# E2-A only (no sustained_drop filter). LH bars 3-11, fb 15.
# Exit 17:00 ET. Single shared pipeline, build trades per N.
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

BACKTEST_START = '2026-01-02'
BACKTEST_END   = '2026-09-20'

TOP_N_SWEEP    = [1, 2, 3, 4, 5]

GAP_MIN_PCT    = 3.0
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
PRE_FILTER_N   = 30

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

EXIT_TIME      = '17:00'
SCAN_TIME_ET   = '09:31'
GAP_CONSUMED_MAX = 0.50

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
            return tk, {'type': res.get('type'), 'name': res.get('name', ''),
                        'shares': res.get('share_class_shares_outstanding')}
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
# PATTERN + ENTRY + HELPERS
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
                return {'entry_price': bars[ci][1]['c'], 'entry_time': bars[ci][0],
                        'entry_type': 'local_high', 'peak_bar': i}
    fb = max(fallback_bar, min_entry_bar) if fallback_bar is not None else None
    if fb is not None and fb < len(bars):
        return {'entry_price': bars[fb][1]['c'], 'entry_time': bars[fb][0],
                'entry_type': 'fallback', 'peak_bar': None}
    return None

def _T_to_min_entry_bar(T):
    h, m = int(T[:2]), int(T[3:5])
    T_min = h * 60 + m
    rth_start = 9 * 60 + 30
    return max(0, T_min - rth_start)

def get_price_at_T(bar_data, T):
    if bar_data is None: return None
    all_bars = (bar_data.get('pm') or []) + (bar_data.get('rth') or [])
    best = None
    for t_str, bd in all_bars:
        if t_str <= T: best = bd['c']
        else: break
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER
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
        fs = _FLOAT.get(tk)
        if fs is not None and fs < FLOAT_MIN: continue
        candidates.append({'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                           'open_price': cur_p, 'scan_date': dt_str,
                           'volume_today': today[tk].get('v', 0)})
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
# SHARED PIPELINE (runs once)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    t_total = time.time()
    timings = {}

    print(f"\n{'='*80}")
    print(f"  AGENT 1 — E2-A TOP_N SWEEP (N={TOP_N_SWEEP})")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Gap >= {GAP_MIN_PCT}%, price >= ${PRICE_MIN}, CS only, SOFT float")
    print(f"  Gap freshness: vs AH ref (last 16:00-20:00 bar on prior day)")
    print(f"  Gap erosion: skip if >{GAP_CONSUMED_MAX*100:.0f}% consumed by scan time")
    print(f"  Re-rank at {SCAN_TIME_ET} ET | Exit {EXIT_TIME} ET | ${CAPITAL:,.0f}/day NC")
    print(f"{'='*80}")

    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"\n  {n_dates} trading days")

    # Phase 1: Scan
    t1 = time.time()
    print(f"\n  Phase 1: Scanning for gap-ups...")
    all_candidates = {}
    total_cands = 0
    for i, dt in enumerate(dates):
        cands = scan_day(dt)
        all_candidates[dt] = cands
        total_cands += len(cands)
        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_cands} total  |  {time.time()-t_total:.0f}s")
    timings['scan'] = time.time() - t1

    # Phase 2: Fetch minute bars (scan day + prior day for AH ref)
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
        if done % 500 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    timings['bars'] = time.time() - t2
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

    # Phase 2a: AH freshness
    gap_min_final = GAP_MIN_PCT / 100.0
    print(f"\n  Phase 2a: AH freshness filter...")
    n_ah_found = 0
    n_stale = 0
    for dt in dates:
        for c in all_candidates.get(dt, [])[:PRE_FILTER_N]:
            tk = c['ticker']
            prior_str = c.get('prior_str')
            if not prior_str: continue
            prior_bars = all_bars.get((tk, prior_str))
            if prior_bars is None: continue
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
    print(f"  → {n_ah_found} with AH data, {n_stale} stale excluded")

    # Phase 2b: Re-rank + gap erosion
    print(f"\n  Phase 2b: Re-rank at {SCAN_TIME_ET} ET + gap erosion...")
    n_reranked = 0
    n_dropped = 0
    n_eroded = 0
    for dt in dates:
        cands = all_candidates.get(dt, [])
        reranked = []
        for c in cands:
            if c.get('ah_stale'): continue
            tk = c['ticker']
            bar_data = all_bars.get((tk, dt))
            price_T = get_price_at_T(bar_data, SCAN_TIME_ET)
            if price_T is None:
                n_dropped += 1
                continue
            ref_price = c.get('ah_ref_price', c['prior_close'])
            gap_T = (price_T - ref_price) / ref_price
            if gap_T < gap_min_final: continue
            if price_T < PRICE_MIN: continue
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
    print(f"  → {n_reranked} re-ranked ({n_dropped} no bar, {n_stale} stale, {n_eroded} eroded)")

    # Phase 3: Ticker reference (CS filter)
    t3 = time.time()
    all_tickers = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            all_tickers.add(c['ticker'])
    print(f"\n  Phase 3: Reference data for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    for c_start in range(0, len(tk_list), 200):
        chunk = tk_list[c_start:c_start + 200]
        fetch_ticker_refs(chunk)
        done = min(c_start + 200, len(tk_list))
        if done % 400 == 0 or done == len(tk_list):
            print(f"    {done}/{len(tk_list)}  |  {time.time()-t_total:.0f}s")
    timings['refs'] = time.time() - t3
    n_cs = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('type') == 'CS')
    print(f"  → {n_cs} CS / {len(all_tickers)} total")

    # Phase 4: Classify + find entries
    t4 = time.time()
    min_eb = _T_to_min_entry_bar(SCAN_TIME_ET)
    pattern_cache = {}
    entry_cache = {}
    for key, bar_data in all_bars.items():
        rth_bars = bar_data.get('rth', []) if isinstance(bar_data, dict) else bar_data
        pattern_cache[key] = classify_open_pattern(rth_bars)
        if rth_bars and len(rth_bars) >= PEAK_WINDOW * 2 + 1:
            entry_cache[key] = find_local_high(rth_bars, min_entry_bar=min_eb)
        else:
            entry_cache[key] = None
    timings['classify'] = time.time() - t4

    # Phase 5: Outcome dates
    outcome_dates = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            outcome_dates.add(c['scan_date'])
            if c.get('t1_date'): outcome_dates.add(c['t1_date'])
            if c.get('t2_date'): outcome_dates.add(c['t2_date'])
    for od in sorted(od for od in outcome_dates if od not in _gd_cache):
        try: fetch_grouped_daily(od)
        except: pass

    elapsed = time.time() - t_total
    print(f"\n  Pipeline done: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return all_candidates, all_bars, pattern_cache, entry_cache, dates, t_total


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD TRADES (parameterized by top_n)
# ═══════════════════════════════════════════════════════════════════════════════

def build_trades(all_candidates, all_bars, pattern_cache, entry_cache, dates, top_n):
    def _build_trade(c):
        tk, dt = c['ticker'], c['scan_date']
        entry = entry_cache.get((tk, dt))
        if entry is None: return None
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
                if t_str > EXIT_TIME: break
                if t_str >= entry_t: exit_price = bar['c']

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
    for dt in dates:
        cands = all_candidates.get(dt, [])
        cs_qualified = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            ref = _ticker_ref.get(tk, {})
            if ref.get('type') != 'CS': continue
            cc = copy.deepcopy(c)
            cc['asset_type'] = 'CS'
            cc['open_pattern'] = pattern_cache.get((tk, dt), 'no_data')
            cs_qualified.append(cc)
        for c in cs_qualified[:top_n]:
            trade = _build_trade(c)
            if trade and trade.get('pnl_eod') is not None:
                trades.append(trade)
    return trades


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

    active_days = sum(1 for ds in daily_stats if ds['n_trades'] > 0)
    tpd = len(trades) / active_days if active_days > 0 else 0

    eq = [CAPITAL]
    for ds in daily_stats:
        eq.append(eq[-1] + ds['day_pnl_dollar'])
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = float(dd.min())

    return {'label': label, 'trades': trades, 'metrics': m,
            'daily_stats': daily_stats, 'cum_nc': cum_nc,
            'active_days': active_days, 'tpd': tpd, 'max_dd': max_dd}


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

all_candidates, all_bars, pattern_cache, entry_cache, dates, t_total = run_pipeline()

results = {}
for n in TOP_N_SWEEP:
    trades = build_trades(all_candidates, all_bars, pattern_cache, entry_cache, dates, top_n=n)
    res = build_result(trades, dates, f'N={n}')
    results[n] = res
    m = res['metrics']
    print(f"  N={n}: {m['n']} trades, avg={m['avg']:+.2f}%, "
          f"Sharpe={m['sharpe']:.3f}, ${res['cum_nc']:+,.0f} NC")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*110}")
print(f"  E2-A TOP_N SWEEP — {BACKTEST_START} → {BACKTEST_END}")
print(f"  CS only, SOFT float, AH freshness, gap erosion, exit {EXIT_TIME} ET")
print(f"{'='*110}")

rows = []
for n in TOP_N_SWEEP:
    r = results[n]
    m = r['metrics']
    ci_str = f"[{m['ci_lo']:+.1f}, {m['ci_hi']:+.1f}]"
    ci_ok = 'YES' if m['ci_lo'] > 0 else 'NO'
    rows.append([
        f'N={n}',
        f"{m['n']}",
        f"{r['active_days']}",
        f"{r['tpd']:.1f}",
        f"{m['avg']:+.2f}%",
        f"{m['med']:+.2f}%",
        f"{m['trimmed_mean']:+.2f}%",
        f"{m['wr']*100:.0f}%",
        f"{m['sharpe']:.3f}",
        f"{m['pf']:.2f}",
        ci_str,
        ci_ok,
        f"${r['cum_nc']:+,.0f}",
        f"{r['max_dd']:.1f}%",
        f"{m['max_loss']:+.1f}%",
    ])
print()
print(tabulate(rows, headers=['Config', 'Trades', 'Days', 'TPD', 'Avg', 'Med',
                               'Trim', 'WR', 'Sharpe', 'PF', '95% CI', 'CI+',
                               '$ NC', 'MaxDD', 'MaxLoss'],
               tablefmt='simple', stralign='right'))

# Delta table vs N=1
print(f"\n  ── Delta vs N=1 ──")
base = results[1]
bm = base['metrics']
drows = []
for n in TOP_N_SWEEP[1:]:
    r = results[n]
    m = r['metrics']
    drows.append([
        f'N={n}',
        f"{m['n'] - bm['n']:+d}",
        f"{m['avg'] - bm['avg']:+.2f}pp",
        f"{m['wr']*100 - bm['wr']*100:+.0f}pp",
        f"{m['sharpe'] - bm['sharpe']:+.3f}",
        f"{m['pf'] - bm['pf']:+.2f}",
        f"${r['cum_nc'] - base['cum_nc']:+,.0f}",
        f"{r['max_dd'] - base['max_dd']:+.1f}pp",
    ])
print()
print(tabulate(drows, headers=['Config', 'ΔTrades', 'ΔAvg', 'ΔWR', 'ΔSharpe',
                                'ΔPF', 'Δ$NC', 'ΔMaxDD'],
               tablefmt='simple', stralign='right'))

# Entry type breakdown per N
print(f"\n  ── Entry type by N ──")
for n in TOP_N_SWEEP:
    trades = results[n]['trades']
    by_et = defaultdict(list)
    for t in trades:
        by_et[t['entry_type']].append(t['pnl_eod'])
    parts = []
    for et in ['local_high', 'fallback']:
        arr = by_et.get(et, [])
        if arr:
            a = np.array(arr)
            sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
            parts.append(f"{et}={len(a)} avg={np.mean(a):+.1f}% Sh={sh:.2f}")
    print(f"    N={n}:  {' | '.join(parts)}")

# Pattern breakdown per N
print(f"\n  ── Pattern breakdown by N ──")
for n in TOP_N_SWEEP:
    trades = results[n]['trades']
    pat_pnls = defaultdict(list)
    for t in trades:
        pat_pnls[t.get('open_pattern', 'unknown')].append(t['pnl_eod'])
    parts = []
    for pat in ['sustained_drop', 'drop_then_recover', 'mild_drop', 'flat_or_bounce']:
        arr = pat_pnls.get(pat, [])
        if arr:
            parts.append(f"{pat[:8]}={len(arr)}")
    print(f"    N={n}:  {', '.join(parts)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

colors = ['#2c3e50', '#e74c3c', '#27ae60', '#f39c12', '#8e44ad']

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'Agent 1 — E2-A TOP_N Sweep (N=1..{max(TOP_N_SWEEP)})\n'
             f'{BACKTEST_START} → {BACKTEST_END}  |  CS, AH fresh, erosion, exit {EXIT_TIME} ET',
             fontsize=11, fontweight='bold')

# 1. Equity curves
ax = axes[0, 0]
origin = datetime.strptime(BACKTEST_START, '%Y-%m-%d')
for i, n in enumerate(TOP_N_SWEEP):
    ds = results[n]['daily_stats']
    d_dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in ds]
    cum = [s['cum_pnl'] for s in ds]
    ax.plot([origin] + d_dates, [0] + cum, color=colors[i], linewidth=2,
            label=f'N={n}: ${results[n]["cum_nc"]:+,.0f}')
ax.axhline(0, color='#333', lw=0.8, ls=':')
ax.set_title('Cumulative P&L (non-compounding)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Sharpe by N
ax = axes[0, 1]
sharpes = [results[n]['metrics']['sharpe'] for n in TOP_N_SWEEP]
bars = ax.bar([f'N={n}' for n in TOP_N_SWEEP], sharpes, color=colors[:len(TOP_N_SWEEP)], alpha=0.8)
for bar, sh in zip(bars, sharpes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{sh:.3f}', ha='center', va='bottom', fontsize=9)
ax.set_title('Sharpe Ratio by N')
ax.set_ylabel('Sharpe')
ax.grid(True, alpha=0.3, axis='y')

# 3. Avg PnL + CI
ax = axes[1, 0]
avgs = [results[n]['metrics']['avg'] for n in TOP_N_SWEEP]
ci_los = [results[n]['metrics']['ci_lo'] for n in TOP_N_SWEEP]
ci_his = [results[n]['metrics']['ci_hi'] for n in TOP_N_SWEEP]
x = np.arange(len(TOP_N_SWEEP))
ax.bar(x, avgs, color=colors[:len(TOP_N_SWEEP)], alpha=0.8)
ax.errorbar(x, avgs, yerr=[np.array(avgs)-np.array(ci_los), np.array(ci_his)-np.array(avgs)],
            fmt='none', ecolor='black', capsize=5, capthick=1.5)
ax.axhline(0, color='#333', lw=1)
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in TOP_N_SWEEP])
ax.set_title('Avg PnL with 95% CI')
ax.set_ylabel('Avg PnL (%)')
ax.grid(True, alpha=0.3, axis='y')

# 4. $ NC + trade count
ax = axes[1, 1]
ncs = [results[n]['cum_nc'] for n in TOP_N_SWEEP]
trades_n = [results[n]['metrics']['n'] for n in TOP_N_SWEEP]
ax.bar(x - 0.2, ncs, 0.4, color='#2c3e50', alpha=0.8, label='$ NC')
ax.set_ylabel('$ NC', color='#2c3e50')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax2 = ax.twinx()
ax2.bar(x + 0.2, trades_n, 0.4, color='#e74c3c', alpha=0.5, label='Trades')
ax2.set_ylabel('Trade Count', color='#e74c3c')
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in TOP_N_SWEEP])
ax.set_title('Total $ NC vs Trade Count')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.92])
p = os.path.join(BASE_DIR, 'agent1_n_sweep.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {p}")

# Save results JSON
import json
out = {}
for n in TOP_N_SWEEP:
    r = results[n]
    m = r['metrics']
    out[f'N={n}'] = {
        'n': m['n'], 'active_days': r['active_days'], 'tpd': round(r['tpd'], 2),
        'avg_pnl': round(m['avg'], 3), 'median_pnl': round(m['med'], 3),
        'trimmed_mean': round(m['trimmed_mean'], 3),
        'win_rate': round(m['wr'], 3), 'sharpe': round(m['sharpe'], 4),
        'profit_factor': round(m['pf'], 3),
        'ci_95_low': round(m['ci_lo'], 3), 'ci_95_high': round(m['ci_hi'], 3),
        'ci_confirmed': bool(m['ci_lo'] > 0),
        'total_dollar_nc': round(r['cum_nc'], 2), 'max_dd': round(r['max_dd'], 2),
        'max_loss': round(m['max_loss'], 2), 'max_gain': round(m['max_gain'], 2),
    }
jp = os.path.join(BASE_DIR, 'results', 'n_sweep_ah_erosion.json')
os.makedirs(os.path.dirname(jp), exist_ok=True)
with open(jp, 'w') as f:
    json.dump(out, f, indent=2)
print(f"  Saved: {jp}")

print(f"\n  Total runtime: {time.time() - t_total:.0f}s ({(time.time() - t_total)/60:.1f} min)")
