# %% Agent 1 — E2: N×Hold Grid + Capital Efficiency + Intraday Exit + Gap Impact (CS+SD)
#
# Part 1: 5×4 grid — N = {1,2,3,4,5} × T = {T+2,T+3,T+4,T+5}
# Part 2: Capital efficiency — N × {EOD,T+1,...,T+5} with pool-adjusted $
# Part 3: Intraday exit timing — N × {10:00..20:00} hourly (RTH + after-hours)
# Part 4: E3b Gap size impact — 6 buckets, deployed config (N=1, exit@17:00)
#
# Applied to CS+SD (forward benchmark from E1)
# Capital model: fixed pool split by hold period
#   EOD → full pool/day | T+2 → pool/3 per day | T+5 → pool/6 per day
#   Intraday exits → same as EOD (full pool, 1 bucket)
#
# Colab-cell friendly. ~8-10 min runtime (API-bound).

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
BACKTEST_END   = '2026-09-12'

GAP_MIN_PCT    = 5.0
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
PRE_FILTER_N   = 15

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

CAPITAL        = 100_000
TOTAL_POOL     = 100_000

N_VALUES       = [1, 2, 3, 4, 5]
HOLD_DAYS      = [2, 3, 4, 5]
MAX_HOLD       = max(HOLD_DAYS)

INTRADAY_EXITS = ['10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
                  '16:00', '17:00', '18:00', '19:00', '20:00']

GAP_BUCKETS = [
    ('5-8%',   0.05, 0.08),
    ('8-12%',  0.08, 0.12),
    ('12-20%', 0.12, 0.20),
    ('20-30%', 0.20, 0.30),
    ('30-50%', 0.30, 0.50),
    ('50%+',   0.50, 999.0),
]

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
# ASYNC MINUTE BAR FETCH (RTH only)
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
            rth = []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                if '09:30' <= t_str < '20:00':
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
            return tk, {
                'type': res.get('type'),
                'name': res.get('name', ''),
                'shares': res.get('weighted_shares_outstanding'),
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
# OPEN PATTERN CLASSIFICATION
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
                    fallback_bar=FALLBACK_BAR):
    if len(bars) < window * 2 + 1:
        return None
    for i in range(window, min(max_peak_bar, len(bars) - window)):
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
    if fallback_bar is not None and fallback_bar < len(bars):
        return {
            'entry_price': bars[fallback_bar][1]['c'],
            'entry_time': bars[fallback_bar][0],
            'entry_type': 'fallback',
            'peak_bar': None,
        }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

def scan_day(dt_str):
    if not _is_td(dt_str): return []
    prior_str = _prior_td(dt_str).strftime('%Y-%m-%d')
    prior = fetch_grouped_daily(prior_str)
    today = fetch_grouped_daily(dt_str)

    gap_min = GAP_MIN_PCT / 100.0
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
            'volume_today': today[tk]['v'],
        })

    candidates.sort(key=lambda x: -x['gap_pct'])

    nxt = _next_n_td(dt_str, MAX_HOLD)
    for s in candidates:
        for i in range(len(nxt)):
            s[f't{i+1}_date'] = nxt[i].strftime('%Y-%m-%d')
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
#  BACKTEST ENGINE — DATA FETCH + CS+SD TRADE BUILDING
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest():
    t_total = time.time()
    timings = {}

    print(f"\n{'='*80}")
    print(f"  AGENT 1 — E2: N×HOLD GRID (CS+SD FORWARD BENCHMARK)")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Grid: N = {N_VALUES} × T = {{T+{',T+'.join(str(h) for h in HOLD_DAYS)}}}")
    print(f"  Filter: CS + sustained_drop (from PRE_FILTER_N={PRE_FILTER_N})")
    print(f"  Entry: local_high + fallback  |  Capital: ${CAPITAL:,.0f}/day NC")
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

    # ── Phase 2: Fetch minute bars for top PRE_FILTER_N per day ──
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
    timings['bars'] = time.time() - t2
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

    # ── Phase 3: Fetch ticker references ──
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
    print(f"  → {n_cs} CS out of {len(all_tickers)} tickers")

    # ── Phase 4: Classify patterns + find entries ──
    t4 = time.time()
    pattern_cache = {}
    entry_cache = {}
    for key, rth_bars in all_bars.items():
        pattern_cache[key] = classify_open_pattern(rth_bars)
        if rth_bars and len(rth_bars) >= PEAK_WINDOW * 2 + 1:
            entry_cache[key] = find_local_high(rth_bars)
        else:
            entry_cache[key] = None
    timings['classify'] = time.time() - t4

    pat_counts = defaultdict(int)
    for p in pattern_cache.values():
        pat_counts[p] += 1
    print(f"\n  Phase 4: Pattern classification")
    for pat in ['sustained_drop', 'drop_then_recover', 'mild_drop', 'flat_or_bounce', 'no_data']:
        cnt = pat_counts.get(pat, 0)
        if pattern_cache:
            print(f"    {pat:25s}  {cnt:>5d}  ({cnt/len(pattern_cache)*100:.1f}%)")

    # ── Phase 5: Fetch ALL outcome dates (up to T+5) ──
    t5 = time.time()
    print(f"\n  Phase 5: Fetching outcome dates (up to T+{MAX_HOLD})...")
    outcome_dates = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            outcome_dates.add(c['scan_date'])
            for i in range(1, MAX_HOLD + 1):
                d = c.get(f't{i}_date')
                if d: outcome_dates.add(d)
    to_fetch = sorted(od for od in outcome_dates if od not in _gd_cache)
    for od in to_fetch:
        try: fetch_grouped_daily(od)
        except: pass
    timings['outcomes'] = time.time() - t5
    print(f"  → Fetched {len(to_fetch)} additional dates ({len(_gd_cache)} total cached)")

    # ── Phase 6: Build CS+SD qualified trades (full list per day) ──
    t6 = time.time()
    print(f"\n  Phase 6: Building CS+SD qualified trades...")

    cssd_daily = {}
    total_qualified = 0
    entry_stats = {'lh': 0, 'fb': 0, 'ne': 0, 'nb': 0}

    for dt in dates:
        cands = all_candidates.get(dt, [])
        qualified = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            key = (tk, dt)
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS':
                continue
            pat = pattern_cache.get(key, 'no_data')
            if pat != 'sustained_drop':
                continue

            entry = entry_cache.get(key)
            if entry is None:
                if key not in all_bars:
                    entry_stats['nb'] += 1
                else:
                    entry_stats['ne'] += 1
                continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = entry['entry_price']
            tc['entry_time'] = entry['entry_time']
            tc['entry_type'] = entry['entry_type']
            tc['open_pattern'] = pat
            tc['asset_type'] = at

            ep = tc['entry_price']
            t0d = _gd_cache.get(dt, {}).get(tk, {})
            tc['pnl_eod'] = round((ep - t0d['c']) / ep * 100, 3) if t0d.get('c') else None
            for i in range(1, MAX_HOLD + 1):
                td = _gd_cache.get(tc.get(f't{i}_date', ''), {}).get(tk, {})
                tc[f'pnl_t{i}'] = round((ep - td['c']) / ep * 100, 3) if td.get('c') else None

            bars = all_bars.get(key, [])
            entry_t = tc['entry_time']
            for exit_h in INTRADAY_EXITS:
                exit_price = None
                for t_str, bar in bars:
                    if t_str > exit_h:
                        break
                    if t_str >= entry_t:
                        exit_price = bar['c']
                pk_h = f'pnl_{exit_h.replace(":", "")}'
                tc[pk_h] = round((ep - exit_price) / ep * 100, 3) if exit_price is not None else None

            entry_stats['lh' if tc['entry_type'] == 'local_high' else 'fb'] += 1
            qualified.append(tc)

        cssd_daily[dt] = qualified
        total_qualified += len(qualified)

    timings['build'] = time.time() - t6
    elapsed = time.time() - t_total

    days_with_trades = sum(1 for dt in dates if cssd_daily.get(dt))
    print(f"  → {total_qualified} total CS+SD qualified trades across {days_with_trades} active days")
    print(f"    local_high: {entry_stats['lh']}  |  fallback: {entry_stats['fb']}  |  "
          f"no_entry: {entry_stats['ne']}  |  no_bars: {entry_stats['nb']}")
    print(f"    Avg qualified/day: {total_qualified/days_with_trades:.2f}" if days_with_trades else '')

    print(f"\n  {'─'*50}")
    for step, dur in [('Scan', timings['scan']), ('Bars', timings['bars']),
                      ('References', timings['refs']), ('Outcomes', timings['outcomes']),
                      ('Build', timings['build'])]:
        print(f"    {step:<20s}  {dur:>6.0f}s  ({dur/elapsed*100:>4.0f}%)")
    print(f"  TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return cssd_daily, dates


# ═══════════════════════════════════════════════════════════════════════════════
# GRID ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_grid(cssd_daily, dates):
    grid = {}

    for n_val in N_VALUES:
        for hold in HOLD_DAYS:
            pnl_key = f'pnl_t{hold}'
            all_trades = []
            daily_pnl = defaultdict(list)

            for dt in dates:
                day_trades = cssd_daily.get(dt, [])[:n_val]
                for t in day_trades:
                    p = t.get(pnl_key)
                    if p is not None:
                        all_trades.append(t)
                        daily_pnl[dt].append(p)

            pnls = [t[pnl_key] for t in all_trades]
            m = compute_metrics(pnls, f'N={n_val} T+{hold}')

            active_days = len(daily_pnl)
            tpd = len(all_trades) / active_days if active_days > 0 else 0

            cum_nc = 0.0
            for dt in dates:
                dp = daily_pnl.get(dt, [])
                if dp:
                    pos_size = CAPITAL / len(dp)
                    cum_nc += sum(pos_size * (r / 100) for r in dp)

            grid[(n_val, hold)] = {
                'metrics': m, 'tpd': tpd, 'cum_nc': cum_nc,
                'ci_confirmed': m['ci_lo'] > 0,
                'trades': all_trades,
                'active_days': active_days,
                'daily_pnl': dict(daily_pnl),
            }

    return grid


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT GRID RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def print_grid(grid, dates):
    print(f"\n\n{'='*90}")
    print(f"  E2: N×HOLD GRID — CS+SD FORWARD BENCHMARK")
    print(f"  {dates[0]} → {dates[-1]}  |  ${CAPITAL:,.0f}/day NC")
    print(f"{'='*90}")

    hold_labels = [f'T+{h}' for h in HOLD_DAYS]

    def _grid_table(title, fn, fmt=''):
        print(f"\n  ── {title} ──")
        headers = [''] + hold_labels
        rows = []
        for n in N_VALUES:
            row = [f'N={n}']
            for h in HOLD_DAYS:
                val = fn(grid[(n, h)])
                row.append(val)
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right'))

    _grid_table('SHARPE RATIO',
                lambda c: f"{c['metrics']['sharpe']:.3f}")

    _grid_table('AVG PnL (%)',
                lambda c: f"{c['metrics']['avg']:+.2f}%")

    _grid_table('MEDIAN PnL (%)',
                lambda c: f"{c['metrics']['med']:+.2f}%")

    _grid_table('TRIMMED MEAN (%)',
                lambda c: f"{c['metrics']['trimmed_mean']:+.2f}%")

    _grid_table('WIN RATE',
                lambda c: f"{c['metrics']['wr']*100:.0f}%")

    _grid_table('PROFIT FACTOR',
                lambda c: f"{c['metrics']['pf']:.2f}")

    _grid_table('TRADE COUNT',
                lambda c: f"{c['metrics']['n']}")

    _grid_table('TRADES/DAY',
                lambda c: f"{c['tpd']:.2f}")

    _grid_table('DOLLAR PnL (NC)',
                lambda c: f"${c['cum_nc']:+,.0f}")

    _grid_table('95% CI',
                lambda c: f"[{c['metrics']['ci_lo']:+.1f}, {c['metrics']['ci_hi']:+.1f}]")

    _grid_table('CI CONFIRMED',
                lambda c: '  ✓' if c['ci_confirmed'] else '  ✗')

    # ── Best cell by Sharpe ──
    best_key = max(grid, key=lambda k: grid[k]['metrics']['sharpe'])
    best = grid[best_key]
    print(f"\n  {'═'*70}")
    print(f"  BEST CELL: N={best_key[0]}, T+{best_key[1]}")
    print(f"    Sharpe {best['metrics']['sharpe']:.3f}  |  "
          f"Avg {best['metrics']['avg']:+.2f}%  |  "
          f"WR {best['metrics']['wr']*100:.0f}%  |  "
          f"n={best['metrics']['n']}  |  "
          f"${best['cum_nc']:+,.0f}")
    print(f"    CI [{best['metrics']['ci_lo']:+.2f}, {best['metrics']['ci_hi']:+.2f}]  "
          f"{'✓ CONFIRMED' if best['ci_confirmed'] else '✗ NOT CONFIRMED'}")

    # ── Validation check ──
    ref = grid.get((5, 2))
    if ref:
        print(f"\n  VALIDATION: N=5 T+2 Sharpe = {ref['metrics']['sharpe']:.3f} "
              f"(expected ~0.255 from E1)")

    # ── Best by dollar PnL ──
    best_dollar = max(grid, key=lambda k: grid[k]['cum_nc'])
    bd = grid[best_dollar]
    print(f"\n  BEST $ CELL: N={best_dollar[0]}, T+{best_dollar[1]}  "
          f"${bd['cum_nc']:+,.0f}  (Sharpe {bd['metrics']['sharpe']:.3f})")

    # ── Detail per cell ──
    print(f"\n  ── DETAIL PER CELL ──")
    for n in N_VALUES:
        for h in HOLD_DAYS:
            c = grid[(n, h)]
            m = c['metrics']
            ci = '✓' if c['ci_confirmed'] else '✗'
            print(f"    N={n} T+{h}:  {m['n']:>4d}t  avg={m['avg']:+7.2f}%  "
                  f"med={m['med']:+7.2f}%  WR={m['wr']*100:>2.0f}%  "
                  f"Sh={m['sharpe']:.3f}  PF={m['pf']:.2f}  "
                  f"TPD={c['tpd']:.2f}  ${c['cum_nc']:>+10,.0f}  CI {ci}")

    # ── Hold period monotonicity check ──
    print(f"\n  ── HOLD PERIOD TREND (Sharpe by T, per N) ──")
    for n in N_VALUES:
        sharpes = [grid[(n, h)]['metrics']['sharpe'] for h in HOLD_DAYS]
        trend = '→'.join(f'{s:.3f}' for s in sharpes)
        best_h = HOLD_DAYS[np.argmax(sharpes)]
        print(f"    N={n}: {trend}  (best: T+{best_h})")

    # ── N monotonicity check ──
    print(f"\n  ── N TREND (Sharpe by N, per T) ──")
    for h in HOLD_DAYS:
        sharpes = [grid[(n, h)]['metrics']['sharpe'] for n in N_VALUES]
        trend = '→'.join(f'{s:.3f}' for s in sharpes)
        best_n = N_VALUES[np.argmax(sharpes)]
        print(f"    T+{h}: {trend}  (best: N={best_n})")

    # ── Top 5 cells by Sharpe ──
    print(f"\n  ── TOP 5 CELLS BY SHARPE ──")
    ranked = sorted(grid.items(), key=lambda x: -x[1]['metrics']['sharpe'])
    for i, ((n, h), c) in enumerate(ranked[:5]):
        m = c['metrics']
        print(f"    #{i+1}  N={n} T+{h}  Sharpe={m['sharpe']:.3f}  "
              f"avg={m['avg']:+.2f}%  WR={m['wr']*100:.0f}%  "
              f"n={m['n']}  ${c['cum_nc']:+,.0f}")

    # ── Worst trades in best cell ──
    best_trades = sorted(grid[best_key]['trades'],
                         key=lambda t: t[f'pnl_t{best_key[1]}'])
    pnl_key = f'pnl_t{best_key[1]}'
    print(f"\n  ── WORST 5 TRADES — BEST CELL N={best_key[0]} T+{best_key[1]} ──")
    for t in best_trades[:5]:
        print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={t['gap_pct']*100:>6.1f}%  "
              f"entry={t['entry_type']:>10s}  pnl={t[pnl_key]:>+8.2f}%")

    print(f"\n  ── BEST 5 TRADES — BEST CELL N={best_key[0]} T+{best_key[1]} ──")
    for t in best_trades[-5:]:
        print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={t['gap_pct']*100:>6.1f}%  "
              f"entry={t['entry_type']:>10s}  pnl={t[pnl_key]:>+8.2f}%")

    print()
    return best_key


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_grid(grid, dates, best_key):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    # ── Figure 1: 2×2 Heatmaps ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'E2: N×Hold Grid — CS+SD\n{dates[0]} → {dates[-1]}',
                 fontsize=12, fontweight='bold')

    panels = [
        ('Sharpe Ratio', lambda c: c['metrics']['sharpe'], '.3f', 'RdYlGn'),
        ('Avg PnL (%)', lambda c: c['metrics']['avg'], '+.1f', 'RdYlGn'),
        ('Dollar PnL ($K, NC)', lambda c: c['cum_nc'] / 1000, '+.0f', 'RdYlGn'),
        ('Win Rate (%)', lambda c: c['metrics']['wr'] * 100, '.0f', 'RdYlGn'),
    ]

    for idx, (title, fn, fmt, cmap) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        data = np.array([[fn(grid[(n, h)]) for h in HOLD_DAYS] for n in N_VALUES])

        im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(HOLD_DAYS)))
        ax.set_xticklabels([f'T+{h}' for h in HOLD_DAYS], fontsize=11)
        ax.set_yticks(range(len(N_VALUES)))
        ax.set_yticklabels([f'N={n}' for n in N_VALUES], fontsize=11)

        for i in range(len(N_VALUES)):
            for j in range(len(HOLD_DAYS)):
                val = data[i, j]
                txt = f'{val:{fmt}}'
                is_best = (N_VALUES[i], HOLD_DAYS[j]) == best_key
                weight = 'bold' if is_best else 'normal'
                color = 'white' if abs(val - data.mean()) > data.std() * 0.8 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=10, fontweight=weight, color=color)
                if is_best:
                    rect = plt.Rectangle((j-0.48, i-0.48), 0.96, 0.96,
                                         linewidth=3, edgecolor='gold', facecolor='none')
                    ax.add_patch(rect)

        ax.set_title(title, fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p1 = os.path.join(BASE_DIR, 'agent1_e2_heatmaps.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")

    # ── Figure 2: Hold period curves (Sharpe + Avg PnL by T, one line per N) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']

    ax = axes[0]
    for i, n in enumerate(N_VALUES):
        sharpes = [grid[(n, h)]['metrics']['sharpe'] for h in HOLD_DAYS]
        ax.plot(HOLD_DAYS, sharpes, 'o-', color=colors[i], linewidth=2,
                markersize=8, label=f'N={n}')
        best_h_idx = np.argmax(sharpes)
        ax.plot(HOLD_DAYS[best_h_idx], sharpes[best_h_idx], '*',
                color=colors[i], markersize=16, zorder=5)
    ax.set_xlabel('Hold Period (T+X)', fontsize=11)
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.set_title('Sharpe vs Hold Period', fontsize=12, fontweight='bold')
    ax.set_xticks(HOLD_DAYS)
    ax.set_xticklabels([f'T+{h}' for h in HOLD_DAYS])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, n in enumerate(N_VALUES):
        avgs = [grid[(n, h)]['metrics']['avg'] for h in HOLD_DAYS]
        ax.plot(HOLD_DAYS, avgs, 'o-', color=colors[i], linewidth=2,
                markersize=8, label=f'N={n}')
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_xlabel('Hold Period (T+X)', fontsize=11)
    ax.set_ylabel('Avg PnL (%)', fontsize=11)
    ax.set_title('Avg PnL vs Hold Period', fontsize=12, fontweight='bold')
    ax.set_xticks(HOLD_DAYS)
    ax.set_xticklabels([f'T+{h}' for h in HOLD_DAYS])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p2 = os.path.join(BASE_DIR, 'agent1_e2_hold_curves.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p2}")

    # ── Figure 3: Best cell equity vs N=5 T+2 baseline ──
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    comparisons = [(best_key, '#e74c3c', 'Best'), ((5, 2), '#2c3e50', 'N=5 T+2 (E1 ref)')]
    if best_key != (5, 2) and best_key not in [(5, 2)]:
        pass
    else:
        best_dollar = max(grid, key=lambda k: grid[k]['cum_nc'])
        if best_dollar != best_key:
            comparisons.append((best_dollar, '#27ae60', f'Best $ (N={best_dollar[0]} T+{best_dollar[1]})'))

    for (n, h), color, lbl in comparisons:
        c = grid[(n, h)]
        cum = 0.0
        cum_series = [0.0]
        d_dates = [datetime.strptime(dates[0], '%Y-%m-%d')]
        for dt in dates:
            dp = c['daily_pnl'].get(dt, [])
            if dp:
                pos_size = CAPITAL / len(dp)
                cum += sum(pos_size * (r / 100) for r in dp)
            cum_series.append(cum)
            d_dates.append(datetime.strptime(dt, '%Y-%m-%d'))

        m = c['metrics']
        ax.plot(d_dates, cum_series, color=color, linewidth=2,
                label=f'N={n} T+{h}: ${cum:+,.0f} Sh={m["sharpe"]:.3f} ({lbl})')

    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_title('Cumulative P&L — Best Cell vs E1 Reference', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = os.path.join(BASE_DIR, 'agent1_e2_equity.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p3}")


# ═══════════════════════════════════════════════════════════════════════════════
# CAPITAL EFFICIENCY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Key insight: with a fixed capital pool, longer holds tie up more capital.
#   Hold EOD → 1 pool  → deploy $TOTAL_POOL/day
#   Hold T+2 → 3 pools → deploy $TOTAL_POOL/3 per day
#   Hold T+5 → 6 pools → deploy $TOTAL_POOL/6 per day
#
# A trade with higher avg PnL at T+5 may be WORSE than T+2 if the
# capital reduction (1/6 vs 1/3) exceeds the per-trade PnL gain.

ALL_HOLDS = [0, 1, 2, 3, 4, 5]

def _pnl_key(h):
    return 'pnl_eod' if h == 0 else f'pnl_t{h}'

def _hold_label(h):
    return 'EOD' if h == 0 else f'T+{h}'


def capital_efficiency_analysis(cssd_daily, dates):
    print(f"\n\n{'='*100}")
    print(f"  CAPITAL EFFICIENCY ANALYSIS")
    print(f"  Pool: ${TOTAL_POOL:,.0f}  |  Split into (hold+1) daily buckets")
    print(f"  EOD → 1 bucket → ${TOTAL_POOL:,.0f}/day")
    print(f"  T+2 → 3 buckets → ${TOTAL_POOL/3:,.0f}/day")
    print(f"  T+5 → 6 buckets → ${TOTAL_POOL/6:,.0f}/day")
    print(f"{'='*100}")

    cap_grid = {}
    for n_val in N_VALUES:
        for hold in ALL_HOLDS:
            pk = _pnl_key(hold)
            slots = hold + 1
            pool_daily = TOTAL_POOL / slots

            all_trades = []
            daily_pnl = defaultdict(list)
            for dt in dates:
                for t in cssd_daily.get(dt, [])[:n_val]:
                    p = t.get(pk)
                    if p is not None:
                        all_trades.append(t)
                        daily_pnl[dt].append(p)

            pnls = [t[pk] for t in all_trades]
            m = compute_metrics(pnls, f'N={n_val} {_hold_label(hold)}')

            active_days = len(daily_pnl)
            tpd = len(all_trades) / active_days if active_days > 0 else 0

            cum_raw = 0.0
            cum_adj = 0.0
            for dt in dates:
                dp = daily_pnl.get(dt, [])
                if dp:
                    n_t = len(dp)
                    cum_raw += sum((CAPITAL / n_t) * (r / 100) for r in dp)
                    cum_adj += sum((pool_daily / n_t) * (r / 100) for r in dp)

            cap_grid[(n_val, hold)] = {
                'metrics': m, 'tpd': tpd,
                'cum_raw': cum_raw, 'cum_adj': cum_adj,
                'pool_daily': pool_daily, 'slots': slots,
                'ci_confirmed': m['ci_lo'] > 0,
                'active_days': active_days,
                'daily_pnl': dict(daily_pnl),
                'trades': all_trades,
            }

    # ── Grid tables ──
    hl = [_hold_label(h) for h in ALL_HOLDS]

    def _tbl(title, fn):
        print(f"\n  ── {title} ──")
        headers = [''] + hl
        rows = [[f'N={n}'] + [fn(cap_grid[(n, h)]) for h in ALL_HOLDS]
                for n in N_VALUES]
        print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right'))

    _tbl('SHARPE', lambda c: f"{c['metrics']['sharpe']:.3f}")
    _tbl('AVG PnL (%)', lambda c: f"{c['metrics']['avg']:+.2f}%")
    _tbl('MEDIAN PnL (%)', lambda c: f"{c['metrics']['med']:+.2f}%")
    _tbl('WIN RATE', lambda c: f"{c['metrics']['wr']*100:.0f}%")
    _tbl('PROFIT FACTOR', lambda c: f"{c['metrics']['pf']:.2f}")
    _tbl('TRADE COUNT', lambda c: f"{c['metrics']['n']}")

    _tbl(f'RAW $ (${CAPITAL/1000:.0f}K/day unlimited)',
         lambda c: f"${c['cum_raw']:+,.0f}")

    _tbl(f'POOL-ADJUSTED $ (${TOTAL_POOL/1000:.0f}K pool)',
         lambda c: f"${c['cum_adj']:+,.0f}")

    _tbl('DAILY ALLOCATION',
         lambda c: f"${c['pool_daily']:,.0f}")

    _tbl('ROI ON POOL (%)',
         lambda c: f"{c['cum_adj']/TOTAL_POOL*100:+.1f}%")

    _tbl('CI CONFIRMED',
         lambda c: '  ✓' if c['ci_confirmed'] else '  ✗')

    # ── Best cells ──
    best_sharpe = max(cap_grid, key=lambda k: cap_grid[k]['metrics']['sharpe'])
    best_raw = max(cap_grid, key=lambda k: cap_grid[k]['cum_raw'])
    best_pool = max(cap_grid, key=lambda k: cap_grid[k]['cum_adj'])

    print(f"\n  {'═'*80}")
    print(f"  ANSWER: OPTIMAL EXIT TIMING (${TOTAL_POOL/1000:.0f}K pool)")
    print(f"  {'═'*80}")

    for label, key in [('Best Sharpe (risk-adjusted quality)', best_sharpe),
                       ('Best Raw $ (unlimited capital)', best_raw),
                       ('Best Pool $ (FIXED CAPITAL — YOUR QUESTION)', best_pool)]:
        c = cap_grid[key]
        m = c['metrics']
        print(f"\n  {label}:")
        print(f"    → N={key[0]}, {_hold_label(key[1])}")
        print(f"      Sharpe={m['sharpe']:.3f}  avg={m['avg']:+.2f}%  "
              f"WR={m['wr']*100:.0f}%  n={m['n']}")
        print(f"      Daily alloc: ${c['pool_daily']:,.0f}/day ({c['slots']} pools)")
        print(f"      Raw $: ${c['cum_raw']:+,.0f}  |  Pool $: ${c['cum_adj']:+,.0f}")
        print(f"      ROI: {c['cum_adj']/TOTAL_POOL*100:+.1f}% on ${TOTAL_POOL/1000:.0f}K")

    ref = cap_grid.get((5, 2))
    if ref:
        m = ref['metrics']
        print(f"\n  Current benchmark (N=5, T+2):")
        print(f"      Sharpe={m['sharpe']:.3f}  avg={m['avg']:+.2f}%  "
              f"WR={m['wr']*100:.0f}%  n={m['n']}")
        print(f"      Daily alloc: ${ref['pool_daily']:,.0f}/day ({ref['slots']} pools)")
        print(f"      Raw $: ${ref['cum_raw']:+,.0f}  |  Pool $: ${ref['cum_adj']:+,.0f}")
        print(f"      ROI: {ref['cum_adj']/TOTAL_POOL*100:+.1f}% on ${TOTAL_POOL/1000:.0f}K")

    # ── All cells ranked by pool $ ──
    print(f"\n  ── ALL 30 CELLS RANKED BY POOL $ ──")
    ranked = sorted(cap_grid.items(), key=lambda x: -x[1]['cum_adj'])
    for i, ((n, h), c) in enumerate(ranked):
        m = c['metrics']
        ci = '✓' if c['ci_confirmed'] else '✗'
        marker = ' ★' if (n, h) == best_pool else ''
        print(f"    #{i+1:>2d}  N={n} {_hold_label(h):>4s}  "
              f"Sh={m['sharpe']:.3f}  avg={m['avg']:+7.2f}%  WR={m['wr']*100:>2.0f}%  "
              f"n={m['n']:>4d}  alloc=${c['pool_daily']:>7,.0f}  "
              f"pool$={c['cum_adj']:>+10,.0f}  CI {ci}{marker}")

    # ── Hold period trend per N ──
    print(f"\n  ── POOL $ BY HOLD PERIOD (per N) ──")
    for n in N_VALUES:
        vals = [cap_grid[(n, h)]['cum_adj'] for h in ALL_HOLDS]
        best_idx = int(np.argmax(vals))
        parts = [f'{_hold_label(ALL_HOLDS[i])}=${vals[i]/1000:+.0f}K' for i in range(len(ALL_HOLDS))]
        print(f"    N={n}: {'  '.join(parts)}  (best: {_hold_label(ALL_HOLDS[best_idx])})")

    # ── N trend per hold period ──
    print(f"\n  ── POOL $ BY N (per hold period) ──")
    for h in ALL_HOLDS:
        vals = [cap_grid[(n, h)]['cum_adj'] for n in N_VALUES]
        best_idx = int(np.argmax(vals))
        parts = [f'N={N_VALUES[i]}=${vals[i]/1000:+.0f}K' for i in range(len(N_VALUES))]
        print(f"    {_hold_label(h):>4s}: {'  '.join(parts)}  (best: N={N_VALUES[best_idx]})")

    print()
    return cap_grid


def plot_capital_efficiency(cap_grid, dates):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    best_pool = max(cap_grid, key=lambda k: cap_grid[k]['cum_adj'])

    # ── Figure 1: 2×2 Heatmaps ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'E2: Capital Efficiency — CS+SD\n'
                 f'{dates[0]} → {dates[-1]}  |  '
                 f'Pool: ${TOTAL_POOL/1000:.0f}K',
                 fontsize=12, fontweight='bold')

    panels = [
        ('Sharpe Ratio', lambda c: c['metrics']['sharpe'], '.3f'),
        ('Avg PnL (%)', lambda c: c['metrics']['avg'], '+.1f'),
        (f'Pool-Adjusted $ ($K)', lambda c: c['cum_adj'] / 1000, '+.0f'),
        ('Win Rate (%)', lambda c: c['metrics']['wr'] * 100, '.0f'),
    ]

    for idx, (title, fn, fmt) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        data = np.array([[fn(cap_grid[(n, h)]) for h in ALL_HOLDS] for n in N_VALUES])
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(ALL_HOLDS)))
        ax.set_xticklabels([_hold_label(h) for h in ALL_HOLDS], fontsize=10)
        ax.set_yticks(range(len(N_VALUES)))
        ax.set_yticklabels([f'N={n}' for n in N_VALUES], fontsize=10)
        for i in range(len(N_VALUES)):
            for j in range(len(ALL_HOLDS)):
                val = data[i, j]
                is_best = (N_VALUES[i], ALL_HOLDS[j]) == best_pool
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
    p1 = os.path.join(BASE_DIR, 'agent1_e2_capital_heatmaps.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")

    # ── Figure 2: Raw $ vs Pool-Adjusted $ curves ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']

    ax = axes[0]
    for i, n in enumerate(N_VALUES):
        raw_vals = [cap_grid[(n, h)]['cum_raw'] / 1000 for h in ALL_HOLDS]
        ax.plot([_hold_label(h) for h in ALL_HOLDS], raw_vals, 'o-',
                color=colors[i], linewidth=2, markersize=8, label=f'N={n}')
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_ylabel('Dollar PnL ($K)')
    ax.set_title(f'Raw $ (${CAPITAL/1000:.0f}K/day unlimited)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, n in enumerate(N_VALUES):
        adj_vals = [cap_grid[(n, h)]['cum_adj'] / 1000 for h in ALL_HOLDS]
        ax.plot([_hold_label(h) for h in ALL_HOLDS], adj_vals, 'o-',
                color=colors[i], linewidth=2, markersize=8, label=f'N={n}')
        best_idx = int(np.argmax(adj_vals))
        ax.plot(_hold_label(ALL_HOLDS[best_idx]), adj_vals[best_idx], '*',
                color=colors[i], markersize=16, zorder=5)
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_ylabel('Dollar PnL ($K)')
    ax.set_title(f'Pool-Adjusted $ (${TOTAL_POOL/1000:.0f}K pool)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p2 = os.path.join(BASE_DIR, 'agent1_e2_capital_curves.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p2}")

    # ── Figure 3: Pool-adjusted equity — best vs benchmark ──
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    best_raw = max(cap_grid, key=lambda k: cap_grid[k]['cum_raw'])
    comparisons = [
        (best_pool, '#e74c3c',
         f'Best Pool (N={best_pool[0]} {_hold_label(best_pool[1])})'),
        ((5, 2), '#2c3e50', 'N=5 T+2 (current benchmark)'),
    ]
    if best_raw != best_pool and best_raw != (5, 2):
        comparisons.append(
            (best_raw, '#27ae60',
             f'Best Raw (N={best_raw[0]} {_hold_label(best_raw[1])})'))

    for (n, h), color, lbl in comparisons:
        c = cap_grid[(n, h)]
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
    ax.set_title(f'Pool-Adjusted Equity — ${TOTAL_POOL/1000:.0f}K Capital Pool',
                 fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = os.path.join(BASE_DIR, 'agent1_e2_capital_equity.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p3}")


# ═══════════════════════════════════════════════════════════════════════════════
# INTRADAY EXIT TIMING ANALYSIS (Part 3)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Break down the intraday period into hourly exit times (10:00–20:00)
# including after-hours. All same-day exits → full pool deployed daily.

def intraday_exit_analysis(cssd_daily, dates):
    print(f"\n\n{'='*100}")
    print(f"  INTRADAY EXIT TIMING — 1-HOUR GRANULARITY (RTH + AFTER-HOURS)")
    print(f"  Pool: ${TOTAL_POOL:,.0f}  |  All intraday exits → full pool daily")
    print(f"  Exit times: {', '.join(INTRADAY_EXITS)}")
    print(f"  16:00 = market close  |  17:00+ = after-hours")
    print(f"{'='*100}")

    intra_grid = {}

    for n_val in N_VALUES:
        for exit_h in INTRADAY_EXITS:
            pk = f'pnl_{exit_h.replace(":", "")}'

            all_trades = []
            daily_pnl = defaultdict(list)

            for dt in dates:
                for t in cssd_daily.get(dt, [])[:n_val]:
                    p = t.get(pk)
                    if p is not None:
                        all_trades.append(t)
                        daily_pnl[dt].append(p)

            pnls = [t[pk] for t in all_trades]
            m = compute_metrics(pnls, f'N={n_val} exit@{exit_h}')

            active_days = len(daily_pnl)
            tpd = len(all_trades) / active_days if active_days > 0 else 0

            cum = 0.0
            for dt in dates:
                dp = daily_pnl.get(dt, [])
                if dp:
                    pos_size = TOTAL_POOL / len(dp)
                    cum += sum(pos_size * (r / 100) for r in dp)

            intra_grid[(n_val, exit_h)] = {
                'metrics': m, 'tpd': tpd, 'cum_pool': cum,
                'ci_confirmed': m['ci_lo'] > 0,
                'active_days': active_days,
                'daily_pnl': dict(daily_pnl),
            }

    hl = INTRADAY_EXITS

    def _tbl(title, fn):
        print(f"\n  ── {title} ──")
        headers = [''] + hl
        rows = [[f'N={n}'] + [fn(intra_grid[(n, h)]) for h in INTRADAY_EXITS]
                for n in N_VALUES]
        print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right'))

    _tbl('SHARPE', lambda c: f"{c['metrics']['sharpe']:.3f}")
    _tbl('AVG PnL (%)', lambda c: f"{c['metrics']['avg']:+.2f}%")
    _tbl('MEDIAN PnL (%)', lambda c: f"{c['metrics']['med']:+.2f}%")
    _tbl('WIN RATE', lambda c: f"{c['metrics']['wr']*100:.0f}%")
    _tbl('PROFIT FACTOR', lambda c: f"{c['metrics']['pf']:.2f}")
    _tbl('TRADE COUNT', lambda c: f"{c['metrics']['n']}")
    _tbl(f'POOL $ (${TOTAL_POOL/1000:.0f}K pool)', lambda c: f"${c['cum_pool']:+,.0f}")
    _tbl('ROI (%)', lambda c: f"{c['cum_pool']/TOTAL_POOL*100:+.1f}%")
    _tbl('CI CONFIRMED', lambda c: '  ✓' if c['ci_confirmed'] else '  ✗')

    # ── Optimal exit per N (by Pool $) ──
    print(f"\n  ── OPTIMAL EXIT TIME (by Pool $, per N) ──")
    for n in N_VALUES:
        vals = [(h, intra_grid[(n, h)]['cum_pool']) for h in INTRADAY_EXITS]
        best = max(vals, key=lambda x: x[1])
        c = intra_grid[(n, best[0])]
        m = c['metrics']
        print(f"    N={n}: best exit = {best[0]}  pool$=${best[1]:+,.0f}  "
              f"Sh={m['sharpe']:.3f}  avg={m['avg']:+.2f}%  WR={m['wr']*100:.0f}%")
        parts = [f'{h}=${v/1000:+.0f}K' for h, v in vals]
        print(f"           {'  '.join(parts)}")

    # ── Optimal exit per N (by Sharpe) ──
    print(f"\n  ── OPTIMAL EXIT TIME (by Sharpe, per N) ──")
    for n in N_VALUES:
        vals = [(h, intra_grid[(n, h)]['metrics']['sharpe']) for h in INTRADAY_EXITS]
        best = max(vals, key=lambda x: x[1])
        print(f"    N={n}: best exit = {best[0]}  Sharpe={best[1]:.3f}")

    # ── Overall best ──
    best_key = max(intra_grid, key=lambda k: intra_grid[k]['cum_pool'])
    c = intra_grid[best_key]
    m = c['metrics']
    print(f"\n  {'═'*80}")
    print(f"  BEST INTRADAY EXIT: N={best_key[0]}, exit at {best_key[1]}")
    print(f"    Sharpe={m['sharpe']:.3f}  avg={m['avg']:+.2f}%  "
          f"WR={m['wr']*100:.0f}%  n={m['n']}")
    print(f"    Pool $: ${c['cum_pool']:+,.0f}  |  "
          f"ROI: {c['cum_pool']/TOTAL_POOL*100:+.1f}%")
    print(f"    CI [{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]  "
          f"{'✓ CONFIRMED' if c['ci_confirmed'] else '✗ NOT CONFIRMED'}")

    # ── After-hours data availability ──
    print(f"\n  ── DATA AVAILABILITY BY EXIT TIME (N=1) ──")
    n1_total = max(intra_grid[(1, h)]['metrics']['n'] for h in INTRADAY_EXITS)
    for h in INTRADAY_EXITS:
        n_h = intra_grid[(1, h)]['metrics']['n']
        pct = n_h / n1_total * 100 if n1_total > 0 else 0
        marker = '  ← market close' if h == '16:00' else ''
        ah = ' (AH)' if h > '16:00' else ''
        print(f"    {h}{ah}: {n_h:>4d} trades ({pct:>5.1f}%){marker}")

    # ── Worst 5 trades at best exit time ──
    pk_best = f'pnl_{best_key[1].replace(":", "")}'
    best_trades_list = []
    for dt in dates:
        for t in cssd_daily.get(dt, [])[:best_key[0]]:
            p = t.get(pk_best)
            if p is not None:
                best_trades_list.append(t)
    best_trades_list.sort(key=lambda t: t[pk_best])
    print(f"\n  ── WORST 5 TRADES — N={best_key[0]} exit@{best_key[1]} ──")
    for t in best_trades_list[:5]:
        print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={t['gap_pct']*100:>6.1f}%  "
              f"entry={t['entry_type']:>10s}  pnl={t[pk_best]:>+8.2f}%")
    print(f"\n  ── BEST 5 TRADES — N={best_key[0]} exit@{best_key[1]} ──")
    for t in best_trades_list[-5:]:
        print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={t['gap_pct']*100:>6.1f}%  "
              f"entry={t['entry_type']:>10s}  pnl={t[pk_best]:>+8.2f}%")

    print()
    return intra_grid


def plot_intraday_exit(intra_grid, dates):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
    x_pos = range(len(INTRADAY_EXITS))
    close_idx = INTRADAY_EXITS.index('16:00')

    # ── Figure 1: 3-panel curves ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Intraday Exit Timing — CS+SD (RTH + After-Hours)\n'
                 f'{dates[0]} → {dates[-1]}  |  Pool: ${TOTAL_POOL/1000:.0f}K',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    for i, n in enumerate(N_VALUES):
        sharpes = [intra_grid[(n, h)]['metrics']['sharpe'] for h in INTRADAY_EXITS]
        ax.plot(x_pos, sharpes, 'o-', color=colors[i], linewidth=2,
                markersize=8, label=f'N={n}')
        best_idx = int(np.argmax(sharpes))
        ax.plot(best_idx, sharpes[best_idx], '*', color=colors[i], markersize=16, zorder=5)
    ax.axvline(x=close_idx, color='gray', ls='--', lw=1, alpha=0.6, label='Close')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(INTRADAY_EXITS, rotation=45, fontsize=9)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe by Exit Time', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, n in enumerate(N_VALUES):
        avgs = [intra_grid[(n, h)]['metrics']['avg'] for h in INTRADAY_EXITS]
        ax.plot(x_pos, avgs, 'o-', color=colors[i], linewidth=2,
                markersize=8, label=f'N={n}')
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.axvline(x=close_idx, color='gray', ls='--', lw=1, alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(INTRADAY_EXITS, rotation=45, fontsize=9)
    ax.set_ylabel('Avg PnL (%)')
    ax.set_title('Avg PnL by Exit Time', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for i, n in enumerate(N_VALUES):
        vals = [intra_grid[(n, h)]['cum_pool'] / 1000 for h in INTRADAY_EXITS]
        ax.plot(x_pos, vals, 'o-', color=colors[i], linewidth=2,
                markersize=8, label=f'N={n}')
        best_idx = int(np.argmax(vals))
        ax.plot(best_idx, vals[best_idx], '*', color=colors[i], markersize=16, zorder=5)
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.axvline(x=close_idx, color='gray', ls='--', lw=1, alpha=0.6, label='Close')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(INTRADAY_EXITS, rotation=45, fontsize=9)
    ax.set_ylabel('Pool-Adjusted PnL ($K)')
    ax.set_title(f'Pool $ by Exit Time (${TOTAL_POOL/1000:.0f}K)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    p1 = os.path.join(BASE_DIR, 'agent1_e2_intraday_exit.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")

    # ── Figure 2: Equity curves — best intraday exit vs 16:00 ──
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    best_key = max(intra_grid, key=lambda k: intra_grid[k]['cum_pool'])
    best_sharpe = max(intra_grid, key=lambda k: intra_grid[k]['metrics']['sharpe'])

    comparisons = [
        (best_key, '#e74c3c', f'Best Pool$ (N={best_key[0]} @{best_key[1]})'),
    ]
    if (1, '16:00') != best_key:
        comparisons.append(((1, '16:00'), '#2c3e50', 'N=1 @16:00 (close)'))
    if best_sharpe != best_key and best_sharpe != (1, '16:00'):
        comparisons.append(
            (best_sharpe, '#27ae60',
             f'Best Sharpe (N={best_sharpe[0]} @{best_sharpe[1]})'))

    for (n, h), color, lbl in comparisons:
        c = intra_grid.get((n, h))
        if c is None:
            continue
        cum = 0.0
        cum_series = [0.0]
        d_dates = [datetime.strptime(dates[0], '%Y-%m-%d')]
        for dt in dates:
            dp = c['daily_pnl'].get(dt, [])
            if dp:
                pos_size = TOTAL_POOL / len(dp)
                cum += sum(pos_size * (r / 100) for r in dp)
            cum_series.append(cum)
            d_dates.append(datetime.strptime(dt, '%Y-%m-%d'))
        m = c['metrics']
        ax.plot(d_dates, cum_series, color=color, linewidth=2,
                label=f'{lbl}: ${cum:+,.0f} Sh={m["sharpe"]:.3f}')

    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_title('Intraday Exit Equity — Best vs Market Close',
                 fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(BASE_DIR, 'agent1_e2_intraday_equity.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p2}")

    # ── Figure 3: Heatmap N × exit time ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle('Intraday Exit Heatmaps — CS+SD', fontsize=12, fontweight='bold')

    best_pool_key = best_key
    for idx, (title, fn, fmt) in enumerate([
        ('Sharpe Ratio', lambda c: c['metrics']['sharpe'], '.3f'),
        (f'Pool $ ($K)', lambda c: c['cum_pool'] / 1000, '+.0f'),
    ]):
        ax = axes[idx]
        data = np.array([[fn(intra_grid[(n, h)]) for h in INTRADAY_EXITS]
                         for n in N_VALUES])
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(INTRADAY_EXITS)))
        ax.set_xticklabels(INTRADAY_EXITS, fontsize=9, rotation=45)
        ax.set_yticks(range(len(N_VALUES)))
        ax.set_yticklabels([f'N={n}' for n in N_VALUES], fontsize=10)
        for i in range(len(N_VALUES)):
            for j in range(len(INTRADAY_EXITS)):
                val = data[i, j]
                is_best = (N_VALUES[i], INTRADAY_EXITS[j]) == best_pool_key
                weight = 'bold' if is_best else 'normal'
                color = 'white' if abs(val - data.mean()) > data.std() * 0.8 else 'black'
                ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                        fontsize=8, fontweight=weight, color=color)
                if is_best:
                    rect = plt.Rectangle((j-0.48, i-0.48), 0.96, 0.96,
                                         linewidth=3, edgecolor='gold', facecolor='none')
                    ax.add_patch(rect)
        ax.set_title(title, fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p3 = os.path.join(BASE_DIR, 'agent1_e2_intraday_heatmaps.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p3}")


# ═══════════════════════════════════════════════════════════════════════════════
# E3b: GAP SIZE IMPACT ON PnL (Part 4)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Deployed config: N=1, exit@17:00. Question: does "biggest gap = best SHORT"
# hold, or is there a sweet spot gap range that should change ranking?

def gap_impact_analysis(cssd_daily, dates):
    pk = 'pnl_1700'

    print(f"\n\n{'='*100}")
    print(f"  E3b: GAP SIZE IMPACT ON PnL")
    print(f"  Config: N=1, exit@17:00 (deployed strategy)")
    print(f"  Buckets: {', '.join(b[0] for b in GAP_BUCKETS)}")
    print(f"  Pool: ${TOTAL_POOL:,.0f}")
    print(f"{'='*100}")

    all_trades = []
    for dt in dates:
        for t in cssd_daily.get(dt, [])[:1]:
            p = t.get(pk)
            if p is not None:
                all_trades.append(t)

    print(f"\n  Total N=1 trades with 17:00 exit data: {len(all_trades)}")

    bucket_results = {}
    for label, lo, hi in GAP_BUCKETS:
        trades = [t for t in all_trades if lo <= t['gap_pct'] < hi]
        pnls = [t[pk] for t in trades]
        m = compute_metrics(pnls, label)

        worst = min(trades, key=lambda t: t[pk]) if trades else None
        best = max(trades, key=lambda t: t[pk]) if trades else None

        cum = 0.0
        daily_pnl = defaultdict(list)
        for t in trades:
            daily_pnl[t['scan_date']].append(t[pk])
        for dt in dates:
            dp = daily_pnl.get(dt, [])
            if dp:
                pos_size = TOTAL_POOL / len(dp)
                cum += sum(pos_size * (r / 100) for r in dp)

        bucket_results[label] = {
            'metrics': m, 'trades': trades, 'cum_pool': cum,
            'worst': worst, 'best': best,
            'gap_lo': lo, 'gap_hi': hi,
            'active_days': len(daily_pnl),
        }

    # ── Summary table ──
    print(f"\n  ── METRICS BY GAP BUCKET ──")
    headers = ['Bucket', 'n', 'Avg PnL', 'Median', 'Trim Mean', 'Sharpe',
               'WR', 'PF', 'CI lo', 'CI hi', 'CI?', 'Pool $']
    rows = []
    for label, _, _ in GAP_BUCKETS:
        r = bucket_results[label]
        m = r['metrics']
        rows.append([
            label, m['n'], f"{m['avg']:+.2f}%", f"{m['med']:+.2f}%",
            f"{m['trimmed_mean']:+.2f}%", f"{m['sharpe']:.3f}",
            f"{m['wr']*100:.0f}%", f"{m['pf']:.2f}",
            f"{m['ci_lo']:+.2f}", f"{m['ci_hi']:+.2f}",
            '✓' if m['ci_lo'] > 0 else '✗',
            f"${r['cum_pool']:+,.0f}",
        ])
    print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right'))

    # ── Worst/best per bucket ──
    print(f"\n  ── WORST & BEST TRADE PER BUCKET ──")
    for label, _, _ in GAP_BUCKETS:
        r = bucket_results[label]
        if r['worst']:
            w = r['worst']
            b = r['best']
            print(f"    {label:>8s}:  worst = {w['ticker']:>6s} {w['scan_date']} "
                  f"gap={w['gap_pct']*100:>6.1f}% pnl={w[pk]:>+8.2f}%  |  "
                  f"best = {b['ticker']:>6s} {b['scan_date']} "
                  f"gap={b['gap_pct']*100:>6.1f}% pnl={b[pk]:>+8.2f}%")
        else:
            print(f"    {label:>8s}:  no trades")

    # ── Monotonicity test ──
    print(f"\n  ── MONOTONICITY TEST ──")
    print(f"  Question: does bigger gap → better SHORT PnL?")
    avgs = []
    sharpes = []
    for label, _, _ in GAP_BUCKETS:
        m = bucket_results[label]['metrics']
        avgs.append(m['avg'])
        sharpes.append(m['sharpe'])
    avg_diffs = [avgs[i+1] - avgs[i] for i in range(len(avgs)-1)]
    sharpe_diffs = [sharpes[i+1] - sharpes[i] for i in range(len(sharpes)-1)]
    avg_mono = all(d > 0 for d in avg_diffs)
    avg_anti = all(d < 0 for d in avg_diffs)
    sharpe_mono = all(d > 0 for d in sharpe_diffs)
    sharpe_anti = all(d < 0 for d in sharpe_diffs)

    labels = [b[0] for b in GAP_BUCKETS]
    print(f"    Avg PnL trend:  {'→'.join(f'{a:+.1f}' for a in avgs)}")
    if avg_mono:
        print(f"    → MONOTONIC INCREASING: bigger gap = better fade ✓")
    elif avg_anti:
        print(f"    → MONOTONIC DECREASING: bigger gap = worse fade ✗")
    else:
        best_bucket_idx = int(np.argmax(avgs))
        print(f"    → NON-MONOTONIC: sweet spot at {labels[best_bucket_idx]} "
              f"({avgs[best_bucket_idx]:+.2f}%)")

    print(f"    Sharpe trend:   {'→'.join(f'{s:.3f}' for s in sharpes)}")
    if sharpe_mono:
        print(f"    → MONOTONIC INCREASING: bigger gap = more reliable ✓")
    elif sharpe_anti:
        print(f"    → MONOTONIC DECREASING: bigger gap = less reliable ✗")
    else:
        best_sharpe_idx = int(np.argmax(sharpes))
        print(f"    → NON-MONOTONIC: Sharpe peak at {labels[best_sharpe_idx]} "
              f"({sharpes[best_sharpe_idx]:.3f})")

    # ── Correlation: gap_pct vs PnL ──
    gap_vals = np.array([t['gap_pct'] * 100 for t in all_trades])
    pnl_vals = np.array([t[pk] for t in all_trades])
    if len(gap_vals) > 2:
        corr = np.corrcoef(gap_vals, pnl_vals)[0, 1]
        print(f"\n    Correlation (gap% vs PnL): r = {corr:+.3f}")
        if abs(corr) < 0.1:
            print(f"    → WEAK: gap size has little linear effect on PnL")
        elif corr > 0.1:
            print(f"    → POSITIVE: bigger gaps fade MORE (supports current ranking)")
        else:
            print(f"    → NEGATIVE: bigger gaps fade LESS (ranking may be wrong)")

    # ── What-if: filter to optimal bucket only ──
    print(f"\n  ── WHAT-IF: TRADE ONLY OPTIMAL BUCKET ──")
    best_sharpe_bucket = max(bucket_results.items(),
                             key=lambda x: x[1]['metrics']['sharpe'] if x[1]['metrics']['n'] >= 10 else -999)
    best_dollar_bucket = max(bucket_results.items(),
                             key=lambda x: x[1]['cum_pool'] if x[1]['metrics']['n'] >= 10 else -999)

    for lbl_type, (blabel, br) in [('Best Sharpe bucket', best_sharpe_bucket),
                                    ('Best Pool$ bucket', best_dollar_bucket)]:
        m = br['metrics']
        print(f"\n    {lbl_type}: {blabel}")
        print(f"      n={m['n']}  avg={m['avg']:+.2f}%  Sharpe={m['sharpe']:.3f}  "
              f"WR={m['wr']*100:.0f}%  PF={m['pf']:.2f}")
        print(f"      Pool $: ${br['cum_pool']:+,.0f}  |  "
              f"CI [{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}] "
              f"{'✓' if m['ci_lo'] > 0 else '✗'}")
        print(f"      Active days: {br['active_days']} / {len(dates)} "
              f"({br['active_days']/len(dates)*100:.0f}%)")

    # ── All trades (no bucket restriction) baseline ──
    m_all = compute_metrics([t[pk] for t in all_trades], 'All')
    cum_all = 0.0
    daily_all = defaultdict(list)
    for t in all_trades:
        daily_all[t['scan_date']].append(t[pk])
    for dt in dates:
        dp = daily_all.get(dt, [])
        if dp:
            cum_all += sum((TOTAL_POOL / len(dp)) * (r / 100) for r in dp)

    print(f"\n    Baseline (all gaps, N=1 @17:00):")
    print(f"      n={m_all['n']}  avg={m_all['avg']:+.2f}%  "
          f"Sharpe={m_all['sharpe']:.3f}  WR={m_all['wr']*100:.0f}%  "
          f"Pool $: ${cum_all:+,.0f}")

    # ── Should ranking change? ──
    print(f"\n  {'═'*80}")
    print(f"  VERDICT: SHOULD RANKING CHANGE?")
    print(f"  {'═'*80}")

    best_avg_idx = int(np.argmax(avgs))
    best_sharpe_idx = int(np.argmax(sharpes))
    biggest_is_best_avg = best_avg_idx == len(avgs) - 1
    biggest_is_best_sharpe = best_sharpe_idx == len(sharpes) - 1

    if biggest_is_best_avg and biggest_is_best_sharpe:
        print(f"  → KEEP CURRENT RANKING: biggest gap wins on BOTH avg PnL and Sharpe")
    elif biggest_is_best_avg or biggest_is_best_sharpe:
        winner = 'avg PnL' if biggest_is_best_avg else 'Sharpe'
        other = 'Sharpe' if biggest_is_best_avg else 'avg PnL'
        other_best = labels[best_sharpe_idx] if biggest_is_best_avg else labels[best_avg_idx]
        print(f"  → MIXED: biggest gap wins on {winner}, "
              f"but {other_best} wins on {other}")
        print(f"    Consider: keep biggest-gap ranking but ADD a gap filter "
              f"to exclude weaker buckets")
    else:
        print(f"  → CONSIDER RE-RANKING: biggest gap is NOT best on either metric")
        print(f"    Best avg PnL: {labels[best_avg_idx]} ({avgs[best_avg_idx]:+.2f}%)")
        print(f"    Best Sharpe:  {labels[best_sharpe_idx]} ({sharpes[best_sharpe_idx]:.3f})")
        print(f"    BUT: restricting to one bucket reduces trade frequency drastically")

    # ── Dollar contribution: which buckets make/lose money? ──
    print(f"\n  ── DOLLAR CONTRIBUTION BY BUCKET ──")
    total_pool = sum(r['cum_pool'] for r in bucket_results.values())
    for label, _, _ in GAP_BUCKETS:
        r = bucket_results[label]
        pct = r['cum_pool'] / total_pool * 100 if total_pool != 0 else 0
        bar = '█' * max(1, int(abs(pct) / 3))
        sign = '+' if r['cum_pool'] >= 0 else '-'
        print(f"    {label:>8s}:  ${r['cum_pool']:>+10,.0f}  "
              f"({pct:>+5.1f}%)  {bar}")

    print()
    return bucket_results


def plot_gap_impact(bucket_results, all_trades_data, dates):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    pk = 'pnl_1700'
    labels = [b[0] for b in GAP_BUCKETS]
    x = np.arange(len(labels))

    # ── Figure 1: 3-panel — Avg PnL, Sharpe, Pool $ by bucket ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'E3b: Gap Size Impact — CS+SD, N=1, exit@17:00\n'
                 f'{dates[0]} → {dates[-1]}',
                 fontsize=12, fontweight='bold')

    avgs = [bucket_results[l]['metrics']['avg'] for l in labels]
    ci_los = [bucket_results[l]['metrics']['ci_lo'] for l in labels]
    ci_his = [bucket_results[l]['metrics']['ci_hi'] for l in labels]
    errs_lo = [a - c for a, c in zip(avgs, ci_los)]
    errs_hi = [c - a for a, c in zip(avgs, ci_his)]

    ax = axes[0]
    colors_bar = ['#e74c3c' if a < 0 else '#27ae60' for a in avgs]
    ax.bar(x, avgs, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.errorbar(x, avgs, yerr=[errs_lo, errs_hi], fmt='none', ecolor='black',
                capsize=4, linewidth=1.5)
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30)
    ax.set_ylabel('Avg PnL (%)')
    ax.set_title('Avg PnL by Gap Bucket', fontweight='bold')
    for i, v in enumerate(avgs):
        n = bucket_results[labels[i]]['metrics']['n']
        ax.text(i, v + (max(avgs) * 0.05 if v >= 0 else min(avgs) * 0.05),
                f'n={n}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    sharpes = [bucket_results[l]['metrics']['sharpe'] for l in labels]
    ax = axes[1]
    colors_sh = ['#e74c3c' if s < 0.2 else '#f39c12' if s < 0.5 else '#27ae60'
                 for s in sharpes]
    ax.bar(x, sharpes, color=colors_sh, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe by Gap Bucket', fontweight='bold')
    best_sh_idx = int(np.argmax(sharpes))
    ax.bar(best_sh_idx, sharpes[best_sh_idx], color='gold', alpha=0.9,
           edgecolor='black', linewidth=1.5)
    for i, v in enumerate(sharpes):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    pools = [bucket_results[l]['cum_pool'] / 1000 for l in labels]
    ax = axes[2]
    colors_p = ['#e74c3c' if p < 0 else '#27ae60' for p in pools]
    ax.bar(x, pools, color=colors_p, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30)
    ax.set_ylabel('Pool PnL ($K)')
    ax.set_title(f'Pool $ by Gap Bucket (${TOTAL_POOL/1000:.0f}K)', fontweight='bold')
    for i, v in enumerate(pools):
        ax.text(i, v + (max(pools) * 0.03), f'${v:+.0f}K', ha='center',
                va='bottom', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    p1 = os.path.join(BASE_DIR, 'agent1_e3b_gap_buckets.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")

    # ── Figure 2: Scatter — gap_pct vs PnL with trend line ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Gap Size vs PnL — Individual Trades', fontsize=12, fontweight='bold')

    gap_vals = np.array([t['gap_pct'] * 100 for t in all_trades_data])
    pnl_vals = np.array([t[pk] for t in all_trades_data])

    ax = axes[0]
    scatter_colors = ['#27ae60' if p > 0 else '#e74c3c' for p in pnl_vals]
    ax.scatter(gap_vals, pnl_vals, c=scatter_colors, alpha=0.5, s=25, edgecolors='none')
    if len(gap_vals) > 2:
        z = np.polyfit(gap_vals, pnl_vals, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(gap_vals.min(), gap_vals.max(), 100)
        ax.plot(x_line, p_line(x_line), 'k--', lw=2, alpha=0.7,
                label=f'Trend: slope={z[0]:+.3f}')
        corr = np.corrcoef(gap_vals, pnl_vals)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:+.3f}\nn = {len(gap_vals)}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_xlabel('Gap %')
    ax.set_ylabel('PnL @ 17:00 (%)')
    ax.set_title('All Trades', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    gap_capped = gap_vals[gap_vals <= 30]
    pnl_capped = pnl_vals[gap_vals <= 30]
    scatter_colors2 = ['#27ae60' if p > 0 else '#e74c3c' for p in pnl_capped]
    ax.scatter(gap_capped, pnl_capped, c=scatter_colors2, alpha=0.5, s=25, edgecolors='none')
    if len(gap_capped) > 2:
        z2 = np.polyfit(gap_capped, pnl_capped, 1)
        p_line2 = np.poly1d(z2)
        x_line2 = np.linspace(gap_capped.min(), gap_capped.max(), 100)
        ax.plot(x_line2, p_line2(x_line2), 'k--', lw=2, alpha=0.7,
                label=f'Trend: slope={z2[0]:+.3f}')
        corr2 = np.corrcoef(gap_capped, pnl_capped)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr2:+.3f}\nn = {len(gap_capped)}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_xlabel('Gap %')
    ax.set_ylabel('PnL @ 17:00 (%)')
    ax.set_title('Zoomed: Gap <= 30% (excludes outliers)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p2 = os.path.join(BASE_DIR, 'agent1_e3b_gap_scatter.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p2}")

    # ── Figure 3: Win rate + trade count distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Gap Bucket Profile', fontsize=12, fontweight='bold')

    wrs = [bucket_results[l]['metrics']['wr'] * 100 for l in labels]
    ax = axes[0]
    ax.bar(x, wrs, color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(50, color='#e74c3c', ls='--', lw=1, alpha=0.6, label='50% baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30)
    ax.set_ylabel('Win Rate (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Win Rate by Bucket', fontweight='bold')
    for i, v in enumerate(wrs):
        ax.text(i, v + 1.5, f'{v:.0f}%', ha='center', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ns = [bucket_results[l]['metrics']['n'] for l in labels]
    ax = axes[1]
    ax.bar(x, ns, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30)
    ax.set_ylabel('Trade Count')
    ax.set_title('Trade Distribution by Gap Size', fontweight='bold')
    total_n = sum(ns)
    for i, v in enumerate(ns):
        ax.text(i, v + 0.5, f'{v}\n({v/total_n*100:.0f}%)', ha='center', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p3 = os.path.join(BASE_DIR, 'agent1_e3b_gap_profile.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p3}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

# Part 1: Original E2 grid (N × T+2..T+5)
cssd_daily, dates = run_backtest()
grid = analyze_grid(cssd_daily, dates)
best_key = print_grid(grid, dates)
plot_grid(grid, dates, best_key)

# Part 2: Capital efficiency (N × EOD..T+5 with pool-adjusted $)
cap_grid = capital_efficiency_analysis(cssd_daily, dates)
plot_capital_efficiency(cap_grid, dates)

# Part 3: Intraday exit timing (N × 10:00..20:00 hourly, including after-hours)
intra_grid = intraday_exit_analysis(cssd_daily, dates)
plot_intraday_exit(intra_grid, dates)

# Part 4: E3b Gap size impact (N=1, exit@17:00 — deployed config)
bucket_results = gap_impact_analysis(cssd_daily, dates)
all_n1_trades = [t for dt in dates for t in cssd_daily.get(dt, [])[:1]
                 if t.get('pnl_1700') is not None]
plot_gap_impact(bucket_results, all_n1_trades, dates)
