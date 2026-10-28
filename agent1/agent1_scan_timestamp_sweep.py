# %% Agent 1 — Scan-Timestamp Sweep
#
# Which minute between 09:15 and 09:45 ET is the best time to rank candidates?
# Everything else frozen from agent1_baseline_bt.py (E2 benchmark).
#
# Pool: top 50 by open gap (open_gap >= 0%), price >= $10, float + exclusion rules.
# At each T, re-rank by gap_T = (price_T - prior_close) / prior_close.
# No fallback to open-gap when PM bars missing — drop the ticker.
# Apply 3% gap threshold, CS filter, sustained_drop, TOP_N=1.
# Entry: local_high anchored to T — scan for peak starts so that
#   confirmation bar (peak + w) >= T's RTH bar index. For T before
#   09:30, this is the baseline (peak bars 3-11, entry bars 6-14).
#   For T after 09:30, entry window shrinks. Fallback = max(bar 15, T bar).
# Exit: 17:00 ET.
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
from matplotlib.ticker import FuncFormatter

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — frozen from baseline except POOL_SIZE and TIMESTAMPS
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_START = '2026-08-01'
BACKTEST_END   = '2026-09-20'

GAP_MIN_PCT    = 3.0           # frozen: 3% threshold applied after re-ranking
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
TOP_N          = 1
POOL_SIZE      = 50            # wider pool: top 50 by open gap per day

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

EXIT_TIME      = '17:00'
CAPITAL        = 100_000

TIMESTAMPS = [f"09:{m:02d}" for m in range(15, 46)]

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
# HELPERS — verbatim from baseline
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
# ASYNC MINUTE BAR FETCH — verbatim from baseline
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
# ASYNC TICKER REFERENCE FETCH — verbatim from baseline
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
# PATTERN CLASSIFICATION + LOCAL HIGH — verbatim from baseline
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
        return {'entry_price': bars[fb][1]['c'],
                'entry_time': bars[fb][0],
                'entry_type': 'fallback', 'peak_bar': None}
    return None


def _T_to_min_entry_bar(T):
    h, m = int(T[:2]), int(T[3:5])
    T_min = h * 60 + m
    rth_start = 9 * 60 + 30
    if T_min <= rth_start:
        return 0
    return T_min - rth_start


def compute_trade_for_T(bar_data, T, tk, dt):
    rth_bars = bar_data.get('rth', [])
    if not rth_bars or len(rth_bars) < PEAK_WINDOW * 2 + 1:
        return None
    min_eb = _T_to_min_entry_bar(T)
    entry = find_local_high(rth_bars, min_entry_bar=min_eb)
    if entry is None:
        return None
    ep = entry['entry_price']
    session_bars = rth_bars + bar_data.get('ah', [])
    exit_price = None
    for t_str, bar in session_bars:
        if t_str > EXIT_TIME: break
        if t_str >= entry['entry_time']:
            exit_price = bar['c']
    if exit_price is None:
        t0d = _gd_cache.get(dt, {}).get(tk, {})
        if t0d.get('c'):
            exit_price = t0d['c']
        else:
            return None
    pnl = round((ep - exit_price) / ep * 100, 3)
    return {
        'ticker': tk, 'scan_date': dt,
        'entry_price': ep, 'entry_time': entry['entry_time'],
        'entry_type': entry['entry_type'], 'exit_price': exit_price,
        'pnl_eod': pnl,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pnl_list):
    if not pnl_list:
        return {'n':0,'avg':0,'med':0,'std':0,'wr':0,'sharpe':0,'pf':0,
                'ci_lo':0,'ci_hi':0}
    a = np.array(pnl_list)
    n = len(a); avg = float(np.mean(a)); med = float(np.median(a))
    std = float(np.std(a, ddof=1)) if n > 1 else 0
    wr = float((a > 0).mean())
    sharpe = avg / std if std > 0 else 0
    se = std / np.sqrt(n) if n > 0 else 0
    w = a[a > 0]; l = a[a < 0]
    pf = float(w.sum() / abs(l.sum())) if len(l) > 0 and l.sum() != 0 else float('inf')
    return {'n':n, 'avg':avg, 'med':med, 'std':std, 'wr':wr,
            'sharpe':sharpe, 'pf':pf, 'ci_lo':avg - 1.96*se, 'ci_hi':avg + 1.96*se}


# ═══════════════════════════════════════════════════════════════════════════════
# POOL BUILDING — wider pool, independent of sweep timestamp
# ═══════════════════════════════════════════════════════════════════════════════

def build_pool_day(dt_str):
    if not _is_td(dt_str): return []
    prior_str = _prior_td(dt_str).strftime('%Y-%m-%d')
    prior = fetch_grouped_daily(prior_str)
    today = fetch_grouped_daily(dt_str)

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
        if gap < 0: continue
        fs = _FLOAT.get(tk)
        if fs is not None and fs < FLOAT_MIN: continue
        candidates.append({
            'ticker': tk, 'open_gap': gap, 'prior_close': prev_c,
            'open_price': cur_p, 'scan_date': dt_str,
        })

    candidates.sort(key=lambda x: -x['open_gap'])
    return candidates[:POOL_SIZE]


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE AT TIMESTAMP T
# ═══════════════════════════════════════════════════════════════════════════════

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
# ═══════════════════════════════════════════════════════════════════════════════
#
#  SHARED PIPELINE — runs once, all timestamps share the result
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def run_shared_pipeline():
    t_total = time.time()

    print(f"\n{'='*80}")
    print(f"  AGENT 1 — Scan-Timestamp Sweep")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Pool: top {POOL_SIZE} by open gap (open_gap >= 0%), price >= ${PRICE_MIN}")
    print(f"  Sweep: {TIMESTAMPS[0]} → {TIMESTAMPS[-1]} ET ({len(TIMESTAMPS)} timestamps)")
    print(f"  Frozen: CS + sustained_drop, TOP_N={TOP_N}, exit {EXIT_TIME} ET")
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

    # ── Phase 1: Build pool per day ──
    t1 = time.time()
    print(f"\n  Phase 1: Building candidate pools (top {POOL_SIZE} by open gap, gap >= 0%)...")
    pool_by_day = {}
    total_pool = 0
    for i, dt in enumerate(dates):
        pool = build_pool_day(dt)
        pool_by_day[dt] = pool
        total_pool += len(pool)
        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_pool} total  |  {time.time()-t_total:.0f}s")
    avg_pool = total_pool / n_dates if n_dates > 0 else 0
    print(f"  → {total_pool} total pool entries, avg {avg_pool:.1f}/day  ({time.time()-t1:.0f}s)")

    # ── Phase 2: Fetch minute bars for all pool tickers ──
    t2 = time.time()
    all_pairs = set()
    for dt, pool in pool_by_day.items():
        for c in pool:
            all_pairs.add((c['ticker'], dt))
    print(f"\n  Phase 2: Fetching 1-min bars for {len(all_pairs)} ticker-day pairs...")

    pairs_list = list(all_pairs)
    all_bars = {}
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start + 100]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        done = min(c_start + 100, len(pairs_list))
        if done % 500 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs  ({time.time()-t2:.0f}s)")

    # ── Phase 3: Fetch ticker references (CS filter) ──
    t3 = time.time()
    all_tickers = set()
    for pool in pool_by_day.values():
        for c in pool:
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
    print(f"  → {n_cs} CS out of {len(all_tickers)} unique tickers  ({time.time()-t3:.0f}s)")

    # ── Phase 4: Classify patterns ──
    t4 = time.time()
    pattern_cache = {}
    for key, bar_data in all_bars.items():
        rth_bars = bar_data.get('rth', [])
        pattern_cache[key] = classify_open_pattern(rth_bars)

    n_sd = sum(1 for p in pattern_cache.values() if p == 'sustained_drop')
    print(f"\n  Phase 4: Pattern classification — {n_sd} sustained_drop "
          f"out of {len(pattern_cache)}  ({time.time()-t4:.0f}s)")

    # ── Phase 5: Build eligible keys (CS + sustained_drop) ──
    t5 = time.time()
    eligible_keys = set()
    for key in all_bars:
        tk, dt = key
        if _ticker_ref.get(tk, {}).get('type') != 'CS': continue
        if pattern_cache.get(key) != 'sustained_drop': continue
        eligible_keys.add(key)

    print(f"\n  Phase 5: {len(eligible_keys)} eligible ticker-days "
          f"(CS + sustained_drop)  ({time.time()-t5:.0f}s)")
    print(f"  Entries will be computed per-T (anchored to scan timestamp)")

    elapsed = time.time() - t_total
    print(f"\n  Shared pipeline complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return pool_by_day, all_bars, eligible_keys, dates


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  TIMESTAMP SWEEP
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def sweep_one_T(T, pool_by_day, all_bars, eligible_keys, dates):
    gap_min = GAP_MIN_PCT / 100.0
    trades = []
    picks = {}
    n_dropped_total = 0
    n_outside_top30 = 0
    n_pick_days = 0

    for dt in dates:
        pool = pool_by_day.get(dt, [])
        if not pool: continue

        n_dropped_day = 0
        ranked = []
        for idx, c in enumerate(pool):
            tk = c['ticker']
            key = (tk, dt)
            bar_data = all_bars.get(key)
            price_T = get_price_at_T(bar_data, T)
            if price_T is None:
                n_dropped_day += 1
                continue
            gap_T = (price_T - c['prior_close']) / c['prior_close']
            if gap_T < gap_min: continue
            if key not in eligible_keys: continue
            ranked.append((tk, gap_T, idx, key))

        n_dropped_total += n_dropped_day
        ranked.sort(key=lambda x: -x[1])

        for pick_tk, _, open_rank, pick_key in ranked[:TOP_N]:
            trade = compute_trade_for_T(all_bars[pick_key], T, pick_tk, dt)
            if trade is None: continue
            picks[dt] = pick_tk
            n_pick_days += 1
            if open_rank >= 30:
                n_outside_top30 += 1
            trades.append(trade)
            break

    m = compute_metrics([t['pnl_eod'] for t in trades])

    daily_pnl = defaultdict(list)
    for t in trades:
        daily_pnl[t['scan_date']].append(t['pnl_eod'])
    cum_nc = 0.0
    equity_nc = [CAPITAL]
    for dt in dates:
        dp = daily_pnl.get(dt, [])
        if dp:
            n = len(dp)
            day_dollar = sum((CAPITAL / n) * (r / 100) for r in dp)
        else:
            day_dollar = 0.0
        cum_nc += day_dollar
        equity_nc.append(CAPITAL + cum_nc)

    eq_arr = np.array(equity_nc)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = float(dd.min())

    max_loss = float(min(t['pnl_eod'] for t in trades)) if trades else 0.0
    n_no_pick = len(dates) - n_pick_days
    pct_outside = n_outside_top30 / n_pick_days * 100 if n_pick_days > 0 else 0

    return {
        'T': T, 'trades': trades, 'picks': picks, 'metrics': m,
        'total_nc': cum_nc, 'max_dd': max_dd, 'max_loss': max_loss,
        'n_no_pick': n_no_pick, 'n_dropped': n_dropped_total,
        'pct_outside_top30': pct_outside,
    }


def run_sweep(pool_by_day, all_bars, eligible_keys, dates):
    print(f"\n{'='*80}")
    print(f"  SWEEP: {len(TIMESTAMPS)} timestamps")
    print(f"{'='*80}")

    results = {}
    for T in TIMESTAMPS:
        res = sweep_one_T(T, pool_by_day, all_bars, eligible_keys, dates)
        m = res['metrics']
        print(f"  {T} ET  |  n={m['n']:<4d}  avg={m['avg']:+7.2f}%  "
              f"Sharpe={m['sharpe']:.3f}  ${res['total_nc']:+10,.0f}  "
              f"dropped={res['n_dropped']}")
        results[T] = res

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PICK AGREEMENT MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def compute_agreement(results):
    n_T = len(TIMESTAMPS)
    matrix = np.zeros((n_T, n_T))
    for i in range(n_T):
        pi = results[TIMESTAMPS[i]]['picks']
        for j in range(n_T):
            pj = results[TIMESTAMPS[j]]['picks']
            common = set(pi.keys()) & set(pj.keys())
            if common:
                agree = sum(1 for d in common if pi[d] == pj[d])
                matrix[i][j] = agree / len(common)
    return matrix


def print_agreement(matrix):
    print(f"\n\n{'='*80}")
    print(f"  PICK-AGREEMENT MATRIX (fraction of shared days with same pick)")
    print(f"{'='*80}\n")

    headers = [''] + [T[3:] for T in TIMESTAMPS]
    rows = []
    for i, T in enumerate(TIMESTAMPS):
        row = [T[3:]]
        for j in range(len(TIMESTAMPS)):
            if i == j:
                row.append('  — ')
            else:
                row.append(f"{matrix[i][j]*100:4.0f}%")
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right'))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_results_table(results):
    ref_T = '09:30'
    ref_picks = results[ref_T]['picks'] if ref_T in results else {}

    print(f"\n\n{'='*120}")
    print(f"  SCAN-TIMESTAMP SWEEP RESULTS")
    print(f"{'='*120}")

    rows = []
    for T in TIMESTAMPS:
        r = results[T]
        m = r['metrics']

        if T == ref_T:
            agree_str = '  — '
        else:
            p_T = r['picks']
            common = set(p_T.keys()) & set(ref_picks.keys())
            if common:
                agree = sum(1 for d in common if p_T[d] == ref_picks[d]) / len(common)
                agree_str = f"{agree*100:.0f}%"
            else:
                agree_str = 'n/a'

        rows.append([
            T, m['n'], f"{m['avg']:+.2f}%", f"{m['med']:+.2f}%",
            f"{m['wr']*100:.0f}%", f"{m['sharpe']:.3f}",
            f"{m['pf']:.2f}" if m['pf'] < 100 else '∞',
            f"${r['total_nc']:+,.0f}", f"{r['max_loss']:+.1f}%",
            f"{r['max_dd']:.1f}%", f"[{m['ci_lo']:+.1f}, {m['ci_hi']:+.1f}]",
            r['n_no_pick'], agree_str, r['n_dropped'],
        ])

    headers = ['T (ET)', 'n', 'avg PnL', 'median', 'WR', 'Sharpe', 'PF',
               '$ NC', 'max loss', 'max DD', '95% CI', 'no-pick', 'vs 09:30',
               'dropped']
    print()
    print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right'))

    print(f"\n  'dropped' = pool tickers with no bar at or before T (summed across all days)")
    print(f"  'vs 09:30' = fraction of shared-pick days where T chose the same ticker as 09:30")

    # ── Entry-type breakdown per timestamp ──
    print(f"\n\n{'='*120}")
    print(f"  ENTRY-TYPE BREAKDOWN PER TIMESTAMP")
    print(f"{'='*120}")

    et_rows = []
    for T in TIMESTAMPS:
        r = results[T]
        all_trades = r['trades']
        lh_trades = [t for t in all_trades if t['entry_type'] == 'local_high']
        fb_trades = [t for t in all_trades if t['entry_type'] == 'fallback']

        lh_pnls = [t['pnl_eod'] for t in lh_trades]
        fb_pnls = [t['pnl_eod'] for t in fb_trades]
        all_pnls = [t['pnl_eod'] for t in all_trades]

        def _row_stats(pnls):
            if not pnls:
                return 0, '—', '—', '—', '—'
            a = np.array(pnls)
            n = len(a)
            avg = float(np.mean(a))
            wr = float((a > 0).mean())
            total_nc = sum((CAPITAL) * (r / 100) for r in a)
            sharpe = avg / float(np.std(a, ddof=1)) if n > 1 and np.std(a, ddof=1) > 0 else 0
            return n, f"{avg:+.2f}%", f"{wr*100:.0f}%", f"${total_nc:+,.0f}", f"{sharpe:.3f}"

        n_all, avg_all, wr_all, nc_all, sh_all = _row_stats(all_pnls)
        n_lh, avg_lh, wr_lh, nc_lh, sh_lh = _row_stats(lh_pnls)
        n_fb, avg_fb, wr_fb, nc_fb, sh_fb = _row_stats(fb_pnls)

        et_rows.append([
            T,
            n_all, avg_all, wr_all, nc_all, sh_all,
            n_lh, avg_lh, wr_lh, nc_lh,
            n_fb, avg_fb, wr_fb, nc_fb,
        ])

    et_headers = [
        'T (ET)',
        'n', 'avg', 'WR', '$ NC', 'Sharpe',
        'LH n', 'LH avg', 'LH WR', 'LH $',
        'FB n', 'FB avg', 'FB WR', 'FB $',
    ]
    print()
    print(tabulate(et_rows, headers=et_headers, tablefmt='simple', stralign='right'))
    print(f"\n  LH = local_high entry (bars 3-11), FB = fallback entry (bar 15)")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sweep(results, agreement, filename='agent1_scan_timestamp_sweep.png'):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    Ts = TIMESTAMPS
    avgs = [results[T]['metrics']['avg'] for T in Ts]
    sharpes = [results[T]['metrics']['sharpe'] for T in Ts]
    ns = [results[T]['metrics']['n'] for T in Ts]

    # 3-point neighborhood means
    avg_smooth = []
    sharpe_smooth = []
    for i in range(len(Ts)):
        lo = max(0, i - 1)
        hi = min(len(Ts), i + 2)
        avg_smooth.append(np.mean(avgs[lo:hi]))
        sharpe_smooth.append(np.mean(sharpes[lo:hi]))

    wrs = [results[T]['metrics']['wr'] * 100 for T in Ts]
    ncs = [results[T]['total_nc'] for T in Ts]

    # entry-type split
    lh_avgs, fb_avgs, lh_wrs, fb_wrs = [], [], [], []
    for T in Ts:
        lh = [t['pnl_eod'] for t in results[T]['trades'] if t['entry_type'] == 'local_high']
        fb = [t['pnl_eod'] for t in results[T]['trades'] if t['entry_type'] == 'fallback']
        lh_avgs.append(np.mean(lh) if lh else 0)
        fb_avgs.append(np.mean(fb) if fb else 0)
        lh_wrs.append((np.array(lh) > 0).mean() * 100 if lh else 0)
        fb_wrs.append((np.array(fb) > 0).mean() * 100 if fb else 0)

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle('Agent 1 — Scan-Timestamp Sweep\n'
                 f'{BACKTEST_START} → {BACKTEST_END}  |  '
                 f'CS + sustained_drop, TOP_N={TOP_N}, exit {EXIT_TIME} ET',
                 fontsize=11, fontweight='bold')

    # ── Top left: avg PnL + Sharpe vs T ──
    ax1 = axes[0, 0]
    x = np.arange(len(Ts))
    color1 = '#2c3e50'
    color2 = '#c0392b'

    ln1 = ax1.plot(x, avgs, 'o-', color=color1, linewidth=2, markersize=5,
                   label='Avg PnL (%)')
    ln1s = ax1.plot(x, avg_smooth, '--', color=color1, linewidth=1.5, alpha=0.5,
                    label='3-pt smooth')
    ax1.set_ylabel('Avg PnL (%)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([T[3:] for T in Ts], rotation=45, fontsize=8)
    ax1.set_xlabel('Scan timestamp (ET)')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(x, sharpes, 's-', color=color2, linewidth=2, markersize=5,
                   label='Sharpe')
    ln2s = ax2.plot(x, sharpe_smooth, '--', color=color2, linewidth=1.5, alpha=0.5,
                    label='3-pt smooth')
    ax2.set_ylabel('Sharpe', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lns = ln1 + ln1s + ln2 + ln2s
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower left', fontsize=8)
    ax1.set_title('Performance vs Scan Timestamp')

    for i, n in enumerate(ns):
        ax1.annotate(f'n={n}', (x[i], avgs[i]), textcoords='offset points',
                     xytext=(0, 10), fontsize=6, ha='center', color='gray')

    # ── Top right: Win rate + Total $ vs T ──
    ax_wr = axes[0, 1]
    ln_wr = ax_wr.plot(x, wrs, 'o-', color='#27ae60', linewidth=2, markersize=5,
                       label='Win Rate (%)')
    ax_wr.set_ylabel('Win Rate (%)', color='#27ae60')
    ax_wr.tick_params(axis='y', labelcolor='#27ae60')
    ax_wr.set_xticks(x)
    ax_wr.set_xticklabels([T[3:] for T in Ts], rotation=45, fontsize=8)
    ax_wr.set_xlabel('Scan timestamp (ET)')
    ax_wr.grid(True, alpha=0.3)

    ax_nc = ax_wr.twinx()
    ln_nc = ax_nc.plot(x, ncs, 's-', color='#8e44ad', linewidth=2, markersize=5,
                       label='Total $ NC')
    ax_nc.set_ylabel('Total $ NC', color='#8e44ad')
    ax_nc.tick_params(axis='y', labelcolor='#8e44ad')
    ax_nc.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'${v:,.0f}'))

    lns_wr = ln_wr + ln_nc
    ax_wr.legend(lns_wr, [l.get_label() for l in lns_wr], loc='lower left', fontsize=8)
    ax_wr.set_title('Win Rate & Total PnL vs Scan Timestamp')

    # ── Bottom left: Entry-type avg PnL breakdown ──
    ax_et = axes[1, 0]
    w = 0.35
    ax_et.bar(x - w/2, lh_avgs, w, color='#2c3e50', alpha=0.8, label='Local High avg')
    ax_et.bar(x + w/2, fb_avgs, w, color='#e67e22', alpha=0.8, label='Fallback avg')
    ax_et.axhline(0, color='#333', lw=0.8, ls=':')
    ax_et.set_xticks(x)
    ax_et.set_xticklabels([T[3:] for T in Ts], rotation=45, fontsize=8)
    ax_et.set_xlabel('Scan timestamp (ET)')
    ax_et.set_ylabel('Avg PnL (%)')
    ax_et.set_title('Avg PnL by Entry Type')
    ax_et.legend(fontsize=8)
    ax_et.grid(True, alpha=0.3, axis='y')

    # ── Bottom right: Agreement heatmap ──
    ax3 = axes[1, 1]
    im = ax3.imshow(agreement * 100, cmap='YlOrRd', aspect='auto',
                    vmin=50, vmax=100)
    ax3.set_xticks(x)
    ax3.set_yticks(x)
    ax3.set_xticklabels([T[3:] for T in Ts], rotation=45, fontsize=7)
    ax3.set_yticklabels([T[3:] for T in Ts], fontsize=7)
    ax3.set_xlabel('Timestamp (ET)')
    ax3.set_ylabel('Timestamp (ET)')
    ax3.set_title('Pick Agreement (%)')
    fig.colorbar(im, ax=ax3, shrink=0.8, label='Agreement %')

    for i in range(len(Ts)):
        for j in range(len(Ts)):
            val = agreement[i][j] * 100
            color = 'white' if val < 70 else 'black'
            ax3.text(j, i, f'{val:.0f}', ha='center', va='center',
                     fontsize=5, color=color)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(BASE_DIR, filename)
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════

def recommend_T(results):
    print(f"\n\n{'='*80}")
    print(f"  RECOMMENDATION")
    print(f"{'='*80}")

    Ts = TIMESTAMPS
    avgs = [results[T]['metrics']['avg'] for T in Ts]
    sharpes = [results[T]['metrics']['sharpe'] for T in Ts]
    ci_los = [results[T]['metrics']['ci_lo'] for T in Ts]
    ci_his = [results[T]['metrics']['ci_hi'] for T in Ts]
    ns = [results[T]['metrics']['n'] for T in Ts]

    # 3-point neighborhood means
    avg_nbr = []
    sharpe_nbr = []
    for i in range(len(Ts)):
        lo = max(0, i - 1)
        hi = min(len(Ts), i + 2)
        avg_nbr.append(np.mean(avgs[lo:hi]))
        sharpe_nbr.append(np.mean(sharpes[lo:hi]))

    print(f"\n  ── 3-Point Neighborhood Means ──\n")
    rows = []
    for i, T in enumerate(Ts):
        rows.append([T, f"{avgs[i]:+.2f}%", f"{avg_nbr[i]:+.2f}%",
                     f"{sharpes[i]:.3f}", f"{sharpe_nbr[i]:.3f}"])
    print(tabulate(rows, headers=['T', 'avg PnL', 'nbr avg', 'Sharpe', 'nbr Sharpe'],
                   tablefmt='simple', stralign='right'))

    best_nbr_sharpe_idx = int(np.argmax(sharpe_nbr))
    best_nbr_avg_idx = int(np.argmax(avg_nbr))
    best_raw_sharpe_idx = int(np.argmax(sharpes))
    best_raw_avg_idx = int(np.argmax(avgs))

    print(f"\n  Best by neighborhood Sharpe: {Ts[best_nbr_sharpe_idx]} "
          f"({sharpe_nbr[best_nbr_sharpe_idx]:.3f})")
    print(f"  Best by neighborhood avg PnL: {Ts[best_nbr_avg_idx]} "
          f"({avg_nbr[best_nbr_avg_idx]:+.2f}%)")
    print(f"  Best by raw Sharpe: {Ts[best_raw_sharpe_idx]} "
          f"({sharpes[best_raw_sharpe_idx]:.3f})")
    print(f"  Best by raw avg PnL: {Ts[best_raw_avg_idx]} "
          f"({avgs[best_raw_avg_idx]:+.2f}%)")

    # CI overlap check
    best_ci_lo = ci_los[best_raw_avg_idx]
    best_ci_hi = ci_his[best_raw_avg_idx]
    worst_avg_idx = int(np.argmin(avgs))
    worst_ci_lo = ci_los[worst_avg_idx]
    worst_ci_hi = ci_his[worst_avg_idx]
    overlaps = best_ci_lo < worst_ci_hi and worst_ci_lo < best_ci_hi

    print(f"\n  ── CI Overlap Check ──")
    print(f"  Best  ({Ts[best_raw_avg_idx]}): CI [{best_ci_lo:+.2f}, {best_ci_hi:+.2f}]")
    print(f"  Worst ({Ts[worst_avg_idx]}): CI [{worst_ci_lo:+.2f}, {worst_ci_hi:+.2f}]")
    print(f"  Overlapping: {'YES — differences NOT statistically distinguishable' if overlaps else 'NO — statistically distinguishable'}")

    # Spread diagnostic
    avg_range = max(avgs) - min(avgs)
    sharpe_range = max(sharpes) - min(sharpes)
    avg_se = np.mean([results[T]['metrics']['std'] for T in Ts]) / np.sqrt(np.mean(ns))
    print(f"\n  Spread: avg PnL range = {avg_range:.2f}pp, Sharpe range = {sharpe_range:.3f}")
    print(f"  Typical SE on avg PnL ≈ {avg_se:.2f}pp")

    # Pick the recommended T
    if overlaps:
        rec_idx = best_nbr_sharpe_idx
        reason = "CIs overlap → pick on stability (neighborhood Sharpe) + operational ease"
    else:
        rec_idx = best_nbr_sharpe_idx
        reason = "neighborhood Sharpe on a stable plateau"

    rec_T = Ts[rec_idx]
    entry_bar3 = '09:33'
    budget_min = (9 * 60 + 33) - (int(rec_T[0:2]) * 60 + int(rec_T[3:5]))

    print(f"\n  ── RECOMMENDED: T = {rec_T} ET ──")
    print(f"  Reason: {reason}")
    print(f"  Decision-to-entry budget: {budget_min} minutes "
          f"({rec_T} → {entry_bar3} earliest entry)")
    print(f"  In that window: fetch grouped daily, compute gaps, look up CS type,")
    print(f"  run float filter, rank. CS lookups must be batched+pre-warmed")
    print(f"  if budget < 5 min (current live code does serial lookups with 50ms sleep).")

    # Pool boundary diagnostic
    pct_outside = [results[T]['pct_outside_top30'] for T in Ts]
    max_outside = max(pct_outside)
    print(f"\n  ── Pool Boundary Diagnostic ──")
    print(f"  Max % of days where pick was outside top-30 by open gap: {max_outside:.1f}%")
    if max_outside > 10:
        print(f"  WARNING: pool boundary may be binding — consider widening POOL_SIZE "
              f"before trusting these results")
    else:
        print(f"  OK: pool boundary is not binding (rank-1 pick rarely falls outside top-30)")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

pool_by_day, all_bars, eligible_keys, dates = run_shared_pipeline()
sweep_results = run_sweep(pool_by_day, all_bars, eligible_keys, dates)
print_results_table(sweep_results)
agreement = compute_agreement(sweep_results)
print_agreement(agreement)
plot_sweep(sweep_results, agreement)
recommend_T(sweep_results)

print(f"\n  Baseline reference (E2 @ open gap): 160 trades, +14.71% avg, "
      f"Sharpe 0.594, $+2.35M NC")
