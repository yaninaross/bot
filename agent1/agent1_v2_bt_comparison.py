# %% Agent 1 v2 — Backtest (100% aligned with agent1_v2.py live scanner)
#
# Critical alignment vs agent1_trade_review.py:
#   1. FILTER FIRST, then top N (v2 order) — old BT took top N then filtered
#   2. No fallback entry — local high or skip
#   3. Exit at 17:00 ET same day (not T+2)
#   4. No SL
#   5. Same 5 strategies, same filter functions, same gap/AH/erosion logic
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
BACKTEST_END   = '2026-10-28'

GAP_MIN_PCT    = 3.0
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
PRE_FILTER_N   = 30
GAP_CONSUMED_MAX = 0.50

SCAN_TIME_ET   = '09:31'
EXIT_TIME      = '17:00'

MIN_RTH_BARS   = 2          # min RTH bars with volume > 0 by scan time
MIN_OPEN_VOL   = 5_000      # min shares in bar 0 (09:30)

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
MAX_PEAK_TIME_ET  = '09:41'
MAX_ENTRY_TIME_ET = '09:44'

CAPITAL        = 100_000

POST_AUG_DATE  = '2026-08-16'

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
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
# STRATEGY DEFINITIONS — identical to agent1_v2.py
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGIES = [
    {'id': 'C5',    'name': 'C5: PM+Red',           'optimal_n': 1},
    {'id': 'C3+C5', 'name': 'C3+C5: SkipT6+PM+Red', 'optimal_n': 1},
    {'id': 'C7',    'name': 'C7: PMTrend<0',         'optimal_n': 1},
    {'id': 'C9',    'name': 'C9: 2BarDrop',           'optimal_n': 2},
    {'id': 'C3',    'name': 'C3: SkipT6',             'optimal_n': 3},
]

N_VALUES = [1, 2, 3, 4, 5]


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
               {"adjusted": "true", "include_otc": "false"})
    out = {}
    for r in data.get('results', []):
        tk = r.get('T', '')
        if not _TK_RE.match(tk): continue
        out[tk] = {'o': float(r.get('o', 0)), 'h': float(r.get('h', 0)),
                   'l': float(r.get('l', 0)), 'c': float(r.get('c', 0)),
                   'v': int(r.get('v', 0))}
    _gd_cache[dt_str] = out
    return out


async def _fetch_bars_one(session, tk, dt, sem):
    url = f"{_BASE}/v2/aggs/ticker/{tk}/range/1/minute/{dt}/{dt}"
    params = {"adjusted": "true", "sort": "asc", "limit": "1000", "apiKey": POLYGON_API_KEY}
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
                bar_d = {'o': float(b['o']), 'h': float(b['h']),
                         'l': float(b['l']), 'c': float(b['c']), 'v': int(b['v'])}
                if '04:00' <= t_str < '09:30':
                    pm.append((t_str, bar_d))
                elif '09:30' <= t_str < '16:00':
                    rth.append((t_str, bar_d))
                elif '16:00' <= t_str <= '20:00':
                    ah.append((t_str, bar_d))
            return (tk, dt), {'pm': pm, 'rth': rth, 'ah': ah}
        return (tk, dt), None

import ssl
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()

async def _fetch_bars_batch(pairs):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=_SSL_CTX)) as session:
        tasks = [_fetch_bars_one(session, tk, dt, sem) for tk, dt in pairs]
        results = await asyncio.gather(*tasks)
    return {key: val for key, val in results if val is not None}

def fetch_minute_bars_batch(pairs):
    if not pairs: return {}
    return asyncio.run(_fetch_bars_batch(pairs))


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
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=_SSL_CTX)) as session:
        tasks = [_fetch_ref_one(session, tk, sem) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: val for tk, val in results}

def fetch_ticker_refs(tickers):
    uncached = [tk for tk in tickers if tk not in _ticker_ref]
    if uncached:
        batch = asyncio.run(_fetch_refs_batch(uncached))
        _ticker_ref.update(batch)


def get_price_at_T(bar_data, T):
    if bar_data is None: return None
    all_bars = (bar_data.get('pm') or []) + (bar_data.get('rth') or [])
    best = None
    for t_str, bd in all_bars:
        if t_str <= T:
            best = bd['c']
        else:
            break
    return best


def compute_metrics(pnl_list, label=''):
    if not pnl_list:
        return {'label': label, 'n': 0, 'avg': 0, 'med': 0, 'std': 0, 'wr': 0,
                'sharpe': 0, 'pf': 0, 'ci_lo': 0, 'ci_hi': 0, 'trimmed_mean': 0}
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
    return {'label': label, 'n': n, 'avg': avg, 'med': med, 'std': std, 'wr': wr,
            'sharpe': sharpe, 'pf': pf, 'ci_lo': avg - 1.96 * se, 'ci_hi': avg + 1.96 * se,
            'trimmed_mean': tm}


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY FILTERS — copied from agent1_v2.py
# ═══════════════════════════════════════════════════════════════════════════════

def check_c5(candidate, bar_data):
    if bar_data is None: return False
    pm_bars = bar_data.get('pm', [])
    rth_bars = bar_data.get('rth', [])
    prior_close = candidate.get('prior_close', 0)

    pm_after_830 = [(t, b) for t, b in pm_bars if '08:30' <= t < '09:30']
    if pm_after_830 and prior_close > 0:
        pm_high = max(b['h'] for _, b in pm_after_830)
        last_pm_close = pm_after_830[-1][1]['c']
        denom = pm_high - prior_close
        pm_retrace = (pm_high - last_pm_close) / denom if denom > 0 else 0
        if pm_retrace < 0.08:
            return False
    else:
        return False

    scan_rth = [(t, b) for t, b in rth_bars if t <= SCAN_TIME_ET]
    if scan_rth:
        bar0 = scan_rth[0][1]
        if bar0['c'] > bar0['o']:
            return False
    else:
        return False

    return True


def check_c3_c5(candidate, bar_data):
    if not check_c3(candidate, bar_data):
        return False
    return check_c5(candidate, bar_data)


def check_c7(candidate, bar_data):
    if bar_data is None: return False
    pm_bars = bar_data.get('pm', [])
    pm_after_830 = [(t, b) for t, b in pm_bars if '08:30' <= t < '09:30']
    if len(pm_after_830) >= 2:
        first_pm = pm_after_830[0][1]['o']
        last_pm = pm_after_830[-1][1]['c']
        pm_trend = (last_pm - first_pm) / first_pm if first_pm > 0 else 0
        return pm_trend < 0
    return False


def check_c9(candidate, bar_data):
    if bar_data is None: return False
    rth_bars = bar_data.get('rth', [])
    scan_rth = [(t, b) for t, b in rth_bars if t <= SCAN_TIME_ET]
    if len(scan_rth) >= 2:
        open_p = scan_rth[0][1]['o']
        bar1_close = scan_rth[1][1]['c']
        drop = (bar1_close - open_p) / open_p if open_p > 0 else 0
        return drop < 0
    return False


def check_c3(candidate, bar_data):
    gap = candidate.get('gap_pct', 0)
    raw_gap = candidate.get('raw_open_gap', gap)
    if (0.15 <= gap < 0.30) or (0.15 <= raw_gap < 0.30):
        return False
    return True


FILTER_FNS = {
    'C5': check_c5,
    'C3+C5': check_c3_c5,
    'C7': check_c7,
    'C9': check_c9,
    'C3': check_c3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL HIGH — copied from agent1_v2.py
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_high(rth_bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    min_entry_bar=0):
    if len(rth_bars) < window * 2 + 1:
        return None
    start = max(window, min_entry_bar - window)
    for i in range(start, min(max_peak_bar, len(rth_bars) - window)):
        if rth_bars[i][0] > MAX_PEAK_TIME_ET:
            break
        ch = rth_bars[i][1]['h']
        if all(rth_bars[i - j][1]['h'] < ch and rth_bars[i + j][1]['h'] < ch
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(rth_bars) and rth_bars[ci][0] <= MAX_ENTRY_TIME_ET:
                return {
                    'entry_price': rth_bars[ci][1]['c'],
                    'entry_time': rth_bars[ci][0],
                    'entry_type': 'local_high',
                    'peak_bar': i,
                    'peak_time': rth_bars[i][0],
                }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
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
            'ticker': tk, 'gap_pct': gap, 'raw_open_gap': gap,
            'prior_close': prev_c, 'open_price': cur_p, 'scan_date': dt_str,
            'prior_str': prior_str,
        })
    candidates.sort(key=lambda x: -x['gap_pct'])
    return candidates


def run_pipeline():
    t_total = time.time()

    print(f"\n{'='*80}")
    print(f"  AGENT 1 v2 — BACKTEST (aligned with agent1_v2.py)")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Exit: {EXIT_TIME} ET  |  No SL  |  No fallback  |  Filter-first")
    print(f"{'='*80}")

    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"\n  {n_dates} trading days")

    # Phase 1: scan all days for gap candidates
    print(f"\n  Phase 1: Scanning for gap candidates...")
    all_candidates = {}
    total_cands = 0
    for i, dt in enumerate(dates):
        cands = scan_day(dt)
        all_candidates[dt] = cands
        total_cands += len(cands)
        if (i + 1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_cands} total  |  {time.time()-t_total:.0f}s")

    # Phase 2: fetch 1-min bars (scan day + prior day for AH ref)
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
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

    # Phase 2a: AH reference prices + align open_price to 09:30 RTH bar
    gap_min_final = GAP_MIN_PCT / 100.0
    print(f"\n  Phase 2a: AH reference prices...")
    n_ah_found = 0
    n_stale = 0
    for dt in dates:
        cands = all_candidates.get(dt, [])
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            sb = all_bars.get((tk, dt))
            if sb and sb.get('rth'):
                c['open_price'] = sb['rth'][0][1]['o']
                if c['prior_close'] > 0:
                    c['raw_open_gap'] = (c['open_price'] - c['prior_close']) / c['prior_close']
            prior_str = c.get('prior_str')
            if not prior_str: continue
            prior_bars = all_bars.get((tk, prior_str))
            if prior_bars is None: continue
            ah_bars = prior_bars.get('ah', [])
            if ah_bars:
                ah_ref = ah_bars[-1][1]['c']
                c['ah_ref_price'] = ah_ref
                n_ah_found += 1
                fresh_gap = (c['open_price'] - ah_ref) / ah_ref
                c['gap_pct'] = fresh_gap
                if fresh_gap < gap_min_final:
                    c['ah_stale'] = True
                    n_stale += 1
    print(f"  → AH ref: {n_ah_found} tickers, {n_stale} stale excluded")

    # Phase 2b: re-rank at SCAN_TIME + gap erosion (strictly on cands[:PRE_FILTER_N] to prevent cross-day cache leak)
    print(f"\n  Phase 2b: Re-rank + erosion...")
    n_eroded = 0
    for dt in dates:
        cands = all_candidates.get(dt, [])
        reranked = []
        for c in cands[:PRE_FILTER_N]:
            if c.get('ah_stale'): continue
            tk = c['ticker']
            bar_data = all_bars.get((tk, dt))
            price_T = get_price_at_T(bar_data, SCAN_TIME_ET)
            if price_T is None: continue
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
        reranked.sort(key=lambda x: -x['gap_pct'])
        all_candidates[dt] = reranked
    print(f"  → {n_eroded} eroded")

    # Phase 3: CS type check
    all_tickers = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            all_tickers.add(c['ticker'])
    print(f"\n  Phase 3: Reference data for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    for c_start in range(0, len(tk_list), 200):
        chunk = tk_list[c_start:c_start + 200]
        fetch_ticker_refs(chunk)

    # Build CS pool per day + liquidity filter
    cs_pool_by_day = {}
    n_illiquid_total = 0
    for dt in dates:
        cands = all_candidates.get(dt, [])
        cs_pool = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            ref = _ticker_ref.get(tk, {})
            at = ref.get('type', 'UNK')
            if at != 'CS': continue
            c['asset_type'] = at
            c['name'] = ref.get('name', '')
            bd = all_bars.get((tk, dt))
            if bd:
                rth = bd.get('rth', [])
                scan_rth = [(t, b) for t, b in rth if t <= SCAN_TIME_ET]
                bars_with_vol = sum(1 for _, b in scan_rth if b['v'] > 0)
                bar0_vol = rth[0][1]['v'] if rth else 0
                if bars_with_vol < MIN_RTH_BARS or bar0_vol < MIN_OPEN_VOL:
                    n_illiquid_total += 1
                    continue
            cs_pool.append(c)
        cs_pool_by_day[dt] = cs_pool
    print(f"  → Liquidity filter: {n_illiquid_total} excluded (need {MIN_RTH_BARS} bars, {MIN_OPEN_VOL:,} vol)")

    print(f"\n  Pipeline done in {time.time()-t_total:.0f}s")
    return cs_pool_by_day, all_bars, dates, t_total


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD TRADES — v2 aligned (filter first, then top N)
# ═══════════════════════════════════════════════════════════════════════════════

def build_trades_v2(cs_pool_by_day, all_bars, dates, strategy_id, top_n):
    check_fn = FILTER_FNS[strategy_id]
    trades = []

    for dt in dates:
        cs_pool = cs_pool_by_day.get(dt, [])
        picks = []

        for c in cs_pool:
            tk = c['ticker']
            bd = all_bars.get((tk, dt))
            if not check_fn(c, bd):
                continue

            rth = bd.get('rth', []) if bd else []
            entry = find_local_high(rth) if len(rth) >= PEAK_WINDOW * 2 + 1 else None
            if entry is None:
                continue

            ep = entry['entry_price']
            exit_price = None
            if bd:
                session_bars = rth + bd.get('ah', [])
                for t_str, bar in session_bars:
                    if t_str > EXIT_TIME:
                        break
                    if t_str >= entry['entry_time']:
                        exit_price = bar['c']

            if exit_price is None:
                gd = _gd_cache.get(dt, {}).get(tk, {})
                if gd.get('c'):
                    exit_price = gd['c']

            if exit_price is None:
                continue

            pnl = round((ep - exit_price) / ep * 100, 3)
            picks.append({
                'ticker': tk,
                'scan_date': dt,
                'gap_pct': c['gap_pct'],
                'open_price': c['open_price'],
                'prior_close': c['prior_close'],
                'entry_price': ep,
                'entry_time': entry['entry_time'],
                'entry_type': entry['entry_type'],
                'peak_bar': entry['peak_bar'],
                'exit_price': exit_price,
                'pnl_eod': pnl,
                'name': c.get('name', ''),
            })
            if len(picks) >= top_n:
                break

        trades.extend(picks)

    return trades



# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON RUNNER — Current BT vs Option A1 (Live Primary) vs Option A2 vs Option B
# ═══════════════════════════════════════════════════════════════════════════════

def get_exit_price(bd, rth, entry_time, dt, tk):
    exit_price = None
    if bd:
        session_bars = rth + bd.get("ah", [])
        for t_str, bar in session_bars:
            if t_str > EXIT_TIME:
                break
            if t_str >= entry_time:
                exit_price = bar["c"]
    if exit_price is None:
        gd = _gd_cache.get(dt, {}).get(tk, {})
        if gd.get("c"):
            exit_price = gd["c"]
    return exit_price


def make_trade_dict(c, dt, entry, exit_price, rank_in_filter=None):
    ep = entry["entry_price"]
    pnl = round((ep - exit_price) / ep * 100, 3)
    return {
        "ticker": c["ticker"],
        "scan_date": dt,
        "gap_pct": c["gap_pct"],
        "open_price": c["open_price"],
        "prior_close": c["prior_close"],
        "entry_price": ep,
        "entry_time": entry["entry_time"],
        "entry_type": entry["entry_type"],
        "peak_bar": entry["peak_bar"],
        "exit_price": exit_price,
        "pnl_eod": pnl,
        "name": c.get("name", ""),
        "filter_rank": rank_in_filter,
    }


# 1. Current Backtest (Hindsight walk-down: skips unconfirmed, takes top_n confirmed by gap rank)
def build_trades_current_bt(cs_pool_by_day, all_bars, dates, strategy_id, top_n):
    check_fn = FILTER_FNS[strategy_id]
    trades = []
    for dt in dates:
        cs_pool = cs_pool_by_day.get(dt, [])
        picks = []
        filter_rank = 0
        for c in cs_pool:
            tk = c["ticker"]
            bd = all_bars.get((tk, dt))
            if not check_fn(c, bd):
                continue
            filter_rank += 1
            rth = bd.get("rth", []) if bd else []
            entry = find_local_high(rth) if len(rth) >= PEAK_WINDOW * 2 + 1 else None
            if entry is None:
                continue
            exit_price = get_exit_price(bd, rth, entry["entry_time"], dt, tk)
            if exit_price is None:
                continue
            picks.append(make_trade_dict(c, dt, entry, exit_price, filter_rank))
            if len(picks) >= top_n:
                break
        trades.extend(picks)
    return trades


# 2. Option A1 (Strict Live Primary: Lock top_n passing check_fn at 09:31 ET, trade only if confirmed)
def build_trades_option_a_primary(cs_pool_by_day, all_bars, dates, strategy_id, top_n):
    check_fn = FILTER_FNS[strategy_id]
    trades = []
    for dt in dates:
        cs_pool = cs_pool_by_day.get(dt, [])
        locked_candidates = []
        for c in cs_pool:
            tk = c["ticker"]
            bd = all_bars.get((tk, dt))
            if check_fn(c, bd):
                locked_candidates.append(c)
                if len(locked_candidates) >= top_n:
                    break

        for rank_idx, c in enumerate(locked_candidates, 1):
            tk = c["ticker"]
            bd = all_bars.get((tk, dt))
            rth = bd.get("rth", []) if bd else []
            entry = find_local_high(rth) if len(rth) >= PEAK_WINDOW * 2 + 1 else None
            if entry is None:
                continue
            exit_price = get_exit_price(bd, rth, entry["entry_time"], dt, tk)
            if exit_price is None:
                continue
            trades.append(make_trade_dict(c, dt, entry, exit_price, rank_idx))
    return trades


# 3. Option A2 (Live show_n = max(opt_n, 2) banner: Lock show_n at 09:31 ET, trade all confirmed in banner)
def build_trades_option_a_banner(cs_pool_by_day, all_bars, dates, strategy_id, top_n):
    show_n = max(top_n, 2)
    return build_trades_option_a_primary(cs_pool_by_day, all_bars, dates, strategy_id, show_n)


# 4. Option B (Causal Chronological First-Confirmed: among all passing check_fn at 09:31 ET,
#    take the first top_n to confirm chronologically; tie-break same minute bar by gap_pct)
def build_trades_option_b_chrono(cs_pool_by_day, all_bars, dates, strategy_id, top_n):
    check_fn = FILTER_FNS[strategy_id]
    trades = []
    for dt in dates:
        cs_pool = cs_pool_by_day.get(dt, [])
        confirmed_today = []
        filter_rank = 0
        for c in cs_pool:
            tk = c["ticker"]
            bd = all_bars.get((tk, dt))
            if not check_fn(c, bd):
                continue
            filter_rank += 1
            rth = bd.get("rth", []) if bd else []
            entry = find_local_high(rth) if len(rth) >= PEAK_WINDOW * 2 + 1 else None
            if entry is None:
                continue
            exit_price = get_exit_price(bd, rth, entry["entry_time"], dt, tk)
            if exit_price is None:
                continue
            confirmed_today.append(make_trade_dict(c, dt, entry, exit_price, filter_rank))

        confirmed_today.sort(key=lambda t: (t["entry_time"], -t["gap_pct"]))
        trades.extend(confirmed_today[:top_n])
    return trades


def run_compounding(trades, dates, start_capital, fixed_slots=None):
    daily_trades = defaultdict(list)
    for t in trades:
        daily_trades[t["scan_date"]].append(t)

    capital = start_capital
    for dt in dates:
        day_trades = daily_trades.get(dt, [])
        if not day_trades or capital <= 0:
            continue
        n_t = len(day_trades)
        denom = max(n_t, fixed_slots) if fixed_slots is not None else n_t
        pos_size = capital / denom
        day_pnl = 0.0
        for t in day_trades:
            capped_pnl = max(t["pnl_eod"], -100.0)
            day_pnl += pos_size * capped_pnl / 100.0
        capital = max(0.0, capital + day_pnl)
    return capital


def summarize_mode(mode_name, builder_fn):
    print(f"\n{'='*115}")
    print(f"  {mode_name}")
    print(f"{'='*115}")
    rows = []
    strat_trades = {}
    for strat in STRATEGIES:
        sid = strat["id"]
        opt_n = strat["optimal_n"]
        trades = builder_fn(cs_pool_by_day, all_bars, dates, sid, opt_n)
        strat_trades[sid] = trades
        pnls = [t["pnl_eod"] for t in trades]
        m = compute_metrics(pnls)
        post_pnls = [t["pnl_eod"] for t in trades if t["scan_date"] >= POST_AUG_DATE]
        pm = compute_metrics(post_pnls)
        final_cap_la = run_compounding(trades, dates, CAPITAL, fixed_slots=None)
        final_cap_causal = run_compounding(trades, dates, CAPITAL, fixed_slots=opt_n)
        worst = min(pnls) if pnls else 0.0
        rows.append([
            strat["name"], opt_n,
            m["n"], f"{m['avg']:+.2f}%", f"{m['wr']*100:.0f}%", f"{m['sharpe']:.3f}", f"{worst:+.2f}%",
            pm["n"], f"{pm['avg']:+.2f}%" if post_pnls else "-", f"{pm['sharpe']:.3f}" if post_pnls else "-",
            f"${final_cap_la:,.0f}", f"${final_cap_causal:,.0f}"
        ])

    combined = []
    for dt in dates:
        seen = {}
        for strat in STRATEGIES:
            sid = strat["id"]
            for t in strat_trades[sid]:
                if t["scan_date"] != dt:
                    continue
                tk = t["ticker"]
                if tk not in seen:
                    seen[tk] = {**t, "strategies": [sid]}
                else:
                    seen[tk]["strategies"].append(sid)
        combined.extend(seen.values())

    cpnls = [t["pnl_eod"] for t in combined]
    cm = compute_metrics(cpnls)
    cpost = [t["pnl_eod"] for t in combined if t["scan_date"] >= POST_AUG_DATE]
    cpm = compute_metrics(cpost)
    cfinal_la = run_compounding(combined, dates, CAPITAL, fixed_slots=None)
    cfinal_3 = run_compounding(combined, dates, CAPITAL, fixed_slots=3)
    cworst = min(cpnls) if cpnls else 0.0
    rows.append([
        "COMBINED (dedup)", "-",
        cm["n"], f"{cm['avg']:+.2f}%", f"{cm['wr']*100:.0f}%", f"{cm['sharpe']:.3f}", f"{cworst:+.2f}%",
        cpm["n"], f"{cpm['avg']:+.2f}%", f"{cpm['sharpe']:.3f}",
        f"${cfinal_la:,.0f}", f"${cfinal_3:,.0f} (cap/3)"
    ])

    print(tabulate(
        rows,
        headers=[
            "Strategy", "N", "Trades", "Avg%", "WR", "Sharpe", "Worst%",
            "PostN", "PostAvg", "PostSh", "Comp$(Cap/n_t)", "Comp$(Causal)"
        ],
        tablefmt="simple",
        stralign="right"
    ))
    return strat_trades, combined


cs_pool_by_day, all_bars, dates, t_total = run_pipeline()

bt_trades, bt_comb = summarize_mode(
    "1. CURRENT BACKTEST (Hindsight Walk-Down: skips unconfirmed at 09:31, walks down pool using full 09:30-09:44 window)",
    build_trades_current_bt
)

optA_trades, optA_comb = summarize_mode(
    "2. OPTION A1 — LIVE PRIMARY (Lock Top N passing filter at 09:31 ET; trade only if those N confirm local high)",
    build_trades_option_a_primary
)

optA2_trades, optA2_comb = summarize_mode(
    "3. OPTION A2 — LIVE BANNER (Lock show_n = max(N, 2) at 09:31 ET; trade if #1 or #2 backup confirms)",
    build_trades_option_a_banner
)

optB_trades, optB_comb = summarize_mode(
    "4. OPTION B — CAUSAL CHRONOLOGICAL (All stocks passing filter at 09:31 ET; take first N to confirm local high in real time)",
    build_trades_option_b_chrono
)

print(f"\n\n{'='*115}")
print("  EXACT TRADE DIFFERENCES: CURRENT BACKTEST vs OPTION A1 (LIVE PRIMARY) & OPTION B (CHRONOLOGICAL)")
print(f"{'='*115}")

for strat in STRATEGIES:
    sid = strat["id"]
    t_bt = {(t["scan_date"], t["ticker"]): t for t in bt_trades[sid]}
    t_a1 = {(t["scan_date"], t["ticker"]): t for t in optA_trades[sid]}
    t_b  = {(t["scan_date"], t["ticker"]): t for t in optB_trades[sid]}

    only_in_bt_vs_a1 = [t_bt[k] for k in sorted(set(t_bt.keys()) - set(t_a1.keys()))]
    print(f"\n  [{strat['name']} (N={strat['optimal_n']})]")
    print(f"    Current BT trades: {len(t_bt)}  |  Option A1 (Live Primary) trades: {len(t_a1)}  |  Option B (Chrono) trades: {len(t_b)}")
    if only_in_bt_vs_a1:
        print(f"    Trades in Current BT that were NOT in Top-{strat['optimal_n']} at 09:31 ET (Rank #{strat['optimal_n']+1}+ taken because higher rank failed to form local high):")
        for t in only_in_bt_vs_a1:
            b_same_day = [x for x in optB_trades[sid] if x["scan_date"] == t["scan_date"]]
            b_str = ", ".join(f"{x['ticker']}@{x['entry_time']}(#{x['filter_rank']})" for x in b_same_day)
            print(f"      - {t['scan_date']}  {t['ticker']:6s}  FilterRank=#{t['filter_rank']}  gap={t['gap_pct']*100:+.1f}%  "
                  f"entry={t['entry_time']}  PnL={t['pnl_eod']:+.2f}%  |  [Option B took: {b_str}]")

    diff_bt_vs_b = set(t_bt.keys()) ^ set(t_b.keys())
    if diff_bt_vs_b:
        print(f"    Dates where Current BT (Gap-Rank Hindsight) != Option B (Real-Time First-Confirmed):")
        diff_dates = sorted({d for d, _ in diff_bt_vs_b})
        for dt in diff_dates:
            bt_d = [x for x in bt_trades[sid] if x["scan_date"] == dt]
            b_d  = [x for x in optB_trades[sid] if x["scan_date"] == dt]
            bt_s = ", ".join(f"{x['ticker']}(#{x['filter_rank']} @{x['entry_time']} PnL={x['pnl_eod']:+.2f}%)" for x in bt_d)
            b_s  = ", ".join(f"{x['ticker']}(#{x['filter_rank']} @{x['entry_time']} PnL={x['pnl_eod']:+.2f}%)" for x in b_d)
            print(f"      * {dt}:  BT=[{bt_s}]   vs   Chrono=[{b_s}]")
