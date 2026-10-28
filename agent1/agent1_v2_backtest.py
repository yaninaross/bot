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
BACKTEST_END   = '2026-10-17'

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

async def _fetch_bars_batch(pairs):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession() as session:
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
    async with aiohttp.ClientSession() as session:
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

    pm_after_830 = [(t, b) for t, b in pm_bars if t >= '08:30']
    if pm_after_830 and prior_close > 0:
        pm_high = max(b['h'] for _, b in pm_after_830)
        last_pm_close = pm_after_830[-1][1]['c']
        denom = pm_high - prior_close
        pm_retrace = (pm_high - last_pm_close) / denom if denom > 0 else 0
        if pm_retrace < 0.08:
            return False
    else:
        return False

    if rth_bars:
        bar0 = rth_bars[0][1]
        if bar0['c'] > bar0['o']:
            return False
    else:
        return False

    return True


def check_c3_c5(candidate, bar_data):
    gap = candidate.get('gap_pct', 0)
    if 0.15 <= gap < 0.30:
        return False
    return check_c5(candidate, bar_data)


def check_c7(candidate, bar_data):
    if bar_data is None: return False
    pm_bars = bar_data.get('pm', [])
    pm_after_830 = [(t, b) for t, b in pm_bars if t >= '08:30']
    if len(pm_after_830) >= 2:
        first_pm = pm_after_830[0][1]['o']
        last_pm = pm_after_830[-1][1]['c']
        pm_trend = (last_pm - first_pm) / first_pm if first_pm > 0 else 0
        return pm_trend < 0
    return False


def check_c9(candidate, bar_data):
    if bar_data is None: return False
    rth_bars = bar_data.get('rth', [])
    if len(rth_bars) >= 2:
        open_p = rth_bars[0][1]['o']
        bar1_close = rth_bars[1][1]['c']
        drop = (bar1_close - open_p) / open_p if open_p > 0 else 0
        return drop < 0
    return False


def check_c3(candidate, bar_data):
    gap = candidate.get('gap_pct', 0)
    return not (0.15 <= gap < 0.30)


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
        ch = rth_bars[i][1]['h']
        if all(rth_bars[i - j][1]['h'] < ch and rth_bars[i + j][1]['h'] < ch
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(rth_bars):
                return {
                    'entry_price': rth_bars[ci][1]['c'],
                    'entry_time': rth_bars[ci][0],
                    'entry_type': 'local_high',
                    'peak_bar': i,
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
            'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
            'open_price': cur_p, 'scan_date': dt_str,
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

    # Phase 2a: AH reference prices
    gap_min_final = GAP_MIN_PCT / 100.0
    print(f"\n  Phase 2a: AH reference prices...")
    n_ah_found = 0
    n_stale = 0
    for dt in dates:
        cands = all_candidates.get(dt, [])
        for c in cands[:PRE_FILTER_N]:
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
                fresh_gap = (c['open_price'] - ah_ref) / ah_ref
                c['gap_pct'] = fresh_gap
                if fresh_gap < gap_min_final:
                    c['ah_stale'] = True
                    n_stale += 1
    print(f"  → AH ref: {n_ah_found} tickers, {n_stale} stale excluded")

    # Phase 2b: re-rank at SCAN_TIME + gap erosion
    print(f"\n  Phase 2b: Re-rank + erosion...")
    n_eroded = 0
    for dt in dates:
        cands = all_candidates.get(dt, [])
        reranked = []
        for c in cands:
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
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

cs_pool_by_day, all_bars, dates, t_total = run_pipeline()

strat_trades = {}
for strat in STRATEGIES:
    sid = strat['id']
    n = strat['optimal_n']
    strat_trades[sid] = build_trades_v2(cs_pool_by_day, all_bars, dates, sid, n)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOUNDING PORTFOLIO SIMULATION
#
# Each strategy runs as its own compounding portfolio:
#   - Start with CAPITAL
#   - Each day, split today's capital equally across that day's trades
#   - PnL compounds: tomorrow's capital = today's ending capital
#
# Combined portfolio:
#   - Merge all strategies, deduplicate tickers per day
#   - Split capital across unique tickers
#   - Track which strategy sourced each trade for attribution
# ═══════════════════════════════════════════════════════════════════════════════

def run_compounding_portfolio(trades, dates, start_capital):
    daily_trades = defaultdict(list)
    for t in trades:
        daily_trades[t['scan_date']].append(t)

    capital = start_capital
    log = []
    daily_returns = []
    blown_up = False
    for dt in dates:
        day_trades = daily_trades.get(dt, [])
        if not day_trades or blown_up:
            daily_returns.append(0.0)
            log.append({'date': dt, 'n_trades': 0, 'capital_start': capital,
                        'day_pnl_dollar': 0, 'capital_end': capital,
                        'day_return_pct': 0, 'trades': []})
            continue

        n_t = len(day_trades)
        pos_size = capital / n_t
        day_pnl = 0.0
        trade_details = []
        for t in day_trades:
            capped_pnl = max(t['pnl_eod'], -100.0)
            trade_dollar = pos_size * capped_pnl / 100
            day_pnl += trade_dollar
            trade_details.append({**t, 'pos_size': pos_size, 'dollar_pnl': trade_dollar,
                                  'pnl_capped': capped_pnl != t['pnl_eod']})

        day_ret = day_pnl / capital * 100 if capital > 0 else 0
        daily_returns.append(day_ret)
        capital_end = capital + day_pnl
        if capital_end <= 0:
            capital_end = 0
            blown_up = True
        log.append({'date': dt, 'n_trades': n_t, 'capital_start': capital,
                    'day_pnl_dollar': day_pnl, 'capital_end': capital_end,
                    'day_return_pct': day_ret, 'trades': trade_details})
        capital = capital_end

    return log, daily_returns, capital


def build_combined_trades(strat_trades_dict, dates):
    combined = []
    for dt in dates:
        seen_tickers = {}
        for strat in STRATEGIES:
            sid = strat['id']
            for t in strat_trades_dict.get(sid, []):
                if t['scan_date'] != dt:
                    continue
                tk = t['ticker']
                if tk not in seen_tickers:
                    seen_tickers[tk] = {**t, 'strategies': [sid]}
                else:
                    seen_tickers[tk]['strategies'].append(sid)
        combined.extend(seen_tickers.values())
    return combined

combined_trades = build_combined_trades(strat_trades, dates)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-STRATEGY COMPOUNDING RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print(f"  COMPOUNDING PORTFOLIO — ${CAPITAL:,.0f} start  |  capital / n_trades per day  |  PnL rolls forward")
print(f"  {BACKTEST_START} → {BACKTEST_END}  |  Exit {EXIT_TIME} ET  |  No SL  |  Filter first → top N")
print(f"{'='*120}")


# ── Per-strategy standalone portfolios ────────────────────────────────────────

print(f"\n\n{'='*100}")
print(f"  PER-STRATEGY STANDALONE (each starts with ${CAPITAL:,.0f})")
print(f"{'='*100}\n")

strat_logs = {}
strat_daily_rets = {}
strat_final = {}
summary_rows = []

for strat in STRATEGIES:
    sid = strat['id']
    trades = strat_trades[sid]
    log, daily_rets, final_cap = run_compounding_portfolio(trades, dates, CAPITAL)
    strat_logs[sid] = log
    strat_daily_rets[sid] = daily_rets
    strat_final[sid] = final_cap

    n_trades = len(trades)
    pnls = [t['pnl_eod'] for t in trades]
    m = compute_metrics(pnls)
    total_return = (final_cap - CAPITAL) / CAPITAL * 100
    profit = final_cap - CAPITAL

    trading_days = [d for d in daily_rets if d != 0]
    daily_m = compute_metrics(trading_days) if trading_days else compute_metrics([])

    post_trades = [t for t in trades if t['scan_date'] >= POST_AUG_DATE]
    post_log = [l for l in log if l['date'] >= POST_AUG_DATE]
    post_trading_rets = [l['day_return_pct'] for l in post_log if l['n_trades'] > 0]
    post_m = compute_metrics([t['pnl_eod'] for t in post_trades]) if post_trades else compute_metrics([])

    summary_rows.append([
        strat['name'], strat['optimal_n'], n_trades,
        f"{m['avg']:+.2f}%", f"{m['wr']*100:.0f}%", f"{m['sharpe']:.3f}",
        f"${profit:+,.0f}", f"${final_cap:,.0f}", f"{total_return:+.1f}%",
        len(post_trades), f"{post_m['avg']:+.2f}%" if post_trades else '-',
        f"{post_m['sharpe']:.3f}" if post_trades else '-',
    ])

print(tabulate(summary_rows,
               headers=['Strategy', 'N', 'Trades', 'Avg%', 'WR', 'Sharpe',
                        'Profit', 'Final$', 'Return',
                        'PostN', 'PostAvg', 'PostSh'],
               tablefmt='simple', stralign='right'))

n_capped_total = 0
for strat in STRATEGIES:
    sid = strat['id']
    capped = [t for log_day in strat_logs[sid] for t in log_day.get('trades', []) if t.get('pnl_capped')]
    n_capped_total += len(capped)
    if capped:
        print(f"  {strat['name']}: {len(capped)} trade(s) capped at -100% "
              f"({', '.join(t['ticker']+' '+t['scan_date'] for t in capped)})")
    blown = strat_final[sid] <= 0
    if blown:
        blow_date = next((e['date'] for e in strat_logs[sid] if e['capital_end'] <= 0), '?')
        print(f"  {strat['name']}: BLOWN UP on {blow_date} (capital hit $0)")
if n_capped_total == 0:
    print("  No trades exceeded -100% loss cap")


# ── Combined portfolio (all strategies, deduplicated) ─────────────────────────

print(f"\n\n{'='*100}")
print(f"  COMBINED PORTFOLIO — all strategies, deduplicate same ticker per day")
print(f"{'='*100}")

comb_log, comb_daily_rets, comb_final = run_compounding_portfolio(combined_trades, dates, CAPITAL)
comb_profit = comb_final - CAPITAL
comb_return = comb_profit / CAPITAL * 100
comb_pnls = [t['pnl_eod'] for t in combined_trades]
comb_m = compute_metrics(comb_pnls)
comb_trading_rets = [r for r in comb_daily_rets if r != 0]
comb_dm = compute_metrics(comb_trading_rets) if comb_trading_rets else compute_metrics([])

print(f"\n  Start:  ${CAPITAL:,.0f}")
print(f"  Final:  ${comb_final:,.0f}  ({comb_return:+.1f}%)")
print(f"  Profit: ${comb_profit:+,.0f}")
print(f"  Trades: {len(combined_trades)}  |  Avg: {comb_m['avg']:+.2f}%  |  "
      f"WR: {comb_m['wr']*100:.0f}%  |  Sharpe: {comb_m['sharpe']:.3f}")
print(f"  Daily (trading days): Avg: {comb_dm['avg']:+.2f}%  |  Sharpe: {comb_dm['sharpe']:.3f}")


# ── Strategy attribution in combined portfolio ────────────────────────────────

print(f"\n\n{'='*100}")
print(f"  STRATEGY ATTRIBUTION — which strategy contributed what to combined PnL")
print(f"{'='*100}\n")

strat_attribution = defaultdict(lambda: {'n': 0, 'pnl_sum': 0.0, 'dollar_sum': 0.0, 'pnls': []})
for entry in comb_log:
    for t in entry['trades']:
        strats = t.get('strategies', ['?'])
        dollar_per_strat = t['dollar_pnl'] / len(strats)
        pnl_per_strat = t['pnl_eod'] / len(strats)
        for sid in strats:
            strat_attribution[sid]['n'] += 1
            strat_attribution[sid]['pnl_sum'] += pnl_per_strat
            strat_attribution[sid]['dollar_sum'] += dollar_per_strat
            strat_attribution[sid]['pnls'].append(t['pnl_eod'])

attr_rows = []
for strat in STRATEGIES:
    sid = strat['id']
    a = strat_attribution[sid]
    if a['n'] == 0:
        attr_rows.append([strat['name'], 0, '-', '-', '-', '$0', '0%'])
        continue
    a_m = compute_metrics(a['pnls'])
    share = a['dollar_sum'] / comb_profit * 100 if comb_profit != 0 else 0
    attr_rows.append([
        strat['name'], a['n'],
        f"{a_m['avg']:+.2f}%", f"{a_m['wr']*100:.0f}%", f"{a_m['sharpe']:.3f}",
        f"${a['dollar_sum']:+,.0f}", f"{share:+.1f}%",
    ])
print(tabulate(attr_rows,
               headers=['Strategy', 'Trades', 'Avg%', 'WR', 'Sharpe', '$Contribution', '% of Total'],
               tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE LOG — combined portfolio with compounding
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*140}")
print(f"  COMBINED TRADE LOG — compounding ${CAPITAL:,.0f}")
print(f"{'='*140}")

trade_rows = []
trade_num = 0
for entry in comb_log:
    if entry['n_trades'] == 0:
        continue
    for t in entry['trades']:
        trade_num += 1
        strats = ', '.join(t.get('strategies', ['?']))
        period = 'PRE' if t['scan_date'] < POST_AUG_DATE else 'POST'
        wl = 'W' if t['pnl_eod'] > 0 else 'L'
        capped = '⚠CAP' if t.get('pnl_capped') else ''
        trade_rows.append([
            trade_num, t['scan_date'], t['ticker'],
            f"{t['gap_pct']*100:.1f}%",
            t['entry_time'],
            f"${t['entry_price']:.2f}",
            f"${t['exit_price']:.2f}",
            f"{t['pnl_eod']:+.2f}%",
            f"${t['pos_size']:,.0f}",
            f"${t['dollar_pnl']:+,.0f}",
            f"${entry['capital_end']:,.0f}",
            strats, wl, period, capped,
        ])
print()
print(tabulate(trade_rows,
               headers=['#', 'Date', 'Ticker', 'Gap', 'EntryT',
                        'EntryP', 'ExitP', 'PnL%', 'PosSize', '$PnL', 'Capital',
                        'Strats', 'W/L', 'Period', ''],
               tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# MONTHLY BREAKDOWN — combined + per-strategy
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*100}")
print(f"  MONTHLY BREAKDOWN — combined compounding portfolio")
print(f"{'='*100}")

monthly_data = defaultdict(lambda: {'trades': [], 'capital_start': 0, 'capital_end': 0, 'pnl': 0})
for entry in comb_log:
    mo = entry['date'][:7]
    if entry['n_trades'] > 0:
        monthly_data[mo]['trades'].extend(entry['trades'])
        monthly_data[mo]['pnl'] += entry['day_pnl_dollar']
    if monthly_data[mo]['capital_start'] == 0:
        monthly_data[mo]['capital_start'] = entry['capital_start']
    monthly_data[mo]['capital_end'] = entry['capital_end']

mo_rows = []
for mo in sorted(monthly_data):
    md = monthly_data[mo]
    trades_in_mo = md['trades']
    n_t = len(trades_in_mo)
    if n_t == 0:
        mo_rows.append([mo, 0, '-', '-', '-', '$0', f"${md['capital_end']:,.0f}", '-', '-'])
        continue
    pnls = [t['pnl_eod'] for t in trades_in_mo]
    arr = np.array(pnls)
    wr = float((arr > 0).mean())
    sh = float(np.mean(arr) / np.std(arr, ddof=1)) if len(arr) > 1 and np.std(arr, ddof=1) > 0 else 0
    mo_ret = md['pnl'] / md['capital_start'] * 100 if md['capital_start'] > 0 else 0
    mo_rows.append([
        mo, n_t, f"{np.mean(arr):+.2f}%", f"{wr*100:.0f}%", f"{sh:.3f}",
        f"${md['pnl']:+,.0f}", f"${md['capital_end']:,.0f}",
        f"{mo_ret:+.1f}%",
        f"{arr.min():+.2f}% / {arr.max():+.2f}%",
    ])
print()
print(tabulate(mo_rows,
               headers=['Month', 'N', 'Avg%', 'WR', 'Sharpe',
                        'Month$', 'Capital', 'Mo Return', 'Worst/Best'],
               tablefmt='simple', stralign='right'))

# Per-strategy monthly
for strat in STRATEGIES:
    sid = strat['id']
    trades = strat_trades[sid]
    if not trades:
        continue
    log = strat_logs[sid]

    print(f"\n  ── {strat['name']} N={strat['optimal_n']} — Monthly ──")
    s_monthly = defaultdict(lambda: {'trades': [], 'pnl': 0, 'cap_start': 0, 'cap_end': 0})
    for entry in log:
        mo = entry['date'][:7]
        if entry['n_trades'] > 0:
            s_monthly[mo]['trades'].extend(entry['trades'])
            s_monthly[mo]['pnl'] += entry['day_pnl_dollar']
        if s_monthly[mo]['cap_start'] == 0:
            s_monthly[mo]['cap_start'] = entry['capital_start']
        s_monthly[mo]['cap_end'] = entry['capital_end']

    smo_rows = []
    for mo in sorted(s_monthly):
        sd = s_monthly[mo]
        st = sd['trades']
        n_t = len(st)
        if n_t == 0:
            smo_rows.append([mo, 0, '-', '-', '$0', f"${sd['cap_end']:,.0f}"])
            continue
        arr = np.array([t['pnl_eod'] for t in st])
        wr = float((arr > 0).mean())
        mo_ret = sd['pnl'] / sd['cap_start'] * 100 if sd['cap_start'] > 0 else 0
        smo_rows.append([
            mo, n_t, f"{np.mean(arr):+.2f}%", f"{wr*100:.0f}%",
            f"${sd['pnl']:+,.0f}", f"${sd['cap_end']:,.0f}",
        ])
    print(tabulate(smo_rows,
                   headers=['Month', 'N', 'Avg%', 'WR', 'Month$', 'Capital'],
                   tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# PER-STRATEGY TRADE LOGS (standalone compounding)
# ═══════════════════════════════════════════════════════════════════════════════

for strat in STRATEGIES:
    sid = strat['id']
    trades = strat_trades[sid]
    log = strat_logs[sid]
    final_cap = strat_final[sid]
    pnls = [t['pnl_eod'] for t in trades]
    m = compute_metrics(pnls)

    print(f"\n\n{'='*130}")
    print(f"  {strat['name']} N={strat['optimal_n']}  —  {m['n']} trades  |  "
          f"Avg {m['avg']:+.2f}%  |  WR {m['wr']*100:.0f}%  |  Sharpe {m['sharpe']:.3f}  |  "
          f"${CAPITAL:,.0f} → ${final_cap:,.0f} ({(final_cap-CAPITAL)/CAPITAL*100:+.1f}%)")
    print(f"{'='*130}")

    pre = [t for t in trades if t['scan_date'] < POST_AUG_DATE]
    post = [t for t in trades if t['scan_date'] >= POST_AUG_DATE]
    if pre:
        pm = compute_metrics([t['pnl_eod'] for t in pre])
        print(f"  Pre-Aug:  {pm['n']} trades  avg={pm['avg']:+.2f}%  WR={pm['wr']*100:.0f}%  Sharpe={pm['sharpe']:.3f}")
    if post:
        pm = compute_metrics([t['pnl_eod'] for t in post])
        print(f"  Post-Aug: {pm['n']} trades  avg={pm['avg']:+.2f}%  WR={pm['wr']*100:.0f}%  Sharpe={pm['sharpe']:.3f}")

    rows = []
    for entry in log:
        for t in entry['trades']:
            wl = 'W' if t['pnl_eod'] > 0 else 'L'
            period = 'PRE' if t['scan_date'] < POST_AUG_DATE else 'POST'
            rows.append([
                t['scan_date'], t['ticker'],
                f"{t['gap_pct']*100:.1f}%",
                t['entry_time'],
                f"${t['entry_price']:.2f}",
                f"${t['exit_price']:.2f}",
                f"{t['pnl_eod']:+.2f}%",
                f"${t['pos_size']:,.0f}",
                f"${t['dollar_pnl']:+,.0f}",
                f"${entry['capital_end']:,.0f}",
                wl, period,
            ])
    print()
    print(tabulate(rows,
                   headers=['Date', 'Ticker', 'Gap', 'EntryT',
                            'EntryP', 'ExitP', 'PnL%', 'PosSize', '$PnL', 'Capital',
                            'W/L', 'Period'],
                   tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

strat_colors = ['#e74c3c', '#9b59b6', '#3498db', '#e67e22', '#27ae60']
comb_color = '#1a1a1a'

d_dates_plot = [datetime.strptime(dt, '%Y-%m-%d') for dt in dates]
aug15 = datetime.strptime('2026-08-15', '%Y-%m-%d')

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle(f'Agent 1 v2 — Compounding Backtest (${CAPITAL/1000:.0f}K start)\n'
             f'{BACKTEST_START} → {BACKTEST_END}  |  Exit {EXIT_TIME} ET  |  '
             f'Capital / n_trades per day  |  PnL compounds',
             fontsize=11, fontweight='bold')


# Chart 1: Per-strategy equity curves (standalone compounding)
ax = axes[0, 0]
for idx, strat in enumerate(STRATEGIES):
    sid = strat['id']
    log = strat_logs[sid]
    caps = [entry['capital_end'] for entry in log]
    ax.plot(d_dates_plot, caps, color=strat_colors[idx], linewidth=2,
            label=f'{strat["name"]} N={strat["optimal_n"]}: ${strat_final[sid]:,.0f}')
ax.axvline(aug15, color='red', ls='--', lw=1, alpha=0.5)
ax.axhline(CAPITAL, color='#333', lw=0.8, ls=':')
ax.set_title('Per-Strategy Compounding Equity')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)


# Chart 2: Combined portfolio equity curve
ax = axes[0, 1]
comb_caps = [entry['capital_end'] for entry in comb_log]
ax.plot(d_dates_plot, comb_caps, color=comb_color, linewidth=2.5,
        label=f'Combined: ${comb_final:,.0f} ({comb_return:+.1f}%)')
ax.fill_between(d_dates_plot, CAPITAL, comb_caps, alpha=0.1, color=comb_color)
ax.axvline(aug15, color='red', ls='--', lw=1, alpha=0.5, label='Aug 15')
ax.axhline(CAPITAL, color='#333', lw=0.8, ls=':')
ax.set_title('Combined Portfolio (all strategies, deduped)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)


# Chart 3: Strategy attribution (bar chart)
ax = axes[1, 0]
strat_names = [s['name'] for s in STRATEGIES]
strat_dollars = [strat_attribution[s['id']]['dollar_sum'] for s in STRATEGIES]
bars = ax.bar(strat_names, strat_dollars, color=strat_colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, strat_dollars):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'${val:+,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(0, color='#333', lw=0.8)
ax.set_title('Strategy $ Contribution to Combined Portfolio')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_xticklabels(strat_names, rotation=15, ha='right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')


# Chart 4: Monthly PnL (combined)
ax = axes[1, 1]
mo_labels = sorted(monthly_data.keys())
mo_pnls = [monthly_data[mo]['pnl'] for mo in mo_labels]
mo_colors = ['#27ae60' if p >= 0 else '#e74c3c' for p in mo_pnls]
bars = ax.bar(mo_labels, mo_pnls, color=mo_colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, mo_pnls):
    y_pos = bar.get_height() if val >= 0 else bar.get_height()
    va = 'bottom' if val >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f'${val:+,.0f}', ha='center', va=va, fontsize=8, fontweight='bold')
ax.axhline(0, color='#333', lw=0.8)
ax.set_title('Monthly PnL (Combined, Compounding)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_xticklabels(mo_labels, rotation=30, ha='right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.92])
p1 = os.path.join(BASE_DIR, 'agent1_v2_backtest.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {p1}")

print(f"\n  Total runtime: {time.time() - t_total:.0f}s ({(time.time() - t_total)/60:.1f} min)")
