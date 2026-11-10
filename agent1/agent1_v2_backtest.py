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
BACKTEST_END   = '2026-10-30'

GAP_MIN_PCT    = 3.0
PRICE_MIN      = 10.0
PRICE_MAX      = 100.0
FLOAT_MIN      = 10_000_000
PRE_FILTER_N   = 30
GAP_CONSUMED_MAX = 0.50

SCAN_TIME_ET      = '09:31'
EXIT_TIME_INST_ET = '11:30' # 08:30 PST — Rule #3 Tier 1: $50+ local_high & C2 <15% gap
EXIT_TIME_LH_ET   = '12:30' # 09:30 PST — Rule #3 Tier 2: <$50 local_high morning spikes
EXIT_TIME         = '14:00' # 11:00 PST — Rule #3 Tier 3: bar0_flush, _B2 bounce, _W11 pyramid (Hard Flat)
RTH_CLOSE_ET      = '15:59' # 13:00 PST — Official RTH close (used by unmodified S6/S1/S12 baselines)

MIN_RTH_BARS   = 2          # min RTH bars with volume > 0 by scan time
MIN_OPEN_VOL   = 5_000      # min shares in bar 0 (09:30)

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
MAX_PEAK_TIME_ET  = '09:41'
MAX_ENTRY_TIME_ET = '09:44'
MIN_PULLBACK_PCT  = 0.5
MAX_MISSING_BARS_MOD_GAP = 1
MAX_OPEN_RUNUP_MOD_GAP   = 0.07
MOD_GAP_PEAK_STOP_BUFFER = None
MOD_GAP_MAX_PRICE        = None
MOD_GAP_MAX_PRIOR_DVOL   = None
USE_PRIORITY_TIER_RANKING    = True
TIER1_MOD_GAP_MAX_PRICE      = 25.0
TIER1_MOD_GAP_MAX_PRIOR_DVOL = 25_000_000

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
_SPAC_KEYWORDS  = ('ACQUISITION', 'BLANK CHECK', 'SPAC')

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
    # ── Legacy v2 Baselines (Fixed 1/N Slot Sizing, 13:00 PST RTH Close) ──────
    {'id': 'S6',         'name': 'S6: Base (Skip12-30, N=3, Fix33%)',      'base_filter': 'S6',  'optimal_n': 3, 'sizing_mode': 'fixed'},
    {'id': 'S1',         'name': 'S1: Base (Skip15-30, N=3, Fix33%)',      'base_filter': 'S1',  'optimal_n': 3, 'sizing_mode': 'fixed'},
    {'id': 'S12',        'name': 'S12: Base (Skip15-30, N=2, Fix50%)',     'base_filter': 'S12', 'optimal_n': 2, 'sizing_mode': 'fixed'},
    # ── Tier A: Morning Entry + Active 50% Sizing + Rule #3 RTH Exits ─────────
    {'id': 'S6+C7',      'name': 'Combo A1: S6+C7 (Bar0+Cont+Act50%)',     'base_filter': 'S6',  'optimal_n': 3, 'sizing_mode': 'active_cap50',
     'allow_bar0_flush': True, 'continue_on_failed_pb': True, 'use_rule3_exit': True},
    {'id': 'S1+C7+1130', 'name': 'Combo A4: S1+C7+08:30Exit(<15%Gap)',     'base_filter': 'S1',  'optimal_n': 3, 'sizing_mode': 'active_cap50',
     'allow_bar0_flush': True, 'continue_on_failed_pb': True, 'hybrid_exit_lt15_1130': True, 'use_rule3_exit': True},
    {'id': 'S12+C7',     'name': 'Combo A3: S12+C7 (Bar0+Cont, N=2 50%)',  'base_filter': 'S12', 'optimal_n': 2, 'sizing_mode': 'fixed',
     'allow_bar0_flush': True, 'continue_on_failed_pb': True, 'use_rule3_exit': True},
    # ── Tier B: 08:00 PST Winner Pyramid Add-On (C8 Hybrid + Rule #3 Exits) ───
    {'id': 'S6+C8',      'name': 'Combo B1: S6+C8 (C7+08:00 Pyramid)',     'base_filter': 'S6',  'optimal_n': 3, 'sizing_mode': 'active_cap50',
     'allow_bar0_flush': True, 'continue_on_failed_pb': True, 'addon_1100_winner': True, 'use_rule3_exit': True},
    # ── Tier C: Deployed Live Combos C2 / C1 / C3 (De-Overfitted + Rule #3 Exits) ────
    {'id': 'S1+C6+1130',  'name': 'Combo C2: S1+C6+08:30 (C7+Bnc+08:30)',   'base_filter': 'S1',  'optimal_n': 3, 'sizing_mode': 'active_cap50',
     'allow_bar0_flush': True, 'continue_on_failed_pb': True, 'hybrid_exit_lt15_1130': True, 'addon_bounce_reshort': True, 'use_rule3_exit': True},
    {'id': 'S6+C6',       'name': 'Combo C1: S6+C6 (C7+Bounce ReShort)',    'base_filter': 'S6',  'optimal_n': 3, 'sizing_mode': 'active_cap50',
     'allow_bar0_flush': True, 'continue_on_failed_pb': True, 'addon_bounce_reshort': True, 'use_rule3_exit': True},
    {'id': 'S6+Full',     'name': 'Combo C3: S6+Full (C7+Bounce+08:00)',    'base_filter': 'S6',  'optimal_n': 3, 'sizing_mode': 'active_cap50',
     'allow_bar0_flush': True, 'continue_on_failed_pb': True, 'addon_bounce_reshort': True, 'addon_1100_winner': True, 'use_rule3_exit': True},
]

STRAT_BY_ID = {s['id']: s for s in STRATEGIES}

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


def _passes_mod_gap_retail_check(candidate):
    nm = (candidate.get('name') or (_ticker_ref.get(candidate.get('ticker', ''), {}) or {}).get('name') or '').upper()
    if any(w in nm for w in _SPAC_KEYWORDS):
        return False
    gap = candidate.get('gap_pct', 0)
    if gap < 0.15:
        if MOD_GAP_MAX_PRICE is not None and candidate.get('open_price', 0) > MOD_GAP_MAX_PRICE:
            return False
        prior_dvol = candidate.get('prior_dvol')
        if prior_dvol is None:
            dt = candidate.get('scan_date')
            tk = candidate.get('ticker')
            if dt and tk:
                p_str = candidate.get('prior_str') or _prior_td(dt).strftime('%Y-%m-%d')
                pv = _gd_cache.get(p_str, {}).get(tk, {}).get('v', 0)
                prior_dvol = pv * candidate.get('prior_close', 0)
        if MOD_GAP_MAX_PRIOR_DVOL is not None and prior_dvol is not None and prior_dvol >= MOD_GAP_MAX_PRIOR_DVOL:
            return False
    return True


def _is_tier1_setup(candidate):
    """Priority Tier 1: Parabolic gaps (>=30%) or Retail Small-Cap Moderate gaps (<15%, Price<=$25, Prior $Vol<$25M).
    Priority Tier 2: Institutional / Mid-Cap Moderate gaps (<15%, Price>$25 or Prior $Vol>=$25M) — used to fill remaining daily slots.
    """
    gap = candidate.get('gap_pct', 0)
    if gap >= 0.30:
        return True
    op = candidate.get('open_price', 0)
    prior_dvol = candidate.get('prior_dvol')
    if prior_dvol is None:
        dt = candidate.get('scan_date')
        tk = candidate.get('ticker')
        if dt and tk:
            p_str = candidate.get('prior_str') or _prior_td(dt).strftime('%Y-%m-%d')
            pv = _gd_cache.get(p_str, {}).get(tk, {}).get('v', 0)
            prior_dvol = pv * candidate.get('prior_close', 0)
            candidate['prior_dvol'] = prior_dvol
    if op <= TIER1_MOD_GAP_MAX_PRICE and (prior_dvol is None or prior_dvol < TIER1_MOD_GAP_MAX_PRIOR_DVOL):
        return True
    return False


def check_s6(candidate, bar_data):
    if candidate.get('open_price', 0) > PRICE_MAX:
        return False
    gap = candidate.get('gap_pct', 0)
    raw_gap = candidate.get('raw_open_gap', gap)
    if (0.12 <= gap < 0.30) or (0.12 <= raw_gap < 0.30):
        return False
    return _passes_mod_gap_retail_check(candidate)


def check_s1(candidate, bar_data):
    if candidate.get('open_price', 0) > PRICE_MAX:
        return False
    gap = candidate.get('gap_pct', 0)
    raw_gap = candidate.get('raw_open_gap', gap)
    if (0.15 <= gap < 0.30) or (0.15 <= raw_gap < 0.30):
        return False
    return _passes_mod_gap_retail_check(candidate)


def check_s12(candidate, bar_data):
    return check_s1(candidate, bar_data)


FILTER_FNS = {
    'S6': check_s6,
    'S1': check_s1,
    'S12': check_s12,
    'C5': check_c5,
    'C3+C5': check_c3_c5,
    'C7': check_c7,
    'C9': check_c9,
    'C3': check_c3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL HIGH & HYBRID ENTRY / ADD-ON HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_high(rth_bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    min_entry_bar=0, min_pullback_pct=MIN_PULLBACK_PCT,
                    open_price=None, gap_pct=None, ref_price=None,
                    allow_bar0_flush=False, continue_on_failed_pb=False,
                    tweezer_tip_guard=False):
    if len(rth_bars) < window * 2 + 1:
        return None

    # U4: Bar 0 Opening Flush (09:30 peak confirmed at 09:33 ET bar 3)
    if allow_bar0_flush and len(rth_bars) >= 4:
        b0 = rth_bars[0][1]
        h0 = b0['h']
        if b0['c'] <= b0['o'] and all(rth_bars[j][1]['h'] < h0 for j in range(1, 4)):
            ci = 3
            ep = rth_bars[ci][1]['c']
            et = rth_bars[ci][0]
            pb_pct = (h0 - ep) / h0 * 100.0 if h0 > 0 else 0.0
            drop_open_pct = (open_price - ep) / open_price * 100.0 if (open_price and open_price > 0) else 0.0
            gap_range = (open_price - ref_price) if (open_price and ref_price and open_price > ref_price) else 0.0
            gap_ret = (ep - ref_price) / gap_range if gap_range > 0 else 0.0
            req_pb = min_pullback_pct if min_pullback_pct is not None else 0.0
            if pb_pct >= req_pb and drop_open_pct >= 1.0 and gap_ret >= 0.35:
                mod_ok = True
                if gap_pct is not None and gap_pct < 0.15:
                    clock_idx = (int(et[:2]) - 9) * 60 + (int(et[3:5]) - 30)
                    if (clock_idx - ci) > MAX_MISSING_BARS_MOD_GAP:
                        mod_ok = False
                    if open_price and open_price > 0 and (ep - open_price) / open_price >= MAX_OPEN_RUNUP_MOD_GAP:
                        mod_ok = False
                if mod_ok:
                    return {
                        'entry_price': ep,
                        'entry_time': et,
                        'entry_bar_idx': ci,
                        'entry_type': 'bar0_flush',
                        'peak_bar': 0,
                        'peak_time': rth_bars[0][0],
                        'peak_high': h0,
                        'pullback_pct': round(pb_pct, 2),
                    }

    start = max(window, min_entry_bar - window)
    for i in range(start, min(max_peak_bar, len(rth_bars) - window)):
        if rth_bars[i][0] > MAX_PEAK_TIME_ET:
            break
        ch = rth_bars[i][1]['h']
        if all(rth_bars[i - j][1]['h'] < ch and rth_bars[i + j][1]['h'] < ch
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(rth_bars) and rth_bars[ci][0] <= MAX_ENTRY_TIME_ET:
                ep = rth_bars[ci][1]['c']
                et = rth_bars[ci][0]
                pb_pct = (ch - ep) / ch * 100.0 if ch > 0 else 0.0
                if min_pullback_pct is not None and pb_pct < min_pullback_pct:
                    if continue_on_failed_pb:
                        continue
                    return None
                # Option 1: Tweezer-Aware Tip Guard (disabled by default after 12M OOS audit)
                if tweezer_tip_guard:
                    max_prev3 = max(rth_bars[i - j][1]['h'] for j in range(1, window + 1))
                    max_prev23 = max(rth_bars[i - j][1]['h'] for j in (2, 3))
                    tip1 = (ch - max_prev3) / ch * 100.0 if ch > 0 else 0.0
                    tip23 = (ch - max_prev23) / ch * 100.0 if ch > 0 else 0.0
                    if not ((tip1 >= 0.15) or (tip23 >= 0.50 and pb_pct >= 3.0)):
                        if continue_on_failed_pb:
                            continue
                        return None
                if gap_pct is not None and gap_pct < 0.15:
                    clock_idx = (int(et[:2]) - 9) * 60 + (int(et[3:5]) - 30)
                    if (clock_idx - ci) > MAX_MISSING_BARS_MOD_GAP:
                        return None
                    if open_price and open_price > 0 and (ep - open_price) / open_price >= MAX_OPEN_RUNUP_MOD_GAP:
                        return None
                stop_price = (
                    round(ch * (1.0 + MOD_GAP_PEAK_STOP_BUFFER), 2)
                    if (MOD_GAP_PEAK_STOP_BUFFER is not None and gap_pct is not None and gap_pct < 0.15)
                    else None
                )
                return {
                    'entry_price': ep,
                    'entry_time': et,
                    'entry_bar_idx': ci,
                    'entry_type': 'local_high',
                    'peak_bar': i,
                    'peak_time': rth_bars[i][0],
                    'peak_high': ch,
                    'pullback_pct': round(pb_pct, 2),
                    'stop_price': stop_price,
                }
            return None
    return None


def find_secondary_bounce_reshort(rth_bars, primary_entry, min_mfe_pct=3.0, min_bounce_pct=1.5, min_profit_pct=0.5):
    """U7b: Check if an open morning trade reaches >=3% MFE, then forms a 3-bar bounce local high (09:55-13:30 ET) while still in profit."""
    ep1 = primary_entry['entry_price']
    entry_idx = primary_entry.get('entry_bar_idx', primary_entry['peak_bar'] + PEAK_WINDOW)
    start_idx = max(entry_idx + 4, 25)
    running_low = ep1
    for k in range(entry_idx + 1, min(start_idx, len(rth_bars))):
        if rth_bars[k][1]['l'] < running_low:
            running_low = rth_bars[k][1]['l']

    window = PEAK_WINDOW
    for i in range(start_idx, min(len(rth_bars) - window, 240)):
        t_str = rth_bars[i][0]
        if t_str < '09:55':
            if rth_bars[i][1]['l'] < running_low:
                running_low = rth_bars[i][1]['l']
            continue
        if t_str > '13:30':
            break
        mfe_pct = (ep1 - running_low) / ep1 * 100.0 if ep1 > 0 else 0.0
        ch = rth_bars[i][1]['h']
        if mfe_pct >= min_mfe_pct:
            bounce_pct = (ch - running_low) / running_low * 100.0 if running_low > 0 else 0.0
            if bounce_pct >= min_bounce_pct:
                if all(rth_bars[i - j][1]['h'] < ch and rth_bars[i + j][1]['h'] < ch for j in range(1, window + 1)):
                    ci = i + window
                    ep2 = rth_bars[ci][1]['c']
                    pb2 = (ch - ep2) / ch * 100.0 if ch > 0 else 0.0
                    cur_profit = (ep1 - ep2) / ep1 * 100.0 if ep1 > 0 else 0.0
                    if pb2 >= 0.4 and cur_profit >= min_profit_pct and rth_bars[ci][0] <= '13:35':
                        return {
                            'entry_price': ep2,
                            'entry_time': rth_bars[ci][0],
                            'entry_bar_idx': ci,
                            'entry_type': 'bounce_reshort',
                            'peak_bar': i,
                            'peak_time': t_str,
                            'peak_high': ch,
                            'pullback_pct': round(pb2, 2),
                        }
        if rth_bars[i][1]['l'] < running_low:
            running_low = rth_bars[i][1]['l']
    return None


def get_target_exit_et(open_price, gap_pct, entry_type, hybrid_exit_lt15_1130=False, use_rule3_exit=True):
    """
    Rule #3 Signal-Adaptive 3-Tier RTH Exit Schedule (returns ET 'HH:MM'; PST = ET - 3h):
      - Tier 1 (08:30 PST / '11:30' ET): $50+ `local_high` morning spikes, or C2 `<15%` gap morning trades
      - Tier 2 (09:30 PST / '12:30' ET): `< $50` `local_high` morning spikes
      - Tier 3 (11:00 PST / '14:00' ET — Hard Flat): `bar0_flush` collapses (except C2 `<15%`), `_B2` bounces, `_W11` pyramids
    Legacy unmodified baselines (`use_rule3_exit=False`) exit at `RTH_CLOSE_ET` ('15:59' ET / 13:00 PST).
    """
    if not use_rule3_exit:
        return RTH_CLOSE_ET
    if entry_type == 'local_high':
        if (open_price is not None and open_price >= 50.0) or (hybrid_exit_lt15_1130 and gap_pct is not None and gap_pct < 0.15):
            return EXIT_TIME_INST_ET
        return EXIT_TIME_LH_ET
    if entry_type == 'bar0_flush':
        if hybrid_exit_lt15_1130 and gap_pct is not None and gap_pct < 0.15:
            return EXIT_TIME_INST_ET
        return EXIT_TIME
    return EXIT_TIME


def get_exit_price_at(bd, rth_bars, entry_time_et, target_exit_et, dt, tk):
    """Get causal RTH exit price at target_exit_et (e.g. '11:30', '12:30', '14:00', or '15:59' RTH close).
    Strictly uses Regular Trading Hours (rth_bars) — never after-hours (ah) dark-pool prints.
    """
    best_p = None
    for t_str, b in rth_bars:
        if t_str <= target_exit_et:
            if t_str >= entry_time_et:
                best_p = b['c']
        else:
            break
    if best_p is None and target_exit_et >= RTH_CLOSE_ET:
        gd = _gd_cache.get(dt, {}).get(tk, {})
        if gd.get('c'):
            best_p = gd['c']
    return best_p


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
        prev_v = prior[tk].get('v', 0)
        cur_p = today[tk]['o']
        if cur_p <= 0 or cur_p < PRICE_MIN: continue
        gap = (cur_p - prev_c) / prev_c
        if gap < gap_min: continue
        fs = _FLOAT.get(tk)
        if fs is not None and fs < FLOAT_MIN: continue
        candidates.append({
            'ticker': tk, 'gap_pct': gap, 'raw_open_gap': gap,
            'prior_close': prev_c, 'prior_dvol': prev_c * prev_v,
            'open_price': cur_p, 'scan_date': dt_str,
            'prior_str': prior_str,
        })
    candidates.sort(key=lambda x: -x['gap_pct'])
    return candidates


def run_pipeline():
    t_total = time.time()

    print(f"\n{'='*80}")
    print(f"  AGENT 1 v2 — BACKTEST (aligned with agent1_v2.py + Rule #3 RTH Exit Schedule)")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  RTH Exits (Rule #3): 08:30 ($50+ LH / C2<15%) | 09:30 (<$50 LH) | 11:00 PST (Flush/B2/W11)")
    print(f"{'='*80}")

    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"\n  {n_dates} trading days")

    # Fast local cache check (avoids redundant 2-min Polygon API fetch when running on cached window)
    try:
        _base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _base_dir = os.getcwd()
    cache_path = os.path.join(_base_dir, f".v2_pipeline_cache_{BACKTEST_END.replace('-', '')}.pkl")
    loaded_from_cache = False
    all_bars = {}
    if os.path.exists(cache_path):
        try:
            import pickle
            with open(cache_path, 'rb') as f:
                cs_pool_cached, cached_bars, cached_dates, cached_gd = pickle.load(f)
            if cached_dates == dates:
                _gd_cache.update(cached_gd)
                all_bars = cached_bars
                for dt_c, pool_c in cs_pool_cached.items():
                    for c_item in pool_c:
                        _ticker_ref[c_item['ticker']] = {'type': 'CS', 'name': c_item.get('name', '')}
                loaded_from_cache = True
                print(f"  [Cache Hit] Loaded grouped daily & 1-min bars from {os.path.basename(cache_path)}")
        except Exception as e:
            print(f"  [Cache Warning] Could not load cache ({e}), fetching from Polygon...")

    # Phase 1: scan all days for gap candidates
    print(f"\n  Phase 1: Scanning for gap candidates...")
    all_candidates = {}
    total_cands = 0
    for i, dt in enumerate(dates):
        cands = scan_day(dt)
        all_candidates[dt] = cands
        total_cands += len(cands)
        if (i + 1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_cands} total  |  {time.time()-t_total:.1f}s")

    # Phase 2: fetch 1-min bars (scan day + prior day for AH ref)
    all_pairs = set()
    for dt, cands in all_candidates.items():
        for c in cands[:PRE_FILTER_N]:
            all_pairs.add((c['ticker'], dt))
            if c.get('prior_str'):
                all_pairs.add((c['ticker'], c['prior_str']))

    if not loaded_from_cache:
        print(f"\n  Phase 2: Fetching 1-min bars for {len(all_pairs)} pairs...")
        pairs_list = list(all_pairs)
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
    if not loaded_from_cache:
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
        if loaded_from_cache and dt in cs_pool_cached and len(cs_pool) == 0 and len(cs_pool_cached[dt]) > 0:
            cs_pool_by_day[dt] = copy.deepcopy(cs_pool_cached[dt])
        else:
            cs_pool_by_day[dt] = cs_pool
    print(f"  → Liquidity filter: {n_illiquid_total} excluded (need {MIN_RTH_BARS} bars, {MIN_OPEN_VOL:,} vol)")

    print(f"\n  Pipeline done in {time.time()-t_total:.1f}s")
    return cs_pool_by_day, all_bars, dates, t_total


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD TRADES — v2 aligned (filter first, then top N + Hybrid Options)
# ═══════════════════════════════════════════════════════════════════════════════

LOCK_PRIMARY_AT_SCAN = True  # True = Option A1 (lock Top-N passing filter at 09:31 ET, matching live)

def build_trades_v2(cs_pool_by_day, all_bars, dates, strategy_id, top_n, strat_cfg=None):
    cfg = strat_cfg or STRAT_BY_ID.get(strategy_id, {})
    base_filter_id = cfg.get('base_filter', strategy_id)
    check_fn = FILTER_FNS[base_filter_id]
    allow_bar0_flush = cfg.get('allow_bar0_flush', False)
    continue_on_failed_pb = cfg.get('continue_on_failed_pb', False)
    hybrid_exit_lt15_1130 = cfg.get('hybrid_exit_lt15_1130', False)
    addon_bounce_reshort = cfg.get('addon_bounce_reshort', False)
    addon_1100_winner = cfg.get('addon_1100_winner', False)
    use_rule3_exit = cfg.get('use_rule3_exit', False)

    trades = []

    for dt in dates:
        cs_pool = cs_pool_by_day.get(dt, [])
        if USE_PRIORITY_TIER_RANKING:
            cs_pool = sorted(cs_pool, key=lambda c: (1 if _is_tier1_setup(c) else 0, c.get('gap_pct', 0)), reverse=True)
        picks = []
        filter_rank = 0

        for c in cs_pool:
            tk = c['ticker']
            bd = all_bars.get((tk, dt))
            if not check_fn(c, bd):
                continue
            filter_rank += 1
            if LOCK_PRIMARY_AT_SCAN and filter_rank > top_n:
                break

            rth = bd.get('rth', []) if bd else []
            ref_p = c.get('ah_ref_price', c.get('prior_close'))
            entry = (
                find_local_high(
                    rth,
                    open_price=c.get('open_price'),
                    gap_pct=c.get('gap_pct'),
                    ref_price=ref_p,
                    allow_bar0_flush=allow_bar0_flush,
                    continue_on_failed_pb=continue_on_failed_pb,
                )
                if len(rth) >= PEAK_WINDOW * 2 + 1
                else None
            )
            if entry is None:
                continue

            ep = entry['entry_price']
            gap = c['gap_pct']
            target_exit = get_target_exit_et(
                c.get('open_price'), gap, entry['entry_type'],
                hybrid_exit_lt15_1130=hybrid_exit_lt15_1130,
                use_rule3_exit=use_rule3_exit,
            )
            stop_price = entry.get('stop_price') if use_rule3_exit else None
            stopped_out = False
            actual_exit_time = target_exit
            exit_price = None
            if stop_price is not None:
                for t_bar, b_bar in rth:
                    if t_bar <= entry['entry_time']:
                        continue
                    if t_bar > target_exit:
                        break
                    if b_bar['h'] >= stop_price:
                        exit_price = round(max(stop_price, b_bar['o']), 2)
                        actual_exit_time = t_bar
                        stopped_out = True
                        break
            if not stopped_out:
                exit_price = get_exit_price_at(bd, rth, entry['entry_time'], target_exit, dt, tk)

            if exit_price is None or exit_price <= 0:
                continue

            pnl = round((ep - exit_price) / ep * 100, 3)
            picks.append({
                'ticker': tk,
                'scan_date': dt,
                'gap_pct': gap,
                'open_price': c['open_price'],
                'prior_close': c['prior_close'],
                'entry_price': ep,
                'entry_time': entry['entry_time'],
                'entry_bar_idx': entry.get('entry_bar_idx', entry['peak_bar'] + PEAK_WINDOW),
                'entry_type': entry['entry_type'],
                'peak_bar': entry['peak_bar'],
                'stop_price': stop_price,
                'stopped_out': stopped_out,
                'exit_price': exit_price,
                'exit_time': actual_exit_time,
                'pnl_eod': pnl,
                'name': c.get('name', ''),
                '_bd': bd,
                '_rth': rth,
            })
            if len(picks) >= top_n:
                break

        # U7b: Optional Midday Bounce Re-Short into idle slots
        if addon_bounce_reshort and len(picks) < top_n:
            for tr in list(picks):
                if len(picks) >= top_n:
                    break
                if tr.get('stopped_out'):
                    continue
                if len(picks) >= top_n:
                    break
                rth = tr.get('_rth', [])
                bd = tr.get('_bd')
                ent2 = find_secondary_bounce_reshort(rth, tr)
                if ent2 is not None:
                    base_tk = tr['ticker']
                    exit_p2 = get_exit_price_at(bd, rth, ent2['entry_time'], EXIT_TIME, dt, base_tk)
                    if exit_p2 and exit_p2 > 0:
                        ep2 = ent2['entry_price']
                        pnl2 = round((ep2 - exit_p2) / ep2 * 100, 3)
                        picks.append({
                            'ticker': f"{base_tk}_B2",
                            'scan_date': dt,
                            'gap_pct': tr['gap_pct'],
                            'open_price': tr['open_price'],
                            'prior_close': tr['prior_close'],
                            'entry_price': ep2,
                            'entry_time': ent2['entry_time'],
                            'entry_bar_idx': ent2['entry_bar_idx'],
                            'entry_type': 'bounce_reshort',
                            'peak_bar': ent2['peak_bar'],
                            'exit_price': exit_p2,
                            'exit_time': EXIT_TIME,
                            'pnl_eod': pnl2,
                            'name': tr.get('name', ''),
                        })

        # U7a: Optional 11:00 ET Winner Pyramid Add-On into idle slots
        if addon_1100_winner and len(picks) < top_n:
            for tr in list(picks):
                if len(picks) >= top_n:
                    break
                if tr.get('entry_type') in ('bounce_reshort', 'winner_1100'):
                    continue
                rth = tr.get('_rth', [])
                bd = tr.get('_bd')
                ep1 = tr['entry_price']
                bars_to_1100 = [(t, b) for t, b in rth if tr['entry_time'] < t <= '11:00']
                if len(bars_to_1100) >= 5:
                    p_1100 = bars_to_1100[-1][1]['c']
                    low_to_1100 = min(b['l'] for _, b in bars_to_1100)
                    max_drop = ep1 - low_to_1100
                    cur_pnl_1100 = (ep1 - p_1100) / ep1 * 100.0 if ep1 > 0 else 0.0
                    bounce_frac = (p_1100 - low_to_1100) / max_drop if max_drop > 0 else 1.0
                    if cur_pnl_1100 >= 3.0 and bounce_frac <= 0.40:
                        base_tk = tr['ticker']
                        exit_p2 = get_exit_price_at(bd, rth, '11:00', EXIT_TIME, dt, base_tk)
                        if exit_p2 and exit_p2 > 0:
                            pnl2 = round((p_1100 - exit_p2) / p_1100 * 100, 3)
                            picks.append({
                                'ticker': f"{base_tk}_W11",
                                'scan_date': dt,
                                'gap_pct': tr['gap_pct'],
                                'open_price': tr['open_price'],
                                'prior_close': tr['prior_close'],
                                'entry_price': p_1100,
                                'entry_time': '11:00',
                                'entry_bar_idx': 90,
                                'entry_type': 'winner_1100',
                                'peak_bar': tr['peak_bar'],
                                'exit_price': exit_p2,
                                'exit_time': EXIT_TIME,
                                'pnl_eod': pnl2,
                                'name': tr.get('name', ''),
                            })

        for p in picks:
            p.pop('_bd', None)
            p.pop('_rth', None)
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
    strat_trades[sid] = build_trades_v2(cs_pool_by_day, all_bars, dates, sid, n, strat_cfg=strat)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOUNDING PORTFOLIO SIMULATION
#
# Each strategy runs as its own compounding portfolio:
#   - Start with CAPITAL
#   - sizing_mode == 'fixed': pos_size = capital / max(n_trades, fixed_slots)
#   - sizing_mode == 'active_cap50': pos_size = capital * min(1.0 / n_trades, 0.50)
#   - PnL compounds: tomorrow's capital = today's ending capital
# ═══════════════════════════════════════════════════════════════════════════════

def run_compounding_portfolio(trades, dates, start_capital, fixed_slots=None, sizing_mode='fixed'):
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
        if sizing_mode == 'active_cap50':
            weight = min(1.0 / n_t, 0.50)
            pos_size = capital * weight
        else:
            n_slots = max(n_t, fixed_slots) if fixed_slots else n_t
            pos_size = capital / n_slots

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


def build_combined_trades(strat_trades_dict, dates, strat_list=None):
    target_strats = strat_list if strat_list is not None else [s for s in STRATEGIES if s['id'] in ('S6', 'S1', 'S12')]
    combined = []
    for dt in dates:
        seen_tickers = {}
        for strat in target_strats:
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
# MULTI-HORIZON REVENUE & PERFORMANCE BENCHMARK (2W OOS | 3M | POST-AUG | 86D)
# ═══════════════════════════════════════════════════════════════════════════════

HORIZONS = [
    ('2W OOS',   '2026-10-15'),
    ('Last 3M',  '2026-07-28'),
    ('Post-Aug', POST_AUG_DATE),
    ('Full 86d', BACKTEST_START),
]

print(f"\n\n{'='*150}")
print(f"  MULTI-HORIZON REVENUE & TOTAL PnL BENCHMARK — Baselines vs Hybrid Combinations (${CAPITAL:,.0f} Start per Horizon)")
print(f"{'='*150}\n")

horizon_logs = {h_name: {} for h_name, _ in HORIZONS}
horizon_finals = {h_name: {} for h_name, _ in HORIZONS}
horizon_cum_pnl = {h_name: {} for h_name, _ in HORIZONS}
multi_rows = []

for strat in STRATEGIES:
    sid = strat['id']
    all_t = strat_trades[sid]
    s_mode = strat.get('sizing_mode', 'fixed')
    opt_n = strat['optimal_n']

    row = [strat['name']]
    for h_name, h_start in HORIZONS:
        h_dates = [d for d in dates if d >= h_start]
        h_trades = [t for t in all_t if t['scan_date'] >= h_start]
        h_log, _, h_final = run_compounding_portfolio(
            h_trades, h_dates, CAPITAL, fixed_slots=opt_n, sizing_mode=s_mode
        )
        horizon_logs[h_name][sid] = h_log
        horizon_finals[h_name][sid] = h_final

        # Build daily cumulative uncompounded PnL% curve for plotting
        cum_s = 0.0
        cum_series = []
        by_dt = defaultdict(list)
        for t in h_trades:
            by_dt[t['scan_date']].append(t['pnl_eod'])
        for d in h_dates:
            cum_s += sum(by_dt.get(d, []))
            cum_series.append(cum_s)
        horizon_cum_pnl[h_name][sid] = (h_dates, cum_series)

        pnls = [t['pnl_eod'] for t in h_trades]
        m = compute_metrics(pnls)
        tot_sum = sum(pnls)
        if h_name == '2W OOS':
            row.extend([len(h_trades), f"{m['wr']*100:.0f}%", f"{m['avg']:+.2f}%", f"{tot_sum:+.1f}%", f"${h_final:,.0f}"])
        else:
            row.extend([len(h_trades), f"{m['wr']*100:.0f}%", f"{m['avg']:+.2f}%", f"{tot_sum:+.1f}%", f"{m['sharpe']:.2f}", f"${h_final:,.0f}"])
    multi_rows.append(row)

print(tabulate(
    multi_rows,
    headers=[
        'Strategy',
        '2W_N', '2W_WR', '2W_Avg', '2W_Sum%', '2W_Final$',
        '3M_N', '3M_WR', '3M_Avg', '3M_Sum%', '3M_Shp', '3M_Final$',
        'PA_N', 'PA_WR', 'PA_Avg', 'PA_Sum%', 'PA_Shp', 'PA_Final$',
        '86d_N', '86d_WR', '86d_Avg', '86d_Sum%', '86d_Shp', '86d_Final$',
    ],
    tablefmt='simple',
    stralign='right',
))


# ═══════════════════════════════════════════════════════════════════════════════
# PER-STRATEGY COMPOUNDING RESULTS (Full 86-Day Window)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print(f"  COMPOUNDING PORTFOLIO — ${CAPITAL:,.0f} start  |  {BACKTEST_START} → {BACKTEST_END}  |  Rule #3 RTH Exits (08:30 / 09:30 / 11:00 PST)")
print(f"{'='*120}\n")

strat_logs = {}
strat_daily_rets = {}
strat_final = {}
summary_rows = []

for strat in STRATEGIES:
    sid = strat['id']
    trades = strat_trades[sid]
    s_mode = strat.get('sizing_mode', 'fixed')
    log, daily_rets, final_cap = run_compounding_portfolio(
        trades, dates, CAPITAL, fixed_slots=strat['optimal_n'], sizing_mode=s_mode
    )
    strat_logs[sid] = log
    strat_daily_rets[sid] = daily_rets
    strat_final[sid] = final_cap

    n_trades = len(trades)
    pnls = [t['pnl_eod'] for t in trades]
    m = compute_metrics(pnls)
    tot_sum = sum(pnls)
    total_return = (final_cap - CAPITAL) / CAPITAL * 100
    profit = final_cap - CAPITAL

    post_trades = [t for t in trades if t['scan_date'] >= POST_AUG_DATE]
    post_m = compute_metrics([t['pnl_eod'] for t in post_trades]) if post_trades else compute_metrics([])

    summary_rows.append([
        strat['name'], strat['optimal_n'], s_mode, n_trades,
        f"{m['avg']:+.2f}%", f"{tot_sum:+.1f}%", f"{m['wr']*100:.0f}%", f"{m['sharpe']:.3f}",
        f"${profit:+,.0f}", f"${final_cap:,.0f}", f"{total_return:+.1f}%",
        len(post_trades), f"{post_m['avg']:+.2f}%" if post_trades else '-',
        f"{post_m['sharpe']:.3f}" if post_trades else '-',
    ])

print(tabulate(summary_rows,
               headers=['Strategy', 'N', 'Sizing', 'Trades', 'Avg%', 'Sum%', 'WR', 'Sharpe',
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


# ── Combined Baseline portfolio (S6 + S1 + S12 deduplicated) ──────────────────

print(f"\n\n{'='*100}")
print(f"  COMBINED BASELINE PORTFOLIO (S6 + S1 + S12 deduplicated)")
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


# ── Strategy attribution in combined baseline portfolio ───────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════════
# MONTHLY BREAKDOWN — per-strategy (Last 4 Months: 2026-07 to 2026-10)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*110}")
print(f"  MONTHLY BREAKDOWN — Baselines vs Hybrid Combinations")
print(f"{'='*110}")

for strat in STRATEGIES:
    sid = strat['id']
    trades = strat_trades[sid]
    if not trades:
        continue
    log = strat_logs[sid]

    print(f"\n  ── {strat['name']} (N={strat['optimal_n']}, {strat.get('sizing_mode', 'fixed')}) — Monthly ──")
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
            smo_rows.append([mo, 0, '-', '-', '-', '$0', f"${sd['cap_end']:,.0f}"])
            continue
        arr = np.array([t['pnl_eod'] for t in st])
        wr = float((arr > 0).mean())
        smo_rows.append([
            mo, n_t, f"{np.mean(arr):+.2f}%", f"{np.sum(arr):+.1f}%", f"{wr*100:.0f}%",
            f"${sd['pnl']:+,.0f}", f"${sd['cap_end']:,.0f}",
        ])
    print(tabulate(smo_rows,
                   headers=['Month', 'N', 'Avg%', 'Sum%', 'WR', 'Month$', 'Capital'],
                   tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS — 6-Panel Revenue & Cumulative PnL Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

STYLE_MAP = {
    'S6':         {'color': '#7f8c8d', 'ls': '--', 'lw': 1.8, 'label': 'S6 Base (Fix 33%)'},
    'S1':         {'color': '#95a5a6', 'ls': ':',  'lw': 1.8, 'label': 'S1 Base (Fix 33%)'},
    'S12':        {'color': '#34495e', 'ls': '-.', 'lw': 1.8, 'label': 'S12 Base (Fix 50%)'},
    'S6+C7':      {'color': '#2980b9', 'ls': '-',  'lw': 2.5, 'label': 'Combo A1: S6+C7'},
    'S1+C7+1130': {'color': '#16a085', 'ls': '-',  'lw': 2.2, 'label': 'Combo A4: S1+C7+08:30'},
    'S12+C7':     {'color': '#8e44ad', 'ls': '-',  'lw': 2.0, 'label': 'Combo A3: S12+C7'},
    'S6+C8':      {'color': '#e67e22', 'ls': '-',  'lw': 2.5, 'label': 'Combo B1: S6+C8 (08:00 Pyr)'},
    'S6+C6':      {'color': '#27ae60', 'ls': '-',  'lw': 2.8, 'label': 'Combo C1: S6+C6 (Bounce)'},
    'S1+C6+1130': {'color': '#2ecc71', 'ls': '--', 'lw': 2.5, 'label': 'Combo C2: S1+C6+08:30'},
    'S6+Full':    {'color': '#c0392b', 'ls': '-',  'lw': 2.8, 'label': 'Combo C3: S6+Full (C7+C6+C8)'},
}

fig, axes = plt.subplots(3, 2, figsize=(22, 18))
fig.suptitle(
    f'Agent 1 v2 — Revenue & Cumulative PnL Backtest: Baselines vs Rule #3 RTH Hybrid Combos (${CAPITAL/1000:.0f}K Start)\n'
    f'Last 3 Months (07/28 → 10/30)  |  Last 2 Weeks OOS (10/15 → 10/30)  |  Full 86-Day (07/01 → 10/30)',
    fontsize=13, fontweight='bold'
)

# ── Panel 1 (0, 0): Last 3 Months (07/28 -> 10/30) Compounding Revenue ($) ────
ax = axes[0, 0]
dates_3m = [datetime.strptime(e['date'], '%Y-%m-%d') for e in horizon_logs['Last 3M']['S6']]
for strat in STRATEGIES:
    sid = strat['id']
    st = STYLE_MAP[sid]
    caps = [e['capital_end'] for e in horizon_logs['Last 3M'][sid]]
    fin = horizon_finals['Last 3M'][sid]
    ax.plot(dates_3m, caps, color=st['color'], ls=st['ls'], lw=st['lw'],
            label=f"{st['label']}: ${fin/1000:,.1f}K ({(fin-CAPITAL)/CAPITAL*100:+.0f}%)")
ax.axhline(CAPITAL, color='#333', lw=0.8, ls=':')
ax.set_title('1. Last 3 Months (07/28 → 10/30) — Compounding Revenue ($100K Start)', fontweight='bold')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x/1000:,.0f}K'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.legend(fontsize=7.5, loc='upper left')
ax.grid(True, alpha=0.3)

# ── Panel 2 (0, 1): Last 3 Months (07/28 -> 10/30) Cumulative Total PnL (Sum %) ─
ax = axes[0, 1]
for strat in STRATEGIES:
    sid = strat['id']
    st = STYLE_MAP[sid]
    _, cum_s = horizon_cum_pnl['Last 3M'][sid]
    tot_s = cum_s[-1] if cum_s else 0.0
    n_t = len([t for t in strat_trades[sid] if t['scan_date'] >= '2026-07-28'])
    ax.plot(dates_3m, cum_s, color=st['color'], ls=st['ls'], lw=st['lw'],
            label=f"{st['label']}: {tot_s:+.1f}% ({n_t}t)")
ax.axhline(0, color='#333', lw=0.8, ls=':')
ax.set_title('2. Last 3 Months (07/28 → 10/30) — Cumulative Uncompounded Total PnL (Sum %)', fontweight='bold')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:+.0f}%'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.legend(fontsize=7.5, loc='upper left')
ax.grid(True, alpha=0.3)

# ── Panel 3 (1, 0): Last 2 Weeks OOS (10/15 -> 10/30) Compounding Revenue ($) ──
ax = axes[1, 0]
dates_2w = [datetime.strptime(e['date'], '%Y-%m-%d') for e in horizon_logs['2W OOS']['S6']]
for strat in STRATEGIES:
    sid = strat['id']
    st = STYLE_MAP[sid]
    caps = [e['capital_end'] for e in horizon_logs['2W OOS'][sid]]
    fin = horizon_finals['2W OOS'][sid]
    ax.plot(dates_2w, caps, color=st['color'], ls=st['ls'], lw=st['lw'], marker='o', markersize=3,
            label=f"{st['label']}: ${fin/1000:,.1f}K ({(fin-CAPITAL)/CAPITAL*100:+.1f}%)")
ax.axhline(CAPITAL, color='#333', lw=0.8, ls=':')
ax.set_title('3. Last 2 Weeks OOS (10/15 → 10/30) — Compounding Revenue ($100K Start)', fontweight='bold')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x/1000:,.1f}K'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.legend(fontsize=7.5, loc='upper left')
ax.grid(True, alpha=0.3)

# ── Panel 4 (1, 1): Last 2 Weeks OOS (10/15 -> 10/30) Cumulative Total PnL (Sum %)
ax = axes[1, 1]
for strat in STRATEGIES:
    sid = strat['id']
    st = STYLE_MAP[sid]
    _, cum_s = horizon_cum_pnl['2W OOS'][sid]
    tot_s = cum_s[-1] if cum_s else 0.0
    n_t = len([t for t in strat_trades[sid] if t['scan_date'] >= '2026-10-15'])
    ax.plot(dates_2w, cum_s, color=st['color'], ls=st['ls'], lw=st['lw'], marker='o', markersize=3,
            label=f"{st['label']}: {tot_s:+.1f}% ({n_t}t)")
ax.axhline(0, color='#333', lw=0.8, ls=':')
ax.set_title('4. Last 2 Weeks OOS (10/15 → 10/30) — Cumulative Uncompounded Total PnL (Sum %)', fontweight='bold')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:+.0f}%'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.legend(fontsize=7.5, loc='upper left')
ax.grid(True, alpha=0.3)

# ── Panel 5 (2, 0): Full 86-Day (07/01 -> 10/30) Cumulative Total PnL (Sum %) ──
ax = axes[2, 0]
d_dates_plot = [datetime.strptime(dt, '%Y-%m-%d') for dt in dates]
aug15 = datetime.strptime('2026-08-15', '%Y-%m-%d')
for strat in STRATEGIES:
    sid = strat['id']
    st = STYLE_MAP[sid]
    _, cum_s = horizon_cum_pnl['Full 86d'][sid]
    tot_s = cum_s[-1] if cum_s else 0.0
    fin86 = strat_final[sid]
    ax.plot(d_dates_plot, cum_s, color=st['color'], ls=st['ls'], lw=st['lw'],
            label=f"{st['label']}: {tot_s:+.0f}% (${fin86/1000:,.0f}K)")
ax.axvline(aug15, color='red', ls='--', lw=1, alpha=0.5, label='Aug 15 Regime Shift')
ax.axhline(0, color='#333', lw=0.8, ls=':')
ax.set_title('5. Full 86-Day (07/01 → 10/30) — Cumulative Uncompounded Total PnL (Sum %)', fontweight='bold')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:+.0f}%'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.legend(fontsize=7.5, loc='upper left')
ax.grid(True, alpha=0.3)

# ── Panel 6 (2, 1): 3-Month Net Revenue ($) & Total PnL (%) Bar Comparison ────
ax = axes[2, 1]
short_labels = [STYLE_MAP[s['id']]['label'].replace('Combo ', '') for s in STRATEGIES]
profits_3m = [(horizon_finals['Last 3M'][s['id']] - CAPITAL) for s in STRATEGIES]
sums_3m = [horizon_cum_pnl['Last 3M'][s['id']][1][-1] for s in STRATEGIES]
bar_colors = [STYLE_MAP[s['id']]['color'] for s in STRATEGIES]

bars = ax.bar(short_labels, profits_3m, color=bar_colors, edgecolor='white', linewidth=0.8)
for bar, prof, sm in zip(bars, profits_3m, sums_3m):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(profits_3m) * 0.015,
        f"+${prof/1000:,.0f}K\n({sm:+.0f}%)",
        ha='center', va='bottom', fontsize=8, fontweight='bold'
    )
ax.set_ylim(0, max(profits_3m) * 1.18)
ax.axhline(0, color='#333', lw=0.8)
ax.set_title('6. Last 3 Months (07/28 → 10/30) — Net Profit (USD on $100K Start) & Total Sum PnL%', fontweight='bold')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x/1000:,.0f}K'))
ax.set_xticks(range(len(short_labels)))
ax.set_xticklabels(short_labels, rotation=22, ha='right', fontsize=8.5)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.95])
p1 = os.path.join(BASE_DIR, 'agent1_v2_backtest.png')
plt.savefig(p1, dpi=160, bbox_inches='tight')
print(f"\n  Saved: {p1}")

print(f"\n  Total runtime: {time.time() - t_total:.1f}s ({(time.time() - t_total)/60:.2f} min)")

