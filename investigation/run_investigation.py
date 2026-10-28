# %% Project X — Semi-Autonomous Strategy Investigation
#
# Universal backtest engine + hypothesis loop.
# Adapts proven Polygon API infrastructure from agent1/agent3.
#
# Period: Aug 1 – Oct 6 2026 (eval), Sep 15 2025 – Oct 6 2026 (full validation)
# Beat:   avg > 2%, Sharpe > 0.20, n >= 15, CI > 0, WR >= 50%
# Pace:   max 15 trades/week, min 1h avg hold, min $5K position
#
# Colab-cell friendly. Config vars at top, no if __name__.

import os, sys, time, re, copy, json, asyncio, hashlib
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


# =============================================================================
# CONFIG
# =============================================================================

EVAL_START     = '2026-08-01'
EVAL_END       = '2026-10-06'
FULL_START     = '2025-09-15'

CAPITAL        = 100_000
TOP_N_DEFAULT  = 5
FLOAT_MIN      = 5_000_000
PRICE_MIN      = 5.0
AVG_VOL_MIN    = 50_000

MAX_HYPOTHESES = 50
MAX_WINNERS    = 3

BOOTSTRAP_N    = 1000
np.random.seed(42)

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')
ET = pytz.timezone('America/New_York')

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
WINNERS_DIR = os.path.join(BASE_DIR, 'winners')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WINNERS_DIR, exist_ok=True)


# =============================================================================
# PASS CRITERIA (from hypothesis_space.py)
# =============================================================================

PASS_CRITERIA = {
    "avg_pnl_gt":          2.0,
    "sharpe_gt":           0.20,
    "n_gte":               15,
    "ci_lower_gt":         0.0,
    "win_rate_gte":        0.50,
    "max_dd_lt":           0.35,
    "trades_per_week_lte": 15,
    "avg_hold_hrs_gte":    1.0,
}

BASELINE = {
    "strategy": "E2-A SHORT",
    "period":   "Aug-Oct 2026",
    "avg_pnl":  2.28,
    "sharpe":   0.185,
    "n":        154,
}


# =============================================================================
# CONSTANTS — excludes, holidays, float dict
# =============================================================================

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

SCANNER_FLOOR = 0.02

SECTOR_GROUPS = {
    'pharma_health_biotech': ['pharmaceutical', 'health', 'biotech', 'medical', 'drug',
                              'surgical', 'dental', 'diagnostic', 'biological'],
    'transport_retail': ['transport', 'retail', 'eating', 'food', 'grocery',
                         'apparel', 'restaurant', 'trucking', 'airline'],
    'utilities_energy': ['utility', 'electric', 'gas', 'oil', 'energy', 'petroleum',
                         'coal', 'solar', 'wind', 'nuclear', 'power'],
}


# =============================================================================
# HELPERS — calendar, API, caching
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
    for _ in range(60):
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


# =============================================================================
# DATA CACHE — grouped daily, minute bars, ticker references
# =============================================================================

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


async def _fetch_bars_one(session, tk, dt, sem):
    url = f"{_BASE}/v2/aggs/ticker/{tk}/range/1/minute/{dt}/{dt}"
    params = {"adjusted":"true","sort":"asc","limit":"1000","apiKey":POLYGON_API_KEY}
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
            pm, rth, ah = [], [], []
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
                'sic_code': res.get('sic_code', ''),
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


# =============================================================================
# PATTERN CLASSIFICATION (gap-up: SHORT; gap-down: LONG)
# =============================================================================

def classify_gap_up_pattern(rth_bars):
    if len(rth_bars) < 60:
        return 'no_data'
    open_price = rth_bars[0][1]['o']
    if open_price <= 0:
        return 'no_data'
    low_30m = min(b[1]['l'] for b in rth_bars[:30])
    close_60m = rth_bars[59][1]['c']
    drop_30m = (low_30m - open_price) / open_price
    recovery_denom = abs(open_price - low_30m)
    recovery = (close_60m - low_30m) / recovery_denom if recovery_denom > 0 else 0

    if drop_30m < -0.03 and recovery <= 0.5:
        return 'sustained_drop'
    elif drop_30m < -0.03 and recovery > 0.5:
        return 'drop_then_recover'
    elif drop_30m >= -0.01:
        return 'flat_or_bounce'
    else:
        return 'mild_drop'

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
# LOCAL HIGH / LOCAL LOW DETECTION (entry mechanics)
# =============================================================================

def find_local_high(bars, window=3, max_peak_bar=12, fallback_bar=15):
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

def find_local_low(bars, window=3, max_bar=12, fallback_bar=None):
    if len(bars) < window * 2 + 1:
        return None
    for i in range(window, min(max_bar, len(bars) - window)):
        cl = bars[i][1]['l']
        if all(bars[i - j][1]['l'] > cl and bars[i + j][1]['l'] > cl
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(bars):
                return {
                    'entry_price': bars[ci][1]['c'],
                    'entry_time': bars[ci][0],
                    'entry_type': 'local_low',
                    'trough_bar': i,
                }
    if fallback_bar is not None and fallback_bar < len(bars):
        return {
            'entry_price': bars[fallback_bar][1]['c'],
            'entry_time': bars[fallback_bar][0],
            'entry_type': 'fallback',
            'trough_bar': None,
        }
    return None


# =============================================================================
# PRICE AT TIME helper
# =============================================================================

def get_bar_close_at(rth_bars, bar_index):
    if rth_bars and bar_index < len(rth_bars):
        return rth_bars[bar_index][1]['c']
    return None

def get_price_at_time(bar_data, time_str):
    if bar_data is None:
        return None
    all_bars = (bar_data.get('pm') or []) + (bar_data.get('rth') or []) + (bar_data.get('ah') or [])
    best = None
    for t_str, bd in all_bars:
        if t_str <= time_str:
            best = bd['c']
        else:
            break
    return best

def time_str_to_bar_index(time_str):
    parts = time_str.split(':')
    h, m = int(parts[0]), int(parts[1])
    return (h - 9) * 60 + m - 30


# =============================================================================
# METRICS
# =============================================================================

def bootstrap_ci(pnls, n_iter=BOOTSTRAP_N, alpha=0.05):
    if len(pnls) < 2:
        m = float(np.mean(pnls)) if pnls else 0.0
        return m, m
    a = np.array(pnls)
    means = np.array([float(np.mean(np.random.choice(a, size=len(a), replace=True)))
                      for _ in range(n_iter)])
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))

def compute_metrics(pnl_list, label=''):
    if not pnl_list:
        return {'label':label,'n':0,'avg':0,'med':0,'std':0,'wr':0,
                'sharpe':0,'pf':0,'ci_lo':0,'ci_hi':0,'trimmed_mean':0,
                'max_loss':0,'max_gain':0,'max_dd_pct':0}
    a = np.array(pnl_list)
    n = len(a); avg = float(np.mean(a)); med = float(np.median(a))
    std = float(np.std(a, ddof=1)) if n > 1 else 0
    wr = float((a > 0).mean())
    sharpe = avg / std if std > 0 else 0
    w = a[a > 0]; l = a[a < 0]
    pf = float(w.sum() / abs(l.sum())) if len(l) > 0 and l.sum() != 0 else float('inf')
    p5, p95 = np.percentile(a, 5), np.percentile(a, 95)
    trimmed = a[(a >= p5) & (a <= p95)]
    tm = float(np.mean(trimmed)) if len(trimmed) > 0 else avg

    if n >= 2:
        ci_lo, ci_hi = bootstrap_ci(pnl_list)
    else:
        se = std / np.sqrt(n) if n > 0 else 0
        ci_lo = avg - 1.96 * se
        ci_hi = avg + 1.96 * se

    # Non-compounding daily DD: each trade risked CAPITAL/n_per_day
    daily_pnl = defaultdict(list)
    for i, p in enumerate(pnl_list):
        daily_pnl[i // 5].append(p)  # approximate: 5 trades per day bucket
    eq = [CAPITAL]
    for day_idx in sorted(daily_pnl.keys()):
        dp = daily_pnl[day_idx]
        n_t = len(dp)
        pos_size = CAPITAL / max(n_t, 1)
        day_dollar = sum(pos_size * (r / 100) for r in dp)
        eq.append(eq[-1] + day_dollar)
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = np.where(peak > 0, (eq_arr - peak) / peak, 0)
    max_dd = float(dd.min())

    return {
        'label': label, 'n': n,
        'avg': round(avg, 3), 'med': round(med, 3),
        'std': round(std, 3), 'wr': round(wr, 4),
        'sharpe': round(sharpe, 4), 'pf': round(pf, 3),
        'ci_lo': round(ci_lo, 3), 'ci_hi': round(ci_hi, 3),
        'trimmed_mean': round(tm, 3),
        'max_loss': round(float(a.min()), 3),
        'max_gain': round(float(a.max()), 3),
        'max_dd_pct': round(max_dd * 100, 2),
    }


# =============================================================================
# =============================================================================
#
#  DATA PIPELINE — shared across all hypotheses
#
# =============================================================================
# =============================================================================

_pipeline_cache = {}

def run_data_pipeline(start_date, end_date):
    cache_key = f"{start_date}_{end_date}"
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]

    t_total = time.time()
    print(f"\n{'='*80}")
    print(f"  PROJECT X — DATA PIPELINE")
    print(f"  {start_date} -> {end_date}")
    print(f"{'='*80}")

    d = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"\n  {n_dates} trading days")

    lookback_start = _nth_prior_td(start_date, 12)
    lb = lookback_start
    lookback_dates = []
    while lb < datetime.strptime(start_date, '%Y-%m-%d').date():
        if _is_td(lb): lookback_dates.append(lb.strftime('%Y-%m-%d'))
        lb += timedelta(days=1)

    t1 = time.time()
    all_fetch_dates = sorted(set(lookback_dates + dates))
    print(f"\n  Phase 1: Fetching grouped daily for {len(all_fetch_dates)} dates...")
    for i, dt in enumerate(all_fetch_dates):
        fetch_grouped_daily(dt)
        if (i+1) % 30 == 0 or i == len(all_fetch_dates) - 1:
            print(f"    [{i+1:>3}/{len(all_fetch_dates)}] {dt}  |  {time.time()-t_total:.0f}s")

    outcome_dates = set()
    for dt in dates:
        for nd in _next_n_td(dt, 12):
            outcome_dates.add(nd.strftime('%Y-%m-%d'))
    print(f"  Fetching {len(outcome_dates)} outcome dates...")
    for od in sorted(outcome_dates):
        if od not in _gd_cache:
            try: fetch_grouped_daily(od)
            except: pass
    print(f"  Phase 1: {time.time()-t1:.0f}s")

    # Phase 2: Scan gap-up AND gap-down candidates (broad net)
    t2 = time.time()
    print(f"\n  Phase 2: Scanning gap-up + gap-down candidates...")
    gap_up_candidates = {}
    gap_down_candidates = {}
    total_up, total_down = 0, 0

    for i, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = _gd_cache.get(prior_str, {})
        today = _gd_cache.get(dt, {})

        up_day, down_day = [], []
        for tk in today:
            if tk in _EXCLUDE_TK: continue
            if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
            if tk not in prior: continue
            prev_c = prior[tk]['c']
            if prev_c <= 0: continue
            cur_o = today[tk]['o']
            if cur_o <= 0 or cur_o < PRICE_MIN: continue
            # Hard float filter: must be in _FLOAT with >= FLOAT_MIN
            fs = _FLOAT.get(tk)
            if fs is None or fs < FLOAT_MIN: continue

            gap = (cur_o - prev_c) / prev_c
            # Cap extreme gaps (penny stock splits/reverse splits)
            if abs(gap) > 0.75: continue

            base = {
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_o, 'scan_date': dt,
                'volume_today': today[tk]['v'],
                'day_high': today[tk]['h'],
                'day_low': today[tk]['l'],
                'day_close': today[tk]['c'],
            }

            if gap >= 0.02:
                up_day.append(base)
            if gap <= -0.02:
                down_day.append(copy.copy(base))

        up_day.sort(key=lambda x: -x['gap_pct'])
        down_day.sort(key=lambda x: x['gap_pct'])
        gap_up_candidates[dt] = up_day
        gap_down_candidates[dt] = down_day
        total_up += len(up_day)
        total_down += len(down_day)

        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  up={total_up}, down={total_down}  |  {time.time()-t_total:.0f}s")
    print(f"  Phase 2: {time.time()-t2:.0f}s")

    # Phase 3: Fetch minute bars for top candidates
    t3 = time.time()
    bar_pairs = set()
    for dt in dates:
        for c in gap_up_candidates.get(dt, [])[:20]:
            bar_pairs.add((c['ticker'], dt))
        for c in gap_down_candidates.get(dt, [])[:20]:
            bar_pairs.add((c['ticker'], dt))

    print(f"\n  Phase 3: Fetching 1-min bars for {len(bar_pairs)} pairs...")
    pairs_list = list(bar_pairs)
    all_bars = {}
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start + 100]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        done = min(c_start + 100, len(pairs_list))
        if done % 500 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    print(f"  -> Got bars for {len(all_bars)}/{len(bar_pairs)} pairs")
    print(f"  Phase 3: {time.time()-t3:.0f}s")

    # Phase 4: Ticker references
    t4 = time.time()
    all_tickers = set()
    for cands in gap_up_candidates.values():
        for c in cands[:20]:
            all_tickers.add(c['ticker'])
    for cands in gap_down_candidates.values():
        for c in cands[:20]:
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

    # Phase 5: Classify patterns
    t5 = time.time()
    pattern_cache_up = {}
    pattern_cache_down = {}
    for key, bar_data in all_bars.items():
        rth = bar_data.get('rth', [])
        pattern_cache_up[key] = classify_gap_up_pattern(rth)
        pattern_cache_down[key] = classify_gap_down_pattern(rth)
    print(f"  Phase 5: Classified {len(pattern_cache_up)} patterns  |  {time.time()-t5:.0f}s")

    # Prior multi-day returns for C1-style hypotheses
    t6 = time.time()
    print(f"\n  Phase 6: Computing prior multi-day returns...")
    prior_returns = {}
    for dt in dates:
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        for cands in [gap_up_candidates.get(dt, [])[:20], gap_down_candidates.get(dt, [])[:20]]:
            for c in cands:
                tk = c['ticker']
                prev_c = c['prior_close']
                for nd_n, nd_lbl in [(1, '2d'), (2, '3d'), (3, '4d'), (4, '5d')]:
                    try:
                        close_ago_date = _nth_prior_td(prior_str, nd_n).strftime('%Y-%m-%d')
                        close_ago_data = _gd_cache.get(close_ago_date, {}).get(tk)
                        if close_ago_data and close_ago_data['c'] > 0:
                            ret = (prev_c - close_ago_data['c']) / close_ago_data['c']
                            prior_returns[(tk, dt, nd_lbl)] = ret
                    except ValueError:
                        pass
    print(f"  Phase 6: {len(prior_returns)} prior return lookups  |  {time.time()-t6:.0f}s")

    # SPY context for regime gates
    spy_context = {}
    for dt in dates:
        spy_today = _gd_cache.get(dt, {}).get('SPY')
        if spy_today:
            spy_context[dt] = {
                'open': spy_today['o'], 'close': spy_today['c'],
                'high': spy_today['h'], 'low': spy_today['l'],
            }

    # Phase 7: Fetch SPY minute bars separately (SPY excluded from ticker universe)
    t7 = time.time()
    spy_bar_pairs = [('SPY', dt) for dt in dates]
    print(f"\n  Phase 7: Fetching SPY minute bars for {len(spy_bar_pairs)} dates...")
    spy_bars = fetch_minute_bars_batch(spy_bar_pairs)
    print(f"  -> Got SPY bars for {len(spy_bars)}/{len(spy_bar_pairs)} dates")
    print(f"  Phase 7: {time.time()-t7:.0f}s")

    # Build sector map from ticker reference SIC codes
    sector_map = {}
    for tk, ref in _ticker_ref.items():
        sic = (ref.get('sic_code') or '').lower()
        name = (ref.get('name') or '').lower()
        sector_map[tk] = sic + ' ' + name
    print(f"  Sector map: {len(sector_map)} tickers mapped")

    elapsed = time.time() - t_total
    print(f"\n  DATA PIPELINE COMPLETE: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    result = {
        'dates': dates,
        'gap_up_candidates': gap_up_candidates,
        'gap_down_candidates': gap_down_candidates,
        'all_bars': all_bars,
        'pattern_cache_up': pattern_cache_up,
        'pattern_cache_down': pattern_cache_down,
        'prior_returns': prior_returns,
        'spy_context': spy_context,
        'spy_bars': spy_bars,
        'sector_map': sector_map,
    }
    _pipeline_cache[cache_key] = result
    return result


# =============================================================================
# =============================================================================
#
#  UNIVERSAL BACKTEST ENGINE
#
# =============================================================================
# =============================================================================

def run_backtest(hypothesis, data):
    h = hypothesis
    h_id = h['id']
    direction = h['direction']
    catalyst = h.get('catalyst', 'gap_down')
    entry_type = h.get('entry', 'open_9:30')
    exit_type = h.get('exit', 'T+2_close')
    filters = h.get('filters', {})
    top_n = filters.get('top_n', TOP_N_DEFAULT)

    gap_min = filters.get('gap_min', 0.02)
    gap_max = filters.get('gap_max', None)
    pattern_filter = filters.get('pattern', 'none')
    fire_type = filters.get('fire_type', 'any')
    lookback_days = filters.get('lookback_days', None)
    decline_thresh = filters.get('decline_thresh', None)
    rise_thresh = filters.get('rise_thresh', None)

    trades = []
    dates = data['dates']

    for dt in dates:
        # Select candidate pool based on catalyst
        if catalyst in ('gap_up', 'earnings_beat', 'analyst_upgrade'):
            cands = data['gap_up_candidates'].get(dt, [])
            cands = [c for c in cands if abs(c['gap_pct']) >= gap_min]
            if gap_max is not None:
                cands = [c for c in cands if c['gap_pct'] <= gap_max]
        elif catalyst in ('gap_down', 'earnings_miss', 'secondary_offering'):
            cands = data['gap_down_candidates'].get(dt, [])
            cands = [c for c in cands if abs(c['gap_pct']) >= gap_min]
        elif catalyst == 'multi_day_decline_reversal':
            cands = data['gap_up_candidates'].get(dt, [])
            cands = [c for c in cands if c['gap_pct'] >= gap_min]
        elif catalyst == 'multi_day_rise_reversal':
            cands = data['gap_down_candidates'].get(dt, [])
            cands = [c for c in cands if abs(c['gap_pct']) >= gap_min]
        else:
            cands = data['gap_up_candidates'].get(dt, []) + data['gap_down_candidates'].get(dt, [])
            cands = [c for c in cands if abs(c['gap_pct']) >= gap_min]

        qualified = []
        for c in cands[:30]:
            tk = c['ticker']

            # CS filter
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS': continue

            # Price floor
            if c['open_price'] < PRICE_MIN: continue

            # Pattern filter
            key = (tk, dt)
            if pattern_filter != 'none':
                if direction == 'SHORT':
                    pat = data['pattern_cache_up'].get(key, 'no_data')
                else:
                    pat = data['pattern_cache_down'].get(key, 'no_data')
                if pat != pattern_filter: continue

            # Multi-day decline/rise filter (C1-style)
            if lookback_days and decline_thresh:
                lbl_map = {2: '2d', 3: '3d', 4: '4d', 5: '5d'}
                lbl = lbl_map.get(lookback_days, f'{lookback_days}d')
                prior_ret = data['prior_returns'].get((tk, dt, lbl))
                if prior_ret is None: continue
                if prior_ret > -decline_thresh: continue

            if lookback_days and rise_thresh:
                lbl_map = {2: '2d', 3: '3d', 4: '4d', 5: '5d'}
                lbl = lbl_map.get(lookback_days, f'{lookback_days}d')
                prior_ret = data['prior_returns'].get((tk, dt, lbl))
                if prior_ret is None: continue
                if prior_ret < rise_thresh: continue

            # Sector filter
            sector_incl = filters.get('sector_incl', 'none')
            sector_excl = filters.get('sector_excl', 'none')
            if sector_incl != 'none':
                sector_keywords = SECTOR_GROUPS.get(sector_incl, [])
                tk_sector = data.get('sector_map', {}).get(tk, '')
                if not any(kw in tk_sector for kw in sector_keywords):
                    continue
            if sector_excl != 'none':
                sector_keywords = SECTOR_GROUPS.get(sector_excl, [])
                tk_sector = data.get('sector_map', {}).get(tk, '')
                if any(kw in tk_sector for kw in sector_keywords):
                    continue

            # SPY regime gate (uses dedicated spy_bars, not all_bars)
            spy_gate = filters.get('spy_gate')
            if spy_gate:
                spy = data['spy_context'].get(dt)
                if not spy: continue
                if spy_gate == 'down_30m':
                    spy_bar_data = data.get('spy_bars', {}).get(('SPY', dt))
                    if spy_bar_data:
                        rth = spy_bar_data.get('rth', [])
                        if len(rth) >= 30:
                            spy_30m_ret = (rth[29][1]['c'] - rth[0][1]['o']) / rth[0][1]['o']
                            if spy_30m_ret >= 0: continue
                        else:
                            continue
                    else:
                        continue

            # Determine entry price
            bar_data = data['all_bars'].get(key)
            rth_bars = bar_data.get('rth', []) if bar_data else []

            ep = None
            entry_time_str = '09:30'
            entry_method = 'open'

            if entry_type == 'open_9:30':
                ep = c['open_price']
                entry_time_str = '09:30'
                entry_method = 'open'

            elif entry_type == 'local_high':
                if not rth_bars: continue
                lh = find_local_high(rth_bars)
                if lh is None: continue
                if fire_type == 'LH_only' and lh['entry_type'] != 'local_high':
                    continue
                ep = lh['entry_price']
                entry_time_str = lh['entry_time']
                entry_method = lh['entry_type']

            elif entry_type == 'local_low':
                if not rth_bars: continue
                ll = find_local_low(rth_bars)
                if ll is None: continue
                ep = ll['entry_price']
                entry_time_str = ll['entry_time']
                entry_method = ll['entry_type']

            elif ':' in entry_type:
                bar_idx = time_str_to_bar_index(entry_type)
                if bar_idx < 0: bar_idx = 0
                ep = get_bar_close_at(rth_bars, bar_idx)
                if ep is None: continue
                entry_time_str = entry_type
                entry_method = f'bar_{entry_type}'

            elif entry_type == 'T+1_open':
                t1 = _next_n_td(dt, 1)
                if not t1: continue
                t1_str = t1[0].strftime('%Y-%m-%d')
                t1_data = _gd_cache.get(t1_str, {}).get(tk, {})
                if not t1_data.get('o'): continue
                ep = t1_data['o']
                entry_time_str = '09:30'
                entry_method = 'T+1_open'

            if ep is None or ep <= 0: continue

            # Gap freshness filter: gap remaining at entry vs original gap
            gap_fresh_min = filters.get('gap_fresh')
            if gap_fresh_min and gap_fresh_min > 0:
                original_gap_pts = c['open_price'] - c['prior_close']
                if abs(original_gap_pts) > 0:
                    if direction == 'SHORT':
                        remaining = ep - c['prior_close']
                    else:
                        remaining = c['prior_close'] - ep
                    freshness = remaining / abs(original_gap_pts)
                    if freshness < gap_fresh_min: continue

            # Volume gate: entry bar volume vs median
            vol_gate = filters.get('vol_gate')
            if vol_gate == 'top_quartile' and rth_bars:
                entry_bar_idx = None
                for bi, (bt, bd) in enumerate(rth_bars):
                    if bt == entry_time_str:
                        entry_bar_idx = bi
                        break
                if entry_bar_idx is not None:
                    all_vols = [b[1]['v'] for b in rth_bars[:min(30, len(rth_bars))]]
                    if all_vols:
                        vol_75 = np.percentile(all_vols, 75)
                        if rth_bars[entry_bar_idx][1]['v'] < vol_75:
                            continue

            # Determine exit price
            exit_price = None
            exit_date = dt

            if exit_type == 'T+1_open':
                t1 = _next_n_td(dt, 1)
                if not t1: continue
                t1_str = t1[0].strftime('%Y-%m-%d')
                t1_data = _gd_cache.get(t1_str, {}).get(tk, {})
                if not t1_data.get('o'): continue
                exit_price = t1_data['o']
                exit_date = t1_str

            elif exit_type == 'T+1_close':
                t1 = _next_n_td(dt, 1)
                if not t1: continue
                t1_str = t1[0].strftime('%Y-%m-%d')
                t1_data = _gd_cache.get(t1_str, {}).get(tk, {})
                if not t1_data.get('c'): continue
                exit_price = t1_data['c']
                exit_date = t1_str

            elif exit_type.startswith('T+') and exit_type.endswith('_close'):
                n_days = int(exit_type.split('+')[1].split('_')[0])
                tn = _next_n_td(dt, n_days)
                if len(tn) < n_days: continue
                tn_str = tn[n_days - 1].strftime('%Y-%m-%d')
                tn_data = _gd_cache.get(tn_str, {}).get(tk, {})
                if not tn_data.get('c'): continue
                exit_price = tn_data['c']
                exit_date = tn_str

            elif exit_type.startswith('T+') and exit_type.endswith('_open'):
                n_days = int(exit_type.split('+')[1].split('_')[0])
                tn = _next_n_td(dt, n_days)
                if len(tn) < n_days: continue
                tn_str = tn[n_days - 1].strftime('%Y-%m-%d')
                tn_data = _gd_cache.get(tn_str, {}).get(tk, {})
                if not tn_data.get('o'): continue
                exit_price = tn_data['o']
                exit_date = tn_str

            elif exit_type == '17:00':
                if bar_data:
                    exit_price = get_price_at_time(bar_data, '17:00')
                if exit_price is None:
                    exit_price = c.get('day_close')

            elif ':' in exit_type:
                bar_idx = time_str_to_bar_index(exit_type)
                if bar_idx < 0: bar_idx = 0
                exit_price = get_bar_close_at(rth_bars, bar_idx)

            if exit_price is None or exit_price <= 0: continue

            # Compute PnL
            if direction == 'LONG':
                pnl = round((exit_price - ep) / ep * 100, 3)
            else:
                pnl = round((ep - exit_price) / ep * 100, 3)

            # Estimate hold period in hours
            if exit_date == dt:
                eparts = entry_time_str.split(':')
                entry_min = int(eparts[0]) * 60 + int(eparts[1])
                if ':' in exit_type:
                    xparts = exit_type.split(':')
                    exit_min = int(xparts[0]) * 60 + int(xparts[1])
                elif exit_type == '17:00':
                    exit_min = 17 * 60
                else:
                    exit_min = 16 * 60
                hold_hrs = max(0, (exit_min - entry_min) / 60)
            else:
                scan_d = datetime.strptime(dt, '%Y-%m-%d')
                exit_d = datetime.strptime(exit_date, '%Y-%m-%d')
                hold_hrs = (exit_d - scan_d).days * 6.5

            qualified.append({
                'ticker': tk, 'scan_date': dt, 'gap_pct': c['gap_pct'],
                'open_price': c['open_price'],
                'entry_price': ep, 'exit_price': exit_price,
                'entry_type': entry_method, 'entry_time': entry_time_str,
                'exit_date': exit_date, 'pnl': pnl,
                'direction': direction, 'hold_hrs': hold_hrs,
                'strategy': h_id,
            })

        if direction == 'SHORT':
            qualified.sort(key=lambda x: -abs(x['gap_pct']))
        else:
            qualified.sort(key=lambda x: x['gap_pct'])

        for tc in qualified[:top_n]:
            trades.append(tc)

    return trades


# =============================================================================
# EVALUATE — check pass criteria
# =============================================================================

def evaluate_hypothesis(hypothesis, trades, dates):
    h = hypothesis
    h_id = h['id']
    pnls = [t['pnl'] for t in trades if t.get('pnl') is not None]
    m = compute_metrics(pnls, h_id)

    n_weeks = len(dates) / 5.0
    tpw = m['n'] / n_weeks if n_weeks > 0 else 0
    avg_hold = np.mean([t.get('hold_hrs', 0) for t in trades]) if trades else 0

    passed = True
    reasons = []

    if m['n'] < PASS_CRITERIA['n_gte']:
        passed = False
        reasons.append(f"n={m['n']} < {PASS_CRITERIA['n_gte']}")

    if m['n'] < 5:
        return {
            'hypothesis': h, 'metrics': m, 'passed': False,
            'reasons': ['INVALID: n < 5'], 'trades_per_week': round(tpw, 2),
            'avg_hold_hrs': round(avg_hold, 1), 'n_trades': m['n'],
        }

    if m['avg'] <= PASS_CRITERIA['avg_pnl_gt']:
        passed = False
        reasons.append(f"avg={m['avg']:.2f}% <= {PASS_CRITERIA['avg_pnl_gt']}%")

    if m['sharpe'] <= PASS_CRITERIA['sharpe_gt']:
        passed = False
        reasons.append(f"Sharpe={m['sharpe']:.3f} <= {PASS_CRITERIA['sharpe_gt']}")

    if m['ci_lo'] <= PASS_CRITERIA['ci_lower_gt']:
        passed = False
        reasons.append(f"CI_lo={m['ci_lo']:.2f} <= 0")

    if m['wr'] < PASS_CRITERIA['win_rate_gte']:
        passed = False
        reasons.append(f"WR={m['wr']*100:.0f}% < {PASS_CRITERIA['win_rate_gte']*100:.0f}%")

    if abs(m.get('max_dd_pct', 0)) > PASS_CRITERIA['max_dd_lt'] * 100:
        passed = False
        reasons.append(f"maxDD={m['max_dd_pct']:.1f}% > {PASS_CRITERIA['max_dd_lt']*100:.0f}%")

    if tpw > PASS_CRITERIA['trades_per_week_lte']:
        passed = False
        reasons.append(f"tpw={tpw:.1f} > {PASS_CRITERIA['trades_per_week_lte']}")

    if avg_hold < PASS_CRITERIA['avg_hold_hrs_gte']:
        passed = False
        reasons.append(f"avg_hold={avg_hold:.1f}h < {PASS_CRITERIA['avg_hold_hrs_gte']}h")

    beats_baseline = (m['avg'] > BASELINE['avg_pnl'] and
                      m['sharpe'] > BASELINE['sharpe'])

    return {
        'hypothesis': h,
        'metrics': m,
        'passed': passed,
        'beats_baseline': beats_baseline,
        'reasons': reasons if not passed else ['ALL PASS'],
        'trades_per_week': round(tpw, 2),
        'avg_hold_hrs': round(avg_hold, 1),
        'n_trades': m['n'],
    }


# =============================================================================
# RESULT LOGGING
# =============================================================================

def log_result(result, results_log):
    h = result['hypothesis']
    m = result['metrics']
    results_log.append(result)

    status = 'WINNER' if result['passed'] else 'FAIL'
    beats = ' [BEATS BASELINE]' if result.get('beats_baseline') else ''

    print(f"\n  {'─'*70}")
    print(f"  [{status}] {h['id']}: {h['desc']}{beats}")
    print(f"  n={m['n']}  avg={m['avg']:+.2f}%  Sharpe={m['sharpe']:.3f}  "
          f"WR={m['wr']*100:.0f}%  CI=[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]")
    print(f"  tpw={result['trades_per_week']:.1f}  avg_hold={result['avg_hold_hrs']:.1f}h  "
          f"maxDD={m.get('max_dd_pct', 0):.1f}%  PF={m['pf']:.2f}")
    if not result['passed']:
        print(f"  Fail reasons: {', '.join(result['reasons'])}")
    print(f"  {'─'*70}")

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, f"{h['id']}.json")
    with open(json_path, 'w') as f:
        json.dump({
            'id': h['id'],
            'desc': h['desc'],
            'direction': h['direction'],
            'catalyst': h.get('catalyst'),
            'entry': h.get('entry'),
            'exit': h.get('exit'),
            'filters': h.get('filters', {}),
            'passed': result['passed'],
            'beats_baseline': result.get('beats_baseline', False),
            'metrics': m,
            'trades_per_week': result['trades_per_week'],
            'avg_hold_hrs': result['avg_hold_hrs'],
            'reasons': result['reasons'],
            'n_trades': m['n'],
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        }, f, indent=2)


# =============================================================================
# WINNER OUTPUT
# =============================================================================

def write_winner(result, trades):
    h = result['hypothesis']
    m = result['metrics']

    md = f"""# WINNER: {h['id']}
**Strategy:** {h['desc']}
**Direction:** {h['direction']}
**Period tested:** Aug 1 - Oct 6 2026
**Avg PnL:** {m['avg']:+.2f}% | **Sharpe:** {m['sharpe']:.3f} | **n:** {m['n']} | **WR:** {m['wr']*100:.0f}%

## How to Trade This (Human Instructions)

### What to do the night before
- Run gap scanner at market close or pre-market
- Filter: {h.get('catalyst', 'gap')} candidates with gap >= {h.get('filters', {}).get('gap_min', 0.02)*100:.0f}%
- Universe: US equities, price >= ${PRICE_MIN}, float >= {FLOAT_MIN/1e6:.0f}M shares
- Expect {result['trades_per_week']:.1f} trades per week on average

### Entry
- Time: {h.get('entry', 'open_9:30')}
- Signal: {h['direction']} on {h.get('catalyst', 'gap')} stocks meeting filters
- Order type: market at specified time
- Max candidates per day: {h.get('filters', {}).get('top_n', TOP_N_DEFAULT)}

### Exit
- Primary: {h.get('exit', 'T+2_close')}
- Average hold: {result['avg_hold_hrs']:.1f} hours

### Position sizing
- ${CAPITAL:,} daily capital, equal-weight across day's trades
- Minimum position: $5,000

### What this strategy does NOT work on
- [Review monthly breakdown for weak periods]
- [Check worst trades for failure modes]

## Backtest Stats
| Metric | Value |
|--------|-------|
| n trades | {m['n']} |
| Avg PnL | {m['avg']:+.2f}% |
| Median PnL | {m['med']:+.2f}% |
| Trimmed Mean | {m['trimmed_mean']:+.2f}% |
| Sharpe | {m['sharpe']:.3f} |
| Win Rate | {m['wr']*100:.0f}% |
| Profit Factor | {m['pf']:.2f} |
| 95% CI | [{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}] |
| Max Loss | {m['max_loss']:+.2f}% |
| Max Gain | {m['max_gain']:+.2f}% |
| Max Drawdown | {m.get('max_dd_pct', 0):.1f}% |
| Trades/Week | {result['trades_per_week']:.1f} |
| Avg Hold | {result['avg_hold_hrs']:.1f} hours |

## Top 5 Worst Trades
| Date | Ticker | Gap | PnL |
|------|--------|-----|-----|
"""
    worst = sorted(trades, key=lambda t: t['pnl'])[:5]
    for t in worst:
        md += f"| {t['scan_date']} | {t['ticker']} | {t['gap_pct']*100:+.1f}% | {t['pnl']:+.2f}% |\n"

    md += f"""
## Top 5 Best Trades
| Date | Ticker | Gap | PnL |
|------|--------|-----|-----|
"""
    best = sorted(trades, key=lambda t: -t['pnl'])[:5]
    for t in best:
        md += f"| {t['scan_date']} | {t['ticker']} | {t['gap_pct']*100:+.1f}% | {t['pnl']:+.2f}% |\n"

    winner_path = os.path.join(WINNERS_DIR, f"{h['id']}_winner.md")
    with open(winner_path, 'w') as f:
        f.write(md)
    print(f"\n  WINNER written: {winner_path}")


# =============================================================================
# HYPOTHESIS GENERATION — autonomous next batch
# =============================================================================

def _dedup_key(h):
    """Canonical key for functional deduplication of hypotheses."""
    f = h.get('filters', {})
    eff_gap = max(f.get('gap_min', 0.02), SCANNER_FLOOR)
    pattern = f.get('pattern', 'none')
    return (h.get('direction', ''), h.get('catalyst', ''), round(eff_gap, 3),
            h.get('entry', ''), h.get('exit', ''), pattern,
            f.get('sector_incl', 'none'), f.get('sector_excl', 'none'),
            f.get('spy_gate', 'none'), f.get('lookback_days', 0),
            f.get('decline_thresh', 0), f.get('rise_thresh', 0))


def generate_next_batch(tested_results, winners, tested_ids, next_id_counter):
    new_hypotheses = []
    tested_keys = set()
    for r in tested_results:
        tested_keys.add(_dedup_key(r['hypothesis']))

    if not tested_results:
        return new_hypotheses, next_id_counter

    # Sort by Sharpe (whether pass or fail)
    valid = [r for r in tested_results if r['metrics']['n'] >= 5]
    if not valid:
        return new_hypotheses, next_id_counter

    top5 = sorted(valid, key=lambda r: -r['metrics']['sharpe'])[:5]

    # Identify patterns in top performers
    top_directions = [r['hypothesis']['direction'] for r in top5]
    top_catalysts = [r['hypothesis'].get('catalyst', '') for r in top5]
    top_entries = [r['hypothesis'].get('entry', '') for r in top5]
    top_exits = [r['hypothesis'].get('exit', '') for r in top5]

    best = top5[0]
    bh = best['hypothesis']
    bf = bh.get('filters', {})

    # Variation 1: Adjust gap threshold on best
    for gap_adj in [0.01, -0.01, 0.02]:
        new_gap = bf.get('gap_min', 0.02) + gap_adj
        if new_gap < 0.01 or new_gap > 0.15: continue
        h_id = f"H{next_id_counter:03d}"
        next_id_counter += 1
        new_filters = dict(bf)
        new_filters['gap_min'] = round(new_gap, 3)
        desc = f"{bh['direction']} {bh.get('catalyst','')} gap>={new_gap*100:.0f}% {bh.get('exit','')}"
        new_h = {
            'id': h_id, 'desc': desc,
            'direction': bh['direction'],
            'catalyst': bh.get('catalyst'),
            'entry': bh.get('entry'),
            'exit': bh.get('exit'),
            'filters': new_filters,
            'hold_approx': bh.get('hold_approx', ''),
            'notes': f'Gap threshold variation on {bh["id"]}',
        }
        if h_id not in tested_ids and _dedup_key(new_h) not in tested_keys:
            new_hypotheses.append(new_h)

    # Variation 2: Different exit on best
    exits_to_try = ['T+1_close', 'T+2_close', 'T+3_close', 'T+5_close', 'T+1_open', '17:00']
    for ex in exits_to_try:
        if ex == bh.get('exit'): continue
        h_id = f"H{next_id_counter:03d}"
        next_id_counter += 1
        desc = f"{bh['direction']} {bh.get('catalyst','')} gap>={bf.get('gap_min',0.02)*100:.0f}% {ex}"
        new_h = {
            'id': h_id, 'desc': desc,
            'direction': bh['direction'],
            'catalyst': bh.get('catalyst'),
            'entry': bh.get('entry'),
            'exit': ex,
            'filters': dict(bf),
            'hold_approx': ex,
            'notes': f'Exit variation on {bh["id"]}',
        }
        if h_id not in tested_ids and _dedup_key(new_h) not in tested_keys:
            new_hypotheses.append(new_h)

    # Variation 3: Different entry on best
    entries_to_try = ['open_9:30', 'local_high', 'local_low', '9:45', '10:00']
    for en in entries_to_try:
        if en == bh.get('entry'): continue
        h_id = f"H{next_id_counter:03d}"
        next_id_counter += 1
        desc = f"{bh['direction']} {bh.get('catalyst','')} entry={en} {bh.get('exit','')}"
        new_h = {
            'id': h_id, 'desc': desc,
            'direction': bh['direction'],
            'catalyst': bh.get('catalyst'),
            'entry': en,
            'exit': bh.get('exit'),
            'filters': dict(bf),
            'hold_approx': bh.get('hold_approx', ''),
            'notes': f'Entry variation on {bh["id"]}',
        }
        if h_id not in tested_ids:
            new_hypotheses.append(new_h)

    # Variation 4: Flip direction on top-2 if they haven't been tried
    for r in top5[:2]:
        rh = r['hypothesis']
        flipped = 'LONG' if rh['direction'] == 'SHORT' else 'SHORT'
        h_id = f"H{next_id_counter:03d}"
        next_id_counter += 1
        desc = f"{flipped} {rh.get('catalyst','')} gap>={rh.get('filters',{}).get('gap_min',0.02)*100:.0f}% {rh.get('exit','')}"
        new_h = {
            'id': h_id, 'desc': desc,
            'direction': flipped,
            'catalyst': rh.get('catalyst'),
            'entry': rh.get('entry'),
            'exit': rh.get('exit'),
            'filters': dict(rh.get('filters', {})),
            'hold_approx': rh.get('hold_approx', ''),
            'notes': f'Direction flip on {rh["id"]}',
        }
        if h_id not in tested_ids and _dedup_key(new_h) not in tested_keys:
            new_hypotheses.append(new_h)

    # Variation 5: Add pattern filter to best if none used
    if bf.get('pattern', 'none') == 'none':
        for pat in ['sustained_drop', 'sustained_rise']:
            h_id = f"H{next_id_counter:03d}"
            next_id_counter += 1
            new_filters = dict(bf)
            new_filters['pattern'] = pat
            desc = f"{bh['direction']} {bh.get('catalyst','')} +{pat} {bh.get('exit','')}"
            new_h = {
                'id': h_id, 'desc': desc,
                'direction': bh['direction'],
                'catalyst': bh.get('catalyst'),
                'entry': bh.get('entry'),
                'exit': bh.get('exit'),
                'filters': new_filters,
                'hold_approx': bh.get('hold_approx', ''),
                'notes': f'Pattern gate on {bh["id"]}',
            }
            if h_id not in tested_ids and _dedup_key(new_h) not in tested_keys:
                new_hypotheses.append(new_h)

    # Cap generated batch
    return new_hypotheses[:10], next_id_counter


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary(results_log):
    print(f"\n\n{'='*100}")
    print(f"  INVESTIGATION SUMMARY — {len(results_log)} hypotheses tested")
    print(f"{'='*100}")

    winners = [r for r in results_log if r['passed']]
    near_miss = [r for r in results_log if not r['passed'] and r['metrics']['n'] >= 5
                 and r['metrics']['sharpe'] > 0.10]

    sorted_all = sorted([r for r in results_log if r['metrics']['n'] >= 5],
                         key=lambda r: -r['metrics']['sharpe'])

    print(f"\n  {'ID':>6s}  {'Description':>40s}  {'n':>4s}  {'avg':>7s}  {'Sharpe':>7s}  "
          f"{'WR':>5s}  {'CI_lo':>7s}  {'tpw':>5s}  {'Status':>8s}")
    print(f"  {'-'*6}  {'-'*40}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*8}")

    for r in sorted_all:
        h = r['hypothesis']
        m = r['metrics']
        status = 'WINNER' if r['passed'] else 'FAIL'
        desc = h['desc'][:40]
        print(f"  {h['id']:>6s}  {desc:>40s}  {m['n']:>4d}  {m['avg']:>+6.2f}%  "
              f"{m['sharpe']:>7.3f}  {m['wr']*100:>4.0f}%  {m['ci_lo']:>+6.2f}  "
              f"{r['trades_per_week']:>5.1f}  {status:>8s}")

    print(f"\n  Winners: {len(winners)}")
    print(f"  Near misses (n>=5, Sharpe>0.10): {len(near_miss)}")
    print(f"  Baseline: avg={BASELINE['avg_pnl']}%, Sharpe={BASELINE['sharpe']}")


# =============================================================================
# =============================================================================
#
#  MAIN INVESTIGATION LOOP
#
# =============================================================================
# =============================================================================

# Import seed hypotheses
sys.path.insert(0, BASE_DIR)
from hypothesis_space import HYPOTHESES

t_start = time.time()
print(f"\n{'='*80}")
print(f"  PROJECT X — SEMI-AUTONOMOUS STRATEGY INVESTIGATION")
print(f"  Eval period: {EVAL_START} -> {EVAL_END}")
print(f"  Baseline to beat: avg > {BASELINE['avg_pnl']}%, Sharpe > {BASELINE['sharpe']}")
print(f"  Max hypotheses: {MAX_HYPOTHESES}, Max winners: {MAX_WINNERS}")
print(f"  Trades/week limit: {PASS_CRITERIA['trades_per_week_lte']}")
print(f"{'='*80}")

# Load data pipeline (cached across hypotheses)
data = run_data_pipeline(EVAL_START, EVAL_END)
dates = data['dates']

# Run investigation loop
results_log = []
winners = []
tested_ids = set()
hypothesis_queue = list(HYPOTHESES)
next_id_counter = len(HYPOTHESES) + 1

batch_num = 0
while hypothesis_queue and len(tested_ids) < MAX_HYPOTHESES and len(winners) < MAX_WINNERS:
    batch_num += 1
    h = hypothesis_queue.pop(0)

    if h['id'] in tested_ids:
        continue
    tested_ids.add(h['id'])

    print(f"\n\n{'═'*80}")
    print(f"  HYPOTHESIS {h['id']} ({len(tested_ids)}/{MAX_HYPOTHESES})")
    print(f"  {h['desc']}")
    print(f"  Direction: {h['direction']}  Entry: {h.get('entry')}  Exit: {h.get('exit')}")
    print(f"{'═'*80}")

    try:
        trades = run_backtest(h, data)
        result = evaluate_hypothesis(h, trades, dates)
        log_result(result, results_log)

        if result['passed']:
            winners.append(result)
            write_winner(result, trades)
            print(f"\n  >>> WINNER #{len(winners)} FOUND! <<<")
            if len(winners) >= MAX_WINNERS:
                print(f"\n  {MAX_WINNERS} winners found — stopping.")
                break

        # Generate next batch after every 5 tested or when queue empty
        if len(hypothesis_queue) == 0 or len(tested_ids) % 5 == 0:
            new_batch, next_id_counter = generate_next_batch(
                results_log, winners, tested_ids, next_id_counter)
            hypothesis_queue.extend(new_batch)
            if new_batch:
                print(f"\n  Generated {len(new_batch)} new hypotheses: "
                      f"{', '.join(h['id'] for h in new_batch)}")

    except Exception as e:
        print(f"\n  ERROR on {h['id']}: {e}")
        results_log.append({
            'hypothesis': h, 'metrics': {'n': 0},
            'passed': False, 'reasons': [f'ERROR: {str(e)}'],
            'trades_per_week': 0, 'avg_hold_hrs': 0, 'n_trades': 0,
        })

# Final summary
print_summary(results_log)

# If any winners found, run full-period validation
if winners:
    print(f"\n\n{'='*80}")
    print(f"  FULL-PERIOD VALIDATION ({FULL_START} -> {EVAL_END})")
    print(f"{'='*80}")
    full_data = run_data_pipeline(FULL_START, EVAL_END)
    for w in winners:
        h = w['hypothesis']
        print(f"\n  Validating {h['id']}: {h['desc']}")
        full_trades = run_backtest(h, full_data)
        full_result = evaluate_hypothesis(h, full_trades, full_data['dates'])
        m = full_result['metrics']
        print(f"  Full period: n={m['n']}  avg={m['avg']:+.2f}%  Sharpe={m['sharpe']:.3f}  "
              f"WR={m['wr']*100:.0f}%  CI=[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]")
        if full_result['passed']:
            print(f"  >> FULL PERIOD PASS — strategy is robust!")
        else:
            print(f"  >> Full period: {', '.join(full_result['reasons'])}")

elapsed = time.time() - t_start
print(f"\n{'='*80}")
print(f"  TOTAL RUNTIME: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"  Hypotheses tested: {len(tested_ids)}")
print(f"  Winners found: {len(winners)}")
print(f"{'='*80}")
