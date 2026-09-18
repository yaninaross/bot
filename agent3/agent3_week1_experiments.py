# %% Agent 3 — Week 1 Experiments
#
# Five priority experiments run in a single pass (data fetched once):
#   C1-E01: Day of week — Wed/Thu only
#   A6-E01: Volume filter — vol_ratio < 2x ADV
#   A6-E02: Regime filter — exclude bull SPY
#   C1-E04: Decline threshold — prior 5d return <= -20%
#   C1-E07: Gap-UP threshold — gap >= 8%
#
# Uses baseline backtest engine (agent3_baseline_bt.py) for data fetching.
# Each experiment filters the baseline trade list — only the filter changes.
# Frozen baselines repeated for side-by-side comparison.
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

BACKTEST_START = '2025-09-15'
BACKTEST_END   = '2026-09-12'
OOS_CUTOFF     = '2026-01-01'

# A6 config (baseline)
A6_GAP_MIN_PCT   = 5.0
A6_PRICE_MIN     = 10.0
A6_TOP_N         = 5
A6_PRE_FILTER_N  = 15
A6_HOLD_DAYS     = 1         # T+1 (E1/E1b: Sharpe 0.600 vs T+2 0.458)

# C1 config (baseline)
C1_GAP_MIN_PCT     = 3.0
C1_GAP_MAX_PCT     = 50.0
C1_PRICE_MIN       = 20.0
C1_DECLINE_5D_MIN  = 0.15
C1_HOLD_DAYS       = 3

# Entry config (A6 local low)
LOW_WINDOW     = 3
MAX_LOW_BAR    = 12
FALLBACK_BAR   = 15
SKIP_BOUNCE_PCT = 0.15

FLOAT_MIN      = 10_000_000
CAPITAL        = 100_000

# Volume filter (A6-E01)
A6_E01_VOL_MAX = 2.0     # exclude if vol_ratio >= 2x ADV
ADV_LOOKBACK   = 20       # 20 trading day average volume

# Regime filter (A6-E02)
A6_E02_SPY_BULL_THRESH = 0.02  # exclude if SPY 5d return > +2%

# Decline threshold (C1-E04)
C1_E04_DECLINE_MIN = 0.20  # prior 5d decline >= 20%

# Gap threshold (C1-E07)
C1_E07_GAP_MIN_PCT = 8.0   # gap-UP >= 8%

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')
ET = pytz.timezone('America/New_York')

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
# GAP-DOWN PATTERN CLASSIFICATION (A6)
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


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL LOW DETECTION (A6 entry)
# ═══════════════════════════════════════════════════════════════════════════════

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


def build_result(trades, dates, label, hold_days):
    pnl_key = f'pnl_t{hold_days}'
    pnls = [t[pnl_key] for t in trades if t.get(pnl_key) is not None]
    m = compute_metrics(pnls, label)

    daily_pnl = defaultdict(list)
    for t in trades:
        if t.get(pnl_key) is not None:
            daily_pnl[t['scan_date']].append(t[pnl_key])

    cum_nc = 0.0
    for dt in dates:
        dp = daily_pnl.get(dt, [])
        if dp:
            n_pos = len(dp)
            pos_size = CAPITAL / n_pos
            cum_nc += sum(pos_size * (r / 100) for r in dp)

    equity = [CAPITAL]
    eq_dates = []
    running_capital = CAPITAL
    for dt in sorted(daily_pnl):
        n_pos = len(daily_pnl[dt])
        pos_size = running_capital / n_pos
        day_dollar = sum(pos_size * (r / 100) for r in daily_pnl[dt])
        running_capital += day_dollar
        equity.append(running_capital)
        eq_dates.append(dt)

    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = float(dd.min())
    active_days = sum(1 for dt in dates if dt in daily_pnl)
    tpd = len(pnls) / active_days if active_days > 0 else 0

    is_pnls = [t[pnl_key] for t in trades
               if t.get(pnl_key) is not None and t['scan_date'] < OOS_CUTOFF]
    oos_pnls = [t[pnl_key] for t in trades
                if t.get(pnl_key) is not None and t['scan_date'] >= OOS_CUTOFF]
    is_m = compute_metrics(is_pnls, f'{label} IS')
    oos_m = compute_metrics(oos_pnls, f'{label} OOS')

    degradation = None
    if is_m['avg'] != 0 and is_m['n'] > 0 and oos_m['n'] > 0:
        degradation = (oos_m['avg'] - is_m['avg']) / abs(is_m['avg']) * 100

    status = 'DEAD'
    if m['ci_lo'] > 0:
        if (oos_m['ci_lo'] > 0 and oos_m['avg'] >= 3.0 and oos_m['sharpe'] >= 0.15
                and m['wr'] >= 0.58 and degradation is not None and degradation > -50):
            status = 'PENSION READY'
        elif (oos_m['avg'] >= 3.0 and oos_m['sharpe'] >= 0.15
              and is_m['n'] < 50):
            status = 'NEAR MISS'
        elif oos_m['n'] > 0 and oos_m['avg'] < is_m['avg'] * 0.6:
            status = 'DEGRADED'
        elif oos_m['avg'] < 3.0 or oos_m['ci_lo'] < 0:
            status = 'DEAD'
        else:
            status = 'NEAR MISS'

    return {
        'label': label, 'trades': trades, 'metrics': m,
        'is_metrics': is_m, 'oos_metrics': oos_m,
        'degradation': degradation, 'status': status,
        'equity': equity, 'eq_dates': eq_dates,
        'cum_nc': cum_nc, 'max_dd': max_dd,
        'final_capital': running_capital,
        'active_days': active_days, 'tpd': tpd, 'hold_days': hold_days,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  BACKTEST ENGINE — Extended with experiment metadata
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest():
    t_total = time.time()
    timings = {}

    print(f"\n{'='*80}")
    print(f"  AGENT 3 — WEEK 1 EXPERIMENTS (LONG)")
    print(f"  {BACKTEST_START} -> {BACKTEST_END}  |  OOS cutoff: {OOS_CUTOFF}")
    print(f"  Universe: _FLOAT ({len(_FLOAT_SET)} tickers with float >= 10M)")
    print(f"  Experiments: C1-E01 WedThu, A6-E01 Vol<2x, A6-E02 Regime,")
    print(f"               C1-E04 Decline>=20%, C1-E07 Gap>=8%")
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

    # ── Extended lookback for ADV computation ──
    adv_lookback_start = _nth_prior_td(BACKTEST_START, ADV_LOOKBACK + 10)
    lb = adv_lookback_start
    lookback_dates = []
    while lb < datetime.strptime(BACKTEST_START, '%Y-%m-%d').date():
        if _is_td(lb): lookback_dates.append(lb.strftime('%Y-%m-%d'))
        lb += timedelta(days=1)

    # ── Phase 1: Fetch grouped daily ──
    t1 = time.time()
    all_fetch_dates = sorted(set(lookback_dates + dates))
    print(f"\n  Phase 1: Fetching grouped daily for {len(all_fetch_dates)} dates...")
    for i, dt in enumerate(all_fetch_dates):
        fetch_grouped_daily(dt)
        if (i+1) % 30 == 0 or i == len(all_fetch_dates) - 1:
            print(f"    [{i+1:>3}/{len(all_fetch_dates)}] {dt}  |  {time.time()-t_total:.0f}s")
    timings['daily'] = time.time() - t1

    # Outcome dates
    outcome_dates = set()
    for dt in dates:
        for nd in _next_n_td(dt, max(A6_HOLD_DAYS, C1_HOLD_DAYS)):
            outcome_dates.add(nd.strftime('%Y-%m-%d'))
    for od in sorted(outcome_dates):
        if od not in _gd_cache:
            try: fetch_grouped_daily(od)
            except: pass
    print(f"    + {len(outcome_dates)} outcome dates fetched")

    all_td_sorted = sorted(set(
        all_fetch_dates + [od for od in outcome_dates if od not in set(all_fetch_dates)]
    ))

    # ── Build ordered list of all trading days for lookback ──
    td_list = sorted(d for d in all_td_sorted if _is_td(d))
    td_to_idx = {d: i for i, d in enumerate(td_list)}

    # ── Compute SPY 5d return for each backtest date (for A6-E02) ──
    spy_5d_return = {}
    for dt in dates:
        spy_today = _gd_cache.get(dt, {}).get('SPY', {})
        if not spy_today or spy_today['c'] <= 0:
            spy_5d_return[dt] = None
            continue
        try:
            spy_5d_ago_date = _nth_prior_td(dt, 5).strftime('%Y-%m-%d')
        except ValueError:
            spy_5d_return[dt] = None
            continue
        spy_5d_ago = _gd_cache.get(spy_5d_ago_date, {}).get('SPY', {})
        if not spy_5d_ago or spy_5d_ago['c'] <= 0:
            spy_5d_return[dt] = None
            continue
        spy_5d_return[dt] = (spy_today['c'] - spy_5d_ago['c']) / spy_5d_ago['c']

    # ── Phase 2: Scan A6 and C1 candidates ──
    t2 = time.time()
    print(f"\n  Phase 2: Scanning candidates...")
    a6_candidates = {}
    c1_candidates = {}
    total_a6 = 0
    total_c1 = 0

    gap_min_a6 = A6_GAP_MIN_PCT / 100.0
    gap_min_c1 = C1_GAP_MIN_PCT / 100.0

    for i, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = _gd_cache.get(prior_str, {})
        today = _gd_cache.get(dt, {})
        nxt_a6 = _next_n_td(dt, A6_HOLD_DAYS)
        nxt_c1 = _next_n_td(dt, C1_HOLD_DAYS)

        # ── A6: Gap-DOWN candidates ──
        a6_day = []
        for tk in _FLOAT_SET:
            if tk in _EXCLUDE_TK: continue
            if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
            if tk not in prior or tk not in today: continue
            prev_c = prior[tk]['c']
            if prev_c <= 0: continue
            cur_o = today[tk]['o']
            if cur_o <= 0 or cur_o < A6_PRICE_MIN: continue
            gap = (cur_o - prev_c) / prev_c
            if gap > -gap_min_a6: continue

            # Compute ADV (20-day average volume) for A6-E01
            dt_idx = td_to_idx.get(dt)
            adv_20 = None
            if dt_idx is not None and dt_idx >= ADV_LOOKBACK:
                vols = []
                for j in range(1, ADV_LOOKBACK + 1):
                    lb_dt = td_list[dt_idx - j]
                    lb_data = _gd_cache.get(lb_dt, {}).get(tk, {})
                    if lb_data and lb_data.get('v', 0) > 0:
                        vols.append(lb_data['v'])
                if len(vols) >= 10:
                    adv_20 = np.mean(vols)

            vol_ratio = None
            if adv_20 is not None and adv_20 > 0:
                vol_ratio = today[tk]['v'] / adv_20

            a6_day.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_o, 'scan_date': dt,
                'volume_today': today[tk]['v'],
                'vol_ratio': vol_ratio,
                'adv_20': adv_20,
                'spy_5d_return': spy_5d_return.get(dt),
                'day_of_week': datetime.strptime(dt, '%Y-%m-%d').weekday(),
                't_dates': [nd.strftime('%Y-%m-%d') for nd in nxt_a6],
            })
        a6_day.sort(key=lambda x: x['gap_pct'])
        a6_candidates[dt] = a6_day
        total_a6 += len(a6_day)

        # ── C1: Gap-UP + prior 5d decline candidates ──
        c1_day = []
        for tk in _FLOAT_SET:
            if tk in _EXCLUDE_TK: continue
            if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
            if tk not in prior or tk not in today: continue
            prev_c = prior[tk]['c']
            if prev_c <= 0: continue
            cur_o = today[tk]['o']
            if cur_o <= 0 or cur_o < C1_PRICE_MIN: continue
            gap = (cur_o - prev_c) / prev_c
            if gap < gap_min_c1: continue
            if gap > C1_GAP_MAX_PCT / 100.0: continue

            try:
                close_5d_ago_date = _nth_prior_td(prior_str, 4).strftime('%Y-%m-%d')
            except ValueError:
                continue
            close_5d_ago_data = _gd_cache.get(close_5d_ago_date, {}).get(tk)
            if not close_5d_ago_data or close_5d_ago_data['c'] <= 0:
                continue
            prior_5d_return = (prev_c - close_5d_ago_data['c']) / close_5d_ago_data['c']
            if prior_5d_return > -C1_DECLINE_5D_MIN:
                continue

            c1_day.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_o, 'scan_date': dt,
                'volume_today': today[tk]['v'],
                'prior_5d_return': prior_5d_return,
                'day_of_week': datetime.strptime(dt, '%Y-%m-%d').weekday(),
                'spy_5d_return': spy_5d_return.get(dt),
                't_dates': [nd.strftime('%Y-%m-%d') for nd in nxt_c1],
            })
        c1_day.sort(key=lambda x: -x['gap_pct'])
        c1_candidates[dt] = c1_day
        total_c1 += len(c1_day)

        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  A6={total_a6}, C1={total_c1}  |  {time.time()-t_total:.0f}s")
    timings['scan'] = time.time() - t2

    # ── Phase 3: Fetch minute bars for A6 candidates ──
    t3 = time.time()
    a6_pairs = set()
    for dt, cands in a6_candidates.items():
        for c in cands[:A6_PRE_FILTER_N]:
            a6_pairs.add((c['ticker'], dt))

    print(f"\n  Phase 3: Fetching 1-min bars for {len(a6_pairs)} A6 pairs...")
    pairs_list = list(a6_pairs)
    all_bars = {}
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start + 100]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        done = min(c_start + 100, len(pairs_list))
        if done % 200 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    timings['bars'] = time.time() - t3
    print(f"  -> Got bars for {len(all_bars)}/{len(a6_pairs)} pairs")

    # ── Phase 4: Fetch ticker reference ──
    t4 = time.time()
    all_tickers = set()
    for cands in a6_candidates.values():
        for c in cands[:A6_PRE_FILTER_N]:
            all_tickers.add(c['ticker'])
    for cands in c1_candidates.values():
        for c in cands:
            all_tickers.add(c['ticker'])
    print(f"\n  Phase 4: Fetching reference data for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    for c_start in range(0, len(tk_list), 200):
        chunk = tk_list[c_start:c_start + 200]
        fetch_ticker_refs(chunk)
        done = min(c_start + 200, len(tk_list))
        if done % 400 == 0 or done == len(tk_list):
            print(f"    {done}/{len(tk_list)}  |  {time.time()-t_total:.0f}s")
    timings['refs'] = time.time() - t4

    # ── Phase 5: Classify A6 patterns + find local low ──
    t5 = time.time()
    pattern_cache = {}
    entry_cache = {}
    for key, rth_bars in all_bars.items():
        pattern_cache[key] = classify_gap_down_pattern(rth_bars)
        if rth_bars and len(rth_bars) >= LOW_WINDOW * 2 + 1:
            entry_cache[key] = find_local_low(rth_bars)
        else:
            entry_cache[key] = None
    timings['classify'] = time.time() - t5

    # ── Phase 6: Build A6 trades (with experiment metadata) ──
    t6 = time.time()
    print(f"\n  Phase 6: Building trade lists...")
    a6_trades = []

    for dt in dates:
        cands = a6_candidates.get(dt, [])
        qualified = []
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
            tc['open_pattern'] = pat
            tc['asset_type'] = at
            tc['strategy'] = 'A6'

            ep = tc['entry_price']
            for hi, td_str in enumerate(tc['t_dates'], 1):
                td_data = _gd_cache.get(td_str, {}).get(tk, {})
                if td_data.get('c'):
                    tc[f'pnl_t{hi}'] = round((td_data['c'] - ep) / ep * 100, 3)
                else:
                    tc[f'pnl_t{hi}'] = None

            qualified.append(tc)

        for tc in qualified[:A6_TOP_N]:
            if tc.get(f'pnl_t{A6_HOLD_DAYS}') is not None:
                a6_trades.append(tc)

    # ── Phase 7: Build C1 trades ──
    c1_trades = []

    for dt in dates:
        cands = c1_candidates.get(dt, [])
        for c in cands:
            tk = c['ticker']
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS': continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = tc['open_price']
            tc['entry_time'] = '09:30'
            tc['entry_type'] = 'open'
            tc['open_pattern'] = 'c1_reversal'
            tc['asset_type'] = at
            tc['strategy'] = 'C1'

            ep = tc['entry_price']
            has_exit = False
            for hi, td_str in enumerate(tc['t_dates'], 1):
                td_data = _gd_cache.get(td_str, {}).get(tk, {})
                if td_data.get('c'):
                    tc[f'pnl_t{hi}'] = round((td_data['c'] - ep) / ep * 100, 3)
                    if hi == C1_HOLD_DAYS:
                        has_exit = True
                else:
                    tc[f'pnl_t{hi}'] = None

            if has_exit:
                c1_trades.append(tc)

    timings['build'] = time.time() - t6
    elapsed = time.time() - t_total

    print(f"\n  -> A6 baseline: {len(a6_trades)} trades")
    print(f"  -> C1 baseline: {len(c1_trades)} trades")

    print(f"\n  {'─'*50}")
    for step, dur in [('Daily', timings['daily']), ('Scan', timings['scan']),
                      ('Bars', timings['bars']), ('Refs', timings['refs']),
                      ('Classify', timings['classify']), ('Build', timings['build'])]:
        print(f"    {step:<12s}  {dur:>6.0f}s  ({dur/elapsed*100:>4.0f}%)")
    print(f"  TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return a6_trades, c1_trades, dates


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  EXPERIMENT FILTERS
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {}


def register_experiments(a6_trades, c1_trades, dates):
    """Apply each experiment's filter and register results."""

    # ── BASELINE-A6 (frozen reference) ──
    EXPERIMENTS['BASELINE-A6'] = {
        'id': 'BASELINE-A6', 'strategy': 'A6',
        'description': 'Gap-down sustained rise, local_low entry, T+1',
        'change': 'None (frozen baseline)',
        'trades': a6_trades,
        'hold_days': A6_HOLD_DAYS,
    }

    # ── BASELINE-C1 (frozen reference) ──
    EXPERIMENTS['BASELINE-C1'] = {
        'id': 'BASELINE-C1', 'strategy': 'C1',
        'description': 'Multi-day reversal, 5d decline >= 15%, gap-up >= 3%, T+3',
        'change': 'None (frozen baseline)',
        'trades': c1_trades,
        'hold_days': C1_HOLD_DAYS,
    }

    # ── C1-E01: Wednesday/Thursday only ──
    c1_e01 = [t for t in c1_trades if t['day_of_week'] in [2, 3]]
    EXPERIMENTS['C1-E01'] = {
        'id': 'C1-E01', 'strategy': 'C1',
        'description': 'C1 on Wed/Thu only',
        'change': 'entry_date.weekday() in [2, 3]',
        'trades': c1_e01,
        'hold_days': C1_HOLD_DAYS,
    }

    # ── A6-E01: Volume < 2x ADV ──
    a6_e01 = [t for t in a6_trades
              if t.get('vol_ratio') is None or t['vol_ratio'] < A6_E01_VOL_MAX]
    EXPERIMENTS['A6-E01'] = {
        'id': 'A6-E01', 'strategy': 'A6',
        'description': 'A6 with vol_ratio < 2x ADV',
        'change': f'EXCLUDE if vol_ratio >= {A6_E01_VOL_MAX}x (include if ADV unavailable)',
        'trades': a6_e01,
        'hold_days': A6_HOLD_DAYS,
    }

    # ── A6-E02: Regime — exclude bull SPY ──
    a6_e02 = [t for t in a6_trades
              if t.get('spy_5d_return') is None or t['spy_5d_return'] <= A6_E02_SPY_BULL_THRESH]
    EXPERIMENTS['A6-E02'] = {
        'id': 'A6-E02', 'strategy': 'A6',
        'description': 'A6 excluding bull regime (SPY 5d > +2%)',
        'change': f'EXCLUDE if spy_5d_return > {A6_E02_SPY_BULL_THRESH*100:.0f}%',
        'trades': a6_e02,
        'hold_days': A6_HOLD_DAYS,
    }

    # ── C1-E04: Decline threshold >= 20% ──
    c1_e04 = [t for t in c1_trades if t['prior_5d_return'] <= -C1_E04_DECLINE_MIN]
    EXPERIMENTS['C1-E04'] = {
        'id': 'C1-E04', 'strategy': 'C1',
        'description': 'C1 with prior 5d decline >= 20%',
        'change': f'prior_5d_return <= -{C1_E04_DECLINE_MIN*100:.0f}%',
        'trades': c1_e04,
        'hold_days': C1_HOLD_DAYS,
    }

    # ── C1-E07: Gap-UP >= 8% ──
    c1_e07 = [t for t in c1_trades if t['gap_pct'] >= C1_E07_GAP_MIN_PCT / 100.0]
    EXPERIMENTS['C1-E07'] = {
        'id': 'C1-E07', 'strategy': 'C1',
        'description': 'C1 with gap-up >= 8%',
        'change': f'gap_pct >= {C1_E07_GAP_MIN_PCT:.0f}%',
        'trades': c1_e07,
        'hold_days': C1_HOLD_DAYS,
    }

    return EXPERIMENTS


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS — per experiment
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_experiments(experiments, dates):
    results = {}
    for exp_id, exp in experiments.items():
        res = build_result(exp['trades'], dates, exp_id, exp['hold_days'])
        res['description'] = exp['description']
        res['change'] = exp['change']
        res['strategy'] = exp['strategy']
        results[exp_id] = res
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT — Master summary table + per-experiment detail
# ═══════════════════════════════════════════════════════════════════════════════

def print_experiment_results(results, dates):
    print(f"\n\n{'='*120}")
    print(f"  AGENT 3 — WEEK 1 EXPERIMENT RESULTS")
    print(f"  {dates[0]} -> {dates[-1]}  |  OOS cutoff: {OOS_CUTOFF}")
    print(f"  Capital: ${CAPITAL:,.0f}/day  |  Non-compounding")
    print(f"{'='*120}")

    # ── Master comparison table ──
    exp_order = ['BASELINE-A6', 'A6-E01', 'A6-E02',
                 'BASELINE-C1', 'C1-E01', 'C1-E04', 'C1-E07']

    headers = ['Experiment', 'n', 'Avg%', 'Med%', 'TrMean%', 'WR%',
               'Sharpe', 'PF', '95% CI', 'CI>0',
               'IS_n', 'IS_avg', 'OOS_n', 'OOS_avg', 'OOS_Sh', 'Deg%',
               'Status', '$ NC']

    rows = []
    for eid in exp_order:
        r = results[eid]
        m = r['metrics']
        im = r['is_metrics']
        om = r['oos_metrics']
        deg = r['degradation']
        deg_str = f"{deg:+.0f}" if deg is not None else '—'
        st = r['status']
        if st == 'PENSION READY':
            st_str = '★ READY'
        elif st == 'NEAR MISS':
            st_str = '◇ NEAR'
        elif st == 'DEGRADED':
            st_str = '✗ DEGR'
        else:
            st_str = '✗ DEAD'

        rows.append([
            eid,
            f"{m['n']}",
            f"{m['avg']:+.2f}",
            f"{m['med']:+.2f}",
            f"{m['trimmed_mean']:+.2f}",
            f"{m['wr']*100:.0f}",
            f"{m['sharpe']:.3f}",
            f"{m['pf']:.2f}",
            f"[{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}]",
            'Y' if m['ci_lo'] > 0 else 'N',
            f"{im['n']}",
            f"{im['avg']:+.2f}",
            f"{om['n']}",
            f"{om['avg']:+.2f}",
            f"{om['sharpe']:.3f}",
            deg_str,
            st_str,
            f"${r['cum_nc']:+,.0f}",
        ])
    print()
    print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right'))

    # ── Per-experiment detail blocks ──
    for eid in exp_order:
        if eid.startswith('BASELINE'): continue
        r = results[eid]
        m = r['metrics']
        im = r['is_metrics']
        om = r['oos_metrics']
        pnl_key = f'pnl_t{r["hold_days"]}'

        # Find baseline for comparison
        base_id = f'BASELINE-{r["strategy"]}'
        bm = results[base_id]['metrics']

        print(f"\n\n{'─'*90}")
        print(f"  {eid}: {r['description']}")
        print(f"  Change: {r['change']}")
        print(f"  Status: {r['status']}")
        print(f"{'─'*90}")

        print(f"\n  vs BASELINE:")
        print(f"    n:      {bm['n']} -> {m['n']}  ({m['n']-bm['n']:+d}, {(m['n']-bm['n'])/bm['n']*100:+.0f}%)")
        print(f"    avg:    {bm['avg']:+.2f}% -> {m['avg']:+.2f}%  ({m['avg']-bm['avg']:+.2f}pp)")
        print(f"    Sharpe: {bm['sharpe']:.3f} -> {m['sharpe']:.3f}  ({m['sharpe']-bm['sharpe']:+.3f})")
        print(f"    WR:     {bm['wr']*100:.0f}% -> {m['wr']*100:.0f}%  ({(m['wr']-bm['wr'])*100:+.0f}pp)")
        print(f"    CI:     [{bm['ci_lo']:+.2f},{bm['ci_hi']:+.2f}] -> [{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}]")
        print(f"    $ NC:   ${results[base_id]['cum_nc']:+,.0f} -> ${r['cum_nc']:+,.0f}")

        print(f"\n  OOS Validation:")
        deg = r['degradation']
        deg_str = f"{deg:+.0f}%" if deg is not None else 'N/A'
        print(f"    IS:  n={im['n']}, avg={im['avg']:+.2f}%, Sharpe={im['sharpe']:.3f}, "
              f"CI [{im['ci_lo']:+.2f},{im['ci_hi']:+.2f}]")
        print(f"    OOS: n={om['n']}, avg={om['avg']:+.2f}%, Sharpe={om['sharpe']:.3f}, "
              f"CI [{om['ci_lo']:+.2f},{om['ci_hi']:+.2f}]")
        print(f"    Degradation: {deg_str}")

        # Worst 5 / Best 5
        valid = [t for t in r['trades'] if t.get(pnl_key) is not None]
        by_pnl = sorted(valid, key=lambda x: x[pnl_key])
        for lbl, sl in [('WORST 5', by_pnl[:5]), ('BEST 5', by_pnl[-5:])]:
            print(f"\n  {lbl}:")
            for t in sl:
                pnl = t[pnl_key]
                gap = t['gap_pct'] * 100
                extras = []
                if t.get('vol_ratio') is not None:
                    extras.append(f"vol={t['vol_ratio']:.1f}x")
                if t.get('prior_5d_return') is not None:
                    extras.append(f"5d={t['prior_5d_return']*100:+.1f}%")
                if t.get('spy_5d_return') is not None:
                    extras.append(f"spy={t['spy_5d_return']*100:+.1f}%")
                dow_names = ['Mon','Tue','Wed','Thu','Fri']
                extras.append(dow_names[t['day_of_week']])
                extra_str = '  '.join(extras)
                print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={gap:>+7.1f}%  "
                      f"pnl={pnl:>+8.2f}%  {extra_str}")

        # Monthly breakdown
        monthly = defaultdict(list)
        for t in r['trades']:
            if t.get(pnl_key) is not None:
                monthly[t['scan_date'][:7]].append(t[pnl_key])
        if monthly:
            print(f"\n  Monthly:")
            mo_rows = []
            for mo in sorted(monthly):
                ps = monthly[mo]
                dollar = sum(CAPITAL / max(1, len(ps)) * (p / 100) for p in ps)
                mo_rows.append([mo, len(ps), f"{np.mean(ps):+.2f}%",
                                f"{np.median(ps):+.2f}%", f"${dollar:+,.0f}"])
            print(tabulate(mo_rows, headers=['Month','n','Avg','Median','$NC'],
                           tablefmt='simple'))

    # ── A6-E01 volume distribution ──
    print(f"\n\n{'─'*90}")
    print(f"  A6-E01 VOLUME RATIO DISTRIBUTION")
    print(f"{'─'*90}")
    a6_base = results['BASELINE-A6']['trades']
    pnl_key_a6 = f'pnl_t{A6_HOLD_DAYS}'
    vol_buckets = {'<0.5x': [], '0.5-1x': [], '1-2x': [], '2-5x': [], '5x+': [], 'N/A': []}
    for t in a6_base:
        pnl = t.get(pnl_key_a6)
        if pnl is None: continue
        vr = t.get('vol_ratio')
        if vr is None:
            vol_buckets['N/A'].append(pnl)
        elif vr < 0.5:
            vol_buckets['<0.5x'].append(pnl)
        elif vr < 1.0:
            vol_buckets['0.5-1x'].append(pnl)
        elif vr < 2.0:
            vol_buckets['1-2x'].append(pnl)
        elif vr < 5.0:
            vol_buckets['2-5x'].append(pnl)
        else:
            vol_buckets['5x+'].append(pnl)
    vb_rows = []
    for bkt in ['<0.5x', '0.5-1x', '1-2x', '2-5x', '5x+', 'N/A']:
        ps = vol_buckets[bkt]
        if ps:
            a = np.array(ps)
            sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
            vb_rows.append([bkt, len(ps), f"{np.mean(ps):+.2f}%",
                            f"{(a>0).mean()*100:.0f}%", f"{sh:.3f}"])
        else:
            vb_rows.append([bkt, 0, '—', '—', '—'])
    print(tabulate(vb_rows, headers=['Vol Ratio', 'n', 'Avg', 'WR', 'Sharpe'],
                   tablefmt='simple'))

    # ── A6-E02 regime distribution ──
    print(f"\n\n{'─'*90}")
    print(f"  A6-E02 SPY REGIME DISTRIBUTION")
    print(f"{'─'*90}")
    regime_buckets = {'Bull (>+2%)': [], 'Flat (±2%)': [], 'Bear (<-2%)': [], 'N/A': []}
    for t in a6_base:
        pnl = t.get(pnl_key_a6)
        if pnl is None: continue
        spy = t.get('spy_5d_return')
        if spy is None:
            regime_buckets['N/A'].append(pnl)
        elif spy > 0.02:
            regime_buckets['Bull (>+2%)'].append(pnl)
        elif spy < -0.02:
            regime_buckets['Bear (<-2%)'].append(pnl)
        else:
            regime_buckets['Flat (±2%)'].append(pnl)
    rb_rows = []
    for bkt in ['Bull (>+2%)', 'Flat (±2%)', 'Bear (<-2%)', 'N/A']:
        ps = regime_buckets[bkt]
        if ps:
            a = np.array(ps)
            sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
            rb_rows.append([bkt, len(ps), f"{np.mean(ps):+.2f}%",
                            f"{(a>0).mean()*100:.0f}%", f"{sh:.3f}"])
        else:
            rb_rows.append([bkt, 0, '—', '—', '—'])
    print(tabulate(rb_rows, headers=['Regime', 'n', 'Avg', 'WR', 'Sharpe'],
                   tablefmt='simple'))

    # ── C1-E01 day-of-week distribution ──
    print(f"\n\n{'─'*90}")
    print(f"  C1-E01 DAY-OF-WEEK DISTRIBUTION (C1 baseline)")
    print(f"{'─'*90}")
    pnl_key_c1 = f'pnl_t{C1_HOLD_DAYS}'
    c1_base = results['BASELINE-C1']['trades']
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    dow_buckets = {i: [] for i in range(5)}
    for t in c1_base:
        pnl = t.get(pnl_key_c1)
        if pnl is None: continue
        dow_buckets[t['day_of_week']].append(pnl)
    dow_rows = []
    for i in range(5):
        ps = dow_buckets[i]
        if ps:
            a = np.array(ps)
            sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
            dow_rows.append([dow_names[i], len(ps), f"{np.mean(ps):+.2f}%",
                             f"{(a>0).mean()*100:.0f}%", f"{sh:.3f}"])
        else:
            dow_rows.append([dow_names[i], 0, '—', '—', '—'])
    print(tabulate(dow_rows, headers=['Day', 'n', 'Avg', 'WR', 'Sharpe'],
                   tablefmt='simple'))

    # ── C1-E04 decline distribution ──
    print(f"\n\n{'─'*90}")
    print(f"  C1-E04 PRIOR 5D DECLINE DISTRIBUTION (C1 baseline)")
    print(f"{'─'*90}")
    dec_buckets = {'-15% to -20%': [], '-20% to -25%': [],
                   '-25% to -30%': [], '-30% to -40%': [], '<-40%': []}
    for t in c1_base:
        pnl = t.get(pnl_key_c1)
        if pnl is None: continue
        p5d = t['prior_5d_return']
        if p5d > -0.20:
            dec_buckets['-15% to -20%'].append(pnl)
        elif p5d > -0.25:
            dec_buckets['-20% to -25%'].append(pnl)
        elif p5d > -0.30:
            dec_buckets['-25% to -30%'].append(pnl)
        elif p5d > -0.40:
            dec_buckets['-30% to -40%'].append(pnl)
        else:
            dec_buckets['<-40%'].append(pnl)
    dec_rows = []
    for bkt in ['-15% to -20%', '-20% to -25%', '-25% to -30%', '-30% to -40%', '<-40%']:
        ps = dec_buckets[bkt]
        if ps:
            a = np.array(ps)
            sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
            dec_rows.append([bkt, len(ps), f"{np.mean(ps):+.2f}%",
                             f"{(a>0).mean()*100:.0f}%", f"{sh:.3f}"])
        else:
            dec_rows.append([bkt, 0, '—', '—', '—'])
    print(tabulate(dec_rows, headers=['Decline', 'n', 'Avg', 'WR', 'Sharpe'],
                   tablefmt='simple'))

    # ── C1-E07 gap distribution ──
    print(f"\n\n{'─'*90}")
    print(f"  C1-E07 GAP-UP TIER DISTRIBUTION (C1 baseline)")
    print(f"{'─'*90}")
    gap_buckets = {'3-5%': [], '5-8%': [], '8-10%': [], '10-15%': [],
                   '15-25%': [], '25-50%': []}
    for t in c1_base:
        pnl = t.get(pnl_key_c1)
        if pnl is None: continue
        g = t['gap_pct'] * 100
        if g < 5:
            gap_buckets['3-5%'].append(pnl)
        elif g < 8:
            gap_buckets['5-8%'].append(pnl)
        elif g < 10:
            gap_buckets['8-10%'].append(pnl)
        elif g < 15:
            gap_buckets['10-15%'].append(pnl)
        elif g < 25:
            gap_buckets['15-25%'].append(pnl)
        else:
            gap_buckets['25-50%'].append(pnl)
    gap_rows = []
    for bkt in ['3-5%', '5-8%', '8-10%', '10-15%', '15-25%', '25-50%']:
        ps = gap_buckets[bkt]
        if ps:
            a = np.array(ps)
            sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
            gap_rows.append([bkt, len(ps), f"{np.mean(ps):+.2f}%",
                             f"{(a>0).mean()*100:.0f}%", f"{sh:.3f}"])
        else:
            gap_rows.append([bkt, 0, '—', '—', '—'])
    print(tabulate(gap_rows, headers=['Gap Range', 'n', 'Avg', 'WR', 'Sharpe'],
                   tablefmt='simple'))

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

COLORS_EXP = {
    'BASELINE-A6': '#27ae60', 'A6-E01': '#2ecc71', 'A6-E02': '#1abc9c',
    'BASELINE-C1': '#2980b9', 'C1-E01': '#e74c3c', 'C1-E04': '#e67e22', 'C1-E07': '#9b59b6',
}

def plot_experiment_results(results, dates):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    # ── Figure 1: Equity curves — A6 experiments ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'Agent 3 — Week 1 Experiments: {dates[0]} → {dates[-1]}', fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.set_title('A6 Experiments — Equity (compounding)')
    for eid in ['BASELINE-A6', 'A6-E01', 'A6-E02']:
        r = results[eid]
        color = COLORS_EXP.get(eid, '#333')
        eq_d = [datetime.strptime(dates[0], '%Y-%m-%d')] + \
               [datetime.strptime(d, '%Y-%m-%d') for d in r['eq_dates']]
        ax.plot(eq_d, r['equity'], color=color, linewidth=2,
                label=f'{eid} (${r["final_capital"]:,.0f})')
    ax.axhline(CAPITAL, color='#999', ls='--', lw=0.8)
    oos_dt = datetime.strptime(OOS_CUTOFF, '%Y-%m-%d')
    ax.axvline(oos_dt, color='red', ls=':', lw=1, alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.set_title('C1 Experiments — Equity (compounding)')
    for eid in ['BASELINE-C1', 'C1-E01', 'C1-E04', 'C1-E07']:
        r = results[eid]
        color = COLORS_EXP.get(eid, '#333')
        eq_d = [datetime.strptime(dates[0], '%Y-%m-%d')] + \
               [datetime.strptime(d, '%Y-%m-%d') for d in r['eq_dates']]
        ax.plot(eq_d, r['equity'], color=color, linewidth=2,
                label=f'{eid} (${r["final_capital"]:,.0f})')
    ax.axhline(CAPITAL, color='#999', ls='--', lw=0.8)
    ax.axvline(oos_dt, color='red', ls=':', lw=1, alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p1 = os.path.join(BASE_DIR, 'agent3_w1_equity.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")

    # ── Figure 2: Summary bar charts — Avg PnL and Sharpe ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Week 1 Experiments — Comparison', fontsize=12, fontweight='bold')

    exp_order = ['BASELINE-A6', 'A6-E01', 'A6-E02',
                 'BASELINE-C1', 'C1-E01', 'C1-E04', 'C1-E07']
    x_labels = exp_order
    x = np.arange(len(exp_order))

    colors = [COLORS_EXP.get(eid, '#333') for eid in exp_order]

    # Avg PnL
    ax = axes[0]
    avgs = [results[eid]['metrics']['avg'] for eid in exp_order]
    bars = ax.bar(x, avgs, color=colors, alpha=0.85)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Avg PnL (%)')
    ax.set_title('Average PnL')
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{v:+.1f}', ha='center', va='bottom', fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')

    # Sharpe
    ax = axes[1]
    sharpes = [results[eid]['metrics']['sharpe'] for eid in exp_order]
    bars = ax.bar(x, sharpes, color=colors, alpha=0.85)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Sharpe')
    ax.set_title('Sharpe Ratio')
    for bar, v in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')

    # OOS avg
    ax = axes[2]
    oos_avgs = [results[eid]['oos_metrics']['avg'] for eid in exp_order]
    bars = ax.bar(x, oos_avgs, color=colors, alpha=0.85)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('OOS Avg PnL (%)')
    ax.set_title('OOS Performance')
    for bar, v in zip(bars, oos_avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{v:+.1f}', ha='center', va='bottom', fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p2 = os.path.join(BASE_DIR, 'agent3_w1_comparison.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p2}")

    # ── Figure 3: IS vs OOS per experiment ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('IS vs OOS — Week 1 Experiments', fontsize=12, fontweight='bold')

    # IS vs OOS avg
    ax = axes[0]
    is_avgs = [results[eid]['is_metrics']['avg'] for eid in exp_order]
    oos_avgs_list = [results[eid]['oos_metrics']['avg'] for eid in exp_order]
    w = 0.35
    ax.bar(x - w/2, is_avgs, w, color='#3498db', alpha=0.8, label='IS')
    ax.bar(x + w/2, oos_avgs_list, w, color='#e67e22', alpha=0.8, label='OOS')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Avg PnL (%)')
    ax.set_title('IS vs OOS — Avg PnL')
    ax.axhline(0, color='#333', lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # IS vs OOS Sharpe
    ax = axes[1]
    is_sh = [results[eid]['is_metrics']['sharpe'] for eid in exp_order]
    oos_sh = [results[eid]['oos_metrics']['sharpe'] for eid in exp_order]
    ax.bar(x - w/2, is_sh, w, color='#3498db', alpha=0.8, label='IS')
    ax.bar(x + w/2, oos_sh, w, color='#e67e22', alpha=0.8, label='OOS')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Sharpe')
    ax.set_title('IS vs OOS — Sharpe')
    ax.axhline(0, color='#333', lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p3 = os.path.join(BASE_DIR, 'agent3_w1_isoos.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p3}")

    # ── Figure 4: PnL distributions overlay ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('PnL Distributions — Week 1', fontsize=12, fontweight='bold')

    # A6 distributions
    ax = axes[0]
    ax.set_title('A6 PnL Distributions')
    for eid in ['BASELINE-A6', 'A6-E01', 'A6-E02']:
        r = results[eid]
        pnl_key = f'pnl_t{r["hold_days"]}'
        pnls = [t[pnl_key] for t in r['trades'] if t.get(pnl_key) is not None]
        if pnls:
            color = COLORS_EXP.get(eid, '#333')
            clip_lo, clip_hi = np.percentile(pnls, 2), np.percentile(pnls, 98)
            bins = np.linspace(clip_lo, clip_hi, 40)
            ax.hist(pnls, bins=bins, color=color, alpha=0.35,
                    label=f'{eid} (n={len(pnls)}, {np.mean(pnls):+.2f}%)')
            ax.axvline(np.mean(pnls), color=color, ls='--', lw=1.5)
    ax.axvline(0, color='#333', lw=1.2)
    ax.set_xlabel('PnL (%)')
    ax.legend(fontsize=8)

    # C1 distributions
    ax = axes[1]
    ax.set_title('C1 PnL Distributions')
    for eid in ['BASELINE-C1', 'C1-E01', 'C1-E04', 'C1-E07']:
        r = results[eid]
        pnl_key = f'pnl_t{r["hold_days"]}'
        pnls = [t[pnl_key] for t in r['trades'] if t.get(pnl_key) is not None]
        if pnls:
            color = COLORS_EXP.get(eid, '#333')
            clip_lo, clip_hi = np.percentile(pnls, 2), np.percentile(pnls, 98)
            bins = np.linspace(clip_lo, clip_hi, 40)
            ax.hist(pnls, bins=bins, color=color, alpha=0.3,
                    label=f'{eid} (n={len(pnls)}, {np.mean(pnls):+.2f}%)')
            ax.axvline(np.mean(pnls), color=color, ls='--', lw=1.5)
    ax.axvline(0, color='#333', lw=1.2)
    ax.set_xlabel('PnL (%)')
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p4 = os.path.join(BASE_DIR, 'agent3_w1_distributions.png')
    plt.savefig(p4, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p4}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

a6_trades, c1_trades, dates = run_backtest()

experiments = register_experiments(a6_trades, c1_trades, dates)
results = analyze_experiments(experiments, dates)
print_experiment_results(results, dates)
plot_experiment_results(results, dates)
