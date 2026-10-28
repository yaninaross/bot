# %% Agent 3 — Revised Backtest Suite (BT-1 through BT-4)
#
# BT-1a: A6 gap>=2%, open entry (no pattern gate), T+2 close
# BT-1b: A6 gap>=2%, delayed entry at 10:35 after SR confirmation, T+2 close
# BT-2:  A6 gap>=2%, 9:45 entry, intraday exit grid (10:00 -> 12:00)
# BT-3:  A6 gap>=2%, open entry, T+1 open exit (overnight capture)
# BT-4:  C1 lookback grid (2d->5d), open entry, T+3 close + hold sweep
#
# Full period: 2025-09-15 -> 2026-10-06
# Splits: IS (Sep15-Jan1), OOS (Jan1-Aug1), Post-Aug (Aug1-Oct6)
# Capital: $100,000/day non-compounding
#
# Colab-cell friendly. ~25-40 min runtime (API-bound).

import os, sys, time, re, copy, json, asyncio
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

BACKTEST_START = '2025-09-15'
BACKTEST_END   = '2026-10-06'
OOS_CUTOFF     = '2026-01-01'
BREAK_DATE     = '2026-08-01'

CAPITAL    = 100_000
TOP_N      = 5
FLOAT_MIN  = 10_000_000

A6_GAP_MIN_PCT  = 2.0     # revised: was 5.0
A6_PRICE_MIN    = 10.0
A6_HOLD_DAYS    = 2        # T+2 for BT-1a/1b
A6_PRE_FILTER_N = 20

C1_GAP_MIN_PCT    = 3.0
C1_GAP_MAX_PCT    = 50.0
C1_PRICE_MIN      = 20.0
C1_HOLD_DAYS      = 3     # T+3
MAX_HOLD_SWEEP    = 10    # for BT-4 hold period sweep

LOW_WINDOW   = 3
MAX_LOW_BAR  = 12
FALLBACK_BAR = 15
SKIP_BOUNCE_PCT = 0.15

BOOTSTRAP_N  = 1000
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
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pre-registered expectations (written before seeing results)
EXPECTATIONS = {
    'BT-1a': 'Post-Aug avg > +2%, Sharpe > 0.30',
    'BT-1b': 'BT-1b avg within 2pp of BT-1a SR subset avg',
    'BT-2':  '10:30 exit optimal, full-period Sharpe > 0.40',
    'BT-3':  'Post-Aug avg > BT-1a avg, overnight_ret > 0 both periods',
    'BT-4':  'Config_B (3d>=10%) > Config_E (5d>=15%)',
}


# =============================================================================
# CONSTANTS
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


# =============================================================================
# HELPERS
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


# =============================================================================
# ASYNC MINUTE BAR FETCH (RTH only)
# =============================================================================

async def _fetch_bars_one(session, tk, dt, sem):
    url = f"{_BASE}/v2/aggs/ticker/{tk}/range/1/minute/{dt}/{dt}"
    params = {"adjusted":"true","sort":"asc","limit":"500","apiKey":POLYGON_API_KEY}
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


# =============================================================================
# ASYNC TICKER REFERENCE FETCH
# =============================================================================

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


# =============================================================================
# PATTERN CLASSIFICATION (A6 gap-down)
# =============================================================================

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

def compute_metrics(pnl_list, label='', use_bootstrap=True):
    if not pnl_list:
        return {'label':label,'n':0,'avg':0,'med':0,'std':0,'wr':0,
                'sharpe':0,'pf':0,'ci_lo':0,'ci_hi':0,'trimmed_mean':0,
                'max_loss':0,'max_gain':0,'max_dd':0,'total_usd_nc':0,
                'active_days':0,'tpd':0}
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

    if use_bootstrap and n >= 2:
        ci_lo, ci_hi = bootstrap_ci(pnl_list)
    else:
        se = std / np.sqrt(n) if n > 0 else 0
        ci_lo = avg - 1.96 * se
        ci_hi = avg + 1.96 * se

    return {'label':label,'n':n,'avg':round(avg,3),'med':round(med,3),'std':round(std,3),
            'wr':round(wr,4),'sharpe':round(sharpe,4),'pf':round(pf,3),
            'ci_lo':round(ci_lo,3),'ci_hi':round(ci_hi,3),
            'trimmed_mean':round(tm,3),'max_loss':round(float(a.min()),3),
            'max_gain':round(float(a.max()),3)}

def _fmt(m):
    if m['n'] == 0: return "  n=0 (no trades)"
    ci_pass = 'YES' if m['ci_lo'] > 0 else 'NO'
    return (f"  n={m['n']}  avg={m['avg']:+.2f}%  Sharpe={m['sharpe']:.3f}  "
            f"WR={m['wr']*100:.0f}%  CI=[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}] ({ci_pass})  "
            f"trimmed={m['trimmed_mean']:+.2f}%")

def compute_nc_dollar(trades, pnl_key='pnl'):
    daily = defaultdict(list)
    for t in trades:
        p = t.get(pnl_key)
        if p is not None:
            daily[t['scan_date']].append(p)
    total = 0.0
    for dt, pnls in daily.items():
        n = len(pnls)
        pos_size = CAPITAL / n
        total += sum(pos_size * (r / 100) for r in pnls)
    return round(total, 2)

def compute_max_dd(trades, dates, pnl_key='pnl'):
    daily_pnl = defaultdict(list)
    for t in trades:
        p = t.get(pnl_key)
        if p is not None:
            daily_pnl[t['scan_date']].append(p)
    equity = [CAPITAL]
    for dt in sorted(dates):
        dp = daily_pnl.get(dt, [])
        if dp:
            n = len(dp)
            pos_size = equity[-1] / n
            day_dollar = sum(pos_size * (r / 100) for r in dp)
            equity.append(equity[-1] + day_dollar)
    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    return round(float(dd.min()), 2)


# =============================================================================
# PERIOD SPLIT
# =============================================================================

def split_by_period(trades, pnl_key='pnl'):
    periods = {
        'full':     lambda dt: True,
        'is':       lambda dt: dt < OOS_CUTOFF,
        'oos':      lambda dt: OOS_CUTOFF <= dt < BREAK_DATE,
        'post_aug': lambda dt: dt >= BREAK_DATE,
    }
    results = {}
    for period_name, filt in periods.items():
        pnls = [t[pnl_key] for t in trades if filt(t['scan_date']) and t.get(pnl_key) is not None]
        m = compute_metrics(pnls, period_name)
        results[period_name] = m
    return results

def print_period_table(splits, bt_id):
    print(f"\n  {'Period':<12s}  {'n':>5s}  {'avg':>8s}  {'Sharpe':>8s}  {'WR':>6s}  {'CI 95%':>22s}  {'CI>0':>5s}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*22}  {'-'*5}")
    for p in ['full', 'is', 'oos', 'post_aug']:
        m = splits[p]
        ci_pass = 'YES' if m['ci_lo'] > 0 else 'NO'
        lbl = {'full':'Full','is':'IS (Sep-Jan)','oos':'OOS (Jan-Aug)','post_aug':'Post-Aug'}[p]
        print(f"  {lbl:<12s}  {m['n']:>5d}  {m['avg']:>+7.2f}%  {m['sharpe']:>8.3f}  "
              f"{m['wr']*100:>5.0f}%  [{m['ci_lo']:>+7.2f}, {m['ci_hi']:>+7.2f}]  {ci_pass:>5s}")

def monthly_breakdown(trades, pnl_key='pnl'):
    monthly = defaultdict(list)
    for t in trades:
        p = t.get(pnl_key)
        if p is not None:
            monthly[t['scan_date'][:7]].append(p)
    rows = []
    for mo in sorted(monthly.keys()):
        pnls = monthly[mo]
        m = compute_metrics(pnls, use_bootstrap=False)
        rows.append({'month': mo, 'n': m['n'], 'avg': m['avg'], 'sharpe': m['sharpe']})
    return rows


# =============================================================================
# =============================================================================
#
#  DATA PIPELINE
#
# =============================================================================
# =============================================================================

def run_data_pipeline():
    t_total = time.time()

    print(f"\n{'='*80}")
    print(f"  AGENT 3 — REVISED BACKTEST SUITE")
    print(f"  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"  Universe: _FLOAT ({len(_FLOAT_SET)} tickers with float >= 10M)")
    print(f"  A6: gap-down >= {A6_GAP_MIN_PCT}%, open/delayed/intraday entry variants")
    print(f"  C1: 5 lookback configs, T+3 + hold sweep T+1..T+10")
    print(f"  Capital: ${CAPITAL:,.0f}/day, non-compounding")
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

    # Lookback: 12 trading days before start (for C1 5d decline + margin)
    lookback_start = _nth_prior_td(BACKTEST_START, 12)
    lb = lookback_start
    lookback_dates = []
    while lb < datetime.strptime(BACKTEST_START, '%Y-%m-%d').date():
        if _is_td(lb): lookback_dates.append(lb.strftime('%Y-%m-%d'))
        lb += timedelta(days=1)

    # Phase 1: Fetch grouped daily
    t1 = time.time()
    all_fetch_dates = sorted(set(lookback_dates + dates))
    print(f"\n  Phase 1: Fetching grouped daily for {len(all_fetch_dates)} dates...")
    for i, dt in enumerate(all_fetch_dates):
        fetch_grouped_daily(dt)
        if (i+1) % 30 == 0 or i == len(all_fetch_dates) - 1:
            print(f"    [{i+1:>3}/{len(all_fetch_dates)}] {dt}  |  {time.time()-t_total:.0f}s")

    # Outcome dates (up to T+10 for BT-4 hold sweep)
    outcome_dates = set()
    for dt in dates:
        for nd in _next_n_td(dt, MAX_HOLD_SWEEP):
            outcome_dates.add(nd.strftime('%Y-%m-%d'))
    print(f"  Fetching {len(outcome_dates)} outcome dates...")
    for od in sorted(outcome_dates):
        if od not in _gd_cache:
            try: fetch_grouped_daily(od)
            except: pass
    print(f"  Phase 1: {time.time()-t1:.0f}s")

    # Phase 2: Scan A6 + C1 candidates
    t2 = time.time()
    print(f"\n  Phase 2: Scanning candidates...")
    a6_candidates = {}
    c1_candidates = {}
    total_a6, total_c1 = 0, 0
    gap_min_a6 = A6_GAP_MIN_PCT / 100.0

    for i, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = _gd_cache.get(prior_str, {})
        today = _gd_cache.get(dt, {})

        # A6: Gap-DOWN candidates (>= 2%)
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
            a6_day.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_o, 'scan_date': dt,
                'volume_today': today[tk]['v'],
                'day_close': today[tk]['c'],
            })
        a6_day.sort(key=lambda x: x['gap_pct'])
        a6_candidates[dt] = a6_day
        total_a6 += len(a6_day)

        # C1: Gap-UP + prior decline (broad: 2d/3d/4d/5d)
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
            if gap < C1_GAP_MIN_PCT / 100.0: continue
            if gap > C1_GAP_MAX_PCT / 100.0: continue

            declines = {}
            for nd_lbl, nd_n in [('2d', 1), ('3d', 2), ('4d', 3), ('5d', 4)]:
                try:
                    close_nd_ago_date = _nth_prior_td(prior_str, nd_n).strftime('%Y-%m-%d')
                    close_nd_ago_data = _gd_cache.get(close_nd_ago_date, {}).get(tk)
                    if close_nd_ago_data and close_nd_ago_data['c'] > 0:
                        declines[nd_lbl] = (prev_c - close_nd_ago_data['c']) / close_nd_ago_data['c']
                except ValueError:
                    pass

            if '5d' not in declines:
                continue

            c1_day.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_o, 'scan_date': dt,
                'volume_today': today[tk]['v'],
                'prior_2d_return': declines.get('2d'),
                'prior_3d_return': declines.get('3d'),
                'prior_4d_return': declines.get('4d'),
                'prior_5d_return': declines['5d'],
                'day_of_week': datetime.strptime(dt, '%Y-%m-%d').strftime('%A'),
            })
        c1_day.sort(key=lambda x: -x['gap_pct'])
        c1_candidates[dt] = c1_day
        total_c1 += len(c1_day)

        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  A6={total_a6}, C1={total_c1}  |  {time.time()-t_total:.0f}s")
    print(f"  Phase 2: {time.time()-t2:.0f}s")

    # Phase 3: Fetch minute bars for A6 candidates (top 20/day)
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
        if done % 500 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    print(f"  -> Got bars for {len(all_bars)}/{len(a6_pairs)} pairs")
    print(f"  Phase 3: {time.time()-t3:.0f}s")

    # Phase 4: Fetch ticker references
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
    print(f"  Phase 4: {time.time()-t4:.0f}s")

    # Phase 5: Classify patterns
    t5 = time.time()
    pattern_cache = {}
    for key, rth_bars in all_bars.items():
        pattern_cache[key] = classify_gap_down_pattern(rth_bars)

    pat_counts = defaultdict(int)
    for p in pattern_cache.values():
        pat_counts[p] += 1
    print(f"\n  Phase 5: Pattern classification ({len(pattern_cache)} candidates)")
    for pat in ['sustained_rise', 'rise_then_drop', 'mild_rise', 'continued_drop', 'no_data']:
        cnt = pat_counts.get(pat, 0)
        pct = cnt / len(pattern_cache) * 100 if pattern_cache else 0
        print(f"    {pat:25s}  {cnt:>5d}  ({pct:.1f}%)")
    print(f"  Phase 5: {time.time()-t5:.0f}s")

    elapsed = time.time() - t_total
    print(f"\n  DATA PIPELINE COMPLETE: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return {
        'dates': dates,
        'a6_candidates': a6_candidates,
        'c1_candidates': c1_candidates,
        'all_bars': all_bars,
        'pattern_cache': pattern_cache,
    }


# =============================================================================
# =============================================================================
#
#  BT-1a: A6 Gap>=2%, Open Entry, No Pattern Gate, T+2 Close
#
# =============================================================================
# =============================================================================

def run_bt1a(data):
    print(f"\n\n{'='*80}")
    print(f"  BT-1a — A6 REVISED: Gap>=2% + Open Entry + T+2 Close")
    print(f"  NO pattern gate at entry. Pattern classified post-hoc.")
    print(f"  Expected: Post-Aug avg > +2%, Sharpe > 0.30")
    print(f"{'='*80}")

    trades = []
    for dt in data['dates']:
        cands = data['a6_candidates'].get(dt, [])
        qualified = []
        for c in cands[:A6_PRE_FILTER_N]:
            tk = c['ticker']
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS': continue

            ep = c['open_price']
            t2 = _next_n_td(dt, 2)
            if len(t2) < 2: continue
            t2_str = t2[1].strftime('%Y-%m-%d')
            t2_data = _gd_cache.get(t2_str, {}).get(tk, {})
            if not t2_data.get('c'): continue
            exit_price = t2_data['c']
            pnl = round((exit_price - ep) / ep * 100, 3)

            pattern = data['pattern_cache'].get((tk, dt), 'no_data')

            qualified.append({
                'ticker': tk, 'scan_date': dt, 'gap_pct': c['gap_pct'],
                'open_price': ep, 'entry_price': ep, 'exit_price': exit_price,
                'entry_type': 'open', 'entry_time': '09:30',
                'exit_date': t2_str, 'pnl': pnl, 'pattern': pattern,
                'strategy': 'BT-1a',
            })

        for tc in qualified[:TOP_N]:
            trades.append(tc)

    print(f"\n  Total trades: {len(trades)}")

    # Period splits
    splits = split_by_period(trades)
    print_period_table(splits, 'BT-1a')

    # Pattern breakdown (post-hoc)
    print(f"\n  Pattern breakdown (post-hoc):")
    print(f"  {'Pattern':>20s}  {'n':>5s}  {'avg':>8s}  {'Sharpe':>8s}  {'WR':>6s}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*6}")
    for pat in ['sustained_rise', 'rise_then_drop', 'mild_rise', 'continued_drop', 'no_data']:
        pnls = [t['pnl'] for t in trades if t['pattern'] == pat]
        if pnls:
            m = compute_metrics(pnls, use_bootstrap=False)
            print(f"  {pat:>20s}  {m['n']:>5d}  {m['avg']:>+7.2f}%  {m['sharpe']:>8.3f}  {m['wr']*100:>5.0f}%")
        else:
            print(f"  {pat:>20s}  {'0':>5s}  {'—':>8s}  {'—':>8s}  {'—':>6s}")

    # Gap tier breakdown
    print(f"\n  Gap tier breakdown:")
    print(f"  {'Tier':>12s}  {'n':>5s}  {'avg':>8s}  {'Sharpe':>8s}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*8}  {'-'*8}")
    tiers = [('2-5%', -0.05, -0.02), ('5-8%', -0.08, -0.05),
             ('8-12%', -0.12, -0.08), ('>12%', -1.0, -0.12)]
    for lbl, lo, hi in tiers:
        pnls = [t['pnl'] for t in trades if lo <= t['gap_pct'] < hi]
        if pnls:
            m = compute_metrics(pnls, use_bootstrap=False)
            print(f"  {lbl:>12s}  {m['n']:>5d}  {m['avg']:>+7.2f}%  {m['sharpe']:>8.3f}")
        else:
            print(f"  {lbl:>12s}  {'0':>5s}  {'—':>8s}  {'—':>8s}")

    # Monthly
    mo = monthly_breakdown(trades)
    print(f"\n  Monthly breakdown:")
    print(f"  {'Month':>8s}  {'n':>5s}  {'avg':>8s}  {'Sharpe':>8s}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*8}")
    for r in mo:
        print(f"  {r['month']:>8s}  {r['n']:>5d}  {r['avg']:>+7.2f}%  {r['sharpe']:>8.3f}")

    return {'bt_id': 'BT-1a', 'trades': trades, 'splits': splits,
            'monthly': mo, 'n': len(trades)}


# =============================================================================
#
#  BT-1b: A6 Gap>=2%, Delayed Entry at 10:35 After SR Confirmation, T+2
#
# =============================================================================

def run_bt1b(data, bt1a_result):
    print(f"\n\n{'='*80}")
    print(f"  BT-1b — A6 REVISED: Gap>=2% + Delayed Entry 10:35 + T+2 Close")
    print(f"  Wait for sustained_rise confirmation, enter at bar 65 close.")
    print(f"  Expected: BT-1b avg within 2pp of BT-1a SR subset avg")
    print(f"{'='*80}")

    # BT-1a SR subset for comparison
    bt1a_sr = [t for t in bt1a_result['trades'] if t['pattern'] == 'sustained_rise']
    bt1a_sr_pnls = [t['pnl'] for t in bt1a_sr]
    bt1a_sr_avg = float(np.mean(bt1a_sr_pnls)) if bt1a_sr_pnls else 0

    trades = []
    skipped_no_sr = 0
    skipped_no_bars = 0

    for dt in data['dates']:
        cands = data['a6_candidates'].get(dt, [])
        qualified = []
        for c in cands[:A6_PRE_FILTER_N]:
            tk = c['ticker']
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS': continue

            key = (tk, dt)
            rth_bars = data['all_bars'].get(key)
            if not rth_bars or len(rth_bars) < 66:
                skipped_no_bars += 1
                continue

            pattern = data['pattern_cache'].get(key, 'no_data')
            if pattern != 'sustained_rise':
                skipped_no_sr += 1
                continue

            ep = rth_bars[65][1]['c']
            if ep <= 0: continue

            t2 = _next_n_td(dt, 2)
            if len(t2) < 2: continue
            t2_str = t2[1].strftime('%Y-%m-%d')
            t2_data = _gd_cache.get(t2_str, {}).get(tk, {})
            if not t2_data.get('c'): continue
            exit_price = t2_data['c']
            pnl = round((exit_price - ep) / ep * 100, 3)

            qualified.append({
                'ticker': tk, 'scan_date': dt, 'gap_pct': c['gap_pct'],
                'open_price': c['open_price'], 'entry_price': ep,
                'exit_price': exit_price,
                'entry_type': 'delayed_confirmed', 'entry_time': '10:35',
                'exit_date': t2_str, 'pnl': pnl, 'pattern': 'sustained_rise',
                'strategy': 'BT-1b',
            })

        for tc in qualified[:TOP_N]:
            trades.append(tc)

    print(f"\n  Total trades: {len(trades)}")
    print(f"  Skipped (no SR): {skipped_no_sr}, Skipped (no bars): {skipped_no_bars}")

    splits = split_by_period(trades)
    print_period_table(splits, 'BT-1b')

    # Delay cost comparison
    bt1b_avg = splits['full']['avg']
    delay_cost = bt1a_sr_avg - bt1b_avg
    print(f"\n  BT-1a SR subset: n={len(bt1a_sr)}, avg={bt1a_sr_avg:+.2f}%")
    print(f"  BT-1b (delayed): n={len(trades)}, avg={bt1b_avg:+.2f}%")
    print(f"  Delay cost: {delay_cost:+.2f}pp")
    print(f"  n lost to non-SR filter: {skipped_no_sr}")

    acceptable = abs(delay_cost) <= 2.0
    print(f"  Delay cost <= 2pp: {'YES' if acceptable else 'NO'}")

    mo = monthly_breakdown(trades)

    return {'bt_id': 'BT-1b', 'trades': trades, 'splits': splits,
            'monthly': mo, 'n': len(trades),
            'bt1a_sr_avg': round(bt1a_sr_avg, 3),
            'delay_cost': round(delay_cost, 3),
            'delay_acceptable': acceptable}


# =============================================================================
#
#  BT-2: A6 Gap>=2%, 9:45 Entry, Intraday Exit Grid
#
# =============================================================================

def run_bt2(data):
    print(f"\n\n{'='*80}")
    print(f"  BT-2 — A6 REVISED: Gap>=2% + 9:45 Entry + Intraday Exit Grid")
    print(f"  No pattern gate. All exits same-day.")
    print(f"  Expected: 10:30 exit optimal, full-period Sharpe > 0.40")
    print(f"{'='*80}")

    EXIT_GRID = [
        (30,  '10:00'),
        (45,  '10:15'),
        (60,  '10:30'),
        (75,  '10:45'),
        (90,  '11:00'),
        (120, '11:30'),
        (150, '12:00'),
    ]

    trades = []
    for dt in data['dates']:
        cands = data['a6_candidates'].get(dt, [])
        qualified = []
        for c in cands[:A6_PRE_FILTER_N]:
            tk = c['ticker']
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS': continue

            key = (tk, dt)
            rth_bars = data['all_bars'].get(key)
            if not rth_bars or len(rth_bars) < 16:
                continue

            ep = rth_bars[15][1]['c']  # 9:45 ET close
            if ep <= 0: continue

            pattern = data['pattern_cache'].get(key, 'no_data')

            tc = {
                'ticker': tk, 'scan_date': dt, 'gap_pct': c['gap_pct'],
                'open_price': c['open_price'], 'entry_price': ep,
                'entry_type': 'delayed_open', 'entry_time': '09:45',
                'pattern': pattern, 'strategy': 'BT-2',
            }

            has_any_exit = False
            for bar_idx, exit_label in EXIT_GRID:
                if bar_idx < len(rth_bars):
                    exit_p = rth_bars[bar_idx][1]['c']
                    tc[f'pnl_{exit_label}'] = round((exit_p - ep) / ep * 100, 3)
                    has_any_exit = True

            if has_any_exit:
                qualified.append(tc)

        for tc in qualified[:TOP_N]:
            trades.append(tc)

    print(f"\n  Total trades: {len(trades)}")

    # Report for each exit time
    best_sharpe = -999
    best_exit = None
    print(f"\n  Exit time grid (full period):")
    print(f"  {'Exit':>8s}  {'n':>5s}  {'avg':>8s}  {'Sharpe':>8s}  {'WR':>6s}  {'CI 95%':>22s}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*22}")

    exit_results = {}
    for _, exit_label in EXIT_GRID:
        pnl_key = f'pnl_{exit_label}'
        pnls = [t[pnl_key] for t in trades if pnl_key in t]
        m = compute_metrics(pnls)
        ci_pass = 'YES' if m['ci_lo'] > 0 else 'NO'
        print(f"  {exit_label:>8s}  {m['n']:>5d}  {m['avg']:>+7.2f}%  {m['sharpe']:>8.3f}  "
              f"{m['wr']*100:>5.0f}%  [{m['ci_lo']:>+7.2f}, {m['ci_hi']:>+7.2f}]")
        exit_results[exit_label] = m
        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_exit = exit_label

    print(f"\n  >> Optimal exit: {best_exit} (Sharpe={best_sharpe:.3f})")

    # Period splits at optimal exit
    pnl_key_opt = f'pnl_{best_exit}'
    for t in trades:
        t['pnl'] = t.get(pnl_key_opt)
    splits = split_by_period(trades)
    print(f"\n  Period splits at optimal exit ({best_exit}):")
    print_period_table(splits, 'BT-2')

    # Pattern breakdown at optimal
    print(f"\n  Pattern breakdown at optimal exit ({best_exit}):")
    for pat in ['sustained_rise', 'rise_then_drop', 'mild_rise', 'continued_drop', 'no_data']:
        pnls = [t.get(pnl_key_opt) for t in trades
                if t['pattern'] == pat and t.get(pnl_key_opt) is not None]
        if pnls:
            m = compute_metrics(pnls, use_bootstrap=False)
            print(f"    {pat:>20s}  n={m['n']:<5d}  avg={m['avg']:>+7.2f}%  Sharpe={m['sharpe']:.3f}")

    mo = monthly_breakdown(trades)

    return {'bt_id': 'BT-2', 'trades': trades, 'splits': splits,
            'exit_results': exit_results, 'optimal_exit': best_exit,
            'optimal_sharpe': round(best_sharpe, 4), 'monthly': mo, 'n': len(trades)}


# =============================================================================
#
#  BT-3: A6 Gap>=2%, Open Entry, T+1 Open Exit (Overnight Capture)
#
# =============================================================================

def run_bt3(data):
    print(f"\n\n{'='*80}")
    print(f"  BT-3 — A6 REVISED: Gap>=2% + Open Entry + T+1 Open Exit")
    print(f"  Capture overnight institutional accumulation.")
    print(f"  Expected: Post-Aug avg > BT-1a avg, overnight_ret > 0 both periods")
    print(f"{'='*80}")

    trades = []
    for dt in data['dates']:
        cands = data['a6_candidates'].get(dt, [])
        qualified = []
        for c in cands[:A6_PRE_FILTER_N]:
            tk = c['ticker']
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS': continue

            ep = c['open_price']  # T+0 open
            t0_close = c.get('day_close')

            t1 = _next_n_td(dt, 1)
            if not t1: continue
            t1_str = t1[0].strftime('%Y-%m-%d')
            t1_data = _gd_cache.get(t1_str, {}).get(tk, {})
            if not t1_data.get('o'): continue
            exit_price = t1_data['o']  # T+1 open

            total_ret = round((exit_price - ep) / ep * 100, 3)

            overnight_ret = None
            intraday_ret = None
            if t0_close and t0_close > 0:
                intraday_ret = round((t0_close - ep) / ep * 100, 3)
                overnight_ret = round((exit_price - t0_close) / t0_close * 100, 3)

            pattern = data['pattern_cache'].get((tk, dt), 'no_data')

            qualified.append({
                'ticker': tk, 'scan_date': dt, 'gap_pct': c['gap_pct'],
                'open_price': ep, 'entry_price': ep, 'exit_price': exit_price,
                'entry_type': 'open', 'entry_time': '09:30',
                'exit_date': t1_str, 'pnl': total_ret,
                'overnight_ret': overnight_ret, 'intraday_ret': intraday_ret,
                'pattern': pattern, 'strategy': 'BT-3',
            })

        for tc in qualified[:TOP_N]:
            trades.append(tc)

    print(f"\n  Total trades: {len(trades)}")

    splits = split_by_period(trades)
    print_period_table(splits, 'BT-3')

    # Decomposition
    print(f"\n  Overnight vs Intraday Decomposition:")
    print(f"  {'Period':>12s}  {'n':>5s}  {'avg_total':>10s}  {'avg_overnight':>15s}  {'avg_intraday':>14s}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*10}  {'-'*15}  {'-'*14}")

    decomp = {}
    for period_name, filt in [('Full', lambda dt: True),
                               ('IS', lambda dt: dt < OOS_CUTOFF),
                               ('OOS', lambda dt: OOS_CUTOFF <= dt < BREAK_DATE),
                               ('Post-Aug', lambda dt: dt >= BREAK_DATE)]:
        period_trades = [t for t in trades if filt(t['scan_date'])]
        totals = [t['pnl'] for t in period_trades if t['pnl'] is not None]
        overnights = [t['overnight_ret'] for t in period_trades if t['overnight_ret'] is not None]
        intradays = [t['intraday_ret'] for t in period_trades if t['intraday_ret'] is not None]
        n = len(totals)
        avg_t = float(np.mean(totals)) if totals else 0
        avg_o = float(np.mean(overnights)) if overnights else 0
        avg_i = float(np.mean(intradays)) if intradays else 0
        decomp[period_name] = {'n': n, 'total': round(avg_t, 3),
                                'overnight': round(avg_o, 3), 'intraday': round(avg_i, 3)}
        print(f"  {period_name:>12s}  {n:>5d}  {avg_t:>+9.2f}%  {avg_o:>+14.2f}%  {avg_i:>+13.2f}%")

    # Validation: overnight_ret > 0 in both pre-Aug and post-Aug
    pre_aug_o = decomp.get('IS', {}).get('overnight', 0) + decomp.get('OOS', {}).get('overnight', 0)
    post_aug_o = decomp.get('Post-Aug', {}).get('overnight', 0)
    print(f"\n  Overnight > 0 pre-Aug: {'YES' if pre_aug_o > 0 else 'NO'} ({pre_aug_o/2:+.2f}% avg)")
    print(f"  Overnight > 0 post-Aug: {'YES' if post_aug_o > 0 else 'NO'} ({post_aug_o:+.2f}%)")

    mo = monthly_breakdown(trades)

    return {'bt_id': 'BT-3', 'trades': trades, 'splits': splits,
            'decomposition': decomp, 'monthly': mo, 'n': len(trades)}


# =============================================================================
#
#  BT-4: C1 Lookback Grid (5 configs) + Hold Period Sweep
#
# =============================================================================

def run_bt4(data):
    print(f"\n\n{'='*80}")
    print(f"  BT-4 — C1 REVISED: Lookback Grid + Hold Period Sweep")
    print(f"  5 configs: 2d>=7%, 3d>=10%, 3d>=12%, 4d>=12%, 5d>=15%")
    print(f"  Expected: Config_B (3d>=10%) > Config_E (5d>=15%)")
    print(f"{'='*80}")

    CONFIGS = [
        ('A', '2d', 'prior_2d_return', 0.07),
        ('B', '3d', 'prior_3d_return', 0.10),
        ('C', '3d', 'prior_3d_return', 0.12),
        ('D', '4d', 'prior_4d_return', 0.12),
        ('E', '5d', 'prior_5d_return', 0.15),
    ]

    config_results = {}
    all_config_trades = {}

    for cfg_id, lookback_lbl, decline_key, decline_min in CONFIGS:
        trades = []
        for dt in data['dates']:
            cands = data['c1_candidates'].get(dt, [])
            qualified = []
            for c in cands:
                tk = c['ticker']
                at = _ticker_ref.get(tk, {}).get('type', 'UNK')
                if at != 'CS': continue

                decline_val = c.get(decline_key)
                if decline_val is None: continue
                if decline_val > -decline_min: continue

                ep = c['open_price']

                # Compute T+1 through T+10 PnL for the hold sweep (Config_B only)
                # and T+3 for all configs
                tc = {
                    'ticker': tk, 'scan_date': dt, 'gap_pct': c['gap_pct'],
                    'open_price': ep, 'entry_price': ep,
                    'entry_type': 'open', 'entry_time': '09:30',
                    'pattern': 'c1_reversal', 'strategy': f'BT-4-{cfg_id}',
                    'decline_val': decline_val, 'day_of_week': c.get('day_of_week', ''),
                }

                if cfg_id == 'B':
                    # Full hold sweep T+1..T+10
                    hold_dates = _next_n_td(dt, MAX_HOLD_SWEEP)
                    has_primary = False
                    for h in range(1, MAX_HOLD_SWEEP + 1):
                        if h <= len(hold_dates):
                            hd_str = hold_dates[h-1].strftime('%Y-%m-%d')
                            hd_data = _gd_cache.get(hd_str, {}).get(tk, {})
                            if hd_data.get('c'):
                                tc[f'pnl_t{h}'] = round((hd_data['c'] - ep) / ep * 100, 3)
                                if h == C1_HOLD_DAYS:
                                    has_primary = True
                    if not has_primary:
                        continue
                    tc['pnl'] = tc.get(f'pnl_t{C1_HOLD_DAYS}')
                else:
                    # Just T+3
                    hold_dates = _next_n_td(dt, C1_HOLD_DAYS)
                    if len(hold_dates) < C1_HOLD_DAYS: continue
                    hd_str = hold_dates[C1_HOLD_DAYS - 1].strftime('%Y-%m-%d')
                    hd_data = _gd_cache.get(hd_str, {}).get(tk, {})
                    if not hd_data.get('c'): continue
                    tc['pnl'] = round((hd_data['c'] - ep) / ep * 100, 3)
                    tc['exit_date'] = hd_str

                qualified.append(tc)

            qualified.sort(key=lambda x: -x['gap_pct'])
            for tc in qualified[:TOP_N]:
                trades.append(tc)

        all_config_trades[cfg_id] = trades
        splits = split_by_period(trades)
        config_results[cfg_id] = {
            'config': f'{lookback_lbl}>={decline_min*100:.0f}%',
            'trades': trades, 'splits': splits, 'n': len(trades),
        }

        print(f"\n  Config {cfg_id} ({lookback_lbl} >= {decline_min*100:.0f}%):")
        print_period_table(splits, f'BT-4-{cfg_id}')

    # Comparison table
    print(f"\n\n  {'─'*80}")
    print(f"  CONFIG COMPARISON (Full Period)")
    print(f"  {'─'*80}")
    print(f"  {'Config':>8s}  {'Lookback':>12s}  {'n':>5s}  {'avg':>8s}  {'Sharpe':>8s}  {'WR':>6s}  {'CI 95%':>22s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*22}")
    for cfg_id, _, _, _ in CONFIGS:
        cr = config_results[cfg_id]
        m = cr['splits']['full']
        print(f"  {cfg_id:>8s}  {cr['config']:>12s}  {m['n']:>5d}  {m['avg']:>+7.2f}%  "
              f"{m['sharpe']:>8.3f}  {m['wr']*100:>5.0f}%  [{m['ci_lo']:>+7.2f}, {m['ci_hi']:>+7.2f}]")

    # Config_B additional analysis
    print(f"\n\n  {'─'*80}")
    print(f"  CONFIG B ADDITIONAL ANALYSIS (3d >= 10%)")
    print(f"  {'─'*80}")

    b_trades = all_config_trades['B']

    # Day-of-week
    print(f"\n  Day-of-week breakdown:")
    print(f"  {'Day':>12s}  {'n':>5s}  {'avg':>8s}  {'Sharpe':>8s}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*8}  {'-'*8}")
    for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        pnls = [t['pnl'] for t in b_trades if t['day_of_week'] == dow and t.get('pnl') is not None]
        if pnls:
            m = compute_metrics(pnls, use_bootstrap=False)
            print(f"  {dow:>12s}  {m['n']:>5d}  {m['avg']:>+7.2f}%  {m['sharpe']:>8.3f}")
        else:
            print(f"  {dow:>12s}  {'0':>5s}  {'—':>8s}  {'—':>8s}")

    # Gap magnitude
    print(f"\n  Gap magnitude breakdown:")
    print(f"  {'Bucket':>12s}  {'n':>5s}  {'avg':>8s}  {'Sharpe':>8s}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*8}  {'-'*8}")
    for lbl, lo, hi in [('3-5%', 0.03, 0.05), ('5-8%', 0.05, 0.08),
                         ('8-12%', 0.08, 0.12), ('>12%', 0.12, 5.0)]:
        pnls = [t['pnl'] for t in b_trades if lo <= t['gap_pct'] < hi and t.get('pnl') is not None]
        if pnls:
            m = compute_metrics(pnls, use_bootstrap=False)
            print(f"  {lbl:>12s}  {m['n']:>5d}  {m['avg']:>+7.2f}%  {m['sharpe']:>8.3f}")
        else:
            print(f"  {lbl:>12s}  {'0':>5s}  {'—':>8s}  {'—':>8s}")

    # Hold period sweep
    print(f"\n  Hold period sweep (Config B):")
    print(f"  {'Hold':>8s}  {'n':>5s}  {'avg':>8s}  {'Sharpe':>8s}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*8}")
    hold_sweep = {}
    for h in range(1, MAX_HOLD_SWEEP + 1):
        pnl_key = f'pnl_t{h}'
        pnls = [t[pnl_key] for t in b_trades if t.get(pnl_key) is not None]
        if pnls:
            m = compute_metrics(pnls, use_bootstrap=False)
            hold_sweep[h] = m
            marker = ' <<<' if h == C1_HOLD_DAYS else ''
            print(f"  {'T+'+str(h):>8s}  {m['n']:>5d}  {m['avg']:>+7.2f}%  {m['sharpe']:>8.3f}{marker}")

    mo = monthly_breakdown(b_trades)

    return {'bt_id': 'BT-4', 'config_results': config_results,
            'all_config_trades': all_config_trades,
            'hold_sweep': hold_sweep, 'monthly_b': mo}


# =============================================================================
# =============================================================================
#
#  COMPARISON TABLE
#
# =============================================================================
# =============================================================================

def run_comparison(bt1a, bt1b, bt2, bt3, bt4):
    print(f"\n\n{'='*110}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*110}")

    bt4_b = bt4['config_results']['B']

    rows_all = [
        ('BT-1a', 'A6 Open T+2',     bt1a['splits']),
        ('BT-1b', 'A6 Delayed T+2',  bt1b['splits']),
        ('BT-2',  f"A6 Intra {bt2['optimal_exit']}", bt2['splits']),
        ('BT-3',  'A6 T+1 Open',     bt3['splits']),
        ('BT-4B', 'C1 3d>=10% T+3',  bt4_b['splits']),
    ]

    for period in ['full', 'is', 'oos', 'post_aug']:
        lbl = {'full':'FULL PERIOD','is':'IN-SAMPLE (Sep15-Jan1)',
               'oos':'OOS (Jan1-Aug1)','post_aug':'POST-AUG (Aug1-Oct6)'}[period]
        print(f"\n  {lbl}")
        print(f"  {'BT':>8s}  {'Description':>20s}  {'n':>5s}  {'avg':>8s}  {'Sharpe':>8s}  "
              f"{'WR':>6s}  {'CI 95%':>22s}  {'CI>0':>5s}")
        print(f"  {'-'*8}  {'-'*20}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*22}  {'-'*5}")
        for bt_id, desc, splits in rows_all:
            m = splits[period]
            ci_pass = 'YES' if m['ci_lo'] > 0 else 'NO'
            print(f"  {bt_id:>8s}  {desc:>20s}  {m['n']:>5d}  {m['avg']:>+7.2f}%  {m['sharpe']:>8.3f}  "
                  f"{m['wr']*100:>5.0f}%  [{m['ci_lo']:>+7.2f}, {m['ci_hi']:>+7.2f}]  {ci_pass:>5s}")

    # Best A6 selection
    print(f"\n\n  {'─'*80}")
    print(f"  BEST A6 EXIT MECHANISM SELECTION")
    print(f"  {'─'*80}")
    print(f"  Criteria: 1) Highest full-period Sharpe  2) CI>0 pre+post  3) Largest post-Aug n")

    a6_candidates_list = [
        ('BT-1a', bt1a['splits']),
        ('BT-1b', bt1b['splits']),
        ('BT-2',  bt2['splits']),
        ('BT-3',  bt3['splits']),
    ]

    print(f"\n  {'BT':>8s}  {'Full Sharpe':>12s}  {'Pre CI>0':>10s}  {'Post CI>0':>10s}  {'Post n':>8s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")
    for bt_id, splits in a6_candidates_list:
        full_sh = splits['full']['sharpe']
        pre_ci = splits['oos']['ci_lo'] > 0 if splits['oos']['n'] > 0 else False
        post_ci = splits['post_aug']['ci_lo'] > 0 if splits['post_aug']['n'] > 0 else False
        post_n = splits['post_aug']['n']
        print(f"  {bt_id:>8s}  {full_sh:>12.3f}  {'YES' if pre_ci else 'NO':>10s}  "
              f"{'YES' if post_ci else 'NO':>10s}  {post_n:>8d}")


# =============================================================================
# VALIDATION GATES
# =============================================================================

def check_validation_gates(bt1a, bt1b, bt2, bt3, bt4):
    print(f"\n\n{'='*80}")
    print(f"  VALIDATION GATES")
    print(f"{'='*80}")

    gates = []

    # BT-1
    pa = bt1a['splits']['post_aug']
    g1 = pa['ci_lo'] > 0
    g2 = pa['sharpe'] >= 0.15
    g3 = pa['n'] >= 20
    gates.append(('BT-1 Post-Aug CI > 0', g1))
    gates.append(('BT-1 Post-Aug Sharpe >= 0.15', g2))
    gates.append(('BT-1 Post-Aug n >= 20', g3))
    gates.append(('BT-1 delay cost <= 2pp', bt1b.get('delay_acceptable', False)))

    # BT-2
    bt2_full = bt2['splits']['full']
    bt2_post = bt2['splits']['post_aug']
    bt2_pre = bt2['splits']['oos']
    g_bt2_1 = bt2_full['sharpe'] >= 0.30
    g_bt2_2 = bt2_post['ci_lo'] > 0
    g_bt2_3 = bt2_pre['sharpe'] >= 0.15 if bt2_pre['n'] > 0 else False
    gates.append((f"BT-2 Full Sharpe >= 0.30 (at {bt2['optimal_exit']})", g_bt2_1))
    gates.append(('BT-2 Post-Aug CI > 0', g_bt2_2))
    gates.append(('BT-2 Pre-Aug Sharpe >= 0.15', g_bt2_3))

    # BT-3
    bt3_post = bt3['splits']['post_aug']
    bt1a_post = bt1a['splits']['post_aug']
    g_bt3_1 = bt3_post['avg'] > bt1a_post['avg']
    g_bt3_2 = bt3['splits']['full']['ci_lo'] > 0
    decomp = bt3.get('decomposition', {})
    pre_overnight = (decomp.get('IS', {}).get('overnight', 0) + decomp.get('OOS', {}).get('overnight', 0)) / 2
    post_overnight = decomp.get('Post-Aug', {}).get('overnight', 0)
    g_bt3_3 = pre_overnight > 0 and post_overnight > 0
    gates.append(('BT-3 Post-Aug avg > BT-1a Post-Aug avg', g_bt3_1))
    gates.append(('BT-3 Full CI > 0', g_bt3_2))
    gates.append(('BT-3 overnight_ret > 0 both periods', g_bt3_3))

    # BT-4
    bt4b_full = bt4['config_results']['B']['splits']['full']
    bt4e_full = bt4['config_results']['E']['splits']['full']
    g_bt4_1 = bt4b_full['sharpe'] >= 0.20
    g_bt4_2 = bt4b_full['ci_lo'] > 0
    g_bt4_3 = bt4b_full['n'] >= 50
    bt4e_pre = bt4['config_results']['E']['splits']['oos']
    bt4b_pre = bt4['config_results']['B']['splits']['oos']
    g_bt4_4 = True
    if bt4e_pre['n'] > 0 and bt4b_pre['n'] > 0:
        g_bt4_4 = (bt4e_pre['avg'] - bt4b_pre['avg']) <= 1.0
    gates.append(('BT-4B Full Sharpe >= 0.20', g_bt4_1))
    gates.append(('BT-4B Full CI > 0', g_bt4_2))
    gates.append(('BT-4B Full n >= 50', g_bt4_3))
    gates.append(('BT-4B pre-Aug not worse than E by >1pp', g_bt4_4))

    # Print
    all_pass = True
    for label, passed in gates:
        status = 'PASS' if passed else 'FAIL'
        if not passed: all_pass = False
        print(f"  [{status:>4s}]  {label}")

    print(f"\n  Overall: {'ALL GATES PASSED' if all_pass else 'SOME GATES FAILED'}")
    return gates


# =============================================================================
# CHARTS
# =============================================================================

def plot_results(bt1a, bt1b, bt2, bt3, bt4, dates):
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(
        f'Agent 3 Revised Backtest Suite  |  {BACKTEST_START} → {BACKTEST_END}\n'
        f'BT-1a: n={bt1a["n"]}  |  BT-1b: n={bt1b["n"]}  |  '
        f'BT-2: n={bt2["n"]}  |  BT-3: n={bt3["n"]}',
        fontsize=11, fontweight='bold')

    colors = {'BT-1a':'#27ae60', 'BT-1b':'#2ecc71', 'BT-2':'#e67e22',
              'BT-3':'#2980b9', 'BT-4B':'#8e44ad'}

    # 1. A6 comparison: cumulative PnL (non-compounding)
    ax = axes[0, 0]
    for bt_id, bt_res, color in [('BT-1a', bt1a, colors['BT-1a']),
                                  ('BT-1b', bt1b, colors['BT-1b']),
                                  ('BT-3',  bt3,  colors['BT-3'])]:
        daily_pnl = defaultdict(list)
        for t in bt_res['trades']:
            if t.get('pnl') is not None:
                daily_pnl[t['scan_date']].append(t['pnl'])
        cum = 0.0
        xs, ys = [], []
        for dt in sorted(dates):
            dp = daily_pnl.get(dt, [])
            if dp:
                n = len(dp)
                cum += sum(CAPITAL / n * (r / 100) for r in dp)
            xs.append(datetime.strptime(dt, '%Y-%m-%d'))
            ys.append(cum)
        ax.plot(xs, ys, color=color, linewidth=1.5, label=f'{bt_id} (${cum:+,.0f})')
    ax.axhline(0, color='#999', ls='--', lw=0.8)
    ax.axvline(datetime.strptime(BREAK_DATE, '%Y-%m-%d'), color='red', ls=':', lw=1, alpha=0.5)
    ax.axvline(datetime.strptime(OOS_CUTOFF, '%Y-%m-%d'), color='blue', ls=':', lw=1, alpha=0.5)
    ax.set_title('A6 Variants: Cumulative P&L (NC)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 2. BT-2 exit time Sharpe curve
    ax = axes[0, 1]
    exit_labels = ['10:00', '10:15', '10:30', '10:45', '11:00', '11:30', '12:00']
    sharpes = [bt2['exit_results'].get(el, {}).get('sharpe', 0) for el in exit_labels]
    wrs = [bt2['exit_results'].get(el, {}).get('wr', 0) * 100 for el in exit_labels]
    x_pos = np.arange(len(exit_labels))
    ax.bar(x_pos, sharpes, color=colors['BT-2'], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exit_labels, fontsize=9)
    ax.axhline(0, color='#333', lw=0.8)
    opt_idx = exit_labels.index(bt2['optimal_exit']) if bt2['optimal_exit'] in exit_labels else 0
    ax.bar(opt_idx, sharpes[opt_idx], color='#c0392b', alpha=0.9)
    ax.set_title(f'BT-2: Sharpe by Exit Time (optimal={bt2["optimal_exit"]})')
    ax.set_ylabel('Sharpe')
    ax.grid(True, alpha=0.2, axis='y')

    # 3. BT-4 config comparison
    ax = axes[0, 2]
    cfg_ids = ['A', 'B', 'C', 'D', 'E']
    cfg_labels = [bt4['config_results'][c]['config'] for c in cfg_ids]
    cfg_sharpes = [bt4['config_results'][c]['splits']['full']['sharpe'] for c in cfg_ids]
    cfg_ns = [bt4['config_results'][c]['splits']['full']['n'] for c in cfg_ids]
    bar_colors = [colors['BT-4B'] if c == 'B' else '#95a5a6' for c in cfg_ids]
    ax.bar(np.arange(len(cfg_ids)), cfg_sharpes, color=bar_colors, alpha=0.8)
    ax.set_xticks(np.arange(len(cfg_ids)))
    ax.set_xticklabels([f'{c}\n{l}\nn={n}' for c, l, n in zip(cfg_ids, cfg_labels, cfg_ns)], fontsize=8)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_title('BT-4: C1 Config Comparison (Full Period Sharpe)')
    ax.set_ylabel('Sharpe')
    ax.grid(True, alpha=0.2, axis='y')

    # 4. BT-4B hold period sweep
    ax = axes[1, 0]
    if bt4.get('hold_sweep'):
        holds = sorted(bt4['hold_sweep'].keys())
        hold_sharpes = [bt4['hold_sweep'][h]['sharpe'] for h in holds]
        hold_avgs = [bt4['hold_sweep'][h]['avg'] for h in holds]
        ax.plot(holds, hold_sharpes, 'o-', color=colors['BT-4B'], linewidth=2, label='Sharpe')
        ax.axvline(C1_HOLD_DAYS, color='red', ls='--', lw=1, alpha=0.5, label=f'T+{C1_HOLD_DAYS}')
        ax.set_xlabel('Hold Period (T+N)')
        ax.set_ylabel('Sharpe')
        ax.set_title('BT-4B: Hold Period Sweep')
        ax.set_xticks(holds)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # 5. Period comparison bar chart
    ax = axes[1, 1]
    bt_ids = ['BT-1a', 'BT-2', 'BT-3', 'BT-4B']
    period_names = ['is', 'oos', 'post_aug']
    period_colors = ['#3498db', '#2ecc71', '#e74c3c']
    bt_splits_map = {'BT-1a': bt1a['splits'], 'BT-2': bt2['splits'],
                     'BT-3': bt3['splits'],
                     'BT-4B': bt4['config_results']['B']['splits']}
    x = np.arange(len(bt_ids))
    w = 0.25
    for pi, (pname, pcol) in enumerate(zip(period_names, period_colors)):
        vals = [bt_splits_map[bt_id][pname]['sharpe'] for bt_id in bt_ids]
        ax.bar(x + (pi - 1) * w, vals, w, color=pcol, alpha=0.8,
               label={'is':'IS','oos':'OOS','post_aug':'Post-Aug'}[pname])
    ax.set_xticks(x)
    ax.set_xticklabels(bt_ids, fontsize=9)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_title('Sharpe by Period')
    ax.set_ylabel('Sharpe')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # 6. Monthly heatmap-style for BT-1a
    ax = axes[1, 2]
    if bt1a.get('monthly'):
        months = [r['month'] for r in bt1a['monthly']]
        avgs = [r['avg'] for r in bt1a['monthly']]
        bar_cols = ['#27ae60' if v > 0 else '#e74c3c' for v in avgs]
        ax.bar(np.arange(len(months)), avgs, color=bar_cols, alpha=0.8)
        ax.set_xticks(np.arange(len(months)))
        ax.set_xticklabels([m[5:] for m in months], fontsize=8, rotation=45)
        ax.axhline(0, color='#333', lw=0.8)
        ax.set_title('BT-1a Monthly Avg PnL (%)')
        ax.set_ylabel('Avg PnL (%)')
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    chart_path = os.path.join(BASE_DIR, 'agent3_revised_bt.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {chart_path}")


# =============================================================================
# JSON SAVE
# =============================================================================

def _make_json_block(splits, config_dict=None):
    block = {}
    if config_dict:
        block['config'] = config_dict
    for period in ['full', 'is', 'oos', 'post_aug']:
        m = splits.get(period, {})
        period_key = {'full':'full_period','is':'is_period','oos':'oos_period','post_aug':'post_aug'}[period]
        block[period_key] = {
            'n': m.get('n', 0), 'avg': m.get('avg', 0), 'median': m.get('med', 0),
            'trimmed_mean': m.get('trimmed_mean', 0), 'sharpe': m.get('sharpe', 0),
            'wr': m.get('wr', 0), 'pf': m.get('pf', 0),
            'ci_95_low': m.get('ci_lo', 0), 'ci_95_high': m.get('ci_hi', 0),
            'max_loss': m.get('max_loss', 0), 'max_gain': m.get('max_gain', 0),
        }
    return block

def save_results(bt1a, bt1b, bt2, bt3, bt4, gates):
    today_str = datetime.now().strftime('%Y%m%d')

    # BT-1a
    j1a = _make_json_block(bt1a['splits'], {'gap_min': 0.02, 'entry': 'open', 'exit': 'T+2_close',
                                             'price_min': A6_PRICE_MIN, 'top_n': TOP_N})
    j1a['bt_id'] = 'BT-1a'
    j1a['run_date'] = today_str
    j1a['monthly'] = bt1a.get('monthly', [])
    p1a = os.path.join(RESULTS_DIR, f'bt_1a_{today_str}.json')
    with open(p1a, 'w') as f: json.dump(j1a, f, indent=2)

    # BT-1b
    j1b = _make_json_block(bt1b['splits'], {'gap_min': 0.02, 'entry': 'delayed_10:35',
                                             'exit': 'T+2_close', 'pattern_gate': 'sustained_rise'})
    j1b['bt_id'] = 'BT-1b'
    j1b['run_date'] = today_str
    j1b['delay_cost_pp'] = bt1b.get('delay_cost', 0)
    j1b['delay_acceptable'] = bt1b.get('delay_acceptable', False)
    p1b = os.path.join(RESULTS_DIR, f'bt_1b_{today_str}.json')
    with open(p1b, 'w') as f: json.dump(j1b, f, indent=2)

    # BT-2
    j2 = _make_json_block(bt2['splits'], {'gap_min': 0.02, 'entry': '09:45',
                                           'exit': bt2.get('optimal_exit', '10:30')})
    j2['bt_id'] = 'BT-2'
    j2['run_date'] = today_str
    j2['optimal_exit'] = bt2.get('optimal_exit', '')
    j2['exit_grid'] = {}
    for el, m in bt2.get('exit_results', {}).items():
        j2['exit_grid'][el] = {'n': m['n'], 'avg': m['avg'], 'sharpe': m['sharpe'],
                                'wr': m['wr'], 'ci_lo': m['ci_lo'], 'ci_hi': m['ci_hi']}
    p2 = os.path.join(RESULTS_DIR, f'bt_2_{today_str}.json')
    with open(p2, 'w') as f: json.dump(j2, f, indent=2)

    # BT-3
    j3 = _make_json_block(bt3['splits'], {'gap_min': 0.02, 'entry': 'open',
                                           'exit': 'T+1_open'})
    j3['bt_id'] = 'BT-3'
    j3['run_date'] = today_str
    j3['decomposition'] = bt3.get('decomposition', {})
    p3 = os.path.join(RESULTS_DIR, f'bt_3_{today_str}.json')
    with open(p3, 'w') as f: json.dump(j3, f, indent=2)

    # BT-4 (all configs)
    j4 = {'bt_id': 'BT-4', 'run_date': today_str, 'configs': {}}
    for cfg_id, cr in bt4.get('config_results', {}).items():
        j4['configs'][cfg_id] = _make_json_block(cr['splits'],
            {'config': cr['config'], 'n_trades': cr['n']})
    if bt4.get('hold_sweep'):
        j4['hold_sweep_b'] = {}
        for h, m in bt4['hold_sweep'].items():
            j4['hold_sweep_b'][f'T+{h}'] = {'n': m['n'], 'avg': m['avg'], 'sharpe': m['sharpe']}
    p4 = os.path.join(RESULTS_DIR, f'bt_4_{today_str}.json')
    with open(p4, 'w') as f: json.dump(j4, f, indent=2)

    # Validation gates
    jg = {'run_date': today_str, 'gates': []}
    for label, passed in gates:
        jg['gates'].append({'label': label, 'passed': passed})
    pg = os.path.join(RESULTS_DIR, f'bt_validation_{today_str}.json')
    with open(pg, 'w') as f: json.dump(jg, f, indent=2)

    print(f"\n  Results saved:")
    for p in [p1a, p1b, p2, p3, p4, pg]:
        print(f"    {p}")


# =============================================================================
# =============================================================================
#
#  MAIN EXECUTION
#
# =============================================================================
# =============================================================================

t_start = time.time()

# Pre-register expectations
print(f"\n{'='*80}")
print(f"  PRE-REGISTERED EXPECTATIONS")
print(f"{'='*80}")
for bt_id, exp in EXPECTATIONS.items():
    print(f"  {bt_id}: {exp}")

# Data pipeline
data = run_data_pipeline()
dates = data['dates']

# Run BT-1a, BT-2, BT-3, BT-4 (no dependencies)
bt1a = run_bt1a(data)
bt2  = run_bt2(data)
bt3  = run_bt3(data)
bt4  = run_bt4(data)

# Run BT-1b (depends on BT-1a SR subset)
bt1b = run_bt1b(data, bt1a)

# Comparison table
run_comparison(bt1a, bt1b, bt2, bt3, bt4)

# Validation gates
gates = check_validation_gates(bt1a, bt1b, bt2, bt3, bt4)

# Charts
plot_results(bt1a, bt1b, bt2, bt3, bt4, dates)

# Save JSON results
save_results(bt1a, bt1b, bt2, bt3, bt4, gates)

elapsed = time.time() - t_start
print(f"\n{'='*80}")
print(f"  TOTAL RUNTIME: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'='*80}")
