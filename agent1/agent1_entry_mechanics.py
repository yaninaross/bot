# %% Agent 1 — E2-A Entry & Trade Mechanics Investigation
#
# 15-experiment investigation of optimal entry/exit mechanics
# for the compressed post-Aug regime.
# Evaluated on Aug-Oct vs Jan-Jul.
#
# Uses full pipeline with AH freshness, gap erosion, CS filter.
# Colab-cell friendly.

import os, sys, time, re, asyncio, copy, json
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
from scipy import stats as sp_stats

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_START = '2026-01-02'
BACKTEST_END   = '2026-10-06'
BREAK_DATE     = '2026-08-01'

TOP_N          = 5

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

EXIT_HOURS = ['10:00','10:30','11:00','11:30','12:00','13:00','14:00','15:00','16:00','17:00']

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


def classify_open_pattern(rth_bars):
    if len(rth_bars) < 60: return 'no_data'
    open_price = rth_bars[0][1]['o']
    low_30m = min(b[1]['l'] for b in rth_bars[:30])
    close_60m = rth_bars[59][1]['c']
    drop_30m = (low_30m - open_price) / open_price if open_price > 0 else 0
    recovery_denom = abs(open_price - low_30m)
    recovery = (close_60m - low_30m) / recovery_denom if recovery_denom > 0 else 0
    if drop_30m < -0.03 and recovery > 0.5: return 'drop_then_recover'
    elif drop_30m < -0.03 and recovery <= 0.5: return 'sustained_drop'
    elif drop_30m >= -0.01: return 'flat_or_bounce'
    else: return 'mild_drop'

def find_local_high(bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    fallback_bar=FALLBACK_BAR, min_entry_bar=0):
    if len(bars) < window * 2 + 1: return None
    start = max(window, min_entry_bar - window)
    for i in range(start, min(max_peak_bar, len(bars) - window)):
        ch = bars[i][1]['h']
        if all(bars[i - j][1]['h'] < ch and bars[i + j][1]['h'] < ch
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(bars):
                return {'entry_price': bars[ci][1]['c'], 'entry_time': bars[ci][0],
                        'entry_type': 'local_high', 'peak_bar': i, 'entry_bar': ci}
    fb = max(fallback_bar, min_entry_bar) if fallback_bar is not None else None
    if fb is not None and fb < len(bars):
        return {'entry_price': bars[fb][1]['c'], 'entry_time': bars[fb][0],
                'entry_type': 'fallback', 'peak_bar': None, 'entry_bar': fb}
    return None

def _T_to_min_entry_bar(T):
    h, m = int(T[:2]), int(T[3:5])
    return max(0, h * 60 + m - 9 * 60 - 30)

def get_price_at_T(bar_data, T):
    if bar_data is None: return None
    all_bars = (bar_data.get('pm') or []) + (bar_data.get('rth') or [])
    best = None
    for t_str, bd in all_bars:
        if t_str <= T: best = bd['c']
        else: break
    return best


def scan_day(dt_str):
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
        if gap < 0.03: continue
        fs = _FLOAT.get(tk)
        if fs is not None and fs < FLOAT_MIN: continue
        candidates.append({'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                           'open_price': cur_p, 'scan_date': dt_str,
                           'volume_today': today[tk].get('v', 0), 'prior_str': prior_str})
    candidates.sort(key=lambda x: -x['gap_pct'])
    nxt = _next_n_td(dt_str, 2)
    t1 = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
    t2 = nxt[1].strftime('%Y-%m-%d') if len(nxt) >= 2 else None
    for s in candidates:
        s['t1_date'] = t1; s['t2_date'] = t2
    return candidates


def compute_metrics(pnl_list, label=''):
    if not pnl_list:
        return {'label':label,'n':0,'avg':0,'med':0,'std':0,'wr':0,
                'sharpe':0,'pf':0,'ci_lo':0,'ci_hi':0,'trimmed_mean':0,
                'max_loss':0,'max_gain':0}
    a = np.array(pnl_list, dtype=float)
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
# EXPERIMENT ENTRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def find_exit_price(bar_data, entry_time, exit_time='17:00'):
    session_bars = bar_data.get('rth', []) + bar_data.get('ah', [])
    exit_p = None
    for t_str, bar in session_bars:
        if t_str > exit_time: break
        if t_str >= entry_time: exit_p = bar['c']
    return exit_p

def short_pnl(entry_price, exit_price):
    if exit_price is None or entry_price is None or entry_price <= 0: return None
    return round((entry_price - exit_price) / entry_price * 100, 3)

def find_orb_entry(rth_bars, range_n, max_wait_bar=30):
    if len(rth_bars) < range_n + 1: return None
    range_low = min(b[1]['l'] for b in rth_bars[:range_n])
    for i in range(range_n, min(max_wait_bar, len(rth_bars))):
        if rth_bars[i][1]['c'] < range_low:
            return {'entry_price': rth_bars[i][1]['c'], 'entry_time': rth_bars[i][0],
                    'entry_bar': i, 'entry_type': 'orb', 'range_low': range_low}
    return None

def compute_vwap_series(rth_bars):
    cum_tp_vol, cum_vol = 0.0, 0.0
    vwaps = []
    for t_str, bar in rth_bars:
        tp = (bar['h'] + bar['l'] + bar['c']) / 3.0
        cum_tp_vol += tp * bar['v']
        cum_vol += bar['v']
        vwap = cum_tp_vol / cum_vol if cum_vol > 0 else tp
        vwaps.append(vwap)
    return vwaps

def find_vwap_entry(rth_bars, min_bar=1, max_bar=30):
    if len(rth_bars) < 3: return None
    vwaps = compute_vwap_series(rth_bars)
    above_vwap = False
    for i in range(len(rth_bars)):
        if i < min_bar: continue
        if i >= max_bar: break
        if rth_bars[i][1]['c'] > vwaps[i]:
            above_vwap = True
        elif above_vwap and rth_bars[i][1]['c'] < vwaps[i]:
            return {'entry_price': rth_bars[i][1]['c'], 'entry_time': rth_bars[i][0],
                    'entry_bar': i, 'entry_type': 'vwap_cross', 'vwap': vwaps[i]}
    return None

def compute_atr14(ticker, date_str, dates_list):
    try:
        idx = dates_list.index(date_str)
    except ValueError:
        return None
    if idx < 14: return None
    prior = dates_list[max(0, idx - 14):idx]
    trs = []
    prev_c = None
    for d in prior:
        gd = _gd_cache.get(d, {})
        tk_data = gd.get(ticker)
        if tk_data is None:
            prev_c = None
            continue
        h, l = tk_data['h'], tk_data['l']
        if prev_c is not None:
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        else:
            tr = h - l
        if tr > 0:
            trs.append(tr)
        prev_c = tk_data['c']
    if len(trs) < 7: return None
    return float(np.mean(trs))

def find_trailing_exit(bar_data, entry_time, entry_price, retrace_pct, exit_time='17:00'):
    session_bars = bar_data.get('rth', []) + bar_data.get('ah', [])
    mfe_low = entry_price
    exit_p, exit_t = None, None
    for t_str, bar in session_bars:
        if t_str < entry_time: continue
        if t_str > exit_time: break
        mfe_low = min(mfe_low, bar['l'])
        mfe_drop = entry_price - mfe_low
        if mfe_drop > 0:
            retrace = (bar['h'] - mfe_low) / mfe_drop
            if retrace >= retrace_pct:
                stop_price = mfe_low + retrace_pct * mfe_drop
                return stop_price, t_str, True
        exit_p = bar['c']
        exit_t = t_str
    return exit_p, exit_t, False

def find_sl_exit(bar_data, entry_time, entry_price, sl_pct, exit_time='17:00'):
    session_bars = bar_data.get('rth', []) + bar_data.get('ah', [])
    exit_p, exit_t = None, None
    for t_str, bar in session_bars:
        if t_str < entry_time: continue
        if t_str > exit_time: break
        if bar['h'] >= entry_price * (1 + sl_pct / 100):
            return entry_price * (1 + sl_pct / 100), t_str, True
        exit_p = bar['c']
        exit_t = t_str
    return exit_p, exit_t, False

def detect_sd_early(rth_bars, check_bar=5):
    if len(rth_bars) < check_bar + 1: return False
    open_price = rth_bars[0][1]['o']
    if open_price <= 0: return False
    low_so_far = min(b[1]['l'] for b in rth_bars[:check_bar + 1])
    drop = (low_so_far - open_price) / open_price
    current = rth_bars[check_bar][1]['c']
    recovery_denom = abs(open_price - low_so_far)
    recovery = (current - low_so_far) / recovery_denom if recovery_denom > 0 else 0
    return drop < -0.03 and recovery < 0.50


def eval_split(pnl_dates, label):
    pre = [(p, d) for p, d in pnl_dates if d < BREAK_DATE]
    post = [(p, d) for p, d in pnl_dates if d >= BREAK_DATE]
    pre_m = compute_metrics([x[0] for x in pre], f'{label} Pre')
    post_m = compute_metrics([x[0] for x in post], f'{label} Post')
    return pre_m, post_m


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    t_total = time.time()
    print(f"\n{'='*80}")
    print(f"  E2-A ENTRY & TRADE MECHANICS INVESTIGATION")
    print(f"  {BACKTEST_START} → {BACKTEST_END}  |  Break date: {BREAK_DATE}")
    print(f"  TOP_N={TOP_N}, CS only, SOFT float, AH fresh, erosion")
    print(f"  Exit {EXIT_TIME} ET | ${CAPITAL:,.0f}/day NC")
    print(f"{'='*80}")

    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    print(f"\n  {len(dates)} trading days")

    print(f"\n  Phase 1: Scanning...")
    all_candidates = {}
    for i, dt in enumerate(dates):
        all_candidates[dt] = scan_day(dt)
        if (i+1) % 30 == 0 or i == len(dates) - 1:
            print(f"    [{i+1:>3}/{len(dates)}] {dt}  |  {time.time()-t_total:.0f}s")

    all_pairs = set()
    for dt, cands in all_candidates.items():
        for c in cands[:PRE_FILTER_N]:
            all_pairs.add((c['ticker'], dt))
            if c.get('prior_str'): all_pairs.add((c['ticker'], c['prior_str']))

    print(f"\n  Phase 2: Fetching {len(all_pairs)} bar pairs...")
    pairs_list = list(all_pairs)
    all_bars = {}
    for cs in range(0, len(pairs_list), 100):
        chunk = pairs_list[cs:cs + 100]
        all_bars.update(fetch_minute_bars_batch(chunk))
        done = min(cs + 100, len(pairs_list))
        if done % 500 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    print(f"  → {len(all_bars)}/{len(all_pairs)} pairs")

    gap_min_final = GAP_MIN_PCT / 100.0
    print(f"\n  Phase 2a: AH freshness...")
    n_stale = 0
    for dt in dates:
        for c in all_candidates.get(dt, [])[:PRE_FILTER_N]:
            tk = c['ticker']
            prior_bars = all_bars.get((tk, c.get('prior_str', '')))
            if prior_bars is None: continue
            ah = prior_bars.get('ah', [])
            if ah:
                ah_ref = ah[-1][1]['c']
                c['ah_ref_price'] = ah_ref
                fresh_gap = (c['open_price'] - ah_ref) / ah_ref
                c['gap_pct_daily'] = c['gap_pct']
                c['gap_pct'] = fresh_gap
                if fresh_gap < gap_min_final:
                    c['ah_stale'] = True; n_stale += 1
    print(f"  → {n_stale} stale excluded")

    print(f"\n  Phase 2b: Re-rank + erosion...")
    n_eroded = 0
    for dt in dates:
        reranked = []
        for c in all_candidates.get(dt, []):
            if c.get('ah_stale'): continue
            tk = c['ticker']
            bar_data = all_bars.get((tk, dt))
            price_T = get_price_at_T(bar_data, SCAN_TIME_ET)
            if price_T is None: continue
            ref_price = c.get('ah_ref_price', c['prior_close'])
            gap_T = (price_T - ref_price) / ref_price
            if gap_T < gap_min_final or price_T < PRICE_MIN: continue
            c['gap_pct'] = gap_T; c['scan_price'] = price_T
            rth = bar_data.get('rth', []) if bar_data else []
            if rth:
                daily_open = rth[0][1]['o']
                scan_bars = [b for b in rth if b[0] <= SCAN_TIME_ET]
                gap_range = daily_open - ref_price
                if scan_bars and gap_range > 0:
                    low_to_scan = min(b[1]['l'] for b in scan_bars)
                    gc = (daily_open - low_to_scan) / gap_range
                    c['gap_consumed'] = gc
                    if gc > GAP_CONSUMED_MAX:
                        n_eroded += 1; continue
            reranked.append(c)
        reranked.sort(key=lambda x: -x['gap_pct'])
        all_candidates[dt] = reranked
    print(f"  → {n_eroded} eroded")

    all_tickers = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]: all_tickers.add(c['ticker'])
    print(f"\n  Phase 3: Refs for {len(all_tickers)} tickers...")
    for cs in range(0, len(sorted(all_tickers)), 200):
        fetch_ticker_refs(sorted(all_tickers)[cs:cs + 200])
    n_cs = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('type') == 'CS')
    print(f"  → {n_cs} CS / {len(all_tickers)}")

    min_eb = _T_to_min_entry_bar(SCAN_TIME_ET)
    pattern_cache, entry_cache = {}, {}
    for key, bar_data in all_bars.items():
        rth_bars = bar_data.get('rth', []) if isinstance(bar_data, dict) else bar_data
        pattern_cache[key] = classify_open_pattern(rth_bars)
        entry_cache[key] = find_local_high(rth_bars, min_entry_bar=min_eb) if rth_bars and len(rth_bars) >= PEAK_WINDOW * 2 + 1 else None

    outcome_dates = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            outcome_dates.add(c['scan_date'])
            if c.get('t1_date'): outcome_dates.add(c['t1_date'])
            if c.get('t2_date'): outcome_dates.add(c['t2_date'])
    for od in sorted(od for od in outcome_dates if od not in _gd_cache):
        try: fetch_grouped_daily(od)
        except: pass

    print(f"\n  Pipeline done: {time.time()-t_total:.0f}s")
    return all_candidates, all_bars, pattern_cache, entry_cache, dates, t_total


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD BASE TRADES
# ═══════════════════════════════════════════════════════════════════════════════

def build_trades(all_candidates, all_bars, pattern_cache, entry_cache, dates, top_n=TOP_N):
    trades = []
    for dt in dates:
        cs_qualified = []
        for c in all_candidates.get(dt, [])[:PRE_FILTER_N]:
            if _ticker_ref.get(c['ticker'], {}).get('type') != 'CS': continue
            cc = copy.deepcopy(c)
            cc['asset_type'] = 'CS'
            cc['open_pattern'] = pattern_cache.get((c['ticker'], dt), 'no_data')
            cs_qualified.append(cc)

        for c in cs_qualified[:top_n]:
            tk, dt2 = c['ticker'], c['scan_date']
            entry = entry_cache.get((tk, dt2))
            if entry is None: continue
            tc = copy.deepcopy(c)
            tc['entry_price'] = entry['entry_price']
            tc['entry_time'] = entry['entry_time']
            tc['entry_type'] = entry['entry_type']
            tc['peak_bar'] = entry.get('peak_bar')
            tc['entry_bar'] = entry.get('entry_bar', FALLBACK_BAR)
            ep = tc['entry_price']

            bar_data = all_bars.get((tk, dt2))
            exit_price = find_exit_price(bar_data, tc['entry_time'], EXIT_TIME) if bar_data else None

            if exit_price is not None:
                tc['pnl_eod'] = round((ep - exit_price) / ep * 100, 3)
            else:
                t0d = _gd_cache.get(dt2, {}).get(tk, {})
                tc['pnl_eod'] = round((ep - t0d['c']) / ep * 100, 3) if t0d.get('c') else None

            tc['exit_price'] = exit_price

            # MFE / MAE
            mfe, mae = 0.0, 0.0
            if bar_data:
                session_bars = bar_data.get('rth', []) + bar_data.get('ah', [])
                for t_str, bar in session_bars:
                    if t_str < tc['entry_time']: continue
                    if t_str > EXIT_TIME: break
                    mfe = max(mfe, (ep - bar['l']) / ep * 100)
                    mae = max(mae, (bar['h'] - ep) / ep * 100)
            tc['mfe'] = round(mfe, 3)
            tc['mae'] = round(mae, 3)

            # PnL at different exit hours
            if bar_data:
                session_bars = bar_data.get('rth', []) + bar_data.get('ah', [])
                for ex_t in EXIT_HOURS:
                    if ex_t <= tc['entry_time']:
                        tc[f'pnl_{ex_t}'] = None; continue
                    ex_p = None
                    for t_str, bar in session_bars:
                        if t_str > ex_t: break
                        if t_str >= tc['entry_time']: ex_p = bar['c']
                    tc[f'pnl_{ex_t}'] = round((ep - ex_p) / ep * 100, 3) if ex_p else None

            # Entry vs open
            tc['entry_vs_open'] = round((ep - tc['open_price']) / tc['open_price'] * 100, 3) if tc['open_price'] > 0 else 0

            if tc.get('pnl_eod') is not None:
                trades.append(tc)
    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE + BUILD
# ═══════════════════════════════════════════════════════════════════════════════

all_candidates, all_bars, pattern_cache, entry_cache, dates, t_total = run_pipeline()
trades = build_trades(all_candidates, all_bars, pattern_cache, entry_cache, dates)

pre_trades = [t for t in trades if t['scan_date'] < BREAK_DATE]
post_trades = [t for t in trades if t['scan_date'] >= BREAK_DATE]
pre_m = compute_metrics([t['pnl_eod'] for t in pre_trades], 'Pre-Aug')
post_m = compute_metrics([t['pnl_eod'] for t in post_trades], 'Post-Aug')

BASELINE_PRE = pre_m['avg']
BASELINE_POST = post_m['avg']
BASELINE_DELTA = BASELINE_POST - BASELINE_PRE

print(f"\n{'='*80}")
print(f"  BASELINE: {len(trades)} trades (Pre: {pre_m['n']}, Post: {post_m['n']})")
print(f"  Pre-Aug:  avg={pre_m['avg']:+.2f}%, Sharpe={pre_m['sharpe']:.3f}, WR={pre_m['wr']*100:.0f}%")
print(f"  Post-Aug: avg={post_m['avg']:+.2f}%, Sharpe={post_m['sharpe']:.3f}, WR={post_m['wr']*100:.0f}%")
print(f"  Delta:    {BASELINE_DELTA:+.2f}pp")
print(f"{'='*80}")


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD CANDIDATE POOL + SPY BARS + ATR
# ═══════════════════════════════════════════════════════════════════════════════

candidate_pool = {}
for dt in dates:
    cs_qualified = []
    for c in all_candidates.get(dt, [])[:PRE_FILTER_N]:
        if _ticker_ref.get(c['ticker'], {}).get('type') != 'CS': continue
        cs_qualified.append(c)
    pool = []
    for c in cs_qualified[:TOP_N]:
        bd = all_bars.get((c['ticker'], dt))
        if bd:
            pool.append((c, bd))
    candidate_pool[dt] = pool

print(f"\n  Fetching SPY minute bars for {len(dates)} days...")
spy_pairs = [('SPY', dt) for dt in dates]
spy_bars_cache = {}
for cs in range(0, len(spy_pairs), 100):
    chunk = spy_pairs[cs:cs + 100]
    spy_bars_cache.update(fetch_minute_bars_batch(chunk))
print(f"  → {len(spy_bars_cache)} SPY bar days")

print(f"  Computing ATR14 for trade tickers...")
atr_cache = {}
trade_tickers = set(t['ticker'] for t in trades)
for tk in trade_tickers:
    for dt in dates:
        atr = compute_atr14(tk, dt, dates)
        if atr is not None:
            atr_cache[(tk, dt)] = atr
print(f"  → {len(atr_cache)} ATR14 values computed")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

exp_results = {}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 1 — LOCAL HIGH DETECTION AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 1 — LOCAL HIGH DETECTION AUDIT")
print(f"{'='*80}")

lh_audit = {'clean': [], 'late': [], 'fallback': []}
for t in trades:
    bar_data = all_bars.get((t['ticker'], t['scan_date']))
    if bar_data is None: continue
    rth = bar_data.get('rth', [])
    eb = t.get('entry_bar', FALLBACK_BAR)
    pb = t.get('peak_bar')

    entry_vs_open = t.get('entry_vs_open', 0)

    # Subsequent PnL from entry at 5, 10, 15 bars
    sub_pnl = {}
    ep = t['entry_price']
    for offset in [5, 10, 15]:
        idx = eb + offset
        if idx < len(rth) and ep > 0:
            sub_pnl[f'bar+{offset}'] = round((ep - rth[idx][1]['c']) / ep * 100, 3)
        else:
            sub_pnl[f'bar+{offset}'] = None

    rec = {'ticker': t['ticker'], 'date': t['scan_date'], 'period': 'pre' if t['scan_date'] < BREAK_DATE else 'post',
           'entry_type': t['entry_type'], 'peak_bar': pb, 'entry_bar': eb,
           'entry_vs_open': entry_vs_open, 'pnl_eod': t['pnl_eod'], **sub_pnl}

    if t['entry_type'] == 'fallback':
        lh_audit['fallback'].append(rec)
    elif pb is not None and pb <= 7:
        lh_audit['clean'].append(rec)
    else:
        lh_audit['late'].append(rec)

print(f"\n  Fire type distribution:")
for ft in ['clean', 'late', 'fallback']:
    recs = lh_audit[ft]
    pre_r = [r for r in recs if r['period'] == 'pre']
    post_r = [r for r in recs if r['period'] == 'post']
    pre_avg = np.mean([r['pnl_eod'] for r in pre_r]) if pre_r else 0
    post_avg = np.mean([r['pnl_eod'] for r in post_r]) if post_r else 0
    print(f"    {ft:10s}: Pre n={len(pre_r):3d} avg={pre_avg:+.2f}%  |  Post n={len(post_r):3d} avg={post_avg:+.2f}%  |  Δ={post_avg-pre_avg:+.2f}pp")

print(f"\n  Entry bar distribution for local_high fires:")
bar_dist = defaultdict(lambda: {'pre': [], 'post': []})
for ft in ['clean', 'late']:
    for r in lh_audit[ft]:
        bar_dist[r['peak_bar']][r['period']].append(r['pnl_eod'])

bar_rows = []
for pb in sorted(bar_dist.keys()):
    pre_p = bar_dist[pb]['pre']
    post_p = bar_dist[pb]['post']
    bar_rows.append([f"bar {pb}", len(pre_p), f"{np.mean(pre_p):+.2f}%" if pre_p else "—",
                     len(post_p), f"{np.mean(post_p):+.2f}%" if post_p else "—"])
print()
print(tabulate(bar_rows, headers=['Peak Bar', 'Pre N', 'Pre Avg', 'Post N', 'Post Avg'], tablefmt='simple'))

print(f"\n  Subsequent PnL from entry point:")
sub_rows = []
for ft in ['clean', 'late', 'fallback']:
    for offset in [5, 10, 15]:
        key = f'bar+{offset}'
        pre_vals = [r[key] for r in lh_audit[ft] if r['period'] == 'pre' and r[key] is not None]
        post_vals = [r[key] for r in lh_audit[ft] if r['period'] == 'post' and r[key] is not None]
        sub_rows.append([f"{ft}@+{offset}", len(pre_vals),
                         f"{np.mean(pre_vals):+.2f}%" if pre_vals else "—",
                         len(post_vals), f"{np.mean(post_vals):+.2f}%" if post_vals else "—"])
print()
print(tabulate(sub_rows, headers=['Type@Offset', 'Pre N', 'Pre Avg', 'Post N', 'Post Avg'], tablefmt='simple'))

print(f"\n  Entry vs open price by fire type:")
for ft in ['clean', 'late', 'fallback']:
    pre_evo = [r['entry_vs_open'] for r in lh_audit[ft] if r['period'] == 'pre']
    post_evo = [r['entry_vs_open'] for r in lh_audit[ft] if r['period'] == 'post']
    print(f"    {ft:10s}: Pre={np.mean(pre_evo):+.2f}% | Post={np.mean(post_evo):+.2f}%" if pre_evo and post_evo else f"    {ft}: insufficient data")

exp_results['exp1'] = {
    'clean_pre': compute_metrics([r['pnl_eod'] for r in lh_audit['clean'] if r['period'] == 'pre']),
    'clean_post': compute_metrics([r['pnl_eod'] for r in lh_audit['clean'] if r['period'] == 'post']),
    'late_pre': compute_metrics([r['pnl_eod'] for r in lh_audit['late'] if r['period'] == 'pre']),
    'late_post': compute_metrics([r['pnl_eod'] for r in lh_audit['late'] if r['period'] == 'post']),
    'fb_pre': compute_metrics([r['pnl_eod'] for r in lh_audit['fallback'] if r['period'] == 'pre']),
    'fb_post': compute_metrics([r['pnl_eod'] for r in lh_audit['fallback'] if r['period'] == 'post']),
}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 2 — OPENING RANGE BREAKOUT ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 2 — OPENING RANGE BREAKOUT (ORB) ENTRY")
print(f"{'='*80}")

orb_results = {}
for range_n in [1, 2, 3, 5]:
    orb_trades = []
    for dt, pool in candidate_pool.items():
        for c, bd in pool:
            rth = bd.get('rth', [])
            entry = find_orb_entry(rth, range_n, max_wait_bar=30)
            if entry is None: continue
            exit_p = find_exit_price(bd, entry['entry_time'], EXIT_TIME)
            pnl = short_pnl(entry['entry_price'], exit_p)
            if pnl is not None:
                orb_trades.append({'pnl': pnl, 'scan_date': dt, 'ticker': c['ticker'],
                                   'entry_bar': entry['entry_bar']})

    pre_m_o, post_m_o = eval_split([(t['pnl'], t['scan_date']) for t in orb_trades], f'ORB-{range_n}')
    orb_results[range_n] = {'pre': pre_m_o, 'post': post_m_o, 'trades': orb_trades}
    delta = post_m_o['avg'] - pre_m_o['avg']
    print(f"  ORB range={range_n}: Pre avg={pre_m_o['avg']:+.2f}% n={pre_m_o['n']}  |  "
          f"Post avg={post_m_o['avg']:+.2f}% n={post_m_o['n']}  |  Δ={delta:+.2f}pp  |  "
          f"vs baseline Δ: {delta - BASELINE_DELTA:+.2f}pp")

print(f"\n  Baseline: Pre={BASELINE_PRE:+.2f}%, Post={BASELINE_POST:+.2f}%, Δ={BASELINE_DELTA:+.2f}pp")

exp_results['exp2'] = {f'orb_{n}': {'pre': orb_results[n]['pre'], 'post': orb_results[n]['post']} for n in orb_results}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 3 — VWAP-RELATIVE ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 3 — VWAP-RELATIVE ENTRY")
print(f"{'='*80}")

vwap_trades = []
for dt, pool in candidate_pool.items():
    for c, bd in pool:
        rth = bd.get('rth', [])
        entry = find_vwap_entry(rth, min_bar=1, max_bar=30)
        if entry is None: continue
        exit_p = find_exit_price(bd, entry['entry_time'], EXIT_TIME)
        pnl = short_pnl(entry['entry_price'], exit_p)
        if pnl is not None:
            vwap_trades.append({'pnl': pnl, 'scan_date': dt, 'ticker': c['ticker'],
                                'entry_bar': entry['entry_bar']})

vwap_pre, vwap_post = eval_split([(t['pnl'], t['scan_date']) for t in vwap_trades], 'VWAP')
delta_v = vwap_post['avg'] - vwap_pre['avg']
print(f"\n  VWAP cross entry:")
print(f"    Pre:  avg={vwap_pre['avg']:+.2f}%, n={vwap_pre['n']}, WR={vwap_pre['wr']*100:.0f}%")
print(f"    Post: avg={vwap_post['avg']:+.2f}%, n={vwap_post['n']}, WR={vwap_post['wr']*100:.0f}%")
print(f"    Δ={delta_v:+.2f}pp  |  vs baseline Δ: {delta_v - BASELINE_DELTA:+.2f}pp")

# Compare: VWAP vs local_high vs fallback (existing trades)
lh_only = [t for t in trades if t['entry_type'] == 'local_high']
fb_only = [t for t in trades if t['entry_type'] == 'fallback']

print(f"\n  Comparison:")
for label, subset in [('local_high', lh_only), ('fallback', fb_only), ('vwap_cross', vwap_trades)]:
    pre_s = [t for t in subset if t.get('scan_date', t.get('date', '')) < BREAK_DATE]
    post_s = [t for t in subset if t.get('scan_date', t.get('date', '')) >= BREAK_DATE]
    pkey = 'pnl_eod' if 'pnl_eod' in (subset[0] if subset else {}) else 'pnl'
    pre_avg = np.mean([t[pkey] for t in pre_s]) if pre_s else 0
    post_avg = np.mean([t[pkey] for t in post_s]) if post_s else 0
    print(f"    {label:12s}: Pre avg={pre_avg:+.2f}% n={len(pre_s)} | Post avg={post_avg:+.2f}% n={len(post_s)} | Δ={post_avg-pre_avg:+.2f}pp")

exp_results['exp3'] = {'pre': vwap_pre, 'post': vwap_post}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 4 — VOLUME-WEIGHTED ENTRY QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 4 — VOLUME-WEIGHTED ENTRY QUALITY")
print(f"{'='*80}")

vol_quality = []
for t in trades:
    bd = all_bars.get((t['ticker'], t['scan_date']))
    if bd is None: continue
    rth = bd.get('rth', [])
    pm = bd.get('pm', [])
    eb = t.get('entry_bar', FALLBACK_BAR)
    if eb >= len(rth): continue

    entry_vol = rth[eb][1]['v']
    pm_total_vol = sum(b[1]['v'] for b in pm) if pm else 0
    first_5_vol = sum(rth[i][1]['v'] for i in range(min(5, len(rth))))
    vol_ratio_pm = entry_vol / pm_total_vol if pm_total_vol > 0 else 0
    vol_ratio_5 = entry_vol / first_5_vol if first_5_vol > 0 else 0

    vol_quality.append({
        'pnl': t['pnl_eod'], 'date': t['scan_date'],
        'entry_vol': entry_vol, 'pm_vol': pm_total_vol,
        'vol_ratio_pm': vol_ratio_pm, 'vol_ratio_5': vol_ratio_5,
        'entry_type': t['entry_type'], 'period': 'pre' if t['scan_date'] < BREAK_DATE else 'post'
    })

# Quartile analysis by entry volume
vol_ratios = [v['vol_ratio_5'] for v in vol_quality if v['vol_ratio_5'] > 0]
if vol_ratios:
    q25, q50, q75 = np.percentile(vol_ratios, [25, 50, 75])
    print(f"\n  Entry bar vol / first-5-bar vol quartiles: Q25={q25:.3f}, Q50={q50:.3f}, Q75={q75:.3f}")

    vol_buckets = {'low': [], 'mid': [], 'high': []}
    for v in vol_quality:
        r = v['vol_ratio_5']
        if r <= q25: vol_buckets['low'].append(v)
        elif r <= q75: vol_buckets['mid'].append(v)
        else: vol_buckets['high'].append(v)

    vol_rows = []
    for bk in ['low', 'mid', 'high']:
        pre_b = [v['pnl'] for v in vol_buckets[bk] if v['period'] == 'pre']
        post_b = [v['pnl'] for v in vol_buckets[bk] if v['period'] == 'post']
        vol_rows.append([bk, len(pre_b), f"{np.mean(pre_b):+.2f}%" if pre_b else "—",
                         len(post_b), f"{np.mean(post_b):+.2f}%" if post_b else "—"])
    print()
    print(tabulate(vol_rows, headers=['Vol Bucket', 'Pre N', 'Pre Avg', 'Post N', 'Post Avg'], tablefmt='simple'))

    # By entry type
    print(f"\n  Volume ratio by entry type:")
    for et in ['local_high', 'fallback']:
        pre_vr = [v['vol_ratio_5'] for v in vol_quality if v['entry_type'] == et and v['period'] == 'pre']
        post_vr = [v['vol_ratio_5'] for v in vol_quality if v['entry_type'] == et and v['period'] == 'post']
        print(f"    {et:12s}: Pre mean={np.mean(pre_vr):.3f} | Post mean={np.mean(post_vr):.3f}" if pre_vr and post_vr else f"    {et}: insufficient data")

exp_results['exp4'] = {bk: {'pre': compute_metrics([v['pnl'] for v in vol_buckets[bk] if v['period'] == 'pre']),
                            'post': compute_metrics([v['pnl'] for v in vol_buckets[bk] if v['period'] == 'post'])}
                       for bk in vol_buckets}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 5 — FALLBACK TIMING GRID
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 5 — FALLBACK TIMING GRID")
print(f"{'='*80}")

min_eb = _T_to_min_entry_bar(SCAN_TIME_ET)
fb_grid_results = {}
for fb_bar in [3, 5, 7, 9, 11, 13, 15, 19]:
    fb_trades = []
    for dt, pool in candidate_pool.items():
        for c, bd in pool:
            rth = bd.get('rth', [])
            if not rth or len(rth) < PEAK_WINDOW * 2 + 1: continue
            entry = find_local_high(rth, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                                    fallback_bar=fb_bar, min_entry_bar=min_eb)
            if entry is None: continue
            exit_p = find_exit_price(bd, entry['entry_time'], EXIT_TIME)
            pnl = short_pnl(entry['entry_price'], exit_p)
            if pnl is not None:
                fb_trades.append({'pnl': pnl, 'scan_date': dt, 'entry_type': entry['entry_type']})

    pre_fb, post_fb = eval_split([(t['pnl'], t['scan_date']) for t in fb_trades], f'FB={fb_bar}')
    fb_only_pre = [t for t in fb_trades if t['scan_date'] < BREAK_DATE and t['entry_type'] == 'fallback']
    fb_only_post = [t for t in fb_trades if t['scan_date'] >= BREAK_DATE and t['entry_type'] == 'fallback']
    fb_grid_results[fb_bar] = {
        'all_pre': pre_fb, 'all_post': post_fb,
        'fb_pre_avg': float(np.mean([t['pnl'] for t in fb_only_pre])) if fb_only_pre else 0,
        'fb_post_avg': float(np.mean([t['pnl'] for t in fb_only_post])) if fb_only_post else 0,
        'fb_pre_n': len(fb_only_pre), 'fb_post_n': len(fb_only_post),
    }

fb_rows = []
for fb_bar in [3, 5, 7, 9, 11, 13, 15, 19]:
    r = fb_grid_results[fb_bar]
    delta = r['all_post']['avg'] - r['all_pre']['avg']
    fb_rows.append([
        f"FB={fb_bar}", r['all_pre']['n'], f"{r['all_pre']['avg']:+.2f}%",
        r['all_post']['n'], f"{r['all_post']['avg']:+.2f}%", f"{delta:+.2f}pp",
        r['fb_pre_n'], f"{r['fb_pre_avg']:+.2f}%", r['fb_post_n'], f"{r['fb_post_avg']:+.2f}%"
    ])
print()
print(tabulate(fb_rows, headers=['Config', 'Pre N', 'Pre Avg', 'Post N', 'Post Avg', 'Δ',
                                  'FB Pre N', 'FB Pre Avg', 'FB Post N', 'FB Post Avg'],
               tablefmt='simple', stralign='right'))

# No-fallback variant
nofb_trades = []
for dt, pool in candidate_pool.items():
    for c, bd in pool:
        rth = bd.get('rth', [])
        if not rth or len(rth) < PEAK_WINDOW * 2 + 1: continue
        entry = find_local_high(rth, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                                fallback_bar=None, min_entry_bar=min_eb)
        if entry is None: continue
        exit_p = find_exit_price(bd, entry['entry_time'], EXIT_TIME)
        pnl = short_pnl(entry['entry_price'], exit_p)
        if pnl is not None:
            nofb_trades.append({'pnl': pnl, 'scan_date': dt})

nofb_pre, nofb_post = eval_split([(t['pnl'], t['scan_date']) for t in nofb_trades], 'No-FB')
print(f"\n  No fallback (LH only): Pre avg={nofb_pre['avg']:+.2f}% n={nofb_pre['n']} | "
      f"Post avg={nofb_post['avg']:+.2f}% n={nofb_post['n']} | Δ={nofb_post['avg']-nofb_pre['avg']:+.2f}pp")

exp_results['exp5'] = {f'fb_{b}': {'all_pre': fb_grid_results[b]['all_pre'], 'all_post': fb_grid_results[b]['all_post']}
                       for b in fb_grid_results}
exp_results['exp5']['no_fb'] = {'pre': nofb_pre, 'post': nofb_post}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 6 — GAP FRESHNESS AT ENTRY VS AT SCAN
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 6 — GAP FRESHNESS AT ENTRY VS AT SCAN")
print(f"{'='*80}")

gap_at_entry = []
for t in trades:
    ref_price = t.get('ah_ref_price', t['prior_close'])
    open_price = t['open_price']
    entry_price = t['entry_price']
    if ref_price <= 0 or open_price <= 0: continue
    gap_at_scan = (open_price - ref_price) / ref_price
    gap_at_e = (entry_price - ref_price) / ref_price
    gap_remaining = gap_at_e / gap_at_scan if gap_at_scan > 0 else 0

    gap_at_entry.append({
        'pnl': t['pnl_eod'], 'date': t['scan_date'],
        'gap_remaining': gap_remaining, 'gap_at_scan': gap_at_scan,
        'gap_at_entry': gap_at_e,
        'period': 'pre' if t['scan_date'] < BREAK_DATE else 'post'
    })

print(f"\n  Gap remaining at entry (entry_gap / scan_gap):")
for period in ['pre', 'post']:
    subset = [g for g in gap_at_entry if g['period'] == period]
    if subset:
        gr = [g['gap_remaining'] for g in subset]
        print(f"    {period:5s}: mean={np.mean(gr):.3f}, med={np.median(gr):.3f}, "
              f"<0.30={sum(1 for x in gr if x < 0.30)}/{len(gr)}, "
              f"<0.50={sum(1 for x in gr if x < 0.50)}/{len(gr)}")

# Gate: trades where >30% of gap filled by entry
thresholds = [0.30, 0.50, 0.70]
gate_rows = []
for thresh in thresholds:
    kept = [g for g in gap_at_entry if g['gap_remaining'] >= thresh]
    pre_k = [g['pnl'] for g in kept if g['period'] == 'pre']
    post_k = [g['pnl'] for g in kept if g['period'] == 'post']
    pre_avg = np.mean(pre_k) if pre_k else 0
    post_avg = np.mean(post_k) if post_k else 0
    gate_rows.append([f">={thresh:.0%} remaining", len(pre_k), f"{pre_avg:+.2f}%",
                      len(post_k), f"{post_avg:+.2f}%", f"{post_avg-pre_avg:+.2f}pp"])

    skipped = [g for g in gap_at_entry if g['gap_remaining'] < thresh]
    pre_s = [g['pnl'] for g in skipped if g['period'] == 'pre']
    post_s = [g['pnl'] for g in skipped if g['period'] == 'post']
    pre_avg_s = np.mean(pre_s) if pre_s else 0
    post_avg_s = np.mean(post_s) if post_s else 0
    gate_rows.append([f"<{thresh:.0%} remaining", len(pre_s), f"{pre_avg_s:+.2f}%",
                      len(post_s), f"{post_avg_s:+.2f}%", f"{post_avg_s-pre_avg_s:+.2f}pp"])

print()
print(tabulate(gate_rows, headers=['Gate', 'Pre N', 'Pre Avg', 'Post N', 'Post Avg', 'Δ'], tablefmt='simple'))

# Correlation: gap_remaining vs PnL
for period in ['pre', 'post']:
    subset = [g for g in gap_at_entry if g['period'] == period and g['gap_remaining'] > 0]
    if len(subset) > 5:
        r, p = sp_stats.pearsonr([g['gap_remaining'] for g in subset], [g['pnl'] for g in subset])
        print(f"  {period} gap_remaining→PnL: r={r:+.3f} (p={p:.3f})")

exp_results['exp6'] = {f'gate_{int(t*100)}': {'pre': compute_metrics([g['pnl'] for g in gap_at_entry if g['gap_remaining'] >= t and g['period'] == 'pre']),
                                               'post': compute_metrics([g['pnl'] for g in gap_at_entry if g['gap_remaining'] >= t and g['period'] == 'post'])}
                       for t in thresholds}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 7 — INTRADAY VOLATILITY AS ENTRY GATE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 7 — INTRADAY VOLATILITY ENTRY GATE")
print(f"{'='*80}")

# Compute first-5-bar realized vol for each trade
for t in trades:
    bd = all_bars.get((t['ticker'], t['scan_date']))
    if bd is None:
        t['first5_vol'] = None; continue
    rth = bd.get('rth', [])
    if len(rth) < 5:
        t['first5_vol'] = None; continue
    first5 = rth[:5]
    hi = max(b[1]['h'] for b in first5)
    lo = min(b[1]['l'] for b in first5)
    op = first5[0][1]['o']
    t['first5_vol'] = round((hi - lo) / op * 100, 3) if op > 0 else None

vol_thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
vol_rows = []
for vt in vol_thresholds:
    kept = [t for t in trades if t.get('first5_vol') is not None and t['first5_vol'] >= vt]
    pre_k = [t['pnl_eod'] for t in kept if t['scan_date'] < BREAK_DATE]
    post_k = [t['pnl_eod'] for t in kept if t['scan_date'] >= BREAK_DATE]
    pre_avg = np.mean(pre_k) if pre_k else 0
    post_avg = np.mean(post_k) if post_k else 0
    vol_rows.append([f"≥{vt:.2f}%", len(pre_k), f"{pre_avg:+.2f}%", f"{compute_metrics(pre_k)['wr']*100:.0f}%",
                     len(post_k), f"{post_avg:+.2f}%", f"{compute_metrics(post_k)['wr']*100:.0f}%",
                     f"{post_avg-pre_avg:+.2f}pp"])
print()
print(tabulate(vol_rows, headers=['Vol Gate', 'Pre N', 'Pre Avg', 'Pre WR', 'Post N', 'Post Avg', 'Post WR', 'Δ'],
               tablefmt='simple', stralign='right'))

# Distribution of first5_vol pre vs post
pre_v5 = [t['first5_vol'] for t in pre_trades if t.get('first5_vol') is not None]
post_v5 = [t['first5_vol'] for t in post_trades if t.get('first5_vol') is not None]
if pre_v5 and post_v5:
    print(f"\n  First-5-bar vol distribution:")
    print(f"    Pre:  mean={np.mean(pre_v5):.2f}%, med={np.median(pre_v5):.2f}%")
    print(f"    Post: mean={np.mean(post_v5):.2f}%, med={np.median(post_v5):.2f}%")

exp_results['exp7'] = {f'vol_{int(vt*100)}': {'pre': compute_metrics([t['pnl_eod'] for t in trades if t.get('first5_vol') is not None and t['first5_vol'] >= vt and t['scan_date'] < BREAK_DATE]),
                                               'post': compute_metrics([t['pnl_eod'] for t in trades if t.get('first5_vol') is not None and t['first5_vol'] >= vt and t['scan_date'] >= BREAK_DATE])}
                       for vt in vol_thresholds}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 8 — EXIT OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 8 — EXIT OPTIMIZATION")
print(f"{'='*80}")

# 8a: Fixed exit time grid
print(f"\n  8a — Fixed exit time grid:")
exit_rows = []
for ex_t in EXIT_HOURS:
    key = f'pnl_{ex_t}'
    pre_hp = [t[key] for t in pre_trades if t.get(key) is not None]
    post_hp = [t[key] for t in post_trades if t.get(key) is not None]
    if pre_hp or post_hp:
        pre_avg = np.mean(pre_hp) if pre_hp else 0
        post_avg = np.mean(post_hp) if post_hp else 0
        exit_rows.append([ex_t, len(pre_hp), f"{pre_avg:+.2f}%", f"{compute_metrics(pre_hp)['wr']*100:.0f}%",
                          len(post_hp), f"{post_avg:+.2f}%", f"{compute_metrics(post_hp)['wr']*100:.0f}%",
                          f"{post_avg-pre_avg:+.2f}pp"])
print()
print(tabulate(exit_rows, headers=['Exit Time', 'Pre N', 'Pre Avg', 'Pre WR',
                                    'Post N', 'Post Avg', 'Post WR', 'Δ'],
               tablefmt='simple', stralign='right'))

# 8b: Trailing stop from MFE
print(f"\n  8b — Trailing stop from MFE:")
trail_results = {}
for retrace in [0.25, 0.33, 0.50]:
    trail_pnls = []
    for t in trades:
        bd = all_bars.get((t['ticker'], t['scan_date']))
        if bd is None: continue
        exit_p, exit_t, stopped = find_trailing_exit(bd, t['entry_time'], t['entry_price'], retrace, EXIT_TIME)
        pnl = short_pnl(t['entry_price'], exit_p)
        if pnl is not None:
            trail_pnls.append((pnl, t['scan_date'], stopped))

    pre_t = [p for p, d, s in trail_pnls if d < BREAK_DATE]
    post_t = [p for p, d, s in trail_pnls if d >= BREAK_DATE]
    pre_stopped = sum(1 for p, d, s in trail_pnls if d < BREAK_DATE and s)
    post_stopped = sum(1 for p, d, s in trail_pnls if d >= BREAK_DATE and s)
    pre_avg = np.mean(pre_t) if pre_t else 0
    post_avg = np.mean(post_t) if post_t else 0
    trail_results[retrace] = {'pre': compute_metrics(pre_t), 'post': compute_metrics(post_t)}
    print(f"    Retrace {retrace:.0%}: Pre avg={pre_avg:+.2f}% n={len(pre_t)} stopped={pre_stopped} | "
          f"Post avg={post_avg:+.2f}% n={len(post_t)} stopped={post_stopped} | Δ={post_avg-pre_avg:+.2f}pp")

print(f"\n  Baseline: Pre={BASELINE_PRE:+.2f}%, Post={BASELINE_POST:+.2f}%, Δ={BASELINE_DELTA:+.2f}pp")

exp_results['exp8'] = {
    'exit_times': {ex_t: {'pre': compute_metrics([t[f'pnl_{ex_t}'] for t in pre_trades if t.get(f'pnl_{ex_t}') is not None]),
                          'post': compute_metrics([t[f'pnl_{ex_t}'] for t in post_trades if t.get(f'pnl_{ex_t}') is not None])}
                  for ex_t in EXIT_HOURS},
    'trailing': {f'retrace_{int(r*100)}': trail_results[r] for r in trail_results}
}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 9 — SUSTAINED DROP FILTER AS ENTRY CONDITION
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 9 — SUSTAINED DROP FILTER AS ENTRY CONDITION")
print(f"{'='*80}")

sd_results = {}
for check_bar in [5, 6, 7]:
    sd_trades = []
    for dt, pool in candidate_pool.items():
        for c, bd in pool:
            rth = bd.get('rth', [])
            if not detect_sd_early(rth, check_bar): continue
            entry = find_local_high(rth, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                                    fallback_bar=FALLBACK_BAR, min_entry_bar=min_eb)
            if entry is None: continue
            if entry['entry_bar'] < check_bar: continue
            exit_p = find_exit_price(bd, entry['entry_time'], EXIT_TIME)
            pnl = short_pnl(entry['entry_price'], exit_p)
            if pnl is not None:
                sd_trades.append({'pnl': pnl, 'scan_date': dt, 'ticker': c['ticker']})

    pre_sd, post_sd = eval_split([(t['pnl'], t['scan_date']) for t in sd_trades], f'SD@bar{check_bar}')
    sd_results[check_bar] = {'pre': pre_sd, 'post': post_sd}
    delta = post_sd['avg'] - pre_sd['avg']
    print(f"  SD confirm at bar {check_bar}: Pre avg={pre_sd['avg']:+.2f}% n={pre_sd['n']} | "
          f"Post avg={post_sd['avg']:+.2f}% n={post_sd['n']} | Δ={delta:+.2f}pp | vs baseline: {delta-BASELINE_DELTA:+.2f}pp")

# Compare to base pattern classification
sd_base = [t for t in trades if t.get('open_pattern') == 'sustained_drop']
pre_sd_base = [t['pnl_eod'] for t in sd_base if t['scan_date'] < BREAK_DATE]
post_sd_base = [t['pnl_eod'] for t in sd_base if t['scan_date'] >= BREAK_DATE]
print(f"\n  Base SD filter (bar 60 classification): Pre avg={np.mean(pre_sd_base):+.2f}% n={len(pre_sd_base)} | "
      f"Post avg={np.mean(post_sd_base):+.2f}% n={len(post_sd_base)}")

exp_results['exp9'] = {f'sd_bar{b}': {'pre': sd_results[b]['pre'], 'post': sd_results[b]['post']} for b in sd_results}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 10 — RE-RANK QUALITY AT 09:31 VS PRE-MARKET
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 10 — RE-RANK QUALITY AUDIT")
print(f"{'='*80}")

rerank_audit = {'reranked_better': 0, 'reranked_worse': 0, 'same': 0}
rerank_trades_pm = []
for dt in dates:
    cands_raw = all_candidates.get(dt, [])
    if not cands_raw: continue

    # Pre-market rank (by gap_pct_daily or gap_pct)
    pm_ranked = sorted(cands_raw, key=lambda x: -x.get('gap_pct_daily', x['gap_pct']))
    pm_cs = [c for c in pm_ranked[:PRE_FILTER_N] if _ticker_ref.get(c['ticker'], {}).get('type') == 'CS'][:TOP_N]

    # Post-rerank (current order in all_candidates is already re-ranked)
    rr_cs = [c for c in cands_raw[:PRE_FILTER_N] if _ticker_ref.get(c['ticker'], {}).get('type') == 'CS'][:TOP_N]

    pm_tickers = set(c['ticker'] for c in pm_cs)
    rr_tickers = set(c['ticker'] for c in rr_cs)

    if pm_tickers == rr_tickers:
        rerank_audit['same'] += 1
    else:
        # Build trades with PM rank
        for c in pm_cs:
            bd = all_bars.get((c['ticker'], dt))
            if bd is None: continue
            rth = bd.get('rth', [])
            entry = find_local_high(rth, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                                    fallback_bar=FALLBACK_BAR, min_entry_bar=min_eb) if rth and len(rth) >= PEAK_WINDOW * 2 + 1 else None
            if entry is None: continue
            exit_p = find_exit_price(bd, entry['entry_time'], EXIT_TIME)
            pnl = short_pnl(entry['entry_price'], exit_p)
            if pnl is not None:
                rerank_trades_pm.append({'pnl': pnl, 'scan_date': dt, 'ticker': c['ticker']})

pm_pre, pm_post = eval_split([(t['pnl'], t['scan_date']) for t in rerank_trades_pm], 'PM-Rank')
print(f"\n  Pre-market rank (no re-rank at 09:31):")
print(f"    Pre:  avg={pm_pre['avg']:+.2f}%, n={pm_pre['n']}, WR={pm_pre['wr']*100:.0f}%")
print(f"    Post: avg={pm_post['avg']:+.2f}%, n={pm_post['n']}, WR={pm_post['wr']*100:.0f}%")
print(f"    Δ={pm_post['avg']-pm_pre['avg']:+.2f}pp")

print(f"\n  Current (with re-rank at 09:31):")
print(f"    Pre:  avg={BASELINE_PRE:+.2f}%, Post: avg={BASELINE_POST:+.2f}%, Δ={BASELINE_DELTA:+.2f}pp")

rr_delta = pm_post['avg'] - pm_pre['avg']
print(f"\n  Re-rank impact: {'HELPS' if BASELINE_DELTA > rr_delta else 'HURTS'} post-Aug "
      f"(baseline Δ={BASELINE_DELTA:+.2f}pp vs PM Δ={rr_delta:+.2f}pp)")

exp_results['exp10'] = {'pm_rank': {'pre': pm_pre, 'post': pm_post},
                        'rerank_stats': rerank_audit}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 11 — PARTIAL ENTRY (SCALE IN OVER FIRST 3 BARS)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 11 — PARTIAL ENTRY (3-TRANCHE SCALE-IN)")
print(f"{'='*80}")

partial_trades = []
for dt, pool in candidate_pool.items():
    for c, bd in pool:
        rth = bd.get('rth', [])
        if not rth or len(rth) < 10: continue
        open_price = rth[0][1]['o']
        if open_price <= 0: continue

        entries = []
        # Tranche 1: local_high or bar 3
        lh = find_local_high(rth, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                             fallback_bar=None, min_entry_bar=min_eb)
        if lh:
            entries.append((lh['entry_price'], lh['entry_time']))
        elif len(rth) > 3:
            entries.append((rth[3][1]['c'], rth[3][0]))

        # Tranche 2: first 1% confirmation drop from open
        for i in range(1, min(30, len(rth))):
            if rth[i][1]['c'] <= open_price * 0.99:
                entries.append((rth[i][1]['c'], rth[i][0]))
                break

        # Tranche 3: VWAP cross
        vwap_e = find_vwap_entry(rth, min_bar=1, max_bar=30)
        if vwap_e:
            entries.append((vwap_e['entry_price'], vwap_e['entry_time']))

        if not entries: continue
        avg_entry = np.mean([e[0] for e in entries])
        latest_time = max(e[1] for e in entries)
        exit_p = find_exit_price(bd, latest_time, EXIT_TIME)
        pnl = short_pnl(avg_entry, exit_p)
        if pnl is not None:
            partial_trades.append({'pnl': pnl, 'scan_date': dt, 'ticker': c['ticker'],
                                   'n_tranches': len(entries)})

part_pre, part_post = eval_split([(t['pnl'], t['scan_date']) for t in partial_trades], 'Partial')
delta_p = part_post['avg'] - part_pre['avg']
print(f"\n  3-tranche scale-in (LH/bar3 + 1% drop + VWAP cross):")
print(f"    Pre:  avg={part_pre['avg']:+.2f}%, n={part_pre['n']}, WR={part_pre['wr']*100:.0f}%")
print(f"    Post: avg={part_post['avg']:+.2f}%, n={part_post['n']}, WR={part_post['wr']*100:.0f}%")
print(f"    Δ={delta_p:+.2f}pp  |  vs baseline: {delta_p-BASELINE_DELTA:+.2f}pp")

tranche_dist = defaultdict(int)
for t in partial_trades:
    tranche_dist[t['n_tranches']] += 1
print(f"    Tranche fill: {dict(tranche_dist)}")

exp_results['exp11'] = {'pre': part_pre, 'post': part_post}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 12 — SPY INTRADAY DIRECTION AS ENTRY FILTER
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 12 — SPY INTRADAY DIRECTION FILTER")
print(f"{'='*80}")

spy_dir_trades = {'spy_down': [], 'spy_up': [], 'no_spy': []}
for t in trades:
    spy_bd = spy_bars_cache.get(('SPY', t['scan_date']))
    if spy_bd is None:
        spy_dir_trades['no_spy'].append(t)
        continue
    spy_rth = spy_bd.get('rth', [])
    if not spy_rth:
        spy_dir_trades['no_spy'].append(t)
        continue

    spy_open = spy_rth[0][1]['o']
    entry_time = t['entry_time']
    spy_at_entry = None
    for ts, bar in spy_rth:
        if ts > entry_time: break
        spy_at_entry = bar['c']

    if spy_at_entry is None:
        spy_dir_trades['no_spy'].append(t)
    elif spy_at_entry <= spy_open:
        spy_dir_trades['spy_down'].append(t)
    else:
        spy_dir_trades['spy_up'].append(t)

print(f"\n  Distribution: SPY down/flat={len(spy_dir_trades['spy_down'])}, "
      f"SPY up={len(spy_dir_trades['spy_up'])}, no data={len(spy_dir_trades['no_spy'])}")

for label, subset in spy_dir_trades.items():
    if not subset: continue
    pre_s = [t['pnl_eod'] for t in subset if t['scan_date'] < BREAK_DATE]
    post_s = [t['pnl_eod'] for t in subset if t['scan_date'] >= BREAK_DATE]
    pre_avg = np.mean(pre_s) if pre_s else 0
    post_avg = np.mean(post_s) if post_s else 0
    print(f"    {label:10s}: Pre avg={pre_avg:+.2f}% n={len(pre_s)} | Post avg={post_avg:+.2f}% n={len(post_s)} | Δ={post_avg-pre_avg:+.2f}pp")

# Also test: first 15 min SPY direction
spy_15m_trades = {'spy_15m_down': [], 'spy_15m_up': []}
for t in trades:
    spy_bd = spy_bars_cache.get(('SPY', t['scan_date']))
    if spy_bd is None: continue
    spy_rth = spy_bd.get('rth', [])
    if len(spy_rth) < 15: continue
    spy_open = spy_rth[0][1]['o']
    spy_15 = spy_rth[14][1]['c']
    if spy_15 <= spy_open:
        spy_15m_trades['spy_15m_down'].append(t)
    else:
        spy_15m_trades['spy_15m_up'].append(t)

print(f"\n  SPY first 15 min direction:")
for label, subset in spy_15m_trades.items():
    if not subset: continue
    pre_s = [t['pnl_eod'] for t in subset if t['scan_date'] < BREAK_DATE]
    post_s = [t['pnl_eod'] for t in subset if t['scan_date'] >= BREAK_DATE]
    pre_avg = np.mean(pre_s) if pre_s else 0
    post_avg = np.mean(post_s) if post_s else 0
    print(f"    {label:16s}: Pre avg={pre_avg:+.2f}% n={len(pre_s)} | Post avg={post_avg:+.2f}% n={len(post_s)} | Δ={post_avg-pre_avg:+.2f}pp")

exp_results['exp12'] = {k: {'pre': compute_metrics([t['pnl_eod'] for t in v if t['scan_date'] < BREAK_DATE]),
                            'post': compute_metrics([t['pnl_eod'] for t in v if t['scan_date'] >= BREAK_DATE])}
                        for k, v in spy_dir_trades.items() if v}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 13 — ATR-NORMALIZED POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 13 — ATR-NORMALIZED POSITION SIZING")
print(f"{'='*80}")

TARGET_RISK = 2000  # target $2000 risk per trade
atr_trades = []
for t in trades:
    atr = atr_cache.get((t['ticker'], t['scan_date']))
    if atr is None or atr <= 0: continue
    atr_pct = atr / t['entry_price'] * 100 if t['entry_price'] > 0 else 0
    shares = TARGET_RISK / atr if atr > 0 else 0
    pos_size = shares * t['entry_price']
    dollar_pnl = pos_size * (t['pnl_eod'] / 100) if t['pnl_eod'] is not None else 0

    atr_trades.append({
        'pnl_pct': t['pnl_eod'], 'dollar_pnl': dollar_pnl,
        'pos_size': pos_size, 'atr': atr, 'atr_pct': atr_pct,
        'date': t['scan_date'], 'ticker': t['ticker'],
        'period': 'pre' if t['scan_date'] < BREAK_DATE else 'post'
    })

print(f"\n  ATR-sized trades: {len(atr_trades)} / {len(trades)} (have ATR14)")

# Compare fixed sizing vs ATR sizing
for period in ['pre', 'post']:
    subset = [t for t in atr_trades if t['period'] == period]
    if subset:
        avg_pos = np.mean([t['pos_size'] for t in subset])
        avg_atr = np.mean([t['atr_pct'] for t in subset])
        total_dollar = sum(t['dollar_pnl'] for t in subset)
        # Fixed sizing baseline (same trades)
        fixed_dollar = sum((CAPITAL / 5) * (t['pnl_pct'] / 100) for t in subset)
        print(f"  {period:5s}: ATR avg pos=${avg_pos:,.0f}, avg ATR%={avg_atr:.2f}%, "
              f"ATR total $={total_dollar:+,.0f} vs Fixed $={fixed_dollar:+,.0f}")

# ATR quartile analysis
if atr_trades:
    atr_pcts = [t['atr_pct'] for t in atr_trades]
    q25, q50, q75 = np.percentile(atr_pcts, [25, 50, 75])
    print(f"\n  ATR% quartiles: Q25={q25:.2f}%, Q50={q50:.2f}%, Q75={q75:.2f}%")

    atr_buckets = {'low_atr': [], 'mid_atr': [], 'high_atr': []}
    for t in atr_trades:
        if t['atr_pct'] <= q25: atr_buckets['low_atr'].append(t)
        elif t['atr_pct'] <= q75: atr_buckets['mid_atr'].append(t)
        else: atr_buckets['high_atr'].append(t)

    atr_rows = []
    for bk in ['low_atr', 'mid_atr', 'high_atr']:
        pre_b = [t['pnl_pct'] for t in atr_buckets[bk] if t['period'] == 'pre']
        post_b = [t['pnl_pct'] for t in atr_buckets[bk] if t['period'] == 'post']
        atr_rows.append([bk, len(pre_b), f"{np.mean(pre_b):+.2f}%" if pre_b else "—",
                         len(post_b), f"{np.mean(post_b):+.2f}%" if post_b else "—"])
    print()
    print(tabulate(atr_rows, headers=['ATR Bucket', 'Pre N', 'Pre Avg', 'Post N', 'Post Avg'], tablefmt='simple'))

exp_results['exp13'] = {
    'atr_pre_dollar': float(sum(t['dollar_pnl'] for t in atr_trades if t['period'] == 'pre')),
    'atr_post_dollar': float(sum(t['dollar_pnl'] for t in atr_trades if t['period'] == 'post')),
    'fixed_pre_dollar': float(sum((CAPITAL/5) * (t['pnl_pct']/100) for t in atr_trades if t['period'] == 'pre')),
    'fixed_post_dollar': float(sum((CAPITAL/5) * (t['pnl_pct']/100) for t in atr_trades if t['period'] == 'post')),
}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 14 — GAP BUCKET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 14 — GAP BUCKET ANALYSIS (HIGH VS LOW GAP)")
print(f"{'='*80}")

gap_buckets = {'<15%': [], '15-30%': [], '>30%': []}
for t in trades:
    gp = t['gap_pct'] * 100
    if gp < 15: gap_buckets['<15%'].append(t)
    elif gp <= 30: gap_buckets['15-30%'].append(t)
    else: gap_buckets['>30%'].append(t)

gap_rows = []
for bk_label in ['<15%', '15-30%', '>30%']:
    subset = gap_buckets[bk_label]
    pre_s = [t['pnl_eod'] for t in subset if t['scan_date'] < BREAK_DATE]
    post_s = [t['pnl_eod'] for t in subset if t['scan_date'] >= BREAK_DATE]
    pre_avg = np.mean(pre_s) if pre_s else 0
    post_avg = np.mean(post_s) if post_s else 0
    gap_rows.append([bk_label, len(pre_s), f"{pre_avg:+.2f}%", f"{compute_metrics(pre_s)['wr']*100:.0f}%",
                     len(post_s), f"{post_avg:+.2f}%", f"{compute_metrics(post_s)['wr']*100:.0f}%",
                     f"{post_avg-pre_avg:+.2f}pp"])
print()
print(tabulate(gap_rows, headers=['Gap Bucket', 'Pre N', 'Pre Avg', 'Pre WR', 'Post N', 'Post Avg', 'Post WR', 'Δ'],
               tablefmt='simple', stralign='right'))

# Entry type breakdown within each gap bucket
print(f"\n  Entry type by gap bucket:")
for bk_label in ['<15%', '15-30%', '>30%']:
    subset = gap_buckets[bk_label]
    for et in ['local_high', 'fallback']:
        pre_et = [t['pnl_eod'] for t in subset if t['entry_type'] == et and t['scan_date'] < BREAK_DATE]
        post_et = [t['pnl_eod'] for t in subset if t['entry_type'] == et and t['scan_date'] >= BREAK_DATE]
        if pre_et or post_et:
            pre_avg = np.mean(pre_et) if pre_et else 0
            post_avg = np.mean(post_et) if post_et else 0
            print(f"    {bk_label:6s} {et:12s}: Pre avg={pre_avg:+.2f}% n={len(pre_et)} | Post avg={post_avg:+.2f}% n={len(post_et)}")

# Fallback collapse by gap bucket
print(f"\n  Fallback PnL collapse by gap bucket:")
for bk_label in ['<15%', '15-30%', '>30%']:
    subset = gap_buckets[bk_label]
    pre_fb = [t['pnl_eod'] for t in subset if t['entry_type'] == 'fallback' and t['scan_date'] < BREAK_DATE]
    post_fb = [t['pnl_eod'] for t in subset if t['entry_type'] == 'fallback' and t['scan_date'] >= BREAK_DATE]
    if pre_fb and post_fb:
        print(f"    {bk_label:6s}: Pre FB avg={np.mean(pre_fb):+.2f}% n={len(pre_fb)} → Post FB avg={np.mean(post_fb):+.2f}% n={len(post_fb)} | Δ={np.mean(post_fb)-np.mean(pre_fb):+.2f}pp")
    elif pre_fb:
        print(f"    {bk_label:6s}: Pre FB avg={np.mean(pre_fb):+.2f}% n={len(pre_fb)} → Post: n=0")

exp_results['exp14'] = {bk: {'pre': compute_metrics([t['pnl_eod'] for t in gap_buckets[bk] if t['scan_date'] < BREAK_DATE]),
                             'post': compute_metrics([t['pnl_eod'] for t in gap_buckets[bk] if t['scan_date'] >= BREAK_DATE])}
                        for bk in gap_buckets}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 15 — STOP LOSS CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  EXP 15 — STOP LOSS CALIBRATION")
print(f"{'='*80}")

# 15a: Fixed % SL
print(f"\n  15a — Fixed % stop loss:")
sl_fixed_results = {}
for sl_pct in [2, 3, 5, 7, 10]:
    sl_pnls = []
    for t in trades:
        bd = all_bars.get((t['ticker'], t['scan_date']))
        if bd is None: continue
        exit_p, exit_t, stopped = find_sl_exit(bd, t['entry_time'], t['entry_price'], sl_pct, EXIT_TIME)
        pnl = short_pnl(t['entry_price'], exit_p)
        if pnl is not None:
            sl_pnls.append((pnl, t['scan_date'], stopped))

    pre_sl = [p for p, d, s in sl_pnls if d < BREAK_DATE]
    post_sl = [p for p, d, s in sl_pnls if d >= BREAK_DATE]
    pre_stopped = sum(1 for p, d, s in sl_pnls if d < BREAK_DATE and s)
    post_stopped = sum(1 for p, d, s in sl_pnls if d >= BREAK_DATE and s)
    pre_avg = np.mean(pre_sl) if pre_sl else 0
    post_avg = np.mean(post_sl) if post_sl else 0
    sl_fixed_results[sl_pct] = {'pre': compute_metrics(pre_sl), 'post': compute_metrics(post_sl)}
    print(f"    SL={sl_pct}%: Pre avg={pre_avg:+.2f}% n={len(pre_sl)} stopped={pre_stopped} | "
          f"Post avg={post_avg:+.2f}% n={len(post_sl)} stopped={post_stopped} | Δ={post_avg-pre_avg:+.2f}pp")

# 15b: ATR-normalized SL
print(f"\n  15b — ATR-normalized stop loss:")
sl_atr_results = {}
for k in [0.25, 0.50, 0.75, 1.0]:
    sl_pnls = []
    for t in trades:
        atr = atr_cache.get((t['ticker'], t['scan_date']))
        if atr is None or atr <= 0 or t['entry_price'] <= 0: continue
        sl_pct_dynamic = k * atr / t['entry_price'] * 100
        bd = all_bars.get((t['ticker'], t['scan_date']))
        if bd is None: continue
        exit_p, exit_t, stopped = find_sl_exit(bd, t['entry_time'], t['entry_price'], sl_pct_dynamic, EXIT_TIME)
        pnl = short_pnl(t['entry_price'], exit_p)
        if pnl is not None:
            sl_pnls.append((pnl, t['scan_date'], stopped))

    pre_sl = [p for p, d, s in sl_pnls if d < BREAK_DATE]
    post_sl = [p for p, d, s in sl_pnls if d >= BREAK_DATE]
    pre_stopped = sum(1 for p, d, s in sl_pnls if d < BREAK_DATE and s)
    post_stopped = sum(1 for p, d, s in sl_pnls if d >= BREAK_DATE and s)
    pre_avg = np.mean(pre_sl) if pre_sl else 0
    post_avg = np.mean(post_sl) if post_sl else 0
    sl_atr_results[k] = {'pre': compute_metrics(pre_sl), 'post': compute_metrics(post_sl)}
    print(f"    SL={k:.2f}×ATR: Pre avg={pre_avg:+.2f}% n={len(pre_sl)} stopped={pre_stopped} | "
          f"Post avg={post_avg:+.2f}% n={len(post_sl)} stopped={post_stopped} | Δ={post_avg-pre_avg:+.2f}pp")

print(f"\n  Baseline (no SL): Pre={BASELINE_PRE:+.2f}%, Post={BASELINE_POST:+.2f}%, Δ={BASELINE_DELTA:+.2f}pp")

# Cross-reference with MFE/MAE
pre_mfe_vals = [t['mfe'] for t in pre_trades if t.get('mfe') is not None]
post_mfe_vals = [t['mfe'] for t in post_trades if t.get('mfe') is not None]
pre_mae_vals = [t['mae'] for t in pre_trades if t.get('mae') is not None]
post_mae_vals = [t['mae'] for t in post_trades if t.get('mae') is not None]
print(f"\n  MFE/MAE reference:")
print(f"    Pre:  MFE mean={np.mean(pre_mfe_vals):+.2f}%, MAE mean={np.mean(pre_mae_vals):+.2f}%")
print(f"    Post: MFE mean={np.mean(post_mfe_vals):+.2f}%, MAE mean={np.mean(post_mae_vals):+.2f}%")

exp_results['exp15'] = {
    'fixed': {f'sl_{s}': {'pre': sl_fixed_results[s]['pre'], 'post': sl_fixed_results[s]['post']} for s in sl_fixed_results},
    'atr': {f'sl_{int(k*100)}atr': {'pre': sl_atr_results[k]['pre'], 'post': sl_atr_results[k]['post']} for k in sl_atr_results}
}


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*110}")
print(f"  SUMMARY — ALL 15 EXPERIMENTS")
print(f"  Baseline: Pre={BASELINE_PRE:+.2f}%, Post={BASELINE_POST:+.2f}%, Δ={BASELINE_DELTA:+.2f}pp")
print(f"{'='*110}")

def _best_variant(exp_data):
    if isinstance(exp_data, dict) and 'pre' in exp_data and 'post' in exp_data:
        return exp_data['pre'], exp_data['post']
    best = None
    best_post = -999
    for k, v in exp_data.items():
        if isinstance(v, dict) and 'pre' in v and 'post' in v:
            if v['post'].get('avg', v['post'].get('avg', 0)) > best_post:
                best = (v['pre'], v['post'], k)
                best_post = v['post'].get('avg', 0)
    if best:
        return best[0], best[1]
    return None, None

summary_rows = []
exp_labels = [
    ('1', 'Local High Audit', 'exp1'),
    ('2', 'ORB Entry', 'exp2'),
    ('3', 'VWAP Entry', 'exp3'),
    ('4', 'Volume Entry Quality', 'exp4'),
    ('5', 'Fallback Timing Grid', 'exp5'),
    ('6', 'Gap at Entry Gate', 'exp6'),
    ('7', 'Intraday Vol Gate', 'exp7'),
    ('8', 'Exit Optimization', 'exp8'),
    ('9', 'SD Filter Entry', 'exp9'),
    ('10', 'Re-rank Quality', 'exp10'),
    ('11', 'Partial Entry', 'exp11'),
    ('12', 'SPY Direction Filter', 'exp12'),
    ('13', 'ATR Position Sizing', 'exp13'),
    ('14', 'Gap Bucket Analysis', 'exp14'),
    ('15', 'Stop Loss Calibration', 'exp15'),
]

for exp_id, exp_name, exp_key in exp_labels:
    data = exp_results.get(exp_key, {})
    pre, post = _best_variant(data)
    if pre is None or post is None:
        summary_rows.append([exp_id, exp_name, '—', '—', '—', '—', '—'])
        continue
    pre_avg = pre.get('avg', 0)
    post_avg = post.get('avg', 0)
    delta = post_avg - pre_avg
    delta_vs_bl = delta - BASELINE_DELTA
    narrows = 'YES' if abs(delta) < abs(BASELINE_DELTA) and delta * BASELINE_DELTA > 0 else 'NO'
    improves_post = 'YES' if post_avg > BASELINE_POST else 'NO'
    summary_rows.append([
        exp_id, exp_name[:22],
        f"{pre_avg:+.2f}%", f"{post_avg:+.2f}%", f"{delta:+.2f}pp",
        narrows, improves_post
    ])

print()
print(tabulate(summary_rows,
               headers=['Exp', 'Name', 'Best Pre', 'Best Post', 'Δ', 'Narrows Gap?', 'Improves Post?'],
               tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

# Chart 1: Entry audit + fallback grid
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Entry Mechanics Investigation — Part 1', fontsize=12, fontweight='bold')

# 1.1: Local high fire type PnL
ax = axes[0, 0]
fire_types = ['clean', 'late', 'fallback']
x_ft = np.arange(len(fire_types))
pre_ft_avg = []
post_ft_avg = []
for ft in fire_types:
    pre_r = [r['pnl_eod'] for r in lh_audit[ft] if r['period'] == 'pre']
    post_r = [r['pnl_eod'] for r in lh_audit[ft] if r['period'] == 'post']
    pre_ft_avg.append(np.mean(pre_r) if pre_r else 0)
    post_ft_avg.append(np.mean(post_r) if post_r else 0)
w = 0.35
ax.bar(x_ft - w/2, pre_ft_avg, w, color='#2c3e50', alpha=0.8, label='Pre-Aug')
ax.bar(x_ft + w/2, post_ft_avg, w, color='#e74c3c', alpha=0.8, label='Post-Aug')
ax.set_xticks(x_ft); ax.set_xticklabels(fire_types)
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Exp 1: PnL by Fire Type'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylabel('Avg PnL %')

# 1.2: Fallback timing grid
ax = axes[0, 1]
fb_bars = [3, 5, 7, 9, 11, 13, 15, 19]
x_fb = np.arange(len(fb_bars))
pre_fb_avgs = [fb_grid_results[b]['all_pre']['avg'] for b in fb_bars]
post_fb_avgs = [fb_grid_results[b]['all_post']['avg'] for b in fb_bars]
ax.plot(x_fb, pre_fb_avgs, 'o-', color='#2c3e50', linewidth=2, label='Pre-Aug')
ax.plot(x_fb, post_fb_avgs, 's-', color='#e74c3c', linewidth=2, label='Post-Aug')
ax.set_xticks(x_fb); ax.set_xticklabels([f'FB={b}' for b in fb_bars], rotation=30)
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Exp 5: Fallback Timing Grid'); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylabel('Avg PnL %')

# 1.3: ORB entry vs baseline
ax = axes[1, 0]
orb_ns = [1, 2, 3, 5]
x_orb = np.arange(len(orb_ns))
pre_orb = [orb_results[n]['pre']['avg'] for n in orb_ns]
post_orb = [orb_results[n]['post']['avg'] for n in orb_ns]
ax.bar(x_orb - w/2, pre_orb, w, color='#2c3e50', alpha=0.8, label='Pre-Aug')
ax.bar(x_orb + w/2, post_orb, w, color='#e74c3c', alpha=0.8, label='Post-Aug')
ax.axhline(BASELINE_PRE, color='#2c3e50', lw=1, ls='--', alpha=0.5)
ax.axhline(BASELINE_POST, color='#e74c3c', lw=1, ls='--', alpha=0.5)
ax.set_xticks(x_orb); ax.set_xticklabels([f'ORB-{n}' for n in orb_ns])
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Exp 2: ORB Entry by Range Width'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylabel('Avg PnL %')

# 1.4: Vol gate sweep
ax = axes[1, 1]
vt_labels = vol_thresholds
vt_pre = [exp_results['exp7'][f'vol_{int(vt*100)}']['pre']['avg'] for vt in vt_labels]
vt_post = [exp_results['exp7'][f'vol_{int(vt*100)}']['post']['avg'] for vt in vt_labels]
vt_pre_n = [exp_results['exp7'][f'vol_{int(vt*100)}']['pre']['n'] for vt in vt_labels]
ax.plot(range(len(vt_labels)), vt_pre, 'o-', color='#2c3e50', linewidth=2, label='Pre-Aug')
ax.plot(range(len(vt_labels)), vt_post, 's-', color='#e74c3c', linewidth=2, label='Post-Aug')
ax2 = ax.twinx()
ax2.bar(range(len(vt_labels)), vt_pre_n, alpha=0.15, color='#2c3e50', label='Pre N')
ax2.set_ylabel('Trade Count', color='#95a5a6')
ax.set_xticks(range(len(vt_labels))); ax.set_xticklabels([f'≥{vt}%' for vt in vt_labels], rotation=30)
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Exp 7: Intraday Vol Gate Sweep'); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylabel('Avg PnL %')

plt.tight_layout(rect=[0, 0, 1, 0.94])
p1 = os.path.join(BASE_DIR, 'entry_mech_part1.png')
plt.savefig(p1, dpi=150, bbox_inches='tight'); plt.show()
print(f"  Saved: {p1}")

# Chart 2: Exit + filters
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Entry Mechanics Investigation — Part 2', fontsize=12, fontweight='bold')

# 2.1: Exit time curves
ax = axes[0, 0]
pre_exit_avgs = []
post_exit_avgs = []
for ex_t in EXIT_HOURS:
    key = f'pnl_{ex_t}'
    pre_hp = [t[key] for t in pre_trades if t.get(key) is not None]
    post_hp = [t[key] for t in post_trades if t.get(key) is not None]
    pre_exit_avgs.append(np.mean(pre_hp) if pre_hp else np.nan)
    post_exit_avgs.append(np.mean(post_hp) if post_hp else np.nan)
ax.plot(range(len(EXIT_HOURS)), pre_exit_avgs, 'o-', color='#2c3e50', linewidth=2, label='Pre-Aug')
ax.plot(range(len(EXIT_HOURS)), post_exit_avgs, 's-', color='#e74c3c', linewidth=2, label='Post-Aug')
ax.set_xticks(range(len(EXIT_HOURS))); ax.set_xticklabels(EXIT_HOURS, rotation=45)
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Exp 8a: Exit Time Curve'); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylabel('Avg PnL %')

# 2.2: Trailing stop results
ax = axes[0, 1]
trail_labels = ['25%', '33%', '50%', 'None']
trail_pre_vals = [trail_results[0.25]['pre']['avg'], trail_results[0.33]['pre']['avg'],
                  trail_results[0.50]['pre']['avg'], BASELINE_PRE]
trail_post_vals = [trail_results[0.25]['post']['avg'], trail_results[0.33]['post']['avg'],
                   trail_results[0.50]['post']['avg'], BASELINE_POST]
x_tr = np.arange(len(trail_labels))
ax.bar(x_tr - w/2, trail_pre_vals, w, color='#2c3e50', alpha=0.8, label='Pre-Aug')
ax.bar(x_tr + w/2, trail_post_vals, w, color='#e74c3c', alpha=0.8, label='Post-Aug')
ax.set_xticks(x_tr); ax.set_xticklabels([f'Trail {l}' for l in trail_labels])
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Exp 8b: Trailing Stop'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylabel('Avg PnL %')

# 2.3: SPY direction filter
ax = axes[1, 0]
spy_labels = [k for k in spy_dir_trades if spy_dir_trades[k]]
spy_pre_vals = []
spy_post_vals = []
for k in spy_labels:
    pre_s = [t['pnl_eod'] for t in spy_dir_trades[k] if t['scan_date'] < BREAK_DATE]
    post_s = [t['pnl_eod'] for t in spy_dir_trades[k] if t['scan_date'] >= BREAK_DATE]
    spy_pre_vals.append(np.mean(pre_s) if pre_s else 0)
    spy_post_vals.append(np.mean(post_s) if post_s else 0)
x_spy = np.arange(len(spy_labels))
ax.bar(x_spy - w/2, spy_pre_vals, w, color='#2c3e50', alpha=0.8, label='Pre-Aug')
ax.bar(x_spy + w/2, spy_post_vals, w, color='#e74c3c', alpha=0.8, label='Post-Aug')
ax.set_xticks(x_spy); ax.set_xticklabels(spy_labels, rotation=20)
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Exp 12: SPY Direction Filter'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylabel('Avg PnL %')

# 2.4: Gap bucket analysis
ax = axes[1, 1]
gap_labels = ['<15%', '15-30%', '>30%']
x_gap = np.arange(len(gap_labels))
gap_pre_vals = [exp_results['exp14'][bk]['pre']['avg'] for bk in gap_labels]
gap_post_vals = [exp_results['exp14'][bk]['post']['avg'] for bk in gap_labels]
ax.bar(x_gap - w/2, gap_pre_vals, w, color='#2c3e50', alpha=0.8, label='Pre-Aug')
ax.bar(x_gap + w/2, gap_post_vals, w, color='#e74c3c', alpha=0.8, label='Post-Aug')
ax.set_xticks(x_gap); ax.set_xticklabels(gap_labels)
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Exp 14: Gap Bucket Analysis'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylabel('Avg PnL %')

plt.tight_layout(rect=[0, 0, 1, 0.94])
p2 = os.path.join(BASE_DIR, 'entry_mech_part2.png')
plt.savefig(p2, dpi=150, bbox_inches='tight'); plt.show()
print(f"  Saved: {p2}")

# Chart 3: SL calibration + summary
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Entry Mechanics Investigation — Part 3', fontsize=12, fontweight='bold')

# 3.1: Fixed SL sweep
ax = axes[0, 0]
sl_pcts = [2, 3, 5, 7, 10]
sl_pre_vals = [sl_fixed_results[s]['pre']['avg'] for s in sl_pcts]
sl_post_vals = [sl_fixed_results[s]['post']['avg'] for s in sl_pcts]
x_sl = np.arange(len(sl_pcts))
ax.plot(x_sl, sl_pre_vals, 'o-', color='#2c3e50', linewidth=2, label='Pre-Aug')
ax.plot(x_sl, sl_post_vals, 's-', color='#e74c3c', linewidth=2, label='Post-Aug')
ax.axhline(BASELINE_PRE, color='#2c3e50', lw=1, ls='--', alpha=0.5, label=f'BL Pre ({BASELINE_PRE:+.1f}%)')
ax.axhline(BASELINE_POST, color='#e74c3c', lw=1, ls='--', alpha=0.5, label=f'BL Post ({BASELINE_POST:+.1f}%)')
ax.set_xticks(x_sl); ax.set_xticklabels([f'SL={s}%' for s in sl_pcts])
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Exp 15a: Fixed % Stop Loss'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
ax.set_ylabel('Avg PnL %')

# 3.2: ATR SL sweep
ax = axes[0, 1]
atr_ks = [0.25, 0.50, 0.75, 1.0]
atr_sl_pre = [sl_atr_results[k]['pre']['avg'] for k in atr_ks]
atr_sl_post = [sl_atr_results[k]['post']['avg'] for k in atr_ks]
x_atr = np.arange(len(atr_ks))
ax.plot(x_atr, atr_sl_pre, 'o-', color='#2c3e50', linewidth=2, label='Pre-Aug')
ax.plot(x_atr, atr_sl_post, 's-', color='#e74c3c', linewidth=2, label='Post-Aug')
ax.axhline(BASELINE_PRE, color='#2c3e50', lw=1, ls='--', alpha=0.5)
ax.axhline(BASELINE_POST, color='#e74c3c', lw=1, ls='--', alpha=0.5)
ax.set_xticks(x_atr); ax.set_xticklabels([f'{k}×ATR' for k in atr_ks])
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Exp 15b: ATR-Normalized Stop Loss'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
ax.set_ylabel('Avg PnL %')

# 3.3: SD filter timing
ax = axes[1, 0]
sd_bars_tested = [5, 6, 7]
x_sd = np.arange(len(sd_bars_tested))
sd_pre_vals = [sd_results[b]['pre']['avg'] for b in sd_bars_tested]
sd_post_vals = [sd_results[b]['post']['avg'] for b in sd_bars_tested]
ax.bar(x_sd - w/2, sd_pre_vals, w, color='#2c3e50', alpha=0.8, label='Pre-Aug')
ax.bar(x_sd + w/2, sd_post_vals, w, color='#e74c3c', alpha=0.8, label='Post-Aug')
ax.axhline(BASELINE_PRE, color='#2c3e50', lw=1, ls='--', alpha=0.5)
ax.axhline(BASELINE_POST, color='#e74c3c', lw=1, ls='--', alpha=0.5)
ax.set_xticks(x_sd); ax.set_xticklabels([f'SD@bar{b}' for b in sd_bars_tested])
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Exp 9: SD Filter Entry Timing'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylabel('Avg PnL %')

# 3.4: Summary — post-Aug avg PnL of best variant per experiment
ax = axes[1, 1]
summary_exp_labels = []
summary_post_vals = []
for exp_id, exp_name, exp_key in exp_labels:
    data = exp_results.get(exp_key, {})
    _, post = _best_variant(data)
    if post is not None:
        summary_exp_labels.append(f'E{exp_id}')
        summary_post_vals.append(post.get('avg', 0))

x_sum = np.arange(len(summary_exp_labels))
colors_sum = ['#27ae60' if v > BASELINE_POST else '#e74c3c' for v in summary_post_vals]
ax.bar(x_sum, summary_post_vals, color=colors_sum, alpha=0.8)
ax.axhline(BASELINE_POST, color='#333', lw=2, ls='--', label=f'Baseline Post ({BASELINE_POST:+.1f}%)')
ax.axhline(0, color='#333', lw=0.5, ls=':')
ax.set_xticks(x_sum); ax.set_xticklabels(summary_exp_labels, rotation=45)
ax.set_title('Post-Aug Avg PnL: Best Variant per Experiment')
ax.set_ylabel('Avg PnL %'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.94])
p3 = os.path.join(BASE_DIR, 'entry_mech_part3.png')
plt.savefig(p3, dpi=150, bbox_inches='tight'); plt.show()
print(f"  Saved: {p3}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE JSON
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_metrics(m):
    if m is None: return None
    out = {}
    for k, v in m.items():
        if isinstance(v, (np.integer, np.int64)): out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)): out[k] = round(float(v), 4)
        elif isinstance(v, np.bool_): out[k] = bool(v)
        elif isinstance(v, dict): out[k] = _clean_metrics(v)
        else: out[k] = v
    return out

json_out = {
    'config': {
        'backtest_start': BACKTEST_START, 'backtest_end': BACKTEST_END,
        'break_date': BREAK_DATE, 'top_n': TOP_N,
    },
    'baseline': {
        'pre': _clean_metrics(pre_m), 'post': _clean_metrics(post_m),
        'delta': round(float(BASELINE_DELTA), 3),
    },
    'experiments': {},
}

for exp_id, exp_name, exp_key in exp_labels:
    data = exp_results.get(exp_key, {})
    json_out['experiments'][exp_key] = _clean_metrics(data) if isinstance(data, dict) else data

jp = os.path.join(BASE_DIR, 'results', 'entry_mechanics_investigation.json')
os.makedirs(os.path.dirname(jp), exist_ok=True)
with open(jp, 'w') as f:
    json.dump(json_out, f, indent=2, default=lambda x: round(float(x), 4) if isinstance(x, (np.floating, np.integer)) else str(x))
print(f"\n  Saved: {jp}")

print(f"\n{'='*80}")
print(f"  TOTAL RUNTIME: {time.time() - t_total:.0f}s ({(time.time() - t_total)/60:.1f} min)")
print(f"{'='*80}")
