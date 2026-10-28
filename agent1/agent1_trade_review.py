# %% Agent 1 — N Sweep for Top 5 Strategies
#
# Sweeps N=1..5 for top 5 strategies from trade review:
#   1. C5: PM+Red        (PostSh 0.928 @ N=1)
#   2. C3+C5             (PostSh 0.883 @ N=1)
#   3. C7: PMTrend<0     (PostSh 0.720 @ N=1)
#   4. C3: SkipT6        (PostSh 0.486 @ N=1)
#   5. C9: 2BarDrop      (PostSh 0.355 @ N=1)
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
BACKTEST_END   = '2026-10-09'

GAP_MIN_PCT    = 3.0
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
PRE_FILTER_N   = 30

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

SD_MAX_PEAK_BAR = 72
SD_FALLBACK_BAR = 75
SD_CONFIRM_BAR  = 59

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
            return tk, {
                'type': res.get('type'),
                'name': res.get('name', ''),
                'shares': res.get('share_class_shares_outstanding'),
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
                return {
                    'entry_price': bars[ci][1]['c'],
                    'entry_time': bars[ci][0],
                    'entry_type': 'local_high',
                    'peak_bar': i,
                }
    fb = max(fallback_bar, min_entry_bar) if fallback_bar is not None else None
    if fb is not None and fb < len(bars):
        return {
            'entry_price': bars[fb][1]['c'],
            'entry_time': bars[fb][0],
            'entry_type': 'fallback',
            'peak_bar': None,
        }
    return None


def _T_to_min_entry_bar(T):
    h, m = int(T[:2]), int(T[3:5])
    T_min = h * 60 + m
    rth_start = 9 * 60 + 30
    if T_min <= rth_start:
        return 0
    return T_min - rth_start


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
        candidates.append({
            'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
            'open_price': cur_p, 'scan_date': dt_str,
            'volume_today': today[tk].get('v', 0),
        })
    candidates.sort(key=lambda x: -x['gap_pct'])
    nxt = _next_n_td(dt_str, 2)
    t1 = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
    t2 = nxt[1].strftime('%Y-%m-%d') if len(nxt) >= 2 else None
    for s in candidates:
        s['t1_date'] = t1
        s['t2_date'] = t2
        s['prior_str'] = prior_str
    return candidates


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
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest():
    t_total = time.time()

    print(f"\n{'='*80}")
    print(f"  AGENT 1 — TRADE REVIEW (7 strategies)")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"{'='*80}")

    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"\n  {n_dates} trading days")

    t1 = time.time()
    print(f"\n  Phase 1: Scanning...")
    all_candidates = {}
    total_cands = 0
    for i, dt in enumerate(dates):
        cands = scan_day(dt)
        all_candidates[dt] = cands
        total_cands += len(cands)
        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_cands} total  |  {time.time()-t_total:.0f}s")

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
        if done % 200 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

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
                c['gap_pct_daily'] = c['gap_pct']
                c['gap_pct'] = fresh_gap
                if fresh_gap < gap_min_final:
                    c['ah_stale'] = True
                    n_stale += 1
    print(f"  → AH ref: {n_ah_found} tickers, {n_stale} stale excluded")

    print(f"\n  Phase 2b: Re-rank + erosion...")
    n_reranked = 0
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
            n_reranked += 1
        reranked.sort(key=lambda x: -x['gap_pct'])
        all_candidates[dt] = reranked

    all_tickers = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            all_tickers.add(c['ticker'])
    print(f"\n  Phase 3: Reference data for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    for c_start in range(0, len(tk_list), 200):
        chunk = tk_list[c_start:c_start + 200]
        fetch_ticker_refs(chunk)

    min_eb = _T_to_min_entry_bar(SCAN_TIME_ET)
    pattern_cache = {}
    entry_cache_bl = {}
    entry_cache_sd = {}
    for key, bar_data in all_bars.items():
        rth_bars = bar_data.get('rth', []) if isinstance(bar_data, dict) else bar_data
        pattern_cache[key] = classify_open_pattern(rth_bars)
        if rth_bars and len(rth_bars) >= PEAK_WINDOW * 2 + 1:
            entry_cache_bl[key] = find_local_high(rth_bars, min_entry_bar=min_eb)
            entry_cache_sd[key] = find_local_high(
                rth_bars, window=PEAK_WINDOW,
                max_peak_bar=SD_MAX_PEAK_BAR, fallback_bar=SD_FALLBACK_BAR,
                min_entry_bar=60)

    print(f"\n  Pipeline done in {time.time()-t_total:.0f}s")
    return all_candidates, all_bars, pattern_cache, entry_cache_bl, entry_cache_sd, dates, t_total


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD TRADES WITH SL
# ═══════════════════════════════════════════════════════════════════════════════

def build_trades(all_candidates, all_bars, pattern_cache, entry_cache, dates,
                 mode='e2a', top_n=1, sl_pct=None):

    require_sd = (mode in ('e2b', 'e2b_pm'))
    skip_fallback = (mode not in ('e2a', 'e2b', 'e2b_pm'))
    skip_t6 = (mode in ('e2a_nofb_skip_t6', 'e2a_nofb_t6_pm'))
    pm_filters = (mode in ('e2a_nofb_pm', 'e2a_nofb_t6_pm', 'e2b_pm'))
    pm_trend_filter = (mode == 'e2a_nofb_pmtrend')
    two_red_filter = (mode == 'e2a_nofb_2red')
    two_bar_drop_filter = (mode == 'e2a_nofb_2drop')

    def _build_trade(c):
        tk, dt = c['ticker'], c['scan_date']
        entry = entry_cache.get((tk, dt))
        if entry is None:
            return None
        tc = copy.deepcopy(c)
        tc['entry_price'] = entry['entry_price']
        tc['entry_time'] = entry['entry_time']
        tc['entry_type'] = entry['entry_type']
        tc['open_pattern'] = pattern_cache.get((tk, dt), 'no_data')
        tc['asset_type'] = _ticker_ref.get(tk, {}).get('type', 'UNK')
        ep = tc['entry_price']

        bar_data = all_bars.get((tk, dt))
        rth = bar_data.get('rth', []) if bar_data else []

        exit_price = None
        exit_reason = 'eod'
        sl_price = ep * (1 + sl_pct / 100) if sl_pct is not None else None

        if bar_data:
            session_bars = rth + bar_data.get('ah', [])
            entry_t = tc['entry_time']
            for t_str, bar in session_bars:
                if t_str > EXIT_TIME:
                    break
                if t_str >= entry_t:
                    if sl_price is not None and bar['h'] >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'stop_loss'
                        break
                    exit_price = bar['c']

        if exit_price is not None:
            tc['pnl_eod'] = round((ep - exit_price) / ep * 100, 3)
        else:
            t0d = _gd_cache.get(dt, {}).get(tk, {})
            if t0d.get('c'):
                tc['pnl_eod'] = round((ep - t0d['c']) / ep * 100, 3)
            else:
                tc['pnl_eod'] = None

        tc['exit_price'] = exit_price
        tc['exit_reason'] = exit_reason
        tc['exit_time'] = t_str if exit_price is not None else None
        return tc

    trades = []
    for dt in dates:
        cands = all_candidates.get(dt, [])
        cs_qualified = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            ref = _ticker_ref.get(tk, {})
            at = ref.get('type', 'UNK')
            if at != 'CS': continue
            pattern = pattern_cache.get((tk, dt), 'no_data')
            if require_sd and pattern != 'sustained_drop': continue
            cc = copy.deepcopy(c)
            cc['asset_type'] = at
            cc['open_pattern'] = pattern
            cs_qualified.append(cc)

        for c in cs_qualified[:top_n]:
            if skip_t6:
                gp = c.get('gap_pct', 0)
                if 0.15 <= gp < 0.30: continue
            if pm_filters:
                tk, ddt = c['ticker'], c['scan_date']
                bar_data = all_bars.get((tk, ddt))
                if bar_data:
                    pm_bars = bar_data.get('pm', [])
                    rth_bars_local = bar_data.get('rth', [])
                    pm_after_830 = [(t_s, b) for t_s, b in pm_bars if t_s >= '08:30']
                    prior_close = c.get('prior_close', 0)
                    if pm_after_830 and prior_close > 0:
                        pm_high = max(b['h'] for _, b in pm_after_830)
                        last_pm_close = pm_after_830[-1][1]['c']
                        denom = pm_high - prior_close
                        pm_retrace = (pm_high - last_pm_close) / denom if denom > 0 else 0
                        if pm_retrace < 0.08: continue
                    if rth_bars_local:
                        bar0 = rth_bars_local[0][1]
                        if bar0['c'] > bar0['o']: continue
            if pm_trend_filter:
                tk, ddt = c['ticker'], c['scan_date']
                bar_data = all_bars.get((tk, ddt))
                if bar_data:
                    pm_bars = bar_data.get('pm', [])
                    pm_after_830 = [(t_s, b) for t_s, b in pm_bars if t_s >= '08:30']
                    if len(pm_after_830) >= 2:
                        first_pm = pm_after_830[0][1]['o']
                        last_pm = pm_after_830[-1][1]['c']
                        pm_trend = (last_pm - first_pm) / first_pm if first_pm > 0 else 0
                        if pm_trend >= 0: continue
                    else:
                        continue
                else:
                    continue
            if two_red_filter:
                tk, ddt = c['ticker'], c['scan_date']
                bar_data = all_bars.get((tk, ddt))
                if bar_data:
                    rth_bars_local = bar_data.get('rth', [])
                    if len(rth_bars_local) >= 2:
                        bar0 = rth_bars_local[0][1]
                        bar1 = rth_bars_local[1][1]
                        if bar0['c'] > bar0['o'] or bar1['c'] > bar1['o']: continue
                    else:
                        continue
                else:
                    continue
            if two_bar_drop_filter:
                tk, ddt = c['ticker'], c['scan_date']
                bar_data = all_bars.get((tk, ddt))
                if bar_data:
                    rth_bars_local = bar_data.get('rth', [])
                    if len(rth_bars_local) >= 2:
                        open_p = rth_bars_local[0][1]['o']
                        bar1_close = rth_bars_local[1][1]['c']
                        drop_2bar = (bar1_close - open_p) / open_p if open_p > 0 else 0
                        if drop_2bar >= 0: continue
                    else:
                        continue
                else:
                    continue
            trade = _build_trade(c)
            if trade:
                if skip_fallback and trade['entry_type'] == 'fallback': continue
                if trade.get('pnl_eod') is not None:
                    trades.append(trade)

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# RUN — N sweep (N=1..5) for top 5 strategies
# ═══════════════════════════════════════════════════════════════════════════════

all_candidates, all_bars, pattern_cache, entry_cache_bl, entry_cache_sd, dates, t_total = run_backtest()

MODES = [
    ('e2a_nofb_pm',      'bl', 'C5: PM+Red'),
    ('e2a_nofb_t6_pm',   'bl', 'C3+C5'),
    ('e2a_nofb_pmtrend', 'bl', 'C7: PMTrend<0'),
    ('e2a_nofb_skip_t6', 'bl', 'C3: SkipT6'),
    ('e2a_nofb_2drop',   'bl', 'C9: 2BarDrop'),
]
N_VALUES = [1, 2, 3, 4, 5]
_CACHES = {'bl': entry_cache_bl, 'sd': entry_cache_sd}

all_results = {}
for mode, cache_key, base_lbl in MODES:
    for n in N_VALUES:
        lbl = f"{base_lbl} N={n}"
        trades = build_trades(all_candidates, all_bars, pattern_cache,
                              _CACHES[cache_key], dates,
                              mode=mode, top_n=n, sl_pct=None)
        all_results[lbl] = trades

print(f"\n\n{'='*120}")
print(f"  N SWEEP — Top 5 Strategies × N=1..5  (no SL, exit {EXIT_TIME} ET)")
print(f"{'='*120}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE — full period + post-Aug
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  ── Full Period ({BACKTEST_START} → {BACKTEST_END}) ──")
full_rows = []
for mode, cache_key, base_lbl in MODES:
    for n in N_VALUES:
        lbl = f"{base_lbl} N={n}"
        trades = all_results[lbl]
        pnls = [t['pnl_eod'] for t in trades]
        m = compute_metrics(pnls, lbl)
        worst = min(pnls) if pnls else 0
        best = max(pnls) if pnls else 0
        dollar = CAPITAL * sum(pnls) / 100 if pnls else 0
        full_rows.append([
            base_lbl, n, m['n'], f"{m['avg']:+.2f}%", f"{m['med']:+.2f}%",
            f"{m['wr']*100:.0f}%", f"{m['sharpe']:.3f}",
            f"${dollar:+,.0f}", f"{worst:+.2f}%", f"{best:+.2f}%",
            f"[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]",
        ])
print()
print(tabulate(full_rows,
               headers=['Strategy', 'N', 'Trades', 'Avg', 'Med', 'WR', 'Sharpe',
                        '$NC', 'Worst', 'Best', '95% CI'],
               tablefmt='simple', stralign='right'))

print(f"\n\n  ── Post-Aug Only (≥ 2026-08-16) ──")
post_rows = []
for mode, cache_key, base_lbl in MODES:
    for n in N_VALUES:
        lbl = f"{base_lbl} N={n}"
        trades = all_results[lbl]
        post_trades = [t for t in trades if t['scan_date'] >= '2026-08-16']
        post_pnls = [t['pnl_eod'] for t in post_trades]
        m = compute_metrics(post_pnls, lbl)
        worst = min(post_pnls) if post_pnls else 0
        best = max(post_pnls) if post_pnls else 0
        dollar = CAPITAL * sum(post_pnls) / 100 if post_pnls else 0
        post_rows.append([
            base_lbl, n, m['n'], f"{m['avg']:+.2f}%", f"{m['med']:+.2f}%",
            f"{m['wr']*100:.0f}%", f"{m['sharpe']:.3f}",
            f"${dollar:+,.0f}", f"{worst:+.2f}%", f"{best:+.2f}%",
            f"[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]",
        ])
print()
print(tabulate(post_rows,
               headers=['Strategy', 'N', 'Trades', 'Avg', 'Med', 'WR', 'Sharpe',
                        '$NC', 'Worst', 'Best', '95% CI'],
               tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMAL N per strategy
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*100}")
print(f"  OPTIMAL N PER STRATEGY (by Post-Aug Sharpe)")
print(f"{'='*100}\n")

for mode, cache_key, base_lbl in MODES:
    best_sh = -999
    best_n = 1
    best_m = None
    for n in N_VALUES:
        lbl = f"{base_lbl} N={n}"
        post_trades = [t for t in all_results[lbl] if t['scan_date'] >= '2026-08-16']
        post_pnls = [t['pnl_eod'] for t in post_trades]
        m = compute_metrics(post_pnls)
        if m['sharpe'] > best_sh:
            best_sh = m['sharpe']
            best_n = n
            best_m = m
    worst = min([t['pnl_eod'] for t in all_results[f"{base_lbl} N={best_n}"]
                 if t['scan_date'] >= '2026-08-16']) if best_m and best_m['n'] > 0 else 0
    dollar = CAPITAL * best_m['avg'] * best_m['n'] / 100 if best_m else 0
    print(f"  {base_lbl:20s}  →  N={best_n}  |  PostSh={best_sh:.3f}  "
          f"PostN={best_m['n']}  Avg={best_m['avg']:+.2f}%  "
          f"WR={best_m['wr']*100:.0f}%  Worst={worst:+.2f}%  ${dollar:+,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE LOGS for best N per strategy
# ═══════════════════════════════════════════════════════════════════════════════

for mode, cache_key, base_lbl in MODES:
    best_sh = -999
    best_n = 1
    for n in N_VALUES:
        lbl = f"{base_lbl} N={n}"
        post_pnls = [t['pnl_eod'] for t in all_results[lbl]
                     if t['scan_date'] >= '2026-08-16']
        m = compute_metrics(post_pnls)
        if m['sharpe'] > best_sh:
            best_sh = m['sharpe']
            best_n = n

    lbl = f"{base_lbl} N={best_n}"
    trades = all_results[lbl]
    pnls = [t['pnl_eod'] for t in trades]
    m = compute_metrics(pnls, lbl)

    print(f"\n\n{'='*130}")
    print(f"  {lbl}  —  {m['n']} trades  |  Avg {m['avg']:+.2f}%  |  "
          f"WR {m['wr']*100:.0f}%  |  Sharpe {m['sharpe']:.3f}  |  "
          f"CI [{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]")
    print(f"{'='*130}")

    pre = [t for t in trades if t['scan_date'] < '2026-08-16']
    post = [t for t in trades if t['scan_date'] >= '2026-08-16']
    if pre:
        pm = compute_metrics([t['pnl_eod'] for t in pre])
        print(f"  Pre-Aug:  {pm['n']} trades  avg={pm['avg']:+.2f}%  WR={pm['wr']*100:.0f}%  Sharpe={pm['sharpe']:.3f}")
    if post:
        pm = compute_metrics([t['pnl_eod'] for t in post])
        print(f"  Post-Aug: {pm['n']} trades  avg={pm['avg']:+.2f}%  WR={pm['wr']*100:.0f}%  Sharpe={pm['sharpe']:.3f}")

    rows = []
    cum_dollar = 0.0
    for i, t in enumerate(sorted(trades, key=lambda x: x['scan_date']), 1):
        pnl = t['pnl_eod']
        dollar = CAPITAL * pnl / 100
        cum_dollar += dollar
        period = 'PRE' if t['scan_date'] < '2026-08-16' else 'POST'
        wl = 'W' if pnl > 0 else 'L'
        rows.append([
            i, t['scan_date'], t['ticker'],
            f"{t['gap_pct']*100:.1f}%",
            t.get('open_pattern', '?')[:8],
            t['entry_type'][:3].upper(),
            t['entry_time'],
            f"${t['entry_price']:.2f}",
            f"${t['exit_price']:.2f}" if t.get('exit_price') else '?',
            f"{pnl:+.2f}%",
            f"${dollar:+,.0f}",
            f"${cum_dollar:+,.0f}",
            wl, period,
        ])
    print()
    print(tabulate(rows,
                   headers=['#', 'Date', 'Ticker', 'Gap', 'Pattern', 'Entry',
                            'Time', 'EntryP', 'ExitP', 'PnL%', '$PnL', 'Cum$',
                            'W/L', 'Period'],
                   tablefmt='simple', stralign='right'))

    monthly = defaultdict(list)
    for t in trades:
        monthly[t['scan_date'][:7]].append(t['pnl_eod'])
    print(f"\n  ── Monthly Breakdown ──")
    mo_rows = []
    for mo in sorted(monthly):
        arr = np.array(monthly[mo])
        sh = float(np.mean(arr) / np.std(arr, ddof=1)) if len(arr) > 1 and np.std(arr, ddof=1) > 0 else 0
        dollar = float(CAPITAL * np.sum(arr) / 100)
        mo_rows.append([mo, len(arr), f"{np.mean(arr):+.2f}%", f"{(arr > 0).mean()*100:.0f}%",
                        f"{sh:.3f}", f"${dollar:+,.0f}", f"{arr.min():+.2f}%", f"{arr.max():+.2f}%"])
    print(tabulate(mo_rows, headers=['Month', 'N', 'Avg', 'WR', 'Sharpe', '$NC', 'Worst', 'Best'],
                   tablefmt='simple', stralign='right'))


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS — Sharpe × N heatmap + equity curves for best N
# ═══════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

strat_colors = ['#e74c3c', '#9b59b6', '#3498db', '#27ae60', '#e67e22']

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle(f'Agent 1 — N Sweep (top 5 strategies × N=1..5)\n'
             f'{dates[0]} → {dates[-1]}  |  Exit {EXIT_TIME} ET  |  SHORT',
             fontsize=11, fontweight='bold')

# Chart 1: Post-Aug Sharpe vs N
ax = axes[0, 0]
for idx, (mode, cache_key, base_lbl) in enumerate(MODES):
    sharpes = []
    for n in N_VALUES:
        lbl = f"{base_lbl} N={n}"
        post_pnls = [t['pnl_eod'] for t in all_results[lbl] if t['scan_date'] >= '2026-08-16']
        m = compute_metrics(post_pnls)
        sharpes.append(m['sharpe'])
    ax.plot(N_VALUES, sharpes, 'o-', color=strat_colors[idx], linewidth=2,
            markersize=8, label=base_lbl)
ax.set_xlabel('TOP_N')
ax.set_ylabel('Post-Aug Sharpe')
ax.set_title('Post-Aug Sharpe vs N')
ax.set_xticks(N_VALUES)
ax.axhline(0, color='#333', lw=0.8, ls=':')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Chart 2: Post-Aug Avg PnL vs N
ax = axes[0, 1]
for idx, (mode, cache_key, base_lbl) in enumerate(MODES):
    avgs = []
    for n in N_VALUES:
        lbl = f"{base_lbl} N={n}"
        post_pnls = [t['pnl_eod'] for t in all_results[lbl] if t['scan_date'] >= '2026-08-16']
        m = compute_metrics(post_pnls)
        avgs.append(m['avg'])
    ax.plot(N_VALUES, avgs, 'o-', color=strat_colors[idx], linewidth=2,
            markersize=8, label=base_lbl)
ax.set_xlabel('TOP_N')
ax.set_ylabel('Post-Aug Avg PnL %')
ax.set_title('Post-Aug Avg PnL % vs N')
ax.set_xticks(N_VALUES)
ax.axhline(0, color='#333', lw=0.8, ls=':')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Chart 3: Post-Aug $ vs N
ax = axes[1, 0]
for idx, (mode, cache_key, base_lbl) in enumerate(MODES):
    dollars = []
    for n in N_VALUES:
        lbl = f"{base_lbl} N={n}"
        post_pnls = [t['pnl_eod'] for t in all_results[lbl] if t['scan_date'] >= '2026-08-16']
        dollars.append(CAPITAL * sum(post_pnls) / 100 if post_pnls else 0)
    ax.plot(N_VALUES, dollars, 'o-', color=strat_colors[idx], linewidth=2,
            markersize=8, label=base_lbl)
ax.set_xlabel('TOP_N')
ax.set_ylabel('Post-Aug Cumulative $')
ax.set_title('Post-Aug $ vs N')
ax.set_xticks(N_VALUES)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.axhline(0, color='#333', lw=0.8, ls=':')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Chart 4: Equity curves for optimal N per strategy
ax = axes[1, 1]
aug15 = datetime.strptime('2026-08-15', '%Y-%m-%d')
for idx, (mode, cache_key, base_lbl) in enumerate(MODES):
    best_sh = -999
    best_n = 1
    for n in N_VALUES:
        post_pnls = [t['pnl_eod'] for t in all_results[f"{base_lbl} N={n}"]
                     if t['scan_date'] >= '2026-08-16']
        m = compute_metrics(post_pnls)
        if m['sharpe'] > best_sh:
            best_sh = m['sharpe']
            best_n = n
    lbl = f"{base_lbl} N={best_n}"
    trades = all_results[lbl]
    daily_pnl = defaultdict(list)
    for t in trades:
        daily_pnl[t['scan_date']].append(t['pnl_eod'])
    cum = 0.0
    d_dates_plot, cum_vals = [], []
    for dt in dates:
        dp = daily_pnl.get(dt, [])
        if dp:
            n_t = len(dp)
            pos_size = CAPITAL / n_t
            cum += sum(pos_size * (r / 100) for r in dp)
        d_dates_plot.append(datetime.strptime(dt, '%Y-%m-%d'))
        cum_vals.append(cum)
    ax.plot(d_dates_plot, cum_vals, color=strat_colors[idx], linewidth=2,
            label=f'{base_lbl} N={best_n}: ${cum:+,.0f}')
ax.axvline(aug15, color='red', ls='--', lw=1, alpha=0.5, label='Aug 15')
ax.axhline(0, color='#333', lw=0.8, ls=':')
ax.set_title('Equity Curves (optimal N per strategy)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.92])
p1 = os.path.join(BASE_DIR, 'agent1_n_sweep_top5.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {p1}")

print(f"\n  Total runtime: {time.time() - t_total:.0f}s ({(time.time() - t_total)/60:.1f} min)")
