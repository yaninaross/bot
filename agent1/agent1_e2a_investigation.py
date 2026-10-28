# %% Agent 1 — E2-A Post-August Performance Investigation
#
# 5-phase investigation of E2-A degradation after Aug 2026:
#   Phase 1: Temporal break analysis (rolling Sharpe, WR, drawdown)
#   Phase 2: Signal audit (gap distribution, signal-outcome correlation)
#   Phase 3: Execution audit (entry quality, fallback rate, data feed)
#   Phase 4: Strategy decomposition (pattern, MFE/MAE, hold period, tickers)
#   Phase 5: Regime classification (SPY vol/trend → E2-A performance)
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

EXIT_HOURS = ['10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00']

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
# HELPERS (same as n_sweep pipeline)
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
                        'entry_type': 'local_high', 'peak_bar': i}
    fb = max(fallback_bar, min_entry_bar) if fallback_bar is not None else None
    if fb is not None and fb < len(bars):
        return {'entry_price': bars[fb][1]['c'], 'entry_time': bars[fb][0],
                'entry_type': 'fallback', 'peak_bar': None}
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


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    t_total = time.time()
    print(f"\n{'='*80}")
    print(f"  E2-A POST-AUGUST INVESTIGATION")
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
# BUILD TRADES
# ═══════════════════════════════════════════════════════════════════════════════

def build_trades(all_candidates, all_bars, pattern_cache, entry_cache, dates, top_n):
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
            ep = tc['entry_price']

            bar_data = all_bars.get((tk, dt2))
            exit_price = None
            if bar_data:
                session_bars = bar_data.get('rth', []) + bar_data.get('ah', [])
                for t_str, bar in session_bars:
                    if t_str > EXIT_TIME: break
                    if t_str >= tc['entry_time']: exit_price = bar['c']

            if exit_price is not None:
                tc['pnl_eod'] = round((ep - exit_price) / ep * 100, 3)
            else:
                t0d = _gd_cache.get(dt2, {}).get(tk, {})
                tc['pnl_eod'] = round((ep - t0d['c']) / ep * 100, 3) if t0d.get('c') else None

            tc['exit_price'] = exit_price

            # MFE / MAE (SHORT: favorable = price drops)
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

            if tc.get('pnl_eod') is not None:
                trades.append(tc)
    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE + BUILD
# ═══════════════════════════════════════════════════════════════════════════════

all_candidates, all_bars, pattern_cache, entry_cache, dates, t_total = run_pipeline()
trades = build_trades(all_candidates, all_bars, pattern_cache, entry_cache, dates, TOP_N)

pre_trades = [t for t in trades if t['scan_date'] < BREAK_DATE]
post_trades = [t for t in trades if t['scan_date'] >= BREAK_DATE]

pre_m = compute_metrics([t['pnl_eod'] for t in pre_trades], 'Pre-Aug')
post_m = compute_metrics([t['pnl_eod'] for t in post_trades], 'Post-Aug')
all_m = compute_metrics([t['pnl_eod'] for t in trades], 'All')

print(f"\n{'='*80}")
print(f"  OVERVIEW: {len(trades)} trades (Pre: {pre_m['n']}, Post: {post_m['n']})")
print(f"  Pre-Aug:  avg={pre_m['avg']:+.2f}%, Sharpe={pre_m['sharpe']:.3f}, WR={pre_m['wr']*100:.0f}%")
print(f"  Post-Aug: avg={post_m['avg']:+.2f}%, Sharpe={post_m['sharpe']:.3f}, WR={post_m['wr']*100:.0f}%")
print(f"  Delta:    avg={post_m['avg']-pre_m['avg']:+.2f}pp, Sharpe={post_m['sharpe']-pre_m['sharpe']:+.3f}")
print(f"{'='*80}")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: TEMPORAL BREAK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  PHASE 1 — TEMPORAL BREAK ANALYSIS")
print(f"{'='*80}")

daily_pnl = defaultdict(list)
for t in trades:
    daily_pnl[t['scan_date']].append(t['pnl_eod'])

daily_df = pd.DataFrame([
    {'date': dt,
     'n_trades': len(daily_pnl.get(dt, [])),
     'avg_pnl': float(np.mean(daily_pnl[dt])) if dt in daily_pnl else np.nan,
     'total_dollar': sum(CAPITAL / len(daily_pnl[dt]) * (r / 100) for r in daily_pnl[dt]) if dt in daily_pnl else 0.0}
    for dt in dates
])
daily_df['date_dt'] = pd.to_datetime(daily_df['date'])
daily_df['cum_dollar'] = daily_df['total_dollar'].cumsum()
daily_df['peak'] = daily_df['cum_dollar'].cummax()
daily_df['drawdown'] = (daily_df['cum_dollar'] - daily_df['peak'])

trade_days = daily_df[daily_df['n_trades'] > 0].copy()
trade_days['roll_avg'] = trade_days['avg_pnl'].rolling(20, min_periods=5).mean()
trade_days['roll_std'] = trade_days['avg_pnl'].rolling(20, min_periods=5).std()
trade_days['roll_sharpe'] = trade_days['roll_avg'] / trade_days['roll_std']
trade_days['roll_wr'] = trade_days['avg_pnl'].apply(lambda x: 1 if x > 0 else 0).rolling(20, min_periods=5).mean()

# SPY daily returns for regime analysis
spy_close = []
for dt in dates:
    gd = _gd_cache.get(dt, {})
    spy = gd.get('SPY', {})
    if spy.get('c'):
        spy_close.append({'date': dt, 'spy_c': spy['c'], 'spy_o': spy.get('o', spy['c'])})
spy_df = pd.DataFrame(spy_close)
if len(spy_df) > 0:
    spy_df['date_dt'] = pd.to_datetime(spy_df['date'])
    spy_df['spy_ret'] = spy_df['spy_c'].pct_change()
    spy_df['spy_vol_10d'] = spy_df['spy_ret'].rolling(10).std() * np.sqrt(252)
    spy_df['spy_trend_20d'] = spy_df['spy_c'].pct_change(20)
    spy_df['spy_ret_5d'] = spy_df['spy_c'].pct_change(5)

break_dt = pd.Timestamp(BREAK_DATE)

# Find the sharpest Sharpe drop
if len(trade_days) > 20:
    rs = trade_days['roll_sharpe'].dropna()
    if len(rs) > 5:
        rs_diff = rs.diff()
        worst_idx = rs_diff.idxmin()
        worst_date = trade_days.loc[worst_idx, 'date'] if worst_idx in trade_days.index else BREAK_DATE
        print(f"\n  Sharpest rolling Sharpe drop: {worst_date}")
        print(f"    Roll Sharpe went from {trade_days.loc[max(0, worst_idx-1), 'roll_sharpe']:.3f}"
              f" to {trade_days.loc[worst_idx, 'roll_sharpe']:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'Phase 1 — Temporal Break Analysis (break={BREAK_DATE})', fontsize=12, fontweight='bold')

ax = axes[0, 0]
td = trade_days
ax.bar(td['date_dt'], td['avg_pnl'], color=['#27ae60' if x > 0 else '#e74c3c' for x in td['avg_pnl']], alpha=0.6, width=1)
if 'roll_avg' in td.columns:
    ax.plot(td['date_dt'], td['roll_avg'], color='#2c3e50', linewidth=2, label='20d rolling avg')
ax.axvline(break_dt, color='red', linewidth=2, linestyle='--', label=f'Break: {BREAK_DATE}')
ax.axhline(0, color='#333', lw=0.8, ls=':')
ax.set_title('Daily Avg PnL (%) + 20d Rolling Mean')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

ax = axes[0, 1]
if 'roll_sharpe' in td.columns:
    ax.plot(td['date_dt'], td['roll_sharpe'], color='#2c3e50', linewidth=2)
    ax.axvline(break_dt, color='red', linewidth=2, linestyle='--')
    ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('20d Rolling Sharpe')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

ax = axes[1, 0]
if 'roll_wr' in td.columns:
    ax.plot(td['date_dt'], td['roll_wr'] * 100, color='#8e44ad', linewidth=2)
    ax.axvline(break_dt, color='red', linewidth=2, linestyle='--')
    ax.axhline(50, color='#333', lw=1, ls=':')
ax.set_title('20d Rolling Win Rate (%)')
ax.set_ylabel('%'); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

ax = axes[1, 1]
ax.fill_between(daily_df['date_dt'], daily_df['drawdown'], 0, color='#e74c3c', alpha=0.3)
ax.plot(daily_df['date_dt'], daily_df['drawdown'], color='#e74c3c', linewidth=1)
ax2 = ax.twinx()
ax2.plot(daily_df['date_dt'], daily_df['cum_dollar'], color='#2c3e50', linewidth=2)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.axvline(break_dt, color='red', linewidth=2, linestyle='--')
ax.set_title('Cumulative P&L (blue) + Drawdown (red)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

plt.tight_layout(rect=[0, 0, 1, 0.94])
p1 = os.path.join(BASE_DIR, 'investigation_phase1.png')
plt.savefig(p1, dpi=150, bbox_inches='tight'); plt.show()
print(f"  Saved: {p1}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: SIGNAL AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  PHASE 2 — SIGNAL AUDIT")
print(f"{'='*80}")

pre_gaps = [t['gap_pct'] * 100 for t in pre_trades]
post_gaps = [t['gap_pct'] * 100 for t in post_trades]
pre_pnls = [t['pnl_eod'] for t in pre_trades]
post_pnls = [t['pnl_eod'] for t in post_trades]

if pre_gaps and pre_pnls:
    corr_pre, p_pre = sp_stats.pearsonr(pre_gaps, pre_pnls)
else:
    corr_pre, p_pre = 0, 1
if post_gaps and post_pnls:
    corr_post, p_post = sp_stats.pearsonr(post_gaps, post_pnls)
else:
    corr_post, p_post = 0, 1

print(f"\n  Gap → PnL correlation:")
print(f"    Pre-Aug:  r={corr_pre:+.3f} (p={p_pre:.3f})")
print(f"    Post-Aug: r={corr_post:+.3f} (p={p_post:.3f})")
print(f"    {'INVERTED' if corr_pre * corr_post < 0 else 'SAME SIGN'}"
      f"  {'→ REGIME FLIP' if corr_pre > 0.05 and corr_post < -0.05 else ''}")

print(f"\n  Gap distribution:")
print(f"    Pre-Aug:  mean={np.mean(pre_gaps):.1f}%, std={np.std(pre_gaps):.1f}%, med={np.median(pre_gaps):.1f}%")
print(f"    Post-Aug: mean={np.mean(post_gaps):.1f}%, std={np.std(post_gaps):.1f}%, med={np.median(post_gaps):.1f}%")

pre_vol = [t.get('volume_today', 0) for t in pre_trades if t.get('volume_today', 0) > 0]
post_vol = [t.get('volume_today', 0) for t in post_trades if t.get('volume_today', 0) > 0]
print(f"\n  Volume (daily):")
print(f"    Pre-Aug:  mean={np.mean(pre_vol)/1e6:.1f}M, med={np.median(pre_vol)/1e6:.1f}M")
print(f"    Post-Aug: mean={np.mean(post_vol)/1e6:.1f}M, med={np.median(post_vol)/1e6:.1f}M")

pre_gc = [t.get('gap_consumed', 0) for t in pre_trades if 'gap_consumed' in t]
post_gc = [t.get('gap_consumed', 0) for t in post_trades if 'gap_consumed' in t]
if pre_gc and post_gc:
    print(f"\n  Gap consumed at entry:")
    print(f"    Pre-Aug:  mean={np.mean(pre_gc)*100:.1f}%, med={np.median(pre_gc)*100:.1f}%")
    print(f"    Post-Aug: mean={np.mean(post_gc)*100:.1f}%, med={np.median(post_gc)*100:.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Phase 2 — Signal Audit', fontsize=12, fontweight='bold')

ax = axes[0, 0]
bins = np.linspace(0, max(max(pre_gaps, default=30), max(post_gaps, default=30)), 30)
ax.hist(pre_gaps, bins=bins, alpha=0.6, color='#2c3e50', label=f'Pre-Aug (n={len(pre_gaps)})', density=True)
ax.hist(post_gaps, bins=bins, alpha=0.6, color='#e74c3c', label=f'Post-Aug (n={len(post_gaps)})', density=True)
ax.set_title('Gap % Distribution: Pre vs Post')
ax.set_xlabel('Gap %'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.scatter([t['gap_pct']*100 for t in pre_trades], [t['pnl_eod'] for t in pre_trades],
           alpha=0.4, c='#2c3e50', s=20, label=f'Pre (r={corr_pre:+.2f})')
ax.scatter([t['gap_pct']*100 for t in post_trades], [t['pnl_eod'] for t in post_trades],
           alpha=0.4, c='#e74c3c', s=20, label=f'Post (r={corr_post:+.2f})')
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Gap % vs PnL (signal-outcome)')
ax.set_xlabel('Gap %'); ax.set_ylabel('PnL %'); ax.legend(); ax.grid(True, alpha=0.3)

# Monthly avg gap, volume, price
monthly = defaultdict(lambda: {'gaps': [], 'vols': [], 'prices': [], 'pnls': []})
for t in trades:
    mo = t['scan_date'][:7]
    monthly[mo]['gaps'].append(t['gap_pct'] * 100)
    monthly[mo]['vols'].append(t.get('volume_today', 0))
    monthly[mo]['prices'].append(t['open_price'])
    monthly[mo]['pnls'].append(t['pnl_eod'])

months = sorted(monthly.keys())
ax = axes[1, 0]
ax.bar(range(len(months)), [np.mean(monthly[m]['gaps']) for m in months],
       color=['#2c3e50' if m < BREAK_DATE[:7] else '#e74c3c' for m in months], alpha=0.8)
ax.set_xticks(range(len(months))); ax.set_xticklabels([m[5:] for m in months])
ax.set_title('Monthly Avg Gap %')
ax.set_ylabel('Gap %'); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
if pre_gc and post_gc:
    gc_bins = np.linspace(0, 0.5, 20)
    ax.hist([x*100 for x in pre_gc], bins=gc_bins*100, alpha=0.6, color='#2c3e50',
            label=f'Pre (mean={np.mean(pre_gc)*100:.0f}%)', density=True)
    ax.hist([x*100 for x in post_gc], bins=gc_bins*100, alpha=0.6, color='#e74c3c',
            label=f'Post (mean={np.mean(post_gc)*100:.0f}%)', density=True)
    ax.set_title('Gap Consumed at Scan: Pre vs Post')
    ax.set_xlabel('% Consumed'); ax.legend()
else:
    ax.text(0.5, 0.5, 'No gap_consumed data', ha='center', va='center', transform=ax.transAxes)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
p2 = os.path.join(BASE_DIR, 'investigation_phase2.png')
plt.savefig(p2, dpi=150, bbox_inches='tight'); plt.show()
print(f"  Saved: {p2}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: EXECUTION AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  PHASE 3 — EXECUTION AUDIT")
print(f"{'='*80}")

pre_fb = [t for t in pre_trades if t['entry_type'] == 'fallback']
post_fb = [t for t in post_trades if t['entry_type'] == 'fallback']
pre_fb_rate = len(pre_fb) / len(pre_trades) * 100 if pre_trades else 0
post_fb_rate = len(post_fb) / len(post_trades) * 100 if post_trades else 0

print(f"\n  Fallback entry rate:")
print(f"    Pre-Aug:  {pre_fb_rate:.0f}% ({len(pre_fb)}/{len(pre_trades)})")
print(f"    Post-Aug: {post_fb_rate:.0f}% ({len(post_fb)}/{len(post_trades)})")

pre_lh = [t for t in pre_trades if t['entry_type'] == 'local_high']
post_lh = [t for t in post_trades if t['entry_type'] == 'local_high']
print(f"\n  Entry type PnL:")
print(f"    Pre  local_high: avg={np.mean([t['pnl_eod'] for t in pre_lh]):+.2f}% (n={len(pre_lh)})" if pre_lh else "    Pre  local_high: n=0")
print(f"    Pre  fallback:   avg={np.mean([t['pnl_eod'] for t in pre_fb]):+.2f}% (n={len(pre_fb)})" if pre_fb else "    Pre  fallback: n=0")
print(f"    Post local_high: avg={np.mean([t['pnl_eod'] for t in post_lh]):+.2f}% (n={len(post_lh)})" if post_lh else "    Post local_high: n=0")
print(f"    Post fallback:   avg={np.mean([t['pnl_eod'] for t in post_fb]):+.2f}% (n={len(post_fb)})" if post_fb else "    Post fallback: n=0")

pre_slip = [(t['entry_price'] - t['open_price']) / t['open_price'] * 100 for t in pre_trades if t['open_price'] > 0]
post_slip = [(t['entry_price'] - t['open_price']) / t['open_price'] * 100 for t in post_trades if t['open_price'] > 0]
print(f"\n  Entry vs open (% from open, positive = higher entry = better SHORT entry):")
print(f"    Pre-Aug:  mean={np.mean(pre_slip):+.2f}%, med={np.median(pre_slip):+.2f}%")
print(f"    Post-Aug: mean={np.mean(post_slip):+.2f}%, med={np.median(post_slip):+.2f}%")

# Data feed: bars per trade day
bars_per_day = defaultdict(int)
for (tk, dt), bd in all_bars.items():
    if dt in [t['scan_date'] for t in trades]:
        bars_per_day[dt] += len(bd.get('rth', []))
trade_dates = sorted(set(t['scan_date'] for t in trades))
print(f"\n  Avg RTH bars per fetched ticker:")
pre_bpd = [bars_per_day[d] for d in trade_dates if d < BREAK_DATE and bars_per_day.get(d)]
post_bpd = [bars_per_day[d] for d in trade_dates if d >= BREAK_DATE and bars_per_day.get(d)]
print(f"    Pre-Aug:  {np.mean(pre_bpd):.0f}" if pre_bpd else "    Pre-Aug: no data")
print(f"    Post-Aug: {np.mean(post_bpd):.0f}" if post_bpd else "    Post-Aug: no data")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: STRATEGY DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  PHASE 4 — STRATEGY DECOMPOSITION")
print(f"{'='*80}")

# 4.1 Pattern breakdown pre vs post
print(f"\n  4.1 — PnL by open pattern:")
patterns = ['sustained_drop', 'drop_then_recover', 'mild_drop', 'flat_or_bounce', 'no_data']
pat_rows = []
for pat in patterns:
    pre_p = [t['pnl_eod'] for t in pre_trades if t.get('open_pattern') == pat]
    post_p = [t['pnl_eod'] for t in post_trades if t.get('open_pattern') == pat]
    if pre_p or post_p:
        pat_rows.append([
            pat[:15],
            f"{len(pre_p)}", f"{np.mean(pre_p):+.2f}%" if pre_p else "—",
            f"{len(post_p)}", f"{np.mean(post_p):+.2f}%" if post_p else "—",
            f"{(np.mean(post_p) if post_p else 0) - (np.mean(pre_p) if pre_p else 0):+.2f}pp"
        ])
print()
print(tabulate(pat_rows, headers=['Pattern', 'Pre N', 'Pre Avg', 'Post N', 'Post Avg', 'Delta'],
               tablefmt='simple', stralign='right'))

# 4.2 MFE / MAE
pre_mfe = [t['mfe'] for t in pre_trades if t.get('mfe') is not None]
pre_mae = [t['mae'] for t in pre_trades if t.get('mae') is not None]
post_mfe = [t['mfe'] for t in post_trades if t.get('mfe') is not None]
post_mae = [t['mae'] for t in post_trades if t.get('mae') is not None]
print(f"\n  4.2 — MFE / MAE:")
print(f"    Pre-Aug  MFE: mean={np.mean(pre_mfe):+.2f}%, med={np.median(pre_mfe):+.2f}%")
print(f"    Pre-Aug  MAE: mean={np.mean(pre_mae):+.2f}%, med={np.median(pre_mae):+.2f}%")
print(f"    Post-Aug MFE: mean={np.mean(post_mfe):+.2f}%, med={np.median(post_mfe):+.2f}%")
print(f"    Post-Aug MAE: mean={np.mean(post_mae):+.2f}%, med={np.median(post_mae):+.2f}%")
print(f"    MFE delta: {np.mean(post_mfe)-np.mean(pre_mfe):+.2f}pp")
print(f"    MAE delta: {np.mean(post_mae)-np.mean(pre_mae):+.2f}pp")
if np.mean(post_mfe) < np.mean(pre_mfe) and abs(np.mean(post_mae) - np.mean(pre_mae)) < 1:
    print(f"    → MFE collapsed, MAE held → upside capture degraded (REGIME)")
elif np.mean(post_mfe) < np.mean(pre_mfe) and np.mean(post_mae) < np.mean(pre_mae):
    print(f"    → Both compressed → low-movement period (SIZING)")
elif np.mean(post_mae) > np.mean(pre_mae):
    print(f"    → MAE expanded → more adverse moves against SHORT (TREND)")

# 4.3 Hold period analysis
print(f"\n  4.3 — Hold period sensitivity (avg PnL at different exit times):")
hp_rows = []
for ex_t in EXIT_HOURS:
    key = f'pnl_{ex_t}'
    pre_hp = [t[key] for t in pre_trades if t.get(key) is not None]
    post_hp = [t[key] for t in post_trades if t.get(key) is not None]
    if pre_hp or post_hp:
        hp_rows.append([
            ex_t,
            f"{np.mean(pre_hp):+.2f}%" if pre_hp else "—",
            f"{np.mean(post_hp):+.2f}%" if post_hp else "—",
            f"{(np.mean(post_hp) if post_hp else 0) - (np.mean(pre_hp) if pre_hp else 0):+.2f}pp"
        ])
print()
print(tabulate(hp_rows, headers=['Exit Time', 'Pre Avg', 'Post Avg', 'Delta'], tablefmt='simple', stralign='right'))

# 4.4 Top/bottom tickers
print(f"\n  4.4 — Top/Bottom tickers by total $ PnL contribution:")
tk_dollar = defaultdict(lambda: {'pre': 0.0, 'post': 0.0, 'n_pre': 0, 'n_post': 0})
for t in pre_trades:
    n_day = len([x for x in pre_trades if x['scan_date'] == t['scan_date']])
    tk_dollar[t['ticker']]['pre'] += (CAPITAL / n_day) * (t['pnl_eod'] / 100)
    tk_dollar[t['ticker']]['n_pre'] += 1
for t in post_trades:
    n_day = len([x for x in post_trades if x['scan_date'] == t['scan_date']])
    tk_dollar[t['ticker']]['post'] += (CAPITAL / n_day) * (t['pnl_eod'] / 100)
    tk_dollar[t['ticker']]['n_post'] += 1

tk_total = {tk: v['pre'] + v['post'] for tk, v in tk_dollar.items()}
sorted_tks = sorted(tk_total.items(), key=lambda x: x[1])

print(f"\n  Bottom 10 (worst):")
for tk, total in sorted_tks[:10]:
    v = tk_dollar[tk]
    print(f"    {tk:6s}  total=${total:+,.0f}  pre=${v['pre']:+,.0f}({v['n_pre']})  post=${v['post']:+,.0f}({v['n_post']})")

print(f"\n  Top 10 (best):")
for tk, total in sorted_tks[-10:][::-1]:
    v = tk_dollar[tk]
    print(f"    {tk:6s}  total=${total:+,.0f}  pre=${v['pre']:+,.0f}({v['n_pre']})  post=${v['post']:+,.0f}({v['n_post']})")


fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Phase 4 — Strategy Decomposition', fontsize=12, fontweight='bold')

ax = axes[0, 0]
x_idx = np.arange(len([p for p in patterns if any(t.get('open_pattern') == p for t in trades)]))
valid_pats = [p for p in patterns if any(t.get('open_pattern') == p for t in trades)]
pre_avgs = [np.mean([t['pnl_eod'] for t in pre_trades if t.get('open_pattern') == p]) if any(t.get('open_pattern') == p for t in pre_trades) else 0 for p in valid_pats]
post_avgs = [np.mean([t['pnl_eod'] for t in post_trades if t.get('open_pattern') == p]) if any(t.get('open_pattern') == p for t in post_trades) else 0 for p in valid_pats]
w = 0.35
ax.bar(x_idx - w/2, pre_avgs, w, color='#2c3e50', alpha=0.8, label='Pre-Aug')
ax.bar(x_idx + w/2, post_avgs, w, color='#e74c3c', alpha=0.8, label='Post-Aug')
ax.set_xticks(x_idx); ax.set_xticklabels([p[:12] for p in valid_pats], rotation=30, ha='right')
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Avg PnL by Pattern: Pre vs Post'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

ax = axes[0, 1]
ax.scatter(pre_mae, pre_mfe, alpha=0.4, c='#2c3e50', s=20, label='Pre-Aug')
ax.scatter(post_mae, post_mfe, alpha=0.4, c='#e74c3c', s=20, label='Post-Aug')
ax.plot([0, max(max(pre_mae, default=0), max(post_mae, default=0))],
        [0, max(max(pre_mae, default=0), max(post_mae, default=0))],
        'k--', alpha=0.3, label='MFE=MAE')
ax.set_xlabel('MAE (adverse, %)'); ax.set_ylabel('MFE (favorable, %)')
ax.set_title('MFE vs MAE Scatter'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for label, subset, color in [('Pre-Aug', pre_trades, '#2c3e50'), ('Post-Aug', post_trades, '#e74c3c')]:
    hp_avgs = []
    for ex_t in EXIT_HOURS:
        key = f'pnl_{ex_t}'
        vals = [t[key] for t in subset if t.get(key) is not None]
        hp_avgs.append(np.mean(vals) if vals else np.nan)
    ax.plot(EXIT_HOURS, hp_avgs, 'o-', color=color, linewidth=2, label=label)
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Avg PnL by Exit Time: Pre vs Post')
ax.set_xlabel('Exit Time ET'); ax.set_ylabel('Avg PnL %')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
mo_pnl = [(m, np.mean(monthly[m]['pnls']), len(monthly[m]['pnls'])) for m in months]
colors_mo = ['#2c3e50' if m < BREAK_DATE[:7] else '#e74c3c' for m, _, _ in mo_pnl]
bars_mo = ax.bar(range(len(mo_pnl)), [x[1] for x in mo_pnl], color=colors_mo, alpha=0.8)
ax2 = ax.twinx()
ax2.plot(range(len(mo_pnl)), [x[2] for x in mo_pnl], 'o-', color='#f39c12', linewidth=2, label='Trade count')
ax.set_xticks(range(len(mo_pnl))); ax.set_xticklabels([m[5:] for m, _, _ in mo_pnl])
ax.axhline(0, color='#333', lw=1, ls=':')
ax.set_title('Monthly Avg PnL + Trade Count')
ax.set_ylabel('Avg PnL %'); ax2.set_ylabel('Trades', color='#f39c12')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.94])
p4 = os.path.join(BASE_DIR, 'investigation_phase4.png')
plt.savefig(p4, dpi=150, bbox_inches='tight'); plt.show()
print(f"  Saved: {p4}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  PHASE 5 — REGIME CLASSIFICATION")
print(f"{'='*80}")

if len(spy_df) > 20:
    spy_merged = spy_df[['date', 'spy_vol_10d', 'spy_trend_20d', 'spy_c']].dropna().copy()

    vol_med = spy_merged['spy_vol_10d'].median()
    spy_merged['regime'] = 'unknown'
    spy_merged.loc[(spy_merged['spy_vol_10d'] <= vol_med) & (spy_merged['spy_trend_20d'] >= 0), 'regime'] = 'calm_bull'
    spy_merged.loc[(spy_merged['spy_vol_10d'] <= vol_med) & (spy_merged['spy_trend_20d'] < 0), 'regime'] = 'calm_bear'
    spy_merged.loc[(spy_merged['spy_vol_10d'] > vol_med) & (spy_merged['spy_trend_20d'] >= 0), 'regime'] = 'vol_bull'
    spy_merged.loc[(spy_merged['spy_vol_10d'] > vol_med) & (spy_merged['spy_trend_20d'] < 0), 'regime'] = 'vol_bear'

    regime_map = dict(zip(spy_merged['date'], spy_merged['regime']))
    for t in trades:
        t['regime'] = regime_map.get(t['scan_date'], 'unknown')

    print(f"\n  SPY regime thresholds:")
    print(f"    Vol median (10d ann): {vol_med*100:.1f}%")
    print(f"    Trend: 20d SPY return ≥ 0 = bull, < 0 = bear")

    regime_rows = []
    for reg in ['calm_bull', 'calm_bear', 'vol_bull', 'vol_bear']:
        pre_r = [t['pnl_eod'] for t in pre_trades if t.get('regime') == reg]
        post_r = [t['pnl_eod'] for t in post_trades if t.get('regime') == reg]
        all_r = pre_r + post_r
        regime_rows.append([
            reg,
            f"{len(pre_r)}", f"{np.mean(pre_r):+.2f}%" if pre_r else "—",
            f"{len(post_r)}", f"{np.mean(post_r):+.2f}%" if post_r else "—",
            f"{len(all_r)}", f"{np.mean(all_r):+.2f}%" if all_r else "—",
        ])
    print()
    print(tabulate(regime_rows,
                   headers=['Regime', 'Pre N', 'Pre Avg', 'Post N', 'Post Avg', 'All N', 'All Avg'],
                   tablefmt='simple', stralign='right'))

    # Which regime dominated pre vs post?
    pre_regime_counts = defaultdict(int)
    post_regime_counts = defaultdict(int)
    for t in pre_trades: pre_regime_counts[t.get('regime', 'unknown')] += 1
    for t in post_trades: post_regime_counts[t.get('regime', 'unknown')] += 1
    print(f"\n  Regime distribution:")
    print(f"    Pre-Aug:  {dict(pre_regime_counts)}")
    print(f"    Post-Aug: {dict(post_regime_counts)}")

    # Check: was pre-Aug performance concentrated in one regime?
    best_pre_regime = max(
        [(reg, np.mean([t['pnl_eod'] for t in pre_trades if t.get('regime') == reg]))
         for reg in ['calm_bull', 'calm_bear', 'vol_bull', 'vol_bear']
         if any(t.get('regime') == reg for t in pre_trades)],
        key=lambda x: x[1], default=('none', 0)
    )
    print(f"\n  Best pre-Aug regime: {best_pre_regime[0]} (avg={best_pre_regime[1]:+.2f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Phase 5 — Regime Classification', fontsize=12, fontweight='bold')

    ax = axes[0]
    reg_colors = {'calm_bull': '#27ae60', 'calm_bear': '#95a5a6',
                  'vol_bull': '#f39c12', 'vol_bear': '#e74c3c', 'unknown': '#bdc3c7'}
    for reg in ['calm_bull', 'calm_bear', 'vol_bull', 'vol_bear']:
        mask = spy_merged['regime'] == reg
        ax.scatter(spy_merged.loc[mask, 'spy_vol_10d'] * 100,
                   spy_merged.loc[mask, 'spy_trend_20d'] * 100,
                   c=reg_colors[reg], s=30, alpha=0.6, label=reg)
    ax.axhline(0, color='#333', lw=1, ls=':')
    ax.axvline(vol_med * 100, color='#333', lw=1, ls=':')
    ax.set_xlabel('SPY 10d Realized Vol (ann, %)')
    ax.set_ylabel('SPY 20d Return (%)')
    ax.set_title('Regime Scatter (each dot = 1 trading day)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    regs = ['calm_bull', 'calm_bear', 'vol_bull', 'vol_bear']
    x_r = np.arange(len(regs))
    pre_ravg = [np.mean([t['pnl_eod'] for t in pre_trades if t.get('regime') == r]) if any(t.get('regime') == r for t in pre_trades) else 0 for r in regs]
    post_ravg = [np.mean([t['pnl_eod'] for t in post_trades if t.get('regime') == r]) if any(t.get('regime') == r for t in post_trades) else 0 for r in regs]
    w = 0.35
    ax.bar(x_r - w/2, pre_ravg, w, color='#2c3e50', alpha=0.8, label='Pre-Aug')
    ax.bar(x_r + w/2, post_ravg, w, color='#e74c3c', alpha=0.8, label='Post-Aug')
    ax.set_xticks(x_r); ax.set_xticklabels(regs, rotation=20, ha='right')
    ax.axhline(0, color='#333', lw=1, ls=':')
    ax.set_title('E2-A Avg PnL by Regime: Pre vs Post')
    ax.set_ylabel('Avg PnL %'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p5 = os.path.join(BASE_DIR, 'investigation_phase5.png')
    plt.savefig(p5, dpi=150, bbox_inches='tight'); plt.show()
    print(f"  Saved: {p5}")
else:
    print("  Insufficient SPY data for regime analysis")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY VERDICT TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*110}")
print(f"  INVESTIGATION SUMMARY — E2-A Performance Break")
print(f"  Period: {BACKTEST_START} → {BACKTEST_END}  |  Break: {BREAK_DATE}  |  N={TOP_N}")
print(f"{'='*110}")

def _verdict(pre_val, post_val, threshold=0.3, invert=False):
    if pre_val == 0: return 'INCONCLUSIVE'
    pct_change = (post_val - pre_val) / abs(pre_val) if pre_val != 0 else 0
    if invert: pct_change = -pct_change
    if abs(pct_change) > threshold: return 'CAUSE'
    elif abs(pct_change) > threshold / 2: return 'CONTRIBUTING'
    return 'NOT CAUSE'

summary = []
summary.append(['1.1', 'Avg PnL', f"{pre_m['avg']:+.2f}%", f"{post_m['avg']:+.2f}%",
                f"{post_m['avg']-pre_m['avg']:+.2f}pp", 'DEGRADATION CONFIRMED'])
summary.append(['1.1', 'Sharpe', f"{pre_m['sharpe']:.3f}", f"{post_m['sharpe']:.3f}",
                f"{post_m['sharpe']-pre_m['sharpe']:+.3f}", 'DEGRADATION CONFIRMED'])
summary.append(['1.1', 'Win Rate', f"{pre_m['wr']*100:.0f}%", f"{post_m['wr']*100:.0f}%",
                f"{(post_m['wr']-pre_m['wr'])*100:+.0f}pp", _verdict(pre_m['wr'], post_m['wr'])])
summary.append(['2.1', 'Gap→PnL corr', f"r={corr_pre:+.3f}", f"r={corr_post:+.3f}",
                f"Δ={corr_post-corr_pre:+.3f}",
                'REGIME FLIP' if corr_pre > 0.05 and corr_post < -0.05 else
                'SIGNAL DEAD' if abs(corr_post) < 0.02 else 'SAME SIGN'])
summary.append(['2.1', 'Avg gap size', f"{np.mean(pre_gaps):.1f}%", f"{np.mean(post_gaps):.1f}%",
                f"{np.mean(post_gaps)-np.mean(pre_gaps):+.1f}pp",
                _verdict(np.mean(pre_gaps), np.mean(post_gaps))])
summary.append(['3.1', 'Fallback rate', f"{pre_fb_rate:.0f}%", f"{post_fb_rate:.0f}%",
                f"{post_fb_rate-pre_fb_rate:+.0f}pp",
                _verdict(pre_fb_rate, post_fb_rate, threshold=0.5)])
summary.append(['3.1', 'Entry vs open', f"{np.mean(pre_slip):+.2f}%", f"{np.mean(post_slip):+.2f}%",
                f"{np.mean(post_slip)-np.mean(pre_slip):+.2f}pp", 'CHECK'])
summary.append(['4.2', 'MFE (favorable)', f"{np.mean(pre_mfe):+.2f}%", f"{np.mean(post_mfe):+.2f}%",
                f"{np.mean(post_mfe)-np.mean(pre_mfe):+.2f}pp",
                _verdict(np.mean(pre_mfe), np.mean(post_mfe))])
summary.append(['4.2', 'MAE (adverse)', f"{np.mean(pre_mae):+.2f}%", f"{np.mean(post_mae):+.2f}%",
                f"{np.mean(post_mae)-np.mean(pre_mae):+.2f}pp",
                _verdict(np.mean(pre_mae), np.mean(post_mae), invert=True)])

print()
print(tabulate(summary,
               headers=['Phase', 'Metric', 'Pre-Aug', 'Post-Aug', 'Delta', 'Verdict'],
               tablefmt='simple', stralign='right'))

# Decision tree output
print(f"\n\n  ── DECISION TREE ──")
if corr_pre > 0.05 and corr_post < -0.05:
    print(f"  → Signal-outcome correlation INVERTED: regime flip detected")
    print(f"  → Action: add regime filter or retrain signal on recent data")
elif abs(corr_post) < 0.02:
    print(f"  → Signal-outcome correlation went to ZERO: signal is dead in current regime")
    print(f"  → Action: investigate what changed in gap-fade mechanics")
if np.mean(post_mfe) < np.mean(pre_mfe) * 0.7:
    print(f"  → MFE collapsed {np.mean(pre_mfe):.1f}% → {np.mean(post_mfe):.1f}%: gap-fade upside capture degraded")
    print(f"  → Action: gaps not fading as deeply post-Aug — possible bull regime preventing mean reversion")
if np.mean(post_mae) > np.mean(pre_mae) * 1.3:
    print(f"  → MAE expanded {np.mean(pre_mae):.1f}% → {np.mean(post_mae):.1f}%: more adverse excursions")
    print(f"  → Action: market trending up harder after gap, need regime gate")
if post_fb_rate > pre_fb_rate * 1.5:
    print(f"  → Fallback rate surged {pre_fb_rate:.0f}% → {post_fb_rate:.0f}%: no clean local highs post-Aug")
    print(f"  → Action: opening price structure changed — investigate if V-pattern replaced local-high pattern")

# Save results
out = {
    'config': {'backtest_start': BACKTEST_START, 'backtest_end': BACKTEST_END,
               'break_date': BREAK_DATE, 'top_n': TOP_N},
    'overview': {
        'pre': {'n': pre_m['n'], 'avg': round(pre_m['avg'], 3), 'sharpe': round(pre_m['sharpe'], 4),
                'wr': round(pre_m['wr'], 3), 'pf': round(pre_m['pf'], 3)},
        'post': {'n': post_m['n'], 'avg': round(post_m['avg'], 3), 'sharpe': round(post_m['sharpe'], 4),
                 'wr': round(post_m['wr'], 3), 'pf': round(post_m['pf'], 3)},
    },
    'signal': {'corr_pre': round(float(corr_pre), 4), 'corr_post': round(float(corr_post), 4)},
    'mfe_mae': {
        'pre_mfe': round(float(np.mean(pre_mfe)), 3), 'post_mfe': round(float(np.mean(post_mfe)), 3),
        'pre_mae': round(float(np.mean(pre_mae)), 3), 'post_mae': round(float(np.mean(post_mae)), 3),
    },
    'entry': {'pre_fb_rate': round(pre_fb_rate, 1), 'post_fb_rate': round(post_fb_rate, 1)},
    'monthly': {m: {'n': len(monthly[m]['pnls']), 'avg_pnl': round(float(np.mean(monthly[m]['pnls'])), 3)}
                for m in months},
}
jp = os.path.join(BASE_DIR, 'results', 'e2a_investigation.json')
os.makedirs(os.path.dirname(jp), exist_ok=True)
with open(jp, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved: {jp}")
print(f"  Total runtime: {time.time() - t_total:.0f}s ({(time.time() - t_total)/60:.1f} min)")
