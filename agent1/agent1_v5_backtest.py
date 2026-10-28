# %% Agent 1 — V5 Strategy Backtest
#
# Tests 4 incremental V5 configurations derived from the
# 15-experiment Entry Mechanics investigation:
#
#   BASELINE:  E2-A as-is (LH + fallback, no filters)
#   V5-A:      LH-only (no fallback)
#   V5-B:      V5-A + SPY-down 30min filter
#   V5-C:      V5-B + vol quality gate + gap freshness ≥70%
#   V5-LATE:   Late fires (bars 8-11) + SPY-down + vol gate + gap ≥70%
#
# Colab-cell friendly. ~20 min runtime (API-bound).

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

# V5 filter thresholds (from Entry Mechanics investigation)
VOL_RATIO_MIN   = 0.153   # top quartile of entry_bar_vol / first_5_bar_vol
GAP_FRESH_MIN   = 0.70    # ≥70% of scan gap must remain at entry

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
# HELPERS (same as entry_mechanics pipeline)
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


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE (identical to entry_mechanics)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    t_total = time.time()
    print(f"\n{'='*80}")
    print(f"  AGENT 1 — V5 STRATEGY BACKTEST")
    print(f"  {BACKTEST_START} → {BACKTEST_END}  |  Break date: {BREAK_DATE}")
    print(f"  TOP_N={TOP_N}, CS only, SOFT float, AH fresh, erosion")
    print(f"  Exit {EXIT_TIME} ET | ${CAPITAL:,.0f}/day NC")
    print(f"  V5 gates: vol_ratio>{VOL_RATIO_MIN}, gap_fresh>={GAP_FRESH_MIN*100:.0f}%")
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
    print(f"  -> {len(all_bars)}/{len(all_pairs)} pairs")

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
    print(f"  -> {n_stale} stale excluded")

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
    print(f"  -> {n_eroded} eroded")

    all_tickers = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]: all_tickers.add(c['ticker'])
    print(f"\n  Phase 3: Refs for {len(all_tickers)} tickers...")
    for cs in range(0, len(sorted(all_tickers)), 200):
        fetch_ticker_refs(sorted(all_tickers)[cs:cs + 200])
    n_cs = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('type') == 'CS')
    print(f"  -> {n_cs} CS / {len(all_tickers)}")

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
# BUILD TRADES WITH V5 ANNOTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_annotated_trades(all_candidates, all_bars, pattern_cache, entry_cache,
                           dates, spy_bars_cache):
    trades = []
    for dt in dates:
        cs_qualified = []
        for c in all_candidates.get(dt, [])[:PRE_FILTER_N]:
            if _ticker_ref.get(c['ticker'], {}).get('type') != 'CS': continue
            cs_qualified.append(c)

        # SPY direction for this date
        spy_data = spy_bars_cache.get(('SPY', dt))
        spy_30m_down = None
        if spy_data:
            spy_rth = spy_data.get('rth', [])
            if len(spy_rth) >= 30:
                spy_open = spy_rth[0][1]['o']
                spy_30m_close = spy_rth[29][1]['c']
                spy_30m_down = spy_30m_close < spy_open

        for c in cs_qualified[:TOP_N]:
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

            # ── V5 annotations ──

            # 1. Fire type classification
            pb = tc.get('peak_bar')
            if tc['entry_type'] == 'fallback':
                tc['fire_type'] = 'fallback'
            elif pb is not None and pb >= 8:
                tc['fire_type'] = 'late'
            else:
                tc['fire_type'] = 'clean'

            # 2. Volume ratio: entry_bar_vol / sum(first 5 bar vols)
            tc['vol_ratio'] = None
            if bar_data:
                rth = bar_data.get('rth', [])
                eb = tc['entry_bar']
                if rth and len(rth) >= 5 and eb < len(rth):
                    first5_vol = sum(b[1]['v'] for b in rth[:5])
                    entry_bar_vol = rth[eb][1]['v']
                    tc['vol_ratio'] = entry_bar_vol / first5_vol if first5_vol > 0 else 0

            # 3. Gap freshness at entry
            tc['gap_remaining'] = None
            ref_price = tc.get('ah_ref_price', tc['prior_close'])
            scan_gap = tc['gap_pct']
            if ref_price > 0 and scan_gap > 0:
                entry_gap = (ep - ref_price) / ref_price
                tc['gap_remaining'] = entry_gap / scan_gap

            # 4. SPY direction
            tc['spy_30m_down'] = spy_30m_down

            if tc.get('pnl_eod') is not None:
                trades.append(tc)

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# V5 STRATEGY FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def filter_v5a(trades):
    """LH-only: remove fallback entries."""
    return [t for t in trades if t['entry_type'] == 'local_high']

def filter_v5b(trades):
    """LH-only + SPY-down 30min."""
    return [t for t in trades
            if t['entry_type'] == 'local_high'
            and t.get('spy_30m_down') == True]

def filter_v5c(trades):
    """LH-only + SPY-down + vol gate + gap freshness >=70%."""
    return [t for t in trades
            if t['entry_type'] == 'local_high'
            and t.get('spy_30m_down') == True
            and t.get('vol_ratio') is not None and t['vol_ratio'] > VOL_RATIO_MIN
            and t.get('gap_remaining') is not None and t['gap_remaining'] >= GAP_FRESH_MIN]

def filter_v5_late(trades):
    """Late fires (bars 8-11) + SPY-down + vol gate + gap freshness >=70%."""
    return [t for t in trades
            if t['fire_type'] == 'late'
            and t.get('spy_30m_down') == True
            and t.get('vol_ratio') is not None and t['vol_ratio'] > VOL_RATIO_MIN
            and t.get('gap_remaining') is not None and t['gap_remaining'] >= GAP_FRESH_MIN]


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_strategy(trades, label, dates):
    pnls = [t['pnl_eod'] for t in trades]
    pre_t = [t for t in trades if t['scan_date'] < BREAK_DATE]
    post_t = [t for t in trades if t['scan_date'] >= BREAK_DATE]
    pre_m = compute_metrics([t['pnl_eod'] for t in pre_t], f'{label} Pre')
    post_m = compute_metrics([t['pnl_eod'] for t in post_t], f'{label} Post')
    all_m = compute_metrics(pnls, label)

    # Monthly breakdown
    monthly = defaultdict(list)
    for t in trades:
        monthly[t['scan_date'][:7]].append(t['pnl_eod'])

    # Daily P&L for equity curve
    daily_pnl = defaultdict(list)
    for t in trades:
        daily_pnl[t['scan_date']].append(t['pnl_eod'])

    equity = [CAPITAL]
    eq_dates = []
    running = CAPITAL
    cum_nc = 0.0
    daily_stats = []
    for dt in dates:
        dp = daily_pnl.get(dt, [])
        if dp:
            n = len(dp)
            pos_size = running / n
            day_dollar = sum(pos_size * (r / 100) for r in dp)
            running += day_dollar
            equity.append(running)
            eq_dates.append(dt)
            nc_dollar = sum(CAPITAL / n * (r / 100) for r in dp)
        else:
            nc_dollar = 0.0
            n = 0
        cum_nc += nc_dollar
        daily_stats.append({'date': dt, 'n': n, 'dollar': nc_dollar, 'cum': cum_nc})

    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = float(dd.min())
    active_days = sum(1 for ds in daily_stats if ds['n'] > 0)
    tpd = len(pnls) / active_days if active_days > 0 else 0

    # Fire type breakdown
    ft_counts = defaultdict(int)
    for t in trades:
        ft_counts[t.get('fire_type', 'unknown')] += 1

    return {
        'label': label, 'trades': trades,
        'pre': pre_m, 'post': post_m, 'all': all_m,
        'monthly': dict(monthly),
        'equity': equity, 'eq_dates': eq_dates,
        'daily_stats': daily_stats,
        'max_dd': max_dd, 'final': running, 'cum_nc': cum_nc,
        'active_days': active_days, 'tpd': tpd,
        'fire_types': dict(ft_counts),
        'dd_series': dd.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

all_candidates, all_bars, pattern_cache, entry_cache, dates, t_total = run_pipeline()

# Fetch SPY bars
print(f"\n  Fetching SPY minute bars for {len(dates)} days...")
spy_pairs = [('SPY', dt) for dt in dates]
spy_bars_cache = {}
for cs in range(0, len(spy_pairs), 100):
    chunk = spy_pairs[cs:cs + 100]
    spy_bars_cache.update(fetch_minute_bars_batch(chunk))
print(f"  -> {len(spy_bars_cache)} SPY bar days")

# Build annotated trades
trades = build_annotated_trades(all_candidates, all_bars, pattern_cache, entry_cache,
                                dates, spy_bars_cache)
print(f"\n  Built {len(trades)} annotated baseline trades")

# Apply strategy filters
strategies = {
    'BASELINE': trades,
    'V5-A':     filter_v5a(trades),
    'V5-B':     filter_v5b(trades),
    'V5-C':     filter_v5c(trades),
    'V5-LATE':  filter_v5_late(trades),
}

strat_order = ['BASELINE', 'V5-A', 'V5-B', 'V5-C', 'V5-LATE']
results = {}
for name in strat_order:
    results[name] = analyze_strategy(strategies[name], name, dates)


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print(f"  V5 STRATEGY COMPARISON  |  {dates[0]} → {dates[-1]}  |  Break: {BREAK_DATE}")
print(f"  Capital: ${CAPITAL:,.0f}/day  |  Non-compounding  |  Exit {EXIT_TIME} ET")
print(f"{'='*120}")

# ── Main comparison table ──
headers = ['Metric'] + strat_order
def _row(name, fn):
    return [name] + [fn(results[s]) for s in strat_order]

tbl = [
    _row('Description', lambda r: {
        'BASELINE': 'E2-A as-is',
        'V5-A': 'LH only',
        'V5-B': '+SPY-down',
        'V5-C': '+vol+gap',
        'V5-LATE': 'late+SPY+vol+gap',
    }.get(r['label'], '')),
    _row('Pre N',       lambda r: f"{r['pre']['n']}"),
    _row('Post N',      lambda r: f"{r['post']['n']}"),
    _row('Total N',     lambda r: f"{r['all']['n']}"),
    _row('Pre Avg',     lambda r: f"{r['pre']['avg']:+.2f}%"),
    _row('Post Avg',    lambda r: f"{r['post']['avg']:+.2f}%"),
    _row('Delta',       lambda r: f"{r['post']['avg'] - r['pre']['avg']:+.2f}pp"),
    _row('All Avg',     lambda r: f"{r['all']['avg']:+.2f}%"),
    _row('Pre WR',      lambda r: f"{r['pre']['wr']*100:.0f}%"),
    _row('Post WR',     lambda r: f"{r['post']['wr']*100:.0f}%"),
    _row('Pre Sharpe',  lambda r: f"{r['pre']['sharpe']:.3f}"),
    _row('Post Sharpe', lambda r: f"{r['post']['sharpe']:.3f}"),
    _row('Pre PF',      lambda r: f"{r['pre']['pf']:.2f}" if r['pre']['pf'] < 100 else 'inf'),
    _row('Post PF',     lambda r: f"{r['post']['pf']:.2f}" if r['post']['pf'] < 100 else 'inf'),
    _row('Pre 95% CI',  lambda r: f"[{r['pre']['ci_lo']:+.1f}, {r['pre']['ci_hi']:+.1f}]"),
    _row('Post 95% CI', lambda r: f"[{r['post']['ci_lo']:+.1f}, {r['post']['ci_hi']:+.1f}]"),
    _row('CI > 0',      lambda r: f"{'YES' if r['post']['ci_lo'] > 0 else 'NO'}"),
    _row('Pre Trimmed',  lambda r: f"{r['pre']['trimmed_mean']:+.2f}%"),
    _row('Post Trimmed', lambda r: f"{r['post']['trimmed_mean']:+.2f}%"),
    _row('Max DD',      lambda r: f"{r['max_dd']:.1f}%"),
    _row('Total $ NC',  lambda r: f"${r['cum_nc']:+,.0f}"),
    _row('Total $ CMP', lambda r: f"${r['final']-CAPITAL:+,.0f}"),
    _row('TPD',         lambda r: f"{r['tpd']:.2f}"),
    _row('Active Days', lambda r: f"{r['active_days']}"),
]
print()
print(tabulate(tbl, headers=headers, tablefmt='simple', stralign='right'))

# ── Fire type breakdown ──
print(f"\n\n{'─'*80}")
print(f"  FIRE TYPE BREAKDOWN")
print(f"{'─'*80}")
for name in strat_order:
    r = results[name]
    ft = r['fire_types']
    parts = [f"{k}={v}" for k, v in sorted(ft.items())]
    print(f"  {name:12s}  {', '.join(parts)}")

# ── Filter funnel ──
print(f"\n\n{'─'*80}")
print(f"  FILTER FUNNEL (how many trades survive each gate)")
print(f"{'─'*80}")

all_lh = [t for t in trades if t['entry_type'] == 'local_high']
all_spy = [t for t in all_lh if t.get('spy_30m_down') == True]
all_vol = [t for t in all_spy if t.get('vol_ratio') is not None and t['vol_ratio'] > VOL_RATIO_MIN]
all_gap = [t for t in all_vol if t.get('gap_remaining') is not None and t['gap_remaining'] >= GAP_FRESH_MIN]
all_late_base = [t for t in trades if t.get('fire_type') == 'late']
all_late_spy = [t for t in all_late_base if t.get('spy_30m_down') == True]
all_late_vol = [t for t in all_late_spy if t.get('vol_ratio') is not None and t['vol_ratio'] > VOL_RATIO_MIN]
all_late_gap = [t for t in all_late_vol if t.get('gap_remaining') is not None and t['gap_remaining'] >= GAP_FRESH_MIN]

def _pre_post(tl):
    pre = sum(1 for t in tl if t['scan_date'] < BREAK_DATE)
    post = len(tl) - pre
    return pre, post

for label, tl in [
    ('Baseline (all)', trades),
    ('  - LH only', all_lh),
    ('    - SPY-down', all_spy),
    ('      - Vol>Q75', all_vol),
    ('        - Gap>=70%', all_gap),
    ('Late fires (all)', all_late_base),
    ('  - SPY-down', all_late_spy),
    ('    - Vol>Q75', all_late_vol),
    ('      - Gap>=70%', all_late_gap),
]:
    pre, post = _pre_post(tl)
    pnls_pre = [t['pnl_eod'] for t in tl if t['scan_date'] < BREAK_DATE]
    pnls_post = [t['pnl_eod'] for t in tl if t['scan_date'] >= BREAK_DATE]
    avg_pre = np.mean(pnls_pre) if pnls_pre else 0
    avg_post = np.mean(pnls_post) if pnls_post else 0
    print(f"  {label:25s}  total={len(tl):>4d}  pre={pre:>4d} ({avg_pre:+.2f}%)  post={post:>3d} ({avg_post:+.2f}%)")

# ── Monthly comparison ──
print(f"\n\n{'─'*80}")
print(f"  MONTHLY COMPARISON")
print(f"{'─'*80}")
all_months = sorted(set(
    mo for r in results.values() for mo in r['monthly']
))
mo_headers = ['Month'] + [f'{s} n' for s in strat_order] + [f'{s} avg' for s in strat_order]
mo_rows = []
for mo in all_months:
    row = [mo]
    for s in strat_order:
        pnls = results[s]['monthly'].get(mo, [])
        row.append(f"{len(pnls)}")
    for s in strat_order:
        pnls = results[s]['monthly'].get(mo, [])
        row.append(f"{np.mean(pnls):+.2f}%" if pnls else '—')
    mo_rows.append(row)
print(tabulate(mo_rows, headers=mo_headers, tablefmt='simple'))

# ── Worst / Best trades for V5-C and V5-LATE ──
for name in ['V5-C', 'V5-LATE']:
    r = results[name]
    valid = sorted(r['trades'], key=lambda x: x['pnl_eod'])
    print(f"\n\n{'─'*80}")
    print(f"  {name} — WORST 5 / BEST 5")
    print(f"{'─'*80}")
    for lbl, sl in [('WORST 5', valid[:5]), ('BEST 5', valid[-5:])]:
        print(f"\n  {lbl}:")
        for t in sl:
            gap = t['gap_pct'] * 100
            vr = t.get('vol_ratio', 0) or 0
            gr = t.get('gap_remaining', 0) or 0
            spy = 'D' if t.get('spy_30m_down') else 'U'
            pb_str = str(t.get('peak_bar', '—'))
            print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={gap:>+6.1f}%  "
                  f"fire={t['fire_type']:>8s}  bar={pb_str:>3s}  "
                  f"vr={vr:.3f}  gr={gr:.2f}  SPY={spy}  pnl={t['pnl_eod']:>+8.2f}%")

# ── V5-C vs BASELINE: per-month delta ──
print(f"\n\n{'─'*80}")
print(f"  V5-C vs BASELINE — MONTHLY DELTA")
print(f"{'─'*80}")
for mo in all_months:
    bl_pnls = results['BASELINE']['monthly'].get(mo, [])
    v5_pnls = results['V5-C']['monthly'].get(mo, [])
    bl_avg = np.mean(bl_pnls) if bl_pnls else 0
    v5_avg = np.mean(v5_pnls) if v5_pnls else 0
    delta = v5_avg - bl_avg
    marker = '<< BREAK' if mo == '2026-08' else ''
    print(f"  {mo}  BL={bl_avg:+.2f}% (n={len(bl_pnls)})  V5-C={v5_avg:+.2f}% (n={len(v5_pnls)})  Δ={delta:+.2f}pp  {marker}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

COLORS = {
    'BASELINE': '#95a5a6',
    'V5-A':     '#3498db',
    'V5-B':     '#2ecc71',
    'V5-C':     '#e74c3c',
    'V5-LATE':  '#9b59b6',
}

# ── Figure 1: Equity curves + Drawdown ──
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle(
    f'Agent 1 — V5 Strategy Comparison  |  {dates[0]} → {dates[-1]}  |  Break: {BREAK_DATE}\n'
    f'BASELINE: {results["BASELINE"]["all"]["n"]}t  |  '
    f'V5-A: {results["V5-A"]["all"]["n"]}t  |  '
    f'V5-B: {results["V5-B"]["all"]["n"]}t  |  '
    f'V5-C: {results["V5-C"]["all"]["n"]}t  |  '
    f'V5-LATE: {results["V5-LATE"]["all"]["n"]}t',
    fontsize=11, fontweight='bold')

# Equity curves (compounding)
ax = axes[0, 0]
for name in strat_order:
    r = results[name]
    eq_d = [datetime.strptime(dates[0], '%Y-%m-%d')] + \
           [datetime.strptime(d, '%Y-%m-%d') for d in r['eq_dates']]
    ax.plot(eq_d, r['equity'], color=COLORS[name], linewidth=2 if name in ('V5-C','V5-LATE') else 1.2,
            alpha=0.5 if name == 'BASELINE' else 1.0,
            label=f'{name} (${r["final"]:,.0f})')
ax.axhline(CAPITAL, color='#999', ls='--', lw=0.8)
ax.axvline(datetime.strptime(BREAK_DATE, '%Y-%m-%d'), color='red', ls=':', lw=1, alpha=0.5)
ax.set_title('Equity Curves (compounding)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.2)

# Cumulative NC P&L
ax = axes[0, 1]
for name in strat_order:
    r = results[name]
    ds = r['daily_stats']
    d_dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in ds]
    cum = [s['cum'] for s in ds]
    ax.plot(d_dates, cum, color=COLORS[name], linewidth=2 if name in ('V5-C','V5-LATE') else 1.2,
            alpha=0.5 if name == 'BASELINE' else 1.0,
            label=f'{name} (${r["cum_nc"]:+,.0f})')
ax.axhline(0, color='#333', lw=0.8, ls=':')
ax.axvline(datetime.strptime(BREAK_DATE, '%Y-%m-%d'), color='red', ls=':', lw=1, alpha=0.5)
ax.set_title('Cumulative P&L (non-compounding, $100K/day)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.2)

# Pre vs Post avg PnL bar chart
ax = axes[1, 0]
x = np.arange(len(strat_order))
w = 0.35
pre_avgs = [results[s]['pre']['avg'] for s in strat_order]
post_avgs = [results[s]['post']['avg'] for s in strat_order]
bars1 = ax.bar(x - w/2, pre_avgs, w, color=[COLORS[s] for s in strat_order], alpha=0.6, label='Pre-Aug')
bars2 = ax.bar(x + w/2, post_avgs, w, color=[COLORS[s] for s in strat_order], alpha=1.0, label='Post-Aug')
ax.set_xticks(x)
ax.set_xticklabels(strat_order, fontsize=9)
ax.axhline(0, color='#333', lw=0.8)
ax.set_title('Avg PnL: Pre-Aug (light) vs Post-Aug (solid)')
ax.set_ylabel('Avg PnL (%)')
for i, (pre, post) in enumerate(zip(pre_avgs, post_avgs)):
    delta = post - pre
    ax.annotate(f'Δ={delta:+.1f}pp', xy=(i, max(pre, post) + 0.3),
                ha='center', fontsize=7, color='red' if delta < -1 else 'green')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

# Drawdown
ax = axes[1, 1]
for name in strat_order:
    r = results[name]
    eq_d = [datetime.strptime(dates[0], '%Y-%m-%d')] + \
           [datetime.strptime(d, '%Y-%m-%d') for d in r['eq_dates']]
    dd = r['dd_series']
    ax.fill_between(eq_d, dd, 0, alpha=0.08, color=COLORS[name])
    ax.plot(eq_d, dd, color=COLORS[name], linewidth=1.2,
            label=f'{name} ({min(dd):.1f}%)')
ax.axvline(datetime.strptime(BREAK_DATE, '%Y-%m-%d'), color='red', ls=':', lw=1, alpha=0.5)
ax.set_title('Drawdown')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.92])
p1 = os.path.join(BASE_DIR, 'v5_backtest_comparison.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {p1}")

# ── Figure 2: Monthly heatmap ──
fig, axes = plt.subplots(len(strat_order), 1, figsize=(16, 3 * len(strat_order)), sharex=True)
fig.suptitle('Monthly Avg PnL by Strategy', fontsize=12, fontweight='bold')
for idx, name in enumerate(strat_order):
    ax = axes[idx]
    r = results[name]
    mo_avgs = [np.mean(r['monthly'].get(mo, [0])) for mo in all_months]
    mo_ns = [len(r['monthly'].get(mo, [])) for mo in all_months]
    colors = ['#27ae60' if v > 0 else '#e74c3c' for v in mo_avgs]
    bars = ax.bar(range(len(all_months)), mo_avgs, color=colors, alpha=0.8)
    for i, (v, n) in enumerate(zip(mo_avgs, mo_ns)):
        ax.text(i, v + 0.15 if v >= 0 else v - 0.5, f'n={n}', ha='center', fontsize=7)
    ax.axhline(0, color='#333', lw=0.8)
    break_idx = next((i for i, mo in enumerate(all_months) if mo >= '2026-08'), len(all_months))
    ax.axvline(break_idx - 0.5, color='red', ls=':', lw=1, alpha=0.5)
    ax.set_ylabel(name, fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(all_months)))
    ax.set_xticklabels([mo[5:] for mo in all_months], fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.95])
p2 = os.path.join(BASE_DIR, 'v5_backtest_monthly.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {p2}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE JSON
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
json_out = {}
for name in strat_order:
    r = results[name]
    json_out[name] = {
        'pre': {k: v for k, v in r['pre'].items() if k != 'label'},
        'post': {k: v for k, v in r['post'].items() if k != 'label'},
        'all': {k: v for k, v in r['all'].items() if k != 'label'},
        'monthly': {mo: {'n': len(pnls), 'avg': float(np.mean(pnls))}
                    for mo, pnls in r['monthly'].items()},
        'max_dd': r['max_dd'],
        'cum_nc': r['cum_nc'],
        'final_capital': r['final'],
        'tpd': r['tpd'],
        'active_days': r['active_days'],
        'fire_types': r['fire_types'],
    }

json_path = os.path.join(BASE_DIR, 'results', 'v5_backtest.json')
with open(json_path, 'w') as f:
    json.dump(json_out, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x) if isinstance(x, (np.floating, np.integer)) else x)
print(f"\n  Saved: {json_path}")

print(f"\n{'='*80}")
print(f"  TOTAL RUNTIME: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f} min)")
print(f"{'='*80}")
